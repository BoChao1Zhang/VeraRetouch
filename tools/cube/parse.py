#!/usr/bin/env python3
"""T2 parse: parse every preset LUT (.cube via colour-science, .3dl via a
minimal OCIO-convention 3DMESH reader), resample to a canonical 33^3 table,
and store as .npy.

Canonical npy: shape (33,33,33,3) float32, index [r,g,b], RGB channels,
implicit domain [0,1] (DOMAIN_MIN/MAX handled by colour's LUT3D.apply;
.3dl shaper axes handled by rectilinear interpolation).

Outputs:
  <npy-dir>/<id>.npy            canonical tables
  <out-dir>/parse_report.jsonl  one row per file (ok/error + metadata)
  <out-dir>/parse_failures.txt  failing paths with reason
  <out-dir>/parse_summary.json  headline numbers

Usage:
  python3 parse.py --manifest <dcube_manifest.jsonl> --npy-dir DIR --out-dir DIR
  python3 parse.py --paths file_with_one_path_per_line --npy-dir DIR --out-dir DIR
  python3 parse.py --selftest            # tiny built-in check on synthetic LUTs
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import CANONICAL_SIZE, DEFAULT_NPY_DIR, identity_table, preset_slug

_GRID33 = None  # lazy per-process identity grid


def _grid33() -> np.ndarray:
    global _GRID33
    if _GRID33 is None:
        _GRID33 = identity_table(CANONICAL_SIZE).astype(np.float64)
    return _GRID33


# ---------------------------------------------------------------------------
# .3dl (Lustre 3DMESH) reader — conventions verified against OpenColorIO
# FileFormat3DL.cpp (see NOTES.md §5.1): blue varies fastest; output depth
# inferred from max value; shaper line = input axis sample positions.
# ---------------------------------------------------------------------------

def _infer_scale(max_value: float) -> int:
    if max_value <= 511:
        return 255
    if max_value <= 2047:
        return 1023
    if max_value <= 8191:
        return 4095
    return 65535


def read_3dl(path: str):
    """Return (table[r,g,b,3] float64 in [0,1], axis positions in [0,1], meta)."""
    shaper = None
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if not toks[0].lstrip("+-").isdigit():
                continue  # 3DMESH / Mesh / LUT8 ... keyword lines
            vals = [int(t) for t in toks]
            if len(vals) > 3:
                shaper = vals
            elif len(vals) == 3:
                rows.append(vals)
            else:
                raise ValueError(f"unexpected token count {len(vals)}")
    if not rows:
        raise ValueError("no 3D data rows")
    n = round(len(rows) ** (1.0 / 3.0))
    if n ** 3 != len(rows):
        raise ValueError(f"row count {len(rows)} is not a cube")
    if shaper is not None and len(shaper) != n:
        raise ValueError(f"shaper length {len(shaper)} != mesh size {n}")

    arr = np.asarray(rows, dtype=np.float64)
    scale = _infer_scale(arr.max())
    # blue-fastest file order -> C-order reshape gives index [r, g, b]
    table = arr.reshape(n, n, n, 3) / scale

    if shaper is not None:
        axis = np.asarray(shaper, dtype=np.float64)
        axis = axis / axis[-1]
    else:
        axis = np.linspace(0.0, 1.0, n)
    uniform_dev = float(np.abs(axis - np.linspace(0, 1, n)).max())
    meta = {
        "orig_size": n,
        "out_scale": scale,
        "shaper_uniform_dev": uniform_dev,
        "shaper_nonuniform": uniform_dev > 0.02,
    }
    return table, axis, meta


def resample_3dl(table: np.ndarray, axis: np.ndarray) -> np.ndarray:
    from scipy.interpolate import RegularGridInterpolator

    rgi = RegularGridInterpolator(
        (axis, axis, axis), table, method="linear", bounds_error=False,
        fill_value=None,  # pyright: ignore[reportArgumentType]  # None=线性外推（scipy 文档语义；stub 标注过窄为 float）
    )
    grid = _grid33()  # (33,33,33,3) with grid[i,j,k]=(r_i,g_j,b_k)
    pts = grid.reshape(-1, 3)
    out = rgi(pts).reshape(CANONICAL_SIZE, CANONICAL_SIZE, CANONICAL_SIZE, 3)
    return out


# ---------------------------------------------------------------------------
# .cube reader via colour-science
# ---------------------------------------------------------------------------

APPLE_DOUBLE_MAGIC = b"\x00\x05\x16\x07"


def _read_cube_with_fallback(path: str, meta: dict):
    """read_LUT_IridasCube with two recovery paths (see NOTES.md §5.2):
    - non-UTF8 titles/comments (GBK in this corpus): re-encode and retry;
    - Resolve dialect (LUT_3D_INPUT_RANGE): colour's Resolve reader.
    AppleDouble resource forks are rejected explicitly."""
    import tempfile

    import colour

    try:
        return colour.io.read_LUT_IridasCube(path)
    except UnicodeDecodeError:
        raw = open(path, "rb").read()
        if raw[:4] == APPLE_DOUBLE_MAGIC:
            raise ValueError(
                "AppleDouble resource fork (macOS metadata), not a LUT") from None
        for enc in ("gbk", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise
        meta["reencoded_from"] = enc
        with tempfile.NamedTemporaryFile("w", suffix=".cube",
                                         encoding="utf-8", delete=False) as tf:
            tf.write(text)
            tmp = tf.name
        try:
            return colour.io.read_LUT_IridasCube(tmp)
        finally:
            os.unlink(tmp)
    except ValueError as e:
        if "INPUT_RANGE" in str(e):
            meta["dialect"] = "resolve"
            return colour.io.read_LUT_ResolveCube(path)
        raise


def parse_cube(path: str):
    from colour.algebra import table_interpolation_tetrahedral
    from colour.io.luts import LUT3D, LUT3x1D
    from colour.utilities import suppress_warnings

    meta = {}
    lut = _read_cube_with_fallback(path, meta)
    grid = _grid33()
    if isinstance(lut, LUT3D):
        meta["lut_type"] = "3D"
        meta["orig_size"] = int(lut.table.shape[0])
        meta["domain_explicit"] = bool(lut.is_domain_explicit())
        dom = np.asarray(lut.domain, dtype=float)
        meta["domain"] = dom.tolist()
        meta["domain_nonunit"] = not (
            dom.shape == (2, 3)
            and np.allclose(dom, [[0, 0, 0], [1, 1, 1]])
        )
        with suppress_warnings(python_warnings=True):
            table33 = lut.apply(grid, interpolator=table_interpolation_tetrahedral)
    elif isinstance(lut, LUT3x1D):
        meta["lut_type"] = "3x1D"
        meta["orig_size"] = int(lut.table.shape[0])
        dom = np.asarray(lut.domain, dtype=float)
        meta["domain"] = dom.tolist()
        meta["domain_explicit"] = bool(lut.is_domain_explicit())
        meta["domain_nonunit"] = True  # informational; apply() handles it
        with suppress_warnings(python_warnings=True):
            table33 = lut.apply(grid)
    elif hasattr(lut, "apply"):  # e.g. LUTSequence from the Resolve reader
        meta["lut_type"] = type(lut).__name__
        meta["orig_size"] = None
        meta["domain_explicit"] = False
        meta["domain_nonunit"] = True  # unknown composition; apply() handles
        with suppress_warnings(python_warnings=True):
            table33 = lut.apply(grid)
    else:  # pragma: no cover
        raise TypeError(f"unsupported LUT class {type(lut).__name__}")
    return table33, meta


def parse_any(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cube":
        return parse_cube(path)
    if ext == ".3dl":
        table, axis, meta = read_3dl(path)
        meta["lut_type"] = "3dl_mesh"
        meta["domain"] = [[0, 0, 0], [1, 1, 1]]
        meta["domain_explicit"] = False
        meta["domain_nonunit"] = False
        return resample_3dl(table, axis), meta
    raise ValueError(f"unsupported extension {ext}")


def _worker(job):
    path, npy_dir = job
    pid = preset_slug(path)
    row = {"id": pid, "path": path,
           "format": os.path.splitext(path)[1].lstrip(".").lower()}
    try:
        table33, meta = parse_any(path)
        table33 = np.asarray(table33, dtype=np.float32)
        if table33.shape != (CANONICAL_SIZE,) * 3 + (3,):
            raise ValueError(f"bad resampled shape {table33.shape}")
        if not np.isfinite(table33).all():
            raise ValueError("non-finite values after resample")
        np.save(os.path.join(npy_dir, pid + ".npy"), table33)
        row.update(meta)
        row["ok"] = True
        rng = (float(table33.min()), float(table33.max()))
        row["value_range"] = rng
        row["range_outside_unit"] = rng[0] < -1e-4 or rng[1] > 1 + 1e-4
    except Exception as e:  # noqa: BLE001 — per-file failure is a data finding
        row["ok"] = False
        row["error"] = f"{type(e).__name__}: {e}"
        row["traceback"] = traceback.format_exc(limit=3)
    return row


def run(paths, npy_dir, out_dir, workers):
    os.makedirs(npy_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(p, npy_dir) for p in paths]
    rows = []
    with mp.Pool(workers) as pool:
        for i, row in enumerate(pool.imap_unordered(_worker, jobs, chunksize=16)):
            rows.append(row)
            if (i + 1) % 500 == 0:
                print(f"  parsed {i + 1}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: r["id"])

    with open(os.path.join(out_dir, "parse_report.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    failures = [r for r in rows if not r["ok"]]
    with open(os.path.join(out_dir, "parse_failures.txt"), "w") as f:
        for r in failures:
            f.write(f"{r['path']}\t{r['error']}\n")

    ok = [r for r in rows if r["ok"]]
    summary = {
        "total": len(rows),
        "ok": len(ok),
        "failed": len(failures),
        "success_rate": round(len(ok) / max(1, len(rows)), 6),
        "by_format": {},
        "by_lut_type": {},
        "orig_sizes": {},
        "domain_nonunit": sum(1 for r in ok if r.get("domain_nonunit")),
        "domain_explicit": sum(1 for r in ok if r.get("domain_explicit")),
        "range_outside_unit": sum(1 for r in ok if r.get("range_outside_unit")),
        "shaper_nonuniform": sum(1 for r in ok if r.get("shaper_nonuniform")),
    }
    for r in rows:
        fmt = r["format"]
        d = summary["by_format"].setdefault(fmt, {"total": 0, "ok": 0})
        d["total"] += 1
        d["ok"] += int(r["ok"])
    for r in ok:
        summary["by_lut_type"][r.get("lut_type", "?")] = (
            summary["by_lut_type"].get(r.get("lut_type", "?"), 0) + 1)
        s = str(r.get("orig_size"))
        summary["orig_sizes"][s] = summary["orig_sizes"].get(s, 0) + 1
    with open(os.path.join(out_dir, "parse_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    return summary


def selftest() -> None:
    """Synthetic round-trip checks that need no corpus files."""
    import tempfile

    from colour import LUT3D
    from colour.io import write_LUT_IridasCube

    with tempfile.TemporaryDirectory() as td:
        # 1) identity cube 33 -> canonical table must equal linear table exactly
        p = os.path.join(td, "id33.cube")
        write_LUT_IridasCube(LUT3D(LUT3D.linear_table(33), "id"), p)
        t, _ = parse_any(p)
        err = np.abs(t - _grid33()).max()
        assert err < 1e-7, f"identity cube roundtrip err {err}"

        # 2) gamma cube 17 -> resampled 33 close to analytic gamma
        p = os.path.join(td, "g17.cube")
        write_LUT_IridasCube(LUT3D(LUT3D.linear_table(17) ** (1 / 2.2), "g"), p)
        t, meta = parse_any(p)
        ref = _grid33() ** (1 / 2.2)
        err = np.abs(t - ref).max()
        assert meta["orig_size"] == 17
        # near-black curvature of x^(1/2.2) dominates: linear interp between
        # nodes 0 and 1/16 is off by ~0.065 at x=1/32 — genuine interp error,
        # inherent to any 17^3 LUT of a gamma curve, not a parser defect.
        assert err < 0.08, f"gamma17 resample err {err}"
        mid_err = np.abs(t[8:, 8:, 8:] - ref[8:, 8:, 8:]).max()
        assert mid_err < 5e-3, f"gamma17 mid/upper-range err {mid_err}"

        # 3) synthetic identity .3dl (10-bit, mesh 17, Lustre shaper axis)
        p = os.path.join(td, "id.3dl")
        n = 17
        axis1023 = np.round(np.linspace(0, 1023, n)).astype(int)
        with open(p, "w") as f:
            f.write("3DMESH\nMesh 4 10\n")
            f.write(" ".join(str(v) for v in axis1023) + "\n")
            for r in range(n):
                for g in range(n):
                    for b in range(n):  # blue fastest
                        f.write(f"{axis1023[r]} {axis1023[g]} {axis1023[b]}\n")
        t, meta = parse_any(p)
        err = np.abs(t - _grid33()).max()
        assert meta["out_scale"] == 1023, meta
        assert err < 1e-6, f"identity 3dl err {err}"
    print("parse.py selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", help="dcube_manifest.jsonl from inventory.py")
    ap.add_argument("--paths", help="text file, one LUT path per line")
    ap.add_argument("--npy-dir", default=DEFAULT_NPY_DIR)
    ap.add_argument("--out-dir")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="parse first N only")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.out_dir:
        ap.error("--out-dir is required (unless --selftest)")

    if args.manifest:
        paths = [json.loads(l)["path"] for l in open(args.manifest)]
    elif args.paths:
        paths = [l.strip() for l in open(args.paths) if l.strip()]
    else:
        ap.error("need --manifest or --paths")
    if args.limit:
        paths = paths[: args.limit]
    run(paths, args.npy_dir, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
