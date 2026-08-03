#!/usr/bin/env python3
"""F5 main pass: re-render every manifest pair through the colour tetrahedral
path and compare against the archived production after image.

Per pair:
  I_in   = preprocess(bank source bytes)      # rendering.preprocess_source math
  table  = colour.io.read_LUT_IridasCube      # independent parse, [r,g,b] axes
  cross  = max|table - load_lut(grid)[b,g,r].T|   # parser consistency
  tet    = colour tetrahedral apply (+ mask composite for l-line)
  tri    = production trilinear math replica (grid[b,g,r] oracle) (+ composite)
  swap   = counterfactual channel-swap bug: apply(img[...,::-1])[...,::-1]
  floor  = ΔE00(jpeg_q95_roundtrip(tri), tri)     # codec-only noise floor
  metrics: ΔE00 quantiles (after vs tet / tri / swap), 3x3 channel corr
  (raw + delta-vs-I_in), channel permutation MAE test.

Usage:
  /home/bc/miniconda3/bin/python3 run_check.py [--workers 8] [--limit N]
  /home/bc/miniconda3/bin/python3 run_check.py --viz g001 l063 ...   # panels
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from dataset_build.lut_io import load_lut  # noqa: E402  (prod parser)


def load_pair_images(row: dict):
    """-> (I_in float32 HW3, after float32 HW3, alpha float32 HW | None)."""
    src = row["source_ref"]
    if src["kind"] == "local":
        with open(src["path"], "rb") as f:
            data = f.read()
    else:
        data = C.read_member(src["tar"], src["offset_data"], src["length"])
    i_in = C.preprocess_source_bytes(data, short_edge=1024)

    a = row["after_ref"]
    after_bytes = C.read_member(a["tar"], a["offset_data"], a["length"])
    from PIL import Image
    import io as _io
    after = np.asarray(Image.open(_io.BytesIO(after_bytes)).convert("RGB"),
                       dtype=np.float32) / 255.0

    alpha = None
    if row.get("cgt_ref"):
        m = row["cgt_ref"]
        mask_bytes = C.read_member(m["tar"], m["offset_data"], m["length"])
        alpha = np.asarray(Image.open(_io.BytesIO(mask_bytes)).convert("L"),
                           dtype=np.float32) / 255.0
    return i_in, after, alpha


def render_all(row: dict, i_in: np.ndarray, alpha: np.ndarray | None):
    """-> (tet, tri, swap, parser_crosscheck_maxabs, domain_default)."""
    path = row["preset_path"]
    table_rgb, domain = C.read_cube_colour(path)
    grid_bgr, dmin, dmax = load_lut(path)
    cross = float(np.max(np.abs(table_rgb - grid_bgr.transpose(2, 1, 0, 3))))
    domain_default = bool(
        np.allclose(domain, [[0, 0, 0], [1, 1, 1]])
        and np.allclose(dmin, 0) and np.allclose(dmax, 1))

    tet = C.apply_lut_tetrahedral_rgb(table_rgb, i_in, domain=domain)
    tri = C.apply_lut_trilinear_bgr(i_in, grid_bgr, dmin, dmax)
    swap = C.apply_lut_tetrahedral_rgb(
        table_rgb, np.ascontiguousarray(i_in[..., ::-1]), domain=domain
    )[..., ::-1]
    if alpha is not None:
        tet = C.composite_prod(i_in, tet, alpha)
        tri = C.composite_prod(i_in, tri, alpha)
        swap = C.composite_prod(i_in, swap, alpha)
    return tet, tri, np.ascontiguousarray(swap), cross, domain_default


def check_pair(row: dict) -> dict:
    try:
        i_in, after, alpha = load_pair_images(row)
        if after.shape != i_in.shape:
            return {"pair_id": row["pair_id"], "error":
                    f"shape mismatch after={after.shape} i_in={i_in.shape}"}
        tet, tri, swap, cross, domain_default = render_all(row, i_in, alpha)

        de_tet = C.delta_e00_tiled(after, tet)
        de_tri = C.delta_e00_tiled(after, tri)
        de_swap = C.delta_e00_tiled(after, swap)
        floor = C.delta_e00_tiled(C.jpeg_roundtrip(tri), tri)

        out = {
            "pair_id": row["pair_id"],
            "line": row["line"],
            "build": row["build"],
            "pool": row["pool"],
            "preset_id": row["preset_id"],
            "preset_path": row["preset_path"],
            "lut_size": row["lut_size"],
            "candidate_id": row["candidate_id"],
            "masked": alpha is not None,
            "mask_mean": float(alpha.mean()) if alpha is not None else None,
            "parser_crosscheck_maxabs": cross,
            "domain_default": domain_default,
            "de00_tet": C.quantiles(de_tet),
            "de00_tri": C.quantiles(de_tri),
            "de00_swap": C.quantiles(de_swap),
            "de00_jpeg_floor": C.quantiles(floor),
            "corr_raw": C.corr_matrix(tet, after),
            "corr_delta": C.corr_matrix(tet - i_in, after - i_in),
            "perm": C.permutation_test(tet, after),
        }
        return out
    except Exception as exc:  # noqa: BLE001
        return {"pair_id": row.get("pair_id"), "error":
                f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------

def viz_pair(row: dict, out_path: str, metrics: dict | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    i_in, after, alpha = load_pair_images(row)
    tet, tri, swap, cross, _ = render_all(row, i_in, alpha)
    de_tet = C.delta_e00_tiled(after, tet)
    de_swap = C.delta_e00_tiled(after, swap)

    ds = 2  # downscale for the panel
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.5))
    panels = [
        (i_in[::ds, ::ds], "I_in (bank source, preprocess 1024)", None),
        (after[::ds, ::ds], "archived after (production, JPEG q95)", None),
        (tet[::ds, ::ds], "re-render: colour tetrahedral", None),
        (swap[::ds, ::ds], "counterfactual: BGR-swapped apply", None),
        (alpha[::ds, ::ds] if alpha is not None
         else np.zeros(i_in.shape[:2])[::ds, ::ds],
         "C_GT alpha (l-line)" if alpha is not None else "no mask (g-line)",
         "gray"),
        (de_tet[::ds, ::ds], "dE00(after, tet)  vmax=4", "magma"),
        (de_swap[::ds, ::ds], "dE00(after, swap) vmax=4", "magma"),
        (np.abs(after - tet).mean(axis=-1)[::ds, ::ds] * 20,
         "|after - tet| x20", "gray"),
    ]
    for ax, (img, title, cmap) in zip(axes.ravel(), panels):
        if cmap == "magma":
            im = ax.imshow(img, cmap=cmap, vmin=0, vmax=4)
            fig.colorbar(im, ax=ax, fraction=0.035)
        elif cmap:
            ax.imshow(np.clip(img, 0, 1), cmap=cmap, vmin=0, vmax=1)
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    m = metrics or {}
    de_t = (m.get("de00_tet") or {})
    de_s = (m.get("de00_swap") or {})
    fl = (m.get("de00_jpeg_floor") or {})
    fig.suptitle(
        f"{row['pair_id']}  {row['line']}-line  {row['build']}  "
        f"preset={os.path.basename(row['preset_path'])} (N={row['lut_size']})  "
        f"pool={row['pool']}\n"
        f"dE00 after-vs-tet p50/p99 = {de_t.get('p50', float('nan')):.3f}/"
        f"{de_t.get('p99', float('nan')):.3f}   "
        f"jpeg-floor p50/p99 = {fl.get('p50', float('nan')):.3f}/"
        f"{fl.get('p99', float('nan')):.3f}   "
        f"swap-counterfactual p50 = {de_s.get('p50', float('nan')):.2f}   "
        f"best_perm={((m.get('perm') or {}).get('best_perm', '?'))}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=C.OUT_DIR)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--viz", nargs="*", default=None,
                    help="pair_ids to render viz panels for (skips metric pass)")
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.out_dir, "manifest.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
    if args.limit:
        rows = rows[: args.limit]

    if args.viz is not None:
        per_pair = {}
        mpath = os.path.join(args.out_dir, "metrics.json")
        if os.path.isfile(mpath):
            with open(mpath) as f:
                per_pair = {p["pair_id"]: p
                            for p in json.load(f).get("pairs", [])}
        viz_dir = os.path.join(args.out_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        by_id = {r["pair_id"]: r for r in rows}
        for pid in args.viz:
            row = by_id[pid]
            m = per_pair.get(pid)
            tag = "consistent"
            # mismatch = channel semantics wrong (perm != RGB) or residual not
            # explained by the per-pair JPEG floor (excess > 0.3 dE00 at p50)
            if m and (not m.get("perm", {}).get("identity_wins", True)
                      or (m.get("de00_tri", {}).get("p50", 0.0)
                          - m.get("de00_jpeg_floor", {}).get("p50", 0.0)) > 0.3):
                tag = "mismatch"
            out_path = os.path.join(viz_dir, f"{tag}_{pid}.png")
            viz_pair(row, out_path, m)
            print(f"[viz] {out_path}")
        return 0

    import multiprocessing as mp
    with mp.Pool(args.workers) as pool:
        results = pool.map(check_pair, rows)

    ok = [r for r in results if "error" not in r]
    errs = [r for r in results if "error" in r]

    def med(key_chain):
        vals = []
        for r in ok:
            v = r
            for k in key_chain:
                v = v[k]
            vals.append(v)
        return float(np.median(vals)) if vals else float("nan")

    def frac(pred):
        return sum(1 for r in ok if pred(r)) / max(1, len(ok))

    summary = {
        "n_pairs": len(results),
        "n_ok": len(ok),
        "n_error": len(errs),
        "identity_wins_all": all(r["perm"]["identity_wins"] for r in ok),
        "n_identity_wins": sum(r["perm"]["identity_wins"] for r in ok),
        "median_de00_tri_p50": med(["de00_tri", "p50"]),
        "median_de00_tri_p99": med(["de00_tri", "p99"]),
        "median_de00_tet_p50": med(["de00_tet", "p50"]),
        "median_de00_tet_p99": med(["de00_tet", "p99"]),
        "median_de00_swap_p50": med(["de00_swap", "p50"]),
        "median_jpeg_floor_p50": med(["de00_jpeg_floor", "p50"]),
        "median_jpeg_floor_p99": med(["de00_jpeg_floor", "p99"]),
        "frac_tri_p99_within_floor_plus_1": frac(
            lambda r: r["de00_tri"]["p99"] <= r["de00_jpeg_floor"]["p99"] + 1.0),
        "frac_tri_p50_le_1": frac(lambda r: r["de00_tri"]["p50"] <= 1.0),
        "max_parser_crosscheck": max((r["parser_crosscheck_maxabs"] for r in ok),
                                     default=float("nan")),
        "n_domain_nondefault": sum(1 for r in ok if not r["domain_default"]),
        "mean_corr_delta_diag": float(np.mean(
            [np.mean(np.diag(np.asarray(r["corr_delta"]))) for r in ok])),
        "mean_corr_delta_offdiag_max": float(np.mean(
            [np.max(np.asarray(r["corr_delta"])
                    - np.diag(np.diag(np.asarray(r["corr_delta"]))))
             for r in ok])),
    }
    out = {"summary": summary, "pairs": ok, "errors": errs}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if errs:
        for e in errs:
            print(f"[err] {e['pair_id']}: {e['error']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
