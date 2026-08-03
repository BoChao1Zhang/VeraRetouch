"""Shared helpers for F5 bgr risk cross-check (tools/bgr_check).

Purpose: verify whether the production renderer's `axis_order:"bgr"` LUT
storage convention yields after-images that are semantically RGB-consistent
with the colour-science tetrahedral path (the RD-G Stage-1 L_cube supervision
path).

Interpreter: /home/bc/miniconda3/bin/python3 (colour-science 0.4.7, numpy,
PIL, matplotlib) — same as tools/cube (see tools/cube/README.md).

Conventions used here (all locally verified, see NOTES.md):
- production parse: dataset_build.lut_io.load_lut -> grid[b, g, r] axis
  order, RGB value channels (rows R-fastest reshaped C-order).
- colour parse: colour.io.read_LUT_IridasCube -> lut.table[r, g, b]
  (rows R-fastest reshaped order='F'), RGB value channels.
- identity relation to verify per pair: lut.table == grid.transpose(2,1,0,3).
"""
from __future__ import annotations

import io
import json
import os
import sys
from typing import Any, Iterator

import numpy as np

sys.path.insert(0, "/home/bc/VeraRetouch")          # dataset_build namespace pkg
sys.path.insert(0, "/home/bc/VeraRetouch/tools/cube")  # cubelib (T2, reviewed)

JOURNAL_ROOT = "/var/cache/veradata/annot_review/journal-archive"
DATASETS_ROOT = "/mnt/nfs/bc/data/datasets"
IMG_BANK_ROOT = os.path.join(DATASETS_ROOT, "img", "unknown")
OUT_DIR = "/home/bc/VeraRetouch/experiments/tooling-wave1/bgr_check"

G_BUILDS = [
    "prod-g1-global25k-20260731",
    "prod-g2-global25k-20260731",
    "prod-g3-global25k-20260801",
]
L_BUILDS = [
    "prod-l1-local17k-20260731",
    "prod-l2-local17k-20260731",
    "prod-l3-local17k-20260731",
    "prod-l4-local17k-20260801",
]

# journal pool -> (img bank dirs, accepted roles); mirrors
# tools/data_splits/action_g_render_audit.py (T1, reviewed)
BANK_FOR_POOL = {
    "unsplash": (["unsplash", "unsplash_work"], {"primary"}),
    "awards": (["awards"], {"primary"}),
    "quandian": (["quandian"], {"primary"}),
    "korean": (["korean"], {"primary"}),
    "greysky": (["greysky"], {"primary"}),
    "ppr10k": (["ppr10k"], {"source"}),
    "raise6k": (["raise6k"], {"preview"}),  # raw NEF not decodable here -> skip
    "fivek_gold": (["fivek_gold"], {"before.jpg"}),
}
FULL_PATH_BANKS = {"fivek_gold"}
LOCAL_PATH_POOLS = {"mmart_ppr10k"}


def pool_of(source_path: str) -> str:
    p = source_path
    if "/presets_sources/" in p:
        return p.split("/presets_sources/")[1].split("/")[0]
    if "/_scratch/unsplash/" in p:
        return "unsplash"
    if "/ppr10k/source/" in p:
        return "ppr10k"
    if "/RAISE-6k/" in p:
        return "raise6k"
    if "/fivek_gold/" in p:
        return "fivek_gold"
    if "MMArt-PPR10k" in p:
        return "mmart_ppr10k"
    return "other"


def stem_of(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0].lower()


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------

def iter_group_lines(build: str, wanted: set[int]) -> Iterator[tuple[int, dict]]:
    """Yield (line_no, parsed_group) for exactly the wanted 0-based lines."""
    path = os.path.join(JOURNAL_ROOT, build, "groups.jsonl")
    left = set(wanted)
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in left:
                left.discard(i)
                yield i, json.loads(line)
                if not left:
                    return


def count_lines(build: str) -> int:
    path = os.path.join(JOURNAL_ROOT, build, "groups.jsonl")
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


# ---------------------------------------------------------------------------
# tar ranged read (ustar, uncompressed; same convention as T5 oracle.py)
# ---------------------------------------------------------------------------

def read_member(tar_path: str, offset_data: int, length: int) -> bytes:
    with open(tar_path, "rb") as f:
        f.seek(offset_data)
        data = f.read(length)
    if len(data) != length:
        raise IOError(f"short read {len(data)}/{length} from {tar_path}@{offset_data}")
    return data


def find_candidates_in_build(build: str, candidate_ids: list[str]) -> dict[str, dict]:
    """candidate_id -> {suffix -> {tar, offset_data, length, sha256}} via one
    rg -F pass over the build's idx.jsonl files."""
    import subprocess
    import tempfile

    root = os.path.join(DATASETS_ROOT, "groups", build)
    out: dict[str, dict] = {}
    if not candidate_ids:
        return out
    with tempfile.NamedTemporaryFile("w", suffix=".pats", delete=False) as tf:
        tf.write("\n".join(candidate_ids) + "\n")
        pats = tf.name
    try:
        proc = subprocess.run(
            ["rg", "-I", "--no-heading", "-F", "-f", pats, "-j", "32",
             "-g", "*.idx.jsonl", root],
            capture_output=True, text=True, check=False)
        for line in proc.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = row.get("sample_id", "")
            pos = sid.rfind("candidate_")
            if pos < 0:
                continue
            cid = sid[pos:]
            if cid not in set(candidate_ids):
                continue
            # logical_path: groups/<build>/<batch>/<member>
            parts = row["logical_path"].split("/")
            batch = parts[2]
            tar = os.path.join(root, batch, "shards", row["shard"] + ".tar")
            out.setdefault(cid, {})[row["suffix"]] = {
                "tar": tar,
                "offset_data": int(row["offset_data"]),
                "length": int(row["length"]),
                "sha256": row.get("sha256", ""),
            }
    finally:
        os.unlink(pats)
    return out


# ---------------------------------------------------------------------------
# img bank source resolution
# ---------------------------------------------------------------------------

class BankResolver:
    """source_path -> member reference {tar, offset_data, length} using
    metadata.jsonl (role rows) + catalog.sqlite3 (members table)."""

    def __init__(self) -> None:
        # bank -> (full_path -> sample_id, stem -> sample_id)
        self._meta: dict[str, tuple[dict[str, str], dict[str, str]]] = {}

    def _load_bank(self, bank: str, roles: set[str]) -> tuple[dict[str, str], dict[str, str]]:
        if bank in self._meta:
            return self._meta[bank]
        by_full: dict[str, str] = {}
        by_stem: dict[str, str] = {}
        meta = os.path.join(IMG_BANK_ROOT, bank, "metadata.jsonl")
        if os.path.isfile(meta):
            with open(meta) as f:
                for line in f:
                    r = json.loads(line)
                    if r.get("role") not in roles:
                        continue
                    sp = r.get("source_path", "")
                    by_full[sp] = r["sample_id"]
                    by_stem[stem_of(sp)] = r["sample_id"]
        self._meta[bank] = (by_full, by_stem)
        return self._meta[bank]

    def resolve(self, pool: str, source_path: str) -> dict | None:
        import sqlite3

        if pool in LOCAL_PATH_POOLS:
            if os.path.isfile(source_path):
                return {"kind": "local", "path": source_path}
            return None
        if pool not in BANK_FOR_POOL:
            return None
        banks, roles = BANK_FOR_POOL[pool]
        want_suffix = os.path.splitext(source_path)[1].lower()
        # exact full-path match across banks first (e.g. _scratch/unsplash work
        # copies live in unsplash_work with the verbatim prod source_path);
        # basename-stem match is only a fallback and may hit a different
        # encode of the same photo.
        ordered: list[tuple[str, str, str]] = []  # (bank, sample_id, matched_by)
        for bank in banks:
            by_full, _ = self._load_bank(bank, roles)
            sid = by_full.get(source_path)
            if sid is not None:
                ordered.append((bank, sid, "full_path"))
        if pool not in FULL_PATH_BANKS:
            for bank in banks:
                _, by_stem = self._load_bank(bank, roles)
                sid = by_stem.get(stem_of(source_path))
                if sid is not None:
                    ordered.append((bank, sid, "stem"))
        for bank, sample_id, matched_by in ordered:
            db = os.path.join(IMG_BANK_ROOT, bank, "indexes", "catalog.sqlite3")
            if not os.path.isfile(db):
                continue
            con = sqlite3.connect(db)
            try:
                cols = [r[1] for r in con.execute("PRAGMA table_info(members)")]
                off = "offset_data" if "offset_data" in cols else "offset"
                ln = "length" if "length" in cols else "size"
                rows = con.execute(
                    f"SELECT suffix, shard, {off}, {ln} FROM members WHERE sample_id=?",
                    (sample_id,)).fetchall()
            finally:
                con.close()
            best = None
            for suffix, shard, offset_data, length in rows:
                if suffix.lower() == want_suffix:
                    best = (suffix, shard, offset_data, length)
                    break
                if best is None and suffix.lower() in (".jpg", ".jpeg", ".png"):
                    best = (suffix, shard, offset_data, length)
            if best is None:
                continue
            tar = os.path.join(IMG_BANK_ROOT, bank, "shards", best[1] + ".tar")
            return {"kind": "bank", "tar": tar,
                    "offset_data": int(best[2]), "length": int(best[3]),
                    "bank": bank, "sample_id": sample_id, "suffix": best[0],
                    "matched_by": matched_by}
        return None


# ---------------------------------------------------------------------------
# image pipeline (replicates dataset_build.src.construct.rendering exactly)
# ---------------------------------------------------------------------------

def decode_oriented_rgb(data: bytes):
    """PIL decode + EXIF orientation + RGB, matching archive_reader.open_rgb."""
    from PIL import Image, ImageOps

    return ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")


def preprocess_source_bytes(data: bytes, short_edge: int = 1024) -> np.ndarray:
    """rendering.preprocess_source semantics on in-memory bytes."""
    from PIL import Image

    # Any: 运行时为 Image.Resampling（新 Pillow）或 Image 模块（旧 Pillow 兜底），
    # 两者均有 LANCZOS；静态收窄到 Module 会误报 LANCZOS 缺失。
    resampling: Any = getattr(Image, "Resampling", Image)
    oriented = decode_oriented_rgb(data)
    scale = short_edge / min(oriented.width, oriented.height)
    width = max(1, int(round(oriented.width * scale)))
    height = max(1, int(round(oriented.height * scale)))
    if oriented.size != (width, height):
        oriented = oriented.resize((width, height), resampling.LANCZOS)
    return np.asarray(oriented, dtype=np.float32) / 255.0


def composite_prod(before: np.ndarray, edited: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """rendering.composite_srgb / GPU masked-blend semantics (endpoint snap)."""
    weight = np.clip(alpha, 0.0, 1.0).astype(np.float32)[..., None]
    mixed = before * (1.0 - weight) + edited * weight
    mixed = np.where(weight == 0, before, np.where(weight == 1, edited, mixed))
    return mixed.astype(np.float32)


def apply_lut_trilinear_bgr(image: np.ndarray, grid_bgr: np.ndarray,
                            domain_min=None, domain_max=None) -> np.ndarray:
    """Trilinear on grid[b,g,r] — verbatim math of the production test oracle
    apply_lut_cpu_oracle (dataset_build/src/construct/rendering.py:77-109),
    which mirrors the GPU grid_sample path."""
    source = np.asarray(image, dtype=np.float32)
    grid = np.asarray(grid_bgr, dtype=np.float32)
    dmin = np.zeros(3, dtype=np.float32) if domain_min is None else np.asarray(domain_min)
    dmax = np.ones(3, dtype=np.float32) if domain_max is None else np.asarray(domain_max)
    span = np.where(dmax == dmin, 1.0, dmax - dmin)
    coords = np.clip((source - dmin) / span, 0.0, 1.0) * (grid.shape[0] - 1)
    lo = np.floor(coords).astype(np.int32)
    hi = np.minimum(lo + 1, grid.shape[0] - 1)
    frac = coords - lo
    r0, g0, b0 = lo[..., 0], lo[..., 1], lo[..., 2]
    r1, g1, b1 = hi[..., 0], hi[..., 1], hi[..., 2]
    fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
    c000 = grid[b0, g0, r0]
    c100 = grid[b0, g0, r1]
    c010 = grid[b0, g1, r0]
    c110 = grid[b0, g1, r1]
    c001 = grid[b1, g0, r0]
    c101 = grid[b1, g0, r1]
    c011 = grid[b1, g1, r0]
    c111 = grid[b1, g1, r1]
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return np.clip(c0 * (1 - fb) + c1 * fb, 0.0, 1.0).astype(np.float32)


def read_cube_colour(path: str):
    """colour parse -> (table_rgb[r,g,b,3] float64, domain (2,3))."""
    import colour

    lut = colour.io.read_LUT_IridasCube(path)
    if not hasattr(lut, "table") or lut.table.ndim != 4:
        raise ValueError(f"not a 3D LUT: {path}")
    return np.asarray(lut.table, dtype=np.float64), np.asarray(lut.domain, dtype=np.float64)


def apply_lut_tetrahedral_rgb(table_rgb: np.ndarray, img: np.ndarray,
                              domain: np.ndarray | None = None,
                              tile_rows: int = 256) -> np.ndarray:
    """colour tetrahedral interpolation (the RD-G L_cube supervision path).

    table_rgb: (S,S,S,3) index [r,g,b]; img: (H,W,3) in [0,1].
    """
    from colour import LUT3D
    from colour.algebra import table_interpolation_tetrahedral

    if domain is not None:
        lut = LUT3D(table_rgb.astype(np.float64), domain=domain.astype(np.float64))
    else:
        lut = LUT3D(table_rgb.astype(np.float64))
    out = np.empty(img.shape, dtype=np.float32)
    for y0 in range(0, img.shape[0], tile_rows):
        y1 = min(y0 + tile_rows, img.shape[0])
        out[y0:y1] = lut.apply(
            img[y0:y1].astype(np.float64),
            interpolator=table_interpolation_tetrahedral,
        ).astype(np.float32)
    return np.clip(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

QUANTS = (50, 90, 95, 99, 99.9)


def delta_e00_tiled(rgb_a: np.ndarray, rgb_b: np.ndarray, tile_rows: int = 256) -> np.ndarray:
    """CIEDE2000 between two gamma-encoded sRGB arrays in [0,1] (D65, via
    colour — same lab basis as tools/cube cubelib)."""
    import colour

    out = np.empty(rgb_a.shape[:2], dtype=np.float32)
    for y0 in range(0, rgb_a.shape[0], tile_rows):
        y1 = min(y0 + tile_rows, rgb_a.shape[0])
        lab_a = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb_a[y0:y1], 0, 1)))
        lab_b = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(np.clip(rgb_b[y0:y1], 0, 1)))
        out[y0:y1] = colour.difference.delta_E(lab_a, lab_b, method="CIE 2000")
    return out


def quantiles(de_map: np.ndarray) -> dict:
    flat = de_map.reshape(-1)
    q = np.percentile(flat, QUANTS)
    return {
        "mean": float(flat.mean()),
        **{f"p{str(p).replace('.', '_')}": float(v) for p, v in zip(QUANTS, q)},
        "max": float(flat.max()),
    }


def corr_matrix(a: np.ndarray, b: np.ndarray) -> list[list[float]]:
    """3x3 Pearson r between channels of a (rows) and b (cols)."""
    fa = a.reshape(-1, 3).astype(np.float64)
    fb = b.reshape(-1, 3).astype(np.float64)
    fa -= fa.mean(axis=0)
    fb -= fb.mean(axis=0)
    sa = np.sqrt((fa ** 2).sum(axis=0))
    sb = np.sqrt((fb ** 2).sum(axis=0))
    sa[sa == 0] = 1.0
    sb[sb == 0] = 1.0
    return (fa.T @ fb / np.outer(sa, sb)).tolist()


PERMS = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]


def permutation_test(pred: np.ndarray, after: np.ndarray) -> dict:
    """MAE of every channel permutation of pred vs after; best should be RGB."""
    maes = {}
    for p in PERMS:
        maes["".join("RGB"[i] for i in p)] = float(
            np.abs(pred[..., list(p)] - after).mean())
    best = min(maes, key=lambda k: maes[k])  # 等价 maes.get；dict.get 的 Optional 返回类型过不了 min 的 key 签名
    return {"mae_by_perm": maes, "best_perm": best, "identity_wins": best == "RGB"}


def jpeg_roundtrip(pred: np.ndarray, quality: int = 95) -> np.ndarray:
    """save_candidate_jpeg encode semantics (round, uint8, PIL JPEG q95
    default options) then decode back to float [0,1]."""
    from PIL import Image

    img = Image.fromarray(
        np.clip(np.asarray(pred) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
