"""LUT selection + GT cache for A0 / E1.

A0 (GLUT protocol): 75 LUTs, 64^3 native resolution (GLUT's 75-LUT anchor
subset is their 64^3 slice; the exact files are not public, so we take a
deterministic 75-LUT sample from the local 447 64^3 cubes -- decision logged
in NOTES.md).  GT is computed from the ORIGINAL cube at native resolution via
colour tetrahedral interpolation (protocol GT path).

E1 (Stage-0): production presets at canonical 33^3 (D-CUBE canonical npy),
400-LUT subset stratified by `minor` from tools/data_splits/splits.sqlite3.

GT cache layout (float16, values in [0,1]):
  {cache}/a0/{lut_id}.train.npy   (2_097_152, 3)   128^3 train colors
  {cache}/a0/{lut_id}.eval.npy    (14_680_064, 3)  full held-out colors
  {cache}/e1/{lut_id}.train.npy   (2_097_152, 3)
  {cache}/e1/{lut_id}.evalsub.npy (2_097_152, 3)   fixed-subsample held-out
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "tools", "cube"))

from cubelib import (  # noqa: E402
    DEFAULT_NPY_DIR,
    apply_lut_tetrahedral,
    hald_eval_image,
    hald_train_image,
)

MANIFEST = os.path.join(
    _REPO, "experiments/tooling-wave1/cube/inventory/dcube_manifest.jsonl")
PARSE_REPORT = os.path.join(
    _REPO, "experiments/tooling-wave1/cube/parse/parse_report.jsonl")
NEAR_IDENTITY = os.path.join(
    _REPO, "experiments/tooling-wave1/cube/selfcheck/near_identity_stats.jsonl")
SPLITS_DB = os.path.join(_REPO, "tools/data_splits/splits.sqlite3")
GT_CACHE = "/var/cache/veradata/dcube/gtcache"
NILUT_LUT01 = "/var/cache/veradata/dcube/nilut/LUT01.cube"

EVALSUB_SEED = 12345
EVALSUB_SIZE = 2 ** 21  # 2,097,152 of the 14,680,064 held-out colors


# ---------------------------------------------------------------------------
# Corpus tables
# ---------------------------------------------------------------------------

def _read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_manifest() -> dict[str, dict]:
    return {r["id"]: r for r in _read_jsonl(MANIFEST)}


def load_parse_report() -> dict[str, dict]:
    return {r["id"]: r for r in _read_jsonl(PARSE_REPORT) if r.get("ok")}


def near_identity_ids() -> set[str]:
    return {r["id"] for r in _read_jsonl(NEAR_IDENTITY) if r["near_identity"]}


# ---------------------------------------------------------------------------
# Selections (deterministic, seed 0)
# ---------------------------------------------------------------------------

def select_a0_75(seed: int = 0) -> list[dict]:
    """75 local 64^3 cubes: non-near-identity, md5-deduped, prod-used first."""
    man = load_manifest()
    rep = load_parse_report()
    near = near_identity_ids()
    cand = []
    seen_md5 = set()
    for lid, r in sorted(rep.items()):
        if r.get("orig_size") != 64 or r.get("lut_type") != "3D":
            continue
        if lid in near or lid not in man:
            continue
        md5 = man[lid]["md5"]
        if md5 in seen_md5:
            continue
        seen_md5.add(md5)
        cand.append({"id": lid, "path": man[lid]["path"],
                     "used": bool(man[lid]["used_in_prod"])})
    used = [c for c in cand if c["used"]]
    unused = [c for c in cand if not c["used"]]
    rng = np.random.default_rng(seed)
    rng.shuffle(used)
    rng.shuffle(unused)
    pick = (used + unused)[:75]
    return sorted(pick, key=lambda c: c["id"])


def select_e1_400(seed: int = 0, n_total: int = 400) -> list[dict]:
    """400 production presets stratified by `minor` (splits.sqlite3), joined
    to cube ids via the manifest; near-identity excluded; canonical npy33."""
    man = load_manifest()
    near = near_identity_ids()
    con = sqlite3.connect(SPLITS_DB)
    minor_of = {pid: (minor, split) for pid, minor, split in
                con.execute("SELECT preset_id, minor, split FROM presets")}
    con.close()
    # cube id -> (minor, preset_id); production cubes only
    rows = []
    for lid, r in sorted(man.items()):
        if not r["used_in_prod"] or lid in near:
            continue
        pids = r.get("preset_ids") or []
        if not pids or pids[0] not in minor_of:
            continue
        minor, split = minor_of[pids[0]]
        npy = os.path.join(DEFAULT_NPY_DIR, f"{lid}.npy")
        if not os.path.exists(npy):
            continue
        rows.append({"id": lid, "npy": npy, "minor": minor,
                     "preset_id": pids[0], "split": split})
    # proportional stratified sample over minors, >=1 per minor
    by_minor: dict[str, list[dict]] = {}
    for r in rows:
        by_minor.setdefault(r["minor"], []).append(r)
    rng = np.random.default_rng(seed)
    minors = sorted(by_minor)
    total = sum(len(v) for v in by_minor.values())
    alloc = {m: max(1, int(round(n_total * len(by_minor[m]) / total)))
             for m in minors}
    # adjust to exactly n_total
    while sum(alloc.values()) > n_total:
        m = max(minors, key=lambda k: (alloc[k], k))
        if alloc[m] > 1:
            alloc[m] -= 1
    while sum(alloc.values()) < n_total:
        m = max(minors, key=lambda k: (len(by_minor[k]) - alloc[k], k))
        alloc[m] += 1
    pick = []
    for m in minors:
        pool = sorted(by_minor[m], key=lambda r: r["id"])
        rng.shuffle(pool)
        pick.extend(pool[:alloc[m]])
    return sorted(pick, key=lambda r: r["id"])[:n_total]


# ---------------------------------------------------------------------------
# GT computation
# ---------------------------------------------------------------------------

def load_lut_native(path: str):
    """Original-resolution colour LUT object (dialect fallbacks included)."""
    sys.path.insert(0, os.path.join(_REPO, "tools", "cube"))
    from parse import _read_cube_with_fallback  # noqa: E402
    meta: dict = {}
    return _read_cube_with_fallback(path, meta), meta


def apply_native_tetrahedral(path: str, colors: np.ndarray,
                             tile: int = 1 << 20) -> np.ndarray:
    """Apply the ORIGINAL cube (native size) to (P,3) colors, tetrahedral."""
    from colour.algebra import table_interpolation_tetrahedral
    from colour.utilities import suppress_warnings
    lut, _ = load_lut_native(path)
    out = np.empty_like(colors, dtype=np.float32)
    with suppress_warnings(python_warnings=True):
        for i in range(0, colors.shape[0], tile):
            j = min(i + tile, colors.shape[0])
            try:
                out[i:j] = lut.apply(
                    colors[i:j].astype(np.float64),
                    interpolator=table_interpolation_tetrahedral,
                ).astype(np.float32)
            except TypeError:  # LUT classes without interpolator kw (3x1D etc)
                out[i:j] = lut.apply(colors[i:j].astype(np.float64)) \
                    .astype(np.float32)
    return np.clip(out, 0.0, 1.0)


def apply_npy33_tetrahedral(npy_path: str, colors: np.ndarray) -> np.ndarray:
    table = np.load(npy_path)
    img = colors.reshape(-1, 1, 3)
    out = apply_lut_tetrahedral(table, img, tile_rows=1 << 18)
    return np.clip(out.reshape(-1, 3), 0.0, 1.0)


_TRAIN_COLORS: np.ndarray | None = None
_EVAL_COLORS: np.ndarray | None = None


def train_colors() -> np.ndarray:
    global _TRAIN_COLORS
    if _TRAIN_COLORS is None:
        _TRAIN_COLORS = hald_train_image().reshape(-1, 3)
    return _TRAIN_COLORS


def eval_colors() -> np.ndarray:
    global _EVAL_COLORS
    if _EVAL_COLORS is None:
        _EVAL_COLORS = hald_eval_image().reshape(-1, 3)
    return _EVAL_COLORS


def evalsub_indices() -> np.ndarray:
    n_eval = 14_680_064
    rng = np.random.default_rng(EVALSUB_SEED)
    return np.sort(rng.choice(n_eval, size=EVALSUB_SIZE, replace=False))


def _cache_path(kind: str, lut_id: str, part: str) -> str:
    return os.path.join(GT_CACHE, kind, f"{lut_id}.{part}.npy")


def build_gt(kind: str, lut_id: str, src_path: str,
             parts: tuple[str, ...]) -> dict[str, str]:
    """Ensure GT cache entries exist; returns {part: path}.

    kind='a0': src_path = original .cube, parts from {train, eval}
    kind='e1': src_path = canonical npy33, parts from {train, evalsub}
    """
    os.makedirs(os.path.join(GT_CACHE, kind), exist_ok=True)
    out = {}
    for part in parts:
        p = _cache_path(kind, lut_id, part)
        out[part] = p
        if os.path.exists(p):
            continue
        if part == "train":
            colors = train_colors()
        elif part == "eval":
            colors = eval_colors()
        elif part == "evalsub":
            colors = eval_colors()[evalsub_indices()]
        else:
            raise ValueError(part)
        if kind == "a0":
            gt = apply_native_tetrahedral(src_path, colors)
        else:
            gt = apply_npy33_tetrahedral(src_path, colors)
        tmp = p + ".tmp.npy"
        np.save(tmp, gt.astype(np.float16))
        os.replace(tmp, p)
    return out


def _gt_worker(job):
    kind, lut_id, src, parts = job
    try:
        build_gt(kind, lut_id, src, tuple(parts))
        return lut_id, None
    except Exception as e:  # noqa: BLE001
        return lut_id, repr(e)


def build_gt_parallel(jobs, workers: int = 8):
    """jobs: [(kind, lut_id, src_path, parts)] -> list of (id, err|None)."""
    from multiprocessing import Pool
    with Pool(workers) as pool:
        return list(pool.imap_unordered(_gt_worker, jobs))
