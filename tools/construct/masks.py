"""Mask sources for the INF-2 construct generator.

Two families:

1. Geometric samplers (radial / linear / elliptical / vignette) with
   smoothstep feathering (half-width = sigma px) and area targeting via
   bisection on a single scalar (mean(mask) is monotone in it).
2. Semantic bank: reads ``.cgt.png`` soft masks and ``.in.*`` source images
   straight out of the l-series build shards using the ``idx.jsonl`` offsets
   (ranged reads, no sequential tar scan), filtered by S-split.

Conventions (repo LUT_RENDERER protocol L303): RGB resize = Lanczos,
soft-mask resize = bilinear. Working resolution: long side 1024.
"""

from __future__ import annotations

import io
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .splits import make_splitter

LONG_SIDE = 1024
GEOM_FAMILIES = ["radial", "linear", "elliptical", "vignette"]
FEATHERS = [0, 2, 8, 24]          # px (PLAN §3)
AREAS = [0.05, 0.15, 0.40, 0.70]  # target mean(mask) (PLAN §3)
AREA_TOL = 0.01                   # |achieved - target| tolerance (abs)

# ----------------------------------------------------------------------------
# Geometric samplers
# ----------------------------------------------------------------------------


def smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def soft_edge(signal_px: np.ndarray, feather_px: float) -> np.ndarray:
    """signal>0 inside. Feathered edge = smoothstep over [-f, +f] px."""
    if feather_px <= 0:
        return (signal_px >= 0.0).astype(np.float32)
    return smoothstep((signal_px + feather_px) / (2.0 * feather_px)).astype(np.float32)


def _grid(h: int, w: int, stride: int = 1):
    ys = np.arange(0, h, stride, dtype=np.float64)
    xs = np.arange(0, w, stride, dtype=np.float64)
    return np.meshgrid(xs, ys)  # X, Y


def _bisect_area(make_mask, lo: float, hi: float, target: float,
                 increasing: bool, tol: float = AREA_TOL, iters: int = 48):
    """Bisect scalar param so mean(make_mask(param)) hits target.

    Returns (param, achieved_mean) or None if the target is unreachable.
    """
    f_lo = float(make_mask(lo).mean())
    f_hi = float(make_mask(hi).mean())
    a, b = (f_lo, f_hi) if increasing else (f_hi, f_lo)
    if target < a - tol or target > b + tol:
        return None
    p_lo, p_hi = lo, hi
    mid, f_mid = lo, f_lo
    for _ in range(iters):
        mid = 0.5 * (p_lo + p_hi)
        f_mid = float(make_mask(mid).mean())
        if abs(f_mid - target) < tol * 0.25:
            break
        if (f_mid < target) == increasing:
            p_lo = mid
        else:
            p_hi = mid
    if abs(f_mid - target) > tol:
        return None
    return mid, f_mid


@dataclass
class GeomMask:
    family: str
    mask: np.ndarray          # float32 HxW in [0,1]
    params: dict              # fully reproducible parameterization
    alpha_target: float | None
    alpha_achieved: float
    feather_px: float


class GeomMaskSampler:
    """Area-targeted geometric mask sampler with smoothstep feathering."""

    def __init__(self, solve_stride: int = 4, max_center_tries: int = 6):
        self.solve_stride = solve_stride
        self.max_center_tries = max_center_tries

    # -- per-family signal functions (px-scaled signed distance, >0 inside) --

    @staticmethod
    def _radial_signal(X, Y, cx, cy, r):
        return r - np.hypot(X - cx, Y - cy)

    @staticmethod
    def _elliptical_signal(X, Y, cx, cy, k, rho, phi):
        u = (X - cx) * np.cos(phi) + (Y - cy) * np.sin(phi)
        v = -(X - cx) * np.sin(phi) + (Y - cy) * np.cos(phi)
        a = k * np.sqrt(rho)
        b = k / np.sqrt(rho)
        d_n = np.sqrt((u / a) ** 2 + (v / b) ** 2)
        return (1.0 - d_n) * k  # approx px distance via geometric-mean radius

    @staticmethod
    def _linear_signal(X, Y, theta, t0):
        return t0 - (X * np.cos(theta) + Y * np.sin(theta))

    @staticmethod
    def _vignette_signal(X, Y, cx, cy, ax, by, d0):
        d_n = np.sqrt(((X - cx) / ax) ** 2 + ((Y - cy) / by) ** 2)
        return (d_n - d0) * np.sqrt(ax * by)  # mask = OUTER ring

    # -- sampling ------------------------------------------------------------

    def sample(self, family: str, h: int, w: int, rng: np.random.Generator,
               alpha: float, feather: float,
               force_center: tuple[float, float] | None = None,
               force_theta: float | None = None,
               force_t0: float | None = None) -> GeomMask | None:
        Xs, Ys = _grid(h, w, self.solve_stride)
        Xf, Yf = _grid(h, w, 1)
        diag = float(np.hypot(h, w))

        if family == "linear":
            theta = force_theta if force_theta is not None else float(rng.uniform(0, 2 * np.pi))
            if force_t0 is not None:
                # L7 mode: boundary pinned through a point, no area targeting
                sig = self._linear_signal(Xf, Yf, theta, force_t0)
                mask = soft_edge(sig, feather)
                return GeomMask("linear", mask,
                                {"theta": round(theta, 6), "t0": round(float(force_t0), 4)},
                                None, float(mask.mean()), feather)
            t = Xs * np.cos(theta) + Ys * np.sin(theta)
            lo, hi = float(t.min()) - 3 * feather - 1, float(t.max()) + 3 * feather + 1
            sol = _bisect_area(
                lambda t0: soft_edge(self._linear_signal(Xs, Ys, theta, t0), feather),
                lo, hi, alpha, increasing=True)
            if sol is None:
                return None
            t0, _ = sol
            mask = soft_edge(self._linear_signal(Xf, Yf, theta, t0), feather)
            return GeomMask("linear", mask,
                            {"theta": round(theta, 6), "t0": round(t0, 4)},
                            alpha, float(mask.mean()), feather)

        if family == "vignette":
            cx = float(rng.uniform(0.45, 0.55)) * w
            cy = float(rng.uniform(0.45, 0.55)) * h
            ax = 0.5 * w * float(rng.uniform(0.85, 1.2))
            by = 0.5 * h * float(rng.uniform(0.85, 1.2))
            d_max = float(np.sqrt((max(cx, w - cx) / ax) ** 2 + (max(cy, h - cy) / by) ** 2))
            sol = _bisect_area(
                lambda d0: soft_edge(self._vignette_signal(Xs, Ys, cx, cy, ax, by, d0), feather),
                0.0, d_max, alpha, increasing=False)
            if sol is None:
                return None
            d0, _ = sol
            mask = soft_edge(self._vignette_signal(Xf, Yf, cx, cy, ax, by, d0), feather)
            return GeomMask("vignette", mask,
                            {"cx": round(cx, 2), "cy": round(cy, 2),
                             "ax": round(ax, 2), "by": round(by, 2), "d0": round(d0, 5)},
                            alpha, float(mask.mean()), feather)

        # radial / elliptical share the center-retry loop
        for attempt in range(self.max_center_tries):
            span = (0.35, 0.65) if alpha >= 0.55 else (0.25, 0.75)
            if force_center is not None:
                cx, cy = force_center[0] * w, force_center[1] * h
            else:
                cx = float(rng.uniform(*span)) * w
                cy = float(rng.uniform(*span)) * h
            if family == "radial":
                sol = _bisect_area(
                    lambda r: soft_edge(self._radial_signal(Xs, Ys, cx, cy, r), feather),
                    1.0, 1.2 * diag, alpha, increasing=True)
                if sol is not None:
                    r, _ = sol
                    mask = soft_edge(self._radial_signal(Xf, Yf, cx, cy, r), feather)
                    return GeomMask("radial", mask,
                                    {"cx": round(cx, 2), "cy": round(cy, 2),
                                     "r": round(r, 2), "center_tries": attempt + 1},
                                    alpha, float(mask.mean()), feather)
            elif family == "elliptical":
                rho = float(rng.uniform(1.5, 4.0))
                phi = float(rng.uniform(0.0, np.pi))
                sol = _bisect_area(
                    lambda k: soft_edge(
                        self._elliptical_signal(Xs, Ys, cx, cy, k, rho, phi), feather),
                    1.0, 2.0 * diag, alpha, increasing=True)
                if sol is not None:
                    k, _ = sol
                    mask = soft_edge(self._elliptical_signal(Xf, Yf, cx, cy, k, rho, phi), feather)
                    return GeomMask("elliptical", mask,
                                    {"cx": round(cx, 2), "cy": round(cy, 2),
                                     "k": round(k, 2), "rho": round(rho, 4),
                                     "phi": round(phi, 5), "center_tries": attempt + 1},
                                    alpha, float(mask.mean()), feather)
            else:
                raise ValueError(f"unknown family {family}")
            if force_center is not None:
                return None
        return None


# ----------------------------------------------------------------------------
# Semantic bank (l-series shards, ranged reads via idx.jsonl)
# ----------------------------------------------------------------------------

DEFAULT_BUILDS = [
    "prod-l1-local17k-20260731",
    "prod-l2-local17k-20260731",
    "prod-l3-local17k-20260731",
    "prod-l4-local17k-20260801",
    "prod-l5-local17k-20260801",
    "prod-l6-local17k-20260801",
]
DEFAULT_ROOT = "/mnt/nfs/bc/data/datasets/sft"
CGT_VALID_RANGE = (0.02, 0.85)  # mean(cgt) admissibility gate


def read_member(tar_path: str | Path, offset: int, length: int) -> bytes:
    with open(tar_path, "rb") as fh:
        fh.seek(offset)
        return fh.read(length)


def _scan_one_batch(root: Path, build: str, batch: str) -> list[dict]:
    """Collect per-sample member offsets + vrmeta source_id for one batch."""
    bdir = root / build / batch
    idx_files = sorted((bdir / "indexes").glob("shard-*.idx.jsonl"))
    samples: dict[str, dict] = {}
    for idx in idx_files:
        with open(idx, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                sid = row["sample_id"]
                suffix = row["suffix"]
                rec = samples.setdefault(sid, {})
                rec[suffix] = {
                    "shard": row["shard"],
                    "offset": row["offset_data"],
                    "length": row["length"],
                }
    rows = []
    for sid, members in samples.items():
        vr = members.get(".vrmeta.json")
        cgt = members.get(".cgt.png")
        in_m = members.get(".in.jpg") or members.get(".in.png")
        if not (vr and cgt and in_m):
            continue
        tar = bdir / "shards" / f"{vr['shard']}.tar"
        try:
            meta = json.loads(read_member(tar, vr["offset"], vr["length"]))
        except Exception:
            continue
        rows.append({
            "sample_id": sid,
            "build": build,
            "batch": batch,
            "source_id": meta.get("source_id"),
            "slot_id": meta.get("slot_id"),
            "winner_confidence": meta.get("winner_confidence"),
            "subject_area": (meta.get("subject") or {}).get("area")
            if isinstance(meta.get("subject"), dict) else None,
            "in": in_m, "cgt": cgt, "vrmeta": vr,
        })
    return rows


def build_catalog(out_path: str | Path,
                  root: str | Path = DEFAULT_ROOT,
                  builds: list[str] | None = None,
                  batches_per_build: int = 4,
                  workers: int = 12) -> dict:
    """Scan idx.jsonl + vrmeta of the first N batches per build -> catalog jsonl."""
    root = Path(root)
    builds = builds or DEFAULT_BUILDS
    jobs = []
    for build in builds:
        batches = sorted(p.name for p in (root / build).glob("batch-*"))[:batches_per_build]
        for batch in batches:
            jobs.append((build, batch))
    all_rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for rows in ex.map(lambda j: _scan_one_batch(root, *j), jobs):
            all_rows.extend(rows)
    all_rows.sort(key=lambda r: r["sample_id"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in all_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {
        "rows": len(all_rows),
        "sources": len({r["source_id"] for r in all_rows}),
        "builds": builds,
        "batches_per_build": batches_per_build,
        "root": str(root),
    }
    return stats


class SemanticBank:
    """Split-filtered access to source images + .cgt.png soft masks."""

    def __init__(self, catalog_path: str | Path,
                 root: str | Path = DEFAULT_ROOT,
                 split_table: dict[str, str] | None = None,
                 split_table_path: str | Path | None = None):
        self.root = Path(root)
        self.splitter, self.split_rule = make_splitter(split_table,
                                                       split_table_path)
        self.by_source: dict[str, list[dict]] = {}
        with open(catalog_path, "r", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                sid = row.get("source_id")
                if not sid:
                    continue
                self.by_source.setdefault(sid, []).append(row)
        for rows in self.by_source.values():
            rows.sort(key=lambda r: r["sample_id"])

    def sources(self, split: str) -> list[str]:
        return sorted(s for s in self.by_source if self.splitter(s) == split)

    def rows_of(self, source_id: str) -> list[dict]:
        return self.by_source[source_id]

    def semantic_rows_of(self, source_id: str) -> list[dict]:
        """Only candidates whose mask slot is semantic (cgt = subject mask).

        The l-series builds mix geometric slots (linear-*/radial-*/band-*) with
        semantic ones (semantic-*); the ladder's semantic levels must not eat
        geometric slot masks (verified empirically: ~13.5% of candidates are
        semantic slots).
        """
        return [r for r in self.by_source[source_id]
                if str(r.get("slot_id", "")).startswith("semantic")]

    def _tar_path(self, row: dict, member: dict) -> Path:
        return self.root / row["build"] / row["batch"] / "shards" / f"{member['shard']}.tar"

    def load_image(self, row: dict) -> np.ndarray:
        """Source image -> float32 sRGB [0,1], long side LONG_SIDE (Lanczos)."""
        m = row["in"]
        raw = read_member(self._tar_path(row, m), m["offset"], m["length"])
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        scale = LONG_SIDE / max(w, h)
        if scale < 1.0:
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.Resampling.LANCZOS)
        return np.asarray(img, dtype=np.float32) / 255.0

    def load_cgt(self, row: dict, size_hw: tuple[int, int]) -> np.ndarray:
        """.cgt.png soft mask -> float32 [0,1] resized (bilinear) to (H, W)."""
        m = row["cgt"]
        raw = read_member(self._tar_path(row, m), m["offset"], m["length"])
        img = Image.open(io.BytesIO(raw)).convert("L")
        h, w = size_hw
        if img.size != (w, h):
            img = img.resize((w, h), Image.Resampling.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0

    @staticmethod
    def cgt_admissible(cgt: np.ndarray) -> bool:
        mean = float(cgt.mean())
        return CGT_VALID_RANGE[0] <= mean <= CGT_VALID_RANGE[1]
