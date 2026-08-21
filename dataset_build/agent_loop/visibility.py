"""P2 §S2.1 local visibility metric family.

Pre-registered columns for one local chain
(``global_after`` = before, ``final_after`` = after, mask ``alpha``):

===============  ==========================================================
``de_in``        alpha-weighted mean CIEDE2000 over pixels with ``alpha > tau``
``de_out``       unweighted mean CIEDE2000 over pixels with ``alpha <= tau``
``de_contrast``  ``de_in - de_out``
``edge_de``      unweighted mean CIEDE2000 over the transition band
                 ``edge_low <= alpha <= edge_high``
``edge_step_p95``  p95 of the per-pixel one-step CIEDE2000 difference inside
                 the transition band (max over right / down neighbour)
``floor_*``      the same three numbers on a deterministic equal-area random
                 rectangle (position drawn from ``sha256(seed_key)``)
===============  ==========================================================

Sampling: every region draws its own fixed budget **from that region's own
index pool**, not from the whole image, so ``n_in`` no longer scales with
``support_frac`` (the defect recorded in DESIGN_databuild_v2 §2.1, where
``render.py:573`` samples 4096 whole-image pixels and the mask keeps only
``4096 * support_frac`` of them).

:func:`visibility_components` (task card D1b) re-measures **the same support
sample** in CIELAB components instead of CIEDE2000: ``dL_* / dC_* / dHue_mean``
magnitudes, the signed means that carry direction, the ``L* > 85`` highlight
subset, and ``dL_grad`` / ``dL_step_mean`` for the structure of the dL map.

Nothing in this module is wired into the render / graph path; it is a
measurement module only.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
from skimage.color import rgb2lab

from .render import delta_e_map

# ------------------------------------------------------------------ constants
VISIBILITY_CONTRACT = "visibility-support-sampled-de00-v1"
COMPONENTS_CONTRACT = "visibility-lab-components-v1"

VISIBILITY_TAU = 0.05
EDGE_BAND_LOW = 0.2
EDGE_BAND_HIGH = 0.8

SAMPLE_BUDGET_MAX = 4096
SAMPLE_PIXELS_IN = 2048
SAMPLE_PIXELS_OUT = 2048
SAMPLE_PIXELS_EDGE = 1024

DE_IN_QUANTILES = (0.5, 0.9)
EDGE_STEP_QUANTILE = 0.95

# D1b: Lab component decomposition of the same support sample.
COMPONENT_QUANTILE = 0.9
HIGHLIGHT_L_THRESHOLD = 85.0

COMPONENT_COLUMNS = frozenset({
    "support_frac", "dL_mean", "dL_p90", "dL_signed_mean",
    "dC_mean", "dC_p90", "dC_signed_mean", "dHue_mean",
    "dL_highlight_mean", "dL_highlight_signed_mean", "highlight_frac",
    "dL_grad", "dL_step_mean",
    "n_in", "n_highlight", "components_contract",
})

REQUIRED_COLUMNS = frozenset({
    "support_frac", "de_in", "de_out", "de_contrast",
    "de_in_p50", "de_in_p90", "edge_de", "edge_step_p95",
    "floor_de_in", "floor_de_out", "floor_de_contrast",
    "de_in_over_floor", "de_in_minus_floor",
    "n_in", "n_out", "n_edge", "n_floor_in", "n_floor_out",
    "visibility_contract",
})


class VisibilityError(ValueError):
    pass


# ------------------------------------------------------------------ sampling
def _seed(seed_key: str, region: str) -> int:
    digest = hashlib.sha256(
        f"{VISIBILITY_CONTRACT}|{seed_key}|{region}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16)


def draw_region(pool: np.ndarray, budget: int, seed_key: str, region: str) -> np.ndarray:
    """Deterministic <=``budget`` subset of one region's flat index pool."""
    if budget > SAMPLE_BUDGET_MAX:
        raise VisibilityError("sample budget above SAMPLE_BUDGET_MAX")
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size <= budget:
        return pool
    rng = np.random.default_rng(_seed(seed_key, region))
    return pool[np.sort(rng.choice(pool.size, budget, replace=False))]


def random_area_mask(
    shape: tuple[int, int], support_frac: float, seed_key: str
) -> np.ndarray:
    """Equal-area random-position rectangle, image aspect ratio, fixed seed."""
    height, width = int(shape[0]), int(shape[1])
    frac = float(min(max(float(support_frac), 0.0), 1.0))
    mask = np.zeros((height, width), dtype=np.float32)
    if frac <= 0.0:
        return mask
    if frac >= 1.0:
        mask[:] = 1.0
        return mask
    scale = float(np.sqrt(frac))
    box_h = int(min(height, max(1, round(height * scale))))
    box_w = int(min(width, max(1, round(width * scale))))
    rng = np.random.default_rng(_seed(seed_key, "floor"))
    top = int(rng.integers(0, height - box_h + 1))
    left = int(rng.integers(0, width - box_w + 1))
    mask[top:top + box_h, left:left + box_w] = 1.0
    return mask


# ------------------------------------------------------------------ delta E
def delta_e_at(before: np.ndarray, after: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
    """CIEDE2000 at the given flat pixel indices of two HxWx3 float images."""
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise VisibilityError("shape mismatch")
    flat_index = np.asarray(flat_index, dtype=np.int64)
    if flat_index.size == 0:
        return np.zeros(0, dtype=np.float32)
    lhs = before.reshape(-1, 3)[flat_index].reshape(-1, 1, 3)
    rhs = after.reshape(-1, 3)[flat_index].reshape(-1, 1, 3)
    return np.asarray(delta_e_map(lhs, rhs), dtype=np.float32).reshape(-1)


def lab_at(image: np.ndarray, flat_index: np.ndarray) -> np.ndarray:
    """CIELAB (D65) of the given flat pixel indices of one HxWx3 float image."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise VisibilityError("shape mismatch")
    flat_index = np.asarray(flat_index, dtype=np.int64)
    if flat_index.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    pixels = image.reshape(-1, 3)[flat_index].reshape(-1, 1, 3)
    return np.asarray(rgb2lab(pixels), dtype=np.float64).reshape(-1, 3)


def lab_components(before_lab: np.ndarray, after_lab: np.ndarray) -> dict[str, np.ndarray]:
    """Per-pixel CIELAB decomposition ``(dL, dC, dH)`` of an edit.

    ``dL = L2 - L1``, ``dC = C*ab2 - C*ab1`` and the CIE hue difference
    ``dH = 2 * sqrt(C1 * C2) * sin(dh / 2)``; by construction
    ``dL**2 + dC**2 + dH**2 == dE*ab**2`` exactly.
    """
    before_lab = np.asarray(before_lab, dtype=np.float64)
    after_lab = np.asarray(after_lab, dtype=np.float64)
    if before_lab.shape != after_lab.shape or before_lab.ndim != 2 or before_lab.shape[1] != 3:
        raise VisibilityError("shape mismatch")
    d_l = after_lab[:, 0] - before_lab[:, 0]
    chroma_1 = np.hypot(before_lab[:, 1], before_lab[:, 2])
    chroma_2 = np.hypot(after_lab[:, 1], after_lab[:, 2])
    d_c = chroma_2 - chroma_1
    hue_1 = np.arctan2(before_lab[:, 2], before_lab[:, 1])
    hue_2 = np.arctan2(after_lab[:, 2], after_lab[:, 1])
    d_hue = np.arctan2(np.sin(hue_2 - hue_1), np.cos(hue_2 - hue_1))
    d_h = 2.0 * np.sqrt(np.maximum(chroma_1 * chroma_2, 0.0)) * np.sin(d_hue / 2.0)
    return {"dL": d_l, "dC": d_c, "dH": d_h}


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    if total <= 0.0:
        raise VisibilityError("zero total weight")
    return float(float((values * weights).sum()) / total)


def _neighbour_step(
    before: np.ndarray, after: np.ndarray, flat_index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel CIEDE2000 and the max one-step |difference| to right/down."""
    height, width = before.shape[0], before.shape[1]
    rows, cols = np.divmod(flat_index, width)
    right = rows * width + np.minimum(cols + 1, width - 1)
    down = np.minimum(rows + 1, height - 1) * width + cols
    stacked = np.concatenate([flat_index, right, down])
    values = delta_e_at(before, after, stacked)
    size = flat_index.size
    here = values[:size]
    step = np.maximum(
        np.abs(here - values[size:2 * size]), np.abs(here - values[2 * size:])
    )
    return here, step


# ------------------------------------------------------------------ metrics
def visibility_metrics(
    before: np.ndarray,
    after: np.ndarray,
    alpha: np.ndarray,
    *,
    seed_key: str,
    tau: float = VISIBILITY_TAU,
    edge_low: float = EDGE_BAND_LOW,
    edge_high: float = EDGE_BAND_HIGH,
) -> dict[str, Any]:
    """The pre-registered visibility column family for one local chain."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if before.shape != after.shape:
        raise VisibilityError("before/after shape mismatch")
    if alpha.shape != before.shape[:2]:
        raise VisibilityError("alpha shape mismatch")

    flat_alpha = alpha.reshape(-1)
    inside_pool = np.flatnonzero(flat_alpha > tau)
    outside_pool = np.flatnonzero(flat_alpha <= tau)
    if inside_pool.size == 0:
        raise VisibilityError("empty mask support")

    support_frac = float(inside_pool.size) / float(flat_alpha.size)

    inside = draw_region(inside_pool, SAMPLE_PIXELS_IN, seed_key, "in")
    de_inside = delta_e_at(before, after, inside)
    weights = flat_alpha[inside]
    de_in = _weighted_mean(de_inside, weights)
    p50, p90 = (float(v) for v in np.quantile(de_inside, DE_IN_QUANTILES))

    if outside_pool.size:
        outside = draw_region(outside_pool, SAMPLE_PIXELS_OUT, seed_key, "out")
        de_outside = delta_e_at(before, after, outside)
        de_out: float | None = float(de_outside.mean())
        n_out = int(outside.size)
    else:
        de_out, n_out = None, 0

    edge_pool = np.flatnonzero((flat_alpha >= edge_low) & (flat_alpha <= edge_high))
    if edge_pool.size:
        edge = draw_region(edge_pool, SAMPLE_PIXELS_EDGE, seed_key, "edge")
        here, step = _neighbour_step(before, after, edge)
        edge_de: float | None = float(here.mean())
        edge_step_p95: float | None = float(np.quantile(step, EDGE_STEP_QUANTILE))
        n_edge = int(edge.size)
    else:
        edge_de, edge_step_p95, n_edge = None, None, 0

    floor_alpha = random_area_mask(before.shape[:2], support_frac, seed_key).reshape(-1)
    floor_in_pool = np.flatnonzero(floor_alpha > tau)
    floor_out_pool = np.flatnonzero(floor_alpha <= tau)
    if floor_in_pool.size:
        floor_in_idx = draw_region(floor_in_pool, SAMPLE_PIXELS_IN, seed_key, "floor_in")
        floor_de_in: float | None = float(
            delta_e_at(before, after, floor_in_idx).mean()
        )
        n_floor_in = int(floor_in_idx.size)
    else:
        floor_de_in, n_floor_in = None, 0
    if floor_out_pool.size:
        floor_out_idx = draw_region(floor_out_pool, SAMPLE_PIXELS_OUT, seed_key, "floor_out")
        floor_de_out: float | None = float(
            delta_e_at(before, after, floor_out_idx).mean()
        )
        n_floor_out = int(floor_out_idx.size)
    else:
        floor_de_out, n_floor_out = None, 0

    floor_contrast = (
        float(floor_de_in - floor_de_out)
        if floor_de_in is not None and floor_de_out is not None else None
    )
    ratio = (
        float(de_in / floor_de_in)
        if floor_de_in not in (None, 0.0) else None
    )
    return {
        "visibility_contract": VISIBILITY_CONTRACT,
        "tau": float(tau),
        "edge_band": [float(edge_low), float(edge_high)],
        "support_frac": support_frac,
        "de_in": de_in,
        "de_in_p50": p50,
        "de_in_p90": p90,
        "de_out": de_out,
        "de_contrast": float(de_in - de_out) if de_out is not None else None,
        "edge_de": edge_de,
        "edge_step_p95": edge_step_p95,
        "floor_de_in": floor_de_in,
        "floor_de_out": floor_de_out,
        "floor_de_contrast": floor_contrast,
        "de_in_over_floor": ratio,
        "de_in_minus_floor": (
            float(de_in - floor_de_in) if floor_de_in is not None else None
        ),
        "n_in": int(inside.size),
        "n_out": n_out,
        "n_edge": n_edge,
        "n_floor_in": n_floor_in,
        "n_floor_out": n_floor_out,
    }


def visibility_components(
    before: np.ndarray,
    after: np.ndarray,
    alpha: np.ndarray,
    *,
    seed_key: str,
    tau: float = VISIBILITY_TAU,
    highlight_l: float = HIGHLIGHT_L_THRESHOLD,
) -> dict[str, Any]:
    """D1b lightness / chroma / hue decomposition on the D1 support sample.

    The support sample is drawn with the same ``draw_region(..., "in")`` call
    and the same ``seed_key`` as :func:`visibility_metrics`, so every column
    here is measured on exactly the pixels that produced ``de_in``.
    """
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    alpha = np.asarray(alpha, dtype=np.float32)
    if before.shape != after.shape:
        raise VisibilityError("before/after shape mismatch")
    if alpha.shape != before.shape[:2]:
        raise VisibilityError("alpha shape mismatch")

    height, width = before.shape[0], before.shape[1]
    flat_alpha = alpha.reshape(-1)
    inside_pool = np.flatnonzero(flat_alpha > tau)
    if inside_pool.size == 0:
        raise VisibilityError("empty mask support")
    support_frac = float(inside_pool.size) / float(flat_alpha.size)

    inside = draw_region(inside_pool, SAMPLE_PIXELS_IN, seed_key, "in")

    # one-step neighbour structure of the dL map, on the same sample
    rows, cols = np.divmod(inside, width)
    right = rows * width + np.minimum(cols + 1, width - 1)
    down = np.minimum(rows + 1, height - 1) * width + cols
    stacked = np.concatenate([inside, right, down])
    parts = lab_components(lab_at(before, stacked), lab_at(after, stacked))
    size = inside.size
    d_l = parts["dL"][:size]
    d_c = parts["dC"][:size]
    d_h = parts["dH"][:size]
    step = np.maximum(
        np.abs(d_l - parts["dL"][size:2 * size]),
        np.abs(d_l - parts["dL"][2 * size:]),
    )

    # highlight subset: support pixels whose *before* L* exceeds the threshold
    lab_support = lab_at(before, inside_pool)
    highlight_pool = inside_pool[lab_support[:, 0] > float(highlight_l)]
    highlight_frac = float(highlight_pool.size) / float(inside_pool.size)
    if highlight_pool.size:
        highlight = draw_region(highlight_pool, SAMPLE_PIXELS_IN, seed_key, "highlight")
        hi_dl = lab_components(lab_at(before, highlight), lab_at(after, highlight))["dL"]
        dl_highlight_mean: float | None = float(np.abs(hi_dl).mean())
        dl_highlight_signed: float | None = float(hi_dl.mean())
        n_highlight = int(highlight.size)
    else:
        dl_highlight_mean, dl_highlight_signed, n_highlight = None, None, 0

    return {
        "components_contract": COMPONENTS_CONTRACT,
        "tau": float(tau),
        "highlight_l": float(highlight_l),
        "support_frac": support_frac,
        "dL_mean": float(np.abs(d_l).mean()),
        "dL_p90": float(np.quantile(np.abs(d_l), COMPONENT_QUANTILE)),
        "dL_signed_mean": float(d_l.mean()),
        "dC_mean": float(np.abs(d_c).mean()),
        "dC_p90": float(np.quantile(np.abs(d_c), COMPONENT_QUANTILE)),
        "dC_signed_mean": float(d_c.mean()),
        "dHue_mean": float(np.abs(d_h).mean()),
        "dL_highlight_mean": dl_highlight_mean,
        "dL_highlight_signed_mean": dl_highlight_signed,
        "highlight_frac": highlight_frac,
        "dL_grad": float(d_l.std()),
        "dL_step_mean": float(step.mean()),
        "n_in": int(inside.size),
        "n_highlight": n_highlight,
    }


def assert_component_columns(row: Mapping[str, Any]) -> None:
    """Runtime assertion: every pre-registered D1b column was produced."""
    missing = sorted(COMPONENT_COLUMNS - set(row))
    if missing:
        raise VisibilityError(f"component columns missing: {missing}")
    if row.get("components_contract") != COMPONENTS_CONTRACT:
        raise VisibilityError("component contract mismatch")


def assert_visibility_columns(row: Mapping[str, Any]) -> None:
    """S2.7 runtime assertion: every pre-registered column was produced."""
    missing = sorted(REQUIRED_COLUMNS - set(row))
    if missing:
        raise VisibilityError(f"visibility columns missing: {missing}")
    if row.get("visibility_contract") != VISIBILITY_CONTRACT:
        raise VisibilityError("visibility contract mismatch")


__all__ = [
    "COMPONENTS_CONTRACT", "COMPONENT_COLUMNS", "COMPONENT_QUANTILE",
    "DE_IN_QUANTILES", "EDGE_BAND_HIGH", "EDGE_BAND_LOW", "EDGE_STEP_QUANTILE",
    "HIGHLIGHT_L_THRESHOLD",
    "REQUIRED_COLUMNS", "SAMPLE_BUDGET_MAX", "SAMPLE_PIXELS_EDGE", "SAMPLE_PIXELS_IN",
    "SAMPLE_PIXELS_OUT", "VISIBILITY_CONTRACT", "VISIBILITY_TAU", "VisibilityError",
    "assert_component_columns", "assert_visibility_columns", "delta_e_at", "draw_region",
    "lab_at", "lab_components", "random_area_mask", "visibility_components",
    "visibility_metrics",
]
