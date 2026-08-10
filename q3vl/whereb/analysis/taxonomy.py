"""Geometry-only classification of a GT mask.

Six single-dimension strata.  The full cross would be 2*3*2*2*3*2 = 144 cells
over 400 local V_where samples, i.e. a median computed on two samples; §13 of the
protocol asks for coverage of "small/medium/large", "ring / thin boundary /
multi-component" masks, which is exactly a set of *marginal* strata.

**Nothing here reads a prediction.**  A class label that depended on the model
would make "the model is bad on class X" unfalsifiable, and the report's whole
point is to be able to say which class is hard.

Every number is defined operationally below and calibrated against synthetic
shapes in ``q3vl/whereb/tests/test_analysis_taxonomy.py`` -- a disc, a square, a
ring, two blobs -- so the definitions are pinned by measurement rather than by
assertion.  Measured on this implementation, 256x256 frame:

===================  ==========  ==========  ==============
shape                perimeter   analytic    circularity
===================  ==========  ==========  ==============
disc r=60            393.99      376.99      0.915  (1.000)
square 120x120       476.00      480.00      0.799  (0.785)
annulus r=60/30      --          --          0.300  (0.334)
===================  ==========  ==========  ==============

i.e. the estimator runs +4.5% on a disc and -0.8% on a square, and the shape
ordering (disc > square > annulus) is preserved.  The residual anisotropy is the
reason the class cut is at 0.30 and not at some value inherited from a paper: it
is set against *this* estimator's measured scale, and the unit test pins those
three numbers so a future change to the estimator cannot silently move the cut.

Softness note: the published ``.maskhi.png`` view is the ``.cgt`` candidate-region
mask, which is **soft-edged by construction** (single channel, anti-aliased and
feathered).  ``soft_frac`` measures how much of the region is in that transition
band, which is the axis the min/max soft-IoU is most sensitive to.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

__all__ = [
    "DIMENSIONS", "AREA_CLASSES", "TaxonomyConfig", "perimeter_marching_squares",
    "mask_geometry", "classify_geometry", "global_labels", "GLOBAL_CLASS",
    "GEOMETRY_QUANTILE_KEYS",
]

#: raw quantities whose p10/p50/p90 the report prints next to the class cuts
GEOMETRY_QUANTILE_KEYS: tuple[str, ...] = (
    "area_frac", "circularity", "centroid_dist_rel", "soft_frac", "mask_max",
    "largest_component_frac",
)

#: the six geometry dimensions, in report order
DIMENSIONS: tuple[str, ...] = (
    "area", "components", "topology", "boundary", "position", "softness",
)

#: what a global (all-ones GT) sample is labelled as on every dimension.  Global
#: samples are not a *kind of region*; they are the absence of one, and mixing
#: them into a geometry stratum would put 496 trivially-perfect rows into the
#: "large / solid / centre" cells and drown the local signal.
GLOBAL_CLASS = "global"


@dataclass(frozen=True)
class TaxonomyConfig:
    """Every threshold, in one recordable object (it is dumped to config/).

    The defaults are **calibrated on the observed V_where distribution**, which
    the task card asks for explicitly ("阈值可配、记录分布后微调").  What was
    measured on the 400 local samples, and what each cut had to answer to:

    ==================  ====================================  =====================
    quantity            observed p10 / p50 / p90              default cut, and why
    ==================  ====================================  =====================
    ``area_frac``       0.104 / 0.409 / 0.764                 (0.05, 0.15, 0.45)
    ``circularity``     0.345 / 0.680 / 0.832                 0.50 -> 24% complex
    ``centroid_dist``   0.059 / 0.181 / 0.370                 0.15 / 0.30
    ``soft_frac``       0.193 / 0.369 / 0.649 (relative)      0.40 -> 47% soft
    ==================  ====================================  =====================

    The area cut keeps the task card's ``< 5%`` boundary as its own class rather
    than folding it away, but 5%/25% alone would have put **279 of 400** samples
    in "large" -- these are ``.cgt`` candidate regions, whose median covers 41%
    of the frame, not COCO-style objects.  A stratum holding 70% of the split
    cannot answer "which kind of region is hard", so the two extra cuts split
    that mass at 15% and 45%.  The p10/p50/p90 row above is reprinted in every
    report, so a reader can always see the cut against the distribution it faced.
    """

    #: binarisation of the soft GT mask for all geometry.  The same 0.5 the
    #: eval's ``gt_area_k`` uses, so "area" here and ``k`` there agree.  Safe on
    #: this data: the smallest per-mask maximum measured on V_where is 0.561.
    binarize: float = 0.5
    #: area_frac cuts, ascending -> classes AREA_CLASSES
    area_cuts: tuple[float, ...] = (0.05, 0.15, 0.45)
    #: a connected component is counted only if it holds at least this share of
    #: the mask's own area AND this many pixels -- anti-aliasing specks along a
    #: feathered edge are not "a second region"
    min_component_frac: float = 0.02
    min_component_px: int = 64
    #: a hole counts only if it is at least this share of the filled area
    min_hole_frac: float = 0.01
    min_hole_px: int = 64
    #: circularity = 4*pi*A / P^2 in [0, 1]; 1.0 is a disc.  Below this the
    #: boundary is long for the area it encloses ("complex")
    compact_min: float = 0.50
    #: |centroid - frame centre| in short-side units, divided by the corner
    #: distance, so it is in [0, 1] for any aspect ratio
    center_max: float = 0.15
    edge_min: float = 0.30
    #: the transition band, **relative to the mask's own maximum**.  The ``.cgt``
    #: view's amplitude is a candidate-confidence scale, not geometry: the
    #: per-mask maximum on V_where runs 0.561 to 1.000 with a median of 1.000, so
    #: an absolute (0.05, 0.95) band labels a 0.64-amplitude mask "100% soft"
    #: when its edge may be one pixel wide -- it would be classifying amplitude.
    #: Rescaling here is a *classification of the GT*; no criterion, and no
    #: colour scale, is ever computed from the rescaled mask (red line).
    soft_lo: float = 0.05
    soft_hi: float = 0.95
    #: share of the support that must be in the band for "soft"
    soft_frac_min: float = 0.40

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: area class names, one more than there are cuts
AREA_CLASSES: tuple[str, ...] = ("tiny", "small", "medium", "large")


# --- perimeter --------------------------------------------------------------

# Marching-square style perimeter: each border pixel is classified by its
# neighbourhood and contributes 1, sqrt(2) or (1+sqrt(2))/2.  A plain
# crack-following count (number of foreground/background pixel edges) is
# strongly anisotropic -- it is exact for an axis-aligned square but 8R for a
# disc of radius R against the true 2*pi*R = 6.28R, a +27% bias that would make
# every round region look "complex" and every rectangle look "compact", i.e. the
# classification would be measuring orientation.  The weighted form is calibrated
# in the unit test on both shapes.
_KERNEL = np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]], dtype=np.int32)
_WEIGHTS = np.zeros(50, dtype=np.float64)
_WEIGHTS[[5, 7, 15, 17, 25, 27]] = 1.0
_WEIGHTS[[21, 33]] = math.sqrt(2.0)
_WEIGHTS[[13, 23]] = (1.0 + math.sqrt(2.0)) / 2.0


def perimeter_marching_squares(binary: np.ndarray) -> float:
    """Perimeter of a binary mask, in pixels."""
    from scipy import ndimage as ndi

    b = np.ascontiguousarray(binary).astype(bool)
    if not b.any():
        return 0.0
    cross = ndi.generate_binary_structure(2, 1)
    eroded = ndi.binary_erosion(b, cross, border_value=0)
    border = (b & ~eroded).astype(np.uint8)
    conv = ndi.convolve(border, _KERNEL, mode="constant", cval=0)
    hist = np.bincount(conv.ravel(), minlength=_WEIGHTS.size)
    return float(hist[: _WEIGHTS.size] @ _WEIGHTS)


# --- geometry ---------------------------------------------------------------

def _corner_distance(h: int, w: int) -> float:
    """Frame-corner distance in the campaign's ``short_side_unit`` coordinates.

    Same convention as ``q3vl.where.phi.norm_coords`` (and therefore as the
    centre-prior field): both axes divided by half the short side, so the short
    side spans [-1, 1] and a cell is square in feature space.
    """
    half = min(h, w) / 2.0
    return math.hypot((h / 2.0) / half, (w / 2.0) / half)


def mask_geometry(mask: Any, cfg: TaxonomyConfig | None = None) -> dict[str, Any]:
    """Raw geometric quantities of one soft GT mask ``(H, W)`` in ``[0, 1]``.

    Returns numbers only -- the bucketing lives in :func:`classify_geometry`, so
    a threshold change never requires re-reading the masks.
    """
    from scipy import ndimage as ndi

    cfg = cfg or TaxonomyConfig()
    m = np.asarray(mask, dtype=np.float64)
    if m.ndim != 2:
        raise ValueError(f"expected a 2D mask, got shape {m.shape}")
    h, w = m.shape
    n_px = h * w
    b = m > cfg.binarize
    area = int(b.sum())
    out: dict[str, Any] = {
        "height": h, "width": w,
        "area_px": area,
        "area_frac": area / n_px,
        "gt_mean": float(m.mean()),
    }
    if area == 0:
        # An all-zero GT is not a region; it is a data defect.  Reported as such
        # rather than silently classified as "small, solid, compact".
        out.update({
            "n_components": 0, "n_components_raw": 0, "largest_component_frac": 0.0,
            "n_holes": 0, "hole_frac": 0.0, "perimeter_px": 0.0,
            "circularity": None, "centroid_y": None, "centroid_x": None,
            "centroid_dist": None, "centroid_dist_rel": None,
            "soft_frac": 0.0, "support_frac": 0.0, "degenerate": "empty_mask",
        })
        return out

    # components: 8-connectivity for the foreground (a diagonal chain is one
    # region), which pairs with 4-connectivity on the background for holes --
    # the standard opposite-connectivity convention that keeps the Euler number
    # consistent.
    lab, n_raw = ndi.label(b, structure=np.ones((3, 3), dtype=int))
    sizes = np.bincount(lab.ravel())[1:] if n_raw else np.zeros(0, dtype=int)
    floor = max(cfg.min_component_px, int(round(cfg.min_component_frac * area)))
    kept = int((sizes >= floor).sum())
    out["n_components_raw"] = int(n_raw)
    out["n_components"] = max(1, kept)      # the largest component always counts
    out["largest_component_frac"] = float(sizes.max() / area) if n_raw else 0.0

    filled = ndi.binary_fill_holes(b)
    holes = filled & ~b
    hlab, n_hraw = ndi.label(holes, structure=ndi.generate_binary_structure(2, 1))
    hsizes = np.bincount(hlab.ravel())[1:] if n_hraw else np.zeros(0, dtype=int)
    hfloor = max(cfg.min_hole_px, int(round(cfg.min_hole_frac * int(filled.sum()))))
    out["n_holes"] = int((hsizes >= hfloor).sum())
    out["hole_frac"] = float(holes.sum() / max(1, int(filled.sum())))

    per = perimeter_marching_squares(b)
    out["perimeter_px"] = per
    # 4*pi*A / P^2: exactly 1 for a disc, pi/4 for a square, -> 0 as the boundary
    # gets long for the area it encloses.  Clipped at 1 because a few-pixel blob's
    # discrete perimeter can undershoot the continuous one.
    out["circularity"] = float(min(1.0, 4.0 * math.pi * area / (per * per))) if per > 0 else None

    ys, xs = np.nonzero(b)
    cy, cx = float(ys.mean()) + 0.5, float(xs.mean()) + 0.5
    half = min(h, w) / 2.0
    dy, dx = (cy - h / 2.0) / half, (cx - w / 2.0) / half
    out["centroid_y"], out["centroid_x"] = cy, cx
    out["centroid_dist"] = float(math.hypot(dy, dx))
    out["centroid_dist_rel"] = float(out["centroid_dist"] / _corner_distance(h, w))

    mx = float(m.max())
    out["mask_max"], out["mask_mean_raw"] = mx, float(m.mean())
    rel = m / mx if mx > 0 else m
    support = rel > cfg.soft_lo
    band = support & (rel < cfg.soft_hi)
    out["support_frac"] = float(support.sum() / n_px)
    out["soft_frac"] = float(band.sum() / max(1, int(support.sum())))
    # kept next to it so the two definitions can be compared in one place: the
    # absolute-band version is what Where-A's maskmeta reports as `frac_soft`,
    # and on a sub-unit-amplitude mask it saturates at 1.0
    abs_support = m > cfg.soft_lo
    out["soft_frac_absolute"] = float(
        (abs_support & (m < cfg.soft_hi)).sum() / max(1, int(abs_support.sum())))
    out["degenerate"] = None
    return out


def classify_geometry(geom: dict[str, Any],
                      cfg: TaxonomyConfig | None = None) -> dict[str, str]:
    """Bucket the raw quantities into one label per dimension."""
    cfg = cfg or TaxonomyConfig()
    if geom.get("degenerate"):
        return {d: f"degenerate:{geom['degenerate']}" for d in DIMENSIONS}
    a = geom["area_frac"]
    area = AREA_CLASSES[sum(1 for c in cfg.area_cuts if a >= c)]
    comps = "single" if geom["n_components"] <= 1 else "multi"
    topo = "holed" if geom["n_holes"] >= 1 else "solid"
    circ = geom.get("circularity")
    boundary = ("unknown" if circ is None
                else ("compact" if circ >= cfg.compact_min else "complex"))
    d = geom["centroid_dist_rel"]
    pos = "center" if d <= cfg.center_max else ("edge" if d > cfg.edge_min else "mid")
    soft = "soft" if geom["soft_frac"] >= cfg.soft_frac_min else "hard"
    return {"area": area, "components": comps, "topology": topo,
            "boundary": boundary, "position": pos, "softness": soft}


def global_labels() -> dict[str, str]:
    """The label set for a global (all-ones GT) sample."""
    return {d: GLOBAL_CLASS for d in DIMENSIONS}
