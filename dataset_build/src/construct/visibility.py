"""Pre-JPEG alpha-weighted CIEDE2000 visibility gate."""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


_RESAMPLING = getattr(Image, "Resampling", Image)


class VisibilityError(ValueError):
    """Raised for invalid image or weight inputs."""


@dataclass(frozen=True, slots=True)
class VisibilityMetrics:
    visible_de: float
    visible_fraction: float
    accepted: bool


@dataclass(frozen=True, slots=True)
class WorkingImageReference:
    source_shape: tuple[int, int, int]
    pixels: np.ndarray
    lab: np.ndarray
    # Per-pixel facts about the BEFORE image that every candidate in the group
    # reuses: the blown-out highlights and the colour-name surface each pixel
    # belongs to.  Both are v5 hint inputs (see below) and both are functions of
    # the before image alone, so they are computed once per group.
    highlight: np.ndarray
    surface: np.ndarray


@dataclass(frozen=True, slots=True)
class TorchLabReference:
    lab: Any
    device: str
    lock: Any
    highlight: Any
    surface: Any


# --- v5 objective edit-direction metrics (WP15c) ---------------------------
#
# The five axis definitions, the colour-surface table and the silence gate below
# are the WP15ab ROC winners, scored on the fresh200 n=160 Direction-List panel
# (/var/cache/veradata/annot_review/wp15_metric_roc/, winning_spec.json).  They
# replace the four first-moment spatial means of v4.1, whose failure mode is
# documented in docs/HINTS_MEAN_BIAS_2026-07-29.md.
#
# EVERY NUMBER IN THIS SECTION IS PART OF A METRIC DEFINITION, NOT A TUNING KNOB.
# None of it is reachable from the TOML on purpose: a build's recorded direction
# only means something relative to the thresholds it was measured against, so
# moving one of these silently invalidates the hints of every earlier build.  If
# one has to move, it moves here, with the ROC evidence written next to it.

# Warm/cool is a direction in the (da*, db*) plane, not b* alone.  WP15a's
# 721780ad moves b* by +4.1 -- "warmer" under the old b*-only rule -- while a*
# moves -9.1, and the blind reviewer reads the frame as cooler and greener;
# 7213ca74 is the mirror image.  The ROC swept fixed angles rather than fitting
# one, and 70 deg on the high-alpha core won (hi.proj70, AUC 0.959, against the
# production support.full.d_b baseline's 0.949).
WARM_ANGLE_DEG = 70.0
# The high-alpha core is the part of the mask the edit actually hits.  The
# threshold is relative to the map's own peak because a linear or radial ramp
# scaled by amount < 1 never reaches an absolute 0.75 anywhere, which would make
# an absolute cut degenerate on exactly the slots that need it most.
HI_ALPHA_RATIO = 0.75
# Blown-out highlights, read on 0-255 sRGB.  A pixel already at the rail cannot
# move further, which is why v5.1 measured contrast on the pixels that were not
# (HINTS_MEAN_BIAS §3.1); v5.2's tone curve does not need that filter -- see the
# block below -- so the rails survive only as the two fractions the journal
# records for diagnosis.
CLIP_HIGH_255 = 250.0

# --- v5.2 contrast: the tone curve, not the tonal spread (WP18 fix 1) -------
#
# v5.1 measured contrast as a difference of alpha-weighted L* standard
# deviations with the clipping rails dropped.  The fresh150 factory panel
# (2026-07-30) proved that proxy structurally blind in both directions, and the
# two reversals it produced are opposite in sign, so no threshold move repairs
# them:
#
#   p066 (sft_85dc238d): the judge fitted a tone curve inside the mask core and
#     found before-L 30-40 mapping to 17.7 and 70-80 mapping to 84.4 -- a 40-unit
#     input range opened to 67 units, a 1.67x expansion, "contrast is clearly
#     RAISED".  The stored hint said "lower contrast" (-3.25) because two thirds
#     of the band is backdrop that the edit flattened to a constant: flattening
#     removes *spread* without touching the slope the eye reads.
#   p050 (sft_a47d94b7): blacks lifted from p5 5.7 to 19.7 and the frame went
#     visibly flat and washed out, yet the stored hint said "higher contrast"
#     (+2.27).  Those crushed blacks sit on the low rail, so the noclip support
#     deleted exactly the pixels carrying the change.
#
# So the axis is now the slope of the tone curve itself, and it is read on the
# high-alpha core rather than on a rail-filtered support -- the curve caps a
# rail's influence at one band on its own (see TONE_BAND_WEIGHT_CAP below),
# which is what the noclip filter was there to do.
#
# Scale.  The reported number stays in L* units and stays comparable with the
# v5.1 figure by construction: for an edit that is a linear tone map
# L2 = g*L1 + c, every band mean satisfies y = g*x + c exactly, so the fitted
# slope is g whatever the band weights are, and
#     (slope - 1) * std(L1) == std(L2) - std(L1)
# identically.  The two statistics differ only where the map is *not* linear --
# which is precisely the p050/p066 structure -- so a dead band in L* units keeps
# meaning what it meant.
#
# Bands are 10 L* wide over the full 0-100 scale, which is the instrument the
# judge quoted ("before-L 30-40", "70-80"), and each occupied band contributes
# one point (mean before-L, mean after-L) to a least-squares fit.
TONE_BANDS = 10
# A band holding less than this share of the core is a handful of stray pixels,
# not a tone anybody looks at; it is dropped rather than allowed to lever the
# fit.  Below 1% the AUC on the WP15a Direction-List contrast labels falls from
# 0.920 to 0.916 and the fit starts chasing single-pixel bands.
TONE_BAND_MIN_MASS = 0.01
# No band may speak for more than an equal share of the curve -- that is the
# whole repair, stated as arithmetic: p066's flattened backdrop owns two thirds
# of the mask and would otherwise decide the slope exactly as it decides the
# variance.  A band thinner than an equal share still counts only its true mass,
# so a sliver cannot outvote a real tone either.  Weighting bands by their raw
# mass instead reproduces the v5.1 blindness (p066 comes back at -2.79);
# weighting them all equally fixes p066 but drops the label AUC to 0.900.
TONE_BAND_WEIGHT_CAP = 1.0 / TONE_BANDS

# Colour-name surfaces.  van de Weijer's w2c LUT is not obtainable without a
# download, so this is the sanctioned fallback carried over verbatim from the
# WP15b reference implementation (dataset_build/tools/metric_bank.py): an
# explicit rule set over CIELAB assigning each pixel one of the 11 Berlin-Kay
# basic colour terms.  A nearest-centroid in Lab was tried first and rejected --
# the canonical sRGB primaries sit at chroma 80+ while real photographic
# surfaces sit at 10-40, so every natural pixel fell into the achromatic
# centroids.  Assignment is hue-angle based for chromatic pixels and lightness
# based for achromatic ones, with brown and pink split off by lightness as the
# colour-naming literature does.
ACHROMATIC_CHROMA = 12.0     # C* below this reads as black / grey / white
SURFACE_HUE_BINS = (         # (name, lo_deg, hi_deg) over h = atan2(b*, a*)
    ("red", 345.0, 25.0),
    ("orange", 25.0, 60.0),
    ("yellow", 60.0, 100.0),
    ("green", 100.0, 180.0),
    ("blue", 180.0, 285.0),
    ("purple", 285.0, 325.0),
    ("pink", 325.0, 345.0),
)
COLOUR_SURFACES = ("black", "grey", "white", "red", "orange", "yellow", "green",
                   "blue", "purple", "pink", "brown")
# Per-surface silence gate.  A surface is only worth naming when it owns real
# estate *and* its chroma moved past the judge's measured discrimination floor:
# WP14 puts sol's colour-axis JND at 1.4-1.8 and luna's at 3.4-7.4, and the
# WP15b operating points for an asserted chroma direction land at |dC| ~ 6 once
# the support is restricted rather than whole-frame.  Below either floor the
# table says nothing rather than handing the annotator a claim it cannot check.
SURFACE_MIN_AREA = 0.05
SURFACE_MIN_D_C = 6.0
# At most two surfaces are named.  A longer list reads as a checklist and the
# WP14 anchoring result (a wrong hint drives luna from 55.8% to 5.8%) says every
# extra asserted line is a liability, not extra information.
SURFACE_TOP_N = 2
# Where the surface sits, in thirds of the frame (WP18 fix 2).  A colour name on
# its own does not say *which* red thing, and the fresh150 panel caught the
# annotator naming the most conspicuous object of that colour anywhere in the
# picture instead of inside the edit: on one row it named a brown guitar that
# "sits at mask value 0.01 and measures L 22.3->22.1, chroma 10.8->10.8, i.e.
# zero change".  The centroid is alpha-weighted over the surface's own pixels,
# so it points at the part of the frame the edit actually reaches.
SURFACE_POSITION_THIRDS = 3
# Below this weighted mass a support set is empty for all practical purposes.
_MIN_SUPPORT_MASS = 1e-6

_HINT_WORDS: dict[str, tuple[str, str]] = {
    "brightness": ("brighter", "darker"),
    "warmth": ("warmer", "cooler"),
    "hue_gm": ("shifted toward magenta/red", "shifted toward green"),
    "chroma": ("richer", "more muted"),
    "contrast": ("higher contrast", "lower contrast"),
}


@dataclass(frozen=True, slots=True)
class HintSupport:
    """Per-pixel support masks the v5 metrics need beyond the Lab pair.

    ``surface`` indexes ``COLOUR_SURFACES`` and is always read off the *before*
    image: the annotator is told "the red things got duller", which is a claim
    about what was red to start with, not about what is red afterwards.

    The highlight masks are the two blown-out fractions the journal records.
    v5.1 also carried an "at either rail" mask, because contrast was measured on
    the pixels it excluded; v5.2's tone curve caps a rail's influence at one
    band, so that support -- and the p050 reversal it caused -- is gone.
    """

    before_highlight: np.ndarray
    after_highlight: np.ndarray
    surface: np.ndarray


def highlight_mask(rgb: np.ndarray) -> np.ndarray:
    """Pixels at the highlight rail in one 0..1 RGB image.

    Written channel-wise rather than as ``(rgb * 255).max(axis=-1)``.  The
    obvious form materialises the whole scaled HxWx3 array and then reduces over
    a length-3 *innermost* axis, which NumPy walks with its generic reduction
    loop: at the 512-short-edge working size that measured 49 ms per candidate,
    which was 84% of the entire visibility-and-hints call and the whole of the
    WP15c render-phase regression.  Scaling after the max is exact -- the
    multiply is monotonic and elementwise -- so this is the same mask.
    """
    channels = np.asarray(rgb, dtype=np.float32)
    red, green, blue = channels[..., 0], channels[..., 1], channels[..., 2]
    highest = np.maximum(np.maximum(red, green), blue)
    return highest * 255.0 >= CLIP_HIGH_255


def colour_surface_labels(lab: np.ndarray) -> np.ndarray:
    """Per-pixel index into ``COLOUR_SURFACES``."""
    lightness, a_star, b_star = np.moveaxis(np.asarray(lab, dtype=np.float64), -1, 0)
    chroma = np.hypot(a_star, b_star)
    hue = np.degrees(np.arctan2(b_star, a_star)) % 360.0
    out = np.full(lightness.shape, COLOUR_SURFACES.index("grey"), dtype=np.int8)

    achromatic = chroma < ACHROMATIC_CHROMA
    out[achromatic & (lightness < 25.0)] = COLOUR_SURFACES.index("black")
    out[achromatic & (lightness >= 75.0)] = COLOUR_SURFACES.index("white")

    chromatic = ~achromatic
    for name, low, high in SURFACE_HUE_BINS:
        if low > high:                   # the red bin wraps through 0
            selected = chromatic & ((hue >= low) | (hue < high))
        else:
            selected = chromatic & (hue >= low) & (hue < high)
        out[selected] = COLOUR_SURFACES.index(name)
    # brown is dark orange/yellow; pink is light, moderate-chroma red
    warm = chromatic & (hue >= 20.0) & (hue < 100.0)
    out[warm & (lightness < 45.0)] = COLOUR_SURFACES.index("brown")
    reddish = chromatic & ((hue >= 325.0) | (hue < 25.0))
    out[reddish & (lightness >= 60.0) & (chroma < 50.0)] = COLOUR_SURFACES.index("pink")
    return out


def hint_support(reference: WorkingImageReference, after: np.ndarray) -> HintSupport:
    """Pair one candidate's after image with the group's cached before masks."""
    return HintSupport(
        before_highlight=reference.highlight,
        after_highlight=highlight_mask(after),
        surface=reference.surface,
    )


def _weighted_mean(values: np.ndarray, weight: np.ndarray, total: float) -> float:
    return float((np.asarray(values, dtype=np.float64) * weight).sum() / total)


def _weighted_std(values: np.ndarray, weight: np.ndarray, total: float) -> float:
    mean = _weighted_mean(values, weight, total)
    return math.sqrt(max(0.0, _weighted_mean((values - mean) ** 2, weight, total)))


def _direction(name: str, value: float) -> str:
    positive, negative = _HINT_WORDS[name]
    if abs(value) < 1e-6:
        return "unchanged"
    return positive if value > 0 else negative


def tone_band_labels(lightness: np.ndarray) -> np.ndarray:
    """Which fixed-width L* band each pixel's BEFORE lightness falls in.

    Deliberately a fixed grid over 0-100 rather than mass quantiles: quantile
    bands are defined by the mass distribution, which is the thing p066 shows
    cannot be trusted to decide the answer, and a fixed grid is also the same
    integer arithmetic on both backends, so band membership is bit-equal.
    """
    width = 100.0 / TONE_BANDS
    return np.clip(
        (np.asarray(lightness, dtype=np.float64) / width).astype(np.int64),
        0, TONE_BANDS - 1,
    )


def _band_weights(mass: np.ndarray, total: float) -> np.ndarray:
    """Per-band fit weight: floored for thinness, capped at an equal share."""
    keep = mass > max(_MIN_SUPPORT_MASS, TONE_BAND_MIN_MASS * total)
    return np.where(keep, np.minimum(mass, TONE_BAND_WEIGHT_CAP * total), 0.0)


def tone_curve_contrast(
    l_before: np.ndarray,
    l_after: np.ndarray,
    weight: np.ndarray,
    total: float,
) -> float:
    """``(tone-curve slope - 1) * spread(before)``, in L* units.

    Falls back to the v5.1 spread difference on the same support when the curve
    cannot be fitted -- fewer than two bands carry weight, or every band that
    does sits at the same lightness.  That is a flat or single-tone region, where
    a slope is undefined and the spread difference is all there is to say.
    """
    spread_before = _weighted_std(l_before, weight, total)
    labels = tone_band_labels(l_before).reshape(-1)
    flat_weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    mass = np.bincount(labels, weights=flat_weight, minlength=TONE_BANDS)
    sum_x = np.bincount(labels, weights=flat_weight * np.asarray(l_before).reshape(-1),
                        minlength=TONE_BANDS)
    sum_y = np.bincount(labels, weights=flat_weight * np.asarray(l_after).reshape(-1),
                        minlength=TONE_BANDS)
    safe_mass = np.where(mass > 0.0, mass, 1.0)
    band_x = sum_x / safe_mass
    band_y = sum_y / safe_mass
    band_weight = _band_weights(mass, total)
    band_total = float(band_weight.sum())
    if band_total > _MIN_SUPPORT_MASS:
        centre_x = float((band_weight * band_x).sum()) / band_total
        centre_y = float((band_weight * band_y).sum()) / band_total
        variance = float((band_weight * (band_x - centre_x) ** 2).sum()) / band_total
        if variance > _MIN_SUPPORT_MASS:
            covariance = float(
                (band_weight * (band_x - centre_x) * (band_y - centre_y)).sum()
            ) / band_total
            return (covariance / variance - 1.0) * spread_before
    return _weighted_std(l_after, weight, total) - spread_before


def _position_word(centre_y: float, centre_x: float) -> str:
    """A weighted centroid in [0, 1]^2 as the coarse place a reader would say."""
    thirds = SURFACE_POSITION_THIRDS
    row = min(thirds - 1, max(0, int(centre_y * thirds)))
    column = min(thirds - 1, max(0, int(centre_x * thirds)))
    vertical = ("upper", "", "lower")[row]
    horizontal = ("left", "", "right")[column]
    if vertical and horizontal:
        return f"{vertical} {horizontal}"
    if vertical:
        return f"{vertical} half"
    if horizontal:
        return f"{horizontal} half"
    return "middle of the frame"


def _surface_rows(
    counts: np.ndarray,
    sums: dict[str, np.ndarray],
    total: float,
    region_d_c: float,
) -> list[dict[str, float | str | bool]]:
    """Gate the 11-surface table down to the at most two worth naming."""
    gated: list[tuple[float, dict[str, float | str | bool]]] = []
    for index, name in enumerate(COLOUR_SURFACES):
        mass = float(counts[index])
        if mass <= _MIN_SUPPORT_MASS:
            continue
        area = mass / total
        d_c = float(sums["d_C"][index]) / mass
        if area < SURFACE_MIN_AREA or abs(d_c) < SURFACE_MIN_D_C:
            continue
        row: dict[str, float | str | bool] = {
            "name": name,
            "area": round(area, 6),
            "d_L": round(float(sums["d_L"][index]) / mass, 6),
            "d_a": round(float(sums["d_a"][index]) / mass, 6),
            "d_b": round(float(sums["d_b"][index]) / mass, 6),
            "d_C": round(d_c, 6),
            "direction": _direction("chroma", d_c),
            "position": _position_word(float(sums["cy"][index]) / mass,
                                       float(sums["cx"][index]) / mass),
            # The whole-support figure is the one the ROC operating point was
            # fitted on, so where a surface disagrees with it in sign the region
            # keeps authority and the surface line is demoted rather than
            # dropped -- dropping it is how v4.1 lost the lipstick case.
            "low_confidence": bool(region_d_c * d_c < 0.0),
        }
        # Salience, as in the WP15b reference implementation: a big change on a
        # surface that owns real estate, not the biggest change anywhere.
        gated.append((abs(d_c) * math.sqrt(area), row))
    gated.sort(key=lambda item: item[0], reverse=True)
    return [row for _, row in gated[:SURFACE_TOP_N]]


def objective_edit_hints_from_lab(
    first: np.ndarray,
    second: np.ndarray,
    *,
    weight: np.ndarray,
    support: HintSupport,
) -> dict[str, Any]:
    """Compute edit hints from a validated, equal-grid Lab pair."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    normalized_weight = np.asarray(weight, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 3 or first.shape[2] != 3:
        raise VisibilityError("Lab arrays must be equal-shape HWC arrays")
    if normalized_weight.shape != first.shape[:2] \
            or not np.isfinite(normalized_weight).all() or np.any(normalized_weight < 0):
        raise VisibilityError("weight map is invalid")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise VisibilityError("Lab arrays contain non-finite values")
    total = float(normalized_weight.sum())
    if not math.isfinite(total) or total <= 0:
        raise VisibilityError("weight map has no mass")
    for mask in (support.before_highlight, support.after_highlight, support.surface):
        if np.shape(mask) != first.shape[:2]:
            raise VisibilityError("hint support masks must match the Lab grid")

    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    d_l, d_a, d_b, d_c = l2 - l1, a2 - a1, b2 - b1, c2 - c1

    # -- whole-support axes (unchanged from v4.1 by ROC verdict) -------------
    brightness = _weighted_mean(d_l, normalized_weight, total)
    chroma = _weighted_mean(d_c, normalized_weight, total)
    hue_gm = _weighted_mean(d_a, normalized_weight, total)

    # -- warmth and contrast on the high-alpha core -------------------------
    # The peak is > 0 because the total is, so this support is never empty.
    core = normalized_weight >= HI_ALPHA_RATIO * float(normalized_weight.max())
    core_weight = normalized_weight * core
    core_total = float(core_weight.sum())
    core_d_a = _weighted_mean(d_a, core_weight, core_total)
    core_d_b = _weighted_mean(d_b, core_weight, core_total)
    radians = math.radians(WARM_ANGLE_DEG)
    warmth = math.cos(radians) * core_d_a + math.sin(radians) * core_d_b
    contrast = tone_curve_contrast(l1, l2, core_weight, core_total)

    # -- colour-surface table ------------------------------------------------
    labels = np.asarray(support.surface, dtype=np.int64).reshape(-1)
    flat_weight = normalized_weight.reshape(-1)
    counts = np.bincount(labels, weights=flat_weight, minlength=len(COLOUR_SURFACES))
    rows, columns = first.shape[:2]
    # Pixel centres in [0, 1], so a one-row image is not pinned to the top edge.
    centre_y = np.repeat((np.arange(rows, dtype=np.float64) + 0.5) / rows, columns)
    centre_x = np.tile((np.arange(columns, dtype=np.float64) + 0.5) / columns, rows)
    sums = {
        field: np.bincount(labels, weights=(flat_weight * delta.reshape(-1)),
                           minlength=len(COLOUR_SURFACES))
        for field, delta in (("d_L", d_l), ("d_a", d_a), ("d_b", d_b), ("d_C", d_c))
    }
    for field, grid in (("cy", centre_y), ("cx", centre_x)):
        sums[field] = np.bincount(labels, weights=(flat_weight * grid),
                                  minlength=len(COLOUR_SURFACES))

    return {
        "brightness": {"delta": round(brightness, 6),
                       "direction": _direction("brightness", brightness)},
        "warmth": {"delta": round(warmth, 6), "direction": _direction("warmth", warmth),
                   "components": {"d_a": round(core_d_a, 6), "d_b": round(core_d_b, 6)}},
        "hue_gm": {"delta": round(hue_gm, 6), "direction": _direction("hue_gm", hue_gm)},
        "chroma": {"delta": round(chroma, 6), "direction": _direction("chroma", chroma),
                   # WP18 fix 3's ``chroma_after_mean``: what colour is left, not
                   # how much moved.  A delta alone cannot tell a real conversion
                   # from a big cut that lands somewhere still vivid.
                   "after_mean": round(_weighted_mean(c2, normalized_weight, total), 6)},
        "contrast": {
            "delta": round(contrast, 6), "direction": _direction("contrast", contrast),
            "clip_frac_before": round(
                float((normalized_weight * support.before_highlight).sum()) / total, 6),
            "clip_frac_after": round(
                float((normalized_weight * support.after_highlight).sum()) / total, 6),
        },
        "surfaces": _surface_rows(counts, sums, total, chroma),
    }


def objective_edit_hints(
    before: np.ndarray,
    after: np.ndarray,
    *,
    weight: np.ndarray | None,
) -> dict[str, Any]:
    """Numerical Lab edit directions from the authoritative pre-JPEG pair."""
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise VisibilityError("before and after must be equal-shape HWC RGB arrays")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise VisibilityError("image arrays contain non-finite values")
    if weight is None:
        normalized_weight = np.ones(before.shape[:2], dtype=np.float64)
    else:
        normalized_weight = np.asarray(weight, dtype=np.float64)
    if normalized_weight.shape != before.shape[:2] or not np.isfinite(normalized_weight).all() \
            or np.any(normalized_weight < 0):
        raise VisibilityError("weight map is invalid")
    total = float(normalized_weight.sum())
    if not math.isfinite(total) or total <= 0:
        raise VisibilityError("weight map has no mass")

    before_lab = srgb_to_lab(before)
    return objective_edit_hints_from_lab(
        before_lab,
        srgb_to_lab(after),
        weight=normalized_weight,
        support=HintSupport(
            before_highlight=highlight_mask(before),
            after_highlight=highlight_mask(after),
            surface=colour_surface_labels(before_lab),
        ),
    )


def _resize_rgb(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(
        np.clip(np.asarray(array) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"
    )
    return np.asarray(image.resize((width, height), _RESAMPLING.BILINEAR),
                      dtype=np.float32) / 255.0


def _resize_alpha(array: np.ndarray, width: int, height: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(array, dtype=np.float32), "F")
    return np.asarray(image.resize((width, height), _RESAMPLING.BILINEAR),
                      dtype=np.float32)


def bounded_working_pair(before: np.ndarray, after: np.ndarray, weight: np.ndarray,
                         short_edge: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if short_edge <= 0:
        raise VisibilityError("short_edge must be positive")
    if before.shape != after.shape or before.ndim != 3 or before.shape[2] != 3:
        raise VisibilityError("before and after must be equal-shape HWC RGB arrays")
    if weight.shape != before.shape[:2]:
        raise VisibilityError("weight map must match image dimensions")
    height, width = before.shape[:2]
    if min(height, width) <= short_edge:
        return before, after, weight
    scale = short_edge / min(height, width)
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    return (
        _resize_rgb(before, target_width, target_height),
        _resize_rgb(after, target_width, target_height),
        _resize_alpha(weight, target_width, target_height),
    )


def prepare_working_reference(before: np.ndarray, short_edge: int) -> WorkingImageReference:
    """Prepare the shared bounded before image and Lab array once per group."""
    before = np.asarray(before, dtype=np.float32)
    if before.ndim != 3 or before.shape[2] != 3:
        raise VisibilityError("before must be an HWC RGB array")
    if not np.isfinite(before).all():
        raise VisibilityError("image arrays contain non-finite values")
    height, width = before.shape[:2]
    if short_edge <= 0:
        raise VisibilityError("short_edge must be positive")
    if min(height, width) <= short_edge:
        working = before
    else:
        scale = short_edge / min(height, width)
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        working = _resize_rgb(before, target_width, target_height)
    lab = srgb_to_lab(working)
    return WorkingImageReference(
        source_shape=tuple(before.shape),
        pixels=working,
        lab=lab,
        highlight=highlight_mask(working),
        surface=colour_surface_labels(lab),
    )


def prepare_working_after(
    reference: WorkingImageReference,
    after: np.ndarray,
    weight: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Put one after image and weight map on a prepared reference's grid."""
    after = np.asarray(after, dtype=np.float32)
    if tuple(after.shape) != reference.source_shape:
        raise VisibilityError("before and after must be equal-shape HWC RGB arrays")
    if not np.isfinite(after).all():
        raise VisibilityError("image arrays contain non-finite values")
    unit_weight = weight is None
    if unit_weight:
        normalized_weight = None
    else:
        normalized_weight = np.asarray(weight, dtype=np.float32)
    if normalized_weight is not None and (
        normalized_weight.shape != after.shape[:2]
        or not np.isfinite(normalized_weight).all()
        or np.any(normalized_weight < 0)
    ):
        raise VisibilityError("weight map is invalid")
    target_height, target_width = reference.pixels.shape[:2]
    if after.shape[:2] != (target_height, target_width):
        after = _resize_rgb(after, target_width, target_height)
        if normalized_weight is not None:
            normalized_weight = _resize_alpha(
                normalized_weight, target_width, target_height
            )
    if unit_weight:
        normalized_weight = np.ones((target_height, target_width), dtype=np.float32)
    assert normalized_weight is not None
    return after, normalized_weight


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    linear = np.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )
    matrix = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    delta = 6.0 / 29.0
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    return np.stack((116.0 * f[..., 1] - 16.0,
                     500.0 * (f[..., 0] - f[..., 1]),
                     200.0 * (f[..., 1] - f[..., 2])), axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    first = np.asarray(lab1, dtype=np.float64)
    second = np.asarray(lab2, dtype=np.float64)
    if first.shape != second.shape or first.shape[-1] != 3:
        raise VisibilityError("Lab arrays must have matching final dimension 3")
    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c7 = c_bar ** 7
    g = 0.5 * (1.0 - np.sqrt(c7 / (c7 + 25.0 ** 7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = np.hypot(a1p, b1)
    c2p = np.hypot(a2p, b2)
    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    h1p = np.where(c1p == 0, 0.0, h1p)
    h2p = np.where(c2p == 0, 0.0, h2p)

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    delta_h = h2p - h1p
    delta_h = np.where(c1p * c2p == 0, 0.0, delta_h)
    delta_h = np.where(delta_h > 180.0, delta_h - 360.0, delta_h)
    delta_h = np.where(delta_h < -180.0, delta_h + 360.0, delta_h)
    delta_hp = 2.0 * np.sqrt(c1p * c2p) * np.sin(np.radians(delta_h / 2.0))

    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_abs = np.abs(h1p - h2p)
    h_bar = np.where(c1p * c2p == 0, h_sum, h_sum / 2.0)
    h_bar = np.where((c1p * c2p != 0) & (h_abs > 180.0) & (h_sum < 360.0),
                     (h_sum + 360.0) / 2.0, h_bar)
    h_bar = np.where((c1p * c2p != 0) & (h_abs > 180.0) & (h_sum >= 360.0),
                     (h_sum - 360.0) / 2.0, h_bar)
    t = (
        1.0
        - 0.17 * np.cos(np.radians(h_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * h_bar))
        + 0.32 * np.cos(np.radians(3.0 * h_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * h_bar - 63.0))
    )
    delta_theta = 30.0 * np.exp(-((h_bar - 275.0) / 25.0) ** 2)
    rc = 2.0 * np.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25.0 ** 7))
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / np.sqrt(20.0 + (l_bar - 50.0) ** 2)
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    dl = delta_lp / sl
    dc = delta_cp / sc
    dh = delta_hp / sh
    return np.sqrt(np.maximum(dl * dl + dc * dc + dh * dh + rt * dc * dh, 0.0))


def _ciede2000_torch(first: Any, second: Any) -> Any:
    """Torch float64 equivalent of the NumPy CIEDE2000 oracle."""
    import torch

    l1, a1, b1 = first.unbind(-1)
    l2, a2, b2 = second.unbind(-1)
    c1 = torch.hypot(a1, b1)
    c2 = torch.hypot(a2, b2)
    c_bar = (c1 + c2) / 2.0
    c7 = c_bar ** 7
    g = 0.5 * (1.0 - torch.sqrt(c7 / (c7 + 25.0 ** 7)))
    a1p = (1.0 + g) * a1
    a2p = (1.0 + g) * a2
    c1p = torch.hypot(a1p, b1)
    c2p = torch.hypot(a2p, b2)
    h1p = torch.remainder(torch.rad2deg(torch.atan2(b1, a1p)), 360.0)
    h2p = torch.remainder(torch.rad2deg(torch.atan2(b2, a2p)), 360.0)
    zero = torch.zeros((), dtype=first.dtype, device=first.device)
    h1p = torch.where(c1p == 0, zero, h1p)
    h2p = torch.where(c2p == 0, zero, h2p)

    delta_lp = l2 - l1
    delta_cp = c2p - c1p
    product = c1p * c2p
    delta_h = h2p - h1p
    delta_h = torch.where(product == 0, zero, delta_h)
    delta_h = torch.where(delta_h > 180.0, delta_h - 360.0, delta_h)
    delta_h = torch.where(delta_h < -180.0, delta_h + 360.0, delta_h)
    delta_hp = 2.0 * torch.sqrt(product) * torch.sin(torch.deg2rad(delta_h / 2.0))

    l_bar = (l1 + l2) / 2.0
    c_bar_p = (c1p + c2p) / 2.0
    h_sum = h1p + h2p
    h_abs = torch.abs(h1p - h2p)
    h_bar = torch.where(product == 0, h_sum, h_sum / 2.0)
    h_bar = torch.where(
        (product != 0) & (h_abs > 180.0) & (h_sum < 360.0),
        (h_sum + 360.0) / 2.0,
        h_bar,
    )
    h_bar = torch.where(
        (product != 0) & (h_abs > 180.0) & (h_sum >= 360.0),
        (h_sum - 360.0) / 2.0,
        h_bar,
    )
    t = (
        1.0
        - 0.17 * torch.cos(torch.deg2rad(h_bar - 30.0))
        + 0.24 * torch.cos(torch.deg2rad(2.0 * h_bar))
        + 0.32 * torch.cos(torch.deg2rad(3.0 * h_bar + 6.0))
        - 0.20 * torch.cos(torch.deg2rad(4.0 * h_bar - 63.0))
    )
    delta_theta = 30.0 * torch.exp(-((h_bar - 275.0) / 25.0) ** 2)
    rc = 2.0 * torch.sqrt(c_bar_p ** 7 / (c_bar_p ** 7 + 25.0 ** 7))
    sl = 1.0 + 0.015 * (l_bar - 50.0) ** 2 / torch.sqrt(
        20.0 + (l_bar - 50.0) ** 2
    )
    sc = 1.0 + 0.045 * c_bar_p
    sh = 1.0 + 0.015 * c_bar_p * t
    rt = -torch.sin(torch.deg2rad(2.0 * delta_theta)) * rc
    dl = delta_lp / sl
    dc = delta_cp / sc
    dh = delta_hp / sh
    return torch.sqrt(torch.clamp(dl * dl + dc * dc + dh * dh + rt * dc * dh, min=0.0))


def _srgb_to_lab_torch(rgb: Any) -> Any:
    import torch

    value = torch.clamp(rgb, 0.0, 1.0)
    linear = torch.where(
        value <= 0.04045,
        value / 12.92,
        ((value + 0.055) / 1.055) ** 2.4,
    )
    matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=rgb.dtype,
        device=rgb.device,
    )
    xyz = linear @ matrix.T
    xyz = xyz / torch.tensor(
        [0.95047, 1.0, 1.08883], dtype=rgb.dtype, device=rgb.device
    )
    delta = 6.0 / 29.0
    f = torch.where(
        xyz > delta ** 3,
        torch.pow(xyz, 1.0 / 3.0),
        xyz / (3.0 * delta ** 2) + 4.0 / 29.0,
    )
    return torch.stack(
        (
            116.0 * f[..., 1] - 16.0,
            500.0 * (f[..., 0] - f[..., 1]),
            200.0 * (f[..., 1] - f[..., 2]),
        ),
        dim=-1,
    )


def _objective_edit_hints_torch(
    first: Any,
    second: Any,
    weight: Any,
    total: float,
    *,
    before_highlight: Any,
    after_highlight: Any,
    surface: Any,
) -> dict[str, Any]:
    """The GPU twin of ``objective_edit_hints_from_lab``.

    This is the implementation production actually runs; the NumPy one is the
    oracle it is held to.  Any change to one belongs in both, and the parity
    test in test_canonical_foundation.py is what keeps them honest.
    """
    import torch

    l1, a1, b1 = first.unbind(-1)
    l2, a2, b2 = second.unbind(-1)
    c1 = torch.hypot(a1, b1)
    c2 = torch.hypot(a2, b2)
    d_l, d_a, d_b, d_c = l2 - l1, a2 - a1, b2 - b1, c2 - c1

    def mean(values: Any, mass: Any = weight, scale: float | Any = total) -> Any:
        return (values * mass).sum() / scale

    brightness = mean(d_l)
    chroma = mean(d_c)
    hue_gm = mean(d_a)

    core_weight = weight * (weight >= HI_ALPHA_RATIO * weight.max())
    core_total = core_weight.sum()
    core_d_a = mean(d_a, core_weight, core_total)
    core_d_b = mean(d_b, core_weight, core_total)
    radians = math.radians(WARM_ANGLE_DEG)
    warmth = math.cos(radians) * core_d_a + math.sin(radians) * core_d_b
    chroma_after = mean(c2)

    # -- the tone curve, on the same core -----------------------------------
    # Band membership is integer arithmetic on the before lightness, so it is
    # bit-equal with the NumPy oracle without shipping a mask.  The two
    # degenerate arms are selected on the device rather than branched on the
    # host: reading the band masses back here would cost a full synchronisation,
    # and each arm is exactly the arithmetic the oracle performs in its branch.
    def spread(values: Any) -> Any:
        centre = mean(values, core_weight, core_total)
        return torch.sqrt(torch.clamp(
            mean((values - centre) ** 2, core_weight, core_total), min=0.0))

    spread_before = spread(l1)
    band_index = torch.clamp(
        (l1.reshape(-1) / (100.0 / TONE_BANDS)).to(torch.int64), 0, TONE_BANDS - 1)
    core_flat = core_weight.reshape(-1)
    band_table = torch.zeros((3, TONE_BANDS), dtype=weight.dtype, device=weight.device)
    band_table[0].index_add_(0, band_index, core_flat)
    band_table[1].index_add_(0, band_index, core_flat * l1.reshape(-1))
    band_table[2].index_add_(0, band_index, core_flat * l2.reshape(-1))
    band_mass, band_sum_x, band_sum_y = band_table.unbind(0)
    zero = torch.zeros((), dtype=weight.dtype, device=weight.device)
    one = torch.ones((), dtype=weight.dtype, device=weight.device)
    safe_mass = torch.where(band_mass > 0.0, band_mass, one)
    band_x = band_sum_x / safe_mass
    band_y = band_sum_y / safe_mass
    floor = torch.clamp(TONE_BAND_MIN_MASS * core_total, min=_MIN_SUPPORT_MASS)
    band_weight = torch.where(
        band_mass > floor,
        torch.minimum(band_mass, TONE_BAND_WEIGHT_CAP * core_total),
        zero,
    )
    band_total = band_weight.sum()
    safe_band_total = torch.where(band_total > _MIN_SUPPORT_MASS, band_total, one)
    centre_x = (band_weight * band_x).sum() / safe_band_total
    centre_y = (band_weight * band_y).sum() / safe_band_total
    variance = (band_weight * (band_x - centre_x) ** 2).sum() / safe_band_total
    covariance = (
        band_weight * (band_x - centre_x) * (band_y - centre_y)
    ).sum() / safe_band_total
    fittable = (band_total > _MIN_SUPPORT_MASS) & (variance > _MIN_SUPPORT_MASS)
    slope = covariance / torch.where(fittable, variance, one)
    contrast = torch.where(
        fittable, (slope - 1.0) * spread_before, spread(l2) - spread_before)

    labels = surface.reshape(-1).to(torch.int64)
    flat_weight = weight.reshape(-1)
    bins = len(COLOUR_SURFACES)
    rows, columns = l1.shape
    grid_y = ((torch.arange(rows, dtype=weight.dtype, device=weight.device) + 0.5)
              / rows).repeat_interleave(columns)
    grid_x = ((torch.arange(columns, dtype=weight.dtype, device=weight.device) + 0.5)
              / columns).repeat(rows)
    table = torch.zeros((7, bins), dtype=flat_weight.dtype, device=flat_weight.device)
    table[0].index_add_(0, labels, flat_weight)
    for row, delta in enumerate((d_l, d_a, d_b, d_c), start=1):
        table[row].index_add_(0, labels, flat_weight * delta.reshape(-1))
    for row, grid in ((5, grid_y), (6, grid_x)):
        table[row].index_add_(0, labels, flat_weight * grid)

    # Two device-to-host transfers for the whole record rather than one per
    # figure.  Each read back is a synchronisation, and eight candidates a group
    # queue behind one CUDA lock; the kernels themselves are ~2 ms.
    scalars = torch.stack((brightness, chroma, hue_gm, warmth, core_d_a, core_d_b,
                           contrast, mean(before_highlight.to(weight.dtype)),
                           mean(after_highlight.to(weight.dtype)),
                           chroma_after)).cpu().tolist()
    counts, *sum_rows = table.cpu().numpy()
    (brightness_value, chroma_value, hue_gm_value, warmth_value, core_a, core_b,
     contrast_value, clip_before, clip_after, chroma_after_value) = scalars
    sums = dict(zip(("d_L", "d_a", "d_b", "d_C", "cy", "cx"), sum_rows))
    return {
        "brightness": {"delta": round(brightness_value, 6),
                       "direction": _direction("brightness", brightness_value)},
        "warmth": {"delta": round(warmth_value, 6),
                   "direction": _direction("warmth", warmth_value),
                   "components": {"d_a": round(core_a, 6), "d_b": round(core_b, 6)}},
        "hue_gm": {"delta": round(hue_gm_value, 6),
                   "direction": _direction("hue_gm", hue_gm_value)},
        "chroma": {"delta": round(chroma_value, 6),
                   "direction": _direction("chroma", chroma_value),
                   "after_mean": round(chroma_after_value, 6)},
        "contrast": {
            "delta": round(contrast_value, 6),
            "direction": _direction("contrast", contrast_value),
            "clip_frac_before": round(clip_before, 6),
            "clip_frac_after": round(clip_after, 6),
        },
        "surfaces": _surface_rows(counts, sums, total, chroma_value),
    }


def prepare_torch_lab_reference(
    reference: WorkingImageReference,
    device: str,
) -> TorchLabReference:
    """Keep the shared before Lab array on the renderer's torch device."""
    try:
        import torch
    except ImportError as exc:
        raise VisibilityError("torch visibility backend requires PyTorch") from exc
    try:
        torch_device = torch.device(device)
    except RuntimeError:
        # Injected renderers may expose an abstract device label such as
        # ``cuda:test``. Keep the torch oracle semantics on CPU; the production
        # GPU renderer validates its concrete CUDA device in assert_ready().
        torch_device = torch.device("cpu")
    lab = torch.as_tensor(reference.lab, dtype=torch.float64, device=torch_device)
    # The clipping rails and the colour-surface labels are derived on the CPU
    # and shipped, not recomputed here: they are threshold comparisons, and
    # recomputing them in float64 on the device would let a pixel sitting
    # exactly on a rail land in a different support set than the NumPy oracle
    # puts it in.  Moving the answer keeps the two implementations bit-equal on
    # membership and leaves only reduction order to the parity tolerance.
    return TorchLabReference(
        lab=lab,
        device=str(lab.device),
        lock=threading.Lock(),
        highlight=torch.as_tensor(reference.highlight, device=torch_device),
        surface=torch.as_tensor(reference.surface, device=torch_device),
    )


def visibility_metrics_from_lab_torch(
    reference: TorchLabReference,
    after_lab: np.ndarray,
    *,
    weight: np.ndarray,
    visible_de_min: float,
    visible_fraction_de: float,
    visible_fraction_min: float,
) -> VisibilityMetrics:
    """Evaluate the NumPy oracle formula on CUDA using float64 math."""
    import torch

    after_lab = np.asarray(after_lab, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float32)
    expected_shape = tuple(reference.lab.shape)
    if tuple(after_lab.shape) != expected_shape or after_lab.ndim != 3 \
            or after_lab.shape[2] != 3:
        raise VisibilityError("Lab arrays must be equal-shape HWC arrays")
    if weight.shape != after_lab.shape[:2] or not np.isfinite(weight).all() \
            or np.any(weight < 0):
        raise VisibilityError("weight map is invalid")
    if not np.isfinite(after_lab).all():
        raise VisibilityError("Lab arrays contain non-finite values")
    weight_sum = float(weight.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise VisibilityError("weight map has no mass")

    # PyTorch's CUDA allocator and default stream serialize these short kernels
    # poorly when eight Python threads enter together. One lock produces a much
    # shorter aggregate critical path while the CPU Lab/hint work stays parallel.
    with reference.lock:
        second = torch.tensor(
            after_lab, dtype=torch.float64, device=reference.device
        )
        torch_weight = torch.tensor(
            weight, dtype=torch.float64, device=reference.device
        )
        delta = _ciede2000_torch(reference.lab, second)
        visible_de = float((torch_weight * delta).sum().item() / weight_sum)
        visible_fraction = float(
            (torch_weight * (delta >= visible_fraction_de)).sum().item() / weight_sum
        )
    accepted = visible_de >= visible_de_min and visible_fraction >= visible_fraction_min
    return VisibilityMetrics(visible_de, visible_fraction, accepted)


def visibility_and_hints_torch(
    reference: TorchLabReference,
    after: np.ndarray,
    *,
    weight: np.ndarray,
    visible_de_min: float,
    visible_fraction_de: float,
    visible_fraction_min: float,
) -> tuple[VisibilityMetrics, dict[str, Any]]:
    """Compute the gate and Lab hints together on the render GPU."""
    import torch

    after = np.asarray(after, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    expected_shape = tuple(reference.lab.shape)
    if tuple(after.shape) != expected_shape or after.ndim != 3 or after.shape[2] != 3:
        raise VisibilityError("image arrays must match the torch reference")
    if weight.shape != after.shape[:2] or not np.isfinite(weight).all() \
            or np.any(weight < 0):
        raise VisibilityError("weight map is invalid")
    if not np.isfinite(after).all():
        raise VisibilityError("image arrays contain non-finite values")
    visibility_total = float(weight.sum())
    hints_total = float(np.asarray(weight, dtype=np.float64).sum())
    if not math.isfinite(visibility_total) or visibility_total <= 0 \
            or not math.isfinite(hints_total) or hints_total <= 0:
        raise VisibilityError("weight map has no mass")

    after_highlight = highlight_mask(after)
    with reference.lock:
        second_rgb = torch.tensor(
            after, dtype=torch.float64, device=reference.device
        )
        second_lab = _srgb_to_lab_torch(second_rgb)
        torch_weight = torch.tensor(
            weight, dtype=torch.float64, device=reference.device
        )
        delta = _ciede2000_torch(reference.lab, second_lab)
        visible_de = float(
            (torch_weight * delta).sum().item() / visibility_total
        )
        visible_fraction = float(
            (torch_weight * (delta >= visible_fraction_de)).sum().item()
            / visibility_total
        )
        hints = _objective_edit_hints_torch(
            reference.lab, second_lab, torch_weight, hints_total,
            before_highlight=reference.highlight,
            after_highlight=torch.as_tensor(after_highlight, device=reference.device),
            surface=reference.surface,
        )
    metrics = VisibilityMetrics(
        visible_de,
        visible_fraction,
        visible_de >= visible_de_min and visible_fraction >= visible_fraction_min,
    )
    return metrics, hints


def visibility_metrics_from_lab(
    before_lab: np.ndarray,
    after_lab: np.ndarray,
    *,
    weight: np.ndarray,
    visible_de_min: float,
    visible_fraction_de: float,
    visible_fraction_min: float,
) -> VisibilityMetrics:
    before_lab = np.asarray(before_lab, dtype=np.float64)
    after_lab = np.asarray(after_lab, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float32)
    if before_lab.shape != after_lab.shape or before_lab.ndim != 3 \
            or before_lab.shape[2] != 3:
        raise VisibilityError("Lab arrays must be equal-shape HWC arrays")
    if weight.shape != before_lab.shape[:2] or not np.isfinite(weight).all() \
            or np.any(weight < 0):
        raise VisibilityError("weight map is invalid")
    if not np.isfinite(before_lab).all() or not np.isfinite(after_lab).all():
        raise VisibilityError("Lab arrays contain non-finite values")
    weight_sum = float(weight.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise VisibilityError("weight map has no mass")
    delta = ciede2000(before_lab, after_lab)
    visible_de = float((weight * delta).sum() / weight_sum)
    visible_fraction = float((weight * (delta >= visible_fraction_de)).sum() / weight_sum)
    accepted = visible_de >= visible_de_min and visible_fraction >= visible_fraction_min
    return VisibilityMetrics(visible_de, visible_fraction, accepted)


def visibility_metrics(
    before: np.ndarray,
    after: np.ndarray,
    *,
    weight: np.ndarray | None,
    short_edge: int,
    visible_de_min: float,
    visible_fraction_de: float,
    visible_fraction_min: float,
) -> VisibilityMetrics:
    reference = prepare_working_reference(before, short_edge)
    working_after, working_weight = prepare_working_after(reference, after, weight)
    return visibility_metrics_from_lab(
        reference.lab,
        srgb_to_lab(working_after),
        weight=working_weight,
        visible_de_min=visible_de_min,
        visible_fraction_de=visible_fraction_de,
        visible_fraction_min=visible_fraction_min,
    )


def assert_alpha_zero_endpoint(before: np.ndarray, after: np.ndarray,
                               alpha: np.ndarray) -> None:
    changed = np.not_equal(np.asarray(before), np.asarray(after))
    changed &= (np.asarray(alpha) == 0)[..., None]
    if np.any(changed):
        raise VisibilityError("alpha == 0 endpoint changed before JPEG encoding")
