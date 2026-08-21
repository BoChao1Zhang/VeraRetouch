"""B12 item 1: the source-side L* / C*ab / hue histogram evidence of one image.

Everything here is a pure, seed-free function over an image. The sample is a fixed
even stride over the flat pixel index (`HISTOGRAM_SAMPLE_PIXELS` positions from
`np.linspace`), so the same file always produces the same numbers on any machine and
no RNG state can leak in.

Columns, in the frozen serialization order:

* `l_bins` - share of sampled pixels in each of `L_BIN_COUNT` equal-width L* bins
  over [0, 100]. Sums to 1.
* `clip_low` / `clip_high` - share of sampled pixels with `L* < CLIP_LOW_L` and
  `L* > CLIP_HIGH_L`. These are *sub*-shares of the first / last L* bin, not extra bins.
* `c_bins` - share of sampled pixels in each C*ab bin cut by `CHROMA_BIN_EDGES`
  (four bins: below the first edge, between the edges, above the last edge). Sums to 1.
* `hue_sectors` - share of sampled pixels in each of `HUE_SECTOR_COUNT` equal-width
  hue sectors of the Lab hue angle, counted over the chromatic pixels only
  (`C*ab >= HUE_CHROMA_MIN`, which is exactly `CHROMA_BIN_EDGES[0]`). The sectors
  therefore sum to `1 - c_bins[0]`, never to 1.

`HISTOGRAM_MATCH_GATE` is the pre-registered retrieval bonus that pairs this source
reading with a LUT's histogram response (`segment_fingerprints` v2). See
`histogram_match_bonus` for the exact arithmetic and the sign convention.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab


class SourceHistogramError(ValueError):
    """The image or the histogram payload violates its contract."""


SOURCE_HISTOGRAM_CONTRACT = "source-histogram-lab-8bin-v1"

# Deterministic pixel budget; the same budget the R6.2 measurement and the frozen reach
# probe use, so all three read comparable population sizes.
HISTOGRAM_SAMPLE_PIXELS = 4096

L_BIN_COUNT = 8
L_RANGE: tuple[float, float] = (0.0, 100.0)
CLIP_LOW_L = 2.0
CLIP_HIGH_L = 98.0

# Four C*ab bins: [0, 10), [10, 25), [25, 50), [50, inf).
CHROMA_BIN_EDGES: tuple[float, ...] = (10.0, 25.0, 50.0)
HUE_SECTOR_COUNT = 6
# A pixel below this chroma has no stable hue angle, so it carries no hue sector.
# Frozen equal to `CHROMA_BIN_EDGES[0]`, which is what makes the sectors sum to
# `1 - c_bins[0]` exactly.
HUE_CHROMA_MIN = CHROMA_BIN_EDGES[0]

ROUND_DIGITS = 6

# Fixed serialization widths. Changing any of them changes the prompt bytes, so they
# are registered constants.
L_BIN_FORMAT = "{:.3f}"
CLIP_FORMAT = "{:.4f}"
C_BIN_FORMAT = "{:.3f}"
HUE_FORMAT = "{:.3f}"

HISTOGRAM_COLUMNS: tuple[str, ...] = (
    "l_bins", "clip_low", "clip_high", "c_bins", "hue_sectors",
)

SOURCE_HISTOGRAM_HEADER = (
    "Source tone/colour histogram of the untouched photo, measured on "
    f"{HISTOGRAM_SAMPLE_PIXELS} deterministic pixel samples. One line, fixed columns: "
    f"L{L_BIN_COUNT} is the share of pixels in {L_BIN_COUNT} equal-width CIE L* bins "
    "from black to white; CLIP is the share below L*=2 and the share above L*=98; "
    f"C{len(CHROMA_BIN_EDGES) + 1} is the share in the CIE C*ab chroma bins cut at "
    + "/".join(f"{edge:g}" for edge in CHROMA_BIN_EDGES)
    + f"; H{HUE_SECTOR_COUNT} is the share in {HUE_SECTOR_COUNT} equal-width Lab hue "
    f"sectors starting at hue 0 deg, counted over the chromatic pixels only "
    f"(C*ab >= {HUE_CHROMA_MIN:g}), so H sums to 1 minus the first C bin."
)

# B12 item 3: the pre-registered histogram retrieval bonus.
#
# `clip_low_min` / `clip_high_min` are the source-side activation thresholds; a source
# whose clipped share sits below them contributes no histogram term at all.
# `delta_scale` is the |aggregate| that saturates one term, `shadow_weight` /
# `highlight_weight` are the saturated bonuses added to `LutCatalog._score`.
#
# `shadow_sign` / `highlight_sign` fix which sign of the LUT aggregate is rewarded:
# the shadow term fires when `shadow_sign * d_shadow > 0` and the highlight term when
# `highlight_sign * d_high > 0`. `d_shadow` / `d_high` are output-minus-input bin-share
# differences (see `segment_fingerprints.derive_histogram_response`), so a LUT that
# lifts the shadows moves pixels *out* of the shadow bins (`d_shadow < 0`) and a LUT
# that tames the highlights moves pixels out of the highlight bins (`d_high < 0`).
# The registered convention rewards `d_shadow < 0` for a clipped-shadow source and
# `d_high < 0` for a clipped-highlight source, which is `-1.0` / `-1.0` here. These two
# constants are the whole sign convention of the term and flipping one is a one-line,
# fingerprinted edit.
HISTOGRAM_MATCH_GATE: dict[str, float] = {
    "clip_low_min": 0.02,
    "clip_high_min": 0.02,
    "delta_scale": 0.05,
    "shadow_weight": 0.5,
    "highlight_weight": 0.5,
    "shadow_sign": -1.0,
    "highlight_sign": -1.0,
}


# ------------------------------------------------------------------ sampling
def _as_rgb(value: Any) -> np.ndarray:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            array = np.asarray(
                ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64
            ) / 255.0
    elif isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float64) / 255.0
    else:
        array = np.asarray(value)
        array = (
            array.astype(np.float64) / 255.0 if array.dtype == np.uint8
            else array.astype(np.float64)
        )
    if array.ndim != 3 or array.shape[2] != 3:
        raise SourceHistogramError("image must be an HxWx3 RGB array")
    return np.clip(array, 0.0, 1.0)


def sample_pixels(image: Any, budget: int = HISTOGRAM_SAMPLE_PIXELS) -> np.ndarray:
    """Deterministic, seed-free even stride of at most `budget` RGB pixels."""
    if budget <= 0:
        raise SourceHistogramError("sample budget must be positive")
    rgb = _as_rgb(image)
    flat = rgb.reshape(-1, 3)
    if flat.shape[0] == 0:
        raise SourceHistogramError("image has no pixel")
    if flat.shape[0] <= budget:
        return flat
    positions = np.linspace(0, flat.shape[0] - 1, budget)
    return flat[np.rint(positions).astype(np.int64)]


# ------------------------------------------------------------------ histograms
def l_bin_shares(lightness: np.ndarray) -> list[float]:
    """Share of `lightness` in each of `L_BIN_COUNT` equal-width L* bins."""
    values = np.asarray(lightness, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise SourceHistogramError("cannot bin an empty L* population")
    low, high = L_RANGE
    width = (high - low) / float(L_BIN_COUNT)
    index = np.clip(
        np.floor((values - low) / width).astype(np.int64), 0, L_BIN_COUNT - 1
    )
    counts = np.bincount(index, minlength=L_BIN_COUNT).astype(np.float64)
    return [round(float(value / values.size), ROUND_DIGITS) for value in counts]


def _chroma_shares(chroma: np.ndarray) -> list[float]:
    total = float(chroma.size)
    edges = np.asarray(CHROMA_BIN_EDGES, dtype=np.float64)
    index = np.searchsorted(edges, chroma, side="right")
    counts = np.bincount(index, minlength=len(CHROMA_BIN_EDGES) + 1).astype(np.float64)
    return [round(float(value / total), ROUND_DIGITS) for value in counts]


def _hue_shares(chroma: np.ndarray, a_star: np.ndarray, b_star: np.ndarray) -> list[float]:
    total = float(chroma.size)
    chromatic = chroma >= HUE_CHROMA_MIN
    counts = np.zeros(HUE_SECTOR_COUNT, dtype=np.float64)
    if bool(chromatic.any()):
        angle = np.degrees(np.arctan2(b_star[chromatic], a_star[chromatic])) % 360.0
        width = 360.0 / float(HUE_SECTOR_COUNT)
        index = np.clip(
            np.floor(angle / width).astype(np.int64), 0, HUE_SECTOR_COUNT - 1
        )
        counts = np.bincount(index, minlength=HUE_SECTOR_COUNT).astype(np.float64)
    return [round(float(value / total), ROUND_DIGITS) for value in counts]


def source_histogram(
    image: Any, budget: int = HISTOGRAM_SAMPLE_PIXELS
) -> dict[str, Any]:
    """The frozen source histogram reading of one image."""
    pixels = sample_pixels(image, budget)
    lab = np.asarray(rgb2lab(pixels.reshape(-1, 1, 3)), dtype=np.float64).reshape(-1, 3)
    lightness, a_star, b_star = lab[:, 0], lab[:, 1], lab[:, 2]
    chroma = np.hypot(a_star, b_star)
    total = float(lightness.size)
    return {
        "histogram_contract": SOURCE_HISTOGRAM_CONTRACT,
        "sample_pixels": int(lightness.size),
        "l_bins": l_bin_shares(lightness),
        "clip_low": round(float((lightness < CLIP_LOW_L).sum() / total), ROUND_DIGITS),
        "clip_high": round(float((lightness > CLIP_HIGH_L).sum() / total), ROUND_DIGITS),
        "c_bins": _chroma_shares(chroma),
        "hue_sectors": _hue_shares(chroma, a_star, b_star),
    }


def assert_histogram_columns(row: Mapping[str, Any]) -> None:
    """Runtime assertion: every pre-registered source-histogram column was produced."""
    missing = sorted(set(HISTOGRAM_COLUMNS) - set(row))
    if missing:
        raise SourceHistogramError(f"source histogram columns missing: {missing}")
    if row.get("histogram_contract") != SOURCE_HISTOGRAM_CONTRACT:
        raise SourceHistogramError("source histogram contract mismatch")
    for name, count in (
        ("l_bins", L_BIN_COUNT),
        ("c_bins", len(CHROMA_BIN_EDGES) + 1),
        ("hue_sectors", HUE_SECTOR_COUNT),
    ):
        values = row.get(name)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) \
                or len(values) != count:
            raise SourceHistogramError(f"{name} must carry exactly {count} numbers")
    for name in ("clip_low", "clip_high"):
        value = row.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)):
            raise SourceHistogramError(f"{name} must be a finite number")


# ------------------------------------------------------------------ serialization
def _numbers(values: Sequence[Any], template: str) -> str:
    return " ".join(template.format(float(value)) for value in values)


def source_histogram_text(row: Mapping[str, Any]) -> str:
    """The one frozen line that enters the global prompt after the diagnosis."""
    assert_histogram_columns(row)
    return (
        f"source_histogram {SOURCE_HISTOGRAM_CONTRACT}"
        f" | L{L_BIN_COUNT} {_numbers(row['l_bins'], L_BIN_FORMAT)}"
        f" | CLIP {_numbers([row['clip_low'], row['clip_high']], CLIP_FORMAT)}"
        f" | C{len(CHROMA_BIN_EDGES) + 1} {_numbers(row['c_bins'], C_BIN_FORMAT)}"
        f" | H{HUE_SECTOR_COUNT} {_numbers(row['hue_sectors'], HUE_FORMAT)}"
    )


def source_histogram_block(row: Mapping[str, Any]) -> str:
    """`SOURCE_HISTOGRAM_HEADER` plus the single serialized line."""
    return SOURCE_HISTOGRAM_HEADER + "\n" + source_histogram_text(row)


# ------------------------------------------------------------------ retrieval bonus
def _finite(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def histogram_match_bonus(
    source: Mapping[str, Any] | None, response: Mapping[str, Any] | None
) -> float:
    """Pre-registered histogram term of `LutCatalog._score`.

    `source` is a `source_histogram()` reading, `response` a LUT's mounted v2
    histogram response (`d_shadow` / `d_mid` / `d_high`). Returns 0.0 whenever either
    side is missing, so an unmounted catalog scores exactly as it did before B12.

    Both terms are one-sided and saturating::

        clip_low  >= clip_low_min  and shadow_sign * d_shadow > 0
            -> shadow_weight * min(|d_shadow| / delta_scale, 1)
        clip_high >= clip_high_min and highlight_sign * d_high > 0
            -> highlight_weight * min(|d_high| / delta_scale, 1)

    With the registered signs (`-1.0` / `-1.0`) a clipped-shadow source rewards
    `d_shadow < 0` (the LUT lifts pixels out of the shadow bins) and a clipped-highlight
    source rewards `d_high < 0` (the LUT pulls pixels out of the highlight bins).
    """
    if not source or not response:
        return 0.0
    gate = HISTOGRAM_MATCH_GATE
    scale = float(gate["delta_scale"])
    if scale <= 0.0:
        raise SourceHistogramError("delta_scale must be positive")
    bonus = 0.0
    pairs = (
        ("clip_low", "clip_low_min", "d_shadow", "shadow_sign", "shadow_weight"),
        ("clip_high", "clip_high_min", "d_high", "highlight_sign", "highlight_weight"),
    )
    for clip_key, threshold_key, delta_key, sign_key, weight_key in pairs:
        if _finite(source.get(clip_key)) < float(gate[threshold_key]):
            continue
        delta = _finite(response.get(delta_key))
        if float(gate[sign_key]) * delta <= 0.0:
            continue
        bonus += float(gate[weight_key]) * min(abs(delta) / scale, 1.0)
    return float(bonus)


__all__ = [
    "CHROMA_BIN_EDGES", "CLIP_FORMAT", "CLIP_HIGH_L", "CLIP_LOW_L", "C_BIN_FORMAT",
    "HISTOGRAM_COLUMNS", "HISTOGRAM_MATCH_GATE", "HISTOGRAM_SAMPLE_PIXELS",
    "HUE_CHROMA_MIN", "HUE_FORMAT", "HUE_SECTOR_COUNT", "L_BIN_COUNT", "L_BIN_FORMAT",
    "L_RANGE", "ROUND_DIGITS", "SOURCE_HISTOGRAM_CONTRACT", "SOURCE_HISTOGRAM_HEADER",
    "SourceHistogramError", "assert_histogram_columns", "histogram_match_bonus",
    "l_bin_shares", "sample_pixels", "source_histogram", "source_histogram_block",
    "source_histogram_text",
]
