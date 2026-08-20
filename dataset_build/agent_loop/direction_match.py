"""R6.2 direction vectors and segmented-fingerprint direction matching.

Everything here is a pure function over plain numbers / arrays. B11 wires it into the
local retrieval: `MaskDirection` + `combined_direction_scores` are the prefilter that
replaced the offline reach subset, and `direction_match_score` is the domain of the
`cast_correction_local` intent.

A `DirectionVector` always encodes an *observed* direction in the five pre-registered
axes (`cast_a`, `cast_b`, `lightness`, `saturation`, `contrast`) plus a `mode`:

* `mode="correction"` - the vector is the defect that must be undone, so a LUT whose
  segmented fingerprint points the *opposite* way scores high (anti-alignment).
* `mode="enhancement"` - the vector is the direction that is wanted, so a LUT whose
  fingerprint points the *same* way scores high (alignment).

Two constructors are offered, per R6.2: keyword parsing of a diagnosis
(`direction_from_diagnosis` / `direction_from_text`) and a two-image measurement
(`measure_direction`, e.g. source vs `global_after` to read the residual direction).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2hsv, rgb2lab

from .segment_fingerprints import (
    SEGMENT_FIELDS, SEGMENT_L_BOUNDS, SEGMENT_NAMES,
)


class DirectionMatchError(ValueError):
    """The direction vector, fingerprint or tonal weighting violates its contract."""


AXIS_NAMES: tuple[str, ...] = (
    "cast_a", "cast_b", "lightness", "saturation", "contrast"
)
DIRECTION_MODES: tuple[str, ...] = ("correction", "enhancement")
DIRECTION_ORIGINS: tuple[str, ...] = ("keywords", "measured", "manual")

# Pre-registered per-axis units, so the fingerprint side and the measured side land on a
# comparable scale before the cosine. `cast_*`/`lightness`/`contrast` are Lab L*a*b*
# units; `saturation` is an HSV saturation percentage on both sides (the catalog bands
# report `d_sat_pct`, and `measure_direction` reports 100 * dS of HSV).
AXIS_SCALES: dict[str, float] = {
    "cast_a": 5.0, "cast_b": 5.0, "lightness": 5.0, "saturation": 10.0, "contrast": 5.0,
}
# Relative axis importance inside the cosine metric. Flat for v1.
AXIS_WEIGHTS: dict[str, float] = {name: 1.0 for name in AXIS_NAMES}

# Deterministic pixel budget for the two-image measurement (same budget as the frozen
# reach probe's `SAMPLE_PIXELS`).
MEASURE_SAMPLE_PIXELS = 4096
# A mask pixel below this alpha carries no weight in a measurement.
MASK_SUPPORT_MIN = 0.05

# --- keyword vocabulary -------------------------------------------------------------
# The first two rows are the saturation words that `LutCatalog._score` has always used;
# they live here now so the scorer and the direction vector cannot drift apart.
SATURATION_LIFT_WORDS: tuple[str, ...] = ("flat", "muted", "dull", "低饱和", "灰")
SATURATION_DROP_WORDS: tuple[str, ...] = ("oversaturated", "too saturated", "过饱和")
# `LutCatalog._score`'s forbidden-direction veto table: needle in `forbidden_directions`
# x aliases in the LUT caption.
FORBIDDEN_CAPTION_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("warm", ("warm", "暖")), ("cool", ("cool", "冷")),
    ("dark", ("dark", "压暗")), ("satur", ("satur", "饱和")),
)

# (aliases, axis, sign). A rule fires at most once per parsed text. Signs describe the
# *observed / described* state, never the remedy.
KEYWORD_AXES: tuple[tuple[tuple[str, ...], str, float], ...] = (
    (("warm", "yellow cast", "偏黄", "发黄", "偏暖", "暖调"), "cast_b", 1.0),
    (("cool", "blue cast", "偏蓝", "发蓝", "偏冷", "冷调"), "cast_b", -1.0),
    (("magenta", "pink cast", "偏洋红", "偏红", "发红"), "cast_a", 1.0),
    (("green cast", "偏绿", "发绿"), "cast_a", -1.0),
    (SATURATION_LIFT_WORDS, "saturation", -1.0),
    (SATURATION_DROP_WORDS, "saturation", 1.0),
    (
        ("low contrast", "hazy", "对比不足", "缺乏对比", "层次不足", "灰蒙", "发闷"),
        "contrast", -1.0,
    ),
    (("harsh", "too contrasty", "对比过强", "反差过大"), "contrast", 1.0),
    (("underexposed", "too dark", "欠曝", "曝光不足", "偏暗", "过暗"), "lightness", -1.0),
    (("overexposed", "blown", "过曝", "曝光过度", "偏亮", "过亮"), "lightness", 1.0),
)
DIAGNOSIS_FIELDS: dict[str, str] = {
    "correction": "correction_needs", "enhancement": "enhancement_opportunities",
}


@dataclass(frozen=True, slots=True)
class DirectionVector:
    cast_a: float = 0.0
    cast_b: float = 0.0
    lightness: float = 0.0
    saturation: float = 0.0
    contrast: float = 0.0
    mode: str = "correction"
    origin: str = "manual"

    def __post_init__(self) -> None:
        if self.mode not in DIRECTION_MODES:
            raise DirectionMatchError(f"unknown direction mode: {self.mode!r}")
        if self.origin not in DIRECTION_ORIGINS:
            raise DirectionMatchError(f"unknown direction origin: {self.origin!r}")
        for name in AXIS_NAMES:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise DirectionMatchError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise DirectionMatchError(f"{name} must be finite")

    @property
    def orientation(self) -> float:
        """-1 for correction (reward the opposite fingerprint), +1 for enhancement."""
        return -1.0 if self.mode == "correction" else 1.0

    def as_array(self) -> np.ndarray:
        return np.array(
            [float(getattr(self, name)) for name in AXIS_NAMES], dtype=np.float64
        )

    def is_zero(self) -> bool:
        return not bool(np.any(self.as_array()))

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            name: round(float(getattr(self, name)), 6) for name in AXIS_NAMES
        }
        payload["mode"] = self.mode
        payload["origin"] = self.origin
        return payload


# --- keyword construction -----------------------------------------------------------

def direction_from_text(text: str, mode: str = "correction") -> DirectionVector:
    """Parse one blob of diagnosis text into the observed direction it describes."""
    if mode not in DIRECTION_MODES:
        raise DirectionMatchError(f"unknown direction mode: {mode!r}")
    lowered = str(text or "").lower()
    axes = dict.fromkeys(AXIS_NAMES, 0.0)
    for aliases, axis, sign in KEYWORD_AXES:
        if any(alias in lowered for alias in aliases):
            axes[axis] += sign
    return DirectionVector(mode=mode, origin="keywords", **axes)


def direction_from_diagnosis(
    diagnosis: Mapping[str, Any], mode: str = "correction"
) -> DirectionVector:
    """`correction_needs` for correction mode, `enhancement_opportunities` otherwise."""
    if mode not in DIRECTION_MODES:
        raise DirectionMatchError(f"unknown direction mode: {mode!r}")
    field = DIAGNOSIS_FIELDS[mode]
    items = diagnosis.get(field) or []
    if isinstance(items, (str, bytes)):
        items = [items]
    return direction_from_text(
        " ".join(str(item) for item in items), mode
    )


# --- two-image measurement ----------------------------------------------------------

def _as_rgb(value: Any, field: str) -> np.ndarray:
    if isinstance(value, (str, Path)):
        with Image.open(value) as image:
            array = np.asarray(
                ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32
            ) / 255.0
    elif isinstance(value, Image.Image):
        array = np.asarray(value.convert("RGB"), dtype=np.float32) / 255.0
    else:
        array = np.asarray(value)
        if array.dtype == np.uint8:
            array = array.astype(np.float32) / 255.0
        else:
            array = array.astype(np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise DirectionMatchError(f"{field} must be an HxWx3 RGB image")
    return np.clip(array, 0.0, 1.0)


def _as_mask(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype == np.uint8:
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.shape != shape:
        raise DirectionMatchError(
            f"mask shape {array.shape} does not match the image shape {shape}"
        )
    return np.clip(array, 0.0, 1.0)


def _sample_indices(eligible: np.ndarray) -> np.ndarray:
    """Deterministic, seed-free even stride over the eligible flat pixel indices."""
    count = int(eligible.size)
    if count == 0:
        raise DirectionMatchError("no pixel is eligible for a direction measurement")
    if count <= MEASURE_SAMPLE_PIXELS:
        return eligible
    positions = np.linspace(0, count - 1, MEASURE_SAMPLE_PIXELS)
    return eligible[np.rint(positions).astype(np.int64)]


def _sampled(
    image: np.ndarray, mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    flat_mask = None if mask is None else mask.reshape(-1)
    if flat_mask is None:
        eligible = np.arange(image.shape[0] * image.shape[1], dtype=np.int64)
    else:
        eligible = np.flatnonzero(flat_mask > MASK_SUPPORT_MIN).astype(np.int64)
    index = _sample_indices(eligible)
    weights = (
        np.ones(index.size, dtype=np.float64) if flat_mask is None
        else flat_mask[index].astype(np.float64)
    )
    total = float(weights.sum())
    if total <= 0.0:
        raise DirectionMatchError("mask weights sum to zero")
    return index, weights / total


def _lab_and_saturation(
    image: np.ndarray, index: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pixels = image.reshape(-1, 3)[index].reshape(-1, 1, 3)
    lab = rgb2lab(pixels).reshape(-1, 3)
    saturation = rgb2hsv(pixels).reshape(-1, 3)[:, 1] * 100.0
    return lab, saturation


def _segment_membership(lightness: np.ndarray) -> dict[str, np.ndarray]:
    low, high = SEGMENT_L_BOUNDS
    return {
        "shadows": lightness < low,
        "mids": (lightness >= low) & (lightness <= high),
        "highlights": lightness > high,
    }


def measure_direction(
    img_a: Any, img_b: Any, mask: Any = None, *, mode: str = "correction"
) -> DirectionVector:
    """Measure the a -> b direction on `MEASURE_SAMPLE_PIXELS` deterministic samples.

    `mask` (HxW alpha in [0, 1]) restricts the sample to its support and weights every
    sampled pixel by its alpha. `contrast` is the highlight-minus-shadow difference of
    dL, with the tonal segments taken from `img_a`'s L* and `SEGMENT_L_BOUNDS`.
    """
    if mode not in DIRECTION_MODES:
        raise DirectionMatchError(f"unknown direction mode: {mode!r}")
    before = _as_rgb(img_a, "img_a")
    after = _as_rgb(img_b, "img_b")
    if before.shape != after.shape:
        raise DirectionMatchError(
            f"img_a shape {before.shape} does not match img_b shape {after.shape}"
        )
    alpha = _as_mask(mask, before.shape[:2])
    index, weights = _sampled(before, alpha)
    lab_before, sat_before = _lab_and_saturation(before, index)
    lab_after, sat_after = _lab_and_saturation(after, index)
    delta = lab_after - lab_before
    membership = _segment_membership(lab_before[:, 0])

    def segment_dl(name: str) -> float | None:
        selected = membership[name]
        total = float(weights[selected].sum())
        if total <= 0.0:
            return None
        return float((weights[selected] * delta[selected, 0]).sum() / total)

    shadow_dl = segment_dl("shadows")
    highlight_dl = segment_dl("highlights")
    contrast = (
        0.0 if shadow_dl is None or highlight_dl is None else highlight_dl - shadow_dl
    )
    return DirectionVector(
        cast_a=float((weights * delta[:, 1]).sum()),
        cast_b=float((weights * delta[:, 2]).sum()),
        lightness=float((weights * delta[:, 0]).sum()),
        saturation=float((weights * (sat_after - sat_before)).sum()),
        contrast=contrast, mode=mode, origin="measured",
    )


def measure_tonal_weights(img: Any, mask: Any = None) -> dict[str, float]:
    """Shadows/mids/highlights share of the (mask-weighted) pixels of one image."""
    image = _as_rgb(img, "img")
    alpha = _as_mask(mask, image.shape[:2])
    index, weights = _sampled(image, alpha)
    lab, _saturation = _lab_and_saturation(image, index)
    membership = _segment_membership(lab[:, 0])
    return {
        name: float(weights[membership[name]].sum()) for name in SEGMENT_NAMES
    }


# --- fingerprint table and scoring --------------------------------------------------

_FIELD_INDEX: dict[str, int] = {field: i for i, field in enumerate(SEGMENT_FIELDS)}


@dataclass(frozen=True, slots=True)
class SegmentFingerprintTable:
    """`values[i, segment, field]` for the whole catalog, ready for one einsum."""

    preset_ids: tuple[str, ...]
    values: np.ndarray

    def __post_init__(self) -> None:
        expected = (len(self.preset_ids), len(SEGMENT_NAMES), len(SEGMENT_FIELDS))
        if self.values.shape != expected:
            raise DirectionMatchError(
                f"fingerprint table must have shape {expected}, got {self.values.shape}"
            )

    def __len__(self) -> int:
        return len(self.preset_ids)

    @staticmethod
    def _row(fingerprint: Mapping[str, Mapping[str, float]]) -> list[list[float]]:
        if set(fingerprint) != set(SEGMENT_NAMES):
            raise DirectionMatchError(
                f"segment fingerprint must have exactly {list(SEGMENT_NAMES)}"
            )
        row: list[list[float]] = []
        for segment in SEGMENT_NAMES:
            values = fingerprint[segment]
            missing = sorted(set(SEGMENT_FIELDS) - set(values))
            if missing:
                raise DirectionMatchError(
                    f"segment {segment} is missing fields: {missing}"
                )
            row.append([float(values[field]) for field in SEGMENT_FIELDS])
        return row

    @classmethod
    def from_mapping(
        cls, fingerprints: Mapping[str, Mapping[str, Mapping[str, float]]]
    ) -> "SegmentFingerprintTable":
        preset_ids = tuple(sorted(fingerprints))
        if not preset_ids:
            raise DirectionMatchError("cannot build an empty fingerprint table")
        values = np.array(
            [cls._row(fingerprints[preset_id]) for preset_id in preset_ids],
            dtype=np.float64,
        )
        return cls(preset_ids, values)

    @classmethod
    def from_records(cls, records: Iterable[Any]) -> "SegmentFingerprintTable":
        """Build from `LutRecord`s that carry a mounted `segment_fingerprint`."""
        mapping: dict[str, Mapping[str, Mapping[str, float]]] = {}
        for record in records:
            fingerprint = getattr(record, "segment_fingerprint", None)
            if fingerprint is None:
                raise DirectionMatchError(
                    f"{getattr(record, 'preset_id', '<unknown>')} has no mounted "
                    "segment fingerprint"
                )
            mapping[str(record.preset_id)] = fingerprint
        return cls.from_mapping(mapping)


def normalize_tonal_weights(tonal_weights: Mapping[str, float]) -> np.ndarray:
    unknown = sorted(set(tonal_weights) - set(SEGMENT_NAMES))
    if unknown:
        raise DirectionMatchError(f"unknown tonal weight segments: {unknown}")
    weights = np.array(
        [float(tonal_weights.get(name, 0.0)) for name in SEGMENT_NAMES],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise DirectionMatchError("tonal weights must be finite and non-negative")
    total = float(weights.sum())
    if total <= 0.0:
        raise DirectionMatchError("tonal weights must sum to a positive number")
    return weights / total


def _axis_metric() -> np.ndarray:
    return np.array(
        [AXIS_WEIGHTS[name] / AXIS_SCALES[name] for name in AXIS_NAMES],
        dtype=np.float64,
    )


def fingerprint_axes(
    table: SegmentFingerprintTable, tonal_weights: Mapping[str, float]
) -> np.ndarray:
    """Collapse (N, 3, 5) fingerprints onto the (N, 5) direction axes of R6.2."""
    weights = normalize_tonal_weights(tonal_weights)
    values = table.values
    shadows = SEGMENT_NAMES.index("shadows")
    highlights = SEGMENT_NAMES.index("highlights")
    axes = np.empty((values.shape[0], len(AXIS_NAMES)), dtype=np.float64)
    axes[:, 0] = values[:, :, _FIELD_INDEX["cast_a"]] @ weights
    axes[:, 1] = values[:, :, _FIELD_INDEX["cast_b"]] @ weights
    axes[:, 2] = values[:, :, _FIELD_INDEX["dL"]] @ weights
    axes[:, 3] = values[:, :, _FIELD_INDEX["dC"]] @ weights
    # The contrast axis is a difference *across* segments, so it cannot be a weighted
    # sum over them. It is scaled by the geometric-mean relevance of the two ends: 1.0
    # when the mask spans shadows and highlights evenly, 0.0 when it sits in one of them.
    axes[:, 4] = (
        values[:, highlights, _FIELD_INDEX["dL"]] - values[:, shadows, _FIELD_INDEX["dL"]]
    ) * (2.0 * math.sqrt(weights[shadows] * weights[highlights]))
    return axes


def direction_match_scores(
    direction: DirectionVector, table: SegmentFingerprintTable,
    tonal_weights: Mapping[str, float],
) -> np.ndarray:
    """Vectorized match of one direction against a whole fingerprint table.

    Returns a cosine in [-1, 1] per preset, oriented by `direction.mode`: correction
    rewards a fingerprint that points against the observed direction, enhancement
    rewards one that points with it. A zero direction or a zero fingerprint scores 0.
    """
    weights = normalize_tonal_weights(tonal_weights)
    metric = _axis_metric()
    query = direction.as_array() * metric
    relevance = 2.0 * math.sqrt(
        weights[SEGMENT_NAMES.index("shadows")] * weights[SEGMENT_NAMES.index("highlights")]
    )
    query[AXIS_NAMES.index("contrast")] *= relevance
    candidates = fingerprint_axes(table, tonal_weights) * metric
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 0.0:
        return np.zeros(len(table), dtype=np.float64)
    candidate_norm = np.linalg.norm(candidates, axis=1)
    scores = candidates @ query
    safe = candidate_norm > 0.0
    result = np.zeros(len(table), dtype=np.float64)
    result[safe] = scores[safe] / (candidate_norm[safe] * query_norm)
    return result * direction.orientation


def direction_match_score(
    direction: DirectionVector,
    segment_fingerprint: Mapping[str, Mapping[str, float]],
    tonal_weights: Mapping[str, float],
) -> float:
    """Single-fingerprint match score; same metric as `direction_match_scores`."""
    table = SegmentFingerprintTable.from_mapping({"_": segment_fingerprint})
    return float(direction_match_scores(direction, table, tonal_weights)[0])


def rank_by_direction(
    direction: DirectionVector, table: SegmentFingerprintTable,
    tonal_weights: Mapping[str, float], limit: int,
) -> list[tuple[str, float]]:
    """Top-`limit` (preset_id, score), ties broken by preset_id for determinism."""
    if limit <= 0:
        raise DirectionMatchError("limit must be positive")
    scores = direction_match_scores(direction, table, tonal_weights)
    order = sorted(
        range(len(table)), key=lambda i: (-scores[i], table.preset_ids[i])
    )[:limit]
    return [(table.preset_ids[i], float(scores[i])) for i in order]


@dataclass(frozen=True, slots=True)
class MaskDirection:
    """B11 item 2 (R7.1): everything the online local retrieval reads off one mask.

    `correction` is the measured `source -> global_after` residual inside the mask (so a
    LUT pointing against it scores high), `enhancement` is the keyword direction parsed
    from the frozen diagnosis' `enhancement_opportunities` (so a LUT pointing with it
    scores high), and `tonal_weights` is the mask's shadows/mids/highlights share, which
    is how the segmented fingerprint is collapsed onto the five axes.
    """

    correction: DirectionVector
    enhancement: DirectionVector
    tonal_weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if self.correction.mode != "correction":
            raise DirectionMatchError("correction must carry mode='correction'")
        if self.enhancement.mode != "enhancement":
            raise DirectionMatchError("enhancement must carry mode='enhancement'")
        normalize_tonal_weights(self.tonal_weights)

    def vectors(self) -> tuple[DirectionVector, ...]:
        """The non-zero directions, in a fixed order."""
        return tuple(
            vector for vector in (self.correction, self.enhancement)
            if not vector.is_zero()
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "correction": self.correction.as_dict(),
            "enhancement": self.enhancement.as_dict(),
            "tonal_weights": {
                name: round(float(self.tonal_weights.get(name, 0.0)), 6)
                for name in SEGMENT_NAMES
            },
        }


def combined_direction_scores(
    direction: MaskDirection, table: SegmentFingerprintTable
) -> np.ndarray:
    """Mean of the available direction scores; the prefilter key of R7.1.

    Both modes are already oriented so that "good for this mask" is large and positive,
    so the plain mean over the non-zero directions is the combination. When neither
    direction carries any signal every preset scores 0 and the prefilter degenerates to
    the deterministic `preset_id` order.
    """
    vectors = direction.vectors()
    if not vectors:
        return np.zeros(len(table), dtype=np.float64)
    total = np.zeros(len(table), dtype=np.float64)
    for vector in vectors:
        total += direction_match_scores(vector, table, direction.tonal_weights)
    return total / float(len(vectors))


def uniform_tonal_weights() -> dict[str, float]:
    share = 1.0 / len(SEGMENT_NAMES)
    return dict.fromkeys(SEGMENT_NAMES, share)


def validate_direction_axes(names: Sequence[str]) -> None:
    if tuple(names) != AXIS_NAMES:
        raise DirectionMatchError(f"direction axes must be {list(AXIS_NAMES)}")


__all__ = [
    "AXIS_NAMES", "AXIS_SCALES", "AXIS_WEIGHTS", "DIAGNOSIS_FIELDS",
    "DIRECTION_MODES", "DIRECTION_ORIGINS", "DirectionMatchError", "DirectionVector",
    "FORBIDDEN_CAPTION_ALIASES", "KEYWORD_AXES", "MASK_SUPPORT_MIN", "MaskDirection",
    "MEASURE_SAMPLE_PIXELS", "SATURATION_DROP_WORDS", "SATURATION_LIFT_WORDS",
    "SegmentFingerprintTable", "combined_direction_scores",
    "direction_from_diagnosis", "direction_from_text",
    "direction_match_score", "direction_match_scores", "fingerprint_axes",
    "measure_direction", "measure_tonal_weights", "normalize_tonal_weights",
    "rank_by_direction", "uniform_tonal_weights", "validate_direction_axes",
]
