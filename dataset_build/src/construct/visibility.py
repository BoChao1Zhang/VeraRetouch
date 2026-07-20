"""Pre-JPEG alpha-weighted CIEDE2000 visibility gate."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


def _weighted_mean(values: np.ndarray, weight: np.ndarray, total: float) -> float:
    return float((np.asarray(values, dtype=np.float64) * weight).sum() / total)


def objective_edit_hints(
    before: np.ndarray,
    after: np.ndarray,
    *,
    weight: np.ndarray | None,
) -> dict[str, dict[str, float | str]]:
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

    first = srgb_to_lab(before)
    second = srgb_to_lab(after)
    l1, a1, b1 = np.moveaxis(first, -1, 0)
    l2, a2, b2 = np.moveaxis(second, -1, 0)
    c1 = np.hypot(a1, b1)
    c2 = np.hypot(a2, b2)
    mean_l1 = _weighted_mean(l1, normalized_weight, total)
    mean_l2 = _weighted_mean(l2, normalized_weight, total)
    contrast1 = math.sqrt(_weighted_mean((l1 - mean_l1) ** 2, normalized_weight, total))
    contrast2 = math.sqrt(_weighted_mean((l2 - mean_l2) ** 2, normalized_weight, total))

    values = {
        "brightness": _weighted_mean(l2 - l1, normalized_weight, total),
        "warmth": _weighted_mean(b2 - b1, normalized_weight, total),
        "chroma": _weighted_mean(c2 - c1, normalized_weight, total),
        "contrast": contrast2 - contrast1,
    }
    labels = {
        "brightness": ("brighter", "darker"),
        "warmth": ("warmer", "cooler"),
        "chroma": ("richer", "more muted"),
        "contrast": ("higher", "lower"),
    }
    result: dict[str, dict[str, float | str]] = {}
    for name, value in values.items():
        positive, negative = labels[name]
        direction = "unchanged" if abs(value) < 1e-6 else positive if value > 0 else negative
        result[name] = {"delta": round(float(value), 6), "direction": direction}
    return result


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
    before = np.asarray(before, dtype=np.float32)
    after = np.asarray(after, dtype=np.float32)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise VisibilityError("image arrays contain non-finite values")
    if weight is None:
        weight = np.ones(before.shape[:2], dtype=np.float32)
    else:
        weight = np.asarray(weight, dtype=np.float32)
    if not np.isfinite(weight).all() or np.any(weight < 0):
        raise VisibilityError("weight map is invalid")
    before, after, weight = bounded_working_pair(before, after, weight, short_edge)
    weight_sum = float(weight.sum())
    if not math.isfinite(weight_sum) or weight_sum <= 0:
        raise VisibilityError("weight map has no mass")
    delta = ciede2000(srgb_to_lab(before), srgb_to_lab(after))
    visible_de = float((weight * delta).sum() / weight_sum)
    visible_fraction = float((weight * (delta >= visible_fraction_de)).sum() / weight_sum)
    accepted = visible_de >= visible_de_min and visible_fraction >= visible_fraction_min
    return VisibilityMetrics(visible_de, visible_fraction, accepted)


def assert_alpha_zero_endpoint(before: np.ndarray, after: np.ndarray,
                               alpha: np.ndarray) -> None:
    zero = np.asarray(alpha) == 0
    if np.any(zero) and not np.array_equal(np.asarray(before)[zero], np.asarray(after)[zero]):
        raise VisibilityError("alpha == 0 endpoint changed before JPEG encoding")
