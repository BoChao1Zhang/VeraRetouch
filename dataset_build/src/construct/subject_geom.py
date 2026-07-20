"""Subject-side radial, band, and linear geometry for canonical masks."""
from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np


MIN_LINEAR_ROOM = 0.20
MARGIN_MAX = 1.60
MARGIN_MIN = 1.02
AREA_SMALL = 0.04
AREA_LARGE = 0.45
MIN_ELONG = 1.35
RADIAL_FEATHERS = (55.0, 70.0, 85.0)
LINEAR_RAMP = (0.22, 0.40)
ANGLE_JITTER = 18.0
LINE_ANGLE_JITTER = 25.0
BAND_MAX_WIDTH = 0.85
BAND_AXIS_LEN = 1.6
RADIAL_MIN_B = 0.12
BAND_MIN_W = 0.12


def adaptive_margin(area: float) -> float:
    value = min(max(float(area), 1e-4), 1.0)
    position = (
        (math.log(value) - math.log(AREA_SMALL))
        / (math.log(AREA_LARGE) - math.log(AREA_SMALL))
    )
    position = min(max(position, 0.0), 1.0)
    return MARGIN_MAX + (MARGIN_MIN - MARGIN_MAX) * position


def _size_factor(area: float) -> float:
    return (adaptive_margin(area) - MARGIN_MIN) / (MARGIN_MAX - MARGIN_MIN)


def _mask_pca(mask01: np.ndarray):
    mask = np.asarray(mask01, dtype=np.float32)
    if mask.ndim != 2 or not mask.size:
        return None
    height, width = mask.shape
    if min(height, width) > 512:
        import cv2

        scale = 512.0 / min(height, width)
        mask = cv2.resize(
            mask,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_NEAREST,
        )
        height, width = mask.shape
    ys, xs = np.nonzero(mask > 0.5)
    if xs.size < 32:
        return None
    points = np.stack([xs / width, ys / height], axis=1).astype(np.float64)
    center = points.mean(axis=0)
    covariance = np.cov((points - center).T)
    _, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, 1]
    minor = eigenvectors[:, 0]
    major_extent = float(np.quantile(np.abs((points - center) @ major), 0.98))
    minor_extent = float(np.quantile(np.abs((points - center) @ minor), 0.98))
    return center, major, major_extent, minor_extent, float(xs.size / (height * width))


def _ellipse_geom(center, major: float, minor: float, angle: float, feather: float) -> dict:
    return {
        "Top": round(center[1] - minor, 4),
        "Bottom": round(center[1] + minor, 4),
        "Left": round(center[0] - major, 4),
        "Right": round(center[0] + major, 4),
        "Angle": round(angle, 2),
        "Feather": feather,
        "Roundness": 0.0,
        "Midpoint": 50.0,
        "Flipped": "true",
    }


def radial_geom(
    mask01: np.ndarray, rng: random.Random, apply_inside: bool = True
) -> Optional[dict]:
    if not apply_inside:
        raise ValueError("canonical radial geometry must affect the subject")
    pca = _mask_pca(mask01)
    if pca is None:
        return None
    center, major, extent_a, extent_b, area = pca
    natural_elongation = extent_a / max(extent_b, 1e-6)
    margin = adaptive_margin(area)
    extent_a *= margin
    extent_b = max(extent_b * margin, RADIAL_MIN_B)
    if natural_elongation < MIN_ELONG or extent_a < extent_b * MIN_ELONG:
        extent_a = extent_b * MIN_ELONG
    if natural_elongation < 1.15:
        angle = rng.uniform(-90.0, 90.0)
    else:
        angle = math.degrees(math.atan2(major[1], major[0])) + rng.uniform(
            -ANGLE_JITTER, ANGLE_JITTER
        )
    return {
        "mask_type": "circulargradient",
        "what": "Mask/CircularGradient",
        "geom": _ellipse_geom(
            center, extent_a, extent_b, angle, float(rng.choice(RADIAL_FEATHERS))
        ),
        "_mode": "radial",
        "_apply": "inside",
    }


def band_geom(
    mask01: np.ndarray, rng: random.Random, apply_inside: bool = True
) -> Optional[dict]:
    if not apply_inside:
        raise ValueError("canonical band geometry must affect the subject")
    pca = _mask_pca(mask01)
    if pca is None:
        return None
    center, major, extent_a, extent_b, area = pca
    if rng.random() < 0.6:
        angle = math.degrees(math.atan2(major[1], major[0])) + rng.uniform(
            -ANGLE_JITTER, ANGLE_JITTER
        )
    else:
        angle = rng.choice((0.0, 90.0)) + rng.uniform(-ANGLE_JITTER, ANGLE_JITTER)
    theta = math.radians(angle)
    normal = (-math.sin(theta), math.cos(theta))
    major_normal = major[0] * normal[0] + major[1] * normal[1]
    subject_half_width = math.sqrt(
        (extent_a * major_normal) ** 2
        + (extent_b ** 2) * max(1.0 - major_normal ** 2, 0.0)
    )
    width = max(subject_half_width * adaptive_margin(area) * 1.05, BAND_MIN_W)
    if 2 * width > BAND_MAX_WIDTH:
        return None
    return {
        "mask_type": "circulargradient",
        "what": "Mask/CircularGradient",
        "geom": _ellipse_geom(
            center, BAND_AXIS_LEN, width, angle, float(rng.choice(RADIAL_FEATHERS))
        ),
        "_mode": "band",
        "_apply": "inside",
    }


def _gradient_geom(
    point, base_axis_degrees: float, half_width: float, toward_high: bool,
    rng: random.Random,
) -> dict:
    theta = math.radians(
        base_axis_degrees + rng.uniform(-LINE_ANGLE_JITTER, LINE_ANGLE_JITTER)
    )
    direction = (math.cos(theta), math.sin(theta))
    sign = 1.0 if toward_high else -1.0
    return {
        "ZeroX": round(point[0] - sign * half_width * direction[0], 4),
        "ZeroY": round(point[1] - sign * half_width * direction[1], 4),
        "FullX": round(point[0] + sign * half_width * direction[0], 4),
        "FullY": round(point[1] + sign * half_width * direction[1], 4),
        "Flipped": "false",
    }


def linear_geom(
    bbox,
    rng: random.Random,
    apply_subject_side: bool = True,
    area: Optional[float] = None,
    side: Optional[str] = None,
) -> Optional[dict]:
    if not apply_subject_side:
        raise ValueError("canonical linear geometry must affect the subject side")
    x0, y0, x1, y1 = bbox
    rooms = {"left": x0, "right": 1 - x1, "top": y0, "bottom": 1 - y1}
    if side is None:
        side, room = max(rooms.items(), key=lambda item: item[1])
    else:
        if side not in rooms:
            raise ValueError(f"invalid linear side: {side}")
        room = rooms[side]
    if room < MIN_LINEAR_ROOM:
        return None
    ramp_width = min(rng.uniform(*LINEAR_RAMP), room * 0.9)
    edge = {"left": x0, "right": x1, "top": y0, "bottom": y1}[side]
    sign = -1 if side in ("left", "top") else 1
    gap = 0.15 + 0.35 * _size_factor(area) if area is not None else 0.35
    center = edge + sign * gap * room
    horizontal = side in ("left", "right")
    point = (center, 0.5) if horizontal else (0.5, center)
    subject_on_low_side = sign > 0
    toward_high = not subject_on_low_side
    return {
        "mask_type": "gradient",
        "what": "Mask/Gradient",
        "geom": _gradient_geom(
            point, 0.0 if horizontal else 90.0, ramp_width / 2, toward_high, rng
        ),
        "_mode": "linear",
        "_side": side,
        "_apply": "subject_side",
    }


__all__ = ["band_geom", "linear_geom", "radial_geom"]
