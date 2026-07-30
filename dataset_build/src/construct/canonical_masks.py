"""Fixed subject-aware eight-slot mask protocol for canonical local groups."""
from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageFilter

from dataset_build.tools.archive_reader import open_image

from .sources import SourceRecord
from .state import stable_id
from .subject_geom import band_geom, linear_geom, radial_geom


GEOMETRY_ATTEMPTS = 12
MODE_COUNTS = {"radial": 2, "semantic": 2, "band": 2, "linear": 2}
_RESAMPLING = getattr(Image, "Resampling", Image)


class MaskPlanError(RuntimeError):
    """A ready cache entry cannot satisfy the fixed geometry contract."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class MaskAsset:
    mask_id: str
    mode: str
    effective_alpha: np.ndarray
    raw_alpha_mean: float
    amount: float
    effective_alpha_mean: float
    geometry: dict[str, Any] | None
    region: str


@dataclass(frozen=True, slots=True)
class MaskSlot:
    slot_id: str
    mode: str
    mode_index: int
    mask: MaskAsset
    pairing_index: int = -1


@dataclass(frozen=True, slots=True)
class MaskPlan:
    slots: tuple[MaskSlot, ...]
    physical_masks: tuple[MaskAsset, ...]


def _seed_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_subject_alpha(source: SourceRecord, width: int, height: int) -> np.ndarray:
    try:
        image = open_image(source.subject_path)
        mask = image.convert("L").resize((width, height), _RESAMPLING.NEAREST)
        hard = np.asarray(mask, dtype=np.float32) / 255.0
    except Exception as exc:  # noqa: BLE001
        raise MaskPlanError("subject_cache_corrupt", "subject.png is unreadable") from exc
    hard = (hard > 0.5).astype(np.float32)
    area = float(hard.mean()) if hard.size else 0.0
    if not np.isfinite(hard).all() or area < 0.005 or area > 0.85:
        raise MaskPlanError("subject_mask_integrity", f"subject mask area is {area:.6f}")
    return hard


def _semantic_alpha(hard: np.ndarray, rng: random.Random) -> np.ndarray:
    area = float(hard.mean())
    radius = min(hard.shape) * (0.008 + 0.017 * math.sqrt(max(area, 1e-4)))
    radius *= rng.uniform(0.7, 1.4)
    image = Image.fromarray((hard * 255.0).astype(np.uint8), "L")
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=max(radius, 0.5))),
                         dtype=np.float32) / 255.0
    # The semantic edit remains confined to the selected subject instance.
    return np.clip(blurred * hard, 0.0, 1.0).astype(np.float32)


def raster_geometry(mask_type: str, geometry: dict[str, Any],
                    height: int, width: int) -> np.ndarray:
    def value(name: str, default: float = 0.0) -> float:
        try:
            return float(str(geometry.get(name, default)).lstrip("+"))
        except (TypeError, ValueError):
            return default

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    x = xx / float(width)
    y = yy / float(height)
    if mask_type == "circulargradient":
        cx = (value("Left") + value("Right")) / 2.0
        cy = (value("Top") + value("Bottom")) / 2.0
        rx = max(abs(value("Right") - value("Left")) / 2.0, 1e-3)
        ry = max(abs(value("Bottom") - value("Top")) / 2.0, 1e-3)
        angle = math.radians(value("Angle"))
        xr = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
        yr = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        distance = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        feather = max(value("Feather", 50.0) / 100.0, 0.05)
        alpha = np.clip((distance - 1.0) / feather + 0.5, 0.0, 1.0)
    elif mask_type == "gradient":
        zx, zy = value("ZeroX"), value("ZeroY")
        fx, fy = value("FullX", 1.0), value("FullY")
        dx, dy = fx - zx, fy - zy
        length_sq = dx * dx + dy * dy + 1e-6
        alpha = np.clip(((x - zx) * dx + (y - zy) * dy) / length_sq, 0.0, 1.0)
    else:
        raise MaskPlanError("unsupported_mask_type", mask_type)
    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    if str(geometry.get("Flipped", "false")).lower().lstrip("+") == "true":
        alpha = 1.0 - alpha
    return alpha.astype(np.float32)


def _coarse_region(alpha: np.ndarray) -> str:
    weights = np.asarray(alpha, dtype=np.float64)
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise MaskPlanError("empty_alpha", "mask has no effective alpha mass")
    yy, xx = np.mgrid[0:weights.shape[0], 0:weights.shape[1]]
    cx = float((weights * xx).sum() / total) / weights.shape[1]
    cy = float((weights * yy).sum() / total) / weights.shape[0]
    horizontal = "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "center"
    vertical = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    if horizontal == "center" and vertical == "middle":
        return "center"
    if horizontal == "center":
        return vertical
    if vertical == "middle":
        return horizontal
    return f"{vertical} {horizontal}"


def linear_strength(raw_alpha: np.ndarray, target: float = 0.5) -> tuple[float, np.ndarray]:
    raw = np.asarray(raw_alpha, dtype=np.float32)
    raw_mean = float(raw.mean())
    if not math.isfinite(raw_mean) or raw_mean <= 0:
        raise MaskPlanError("empty_linear_alpha", "linear alpha has no mass")
    amount = 1.0 if raw_mean <= target else target / raw_mean
    if amount < 0.5 - 1e-7:
        raise MaskPlanError("linear_amount_invariant", f"linear amount is {amount}")
    return amount, np.clip(raw * amount, 0.0, 1.0).astype(np.float32)


def _asset(
    *,
    build_id: str,
    source_id: str,
    physical_key: str,
    mode: str,
    raw: np.ndarray,
    geometry: dict[str, Any] | None,
    linear_target: float,
) -> MaskAsset:
    raw = np.clip(np.asarray(raw, dtype=np.float32), 0.0, 1.0)
    raw_mean = float(raw.mean())
    if mode == "linear":
        amount, effective = linear_strength(raw, linear_target)
    else:
        amount, effective = 1.0, raw
    effective_mean = float(effective.mean())
    if effective_mean <= 0:
        raise MaskPlanError("empty_alpha", f"{mode} mask has no alpha mass")
    return MaskAsset(
        mask_id=stable_id("mask", build_id, source_id, physical_key),
        mode=mode,
        effective_alpha=effective,
        raw_alpha_mean=raw_mean,
        amount=amount,
        effective_alpha_mean=effective_mean,
        geometry=geometry,
        region=_coarse_region(effective),
    )


def _sample_geometry(
    mode: str,
    mask: np.ndarray,
    bbox: tuple[float, float, float, float],
    area: float,
    rng: random.Random,
) -> dict[str, Any]:
    for _ in range(GEOMETRY_ATTEMPTS):
        if mode == "radial":
            spec = radial_geom(mask, rng, apply_inside=True)
        elif mode == "band":
            spec = band_geom(mask, rng, apply_inside=True)
        elif mode == "linear":
            rooms = {
                "left": bbox[0], "right": 1.0 - bbox[2],
                "top": bbox[1], "bottom": 1.0 - bbox[3],
            }
            sides = [side for side, room in rooms.items() if room >= 0.20]
            if not sides:
                spec = None
            else:
                spec = linear_geom(
                    bbox, rng, apply_subject_side=True, area=area, side=rng.choice(sides)
                )
        else:
            raise MaskPlanError("invalid_mode", mode)
        if spec is not None:
            if spec.get("_apply") in {"outside", "env_side", "one_side"}:
                raise MaskPlanError("background_inversion", f"invalid {mode} apply side")
            return spec
    raise MaskPlanError("geometry_unsatisfied", f"cannot sample required {mode} geometry")


def build_mask_plan(
    source: SourceRecord,
    *,
    build_id: str,
    seed: int,
    width: int,
    height: int,
    linear_target: float = 0.5,
) -> MaskPlan:
    hard = load_subject_alpha(source, width, height)
    ys, xs = np.nonzero(hard > 0.5)
    if not len(xs):
        raise MaskPlanError("empty_subject", "ready subject mask is empty")
    bbox = (
        float(xs.min()) / width,
        float(ys.min()) / height,
        float(xs.max() + 1) / width,
        float(ys.max() + 1) / height,
    )
    area = float(hard.mean())
    slots: list[MaskSlot] = []
    physical: list[MaskAsset] = []

    for mode in ("radial",):
        for index in range(2):
            rng = random.Random(_seed_int(build_id, seed, source.source_id, mode, index))
            spec = _sample_geometry(mode, hard, bbox, area, rng)
            raw = raster_geometry(spec["mask_type"], spec["geom"], height, width)
            asset = _asset(
                build_id=build_id, source_id=source.source_id,
                physical_key=f"{mode}-{index}", mode=mode, raw=raw,
                geometry=dict(spec["geom"]), linear_target=linear_target,
            )
            physical.append(asset)
            slots.append(MaskSlot(f"{mode}-{index}", mode, index, asset))

    semantic_rng = random.Random(_seed_int(build_id, seed, source.source_id, "semantic"))
    semantic = _asset(
        build_id=build_id, source_id=source.source_id, physical_key="semantic-shared",
        mode="semantic", raw=_semantic_alpha(hard, semantic_rng), geometry=None,
        linear_target=linear_target,
    )
    physical.append(semantic)
    slots.extend((
        MaskSlot("semantic-0", "semantic", 0, semantic),
        MaskSlot("semantic-1", "semantic", 1, semantic),
    ))

    for mode in ("band", "linear"):
        for index in range(2):
            rng = random.Random(_seed_int(build_id, seed, source.source_id, mode, index))
            spec = _sample_geometry(mode, hard, bbox, area, rng)
            raw = raster_geometry(spec["mask_type"], spec["geom"], height, width)
            asset = _asset(
                build_id=build_id, source_id=source.source_id,
                physical_key=f"{mode}-{index}", mode=mode, raw=raw,
                geometry=dict(spec["geom"]), linear_target=linear_target,
            )
            physical.append(asset)
            slots.append(MaskSlot(f"{mode}-{index}", mode, index, asset))

    if CounterLike(slot.mode for slot in slots) != MODE_COUNTS:
        raise MaskPlanError("slot_count_invariant", "mask plan is not 2+2+2+2")
    if len({asset.mask_id for asset in physical}) != 7:
        raise MaskPlanError("physical_mask_invariant", "mask plan does not have seven assets")
    return MaskPlan(tuple(slots), tuple(physical))


def pair_mask_slots(plan: MaskPlan, *, build_id: str, seed: int,
                    source_id: str) -> tuple[MaskSlot, ...]:
    slots = list(plan.slots)
    random.Random(_seed_int(build_id, seed, source_id, "mask-pairing")).shuffle(slots)
    return tuple(
        MaskSlot(slot.slot_id, slot.mode, slot.mode_index, slot.mask, pairing_index=index)
        for index, slot in enumerate(slots)
    )


def CounterLike(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts
