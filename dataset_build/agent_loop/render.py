"""First-class global/local strength calibration and render caching."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab

from .api_cache import canonical_json
from .artifacts import ArtifactStore
from .candidates import BACKGROUND_ROLE_GATE, LutCatalog, LutRecord
from .direction_match import (
    MaskDirection, direction_from_diagnosis, measure_direction,
    measure_tonal_weights,
)
from .models import (
    GLOBAL_DELTA_E_TARGETS, LOCAL_DELTA_E_TARGETS, LOCAL_VISIBILITY_FLOOR,
    MASK_REACH_GATE, SUBJECT_HEADROOM_GATE, resolve_local_target,
)
from .persistence import AuditStore


class RenderError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code


class FullPresetRenderer(ABC):
    @abstractmethod
    def render_full(self, input_path: Path, preset: LutRecord) -> np.ndarray: ...


class CanonicalCpuLutRenderer(FullPresetRenderer):
    """Canonical packed-LUT CPU oracle used by reach probing and final rendering."""

    def __init__(self, databuild_config: Path, catalog: LutCatalog) -> None:
        from .source_reach import configured_lut_loader

        self._loader = configured_lut_loader(databuild_config)
        self._preset_ids = frozenset(row.preset_id for row in catalog.records)

    @property
    def renderable_preset_ids(self) -> frozenset[str]:
        return self._preset_ids

    def render_full(self, input_path: Path, preset: LutRecord) -> np.ndarray:
        from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

        try:
            grid, domain_min, domain_max = self._loader.load(Path(preset.path))
            return apply_lut_cpu_oracle(
                load_rgb(input_path), grid, domain_min=domain_min, domain_max=domain_max
            ).astype(np.float32)
        except Exception as exc:
            raise RenderError("preset_not_cpu_renderable", preset.preset_id) from exc


class CanonicalGpuRenderer(FullPresetRenderer):
    """Adapter over the existing immutable local-GPU renderer."""

    def __init__(self, databuild_config: Path, catalog: LutCatalog) -> None:
        try:
            from construct.config import load_config
            from construct.presets import (
                PresetCatalog as CanonicalCatalog,
                PresetRecord as CanonicalRecord,
                TaxonomyLink as CanonicalLink,
                packed_lut_paths,
            )
            from construct.rendering import LocalGpuOnlyRenderer
        except ImportError:
            from dataset_build.src.construct.config import load_config
            from dataset_build.src.construct.presets import (
                PresetCatalog as CanonicalCatalog,
                PresetRecord as CanonicalRecord,
                TaxonomyLink as CanonicalLink,
                packed_lut_paths,
            )
            from dataset_build.src.construct.rendering import LocalGpuOnlyRenderer

        config = load_config(databuild_config, require_private=True, validate_paths=True)
        lut_config = dataclasses.replace(config, preset_filter="lut")
        packed_paths = packed_lut_paths(config.presets.bank_dir)
        links = []
        for record in catalog.records:
            path = Path(record.path)
            if os.path.realpath(path) not in packed_paths:
                continue
            preset = CanonicalRecord(
                preset_id=record.preset_id, path=path, format="lut", kind="lut",
                style_name=record.name, fidelity_de=None, render_engine="gpu_lut",
            )
            links.append(CanonicalLink(
                preset=preset, major=record.style_major, minor=record.style_minor,
            ))
        canonical = CanonicalCatalog.from_links(links)
        if not canonical.by_id:
            raise RenderError("catalog_binding_empty")
        self._renderer = LocalGpuOnlyRenderer.create(lut_config)
        self._renderer.bind_catalog(canonical)
        self._canonical = canonical.by_id

    @property
    def renderable_preset_ids(self) -> frozenset[str]:
        return frozenset(self._canonical)

    def render_full(self, input_path: Path, preset: LutRecord) -> np.ndarray:
        try:
            from construct.rendering import preprocess_source
        except ImportError:
            from dataset_build.src.construct.rendering import preprocess_source

        bound = self._canonical.get(preset.preset_id)
        if bound is None:
            raise RenderError("preset_not_gpu_renderable", preset.preset_id)
        prepared = preprocess_source(input_path, short_edge=1024)
        rendered = self._renderer.render(prepared, bound, None)
        return np.asarray(rendered.pixels, dtype=np.float32)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def load_alpha(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        alpha = image.convert("L")
        if alpha.size != size:
            alpha = alpha.resize(size, getattr(Image, "Resampling", Image).BILINEAR)
        return np.asarray(alpha, dtype=np.float32) / 255.0


def apply_global_strength(before: np.ndarray, full_edit: np.ndarray, strength: float) -> np.ndarray:
    if before.shape != full_edit.shape:
        raise RenderError("render_shape_mismatch")
    value = float(np.clip(strength, 0.0, 1.0))
    return (before * (1.0 - value) + full_edit * value).astype(np.float32)


def apply_local_strength(
    before: np.ndarray, full_edit: np.ndarray, alpha: np.ndarray, strength: float
) -> np.ndarray:
    if before.shape != full_edit.shape or alpha.shape != before.shape[:2]:
        raise RenderError("render_shape_mismatch")
    weight = np.clip(alpha * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
    result = before * (1.0 - weight[..., None]) + full_edit * weight[..., None]
    return result.astype(np.float32)


def delta_e_map(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    return np.asarray(deltaE_ciede2000(rgb2lab(before), rgb2lab(after)), dtype=np.float32)


def _sample_indices(size: int, request_hash: str, maximum: int = 4096) -> np.ndarray:
    if size <= maximum:
        return np.arange(size, dtype=np.int64)
    seed = int(request_hash[:16], 16)
    return np.sort(np.random.default_rng(seed).choice(size, maximum, replace=False))


def _sample_metric(
    before: np.ndarray, full: np.ndarray, alpha: np.ndarray | None,
    strength: float, indices: np.ndarray,
) -> tuple[float, float]:
    before_sample = before.reshape(-1, 3)[indices].reshape(-1, 1, 3)
    full_sample = full.reshape(-1, 3)[indices].reshape(-1, 1, 3)
    if alpha is None:
        after = apply_global_strength(before_sample, full_sample, strength)
        actual = float(delta_e_map(before_sample, after).mean())
    else:
        alpha_sample = alpha.reshape(-1)[indices].reshape(-1, 1)
        after = apply_local_strength(before_sample, full_sample, alpha_sample, strength)
        delta = delta_e_map(before_sample, after)
        actual = float((delta * alpha_sample).sum() / max(float(alpha_sample.sum()), 1e-8))
    return actual, _new_clip_fraction(before_sample, after)


def _applied_alpha_metrics(applied: np.ndarray, subject: np.ndarray) -> dict[str, float]:
    selected = subject > 0.5
    values = applied[selected]
    if values.size == 0:
        raise RenderError("empty_subject_mask")
    outside = applied[~selected]
    return {
        "effective_alpha_mean": float(applied.mean()),
        "subject_high_coverage": float((values >= 0.5).mean()),
        "subject_support_coverage": float((values > 0.05).mean()),
        "subject_alpha_mean": float(values.mean()),
        "background_alpha_mean": float(outside.mean()) if outside.size else 0.0,
        "half_area": float((applied >= 0.5).mean()),
    }


def _applied_alpha_rejected(
    metrics: Mapping[str, float], mask: Mapping[str, Any]
) -> bool:
    """Role-aware applied-alpha gate.

    Subject masks keep the frozen Mask v2 coverage gate.

    Background masks: the avoidance half of `BACKGROUND_ROLE_GATE`
    (`subject_alpha_mean` / `subject_high_coverage`) is NOT re-checked here. Applied
    alpha is `alpha * strength` with `strength` in [0, 1], so both avoidance numbers
    are monotonically non-increasing in strength; the build-time gate on the raw
    alpha already bounds them, which made the re-check vacuously true (B6 blocker 5).
    It is replaced by the pre-registered visibility floor: the calibrated alpha must
    still average at least `applied_background_alpha_mean_min` over the background
    region, so a strength small enough to make the background edit invisible fails.
    """
    if str(mask.get("role") or "subject") == "background":
        return metrics["background_alpha_mean"] < \
            BACKGROUND_ROLE_GATE["applied_background_alpha_mean_min"]
    return str(mask.get("family")) in {"radial", "band"} and (
        metrics["effective_alpha_mean"] <= 0.45
        or metrics["subject_high_coverage"] < 0.98
        or metrics["subject_support_coverage"] < 1.0
    )


def _new_clip_fraction(before: np.ndarray, after: np.ndarray) -> float:
    threshold = 1.0 / 255.0
    before_clip = (before <= threshold) | (before >= 1.0 - threshold)
    after_clip = (after <= threshold) | (after >= 1.0 - threshold)
    return float(np.logical_and(after_clip, ~before_clip).mean())


def _high_clipped(pixels: np.ndarray) -> np.ndarray:
    return (pixels >= 1.0 - 1.0 / 255.0).any(axis=-1)


def subject_clip_regression(
    before: np.ndarray, after: np.ndarray, subject: np.ndarray
) -> float:
    """B2 posterior column: subject pixels newly highlight-clipped by the local edit."""
    selected = subject > 0.5
    if not selected.any():
        raise RenderError("empty_subject_mask")
    was = _high_clipped(before[selected])
    now = _high_clipped(after[selected])
    return float(np.logical_and(now, ~was).mean())


def subject_highlight_headroom(
    rgb: np.ndarray, subject: np.ndarray
) -> dict[str, float | bool]:
    """B2 candidate-side reading of the subject region of global_after.

    `near_clip_fraction` is the share of subject pixels whose max channel already sits
    at or above 250/255; `p99_luma` is the 99th percentile of that max channel.
    """
    selected = subject > 0.5
    if not selected.any():
        raise RenderError("empty_subject_mask")
    values = rgb[selected]
    channel_max = values.max(axis=-1)
    channel_min = values.min(axis=-1)
    near = float((channel_max >= SUBJECT_HEADROOM_GATE["near_clip_level"]).mean())
    p99 = float(np.quantile(channel_max, 0.99))
    saturation = float(
        ((channel_max - channel_min) / np.maximum(channel_max, 1e-6)).mean()
    )
    pressure = near > SUBJECT_HEADROOM_GATE["near_clip_fraction_max"] \
        or p99 > SUBJECT_HEADROOM_GATE["p99_luma_max"]
    return {
        "near_clip_fraction": near, "p99_luma": p99,
        "subject_saturation_mean": saturation,
        "subject_pixels": float(int(selected.sum())),
        "highlight_pressure": bool(pressure),
        "headroom_ok": bool(not pressure),
    }


def measure_subject_headroom(
    artifacts: ArtifactStore, image: Mapping[str, Any], subject: Mapping[str, Any]
) -> dict[str, float | bool]:
    rgb = load_rgb(artifacts.path_for(dict(image)))
    alpha = load_alpha(
        artifacts.path_for(dict(subject)), (rgb.shape[1], rgb.shape[0])
    )
    return subject_highlight_headroom(rgb, alpha)


MASK_REACH_CONTRACT = "mask-reach-sample1024-full-strength-de00-v1"


def _mask_reach_indices(support: np.ndarray, seed_hash: str) -> np.ndarray:
    """Deterministic <=`sample_pixels` subset of the mask support (B8 item 4)."""
    budget = int(MASK_REACH_GATE["sample_pixels"])
    if support.size <= budget:
        return support
    seed = int(seed_hash[:16], 16)
    picked = np.random.default_rng(seed).choice(support.size, budget, replace=False)
    return support[np.sort(picked)]


class MaskReachProbe:
    """R1: mask-conditioned reachable ΔE of one (mask, LUT) pair.

    The frozen `d_full` reach probe is a whole-image reading; local calibration is a
    mask-weighted one. This probe answers the mask-conditioned question with the exact
    quantity local calibration maximizes: the alpha-weighted mean CIEDE2000 of
    `apply_local_strength(..., strength=1.0)` on a deterministic sample of the mask
    support. It is therefore the upper bound of the calibration objective on that mask.

    The LUT is always applied through the packed-LUT CPU oracle (the same oracle the
    frozen reach probe uses), independent of the configured render backend.
    """

    def __init__(
        self, artifacts: ArtifactStore, catalog: LutCatalog, *,
        input_artifact: Mapping[str, Any], source_sha256: str, loader: Any,
    ) -> None:
        self.artifacts = artifacts
        self.catalog = catalog
        self.input_artifact = dict(input_artifact)
        self.source_sha256 = str(source_sha256)
        self.loader = loader
        self._before: np.ndarray | None = None
        self._samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._values: dict[tuple[str, str], float] = {}

    def _image(self) -> np.ndarray:
        if self._before is None:
            self._before = load_rgb(self.artifacts.path_for(self.input_artifact))
        return self._before

    def _sample(self, mask: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        alpha_ref = dict(mask["alpha_artifact"])
        key = str(alpha_ref["sha256"])
        cached = self._samples.get(key)
        if cached is not None:
            return cached
        before = self._image()
        alpha = load_alpha(
            self.artifacts.path_for(alpha_ref), (before.shape[1], before.shape[0])
        ).reshape(-1)
        support = np.nonzero(alpha > MASK_REACH_GATE["alpha_min"])[0]
        seed_hash = hashlib.sha256(canonical_json({
            "contract": MASK_REACH_CONTRACT, "source_sha256": self.source_sha256,
            "mask_sha256": key,
        }).encode()).hexdigest()
        indices = _mask_reach_indices(support, seed_hash)
        sample = (
            before.reshape(-1, 3)[indices].reshape(-1, 1, 3),
            alpha[indices].reshape(-1, 1),
        )
        self._samples[key] = sample
        return sample

    def measure(self, mask: Mapping[str, Any], preset_id: str) -> float:
        """Alpha-weighted mean ΔE00 of `preset_id` at full strength on `mask`."""
        key = (str(mask["alpha_artifact"]["sha256"]), str(preset_id))
        cached = self._values.get(key)
        if cached is not None:
            return cached
        from dataset_build.src.construct.rendering import apply_lut_cpu_oracle

        pixels, weights = self._sample(mask)
        if pixels.size == 0 or float(weights.sum()) <= 0.0:
            self._values[key] = 0.0
            return 0.0
        record = self.catalog.get(str(preset_id))
        try:
            grid, domain_min, domain_max = self.loader.load(Path(record.path))
            full = np.asarray(apply_lut_cpu_oracle(
                pixels, grid, domain_min=domain_min, domain_max=domain_max
            ), dtype=np.float32)
        except Exception as exc:
            raise RenderError("mask_reach_not_renderable", str(preset_id)) from exc
        after = apply_local_strength(pixels, full, weights, 1.0)
        delta = delta_e_map(pixels, after)
        value = float((delta * weights).sum() / float(weights.sum()))
        if not np.isfinite(value):
            raise RenderError("mask_reach_nonfinite", str(preset_id))
        self._values[key] = value
        return value


class LocalDirectionProbe:
    """R7.1/R6.2: the two direction vectors and the tonal weighting of one mask.

    `correction` is measured on `source -> global_after` restricted to the mask, which
    is the residual the global edit left behind there; `enhancement` is the keyword
    direction of the frozen diagnosis' `enhancement_opportunities` and is therefore the
    same for every mask of a source. `tonal_weights` is read on `global_after`, the
    image the local LUT will actually be applied to.
    """

    def __init__(
        self, artifacts: ArtifactStore, *, source_artifact: Mapping[str, Any],
        global_after_artifact: Mapping[str, Any], diagnosis: Mapping[str, Any],
    ) -> None:
        self.artifacts = artifacts
        self.source_artifact = dict(source_artifact)
        self.global_after_artifact = dict(global_after_artifact)
        self.enhancement = direction_from_diagnosis(dict(diagnosis), "enhancement")
        self._before: np.ndarray | None = None
        self._after: np.ndarray | None = None
        self._directions: dict[str, MaskDirection] = {}

    def _images(self) -> tuple[np.ndarray, np.ndarray]:
        if self._before is None or self._after is None:
            self._before = load_rgb(self.artifacts.path_for(self.source_artifact))
            self._after = load_rgb(
                self.artifacts.path_for(self.global_after_artifact)
            )
            if self._before.shape != self._after.shape:
                raise RenderError(
                    "direction_shape_mismatch",
                    f"{self._before.shape} vs {self._after.shape}",
                )
        return self._before, self._after

    def direction(self, mask: Mapping[str, Any]) -> MaskDirection:
        key = str(mask["alpha_artifact"]["sha256"])
        cached = self._directions.get(key)
        if cached is not None:
            return cached
        before, after = self._images()
        alpha = load_alpha(
            self.artifacts.path_for(dict(mask["alpha_artifact"])),
            (before.shape[1], before.shape[0]),
        )
        result = MaskDirection(
            correction=measure_direction(before, after, alpha, mode="correction"),
            enhancement=self.enhancement,
            tonal_weights=measure_tonal_weights(after, alpha),
        )
        self._directions[key] = result
        return result


class StrengthCalibrator:
    def __init__(
        self, renderer: FullPresetRenderer, catalog: LutCatalog, artifacts: ArtifactStore,
        audit: AuditStore, *, renderer_revision: str, search_steps: int = 5,
        clip_fraction_max: float = 0.01, concurrency: int = 2, backend: str = "",
    ) -> None:
        self.renderer = renderer
        self.catalog = catalog
        self.artifacts = artifacts
        self.audit = audit
        self.renderer_revision = renderer_revision
        self.search_steps = search_steps
        self.clip_fraction_max = clip_fraction_max
        # C1b item 6: `render.backend` picks a different `FullPresetRenderer`
        # implementation for the same `renderer_revision`, so it is part of what
        # produced a cached calibration.
        self.backend = str(backend or "")
        self._semaphore = threading.BoundedSemaphore(max(1, concurrency))

    def calibrate_global(
        self, *, source_sha256: str, branch_id: str, input_artifact: Mapping[str, Any],
        preset_id: str, strength_bin: str,
    ) -> dict[str, Any]:
        if strength_bin not in GLOBAL_DELTA_E_TARGETS:
            raise RenderError("invalid_global_strength_bin")
        low, high, inclusive = GLOBAL_DELTA_E_TARGETS[strength_bin]
        return self._calibrate(
            source_sha256=source_sha256, branch_id=branch_id, stage="global",
            input_artifact=input_artifact, preset_id=preset_id, mask=None,
            strength_bin=strength_bin, target=(low, high, inclusive),
        )

    def calibrate_local(
        self, *, source_sha256: str, branch_id: str, input_artifact: Mapping[str, Any],
        preset_id: str, mask: Mapping[str, Any], strength_bin: str, intent: str,
        subject_p99_luma: float | None = None,
    ) -> dict[str, Any]:
        """B7 items 1-2: the target band is looked up per intent, and a capped intent
        on a near-white subject is served one bin lower."""
        if intent not in LOCAL_DELTA_E_TARGETS:
            raise RenderError("invalid_local_intent")
        if strength_bin not in LOCAL_DELTA_E_TARGETS[intent]:
            raise RenderError("invalid_local_strength_bin")
        low, high, inclusive, effective_bin, capped = resolve_local_target(
            intent, strength_bin, subject_p99_luma
        )
        # B8 item 1 runtime assertion of the pre-registered transition floor. It also
        # covers the false-white soft cap: a capped intent served one bin lower is
        # rejected here whenever that lower band opens below the floor.
        if float(low) < LOCAL_VISIBILITY_FLOOR:
            raise RenderError(
                "local_visibility_floor",
                f"{intent}/{effective_bin} target {low} < {LOCAL_VISIBILITY_FLOOR}",
            )
        result = self._calibrate(
            source_sha256=source_sha256, branch_id=branch_id, stage="local",
            input_artifact=input_artifact, preset_id=preset_id, mask=mask,
            strength_bin=strength_bin, target=(low, high, inclusive),
        )
        result["luma_capped"] = bool(capped)
        result["effective_strength_bin"] = effective_bin
        return result

    def _calibrate(
        self, *, source_sha256: str, branch_id: str, stage: str,
        input_artifact: Mapping[str, Any], preset_id: str,
        mask: Mapping[str, Any] | None, strength_bin: str,
        target: tuple[float, float, bool],
    ) -> dict[str, Any]:
        preset = self.catalog.get(preset_id)
        input_sha = str(input_artifact["sha256"])
        alpha_sha = str(mask["alpha_artifact"]["sha256"]) if mask else "global"
        request_key = {
            # C1b item 6: `clip_fraction_max` vetoes candidate strengths inside the
            # bisection and `backend` decides which renderer produced `full`; both
            # change the result for an otherwise identical request, so both are part
            # of the cache key. Bumped v2 -> v3 so pre-C1b cached rows never resolve.
            "calibration_contract": "sample4096-bisect-applied-alpha-v3",
            "renderer_revision": self.renderer_revision, "input_image_sha256": input_sha,
            "preset_id": preset_id, "mask_sha256_or_global": alpha_sha,
            "strength_bin": strength_bin, "search_steps": self.search_steps,
            "target": target, "clip_fraction_max": self.clip_fraction_max,
            "backend": self.backend,
        }
        request_hash = hashlib.sha256(canonical_json(request_key).encode()).hexdigest()
        resolved_hash = self.audit.get_render_calibration(request_hash)
        cached_request = self.audit.get_render(resolved_hash) if resolved_hash else None
        if cached_request and cached_request.get("status") == "accepted":
            artifact = cached_request.get("artifact_json")
            if artifact:
                try:
                    self.artifacts.path_for(artifact)
                    parameters = cached_request["parameters_json"]
                    applied = parameters.get("applied_alpha_artifact")
                    if mask and not applied:
                        raise FileNotFoundError("cached local render lacks applied alpha")
                    if applied:
                        self.artifacts.path_for(applied)
                except FileNotFoundError:
                    pass
                else:
                    return {
                        "render_hash": resolved_hash, "artifact": artifact,
                        "applied_alpha_artifact": applied,
                        "metrics": cached_request["metrics_json"],
                        "parameters": parameters, "cache_hit": True,
                    }
        wall_started = time.perf_counter()
        cpu_started = time.thread_time()

        def add_timing(values: dict[str, float]) -> dict[str, float]:
            values["render_wall_seconds"] = time.perf_counter() - wall_started
            values["render_thread_cpu_seconds"] = time.thread_time() - cpu_started
            return values

        before = load_rgb(self.artifacts.path_for(dict(input_artifact)))
        with self._semaphore:
            full = self.renderer.render_full(self.artifacts.path_for(dict(input_artifact)), preset)
        if full.shape != before.shape:
            image = Image.fromarray(np.clip(full * 255 + 0.5, 0, 255).astype(np.uint8), "RGB")
            image = image.resize((before.shape[1], before.shape[0]),
                                 getattr(Image, "Resampling", Image).LANCZOS)
            full = np.asarray(image, dtype=np.float32) / 255.0
        if not np.isfinite(full).all():
            raise RenderError("nonfinite_full_render")
        alpha = load_alpha(
            self.artifacts.path_for(mask["alpha_artifact"]), (before.shape[1], before.shape[0])
        ) if mask else None
        low, high, inclusive = target
        center = (low + high) / 2.0
        indices = _sample_indices(before.shape[0] * before.shape[1], request_hash)
        candidates: list[tuple[float, float, float, float]] = []
        left, right = 0.0, 1.0
        for _ in range(self.search_steps):
            strength = (left + right) / 2.0
            actual, sampled_clip = _sample_metric(before, full, alpha, strength, indices)
            candidates.append((abs(actual - center), strength, actual, sampled_clip))
            if actual < center:
                left = strength
            else:
                right = strength
        for strength in (left, right, 1.0):
            actual, sampled_clip = _sample_metric(before, full, alpha, strength, indices)
            candidates.append((abs(actual - center), strength, actual, sampled_clip))
        best: tuple[float, float, dict[str, float]] | None = None
        for deviation, strength, actual, sampled_clip in candidates:
            in_range = low <= actual <= high if inclusive else low <= actual < high
            if in_range and sampled_clip <= self.clip_fraction_max:
                metrics = {
                    "delta_e": actual, "clip_fraction_sampled": sampled_clip,
                    "delta_e_sample_size": float(indices.size),
                }
                candidate = (deviation, strength, metrics)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
        if best is None:
            # B8 item 1: a local leaf that never reached the transition floor at any
            # searched strength is a visibility rejection, not a band-placement miss.
            reached = max((actual for _d, _s, actual, _c in candidates), default=0.0)
            floored = stage == "local" and reached < LOCAL_VISIBILITY_FLOOR
            code = "local_visibility_floor" if floored else "strength_target_unreachable"
            self.audit.record_render({
                "render_hash": request_hash, "source_sha256": source_sha256,
                "branch_id": branch_id, "stage": stage, "input": request_key,
                "parameters": {"strength_bin": strength_bin},
                "metrics": add_timing({
                    "target_low": low, "target_high": high,
                    "delta_e_reached_max": float(reached),
                    "local_visibility_floor": LOCAL_VISIBILITY_FLOOR,
                }),
                "artifact": None,
                "status": code if floored else "calibration_failed",
            })
            raise RenderError(code)
        _deviation, strength, metrics = best
        applied_alpha = np.clip(alpha * strength, 0.0, 1.0) if alpha is not None else None
        applied_alpha_artifact = None
        subject = None
        if applied_alpha is not None:
            subject_ref = mask.get("subject_artifact") if mask else None
            if not subject_ref:
                # C1b item 7: the applied-alpha role gate is a pre-registered judgement
                # column, and it can only run against the subject mask. A local render
                # that arrives without `subject_artifact` used to skip the gate in
                # silence and produce an ungated leaf that still looked accepted; it is
                # a wiring bug and it fails loud.
                self.audit.record_render({
                    "render_hash": request_hash, "source_sha256": source_sha256,
                    "branch_id": branch_id, "stage": stage, "input": request_key,
                    "parameters": {"strength_bin": strength_bin,
                                   "local_strength": strength},
                    "metrics": add_timing(metrics), "artifact": None,
                    "status": "subject_artifact_missing",
                })
                raise RenderError(
                    "subject_artifact_missing",
                    f"mask {mask.get('mask_id') if mask else '<none>'} carries no "
                    "subject_artifact, so the applied-alpha gate cannot run",
                )
            subject = load_alpha(
                self.artifacts.path_for(subject_ref), (before.shape[1], before.shape[0])
            )
            applied_metrics = _applied_alpha_metrics(applied_alpha, subject)
            metrics.update({f"applied_alpha_{key}": value
                            for key, value in applied_metrics.items()})
            if _applied_alpha_rejected(applied_metrics, mask):
                self.audit.record_render({
                    "render_hash": request_hash, "source_sha256": source_sha256,
                    "branch_id": branch_id, "stage": stage, "input": request_key,
                    "parameters": {"strength_bin": strength_bin,
                                   "local_strength": strength},
                    "metrics": add_timing(metrics), "artifact": None,
                    "status": "applied_alpha_rejected",
                })
                raise RenderError("applied_alpha_mask_gate")
        after = apply_global_strength(before, full, strength) if alpha is None else \
            apply_local_strength(before, full, alpha, strength)
        metrics["clip_fraction_new"] = _new_clip_fraction(before, after)
        metrics["finite"] = float(np.isfinite(after).all())
        if metrics["clip_fraction_new"] > self.clip_fraction_max:
            raise RenderError("clip_fraction_exceeded")
        if stage == "local" and subject is not None:
            # B2 posterior guard: `before` is global_after for a local render, so this
            # is the newly clipped subject share the local edit itself introduced.
            regression = subject_clip_regression(before, after, subject)
            metrics["subject_clip_regression"] = regression
            metrics["subject_clip_regression_max"] = \
                SUBJECT_HEADROOM_GATE["subject_clip_regression_max"]
            if regression > SUBJECT_HEADROOM_GATE["subject_clip_regression_max"]:
                self.audit.record_render({
                    "render_hash": request_hash, "source_sha256": source_sha256,
                    "branch_id": branch_id, "stage": stage, "input": request_key,
                    "parameters": {"strength_bin": strength_bin,
                                   "local_strength": strength},
                    "metrics": add_timing(metrics), "artifact": None,
                    "status": "highlight_clip_regression",
                })
                raise RenderError("highlight_clip_regression")
        if applied_alpha is not None:
            applied_alpha_artifact = self.artifacts.put_alpha(
                applied_alpha, retention="quarantine"
            ).to_dict()
        artifact = self.artifacts.put_image_array(after, retention="quarantine")
        add_timing(metrics)
        parameters = {
            "preset_id": preset_id,
            "global_strength": strength if stage == "global" else None,
            "local_strength": strength if stage == "local" else None,
            "mask_id": mask.get("mask_id") if mask else None,
            "applied_alpha_artifact": applied_alpha_artifact,
            "strength_bin": strength_bin,
        }
        final_hash = hashlib.sha256(canonical_json({
            "renderer_revision": self.renderer_revision, "input_image_sha256": input_sha,
            "preset_id": preset_id, "calibrated_strength": strength,
            "mask_sha256_or_global": alpha_sha,
            "render_settings": {"jpeg_quality": 95},
        }).encode()).hexdigest()
        row = {
            "render_hash": final_hash, "source_sha256": source_sha256,
            "branch_id": branch_id, "stage": stage, "input": request_key,
            "parameters": parameters, "metrics": metrics,
            "artifact": artifact.to_dict(), "status": "accepted",
        }
        self.audit.record_render(row)
        self.audit.record_render_calibration(request_hash, final_hash)
        return {
            "render_hash": final_hash, "artifact": artifact.to_dict(), "metrics": metrics,
            "applied_alpha_artifact": applied_alpha_artifact,
            "parameters": parameters, "cache_hit": False,
        }


__all__ = [
    "CanonicalCpuLutRenderer", "CanonicalGpuRenderer", "FullPresetRenderer",
    "LocalDirectionProbe", "MASK_REACH_CONTRACT", "MaskReachProbe", "RenderError",
    "StrengthCalibrator", "apply_global_strength", "apply_local_strength", "delta_e_map",
    "load_alpha", "load_rgb", "measure_subject_headroom", "subject_clip_regression",
    "subject_highlight_headroom",
]
