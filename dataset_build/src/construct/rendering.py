"""Immutable local-GPU-only renderer for canonical databuild."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from dataset_build.lut_io import load_lut

from .canonical_masks import MaskAsset
from .config import DatabuildConfig
from .presets import PresetCatalog, PresetRecord
from .visibility import assert_alpha_zero_endpoint


_RESAMPLING = getattr(Image, "Resampling", Image)


class GpuRenderError(RuntimeError):
    """Canonical rendering failed closed instead of using farm or CPU operators."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class PreparedSource:
    pixels: np.ndarray
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class RenderedCandidate:
    pixels: np.ndarray
    engine: str
    diagnostics: dict[str, Any]


def preprocess_source(path: str | os.PathLike[str], short_edge: int = 1024) -> PreparedSource:
    if short_edge <= 0:
        raise ValueError("short_edge must be positive")
    with Image.open(path) as image:
        image.load()
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        scale = short_edge / min(oriented.width, oriented.height)
        width = max(1, int(round(oriented.width * scale)))
        height = max(1, int(round(oriented.height * scale)))
        if oriented.size != (width, height):
            oriented = oriented.resize((width, height), _RESAMPLING.LANCZOS)
        pixels = np.asarray(oriented, dtype=np.float32) / 255.0
    return PreparedSource(pixels=pixels, width=width, height=height)


def composite_srgb(before: np.ndarray, edited: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    base = np.asarray(before, dtype=np.float32)
    complete = np.asarray(edited, dtype=np.float32)
    weight = np.asarray(alpha, dtype=np.float32)
    if base.shape != complete.shape or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("before and edited must be matching HWC RGB arrays")
    if weight.shape != base.shape[:2]:
        raise ValueError("alpha must match image dimensions")
    weight = np.clip(weight, 0.0, 1.0)
    mixed = base * (1.0 - weight[..., None]) + complete * weight[..., None]
    mixed[weight == 0] = base[weight == 0]
    mixed[weight == 1] = complete[weight == 1]
    return mixed.astype(np.float32)


def apply_lut_cpu_oracle(image: np.ndarray, grid_bgr: np.ndarray,
                         domain_min: np.ndarray | None = None,
                         domain_max: np.ndarray | None = None) -> np.ndarray:
    """Test-only trilinear oracle for standard .cube B/G/R storage order."""
    source = np.asarray(image, dtype=np.float32)
    grid = np.asarray(grid_bgr, dtype=np.float32)
    if grid.ndim != 4 or grid.shape[-1] != 3 or len(set(grid.shape[:3])) != 1:
        raise ValueError("LUT grid must be cubic BGRxRGB")
    dmin = np.zeros(3, dtype=np.float32) if domain_min is None else np.asarray(domain_min)
    dmax = np.ones(3, dtype=np.float32) if domain_max is None else np.asarray(domain_max)
    span = np.where(dmax == dmin, 1.0, dmax - dmin)
    coords = np.clip((source - dmin) / span, 0.0, 1.0) * (grid.shape[0] - 1)
    lo = np.floor(coords).astype(np.int32)
    hi = np.minimum(lo + 1, grid.shape[0] - 1)
    frac = coords - lo
    r0, g0, b0 = lo[..., 0], lo[..., 1], lo[..., 2]
    r1, g1, b1 = hi[..., 0], hi[..., 1], hi[..., 2]
    fr, fg, fb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]
    c000 = grid[b0, g0, r0]
    c100 = grid[b0, g0, r1]
    c010 = grid[b0, g1, r0]
    c110 = grid[b0, g1, r1]
    c001 = grid[b1, g0, r0]
    c101 = grid[b1, g0, r1]
    c011 = grid[b1, g1, r0]
    c111 = grid[b1, g1, r1]
    c00 = c000 * (1 - fr) + c100 * fr
    c10 = c010 * (1 - fr) + c110 * fr
    c01 = c001 * (1 - fr) + c101 * fr
    c11 = c011 * (1 - fr) + c111 * fr
    c0 = c00 * (1 - fg) + c10 * fg
    c1 = c01 * (1 - fg) + c11 * fg
    return np.clip(c0 * (1 - fb) + c1 * fb, 0.0, 1.0).astype(np.float32)


class _LutLoader:
    def __init__(self, bank_dir: Path):
        self.bank_dir = bank_dir
        self._lock = threading.Lock()
        self._cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._packed = None
        self._packed_by_path: dict[str, tuple[str, np.ndarray, np.ndarray]] = {}
        meta_path = bank_dir / "luts_meta.json"
        packed_path = bank_dir / "luts.npz"
        if meta_path.is_file() and packed_path.is_file():
            with meta_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self._packed = np.load(packed_path, allow_pickle=False)
            for preset_id, row in metadata.items():
                path = os.path.realpath(str(row.get("path") or ""))
                self._packed_by_path[path] = (
                    preset_id,
                    np.asarray(row.get("dmin", (0, 0, 0)), dtype=np.float32),
                    np.asarray(row.get("dmax", (1, 1, 1)), dtype=np.float32),
                )

    def load(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        real = os.path.realpath(path)
        with self._lock:
            cached = self._cache.get(real)
            if cached is not None:
                return cached
            packed = self._packed_by_path.get(real)
            if packed is not None and self._packed is not None:
                preset_id, dmin, dmax = packed
                grid = np.asarray(self._packed[preset_id], dtype=np.float32)
                result = (grid, dmin, dmax)
            else:
                result = load_lut(real)
            self._cache[real] = result
            return result


class LocalGpuOnlyRenderer:
    """Dedicated renderer with no farm or CPU-operator fallback edge."""

    def __init__(self, config: DatabuildConfig):
        self.config = config
        self.device = ""
        self._torch = None
        self._semaphore = threading.BoundedSemaphore(config.render.gpu_concurrency)
        self._parse_lock = threading.Lock()
        self._preset_cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._allowed_presets: MappingProxyType | None = None
        self._lut_loader = _LutLoader(config.presets.bank_dir)
        self._preflight()

    @classmethod
    def create(cls, config: DatabuildConfig) -> "LocalGpuOnlyRenderer":
        return cls(config)

    def _preflight(self) -> None:
        try:
            import torch
        except ImportError as exc:
            raise GpuRenderError("torch_unavailable", "PyTorch is required for GPU rendering") from exc
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise GpuRenderError("cuda_unavailable", "canonical databuild requires CUDA")
        self.device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
        configured = os.environ.get("MONETGPT_TORCH_DEVICE")
        if configured and configured != self.device:
            raise GpuRenderError(
                "gpu_redirection_rejected",
                f"MONETGPT_TORCH_DEVICE cannot redirect canonical rendering from {self.device}",
            )
        os.environ["MONETGPT_TORCH_DEVICE"] = self.device
        self._torch = torch
        probe = torch.zeros((1,), device=self.device)
        self._assert_gpu_tensor(probe)
        del probe

    def assert_ready(self) -> None:
        torch = self._torch
        if torch is None or not torch.cuda.is_available():
            raise GpuRenderError("cuda_lost", "CUDA became unavailable after startup")
        current = os.environ.get("MONETGPT_TORCH_DEVICE")
        if current != self.device:
            raise GpuRenderError("gpu_redirection_rejected", "GPU backend changed after preflight")

    @staticmethod
    def _expected_engine(preset_format: str) -> str:
        if preset_format == "lut":
            return "gpu_lut"
        if preset_format in {"xmp", "lrtemplate"}:
            return "gpu_local_preset"
        raise GpuRenderError("unsupported_format", preset_format)

    @classmethod
    def _preset_signature(cls, preset: PresetRecord) -> tuple[str, str, str]:
        expected_engine = cls._expected_engine(preset.format)
        if preset.render_engine != expected_engine:
            raise GpuRenderError(
                "capability_binding_rejected",
                f"preset {preset.preset_id} engine does not match {preset.format}",
            )
        suffix = preset.path.suffix.lower()
        valid_suffixes = {
            "lut": {".cube", ".3dl"},
            "xmp": {".xmp"},
            "lrtemplate": {".lrtemplate"},
        }[preset.format]
        if suffix not in valid_suffixes or not preset.path.is_file():
            raise GpuRenderError(
                "capability_binding_rejected",
                f"preset {preset.preset_id} path no longer matches its capability scan",
            )
        return os.path.realpath(preset.path), preset.format, preset.render_engine

    def bind_catalog(self, catalog: PresetCatalog) -> None:
        """Freeze the startup capability result used by every later render call."""
        allowed: dict[str, tuple[str, str, str]] = {}
        for link in catalog.links:
            preset = link.preset
            if preset.format not in self.config.effective_formats:
                raise GpuRenderError(
                    "capability_binding_rejected",
                    f"preset {preset.preset_id} uses a disabled format",
                )
            signature = self._preset_signature(preset)
            previous = allowed.setdefault(preset.preset_id, signature)
            if previous != signature:
                raise GpuRenderError(
                    "capability_binding_rejected",
                    f"preset {preset.preset_id} has conflicting capability records",
                )
        if not allowed:
            raise GpuRenderError("capability_binding_rejected", "catalog has no GPU presets")
        self._allowed_presets = MappingProxyType(allowed)

    def _assert_bound_preset(self, preset: PresetRecord) -> None:
        if self._allowed_presets is None:
            raise GpuRenderError(
                "capability_binding_missing", "renderer was not bound to the preflight catalog"
            )
        if self._allowed_presets.get(preset.preset_id) != self._preset_signature(preset):
            raise GpuRenderError(
                "capability_binding_rejected",
                f"preset {preset.preset_id} differs from the preflight capability result",
            )

    def render(self, source: PreparedSource, preset: PresetRecord,
               mask: MaskAsset | None = None) -> RenderedCandidate:
        self.assert_ready()
        self._assert_bound_preset(preset)
        with self._semaphore:
            before = self._upload(source.pixels)
            try:
                if preset.format == "lut":
                    edited, diagnostics = self._apply_lut(before, preset)
                    engine = "gpu_lut"
                elif preset.format in {"xmp", "lrtemplate"}:
                    edited, diagnostics = self._apply_param(before, preset)
                    engine = "gpu_local_preset"
                else:
                    raise GpuRenderError("unsupported_format", preset.format)
                self._assert_gpu_tensor(edited)
                if mask is None:
                    output = edited
                else:
                    if mask.effective_alpha.shape != (source.height, source.width):
                        raise GpuRenderError("mask_shape_mismatch", "C_GT does not match source grid")
                    alpha = self._torch.as_tensor(
                        mask.effective_alpha, dtype=before.dtype, device=self.device
                    )[None, None]
                    mixed = before * (1.0 - alpha) + edited * alpha
                    output = self._torch.where(
                        alpha == 0, before, self._torch.where(alpha == 1, edited, mixed)
                    )
                self._assert_gpu_tensor(output)
                array = output[0].permute(1, 2, 0).detach().cpu().numpy().astype(np.float32)
            except GpuRenderError:
                raise
            except Exception as exc:  # noqa: BLE001 - fail closed; never select another backend
                raise GpuRenderError("gpu_render_failed", f"{type(exc).__name__}: {exc}") from exc
            finally:
                del before
        if mask is not None:
            assert_alpha_zero_endpoint(source.pixels, array, mask.effective_alpha)
        return RenderedCandidate(array, engine, diagnostics)

    def _upload(self, pixels: np.ndarray):
        tensor = self._torch.from_numpy(
            np.ascontiguousarray(pixels.transpose(2, 0, 1))[None]
        ).to(self.device, dtype=self._torch.float32)
        self._assert_gpu_tensor(tensor)
        return tensor

    def _apply_param(self, before, preset: PresetRecord):
        from gpu_render.gpu.gpu_replay import replay_batch
        from gpu_render.local_apply import FITS_DIR

        key = (os.path.realpath(preset.path), preset.path.stat().st_mtime_ns)
        with self._parse_lock:
            parsed = self._preset_cache.get(key)
            if parsed is None:
                from gpu_render.replay import parse_preset

                parsed = parse_preset(str(preset.path), preset.format)
                if parsed.get("locals") or any(str(key).startswith("Local")
                                               for key in parsed.get("attrs", {})):
                    raise GpuRenderError(
                        "embedded_local_rejected", "preset contains nested local corrections"
                    )
                self._preset_cache[key] = parsed
        edited, info = replay_batch(before.clone(), parsed, fits_dir=FITS_DIR, fallback="skip")
        residual_id = preset.preset_id
        from gpu_render.residual import load_residual

        residual = load_residual(residual_id)
        if residual is None:
            residual_id = "_global"
            residual = load_residual(residual_id)
        residual_applied = False
        if residual is not None:
            from gpu_render.gpu.residual_gpu import apply_residual_batch

            edited = apply_residual_batch(edited, *residual)
            residual_applied = True
        public = {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in dict(info or {}).items()
            if key != "alpha"
        }
        public.update({"residual_id": residual_id, "residual_applied": residual_applied})
        return edited, public

    def _apply_lut(self, before, preset: PresetRecord):
        torch = self._torch
        import torch.nn.functional as functional

        grid, dmin, dmax = self._lut_loader.load(preset.path)
        volume = torch.from_numpy(grid).permute(3, 0, 1, 2)[None].to(self.device)
        domain_min = torch.as_tensor(dmin, dtype=before.dtype, device=self.device)[None, :, None, None]
        span_np = np.where(dmax == dmin, 1.0, dmax - dmin)
        span = torch.as_tensor(span_np, dtype=before.dtype, device=self.device)[None, :, None, None]
        coords = ((before - domain_min) / span).clamp(0.0, 1.0)
        points = (coords.permute(0, 2, 3, 1) * 2.0 - 1.0)[:, None]
        edited = functional.grid_sample(
            volume.expand(before.shape[0], -1, -1, -1, -1),
            points, mode="bilinear", padding_mode="border", align_corners=True,
        )[:, :, 0]
        return edited, {"lut_size": int(grid.shape[0]), "axis_order": "bgr"}

    def _assert_gpu_tensor(self, tensor) -> None:
        if not getattr(tensor, "is_cuda", False):
            raise GpuRenderError("cpu_backend_rejected", "render tensor is not CUDA-backed")
        if str(tensor.device) != self.device:
            raise GpuRenderError(
                "gpu_redirection_rejected",
                f"render tensor moved to {tensor.device}; expected {self.device}",
            )


def save_candidate_jpeg(pixels: np.ndarray, path: str | os.PathLike[str], quality: int = 95) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    image = Image.fromarray(
        np.clip(np.asarray(pixels) * 255.0 + 0.5, 0, 255).astype(np.uint8), "RGB"
    )
    image.save(tmp, format="JPEG", quality=quality)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def save_cgt_png(mask: MaskAsset, path: str | os.PathLike[str]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    image = Image.fromarray(
        np.clip(mask.effective_alpha * 255.0 + 0.5, 0, 255).astype(np.uint8), "L"
    )
    image.save(tmp, format="PNG", compress_level=6)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target
