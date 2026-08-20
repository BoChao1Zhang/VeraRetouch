"""Deterministic source-specific LUT strength reachability probing."""
from __future__ import annotations

import hashlib
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
from PIL import Image, ImageOps
from skimage.color import deltaE_ciede2000, rgb2lab

from dataset_build.lut_io import load_lut
from dataset_build.src.construct.rendering import _LutLoader, apply_lut_cpu_oracle

from .candidates import LutRecord
from .models import GLOBAL_DELTA_E_TARGETS


REACH_SCHEMA = "local-retouch-preset-reach-v1"
SAMPLER_REVISION = "sha256-coprime-stride-4096-v1"
SAMPLE_PIXELS = 4096

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib


@lru_cache(maxsize=4)
def configured_lut_loader(databuild_config: str | Path) -> _LutLoader:
    config_path = Path(databuild_config).expanduser().resolve()
    with config_path.open("rb") as handle:
        data = tomllib.load(handle)
    bank_dir = Path(str((data.get("presets") or {}).get("bank_dir") or ""))
    if not bank_dir.is_dir():
        raise ValueError(f"configured LUT bank is missing: {bank_dir}")
    return _LutLoader(bank_dir)


def catalog_digest(records: Iterable[LutRecord]) -> str:
    payload = "\n".join(
        f"{row.preset_id}\t{Path(row.path).resolve()}" for row in records
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sample_indices(pixel_count: int, source_sha256: str) -> np.ndarray:
    count = min(SAMPLE_PIXELS, pixel_count)
    if count == pixel_count:
        return np.arange(pixel_count, dtype=np.int64)
    seed = int(source_sha256[:16], 16)
    offset = seed % pixel_count
    stride = max(1, pixel_count // count)
    stride += int(source_sha256[16:24], 16) % max(stride, 1)
    while math.gcd(stride, pixel_count) != 1:
        stride += 1
    return (offset + np.arange(count, dtype=np.int64) * stride) % pixel_count


def sample_source_pixels(path: str | Path, source_sha256: str) -> np.ndarray:
    try:
        from dataset_build.tools.archive_reader import open_image

        image = open_image(path)
    except (ImportError, FileNotFoundError, ValueError):
        image = Image.open(path)
    with image:
        rgb = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float32)
    flat = (rgb.reshape(-1, 3) / 255.0).astype(np.float32)
    return np.ascontiguousarray(flat[_sample_indices(len(flat), source_sha256)])


def probe_preset_reach(
    source_path: str | Path, source_sha256: str, records: Iterable[LutRecord],
    *, loader: Callable[[Path], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> dict[str, Any]:
    ordered = tuple(sorted(records, key=lambda row: row.preset_id))
    pixels = sample_source_pixels(source_path, source_sha256)
    before_lab = rgb2lab(pixels.reshape(-1, 1, 3))
    presets: dict[str, dict[str, Any]] = {}
    load = loader or load_lut
    for row in ordered:
        try:
            grid, domain_min, domain_max = load(Path(row.path))
            rendered = apply_lut_cpu_oracle(
                pixels, grid, domain_min=domain_min, domain_max=domain_max
            )
        except Exception as exc:
            raise ValueError(f"cannot probe viable LUT {row.preset_id}") from exc
        d_full = float(deltaE_ciede2000(
            before_lab, rgb2lab(rendered.reshape(-1, 1, 3))
        ).mean())
        if not math.isfinite(d_full):
            raise ValueError(f"non-finite reach for viable LUT {row.preset_id}")
        achievable = [
            strength_bin for strength_bin, (low, _high, _inclusive)
            in GLOBAL_DELTA_E_TARGETS.items() if d_full >= low
        ]
        presets[row.preset_id] = {
            "d_full": round(d_full, 6), "achievable_bins": achievable,
        }
    return {
        "schema": REACH_SCHEMA,
        "sampler_revision": SAMPLER_REVISION,
        "requested_pixels": SAMPLE_PIXELS,
        "sampled_pixels": int(len(pixels)),
        "catalog_sha256": catalog_digest(ordered),
        "preset_count": len(ordered),
        "presets": presets,
    }


def validate_preset_reach(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != REACH_SCHEMA:
        raise ValueError("invalid frozen preset reach schema")
    if value.get("sampler_revision") != SAMPLER_REVISION:
        raise ValueError("invalid frozen preset reach sampler")
    presets = value.get("presets")
    if not isinstance(presets, dict) or int(value.get("preset_count", -1)) != len(presets):
        raise ValueError("invalid frozen preset reach catalog")
    allowed_bins = set(GLOBAL_DELTA_E_TARGETS)
    for preset_id, row in presets.items():
        if not isinstance(preset_id, str) or not isinstance(row, dict):
            raise ValueError("invalid frozen preset reach entry")
        d_full = row.get("d_full")
        bins = row.get("achievable_bins")
        if not isinstance(d_full, (int, float)) or not math.isfinite(float(d_full)) \
                or float(d_full) < 0:
            raise ValueError(f"invalid frozen d_full for {preset_id}")
        if not isinstance(bins, list) or len(bins) != len(set(bins)) \
                or not set(bins).issubset(allowed_bins):
            raise ValueError(f"invalid frozen achievable bins for {preset_id}")
        expected = [name for name, (low, _high, _inclusive)
                    in GLOBAL_DELTA_E_TARGETS.items() if float(d_full) >= low]
        if bins != expected:
            raise ValueError(f"inconsistent frozen achievable bins for {preset_id}")
    return value


__all__ = [
    "REACH_SCHEMA", "SAMPLE_PIXELS", "SAMPLER_REVISION", "catalog_digest",
    "configured_lut_loader", "probe_preset_reach", "sample_source_pixels",
    "validate_preset_reach",
]
