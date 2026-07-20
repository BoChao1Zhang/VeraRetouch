"""Compatibility entry point for the canonical geometry rasterizer.

Production mask planning lives in :mod:`construct.canonical_masks`.  This module
retains the historical ``cgt_raster`` name for GPU parity checks only; it does not
construct sample plans, render candidates, access PostgreSQL, or expose a CLI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .canonical_masks import raster_geometry


def cgt_raster(mask_type: str, geom: dict[str, Any], h: int, w: int) -> np.ndarray:
    """Rasterize canonical geometry on CPU for parity tests."""
    return raster_geometry(mask_type, geom, h, w)


__all__ = ["cgt_raster"]
