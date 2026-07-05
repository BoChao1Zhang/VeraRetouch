"""ColorNR / FringeProbe —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

Defringe(Purple/Green)/AutoLateralCA 在 JPEG 重渲染语境实测 ≈ no-op（依据存
fits/FringeProbe.json noop_points），保持恒等。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_color_nr, apply_manual_distortion


def _color_nr(img: np.ndarray, ctx: dict) -> np.ndarray:
    amount = float(ctx.get("attrs", {}).get("ColorNoiseReduction", 0))
    return apply_color_nr(np.asarray(img, dtype=np.float32), amount)


def _fringe_probe(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs", {})
    if "LensManualDistortionAmount" in attrs:
        amount = float(attrs["LensManualDistortionAmount"])
        return apply_manual_distortion(np.asarray(img, dtype=np.float32), amount)
    return img.astype(np.float32, copy=False)


OPS_V2 = {"ColorNR": _color_nr, "FringeProbe": _fringe_probe}
