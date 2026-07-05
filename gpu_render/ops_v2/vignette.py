"""Vignette —— thin 适配器：实现与标定常数已合入 image_ops/non_gimp_ops.py。

历史实现/标定说明见 non_gimp_ops.py 的 Vignette 段与 scratch/vignette_v2_*.py。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_lens_vignette, apply_postcrop_vignette


def _vignette_v2(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    out = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    pc = float(attrs.get("PostCropVignetteAmount", 0) or 0)
    if pc != 0.0:
        out = apply_postcrop_vignette(
            out, pc,
            midpoint=float(attrs.get("PostCropVignetteMidpoint", 50) or 50),
            feather=float(attrs.get("PostCropVignetteFeather", 50) or 50),
            roundness=float(attrs.get("PostCropVignetteRoundness", 0) or 0),
            style=int(float(attrs.get("PostCropVignetteStyle", 1) or 1)))
    lens = float(attrs.get("VignetteAmount", 0) or 0)
    if lens != 0.0:
        out = apply_lens_vignette(out, lens,
                                  midpoint=float(attrs.get("VignetteMidpoint", 50) or 50))
    return out


OPS_V2 = {"Vignette": _vignette_v2, "VignetteParams": _vignette_v2}
