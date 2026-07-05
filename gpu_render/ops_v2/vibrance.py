"""Vibrance —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

达标参数（原 fits/Vibrance.json v2_params）已烘焙进 non_gimp_ops._VIB_DEFAULTS。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_vibrance_lr


def _vibrance_entry(img: np.ndarray, ctx: dict) -> np.ndarray:
    pct = float(ctx["attrs"].get("Vibrance", ctx["label"]))
    return apply_vibrance_lr(np.asarray(img, dtype=np.float32), pct)


OPS_V2 = {"Vibrance": _vibrance_entry}
