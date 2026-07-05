"""Clarity —— thin 适配器：实现与烘焙标定表已合入 image_ops/non_gimp_ops.py。"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import adjust_clarity_v2


def _clarity_entry(img: np.ndarray, ctx: dict) -> np.ndarray:
    v = float(ctx["label"])
    return adjust_clarity_v2(np.asarray(img, dtype=np.float32), v)


OPS_V2 = {"Clarity": _clarity_entry}
