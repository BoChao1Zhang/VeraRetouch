"""Shadows —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

拟合曲线（原 fits/Shadows.json["shadows_v2"]）已烘焙进 non_gimp_ops._SHADOWS_V2_FIT。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import _apply_shadows_v2


def _shadows_entry(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    v = float(attrs.get("Shadows2012", ctx["label"]))
    return _apply_shadows_v2(np.clip(img, 0.0, 1.0).astype(np.float32), v)


OPS_V2 = {"Shadows": _shadows_entry}
