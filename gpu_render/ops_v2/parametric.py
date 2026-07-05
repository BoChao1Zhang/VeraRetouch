"""参数曲线四滑杆 —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

标定曲线（原 fits/Parametric*.json）已烘焙进 non_gimp_ops._PARAMETRIC_FITS。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_lr_parametric


def _make(op_name: str):
    def fn(img: np.ndarray, ctx: dict) -> np.ndarray:
        raw = ctx["attrs"].get(op_name, ctx.get("label", "0"))
        return apply_lr_parametric(np.asarray(img, dtype=np.float32), op_name, float(raw))
    return fn


OPS_V2 = {f"Parametric{n}": _make(f"Parametric{n}")
          for n in ("Shadows", "Darks", "Lights", "Highlights")}
