"""Camera Calibration 面板 —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

锚点矩阵（原 fits/Calib*.json）已烘焙进 non_gimp_ops._CALIB_FITS；
ShadowTint 对 JPEG 源实测为恒等（矩阵=单位阵），保留参数入口。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_lr_calibration_panel

_CALIB_OPS = ("CalibRedHue", "CalibRedSaturation", "CalibGreenHue", "CalibGreenSaturation",
              "CalibBlueHue", "CalibBlueSaturation", "CalibShadowTint")


def _make(op: str):
    def fn(img: np.ndarray, ctx: dict) -> np.ndarray:
        sliders = dict(ctx.get("attrs") or {})
        return apply_lr_calibration_panel(np.asarray(img, dtype=np.float32), sliders)
    fn.__name__ = f"calib_{op.lower()}"
    return fn


OPS_V2 = {op: _make(op) for op in _CALIB_OPS}
