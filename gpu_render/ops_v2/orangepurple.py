"""HSL Orange/Purple 六算子 —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

拟合参数（原 fits/{Hue,Saturation,Luminance}Adjustment{Orange,Purple}.json）
已烘焙进 non_gimp_ops._ORANGE_PURPLE_FITS。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_hsl_orange_purple_v2


def _make(op_name: str, color: str, kind: str):
    def _fn(img: np.ndarray, ctx: dict) -> np.ndarray:
        lr_v = float(ctx["attrs"].get(op_name, ctx.get("label", 0.0)))
        if lr_v == 0.0:
            return img
        return apply_hsl_orange_purple_v2(
            np.asarray(img, dtype=np.float32), color, **{kind: lr_v})

    _fn.__name__ = f"lr_{op_name.lower()}"
    return _fn


OPS_V2 = {
    f"{pfx}{color}": _make(f"{pfx}{color}", color.lower(), kind)
    for pfx, kind in (("HueAdjustment", "hue"),
                      ("SaturationAdjustment", "sat"),
                      ("LuminanceAdjustment", "lum"))
    for color in ("Orange", "Purple")
}
