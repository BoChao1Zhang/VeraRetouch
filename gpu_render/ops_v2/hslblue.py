"""HSL Blue 三算子 —— thin 适配器：实现与响应表已合入 image_ops/non_gimp_ops.py。

响应表再生成请用 scratch/hueadjustmentblue_v3_fit.py（勿手改 non_gimp_ops 数值）。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_hsl_blue_v2


def _ctx_value(op: str, ctx: dict) -> float:
    attrs = ctx.get("attrs") or {}
    if op in attrs:
        return float(attrs[op])
    return float(ctx.get("label", 0.0))


def _mk(op: str, kw: str):
    def fn(img: np.ndarray, ctx: dict) -> np.ndarray:
        return apply_hsl_blue_v2(np.asarray(img, dtype=np.float32), **{kw: _ctx_value(op, ctx)})
    fn.__name__ = f"apply_{op}"
    return fn


OPS_V2 = {
    "HueAdjustmentBlue": _mk("HueAdjustmentBlue", "hue"),
    "SaturationAdjustmentBlue": _mk("SaturationAdjustmentBlue", "sat"),
    "LuminanceAdjustmentBlue": _mk("LuminanceAdjustmentBlue", "lum"),
}
