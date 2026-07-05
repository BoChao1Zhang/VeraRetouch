"""Dehaze —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

per_value 参数（原 fits/Dehaze.json）已烘焙进 non_gimp_ops._DEHAZE_DEFAULT_PARAMS；
+70/+100 于 2026-07 重拟合（scratch/dehaze_p70_refit.py，新增 kd/hk/acx/wvl 机制）。
已知缺口：+70/+100 仍未达 2.0/4.0 门（全分辨率 2.82/4.08），为传输图逐像素
结构性残差；详见 non_gimp_ops._DEHAZE_DEFAULT_PARAMS 上方注释。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import lr_dehaze


def _op_dehaze(img: np.ndarray, ctx: dict) -> np.ndarray:
    v = float(ctx["attrs"].get("Dehaze", ctx.get("label", "0")))
    return lr_dehaze(np.asarray(img, dtype=np.float32), v)


OPS_V2 = {"Dehaze": _op_dehaze}
