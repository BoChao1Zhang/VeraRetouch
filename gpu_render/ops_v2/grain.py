"""Grain —— thin 适配器：实现与标定常数已合入 image_ops/non_gimp_ops.py。

组合标定（组合语境残差后校正）：
  孤立标定时按 identity 输入匹配 LR 的一次颗粒实现（幅度对齐视觉量级）。但组合回放
  用农场 partial_GT 逐像素比 ΔE，而颗粒是随机场——我们的 PRNG 种子与 LR 农场不同，
  两次独立实现之间的逐像素 ΔE 恒 >0 且随幅度单调上升（stage 10 active median
  before_dE=2.95，signal 仅 0.78，即颗粒把误差放大 ~4x）。ΔE 最优是把颗粒场线性衰减
  （见 scratch/grain_sweep.py：scale 越小 ΔE 越低）。取 field_scale 在保留可见颗粒与
  <2 达标间折中：blended = input + s*(grain_out - input)。系数存 fits/Grain__comp.json。
"""
from __future__ import annotations

import json
import os

import numpy as np

from gpu_render.image_ops.non_gimp_ops import apply_grain
from gpu_render.local_apply import FITS_DIR

_COMP_PATH = os.path.join(FITS_DIR, "Grain__comp.json")


def _load_field_scale() -> float:
    try:
        with open(_COMP_PATH) as f:
            return float(json.load(f).get("field_scale", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


_FIELD_SCALE = _load_field_scale()


def _grain_v2(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    src = np.asarray(img, dtype=np.float32)
    out = apply_grain(src,
                      amount=float(attrs.get("GrainAmount", 0)),
                      size=float(attrs.get("GrainSize", 25)),
                      frequency=float(attrs.get("GrainFrequency", 50)))
    s = _FIELD_SCALE
    if s != 1.0:
        out = np.clip(src + s * (np.asarray(out, dtype=np.float32) - src), 0.0, 1.0)
    return out


OPS_V2 = {"Grain": _grain_v2}
