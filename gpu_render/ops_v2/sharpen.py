"""Sharpness / SharpenParams —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

组合标定（组合语境残差后校正）：
  孤立标定在 identity 输入（干净 GT）上对齐 LR 的锐化增益。组合回放时输入是农场
  partial_GT（多阶段 JPEG 后的中间态），高频内容已被压缩衰减，而 target 同样是 JPEG 态，
  故我们全强度锐化会加入 target 里没有的清晰高频 → 过冲（stage 9 detail：38/62 对
  full>id，active median before 1.30，p90 2.16）。amount_scale 线性压低锐化强度即可
  （见 scratch/detail_decomp.py：0.4 处 median 1.06 / p90 1.35 / 过冲对降到 ~11）。
  系数存 fits/Sharpness__comp.json。
"""
from __future__ import annotations

import json
import os

import numpy as np

from gpu_render.image_ops.non_gimp_ops import lr_sharpen
from gpu_render.local_apply import FITS_DIR

_COMP_PATH = os.path.join(FITS_DIR, "Sharpness__comp.json")


def _load_amount_scale() -> float:
    try:
        with open(_COMP_PATH) as f:
            return float(json.load(f).get("amount_scale", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


_AMOUNT_SCALE = _load_amount_scale()


def _attr_f(attrs: dict, key: str, default: float) -> float:
    v = attrs.get(key, default)
    try:
        return float(str(v).replace("+", ""))
    except ValueError:
        return default


def _apply(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    return lr_sharpen(
        np.asarray(img, dtype=np.float32),
        amount=_attr_f(attrs, "Sharpness", 0.0) * _AMOUNT_SCALE,
        radius=_attr_f(attrs, "SharpenRadius", 1.0),
        detail=_attr_f(attrs, "SharpenDetail", 25.0),
        masking=_attr_f(attrs, "SharpenEdgeMasking", 0.0),
    )


OPS_V2 = {"Sharpness": _apply, "SharpenParams": _apply}
