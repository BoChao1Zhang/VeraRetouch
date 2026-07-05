"""SplitToning / ColorGrade 家族 —— thin 适配器：核心 lr_color_grade 已合入
image_ops/non_gimp_ops.py（常量由 scratch/colorgrade_genmod5.py 生成，改动请改生成器）。

组合标定后校正（teacher-forced stage_pairs(7)）：孤立标定输入是 identity，组合中输入是
上一阶段 GT。诊断发现 lr_color_grade 在 |SplitToningBalance| 偏大时对 split-tone tint
过冲（权重区扩太大）。fits/SplitToneShadow__comp.json 的 balance_alpha 曲线按 |balance|
把 color-grade delta 缩回：out = inp + alpha(|bal|)*(out-inp)。|bal|=0 → alpha=1（不动
已达标的 bal≈0 主流）。文件缺失时 alpha=1（等价原行为，签名/输出不变）。
"""
from __future__ import annotations

import json
import os

import numpy as np

from gpu_render.image_ops.non_gimp_ops import lr_color_grade

_COMP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "fits", "SplitToneShadow__comp.json")
_COMP_CACHE: dict = {}


def _balance_alpha(balance: float) -> float:
    """读组合残差曲线，返回 |balance| 对应的振幅缩放 alpha（默认 1.0 = 不校正）。"""
    if "loaded" not in _COMP_CACHE:
        try:
            _COMP_CACHE["ba"] = json.load(open(_COMP_PATH)).get("balance_alpha")
        except (OSError, ValueError):
            _COMP_CACHE["ba"] = None
        _COMP_CACHE["loaded"] = True
    ba = _COMP_CACHE.get("ba")
    if not ba:
        return 1.0
    return float(np.interp(abs(float(balance)), ba["x"], ba["y"]))


def _cg_apply(img: np.ndarray, ctx: dict) -> np.ndarray:
    at = {k: float(v) for k, v in ctx.get("attrs", {}).items()}
    shadow = (at.get("SplitToningShadowHue", 0.0),
              at.get("SplitToningShadowSaturation", 0.0),
              at.get("ColorGradeShadowLum", 0.0))
    highlight = (at.get("SplitToningHighlightHue", 0.0),
                 at.get("SplitToningHighlightSaturation", 0.0),
                 at.get("ColorGradeHighlightLum", 0.0))
    midtone = (at.get("ColorGradeMidtoneHue", 0.0),
               at.get("ColorGradeMidtoneSat", 0.0),
               at.get("ColorGradeMidtoneLum", 0.0))
    global_ = (at.get("ColorGradeGlobalHue", 0.0),
               at.get("ColorGradeGlobalSat", 0.0),
               at.get("ColorGradeGlobalLum", 0.0))
    balance = at.get("SplitToningBalance", 0.0)
    inp = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    out = lr_color_grade(inp, shadow=shadow, midtone=midtone, highlight=highlight,
                         global_=global_, balance=balance,
                         blending=at.get("ColorGradeBlending", 50.0))
    out = np.asarray(out, dtype=np.float32)
    alpha = _balance_alpha(balance)
    if alpha != 1.0:
        out = np.clip(inp + alpha * (out - inp), 0.0, 1.0)
    return out


OPS_V2 = {op: _cg_apply for op in (
    "SplitToneShadow", "SplitToneHighlight", "SplitToneBalance",
    "ColorGradeMid", "ColorGradeGlobal", "ColorGradeLum", "ColorGradeBlending")}
