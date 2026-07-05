"""Temperature / AbsTemperature —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

wb_cat_v3 模型参数（原 fits/Temperature.json）已烘焙为 non_gimp_ops 模块常量，
apply_lr_temperature 缺省即用烘焙模型。

AbsTemperature（WhiteBalance=Custom + 绝对 kelvin Temperature/Tint）：孤立标定时对
单张 JPEG 探针实测近 no-op，但组合标定（LR 农场 partial_GT_0 vs identity）显示它在
真实 preset 群里是显著白平衡（中位 ΔE4.5）——因为 identity 渲染用的是各图 as-shot WB，
preset 的绝对 Temperature 把它拉到目标色温。fits/AbsTemperature.json 里的线性模型
（it/ti = slope*绝对值 + intercept，以隐含参考 WB≈5029K 为原点）把绝对 kelvin/tint
映到等效 IncrementalTemperature/Tint，经 apply_lr_temperature + adjust_tint 复现。
模型缺省（fit 无 it_slope）时保持历史 no-op，签名不变。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import adjust_tint, apply_lr_abs_temperature, apply_lr_temperature


def _temperature(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    v = float(attrs.get("IncrementalTemperature", ctx.get("label", 0.0)))
    return apply_lr_temperature(np.asarray(img, dtype=np.float64), v)


def _fnum(attrs: dict, key: str) -> float:
    try:
        return float(str(attrs.get(key, "")).lstrip("+"))
    except (TypeError, ValueError):
        return 0.0


def _abs_temperature(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    fit = ctx.get("fit") or {}
    out = np.asarray(img, dtype=np.float32)
    # 未标定 → 保持历史 no-op（回归安全）
    if "it_slope" not in fit:
        return apply_lr_abs_temperature(out)
    temp_k = _fnum(attrs, "Temperature")
    tint_v = _fnum(attrs, "Tint")
    if temp_k <= 0 and tint_v == 0:
        return out
    it_clip = float(fit.get("it_clip", 60))
    ti_clip = float(fit.get("ti_clip", 30))
    if temp_k > 0:
        it = float(np.clip(fit["it_slope"] * temp_k + fit["it_intercept"], -it_clip, it_clip))
        if abs(it) >= 0.5:
            out = np.clip(apply_lr_temperature(out.astype(np.float64), it), 0.0, 1.0).astype(np.float32)
    # 绝对 tint（含小的系统截距，标定所得）：仅在有绝对 WB 时应用
    ti = float(np.clip(fit["ti_slope"] * tint_v + fit["ti_intercept"], -ti_clip, ti_clip))
    if abs(ti) >= 0.5:
        out = np.clip(adjust_tint(out, ti / 100.0), 0.0, 1.0).astype(np.float32)
    return out


OPS_V2 = {"Temperature": _temperature, "AbsTemperature": _abs_temperature}
