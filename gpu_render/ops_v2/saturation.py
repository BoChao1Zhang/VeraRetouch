"""Saturation —— sat 阶段组合标定实现。

背景：旧路径 replay.scalar("Saturation") 在 REGISTRY 无 Saturation 时落到
apply_scalar_op -> apply_non_gimp_config({"Saturation": local})。而
non_gimp_ops._normalize_global_compat_intensity 用 |v|>1 才 /100 的启发式判定单位：
value_map 把小 LR 值（如 -1）映射到 |local|<1（-0.583），被误判为归一化 [-1,1] 单位，
经 adjust_saturation 再 ×100 → 实际施加 -58.3% 饱和度（应为 -0.58%），组合回放里
Saturation≈±1 的 preset 因此爆到 ΔE 6~7。isolated 扫描只有 ±30/60/100（|local|>1）
从未触发该 bug。

本模块注册 "Saturation" 进 REGISTRY，使 replay 走 reg 路径，直接以 LR 百分比语义
调用 _apply_saturation_np（与 isolated 标定在大值处逐位等价），彻底消除单位歧义；
并读 fits/Saturation.json(value_map) + 可选 fits/Saturation__comp.json(组合残差后校正)。
只加实现、不改签名、不动 non_gimp_ops。
"""
from __future__ import annotations

import json
import os

import numpy as np

from gpu_render.image_ops.non_gimp_ops import _apply_saturation_np

_HERE = os.path.dirname(os.path.abspath(__file__))
_FITS = os.path.join(os.path.dirname(_HERE), "fits")
_LUMA_W = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
_COMP_CACHE: dict = {}


def _load_json(name: str) -> dict:
    p = os.path.join(_FITS, name)
    if p in _COMP_CACHE:
        return _COMP_CACHE[p]
    d = {}
    if os.path.exists(p):
        try:
            d = json.load(open(p))
        except Exception:  # noqa: BLE001
            d = {}
    _COMP_CACHE[p] = d
    return d


def _sat_pct(lr_v: float, fit: dict) -> float:
    """LR 滑杆值 -> native saturation 百分比（LR 单位，直接进 _apply_saturation_np）。"""
    vm = fit.get("value_map")
    if vm:
        xs, ys = zip(*sorted(vm))
        return float(np.interp(lr_v, xs, ys))
    return float(lr_v)


def _apply_comp(out: np.ndarray, comp: dict) -> np.ndarray:
    """组合残差后校正：luma 残差曲线 + 逐通道 RGB LUT（值无关，全局单表）。"""
    if not comp:
        return out
    pts = comp.get("post_luma_lut")
    if pts:
        xs, ys = zip(*sorted(pts))
        luma = out @ _LUMA_W
        gain = (np.interp(luma, xs, ys).astype(np.float32) - luma)
        out = np.clip(out + gain[..., None], 0.0, 1.0)
    ch = comp.get("post_rgb_lut")
    if ch:
        o = out.copy()
        for i, c in enumerate(("r", "g", "b")):
            p = ch.get(c)
            if p:
                xs, ys = zip(*sorted(p))
                o[..., i] = np.interp(out[..., i], xs, ys).astype(np.float32)
        out = np.clip(o, 0.0, 1.0)
    return out


def _saturation_entry(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    raw = attrs.get("Saturation", ctx.get("label", 0))
    try:
        lr_v = float(str(raw).lstrip("+"))
    except (TypeError, ValueError):
        return np.asarray(img, dtype=np.float32)
    if abs(lr_v) < 1e-8:
        return np.asarray(img, dtype=np.float32)
    fit = ctx.get("fit") or _load_json("Saturation.json")
    pct = _sat_pct(lr_v, fit)
    out = _apply_saturation_np(np.asarray(img, dtype=np.float32), pct)
    return _apply_comp(out, _load_json("Saturation__comp.json"))


OPS_V2 = {"Saturation": _saturation_entry}
