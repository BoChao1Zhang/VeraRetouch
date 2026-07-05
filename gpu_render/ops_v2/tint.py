"""IncrementalTint —— 组合标定重写：native adjust_tint + 干净 value_map。

为何不走 apply_scalar_op（标量路径）：
  1. 孤立标定的 post_rgb_lut 以扫描值 {±30,±60,±100} 为键、最近邻快照。组合语境里
     preset 的 tint 多为小值（-9..35），会被快照到满量程 ±30 LUT，产生 ΔE15+ 的满
     量程色偏伪影。
  2. apply_scalar_op 把 value_map 输出交给 _normalize_global_compat_intensity，后者把
     |v|<=1 当作已归一化分数 [-1,1]（×150 tint_units），于是 value_map 输出 0.8（LR
     tint≈+2 时）被当成 0.8 满量程，实测 ΔE20。
本模块直接读 value_map 插值得 local config（[-100,100] 量纲），按 local/100 归一后调用
adjust_tint，规避两坑。replay 的 scalar('Tint') 会先命中 REGISTRY['Tint']（本模块），
不再回退标量路径。value_map 缺失时用内置缺省。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import adjust_tint

_DEFAULT_VM = [[-100, -74], [-35, -26], [-18, -12], [-10, -5], [-8, -4], [-5, -2],
               [0, 0], [5, 2], [8, 4], [10, 5], [18, 12], [35, 26], [100, 74]]


def _tint(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    raw = attrs.get("IncrementalTint", ctx.get("label", 0.0))
    try:
        v = float(str(raw).lstrip("+"))
    except (TypeError, ValueError):
        v = 0.0
    if abs(v) < 1e-9:
        return np.asarray(img, dtype=np.float32)
    vm = (ctx.get("fit") or {}).get("value_map") or _DEFAULT_VM
    xs, ys = zip(*sorted(vm))
    local = float(np.interp(v, xs, ys))
    return np.clip(adjust_tint(np.asarray(img, dtype=np.float32), local / 100.0),
                   0.0, 1.0).astype(np.float32)


OPS_V2 = {"Tint": _tint}
