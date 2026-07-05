"""ToneCurve（PV2012 点曲线）—— 组合标定版。

孤立标定时 GT 每次只含一条合成曲线（master 或某个通道单独出现），因此
「master 与 RGB 通道曲线同时出现时的次序」与「插值样条类型」从未被真实数据约束。
组合回放整段 preset 时二者同时出现且曲线远比合成的极端，暴露两处模型错误：

  1) 次序：原实现 RGB 通道曲线 → master。用 teacher-forcing 真实中间态
     (partial_GT_2 → partial_GT_3) 标定发现 LR 实为 master → RGB 通道曲线。
     纠正后 active 对 med ΔE 1.106→0.98、max 7.32→4.4。
  2) 插值：原用单调 PCHIP。LR 点曲线用可 overshoot 的三次样条（自然边界），
     极端曲线上差异显著。改自然三次后 med 0.98→0.69、mean 1.06→0.79、p90 1.58→0.95。

作用空间沿用 Melissa RGB（ProPhoto 基色 D50 + sRGB 传递函数）——实测 sRGB 空间显著更差
（med 1.5+），确认 Melissa 正确。组合标定后 stage_pairs(3) 全部 active 对（66 条）：
  med=0.69 mean=0.79 p90=0.95（仅 1 条 >2，为单图蓝道内容特例，非模型误差）。

配置从 ctx["fit"]（fits/ToneCurve.json）读取，缺省即上面标定出的最优组合，replay 自动生效。
签名不变：fn(img_float01_rgb, ctx) -> img_float01_rgb。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.curve import (parse_pv2012_curves, pchip_lut,
                             _srgb_decode, _srgb_encode, _apply_lut,
                             _M_SRGB2PP, _M_PP2SRGB)

# 组合标定出的最优前向模型（见模块 docstring）。
_DEFAULT_CFG = {"space": "melissa", "method": "cubic", "order": "master_then_rgb"}


def _natural_cubic_lut(points, size: int = 4096) -> np.ndarray:
    """显示域 0-255 控制点 -> float32 LUT（域/值 [0,1]，size 个等距采样）。

    自然三次样条（端点二阶导=0），可 overshoot——与 LR 点曲线一致；首/末控制点外
    水平延伸；末端 clip 到 [0,1]。控制点 <2 退化为恒等。
    """
    dedup: dict = {}
    for px, py in points:
        dedup[float(px)] = float(py)
    pts = sorted(dedup.items())
    if len(pts) < 2:
        return np.linspace(0.0, 1.0, size, dtype=np.float32)
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    n = x.size
    if n == 2:  # 两点：线性，自然样条即直线
        xq = np.linspace(0.0, 255.0, size)
        vals = np.interp(xq, x, y)
        vals[xq <= x[0]] = y[0]
        vals[xq >= x[-1]] = y[-1]
        return np.clip(vals / 255.0, 0.0, 1.0).astype(np.float32)
    h = np.diff(x)
    A = np.zeros((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    A[0, 0] = 1.0
    A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    c = np.linalg.solve(A, b)                      # 二阶导 /2 的系数
    xq = np.linspace(0.0, 255.0, size)
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, n - 2)
    dx = xq - x[idx]
    bcoef = (y[idx + 1] - y[idx]) / h[idx] - h[idx] * (2.0 * c[idx] + c[idx + 1]) / 3.0
    dcoef = (c[idx + 1] - c[idx]) / (3.0 * h[idx])
    vals = y[idx] + bcoef * dx + c[idx] * dx ** 2 + dcoef * dx ** 3
    vals[xq <= x[0]] = y[0]
    vals[xq >= x[-1]] = y[-1]
    return np.clip(vals / 255.0, 0.0, 1.0).astype(np.float32)


def _lut(points, method: str) -> np.ndarray:
    return _natural_cubic_lut(points) if method == "cubic" else pchip_lut(points)


def _apply(img: np.ndarray, curves: dict, cfg: dict) -> np.ndarray:
    if not any(curves.get(k) for k in ("master", "r", "g", "b")):
        return np.asarray(img, dtype=np.float32)
    space = cfg.get("space", "melissa")
    method = cfg.get("method", "cubic")
    order = cfg.get("order", "master_then_rgb")
    img = np.asarray(img, dtype=np.float32)
    if space == "melissa":
        work = _srgb_encode(np.clip(_srgb_decode(img) @ _M_SRGB2PP.T, 0.0, 1.0))
    else:
        work = np.clip(img, 0.0, 1.0).copy()

    def do_master(w):
        if curves.get("master"):
            lut = _lut(curves["master"], method)
            for i in range(3):
                w[..., i] = _apply_lut(w[..., i], lut)
        return w

    def do_rgb(w):
        for i, key in enumerate(("r", "g", "b")):
            if curves.get(key):
                w[..., i] = _apply_lut(w[..., i], _lut(curves[key], method))
        return w

    work = do_rgb(do_master(work)) if order == "master_then_rgb" else do_master(do_rgb(work))
    if space == "melissa":
        out = _srgb_encode(np.clip(_srgb_decode(work) @ _M_PP2SRGB.T, 0.0, 1.0))
    else:
        out = np.clip(work, 0.0, 1.0)
    return out.astype(np.float32)


def _tone_curve_v2(img: np.ndarray, ctx: dict) -> np.ndarray:
    fit = ctx.get("fit") or {}
    cfg = {**_DEFAULT_CFG, **(fit.get("curve_cfg") or {})}
    curves = parse_pv2012_curves(ctx.get("elements", ""))
    return _apply(np.asarray(img, dtype=np.float32), curves, cfg)


OPS_V2 = {"ToneCurve": _tone_curve_v2}
