"""curve 组 GPU 实现（torch / cuda:1）：ToneCurve(PV2012 点曲线) + Parametric 四滑杆。

复现 ops_v2 的 numpy 标定输出（含 fits 校正），BCHW float32 [0,1] batch 并行。
导出 OPS_GPU = {op_name: fn}；fn(img_bchw_float01, ctx) -> 同形张量（device 沿用输入张量）。

设计（对齐 DESIGN.md）：
- 1D LUT / Δ 曲线 **只依赖 ctx**（控制点/滑杆值），不依赖像素——故在 host(numpy) 端用与
  numpy 参考完全相同的函数构建（保证 parity），再搬到 GPU 对整个 batch 一次施加。
- ToneCurve：Melissa/ProPhoto D50 显式矩阵往返 + master/RGB 自然三次样条 LUT（master_then_rgb 次序）。
- Parametric：标定 Δ 曲线逐通道点曲线语义 + blend<1 时少量 luma 乘性分量。

numpy 参考（parity 对齐目标）：
- ToneCurve  = gpu_render.ops_v2.REGISTRY["ToneCurve"]（ops_v2/tonecurve.py）
- Parametric* = gpu_render.ops_v2.REGISTRY["Parametric*"]（-> non_gimp_ops.apply_parametric_slider）
"""
from __future__ import annotations

import os
# 硬约束：只用 cuda:1。张量显式沿用输入 device（parity 已把张量搬到 MONETGPT_TORCH_DEVICE=cuda:1）。
os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")

import numpy as np
import torch

# —— numpy 参考侧的 LUT/Δ 构建器与常量（只 import 不改，保证与 numpy 参考逐位一致）——
from gpu_render.image_ops.curve import (parse_pv2012_curves, pchip_lut,
                             _M_SRGB2PP, _M_PP2SRGB)
from gpu_render.image_ops.non_gimp_ops import (_parametric_delta_curve, _PARAMETRIC_FITS,
                                    _LUMA709, _EPS)
from gpu_render.ops_v2.tonecurve import _natural_cubic_lut, _DEFAULT_CFG


# ---------------------------------------------------------------------------
# 设备常量缓存（3×3 色彩矩阵 / luma 权重）—— 按 device 缓存 torch buffer
# ---------------------------------------------------------------------------
_CONST_CACHE: dict = {}


def _consts(device):
    c = _CONST_CACHE.get(device)
    if c is None:
        c = {
            "srgb2pp": torch.from_numpy(np.ascontiguousarray(_M_SRGB2PP)).to(device),
            "pp2srgb": torch.from_numpy(np.ascontiguousarray(_M_PP2SRGB)).to(device),
            "luma": torch.from_numpy(np.ascontiguousarray(_LUMA709)).to(device).view(1, 3, 1, 1),
        }
        _CONST_CACHE[device] = c
    return c


# ---------------------------------------------------------------------------
# 逐点数学基元（batched，BCHW）
# ---------------------------------------------------------------------------
def _srgb_decode_t(v: torch.Tensor) -> torch.Tensor:
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _srgb_encode_t(v: torch.Tensor) -> torch.Tensor:
    v = v.clamp(0.0, 1.0)
    return torch.where(v <= 0.0031308, v * 12.92, 1.055 * v ** (1.0 / 2.4) - 0.055)


def _matmul_channels(x: torch.Tensor, m3x3: torch.Tensor) -> torch.Tensor:
    """BCHW 上逐像素 3×3 通道混合：out[:,c]=Σ_i m[c,i]*x[:,i]（等价 numpy 的 x_hwc @ m.T）。"""
    return torch.einsum("ci,bihw->bchw", m3x3, x)


def _apply_lut_uniform(x: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """把均匀 [0,1] 采样的 1D LUT（size 个点）线性插值施加到 x（等价 np.interp(x, linspace(0,1,L), lut)）。"""
    L = lut.shape[0]
    pos = x.clamp(0.0, 1.0) * (L - 1)
    lo = torch.floor(pos).to(torch.long).clamp_(0, L - 1)
    hi = (lo + 1).clamp_(0, L - 1)
    frac = pos - lo.to(pos.dtype)
    return lut[lo] * (1.0 - frac) + lut[hi] * frac


def _interp_np(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """batched np.interp 语义（xp 升序，区间外夹到端点）。用于 Parametric 的非均匀 grid。"""
    n = xp.shape[0]
    xc = x.contiguous()
    idx = torch.searchsorted(xp, xc, right=True).clamp_(1, n - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    res = y0 + (y1 - y0) / (x1 - x0) * (x - x0)
    res = torch.where(x < xp[0], fp[0], res)
    res = torch.where(x > xp[-1], fp[-1], res)
    return res


# ---------------------------------------------------------------------------
# ToneCurve（PV2012 点曲线，Melissa/ProPhoto 空间，master_then_rgb 次序）
# ---------------------------------------------------------------------------
def _lut_np(points, method: str) -> np.ndarray:
    return _natural_cubic_lut(points) if method == "cubic" else pchip_lut(points)


def _tone_curve_gpu(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    curves = parse_pv2012_curves(ctx.get("elements", ""))
    if not any(curves.get(k) for k in ("master", "r", "g", "b")):
        return img
    fit = ctx.get("fit") or {}
    cfg = {**_DEFAULT_CFG, **(fit.get("curve_cfg") or {})}
    space = cfg.get("space", "melissa")
    method = cfg.get("method", "cubic")
    order = cfg.get("order", "master_then_rgb")
    dev = img.device
    C = _consts(dev)

    def lut_t(pts):
        arr = np.ascontiguousarray(_lut_np(pts, method))
        return torch.from_numpy(arr).to(dev)

    if space == "melissa":
        work = _srgb_encode_t(_matmul_channels(_srgb_decode_t(img), C["srgb2pp"]).clamp(0.0, 1.0))
    else:
        work = img.clamp(0.0, 1.0)

    def do_master(w):
        if curves.get("master"):
            w = _apply_lut_uniform(w, lut_t(curves["master"]))  # 同一 LUT 施加到全部 3 通道
        return w

    def do_rgb(w):
        for i, key in enumerate(("r", "g", "b")):
            if curves.get(key):
                lut = lut_t(curves[key])
                w = w.clone()
                w[:, i] = _apply_lut_uniform(w[:, i], lut)
        return w

    work = do_rgb(do_master(work)) if order == "master_then_rgb" else do_master(do_rgb(work))

    if space == "melissa":
        out = _srgb_encode_t(_matmul_channels(_srgb_decode_t(work), C["pp2srgb"]).clamp(0.0, 1.0))
    else:
        out = work.clamp(0.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# Parametric 四滑杆（标定 Δ 曲线，点曲线语义 + luma 乘性分量）
# ---------------------------------------------------------------------------
def _make_parametric_gpu(op_name: str):
    fit = _PARAMETRIC_FITS.get(op_name) or {}

    def fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
        raw = ctx["attrs"].get(op_name, ctx.get("label", "0"))
        v = float(raw)
        gd = _parametric_delta_curve(fit, v)  # (grid, delta, blend)；|v|<eps -> None（恒等）
        if gd is None:
            return img
        grid_np, delta_np, blend = gd
        dev = img.device
        grid = torch.from_numpy(np.ascontiguousarray(grid_np)).to(dev)
        delta = torch.from_numpy(np.ascontiguousarray(delta_np)).to(dev)
        out = img + _interp_np(img, grid, delta)
        if blend < 1.0 - _EPS:
            luma = _consts(dev)["luma"]
            y = (img * luma).sum(dim=1, keepdim=True)                 # (B,1,H,W)
            y2 = (y + _interp_np(y, grid, delta)).clamp(0.0, 1.0)
            lm = img * (y2 / (y + _EPS))
            out = blend * out + (1.0 - blend) * lm
        return out.clamp(0.0, 1.0)

    return fn


OPS_GPU = {"ToneCurve": _tone_curve_gpu}
OPS_GPU.update({f"Parametric{n}": _make_parametric_gpu(f"Parametric{n}")
                for n in ("Shadows", "Darks", "Lights", "Highlights")})
