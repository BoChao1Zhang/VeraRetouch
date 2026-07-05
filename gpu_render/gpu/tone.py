"""GPU (torch, cuda:1) 复现 ops_v2 的 tone 面板算子 —— 批处理并行渲染。

复现目标 = numpy 标定参考（含 fits 校正）：
  - Exposure/Contrast/Highlights/Whites/Blacks 走 `replay.apply_scalar_op`
    （value_map → op → post_luma_lut 相邻值混合 → post_rgb_lut 最近值快照）。
  - Shadows 走 `ops_v2.REGISTRY["Shadows"]`（图像自适应乘性 luma 曲线）。

张量约定：BCHW float32 [0,1] RGB on cuda:1（支持 batch 维，这是并行的关键）。
导出 OPS_GPU = {op_name: fn}，fn(img_bchw, ctx) -> 同形张量。ctx 与 ops_v2 一致
  {"label","attrs","elements","fit"}。scalar 值从 ctx["attrs"][crs] 或 ctx["label"] 取，
  fits 从 ctx["fit"] 取。标量参数逐 ctx 解析、广播到整个 batch。

parity: 见 tools/lr_calib/gpu/scratch/tone_parity.py（对每算子多代表性 ctx 验 de_mean<0.5）。
"""
from __future__ import annotations

import math
import os

os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")

import numpy as np
import torch
import torch.nn.functional as F

# 只读 import（host 端参数/曲线预计算，保证与 numpy 标定逐位一致；不修改本体）
from gpu_render.replay import _lut_at
from gpu_render.sweeps import OPS, fmt_value
from gpu_render.local_apply import local_value as _np_local_value
from gpu_render.image_ops.non_gimp_ops import _SHADOWS_V2_FIT, _shadows_delta_curves
from gpu_render.image_ops.curve import build_blacks_lut

DEVICE = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1")

_W_LUMA_LR = (0.2126, 0.7152, 0.0722)   # post_luma_lut / shadows_v2 用（Rec.709）
_W_LUMA_601 = (0.299, 0.587, 0.114)     # gegl tone core（Highlights/Whites）用

_CRS = {
    "Exposure": "Exposure2012", "Contrast": "Contrast2012",
    "Highlights": "Highlights2012", "Whites": "Whites2012",
    "Blacks": "Blacks2012", "Shadows": "Shadows2012",
}


# ---------------------------------------------------------------------------
# 基础 torch 工具
# ---------------------------------------------------------------------------
def _luma(img: torch.Tensor, w) -> torch.Tensor:
    """(B,3,H,W) -> (B,1,H,W) 加权亮度。"""
    return (img[:, 0] * w[0] + img[:, 1] * w[1] + img[:, 2] * w[2]).unsqueeze(1)


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp 语义（xp 升序、区间外钳到端点），xp/fp 为共享 1D。"""
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp_(1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = (x - x0) / (x1 - x0).clamp_min(1e-12)
    y = y0 + t * (y1 - y0)
    y = torch.where(x <= xp[0], fp[0], y)
    y = torch.where(x >= xp[-1], fp[-1], y)
    return y


def _interp1d_batched(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """xp 共享 1D (K,)、fp 逐样本 (B,K)、x (B,1,H,W)；np.interp 语义。"""
    B = x.shape[0]
    K = xp.numel()
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp_(1, K - 1)  # (B,1,H,W)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    i0 = (idx - 1).reshape(B, -1)
    i1 = idx.reshape(B, -1)
    y0 = torch.gather(fp, 1, i0).reshape_as(x)
    y1 = torch.gather(fp, 1, i1).reshape_as(x)
    t = (x - x0) / (x1 - x0).clamp_min(1e-12)
    y = y0 + t * (y1 - y0)
    lo = fp[:, 0].view(B, 1, 1, 1)
    hi = fp[:, -1].view(B, 1, 1, 1)
    y = torch.where(x <= xp[0], lo, y)
    y = torch.where(x >= xp[-1], hi, y)
    return y


def _smoothstep(x: torch.Tensor, e0: float, e1: float) -> torch.Tensor:
    t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# cv2 忠实双边（Highlights base；下采样 //4 → 双边 → 上采样）
# ---------------------------------------------------------------------------
_BILAT_CACHE: dict = {}


def _bilat_offsets(sigma_space: float):
    """复现 cv2 d<=0 时的圆形支撑 + 高斯空间权重。"""
    key = round(sigma_space, 4)
    if key in _BILAT_CACHE:
        return _BILAT_CACHE[key]
    radius = int(round(sigma_space * 1.5))
    radius = max(radius, 1)
    gsc = -0.5 / (sigma_space * sigma_space)
    offs = []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            rr = i * i + j * j
            if rr > radius * radius:
                continue
            offs.append((i, j, math.exp(rr * gsc)))
    _BILAT_CACHE[key] = (radius, offs)
    return radius, offs


def _bilateral_cv2(light: torch.Tensor, sigma_color: float, sigma_space: float) -> torch.Tensor:
    """cv2.bilateralFilter(d=0, sigmaColor, sigmaSpace) 忠实复现，单通道 (B,1,H,W)。"""
    radius, offs = _bilat_offsets(sigma_space)
    gcc = -0.5 / (sigma_color * sigma_color)
    H, W = light.shape[-2:]
    pad = min(radius, H - 1, W - 1)
    lp = F.pad(light, (pad, pad, pad, pad), mode="reflect")  # reflect = BORDER_REFLECT_101
    acc = torch.zeros_like(light)
    wsum = torch.zeros_like(light)
    for i, j, sw in offs:
        ii = pad + i
        jj = pad + j
        if ii < 0 or jj < 0 or ii + H > lp.shape[-2] or jj + W > lp.shape[-1]:
            continue  # 半径被 pad 上限截断时跳过越界抽头
        neigh = lp[..., ii:ii + H, jj:jj + W]
        d = neigh - light
        w = sw * torch.exp(d * d * gcc)
        acc += neigh * w
        wsum += w
    return acc / wsum.clamp_min(1e-12)


def _bilateral_base_from_luma(light: torch.Tensor, radius: float = 100.0) -> torch.Tensor:
    """_bilateral_base_from_luma_np 的 torch 版：下采样 //4 → 双边 → 上采样。"""
    H, W = light.shape[-2:]
    sh = max(1, H // 4)
    sw = max(1, W // 4)
    small = F.interpolate(light, size=(sh, sw), mode="area")
    sigma_space = max(2.0, float(radius) * 0.12)
    base_small = _bilateral_cv2(small, sigma_color=0.08, sigma_space=sigma_space)
    return F.interpolate(base_small, size=(H, W), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# host 端标量参数（复现 apply_scalar_op：value_map + post-LUT 预计算）
# ---------------------------------------------------------------------------
def _lr_value(ctx: dict, op: str) -> float:
    attrs = ctx.get("attrs") or {}
    raw = attrs.get(_CRS[op], ctx.get("label", 0))
    try:
        return float(str(raw).lstrip("+"))
    except (TypeError, ValueError):
        return 0.0


def _local_value(op: str, lr_v: float, fit: dict) -> float:
    # 直接复用 numpy 版（value_map 插值；无 value_map 时回退 baseline_local_config），
    # 保证 host 端 LR值→local值 与标定参考逐位一致。
    return float(_np_local_value(op, lr_v, fit))


def _normalize_compat(v: float) -> float:
    return v / 100.0 if abs(v) > 1.0 else v


def _post_luma_pts(op: str, lr_v: float, fit: dict):
    vals = [float(v) for v in OPS[op]["values"]]
    pts = _lut_at(fit.get("post_luma_lut") or {}, vals, lr_v)
    if not pts:
        return None
    xs, ys = zip(*sorted(pts))
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def _post_rgb_pts(op: str, lr_v: float, fit: dict):
    rgb_luts = fit.get("post_rgb_lut") or {}
    vals = [float(v) for v in OPS[op]["values"]]
    if not (rgb_luts and vals):
        return None
    nearest = min(vals, key=lambda g: abs(g - lr_v))
    chl = (rgb_luts.get(fmt_value(nearest))
           or rgb_luts.get(str(int(nearest)) if float(nearest).is_integer() else str(nearest)))
    if not chl or abs(nearest - lr_v) > (max(vals) - min(vals)):
        return None
    out = {}
    for ch in ("r", "g", "b"):
        p = chl.get(ch)
        if p:
            xs, ys = zip(*sorted(p))
            out[ch] = (np.asarray(xs, np.float32), np.asarray(ys, np.float32))
    return out or None


def _apply_post_luma(out: torch.Tensor, pts) -> torch.Tensor:
    if pts is None:
        return out
    xs = torch.as_tensor(pts[0], device=out.device)
    ys = torch.as_tensor(pts[1], device=out.device)
    luma = _luma(out, _W_LUMA_LR)
    gain = _interp1d(luma, xs, ys) - luma
    return (out + gain).clamp(0.0, 1.0)


def _apply_post_rgb(out: torch.Tensor, luts) -> torch.Tensor:
    if luts is None:
        return out
    o = out.clone()
    for i, ch in enumerate(("r", "g", "b")):
        if ch in luts:
            xs = torch.as_tensor(luts[ch][0], device=out.device)
            ys = torch.as_tensor(luts[ch][1], device=out.device)
            o[:, i] = _interp1d(out[:, i], xs, ys)
    return o.clamp(0.0, 1.0)


def _scalar_post(out: torch.Tensor, op: str, lr_v: float, fit: dict) -> torch.Tensor:
    out = _apply_post_luma(out, _post_luma_pts(op, lr_v, fit))
    out = _apply_post_rgb(out, _post_rgb_pts(op, lr_v, fit))
    return out


# ---------------------------------------------------------------------------
# 算子核（逐点 / 空间），全部 batched，param 广播
# ---------------------------------------------------------------------------
def _exposure_core(img: torch.Tensor, ev: float) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    white = float(2.0 ** (-ev))
    gain = 1.0 / max(white, 1e-6)                       # black_level = 0
    adjusted = (img * gain).clamp_min(0.0)
    pivot = 0.18
    gamma = 1.0 / (1.0 + 0.25 * ev) if ev >= 0.0 else 1.0 - 0.20 * ev
    normalized = (adjusted / max(pivot, 1e-6)).clamp_min(0.0)
    out = pivot * normalized.pow(gamma)
    return out.clamp(0.0, 1.0)


def _contrast_core(img: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    img = img.clamp(0.0, 1.0)
    normalized = contrast_factor - 1.0
    if normalized >= 0.0:
        slope = 2.0 + 6.0 * min(max(normalized, 0.0), 1.0)
        raw = torch.sigmoid(slope * (img - 0.5))
        lo = 1.0 / (1.0 + math.exp(slope * 0.5))        # sigmoid(-slope*0.5)
        hi = 1.0 / (1.0 + math.exp(-slope * 0.5))       # sigmoid(slope*0.5)
        out = (raw - lo) / max(hi - lo, 1e-6)
    else:
        flatten = max(0.05, 1.0 + min(max(normalized, -0.95), 0.0))
        out = (img - 0.5) * flatten + 0.5
    return out.clamp(0.0, 1.0)


def _highlights_core(img: torch.Tensor, highlights: float) -> torch.Tensor:
    """gegl tone core（highlights-only, response_scale=1, ccorrect=0.5→无饱和校正）。"""
    l = _luma(img, _W_LUMA_601)
    base = _bilateral_base_from_luma(l, radius=100.0)
    tb0 = (1.0 - base).clamp(0.0, 1.0)
    c = 0.5
    eps = 1e-8
    ta = l
    h_scaled = 2.0 * min(max(highlights, -1.0), 1.0) * 1.0
    if h_scaled != 0.0:
        hx = (1.0 - tb0 / (1.0 - c + eps)).clamp(0.0, 1.0)
        highlights2 = h_scaled * h_scaled
        sign_neg = -1.0 if h_scaled > 0.0 else 1.0
        while highlights2 > 0.0:
            la = ta
            lb = ((tb0 - 0.5) * sign_neg * torch.sign(1.0 - la) + 0.5).clamp(0.0, 1.0)
            chunk = 1.0 if highlights2 > 1.0 else highlights2
            optrans = chunk * hx
            highlights2 -= 1.0
            mapped = torch.where(la > 0.5,
                                 1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb),
                                 2.0 * la * lb)
            ta = (la * (1.0 - optrans) + mapped * optrans).clamp(0.0, 1.0)
    ratio = ta / (l + 1e-8)
    return (img * ratio).clamp(0.0, 1.0)


def _whites_core(img: torch.Tensor, whites: float) -> torch.Tensor:
    """gegl tone core（whites-only；纯逐点，无空间 base）。whites_gain=0.9。"""
    w_val = min(max(whites, -1.0), 1.0) * 0.9
    l = _luma(img, _W_LUMA_601)
    mask = _smoothstep(l, 0.72, 0.98)
    ta = (l + 0.35 * min(max(w_val, -1.0), 1.0) * mask).clamp(0.0, 1.0)
    ratio = ta / (l + 1e-8)
    return (img * ratio).clamp(0.0, 1.0)


def _blacks_core(img: torch.Tensor, amount: float) -> torch.Tensor:
    """adjust_blacks：HSV-V 通道 1D LUT，等价 rgb *= LUT(max)/max。"""
    img = img.clamp(0.0, 1.0)
    lut = build_blacks_lut(amount)                      # 256-entry uint8, host
    xs = torch.linspace(0.0, 1.0, lut.shape[0], device=img.device)
    ys = torch.as_tensor(lut.astype(np.float32) / 255.0, device=img.device)
    V = img.amax(dim=1, keepdim=True)                   # cv2 HSV V = max(rgb)
    Vp = _interp1d(V, xs, ys).clamp(0.0, 1.0)
    scale = torch.where(V > 1e-8, Vp / V.clamp_min(1e-8), torch.ones_like(V))
    return (img * scale).clamp(0.0, 1.0)


def _shadows_core(img: torch.Tensor, v: float) -> torch.Tensor:
    """_apply_shadows_v2：图像自适应乘性 luma 曲线（逐样本 mean）。"""
    if abs(v) < 1e-6:
        return img.clamp(0.0, 1.0)
    centers, a, b = _shadows_delta_curves(_SHADOWS_V2_FIT, v)   # host, float64
    dev = img.device
    centers_t = torch.as_tensor(centers, dtype=torch.float32, device=dev)     # (K,)
    a_t = torch.as_tensor(a, dtype=torch.float32, device=dev)
    b_t = torch.as_tensor(b, dtype=torch.float32, device=dev)
    K = centers_t.numel()
    B = img.shape[0]
    luma = _luma(img, _W_LUMA_LR)                       # (B,1,H,W)
    m = luma.mean(dim=(2, 3), keepdim=True).view(B, 1)  # (B,1) 逐样本均值
    tone_pre = centers_t.view(1, K) + a_t.view(1, K) + b_t.view(1, K) * m      # (B,K)
    tone = torch.cummax(tone_pre, dim=1)[0]             # 单调守卫
    fp = tone - centers_t.view(1, K)                    # (B,K)
    delta = _interp1d_batched(luma, centers_t, fp)      # (B,1,H,W)
    ratio = (luma + delta) / luma.clamp_min(1e-4)
    return (img * ratio).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# OPS_GPU 入口（fn(img_bchw, ctx) -> 同形张量）
# ---------------------------------------------------------------------------
def _exposure(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    fit = ctx.get("fit") or {}
    lr_v = _lr_value(ctx, "Exposure")
    local = _normalize_compat(_local_value("Exposure", lr_v, fit))
    ev = min(max(local, -1.0), 1.0) * 5.0
    out = img if abs(ev) < 1e-8 else _exposure_core(img, ev)
    return _scalar_post(out.clamp(0.0, 1.0), "Exposure", lr_v, fit)


def _contrast(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    fit = ctx.get("fit") or {}
    lr_v = _lr_value(ctx, "Contrast")
    local = _normalize_compat(_local_value("Contrast", lr_v, fit))
    cf = 1.0 + min(max(local, -1.0), 1.0)
    out = img if abs(cf - 1.0) < 1e-8 else _contrast_core(img, cf)
    return _scalar_post(out.clamp(0.0, 1.0), "Contrast", lr_v, fit)


def _highlights(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    fit = ctx.get("fit") or {}
    lr_v = _lr_value(ctx, "Highlights")
    local = _normalize_compat(_local_value("Highlights", lr_v, fit))
    hl = min(max(local, -1.0), 1.0)
    out = img if abs(hl) < 1e-8 else _highlights_core(img, hl)
    return _scalar_post(out.clamp(0.0, 1.0), "Highlights", lr_v, fit)


def _whites(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    fit = ctx.get("fit") or {}
    lr_v = _lr_value(ctx, "Whites")
    local = _normalize_compat(_local_value("Whites", lr_v, fit))
    w = min(max(local, -1.0), 1.0)
    out = img if abs(w) < 1e-8 else _whites_core(img, w)
    return _scalar_post(out.clamp(0.0, 1.0), "Whites", lr_v, fit)


def _blacks(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    fit = ctx.get("fit") or {}
    lr_v = _lr_value(ctx, "Blacks")
    local = _normalize_compat(_local_value("Blacks", lr_v, fit))
    amt = min(max(local, -1.0), 1.0)
    out = img if abs(amt) < 1e-8 else _blacks_core(img, amt)
    return _scalar_post(out.clamp(0.0, 1.0), "Blacks", lr_v, fit)


def _shadows(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    attrs = ctx.get("attrs") or {}
    v = float(attrs.get("Shadows2012", ctx.get("label", 0)))
    return _shadows_core(img, v)


OPS_GPU = {
    "Exposure": _exposure,
    "Contrast": _contrast,
    "Highlights": _highlights,
    "Whites": _whites,
    "Blacks": _blacks,
    "Shadows": _shadows,
}
