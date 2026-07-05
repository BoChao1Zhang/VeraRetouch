"""GPU (torch, cuda:1) 实现 —— finish 组: Grain, Dehaze。

复现 tools/lr_calib/ops_v2 的 numpy 标定输出（含 fits 校正），支持 BCHW batch 并行。
导出 OPS_GPU = {op_name: fn}；fn(img_BCHW_float01_cuda, ctx) -> 同形张量。

对齐来源（单一真相，直接 import 标定常数/参数，绝不复制副本）:
  image_ops.non_gimp_ops.apply_grain / lr_dehaze 的实现与常数
  fits/Grain__comp.json (field_scale) / fits/Dehaze.json (per_value 已烘焙进 non_gimp_ops)

设计要点:
- Grain 是伪随机场。孤立测试证明两次独立 PRNG 实现的逐像素 ΔE=0.7~1.9 (>0.5 门)，
  故白噪声底 n 必须复现 numpy PCG64（用 numpy default_rng(seed).standard_normal 生成、
  上传 cuda:1）。空间/色调处理（高通高斯、归一、亮度权重、blend）全部在 GPU 上 batch 化。
  种子 = 图像内容 crc32（与 non_gimp_ops._grain_seed 逐字节一致）。
- Dehaze 纯确定性：暗通道 + 引导滤波传输图。box(cv2 BORDER_REFLECT) 用 symmetric-pad +
  double 积分图精确复现；引导滤波下采样用 area、上采样用 bilinear(align_corners=False)；
  erode 用 -max_pool2d(-x)。全部 batch 化。
"""
from __future__ import annotations

import os

os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")

import json
import zlib

import numpy as np
import torch
import torch.nn.functional as F

from gpu_render.image_ops.non_gimp_ops import (
    _GRAIN_AMP_X, _GRAIN_AMP_Y, _GRAIN_HIGH_W, _GRAIN_HIGH_Y0, _GRAIN_HIGH_Y1,
    _GRAIN_HP_MIX, _GRAIN_HP_SIGMA, _GRAIN_RES_K, _GRAIN_SHADOW_W,
    _GRAIN_SHADOW_Y0, _GRAIN_SHADOW_Y1, _LUMA_W, _dehaze_resolve_params,
)
from gpu_render.local_apply import FITS_DIR

_GRAIN_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _load_field_scale() -> float:
    try:
        with open(os.path.join(FITS_DIR, "Grain__comp.json")) as f:
            return float(json.load(f).get("field_scale", 1.0))
    except (OSError, ValueError, TypeError):
        return 1.0


_FIELD_SCALE = _load_field_scale()


# ============================ 通用空间算子 ============================

def _sympad(x: torch.Tensor, r: int) -> torch.Tensor:
    """对称填充（numpy 'symmetric' / cv2 BORDER_REFLECT：边缘像素重复）。"""
    if r <= 0:
        return x
    x = torch.cat([x[..., :r].flip(-1), x, x[..., -r:].flip(-1)], dim=-1)
    x = torch.cat([x[..., :r, :].flip(-2), x, x[..., -r:, :].flip(-2)], dim=-2)
    return x


def _box(x: torch.Tensor, r: int) -> torch.Tensor:
    """归一化 box 滤波（cv2.boxFilter BORDER_REFLECT 精确复现，double 积分图）。"""
    if r <= 0:
        return x
    B, C, H, W = x.shape
    xp = _sympad(x, r).double()
    ii = xp.cumsum(-2).cumsum(-1)
    ii = F.pad(ii, (1, 0, 1, 0))
    k = 2 * r + 1
    S = (ii[..., k:k + H, k:k + W] - ii[..., 0:H, k:k + W]
         - ii[..., k:k + H, 0:W] + ii[..., 0:H, 0:W])
    return (S / (k * k)).to(x.dtype)


def _guided_gray(guide: torch.Tensor, src: torch.Tensor, radius: int, eps: float,
                 sub: int = 4) -> torch.Tensor:
    """灰度引导滤波（可下采样加速），复现 non_gimp_ops._dehaze_guided_gray。"""
    if sub > 1:
        B, C, H, W = guide.shape
        sh, sw = max(1, H // sub), max(1, W // sub)
        g = F.interpolate(guide, size=(sh, sw), mode="area")
        s = F.interpolate(src, size=(sh, sw), mode="area")
        f = _guided_gray(g, s, max(1, radius // sub), eps, sub=1)
        return F.interpolate(f, size=(H, W), mode="bilinear", align_corners=False)
    mean_i, mean_p = _box(guide, radius), _box(src, radius)
    corr_i = _box(guide * guide, radius)
    corr_ip = _box(guide * src, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, radius) * guide + _box(b, radius)


# ============================ Grain ============================

def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """scipy.ndimage.gaussian_filter 等价（truncate=4, mode='reflect'=symmetric）。

    x: B,1,H,W 单通道。分离核，symmetric 边界。
    """
    radius = int(4.0 * sigma + 0.5)
    if radius < 1:
        return x
    xg = torch.arange(-radius, radius + 1, device=x.device, dtype=torch.float64)
    k = torch.exp(-0.5 / (sigma * sigma) * xg * xg)
    k = (k / k.sum()).to(x.dtype)
    B, C, H, W = x.shape
    xp = _sympad(x, radius)                         # B,C,H+2r,W+2r
    kh = k.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    x1 = F.conv2d(xp, kh, groups=C)                 # 沿 W valid → B,C,H+2r,W
    kv = k.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    return F.conv2d(x1, kv, groups=C)               # 沿 H valid → B,C,H,W


def _grain_luma_weight(luma: torch.Tensor) -> torch.Tensor:
    """亮度权重：中间调平台 1.0，极暗/极亮线性 ramp 衰减（复现 non_gimp_ops）。"""
    w = torch.ones_like(luma)
    if _GRAIN_SHADOW_Y1 > _GRAIN_SHADOW_Y0:
        t = ((luma - _GRAIN_SHADOW_Y0) / (_GRAIN_SHADOW_Y1 - _GRAIN_SHADOW_Y0)).clamp(0, 1)
        w = w * (_GRAIN_SHADOW_W + (1.0 - _GRAIN_SHADOW_W) * t)
    if _GRAIN_HIGH_Y1 > _GRAIN_HIGH_Y0:
        t = ((luma - _GRAIN_HIGH_Y0) / (_GRAIN_HIGH_Y1 - _GRAIN_HIGH_Y0)).clamp(0, 1)
        w = w * (1.0 + (_GRAIN_HIGH_W - 1.0) * t)
    return w


def _seed_from_slice(sl_hwc: np.ndarray) -> int:
    """复现 non_gimp_ops._grain_seed body（输入已 [::16,::16] 切片的 HWC float01）。"""
    q = np.clip(sl_hwc * 255.0, 0, 255).astype(np.uint8)
    return zlib.crc32(np.ascontiguousarray(q).tobytes()) & 0xFFFFFFFF


def _grain(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    attrs = ctx.get("attrs") or {}
    amount = float(attrs.get("GrainAmount", 0))
    if amount <= 0:
        return img
    size = float(attrs.get("GrainSize", 25))
    frequency = float(attrs.get("GrainFrequency", 50))
    B, C, H, W = img.shape
    dev = img.device

    # 白噪声底：逐图 crc32 种子 → numpy PCG64（复现标定参考的逐像素场）
    sl = img[:, :, ::16, ::16].detach().permute(0, 2, 3, 1).cpu().numpy()  # B,h',w',C
    noise = np.empty((B, H, W), dtype=np.float32)
    for b in range(B):
        seed = _seed_from_slice(sl[b])
        noise[b] = np.random.default_rng(seed).standard_normal((H, W), dtype=np.float32)
    n = torch.from_numpy(noise).to(dev).unsqueeze(1)                       # B,1,H,W

    # 谱形高通
    sg = _GRAIN_HP_SIGMA * max(size, 1.0) / 25.0
    m = min(_GRAIN_HP_MIX * frequency / 50.0, 0.95)
    g = n - m * _gaussian_blur(n, sg)
    std = g.reshape(B, -1).std(dim=1, unbiased=False).clamp_min(1e-8).view(B, 1, 1, 1)
    g = g / std

    # 幅度 × 分辨率因子
    sigma = float(np.interp(amount, _GRAIN_AMP_X, _GRAIN_AMP_Y))
    sigma *= 1.0 + _GRAIN_RES_K / float(min(H, W))

    lw = torch.tensor(_GRAIN_LUMA, device=dev).view(1, 3, 1, 1)
    luma = (img * lw).sum(1, keepdim=True)                                 # B,1,H,W
    field = sigma * g * _grain_luma_weight(luma)                          # B,1,H,W
    out = (img + field).clamp(0.0, 1.0)                                    # 广播到 3 通道
    if _FIELD_SCALE != 1.0:
        out = (img + _FIELD_SCALE * (out - img)).clamp(0.0, 1.0)
    return out


# ============================ Dehaze ============================

def _srgb2lin(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin2srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _sat_gain(img: torch.Tensor, amount: float) -> torch.Tensor:
    if abs(amount) < 1e-6:
        return img
    chroma = img.max(1, keepdim=True).values - img.min(1, keepdim=True).values
    roll = (1.0 - chroma / 0.85).clamp(0.0, 1.0)
    gain = 1.0 + amount * roll
    mean = img.mean(1, keepdim=True)
    return mean + (img - mean) * gain


def _dehaze(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    v = float(ctx["attrs"].get("Dehaze", ctx.get("label", "0")))
    if abs(v) < 1e-3:
        return img
    p = _dehaze_resolve_params(v, None)               # 与 numpy 同一插值路径
    B, C, H, W = img.shape
    dev = img.device
    short = min(H, W)
    lw = torch.tensor(_LUMA_W, device=dev).view(1, 3, 1, 1)
    lw1 = torch.tensor(_LUMA_W, device=dev)           # (3,)

    # --- 雾特征：细化暗通道 + 大气光 A ---
    luma = (img * lw).sum(1, keepdim=True)
    dc_raw = img.min(dim=1, keepdim=True).values
    er = max(1, int(round(short * 0.01)))
    dc = -F.max_pool2d(-dc_raw, kernel_size=2 * er + 1, stride=1, padding=er)  # erode
    dc = _guided_gray(luma, dc, radius=max(4, int(round(short * 0.04))), eps=1e-3)
    dcf = dc.clamp(0.0, 1.0)
    n_px = H * W
    k = max(1, int(n_px * 0.01))
    idx = dcf.reshape(B, n_px).topk(k, dim=1).indices                     # B,k (top-1%)
    img_flat = img.reshape(B, 3, n_px)
    a_top = torch.gather(img_flat, 2, idx.unsqueeze(1).expand(B, 3, k)).mean(dim=2)
    a_q95 = torch.quantile(img_flat, 0.95, dim=2)
    a_col = torch.maximum(a_top, a_q95).clamp(0.0, 1.0)                    # B,3

    dcc = _box(dc_raw, max(8, int(round(short * 0.25))))                   # 粗暗通道

    a_mix = float(p.get("a_mix", 1.0))
    A = ((1.0 - a_mix) + a_mix * a_col).view(B, 3, 1, 1)
    a_luma = ((a_col * lw1).sum(1) * a_mix + (1.0 - a_mix)).clamp_min(0.4).view(B, 1, 1, 1)

    k0 = float(p.get("k0", 0.0))
    k1 = float(p.get("k1", 0.0))
    k2 = float(p.get("k2", 0.0))
    sp = float(p.get("sp", 0.0))
    driver = ((1.0 - sp) * dc_raw + sp * dcf) / a_luma
    gdc = float(p.get("gdc", 1.0))
    dthr = float(p.get("dthr", 0.12))
    dark_frac = (driver < dthr).float().mean(dim=(1, 2, 3)).view(B, 1, 1, 1)
    kq = float(p.get("kq", 0.0))
    if abs(kq) > 1e-6:
        q95 = torch.quantile(driver.reshape(B, -1), 0.95, dim=1).view(B, 1, 1, 1)
        k1_t = k1 * (1.0 + kq * (q95 - 0.7)).clamp_min(0.0)
    else:
        k1_t = torch.full((B, 1, 1, 1), k1, device=dev)
    if abs(gdc - 1.0) > 1e-6:
        driver = driver.clamp(0.0, 1.5).pow(gdc)
    field = k0 + k1_t * driver + k2 * dcc
    # 暗部保护（2026-07 +70/+100 重拟合）：同 numpy 版 kd 机制
    kd = float(p.get("kd", 0.0))
    if abs(kd) > 1e-6:
        field = field * (1.0 + kd * (0.2 - dark_frac)).clamp_min(0.0)

    # 效果下限（直方图自适应 floor）
    fl0 = float(p.get("fl0", 0.0))
    flg = float(p.get("flg", 0.0))
    vk = float(p.get("vk", 0.0))
    if abs(flg) > 1e-6:
        floor_v = (fl0 + flg * (-(dark_frac + 0.005).log())).clamp_min(0.0)
    else:
        floor_v = fl0 * torch.exp(-float(p.get("fl1", 2.6)) * dark_frac)
    if abs(fl0) > 1e-9 or abs(flg) > 1e-9:
        if vk > 1e-6:
            applied = field + floor_v * torch.exp(-vk * field.clamp_min(0.0))
        else:
            applied = torch.maximum(field, floor_v)
        field = torch.where(floor_v > 1e-9, applied, field)               # 逐图匹配 numpy skip

    if v > 0:
        # 亮度倾斜（2026-07 重拟合）：同 numpy 版 hk 机制
        hk = float(p.get("hk", 0.0))
        if abs(hk) > 1e-6:
            field = field * (1.0 + hk * (luma - 0.6))
        conv = float(p.get("conv", 0.0))
        if conv > 1e-6:
            field = field / (1.0 - conv * field.clamp(0.0, 0.92))
        s = field.clamp(0.0, float(p.get("smax", 3.0)))
        # 大气光色度放大（2026-07 重拟合）：同 numpy 版 acx 机制
        acx = float(p.get("acx", 0.0))
        if abs(acx) > 1e-6:
            A = a_luma + (1.0 + acx) * (A - a_luma)
        apiv = float(p.get("apiv", 1.0))
        if float(p.get("lin", 0.0)) > 0.5:
            img_l = _srgb2lin(img)
            A_l3 = _srgb2lin(A.clamp(0.0, 1.0)) * apiv
            out = _lin2srgb((img_l - A_l3) * (1.0 + s) + A_l3)
        else:
            A_piv = A * apiv if apiv > 1.0 else A
            # 波长相关传输（2026-07 重拟合）：同 numpy 版 wvl 机制
            wvl = float(p.get("wvl", 0.0))
            if abs(wvl) > 1e-6:
                d_c = torch.tensor([-0.4, 0.0, 1.0], device=dev).view(1, 3, 1, 1)
                out = (img - A_piv) * (1.0 + s * (1.0 + wvl * d_c)) + A_piv
            else:
                out = (img - A_piv) * (1.0 + s) + A_piv
        out = _sat_gain(out, float(p.get("sat", 0.0)))
        lc = float(p.get("lc", 0.0))
        if abs(lc) > 1e-6:
            luma2 = (out.clamp(0.0, 1.0) * lw).sum(1, keepdim=True)
            base = _guided_gray(luma2, luma2, radius=max(4, int(round(short * 0.05))),
                                eps=4e-3)
            out = out + lc * (luma2 - base)
    else:
        veil = field.clamp(0.0, float(p.get("vmax", 0.95)))
        out = img * (1.0 - veil) + A * veil
        desat = float(p.get("desat", 0.0))
        if abs(desat) > 1e-6:
            luma2 = (out * lw).sum(1, keepdim=True)
            w = (desat * veil).clamp(0.0, 1.0)
            out = out + w * (luma2 - out)
    return out.clamp(0.0, 1.0)


OPS_GPU = {"Grain": _grain, "Dehaze": _dehaze}
