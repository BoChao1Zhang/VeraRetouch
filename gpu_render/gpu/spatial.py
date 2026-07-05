"""spatial —— GPU(torch/cuda:1) 移植: 空间滤波类算子, 复现 ops_v2 的 numpy 标定输出.

移植算子 (对齐 tools/lr_calib/ops_v2 + image_ops/non_gimp_ops 的 numpy 标定版):
    Clarity                     (adjust_clarity_v2; Lab-L 大半径 bilateral 局部对比 + 2D 响应表)
    Sharpness / SharpenParams   (lr_sharpen; USM + 软限幅 + 色调权重 + edge-masking, comp amount_scale)
    Texture                     (adjust_texture; 多尺度 band-pass + 中间调 mask, value_map remap)
    LuminanceNoiseReduction     (adjust_luminance_noise_reduction; spike 清理 + 保边 bilateral 混合)
    ColorNR                     (apply_color_nr; opponent 色度 bilateral 平滑)
    Vignette / VignetteParams   (apply_postcrop_vignette + apply_lens_vignette; 径向场 + 锚点)

张量约定: BCHW float32 [0,1] on cuda:1 (支持 batch, 并行渲染的关键). ctx 与 ops_v2 一致
    {"label","attrs","elements","fit"}. 导出 OPS_GPU = {op_name: fn}, fn(x_bchw, ctx) -> 同形张量.

标定表/常数直接从 image_ops.non_gimp_ops 的烘焙常量读取 (数据, 非算法), 保证与 numpy 版逐比特一致.
GPU 空间滤波: kornia gaussian/median; cv2-faithful bilateral 用滑窗 (圆形邻域 + reflect101, 显存安全).
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import kornia

# --- 仅暴露 cuda:1 时代码里设备名为 cuda:0; 否则直接 cuda:1. parity.DEVICE 与此一致 ---
DEVICE = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1")

# 亮度权重 (与 numpy 版一一对应)
_W709 = torch.tensor([0.2126, 0.7152, 0.0722], dtype=torch.float32)   # lr_sharpen / vignette luma
_W601 = torch.tensor([0.299, 0.587, 0.114], dtype=torch.float32)      # texture/lumNR/colorNR luma

# ---------------------------------------------------------------------------
# 标定常量 (从 non_gimp_ops 读取烘焙数据, 保证与 numpy 版一致)
# ---------------------------------------------------------------------------
from gpu_render.image_ops import non_gimp_ops as _N  # noqa: E402

_CLARITY_TABLES = _N._load_tables(None)     # 烘焙的 v2 clarity 响应表 (pos/neg)
_VIG = _N.VIG_CONSTS
_COLOR_NR_PARAMS = _N.COLOR_NR_PARAMS

_FITS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fits")


def _load_json(name: str) -> dict:
    p = os.path.join(_FITS_DIR, name)
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# lr_sharpen 组合标定强度缩放 (与 ops_v2/sharpen.py 一致)
_SHARP_AMOUNT_SCALE = float(_load_json("Sharpness__comp.json").get("amount_scale", 1.0))
_TEXTURE_VALUE_MAP = _load_json("Texture.json").get("value_map")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def _luma(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x:(B,3,H,W) -> (B,1,H,W) 亮度."""
    ww = w.to(x.device).view(1, 3, 1, 1)
    return (x * ww).sum(dim=1, keepdim=True)


def _smoothstep(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    t = ((x - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """匹配 scipy.ndimage.gaussian_filter(truncate=4.0): kernel radius = int(4*sigma+0.5)."""
    if sigma <= 0:
        return x
    r = int(4.0 * sigma + 0.5)
    r = max(1, r)
    k = 2 * r + 1
    return kornia.filters.gaussian_blur2d(x, (k, k), (sigma, sigma), border_type="reflect")


def _bilateral_cv(x: torch.Tensor, d: int, sigma_color: float, sigma_space: float) -> torch.Tensor:
    """cv2.bilateralFilter 忠实复现 (单通道友好, 多通道逐通道). x:(B,C,H,W).

    圆形邻域 (r^2<=radius^2) + Gaussian 空间/值权 + BORDER_REFLECT_101(=torch reflect).
    逐 kernel 行向量化 (整行 dj 一次算完): python 循环仅 2r+1 次, 大幅降 kernel-launch 开销;
    显存 O(B*C*H*W*(2r+1)) 有界 (大半径 ColorNR 在共享 GPU 上也安全).
    """
    if d <= 0:
        radius = int(round(sigma_space * 1.5))
    else:
        radius = d // 2
    radius = max(1, radius)
    ks = 2 * radius + 1
    gc = -0.5 / (sigma_color * sigma_color)
    gs = -0.5 / (sigma_space * sigma_space)
    B, C, H, W = x.shape
    xpad = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    r2max = radius * radius
    dj = np.arange(ks) - radius
    xc = x.unsqueeze(-2)                                  # (B,C,H,1,W)
    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    for i in range(ks):
        di = i - radius
        rr = di * di + dj * dj                            # (ks,)
        keep = rr <= r2max
        if not keep.any():
            continue
        sw = np.where(keep, np.exp(rr * gs), 0.0).astype(np.float32)
        row = xpad[:, :, i:i + H, :]                      # (B,C,H,W+2r)
        # 展开该行所有 dj -> (B,C,H,ks,W): patches[...,j,x] = xpad[..., x+j] = 原图偏移 dj=j-radius
        patches = row.unfold(3, W, 1)                     # (B,C,H,ks,W)
        swt = torch.from_numpy(sw).to(x.device).view(1, 1, 1, ks, 1)
        diff = patches - xc
        w = torch.exp(diff * diff * gc) * swt
        acc += (w * patches).sum(dim=3)
        wsum += w.sum(dim=3)
    return acc / wsum


def _interp1d(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """np.interp 等价: xp 单调增 1D, 越界钳到端点."""
    xp = xp.to(x.device)
    fp = fp.to(x.device)
    xc = x.clamp(float(xp[0]), float(xp[-1]))
    idx = torch.searchsorted(xp, xc.reshape(-1), right=False).clamp(1, len(xp) - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    w = (xc.reshape(-1) - x0) / (x1 - x0)
    return (y0 + w * (y1 - y0)).reshape(x.shape)


def _interp_table_2d(table: torch.Tensor, xq: torch.Tensor, yq: torch.Tensor) -> torch.Tensor:
    """复现 non_gimp_ops._interp_table_2d: (nx,ny) 表在 [0,1]^2 bin 中心双线性插值.

    xq/yq 同形任意, table (nx,ny) on device.
    """
    nx, ny = table.shape
    fx = (xq * nx - 0.5).clamp(0.0, nx - 1.0)
    fy = (yq * ny - 0.5).clamp(0.0, ny - 1.0)
    x0 = torch.floor(fx).long()
    y0 = torch.floor(fy).long()
    x1 = torch.clamp(x0 + 1, max=nx - 1)
    y1 = torch.clamp(y0 + 1, max=ny - 1)
    tx = (fx - x0.float())
    ty = (fy - y0.float())
    flat = table.reshape(-1)
    v00 = flat[x0 * ny + y0]
    v01 = flat[x0 * ny + y1]
    v10 = flat[x1 * ny + y0]
    v11 = flat[x1 * ny + y1]
    return (v00 * (1 - ty) + v01 * ty) * (1 - tx) + (v10 * (1 - ty) + v11 * ty) * tx


def _lab_light(x: torch.Tensor) -> torch.Tensor:
    """Lab L/100 in [0,1], (B,3,H,W)->(B,1,H,W). kornia rgb_to_lab (== cv2 RGB2LAB, ΔL<0.003)."""
    lab = kornia.color.rgb_to_lab(x.clamp(0.0, 1.0))
    return (lab[:, 0:1] / 100.0).clamp(0.0, 1.0)


def _median3(x: torch.Tensor) -> torch.Tensor:
    return kornia.filters.median_blur(x, (3, 3))


def _local_variance(light: torch.Tensor, sigma: float) -> torch.Tensor:
    mean = _gaussian(light, sigma)
    mean_sq = _gaussian(light * light, sigma)
    return (mean_sq - mean * mean).clamp(min=0.0)


def _attr_f(attrs: dict, key: str, default: float) -> float:
    v = attrs.get(key, default)
    try:
        return float(str(v).replace("+", ""))
    except (ValueError, TypeError):
        return default


def _recompose(x: torch.Tensor, src_luma: torch.Tensor, dst_luma: torch.Tensor) -> torch.Tensor:
    ratio = dst_luma / (src_luma + 1e-6)
    return (x * ratio).clamp(0.0, 1.0)


# ===========================================================================
# Clarity  (adjust_clarity_v2)
# ===========================================================================
def _clarity(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    v = float(np.clip(float(ctx["label"]), -100.0, 100.0))
    if abs(v) < 1e-6:
        return x.clamp(0.0, 1.0)
    x = x.clamp(0.0, 1.0)
    B, _, H, W = x.shape
    light = _lab_light(x)                      # (B,1,H,W)
    short = float(min(H, W))

    if v > 0:
        t = _CLARITY_TABLES["pos"]
        sig = float(t["sig_frac"]) * short
        # 下采样 bilateral base (cv2: INTER_AREA -> bilateral -> INTER_LINEAR)
        ds = max(1, int(round(min(H, W) / 240.0)))
        small = (F.interpolate(light, size=(max(2, H // ds), max(2, W // ds)), mode="area")
                 if ds > 1 else light)
        d = max(3, int(round(2.0 * sig / ds)) | 1)
        base = _bilateral_cv(small, d, float(t.get("sig_color", 0.2)), sig / ds)
        if ds > 1:
            base = F.interpolate(base, size=(H, W), mode="bilinear", align_corners=False)
        c = light - base
        # 逐图亮度分位 u (CDF), [::2,::2] 采样
        sample = torch.sort(light[:, :, ::2, ::2].reshape(B, -1), dim=1).values
        u = (torch.searchsorted(sample, light.reshape(B, -1), right=False).float()
             / sample.shape[1]).reshape(B, 1, H, W)
        A = torch.tensor(np.asarray(t["A"], np.float32), device=x.device)
        Bt = torch.tensor(np.asarray(t["B"], np.float32), device=x.device)
        c_nodes = torch.tensor(np.asarray(t["c_nodes"], np.float32), device=x.device)
        shape_v = torch.tensor(np.asarray(t["shape"], np.float32), device=x.device)
        shape = _interp1d(c, c_nodes, shape_v)
        sv = t["s_v"]
        scale = float(np.interp(v, [p[0] for p in sv], [p[1] for p in sv]))
        delta = scale * (_interp_table_2d(A, light, u) + _interp_table_2d(Bt, light, u) * shape)
    else:
        t = _CLARITY_TABLES["neg"]
        sig = float(t["sig_frac"]) * short
        c = light - _gaussian(light, sig)
        Wt = torch.tensor(np.asarray(t["W"], np.float32), device=x.device)
        c_nodes = torch.tensor(np.asarray(t["c_nodes"], np.float32), device=x.device)
        shape_v = torch.tensor(np.asarray(t["shape"], np.float32), device=x.device)
        shape = _interp1d(c, c_nodes, shape_v)
        sv = t["s_v"]
        scale = float(np.interp(-v, [p[0] for p in sv], [p[1] for p in sv]))
        # _interp_table_1d: centers=(arange(n)+0.5)/n
        n = len(Wt)
        centers = (torch.arange(n, device=x.device, dtype=torch.float32) + 0.5) / n
        delta = scale * _interp1d(light, centers, Wt) * shape

    new_light = (light + delta).clamp(0.0, 1.0)
    ratio = (new_light / light.clamp(min=1e-4)).clamp(0.0, 4.0)
    return (x * ratio).clamp(0.0, 1.0)


# ===========================================================================
# Sharpness / SharpenParams  (lr_sharpen; ops_v2 comp amount_scale)
# ===========================================================================
def _lr_sharpen(x: torch.Tensor, amount: float, radius: float, detail: float,
                masking: float) -> torch.Tensor:
    amount = float(np.clip(amount, 0.0, 150.0))
    if amount < 1e-6:
        return x
    x = x.clamp(0.0, 1.0)
    radius = float(np.clip(radius, 0.5, 3.0))
    detail = float(np.clip(detail, 0.0, 100.0))
    masking = float(np.clip(masking, 0.0, 100.0))
    luma = _luma(x, _W709)
    d = max(detail / 25.0, 0.05)
    sigma = 0.75 * radius
    sigma *= d ** (-0.24 if d < 1.0 else -0.123)
    hf = luma - _gaussian(luma, sigma)
    a = amount * (0.019515 + 0.00047057 * amount)
    a *= radius ** -1.15
    a *= d ** 0.7
    cap = 0.165 * (0.75 + 0.25 * d)
    delta = a * hf / (1.0 + (a / cap) * hf.abs())
    tone_w = _smoothstep(luma, 0.02, 0.32) * (1.0 - 0.55 * _smoothstep(luma, 0.70, 1.02))
    delta = delta * tone_w
    if masking > 1e-6:
        m = masking / 100.0
        edge = _gaussian(hf.abs(), 2.0)
        t0 = 0.002 + 0.020 * m * m
        delta = delta * _smoothstep(edge, t0 * 0.5, t0 * 2.0)
    out_luma = (luma + delta).clamp(0.0, 1.0)
    ratio = out_luma / luma.clamp(min=1e-6)
    return (x * ratio).clamp(0.0, 1.0)


def _sharpness(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    attrs = ctx.get("attrs") or {}
    return _lr_sharpen(
        x,
        amount=_attr_f(attrs, "Sharpness", 0.0) * _SHARP_AMOUNT_SCALE,
        radius=_attr_f(attrs, "SharpenRadius", 1.0),
        detail=_attr_f(attrs, "SharpenDetail", 25.0),
        masking=_attr_f(attrs, "SharpenEdgeMasking", 0.0),
    )


# ===========================================================================
# Texture  (adjust_texture; value_map remap + /100 归一)
# ===========================================================================
def _texture_core(x: torch.Tensor, amount: float) -> torch.Tensor:
    amount = float(np.clip(amount, -1.0, 1.0))
    if abs(amount) < 1e-8:
        return x
    x = x.clamp(0.0, 1.0)
    luma = _luma(x, _W601)
    g1 = _gaussian(luma, 0.7)
    g2 = _gaussian(luma, 2.0)
    g3 = _gaussian(luma, 6.0)
    band = 0.75 * (g1 - g2) + 0.35 * (g2 - g3)
    activity = _smoothstep(band.abs(), 0.001, 0.020)
    broad = _smoothstep(luma, 0.10, 0.40) * (1.0 - _smoothstep(luma, 0.60, 0.90))
    center = 1.0 - _smoothstep((luma - 0.5).abs(), 0.12, 0.34)
    midtone = (broad * (0.55 + 0.90 * center)).clamp(0.0, 1.0)
    mask = activity * (0.35 + 0.65 * midtone)   # tone_strength=0.65
    out_luma = (luma + amount * 1.45 * band * mask).clamp(0.0, 1.0)   # gain_scale=1.45
    return _recompose(x, luma, out_luma)


def _texture(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    lr_v = float(ctx["label"])
    if _TEXTURE_VALUE_MAP:                       # value_map: LR 值 -> 本地值
        xs = [p[0] for p in _TEXTURE_VALUE_MAP]
        ys = [p[1] for p in _TEXTURE_VALUE_MAP]
        local_v = float(np.interp(lr_v, xs, ys))
    else:
        local_v = lr_v
    amount = local_v / 100.0 if abs(local_v) > 1.0 else local_v   # _normalize_global_compat
    return _texture_core(x, amount)


# ===========================================================================
# LuminanceNoiseReduction  (adjust_luminance_noise_reduction)
# ===========================================================================
def _lumnr_core(x: torch.Tensor, amount: float) -> torch.Tensor:
    amount = float(np.clip(amount, 0.0, 1.0))
    if amount <= 0:
        return x
    x = x.clamp(0.0, 1.0)
    raw_luma = _luma(x, _W601)
    median3 = _median3(raw_luma)
    base_noise = _local_variance(median3, 1.0).sqrt()
    spike_floor = torch.clamp(4.0 * base_noise + 0.02, min=0.08)
    spike_mask = (raw_luma - median3).abs() > spike_floor        # (B,1,H,W)
    med_rgb = _median3(x)
    work = torch.where(spike_mask, med_rgb, x)
    luma = _luma(work, _W601)
    preclean = luma

    noise_level = _local_variance(preclean, 1.0).sqrt()
    flat_mask = 1.0 - _smoothstep(noise_level, 0.01, 0.045)
    diameter = max(5, 5 + int(round(amount * 4.0)) * 2)
    sigma_color = 0.04 + amount * 0.03
    sigma_space = 2.0 + amount * 2.5
    smooth = _bilateral_cv(preclean, diameter, sigma_color, sigma_space)

    detail_ref = (preclean - smooth).abs()
    detail_mask = 1.0 - _smoothstep(detail_ref, 0.015, 0.06)
    variance_mask = 1.0 - _smoothstep(_local_variance(preclean, 1.2).sqrt(), 0.015, 0.05)
    shadow_weight = 1.0 - 0.25 * _smoothstep(preclean, 0.72, 0.95)
    blend = (amount * (0.15 + 0.85 * flat_mask) * (0.25 + 0.75 * detail_mask)
             * (0.3 + 0.7 * variance_mask) * shadow_weight).clamp(0.0, 1.0)
    out_luma = (preclean * (1.0 - blend) + smooth * blend).clamp(0.0, 1.0)
    return _recompose(work, luma, out_luma)


def _lumnr(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    lr_v = float(ctx["label"])                   # LumNR 无 value_map (fit 缺失) -> 直用 LR 值
    amount = lr_v / 100.0 if abs(lr_v) > 1.0 else lr_v
    return _lumnr_core(x, amount)


# ===========================================================================
# ColorNR  (apply_color_nr; opponent 色度 bilateral)
# ===========================================================================
def _interp_rows(rows: list, xval: float) -> list:
    rows = sorted(rows)
    xs = [r[0] for r in rows]
    return [float(np.interp(xval, xs, [r[i] for r in rows])) for i in range(1, len(rows[0]))]


def _color_nr(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    amount = float(ctx.get("attrs", {}).get("ColorNoiseReduction", 0))
    if amount <= 0:
        return x
    sigma_s, sigma_c, alpha = _interp_rows(_COLOR_NR_PARAMS, amount)
    if sigma_s <= 0 or alpha <= 0:
        return x
    x = x.clamp(0.0, 1.0)
    y = _luma(x, _W601)                          # (B,1,H,W)
    cb = x[:, 2:3] - y
    cr = x[:, 0:1] - y
    # 色度双边在 1/s 分辨率上算（色度低频 + sigma_s≥8px 大核）：O(ks²) 代价降 s⁴ 倍。
    # amount=25(库内主档) ks 31→17；近似误差实测 ΔE00 ≪ 0.1（vs 全分辨率精确版，
    # 见 scratch/colornr_fast_check.py），远低于 0.5 parity 门与 1.03 的 LR 标定误差。
    s = max(1, int(sigma_s // 5))
    if s > 1:
        B, _, H, W = x.shape
        hs, ws = max(1, H // s), max(1, W // s)
        c2 = torch.cat([cb, cr], dim=1)
        c2d = F.interpolate(c2, size=(hs, ws), mode="area")
        f2d = _bilateral_cv(c2d, 0, sigma_c, sigma_s / s)
        f2 = F.interpolate(f2d, size=(H, W), mode="bilinear", align_corners=False)
        fcb, fcr = f2[:, 0:1], f2[:, 1:2]
    else:
        fcb = _bilateral_cv(cb, 0, sigma_c, sigma_s)
        fcr = _bilateral_cv(cr, 0, sigma_c, sigma_s)
    cb = cb + alpha * (fcb - cb)
    cr = cr + alpha * (fcr - cr)
    r = y + cr
    b = y + cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return torch.cat([r, g, b], dim=1).clamp(0.0, 1.0)


# ===========================================================================
# Vignette / VignetteParams  (apply_postcrop_vignette + apply_lens_vignette)
# ===========================================================================
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _anchor(xval: float, anchors: list) -> float:
    xs = [a[0] for a in anchors]
    ys = [a[1] for a in anchors]
    return float(np.interp(xval, xs, ys))


def _vig_field(H: int, W: int, roundness: float, device) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, H, device=device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1.0, 1.0, W, device=device).view(1, W).expand(H, W)
    d_ell = torch.sqrt(xx * xx + yy * yy) / float(np.sqrt(2.0))
    rho = float(np.clip(roundness / 100.0, -1.0, 1.0))
    if rho == 0.0:
        return d_ell
    if rho > 0:
        mn = float(max(1, min(H, W)))
        sx, sy = W / mn, H / mn
        norm = float(np.sqrt(sx * sx + sy * sy))
        d_cir = torch.sqrt((xx * sx) ** 2 + (yy * sy) ** 2) / norm
        s = rho / 0.7
        k = min(1.0, s * float(_VIG["round_pos_k"]))
        g = 1.0 + s * (float(_VIG["round_pos_gamma"]) - 1.0)
        d = ((1.0 - k) * d_ell + k * d_cir) ** g
    else:
        d_box = torch.maximum(xx.abs(), yy.abs())
        s = -rho / 0.7
        alpha = s * float(_VIG["round_neg_alpha"])
        beta = 1.0 + s * (float(_VIG["round_neg_beta"]) - 1.0)
        d = d_box ** alpha * d_ell ** beta
    return d


def _vig_mask(u: torch.Tensor, midpoint: float, feather: float, roundness: float) -> torch.Tensor:
    rho = float(np.clip(roundness / 100.0, -1.0, 1.0))
    s = abs(rho) / 0.7
    sfx = "neg" if rho < 0 else "pos"
    u0 = (_anchor(midpoint, _VIG["u0_mid"]) + _anchor(feather, _VIG["u0_fea_delta"])
          + s * float(_VIG.get(f"round_u0_delta_{sfx}", 0.0)))
    w = (_anchor(feather, _VIG["w_fea"]) + _anchor(midpoint, _VIG["w_mid_delta"])
         + s * float(_VIG.get(f"round_w_delta_{sfx}", 0.0)))
    w = max(1e-3, w)
    t = ((u - u0) / w).clamp(0.0, 1.0)
    pt = torch.tensor(_VIG["profile_t"], dtype=torch.float32, device=u.device)
    pc = torch.tensor(_VIG["profile_c"], dtype=torch.float32, device=u.device)
    return _interp1d(t, pt, pc)


def _darken_hp(x: torch.Tensor, v: float, mask: torch.Tensor) -> torch.Tensor:
    pd = float(_VIG["dark_pd"])
    h = float(_VIG["dark_prot_h"])
    q = float(_VIG["dark_prot_q"])
    luma = _luma(x, _W709).clamp(0.0, 1.0)       # (B,1,H,W)
    prot = 1.0 - h * luma ** q
    sub = v * mask * prot                          # (B,1,H,W) broadcast
    if pd == 1.0:
        out = x - sub
    else:
        e = x.clamp(0.0, 1.0) ** (1.0 / pd)
        out = (e - sub).clamp(0.0, 1.0) ** pd
    return out.clamp(0.0, 1.0)


def _brighten_hp(x: torch.Tensor, v: float, mask: torch.Tensor) -> torch.Tensor:
    beta = float(_VIG["bright_beta"])
    lin = _srgb_to_linear(x)
    veff = (v * mask ** beta).clamp(0.0, 1.0)
    out = 1.0 - (1.0 - lin) * (1.0 - veff)
    return _linear_to_srgb(out)


def _darken_cp(x: torch.Tensor, v: float, mask: torch.Tensor) -> torch.Tensor:
    pd = float(_VIG["dark_pd"])
    h = float(_VIG["dark_prot_h"])
    q = float(_VIG["dark_prot_q"])
    luma = _luma(x, _W709).clamp(1e-4, 1.0)
    prot = 1.0 - h * luma ** q
    sub = v * float(_VIG["style2_v_scale"]) * mask * prot
    l2 = (luma ** (1.0 / pd) - sub).clamp(0.0, 1.0) ** pd
    gain = _srgb_to_linear(l2) / _srgb_to_linear(luma).clamp(min=1e-6)
    out = _srgb_to_linear(x) * gain
    return _linear_to_srgb(out)


def _apply_postcrop(x: torch.Tensor, amount: float, midpoint: float, feather: float,
                    roundness: float, style: int) -> torch.Tensor:
    amount = float(np.clip(amount, -100.0, 100.0))
    if amount == 0.0:
        return x
    x = x.clamp(0.0, 1.0)
    B, _, H, W = x.shape
    u = _vig_field(H, W, roundness, x.device).view(1, 1, H, W)
    mask = _vig_mask(u, midpoint, feather, roundness)
    if amount < 0:
        v = _anchor(-amount, _VIG["dark_v"])
        if int(style) == 2:
            return _darken_cp(x, v, mask)
        return _darken_hp(x, v, mask)
    v = _anchor(amount, _VIG["bright_v"])
    return _brighten_hp(x, v, mask)


def _apply_lens(x: torch.Tensor, amount: float, midpoint: float) -> torch.Tensor:
    amount = float(np.clip(amount, -100.0, 100.0))
    if amount == 0.0:
        return x
    x = x.clamp(0.0, 1.0)
    B, _, H, W = x.shape
    yy = torch.linspace(-1.0, 1.0, H, device=x.device).view(H, 1).expand(H, W)
    xx = torch.linspace(-1.0, 1.0, W, device=x.device).view(1, W).expand(H, W)
    mn = float(max(1, min(H, W)))
    sx, sy = W / mn, H / mn
    u = torch.sqrt((xx * sx) ** 2 + (yy * sy) ** 2) / float(np.sqrt(sx * sx + sy * sy))
    u = u.clamp(0.0, 1.0)
    pt = torch.tensor(_VIG["lens_profile_t"], dtype=torch.float32, device=x.device)
    pc = torch.tensor(_VIG["lens_profile_c"], dtype=torch.float32, device=x.device)
    p = _interp1d(u, pt, pc)
    amp = _anchor(amount, _VIG["lens_amp"])
    gain = (2.0 ** (amp * p)).view(1, 1, H, W)
    lin = _srgb_to_linear(x) * gain
    return _linear_to_srgb(lin)


def _vignette(x: torch.Tensor, ctx: dict) -> torch.Tensor:
    attrs = ctx.get("attrs") or {}
    out = x.clamp(0.0, 1.0)
    pc = float(attrs.get("PostCropVignetteAmount", 0) or 0)
    if pc != 0.0:
        out = _apply_postcrop(
            out, pc,
            midpoint=float(attrs.get("PostCropVignetteMidpoint", 50) or 50),
            feather=float(attrs.get("PostCropVignetteFeather", 50) or 50),
            roundness=float(attrs.get("PostCropVignetteRoundness", 0) or 0),
            style=int(float(attrs.get("PostCropVignetteStyle", 1) or 1)))
    lens = float(attrs.get("VignetteAmount", 0) or 0)
    if lens != 0.0:
        out = _apply_lens(out, lens, midpoint=float(attrs.get("VignetteMidpoint", 50) or 50))
    return out


OPS_GPU = {
    "Clarity": _clarity,
    "Sharpness": _sharpness,
    "SharpenParams": _sharpness,
    "Texture": _texture,
    "LuminanceNoiseReduction": _lumnr,
    "ColorNR": _color_nr,
    "Vignette": _vignette,
    "VignetteParams": _vignette,
}
