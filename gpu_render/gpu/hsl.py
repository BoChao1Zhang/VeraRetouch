"""GPU (torch, cuda:1) 实现的 HSL 组算子 —— 复现 ops_v2 的 numpy 标定输出。

覆盖：
  - Blue 三兄弟   HueAdjustmentBlue / SaturationAdjustmentBlue / LuminanceAdjustmentBlue
                  （逐带响应表 dh/w_sr/SR1/w_dl/DL，v→表插值 + 逐像素 hue/(c,L) 双线性）
  - Orange/Purple 六算子 {Hue,Saturation,Luminance}Adjustment{Orange,Purple}
                  （raised-cosine 带权重 + hue/sat/lum 耦合，参数走 _interp_params）
  - 稀有 5 色带 15 算子 {Hue,Saturation,Luminance}Adjustment{Red,Green,Yellow,Aqua,Magenta}
                  （gimp_stable 路径：7 桶带映射 + stability 链，复现 replay.apply_scalar_op
                  的完整旧 fits 标量语义 value_map → 核 → post_luma_lut → post_rgb_lut）
  - BlackWhite    灰度混合（8 带 GrayMixer + HSV 权重）
  - Calib 面板 7  {Red,Green,Blue}{Hue,Saturation} + ShadowTint（线性域 3x3 通道混合）

约定：img 为 BCHW float32 [0,1] RGB（on cuda:1，见 parity.py）。fn(img, ctx) -> 同形张量。
逐 preset 的标量/表插值在 host 端用 non_gimp_ops 的 numpy helper 精确算好，逐像素数学在 torch。
一次 kernel 摊平整个 batch（B 维天然并行），param=v 由单一 ctx 决定并作用于全 batch。

只读依赖 image_ops.non_gimp_ops 的标定常量与标量 helper（禁改本体，仅 import）。
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

os.environ.setdefault("MONETGPT_TORCH_DEVICE", "cuda:1")

# ---- 从本体导入标定常量 / host 端标量 helper（read-only）----
from gpu_render.image_ops.non_gimp_ops import (  # noqa: E402
    BW_BANDS,
    BW_BAND_PARAMS,
    _C_ANCHORS,
    _CALIB_FITS,
    _HSL_BLUE_TABLES,
    _HUE_ANCHORS_DEG,
    _L_ANCHORS,
    _ORANGE_PURPLE_FITS,
    _build_all_hsl_adjustments,
    _calib_anchors,
    _interp_matrix,
    _interp_params,
    _interp_v,
    _resolve_hsl_stability_cfg,
    _safe_float,
    get_hsl_config,
)
# 旧 fits 标量路径的 host 端 helper（value_map 插值 / LUT 相邻扫描值混合 / 扫描网格）
from gpu_render.local_apply import local_value  # noqa: E402
from gpu_render.sweeps import OPS as _SWEEP_OPS, fmt_value  # noqa: E402

import torch.nn.functional as F  # noqa: E402

_LUMA_LIN = (0.2126, 0.7152, 0.0722)


# ===========================================================================
# 颜色空间 / 逐像素 primitive（全部 BCHW batched）
# ===========================================================================
def _srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055)


def _rgb_to_hls(rgb: torch.Tensor):
    """BCHW RGB -> (H, L, S) 各 (B,H,W)，与 colorsys / rgb_to_hls_np 一致。"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    maxc = rgb.amax(1)
    minc = rgb.amin(1)
    L = (maxc + minc) * 0.5
    delta = maxc - minc
    small = delta < 1e-20
    ds = torch.where(small, torch.ones_like(delta), delta)
    H = torch.zeros_like(L)
    H = torch.where((r == maxc) & (~small), (g - b) / ds, H)
    H = torch.where((g == maxc) & (~small), 2.0 + (b - r) / ds, H)
    H = torch.where((b == maxc) & (~small), 4.0 + (r - g) / ds, H)
    H = torch.remainder(H / 6.0, 1.0)
    eps = 1e-8
    sumc = maxc + minc
    S = torch.where(
        small,
        torch.zeros_like(L),
        torch.where(L <= 0.5, delta / (sumc + eps), delta / (2.0 - sumc + eps)),
    )
    return H, L, S


def _hls_to_rgb(H: torch.Tensor, L: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """(H,L,S) 各 (B,H,W) -> BCHW RGB，与 hls_to_rgb_np 一致（S=0 自然退化为 L）。"""
    def h2c(m1, m2, h):
        h = torch.remainder(h, 1.0)
        return torch.where(
            h < 1.0 / 6.0,
            m1 + (m2 - m1) * 6.0 * h,
            torch.where(
                h < 0.5,
                m2,
                torch.where(h < 2.0 / 3.0, m1 + (m2 - m1) * 6.0 * (2.0 / 3.0 - h), m1),
            ),
        )

    m2 = torch.where(L < 0.5, L + L * S, L + S - L * S)
    m1 = 2.0 * L - m2
    R = h2c(m1, m2, H + 1.0 / 3.0)
    G = h2c(m1, m2, H)
    B = h2c(m1, m2, H - 1.0 / 3.0)
    return torch.stack([R, G, B], dim=1)


def _hsv_hue_sat(rgb: torch.Tensor):
    """HSV 色相(度) 与 饱和度，各 (B,H,W)，与 _hsv_hue_sat(numpy) 一致。"""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    mx = rgb.amax(1)
    mn = rgb.amin(1)
    c = mx - mn
    cs = torch.where(c > 1e-8, c, torch.ones_like(c))
    m = c > 1e-8
    h = torch.zeros_like(mx)
    h = torch.where(m & (mx == r), torch.remainder((g - b) / cs, 6.0), h)
    h = torch.where(m & (mx == g) & (mx != r), (b - r) / cs + 2.0, h)
    h = torch.where(m & (mx == b) & (mx != r) & (mx != g), (r - g) / cs + 4.0, h)
    h = h * 60.0
    s = torch.where(mx > 1e-8, c / torch.clamp(mx, min=1e-8), torch.zeros_like(mx))
    return h, s


def _interp1d(x: torch.Tensor, xa: torch.Tensor, ya: torch.Tensor) -> torch.Tensor:
    """沿 1D 单调 anchors 的线性插值（np.interp 语义：越界夹到端点）。x 任意形状。"""
    K = xa.shape[0]
    i = torch.clamp(torch.searchsorted(xa, x.contiguous(), right=True) - 1, 0, K - 2)
    x0 = xa[i]
    x1 = xa[i + 1]
    t = ((x - x0) / (x1 - x0)).clamp(0.0, 1.0)
    return ya[i] + (ya[i + 1] - ya[i]) * t


def _bilinear_cl(c: torch.Tensor, l_: torch.Tensor, tab: torch.Tensor,
                 ca: torch.Tensor, la: torch.Tensor) -> torch.Tensor:
    """(nc,nl) 表在 (chroma,L) 上双线性插值（越界夹端点），复现 _bilinear_cl(numpy)。"""
    nc, nl = tab.shape
    i = torch.clamp(torch.searchsorted(ca, c.contiguous()) - 1, 0, nc - 2)
    j = torch.clamp(torch.searchsorted(la, l_.contiguous()) - 1, 0, nl - 2)
    c0, c1 = ca[i], ca[i + 1]
    l0, l1 = la[j], la[j + 1]
    wc = ((c - c0) / (c1 - c0 + 1e-12)).clamp(0.0, 1.0)
    wl = ((l_ - l0) / (l1 - l0 + 1e-12)).clamp(0.0, 1.0)
    return (tab[i, j] * (1 - wc) * (1 - wl) + tab[i, j + 1] * (1 - wc) * wl
            + tab[i + 1, j] * wc * (1 - wl) + tab[i + 1, j + 1] * wc * wl)


def _t(a, device, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(a), device=device, dtype=dtype)


# ===========================================================================
# Blue 三兄弟：逐带响应表
# ===========================================================================
def _blue_ctx_value(op: str, ctx: dict) -> float:
    attrs = ctx.get("attrs") or {}
    if op in attrs:
        return float(attrs[op])
    return float(ctx.get("label", 0.0))


def _make_blue(op_key: str):
    def fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
        v = _blue_ctx_value(op_key, ctx)
        if abs(v) < 1e-6 or op_key not in _HSL_BLUE_TABLES:
            return img
        tabs = _HSL_BLUE_TABLES[op_key]
        # host 端沿扫描值插值（v 是逐 preset 标量）
        dh_cur = _interp_v(np.asarray(tabs["dh"], np.float32), v)     # (nh,)
        wsr_cur = _interp_v(np.asarray(tabs["w_sr"], np.float32), v)  # (nh,)
        sr1_cur = _interp_v(np.asarray(tabs["SR1"], np.float32), v)   # (nc,nl)
        wdl_cur = _interp_v(np.asarray(tabs["w_dl"], np.float32), v)  # (nh,)
        dl_cur = _interp_v(np.asarray(tabs["DL"], np.float32), v)     # (nc,nl)

        dev = img.device
        ha = _t(_HUE_ANCHORS_DEG, dev)
        ca = _t(_C_ANCHORS, dev)
        la = _t(_L_ANCHORS, dev)
        dh_t = _t(dh_cur, dev)
        wsr_t = _t(wsr_cur, dev)
        wdl_t = _t(wdl_cur, dev)
        sr1_t = _t(sr1_cur, dev)
        dl_t = _t(dl_cur, dev)

        rgb = img.clamp(0.0, 1.0)
        H, L, S = _rgb_to_hls(rgb)
        h_deg = H * 360.0
        c = rgb.amax(1) - rgb.amin(1)

        dh = _interp1d(h_deg, ha, dh_t)      # 端点=0 → 带外恒等
        w_sr = _interp1d(h_deg, ha, wsr_t)
        w_dl = _interp1d(h_deg, ha, wdl_t)
        sr1 = _bilinear_cl(c, L, sr1_t, ca, la)
        dl = _bilinear_cl(c, L, dl_t, ca, la)

        h_out = torch.remainder(h_deg + dh, 360.0) / 360.0
        s_out = (S * (1.0 + w_sr * sr1)).clamp(0.0, 1.0)
        l_out = (L + w_dl * dl).clamp(0.0, 1.0)
        return _hls_to_rgb(h_out, l_out, s_out).clamp(0.0, 1.0)

    fn.__name__ = f"gpu_{op_key}"
    return fn


# ===========================================================================
# Orange / Purple：raised-cosine 带 HSL
# ===========================================================================
def _band_weight(hue01: torch.Tensor, hc: float, wl: float, wr: float, p: float) -> torch.Tensor:
    d = torch.remainder(hue01 * 360.0 - hc + 180.0, 360.0) - 180.0
    t = torch.where(d < 0.0, -d / max(wl, 1e-3), d / max(wr, 1e-3))
    w = 0.5 + 0.5 * torch.cos(math.pi * torch.clamp(t, max=1.0))
    if abs(p - 1.0) > 1e-6:
        w = torch.clamp(w, min=0.0) ** p
    return w


def _apply_band_hsl(img: torch.Tensor, kind: str, prm: dict) -> torch.Tensor:
    """复现 _apply_lr_band_hsl(numpy)：单带 LR 风格 HSL 调整。"""
    rgb = img.clamp(0.0, 1.0)
    H, L, S = _rgb_to_hls(rgb)
    c = rgb.amax(1) - rgb.amin(1)

    w = _band_weight(H, prm["hc"], prm["wl"], prm["wr"], prm["p"])
    w = w * (c / 0.004).clamp(0.0, 1.0)

    hue_out, light_out, sat_out = H, L, S
    if kind == "hue":
        rot = prm["amp"] * (1.0 + prm["q"] * (c - 0.2))
        hue_out = torch.remainder(H + w * rot / 360.0, 1.0)
        sat_out = (S * (1.0 + prm["kc"] * w)).clamp(0.0, 1.0)
        light_out = (L + prm["kl"] * c * w).clamp(0.0, 1.0)
    elif kind == "sat":
        sat_out = (S * (1.0 + prm["gain"] * w)).clamp(0.0, 1.0)
        light_out = (L + prm["kl"] * c * w).clamp(0.0, 1.0)
        rot = prm["kh"] * (c / 0.3).clamp(0.0, 1.0)
        hue_out = torch.remainder(H + w * rot / 360.0, 1.0)
    elif kind == "lum":
        ramp = (c / max(prm["c0"], 1e-3)).clamp(0.0, 1.0)
        eff = prm["amp"] * w * ramp
        if prm["amp"] < 0.0:
            light_out = L * torch.clamp(1.0 + eff, min=0.0)
        else:
            m = float(np.clip(prm["m"], 0.0, 1.0))
            light_out = L + eff * (m * (1.0 - L) + (1.0 - m) * L)
        light_out = light_out.clamp(0.0, 1.0)
        f0 = 2.0 * torch.minimum(L, 1.0 - L)
        f1 = 2.0 * torch.minimum(light_out, 1.0 - light_out)
        beta = float(np.clip(prm["beta"], 0.0, 1.0))
        sat_out = (S * (f0 + beta * (f1 - f0)) / torch.clamp(f1, min=1e-4)).clamp(0.0, 1.0)
    else:  # pragma: no cover
        raise ValueError(f"unknown band-HSL kind: {kind}")

    return _hls_to_rgb(hue_out, light_out, sat_out).clamp(0.0, 1.0)


def _make_orange_purple(op_name: str, color: str, kind: str):
    def fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
        lr_v = float((ctx.get("attrs") or {}).get(op_name, ctx.get("label", 0.0)))
        if lr_v == 0.0:
            return img
        op_fit = _ORANGE_PURPLE_FITS.get(op_name) or {}
        prm = _interp_params(op_fit, color, kind, lr_v)  # host 端标量参数插值
        amp_key = "gain" if kind == "sat" else "amp"
        if abs(prm.get(amp_key, 0.0)) < 1e-9 and all(
            abs(prm.get(k, 0.0)) < 1e-9 for k in ("kc", "kl", "kh", "q") if k in prm
        ):
            return img
        return _apply_band_hsl(img, kind, prm)

    fn.__name__ = f"gpu_{op_name}"
    return fn


# ===========================================================================
# BlackWhite：8 带灰度混合
# ===========================================================================
def _bw_fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    attrs = ctx.get("attrs") or {}
    rgb = img.clamp(0.0, 1.0)
    lin = _srgb_to_linear(rgb)
    yw = torch.as_tensor(_LUMA_LIN, device=img.device, dtype=img.dtype)
    y = (lin * yw.view(1, 3, 1, 1)).sum(1)          # (B,H,W)
    hue, sat = _hsv_hue_sat(rgb)

    f = torch.zeros_like(y)
    for band in BW_BANDS:
        p = BW_BAND_PARAMS[band]
        m = float(attrs.get(f"GrayMixer{band}", p["default"]))
        if abs(m) < 1e-9:
            continue
        center = float(p["center"])
        width = max(float(p["width"]), 1.0)
        gain = float(p["gain"])
        dh = torch.abs(torch.remainder(hue - center + 180.0, 360.0) - 180.0)
        t = (dh / width).clamp(max=1.0)
        w = torch.cos(0.5 * math.pi * t) ** 2
        f = f + gain * w * (m / 100.0)
    gray = y * torch.clamp(1.0 + sat * f, min=0.0)
    out = _linear_to_srgb(gray)                     # 内部夹 [0,1]
    return torch.stack([out, out, out], dim=1)


# ===========================================================================
# Camera Calibration：线性域 3x3 通道混合
# ===========================================================================
def _compose_calib(sliders: dict):
    """复现 apply_lr_calibration_panel 的矩阵合成（float64），返回 (M3x3, b3, active)。"""
    M = np.eye(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    active = False
    for name, v in sliders.items():
        f = _CALIB_FITS.get(f"Calib{name}")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not f or not f.get("matrix") or abs(v) < 1e-9:
            continue
        Mi, bi = _interp_matrix(_calib_anchors(f), v)
        Mi = np.asarray(Mi, dtype=np.float64)
        M = Mi @ M
        b = Mi @ b + np.asarray(bi, dtype=np.float64)
        active = True
    return M, b, active


def _make_calib(op: str):
    def fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
        sliders = dict(ctx.get("attrs") or {})
        M, b, active = _compose_calib(sliders)
        if not active:
            return img
        Mt = torch.as_tensor(M, device=img.device, dtype=torch.float32)
        bt = torch.as_tensor(b, device=img.device, dtype=torch.float32).view(1, 3, 1, 1)
        lin = _srgb_to_linear(img)
        out = torch.einsum("ij,bjhw->bihw", Mt, lin) + bt
        return _linear_to_srgb(out)

    fn.__name__ = f"gpu_{op}"
    return fn


# ===========================================================================
# 稀有 5 色带（Red/Yellow/Green/Aqua/Magenta × Hue/Sat/Lum，共 15 算子）
# ===========================================================================
# numpy 参考 = replay.apply_scalar_op：
#   local_v = value_map 插值 → apply_non_gimp_config({op: local_v})（即 execute_hsl 的
#   gimp_stable 路径 _apply_gimp_hsl_transform + hsl.yaml stability 链）
#   → post_luma_lut 相邻扫描值混合 → post_rgb_lut 最近扫描值快照。
# 本节把该链逐步在 torch 复现：host 端算 fits 标量 / 7 桶 adj 数组，逐像素数学全 GPU。
# stability 链的两个空间步：
#   - luma_guard.guided：本 env 无 cv2.ximgproc → numpy 侧实际走 cv2.bilateralFilter
#     回退。_bilateral_cv2_t 精确复现 OpenCV bilateralFilter_32f（圆形支撑、中心权重 1、
#     BORDER_REFLECT_101；LUT 用解析 exp 替代，实测与 OpenCV 输出差 ~1e-7）。
#   - luma_guard.speckle：连通域面积筛选 → _small_component_mask 用迭代 min-label
#     传播（收敛）复现 cv2.connectedComponentsWithStats(8 连通) 的 area<=min_area 判定。

_FLT_EPSILON = 1.1920928955078125e-07   # OpenCV 常量场 shortcut 阈值


def _smoothstep_t(x: torch.Tensor, e0: float, e1: float) -> torch.Tensor:
    """复现 non_gimp_ops.smoothstep（含 +1e-8 分母）。"""
    t = ((x - e0) / (e1 - e0 + 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _median_filter_t(x: torch.Tensor, kernel: int) -> torch.Tensor:
    """复现 scipy.ndimage.median_filter(size=kernel, mode='nearest')。x: (B,C,H,W)。"""
    p = kernel // 2
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    win = xp.unfold(2, kernel, 1).unfold(3, kernel, 1)
    return win.reshape(*x.shape, kernel * kernel).median(dim=-1).values


def _bilateral_cv2_t(delta: torch.Tensor, d: int, sigma_color: float,
                     sigma_space: float) -> torch.Tensor:
    """复现 cv2.bilateralFilter(float32 单通道) 的 C++ 路径。delta: (B,H,W)。
    圆形支撑 r<=radius（不含中心，中心权重恒 1）、reflect101 边界、
    (max-min)<FLT_EPSILON 的常量场直通（逐 batch 图独立）。"""
    radius = max(d // 2, 1)
    B, Hh, Ww = delta.shape
    gcc = -0.5 / (sigma_color * sigma_color)
    gsc = -0.5 / (sigma_space * sigma_space)
    pad = F.pad(delta.unsqueeze(1), (radius,) * 4, mode="reflect").squeeze(1)
    s = torch.zeros_like(delta)
    ws = torch.zeros_like(delta)
    for di in range(-radius, radius + 1):
        for dj in range(-radius, radius + 1):
            r = math.sqrt(di * di + dj * dj)
            if r > radius or (di == 0 and dj == 0):
                continue
            w0 = math.exp(r * r * gsc)
            val = pad[:, radius + di:radius + di + Hh, radius + dj:radius + dj + Ww]
            w = torch.exp((val - delta) ** 2 * gcc) * w0
            s = s + val * w
            ws = ws + w
    out = (s + delta) / (ws + 1.0)
    rng = delta.amax(dim=(1, 2)) - delta.amin(dim=(1, 2))
    return torch.where((rng < _FLT_EPSILON).view(B, 1, 1), delta, out)


def _small_component_mask(mask: torch.Tensor, min_area: int,
                          max_iters: int = 8192) -> torch.Tensor:
    """8 连通域中 area<=min_area 像素的掩码（逐 batch 图独立），复现
    cv2.connectedComponentsWithStats 的面积筛选。迭代 3x3 min-label 传播至收敛：
    组件内标签收敛到最小像素索引，再按标签计数取面积。mask: (B,H,W) bool。"""
    B, Hh, Ww = mask.shape
    idx = torch.arange(B * Hh * Ww, device=mask.device, dtype=torch.float32).view(B, 1, Hh, Ww)
    big = torch.full_like(idx, float(B * Hh * Ww + 1))
    m4 = mask.unsqueeze(1)
    lab = torch.where(m4, idx, big)
    for _ in range(max_iters):
        new = torch.where(m4, -F.max_pool2d(-lab, 3, 1, 1), big)
        if torch.equal(new, lab):
            break
        lab = new
    flat = lab.view(-1).long()
    sel = mask.view(-1)
    labs = flat[sel]
    uniq, inv, counts = torch.unique(labs, return_inverse=True, return_counts=True)
    small = torch.zeros_like(sel)
    small[sel.nonzero(as_tuple=True)[0]] = counts[inv] <= min_area
    return small.view(B, Hh, Ww)


def _gimp_map_light_t(values: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """_gimp_map_lightness：v = light_all + light_range（含 clip）。"""
    return torch.where(v < 0.0, values * (v + 1.0),
                       values + v * (1.0 - values)).clamp(0.0, 1.0)


def _gimp_hue_ranges_t(hue: torch.Tensor, overlap: float):
    """复现 _resolve_gimp_hue_ranges（7 桶主/次带 + overlap 权重 + 红带回绕）。"""
    hue = torch.remainder(hue, 1.0)
    h6 = hue * 6.0
    overlap = float(np.clip(overlap, 0.0, 1.0))
    oh = overlap * 0.5
    prim = torch.zeros_like(hue, dtype=torch.long)
    sec = torch.zeros_like(hue, dtype=torch.long)
    use_sec = torch.zeros_like(hue, dtype=torch.bool)
    p_int = torch.ones_like(hue)
    s_int = torch.zeros_like(hue)
    assigned = torch.zeros_like(hue, dtype=torch.bool)
    for hc in range(7):
        thr = float(hc) + 0.5
        m = (~assigned) & (h6 < (thr + oh))
        prim = torch.where(m, torch.full_like(prim, hc), prim)
        if oh > 0.0:
            sm = m & (h6 > (thr - oh))
            sv = ((h6 - thr + oh) / (2.0 * oh)).clamp(0.0, 1.0)
            use_sec = use_sec | sm
            sec = torch.where(sm, torch.full_like(sec, hc + 1), sec)
            s_int = torch.where(sm, sv, s_int)
            p_int = torch.where(sm, 1.0 - sv, p_int)
        assigned = assigned | m
    wrap_p = prim >= 6
    prim = torch.where(wrap_p, torch.zeros_like(prim), prim)
    sec = torch.where(wrap_p, torch.zeros_like(sec), sec)
    use_sec = use_sec & ~wrap_p
    p_int = torch.where(wrap_p, torch.ones_like(p_int), p_int)
    s_int = torch.where(wrap_p, torch.zeros_like(s_int), s_int)
    sec = torch.where(sec >= 6, torch.zeros_like(sec), sec)
    return prim + 1, sec + 1, use_sec, p_int, s_int


def _gimp_hsl_kernel(img: torch.Tensor, hue_adj: np.ndarray, sat_adj: np.ndarray,
                     light_adj: np.ndarray, overlap: float, stab: dict) -> torch.Tensor:
    """复现 _apply_gimp_hsl_transform（含 stability 链），BCHW batched。
    adj 数组为 (7,)：[0]=all，[1..6]=Red..Magenta 桶。"""
    dev = img.device
    rgb = img.clamp(0.0, 1.0)
    ha = _t(hue_adj, dev)
    sa = _t(sat_adj, dev)
    la = _t(light_adj, dev)

    H, L, S = _rgb_to_hls(rgb)
    chroma = rgb.amax(1) - rgb.amin(1)
    prim, sec, use_sec, p_int, s_int = _gimp_hue_ranges_t(H, overlap)

    all_h, all_s, all_l = ha[0], sa[0], la[0]
    hp, hs2 = ha[prim], ha[sec]
    sp, ss2 = sa[prim], sa[sec]
    lp, ls2 = la[prim], la[sec]

    # secondary（overlap 混合）：hue 权重先混 adj，sat/light 各自 map 后按权重混
    hue_sec = torch.remainder(H + (all_h + hp * p_int + hs2 * s_int) * 0.5, 1.0)
    sat_sec = ((S * (all_s + sp + 1.0)).clamp(0.0, 1.0) * p_int
               + (S * (all_s + ss2 + 1.0)).clamp(0.0, 1.0) * s_int)
    light_sec = (_gimp_map_light_t(L, all_l + lp) * p_int
                 + _gimp_map_light_t(L, all_l + ls2) * s_int)
    # primary-only（chroma 路径）
    hue_pri = torch.remainder(H + (all_h + hp) * 0.5, 1.0)
    sat_pri = (S * (all_s + sp + 1.0)).clamp(0.0, 1.0)
    light_pri = _gimp_map_light_t(L, all_l + lp)
    # achromatic（无 clip，与 numpy 一致）
    light_ach = torch.where(all_l < 0.0, L * (all_l + 1.0), L + all_l * (1.0 - L))

    _ACH = 0.005
    ach = (~use_sec) & (S <= _ACH)
    chr_m = (~use_sec) & (S > _ACH)
    hue_out = torch.where(use_sec, hue_sec, torch.where(chr_m, hue_pri, H))
    sat_out = torch.where(use_sec, sat_sec, torch.where(chr_m, sat_pri, S))
    light_out = torch.where(use_sec, light_sec,
                            torch.where(chr_m, light_pri,
                                        torch.where(ach, light_ach, L)))

    stab_on = isinstance(stab, dict) and stab.get("enabled", True)
    if stab_on:
        conf = (_smoothstep_t(S, stab["sat_floor_low"], stab["sat_floor_high"])
                * _smoothstep_t(chroma, stab.get("chroma_floor_low", 0.006),
                                stab.get("chroma_floor_high", 0.040)))
        hue_delta = torch.remainder(hue_out - H + 0.5, 1.0) - 0.5
        hue_out = torch.remainder(H + hue_delta * conf, 1.0)
        sat_out = (S + (sat_out - S) * conf).clamp(0.0, 1.0)

        lg = stab.get("luma_guard", {})
        if isinstance(lg, dict) and lg.get("enabled", True):
            lconf = (_smoothstep_t(S, lg.get("sat_floor_low", 0.06),
                                   lg.get("sat_floor_high", 0.22))
                     * _smoothstep_t(chroma,
                                     lg.get("chroma_floor_low", stab.get("chroma_floor_low", 0.006)),
                                     lg.get("chroma_floor_high", stab.get("chroma_floor_high", 0.040))))
            delta = (light_out - L) * lconf * float(np.clip(_safe_float(lg.get("strength", 1.0), 1.0), 0.0, 1.0))

            g = lg.get("guided", {})
            if isinstance(g, dict) and g.get("enabled", True):
                # 本 env 无 cv2.ximgproc → numpy 参考走 bilateralFilter 回退分支
                radius = max(1, int(_safe_float(g.get("radius", 5), 5)))
                d = max(3, radius * 2 + 1)
                sigma_space = float(max(1.0, _safe_float(g.get("sigma_space", float(radius)), float(radius))))
                sigma_color = float(max(1e-6, _safe_float(g.get("sigma_color", 0.06), 0.06)))
                delta = _bilateral_cv2_t(delta, d, sigma_color, sigma_space)

            sp_cfg = lg.get("speckle", {})
            if isinstance(sp_cfg, dict) and sp_cfg.get("enabled", True):
                low = ((S <= float(sp_cfg.get("low_sat_threshold", 0.22)))
                       | (chroma <= float(sp_cfg.get("low_chroma_threshold", 0.06))))
                spike = low & (delta.abs() >= float(sp_cfg.get("delta_thresh", 0.06)))
                if bool(spike.any()):
                    small = _small_component_mask(spike, max(1, int(sp_cfg.get("min_area", 6))))
                    med = _median_filter_t(delta.unsqueeze(1), int(sp_cfg.get("kernel", 3)))[:, 0]
                    blend = float(np.clip(_safe_float(sp_cfg.get("blend", 0.85), 0.85), 0.0, 1.0))
                    mixed = med if blend >= 1.0 else (1.0 - blend) * delta + blend * med
                    delta = torch.where(small, mixed, delta)

            light_out = (L + delta).clamp(0.0, 1.0)

    out = _hls_to_rgb(hue_out, light_out, sat_out)

    if stab_on:
        ag = stab.get("artifact_guard", {})
        if isinstance(ag, dict) and ag.get("enabled", True):
            diff = (out - rgb).abs().mean(1)
            low = ((S <= float(ag.get("low_sat_threshold", stab["sat_floor_high"])))
                   | (chroma <= float(ag.get("low_chroma_threshold", stab.get("chroma_floor_high", 0.040)))))
            spike = low & (diff >= float(ag.get("diff_thresh", 0.10)))
            if bool(spike.any()):
                filt = _median_filter_t(out, max(1, int(ag.get("kernel", 3))))
                blend = float(np.clip(_safe_float(ag.get("blend", 0.85), 0.85), 0.0, 1.0))
                mixed = filt if blend >= 1.0 else (1.0 - blend) * out + blend * filt
                out = torch.where(spike.unsqueeze(1), mixed, out)

    return out.clamp(0.0, 1.0)


def _apply_scalar_post_luts(out: torch.Tensor, op: str, lr_v: float, fit: dict) -> torch.Tensor:
    """复现 replay.apply_scalar_op 的核后校正：post_luma_lut 相邻扫描值线性混合 +
    post_rgb_lut 最近扫描值每通道 1D LUT。这 15 个色带的 fits 目前均无 LUT（自然直通），
    保留完整语义以防 fits 精调后补充。"""
    from gpu_render.replay import _lut_at
    dev = out.device
    vals = [float(v) for v in _SWEEP_OPS[op]["values"]]
    pts = _lut_at(fit.get("post_luma_lut") or {}, vals, lr_v)
    if pts:
        xs, ys = zip(*sorted(pts))
        luma_w = torch.as_tensor([0.2126, 0.7152, 0.0722], device=dev,
                                 dtype=out.dtype).view(1, 3, 1, 1)
        luma = (out * luma_w).sum(1)
        gain = _interp1d(luma, _t(xs, dev), _t(ys, dev)) - luma
        out = (out + gain.unsqueeze(1)).clamp(0.0, 1.0)
    rgb_luts = fit.get("post_rgb_lut") or {}
    if rgb_luts and vals:
        nearest = min(vals, key=lambda gv: abs(gv - lr_v))
        chl = (rgb_luts.get(fmt_value(nearest))
               or rgb_luts.get(str(int(nearest)) if float(nearest).is_integer() else str(nearest)))
        if chl and abs(nearest - lr_v) <= (max(vals) - min(vals)):
            chans = []
            for i, ch in enumerate(("r", "g", "b")):
                p = chl.get(ch)
                if p:
                    xs, ys = zip(*sorted(p))
                    chans.append(_interp1d(out[:, i], _t(xs, dev), _t(ys, dev)))
                else:
                    chans.append(out[:, i])
            out = torch.stack(chans, dim=1).clamp(0.0, 1.0)
    return out


# torch.compile 融合 elementwise 链：实测 11.8×（292→25 ms/batch16 全分辨率, 2026-07-05）。
# 与 eager 的差异仅桶带/阈值边界像素的浮点比较翻转（0.56% 像素、ΔE00≈0.014），
# 远低于 0.5 parity 门。MONETGPT_GPU_COMPILE=0 回 eager（逐位复现 numpy 参考）。
if os.environ.get("MONETGPT_GPU_COMPILE", "1") != "0":
    _gimp_hsl_kernel = torch.compile(_gimp_hsl_kernel, dynamic=True)


def _make_rare_band(op_name: str, color: str, kind: str):
    """kind ∈ {HueAdjustment, SaturationAdjustment, LuminanceAdjustment}。"""
    def fn(img: torch.Tensor, ctx: dict) -> torch.Tensor:
        lr_v = float((ctx.get("attrs") or {}).get(op_name, ctx.get("label", 0.0) or 0.0))
        if lr_v == 0.0:
            return img
        fit = ctx.get("fit") or {}
        local_v = local_value(op_name, lr_v, fit)   # host：value_map 插值（无 fit 时恒等）
        # host 端复现 execute_hsl 的 adj 数组 / overlap / stability 解析
        all_cfg = get_hsl_config()
        adj = {"HueAdjustment": 0.0, "SaturationAdjustment": 0.0, "LuminanceAdjustment": 0.0}
        adj[kind] = local_v
        hue_adj, sat_adj, light_adj = _build_all_hsl_adjustments(all_cfg, {color: adj})
        if (np.max(np.abs(hue_adj)) < 1e-8 and np.max(np.abs(sat_adj)) < 1e-8
                and np.max(np.abs(light_adj)) < 1e-8):
            out = img
        else:
            first_cfg = next(iter(all_cfg.values()), {})
            overlap = float(np.clip(_safe_float(first_cfg.get("overlap_default", 0.0), 0.0), 0.0, 1.0))
            stab = _resolve_hsl_stability_cfg(first_cfg, stability_override=None)
            out = _gimp_hsl_kernel(img, hue_adj, sat_adj, light_adj, overlap, stab)
        return _apply_scalar_post_luts(out, op_name, lr_v, fit)

    fn.__name__ = f"gpu_{op_name}"
    return fn


# ===========================================================================
# 注册表
# ===========================================================================
OPS_GPU: dict = {}
for _op in ("HueAdjustmentBlue", "SaturationAdjustmentBlue", "LuminanceAdjustmentBlue"):
    OPS_GPU[_op] = _make_blue(_op)
for _pfx, _kind in (("HueAdjustment", "hue"), ("SaturationAdjustment", "sat"),
                    ("LuminanceAdjustment", "lum")):
    for _color in ("Orange", "Purple"):
        OPS_GPU[f"{_pfx}{_color}"] = _make_orange_purple(f"{_pfx}{_color}", _color.lower(), _kind)
for _pfx in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
    for _color in ("Red", "Yellow", "Green", "Aqua", "Magenta"):
        OPS_GPU[f"{_pfx}{_color}"] = _make_rare_band(f"{_pfx}{_color}", _color.lower(), _pfx)
OPS_GPU["BlackWhite"] = _bw_fn
for _op in ("CalibRedHue", "CalibRedSaturation", "CalibGreenHue", "CalibGreenSaturation",
            "CalibBlueHue", "CalibBlueSaturation", "CalibShadowTint"):
    OPS_GPU[_op] = _make_calib(_op)

__all__ = ["OPS_GPU"]
