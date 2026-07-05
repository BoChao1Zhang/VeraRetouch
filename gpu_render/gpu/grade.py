"""grade 组 GPU 实现（torch / cuda:1）—— SplitToning + ColorGrade 统一 3-way color grade。

复现 ops_v2/colorgrade.py 的 numpy 标定输出（含 fits/SplitToneShadow__comp.json 的
balance_alpha 组合残差校正）。numpy 参考: image_ops.non_gimp_ops.lr_color_grade。

策略
----
lr_color_grade 的所有"标定量"（tint 向量、zone 权重 W(Y)、cap(x)、lum LUT、低秩残差表、
双 wheel 幅度 amp、balance_alpha）都是 **ctx 标量的纯函数、与图像无关**。因此直接调用
non_gimp_ops 里已标定的 numpy 辅助函数把这些小 LUT/向量算出来（保证标定常数逐位一致），
只把**逐像素**部分（srgb<->lin、luma、1-D LUT 插值、逐元素加权求和）搬到 torch/cuda:1，
天然支持 BCHW batch 维（并行的关键）。np.interp 与本文件的 _interp1d 都是分段线性 → 数值
对齐到 float32 精度，deltaE≈0。

7 个算子（SplitToneShadow/Highlight/Balance, ColorGradeMid/Global/Lum/Blending）在 ops_v2
里映射到同一个 _cg_apply，这里同样用一个 fn 注册到 7 个名字。
"""
from __future__ import annotations

import os
import threading

import numpy as np
import torch

import gpu_render.image_ops.non_gimp_ops as N
from gpu_render.ops_v2.colorgrade import _balance_alpha

DEVICE = os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1")

# ---- 网格（device 常量，缓存）----
_YC = torch.from_numpy(np.ascontiguousarray(N._CG_YC)).to(DEVICE)      # (48,) 亮度分箱中心
_XCAP = torch.from_numpy(np.ascontiguousarray(N._CG_XCAP)).to(DEVICE)  # (40,) cap 的 x 分箱
_XC = torch.from_numpy(np.ascontiguousarray(N._CG_XC)).to(DEVICE)      # (40,) lum 的 x 分箱
_LUMA = torch.tensor(N._CG_LUMA.tolist(), dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)

_ZONES = ("shadow", "mid", "highlight", "global")

_PREP_CACHE: dict = {}
_PREP_LOCK = threading.Lock()


def _interp1d(q: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """torch 版 np.interp（xp 单调增；范围外夹取端值）。q 任意形状；xp/fp 1-D。"""
    n = xp.numel()
    idx = torch.searchsorted(xp, q.contiguous(), right=True).clamp(1, n - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    t = (q - x0) / (x1 - x0)
    out = y0 + t * (y1 - y0)
    out = torch.where(q <= xp[0], fp[0], out)
    out = torch.where(q >= xp[-1], fp[-1], out)
    return out


def _srgb2lin(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin2srgb(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055)


def _t(arr) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(np.asarray(arr, dtype=np.float32))).to(DEVICE)


def _terms_to_device(terms) -> list:
    """低秩残差表 terms[c] = [(ay(48,), bx(40,)), ...] → 每项 (device_ay, device_bx)。"""
    return [[(_t(ay), _t(bx)) for (ay, bx) in terms[c]] for c in range(3)]


def _attrs_key(at: dict) -> tuple:
    keys = ("SplitToningShadowHue", "SplitToningShadowSaturation", "ColorGradeShadowLum",
            "SplitToningHighlightHue", "SplitToningHighlightSaturation", "ColorGradeHighlightLum",
            "ColorGradeMidtoneHue", "ColorGradeMidtoneSat", "ColorGradeMidtoneLum",
            "ColorGradeGlobalHue", "ColorGradeGlobalSat", "ColorGradeGlobalLum",
            "SplitToningBalance", "ColorGradeBlending")
    return tuple(round(float(at.get(k, 50.0 if k == "ColorGradeBlending" else 0.0)), 6) for k in keys)


def _prep(ctx: dict) -> dict:
    """把 ctx 的所有标定量（与图像无关）算成 device 小张量 + 标量。带缓存。"""
    at = {k: float(v) for k, v in ctx.get("attrs", {}).items()}
    key = _attrs_key(at)
    hit = _PREP_CACHE.get(key)
    if hit is not None:
        return hit

    shadow = (at.get("SplitToningShadowHue", 0.0), at.get("SplitToningShadowSaturation", 0.0),
              at.get("ColorGradeShadowLum", 0.0))
    highlight = (at.get("SplitToningHighlightHue", 0.0), at.get("SplitToningHighlightSaturation", 0.0),
                 at.get("ColorGradeHighlightLum", 0.0))
    midtone = (at.get("ColorGradeMidtoneHue", 0.0), at.get("ColorGradeMidtoneSat", 0.0),
               at.get("ColorGradeMidtoneLum", 0.0))
    global_ = (at.get("ColorGradeGlobalHue", 0.0), at.get("ColorGradeGlobalSat", 0.0),
               at.get("ColorGradeGlobalLum", 0.0))
    balance = at.get("SplitToningBalance", 0.0)
    blending = at.get("ColorGradeBlending", 50.0)

    zvals = {"shadow": shadow, "mid": midtone, "highlight": highlight, "global": global_}

    sat_terms = []   # 每 active zone: (v(3,), W_lut(48,), cap(40,), mid_w or None)
    lum_terms = []   # 每 l!=0 zone: (g(40,), k)
    for zone in _ZONES:
        h, s, l = zvals[zone]
        if s and s > 0:
            v = np.asarray(N._cg_tint_vec(zone, h, s), dtype=np.float32)          # (3,)
            W_lut = np.asarray(N._cg_zone_weight(zone, N._CG_YC, float(balance), float(blending)),
                               dtype=np.float32)                                   # (48,) 恒等插值→原 W
            cap = np.asarray(N._cg_zone_cap(zone, float(s), float(balance), float(blending)),
                             dtype=np.float32)                                     # (40,)
            mid_w = None
            if zone == "mid" and s > 60.0:
                mid_w = (min(float(s), 100.0) - 60.0) / 40.0
            sat_terms.append({
                "v": torch.tensor(v.tolist(), dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1),
                "W": _t(W_lut), "cap": _t(cap), "mid_w": mid_w})
        if l:
            g = np.asarray(N._CG_LUM[zone + ("+" if l > 0 else "-")], dtype=np.float32)  # (40,)
            k = abs(float(l)) / 60.0
            lum_terms.append({"g": _t(g), "k": k})

    # 双 wheel 复合残差（sat50 标定，按幅度缩放）
    dual = None
    s_sh, s_hi = float(shadow[1] or 0.0), float(highlight[1] or 0.0)
    if s_sh > 0 and s_hi > 0:
        amp = 0.5 * (np.interp(min(s_sh, 100.0), *N._CG_SAT["shadow"])
                     / max(np.interp(50.0, *N._CG_SAT["shadow"]), 1e-6)
                     + np.interp(min(s_hi, 100.0), *N._CG_SAT["highlight"])
                     / max(np.interp(50.0, *N._CG_SAT["highlight"]), 1e-6))
        bal = float(np.clip(balance, -100.0, 100.0))
        bl = float(np.clip(blending, 0.0, 100.0))
        dual = {}
        if bal != 0.0:
            dual["bal"] = (_terms_to_device(N._CG_BAL_R[-70 if bal < 0 else 70]),
                           float(amp) * abs(bal) / 70.0)
        if bl != 50.0:
            dual["blend"] = (_terms_to_device(N._CG_BLEND_R[0 if bl < 50 else 100]),
                             float(amp) * abs(bl - 50.0) / 50.0)
    mid_satd_dev = None
    if any(t.get("mid_w") for t in sat_terms):
        mid_satd_dev = _terms_to_device(N._CG_MID_SATD)

    prep = {"sat": sat_terms, "lum": lum_terms, "dual": dual,
            "mid_satd": mid_satd_dev, "alpha": float(_balance_alpha(balance))}
    with _PREP_LOCK:
        if len(_PREP_CACHE) > 256:
            _PREP_CACHE.clear()
        _PREP_CACHE[key] = prep
    return prep


def _lowrank_add(d: torch.Tensor, y: torch.Tensor, x: torch.Tensor, terms_dev: list, w: float):
    """d[:,c] += w * Σ_k interp(y,YC,ay) * interp(x_c,XCAP,bx)。y,x_c: B1HW。"""
    if w == 0.0:
        return
    for c in range(3):
        acc = torch.zeros_like(y)
        xc = x[:, c:c + 1]
        for ay, bx in terms_dev[c]:
            acc = acc + _interp1d(y, _YC, ay) * _interp1d(xc, _XCAP, bx)
        d[:, c:c + 1] = d[:, c:c + 1] + w * acc


def cg_apply_gpu(img: torch.Tensor, ctx: dict) -> torch.Tensor:
    """统一 3-way color grade（BCHW float32 [0,1] on cuda:1）。复现 ops_v2 _cg_apply。"""
    if img.dim() == 3:
        img = img.unsqueeze(0)
    x = img.to(DEVICE, dtype=torch.float32).clamp(0.0, 1.0)
    p = _prep(ctx)

    y = (x * _LUMA).sum(dim=1, keepdim=True)          # B,1,H,W 亮度
    xl = _srgb2lin(x)
    d = torch.zeros_like(xl)

    for st in p["sat"]:
        Wq = _interp1d(y, _YC, st["W"])               # B,1,H,W
        capx = _interp1d(x, _XCAP, st["cap"])         # B,3,H,W（同一 cap 施于每通道）
        d = d + Wq * st["v"] * capx
        if st["mid_w"] is not None and p["mid_satd"] is not None:
            _lowrank_add(d, y, x, p["mid_satd"], st["mid_w"])

    for lt in p["lum"]:
        gx = _interp1d(xl, _XC, lt["g"])              # B,3,H,W（同一 g 施于每通道）
        d = d + lt["k"] * gx

    if p["dual"] is not None:
        if "bal" in p["dual"]:
            terms, w = p["dual"]["bal"]
            _lowrank_add(d, y, x, terms, w)
        if "blend" in p["dual"]:
            terms, w = p["dual"]["blend"]
            _lowrank_add(d, y, x, terms, w)

    out = _lin2srgb(xl + d)                            # 已 clamp 到 [0,1]
    alpha = p["alpha"]
    if alpha != 1.0:
        out = (x + alpha * (out - x)).clamp(0.0, 1.0)
    return out


OPS_GPU = {op: cg_apply_gpu for op in (
    "SplitToneShadow", "SplitToneHighlight", "SplitToneBalance",
    "ColorGradeMid", "ColorGradeGlobal", "ColorGradeLum", "ColorGradeBlending")}
