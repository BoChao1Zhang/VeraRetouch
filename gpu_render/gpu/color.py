"""GPU (torch, cuda:1) 移植：color 组算子 —— Temperature / AbsTemperature / Tint /
Vibrance / Saturation。复现 tools/lr_calib/ops_v2 的 numpy 标定输出（含 fits 校正）。

约定：fn(img_bchw_float01, ctx) -> 同形 BCHW 张量，运行于输入张量所在设备（cuda:1）。
ctx 与 ops_v2 一致：{"label","attrs","elements","fit"}。支持 batch 维（并行渲染关键），
Temperature 的逐图白点自适应在 batch 内 per-sample 计算。

标定参数（temperature wb_cat_v3 模型、vibrance v2_params 节点、tint/sat value_map）
直接复用 image_ops.non_gimp_ops / ops_v2 的 numpy 常量，保证与参考管线逐值一致；
所有 per-pixel 数学在 torch/GPU 上重写。skimage Lab (D65/2) 在 torch 上精确复刻。
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch

# --- 复用 numpy 参考的标定常量（只读；per-pixel 数学在下面用 torch 重写） ---
from gpu_render.image_ops.non_gimp_ops import (
    M_PP2SRGB,
    M_SRGB2PP,
    Y_PP,
    _interp_model,
    _lr_temperature_model,
    _RGB_LUMA,
    _VIB_C_NODES,
    _VIB_DEFAULTS,
    _VIB_H_NODES,
    _VIB_L_NODES,
    _vib_row,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
_FITS = os.path.join(os.path.dirname(_HERE), "fits")

# skimage rgb2lab/lab2rgb 常量（sRGB, D65/2 观察者）
_XYZ_FROM_RGB = np.array([[0.412453, 0.35758, 0.180423],
                          [0.212671, 0.71516, 0.072169],
                          [0.019334, 0.119193, 0.950227]], dtype=np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_REF_WHITE = np.array([0.95047, 1.0, 1.08883], dtype=np.float64)

_LUMA_709 = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)

# Tint value_map 缺省（镜像 ops_v2/tint._DEFAULT_VM；仅在 fit 缺失时用）
_TINT_DEFAULT_VM = [[-100, -74], [-35, -26], [-18, -12], [-10, -5], [-8, -4], [-5, -2],
                    [0, 0], [5, 2], [8, 4], [10, 5], [18, 12], [35, 26], [100, 74]]

_JSON_CACHE: dict = {}
_TENSOR_CACHE: dict = {}


def _load_json(name: str) -> dict:
    p = os.path.join(_FITS, name)
    if p in _JSON_CACHE:
        return _JSON_CACHE[p]
    d = {}
    if os.path.exists(p):
        try:
            d = json.load(open(p))
        except Exception:  # noqa: BLE001
            d = {}
    _JSON_CACHE[p] = d
    return d


def _const(key: str, np_arr, device, dtype):
    """按 (key, device, dtype) 缓存常量张量，避免每帧 host->device 拷贝。"""
    k = (key, str(device), str(dtype))
    t = _TENSOR_CACHE.get(k)
    if t is None:
        t = torch.as_tensor(np.asarray(np_arr), device=device, dtype=dtype)
        _TENSOR_CACHE[k] = t
    return t


def _t(np_arr, device, dtype):
    return torch.as_tensor(np.asarray(np_arr), device=device, dtype=dtype)


# ----------------------------------------------------------------------------
# 共享 torch 基元
# ----------------------------------------------------------------------------
def _interp1d(x, xp, fp):
    """np.interp 等价（端点钳位）。xp 递增 1D；x 任意形状；同 device/dtype。"""
    idx = torch.searchsorted(xp, x.contiguous(), right=True).clamp(1, xp.numel() - 1)
    x0 = xp[idx - 1]
    x1 = xp[idx]
    y0 = fp[idx - 1]
    y1 = fp[idx]
    denom = x1 - x0
    t = (x - x0) / torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    y = y0 + t * (y1 - y0)
    y = torch.where(x <= xp[0], fp[0], y)
    y = torch.where(x >= xp[-1], fp[-1], y)
    return y


def _srgb_to_lin(x):
    return torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def _lin_to_srgb(x):
    x = x.clamp(0.0, 1.0)
    return torch.where(x > 0.0031308, 1.055 * x ** (1.0 / 2.4) - 0.055, 12.92 * x)


def _rgb_to_lab(rgb):
    """skimage-exact rgb2lab；rgb 末轴=通道，[0,1]。"""
    dev, dt = rgb.device, rgb.dtype
    M = _const("xyz_from_rgb_T", _XYZ_FROM_RGB.T, dev, dt)
    rw = _const("ref_white", _REF_WHITE, dev, dt)
    lin = torch.where(rgb > 0.04045, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    arr = (lin @ M) / rw
    arr = torch.where(arr > 0.008856, torch.clamp(arr, min=0.0) ** (1.0 / 3.0),
                      7.787 * arr + 16.0 / 116.0)
    x, y, z = arr[..., 0], arr[..., 1], arr[..., 2]
    return torch.stack([116.0 * y - 16.0, 500.0 * (x - y), 200.0 * (y - z)], dim=-1)


def _lab_to_rgb(lab):
    """skimage-exact lab2rgb（z<0 钳零）；返回未 clip 的 rgb，末轴=通道。"""
    dev, dt = lab.device, lab.dtype
    Mi = _const("rgb_from_xyz_T", _RGB_FROM_XYZ.T, dev, dt)
    rw = _const("ref_white", _REF_WHITE, dev, dt)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    y = (L + 16.0) / 116.0
    x = a / 500.0 + y
    z = torch.clamp(y - b / 200.0, min=0.0)
    arr = torch.stack([x, y, z], dim=-1)
    arr = torch.where(arr > 0.2068966, arr ** 3, (arr - 16.0 / 116.0) / 7.787)
    rgb = (arr * rw) @ Mi
    return torch.where(rgb > 0.0031308,
                       1.055 * torch.clamp(rgb, min=0.0) ** (1.0 / 2.4) - 0.055,
                       12.92 * rgb)


def _hue_interp(h_deg, vals_t, hnodes_t):
    """周期色相剖面插值（节点 _VIB_H_NODES，360° 回绕）。vals_t: (12,) 张量。"""
    xp = torch.cat([hnodes_t, hnodes_t[:1] * 0 + 360.0])
    yp = torch.cat([vals_t, vals_t[:1]])
    return _interp1d(torch.remainder(h_deg, 360.0), xp, yp)


def _fnum(attrs: dict, key: str, default=0.0) -> float:
    try:
        return float(str(attrs.get(key, default)).lstrip("+"))
    except (TypeError, ValueError):
        return float(default)


# ----------------------------------------------------------------------------
# Saturation
# ----------------------------------------------------------------------------
def _sat_pct(lr_v: float, fit: dict) -> float:
    vm = fit.get("value_map")
    if vm:
        xs, ys = zip(*sorted(vm))
        return float(np.interp(lr_v, xs, ys))
    return float(lr_v)


def _apply_comp(out, comp: dict):
    """组合残差后校正（post_luma_lut + 逐通道 post_rgb_lut）；BHWC 张量。"""
    if not comp:
        return out
    dev, dt = out.device, out.dtype
    pts = comp.get("post_luma_lut")
    if pts:
        xs, ys = zip(*sorted(pts))
        w = _const("luma709", _LUMA_709, dev, dt)
        luma = out @ w
        gain = _interp1d(luma, _t([x for x in xs], dev, dt), _t([y for y in ys], dev, dt)) - luma
        out = (out + gain.unsqueeze(-1)).clamp(0.0, 1.0)
    ch = comp.get("post_rgb_lut")
    if ch:
        chans = list(out.unbind(-1))
        for i, c in enumerate(("r", "g", "b")):
            p = ch.get(c)
            if p:
                xs, ys = zip(*sorted(p))
                chans[i] = _interp1d(chans[i], _t([x for x in xs], dev, dt),
                                     _t([y for y in ys], dev, dt))
        out = torch.stack(chans, dim=-1).clamp(0.0, 1.0)
    return out


def _saturation(img, ctx: dict):
    attrs = ctx.get("attrs") or {}
    raw = attrs.get("Saturation", ctx.get("label", 0))
    try:
        lr_v = float(str(raw).lstrip("+"))
    except (TypeError, ValueError):
        return img.clamp(0.0, 1.0)
    if abs(lr_v) < 1e-8:
        return img.clamp(0.0, 1.0)
    fit = ctx.get("fit") or _load_json("Saturation.json")
    pct = _sat_pct(lr_v, fit)
    scale = max(0.0, 1.0 + pct / 100.0)

    x = img.movedim(1, -1).clamp(0.0, 1.0)          # BHWC
    lab = _rgb_to_lab(x)
    a = (lab[..., 1] * scale).clamp(-128.0, 127.0)
    b = (lab[..., 2] * scale).clamp(-128.0, 127.0)
    rgb = _lab_to_rgb(torch.stack([lab[..., 0], a, b], dim=-1)).clamp(0.0, 1.0)
    rgb = _apply_comp(rgb, _load_json("Saturation__comp.json"))
    return rgb.movedim(-1, 1)


# ----------------------------------------------------------------------------
# Vibrance（LR-faithful；Lab 色度乘性增益 + 肤色带 + 负向灰目标）
# ----------------------------------------------------------------------------
def _vibrance(img, ctx: dict):
    attrs = ctx.get("attrs") or {}
    pct = float(np.clip(float(attrs.get("Vibrance", ctx.get("label", 0.0))), -100.0, 100.0))
    x = img.movedim(1, -1).clamp(0.0, 1.0)          # BHWC
    if abs(pct) < 1e-8:
        return x.movedim(-1, 1)
    dev, dt = x.device, x.dtype
    p = _VIB_DEFAULTS

    lab = _rgb_to_lab(x)
    a, b = lab[..., 1], lab[..., 2]
    chroma = torch.hypot(a, b)
    hue = torch.rad2deg(torch.atan2(b, a))
    C = _const("vib_C", _VIB_C_NODES, dev, dt)
    H = _const("vib_H", _VIB_H_NODES, dev, dt)
    Ln = _const("vib_L", _VIB_L_NODES, dev, dt)

    if pct > 0.0:
        pp = p["pos"]
        k_nodes = _vib_row(pct, pp["amts"], pp["k"], np.zeros(len(_VIB_C_NODES)))
        mh_nodes = _vib_row(pct, pp["amts"], pp["mh"], pp["mh"][0])
        dl_nodes = _vib_row(pct, pp["amts"], pp["dl"], np.zeros(len(_VIB_H_NODES)))
        ramp = _vib_row(pct, pp["amts"], pp["dl_ramp"], pp["dl_ramp"][0])
        ml_nodes = _vib_row(pct, pp["amts"], pp["ml"], pp["ml"][0])

        k = _interp1d(chroma, C, _t(k_nodes, dev, dt))
        mh = _hue_interp(hue, _t(mh_nodes, dev, dt), H)
        mh = mh * _interp1d(lab[..., 0], Ln, _t(ml_nodes, dev, dt))
        gain = 1.0 + torch.clamp(k * mh, min=0.0)
        a2 = a * gain
        b2 = b * gain
        c0 = float(ramp[0])
        c1 = float(max(ramp[1], ramp[0] + 1e-3))
        tt = ((chroma - c0) / (c1 - c0)).clamp(0.0, 1.0)
        w = tt * tt * (3.0 - 2.0 * tt)
        dl = _hue_interp(hue, _t(dl_nodes, dev, dt), H)
        L2 = (lab[..., 0] + dl * w).clamp(0.0, 100.0)
        out = _lab_to_rgb(torch.stack([L2, a2, b2], dim=-1)).clamp(0.0, 1.0)
        return out.movedim(-1, 1)

    # negative branch（RGB 域灰目标插值）
    pn = p["neg"]
    amt = -pct
    r_nodes = _vib_row(amt, pn["amts"], pn["r"], np.ones(len(_VIB_C_NODES)))
    beta = float(np.interp(amt, [0.0] + list(pn["amts"]), [0.0] + list(pn["beta"])))
    r = _interp1d(chroma, C, _t(r_nodes, dev, dt))
    w709 = _const("luma709", _LUMA_709, dev, dt)
    mean_rgb = x.mean(dim=-1)
    luma = x @ w709
    gray = (1.0 - beta) * mean_rgb + beta * luma
    gs_nodes = _vib_row(amt, pn["amts"], pn["gs"], np.zeros(len(_VIB_H_NODES)))
    gsc = _vib_row(amt, pn["amts"], pn["gsc"], pn["gsc"][0])
    c0 = float(gsc[0])
    c1 = float(max(gsc[1], gsc[0] + 1e-3))
    tt = ((chroma - c0) / (c1 - c0)).clamp(0.0, 1.0)
    wc = 1.0 - tt * tt * (3.0 - 2.0 * tt)
    s = _hue_interp(hue, _t(gs_nodes, dev, dt), H) * wc
    gray = gray + s * (x.amin(dim=-1) - gray)
    out = gray.unsqueeze(-1) + (x - gray.unsqueeze(-1)) * r.unsqueeze(-1)
    return out.clamp(0.0, 1.0).movedim(-1, 1)


# ----------------------------------------------------------------------------
# Tint（native adjust_tint：luma 保持的 RGB 乘性增益）
# ----------------------------------------------------------------------------
def _tint_gain_bchw(img, scalar: float):
    """adjust_tint(img, scalar) 的 GPU 版：scalar 为传给 adjust_tint 的标量。"""
    n = float(np.clip(scalar, -1.0, 1.0))
    tint_units = n * 150.0
    if abs(tint_units) < 1e-8:
        return img
    coeffs = np.array([np.exp(0.18 * n), np.exp(-0.28 * n), np.exp(0.18 * n)],
                      dtype=np.float32)
    coeffs /= max(float(np.dot(coeffs, _RGB_LUMA)), 1e-6)
    dev, dt = img.device, img.dtype
    c = _t(coeffs, dev, dt).view(1, 3, 1, 1)
    return (img.clamp(0.0, 1.0) * c).clamp(0.0, 1.0)


def _tint(img, ctx: dict):
    attrs = ctx.get("attrs") or {}
    raw = attrs.get("IncrementalTint", ctx.get("label", 0.0))
    try:
        v = float(str(raw).lstrip("+"))
    except (TypeError, ValueError):
        v = 0.0
    if abs(v) < 1e-9:
        return img
    vm = (ctx.get("fit") or {}).get("value_map") or _TINT_DEFAULT_VM
    xs, ys = zip(*sorted(vm))
    local = float(np.interp(v, xs, ys))
    return _tint_gain_bchw(img, local / 100.0)


# ----------------------------------------------------------------------------
# Temperature（LR IncrementalTemperature；wb_cat_v3；逐图白点 batch 内 per-sample）
# ----------------------------------------------------------------------------
def _lr_temperature_core(img, value: float):
    """apply_lr_temperature 的 GPU 版。img: BCHW [0,1]。返回 BCHW。"""
    if abs(value) < 1e-9:
        return img
    model = _lr_temperature_model()
    dev, dt = img.device, img.dtype

    grid = _const("temp_grid", model["grid_log10x"], dev, dt)
    u = _const("temp_u", model["u"], dev, dt)
    shoulder = float(model.get("shoulder", 0.0))
    Bmat = _const("temp_B", np.asarray(model["B"]), dev, dt)
    Bi = _const("temp_Bi", np.linalg.inv(np.asarray(model["B"], dtype=np.float64)), dev, dt)
    feat_center = _const("temp_fc", np.asarray(model["feat_center"]), dev, dt)
    M_s2p_T = _const("M_SRGB2PP_T", np.asarray(M_SRGB2PP).T, dev, dt)
    M_p2s_T = _const("M_PP2SRGB_T", np.asarray(M_PP2SRGB).T, dev, dt)
    Ypp = _const("Y_PP", np.asarray(Y_PP), dev, dt)

    M_sh_np, W_np = _interp_model(value, model)     # 逐值标量插值（numpy, 3x3 / 4x3）
    M_sh = _t(M_sh_np, dev, dt)
    W = _t(W_np, dev, dt)

    x = img.movedim(1, -1).clamp(0.0, 1.0)          # BHWC
    B, Hh, Ww, _ = x.shape
    lin = _srgb_to_lin(x)
    pp = torch.clamp(lin @ M_s2p_T, min=1e-7)       # BHWC linear ProPhoto

    # 逐图统计特征（batch 内 per-sample）
    lms = torch.clamp(pp @ Bmat.t(), min=1e-7)      # BHWC
    lum = pp @ Ypp                                  # BHW
    flat_lms = lms.reshape(B, -1, 3)
    flat_lum = lum.reshape(B, -1)
    gw = flat_lms.mean(dim=1)                        # (B,3)
    lo = torch.quantile(flat_lum, 0.60, dim=1, keepdim=True)
    hi = torch.quantile(flat_lum, 0.95, dim=1, keepdim=True)
    sel = (flat_lum >= lo) & (flat_lum <= hi)       # (B,N)
    cnt = sel.sum(dim=1)                            # (B,)
    selm = sel.unsqueeze(-1).to(dt)
    br = (flat_lms * selm).sum(dim=1) / selm.sum(dim=1).clamp(min=1.0)
    br = torch.where((cnt > 50).unsqueeze(-1), br, gw)
    f = torch.stack([torch.log(gw[:, 0] / gw[:, 1]), torch.log(gw[:, 2] / gw[:, 1]),
                     torch.log(br[:, 0] / br[:, 1]), torch.log(br[:, 2] / br[:, 1])],
                    dim=1) - feat_center            # (B,4)
    d = f @ W                                        # (B,3) log 增益
    expd = torch.exp(d)
    # M = Bi @ diag(expd) @ B @ M_sh  （per-sample）
    M_core = (Bi.unsqueeze(0) * expd.unsqueeze(1)) @ Bmat.unsqueeze(0)   # (B,3,3)
    M = M_core @ M_sh.unsqueeze(0)                                       # (B,3,3)

    grid0 = grid[0]
    pp_c = torch.clamp(pp, min=float(10.0 ** float(grid0.item())), max=1.0)
    scene = torch.exp(_interp1d(torch.log10(pp_c), grid, u))             # BHWC
    scene2 = torch.clamp(torch.einsum("bhwj,bij->bhwi", scene, M), min=1e-9)
    u2 = torch.log(scene2)
    if shoulder > 1e-4:
        knee = u[-1] - shoulder
        u2 = torch.where(u2 > knee, knee + shoulder * torch.tanh((u2 - knee) / shoulder), u2)
    pp2 = 10.0 ** _interp1d(u2, u, grid)
    ppM = torch.clamp(torch.einsum("bhwj,bij->bhwi", pp, M), min=0.0)
    dark = pp < (10.0 ** grid0)
    pp2 = torch.where(dark, ppM, pp2)
    out = _lin_to_srgb(pp2 @ M_p2s_T)
    return out.movedim(-1, 1)


def _temperature(img, ctx: dict):
    attrs = ctx.get("attrs") or {}
    v = float(attrs.get("IncrementalTemperature", ctx.get("label", 0.0)))
    return _lr_temperature_core(img, v)


def _abs_temperature(img, ctx: dict):
    attrs = ctx.get("attrs") or {}
    fit = ctx.get("fit") or {}
    if "it_slope" not in fit:                        # 未标定 → 历史 no-op
        return img
    temp_k = _fnum(attrs, "Temperature")
    tint_v = _fnum(attrs, "Tint")
    if temp_k <= 0 and tint_v == 0:
        return img
    it_clip = float(fit.get("it_clip", 60))
    ti_clip = float(fit.get("ti_clip", 30))
    out = img
    if temp_k > 0:
        it = float(np.clip(fit["it_slope"] * temp_k + fit["it_intercept"], -it_clip, it_clip))
        if abs(it) >= 0.5:
            out = _lr_temperature_core(out, it).clamp(0.0, 1.0)
    ti = float(np.clip(fit["ti_slope"] * tint_v + fit["ti_intercept"], -ti_clip, ti_clip))
    if abs(ti) >= 0.5:
        out = _tint_gain_bchw(out, ti / 100.0).clamp(0.0, 1.0)
    return out


OPS_GPU = {
    "Temperature": _temperature,
    "AbsTemperature": _abs_temperature,
    "Tint": _tint,
    "Vibrance": _vibrance,
    "Saturation": _saturation,
}
