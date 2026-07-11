"""Local edit 的 GPU 镜像（torch, BCHW）。语义/数值与 local_replay.py 逐行对齐：
α 光栅同一套数学（torch 版），Local* 算子走 gpu.REGISTRY（fits 已内建），
未覆盖算子按 replay_batch 的约定 CPU 回退。out = out·(1-α) + edited·α。"""
from __future__ import annotations

import math

import numpy as np
import torch

from gpu_render.local_replay import (
    _EPS, _GMAX, _GMIN, LOCAL_OP_MAP, _f, _smoothstep, _use_smooth_gain)


def raster_alpha_t(mask_type: str, geom: dict, h: int, w: int,
                   device, *, smoothstep: bool = False) -> torch.Tensor:
    """Torch mirror of ``local_replay.raster_alpha`` on ``device``."""
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32),
                            indexing="ij")
    x, y = xx / w, yy / h
    if mask_type == "circulargradient":
        cx = (_f(geom, "Left") + _f(geom, "Right")) / 2
        cy = (_f(geom, "Top") + _f(geom, "Bottom")) / 2
        rx = max(abs(_f(geom, "Right") - _f(geom, "Left")) / 2, 1e-3)
        ry = max(abs(_f(geom, "Bottom") - _f(geom, "Top")) / 2, 1e-3)
        ang = math.radians(_f(geom, "Angle"))
        xr = (x - cx) * math.cos(ang) + (y - cy) * math.sin(ang)
        yr = -(x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)
        d = torch.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        feather = max(_f(geom, "Feather", 50) / 100.0, 0.05)
        m = ((d - 1.0) / feather + 0.5).clamp(0.0, 1.0)
    else:
        zx, zy = _f(geom, "ZeroX"), _f(geom, "ZeroY")
        fx, fy = _f(geom, "FullX", 1), _f(geom, "FullY")
        dxv, dyv = fx - zx, fy - zy
        L2 = dxv * dxv + dyv * dyv + 1e-6
        m = (((x - zx) * dxv + (y - zy) * dyv) / L2).clamp(0.0, 1.0)
    if smoothstep:
        m = _smoothstep(m)
    if str(geom.get("Flipped", "false")).lower().lstrip("+") == "true":
        m = 1.0 - m
    return m


def smooth_gain_t(base: torch.Tensor, edited: torch.Tensor,
                  alpha: torch.Tensor) -> torch.Tensor:
    """local_replay.smooth_gain_np 的 torch 版:线性域 α 调制平滑增益。base/edited:(B,3,H,W)。"""
    from gpu_render.gpu.spatial import _gaussian, _linear_to_srgb, _srgb_to_linear
    _, _, h, w = base.shape
    sigma = max(2.0, min(h, w) * 0.015)
    lin_b = _srgb_to_linear(base)
    lin_e = _srgb_to_linear(edited)
    bb = _gaussian(lin_b, sigma)
    be = _gaussian(lin_e, sigma)
    G = ((be + _EPS) / (bb + _EPS)).clamp(_GMIN, _GMAX)
    a = alpha[None, None]
    out_lin = lin_b * (1.0 + a * (G - 1.0))
    return _linear_to_srgb(out_lin.clamp(0.0, 1.0))


def corr_alpha_t(corr: dict, h: int, w: int, device) -> torch.Tensor:
    if corr.get("alpha") is not None:
        a = torch.as_tensor(np.asarray(corr["alpha"], np.float32), device=device)
        if a.shape != (h, w):
            a = torch.nn.functional.interpolate(
                a[None, None], size=(h, w), mode="bilinear",
                align_corners=False)[0, 0]
    else:
        ms = [raster_alpha_t(m["mask_type"], m["geom"], h, w, device)
              for m in corr["masks"]]
        a = torch.stack(ms).amax(0) if ms else torch.zeros((h, w), device=device)
    return (a * float(corr.get("amount", 1.0))).clamp(0.0, 1.0)


def apply_locals_batch(out: torch.Tensor, corrections: list, fits_dir: str,
                       fallback: str = "cpu") -> tuple:
    """镜像 local_replay.apply_locals。out: (B,3,H,W)。返回 (out, fallback_ops)。"""
    from gpu_render.gpu import REGISTRY as GPU_REG
    from gpu_render.local_apply import load_fit
    from gpu_render.replay import apply_scalar_op
    from gpu_render.sweeps import OPS

    fb_ops: list = []
    _, _, h, w = out.shape

    def scalar(img: torch.Tensor, op: str, lr_v: float) -> torch.Tensor:
        fn = GPU_REG.get(op)
        if fn is not None:
            ctx = {"label": f"{lr_v:+g}", "attrs": {OPS[op]["crs"]: lr_v},
                   "elements": "", "fit": load_fit(op, fits_dir)}
            return fn(img, ctx).clamp(0.0, 1.0)
        if fallback != "cpu":
            return img
        fb_ops.append(op)
        from gpu_render.gpu.gpu_replay import _apply_cfg_np
        arr = img.detach().cpu().numpy()
        fit = load_fit(op, fits_dir)
        res = np.empty_like(arr)
        for b in range(arr.shape[0]):
            hwc = np.ascontiguousarray(arr[b].transpose(1, 2, 0))
            o = apply_scalar_op(hwc, op, lr_v, fit, _apply_cfg_np)
            res[b] = np.asarray(o, np.float32).transpose(2, 0, 1)
        return torch.from_numpy(res).to(img.device, torch.float32).clamp(0.0, 1.0)

    for corr in corrections:
        params = corr.get("params") or {}
        ops = sorted(((LOCAL_OP_MAP[k][1], LOCAL_OP_MAP[k][0], v)
                      for k, v in params.items() if k in LOCAL_OP_MAP and v),
                     key=lambda t: t[0])
        if not ops:
            continue
        alpha = corr_alpha_t(corr, h, w, out.device)
        if float(alpha.max()) <= 0.0:
            continue
        edited = out.clone()
        for _, op, v in ops:
            edited = scalar(edited, op, v)
        if _use_smooth_gain(corr):
            out = smooth_gain_t(out, edited, alpha)
        else:
            a = alpha[None, None]       # (1,1,H,W) 广播整个 batch
            out = (out * (1.0 - a) + edited * a).clamp(0.0, 1.0)
    return out, fb_ops
