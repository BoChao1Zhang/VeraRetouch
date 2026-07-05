"""每 preset 残差 LUT 的 torch 批量应用（cuda:1）。

与 tools/lr_calib/residual.apply_lut 逐位对应（多线性插值 = map_coordinates order=1）。
4D（RGB+局部 luma 上下文）时上下文用可分离高斯模糊，核大小按 cv2 约定
（ksize = 2*round(4*sigma)+1，float 输入），与 numpy 版 parity 由 --parity 验证。

无 OPS_GPU 导出——不是管线算子，由批渲染调用方在整条链后按 preset_id 调用：
    from gpu_render.residual import load_residual
    from gpu_render.gpu.residual_gpu import apply_residual_batch
    res = load_residual(pid)
    if res is not None:
        out = apply_residual_batch(out, *res)
"""
from __future__ import annotations

import numpy as np
import torch


def _gauss_kernel1d(sigma: float, device, dtype):
    r = int(round(4.0 * sigma))
    x = torch.arange(-r, r + 1, device=device, dtype=dtype)
    k = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    return k / k.sum()


def _ctx_map_torch(out_bchw: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) -> (B,H,W) 局部 luma 上下文，对齐 residual._ctx_map。"""
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=out_bchw.device,
                     dtype=out_bchw.dtype).view(1, 3, 1, 1)
    luma = (out_bchw * w).sum(1, keepdim=True)          # B,1,H,W
    sigma = max(4.0, min(out_bchw.shape[-2:]) / 24.0)
    k = _gauss_kernel1d(sigma, out_bchw.device, out_bchw.dtype)
    r = (k.numel() - 1) // 2
    luma = torch.nn.functional.pad(luma, (r, r, r, r), mode="reflect")
    luma = torch.nn.functional.conv2d(luma, k.view(1, 1, 1, -1))
    luma = torch.nn.functional.conv2d(luma, k.view(1, 1, -1, 1))
    return luma[:, 0]


def apply_residual_batch(out_bchw: torch.Tensor, delta: np.ndarray, dims: tuple) -> torch.Tensor:
    """out (B,3,H,W) float01 + delta (prod(dims),3) -> 校正后 (B,3,H,W)。"""
    B, C, H, W = out_bchw.shape
    dev = out_bchw.device
    D = len(dims)
    pts = [out_bchw[:, c] for c in range(3)]            # 各 (B,H,W)
    if D == 4:
        pts.append(_ctx_map_torch(out_bchw))
    dt = torch.from_numpy(np.ascontiguousarray(delta)).to(dev, torch.float32)  # (P,3)
    dims_t = torch.tensor(dims, device=dev, dtype=torch.float32)
    strides = [1] * D
    for d in range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * dims[d + 1]
    i0, f = [], []
    for d in range(D):
        g = pts[d].clamp(0.0, 1.0) * (dims_t[d] - 1)
        idx = g.floor().long().clamp(max=dims[d] - 2)
        i0.append(idx)
        f.append(g - idx.to(g.dtype))
    corr = torch.zeros((B, H, W, 3), device=dev, dtype=torch.float32)
    for k in range(2 ** D):
        w = torch.ones((B, H, W), device=dev, dtype=torch.float32)
        idx = torch.zeros((B, H, W), device=dev, dtype=torch.long)
        for d in range(D):
            b = (k >> d) & 1
            w = w * (f[d] if b else 1.0 - f[d])
            idx = idx + (i0[d] + b) * strides[d]
        corr += w.unsqueeze(-1) * dt[idx.reshape(-1)].reshape(B, H, W, 3)
    return (out_bchw + corr.permute(0, 3, 1, 2)).clamp(0.0, 1.0)


def _parity(n_presets: int = 6) -> None:
    import json
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # VeraRetouch 根（gpu_render 包父目录）
    from gpu_render.residual import RES_DIR, _load, apply_lut, load_residual, _de
    from gpu_render.sweeps import CALIB_ROOT
    id_dir = os.path.join(CALIB_ROOT, "gt", "Identity")
    probe = os.path.join(id_dir, sorted(os.listdir(id_dir))[0])
    img = _load(probe)
    fitted = sorted(f[:-4] for f in os.listdir(RES_DIR) if f.endswith(".npz") and not f.startswith("_"))
    for pid in fitted[:n_presets]:
        res = load_residual(pid)
        ref = apply_lut(img, res[0], res[1])
        t = torch.from_numpy(img.transpose(2, 0, 1)[None]).to("cuda:1")
        got = apply_residual_batch(t, res[0], res[1])[0].cpu().numpy().transpose(1, 2, 0)
        print(f"{pid}: parity ΔE={_de(ref, got):.4f} max|diff|={np.abs(ref-got).max():.5f}")


if __name__ == "__main__":
    _parity()
