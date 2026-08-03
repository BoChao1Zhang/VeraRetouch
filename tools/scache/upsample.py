"""s 场引导上采样（INF-5 / 任务卡 T5）。

32x32 s 场 -> bilinear 到目标分辨率 -> kornia.filters.guided_blur 引导精修，guide=原图。

已核实（kornia==0.8.2 源码 kornia/filters/guided.py）：
    guided_blur(guidance, input, kernel_size, eps, border_type='reflect', subsample=1)
    - guidance/input 均 (B,C,H,W)，batch 与空间尺寸须一致，通道可不同；
    - subsample>1 走 Fast Guided Filter（He 2015），但内部 interpolate 下/上采样
      要求 H、W 能被 subsample 整除，否则形状不匹配 → 本模块先 replicate pad
      到整除，算完裁回。

eps 用归一化域（guide、s 均在 [0,1]）量级 1e-4 ~ 1e-2，默认 1e-3。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from kornia.filters import guided_blur

DEFAULT_EPS = 1e-4
DEFAULT_SUBSAMPLE = 1


def _to_tensor(x: Union[np.ndarray, torch.Tensor], device: torch.device) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x))
    return x.to(device=device, dtype=torch.float32)


def auto_kernel_size(in_hw: Tuple[int, int], out_hw: Tuple[int, int]) -> int:
    """默认核尺寸：约 1/8 个低分辨率单元的窗口（奇数）。

    经验值：T5 自检在 prod-l1 上扫描 k∈{9..97}，小核（~scale/4）IoU 一致更优
    （大核会把小掩膜的质量摊薄到 0.5 阈值以下）；32→1536 时 k=13。
    """
    scale = max(out_hw[0] / in_hw[0], out_hw[1] / in_hw[1])
    k = 2 * int(round(scale / 8)) + 1
    return max(k, 3)


@torch.no_grad()
def upsample_s(
    s: Union[np.ndarray, torch.Tensor],
    guide: Union[np.ndarray, torch.Tensor],
    kernel_size: Optional[int] = None,
    eps: float = DEFAULT_EPS,
    subsample: int = DEFAULT_SUBSAMPLE,
    gray_guide: bool = False,
    clamp: Optional[Tuple[float, float]] = (0.0, 1.0),
    device: Optional[str] = None,
) -> np.ndarray:
    """把低分辨率 s 场上采样到 guide 分辨率并做引导精修。

    Parameters
    ----------
    s     : (h, w) float，低分辨率 s 场（归一化域）。
    guide : (H, W, 3) 或 (H, W) float [0,1]，原图（guide）。
    kernel_size : None = 自动（约一个低分辨率单元）。
    eps   : guided filter 正则；归一化域量级 1e-4 ~ 1e-2。
    subsample : Fast Guided Filter 下采样因子；1 = 精确版。
    gray_guide : True 时把 RGB guide 转灰度（单通道路径，更快）。
    clamp : 输出裁剪范围；None 不裁剪（s 非掩膜语义时用）。
    device: 'cuda' / 'cpu' / None（自动：有 CUDA 用 cuda）。

    Returns
    -------
    np.ndarray float32 (H, W)
    """
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    s_t = _to_tensor(np.asarray(s, dtype=np.float32) if isinstance(s, np.ndarray) else s, dev)
    if s_t.ndim != 2:
        raise ValueError(f"s 须为二维 (h,w)，得到 {tuple(s_t.shape)}")
    g_t = _to_tensor(guide, dev)
    if g_t.ndim == 2:
        g_t = g_t.unsqueeze(-1)
    if g_t.ndim != 3:
        raise ValueError(f"guide 须为 (H,W,C) 或 (H,W)，得到 {tuple(g_t.shape)}")
    g_t = g_t.permute(2, 0, 1).unsqueeze(0)          # 1,C,H,W
    if gray_guide and g_t.shape[1] == 3:
        w = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)
        g_t = (g_t * w).sum(dim=1, keepdim=True)

    H, W = g_t.shape[-2:]
    in_hw = (int(s_t.shape[0]), int(s_t.shape[1]))
    if kernel_size is None:
        kernel_size = auto_kernel_size(in_hw, (H, W))
    if kernel_size % 2 == 0:
        raise ValueError(f"kernel_size 须为奇数: {kernel_size}")

    up = F.interpolate(
        s_t.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )                                                # 1,1,H,W

    pad_h = pad_w = 0
    if subsample > 1:
        pad_h = (-H) % subsample
        pad_w = (-W) % subsample
        if pad_h or pad_w:
            g_t = F.pad(g_t, (0, pad_w, 0, pad_h), mode="replicate")
            up = F.pad(up, (0, pad_w, 0, pad_h), mode="replicate")

    out = guided_blur(g_t, up, kernel_size, eps, subsample=subsample)

    if pad_h or pad_w:
        out = out[..., :H, :W]
    if clamp is not None:
        out = out.clamp(*clamp)
    return out[0, 0].float().cpu().numpy()


def iou_at(a: np.ndarray, b: np.ndarray, thr: float = 0.5) -> float:
    """两归一化掩膜在阈值 thr 下的 IoU；双空记 1.0。"""
    if a.shape != b.shape:
        raise ValueError(f"形状不一致: {a.shape} vs {b.shape}")
    am, bm = a > thr, b > thr
    union = np.logical_or(am, bm).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(am, bm).sum() / union)
