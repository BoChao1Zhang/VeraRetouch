"""INF-1 统一评测 harness — 核心指标.

口径锁定（见 NOTES.md / IMPL_DOSSIER §4.1 条 6 / PLAN §3 M7）:
- 全图 PSNR: round(x*255) 后算 MSE, 10*log10(255^2/mse), 逐图 (batch=1).
- masked PSNR 三分: 掩膜内核 / 边界带 ±band_px（可配, 二值边界双侧欧氏距离 ≤ band_px）/ 掩膜外核.
  三区互斥且并集为全图. 量化口径与全图 PSNR 相同.
- ΔE00: skimage rgb2lab (D65, 默认) -> deltaE_ciede2000, 像素均值.
- SSIM: skimage structural_similarity, round×255 值域, gaussian_weights=True (11x11 高斯窗,
  sigma=1.5), use_sample_covariance=False, data_range=255.
- LPIPS: 可选懒加载 (net='alex'), 输入变换到 [-1,1].

所有图像输入: HxWx3 RGB, float ∈ [0,1] 或 uint8 (自动 /255). 掩膜: HxW 单通道,
float ∈ [0,1] (软边) 或 uint8/bool.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity

DEFAULT_CAP_DB = 100.0
DEFAULT_BAND_PX = 3
DEFAULT_MASK_THR = 0.5

_lpips_cache: dict = {}


# ---------------------------------------------------------------- 基础转换

def to_float01(img: np.ndarray) -> np.ndarray:
    """任意输入统一为 float64 RGB ∈ [0,1]. uint8 -> /255; float 假定已在 [0,1] 并 clip."""
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float64) / 255.0
    return np.clip(img.astype(np.float64), 0.0, 1.0)


def quantize255(img: np.ndarray) -> np.ndarray:
    """round(x*255) 口径的量化 (返回 float64 的整数值, 0..255)."""
    return np.round(to_float01(img) * 255.0)


def to_mask01(mask: np.ndarray) -> np.ndarray:
    """掩膜统一为 float64 ∈ [0,1] 的 HxW. 多通道取第 0 通道."""
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype == np.uint8:
        return mask.astype(np.float64) / 255.0
    if mask.dtype == bool:
        return mask.astype(np.float64)
    return np.clip(mask.astype(np.float64), 0.0, 1.0)


# ---------------------------------------------------------------- PSNR

def _psnr_from_mse(mse: float, cap_db: float) -> float:
    if mse <= 0.0:
        return cap_db
    return min(10.0 * math.log10(255.0 ** 2 / mse), cap_db)


def psnr_full(pred: np.ndarray, gt: np.ndarray, cap_db: float = DEFAULT_CAP_DB) -> float:
    """全图 PSNR, round×255 口径. 完全一致时返回 cap_db (默认 100)."""
    qp, qg = quantize255(pred), quantize255(gt)
    if qp.shape != qg.shape:
        raise ValueError(f"shape mismatch: {qp.shape} vs {qg.shape}")
    mse = float(np.mean((qp - qg) ** 2))
    return _psnr_from_mse(mse, cap_db)


def region_partition(
    mask: np.ndarray,
    band_px: int = DEFAULT_BAND_PX,
    thr: float = DEFAULT_MASK_THR,
) -> dict:
    """软掩膜 -> 三个互斥布尔区域 {'in','band','out'}, 并集=全图.

    band = 距二值化边界欧氏距离 ≤ band_px 的双侧像素;
    in   = 掩膜内且不在 band; out = 掩膜外且不在 band.
    全 True / 全 False 掩膜没有边界, band 为空.
    """
    m = to_mask01(mask) >= thr
    if m.all() or (~m).all():
        band = np.zeros_like(m)
    else:
        # 距离变换: 每个前景像素到最近背景像素的距离 (及反之).
        # cast: 默认参数 (return_distances=True, return_indices=False) 下恒返回
        # ndarray, scipy 类型存根的 tuple/None 分支不可达 — 仅类型收窄, 无逻辑变化.
        d_in = cast(np.ndarray, distance_transform_edt(m))     # 掩膜内像素离掩膜外的距离
        d_out = cast(np.ndarray, distance_transform_edt(~m))   # 掩膜外像素离掩膜内的距离
        band = (m & (d_in <= band_px)) | (~m & (d_out <= band_px))
    return {"in": m & ~band, "band": band, "out": ~m & ~band}


def masked_psnr(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    band_px: int = DEFAULT_BAND_PX,
    thr: float = DEFAULT_MASK_THR,
    cap_db: float = DEFAULT_CAP_DB,
) -> dict:
    """masked PSNR 三分 (PLAN §3 M7). 空区域 -> NaN.

    返回 {'psnr_in','psnr_band','psnr_out','frac_in','frac_band','frac_out'}.
    """
    qp, qg = quantize255(pred), quantize255(gt)
    if qp.shape != qg.shape:
        raise ValueError(f"shape mismatch: {qp.shape} vs {qg.shape}")
    regions = region_partition(mask, band_px=band_px, thr=thr)
    sq = (qp - qg) ** 2
    n_total = float(qp.shape[0] * qp.shape[1])
    out: dict = {}
    for name, reg in regions.items():
        n = int(reg.sum())
        out[f"frac_{name}"] = n / n_total
        if n == 0:
            out[f"psnr_{name}"] = float("nan")
        else:
            mse = float(sq[reg].mean())
            out[f"psnr_{name}"] = _psnr_from_mse(mse, cap_db)
    return out


# ---------------------------------------------------------------- ΔE00 / SSIM / LPIPS

def delta_e00(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> float:
    """CIEDE2000 像素均值. rgb2lab 默认 D65 (skimage 0.25.2, 签名已核实).

    mask 给定时只在 mask>=0.5 的像素上取均值 (布尔/软掩膜均可).
    """
    lab_p = rgb2lab(to_float01(pred))
    lab_g = rgb2lab(to_float01(gt))
    de = deltaE_ciede2000(lab_p, lab_g)
    if mask is not None:
        sel = to_mask01(mask) >= 0.5
        if not sel.any():
            return float("nan")
        return float(de[sel].mean())
    return float(de.mean())


def ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    """SSIM, round×255 值域, 高斯窗 11x11 (sigma=1.5), 总体协方差."""
    qp, qg = quantize255(pred), quantize255(gt)
    score = structural_similarity(
        qp,
        qg,
        data_range=255,
        channel_axis=-1,
        gaussian_weights=True,
        sigma=1.5,
        use_sample_covariance=False,
    )
    # cast: full=False, gradient=False (默认) 下返回标量, tuple 分支不可达 —
    # 仅类型收窄, 口径不变.
    return float(cast(float, score))


def lpips_dist(
    pred: np.ndarray,
    gt: np.ndarray,
    net: str = "alex",
    device: str = "cpu",
) -> float:
    """LPIPS (懒加载, 模块级缓存模型). 输入变换到 [-1,1]. 失败会抛 ImportError/RuntimeError."""
    import torch  # 环境已有

    key = (net, device)
    if key not in _lpips_cache:
        import lpips as _lpips

        model = _lpips.LPIPS(net=net, verbose=False).to(device).eval()
        _lpips_cache[key] = model
    model = _lpips_cache[key]

    def _t(img: np.ndarray) -> "torch.Tensor":
        x = to_float01(img) * 2.0 - 1.0
        return torch.from_numpy(x.transpose(2, 0, 1)[None]).float().to(device)

    with torch.no_grad():
        return float(model(_t(pred), _t(gt)).item())


# ---------------------------------------------------------------- 组合入口

def evaluate_pair(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None = None,
    band_px: int = DEFAULT_BAND_PX,
    thr: float = DEFAULT_MASK_THR,
    cap_db: float = DEFAULT_CAP_DB,
    with_lpips: bool = False,
    lpips_device: str = "cpu",
) -> dict:
    """单图全指标. mask=None 时只出全图指标 (三分为 NaN)."""
    res = {
        "psnr_full": psnr_full(pred, gt, cap_db=cap_db),
        "ssim": ssim(pred, gt),
        "delta_e00": delta_e00(pred, gt),
    }
    if mask is not None:
        res.update(masked_psnr(pred, gt, mask, band_px=band_px, thr=thr, cap_db=cap_db))
        res["delta_e00_in"] = delta_e00(pred, gt, mask=mask)
    else:
        res.update(
            {k: float("nan") for k in ("psnr_in", "psnr_band", "psnr_out")}
        )
    if with_lpips:
        res["lpips"] = lpips_dist(pred, gt, device=lpips_device)
    return res


def evaluate_batch(samples, **kwargs) -> dict:
    """samples: 可迭代 (pred, gt, mask_or_None) 或 dict{'pred','gt','mask'?,'id'?}.

    返回 {'per_image': [...], 'mean': {...}, 'n': int}. 均值为逐图 nanmean (batch=1 口径),
    mean 里附带 n_valid_<key> 记录各指标有效图数.
    """
    per_image = []
    for i, s in enumerate(samples):
        if isinstance(s, dict):
            r = evaluate_pair(s["pred"], s["gt"], s.get("mask"), **kwargs)
            r["id"] = s.get("id", i)
        else:
            pred, gt, mask = (s + (None,))[:3] if len(s) < 3 else s
            r = evaluate_pair(pred, gt, mask, **kwargs)
            r["id"] = i
        per_image.append(r)
    keys = [k for k in per_image[0] if k != "id"] if per_image else []
    mean: dict = {}
    for k in keys:
        vals = np.array([r[k] for r in per_image], dtype=np.float64)
        n_valid = int(np.sum(~np.isnan(vals)))
        mean[k] = float(np.nanmean(vals)) if n_valid else float("nan")
        mean[f"n_valid_{k}"] = n_valid
    return {"per_image": per_image, "mean": mean, "n": len(per_image)}
