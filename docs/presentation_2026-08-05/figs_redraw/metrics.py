"""重绘配图的**算数**部分——与着色完全分开（CLAUDE.md：着色归着色、算数归算数）。

口径逐字复用 `RO9c/diag_four_fields.py`（已冻结的 `METRIC_SPEC`）：

- 阈值化 = **面积匹配 top-k**（k = 该源 valid 格内 GT 正格数），纯秩次 ⇒
  对任何单调变换不变，**不需要任何逐图 min-max**，阈值也不再是自由参数。
- soft_iou / hard_iou / bf1_grid（16×16 网格、1 格容差）。
- 中心先验列 `-sqrt((y-7.5)^2+(x-7.5)^2)`：零参数、**完全不看图像**。
  CLAUDE.md 规定「任何『我们的场找到了主体』的主张必须出示配对 Δ」。

⚑ **AUC 全实验禁用**（CLAUDE.md 2026-08-05）。本文件提供 `legacy_auc` 仅用于与
   已落盘的历史数字对表（RO-9b REPORT / G1 REPORT），引用时必须标注
   「判据不完整，结论限于排序」。**不得进任何新判据表、不得上图。**
"""
from __future__ import annotations

import numpy as np

GRID = 16
_yy, _xx = np.mgrid[0:GRID, 0:GRID]
CENTER_PRIOR = (-np.sqrt((_yy - (GRID - 1) / 2.0) ** 2
                         + (_xx - (GRID - 1) / 2.0) ** 2)).astype(np.float32)


def topk_mask(f: np.ndarray, valid: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros(f.shape, dtype=bool)
    idx = np.flatnonzero(valid.ravel())
    if k <= 0 or len(idx) == 0:
        return out
    k = min(k, len(idx))
    out.ravel()[idx[np.argsort(-f.ravel()[idx], kind="stable")][:k]] = True
    return out


def soft_iou(pred: np.ndarray, m_soft: np.ndarray, valid: np.ndarray) -> float:
    p = pred[valid].astype(np.float64)
    m = np.clip(m_soft[valid].astype(np.float64), 0, 1)
    den = np.maximum(p, m).sum()
    return float(np.minimum(p, m).sum() / den) if den > 0 else float("nan")


def hard_iou(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray) -> float:
    p, g = pred[valid], gt[valid]
    den = (p | g).sum()
    return float((p & g).sum() / den) if den > 0 else float("nan")


def _boundary(b: np.ndarray) -> np.ndarray:
    p = np.pad(b, 1, constant_values=False)
    nb = (p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return p[1:-1, 1:-1] & ~nb


def bf1_grid(pred: np.ndarray, gt: np.ndarray, r: int = 1) -> float:
    """grid 级边界 F1（16×16，1 格容差）。禁用像素级 3px 版（CLAUDE.md）。"""
    from scipy.ndimage import distance_transform_edt

    bg, bp = _boundary(gt), _boundary(pred)
    if not bg.any() or not bp.any():
        return float("nan")
    dg, dp = distance_transform_edt(~bg), distance_transform_edt(~bp)
    prec, rec = float((dg[bp] <= r).mean()), float((dp[bg] <= r).mean())
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def score(field: np.ndarray, m_soft: np.ndarray, valid: np.ndarray) -> dict:
    """一个场对一个（软）目标区域的三列判据。"""
    gt = (m_soft >= 0.5) & valid
    k = int(gt.sum())
    pk = topk_mask(field, valid, k)
    return {"k": k, "soft_iou": soft_iou(pk, m_soft, valid),
            "hard_iou": hard_iou(pk, gt, valid), "bf1_grid": bf1_grid(pk, gt)}


def legacy_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """⚠️ 历史对表专用（判据不完整，结论限于排序）。禁止进新判据表、禁止上图。"""
    from scipy.stats import rankdata

    s = np.asarray(scores, dtype=np.float64).ravel()
    y = np.asarray(labels).ravel().astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def pearson_valid(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    x, y = a[valid].ravel().astype(np.float64), b[valid].ravel().astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank_pct(f: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """valid 格内的秩次百分位（0–1），补边格 NaN。用于"两场差异有多大"的秩次口径。"""
    from scipy.stats import rankdata

    out = np.full(f.shape, np.nan, dtype=np.float64)
    sel = valid.ravel()
    out.ravel()[sel] = (rankdata(f.ravel()[sel]) - 1) / max(1, int(sel.sum()) - 1)
    return out
