"""INF-1 配对统计 — 逐图配对 ΔPSNR + bootstrap 95% CI + 符号检验.

口径 (PLAN §3): 主指标配对报告 = 逐图 ΔPSNR 均值 + percentile bootstrap 95% CI + 符号检验.
符号检验: 双侧精确二项检验 (scipy.stats.binomtest, n = 非零 delta 数).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def paired_delta(psnr_a, psnr_b) -> np.ndarray:
    """逐图配对差 a−b (a=待评方法, b=基线). 长度必须一致, NaN 成对剔除."""
    a = np.asarray(psnr_a, dtype=np.float64)
    b = np.asarray(psnr_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: {a.shape} vs {b.shape}")
    ok = ~(np.isnan(a) | np.isnan(b))
    return (a - b)[ok]


def bootstrap_ci(
    x,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple:
    """均值的 percentile bootstrap CI. 返回 (lo, hi). 全同值时退化为 (v, v)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("empty array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def sign_test(x) -> dict:
    """双侧精确符号检验. 零 delta 按惯例剔除; 全零时 p=1.0."""
    x = np.asarray(x, dtype=np.float64)
    n_pos = int(np.sum(x > 0))
    n_neg = int(np.sum(x < 0))
    n_zero = int(np.sum(x == 0))
    n = n_pos + n_neg
    p = 1.0 if n == 0 else float(binomtest(n_pos, n, p=0.5, alternative="two-sided").pvalue)
    return {"n_pos": n_pos, "n_neg": n_neg, "n_zero": n_zero, "p_value": p}


def compare(
    psnr_a,
    psnr_b,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """完整配对比较报告: ΔPSNR 均值/中位数 + bootstrap CI + 符号检验.

    直接可写进 metrics.json 的 'ci' 节.
    """
    d = paired_delta(psnr_a, psnr_b)
    if d.size == 0:
        raise ValueError("no valid pairs after NaN removal")
    lo, hi = bootstrap_ci(d, n_boot=n_boot, alpha=alpha, seed=seed)
    st = sign_test(d)
    return {
        "delta_psnr_mean": float(d.mean()),
        "delta_psnr_median": float(np.median(d)),
        "ci95": [lo, hi],
        "ci_alpha": alpha,
        "n_boot": n_boot,
        "boot_seed": seed,
        "sign_test": st,
        "n_pairs": int(d.size),
    }
