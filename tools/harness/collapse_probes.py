"""INF-1 塌陷探针 — Δ_const / Δ_shuffle 评测器 + σ_s 分布统计器.

口径 (PLAN §3 M1–M3):
- Δ_const   = PSNR(s) − PSNR(s_∅)      红线: <0.05 dB 且 loss 仍降 = 已塌陷
- Δ_shuffle = PSNR(s) − PSNR(跨图置换 s) 红线: <0.3 dB 一票否决
- M3: σ_s 贴上界比例                     红线: >80%

接口约定:
- render_fn(image, s) -> out. image/out: HxWx3 RGB float[0,1] 或 uint8; s: 任意 ndarray
  (harness 原样透传, **绝不做任何逐图归一化** — 红线).
- samples: 可迭代 dict {'image':..., 'gt':..., 's':..., 'id':...}; 或用 load_samples()
  从目录装配 (s 目录布局按 INF-5: {img_id}.npy 或 {img_id}_{instr_hash}.npy).
- s_∅ 取法见 NOTES.md 待决策 2: 默认全评测集 s 的逐通道均值广播成常量场;
  模型自带 s_∅ 时用 s_null= 传入 (逐样本 dict 或单个 ndarray).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from metrics import DEFAULT_CAP_DB, psnr_full


# ---------------------------------------------------------------- 样本装配

def load_samples(
    image_dir: str | Path,
    gt_dir: str | Path,
    s_dir: str | Path,
    image_suffix: str = ".in.jpg",
    gt_suffix: str = ".jpg",
) -> list:
    """按共同 stem 从三个目录装配样本列表 (惰性读大文件不必要, 数据量为评测集量级).

    s 目录: {stem}.npy 优先; 否则取 {stem}_*.npy 的第一个 (INF-5 命名含 instr_hash).
    只保留三者齐全的样本; 缺失的打印警告.
    """
    from PIL import Image

    image_dir, gt_dir, s_dir = Path(image_dir), Path(gt_dir), Path(s_dir)
    samples = []
    for img_path in sorted(image_dir.glob(f"*{image_suffix}")):
        stem = img_path.name[: -len(image_suffix)]
        gt_path = gt_dir / f"{stem}{gt_suffix}"
        s_path = s_dir / f"{stem}.npy"
        if not s_path.exists():
            cand = sorted(s_dir.glob(f"{stem}_*.npy"))
            s_path = cand[0] if cand else None
        if not gt_path.exists() or s_path is None:
            print(f"[collapse_probes] skip {stem}: missing gt or s", file=sys.stderr)
            continue
        samples.append(
            {
                "id": stem,
                "image": np.asarray(Image.open(img_path).convert("RGB")),
                "gt": np.asarray(Image.open(gt_path).convert("RGB")),
                "s": np.load(s_path),
            }
        )
    return samples


def _mean_s_null(s_list: list) -> np.ndarray:
    """默认 s_∅: 全评测集 s 的逐通道均值, 广播为常量场 (形状同单个 s).

    对 HxW / HxWxC / 向量 s 通吃: 除最后一维 (通道) 外全部平均; 无通道维则全均值.
    """
    stacked = [np.asarray(s, dtype=np.float64) for s in s_list]
    if stacked[0].ndim >= 2:
        ch_means = np.mean([s.reshape(-1, s.shape[-1]).mean(axis=0) for s in stacked], axis=0)
        return np.broadcast_to(ch_means, stacked[0].shape).copy()
    return np.full_like(stacked[0], float(np.mean([s.mean() for s in stacked])))


def _psnr_of(render_fn, samples, s_for, cap_db: float) -> np.ndarray:
    return np.array(
        [
            psnr_full(render_fn(smp["image"], s), smp["gt"], cap_db=cap_db)
            for smp, s in zip(samples, s_for)
        ]
    )


# ---------------------------------------------------------------- Δ_const / Δ_shuffle

def delta_const(
    render_fn,
    samples,
    s_null=None,
    cap_db: float = DEFAULT_CAP_DB,
) -> dict:
    """M1: Δ_const = mean PSNR(true s) − mean PSNR(s_∅). 逐图 PSNR, 均值相减."""
    samples = list(samples)
    if not samples:
        raise ValueError("empty samples")
    true_s = [smp["s"] for smp in samples]
    if s_null is None:
        null_field = _mean_s_null(true_s)
        null_s = [null_field] * len(samples)
        mode = "dataset_mean"
    elif isinstance(s_null, (list, tuple)):
        null_s, mode = list(s_null), "provided_per_sample"
    else:
        null_s, mode = [s_null] * len(samples), "provided_constant"
    p_true = _psnr_of(render_fn, samples, true_s, cap_db)
    p_null = _psnr_of(render_fn, samples, null_s, cap_db)
    return {
        "delta_const": float(p_true.mean() - p_null.mean()),
        "delta_const_mode": mode,
        "psnr_true": float(p_true.mean()),
        "psnr_null": float(p_null.mean()),
        "per_image_delta": (p_true - p_null).tolist(),
        "n": len(samples),
    }


def delta_shuffle(
    render_fn,
    samples,
    seed: int = 0,
    cap_db: float = DEFAULT_CAP_DB,
) -> dict:
    """M2: Δ_shuffle = mean PSNR(true s) − mean PSNR(跨图置换 s).

    置换 = seeded 随机置换后整体轮移 1 位 => 保证无不动点 (derangement). n≥2.
    要求各样本 s 形状一致 (INF-5 缓存为统一 32×32), 不一致直接报错, 不做静默 resize.
    """
    samples = list(samples)
    if len(samples) < 2:
        raise ValueError("delta_shuffle needs >=2 samples")
    shapes = {np.asarray(s["s"]).shape for s in samples}
    if len(shapes) != 1:
        raise ValueError(f"s shapes differ across images: {shapes}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(samples))
    derangement = np.roll(order, 1)  # order[i] 的 s 给 order[roll] — 无不动点
    assign = np.empty(len(samples), dtype=int)
    assign[order] = derangement
    true_s = [smp["s"] for smp in samples]
    shuf_s = [samples[assign[i]]["s"] for i in range(len(samples))]
    p_true = _psnr_of(render_fn, samples, true_s, cap_db)
    p_shuf = _psnr_of(render_fn, samples, shuf_s, cap_db)
    return {
        "delta_shuffle": float(p_true.mean() - p_shuf.mean()),
        "psnr_true": float(p_true.mean()),
        "psnr_shuffled": float(p_shuf.mean()),
        "per_image_delta": (p_true - p_shuf).tolist(),
        "seed": seed,
        "n": len(samples),
    }


def run_probes(render_fn, samples, s_null=None, seed: int = 0, cap_db: float = DEFAULT_CAP_DB) -> dict:
    """一次跑齐 M1+M2, 返回扁平 dict (可直接并入 metrics.json 的 metrics 节)."""
    samples = list(samples)
    dc = delta_const(render_fn, samples, s_null=s_null, cap_db=cap_db)
    ds = delta_shuffle(render_fn, samples, seed=seed, cap_db=cap_db)
    return {
        "delta_const": dc["delta_const"],
        "delta_const_mode": dc["delta_const_mode"],
        "delta_shuffle": ds["delta_shuffle"],
        "psnr_true": dc["psnr_true"],
        "psnr_null": dc["psnr_null"],
        "psnr_shuffled": ds["psnr_shuffled"],
        "shuffle_seed": seed,
        "n": len(samples),
    }


# ---------------------------------------------------------------- σ_s 分布统计 (M3)

def sigma_stats(
    params: dict,
    sigma_min: float,
    sigma_max: float,
    key: str = "sigma_s",
    edge_frac: float = 0.01,
) -> dict:
    """M3: σ_s 分布统计. params: 参数字典 (含 key -> array-like 的 σ_s 值).

    「贴上界」= σ ≥ σ_max − ε, ε = edge_frac*(σ_max−σ_min) (NOTES.md 假设 8).
    红线: frac_at_upper > 0.80 (PLAN §3 M3).
    """
    if key not in params:
        raise KeyError(f"params has no key '{key}' (available: {list(params)})")
    sig = np.asarray(params[key], dtype=np.float64).ravel()
    if sig.size == 0:
        raise ValueError("empty sigma_s array")
    eps = edge_frac * (sigma_max - sigma_min)
    qs = np.quantile(sig, [0.05, 0.25, 0.50, 0.75, 0.95])
    return {
        "n": int(sig.size),
        "frac_at_upper": float(np.mean(sig >= sigma_max - eps)),
        "frac_at_lower": float(np.mean(sig <= sigma_min + eps)),
        "mean": float(sig.mean()),
        "std": float(sig.std()),
        "q05": float(qs[0]),
        "q25": float(qs[1]),
        "q50": float(qs[2]),
        "q75": float(qs[3]),
        "q95": float(qs[4]),
        "sigma_min": float(sigma_min),
        "sigma_max": float(sigma_max),
        "edge_frac": float(edge_frac),
        "m3_red_line": bool(np.mean(sig >= sigma_max - eps) > 0.80),
    }
