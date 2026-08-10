"""tools/ceiling 自检：合成情形下验证 Δ_ceil 估计器的语义正确。

用法: CUDA_VISIBLE_DEVICES=1 python3 tools/ceiling/selfcheck.py [--device cuda]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from tools.ceiling.delta_ceil import ceiling, confound_stats, s_quantile_buckets
import torch


def _rand_img(n: int, seed: int, palette: int = 400) -> np.ndarray:
    """照片式伪图：小调色板 + 逐像素 iid 采样。

    颜色**空间上独立同分布** → 任何空间掩膜都与颜色统计独立，
    Δ_ceil 才能干净地度量「s 相对颜色的增量信息」（真实照片里颜色与区域相关，
    那份混杂正是本实验要通过 donor/control 档量化的）。
    """
    rng = np.random.default_rng(seed)
    pal = rng.random((palette, 3)).astype(np.float32) * 0.8 + 0.1
    idx = rng.integers(0, palette, n)
    return np.clip(pal[idx] + rng.normal(0, 0.004, (n, 3)).astype(np.float32), 0, 1)


def _lut_a(x):
    return np.clip(x ** 0.8 * 0.9 + 0.05, 0, 1)


def _lut_b(x):
    y = x.copy()
    y[:, 0] = np.clip(x[:, 0] * 1.3, 0, 1)
    y[:, 2] = np.clip(x[:, 2] * 0.6, 0, 1)
    return y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = args.device
    checks: list[dict] = []

    def rec(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    side = 512
    n = side * side
    x = _rand_img(n, 0)
    s_rand = np.random.default_rng(1).random(n).astype(np.float32)
    # 空间连续（上半图）但与颜色独立的掩膜：Δ_ceil 该大，Moran's I 也该高。
    half = np.zeros((side, side), dtype=np.float32)
    half[: side // 2] = 1.0
    half = half.reshape(-1)

    # 1) 纯全局 3D LUT 编辑 + 随机 s: Δ_ceil 应很小（3D 已经解释一切）
    y = _lut_a(x)
    r = ceiling(x, y, s_rand, device=dev)
    rec("global_lut_delta_small", r.delta < 0.5,
        f"psnr_3d={r.psnr_3d:.2f} delta={r.delta:.3f} delta_cv={r.delta_cv:.3f}")

    # 2) 掩膜依赖编辑（两个不同 LUT）+ s=掩膜: Δ_ceil 应很大
    y2 = np.where(half[:, None] > 0.5, _lut_b(x), _lut_a(x))
    r2 = ceiling(x, y2, half, device=dev)
    rec("masked_two_lut_delta_large", r2.delta > 8.0,
        f"psnr_3d={r2.psnr_3d:.2f} psnr_4d={r2.psnr_4d:.2f} delta={r2.delta:.3f}")
    rec("masked_two_lut_delta_cv_large", r2.delta_cv > 8.0,
        f"delta_cv={r2.delta_cv:.3f}")

    # 3) 同样的编辑但 s 换成与掩膜无关的随机场: Δ_ceil 应塌回小值
    r3 = ceiling(x, y2, s_rand, device=dev)
    rec("masked_two_lut_wrong_s_delta_small", r3.delta < r2.delta * 0.25,
        f"delta_wrong_s={r3.delta:.3f} vs delta_gt_s={r2.delta:.3f}")

    # 4) Δ_ceil ≥ 0 恒成立（构造上保证）
    ds = [ceiling(x, _lut_a(x) if i % 2 else y2,
                  s_rand if i % 3 else half, device=dev).delta for i in range(6)]
    rec("delta_nonneg", all(d >= -1e-9 for d in ds), f"deltas={[round(d,4) for d in ds]}")

    # 5) 恒等编辑: Δ_ceil 应≈0（psnr_3d 受**分箱量化地板**限制, 只作诊断不设判据）
    r5 = ceiling(x, x, s_rand, device=dev)
    rec("identity_edit_delta_zero", abs(r5.delta) < 0.5,
        f"psnr_3d(分箱地板)={r5.psnr_3d:.2f} delta={r5.delta:.3f}")

    # 5b) Δ_const 恒等于 0（s≡1 时 4D 分箱与 3D 分箱完全相同）——实现自检
    rc = ceiling(x, y2, np.ones(n, dtype=np.float32), device=dev)
    rec("delta_const_exactly_zero", abs(rc.delta) < 1e-9 and abs(rc.delta_cv) < 1e-9,
        f"delta_const={rc.delta:.3e} delta_const_cv={rc.delta_cv:.3e}")

    # 6) 分位桶: 退化 s（全 0）应只有 1 桶；均匀 s 应有 8 桶
    b0 = s_quantile_buckets(torch.zeros(1000, device=dev))
    b1 = s_quantile_buckets(torch.rand(100000, device=dev))
    rec("s_buckets_degenerate", int(b0.max()) + 1 == 1, f"n_buckets={int(b0.max())+1}")
    rec("s_buckets_uniform", int(b1.max()) + 1 == 8, f"n_buckets={int(b1.max())+1}")

    # 7) 混淆统计: 成簇残差 Moran's I 应显著高于白噪残差
    hw = (512, 512)
    cs_local = confound_stats(x, y2, hw)
    noise = np.clip(x + np.random.default_rng(2).normal(0, 0.02, x.shape).astype(np.float32), 0, 1)
    cs_noise = confound_stats(x, noise, hw)
    rec("moran_local_gt_noise", cs_local["moran_i"] > cs_noise["moran_i"] + 0.3,
        f"local={cs_local['moran_i']:.3f} noise={cs_noise['moran_i']:.3f}")
    rec("blur_drop_noise_high", cs_noise["blur_drop"] > 0.6,
        f"noise_blur_drop={cs_noise['blur_drop']:.3f} local={cs_local['blur_drop']:.3f}")

    n_fail = sum(1 for c in checks if not c["pass"])
    print(json.dumps({"n_checks": len(checks), "n_fail": n_fail}, ensure_ascii=False))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
