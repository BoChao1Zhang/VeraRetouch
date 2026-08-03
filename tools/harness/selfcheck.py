"""INF-1 合成自检 — 已知答案对拍 (任务卡 T3 判据).

构造: 常量底图 + 掩膜内 +Δ 偏移 (全部取值在 1/255 整数格上, round×255 量化零损失),
对每个指标用**独立解析公式**算期望值, 与实现对拍到 1e-6 (恒等类检查到 1e-12).

运行: /home/bc/miniconda3/bin/python3 tools/harness/selfcheck.py
退出码 0 = 全部通过.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bake_consistency as bake
import metrics
import stats
from collapse_probes import delta_const, delta_shuffle, sigma_stats
from leaderboard import build_leaderboard

TOL = 1e-6
TOL_EXACT = 1e-12
PASSED, FAILED = [], []


def check(name: str, got: float, want: float, tol: float = TOL) -> None:
    ok = (math.isnan(want) and math.isnan(got)) or abs(got - want) <= tol
    (PASSED if ok else FAILED).append(name)
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}: got={got!r} want={want!r} tol={tol:g}")


def check_true(name: str, cond: bool) -> None:
    (PASSED if cond else FAILED).append(name)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


# ================================================================ 合成场景
# 64x64 常量底图 a=100/255; 掩膜=矩形 [16,48)x[16,48); pred = gt + 10/255 (掩膜内).
H = W = 64
R0, R1, C0, C1 = 16, 48, 16, 48
A, D = 100, 10  # /255 单位
BAND = 2

gt = np.full((H, W, 3), A / 255.0)
pred = gt.copy()
pred[R0:R1, C0:C1] += D / 255.0
mask = np.zeros((H, W), dtype=np.float64)
mask[R0:R1, C0:C1] = 1.0


def analytic_regions() -> dict:
    """独立解析几何: 矩形掩膜的三分区布尔图 (闭式距离公式, 不调 region_partition).

    掩膜内像素到背景的欧氏距离 = min(轴向距离) (背景四面包围矩形);
    掩膜外像素到掩膜的欧氏距离 = sqrt(clamp_dr^2 + clamp_dc^2) (点到矩形距离).
    """
    inside = np.zeros((H, W), bool)
    band = np.zeros((H, W), bool)
    for i in range(H):
        for j in range(W):
            in_m = R0 <= i < R1 and C0 <= j < C1
            if in_m:
                d = min(i - (R0 - 1), R1 - i, j - (C0 - 1), C1 - j)
                (band if d <= BAND else inside)[i, j] = True
            else:
                dr = max(0, R0 - i, i - (R1 - 1))
                dc = max(0, C0 - j, j - (C1 - 1))
                if math.sqrt(dr * dr + dc * dc) <= BAND:
                    band[i, j] = True
    return {"in": inside, "band": band, "out": ~inside & ~band}


def psnr_of_mse(mse: float) -> float:
    return 10.0 * math.log10(255.0 ** 2 / mse)


print("=" * 72)
print("§1 全图 PSNR (round×255 口径)")
n_shift = (R1 - R0) * (C1 - C0)
mse_full = n_shift * D * D / (H * W)  # 每 shifted 像素 3 通道各贡献 D², 通道均摊后不变
check("psnr_full", metrics.psnr_full(pred, gt), psnr_of_mse(mse_full))
check("psnr_full identity -> cap", metrics.psnr_full(gt, gt), 100.0, TOL_EXACT)

print("=" * 72)
print("§2 masked PSNR 三分 (band_px=2)")
reg = analytic_regions()
got = metrics.masked_psnr(pred, gt, mask, band_px=BAND)
# 期望: in 区全 shifted -> mse=D²; band 区 shifted 数 = band∩掩膜; out 区零误差 -> cap
n_band = int(reg["band"].sum())
n_band_shift = int((reg["band"] & (mask >= 0.5)).sum())
exp_in = psnr_of_mse(D * D)
exp_band = psnr_of_mse(n_band_shift * D * D / n_band)
check("psnr_in", got["psnr_in"], exp_in)
check("psnr_band", got["psnr_band"], exp_band)
check("psnr_out (zero err -> cap)", got["psnr_out"], 100.0, TOL_EXACT)
for k in ("in", "band", "out"):
    check(f"frac_{k}", got[f"frac_{k}"], reg[k].sum() / (H * W), TOL_EXACT)
# 区域划分逐像素对拍
impl_reg = metrics.region_partition(mask, band_px=BAND)
check_true("region_partition pixelwise == analytic", all(np.array_equal(impl_reg[k], reg[k]) for k in reg))
check_true(
    "regions disjoint & cover",
    int(impl_reg["in"].sum() + impl_reg["band"].sum() + impl_reg["out"].sum()) == H * W,
)

print("=" * 72)
print("§3 ΔE00 (D65)")
check("deltaE00 identity", metrics.delta_e00(gt, gt), 0.0, TOL_EXACT)


def srgb_to_L(v: float) -> float:
    """中性灰 (R=G=B=v) 的解析 L*: X/Xn=Y/Yn=Z/Zn=srgb_linear(v) (白点无关)."""
    lin = v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    f = lin ** (1 / 3) if lin > (6 / 29) ** 3 else lin / (3 * (6 / 29) ** 2) + 4 / 29
    return 116.0 * f - 16.0


def ciede2000_neutral(l1: float, l2: float) -> float:
    """CIEDE2000 在两中性色 (C=0) 下的解析简化: ΔE = ΔL' / S_L."""
    lbar = (l1 + l2) / 2.0
    s_l = 1.0 + 0.015 * (lbar - 50.0) ** 2 / math.sqrt(20.0 + (lbar - 50.0) ** 2)
    return abs(l2 - l1) / s_l


g1, g2 = 100 / 255.0, 130 / 255.0
img1 = np.full((8, 8, 3), g1)
img2 = np.full((8, 8, 3), g2)
exp_de = ciede2000_neutral(srgb_to_L(g1), srgb_to_L(g2))
check("deltaE00 neutral-gray analytic", metrics.delta_e00(img1, img2), exp_de)
# 掩膜内 ΔE: 半图掩膜, 掩膜内=g1 vs g2, 掩膜外恒等 -> 掩膜内均值不被稀释
m_half = np.zeros((8, 8))
m_half[:, :4] = 1.0
img2b = img1.copy()
img2b[:, :4] = g2
check("deltaE00 masked", metrics.delta_e00(img1, img2b, mask=m_half), exp_de)

print("=" * 72)
print("§4 SSIM (常量图解析值)")
# 常量 a vs a+d: σ 项全零 -> SSIM = (2ab+C1)/(a²+b²+C1), a=100, b=110 (量化后), C1=(0.01*255)²
a_, b_ = float(A), float(A + D)
c1 = (0.01 * 255.0) ** 2
exp_ssim = (2 * a_ * b_ + c1) / (a_ * a_ + b_ * b_ + c1)
const_pred = np.full((32, 32, 3), (A + D) / 255.0)
const_gt = np.full((32, 32, 3), A / 255.0)
check("ssim constant-pair analytic", metrics.ssim(const_pred, const_gt), exp_ssim)
check("ssim identity", metrics.ssim(gt, gt), 1.0, TOL_EXACT)

print("=" * 72)
print("§5 stats: bootstrap CI + 符号检验")
const_d = np.full(20, 0.7)
lo, hi = stats.bootstrap_ci(const_d, n_boot=200, seed=0)
check("bootstrap CI degenerate lo", lo, 0.7, TOL_EXACT)
check("bootstrap CI degenerate hi", hi, 0.7, TOL_EXACT)
st = stats.sign_test(const_d)
check("sign test all-pos p (2^-19)", st["p_value"], 2.0 ** -19, TOL_EXACT)
# 混合案例 n_pos=3, n_neg=1, n_zero=1: 双侧精确 p = 10/16
st2 = stats.sign_test([1.0, 2.0, 0.5, -0.3, 0.0])
check("sign test mixed p (10/16)", st2["p_value"], 10.0 / 16.0, TOL_EXACT)
check_true("sign test counts", (st2["n_pos"], st2["n_neg"], st2["n_zero"]) == (3, 1, 1))
rep = stats.compare([30.0, 31.0, 32.0], [29.0, 30.5, 31.0], n_boot=100, seed=1)
check("compare delta mean", rep["delta_psnr_mean"], (1.0 + 0.5 + 1.0) / 3.0, TOL_EXACT)

print("=" * 72)
print("§6 collapse probes: Δ_const / Δ_shuffle / σ_s")
# 8 样本玩具: image=100/255 常量, s_i = c_i/255 常量场, gt_i = image + s_i,
# render(image, s) = image + s + 1/255 (故意 1 单位残差, 避免 PSNR 触 cap)
C_VALS = [0, 4, 8, 12, 16, 20, 24, 28]  # 均值=14 整数 -> 量化无 .5 歧义
toy = []
base = np.full((16, 16, 3), 100 / 255.0)
for idx, c in enumerate(C_VALS):
    s_field = np.full((16, 16, 1), c / 255.0)
    toy.append({"id": idx, "image": base, "gt": base + c / 255.0, "s": s_field})


def toy_render(image, s):
    return metrics.to_float01(image) + np.broadcast_to(np.asarray(s), image.shape[:2] + (1,)) + 1.0 / 255.0


p_true_exp = psnr_of_mse(1.0)  # 残差恒为 1 量化单位
dc = delta_const(toy_render, toy, s_null=None)
mean_c = sum(C_VALS) / len(C_VALS)
p_null_exp = float(np.mean([psnr_of_mse((mean_c - c + 1.0) ** 2) for c in C_VALS]))
check("delta_const psnr_true", dc["psnr_true"], p_true_exp)
check("delta_const psnr_null", dc["psnr_null"], p_null_exp)
check("delta_const", dc["delta_const"], p_true_exp - p_null_exp)
check_true("delta_const mode recorded", dc["delta_const_mode"] == "dataset_mean")

# Δ_shuffle: derangement 性质检查 (记录第二遍 pass 收到的 s, 必须无一等于自身)
for seed in range(5):
    seen: list = []

    def spy_render(image, s, _seen=seen):
        _seen.append(float(np.asarray(s).ravel()[0]))
        return toy_render(image, s)

    ds = delta_shuffle(spy_render, toy, seed=seed)
    shuf_pass = seen[len(toy):]  # 后一半调用 = shuffled s
    check_true(
        f"delta_shuffle derangement (seed={seed})",
        all(abs(sv - c / 255.0) > 1e-12 for sv, c in zip(shuf_pass, C_VALS)),
    )
    # 期望值: 用 spy 记录的实际置换独立算解析 PSNR
    p_shuf_exp = float(
        np.mean([psnr_of_mse((sv * 255.0 - c + 1.0) ** 2) for sv, c in zip(shuf_pass, C_VALS)])
    )
    check(f"delta_shuffle (seed={seed})", ds["delta_shuffle"], p_true_exp - p_shuf_exp)

# σ_s 统计: 已知构造 20% 贴上界
sig = np.concatenate([np.full(8, 0.30), np.full(32, 0.10)])
ss = sigma_stats({"sigma_s": sig}, sigma_min=0.025, sigma_max=0.30)
check("sigma frac_at_upper", ss["frac_at_upper"], 0.2, TOL_EXACT)
check_true("sigma m3 red line not triggered", ss["m3_red_line"] is False)
ss2 = sigma_stats({"sigma_s": np.full(10, 0.2999)}, sigma_min=0.025, sigma_max=0.30)
check_true("sigma m3 red line triggered", ss2["m3_red_line"] is True)

print("=" * 72)
print("§7 leaderboard: 红线列强制 N/A + 警告")
with tempfile.TemporaryDirectory() as td:
    p_full = Path(td) / "e1" / "metrics.json"
    p_miss = Path(td) / "e2" / "metrics.json"
    p_full.parent.mkdir()
    p_miss.parent.mkdir()
    p_full.write_text(json.dumps({
        "exp_id": "arm-full", "n_images": 10,
        "metrics": {"psnr_in": 30.0, "psnr_band": 28.0, "psnr_out": 40.0, "psnr_full": 35.0,
                    "ssim": 0.9, "delta_e00": 2.0, "delta_const": 1.2, "delta_shuffle": 3.4},
        "ci": {"delta_psnr_mean": 0.5, "ci95": [0.2, 0.8], "sign_test": {"p_value": 0.01}},
    }))
    p_miss.write_text(json.dumps({
        "exp_id": "arm-missing", "n_images": 10,
        "metrics": {"psnr_in": 33.0, "delta_const": 0.9},
    }))
    md, warns = build_leaderboard([p_full, p_miss])
    check_true("leaderboard: 缺失 Δ_shuffle 标 N/A ⚠", "N/A ⚠" in md)
    check_true("leaderboard: 缺失触发警告", any("delta_shuffle" in w for w in warns))
    check_true("leaderboard: 完整行含 3.400", "3.400" in md)
    rows = [l for l in md.splitlines() if l.startswith("| arm-")]
    check_true("leaderboard: psnr_in 降序", rows[0].startswith("| arm-missing"))

print("=" * 72)
print("§8 烘焙一致性 (INF-1 B2 补件: 33³ 采样 -> LUT -> 四面体回读, 128³ 留出色)")
# 8a. 恒等渲染器: 恒等 LUT 的四面体回读逐位精确 -> 烘焙一致性 ΔE = 0, PSNR 触 cap.
bc_id = bake.bake_consistency(lambda rgb: np.asarray(rgb))
check("bake identity de00_max = 0", bc_id["heldout"]["de00_max"], 0.0, TOL_EXACT)
check("bake identity max_abs_err = 0", bc_id["heldout"]["max_abs_err"], 0.0, TOL_EXACT)
check("bake identity psnr -> cap", bc_id["heldout"]["psnr_direct_vs_baked"], 100.0, TOL_EXACT)
check_true("bake identity passed (p99 < 0.5)", bc_id["passed"] is True)

# 8b. 已知 gamma(=2) 渲染器: 烘焙误差为小的非零值, 且可闭式预估.
#   四面体插值对逐通道可分函数 == 逐通道 1D 线性插值 (方案对线性函数精确
#   => x1 面权重和 = 分数坐标 fx). f(x)=x², f''=2 -> 单元 [x0, x0+h] 内
#   线性插值误差 e(x) = (x-x0)(x0+h-x), h=1/32; 对留出色格独立算 max.
vals_odd = np.arange(1, 256, 2, dtype=np.float64) / 255.0
cell_lo = np.floor(vals_odd * 32.0) / 32.0
exp_gamma_err = float(np.max((vals_odd - cell_lo) * (cell_lo + 1.0 / 32.0 - vals_odd)))
rng_nat = np.random.default_rng(7)
bc_g = bake.bake_consistency(
    lambda rgb: np.asarray(rgb) ** 2,
    natural_images=[rng_nat.random((64, 64, 3))],
)
check("bake gamma2 max_abs_err (解析预估)", bc_g["heldout"]["max_abs_err"], exp_gamma_err, 1e-6)
check_true(
    "bake gamma2 de00_p99 小的非零值 (0 < p99 < 0.5)",
    0.0 < bc_g["heldout"]["de00_p99"] < bake.DE00_P99_PASS,
)
check_true("bake gamma2 natural image 面也过判据", bc_g["natural"][0]["passed"] is True)
check_true("bake gamma2 passed", bc_g["passed"] is True)

print("=" * 72)
print(f"selfcheck: {len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("FAILED:", *FAILED, sep="\n  - ")
    sys.exit(1)
print("ALL PASS")
