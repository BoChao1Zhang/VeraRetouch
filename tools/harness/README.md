# tools/harness — INF-1 统一评测 harness

所有实验臂共用的评测与监控件（EXPERIMENTS_v3 §1 INF-1 / PLAN §3 M1–M9 / IMPL_DOSSIER §4.1）。
只依赖环境已有库（numpy / scipy / skimage / torch / PIL；LPIPS 懒加载）。

```
python: /home/bc/miniconda3/bin/python3
自检:   python3 tools/harness/selfcheck.py        # 合成已知答案对拍, 42 项, 1e-6
冒烟:   python3 tools/harness/smoke_real.py --lpips  # 真实数据小样本全链路
```

## 口径（锁死, 勿改）

| 指标 | 口径 | 出处 |
|---|---|---|
| PSNR（全图/分区） | `round(x*255)` 后算 MSE，`10*log10(255²/mse)`，逐图 (batch=1) 再平均；mse=0 → `cap_db`(默认 100) | IMPL_DOSSIER §4.1-6 |
| masked PSNR 三分 | 软掩膜按 `thr`(0.5) 二值化；band = 二值边界双侧欧氏距离 ≤ `band_px`(默认 3, 可配) 的像素；in/out = 掩膜内/外去掉 band。**三区互斥, 并集=全图** | PLAN §3 M7 |
| ΔE00 | skimage `rgb2lab`（默认 D65/2°）→ `deltaE_ciede2000`，像素均值 | IMPL_DOSSIER §4.2-8 |
| SSIM | round×255 值域，`gaussian_weights=True, sigma=1.5`（11×11 高斯窗），`use_sample_covariance=False` | Wang04 标准配置 |
| LPIPS | net='alex'，输入 [-1,1]，懒加载 + 模块级缓存 | — |
| Δ_const / Δ_shuffle | mean PSNR(s) − mean PSNR(s_∅ / 跨图置换 s)；置换保证无不动点 | PLAN §3 M1/M2 |
| 配对统计 | 逐图 ΔPSNR + percentile bootstrap 95% CI + 双侧精确符号检验 | PLAN §3 |

红线遵守：harness **绝不对 s 做逐图归一化**；leaderboard 每行强制 Δ_const/Δ_shuffle 列。

## 模块与接口

### metrics.py

图像输入统一 HxWx3 RGB，float∈[0,1] 或 uint8；掩膜 HxW（软边 float / uint8 / bool）。

```python
psnr_full(pred, gt, cap_db=100.0) -> float
region_partition(mask, band_px=3, thr=0.5) -> {'in','band','out'}   # 互斥布尔图
masked_psnr(pred, gt, mask, band_px=3, thr=0.5, cap_db=100.0)
    -> {'psnr_in','psnr_band','psnr_out','frac_in','frac_band','frac_out'}  # 空区 NaN
delta_e00(pred, gt, mask=None) -> float          # mask 给定时只算 mask>=0.5 像素
ssim(pred, gt) -> float
lpips_dist(pred, gt, net='alex', device='cpu') -> float   # 懒加载
evaluate_pair(pred, gt, mask=None, band_px=3, thr=0.5, cap_db=100.0,
              with_lpips=False, lpips_device='cpu') -> dict   # 单图全指标
evaluate_batch(samples, **kwargs)
    -> {'per_image': [...], 'mean': {...含 n_valid_*}, 'n': int}
    # samples: 迭代 (pred, gt[, mask]) 或 {'pred','gt','mask'?,'id'?}; 均值 = 逐图 nanmean
```

### collapse_probes.py

```python
# render_fn(image, s) -> out; samples: 迭代 {'image','gt','s','id'?}
delta_const(render_fn, samples, s_null=None, cap_db=100.0)
    -> {'delta_const','delta_const_mode','psnr_true','psnr_null','per_image_delta','n'}
    # s_null: None=全集逐通道均值常量场(默认); ndarray=统一常量; list=逐样本(模型自带 s_∅)
delta_shuffle(render_fn, samples, seed=0, cap_db=100.0)
    -> {'delta_shuffle','psnr_true','psnr_shuffled','per_image_delta','seed','n'}
    # 置换 = seeded 随机置换 + 轮移1位 => derangement; 要求各样本 s 同形状, 不做静默 resize
run_probes(render_fn, samples, s_null=None, seed=0) -> 扁平 dict, 可直接并入 metrics.json
load_samples(image_dir, gt_dir, s_dir, image_suffix='.in.jpg', gt_suffix='.jpg') -> list
    # s 目录布局按 INF-5: {stem}.npy 或 {stem}_{instr_hash}.npy
sigma_stats(params, sigma_min, sigma_max, key='sigma_s', edge_frac=0.01)
    -> {'frac_at_upper', 'frac_at_lower', 分位数, 'm3_red_line'(>80% 贴上界), ...}
```

### stats.py

```python
paired_delta(psnr_a, psnr_b) -> ndarray            # a−b, NaN 成对剔除
bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0) -> (lo, hi)   # 均值 percentile bootstrap
sign_test(x) -> {'n_pos','n_neg','n_zero','p_value'}  # 双侧精确二项; 零剔除
compare(psnr_a, psnr_b, ...) -> dict               # 完整报告, 可直接作 metrics.json 的 'ci' 节
```

### leaderboard.py

```bash
python3 tools/harness/leaderboard.py "experiments/*/metrics.json" -o leaderboard.md
```

按 `psnr_in` 降序；**Δ_const/Δ_shuffle 缺失 → 该格标 `N/A ⚠` 且 stderr 警告**（PLAN §3 红线）。
库接口：`build_leaderboard(paths) -> (markdown, warnings)`。

### metrics.json schema（leaderboard 解析口径）

```json
{
  "exp_id": "arm-xxx",            // 必填; 缺省用父目录名
  "n_images": 100,
  "metrics": {
    "psnr_in": 0.0, "psnr_band": 0.0, "psnr_out": 0.0, "psnr_full": 0.0,
    "ssim": 0.0, "delta_e00": 0.0, "lpips": 0.0,
    "delta_const": 0.0, "delta_shuffle": 0.0      // 红线列, 缺失标 N/A ⚠
  },
  "ci": {"delta_psnr_mean": 0.0, "ci95": [0.0, 0.0], "sign_test": {"p_value": 1.0}}
}
```

### selfcheck.py / smoke_real.py

- `selfcheck.py`：合成图（常量底图 + 掩膜内 +Δ 偏移，全部取值在 1/255 格上）对每个指标
  独立解析求期望，与实现对拍 1e-6（恒等类 1e-12）。42 项，退出码 0 = 全过。
- `smoke_real.py`：从落盘 build（mini30-v52）shard 抽带 `.cgt.png` 的样本，跑全指标 +
  oracle 渲染的 Δ_const/Δ_shuffle + 配对统计，产出 metrics.json / viz / leaderboard demo
  到 `experiments/tooling-wave1/harness/`。

## 已知边界

- 本交付**不含**烘焙一致性（四面体插值回读）评测器——任务卡 T3 范围外，见 NOTES.md 待决策 3。
- `band_px` 默认 3 是保守占位，正式实验前应在 EXPERIMENTS_v3 统一定值（NOTES.md 待决策 1）。
- Δ_const 的 s_∅ 默认口径 = 全评测集均值常量场；模型自带 s_∅ 时务必显式传 `s_null`，
  榜单 `delta_const_mode` 字段记录口径（NOTES.md 待决策 2）。
