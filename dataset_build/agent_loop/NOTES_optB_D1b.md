# NOTES · 任务卡 D1b（可见性指标的明度/彩度/色相拆解 + 同链回填 + 同标注验证）

在 D1 基建上加列，**不接线**：`visibility.py` 仍无调用方（除离线工具与单测），未改门、未 bump
revision、未动 `graph.py` / `render.py` / `models.py`，未覆盖任何 D1 产物。

## 1. 落盘文件

| 路径 | 内容 |
|---|---|
| `dataset_build/agent_loop/visibility.py` | 新增 `lab_at` / `lab_components` / `visibility_components` / `assert_component_columns` |
| `dataset_build/tools/backfill_visibility_metrics.py` | `backfill --kind components --from-chains`、`validate --columns components` |
| `dataset_build/tests/test_visibility.py` | 新增 10 个单测（共 25 个，全绿） |
| `/home/bc/data/scratch/visibility_metrics/components.jsonl` | 增量回填，1562 行（D1 的 `chains.jsonl` 未改） |
| `/home/bc/data/scratch/visibility_metrics/validation_components.json` | 验证报告 |
| `/home/bc/data/scratch/visibility_metrics/joined_components.jsonl` | 554 条 join 后逐条记录 |

## 2. 预注册常量与定义

```
COMPONENTS_CONTRACT   = "visibility-lab-components-v1"
COMPONENT_QUANTILE    = 0.90
HIGHLIGHT_L_THRESHOLD = 85.0        # global_after 的 L*
（tau / SAMPLE_PIXELS_IN 沿用 D1：0.05 / 2048）
```

支撑集样本 = `draw_region(pool, SAMPLE_PIXELS_IN, seed_key=branch_id, "in")`，与 D1 的 `de_in`
**逐点相同**（单测 `test_components_share_the_support_sample_of_the_de00_family` 断言）。

逐像素分解（CIELAB, D65，与 `render.delta_e_map` 同一 `rgb2lab`）：

```
dL = L2 - L1
dC = C*ab2 - C*ab1                       C*ab = hypot(a, b)
dH = 2 * sqrt(C1*C2) * sin(dh / 2)       dh = wrap(h2 - h1) 到 (-pi, pi]
恒等式 dL^2 + dC^2 + dH^2 == dE*ab^2     （单测逐点断言，rel 1e-6）
```

列：

| 列 | 定义（均在支撑集样本上） |
|---|---|
| `dL_mean` / `dL_p90` | `mean(|dL|)` / `p90(|dL|)` |
| `dL_signed_mean` | `mean(dL)` |
| `dC_mean` / `dC_p90` | `mean(|dC|)` / `p90(|dC|)` |
| `dC_signed_mean` | `mean(dC)` |
| `dHue_mean` | `mean(|dH|)` |
| `dL_highlight_mean` / `dL_highlight_signed_mean` | 高亮子集上的 `mean(|dL|)` / `mean(dL)` |
| `highlight_frac` | 支撑集里 `L*(global_after) > 85` 的像素占比 |
| `dL_grad` | 支撑集样本上 `std(dL)` |
| `dL_step_mean` | 每个采样点与右邻/下邻的 `|dL 差|` 取大者，再取均值 |
| `n_in` / `n_highlight` | 两个样本的实际条数 |

## 3. 假设与保守默认（未静默拍板）

1. **`dC_mean` / `dHue_mean` 取绝对值**。任务卡写 `dC_mean = mean(ΔC*ab)`，而并列的
   `dL_mean` 明确写了绝对值。保守默认：幅度族一律取绝对值（`dL_mean`/`dC_mean`/`dHue_mean`），
   方向另开带符号列（`dL_signed_mean`/`dC_signed_mean`/`dL_highlight_signed_mean`），
   两种口径都落盘，不需要重算。
2. **`dL_grad` 取「支撑集样本内 dL 的标准差」**（任务卡原文「ΔL 图的空间标准差」）。
   列名带 `grad` 但定义是 std，故另加真·局部梯度列 `dL_step_mean`（一步邻域差分，
   口径与 D1 的 `edge_step_p95` 同构），两列并排。
3. **高亮子集从「整个支撑集池」筛后再抽 2048**，不是从已抽的 2048 里筛（否则 n 随
   `highlight_frac` 缩水，正是 D1 修掉的那个缺陷）。`highlight_frac` 分母是支撑集大小。
   支撑集内无 `L*>85` 像素时两列记 `None`（1562 条里 69 条；554 条标注里 26 条）。
4. **链集合来自 D1 的 `chains.jsonl`（`--from-chains`）而非重查 postgres**，保证与 D1 逐条同集
   （1562 条 status=ok）；D1 那 3 条 `blob_missing` 不再重试。
5. `de_masked` / `support_frac` 随增量文件一并写出，供并排对照；`support_frac` 与 D1 逐条相等
   （同 tau、同 alpha）。

## 4. 确定性

- `--workers 12` 与 `--workers 5` 的前 200 行 **逐字节相同**（已实测）。
- 输出按 `(campaign_id, branch_id)` 排序写出。
- `visibility_components` 同 seed_key 两次调用 dict 全等；换 seed_key 抽样变（单测断言）。

## 5. 运行时断言

- `assert_component_columns` 在每条链回填后调用（`compute_chain`）。
- `validate` 在出表前断言：预注册列里若有列在 554 条 join 中**全为 None**，直接抛
  `VisibilityError`（「定义了没接线」守卫）。

## 6. 复现命令

```bash
python3 dataset_build/tools/backfill_visibility_metrics.py backfill \
  --kind components --workers 12 \
  --from-chains /home/bc/data/scratch/visibility_metrics/chains.jsonl \
  --out /home/bc/data/scratch/visibility_metrics/components.jsonl

python3 dataset_build/tools/backfill_visibility_metrics.py validate \
  --chains /home/bc/data/scratch/visibility_metrics/components.jsonl \
  --columns components \
  --round "r1_iter2=docs/assets/lut_cluster_pilot_20260819/intent_quality_200/item_key.json=docs/assets/questionnaire/intentq.csv" \
  --round "r2_iter3=docs/assets/lut_cluster_pilot_20260819/intent_quality_v2/item_key.json=docs/assets/questionnaire/intentq (1).csv" \
  --round "r3_iter4=docs/assets/lut_cluster_pilot_20260819/intent_quality_v3/item_key.json=docs/assets/questionnaire/intentq (2).csv" \
  --out /home/bc/data/scratch/visibility_metrics/validation_components.json \
  --joined-out /home/bc/data/scratch/visibility_metrics/joined_components.jsonl
```

## 7. 结果（只列数字）

### 7.1 新列分布（1562 条链）

| col | n | mean | p25 | p50 | p75 | p90 | min | max |
|---|---|---|---|---|---|---|---|---|
| dL_mean | 1562 | 2.209 | 1.160 | 2.127 | 3.050 | 4.059 | 0.127 | 7.049 |
| dL_p90 | 1562 | 4.225 | 2.260 | 4.030 | 5.874 | 7.575 | 0.302 | 17.828 |
| dL_signed_mean | 1562 | -0.646 | -2.327 | -0.496 | 0.670 | 2.425 | -7.018 | 5.209 |
| dC_mean | 1562 | 2.648 | 1.661 | 2.415 | 3.254 | 4.496 | 0.127 | 10.174 |
| dC_p90 | 1562 | 5.673 | 3.519 | 5.073 | 6.993 | 9.497 | 0.415 | 36.270 |
| dC_signed_mean | 1562 | -0.742 | -2.094 | -0.634 | 0.681 | 1.860 | -10.169 | 8.768 |
| dHue_mean | 1562 | 1.640 | 0.914 | 1.527 | 2.179 | 2.987 | 0.067 | 6.861 |
| dL_highlight_mean | 1493 | 1.869 | 0.456 | 0.991 | 2.213 | 5.068 | 0.000 | 17.433 |
| dL_highlight_signed_mean | 1493 | -0.956 | -1.686 | -0.079 | 0.617 | 1.475 | -17.433 | 4.998 |
| highlight_frac | 1562 | 0.165 | 0.010 | 0.076 | 0.240 | 0.467 | 0.000 | 1.000 |
| dL_grad | 1562 | 1.750 | 0.876 | 1.601 | 2.390 | 3.246 | 0.171 | 6.879 |
| dL_step_mean | 1562 | 0.378 | 0.228 | 0.312 | 0.448 | 0.618 | 0.027 | 3.453 |
| de_masked | 1562 | 3.633 | 3.014 | 3.848 | 4.024 | 4.672 | 2.503 | 6.106 |
| support_frac | 1562 | 0.683 | 0.536 | 0.743 | 0.865 | 0.883 | 0.153 | 1.000 |

`dL_highlight_*` 为 None：69 / 1562。

### 7.2 Spearman(metric, rating)，n=554（同 D1 §3.2 格式）

| col | n | rho | p |
|---|---|---|---|
| support_frac | 554 | 0.151 | 0.0004 |
| highlight_frac | 554 | -0.146 | 0.0006 |
| de_masked | 554 | 0.082 | 0.0549 |
| dL_mean | 554 | -0.059 | 0.1663 |
| dC_mean | 554 | -0.055 | 0.1948 |
| dL_highlight_mean | 528 | -0.054 | 0.2116 |
| dC_signed_mean | 554 | 0.047 | 0.2657 |
| dL_highlight_signed_mean | 528 | 0.040 | 0.3547 |
| dC_p90 | 554 | -0.039 | 0.3597 |
| dL_p90 | 554 | -0.033 | 0.4322 |
| dL_signed_mean | 554 | -0.028 | 0.5091 |
| dHue_mean | 554 | 0.023 | 0.5819 |
| dL_step_mean | 554 | 0.017 | 0.6926 |
| dL_grad | 554 | -0.002 | 0.9635 |

D1 最强信号 `support_frac` = 0.151；D1 最强 ΔE00 列 `edge_de` = 0.127。

### 7.3 rating==4 vs rating in {3,5}（同 D1 §3.3 格式）

| col | mean4 | mean35 | mean_gap | p50_gap | rank_r_is4 | p |
|---|---|---|---|---|---|---|
| dL_mean | 2.344 | 2.196 | 0.148 | 0.337 | 0.059 | 0.2098 |
| dL_p90 | 4.544 | 4.324 | 0.220 | 0.485 | 0.054 | 0.2522 |
| dL_signed_mean | -0.819 | -0.575 | -0.244 | -0.301 | -0.066 | 0.1612 |
| dC_mean | 2.669 | 2.701 | -0.032 | -0.122 | -0.026 | 0.5878 |
| dC_p90 | 5.808 | 5.851 | -0.044 | -0.041 | -0.040 | 0.3996 |
| dC_signed_mean | -0.988 | -0.566 | -0.422 | -0.301 | -0.064 | 0.1763 |
| dHue_mean | 1.691 | 1.681 | 0.010 | 0.051 | 0.002 | 0.9665 |
| dL_highlight_mean | 1.948 | 1.825 | 0.123 | -0.009 | 0.025 | 0.6005 |
| dL_highlight_signed_mean | -1.170 | -0.867 | -0.304 | -0.204 | -0.074 | 0.1266 |
| highlight_frac | 0.144 | 0.147 | -0.002 | 0.040 | 0.057 | 0.2297 |
| dL_grad | 1.871 | 1.871 | -0.000 | 0.103 | 0.011 | 0.8099 |
| dL_step_mean | 0.405 | 0.388 | 0.017 | 0.014 | 0.043 | 0.3658 |
| de_masked | 3.802 | 3.757 | 0.045 | 0.021 | 0.033 | 0.4856 |
| support_frac | 0.684 | 0.712 | -0.028 | -0.083 | -0.096 | 0.0408 |

### 7.4 per-intent Spearman(metric, rating)（同 D1 §3.4 格式）

| col | background_swap | highlight_recover | hue_shift | luminance_pop | sat_boost | warm_cool_shift | zonal_contrast |
|---|---|---|---|---|---|---|---|
| n | 135 | 30 | 68 | 103 | 76 | 10 | 132 |
| mean_rating | 3.55 | 3.17 | 3.71 | 3.56 | 3.84 | 2.40 | 3.72 |
| dL_mean | 0.103 | -0.160 | 0.015 | -0.139 | 0.024 | 0.342 | 0.072 |
| dL_p90 | 0.117 | -0.002 | 0.045 | -0.090 | 0.030 | 0.080 | 0.021 |
| dL_signed_mean | -0.072 | 0.140 | 0.013 | -0.052 | 0.041 | 0.060 | -0.087 |
| dC_mean | -0.036 | -0.044 | -0.066 | -0.219 | -0.063 | -0.523 | -0.065 |
| dC_p90 | -0.038 | 0.107 | -0.036 | -0.182 | -0.026 | -0.402 | -0.094 |
| dC_signed_mean | 0.006 | 0.023 | -0.007 | 0.126 | 0.119 | -0.201 | -0.026 |
| dHue_mean | -0.014 | 0.077 | 0.058 | -0.011 | -0.092 | -0.362 | -0.079 |
| dL_highlight_mean | -0.133 | 0.040 | 0.169 | -0.155 | 0.080 | 0.241 | -0.005 |
| dL_highlight_signed_mean | 0.155 | -0.043 | -0.112 | -0.073 | 0.040 | 0.422 | 0.051 |
| highlight_frac | -0.145 | -0.072 | -0.112 | -0.329 | 0.023 | -0.080 | -0.147 |
| dL_grad | 0.148 | 0.433 | 0.029 | -0.093 | 0.107 | 0.241 | -0.020 |
| dL_step_mean | 0.195 | 0.326 | 0.059 | -0.125 | -0.079 | -0.080 | -0.073 |
| de_masked | 0.113 | 0.106 | -0.010 | 0.042 | -0.064 | -0.121 | 0.125 |
| support_frac | 0.105 | 0.045 | 0.088 | 0.241 | 0.093 | -0.334 | 0.107 |

## 8. 待用户决定（本卡不做）

- `dC_mean` / `dHue_mean` 是否改成带符号口径（§3.1）。
- `dL_grad` 是否改判为 `dL_step_mean`（§3.2）。
- 高亮阈值 85 是否改（`HIGHLIGHT_L_THRESHOLD`）。
- 是否把该列族接进 `render.py` / `graph.py`（与 D1 同一悬置项）。
