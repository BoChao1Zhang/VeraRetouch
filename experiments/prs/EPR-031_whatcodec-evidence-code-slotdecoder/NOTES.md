# EPR-031 · NOTES

## C0+C1 实施（2026-08-16）

只写假设、待决策、实测计数。不写解读、不下结论。

### 落点（全部新建，未改任何现有文件）

| 文件 | 内容 |
|---|---|
| `q3vl/whatb/codec/__init__.py` | 包说明 |
| `q3vl/whatb/codec/lutcode.py` | C0：残差矩阵、whitened PCA、`cumulative_dims`、`code_recon_de00`、拟合集守卫 |
| `q3vl/whatb/codec/tables.py` | C1：`DirectCarrierTable`（Θ 表）、`STAGES` 档位表、paired anchors、颜色组 mining、`R_line` |
| `q3vl/whatb/scripts/build_lut_code.py` | C0 入口（CPU） |
| `q3vl/whatb/scripts/run_whatcodec_arm.py` | C1 入口（O0） |
| `q3vl/whatb/tests/test_whatcodec_c0c1.py` | 37 条 CPU 单测 |

C0 产物：`/home/bc/data/runs/whatb/lutcode_17c_pca/{code.npz, manifest.json}`，
`code.npz` sha256 `9a06ab9aaf84b6da4d5961bd43b61d1780f7f59dc7ba108dbda7542096c1311e`。
两个文件都先写同目录 `*.tmp` 再 `os.replace`（`code.npz` 另加 `fsync`）。

### 假设（实施时按保守默认继续，未静默拍板）

1. **拟合集与 V_what / T_final 的 lut_id 关系（需用户裁定）。**
   任务卡单测第 1 条要求「拟合集里没有 V_what / T_final / T_lut_unseen 的任何 lut_id」。
   实测：train index uniq lut_id = **3,149**；`V_what` 531 个 lut_id **531 个全部**在 train 内；
   `T_final` 577 个 **577 个全部**在 train 内；`V_where` 530 个亦然；
   `T_lut_unseen` 259 个与 train 交集 **0**。三个评测集的 lut_id 并集与 train 的交集 = 1,093，
   若全部排除则拟合集为 **2,056** 条，与 §1「canonical code 拟合集 = train 的 3,149 条」冲突。
   保守默认：按 §1 用全部 3,149 条拟合；运行时守卫 `assert_fit_set_clean` 只对
   **LUT 不相交的** `T_lut_unseen` 硬报错；`--exclude-eval-luts`（默认 off）提供另一档。
   单测第 1 条按此实现：断言 `T_lut_unseen ∩ 拟合集 = 0`，并把
   V_what / T_final 的全包含关系一并断言落盘。

2. **9³ 对照的 `Lib_tr` 抽样种子未知。**
   既有底账协议为「从 train index 随机采 2500 行 → 1137 个 lut_id」，
   但六份提案都未记种子。本次用 `random.Random(20260810).sample(train_index_rows, 2500)`
   （与 `run_carrier_arm._library_ids` 同一抽样形状），得到 **1,166** 个 lut_id，
   与底账的 1,137 不同。**未换种子去凑**；manifest 里 `matches_ledger_id_count=false`。
   9³ 对照同时并排给出「全部 3,149 条」的一版。

3. **9³ PCA 的向量空间取 sRGB 残差。**
   底账原文「9³ 均匀 sRGB 网格；每色转 CIELab 后取 L2（ΔE76）」描述的是**距离度量**，
   「1137 条 LUT 在该 2187 维空间做 PCA」未指明 PCA 的向量是 sRGB 还是 Lab。
   本次两处（17³ 主档、9³ 对照）一律用 **sRGB 残差**。
   注：PCA 先去均值，所以「残差 `L(x)−x`」与「原始输出 `L(x)`」的成分/解释方差**完全相同**
   （二者只差常向量 `x`），该项无歧义；sRGB vs Lab 有歧义，待裁定。

4. **`code_recon_de00` 未做 [0,1] 裁剪为主数字**，同时并排给出裁剪版（`clamped=true`）。

5. **`d_LUT` 三档取自同一次 SVD 的前缀切片**（`transform(x, k)` 取前 k 列），
   落盘 `c_star` 宽度 256，`components` (256, 14739)，`explained_variance_ratio` 落**满秩**长度
   （2,187/3,148），以便从产物本身重算 90/95/99。

6. **Θ 表用单个 `nn.Parameter(n_lut, D_θ)` 扁平表**（不是 `nn.Embedding`），
   按 `theta_fields(mode, N)` 的定长布局切片成 `G4DParams`；
   参数量恒等于 `n_lut × n_params_g4d(N, mode)`，行间梯度不串（单测钉死）。
   表的每一行从同一组初值出发（EPR-028 R1 §8.4），`mu_s` 只抽一次共享。

7. **mining 斜坡的 epoch 口径。** GLUT App A.1 的 mining 斜坡（epoch 5→20，10%→40%）
   按 epoch 定义，而 O0 没有 record 级 epoch。本次把斜坡铺在本档自身步数上：
   `epoch_equiv = --mining-epochs(默认 40) × step / total_steps`，写进 `run_setup.json`。
   S1 档 `mining=False`（照 §7 表）。

8. **A1 档 `s=0` 列结构性为 0。** A1 的前向是 `mix_alpha(x, T(x), s)`，
   `rendering.py:311-313` 的端点吸附使 `s=0` 时输出逐位等于 `x`，
   目标 `(1−s)x + sL(x)` 在 `s=0` 时亦为 `x`，故 `grid_s0` 恒为 0.0。A3 档不具此性质。

9. **O0 不出 headline 板。** `metrics.json` 写 `published=false`、`oracle_reference=true`、
   `headline_board=false`；不读图像 / z 缓存 / pred_field，也不把它们记为「缺失」。

10. **`--weight-decay` 默认 0、无 warmup**（照 EPR-028 R1 §8.4 冻结口径）。
    R1 §11-1 / EPR-031 §9-1 的「StatLUT 原配方 AdamW wd=0.05 + 5-epoch warmup」**未自行改**。

### 待决策（列给用户，未拍板）

- 假设 1：`--exclude-eval-luts` 是否要开（拟合集 3,149 → 2,056）。
- 假设 2：9³ 对照要不要换一个能复现 1,137 的抽样口径（本次未凑）。
- 假设 3：9³ / 17³ 的 PCA 向量空间是 sRGB 还是 Lab。
- §9-2 的 `d_LUT` 主档 192、§9-3 的主格 17³：本次按提案默认实现，三档 128/192/256 均已落盘。
- 假设 7：mining 斜坡的 epoch 口径。
- `--quick-eval-luts` 默认 32（快评只打 32 条 LUT，终评打全池）——为省 quick eval 时间，非提案规定。

### 实测计数

**数据侧**

| 项 | 实测 |
|---|---|
| train index 行数 / normal-only | 159,215 / 93,934 |
| train uniq lut_id（全部行 / normal-only 行） | **3,149** / 3,081 |
| V_what / V_where / T_final / T_lut_unseen 的 uniq lut_id | 531 / 530 / 577 / 259 |
| 上述四者与 train uniq lut_id 的交集 | 531 / 530 / 577 / **0** |

**C0（17³ 主档，3,149 条，14,739 维，float64）**

| 项 | 实测 |
|---|---|
| 解释方差累计 90% / 95% / 99% 所需维数 | **14 / 28 / 117** |
| 秩上限 `min(n−1, d)` | 3,148 |
| 落盘成分数 | 256 |
| 总方差 | 432.5652358606418 |
| LUT 求值耗时 / SVD 耗时 / 总耗时 | 10.5 s / 20.7 s / 71.7 s（CPU，OMP 16 线程） |
| `code.npz` 大小 | 37,153,964 B |

`code_recon_de00`（17³ 网格，对 GT LUT 的 ΔE00 均值，纯几何量）

| `d_LUT` | 未裁剪 | 裁剪到 [0,1] |
|---|---|---|
| 128 | **1.4121292022147434** | 1.3986988927131874 |
| 192 | **1.0885775465429532** | 1.0787805691491754 |
| 256 | **0.8978493688937816** | 0.8900381266275258 |

**C0（9³ 对照，2,187 维）**

| 池 | n_lut | 秩 | 90% | 95% | 99% |
|---|---|---|---|---|---|
| 底账（六份提案逐字） | 1,137 | — | 15 | 28 | 99 |
| 本次 redraw（seed 20260810，2500 行） | 1,166 | 1,165 | **15** | **29** | **109** |
| 全部 train 池 | 3,149 | 2,187 | **15** | **29** | **120** |

**C1（档位解析，`--dry-run` 实测）**

| stage | carrier | LUT 池 | b × q × anchors = 每步 | 步数 | mining | `D_θ`/条 | Θ 参数量 | rc |
|---|---|---|---|---|---|---|---|---|
| S1 | A1 | 1 | 1 × 2048 × 4 = 8,192 | 2,000 | False | 1,068 | 1,068 | 0 |
| S2 | A1 | 32 | 32 × 512 × 4 = 65,536 | 4,000 | True | 1,068 | 34,176 | 0 |
| S2 | A3 | 32 | 32 × 512 × 4 = 65,536 | 4,000 | True | 1,308 | 41,856 | 0 |
| S3 | A1/A3 | 3,149 | 256 × 2048 × 4 = 2,097,152 | 18,760 | True | 1,068 / 1,308 | 3,363,132 / 4,118,892 | — |

**C1（CPU 冒烟，用于排期，非结果）**

| 项 | 实测 |
|---|---|
| S1 / A1，50 步（CPU，8,192 pairs/步，含首步断言） | **2.0824 s**（41.6 ms/步） |
| S1 / A3，50 步（同上） | **2.3426 s**（46.9 ms/步） |
| S1 / A1，5 步 + 一次 quick eval（1 条 LUT，17³×5 s） | rc 0，`grid_de00_mean` 3.5605798244476317 |
| S2 / A3，3 步 + quick eval（2 条 LUT）+ 终评（32 条） | rc 0，`grid_de00_mean` 6.1420070756226774 |

首步 `steps.jsonl` 行（S2/A3，3 步档）实际出现的列：
`L_rec, L_hc, L_sparse, n_hc_masked, R_line, L_total, n_colors, n_colors_distinct,
n_luts_in_batch, mining_ratio, lr, null_mass_mean, null_mass_p99,
cholesky_info_nonzero, opacity_p05/p50/p95, n_pairs_s, weight_underflow_frac,
tau_p05/p50/p95, beta_absmean, gnorm`，
`g4d.assert_step_row` 在首步就地断言通过。

**单测**：`CUDA_VISIBLE_DEVICES="" pytest q3vl/whatb/tests/test_whatcodec_c0c1.py -q`
→ **37 passed**。全量 `q3vl/whatb/tests` → **788 passed**（新增文件未破坏既有 751 条）。

### 未做 / 越界的事

- 未提交任何作业（无 `q submit`、无 nohup / setsid / `&` 的训练进程）；GPU 全程未占。
- 未改任何现有文件（`arms/g4d.py`、`scripts/run_g4d_arm.py` 及 8 个在跑作业依赖的模块 mtime 未变）。
- `--mem-peak`：**需冒烟实测，禁估算**（本次不碰 GPU，给不出 `torch.cuda.max_memory_reserved`）。
