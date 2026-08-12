# 汇报配图素材清单 · 读出线（VLM 能不能读出 where）

- 生成时间：2026-08-05
- 范围：RO9c / G1 / RO9-L / RO9b / G1b / RO-3 / RO-1 / RO-X1 / MCQ 条件消融 + 6-09 历史归档
- 纪律：本文件只做清单，**未生成任何新图**；所有列出的"已有素材"路径均已 `PIL.Image.open` 验证真实存在并读出尺寸
- 6-09 历史目录在回收站（`/home/bc/.local/share/Trash/files/research/attention_grounding/`），**只读引用，未移动未修改**

---

## 全场通用警告（凡涉及可视化的页都要提示）

来自 RO9c 补件 B（`experiments/RO9c_subject_repro_20260805/REPORT.md` §9.1）：

| # | 坑 | 实测 | 影响哪些图 |
|---|---|---|---|
| **A** | **`expand2square` 黑边格吃掉大部分 image 注意力质量** | 16 格中约 **5.3 格是纯黑边**（有效格占比中位 0.625）；黑边格占 image 注意力质量中位 **68.8%（L）/ 53.4%（GC）/ 74.4%（SC）**；**93.0%** 的源全局 argmax 落在黑边格。而 AUC 只在 valid 格上算 ⇒ **该 sink 从未进入过 AUC** | 所有 6-09 原版式图；RO-3 单场（黑边质量 0.435 / argmax 落黑边 60.6%）与 RO9b（0.523 / 0.506）**也中招**（§10.5） |
| **B** | **6-09 `render()` 网格错位**：把 16×16 格直接 `resize` 到**未 pad** 的图上，而网格覆盖的是 **pad 后的方形** | 约 **1/3** 的系统性错位 | 6-09 原始两张图；RO9c 保留的 `repro_*_raw.png` / `_m1.png`（**故意保留错位作复现对照**）。RO9c 主交付图已用 `make_figs.grid_to_img` 修正，标题带 `[GRID-ALIGNED]` |
| **C** | **逐图 min-max 着色会说谎** | 同一张 raw 场：min-max 着色说"一片蓝、什么也没有"，秩次 AUC 说 **0.736–0.784** | 任何"raw 没信号"的口头说法都要配 `diag_raw_colormap.png` 第 4 列 |

---

## P3 — 6-09 原始效果图 + 这次复现的七列消融图

### P3-a　6-09 原始两张（历史归档，只读）

| 用途 | 绝对路径 | 尺寸 / 大小 |
|---|---|---|
| **6-09 raw（"一片蓝、只有左上角一个 sink"）** | `/home/bc/.local/share/Trash/files/research/attention_grounding/figs/veraretouch_special_token_attn_raw.png` | 6852×1280，2.23 MB |
| **6-09 M1（共模消除后，GC 落到主体花簇）** | `/home/bc/.local/share/Trash/files/research/attention_grounding/figs/veraretouch_special_token_attn.png` | 6852×1280，2.73 MB |
| 元数据（可直接引用的设置数字） | `/home/bc/.local/share/Trash/files/research/attention_grounding/figs/veraretouch_special_token_meta.json` | `num_patches=256`, `grid=[16,16]`, `merged_seq=329`, `rt_pos=[326,327,328]`, `method="M1_diffLMM (seq-mean subtraction)"`, `image=data_samples/input/sample_flower.jpg` |
| 生成脚本（RO9c 逐行照抄的对象） | `/home/bc/.local/share/Trash/files/research/attention_grounding/scripts/veraretouch_special_token_attn.py` | — |

⚠ 这两张图 **n=1**（`sample_flower.jpg`，该图已不在仓库），**含警告 B 的网格错位**，**raw 那张的"一片蓝"是警告 C**。

同批可选备用（同一回收站目录 `figs/` 下，均已验证存在）：

| 用途 | 相对 `figs/` 的路径 | 尺寸 |
|---|---|---|
| VeraRetouch **special token vs post-`retouch_head`-MLP** 对比 | `three_model/veraretouch_token_vs_mlp.png` | 942×2567 |
| 三模型纵向对比联图（base Qwen3-VL-8B / monetGPT / VeraRetouch） | `three_model/three_model_vertical_compare.png` | 1430×2930（另有 `maps.jsonl` / `summary.json`） |
| DiffLMM 原论文的 pipeline / teaser 图（可用于讲 M1 是什么） | `paper_attend_and_segment_pipeline.png` / `paper_attend_and_segment_teaser.png` | 0.74 / 1.13 MB |
| 6-08 那批"涌现 grounding"可视化 | `grounding4_panel.png`（3.99 MB）/ `grounding4_flower.png` / `grounding4_leaves.png` / `grounding4_background.png` / `grounding_alltext.png` | — |

配套 6-09 文字底账（引用结论时的出处）：`attention_method_experiment_2026-06-09.md`、`DiffLMM_method_and_conclusion_2026-06-09.md`（同目录，后者含方程级 M1 定义与 latex 版）。

### P3-b　RO9c 的七列消融图（**这次复现的主图**）

| 用途 | 绝对路径 | 尺寸 / 大小 |
|---|---|---|
| **七列消融 · GC（预注册主 token）** | `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_raw_colormap.png` | 2696×2261，7.13 MB |
| **七列消融 · L（raw AUC 最高的 token）** | `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_raw_colormap_L.png` | 2696×2261，6.95 MB |

七列的确切标题（出处 `diag_raw_colormap.py:136-142`）：

```
1. input
2. raw min-max  [6-09 VERBATIM]        ← 一片蓝
3. + grid alignment fixed              ← 仍然一片蓝（⇒ 错位不是主因）
4. + colour scale from VALID cells     ← 主体立刻显形（⇒ 单变量证明）
5. raw RANK pct  (= what AUC sees)
6. M1 min-max (aligned)
7. GT SAM3 subject
```

- 5 源 **effect-blind** 选出（规则 `config/figure_picks.json`：`winner_confidence != low` ∧ `subject_area ∈ [0.08,0.35]`，按 pool / region_b_kind 分层取 `img_id` 字典序第一）
- **这是本页的杀手锏**：col2 与 col3 都是一片蓝、col4 主体显形 ⇒ 压平 raw 图的不是"没信号"，是位于黑边、且从不参与 AUC 的 sink 在做 min-max 的分母

### P3-c　RO9c 的 6-09 原版式复现（可与 P3-a 并排放）

5 源 × 3 张，路径前缀 `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/`：

| 源 | raw（6-09 逐字原版式，**含其错位**） | M1（同上） | pair（PPT 合并版，**已 grid-aligned**） |
|---|---|---|---|
| `ppr10k_0579_a` | `repro_ppr10k_0579_a_raw.png` (2072×768) | `repro_ppr10k_0579_a_m1.png` (2072×768) | `repro_ppr10k_0579_a_pair.png` (2592×1578, 5.0 MB) |
| `ppr10k_0815_a` | `repro_ppr10k_0815_a_raw.png` | `repro_ppr10k_0815_a_m1.png` | `repro_ppr10k_0815_a_pair.png` |
| `ppr10k_2164_a` | `repro_ppr10k_2164_a_raw.png` | `repro_ppr10k_2164_a_m1.png` | `repro_ppr10k_2164_a_pair.png` |
| `src_08aef2f575be97b1` | `repro_src_08aef2f575be97b1_raw.png` | `repro_src_08aef2f575be97b1_m1.png` | `repro_src_08aef2f575be97b1_pair.png` |
| `src_0e6178111d466cfb` | `repro_src_0e6178111d466cfb_raw.png` | `repro_src_0e6178111d466cfb_m1.png` | `repro_src_0e6178111d466cfb_pair.png` |

`pair` 版式 = input | L | GC | SC，raw 上 / M1 下。

> **必须配的一句限定**（REPORT §八）：raw 那一行之所以"只有一个角"，是逐图 min-max 着色所致；秩次口径下 raw 本来就找得到主体（AUC 0.736–0.784）。

---

## P4 — 找主体成立的证据图

⚠ **先读这一段再选图**：RO9c 补件 B/C 已经把"找主体成立"这句话的范围收紧了 ——

- **RO9c 自己的 6 个读出（raw L/GC/SC、M1 L/GC/SC）全部输给一个不看图像的几何中心先验（0.836）**，四个 `subject_area` 分档上无一跑赢（`REPORT.md` §9.2）
- **战役里真正跑赢中心先验的只有两条**（§10.1，同批源、同掩膜、同一套 AUC）：
  - **RO-1 ClearCLIP `+A3+A1+A2`**：0.961（canonical16）/ 0.930（原生网格），配对 Δ=+0.072 / +0.056，p ≤ 7.2e-06
  - **RO-3 L11H5 区域对立差分场**：0.927，Δ=+0.089，p=3.1e-19
  - 两者在四个 `subject_area` 档上**全胜**，主体越大优势越大

### P4-a　最诚实的一张：跨臂 vs 中心先验（**推荐做本页头图**）

| 用途 | 绝对路径 | 尺寸 |
|---|---|---|
| **战役全部落盘读出场 vs "不看图像的场"的配对 Δ** | `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_center_prior_crossarm.png` | 2185×977，0.18 MB |
| 中心先验分档曲线 + 配对散点 + 净增益 vs subject_area（**只覆盖 RO9c 自己的读出**） | `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_center_prior.png` | 2040×576，0.12 MB |

### P4-b　逐案例成功图（按"谁跑赢中心先验"排序取用）

| 臂 | 路径（前缀 `/home/bc/VeraRetouch/experiments/`） | 尺寸 | 版式 |
|---|---|---|---|
| **RO-3 差分场（跑赢）** | `RO3_layerhead_scan_20260803/viz/success_diffield_L11H5_auc0.991_ppr10k_6226_a.png`（另有 `auc0.992_ppr10k_1046_a` / `auc0.993_src_bad7b0ad1e9e5c60`） | 2400×390 | 源图 / 掩膜 / s(reg_a) / s(reg_b) / **差分场** / **同区域对照差分** / luma |
| **RO-1 ClearCLIP（跑赢）** | `RO1_selfself_20260803/viz/success_00_auc0.996_src_225a0cd3296ee6b1.png`（另 3 张 `success_01..03`，AUC 0.992–0.994） | 1985×393 | 原图 / GT / s(目标名词) / s(错位名词) / s(补集词) |
| RO-9c M1（**跑输中心先验，须配限定**） | `RO9c_subject_repro_20260805/viz/success_aucM10.981_ppr10k_2067_a.png`（另 `success_aucM10.981_src_ab66bfff2833c0b2` / `success_aucM10.982_ppr10k_6993_a`） | 3872×1066，3.6 MB | 6-09 版式 + GT |
| RO9b 修复后（**跑输**） | `RO9b_readout_fix_20260803/viz/success_03_ppr10k_8652_a_auc0.901.png`（另 3 张 0.681–0.888） | — | 原图/掩膜/RO-9 原样/RO9b 修复后/reg_b/错位/视觉塔 vanilla/视觉塔最优，共 8 格 |

### P4-c　"形状对不对"的证据（补件 D，说明 AUC 高 ≠ 能当 mask）

| 用途 | 绝对路径 | 尺寸 |
|---|---|---|
| 9 源 × 6 列：input / 中心先验 / raw L / raw GC / M1 GC / GT，白实线=面积匹配 top-k、红虚线=GT | `/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_four_fields.png` | 2190×3728，**9.54 MB**（PPT 里需先压） |

读法：中心先验那一列永远是画面中央**一个紧凑圆斑**（连通块数中位 1、边界格 12）；attention 那三列是**散落 7–10 块**碎片（边界格 23）。重叠类指标（AUC / soft-IoU / hard-IoU / point∈GT）中心先验 **4/4 第一**；边界类指标（bf1 3px / bf1_grid）中心先验 **4/4 垫底**。

---

## P7 — G1 核心图：同一张图两条相反指令 → 两张几乎相同的注意力场

**全部现成，无需生成。** 路径前缀 `/home/bc/VeraRetouch/experiments/G1_s_identifiability_20260803/viz/`。

### P7-a　⚑ 首选：`region_FAIL_rho1.00_maxrho_*`（ρ≈1.00 的极端，**6 列含差图**）

| 文件 | 尺寸 |
|---|---|
| `region_FAIL_rho1.00_maxrho_ppr10k_1046_a.png` | 2618×446 |
| `region_FAIL_rho1.00_maxrho_ppr10k_8693_a.png` | 2618×446 |
| `region_FAIL_rho0.99_maxrho_src_0b9883296a73e9f7.png` | 2618×446 |
| `region_FAIL_rho0.98_maxrho_src_8fb50b056e50afe1.png` / `_src_547b03c38c66d6e4.png` / `_src_23547f0fcb1b7bb1.png` | 2618×446 |

版式（REPORT §六）= 源图 / **s(reg_a：点名主体)** / **s(reg_b：点名补集)** / **s(reg_a) − s(reg_b) 差图** / SAM3 主体掩膜（带 AUC(a)、AUC(b)）/ luma16。

> **差图是 ρ 之外最直观的一张**：即使在最分离的样本上，差图也是一片**接近均匀的整体偏移**，而不是"主体区亮起、补集区暗下去"的区域交换。

### P7-b　⚑ 配套：`failure_instrblind_*`（"说背景反而更压主体"，**带 AUC 数字**）

| 文件 | 尺寸 | Δ=AUC(b)−AUC(a) |
|---|---|---|
| `failure_instrblind_1_aucB0.72_gt_aucA0.63_src_6323567d36a72342.png` | 2100×600 | +0.090 |
| `failure_instrblind_2_aucB0.84_gt_aucA0.77_ppr10k_7306_a.png` | 2100×600 | +0.074 |
| `failure_instrblind_3_aucB0.70_gt_aucA0.64_src_562b7ed9fceb7d49.png` | 2100×600 | +0.068 |

版式 = 源图 / SAM3 主体掩膜 / s(A：说主体) / s(B：说背景)，两张 s 叠主体掩膜红色轮廓，各带 AUC。
选样规则**代码常量写死、非事后挑图**：`region_b_kind=background` ∧ `AUC(b)>AUC(a)` ∧ `AUC(b)≥0.70` ∧ 主体面积≥6%，按 Δ 降序取前 3；**候选池 17 源 / 区域批 214**。

### P7-c　"连最好的那几张也远离判据"（可选，做 backup）

`region_FAIL_rho0.58_minrho_ppr10k_2619_a.png` 起 6 张（ρ 0.585–0.65，**全部仍 FAIL**，判据 <0.30）。

### P7-d　同一现象的另一条独立证据（G1b，A&S 读法下）

| 用途 | 绝对路径 | 尺寸 |
|---|---|---|
| **A&S 相对 canonical 分离最少的 3 例 = 两条对立指令下的场几乎重合** | `G1b_difflmm_20260803/viz/aasworst_paired_aas0.876_canon0.772_src_61a87d742f415b8d.png`（另 `aas0.709_...` / `aas0.739_...`） | 2365×968 |
| A&S 分离最多的 3 例（**即使这里 A&S 的场也只是 1–2 个孤立亮格，与 GT 无关**） | `G1b_difflmm_20260803/viz/aasbest_paired_aas-0.027_canon0.970_ppr10k_7793_a.png`（另 2 张） | 2365×968 |

版式 = 2 行（canonical / attend-and-segment）× 5 列：源图·GT(C_GT) / s(reg_a) / s(reg_b) / **s(reg_a_para 同区域换句框＝匹配噪声地板)** / s(a)−s(b) 差分场。

---

## P9 — 端到端条件消融（给指令 / 固定指令 / 不给指令）

### ✅ 已生成

| 用途 | 绝对路径 | 尺寸 / 大小 |
|---|---|---|
| **三档并排柱状图**（掩膜 AUC / 颜色方差比 / Δ_shuffle 三面板） | `/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs/P9_condition_ablation.png` | 2625×1140，0.44 MB |
| 生成脚本（可复核，零 GPU，<1 min） | `/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs/make_P9_condition_ablation.py` | — |

图上数字全部由脚本从 `runs/<arm>/metrics.json → .final.select` 读出，**脚本内不硬编码任何实测值**；
另断言三个 arm 的 `n` 与 `manifest_digest` 一致（唯一变量确实只有 condition）。
**纵轴一律从 0 起、未截断** —— 本页的说服力来自「三根柱子一样高」。

### 三个 arm 的 condition_mode 语义

（`train.py:113-135` `condition_inputs`）：

| arm 目录 | `condition_mode` | 实际喂进去的东西 |
|---|---|---|
| `runs/config_a` | `full` | 图 + **本样本真实指令** ← 完整输入 |
| `runs/condition_image_only` | `image_only` | 图 + **一条中性固定句** `"Retouch the requested region of this image."`（`train.py:115`） ← **"不给指令"** |
| `runs/condition_fixed_shuffle` | `fixed_shuffle` | 图 + **别的样本的指令**（确定性错位，`data.py:116`） ← **"给错指令"** |
| `runs/condition_instruction_only` | `instruction_only` | **全零图** + 真实指令（`train.py:132-133`） ← "不给图" |
| `runs/condition_short` | `short` | 图 + `instruction_short`（形态对照，非因果对照） |

### 已落盘的数字（**这就是柱状图的全部输入**）

来源：`experiments/MCQ_full_local_l1l6_20260804/runs/<arm>/metrics.json → .final.select`（n=4225，完整 select 池）

| arm | ΔE00 p50 ↓ | ΔE00 p90 ↓ | PSNR_in p50 ↑ | **AUC_cgt p50** | soft-IoU p50 | **Δ_shuffle (dB)** | Δ_const (dB) | Δ_AUC_shuffle |
|---|---|---|---|---|---|---|---|---|
| **config_a（完整）** | **3.560** | 7.086 | **22.59** | 0.9469 | **0.605** | **1.901** | 0.780 | +0.0024 |
| condition_short | 3.545 | 7.074 | 22.53 | 0.9461 | 0.584 | 1.787 | 0.829 | +0.0018 |
| **condition_image_only（不给指令）** | **3.995** | 8.310 | **20.29** | **0.9481** | 0.509 | **0.000** | 0.244 | **0.000** |
| **condition_fixed_shuffle（给错指令）** | **4.011** | 8.266 | **20.41** | **0.9498** | 0.508 | **−0.000** | 0.221 | **−1.616** |
| condition_instruction_only（不给图） | 3.811 | 7.467 | 22.03 | 0.8110 | 0.448 | 1.407 | 0.455 | +0.0074 |

**⚑ 本页最该讲的一句**：**掩膜 AUC 三档几乎相同（0.9469 / 0.9499 / 0.9481），给错指令的那档甚至最高**；真正分开的是颜色方差比（0.317 → 0.067 / 0.063）、像素指标（ΔE00 3.560 → 4.011 / 3.995，PSNR_in 22.59 → 20.41 / 20.29）和 Δ_shuffle（1.901 dB → ≈0）。

### ⚠ 讲这页时的用词提醒

`condition_image_only` 喂的是**一条对所有图逐字相同的中性句**（不是「什么都不喂」），
`condition_fixed_shuffle` 喂的是**别的样本的真实指令**（逐样本不同、但都是错的）。
图上已按代码语义标注，PPT 转述时别把两者对调。

### 若还要分层版 / 带 CI 的版本

- 分层：`runs/<arm>/metrics.json → .final.select.by_level`（L1–L6 六档，字段与总表相同）
- 逐样本（做配对检验用）：`runs/<arm>/per_sample_select.jsonl`（每行含 `uid/source_id/level/psnr_in/de00/auc_cgt/soft_iou/delta_const_db/delta_shuffle_db/delta_auc_shuffle`）

### 若还要并排样例图（需 GPU）

- **`condition_*` 四个 arm 都没有出过 gallery**；只有 `viz/config_a/`、`viz/config_b/`、`viz/renderer_gaussian3d_alpha/` 三个 arm 有
- 现成可用：`/home/bc/VeraRetouch/experiments/MCQ_full_local_l1l6_20260804/viz/config_a/gallery20_overview.png`（528×1080）+ `gallery20_p01..p04.png`
- 生成 condition arm 的：
  ```
  python experiments/MCQ_full_local_l1l6_20260804/visualize_final.py \
    --run-dir experiments/MCQ_full_local_l1l6_20260804/runs/condition_image_only \
    --device cuda:X \
    --out-dir experiments/MCQ_full_local_l1l6_20260804/viz/condition_image_only
  ```
  20 个固定样本，需加载 `best.pt` + VLM 前向 ⇒ **单卡数分钟 / arm**（`logs/viz_config_a.log` 那次的产物落盘在启动后 ~10 s 内完成，模型加载是主要开销）

---

## P10 — RO-3 query 端对照（全场最有说服力的一张）

### ⚠ 数字更正（**任务卡里的 0.58 → 0.93 把两件事混了**）

从 `experiments/RO3_layerhead_scan_20260803/metrics_diffield.json` 按其 `index_note`
（`r = ((mp*24 + layer)*14 + head)`，`MP = [(pre,instr),(pre,last),(pre,alltxt),(pre,gl),(post,instr),(post,last),(post,alltxt),(post,gl)]`，
出处 `analyze_ro3.py:37-39`）直接取值：

| 口径 | `gl` 池（special token 当 query） | `instr` 池（指令文本 token 当 query） |
|---|---|---|
| **同一层同一头 L11H5，`pre`** | **0.4991**（FWER p = **1.0**） | **0.9298**（FWER p = **0.000**） |
| 同一层同一头 L11H5，`post` | 0.4678（p = 1.0） | 0.8987（p = 0.000） |
| **全 336 个 (层,头) 取 max，`pre`** | **0.5839**，argmax = **L8H9**（**不是 L11H5**） | 0.9298，argmax = L11H5 |
| 全 336 取 max，`post` | 0.5982，argmax = L11H8 | 0.8987，argmax = L11H5 |

⇒ **真正的"同一层同一个头"对照是 0.499 → 0.930**，比 0.58→0.93 **更强**。
**0.5839 是"允许 gl 池在全场 336 个头里挑最好的一个"的上限**（且它落在 L8H9，FWER p=0.982，连置换零分布 q95=0.596 都没超过）。
汇报时二选一，别混着说：
- 想说"同层同头"⇒ **0.50 → 0.93**
- 想说"给 special token 全场最优待遇也不行"⇒ **0.58（全场 max，p=0.98） vs 0.93**

### 已有素材（能立刻用，但不是"两格并排"的形态）

| 用途 | 绝对路径 | 尺寸 |
|---|---|---|
| **8 面板 × 24 层 × 14 头差分场 AUC 热力图（SAM3 GT）** —— `pre/instr` 面板 L11H5 一个亮点，`pre/gl` 面板全场冷 | `/home/bc/VeraRetouch/experiments/RO3_layerhead_scan_20260803/viz/heatmap_layerhead_DIFFFIELD_auc.png` | 2420×1210，0.20 MB |
| **同上但 GT=`.cgt.png`（论文主图）**，带**绿等高线 = max-stat 置换零分布 95% 阈值 0.739** | `.../viz/heatmap_336_layerhead_DIFFFIELD_cgtGT.png` | 2645×1322，0.32 MB |
| AUC_target 口径同网格 8 面板 | `.../viz/heatmap_layerhead_auc_target_ALL.png` | 2420×1210 |
| 差分场 AUC / 同区域对照 / FWER p 三张 + **置换零分布直方图与实测值** | `.../viz/diffield_best_and_null.png` | 2760×690，0.11 MB |
| 逐层曲线（中层带蓝、末两层带红） | `.../viz/layer_curves.png` | 1725×598 |
| 融合 top-k 曲线（含 0.75 晋级门） | `.../viz/fusion_topk_curve.png` | 864×552 |

**只用现成图的方案**：把 `heatmap_layerhead_DIFFFIELD_auc.png` 裁出 `pre/instr` 与 `pre/gl` 两个面板并排 —— 零成本，但读者需要看色标才能读出 0.93 vs 0.50。

### ✅ 已生成（两档都做了，零 GPU，合计 wall < 10 s）

| 用途 | 绝对路径 | 尺寸 / 大小 |
|---|---|---|
| **档 1 · 同层同头对照柱状图**（gl 0.4991 vs instr 0.9298 + 同区域零对比度对照 0.5220） | `/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs/P10_query_position.png` | 2130×1410，0.31 MB |
| **档 2 · 逐案例 4 列图**（5 源 × `input / gl 场 / instr 场 / GT`） | `/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs/P10_query_position_cases.png` | 2106×2275，1.6 MB |
| 生成脚本（两张图一起出；`--verify` 可复核） | `/home/bc/VeraRetouch/docs/presentation_2026-08-05/figs/make_P10_query_position.py` | — |

**档 1 图上的数与出处**：`gl` 0.4991（FWER p = 1）／`instr` 0.9298（FWER p = 0），
均由脚本按 `metrics_diffield.json` 的 `index_note` 公式从 `scan_auc_diff_bg` / `scan_fwer_p_bg` 取出；
两条参考线 = 随机基线 0.5 与 `verdict.null_max_q95` = **0.8416**（max-stat 置换零分布 95% 阈值，已控 2688 路多重比较）；
第三根灰柱 = `verdict.best.auc_diff_ctrl_same_region` = **0.5220**（同一读出、同区域零对比度差分）。
图内橙色提示框已按裁定**单列**两个全池上限、明确写"不可混着说"：
差分场口径 **0.5839 @ L8H9**（min FWER p = 0.982）／AUC_target 口径 **0.5542 @ L6H5**（低于该口径零分布 q95 0.5958）。

**档 2 的正确性自检（脚本内置 `--verify`，已跑过）**：本脚本的轻量场提取在全部 214 源上复算
background 子集中位 = **gl 0.4991 / instr 0.9298**，与 `metrics_diffield.json` **逐位一致**（n_used=212 / n_bg=165，与落盘 counts 相同）。
场的提取与 D-0 修复复用 `analyze_ro3.interp_batch`，AUC 用 `analyze_g1.roc_auc`，掩膜用 `analyze_g1.SubjectMaskBank`。

**档 2 的选源规则（effect-blind，写进图注）**：直接复用 RO-9c 已冻结的
`experiments/RO9c_subject_repro_20260805/config/figure_picks.json` —— 与 P3 的七列消融图**同一批源**。
逐源实测（脚本打印，可复核）：

| 源 | pool / reg_b | 主体面积 | gl 差分场 AUC | instr 差分场 AUC |
|---|---|---|---|---|
| `src_08aef2f575be97b1` | awards / spatial | 0.263 | 0.189 | **0.927** |
| `src_0e6178111d466cfb` | unsplash / background | 0.253 | 0.657 | **0.972** |
| `ppr10k_0579_a` | ppr10k / background | 0.318 | 0.573 | **0.959** |
| `ppr10k_2164_a` | ppr10k / spatial | 0.156 | 0.380 | **0.904** |
| `ppr10k_0815_a` | ppr10k / background | 0.169 | 0.262 | **0.990** |

着色纪律：发散色标以 0 为中心、**只用有效格定对称上下限**（RO-9c C1 的教训：不许让黑边格决定 min-max 的分母）；
黑边格画成白色并在图注注明；**着色仅用于出图，所有 AUC 都是秩次量、不经过着色**。

### 配套可引用的对照数（写在图注里）

| 量 | `gl` 池 | `instr` 池 |
|---|---|---|
| max AUC_target（全 336 头） | **0.5542**（零分布 q95 = 0.596，**没超过零分布**） | **0.7660**（L11H5，FWER p=0.000） |
| max Δ_shuffle | +0.0117 | +0.0985 |
| head-mean 通道（= RO-9 落盘口径）的差分场 AUC | **0.5129**，同区域对照 **0.5293**（对照还更高） | — |
| 生成位 vs 追加位（U3 补件 §12.2） | 追加位 0.5743 → 生成位 0.5908（Δ+0.017），**两位都低于零分布 q95 0.596** | 追加位 0.7541 → 生成位 0.7549（Δ+0.0008，**等价**） |

> ⚑ **U3 补件必须提**：RO-3 的 `gl` 池是在 **prefill 末尾追加** special token，与 G1/RO-9 读**生成位**不同。补件 §12 用 100 源 × {reg_a,reg_b} 带 greedy 生成专门比过这两个位置：`instr` 池两位**逐元素等价**（空间 ρ 中位 0.999936），`gl` 池两位**不是同一张图**（ρ 0.36–0.61，argmax 从 L06H05 漂到 L19H07），生成位略强但**仍在噪声地板内**。

---

## P11 — 三个"漂亮数字骗人"的配图

### ① RO-X1：一条对所有图相同的短语拿到几乎相同的 AUC，但可控性归零

- 数字：`X_deictic = "the main subject"` AUC **0.907** vs 逐源长描述 `N_desc` **0.930**，配对差 **+0.009，p=0.133**；同一条件下 **AUC_target 塌到 0.523**（CI [0.396, 0.649]）
- **图（现成，且自带反转叙事）**：`/home/bc/VeraRetouch/experiments/ROX1_clipside_20260803/viz/failure_nounsep_02_ppr10k_0860_a.png`（1985×390）
  —— 同一张图：`N_desc`（**有名词**）AUC **0.084**（场压在铁丝网上），`X_deictic`（**无名词固定短语**）AUC **0.939**（干净压在主体上）。**给了名词反而全错，不给名词反而全对。**
- 备选：`ROX1/viz/success_nounsep_00_src_c6c901e4652afd90.png`（1985×397，名词确实带来增益的一端）
- 版式 = 原图 / GT 掩膜 / s(`N_desc`) / s(`X_deictic`) / s(`X_style`)，后三格叠 GT 红轮廓并标 AUC

### ② RO-9b：四处修复把 AUC 拉到 0.92–0.93（追平免费 CLIP），但一分指令依赖也没买到

- 数字：AUC **0.661 → 0.9200**（零学习参数，CI [0.907,0.927]）/ **0.9349**（学 24 个层权重）；同一路径上 **AUC_target 0.504 / 0.385**，**Δ_shuffle +0.0003 (p=0.813) / −0.0006 (p=0.365)**；指令依赖天花板只有 AUC_target **0.5719**（CI 下界 0.524，**刚越过 0.5**），Δ_shuffle **+0.022**（RO-1 的 **1/10**）
- **主图（现成）**：`/home/bc/VeraRetouch/experiments/RO9b_readout_fix_20260803/viz/waterfall.png`（1875×2100，0.39 MB）
  —— 四段增量 × 两个准则 × 受限/全档共**四条路径**，带 CI 与四条基线锚点横线
- **逐样本配套（现成）**：`RO9b/viz/failure_instrblind_00_src_8128f123621c4cbf_auc0.610.png` 等 **4 张** —— `AUC(错位指令) ≥ AUC(自己的指令)` 的源（0.448–0.691），"AUC 高 ≠ 听懂指令"的逐样本证据
- 备选：`RO9b/viz/layer_curves.png`（1170×585，`AUC(s_a,M)` 与 `AUC(s_b,M)` 逐层几乎重合）

### ③ RO-9c：一个完全不看图像的几何中心先验，打赢本实验全部读出

- 数字：中心先验主体 AUC **0.836**（`−到画幅中心的格距离`，**完全不看图像**）> raw L **0.784** > raw SC 0.749 > raw GC 0.736 > M1 GC **0.695** > M1 L 0.669 > M1 SC 0.475；配对 Δ = −0.064 … −0.320，p ≤ 4.2e-4，胜出率仅 **6–35%**；四个 `subject_area` 分档**无一跑赢**
- **图（现成）**：`/home/bc/VeraRetouch/experiments/RO9c_subject_repro_20260805/viz/diag_center_prior.png`（2040×576）
  —— 三面板：分档曲线 / raw L vs 中心先验配对散点 / 净增益 vs subject_area
- **同一页可加的第二张（现成）**：`RO9c/viz/diag_raw_colormap.png` —— **同一张场，min-max 着色说"raw 无信号"，AUC 说 0.784**（着色骗人）
- **中心先验没有可调自由参数**（补件 D §11.4，数值验证）：高斯 σ=2/3/5 与主档 `−L2` **全部指标逐位相同**；换度量（L1 0.8382 / L∞ 0.8121）结论稳健，**且主档不是最高的那个**

### 备选④（若想讲"ρ 掉了看起来像巨大进步"）—— G1b

- 数字：A&S 把 ρ_region_opp 从 **0.884 → 0.439**（p=8.8e-37，35/214 跌破 0.3 判据），**看上去是巨大进步**；但**匹配噪声地板同步塌到 0.445**，`sep_matched = +0.0011`（p=0.46，**逐源胜出率 50.0%**）；`AUC_target` 仍 **0.496**（p=0.77）
- 图（现成）：`/home/bc/VeraRetouch/experiments/G1b_difflmm_20260803/viz/aasbest_paired_aas-0.027_canon0.970_ppr10k_7793_a.png`（2365×968）—— 即使在 A&S 分离最多的样本上，场也只是 1–2 个孤立亮格
- 配套数：A&S 减完共模后 **单个格子吃掉 41.7% 的空间方差**（canon 0.083，均匀场参照 0.006）

### 备选⑤（端到端侧的同一现象）—— MCQ

- 数字：**不给指令的 `condition_image_only` 掩膜 AUC 0.9481 ≥ 完整输入 config_a 的 0.9469**；给错指令的 `condition_fixed_shuffle` 更高（0.9498）
- 图：与 **P9 的柱状图共用一张**（AUC 那一组三档打平，ΔE00/PSNR/Δ_shuffle 那几组分开）

### 备选⑥（若要讲"指标本身选错了"）—— RO-9c 补件 D

- 数字：**3px 像素级边界 F1 的随机 top-k 零模型 0.0394 > 中心先验 0.0327**（该口径在 16×16 上主要在测"预测边界有多长"）；而网格级 `bf1_grid` 的随机零模型 0.378 **显著低于**中心先验 0.548（p=1.1e-11）⇒ 只有后者可信
- 图（现成）：`RO9c/viz/diag_four_fields.png`（2190×3728，9.54 MB）

---

## 附：本清单核对过但**没有**现成图的项

| 想要的图 | 状态 | 最省的生成路径 |
|---|---|---|
| ~~P9 三档条件消融柱状图~~ | ✅ **已生成** | `figs/P9_condition_ablation.png`（脚本 `figs/make_P9_condition_ablation.py`） |
| ~~P10 "同层同头两个 query 并排"专图~~ | ✅ **已生成（两档）** | `figs/P10_query_position.png` + `figs/P10_query_position_cases.png`（脚本 `figs/make_P10_query_position.py`，带 `--verify`） |
| P9 condition_* arm 的 20 样本并排 | **无**（只有 config_a/config_b/renderer_gaussian3d_alpha 有） | `visualize_final.py --run-dir runs/condition_image_only ...`，单卡数分钟/arm。**本轮未做**（柱状图已足够说清三档） |
| RO9b "0.920/0.935 档 vs 中心先验"的配对 | **无法配对，需重跑**（RO9c §10.6）：场未落盘，`config/fields_final.npz` 只存了 `auc_target` 选优档（AUC 0.728）；0.935 那档需按 5 折 OOF 重新拟合层权重 | 逐头栈在 `/var/cache/veradata/ro9b_stacks_20260803`，**零 GPU 可重跑**。重跑必须标注：该档层权重是**在目标 GT 上有监督拟合**的 |
| RO-1 / RO-X1 其余组合 vs 中心先验的口径 A 配对 | 场未落盘（RO-1 只有 `clearclip × +A3+A1+A2 × desc` 一档写进 scache；RO-X1 9 条短语全未落盘） | 已用口径 C（原生网格 + 落盘逐源 AUC）覆盖，见 RO9c §10.3 |
