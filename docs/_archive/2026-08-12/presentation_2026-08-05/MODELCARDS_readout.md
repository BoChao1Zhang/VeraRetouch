# 模型细节卡 · 读出线（VLM 能不能读出 where）

- 生成时间：2026-08-05
- 规则：**每个数字都注明出处**（文件:行 或 json 键路径）。追不到的一律写「**缺**」，不估算、不编造。
- 出处路径若以 `experiments/` 开头，根为 `/home/bc/VeraRetouch/`。

---

## 卡 0 —— 共享底座（RO9c / G1 / RO9-L / RO9b / G1b / RO-3 / MCQ 七张卡共用，先讲一次）

| 项 | 值 | 出处 |
|---|---|---|
| 权重路径 | `/home/bc/data/models/VeraRetouch` | `experiments/*/config/env.json → model_path` |
| 架构名 | `SamanthaForCausalLLM_Unified` / `model_type = llava_qwen2` | `/home/bc/data/models/VeraRetouch/config.json` |
| **语言塔** | **Qwen2-0.5B**：`num_hidden_layers=24`，`hidden_size=896`，`num_attention_heads=14`，`num_key_value_heads=2`（**GQA groups=7**），`intermediate_size=4864`，`rope_theta=1e6`，`max_position_embeddings=32768` | 同上 config.json；`experiments/RO3_layerhead_scan_20260803/config/verify_env.json → V1_versions` |
| **视觉塔** | `mm_vision_tower = mobileclip_l_1024` = **FastViTHD**；`layers=[2,12,24,4,2]`，`embed_dims=[96,192,384,768,1536]`，`token_mixers=(repmixer,repmixer,repmixer,attention,attention)`；**只有 stage3（`network[7]`，4 块，输入 (1,768,32,32)，24 头）与 stage4（`network[10]`，2 块，输入 (1,1536,16,16)，48 头，head_dim 32）有 self-attention**；**全程 (B,C,H,W)，没有 cls token、没有 register token**（全局表征靠 `GlobalPool2D`） | `experiments/RO9b_readout_fix_20260803/REPORT.md` §5.1（一手核实 Apple `ml-fastvlm` 源码 + 本机 `config/verify_env.json` V1–V3） |
| **connector** | `mm_projector_type = mlp2x_gelu`，`mm_hidden_size = 3072 → 896` | config.json |
| **image token 数与网格** | **256 个 image token = 16×16**；`<image>` 走 llava 占位 `IMAGE_TOKEN_INDEX=-200`，由 `prepare_inputs_labels_for_multimodal` 展开成 256 个视觉 embedding | `verify_env.json → V5_n_image_tokens / V5_grid / V6_note` |
| **image span 位置** | prompt 里 image token 段 = **(14, 270)**，恒定；**指令排在 image 之后** | `verify_env.json → V5_image_span`；`experiments/G1b_difflmm_20260803/REPORT.md` §1.3（与 RO-2 逐位一致） |
| 图像预处理 | `image_aspect_ratio = "pad"` ⇒ **`expand2square` 补黑边成方形** | config.json；黑边后果见下方 ⚠ |
| **三个 retouch special token** | `<retouch_light>` **151646** / `<retouch_color&temp>` **151647** / `<retouch_colormixer>` **151648**，均为单 token | `verify_env.json → V6_token_ids`；`G1b/REPORT.md` §1.1（`added_tokens.json` + tokenizer 运行时实测两处一致） |
| **它们在序列里的位置** | prompt 里**一个都没有**，全部在 assistant **生成段末尾且相邻**（典型 `gen_len=311` 时 light=307 / colortemp=308 / colormixer=309）⇒ **因果掩码下完全看得见指令** | `G1b/REPORT.md` §1.2（实测） |
| **checkpoint 参数量（实测，从 safetensors header 逐张量累加）** | **总计 634,075,663**；其中 LM 24 层 **358,062,336** + `embed_tokens` **135,890,944**（tied，无独立 lm_head）= **493,953,280 ≈ 0.5B**；视觉塔 **125,109,260**；`mm_projector` **3,557,120**；生产用 `retouch_head`/`retouch_decoder` **11,620,227** | 本次实测 `/home/bc/data/models/VeraRetouch/model.safetensors` header（可复算） |
| **⚑ 是 0.5B 不是 0.6B** | 语言塔 493.95M ≈ 0.5B（Qwen2-0.5B）。**634M 是含视觉塔+projector+生产头的整包数**，不是语言塔规模 | 同上 |
| attention 实现 | **eager**，加载后断言 `config._attn_implementation == "eager"`，前向后断言 `attentions is not None` 且层数==24；不满足直接 raise | `RO9c/config/env.json → attn_implementation`；`G1b/REPORT.md` §二「红线守卫」 |
| **实测红线**：sdpa 不能用 | transformers 4.57.1 下 `attn_implementation="sdpa"` + `output_attentions=True` → **`attentions is None`，只 warning、不回退**；eager 下返回 24 个 **(1, 14, S, S)** 张量 | `verify_env.json → V3_sdpa_attentions_is_none / V2_attn_shape` |
| pre- 与 post-softmax 一致性 | `softmax(捕获的 pre-softmax logit)` 与 eager 返回权重逐元素 `max|Δ| = 1.95e-3`（= bf16 舍入量级） | `verify_env.json → V4_max_abs_diff_softmax_vs_returned` |

### ⚠ 三条会被 mentor 追问的已知缺陷（涉及可视化的页必提）

| # | 事实 | 数字 | 出处 |
|---|---|---|---|
| 1 | **`expand2square` 黑边格吃掉大部分 image 注意力质量** | 16 格中约 **5.3 格是纯黑边**（有效格占比中位 0.625）；黑边格占 image 注意力质量中位 **68.8%(L) / 53.4%(GC) / 74.4%(SC)**；**93.0%** 的源全局 argmax 落在黑边格。**AUC 只在 valid 格上算 ⇒ 该 sink 从未进入过 AUC** | `RO9c/REPORT.md` §9.1；`RO9c/metrics.json → sink_location_diagnostic` |
| 2 | **6-09 的 `render()` 网格错位** | 把 16 格直接 `resize` 到**未 pad** 的图上，网格却覆盖 **pad 后的方形** ⇒ 约 **1/3** 系统性错位。RO9c 主交付图已用 `make_figs.grid_to_img` 修正（标题带 `[GRID-ALIGNED]`），`repro_*_raw/_m1.png` **故意保留原错位**作对照 | `RO9c/REPORT.md` §9.1 混杂 B、§9.5 |
| 3 | 黑边问题**不只 RO9c 中招** | RO-3 L11H5 单场：黑边质量 **0.435**、argmax 落黑边 **60.6%**；RO9b `s_fin_a`/`s_base_a`：**0.523/0.506**；RO-3 **差分场**把它消掉了（0.223/0.138，argmax 落黑边 1.7%） | `RO9c/REPORT.md` §10.5 |
| 4 | **共享模型目录被 6-09 那次实验改坏后从未恢复** | `/home/bc/data/models/VeraRetouch/generation_config.json` 被 `os.rename` 成 `.generation_config.json`，**本轮战役全部读出臂都是在这个状态下跑的**；RO9c 未擅自恢复（恢复会让后续臂与已完成臂处在不同 generation 默认值下）。**这是一个待主 agent 处置项 D-6** | `RO9c/REPORT.md` §八 D-6；`RO9c/config/env.json → model_dir_note`；本次实测目录列表确认该文件仍是 `.generation_config.json` |

---

## 卡 1 —— RO-9c（6-09 复现）

> ⚠ 该目录**另一个 agent 正在写**，本卡数字取自 2026-08-05 16:01 的 `REPORT.md` 与 15:53 的 `metrics.json`。

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | 6-09 那张"retouch token 注意力经 DiffLMM 序列均值减法（M1）后落在主体上"的图，**是不是单张 `sample_flower.jpg` 的偶然**——用 214 源 + SAM3 掩膜量化。 |
| **模型** | 卡 0 底座。RO9c 自己的 env 复述：`24 层 / 14 头 / 256 image token / 16×16 网格`（`RO9c/config/env.json → n_layers, n_heads, n_image_tokens, grid`）。视觉塔 mobileclip_l_1024。参数量见卡 0。 |
| **输入** | 图 → `expand2square` → 视觉塔 → 256 image embedding，插在 prompt 的 (14,270)；指令排在其后；**3 个 retouch special token teacher-force 接在 prompt 末尾**（= 6-09 口径，**不是**生成轨迹内位置）。一次 eager prefill，**无 `generate()` 调用**。序列长度中位 **330**（`metrics.json → run_meta.seq_len_median`）。attention 每层 `(1, 14, S, S)`。 |
| **输出** | 每次前向落一个 npz（`run/stacks/<img_id>__<cond>.npz`）：raw 逐层注意力 / `common` / `common_prompt` / D-0 掩膜。读出量：`red = A.mean(layer).mean(head)` → 取 256 个 image 列 → `common = red[图像之后所有 query 行].mean(0)`（`n_post_rows_median = 60`）→ `M1 = row − common` ⇒ **每 token 一张 16×16 场**。共 **856 条**（`config/env.json → n_forward_passes`，`metrics.json → n_readouts_loaded = 856`）。 |
| **训练/推理** | **零训练、纯推理**。prefill-only，**856 次前向**，GPU 1（H100），wall **3.9 min**，err=0。（`config/env.json → wall_clock_readout`） |
| **数据** | **S-val**，复用 G1 已冻结的区域对立批 `experiments/G1_s_identifiability_20260803/config/g1_region_opp.json`，**214 源、零重新采样**（快照 `config/g1_region_opp.SNAPSHOT.json`）。池 awards 76 / unsplash 75 / ppr10k 63；`region_b_kind` background 165 / spatial 49；`winner_confidence` normal 152 / low 62。**有效 n=194**（20 源被 AUC 有效性门剔除：二值化后主体<4 格或背景<4 格）。4 个 prompt 条件：`auto`（6-09 原样零指令，主档）/ `reg_a` / `reg_b` / `fixed`。GT = D-MASKBANK 的 SAM3 `cache/subject` 软掩膜，经 G1 的 `SubjectMaskBank` + `luma_to_grid`（复用 `G1b/config/subject_masks16.npz`，214/214 覆盖）。（`REPORT.md` §三；`config/env.json → data`） |
| **判据（预注册 vs 实测）** | 见下表 |
| **seed / commit** | seed **20260805**；commit **a8d10a6460c8be0748dd6aea1e17cdb7a96f1add**，branch `lens-exp`（`config/env.json`） |

| 判据 | 预注册 | 实测 | 判 | 出处（json 键路径） |
|---|---|---|---|---|
| 主判据-a `mass-in-GT(M1) > (raw)` | Wilcoxon p < 0.01 | Δ = **+0.03950**，**p = 2.81e-19**，80.9% 源胜出 | **PASS** | `metrics.json → verdict.main_mass_m1_gt_raw` |
| 主判据-b M1 主体 AUC | **≥ 0.75** | **0.6949** [0.666, 0.714] | **FAIL** | `verdict."main_subject_auc_m1_ge_0.75"`；`per_condition.auto.m1.colortemp.auc.median` |
| 定性-a raw 三 token 两两余弦 | > 0.90（三对全部） | L~GC **0.9098** ✔ / L~SC **0.9684** ✔ / GC~SC **0.8821** ✘ | **FAIL**（差 0.018） | `verdict."qual_raw_cross_token_cos_gt_0.9"` |
| 定性-b M1 后可分性上升（复现 6-09 的 0.714） | M1 > raw | raw **0.0817** → M1 **0.8236**（6-09 记载 0.714） | **PASS**（比 6-09 更强） | `verdict.qual_m1_cos_drops` |
| 边界 `AUC_target` | 落在 0.45–0.60；≥0.65 标红 | **0.4871** [0.456,0.508]，vs 0.5 p=0.183 | **红旗未触发** | `verdict.boundary_auc_target` |
| **整体** | — | `main_criterion_overall_pass = false` | **MIXED** | `verdict.main_criterion_overall_pass` |

**补件 B / C / D 的三个必讲数（都在 `metrics.json` 里可复算）**

| 结论 | 数字 | 键路径 |
|---|---|---|
| 中心先验（**不看图像**，`−到画幅中心的格距离`）主体 AUC | **0.8358**，**高于 RO9c 全部 6 个读出**（raw L 0.784 / raw SC 0.749 / raw GC 0.736 / M1 GC 0.695 / M1 L 0.669 / M1 SC 0.475），配对 Δ −0.064…−0.320，p ≤ 4.2e-4 | `center_prior_baseline.overall` / `.paired_vs_center_prior` |
| 战役里**真正跑赢**中心先验的两条 | RO-1 ClearCLIP **0.961**（Δ+0.0721，p=2.0e-09）；RO-3 L11H5 区域对立**差分场** **0.927**（Δ+0.0885，p=3.1e-19）。**同一个 L11H5 头的单场只有 0.761/0.822（跑平/跑输）** ⇒ 跑赢的是「共模消除」不是「读出本身准」 | `center_prior_crossarm` |
| 四指标翻转检验 | 重叠类（AUC / soft-IoU / hard-IoU / point∈GT）中心先验 **4/4 第一**；边界类（bf1 3px / bf1_grid）中心先验 **4/4 垫底**；但 3px 那档的**随机 top-k 零模型 0.0394 > 中心先验 0.0327** ⇒ 该口径不可采信 | `four_field_multimetric` |

---

## 卡 2 —— G1 · s 可辨识性体检（Gate D1）

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | 冻结 VLM 里 `<retouch_light>` 打向 image token 的 **pre-softmax** 注意力场 s，**是不是一张跟着指令走的空间寻址图**（能不能零训练当 where 通路）。 |
| **模型** | 卡 0 底座，bf16，eager。env 自述：`VeraRetouchForCausalLLM_Unified (llava_qwen2, Qwen2-0.5B: 24L/14H/2KV/hidden896)`，视觉塔 `mobileclip_l_1024 (FastViTHD, 1024², patch64 → 16×16=256 image tokens)`。（`experiments/G1_s_identifiability_20260803/config/env.json → model`） |
| **输入** | style 模板 prompt（`data/infer_dataset.py:141`）；图 + 指令拼法同卡 0（image 在前、指令在后）。**带 greedy 生成**：`do_sample=false, num_beams=1, max_new_tokens=512`（`config/env.json → decode`）。**query 取生成出来的 GL token 位置**；若该轮没自然产出 GL token 则 fallback 到"末尾追加该 token 当 query"——**fallback 率 6.3%，全部来自 style 段，local 段 0.000**（`REPORT.md` §二）。 |
| **输出** | `s` = GL token query → 256 image token 的 **pre-softmax logit**，head-mean，**L8–15 平均**（canonical），D-0 修复（范数 MAD3 outlier 剔除 + 4 邻域插值），**无逐图归一化** ⇒ 每 (img, instruction) 一张 **16×16** 场。scache arm `ro9`（16×16 fp16，`/var/cache/veradata/scache/ro9/`）。 |
| **训练/推理** | **零训练**。读出总量 **1388 次**：主批 900（300 源 × 3 指令）+ 区域对立批 428（214 × 2）+ 跨图错位对照 60。（`REPORT.md` §设置） |
| **数据** | **S-val 源 300 张**（unsplash / awards / ppr10k 各 100，`tools/data_splits/splits.sqlite3` 旁表）；builds `prod-g1..g3-global25k` + `prod-l1..l4-local17k`。区域对立批 214 源。GT = veradata 银行 `cache/subject` 的 SAM3 主体软掩膜（300/300 命中），16×16 格覆盖 ≥0.5 判正，只在 valid（非黑边 pad）格上算，valid_frac 均值 0.674。 |
| **judge/commit** | seed 20260803；commit `0e5d04a`，branch `lens-exp`（`config/env.json`） |

| 判据 | 预注册（过 / 死） | 实测 | 判 | 出处 |
|---|---|---|---|---|
| ρ_syn 中位（同义稳定） | >0.7 / <0.5 | **0.867**（n=300，Spearman 0.763） | PASS | `metrics.json → aggregate`（REPORT §一） |
| \|ρ_Y\| 中位（非亮度马甲） | <0.5 / >0.8 | **0.177** | PASS（**只是"未触发死刑"**，D-31） | 同上 |
| **ρ_region_opp 中位**（同方向词、**不同区域**） | **< 0.3** | **0.884**（n=214；**n_below_0.3 = 0/214**；全批最小 0.585） | **FAIL** | `metrics.json → region_opposition` |
| **ρ_shuf**（把别的图的指令扣到本图） | 应显著 < ρ_syn | **0.895**（n=60；配对同源 ρ_syn=0.842；配对差 **−0.015**；ρ_syn 胜出率 **40%**，binomial p=0.155） | **FAIL（负控制被击穿）** | `metrics.json → shuffle_control` |
| AUC 配对（说主体 vs 说背景） | — | AUC(reg_a)=**0.648** vs AUC(reg_b)=**0.642**，配对差 **+0.0005**，Wilcoxon **p=0.121**；**AUC(b)>AUC(a) 占 48.6%（103/212）** | 佐证 FAIL | `metrics.json → subject_auc.paired_rega_vs_regb_light` |
| **Gate** | `gate_keys = [rho_syn, rho_region_opp, rho_y]` | **FAIL**（非 DEAD：两条死刑线都没踩） | — | `metrics.json → gate / verdict` |

**层扫描（说明"换层救不了"）**：24 层逐层 `sep = ρ_syn − ρ_region_opp` 最大只有 **+0.007**（L20），canonical L8–15 是 **−0.017**；10 个区间全部 ≤ +0.003。主体显著性口径下 AUC_light 在 **L22 最高 0.834**、L20–22 区间 0.825（vs canonical 0.648）。（`metrics.json → layer_scan / band_scan`）

---

## 卡 3 —— RO-9L（`RO9_layer_verdict_20260804`）· 层判决

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | RO-9 拿不到 where 是不是**层选错了**——早期层 ρ_region_opp 很低（L0=0.045 / L4=0.270），这到底是"真随指令变"还是"场不可复现"。 |
| **模型** | **不加载模型**。本实验**零前向**，直接读 G1 已落盘的逐层 s 场。 |
| **输入** | `experiments/G1_s_identifiability_20260803/run_{regfull,regsmoke20,full,smoke30,shufctrl}/stacks` 共 **1388 个 npz**。（`REPORT.md` §一） |
| **输出** | 逐层 24 行 + 逐案例 212 行的判据量表 + 3 张曲线图。 |
| **训练/推理** | **零训练、零 GPU**。纯 CPU numpy，wall-clock **107 s**，seed 20260804，空间置换零模型 32 次/场/层。 |
| **数据** | 区域对立批 214 源（S-val）→ **有效 212**（2 源主体掩膜 16×16 占比为 0）；分层 background **165**（主判据口径）/ spatial 47；主体面积16 中位 **0.138**。 |
| **commit** | `0e5d04a`，branch `lens-exp`，numpy 2.5.1 / scipy 1.17.1（`config/env.json`） |

| 判据 | 预注册（写死在 `analyze_layers.py:VERDICT_RULE`） | 实测 | 判 |
|---|---|---|---|
| ① 存在层使 **AUC_target ≥ 0.65** | ≥0.65 | **24/24 层不过线**；全层区间 **0.479–0.526**，最好层 L20=**0.526**，CI95 [0.390, 0.630]（含 0.5） | **FAIL** |
| ② 该层 ρ_region_opp < 0.30 | <0.30 | 只有 L0(0.045) 与 L4(0.270) 满足，但它们的 AUC_target = 0.487/0.495 | 条件满足但①不满足 |
| ③ 该层非白噪声 | — | L0/L4 均**非**白噪声（Moran's I 0.446/0.232 ≫ 零模型 −0.007；erank 5.99/7.92 < 零模型 8.98）→ 场是**行状条带**，空间有结构、语义无内容 | 条件满足但①不满足 |
| **裁决** | AUC 接近 0.5 或白噪声 ⇒ 判死 | **RO-9 判死（DEAD）** | — |

**必讲的一组对照数**：L22 上 `AUC(s_a,M) = 0.834` 但 `AUC(s_b,M) = 0.822` —— 指令从"点名主体"换成"点名背景"，s 场几乎不动。全 24 层 `max|ΔAUC 配对| = 0.0255`（L14）。（`metrics.json → layers`）

---

## 卡 4 —— RO-9b · RO-9 读出算子的四处修复

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | RO-9 的 AUC=0.648 **是读出算子的问题还是 VLM 里没有 where**——同一批 attention 张量换一套聚合方式能不能追平零训练 CLIP 的 0.941。 |
| **模型** | 卡 0 底座（bf16、eager）。**另加**：在自家视觉塔 **FastViTHD** 上原样施加 RO-1 的三个 self-self 算子（13 个臂）。视觉塔结构见卡 0；读出点 = `emb` logit lens。 |
| **输入** | 生成 plan（**批量生成 `gen-batch=8`**，见下方妥协登记）→ teacher-forced eager 前向 → 取 GL token（实测位于生成序列**倒数第 4 个 token**）对 256 个 image token 的 pre-softmax attention，**保留 head 维与全 24 层** ⇒ 每源一个 `(24, 14, 256)` 逐头栈；同时导出 post-softmax / k-k / q-q / `K_img`。（`REPORT.md` §一） |
| **输出** | `(24,14,16,16)` 逐头栈落盘于 `/var/cache/veradata/ro9b_stacks_20260803`（**642 条**）；最终场 `config/fields_final.npz`；scache arm `/var/cache/veradata/scache/ro9b-fixed`（**424 条 = 212 源 × {reg_a,reg_b}**，16×16 fp16，`norm.domain=[0,1]`，`lo=−1.6615/hi=0.8037` 为**整臂两个常量**，实测 >0 的格占 **99.6%**）。 |
| **训练/推理** | **主体零训练**，但**有折内选择/拟合**：5 折 **source-level** CV（seed 20260803），逐头/逐层 z-score 统计量、逐头权重、层段选择、交互与后处理档**全部只在 fit 折估、报 out-of-fold**。**受限档**（禁学习式层加权，层选择只允许 0–1 个自由参数）是主叙事；**全档**的 `linfit` 另学了 **24 个连续层权重**（⇒ **不再是零训练读出**，与 RO-1 的零标签 0.941 不同口径）。算力：卡 0，3 分片并行，各进程显存峰值 4.4 GB；LM 导出 642 作业 wall **25 min / 0 error**，组 A 13 臂 **13 min**，桥接臂 642 作业 0 error；分析纯 CPU（主 26 min + 受限 17 min）。 |
| **数据** | `G1/config/g1_region_opp.json` **214 源（S-val），零重新采样**；有效 **212**；background **165**（主判据口径）/ spatial 47。GT = SAM3 主体掩膜银行，16×16 覆盖 ≥0.5 判正，只取 valid 格。 |
| **commit** | `bc888fe147af76f2f8ef0299c85de856b9cd4adf`，branch `lens-exp`（`config/env.json`） |

| 判据 | 预注册 | 实测 | 判 |
|---|---|---|---|
| 晋级线 **AUC ≥ 0.85**（定位质量） | ≥0.85 | 受限档 **0.9200** [0.907,0.927]（零学习参数）／全档 **0.9349** [0.929,0.943]（24 个学习层权重） | **达成** |
| 指令依赖 `AUC_target` | 并报，缺一不判（UX-2） | 天花板 **0.5719** [0.524, 0.599]（**CI 下界刚越过 0.5**）；**定位质量最优的那条路径上反而 0.5042 / 0.3846** | **未救回** |
| `Δ_shuffle` | 每行必带 | 最好 **+0.0220 (p=5.7e-4)** = **RO-1 的 1/10**（RO-1 +0.220, p=5.5e-17）；定位最优路径上 **+0.0003 (p=0.813) / −0.0006 (p=0.365)** | **未救回** |
| 连续性锚（复刻 RO-9） | — | RO-9 已发表 0.648/0.504；本臂 S0 **0.6607 / 0.5083**；差 **+0.0145 AUC** 来自批量生成的工程妥协（25.5% 样本 plan 逐 token 相同 ⇒ 场**逐比特相同 ρ=1.000000**，其余 ρ 中位 0.927） | 已量化 |

**四处修复各值多少（`REPORT.md` §2d）**：① 层选择 **+0.179**（换最好单层 L22，1 个自由参数）／+0.274（学 24 个层权重）；② head 聚合 **+0.029**（逐层 top-3 头），**且是指令依赖上唯一有效的一处**（AUC_target +0.047）；③ self-self **+0.026**（LM 侧 q-q）；④ 后处理 **+0.026**（`aff` = K 自相似传播），D-0 本身只值 **+0.0036**。**③④ 在层权重学好后全部塌成 0** ⇒ 是代偿不是独立信息源。

**组 A（自家视觉塔 13 臂）必讲**：未改造 FastViTHD **AUC 0.7297 / AUC_target 0.6576**（vs 无关词 `calculator` 0.5767，配对差 +0.120，p<1e-3）；最好的手术只值 **+0.010**；ClearCLIP 原方把它打到 **0.4402**；只丢残差丢 FFN（`dropres`）打到 **0.3904**。机制：**FastViTHD 没有 cls token**（一手核实）⇒ 没有 SCLIP/NACLIP/ClearCLIP 要治的病。

---

## 卡 5 —— G1b / RO-D · DiffLMM（attend-and-segment）读出

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | G1 读不出指令依赖是不是**读法**问题——换成 DiffLMM 论文的 attend-and-segment 读出（**同一模型、同一批样本、同一套判据、前向逐比特不变**），同一份注意力里能不能读出随指令改变的空间场。**A&S 的 Eq.3 就是"解 sink"那一步。** |
| **模型** | 卡 0 底座，bf16，eager。**前向逐比特不变**（A&S 是纯读出：官方 `aas/infer_attn.py` 唯一模型调用是一次普通 `model.generate(..., output_attentions=True)`，归一化在读出侧）。 |
| **输入** | style 模板 prompt（74 token 实测）+ **带生成**；三个 retouch token 在**生成段末尾相邻**产出。`prompt 顺序 2×2` 因子：`image_first`（原样，img_span 恒 (14,270)）vs `instr_first`（把 `Instruction:` 移到 `<image>` 之前，img_span 变成 (63,319)/(72,328)）。 |
| **输出** | 一次前向同时导出 **pre-softmax（G1 口径）与 post-softmax + Eq.3 归一化（A&S 口径）**；**8 变体消融梯子 × 3 个 special token**。A&S 定义（按官方代码逐行核对）：取 image 列 → **对全部 24 层与全部 14 头求平均** → `A_norm = A_reduced − (1/r)Σ A_j`（沿输出 token 轴减均值）⇒ **16×16 场**。**落盘精度 float32**（post-softmax 概率 ~1e-3、减均值残差 ~1e-5，fp16 会把信号量化掉）。scache arm `/var/cache/veradata/scache/difflmm-light-L0-23/`，**822 条**，16×16 float16，`meta.norm.scale=1000.0`（`s_raw = s_stored/1000`），**存储值域 [−12.53, +9.34]，含负数** ⇒ 消费必须 `clamp=None`。 |
| **训练/推理** | **零训练**。**1182 次读出，0 次失败**：region 428（214×2）/ floor 214 / region_instr_first 240（120×2）/ floor_instr_first 120 / syn 120（60×2）/ shuf 60。卡 1。 |
| **数据** | G1 的 `g1_region_opp.json` / `g1_samples.json` / `g1_shuffle_ctrl.json` **原样复用，零重新采样**。GT 两套并报：`.cgt`（`config/cgt_masks16.npz`）与 SAM3 主体软掩膜（`config/subject_masks16.npz`，与 G1 同一 bank 同一 `luma_to_grid`）。 |
| **链路正确性硬核验** | `canon` 变体与 G1 落盘 npz **逐格对拍**：n=40，**Pearson 中位 0.99999992**，max_abs_diff 中位 0.0039（bf16 噪声）；`canon × image_first` 的 ρ_region_opp = **0.884**，与 G1 metrics.json **逐位一致**。（`metrics.crosscheck_vs_G1_canonical`） |
| **commit** | `bc888fe147af76f2f8ef0299c85de856b9cd4adf`（`config/g1b_config_report.json` 同批） |

| 判据 | 预注册 | 实测 | 判 |
|---|---|---|---|
| ρ_region_opp 中位 | **< 0.3**（沿用 G1 D-17） | `canon` **0.884**（0/214 达标）；**`aas` 0.439**（35/214 达标） | 表面上"大进步" |
| **⚑ 匹配噪声地板 `sep_matched = ρ_floor − ρ_region_opp`**（新增判别量：同区域只换句框，且表面形式扰动**更大**——`reg_a↔reg_a_para` 恒 15 词 vs `reg_a↔reg_b` 中位 12 词） | 应显著 > 0 | `canon` **−0.0009**（p=0.79）；**`aas` +0.0011（p=0.46，逐源胜出率 50.0%）** | **⇒ 换区域与只换句框对 s 的破坏完全相同；A&S 的 ρ 下降是噪声放大** |
| **AUC_target**（唯一单解度量） | 显著 > 0.5 | `canon` **0.4989**（p=0.80）；**`aas` 0.4957（p=0.77）**；三个 token、两套 GT、四个格全部 ≈0.5 | **否定** |
| 去共模差分场 AUC（配同区域对照） | 对立应 > 同区域 | `aas` **0.517** vs 同区域对照 **0.524**（对照更高）；`canon` 0.507 vs 0.525 | **否定** |
| ρ_shuf 负控制 | 应显著 < ρ_syn | `canon` 0.842 vs **0.895（反超，复现 G1）**；`aas` 0.360 vs 0.321（顺序正常但**两者都掉到噪声位**） | 不构成正面证据 |
| 场的形态 | — | **A&S 减完共模后单个格子吃掉 41.7% 的空间方差**（`canon` 0.083，均匀场参照 0.006）⇒ 低 ρ 来自**孤立尖刺**不是区域交换 | — |
| 与 RO-3 横比（**用 G1b 自己的掩膜与 AUC 代码重算**） | — | `canon` AUC_target[cgt] 0.499 / `aas` 0.496 vs **`ro3-l11-h5` 0.667 / `ro3-fused` 0.739**；配对 Δρ `aas − ro3-fused` = **+0.371**（p=9.0e-19） | **差距是数量级的** |
| prompt 顺序 2×2 | — | `canon` AUC_target 0.499→0.505；`aas` 0.496→0.503（**纹丝不动**） | 与 RO-2 的 C7 独立同结论 |

---

## 卡 6 —— RO-3 · LMM 全层全头 text→image attention 扫描（**P10 的来源**）

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | 在**不预设 token、不预设层**的前提下，冻结 VLM 的**某些单个 (层, 头)** 上、**instruction 文本 token → image token** 的注意力，是不是一张跟着指令改指区域的空间场 ⇒ G1/RO-9 的 FAIL 是**读法错**还是**假设错**。 |
| **模型** | 卡 0 底座，bf16，eager。**实测常量（从 config 与 tokenizer 直读，非转述）**：`24 层 / 14 heads / 2 KV heads（GQA groups=7）/ 256 image tokens = 16×16`（`config/verify_env.json`）。 |
| **输入** | **prefill-only，无生成**。图 + 指令拼法同卡 0；prompt 实测 87 token、展开后 **342** token，image span (14,270)，指令段 30 token（`verify_env.json → V7_*`）。attention 每层 `(1, 14, 342, 342)`。**4 个 query 池**：`instr`（instruction 段 text token 均值）/ `last`（prompt 末 token）/ `alltxt`（image 段之后全部 text token）/ **`gl`（桥接项：prompt 末尾追加 `<retouch_light>`，与 RO-9 并排读）**。 |
| **输出** | **2688 个候选读出** = 2 读法（`pre` = q·kᵀ/√d + causal mask；`post` = **整行**含 sink softmax 后取 image 列）× 4 query 池 × 24 层 × 14 头，每个是一张 **16×16** 场。scache arms：`ro3-l11-h5`（772 条，值域 [1.79e-7, 0.0850]）、`ro3-fused`（772 条，值域 **[−3.256, 8.805]**，58.8% 的格 <0 ⇒ 消费必须走 `_ARM_INFO.json` 的 `consume_recipe`）；另有 construct 版各 1600 条。 |
| **训练/推理** | **主体零训练**。**1708 次 eager prefill**，0.22 s/样本，**卡 1**，显存峰值 **1.46 GB**（限额 20 GB）。融合（可学凸组合）= `softmax(θ)` 严格单纯形，目标 = AUC 的成对 logistic 松弛；**选头/定向/全局 z-score 统计量全部只在训练折上估，报 out-of-fold**（5 折 source-level CV）。 |
| **数据** | **零重新采样**：G1 冻结的 `g1_{samples,region_opp,shuffle_ctrl}.json`（S-val **300 / 214 / 60**）+ D-CONSTRUCT L1（S-val **22** 评测 / S-train **181** 仅供跨 split 拟合）。GT 三列并报：`.cgt.png`（主列，n=81 background×normal）/ D-CONSTRUCT 自带 GT（n=22）/ SAM3（对照列，n=165）。 |
| **commit** | 见 `metrics.json → git_commit`（`REPORT.md` 抬头，branch `lens-exp`，2026-08-03） |

| 判据 | 预注册（写死在 `analyze_ro3.PREREG` / `analyze_diffield.PREREG_DIFF`） | 实测 | 判 |
|---|---|---|---|
| ① 晋级：融合 AUC_target ≥ **0.75** | ≥0.75 | **0.8302**（out-of-fold，5 折，k=32）CI95 [0.816, 0.841] | **PASS** |
| ② 晋级：最佳层在**中层**（L8–15） | L8–15 | 融合 top-weight 层 = **L11**；加权平均层 11.51；最佳单读出层 = **L11** | **PASS** |
| ③ 淘汰：最佳层落末两层 | L22–23 | top-50 的层分布：中层 38 / 深层 7 / 早层 5 / **末两层 0** | **未触发** |
| ④ AUC_target（G1 口径） | 必报 | 最佳单读出 `post/instr` **L11H5 = 0.7660**；max-stat 置换零分布中位 0.558 / **q95 0.596** → **FWER p = 0.000** | 达标 |
| ⑥ shuffle 前置判别 | 必做 | 单读出 **Δ_shuffle = +0.0733**；融合 k=32 **+0.1079**；**ρ_syn 0.833 > ρ_shuf 0.651**（G1 是 0.842 < 0.895，**方向翻转**） | **PASS** |
| ⑨ 差分场 AUC ≥ 0.65 且 FWER p<0.05 ⇒ (b) 读法错确证 | ≥0.65, p<0.05 | **L11H5 `pre/instr`：0.9298（SAM3）/ 0.8405（`.cgt` 主列）/ 0.9217（构造 GT）**，FWER p = **0.000** | **确证** |
| ⑩ 同区域零对比度对照 | 必做 | 同一读出的 `d_ctrl = s(syn_a) − s(syn_b)` AUC = **0.5220**；配对 Wilcoxon **p = 5.1e-35**（n=212） | **PASS** |

### ⚑ P10 用的那组数（**任务卡的 0.58→0.93 需要修正**）

从 `metrics_diffield.json` 按其 `index_note`（`r = ((mp*24+layer)*14+head)`，`MP = [(pre,instr),(pre,last),(pre,alltxt),(pre,gl),(post,instr),(post,last),(post,alltxt),(post,gl)]`，出处 `analyze_ro3.py:37-39`）直接取值：

| 口径 | `gl` 池（special token 当 query） | `instr` 池（指令文本 token 当 query） |
|---|---|---|
| **同一层同一头 L11H5，`pre`** | **0.4991**（FWER p = **1.0**） | **0.9298**（FWER p = **0.000**） |
| 同一层同一头 L11H5，`post` | **0.4678**（p = 1.0） | **0.8987**（p = 0.000） |
| 全 336 个 (层,头) 取 max，`pre` | **0.5839**，argmax = **L8H9**（**不是 L11H5**），min FWER p = **0.982** | 0.9298，argmax = L11H5 |
| 全 336 取 max，`post` | 0.5982，argmax = L11H8，p = 0.9575 | 0.8987，argmax = L11H5 |
| max **AUC_target** | **0.5542**（**低于零分布 q95 = 0.596**） | **0.7660** |
| head-mean 通道（= RO-9 落盘口径，`ro9_gl_attention.py:378` 落盘前就 `raw.mean(axis=2)`） | 区域对立 **0.5129**，同区域对照 **0.5293**（**对照更高**），spatial 子类 0.4929 | — |

⇒ **"同一层同一个头"的正确说法是 0.499 → 0.930**（比 0.58→0.93 更强）；**0.5839 是"允许 gl 池在全场 336 个头里挑最好的一个"的上限**。二者不可混着说。（本次实测复算，脚本一行 numpy 索引即可复现）

### ⚑ U3 补件（`REPORT.md` §十二，`metrics_genpos.json`）—— prefill 追加位 vs 生成位

**协议**：G1 区域对立批**前 100 源** × {reg_a, reg_b}（background 子集 72），**带 greedy 生成**（中位生成长度 325.5 token，GL fallback 率 0.000），脚本 `probe_genpos.py`，wall 51 min。

| 池 | 追加位 → 生成位（AUC_target） | 追加位 → 生成位（差分场 AUC） | 两位空间 ρ 中位 | argmax 漂移 |
|---|---|---|---|---|
| `pre/instr` | 0.7317 → 0.7318（**Δ +0.0001**） | 0.9315 → 0.9306 | **0.999936** | L11H05（两位都是） |
| `post/instr` | 0.7541 → 0.7549（Δ +0.0008） | 0.8938 → 0.8935 | 0.999869 | L11H05 |
| **`pre/gl`** | 0.5743 → **0.5908**（Δ +0.0165） | 0.5931 → **0.7010**（Δ +0.1078） | **0.607** | **L06H05 → L19H07** |
| **`post/gl`** | 0.5743 → 0.5871 | 0.6041 → 0.6418 | **0.361** | L06H05 → L19H07 |

**读法**：`instr` 池两位**实测等价**（因果掩码下 instruction token 的注意力行不受其后 token 影响）；**`gl` 池两位不是同一张图**（ρ 0.36–0.61），生成位略强但 **0.5908 < 零分布 q95 0.596，仍在噪声地板内**。⇒ 位置值 ≤0.017，**query token 值 0.16–0.24**。与 RO-9b 的生成位读数（AUC_target 0.5719）相差 0.019，**三家口径已对齐**。

---

## 卡 7 —— RO-1 · self-self 零训练 CLIP 读出（基准线）

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | **零训练的 CLIP self-self 稠密读出**能不能给出 AUC≥0.80 且真正跟着目标名词走的空间场 ⇒ 它是所有 VLM 读出臂必须超过的**免费基准线**。 |
| **模型** | **OpenAI CLIP ViT-B/16**，`/home/bc/data/models/openai_clip/ViT-B-16.pt`，sha256 `5806e77c…f416f`（与官方 URL 内嵌期望哈希一致），**fp32，冻结**。**参数量（本次从 checkpoint state_dict 逐张量实测）：总计 149,620,740**；其中**视觉塔 86,192,640**、文本塔 63,428,100。**视觉塔 12 个 resblock，宽度 768，12 头**（`vision_heads = vision_width // 64`，`tools/readout/clip_naclip/model.py:283`）；**文本塔 12 层，宽度 512，8 头**（`transformer_heads = transformer_width // 64`，同文件:416）；词表 49408。 |
| **输入** | 图：**短边 448、保长宽比、裁到 patch 整数倍、整图前向**（voc21 的 `slide_crop=0` 档）⇒ **patch 原生网格 ~28×42**（另跑 336 全量作稳健性行）。文本：目标短语走 **80 条 `openai_imagenet_template` 集成**（逐模板 encode→L2→均值→再 L2，NACLIP `naclip.py:30-42` 逐字口径）。**图与指令不拼在一起**——这是双塔，各自编码后取余弦。**无 teacher forcing**（不是自回归模型）。 |
| **输出** | `s = cos(patch_feat, text_emb)`，`patch_feat` = `encode_image(return_all=True)` 丢 CLS 后 L2 归一 ⇒ **每 (图, 文本) 一张 ~28×42 的余弦场**，**全程无逐图归一化**。scache arm `/var/cache/veradata/scache/ro1-clearclip-l11`（**237 条**，32×32 fp16；`norm` 为**整臂两个常量** μ=0.24908 / σ=0.06722，`per_image=false`）。 |
| **训练/推理** | **零训练、零标签**。三算子只改**末层 attention**：`sclip` = arch vanilla + attn `csa`｜`naclip` = arch reduced + attn `naclip`(std=5)｜`clearclip` = arch reduced + attn `clearclip`（qqᵀ，丢残差丢 FFN）。**仅卡 1**，显存峰值 **1.72 GB**（限额 20 GB），wall-clock **78 min**（其中全分辨率 guided filter 块 64 min）。 |
| **数据** | **S-val 237 样本** = `construct_l1`（D-CONSTRUCT L1(S-val) **23**，T4 冻结 sanity 批，1/24 因 journal 无主体描述剔除）+ `subject_sam3`（G1 区域对立批 **214** 源 × veradata 银行 SAM3 主体软掩膜）。**复用 `g1_region_opp.json`，未重新采样**。目标短语主口径 = `local.subject.description`（长描述），`subject.name`（短名词）并列同报。标签口径：掩膜面积均值下采到评分网格，**≥0.5 判正**（G1 同口径）。 |
| **commit / seed** | `bc888fe147af76f2f8ef0299c85de856b9cd4adf`；seed 20260803（`config/env.json`） |

| 判据 | 预注册（写死在 `run_ro1.py` 顶部 `PREREG`） | 实测 | 判 |
|---|---|---|---|
| Gate：**两个评分集 median AUC 均 ≥0.80** | 晋级 ≥0.80；淘汰 <0.75 | D-CONSTRUCT L1 (n=23) **0.939** [0.832, 0.966]；SAM3 (n=214) **0.930** [0.901, 0.945] | **PROMOTE** |
| 符号检查（**禁事后翻转**） | 中位 AUC>0.5 且逐样本 AUC>0.5 占比 ≥0.90 | 三算子全 PASS；**`vanilla`（未改造 CLIP）FAIL**：AUC **0.214**，96.7% 的样本 AUC<0.5（**系统性反相关**） | 按纪律**不翻转** |
| **Δ_shuffle**（跨图错位名词，配对） | 每行必带 | L1 集 **+0.314，p=2.6e-5**，胜率 0.826；SAM3 集 **+0.220，p=5.5e-17**，胜率 0.729（错位后 AUC 掉到 0.516 / 0.653） | **正面证据**（**这正是 RO-9 缺的那一列**） |
| AUC_target（G1 口径） | — | background 子集 **0.736–0.755**；spatial 子集 0.689–0.789（对照：RO-9 全 24 层 0.479–0.526） | — |
| Δ_luma / Δ_const | 每行必带 | +0.47 / +0.41；Δ_const 恒 = median AUC − 0.5 | 未触发死刑 |
| **⚠ 必须与 0.930 一同引用的限定（C9）** | — | RO-X1 实测：**一条对所有图相同的 `"the main subject"` 拿到 0.907**，与 0.930 **统计上不可区分**（配对差 +0.009，p=0.133）；同条件 **AUC_target 塌到 0.523** | **RO-1 的强项是"找主体"不是"听指令"** |

**必讲的一句**：「**改末层 attention 这一步值 +0.70 AUC，怎么改只值 0.02**」——vanilla 0.214 → 三算子 0.90–0.93（+0.69~+0.72），而三算子彼此差 0.014–0.028、CI 重叠、换短名词口径后排名翻转（NACLIP 0.962 vs ClearCLIP 0.961）。

**逐样本仍有 ~18% 落在淘汰线下**（41/214 < 0.75，31/214 < 0.70），且失败时是**整体翻转**（AUC 0.08–0.27）不是温和退化。

---

## 卡 8 —— RO-X1 · CLIP 侧半场（有名词 / 无名词分离）

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | 把指令里的可定位名词拿掉，CLIP 的区域可控性会不会塌 ⇒ **VLM 在 where 这条线上到底可不可替代**。 |
| **模型** | 与卡 7 **完全相同的一份权重与配置，不重扫**：OpenAI CLIP ViT-B/16（149,620,740 参数；视觉塔 86.19M / 12 层 / 宽 768 / 12 头）· `clearclip` · `+A3+A1+A2` · 短边 448。（`config/env.json → operator, rung, side, weights`） |
| **输入** | 同卡 7（双塔，图文各自编码）。11 个文本条件：`N_desc`（逐源长描述）/ `N_name`（逐源短名词）/ `N_shuf`（别的图的描述，负控制）/ **5 条 `X_*` 对全体样本相同的固定短语**（`X_deictic` = `"the main subject"`、`X_deictic_comp` = `"everything except the main subject"`、`X_intent`、`X_style`、`X_mood`、`X_tonal`）。固定短语与逐源名词短语走 80 模板集成；**真实整句指令不套模板**（`truncate=True` 并报截断率）。 |
| **输出** | 同卡 7：`~28×42` 余弦场（另报 16×16 列）。**场未落盘**（RO9c §10.6 登记：口径 A 无法配对，只能用口径 C）。 |
| **训练/推理** | **零训练**。**仅卡 1**，显存峰值 **1.01 GB**，wall-clock **9 分 14 秒**（323 源 × 11 个文本条件）。 |
| **数据** | Arm B（决定性）：**同 214 张 S-val 图、同一张 SAM3 主体掩膜、同一个算子，只换文本**。Arm A（数据侧）：按 journal `task_type` 把 300 个 S-val 源切成 local(214) / style(86)。 |
| **commit / seed** | `bc888fe147af76f2f8ef0299c85de856b9cd4adf`；seed 20260803 |

| 判据 | 预注册（写死在 `run_rox1.py` 顶部 `PREREG`/`NOUNFREE`） | 实测 | 判 |
|---|---|---|---|
| ① 主判据 配对 ΔAUC = median[AUC(N_desc) − AUC(X_deictic)] | 显著分离 ⟺ ≥0.05 且 p<0.01 | **+0.009，p=0.133，胜率 0.528**（n=214） | **不分离** |
| ② 负结论线 | 无分离（重大负面结论）⟺ 中位差 < 0.02 | +0.009 < 0.02 | **触发** |
| ③ 辅判据 AUC_target(X_deictic vs 其补集) | ≤0.60 ⇒ 无名词指称下区域可控性丧失 | **0.523**，CI95 [0.396, 0.649] | **触发** |
| ④ Arm A 作废条件 | 两个名词检测器在两半集上无显著差异 ⇒ 半集标签作废 | style 半集 **D1 命中率 0.884 / D2 命中率 0.942**，**严格无名词子集仅 2/86 源** | **作废条件触发**（Arm A 降为描述性） |

**关键数字表（Arm B，n=214，S-val，SAM3 GT）**

| 文本条件 | 有名词? | 逐图不同? | median AUC | 16×16 |
|---|---|---|---|---|
| `N_name` 逐源短名词 | ✔ | ✔ | **0.961** | 0.976 |
| `N_desc` 逐源长描述（主协议） | ✔ | ✔ | **0.930** | 0.941 |
| **`X_deictic` = "the main subject"** | ✘ | **✘（214 张图同一条）** | **0.907** | 0.926 |
| `X_deictic_comp` = "everything except the main subject" | ✘ | ✘ | **0.887**（**要求"除主体以外的一切"，场仍压在主体上**） | 0.915 |
| `X_intent` / `X_style` / `X_mood` / `X_tonal` | ✘ | ✘ | 0.773 / 0.738 / 0.735 / 0.605 | — |
| `N_shuf` 别的图的描述（负控制） | ✔ | ✔ | **0.653** | 0.656 |

**Arm A 描述性数字**：`ρ_shuf` local **0.073** → style **0.536**；`sep = ρ_syn − ρ_shuf` local **0.864** → style **0.361**（跨半集 Mann-Whitney **p=3.8e-6**）；`Δ_shuffle` local **+0.049 (p=4.7e-4)** → style **−0.023 (p=0.885)**（**与 G1 里 RO-9 的签名 +0.0005 / p=0.121 几乎逐字相同**）。

---

## 卡 9 —— MCQ-L 条件消融（端到端，**唯一一张"训练了"的卡**）

| 字段 | 内容 |
|---|---|
| **实验 / 一句话在验什么** | 端到端训练的 MetaCanvas 端点，**给指令 / 给中性固定句 / 给错指令 / 只给指令不给图**四档因果对照下，像素质量与"掩膜 AUC"分别掉多少 ⇒ 分开"instruction-conditioned color"与"图像主体先验"。 |
| **模型** | **冻结 VeraRetouch VLM（卡 0 底座，bf16，eager）+ LoRA r32 + MetaCanvas 连接器 + 双读出头**。<br>· LoRA：`r=32, alpha=64, dropout=0.05, bias=none`，target = `q/k/v/o_proj, gate/up/down_proj`（`MCQ_e2e_whatwhere_20260803/canvas_model.py:202-206`）<br>· 加载后 **`del base.retouch_decoder` / `del base.retouch_head`**（生产头与本实验无关，`canvas_model.py:196-197`）<br>· **MetaCanvas query：256 个可学 query（16×16），维度 = HIDDEN 896**，加 `Fourier2D(HIDDEN)` 位置特征（`canvas_model.py:22-23, 218-221`）<br>· **读 L11 / L17 / L23 三个 tap，路由权重 `route` 3 维可学**（`--taps 11,17,23`）<br>· **connector `PatchCanvasConnector`：in 896 → dim 384，1 层 `TransformerEncoderLayer`，nhead=8，FFN = 4×384 = 1536，dropout 0.1，`norm_first=True`，残差经 zero-init `nn.Linear(384,384)`**（`canvas_model.py:77-100`）<br>· **spatial head `DenseSpatialHead`：LayerNorm → Conv2d(384,128,3) → GELU → Conv2d(128,128,3) → GELU → Conv2d(128,1,1)（zero-init）⇒ 16×16 mask logits**（`local_model.py:113-130`）<br>· **parameter head `GaussianParameterReadout(metaquery)`：`QueryDecoder(49 query, dim=384, 3 层 TransformerDecoderLayer, nhead=8, FFN 1536)` → 前 48 个 query 各出 `raw_dim=27`（gaussian4d）、第 49 个出 12 维 global affine，两个投影都是 zero-init ⇒ `ParamHead4D(48, anchors)`**（`local_model.py:48-90`） |
| **参数量** | **`counts.total = 650,669,771` / `counts.trainable = 28,250,091`（可训练占比 4.34%）**（`runs/config_a/run.json → counts`，五个 condition arm 逐位相同）。与卡 0 的 checkpoint 634,075,663 对账：634,075,663 − 11,620,227（删掉的 retouch 头）= 622,455,436；622,455,436 + 28,250,091 = 650,705,527，**与 650,669,771 差 35,756（0.006%）未追平**，疑为 checkpoint 里的非 Parameter buffer —— **如实登记**。 |
| **输入** | · VLM 侧：`build_prompt(instruction)`（`canvas_model.py:40-51`，qwen_2 模板 + `TASK_STYLE_RETOUCH_TOKEN` + `Instruction: {…}`），图走模型自带 image processor（`expand2square` → 256 image token）<br>· 渲染/监督侧：源图 EXIF transpose → LANCZOS 短边 1024 → **`source_rgb` / `target_rgb` / `mask` 全部 resize 到 128×128**（`data.py:14 RENDER_SIZE = 128`）<br>· **模型输入不含 `I_tar` 或 `.cgt`**（`metrics.json → target_not_in_condition = true`）；`.cgt` 只作训练标签与评测真值<br>· **无 teacher forcing**（不做自回归解码，只取 hidden state 做 cross-attention 读出）<br>· 训练时 **instruction condition dropout p=0.15**（`run.json → config.condition_dropout`） |
| **输出** | · `mask_logits`：**(B, 16, 16)** → `sigmoid(bilinear 上采到 128×128)` = 监督用 mask 概率（`train.py:184-196`）<br>· renderer 参数：48 个 4D anchored Gaussian primitive（各 27 维 raw）+ 12 维 global affine → `render4d` → 预测图 **(B,128,128,3)** |
| **训练** | **6000 step，batch 8**（`run.json → config.steps / batch`）；lr **2e-4**（main）/ **2e-5**（LoRA），warmup **200**，wd **0.05**（config_a = 胜出的 lr 尺度，四个 condition arm 全部沿用）；每样本 **1024 个分层像素**；`eval_every=500`、`checkpoint_every=500`、`eval_limit=384`（select-core 只用于 checkpoint 选择）；**checkpoint 不按 val loss 选**（红线），依据 ΔE00 p50/p90、mask 内/边界/外 PSNR、outside leakage、`delta_const`、`delta_shuffle`、AUC/AUC-shuffle，并对低于 3 dB 的 const/shuffle 施加 gate penalty，**IoU 只作描述统计**（`train.py:255-268` selection 公式落在 `metrics.json → final.select.selection_formula`）。<br>· **优化器：`torch.optim.AdamW`，`betas=(0.9, 0.95)`（eps 用默认），两个 param group —— 非 LoRA 参数 lr=2e-4 / wd=0.05，LoRA 参数 lr=2e-5 / wd=0.0**（`train.py:519-527`）<br>· **scheduler：`LambdaLR`，前 200 step 线性 warmup，之后余弦退火到 1% 地板 —— `0.01 + 0.99*0.5*(1+cos(pi*t))`**（`train.py:529-535`）<br>· seed **20260804**（像素采样另用 `seed+29` 的独立 generator，`train.py:536`）。<br>wall-clock：config_a **4392.6 s**；image_only 4785.7 s；instruction_only 4868.9 s；fixed_shuffle 4395.2 s；short 4395.5 s（`runs/*/metrics.json → wall_sec`）。选中 step：config_a/fixed_shuffle/short **5000**，image_only/instruction_only **4500**（`→ selected_step`）。 |
| **数据** | 六个 durable SFT build `prod-l1..l6-local17k-*`，只保留 `winner_confidence == normal` ∧ `task_type == local` ∧ 有 instruction/source/target/cgt。split 权威 = `tools/data_splits/splits.sqlite3`；sweep 用 S-train 内按 `source_id` 冻结的 `fit/select`（**两个 pool source-disjoint**）。**训练 37,370 条（`fit` pool），评测 4,225 条（完整 `select` pool）**（`runs/*/metrics.json → data`）。manifest = `/mnt/nfs/bc/data/datasets/derived/metacanvas-local-l1l6-v2-20260804`，digest `3ad571a9…9ddd8485`（五个 arm 逐位相同 ⇒ 唯一变量确实只有 condition）。 |
| **判据（预注册，`EXPERIMENT_SCHEDULE.md`「Instruction 到颜色的判据」）** | 「**完整输入只有同时优于 image-only 与 fixed wrong instruction，并在固定图像交换 instruction 时改变 WHAT，才能称为 instruction-conditioned color；普通 mask AUC 高但同图反事实不变，只能解释为图像主体/位置先验**」。**image-only / instruction-only / fixed wrong instruction 是因果控制，即使像素指标高也不进入最终候选。** |

### 四档实测（全部取 `runs/<arm>/metrics.json → final.select`，n=4225）

| arm（`condition_mode`） | 喂进去的东西 | ΔE00 p50 ↓ | ΔE00 p90 ↓ | PSNR_in p50 ↑ | **AUC_cgt p50** | soft-IoU p50 | **Δ_shuffle (dB)** | Δ_const (dB) | Δ_AUC_shuffle |
|---|---|---|---|---|---|---|---|---|---|
| **`config_a`（`full`）** | 图 + 真实指令 | **3.560** | 7.086 | **22.59** | 0.9469 | **0.605** | **1.901** | 0.780 | +0.0024 |
| `condition_short`（`short`） | 图 + `instruction_short` | 3.545 | 7.074 | 22.53 | 0.9461 | 0.584 | 1.787 | 0.829 | +0.0018 |
| **`condition_image_only`** | 图 + **中性固定句** `"Retouch the requested region of this image."`（`train.py:115`） | **3.995** | 8.310 | **20.29** | **0.9481** | 0.509 | **0.000** | 0.244 | **0.000** |
| **`condition_fixed_shuffle`** | 图 + **别的样本的指令**（确定性错位，`data.py:116`） | **4.011** | 8.266 | **20.41** | **0.9498** | 0.508 | **−0.000** | 0.221 | **−1.616** |
| `condition_instruction_only` | **全零图** + 真实指令（`train.py:132-133`） | 3.811 | 7.467 | 22.03 | **0.8110** | 0.448 | 1.407 | 0.455 | +0.0074 |

**指标定义（`train.py:290-348`）**：`Δ_const` = `PSNR(全图, 用把 s 场压成常数后重渲的图)` 与真渲的差；`Δ_shuffle` = 与「batch 内循环移位指令重跑一遍」的 PSNR 差；`Δ_AUC_shuffle` = `auc_cgt − auc_cgt(错位指令)`；`auc` 与 `soft_iou` 都在 128×128 的 mask 概率上算。

**⚑ 这张卡最该被记住的一行**：**掩膜 AUC 三档几乎相同（0.947 / 0.948 / 0.950），"不给指令"甚至最高**；分开四档的是 ΔE00（3.56 → 4.00）、PSNR_in（22.6 → 20.3）与 Δ_shuffle（1.90 dB → 0）。⇒ **mask AUC 在这条端到端线上同样是"漂亮数字骗人"的那一类。**

### 本卡的「缺」

| # | 缺什么 | 影响 | 怎么补 |
|---|---|---|---|
| 1 | **优化器超参只在源码里、未落进 `run.json`/`metrics.json`**（本卡已从 `train.py:519-535` 追到并写全，但机器可读产物里没有） | 低（已可追源码） | 下次跑把 optimizer/betas/scheduler 写进 `run.json → config` |
| 2 | **MCQ 目录没有 `REPORT.md`**（只有 `NOTES.md` / `EXPERIMENT_SCHEDULE.md` / `AB_VISUALIZATION.md` / `RENDERER_GAUSSIAN3D_ALPHA_VISUALIZATION.md`），**因此没有"预注册数字 vs 实测数字并排表"** | 条件消融的判据只在 `EXPERIMENT_SCHEDULE.md` 里以散文形式写着，没有阈值化的数字判据 | 若要进汇报，建议补一张并排表；或明确说明"这一波是结构矩阵、判据是排序不是阈值" |
| 3 | **同图反事实（instruction swap）尚未做** | `EXPERIMENT_SCHEDULE.md` 自己写了「现有 batch 内循环 shuffle **不是**严格同图反事实」，因此 Δ_shuffle=1.90 dB 只能读作"换指令会变"，**不能读作"按指令变对了"** | 按 schedule 的「终局纪律」补 paired evaluation（固定 `source_id` 做 instruction swap / 方向相反 / synonym） |
| 4 | 条件消融四个 arm **没有 20 样本并排可视化** | P9 若要样例图需现跑 | `visualize_final.py --run-dir runs/<arm>`，单卡数分钟/arm |

---

## 汇总：全部「缺」项一览（共 6 处）

| # | 卡 | 缺什么 | 严重度 | 补法 |
|---|---|---|---|---|
| 1 | 卡 9 MCQ | **无 `REPORT.md`、无「预注册数字 vs 实测数字」并排表** —— 条件消融的判据只在 `EXPERIMENT_SCHEDULE.md` 里以散文形式写着，没有阈值化数字 | **中（汇报里最可能被追问的一处）** | 补一张并排表，或明确说明这一波是**排序**不是阈值 |
| 2 | 卡 9 MCQ | **同图反事实（instruction swap）未做** —— `EXPERIMENT_SCHEDULE.md` 自己承认「现有 batch 内循环 shuffle 不是严格同图反事实」 | **中** ⇒ Δ_shuffle=1.90 dB 只能读作"换指令会变"，**不能读作"按指令变对了"** | 按 schedule 终局纪律补 paired evaluation（固定 `source_id` 做 instruction swap / 方向相反 / synonym） |
| 3 | 卡 4 RO-9b | **「最佳主体定位档」0.920 / 0.935 的场未落盘**，无法与中心先验配对（RO9c §10.6）；`config/fields_final.npz` 只有 `auc_target` 选优档（AUC 0.728） | **中（直接影响 P4/P11 的可比性）** | 逐头栈仍在 `/var/cache/veradata/ro9b_stacks_20260803`，**零 GPU 可重跑**；重跑必须标注该档层权重是**在目标 GT 上有监督拟合**的 |
| 4 | 卡 9 MCQ | 五个 condition arm 的 `counts` 与 checkpoint 逐张量对账**残差 35,756（0.006%）未追平** | 低 | 逐参数名比对一次即可；不影响任何结论 |
| 5 | 卡 9 MCQ | **优化器超参只在源码里**（本卡已从 `train.py:519-535` 追全），机器可读产物里没有；condition arm 也没有 20 样本并排图 | 低 | 下次跑写进 `run.json → config`；图用 `visualize_final.py`，单卡数分钟/arm |
| 6 | 卡 8 RO-X1 / 卡 7 RO-1 | **RO-X1 的 9 条短语场完全未落盘**；RO-1 只有 `clearclip × +A3+A1+A2 × desc` 一档写进 scache | 低 | 已用口径 C（原生网格 + 落盘逐源 AUC）覆盖，见 RO9c §10.3 |

> 此外有一处**不是"缺"而是"任务卡数字需修正"**：卡 6 RO-3 的 P10 对照，「同一层同一个头」的正确数字是 **0.4991 → 0.9298**（L11H5, `pre`），不是 0.58 → 0.93；**0.5839 是 `gl` 池在全 336 头上取 max 的值，argmax 在 L8H9**。两者不可混着说。
