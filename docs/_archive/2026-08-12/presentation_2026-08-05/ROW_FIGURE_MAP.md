# 单行配图映射表（旧多行联图 → 新单行图）

生成日期：2026-08-05　　产出目录：`docs/presentation_2026-08-05/assets_row/`
重绘脚本：`docs/presentation_2026-08-05/figs/`（`rowlib.py` + `make_row_*.py`）
逐图机器可读事实：`figs/row_facts_*.json`

## 版式约定（按评审两轮要求收紧后的最终形态）

- **一页一图，一张图最多 1 行**；每格短边 ≥ 440 px。
- **画面里只有图像内容 + 一行简短列标题**：不放图注、不放 source id、不放格内数值、
  不叠加真值轮廓线。真值单独占一列。
- 所有被移出图面的信息（选源规则 / source id / 逐格指标 / 口径）在本表逐行给出，
  供报告正文与 PPT 备注引用。
- 全程零 GPU：脚本档只读已落盘的 stacks / `features20.npz` / 推理缓存，裁剪档逐像素搬运原图。
  未训练、未重跑推理、未改动任何 `experiments/` 下的产物。

## 检查结论一览

| 项 | 数 |
|---|---|
| 报告 `BIWEEKLY_REPORT.md` 引用的 `assets/*.png` | 34 |
| 逐张开图判定为「超过 1 行」的 | 14 |
| 判定保持原样的 | 20（其中 1 张为 2 行热力图矩阵，属纯图表，见文末） |
| 产出的单行图 | 38（含未被报告引用但任务点名的 `P3d` 3 张、`P5a` 2 张） |

---

## 一、需要替换的图（按报告出现顺序）

### 图 2(b) · 色标消融

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P3c_seven_col_ablation_GC.png`（5 行 × 7 列） |
| 新路径（推荐顺序） | 1. `assets_row/P3c_seven_col_ablation_GC_row2.png`<br>2. `assets_row/P3c_seven_col_ablation_GC_row3.png`<br>3. `assets_row/P3c_seven_col_ablation_GC_row1.png` |
| 读图要点 | 七列只改一个自变量：第 4 列**只把色标统计范围换成有效格**，主体轮廓立刻显现，而场本身一字未改。 |

- 选源规则（effect-blind）：直接沿用 RO-9c 已冻结的 `config/figure_picks.json`——分层取样，
  只用 G1 配置里的源属性（pool / region_b_kind / subject_area / winner_confidence），
  每层按 `img_id` 字典序取第一个；过滤 `winner_confidence != low`、`subject_area ∈ [0.08, 0.35]`。
  单行版按该清单顺序取前 3 个源，不看任何读出结果。
- 逐行事实（token GC `<retouch_color&temp>`，prompt=auto，模型冻结、零训练）：

| 新图 | source | 主体面积 | 补边格占 image 注意力质量 | argmax 在补边格 | raw AUC | 共模消除 AUC |
|---|---|---|---|---|---|---|
| `_row1` | `src_08aef2f575be97b1` | 0.263 | 55.7% | 是 | **0.480** | 0.374 |
| `_row2` | `src_0e6178111d466cfb` | 0.253 | 51.6% | 是 | 0.735 | 0.809 |
| `_row3` | `ppr10k_0579_a` | 0.318 | 55.9% | 是 | 0.704 | 0.733 |

- **推荐顺序为什么不是清单顺序**：`_row1` 的源是一张 2.4:1 的拼接图，单行版整体宽高比达 12:1，
  投影时偏扁；且该源 raw AUC 仅 0.480，是规则内的弱例。**它按规则照登、不删**，
  但放在第 3 页；`_row2` 现象最典型，放第 1 页。这是**排版顺序**的选择，不是选源，已在此声明。
- 口径：AUC 只在非补边（valid）格上算、用未归一化原始场；16×16 场叠回原图走
  `make_figs.grid_to_img` 的严格逆映射，未直接 resize。
  ※ 本页 AUC 仅用于复现 6-09 的历史论证；新实验判据已改为 IoU + grid 边界 F1 + 中心先验列。

### （附）图 2(b)-L · 同一消融的光照 token 版

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P3d_seven_col_ablation_L.png`（报告当前未引用，任务点名要求处理） |
| 新路径（推荐顺序） | 1. `assets_row/P3d_seven_col_ablation_L_row1.png`<br>2. `assets_row/P3d_seven_col_ablation_L_row3.png`<br>3. `assets_row/P3d_seven_col_ablation_L_row2.png` |
| 读图要点 | 换成 `<retouch_light>` token，同一条「只改色标」的结论照样成立（补边格吃掉 69–74% 的注意力质量）。 |

逐行：`_row1` src_08aef2f575be97b1 pad 73.7% / raw AUC 0.695；`_row2` src_0e6178111d466cfb pad 69.2% / 0.471；`_row3` ppr10k_0579_a pad 69.5% / 0.627。选源规则同上。

### 图 6 · 三档条件输入的真实渲染

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P53a_condition_render_compare.png`（6 行 × 8 列，且每格挂 Δ 图与三行数字） |
| 新路径（推荐顺序） | 1. `assets_row/P53a_condition_render_compare_row1.png`（L2，8 列渲染 / 掩膜）<br>2. `assets_row/P53a_condition_render_compare_row2.png`（L2，5 列 Δ 图）<br>3. `assets_row/P53a_condition_render_compare_row3.png`（L6 反例，5 列 Δ 图） |
| 读图要点 | row1 看第 6–8 列：真值掩膜与两档预测掩膜位置几乎重合 → 指令没改「改哪」；row2 看 Δ 图：只有真实指令那一档有明显色块 → 改的是「改多狠」；row3 是方向相反的反例。 |

- 拆法说明：原图把 Δ 图挂在每格下方，压成单行会变成两层。故改为**渲染档与 Δ 档各占一页**，
  两页都是严格单行。Δ 图是图像内容（= 0.5 + 3.3×(输出−输入)，全图共用同一常量增益、
  无逐图归一化；中性灰 = 完全没改，|Δ| ≥ 0.15 处饱和），不是标注。
- 选源规则：源来自 config_a 于 2026-08-04 冻结的 20 例可视化清单
  （`LocalDataset(pool="test", limit=20)` → stratified_limit，按 uid 的 sha256 排序、层内先保证 source 不重复）。
  单行版取 L2 层首例；再取 L6 —— 它是 6 行里**唯一**一行「打乱指令的 Δ 幅度反而更大」，
  属报告 §5.3 已声明的反例，规则是反例必须照登、不做替换。
- 逐行事实：

| 新图 | level / uid | soft-IoU（真实 / 打乱 / 中性） | ΔE00 | Δ 幅度比 |
|---|---|---|---|---|
| `_row1`,`_row2` | L2 · `…candidate_8adc51eed600186b2dead889d7e74131` | 0.734 / 0.718 / 0.708 | 3.23 / 4.80 / 4.84 | **1.005** / 0.155 / 0.065 |
| `_row3` | L6 · `…candidate_2d188afc325c1d3ce20fabe63006287c` | 0.731 / 0.741 / 0.744 | 3.77 / 4.16 / 4.11 | 0.065 / **0.466** / 0.293 ⚠反例 |

- 整池数字（`runs/<arm>/metrics.json → .final.select`，n=4225，三档同一批样本）：
  真实指令 soft-IoU 0.605 / PSNR 22.59 dB / ΔE00 3.560 / 方差比 0.317；
  打乱 0.508 / 20.41 / 4.011 / 0.067；中性句 0.509 / 20.29 / 3.995 / 0.063。
  20 例上三档预测掩膜逐图相关中位 **0.963**；真实指令的 Δ 幅度比在 **17/20** 例为三档最高。
- 三档语义（`train.py:113-135 condition_inputs`）：真实指令 = `condition_mode="full"`；
  别的样本的指令 = `"fixed_shuffle"`（`data.py:109-121` 确定性错位，跨 source 且跨 preset）；
  全体相同中性句 = `"image_only"`，逐字为 “Retouch the requested region of this image.”

### 图 9 · 只换 query token 的逐案例

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P11b_query_position_cases.png`（5 行 × 4 列） |
| 新路径（推荐顺序） | 1. `assets_row/P11b_query_position_cases_row1.png`<br>2. `assets_row/P11b_query_position_cases_row3.png`<br>3. `assets_row/P11b_query_position_cases_row2.png` |
| 读图要点 | 同一层同一个头（L11 H5，pre-softmax），只把 query 从 special token 换成指令文本 token，区域对立差分场就从「几乎没结构」变成「贴着主体」。 |

- 选源规则同图 2(b)（RO-9c 冻结清单，按顺序取前 3 个源）。
- 逐行 AUC（special token 当 query → 指令文本 token 当 query）：
  `_row1` src_08aef2f575be97b1（awards，面积 0.263）0.189 → **0.927**；
  `_row2` src_0e6178111d466cfb（unsplash，0.253）0.657 → 0.972；
  `_row3` ppr10k_0579_a（ppr10k，0.318）0.574 → 0.959。
- 版式说明：新图**去掉了原图叠在热力图上的青色 GT 轮廓**，真值改为最后一列独立呈现；
  `expand2square` 的补边格已裁出视野（补边格不是数据）。
- 正确性自检：同一实现在全部 214 源上复算 background 子集中位 = gl 0.4991 / instr 0.9298，
  与 `RO3_layerhead_scan_20260803/metrics_diffield.json` 逐位一致
  （`figs/make_P10_query_position.py --verify` 可复核）。
  ※ 该 AUC 属已落盘的历史口径，结论限于「同一头内换 query 位置的排序」。

### 图 11 · dense 头的边缘退化

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P13a_dense_edge_degradation.png`（5 行 × 7 列） |
| 新路径（推荐顺序） | 1. `assets_row/P13a_dense_edge_degradation_row3.png`<br>2. `assets_row/P13a_dense_edge_degradation_row1.png`<br>3. `assets_row/P13a_dense_edge_degradation_row2.png` |
| 读图要点 | 第 5 列真值区域是清晰剪影，第 6/7 列 A/B 两配置的预测都是一团模糊竖斑——16×16 mask logits 双线性上采到 128×128，边缘分辨率被 query 数锁死。 |

- 选源规则：按原图行序取前 3 行（原图共 5 行，来自 config_a 的冻结 20 例清单，
  `compare_ab_visualizations.py` 生成，未按效果挑）。
- 逐行 uid（均为 L1）：`_row1` `…candidate_35bdf6a553fb…`；`_row2` `…candidate_200922a31ac6…`；
  `_row3` `…candidate_442b4263a7df…`（真值是人物剪影，边缘退化最直观，故排第一）。
- 分辨率说明：每格原生 128×128（模型渲染分辨率），放大到 440 不引入新信息。

### 图 12 · MetaQuery 读出不随指令改变

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P13b_metaquery_not_instruction_conditioned.png`（6 行 × 5 列） |
| 新路径（推荐顺序） | 1. `assets_row/P13b_metaquery_not_instruction_conditioned_row1.png`<br>2. `assets_row/P13b_metaquery_not_instruction_conditioned_row2.png`<br>3. `assets_row/P13b_metaquery_not_instruction_conditioned_row3.png` |
| 读图要点 | row1 与 row2 是**同一张图**的两条相反指令（目标区域一个是圆斑、一个是斜条），第 4 列的预测 mask 几乎一模一样——切指令时预测不动。 |

- 选源规则：按原图行序取前 3 行（原图 = source-disjoint val 的 L6 UID 等距抽样，每个 UID 同时展示 rega/regb，
  `visualize_best.py` 生成，未按逐样本 AUC 挑图）。row1/row2 必须成对使用。
- 逐行（普通 AUC / AUC_target）：`_row1` L6_val_0000 区域 A：0.958 / 0.569；
  `_row2` L6_val_0000 区域 B：0.941 / 0.737；`_row3` L6_val_0011 区域 A：0.843 / 0.508。
  checkpoint = MetaQuery / w14_metaquery@2500。

### 图 13 · 基函数长什么样

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P14a_basis_functions_14.png`（4×4 图鉴） |
| 新路径（推荐顺序） | 1. `assets_row/P14a_basis_functions_14_row1.png`（几何基 8 维）<br>2. `assets_row/P14a_basis_functions_14_row2.png`（VLM 语义基 e1..e6） |
| 读图要点 | row1 的 8 个几何基是干净的解析图案与图像亮度/饱和度；row2 的 6 个 VLM 语义基明显是块状噪点——「语义基很脏」这件事一眼可见。 |

- 拆法：原图是 14 基的 4×4 图鉴（本质是字典而非样本联图），按**基的定义顺序**拆成
  「几何 8 + 语义 6」两张单行，每张 ≤ 8 列。样本 = 冻结清单第 1 个 uid
  `…candidate_35bdf6a553fb…`（e1..e6 与图像相关，故必须绑定具体样本）。
- 重绘自 `experiments/MCQ_basis_where_l1l6_20260804/viz/basis_vlm14_full/features20.npz`，
  灰度拉伸口径与 `visualize_final.basis_montage` 逐字一致（2/98 分位）；每格 16×16 原生分辨率，
  最近邻放大，不插值、不造边界。

### 图 14 · 系数怎么组合成场

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P14b_basis_weighted_contrib.png`（4 行 × 8 列） |
| 新路径（推荐顺序） | 1. `assets_row/P14b_basis_weighted_contrib_row1.png`<br>2. `assets_row/P14b_basis_weighted_contrib_row2.png` |
| 读图要点 | 8 列 = `wᵢ·φᵢ` 的加权贡献图，同一行共享色标；模型实际只把权重压在 2–3 个低阶基上（`y`、`P₂(x)`、`P₂(y)`），其余接近全白。 |

- 选源规则：按冻结清单顺序取前 2 个样本（共 20 个）。
- 逐行系数（原图印在列标题里的那组数，现移到本表）：
  `_row1`（`…35bdf6a553fb…`）1 −0.229 / x −0.011 / y **+0.734** / P2(x) −0.252 / P2(y) −0.359 / xy +0.002 / L +0.003 / S −0.021，共享色标 ±0.597；
  `_row2`（`…200922a31ac6…`）1 −0.114 / x −0.145 / y +0.082 / P2(x) **−0.727** / P2(y) −0.235 / xy +0.007 / L −0.012 / S −0.000，共享色标 ±0.359。

### 图 15 · 环形掩膜：单调 vs 带通

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P15a_ring_monotone_vs_bandpass.png`（上散点 + 下样例行的复合版式） |
| 新路径（推荐顺序） | 1. `assets_row/P15a_ring_monotone_vs_bandpass_row2.png`（4 列样例）<br>2. `assets_row/P15a_ring_monotone_vs_bandpass_row1.png`（散点，纯图表） |
| 读图要点 | 样例页：单调读出把薄环画成一个实心圆盘（第 3 列），带通读出画得出薄环（第 2 列）——s 是坐标轴不是 alpha。散点页给全体分布：单调 med 0.34 / 带通 med 0.98。 |

- 按任务要求把复合版式**上下拆成两张独立图**，不强行压成一行。散点页原样搬运（未改内容、未加文字）。
- 样例行原样拆出，未挑图；4 列 = 目标环形掩膜 / 带通拟合 IoU=0.98 / 单调拟合 IoU=0.28 / 拟合出的 s 场。

### 图 16 · 受约束 s 轴

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P15b_constrained_axis.png`（上散点 + 下样例行） |
| 新路径（推荐顺序） | 1. `assets_row/P15b_constrained_axis_row2.png`（5 列样例）<br>2. `assets_row/P15b_constrained_axis_row1.png`（散点，纯图表） |
| 读图要点 | 样例页：受约束的 M=12 s 轴（第 2 列）合成出的平顶环与自由带通读出（第 3 列）几乎无差；散点页给出 0.975 vs 0.975、geometry-only 侧 0.990 反超 0.977。 |

- 同上，上下拆分；5 列 = 目标环形掩膜 / 受约束 s 轴 IoU=0.975 / 带通读出 0.974 / 单高斯 0.791 / 单调读出 0.479。

### 图 17 · 语义族成功案例

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P15c_semantic_fit.png`（3 行 × 3 列） |
| 新路径（推荐顺序） | 1. `assets_row/P15c_semantic_fit_row1.png`<br>2. `assets_row/P15c_semantic_fit_row2.png` |
| 读图要点 | 14 维基底的单调读出能把真实语义掩膜（海雕 / 海豹）拟合到 IoU 0.97——基底本身不是瓶颈。 |

- 选源规则：按原图行序取前 2 行（原图共 3 行；第 3 行是同一张海雕图的带通档，与第 1 行重复，故未取）。
- 逐行：`_row1` `semantic_0099`，单调拟合 IoU=**0.973**；`_row2` `semantic_0082`，单调拟合 IoU=**0.969**。

### 图 18 · 语义族失败案例

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P15d_semantic_failure.png`（3 行 × 3 列） |
| 新路径（推荐顺序） | 1. `assets_row/P15d_semantic_failure_row1.png`<br>2. `assets_row/P15d_semantic_failure_row2.png` |
| 读图要点 | 目标是多个互不相连的小实例 + 细碎边界时，低维基底拟合不出来——这是 limitation 的主证据。 |

- 选源规则：按原图行序取前 2 行（共 3 行）。
- 逐行：`_row1` `semantic_0115`（美术馆里的多组人像），单调拟合 IoU=**0.457**；
  `_row2` `semantic_0069`（逆光花枝），单调拟合 IoU=**0.559**。

### 图 19(a)(b) · geo8 / vlm14 的逐样本细节

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P64a_geo8_detail.png`、`assets/P64b_vlm14_detail.png`（各 5 行 × 8 列） |
| 新路径（推荐顺序） | geo8：1. `assets_row/P64a_geo8_detail_row1.png`　2. `assets_row/P64a_geo8_detail_row2.png`<br>vlm14：1. `assets_row/P64b_vlm14_detail_row1.png`　2. `assets_row/P64b_vlm14_detail_row2.png` |
| 读图要点 | 第 2/3 列真值 vs 预测 mask 的形状差、第 4 列 mask 差，加上最后两列「主导基 × 系数」——模型把区域压在 1–2 个低阶几何基上，所以边界只能是平滑的椭圆状。geo8 与 vlm14 同一样本几乎给出同一张图（soft-IoU 0.544 vs 0.543），说明加 6 维语义基没带来增益。 |

- 选源规则：按冻结清单顺序取前 2 个样本（共 20 个）。
- 逐行：geo8 `_row1` uid `…35bdf6a553fb…` soft-IoU 0.544，主导基 `y w=+0.734`、`P2(y) w=−0.359`；
  `_row2` uid `…200922a31ac6…` soft-IoU 0.505，主导基 `P2(x) w=−0.727`、`P2(y) w=−0.235`。
  vlm14 `_row1` 同 uid soft-IoU 0.543，主导基 `y w=+0.727`、`P2(y) w=−0.301`；
  `_row2` soft-IoU 0.485，主导基 `P2(x) w=−0.490`、`P2(y) w=−0.200`。
- 重绘自各自的 `features20.npz`（gt_mask / pred_mask / latent_s / renderer_s / spatial_basis / spatial_coeff
  均为落盘原生数据），着色沿用 `visualize_final.mask_image / signed_image`；
  第 1 列「输入」自同目录 `gallery20_p01.png` 的 128 px 原图裁出。

### 图 A2（附录）· E1b 低秩重建失败

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/APX_e1b_lowrank_failure.png`（3 行 × 3 列） |
| 新路径（推荐顺序） | 1. `assets_row/APX_e1b_lowrank_failure_row1.png`<br>2. `assets_row/APX_e1b_lowrank_failure_row2.png` |
| 读图要点 | 秩 48 重建把原 LUT 的高频条纹抹平，ΔE00 图上残差沿条纹密集分布——低秩不足以复现这批 preset。 |

- 选源规则：按原图行序取前 2 行（共 3 行）。
- 逐行：`_row1` `quandian__quandian_003473`，r=48 平均 ΔE00 = **9.76**；
  `_row2` `quandian__quandian_003325`，平均 ΔE00 = **9.48**。
- 第 3 列保留了原图自带的那一根 ΔE00 色标条。

### （附）P5a · 四个场并排（报告当前未引用，任务点名要求处理）

| 项 | 内容 |
|---|---|
| 旧路径 | `assets/P5a_four_fields.png`（多行 × 6 列，且每格叠了 top-k 实线 + GT 虚线 + 指标文字） |
| 新路径（推荐顺序） | 1. `assets_row/P5a_four_fields_row1.png`<br>2. `assets_row/P5a_four_fields_row2.png` |
| 读图要点 | 中心先验场（第 2 列，完全不看图像）是一团居中圆斑，AUC 却不低；真正贴着主体的是第 3–5 列的注意力场——AUC 判不出「能不能当 mask 用」。 |

- 版式说明：原图把 top-k 预测实线与 GT 虚线叠在每个场上，按评审要求**全部去掉**；真值单独占最后一列。
  着色规则逐字沿用 `diag_four_fields.field_tile`（只取有效格的 min-max、不做 relu，四个场同一条规则）。
- 选源规则（effect-blind）：`config/four_field_picks.json` —— 先取补件 A 的 5 个源，
  再对四个 subject_area 档补齐到每档 ≥2，补齐时只用 `winner_confidence != low` + `img_id` 字典序。
- 逐行（AUC / soft-IoU / 边界 F1）：

| 新图 | source | 主体面积 | 中心先验 | raw L | raw GC | 共模消除 GC |
|---|---|---|---|---|---|---|
| `_row1` | `ppr10k_2619_a` | 0.035 | 0.630 / **0.000** / 0.000 | 0.961 / 0.212 / 0.081 | 0.949 / 0.319 / 0.133 | 0.833 / 0.269 / 0.103 |
| `_row2` | `ppr10k_2067_a` | 0.036 | 0.814 / **0.004** / 0.020 | 0.950 / 0.019 / 0.051 | 0.986 / 0.292 / 0.149 | 0.981 / 0.292 / 0.149 |

---

## 二、判断应当保持原样的图（不动）

| 报告图号 | 路径 | 理由 |
|---|---|---|
| 图 2(a) | `assets/P5b_sink_diagnostic.png` | 1×3 纯图表（直方图 + 两张配对散点） |
| 图 3 | `assets/P4a_crossarm_forest.png` | 森林图 |
| 图 4 | `assets/P8a_G1_two_opposite_instructions.png` | 已是单行 6 列 |
| 图 5(a)(b)(c) | `assets/P52a_…png`、`P52b_…png`、`P52c_…png` | 已是单行 7–8 列 |
| 图 7 | `assets/P10a_condition_ablation.png` | 1×3 柱状图 |
| 图 8 | `assets/P11a_query_position.png` | 单幅柱状图 |
| **图 10** | `assets/P11c_layerhead_heatmap.png` | **2 行 × 4 列，但每格是 layer×head 热力图矩阵，属纯图表而非样本联图**；拆行会破坏 pre/post 两档的并排对照，故按「纯图表不动」处理。若评审仍要求一页一行，建议按 pre 档 / post 档拆成两张 1×4，**需先确认**——本次未擅自裁剪。 |
| 图 11 之外 | `assets/P11d_diffield_null.png` | 1×4 纯图表（热力图 + 直方图） |
| — | `assets/P12a_fixed_phrase_trap.png` | 已是单行 5 列 |
| 图 20 | `assets/P16b_coefficient_heatmap.png` | 系数热图矩阵，本质是表格，任务明确要求不动 |
| 图 22 / 23 | `assets/P18a_3D_insufficient.png`、`P18b_3D_sufficient.png` | 已是单行 6 列 |
| 图 24 | `assets/P18c_delta_ceil_distribution.png` | 分布图 |
| 图 25 / 26 | `assets/P20a_rdg_success.png`、`P20b_rdg_failure.png` | 已是单行 6 列 |
| 图 28 / A1 | `assets/APX_e1b_rank_curve.png`、`APX_e1b_spectrum.png` | 秩曲线 / 谱曲线 |
| 图 A3 / A4 / A5 | `assets/APX_a0_*.png` | 各为 1×3 纯图表 |

---

## 三、复现方式

```bash
cd /home/bc/VeraRetouch/docs/presentation_2026-08-05/figs
P=/home/bc/miniconda3/bin/python
$P make_row_ro9c_colormap.py --rows 3   # 图 2(b) / 2(b)-L      零 GPU
$P make_row_condition_render.py         # 图 6                  零 GPU（读推理缓存）
$P make_row_query_position.py --rows 3  # 图 9                  零 GPU
$P make_row_four_fields.py --rows 2     # P5a                   零 GPU
$P make_row_crops.py                    # 其余裁剪档 / npz 重绘  零 GPU
```

`make_P_condition_render_compare.py --refresh` 才会重跑三臂推理；本次**没有**跑，
全部图来自 `figs/_cache_P_condition_render_compare.npz` 等已落盘产物。
`experiments/` 目录全程只读。
