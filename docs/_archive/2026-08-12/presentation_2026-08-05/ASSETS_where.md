# 汇报配图素材清单 · Where 线（精准 mask 预测）

生成日期：2026-08-05
适用页码：P13–P17
状态标记：**[有]** = 文件已用 `ls` 核实存在；**★** = 我已开图逐张核对过内容；**[生成]** = 需跑脚本；**[自绘]** = 无现成素材，需画。

叙事主线（供排版参考）：
P13 直接预测像素崩掉 → P14 insight：掩膜可由一组基表示 → P15 E2 证明基的上限 → P16 但模型读不出好系数 → P17 改进方案。

---

## P13 · 直接预测像素崩掉的证据

### 已有素材

| # | 绝对路径 | 内容（已核对的写实描述） | 状态 |
|---|---|---|---|
| 1 | `/home/bc/VeraRetouch/experiments/MCQ_full_local_l1l6_20260804/viz/config_ab_compare/compare20_p01.png` | 5 行样本 × 7 列：`输入｜目标｜A 最终渲染｜B 最终渲染｜GT 区域｜A 预测区域｜B 预测区域`。第 3 行 GT 是清晰的人物剪影，预测是一团竖直模糊斑块；第 4 行 GT 是斜向分割，预测几乎是一张平灰图。**这是"16×16 dense 头 → 双线性上采到 128×128"导致的边缘退化的最直接一张。** | ★ **[有]** |
| 2 | 同目录 `compare20_p02.png` / `compare20_p03.png` / `compare20_p04.png` | 同版式，第 6–20 条样本（L1–L6 分层） | **[有]**（未逐张开图） |
| 3 | 同目录 `compare20_overview.png` | 20 条样本缩略总览 | **[有]** |
| 4 | `/home/bc/VeraRetouch/experiments/MCQ_e2e_whatwhere_20260803/viz/best_where_w14_metaquery.png` | 3 个 L6 UID × rega/regb 共 6 行 × 5 列：`输入(带指令原文)｜目标区域 mask｜同图另一候选区｜预测 mask(标普通 AUC 与 AUC_target)｜叠加图`。预测是一大团 viridis 色斑，**同一张图切 A/B 指令时几乎不动**；标题上写着 `普通 AUC=0.958，AUC_target=0.569`。这张同时能当 P13（读不出精准区域）和 P16 的引子用。 | ★ **[有]** |
| 5 | 同目录 `gallery20_where_w14_metaquery_overview.png` + `gallery20_where_w14_metaquery_p01..p04.png` | 20 张源图 × A/B 两条指令 = 40 次读出的稳定采样联图（青=GT，红=预测） | **[有]**（5 个文件，未逐张开图） |

### 需生成

| # | 产出 | 脚本 + 已落盘数据 | 成本 |
|---|---|---|---|
| G1 | `spatial_metaquery8`（8 个全局 query 直接解码 16×16 logits）的 20 样本 where 联图 | `/home/bc/VeraRetouch/experiments/MCQ_full_local_l1l6_20260804/visualize_final.py --run-dir runs/spatial_metaquery8 --out-dir viz/spatial_metaquery8`；checkpoint `runs/spatial_metaquery8/best.pt`（120 MB，已存在，step≈2500） | 单卡约 5 分钟（含 VLM 加载） |
| G2 | 坍缩曲线：`var_ratio` 从 **0.00087@step500** 爬到 0.117@step2500，同期 AUC 只有 0.80–0.82 | 直接从 `runs/spatial_metaquery8/metrics_partial.json` 的 `.history[].eval` 取 `var_ratio / auc_cgt_p50 / soft_iou_p50`，matplotlib 折线 | CPU < 1 分钟 |
| G3 | 三档并排柱图：dense(config_a) 0.947 / basis_geo8 0.915 / metaquery8 0.820 的 AUC 与 soft-IoU | 三份 `metrics.json`（`.final.select.auc_cgt_p50`、`.soft_iou_p50`；metaquery8 取 `metrics_partial.json` 最后一次 eval） | CPU < 1 分钟 |

> **必须在这一页说清的口径**：dense 头在数字上并不比 basis 差（soft-IoU 0.605 vs 0.528，AUC 0.947 vs 0.915）。它被停掉的理由是**结构性**的，不是分数：mask logits 只有 16×16，边缘分辨率被 query 数锁死（`train.py:184-196` 双线性上采到 128×128），换成 `metaquery8` 后立刻出现近乎坍缩（var_ratio 8.7e-4）。权威原话见 `runs/../matrix_state.json` 的 `pause_reason`（P13 讲稿可直接引用，见 MODELCARDS 卡 2）。

---

## P14 · 「预测基而不是预测像素」的示意图

### 需自绘（主图）

| 元素 | 要画什么 | 数据依据 |
|---|---|---|
| 左侧：基函数长什么样 | 一排 8–14 个小方图，依次是常数 `1`、`x`、`y`、`P₂(x)`、`P₂(y)`、`x·y`、`L`（亮度）、`S`（饱和度），可再接 `e₁..e₆`（语义通道）。前 6 个是解析几何图案（渐变/二次条纹），`L/S` 是图像本身的亮度/饱和度图，`e₁..e₆` 是图像相关的语义响应图。 | 真实版本见下方"可直接抠图替代"栏；定义在 `experiments/E2_basis_fit_20260803/e2lib.py:4-11` 与 `docs/METACANVAS_..._PROTOCOL_2026-08-04.md:167-177` |
| 中间：系数 | 一列 14 个数字（或颜色条），标注"模型唯一要预测的东西 = 这 14 个标量" | `MCQ_basis_where_l1l6_20260804/viz/compare/coefficients_compare20.png` 有真值 |
| 右侧：组合成场 | 公式三行：`q(p) = Σᵢ wᵢ·φᵢ(p)` → `s(p) = 3·tanh(q/3)` → `m(p) = R(s(p))`；配一张最终 mask | 公式逐字在 `MCQ_basis_where_l1l6_20260804/EXPERIMENT_PROTOCOL.md:26-31`，实现在 `MCQ_full_local_l1l6_20260804/local_model.py:254-258` |
| 底部一行小字 | "从 H×W 个自由度 → 14 个自由度；边缘不再由 query 数决定" | — |

### 可直接抠图替代自绘（省时间，且是真实数据）

| # | 绝对路径 | 内容 | 状态 |
|---|---|---|---|
| 6 | `/home/bc/VeraRetouch/experiments/MCQ_basis_where_l1l6_20260804/viz/basis_vlm14_full/feature_spatial_basis.png` | **14 张基函数图**（16×16 分辨率），标题逐张写着 `00 1 / 01 x / 02 y / 03 P2(x) / 04 P2(y) / 05 xy / 06 L / 07 S / 08 e1 … 13 e6`。前 6 张是干净的解析图案，`L/S` 是图像结构，`e1..e6` 明显是块状噪点——**这一张同时把"基长什么样"和"VLM 语义基很脏"两件事都说了**。 | ★ **[有]** |
| 7 | `/home/bc/VeraRetouch/experiments/MCQ_basis_where_l1l6_20260804/viz/basis_geo_range8_full/where_basis_contrib20_p01.png`（共 p01–p05） | **系数怎么组合成场**：每行一个样本，8 列 = `基ᵢ × wᵢ` 的加权贡献图，列标题写着 `P2(x) w=-0.73` 这样的实测系数，同一样本内共享色标。 | ★ **[有]** |
| 8 | `/home/bc/VeraRetouch/experiments/MCQ_basis_where_l1l6_20260804/viz/basis_geo_range8_full/feature_spatial_basis.png` | Geo8 的 8 张基函数图（无 e₁..e₆） | **[有]** |

> 建议：P14 用 **自绘的干净示意图**（8 个基 + 系数条 + 组合箭头）做主视觉，把 #6/#7 作为"这不是画出来的，是真跑出来的"缩略角标放右下。

---

## P15 · E2 的基底上限

### 已有素材（全部在 `/home/bc/VeraRetouch/experiments/E2_basis_fit_20260803/viz/`）

| # | 文件名 | 内容 | 状态 |
|---|---|---|---|
| 9 | **`ring_evidence.png`** | **本页第一主图。** 上半：4 组散点带中位线 —— `monotone(14维) med 0.34`、`bandpass(14维) med 0.98`、`monotone(geo-only) med 0.29`、`bandpass(geo-only) med 0.98`，画着两条门线（`monotone gate <=0.40` 虚线、`bandpass gate >=0.90` 点线）。下半：4 张图 `target ring｜bandpass fit IoU=0.98｜monotone fit IoU=0.28｜fitted s field`——单调读出把薄环画成一个实心圆盘。**这就是"s 是坐标轴不是 alpha"的那张关键证据图。** | ★ **[有]** |
| 10 | **`constrained_axis_evidence.png`** | **补件的那张，务必和 #9 一起放。** 6 种读出并排（monotone / single Gaussian / constrained single prim. / constrained no-norm 0.963 / **CONSTRAINED M=12 axis 0.975** / unconstrained flat-top 0.975），左右两块分别是 14 维基底与 geometry-only 控制（右块约束档 **0.990** 反超自由读出 0.977）。下半 5 张同一个环的五种渲染。**这张回答的是"0.975 是不是你挑了个自由读出凑的"——不是，渲染器自己那根受约束的 s 轴就能合成那个平顶。** | ★ **[有]** |
| 11 | `success_ring.png` | 3 行 × 3 列 `图像｜target mask｜fit`。前两行是 monotone 拟合（IoU 0.596 / 0.583，画成实心圆盘），第 3 行是 bandpass（IoU 0.985，画出薄环）。**薄环那一例。** | ★ **[有]** |
| 12 | `success_semantic.png` | 同版式 3×3。第 1 行 monotone IoU=0.973（海雕轮廓）、第 2 行 monotone IoU=0.969（海豹）、第 3 行 bandpass IoU=0.973。**语义那一例。** | ★ **[有]** |
| 13 | `success_linear.png` | 同版式，线性渐变族 | **[有]**（未开图） |
| 14 | `success_radial_ell.png` | 同版式，径向/椭圆族 | **[有]**（未开图） |
| 15 | `failure_ring.png` | 单调读出的预期内失败：薄环被画成覆盖圆盘或其补集 | **[有]** |
| 16 | `failure_gauss_ring.png` | 表达力失败：幅值恒 1 的单高斯把薄环画成一圈渐晕（无平顶），IoU 0.79 | **[有]** |
| 17 | `failure_semantic.png` | 语义族尾部失败（p10 0.736 / min 0.457）：多个互不相连的小实例 + 细碎边界。REPORT §7 建议把这张当论文的 limitation 主证据。 | **[有]** |
| 18 | `ablation_dims.png` | 维度消融（lin3 / geo6 / main14 / cubic18），对应 REPORT §6 的表 | **[有]** |
| 19 | `constrained_axis_response.png` | 响应级：拟合出的响应曲线 vs 理想平顶 + 12 张带通掩膜 + σ_max 扫描 | **[有]** |
| 20 | `beta_reach_vs_M.png` | wave 3：β 可达区间随 M 变化（M=6 有 44.5% 的环落在等不透明度可达集之外） | **[有]** |
| 21 | `constrained_axis_sweep.png` | A-4 预算对照（mask 级扫描面板已标注未跑） | **[有]** |

### 排版建议

- 主位：**#9（0.34 vs 0.98）**；副位：#11 薄环 + #12 语义 + #13 线性 + #14 径向 四张各取一例，拼成"四族上限"一行。
- **#10 必须出现在同一页或紧邻页**，否则审稿人的第二问（"平顶是你挑的读出凑的"）没有答案。这一点在 E2 的 REPORT.md L5/L9 里写得很死。
- 注意口径：图上标的是 `med 0.34 / med 0.98`（四舍五入），报告与 `metrics.json` 里的精确值是 **0.34227 / 0.97528**（`metrics.json → .criteria.ring_mono_max040.measured_median` / `.criteria.ring_band_min090_CORE.measured_median`）。讲稿写 0.342 / 0.975，别照图念 0.98。

---

## P16 · Geo8 vs VLM14 并排对比 + 系数热图

### 已有素材（全部在 `/home/bc/VeraRetouch/experiments/MCQ_basis_where_l1l6_20260804/viz/`）

**先纠正一个前提**：`viz/compare/` 下**不是 20 张并排图，而是 3 个文件，每个文件内部含 20 条样本**。核实结果：

| # | 绝对路径 | 内容（已核对） | 状态 |
|---|---|---|---|
| 22 | **`viz/compare/where_compare20.png`**（755 KB） | **本页主图。** 左半 = Geo8，右半 = VLM14；20 条 L1–L6 分层样本，每条 4 列：`输入｜GT mask｜预测 mask｜latent s`（蓝-白-红发散色图）。每格标题写着 `01 L1 IoU=0.54` 这样的逐样本 IoU。**肉眼可读的结论：Geo8 全是低频椭圆/斜带；VLM14 出现了人形、纹理等图像相关细节但很噪；两边 GT 是清晰剪影时都画不出来。** | ★ **[有]** |
| 23 | **`viz/compare/coefficients_compare20.png`**（160 KB） | **本页的系数热图，正是"证明 Geo8 全压在低频几何项上"那张。** 左表 8 列（`1, x, y, P2(x), P2(y), xy, L, S`，共享色标 ±0.827），右表 14 列（多 `e1..e6`，共享色标 ±0.560）。逐格印着数值：Geo8 的大值集中在 `1 / y / P2(x) / P2(y)`（如 `P2(x) -0.84`、`y +0.75`、`1 -0.85`），而 **`L` 与 `S` 全列都在 ±0.06 以内**；VLM14 的 `e1..e6` 也全在 ±0.12 以内。 | ★ **[有]** |
| 24 | `viz/compare/render_compare20.png`（974 KB） | 20 条样本的最终渲染并排：`输入｜目标｜预测`，Geo8 与 VLM14 左右分块 | **[有]** |
| 25 | `viz/basis_geo_range8_full/where_coefficients20.png` | Geo8 单臂的 8 维系数热图（若嫌 #23 太宽可用这张） | **[有]** |
| 26 | `viz/basis_vlm14_full/where_coefficients20.png` | VLM14 单臂的 14 维系数热图 | **[有]** |
| 27 | `viz/basis_{geo_range8,vlm14}_full/where20_overview.png` + `where20_p01..p04.png` | 单臂 20 张 where 总览与高分辨率分页（各 5 个文件，共 10 个） | **[有]** |
| 28 | `viz/basis_{geo_range8,vlm14}_full/where_basis_contrib20_p01..p05.png` | 每条样本的逐基加权贡献图（各 5 页，共 10 个）——**如果 mentor 追问"到底哪几个基在起作用"，翻这个** | ★ **[有]**（Geo8 p01 已核对） |
| 29 | `viz/basis_{geo_range8,vlm14}_full/feature_spatial_basis.png` | 8 / 14 张基函数本体 | ★ **[有]** |

### 需生成（可选）

| # | 产出 | 方法 | 成本 |
|---|---|---|---|
| G4 | 全量 4,225 条 select 上的**系数幅值统计**（每个基的 \|w\| 中位数柱图），把"Geo8 压在低频几何项"从 20 条样本升级为全量结论 | 现有 `features20.npz` 只有 20 条；需改 `visualize_final.py` 让它在 select 全量上只导出 `spatial_coeff`（不出图） | 单卡约 15–25 分钟/臂 |

> **数字口径（讲稿用，出处见 MODELCARDS 卡 4）**：完整 4,225 条 select 上 Geo8 selection score **42.427** / AUC **0.9152** / soft-IoU **0.5277**；VLM14 **46.319** / **0.8658** / **0.5030**。VISUALIZATION_REPORT.md 第 11–12 行的表就是这组数。

---

## P17 · 改进方案示意（Stage-Where-A / Stage-Where-B）

**无任何现成素材**——该方案的状态在文档首行写着 `DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED`（`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md:3`）。以下全部 **[自绘]**。

### 要画的主流程（从左到右一条链）

| 环节 | 画什么 | 关键标注（mentor 会追问的数字） | 出处 |
|---|---|---|---|
| ① merger 前取特征 | 一个 Qwen3-VL vision tower 方块，箭头从**最后一个 Vision block 之后、主 merger 之前**引出 | `F_pre ∈ R^[B × H/16 × W/16 × 1024]`；短边 512、保持比例、32 对齐、长边 ≤2048；**不是额外跑一次 DenseCLIP，复用同一次视觉前向** | 协议 §4.1（L154-162） |
| ② Phi-64 | 一个"基底堆叠"方块：`geo5(5) + L,S(2) + semantic(64)` = **71 维方向特征** | `B: 1024→64` 线性投影；64 个语义通道**逐图对 `[1, geo5, L, S]` 做最小二乘残差化再标准化**，防止语义 projector 偷学坐标/亮度/饱和度 | 协议 §4.2（L164-177） |
| ③ 全局参数 | 从 MetaCanvas 引一条箭头进来，标 `w0, w_dir, alpha, rho`；`w_dir = normalize(w_raw)`、`alpha = softplus(alpha_raw)`、符号按"绝对值最大系数为正"定死 | 输出头**只产生全局参数，不产生 dense logits** | 协议 §4.2 L178-181、§5.1 L246 |
| ④ 组合成标量 | 一个求和符号 → `s_low(p) = 3·tanh((w0 + alpha·⟨phi_dir, w_dir⟩)/3)` | — | 协议 §4.2 L180 |
| ⑤ guided upsample | **一条粗箭头，旁边加一个红色禁止标**：只对**标量 `s_low`** 做一次 edge-aware guided upsample 得到原图比例的 `s(p)`；**不能先上采样 64 个通道再组合** | 逐字："组合完成后只对标量 `s_low` 做一次 edge-aware guided upsample …**不能先上采样 64 个通道再组合**"（协议 L183） | 协议 §4.2 L183 |
| ⑥ 一句话结论框 | "边缘分辨率由 `F_pre` 的 `H/16 × W/16` 网格 + 原图 guidance 决定，**不再由 MetaCanvas 的 8×8 / 16×16 query 数决定**" | 这句话正是 P13 那个 bug 的解药，务必画出来 | 协议 §4.2 L185 |
| ⑦ readout | 分叉两条：`R-Band`（可学习平顶 bandpass + polarity，`h>0`、`k∈[1,40]`、`pi=sigmoid`）与 `R-CBand12`（E2 已验证的固定中心 12 高斯归一化竞争，`mu_i=linspace(-3,3,12)` 固定、`sigma_i∈[0.025,0.30]`、`o_i,c_i∈(0,1)`） | 旁注："固定中心不是低分辨率 mask，它是沿一维连续 s 轴的 readout，空间细节仍在 s(p) 里" | 协议 §4.3 L187-208 |

### 要画的第二张（实验矩阵，建议做成小表格而非流程图）

| 块 | 内容 |
|---|---|
| Stage-Where-A 四臂 | `BA-0-Fixed`（seeded orthogonal 1024→64，不训练，no-calibration 控制）｜`BA-1-Band`｜`BA-2-CBand12`｜**`BA-3-Joint`（预注册主方案，后续 8 个主臂的固定 projector，不事后切换）** — 协议 §4.4 L212-223 |
| Stage-Where-B 四种 MetaCanvas 结构 | `MC8-Joint`（8×8=64 query）｜`MC16-Joint`（16×16=256）｜`MC16-SplitHead`（w 与 rho 独立 attention pool + 独立 head）｜`MC16-DualCanvas`（两套独立 query bank/connector stream，只共享 frozen VLM） — 协议 §5.2 L248-257 |
| 8 个主臂 | 四结构 × 两 readout = `W01..W08`，笛卡尔积，**不做短跑筛选、不中途淘汰** — 协议 §5.3 L259-273 |
| Where Loss | `L_mask = (1-softIoU) + 0.25·balanced_BCE + 0.10·boundary_F1_3px`；oracle 辅助 `L_s`(Huber) / `L_curve` / `L_dir`；**两段权重**：前 30% steps `1.00/1.00/0.10`，后 70% `0.25/0.25/0.05` — 协议 §5.5 L291-319 |
| Where gate（9 条，必须同时满足） | soft-IoU ≥0.75｜相对逐图 oracle ≥85%｜p10 ≥0.55｜AUC_target ≥0.80｜boundary F1/oracle ≥75%｜shuffle 后 IoU 降幅 ≥0.20｜`std(s_pred)/std(s*)` ≥0.60｜global mask soft-IoU ≥0.98｜GT/generated context gap ≤0.05 — 协议 §5.6 L321-344 |

> **画图时的两个"别画错"**：
> 1. 这一版的底座是 **Qwen3-VL**（协议标题 + §4.1），**不是**现在跑的 VeraRetouch/Llava-Qwen2。P17 上如果画成同一个模型，mentor 一追问结构就露馅。
> 2. connector 宽度固定 512、**6 个 pre-norm Transformer blocks、8 heads、FFN 2048**，顺序是 `self-attn(Q) → cross-attn(Q, H_where) → cross-attn(Q, F_pre) → FFN`，各 cross-attention residual gate 零初始化；`Q_where` **只读 `<where>...</where>` 的 token hidden + `F_pre` + 真实宽高比的二维位置编码，不读 `<color>` hidden、不读 `I_tar`**（协议 §5.1 L229-246）。

---

## 汇总

按**唯一文件**计（编号 #1–#29 中有多个条目含多文件，此处已展开去重）：

| 页 | 现成文件数 | 明细 |
|---|---:|---|
| P13 | **11** | `compare20_p01..p04` + `compare20_overview`（5）｜`best_where_w14_metaquery.png`（1）｜`gallery20_where_w14_metaquery_overview` + `p01..p04`（5） |
| P14 | **7** | 两臂 `feature_spatial_basis.png`（2）｜Geo8 `where_basis_contrib20_p01..p05`（5） |
| P15 | **13** | E2 viz 下 `ring_evidence` / `constrained_axis_evidence` / `constrained_axis_response` / `constrained_axis_sweep` / `beta_reach_vs_M` / `ablation_dims` / `success_{ring,semantic,linear,radial_ell}` / `failure_{ring,gauss_ring,semantic}` |
| P16 | **20** | `compare/` 3 张｜两臂 `where_coefficients20.png`（2）｜两臂 `where20_overview + p01..p04`（10）｜VLM14 `where_basis_contrib20_p01..p05`（5） |
| P17 | **0** | 方案状态 = `DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED` |

| 类别 | 数量 |
|---|---:|
| 现成图（已 `ls` 核实存在，去重后） | **51** |
| 其中我已开图逐张核对过内容 | **10**（#1、#4、#6、#7、#9、#10、#11、#12、#22、#23） |
| 需生成 | **4**（G1 单卡 ~5 min｜G2 CPU <1 min｜G3 CPU <1 min｜G4 可选，单卡 15–25 min/臂） |
| 需自绘 | **2**（P14 基底示意图；P17 改进方案结构图，建议拆成流程图 + 实验矩阵表两张） |
</content>
</invoke>
