# Instruction-Guided Local Photo Retouching —— 投稿盘点报告

日期：2026-08-16 20:xx（写作时 gpu0/gpu1 仍在跑 E031_MLP_CD256_L3 / E031_MLP_LR1E3）
范围：what 分支（`q3vl/whatb`）+ where 分支（`q3vl/whereb`）
纪律：**§3 / §4 的实验小节只列数字**，一切叙事性主张集中在 §2（contribution，标注为投稿叙事）
与 §5（计划）。数字均从 `eval_final/metrics.json` / `steps.jsonl` / 提案 RESULT 节直读。

---

## 1. 任务与当前系统形态

### 1.1 任务定义

输入：图像 $I$ + 自然语言修图指令。输出：局部精修后的图像

$$\hat I = (1-\alpha)\odot I + \alpha\odot f(I)$$

- $\alpha(x,y)\in[0,1]$：**where 分支**产出的连续空间场（不是二值 mask）。
- $f:[0,1]^3\to[0,1]^3$：**what 分支**产出的全局颜色变换，参数化为 GLUT
  （$N=48$ 个 3D 色域高斯：$\mu$ 3 + Cholesky 6 + opacity 1 + 局部仿射 $M$ 9 + $b$ 3，
  外加全局仿射 $G$ 9 + $g$ 3，共 $22N+12=1068$ 维）。

两分支从**同一个冻结 VLM**（Qwen3-VL-4B，base SFT `q3vl_base_sft_v2seg_20260814/checkpoint-4976`）
的两个专用 token 上读出：`<seg_where>` → $\alpha$，`<seg_color>` → $z\in\mathbb R^{2560}$ → $\theta$ → $f$。
整条链路可微。

### 1.2 数据

由 `dataset_build/` 的**逆向退化管线**构造：对真实预设库（`preset_bank_full`，4,051 条 LUT
= cube 4,000 + 3dl 51，另 3,727 条 param 预设）在真实源图上重放，SAM3 出主体掩膜，
`raster_geometry` 按闭式 mask 族（radial / band / linear / semantic）渲染 `.cgt` 作为 $\alpha$ 真值，
`.vrmeta.json` 记 family/geom 参数与 `lut_id`，OneAlign 排 8 候选出 winner。

| 集合 | n | normal-only n | 用途 |
|---|---|---|---|
| train（`sft2seg-20260804` + L8 `prod-l8-local400k-20260812`） | 159,215 + 46,129 | 93,934 + 25,894 = **119,828** | 训练 |
| V_where | 896 | **515** | where 选型 |
| V_what | 897 | **567** | what 选型（唯一） |
| T_final | 918 | 533 | 终测（LUT 见过），每 arm 一次 |
| T_lut_unseen | 433 | 252 | 终测（`lut_id ∩ train = 0`），每 arm 一次 |

切分用 sha1 规则族（`verasplit-v1`）；`winner_confidence=low` 不进训练也不进评测 GT；
评测 headline 一律 normal-only。四个评测集与合并训练集的 `sft_id` / `source_image_id` 交集
**全部为 0**（2026-08-16 独立复核）。

---

## 2. 三点 Contribution（投稿叙事，与既有工作的区分）

> 本节是**主张**，不是实验结论。每条后面挂的是已出板的支撑数字。

### C1 · 把「局部精修」拆成可分别监督、端到端可微的 where × what 因子分解

**主张**：现有工作要么只做全局（一个 LUT/一组参数管整张图），要么靠外部不可微工具拿局部掩膜。
本工作把局部精修写成 $\hat I=(1-\alpha)\odot I+\alpha\odot f(I)$，其中 $\alpha$ 是**指令条件的连续场**、
$f$ 是**基元参数化的颜色变换**，两者各自在**参数空间**（而非仅像素空间）有真值、可独立消融，
且合成式全程可微，无外部软件调用。

**与谁不同**：

| 工作 | 局部能力 | 颜色变换载体 | 可微性 | 指令条件 |
|---|---|---|---|---|
| **VeraRetouch** (arXiv 2604.27375) | 无（全局三路 latent：lighting / global color / specific color） | ConditionalMLPDecoder（Retouch Renderer，2,577,795 参数） | 全可微 | 有（0.5B VLM 出 plan） |
| **AceTone** (arXiv 2604.00530) | 无 | LUT 体素 VQ tokenizer + 自回归 token | tokenizer 可微，整链不端到端 | 有 |
| **JarvisArt** (arXiv 2506.17612) | 有，但掩膜来自 Adobe Lightroom | Lightroom 200+ 参数 | **不可微**（外部软件） | 有 |
| **JarvisEvo** (arXiv 2511.23002) | 有，同样接 Lightroom；iMCoT + SEPO 自演化 | Lightroom | **不可微** | 有 |
| **GLUT / CGLUT** (arXiv 2605.19889) | 无 | 3D 色域高斯（本工作的载体来源） | 全可微 | **无**——条件是闭集可学查表 embedding $e_\ell=E[\ell]$，$L\in\{7,75,225\}$ |
| **本工作** | 有，$\alpha$ 为连续场，参数空间有闭式真值 | GLUT（$22N+12$） | 全可微 | 有，且**开集**（$z$ 来自语言，非查表） |

**已出板支撑**：
- where 侧最优臂 ST（多 special token + 视觉 LoRA）matched-area top-k IoU **0.77390 / 0.76590**（双种子），
  vs 中心先验 0.3242（m_sem 层）、随机 top-k 地板 0.2254。
- what 侧 `E030_P4_MLP` headline ΔE00 **4.7338**（seed2 4.6254），vs 恒等 8.2926、
  库平均 7.6323、桶检索 6.1553、库内 oracle 0.8253；PSNR 28.0184 vs 恒等 22.6101。
- GLUT 的闭集 embedding → 语言 $z$ 的替换是本工作 EPR-024 的定义性改动（CGLUT 全文无 held-out 风格实验）。

### C2 · 「写入式 token + 多假设 query 解码」的空间场读出，且给出增益归因的因子分解

**主张**：把语言侧条件接进空间头，**放在哪里（in-context vs 外挂桥）不重要，能不能让模型写入 token 才是增益开关**。
本工作用 $K$ 个新词表 token（embedding 可训、forward hook 写入）+ 语言侧-only LoRA，
下游 $K$ 个 query 各出一张假设场 + 选择头，构成 where 分支。

**已出板支撑（EPR-011，19 臂，全部预注册后出数，配对 sign-flip permutation $10^4$，n=224）**：

| 因子 | 配对 Δ vs 对照 | p |
|---|---|---|
| ST（in-context token 写入 + LoRA）vs 1200 步锚点 | **+0.0170 / +0.0180**（双种子） | 0.0082 / 0.0089 |
| MQ（in-context 冻结）vs D（外桥冻结） | −0.0035 | 0.170 |
| **ST − MQ** | **+0.0171** | 0.0101 |
| LORA_UNIQK8_LANG（外桥 + 语言 LoRA）vs D | **−0.0054** | 0.0015 |
| P3_LANG（P3′ 头 + 语言 LoRA）vs 锚点 | +0.0002 | 0.926 |
| 2×2 交互项（ST 为全目标，近似） | **+0.0225** | — |

即：冻结档下 in-context 与外桥无差（|Δ|<0.005）；加微调后只有 in-context 写入通路兑现，
外桥配微调为**显著负**，P3′ 头配微调为**零**。

**附带的第二条可报事实**：多假设场**集体持有**的信息高于当前单一输出——
K=8 续训 3500 步的 `best-of-K = 0.8072 > M0 0.79095`，而 `sel_is_best` 仅 **0.128**，
选择缺口 0.757 → 0.807 = **0.050**。

### C3 · 反证优先的评测协议 + 可反事实配对的数据集

**主张**：指令引导修图的既有评测（PSNR / SSIM / ΔE 对单一 GT）**无法区分**「理解了指令」与
「学到了一个好的无条件平均编辑」。本工作把评测本身作为贡献：同图配对反事实指令三负控制
（shuffle / irrelevant_words / fixed_phrase）+ antonym + subject-swap 门 + 五条平凡基线阶梯
+ 先验地板列 + 预注册判据的**运行时断言**（eval 启动时校验判据函数确实被调用）。

**这不是形式主义，有三次实证**：

1. **AUC 被处决**：中心先验 AUC 0.836 跑赢全部读出；AUC 看不见 2.2 dB 的真实差异。
   本工作因此禁用 AUC，空间场判据强制三列并排：matched-area top-k soft-IoU
   + grid 级边界 F1 + 中心先验基线列（任何定位主张必须出示配对 Δ + p）。
2. **IDGATE 臂**：headline 10.149，而**三个负控制 Δ 全为 0**——该臂完全不看指令，
   但在只报 ΔE00 的表里仍是一个「能跑的数字」。
3. **CONDINST / PRND 臂**：field 逐格恒为 1.0，`swap_delta = 0.0`、$\Delta_{\text{fix}}=\Delta_{\text{irr}}=0.0$，
   而 top-k IoU 仍报出 0.169 / 0.172 —— 实测 12/12 证实其 top-k mask 等于「展平光栅序前 k 格」，
   是 `torch.topk` 的 tie-break 产物，不携带任何预测。

**数据侧的支撑**：`.cgt` 由 `raster_geometry` 从闭式 mask 族解析渲染（可任意分辨率求值），
`.vrmeta.json` 携带 family/geom 参数与 `lut_id` ⇒ **$\alpha$ 与 $f$ 都在参数空间有真值**，
可以做「同图换指令」的反事实配对；FiveK / PPR10K 只有 (before, after) 像素对，做不出这种配对。
另外 `T_lut_unseen`（$lut\_id\cap train=0$，252 条 normal-only）给出真正的 held-out 风格集。

---

## 3. where 分支盘点

### 3.1 要解决的问题（按已测余量排序）

| # | 问题 | 已测数字 |
|---|---|---|
| P1 | **选择瓶颈**（最大可测余量 ≈ 5 分） | best-of-K 0.8072 vs 按 sel argmax 0.7571；`sel_is_best` 0.128 |
| P2 | **缩放瓶颈**（统一 query 头不随训练缩放） | K8@3500 步 0.7571 vs M0(P3′ 续训@3500) **0.79095**，配对 Δ **−0.0362**（p=1e-4，CI95 [−0.0479,−0.0249]） |
| P3 | **分辨率 / 边界瓶颈** | 监督在 grid 32×48（每格约 16 px）算 BCE；已登记现象：边缘不清晰、band/linear 预测退化成 radial 形 |
| P4 | **契约瓶颈**（视觉塔 LoRA 与冻结 sim-norm 契约结构性不相容） | 三次域断言处决，死亡速率 = 头对 sim 通道依赖度：P3′ step130 / 桥 step400 / in-context 满 1200 但续训 step≈1720 死 |
| P5 | **口径风险**（审稿人视角） | headline 是 GT 面积 oracle top-k（外部不可比、系统性偏高）；**固定阈值列至今未实现**（审阅意见 W1） |
| P6 | **能力上界未知** | U_replay 天花板测量已整体撤回（K17）：几何族参数是 subject mask 的确定性导出，重放把确定量当自由量重抽，测出的「歧义」是协议自造的。当前**没有任何有效的上界测量** |

参照线：锚点 P3′@1200 = **0.74172**；现役最优 M0 = **0.79095**；用户口径可用线 **0.85**；
oracle 0.9737；中心先验 0.5088；随机 top-k 地板 0.2582（V_where 全体口径）/ 0.2254（可视化 campaign 口径）。

### 3.2 调研并忠实移植的工作

| 方向 | 参考工作 | 落成的臂 |
|---|---|---|
| `[SEG]` token → 分割头 | **LISA** | EPR-018 `SEGSAM` |
| SAM mask decoder / iou_token / MLP 质量头 | **SAM** (2304.02643)、**SAM2** (2408.00714) | EPR-012 `iousel`、EPR-019 `SAMDEC` |
| masked cross-attention 细化 | **Mask2Former**（+ EoMT 的退火 poly_power 0.9） | EPR-013 `mar` / `mar2` |
| 门控核更新迭代 | **K-Net** (`kernel_updator.py`) | EPR-014 `knet` |
| 一对多辅助 query / 松弛 WTA | **DETR / Mask2Former** 的匈牙利匹配族；SAM2 的 `supervise_all_iou` | EPR-015 `rwta` / `rwta_e01`、EPR-016 `o2m` |
| 点采样监督 | **PointRend** | EPR-020 `PRND` |
| 隐式坐标场 | **LIIF** | EPR-021 `LIIF` |
| 抠图细节解码器 | **ViTMatte**（Detail_Capture） | EPR-022 `MATTE` |
| 条件卷积分割 | **CondInst** | EPR-023 `CONDINST` |
| in-context query 写入 | **MetaQueries**、**MGIE**、**PixelLM**、**CogACT** | EPR-011 `MQ` / `ST` / `ST_LANG` |
| 读出归一化（余弦点积 + 跨 query logit 归一） | — | EPR-017 **未排作业** |

### 3.3 已跑实验与结果

口径：V_where **local / generated / normal-only**，matched-area top-k IoU（$k=$ GT 面积），
checkpoint = `amort_final.pt` = step1200（除注明），配对 sign-flip permutation $10^4$。

**(a) EPR-011 主表（八臂，1200 步，全臂 gate PASS，M3 守卫零触发）**

| 臂 | 配置 | headline | 配对 vs 锚点 (p) | 配对 vs B (p) | best-of-K | sel_is_best |
|---|---|---|---|---|---|---|
| A | K4 + Fourier(16,1.0) | 0.7360 | −0.0025 (0.427) | +0.0023 (0.360) | 0.7526 | 0.228 |
| B | K4 无坐标（参照） | 0.7265 | −0.0048 (0.089) | — | 0.7382 | 0.081 |
| C | K1 | 0.7364 | −0.0038 (0.074) | +0.0010 (0.619) | 0.7348 | 1.000\* |
| **D** | **K8** | **0.7495** | +0.0034 (0.203) | **+0.0082 (0.0003)** | 0.7514 | 0.206 |
| E | K4 + Fourier(16,3.0) | 0.7398 | −0.0026 (0.418) | +0.0022 (0.275) | 0.7546 | 0.184 |
| P | K4 + presence | 0.7436 | −0.0024 (0.385) | +0.0024 (0.182) | 0.7499 | 0.100 |
| PIXQ | K4 + query→pixel attn | 0.7436 | −0.0009 (0.715) | **+0.0039 (0.010)** | 0.7421 | 0.147 |
| FQ | K4 按族指派赢家 | 0.7382 | **−0.0065 (0.004)** | −0.0017 (0.370) | 0.7473 | 0.331 |

\* K=1 的 `sel_is_best` 恒 1，无信息。

K 曲线：K1 0.7364 / K4 0.7265 / **K8 0.7495** / K16 0.7436（K16−K8 = **−0.0076**, p=0.0003）。

读出形态（K=4 冻结，配对 vs hidden 读出）：解析码 **+0.0052**（p=0.0011，唯一显著）>
attention map +0.0027（p=0.110）≈ hidden（基准）。

因子组合（三个独立样本，交互项全负）：
K8PIXQ−D = −0.0025（交互 −0.0064）；RCODE_K8−D = −0.0059（交互 −0.0111）。

**(b) EPR-012~016 五特性（1200 步，基底 = ST_LANG 0.7455）**

| EPR | 臂 | headline | $\Delta_{\text{fix}}$ | $\Delta_{\text{irr}}$ | gate |
|---|---|---|---|---|---|
| — | `ST_LANG`（基底） | **0.7455** | 0.0179 (1e-4) | 0.0205 (1e-4) | PASS |
| 012 | `iousel`（IoU 回归选择头，全候选 L1） | **0.7451** | 0.0211 | 0.0224 | PASS |
| 012 | `iousel_wonly`（只监督赢家） | 0.7436 | 0.0192 | 0.0212 | PASS |
| 013 | `mar`（masked cross-attn 细化，1 层） | 0.7437 | 0.0182 | 0.0329 | PASS |
| 013 | `mar2`（2 层） | 0.7315 | 0.0128 | 0.0148 | PASS |
| 014 | `knet`（K-Net 门控核更新） | 0.7447 | 0.0111 (3.9e-3) | 0.0085 (2.4e-2) | PASS |
| 015 | `rwta`（松弛 WTA，ε 默认） | **0.7482** | 0.0119 (2.2e-2) | 0.0211 | PASS |
| 015 | `rwta_e01`（ε=0.1） | 0.7270 | 0.0027 (**0.597**) | 0.0034 (**0.501**) | PASS |
| 016 | `o2m`（一对多辅助 query 组） | 0.7436 | 0.0172 | 0.0153 | PASS |

（`rwta_e01` 两个负控制 Δ 的 p 值分别 0.597 / 0.501，CI95 跨 0。）

**(c) EPR-018~023 六个像素分辨率头（1200 步，2026-08-14）**

| EPR | 臂 | headline | m_sem top-k IoU | 边界 F1 | gate | $\Delta_{\text{fix}}$ | $\Delta_{\text{irr}}$ | swap_delta |
|---|---|---|---|---|---|---|---|---|
| — | `ST_LANG`（对照） | **0.7455** | 0.8198 | 0.9646 | PASS | 0.0179 | 0.0205 | 0.0523 |
| 018 | `SEGSAM`（LISA [SEG]→SAM decoder） | **0.7208** | 0.7928 | 0.9523 | PASS | 0.0358 | **0.2650** | 0.0852 |
| 022 | `MATTE`（ViTMatte Detail_Capture） | 0.7005 | 0.7716 | 0.9230 | **FAIL**（area_ratio 0.776 < 0.8） | 0.0182 | 0.0388 | 0.0089 |
| 019 | `SAMDEC`（SAM 头从零训 + 像素监督） | 0.6938 | 0.7304 | 0.9147 | PASS | 2.9e-05 (**0.110**) | −5.6e-06 (**0.735**) | 1.4e-06 |
| 021 | `LIIF`（隐式坐标场） | 0.6418 | 0.6219 | 0.8366 | **FAIL**（area_ratio 0.698） | 0.0965 | 0.1443 | 0.0441 |
| 020 | `PRND`（PointRend 点采样） | 0.1717 | **0.0000** | **0.0000** | **FAIL**（swap_delta 0.0） | **0.0** | 2.8e-06 | **0.0** |
| 023 | `CONDINST`（条件卷积） | 0.1692 | **0.0000** | **0.0000** | **FAIL**（area_ratio 2.44、swap_delta 0.0） | **0.0** | **0.0** | **0.0** |

随机 top-k 地板（campaign 口径）**0.2254**；中心先验（m_sem 层）0.3242。
`SAMDEC` 三个负控制的 Δ 量级为 $10^{-5}\sim10^{-6}$、p 分别 0.110 / 0.735。
`PRND` / `CONDINST` 实测 12/12：top-k mask 等于展平光栅序前 k 格（field 逐格恒 1.0）。

**(d) 尚未出数 / 已注销**

- EPR-017（读出归一化）：**从未排作业**。
- `ST_LANG_CONT@3500 vs M0` 的决战实验：**从未跑成**（视觉 LoRA 路线被域断言锁死；语言侧路线未接续）。
- 固定阈值列（审阅 W1）：未实现。

### 3.4 可视化结果

产物：
`docs/assets/where_arms_20260815/where_arms_heatmap.png`（纯场热图）
`docs/assets/where_arms_20260815/where_arms_overlay.png`（场叠原图，严格逆映射 `viz_grid_to_img`，alpha 0.55）
`docs/assets/where_arms_20260815/samples.json`（12 例的选样规则、逐臂 provenance、重算校验）

图面纪律（已落实）：
- **色标固定 0..1**，每个 mask 面板同一条标尺，**禁逐图 min-max**；
- 每格标注读的是**未归一化原始场**的 IoU / std / 唯一值数（`u`）；
- 12 例按 family 分四组（radial / band / linear / semantic），每组 3 例，选样规则写在 `samples.json`
  （按 7 臂间 max−min 的跨臂分歧度选，再在 5 个非退化臂内取分歧最大的 3 例，去重）；
- `reconstruction_check_max_abs_delta = 0.0`（7 臂全部）：图上的 top-k IoU 与
  `eval_final/per_sample.jsonl` 逐位一致；
- `topk_iou_cpu_recompute` 列并排保留（CPU/CUDA 的 topk tie-break 不同，差异显形而非静默）。

图面可直读的现象（只描述像素，不做机制解释）：
- `CONDINST` 全部 12 例整幅纯黄（场恒 1.0）；`PRND` 全部 12 例整幅纯青，仅个别孤立格点异色（`u`=1~14）。
- `radial` 三例：`ST_LANG` 与 GT 的亮斑位置、尺度一致（IoU 0.881~0.889）；`SEGSAM`/`MATTE`/`SAMDEC`
  的亮区被打散成多个团块（IoU 0.532~0.635）。
- `band` 三例：GT 是斜条带；`ST_LANG` 在 `2df633d05b71` 上给出近乎均匀场（`s=0.0330`，IoU 0.410），
  该例 `LIIF` 反而最高（0.696）。
- `linear` 三例：GT 是半平面渐变（area 0.539~0.667）；`ST_LANG` 在 `656d42faaee3` / `f436188d3677`
  上 IoU 0.910 / 0.876，在 `b5dd93fd8aa6` 上 0.483。
- `semantic` 三例：`MATTE` 在 `8b59235a1c65` / `6ec6373df6d9` 上 IoU 0.870 / 0.863 高于 `ST_LANG`
  的 0.800 / 0.722；`85c4bbc5d63c` 上 `ST_LANG` 0.910。

> ⚠ **图上有一处必须在投稿前修掉的不一致**：标题栏写 `ST_LANG 0.774`，但
> `samples.json.provenance.ST_LANG.run_dir = amort_UNIQ_stlang_20260813`，该 run 的
> `topk_iou_median_normal_only = 0.745502`。**0.7739 是 `amort_UNIQ_st_20260812`（视觉 LoRA 的 ST 臂）的值**。
> 面板里的逐例 IoU 与 `stlang` 的 per_sample 逐位一致（重算校验 0.0），所以是**标题数字取错**，
> 不是面板取错。

---

## 4. what 分支盘点

### 4.1 要解决的问题

| # | 问题 | 已测数字 |
|---|---|---|
| Q1 | **打不过平凡基线**：早期臂连「不看指令用训练集平均 LUT」都打不过 | CARRIER 7.895 vs B1_libmean 7.6323；IDGATE 10.149 vs B0_identity 8.2926 |
| Q2 | **loss 配方量纲失衡** | `L_hc` 的 chroma 未归一化（实测 mean $C$=32.5）⇒ `10·L_hc` 等效约 **325×** `L_rec`；CARRIER 全量 117,399 步把 `L_hc` 压了 −92%，而 `L_rec` 0.5596→0.1770，**恒等变换的 `L_rec` = 0.1756** |
| Q3 | **条件通路死亡** | `head_color` 末层隐层 ReLU 全局死亡率 0.469(s0) → 0.938(s500) → **1.0000(s1300)**；之后只剩末层 bias，32 个样本预测逐位相同，梯度精确为 0 |
| Q4 | **饱和点零梯度** | GLUT 全局分支硬 `clamp(0,1)` 从 step 50 起 **100%** 查询点被裁，该分支回传精确零梯度 |
| Q5 | **一对多**（同一句「暖一点」对应库里多条 LUT） | L1 的最优解是条件中位数，L2 是条件均值；当前 headline 4.7338 vs 桶检索下界 6.1553 vs oracle 0.8253 |
| Q6 | **大 batch 下的无条件解塌缩** | 色批 ×256 配 lr 1.6e-2：7 行 7 死，其中两行不是 NaN 而是**条件性塌到 2.02e-09**（地板 1e-4），`std_over_queries` 0.233 ✓ 但 32 个样本给出同一个变换 |

### 4.2 调研的工作（`docs/RESEARCH_whatb_loss_arch_2026-08-15.md`，24 篇逐条打开原始文件核实）

**监督位置**（横向事实）：24 篇里在**函数值 / 参数空间**直接监督的只有三家——
**NILUT**（RGB 函数值 L1，与本项目 B×Q 采样口径同构）、**AceTone**（voxel MSE）、
**StatLUT**（SmoothL1，且是唯一两边同时挂：$\lambda_{lut}=1.0+\lambda_{img}=0.5$）；其余全部只在图像空间监督。

**ΔE 的位置**：本轮打开的全部来源中，ΔE00/ΔE76 只出现在 (a) 评测指标、(b) RL reward（AceTone
`1/(max(2,ΔE)−1)` + DeQA）。**没有找到把完整 ΔE00 当可微训练 loss 的已发表工作或公开实现。**

**正则权重的稳定族值**：TV/smooth **1e-4**、monotonicity **10**、融合权重 L2（sparse）**1e-4**
（3DLUT / AdaInt / SepLUT / 4D LUT / DualBLN 同值）。

**结构侧借鉴**（EPR-030 §1.4，全部当日打开 raw 文件核实）：
StatLUT（query = 格点 + 可学 3D 位置编码 $PE_R\oplus PE_G\oplus PE_B$；decoder 6 层 / $d_{model}$=512 / 8 头；
零初始化末层 FFN 投影）、Bias-HyperInit（PMLR v205 beck23a：weight 全零 + bias = 目标初值）、
Text-to-LoRA（2506.06105：`nn.init.zeros_(head.weight)` + `head.bias.copy_(init_bias)` 的可运行实现）、
Splatter Image（逐参数组给 gain 与 bias）、NILUT（`# more stable than L2`）、GELU（1606.08415）、
Dying ReLU（1903.06733v3 Thm 3.4）、GradNorm（只当监控落盘，不调权重）。


### 4.3 已跑实验与结果

口径：V_what **normal-only n=567**，headline = `.contexts.*.headline_normal_only`（ΔE00，越小越好），
GT $\alpha$，短边 512，`area_resize`；配对 Δ + 10k bootstrap CI + Wilcoxon p。
**禁用顶层 pooled `.baselines`**（混用少算约 0.031）。

**(a) 五条平凡基线（不随训练变，逐块板自算且逐位一致）**

| 列 | 值 |
|---|---|
| `B0_identity`（什么都不做） | **8.2926** |
| `B1_libmean`（不看指令，用训练集平均 LUT） | **7.6323** |
| `B2_librandom`（R=8） | **10.0989** |
| **`B3_bucket_retrieval`（真正要打的线，桶级下界）** | **6.1553** |
| `B4_oracle`（库内最优） | **0.8253** |

**(b) 旧口径全量最优板 `E030_P4_MLP`**（`--backbone mlp`，**447,212** 参数，纯 L1，
`--data v2seg+l8`，`--batch-split 32x256` = 8192 色/步，149,800 步，lr 1e-3）

| 列 | mean | 配对 Δ（臂 − 基线） | p_wilcoxon |
|---|---|---|---|
| **headline_normal_only** | **4.7338** | — | — |
| B0 | 8.2926 | **−3.5587** | 1.2e-82 |
| B1 | 7.6323 | **−2.8985** | 1.3e-70 |
| B2 | 10.0989 | −5.3651 | 5.1e-94 |
| **B3** | 6.1553 | **−1.4215** | 3.5e-28 |
| B4 | 0.8253 | +3.9085 | 1.3e-82 |

三负控制 Δ：shuffle **4.4937**（p=2.2e-81）／irrelevant **3.4245**（p=2.2e-77）／const **3.4903**（p=2.7e-79）。
种子复现 `E030_P4_MLP_SEED2` = **4.6254** ⇒ **run-to-run 方差 0.1084**（B0 Δ −3.6672 / B1 Δ −3.0032 /
B3 Δ −1.5367 / $\Delta_{\text{shuffle}}$ 4.7846）。

PSNR（额外诊断，不在 12 个预注册键内，`experiments/prs/EPR-030_shared-query-backbone/psnr_diagnostic.json`）：

| 列 | `E030_P4_MLP` mean / p50（n） | `SEED2` mean / p50（n） |
|---|---|---|
| **arm** | **28.0184 / 27.8731**（567） | **28.2344 / 27.6621**（567） |
| B0_identity | 22.6101 / 22.2634（567） | 22.6101 / 22.2634（567） |
| B1_libmean | 23.1583 / 23.0342（567） | 23.1507 / 23.0795（567） |
| B2_librandom | 20.3989 / 20.0082（567） | 20.3835 / 20.0093（567） |
| B3_bucket | 23.7488 / 23.5402（**563**） | 23.7392 / 23.2778（**563**） |
| B4_oracle | 30.5835 / 29.7588（**122**） | 30.2286 / 29.0637（**114**） |

n<567 的两列：mse=0 的样本被计数并剔除（`B4_oracle` 445/567、`B3` 4/567），**不可与其它行并排读**。
ΔE00 交叉验证：重算 vs 板上六列差**全部 0.0**。

**(c) 结构消融（EPR-030 §4.3，各行 1 epoch = 3,745 步，同步数比较，叠加式写法）**

容量轴：基线 = qdec d=512 L=6 M=1 + GELU + Bias-HyperInit + 直通 clamp + 纯 L1，headline **6.7449**；
d 512→256，变动 **−0.0768**（6.6681）；再→128，**−0.1584**（6.5865，参数量 **1,534,626**，
低于 VeraRetouch Retouch Renderer 的 2,577,795）；L 6→4，**+0.2265**（6.9714）；L→2，**NaN**；
主干换回 MLP（447,212 参数），**−0.5668**（6.1781）。

结构轴（基底 = d=256 行 6.6681）：条件注入 1 行→4 行 **NaN**；→8 行 **NaN**；
GELU→ReLU **+0.2400**（`act_dead_max` 稳在 0.8398~0.8555，GELU 基底 6.2e-07）；
Bias-HyperInit→全零头 **−0.0510**；直通 clamp→硬 clamp **+0.2914**；加 self-attention **+0.1370**；
输出头从 0.1× 组挪回基础 lr **NaN**。

数据轴（固定 2,936 步）：消融掉 L8（n=93,934）**6.3070** → 增加 L8（n=119,828）**+0.6172**（6.9242）。

**(d) 两条已实证的方法论事实**

1. **`--qdec-mem-rows 1` 时 cross-attention 是退化的**：softmax 在单 key 上恒为 1 ⇒
   $\mathrm{cross\_attn}(h,mem,mem)=W_oW_vz$，与 query 无关。fp32 实测 5 个不同 query 的输出最大偏差：
   M=1 → **2.98e-08**（float32 噪声）；M=4 → 2.58e-01。
   ⇒ 默认档跑的不是 attention 解码器，**「MLP 打赢 transformer」目前站不住**。
2. **守卫只绑首次 quick eval ⇒ NaN 的全量会发布 `headline = 0.0` 的假板**：
   `E030_P4_MLP_NOL8` 在 step≈131,049 NaN，板上 `headline_normal_only` = **0.0**（std 0、p10/p50/p90 全 0），
   三负控制 Δ 全 0，B0/B1/B3 的 Δ 恰等于 −基线值，且 `published=true`、rc=0。复跑逐位相同。
   **采信纪律**：读任何全量板前先 `jq -c 'select(.step!=null)|{step,L_rec}' <run>/steps.jsonl | tail -3`，见 `null` 即作废。

**(e) lr 缩放判决**：色批 ×256 → lr ×16（1.6e-2）**7 行 7 死**（5 行 50 步内 NaN，2 行 step 469
条件性塌到 2.02e-09，1 行 step≈5949 NaN）；lr 1e-3 / 4e-3 / 3e-4 至今零死。

**(f) EPR-028 R1（4D 高斯条件切片）已出的 oracle 板**

G1（单 LUT overfit，A1，2,000 步，8,192 色/步）17³ 网格 ΔE00 mean **0.2594**（`R_line` 恒 0，
`cholesky_info_nonzero` 0）。同条件的 EPR-031 O0（无 generator，每 LUT 独立参数）mean **0.3375**，
step0 的 `L_rec` 两行逐位相同（0.0393543615937233）。

G2（32 LUT oracle，4,000 步，65,536 色/步）3-seed：

| 臂 | seed 810 | seed 811 | seed 812 | mean | std | 变异系数 |
|---|---|---|---|---|---|---|
| A1 | 0.29624 | 0.36402 | 0.28294 | 0.31440 | 0.04348 | 13.83% |
| A2 | 0.42584 | 0.42314 | 0.41191 | 0.42030 | 0.00739 | 1.76% |
| A3 | 0.42592 | 0.41691 | 0.39934 | 0.41406 | 0.01351 | 3.26% |

逐 seed 配对 `A3−A2` mean = **−0.00624**、std = **0.00632**（|mean| 与 std 同量级）。

**(g) EPR-031 C0（canonical LUT code，17³ = 14,739 维，3,149 条，CPU）**

解释方差累计 90% / 95% / 99% 所需维数 = **14 / 28 / 117**（秩上限 3,148）。
`code_recon_de00`（对 GT LUT 的 ΔE00 均值，纯几何量）：$d_{LUT}$=128 → **1.4121**；192 → **1.0886**；256 → **0.8978**。
9³ 对照（2,187 维）：本次 redraw（n=1,166）90/95/99 = 15/29/109；全 train 池（n=3,149）= 15/29/120。

### 4.4 在跑 / 刚死的（截至 2026-08-16 20:30）

新口径：`--batch-split 256x8192` = **2,097,152 色/步**（旧口径的 256×），`--data v2seg+l8` n=119,828，
**469 步/epoch × 40 = 18,760 步**，lr 1e-3，纯 L1。**与 §4.3(b) 的 4.7338 不可比**。

| id | 作业 | 变量 | 状态（末行 steps.jsonl） |
|---|---|---|---|
| 243 | `E031_MLP_LR1E3` | 新口径基准 | **在跑**，step 11,149，L_rec 0.1249 |
| 245 | `E031_MLP_CD256_L3` | `--cond-dim 256` | **在跑**，step 15,649，L_rec 0.1151 |
| 246 | `E031_MLP_CD512_L3` | `--cond-dim 512` | step 15,599，L_rec 0.1167 |
| 249 | `E031_QDEC_LR4E3` | lr 4e-3 | step 10,849，L_rec 0.1288，gnorm 11.8 |
| 251 | `E031_QDEC_MEM4_LR3E4` | mem4 + lr 3e-4 | step 8,599，L_rec 0.1273，gnorm 11.8 |
| 252 | `E031_MLP_NOL8_L3` | `--data v2seg`（L8 A/B） | step 9,699，L_rec 0.1180 |
| 244 | `E031_QDEC_LR1E3` | qdec d=256 | **NaN @ ≈6,949**，Killed |
| 250 | `E031_QDEC_MEM2_L3` | mem-rows 2（唯一非退化 attention） | **NaN @ ≈5,699**，Killed |
| 247/248 | `E031_QDEC_MEM4/MEM8_L3` | mem 4 / 8 | Killed |
| 276 | `EPR028R1_G3_A3` | 全 LUT oracle，18,760 步 | **Queued** |

⇒ **memory 行数扫描的现状：M=1（退化）稳；M=2 在 5,699 步 NaN；M=4 配 lr 3e-4 至 8,599 步仍稳；M=8 早死。**
非退化 attention 目前**没有一条活到出板**。

**恒等参考**：`f ≡ identity` 时 `L_rec = 0.1756`（新口径下上表六行的 L_rec 均低于该值）。

---

## 5. 接下来的计划

### 5.1 T+0 ~ T+1 周：把现有板收干净（不加新方法）

| # | 事项 | 判据 / 交付 |
|---|---|---|
| 1 | **修 §3.4 的图题数字**（`ST_LANG 0.774` → `0.7455`，或把 provenance 换成 `amort_UNIQ_st_20260812` 并改标签为 `ST`） | 重出两张图 + `samples.json` |
| 2 | **修守卫洞**：quick eval 每次都查 `steps.jsonl` 的 `L_rec` 是否 `null`，不只首次 | `E030_P4_MLP_NOL8` 的假板重跑后应 rc≠0 而非 `published=true` |
| 3 | **补 where 的固定阈值列**（审阅 W1，eval-only 重扫已有 per_sample 即可） | 全部已出板臂补一列；这是「0.85 可用线」讨论的前提 |
| 4 | **补 where 的 ORACLE-SUBJECT 上界**（喂 GT subject mask 测几何导出上限） | 把「找主体」与「导几何」的缺口拆开；m_sem 0.820 vs oracle 的差 |
| 5 | **确定性验证**：同 subject mask 重跑几何导出应逐位复现 `.cgt` | CPU 级，半天 |
| 6 | 收 E031 六行 + G3 三行；决 what 主干 | 采信前必查 `steps.jsonl` 尾三行 |
| 7 | 跑 EPR-017（读出归一化）——唯一提了案但从未排作业的 where 臂 | 补齐消融矩阵 |

### 5.2 T+1 ~ T+2 周：where 的两个主攻

- **选择头攻坚**（最大已测余量 0.050）：`--uniq-sel-weight` 扫描是零代码起点；
  两阶段 rerank / 场级置信估计需新文件。**已判负的路不要再走**：FQ（按族指派赢家）−0.0065（p=0.004）。
- **`ST_LANG_CONT@3500 vs M0`**：唯一无漂移的缩放路线（视觉 LoRA 被域断言锁死三次）。
  这是 where 分支能否声称「超过现役最优」的**决战实验**，至今一次都没跑成。

### 5.3 T+2 ~ T+3 周：公开数据集（FiveK / PPR10K）

**问题**：what 分支的监督在**函数值空间**（$B$ 条 LUT × $Q$ 个查询色，比 $f(x)$ 与 $L_\ell(x)$），
而 MIT-Adobe FiveK 与 PPR10K 只有 (before, after) 像素对，**没有 LUT 真值**。

**处理方案（推荐 A，B 作对照）**：

**A · 逐对拟合 GLUT 参数（复用已有 O0 通路，改动最小）**
EPR-031 的 O0 档已经是「每条 LUT 一组独立参数 Θ，无 generator」的直接载体表
（`DirectCarrierTable`，`q3vl/whatb/codec/tables.py`）。把它的目标从「LUT 的 17³ 网格值」
换成「(before, after) 的逐像素配对色」，即
$$\min_{\theta_i}\;\big\|f_{\theta_i}(I^{\text{in}}_i) - I^{\text{out}}_i\big\|_1$$
在图像的像素颜色分布上采样（而不是均匀网格）。产物：每张图一组 1068 维 GLUT 参数，
直接作为 what 分支的 GT。**这条路不需要新载体、不需要新 loss、不需要新评测口径。**

**B · 逐对拟合标准 3D LUT（对照 + 与 3DLUT/AdaInt/SepLUT 可比）**
按 Image-Adaptive-3DLUT 的族值解 $\min_L \|L(I^{in})-I^{out}\|_2^2 + 10\cdot L_{mono} + 10^{-4}\cdot L_{TV}$，
33³ 网格。用途：(a) 给 A 一个独立的拟合残差参照；(b) 让 what 分支能和只报 LUT 的 baseline 直接比。

**两条路共用的三道门（必须做，否则数字不可信）**：

1. **拟合残差门**：报 $\Delta E_{00}(f_\theta(I^{in}), I^{out})$ 的**分布**（不是均值一个数）。
   FiveK/PPR10K 的专家编辑**不保证是全局 LUT 可表达的**（有局部 dodge/burn、空间变化的调整）。
   残差就是「本任务的表达上界」，必须作为 data card 报出去，并设阈值（本仓库的 databuild
   用的是 `fidelity_de_max = 6.0`，建议沿用同一条线，公开集单独重标定）。
   **超阈值的对不进 what 的 GT，但可以进 where 的训练**（局部残差正是 $\alpha$ 该覆盖的地方）。
2. **PPR10K 污染门**：本仓库已记录 —— `mmart_ppr10k` 池基于 PPR10K 原图构建，
   目录名 `<groupid>_<photoid>`，实测 group id 上至 1679；PPR10K 官方 split 是
   「前 8,875 个文件训练、后 2,286 验证」⇒ 高 group id 段几乎必然落官方 val。
   `tools/data_splits/verify_ppr10k.py` 已把分布写进报告，**处置至今未拍板**。
   发 PPR10K 数字前必须处置（建议：把 `mmart_ppr10k` 源强制归入我们的 train，
   或整体从公开集评测口径剔除，并在论文里写明）。
3. **指令缺失门**：FiveK/PPR10K **没有指令**。因此公开集只能支撑两种协议：
   - **协议 (a) auto retouch（无指令）**：与 3DLUT / AdaInt / SepLUT / StatLUT / NamedCurves
     的 PSNR / SSIM / $\Delta E_{ab}$ 直接可比。这是「我们的载体够不够强」的答案。
   - **协议 (b) 指令条件**：需要给公开集**合成指令**（用 VLM 标注 before/after 的差异描述）。
     一旦合成，**三负控制必须同步合成**（shuffle / irrelevant / fixed_phrase），否则协议 (b)
     的数字在 C3 的框架下没有意义。
   **不要把 (a) 的数字当作指令条件性的证据** —— 这正是 §2 C3 第 2 条（IDGATE 三负控制全 0）
   要防的失败。

**where 分支在公开集上的处理**：PPR10K 自带人像区 mask，可直接作为 semantic family 的
$\alpha$ 参照（PPR10K 原文 $L_{HC}$ 就是人像区 w=5 / 背景 w=1 的加权 MSE）；
FiveK 无任何 mask，只能进协议 (a)。

### 5.4 T+3 ~ T+4 周：后训练（PSNR / LPIPS）

**现状**：what 分支监督在函数值空间（L1 on $f(x)$ vs $L_\ell(x)$），where 分支监督在
grid 32×48 的 BCE。**两者都没有优化图像空间的 PSNR / LPIPS。**
`E030_P4_MLP` 的 PSNR 28.0184 是**顺带测出来的**，不是训出来的。

**Stage-1（推荐先做）· 图像空间联合微调**
冻结 VLM，放开 conditioner + $\alpha$ 头，在合成式 $\hat I=(1-\alpha)\odot I+\alpha\odot f(I)$ 上加：
$$L = \lambda_{\text{fn}}\underbrace{\|f(x)-L_\ell(x)\|_1}_{\text{保住函数值监督}} + \lambda_{\text{img}}\|\hat I - I^*\|_1 + \lambda_{\text{lpips}}\,\mathrm{LPIPS}(\hat I, I^*)$$
配方族值从 §4.2 的横表取（**都是打开原文核实过的**）：
FlowLUT `L_MSE + 0.1·LPIPS`；GLARE Stage III `L1 + 0.2·SSIM + 0.01·perceptual`；
StatLUT `λ_lut=1.0 + λ_img=0.5`（唯一两边同时挂的）；NamedCurves `α=0.5 + SSIM`。
**建议起点取 StatLUT 形制**（$\lambda_{\text{fn}}=1.0$、$\lambda_{\text{img}}=0.5$）+ FlowLUT 的 `0.1·LPIPS`。

**Stage-2（可选）· 偏好 / RL 后训练**
同族已有三条：VeraRetouch 的 DAPO-AE、AceTone 的 GRPO（reward `1/(max(2,ΔE)−1)` + DeQA 美学分）、
JarvisEvo 的 SEPO（editor-evaluator 协同、无外部 reward）。
**若做，reward 必须包含指令条件项**，否则会直接奖励「输出一个好看的平均编辑」。

**后训练的四条硬纪律（不遵守则前面所有 C3 的主张作废）**：

1. **checkpoint 选择禁用 val loss**，也禁用 PSNR/LPIPS —— 沿用 quick eval 硬门 + headline 选优。
2. **两块板并存**：预注册判据板（ΔE00 / top-k IoU / 五基线 / 三负控制 / 12 个预注册键 + 运行时断言）
   与图像质量板（PSNR / LPIPS / SSIM）。**后者不得改写前者的任何一列**。
3. **后训练之后必须重跑三负控制**。图像空间 loss 天然奖励无条件平均解
   （B1_libmean 的 PSNR 23.16 已经高于恒等 22.61）—— 如果 $\Delta_{\text{shuffle}}$ 在后训练后掉下来，
   说明换来的 PSNR 是拿指令条件性买的，必须如实报。
4. **PSNR 的 n 要报**：`B4_oracle` 那一列 n 只有 122/567（445 张 mse=0 被剔除），
   **不可与 n=567 的行并排读**。这条已经在 `psnr_diagnostic.json` 里落过一次。

### 5.5 需要用户拍板的（未静默决定）

1. what 侧优化器口径要不要加 warmup / wd（StatLUT 原配方 AdamW wd=0.05 + 5-epoch 线性 warmup；
   本战役冻结的是 Adam / lr 1e-3 cosine / 无 warmup / wd=0，那是给 0.45M MLP 写的）。
   本会话共 10+ 条 NaN，全部是 gnorm 尖峰后炸。
2. cross-attention 的 memory 该放什么（M=1 是退化的，M≥2 至今没活到出板）。
   已探明基础设施支持 `(n, K, 2560)` 三维缓存；候选：`<seg_color>` 1 行 ⊕ 色彩推理六段各池化 6 行
   ⊕ 视觉 token 2×2 池化 4 行。**代价：缓存重生成约 13.5 GB / 两卡数小时。**
3. EPR-031 拟合集是否排除评测集 lut_id（3,149 → 2,056）；9³ 对照是否换抽样口径去复现底账的 1,137；
   PCA 向量空间取 sRGB 还是 Lab。
4. PPR10K / MMArt-PPR10k 的污染处置（见 §5.3 门 2）。
5. `pueue parallel -g gpu0` 目前被改成 5，用完要改回 1。

---

## 6. 投稿风险清单（审稿人一定会问的）

| # | 风险 | 现状 |
|---|---|---|
| R1 | headline 用 GT 面积 oracle top-k ⇒ **外部不可比、系统性偏高** | 固定阈值列未实现（W1）；不补则 where 的所有数字都只能内部比 |
| R2 | where 最优 0.7739 vs 现役最优 M0 0.79095 vs 可用线 0.85 | 统一 query 头**至今没有任何配置超过 M0**；决战实验未跑成 |
| R3 | 「MLP 打赢 transformer」 | **站不住**：赢的那个 qdec 跑在 M=1，cross-attention 数值上退化（偏差 2.98e-08） |
| R4 | 非退化 attention 全部 NaN | M=2 死于 5,699 步、M=4 死于 ≈2,199（换 lr 3e-4 后越过）、M=8 死于 ≈749 |
| R5 | 假板机制 | 守卫只绑首次 quick eval，NaN 会发布 `headline=0.0` 的板且 `published=true` |
| R6 | what 的新口径板一条都没出 | §4.3(b) 的 4.7338 是旧色批（8192）口径，与在跑的 2,097,152 色/步**不可比** |
| R7 | 六个像素分辨率头**无一超过** ST_LANG | 最好的 SEGSAM 0.7208 < 0.7455；两个（PRND/CONDINST）输出恒定场 |
| R8 | 可视化图题数字取错 | §3.4 的 ⚠ |
| R9 | 上界未知 | U_replay 已整体撤回（K17），当前没有任何有效的天花板测量 |

---

## 附：数据出处

- where 板：`/home/bc/data/runs/where_b/*/eval_final/metrics.json`（68 个 run 全量扫描）
- what 板：`/home/bc/data/runs/what_b/*/{metrics.json,steps.jsonl}`；
  `experiments/prs/EPR-030_shared-query-backbone/{PROPOSAL.md §4, psnr_diagnostic.json}`
- 可视化：`docs/assets/where_arms_20260815/{where_arms_heatmap.png,where_arms_overlay.png,samples.json}`
- 调研：`docs/RESEARCH_whatb_loss_arch_2026-08-15.md`（24 篇 loss 配方 + 13 个候选结构 + 11 条编造记录）
- 提案：`experiments/prs/EPR-011 ~ EPR-031`
- 外部核实（本轮当日打开）：JarvisArt `arxiv.org/abs/2506.17612`；
  JarvisEvo `arxiv.org/abs/2511.23002v2`（export API 原文，31 pages, 18 figures，接 Adobe Lightroom）
