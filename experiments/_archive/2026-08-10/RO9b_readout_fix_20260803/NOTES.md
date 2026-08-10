# NOTES — RO-9b · RO-9 读出算子的四处修复

## 〇、落盘核对声明

本目录所有数字来自 `metrics.json`（`analyze_ro9b.py` 一次跑出，seed 20260803）与
`config/verify_env.json`（实测核实）。与任务卡的出入**如实登记**，见 §四。

---

## 一、实施前核实记录（CLAUDE.md 派工协议第 1、2 条）

### 1.1 读过的文档节（只读引用节，不读全文档）

| 文档 | 节 | 取到的东西 |
|---|---|---|
| `experiments/RO1_selfself_20260803/REPORT.md` | 全 | 三算子定义与出处、后处理阶梯 A1/A2/A3/A4 的实测增量（A1 +0.009~0.012、A3 对 ClearCLIP +0.015、A2 ≈0、guided filter ≤+0.002）、vanilla CLIP 0.214 的符号反转、16×16 同网格 0.941 |
| 同上 | `tools/readout/ro1_selfself.py` 全 | `outlier_mask_mad` / `interpolate_outliers` / `find_register_neurons` / `RegisterIntervention` / `roc_auc` / `mask_to_grid` —— **直接复用口径**（本臂的 `interpolate_masked` 与 `rank_auc` 与之数值等价） |
| 同上 | `clip_naclip/VENDOR.md` | `csa` / `naclip(kkᵀ+高斯,std=5)` / `clearclip(qqᵀ, arch=reduced)` 的逐行出处与 commit —— 组 A 的算子定义**照抄这一份**，未另找来源 |
| `experiments/ROX1_clipside_20260803/REPORT.md` | 全 | 新基线锚点 `X_deictic` AUC 0.907 / AUC_target 0.523；"AUC 会骗人"的直接反例；主判据改 AUC_target 的理由 |
| `experiments/RO9_layer_verdict_20260804/REPORT.md` + `analyze_layers.py` | 全 | **AUC_target 的权威定义**（`auc_target_list`：reg_a→M 与 1−AUC(s_b,M) 混池取中位）、逐层曲线（最好层 L22 AUC 0.834 / canonical L8–15 0.648）、valid 格与掩膜口径、`SubjectMaskBank` |
| `experiments/RO3_layerhead_scan_20260803/NOTES.md` | §〇.3 / §一.2 / §二 / §三 | **逐头栈复用可行性判定**（见 §二.1）、V1–V8 实测事实（24 层/14 头/2 KV 头/span [14,270)）、5 折 source-level CV + 权重只在 fit 折估的做法 |
| `experiments/RO2_logitlens_20260803/REPORT.md` | 全 | `emb`（mm_projector 输出）是**唯一词特异**的读出点（GT 掩膜集 AUC 0.829 vs `calculator` 0.523；region 原始 logit 0.747 vs 0.534）；LLM 前向把词特异性抹掉；`word_token_ids` / `last_word` 的选词口径；指令与 image token 表示逐比特无关（ρ=0.99995） |
| `experiments/G1_s_identifiability_20260803/` | `config/g1_region_opp.json`、`analyze_g1.py` | 214 源区域对立批（**零重新采样**直接复用）、掩膜银行读取 |
| `docs/EXPERIMENTS_v3_2026-08-02.md` | §2.1 RO-9 行 + Changelog | RO-9 已判死为 where 读出、降级用途；RO 系公共判据 |
| `docs/IMPL_DOSSIER_2026-08-02.md` | §5.1 / §5.4 + 附录 B | eager 与 output_attentions 互斥的红线；RO-1 已核出的 5 处订正（`gem.create_model_and_transforms` 参数错位 / `gem_depth=7` 实为 6 层 / **GEM 默认 min-max 会踩禁逐图归一化红线** / open_clip pin `==2.24.0` / pkg_resources 弃用）——本臂**不使用 GEM**，故该 5 条不构成本臂的依赖 |

### 1.2 在线核实（IMPL_DOSSIER 附录 B 之外的外部事实，逐条打开原始来源）

任务卡点名必核：**MobileCLIP / FastViTHD 的官方实现与架构**。

| # | 核实项 | 来源（一手） | 结论 |
|---|---|---|---|
| W1 | `fastvithd` 的 layers / embed_dims / token_mixers / pos_embs | `raw.githubusercontent.com/apple/ml-fastvlm/main/llava/model/multimodal_encoder/mobileclip/mci.py` | `layers=[2,12,24,4,2]`、`embed_dims=[96,192,384,768,1536]`、`token_mixers=("repmixer","repmixer","repmixer","attention","attention")`、`pos_embs=[None,None,None,RepCPE(7,7),RepCPE(7,7)]`、`downsamples=[True]*5` —— **与本仓库 vendored `llava/model/multimodal_encoder/mobileclip/mci.py` 逐字一致** |
| W2 | 有无 cls / register token | 同上（`FastViT.forward` / `GlobalPool2D`） | **无**。全程 `(B,C,H,W)`，MHSA 内部 `flatten(2).transpose` 成 `(B,N,C)` 再还原；全局表征靠 `GlobalPool2D` 的空间均值，不靠 cls token。⇒ **不需要剥离任何前缀 token**，也意味着 RO-1 的 A3（test-time registers，要往 token 序列追加寄存器）**在 FastViTHD 上无处可加**（§四 U3） |
| W3 | 哪些 stage 有 self-attention、末层是哪一层 | 同上 + 本机实测（`config/verify_env.json` V1/V2） | 只有 stage 3 与 stage 4：`network[7]`（4 个 `AttentionBlock`，输入 `(1,768,32,32)`，24 头）与 `network[10]`（2 个，输入 `(1,1536,16,16)`，48 头，head_dim=32，scale=0.17678）。**末层 = `network[10][1].token_mixer`，正好在 16×16 网格上 = 256 个 image token 的原生网格** |
| W4 | FastViTHD 的设计意图（自注意力下采样 32×） | Apple ML Research "FastVLM" 页 + arXiv 2412.13303 | "self-attention operates on tensors downsampled by a factor of 32 rather than 16"——与 W3 实测的 32×32/16×16 两档一致 |
| W5 | `mobileclip_l.json` 配置 | `github.com/apple/ml-fastvlm/blob/main/llava/model/multimodal_encoder/mobileclip/configs/mobileclip_l.json` | `embed_dim=768`、`image_cfg{image_size:1024, model_name:fastvithd, embed_dim:3072, patch_size:64}`、`text_cfg{12 层, dim 768, vocab 49408, ctx 77}` —— 与本仓库 vendored 文件逐字一致 |
| W6 | **是否存在可对齐的 MobileCLIP 文本塔权重** | `apple/ml-mobileclip` / HF `apple/MobileCLIP2-*` 检索 | **未找到 `mobileclip_l`（fastvithd 视觉塔）对应的公开文本塔 checkpoint**：公开 MobileCLIP/MobileCLIP2 变体是 S0/S1/S2/S4/B（ViT/MobileOne 视觉塔），与 `fastvithd` 不是同一个视觉塔，其文本塔**与本项目视觉塔不共享嵌入空间**。本机 `/home/bc/data/models/` 下亦无任何 mobileclip 权重（`find` 实测 0 命中）。⇒ 组 A **不能**用 CLIP 余弦口径，改用 `emb` logit lens（§三 U2） |

**未在线核实的部分**：三个 self-self 算子（CSA / kkᵀ+高斯 / qqᵀ）的定义**不重新在线核实**，
直接引用 RO-1 已一手核实并落盘的 `clip_naclip/VENDOR.md`（含三个上游仓库 commit 与逐行行号）
—— 这属于"本仓库已核实事实"，不是新的外部断言。

### 1.3 本机实测核实（`config/verify_env.py` → `config/verify_env.json`）

| # | 待核实 | 实测结果 |
|---|---|---|
| V1 | FastViTHD stage 结构 | `network` = [Seq(2×RepMixer), PatchEmbed, Seq(12×RepMixer), PatchEmbed, Seq(24×RepMixer), PatchEmbed, RepCPE, **Seq(4×AttentionBlock)**, PatchEmbed, RepCPE, **Seq(2×AttentionBlock)**]；无 cls/register ✅ |
| V2 | attention stage 的网格与头数 | `network[7][*]`: (1,768,32,32)、24 头；`network[10][*]`: (1,1536,16,16)、48 头、head_dim 32；视觉特征 `(1,256,3072)` ✅ |
| V3 | 视觉塔权重确实来自 checkpoint（不是随机初始化） | `model.vision_tower.vision_tower.model.network.10.1.token_mixer.qkv.weight` 在 safetensors 中存在，与运行时张量 **max\|Δ\| = 0.0**；视觉塔键 629 个；`head.proj [3072,768]` 存在 ✅ **组 A 不是在读随机权重** |
| V4 | eager + 逐头捕获 | `_attn_implementation='eager'`；24 层/14 头/2 KV 头；捕获形状 `pre/post/kk/qq (24,14,16,16)`、`kimg (24,2,256,64)`、`hnorm (24,256)` ✅ |
| V5 | post-softmax 行内 image 段占比 | 中位 **0.136**（min 1.7e-18，max 0.997）—— GL token 的注意力**86% 落在非 image 位置**（sink/文本），这本身就是"pre-softmax 直接平均会被 sink 主导"的旁证 |
| V7 | `VisionSurgery` 的正确性 | `attn=vanilla, arch=vanilla` **严格 no-op**（max\|Δ\|=0.0）；**手写 `_block_forward`+`_mhsa_forward` 复刻原 `AttentionBlock.forward` 逐元素完全相同（max\|Δ\|=0.0）** ⇒ 手术代码本身不引入任何偏差；`clearclip/reduced` 确实改变特征（max\|Δ\|=11.9） |
| V8 | emb 读出可用 | 目标词概率场 [2e-22, 0.956]，`calculator` 场 [4e-21, 1.3e-5]（量级差 5 个数量级，与 RO-2 的词特异性一致）；峰值显存 **2.48 GB** |
| V-speed | **卡 0 当前吞吐** | `config/probe_speed.json`：单样本生成 **78–93 s**（3.4–4.0 tok/s）；G1 在空闲卡上是 9.4 s（≈34 tok/s）⇒ **约 9× 的他人作业竞争**。`<retouch_light>` 首现位置 = 生成序列的 **倒数第 4 个 token**（311/315、309/313、304/308…）⇒ 提前停止省不下时间 |
| V-batch | **批量生成 vs 单样本生成的等价性** | `/var/cache/veradata/ro9b_stacks_20260803/batch_equivalence.json`：4/4 **不逐 token 相同**（长度 316/315、271/313、304/308、…）。bf16 下 greedy 近似并列被 batch 维与左 pad 改变了。处置见 §三 U1 |

---

## 二、方法学决策（自行核实后采用，非拍板）

### 2.1 与 RO-3 的分工与复用（任务卡 ⚑ 条）

去看了 `experiments/RO3_layerhead_scan_20260803/`：逐头栈已落盘
`/var/cache/veradata/ro3_stacks_20260803/`（1709 个 npz，`pre_{instr,last,alltxt,gl}` /
`post_*`，形状 `(24,14,16,16)` float32）。

**结论：不能直接复用它的 `gl` 池当 RO-9 的读出点，必须重跑前向。** 理由是 RO-3 自己
NOTES §三 U3 写明的：RO-3 的 `gl` 是 **prompt 末尾追加**的 `<retouch_light>`（prefill-only），
而 RO-9 读的是**生成出来的 plan 里**首现的那个 GL token（实测在生成序列倒数第 4 位）；
两者在 local 段逐层 ρ 只有 **0.3–0.9**。本臂的第一件事是"复刻 RO-9 的 0.648 再往上修"，
读错位置就没有可比性。**复用的是它的方法学**（5 折 source-level CV、权重只在 fit 折估、
逐头 z-score 用数据集级统计量而非逐图统计量）与**它的结论边界**（head-mean 会把单头信号平均掉）。

分工按任务卡：RO-3 回答"哪个头有指令依赖"（逐头差分场 AUC_target + max-stat 置换），
本臂回答"怎么聚合头能提高定位质量"。两边的判据量（AUC_target，阈值 0.65/0.30）刻意保持同一口径。

### 2.2 四条实验轴的落地

| 轴 | 落地 |
|---|---|
| ① 层选择 | 导出保留全 24 层；层选择 5 档：`canon`(L8–15，RO-9 原样) / `best1`(fit 折最优单层) / `band`(fit 折最优**连续**层段，300 个候选段全扫) / `wsum`(权重 ∝ relu(层 AUC_target−0.5)，fit 折估) / `linfit`(**学习式层加权** = fit 折格级岭 logistic，只学 24 个层权重，不引入任何空间/邻域特征) |
| ② head 聚合 | 导出保留 head 维；5 档：`mean`(RO-9 原样) / `zmean`(逐头**数据集级** z-score 后平均) / `aucw`(权重 ∝ 逐头 AUC_target−0.5，带符号) / `top3` / `top5`(逐层 top-k 头凸组合)。另有 `post` interaction = **逐头 softmax 后**再聚合（量纲统一到概率）——它作为 ③ 的一档并排 |
| ③ self-self | **组 A**：视觉塔 FastViTHD 末层 MHSA 手术 13 臂（见 `export_vision.py::ARMS`），读出用 `emb` logit lens；**组 B**：LM 侧 GL→image 的 `pre`(q-k, 原样) / `post`(整行 softmax 后切 image 列) / `kk`(k_GL·k_img) / `qq`(q_GL·q_img) 四种交互并排；另有**桥接臂**：视觉塔手术 + LM 读出（teacher-force 同一段 `gen_ids`，唯一变量 = 视觉特征） |
| ④ 后处理 | `none` / `a1`(D-0 高范数 outlier 剔除+4 邻域插值，RO-9 原样) / `sink`(**attention-sink 版 outlier**：跨全部层/头平均的 post-softmax 权重的高值 outlier) / `aff`(**self-self 亲和度传播**：`A = rownorm softmax(K_img K_imgᵀ/√d)`，`s ← A s`) |

### 2.5 两个档位预算（跑数后追加说明，规则本身在 `supp_waterfall.py` 顶部）

主瀑布是**贪心**的，AUC 准则下 S1 一步就把 `linfit`（24 个层权重的 fit 折岭 logistic）选出来，
AUC 从 0.661 直接到 0.935，后三段全 +0.000 —— 那条路径回答"信息在不在张量里"，
但它**用 GT 掩膜拟合了 24 个连续参数**，与 RO-1 的零训练读出不可直接横比。
故追加 **受限瀑布**（`supp_waterfall.py`，`metrics_supp_restricted.json`）：
层选择只允许 **0–1 个自由参数**（canon / 最优单层 / 最优连续层段），其余三段照旧。
受限档四段各自都有贡献（+0.179 / +0.029 / +0.026 / +0.026 → **0.9200**），
**REPORT 的主叙事用受限档**。

### 2.3 组 A 的读出点为什么是 `emb` 而不是 CLIP 余弦

- **不是选择偏好，是可行性**：W6 核实的结果是本项目视觉塔（`fastvithd`）**没有公开可对齐的文本塔**。
  强行拿 MobileCLIP2-B/S 的文本塔与 `head.proj` 相乘会是把两个不共享嵌入空间的向量点积，
  得到的数字没有意义。
- `emb` 是 RO-2 已经**实测过词特异性**的读出点（GT 掩膜集 AUC 0.829 vs `calculator` 0.523，
  Δ=+0.215），而且它就是**真正流进 LLM 的那个量**，回答"where 在不在 VLM 里"比 CLIP 余弦更直接。
- 代价：`emb` 的绝对 AUC 与 RO-1 的 CLIP 余弦**不是同一个刻度**，两者的绝对值不可直接横比；
  可比的是**同一读出点内 vanilla vs 手术后的配对增量**，以及**与无关对照词 `calculator` 的差**。
  这条限定写进 REPORT。

### 2.4 数据纪律

- 样本与指令 **100% 复用**冻结的 `G1/config/g1_region_opp.json`（214 源，S-val），**零重新采样**。
- shuffle donor：按 `direction` 分组内确定性平移（`build_jobs.py`），**方向词匹配**、
  保证 `donor != self`；实测 15.4% 的 donor 与本源同 `target_name`（RO-2 §三.b 指出 RO-1 的
  无约束 derangement 有 20.1% 同词，本臂略好但同量级，属**保守方向**：同词 donor 会压低 Δ_shuffle）。
- 5 折 CV 是**臂内拟合/评测切分**，不是数据集切分；全部 214 源都是 S-val，未触碰 S/P split 纪律。

---

## 三、待主 agent 决策

> 按协议，下列属"两种做法都合理、影响后续"的决策项；已采用**保守默认**继续，未静默拍板。

- **U1（批量生成 vs 单样本生成）**。卡 0 被他人作业占满，单样本生成 85 s × 642 作业 = **15 h**
  不可行。批量生成（gen-batch=8）把它压到约 1.5 h，但实测 **4/4 不逐 token 相同**
  （bf16 greedy 近似并列被 batch 维/左 pad 改变）。
  **保守默认 = 用批量生成，并把风险转成可测量的数字。事后实测（`metrics.json →
  ro9_published_reference`）**：
  ① 直接读 G1 落盘的 RO-9 单样本生成场，在同一批 212 源上复算 = **AUC 0.6462 / AUC_target
  0.5038**（已发表 0.648 / 0.504），本臂 S0 = 0.6607 / 0.5083 ⇒ **净影响 +0.0145 AUC**；
  ② **25.5% 的样本批量生成给出逐 token 相同的 plan**，这些样本上本管线与 RO-9 落盘的场
  **ρ = 1.000000（逐比特相同）** —— 这既解决了"管线是否正确"，也把"plan 措辞不同"隔离成
  唯一残差；其余样本 canonical 场 ρ 中位 **0.927**、p10 **0.749**。
  ③ 因此**不再另跑单样本对照子集**（①②已给出更强的量化）。
  若主 agent 认为必须逐 token 复刻，唯一办法是排空卡或等空闲窗口重跑 15 h（本臂已把
  `--gen-batch 1` 保留为开关，重跑不需改代码）。
- **U2（组 A 的读出口径）**：`emb` logit lens（保守默认，理由见 §2.3）vs 训练一个 1×1 探针
  vs 引入外部文本塔。后两者都超出"零训练读出"的定义或引入未对齐的外部权重。
  若主 agent 要与 RO-1 的 0.941 做**绝对值**横比，必须先决定一个共同刻度——本臂做不到。
- **U3（A3 test-time registers 在 FastViTHD 上不可直接移植）**：官方 A3 要往 token 序列
  追加寄存器 token 并把特定 MLP 神经元的激活搬过去；FastViTHD 的 stage 之间是**卷积**
  （RepCPE / ConvFFN / PatchEmbed 全是空间卷积），token 维只在 MHSA 内部临时存在，
  **追加的寄存器出了 MHSA 就无处安放**。保守默认 = **不做官方 A3**，改报两个可移植的替代：
  ④ 的 `sink`（attention-sink 版 outlier 剔除）与 `aff`（K 自相似传播）。
  若主 agent 认为必须有 A3，可行的最小版本是"只在末个 MHSA 内部追加 N 个零初始化 token、
  出块即丢"（不含官方的神经元定位步），属另一条工具线。
- **U4（`region_b_kind=spatial` 的 49 源）**：主判据按 RO-9/G1 惯例取 `background` 子集
  （n=165，reg_b 字面 = 主体补集）；`all`（214）一并报。若主 agent 要把 spatial 也纳入主判据，
  需先解决"左半/右半"这类非实体指称的可定位性问题（RO-1 §6.2 登记的同一个坑）。
- **U5（scache 的归一化域）**：本臂落盘用**全局仿射**（整臂两个常量 lo/hi = 全体源全体格的
  1%/99% 分位）映射到 `[0,1]` 并 clip，`meta.norm.domain=[0,1]` 显式写出，`invert` 字段给反算式。
  这样消费端 `tools/scache/upsample.py` 的默认 `clamp=(0,1)` 不会静默把场清零
  （`ro9` 臂的已知埋雷）。**代价是 clip 掉了 2% 的极值**；若下游要无损原值，应改成
  `domain=null` 并要求消费端显式传 `clamp=None`——这需要主 agent 在 scache 约定层面定一次。

---

## 四、与任务卡的出入（如实登记）

1. **任务卡说"RO-3 的逐头栈若已落盘，直接复用不要重跑前向"**——实际**不能**复用（§2.1），
   已重跑 LM 侧前向。复用的是方法学与样本/指令清单（零重新采样）。
2. **任务卡说组 A 用"RO-1 的三个算子原样搬到自家视觉塔"读出 patch 级空间场**——算子确实原样搬了，
   但**读出的 query 侧换成了 `emb` logit lens**，因为自家视觉塔没有可对齐的文本塔（W6、§2.3）。
3. **官方 A3（test-time registers）在 FastViTHD 上不可移植**（U3），用两个可移植替代顶上。

---

## 五、红线自查（对照 CLAUDE.md 速查表）

| 红线 | 本实验状态 |
|---|---|
| attention 导出必须 eager（FA2/SDPA 返回 None **不回退**） | ✅ `load_model` 加载即断言 `_attn_implementation=='eager'`；每样本断言 `fwd.attentions[0] is not None`；RO-3 已在同一环境实测 sdpa 确实返回 None |
| s 禁逐图 min-max/softmax 归一化 | ✅ 落盘全是原始 logit / 原始概率；逐头与逐层 z-score 的 μ/σ 在 **fit 折的全部图全部格**上估（数据集级）；scache 用**整臂两个常量**的全局仿射。`post` 档的 softmax 是**注意力行内**的 softmax（模型本身的算子），不是空间维归一化 |
| 每个消融行必带 Δ_const / Δ_shuffle 列 | ✅ `headline()` 每行都有 `delta_const`（= median AUC − 0.5）、`delta_shuffle`（配对 Wilcoxon + 胜率）、`delta_luma` |
| VLM 干预对象 = 整段 image tokens 非 last token | ✅ 读出对象是 GL token → **整段 256 个 image token**；组 A 的手术作用在视觉塔全部 256 个 token 上 |
| IoU 禁当优化目标 | ✅ 全程未出现 IoU；拟合准则是 fit 折的 AUC_target |
| checkpoint 选择禁用 val loss | n/a（零训练；`linfit` 的层权重用 fit 折格级 logistic，选择准则是 fit 折 AUC_target，报的是 hold-out） |
| 逐像素算子禁 (x,y)/邻域/MLP/排序 | ✅ 该红线约束的是**渲染器的逐像素算子**；本臂不训练渲染器。④ 的 `a1`/`sink` 用了 4 邻域插值，那是 **D-0 伪影修复**（PLAN §2.2 明列的三件套之一），不是渲染算子 |
| 全局仿射 G 初始化 / σ 参数化 / s 轴平滑正则 | n/a |

---

## 六、复现

```bash
# 1) 实测核实（必跑，产出 config/verify_env.json）
CUDA_VISIBLE_DEVICES=0 .venv-lens/bin/python \
  experiments/RO9b_readout_fix_20260803/config/verify_env.py

# 2) 作业清单（零重新采样，全部读 G1/RO-2 冻结 config）
.venv-lens/bin/python experiments/RO9b_readout_fix_20260803/config/build_jobs.py

# 3) 组 A：视觉塔 self-self 13 臂（GPU，prefill-only）
CUDA_VISIBLE_DEVICES=0 .venv-lens/bin/python \
  experiments/RO9b_readout_fix_20260803/export_vision.py --mem-frac 0.06 --skip-existing

# 4) LM 侧逐头栈（GPU，642 作业；卡空闲时可用 --gen-batch 1 逐样本复刻 RO-9）
for i in 0 1 2; do CUDA_VISIBLE_DEVICES=0 .venv-lens/bin/python \
  experiments/RO9b_readout_fix_20260803/export_lm.py --shard $i/3 \
  --gen-batch 8 --mem-frac 0.045 --skip-existing & done

# 5) 桥接臂：视觉塔手术 + LM 读出（teacher-force 同一段 gen_ids）
CUDA_VISIBLE_DEVICES=0 .venv-lens/bin/python \
  experiments/RO9b_readout_fix_20260803/export_lm.py \
  --vis-attn clearclip --vis-arch reduced --suffix .cc \
  --gen-cache-dir /var/cache/veradata/ro9b_stacks_20260803 \
  --out-dir /var/cache/veradata/ro9b_stacks_cc_20260803 --itags reg_a,reg_b,shuf

# 6) 分析 + 瀑布 + 组 A（纯 CPU）
OMP_NUM_THREADS=4 .venv-lens/bin/python \
  experiments/RO9b_readout_fix_20260803/analyze_ro9b.py

# 7) viz + scache
.venv-lens/bin/python experiments/RO9b_readout_fix_20260803/make_viz_scache.py
```
