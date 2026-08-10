# Where 头重设计 · 需求与资源盘点（2026-08-10）

> 背景一句话：现方案（MetaCanvas query → cross-attn(H_where, F_pre) → 全局 (w, ρ)）在 159k 全量训练下失败
> （W01/W02 终局 mIoU 0.557/0.562、vs oracle 57.9%/58.2%、gate 4/10 全 WHERE-GATE-FAILED、step 3500 起平台），
> 用户裁定设计失败，W03-W08 容量变体已取消（**注意：容量未被证否——MC16 六臂从未跑过**；判定依据是下述机制
> 证据）。正式失败措辞是「**机制不匹配**」而非「容量到顶」；结果审阅确认的开放替代解释（未扫过）：canvas
> 容量、lr/clip（grad_norm 稳态 2.1-5.5 而 clip=1.0，全程在裁剪区）、H_where 只取最后一层（与发现三冲突）。
> 完整终局审阅见 `experiments/.../where_b/REVIEW-result-W1.md`。本文只写**要实现什么**与**手上有什么**。

---

## 一、要实现的功能（需求）

### 1.1 核心功能

**输入**（推理时只有这两样，红线）：
- `I_in`（原图，短边 512 / 等比 / 32 对齐 / 长边 ≤2048）
- `instruction`（用户编辑指令文本）
- 允许使用 SFT VLM 前向的一切中间产物：`<where>` 推理文本（自回归生成）、任意层 hidden、任意 attention、F_pre 视觉特征
- **禁止**：`I_tar`、GT mask、GT LUT、oracle latent 进入推理输入

**输出**：
- 软掩膜 `m_pred(p) ∈ [0,1]`，交付分辨率 = 原图尺寸，指定「这条指令要编辑的语义区域」
- （供 Stage-What 消费）掩膜之外至少还需：`F_roi/F_bg`（m 加权的 F_pre 池化，现成公式）；可选一个全局空间 latent（原 z_where 位）

### 1.2 质量目标（预注册，V_where generated-context 主榜）

| 指标 | 目标 | 现方案实测（参照系） |
|---|---|---|
| local soft-IoU (min/max) 中位 | ≥ 0.75（协议 §5.6 gate） | 0.557/0.562（终局） |
| 相对 oracle 上界比值 | ≥ 85% | 57.9%/58.2%（终局） |
| p10 | ≥ 0.55 | 0.236/0.297（终局） |
| grid 边界 F1 / oracle | ≥ 75% | **32.1%/33.1%**（形状是最硬的失败证据） |
| 指令条件性 | 同图不同指令产生不同区域（配对 Δ>0）；shuffle 降幅 ≥0.20 | shuffle 降幅仅 0.145/0.126；见 §1.4 精确画像 |
| global 样本（全图编辑） | soft-IoU ≥ 0.98 | 可达（全 1 mask 平凡解） |

日常监控主读数已简化为一个：**soft_iou_vs_oracle**（预测 vs oracle mask 的 min/max soft-IoU，直接值）。
完整板（七 context/负控制/中心先验/分层）只在选型时离线跑一次。

### 1.3 训练约束

- 监督信号：GT mask（`.cgt.png` 软边掩膜，local train 75,544 条）；oracle latents（可选辅助）
- VLM 冻结边界：现协议冻结整个 SFT VLM。**若新方案需微调 VLM 任何部分（如 LISA 式 seg-token），须先经用户批准**——这是唯一需要用户拍板的边界
- 单卡一臂、effective batch 32、1 epoch 量级；bf16；无人值守可跑（接入现有队列）
- 评测按 WEVAL 分层工具离线归因（面积/位置/边界复杂度/软边等六维 + 长尾机制归因）

### 1.4 失败教训（新方案必须回应的三个问题）

1. **间接映射学不出来**：文本 hidden →（query 读出）→ 71 维全局线性方向 w 的映射，在有 oracle 蒸馏监督
   （L_s/L_curve/L_dir）的情况下仍然崩塌——监督不缺，是表征/机制不匹配。新方案要说明它的「文本→空间」通路为何更直接。
2. **指令只承载了尺度，没有承载位置/形状**（终局审阅的精确画像，取代早先「过覆盖铺全局」的 step1500 暂态描述）：
   - pred/GT 面积 Spearman 0.665/0.691（真实指令）→ 0.19/0.15（同图换指令）→ ≈0（无语言，场退化近常数）——
     模型从指令读出了「区域多大」，**扣掉面积后指令条件性 ≈ 0**（面积均衡的中心先验校准配对 Δ = +0.003 p=0.46 / +0.011 p=0.03）；
   - 面积**向数据集均值收缩**（小目标过覆盖 2.55×、大目标欠覆盖 0.99×、整体面积比中位 1.22）；`L_mask` 的
     `0.25×balanced_BCE`（正类权 ∝1/面积）本身为小目标过预测付钱——**loss 侧解释未被排除**，新方案的 loss 设计要正面处理；
   - 形状：grid 边界 F1 停在 oracle 的 32-33%，末段零净进展；归一化后模型只补上「中心先验→oracle」差距的
     6.7-10.1%（覆盖）/ 15.1-16.6%（形状）；**21.5%/17.5% 的样本不如全 1 平凡掩膜**；
   - antonym 不变性 PASS（场不偷读颜色方向词）——错的不是读了不该读的，是该读的（主体/位置）没读出来。
3. **历史坍缩**：直接空间 logits（旧 MCQ 路线）也塌过。「换个头」不等于解决——需要能定位指令条件信号
   实际存在于模型的哪个部位（历史证据见 §2.4）。

---

## 二、现有资源（全部可复用）

### 2.1 模型资产

| 资产 | 位置 | 质量证据 |
|---|---|---|
| Base SFT checkpoint-4976 | `/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976`（NFS 备份已同步） | 两段式结构率 64/64 全满分；`<where>` 文本 token-F1 p50=1.0、53% 样本与 GT 逐字一致——**语言侧把区域"说"对了** |
| 同 checkpoint-2488（0.5 epoch） | 同目录 | 备用对照 |
| 原始基模 Qwen3-VL-4B-Instruct | `/home/bc/data/models/Qwen3-VL-4B-Instruct` | |

### 2.2 数据资产（全部 indexed tar shards，读走 /mnt/nfs-ro）

| 资产 | 规模 | 位置 |
|---|---|---|
| sft2seg 训练/评测数据 | train 159,215（local 75,544）+ V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433 | `/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/`；**本地缓存 27.5 GiB 已预热**（`/home/bc/data/shard_cache`） |
| GT maskviews（低分辨率+交付分辨率） | 五 split 全量 | `where_a-20260805/maskviews/` |
| oracle latents（w*, ρ*, s*, r*(z) 257 点, std(s*), cband_normalization） | 五 split 全量（train 75,544×2 readout） | `where_a-20260805/oracle/BA-3-Joint/s5/` |
| genctx two_segment（`<where>`+`<color>` 生成 token ids） | train+V_where+V_what 全量，零截断零格式失败 | `where_b-20260805/genwhere/` |
| genctx forced_color | train+V_what | 同上 `-forced_color` |
| V_where 掩膜几何分类缓存 | 400 条六维几何标签 | `experiments/.../where_b/geometry_*.json` |

### 2.3 代码资产（q3vl 五包，1000+ 测试，8 轮审阅闭环）

| 资产 | 说明 |
|---|---|
| F_pre 提取管线 | `q3vl/where/fpre.py`：merger 前 H/16×W/16×1024，hook 式，真实宽高比 |
| Phi-71 basis + guided upsample | oracle 上界 0.97（Where-A 实证）；`radius=1, eps=1e-2` 终值；s 域契约与越域断言齐全 |
| 两种 readout（Band/CBand12） | 公式经逐符号审阅；CBand12 logsumexp 抗塌陷 |
| 评测器 | `--online-eval quick`（vs-oracle 主读数）/ `--final-eval full`（§5.6 完整板）；per-sample 落盘 |
| WEVAL 分层归因工具 | `q3vl/whereb/analysis/`：六维分类 mIoU + 长尾机制归因（area_mismatch/s_collapse/oracle_ceiling/...） |
| IO 层 | 本地 shard cache + 有序 prefetch + fd 池：数据供给 0.063 s/step（35×），带「不改变任何权重」等价性测试 |
| context 数据流 | teacher/generated 50/50、forced-prefix、七种 context 构造（负控制备用） |
| 训练循环 | 单卡臂、D-20、eval 防御（eval 崩溃不杀训练）、checkpoint 保护 |

### 2.4 已确立的机制发现（新方案设计的信号地图）

以下发现均有归档实验背书，是「指令条件的空间信号到底在模型哪里、怎么读才读得出来」的已知答案。
历史 AUC 口径的结论一律**限于排序**（AUC 已全面禁用为判据，见 CLAUDE.md 红线）。

**发现一：定位信息在 attention 里涌现，但被 padding 格与 attention sink 掩埋——去掉它们才看得见。**
- `expand2square` 的 pad 格只占 16×16 网格约 5.3 格，却吃掉 **53-74% 的注意力质量**，**93% 的源 argmax 落在 pad 里**（RO-9c 补件）；逐图 min-max 的分母被 pad 支配，导致原始场"看起来只有一个 sink"，而实际**有效区内主体信号清晰存在**——此前"必须做共模消除才能解锁 grounding"的整条结论就是这个假象造成的。
- 排除 pad/sink + 修层 + head 聚合 + self-self 算子后，主体定位从 0.661 提到 **0.92-0.935**（RO9b，base 模型）。
- 推论：任何 attention 读出路线，**pad/sink 排除是第一前提**，不是后处理选项。
- 位置：`experiments/_archive/2026-08-10/RO9b_readout_fix_20260803/`、CLAUDE.md「空间场可视化纪律」节。

**发现二：指令条件信号存在于特定层/头的差分 attention，不在 head 均值。**
- 单头 **L11H5 差分场**对 `.cgt` 有信号、零对比度对照 0.522；OOF 逐层头融合有指令条件定位（RO-3，AUC 口径限排序）。
- G1（旧失败）的主因被证明是 **query token 选择与 head-mean 读法错误**，不是信号不存在。
- **未测的关键变量**：这些全是 base 模型上的结论；SFT 已全参训练 merger+LLM，`<where>` 段 hidden 被显式塑形过（文本 F1=1.0），SFT 后模型的 `<where>` token→image attention 大概率显著更强——一个半天级探针即可测。
- 位置：`.../RO3_layerhead_scan_20260803/REPORT.md`。
- 附：**attention 导出必须 eager**（FA2/SDPA 返回 None，不回退——红线）。

**发现三：词-视觉对齐在进 LLM 之前最强，LLM 前向逐层抹掉词特异性。**
- emb 层（进 LLM 前）目标词对应区域可分（RO-2，限排序）；越往深层走词特异性越弱。
- 推论：「哪个词」的空间信息宜在浅层/入口处取，「指令整体语义」在深层 hidden——两者可能要在不同深度读。
- 位置：`.../RO2_logitlens_20260803/REPORT.md`。

**发现四：主体显著性先验极强，会冒充指令理解——负控制不可省。**
- 一句对全体样本相同的 "the main subject" 就能拿到高定位分（RO-1/RO-X1）；零参数中心先验场同样能赢过弱读出。
- 本次 W01/W02：step1500 时 center 类曾输给中心先验（Δ=−0.05），**终局收敛为打平**（−0.005 p=0.67 / +0.019 p=0.12）；
  整臂对中心先验的正 Δ 几乎全部来自偏心/小/复杂三类（那些类里中心先验基线只有 0.14-0.32）——「赢过中心先验」
  这个读数必须分层看，整臂聚合会高估模型。
- 推论：新方案验证必须自带 shuffle/固定短语/中心先验三件套（且分层报告），哪怕探针阶段。

**发现五：直接生成空间 logits 会坍缩（两次实证）。**
- 旧 MCQ 路线（语言模型直接出空间图）坍缩 + 边缘退化（`.../MCQ_*/`）；本次 MetaCanvas query→全局 w 读出同样崩塌但模式不同（过覆盖铺全局）。
- 两次失败夹出的空间：**信号要从模型内部"读"出来（attention/相似度/浅层特征），不是让模型"生成"出来**——这正是发现一、二、三共同指向的方向。

**发现六：本次 W01/W02 的失败画像（终局口径，逐样本归因在盘）。**
- 指令只承载尺度（详见 §1.4 第 2 条）；长尾（<0.3）终局缩到 65/43 个，area_mismatch primary W01 43.1% / W02 69.8%
  （step1500 时两臂都 ~70%——**训练后期 W01 的归因结构变了**，中期快照不能替代终局）；面积比中位终局 3.26/2.76（尾部）、
  1.22（整体，向均值收缩非铺全局）；oracle 天花板解释 ~12% 尾部、最差 10% 深尾升到 27.5%。
- gate 终局 4/10，通过的四行没有一行是正面证据（global 全 1 平凡解、GT/gen gap 小=两者一样差、中心先验 Δ 靠偏心类抬起）。
- 位置：`/home/bc/data/runs/where_b/W0{1,2}/` + `analysis_W0{1,2}_step1500/` + **`REVIEW-result-W1.md`（终局权威）**。

**其他已确立事实**：F_pre 对 SFT 不变（vision 全冻结，WA-P4b 实测 diff=0——attention/相似度类方案可复用全部 Where-A 资产）；灰度照片占 local 池 3.3-3.8%（S 通道恒零，读出要能容忍退化维）；单活跃基元解在 guided upsample 下脆弱（30.8% 样本，hi 档落差翻倍）。

### 2.5 算力与运维

- 2×H100 95GB；W01/W02 收尾后全空
- pueue 无人值守队列（`gpu-queue` skill、事件日志、产物 gate、systemd 自愈）；`q status` 一条命令全景
- 训练墙钟参照：IO 修复后单臂（159k 样本 1 epoch，冻结 VLM 前向 + 小头训练）约 3.5-4 h

---

## 三、交付期望

新方案的立项材料应包含：机制一句话、对 §1.4 三个问题的回答、最小验证探针实验（≤半天 GPU，
用 V_where + oracle + SFT checkpoint 快速证伪核心假设）、对协议的改动清单（特别是冻结边界）。
探针通过后再谈全量臂。
