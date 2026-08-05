# Qwen3-VL MetaCanvas Where/What 最终实验方案（2026-08-04）

> 状态：**DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED**  
> 版本：v1.0  
> 上游定档：[`QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md`](QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md)  
> 边界：本文冻结实验问题、模型接口、训练矩阵、Loss、选择协议和交付物；不表示代码、数据派生物、checkpoint 或实验结果已经产生。

---

## 0. 最终决策

完整路线固定为三个串行阶段：

```text
Base SFT
  I_in + instruction
      -> <where>...</where><color>...</color>

Stage-Where
  frozen Qwen3-VL + Q_where MetaCanvas + connector
      -> 全局 basis 参数 (w, rho)
      -> s(p) = Phi(I,p) @ w
      -> m(p) = R(s(p); rho)

Stage-What
  frozen Qwen3-VL + frozen Where + independent Q_color
      -> continuous z_style in R^1024
      -> 48-Gaussian LUT function T_pred
      -> standard 33^3 LUT

最终图像
  I_out(p) = I_in(p) + m_pred(p) * (T_pred(I_in(p)) - I_in(p))
```

最终网络只接收 `I_in + instruction`。`I_tar`、GT mask、GT LUT、oracle latent 都不能作为推理输入；它们只在各自阶段作为监督或评测真值。

本轮不再让 MetaCanvas 直接预测低分辨率 mask，也不让语言模型直接生成逐像素场。MetaCanvas 预测的是一组**全局的、可解释的 basis/readout 参数**；高分辨率边缘来自 merger 前视觉特征与解析 basis，而不是从 8x8 或 16x16 query logits 上采样。

Stage-What 固定比较两个生成器：

1. `WHAT-FG48`：每个样本完整生成 48 个 Gaussian 的全部参数；
2. `WHAT-SB48`：共享 Transformer backend 和跨样本共享的 Gaussian geometry，只生成每个样本的激活与颜色载荷。

两者都输出连续 LUT 函数，不做 3500/4000 类 LUT-ID 分类，也不查 LUT embedding 表。因此参数量不随 LUT 数量增长，结构上支持 4000 个及更多 LUT；是否能泛化必须由 LUT-ID-disjoint 测试证明，不能仅凭结构宣称。

---

## 1. 研究依据与不可迁移边界

### 1.1 MetaCanvas

MetaCanvas（arXiv:2512.11464）的可迁移结论是：冻结 MLLM，使用可学习的多维 canvas tokens，经 Transformer/DiT connector 读取语言与视觉状态，并用 zero-init residual 稳定接入下游生成器。

官方训练目标只有 diffusion/flow-matching loss，没有 mask、basis 或 LUT auxiliary loss。该 loss 与本任务的确定性空间读出和 LUT 函数回归不匹配，因此本方案只迁移**结构与冻结边界**，不迁移 flow-matching loss。

### 1.2 GLUT/CGLUT

GLUT/CGLUT（arXiv:2605.19889）支持两个关键设计：

- Full Generation 具有更高的逐风格表达容量；
- Shared Geometry/Backend 具有更强的跨风格连续性，但可能牺牲容量。

官方主损失由 RGB 重建、CIELab hue-chroma 和 opacity binary-entropy sparsity 构成，公开权重为 `1 / 10 / 0.001`。本方案保留这三类信号，但把 RGB 重建改为更直接的 LUT function supervision，并增加 style-code 防坍缩和 33^3 交付一致性。

### 1.3 本仓库已有证据

- `experiments/E2_basis_fit_20260803/` 已证明固定中心、归一化 Gaussian competition 可以形成平顶带通响应；单 Gaussian 对环形区域不足。
- `experiments/RDG_transformer_20260803/` 支持参数生成端使用 Transformer，而非小 MLP；标准 33^3 烘焙与生产四面体插值回读误差可以接近零。
- 旧 MCQ 直接空间 logits 路线出现明显坍缩和边缘退化，因此本方案将 MetaCanvas 从“空间图生成器”降为“全局 basis/readout 参数生成器”。
- 旧 RD-G 中任何 `(I_in, after=I_tar)` 输入都必须删除。`I_tar` 不得进入新模型条件。

### 1.4 明确不采用的 Loss

- 不用 MetaCanvas 的 flow-matching loss；任务域不同。
- 不用语言 causal-SFT loss训练 Where/What connector；SFT 已在上游独立完成。
- 不对 Gaussian LUT 机械叠加 direct-grid LUT 的 TV/monotonicity；Gaussian 的协方差、稀疏门控和 bake consistency 已提供结构约束，重复正则会压低局部颜色变换容量。
- 不把 CIEDE2000 作为高权重反向主损失；它用于 checkpoint 选择和报告，避免复杂分段梯度主导训练。
- 不使用离散 LUT 分类损失；`z_style` 是连续函数码，不是 LUT ID。

---

## 2. 数据、隔离 split 与存储契约

### 2.1 数据范围

所有实验使用相同的数据范围：

```text
global: g1, g2, g3, g4
local:  l1, l2, l3, l4, l5, l6
```

上游 SFT 原始 authority 为：

```text
train: 169,260
eval:    3,320
```

实际数量以完成图像、标签、LUT、mask、长度和 checksum 校验后的 terminal manifest 为准。各 arm 必须读取同一 manifest、同一数据顺序和同一采样种子，不得为某个 arm 单独清洗数据。

### 2.2 三个相互隔离的评测集合

现有 eval 集按稳定 `source_image_id`、`lut_id` 和 build 分组后，确定性划分为三个互斥子集，目标比例约为 `1:1:1`；保持完整 group，不为凑整数拆组：

| split | 用途 | 禁止用途 |
|---|---|---|
| `V_where` | 选择 basis calibration 与 8 个 Where 主臂 | 不选择 What checkpoint |
| `V_what` | 选择 8 个 What 主组合及控制臂 | 不回头选择 Where |
| `T_final` | 最终一次性报告、20 图可视化、置信区间 | 不参与任何模型/阈值选择 |

必须额外生成 `T_lut_unseen`：其中 `lut_id` 在 Base SFT、Where 和 What 的训练 manifest 中均未出现。若现有 split 无法满足这一点，则报告只能写“held-out sample”，不能写“unseen LUT generalization”。在 Base SFT 尚未启动的前提下，应优先通过 group-level reserve 保住该集合。

所有拆分同时满足：

- source image 不跨 train/select/test；
- 同一 LUT identity 不跨 `T_lut_unseen` 与任何训练集；
- 近重复图像、同一 Lightroom recipe 派生物不跨集合；
- 按 g1-g4、l1-l6、mask 面积和指令类型分层报告；
- split manifest 和 digest 对所有 arm 固定。

### 2.3 indexed-tar-shard 契约

新产生的 oracle latent、basis 元数据、连续 LUT code 或可视化索引也是数据集派生物，必须遵守冷热分层与 indexed tar shard 契约：

- durable 数据写入明确配置的 NFS 根；本地只保留容量有上限、可重建的 scratch/cache；
- 同一样本的图像、mask、LUT、oracle 参数和结构化标签使用 stable sample ID，并尽量位于同一 shard；
- shard 使用不压缩 tar，目标 1--4 GiB；不使用整 shard gzip；
- index 至少记录 `shard/member/offset/length/size/checksum`、schema version、shard digest 和样本计数；
- 临时 shard 完成 fsync、边界、计数、checksum 和随机读取验证后才能原子发布；
- resume authority 只来自 terminal manifest 与 shard index；不依赖 mtime 或全量小文件目录。

若不持久化大体积 `F_pre`，训练时直接复用冻结 Qwen3-VL 的同一次视觉前向；不得再额外运行 CLIP/DenseCLIP。任何全量 feature cache 若需要发布，也必须是 indexed shards，而不是百万级 `.npy` 小文件。

---

## 3. 三阶段参数边界

| 阶段 | 冻结 | 训练 |
|---|---|---|
| Base SFT | 24 个 Vision blocks | 主 merger、3 个 deepstack mergers、Language、4 个 special tokens |
| Where-A basis calibration | 整个 SFT VLM | shared basis projector `B` 与离线逐图 oracle latent |
| Where-B MetaCanvas | 整个 SFT VLM、已定档 `B` | `Q_where`、Where connector、readout/output heads |
| What | 整个 SFT VLM、完整 Where checkpoint | 独立 `Q_color`、style connector、Transformer backend、48 heads；SB48 另训练共享 geometry |

Where 和 What 不共享 query bank、attention pool 或输出头。What 不向 Where 反传，Where 不向 VLM 反传。这样可以直接检验此前颜色崩塌是否来自空间/颜色 query 的梯度冲突。

---

## 4. Stage-Where-A：校准高分辨率 basis

### 4.1 merger 前特征

从 Qwen3-VL 最后一个 Vision block、主 merger 之前读取：

```text
F_pre in R^[B x H/16 x W/16 x 1024]
```

图像继续采用短边 512、保持比例、32 对齐、长边不超过 2048 的上游契约。二维坐标按每张图真实宽高比生成，不将图像压成 512x512。

`F_pre` 是主 VLM 视觉前向的中间结果，不是额外的 DenseCLIP 前向。因此新图像确实要计算 basis，但该计算复用本来就必须执行的 Qwen3-VL vision tower；额外成本只有 `1024 -> 64` 投影、残差化和一次 guided upsample。

### 4.2 Phi-64 定义

```text
geo5(p) = [x, y, P2(x), P2(y), x*y]
range(p) = [L, S]
semantic_low(p) = B(F_pre(p)),  B: 1024 -> 64
```

`L,S` 按逐图固定统计口径标准化。64 个 semantic 通道逐图对 `[1, geo5, L, S]` 做最小二乘残差化，再做零均值/单位方差标准化，避免 semantic projector 偷学坐标、亮度或饱和度。

最终方向特征为：

```text
phi_dir(p) = [geo5(p), L(p), S(p), semantic_1..64(p)] in R^71
w_dir = normalize(w_raw)
alpha = softplus(alpha_raw)
s_low(p) = 3 * tanh((w0 + alpha * <phi_dir(p), w_dir>) / 3)
```

`w_dir` 的符号按“绝对值最大系数为正”确定，消除 `w -> -w` 的非辨识性。组合完成后只对标量 `s_low` 做一次 edge-aware guided upsample，得到原图比例下的 `s(p)`；不能先上采样 64 个通道再组合。

因此边缘分辨率由 `F_pre` 的 `H/16 x W/16` 空间网格和原图 guidance 决定，而不是由 MetaCanvas 的 8x8/16x16 query 数决定。

### 4.3 两种 readout

`R-Band` 使用可学习平顶 bandpass 与 polarity：

```text
b(z) = sigmoid(k * (z - mu + h)) - sigmoid(k * (z - mu - h))
m(z) = pi * b(z) + (1 - pi) * (1 - b(z))
```

其中 `h>0`、`k in [1,40]`、`pi=sigmoid(pi_raw)`。

`R-CBand12` 使用 E2 已验证的固定中心归一化 Gaussian competition：

```text
mu_i = linspace(-3, 3, 12)                      # 固定
sigma_i in [0.025, 0.30]
g_i(z) = o_i * exp(-0.5 * ((z-mu_i)/sigma_i)^2)
m(z) = sum_i c_i*g_i(z) / (sum_i g_i(z) + eps)
o_i,c_i in (0,1)
```

固定中心不是低分辨率 mask；它是沿一维连续 `s` 轴的 readout。空间细节仍在 `s(p)` 中。

### 4.4 basis calibration 臂

Where-A 只使用 local l1-l6 的 GT mask，逐图用多起点 L-BFGS 拟合 oracle `w*,rho*`，同时校准共享 `B`。全量运行以下四个归因臂：

| ID | projector 校准目标 | 作用 |
|---|---|---|
| `BA-0-Fixed` | seeded orthogonal `1024->64`，不训练 | no-calibration 控制 |
| `BA-1-Band` | 只由 `R-Band` oracle 反传 | 单 readout 控制 |
| `BA-2-CBand12` | 只由 `R-CBand12` oracle 反传 | 单 readout 控制 |
| `BA-3-Joint` | Band/CBand 独立 oracle latent，共享 `B` 联合校准 | 预注册主方案 |

校准目标是 oracle mask 表达力，不使用 instruction。`BA-3-Joint` 是后续 8 个 Where 主臂的固定 projector，不根据 Where-B 结果事后切换；其他三臂用于回答共享 projector 的收益来自哪里。

校准结束后固定 `B`，分别用多起点 L-BFGS 重拟合 Band/CBand 最终 `w*,rho*,s*,r*(z)`。逐图 oracle 参数只作为监督和 ceiling，推理时不存在。

---

## 5. Stage-Where-B：MetaCanvas 预测全局参数

### 5.1 Query 输入与 connector

`Q_where` 是带可学习 2D 位置的 MetaCanvas query，只读取：

- `<where>...</where>` 全部 token hidden，记为 `H_where`；
- merger 前 `F_pre`；
- 真实宽高比下的二维位置编码。

它不读取 `<color>` hidden，也不读取 `I_tar`。Connector 宽度固定为 512，使用 6 个 pre-norm Transformer blocks、8 heads、FFN 2048：

```text
self-attn(Q)
 -> cross-attn(Q, H_where)
 -> cross-attn(Q, F_pre)
 -> FFN
```

各 cross-attention residual gate 采用 zero initialization；`H_where` 和 `F_pre` 分别线性投影到 512。输出头只产生全局 `w0,w_dir,alpha,rho`，不产生 dense logits。

### 5.2 四种 MetaCanvas 结构

| ID | Query 布局 | 参数读出 |
|---|---:|---|
| `MC8-Joint` | 8x8，64 queries | 同一个 attention pool 和 joint head 预测 `w,rho` |
| `MC16-Joint` | 16x16，256 queries | 同上，用于验证 canvas 分辨率/容量 |
| `MC16-SplitHead` | 16x16，shared canvas | `w` 与 `rho` 使用独立 attention pool 和独立 head |
| `MC16-DualCanvas` | 两套独立 16x16 query bank/connector stream | 一套预测 `w`，一套预测 `rho`，只共享 frozen VLM |

`SplitHead` 检验冲突是否只发生在输出头；`DualCanvas` 检验是否需要从 query memory 开始完全解耦。根据用户定档，本轮不再跑 MetaQuery baseline。

### 5.3 8 个全量主臂

每个 MetaCanvas 结构与两个 readout 做笛卡尔积，不做短跑筛选或中途淘汰：

| Arm | MetaCanvas | Readout |
|---|---|---|
| `W01` | MC8-Joint | R-Band |
| `W02` | MC8-Joint | R-CBand12 |
| `W03` | MC16-Joint | R-Band |
| `W04` | MC16-Joint | R-CBand12 |
| `W05` | MC16-SplitHead | R-Band |
| `W06` | MC16-SplitHead | R-CBand12 |
| `W07` | MC16-DualCanvas | R-Band |
| `W08` | MC16-DualCanvas | R-CBand12 |

### 5.4 context 训练与评测

训练 batch 固定 50% teacher context、50% generated context：

- teacher：GT `<where>` token hidden；
- generated：Base SFT 自回归生成的 `<where>` token hidden。

generated 样本缺失闭合标签时不得回退到 GT；按固定最大 token 边界截取并记录格式失败。模型选择以 generated context 为主。

每个 checkpoint 必须分开报告四种上下文，不能混成一个均值：

```text
GT / generated / null / shuffled
```

`shuffled` 在同一图像、同一局部层级内交换 instruction/where context，防止用图像主体显著性冒充指令理解。

### 5.5 Where Loss

令 `m_pred=R(s_pred;rho_pred)`，GT mask 为 `m_gt`。主 mask loss 为：

```text
L_mask = (1 - softIoU(m_pred,m_gt))
       + 0.25 * balanced_BCE(m_pred,m_gt)
       + 0.10 * boundary_F1_loss_3px(m_pred,m_gt)
```

oracle auxiliary supervision 为：

```text
L_s     = Huber(s_pred/3, s*/3)
L_curve = mean_z |R(z;rho_pred) - r*(z)|, z=linspace(-3,3,257)
L_dir   = 1 - cosine(w_dir_pred, w_dir*)
```

总 loss 使用两段 schedule：

```text
前 30% optimizer steps:
L_where = L_mask + 1.00 L_s + 1.00 L_curve + 0.10 L_dir

后 70% optimizer steps:
L_where = L_mask + 0.25 L_s + 0.25 L_curve + 0.05 L_dir
```

前段用 oracle latent 解决 `s/readout` 联合优化的非辨识和早期坍缩，后段降低 auxiliary 权重，让最终 mask 监督主导。任何 loss 都同时在 GT/generated context 子批次上计算。

### 5.6 Where gate 与选择规则

在 `V_where` 的 generated-context 主榜上，候选必须同时满足：

> **本表已由 amendment A-5（§17.2，2026-08-05）修订**：删除 `AUC_target` 行、
> 「3px boundary F1」改为 **grid 级** boundary F1、新增中心先验基线的配对 Δ 与 p 值两行。
> 下表为**修订后**的现行判据；§5.5 的 loss 一个字未动。
>
> **脚注 1 · soft-IoU 一律为 min/max 形式** `sum(min(p,g)) / sum(max(p,g))`，
> **积形式 `sum(p·g)/sum(p+g−p·g)` 全实验禁用**。依据 Where-A 结果审阅 B-1 的实测：
> 积形式与 **GT mask 软度相关 0.955**、与**拟合质量相关 −0.003**，
> 且对**完美预测**的天花板中位仅 **0.786**。用积形式读本表的 `>= 0.75`，
> 排的是「GT 有多软」而不是「场对不对」——与 AUC 被禁是同一个失败机制。
> min/max 对软 GT 的完美预测恰好给 1.0（本仓库实测 1.0000 vs 积形式 0.5054）。
>
> **脚注 2 · 「相对 oracle」的分母** = Where-A 逐图 oracle 天花板，
> **low 档、min/max 形式**：`band 0.827` / `cband12 0.835`。
> 换一个档位或换成积形式都会**静默改变这条 gate**，因此分母在
> `q3vl/whereb/config.py:ORACLE_CEILING_LOW_MINMAX` 里钉死。
> 顺带一个读表时有用的推论：天花板约 0.83 时，绝对值那行 `>= 0.75`
> 对应约 **90.7%**（band）/ **89.8%**（cband12）的 oracle 占比，
> **高于本行的 85%** —— 即两行同时生效时，**绝对值那行才是更紧的那个**；
> 看到「相对 oracle 差一点」时先确认到底是哪一行在卡。

| 指标 | Gate |
|---|---:|
| local `.cgt` median soft-IoU | `>= 0.75` |
| 相对逐图 oracle 的 soft-IoU〔分母见脚注 2〕 | `>= 85%` |
| local soft-IoU p10 | `>= 0.55` |
| **grid 级** boundary F1 / oracle | `>= 75%` |
| **中心先验配对 Δ(hard-IoU)** | `> 0` |
| **该 Δ 的配对 p 值** | `<= 0.05` |
| instruction shuffle 后 IoU 降幅 | `>= 0.20` |
| `std(s_pred) / std(s*)` 中位数 | `>= 0.60` |
| global mask soft-IoU | `>= 0.98` |
| GT/generated context IoU gap | `<= 0.05` |

通过 gate 后按以下顺序选择唯一冻结 Where checkpoint：

1. generated-context local median soft-IoU；
2. **grid 级** boundary F1（A-5）；
3. p10 soft-IoU；
4. 参数量、峰值显存与延迟。

若无 arm 全部过门，仍选 lexicographic best 供后续诊断，但必须标记 `WHERE-GATE-FAILED`；后续 What 结果不能宣称完整方法已成立。

---

## 6. Where 到 What 的四种接口

冻结最佳 Where checkpoint，构造：

```text
F_roi = sum_p m_pred(p) * F_pre(p) / (sum_p m_pred(p) + eps)
F_bg  = sum_p (1-m_pred(p)) * F_pre(p) / (sum_p (1-m_pred(p)) + eps)
z_where = AttentionPool(Q_axis, Q_readout)
```

`Q_axis,Q_readout` 分别是预测 `w` 与 `rho` 的最终 query state。四个接口如下：

| ID | 给 Stage-What 的信息 | 科学问题 |
|---|---|---|
| `WC-0 ColorOnly` | `H_color` + 全局视觉池化；无显式 Where 输出 | 颜色 reasoning 自身能做到多少 |
| `WC-1 MaskPool` | WC-0 + `m_pred,F_roi,F_bg` | dense region pooling 是否足够 |
| `WC-2 QueryState` | WC-0 + `z_where,w,rho` | 全局空间 latent 是否比 mask pooling 更有效 |
| `WC-3 FullWhere` | WC-1 + WC-2 | dense 与 latent 是否互补 |

Where checkpoint 对所有 What arm 完全相同且冻结。`WC-0` 仍可能因 causal language hidden 间接含有此前 `<where>` 文本，所以另设严格 no-where control，见 §8.2。

---

## 7. Stage-What：独立 Color queries 与准确融合

### 7.1 Q_color 与连续 style code

`Q_color` 与 `Q_where` 完全独立。16 个 learnable color queries 只 cross-attend `<color>...</color>` 的 hidden：

```text
M_color = ColorConnector(Q_color, H_color) in R^[16 x 512]
z_style = LN(MLP(AttentionPool(M_color))) in R^1024
```

`Q_color` 不直接读取 `H_where`。Where 信息只通过四个显式 WC 接口进入，这样每个接口的增益可以归因。

> **修订说明**：`H_color` 的来源已由 **amendment A-4（2026-08-05）** 定档为「训练 50% GT / 50% Base SFT 自回归生成，评测两种 context 分开报告，`V_what` 选择以 generated 为主榜，C01/C02 用 forced `<color>` prefix 生成」。原文未声明此事，等价于 100% teacher-forced。见本文末尾 changelog 的 A-4 条目。

`z_style` 是连续 LUT function code，不是整数 style ID。其监督目标由冻结、确定性的 functional encoder 给出：

```text
u(T) = flatten(T(x)-x), x 为固定 17^3 RGB grid
z_gt = L2Norm(SRHT_1024(u(T)-mean_train_u))
```

SRHT 使用固定公开 seed，只在 train LUT 上计算中心，不学习 LUT lookup。随机投影保留函数距离，避免额外训练一个可能泄漏或坍缩的 target encoder。

### 7.2 Gaussian-aligned visual pooling

视觉融合按每个 Gaussian 在当前图片中的真实支持区域完成：

```text
a_i(p) = m_pred(p) * N(I_in(p); mu_i, Sigma_i)
v_i = sum_p a_i(p) * [RGB(p), Lab(p), V(F_pre(p))] / (sum_p a_i(p) + eps)
```

其中 `V:1024->256`。`v_i` 另外拼接 `log(sum a_i)` 与 valid bit；无有效像素时回退到 masked global pool，而不是产生 NaN。

这一步同时保留：

- `M_color` 的 token-level 语义细节；
- `z_style` 的 1024 维全局连续风格；
- `F_pre` 的局部视觉内容；
- Where 的显式 ROI 约束；
- 每个 Gaussian 对应的真实颜色分布。

### 7.3 Shared Transformer backend

两个 generator 使用同一容量等级的 48-slot Transformer，便于将差异归因于“完整生成 vs 共享 geometry”，而不是小 MLP 容量：

```text
48 learned slot embeddings + WC tokens
 -> 2 seed Transformer blocks
 -> provisional/fixed Gaussian geometry
 -> Gaussian-aligned pooling v_i
 -> 4 refinement Transformer blocks
 -> 48 independent decoder heads + one global head
```

每个 block：宽度 512、8 heads、FFN 2048、pre-norm。每个 block 都：

1. 在 48 slots 间 self-attention，建模 Gaussian 之间的互补与竞争；
2. cross-attend 完整 `M_color`，避免所有语义被压进单个向量；
3. 接收投影后的 `v_i` 与 WC tokens；
4. 使用 `z_style` 生成的 ModLN scale/shift 调制；
5. 采用 zero-init gated residual，防止训练初期视觉或 Where 分支压倒颜色语义。

独立 decoder head 接收 `[h_i, P(z_style)]`。不存在 LUT-ID embedding，也不存在只靠一个小 MLP 从 pooled feature 直接吐 1000 多个参数的路径。

### 7.4 `WHAT-FG48`：Full Generation

每个样本生成全部 48 个 Gaussian 参数：

| 参数/primitive | 维度 |
|---|---:|
| `mu_i` | 3 |
| SPD `Sigma_i` 的 Cholesky 参数 | 6 |
| opacity | 1 |
| existence | 1 |
| residual affine `M_i` | 9 |
| residual bias `b_i` | 3 |
| 合计 | 23 |

另有全局 `G,b` 共 12 维，因此每图生成：

```text
48 * 23 + 12 = 1116 parameters
```

前 2 个 seed blocks 先预测 provisional `mu,Sigma` 用于 aligned pooling，后 4 个 blocks 再联合细化全部参数。梯度可穿过 pooling 回到 geometry。

### 7.5 `WHAT-SB48`：Shared Backend/Geometry

跨样本共享 48 个可训练 `mu_i,Sigma_i`；它们以固定 4x4x3 RGB anchors 初始化，然后在全部训练 LUT 上联合优化。每个样本只生成：

```text
opacity_i, existence_i, M_i, b_i   # 每 primitive 14 维
G,b                                 # 全局 12 维
```

48 个 Gaussian 各有独立 decoder head，`z_style` 通过 ModLN 和 head conditioning 同时作用于每个 slot。共享的是 geometry 和 Transformer backend，不是输出值；每个样本仍可得到不同的稀疏激活、仿射颜色载荷与全局变换。

FG48/SB48 的 Transformer 深度、宽度和 query 数相同。通过调整 decoder head bottleneck，使两臂总 trainable parameters 相差不超过 2%；报告必须同时给出原始参数量、FLOPs、峰值显存、VLM 后增量延迟、33^3 bake 延迟，不能只比较质量。

### 7.6 LUT 函数与约束参数化

Gaussian 中心用 sigmoid 限定在 RGB cube；协方差由 Cholesky 构造并对角 softplus，保证 SPD；opacity/existence 使用 sigmoid。全局与局部 affine 都采用 identity-centered residual 参数化。

令归一化权重为 `q_i(x)`，则：

```text
# SUPERSEDED by amendment A-1 (2026-08-05) -- see the changelog at the end of
# this document.  The global term is now a pure residual `G x`, G zero-init;
# the form below gives T(x) = 2x at zero initialisation.
T_pred(x) = clamp(
    (I + DeltaG) x + b_g
    + sum_i q_i(x) * (M_i x + b_i),
    0, 1)
```

> **修订说明**：上式的全局项已由 **amendment A-1（2026-08-05）** 改写为纯残差 `G x + b_g`（`G` 零初始化）。字面的 `(I + DeltaG) x` 与归一化 `q_i` 联立会在零初始化时给出 `f(x) = 2x`，与战役红线冲突。以 changelog 的 A-1 条目为准，见本文末尾。

所有输出均可在固定 33^3 RGB lattice 上求值并烘焙成标准 LUT。生产评测一律使用与交付一致的 tetrahedral interpolation 回读。

---

## 8. What 实验矩阵

### 8.1 8 个全量主组合

| Arm | Where 接口 | Generator |
|---|---|---|
| `T01` | WC-0 ColorOnly | WHAT-FG48 |
| `T02` | WC-1 MaskPool | WHAT-FG48 |
| `T03` | WC-2 QueryState | WHAT-FG48 |
| `T04` | WC-3 FullWhere | WHAT-FG48 |
| `T05` | WC-0 ColorOnly | WHAT-SB48 |
| `T06` | WC-1 MaskPool | WHAT-SB48 |
| `T07` | WC-2 QueryState | WHAT-SB48 |
| `T08` | WC-3 FullWhere | WHAT-SB48 |

完整 `4 WC x 2 generator` 必须全量跑完。主报告必须给出 WC 与 generator 的 interaction，而不是分别列两个边际排名。

### 8.2 四个必要控制臂

| Arm | 设置 | 作用 |
|---|---|---|
| `C01 NoWhere-FG48` | 去掉 `<where>` prefix、所有 Where 输出和 mask-conditioned pooling | FG 严格 no-where 下界 |
| `C02 NoWhere-SB48` | 同上 | SB 严格 no-where 下界 |
| `C03 OracleWhere-FG48` | GT mask + oracle `w,rho,z_where` | FG 的 Where 上界 |
| `C04 OracleWhere-SB48` | 同上 | SB 的 Where 上界 |

因此 Stage-What 共 12 个完整训练臂。Oracle 输入只存在于 ceiling control，不能进入主榜。

---

## 9. What Loss：最终定档

### 9.1 LUT function supervision

每个样本取 2048 个 RGB 查询点：

- 1024 个来自固定分层 uniform-RGB sampler；
- 1024 个来自当前 `I_in` 的 natural RGB，local 样本按 frozen `m_pred` 加权采样，global 样本从全图采样。

GT 为 LUT 函数 `T_gt(x)`，不是 `I_tar`。主损失：

```text
L_func = mean Charbonnier(T_pred(x) - T_gt(x))
```

uniform 与 natural 两部分各占 50%，分别报告，避免只拟合自然图中出现频率高的颜色。

### 9.2 hue-chroma loss

按 GLUT/CGLUT 的方向使用 CIELab hue cosine，并以 GT chroma 加权：

```text
L_hc = mean [ normalize(C_gt) * (1 - cos(h_pred - h_gt)) ]
```

Lab/chroma 必须归一化到无量纲范围后再进入 loss，不能复现旧 A0 中绝对 Lab 数值造成梯度约 229 倍放大的单位错误。训练日志单独记录 `grad_norm(L_func)` 与 `grad_norm(10*L_hc)` 的比例。

### 9.3 sparsity 与 style-code 防坍缩

```text
R_sparse     = mean(binary_entropy(opacity) + binary_entropy(existence)) / 2
L_style_cos  = 1 - cosine(z_style, z_gt)
L_style_dist = Huber(d_style(i,j), d_func(T_i,T_j))  # batch 内成对距离
L_var        = mean max(0, 1 - std(z_style_dim))
L_cov        = normalized squared off-diagonal covariance
```

`L_var/L_cov` 使用跨两卡 all-gather 后的 batch 统计；若各 arm 单卡并行，则使用 stop-gradient FIFO statistics queue，队列内容和更新规则对所有 arm 固定。它们只防止连续 code 坍缩，不承担 LUT 重建主任务。

### 9.4 33^3 bake consistency

每次训练前向把解析 Gaussian 函数在 33^3 lattice 上烘焙，用生产 tetrahedral interpolation 在同一批查询点回读：

```text
L_bake = mean Charbonnier(T_tetra(bake_33(T_pred), x) - T_pred(x))
```

这项约束训练出来的函数在交付形态下仍成立，而不是只保证解析 renderer 好看。

### 9.5 总 Loss 与固定权重

```text
L_what = 1.00  L_func
       + 10.00 L_hc
       + 0.001 R_sparse
       + 0.05  L_style_cos
       + 0.05  L_style_dist
       + 0.02  (L_var + L_cov)
       + 0.10  L_bake
```

该配方是所有 12 个 What 臂的统一起点和主配置，不为单个 arm 单独调权重。只允许在 preflight 阶段修复单位、非有限值或实现错误；不得根据主实验结果事后改 loss 再只重跑失败臂。

`I_tar` 不进入 `L_what`。最终渲染图在评测时由预测 mask 与预测 LUT 显式合成，再和 `I_tar` 比较；这能把 Where 误差、LUT function 误差和最终成像误差分开归因。

---

## 10. 优化器、训练长度与 checkpoint

### 10.1 Base SFT

严格使用上游定档：两卡 ZeRO-3、global batch 32、AdamW LR `1e-5`、WD 0、warmup 3%、cosine、BF16、clip 1、1 epoch，保护 0.5/1.0 epoch checkpoint。

### 10.2 Where-A

```yaml
optimizer: AdamW
projector_lr: 1.0e-4
weight_decay: 0.01
warmup_ratio: 0.03
scheduler: cosine
precision: bf16_forward_fp32_oracle_fit
epochs: 1.0 over all eligible local train samples
```

L-BFGS oracle fit 使用 float64、多起点和固定容差；失败样本进入显式 rejection/fit report，不静默换成零向量。

### 10.3 Where-B

```yaml
optimizer: AdamW
learning_rate: 2.0e-4
weight_decay: 0.01
warmup_ratio: 0.03
scheduler: cosine
max_grad_norm: 1.0
precision: bf16
effective_batch_per_arm: 32
epochs: 1.0
eval_steps: 500
save_steps: 500
```

单卡独立跑一个 arm。micro-batch 先做显存探测，再调整 gradient accumulation，effective batch 始终为 32。冻结 VLM 不建立其参数梯度，但视觉与语言前向均使用真实模型，不用伪特征。

### 10.4 Stage-What

```yaml
optimizer: AdamW
query_connector_backend_lr: 1.0e-4
decoder_head_lr: 1.0e-4
shared_geometry_lr: 5.0e-5       # 仅 SB48
weight_decay: 0.01
warmup_ratio: 0.03
scheduler: cosine
max_grad_norm: 1.0
precision: bf16
effective_batch_per_arm: 32
epochs: 1.0
eval_steps: 500
save_steps: 500
```

geometry、bias、LayerNorm 和 ModLN 参数不做 weight decay。保护 0.5/1.0 epoch checkpoint；普通 checkpoint 最多保留 3 个，但被选中和保护的 checkpoint 不受滚动删除影响。

Stage-Where 与 Stage-What 的正式主臂都不做低成本短跑筛选，不基于早期指标中止。只允许因 OOM、NaN/Inf、manifest/checkpoint digest 不一致或数据读取错误终止并修复后从同一 authority 恢复。

所有主臂先用同一训练 seed 完整跑完；选择出的前两种配置各追加两个完整 seed，用于确认排序是否由初始化造成。最终统计同时给出 across-seed 范围和 sample-level paired bootstrap 95% CI。

---

## 11. 两卡实验排期

Base SFT 依赖两卡共同运行；完成并冻结 checkpoint 后，Where/What 阶段每张卡独立运行一个完整 arm。

| Wave | GPU 0 | GPU 1 | 依赖 |
|---|---|---|---|
| S0 | Base SFT Arm B（ZeRO-3） | Base SFT Arm B（ZeRO-3） | 模型与数据 preflight |
| A1 | BA-0-Fixed | BA-1-Band | S0 |
| A2 | BA-2-CBand12 | BA-3-Joint | A1 |
| W1 | W01 | W02 | BA-3-Joint 冻结 |
| W2 | W03 | W04 | 同上 |
| W3 | W05 | W06 | 同上 |
| W4 | W07 | W08 | 同上 |
| T1 | T01 | T05 | 最佳 Where 冻结 |
| T2 | T02 | T06 | 同上 |
| T3 | T03 | T07 | 同上 |
| T4 | T04 | T08 | 同上 |
| C1 | C01 | C02 | 同上 |
| C2 | C03 | C04 | 同上 |

每个 Wave 的两个 arm 使用相同数据次序与采样 seed。后续 top-2 多 seed 复跑同样一张卡一个 arm，不与主矩阵混合命名。

Stage 之间有真实依赖，不能把 What 抢跑在 Where 定档前。Stage 内不做 winner-takes-all 排队，表中每一臂都完整训练。

---

## 12. What checkpoint 选择与最终判据

### 12.1 LUT function 指标

在固定 uniform、natural 和完整 33^3 grid 上分别报告：

- RGB MAE、RMSE、PSNR；
- CIEDE2000 mean/median/p90/p95；
- hue angular error 与 chroma error；
- analytic Gaussian 与 baked 33^3 tetrahedral 回读的 MAE/p99/max；
- LUT cube 内 out-of-range 比例与非有限值计数。

生产 bake gate：

```text
analytic vs 33^3 tetrahedral readback:
mean RGB MAE <= 1e-4
p99 RGB error <= 5e-4
non-finite = 0
```

### 12.2 最终图像指标

按 `I_out = I_in + m*(T(I_in)-I_in)` 渲染后，对 `I_tar` 报告：

- PSNR、SSIM、LPIPS、CIEDE2000；
- mask 内部、3px 边界带、mask 外部分区指标；
- g1-g4、l1-l6、source、L-level 和 mask 面积分层；
- global/local 分开，不用总体均值掩盖局部退化。

### 12.3 因果负控制与防坍缩指标

- instruction shuffle/swap 后 LUT function error 必须显著恶化；
- image shuffle 后 aligned visual pooling 的收益必须下降；
- `z_style` effective rank、每维方差、pairwise distance 与 GT LUT function distance 的 Spearman 相关；
- 48 个 existence/opacity 的激活数分布，防止全开或全关；
- predicted geometry/payload 跨样本方差；
- `WC-1/2/3` 相对 `WC-0` 的 paired improvement；
- 主 arm 与 strict no-where、oracle-where 的距离。

### 12.4 选择规则

只在 `V_what` 选择 checkpoint：

1. 先满足 bake gate、finite gate 和 instruction-shuffle 正向依赖；
2. 主排序为 local 最终图像 median CIEDE2000；
3. 次排序为 LUT function CIEDE2000 p90；
4. 再看 boundary-band CIEDE2000、参数量和延迟。

同一 arm 的多个 step 只保留最佳 checkpoint 进入跨 arm 排名。最终 top-2 必须是两个不同配置，而不是同一 arm 的两个相邻 step。

最终只在所有选择冻结后打开 `T_final` 一次。主差异用 paired bootstrap 95% CI；报告 WC、generator 和二者 interaction。`T_lut_unseen` 单列，不能和 seen-LUT 样本混合。

---

## 13. 可视化与中文报告交付

最终报告全部使用中文。对 `V_what` 选出的两个最佳 checkpoint，在冻结的 `T_final` 可视化清单上各展示 20 张，不得按效果人工挑图。20 张至少覆盖：

- g1-g4 与 l1-l6；
- 小/中/大 mask；
- 低饱和、高饱和、肤色、天空/植被/建筑等代表颜色；
- 环形/细边界/多连通区域；
- 典型成功和预注册尾部样本。

每张联图固定包含：

```text
I_in | instruction | GT mask | pred s | pred mask
GT LUT render | pred analytic render | pred 33^3 render | I_tar | abs error
```

另为两个最佳 checkpoint 各生成以下 feature 可视化：

1. `Q_where` 最后一个 MetaCanvas block 的 token PCA/UMAP、token norm、attention entropy，并恢复成真实 2D canvas；
2. `w`-canvas 与 `rho`-canvas/heads 的差异图；
3. 64 个 semantic basis 的代表通道、组合后的 `s_low`、guided-upsample `s` 与最终 readout；
4. `Q_color`/`M_color` 最终 feature 的 PCA、effective rank 和对 `<color>` token 的 cross-attention；
5. 1024 维 `z_style` 在 20 样本上的相似度矩阵与 GT function-code 相似度矩阵；
6. 48 个 Gaussian slot 最终 feature、geometry、existence/opacity、ROI mass 和激活热图；
7. FG48 与 SB48 的 Gaussian 覆盖和 payload 差异。

本轮已按用户决策跳过 MetaQuery，因此新报告不伪造“MetaQuery vs MetaCanvas”图。对应替代物是 `Q_where MetaCanvas`、独立 `Q_color` 与 48 个 Gaussian decoder slots；历史 MetaQuery 图仍保留在旧实验报告中。

除 20 张固定样本外，单列至少 5 个最差案例，按 Where 错误、Color/LUT 错误、bake 误差、reasoning 格式错误分类，不得只展示成功样本。

### 13.1 必交文件

建议实验根目录：

```text
experiments/Q3VL_metacanvas_where_what_20260804/
```

每个阶段最终至少交付：

- `EXPERIMENT_PROTOCOL.md` 或本文的不可变副本及 digest；
- `config/*.yaml` 与 resolved config；
- `manifest.json`、split/index/shard digests；
- trainable/frozen 参数清单与参数量；
- 每臂完整 log、`job.marker`、checkpoint index；
- `metrics.json`、逐样本 `per_sample.jsonl`、bootstrap CI；
- 中文 `REPORT.md`；
- top-2 各 20 张联图、feature 图、失败案例；
- top-2 的标准 33^3 LUT 与 tetrahedral readback 验证记录。

---

## 14. 运行前强制 preflight

在任何正式训练启动前，必须完成并保存：

1. Qwen3-VL 权重 shard/checksum 完整性；
2. Base SFT 四个 special token 与两段式输出校验；
3. Vision/merger/Language、Where、What 的 frozen/trainable 参数逐名清单；
4. `F_pre` 的真实形状、宽高比、位置编码和一次 guided upsample 校验；
5. `BA-3-Joint` 的 residualization 正交误差、condition number 和 oracle fit 成功率；
6. `w_dir` 符号、单位范数、`alpha` 正值和 readout 边界测试；
7. GT/generated/null/shuffled context 数据流无串线；
8. 证明 `Q_where` 不读 `H_color`，`Q_color` 不读 `H_where`；
9. 证明任何主 arm 的输入都不含 `I_tar`、GT mask、GT LUT 或 oracle latent；
10. indexed-shard 随机读、checksum、原子发布与 resume 测试；
11. train/`V_where`/`V_what`/`T_final`/`T_lut_unseen` 的 group-level 无交集审计；
12. 48 Gaussian SPD、权重归一化、identity 初始化、finite gradient 测试；
13. Lab loss 单位与分项 gradient norm 校验；
14. analytic -> 33^3 bake -> tetrahedral readback 数值单元测试；
15. 单 batch 显存、吞吐、effective batch 32 与两卡一臂调度验证。

任一项失败时只允许修复实现并重做 preflight，不能把 smoke 数字写成实验结果。

---

## 15. 最终要回答的论文问题

该矩阵不是只找一个最好分数，而是回答五个可归因的问题：

1. merger 前 Qwen3-VL feature 经共享 64 维 projector 后，是否达到 E2 oracle basis 的空间表达上界？
2. MetaCanvas 是否能从 `I_in + instruction` 稳定预测全局 `w,rho`，并在 generated reasoning 下保持清晰边界和指令依赖？
3. `MaskPool`、`QueryState` 和 `FullWhere` 中，哪种 Where 信息真正改善 LUT 预测，二者是否互补？
4. Transformer Full Generation 的容量收益，是否值得相对 Shared Backend/Geometry 的参数、延迟和跨 LUT 连续性代价？
5. 解析 48-Gaussian renderer 能否无损落成标准 33^3 LUT，并在 LUT-ID-disjoint 的约 4000 LUT 范围保持泛化？

只有在 Where gate、What 负控制、unseen-LUT split 和 33^3 bake gate 同时成立时，才能把结果写成完整方法贡献。否则应按失败所在阶段分别报告，不能用最终平均图像指标掩盖 Where 坍缩、style collapse 或 LUT 泄漏。

---

## 16. 实验规模摘要

| 部分 | 全量训练臂 | 选择集 |
|---|---:|---|
| Base SFT | 1 | 原 SFT eval / 语言指标 |
| Where-A basis calibration | 4 | `V_where` oracle 指标 |
| Where-B MetaCanvas | 8 | `V_where` generated-context 指标 |
| What 主矩阵 | 8 | `V_what` |
| What controls | 4 | `V_what`，不进主榜 |
| Top-2 稳定性复跑 | 4 个额外完整 run | `V_what`，配置已冻结 |
| 最终测试 | 0 个训练臂 | `T_final` 与 `T_lut_unseen` 各打开一次 |

核心主比较为：

```text
Where: 4 MetaCanvas x 2 Readout = 8
What:  4 Where interfaces x 2 Generators = 8
Controls: 2 No-Where + 2 Oracle-Where = 4
```

这就是本轮最终实验方案。后续实现不得减少主组合、混用选择集、把 oracle 结果放进主榜，或在看到结果后为某个 arm 单独改 Loss。

---

## 17. Changelog

### 2026-08-05 · amendment A-1（D-W1）：§7.6 全局 affine 改为纯残差

**依据**：WHAT-IMPL 实现探查发现并实测，独立实现审阅（`docs/reviews/REVIEW-impl-What.md` §一）审定批准。

**改写内容**。§7.6 的第二段与公式段以本条为准：

> Gaussian 中心用 sigmoid 限定在 RGB cube；协方差由 Cholesky 构造并对角 softplus（带正下界 `sigma_lo = 0.02`），保证 SPD；opacity/existence 使用 sigmoid。
>
> **局部** affine 采用 identity-centered residual 参数化 `M_i = I + DeltaM_i`；**全局** affine 采用**纯残差**参数化，`G` 零初始化（**不是** `I`）。令归一化权重为 `q_i(x)`（`sum_i q_i <= 1`，分母含 `eps = 1e-6`），则：
>
> ```text
> T_pred(x) = clamp( G x + b_g + sum_i q_i(x) * (M_i x + b_i), 0, 1 )
> ```

**为什么不是 `(I + DeltaG) x`**。`q_i` 是归一化权重，`sum_i q_i(x) * M_i x ≈ x` 已经在混合项里给出了恒等；若全局项再写成 `(I + DeltaG) x`，零初始化时 `T(x) = 2x`——正是战役红线点名的「全局仿射 G 初始化 = 0 不是 I（否则 f(x)=2x）」。

三者「identity-centered 局部 affine」+「输出头零初始化」+「归一化权重」是联立不可能的，必须改其中一条；而输出头必须零初始化，否则第一步的 `T_pred` 就是一个随机 LUT，`L_bake`（权重 0.10）会在训练开始就把一个随机函数烘进 33^3 格点。

**实测证据**（两方独立复算，互不调用对方代码）：

| 量 | 实现者实测 | 独立审阅复算 |
|---|---|---|
| `sum_i q_i` 取值域 | `[0.9999965, 1.0000002]` | `[0.9999975, 1.0]` |
| 字面式 `mean T(x)/x`（零初始化） | `2.0000` | `1.99999988` |
| 字面式 `max|T(x) - x|`（不 clamp） | `> 0.9` | `0.99984` |
| 改写式 `max|T(x) - x|`（零初始化） | `1.49e-6` | `2.4e-6` |

落盘位置：`experiments/Q3VL_metacanvas_where_what_20260804/what/preflight/preflight_what.json` 的 `WT-P-zero-init-identity`。

**科学问题不变，已证明**。令 `G' = G + I`，则

```text
{ x -> (I + DeltaG) x + b_g + sum_i q_i (M_i x + b_i) }
{ x -> G' x        + b_g + sum_i q_i (M_i x + b_i) }
```

在 `DeltaG, G' ∈ R^{3x3}` 上取遍**同一函数集**。差别只在参数化的原点落在哪个函数上；由于 decoder head 的第二个 Linear 零初始化，那个原点恰好就是初始化点。因此本 amendment 改的是**初始化**，不是**假设类**：

- 表达容量不变；
- 每样本参数量不变（FG48 = 1116，SB48 = 684，§7.4 / §7.5 的数字都不动）；
- §12 的任何指标定义不动；
- §15 的五个论文问题一个都不动。

**实现对应**：`q3vl/what/config.py::GLOBAL_AFFINE_MODE = "residual_zero"`，逐项等于 `model/glut_repro/model_rdg.py::render`（已由 `ci_checks_rdg.check_render_matches_batched_glut` 对拍 `BatchedGLUT` / GLUT Eq.1-3 到 1e-5）。`"identity_centered"` 分支保留，**只**为让 2x 可被测量而不是被断言。

### 2026-08-05 · amendment A-2（D-W9）：§9.3 `L_style_dist` 的 `d_func` 口径

**依据**：REVIEW-impl-What B-3；主 agent 裁定。

§9.3 的 `L_style_dist = Huber(d_style(i,j), d_func(T_i, T_j))` 中，两个距离的口径固定为：

```text
d_func(T_i, T_j) = || u(T_i) - u(T_j) ||_2 / C
d_style(i, j)    = || z_style_i / ||z_style_i||  -  z_style_j / ||z_style_j|| ||_2
```

其中 `u(T)` 是 §7.1 的 `flatten(T(x) - x)`（固定 17^3 grid，**未经 SRHT、未经 L2 归一化**），`C` 是**在 train LUT 集上预计算的固定整体常量**：全体 train LUT 两两距离的均方根，与 `mean_train_u` 在同一次遍历中算出并随之发布。**禁止逐 batch / 逐图归一化**（与 s 轴的同类纪律一致）。

理由：`z_gt = L2Norm(SRHT(u - mean))` 会丢掉 `u` 的**幅度**信息，用它当 `d_func` 会让「同一 look 的强度变体」之间的目标距离为 0，而真实函数距离很大；叠加 `L_style_cos` 也只管方向，结果是 `L_what` 里没有任何一项监督 `z_style` 的编辑幅度。本仓库语料是 Lightroom preset，强度变体是常见的。

`C` 的闭式（一次遍历即可精确求出全部 `N(N-1)/2` 对，无需抽样）：

```text
C^2 = 2 * ( N * sum_i ||u_i||^2 - || sum_i u_i ||^2 ) / ( N * (N-1) )
```

**配套**：§12.3 的「`z_style` pairwise distance 与 GT LUT function distance 的 Spearman 相关」必须用**未归一化的原始** `|| u_i - u_j ||`，与优化项脱钩；`z_gt` 方向距离的相关系数可并列报告，但不得作为该诊断的唯一读数。

### 2026-08-05 · amendment A-3（D-W10）：§9.1 natural 采样的 mask 对 12 臂统一

**依据**：REVIEW-impl-What B-5；主 agent 裁定。

§9.1「local 样本按 frozen `m_pred` 加权采样」中的 `m_pred` 是**监督侧的冻结量**，与 `T_gt` 同性质，**不是模型输入**。因此：

- **12 个臂（含 C01-C04）的 loss 查询点 natural 半区，一律用同一个冻结 Where checkpoint 的 `m_pred` 加权**；
- `where_source`（`predicted` / `none` / `oracle`）**只影响模型输入侧**，不影响 loss 的查询色分布；
- 具体地：`C01`/`C02`（strict no-where）与 `T01`/`T05`（WC-0）的**唯一**差别是语言序列里有没有 `<where>` 段；`C03`/`C04`（oracle-where）的 GT mask 与 oracle latent 只进模型输入，其 loss 查询色分布与主臂完全相同。

理由：§9.5 明写「该配方是所有 12 个 What 臂的统一起点和主配置」。若 no-where 臂的 natural 半区退回全图采样，则它与主臂的目标分布不同，控制臂就不再是干净的下界，而是「下界 + 一次目标分布消融」的混合体。

**实现要求**：每条 per-sample 记录写 `natural_weighting` 字段（取值 `frozen_m_pred` / `global_uniform`，后者只用于 global 样本），使该性质可被审计。


### 2026-08-05 · amendment A-4（NF-1）：Stage-What 的 `<color>` context 定档为 50/50 teacher/generated

**依据**：`docs/reviews/REVIEW-impl-What.md` 聚焦复审 §十 NF-1（新增 BLOCKER）+ 主 agent 裁定（采纳审阅者选项 (a)）。

**问题**。初版 Stage-What 实现的 `color_ids` 与 `where_ids` 全部来自 record 的 **GT** 文本，即训练与评测 100% teacher-forced，且此决策从未声明。后果有三：

1. **§0 的核心主张失去证据**。「最终网络只接收 `I_in + instruction`」在 Stage-What 的任何一个交付数字里都没有被检验过；§15 的问题 3/4/5 的答案全部条件在 GT 推理文本上。
2. **amendment A-3 抬高了赌注**。冻结 Where 的 `m_pred` 现在是全部 12 臂的**监督**掩膜，而那个 `m_pred` 由 GT `<where>` 文本算出——包括按构造永远看不到 `<where>` 的 C01/C02。
3. **train/test 失配无法事后补救**。12 臂若在 100% teacher 上训完，再改用 generated 上下文评测，掉分是必然的，唯一正确动作是**重训 12 臂**。Where-B 的 §5.4 用 50/50 训练正是为了避免这件事。

**定档内容**（四点）：

1. **训练 50/50**。每个 **micro**-batch 固定 50% teacher context（GT `<color>` token hidden）/ 50% generated context（Base SFT 自回归生成的 `<color>` token hidden）。在 micro-batch 上强制，因而对任何梯度累积倍数下的 effective batch 都成立。generated 样本缺失闭合标签时**不得回退 GT**：按固定 token 边界截断并记录 format 失败（与 §5.4 逐字同构）。缓存的是 **token ids** 而非 hidden；hidden 在训练时由 teacher context 所用的同一个 encode 重放，使「同层、同位置、同归一化」成为构造性事实。
2. **评测分报**。GT 与 generated 两种 context **分开报告**，永不混成一个均值；`arm_metrics` 的每个 checkpoint 产出**每 context 一行**。
3. **generated 主榜**。`V_what` 的 checkpoint 选择以 **generated-context** 榜为准（`main_board` 增加 context 维度，默认 `generated`）。teacher-context 榜并列报告，两者之差（generated − teacher）是「该臂对 GT 推理文本的依赖程度」的直接读数。
4. **控制臂的 generated 语义**。`C01`/`C02` 按构造去掉 `<where>` prefix，因此其 generated context 必须由**同样的**方式生成：prompt 不含 `<where>`，`<color>` 开标签作为 **forced prefix**，其后自回归生成。若给它们重放「`<where>` 之后生成的」`<color>`，where 推理会经由 token ids 回到严格 no-where 控制臂里，正是该控制臂要排除的东西。

**上游依赖**。generated context 由**扩展后的** Where-B 生成作业产出（`q3vl.whereb.scripts.make_generated_context`，schema `q3vl.where_b.genwhere/2` = v1 + `<color>` 段；新增 forced-prefix CLI 模式）。Base SFT 本就一次生成 `<where>...</where><color>...</color>`，v2 只是把第二段一并留下。每条记录必带 `mode` 字段（`with_where_prefix` / `forced_color_prefix`），消费侧逐样本断言其与本臂所需一致。

**token 边界**。`<color>` 段边界 = **384**。实测依据：跨五个 split 抽样 3,745 条 record 的 `tokens.color`，min 108 / p50 178 / p95 246 / p99 285 / **max 324**；384 = max + 两个标签 + 约 18% 余量（与 Where-B 的 96 对 measured max 79 同样的余量）。teacher 侧超界**报错**而非截断——边界是对语料的断言，不是截断路径。

**科学问题的影响**：本 amendment **恢复**（而非改变）§0 与 §15 问题 3/4/5 的可回答性——它们本就要求 generated 语境下的数字。§7.4/§7.5 的每样本参数量、§9.5 的七个权重、§12.1 的 gate 阈值、§12.4 的字典序五键、§8 的 12 臂矩阵均**不变**。


---

## 17.2 Amendment A-5：判据侧对齐 2026-08-05 用户红线（2026-08-05）

**依据**：`CLAUDE.md` 2026-08-05 新增两节红线——「AUC 全实验禁用」与「空间场可视化纪律」。
本 amendment **只动判据/gate/报告列与可视化**，**§5.5 的 loss 一个字不动**
（§9.5/§10.4 明文禁止事后改 loss；红线针对的是判据，不是优化目标）。

### 删了什么

1. **§5.6 的 `AUC_target >= 0.80` 行整行删除**，且新实验**不再产出**该指标
   （`q3vl/whereb/metrics.py` 里 `auc_target()` 已删除，非保留不用）。
   三次被误导的机制各不相同，所以不是「小心用」能解决的：
   - 对全体样本相同的一句 `"the main subject"` 拿到 AUC 0.907，而 `AUC_target` 只有 0.523（RO-X1）；
   - 零参数中心先验场 AUC **0.836**，跑赢 RO-9c 全部 6 个 attention 读出（0.695–0.784）；
   - MCQ-L 三档条件消融 AUC 为 0.9469/0.9499/0.9481（真实指令甚至最低），
     而同批 checkpoint 的 soft-IoU 是 0.605/0.508/0.509、PSNR_in 差 2.2 dB。
2. **像素级 3px boundary F1 不再作判据**：同支撑同 k 的随机 top-k 在该列得 **0.0394**，
   高于中心先验的 **0.0327**——它主要在测边界**长度**，越碎越占便宜。

### 换成了什么

§5.6 判据表改为红线规定的三列（缺一不可），全部在 `F_pre` 网格上、
阈值化**统一为「匹配 GT 面积的 top-k」**（禁逐场调阈值）：

| 列 | 说明 |
|---|---|
| soft-IoU / hard-IoU | 覆盖对不对 |
| **grid 级** boundary F1 | 形状跟不跟。网格上两个场各占恰好 `k` 格，没有「边界长度」可薅 |
| **中心先验基线** | 零参数 `-到画幅中心距离`，**同支撑、同 top-k 规则**。任何「场找到了主体」的主张必须出示**配对 Δ 与 p 值**（bootstrap，2000 次重采样） |

**指令条件性**改用配对差分框架，不得用任何 AUC 变体代替。**已实现**（A5-B1 按路线 (a) 落地）：

- **三条负控制全部可产出**，均为 `CONTEXT_MODES` 的一等模式，`evaluate_arm` 各出一块板：
  `shuffled`（同图伙伴的真实指令，§5.4 原有）、
  `irrelevant_words`（固定词表 + 逐样本种子抽 12 个无关名词）、
  `fixed_phrase`（对全体样本相同的 `"the main subject"`——红线自己的反例，
  它曾拿到 AUC 0.907 而 `AUC_target` 只有 0.523）。
  三者都同时替换 prompt 里的 instruction 与 `<where>` 正文（与 D-B15 同一规则：
  留着真实指令等于让正确答案仍然可达）。
- **配对差分 = 同图两条不同指令**：对同一 `source_image_id` 下目标区域不同的两条样本 `A`/`B`，
  取 `d_A = IoU(field_A, GT_A) − IoU(field_A, GT_B)`（以及对称的 `d_B`），
  用 sign-flip 置换检验出 Δ 与 p。图像被固定，因此图像显著性与中心先验成对抵消。

> **与红线原文的关系（主 agent 已裁定，开放项关闭）**：红线写「同图两条**相反**指令」。
> 实测 `V_where` 本地 400 条 / 114 个图像组：同图不同指令的样本对 **712** 对，
> 其中语义相反（明暗/冷暖/饱和任一轴）**147** 对；但「同主体 + 颜色方向相反 + **GT mask 相同**」
> 只有 **2 对**（不同主体的 665 对里 mask 相同的仅 1 对；96 个多样本组全部含 ≥2 个不同 mask）——
> **字面对照在现有语料里不存在**，因为每条指令都绑定它自己的候选区域。
>
> **裁定：两类控制都进评测**，方向相反、各司其职：
>
> | 控制 | 构造 | 期望 | 角色 |
> |---|---|---|---:|
> | **方向性配对 Δ**（校准后） | 同图**不同指令**（目标区域不同）的 `IoU(自 GT) − IoU(伙伴 GT)`，**减去同一流程下中心先验场的同一量** | 校准后 `Δ_field − Δ_prior > 0`，sign-flip p | **报告列**，非 gate |
> | **antonym 不变性** | 固定反义词表翻转指令里的颜色方向词，**主体短语不变** | **中位 \|Δ_IoU\| ≤ 0.05**（预注册宽松阈值） | 负控制列，**非 gate** |
>
> **两条都不是 gate**：`config.GATES` 恒为十行，不含任何 `instruction_paired_*` 项
> （审阅 F-B2：文档一度把配对 Δ 写成 gate，而代码里没有）。
> **为什么现在不能升为 gate**（审阅 F-B1）：原始配对 Δ 有**面积混杂**——预测被二值化成
> `k = |GT_A|` 格，于是 `IoU(pred_k, GT_B) ≤ |GT_A|/|GT_B|`，伙伴区域更大时 cross 分数被
> 机械压低，`self − cross` 天然为正。实测**零参数中心先验场**在同心构造（两张 GT 圆心相同、
> 只差面积）下拿到 **+0.7590**。修法是**中心先验校准列**：同一套交叉打分流程对中心先验再跑一遍，
> 报告 `Δ_field − Δ_prior`（面积效应对两者一视同仁，相减即消），并另报
> 面积比 ∈ [0.5, 2] 的**面积均衡子集**作为更保守的第二视图。
> **若将来要把校准后的 Δ 升为 gate，必须走 amendment**，且前提是真实板上中心先验的净 Δ ≈ 0。
>
> antonym 控制抓的是「Where 场偷读颜色方向词」：Where 的输出应当只由**主体**决定，
> 翻转 darker↔brighter 不该让 mask 动。词表是**固定落盘**的常量
> （`q3vl/whereb/antonyms.py`，带 sha256 digest，非生成 ⇒ 无编造风险），覆盖
> luminance / temperature / saturation 三轴。实测覆盖率：**98.5%** 的指令含可翻转词
> （luma 352 / temp 341 / sat 253，n=400），而 `<where>` 段只有 **1%** ——
> 主体文本本来就不谈颜色，这正是该控制干净的原因；翻转在全语料上是**对合**（400/400 往返一致）。

### soft-IoU 的形式（2026-08-06，Where-A 结果审阅 B-1 触发）

本 amendment 顺带把一个**一直隐含**的口径写成明文：**soft-IoU 恒为 min/max 形式**，
`sum(min(p,g)) / sum(max(p,g))`；**积形式禁用**。

Where-A 结果审阅 B-1 实测积形式：与 **GT mask 软度相关 0.955**、
与**拟合质量相关 −0.003**、**完美预测的天花板中位 0.786**。
若 §5.6 的 `soft-IoU >= 0.75` 用积形式读，它就是**第二个 AUC**——
排的是 GT 软度而非空间正确性。

本仓库现状核实（2026-08-06）：`q3vl/whereb/metrics.py` 的 `soft_iou_value`
**本来就是 min/max**（实测：对软 GT 的完美预测得 **1.0000**，积形式为 **0.5054**）。
本次把它从「默认参数恰好是对的」升级为「**读 `config.SOFT_IOU_KIND` 的显式契约**」，
并补了负向断言测试 `tests/test_soft_iou_form.py`（14 条）：
天花板恒为 1.0、对软度不敏感（min/max 极差 <1e-4，积形式 >0.5）、
criteria 路径的 AST 扫描里不得出现 `"prod"`。

### loss 不变（明确声明）

`L_mask = (1 − softIoU) + 0.25·balanced_BCE + 0.10·boundary_F1_loss_3px` **原样保留**，
其中 `boundary_F1_loss_3px` 仍是**像素级、3px 容差**的那一个。
判据用 grid 级、loss 用像素级，二者**故意不同**且必须在 REPORT 里写清：
这恰好让判据列不再是被直接优化的量（归因价值反而更高）。

### 可视化纪律（落码，`q3vl/whereb/viz.py`）

- **禁逐图 min-max 着色**：`color_scale(mode="per_image_minmax")` 直接抛
  `PerImageMinMaxError`，不是给个警告了事（一个键之遥的默认值不算禁用）。
  依据：RO-9c 的 pad 格占 16×16 中约 5.3 格、吃掉 53–74% 注意力质量、93% 源 argmax 落在 pad 里。
- **色标只取有效格**；pad 格显式画白或打叉（`pad_style`），不许静默填补。
- **叠回原图用严格逆映射** `grid_to_img`（整数边界，最近邻整数倍展开），**禁直接 resize**。
- **着色归着色、算数归算数**：`FieldRender.raw_stats` 来自未归一化原始场，
  与色标端点分开返回；`viz.py` 不返回任何判据数字。

### 不受影响的部分

§5.5 loss、§5.1–5.4 的结构与 context 定义、§10.3 优化配置、§5.3 的 8 臂矩阵、
§5.6 的字典序选择规则（第 2 键由「3px boundary F1」改为「grid 级 boundary F1」）均不变。
