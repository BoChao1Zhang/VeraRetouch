# EPR-031 · WhatCodec：VLM Evidence Resampler → Canonical LUT Code → Anchored Primitive Slot Decoder

状态：提案，2026-08-16。**本 EPR 只改 conditioner，不改 carrier。**
carrier 一律取 EPR-028 R1 的 `A1`（3D explicit gate）与 `A3`（4D joint），
它们的数学在 `EPR-028_4d-gaussian-conditional-slice/PROPOSAL_R1.md`，本文不重复。

---

## 0. 本轮核实过的外部事实（当日 curl / 本地文件，逐条给出处）

| 事实 | 出处 | 核实结果 |
|---|---|---|
| Qwen3-VL 技术报告 arXiv:2511.21631 | `curl https://arxiv.org/abs/2511.21631` | HTTP 200，`<title>[2511.21631] Qwen3-VL Technical Report</title>` |
| 本战役基座 LLM **36 层 / hidden 2560** | `<ckpt>/config.json`（`checkpoint-4976`） | `num_hidden_layers=36`, `hidden_size=2560` |
| 视觉塔 **24 层 / hidden 1024** | 同上 `vision_config` | `depth=24`, `hidden_size=1024` |
| **DeepStack 注入的是 LLM 第 0/1/2 层** | `transformers/models/qwen3_vl/modeling_qwen3_vl.py:862` 的 `layer_idx in range(len(deepstack_visual_embeds))`，且 `deepstack_visual_indexes=[5,11,17]`（len=3） | 成立。视觉特征取自视觉塔第 5/11/17 层，注入到 LLM 前 **3** 层 |
| BLIP-2 learnable queries | `curl https://arxiv.org/abs/2301.12597` | HTTP 200，标题相符 |
| InstructBLIP instruction-aware Q-Former | `curl https://arxiv.org/abs/2305.06500` | HTTP 200，标题相符 |
| 库几何 PCA **15 / 28 / 99** 维（90%/95%/99% 方差） | 本仓库六份提案逐字一致的实测协议（`EPR-024:659` 等）：9³ 网格、2187 维、1137 个 lut_id | 引用准确 |
| zcache 已支持 `(n, K, 2560)` 三维缓存 | `q3vl/whatb/zcache.py:224,333,393-422` 的 `k_rows` | 成立，evidence bank 不需要新建存储层 |

**「第 3 层」这个取值的依据**：`hidden_states` 元组是 37 个元素（idx 0 = embedding 输出，
idx 1..36 = 各层输出）。DeepStack 在 layer_idx 0/1/2 之后注入，所以
**`hidden_states[3]` 是第一个吃到全部三次视觉注入的层输出**。

---

## 1. 数据

沿用 EPR-030 口径层 `q3vl/whatb/caliber.py`，不新建第二套。

| 项 | 值 |
|---|---|
| 训练集 | `--data v2seg+l8`，`split=train` 且 `winner_confidence=normal`，**n = 119,828** |
| LUT 池 | **本轮实测澄清口径**：train 全体（含 low）uniq lut_id = **3,149**；**train normal-only = 3,081**；`v2seg+l8` = **3,166**。预设库总量 3,522 |
| 选型集 | V_what normal-only **n = 567** |
| 终测 | T_final n=533 / T_lut_unseen n=252（normal-only，LUT-id 与 train 交集 **0**），**本 EPR 不碰** |
| canonical code 拟合集 | **train 全体 3,149 条**。主 agent 2026-08-16 先裁定 3,081（normal-only）、**同日改判为 3,149**：PCA 是无监督基，多 68 条只出现在 `low` 里的 LUT 只增强基的代表性，且 `c*` 按 lut_id 索引，训练时自然只取到可见的 3,081 行 —— 不构成泄漏也不造成口径不一致 |
| 评测 LUT 与拟合集的关系 | **实测：V_what 的 531 条、T_final 的 577 条 lut_id 是 train 3,149 的 100% 子集**，所以「V/T 的 LUT 一条都不进 PCA」在 n=3,149 上不可满足。裁定：`--exclude-eval-luts` **保持 off**（开了拟合集缩到 2,056，基更差）。真正无泄漏的终测是 **T_lut_unseen，实测与 train 交集 = 0**，守卫只对它硬失败 |

V_where / V_what / T_final 永不进训练。切分走 sha1 规则族。

---

## 2. 模型（伪代码）

```text
Frozen Qwen3-VL-4B
  └─ evidence bank:  layers {3,12,24,36} × token types {seg_color, color_span_pool,
                     instruction_pool, reasoning_pool}  → 最多 16 × 2560
                                │
                    per-layer projection P_l : 2560 → 256  (+ layer emb + type emb)
                                │
                  ┌─────────────▼─────────────┐
                  │  What Evidence Resampler  │  8 queries × 256, 2 blocks, 8 heads, FFN 1024
                  └─────────────┬─────────────┘
                     Z_what ∈ R^{8×256}
                                │
                  ┌─────────────┴─────────────┐
                  ▼                           ▼
          canonical LUT code ĉ ∈ R^192    latent tokens Z_what
                  │                           │
                  └─────────────┬─────────────┘
                  ┌─────────────▼─────────────┐
                  │ Anchored Primitive Decoder│  48 RGB-anchored slots + 1 global slot
                  │  d=256, 4 blocks, 8 heads │  cross-attn 的 K/V 是 Z_what
                  └─────────────┬─────────────┘
                       per-slot shared heads
                                │
                    EPR-028 R1 的 A1 / A3 参数
```

### 2.1 Evidence bank

```python
e[l,t] = P_l(RMSNorm(h[l,t])) + emb_layer[l] + emb_type[t]      # (16, 256)
```
每层独立 projection（第一版不共享，避免强迫不同层落在同一统计空间）。

### 2.2 Resampler

```python
q[0] = W_seg @ concat(h_seg[3], h_seg[12], h_seg[24], h_seg[36])   # 主决策 token，不随机
q[1:8] = learnable                                                  # 7 个自由 query
for _ in range(2):
    Q = Q + XAttn(Q, E); Q = Q + FFN(Q)
Z_what = Q                                                          # (8, 256)
```

reasoning 与 image 都是**零初始化门控的旁路**，不是主信息流：

```python
Z = Z + gamma_reason * XAttn(Z, H_reason)      # gamma_reason(0) = 0，训练期 dropout 0.3~0.5
Z = Z + gamma_img    * XAttn(Z, H_image)       # gamma_img(0)    = 0
```

### 2.3 Anchored Primitive Slot Decoder

```python
a[i] ∈ [0,1]^3                       # 第 i 枚 Gaussian 的固定 RGB anchor（uniform_grid_positions）
q[i] = emb_slot[i] + phi(a[i])       # i = 0..47
q[g] = emb_global                    # 第 49 个 slot
for _ in range(4):
    Q = Q + SelfAttn(Q); Q = Q + XAttn(Q, Z_what); Q = Q + FFN(Q)
theta[i] = theta0[i] + scale[i] * tanh(Head(q[i]))     # Head 逐参数组共享权重
```

`theta0` = identity carrier 的稳定初值（EPR-028 R1 §8.4 的那组）；`scale` 是每类参数的允许幅度。

每 slot 输出（A3 口径，27/基元）：

| 参数组 | 维/基元 | 参数化 |
|---|---|---|
| `μ_rgb` | 3 | anchor + bounded residual，`‖Δμ‖_∞ ≤ 0.1` |
| `μ_s` | 1 | bounded |
| conditional Cholesky `L_C` | 6 | FP32 激活 |
| `β` | 3 | bounded，`‖β‖_∞ ≤ β_max`，**最后解冻** |
| `τ` | 1 | `0.1 + 0.9·σ(r)` |
| opacity | 1 | sigmoid |
| local affine residual | 12 | **最先训练** |

global slot 输出 12 维 global affine residual。A1 口径去掉 `μ_s / β / τ`，per-prim 22。

---

## 3. 数学公式

### 3.1 Canonical LUT function code（本 EPR 的核心定义）

对每条 **train** LUT 在固定颜色网格上取 residual：

```
r_ℓ = vec{ L_ℓ(x) − x }_{x ∈ X}          X = 17³ 均匀 sRGB 网格 ⇒ dim(r_ℓ) = 14,739
c*_ℓ = WhitenedPCA_192(r_ℓ)              PCA 只在 train 的 3,149 条上拟合
```

并排报 9³ 网格（2187 维）的对照。**本轮实测见 §10.2**：90% 维数复现为 15，
95% 为 29（底账 28），99% 为 109（重抽 1,166 条）/ 120（全池 3,149 条），底账记的是 99。
底账的抽样 seed 与 PCA 向量空间（sRGB 还是 Lab）均未记录，本轮未试 seed 去凑 1,137。

`c*` 是**固定的、有函数意义的坐标**，不是任意可学习 embedding —— 后者存在任意旋转/缩放/
重参数化，会让 VLM 去学一个任意坐标系。

### 3.2 损失

```
L_func   = ‖ f_{D(ĉ)}(x,s) − [x + s(L_ℓ(x) − x)] ‖₁            # 函数值，主项
L_code   = Huber( ĉ_ℓ , c*_ℓ )                                  # 规范坐标回归
L_sameLUT= E_ℓ E_{r_a,r_b ~ ℓ} ‖ c(r_a) − c(r_b) ‖₁            # 同 LUT 不同 record 的一致性
L_metric = | ‖ĉ_a − ĉ_b‖₂ − κ·D_LUT(L_a, L_b) |                # latent 距离 ∝ LUT 函数距离
L_proxy  = CE( W_proxy ĉ , lut_id )                             # 辅助项，不参与推理
```

conditioner 阶段（O3 / R 系列）的总损失：

```
L = L_func + 0.5·L_code + 0.1·L_sameLUT + 0.05·L_metric
```

`L_proxy` 预注册权重 **0**（默认关），是一条消融行。
`D_LUT(L_a, L_b)` = 两条 LUT 在 9³ 网格上的 ΔE76 均值（与仓库既有实测协议同一口径）。

### 3.3 采样

carrier / decoder 阶段（O0–O2）：每步 256 条 LUT × 2048 色 × 4 个 s-anchor（EPR-028 R1 §8.1）。
conditioner 阶段（O3 / R 系列）：**same-LUT paired batch**

```
16 LUT × 2 records/LUT = 32 records      两条 record 的 source image 尽量不同
```

---

## 4. 优化器参数

| 项 | 值 |
|---|---|
| 优化器 | Adam，`--base-lr 1e-3`，cosine，无 warmup，wd=0（EPR-030 冻结口径） |
| lr 依据 | EPR-030 §3.1：色批 2,097,152 上 lr 1.6e-2 七行七死；{1e-3, 4e-3, 3e-4} 零死 |
| geometry 分级 lr | `μ_rgb` / `β` 解冻时用 **0.1×** base lr |
| grad clip | 1.0 |
| 精度 | VLM cache / projection / MLP 可 bf16；covariance 激活、Cholesky、Mahalanobis、logdet、高斯权重、RGB→Lab **强制 FP32**（EPR-028 R1 §7） |
| 分级解冻 | ① local/global affine → ② opacity, `μ_s`, `τ` → ③ conditional covariance → ④ 0.1× lr 解冻 `μ_rgb`, `β` |

**待用户裁定**：是否加 StatLUT 原配方的 AdamW wd=0.05 + 5-epoch 线性 warmup（未自行改）。

---

## 5. 四级容量分解（本 EPR 的主结果表）

这是把「训练不好」拆成四个可归因来源的唯一手段。四行**同 carrier、同步数、同评测集**。

| 级 | 条件来源 | decoder | 回答 |
|---|---|---|---|
| **O0** Direct carrier table | 无（每条 LUT 一组独立参数 `Θ ∈ R^{3149×D_θ}`） | 无 | N=48 的 carrier 本身能不能拟合全部 LUT |
| **O1** Free latent + shared decoder | 自由学习 `e_ℓ ∈ R^192` | slot decoder | shared decoder 容量够不够 |
| **O2** Canonical code + shared decoder | 固定 `c*_ℓ`（§3.1） | slot decoder | 192 维规范坐标够不够 |
| **O3a** VLM readout + frozen decoder | `ĉ = R(H_VLM)` | **冻结**的 O2 decoder | VLM 读到 LUT 信息没有 |
| **O3b** VLM readout + tuned decoder | `ĉ = R(H_VLM)` | 从 O2 权重出发、允许微调 | 同上，去掉 decoder 的噪声脆性 |

**O3 必须是两行，不能只有冻结那一行**：O2 的 decoder 是在干净的 `c*` 上训的，
O3 喂进去的是带预测噪声的 `ĉ` —— decoder 从没见过带噪 code。只跑 O3a 的话
`Δ_readout` 里同时混着「VLM 读不准」和「decoder 对 code 噪声不鲁棒」两件事，
而后者是本 EPR 自己造出来的，不是 VLM 的问题。两条配套措施：

1. **O2 训练时对 `c*` 注入噪声**：`c̃ = c* + σ·ε`，`ε ~ N(0, I)`，`σ` 取 O1 阶段
   实测的 `‖ĉ − c*‖` 量级（预注册三档 `σ ∈ {0, 0.5σ̂, 1.0σ̂}`，默认 `0.5σ̂`）。
2. **并排报 O3a 与 O3b**：`Δ_readout = E_O3b − E_O2` 是 readout 的归因量，
   `E_O3a − E_O3b` 单独一列，量的是 decoder 的噪声脆性。

预注册的四个差分（只列数字，不下结论）：

```
E_O0                      # carrier 地板
Δ_decoder = E_O1 − E_O0   # generator 容量
Δ_code    = E_O2 − E_O1   # latent 定义/维度
Δ_readout = E_O3b − E_O2  # VLM readout（用 tuned decoder 那行，见上）
Δ_brittle = E_O3a − E_O3b # decoder 对 code 噪声的脆性，单独一列
```

`E` 的定义：V_what normal-only 567 条上，17³ 网格的 ΔE00 均值（`grid_de00_mean`），
分 `s ∈ {0, 0.25, 0.5, 0.75, 1}` 五列报。**O0–O2 是 oracle（见 GT lut_id），
`published=false`、板上写 `oracle_reference=true`，不可与任何 arm 的 headline 并排读**
（EPR-028 R1 §9.1）。只有 O3 出完整板（headline + 五平凡基线 + 三负控制）。

---

## 6. 预注册结构消融（叠加式，一行只改一个变量）

### 6.1 Readout 系列（decoder 固定为 O2 训好的那个，全部冻结）

| 行 | VLM readout | LUT code | decoder |
|---|---|---|---|
| **R0** | last-layer `<seg_color>` 单向量（**现行结构**） | 64 | 3×128 flat MLP |
| **R1** | layers {3,12,24,36} 的 `<seg_color>` 标量混合 | 192 | 固定 slot decoder |
| **R2** | R1 + color span pooling + instruction pooling | 192 | 固定 slot decoder |
| **R3** | 8-query What Resampler | 192 | 固定 slot decoder |
| **R4** | R3 + 零初始化门控 image evidence | 192 | 固定 slot decoder |
| **R5** | R3 + 最后 6~8 个 LLM block 的 rank-8/16 LoRA（q/k/v/o） | 192 | 固定 slot decoder |

**R0 → R3 之间禁止一次同时换读出、换 64→192、换 flat MLP→slot decoder**，
否则整段提升不可归因。R1/R2/R3 是逐级叠加。

### 6.2 Generator 系列（code source 固定）

| 行 | code source | decoder |
|---|---|---|
| **G0** | free LUT embedding | 6×512 residual MLP |
| **G1** | free LUT embedding | 48-slot primitive decoder |
| **G2** | PCA-192 | 48-slot primitive decoder |
| **G3** | PCA-256 | 48-slot primitive decoder |
| **G4** | PCA-192 | slot decoder + 8-expert FFN, top-2 routing, load-balancing loss |

`d_LUT` 预注册三档 **128 / 192 / 256**，默认 **192**。
**G4 是条件行**：只在 O1（free-code oracle decoder）仍明显欠拟合时才跑。
**禁止 3149/4000 个 expert**（会退化成隐式 LUT-ID 查表，对 unseen LUT 更差）。

### 6.3 图像证据的两档

| 档 | LUT code 读出 |
|---|---|
| Text-first | instruction、color span、`<seg_color>`，**不直接读 image tokens** |
| Multimodal residual | Text-first + 零初始化 image cross-attention（R4） |

判据：若加入 image evidence 后 `within-LUT variance` 上升且 T_lut_unseen 不提升，删除 image 分支。

### 6.4 LoRA 的准入条件（R5 不得提前跑）

全部满足才准提交：① O2 好；② O3 明显差；③ 单 token / 多 token / 多层读出三档都已出数；
④ `L_code` 与 `L_sameLUT` 已稳定；⑤ frozen readout 仍降不下 oracle gap。
视觉塔一律冻结，`<seg_color>` embedding 可训。

---

## 7. 执行顺序（gate）

| Gate | 内容 | 卡 | 通过条件 |
|---|---|---|---|
| **C0** | 建 canonical code：3,149 条 train LUT 的 17³ residual → whitened PCA-192，落盘 + 复现 9³ 的 15/28/99 | CPU | 重建误差与解释方差表落盘；`c*` 的 sha256 冻结 |
| **C1** | **O0** direct carrier table（无 VLM、无 decoder） | 小 | 收敛、零 NaN；`E_O0` 落盘 |
| **C2** | **O1** free latent + slot decoder（3 seed） | 小 | `Δ_decoder` 落盘 |
| **C3** | **O2** canonical code + slot decoder，扫 `d_LUT ∈ {128,192,256}` | 中 | `Δ_code` 落盘；decoder 冻结存档 |
| **C4** | evidence bank 重生成（layers {3,12,24,36} × 4 token types） | 两卡数小时 | 缓存 `k_rows=16`、sha256、抽样回放断言 |
| **C5** | **O3 / R0→R3** readout 逐级叠加 | 中 | `Δ_readout` + 完整板 |
| **C6** | R4（image gate）、R5（LoRA，须满足 §6.4）、G4（须满足 §6.2） | 中 | 条件行 |

**C1–C3 完全不需要 VLM、不需要 z 缓存、不需要图像**，只需要预设库 `luts.npz`，
显存是百 MB 级 —— 可与在跑作业共存挂载（显存规则：已占 + 新任务峰值 < 65 GB）。
**C4 的缓存重生成是本 EPR 唯一的大额算力开销（约 13.5 GB 产物）**，
排在 `Δ_code` 出数之后：若 `Δ_code` 已经很大，加宽 readout 不解决问题，C4 可以不做。

---

## 8. 判据与遥测

判据键不动：headline 取 `.contexts.all.headline_normal_only`（**禁用顶层 pooled `.baselines`**）、
五条平凡基线（B0 8.2926 / B1 7.6323 / B2 10.0989 / B3 6.1553 / B4 0.8253）、三负控制、
12 个预注册键 + **每键运行时断言**。**AUC 禁用**；checkpoint 选择禁 val loss；
禁逐图 min-max；跨行比较必须步数匹配（U4）。

本 EPR 追加的必报列：

| 列 | 含义 |
|---|---|
| `within_lut_var` | 同一 lut_id 不同 record 的 `ĉ` 方差（§3.2 的 `L_sameLUT` 是它的训练侧对偶） |
| `oracle_gap` | `E_O3b − E_O2`（`E_O3a − E_O3b` 另列，见 §5） |
| `code_recon_de00` | `c*` 经 PCA 逆变换重建 LUT residual 的 ΔE00（code 的信息上限，纯几何量） |
| `gamma_reason` / `gamma_img` | 两个门控的当前标量值（是否被学起来） |
| `slot_attn_entropy` | 48 个 slot 对 8 个 latent token 的注意力熵（是否塌成全看同一个 token） |
| `beta_absmean`（A3） | cross term 是否收缩到零 |
| `null_mass_mean` / `cholesky_info_nonzero` / `tau_p05/p50/p95` / `gnorm` | EPR-028 R1 §10 的稳定性遥测 |

采信纪律：**每次** quick eval 都复查 `L_rec` 是否 NaN/Inf（不只首次）。
NaN 预测的 ΔE00 算出来是 0.0 而非 NaN，不复查就会发布 `headline = 0.0` 的假板。

结果表规则：测什么指标只展示该指标数字；消融只用叠加式写法；**禁下任何结论、
禁一切揣测性表述**。

---

## 9. 待用户裁定（未静默拍板）

1. §4 的 warmup / wd。
2. `d_LUT` 默认 192（报告建议值），预注册扫 128/192/256 —— 是否接受 192 作为主档。
3. canonical code 的网格：主档 17³（14,739 维）+ 9³ 对照，还是反过来。
4. C4（evidence bank 重生成，约 13.5 GB / 两卡数小时）的排期 —— 本提案排在 `Δ_code` 出数之后。
5. `L_proxy`（LUT-ID 分类辅助项）预注册权重 0，是否要开一条非零行。

---

## 10. C0 实测结果（2026-08-16）

产物：`/home/bc/data/runs/whatb/lutcode_17c_pca/{code.npz, manifest.json}`，
`code.npz` sha256 `9a06ab9aaf84b6da4d5961bd43b61d1780f7f59dc7ba108dbda7542096c1311e`，37,153,964 B。
float64；LUT 求值 10.5 s、SVD 20.7 s、合计 71.7 s。

### 10.1 主档 17³（3,149 条 LUT，14,739 维）

累计方差 90% / 95% / 99% 所需维数 = **14 / 28 / 117**（rank 上限 3,148，保留 256 个成分，
总方差 432.5652358606418）。

`code_recon_de00`（17³ 网格、ΔE00 均值、对 GT LUT）：

| `d_LUT` | unclamped | clamped [0,1] |
|---|---|---|
| 128 | 1.4121292022147434 | 1.3986988927131874 |
| 192 | 1.0885775465429532 | 1.0787805691491754 |
| 256 | 0.8978493688937816 | 0.8900381266275258 |

### 10.2 对照 9³（2,187 维）

| 池 | n_lut | 90% | 95% | 99% |
|---|---|---|---|---|
| 既有底账（六份提案逐字记录） | 1,137 | 15 | 28 | 99 |
| 本轮重抽（seed 20260810，2,500 行） | **1,166** | 15 | 29 | 109 |
| 全 train 池 | 3,149 | 15 | 29 | 120 |

重抽得到 1,166 个 id，与底账的 1,137 不等：**底账的抽样 seed 从未记录**，
本轮**没有去试 seed 凑 1,137**。底账的 PCA 向量空间（sRGB 还是 Lab）同样未记录。

### 10.3 C1 dry-run（rc=0）

| stage | carrier | 池 | b × q × 4 = 每步 | 步数 | mining | `D_θ` | `Θ` 参数量 |
|---|---|---|---|---|---|---|---|
| S1 | A1 | 1 | 1×2048×4 = 8,192 | 2,000 | 否 | 1,068 | 1,068 |
| S2 | A1 | 32 | 32×512×4 = 65,536 | 4,000 | 是 | 1,068 | 34,176 |
| S2 | A3 | 32 | 32×512×4 = 65,536 | 4,000 | 是 | 1,308 | 41,856 |

S1 在 CPU 上 50 步实测：A1 **2.0824 s**（41.6 ms/步）、A3 **2.3426 s**（46.9 ms/步）。

测试：`test_whatcodec_c0c1.py` **37 passed**；`q3vl/whatb/tests` 全量 **788 passed**。

### 10.4 EPR-028 R1 G1 实测（同日，carrier 侧的对照读数）

单 LUT overfit、A1、oracle、2,000 步：`grid_de00_mean` **0.2594478905200958**；
分 s：`s=0` **0**、`s=0.25` **0.12694013118743896**、`s=0.5` **0.2558041214942932**。
`L_rec` 0.0393543615937233 → 0.0017211547819897532；`R_line` 恒 0；
`cholesky_info_nonzero` 0；`opacity_p50` 0.46781861782073975；`void_reason` null。
G2 起跑读数：A2@138 `L_rec` 0.010402662679553032、`null_mass_mean` 2.02926045744789e-07、
`beta_absmean` **null**；A3@104 `L_rec` 0.013794191181659698、
`null_mass_mean` 1.7438620147913753e-07、`beta_absmean` **0.08424273878335953**。

显存实测（`nvidia-smi --query-compute-apps`）：G2/A2 **1452 MiB**、G2/A3 **1452 MiB**
（65,536 色/步）。G3（2,097,152 色/步）的 `--mem-peak` **待冒烟实测，禁外推**。

---

## 11. C1 / O0 全量结果（2026-08-17）

`--level O0 --stage S3`：全 3,149 条 train LUT，每条一组独立 carrier 参数（无 generator、
无 VLM），18,760 步，2,097,152 色/步（256 LUT × 2048 色 × 4 个 s-anchor），
seed 20260810，pure L1 + `R_line`（λ=0.1）。评测：17³ 均匀 sRGB 网格 × 5 个 s 值，
逐 LUT 聚合，**n = 3,149**。

| s | A1（3D explicit gate，`D_θ`=1068） | A3（4D joint，`D_θ`=1308） | A3 − A1 |
|---|---|---|---|
| 0 | 0 | 0.13438700737562 | +0.13438700737562 |
| 0.25 | 0.13747021374812 | 0.29191215464472 | +0.15444194089660 |
| 0.50 | 0.28829179641934 | 0.45569843628715 | +0.16740663986781 |
| 0.75 | 0.46157536628736 | 0.67702913437915 | +0.21545376809179 |
| 1.00 | 0.68552007660726 | 0.97133202078611 | +0.28581194417885 |
| **mean（`E_O0`）** | **0.31457149061242** | **0.50607175069455** | **+0.19150026008213** |

末尾 `L_rec`：A1 0.002534511499107、A3 0.004330772906542。
A3 末行遥测：`beta_absmean` 0.09337211400270、`tau_p50` 0.44187074899673、
`opacity_p50` 0.50492781400681、`null_mass_mean` 1.9083435631728e-07、
`cholesky_info_nonzero` 0。两条 `void_reason` 均为 null，`published=false`、
`oracle_reference=true`。

规模对照（同 A1、direct table）：S2（32 LUT，4,000 步）0.24604636839358；
S1（1 LUT，2,000 步）0.33747351765632。

同量纲参照（17³ ΔE00）：`code_recon_de00` @ `d_LUT` 128/192/256 =
1.41212920221474 / 1.08857754654295 / 0.89784936889378。

---

## 12. 两条待修的规格漏洞（2026-08-17 实测暴露）

### 12.1 `Δ_decoder` 目前算不出来：两侧评测口径不同

§5 要求四级分解「同 carrier、同步数、**同评测集**」，但实现落成了两套：

| 侧 | 评测集 | 聚合单位 | n |
|---|---|---|---|
| EPR-031 `O0`（`E_O0`） | 3,149 条 train LUT 的 17³ 网格 | per-LUT | 3,149 |
| EPR-028 R1 `G3`（`grid_s*`） | V_what 的 record | per-record | 567 |

两者的 `grid_s*` 名字相同、量纲相同（ΔE00），但集合与聚合单位都不同，
**相减无意义**。本轮实测的 G3_A1 五档平均 1.19679722803821 与
O0_S3_A1 的 0.31457149061242 之差 0.88222573742579 **不是 `Δ_decoder`**，不得引用。

修法（未做）：让两侧共用一个评测入口，或给 O0 增加一条 V_what per-record 的并行列。

### 12.2 `B4_oracle` 的定义没有跟着 R1 §5 的形成式一起改

| 口径 | 形成式 | `B4_oracle` 实测 |
|---|---|---|
| E031（EPR-030） | `Î = (1−α)I + α·f̂(I)` | 0.82529450538100 |
| EPR-028 R1 A1（G3） | `Î = f(I, S)` | 2.97129814026835 |

E031 口径下把 `f̂` 取成真 LUT 会还原出 GT，所以 B4 接近 0；R1 口径下 B4 的 `f̂`
不吃 `s`，于是 `Î = L(I) ≠ GT`。**R1 §5 改形成式时没有同步重新定义 B4_oracle**，
两个口径的 B4 不可并排读。

修法（未做）：R1 口径下的 B4 应定义为 `f̂(x,s) = (1−s)x + s·L_ℓ(x)`（即把真 LUT
按同一条生成律接上 s 轴），否则这条基线在 A1–A3 上没有意义。
