## 要验证的结论

如果这个实验成功，我们就能说「**零训练的 CLIP self-self 稠密读出在 S-val 上能给出 AUC≥0.80、
且真正跟着目标名词走的空间场**，所以它是所有 VLM 读出臂（RO-9/RO-3/RO-2/RO-5）必须超过的免费基准线，
H2『图文模型里有 where』至少在 CLIP 这一侧成立」；失败（AUC<0.75）就不能说这句，
只能说免费读出的天花板不够，where 必须外挂 DINO（RO-4）或靠训练拿。

## 为什么需要验证它

G1 刚判 Gate D1 FAIL，被证伪的是「GL token attention 这个**读法**」而不是 H2；
不先立起一条能跑通的零训练读出，审稿人会直接问「你们连免费的 CLIP 稠密读出都拿不到 where，
凭什么说 VLM 里有 where？」——而且没有 RO-1 这条线，RO-9/RO-3 的任何数字都无从判强弱
（EXPERIMENTS_v3 的 RO-9 淘汰判据字面就是「弱于 RO-1 → 降级为 analysis」）。

## 怎么验的

把 SCLIP / NACLIP / ClearCLIP 三种末层 attention 手术实现在**同一份 OpenAI CLIP ViT-B/16 权重**上，
在 **S-val 的 237 个样本**（D-CONSTRUCT L1 语义掩膜 23 + G1 区域对立批的 SAM3 主体掩膜 214）上，
以「patch 特征 × 目标名词文本嵌入的**原始余弦**（无任何逐图归一化）」为 s 场，
逐算子 × 逐后处理档（伪影三件套 A1/A2/A3 + guided filter）报 ROC-AUC，
并配四条对照列（跨图错位名词 / 全体共用固定名词 / 亮度 / 常数场）。

---

# REPORT — RO-1 · self-self 零训练读出（EXPERIMENTS_v3 §2.1）

- **日期** 2026-08-03 ｜ 分支 `lens-exp` ｜ commit `bc888fe147af76f2f8ef0299c85de856b9cd4adf`
- **一句话结论**：**Gate = PROMOTE**。最优 = **ClearCLIP + 伪影三件套**，两个评分集 median AUC
  **0.939 / 0.930**（判据 ≥0.80），且**跟着指令走有正面证据**（配对 Δ_shuffle = +0.42 / +0.28，
  p = 2.6e-5 / 5.5e-17）。

> ### 本实验最该被记住的一句话
> **「改末层 attention」这一步值 +0.70 AUC，「怎么改」只值 0.02。**
> 未改造 CLIP 的稠密余弦图对目标掩膜 AUC = **0.214**（系统性反相关，96.7% 样本 <0.5）；
> 三个算子把它抬到 0.90–0.93，**彼此只差 0.014–0.028、CI 重叠、换短名词口径后排名翻转**。
> 所以「SCLIP/NACLIP/ClearCLIP 谁强」不构成可靠排序，选实现最简单的即可
> （ClearCLIP = 一行 `softmax(qqᵀ)` + 丢残差丢 FFN）。

- **⚑ 主 agent 定案（2026-08-03）**：① **U-RO1-2 — 主协议用长描述，短名词作对照**
  （理由：SFT 指令里出现的就是长描述，主协议必须与下游真实使用一致）；
  ② **C2 采纳 — RO-4 判据改为「在 RO-1 失败的那 18% 子集上 +≥0.05」**，
  该子集已落盘 `config/ro4_failure_subset.json`（45 源：SAM3 41 + L1 4，
  RO-1 在其上的 median AUC = **0.663**（SAM3 段）/ 0.637（全体）），RO-4 直接取用；
  ③ scache `meta.norm` 已补显式 `domain` 字段（见 §十三）。
- **后续**：RO-X1 的 CLIP 侧半场（有名词/无名词分离）已作为追加任务单独交付于
  `experiments/ROX1_clipside_20260803/` —— 它回答的是本实验逼出来的那个问题：
  **既然免费的 CLIP 就能做到 0.93 且随指令变，为什么还需要 VLM？**

## 一、设置

| 项 | 值 |
|---|---|
| 骨干 | **OpenAI CLIP ViT-B/16**（`/home/bc/data/models/openai_clip/ViT-B-16.pt`，sha256 `5806e77c…f416f`，与官方 URL 内嵌期望哈希一致），fp32，冻结 |
| 三算子 | `sclip` = arch `vanilla` + attn `csa`（末层保残差+FFN）｜ `naclip` = arch `reduced` + attn `naclip`, std=5 ｜ `clearclip` = arch `reduced` + attn `clearclip`（qqᵀ，丢残差丢 FFN）。**跑在同一份权重与同一份预处理上**，逐行出处见 `tools/readout/clip_naclip/VENDOR.md` 与 `config/vendored_model_py.diff`（相对 NACLIP 上游只有 3 处改动，其余文件 `cmp` 逐字节相同） |
| 对照臂 | `vanilla` = 未改造 CLIP 末层（**不是**三算子之一，作零点参照） |
| s 定义 | `s = cos(patch_feat, text_emb)`；`patch_feat` = `encode_image(return_all=True)` 丢 CLS 后 L2 归一（三个官方 segmentor 的同一个量）；`text_emb` = 80 条 `openai_imagenet_template` 集成（逐模板 encode→L2→均值→再 L2，NACLIP `naclip.py:30-42` 逐字口径）。**全程无逐图归一化**（红线） |
| 输入 | 短边 448、保长宽比、裁到 patch 整数倍、**整图前向**（voc21 的 `slide_crop=0` 档）；patch 原生网格 ~28×42。336 全量复跑一遍作稳健性行 |
| 评分集 | **S-val 237 样本**：`construct_l1` = D-CONSTRUCT L1(S-val) 23（T4 冻结 sanity 批，split 由 T1 旁表逐条背书；1/24 因 journal 无主体描述剔除）＋ `subject_sam3` = G1 区域对立批 214 源 × veradata 银行 `cache/subject/.subject.png`（= D-SFT-L(S-val) 的 SAM3 主体掩膜档）。**复用 `g1_region_opp.json`，未重新采样** |
| 目标短语 | 主口径 = l 系 journal 的 `local.subject.description`（长描述）；`subject.name`（短名词）作并列列全程同报 |
| 标签口径 | 掩膜面积均值下采到评分网格，**≥0.5 判正**（G1 同口径）；AUC = 归一化 Mann-Whitney U（与 `analyze_g1.py::roc_auc` 同实现，已对 `sklearn.roc_auc_score` 校验，200 个含大量并列的随机用例最大差 1.1e-16） |
| 计算 | **仅卡 1**（`CUDA_VISIBLE_DEVICES=1`），**显存峰值 1.72 GB**（限额 20 GB），wall-clock 78 min（其中全分辨率 guided filter 块 64 min），`job.marker` 有 PID/命令/日志 |

## 二、预注册判据 vs 实测（并排）

判据、算子配置、后处理阶梯、选优规则、viz 选例规则**全部写死在 `run_ro1.py` 顶部常量**
（`PREREG` / `OPERATORS` / `RUNGS` / `VIZ_RULE`），跑数前固定。

最优 = **`clearclip` · `+A3+A1+A2`**（选优规则：三算子×六档中，**两个评分集 median AUC 的最小值**最大者；
全 18 格得分见 `metrics.json → best.all_scores`）。

| 评分集 | n | 预注册晋级 | 预注册淘汰 | **实测 median AUC** | bootstrap CI95 | AUC≥0.80 的样本占比 | 判定 |
|---|---|---|---|---|---|---|---|
| D-CONSTRUCT L1 (S-val) | 23 | ≥0.80 | <0.75 | **0.939** | [0.832, 0.966] | 0.783（18/23） | ≥0.80 达成 |
| SAM3 主体掩膜 (S-val) | 214 | ≥0.80 | <0.75 | **0.930** | [0.901, 0.945] | 0.757（162/214） | ≥0.80 达成 |

**Gate 判定：PROMOTE**（预注册规则：两集均 ≥0.80 → 晋级；两集均 <0.75 → 转 RO-4；否则判「分裂」交主 agent）。
淘汰线 AUC<0.75 的样本：construct_l1 4/23、subject_sam3 41/214 —— **逐样本仍有约 18% 落在淘汰线下**，
中位数高不等于每张图都好（失败案例见 §八）。

### 表述纪律（D-31）—— 这次的 PASS 各自属于哪一类

| 结论 | 类别 |
|---|---|
| 两集 median AUC ≥0.80 | **正面证据**（场与掩膜对齐，且逐样本 75–78% 单独达标） |
| 符号检查通过 | **未触发死刑**（只排除「场整体反号、要靠事后翻转才能用」这一种失败模式） |
| Δ_luma = +0.47 / +0.41 | **未触发死刑**（排除「s 是亮度马甲」，不构成有用性证据） |
| **Δ_shuffle = +0.42 / +0.28，配对 p=2.6e-5 / 5.5e-17，胜率 0.83 / 0.73** | **正面证据**（把别的图的名词扣上去，场就掉到 AUC 0.52 / 0.65 —— 场确实跟着指令走）。**这正是 RO-9 缺的那一列** |
| A2（周期性陷波）Δ≈0 | **阴性诊断**，按 D-0 自身的失败判据即「此骨干无此病，跳过」 |
| guided filter Δ ≤ +0.002 | **阴性**：在本评分口径下边界精修没有可测收益（不等于视觉上无用，见 §七） |

## 三、符号检查（预注册，禁事后翻转）

符号约定 `s = +cos(patch_feat, text_emb)` 写死在代码里，**代码中没有任何 flip 分支**。
通过线 = 中位 AUC>0.5 且逐样本 AUC>0.5 的占比 ≥0.90。

| 评分集 | 算子 | median AUC(base 档) | 逐样本 AUC>0.5 占比 | 判定 |
|---|---|---|---|---|
| construct_l1 | sclip | 0.934 | 0.957 | **PASS** |
| construct_l1 | naclip | 0.927 | 0.957 | **PASS** |
| construct_l1 | clearclip | 0.926 | 1.000 | **PASS** |
| construct_l1 | vanilla | 0.296 | 0.130 | **FAIL**（符号系统性反向） |
| subject_sam3 | sclip | 0.890 | 0.925 | **PASS** |
| subject_sam3 | naclip | 0.907 | 0.925 | **PASS** |
| subject_sam3 | clearclip | 0.913 | 0.939 | **PASS** |
| subject_sam3 | vanilla | 0.214 | 0.033 | **FAIL**（符号系统性反向） |

**三算子的符号稳定，不需要任何事后翻转。** `vanilla` 的 FAIL 不是 bug 而是结论：
**未改造 CLIP 的稠密余弦图与目标掩膜系统性反相关**（AUC 0.214，96.7% 的样本 AUC<0.5）——
这正是 SCLIP/NACLIP/ClearCLIP 三篇论文共同的出发点，本实验独立复现了它。
**按纪律它保持原样报出，不翻转**（若翻转会得到 0.786 的假象，那是事后拟合符号）。

## 四、三算子 × 后处理阶梯（主协议 448，patch 原生网格，median AUC(s,M)）

每行都带 Δ_shuffle / Δ_fixed / Δ_luma / Δ_const 四列（红线要求）。
Δ_const 恒等于 median AUC − 0.5（常数场 AUC ≡ 0.5，定义使然，只作口径声明）。

| 算子 | 档 | L1 (n=23) | SAM3 (n=214) | Δ vs base (SAM3) | Δ_shuffle | Δ_fixed | Δ_luma | Δ_const |
|---|---|---|---|---|---|---|---|---|
| **sclip** | base | 0.934 | 0.890 | +0.000 | 0.369 | 0.349 | 0.430 | 0.390 |
| sclip | +A1_outlier | 0.933 | 0.902 | **+0.012** | 0.375 | 0.331 | 0.441 | 0.402 |
| sclip | +A1+A2_notch | 0.933 | 0.902 | +0.012 | 0.375 | 0.330 | 0.441 | 0.402 |
| sclip | +A3_ttreg | 0.931 | 0.890 | +0.000 | 0.308 | 0.321 | 0.430 | 0.390 |
| sclip | +A3+A1 | 0.941 | 0.894 | +0.004 | 0.307 | 0.301 | 0.434 | 0.394 |
| sclip | +A3+A1+A2 | 0.941 | 0.894 | +0.004 | 0.307 | 0.300 | 0.434 | 0.394 |
| **naclip** | base | 0.927 | 0.907 | +0.000 | 0.265 | 0.148 | 0.447 | 0.407 |
| naclip | +A1_outlier | 0.934 | **0.916** | **+0.009** | 0.276 | 0.122 | 0.456 | 0.416 |
| naclip | +A1+A2_notch | 0.934 | 0.916 | +0.009 | 0.276 | 0.121 | 0.455 | 0.416 |
| naclip | +A3_ttreg | 0.933 | 0.912 | +0.005 | 0.305 | 0.149 | 0.452 | 0.412 |
| naclip | +A3+A1 | 0.931 | 0.913 | +0.005 | 0.291 | 0.134 | 0.452 | 0.413 |
| naclip | +A3+A1+A2 | 0.931 | 0.911 | +0.004 | 0.290 | 0.133 | 0.451 | 0.411 |
| **clearclip** | base | 0.926 | 0.913 | +0.000 | 0.252 | 0.099 | 0.453 | 0.413 |
| clearclip | +A1_outlier | 0.944 | 0.923 | +0.010 | 0.243 | 0.076 | 0.462 | 0.423 |
| clearclip | +A1+A2_notch | 0.945 | 0.923 | +0.010 | 0.243 | 0.074 | 0.463 | 0.423 |
| clearclip | +A3_ttreg | 0.936 | 0.928 | **+0.015** | 0.288 | 0.112 | 0.468 | 0.428 |
| clearclip | +A3+A1 | 0.939 | 0.930 | +0.016 | 0.276 | 0.096 | 0.469 | 0.430 |
| **clearclip** | **+A3+A1+A2（最优）** | **0.939** | **0.930** | **+0.016** | **0.276** | 0.096 | 0.469 | 0.430 |
| vanilla（对照） | base | 0.296 | 0.214 | +0.000 | −0.134 | −0.068 | −0.246 | −0.286 |
| vanilla | +A3+A1+A2 | 0.267 | 0.219 | +0.005 | −0.183 | −0.105 | −0.241 | −0.281 |

**读法：**

1. **三算子排名（SAM3 集，各自最优档）：ClearCLIP 0.930 > NACLIP 0.916 > SCLIP 0.902。**
   差距 0.014–0.028，bootstrap CI 有重叠，**「ClearCLIP 最强」只能当弱排序，不能当强断言**；
   L1 集上三家几乎打平（0.931–0.945），换短名词后 NACLIP 与 ClearCLIP 打平（0.962 vs 0.961，§七）。
2. **真正的鸿沟在「改不改末层 attention」，不在「怎么改」**：未改造 CLIP AUC = 0.214，
   三算子相对它的增量是 **+0.69 ~ +0.72**，比它们彼此之间的 0.02 大一个半数量级。
3. **伪影三件套的贡献很小但为正**：A1（高范数 outlier 剔除+4 邻域插值）在三算子上一致
   **+0.009 ~ +0.012**；A3（测试时寄存器）只对 ClearCLIP 有效（**+0.015**），对 SCLIP 是 0 或负；
   A2（周期性陷波）**全部 ≈ 0.000**（诊断阴性，见 §五）。三件套叠加天花板 **+0.016**。
4. **Δ_fixed 是三算子分化最大的一列**：SCLIP 0.300、NACLIP 0.133、ClearCLIP 0.096。
   全体共用的固定短语（`"object"`）在 ClearCLIP 上就能拿到 AUC 0.834 —— 说明
   **ClearCLIP 的场里「主体显著性」成分最重、「名词特异性」成分相对最轻**。
   选臂时要警惕：ClearCLIP 的 AUC 领先里，有一部分是「它更会找主体」而不是「它更会听指令」。
   **Δ_shuffle 才是听指令的证据，在这一列上 SCLIP（0.307）反而最高。**

## 五、D-0 伪影三件套的诊断（决定该不该做，不是做完再说）

| 评分集 | outlier 占比（中位） | 末层前 hidden norm 最大值（中位） | Nyquist ratio（中位） | A2 判阳性？ |
|---|---|---|---|---|
| construct_l1 | 0.083 | 96.1 | 0.518–0.750 | **否** |
| subject_sam3 | 0.080 | 95.8 | 0.530–0.684 | **否** |

- **高范数伪影确实存在且很强**：末层前 hidden 范数中位约 14，最大值约 96（>6×），
  `median+3·MAD` 判出 **8% 的 patch 是 outlier**。这解释了 A1 一致为正。
  register neuron 检测在 **100/100 张 S-train 图**上都找到超阈 token（阈值 30，
  官方 `configs/openai_clip_base.yaml`），发现的神经元 =
  `{L5: [924,2256,1541,112,2562], L4: [1606,447,722], L3: [803,2884]}`；
  用 20 张与 100 张图检出结果**完全相同**（稳定）。
- **周期性伪影不存在**：预注册阳性线 `nyquist_ratio > 2.0`，实测 0.52–0.75（棋盘格分量功率
  **低于**同环带中位）。按 D-0 自身的失败判据「outlier 不降且 s 场无差别 → 此骨干无此病，跳过」，
  **A2 这一件在 CLIP ViT-B/16 上判为不适用**；DVT 相应**未做**（需逐图优化，与「零训练一次前向」
  的 RO-1 定义冲突，见 NOTES §五 U-RO1-5）。

## 六、指令条件性 —— 本臂相对 RO-9 的关键分离

### 6.1 跨图错位（shuffle）负控制：**通过，且幅度巨大**

| 评分集 | AUC(自己的名词) | AUC(别的图的名词) | 中位配对差 | Wilcoxon p | 胜率 |
|---|---|---|---|---|---|
| construct_l1 (n=23) | 0.939 | **0.516** | **+0.314** | **2.6e-5** | 0.826 |
| subject_sam3 (n=214) | 0.930 | **0.653** | **+0.220** | **5.5e-17** | 0.729 |

把别的源的主体描述扣到本图上，场立刻掉到接近随机（0.52）或明显下滑（0.65）。
**对照 G1 的 RO-9：ρ_shuf 0.895 ≥ ρ_syn 0.842、AUC 配对差 +0.0005（p=0.121）。**
同一条负控制上，RO-1 的分离与 RO-9 的零分离形成直接对比 —— 这是
「被证伪的是读法不是假设」的实证。

### 6.2 AUC_target（G1 / RO9-L 口径，混池两种指令条件；最优档）

`AUC_target = median{ AUC(s_desc, M) , AUC(s_comp, ¬M) }`；与指令无关 ⟺ 塌到 0.5。

| 算子 | 子集 | n 对 | AUC(s_desc, M) | AUC(s_comp, M) | **AUC_target** | CI95 | 配对 p |
|---|---|---|---|---|---|---|---|
| sclip | background | 165 | 0.895 | 0.564 | **0.736** | [0.666, 0.802] | <1e-4 |
| naclip | background | 165 | 0.896 | 0.477 | **0.755** | [0.691, 0.818] | <1e-4 |
| clearclip | background | 165 | 0.925 | 0.669 | **0.736** | [0.667, 0.788] | <1e-4 |
| sclip | spatial | 49 | 0.863 | 0.337 | **0.787** | [0.741, 0.826] | <1e-4 |
| naclip | spatial | 49 | 0.947 | 0.520 | **0.789** | [0.677, 0.879] | <1e-4 |
| clearclip | spatial | 49 | 0.941 | 0.685 | **0.689** | [0.592, 0.824] | <1e-4 |
| vanilla | background | 165 | 0.235 | 0.469 | **0.349** | [0.322, 0.381] | <1e-4 |

**AUC_target 0.69–0.79，全部显著高于 0.5**（对照：RO-9 全 24 层是 0.479–0.526）。
但它明显低于 `AUC(s_desc, M)` 的 0.90+，原因在 `¬M` 那一半：CLIP 对「the background」
「the right side of the image」这类**非实体指称**定位能力弱（`AUC(s_comp, M)` 中位 0.48–0.69，
理想应远小于 0.5）。这是 **prompt 可定位性**的问题，不是读出方法的问题 —— 已在
NOTES §五 U-RO1-4 登记为待决策项。**因此本臂的指令条件性主证据用 6.1 的 shuffle，
AUC_target 作为与 RO-9 横向可比的第二证据。**

## 七、稳健性与并列口径

| 表 | 结论 |
|---|---|
| **输入尺度 336 vs 448**（SAM3 集，最优档） | sclip +0.009 / naclip −0.006 / clearclip −0.001。**排名不翻转**（336 上 clearclip 0.928 > naclip 0.905 > sclip 0.903），尺度不是结论的驱动因素 |
| **短名词 vs 长描述**（最优档） | 短名词**一致更好**：SAM3 集 naclip **+0.050**、clearclip **+0.031**、sclip +0.022；L1 集 naclip +0.035、clearclip +0.025。⚠️ 主口径（长描述）**低估**了 RO-1 的天花板；用短名词时 NACLIP **0.962** / ClearCLIP **0.961** —— **排名在短名词口径下翻转（ClearCLIP 领先变成打平），CI 重叠，三算子差异不构成可靠排序**。**⚑ 主 agent 定案（U-RO1-2）：主协议用长描述**（与 SFT 指令的真实措辞一致），短名词保留为对照 |
| **16×16 网格**（与 RO-9 同网格） | clearclip **0.941**（SAM3 集）vs **RO-9 canonical L8–15 = 0.648 / 最好层 L22 = 0.834**。即使拿 RO-9 最有利的层比，RO-1 仍高 **+0.107**；而 RO-9 那 0.834 还是**与指令无关**的主体显著性 |
| 亮度基线 | AUC_luma 中位：L1 集 0.559 / SAM3 集 0.461（与 G1 报的 0.465 一致） |

## 八、全分辨率块（D-3 收尾：bilinear vs guided filter）

s 场从 patch 网格（约 28×42）上到原图分辨率（中位 1.5 MP，最大 6.4 MP），
`kornia.filters.guided_blur(guidance=原图, eps=1e-4, subsample=4)`。

| 评分集 | 算子 | bilinear | +guided filter | Δ |
|---|---|---|---|---|
| construct_l1 | sclip | 0.961 | 0.963 | **+0.0022** |
| construct_l1 | naclip | 0.948 | 0.947 | −0.0005 |
| construct_l1 | clearclip | 0.950 | 0.950 | +0.0004 |
| subject_sam3 | sclip | 0.920 | 0.921 | +0.0013 |
| subject_sam3 | naclip | 0.930 | 0.932 | +0.0011 |
| subject_sam3 | clearclip | 0.948 | 0.948 | −0.0001 |
| subject_sam3 | vanilla | 0.182 | 0.182 | −0.0008 |

**guided filter 在 AUC 口径下的收益 ≤ +0.002，可判为噪声级。** 两点必须说清楚：

1. AUC 是**全图逐像素排序**指标，绝大多数像素远离边界，边界带的改善被稀释 ——
   **本表不能用来断言「guided filter 没用」**，只能说「在全图 AUC 口径下测不出来」。
   PLAN §2.2 D-3 的失败判据本来就是「**边界 PSNR** 回收 <0.05 dB」，那要等 RD 臂接上渲染器
   才测得了；本轮无渲染器，故 **D-3 判据未被检验**（如实登记，不算过也不算不过）。
2. 也正因如此，**FeatUp / LoftUp 本轮不做**：零参的 guided filter 都还测不出增量，
   先上学习式上采样器没有判据支撑（FeatUp 调用签名已核实，随时可接，见 NOTES §1.5）。

## 九、可视化（`viz/`，选例规则跑数前写死在 `VIZ_RULE`）

面板 = 原图 / GT 掩膜 / s(目标名词) / s(**错位**名词) / s(补集词)，后三格叠 GT 掩膜红色轮廓并标 AUC；
**色标是原始余弦值**（无逐图归一化，故各格色标范围不同，这正是「禁逐图归一化」的可视化后果）。
筛选池 = SAM3 集主体面积 4%–60%（178/214 源）。

| 前缀 | 张数 | 含义 |
|---|---|---|
| `success_*` | 4 | 最优档 AUC 最高（0.992–0.996） |
| **`failure_*`** | 4 | 最优档 AUC 最低：**0.084 / 0.108 / 0.153 / 0.270** —— 全部低于随机，即**场压在主体的补集上** |
| **`failure_instrblind_*`** | 4 | `AUC(错位名词) ≥ AUC(目标名词)` 的源（0.153 / 0.450 / 0.464 / 0.483）——「没跟着指令走」的具体样本 |
| `success_l1_*` / `failure_l1_*` | 2 + 2 | L1 集两端（0.994 / 0.978 与 0.520 / 0.521） |

**典型失败归因**（`failure_00_auc0.084_ppr10k_0860_a.png`，AUC=0.084）：一位女性坐在球场草地上、
身后是铁丝网。目标名词是这位女性，但 s 场把高值全给了**铁丝网与草地纹理**，主体区反而是全图最低。
四张失败图的共性：**主体是人、背景有强纹理或强语义竞争物**（铁丝网、栏杆、密集草地），
CLIP 的 patch 特征被纹理主导。这与 §四第 4 点一致 —— 偏「显著性」的场遇到
「显著的不是目标」时就整体翻车，而不是温和退化。

## 十、结论

1. **Gate = PROMOTE**（正面证据）。零训练 CLIP self-self 读出在 S-val 上 median AUC
   **0.930 / 0.939**，超过 0.80 晋级线；**RO-1 作为基准线成立**，后续所有 VLM 读出臂的
   AUC 必须与 **0.930（patch 网格）/ 0.941（16×16）/ 0.961（短名词）** 对表。
2. **三家谁强：ClearCLIP ≳ NACLIP > SCLIP，但这是弱排序**（Δ 0.014–0.028，CI 重叠，
   换短名词后 NACLIP 与 ClearCLIP 打平）。**真正的结论是「改末层 attention」这一步值约 +0.70 AUC，
   「怎么改」只值 0.02。** 实际含义：三家任选其一都行，选实现最简单的
   （ClearCLIP = 单行 `softmax(qqᵀ)` + 丢残差丢 FFN）。
3. **免费读出的天花板约 0.93–0.96**（长描述 0.93，短名词 0.96）。
4. **伪影三件套净收益 +0.016**：A1 稳定 +0.01，A3 只对 ClearCLIP 有效（+0.015），
   **A2 为 0（诊断阴性，按 D-0 自身判据判为此骨干无此病）**。值得做（几乎零成本），但不是胜负手。
5. **guided filter 在 AUC 口径下无可测收益（≤+0.002）；D-3 的正式判据（边界 PSNR）本轮未被检验。**
6. **指令条件性拿到了正面证据**（Δ_shuffle 配对 p=5.5e-17，胜率 0.73），这正是 RO-9 缺的那一列。
   **H2「图文模型里有 where」在 CLIP 一侧成立**；G1 证伪的确实只是「GL token attention 这个读法」。
7. **仍有约 18% 的样本落在淘汰线以下**（41/214 < 0.75，31/214 < 0.70），
   且失败时是**整体翻转**（AUC 0.08–0.27）而非温和退化。中位数漂亮不等于可以裸用；
   下游若要当掩膜，必须配一个「这张图的读出可不可信」的门。

## 十一、建议下一步（指向 EXPERIMENTS_v3 的具体行）

| # | 建议 | 指向 |
|---|---|---|
| C1 | **RO-1 行标记为已晋级**，把 **0.930（patch 网格）/ 0.941（16×16）/ 0.961（短名词）** 三个数写进该行当作后续所有 RO 臂的对表基线 | §2.1 RO-1 行 |
| C2 | **复核 RO-4 的晋级判据**：「比 RO-1 AUC +≥0.02」在 0.93 基线上意味着要做到 0.95，头顶只剩 0.04–0.07。建议改成「在 RO-1 失败的那 18% 样本子集上 +≥0.05」——那才是 DINO 该发力的地方，也才是一次额外前向值不值的真问题 | §2.1 RO-4 行 |
| C3 | **RO-9 行**：本实验给出了「弱于 RO-1」的量化证据 —— 同网格（16×16）下 RO-1 0.941 vs RO-9 最好层 0.834，且 RO-9 的 0.834 与指令无关。按 RO-9 行自身的淘汰判据，**降级为 analysis 的依据已闭合** | §2.1 RO-9 行 |
| C4 | **给所有 RO 臂加一列强制 Δ_shuffle（配对检验）**：本实验证明这一列能一眼分开「真读出」（+0.28，p=5e-17）与「假读出」（RO-9 的 −0.015，p=0.155） | §2.1 表头 / E23 负控制行 |
| C5 | **RO-X2（归一化四档）可以直接开跑**：本臂 s 场已全量落进 scache（`ro1-clearclip-l11`，237 条，全局固定仿射），换归一化档零成本 | 表下 RO-X2 |
| C6 | **短名词 vs 长描述需要主 agent 定一次口径**：短名词一致 +0.02~+0.05，但 SFT 指令里出现的是长描述。若 RO 系统一改短名词，本轮所有数字整体上移，RO-X1（有名词/无名词分离）的设计要跟着调 | NOTES §五 U-RO1-2 |
| C7 | **失败样本子集值得单列**：41 个 AUC<0.75 的源里失败模式是「强纹理背景吃掉主体」。这既是 RO-4（DINO 边界/语义）的靶子，也是 RO-5 探针头最该学的部分 | RO-4 / RO-5 行 |
| C8 | **IMPL_DOSSIER 需要 5 处订正**（gem_torch 的 API bug、gem_depth 语义、min-max 可关、open_clip pin 收紧、pkg_resources） | NOTES §三 → DOSSIER §5.1 / 附录 B |
| **C9** | **⚠️ 本报告的 0.930 需要与 RO-X1 的结果一起读**：`experiments/ROX1_clipside_20260803/` 实测——一条**对所有图相同**的无名词短语 `"the main subject"` 就能拿到 **AUC 0.907**，与本报告的 0.930 **统计上不可区分**（配对差 +0.009，p=0.133）。即 **RO-1 的 AUC 里，指令带来的净增量不显著**；同条件下 **AUC_target 塌到 0.523**。**RO-1 的强项是"找主体"不是"听指令"**，这条限定必须与 0.930 一同引用 | §2.1 RO-1 行 + 表下 RO-X1 |

## 十二、本实验自身的局限（供结果审阅 agent 核）

- **L1 集只有 23 个样本**，bootstrap CI 宽达 [0.832, 0.966]；gate 在该集上的「达成」功效很低，
  真正承重的是 SAM3 集（n=214，CI [0.901, 0.945]）。
- **两个评分集的掩膜语义不同**：L1 是 l 系生产 build 的 `semantic-*` 槽 C_GT（软边二值化），
  SAM3 集是主体软掩膜银行。两者在 G1 的 79 源交集上 IoU 0.913 / ρ 0.997，但不是同一分布。
- **AUC_target 的 `¬M` 侧受限于 prompt 可定位性**（§6.2），该指标在本臂上偏低是 prompt 的账，
  与 RO-9 横向比较时要同时看 6.1。
- **guided filter 的正式判据（边界 PSNR）未测**，本轮只有 AUC 口径的阴性结果。
- **GEM 未入榜**（环境风险，NOTES §二 D / §五 U-RO1-6），故「training-free 读出」的
  搜索空间不是穷举的；GEM 改的是最后 `depth−1` 层，与本轮三算子（只改末层）不同档。
- **未测下游收益**：本轮只测「s 落在哪、跟不跟指令」，没测「s 当渲染条件时的 PSNR/ΔE00」。

## 十三、交付物

```
experiments/RO1_selfself_20260803/
  REPORT.md                  # 本文件
  NOTES.md                   # 核实记录（三个官方仓库逐行 + arXiv 2506.08010 + open_clip pin）
                             #  §三 与 DOSSIER 不符的 5 条 ｜ §五 待主 agent 决策 6 条 ｜ §六 红线自查
  STATUS.md
  metrics.json               # 2.8 MB：sides{448,336} × sets × ladder(4 算子×6 档×5 条件)
                             #  + auc_target + fullres + diagnostics + sign_check + gate + per_case 237 行
  job.marker                 # PID / 完整命令 / 卡 1 / 日志路径
  build_scoring_set.py       # 评分集构造（journal 反查主体描述 + SAM3 掩膜银行 + 确定性错排）
  run_ro1.py                 # 一次跑出全部数字与图（预注册常量在文件头）
  report_tables.py           # 只读 metrics.json 渲染本报告的表
  viz/success_* (4+2)  viz/failure_* (4+2)  viz/failure_instrblind_* (4)
  config/scoring_set.json    # 237 行评分集快照（含 shuffle donor）
  config/env.json            # commit / seed / 环境 / 显存峰值 / 权重 sha256 / 预注册规则
  config/vendored_model_py.diff   # vendored CLIP fork 相对上游的全部改动（104 行）
  config/upstream_commits.json    # 四个上游仓库的 commit
  config/run_ro1.log  config/nohup_ro1.log

tools/readout/clip_naclip/   # vendored OpenAI-CLIP fork（NACLIP MIT）+ VENDOR.md（逐行出处）
tools/readout/ro1_selfself.py# 算子库 + D-0 三件套 + 测试时寄存器 + 评分口径

/var/cache/veradata/scache/ro1-clearclip-l11/   # 237 × (npy + meta.json)，32×32 fp16
```

**scache arm**：`/var/cache/veradata/scache/ro1-clearclip-l11`（arm 名 = `ro1-<算子>-<层>`，
层 = ViT-B/16 最后一个 resblock，0-index L11）。
`norm = {"kind":"global-affine","per_image":false,"mu":0.24908,"sigma":0.06722,
"base":"cosine(patch_feat, text_emb_80template)"}` —— **μ/σ 是整臂两个常量**，
反算 `cos = z*σ+μ`；**不是逐图归一化**。每条 meta 另含 `operator/rung/backbone/
input_short_side/native_grid/img/target_phrase`，RD 臂换个 arm 名即可零成本接入。
