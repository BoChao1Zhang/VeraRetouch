## 要验证的结论

如果 G1b 成功，我们就能说「G1 读不出指令依赖是**读法**的问题——换成 DiffLMM 的
attend-and-segment 读出（同一模型、同一批样本、同一套判据、前向逐比特不变），
同一份注意力里就能读出随指令改变聚焦的空间场」；失败就不能说这句，
也就不能再把「信息在里面、只是被 sink 淹了」当作 G1 负结果的解释。

## 为什么需要验证它

整个项目的出发点是「SFT 后 special token 的 attention 里涌现了空间定位信息，只是被 sink 了」，
而这个观察本身就来自 DiffLMM 的读出方式；G1 用廉价 canonical 读出没读出来（Gate D1 FAIL），
于是「假设错 / 读法错 / 代码错」三解并存——不把读法这一支单独证否或证成，
论文的 where 通路主张就悬在空中，审稿人一句「你换个读法试过吗」即可击穿。

## 怎么验的

在 G1 的**同一批源与同一批指令**上重跑读出（不重新采样），一次前向同时导出
**pre-softmax（G1 口径）与 post-softmax + Eq.3 归一化（attend-and-segment 口径）**，
做 8 变体消融梯子 × 3 个 special token，并新增两条 G1 没有的对照：
**匹配噪声地板**（同区域、只换句框）与 **prompt 顺序 2×2**（指令在图之后 / 之前）。

---

# REPORT — G1b / RO-D · DiffLMM（attend-and-segment）读出重跑 G1

- **实验编号**：G1b / RO-D（EXPERIMENTS_v3 §2.1 RO-9 行的读法变体；PLAN_v2 §3 Gate D1 复检）
- **日期**：2026-08-03 ｜ 分支 `lens-exp` ｜ 卡 1
- **判据口径**：**与 G1 逐行一致**——`analyze_g1b.py` 直接 `import` G1 的
  `pearson / spearman / roc_auc / SubjectMaskBank / luma_valid_from_image`，
  唯一变化的自变量是**读出方式**（以及新增的 prompt 顺序因子）
- **数据**：G1 的 `g1_region_opp.json` / `g1_samples.json` / `g1_shuffle_ctrl.json`
  **原样复用**，零重新采样、零 ad-hoc 切分
- **数字来源**：本目录 `metrics.json`。引用 G1 的数字一律以
  `experiments/G1_s_identifiability_20260803/metrics.json` 为准（不引文档转述）

## 〇、被复现方法的核实（任务卡第 1 优先事项）

完整记录见 `NOTES.md §一`（每条给出实际打开过的 URL）。三条最要紧的：

### 0.1 论文身份 —— 任务卡口头描述有两处需更正

| 项 | 核实结果 |
|---|---|
| 标题 | **Emergent Visual Grounding in Large Multimodal Models Without Grounding Supervision** |
| 作者 | Shengcao Cao, Liang-Yan Gui, Yu-Xiong Wang（UIUC） |
| arXiv | **2410.08209**（v1 2024-10-10 / v2 2025-10-16） |
| venue | **ICCV 2025 Findings**（arXiv abs 页 Comments 字段） |
| 先前被拒 | ICLR 2025 投稿，旧题 *Emerging Pixel Grounding in ...*（OpenReview `UFKC0lMTdK`） |
| 官方仓库 | https://github.com/Shengcao-Cao/groundLMM （Apache 2.0；`aas/` = attend-and-segment） |

- **更正 1**：不是 ECCV，是 **ICCV 2025 Findings**（先投 ICLR 被拒属实）。
- **更正 2（更重要）**：**「DiffLMM」是论文两个贡献之一，而且不是那个读出方法。**
  - **attend-and-segment（A&S）= 读出方法**，不改架构、不加训练 —— **这才是「DiffLMM 的读出方式」**；
  - **DiffLMM = 模型**（视觉编码器换成 diffusion 特征与 CLIP 拼接
    `V = concat(V_SD, V_CLIP) + PE`，重训 projector）—— 另一个模型，**套不到 VeraRetouch**（要重训）。

  本实验做的是 **A&S 读出**，REPORT 全篇按此措辞。

### 0.2 ⚑ 一票否决核实项：A&S 是**纯读出**，前向逐比特不变

论文原文：`"we propose attend-and-segment, a simple yet effective method for grounding LMMs
without changing their architecture or requiring additional training"`。
官方 `aas/infer_attn.py` 唯一的模型调用是一次普通
`model.generate(..., output_attentions=True, return_dict_in_generate=True)`；
没有 hook 改激活、没有额外模块、没有多步去噪。归一化也在读出侧（`aas/gcg.py`：
`attn_mean = attentions.mean(dim=0); attentions = attentions - attn_mean`）。

⇒ **任务卡「若改前向则必须补基座模型对照」的分支不触发**；「读出 vs 创造」在机制层面被排除，
不只是靠论文的一句自述。负责人「非常影响推理」的直觉指向的是**开销**
（要导出全层全头 attention + SAM 分割），不是前向被改 —— **开销大 ≠ 改前向**。

### 0.3 A&S 的确切定义（按官方代码逐行核对）

设输出（生成）序列 `o_1..o_r`，image token 有 `h×w` 个：

| 步 | 论文定义 | 官方代码 |
|---|---|---|
| 1 | `A_i^raw ∈ [0,1]^(n_layer × n_head × …)`：生成 `o_i` 时的注意力 | `output_ids['attentions'][i]` |
| 2 | 只取 image token 列，**对全部层与全部头求平均** → `A_i^reduced ∈ [0,1]^(h×w)` | `.mean(dim=(0,1))` |
| 3 | **Eq.3**：`A_i^norm = A_i^reduced − (1/r) Σ_j A_j^reduced` | `attentions - attentions.mean(dim=0)` |
| 4 | 取最大值点当 point prompt 喂 SAM 出掩膜 | `aas/gcg.py`（**本实验不用这步**，要的是连续 s 场） |

**Eq.3 就是「解 sink」的那一步**：论文附录说明动机是
`"uninformative visual tokens (usually in background) attract more attention"`、
这些伪影在输出序列上 `"relatively stable"` —— 沿输出 token 轴求平均正好把共模估计出来减掉。
论文 Table 4 消融：去掉这一步 GCG mask recall 从 **46.4 → 43.9**。

**与 G1 canonical 的差异**（本实验要隔离的自变量）：pre-softmax → **post-softmax**；
L8–15 → **全部 24 层**；无归一化 → **Eq.3 减均值**；外加 query 行约定
（token 自身位置 vs 产生该 token 的前一位置）。八变体消融梯子见 §二。

**偏离说明**：未装官方仓库（它是 LLaVA fork，装了只会引版本冲突）；Eq.1–3 全部照搬，
只省略 Eq.4 之后的 SAM point-prompt 步骤（不影响 ρ / AUC 判据）。

## 一、三个 special token 与 causal 可见性（任务卡第 2 项）

### 1.1 字面与 id（`added_tokens.json` + tokenizer 运行时实测，两处一致）

| # | 字面 | id | 实测 |
|---|---|---|---|
| 1 | `<retouch_light>` | **151646** | 单 token ✓ |
| 2 | `<retouch_color&temp>` | **151647** | 单 token ✓ |
| 3 | `<retouch_colormixer>` | **151648** | 单 token ✓ |

三者在同一次前向里同时导出，**逐 token 单独报**（§三）。

### 1.2 ⚑ 它们排在指令**之后** —— causal 可见性的明确答案

实测 prompt（style 模板，74 token）：

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
<image>
<Style_Retouch_Task>
Now, you are acting as a Retouch Agent. I will provide an image and an instruction, please give me a retouch plan and retouch tokens.
 Instruction: {instruction}<|im_end|>
<|im_start|>assistant
```

- **三个 retouch token 在 prompt 里一个都没有**（`151646/151647/151648 in prompt_ids → False×3`），
  全部在 **assistant 生成段**产出，实测位于生成序列**末尾且相邻**
  （典型 `gen_len=311` 时 `light=307, colortemp=308, colormixer=309`）。
- ⇒ **因果掩码下它们完全看得见指令。**「special token 排在指令之前所以看不到指令」
  这个架构性解释**被证否**，不能用来解释 G1 的失败。
- 附带含义：`prev` 与 `self` 两种 query 行约定在这三个相邻 token 上只差一个位置，
  **高度重叠、不是独立证据**（两者都报时须知）。

### 1.3 ⚑ 但受限的是 **key 侧**：`<image>` 排在指令**之前**

实测 image token span = **(14, 270)**，指令在其后（**与 RO-2 报的 span 逐位一致**）。
因果掩码下 image token 的 hidden/key **只依赖（系统前缀 + 图像）**、与指令无关
（RO-2 数值侧证实：跨图打乱指令下 image 表示 ρ = 0.99995，bit 相同输入 ρ = 1.000000）。

**两条结论合起来才是完整图景：**

> **query 侧（三个 special token）看得见指令；key 侧（image token）看不见。**
> 指令条件性必须、且只能经由 query 向量进入：`s = Kᵀq`，`K` 是这张图的冻结矩阵。

**A&S 读的对象在当前 prompt 下是否可能含指令信息**（主 agent 明确要求 REPORT 标注）：

- **A&S 读的不是 image token 的表示，而是 attention** —— instruction-**dependent** 的 query
  与 instruction-**independent** 的 key 之间的双线性形式。
  **所以它不属于「读一个本身不含指令信息的对象」那一类**，那堵墙不能先验判它死刑。
- 但那堵墙确实把指令条件性压成**低秩**：每层每头只有一个 64 维 query 可用。
- **旁证**：A&S 原论文的 LLaVA prompt 同样是 `<image>\n{question}`（image 在前），
  在同一堵墙后面仍做到 GCG mask recall 44.2、超过全监督的 GLaMM
  ⇒ **「image 在前」本身不足以解释读不出来。**
- 为把这个因素**实测**掉而不是靠论证，本实验加了 **prompt 顺序 2×2**（§四）。

## 二、设置

| 项 | 值 |
|---|---|
| 模型 | `/home/bc/data/models/VeraRetouch`（llava_qwen2 0.5B + mobileclip_l_1024，bf16，**eager attention**） |
| 红线守卫 | ①加载时断言 `config._attn_implementation == "eager"`；②前向后断言 `attentions is not None`；③patch 内断言取到的 post-softmax 权重非 None；④断言 24 层全捕获、image span == 256、展开长度一致。**任一不满足直接 raise，不回退** |
| 归一化 | **无任何逐图 min-max / softmax 归一化**。Eq.3 减均值是被检验方法自身的定义；**未减均值的原始分量与均值图全部原样落盘**，任何人可复算未归一化版本 |
| 落盘精度 | **float32**（post-softmax 概率 ~1e-3、减均值残差 ~1e-5，fp16 会把 A&S 信号量化掉） |
| 掩膜 | veradata `cache/subject` 的 SAM3 主体软掩膜，与 G1 **同一 bank、同一 `luma_to_grid` 对齐路径** |

### 2.1 读出变体（消融梯子，同一次前向，零额外 GPU 成本）

| 变体 | softmax | 层 | query 行 | Eq.3 | 含义 |
|---|---|---|---|---|---|
| `canon` | pre | L8–15 | self | ✗ | **G1/RO-9 原口径** |
| `pre_all` | pre | 全 | self | ✗ | 只换层 |
| `post_all` | post | 全 | self | ✗ | A&S 的 `A_i^reduced`（未归一化） |
| **`aas`** | post | 全 | **prev** | ✓ | **忠实 attend-and-segment** |
| `aas_self` | post | 全 | self | ✓ | 行约定对照 |
| `post_canon_norm` | post | L8–15 | self | ✓ | 层口径对照 |
| `pre_norm` | pre | 全 | self | ✓ | softmax 是否必要 |
| `pre_canon_norm` | pre | L8–15 | self | ✓ | **G1 口径 + 只加 Eq.3**（最小改动） |

### 2.2 ⚑ 读出链路正确性硬核验（不是声称，是实测）

`canon` 变体与 G1 落盘 npz **逐格对拍**（`metrics.crosscheck_vs_G1_canonical`）：
**n=40，Pearson 中位 `0.99999992`，max_abs_diff 中位 `0.0039`**（logit 值域 ~[−13, 1]，纯 bf16 噪声）。

⇒ **「同一批样本、同一套判据、唯一变量是读法」这个前提已被证实。**
同时排除了「抓错行 / 列 / 层」这一类实现错误。

### 2.3 判据（预注册，分析前写定于 `NOTES.md §三`）

| 判据 | 阈值 | 角色 |
|---|---|---|
| **ρ_region_opp 中位** | **< 0.3** | **主判据**（沿用 G1 D-17，口径不变） |
| Δρ_region_opp（A&S − canonical）配对 | 中位 + bootstrap 95% CI + Wilcoxon | **核心产出** |
| **AUC_target**（pooled 中位） | 显著 > 0.5 | **唯一单解度量** |
| **ρ_floor（匹配噪声地板）** | 应显著高于 ρ_region_opp | **新增判别量** |
| ρ_syn / \|ρ_Y\| | >0.7 过、<0.5 死 / <0.5 过、>0.8 死 | 稳定性 / 非亮度马甲 |
| ρ_shuf | 应显著 < ρ_syn | **负控制，必跑** |

**AUC_target 定义**：`reg_a` 点名主体 ⇒ 目标 = M，`AUC_target = AUC(s_a, M)`；
`reg_b` 点名补集 ⇒ 目标 = ¬M，`AUC_target = AUC(s_b, ¬M) = 1 − AUC(s_b, M)`；两条件 pool 取中位。
**s 与指令无关 ⟺ 中位塌到 0.5**（此时 `AUC(s,M)` 与 `1−AUC(s,M)` 关于 0.5 对称）。
`region_b_kind=background` 子集上 `reg_b` 的目标**字面就是 ¬M**，正表两者都报。

### 2.4 新增对照（应主 agent 反馈，均写在分析前）

**（1）⚑ 匹配噪声地板 `reg_a_para`** —— 保持**区域串与方向词逐字不变**，只重写句框：

| 指令 | 文本模板 |
|---|---|
| `reg_a` | `Please {d} {region}, keeping the rest of the image unchanged.` |
| `reg_a_para`（新增） | `Without altering any other part of the photo, please {d} {region}.` |
| `reg_b` | `Please {d} {complement}, keeping the rest of the image unchanged.` |

- `ρ_floor` = corr(s(reg_a), s(reg_a_para))：**换句框、同区域** → s 若追区域应**高**
- `ρ_region_opp` = corr(s(reg_a), s(reg_b))：**同句框、换区域** → s 若追区域应**低**
- **`sep_matched = ρ_floor − ρ_region_opp`（配对）= 本实验最干净的判别量**

表面形式扰动实测（词级 Levenshtein，`config/g1b_config_report.json`）：
`reg_a↔reg_a_para` = **15 词（恒定）**；`reg_a↔reg_b` = 中位 **12**、均值 13.6。
⇒ **地板侧扰动更大，是保守（偏难）的地板。**

这条同时回答主 agent §四 的质疑「判据 0.3 是否从设计上不可达」：
G1 用 `ρ_syn`（两句**完全不同的自然指令**）当参照，与只差一个名词的 reg 对**根本不可比**；
本实验给出的是**表面形式扰动同阶**的地板。

**（2）去共模差分场 AUC**：报 `AUC(s(reg_a) − s(reg_b), M)`，
配**同区域对照** `AUC(s(syn_a) − s(syn_b), M)`（主 agent §三）。

**（3）跨 token ρ 当动态范围参照**：同图同指令、只换 query token（主 agent §四）。

**（4）生成文本落盘**：`meta.gen_text` 逐样本入 npz（主 agent §五 的可观测性补丁）。

## 三、结果

**数据量**：**1182 次读出，0 次失败**。
`region 428（214 源 ×2）/ floor 214 / region_instr_first 240（120 源 ×2）/ floor_instr_first 120 /
syn 120（60 源 ×2）/ shuf 60`。image_first 列的 n **= G1 的 214 源，逐源一致**。

### 3.1 主表：{prompt 顺序} × {读出方式}（`<retouch_light>`）

| 格 | ρ_region_opp | n<0.3 | **ρ_floor**（匹配地板） | **sep_matched** | **AUC_target[cgt]** | AUC_target[sam3] | 差分场 AUC[cgt] | fallback |
|---|---|---|---|---|---|---|---|---|
| image_first × **canon**（=G1） | **0.884** | **0/214** | 0.878 | **−0.001** | **0.499** | 0.507 | 0.507 | 0.028 |
| image_first × **aas**（A&S） | **0.439** | 35/214 | 0.445 | **+0.001** | **0.496** | 0.496 | 0.517 | 0.028 |
| instr_first × canon | 0.883 | 0/120 | 0.876 | −0.002 | 0.505 | 0.508 | 0.512 | 0.017 |
| instr_first × **aas** | 0.592 | 13/120 | 0.621 | +0.027 | 0.503 | 0.469 | 0.530 | 0.017 |

> `canon × image_first` 实测 **ρ_region_opp = 0.884**，与 G1 `metrics.json` 的 0.884 **逐位一致**；
> 加上 §2.2 的逐格对拍（Pearson 0.9999999），**本实验与 G1 是同一条读出链路**已被双重证实。

### 3.2 ⚑ 判决所依据的三个数

**(1) A&S 确实把 ρ 打下来了 —— 但地板同步塌陷，所以那不是「聚焦」**

| 变体 | Δρ_region_opp vs canonical（配对） | 95% CI | Wilcoxon p |
|---|---|---|---|
| `aas` | **−0.423** | [−0.444, −0.402] | 8.8e-37 |
| `aas_self` | −0.126 | [−0.143, −0.102] | 2.3e-24 |
| `pre_canon_norm`（只加 Eq.3） | −0.363 | [−0.404, −0.316] | 7.3e-37 |

ρ 掉了 0.42，**看上去像巨大进步**。但**匹配噪声地板**（同区域、只换句框，且表面形式扰动**更大**）
同步掉到 0.445：

| 变体 | **sep_matched = ρ_floor − ρ_region_opp**（配对） | 95% CI | p | 逐源 sep>0 占比 |
|---|---|---|---|---|
| `canon` | **−0.0009** | [−0.0090, +0.0057] | 0.79 | 49.5% |
| **`aas`** | **+0.0011** | [−0.0166, +0.0244] | **0.46** | **50.0%** |
| `aas_self` | +0.0107 | [−0.0034, +0.0269] | 0.044 | 54.7% |
| `post_canon_norm` | +0.0154 | [+0.0067, +0.0263] | 5.8e-6 | 67.3% |
| `pre_canon_norm` | −0.0075 | [−0.0348, +0.0282] | 0.97 | 47.2% |

**⇒ 换掉区域 与 只换句框，对 s 场的破坏程度在统计上完全相同（逐源胜出率 50.0%）。**
A&S 的 ρ 下降**与「指令指向哪块区域」无关**，是把噪声放大了。
（唯一显著的 `post_canon_norm` sep = +0.015，相对判据 0.3 小两个量级。）

**这条同时回答了主 agent §四「判据 0.3 是否从设计上不可达」**：在这条通道上，
「同一区域、换个说法」的可达上界就是 0.878（canon）/ 0.445（aas）——
**地板与主判据重合**，所以 0.3 这个门在此通道上**不是难，是无意义**：
它测的东西（区域敏感性）在该通道上不存在。

**(2) AUC_target 全线塌在 0.5 —— 唯一单解度量给出否定答案**

`AUC_target` 的解析性质：s 与指令无关 ⟺ 中位 = 0.5。两个 GT、四个格、三个 token 全部如此：

| 变体（image_first） | AUC_target[cgt] | 95% CI | p vs 0.5 | 背景子集 | normal-only |
|---|---|---|---|---|---|
| `canon` | 0.4989 | [0.474, 0.527] | **0.80** | 0.4996 | 0.4953 |
| **`aas`** | **0.4957** | [0.457, 0.541] | **0.77** | 0.4957 | 0.5000 |
| `aas_self` | 0.5015 | [0.492, 0.518] | 0.47 | 0.5092 | 0.5002 |
| `post_canon_norm` | 0.5159 | [0.493, 0.536] | 0.037 | 0.5159 | 0.5166 |

**没有任何变体在任何 GT / 任何子集上给出有意义的 > 0.5。** 唯一 p<0.05 的
`post_canon_norm`（0.516）在背景子集上就掉到 p=0.054，且效应量 0.016——
按 **D-31**：这**不是**正面证据。

**(3) ⚑ 去共模差分场：负责人在等的那个数字 —— 否定**

主 agent §三 要求的检验（`AUC(s(reg_a) − s(reg_b), GT)`，配**同区域对照**
`AUC(s(syn_a) − s(syn_b), GT)`），`cgt` 列、image_first：

| 变体 | 区域对立差分场 | 95% CI | **同区域对照** | 结论 |
|---|---|---|---|---|
| `canon` | 0.507 | [0.495, 0.517] | **0.525** | 对照**更高** |
| **`aas`** | **0.517** | [0.504, 0.533] | **0.524** | 对照**更高** |
| `aas_self` | 0.515 | [0.498, 0.529] | 0.504 | 打平 |
| `pre_canon_norm` | 0.504 | [0.495, 0.516] | **0.527** | 对照**更高** |

**去掉共模之后，「点名对立区域」的残差携带的区域信息，不比「点名同一区域」更多。**
canonical 通道上的这个否定结论（主 agent §三 报的 0.5129 vs 0.5293）
**在 A&S 通道上完整复现**（0.517 vs 0.524）。

⇒ **「信息在里面、只是被 sink 淹了」这个解释被证否**——因为 A&S 的 Eq.3
**就是**去 sink 那一步，做了之后差分场依旧读不出区域。

### 3.3 A&S 的场是什么形态：尖刺，不是掩膜

| 变体 | top-1 格占场方差比例（中位） |
|---|---|
| 均匀场参照 | 0.006 |
| `canon` | 0.083 |
| `post_all`（A&S 未归一化） | 0.202 |
| **`aas`** | **0.417** |

**A&S 减完共模后，单个格子吃掉 42% 的空间方差。** 低 ρ 的来源是这些孤立尖刺，
不是区域交换。`viz/aasbest_*` 三张图上肉眼可见（同一图两条对立指令，
A&S 场都是 1–2 个亮格，位置还不一样）。

配套读数：`aas_residual_std / raw_std = 0.678`（残差非零，A&S 没有把场减没），
但残差的形态是噪声。

### 3.4 三个 special token 分别报

| token | canon ρ_ro | canon AUC_t[cgt] | **aas ρ_ro** | **aas AUC_t[cgt]** | aas AUC(reg_a)[cgt] |
|---|---|---|---|---|---|
| `<retouch_light>` | 0.884 | 0.499 | **0.439** | **0.496** | 0.346 |
| `<retouch_color&temp>` | 0.841 | 0.496 | 0.734 | 0.502 | 0.497 |
| `<retouch_colormixer>` | 0.804 | 0.496 | 0.776 | 0.507 | 0.452 |

- **三者的 AUC_target 全部 ≈ 0.5**：不是「选错 token」。
- `<retouch_light>` 在 A&S 下 ρ 最低（0.439），但那是**噪声最强**而非信息最多
  （它的尖刺度也最高、AUC(reg_a) 0.346 偏离 0.5 最远但**方向是反的**）。
- **跨 token ρ**（同图同指令、只换 query token）：canon **0.81–0.94**（复现 G1 的 0.809–0.928）；
  **aas 只有 0.19–0.57**。底层注意力高度同源却在 A&S 下彼此不相关 ⇒ 又一条噪声证据。

### 3.5 shuffle 负控制（任务卡必跑）与「反超」现象

image_first、n=60 配对：

| 变体 | ρ_syn（同义） | ρ_shuf（跨图错位） | 反超？ |
|---|---|---|---|
| `canon` | **0.842** | **0.895** | **是**（复现 G1 逐位数字） |
| `post_all` | 0.880 | 0.874 | 否 |
| **`aas`** | **0.360** | **0.321** | **否** |
| `aas_self` | 0.703 | 0.704 | 打平 |

**结论**：**反超是 canonical 读出特有的伪影，不是协议层共性问题**——A&S 下顺序恢复正常
（0.360 > 0.321）。但**两者都掉到噪声位**，所以「顺序正常」在这里不构成正面证据。
最可能的机制：canonical 场几乎全由图像驱动，ρ 主要由**句长/表面形式**决定，
而 shuf 与 syn_a 都是完整自然指令、`syn_b` 是 `instruction_short`（更短），
故 syn 对的表面形式距离**更大** ⇒ ρ 更低。这正是本实验引入**匹配地板**的原因。

### 3.6 层段扫描（A&S）

| 层段 | L0–23（官方口径） | L8–15 | L0–7 | L16–23 | L20–22 |
|---|---|---|---|---|---|
| sep（ρ_floor − ρ_region_opp） | 0.006 | 0.019 | 0.005 | 0.003 | 0.017 |

**全部 ≤ 0.019**，换层段救不回来（与 G1 的逐层结论一致）。

### 3.7 ⚑ 与 RO-3 已找到的读出配对比较（主 agent 追加的新判据）

RO-3 的 scache 臂覆盖本区域批 **214/214 源 × {reg_a, reg_b}**、instr_hash 同源，
故用**我方完全相同的 valid / 掩膜 / AUC 代码重算**（不是引用其 REPORT 数字）：

| 读出 | query 端 | ρ_region_opp | n<0.3 | **AUC_target[cgt]** | AUC_target[sam3] | 差分场 AUC[cgt] |
|---|---|---|---|---|---|---|
| `canon`（G1/RO-9） | **special token** | 0.884 | 0/214 | 0.499 | 0.507 | 0.507 |
| **`aas`（本实验）** | **special token** | 0.439 | 35/214 | **0.496** | 0.496 | 0.517 |
| **`ro3-l11-h5`** | **instruction 文本 token** | **0.172** | 144/214 | **0.667** | 0.760 | **0.809** |
| **`ro3-fused`（32 头）** | **instruction 文本 token** | **0.044** | 172/214 | **0.739** | 0.833 | **0.844** |

配对差（同源、同口径）：

| 对比 | Δρ_region_opp 中位 | 95% CI | p |
|---|---|---|---|
| `canon` − `ro3-fused` | **+0.794** | [0.736, 0.861] | 2.0e-21 |
| `aas` − `ro3-fused` | **+0.371** | [0.316, 0.443] | 9.0e-19 |
| `aas` − `ro3-l11-h5` | +0.269 | [0.215, 0.315] | 1.9e-13 |

**A&S 显著劣于 RO-3 的读出，差距是数量级的（AUC_target 0.496 vs 0.739）。**
这同时**独立复现了 RO-3 的结论**（我用自己的掩膜与 AUC 代码重算，
`ro3-l11-h5` 在 sam3 列得 0.760，与其 REPORT 的 0.766 一致）。

### 3.8 prompt 顺序维度（与 RO-2 的 C7 互为独立验证）

主 agent 后续通知 RO-2 已单独判定「prompt 顺序」为 false。**本实验在收到该通知前已把这一维跑完**，
结果**独立复现了同一结论**：把 `Instruction:` 移到 `<image>` 之前（实测
`img_span` 从恒定的 (14,270) 变成随指令长度浮动的 (63,319)/(72,328)，
指令确实进了 image token 的可见范围）后——

- `canon`：ρ_region_opp 0.884 → 0.883，AUC_target[cgt] 0.499 → 0.505（**纹丝不动**）
- `aas`：ρ 0.439 → 0.592，AUC_target[cgt] 0.496 → 0.503（**仍是 0.5**）
- 生成未崩：fallback 0.028 → 0.017，gen_len 同量级

⇒ **把指令喂进 image token，读出没有变好。** 与 RO-2 的 C7 一致，两条独立路径同结论。

## 四、可视化（`viz/`，6 张，成功与失败两端各 3）

面板 = **2 行（canonical / attend-and-segment）× 5 列**：
源图·GT(C_GT) ｜ s(reg_a 点名主体) ｜ s(reg_b 点名补集) ｜ **s(reg_a_para 同区域换句框＝匹配地板)** ｜ s(a)−s(b) 差分场。
两张 s 图叠 GT 红色轮廓，各带 AUC；差分场带 ρ 与 diffAUC。**这是本实验最有说服力的一张图。**

| 前缀 | 张数 | 含义 |
|---|---|---|
| `aasbest_paired_*` | 3 | A&S 相对 canonical **分离最多**的 3 例（ρ 0.076 / 0.092 / 0.110 vs canonical 0.83–0.91）。**即使在这里，A&S 的场也只是 1–2 个孤立亮格，位置与 GT 无关**——最有利的样本恰恰最能看出低 ρ 来自尖刺 |
| `aasworst_paired_*` | 3 | A&S 相对 canonical **分离最少**的 3 例（ρ 0.56–0.74）——**典型失败案例**：A&S 与 canonical 一样，两条对立指令下的场几乎重合 |

选样规则**单条、写死为代码常量**（按 `ρ_aas − ρ_canon` 排序取两端），非事后挑图。
**文件名只陈述实测 ρ 值**，不含 success/好坏这类价值判断词（D-30）。

## 五、结论

### 5.1 判决：(b) 读法错 —— 但「读法」指的是 **query 端读了哪个 token**，不是 sink

任务卡的三选一，逐条给证据强度：

| 解释 | 判定 | 证据 |
|---|---|---|
| **(c) 代码错** | **排除** | 主 agent 的四视角审查零 blocker；**本实验独立复算**：`canon` 与 G1 落盘 npz 逐格 Pearson **0.99999992**、ρ_region_opp **0.884 逐位一致** |
| **(a) 假设错**（VLM 里没有随指令变的 where 信息） | **排除**（但只在整机层面） | RO-3 的 `L11H5 + instruction token 当 query` 在**本实验同一批 214 源、同一套掩膜与 AUC 代码**下重算得 **AUC_target[cgt] 0.667 / 差分场 0.809**；`ro3-fused` 得 **0.739 / 0.844**。信息**在** |
| **(b) 读法错** | **成立** | 同一模型、同一批样本，**换 query token** 就从 0.496 跳到 0.739；**而换 sink 处理（A&S Eq.3）纹丝不动** |

### 5.2 ⚑ 「被 sink 淹了」这个解释：**证否**

负责人的原判断是「有空间感知信息，但是被 sink 了，解决了这个 sink 就是核心问题」。
**attend-and-segment 的 Eq.3 就是解 sink 那一步**（论文附录明写动机是背景 token
`"attract more attention"`、这些伪影在输出序列上 `"relatively stable"`，减掉输出序列均值即可）。
本实验把它原样实施在 VeraRetouch 上：

- 它**确实起作用了**（残差 std/raw std = 0.678，不是把场减没）；
- 它**确实把 ρ_region_opp 打下来 0.42**（p=9e-37）；
- **但匹配地板同步塌陷（sep = +0.001，p=0.46，逐源胜出率 50.0%）**，
  **AUC_target 仍是 0.496（p=0.77）**，**差分场 0.517 ≤ 同区域对照 0.524**。

⇒ **sink 不是 G1 失败的原因。** 把 sink 去掉，special token 的注意力里**依然没有**
指令条件的 where 信息。真正的原因是 **query 端用了 special token**——
RO-3 已实测 `gl` 池在全部 336 个 (层,头) 上 AUC_target 上限只有 0.554。

### 5.3 措辞纪律（D-31）—— 本报告里哪些是正面证据，哪些只是「未触发死刑」

| 读数 | 性质 |
|---|---|
| `aas` 的 ρ_region_opp 从 0.884 → 0.439，35/214 跌破 0.3 | **不是正面证据**。匹配地板同步塌陷（sep=+0.001, p=0.46）证明这是噪声放大 |
| `aas` 的 AUC_target = 0.496（p=0.77） | **负面证据**（唯一单解度量，明确否定） |
| shuffle 反超在 A&S 下不复现 | **不是正面证据**。ρ_syn 与 ρ_shuf 都掉到 0.32–0.36 的噪声位，顺序正确不代表有信号 |
| A&S 的 AUC(reg_a) = 0.346（翻号 0.654） | 只说明 A&S 的场**反向**弱相关于编辑区域，**与指令无关**（reg_a 0.346 vs reg_b 相近） |
| `post_canon_norm` 的 sep=+0.015（p=5.8e-6）与 AUC_target=0.516（p=0.037） | **统计显著但效应量比判据小两个量级**，且背景子集 p=0.054。按 D-31 **不追认为正面证据** |

### 5.4 结论的边界（被证伪与未被证伪）

- **被证伪**：①「DiffLMM 的读出方式能从 VeraRetouch 的 special token 注意力里读出指令条件的 where」；
  ②「G1 的失败是因为信息被 sink 淹了」；③「换 prompt 顺序能救」。
- **未被证伪**：「VLM 里有随指令变的 where 信息」——**恰恰相反，RO-3 已证其存在**，
  本实验用自己的口径独立复算确认。
- **本实验只否定了 query = special token 的那一族注意力读出**（含 canonical、A&S、
  8 个变体 × 3 token × 5 层段 × 2 prompt 顺序 × 2 GT）。

## 六、建议下一步

| # | 建议 | 指向 |
|---|---|---|
| B1 | **RO-9 / A&S 读出一并按失败判据处置**：不是「换个更好的注意力读法」的问题，**query = special token 这一族整体不成立**。DiffLMM/A&S 是本项目对该族的最后一次、也是最强的一次尝试（它专门为解 sink 而设计），已排除 | EXPERIMENTS_v3 RO-9 行 + 新增 RO-D 行 |
| B2 | **where 通路改挂 RO-3 的 instruction-token query 读出**（`ro3-fused` AUC_target[cgt] 0.739）。本实验用独立口径复算确认了它的优势是数量级的 | RO-W / RO-5 上游 |
| B3 | **论文里「sink」这条叙事必须改写**：本实验是「解了 sink 也没用」的直接反例，且用的是提出该解法的原论文方法。**这条负面结果本身有发表价值**——它说明 attend-and-segment 的有效性依赖于 query 是**语义文本 token**，把它套到 SFT 出来的控制类 special token 上会失效 | 论文 analysis 章 |
| B4 | **匹配噪声地板应升格为所有 ρ 类判据的标配**。本实验证明：不带地板时，A&S 的 ρ 从 0.884 掉到 0.439 会被读成巨大进步，实际 sep=0.001 | EXPERIMENTS_v3 全部 RO 行 |
| B5 | **AUC 的 GT 一律用 `.cgt.png`**：实测 C_GT 与 SAM3 主体掩膜 **IoU 中位仅 0.374、61.5% 的源 <0.5、面积差 3.2×**，两者不可互换 | DATA_ASSIGNMENT 统一评分集 |

## 七、给 RO-W 的交付（蒸馏目标 scache arm）

> **⚑ 先读这一节再消费**：本臂的 s 场**不在 [0,1]**，`tools/scache/upsample.py` 的
> `upsample_s` 默认 `clamp=(0.0, 1.0)` 会把负值**静默吃成 0 且不报错**（`ro9` 臂已踩中）。

| 项 | 值 |
|---|---|
| 路径 | **`/var/cache/veradata/scache/difflmm-light-L0-23/`** |
| 条目数 | **822**（image_first 的 region+floor+syn+shuf 全部条件；16×16 float16 + 逐条 `.meta.json`） |
| 内容 | `aas` 变体（忠实 attend-and-segment，Eq.3，全 24 层，`<retouch_light>`，prev 行约定） |
| **取值域（存储）** | `[-12.53, +9.34]`，**含负数** |
| **全局缩放** | `meta.norm.scale = 1000.0`；**`s_raw = s_stored / 1000`** |
| 取值域（原始） | `[-0.01253, +0.00934]`，\|s\| 的 p99.9 = 2.48e-3 |
| 逐图归一化 | **无**（`meta.norm.per_image_normalization = false`；scale 是全批**单一常数**） |
| 消费方式 | `upsample_s(..., clamp=None)`，或先用全局仿射映到 [0,1] 再上采样 |
| 往返精度 | 写→读→除以 scale 复原，最大相对误差中位 **2.1e-4** |

**但请注意本实验的结论**：这个臂的场 **AUC_target = 0.496**，即**不含指令条件信息**。
**不建议**把它当 RO-W 的蒸馏目标——**应改用 `/var/cache/veradata/scache/ro3-fused/`**
（本实验用同一套口径复算得 AUC_target[cgt] **0.739**、差分场 0.844）。
本臂交付的价值是**可复现的负结果基线**与消融对照。
