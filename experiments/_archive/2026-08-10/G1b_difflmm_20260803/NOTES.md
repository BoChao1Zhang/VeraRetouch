# NOTES — G1b / RO-D：DiffLMM「attend-and-segment」读出重跑 G1

日期：2026-08-03 ｜ 编码 subagent ｜ 分支 `lens-exp` ｜ 卡 1（`CUDA_VISIBLE_DEVICES=1`）

---

## 一、在线核实记录（任务卡第 1 优先事项）

> 纪律：`IMPL_DOSSIER 附录 B` 之外的每一条 URL / 数字 / 结论都必须打开原始来源。
> 下表每一行的「核实方式」列给出**我实际打开过的 URL**。

### 1.1 论文身份 —— ⚑ 任务卡的口头描述有一处需要更正

| 项 | 核实结果 | 核实方式（实际打开的 URL） |
|---|---|---|
| 标题 | **Emergent Visual Grounding in Large Multimodal Models Without Grounding Supervision** | https://arxiv.org/abs/2410.08209 |
| 作者 | **Shengcao Cao, Liang-Yan Gui, Yu-Xiong Wang**（UIUC） | 同上 |
| arXiv | **2410.08209**（v1 2024-10-10；v2 2025-10-16） | 同上 |
| venue | **ICCV 2025 Findings**（abs 页 Comments 字段） | 同上 + https://yxw.cs.illinois.edu/files/GroundLMM_ICCV2025.pdf（文件名即 ICCV2025） |
| 先前投稿被拒 | **ICLR 2025 投稿**，旧题 *Emerging Pixel Grounding in Large Multimodal Models Without Grounding Supervision* | https://openreview.net/forum?id=UFKC0lMTdK |
| 官方仓库 | **https://github.com/Shengcao-Cao/groundLMM**（Apache 2.0；`aas/` 目录即 attend-and-segment） | https://github.com/Shengcao-Cao/groundLMM |
| 项目页 | https://groundlmm-iccv.github.io/ | 同上 |

**⚑ 更正 1**：任务卡说「先投 ICLR 被拒、后转 **ECCV**」。前半段属实（OpenReview UFKC0lMTdK
是 ICLR 2025 投稿），**后半段不对：落地的是 ICCV 2025 Findings，不是 ECCV**。
（我没有把 ECCV 当既定事实去检索，而是从 arXiv abs 页的 Comments 字段读到的 venue。）

**⚑ 更正 2（更重要）**：**「DiffLMM」是论文里的两个贡献之一，而且不是那个读出方法。**
论文有两个可分离的贡献：

| 贡献 | 是什么 | 改前向吗 | 对本实验的可用性 |
|---|---|---|---|
| **attend-and-segment（A&S）** | **读出方法**：从 LMM 的 attention 里读空间图 | **不改** | ✅ 这才是「DiffLMM 的读出方式」，可直接套到 VeraRetouch |
| **DiffLMM** | **模型**：把视觉编码器换成 diffusion 特征 + CLIP 拼接，重训 projector | **改**（换编码器 + 重训） | ❌ 是另一个模型，套不到 VeraRetouch（要重训） |

论文自述 A&S：**"without changing their architecture or requiring additional training"**
（arXiv HTML 全文，https://arxiv.org/html/2410.08209v2）。A&S 在**原版 LLaVA** 上就能用，
DiffLMM 只是让它效果更好的一个**更换编码器的模型**。

⇒ 本实验做的是 **attend-and-segment 读出**，不是 DiffLMM 模型。REPORT 全篇按此措辞。

### 1.2 ⚑ 一票否决级核实项：A&S 是纯读出，还是改了前向？

**答案：A&S 是纯读出（任务卡分类 (i)），前向逐比特不变。**三重证据：

1. **论文原文**：`"we propose attend-and-segment, a simple yet effective method for grounding
   LMMs without changing their architecture or requiring additional training"`
   （https://arxiv.org/html/2410.08209v2）。
2. **官方代码**（https://raw.githubusercontent.com/Shengcao-Cao/groundLMM/main/aas/infer_attn.py）：
   唯一的模型调用是一次普通 `model.generate(..., output_attentions=True,
   return_dict_in_generate=True)`。没有 hook 改激活、没有额外模块、没有多步去噪。
   后处理是纯 numpy/torch：
   ```python
   save_attn_i = torch.cat([x[:, :, -1, image_token_start_index:image_token_end_index]
                            for x in save_attn_i])
   save_attn_i = save_attn_i.mean(dim=(0, 1)).reshape(feature_height, feature_width)
   ```
3. **归一化也在读出侧**（https://raw.githubusercontent.com/Shengcao-Cao/groundLMM/main/aas/gcg.py）：
   ```python
   attn_mean = attentions.mean(dim=0)
   attentions = attentions - attn_mean
   ```

**因此任务卡里「若确认是 (ii) 必须补基座模型对照」这一分支不触发。**
但负责人「非常影响推理」的直觉不是空穴来风——它指向的是 (i) 里的**开销**：
A&S 要 `output_attentions=True` 拿全部层×头的 attention（本实验实测前向从 0.24 s → 见 §五），
而且 SAM 分割那一步很贵。**开销大 ≠ 改前向**，科学含义上是安全的。

我另外补了两条**比基座对照更贴题**的对照（见 §三 判据表），因为真正的风险不是
「DiffLMM 造信息」，而是「减均值这个算子把噪声放大成了看起来像指令依赖的东西」：
- **shuffle 负控制**（任务卡必跑）；
- **frame paraphrase 匹配噪声地板**（新增，见 §二.3）。

### 1.3 A&S 的确切数学定义（已按官方代码逐行核对）

设生成的输出序列为 `o_1..o_r`（r = 输出序列长度），image token 有 `h×w` 个。

| 步 | 论文定义 | 代码对应 |
|---|---|---|
| 1 | `A_i^raw ∈ [0,1]^(n_layer × n_head × (p+hw+q+i−1))`：生成 `o_i` 时的注意力 | `output_ids['attentions'][i]` |
| 2 | 只取 `h×w` 个 image token 列，**对层与头求平均** → `A_i^reduced ∈ [0,1]^(h×w)` | `cat(...)` 后 `.mean(dim=(0,1))` |
| 3 | **Eq.3**：`A_i^norm = A_i^reduced − (1/r) Σ_{j=1..r} A_j^reduced` | `attentions - attentions.mean(dim=0)` |
| 4 | 取 `A_i^norm` 最大值点坐标当 point prompt 喂 SAM → 二值掩膜 | `aas/gcg.py` |

**与 G1 canonical 读出的三处差异**（本实验要隔离的自变量）：

| 维度 | G1 canonical（RO-9） | A&S |
|---|---|---|
| softmax | **pre-softmax logit** | **post-softmax 概率** ∈[0,1] |
| 层 | L8–15 | **全部 24 层** |
| **归一化** | **无** | **Eq.3：减去沿输出 token 轴的平均图** |
| query 行 | special token **自身位置** | 官方取 `[:, :, -1, :]` = **产生该 token 的前一位置** |

**⚑ Eq.3 就是「解 sink」的那一步**：论文附录说明动机是
`"uninformative visual tokens (usually in background) attract more attention than other
visual tokens"`、这些是 `"repurposed for internal computations"` 的伪影、且在输出序列上
`"relatively stable"`——**沿输出 token 轴求平均正好把这个稳定的共模分量估计出来并减掉**。
这与负责人「有空间感知信息，但是被 sink 了」的判断在机制上完全对得上。
论文 Table 4 消融：去掉这一步 GCG mask recall 从 **46.4 → 43.9**。

本实验因此不只测「A&S 好不好」，而是把三个差异做成**消融梯子**（8 个变体，同一次前向，
零额外成本），直接回答**哪一个成分才是关键**。

### 1.4 ⚑ 主 agent §二（image token 排在指令之前）对 A&S 是否致命

主 agent 反馈指出：`<image>` 在 `Instruction:` **之前**，因果掩码下 image token 的
key/value 与指令逐比特无关，`s = Kᵀq` 里只有 64 维 query 承载指令。

**核实结论：A&S 读的是同一族交互（输出 token 的 query × image token 的 key），
它并不绕开这堵墙，而是——原论文的 LLaVA 也在同一堵墙后面。**

- LLaVA 的 prompt 同样是 `<image>\n{question}`，image token 也排在问题文本之前；
  A&S 在这个前提下依然做到 GCG mask recall 44.2、超过全监督的 GLaMM。
- 所以这堵墙的正确读法是：它使指令条件性**低秩**（只能靠 query 侧从冻结的 `K` 里挑方向），
  **不是**使其为零。它**不能**单独解释 G1 的失败。
- **反过来说，这也正是本实验能干净归因的原因**：A&S 与 canonical 读的是同一个 `Kᵀq`，
  唯一变的是怎么把它读出来。若 A&S 能分离而 canonical 不能，(b)「读法错」成立且机制清楚。

⇒ 同时这也再次确认 **A&S 不属于「靠改前向让 image token 看见指令」的那类方法**，
「读出 vs 创造」的疑虑在机制层面被排除，不只是靠论文的一句自述。

### 1.5 复现成本

- **不需要官方仓库的任何代码**：A&S 是 4 行算术，本项目 `tools/readout/ro9_gl_attention.py`
  已有 eager attention 导出、token 定位、D-0、网格对齐的全套设施，直接扩写即可。
  官方仓库是 LLaVA fork（`llava/` 全量），装它只会引入版本冲突，**不装**。
  偏离说明：**未使用 SAM 分割那一步**——本实验要的是连续 s 场（16×16），不是二值掩膜；
  Eq.1–3 全部照搬，只省略 Eq.4 之后的 SAM point-prompt 步骤（那一步不影响 ρ / AUC 判据）。
- 显存/耗时：见 §五实测。

---

## 二、VeraRetouch 三个 special token（任务卡第 2 项）

### 2.1 字面与 id（`added_tokens.json` + tokenizer 运行时实测，两处一致）

| # | 字面 | id | tokenizer 实测 |
|---|---|---|---|
| 1 | `<retouch_light>` | **151646** | `ids=[151646]`（单 token）✓ |
| 2 | `<retouch_color&temp>` | **151647** | `ids=[151647]`（单 token）✓ |
| 3 | `<retouch_colormixer>` | **151648** | `ids=[151648]`（单 token）✓ |

常量定义在 `llava/constants.py:17-19`；模型侧 `register_special_token_idx`
（`llava/model/VeraRetouch.py:39`）把三者注册给 `retouch_head`。

### 2.2 ⚑ 它们在模板里排在指令之**后** —— causal 可见性问题的明确答案

实测 prompt 原文（style 模式模板，`data/infer_dataset.py:141` 同款；74 个 token）：

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

实测结论：

- **三个 retouch token 在 prompt 里一个都没有**（`151646/151647/151648 in prompt_ids → False, False, False`）。
  它们全部出现在 **assistant 生成段**，即指令文本（prompt 第 40–70 号 token 附近）**之后**约 350 个 token 处。
- ⇒ **因果掩码下它们完全看得见指令。**「special token 排在指令之前所以看不到指令」这个
  架构性解释 **被证否**，不能拿它解释 G1 的失败。
- 与主 agent §二 的发现**不矛盾、但对象不同**：主 agent 说的是 **image token** 排在指令之前
  （属实，`<image>` 是 prompt 第 14 号 token，指令在其后）。所以受限的是 **key 侧（`K`）**，
  不是 **query 侧（三个 special token）**。两条结论合起来才是完整图景：
  **query 看得见指令，key 看不见** ⇒ 指令条件性必须、且只能经由 query 侧进入。

### 2.3 三 token 全部单独报

三个 token 在同一次前向里同时导出（零额外成本），metrics 与 REPORT 逐 token 分列。

---

## 三、判据（预注册，分析前写定）

沿用 G1 口径，**判据函数直接 `import` G1 的 `analyze_g1.py`**（`pearson` / `spearman` /
`roc_auc` / `SubjectMaskBank` / `luma_valid_from_image`），保证口径逐行一致、可配对比较。

| 判据 | 阈值 | 角色 |
|---|---|---|
| **ρ_region_opp 中位** | **< 0.3** | **主判据**（G1 原判据，口径不变） |
| Δρ_region_opp（A&S − canonical）配对 | 中位 + bootstrap 95% CI + Wilcoxon | **本实验核心产出** |
| **AUC_target**（pooled 中位） | > 0.5 显著 | 唯一单解度量（见下） |
| ρ_syn 中位 | > 0.7 过 / < 0.5 死 | 稳定性 |
| \|ρ_Y\| 中位 | < 0.5 过 / > 0.8 死 | 非亮度马甲 |
| ρ_shuf | 应显著 < ρ_syn | **负控制，必跑** |

**AUC_target 定义**（任务卡指定，G1 口径）：`reg_a` 点名主体 ⇒ 目标 = M，`AUC_target = AUC(s_a, M)`；
`reg_b` 点名补集 ⇒ 目标 = ¬M，`AUC_target = AUC(s_b, ¬M) = 1 − AUC(s_b, M)`。把两条件 pool 后取中位。
**s 与指令无关 ⟺ 中位塌到 0.5**（因为此时 `AUC(s,M)` 与 `1−AUC(s,M)` 关于 0.5 对称）。
`region_b_kind == "background"` 的 165 源上 `reg_b` 的目标**字面就是 ¬M**，故该子集是最干净的读数，正表两者都报。

### 新增判据（应主 agent 反馈，均写在分析前）

1. **⚑ 匹配噪声地板 ρ_floor（§四 反馈：0.3 可能不可达）**——新增 `reg_a_para` 臂：
   保持**区域串与方向词逐字不变**，只重写句框：
   - `reg_a`      = `Please {d} {region}, keeping the rest of the image unchanged.`
   - `reg_a_para` = `Without altering any other part of the photo, please {d} {region}.`

   于是得到一对变量隔离干净的量：
   - `ρ_floor`      = corr(s(reg_a), s(reg_a_para))：**换句框、同区域** → 若 s 追区域则应**高**
   - `ρ_region_opp` = corr(s(reg_a), s(reg_b))：**同句框、换区域** → 若 s 追区域则应**低**
   - **`sep_matched = ρ_floor − ρ_region_opp`（配对）= 本实验最干净的判别量**

   表面形式扰动量级实测（词级 Levenshtein，`config/g1b_config_report.json`）：
   `reg_a↔reg_a_para` = **15 词（恒定）**；`reg_a↔reg_b` = 中位 **12**、均值 13.6。
   **地板侧的扰动比区域侧还大** ⇒ 这是一个**保守**（偏难）的地板，
   若仍有 `ρ_floor > ρ_region_opp`，差异只能归给「区域变了」。
   这条直接回答「ρ_region_opp > ρ_syn 是不是表面形式距离伪影」——G1 的 syn 对是两句
   **完全不同的自然指令**，与只差一个名词的 reg 对**根本不可比**。

2. **⚑ 去共模差分场 AUC（§三 反馈）**：报 `AUC(s(reg_a) − s(reg_b), M)`，
   并配**同区域对照** `AUC(s(syn_a) − s(syn_b), M)`。只报原始场 ρ 会被主体显著性主导。
3. **跨 token ρ 当动态范围参照（§四 反馈）**：同图同指令、只换 query token 的 ρ，
   canonical 与 A&S 各报一套。
4. **生成文本落盘（§五 反馈）**：`meta.gen_text`（3000 字符）逐样本入 npz，并报
   区域对立两条指令的生成长度差分布（「指令确实被消费了」的可观测证据）。

**表述纪律（D-31）**：任何「PASS」逐条标注属于**未触发死刑**还是**获得正面证据**。

---

## 四、假设与待确认清单

### 已自行核实

- [核实] 三 token 字面 / id / 单 token 性（tokenizer 运行时）；prompt 内不含三者。
- [核实] A&S 为纯读出（论文原文 + 官方 `aas/infer_attn.py` + `aas/gcg.py` 三重）。
- [核实] Eq.3 减均值的轴 = 输出 token 轴（官方 `attentions.mean(dim=0)`）。
- [核实] 官方对层/头是**全部平均**（`mean(dim=(0,1))`，无层选择）。
- [核实] 官方 query 行取 `[:, :, -1, :]`（HF generate 每步最后一行）= **产生** `o_i` 的行。
- [核实] G1 全部配置文件可直接复用；源图 300/300 在 `/var/cache/veradata/g1_srcimg_20260803`。
- [核实] 卡 1 当前占用（见 STATUS.md 实测），`set_per_process_memory_fraction` 已设。
- [假设→已加断言] teacher-forced 复算与生成时注意力逐位一致（greedy、同一 KV 数学）。
  **额外核验**：canonical 变体与 G1 原 npz 逐格对拍（`metrics.crosscheck_vs_G1_canonical`），
  这同时验证「我的读出链路 = G1 的读出链路」。

### 「待主 agent 决策」（**采用保守默认继续，未拍板**）

| # | 决策点 | 两种做法都合理在哪 | **本实验采用的保守默认** |
|---|---|---|---|
| **Da** | **query 行约定**：special token **自身位置**（G1 口径，也是 `retouch_head` 实际取 hidden state 的位置）vs **前一位置**（A&S 官方口径，「产生该 token 的注意力」） | 前者与 G1 配对最干净、且与模型实际用法一致；后者忠实于被复现的方法 | **两个都跑、都报**（同一次前向，零成本）。**忠实 A&S（`prev`）作为 headline**，`self` 作为与 G1 对齐的配对基线。不擅自只报好看的那个 |
| **Db** | **层聚合**：A&S 官方全部 24 层 vs G1 canonical L8–15 | 忠实复现 vs 与 G1 配对 | **headline 用官方全层**；`band_scan` 另报 5 个层段，且 `post_canon_norm` 变体专门隔离「层」这一个变量 |
| **Dc** | **归一化前是否对 image token 列重新归一** 使其和为 1 | 官方不做（直接用原概率）；重归一会让不同样本可比 | **不做**（忠实官方；且逐图重归一属红线「逐图归一化」的灰区） |
| **Dd** | **A&S 减均值是否算「逐图归一化」红线** | 它是**被检验方法自身的定义**，不是我们加的预处理；但它确实是逐 (图,指令) 的空间相关操作 | **原始分量（未减均值的 `post_self`/`post_prev` 与均值图）全部原样落盘**，减法只在分析侧做，任何人可以复算未归一化版本。**不做任何 min-max / softmax 逐图归一化** |
| **De** | **样本量**：是否把 shuffle 对照从 G1 的 n=60 扩到 ≥200（G1 自评局限点） | 扩大能把 binomial p=0.155 变成强断言；但 +280 次读出、~1 h | **沿用 n=60**（口径与 G1 一致才能配对比较）。若主 agent 要强断言，配置现成，加跑即可 |
| **Df** | **基座（pre-SFT）对照** | 任务卡要求「若 A&S 改前向则必须补」；实测 **A&S 不改前向 ⇒ 该分支不触发**。但 G1 建议 B4 独立地想要这个对照 | **本实验不做**（触发条件不成立，且基座词表里没有三个 special token，需要另设 query 定义 = 另一个实验）。改由 §三 的 shuffle + frame 地板两条对照承担「读出 vs 创造」的判别 |

---

## 五、运行记录

（排卡与吞吐见 STATUS.md；本节记**影响结果解读**的实测事实。）

### 5.1 ⚑ 三个 special token 在生成序列里是**相邻**的

实测（`meta.gen_idx`，典型样本 `gen_len=311`）：
`light=307, colortemp=308, colormixer=309` —— 生成文本末尾固定是
`Retouch Tokens:\n<retouch_light><retouch_color&temp><retouch_colormixer>`。

后果（影响决策 Da 的解读）：**`prev` 与 `self` 两种行约定只差一个位置**，且
`prev(colortemp) == self(light)`、`prev(colormixer) == self(colortemp)`。
所以「忠实 A&S（prev）」与「G1 口径（self）」在这三个 token 上是同一组行的一位平移，
两者都报时读者要知道它们**高度重叠、不是独立证据**。

另一后果：三个 token 位于生成序列的**最末尾**（307/311），指令在 prompt 里、
距离约 350 个 token。主 agent §五 指出的「指令可能已在 plan 文本里被消费掉」
在位置上完全成立——`meta.gen_text` 已逐样本落盘，可离线核实。

### 5.2 ⚑ 一条被实测**推翻的**先验推理（诚实记录，含更正）

**我原来的推理（错）**：同一 query 行内 `post = exp(pre)/Z`（Z 是该行常数），softmax 行内单调
⇒ 单层上 pre 与 post 的空间排序应逐格一致，Spearman 应 = 1.0，因此「换 softmax」
不可能改变任何基于秩的度量，Eq.3 才是唯一真自变量。

**实测（n=63 层×源）：Spearman 中位 = 0.873，不是 1.0。**

**原因（已定位，不是 bug）**：单调性只在**单个 head 内**成立，而两边都做了 **head-mean**——
`mean_h softmax_h(·)` ≠ `softmax(mean_h ·)`。pre 侧的 head-mean 被 **logit 尺度大的 head 主导**
（实测 pre 值域跨到 [−47, +192]），post 侧则给每个 head **等权**（每头的概率都在 [0,1]）。

**结论修正**：**「post-softmax」不是可以先验降权的装饰项，它是一次真正的重加权**——
把「按 logit 尺度加权」换成「按概率等权」。A&S 的三个成分（softmax / 全层 / Eq.3）
**都必须实测**，消融梯子的 8 个变体缺一不可。

**读出链路本身的正确性另有硬背书**（不依赖上面这条推理）：
`metrics.crosscheck_vs_G1_canonical` —— 我的 `canon` 变体与 G1 落盘 npz 逐格对拍，
n=40，**Pearson 中位 0.99999992，max_abs_diff 中位 0.0039**（logit 值域 ~[−13,1]，纯 bf16 噪声）。
⇒ 抓的行/列/层没有错位，「同一批样本同一套判据」这个前提成立。
另：实测 `img_span = (14, 270)`，与 RO-2 报的 span **逐位一致**。

### 5.3 A&S 残差确实非零（不是把场减没了）

单样本实测：`post_self` 峰值 **0.0825**（GL token 给单个 image token 的最大注意力概率），
`post_mean`（输出序列平均图）峰值 **0.0343**。GL token 的图比「平均输出 token 的图」
**峰锐约 2.4 倍** ⇒ 减完共模后残差非零，A&S 不是在做无用功。
全批的残差幅度比 `‖s_aas‖/‖s_post‖` 报在 `metrics.sanity_and_residual`。

### 5.4 吞吐与两次返工

- **bug 1**：`np.savez_compressed` 自动补 `.npz`，原子写的 tmp 名变成 `.npz.tmp.npz`，
  `os.replace` 找不到源文件 → 改 `<key>.tmp.npz`。
- **性能返工**：首测 `t_gen = 139.6 s/样本`（G1 当时 10 s）。定位为 **CPU 线程超订**——
  进程内 121 线程 vs 机器 load 141/48 核。设 `OMP/MKL_NUM_THREADS=4` +
  `torch.set_num_threads(3)` 后降到 **53 s**；再改 4 路分片并行（GPU 实测仅
  **2.5 GB/进程**）后聚合吞吐 ≈ **172 次读出/小时**。
  **全程未 kill / 未 SIGSTOP 任何不属于本实验的进程。**

## 六、分析前预注册的结果解读（防事后合理化）

写定于跑批之前：

1. **若 `ρ_floor` 高而 `ρ_region_opp` 低（sep_matched 显著 > 0）且 AUC_target 显著 > 0.5**
   → 判 **(b) 读法错**。G1 的 FAIL 是 canonical 读法的伪影，信息确实在 VLM 里。
   后续降级为工程问题：训一个廉价读出去蒸馏 A&S 的场。
2. **若 A&S 的 ρ 全线走低（`ρ_floor` 也低）而 AUC_target ≈ 0.5**
   → **不是**正面证据。这只说明减均值放大了噪声。必须明确写成
   「低 ρ 有噪声与真信号两解，AUC_target 判给噪声」——这是 G1 §五 关掉 D-32 时用过的同一把尺。
3. **若 `ρ_region_opp` 依旧 ≈ `ρ_floor` ≈ 0.88 且 AUC_target ≈ 0.5**
   → 判 **(a) 假设错**（在这条 query×key 通道上没有随指令变的 where 信息）。
   **重大负面结论，如实报**；同时注意其边界：只证否了「输出 token query × image token key」
   这一族读出，未证否「VLM 内部有 where 信息」（那由 RO-1/RO-2/RO-3 的探针回答）。
4. **shuffle 反超**：若 A&S 下 `ρ_shuf > ρ_syn` 复现 → 协议/构造层共性问题
   （最可能是表面形式距离：shuf 与 syn_a 都是完整自然指令，而 syn_b 是 `instruction_short`，
   句长差异大）；若不复现 → canonical 读出特有的伪影。

---

## 七、AUC 的 GT 换列（主 agent 追加要求，2026-08-03 17:2x）

### 7.1 问题与处置

主 agent 指出：G1 / RO-1 / RO-X1 的 AUC 用 **SAM3 主体掩膜**当 GT，测的是「能不能找到主体」，
不是「能不能找到指令说的那块」；RO-X1 已用一条对 214 张图**完全相同**的短语
`"the main subject"` 拿到 AUC **0.907** / AUC_target **0.523**，把这个基线钉死。

处置：**s 场已落盘，换 GT 只是重算 AUC，零 GPU 成本**。本实验 AUC 一律**并排两列**
（第三列见 7.3 的待决策）：

| 列 | GT | 角色 |
|---|---|---|
| `cgt` | **`.cgt.png` 逐候选区域掩膜** = 指令实际改动的那块 | **主列** |
| `sam3` | SAM3 主体掩膜 | **对照列**，只用来量化「光靠找主体能拿多少分」 |

**不依赖 GT 的指标**（ρ_region_opp / ρ_floor / ρ_shuf / 跨 token ρ / 尖刺度）**口径不变**，
与 G1 保持配对可比。

### 7.2 C_GT 链路（逐环节实测，不猜；`config/build_cgt_masks.py`）

```
区域批配置 sft_id → journal sft.jsonl 同 id 行 → candidate_id + local.mask_id
                 → 银行 groups/prod-l*/batch-* 里 meta.candidate_id 相同的 sample
                 → 该 sample 的 .cgt.png 成员（sqlite 索引 + seek 直读 tar）
```

- 覆盖：**sft 行 214/214 命中，候选 214/214 在银行命中，C_GT 掩膜 214/214 落盘**，缺失 0。
- 对齐口径与 luma/SAM3 **同一函数同一路径**（短边 512 bilinear → expand2square 黑边 pad
  → 面积均值下采样 16×16）。
- **软边阈值**：缓存存**连续值**，分析侧二值化阈值由 `--cgt-thr` 控制，**默认 0.5**
  （与 SAM3 列的 `MASK_BIN_THR` 一致，保证两列可比）；敏感性由改该参数复跑得到。
- `winner_confidence=low` 按数据纪律**单独出一套** `auc_target_pooled_conf_normal`
  （区域批 152 normal / 62 low），正表主列用全体、并列 normal-only。
- 一图多候选的对应方式：**不靠猜**——`sft_id` 唯一确定 `candidate_id`，
  而 reg_a 的区域串正是同一行的 `local.subject.description`，所以掩膜与指令是**同一行的**。

### 7.3 ⚑ 实测：C_GT 与 SAM3 主体掩膜**不是同一个东西**（主 agent 的担心成立）

在区域批全部 214 源上（valid 格、阈值 0.5）：

| 量 | 值 |
|---|---|
| IoU(C_GT, SAM3 主体) 中位 | **0.374**（p25 0.194 / p75 0.750） |
| Pearson 中位 | 0.609 |
| **IoU < 0.5 的源占比** | **61.5%** |
| 面积中位 | C_GT **0.289** vs SAM3 **0.090**（差 3.2×） |

⇒ **两个 GT 在多数源上指向明显不同的区域**，「换 GT 不影响结论」这句话必须实测而不能假设。
（注：G1 NOTES §9.2 报过「C_GT 与 SAM3 IoU 0.913」，但那是 T4 `source_catalog` 的
`semantic-*` 槽、仅 n=79 的子集，**与本实验用的逐候选 `.cgt.png` 不是同一批掩膜**，
不能据此认为两者可互换。）

### 7.4 待主 agent 决策（新增，保守默认已采用）

| # | 决策点 | 保守默认 |
|---|---|---|
| **Dg** | **D-CONSTRUCT 第三列**（主 agent 要求的诊断列）：`T4_construct/sanity/val/L1` 的 `manifest.jsonl` **没有自然语言指令字段**（只有 `mask_kind=semantic_binary` / `transform` / `provenance`）。而本实验的读出**必须有指令文本**才能条件化，所以这一列不能像换 GT 那样零成本重算：要么从 `provenance.sample_id → candidate → group → sft 行` 反查区域描述再合成指令（可行，但合成的指令不再是「与掩膜同时构造」），要么另跑 24 源 × 2 指令 ≈ 48 次读出（约 25 min GPU） | **本轮不做**，如实标注原因。替代物：①主列已换成 `.cgt.png`（同样是「指令实际改动的那块」，且 n=214 远大于 24）；②「光靠找主体能拿多少分」用 RO-X1 已钉死的 `X_deictic` 基线（固定短语 AUC 0.907 / AUC_target 0.523）当参照。若主 agent 认为必须补，配置与代码路径已就绪 |
| **Dh** | C_GT 软边二值化阈值 | **0.5**（与 SAM3 列一致以便并排）；连续掩膜已缓存，改 `--cgt-thr` 即可扫敏感性 |
