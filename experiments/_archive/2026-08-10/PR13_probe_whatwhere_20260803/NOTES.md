# NOTES — PR-1（颜色探针逐层扫描）+ PR-3（空间探针 + H3 判决图）

- **实验编号**：EXPERIMENTS_v3 §2.2 探针臂 PR-1 行 + PR-3 行
- **分支**：`lens-exp`｜**卡**：`CUDA_VISIBLE_DEVICES=0`｜**实测峰值显存 1.64 GB**（上限 10 GB）
- **日期**：2026-08-03

---

## 〇、实施前三件事（CLAUDE.md 派工协议）

### 1. 已读的文档节（只读引用节）

| 文档 | 读了哪一节 | 取到的硬约束 |
|---|---|---|
| `EXPERIMENTS_v3_2026-08-02.md` | §2.2 探针臂表（PR-1/PR-3 两行）+ 全部 Changelog | PR-1 四条预注册判据；PR-3「两峰层距 ≤1/4 深度」；Changelog 2026-08-04 条：G1 终判 FAIL，**被证伪的是读法不是假设**（H2 未证伪）；D-31 措辞纪律 |
| `PLAN_v2_local-retouch_2026-07-31.md` | §3 第三级「腿 A 颜色探针」全节 + 「腿 A' 空间侧」+ §0 的 H1/H2/H3 原文 + §9.3 禁写清单 | A1–A6 属性定义（含命门 A5/A6）；探针阶梯 P0–P4；C1–C5 对照；行为五档；预注册阈值原文；干预对象=整段 image token |
| `DATA_ASSIGNMENT_2026-08-02.md` | §1 切分纪律 + §2 D-PROBE/D-MASKBANK 行 + §3.4 PR 系表 | **PR 系一律 S-val 源**；D-PROBE「RAISE 池优先」；A5 用人像池；「按源再切折，杜绝同源跨折」 |
| `experiments/G1_s_identifiability_20260803/REPORT.md` | 全文 | G1 = FAIL 的具体读数（ρ_region_opp 0.884、AUC(a)−AUC(b)=+0.0005、p=0.121）；canonical L8–15；214 源区域对立批 |
| `experiments/RO9_layer_verdict_20260804/REPORT.md` | 全文 | 逐层 `auc_target_light_bg` 24 层全在 0.479–0.526；建议「AUC_target 升为 RO 系通用判据」——本实验采纳 |
| `tools/readout/ro9_gl_attention.py` | 全文 | 复用：模型加载参数、`luma_to_grid`（expand2square pad 对齐）、D-0 outlier 口径、GL token id 151646 |
| `tools/harness/stats.py` | 全文 | 配对 bootstrap CI / 符号检验口径 |

### 2. 在线核实记录（IMPL_DOSSIER 附录 B 之外的每一条）

> 本项目检索引擎有编造前科。以下每一条都**打开了原始来源**并抄回原文，不是转述。

| # | 事实 | 来源（已打开） | 核实到的原文/数字 |
|---|---|---|---|
| V1 | **selectivity 的定义** | Hewitt & Liang, EMNLP 2019, ACL Anthology **D19-1275** PDF（本地抽取正文） | Fig.2 题注原文：*"Selectivity is defined as the difference between linguistic task accuracy and control task accuracy"*。control task 原文：*"associate word types with random outputs"*，且 Voita&Titov 转述得更细：*"each word token is assigned its type's output, regardless of context"* |
| V2 | **selectivity 的正则纪律** | 同上 | 摘要原文：*"dropout, commonly used to control probe complexity, is ineffective for improving selectivity of MLPs, but that other forms of regularization are effective"* → 本实验探针**只用 weight decay / ridge α，不用 dropout** |
| V3 | **MDL online（prequential）码长公式** | Voita & Titov, EMNLP 2020, **arXiv:2003.12298** PDF 式 (4)（本地抽取正文） | `L_online(y_1:n|x_1:n) = t1·log2 K − Σ_{i=1}^{S−1} log2 p_θi(y_{ti+1:ti+1}|x_{ti+1:ti+1})` |
| V4 | **MDL 的 timesteps 与 compression 定义** | 同上，脚注 4 与 Table 6 题注 | 脚注 4 原文：timesteps = *"0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.25, 12.5, 25, 50, 100 percent of the dataset"*；Table 6 题注：*"compression – with respect to the corresponding uniform code"* → compression = n·log2K / L_online |
| V5 | **LEACE 的编号与保证** | **arXiv:2306.03819** abs 页 | *"a closed-form method which provably prevents all linear classifiers from detecting a concept while changing the embedding as little as possible"*；概念擦除跨层版叫 "concept scrubbing" |
| V6 | **CIE D 系日光轨迹公式** | Wikipedia *Standard illuminant* 条目 | 4000–7000 K：`x=0.244063+0.09911(1e3/T)+2.9678(1e6/T²)−4.6070(1e9/T³)`；7000–25000 K：`x=0.237040+0.24748(...)+1.9018(...)−2.0064(...)`；`y=−3.000x²+2.870x−0.275` |
| V7 | **sRGB↔XYZ 矩阵与 EOTF** | Wikipedia *sRGB* 条目（IEC 61966-2-1 + 2003 修订） | 正矩阵 `[[0.4124,0.3576,0.1805],[0.2126,0.7152,0.0722],[0.0193,0.1192,0.9505]]`；逆矩阵 7 位版 `[[3.2406255,−1.5372080,−0.4986286],...]`；EOTF 阈值 0.04045 / 0.0031308，a=0.055，指数 2.4，线性段 12.92 |
| V8 | **本项目模型真实结构** | **从 `/home/bc/data/models/VeraRetouch/config.json` 实测**（不信任务卡转述） | `model_type=llava_qwen2`，`num_hidden_layers=24`，`num_attention_heads=14`，`num_key_value_heads=2`，`hidden_size=896`，`mm_hidden_size=3072`，`mm_vision_tower=mobileclip_l_1024`，`mm_projector_type=mlp2x_gelu`，`image_aspect_ratio=pad`，`tie_word_embeddings=true`，`vocab_size=151664` |
| V9 | **image token 数与视觉塔栈深** | 冒烟前向实测（`scratchpad/smoke1.log`） | image span = **256** token（16×16）；视觉塔 `FastViTHD.network` = **11 个 stage**（任务卡未提，实测得）；探针位点总数 **159**，拼接维 D 见 `config/env.json` |
| V10 | **线上颜色读出口的确切形状** | 源码 `llava/model/VeraRetouch.py:404-425` + `configs/infer_config.yaml` | `retouch_latent = retouch_head(concat(h[light], h[colortemp], h[colormixer]))`，三个 latent 取**末层**（`lens_readout_layer` 默认 −1），`retouch_head_in_dim=2688 = 3×896` → 「3 个 latent query token」的说法在代码层面成立，本实验的 `readout.actual` 位点 = `llm.lat3.L23` 与之逐元素相同 |

**未能核实 / 不可得**：
- Wyman-Sloan-Shirley (JCGT 2013) 的 CIE CMF 解析拟合：JCGT 页面抓不到正文。→ **规避**：A1/A6 的色温改用 **CIE D 系日光轨迹闭式**（V6 已核实），不做 CMF 积分，因此不需要该拟合。
- Cube+ / NUS-8 光源 GT 数据集：本机 `/home/bc/data`、`/mnt/nfs/bc/data` 全盘搜索**无命中**（见「待主 agent 决策 D3」）。

### 3. 假设与待确认清单

已当场自行核实的（不占决策）：模型层数/头数/token 数（V8/V9）；SAM3 主体掩膜银行可用且与 G1 同一份；`splits.sqlite3` 的 S-val 池规模（raise6k 118 / unsplash 541 / ppr10k 208 / korean 95 / awards 331）；`.venv-lens` 无 `colour-science` 与 `concept-erasure`（故两者均自实现，见 `tools/probe/colorops.py` 与 `run_causal.py`）。

---

## 一、待主 agent 决策（**保守默认已采用并继续**，未静默拍板）

| # | 决策点 | 两种做法 | **本实验采用的保守默认** | 影响 |
|---|---|---|---|---|
| **D1** | **selectivity 的回归版口径** | (a) `R² − R²(标签置换)`：在分组折下 control 必然 ≈0，该数 ≈ R²，**不是独立证据**；(b) H&L 忠实版：control task = 「按 source_id 分配随机目标」，且必须在**样本级折**（同源可跨折）上算才让 control「原则上可被记忆」 | **两个都算、都报**；预注册的 `selectivity≥0.25` 判据要求**两个都过**（更严）。主表用 (b) `selectivity_hl` | 若主 agent 认为只该看 (a)，判据会变松；反之若只看 (b)，样本级折的 R² 不能当主 R² 用（同源跨折） |
| **D2** | **A6 的光源 GT 来源** | (a) 外部 Cube+/NUS-8（DATA_ASSIGNMENT 指定，**本机不可得**）；(b) 自建 gray-world 陷阱集：从 S-val 里挑全局色度偏中性最远的头部源，施加**已知** D 系光源，GT 由构造给出 | **(b)**，并把「陷阱度」`chroma_bias` 随样本落盘。理由：(a) 需要下载外部数据且与 S-val 纪律无关；(b) 的 GT 精确、陷阱可量化 | 若主 agent 要求对外发表时用公开光源基准，需补跑 (a)。本实验的 A6 结论只能声称「在我方构造的陷阱集上」 |
| **D3** | **A6 用绝对特征而非配对 delta** | (a) 严格照 PLAN「全部配对 delta」；(b) A6 用单图绝对特征 | **(b)**。理由：配对 delta 下**像素差本身就精确编码了施加的光源**，C5 会接近满分，「gray-world 陷阱」这个设计意图被自己拆掉——那样测的不再是色恒常 | A6 与 A1–A5 的口径不同，报表中已单列 |
| **D4** | **三个 retouch latent 的取位** | (a) 让模型 greedy 生成到自然产出这三个 token（=线上真实位置，但 G1 实测 40–60 s/样本，4000+ 次前向不可行）；(b) 在 prompt 末尾**固定追加**三个 token，一次 teacher-forced 前向读它们的 hidden state | **(b)**。G1 的 fallback 分支用的就是追加法，且 local 段 fallback 率为 0 说明两者在局部任务上差异小 | 若「读出口 R² 低」是追加位置造成的伪影，结论会翻。**已加一致性检查**：`llm.last.L23`（真正的末位置）单列一行，两者同时低才下结论 |
| **D5** | **颜色探针与空间探针共用中性 prompt** | (a) 各自用最有利的 prompt；(b) 两者共用同一条固定中性指令 `"Retouch this photo."` | **(b)**。H3 判据是「两条曲线的峰值层距」，若两条曲线来自不同输入分布，层距没有意义 | 空间探针的**指令条件**分支（S2）另用 G1 的 reg_a/reg_b 指令，单列不进 H3 主曲线 |
| **D6** | **C2 随机骨干的范围** | (a) 只随机 LLM；(b) 视觉塔+connector+LLM 全随机 | **两个都跑**（C2a / C2b）。只做 (a) 时 `conn.out` 位点与真实模型完全相同，会被审稿人指出对照不完整 | — |
| **D7** | **A5 的「肤区」定义** | (a) 颜色阈值肤色检测；(b) SAM3 主体掩膜（语义定义） | **(b)**。(a) 会让「需要语义才算对」的命门自我拆台——用颜色定义区域，像素基线当然也能算 | A5 的区域是「主体」而非严格「皮肤」，故属性名在报表中写作 **A5 语义区域记忆色偏差（人像池）** |
| **D8** | **对照 run 的规模** | (a) 与主 run 同规模（每个对照 +4260 次前向）；(b) 60 源 × A1/A2/A3 | **(b)**，并在报表中标注对照的 n 小于主 run | 对照的 CI 更宽；若某个对照贴着判据线，需按 (a) 补跑 |

---

## 二、设计要点（为什么这么测）

### 2.1 颜色探针（PR-1）

- **配对 delta**（PLAN §3）：同一张图施加已知扰动，探**两版表征之差**。这样逐图的内容差异被自动消掉，标签是纯粹的扰动量。
- **六个属性**：A1 ΔCCT(mired) / A2 ΔWB(δ) / A3 ΔEV(stop) / A4 tone-curve 斜率 / **A5 语义区域记忆色偏差**（命门）/ **A6 gray-world 陷阱光源**（命门，绝对量，见 D3）。
- **位点**：159 个（视觉塔 13 + connector 1 + LLM 24 层 × 6 家族 + 线上读出口）。**LLM 每层三位点** = resid / attn 子层输出 / mlp 子层输出；**control latent 单独一路**（`llm.lat3.L*`）；**sink/register 单独一路**（`llm.sink.L*` = 该层范数最大的 image token）。
- **关键对比**：`max_L R²(llm.resid.L)` vs `R²(readout.actual)`。前者高、后者低 = 「信息在里面，读出层丢了」的**直接测量**（负责人的主张）。

### 2.2 空间探针（PR-3）

- 两个任务：
  - **S1 图像驱动 where**：中性指令下，token 级线性探针预测 SAM3 主体掩膜（16×16，覆盖≥0.5 判正，只取非 pad 格）。这条曲线进 H3 主图。
  - **S2 指令条件 where**：用 **G1 区域对立批的同一批 214 源、同一对指令（reg_a 点名主体 / reg_b 点名补集）**，探针目标 = **被指令点名的区域**。混池 AUC>0.5 才说明读到了指令条件性——纯主体检测器在混池上必然回到 0.5。判据量沿用 RO-9 REPORT §五.4 建议的 `AUC_target`（阈值 0.65）。
- **为什么必须是同一批源**：只有同源同掩膜同指令，才能把「探针能读到」与「RO-9 的 attention 读不到」并排，从而定位是**哪一步**坏的。

### 2.3 H3 判据的两种读法（本实验都报）

- PLAN §0 的 H3 原文：「存在层 L，颜色探针与空间探针同时接近各自最优（**统一读出层**）」，证伪条件 = 两峰层号相差 > 1/4 深度。
- 任务卡的读法：峰距 ≤ 1/4 深度 ⇒ what/where **纠缠**，分头读没有表征依据；> 1/4 ⇒ **可分离**，分头读由结构决定。
- 两者是**同一个数的两种叙事**（H3 成立 = 统一读出层 = 纠缠；H3 证伪 = 双层读出 = 可分离）。REPORT 中两种表述并列，不许只写对自己有利的那一种。

---

## 三、红线自查

| 红线 | 本实验的落实 |
|---|---|
| attention 导出必须 eager，返回 None 不回退 | 模型以 `attn_implementation="eager"` 加载并**运行时断言** `config._attn_implementation == "eager"`（`tools/probe/vlm_features.py`）。本实验主体**不导出 attention**（探针读 hidden state），G1 对照曲线直接引用 RO-9 已落盘的 eager 结果 |
| 干预对象 = 整段 image tokens 非 last token | `run_causal.py::Intervention._hook` 只改写 `lens_image_spans[0]` 给出的 `[i0:i1)` 共 256 个位置；C3 token 置换同样作用于整段 image token |
| 探针禁逐图归一化特征 | 特征侧只做「跨 token 平均池化」；标准化在探针拟合端，且**只用外层训练折**的均值/方差（`probes.ridge_cv`） |
| 每个探针结论必须带对照组 | C1（标签置换 + H&L control task）、C2a/C2b（随机骨干）、C3（token 置换）、C4（灰度输入）、C5（像素统计上界基线 + C5+ 缩略图强化版）全部实现；空间侧另有 C1a 格位固定随机标签 |
| 判据 PASS 必须区分「未触发死刑」与「正面证据」（D-31） | `metrics.json` 每条结论带 `evidence_class ∈ {positive, not-death, causal}`；REPORT 结论章逐条标注 |
| S 折纪律 | 全部折按 `img_id` 分组（`probes.grouped_folds`）；唯一的样本级折用在 D1 的 H&L selectivity，且已明确标注不能当主 R² |

## 四、算力与排卡

- 卡 0，`CUDA_VISIBLE_DEVICES=0`，**实测峰值 1.64 GB**（冒烟 `scratchpad/smoke1.log`）。抽取阶段最多 2 个进程并行 → ≤3.5 GB，远低于 10 GB 上限。
- CPU：所有拟合走 GPU 张量运算，未调用 sklearn（`n_jobs` 无从加剧）；无 dataloader worker。
- **未 kill 任何非本任务进程**。
- 长任务 nohup 后台 + `job.marker`（PID / 完整命令 / 日志路径）。

## 五、已知局限（供结果审阅 agent 核）

1. A5 的区域是 SAM3 **主体**掩膜（人像池上即人物），不是严格的皮肤分割；「记忆色」参照取该源自身扰动前的区域 Lab，故标签是**区域色位移**而非绝对记忆色偏差。
2. A6 陷阱集是自建的（D2），外部公开光源基准未跑。
3. 三个 retouch latent 用**追加**位置读（D4），非生成位置；已用 `llm.last.L23` 做一致性旁证。
4. 对照 run 规模小于主 run（D8）。
5. 行为侧的双联图输入与探针主 run 的单图输入不同；因此「行为 vs 探针」的比较**只用同一次前向的 matched probe**，逐层曲线不参与该比较。

---

## 六、实施过程中发现的事实与新增待决策（跑数后追加）

### 6.1 三条实现层面的坑（都会**系统性虚高 VLM 相对基线的优势**，故必须留档）

| # | 现象 | 根因 | 修法 | 修前 → 修后 |
|---|---|---|---|---|
| T1 | C5 像素基线折外 R² = **−18.6** | 近似常数的特征列标准化后变成数值巨大的噪声主方向 | 训练折内 std < 1e-8·max(std) 的列直接置零 | −18.6 → −0.39 |
| T2 | 同上仍为 −0.39；A6 的 VLM 探针 R² = **−6.1** | α 选择用了**按样本**的留一 / 闭式 block-LOGO：(a) 同源 4 个扰动样本互为近邻，误差被严重低估；(b) block-LOGO 在近插值时 (I−H_BB) 与残差同时趋 0，数值上给出**虚假的低误差** | 改为**按源分组的 3 折嵌套 CV + 1-SE 规则** | C5 −0.39 → **+0.98**；A6 −6.1 → **+0.13** |
| T3 | 第一版 matched probe 的 MAE **恰好等于常数预测器** | 双联图下把 256 个 image token **整段平均**，正好把「左右两半之差」这个待读量本身平均掉 | 改为**左半均值 / 右半均值 / 二者之差**三路（`run_matched_probe.py`） | MAE 44.5（=常数） → **24.4**（R²=0.61） |

T1/T2 的说明已写死在 `tools/probe/probes.py` 的 `_standardizer` 与 `ridge_cv` 文档字符串里，
防止后续实验重蹈。

### 6.2 一条**未被任何文档预见**的结构事实

`<image>` 在 prompt 的第 75 字符、`Instruction:` 在第 239 字符 —— **image token 在因果序里
先于指令**。decoder-only 因果注意力下，image token 的 hidden state **在结构上**不可能依赖指令。
实测复核：reg_a/reg_b 下 image token 的相对 L2 差 0.0033（bf16 舍入量级）、
「哪条指令」从 image token 解码 AUC = 0.503、线性 S2 探针 24 层全部恰为 0.500。

**这直接改变 RO-9/G1 结论的归因**：RO-9 的失败**不需要**用「表征里没有 where」解释，
用因果序就能解释。见 REPORT §2.4 与建议 N3（把指令前置的一行消融，成本 ~1 GPU-hour）。

### 6.3 新增「待主 agent 决策」

| # | 决策点 | 说明 | 本实验采用的保守默认 |
|---|---|---|---|
| **D9** | **要不要做 prompt 布局消融（N3）** | 若把 `Instruction:` 移到 `<image>` 之前就能让 image token 携带指令，则整条 where 叙事（含 RO-9 判死、本实验的双线性读出）都要重写。这是一个改 prompt 模板的动作，会影响**线上模型的输入格式**，属于跨实验决策 | **不改**，只把它写进建议 N3 |
| **D10** | **what 线的叙事怎么改** | 本实验给出直接反证：颜色到 3-latent 读出口仍有 R² 0.85–0.90，且打不过像素统计。是把 what 线降级为「够用即可」，还是把主张限缩到 A6 类语义色彩量（断点在 connector） | **只报事实，不替主 agent 定调**；REPORT §五给出两个选项 |
| **D11** | **S2-bilinear 的二义性怎么消** | 每源只有两条指令，无法区分「读懂区域描述」与「识别主体/背景二分」。需要一图 ≥3 个命名区域的样本，而 G1 NOTES D8 已记录该数据不存在（需新建 build） | **如实标注为局限**，不做数据新建 |
| **D12** | **MDL 要不要保留** | 当前 online code 的 compression 在探针与 C5 上都 <1（0.51–0.89），不具判别力。要用需要先降维或换强正则探针族 | **保留数字但不据此下任何结论**；REPORT §七.5 已声明 |

### 6.4 红线自查（跑完后复核）

- 干预对象：`run_causal.py::Intervention._hook` 实测只改写 `lens_image_spans[0]` 的
  `[i0:i1)` 共 **256** 个位置；C3 token 置换同样作用于整段 image token。✅
- attention：本实验主体不导出 attention；模型仍以 `attn_implementation="eager"` 加载并断言。
  G1/RO-9 的对照曲线直接引用其已落盘的 eager 结果。✅
- 特征禁逐图归一化：只做跨 token 平均池化；标准化只用外层训练折统计量。✅
- 每个探针结论带对照：C1/C2a/C2b/C3/C4/C5 六组全部实跑并进表。✅
- D-31：`metrics.json` 每条判据带 `evidence_class`；REPORT §三专列相关性/因果性/结构性三分。✅
