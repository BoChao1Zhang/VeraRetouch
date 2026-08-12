# 低维几何码提取与注入架构调研（Where 头重设计支撑文档）

- 日期：2026-08-11
- 输入：六路文献调研（parsing / readout-token / query-bridge / pretrained-decoder / assembly / selection，全部条目 VERIFIED）+ 一轮完整性审计
- 性质：文献综合 + 选型处方 + 预注册实验设计。**本文不含任何本地实验新数据**；引用的项目内数字（0.7622、82%、+0.0007 等）均来自任务卡给定的问题设定。

---

## 1. 问题设定复述

**任务**：条件 dense 场预测——(image, edit instruction) → 软掩膜场。现状：冻结多模态 VLM（Qwen3-VL 系）+ 轻 conv 头（~4M）已把 soft-IoU 推到 (image, instruction) 的信息论天花板（重放上界实测 0.76–0.80，模型 0.7622）。

**超越天花板的唯一已定位信息源**：VLM 自回归生成的 reasoning 文本段。按 SFT 数据设计，它描述掩膜几何（形状家族 / 方向 / 范围），实测词级准确率 82%。核心问题：**如何把这份低维几何信息（约 20 维离散槽 + 少量连续量）从 VLM 中提取出来并注入 dense 头**。

**已确立的实证约束**：

1. **pooled hidden + FiLM 丢失该信息**：teacher-forced 完美 reasoning 文本只动 IoU +0.0007~+0.0087；
2. **历史教训**：learnable query cross-attn 从 VLM hidden 读**高维空间场**再回归系数曾灾难性失败（池化成单向量抹掉空间排布 + 回归目标病态）。本次目标是 ~20 维低维码，该死因是否豁免需要论证 + 实证（见 §2.C、§4.0）；
3. **VLM 默认冻结**；需要重新 SFT 的方案（如预留 special token）不是禁区，但成本与风险必须单列；
4. **现行对照（A 基线）**：冻结词表解析 reasoning 几何短语 → 20 维多热码 → broadcast 通道注入。脆弱性：文本模板规整性、解析器覆盖、82% 上游准确率；
5. **预算红线**：轻头 ~4M、数万样本、单卡数小时、推理低开销。

**一个必须显式写出的框架层事实**（审计 (4)-4）：推理期 reasoning 是从同一 (image, instruction) 自回归生成的，数据处理不等式下它**不含输入之外的新信息**；增益的真实来源只能是「VLM 权重里学到的标注约定先验，被自回归解码显式化，并以 dense 头能消费的形式重新注入」。teacher-forced GT 文本实验注入的是**额外**信息（真上界），与推理期可实现增益不是一回事。因此任何选型结论都必须挂在 §4.1 的配对上下界（GT 码 / 生成码 / 无码）上，这是整个方向的 go/no-go 数字。

---

## 2. 谱系地图

### 2.A 符号解析式（现行对照的可靠性工程上限）

**代表工作**：XGrammar (MLSys'25)、JSONSchemaBench (2501.10868)、OpenAI Structured Outputs（生产数字）、dottxt "Say What You Mean"、BAML Schema-Aligned Parsing、Grammar-Aligned Decoding (NeurIPS'24)、Know Your Limits 弃权综述 (TACL'24)、StructuredRAG、Let Me Speak Freely (EMNLP'24)、NuExtract、Prompt Formatting (MSR)。

**机制（三层结论）**：

1. **解析失败率可以工程化归零**，最优解是「把解析器当生成器跑」：推理期 constrained decoding 对生成段做 logit 掩码，把结构化摘要段约束进 20 槽封闭词表 CFG——解析覆盖率**按构造 = 100%**，XGrammar 证明开销近零（已入 vLLM/SGLang 生态）。OpenAI 生产线同款双层路线：SFT 学格式（93%）+ CFG 兜底（100%）。20 槽扁平枚举 + 数值正则属 JSONSchemaBench 里所有框架都覆盖良好的最简类别。
2. **若保留自由文本 + 事后解析**：严格正则是最差档（dottxt 实测 0.35，手工柔性正则 0.61，还赢过 LLM 当解析器 0.57）；BAML SAP（schema 感知最小编辑距离容错解析）在 BFCL 上 92–94%。StructuredRAG 警示纯 prompt 裸奔的格式层失败均值 ≈17% 且尾部极坏，列表型（多热码）恰属更难类别——**要么上约束解码，要么上容错解析，不能裸奔**。
3. **置信门控**：约束解码下每槽在允许 token 上重归一化的概率是天然置信度,但 GAD (NeurIPS'24) 证明 greedy 掩码使它**有偏**（虚高）——须用未掩码前原始 logit margin + 少量 val 样本做校准；三态门（置信 ≥τ 注入 / <τ 置 unknown、注入通道置零、退回 0.7622 基线）是 selective prediction 标准做法，**保证条件注入只增不减**。schema 里显式留 abstain/unknown 枚举值，避免强制填槽把不确定性变成静默错误。

**风险与规避**：Let Me Speak Freely 报格式约束伤推理，dottxt 复现反驳（prompt-结构对齐即不伤）；保守设计 = **两段式**——reasoning 段自由生成，尾部追加受约束的低维码摘要段，约束只落在码段。模板漂移防护：解析覆盖率 + 掩码触发率监控；漂移后备胎 = NuExtract 路线（数百条「reasoning→码」标注微调 0.5B 抽取器，单卡分钟级）。

**成本**：整层**零训练、不动 VLM**（若在 SFT 数据里加摘要段/abstain 值则动 VLM，须单列）。**失效条件**：格式层归零后，**瓶颈只剩上游 82% 语义准确率**——本路线对它无解（提升手段文献线缺失，见 §5）。

**与约束 2 的关系**：不经过 query/hidden，天然免疫历史死因；但其 broadcast 注入接口被四条独立证据（SPADE / READ / EVF-SAM / assembly 共识）判为**下界**，构成横向对比的混淆（见 §4.2 的接口对齐裁定）。

### 2.B special / readout token 式（两档，风险剖面相反）

**档 2（动权重）代表工作**：LISA (CVPR'24) → PixelLM (CVPR'24) → GLaMM (CVPR'24) → PSALM (ECCV'24) → OMG-LLaVA (NeurIPS'24) → Sa2VA (2025)。统一配方 = 词表加 <SEG>/mask token，其末层 hidden 投影后直接当 mask decoder 条件。PixelLM 证明无需 SAM、轻 decoder + 多 token codebook 即可；PSALM 把读出 token 显式插入输入 schema、hidden 当 Mask2Former query；Sa2VA 证明配方可装 Qwen-VL 且只需少量 instruction tuning（LoRA 档，8×A800×48h、1.1M 样本量级——远超本项目预算，仅作上界参照）。

**档 2 致命风险**：F-LMM 系统实测该类 SFT 使通用 QA / 对话能力**大幅退化**——对本项目致命，因为超越天花板的信息恰来自 reasoning 文本；若走此档必须把 reasoning 文本质量当一等监控指标单列。

**机制层关键证据（READ, CVPR'25）**：<SEG> 末层 hidden 本身只是语义查询向量，**几何信息不在 token 向量内部，而在「token 作 query × image token 作 key」的相似度图里**。这同时解释了本项目 pooled+FiLM 失败（broadcast 抹掉了交互项），并指明注入接口应为「码对 image/dense 特征算相似度出空间图」而非 broadcast 常量通道——本文所有注入设计据此裁定。

**档 1（不动权重）代表工作**：F-LMM（完全冻结 LMM，读回答 token→image token 的 word-pixel attention，仅训几层 CNN + SAM refiner，对话能力零损伤）；Petrov 理论 (ICLR'24)：soft prompt/prefix 不能造新 attention 模式，**只能引出模型已计算的信息**；ViT Registers (ICLR'24) 与 test-time registers (2506.08010)：外挂 token 天然被当聚合槽，且**测试时外挂未训练 token 即可吸收 sink**（直接呼应本项目 pad/sink 实测）；Readout Guidance (CVPR'24)：冻结生成模型 + 轻读出头提取低维几何属性的跨域先例（按属性分头、从中间层读，与本项目「层头差分」发现一致）。

**成本**：档 1 零 VLM 改动、只训轻 CNN/readout 头，在预算内；档 2 LoRA SFT + decoder 训练，超预算且带对话退化税。

**失效条件**：档 1 依赖冻结模型 attention 中天然存在 word-pixel 对应——Qwen3-VL 上 sink/pad 占比高则上限受限（须先做 attention 信息性探针 + pad 排除 + 可选 test-time register 净化；register 定位结论只在 CLIP/DINOv2 验证过，AR VLM 未证）；档 2 失效 = reasoning 能力退化吃掉信息源本身。

**与约束 2 的关系（裁定性）**：Petrov 理论给出干净分界——readout token 在冻结 VLM 上是「选择器」不是「计算器」。本项目 20 维几何码**已被模型计算**（能写成 reasoning 文本，82% 词级准确率），属「已有信息的引出」，恰在理论允许范围；历史灾难（从 hidden 回归高维空间场 = 要求新计算）的死因不适用。注意：该定理设定是 prefix 影响下游计算，外推到「尾部追加读出」超出原定理范围（审计已标，属合理但未证外推）；且理论只保证存在性、不保证优化可达——仍需 §4 的实证。

### 2.C learnable query 桥接式（MetaQuery 谱系）

**代表工作**：**MetaQueries**（Meta+NYU，arXiv 2504.06256，2025-04；锚点，核实详见 §2.F）、OpenUni（全开源复现，2505.23661）、MQT-LLaVA（NeurIPS'24，容量曲线）、DeCo（死因诊断，2405.20985）、Bifrost-1（反例，2508.05954）、Honeybee（CVPR'24，局部性）。

**机制**：在冻结 MLLM 序列尾拼 N 个可学习 query（保持因果掩码），last hidden 经可训连接器（Enc-Proj 序优于 Proj-Enc）喂下游；仅下游损失端到端训 query+连接器。关键实证：64 个 query 即匹配整段 last-layer 序列做条件（512 反超）；**query 桥在依赖推理的指标上显著优于把 LLM 当文本编码器的 last-layer embedding 读出**（WISE 0.55 vs 0.48、CommonsenseT2I 57.7 vs 52.8）——query 收割的是前向中「推理后」才出现的信息，正是本项目 reasoning 段几何信息的同类物。

**约束 2 的三角论证**（无直接先例，四路间接证据一致）：

- **DeCo**：query 压缩的已证死因是 double abstraction——丢的是**高维空间排布**（视觉定位差 7.1%），而 query 抽象的天然产物恰是「有限语义概念」= 低维槽位。死因机制本身预测 20 维码不受害；
- **MQT-LLaVA**：query 数-任务曲线显示 2 个 query 仍保住语义任务（ScienceQA/MMMU 仅 -3%/-6%），空间细节任务最先塌——低维载荷在极端压缩下存活；
- **MetaQueries**：冻结 MLLM + query 成立（其消融显示冻结与全调相当）；
- **Bifrost-1（反向划界）**：高维空间载荷应走空间索引化接口（patch 级 latent）而非 query——确认分工原则「**几何码走 query，空间场走 dense 头自己的通路**」；Honeybee 从投影器侧独立支持（query 类 abstractor 丢局部性，空间信息必须走保局部算子）。

**残余风险（两条警告合并成一条设计裁定）**：DeCo 警告 query 语义抽象丢连续量精度，READ 结论说几何在交互图不在 token 向量——**少量连续量槽（范围/程度）不走 query 分类，走独立回归旁路或解析通道**。

**落地配方**：reasoning 解码完成后在 KV cache 上追加 K=8–32 个 query 再走一次增量前向；last hidden → ≤4M 小连接器 → 20 槽分类 logits + 连续量回归。监督直接用 GT 几何码（SFT 数据可自动导出；若 GT 覆盖不全须澄清，见 §5）。Matryoshka 随机截断训练可**一次训练扫出整条 K 容量曲线**。

**成本**：VLM 完全冻结；连接器可裁到 <4M（MetaQueries 的 316M 连接器 + 百万级预训练是为对齐扩散条件这一高维目标，低维监督下可跳过——此点是推断非其实验结论）。**失效条件**：query 数不足时高维载荷先塌（对 20 维码不构成约束）；**全谱系无「20 维离散码 + 数万样本 + 冻结 VLM」的直接先例，低数据量收敛证据须自己的实验补**（query-bridge 调研自认，审计确认）。

### 2.D 预训练分割 decoder + VLM 特征对齐

**代表工作**：EVF-SAM（2406.20076）、F-LMM（整机模板）、SAMWISE (CVPR'25)、MAM（2306.05399）、SAM4MLLM (ECCV'24)、Grounded SAM、Sa2VA / LISA（SFT 档）、PSALM（Mask2Former 谱系，全参联训，仅借 schema）。

**四个结论**：

1. **冻结 decoder 可行、冻结条件编码器不可行**（全方向最硬单条证据）：EVF-SAM 消融——可训编码器 + 全冻 SAM(prompt encoder+mask decoder) = 82.9 vs 全调 83.7（仅 -0.8 cIoU），而**冻结编码器只训 decoder 崩至 21.2**。对齐负担必须落在「信号→prompt 空间」一侧。移植：20 维码经可训 embedding + 2 层 MLP 升成单个 256 维 sparse token、zero-init 拼接注入——单 token 容量足够，且 token 参与 decoder 逐层 cross-attn，优于一次性 FiLM（与本项目 FiLM 失败同构对照；「冻结 hidden 池化直连必败」的第二份独立证据 = 其 21.2 崩溃模式）。
2. **F-LMM 是 E 组装的现成模板**：冻结 LMM + 冻结 SAM ViT，只训 3-stage U-Net + refiner（数 M 参数）；空间先验场走 SAM prompt encoder 的 dense-prompt 通道，box + 逐层文本 hidden（M 个可学标量加权和）走 sparse 通道。消融：U-Net vs 平 CNN 仅 +0.4——**头的容量不是瓶颈，先验场质量才是**。SAMWISE 补充：中间层 <5M cross-modal adapter 注入也成立（多尺度逐层注入，规避单点注入被下采样冲淡）。
3. **decoder 先验 vs 软衰减场的风险有实证**：MAM **不**复用 SAM decoder 出 alpha，而旁挂 2.7M M2M 模块读 SAM 特征迭代细化——锐边界/objectness 先验与羽化场冲突。**推荐配方 = decoder 只供粗几何，软场仍由自训头输出**；但注意（审计 (3)-2）：该组合配方**本身无 verified 先例**，是从 MAM 反推的假设，必须以消融 {自训头 only} vs {自训头 + decoder 粗几何} 预注册验证。且 MAM 的 alpha 连续性集中在发丝级窄带,与本项目大面积平滑衰减场（天空渐变、肤色区）分布不同——直接把 SAM decoder finetune 到软目标在 2024–26 文献中**无成功先例，高风险路径**。
4. **免训下限与 SFT 上限**：SAM4MLLM 证明文本几何原语（点/框）可直接喂冻结 SAM——A 路的上限形态，受 82% 封顶，与 A 同池不提供独立增量；Sa2VA/LISA 的 [SEG]-hidden→linear→decoder 需 LoRA SFT，列「动 VLM」档（Sa2VA 关键消融：去掉 image seg 联训数据 RefCOCO 崩至 20.2——该能力靠 SFT 数据喂出，冻结 VLM 拿不到；另注意其 decoder 冻结/可调存在论文内口径差，引用须注明）。Grounded SAM 是级联误差反面教材：接口越低维越免训，但误差不可恢复，纯级联不可取，须保留 dense 兜底通路。

**成本**：对齐层路线不动 VLM，可训参数 2.7–5M 量级，契合预算（F-LMM 原配方 190k 样本 8×A800×20h 高于预算,缩到 3–5 万样本未验证）。**失效条件**：SAM prompt 空间带强 objectness 先验，修图域软边/非物体区域掩膜可能系统性不贴合，须先小样本域验证；EVF-SAM 横评警告**纯语言侧 embedding 对齐 SAM 最差**——码须与图像特征融合后再投,不可单独直投 prompt。

**概念性 caveat（审计 (3)-3，本文显式写出）**：重放上界 0.76–0.80 是 (image, instruction) 的**歧义界**；SAM 的 objectness 先验不携带本数据集 GT 约定的信息。故 D 的增量**只能经由「让几何码被更好地利用」实现**（更强的码→形状展开先验），不存在「decoder 强 = 必有增量」。D 档的价值命题是接口质量，不是新信息。

**与约束 2 的关系**：D 本身不经过 query；其与 C 组合时（query 读码 → decoder prompt），沿用 §2.C 的豁免论证。

### 2.E 组装范式（码 + 场 + 头/decoder 的接口设计）

**代表工作**：SAM 三轨接口（论文 + mask_decoder.py 源码级核实）、SurgicalSAM (AAAI'24)、F-LMM、SEEM (NeurIPS'23)、AnyControl (ECCV'24)、Uni-ControlNet (NeurIPS'23)、Compose and Conquer (ICLR'24)、Composer、SPADE (CVPR'19)、EVF-SAM、PSALM schema、Text4Seg (ICLR'25)、MM1。

**共识一：低维码与空间场走不同接口，不共用一路 broadcast**。SAM 源码三轨是现成蓝本：

| 轨 | 载荷 | 实现 | 本项目对应 |
|---|---|---|---|
| dense prompt | 空间场 | conv 嵌入后逐元素加到 dense 特征（`src + dense_prompt_embeddings`） | 空间先验场 → conv 头/decoder 的 dense 侧 |
| sparse token | 低维码 | token 化进双向 transformer cross-attn | 20 维码 → 条件 token 逐层交互 |
| hypernetwork | 全局码→逐像素 | mask token 经 3 层 MLP 变分类器向量，与上采样 dense 特征点积 | 分辨率失配的标准桥：码→逐像素分类器权重，**非 broadcast** |

SurgicalSAM 是「~20 维离散码→预训练 decoder」最直接先例：离散类 ID→可学习原型→prompt embedding 喂冻结 SAM decoder（<1% 参数）；对比原型学习拉开相近几何类（椭圆 vs 圆角矩形）。移植：20 槽各设原型，多热码检索原型加权和→线性投影成 1–N 个 sparse token；连续量作原型插值系数或走旁路。

**共识二：冲突处理无手写规则的成功先例，全靠训练策略**：SEEM 联合 prompt 空间 + 随机组合训练；Composer / Uni-ControlNet 逐条件 dropout 换推理期任意组合（数万样本下须提高「保留全部条件」采样比）；AnyControl 用少量 query 学习性仲裁多条件；CnC 的 **soft guidance** 近零成本可移植——用空间先验场作 attention bias/mask 限制码 cross-attn 的作用域，码与场从两条独立证据变成**乘性绑定**（场限定「在哪」、码限定「什么形状」），须留一路不受 mask 的旁路防场错位。**训练期随机屏蔽码/场之一 + 构造码-场矛盾负样本**，否则推理期矛盾行为未定义。

**共识三：注入位置**。2024–26 **没有**干净的 broadcast vs 调制 vs cross-attn 三方系统对比（两轮检索确认）。最近证据：EVF-SAM（早融合显著优于晚期 prompt 注入——融合深度 > prompt 形式）、SPADE（输入 concat 被归一化层逐层洗掉——本项目 broadcast 基线 = 输入 concat，症状与 pooled+FiLM 失败同构）。MM1 泼冷水：低维信息下接口微结构差异会被训练规模抹平——**审计裁定（采纳）**：只做三类粗档对比（文本/broadcast、attention 切片、query），不做同类内细扫；预算优先花在「信息是否到场」的上下界验证。broadcast 通道基线大概率是下界。

**Text4Seg 旁证**：16×16 文本化掩膜（比 20 维码大两个数量级）经共设计表示 + SFT 可靠传输——**文本通道容量不构成 A 的上限**，上限在 schema/解析器共设计与 82%。

### 2.F MetaQuery / MetaCurves 核实结论（单列）

- 用户提及的 **"metaCurves" 查无此文**：arXiv 全字段检索零结果（"Sorry, your query for all: MetaCurves produced no results"），学术库检索亦无。
- 判定为 **MetaQueries**（*Transfer between Modalities with MetaQueries*，Meta + NYU，Xichen Pan & Saining Xie 等，arXiv **2504.06256**，2025-04）之误记。佐证：语境吻合（连接冻结 MLLM 与下游的 learnable query 接口、2025 年、名称高度相近），两路调研独立得出一致结论。
- 诚实边界：只能证 arXiv 与主要检索无果，**无法证明全网不存在**。建议主 agent 向用户做一次最终确认；若用户另有出处需提供线索再补核。

---

## 3. 横向对比表

| 方案档 | 信息保真上限 | 训练成本 | 推理成本 | 脆弱性 | 冻结约束友好度 |
|---|---|---|---|---|---|
| **A0 现行**：冻结词表解析 + broadcast 注入 | 82% 语义 × 解析覆盖 ×（**接口下界**：concat 被归一化洗掉） | 零 | 近零 | 高：模板漂移静默降覆盖、严格解析 0.35 档 | 完全冻结 ✓ |
| **A1 升级**：两段式约束解码 + 三态门 + 接口对齐注入 | 82% 语义（格式层按构造 100%）；门控保证只增不减 | 零（约束层）+ 注入头数小时 | 近零（XGrammar） | 低：漂移不可能（结构在解码器里）；残余 = GAD 置信偏差（可校准） | 完全冻结 ✓（若 SFT 加摘要段则单列） |
| **B1 冻结 attention 读出**（F-LMM 式） | 不经 20 维瓶颈,读 hidden/attention 全量；上限 = 冻结模型 word-pixel 对应质量（Petrov：只能引出已算信息） | 轻 CNN 头,数万样本单卡数小时 | 一次 eager attention 导出 | 中：sink/pad 污染、层头筛选依赖、transformers 版本敏感（F-LMM 仓库自注） | 完全冻结 ✓ |
| **B2 SFT 预留 seg/geo-token** | 端到端学码,绕过 82% 文本上限（理论最高） | LoRA SFT + decoder,8 卡×2 天、百万样本量级（Sa2VA 档） | 低 | **对话/reasoning 退化实测严重（F-LMM）——恰好吃掉本项目信息源**；数据量超预算 20 倍+ | 动 VLM ✗（单列档） |
| **C query 桥**（MetaQueries 式） | hidden 直读,可能 > 82%（生成采样有损、hidden 无损——待 P0 probe 证实）;推理后信息可达（WISE 0.55 vs 0.48） | query + ≤4M connector,数万样本单卡数小时 | 一次增量前向（复用 KV cache） | 中：无同设定先例,低数据收敛未证；连续量走 query 有精度风险（走旁路规避） | 完全冻结 ✓ |
| **D 免训**：文本原语→冻结 SAM（SAM4MLLM 式） | = A 同池（82% 封顶）,无独立增量 | 零（复用 reasoning 文本） | 加一次 SAM 前向 | 级联误差不可恢复；点/框对软边表达力不足 | 完全冻结 ✓ |
| **D 对齐**：码→256 维 token→冻结 SAM2 decoder + 自训软场头 | 同上游码质量；增量 = decoder 形状先验对码的展开能力（无新信息,见 §2.D caveat） | 对齐 MLP + 2.7–5M 旁挂头,数万样本 | 加一次 decoder 前向 | objectness/锐边先验 vs 大面积软衰减场错配（无先例配方,须消融自证）；纯语言 embedding 直投最差（须先融合） | 完全冻结 ✓（SAM2 亦冻结） |

**读表结论**：完全冻结、预算内、且有可能突破 82% 上限的只有 **C**（hidden 直读）与 **B1**（attention 直读）；A1 是可靠性收敛后的最强符号基线；B2 高上限但税负与预算双重出局（保留为远期档）；D 对齐是接口增强件而非独立信息源，挂在其他臂后面做可选组件。

---

## 4. 选型处方与对照实验设计

### 4.0 前置探针 P0（先于一切臂,单卡小时级,把「推断」变「证据」）

- **P0a：hidden 线性/MLP probe**。在 reasoning 段（几何短语 token 位置及段尾）hidden 上训线性/2 层 MLP probe → 20 维码,S-val 报逐槽与宏平均准确率。**判读**：probe ≥ 82%（文本词级）⇒ hidden 读出存在「免费上限」,C/B1 前提成立且可能白赚上游精度；probe < 70% ⇒ 冻结读出档前提削弱,降优先级、A1 升主力。顺带补掉 probing 文献线缺失（§5）。
- **P0b：attention 切片信息性探针**。复用已有 eager 导出管线：几何短语 token → image token 的 attention（层×头筛选、pad 格显式排除、可选 test-time register 净化）,逐槽做线性可分性测试,与 shuffle 对照。不过线 ⇒ Arm 2 不全训。
- **P0c：Matryoshka 容量曲线**（可并入 Arm 1 训练）：随机截断 K∈{2,4,8,16,32},一次训练扫出 query 容量曲线,直接回答约束 2 的「多少 query 够 20 维码」。

### 4.1 Go/no-go 配对上下界（整个战役的裁决数字,预注册）

同一注入接口（§4.2 共享注入模块）下三档配对：

| 档 | 定义 | 角色 |
|---|---|---|
| U_GT | GT 几何码注入 | 可实现增益上界（含标注约定信息） |
| R_gen | 推理期生成码注入（各臂各自的码） | 可实现增益 |
| 基线 | 无码（0.7622 现行头） | 零点 |

**裁决**：
- **U_GT − 基线 < +0.005 soft-IoU ⇒ 方向 no-go**（信息或接口不成立；先排查接口——换 §4.2 三粗档最强者复测一次——再判死）。
- 各臂捕获率 = (R_gen − 基线)/(U_GT − 基线)；**晋级线：捕获率 ≥ 0.5 且配对 p<0.05 且 |Δ_shuffle| < 0.002**。

### 4.2 共享注入模块（先裁定,消除 A-vs-B/C 的接口混淆）

审计 (4)-2 指出的混淆必须先清：若 A 保持 broadcast 而其他臂用交互式注入,输掉的可能是接口而非符号解析范式。**处方：所有臂共享同一注入模块,横向对比只在「码的提取方式」上变化。**

注入模块设计（SurgicalSAM 原型 + SAM 双轨 + CnC soft guidance 合成）：
- 20 个离散槽各设可学习原型 embedding,多热码检索加权和 → 线性投影成 2–4 个条件 token;
- 条件 token 与 conv 头多尺度 dense 特征做逐层 cross-attn（READ/EVF-SAM「码×特征交互 + 早融合」依据）,并经 hypernetwork 出逐像素分类器权重与末层特征点积（SAM 第三轨,分辨率桥）;
- 可选 soft guidance：用头自身中间场作 attention bias 限定码作用域,留一路无 mask 旁路;
- **连续量槽不进原型检索,走独立小回归旁路直接进头**（DeCo+READ 合并裁定）;
- 训练期对码通道做 dropout（含全屏蔽 = 退回基线）+ 少量码-场矛盾负样本;
- 注入形式只做一次**三类粗档**对比选定（broadcast 下界 / 多层调制 SPADE 式 / 上述 cross-attn+hypernetwork）,选定后冻结设计全臂共用,不做同类内细扫（MM1 裁定）。

### 4.3 对照臂（3 臂 + 1 可选组件）

**Arm 0（基线,必跑）：A1 符号解析升级臂**
- 结构：两段式生成——reasoning 自由,尾部受约束摘要段（20 槽封闭词表 CFG + abstain 枚举,XGrammar logit 掩码;若暂不能改生成段,退 BAML SAP 式编辑距离容错解析器 + 柔性正则族）;三态置信门（置信 = 未掩码原始 logit margin,τ 在 S-val 风险-覆盖率曲线上预注册,禁逐图调）;码走共享注入模块。
- 训练量：约束层零训练;注入头 3–5 万样本、单卡数小时。
- 监控：解析覆盖率、掩码触发率（漂移报警）。
- 判据：R_gen(A1) vs 基线配对 Δ、Δ_const/Δ_shuffle 列、捕获率。**证伪数字**：若 A1 捕获率 < 0.3 且 P0a probe ≥ 82%,说明文本通道（生成采样）损失显著,权重移向 C/B1。

**Arm 1（主推）：C query 桥臂**
- 结构：reasoning 解码完成后在 KV cache 上追加 K 个可学习 query（因果掩码,增量前向）;last hidden → ≤4M connector（Enc-Proj 序）→ 20 槽 logits + 连续量回归;监督用 GT 码（数据事实待澄清,见 §5;若部分伪标签须标注噪声来源）;Matryoshka 截断训练扫 K;输出码走共享注入模块（亦可另测 hidden 直连注入,作为「绕过 20 维显式瓶颈」的加档）。VLM 全程冻结。
- 训练量：3–5 万样本、单卡数小时。
- 判据：槽准确率 vs A1 解析准确率 vs P0a probe（三点定位信息损失在哪一层）;端到端 R_gen(C)、捕获率、Δ_const/Δ_shuffle。**证伪数字**：R_gen(C) ≤ R_gen(A1) + 0.003 soft-IoU（配对,p<0.05 不显著）⇒ query 桥不优于符号解析,按奥卡姆保 A1;槽准确率 < 解析准确率 ⇒ 约束 2 豁免论证在本设定失败,记录为「低数据量收敛不可达」。
- 该臂同时是约束 2 的直接实证：四条件（20 维码/数万样本/冻结 VLM/query 读出）首个同时满足的数据点。

**Arm 2：B1 attention 读出臂（P0b 过线才全训）**
- 结构：几何短语 token → image token 的 attention 图（eager 导出,层×头筛选,pad 格显式排除并单列报告,可选 test-time register）堆多通道 → 轻 CNN（<4M）→ 作为额外空间条件通道拼进现有 conv 头。免解析、免 query、不经 20 维瓶颈——与 Arm 0/1 是**不同信息通路**（空间投影 vs 符号码）,故不强制共享注入模块,但判据表同构。
- 训练量：3–5 万样本、单卡数小时（attention 导出可离线缓存）。
- 判据：R_gen(B1)、捕获率、Δ_shuffle（打乱短语-图对应）;与 Arm 1 可加性测试（B1+C 双通道 vs 各自单通道）。**证伪数字**：P0b 线性可分性不显著高于 shuffle ⇒ 不全训;全训后 R_gen(B1) < +0.003 ⇒ 判「几何信息只在 token 语义不在 attention 空间分布」,该结论本身回填 READ 在冻结模型上的适用性。

**可选组件（非独立臂）：D 对齐最小可行版**
- 触发条件：Arm 0–2 中任一晋级臂的失败案例归因显示「码对了但形状展开不行」（粗几何是瓶颈）。
- 结构：晋级臂的码 + 图像融合特征（EVF-SAM 警告:不可纯码直投）→ 2 层 MLP → 单个 256 维 sparse token（zero-init）→ **冻结** SAM2 decoder 出粗几何 mask;自训 2.7–4M 旁挂头（MAM M2M 量级）读粗 mask + 自身特征出软场残差;现有 dense 头保留兜底通路（禁纯级联）。
- 预注册消融：{自训头 only} vs {自训头 + decoder 粗几何},配对 Δ ≥ +0.003 soft-IoU 才保留该组件。**证伪数字**：Δ 不显著或边界 F1 反降（objectness 锐边先验伤软场）⇒ 移除,并记录「冻结 decoder 粗几何 + 软场旁挂」配方在本域不成立。
- 明示：该组件无新信息,增量只能来自形状先验对码的展开（§2.D caveat）。

### 4.4 判据与纪律（全臂统一,预注册）

- **指标三列缺一不可**：soft-IoU / hard-IoU（匹配 GT 面积 top-k 阈值化,禁逐场调阈值）、grid 级边界 F1（禁像素级 3px）、中心先验基线列（配对 Δ 与 p 值）。**AUC 全面禁用**（项目红线）。
- 每个消融行必带 **Δ_const / Δ_shuffle** 列;指令条件性用配对差分 + 三条负控制,不用任何 AUC 变体。
- checkpoint 选择按预注册主指标在 S-val,禁 val loss;数据一律 S/P split 旁表,探针类走 S-val 源;`eval100-annotqa-20260727` 永不进训练。
- attention 导出必须 eager;可视化禁逐图 min-max、pad 格显式画白、叠图走 `grid_to_img` 严格逆映射。
- 交付按 REPORT.md 强制三行 + viz/success_* + viz/failure_*（必须有失败案例）。

### 4.5 排期建议（预算内）

P0a/b/c（1–2 天,单卡）→ 共享注入模块三粗档选型 + U_GT 上界（go/no-go,1–2 天）→ Arm 0 与 Arm 1 并行（各数小时训练 + 评测）→ Arm 2（视 P0b）→ 复盘裁定后视归因挂 D 组件。全程 VLM 零改动;B2（SFT 档）不入本轮,记为远期升级路径,若入须单列成本（8 卡×天级、百万样本、reasoning 退化监控为一等指标）。

---

## 5. 缺口与未决问题（如实列）

1. **约束 2 无直接先例**：「20 维离散码 + 数万样本 + 冻结 VLM + query 读出」四条件同时满足的已发表工作为**零**;现有豁免论证是 DeCo/MQT/MetaQueries/Petrov 的三角推断。由 P0a + Arm 1 自证,失败即记录。
2. **Petrov 定理外推未证**：原定理是 prefix 影响下游计算,「尾部追加读出」超出设定;且存在性 ≠ 优化可达。
3. **hidden probing / representation reading 文献线整体缺失**（Patchscopes、线性探针、"模型知道的比说出来的多"系）——82% 可能不是 hidden 层的真上限,这是唯一可能免费抬高上游上限的冻结路线;P0a 是其代偿,但缺文献校准预期。
4. **82% 语义准确率的提升手段几乎空白**：self-consistency / best-of-N 槽位投票 / 生成后自校验仅 SAM4MLLM inquiry 一条（且需 SFT）。A 方向自己的结论就是「瓶颈只剩 82%」,却无攻它的文献。
5. **低维形状码→dense 场的解码器谱系缺失**（DeepSDF/占据场类条件化、参数化掩膜头、槽位→解析先验场可微渲染）——「码→软场」最直接的结构先验,组装节只有 SAM hypernetwork 一条近亲。
6. **D 组件配方无先例**：「冻结 decoder 供粗几何 + 自训头出软场」是从 MAM 反推的假设;MAM 的窄带 alpha 与本项目大面积渐变场分布不同,referring/in-context matting 补线缺失。
7. **冻结 Mask2Former decoder + 轻对齐无先例**（任务卡点名);该分支目前以 SAM2 系为唯一候选。
8. **连续量槽的编码方式无文献支撑**（bucket 化 token vs 回归头);本文裁定走回归旁路属保守默认,非证据裁定。
9. **2026 窗口零覆盖**：六路最新条目止于 2025-08（Bifrost-1);MetaQueries 谱系一年间后续与 SAM2 软场应用最可能有增量,建议补一轮定向检索。
10. **AR VLM attention sink 文献缺失**（StreamingLLM/Massive Activations 系);register 证据只在 ViT encoder 侧,test-time register 在 Qwen3-VL 上须先复现「少数神经元负责 sink」。
11. **数据事实待主 agent 澄清**：GT 几何码是否全量可从 SFT 数据导出（则 Arm 1 直接用 GT 监督,「解析器产伪标签」说法作废）;若只覆盖部分,缺 noisy-label 处理线。
12. **口径不一致待统一**：Sa2VA decoder 冻结/可调（论文内两说,引用须注明）;F-LMM 会议年份两节不一致（2024 挂网 / CVPR 2025,采信 CVPR 2025 需最终核对）;PixelLM「轻 decoder」参数量未核实。
13. **MetaCurves 需用户最终确认**（见 §2.F）。
14. **已裁定的内部矛盾（记录裁定）**:MM1 vs assembly 三方消融之争 → 折中为三类粗档各一,不同类内细扫;teacher-forced ≠ 可实现增益 → 以 §4.1 配对上下界量化。

---

## 6. VERIFIED 参考文献表

### A 符号解析式
| # | 工作 | 来源 |
|---|---|---|
| 1 | JSONSchemaBench (Guidance+EPFL/MSR) | https://arxiv.org/abs/2501.10868 |
| 2 | XGrammar (MLSys 2025) | https://arxiv.org/abs/2411.15100 |
| 3 | OpenAI Structured Outputs（生产数字） | https://openai.com/index/introducing-structured-outputs-in-the-api/ |
| 4 | StructuredRAG (Weaviate) | https://arxiv.org/abs/2408.11061 |
| 5 | Let Me Speak Freely? (EMNLP 2024 industry) | https://arxiv.org/abs/2408.02442 |
| 6 | Say What You Mean（dottxt 复现反驳） | https://blog.dottxt.co/say-what-you-mean.html |
| 7 | Grammar-Aligned Decoding (NeurIPS 2024) | https://arxiv.org/abs/2405.21047 |
| 8 | Know Your Limits: Abstention Survey (TACL 2024) | https://arxiv.org/abs/2407.18418 |
| 9 | Does Prompt Formatting Have Any Impact? (Microsoft) | https://arxiv.org/abs/2411.10541 |
| 10 | BAML Schema-Aligned Parsing (BoundaryML) | https://www.boundaryml.com/blog/schema-aligned-parsing |
| 11 | NuExtract (NuMind) | https://web.archive.org/web/2024/https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction |

### B readout / seg-token
| # | 工作 | 来源 |
|---|---|---|
| 12 | LISA (CVPR 2024) | https://arxiv.org/abs/2308.00692 |
| 13 | PixelLM (CVPR 2024) | https://arxiv.org/abs/2312.02228 |
| 14 | GLaMM (CVPR 2024) | https://arxiv.org/abs/2311.03356 |
| 15 | PSALM (ECCV 2024) | https://arxiv.org/abs/2403.14598 |
| 16 | OMG-LLaVA (NeurIPS 2024) | https://arxiv.org/abs/2406.19389 |
| 17 | Sa2VA (2025) | https://arxiv.org/abs/2501.04001 |
| 18 | READ: How <SEG> Token Works (CVPR 2025) | https://arxiv.org/abs/2412.17741 |
| 19 | F-LMM: Grounding Frozen LMMs | https://arxiv.org/abs/2406.05821 |
| 20 | When Do Prompting/Prefix-Tuning Work? (ICLR 2024, Petrov) | https://arxiv.org/abs/2310.19698 |
| 21 | Universality and Limitations of Prompt Tuning (NeurIPS 2023, 补充边界) | https://arxiv.org/abs/2305.18787 |
| 22 | Vision Transformers Need Registers (ICLR 2024) | https://arxiv.org/abs/2309.16588 |
| 23 | ViTs Don't Need Trained Registers (2025) | https://arxiv.org/abs/2506.08010 |
| 24 | Readout Guidance (CVPR 2024) | https://arxiv.org/abs/2312.02150 |

### C query 桥接
| # | 工作 | 来源 |
|---|---|---|
| 25 | **MetaQueries** (Meta+NYU, 2025) | https://arxiv.org/abs/2504.06256 |
| 26 | OpenUni（开源复现） | https://arxiv.org/abs/2505.23661 |
| 27 | MQT-LLaVA (NeurIPS 2024) | https://arxiv.org/abs/2405.19315 |
| 28 | DeCo（double abstraction 诊断） | https://arxiv.org/abs/2405.20985 |
| 29 | Bifrost-1 (2025-08) | https://arxiv.org/abs/2508.05954 |
| 30 | Honeybee (CVPR 2024) | https://arxiv.org/abs/2312.06742 |

### D 预训练 decoder 对齐
| # | 工作 | 来源 |
|---|---|---|
| 31 | EVF-SAM（冻结消融 82.9/83.3/83.7 vs 21.2） | https://arxiv.org/abs/2406.20076 |
| 32 | SAMWISE (CVPR 2025) | https://arxiv.org/abs/2411.17646 |
| 33 | SAM4MLLM (ECCV 2024) | https://arxiv.org/abs/2409.10542 |
| 34 | Matting Anything (MAM) | https://arxiv.org/abs/2306.05399 |
| 35 | Grounded SAM | https://arxiv.org/abs/2401.14159 |

### E 组装
| # | 工作 | 来源 |
|---|---|---|
| 36 | Segment Anything（三轨接口,源码级核实） | https://arxiv.org/abs/2304.02643 |
| 37 | SurgicalSAM (AAAI 2024) | https://arxiv.org/abs/2308.08746 |
| 38 | SEEM (NeurIPS 2023) | https://arxiv.org/abs/2304.06718 |
| 39 | AnyControl (ECCV 2024;检索引擎初给编造号,已纠正) | https://arxiv.org/abs/2406.18958 |
| 40 | Uni-ControlNet (NeurIPS 2023) | https://arxiv.org/abs/2305.16322 |
| 41 | Compose and Conquer (ICLR 2024) | https://arxiv.org/abs/2401.09048 |
| 42 | Composer | https://arxiv.org/abs/2302.09778 |
| 43 | SPADE (CVPR 2019,背景证据) | https://arxiv.org/abs/1903.07291 |
| 44 | Text4Seg (ICLR 2025) | https://arxiv.org/abs/2410.09855 |
| 45 | MM1 (Apple) | https://arxiv.org/abs/2403.09611 |

> 备注：检索引擎在调研过程中多次编造（假 arXiv 号、零来源条目）,上表全部条目经原始来源打开核实;F-LMM/LISA/PSALM/Sa2VA/EVF-SAM/SAM4MLLM 在多节复用,表中只列一次。
