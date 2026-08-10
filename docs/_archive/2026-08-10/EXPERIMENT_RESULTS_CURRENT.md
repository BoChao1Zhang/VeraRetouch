# 当前实验结果：什么成立、什么不成立、下一步借哪篇论文

> 这是当前结果的叙事视图；逐实验状态与证据路径见
> [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)。所有“完成”均指当前工作区已有可查产物，
> 不把排期表当结果。

## 1. 先给结论

### 已经站得住的好结论

1. **局部渲染器值得做，而且 oracle s 下容量很强。** G2 在真实局部数据上测到约 8.04 dB 的
   s 条件收益，全局对照约 0.04 dB；RD-STD 在 L1–L4 上相对同 N 3D 高出 20.6–23.5 dB。
2. **单标量 s 不只是 alpha 混合。** E2 的薄环单调读出 0.342、带通 0.975；最新补件证明
   受约束 M=12 的渲染器 s 轴本身也能合成该平顶，旧证据链缺口已经闭合。
3. **VLM/视觉表征里确实有 where，但免费读法选错了。** RO-3 的 text→image 单头/融合达到
   AUC_target 0.830，PR13 的双线性交互达到 0.917，shuffle 后回到约 0.54。
4. **低层 what 大多已经在线性可解码。** 曝光、白平衡、色调等 A1–A5 的 R² 为 0.89–0.93；
   信息不是到最后三枚 latent 才突然消失。
5. **参数生成端需要比小 MLP 更强的结构。** RD-G Stage-1 当前机器结果明显支持 transformer；
   标准 33³ LUT 烘焙回读误差近零，轻量、可导出的交付形态保住了。

### 已被证伪或必须降级的坏结论

1. **“直接拿 `<retouch_light>` attention 当 mask”失败。** G1/RO-9 全层 AUC_target 约 0.5；
   修聚合只能救主体定位，救不回指令依赖。
2. **普通主体 AUC 会骗人。** 一条对所有图相同的 `the main subject` 就有 AUC 0.907，
   但 AUC_target 只有 0.523。以后不能用“主体找得准”替代“按指令改指区域”。
3. **“颜色信息被 3-latent 读出口压掉”不成立。** A1–A5 在最终读出口仍保留峰值的 93–99%，
   且像素统计在 6 个属性中的 5 个更强。颜色通路不是当前最大瓶颈。
4. **“what 与 where 分处不同层”不成立。** PR13 两条逐层曲线都很平，峰值层差 8 主要是 argmax
   噪声；可写的是二者在多层同时存在、读出机制不同。
5. **N*=32、低秩 r≈384 泛化、12× 容量效率都已撤回。** 当前是 N*=48；r≈384 只适用于
   记忆口径，泛化需 r≥1536；高斯的价值在局部结构，不是一个错误的倍数。
6. **“朴素 4D 一定会塌”尚未证实。** G3 看到逃逸方向性签名，但没有在有效信号档复现塌陷；
   R-2/R-3 目前只能作为低成本结构保险，不能写成已由实验必然证明。

### 仍然未知，不能提前写进论文的部分

- **端到端链路是否成立**：RO-W 与 RD-G Stage-2 尚未给出“真实指令 → where → renderer”的终态结果。
- **多区域/重叠编辑**：单轴在 L6 明显不足；双轴 RD-F 尚无正式结果。
- **未见指令、未见 LUT、真实用户偏好**：E19/E20/E26 尚未完成。
- **transformer 最终上限**：RD-G 报告仍混有不同 step 的旧口径，需用 `metrics_converged.json`
  重写报告并完成有 s 的 Stage-2 后再定稿。

## 2. 按模块解释：Render 完成了什么

| 问题 | 已做实验 | 当前答案 | 论文里可写到哪 |
|---|---|---|---|
| 3D 颜色算子为何不够？ | G2 | 同 RGB 跨区域需要不同输出；真实档约 +8.04 dB，全局对照约 0.04 dB | 可写“数据层存在性证据”，不能写真实 readout 已解决 |
| s 轴是否有表达力？ | E2、RD-ORACLE | 基底能画常见 what/where；受约束带通闭环；L1–L4 容量大幅过门 | 可写 oracle/representation capacity |
| 单轴的上限？ | G2 L6、RD L6 | 重叠软掩膜只 +4.97 dB，低于 +8 门 | 可作为双轴/多步机制的必要性证据 |
| 能否退化为全局？ | RD L0 | RD-STD 收敛后仍比 3D 低 0.93 dB；RD-C/RD-E 能过 | 不能说所有 4D 参数化“零代价退化” |
| 条件通道会不会塌？ | G3 | 有效信号档未塌；无信号档出现 σs↑/sensitivity↓ | 只能写规模相关风险与保险设计 |
| 生成器是否太小？ | RD-G Stage-1 | transformer 明显优于小/宽 MLP，烘焙几乎零代价 | 可写阶段性结构证据；待同步收敛报告与 Stage-2 |

Render 线当前最好的方法判断是：**保留轻量、可烘焙的像素算子，把容量投到“条件生成/读出端”；
但必须同时修 L0 全局退化和 L6 多区域表达。**

## 3. 按 what / where 解释：VLM 完成了什么

### 3.1 What：要改什么、改多少

| 观察 | 证据 | 结论 |
|---|---|---|
| A1–A5 颜色/曝光量线性可解码，R² 0.89–0.93 | PR13 逐层 probe + label permutation / random backbone | 表征里有 what |
| 最终 3-latent 仍保留 93–99% 峰值 | PR13 visual tower → connector → LLM → latent 链 | “最终读出口把颜色压没”被否决 |
| 像素统计 5/6 更强 | PR13 C5 | 这批低层 what 不是 VLM 的独特增量 |
| A6 色恒常在视觉塔 0.44，过 connector 后 0.018 | PR13 | 需要语义的颜色量有真实 connector 断点；这是更窄、更可信的 what 研究点 |
| 模型自由回答接近常数预测器 | PR13 behavior | 说明模型没被训练回答数值问题，不能单独归因“读出口坏” |

**当前取舍**：不要另建一条昂贵的通用颜色 readout 线。把 what 叙事收缩为两类：

- 已有全局 latent/像素统计能解决的低层量，直接复用；
- A6 这类需要语义/色恒常的量，专门研究 connector 多层特征，而不是泛称“颜色都丢了”。

### 3.2 Where：改哪里

| 读法 | 主体定位 | 指令可控性 | 当前判断 |
|---|---:|---:|---|
| GL token head-mean attention（G1/RO-9） | 中等 | AUC_target ≈0.5 | 淘汰为 where 读出；可留作负结果 |
| 修层/head 聚合（RO9b） | 0.92–0.935 | 最高约 0.572 | 只救定位，不救指令依赖 |
| CLIP self-self（RO-1） | 0.930 | 固定 deictic 已 0.907；target 0.523 | 强主体先验/伪标签种子，不是最终控制器 |
| logit lens（RO-2） | `emb` 目标词 0.829 | 绝对刻度失败；LLM 层抹除词特异性 | 适合诊断，不适合直接接 renderer |
| 单头/融合 text→image attention（RO-3） | `.cgt` 0.840 | OOF AUC_target 0.830 | 当前最好零训练/轻训练候选 |
| image×instruction 双线性 probe（PR13） | 0.917 | shuffle 后约 0.54 | 最直接支持“训练一个小 where 头” |
| VLM→14 维系数（RO-W） | 尚无终态 | 尚无终态 | 当前端到端关键缺口 |

**核心机制解释**：image token 在默认 decoder-only prompt 中位于指令之前，它自身不可能随指令改变；
可行信号必须来自“见过指令的 text token 作为 query”与 image token 的交互。这个解释同时统一了
G1 的失败、RO-3 的成功和 PR13 的双线性结果。

## 4. docs 里的 paper list 有没有解法

有，但不是一篇论文包办全部链路。最可信的组合是“**训练式 where 读出 + 语义第 4 轴 renderer +
区域保真评测**”，而不是继续寻找免费 attention 捷径。

| 当前问题 | paper list 中最有用的方案 | 可迁移机制 | 最小验证实验 | 风险 / 不应夸大的地方 |
|---|---|---|---|---|
| 免费 attention 不听指令 | **SWIM**、**Preserving Localized Patch Semantics in VLMs**、**When Sinks Help or Hurt**（见 [`SURVEY`](SURVEY_papers_2026-07-31.md) 的 VLM readout 小节） | 用 referring-expression/mask supervision 约束 text→patch；按层调制而不是平均所有 head | 在 RO-3 最佳头上加小型 mask loss；对照 fixed deictic、无关词、shuffle、灰图 | sink 不是本项目主因；不能只做去 sink，不改 query |
| probe 高但可能是先验 | **Decodable Is Not Grounded** | blank/gray、token permutation、cross-image replacement 等 causal arbiter | PR13/RO-3 增加 blank-image 与多区域同图对照 | probe/steering 高不等于 grounded；现有主体/背景二分仍太容易 |
| 需要训练式、稠密的 connector | **Dense Connector**、**DeCo**、**Perceiver IO / DETR query decoder** | 多层 dense patch 特征 + 少量 query cross-attention，避免先池化再语义化 | N 个 primitive query 读取 image tokens + instruction token，直接回归 E2 的 14 维 w 或 GLUT 参数 | query 可能冗余；当前 RD-G 没看到原语塌陷，暂不先加复杂匹配损失 |
| s 作为第 4 轴 | [**4D-LUT (P0_004)**](plan/paper_digests/P0_004_digest.md)、[**SA-LUT (P0_003)**](plan/paper_digests/P0_003_digest.md) | RGBC 四线性插值/四维寻址；context map 是 address，不是 alpha | 同一 GT 和同一 s，比较 RD-STD、4D-LUT、SA-LUT context baseline | 两篇多为隐式 context、k=1；不直接证明语言条件 where |
| 4D 高斯不是唯一 renderer | [**Deep Bilateral Learning / HDRNet (P1_033)**](plan/paper_digests/P1_031_digest.md) | 低分辨率 bilateral grid + full-res guidance slicing | 用同一 VLM s 作 guidance，比较容量、边界、速度、L0 退化 | grid 分辨率可能成为新瓶颈；正适合作为 RD 的 plan B |
| 局部训练样本与指标 | [**RC-GRPO (P1_040)**](plan/paper_digests/P1_040_digest.md)、[**Qwen-Edit+ (P1_038)**](plan/paper_digests/P1_037_digest.md)、[**AceTone (P0_002)**](plan/paper_digests/P0_002_digest.md) | inside reward、outside preservation、coverage 调权；LUT 合成 → SFT → preference | 已知 mask 的局部曝光/WB/色调合成；报 inside ΔE、outside ΔE、boundary PSNR、AUC_target | 合成分布窄；reward 可迁移，扩散模型本体不一定可迁移 |
| 多区域/单轴不足 | SA-LUT 多 context bins、分步 Exposure/白盒编辑、双轴 RD-F | 每步一个 s 或 k 个条件轴，明确组合次序 | 专打 L6：单轴、双轴、两步串联同预算比较 | 多轴会损失紧凑性；必须先定导出契约 |

### 最推荐的论文组合

1. **读出**：采用 RO-3/PR13 已验证的 instruction-side bilinear 或 query decoder；借 SWIM 的 mask 对齐，
   借 Decodable Is Not Grounded 的 causal controls。
2. **渲染**：主线保留当前 4D GLUT；把 4D-LUT/SA-LUT 设为结构 baseline，HDRNet 设为 plan B。
3. **训练**：先用可解析的 paired synthetic/local reconstruction，后加真实数据与 outside-preservation；
   不要一开始用偏好/RL 掩盖基本容量问题。
4. **评测**：每个结果同时报 inside、outside、boundary、全图和指令负控制；不再用单列主体 AUC。

## 5. 下一轮实验的优先顺序

1. **先收口现有证据**：用 RD-G `metrics_converged.json` 重写 REPORT，完成并审阅 Stage-2；修 RO-3
   陈旧 `STATUS.md`。这些是文档一致性问题，不需要重跑已有计算。
2. **完成 RO-W 的真实 D-SFT-L 主档**：这是现有“表征可读”与“renderer 有容量”之间唯一缺失的桥。
3. **做一图多区域的 where 数据**：至少主体、天空、地面、背景四类，避免模型只学会主体/补集二分。
4. **跑最小端到端四臂**：oracle s / RO-3 双线性 s / RO-1 主体先验 / random-shuffle s，同一 renderer、
   同一预算；主指标是 inside/outside/boundary，而不是普通 AUC。
5. **专打两个结构缺口**：L0 全局退化（RD-C/RD-E 为什么过、RD-STD 为什么不过）与 L6 双轴/串联。

做到第 4 步之前，论文最稳的故事是“**发现并诊断了 where 读出错位，分别证明了可读表征和可用
renderer**”，不是“完整端到端系统已经赢了”。
