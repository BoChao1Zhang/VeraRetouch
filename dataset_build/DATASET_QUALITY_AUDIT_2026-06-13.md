# VeraRetouch 数据集质量测评报告 (S1–S8, 2026-06-13)

> 测评对象：Direction-A 1M 构建（datagen v2），分支 `datagen-v2-databuild`。
> 数据快照：`/home/bc/data/datasets/vera_directionA_1M/shards/<Sx>/*.jsonl`，统计产物 `/home/bc/data/datasets/vera_directionA_1M/_eval_2026-06-13/stats.json`。
> **本次构建仍在追加中（in-progress build）**：本报告快照时全库 `OVERALL.count = 770,765`（目标 1M），下列计数与比率均为快照值，会随构建增长，但比率与代码根因结论不随计数漂移。
> 数据集用途：训练「修图推理 VLM」的 SFT/GRPO 语料。每条样本教模型：给定 输入图 + 编辑指令，产出 `<think>` 推理链 + `<answer>`（Lightroom 风格的 VeraRetouch 参数字典 = GT 修正量），region-local 流另带 `C_GT` mask。

---

## 0. 执行摘要

### 0.1 语料规模（快照）

| Stream | Builder | n | 占比 | answer_null | instr_empty | think 模板率 |
|---|---|---:|---:|---:|---:|---:|
| S1 | InverseDegradeStream (region) | 310,535 | 40.3% | 0% | 0% | **100%** |
| S2 | RecipeXSourceStream (region, SAM3) | 154,379 | 20.0% | 19.2% | 2.6% | 0% |
| S3 | MMArtTextStream (region) | 11,823 | 1.5% | 0% | 0% | 0% |
| S4 | PPR10KStream (region, real mask) | 26,529 | 3.4% | 0% | **31.1%** | 0% |
| S5 | Tier1ExpertStream GREYSKY (global) | 20,000 | 2.6% | 0% | 10.3% | 0% |
| S6 | RecipeXSourceStream (global) | 182,499 | 23.7% | 23.7% | **18.6%** | 0% |
| S7 | InverseDegradeStream (global) | 55,000 | 7.1% | 0% | 0% | **100%** |
| S8 | FivekGoldStream (global) | 10,000 | 1.3% | **59.5%** | 0% | 0% |
| **OVERALL** | | **770,765** | 100% | **10.2%** | **6.3%** | **47.4%** |

数据来源：`stats.json`（各 stream 节点）。两个合成模板流（S1+S7）合计 **365,535 行 = 47.4% 语料**，其 `think` 字段是**逐字节相同的同一条 boilerplate**，这也是 `OVERALL.think_boilerplate_rate = 0.474` 的来源。

### 0.2 整体结论

1. **LABEL（answer + mask）整体可信**：合成流 S1/S7 的 GT 是「退化的精确逆」，aspect↔param 家族 0 错配（S1 0/310,535，S7 0/35）、几乎无 clip（S1 全库仅约 10 行触界）；真实专家流 S4/S5（GREYSKY/PPR10K）与 S3（MMArt）的 answer 是真实人手/专家 grade。**核心资产是参数标签本身。**

2. **`think`（推理链）是全库最严重、最系统化的缺陷**，且**直接是 SFT/GRPO 的监督生成目标**：
   - **47.4% 语料（S1+S7）的 think 是单一 boilerplate**，泄露 pipeline 内部词（"synthetic inverse-degradation sample" / "teacher-rendered" / "search the best renderer correction"），不含任何 slider/方向/幅度，0 推理。
   - **所有 VLM 流（S2/S4/S5/S6，约 38.3 万行 / 49.7% 语料）的 think 在结构上对 stored answer 不忠实**：根因 `config.yaml:275 vlm_override_answer=false` + `streams.py:1110-1112`，think 推理朝向的是 VLM 自己那个被丢弃的 answer，而 stored answer 是确定性 recipe/专家 GT。两者方向冲突在 S2 达 33%（15/46），S6 param-kind 达 ~34%（10/29）。

3. **verify 质量门形同虚设**：`enable_verify=true` 但 `verify_gate_mode="log"` + `verify_sample_rate=0.1` + `verify_degrade_streams=false`（`config.yaml:284-291`）。后果：`aesthetic / mllm_score / look_match / param_sane / er_recon_psnr / histsim` 在绝大多数记录上为 null，且 **QAGate 从不因质量拒绝任何样本**（`streams.py:359-401` 所有输入 None 即跳过）。所有缺陷全部出库。

4. **空指令与 null answer 成片出库**：S4 `instruction_empty=31.1%`（8,243 行）、S6 `18.6%`（33,944 行）是按体量计最大的 INSTRUCT 黑洞；S8 `answer_null=59.5%`（5,946 行）、S6 `23.7%`、S2 `19.2%` 是按体量计最大的 LABEL 黑洞。

### 0.3 按严重度排序的 Top 问题

| # | 问题 | 维度 | 受影响 stream | 量化普遍性 | 训练影响 | 状态 |
|---|---|---|---|---|---|---|
| 1 | think↔answer 结构性不忠实（override=false 解耦） | REASONING | S2,S4,S5,S6（+S8 部分） | ~100% VLM 流记录；方向翻转 S2 33%、S6 param 34% | 教模型推理朝向「不能输出」的 answer，GRPO reward hacking 风险高 | **confirmed** (critical) |
| 2 | think 是单一 boilerplate、泄露 pipeline、零推理 | REASONING | S1,S7 | 100%（365,535 行，1 条唯一 think） | 占 47.4% 语料把模型 think 拉向定式泄露 stub | **confirmed** (critical) |
| 3 | answer=null 占多数（think 截断丢失目标参数） | AESTHETIC | S8 | 59.5%（5,946/10,000） | 该流既不教真实专家 look 也不教一致 answer | **confirmed** (critical) |
| 4 | stored answer 是 VLM 猜测而非 fivek 专家 GT | AESTHETIC | S8 | 40.5%（非 null 全部 VLM 自产） | 与设计意图冲突；但真实专家 look 以 pixel(REAL_JPG) 存在（见 §2） | **confirmed**，harm 框架 **partial** |
| 5 | 空指令逐字出库且仍生成 think | INSTRUCT | S4,S6（+S2,S5） | S4 31.1%、S6 18.6%、S5 10.3%、S2 2.6% | 无 grounding 的 think + 空请求，训练噪声 | **confirmed** (critical/high) |
| 6 | 严重指令模板化 + "in this any photo" 语法错误 | INSTRUCT | S1,S7 | S1 14 长串/全库，88.2% 含病句；S7 14 串，91.1% 病句 | 教模型固化 7~14 句词汇、零图像 grounding | **confirmed** (high) |
| 7 | 白平衡意图不可学：Temp/Tint=0 占 100% GT | AESTHETIC | S4（机理同 S5） | S4 0/26,529 非零 WB，40/60 指令要求 warm/cool | "调暖/调冷"永远学不到 | **confirmed** (high) |
| 8 | think 截断、信封从不闭合 | REASONING | S6（+S8） | S6 n=3000 中 0% 含 `</think>`、~50.5% 无终止符 | 截断 scratchpad 被当 CoT 存；S8 因此 null | **confirmed** (high)；S6 answer 丢失 **refuted** |
| 9 | instruction_short 全缺失 | INSTRUCT | S3 | 100%（11,823/11,823） | 缺 terse-imperative 信号 | **confirmed** (medium) |
| 10 | 指令第三人称分析腔（非祈使） | INSTRUCT | S2,S6 | S2 53%（32/60）、S6 ~50% | 学到「描述图片」而非「下指令」 | **confirmed** (medium) |

---

## 1. INSTRUCT 维度

### 1.1 [HIGH, confirmed] 严重指令模板化（S1: 14 长串 / 7 短串覆盖 310k 行）

- **现象**：S1 全部指令落入 `f"Restore the {aspect_text} of {scope} in this {scene} photo."` 的有限取值。
- **量化**：`stats.json` S1 `instruction_distinct=14`，`instruction_distinct_ratio=4.5e-05`，`instruction_exact_dup_rate=0.99995`；top 长串 `"Restore the specific colors of the masked region in this any photo."` 重复 54,292 次。S7 同构：`instruction_distinct=14`，`distinct_ratio=2.5e-04`。
- **受影响 stream**：S1（310,535）、S7（55,000）。
- **代码根因**：`streams.py:1873` f-string；`aspect_text` 来自 3 标签集 `_aspect_label`（`streams.py:1855-1859`，{L:lighting, GC:global color, SC:specific colors}→7 非空组合），`scope` 固定，无 subject/scene/palette 注入。
- **修复建议**：(a) 把 S1/S7 当作确定性自监督 label，**从 instruction-following SFT 中降权/隔离**；或 (b) build 时从 ≥30 同义祈使句库随机抽取，并注入 `tag_cache` 已有的 subject/scene 标签提升唯一性与 grounding。
- **状态**：**confirmed**（code + impact 双 lens 一致）。impact lens 提示：作为「仅 instruct」缺陷，因 answer 仍可学，单独严重度为中；但叠加 §3 think 缺陷后，S1/S7 文本两维整体不可用于 SFT。

### 1.2 [HIGH, confirmed] 未加守卫的 `scene='any'` 造成病句 "in this any photo"

- **现象**：scene 缺失回退 `"any"`，直接代入 `{scene}` 槽，产出 "...in this any photo."。
- **量化**：S1 全库 `instruction_collisions.in_this_any_photo = 273,899/310,535 = 88.2%`；S7 50,080/55,000 = 91.1%（证据包数据）。
- **代码根因**：`streams.py:1872` `scene = sample.scene_meta.scene or scene_of(source)`，`scene_of`（`streams.py:254-255`）返回 `(item.scene or "any").lower()`；TAD66K 等源无 scene 标签，无语法守卫地代入 `streams.py:1873`。
- **修复建议**：一行守卫——`scene in {None,'any',''}` 时输出 "in this photo."（去掉 scene 词）；或从 `tag_cache` 回填真实 scene。
- **状态**：**confirmed**（code 确认；impact lens 标 partial 仅因病句严重度判断保守，根因与普遍性无争议）。

### 1.3 [HIGH/CRITICAL, confirmed] 空指令逐字出库，且仍在空串上生成 think

- **现象**：VLM 返回空 instruction 被原样持久化，且仍调用 `reason_params('')` 产出无 grounding 的 think。
- **量化**：S4 `instruction_empty_rate=0.3107`（8,243 行，**按体量计最大的单一 INSTRUCT 黑洞**）；S6 `0.1860`（33,944 行）；S5 `0.1029`（2,058 行）；S2 `0.0264`（4,068 行）。short 1:1 跟随。
- **代码根因**：`vlm_clean.py:696` `gen_instruction` 返回 `str(d.get('instruction_long','')).strip()`，空串**不抛异常、不拒绝**；`streams.py:1095-1103` 仅在**抛异常**时回退 `_offline_annotate`，空串被接受存储；`streams.py:1109` 随后用空串调 `reason_params`。`QAGate.accept`（`streams.py:359-401`）无空指令/最短长度门。
- **修复建议**：`gen_instruction` 把空/纯空白 `instruction_long` 视为失败（抛异常或返回哨兵）；`_annotate` 在 VLM 调用后若 instruction 为空则回退 `_offline_annotate`；对非模板流在 QAGate 加空指令拒绝门；空指令时不调 `reason_params`。
- **状态**：**confirmed**（S4/S6 code+impact 双确认；根因链精确到行）。

### 1.4 [MEDIUM, confirmed] 指令为第三人称分析腔而非祈使

- **现象**：指令以 "The image appears/shows..." 描述分析开头，或夹带 "the request should focus on..." / "I need to..." 分析腔。
- **量化**：S2 32/60（53%）描述腔开头、17/60（28%）正文夹分析腔；S6 非空指令约 50% 第三人称 scene 分析（persona slip）。
- **代码根因**：`gen_instruction` prompt 明确要求 imperative user-request voice，但**无后置校验强制**（`vlm_clean.py:~657-700`）。
- **修复建议**：在 `gen_instruction` 加轻量正则后校验，拒绝以 "The image/photo/scene" 开头或含 "the request should"/"I would request" 的输出并重 prompt/改写；或加一个 imperative one-shot 示例对并降温。
- **状态**：**confirmed**（secondary）。

### 1.5 [MEDIUM, confirmed] S3 `instruction_short` 全缺失

- **现象**：S3 终端短指令字段恒空。
- **量化**：`stats.json` S3 `instruction_short_empty_rate = 1.0`（11,823/11,823）。
- **代码根因**：`MMArtTextStream`（`streams.py:1637`）走 MMArt 源文本路径，未填 short 字段。
- **修复建议**：从 MMArt 长文本派生一个 ≤12 词祈使短句，或调 `gen_instruction` 仅取 short。
- **状态**：**confirmed**（与 S3 摘要一致）。

### 1.6 [MEDIUM/LOW] S1 区域 grounding 与 short 退化（次要）

- **S1 'masked region' 无概念名词，且部分"区域"覆盖全帧**（cross 维度，medium）：全库 coverage 中位 0.105，但 49,749/310,535（16.0%）coverage>0.5、367 行 >0.9，70~90% mask 实为全局，与"region-local"矛盾。建议：>0.6 高覆盖样本封顶/记录；若有廉价 concept 则用名词（"the subject"/"the sky"）替代 "the masked region"。
- **S1 `instruction_short` 是 long 的逐字切片**（low）：35/35 short == `'restore '+aspect_text`，全库 7 唯一 short。结构上存在且一致（相对其他流是正面），低优先；若要更自然短祈使，复用 §1.1 的同义句库。

---

## 2. 美学 (AESTHETIC) 维度

### 2.1 [CRITICAL, confirmed] S8 多数 answer=null（目标参数被 think 截断吞没）

- **现象**：S8 多数记录 `answer=null`；其余携带的是 VLM 自产参数（见 §2.2），均非 fivek 专家 GT。
- **量化**：`stats.json` S8 `answer_null_rate=0.5946`（5,946/10,000）。证据包全扫：所有 5,946 null 均缺 `</answer>` 标签，约 25%（1,493）以字面开 JSON 尾结束（如 `...{ "Highlights2012": -90,`）；null-think 均长 2,726 字符 vs 非 null 1,934，证实长推理→更易截断。
- **代码根因**：`config.yaml:83 max_tokens=768`（`_chat_full` at `vlm_clean.py:615`）对 `reason_params` 的冗长 `<think>` 太小，预算在 `</think><answer>{...}</answer>` 前耗尽；`split_think_answer`（`vlm_clean.py:280-292`）找不到 `<answer>` 即把整段当 think，`loads_lenient` 在截断串上失败→`params=None`→answer null。
- **修复建议**：`reason_params` 专用 `max_tokens` 提到 1536–2048；或令 prompt **先输出 `<answer>` JSON**；在 `split_think_answer` 加 salvage（从 think 尾抽最后一个 `{...}`）。更优解见 §2.2。
- **状态**：**confirmed**（code+impact 双确认，截断机制复现）。

### 2.2 [CRITICAL, confirmed；harm 框架 partial] S8 stored answer 是 VLM 猜测而非 fivek 专家 GT

- **现象**：S8 设计本应 `answer=None`（gold real-JPG 行无 teacher params），但 `_annotate` 用 VLM 幻觉参数填充。
- **量化**：S8 非 null 的 4,054 行（40.5%）全部 VLM 自产。
- **代码根因（设计相撞）**：`FivekGoldStream.build_one` 硬置 `answer=None`（`streams.py:1605`，注释 "no teacher params for gold real-JPG rows"），`recipe.params=_empty_params()`，真实专家 look 仅存于 `recipe.meta['expert_after_jpg']`（`streams.py:1616-1619`），`after_source=REAL_JPG`。但 `_annotate`（`streams.py:1111-1112`）的 `sample.answer is None` 短路触发，把 VLM 提案当 GT 存入。
- **修复建议**：在 `FivekGoldStream` 显式禁止 `_annotate` 覆盖故意为 null 的 gold answer（传 per-stream `allow_vlm_answer=False` 标志取代 `answer is None` 判断）；并文档化 answer=null 对 real-JPG 流是正确的，确保 `expert_after_jpg` 接入训练。
- **状态**：**confirmed**（机制 + 40.5% 普遍性属实）。**但 harm 框架 partial**：impact lens 证实**真实专家 GT 确以 pixel 形式存在**——`reproduce.py:155-167` 让每条 S8 行走 REAL_JPG 契约（`target_rgb` = 真实 `expert_after_jpg` 的 `processed.jpg`），在 PARAM 分支（`reproduce.py:169`）前 return。即「专家 look 从未存储」的说法仅对「以 params 形式」成立，对「以像素 target 形式」不成立。报告据此**不把"专家 GT 彻底丢失"作为事实陈述**。

### 2.3 [HIGH, confirmed] 白平衡意图不可学：S4 GT 的 Temperature/Tint 恒为 0

- **现象**：40/60 指令、57/60 think 明确要求 warm/cool，但 GT 的 `IncrementalTemperature/IncrementalTint` 永远 0。
- **量化**：S4 全库 0/26,529 非零 `IncrementalTemperature` 或 `IncrementalTint`。
- **代码根因（比原 finding 更精确）**：PPR10K 专家 XMP 把 WB 编码为**绝对 Kelvin**（实测 `crs:Temperature` 范围 2400–6250、均值 ~4894、std 571；`crs:Tint` 绝对值 90% 非零）；`recipes.py:167-188 _crs_attrs_to_params` 只保留 38 个 `PARAM_KEYS`（含 `Incremental*` 相对键），绝对 Kelvin 被丢弃；`vlm_clean.py` 让模型输出 Temperature 但该 answer 被丢弃（override=false）。
- **修复建议**：决定 PPR10K 相对 WB 是否进 GT——若专家 XMP 编码绝对 Kelvin，转换为 `IncrementalTemperature` 使 "warm it up" 可学；否则**从 S4 指令与 think 中剥离 temperature/tint 措辞**，使文本不承诺 GT 无法兑现的编辑。
- **状态**：**confirmed**（code+impact 双确认，根因下沉到绝对 Kelvin 解析丢失）。**机理同 S5**（S5 36/60 要求 warm 但 `IncrementalTemperature/Tint=0`）。

### 2.4 [HIGH/MEDIUM] param 极值/clipping 与 recipe 自相矛盾

- **现象**：recipe-param GT 频繁触界或自相矛盾。
- **量化**：S2 8/33 param GT 有 ±100 clip、18/33（55%）携带 ≥8 个指令未要的密集 HSL band、个别记录 tone curve 自相矛盾（Shadows+100 & Blacks+100 & Highlights-100 同存）或极端 Exposure（+1.55 stops）；S6 param-kind ~28% 至少一个 slider 触 ±100 clip（多在 HSL）。**对照 S1/S7**：合成流 GT 几乎不触界（S1 全库 `rail_counts` 仅个位数：`hsl_colortemp_abs>=100`=6、`Highlights2012<=-100`=2、`light_slider_abs>=100`=4、`Shadows2012>=99`=1；S7 0/35 clip），证明 clip 集中在 recipe 源而非合成源。
- **代码根因**：S2/S6 用 `RecipeXSourceStream`（`streams.py:1344-1360`）的预设 recipe params，复杂预设本身带极值与密集 HSL；无 param 极值过滤门（verify log-mode，`param_sane` 全 null）。
- **修复建议**：(a) 对 recipe param 加 build 时 `heuristic_param_sane`（`vlm_clean.py:396`）盖章并对全帧密集 HSL 触界做软门；(b) 接受 recipe 为「真实 look」但在 think 侧与指令对齐措辞。
- **状态**：**confirmed**（与 S2/S6 摘要一致）。

### 2.5 [MEDIUM/LOW] aesthetic 分覆盖率与 verify 缺位

- **现象**：`aesthetic / param_sane / mllm_score / er_recon_psnr / histsim / look_match` 在合成流与多数 VLM 流上为 null。
- **量化**：S1/S7 这些字段 100% null（`stats.json` 无填充）；VLM 流（S2–S6/S8）摘要显示 verify 仅 0.1 采样且 log mode，多数记录 null。**例外正面信号**：S2/S3/S4/S5/S6/S8 的 `aesthetic` 分在采样中存在（S2 mean 5.42、S3 5.67、S4 5.60、S5 present、S6 部分），说明 aesthetic 评分管线本身可用，只是覆盖率低。
- **代码根因**：`config.yaml:284-291` `verify_gate_mode="log"` + `verify_sample_rate=0.1` + `verify_degrade_streams=false`；`QAGate`（`streams.py:359`）输入全 None 即跳过每个门。
- **修复建议**：对合成流至少 build 时跑 `heuristic_param_sane` 盖章（近免费，answer 本地已知）以暴露触界行；对 VLM 流把 `verify_sample_rate` 提向 1.0 做质量优先 pass，并先 log 分布再切 enforce。
- **状态**：**confirmed**（secondary，S1 medium）。

### 2.6 [LOW] S3 答案 HSL 子字典复用

- 22/60 S3 answer 含完全相同的 9 键 HSL 簇，26/60 共享 ≥7 键。非 correctness bug，但压低答案多样性，并解释了 `SatYellow=-23` 与 "make yellows pop" 矛盾。建议对近似 answer 子字典下采样/去重或降权 S3 color-mixer 键。状态：**confirmed**（low）。

> **AESTHETIC 维度正面校准**：S1/S7 的合成 answer 是退化精确逆、幅度得体（mean |v| ~12–20、p95 ~30–49、`answer_allzero_rate≈3e-06`、`answer_full_38key=0`），param label 可信可学。S3/S4/S5 的真实人手/专家 grade 在审美上是得体的。AESTHETIC 的问题集中在 (a) S8 的 null/VLM 替换、(b) recipe 流极值、(c) WB 不可学、(d) 评分覆盖率，而非「修图品味」本身崩坏。

---

## 3. REASONING 维度

> 关键前提（已用 code 确认）：`think` 是 dataset 明示的监督生成目标（`pack.py:158` 逐字透传），不是元数据。因此 think 的任何不忠实/泄露都会直接进入 SFT loss 和 GRPO reward 信号。

### 3.1 [CRITICAL, confirmed] think↔answer 结构性不忠实（vlm_override_answer=false 解耦）

这是全库 **最高影响** 的 REASONING 缺陷，跨所有 VLM 流同一根因。

- **现象**：`sample.think` 始终存 VLM 朝向**自己那个被丢弃的 answer** 的推理；`sample.answer` 却是确定性 recipe/专家 GT。二者常方向冲突。
- **量化**：
  - **S2**：15/46 非 null answer（33%）有显式 slider 方向翻转；另 14/60 answer=null、7/60 全零 answer 而 think 提大幅具体编辑。
  - **S4**：22/60 有 Exposure 符号或幅度断裂；32/60 think 提出 answer 永不含的具体 Temperature；机制覆盖 ~100% S4 行。
  - **S5**：60/60（结构性）think 朝向 VLM 自产参数；stored answer 是无关 GREYSKY 预设。
  - **S6**：param-kind 29/29 采样（52.8% stream ≈ 96,328 行），其中 24/29 think 从不点名 answer 实际设置的任何 HSL 通道，10/29 直接与 answer 符号冲突。n=3000 实测 param 符号一致仅 57.5%（vs lut 91.6%），缺陷隔离在 param-kind。
- **代码根因**：`streams.py:1110` 无条件 `sample.think = rp.get("think")`；`streams.py:1111-1112` 仅当 `vlm_override_answer or sample.answer is None` 才采用 VLM answer；`config.yaml:275 vlm_override_answer=false`。S2/S4/S5/S6 在 build_one 预置非 null 确定性 GT（如 PPR10K `streams.py:2012 answer=params`、S5 `streams.py:1222 answer=params`），故 VLM answer 被丢弃、think 留下。`reason_params` prompt（`vlm_clean.py:719-720`）甚至写明 answer "does NOT override any ground-truth recipe"，坐实解耦。
- **修复建议（首选 faithful-CoT 重生）**：在 GT answer 固定**之后**，把 GT 参数喂进 `reason_params`，让模型论证**那些确切 slider/方向/幅度**；或对 VLM 流设 `vlm_override_answer=true` 保持 (think,answer) 配对；或对 recipe-GT 记录干脆**丢弃 think**、从 GT 合成模板化忠实 think。最低限度：在 QAGate 加 think 命名 slider 与 stored answer 的符号/存在一致性检查并 reject/repair。
- **状态**：**confirmed**（S2/S4/S5/S6 均 code+impact 双确认）。

### 3.2 [CRITICAL, confirmed] S1/S7 think 是单一 boilerplate、零推理、泄露 pipeline

- **现象**：全部 365,535 行（S1+S7）共享同一条 think，含 pipeline 内部词，未点名任何 slider/方向/幅度。
- **量化**：S1+S7 各 `think_boilerplate_rate=1.0`；全扫各 1 条唯一 think，且两流 think **逐字节相同**（合计 365,535 行 = 47.4% 语料）。`stats.json` S1 `think_len` min=median=max=135。
- **代码根因**：`streams.py:1875-1878` `_offline_degrade_annotate` 硬编码 think 字符串，从不检视 `sample.answer`；`build_one` 直接调它（S7 `streams.py:1861-1882`、S1 `streams.py:1931`），绕过 VLM。
- **修复建议**：从已知 answer 字典确定性合成忠实 think——枚举所选 slider、符号→动词、`|value|`→幅度桶，按 aspect 分组（如 "Lift shadows (+41) and reduce contrast (-32) to recover crushed lighting; cool the global tint (-19); boost saturation (+37)."）。剥离所有 teacher/renderer/synthetic 词。**廉价（无需 VLM），因 answer 已知。**
- **状态**：**confirmed**（code+impact 双确认；明确缺陷跨 S1+S7 而非 S7-only）。

### 3.3 [CRITICAL/HIGH, confirmed] think 截断、信封从不闭合

- **现象**：冗长 CoT 在 `</think>`/`<answer>` 前被 token cap 砍断，截断 scratchpad 被当 think 存。
- **量化**：S6 30/60 采样 think 以 mid-token 结束；n=3000 子样：**0% 含 `</think>`、0.1% 含 `<answer>`、~50.5% 无终止标点**；think 词数封顶 ~607（p90=515），与 768 token 上限一致。
- **代码根因**：`config.yaml:83 max_tokens=768`，被 `vlm_clean.py:615 _chat_full` 用于 `reason_params`（无独立 `reason_params_max_tokens`）；`split_think_answer`（`vlm_clean.py:280-287`）回退「整段=think」。
- **修复建议**：`reason_params` 专用 `max_tokens` 提到 2048–3072，并加「信封完成」检查（缺 `</think>` 或不可解析 `<answer>` 即 reject/重生）；收紧 prompt 限 CoT 长度；结合 §3.1 在 answer 之后生成 think 使长度可预测。
- **状态**：**confirmed**（截断机制复现）。**注意 partial**：impact lens 证实在 S6（override=false）**answer 并未因截断丢失**（answer 来自 recipe 而非 think 尾），故「答案被截断吞没」对 S6 是 **refuted**；但该机制对 **S8 成立**（§2.1，S8 走 `answer is None` 短路，答案真来自 think 尾）。

### 3.4 [HIGH, confirmed] think 泄露 pipeline/prompt 内部

- **现象**：think 复述系统 prompt 元指令。
- **量化**：S2 44/60（73%）think 含 `reason_params` prompt 字面（"Edit request:"、"translate this request"、"Global adjustments only"、"standard English Lightroom identifiers"）；S5 51/60 泄露 meta；S6 ~83% 泄露；S8 65.3%（6,526/10,000）引用 "the prompt"/"Global adjustments only"/"`<answer>` JSON"/"MaskGroupBasedCorrections"。S1/S7 100% 泄露（§3.2）。
- **代码根因**：`reason_params`（`vlm_clean.py:702-720`）让模型对着自己的系统 prompt 出声推理；persona/system prompt（`vlm_clean.py:90-100`）。
- **修复建议**：在 prompt 中明令不得复述指令/键白名单；后置正则剥离泄露句；或随 §3.1 faithful-CoT 重生时只暴露 GT 参数、不暴露元指令。
- **状态**：**confirmed**（跨多流一致）。

### 3.5 [HIGH, confirmed] think 引用 answer schema 缺失的 slider 名

- **现象**：think 用 Lightroom-UI 名（Temperature、BlueSaturation、GreenHue、OrangeLuminance）推理，而 schema 是 `IncrementalTemperature`/`SaturationAdjustmentBlue`/`HueAdjustmentGreen`/`LuminanceAdjustmentOrange`。
- **量化**：S5 60/60；S2 24/60 提 Clarity/Texture/Dehaze 等 schema 无键；S3 60/60 承诺 sharpening 步（schema 无 sharpness 键），31/60 提 warming WB（answer 0/11,823 含任何 Temperature/Tint），7/60 提 tone curve、7/60 提降噪。
- **代码根因**：`reason_params` prompt（`vlm_clean.py:717-719,730`）字面列出 `Temperature, Tint, ... BlueSaturation, OrangeLuminance, GreenHue` 等非 schema 名。
- **修复建议**：把 `reason_params` 键词汇与 stored answer schema 对齐（用 `IncrementalTemperature/Tint`、`SaturationAdjustment<Band>` 等），或后置把 think 名映射到 schema 名。
- **状态**：**disputed → partial**。两 lens 均确认根因（prompt 列错键），但 **CRITICAL 框架被驳**：(a) `parse_answer`（`vlm_clean.py:312-346`）已通过 `_KEY_ALIASES`（`vlm_clean.py:106-125`）把 UI 名 alias 到 canonical 键，**模型最终要学的 answer 输出不受影响**；(b) S8 摘要确认 `BlueSaturation→SaturationAdjustmentBlue` 这类"键错配"是 alias 重命名、非真 drift。报告将其降为 **HIGH-表面/实际中度**：think prose 里的 UI 名会污染 CoT 自然度，但不构成 answer 不一致。

### 3.6 [MEDIUM, confirmed] think 提符号与 answer 相反 / 引用 schema 外控件（次要）

- S3：8/60 brighten 措辞配 `Exposure2012<0`；2/60 要 WARMER 但 think 冷化图像。S2：24/60 提 Clarity/Texture/Dehaze 等 schema 外键。S3：7/60 tone-curve（全库仅 63/11,823 answer 含任何 Parametric 键）、7/60 降噪（无键）。建议：从数字 answer 重生 think，或加 Exposure 符号一致性检查；把 reason_params 约束到 answer schema 键白名单。状态：**confirmed**（secondary，medium）。

> **REASONING 维度正面信号**：S8 在存活处（25/60 非 null）think 是连贯、image-grounded、忠实 justify answer 的真 CoT（正确 slider/方向/幅度，键错配只是 alias）。S3 的 think 是真 image-grounded 散文（源自人手 MMArt）。说明 VLM 推理能力本身在线，缺陷源自 **管线解耦（override=false）+ 截断 + prompt 泄露**，而非模型不会推理。

---

## 4. 跨 stream 严重度矩阵

单元格式：`严重度 / 普遍性`。`—` 表示该流不受此问题影响或不适用。严重度：C=critical, H=high, M=medium, L=low。

| 问题 \ Stream | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 |
|---|---|---|---|---|---|---|---|---|
| **R1** think↔answer 不忠实（override=false） | — | C/33%翻转 | — | C/~100% | C/100%结构 | C/param 34%翻转 | — | 部分/40% |
| **R2** think 单一 boilerplate+泄露+零推理 | C/100% | — | — | — | — | — | C/100% | — |
| **R3** think 截断不闭合 | — | M/57%截断 | — | — | H/29% | H/~50% | — | C/驱动 null |
| **R4** think 泄露 pipeline/prompt | H/100% | H/73% | — | M/空行 | H/85% | H/83% | H/100% | H/65% |
| **R5** think 引用 schema 缺失键 | — | M/40% | H/100% | H/53% | H/100% | M | — | L/alias |
| **A1** answer=null（目标丢失） | — | H/19.2% | — | — | — | H/23.7% | — | C/59.5% |
| **A2** stored answer 是 VLM 猜测非专家 GT | — | — | — | — | — | — | — | C/40.5% |
| **A3** WB(Temp/Tint)=0 不可学 | — | — | — | H/100% | H/100% | — | — | — |
| **A4** param 极值/clip/自相矛盾 | L/~10行 | H/55% HSL | L | — | M/激进 | M/28% clip | L/0 | — |
| **A5** aesthetic/质量分 null | M/100% | M/采样 | L | L | M | M | M/100% | M/100% |
| **I1** 指令模板化（极低唯一） | H/100% | — | — | — | — | — | H/100% | — |
| **I2** "in this any photo" 病句 | H/88.2% | — | — | — | — | scene='any'~75% | H/91.1% | scene='any'100% |
| **I3** 空指令逐字出库 | — | M/2.6% | — | H/31.1% | M/10.3% | C/18.6% | — | — |
| **I4** 第三人称分析腔非祈使 | — | M/53% | — | — | — | M/~50% | — | M/~37% |
| **I5** instruction_short 缺失 | L/切片 | — | M/100% | — | — | — | — | — |
| **X1** region-local 但 mask 近全局 | M/16% cov>0.5 | — | M/24/60<0.15 | — | — | — | — | — |

---

## 5. Pipeline 根因总结

绝大多数缺陷可归并到 **4 个 pipeline 设计/配置根因**：

### RC-1 `vlm_override_answer=false` 把 think 与 answer 解耦（最高影响，覆盖 ~49.7% 语料）
- **配置/代码**：`config.yaml:275 vlm_override_answer: false`；`streams.py:1110`（无条件存 VLM think）+ `streams.py:1111-1112`（仅 `vlm_override_answer or answer is None` 才采用 VLM answer）。
- **症状**：R1（S2/S4/S5/S6 全部）、R5（think 名与 answer 名不一致的次因之一）、S8 的 answer 被 VLM 猜测填充（A2，`answer is None` 短路的另一面）。
- **本质**：设计本意是「VLM 解释确定性 GT」，但实现是「VLM 解释自己的另一个答案」。**修这一个根因即可拆掉大半 REASONING 缺陷。**

### RC-2 S1/S7 合成模板（覆盖 47.4% 语料的文本两维）
- **代码**：`streams.py:1873`（指令 f-string）、`streams.py:1875-1878`（boilerplate think）、`_aspect_label streams.py:1855-1859`、`scene_of streams.py:254-255`（无语法守卫的 "any"）。
- **症状**：I1、I2、R2、R4（S1/S7 部分）。
- **本质**：自监督 label 优秀，但文本是占位符。answer 已知 → think/指令均可**确定性、廉价地从 answer 重生**，无需 VLM。

### RC-3 verify 跑在 log-mode + 0.1 采样 + 跳过 degrade 流（质量门形同虚设）
- **配置**：`config.yaml:284 verify_gate_mode="log"`、`config.yaml:291 verify_sample_rate=0.1`、`config.yaml:283 verify_degrade_streams=false`；`QAGate streams.py:359-401` 输入全 None 即跳过。
- **症状**：A5（aesthetic/param_sane/mllm/psnr/histsim/look_match 多数 null），且**所有 §1–§3 缺陷全部不被拒绝**（QAGate 从不因质量 reject）。空指令（I3）、null answer（A1）、param 极值（A4）、think 不忠实（R1）全部出库。
- **本质**：没有任何运行中的硬质量门；缺陷靠后处理或重建消除，而非 build 时拦截。

### RC-4 `gen_instruction`/`reason_params` 的空串与截断容忍 + prompt 泄露
- **代码/配置**：`vlm_clean.py:696`（空串不抛异常不拒绝）、`streams.py:1095-1103`（仅异常才回退）、`config.yaml:83 max_tokens=768`（`vlm_clean.py:615 _chat_full`，无 `reason_params` 专用上限）、`split_think_answer vlm_clean.py:280-292`（截断回退整段=think）、`reason_params` prompt（`vlm_clean.py:702-720`）暴露元指令。
- **症状**：I3（空指令出库）、R3（截断不闭合）、A1 的 S8 部分、R4（泄露）。
- **本质**：VLM IO 边界缺少「非空 + 信封闭合 + 不泄露」三道校验。

---

## 6. 优先级修复路线图

> 原则：先修覆盖语料最大、训练危害最直接、改动最局部的根因。所有改动点标 `file:line`/config key。

### P0（必须，阻断 SFT/GRPO 训练正确性）

| ID | 改动点 | 具体改动 | 预期收益 | 风险 |
|---|---|---|---|---|
| P0-1 | RC-1：`config.yaml:275` + `streams.py:1110-1112` | **faithful-CoT 重生**：GT answer 固定后，把 GT 参数喂回 `reason_params` 生成 think；或对 recipe-GT 流丢弃 think。**不可只翻 `vlm_override_answer=true`**（会丢确定性专家 GT）。 | 拆掉 ~49.7% 语料的 think↔answer 不忠实（R1） | 需一次额外 VLM pass；须保证新 think 不再泄露元指令（配 P0-4） |
| P0-2 | RC-2：`streams.py:1875-1878` | 从 `spec.op_params` **确定性合成忠实 think**（枚举 slider/符号/幅度桶），剥离 teacher/renderer/synthetic 词。无需 VLM。 | 修 365,535 行（47.4%）R2/R4；think 变可学 | 极低（纯确定性，answer 已知） |
| P0-3 | A1/A2：`streams.py:1605` + `streams.py:1111-1112` | S8 传 per-stream `allow_vlm_answer=False`，禁止覆盖故意 null 的 gold answer；文档化 answer=null 对 REAL_JPG 正确，确认 `expert_after_jpg`→`reproduce.py:155-167` 训练接入。 | 修 S8 100% answer 语义（5,946 null + 4,054 VLM 猜测） | 须确认训练侧消费 REAL_JPG pixel target（impact lens 已证存在） |
| P0-4 | RC-4：`vlm_clean.py:696` + `streams.py:1095-1103` | 空 `instruction_long` 视为失败 → 回退 `_offline_annotate`；空串不调 `reason_params`；QAGate 加非模板流空指令拒绝门。 | 修 S4 31.1% + S6 18.6% + S5 10.3% + S2 2.6% 空指令（I3） | 低；回退路径已存在 |

### P1（强烈建议，显著提质）

| ID | 改动点 | 具体改动 | 预期收益 | 风险 |
|---|---|---|---|---|
| P1-1 | RC-4：`config.yaml:83` + `vlm_clean.py:615` + `split_think_answer` | 加 `reason_params_max_tokens=2048`；加信封完成检查（缺 `</think>`/`<answer>` 即重生）；`split_think_answer` 加 salvage 抽尾部 `{...}`。 | 修 R3 截断（S6 ~50%、S5 29%）+ S8 截断 null | 推理成本上升；需限 CoT 长度 |
| P1-2 | I1/I2：`streams.py:1873` + `scene_of streams.py:255` | scene∈{None,'any',''} 输出 "in this photo."；从 ≥30 同义祈使句库抽 S1/S7 指令并注入 `tag_cache` subject/scene。 | 修 273,899(S1)+50,080(S7) 病句 + 模板化 | 低；或直接将 S1/S7 从 instruct-following SFT 降权/隔离 |
| P1-3 | A3：`recipes.py:167-188` + S4/S5 指令生成 | PPR10K 绝对 Kelvin → `IncrementalTemperature` 转换使 WB 可学；否则从 S4/S5 指令/think 剥离 warm/cool 措辞。 | 修 S4/S5 的 WB 不可学（40/60 指令落空） | 需校准 Kelvin→Incremental 映射 |
| P1-4 | I4：`vlm_clean.py:~657-700` gen_instruction | 后置正则拒绝 "The image/photo/scene" 开头与分析腔；加 imperative one-shot + 降温。 | 修 S2 53% + S6 ~50% 第三人称腔 | 低 |
| P1-5 | R4/R5：`reason_params` prompt `vlm_clean.py:717-720` | 键白名单对齐 schema（`IncrementalTemperature`、`SaturationAdjustment<Band>`…）；prompt 明令不复述元指令；后置剥离泄露句。 | 修 R4 泄露（多流 65–100%）+ R5 键名 | 低（配 P0-1 重生时一并做） |

### P2（建议，收尾与监控）

| ID | 改动点 | 具体改动 | 预期收益 |
|---|---|---|---|
| P2-1 | RC-3：`config.yaml:283-291` + QAGate | 质量优先 pass：`verify_sample_rate`→1.0、先 log 分布、再切 `verify_gate_mode="enforce"`；合成流 build 时跑 `heuristic_param_sane`（`vlm_clean.py:396`）盖章。 | 暴露并拦截 param 极值/不一致；填 A5 质量分 |
| P2-2 | A4：S2/S6 recipe param | 对全帧密集 HSL 触界做软门/去重；接受 recipe look 但对齐 think 措辞。 | 降 S2 55% HSL / S6 28% clip 噪声 |
| P2-3 | I5：S3 `streams.py:1637` | 从 MMArt 长文本派生 ≤12 词祈使 short。 | 修 S3 100% short 缺失 |
| P2-4 | X1：S1 region-local | coverage>0.6 的 region-local 封顶/记录；有 concept 时用名词替 "the masked region"。 | 修 16% 近全局 region 误标 |
| P2-5 | A6：S3 answer | 近似 HSL 子字典去重/降权。 | 提 S3 答案多样性 |

---

## 7. 附录

### 7.1 关键统计表（引用 `stats.json`）

来源：`/home/bc/data/datasets/vera_directionA_1M/_eval_2026-06-13/stats.json`（构建快照 2026-06-13）。

**null/empty 率（全库 OVERALL：count=770,765）**

| 字段 | OVERALL | 最严重 stream |
|---|---|---|
| `answer_null_rate` | 0.1022（78,784） | S8 0.5946、S6 0.2366、S2 0.1921 |
| `instruction_empty_rate` | 0.0627（48,313） | S4 0.3107、S6 0.1860、S5 0.1029 |
| `instruction_short_empty_rate` | 0.0780 | S3 1.000、S4 0.3100、S6 0.1856 |
| `think_empty_rate` | 0.0000 | —（think 从不空，但 47.4% 是 boilerplate） |
| `think_boilerplate_rate` | 0.4742（≈365,535） | S1 1.000、S7 1.000 |
| `instruction_distinct_ratio` | 0.4831 | S1 4.5e-05、S7 2.5e-04（极度模板化） |

**S1 param 健康度（核心资产，全库 310,535）**：`answer_null=0`；`answer_allzero_records=1`（rate 3.2e-06）；`answer_full_38key_records=0`；触界 `rail_counts` 合计仅约 13 行（`hsl_colortemp_abs>=100`=6、`Highlights2012<=-100`=2、`light_slider_abs>=100`=4、`Shadows2012>=99`=1）；`Exposure2012` ∈[-2.061, 1.921]（in-stops）；`answer_nonzero_param_count` median=3 / p90=6 / max=9。

**S1 文本退化**：`instruction_distinct=14`、`instruction_len` median=77；`think_len` min=median=max=135（单串）；`in_this_any_photo=273,899`（88.2%）。

### 7.2 方法与样本量

- **量化扫描**：全库统计来自 `stats.json`（per-stream + OVERALL 全量计数，非抽样），覆盖 count/null/empty/distinct/boilerplate/param_minmax/rail_counts。
- **逐记录人工核验**：每 stream 抽 60 条（部分 35 条）做指令/think/answer 三维交叉核验，配 source image grounding 抽查（S1 dog 肖像、S2 cottage/river、S3 boy 肖像、S4 orange-wall + green-sofa、S8 yellow-tulip 等均逐图核对）。
- **代码根因**：每条 finding 经 code lens（精确到 `file:line`）+ impact lens（数据复现）双重对抗验证，verdict 记 confirmed/partial/disputed/refuted。
- **think↔answer 忠实性**：在采样上以「think 命名 slider 的符号/存在 vs stored answer」逐条比对；并在 n=3000/6000 子样做符号一致率统计（S6 param 57.5% vs lut 91.6%）。

### 7.3 不确定性 / 未验证项（诚实标注）

1. **S8 "专家 GT 丢失" 框架为 partial/refuted**：真实专家 look 以 `expert_after_jpg` pixel target 存在（`reproduce.py:155-167`），并非彻底丢失。报告未把"专家 GT 从未存储"当事实，仅确认"以 params 形式未存储 + 40.5% 被 VLM 猜测填充"。**未验证**：训练侧是否真正消费该 REAL_JPG pixel 契约（需查训练数据加载器）。
2. **S5 R5（schema 缺失键）为 disputed→partial**：`parse_answer` 的 `_KEY_ALIASES`（`vlm_clean.py:106-125`）已把 UI 名映射到 canonical 键，故对 answer 输出无害；降为表面/中度，不按 critical 计。
3. **S6 R3「answer 被截断吞没」对 S6 refuted**（answer 来自 recipe 非 think 尾），仅对 S8 成立——R3 的 critical 仅适用 S8。
4. **快照漂移**：构建仍在追加（770,765 → 目标 1M）。比率与代码根因结论稳定，但绝对计数会增长；各 stream 内部比率（如 S8 null 59.5%、S4 空 31.1%）预计随构建保持同量级。
5. **verify 分布缺测**：因 `verify_gate_mode="log"` + 0.1 采样，look_match/er_recon_psnr/histsim 分布在多数记录上不可得，无法据此量化 edit-vs-instruction 的逐像素一致性；当前一致性结论基于 think/指令/answer 文本交叉核验与抽样图像 grounding。
6. **样本量**：逐记录核验每流 35–60 条，方向翻转/泄露率等百分比为采样估计（已标 n/分母）；全库 distinct/null/boilerplate 为全量精确值。
