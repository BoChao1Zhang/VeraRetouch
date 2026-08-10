# NOTES — RO-3 · LMM text→image attention 全层全头扫描

## 〇、落盘核对声明

本报告所有数字来自 `metrics.json`（本目录，`analyze_ro3.py` 一次跑出，seed 20260803）。
下列**与任务卡的出入**已如实登记，不做粉饰：

1. **任务卡写「200 张构造样本：`.../sanity/val/L1`」——实际该目录只有 24 个样本。**
   `T4_construct/REPORT.md` 白纸黑字：train 套 = 8 级 × **200**（S-train），
   val 套 = 8 级 × **24**（S-val）。数据纪律（DATA_ASSIGNMENT / CLAUDE.md）禁止 ad-hoc 切分，
   我**没有**把 train 拿来充数当评测集。处置见 §二.1（保守默认 + 待决策 U1）。
2. **构造集样本自带 GT 掩膜但不带指令**。manifest 只有 `slot_id=semantic-*` 与变换参数，
   没有区域的自然语言描述。处置：用 `provenance.candidate_id` 反查
   `journal-archive/prod-l*/sft.jsonl` 的 `local.subject.description` + `local.region`，
   套 **G1 区域对立批逐字相同的模板**生成 reg_a / reg_b（§二.2）。
   join 命中率：val **22/24**、train **181/200**（缺的是 journal 里 `local` 行不含
   subject.description 或 region 不在 COMPLEMENT 表内的候选）。
3. **G1 已缓存的 s 场不能直接复用**：G1/RO-9 的 `stacks/*.npz` 是 **head-mean**（`stack` 的
   形状是 `(3, 24, 16, 16)`，头维已被平均掉），而 RO-3 的全部意义就在逐头。
   任务卡"复用 G1 已缓存的样本与指令清单，不要重新采样"我按**字面**执行：
   **样本与指令清单 100% 复用冻结的 `config/g1_{samples,region_opp,shuffle_ctrl}.json`，
   零重新采样**；前向重跑（prefill-only，无生成，0.22 s/样本）。
4. **首轮 fp16 落盘已作废并整批重导为 float32**。fp16 下 `post`（softmax 概率）在
   sink 主导头上整片下溢为 0（实测单样本 336 个 (层,头) 里 **2 个全 0、36 个 >10% 零**），
   会伪造并列秩、直接污染本臂的核心对比（pre-softmax 能否绕开 sink）。
   `pre` 侧亦有 4/336 个格的"图内 std / fp16 量化步长"< 10。两处一并按 float32 重导
   （`logs/export_fp32.log`），首轮 fp16 数据已删除，任何数字都不来自它。

## 一、实施前核实记录（CLAUDE.md 派工协议第 1、2 条）

### 1.1 读过的文档节（只读引用节）

| 文档 | 节 | 取到的东西 |
|---|---|---|
| `EXPERIMENTS_v3_2026-08-02.md` | §2.1 RO-3 行 / RO-9 行 / Changelog `2026-08-04` 整条 | RO-3 的晋级判据「融合 AUC≥0.75 且最佳层在中层」、淘汰判据「最佳层在末两层」、FA 互斥红线、RO 臂优先级重排、**表述纪律 D-31** |
| `IMPL_DOSSIER_2026-08-02.md` | §5.3 / §5.4 + 附录 B | eager 互斥的具体表现、视觉塔权重被丢弃、显存 O(L·H·N²) 的分层导出建议 |
| `PLAN_v2_local-retouch_2026-07-31.md` | §2.2 D-0 行（L141） | D-0 三件套 = ① test-time registers ② `norm > median+3·MAD` 剔除插值 ③ 16px 周期功率谱→DVT；**D-0 判据**：outlier 不降且 s 场无差别 ⇒ 此骨干无此病，跳过 |
| `experiments/G1_s_identifiability_20260803/` | REPORT.md 全 + NOTES.md | ρ_region_opp / AUC / valid-mask 口径；ρ_syn 0.867、ρ_region_opp 0.884、ρ_shuf 0.895 的终判数字 |
| `experiments/RO9_layer_verdict_20260804/` | REPORT.md + NOTES.md + `analyze_layers.py` | **AUC_target 的权威定义**（`auc_target_list`：reg_a→M 与 1−AUC(s_b,M) 混池取中位）、0.65/0.30 两档门、以及 §五.2 对 RO-3 的明确交办（head-mean 会把单头信号平均掉，RO-3 应逐头复用同一判据量与阈值） |
| `tools/readout/ro9_gl_attention.py` | 全文 | 直接 import 复用：`load_model`（eager+双重断言）/ `build_inputs` / `luma_to_grid` / `outlier_mask_from_norms`；未重写 |
| `tools/scache/README.md` | 全文 | arm 目录约定、`instr_hash`、`SCache.write` 的必含字段（`extra_meta` 不得覆盖 `layer`/`norm`/`arm`） |

### 1.2 **实测**核实（不是查文档，脚本 `config/verify_env.py` → `config/verify_env.json`）

| # | 待核实 | 实测结果 |
|---|---|---|
| V1 | 模型规模（不信转述，从 config 直读） | **24 层 / 14 heads / 2 KV heads（GQA groups=7）/ hidden 896 / `mobileclip_l_1024` / 全 24 层 `full_attention`**；transformers **4.57.1**、torch 2.10.0+cu128 |
| V2 | eager + `output_attentions=True` | 返回 **24 个** 非 None 张量，shape **(1, 14, 342, 342)** ✅ |
| V3 | **红线**：sdpa + `output_attentions=True` | `fwd.attentions` **is None**，只 warning（`sdpa attention does not support output_attentions=True`），**不回退**。DOSSIER §5.4 在本环境**实测成立** ✅ |
| V4 | pre-softmax 捕获是否等价 | `softmax(捕获 logit)` vs eager 返回的权重，逐元素 **max\|Δ\| = 1.95e-3**（= bf16 舍入量级；返回值本身是 bf16）→ 捕获正确 ✅ |
| V5 | image token 段 | span **[14, 270)**，**256 个**，`√256 = 16` → 16×16 网格 ✅ |
| V6 | 视觉 token 在序列中的定位方式 | 本项目 `<image>` 走 llava 占位 **`IMAGE_TOKEN_INDEX = -200`**（负数、不在词表），由 `prepare_inputs_labels_for_multimodal` 就地展开成 256 个视觉 embedding。**DOSSIER §5.3 的 151652/151653/151655 是 Qwen2.5-VL 的，本项目不适用**——任务卡的提醒成立。special token 实测：`<retouch_light>`=**151646** / `<retouch_color&temp>`=151647 / `<retouch_colormixer>`=151648 |
| V7 | instruction 文本 token 定位 | 朴素"子串 id 匹配"**失败**（hit=−1：slow tokenizer 在上下文里对同一串切分不同）。改用**空指令 prompt 的公共前缀/后缀差分**定位，落在 `ro3_layerhead_scan.locate_pools`，每条样本把定位到的 instruction 解码串写进 npz meta 可核 |
| V8 | 与 RO-9 管线的一致性 | 同一 (img, reg_a) 上：`valid16` 与 G1 重算**逐格一致**（agree=1.0）、`luma16` **max\|Δ\|=0.0**；`pre_gl` 的 head-mean 与 G1 `stack_raw` 逐层 ρ = 0.3–0.9（不等于 1 是**预期**：RO-9 的 GL token 位于生成出来的 plan 文本之后，RO-3 的 gl 桥接位是 prompt 末尾追加位） |

### 1.3 未在线核实的外部事实

本任务**没有引用任何 IMPL_DOSSIER 附录 B 之外的外部 URL / 论文数字 / 第三方 API**：
用到的全部是本仓库代码、本机模型 config/tokenizer、以及 G1/RO-9 两个交付文件夹里的数字，
且逐条以**实测**（§1.2）或**读源码**方式核过。故无网络核实项。
`transformers` 的 FA/SDPA 行为虽然写在 DOSSIER §5.4（权威文档内），仍按任务卡要求**实测**（V3）。

## 二、方法学决策（自行核实后采用，非拍板）

1. **构造集 24 vs 200 的处置**（对应 §〇.1）。保守默认：
   - **评测**只用 **S-val**：G1 区域对立批 214 源（主判据）+ 构造集 val **22 源**（GT 掩膜复核）；
   - **S-train 的构造集 181 源只用于"融合权重跨 split 拟合"**（`fusion_cross_split`），
     其结果单独成行、**不进主判据**，且因为拟合集在 S-train、评测集在 S-val，
     两侧源不相交，不存在泄漏；
   - 没有把 train 当评测集，也没有 ad-hoc 重切分。
2. **构造集指令的生成方式**。用与 G1 区域对立批**逐字相同**的模板与 COMPLEMENT 表
   （`config/build_jobs.py` 从 `G1/config/build_region_opp.py` 抄常量），区域描述来自
   journal 的 `local.subject.description`——即**构造集掩膜与指令指的是同一个语义槽**，
   这正是把"GT 掩膜"当标签的合法性前提。方向词取自构造变换的 `sign`（+→brighten，−→darken）。
3. **query 池不预设，四池全导**（本臂"不预设读哪个 token"的落地）：
   `instr`（instruction 段 text token 均值）、`last`（prompt 末 token）、
   `alltxt`（image 段之后的全部 text token 均值）、`gl`（**桥接项**：prompt 末尾追加
   `<retouch_light>` 的 teacher-forced 位，用来与 RO-9 并排读）。
   pooling = **未加权算术平均，无任何逐图归一化**。
4. **pre / post 两读法的定义**。`pre` = `q·kᵀ/√d + causal_mask` 的 image 列；
   `post` = 在**整行（全 key 集合，含 sink）**上 softmax 之后再切 image 列。
   post 保留了 sink 对分母的贡献——这才是"pre-softmax 能否绕开 sink"的正确对照。
5. **D-0 只落盘掩膜、修复在分析端**，从而 D-0 on/off 可 A/B（PLAN 的 D-0 判据要求"s 场无差别
   ⇒ 跳过"，不 A/B 就无法回答）。实现的是 D-0 的**第二件**（norm>median+3·MAD 剔除+4 邻域插值），
   与 RO-9 逐场版数值等价（同迭代数、同兜底），只是批量化。
   第一件（test-time registers）需要改模型结构、属训练侧改造，零训练扫描臂不做；
   第三件（16px 周期功率谱→DVT）在 16×16 token 网格上不可分辨（16px 周期 < 1 个 token），
   本臂改报 outlier 占比逐层曲线代替。**两项缺席已记在 §三 待决策 U2。**
6. **多重比较控制**（G1 翻车教训的直接对策）。2688 个候选读出取 max 一定虚高，故：
   - 单头判据配**条件角色配对置换的最大统计量零分布**（B=500）→ 家族错误率 p；
   - 融合走 **5 折 source-level CV**，**选头 / 定向 / z-score 统计量全部只在训练折上估**，
     报 out-of-fold 数字；"融合前"对照 = 折内选出的单个最优头在留出折上的 AUC_target。
   预注册规则写死在 `analyze_ro3.PREREG`，跑数前提交。
7. **凸组合的两个必要配件**（都在 fit 折上确定，不看留出折）：
   - **定向**：`sign = +1 if AUC_target_fit ≥ 0.5 else −1`（凸组合的非负权重无法翻符号）；
   - **逐头全局 z-score**：均值/方差在 fit 折的**全部图、全部格**上估，是**数据集级**统计量，
     **不是逐图统计量**——红线"s 禁逐图 min-max/softmax 归一化"未触碰。
8. **"中层"的定义不自创**：取 `ro9_gl_attention.CANON_LAYERS = range(8,16)`，即 G1/RO-9 的
   canonical 段 L8–15。末两层 = L22–23（0-indexed，共 24 层）。写在 `PREREG.bands`。

## 三、待主 agent 决策

> 按协议，下列属"两种做法都合理、影响后续"的决策项；已采用**保守默认**继续，未静默拍板。

- **U1（构造集规模）**：任务卡要 200 张构造样本，S-val 只有 24（join 后 22）。
  保守默认 = 只用 22 张做 GT 掩膜复核、不进主判据；主判据仍挂 G1 区域对立批 214 源。
  若要把构造集升为主判据源，需要先跑 `tools/construct/generate --split val --per-level 200`
  重生成 S-val 套（约 215 s × 8 级，属数据侧动作，我没有擅自改动冻结数据集）。
- **U2（D-0 三件套只落地了第二件）**：第一件 test-time registers 要改模型结构，
  第三件 16px 周期谱在 16×16 token 网格上不可分辨。若审阅认为必须齐三件，
  第三件应移到**视觉塔 patch 网格（1024²/patch 后）**上做，属另一条工具线。
- **U3（`gl` 桥接位的语义）**：RO-9 读的是**生成出来**的 GL token 位置，RO-3 为省算力
  用的是 prompt 末尾**追加**位（prefill-only）。两者在 local 段的 s 场逐层 ρ = 0.3–0.9。
  保守默认 = `gl` 池只当**与 RO-9 并排读的桥接项**，不作主判据；若要严格复刻 RO-9 的位置，
  需带生成跑一遍（成本 ×50，约 10 h）。
- **U4（scache 融合臂的权重取哪一折）**：评测口径是 out-of-fold（5 组权重），
  但落盘的 `ro3-fused` arm 必须有**一份确定的**权重。保守默认 = 取 fold 0 的权重并在
  meta 里写明 `fold_of_weights: 0` 与 z-score 统计量口径。若下游渲染臂要用，
  建议改成"全区域批重拟合一版"并单独记账。

## 四、红线自查（对照 CLAUDE.md 速查表）

| 红线 | 本实验状态 |
|---|---|
| attention 导出必须 eager（FA2/SDPA 返回 None 不回退） | ✅ 加载即断言 `_attn_implementation == "eager"`（`ro9_gl_attention.load_model`），每样本断言 `fwd.attentions[0] is not None`；**并在 `verify_env.py` 实测了 sdpa 确实返回 None** |
| s 禁逐图 min-max/softmax 归一化 | ✅ 落盘为原始 logit / 原始概率；判据量 AUC 与 Pearson 对逐场仿射不变；融合用的是 fit 折上估的**全局**逐头 z-score（数据集级，非逐图） |
| VLM 干预对象 = 整段 image tokens 非 last token | ✅ 本臂只读不干预；读出对象是 query→**整段 256 个 image token** 的注意力；`last` 只是 **query 端**的一个池，且不是唯一池（四池全导） |
| 每个消融行必带 Δ_const/Δ_shuffle 列 | ✅ 扫描表逐行带 `dauc_shuffle`（= AUC(s_a,M) − AUC(s_shufa,M)，s_shufa 用**别的源**的 reg_a，方向词已匹配）与 `rho_shuf`；常量基线 = AUC_target 的 0.5 与 luma16 基线 |
| IoU 禁当优化目标 | ✅ 未出现 IoU |
| checkpoint 选择禁用 val loss | n/a（零训练；融合权重的选择用 out-of-fold AUC，不是 loss） |
| 全局仿射 G 初始化 / σ 参数化 / s 轴平滑正则 / 逐像素算子 | n/a（本臂不训练渲染器） |

## 五、复现

```bash
# 1) 环境实测核实（必跑，产出 config/verify_env.json）
CUDA_VISIBLE_DEVICES=1 .venv-lens/bin/python \
  experiments/RO3_layerhead_scan_20260803/config/verify_env.py

# 2) 作业清单（零重新采样，全部读 G1 冻结 config）
.venv-lens/bin/python experiments/RO3_layerhead_scan_20260803/config/build_jobs.py

# 3) 导出（GPU，卡 1，prefill-only，1708 次前向）
CUDA_VISIBLE_DEVICES=1 .venv-lens/bin/python tools/readout/ro3_layerhead_scan.py \
  --jobs-json experiments/RO3_layerhead_scan_20260803/config/ro3_jobs.json \
  --out-dir /var/cache/veradata/ro3_stacks_20260803 --skip-existing

# 4) 分析 + 判决 + viz + scache（纯 CPU）
.venv-lens/bin/python experiments/RO3_layerhead_scan_20260803/analyze_ro3.py
```

---

# 追加记录（主 agent 2026-08-03 两条追加任务）

## 六、追加①：逐头差分场 + 同区域对照 + max-stat 置换零分布

- 脚本：`analyze_diffield.py`（预注册规则 `PREREG_DIFF` 写在跑数之前）+ `viz_diffield_cases.py`
- **零额外 GPU**：全部读已落盘的 `/var/cache/veradata/ro3_stacks_20260803`。
- 判据量 `AUC_diff = AUC(s(reg_a) − s(reg_b), M)`；零模型 = **逐源符号翻转**（`d ↔ −d`，
  等价于交换 a/b 角色；在「s 不依赖指令」的原假设下二者可交换）× 2000 次，
  每次取 **2688 路的 max** 构成零分布 → 单步 maxT 家族错误率。
- 同区域对照 `d_ctrl = s(syn_a) − s(syn_b)`（同源，n=212）。**这条缺席则任何正向读数不成立**，已跑。
- 结果：L11H5 `pre/instr` = **0.9298**，对照 **0.5220**，配对 Wilcoxon **p=5.1e-35**，FWER p = **0.000**。
- **对主 agent head-mean 否定结论的解释**：head-mean 通道 0.5129 vs 逐头 0.9298 →
  **「被平均掉」拿到正面证据**。同时提醒：同区域对照的 **max**（跨 2688 路）也能到 0.808，
  所以判读**必须**用同一读出的配对量，不能拿两个 max 相比。

## 七、追加②：AUC 三列口径

- 脚本：`analyze_gt3col.py`（含 `CgtMaskBank`）。
- **`.cgt.png` 取用路径**：`g1_region_opp.json` 里**逐源冻结的 `sft_id`** → journal
  `prod-l*/sft.jsonl` 的 `candidate_id`（命中 **214/214**）→ build shard 索引
  `(shard, offset_data, length)` **seek 直读**（与 `tools/scache/oracle.py:read_member` 同路径，无解包，
  shard 命中 **214/214**）。**一源恰一候选**——区域对立批构造时 `build_region_opp.py` 已按
  winner_confidence 排序选定一条 sft 行，不存在多候选歧义。
- **软边阈值**：`.cgt` 是软边单通道。口径 = 原图 → 短边 512 bilinear → expand2square 黑边 pad →
  面积均值下采样 16×16 →（与 SAM3/luma 完全一致）→ **阈值二值化**。
  **敏感性**：0.3 / 0.5 / 0.7 三档下 L11H5 的差分场 AUC = **0.8366 / 0.8405 / 0.8459**，结论不变。
- **`winner_confidence=low` 排除**：主列口径为 `normal`（n=81）；含 low 的并列口径为 113，
  数值 0.8456（更高），即排除 low 是**保守**方向。
- **入选 159/214**：55 源的 `.cgt` 在 16×16 上正类或负类为空（掩膜过小/过大），AUC 无定义，跳过。
- **两套 GT 差异有多大**：`.cgt` 与 SAM3 主体掩膜的 16×16 IoU 中位 **0.459**（q10–q90 0.14–0.97）。
  入选源的 slot_mode：**semantic 50 / radial 48 / band 46 / linear 15** —— 三分之二是几何槽，
  掩膜根本不是主体。所以三列不是同义复述。

## 八、G1 数字的真值源

按主 agent 要求，凡引用 G1 数字一律取 `experiments/G1_s_identifiability_20260803/metrics.json`：
**已核对** `aggregate.rho_y_abs_median = 0.1771`（→ 0.177，非文档里流通的 0.203 / 0.247）、
`aggregate.rho_syn_median = 0.867`、`aggregate.rho_opp_median = 0.8696`。
本报告引用的 ρ_region_opp 0.884 / ρ_shuf 0.895 / 配对 ρ_syn 0.842 来自同一 metrics.json 的
`region_opposition` 与 `shuffle_control` 节。

## 九、scache 取值域埋雷（主 agent 警告，已独立复核 + 已处置）

- **复核**：`/var/cache/veradata/scache/ro9` 前 300 条实测取值域 **[−13.703, +1.056]**、
  **>0 的格占 0.0093**，`meta.norm = {"kind": "presoftmax-logit-raw", "note": ...}` **无 `domain`**；
  `tools/scache/upsample.py:53` 的 `upsample_s(..., clamp=(0.0,1.0))` 是默认参数。
  与主 agent 给的 [−12, +1.2] / ~1% 一致。
- **处置**：`config/finalize_scache.py` 给本臂两个 arm 写了 `_ARM_INFO.json`
  （全 arm 分位数 + `consume_recipe` + WARNING），并把 `domain_global` /
  `domain_global_p1_p99` / `consume` 回填进全部 772×2 条 `*.meta.json` 的 `norm`。
  `analyze_ro3.write_scache` 也已改为逐条写 `domain`。
- **本臂取值域**：`ro3-l11-h5` = [1.79e-7, 0.0850]（post-softmax 概率，全正但量级 1e-3，
  clamp 不会归零但会视觉塌陷）；`ro3-fused` = [−3.256, 8.805]，**58.8% 的格 < 0**，
  clamp=(0,1) 会吃掉大半。两者都必须先按 `_ARM_INFO.json` 映射再消费。

## 十、追加的待主 agent 决策

- **U5（`.cgt` 主列 n 偏小）**：主列口径（background × normal）只有 81 源，CI95 [0.805, 0.883]。
  保守默认 = 主列照报、并列口径（113 / 159）同表列出，不挑最大的那个当结论。
  若要收窄 CI，需扩区域对立批（G1 现成脚本 `build_region_opp.py`，约 428 次前向 ≈ 2 min）。
- **U6（RO-3 的 where 是否只是「名词接地」）**：本臂信号来自 instruction 的名词短语。
  这属于**结论范围**问题而非实现问题，必须由 RO-X1（有名词/无名词分离）回答。
  保守默认 = REPORT 只写「指令条件的区域定位」，**不写**「编辑意图理解」。
- **U7（嵌套 CV）**：折内 CV 控住了 2688 路选头，但 k（6 路）与 (读法, 池)（8 路）
  的选择没有进 out-of-fold。保守默认 = 全表列出、不只报最好一行；若审阅要求严格，
  需再套一层外循环（成本 ×6，缓存已在 `/var/cache/veradata/ro3_analysis_cache.pkl`）。

## 十一、执行过程中的两次返工（如实登记）

1. **fp16 → float32 重导**（§〇.4）：首轮 fp16 落盘下 `post` 的 sink 主导头整片下溢为 0
   （单样本 336 个 (层,头) 里 2 个全 0、36 个 >10% 零），会伪造并列秩、污染本臂核心对比。
   整批重导，fp16 数据已删除。
2. **压缩 → 不压缩 / 线程限流**：`np.savez_compressed` 的 zlib 在 float32 上耗时约为前向的 6 倍；
   改 `np.savez`（4.6 GB）。分析端 `torch` 默认起 121 线程做 (336,64,64) 的小张量运算，
   在本机（48 核、常年被其他 agent 的 36-worker 作业占满、load ~120）全部时间花在线程栅栏上，
   单个 k 值 >25 min；限到 4 线程后恢复正常。同时加了重载阶段缓存
   `/var/cache/veradata/ro3_analysis_cache.pkl`（约 25 min 的 npz 重载可复用）。
   **两次返工都只涉及存储/调度，未改判据、未改数据、未改采样。**

---

# 补件记录（主 agent 2026-08-03 晚：解除 RO-W 阻塞 + U3 口径钉死）

## 十二、补件① D-CONSTRUCT **S-train 全量 1600** 读出（已完成）

- **指令模板 100% 复用 RO-W**：`config/build_jobs_construct.py` **import**
  `experiments/ROW_basis_coeff_20260803/rowlib.py:instruction_for`（不复制、不另造），
  `class5` 从 RO-W 已落盘的 `/home/bc/data/row_basis_20260803/meta_train.s*.jsonl` 读，
  **逐字节复现 RO-W 的 `itag="real"` 指令串**。
  实测：1600 条全部拿到 class5，**239 个不同指令串**（模板必然大量重复，`uid` 保证键唯一）。
  只出 `real` 一档；`shuf`/`shufin`/L6 `rega|regb` 是 RO-W 自己训练侧的对照档，不属本补件。
- **导出**：卡 0，1600/1600，**0 error**，wall **18.6 min**，显存峰值 **1.41 GB**。
  栈落 `/home/bc/data/ro3_stacks_construct_20260803`（4.2 GB，放 /home 是因为 `/` 只剩 57 GB）。
- **两个 arm**（`write_construct_arms.py`）：
  | arm | 条目 | 取值域（arm 级） | p1–p99 | >0 占比 |
  |---|---|---|---|---|
  | `ro3-l11-h5-construct` | **1600** | [0.0, 0.1749] | 0.0 – 0.0183 | 1.00 |
  | `ro3-fused-construct` | **1600** | [−3.2695, 12.6484] | −1.3926 – 1.4971 | **0.291** |
- **与已有 arm 严格同尺度**：融合的逐头全局 z-score 统计量 μ,σ **没有在构造集上重估**，
  而是从 `/var/cache/veradata/ro3_analysis_cache.pkl`（G1 区域批 fields）按 `ro3-fused`
  落盘时的**同一口径**重算（全区域批、a/b 两条件、同一 r_idx）；r_idx / sign / w 也全部
  取自 `metrics.json → fusion["32"].folds[0]`。meta 里写了 `zscore_stats_scope` 与
  `sibling_arm` 备查。**否则两个 fused arm 不同尺度，RO-W 蒸馏会踩隐性域漂移。**
- **键对齐自检（从 RO-W 侧独立复算）**：用 RO-W 自己的 `rowlib.instruction_for` +
  `api.instr_hash` 重新生成 1600 个 `uid__instr_hash`，对两个 arm 逐条 `exists()` →
  **1600/1600 命中，0 miss**。RO-W 可直接消费。
- `meta.norm.domain` / `domain_global` / `domain_global_p1_p99` / `consume` 全部写齐
  （1600×2 条 meta 已由 `config/finalize_scache.py` 回填，每个 arm 目录带 `_ARM_INFO.json`）。

## 十三、补件② U3：prefill 位 vs 生成位（进行中，见 §十四）

主 agent 澄清：RO-3 是 **prefill**（EXPERIMENTS_v3 §2.1 明写"prefill 缓存全层全头"），
RO-9 / RO-9b 是**生成位**——两家本来就不一致，这个对照要钉的正是这一差。
RO-9b 新判据（special token 通道 AUC 0.920 但 AUC_target 0.5719、Δ_shuffle 是 RO-1 的 1/10）
读的是生成位，若两位有系统差异则三家横比要重算。

探针 `probe_genpos.py`（G1 区域对立批前 100 源 × reg_a/reg_b，带 greedy 生成）同时回答：
- **Q1（决定 RO-W 要不要带生成）**：`instr` 池两位是否等价。
  **因果掩码下 instruction token 的注意力行不受其后 token 影响，理论上应当等价**——
  脚本**不假设**，实测逐元素差、逐读出空间 ρ、以及 336 路 AUC_target 的逐点差。
- **Q2（决定 RO-3 §四 的表述能写多硬）**：`gl` 池在**追加位** vs **自然生成位**的
  AUC_target / 差分场 AUC 差多少。RO-3 §四 判「special token 读不到」用的是追加位；
  若生成位显著更强，该结论必须收窄成"追加位读不到"。

## 十四、三条来自主 agent 的更正/警示（已核对并落到本臂）

1. **journal 的 `I_in` 路径已死（D-25 活实例）**。RO-W 实测 5/5 指向已清理的
   `/mnt/ramstage/...`。**本臂两批数据都不受影响**：D-CONSTRUCT 读的是
   `T4_construct/sanity/*/L*/*_in.png`（本地实文件）；G1 批读的是
   `/var/cache/veradata/g1_srcimg_20260803/`（G1 暂存的实拷贝）。
   **而且本臂的 `.cgt.png` 取用走的正是安全路径**——`analyze_gt3col.CgtMaskBank` 按
   shard 索引的 `(shard, offset_data, length)` **seek 直读 tar 成员**，从不碰 journal 的
   `C_GT` 绝对路径（那些同样指向 `/mnt/ramstage`）。**建议 N5 复用本 bank 时保留这一点**：
   任何走 journal 绝对路径的实现都会踩雷，输入图应同理改读 shard 的 `.in.jpg`。
2. **`pgrep -f` 的第四种表面**：`pgrep -f <pattern>` 会匹配到**正在 grep 它的那条 shell
   自己**——判活、判忙都一样。本臂前期确实踩过（`until ! pgrep -f "analyze_ro3.py"`
   的等待循环自匹配、永不返回，只好改手工轮询）。**本次补件的两个作业全程只用
   `ps -p $PID`**（提交四步 + 等待循环），`job.marker` 里记了每一步的实证结果。
3. **RO-W 的可复用件不重造**：`rowlib.instruction_for` / `instruction_l6_regions` 已 import；
   无名词模板在 `cache_vlm2.py::NONOUN`，本臂建议 N4（RO-X1 有名词/无名词分离）时
   **直接 import 那一个，不要另写**；D-SFT-L 装配走 `build_sftl.py`
   （3,521 → S-train 3,302 / S-val 219，已排除 3,082 条 `winner_confidence=low`，S-test 189 未动）。

## 十五、U3 实测结果（补件② 完成）

- 协议：G1 区域对立批**前 100 源** × reg_a/reg_b（bg 子集 72），带 greedy 生成，
  中位生成长度 325.5，GL fallback 率 0.000；prefill 侧读已落盘 npz（零重复前向）。
  卡 1，wall **51 min**。**本节数字在 100 源子集上，与主报告的 165 bg 子集不可混读。**
- **Q1（`instr` 池）**：空间 ρ 中位 **0.99994 / 0.99987**，AUC_target@L11H5 Δ = **+0.0001 / +0.0008**，
  336 路 |Δ| 中位 0.0006–0.0008。**等价，实测确认（非假设）**。
  残差是 bf16 累加序噪声（序列长度不同 → matmul tiling 不同），只在近简并头上放大
  （`post` 的 ρ 最小 0.261 那一格是图像侧质量近零的 sink 主导头，AUC 本就 ≈0.5，不进判据）。
  ⇒ **RO-W 用 prefill-only，不必带生成。**
- **Q2（`gl` 池）**：生成位比追加位**略强**（AUC_target +0.0165 / 差分场 +0.1078），
  两位空间 ρ 中位仅 0.36–0.61、argmax 从 L06H05 漂到 L19H07 —— **两位确实不是同一张图**。
  但生成位的 max AUC_target = **0.5908**，仍 < 主报告零分布 **q95 0.596**（且本子集源更少、
  零分布只会更宽）；差分场 0.7010 也远低于 0.842。**属「未触发死刑」，不是正面证据。**
- **与 RO-9b 横比**：RO-9b 在生成位得 AUC_target **0.5719**；本臂在生成位取 336 头 × 2 读法的
  **max**（口径偏高）得 **0.5908**。差 0.019，同结论同量级 → **三家口径已对齐，横比不必重算。**
- **对主报告的更正**：§十一「局限」里「gl 桥接位不等于 RO-9 原位、需带生成重跑（×50，≈10h）」
  **该局限已消除**（本节即那次重跑，实际 51 min）。措辞收窄为：**两位都读不到指令条件的
  where，生成位略强但仍未越过零分布 q95**。U3 **关闭**。
