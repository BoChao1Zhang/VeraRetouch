# NOTES — RO-2 · logit lens 概率场（实施前核实 + 假设 + 待决策）

- 实验编号：EXPERIMENTS_v3 §2.1 **RO-2 行**（logit lens 概率场）
- 日期：2026-08-03｜分支 `lens-exp`｜卡 1（`CUDA_VISIBLE_DEVICES=1`）
- 引用文档节（只读引用节，未读全文档）：
  - `docs/EXPERIMENTS_v3_2026-08-02.md` §2.1 RO-2 行 + RO-X2 说明（L56）+ Changelog 末三条
  - `docs/IMPL_DOSSIER_2026-08-02.md` §5.3（Qwen 系 VLM 适配点）+ §六 探针工具链表（logit lens 行）
  - `docs/DATA_ASSIGNMENT_2026-08-02.md` §1.1/§1.2 切分纪律 + §3.3 L83（RO 零训练臂统一评分集）
  - `experiments/G1_s_identifiability_20260803/REPORT.md`（三元组协议、AUC_target 判据、D-31 表述纪律）
  - `tools/readout/ro9_gl_attention.py`（模型加载 / prompt / token span / luma16 对齐 **复用**）
  - `tools/scache/README.md`（arm 目录约定）、`tools/harness/`（AUC/Pearson 口径）

---

## 一、实施前核实记录（全部**当场实测**，不照抄文档）

DOSSIER §5.3 写的是 **Qwen2.5-VL / Qwen3-VL**，本项目是 **llava_qwen2 0.5B + mobileclip_l_1024**。
逐条核实结果如下，**四条与 DOSSIER 不一致，照抄会得到错误读数**。

| # | 待核事实 | DOSSIER 记载 | **本项目实测** | 影响 |
|---|---|---|---|---|
| 1 | logit lens 模块路径 | `lm_head(language_model.norm(h_l))` | **`model.lm_head(model.model.norm(h_l))`**；`model.model` 是 `LlavaQwen2Model`（`LlavaMetaModel + Qwen2Model`），**没有 `language_model` 子模块** | 照抄 → `AttributeError` |
| 2 | 终端归一化 | 必须先过 RMSNorm | **确认**：`type = Qwen2RMSNorm`，`eps = 1e-6`。已实现 | 漏了读数全错（本实现未漏） |
| 3 | `tie_word_embeddings` | 2.5-VL-3B tied / 7B untied / Qwen3-VL-8B untied | **本项目 = `True`**，且**实测 `model.lm_head.weight is model.model.embed_tokens.weight` → True（同一对象、同一 `data_ptr`）**；`model.safetensors` 里**根本没有 `lm_head.weight` 键**（953 个键中只有 `model.embed_tokens.weight` / `model.norm.weight`） | 解释见 §二 |
| 4 | 视觉 token 定位 | id `151652/151653/151655` | **完全不适用**。本项目走 LLaVA 占位符 `IMAGE_TOKEN_INDEX = -200`，在 `prepare_inputs_labels_for_multimodal` 里就地展开成 256 个 embedding，序列里**没有**视觉 marker token。而 id 151652/151653/151655 在**本项目 tokenizer** 里实测是 `<problem_light_start>` / `<problem_globalcolor_start>` / `<problem_light_end>` ——**用这三个 id 定位就是定位到三个完全无关的文本 special token** | 用 `model.lens_image_spans`（lens-exp 插桩），实测 span=(14,270)，长度恰 256 |
| 5 | deepstack 式中间层视觉二次注入 | Qwen3-VL 在 [8,16,24] 层二次注入，lens 读数突变 | **本项目无此结构**：全仓库无 `deepstack` 字样；`mm_projector`（`Sequential`, mlp2x_gelu）只在**输入端注入一次**，24 层 `layer_types` 全 `full_attention`，无二次注入 | 逐层曲线的任何突变**不能**归因于二次注入；已在 REPORT 标注 |
| 6 | `hidden_states` 索引语义（**DOSSIER 未记，实测新增陷阱**） | — | transformers **4.57.1** 下 `output_hidden_states=True` 返回 **25 项**，**最后一项已经过 `model.norm`**（实测 ‖h‖ 均值 L22=332 → 末项 114；`allclose(hs[-1], norm(pre_norm_input))=True`）。若按 DOSSIER 写法对 `hidden_states[24]` 再 norm 一次 = **双重归一化** | 本实现用 `model.model.norm` 的 `forward_pre_hook` 截末层**未归一化**输出；`emb=hs[0]`、`L{li}=hs[li+1]`(li≤22)、`L23=hook` |
| 7 | 特殊 token id | — | `<retouch_light>`=151646 / `<retouch_color&temp>`=151647 / `<retouch_colormixer>`=151648（与 G1 一致） | RO-2 不用，仅登记 |
| 8 | logit lens 方法本身（**外部事实，已开原始来源核实**） | DOSSIER 只说「自实现 3 行」 | 原始来源 nostalgebraist《interpreting GPT: the logit lens》(LessWrong, 2020) 原文：*"the 'activations' here are the block outputs **after layer norm**, but before the learned point-wise transformation"*，且作者本人用的就是 tied unembedding。**"先过终端 norm 再乘 unembedding" 是方法定义的一部分，不是可选项** | 与 §一.2 一致 |

**核实方式**：#1–#7 由 `config/introspect_model.json`（本目录 `config/` 下的机器可读快照）落盘，
脚本 `tools/readout/ro2_logit_lens.py` 每次运行都把关键项复写进 `export_meta.json`，审阅可直接核。
#8 用 WebFetch 打开 LessWrong 原文核实（附录 B 之外的外部事实，按 CLAUDE.md 派工协议第 2 条办理）。

## 二、`tie_word_embeddings=True` 对 lens 读数解释的影响（重要）

tied 时的标准担忧是「早层 lens 会读出**输入 token 自身**（输入自反射偏置）」。
**本实验里这条担忧对 image token 位置不直接成立**，理由：

- image token 位置的「输入 embedding」**不是词表 embedding**，而是 `mm_projector(MobileCLIP 特征)`。
  它不在 unembedding 矩阵的行空间里，所以不存在「读回输入词」这回事。
- 但 tied 带来另一个**必须显式声明的后果**：unembedding 方向 = 输入 embedding 方向，
  于是 `emb` 层的 lens 读数直接反映「投影器输出和哪个词的 embedding 对齐」。
  实测 `emb` 层已能读出图像相关名词（见 REPORT §三的 top-token 表）——
  **这是 mm_projector 已把视觉特征对齐到词 embedding 空间的证据，不是 LLM 层的功劳**。
  逐层曲线在早层的任何强度**必须**这样解释，不能说成「LLM 早层就懂了」。
- untied 模型上同一实验的 `emb` 层读数会显著不同；**本臂结论不可直接外推到 untied 模型**（写进 REPORT 局限）。

## 三、架构性事实：image token 的 lens 场**必然与指令无关**（本臂最重要的前置）

部署 prompt 的 token 顺序（`ro9_gl_attention.build_inputs`，与 G1/RO-9/RO-3 逐字相同）：

```
[system prompt] <image>(→256 image token) \n <task_style> ...Now, you are acting as a Retouch Agent...
    Instruction: {instruction}
```

image token 严格**早于**指令文本（实测 span=(14,270)，序列总长 336，指令在 270 之后），
而 LLM 是**因果**注意力 ⇒ image token 的隐状态 h_l[p] 是 (图像, 前 14 个 system token) 的函数，
**与指令逐 bit 无关**。因此：

1. **「跨图打乱指令」对照批（任务卡硬要求）在 RO-2 上是架构性退化对照**：它必然给出 ρ≈1。
   本实验**仍然全批跑了**（`regictrl` 428 次前向），并额外加了 `numctrl` 批
   （同指令重跑 `dup` + 只加尾部无关句 `pad`）把 ρ 残差归因到 bf16 数值噪声——
   见 REPORT §五。缺了这批就无法排除「s 不依赖指令」，所以必须跑；但**跑出的 ρ≈1
   不是负面结果**，它只是复述了因果掩码。
2. **RO-2 真正有判别力的负控制是「跨图打乱目标词」**（`shuf_name_last`）：
   读别的图的主体词在本图上的概率场，标签仍用本图掩膜。这是 RO-1 同款负控制，
   本实验以它作为 Δ_shuffle 的承重列。REPORT 会把两种 shuffle 分开陈述，不混为一谈。
3. 推论（对 RO-5/RO-6 有用）：**任何从单次 prefill 的 image token 位置做的读出，
   在本部署 prompt 下都不可能有指令条件性**——这不是 RO-2 的缺陷，是 prompt 顺序的后果。
   见「待主 agent 决策」A2。

## 四、判据口径（与 G1 / RO-1 / RO-3 严格一致，否则不可横向比较）

- `AUC` = 归一化 Mann-Whitney U（并列平均秩），**只在 valid 格上算**
  （`expand2square` 黑边 pad 剔除，`luma_to_grid` 逐字复用 RO-9 实现）。
- `AUC_target` = median over pooled `{AUC(s_a, M)} ∪ {1 − AUC(s_b, M)}`
  （定义出处 `experiments/RO9_layer_verdict_20260804/analyze_layers.py:auc_target_list`，
  RO-3 `analyze_ro3.py` 沿用）。主口径取 **background 子集**（reg_b 字面 = 主体补集），
  同时报 all 子集。
  - **RO-2 的 s_a / s_b 来自同一次前向的两个词**（s_a = 主体词概率场，s_b = "background" 概率场），
    因为 RO-2 的读出量是「词 w 在位置 p 的概率」，换的是词不是指令——这与 RO-1（换文本查询）同构，
    与 RO-3（换指令跑两次前向）不同。已在 REPORT 明写。
- `Δ_shuffle` = median_i [AUC(s_a,M_i) − AUC(s_shufword,M_i)]。
- 掩膜二值化阈值 0.5（16×16 格覆盖率），与 G1/RO-3 一致。
- 逐层：`emb` + `L0..L23`，共 25 个读出点。

## 五、预注册（写在跑数之前）

| 量 | 预注册值 | 出处 |
|---|---|---|
| 晋级：跨图刻度标准差 | **< 0.15** | EXPERIMENTS_v3 §2.1 RO-2 行 |
| 晋级：AUC | **≥ 0.70** | 同上（两条**都**要满足） |
| 主判据 | **AUC_target**（G1 口径） | 任务卡；G1 教训：ρ 高有两解，AUC_target 只有一解 |
| 淘汰去向 | → RO-5（探针头） | EXPERIMENTS_v3 §2.1 RO-2 行 |
| 负控制 | 跨图打乱**词** Δ_shuffle > 0；跨图打乱**指令** ρ（架构退化，只报不判） | G1 审阅 B3 |
| 选词敏感性 | 7 个 scheme 全报；结论若强依赖选词 = 重要负面发现 | 任务卡 |

**刻度标准差的可操作定义**（预注册，见「待决策 A3」）：同语义 200 图、**固定同一个词**
（`person`）、每图算 GT 掩膜内均值 μ_in(i)，再取跨图标准差。同时报三档：

- `scale_std_raw` = std_i(μ_in)，**原始概率单位**；
- `scale_std_globalcal` = 先用**全局**（整个数据集一次估计，**非逐图**）1%/99% 分位线性映射到 [0,1]，
  再算 std_i —— **这一档对 0.15 判据**（RO-X2 的「全局 CDF」档，不违反禁逐图归一化红线）；
- `scale_cv` = std/mean（防「均值≈0 ⇒ 标准差自动<0.15」的退化 PASS）。

**D-31 退化 PASS 防护（预注册）**：`scale_std < 0.15` 单独成立**不算正面证据**——
一个恒等于 0 的场也满足。只有在 **AUC ≥ 0.70 同时成立**、且
**判别比 = mean_i(μ_in − μ_out) / std_i(μ_in) > 1** 时，才可写「获得跨图绝对刻度的正面证据」。
判别比一并预注册报出。

## 六、假设与自检

1. **假设**：`target_name`（journal `local.subject.name`）是可靠的目标语义词。已抽检：
   `person`/`woman`/`flower`/`bald eagle` 等，均为干净名词短语。多词短语取**末词**（`name_last`）。
2. **自检**：`name_first`（整短语首 token）作为对照 scheme，验证「取末词」不是结论的来源。
3. **自检**：`ctrl` scheme = 固定无关名词 `calculator`，AUC 应 ≈ 0.5；若显著 >0.5 说明
   概率场被某种与词无关的伪影（token 范数 outlier / 位置偏置）主导。
4. **自检**：D-0（token 范数 MAD3 outlier 剔除 + 4 邻域插值）**开/关两档都报**，
   与 G1/RO-3 同实现（`interpolate_masked`）。
5. **红线自查**：softmax 只在**词表维**（方法定义）；空间维**零归一化**——
   落盘的是原始概率 p∈[0,1]；`scale_std_globalcal` 用的是**全局**分位（一次估计、所有图共用），
   不是逐图统计量。AUC / Pearson 对逐场单调仿射不变，也不引入逐图归一化。
6. **fp16 存盘误差**：post-RMSNorm 隐状态以 fp16 落盘用于离线换词；判据用的概率场是
   **导出时 fp32 算好的**，不受 fp16 影响。

## 七、待主 agent 决策（**已采用保守默认继续，未静默拍板**）

### A1 · 「200 张同语义图」的构造方法（现成数据里没有明确的同语义分组）

- **事实**：RO 统一评分集的 214 源里 `target_name` 有 53 个不同值，最大组 `woman`(69)+`person`(64)=133，
  **不足 200**。
- **保守默认（已执行）**：从 `splits.sqlite3` 的 **split='val'** 源 ∩ veradata 银行 `cache/subject`
  中 `eligible ∧ subject.name == 'person'` 的 **485 个 S-val 源**里，按 `source_id` **字典序取前 200**
  （无随机、完全可复现）。**这是在已冻结的 S-val 集合内做选择，不是新的切分**，符合
  DATA_ASSIGNMENT §1.2「只读旁表，禁 ad-hoc 切分」。掩膜 = 同一 SAM3 主体软掩膜（D-MASKBANK）。
  与 region 批的重叠 **39/200**，已记账。
- **待决策**：① 是否接受这种「按 subject.name 分组」当同语义定义（另一合理做法是按 CLIP 文本
  相似度聚类，会引入外部模型的口径）；② 200 图里 39 个与 region 批重叠，是否要求完全不相交
  （代价：同语义组只剩 446 个候选，仍够，但与 RO-1/RO-3 的可比性下降）。
  默认：接受 subject.name 分组、允许重叠并记账。

### A2 · 是否加一档「指令前置」prompt 变体

- **事实**（§三）：部署 prompt 里 image token 早于指令 ⇒ image 位置的任何读出**架构上**无指令条件性。
- **保守默认（已执行）**：**不改部署 prompt**，按线上口径读，RO-2 的判据全部在部署口径下给出。
- **待决策**：是否值得单开一个小批（如 60 源 × 2 指令）把指令放到 `<image>` **之前**，
  测「表征端能否吃到指令」。这对 RO-5/RO-6/RO-7 是关键情报（若前置后出现指令条件性，
  说明「VLM 里有 where」只是被 prompt 顺序挡住了），但它**改变了部署契约**，属于计划变更，
  不由 subagent 拍板。**本轮未跑**。

### A3 · 「刻度标准差」的口径

- **事实**：EXPERIMENTS_v3 只写「刻度标准差<0.15」，未定义在什么单位上算。
- **保守默认（已执行）**：三档并报（§五），**以 `scale_std_globalcal` 对 0.15 判据**
  （唯一有量纲意义的一档：全局校准后 s∈[0,1]，0.15 = 满量程的 15%）。
  另加判别比与 CV 防退化 PASS。
- **待决策**：主 agent 若认为应以 `scale_std_raw` 对判据，请指定；本报告两档数字都在表里。

### A4 · 补集词的选取

- **事实**：区域批 214 源里 165 源的 reg_b 字面是「the background」，49 源是空间对侧
  （「the upper part of the image」等）——**空间短语没有可读的名词**，lens 读不了。
- **保守默认（已执行）**：`compl` scheme 一律用固定词 `background`；
  **AUC_target 主口径取 background 子集（165 源）**，与 RO-9/RO-3 的 `auc_target_*_bg` 完全同口径；
  all 子集（214）一并报，但标注 49 个 spatial 源的 s_b 是「background 词场」而非「空间对侧场」。
- **待决策**：是否要为 spatial 源另造可 lens 化的补集词（如「sky」「ground」，但会引入图像内容依赖）。

### A5 · scache arm 的层与 scheme

- **保守默认（已执行）**：按 **AUC_target(background 子集) 最高的层 + `name_last` scheme** 落 arm，
  arm 名 `ro2-l<层号>`。若最优层落在末两层，仍照落但在 REPORT 标注
  （RO-3 的「末两层 = 读到答案聚合」淘汰规则是 RO-3 行的判据，不自动适用于 RO-2）。
- **待决策**：RD 臂接入时是否要求 arm 用全局 CDF 校准后的场而非原始概率（RO-X2 的事）。
  默认落**原始概率**（零归一化），校准留给 RO-X2。

---

## 八、执行记录

见 `STATUS.md` 与 `config/job.marker`。

---

## 九、跑数后追加的实施记录（2026-08-03 晚）

### 9.1 跑数中发现并修正的两处方法问题（**分析前做的，不是事后挑口径**）

1. **RO-1 冻结集的 shuffle donor 不能直接用作 RO-2 的负控制**：它是**无约束** derangement，
   而评分集里 woman(69)/person(64) 各占 ~30% ⇒ **20.1% 的源拿到与自己同词的 donor**，
   那些源的 Δ_shuffle **恒等于 0**，中位数被压成 0。
   修正：另建两档**确定性** donor（按词分组排序后整体平移 max_group 位，保证异词；
   以及按语义大类 round-robin，保证异类），`distinct_words_check` / `distant_cat_check` 均为 true。
   **RO-1 原 donor 的数字一并报**，供与 RO-1 横比。
2. **AUC_target 无法区分「读到目标词」与「读到与词无关的图/底先验」**。
   发现动机：`ctrl` scheme（固定无关词 `calculator`）在 L0 的 AUC_target **高于**目标词。
   于是补跑 `supp_nosoftmax.py`，在 region/cval/samesem 三个集合上逐读出点报
   **AUC(目标词) − AUC(`calculator`)** 配对差（概率版与原始 logit 版各一份）。
   这是本报告 §二的承重证据，也是建议 C3（给所有 RO 臂加无关词对照列）的来源。

### 9.2 跑数后新增的事实（写进 REPORT，此处只记溯源）

- `hidden_states` 末项已过 `model.norm` —— 实测 `allclose(hs[-1], norm(hook_input))=True`，
  ‖h‖ 均值 L22=332 → 末项 114。本实现用 forward_pre_hook 取未归一化输出。
- 指令对照的数值基线：`dup`（bit 级相同输入）ρ **恰好 1.000000**、max|Δ| **恰好 0**；
  `pad`（只加尾部无关句）ρ 0.999934、max|Δ| 3.1e-2 ⇒ `reg_b`/`shufa` 的残差（ρ 0.99995、
  max|Δ| 5.2e-2）与「只改长度」同量级 ⇒ 换指令的差异全部是 bf16 数值噪声。
- 离线换词（fp16 隐状态重算）相对误差中位 **0.0025**；判据用的是导出时 fp32 概率，不受影响。
- D-0 逐读出点 outlier 占比：emb 0.097、L0 0.004、L7–L15 约 0.042、L23 0.017；
  D-0 开/关的 AUC_target 差 0.0015。

### 9.3 待主 agent 决策项的最终状态

| 编号 | 状态 |
|---|---|
| **A1**（同语义 200 图的构造） | 已按保守默认执行（S-val ∩ subject.name==person 字典序前 200，重叠 39 已记账）。**仍待主 agent 追认**；结论对该选择不敏感（判别比 <1 在 25/25 读出点成立） |
| **A2**（是否加「指令前置」prompt 变体） | **未跑**（改变部署契约，属计划变更）。本轮拿到了它的动机证据：架构上封死了表征端指令条件性，见 REPORT §三(a) 与建议 **C5** |
| **A3**（刻度标准差口径） | 三档全报。**新增发现：这条判据线 25/25 读出点全过，没有筛选力**，建议改判别比 >1（REPORT 建议 **C4**） |
| **A4**（spatial 源的补集词） | 已按保守默认（固定 `background`，主口径取 background 子集 165）。all 子集一并报，差 0.003 |
| **A5**（scache arm 的层与 scheme） | **落了两个 arm**：`ro2-l0`（预注册判据最优点，各 437 条）与 `ro2-lemb`（唯一词特异的读出点，各 437 条）。**建议 RD/RO-5 接 `ro2-lemb`**，理由见 REPORT §二；`ro2-l0` 保留仅为让审阅能复核判据数字 |

### 9.4 新增待决策（本轮结果驱动）

- **B1**：REPORT 的形式判定是 `promote=true`（两条预注册线都过），实质判定是**淘汰→RO-5**。
  两者冲突时以哪个为准，需要主 agent 在 EXPERIMENTS_v3 里明文化
  （建议：D-31 的实质判定优先，并把 C4 的判据修订写进 changelog）。
- **B2**：`calculator` 是**单一**对照词。是否要求所有 RO 臂用一组（≥8 个）无 referent 词的分布？
  代价很低（离线换词即可，不需重跑前向）。默认：本轮只报单词结果并在局限里声明。

---

## 十、A2「指令前置」对照 —— 预注册（**写在数据落地之前**）

- 主 agent 于本轮拍板执行（原 NOTES §七 A2 待决策项，现已关闭）。
- 变体定义（`tools/readout/ro2_logit_lens.py::build_inputs_ordered`）：
  - `image_first`（部署口径，与 G1/RO-9/RO-3 逐字一致）：`<image>\n {TASK}…Instruction: {instr}`
  - `instr_first`（A2）：`{TASK}…Instruction: {instr}\n<image>`
  - **除 `<image>` 的位置外，文字逐字相同**。实测 token 差异仅一处 BPE 边界合并
    （`image_first` 有 `"."`+`"Ċ"` 两个 token，`instr_first` 合并成 `".Ċ"`），
    序列长 87 vs 86。**这个 1-token 差异远小于 numctrl `pad` 对照**
    （`pad` 追加 ~12 token，实测 ρ 仍有 0.99993）——所以长度不是解释。
  - 实测 span：`image_first` (14,270)/总长 342；`instr_first` (80,336)/总长 341。
    导出工具加了**顺序守卫**（写错顺序直接 raise，不静默出错误结论）。
- 样本：**与 image_first 完全同一批 925 作业**（同 `ro2_jobs.json`，同图同指令同 donor），
  只换 `--prompt-order`，产物落 `/home/bc/data/ro2_lens_instrfirst_20260803/`。判据与聚合口径不变。

### 预注册判据

| 结果 | 判定 |
|---|---|
| ρ（跨图打乱指令）**显著低于** image_first 的 0.99995，**且**其偏离度 (1−ρ) 显著高于**同顺序**下的 `pad` 数值噪声地板 | **指令信息确实能进入 image token 表示** ⇒ G1 的失败可定性为 **prompt 顺序问题**（一行代码） |
| ρ 仍 ≈ 1，且与同顺序 `pad` 地板无显著差异 | **排除 prompt 顺序这个解释**，问题在更深处（那时才轮到「信息真的不在」这一档） |

- 统计量用 **1−ρ（偏离度）**：ρ 都贴着 1，直接比 ρ 看不出量级。
- 判据阈值（预注册）：`dev(reg_b) / dev(pad) > 2.0` **且** 配对 Wilcoxon `p < 0.01`
  才判「指令进入表示」。配对在**同一批 30 个 numctrl 源**上做。
- 同时报（主 agent 要求）：**AUC_target、Δ_shuffle（异词/异类两档）、判别比**（A3 采纳的新刻度判据），
  全部与 image_first 配对并排。

### 必测的两件事（否则结论会被打回）

1. **模型行为是否退化**（`behavior_check.py`）：50 源 × 两种顺序各一次 greedy 生成。
   **合法性口径取自部署代码本身**：`llava/model/VeraRetouch.py::_generate` 对三个 retouch token
   各取 `torch.where(mask)[0][0]`，**任一缺失即 IndexError、模型产不出图** ⇒
   `pipeline_legal ⟺ 三个 token 全在`。另报生成长度分布、6 个 plan/problem 结构块的成对闭合率、
   是否撞 max_new_tokens、与部署顺序生成文本的字符级相似度。
   **判读规则（预注册）**：读出变好但 `pipeline_legal_rate` 显著下降 ⇒ 该方案**不能直接用**，
   从「一行代码修好」降级为「需要重新 SFT 才能用」，代价完全不同，必须在 REPORT 明写。
2. **前置顺序下的 `dup` / `pad` 数值对照**（已在同一批作业里，无需另跑）：
   新顺序下 ρ 的下降必须显著超过 bf16 噪声地板，否则又是一次假阳性。

### 排卡记录

卡 1；启动时实测 `nvidia-smi` = 68.9 GB / 97.9 GB 已用（余 29 GB），本作业实测峰值 reserved 3.48 GB；
未 kill 任何非本人进程；两个作业（导出 + 行为对照）均写入 `job.marker`。

### 10.6 A2 结果与预注册的对照（跑数后追记）

| 预注册分支 | 实测 | 命中 |
|---|---|---|
| ρ 显著下降 **且** dev(reg_b)/dev(pad) > 2.0 **且** p < 0.01 ⇒「指令进入表示、G1 = prompt 顺序问题」 | ρ 0.999944→0.991870（dev ×146）；比值 **1.478**（<2.0）；p **2.3e-4**（<0.01） | **部分**：比值门未过 ⇒ 判 false |
| ρ 仍 ≈1 且与 pad 地板无差异 ⇒「排除 prompt 顺序」 | ρ 并非仍 ≈1（确实降了），但**与 pad 地板只差 1.48×** | **部分** |

**两个分支都不完全命中，因此按预注册的更强证据（读出判据本身）收口**：
12 条读出判据在 ±0.006 内全部不变（词特异性 Δ @emb 两位顺序**完全相同到小数点后四位**，
因为 emb 是 mm_projector 输出、是图像的纯函数——这同时是一次管线自检）。
⇒ **实质判定：prompt 顺序被排除。**「指令进得来但读出不变好」比任何一个预注册分支都更强，
且方向唯一。已在 REPORT §十.4 逐条写明，未事后改判据。

### 10.7 scache 埋雷修复（主 agent 提醒）

`ro9` 臂写 [−12,+1.2] 原始 logit 但 `meta.norm` 无 `domain`，消费端 `upsample.py::upsample_s`
默认 `clamp=(0,1)` ⇒ 下游静默拿到全 0。本臂两个 arm 已补齐（与 `ro1-clearclip-l11` 同款约定）：

- `domain` / `domain_p01_p99` / `median`：整臂实测值域。`ro2-lemb` min/中位/p99 = 2.65e-27 / 1.61e-08 / 2.78e-01。
- `domain_note`：明写两个坑——概率量级 1e-5，`clamp=(0,1)` 虽不截断，但 `guided_blur` 的
  `eps=1e-4` 是按 [0,1] 满量程调的，直接喂会让 guide 项压过 s 项；任何 0.5 绝对阈值得到全 0。
- `recommended_map`（线性，全局 p01/p99）+ **`recommended_map_log`（重尾分布推荐档）**，
  两档都是**整臂常量、所有图共用**，属 RO-X2「全局 CDF」档，不触红线。
  并注明：AUC / 单一全局阈值 Dice 对任何**全局单调映射**不变 ⇒ 换 log 档不改 REPORT 任何 AUC 数字，
  只有 scale_std / 判别比会变（REPORT §四用线性档）。
- `caveat`：指向 REPORT §二（本臂的场不是目标词条件的；唯一词特异档是 `ro2-lemb`）。
