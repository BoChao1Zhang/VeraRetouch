# RO-9c · 实施前核实记录 / 假设清单 / 待主 agent 决策

实验：`RO9c` — 6-09 主体读出（M1_diffLMM）复现 + 首次量化
日期：2026-08-05　｜　GPU：**卡 1**（卡 0 有他人作业 42 GB / 86%）
环境：`/home/bc/VeraRetouch/.venv-lens/bin/python`（torch 2.10.0+cu128 / transformers 4.57.1）

---

## 一、读了哪些文档节（只读引用节，不读全文档）

| 来源 | 读了什么 | 取到的事实 |
|---|---|---|
| `CLAUDE.md`（全文，硬要求） | 红线速查 / 长任务提交纪律 / 交付物规范 / 数据纪律 | 见下各节 |
| 回收站 `attention_method_experiment_2026-06-09.md` **§二、§五、§七** | VeraRetouch special-token 实验设置、指标定义、aspect 可分性 | 指标四件套 `point∈GT`(±6px) / `mass-in-GT` / `IoU` / 指称可分性=`1−平均余弦`；§五「raw 三 token 几乎相同、只有左上角 sink；M1 后 GC→中心主体、SC→局部、L→弥散」；§七 token-attn aspect 可分性 **0.714**、post-MLP 0.954 |
| 回收站 `DiffLMM_method_and_conclusion_2026-06-09.md` **§二、§三** | Eq.3 方程级定义 + 落地伪代码 | `A_i^norm = A_i^reduced − (1/r)Σ_j A_j^reduced`；落地伪代码 `common = red[图像之后所有 query 行].mean(0)` |
| 回收站 `scripts/veraretouch_special_token_attn.py`（全文 109 行） | 被复现脚本本体 | 见 §二「照抄了什么」 |
| `experiments/G1_s_identifiability_20260803/analyze_g1.py`（`SubjectMaskBank` / `luma_valid_from_image` / `roc_auc`） | 主体掩膜银行与对齐口径 | 直接 `import` 复用，零重写 |
| `tools/readout/ro9_gl_attention.py`（`load_model` / `build_inputs` / `luma_to_grid` / `outlier_mask_from_norms`） | 模型加载、prompt、网格对齐、D-0 | 直接 `import` 复用 |
| `tools/readout/rod_difflmm_aas.py` + `experiments/G1b_difflmm_20260803/analyze_g1b.py` | G1b 已核实过的 A&S 实现与 Eq.3 落地 | 见 §三 核实-2 |
| `docs/EXPERIMENT_REGISTRY.md` §3.1 | G1 / RO9-L / RO9b / RO-1 / RO-X1 / G1b 的现有判决 | 见 §六 边界 |
| `experiments/ROX1_clipside_20260803/REPORT.md`（表 §三、建议 X1/X4） | 固定短语基线数字 | `the main subject` 主体 AUC **0.907**、AUC_target **0.523** |

---

## 二、照抄了什么（不发明）

`tools/readout/ro9c_m1_teacherforced.py` 逐行对应原脚本第 60–70 行与 81–83 行：

| 原脚本 | 本工具 | 说明 |
|---|---|---|
| `A = stack([a[0] for a in out.attentions])` | `out.attentions[li][0]` 逐层处理 | **post-softmax 概率**，eager |
| `red = A.mean(0).mean(0)` | 层内 `mean(0)`（头）+ 分析侧 `layers.mean(0)`（层） | 拆成两步只为把逐层分量落盘，数值等价 |
| `red_img = red[:, img_block]` | `[:, :, img_start:img_end]` | 只留 256 个 image patch 列 |
| `post = arange(img_block[-1]+1, seq)` | `post_lo = img_end; n_post = total − img_end` | 图像之后的**所有** query 行 |
| `common = red_img[post].mean(0)` | `common[li] = red_np.mean(0)` | **M1 的共模** |
| `rt_pos = [seq-3, seq-2, seq-1]` | `rt_rows = [total-3, total-2, total-1]` | teacher-force 的 3 个 retouch token 行 |
| `v = vec − common` | `field(..., "m1")` | **M1_diffLMM** |
| `render()` 里的 jet / 3×3 smooth / blend 0.55 / argmax 双圈 / 标签 | `make_figs.py:render()` | **逐字复刻**，仅用于着色 |

**唯一的扩展**：把「一张 `sample_flower.jpg`」换成 G1 冻结的 214 源 × 4 条件，并把
`raw`（未减均值的原始分量，逐层）、`common`（均值图，逐层）、`common_prompt`、
D-0 outlier 掩膜、`n_post_rows` **原样落盘**，任何人可复算（任务卡明确要求）。

---

## 三、在线/原始来源核实记录

**核实-1 · 被复现对象本身**：三份 6-09 文件均以只读方式打开（`Read`），**未修改、未移动**。
数字 0.714（aspect 可分性）、"GC→主体 / SC→局部 / L→弥散"、"raw 只有左上角 sink"
均直接取自原文（`attention_method_experiment_2026-06-09.md` §五、§七），非二手转述。

**核实-2 · DiffLMM Eq.3**：按任务卡指示，**优先采用仓库内 G1b 已核实实现**而非重新上网。
`tools/readout/rod_difflmm_aas.py` 模块 docstring 记录了官方实现 `aas/gcg.py` 的
`attn_mean = attentions.mean(dim=0); attentions = attentions − attn_mean`（dim=0 = 输出
token 轴）与论文 Eq.3 的对应，并注明 arXiv:2410.08209 / ICCV 2025 / Shengcao Cao 等。
`DiffLMM_method_and_conclusion_2026-06-09.md` §附录另注「机制数字经 arXiv HTML + ICCV
PDF 逐字核验」。**两处独立记录一致**，故本实验直接采用，未新增外部引用。
⚠️ 本 REPORT 不引用任何 6-09 文档与 G1b 之外的新数字/新 URL。

**核实-3 · 模型目录不被污染 —— ⚠️ 顺带查出一处历史遗留污染**：
原脚本第 27–28 行会 `os.rename` 掉 `/home/bc/data/models/VeraRetouch/generation_config.json`。
本工具**不调用 `generate()`**（teacher-forced 单次 prefill 即可），因此根本不需要动它；
`rg generation_config` 对本实验全部代码返回**零命中**（只有一行注释）。

但跑后核验发现：**共享模型目录里 `generation_config.json` 当前并不存在**，只有
`.generation_config.json`（153 B，mtime `May 17 13:37`——`os.rename` 保留 mtime，故这个
时间戳是文件创建时间而非改名时间）。即 **6-09 那次改名从未被恢复**，该目录已带着这个
状态被 G1 / RO-9 / RO-9b / RO-3 / G1b 等全部读出臂使用至今。内容为：
`{"_from_model_config": true, "eos_token_id": [151645], "pad_token_id": 151643,
"transformers_version": "4.57.1", "use_cache": false}`。
⚑ 这**不是本实验造成的**，本实验也**未擅自恢复**（见决策 **D-6**）。

**核实-4 · eager 与三 token id**：加载后断言 `config._attn_implementation == "eager"`
（实测 `eager`）、`out.attentions is not None`、层数 == 24；token id 实测
`L=151646 / GC=151647 / SC=151648`，与 6-09 文档 §五记载的 `151646/151647/151648` **完全一致**
（说明 checkpoint 未变）。任一断言不满足直接 `raise`，**无回退分支**。

**核实-5 · 6-09 用的 prompt 到底是哪条**：原脚本用 `Infer_Auto_Dataset`。读
`data/infer_dataset.py:60-64` 确认它是 `<Auto_Retouch_Task>` 模板、**完全不含指令**
（与 G1/RO-9 用的 `<Style_Retouch_Task>`＋逐源指令是两条不同 prompt）。本实验因此
**同时跑两者**（见决策 D-1）。

**核实-6 · 掩膜与对齐同源**：主体掩膜与 luma/valid 直接复用
`experiments/G1b_difflmm_20260803/config/{subject_masks16,luma_valid16}.npz`——
它们由 G1 的 `SubjectMaskBank` + `luma_to_grid` 生成（同一 bank、同一对齐路径），
实测对 214 源**覆盖 214/214**。`point∈GT` 需要全分辨率掩膜，另经同一 `SubjectMaskBank`
现场重建（`config/point_ok16.npz`）。**零重新采样、零 ad-hoc 切分。**

---

## 四、预注册（跑数之前写进代码常量）

写死在 `analyze_ro9c.py` 顶部 `CRITERIA` / `PRIMARY_*`，本文件与代码一致：

- **主判据**：M1 的 `mass-in-GT` 相对 raw 有显著提升（配对 Wilcoxon **p<0.01**），
  且 M1 的主体 **AUC ≥ 0.75**。
- **定性判据**：raw 三 token 两两余弦 **>0.9**；M1 后可分性上升（对照 6-09 的 **0.714**）。
- **边界判据**：`AUC_target` 预期 **0.45–0.60**；**≥0.65 必须标红另立节并上报主 agent**。
- **主 token = GC（`colortemp`）**，依据 6-09 §五原文「GC（全局色温）强注意力集中在中心主体
  花簇」。⚑ 该指派在看到任何本实验数字**之前**固化，不是事后挑选；三 token 全表照报。
- **主条件 = `auto`**（6-09 原脚本那条 prompt），**主层段 = 全 24 层**（6-09 `A.mean(0)`）。
- 随机场对照：**同支撑空间置换**（valid 格集合不变，值置换），seed=20260805，20 次取均值。

---

## 五、假设与已当场核实项

| # | 假设 | 处理 |
|---|---|---|
| A1 | `out.attentions` 是 post-softmax 概率 | ✔ 核实：eager 路径返回 softmax 后权重；实测每行 image 列求和 ≈0.34–0.47（<1，因为还有文本列），非 logit 量级 |
| A2 | teacher-force 的 3 个 token 就在展开序列末尾 3 行 | ✔ 断言 `total == ids.shape[1] + 255` 且 `attn.shape[-1] == total`，214×4 全通过 |
| A3 | image span 是 `lens_image_spans` 给的 256 连续列 | ✔ 断言 `img_end − img_start == 256` |
| A4 | 6-09 的 `img_tok_pos` 取自未展开 input_ids 仍然正确 | ✔ 因 `<image>` 占位在 prompt 前部（实测 span 起点 14），其前无文本被展开影响；本工具直接用 `lens_image_spans`，规避该隐患 |
| A5 | 三 token id 与 6-09 当时一致 | ✔ 见核实-4 |
| A6 | 主体掩膜 16×16 覆盖全部 214 源 | ✔ 实测 214/214 |
| A7 | `mass-in-GT` 的"归一化"不违红线 | ✔ 它是**比值**（GT 内质量 ÷ 总质量），尺度不变、不做平移，是 6-09/DiffLMM 的原定义；另并排给 `AUC`（纯秩次，任何单调变换不变）与 `smd`（不做任何归一化）作旁证 |

---

## 六、⚑ 待主 agent 决策（已采用保守默认继续，未静默拍板）

### D-1 · 复现用哪条 prompt？
- **事实**：6-09 用 `Infer_Auto_Dataset` = `<Auto_Retouch_Task>`，**零指令**；G1 区域批的
  `reg_a/reg_b` 用 `<Style_Retouch_Task>` + 逐源指令。前者无法算 `AUC_target`（需要两个
  区域条件），后者不是 6-09 的原 prompt。
- **保守默认（已执行）**：**两者都跑**，共 4 条件 ×214 源 = 856 次前向（prefill-only，≈4 min）：
  `auto`（6-09 原样，主条件）/ `reg_a` / `reg_b`（供 `AUC_target`）/ `fixed`（同 reg 句框、
  对 214 张图逐字相同的 deictic 指令，固定短语对照）。
- **留给主 agent**：REPORT 的头条数字该以 `auto` 还是 `reg_a` 为准。本报告以 `auto` 为主
  （忠实复现 + 零指令因此"找主体"的说法最干净），`reg_a` 全表并排。

### D-2 · `common`（共模）的取法
- **事实**：论文 Eq.3 沿**输出 token 轴**取均值；6-09 脚本落地成「**图像之后的所有 query 行**」
  的均值（teacher-forced 探针里没有生成段，只有 prompt 尾部 + 3 个 retouch 行）。两者在本
  设置下不等价。
- **保守默认（已执行）**：**忠实 6-09**（`common` = 图像之后所有行的均值）。同时落盘
  `common_prompt`（排除末尾 3 个 retouch 行）作为鲁棒性档 `m1p`，指标表并排给出。
- **留给主 agent**：若日后要与 G1b 的 `aas` 严格对齐（那边 common 只取生成段），
  需另开一档；本实验不替它拍板。

### D-3 · M1 场含负值时 `mass-in-GT` 怎么算
- **保守默认（已执行）**：先 `relu` 再算比值（DiffLMM 的 point/mass 口径本就只看正响应，
  原脚本 `render()` 也是 `np.maximum(m,0)`）。**同时**并排报 `AUC`（纯秩次、含负值区）
  与 `smd`（GT 内外均值差 ÷ 场标准差，**完全不归一化**），三者若结论一致则不依赖该选择。

### D-4 · D-0（token 范数 outlier 修复）要不要开
- **事实**：G1/RO-9/G1b 一律开；6-09 原脚本**没有** D-0。实测本批 outlier 占比中位 ≈1–5%。
- **保守默认（已执行）**：主表**关闭 D-0**（忠实 6-09）；outlier 掩膜已逐层落盘，
  `analyze_ro9c.py --repair` 可一键出修复档做鲁棒性对照。

### D-6 · 共享模型目录的 `generation_config.json` 要不要恢复？
- **事实**：见核实-3。`/home/bc/data/models/VeraRetouch/generation_config.json` 已被 6-09
  那次实验改名为 `.generation_config.json` 且**从未恢复**。本轮战役的**全部**读出臂
  （G1 / RO-9 / RO-9b / RO-3 / G1b / RO-W…）都是在这个状态下跑的。
- **影响面**：文件里有 `eos_token_id=[151645]`、`pad_token_id=151643`、`use_cache=false`。
  缺席时 HF 会从 `config.json` 兜底构造 generation_config；`tools/readout/ro9_gl_attention.py`
  的 `load_model` 又在内存里显式设了 `pad_token_id`。因此对已完成实验多半无实质影响，
  但**这是猜测，不是核实**。
- **保守默认（已执行）**：**不动它**。理由：现在恢复会让本战役后续跑的臂与已完成的臂
  处在**不同的 generation 默认值**下，制造一个新的、更难发现的不可比来源；而且卡 0/卡 1
  当前有他人作业在跑，改共享目录属越权。
- **留给主 agent**：(a) 是否恢复、何时恢复（建议：在没有任何读出/生成作业在跑的窗口
  统一恢复，并记录一条 changelog）；(b) 是否需要抽查一条 `generate()` 路径的臂
  （G1/RO-9 用 greedy `generate`）确认结论不受影响。本实验自身是 prefill-only、
  不调用 `generate()`，**不受该项影响**。

### D-5 · `point∈GT` 的 ±6px 容差在 16×16 网格上几乎无效
- **事实**：canonical square 边长约 512–768 px，一格 = 32–48 px，±6px < 0.2 格。
- **保守默认（已执行）**：仍按 6-09 口径在**全分辨率**掩膜上做 6px 膨胀
  （`config/point_ok16.npz`：每格中心是否落在膨胀后掩膜内），并在 REPORT 注明该容差在本
  分辨率下近乎无效果，故 `point∈GT` 实质等于「峰值格是否是主体格」。

---

## 七、边界（本实验不越界的部分）

- 本实验**只坐实「找主体」**。「跟着指令走」已由 G1/RO9-L/RO9b/G1b 四个实验判死
  （`docs/EXPERIMENT_REGISTRY.md` §3.1），本实验不去救它，结论里不写任何相反表述。
- 普通主体 AUC **不能单独支持任何结论**（RO-X1：一条对所有图相同的 `the main subject`
  也有 AUC 0.907 / AUC_target 0.523）。因此本实验的每个主体分数都必须与
  `auto` / `fixed` 两个零指令对照、以及置换零模型并排读。
- 6-09 的原始结论是**单图定性**；本实验第一次给它上量化，因此**允许出现"定性成立、
  量化不支持"的结果**，那本身就是合格交付，不得为了对齐 6-09 的说法而选择性报数。

---

## 八、D-20 长任务提交纪律执行记录

见 `job.marker`。要点：`rm -f` 日志 → `ps -p 2144280` 实证存活（**全程未用 `pgrep`**）
→ `tail` 见到 `[load] ok · attn_impl=eager` 与 `[20/214] done=80 err=0` 实质输出
→ 才写 marker 与上报。

---

# 九、补件 D（四场多指标翻转检验）· 核实记录 / 假设 / 待决策

## 9.1 实施前三件事

**读了哪些节**（只读引用节）：
- 本实验 `REPORT.md` §九（补件 B 全文，特别是 9.2 中心先验对照表与 C6 建议）；
- `metrics.json → center_prior_baseline` / `sink_location_diagnostic` 全量；
- `diag_raw_colormap.py:197-252`（中心先验的构造与配对代码，**逐字对照后复用**）；
- `analyze_ro9c.py` 的 `field / roc_auc / paired / med / boot_ci` 与 `build_point_ok`（口径复用，不重写）。

**在线核实**：本节**没有引入任何外部事实**（无新 URL / 无新超参 / 无新论文数字），
全部数字来自本仓库已落盘产物的重算，故无需外部核实。唯一沿用的方法学定义是
**边界 F 度量（DAVIS / BSDS 口径：双向容差匹配率的调和平均）**，按标准定义在本文件内自实现
（`boundary_f1`），**未引用任何未核实的外部数字**。

**中心先验定义逐字一致性核对**：
`diag_raw_colormap.py:198-199` = `center = -np.sqrt((yy-(GRID-1)/2.0)**2 + (xx-(GRID-1)/2.0)**2)`；
`diag_four_fields.py::center_variants()["center_prior"]` 用**同一表达式**。
**复算校验**：本节重算得 center 0.8358 / raw L 0.7845 / raw GC 0.7362 / M1 GC 0.6949 /
point 率 raw GC 0.510，与补件 B 与 §4.1 的落盘数字**逐位一致** ⇒ 重算链路正确。

## 9.2 口径决策（都写进 `metric_spec`，未静默拍板）

| # | 决策 | 采用 | 理由 / 保守性 |
|---|---|---|---|
| DD-1 | 阈值化 | **面积匹配 top-k**（k = GT 正格数） | 纯秩次 ⇒ 对四个场值域差异免疫、无自由参数、**不触"s 禁逐图 min-max"红线**。任务卡建议此法，本节采纳 |
| DD-2 | soft-IoU 定义 | `Σmin(pred,m)/Σmax(pred,m)`，pred 二值、m 为软 GT | 若用场值做 soft-IoU 就必须先归一化 ⇒ 撞红线。另报 hard-IoU 作旁证，两者同向 |
| DD-3 | 边界 F1 的上采样 | **最近邻**（非双三次） | 双三次会让"天生平滑"的中心先验拿到更光滑的边界，属于给某一方开小灶；最近邻对四个场**同等**handicap 到 16×16 的分辨力 |
| DD-4 | 3px 的坐标系 | canonical square 原生尺度，逐图换算 `r=max(1,round(3*512/side))` | 与主实验 `point∈GT` 的 ±6px **同一换算式**，不另立口径 |
| DD-5 | 出图着色 | valid-only min-max，**不做 relu**，四场同规则 | 补件 B 的 col4 规则对 raw（非负）等价；但 relu 会把**全负**的中心先验场压成空白。这是为"四场同规则"必须做的**唯一**改动，已在 REPORT §11.1 单列 |
| DD-6 | 二值量的集中量 | `point` 报**率(均值)**，不报中位数 | 中位数在二值量上退化成 0/1（首轮跑出四个场都是 1.000 的无信息表），已修正 |
| DD-7 ⚑ | **新增伪影守卫** | 同支撑同 k 的**随机 top-k 零模型** | 任务卡未要求，但不加就无法把"attention 边界更准"与"attention 更碎"分开。实测它在 3px 口径上（0.0394）**高于中心先验**（0.0327），直接改变了该口径的可采信度 |

## 9.3 ⚑ 待主 agent 决策

### D-7 · 3px 像素级边界 F1 这一列要不要留在判据里
本节实测该口径的**随机零模型高于中心先验**（0.0394 vs 0.0327；`subject_area≥0.2` 时
0.055–0.069 vs 0.023–0.033），即在 16×16 分辨率下它主要在测边界长度而非边界准确度。
- **保守默认（本报告采用）**：**两个边界口径并排报**，判读只采信 `bf1_grid`（1 格容差，
  随机零模型 0.378 显著低于中心先验 0.548），3px 列标注"不可直接采信"。
- 另一种合理做法：直接从判据里删掉像素级 3px 列。**未擅自删**，因为任务卡点名要了这一列。

### D-8 · 判读分支怎么记
任务卡的两分支是二选一，实测是**部分翻转**（重叠类不翻、边界类翻）。
- **保守默认（本报告采用）**：按分支定义（"IoU/边界显著差"）判为**"不翻转"分支**，
  补件 B 结论**原样成立、不软化**；同时把边界类的真实翻转单列成一条限定，不藏。
- 若主 agent 认为"任一形状指标翻转即算翻转"，则该改判为翻转分支 —— **这是判读规则的选择，不是数据分歧**，两种读法用的是同一张表。

### D-9 · `bf1_grid` 的 1 格容差是否偏松
16×16 上 1 格容差 ≈ 32–64 原生 px，相当于"边界位置对到相邻格即算命中"。这是分辨率允许的
最细口径，但也确实宽松。**未加更严的网格级容差**（0 格容差会让所有场都接近 0）。

---

# 十、补件 C / C6（跨臂中心先验配对）· 核实记录 / 假设 / 待决策

## 10.1 实施前三件事

**读了哪些节**：本实验 REPORT §九（含 C6）；`diag_raw_colormap.py:197-252`；
`RO1_selfself_20260803/REPORT.md` §一/§二/§三/§四 + `run_ro1.py:100-230, 481-521`（GT 口径、
网格语义、scache 写入）；`RO3_layerhead_scan_20260803/analyze_ro3.py:138-232`（`Store.load` /
`mask_grid_from_path` / D-0）与 `analyze_diffield.py` 头部（差分场定义）+ `metrics.json →
diff_field_addendum.top30`；`RO9b_readout_fix_20260803/analyze_ro9b.py:200-380`（`Agg` 聚合器）
+ `config/report_tables.md`；`ROX1_clipside_20260803/run_rox1.py:60-120, 400-470`（短语表与 GT）。

**在线核实**：本节**同样没有引入任何外部事实**。所有数字来自本仓库/本机缓存的落盘产物重算，
或来自各实验 `metrics.json` 的**机器可读逐源字段**（不是 REPORT 里的转述）。

**⚑ 主动核查过的三处"报告数字 vs 落盘实物"不一致**（都已在 REPORT §10 里如实反映）：
1. **RO-3 的 0.930 是 `pre/instr` L11H5（r=159）**，而落盘的 scache `ro3-l11-h5` 是
   **`post/instr`（r=1503，差分 AUC 0.899）**。本节两条都重算并分别报，未混为一谈。
2. **RO9b 落盘的 `fields_final.npz` 是 `auc_target` 选优档（AUC 0.728）**，不是报告头条的
   0.920/0.935。后者需重新 OOF 拟合层权重 ⇒ 判"无法配对，需重跑"。
3. **任务卡里的"RO-1 固定无名词短语档"实际是两个不同的东西**：RO-1 自己的 `fixed` 列短语是
   `"object"`（`run_ro1.py:78`），RO-X1 的 `X_deictic` 才是 `"the main subject"`。
   两者结论相反（平 vs 赢），已分开报。

## 10.2 GT 重建的正确性验证（口径 C 的立身之本）

口径 C 用各臂**落盘的逐源 AUC**，中心先验由本轮在同网格重算 —— 只有当我方重建的 GT 与
原实验**逐位相同**时这个配对才成立。验证方式：逐源比对我方 `(m_grid>=0.5).sum()` 与落盘的
`n_pos_grid` / `n_pos`。**RO-1：214/214 全部相等；RO-X1：214/214 全部相等；0 例不符。**
不符者的处置是**剔除该源**（宁缺毋滥），实际未触发。

## 10.3 假设与已当场核实项

| # | 假设 | 处置 |
|---|---|---|
| A-1 | RO-1/RO-X1 的 CLIP 侧输入**不做 `expand2square`**，故无黑边格、无 valid 掩膜 | 已核实 `run_ro1.py::preprocess_image`：只按短边缩放 + 裁到 patch 整数倍，无 pad。REPORT §10.5 据此把 RO-1 那一行标为"不适用" |
| A-2 | RO-3 逐头栈的 `valid16` 与 RO-9c 的 `luma_valid16` 同源同实现 | 两者都走 `ro9_gl_attention.luma_to_grid`。另实测 `reg_a`/`reg_b` 两条指令的 valid16 不一致的源数 = 0 |
| A-3 | RO-3 的 D-0 应当开 | 沿用 RO-3 自己的 `Store.load(d0=True)`（其 scache meta 也写着 `"d0": "norm-MAD3 + interp"`）。**未为了与 RO-9c（D-0 关）对齐而改动**——改动会变成"我方造的臂" |
| A-4 | RO-1 的 32×32 scache 覆盖的是**未 pad 的整幅图像**（`bilinear_to(f,(32,32))` 是对原生 patch 网格的各向异性压缩） | 已核实 `run_ro1.py:490-505`。口径 A 的 `to_canonical16` 据此按 `luma_to_grid` 的逆几何还原 |
| A-5 | `preprocess_image` 裁到 patch 整数倍会丢掉 ≤15px 边缘，而 GT 用的是未裁的全图 | 这是 RO-1 原实验就有的微小不一致（448 短边下 <3.4%），**照原样沿用，未修**；口径 C 完全避开了这个问题（直接用落盘 AUC） |

## 10.4 ⚑ 待主 agent 决策

### D-10 · 口径 A 里 RO-1 的重采样对它**有利**，要不要以口径 C 为准
把 RO-1 的 32×32 场重采样到 canonical 16×16，AUC 从 **0.930 → 0.961**（格变粗、边界误差被抹平）。
- **保守默认（本报告采用）**：**两个口径都报**，并明说口径 C（0.930，原生网格，未经我方任何
  重采样）是保守读数。两个读数都跑赢中心先验，结论不依赖这次调整。
- 若主 agent 要单一数字，建议取口径 C。

### D-11 · RO9b 的 0.920/0.935 要不要补跑
场未落盘，逐头栈仍在 `/var/cache/veradata/ro9b_stacks_20260803`（**零 GPU 可重跑**）。
- **保守默认（本报告采用）**：判"无法配对，需重跑"，**不用 REPORT 数字凑表**。
- ⚑ 补跑时必须标注：该档层权重是**在目标 GT 上有监督 OOF 拟合**的（`crit='auc'`），
  与无监督几何先验的对比属于不同性质，不能与 RO-1/RO-3 那两条并列成一张表。

### D-12 · RO-3 的哪个 GT 列进本节
RO-3 报告的**主列是 `.cgt`（0.840）**，SAM3 是对照列（0.925/0.930）。本节比的是"找主体"，
故**只用 SAM3 列**；`.cgt` 列**未参与**本节任何比较（它测的是指令改动区域，不是主体）。
若主 agent 希望也看 `.cgt` 列 vs 中心先验，需要另跑（`.cgt` 掩膜与 SAM3 的 16×16 IoU 中位仅 0.459）。

### D-13 · 本节改变了补件 B 的哪一句话
补件 B §9.2 写的是"**没有任何一档、任何一个读出跑赢中心先验**"——该句的实测支撑只覆盖
**RO-9c 自己的 6 个读出**。本节把它的适用范围**收紧**为"RO-9c 的 raw/M1 读出全部跑输"，
并新增"RO-1 ClearCLIP 与 RO-3 差分场跑赢"。
**本节未改写任何原实验的判决**（RO-1 的 PROMOTE、RO-3 的晋级、RO9b/RO-9c 的判决一律不动）——
那是主 agent 的事。

## 10.5 零 GPU 确认
补件 C 与补件 D **全程零 GPU**：所有场来自已落盘产物
（`run/stacks`、`/var/cache/veradata/scache/ro1-clearclip-l11`、
`/var/cache/veradata/ro3_stacks_20260803`、`RO9b/config/fields_final.npz`）。
未提交任何后台长任务，故 D-20 四步纪律不适用；两个脚本均为前台 CPU 作业，
日志 `logs/diagD.log` / `logs/crossarm.log`（重定向前已 `rm -f`）。
