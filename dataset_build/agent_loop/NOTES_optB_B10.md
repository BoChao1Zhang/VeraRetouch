# NOTES · optB · B10

范围：新增 `dataset_build/agent_loop/segment_fingerprints.py`、
`dataset_build/agent_loop/direction_match.py`、`dataset_build/tests/test_direction_match.py`；
改 `dataset_build/agent_loop/{candidates,config}.py`、`dataset_build/tests/test_agent_loop.py`。
规格依据 `docs/REQUIREMENTS_local_visibility_lut_selection_20260820.md` 的 R6.1 + R6.2。
**只做指纹层，未接线 shortlist / packet / prompt / graph（B11）**。不跑 run，无外部 API。

## 1. R6.1 分段响应指纹（离线派生，零新测量）

### 派生口径（写死在 `segment_fingerprints.py` 模块 docstring 与 `derivation` 报告块）

- 派生版本 `DERIVATION_REVISION = "ramp-band-lumsign-v1"`，schema
  `lut-segment-fingerprint-v1`。loader 见到别的版本串直接报错。
- **ΔL 与中性色偏 `cast_a`/`cast_b`** 取自 `hsl_features.neutral_ramp` 的 5 个中性探点：
  - `shadows` = in 0.15 + 0.30，`mids` = in 0.50，`highlights` = in 0.70 + 0.85（任务卡指定）。
  - `dL` = 段内各点 `L_out - L_in` 的算术平均；`cast_a`/`cast_b` = 段内 `a_out`/`b_out` 的算术平均。
  - 全库 4,051 条 ramp 的 `in` 集合完全一致（(0.15,0.3,0.5,0.7,0.85) × 4051），无缺点位。
  - `mids.dL` 与 `summary.mid_gray_dL` 定义上等价；全库 4,051 条实测差值
    max = 0.1、p99 = 0.1、超过 0.11 的 0 条（`L_in`/`L_out` 落盘只保留 1 位小数）；
    `shadows.dL` / `highlights.dL` 是**两点平均**，与 `summary.shadow_dL` /
    `summary.highlight_dL`（单点读数）不是同一个数，不要互相代入。
- **ΔC 与 Δhue** 取自 8 色相 `bands` 的 `d_sat_pct` / `d_hue_deg`。
  - **近似口径（必须记录）**：bands 本身**没有明度分段**。按任务卡指定，用该 band 的
    `d_lum_pct` 符号近似分配到一个段：`d_lum_pct > +2.0` → highlights，
    `< -2.0` → shadows，其余 → mids（阈值常量 `BAND_SEGMENT_LUM_THRESHOLD = 2.0`，
    预注册初值）。段内 `dC` / `d_hue` = 被分到该段的 band 的 `d_sat_pct` / `d_hue_deg`
    **算术平均**（`d_hue_deg` 是有符号旋转量而非绝对色相，故用普通均值不用圆均值）。
  - **空段回退**：某段一个 band 都没分到时，该段的 `dC`/`d_hue` 退回**全 8 band 均值**。
    每行都带 `bands_per_segment` 记录实际分配数，回退与否可事后判读。
    全库实测：分配总数 highlights 14,556 / shadows 12,706 / mids 5,146（合计 32,408 = 4051×8）；
    出现空段而走回退的 preset 数 highlights 713 / mids 1,599 / shadows 1,251。
  - 该近似只影响 `dC`/`d_hue` 两列；`dL`/`cast_a`/`cast_b` 三列是 ramp 的直接读数，无近似。
- 每段落 5 个字段：`dL, dC, d_hue`（R6.1 的九元组本体）+ `cast_a, cast_b`
  （中性色偏，R6.2 的色偏轴需要每段一个 Lab 方向，故随九元组一起落盘）。
  数值一律 `round(..., 4)`。

### 落盘

- 独立派生文件：`/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v1.jsonl`
  （+ 同名 `.report.json`）。**closed-v1 原文件一字未改**，本卡不动
  `lut_annotations.py` 的 schema/revision。
- 每行：`schema`、`preset_id`、`derivation_revision`、`source_sha256`（closed-v1 整文件 SHA）、
  `hsl_features_sha256`（该条 `hsl_features` 的 canonical JSON SHA）、`bands_per_segment`、`segments`。
- 确定性：行按 `preset_id` 排序，canonical JSON（`sort_keys` + 紧凑分隔符），
  一次性拼 bytes 后原子替换。同一输入连跑两次、写到不同目录，输出 SHA 均为
  `bac4db04e402b0eaefb702502027f7287b99ee50e8fa76b6b48871a9f0774939`。
- CLI：`python -m dataset_build.agent_loop.segment_fingerprints`（默认输入输出即上述路径）。

### 挂载

- `CatalogConfig` 新增可选字段 `segment_fingerprints: Path | None = None`
  （TOML `[catalog] segment_fingerprints = "..."`）。
- `LutRecord` 新增**带默认值 None** 的尾字段 `segment_fingerprint`；未挂载时为 `None`，
  `prompt_view()` / `fingerprint()` / `direction_vector()` 一律不读它 → **序列化面零变化**。
- `LutCatalog.load` 在配置给了路径时读入并逐条挂载；**指纹文件缺任一 catalog preset 直接
  `CandidateError`**（不静默留 None）。另加 `segment_fingerprints_mounted` 与
  `segment_fingerprint_table()`（(N,3,5) ndarray，进程内缓存一次）。
- `thread_revision` 的 `catalog_contract` 是白名单，新字段不入 → **thread_revision 逐字节不变**
  （实测 `local-agent-v1-b1a3fcda0466` 挂载前后相同，单测已固定这条）。
  `sanitized_dict()` 里把该 Path 显式转成 str（否则审计 JSON 不可序列化）。
- **本卡没有改任何 `configs/*.toml`**：生产配置里该键仍未设置，即线上行为逐字节不变。
  打开挂载是 B11 的接线动作。

## 2. R6.2 方向匹配接口（`direction_match.py`，全纯函数）

### `DirectionVector`

五轴（`AXIS_NAMES`）：`cast_a`、`cast_b`（Lab a/b 色偏轴）、`lightness`（明度轴）、
`saturation`（饱和轴）、`contrast`（对比轴）；外加 `mode` 与 `origin`。

- **符号约定（统一，唯一一处口径）**：向量永远记录**观察到 / 被描述到的方向**，不是补救方向。
- `mode="correction"` → `orientation = -1`：指纹与方向**反向**得高分（纠偏）。
- `mode="enhancement"` → `orientation = +1`：指纹与方向**同向**得高分（增强）。

两个构造口径：

1. **诊断文本关键词**：`direction_from_text(text, mode)` /
   `direction_from_diagnosis(diagnosis, mode)`（correction 读 `correction_needs`，
   enhancement 读 `enhancement_opportunities`）。关键词表 `KEYWORD_AXES` 为
   (别名组, 轴, 符号) 三元组，一条规则每次解析最多命中一次。
   - **从 `LutCatalog._score` 抽出**的两组饱和词（`SATURATION_LIFT_WORDS` =
     flat/muted/dull/低饱和/灰；`SATURATION_DROP_WORDS` = oversaturated/too saturated/过饱和）
     与 forbidden 别名表 `FORBIDDEN_CAPTION_ALIASES` 现在**只存在于 `direction_match.py`**，
     `_score` 改为 import 使用，**算术一字未动**（既有 shortlist 单测全过，可证打分未漂移）。
   - 新增的四组词（暖/冷、洋红/绿、对比不足/对比过强、欠曝/过曝）是本卡新写的**预注册词表**，
     不参与 `_score`，只喂 `DirectionVector`。
2. **两图实测**：`measure_direction(img_a, img_b, mask=None, *, mode="correction")`
   返回 a→b 的方向（B11 用 source vs global_after 测残差）。
   - 采样 `MEASURE_SAMPLE_PIXELS = 4096`（与冻结 reach 探针同一预算），
     **确定性且无种子**：在合格像素的扁平下标数组上取 `linspace` 等距点，
     同输入逐位可复现（单测断言两次调用完全相等）。
   - `mask` 可选：只在 `alpha > MASK_SUPPORT_MIN (0.05)` 的支撑里采样，且每个样本按 alpha 加权
     （权重归一化）。
   - `lightness` / `cast_a` / `cast_b` = Lab ΔL/Δa/Δb 的加权均值；
     `saturation` = HSV S 的加权均值差 ×100（**与 bands 的 `d_sat_pct` 同单位**，
     不是 Lab chroma，故实测侧与指纹侧的饱和轴可比）。
   - `contrast` = 高光像素的加权 ΔL 均值 − 阴影像素的加权 ΔL 均值；
     段划分按 **img_a 的 L\***，边界 `SEGMENT_L_BOUNDS = (43.0, 63.0)`——
     这两个数就是 ramp 相邻探点 L_in 的中点（≈32.5/53.4/72.8），与 R6.1 的分段同源。
     任一端为空（如全中灰图）→ `contrast = 0.0`。
   - 配套 `measure_tonal_weights(img, mask=None)` 给出该 mask 的
     shadows/mids/highlights 权重（和为 1），即下面打分要的 `tonal_weights`。

### `direction_match_score(direction, segment_fingerprint, tonal_weights)`

- 先把 (3 段 × 5 字段) 指纹按 `tonal_weights` 压到五轴（`fingerprint_axes`）：
  `cast_a/cast_b/lightness(=dL)/saturation(=dC)` 都是三段的加权和；
  **`contrast` 是跨段之差**（`highlights.dL - shadows.dL`），无法写成段的加权和，
  故额外乘一个**相关度因子 `2*sqrt(w_shadows*w_highlights)`**：mask 均匀跨两端时 = 1，
  mask 全落在单一段时 = 0（即「这块 mask 没有对比可读」）。同一因子也乘到查询向量的对比轴上。
- 每轴先除以预注册尺度 `AXIS_SCALES`（cast/明度/对比 5.0 Lab 单位，饱和 10.0 百分点）
  再乘 `AXIS_WEIGHTS`（v1 全 1.0），构成一个对角度量。
- 分值 = 该度量下的**余弦** ∈ [-1, 1]，再乘 `direction.orientation`。
  查询或指纹任一为零向量 → 0.0。
- 向量化版 `direction_match_scores(direction, SegmentFingerprintTable, tonal_weights)`
  一次算全库；`direction_match_score` 是它的 N=1 包装（单测断言两者逐值相等）。
  `rank_by_direction(...)` 给 top-k，同分按 `preset_id` 破平（确定性）。

### 实测耗时（本机，4,051 条真实指纹）

- `direction_match_scores` 全库一次：mean **0.377 ms** / median 0.374 / p95 0.405（n=50）。
  判据 <10 ms 达标（余量 ~26×）。
- `rank_by_direction(top-50)`：mean 3.43 ms（含 Python 排序）。
- 一次性成本：读派生文件 83.6 ms，建 (4051,3,5) 表 23.5–27.9 ms（`LutCatalog` 内缓存，只建一次）。
- 派生文件整库重建：999 ms。

## 待决策 / 保守默认（未静默拍板）

- **band → 段的分配是近似**（bands 无明度分段，只能按 `d_lum_pct` 符号猜）。阈值 2.0 与
  「空段回退全 band 均值」都是本卡预注册的保守初值，未做任何调参。若 B11/R6.3 需要
  真正分段的色相响应，只能重测 LUT（本卡限定「零新测量」，故未做）。
- `d_hue`（Δhue）落盘了但**不进 R6.2 的五轴打分**：R6.2 明确的轴是色偏/对比/饱和/明度，
  把 band 的色相旋转折进色偏轴属于自造口径，未做。需要的话给口径再加。
- `AXIS_SCALES` / `AXIS_WEIGHTS` 是预注册初值（cast、明度、对比同为 5.0 Lab 单位，
  饱和 10 百分点，轴权重全 1）。没有任何数据用来标定它们，故未标定。
- 对比轴的相关度因子取几何均值 `2*sqrt(w_sh*w_hi)`，是本卡为「mask 落在单一色调段时对比轴
  应当失效」写的最简式子，非规格给定数字。
- `contrast` 的实测段界 (43, 63) 由 ramp 探点中点推得，与 R6.1 的 ramp 分段同源；
  若日后 ramp 探点变了，这两个数要跟着变（同一模块常量，只有一处）。
- 生产 TOML 未开挂载（见上）。B11 接线时需要：`[catalog] segment_fingerprints = ...` +
  一条运行时断言（`catalog.segment_fingerprints_mounted` 为假直接失败），
  否则又是「定义了没接线」。本卡没有权限替 B11 决定断言落点，只在此登记。

## 测试

- 新增 `dataset_build/tests/test_direction_match.py`：**19 passed**。
  - R6.1：派生公式逐字段断言（ramp 平均 / band 按 `d_lum` 符号分配 / 空段回退）；
    截断 ramp 与空 bands 报错；建两次 SHA 一致 + 载回一致 + 行按 preset_id 排序 +
    `source_sha256` 全行同值；异版本 `derivation_revision` 被 loader 拒绝。
  - 挂载：未配置时 `segment_fingerprint is None` 且 `segment_fingerprint_table()` 报错；
    配置后挂上且 `prompt_view()`/`fingerprint()` 与未挂载逐值相同（向后兼容）；
    指纹文件漏一个 preset → `CandidateError`。
  - R6.2 方向语义：暖指纹 vs「整体偏黄」correction 得 −1.0、冷指纹得 +1.0；
    enhancement 模式符号整体翻转；正交轴得 0；零向量得 0。
  - R6.2 对比轴：shadows −6 / highlights +6 的指纹对「对比不足」correction 得 +1.0，
    反向指纹得 −1.0；mask 全在 mids 时得 0.0；偏斜权重仍 >0。
  - R6.2 色调权重：同一指纹在 `{shadows:1}` 与 `{highlights:1}` 下符号相反；
    非法权重（全零 / 负 / 未知段名）报错。
  - 单条与批量打分逐值相等；排序确定性（同分按 preset_id）。
  - 4,051 条合成表打分 5 次取 min **< 10 ms** 的运行时断言。
  - 实测：a→b 色偏 / 明度读数、mask 加权（mask 内读数 ≈ 全幅 2×、mask 外为零向量）、
    形状不匹配报错、对比轴按 source L\* 分段、`measure_tonal_weights` 和为 1、
    空 mask 报错；以及「实测残差 → 排序」的端到端一条。
- `dataset_build/tests/test_agent_loop.py`：114 → **115 passed**（新增
  `test_mounting_segment_fingerprints_leaves_the_thread_revision_alone`）。
- `dataset_build/tests/test_lut_annotations.py`：7 passed。
- 三个文件合计 **141 passed**。
- 全目录 `dataset_build/tests`（忽略两个既有 collection error 文件）：
  **31 failed / 591 passed**——失败集合与 B8 记录的 31 failed / 572 passed 完全一致
  （均为 `ModuleNotFoundError: No module named 'construct'` 与 config 样例断言，
  与 agent_loop 无 import 关系），passed 增量 19 = 本卡新增用例。
- `python -m ruff check dataset_build/agent_loop dataset_build/tests/...`：All checks passed。
