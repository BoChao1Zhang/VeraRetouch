# NOTES · optB · B8

范围：`dataset_build/agent_loop/{models,render,candidates,prompts,graph}.py`、
`dataset_build/tests/test_agent_loop.py`。规格依据
`docs/REQUIREMENTS_local_visibility_lut_selection_20260820.md` 的 P0(R5.1+R4) 与 P1(R1)。
R2/R3 本卡不做。不跑 run。

## 实现口径与假设

### 1. R5.1 过渡强度下限（item 1）

- 选择：**`low` 梯子 `subtle` 的下沿由 2.5 抬到 3.5，带宽保持 1.0**（即 (2.5,3.5)→(3.5,4.5)），
  **不删除该档**。理由：
  - 删档会让 `low` intent（luminance_pop / hue_shift）失去 `subtle` 这个 bin 名，
    而 `LOCAL_PROPOSAL_SCHEMA` 的 enum、`LOCAL_BIN_ORDER`、`INTENT_BIN_QUOTA` 配额、
    以及 B7 假白软帽的 `LOCAL_BIN_DOWNSHIFT`(natural→subtle) 都需要这个落点；
  - 抬下沿只改一个数字，所有既有接线不动。
  - **副作用（需记录）**：`low` 梯子上 `subtle` 与 `natural` 现在是同一条带 (3.5,4.5)。
    模型选 subtle 还是 natural 对 `low` intent 已无区别；`local_target_center`
    `luminance_pop/subtle(capped)` 由 3.0 变为 4.0。
- `LOCAL_VISIBILITY_FLOOR = 3.5` 落在 `models.py`；`assert_local_visibility_floor()`
  在模块导入时执行（任一 active intent 的任一档下沿 < 3.5 直接 ValueError）。
- 校准侧两处运行时判据：
  1. `calibrate_local()` 解析出目标带后先断言 `low >= 3.5`，否则
     `RenderError("local_visibility_floor")`——**这条同时覆盖假白软帽**：软帽下调一档后
     若该档下沿低于 3.5，在渲染前就拒绝（软帽本身保留，未改触发条件与下调表）。
  2. `_calibrate()` 找不到落带候选时，取所有搜索点的 `max(actual)`；
     `stage == "local"` 且该最大值 < 3.5 → 状态 `local_visibility_floor` +
     `RenderError("local_visibility_floor")`；否则仍是 `calibration_failed` +
     `strength_target_unreachable`（够得着 3.5 但没落进目标带 = 普通落带失败，不算可见性）。
     审计 metrics 多两列 `delta_e_reached_max` / `local_visibility_floor`。
- 叶子侧：`graph.calibrate_local_render` 的既有 `except RenderError` 分支把它记成
  `status="render_rejected"`、`reject_reason="render:local_visibility_floor"`。
  **未新增 status 值**（沿用 render_rejected），reason 串区分。

### 2. R4.1 band 几何门（item 2）

- `BAND_GEOMETRY_GATE = {"min_width_short": 0.18, "aspect_max": 4.0}`（models.py，预注册初值）。
- 量法：在**全分辨率 alpha ≥ 0.5 的支撑**上，把像素投影到 band 法线、按 1 px 分箱，
  取连续占用段（slab）。plain band 一段；background 的 complement band 最多两段，
  **逐段量**（这正是 R4 说的「细长窄条」形态：宽 stripe 的补集 = 两条边缘窄条）。
  - `band_min_width = min(slab 宽 / 短边)`，`band_aspect = max((slab 像素数 / slab 宽) / slab 宽)`。
    长度用 `面积/宽` 的平均延展，避免为旋转 band + 画幅裁切写弦长公式。
- 判定放在 `build_mask_bank` 的 role 分支**之前**，故 subject/background 同一套门；
  不满足走既有槽位丢弃通道（`diagnostics` 追加 `reason` = `band_min_width` /
  `band_aspect`，带实测值与门限），不抛异常。
- 通过的 band 记录带 `band_min_width` / `band_aspect` 两列（进 mask bank、进 audit），
  `validate_mask_bank` 对**任何 role 的 band** 做运行时断言（缺列也报错）。
- B7 的 subject band `half_area >= 0.28` 保留不动（只增不减）。
- **实测（fixture, 96×64, 短边 64）**：
  - subject band：宽 0.703–0.953 短边、aspect 1.05–1.44 → 全过。
  - background band（small）：(0.438, 3.05)、(0.641, 1.43) → 全过。
  - background band（large）：(0.219, **4.55**) → **被 `band_aspect` 丢弃**；
    (0.312, 2.65) → 过。即该门在现有 fixture 上非空转也非全杀。
  - 结构性观察（不作结论，仅记录量纲）：对一条横跨画幅的 band，
    aspect ≤ 4 比 width ≥ 0.18 更紧（长度≈画幅边长时 width 需 ≥ 0.25 短边）。
    R4.3 允许 pilot 后校正一次。

### 3. R4.2 背景面积门（item 3）

- `BACKGROUND_ROLE_GATE["half_area_min"] 0.12 → 0.20`。`_covers_background` 与
  `_validate_background_mask` 都读同一常量，无第二处硬编码。
- fixture 上现有 background mask 的 `half_area` 为 0.2904–0.9414，故此改动在 fixture 上不触发。

### 4. R1 mask 条件化可达（item 4）

- `render.MaskReachProbe`：
  - 输入 = `global_after` 图（与 local 校准的输入同一张）、mask alpha、LUT 路径。
  - 采样 = mask `alpha > 0.05` 的支撑里确定性取 ≤1024 像素；种子 =
    `sha256(canonical_json({contract, source_sha256, mask_sha256}))`，
    `np.random.default_rng(seed).choice(...)` 后排序 → 同源同 mask 逐位可复现，
    换 source 或换 mask 就换样本。
  - 度量 = `apply_local_strength(before, full, alpha, strength=1.0)` 后的
    `deltaE_ciede2000`，按 alpha 加权求均值。**这正是 local 校准目标在 strength=1 处的取值**，
    即校准可达的上界，所以能直接和 3.5 下限比。
  - LUT 一律走 `apply_lut_cpu_oracle`（与冻结 reach 探针同一 oracle），
    **与配置的渲染后端无关**：GPU 后端下该读数仍来自 CPU oracle。假设记录在此。
  - 内部缓存：图片一次、每 mask 采样一次、每 (mask, preset) 一次。
- `build_local_packets(..., mask_reach=probe)`：
  - `mask_reach_de < 3.5`（`MASK_REACH_GATE["reach_de_min"]`）的行不进该 mask 的包；
  - 全被砍 → notes 追加 `reason="no_row_reaches_mask"`（带 `mask_reach_de_max`、
    `dropped_rows`、门限），该 (mask,intent) 不产生包；所有包都空时沿用既有
    `no_intent_packet` 降级（graph 里未改）。
  - 存活行带 `mask_reach_de`（6 位小数），并**进扁平 shortlist 的去重键**
    `(preset_id, achievable_bins, mask_reach_de)`——同一 preset 在两个 mask 上读数不同
    就是两行（与 B7 对 achievable_bins 的处理同理，因为该数字会被序列化进 prompt）。
  - 返回值新增 `mask_reach_applied`；`mask_reach=None` 时不测、不过滤、行上无该列
    （既有单测与离线工具可继续用）。
- 接线断言：`graph._intent_packets`（首轮与 repair 两条路径共用）构造 probe 并在
  build 后断言 `mask_reach_applied`，否则 `RuntimeError("mask_reach_gate_not_wired")`。
- Prompt：`shortlist_row_text` 在**行末**追加 `| {mask_reach_de:.2f}`，仅当行带该列
  （global 行不带 → 全局表逐字节不变）；`local_shortlist_text` 头部加
  `LOCAL_SHORTLIST_NOTE` 一句说明该列。
- 审计：`_enrich_local_proposals` 把行上的 `mask_reach_de` 带到 proposal，
  `calibrate_local_render` 把它落到 leaf 的 `base`（与 B7 的 `luma_capped` 同一层）。
- **单对耗时实测**（1024×1536 图、支撑 786432 px、采样 1024、33³ cube）：
  - 单对（图/样本/LUT 均已缓存）：mean 1.382 ms，median 1.373 ms，p95 1.455 ms（n=60）。
  - 每分支首次调用（读图 + 读 alpha + 采样 + 第一对）：144.11 ms。
  - 若该 LUT 的 33³ .cube 需冷解析：51.6 ms/对（生产 bank 走 packed `luts.npz` +
    进程内共享 `_LutLoader` 缓存，不是常态成本）。
  - 包规模上界 3 mask × 5 intent × 4 行，按 (mask,preset) 去重后约 ≤36 对 ≈ 50 ms/分支。

### 5. revision（item 5）

- `CANDIDATE_SERIALIZATION_REVISION`: `lut-intent-v6.2-optB` → `lut-intent-v6.3-optB`。
- `prompt_revision_fingerprint()` 新纳入：`local_shortlist_note`、
  `local_visibility_floor`、`band_geometry_gate`、`mask_reach_gate`、
  `background_role_gate`、`subject_band_gate`。后两个用**函数内惰性 import**
  从 `candidates` 取（`candidates` 从不 import `prompts`，不会成环；
  与 `config.py` 既有写法一致）。

## 待决策 / 保守默认（未静默拍板）

- `low` 梯子 subtle==natural 的重复带：等 R3 的 V 下限落地后梯子会重做，本卡不自造新数字
  （不去发明第三条带宽）。若用户希望 `low` intent 仍保留三档可分辨的强度，需要给新数字。
- mask 可达一律用 CPU oracle，GPU 后端下与实际渲染器不是同一实现。
  保守默认：与冻结 reach 探针保持同一 oracle（可比性优先）。若要求与后端一致，需另派卡。
- `MaskReachProbe` 每次 `_intent_packets` 调用新建一个（缓存生命周期 = 单个 global 分支）。
  同一 source 的多个 global 分支输入图不同，跨分支复用价值低，故未提升到 `AgentServices`。
- 叶子拒绝仍复用 `status="render_rejected"`，只在 `reject_reason` 里区分
  `render:local_visibility_floor`；未新增 leaf status 枚举值（避免动下游统计口径）。
- band 几何门只作用于 `family == "band"`。`linear`（仅 subject 侧还在用的边缘渐变）
  与 `radial` 未加宽度/长宽比门，按规格 R4.1 字面「全角色 band」执行。

## 测试

- `dataset_build/tests/test_agent_loop.py`：106 → **114 passed**（只增不减）。
- 新增：
  - `test_local_visibility_floor_covers_every_active_intent`（常量 + 导入期断言函数）
  - `test_local_calibration_rejects_a_leaf_below_the_visibility_floor`
    （2.807 dE → 拒绝并落 `local_visibility_floor` 审计行；4.098 dE → 通过；
     够得着但没落带 → 仍是 `strength_target_unreachable`；软帽下调到 2.5 档 → 渲染前拒绝）
  - `test_band_geometry_gate_applies_to_both_roles`（parametrize small/large，共 2；
    含 large fixture 上 `band_aspect` 实丢弃、门抬到 0.99 后 `band_min_width` 全丢弃、
    `validate_mask_bank` 运行时断言）
  - `test_band_geometry_reading_measures_each_slab_of_a_complement`
  - `test_mask_reach_probe_is_deterministic_and_reads_the_mask_region`
    （同源两个 probe 逐位一致、换 source 换样本、1024 采样上限、identity LUT 读 0、
     alpha≤0.05 支撑为空读 0、缓存不重算）
  - `test_dead_zone_lut_never_enters_a_local_packet`
    （死区行不进包 + `no_row_reaches_mask` note + 行带 `mask_reach_de` +
     `mask_reach_applied` + 同 preset 两 mask 拆两行）
  - `test_mask_reach_gate_is_wired_into_the_branch_and_the_leaf_audit`
    （整链 run：叶子 `mask_reach_de` ≥ 3.5、identity LUT p0 不再进任何 local proposal；
     以及 `mask_reach_applied=False` 时 `mask_reach_gate_not_wired` 硬失败）
- 改判据重写：`test_local_strength_targets_are_the_b7_per_intent_ladders`
  → `..._b8_floored_per_intent_ladders`；`test_luma_soft_cap_...`（下调后的带）；
  `test_shortlist_row_serialization_is_byte_stable`（revision 串 + 新行末列）；
  `test_repair_and_sibling_prompt_prefixes_are_stable`（local 表含说明句与该列）；
  `test_mask_v2_background_role_gate_and_role_packet`（+ `half_area_min == 0.20`）。
- `dataset_build/tests/test_lut_annotations.py`：7 passed。
- `python -m py_compile`、`python -m ruff check`：通过。
- 全目录 `dataset_build/tests`（忽略两个 collection error 文件）：
  **31 failed / 572 passed**，与 B7 记录的既存失败集合完全一致
  （`test_canonical_foundation` / `test_iaa_batch` / `test_source_window_and_cgt` /
  `test_winner_margin` 等，原因均为 `ModuleNotFoundError: No module named 'construct'`
  与 config 样例断言），与 agent_loop 无 import 关系。
