# NOTES · optB · B7

范围：`dataset_build/agent_loop/{models,render,graph,candidates,prompts}.py`、
`dataset_build/tools/export_agent_loop_scene_samples.py`、
`dataset_build/tests/test_agent_loop.py`。不跑 run。

## 实现口径与假设

1. **per-intent 强度目标（item 1）**
   - `LOCAL_DELTA_E_TARGETS` 由 `dict[bin, band]` 改为 `dict[intent, dict[bin, band]]`，
     底层是两张梯子 `LOCAL_DELTA_E_LADDERS = {"high", "low"}`，
     `LOCAL_INTENT_LADDER` 做 intent→梯子映射。
   - 已停用的 `warm_cool_split` / `highlight_rescue` 保留 `high` 梯子条目，
     以便历史 audit 行仍可解析（沿用 B6 对 `warm_cool_split` 的处理）。
   - reach 可达推导（`achievable_bins`）在 `LutCatalog.intent_rows` /
     `_row_view(level="local")` 内按 intent 的梯子算；`_row_view` 在
     `level="local"` 且未传 intent 时抛 `CandidateError`（防止"定义了没接线"）。
   - **副作用（需记录）**：同一 preset 可能同时被一个 `low` intent 和一个 `high`
     intent 收录（例如 dL=2.0 且 dSat=20.0 同时满足 luminance_pop 与 sat_boost），
     两者宣称的 `achievable_bins` 不同。原先扁平化 shortlist 以 `preset_id` 去重，
     会让后写入的 intent 复用前一个 intent 的 `achievable_bins`，
     进而让 `local_strength_bin_unreachable` 校验读到错误的行。
     改为以 `(preset_id, tuple(achievable_bins))` 去重：可达集合相同的仍共用一行，
     不同的拆成两行。测试 `test_shortlist_splits_one_preset_across_two_intent_ladders`。

2. **luminance_pop 假白软帽（item 2）**
   - 实现口径选"整条链目标 ΔE 区间下调一档"（任务卡给的两个选项中较简者，
     且比在 bisection 上界 clamp 更可审计：落盘的是目标区间本身，
     而 `target` 已经在 `render` 的 `request_key` 里，缓存键自动区分）。
   - 触发：`intent ∈ {"luminance_pop"}` 且 `subject_headroom.p99_luma > 0.96`
     （`LUMA_SOFT_CAP`）。`p99_luma` 是 B2 已有的、在 global_after 主体区上量的读数。
   - 下调表 `LOCAL_BIN_DOWNSHIFT = {strong→natural, natural→subtle, subtle→subtle}`。
     **`subtle` 没有更低一档**：区间不变，但 `luma_capped` 仍记 true，
     同时落 `effective_strength_bin`，两列一起看即可分辨"帽子生效但无位可降"。
   - 审计字段：`calibrate_local()` 返回值带 `luma_capped` / `effective_strength_bin`；
     leaf 的 `base` 字典带 `luma_capped`（由 `models.luma_capped()` 同一函数算，
     与 render 侧同源）。
   - 与 B2 既有守卫的关系：`intent_offered` 里 `highlight_pressure`
     （p99>0.98 或 near_clip>0.005）**直接不发 luminance_pop 包**；
     新软帽阈值 0.96 落在其下方，覆盖"有 headroom 但已经很亮"的区间。

3. **停用 `highlight_rescue`（item 3）**
   - 加入 `DISABLED_LOCAL_INTENTS`（现为 `{warm_cool_split, highlight_rescue}`），
     `ACTIVE_LOCAL_INTENTS` 6→5。
   - `intent_offered` 的 disabled 分支在 pressure 分支之前，
     因此返回值由 `no_highlight_pressure` / `offered` 统一变为 `intent_disabled`；
     且它不再进入 `build_local_packets` 的循环，连 `local_packet_notes` 行都不产生。
   - `prompts._LOCAL_INTENT_GUIDE` 删去 highlight_rescue 行（与 warm_cool_split 同待遇）。
   - `intent_admits("highlight_rescue", ...)` 的方向域函数保留未动（历史行仍可判定）。
   - 停用依据（数据，用户已批准）：v1/v2 两版 mean 3.07 / 3.25，劣化率 36% / 38%，
     headroom 分组无差异 3.25 = 3.25。

4. **subject band 最小面积门（item 4）**
   - `SUBJECT_BAND_GATE = {"half_area_min": 0.28}`，只作用于 `role="subject"` 且
     `family="band"`；radial / linear / semantic 未动。
   - 判定放在 `build_mask_bank` 的全分辨率 alpha 上（即入库那张），不在
     `_bands()` 的降采样搜索里；不满足的槽位走**现有槽位丢弃机制**
     （`diagnostics` 追加 `{"role","family","slot","reason":"subject_band_min_area",
     "half_area","half_area_min"}` 后 `continue`），不抛异常。
   - `validate_mask_bank` 相应放宽：band 由"精确等于"改为"不超过"预期数量，
     其余 family 仍精确相等；并新增运行时断言——任何入库的 subject band
     `half_area < 0.28` 直接 `CandidateError`（预注册判据必须有运行时断言）。
   - **实测余量**：现有两个 fixture（large / small subject）的 subject band
     `half_area` 落在 0.552–0.637，全部远高于 0.28，该门在当前几何下基本不触发。
     测试用 monkeypatch 把门抬到 0.99 来实跑丢弃路径。

5. **revision（item 5）**
   - `CANDIDATE_SERIALIZATION_REVISION`: `lut-intent-v6.1-optB` → `lut-intent-v6.2-optB`。
   - `prompt_revision_fingerprint()` 额外纳入 `local_intent_ladder` 与
     `local_delta_e_ladders`（梯子直接决定 shortlist 行宣称的 bins）。

## 待决策 / 保守默认（未静默拍板）

- `select_committed_leaves` 的 deviation 现在需要 leaf 上的 `intent`（以及
  `luma_capped`）才能算目标中心。**保守默认取"硬失败"**：leaf 缺 `intent` 或
  intent 不在表里时抛 `ValueError`，不静默回退到某条梯子。影响面：只影响
  B7 之前落盘、且要重新跑 commit 选择的旧 leaf；当前代码路径里 leaf 的
  `intent` 由 `_enrich_local_proposals` + `_validate_local_response` 保证非空。
  若需要对旧 tree 做回溯选择，需要用户决定回退梯子。
- `export_agent_loop_scene_samples.py` 是 intent-free 的场景抽样导出器，
  B7 后没有 intent 可查表。**保守默认读 `high` 梯子的 `strong`**
  （即维持 B6 后该工具的既有数字，5 个 active intent 里 3 个仍用它）。
  若该工具后续要按 intent 分层，需要另派任务卡。
- `luma_capped` 目前只在 leaf 与 `calibrate_local` 返回值上落盘，
  没有单独进 `agent_branch` 的顶层列（沿用 leaf dict 整体落 `result_json` 的现状）。

## 测试

- `dataset_build/tests/test_agent_loop.py`：101 → 106 passed（数量只增不减）。
- 新增：`test_subject_band_minimum_area_gate`（parametrize small/large，共 2）、
  `test_luma_soft_cap_shifts_the_luminance_pop_ladder_down_one_bin`、
  `test_shortlist_splits_one_preset_across_two_intent_ladders`、
  `test_local_calibration_is_wired_to_the_intent_and_the_luma_cap`。
- 改判据重写：`test_local_strength_targets_are_the_b6_recalibrated_bands`
  → `..._b7_per_intent_ladders`；`test_local_reach_bins_follow_the_recalibrated_targets`
  → `..._follow_the_intent_ladder`；`test_intent_conditions_read_the_subject_headroom`；
  `test_build_local_packets_pairs_each_mask_with_intent_row_subsets`;
  `test_strength_contract_and_render_cache`;
  `test_shortlist_row_serialization_is_byte_stable`（revision 串）。
- `dataset_build/tests/test_lut_annotations.py`：7 passed（未受影响）。
- 全目录 `dataset_build/tests` 另有 31 failed / 2 collection error，
  全部落在 `test_canonical_foundation.py` / `test_iaa_batch.py` /
  `test_source_window_and_cgt.py` / `test_winner_margin.py` /
  `test_annot_contract_v4.py` / `test_annotation_interleave.py`，
  原因为 `ModuleNotFoundError: No module named 'construct'` 及 config 样例断言，
  与 agent_loop 无 import 关系，B7 之前既存。
