# NOTES · B1.5 + B2 意图化候选包与亮度余量守卫

规格依据：`docs/DECISIONS_agent_loop_local_intent_optB_20260819.md` §2（C2/C3）、§3（intent 词表 v1）、
§4（B1.5 / B2）。上游：`NOTES_optB_B1.md`（背景角色 mask，已落地）。
代码：`dataset_build/agent_loop/{models,candidates,render,prompts,graph}.py`、
`dataset_build/tests/test_agent_loop.py`。

## 1. 用户追加决定：背景几何池去掉 linear

- `BACKGROUND_FAMILY_COUNTS` 由 `{radial:2, band:2, linear:2}` 改为 `{radial:2, band:2}`；
  `_background_linears`、`BACKGROUND_LINEAR_FALLOFF`、`_linear_axis` 与 `_evaluate_geometry` /
  `_expanded_geometry` 的 linear 分支一并删除（删除后无任何调用方；subject 角色的 `_linears`
  不走 geometry dict，未受影响）。
- 背景槽位数 6 → 4。B1 pilot 的槽位丢弃计数里 `linear:background_coverage_search` 70 /
  `linear:full_resolution_gate` 3 不再产生；过门背景 mask 的 family 分布口径变为 radial/band 两列
  （B1 pilot 页面未重出，`docs/assets/mask_role_pilot_20260819/` 仍是三 family 的旧统计）。
- 单测 `test_background_geometry_pool_has_no_linear_family` 断言池内无 linear。

## 2. 假设与保守默认（未静默拍板项）

- **B1「相对 global_after」的两种读法**。§3 表头写「指纹过滤（相对 global_after）」，任务卡写
  「相对量以 global 选中 preset 的指纹为参照」。两种读法：(a) 用 local LUT 自身的绝对指纹，
  (b) 用 `local指纹 − global指纹` 的差。本次取 (a) 作主口径：local LUT 是叠加在 global_after
  图上渲染的，它对那些像素的改动方向就是它自身的指纹；(b) 会让「global 提亮 5、local 提亮 3」
  被判成压暗。**只有两个配对 intent 额外读 global 指纹**（它们的语义就是「与 global 的方向形成对比」）：
  `zonal_contrast` 要求 global `dL ≥ 0`，`warm_cool_split` 要求 global 的 cast b* `≥ 0`。
  口径待用户确认；改口径只需改 `intent_admits`。
- **B2 配对 intent 的 v1 简化（已按任务卡保守实现）**。读代码确认：一条链只能有一个 local——
  `graph.py` 的 `route_local_batch` 把每个 local proposal 用 `Send` 并行扇出成兄弟叶子，
  全部作用在同一张 `global_after` 上，没有「local 之后再 local」的边。因此 `zonal_contrast` /
  `warm_cool_split` 在 v1 落成**背景 mask 单 mask intent**：
  `INTENT_V1_VARIANTS = {zonal_contrast: background_darken, warm_cool_split: background_cool}`，
  该字段进包、进 proposal、进叶子（`intent_variant`），审计可查。主体侧交给
  `luminance_pop` / `sat_boost` / `hue_shift` 组合覆盖。
- **B3 亮度余量后验门算在全分辨率，不是 4096 采样**。任务卡写「在现有 4096 采样框架算」；
  采样只在 bisect 阶段用（`_sample_metric`），最终 `after` 是全分辨率整图，现成的
  `clip_fraction_new` 也在全分辨率上算。主体面积小的源（0.5%–5%）在 4096 采样里只有几十个
  主体像素，0.5% 的门会被量化噪声支配，所以 `subject_clip_regression` 与 `clip_fraction_new`
  同一行、同一全分辨率口径。改回采样口径只需换一行。
- **B4 applied-alpha 门必须角色化（实现中发现的规格缺口）**。`StrengthCalibrator` 原来对
  family∈{radial, band} 一律要求 applied alpha 的 `effective_alpha_mean > 0.45`、
  `subject_high_coverage ≥ 0.98`；背景角色复用 radial/band 两个 family 名，按定义主体覆盖为 0，
  于是**每一个背景 local 渲染都被这道门打掉**（第一次 5 源 smoke：`applied_alpha_mask_gate`
  15 次）。改为 `_applied_alpha_rejected()`：subject 角色沿用原门；background 角色改判镜像门的
  **避让半边**（`subject_alpha_mean ≤ 0.15`、`subject_high_coverage ≤ 0.02`，数值取自
  `BACKGROUND_ROLE_GATE`）。**覆盖半边（background_alpha_mean ≥ 0.35、half_area ≥ 0.12）在
  强度缩放后不再复检**：alpha×strength 对覆盖是单调下降的，复检等于给背景 local 强加强度下界，
  会和 ΔE 目标打架。此项属预注册判据的新增列，请用户确认。
- **B5 每包 LUT 行数与 bin 配额**。合同只写「每包 ≤4 行」。实现里 `INTENT_BIN_QUOTA = 1`：
  每包先按 subtle/natural/strong 各保 1 行，再按打分补满 4 行；覆盖不到的 bin 记进
  `quota_deficits`（沿用 `local_shortlist_deficits` 这一列，落 branch 行与 tree）。
  原来的按 major 的 `LutCatalog.local_shortlist`（三 bin × 3 行、9 行上限）随 C2 一起删除，
  被 `LutCatalog.intent_rows` 取代。
- **B6 `scorer_rank_offered` 的含义变了**。以前是「模型看到的那张 shortlist 内的名次」；
  现在 shortlist 是多个 intent 包摊平后的并集，同一个 preset 可能同时属于两个包，
  该列取**第一个引入它的包内名次**。`scorer_rank_raw`（打分排序中的原始名次，去簇前）不变。
- **B7 repair 的排除口径改成 preset ID**。row_index 是摊平 shortlist 里的下标，而 repair 会
  重建 shortlist（mask 换了、intent 包换了），旧下标在新表里无意义。改为把首轮用过的 preset
  从候选池里排掉（`repair.excluded_preset_ids`），prompt 里的 `excluded_row_indices` 因此恒为空。
- **B8 intent 未写进 `proposal_audit` 表**。该表是定长列（sqlite + PG 两套 DDL，线上库已建）。
  intent / intent_variant / mask_role 落在 `agent_branch.proposal_json` 与 `result_json`（JSON 列，
  可 `->>` 查询），tree manifest 的 branch 里另存 `local_intent_packets`、`local_packet_notes`、
  `role_packet_note`、`subject_headroom`。要加表列需要一次 PG 迁移，未擅自做。
- **B9 5k 离线标注仍属旧 revision**。`_DIAGNOSE_RULES` 加了两条修正（evidence 左右一律观者视角、
  禁分辨率断言）并新增 `DIAGNOSE_PROMPT_REVISION = "diagnose-v2-viewer-orientation"` 进
  `prompt_revision_fingerprint()`。已冻结的 5,000 条标注不带 prompt revision 字段，
  `validate_source_annotation` 只按 schema + 内容哈希校验，**继续有效**（质检 96.7% 已放行）；
  新 revision 只影响之后新产的标注批。

## 3. 预注册数字（代码常量）

`models.py` `INTENT_FINGERPRINT_GATE`（指纹域，单位同八数指纹）：

| 键 | 值 | 用在哪个 intent |
| --- | --- | --- |
| `dL_positive_min` | 0.5 | luminance_pop |
| `dL_negative_max` | −0.5 | highlight_rescue、zonal_contrast |
| `dL_nonpositive_max` | 0.0 | background_control、zonal_contrast 的 global 参照 |
| `dL_small_abs_max` | 3.0 | sat_boost、hue_shift |
| `dSat_positive_min` | 3.0 | sat_boost |
| `dSat_small_abs_max` | 15.0 | hue_shift |
| `dSat_nonpositive_max` | 0.0 | background_control |
| `cast_mag_mid_min` / `cast_mag_mid_max` | 1.5 / 12.0 | hue_shift |
| `cast_mag_small_max` | 3.0 | background_control |
| `highlight_dL_negative_max` | −0.5 | highlight_rescue |
| `cast_b_cool_max` | −1.0 | warm_cool_split |

`SUBJECT_HEADROOM_GATE`：`near_clip_level = 250/255`、`near_clip_fraction_max = 0.005`、
`p99_luma_max = 0.98`、`sat_mean_max = 0.55`、`subject_clip_regression_max = 0.005`。

阈值取自 4,051 条 closed-v1 LUT 标注的全表分布（选阈值时只看「该 intent 域内还剩多少条」，
不看任何链质量指标）：luminance_pop 2282、zonal_contrast 1460、highlight_rescue 1063、
warm_cool_split 926、background_control 675、hue_shift 337、sat_boost 268（全表 4051）。

## 4. 条件触发与守卫接线

- `render.subject_highlight_headroom(global_after_rgb, subject_alpha)` 出三列：
  `near_clip_fraction`（主体像素 max-channel ≥ 250/255 的占比）、`p99_luma`（同一 max-channel 的
  99 分位）、`subject_saturation_mean`；`highlight_pressure = near_clip_fraction > 0.005 或
  p99_luma > 0.98`。在 `build_local_packet` 节点上对 `global_after` × 主体 mask 测一次，
  整条 branch 共用，落 `subject_headroom`（触发与否都留审计行）。
- candidate 侧：`highlight_pressure` 为真 → `luminance_pop` 不进包（`highlight_headroom_exhausted`）；
  为假 → `highlight_rescue` 不进包（`no_highlight_pressure`）；
  `subject_saturation_mean > 0.55` → `sat_boost` 不进包（`subject_saturation_high`）。
  每一次不进包都写一行 `local_packet_notes`。
- 后验侧：local 渲染在 `StrengthCalibrator._calibrate` 末尾算
  `subject_clip_regression`（final_after 相对 global_after，主体区新增高光剪切像素占比），
  超过 0.005 → `render_record.status = highlight_clip_regression` +
  `RenderError("highlight_clip_regression")`，叶子落 `render:highlight_clip_regression`。
  未超时该列也写进 `metrics`（含 `subject_clip_regression_max`）。

## 5. revision

- `CANDIDATE_SERIALIZATION_REVISION`：`lut-fingerprint-v5.1-optA` → `lut-intent-v6-optB`
- `MASK_SUMMARY_REVISION`：`mask-summary-v3-role`（B1 已 bump，未动）
- 新增 `DIAGNOSE_PROMPT_REVISION = "diagnose-v2-viewer-orientation"`
- 新增进指纹链的键：`local_intents`、`rules.local_intent_guide`
- 新老 campaign 靠 `thread_revision` 隔离（`prompt_revision_fingerprint()` 已变）。

## 6. 5 源端到端验证（campaign `local-v2-b15-smoke`）

配置 `configs/agent_loop.local-v2-b15-smoke.toml`（PG 同库、两 lane 真实 Terra、CPU LUT 渲染、
validator 关闭、landing 关闭）。源：`/home/bc/data/agent_loop/local-v1/sources5k.annotated.jsonl`
按 `sha1(source_id)` 升序、跳过 `subject.mask_area > 0.60` 取前 5 条（复用 5k 冻结标注）。

- pass-0（首跑，B4 未修）：accepted 2 / source_rejected 2 / error 1（error 为
  provider `server_is_overloaded`）；local 叶子 `render:applied_alpha_mask_gate` 15 次，
  全部落在背景角色 mask 上 → 触发 §2-B4 的修复。
- pass-1（B4 修好后重跑，`pass_index=1`）：**accepted 4 / source_rejected 1**。
  被拒源 `src_0469da8eef67330c` 的两个 global 分支里有一个
  `render:strength_target_unreachable`（global 渲染标定失败），reject_reasons =
  `formal_globals_lt_2` + `global_strength_bins_lt_2`。

pass-1 每源统计（branches / intent 包数 / 叶子 / committed）：

| source | 状态 | branches | 包数 | 叶子 | committed |
| --- | --- | --- | --- | --- | --- |
| src_aeae0aacd7c1c68e | accepted | 3 | 17 | 7 | 6 |
| src_3fa2daaf5eb8b9d0 | accepted | 2 | 14 | 4 | 2 |
| src_347cf4dbbbe16d7d | accepted | 3 | 18 | 9 | 5 |
| src_0f68bbe15e69d924 | accepted | 3 | 23 | 4 | 3 |
| src_0469da8eef67330c | source_rejected | 2 | 6 | 3 | 0 |

- 包 intent 分布（78 个包）：hue_shift 23、luminance_pop 16、sat_boost 12、
  background_control 10、zonal_contrast 9、highlight_rescue 7、warm_cool_split 1。
- 进包被挡（`local_packet_notes` 53 行）：`no_row_in_intent_domain` 19、
  `no_highlight_pressure` 16、`subject_saturation_high` 11、`highlight_headroom_exhausted` 7。
- 被接受叶子的 intent（20 个）：background_control 6、zonal_contrast 6、hue_shift 3、
  luminance_pop 3、highlight_rescue 1、sat_boost 1。
- 角色分配：13 个 branch 全部 `fallback = null`，`role_counts` 为 `{subject:2, background:1}`
  12 个、`{subject:1, background:2}` 1 个。
- 亮度余量守卫：12 个进到 local 阶段的 branch 全部写了 `subject_headroom` 审计行，
  `highlight_pressure = true` 4 个 / `false` 8 个（另 1 个 branch 在 global 渲染就失败，无该行）。
  候选侧因此挡掉 `luminance_pop` 7 次、放行 `highlight_rescue` 7 次。
  后验侧：28 条 accepted 的 local 渲染全部带 `subject_clip_regression` 列，最大值 0.00000，
  本批 0 次 `highlight_clip_regression` 拒绝。
- 其余 local 渲染拒绝（pass-1）：subject 角色 `applied_alpha_mask_gate` 6 次
  （hue_shift 3、luminance_pop 2、sat_boost 1，均为冻结的 Mask v2 主体覆盖门），
  background 角色 `strength_target_unreachable` 1 次；**背景角色 0 次 applied_alpha 门拒绝**。
- 旁证（与本次改动无关的既有性质）：`agent_branch.branch_id` 不含 campaign_id，
  `stable_id("global", source_sha256, proposal_id, preset_id, strength_bin)` 在跨 campaign
  复用同一源时会撞主键，UPSERT 只更新 status/result_json，因此 SQL 里这些行仍挂在旧 campaign 名下
  （本批 12 个 global branch 里有 10 个挂在 `local-v2-a6-lane-smoke` / `local-v1-iter1` 名下）。
  tree manifest 是完整的。未改。
