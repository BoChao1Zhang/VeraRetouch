# NOTES · B6 意图问卷修正落地（revision `lut-intent-v6.1-optB`）

上游：`NOTES_optB_B15.md`（B1.5 + B2）、`docs/DECISIONS_agent_loop_local_intent_optB_20260819.md`。
输入：200 条 intent 问卷标注 + 独立审阅报告（用户已批准四项参数/判据变更 + 三项 blocker 处置 +
三项 warning）。代码：`dataset_build/agent_loop/{models,candidates,render,prompts,graph,persistence}.py`、
`dataset_build/tests/test_agent_loop.py`。**本次不跑 run。**

## 1. 逐项落点

| # | 项 | 落点 |
| --- | --- | --- |
| 1 | 禁用 `warm_cool_split` | `models.DISABLED_LOCAL_INTENTS = {"warm_cool_split"}`、`ACTIVE_LOCAL_INTENTS`；`build_local_packets` 只遍历 active；`intent_offered` 对禁用 intent 返回 `(False, "intent_disabled")`；prompt 的 `_LOCAL_INTENT_GUIDE` 删掉该行 |
| 2 | sibling 角色配比 2+1 | `candidates.ROLE_PACKET_TARGET = {"subject": 2, "background": 1}`；`allocate_role_packets` 先求精确 2:1，不可行再退 ≥1/≥1，再退 subject-only（`fallback="background_infeasible"`）；`role_packet_note` 新增 `role_target` / `role_target_met` |
| 2b | commit ≥1 subject 叶 | `select_committed_leaves` 的锚定 pair 候选先过滤到「含 ≥1 个 `mask_role=="subject"` 叶」；无此类 pair 时不过滤（豁免） |
| 3 | local 强度目标上移 | `LOCAL_DELTA_E_TARGETS` 见 §2；local reach 可达（`d_full ≥ low`）在 `_row_view` / `intent_rows` 里直接读该常量，自动跟随 |
| 4 | highlight_rescue 收紧 | `INTENT_FINGERPRINT_GATE["highlight_dL_negative_max"]`：−0.5 → −2.0 |
| 5 | 背景 applied-alpha 门 | 见 §3 |
| 6 | intent 域计数可复现 | 见 §4 |
| 7 | 跨 campaign 覆写 | 见 §5 |
| 8 | C3 供给侧 | `_validate_local_response` 的多样性下限改为 `min(2, 本 branch 供给的 intent 数)`；`intent_supply` 作为审计列进 `GlobalBranchState`、`agent_branch.result_json` 与 tree manifest 的 branch 行 |
| 9 | repair 载荷去 preset ID | `graph.local_repair` 的 `repair` 字典删掉 `excluded_preset_ids`；排除仍在检索侧生效（`_intent_packets(..., exclude=used_presets)`） |
| 10 | 包契约进 revision | `LOCAL_PACKET_ROW_LIMIT` 与 `INTENT_BIN_QUOTA` 移到 `models.py`，以 `local_packet_contract` 键进 `prompt_revision_fingerprint()`；`INTENT_BIN_QUOTA` 从 `candidates` 继续再导出 |

## 2. 新旧判据数字对照

| 判据 | 旧值 | 新值 |
| --- | --- | --- |
| `LOCAL_DELTA_E_TARGETS["subtle"]` | [2.5, 3.5) | [3.5, 4.5) |
| `LOCAL_DELTA_E_TARGETS["natural"]` | [3.5, 4.5) | [4.5, 5.5) |
| `LOCAL_DELTA_E_TARGETS["strong"]` | [4.5, 5.5] | [5.5, 6.5] |
| local reach 可达门槛（`d_full ≥ low`） | 2.5 / 3.5 / 4.5 | 3.5 / 4.5 / 5.5 |
| `INTENT_FINGERPRINT_GATE["highlight_dL_negative_max"]` | −0.5 | −2.0 |
| 背景 local 渲染的 applied-alpha 门 | `subject_alpha_mean ≤ 0.15` 且 `subject_high_coverage ≤ 0.02`（复检） | `applied_background_alpha_mean_min = 0.18`（新增预注册列） |
| sibling 角色要求 | ≥1 subject 且 ≥1 background | 目标 2 subject + 1 background |
| committed 集合角色要求 | 无 | ≥1 subject 角色叶（无 subject 叶时豁免） |
| local sibling intent 多样性下限 | 恒为 2 | `min(2, intent_supply)` |
| candidate 包内 intent 数（角色匹配后可选） | 7 | 6（`warm_cool_split` 禁用） |
| `CANDIDATE_SERIALIZATION_REVISION` | `lut-intent-v6-optB` | `lut-intent-v6.1-optB` |

其余预注册数字未动：`SUBJECT_HEADROOM_GATE`（`near_clip_level = 250/255`、
`near_clip_fraction_max = 0.005`、`p99_luma_max = 0.98`、`sat_mean_max = 0.55`、
`subject_clip_regression_max = 0.005`）、`BACKGROUND_ROLE_GATE` 的建库四列
（0.15 / 0.02 / 0.35 / 0.12）、`GLOBAL_DELTA_E_TARGETS`、`LOCAL_PACKET_ROW_LIMIT = 4`、
`INTENT_BIN_QUOTA = 1`。

## 3. 背景 applied-alpha 门（blocker 5）

- 删除的复检：`_applied_alpha_rejected()` 里背景分支原来复检
  `subject_alpha_mean > 0.15 or subject_high_coverage > 0.02`。
  applied alpha = `clip(alpha × strength, 0, 1)`，`strength ∈ [0, 1]`，所以这两列对 strength
  单调不增；建库门已在原始 alpha 上把它们卡在 0.15 / 0.02 以内，复检因此恒真（永远判过）。
  **避让侧改为信任建库门，不复检**，理由即上述单调性。
- 新增的门：**背景可见性下限**，`applied_background_alpha_mean_min = 0.18`（预注册数字），
  读 `_applied_alpha_metrics` 的 `background_alpha_mean`（非主体像素上 applied alpha 的均值）。
  低于该值 → `render_record.status = applied_alpha_rejected` + `RenderError("applied_alpha_mask_gate")`，
  与主体侧同一拒绝码。
- 与建库门的关系：建库门要求原始 `background_alpha_mean ≥ 0.35`，因此新门等价于
  `strength ≥ 0.18 / background_alpha_mean`（`≤ 0.514`）。ΔE 目标上移（§2）会把 strength 推高，
  两者同向。
- 数字来源：0.18 由用户批准，预注册，本次无数据拟合。

## 4. intent 指纹域计数（blocker 6，阈值与计数同一次运行产出）

口径（`>=` / `<=` 全部走代码里的 `intent_admits`，不复述阈值）：

- 表：`/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl`，
  筛 `ok == true` 且 `style_major` / `style_minor` 非空 → **4051 行**（不接 `features.jsonl`
  的可渲染过滤，故与 `LutCatalog.load` 的行集可能不同）。
- 每行用 `LutRecord(hsl_features=...).fingerprint()` 取八数指纹。
- 两个配对 intent 额外读 global 指纹；本次统计取中性 global 参照
  `{"dL": 0.0, "cast_hue": 0.0, "cast_mag": 0.0}`，使 global 半边条件恒真，
  统计的是 local 半边的域大小。
- 运行日期 2026-08-20，常量即本次落盘的 `INTENT_FINGERPRINT_GATE`
  （含 `highlight_dL_negative_max = −2.0`）。

| intent | 域内条数 |
| --- | --- |
| luminance_pop | 2304 |
| zonal_contrast | 1486 |
| highlight_rescue | 942 |
| ~~warm_cool_split~~（已禁用，仅存档） | 925 |
| background_control | 639 |
| hue_shift | 338 |
| sat_boost | 273 |
| 全表 | 4051 |

同一次运行的 `highlight_rescue` 阈值敏感度：`highlight_dL ≤ −0.5` → **1093**；
`≤ −2.0` → **942**。

`NOTES_optB_B15.md` §3 的旧计数（luminance_pop 2282 / zonal_contrast 1460 /
highlight_rescue 1063 / warm_cool_split 926 / background_control 675 / hue_shift 337 /
sat_boost 268）**未记录行集与 global 参照口径，本次无法复现**，以上表为准。

复现脚本（逐字）：

```python
import json
from dataset_build.agent_loop.candidates import LutRecord, intent_admits
from dataset_build.agent_loop.models import LOCAL_INTENTS

path = "/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl"
records = []
with open(path, encoding="utf-8") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        if not str(row.get("style_major") or "").strip():
            continue
        if not str(row.get("style_minor") or "").strip():
            continue
        records.append(LutRecord(
            preset_id="x", path="", format="lut", name="", style_major="m",
            style_minor="n", scene_affinity=(), de_med=0.0, caption="",
            per_probe={}, hsl_features=dict(row.get("hsl_features") or {})))
neutral = {"dL": 0.0, "cast_hue": 0.0, "cast_mag": 0.0}
for intent in LOCAL_INTENTS:
    print(intent, sum(1 for r in records
                      if intent_admits(intent, r.fingerprint(), neutral)))
```

## 5. 主键迁移（blocker 7）

选型：**改主键**为 `(campaign_id, branch_id)`，不改 `branch_id` 生成规则。侵入面对比——
改 `stable_id` 会改变 `agent_branch` / `proposal_audit` / `render_record.branch_id` /
`validation_record.branch_id` / tree manifest / 三个导出与问卷工具里的全部 ID，且已落盘的
旧 ID 与新 ID 不可互查；改主键只碰两张表的 DDL 与两条 UPSERT 的冲突目标。

DDL（sqlite 与 PG 同构，两处列顺序不变，主键改成表级约束）：

```sql
-- agent_branch / proposal_audit 同款
branch_id TEXT NOT NULL, campaign_id TEXT NOT NULL, ...,
PRIMARY KEY(campaign_id, branch_id)
```

UPSERT 冲突目标：`ON CONFLICT(branch_id)` → `ON CONFLICT(campaign_id,branch_id)`（四处）。

`setup()` 里的一次性幂等迁移：

- **PostgreSQL**（`PostgresAuditStore._migrate_campaign_primary_key`）：读
  `pg_constraint`（`contype='p'`、`conrelid = to_regclass(<table>)`）取现有主键列名集合；
  等于 `{campaign_id, branch_id}` 则跳过，否则
  `ALTER TABLE <t> DROP CONSTRAINT "<pkname>"` + `ALTER TABLE <t> ADD PRIMARY KEY (campaign_id, branch_id)`。
  （PG 的 `ADD PRIMARY KEY` 自动补 `NOT NULL`。）
- **SQLite**（`SQLiteAuditStore._migrate_campaign_primary_key`）：`PRAGMA table_info` 取 pk 列集合，
  不等则建 `<t>_pkmig`（DDL 从 `_SQLITE_SCHEMA` 里按表名切出来）→
  `INSERT INTO <t>_pkmig(<原列>) SELECT <原列> FROM <t>` → `DROP TABLE <t>` →
  `ALTER TABLE <t>_pkmig RENAME TO <t>`；重建表会连带删掉挂在表上的索引，故 `setup()`
  在迁移后再跑一遍 `_SQLITE_SCHEMA`（全部 `IF NOT EXISTS`）把 `idx_branch_source` 补回来。
- 幂等：连跑两次 `setup()` 第二次全跳过（单测覆盖）。
- 数据：已有快照 `/home/bc/data/agent_loop/local-v1/pg_snapshot_before_b_fix_20260820_0050.sql.gz`，
  本次**未做任何数据恢复**，只做 schema 迁移；被旧主键覆写掉的历史行不会因迁移回来。

## 6. 假设与保守默认（未静默拍板）

- **B6-a `warm_cool_split` 的保留位置**。任务卡写「`INTENT_ROLES` 中移除或标禁用」。取
  **标禁用**：`LOCAL_INTENTS` 与 `INTENT_ROLES` 都保留该键，只加 `DISABLED_LOCAL_INTENTS`。
  理由：`INTENT_V1_VARIANTS`、旧 branch 行的 `intent_variant = background_cool`、
  `_enrich_local_proposals` 的 variants 映射都按键查；移除会让旧审计行解析不出角色。
  代价：`prompt_revision_fingerprint()` 里 `local_intents` 仍列 7 项，另加
  `active_local_intents` 列 6 项。
- **B6-b 禁用 intent 不写 `local_packet_notes`**。`build_local_packets` 直接遍历
  `ACTIVE_LOCAL_INTENTS`，因此不会为 `warm_cool_split` 逐 mask 逐 branch 写
  `intent_disabled` 审计行（否则每个背景 mask 每个 branch 多一行常量噪声）。
  `intent_offered("warm_cool_split", ...)` 仍返回 `(False, "intent_disabled")`，直调可查。
- **B6-c commit ≥1 subject 叶的执行点**。放在**锚定 pair 的候选过滤**上（锚定对必进
  committed 集合），不改贪心补选的打分。若不存在「含 subject 叶」的合法 pair
  （pair 还要满足 global branch 不同 + global bin 不同），则不过滤 → 豁免。
  「背景不可行源」在这个实现下自动落进豁免分支（那种源根本没有背景叶）。
- **B6-d 2:1 配比在 `packet_size < 3` 时不适用**。`allocate_role_packets(packet_size=1或2)`
  拿不到 3 个槽位，直接走 ≥1/≥1 或 subject-only；`role_target_met = false`。
  线上 `max_local_proposals = 3`，正常路径不触发。
- **B6-e `intent_supply` 落在 JSON 列**。沿用 B1.5-B8 的结论：`proposal_audit` 是定长列，
  加列要迁移；`intent_supply` 写进 `agent_branch.result_json` 与 tree manifest 的 branch 行
  （`->>'intent_supply'` 可查），未加 SQL 列。
- **B6-f 强度目标上移对 reach 的连带影响未跑数**。`d_full ≥ low` 的门槛整体抬了 1.0，
  可达 `subtle` 的 preset 数会减少；本次不跑 run，无实测数字。

## 7. 判据

- `pytest dataset_build/tests/test_agent_loop.py dataset_build/tests/test_lut_annotations.py -q`：
  **108 passed**（改动前 100 passed；新增 8 个用例，无删除）。
- `python -m py_compile dataset_build/agent_loop/*.py dataset_build/tests/test_agent_loop.py
  dataset_build/tools/*.py`：通过。
- `python -m ruff check dataset_build/agent_loop/ dataset_build/tests/`：All checks passed。

新增用例：`test_packet_row_budget_enters_prompt_revision_chain[2]`、
`test_local_strength_targets_are_the_b6_recalibrated_bands`、
`test_local_reach_bins_follow_the_recalibrated_targets`、
`test_committed_set_must_carry_a_subject_role_leaf`、
`test_single_intent_supply_relaxes_the_sibling_diversity_rule`、
`test_repair_payload_never_carries_preset_ids`、
`test_agent_branch_and_proposal_audit_are_campaign_scoped`。

改判据数字后按新数字重写的旧用例：`test_shortlist_row_serialization_is_byte_stable`（revision 串）、
`test_intent_conditions_read_the_subject_headroom`（禁用 intent）、
`test_build_local_packets_pairs_each_mask_with_intent_row_subsets`（包内 intent 集合）、
`test_applied_alpha_gate_mirrors_for_the_background_role`（拒绝样例改为可见性下限）、
`test_shortlist_excludes_unreachable_presets_and_exposes_reachable_bins` 与
`test_intent_rows_fill_one_per_bin_and_report_shortfalls`（`d_full` 夹具随新 bin 下界上调）、
`test_mask_v2_background_role_gate_and_role_packet`（2:1 配比 + 退化路径）。

## 8. revision

- `CANDIDATE_SERIALIZATION_REVISION`：`lut-intent-v6-optB` → **`lut-intent-v6.1-optB`**
- `prompt_revision_fingerprint()` 新增键：`active_local_intents`、
  `local_packet_contract = {"row_limit": 4, "bin_quota": 1}`；`_LOCAL_INTENT_GUIDE` 文本变短一行
- `PROMPT_REVISION` / `MASK_SUMMARY_REVISION` / `DIAGNOSE_PROMPT_REVISION` 未动
- 新老 campaign 仍靠 `thread_revision` 隔离（指纹已变）
