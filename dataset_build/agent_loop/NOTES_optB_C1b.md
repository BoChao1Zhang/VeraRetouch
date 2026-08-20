# C1b — C1 审查 12 个 BLOCKER 修复（2026-08-20）

任务卡：C1b。范围＝只修 C1 审查列出的 12 个 BLOCKER + 给 fingerprint registry 补结构性测试。
W/N 级问题不修。不跑 run。

---

## 1. `prompt_revision_fingerprint()` 纳入 11 个缺失常量 + 结构性测试

`dataset_build/agent_loop/prompts.py`

- 函数拆成两层：`prompt_registry() -> dict`（可被测试直接读）与
  `prompt_revision_fingerprint() -> str`（对 registry 做 canonical-JSON SHA-256）。
  两者都进 `__all__`。
- 新纳入 11 个键（注释里逐条写了「为什么它决定模型看到什么」）：

  | registry 键 | 常量 | 来源模块 |
  |---|---|---|
  | `intent_fingerprint_gate` | `INTENT_FINGERPRINT_GATE` | models |
  | `global_delta_e_targets` | `GLOBAL_DELTA_E_TARGETS` | models |
  | `subject_headroom_gate` | `SUBJECT_HEADROOM_GATE` | models |
  | `role_packet_target` | `ROLE_PACKET_TARGET` | candidates |
  | `background_family_counts` | `BACKGROUND_FAMILY_COUNTS` | candidates |
  | `global_bin_quota` | `GLOBAL_BIN_QUOTA` | candidates |
  | `axis_scales` | `AXIS_SCALES` | direction_match |
  | `axis_weights` | `AXIS_WEIGHTS` | direction_match |
  | `keyword_axes` | `KEYWORD_AXES` | direction_match |
  | `measure_sample_pixels` | `MEASURE_SAMPLE_PIXELS` | direction_match |
  | `band_segment_lum_threshold` | `BAND_SEGMENT_LUM_THRESHOLD` | segment_fingerprints |

- `direction_match` 与 `segment_fingerprints` 走模块顶层 import（两者都不 import `prompts`，
  无环）；`candidates` 三个常量沿用原有的函数内局部 import（`candidates` → `prompts` 会成环）。
- 结构性测试 `PROMPT_REGISTRY_KEYS`（显式 41 键元组）+
  `test_prompt_registry_key_set_is_the_explicit_frozen_list`：断言
  `sorted(prompt_registry()) == sorted(PROMPT_REGISTRY_KEYS)` 且键数 == 41。
  **新增或删除任何 registry 键都必须同时改这个清单和这个数字**，不可能悄悄漏掉。
  另配：11 个常量各一条 monkeypatch 参数化用例（改常量 → 指纹必变）、
  11 个键各一条「值非空且可 JSON 序列化」用例、一条「指纹 == registry 的 SHA-256」用例。

**指纹变更（语义等价 bump）**：
- 旧：`d770156b90edbafca19f9eba8669419d08c1ec12598bef19db5481a660faf04b`（B11）
- 新：`ca45918582823317c0be51896c9369e422eb086c85f10af793000651b5fde60e`

这次纳入没有改任何常量的取值，只是把已经在起作用的常量登记进 registry，因此是**语义等价的
revision bump**：模型看到的 prompt / 候选行完全没变，但 `thread_revision` 会变，新老 campaign
按既有规则靠 `thread_revision` 隔离。

## 2. `thread_revision` 的 `catalog_contract` 加 `segment_fingerprints` + 启动断言

`dataset_build/agent_loop/config.py`、`runtime.py`

- `catalog_contract` 新增 `segment_fingerprints`（配置路径，未挂载为 `None`）。
- `runtime.require_frozen_segment_fingerprints(catalog, *, expected_sha256=SEGMENT_FINGERPRINT_TABLE_SHA256)`：
  未挂载 → `ConfigError`；挂载的表 SHA ≠ 常量 → `ConfigError`。在 `create_services()` 里
  `LutCatalog.load()` 之后立刻调用（生产唯一装配入口）。
- `create_services()` 新增 kwarg `expected_segment_fingerprint_sha256`，默认即冻结常量。
  **只有测试**会传别的值——fixture 表天然不可能等于生产表 SHA。这是一处显式、可见的测试旁路，
  不是静默跳过。
- 过时不变式测试改写：
  `test_mounting_segment_fingerprints_leaves_the_thread_revision_alone`
  → `test_mounted_segment_fingerprint_path_enters_the_thread_revision`
  （B10 时该表不进请求面，B11 之后它是在线 local 检索的整个候选宇宙，语义已反转）。
  新增 `test_startup_rejects_a_segment_fingerprint_table_that_is_not_the_frozen_one`。

## 3/4/5. 三个问卷工具带出 `winner_confidence`

共享实现放在 `dataset_build/tools/export_agent_loop_review.py`
（`chain_quality` / `intent_quality` 都已 import 该模块；`recheck` 也新增这一条 import）：

- `winner_confidence_counts(values)`：按取值计数，`None`/空串归入 `unknown`。
- `winner_confidence_warning(counts, *, filtered)`：low+unknown 为 0 时返回 `""`，否则返回一行
  `WARNING winner_confidence: N/M ... filtering is ON/OFF ...`。
- `print_winner_confidence_warning(...)`：打到 **stderr**，并把该行返回以便写进 JSON。

三个工具各自：

| 工具 | 抽样查询带出 | item_key | 汇总 |
|---|---|---|---|
| `chain_quality_questionnaire` | `result.winner_confidence` 进 chain dict | `items[*].winner_confidence` | build stats 顶部 3 个键 + snapshot 里 `chain_population_winner_confidence`；analyze 也算被评分子集 |
| `intent_quality_questionnaire` | 同上 | 同上 | 同上（含 `chain_population_winner_confidence`） |
| `recheck_questionnaire` | 从来源轮次 item_key 继承 | `CARRIED_FIELDS` 新增 `winner_confidence` | build/analyze stats 同上 |

**默认不过滤**：`WINNER_CONFIDENCE_FILTERED = False`（三个模块各自声明，测试断言三者都是 False）。
理由：当前战役 committed leaf 全是 `low`（`winner_confidence` 只在 validator 判 passed 时才是
`normal`），过滤会得到空集。因此警示行是唯一的护栏，必须打印。
C1b 之前建的 recheck 轮次 item_key 里没有这个键，`CARRIED_FIELDS` 取到 `None`，计数器报 `unknown`
（不是伪装成 `normal`）。

## 6. `render.request_key` 纳入 `clip_fraction_max` 与 `render.backend`

`dataset_build/agent_loop/render.py`、`runtime.py`

- `StrengthCalibrator.__init__` 新增 `backend: str = ""`；`runtime.create_services` 传
  `config.render.backend`。
- `request_key` 新增 `clip_fraction_max`、`backend`；`calibration_contract` 由
  `sample4096-bisect-applied-alpha-v2` bump 到 `-v3`，保证 C1b 之前的缓存行不会被解析命中。
- 测试 `test_calibration_request_key_covers_clip_ceiling_and_backend` 用 `cache_hit` 判定
  （返回的 `render_hash` 是内容派生的 `final_hash`，同强度下必然相同，不能用来判 request_key）。

## 7. applied-alpha 闸：`subject_artifact` 缺失 fail loud

`dataset_build/agent_loop/render.py`

- 原 `if subject_ref:` 静默跳过 → 现在缺失时写一条 `status="subject_artifact_missing"` 的
  render_record 并抛 `RenderError("subject_artifact_missing", ...)`。
- 该分支只在 `applied_alpha is not None`（即 local render）时触发，global 不受影响。
- 两个旧测试的 fixture mask 补上 `subject_artifact`（`test_local_calibration_rejects_a_leaf_below_the_visibility_floor`、
  `test_strength_contract_and_render_cache`）；新增
  `test_local_render_without_subject_artifact_fails_loud`。

## 8. `lut_render_distance.read_ratings` 兼容 `item_id`

- 与 `lut_pair_questionnaire.read_ratings` 同口径：`pair_id` 优先，缺失/空则回落 `item_id`。
- 两列都没有 → `SystemExit(f"{path}: no pair_id/item_id column")`（原来是静默读出 0 条评分）。
- 保留原有 duplicate id 的 fail loud。

## 9. Spearman n<3 → null + `insufficient_pairs`

- 新增 `SPEARMAN_MIN_PAIRS = 3`。
- `len(rows) < 3` 时 `rho`/`p` 全部落 `None`，`spearman` 块新增 `min_pairs` 与
  `insufficient_pairs` 两个字段。原来的 `(0.0, 1.0)` 与「真实测出的零相关」无法区分，已删除。

## 10. 落盘顺序：关键产物先写

| 工具 | 改动 |
|---|---|
| `lut_metric_p1` | 新增 `flush_stage(name)`：每个阶段（inputs / M1 / M3 / M4 / M2 / evaluate）完成即写 `stage_progress.json` + `pair_scores.partial.npz`（含当前已算出的全部列）。M4 CUDA OOM 不再毁掉 M1/M3 的小时级结果 |
| `lut_metric_ab` | `pair_delta_stats.npz`（render + 全对 CIEDE2000，唯一昂贵阶段）在模型拟合/候选扫描/report 之前落盘，不再在最后统一写 |
| 三个 questionnaire builder | 顺序改为 **item_key.json → csv → page → (可选) prune**；prune 的 `OSError` 被捕获成 `thumbnails_prune_error` 字段而不是抛出。recheck 的 leak_scan 依赖 page，因此 item_key 写两次（先落全部参数，leak 扫完补 `leak_hits` 再落一次） |
| `sample_mask_role_pilot` | `samples.jsonl` 改为逐条 append+flush；`_write_index` 包 try/except 成 `index_error` 字段 |
| `sample_mask_context_pilot` | 同上。原来 bucket 配额不足会 `raise RuntimeError` 而 `samples.jsonl` 还没写，整批已渲染样本全丢；现在 raise 之前记录已在盘上 |

## 11. 四个 metric/聚类工具的 manifest 补 `features.jsonl` 指纹

- `lut_render_distance.features_jsonl_path(databuild)` / `features_inputs(databuild)`：读
  databuild toml 的 `presets.bank_dir`，返回
  `{features_jsonl, features_jsonl_sha256, features_jsonl_rows}`；文件不存在时三项为路径 + 两个
  `None`（不伪造）。
- 接入 `lut_render_distance`、`lut_metric_ab`、`lut_metric_p1` 的 manifest `inputs`。
- `cluster_lut_effects` 有自己的 `load_catalog`/`sha256_file`，为不新增 import 边，在该模块内实现
  一份同名同语义的 `features_inputs`（注释说明了这个选择）。
- 理由：`LutCatalog.load` 只保留同时出现在 `features.jsonl` 且路径可渲染的 preset，因此 bank 内容
  决定了任何 metric/聚类 run 究竟看到了哪些 LUT；只记 `annotations_sha256` 只覆盖一半 catalog 身份。

## 12. `artifacts.discard()` 改为引用检查（选：先认领后 unlink）

**选择的方案**：把 `mark_artifact_purged` 变成**原子认领**，`discard` 只在认领成功后执行。

`dataset_build/agent_loop/persistence.py`、`graph.py`

- `AuditStore.mark_artifact_purged(sha256) -> bool`（协议 + SQLite + Postgres 三处签名同步）。
  - SQLite：返回 `cursor.rowcount > 0`。
  - Postgres：`... AND retention='quarantine' RETURNING sha256`，返回 `bool(rows)`。
- `graph._discard_uncommitted_renders`：原来是 `discard() → mark_purged()`，现在改成
  `mark_artifact_purged() → discard()`，只有认领赢家 unlink。

**为什么这就是引用检查**：`record_artifact` 的 upsert 里 `accepted` 是**粘性**的
（`CASE WHEN artifact_record.retention='accepted' THEN 'accepted' ELSE excluded.retention END`）。
任何被某个 committed row 引用过的 blob 一定是 `accepted`，条件 UPDATE 匹配 0 行 → 返回 False →
不 unlink。这正是「PG 里同 sha 是否被其他 row 引用」的等价判定，且不需要新扫 `render_record`
（`render_record` 里 `status='accepted'` 的行本身就是通过 `promote()` 把 blob 打成 `accepted` 的）。

**跨 pass 竞态（明确记录）**：
- 已关闭的窗口：两个 pass 同时清理同一个去重 blob——条件 UPDATE 只有一个赢，不会双删；
  以及本 source 的 `protected_sha256` 不认识别的 source 已 commit 的同 sha blob——现在被
  `accepted` 挡住。这两个都是原实现会真删数据的路径。
- **仍存在的窗口**：认领成功（retention 已置 `purged`）到 `path.unlink()` 之间，如果另一个 pass
  恰好 `promote()` 了同一个 sha，promote 会把 retention 拉回 `accepted`，但 blob 已被 unlink。
  该窗口是微秒级且只在完全相同内容的渲染同时发生时出现；`ArtifactStore.path_for` 有 catalog_db
  回填路径，`promote` 本身也会先 `path_for`。彻底关闭需要 blob 级引用计数或把 unlink 放进同一个
  事务，两者都超出 C1b 范围。**未做，明确留账。**
- `retention.apply_cleanup_manifest` 里是 `unlink() → mark_purged()` 的老顺序，有同类窗口。
  它走的是人工确认的 dry-run manifest（两阶段 + size 校验），不在本任务卡的 12 项内，**未改**。
- 新增测试：`test_terminal_cleanup_keeps_a_blob_another_source_committed`、
  `test_purge_claim_is_won_exactly_once`。

---

## 测试与检查

| 项 | 结果 |
|---|---|
| 基线（`test_agent_loop.py` + `test_direction_match.py` + `test_lut_annotations.py`） | 147 passed |
| 改后（同三文件） | 174 passed（+27） |
| 新增 `dataset_build/tests/test_lut_tools_c1b.py` | 10 passed |
| 四文件合计 | **186 passed**（+39，只增不减） |
| `py_compile` 全部改动文件 | OK |
| 全量 `dataset_build/tests/`（排除 4 个 torch 缺失的收集失败文件） | 改前 120 failed / 271 passed，改后 120 failed / 457 passed。失败集完全一致，全部是 `ModuleNotFoundError: No module named 'construct'` 之类的**既有环境问题**，落在本次未触碰的 `test_stop_fill.py` / `test_winner_margin.py` 等文件 |
| ruff | `.venv` 里没有 ruff（C1 报告属实），用 `/home/bc/miniconda3/bin/ruff` 0.15.2 跑 `dataset_build/agent_loop/ dataset_build/tools/` + 两个测试文件：14 个 error，**全部在本次未改的文件**（`bake_luts.py`/`build_taxonomy.py` 的 E741、`reeval_annot_mech.py`/`reeval_relay.py` 的 F401、`source_reuse_dryrun.py` 的 F841）。本次改动文件 0 findings |

## 未修（W/N 级 / 越界）

- `retention.apply_cleanup_manifest` 的 unlink/mark 顺序（见 12）。
- `discard()` 本身仍是无条件 unlink 的低层原语；引用检查放在唯一的调用点
  `_discard_uncommitted_renders`。若将来出现第二个调用点，需要把该守卫下沉到 `ArtifactStore`。
- 三个 questionnaire 工具的 `read_ratings` 各自一份（chain/intent 只认 `item_id`，
  `lut_pair_questionnaire` 认两者）——本次只按 blocker 8 统一了 `lut_render_distance` 一处。
