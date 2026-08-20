# NOTES · optB · B11

规格依据：`docs/REQUIREMENTS_local_visibility_lut_selection_20260820.md` R7.1–R7.4 + R6.2 接线。
上游：`NOTES_optB_B10.md`（分段指纹 + 方向匹配，纯函数层）、`NOTES_optB_B8.md`（mask reach 探针）。
改动文件：`dataset_build/agent_loop/{models,candidates,direction_match,render,prompts,graph,
segment_fingerprints}.py`、`dataset_build/tests/test_agent_loop.py`、`configs/agent_loop.*.toml`
（含新增 `configs/agent_loop.local-v2-b11-smoke.toml`）。

## 1. 挂载与启动断言（任务卡 item 1）

- 全部 11 个 `configs/agent_loop.*.toml` 的 `[catalog]` 加
  `segment_fingerprints = "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v1.jsonl"`
  （iter2/3/4/5/5b、b15-smoke、b11-smoke、local-v1、local-v1.annotate、local-v1.run、
  a6-lane-smoke、terra-smoke）。单测 `test_graph_build_requires_a_mounted_segment_fingerprint_table`
  遍历 `configs/agent_loop.*.toml` 断言每个文件都有这一行。
- `build_graph()` 首行断言 `services.catalog.segment_fingerprints_mounted`，否则
  `RuntimeError("segment_fingerprints_not_mounted")`（B10 NOTES 留的接线点）。
- `LutCatalog` 新增 `segment_fingerprints_sha256`（挂载文件的 SHA-256，未挂载为 ""），
  `restrict()` 透传。生产实测 `bac4db04…4939`，与 B10 落盘一致。

## 2. R7.1 local 在线检索（任务卡 item 2）

`build_local_packets` 的 local 侧现在**完全不读** `preset_reach`（离线 top-300 reach 集）；
该参数已从签名删除，只有 global 轮的 `global_shortlist` / `_row_view(level="global")` 还在用它
（`_row_view` 现在见到 `level != "global"` 直接报错，防止回流）。

每个 mask 一次（不是每个 intent 一次）：

1. `LocalDirectionProbe.direction(mask)`（新增在 `render.py`，与 `MaskReachProbe` 并列）出
   `MaskDirection`：
   - `correction` = `measure_direction(source_render, global_after, mask_alpha, mode="correction")`，
     即 global 编辑在该 mask 里留下的残差；
   - `enhancement` = `direction_from_diagnosis(diagnosis, "enhancement")`（关键词口径，
     同一 source 的所有 mask 共用一条）；
   - `tonal_weights` = `measure_tonal_weights(global_after, mask_alpha)`。
   probe 按 mask alpha 的 sha256 缓存，两张图各读一次。
2. `LutCatalog.direction_scores(direction)` 在**全库 4,051** 上一次向量化出两列：
   `combined`（预筛键）与 `correction`（`cast_correction_local` 的域判据）。
   `combined = mean(非零方向的 direction_match_scores)`，两个方向都已按 mode 定向
   （correction 反向得高分、enhancement 同向得高分），故直接取算术平均。
   **预注册**：两方向权重各 1.0；两条都是零向量时全库同分 0，预筛退化为 `preset_id` 字典序。

每个 (mask, intent)：

3. `intent_admits` 在**全库**过 intent 域（含 global preset 排除、禁用 intent 排除照旧）；
4. 按 `combined` 降序（同分 `preset_id`）取 **top-50**（`LOCAL_ONLINE_RETRIEVAL["prefilter_top_k"]`）；
5. `MaskReachProbe.measure` 只测这 ≤50 条，`mask_reach_de < 4.0` 丢弃（探针按 (mask,preset)
   缓存，跨 intent 复用）；
6. 幸存者按既有 `LutCatalog._score`（palette/diagnosis 启发式打分）排序 → 簇去重 →
   `_quota_select`（每 bin ≥1 行）→ 每包 ≤4 行。**行内排序算法一字未改**，方向分只做预筛。

**`achievable_bins` 换口径（实现被迫的规格补齐，请用户确认）**：local 行原来按离线全图
`d_full ≥ bin 下界` 推可达 bin；全库在线检索后绝大多数 preset 根本不在离线 top-300 里，
没有 `d_full` 可读。现改为按 **mask 条件化可达** 推：`achievable_bins = {bin : mask_reach_de ≥ 下界}`。
这与 R1.1「mask 加权 ΔE 才是 local 校准最大化的量」同口径，且因为 reach 门 = 4.0 = 每条梯子
`subtle` 的下界，任何进包的行至少有 `subtle`，不会出现空 bin 行。global 轮不受影响。

新增审计列 `local_retrieval`（落 branch 行 + tree manifest）：每 (mask, intent) 一行
`{domain_size, prefiltered, reach_dropped, rows}`。行级新增 `direction_score`（预筛分，
**不进 prompt**，只进审计）。

运行时断言：`build_local_packets` 返回 `mask_reach_applied` 与 `direction_prefilter_applied`，
graph 侧任一为假直接 `RuntimeError`（`mask_reach_gate_not_wired` / `direction_prefilter_not_wired`）；
两个 probe 现在都是**必填**，`None` 直接 `CandidateError`。

### 实测耗时（本机，真实 4,051 条 catalog + 真实源图，3 mask/分支）

| source | 单分支 `build_local_packets` | (mask,preset) 探针对数 | 包数 | 摊平行数 |
| --- | --- | --- | --- | --- |
| src_0f68bbe15e69d924 | 1.786 s | 463 | 13 | 45 |
| src_0469da8eef67330c | 1.687 s | 404 | 13 | 35 |
| src_3fa2daaf5eb8b9d0 | 1.607 s | 465 | 13 | 36 |

均值 **1.694 s/分支**。构成：方向打分全库一次 0.38 ms（B10 实测）× 3 mask，其余几乎全部是
mask reach 探针（约 3.6 ms/对 × ~450 对）。B8 时代每分支约 36 对 ≈ 50 ms，本卡放大约 34×。

同批实测的域规模（全库 4,051 条里落在各 intent 域的条数，与 mask 无关的部分）：
`luminance_pop` 2,304、`hue_shift` 338、`sat_boost` 273、`contrast_boost` 991、
`cast_correction_local` 701–1,204（随实测残差变化）。

## 3. R7.3 两个新 intent（任务卡 item 3）

`ACTIVE_LOCAL_INTENTS` 5 → **7**：新增 `contrast_boost`、`cast_correction_local`。

- **角色**：`INTENT_ROLE_DOMAINS`（新表）允许一个 intent 服务多个角色；两个新 intent 都是
  `("subject", "background")`，v1 七个 intent 保持单角色。`INTENT_ROLES` 保留为
  `domains[0]`（冻结审计行照旧可解析），角色判定统一走 `intent_serves_role()`。
- **`contrast_boost` 域**：`segment_fingerprint["shadows"]["dL"] ≤ −1.0` 且
  `["highlights"]["dL"] ≥ +1.0`（`INTENT_SEGMENT_GATE`，预注册初值，单位 Lab L*）。
  全库命中 991 条。
- **`cast_correction_local` 域**：`direction_match_score(实测残差, correction 模式) ≥ 0.5`
  （`INTENT_DIRECTION_GATE["cast_correction_match_min"]`，预注册初值，余弦 ∈ [−1,1]）。
- 两个域的输入缺失时 `intent_admits` **直接报错**（不静默放行），单测覆盖。
- **梯子**：两者都放 `low`（`subtle`/`natural` 同为 [4.0,4.5)、`strong` [4.5,5.5)）。
- **排序倾斜**：`INTENT_PACKET_PRIORITY = {cast_correction_local:0, contrast_boost:1,
  zonal_contrast:2}`，其余 3；`intent_packet_order()` 用 (priority, 声明序) 排序，
  同一 mask 的包按此序发出，也就是摊平 shortlist 的行序与 `offered_intents` 的顺序。
  **是排序键不是配额**：没有任何 intent 因为排在后面被丢。
- prompt 的 `_LOCAL_INTENT_GUIDE` 加两条说明，且这两条排在最前。

## 4. R7.2 local prompt 双图（任务卡 item 4）

`local_request(endpoint, source, global_after, ...)` 新增 `source` 位参。

- 稳定前缀 = `[规则(developer), intent 指南, diagnosis, **source 图**, local shortlist 文本]`；
  尾部 = `[**global_after 图**, global_decision+actual_render, excluded/assigned/task, repair?]`。
- 图片序因此是 **source → global_after**。source 图放在 shortlist 文本**之前**，
  所以同一 source 的所有 global sibling 与 repair 共享的前缀更长（规则+指南+diagnosis+source 图
  这四段逐字节相同；shortlist 每分支不同，本来就没共享过）。`prompt_cache_key` 的前缀
  仍只算 `stable_user`，sibling/repair 三者 key 相同这条既有单测未变。
- `_LOCAL_RULES` 开头改写为「两张图，第一张是未编辑原图，第二张是真实渲染的全局结果；
  把它们当 before/after 读：全局改掉了什么、留下了什么、区域里还有什么空间。你选的每个
  local 编辑都作用在第二张图上」。
- graph 的两个调用点（`local_propose_batch` / `local_repair`）都传 `state["source_render_artifact"]`
  （512px 归一化 render，与 global 渲染的输入同一张，尺寸与 global_after 必然一致；
  `LocalDirectionProbe` 也用这一张，形状不一致直接 `RenderError("direction_shape_mismatch")`）。

## 5. R7.4 强度下沿 3.5 → 4.0（任务卡 item 5）

- `LOCAL_VISIBILITY_FLOOR` 3.5 → **4.0**；`MASK_REACH_GATE["reach_de_min"]` 跟随（同一常量）。
- 梯子：所有下界原为 3.5 的带整体抬到 4.0，上界不动 →
  `high = {subtle:(4.0,4.5), natural:(4.5,5.5), strong:(5.5,6.5)}`，
  `low = {subtle:(4.0,4.5), natural:(4.0,4.5), strong:(4.5,5.5)}`。带名一个没删，
  `LOCAL_BIN_DOWNSHIFT` 仍有落点。副作用：`local_target_center("luminance_pop","strong",
  capped=True)` 4.0 → 4.25。
- 导入期断言 `assert_local_visibility_floor()` 未改，自动覆盖两个新 intent。

## 6. revision（任务卡 item 6）

- `CANDIDATE_SERIALIZATION_REVISION`：`lut-intent-v6.3-optB` → **`lut-intent-v7-optB`**。
- `prompt_revision_fingerprint()` 新纳入 6 个键：
  `segment_fingerprint_table_sha256`（= `SEGMENT_FINGERPRINT_TABLE_SHA256`，新常量，
  值为生产指纹表的 SHA `bac4db04…4939`）、`local_online_retrieval`、`intent_role_domains`、
  `intent_segment_gate`、`intent_direction_gate`、`intent_packet_priority`、
  `intent_packet_order`；`local_intents` / `active_local_intents` / `local_delta_e_ladders` /
  `local_visibility_floor` / `mask_reach_gate` / `rules.local` / `rules.local_intent_guide`
  的值本卡也都变了。
- 生产实测：`prompt_revision_fingerprint() = d770156b90edbafca19f9eba8669419d08c1ec12598bef19db5481a660faf04b`，
  `thread_revision = local-agent-v1-06abb9da455f`。新老 campaign 靠 thread_revision 隔离。

## 待决策 / 保守默认（未静默拍板）

- **`achievable_bins` 改由 mask reach 推导**（见 §2 末）。这是「全库在线检索」的直接后果，
  不是本卡自选的口径变更，但它改变了 prompt 里 `achievable_bins` 这一列的含义（全图 → mask 内），
  请用户确认。回退需要为全库 4,051 条补离线 `d_full`。
- **两个新 intent 的角色域取「主体/背景皆可」**。任务卡只对 `contrast_boost` 明写「主体/背景皆可」，
  `cast_correction_local` 未写。两者的定义都只提「mask 区」，故一并给双角色；
  收窄成单角色只需改 `INTENT_ROLE_DOMAINS` 一行。
- **两个新 intent 的梯子取 `low`**。没有任何数据可以把它们放到 `high`，取更弱的一档为保守默认。
- **`cast_correction_local` 阈值 0.5**、**`contrast_boost` 的 ∓1.0 Lab L***、
  **预筛 top-50**、**两方向权重各 1.0**：全部是预注册初值，没有用任何数据标定。
- **预筛的两方向合成取算术平均**。任务卡写「按 (方向, mask 色调权重) 打分预筛」，未给合成式；
  取最简的等权平均。改成 max / 加权只需改 `combined_direction_scores` 一处。
- **`enhancement` 方向来自诊断关键词，不是实测**。任务卡写「measure_direction(...) 测残差方向
  （纠偏模式）+ 诊断 enhancement 方向（增强模式）」，按字面执行：纠偏侧实测、增强侧读
  `enhancement_opportunities`。因此增强方向对同一 source 的所有 mask 相同，只有色调权重按 mask 变。
- **每分支包构建 1.69 s（B8 时代 ~50 ms）**。全部来自 mask reach 探针从 ~36 对涨到 ~450 对。
  200 源 × 约 3 分支 ≈ 17 分钟纯 CPU。若要压回去，唯一的旋钮是 `prefilter_top_k`
  （50 → 20 大约省 60%），未擅自调。
- **`direction_score` 未进 prompt**。序列化面只多了 B8 已有的 `mask_reach_de` 一列，
  行文本格式一字未改。要让模型看到方向分需要改 shortlist 列，未做。
- `segment_fingerprint_table_sha256` 是**模块常量**，不是运行时读到的文件 SHA：
  `prompt_revision_fingerprint()` 是纯函数、拿不到 config。启动断言只查「挂上了」，
  挂错文件不会被这条常量抓住（`catalog.segment_fingerprints_sha256` 落审计可事后核对）。
  要做成硬断言需要把 config 传进 fingerprint 链，那会改 revision 的定义域，未擅自做。

## 测试

- `dataset_build/tests/test_agent_loop.py`：115 → **121 passed**（只增不减）。
  新增 6 条：
  - `test_new_intent_domains_read_the_segment_fingerprint_and_the_residual`
    （两个新域各只放行一条；两侧闭区间边界；反向 LUT 得负分；输入缺失报错）
  - `test_intent_packet_order_leans_on_the_two_new_intents`
    （排序键前三、并列回落声明序、双角色域、v1 单角色）
  - `test_online_retrieval_reads_the_whole_catalog_and_is_deterministic`
    （60 条域 → 预筛恰好 50 条 → 只有这 50 条被探针测过；两次调用逐值相等；排除生效）
  - `test_local_direction_probe_measures_the_source_to_global_after_residual`
    （mask 内读到残差、mask 外零向量、色调权重和为 1、增强向量跨 mask 相同、按 mask 缓存）
  - `test_graph_build_requires_a_mounted_segment_fingerprint_table`
    （挂上才能 build；卸掉 → `segment_fingerprints_not_mounted`；全部生产 config 都挂了）
  - `test_prompt_revision_fingerprint_covers_the_b11_retrieval_contract`
    （6 个新键逐个 monkeypatch 都能改 fingerprint）
- 改判据重写（口径变了，不是放宽）：
  `test_local_strength_targets_...`（4.0 梯子 + 两个新 intent 的梯子）、
  `test_luma_soft_cap_...`（下调后的带 4.0/4.25）、
  `test_local_visibility_floor_covers_every_active_intent`（4.0）、
  `test_local_calibration_rejects_a_leaf_below_the_visibility_floor`
  （dE 3.845 拒 / 4.348 过，原来是 2.807 / 4.098）、
  `test_local_reach_bins_follow_the_intent_ladder`（bin 改由 mask reach 推）、
  `test_intent_rows_fill_one_per_bin_and_report_shortfalls`（新签名）、
  `test_shortlist_excludes_unreachable_presets_and_exposes_reachable_bins`（local 半段改写）、
  `test_build_local_packets_...` / `test_dead_zone_lut_never_enters_a_local_packet`
  （两 probe 必填 + 7 intent + 包序 + `local_retrieval` 列）、
  `test_repair_and_sibling_prompt_prefixes_are_stable`（双图序 + 前缀 4 段）、
  `test_shortlist_row_serialization_is_byte_stable`（revision 串）、
  `test_intent_packets_and_packet_notes_land_in_branch_rows_and_tree`
  （`contrast_boost` 在 v1 主体 intent 全空的夹具上仍能成包）、
  `test_mask_reach_gate_is_wired_into_the_branch_and_the_leaf_audit`（两条接线断言）、
  `test_mounting_segment_fingerprints_leaves_the_thread_revision_alone`
  （夹具现在默认挂载，改为显式卸载后对比）。
- `dataset_build/tests/test_direction_match.py` + `test_lut_annotations.py`：**26 passed**（未改）。
- 全目录 `dataset_build/tests`（忽略两个既有 collection error 文件
  `test_annot_contract_v4.py` / `test_annotation_interleave.py`）：
  **31 failed / 598 passed**——失败集合与 B10 记录的 31 failed / 591 passed 完全一致
  （`ModuleNotFoundError: No module named 'construct'` 与 config 样例断言，与 agent_loop 无关），
  passed 增量 +7。
- `python -m py_compile`、`python -m ruff check dataset_build/agent_loop dataset_build/tests/...`：通过。

## 5 源端到端验证（campaign `local-v2-b11-smoke`）

配置 `configs/agent_loop.local-v2-b11-smoke.toml`（PG 同库、两 lane 真实 Terra、CPU LUT 渲染、
validator 关闭、landing 关闭）。源清单 `/home/bc/data/agent_loop/local-v1/b11_smoke5.jsonl`：
`sources5k.annotated.jsonl` 按 `sha1(source_id)` 升序、跳过 `subject.mask_area > 0.60` 取前 5 条，
与 B15 的 5 源**完全相同**（src_0f68bbe15e69d924 / src_0469da8eef67330c / src_3fa2daaf5eb8b9d0 /
src_347cf4dbbbe16d7d / src_aeae0aacd7c1c68e）。

### 验证结果：**未完成，被 provider 侧故障挡住**（2026-08-20 10:20–10:57）

按顺序发生的事实，全部有日志（`/home/bc/data/agent_loop/local-v1/b11/preflight.log`）：

1. 10:20–10:44，`preflight` 连续 12 次失败：`CacheExhausted: 503:api_error`
   （夹杂 2 次 `429:rate_limit_error`）。用 openai SDK 直发一条 `input="ping"`
   同样返回 `503 {'message': 'Service temporarily unavailable', 'type': 'api_error'}`，
   与本卡改动无关。
2. 10:44 起错误变成 `404:model_not_found`。`client.models.list()` 返回 13 个模型：
   `gpt-5.2 / gpt-5.2-2025-12-11 / gpt-5.2-chat-latest / gpt-5.2-pro / gpt-5.2-pro-2025-12-11 /
   gpt-5.3-codex / gpt-5.3-codex-spark / gpt-5.4-2026-03-05 / gpt-5.5 / gpt-5.6 / gpt-5.6-sol`
   （+2 条非 gpt-5 系列），**没有 `gpt-5.6-terra`**。直发报
   `Model "gpt-5.6-terra" is not supported by any configured account in this group`。
   即：**provider 把此前所有 campaign 用的 `gpt-5.6-terra` 从本账号组下线了**。
3. 因此把 **smoke 配置**（且只有 smoke 配置）的 `[terra] model` 临时改为 `gpt-5.6`，
   配置里写了醒目注释。**生产 config（iter 系列 / annotate / run）一律未动，仍是
   `gpt-5.6-terra`。** 这是被动应对，不是选型决定，请用户裁决：是改模型、还是等 provider
   恢复 terra、还是换 provider。
4. 10:49 起以 `gpt-5.6` 重试 preflight，到 11:12 为止连续 20 次全部 `503:api_error`。
   provider 整体仍不可用。

**结论：5 源端到端验证（≥3/5 accepted / intent 分布 / 双图请求 / token）本卡未取得。**
重试脚本 `/home/bc/data/agent_loop/local-v1/b11/preflight_retry_long.sh` 仍在后台跑
（最多 200 次 × 60 s，preflight 一过就自动执行 `run5.sh`），日志同目录
`preflight.log` / `run5.log`。
用户可在 provider 恢复后直接看这两个文件，或重跑：

```
export VERARETOUCH_AGENT_POSTGRES_DSN='postgresql://research:research@127.0.0.1:5432/agent_loop?options=-c%20search_path%3Dagent_loop%20-c%20default_tablespace%3Dagent_loop_local'
python -m dataset_build.agent_loop.cli --config configs/agent_loop.local-v2-b11-smoke.toml preflight
/home/bc/data/agent_loop/local-v1/b11/run5.sh
```

### 离线替代测量（不需要 API，同 5 源、真实 catalog、真实 mask、真实 CPU 渲染的 global_after）

不是端到端验证，只是把「双图」「token」两项能离线算的部分算了：

| source | local prompt 文本 token | 图片数 | 摊平行数 | 包数 |
| --- | --- | --- | --- | --- |
| src_0f68bbe15e69d924 | 5,150 | 2 | 45 | 13 |
| src_0469da8eef67330c | 4,401 | 2 | 35 | 13 |
| src_3fa2daaf5eb8b9d0 | 4,448 | 2 | 36 | 13 |
| src_347cf4dbbbe16d7d | 3,527 | 2 | 25 | 11 |
| src_aeae0aacd7c1c68e | 4,114 | 2 | 34 | 11 |
| **均值** | **4,328** | **2** | 35.0 | 12.2 |

口径：`tiktoken` `o200k_base` 对 `local_request(...)` canonical input 里全部
`input_text` 段求和；**不含图片 token、不含 provider 侧模板开销**，因此与审计里 provider
报的 `input_tokens` 基线 6,386 **不同口径，不能直接相减**。图片数 = 2（source + global_after），
即 R7.2 的双图确实进了请求。真实 token 变化待 provider 恢复后从审计 `usage` 读。
