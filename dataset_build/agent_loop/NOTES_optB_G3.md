# NOTES · G3 global 提案按增强方向扇出

日期:2026-08-24。范围:只改代码与测试,未动 config / 未起进程 / 未提交。

## 实现落点

- `prompts.py`
  - `_GLOBAL_RULES`:只替换一句(「cover different diagnosed directions rather than
    variants of one look」→「you are given the corrections and ONE enhancement
    direction; serve the corrections first, then realize THIS direction」),其余原文未动。
  - 新增 `single_direction_diagnosis(diagnosis, index)`:只把
    `enhancement_opportunities` 收窄为一项,其余字段原样透传。
  - `global_request(..., enhancement_index=None)`:prefix 顺序不变
    (rules → 源图 → diagnosis → histogram → shortlist),方向不同 ⇒ diagnosis 块起
    分叉 ⇒ `prompt_cache_key` 不同。
  - 新增常量 `GLOBAL_DIRECTION_FANOUT` + `global_direction_budget()`;
    registry 增键 `global_direction_fanout`,`PROMPT_REGISTRY_KEYS` 46 → 47,
    计数断言同步。
- `graph.py`
  - `_global_propose_node`:按 diagnosis 的每个增强方向各发一次请求;机会为空则退化为
    单次调用、diagnosis 原样。
  - `_global_response_validator(state, min_proposals)`:原校验逻辑抽出,下限改成
    「本次调用的 min」。
  - `_merge_direction_proposals(per_direction, budget)`:去重 → 轮转截断 → 按方向拼接
    (每方向内按 scorer 顺序)。
  - 运行时断言:方向数 ≥ 2 且存活提案只来自一个方向 ⇒
    `RuntimeError("global_direction_fanout_degenerate")`。
  - 每个提案增键 `enhancement_index` / `enhancement_direction` / `style_major`
    (既有键未动),随 branch 记录进审计 JSON。

## 保守默认(未静默拍板,列此备查)

1. **去重键实际是 `(style_major, row_index)`**,不是裸 `row_index`。原因:每个方向的调用
   各自选 major,而 row_index 只在某个 major 内唯一;一个 preset 只属于一个 major,所以
   在「各方向选同一 major」的常规情形下与裸 row_index 去重完全等价。
2. **`selected_major` 取第一个方向响应的 major**(旧行为是唯一那次调用的 major)。
   manifest 的 `style_major` 因此只记第一方向的 major。跨 major 时
   `proposal_id = g{row}-{bin}` 可能在两个 major 间重名:`branch_id` 含 preset_id 仍唯一,
   但 mask packet 的哈希种子会撞(两支拿到同一套 mask 组合)。未改 proposal_id 格式。
3. **单次调用(无增强机会)保留 `config.min_global_proposals` 作为下限**,max 仍是
   `config.max_global_proposals`;若改成 min=1,correction_led 源可能只回 1 个提案而被
   `formal_globals_lt_2` 判掉,属行为回退,故不取。
4. **截断预算**:`max_global_proposals < 方向数 n` 时,轮转到预算耗尽即止,靠后的方向拿不到
   名额(当前配置 6 ≥ n∈[2,4],不会发生)。
5. **prompt registry 变更 ⇒ `thread_revision` / `thread_id` 改变**:与正在跑的 g30
   campaign 线程不兼容(不同 thread_id,不会续跑旧 checkpoint)。这是预期,不是回归。
6. 每方向 max = `max(2, max_global_proposals // n)`,再被 `max_global_proposals` 封顶;
   n=2→3、n=3→2、n=4→2(按 max=6)。

## 待决策(需用户/主 agent 裁决,本卡未做)

- **per-direction shortlist 是否也按方向重召回**:当前所有方向共用同一份
  `build_global_shortlist` 的召回结果(按完整 diagnosis 召回)。「每方向各自召回一份
  shortlist」会让方向间候选池本身分开,但会打掉 shortlist 在 prefix 中的共享、并使
  `selected_major` / 去重口径重新定义。**本卡不做,登记为待决策。**
- **退化断言的严厉程度**:现按任务卡实现为 `RuntimeError` 直接杀掉该源。真实模型在候选行
  很少(某 major 只有 2–3 行)时可能各方向选到同一批行,从而整源失败。备选口径:记审计列
  + 继续跑,由离线统计方向覆盖率。需裁决。
- **跨 major 的 proposal_id 重名**(见保守默认 2)是否要把方向号并进 `proposal_id`。
  改了会变更既有 id 格式与历史可比性,本卡未改。
- **`min_global_proposals` 的语义**:现在只在「单次调用」路径生效,扇出路径的下限固定为 1。
  是否要给合并后的总数也加一条下限(现无此断言)。

## 未完成/环境

- **ruff 未跑**:本机 `.venv` 与 PATH 内均无 ruff 可执行文件(`.venv/bin/ruff` 不存在,
  `which ruff` 为空)。已改行长均 ≤ 92,与文件既有风格一致;新增 import 均被使用。
- `py_compile` 通过。
- 运行判据命令:`.venv/bin/python -m pytest dataset_build/tests/test_agent_loop.py
  dataset_build/tests/test_source_histogram.py dataset_build/tests/test_histogram_board.py -q`
  → `207 passed`。
- 全量 `dataset_build/tests`(排除 4 个需 torch 的收集失败文件):改前 120 failed / 535
  passed,改后 120 failed / 539 passed(+4 = 本卡新增用例),无新增失败。
