# 实现审阅 · TOOL-SourceReuse-1（`[sources] max_source_uses` 源图重复采样开关）

> 审阅人：实现审阅 subagent（独立，与编码 agent 无关）
> 日期：2026-08-12（一轮）／2026-08-12 复核（**终版**）｜ 分支：`lens-exp`
> 审阅对象：`dataset_build/src/construct/{agent,config,sources,state,presets}.py`、
> `dataset_build/tests/test_source_reuse.py`、`dataset_build/tests/test_source_window_and_cgt.py`
> 设计文档：`docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md`
> 纪律：只读审阅、未改任何实现、未跑 build、未碰 `/mnt/nfs`、未读 `trash/`。
> 行号：§1-§11 为**一轮**工作树行号，§12（复核）为**修复后**行号。

---

## 0. 终版结论

# ✅ **PASS —— 0 blocker，可用于 L8**

| 审阅项 | 一轮 | 复核后 |
|---|---|---|
| ① 默认路径（`max_source_uses=1`）逐字节等价 | pass（2 nit） | **pass**（N1 已修；clamp 使其更强） |
| ② resume 安全 | **blocker B1** | **pass —— B1 已清**（§12.1） |
| ③ `group_id` 后缀方案与全链路 | pass | **pass**（未受修复影响） |
| ④ 分配 bug 修复只在开关打开时生效 | pass | **pass** |
| ⑤ C_GT / mask 复用决策 | pass（2 nit） | **pass**（N3 已修，跨 batch 复测仍通过） |
| ⑥ `preset_inventory_exhausted` 按源终态化 | pass（1 nit） | **pass**（N5 采纳为 L8 设计建议） |
| ⑦ 测试有效性 | **blocker B2** | **pass —— B2 已清**（§12.2） |
| ⑧ 协议红线扫描 | pass（全部无涉） | **pass** |
| **合计** | 2 blocker / 8 nit | **0 blocker / 2 nit（均不阻塞）** |

**两个 blocker 都是真清了，不是纸面回复**：

- **B1**：我把一轮写的两个复现脚本（`repro_resume.py` / `repro_drain.py`）**原样重跑**
  —— 崩溃重启从"0 新 group"变成 **6/6 groups、`complete`、shortfall 0**；
  SAM3 救回的源从"只拿 1 次"变成**拿满 budget**。另外我独立扫了 clamp 的
  **232 个配置**（含 `budget=0` 不限模式）：无负 index、无源越 budget、
  无重复 `(source, use_index)`、无不终止。
- **B2**：我**重跑了实现方的真实 bank dry run**（26,460 源 × 12 趟 = 317,520 组），
  输出与 NOTES §七 **逐位一致**（重复率 5.2484%、max overlap 7/8、整组重复 0）。
  manifest 三列已落地，并有一条退化目录用例证明这三列**真的会动**。

**测试现状（复核实测）**：`test_source_reuse.py` **39 passed / 32 subtests**；
全套 `dataset_build/tests/` **30 failed / 413 passed**，30 条失败逐条确认
**全部**是 `databuild.example.toml` 缺失（NOTES D1，先于本改动存在，见 §9）。

**放行时仍建议做的两件事**（都不是 blocker，见 §12.6）：
恢复 `databuild.example.toml` 并在 `[sources]` 写上本 key（否则 L8 操作者
没有任何配置模板能看到这个开关存在）；把 §12.5 的 N9（resume 空转）
和 D2 的记账缓存合成一张卡，在 L8 首次重启前处理。

---

## 1. ① 默认路径逐字节等价 —— **pass**（2 nit）

不看测试，逐分支读码确认。四个派生点全部走"空元组拼接"，`use_index = 0` 时
传给 `stable_id` 的 `*parts` 与改动前**完全相同的元组**：

| 派生点 | 位置 | 默认路径 |
|---|---|---|
| `group_id` | `agent.py:1900-1904` | `*(() if not use_index else (use_index,))` → 空 |
| `reservation_id` | `presets.py:414-417` | 同上 |
| `render-source` task_id | `agent.py:2122-2125` | 同上 |
| group 行 `source_use_index` | `agent.py:2028-2032` | `**({} if not use_index else {...})` → 不写键 |

`stable_id`（`state.py:36-38`）是 `f"{kind}_{sha256(...)[:32]}"`，
参数元组相同 ⇒ 字符串逐位相同。已用 `stable_id("group", build_id, source_id, "global", 0)`
在单测里对拍（`test_source_reuse.py:365-369`）。

**跳过判据等价性**（`agent.py:2350-2352`）：
`completed_source_uses().get(sid, 0) > 0` ⟺ `sid in completed_sources()`，
因为 `completed_source_uses`（`state.py:372-386`）就是同一次 `self.groups.values()`
遍历的计数版，键集合完全相同。**等价。**

**`_chunk_paths` 等价性**（`agent.py:2176-2199`）：
旧 `done = completed_sources() | skip`；新 `sid not in skip and used.get(sid,0) <= use_index`。
`use_index = 0` 时 `<= 0` ⟺ 计数为 0 ⟺ 不在 `completed_sources()`。**等价。**

**`allocate_sources`**（`sources.py:495-524`）：`max_source_uses == 1` 分支里的
四行是旧代码**逐字搬进 `if`**（我逐字符比对过 diff），`target_prefix` 语义未动。
后续 `shuffle` / `replacement_labels` / `zip` 全部在分支外，未改。**等价。**

**`_typed` 对 bool 的处理**（`config.py:312-315`）：`typ is int and isinstance(value, bool)`
显式拒绝，所以 `max_source_uses = true` 不会被当成 1 混进来。已有单测覆盖
（`test_out_of_range_and_non_integer_budgets_are_refused`）。**正确。**

### nit N1 — 默认路径多了一次全量 group 扫描
`_fill_mode`（`:2277`）在调用 `_fill_initial_mode` 前先算一次 `before = len(self._mode_groups(mode))`，
`budget = 1` 时这次扫描的结果**永远用不上**（`:2280` 直接 return）。
每个 mode 每次 `_fill_mode` 一次，按 NOTES D2 的实测 `_mode_groups ≈ 43 ms @200k groups`，
总代价可忽略，但"默认路径逐字节不变"的措辞严格说只覆盖落盘产物，不覆盖调用序列。
建议把 `before` 挪到 budget 判定之后。

### nit N2 — `_try_ready_relabel` 在默认路径新增了一次 O(groups) 扫描
`agent.py:2411-2414` 把 `use_index` 改为 `completed_source_uses().get(sid, 0)` 现算。
逻辑正确（`_pending_sam3_ids()` 在 `:823` 显式 `.difference(completed_sources())`，
所以这里恒为 0，默认路径行为不变），但它在**每次 relabel 尝试**都做一次全量 group 遍历。
L7 口径（27k groups、relabel 次数不多）可忽略；L8 口径下 relabel 循环里已经有
`_pending_sam3_ids()` / `_mode_groups()` 的同量级扫描，属同一笔既有账（NOTES D2），
不单独定级。注释写的"keeps the ID namespace right if that filter ever loosens"成立。

---

## 2. ② resume 安全 —— **BLOCKER B1**

### B1（blocker）`_fill_mode` 的收工护栏把「这一趟早就走完了」误判成「池子榨干了」

**位置**：`agent.py:2262-2288`

```python
budget = self.config.sources.max_source_uses
use_index = 0
while True:
    before = len(self._mode_groups(mode))
    self._fill_initial_mode(mode, sources, target, use_index=use_index)
    use_index += 1
    if budget and use_index >= budget:      return
    if len(self._mode_groups(mode)) >= target: return
    if len(self._mode_groups(mode)) <= before: return   # ← 这里
```

`use_index` 是**函数局部变量、每次调用从 0 起**，不是从 journal 恢复的游标。
`_fill_initial_mode(use_index=0)` 会跳过所有 `count > 0` 的源（`:2350`）。
于是**任何"第 0 趟已经走完"的局面**——重启、或 `_fill_mode` 被第二次调用——
都会让第 0 趟产出 0 个 group，撞上 `:2284` 的护栏直接 return，
**第 1..budget-1 趟永远不会开始**。

NOTES §二.3 声称 `use_index` 是"resume-stable cursor"、
`state.py:372-386` 的 docstring 也叫它"the resume-stable ``use_index`` cursor"——
**游标本身确实是幂等可重建的，但没有人用它决定从第几趟开始走**。

**说清楚它不坏在哪**：一次**全新、不崩溃、且没有源被 SAM3 救回**的 build，
`_fill_mode` 只被调用一次、`use_index` 从 0 连着走到 budget，**完全正确**——
交付的 12 个 pipeline 用例走的都是这条路，所以它们全绿。
坏的是另外两条路，而 L8 两条都会走上。

#### 复现 1：中途重启（这是 L8 的正脸）

脚本：`/tmp/.../scratchpad/repro_resume.py`（基于 `ReusePipelineTests` 夹具）。
target=6、pool=2 源、`max_source_uses=3`（容量 6，足够达标）。
在第 1 趟后于 `_land_checkpoint` 注入一次崩溃，再用干净依赖重跑同一 config：

```
after crash : groups = 4  [(source-1,0),(source-0,0),(source-1,1),(source-0,1)]
after resume: status = complete_with_failures  groups = 4   ← 一个都没加
              exhaustion = {'local_shortfall': 0, 'global_shortfall': 2}
              source_reuse = {"max_source_uses":3,"distinct_sources":2,"groups":4,
                              "max_observed":2,"uses_histogram":{"2":2}}
```

`max_observed = 2 < max_source_uses = 3` 且 target 未达成，**但 build 判定自己"没有进展、
收工"**。注意它**不报错**，只在 shortfall 里留一条数——正是 CLAUDE.md
「s 缓存契约」那一节反复讲的**静默失败**形状。
L8 口径：12 趟 / 400k 目标，第 3 趟崩一次 → 重启后 0 产出，
约 30 万 group 静默作废，manifest 只多一行 `local_target_shortfall`。

#### 复现 2：不用崩溃也会中招 —— SAM3 救回来的源只拿到 1 次使用

`_drain_sam3_and_replacements`（`:2503`）在 `_run_phases`（`:2550-2555`）之后
**又调用一次** `_fill_mode`，同样从 `use_index = 0` 起步。
脚本：`/tmp/.../scratchpad/repro_drain.py`。local 模式、2 源（其中 1 源 border mask
→ 第 0 趟进 `sam3_queued`）、relabeler 修好掩膜、target=6、budget=3：

```
source_reuse = {"max_source_uses":3,"distinct_sources":2,"groups":4,
                "max_observed":3,"uses_histogram":{"1":1,"3":1}}
exhaustion   = {'local_shortfall': 2, 'global_shortfall': 0}
```

被 SAM3 救回来的那个源只拿到 **1** 次使用（应为 3）。
**这条不需要崩溃、不需要 resume，一次干净的 L8 正跑就会发生**，
损失 ≈ `(budget - 1) × 被 relabel 救回的源数`。

#### 修法建议（三选一，都很小）

1. 起始趟数从游标推导：`use_index = min(used.get(s.source_id, 0) for s in 非 terminal 源)`，
   或直接 `while use_index < budget` 每趟都走一遍（已走完的源在 `:2350` 自然跳过，
   代价是一次 O(pool) 的空扫，对 26k 源可忽略）；
2. 让 `_fill_initial_mode` 回报"本趟是否有源因**游标**被跳过"，
   护栏改成「无产出 **且** 无游标跳过」才收工；
3. 最小改动：把 `:2284` 的 `return` 改成"连续两趟无产出才 return"——**不推荐**，
   在复现 1 里第 1 趟同样无产出，仍会误停。

#### resume 的其余部分 —— pass

- **`_RESUME_NEUTRAL_DEFAULTS` 新条目**（`agent.py:471-482`）：机制本身
  （`_resume_config_differences`，`:499-513`）只在 `isinstance(section, dict) and key not in section`
  时补默认值，**只豁免"旧 manifest 缺 key"**。三种情形都有单测且我逐条复核：
  缺 key vs 新配置=1 → 无差异；缺 key vs 新配置=3 → 报差异；
  已写 1 vs 新配置=3 → 报差异。**范围正确，未过度豁免。** pass。
- **`completed_source_uses()` 由 groups.jsonl 重建的幂等性**（`state.py:372-386`）：
  纯 `self.groups.values()` 聚合，`groups` 由 journal 去重装载（`_load_unique`，主键 `group_id`），
  重复装载同一行不会双计。**幂等。** pass。
- **lost group 计入 use** 是刻意的且正确：group_id 已被占用，
  同一 `(source, use_index)` 再渲会撞 `append_group`。docstring 已写明。pass。
- 一个连带观察（**非本次引入**）：`_mode_groups` 走 `_live_groups()` 过滤 lost，
  而 `completed_source_uses()` 不过滤。两者口径不同是对的（一个数产能、一个数 id 占用），
  但意味着"资产全丢的一趟"既不计入 target 也不释放 use ——
  默认路径下这个行为改动前就存在，不定级。

---

## 3. ③ `group_id` 后缀方案 —— **pass**

**关键事实：后缀根本不改变 `group_id` 的字符串形状。**
`stable_id`（`state.py:36-38`）把所有 part 用 `\x1f` 连成 payload 后取 sha256 前 32 hex，
输出恒为 `group_<32 hex>`。`use_index` 只进哈希输入，不进输出格式。
所以"后缀撞解析正则"这类风险**在结构上不存在**。

全仓验证（排除 `trash/`）：

- 搜 `group_id` 的 `split/rsplit/partition/startswith/removeprefix` 与
  `re.match|search|fullmatch`：**0 命中**（唯二命中是 `q3vl/whereb/amort/model.py:364`、
  `pch_full.py:493` 的 `name.split(".")[0]`，参数名叫 `groups`，与 group_id 无关）。
- 搜在 construct 之外**重新推导** `stable_id("group"/"candidate"/...)` 的地方：
  只有 `legacy_import.py:184`（LegacyImportPipeline，不经 `_fill_mode`，NOTES 已列）
  与各测试文件。**没有下游会自己算 group_id。**
- **projection**（`projection.py:22-38 / 39-59`）：`canonical_groups` 主键 `group_id`，
  `source_id` 只有普通 INDEX（`:36-38`）；`canonical_candidates` 是 `UNIQUE(group_id, slot_index)`。
  新增的 `source_use_index` 落进 `payload JSONB`，**没有列白名单校验会拒绝未知键**
  （我确认 `state.py` 里没有 group 行 schema 校验）。pass。
- **S/P split**：`tools/data_splits/vr_common.py:47-62` 的 `s_split()` 是
  `sha1(f"{SPLIT_SEED}:{source_id}")` 的**纯函数**；`build_splits.py:69-96` 的
  `sources: dict[source_id -> pool]` 对同一 source_id 重复出现只会推出同一个 pool，
  `elif prev != pool` 永不触发。**同源多组不会跨 split 泄漏。** 复核 NOTES 结论属实。
- **`q3vl/data/splits.py`** 在切分前按 `source_id` 并组，同源多样本本就是它的设计目标。
- **viewer 的唯一行为变化**已复核属实（NOTES D3）：
  `databuild_viewer/backend/repository.py:192-216`（JSONL 路径，判据在 `:205-213`）与
  `:739-746`（PG `ff.group_id IS NULL AND ff.candidate_id IS NULL AND ff.source_id=g.source_id`）
  用 `source_id` 把**孤儿 failure**挂到 group 上；同源多组时一条源级失败会显示在该源
  **每一个** group 下。纯展示归因偏移，不丢数据、不崩。同意保守默认（不动 viewer）。
- 全仓 `UNIQUE(...source...)` / `PRIMARY KEY(...source...)`：只有
  `tools/data_splits/build_splits.py:60` 的 `sources` 旁表（按源一行，本就该如此）。

---

## 4. ④ 分配 bug 修复的作用域 —— **pass**

`allocate_sources`（`sources.py:506`）的分支条件是
`if max_source_uses == DEFAULT_MAX_SOURCE_USES:` → 走**逐字保留的老代码**；
`_reuse_prefix_labels`（`:458-492`）**只在 `else` 分支**被调用。
即"local 先吃满饿死 global"这个病在 `max_source_uses = 1` 下**照旧存在**，
且有成对单测把它钉死（`test_the_default_still_starves_the_second_mode_on_a_scarce_pool`
断言 40 源 / 100 目标 / 0.7-0.3 mix 下 `len(local)=40, len(global_)=0`，
并额外断言"显式传默认值"与"省略参数"两条调用产生相同分配）。
**没有污染默认行为。** pass。

`_reuse_prefix_labels` 的算术我按 L8 口径手算复核：
pool=37800、target=400000、mix 0.7/0.3 → `need = {local: ⌈280000/12⌉ = 23334,
global: ⌈120000/12⌉ = 10000}`，和 33334 ≤ 37800 → 走 `counts = need`，
余 4466 按 mix 分（3126/1340）→ local 26460、global 11340；
容量 26460×12 = 317520 ≥ 280000、11340×12 = 136080 ≥ 120000。**自洽**，
与 NOTES §六 / D2 的 26,460 一致。
L7 实配是 `local = 1.0 / global = 0.0`（`databuild.prod-l7-local400k-20260811.toml`），
此时 `need.global = 0`、稀缺分支的 `min(counts, need)` 封顶正确防止把源浪费给零权 mode，
已有单测 `test_a_zero_weight_mode_is_never_handed_sources_it_cannot_render`。

边界复核：`uses = 0` 时 `per_source = max(1, target_groups)` → `need` 每 mode 为 1，
绝大多数源变成 "replacement"。因为 `local_rows/global_rows` 是
`zip(ordered, [*prefix_labels, *replacement_labels])`（`:537-538`），
replacement 同样进池、同样保持 `ordered` 的场景分层序，**不影响轮转顺序**，
只是让 manifest 的 `replacement_capacity` 变得很大——NOTES §二.7 已就此加了旁注。可接受。

`target = 0` / 空池 / `budget = 0` 三种退化输入下 `_fill_mode` 都能终止
（`0 >= 0` 或"无产出"两个 return 之一），**无死循环**。

---

## 5. ⑤ C_GT / mask 复用决策 —— **pass**（2 nit + 1 风险提示）

**决策本身正确**：mask 种子只依赖 `source_id`（我实读确认：
`canonical_masks.py:245/270` 的 `_seed_int(build_id, seed, source.source_id, mode, index)`、
`:256` 的 `..., source_id, "semantic"`、`:291` 的 `..., source_id, "mask-pairing"`，
**均不含 `use_index`**），
所以同源各趟重新 plan 出来的 7 张物理 mask 与 `mask_id` **逐位相同**，
`_write_cgt_once`（`agent.py:189-202`）的 "adopt existing file" 返回的就是对的像素。
NOTES §二.4 拒绝给 mask 加 `use_index` 的三条理由（C_GT 失效契约、
`_sample_geometry` 重试可能让第二趟几何失败并触发一条被
`_pending_sam3_ids()` 静默吞掉的 relabel、多样性由 preset 提供）我逐条复核，**成立**。

**跨 batch 落地已实证**（交付的单测只覆盖同一 batch，我补测了跨 batch）：
`/tmp/.../scratchpad/repro_cgt_batches.py`，local 模式、3 源、budget=2、
`source_window=1` + `LAND_WATERMARK_BYTES=1` 强制多次 land →
**4 个 batch，每个 `verify_dataset(members) == len(metadata_rows)`，
21 个 cgt_path + 48 个 after_path 全部在归档里，缺失 0。**
第 2 趟遇到"C_GT 已 land 被 unlink"时按预期重新编码，遇到"还在"时复用，两条路都通。

### nit N3 — `_write_cgt_once` 的硬约束注释已经落后于代码
`agent.py:194-198` 写着：

> Today that holds because SAM3 relabel only ever runs after ``build_mask_plan``
> raised — that is, before ``_start_cgt_writes`` submitted anything for that source.
> Any change that lets an already rendered source be re-planned has to invalidate
> (delete) the affected C_GT files first, or this returns a stale mask.

**本改动正是"让一个已渲染的源被重新 plan"的改动**，而且**没有**删任何 C_GT——
靠的是"重新 plan 的结果逐位相同"这条**新的**豁免理由。
契约注释是这段代码唯一的规格，必须把第二条 re-plan 路径（source reuse）
及其安全性依据（种子不含 use_index ⇒ 重放确定性）写进去，
否则下一个人读到的规格与代码不符。**改注释即可，不改行为。**

### nit N4 — `source_reuse.max_observed` 没有硬校验
`_source_reuse_counts`（`agent.py:778-794`）的 docstring 说
"a bucket above ``max_source_uses`` would mean the cursor and the journal disagree"，
但没有任何地方检查它。我确认当前**不可能**超（`_try_ready_relabel` 只在
`count == 0` 的源上跑，`_fill_mode` 有 budget 门），所以只是纸面不变式。
建议在 `_manifest` 里加一行断言或一条 failure 事件，让这个不变式**自己会叫**。

### 风险提示（不是 blocker，给主 agent 定夺）：同源多组的下游耦合

同一张 `I_in` 现在最多产出 `max_source_uses` 个 group、共享**同一套 7 张 C_GT 区域池**。

- **不构成 split 泄漏**：S-split 是 `source_id` 的纯函数（§3 已证），同源必同 split。
- **但训练集冗余度上升到 12×/图**：Where 头面对的是「同一张图、同一个 7 掩膜区域池、
  12 条不同指令」。CLAUDE.md 红线要求每个消融行必带 `Δ_const/Δ_shuffle` 列——
  在这个分布下，"打乱指令仍能预测区域"的基线**会被结构性抬高**，
  因为图像→区域的映射被重复了 12 次。这不使开关本身失效，
  但 **L8 数据上训出来的 Where 头，其指令条件性判据必须在新分布下重新标定**，
  不能直接沿用 L7 数据上的 `Δ_shuffle` 门槛。建议写进 L8 的预注册。
- L_cube 监督从 `recipe.preset` 渲染，同源各组 preset 不同 ⇒ 该路无耦合。

---

## 6. ⑥ `preset_inventory_exhausted` 按源终态化 —— **pass**（1 nit）

机制复核属实：`_render_source` 试遍 `len(selector.majors)` 个 major 仍拿不到 8 个候选
→ `_terminal_source(..., "preset_inventory_exhausted")`（`agent.py:2095-2099`），
而 `_terminal_source_ids()`（`:809-815`）是 **per-source、跨趟共享**的，
所以第 7 趟的一次 exhausted 会让该源第 8..12 趟一并退休。

**偏差是真的，但它是可审计的**，三条独立记录：

1. `sources.source_reuse.uses_histogram`（`agent.py:790-793`）——
   早退休的源会落在低 bucket，分布倾斜一眼可见；
2. `failures.by_code["preset_inventory_exhausted"]` + `sources.terminal` 计数；
3. **`source_use_index` 落在 group 行上**（`:2028-2032`，缺失即 0），
   所以「按 pass 切片重算场景/池分布」是 groups.jsonl 上的一次 group-by，**事后可算**。

因此判 **pass**：D4 的"提示，不需裁定"定性正确，manifest 有审计记录。

### nit N5 — 建议把"按 pass 的池组成漂移"列进 L8 REPORT 的必看项
退休不是随机的（它与源内容相关：难渲出可见改动的图更容易 exhausted），
所以后几趟的池会系统性偏向"好渲"的源。既然 `source_use_index` 已经落盘，
建议 L8 的 REPORT 直接出一张
`source_use_index × {scene, pool, mask_area 分位}` 的表，
而不是只报一个 histogram。这属于实验设计建议，不阻塞合入。

---

## 7. ⑦ 测试有效性 —— **BLOCKER B2**（3 nit）

### 覆盖度核对（对 §0 的 1-6 项）

| 审阅项 | 被 29 例覆盖？ |
|---|---|
| ① 默认等价 | **是**，且是成对写法（4 例，含 `stable_id` 对拍与"显式默认 == 省略参数"） |
| ② resume | **仅覆盖"已完成的 build 重跑不重渲"与三条 config 比较**；**未覆盖中途重启** → B1 从这个洞里漏过去 |
| ③ group_id 全链路 | 覆盖 construct 内（32 个互异 candidate_id、land 可寻址）；construct 外靠 NOTES §三.B 的扫描，我已独立复核（§3） |
| ④ 分配修复作用域 | **是**，成对（关=老病照旧 / 开=两 mode 都够） |
| ⑤ C_GT 共享 | 覆盖同 batch；**跨 batch 未覆盖**，我补测通过（§5） |
| ⑥ D4 终态化 | **未覆盖**（无 exhausted-mid-reuse 用例）；属"现有语义不改"，可接受 |

### 变异测试复核（抽查三处，用影子包 `PYTHONPATH` 覆盖，未改工作树）

| 改回旧写法 | NOTES 声称 | 我实测 |
|---|---|---|
| `_fill_mode` 的 `budget` 硬编码为 1 | 11 failed | **13 failed / 20 passed** |
| `group_id` 去掉 `use_index` 后缀 | 8 failed | **10 failed / 19 passed** |
| `_chunk_paths` 换回 `completed_sources()` 成员判定 | 2 failed（`not found in []`） | **2 failed**（`SUBFAILED(source=source-0.jpg / source-1.jpg)`，形状一致） |

**三处都真的会咬**，结论方向正确。但两处的数字对不上（见 nit N7）。

### B2（blocker）"同源两组 preset 交集为空 / major 也不同"是玩具目录的产物，不是机制保证

NOTES §一 把它写成机制事实：

> 同源两组拿到的 major/minor/preset 由「第一遍已经消耗掉哪些」决定，**天然不同** ——
> 不需要改种子，只需要不让它们撞 id。

单测 `test_the_repeat_pass_draws_a_completely_different_preset_set`
（`test_source_reuse.py:427-450`）断言 **8 个 preset 交集为空 + major 不同**。
但它跑在交付**专门为此重写的** catalog 上（`:313-339`）：
2 major × 4 minor × 4 preset = 每 major 16 个 preset，一组吃 8 个，
**第二趟不换 major 就凑不齐**——**disjoint 是被目录规模逼出来的，不是被机制保证的**。

**真实 bank（L7 manifest 实测，`/mnt/ramstage/prod-l7-local400k-20260811/manifest.json`）：
`{"majors": 10, "minors": 85, "presets": 3522}`。**

1. **"major 也不同"在 L8 口径下被鸽巢原理直接判死**：
   `max_source_uses = 12 > 10 majors`，同源必有至少两组共用 major。这条不需要实验。
2. **preset 重复是可测的、且不小**。我用 8 major × 4 minor × 8 preset = **256 preset**
   的目录、60 源、6 趟、360 group 实测（`/tmp/.../scratchpad/repro_preset_dupes.py`）：

   ```
   TOTAL duplicate (source, preset) draws: 214 of 2880      (7.4%)
   exact identical 8-preset sets: 0
   pairwise preset overlap: mean 0.25, max 6/8, nonzero 105/900 pairs
   sources reusing a major across passes: 96 groups /360
   ```

   即：**任务卡那句"同源不同 group 必须渲出不同 preset 集合"字面上成立**
   （exact 重复集合 = 0），**但两组共享最多 6/8 个 preset 的情况真实发生**，
   且 7.4% 的抽取是该源已经渲过的 preset。

3. **重复 (source, preset) 的后果**：global 模式下 = 同一张 `I_in` + 同一条 LUT
   = **逐位相同的 `I_tar`**，只是换了个 `candidate_id`/`sft_id`。
   `sft_pack.py:69-100`（`load_sft_rows`，`seen: set[str]` 按 `sft_id`）与
   `q3vl/data/scan.py:39-76`（按 `sample_id` → `sft_id` 建表）我都实读确认是**按 ID 去重**，
   **都抓不到内容重复**。local 模式下还要同槽位才完全重合，概率低但非零。

**为什么定 blocker 而不是 nit**：交付把一条**未在目标口径验证过的性质**当成已证事实
写进设计文档并用一个regime-locked 的单测背书，而这条性质正是任务卡列的三项规格之一
（"同源不同 group 必须渲出不同 preset 集合"）。这与 CLAUDE.md
「任何"真实档打不过 oracle 档"的结论必须先出示消费侧断言」同源：**声明的性质要有
在目标域上的证据**。

**清 blocker 的最小代价（不必改算法）**：
- 把 `source_reuse` 里加两个统计量：
  `duplicate_source_preset_pairs`（同源重复抽到的 preset 次数）与
  `max_pair_overlap`（同源两组的 preset 交集上限）——两者都是 groups.jsonl 上的一次聚合；
- NOTES §一 与那条单测的 docstring 改成"在 preset/major 数远小于 uses 时会重复，
  实测重复率 X%"，把 disjoint 的断言限定在该目录规模内（或干脆把断言从
  "交集为空"降为"两组不是同一个集合"，那才是任务卡要求的）；
- L8 开跑前用真实 bank 做一次小规模 dry run，把上面两个数报出来。

### nit N6 — 缺两条关键回归测试
B1 的两个形状都没有测试：(a) 中途重启后继续走剩余趟数；
(b) `_drain_sam3_and_replacements` 二次调用 `_fill_mode` 后被 relabel 救回的源拿满 budget。
我的两个复现脚本（`repro_resume.py` / `repro_drain.py`）可以直接改写成用例。

### nit N7 — NOTES §五 的变异测试数字不可复现，且"四处"只列了三行
声称 11 / 8 / 2，实测 13 / 10 / 2（复核方法见上，影子包整树 cp，无其它改动）。
数字本身不影响结论，但"实测"标注的数应当可复现；另 §五 正文写"四处"、表里只有三行。

### nit N8 — `MINIMAL_TOML` 与 `databuild.example.toml` 的关系需要在恢复后收敛
`test_source_reuse.py:47-112` 自带一份完整 TOML 常量。若采纳 D1 恢复 example，
这份常量应改成读该文件（NOTES 已自陈），否则又多一处需要同步的配置真值。

---

## 8. ⑧ 协议红线扫描 —— **pass（全部无涉）**

对 `dataset_build/src/construct/` 的 diff（+252 行）逐条 grep
`auc|roc|min-?max|softmax|normali[sz]|val_loss|checkpoint selection|smooth|resize`：**0 命中**。
逐项确认：

| 红线 | 是否被牵动 |
|---|---|
| AUC 禁作空间场判据 | **无涉**（本改动不产出任何指标） |
| 空间场可视化（禁逐图 min-max / 色标只取有效格 / 叠图禁 resize） | **无涉** |
| 掩膜协议（`C_GT` = 逐候选区域掩膜、软边单通道；`L_cube` 从 `recipe.preset` 渲染） | **无涉**：mask 种子、`save_cgt_png`、`recipe` 结构一字未动；C_GT 只是被**共享**给同源多组，像素与协议不变 |
| s 轴 / scache 消费契约 / 逐图归一化 | **无涉** |
| checkpoint 选择禁用 val loss | **无涉** |
| G 初始化 = 0、σ 有界 sigmoid、逐像素算子约束 | **无涉** |
| 数据纪律：S/P split 不得 ad-hoc、同源不跨 split | **满足**（§3 已证 S-split 是 `source_id` 纯函数） |
| `winner_confidence=low` 不进 SFT 主训 | **无涉**（排序与 margin 逻辑未动） |
| `eval100-annotqa-20260727` 永不进训练 | **无涉** |

---

## 9. 对 NOTES 三项待决策的独立意见

- **D1（`databuild.example.toml` 被 83da846 误删）—— 复核属实，建议恢复。**
  `git show 83da846 --stat` 确认该 commit 一次删了 `LICENSE` / `utils.py` /
  `databuild.example.toml`。当前树 `pytest dataset_build/tests/` = **30 failed / 403 passed**，
  30 条全是 `FileNotFoundError: /home/bc/VeraRetouch/databuild.example.toml`，
  分布在 `test_canonical_foundation` / `test_iaa_batch` / `test_source_window_and_cgt` /
  `test_winner_margin` 四个文件的 ConfigTests。
  **它不是上一战役遗留，是当前 canonical databuild 的配置模板兼测试夹具。**
  同意恢复，并在 `[sources]` 段补 `max_source_uses` 的注释与默认值
  （其余四个可选 key 都在该文件有注释，本 key 目前**无处落脚**——
  这意味着 L8 的操作者没有任何文档能看到这个开关存在）。
- **D2（O(groups × sources) 记账）—— 同意保守默认不动，但 B1 的修法会加一点账。**
  若采纳 §2 修法 1（每趟都走一遍全池），L8 增加约 `12 × 26460` 次跳过判定，
  每次一个 `completed_source_uses()`（NOTES 实测 29.2 ms @200k groups）——
  这**会**让 D2 从"1.5% 开销"变成需要认真对待的量。
  建议 B1 的修复顺手把 `completed_source_uses()` 提到循环外算一次快照
  （一趟之内它只在 `finish_oldest` 后变化，而跳过判定发生在提交之前，
  用趟首快照 + 本趟已提交集合即可，仍然精确）。这条与 D2 的缓存卡可以合并成一张。
- **D3（viewer 孤儿 failure 归因）—— 同意保守默认不动。** 纯展示、不落盘、不影响判据。

---

## 10. 范围外观察

工作树里与本任务**无关**的未提交改动：
`dataset_build/core/responses_vlm.py`（+65）、
`dataset_build/source_qa/sam3_subject_instances.py`（+46）、
`dataset_build/tools/eval_subject_instance_selector.py`（+42）。
均不含 `max_source_uses` / `use_index`，应属另一条任务线，**不要与本改动同批 commit**。

本次唯一改到的既有测试文件 `dataset_build/tests/test_source_window_and_cgt.py`
（两个 spy 改成 `**kwargs` 透传，断言一字未动）**合理，pass**。

---

## 11. 一轮放行条件（已全部满足，判定见 §12）

1. **清 B1**：`_fill_mode` 的起始趟数从 use 游标推导（或护栏区分"游标跳过" vs "池子拒绝"），
   并补上 §7 nit N6 的两条回归测试（中途重启 / SAM3 救回的源拿满 budget）。
   → **两条都做了**（实现方两条一起上），§12.1。
2. **清 B2**：`source_reuse` 增加 `duplicate_source_preset_pairs` 与 `max_pair_overlap`；
   NOTES §一 与 `test_the_repeat_pass_draws_a_completely_different_preset_set` 的
   disjoint 主张限定到目录规模，或降为任务卡实际要求的"集合不相同"；
   L8 开跑前用真实 bank（3522 preset / 10 major）做一次 dry run 把这两个数报出来。
   → **三项都做了**，§12.2。
3. 建议同批清掉 N3 与 D1。→ N3 **已修**（§12.4）；D1 **未做**，仍是 §12.6 的建议项。

---

## 12. 二轮复核（终版判定）

> 复核依据：修复后的工作树 diff（`construct/` 由 +252 行增至 **+283 行 agent.py** 等，
> 共 +523/-57）。行号为**修复后**工作树。
> 复核方法：一轮的复现脚本原样重跑 + 我自己新写的边界扫描 + 变异抽查 + 性能实测 +
> dry run 全量重算。**没有只读实现方的自述。**

### 12.1 B1 —— **已清（pass）**

修法采纳了我建议的 1 + 2 两条，且**两条一起上**，`agent.py:2333-2390`：

```python
used = self.store.completed_source_uses()
use_index = min((used.get(row.source_id, 0) for row in sources), default=0)
if budget:
    use_index = min(use_index, budget - 1)      # ← 实现方自抓的越权 bug
while True:
    final = bool(budget) and use_index + 1 >= budget
    before = None if final else len(self._mode_groups(mode))
    cursor_skips = self._fill_initial_mode(mode, sources, target, use_index=use_index)
    use_index += 1
    if final: return
    if len(self._mode_groups(mode)) >= target: return
    if len(self._mode_groups(mode)) <= before and not cursor_skips: return
```

`_fill_initial_mode`（`:2393-2470`）改为返回 `cursor_skips`，且**只把游标跳过计数，
terminal / SAM3-pending 明确不计**（`:2469-2474`）—— 这正是"拒绝 vs 延后"的分界。

**(a) 一轮的两个复现脚本，原样重跑（未改一个字符）**

| 脚本 | 一轮（坏） | 复核（好） |
|---|---|---|
| `repro_resume.py`（第 1 趟后崩溃 → 重启） | 4 组、`complete_with_failures`、`global_shortfall: 2`、**0 新增** | **6 组、`complete`、shortfall 0**、`uses_histogram {"3": 2}`、use 序列 `[0,0,1,1,2,2]` |
| `repro_drain.py`（SAM3 救回的源） | `uses_histogram {"1":1,"3":1}`、`local_shortfall: 2` | **两源各 3 次、shortfall 0** |

**(b) clamp 的边界我自己扫了一遍**（`scratchpad/probe_clamp.py`，把
`:2364-2373` 的算术逐字抄出来 + 复刻整个 `while` 控制流）：

- **`budget = 0`（不限）不会 clamp 到 −1**：`if budget:` 是假，整段跳过。
  实测 `budget=0, used=[13,13] → start=13`，**无负值**。协调者问的这条**不成立**（安全）。
- `budget=1` 在任何 `used` 下都 `start=0`（`used=[13,13] → 0`）——
  默认路径永远不会打开一趟它没有的额度。
- `budget=12, used=[12,12] → 11`、`used=[13,13] → 11`（clamp 生效）。
- **穷举扫描 225 个配置**（budget ∈ {1,2,3} × used ∈ [0,4]² × live_fraction ∈ {0,0.5,1}）
  ＋ 7 个命名场景（含"全部 group 丢失"、"参差 + 半数丢失"、"terminal 型源卡在 0"）：
  **0 个源越 budget、0 个重复 `(source, use_index)`、0 个不终止**。
- **不限模式的终止性**单独扫过（含"resume + 全部 group 丢失"）：均终止。
  机理：产出为 0 时 `used` 不变，而 `use_index` 每轮 +1，
  必然在 `max(used)` 轮内让 `cursor_skips` 归零并撞上护栏。

**(c) clamp 是**默认路径**的安全件，不只是 reuse 的**。删掉 clamp 后
`test_a_lost_group_never_buys_a_source_an_extra_use`（`max_source_uses = 1`）会失败 ——
即：一个 group 落盘后资产丢失的默认 build，在没有 clamp 时会**自己把开关打开**，
给该源发第二个 group。这个 bug 一轮我没抓到（我的 clamp 分析当时不存在），
实现方自查抓到并配了两条专项测试。**加分项。**

**(d) 游标单调性**（B1 修复正确性的地基，我独立验证）：
`ArtifactStore.groups` 只在 `_load_unique` 时整体赋值（`state.py:214`），
全仓**没有任何 `groups.pop / del / clear`**；丢失的 group 留在 `groups` 里，
只经 `lost_group_ids()`（`:355-367`，从 failure 事件派生）被 `_live_groups` 减掉。
所以 `completed_source_uses()` **只增不减** ⇒ 一个源不可能被重新发放已花掉的
`use_index` ⇒ 不可能撞 `group_id`。

**(e) 默认路径未被修复破坏**（重点复查，因为修复动了 `_fill_initial_mode` 内部）：
`used` 从"每源一次 live 扫描"改成"每趟一次快照"。等价性证明：
`sources` 内 `source_id` 唯一（`_require_unique_source_ids`），
所以某源的计数只可能在**它自己那一轮之后**变化，绝不会在读它的那次判定之前变化；
`_chunk_paths` 只看当前位置及之后的源，同理。
落盘产物逐字不变，且默认路径的扫描次数从 `N × completed_sources()`
降到 **2 次**（一轮 nit N1 顺带被超额修掉）。

### 12.2 B2 —— **已清（pass）**

**(a) 真实 bank dry run 我全量重算了一遍**
（`scratchpad/dryrun_real_bank.py`，我先审了脚本：它 `begin_group` →
8 × `reserve_candidate`/`accept` → `commit`，失败换 major 重试，
与 `_render_source` 对选择器的调用序列一致）：

```
inventory (local ): presets=3522 majors=10 minors=85
allocation        : pool=26460 target=317520 uses=12
drew              : 317520 groups (0 refused) in 268.4s

(source, preset) draws        : 2540160
duplicate_source_preset_pairs : 133318
duplicate_source_preset_rate  : 5.2484%
max_pair_overlap (of 8)       : 7
exact identical 8-preset sets : 0
pairwise overlap mean         : 0.084
pairs with any overlap        : 87079/1746360
```

**与 NOTES §七 的每一个数逐位一致**（5.25% / 7 / 0 / 0.084 / 87,079）。
200 源口径也复算了：1004 / 19200 = **5.2292%**、max overlap 5、exact 0、647/13200，
同样逐位命中。**⑤ 复算通过。**

**(b) 三件补救都落地了，且我确认它们不是摆设**：

| 补救 | 复核 |
|---|---|
| manifest 三列 + `budget_exceeded` | `agent.py:789-861`。我在 `repro_resume.py` 的输出里看到它们真的在动：`duplicate_source_preset_pairs: 10, rate: 0.208333, max_pair_overlap: 6` |
| 单测主断言降为"两组不是同一个集合" | `test_source_reuse.py:434-482`。disjoint 与 major 两条断言**保留但打上 `regime="2x16 toy catalog"` 标签**，并在注释里写明鸽巢原理。这比删掉更好——它把"这个结论只在这个 regime 成立"钉在了代码里 |
| 退化目录用例证明三列会动 | `test_the_redundancy_columns_count_a_real_overlap`（`:498-527`）：8-preset 目录 → 断言 `16 / 0.5 / 8`。**没有这条，那三列等于没测** |

**(c) NOTES §一 的措辞修正是诚实的**：它自己写明"这句话是错的"、
给出真正的保证（`GroupReservation.commit` 的 `len(set(preset_ids)) != 8`，
我核到实际位置是 `presets.py:563-564`，NOTES 写 `:549-552` 差了十几行，见 N10）、
写明鸽巢原理判死 major、并列出下游按 ID 去重抓不到内容重复的行号。

### 12.3 变异表抽查（③，抽两行，均命中）

| 改回 | NOTES 声称 | 我实测 |
|---|---|---|
| 删掉 `use_index` 的 `budget - 1` clamp | 2 failed：两条 lost-group 用例 | **2 failed**，且正是 `test_a_lost_group_does_not_push_a_reuse_build_over_budget` + `test_a_lost_group_never_buys_a_source_an_extra_use` |
| B1 只回退「游标感知护栏」 | 1 failed：`test_the_walk_resumes_past_a_source_that_can_never_render` | **1 failed**，同一条 |

一轮 nit **N7 已清**：新表的数字**可复现**，且 NOTES 在表下加了澄清框
解释初稿 11/8/2 与我实测 13/10/2 的差异来源（测试集从 29 例变 39 例 + 影子包收集口径）。

### 12.4 一轮 8 个 nit 的处置复核

| nit | 处置 | 我的复核 |
|---|---|---|
| N1 默认多一次 `_mode_groups` | 已修（`final` 时 `before = None`） | **确认**，`agent.py:2377-2378`；且默认路径总扫描数反而从 N 次降到 2 次 |
| N2 `_try_ready_relabel` 多一次扫描 | 不改 | **同意**。理由与我自己的定级一致（恒为 0、属 D2 同一笔账、换成常量会让隐患回来） |
| N3 `_write_cgt_once` 契约注释 | 已修 | **确认且质量很高**（`agent.py:189-210`）：列出**两条** re-plan 路径各自的安全性依据，并写死"给 mask plan 任何 per-pass 熵会同时打破两条论证" |
| N4 `max_observed` 无硬校验 | 已修 | **确认**，`source_reuse.budget_exceeded` 布尔位（`:854`） |
| N5 L8 REPORT 按 pass 切片 | 采纳为 L8 设计建议 | 同意（非本卡代码） |
| N6 缺 B1 两条回归测试 | 已加，另加 4 条 | **确认**：崩溃重启 / SAM3 救回 / 两个机制各一条 / 两条不终止防护 |
| N7 变异数字不可复现 | 已修 | **确认**，见 §12.3 |
| N8 `MINIMAL_TOML` | 不改 | **同意**（当前 example TOML 根本不存在，改了测试就跑不了） |

### 12.5 复核新增的 2 个 nit（均不阻塞）

**N9（nit）「最小值起步」这个优化在 L8 上基本不会生效，resume 会空转数小时**

`use_index = min(used.get(sid, 0) for sid in sources)` 的 min **取遍整池，
包含永久 terminal 的源**。terminal 源永远 0 组 ⇒ min 恒为 0。

- **L7 实测有 898 个 terminal 源**（manifest `sources.terminal: 898`，
  全部 `sam3_relabel_failed`，即 `build_mask_plan` 抛错、从未渲染 ⇒ 0 组）。
  L8 同形状 ⇒ 这个优化**从第 0 趟起就失效**。
- 代价我实测了：`_mode_groups` 在 317,520 组上 **129.9 ms**，
  而它在 `_fill_initial_mode` 里是**每个源调用一次、且在游标判定之前**
  （`agent.py:2449`）⇒ 一趟空走 = 26,460 × 129.9 ms ≈ **57 分钟**；
  在第 6 趟重启 ⇒ **约 5.7 小时的纯记账**才会渲出第一张图。
- **不是正确性问题**（护栏兜住了，`test_the_walk_resumes_past_a_source_that_can_never_render`
  正是钉这个场景的，实现方的 docstring 也明说"该优化在此提供不了任何帮助"）。
- **一行可修**：min 时减去 `self._terminal_source_ids()`。
  建议与 D2 的记账缓存合成一张卡，在 L8 第一次重启之前处理。

**N10（nit）NOTES §一 的 `presets.py:549-552` 行号指错**
实际的 `len(set(preset_ids)) != 8` 检查在 `presets.py:563-564`（`commit` 内），
`:549-552` 落在 `accept` 里。结论不受影响，改个数字即可。

另附一条**方法学备注**（不计 nit）：dry run 脚本对每个 slot 都直接 `accept`，
不走真实 `_render_source` 的 `reservation.reject`（`agent.py:1798`，可见性门失败重抽）。
真实 build 每组会消耗更多 preset，对重复率的影响方向不显然。
5.25% 应读作"同一抽取算法下的量级"，而非精确预测 —— NOTES 未声称更多，可接受。

### 12.6 ④ manifest 统计开销 —— **可接受**

我在 L8 形状（26,460 源 × 12 趟 = 317,520 组、**2,540,160 条候选**）上实测
`_source_reuse_counts`：

| 数据分布 | 三次取最小 / 中位 |
|---|---|
| 无重复 | 1.95 / 2.38 s |
| ~5% 重复（真实 bank 形状） | 1.92 / 2.02 s |
| 饱和（提前命中 8/8，二次项提前退出） | 1.76 / 1.78 s |

**与实现方声称的 2.2–2.4 s 一致**（我的中位落在 1.8–2.4 s）。

写入频率：`_manifest()` 在 `_land_checkpoint` 的 mirror 分支（`:1524`）
与 `_write_phase`（`:1116`）被调。L7 实测 **13 次 land checkpoint / 27,291 组**
（`LAND_WATERMARK_BYTES = 8 GiB`，约 2,100 组一次）
⇒ L8 的 317,520 组约 **151 次**。

**151 × ≤2.4 s ≈ 6 分钟**，且早期 checkpoint 的组数少、成本按比例更低（实际约 3 分钟）。
对照 L8 的周级 GPU 时间 ⇒ **< 0.1%**。而且 `_manifest()` 本来就已经有多次
O(groups) 与 O(groups × candidates) 的聚合（`_mode_groups` 单次就 130 ms、
preset usage 要遍历全部候选），这两列不是新的主导项。**判定：可接受，不需优化。**

### 12.7 回归复测

| 项 | 结果 |
|---|---|
| `test_source_reuse.py` | **39 passed / 32 subtests** |
| `dataset_build/tests/` 全套 | **30 failed / 413 passed**；30 条逐条确认全是 `databuild.example.toml` 缺失（D1，先于本改动） |
| 跨 batch C_GT（我一轮补的 `repro_cgt_batches.py`，原样重跑） | **通过**：4 个 batch、`verify_dataset` 全绿、21 cgt + 48 after 全在归档、缺失 0 |
| `③ group_id` 全链路 | 未受修复影响（`stable_id` 形状不变、无解析点） |
| ⑧ 红线 | 修复后的 diff 再次 grep `auc\|roc\|min-?max\|softmax\|normali[sz]\|val_loss\|smooth\|resize`：**0 命中** |

### 12.8 终版放行判定

**PASS。0 blocker。** 开关可以用于 L8。

放行时建议（都不阻塞，按优先级）：

1. **D1 恢复 `databuild.example.toml`** 并在 `[sources]` 段写上 `max_source_uses` 的
   注释与默认值 —— 这是唯一一件**会影响 L8 操作者**的事：现在没有任何配置模板
   能让人发现这个开关存在，而 30 条既有测试也还是红的。
2. **N9 + D2 合成一张卡**（terminal 源从 min 里剔除 + 记账缓存），
   在 L8 第一次重启之前处理；不处理的后果是重启后数小时空转，不是数据错误。
3. **D5 写进 L8 预注册**：同图 12 组共享一套区域池，`Δ_shuffle` 门槛
   **不得沿用 L7**，须在新分布上重标。这条是实现方从我一轮的风险提示里
   自己提炼出来的，方向正确。
4. **N10** 改一个行号。
5. 分批 commit：`core/responses_vlm.py`、`source_qa/sam3_subject_instances.py`、
   `tools/eval_subject_instance_selector.py` 属另一条任务线（NOTES §八 已自陈），
   不要与本改动同批。

---

# ⚠ 事故：本文件在 §16 审阅期间被并发删除（2026-08-13 17:22）

**这份文件不完整。** §16 审阅进行中（17:17 取快照，17:39 完成实验）时，整个 `docs/`
树被一个并发进程删除，只剩 `docs/assets` 与 `docs/papers_extracted`。本文件与
`docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md` 从未进入过 git 索引（`git ls-files` 零命中），
因此**无法从提交历史恢复**。

已做的恢复与其边界：

| 段落 | 状态 | 来源 |
|---|---|---|
| §0-§12 | **逐字节完整** | 悬空 blob `9d800fcd37e7f5037238576bea04c58f1f0448f8`（44,973 B / 719 行），曾被 `git add` 过 |
| §13（Sam3Manifest-1）、§14（drain 账本） | **永久丢失** | 无 blob、无提交、无编辑器备份；仅存标题（见下） |
| §15（Cadence-3） | **由本轮审阅 agent 的读取缓冲原样回填** | 本 agent 在 17:18 完整读过 1177-1513 行，下方为原文；非重写、非摘要 |
| §16 | 本轮新增 | —— |

丢失段落的标题（供重建时定位）：§13.1-13.8 + §13 nit 清单；§14.0-14.11（① 增量正确性的地基 /
② 两处过滤差异 / ③ 每次读之前 refresh / ④ round-count 断言 / ⑤ 变异测试 / ⑥ 回归复核 /
⑦ oracle 与基线 / ⑧ 性能自证 ×5,086 / ⑨ 红线扫描 / 7 条 nit）。

`docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md` 同样被删。悬空 blob
`30108ce82b3e9af8c660c494b4e2bdb32a05a3ef`（53,962 B / 804 行）可恢复到
`HOTFIX-Sam3Manifest-1` 为止；**§D（drain）/§M（Cadence-3）/§W（Watermark-4）/§S（MirrorScrub-5）
四节丢失**。两份 blob 已另存到本轮 scratchpad 的 `recovered/`。

**这是本审阅链第四次遭遇工作树并发改写**（§13.7 / §14.9 / §15.8 是前三次，都只是改写；
这次是删除）。建议主 agent：(1) 把 `docs/reviews/` 与 NOTES 纳入 git 跟踪，别让审阅底账
只活在工作树里；(2) 查清 17:22 那次 `docs/` 删除是谁做的、是否还有别的未跟踪产物一起没了
（`docs/` 下大量**已跟踪**文件也处于删除状态，可用 `git checkout -- docs/` 恢复，但那是主
agent 的编排决定，本 agent 未执行）。

---
## 15.0 结论

# ✅ **PASS —— 0 blocker，可用于 L8 重启**

| 审阅项（对应任务卡） | 判定 | 依据 |
|---|---|---|
| ① 恢复语义：两个声称亲手验证 + `restore_mirror` 同时杀两种截断 | **pass** | §15.1，两个声称**全部成立**，并跑通 4 条 crash→restore→resume→再镜像全环 |
| ② 四重防御逐个触发 + 中断追加回滚 | **pass** | §15.2，我自己手工触发 7 个分支，全部落到「事件已记 + 字节最终一致」 |
| ③ 写序 / fsync / NFS 语义 | **pass（NOTES 的危害论述有误，代码更安全）** | §15.3，实测反序**不会**造成空洞，被防御 ② 兜住并回落整拷 |
| ④ 节流门四维真值表 / 播种 / tmpfs 余量算术 | **pass** | §15.4，32 行真值表复跑 + 3.87 MB/组实测 + 窗口最坏 ≈ 1.0 GiB |
| ⑤ 迁移：无 state 的存量镜像首次即追加 | **pass** | §15.5，用**真实 L8 的 397 MB 字节**彩排两遍，逐字节 sha256 对拍相等 |
| ⑥ 变异 10/10 复跑（我全部重写）+ 全套回归 | **pass（1 个变异的表述需修正）** | §15.6，9/10 被杀；第 10 个我的写法**存活**且证明它本就无害 |
| ⑦ 协议红线扫描 | **pass（全部无涉）** | §15.7 |
| **合计** | **0 blocker / 9 nit** | 全部不阻塞 |

**一句话**：增量镜像在 7 个损坏分支上都收敛到「镜像 = 源的字节前缀」，`restore_mirror`
的换行边界截断堵住了 `scan_jsonl` 唯一看不见的那一刀（我用反变异体实测到**下一次
resume 直接 StateError 起不来**），节流门的真值表与 force 豁免与播种全部符合规格，
真实 L8 字节的迁移彩排零整拷。**可以放重启窗口。**

最需要主 agent 知道的一条不是 blocker 而是**语义降级**：旧的整拷让镜像**每次
checkpoint 自愈**，新方案对 64 KiB 窗口之外的篡改**既检测不到、也永不修复**
（§15.9 N-M1，附实测）。建议加一个「每 K 次 checkpoint 整拷一次」的擦洗，代价可算。

---

## 15.1 ① 恢复语义 —— **pass**（两个声称亲手验证）

**声称 A：`scan_jsonl` / `JsonlJournal` 本就 truncate-repair 坏尾。** 成立。
读码：`state.py:70-72` 仅当 `end == size` 才把不可解析行当 torn tail 返回，否则
`raise StateError`；`state.py:93-97` 开局 `scan.torn_tail` 则 `truncate(valid_bytes)`。
实测（scratchpad `v1_journal.py`）：

```
A  scan: 3 records torn=True valid=24 size=35   → 开 journal 后 size=32，4 条记录，干净
A2 中段坏行 → StateError: malformed non-tail JSONL record at …:3
```

**声称 B：「缺换行的完整记录」会被下次 append 粘连成中段损坏。** 成立，且比 NOTES
写的更精确——**要两次 append 才引爆**：

```
B  scan: 4 records torn=False valid_bytes=31 size=31     ← 缺换行那条被当正常记录收下
   append 一条 → b'…{"i":3}{"i":4}\n'  → 重扫 3 条、无异常（粘行恰在文件末尾，被当 torn tail 吞掉）
   再 append 一条 → b'…{"i":3}{"i":4}\n{"i":5}\n' → StateError: malformed non-tail … :4
```

即：第一次 append 把 `{"i":3}` 与 `{"i":4}` **两条一起静默作废**（下次开 journal 时被
truncate 掉），第二次 append 才把坏行推到中段、**让整个 build 起不来**。两种后果都成立，
NOTES §M-三 只写了后者。

**声称 C：`restore_mirror` 的换行截断同时杀掉两种情况。** 成立。我跑的是**全环**
（crash → `restore_mirror` → 真 `ArtifactStore` resume 追加 5 组 → 再 `mirror_artifacts`
→ 再 resume），四种尾巴形态：

| 镜像尾部 | restore 出的记录数 | torn | resume 读到 | 再追加后 | 再镜像事件 | 镜像==源 | 第二次 resume |
|---|---|---|---|---|---|---|---|
| 完好 | 20 | False | 20 | 25 | `[]` | True | clean 25 |
| 不可解析半行 | 20 | False | 20 | 25 | `interrupted_append` | True | clean 25 |
| **完整记录缺换行** | 20 | False | 20 | 25 | `interrupted_append` | True | **clean 25** |
| 2 整行 + 半行 | 22 | False | 22 | 25 | `interrupted_append` | True | clean 25 |

**反变异体对照**（把 `limit = _whole_records_limit(...)` 改成 `limit = None`，其余不动）：
前两行照旧过，**第三行在 resume 之后立刻**
`StateError: malformed non-tail JSONL record at …/resumed/groups.jsonl:21`。
这一刀是**必需**的，不是防御性冗余。

顺带核实 `_whole_records_limit` 的三个边界：文件以 `\n` 结尾 → 读 1 字节返回 `None`
（零成本）；空文件 → `None`；**通篇无换行** → 返回 `0`，`_copy_stream(limit=0)` 的
`while 0 < 0` 不进循环、写出空文件（正确：整份镜像只有半条记录时，恢复出的就该是空账本）。
JSONL 记录里不可能出现裸 `\n`（`json.dumps` 恒转义，与 `ensure_ascii` 无关），
所以「最后一个换行」就是「最后一条完整记录的边界」。

---

## 15.2 ② 四重防御逐个触发 —— **pass**（我手工触发 7 个分支）

不看测试、自己搭台（scratchpad `defense.py`，400 条 ×~320 B 的账本 + 独立
`_atomic_copy` 探针），逐个把不变量打破：

| 分支 | 触发手法 | 记录的 reason / action / journal | 整拷了谁 | 结束时镜像==源 | state.bytes==镜像大小 |
|---|---|---|---|---|---|
| 源短于 offset | 源被改写成 10 条 | `source_shorter_than_mirror` / full_copy / True | groups | ✅ | ✅ |
| 镜像短于 offset | 镜像 truncate 到 500 B | `mirror_truncated` / full_copy / True | groups | ✅ | ✅ |
| 尾部字节不符 | 窗口内翻一个 bit | `prefix_diverged` / full_copy / True | groups | ✅ | ✅ |
| **sha 不符** | 只篡改 state 里的 `tail_sha256` | `mirror_tail_changed` / full_copy / True | groups | ✅ | ✅ |
| 镜像不存在（有 state） | 删镜像文件 | `mirror_missing` / full_copy / True | groups | ✅ | ✅ |
| 首次镜像（无 state） | —— | 不记事件（`journal=False`） | groups | ✅ | ✅ |
| **中断追加（半行）** | 往镜像尾巴写半条 | `interrupted_append` / **rollback** / True | **只有 manifest** | ✅ | ✅ |
| **中断追加（整行）** | 往镜像尾巴写 2 整条 | `interrupted_append` / **rollback** / True | **只有 manifest** | ✅ | ✅ |

「中断追加」两例都**没有**整拷（`copies == ['manifest.json']`），确认是 truncate+重追
而不是偷偷退化成整拷 —— 这正是任务卡点名要构造的那条。

`mirror_state_unusable` 我另按 `True / 1.5 / "12" / None / -1` 五种偏移各跑一遍，
全部整拷（`bool` 是 `int` 这条陷阱确实被 `isinstance(..., bool)` 挡住了；变异 M8 去掉
它 → 2 例 FAIL）。

**边界确认（不是缺陷，是已写明的局限）**：把 320 KB 镜像的**第 100 字节**改掉，
64 KiB 窗口够不着 → `events=[]`、镜像与源**永久不一致**。见 §15.9 N-M1。

---

## 15.3 ③ 写序 / fsync / NFS 语义 —— **pass**（NOTES 的危害论述有误）

**写序在代码里确实是 append → fsync → state**：`_mirror_ledger` 里
`writer.flush(); os.fsync(writer.fileno())` 在 `with` 内，`mirror_artifacts` 的
`write_json_atomic(… MIRROR_STATE_NAME …)` 在**整个 for 循环之后**（`:656`）。
rollback 分支的 `truncate` 同样 flush+fsync。核对无误。

**但 NOTES §M-三「反过来会让 state 指向一个镜像还没到达的偏移，下一轮就会从空洞
后面继续追加（灾难）」不成立。** 我直接构造了那个状态（state 声称 11,544，镜像实有
10,890，源 11,544 后再涨到 12,000）：

```
state claims 11544 | mirror really has 10890 | source 11544
events: [('mirror_truncated', 'full_copy')]
mirror == source: True
```

**防御 ②（`committed > target_size`）就是反序的保险丝**：代价是一次整拷，不是空洞。
这不是代码问题（代码更安全），是**安全论证写错了**——而错误的安全论证会诱使后人
把「冗余的」防御 ② 优化掉。变异 M9（去掉 `mirror_truncated` 分支）确实被测试杀掉
（2 例 FAIL），所以测试守住了这条线；建议只改 NOTES 措辞（N-M2）。

至于「反序是否被测试钉住」：我的**忠实版**反序变异（循环前按源大小预写 state、
循环后照旧重写）**存活 41/41**——因为它在无崩溃时语义等价。把 state 写**移到**循环前
（M1b，循环后不再写）则 **14 例 FAIL**。结论：测试钉住的是「state 必须反映本轮实际
落地量」，不是「写序」；写序本身由防御 ② 兜底。

**fsync 在 NFS 上的语义** —— 没有想当然，但也没有多余保证：
- `os.fsync(fd)` 在 Linux NFS 客户端上发 COMMIT，是正确原语；追加与 truncate 都做了。
- 校验读每次**重新 `open`**，走 close-to-open 重验（GETATTR），不吃陈旧页缓存。
- `landed = target.stat().st_size` 是**路径 stat**、且在 `with` 关闭之后；本进程是唯一
  写者，客户端会在自身写后失效属性缓存，故可信。用 `os.fstat(writer.fileno())` 在关闭前
  取会更强一档（N-M6，nit）。
- 目录项 fsync 两版都没有（`_atomic_copy` 的 `os.replace` 之后不 fsync 目录）——**既有行为**，
  本卡未加重。

---

## 15.4 ④ 节流门 —— **pass**（真值表 / 播种 / 余量算术全部复跑）

**四维真值表**：我按 `groups ∈ {0, 24, 25, 26} × elapsed ∈ {0, 299, 300, 301} × pressure ∈ {F, T}`
跑满 **32 行**，全部符合 `(groups ≥ 25 ∧ elapsed ≥ 300) ∨ pressure`：`pressure=True` 的
16 行全 True；`pressure=False` 的 16 行里只有 `(25|26) × (300|301)` 这 4 行为 True。
边界取的是 `>=` 而不是 `>`（299/300 分界正确）。变异 M4（`and` → `or`）2 例 FAIL。

**force 豁免**（用桩直接打 `_land_checkpoint`）：门关着 + `force=True` → `checkpoint`／
`write_manifest`／`_record_mirror_events` 三步全跑，**且 `_last_land_at` / `_last_land_groups`
照样推进**（force 也复位游标，所以 phase 边界的强制落地之后不会紧跟一次非 force 落地）；
同一个桩 `force=False` → 什么都不做。变异 M6（让节流也管 force）→ **10 failures + 4 errors**。

**泄压阀**：`shutil.disk_usage(self.store.root).free < 4 GiB`，读的是输出根所在挂载点
而不是 `_staged_bytes`（理由正确：后者含 prefetch buffer，正是病根）；`OSError` 吞掉
返回 False（不可读 ≠ 有压力，正确）。变异 M7 → 1 例 FAIL。

**播种**：`__init__:1098-1099` 取 `dependencies.now()` 与 `len(store.groups)`。实测
（`_last_land_groups = 19991` 模拟 L8 续跑）：新增 0 组 → False、24 组 → False、
25 组 → True。**重启后既不会立刻触发、也不会永不触发**，符合规格。

**节流窗口内 tmpfs 最坏占用 —— 我自己算的**（不采信 NOTES 的折算）：

| 量 | 值 | 出处（实测） |
|---|---|---|
| 每组 staged 字节 | **3.87 MB** | archive `batch-0000` = 7,722,411,417 B / 1,994 组；`batch-1245` = 4.72 MB / 1 组 |
| 当前 tmpfs | 24 GiB 总 / **9.1 GiB 已用** / 15 GiB 空 | `df /mnt/ramstage`（buffer 8.7 G + 账本 0.4 G） |
| 窗口长度 | `max(到 25 组的时间, 300 s)` | 健康速率 3,170 组/时 → 25 组只要 28 s，故窗口 = **300 s** |
| 窗口内新增组 | 3,170/h × 300 s = **264 组** | |
| 窗口内新增字节 | 264 × 3.87 MB = **1.02 GiB** | |
| 落地瞬时峰值 | +2.6 MB/组（模块注释：1,994 组 7.8 GiB → 峰值 ~13 GiB）→ 264 组 ≈ **0.69 GiB** | |
| **落地时刻最坏占用** | 9.1 + 1.02 + 0.69 ≈ **10.8 GiB / 24 GiB** | 余量 ≈ 13 GiB |

余量充足。**泄压阀实际不可达**：free 跌到 4 GiB 意味着 pending 资产 > 10.9 GiB ≈ 2,800 组，
而 2,800 组在 300 s 内完成 = 33,600 组/时 = 最佳实测速率的 **10.6 倍**。相应地
「阀门开火时峰值放不下」这个理论隐患（4 GiB 阀 < 0.6×8 GiB 峰值需求）也不可达，
仅记 nit（N-M7）。

**唯一真实的新增退化**：resume 若继承了 ≥ 水位的**可回收** staged 资产，改动前第一次
`not inflight` 就落地并回收，改动后要等 25 个**新**组。L8 现状 `assets = 3.2 MB`，
本次重启不会命中（N-M8）。

---

## 15.5 ⑤ 迁移路径 —— **pass**（真实 L8 字节彩排，两遍）

从 `/mnt/nfs-ro` 只读拷出 L8 真实镜像（`groups.jsonl` 397,697,832 B），从 `/mnt/ramstage`
拷出真实源（397,716,838 B），**不给 state 文件**，在 scratchpad 里跑：

| 彩排 | 设置 | 整拷了谁 | 追加 / 整拷字节 | 事件 | sha256 四文件逐一相等 | restore 后 |
|---|---|---|---|---|---|---|
| **A** | 真实滞后（1 组） | `['manifest.json']` | 0.019 / 0.349 MB | `[]` | ✅ | 19,991 组，torn=False，与源 sha 相同 |
| **B** | 镜像截到 350 MB（**记录中间**）+ 195 MB 陈旧 `.tmp` | `['manifest.json']` | 47.717 / 0.349 MB | `[]` | ✅ | 19,991 组，torn=False |

两点确认：(a) **账本零整拷**，L8 重启后第一次 checkpoint 就是增量；(b) 镜像截在**记录
中间**照样正确追加——因为不变量是「字节前缀」而不是「记录前缀」，记录边界由
`restore_mirror` 负责，两者职责分离正确。陈旧 `.tmp` 全程未被 `_artifact_names` 的
`*.jsonl` 匹配到（195,035,136 B 原封不动）。

**性能我自己测**（同一批真实字节，`os.sync()` 隔开夹具脏页，新路径取 5 次最优、
旧路径取 3 次最优）：

| | 写往镜像的字节 | 本地墙钟 | 折算 98 MB/s |
|---|---|---|---|
| 整拷四文件 | **409.2 MB** | 1.715 s | 4.18 s |
| 增量（1 组 delta） | **0.368 MB** | 0.0031 s | 0.0038 s |
| 比值 | **×1,111** | **×559** | ×1,100 |

按 25 组 delta 折算是 (25×19 KB + 0.349 MB) = 0.82 MB → **×499**，与 NOTES 的 ×516/×517
一致（差别只是我的 delta 是 1 组、它的是 25 组）。**NOTES §M-五 的量级成立。**

`.mirror_state.json` 的下游影响我另查了一遍：全仓没有一处对 build 目录做 `*.json`
通配（`q3vl/data/scan.py` 只 glob `batch_dir/indexes/shard-*.idx.jsonl`，
`databuild_viewer` 与 `tools/reeval_*` 全按具名文件读），且 `.mirror_state.json`
不匹配 `*.jsonl`。**无影响属实。**

---

## 15.6 ⑥ 变异与回归 —— **pass**（10 个我全部重写复跑）

用符号链接搭了个影子包（`scratchpad/mut/construct/`，只有 `agent.py` 是可变异的真文件），
**全程没有改过工作树里的 `agent.py`** —— 这也是与并发的 Watermark-4 隔离的手段。
影子基线 41/41 OK。

| # | 变异 | 我的结果 | NOTES 记的 |
|---|---|---|---|
| M5 | 去掉尾部字节比对 | FAIL 2 | FAIL 2 ✅ |
| M9 | 去掉「镜像短于 offset」 | FAIL 2 | FAIL ✅ |
| M10 | 去掉「源短于 offset」 | FAIL 2 | FAIL ✅ |
| M2 | 中断残留不回滚、直接从末尾追加 | FAIL 1 | FAIL ✅ |
| M3 | restore 不做换行边界截断 | FAIL 2 | FAIL 2（含缺换行那条）✅ |
| M4 | 节流 `and` → `or` | FAIL 2 | FAIL 2 ✅ |
| M6 | 节流也管住 force | FAIL 10 + ERR 4 | 9 failures + 4 errors（差 1，见 N-M3） |
| M7 | 去掉溢出阀 | FAIL 1 | FAIL ✅ |
| M8 | 允许 `bool` 当偏移 | FAIL 2 | FAIL ✅ |
| M1 | state 写在追加**之前** | **存活 41/41**（我的写法）/ M1b 变体 **FAIL 14** | 1 failure + 9 errors（对不上，见 N-M2） |

**M1 是有信息量的那个**：我的写法（循环前预写、循环后照旧重写）在无崩溃时语义等价，
所以存活是**正确**的；真正被杀的是「state 只在循环前写」（M1b，14 例）。而 §15.3 已证明
即便 state 真的跑到镜像前面，防御 ② 也会兜住。

**回归**（当前工作树，即已含 Watermark-4）：

| 跑法 | 结果 |
|---|---|
| `test_mirror_cadence`（新，22 例，`-v` 逐例） | **22/22 OK** |
| `test_source_reuse + test_sam3_drain_ledger + test_sam3_ready_index + test_land_integration + test_mirror_cadence + test_land` | **123/123 OK** |
| 全套 `discover` | Ran **510**，failures **0**，errors **31** |

31 条噪音**逐条核对**（按异常类型聚类）：`30 × FileNotFoundError: databuild.example.toml`
+ `1 × ModuleNotFoundError: No module named 'uvicorn'`，与 §14.6 的已知名单**完全一致**，
**本改动 0 个新增失败**。（总数从 492 涨到 510 是 Watermark-4 新增的测试。）

既有测试只动了 `test_land_integration.py` 一处（镜像目录断言随 state 文件更新 +
state 记的 offset == 镜像各文件实际大小），改法正确；同文件里 `prefetch` 计数那一处
是 Watermark-4 的，不属本卡。

---

## 15.7 ⑦ 协议红线扫描 —— **pass（全部无涉）**

| 红线 | 涉及 |
|---|---|
| AUC 作空间场判据 / IoU / 边界 F1 / 中心先验 | 无涉（纯 IO 记账，`rg -i 'auc'` 零命中） |
| 空间场可视化（逐图 min-max / 色标 / 叠图 resize） | 无涉 |
| G 初始化 = 0、σ 裸 exp、s 轴平滑正则、逐像素算子 | 无涉 |
| checkpoint 选择禁用 val loss | 无涉（此处的 "checkpoint" 是落地检查点，非模型权重） |
| s 缓存消费契约（域声明） | 无涉 |
| 长任务提交纪律（noclobber / `pgrep` 判活） | 无涉：新代码零 `pgrep`/`nohup`；我自己也未提交任何后台任务 |
| NFS 纪律 | **遵守**：只从 `/mnt/nfs-ro` 读（409 MB → scratchpad），未写 `/mnt/nfs`、未用也不需要 `nfsx` |
| 数据纪律（split / eval100 / low-confidence） | 无涉 |

补充：`_record_mirror_events` 新增的 `stage="mirror"` 行不会污染任何既有计数——
`_landing_counts` 按 `stage == "landing"` 过滤（`:1535`），mirror 行进不去
`checkpoint_summaries` / `i_in_members` / `sft_winners`；`terminal=False` 故不改 `status`
（测试已断言 `complete`）；只在 `failures.by_code` 多出 `mirror_full_copy` / `mirror_rollback`
两个计数，是有意的可见性。`append_failure` 对**完全相同**的记录是幂等 no-op
（`state.py:335-339` `existing == stored` → `return False`），所以即便注入冻结时钟
导致 event_id 重复也不会抛 `StateError`。

---

## 15.8 ⚠ 审阅期间 HOTFIX-Watermark-4 落入同文件（第三次，已处置）

与 §13.7 / §14.9 同类。时间线：我在 16:12 取快照（md5 `6cdeca90…`）并完成全部读码与
手工实验；约 16:30 工作树被改写为 md5 `701b63b0…`。**处置方式：AST 函数级对拍**
（`scratchpad/astdiff.py`），确认两版之间的差异集为：

```
CHG __init__ / _account_asset / _asset_directories / _clean_orphan_assets /
    _discard_prefetched / _fill_initial_mode / _prefetch_counts / _recalibrate_staged /
    _rotate_prefetch / _staged_size / _tmpfs_under_pressure / _land_checkpoint
ADD _buffer_size / _is_buffer / _staging_full / _trim_prefetch_buffer
+ 模块常量 PREFETCH_BUFFER_BYTES
```

**本卡的九个核心函数（`_copy_stream` / `_atomic_copy` / `_read_mirror_state` /
`_mirror_state_entry` / `_mirror_ledger` / `MirrorReport` / `mirror_artifacts` /
`_whole_records_limit` / `restore_mirror` / `_land_cadence_ready` / `_record_mirror_events`）
逐字未变**，§15.1-15.3、15.5、15.6 的全部实验对两版同样有效（变异实验跑的是快照，
回归实验跑的是当前树，两边都绿）。

两处交集函数的实际变化：
- `_land_checkpoint`：`self._staged_bytes < LAND_WATERMARK_BYTES` → `not self._staging_full()`，
  **MC-3 的门结构（`… or not self._land_cadence_ready()`）原样保留**，force 豁免不变。
- `_tmpfs_under_pressure`：**只改了 docstring**，函数体逐字相同。

顺带一条**跨卡观察**（不是本卡的 blocker，但两卡合入后需一起验）：Watermark-4 在
`_fill_initial_mode` 的 while 条件里新加了 `or self._staging_full()`，而 MC-3 的节流门
可以在 `_staging_full()` 为真时**拒绝落地**——这段时间里 while 会把在途队列抽干，
源窗口塌回 1，最长持续 `LAND_MIN_INTERVAL_SECONDS`。可达性与 N-M7 同界（需在上次
checkpoint 后 300 s 内重新越过 8 GiB 可回收水位 ≈ 24,000 组/时 = 最佳实测的 7.6 倍），
现实不可达；但这正是 Watermark-4 要消灭的那个病，**建议重启后同时盯这两个数**
（`landing.checkpoints` 增速 与 组/小时）。

---

## 15.9 nit 清单（9 条，均不阻塞）

| # | 位置 | 内容 |
|---|---|---|
| **N-M1**（最值得处理） | `_mirror_ledger` + NOTES §M-三「校验窗口只有 64 KiB」 | NOTES 只写了「检测不到」，**没写「也永不修复」**。实测：320 KB 镜像的第 100 字节被改后，连跑 5 次 checkpoint `events=[]`、镜像与源**始终不一致**，且 `restore_mirror` 会把坏字节原样恢复进账本（我的样例恰好仍可解析 → **静默错数据**；若坏在结构字符上则是中段 `StateError`、build 起不来）。**旧的整拷每次 checkpoint 都自愈**，这是本改动唯一一处真实的语义降级，而 `force=True` 也**不**触发整拷，故全程没有任何擦洗点。建议：每 K 次 checkpoint（或每次 phase 边界 force）强制一次整拷。代价可算：K=25 → 每 625 组一次 400 MB ≈ 0.64 MB/组，仍是现状 405 MB/组 的 1/630 |
| **N-M2** | NOTES §M-三 写序段 | 「反过来…下一轮就会从空洞后面继续追加（**灾难**）」与实测不符：防御 ②（`committed > target_size` → `mirror_truncated`）会兜住，代价是一次整拷。建议改写为「写序是**成本**优化不是**正确性**依赖；正确性由防御 ② 保底」——否则后人会以为防御 ② 是冗余 |
| **N-M3** | NOTES §M-六 变异表两行 | M1「state 写在追加之前 → 1 failure + 9 errors」我复现不出（忠实写法**存活**，移动写点的写法是 14 failures）；M6「节流也管住 force → 9 failures + 4 errors」我测到 **10 + 4**。建议把变异补丁文本（或 patch 路径）与计数并排存档，同 §14 的 N-D4 |
| **N-M4** | `_read_mirror_state` docstring + `test_unparsable_state_costs_a_copy_and_nothing_else` | 二者都说「不可解析的 state 只costs one whole-file copy」，**实际是零整拷**：state 文件整体解析失败 → 全部条目丢失 → 走迁移路径（`committed = 镜像实际大小` + 尾部比对）→ 直接追加。测试自己的断言 `report.events == ()` 就是这个意思，只是名字与 docstring 反了。改文字即可 |
| **N-M5** | `_mirror_ledger` 的两级容错不对称 | **整份** state 坏掉 → 追加（迁移路径）；**单个字段**坏掉 → 整拷。这是有意的（文件级解析失败等同于「老镜像没 state」），但代码里没写，读起来像不一致。加一行注释 |
| **N-M6** | `_mirror_ledger` 的 `tail_bytes` 未做 `bool` 排除 | `bytes` 有 `isinstance(..., bool)` 排除，`tail_bytes` 没有：`true` → 窗口塌成 1 字节。今天无害，因为记录的 `tail_sha256` 随后对不上 → `mirror_tail_changed` → 整拷。但若哪天有人认为 sha 校验与字节比对重复而删掉它，这个 1 字节窗口就活了。加同款排除或一句注释 |
| **N-M7** | `LAND_FREE_BYTES_FLOOR = 4 GiB` | 小于水位隐含的落地瞬时峰值（模块注释：0.6 × 8 GiB ≈ 4.8 GiB）。阀门若真开火且此时 pending 是满批，峰值可能放不下。**实测不可达**（见 §15.4 的 10.6 倍速率论证），但常量之间的这个不等式值得在注释里点明，或把阀提到 6 GiB |
| **N-M8** | `__init__` 的游标播种 / NOTES §M-四 | resume 若继承 ≥ 水位的**可回收** staged 资产，改前第一次 `not inflight` 即落地回收，改后要等 25 个新组（那 25 组还会以塌缩窗口渲染，见 §15.8）。L8 现状 `assets = 3.2 MB`，本次重启不命中。建议在 §M-四 的丢失窗口表里补这一行 |
| **N-M9** | `MirrorReport` 计数 + `landed != committed + written` 分支 | 该分支返回 `written + copied` 且 `action="full_copy"`，于是**已追加的字节被记到 `copied_bytes`**。仅影响诊断口径（NOTES 自己说「`copied_bytes` 贴着账本大小 = 增量没生效」的信号），不影响正确性；另 `landed = target.stat()` 用路径 stat，改 `os.fstat(writer.fileno())` 在关闭前取会更强一档（NFS 属性缓存） |

---

# **PASS。0 blocker。L8 重启可以带上这一版。**

> 唯一建议在重启**之前**顺手做的：N-M1 的周期性整拷擦洗（一个常量 + 三行），
> 它补回旧方案免费提供的「镜像自愈」。其余 8 条都是文字与注释。
---

# 16. 终审：HOTFIX-Watermark-4 + HOTFIX-MirrorScrub-5（重启窗口唯一门禁）

> 审阅人：实现审阅 subagent（独立，与编码 agent 无关）｜ 日期：2026-08-13
> 审阅对象：`dataset_build/src/construct/agent.py`（md5 `909a04fc651186a912be59cf0a2dfa3c`，
> 17:17 取快照，17:40 复核**未变**）、`tests/test_land_watermark.py`（18 例）、
> `tests/test_mirror_cadence.py`（26 例）
> 依据：NOTES §W-一~八、§S-一~七（**审阅期间被删，见上方事故说明**）、本文件 §15（N-M1 / §15.8）
> 纪律：只读审阅；变异全部在影子包 `scratchpad/mut/construct/` 上做，**工作树的 `agent.py`
> 全程一字未改**（md5 两次一致）；未碰运行中的 pid 1433269（实证仍在跑，etime 03:13）；
> 未读也未写 `trash/`；NFS 只读走 `/mnt/nfs-ro`，未写 `/mnt/nfs`、未用 `nfsx`。

## 16.0 结论

# ✅ **PASS —— 0 blocker，两卡可进重启窗口**

| 审阅项（对应任务卡） | 判定 | 依据 |
|---|---|---|
| ① WM4 拆分完备性（全仓找读点 + 当前 chunk 保护的并发论证） | **pass** | §16.1，`_staged_bytes` 生产读点**只有一个**，语义正确；无写者声称成立 |
| ② 淘汰安全：被淘汰的 SAM3 停泊副本 | **pass —— 既不重新预取，也不失败** | §16.2，回落归档读，且**不可能失败**（有构造性论证） |
| ③ 窗口恢复用例 + 我构造的第三场景 | **pass** | §16.3，原对子复跑通过；第三场景我写了并且它**独立杀掉变异 M-A** |
| ④ Scrub 语义（缺 state / 次序 / 跨重启 / 治愈） | **pass** | §16.4，四条全部复跑；缺 state → 刷洗**确认更安全** |
| ⑤ 五卡叠加 AST 一致性 + §15.8 跨卡交互独立判断 | **pass** | §16.5，四处交叠语义清晰；§15.8 我给出**更低**的风险评级并附理由 |
| ⑥ 变异 4 个 + 全套回归 + 31 错误名单 | **pass** | §16.6，4/4 被杀；514 跑，0 failure，31 error 逐条一致 |
| ⑦ 协议红线扫描 | **pass（全部无涉）** | §16.7 |
| **合计** | **0 blocker / 6 nit** | 全部不阻塞 |

**一句话**：Watermark-4 把水位真正接到了「landing 能回收的那本账」上——`_staged_bytes`
在全仓只剩**一个**生产读点，且这个读点就是水位；MirrorScrub-5 把 §15.9 N-M1 那处唯一的
语义降级补了回来，并且顺带把 N-M4 / N-M5 一并变成了真命题。**两卡可以一起上重启窗口。**

主 agent 最该知道的两条（都不是 blocker）：一是**重启后第一次 checkpoint 是一次约 418 MB
的整拷**（L8 镜像现在确实没有 `.mirror_state.json`，我已核实），这是有意的、买一个可信基线；
二是那份陈旧的 `groups.jsonl.tmp` **不需要手工清**——第一次刷洗的 `_atomic_copy` 会覆盖并
`os.replace` 掉它（NOTES §S-六 让主 agent 择机清，其实是多余的动作）。

---

## 16.1 ① WM4 拆分的完备性 —— **pass**

### 全仓读点普查（任务卡点名要求）

`rg` 全仓（`--type py`，排除 `agent.py` 后**只剩测试**），两本账的读写点是封闭的：

| 计数器 | 写点 | 读点 | 每个读点拿到的语义 |
|---|---|---|---|
| `_staged_bytes` | `_recalibrate_staged` / `_account_asset` / `_clean_orphan_assets` | **`_staged_size()` 一个** | —— |
| `_staged_size()` | —— | **`_staging_full()` 一个** | 可回收 assets ✅ |
| `_staging_full()` | —— | `_land_checkpoint:2163`、`_fill_initial_mode:3152` | 两处都是水位语义 ✅ |
| `_buffer_bytes` | `_recalibrate_staged` / `_account_asset` / `_trim_prefetch_buffer` / `_discard_prefetched` | `_buffer_size()` | —— |
| `_buffer_size()` | —— | `_trim_prefetch_buffer:2082`（上限）、`_prefetch_counts:1606`（manifest） | 缓冲预算 ✅ |

即任务卡的三问逐条落地：**水位读 assets** ✅（唯一读点就是 `_staging_full`）、**孤儿清扫仍看全集**
✅（`self._staged` 这本**字典**没有拆，拆的只是字节；`_clean_orphan_assets` 照旧遍历全集，
再用 `_is_buffer` 排除缓冲）、**manifest 的 prefetch 计数** ✅（`bytes` 取 `_buffer_size()`，
不是水位）。

与 HEAD 的函数级对拍确认 `_clean_orphan_assets` 的改动是**纯重构**（内联前缀测试 → `_is_buffer`），
判定逻辑逐字等价。`_discard_prefetched` 的 `_staged_bytes -= …` → `_buffer_bytes -= …` 是本卡
的实质修正且方向正确（预取路径本就是缓冲）。

### 分类一致性

`_recalibrate_staged` 按**目录**分类（`_is_buffer(str(directory) + os.sep)`），`_account_asset`
按**路径**分类，两者都走同一个 `_buffer_prefix`（带尾分隔符），所以 `prefetch_old/` 这类兄弟
目录不会被误判。`__init__` 的初始化次序也对：`_prefetch`(1177) → `_buffer_prefix`(1185) →
`_recalibrate_staged()`(1188)，而 `_recalibrate_staged` 是三个字段的**唯一**初始化点。

### 「当前 chunk 保护在并发 rotate 下真无写者」—— 读码验证，声称成立

`_trim_prefetch_buffer` 的 docstring 声称「调用时没有预取在途」。逐层验证：

1. `_SourcePrefetch` 是 `max_workers=1` 的单线程池，且只保留**一个** `_pending` future；
2. `take()` 里 `pending.result()` 是**阻塞**的，返回前把 `_pending` 置 `None`——所以
   `_account_prefetched()` 之后、下一次 `submit()` 之前，**既没有在跑的预取，也没有排队的预取**；
3. `_rotate_prefetch` / `_discard_prefetched` / `finish_oldest` **全在主线程**（`_fill_initial_mode`
   的那一层循环），所以也不存在第二个写者；
4. 渲染工作线程只**读**缓冲，不写。

`RotationWiringTests` 把 `take → trim → submit` 这个次序钉死了（我复跑通过）。**声称成立。**

补一条它没说、但成立的：即便真有读者撞上 unlink 也**无害**——已 open 的 fd 在 POSIX 下
继续读到旧 inode；尚未 open 的路径拿到 `OSError`，被 `_local_or_prefetch` 的
`except OSError: return None` 接住并回落归档。

---

## 16.2 ② 淘汰安全 —— **pass**，答案是「回落归档读」，**不是静默失败**

任务卡问：被淘汰的 SAM3 停泊副本在 drain 需要时——重新预取？还是 drain 失败？
**两者都不是，是回落一次归档读**，而且我能给出**构造性**的不失败论证：

**为什么一定读得到。** `tools/prefetch.py` 的 `prefetch()` 只物化两类之外的路径：
本机已存在的路径被**跳过**（`read_bytes` 本就 local-first），未在目录里 `_locate()`
到的路径也被**跳过**。所以**缓冲里的每一个文件，其源路径按构造必然可由 catalog 解析**。
淘汰之后 `read_bytes` 走 `_local_or_prefetch`（本地 → 缓冲 → `OSError` → `None`）
→ `_shared(db).read()` → 归档 pread。**不存在「缓冲里有、归档里没有」的条目**，
所以淘汰不可能把一次读变成一次失败。

**drain 的读路径逐点核过**（顺着任务卡要求的方向查）：
`_drain_sam3_and_replacements` → `_try_ready_relabel` → `refresh_source_record` →
`sources.py::_inspect_cache_dir`，其中 `subject.json` / `subject.png` / `source_path`
分别走 `read_bytes` / `path_exists` / `read_bytes`。`path_exists` 是 local-then-catalog，
**从不看缓冲**。全仓 `path_for` 只有两个调用者（`_discard_prefetched` 与
`_trim_prefetch_buffer`），**没有任何 drain 侧的门用 `Path.is_file()` 去问缓冲路径**。
故淘汰对 drain 只有吞吐影响，没有正确性影响。**不是 blocker。**

**不会重新预取**：停泊源在 `pending` 里，而 `_chunk_paths` 过滤 `skip = terminal | pending`，
所以它们永远不进 chunk、也永远不进 `keep`。

**一条结构性观察（nit N-W2）**：mtime 最冷优先与「给 drain 留副本」这个优化是
**系统性相冲**的——停泊副本按构造一定比在渲染的 chunk 更冷，所以每次修剪都优先淘汰它们。
5,003 个停泊（≈9 GiB）对 3 GiB 上限，意味着最老的约 2/3 必然被淘汰。NOTES §W-七 待决策二
已经把这条写清楚了（「drain 阶段大约 2/3 的重渲会回落到归档读」），**表述准确、没有夸大**；
我只补一句：这不是偶然的命中率损失，而是淘汰序与停泊语义的结构冲突，调 4 GiB 只能缓解不能消除。

---

## 16.3 ③ 窗口恢复用例 —— **pass**，第三场景我写了

**原对子复跑**（`-v` 逐例）：
`test_a_resident_buffer_over_the_mark_leaves_the_window_open` → 峰值 ≥ 5 ✅；
`test_real_staged_assets_over_the_mark_still_close_it` → 峰值 == 1 ✅。
这一对确实只差「字节记在哪本账上」一个变量，设计正确。

**但它们从不真正触发一次淘汰**——两例都只是**记账**，`_trim_prefetch_buffer` 一次也没跑。
所以我按任务卡补了第三场景（写在 scratchpad 的影子测试里，**未落入工作树**）：

> `TrimmedBufferStillLeavesTheWindowOpenTests` —— 播 prod-l8 形状（一个超额停泊副本 + 一个热副本，
> assets = 0），把上限 patch 到 4 KiB，**真的驱动一次 `_trim_prefetch_buffer`**，
> 淘汰掉最冷的那个，然后才测在途峰值。

断言链：淘汰前 `buffer > 水位 ∧ 水位读数 == 0 ∧ _staging_full() == False`；
淘汰 1 个；淘汰后 `buffer ≤ 上限` ∧ **水位读数仍然是 0** ∧ `_staging_full()` 仍为 False；
幸存者是热副本；**峰值 ≥ 5**。**通过。**

它不是摆设：变异 M-A（修剪去扣 `_staged_bytes`）把它**独立杀掉**，与另外 3 个既有用例一起。
即「缓冲被 trim 到上限以下之后窗口仍不塌」这条现在有钉子了。

---

## 16.4 ④ Scrub 语义 —— **pass**

### 缺 state → 刷洗，**确认更安全而非回归**

Cadence-3 在无 state 时取 `committed = target_size`，只比对最后 64 KiB 就继续追加——
即**在一段没人验证过的前缀后面接着写**，而这恰恰是 L8 的真实处境（那份镜像是旧代码写的）。
MirrorScrub-5 改成付一次整拷买一个逐字节可信基线。方向正确，且**不闩锁**：
`test_an_absent_state_file_scrubs_once_and_then_appends` 断言下一拍只整拷 `manifest.json`。
变异 M-D（退回裸追加）被 2 例杀掉。

顺带：这一改还把 §15.9 的 **N-M4 与 N-M5 变成了真命题**——N-M4 说 docstring 与测试名都写着
「不可解析的 state 花费一次整拷」而实际是零整拷，现在**真的**是一次整拷了；N-M5 说的
「整份 state 坏 → 追加 / 单字段坏 → 整拷」这个不对称，现在两边都是整拷，**不对称消失了**。
两条 nit 由本卡顺带闭环。

### `reason="scrub"` 与四防御事件共存时的次序

`if reason is None and scrub` 位于**四条防御分支之后、并且在尾部比对之后**，所以
「既是刷洗又是异常」的那一拍报的是异常的 reason。变异 M-C（把 scrub 挪到最前）→ 4 例 FAIL，
含点名的 `test_an_anomaly_on_a_scrub_checkpoint_keeps_its_own_reason`（期望
`["mirror_truncated","scrub"]`）。**次序正确且被钉住。**

### 计数器只活在 state 文件里、跨重启延续

`_read_mirror_state` 返回 `(files, seq)`，`seq` 对非 int / `bool` / 负数一律给 `None`
（老陷阱同款处理）。`sequence = (checkpoints or 0) + 1` 对 `checkpoints == 0` 也正确。
全仓 grep 确认**进程内不存在任何刷洗游标**。复跑
`test_the_counter_lives_in_the_state_file_and_survives_a_restart` ✅：三次独立调用记 1/2/3，
手工把盘上计数改成 K−1 后下一拍即刷洗，刷洗后回到追加。

### 治愈用例（byte 100）

复跑 `test_the_scrub_heals_damage_the_tail_window_cannot_see` ✅。用例质量值得一提：
它把**字节治愈断言放在事件断言之前**，所以去掉刷洗会挂在字节上（变异 M-C 实测
`AssertionError` 出现在治愈处），而不是只挂在「少了一行日志」上——这正是 N-M1 要的那种钉子。

### 一处新增的成本路径（nit N-S1）

`_read_mirror_state` 把「文件不存在」与「读失败」并进同一个 `except (OSError, ValueError)`，
所以 NFS 的一次**瞬时**读失败现在会换来一次 400 MB 整拷，而 Cadence-3 下只是一次廉价追加。
方向上这是对的（本项目 s 缓存契约那一节的教训就是「宁可吵闹昂贵，不要静默错误」），
所以我不认为要改行为；但值得把 `FileNotFoundError` 与其它 `OSError` 分开注释一句，
否则后人会以为整拷是迁移专用路径。

---

## 16.5 ⑤ 五卡叠加的整体一致性 —— **pass**

AST 函数级对拍（HEAD → 工作树）共 45 个单元变动。按卡归属后，**被一张以上卡触碰的只有四处，
且每处的交叠在文本上是分离的语句**：

| 交叠单元 | 涉及卡 | 交叠是否清晰 |
|---|---|---|
| `__init__` | MC-3（`_last_land_at/_last_land_groups`）+ WM-4（`_buffer_prefix`/`_buffer_evicted`） | ✅ 不同语句；且 WM-4 需要的次序（`_prefetch` → `_buffer_prefix` → `_recalibrate_staged`）成立 |
| `_land_checkpoint` | MC-3（节流门）+ WM-4（`_staging_full`） | ✅ `not force and (not _staging_full() or not _land_cadence_ready())`，门 A 门 B 与短路次序均保留 |
| `_tmpfs_under_pressure` | MC-3 新增 + WM-4 | ✅ WM-4 只改 docstring（§15.8 已核，仍然成立） |
| `mirror_artifacts` / `_mirror_ledger` / `_read_mirror_state` | MC-3 新增 + S-5 加刷洗 | ✅ S-5 只加了 3 行 + 一个 kwarg + 返回值一元；MC-3 的四条防御分支**未被重排或删除**（M-C/M-D 双向证明） |

manifest 卡（×6,381）与 drain 账本卡（×5,257）与新计数器的接触面只有两处，且都是有意的：
`_manifest` → `_prefetch_counts()` 多两次 `_staged_lock` 下的读（每次写 manifest 一次，
不是每组）；drain → `_fill_mode` → `_staging_full()` / `_trim_prefetch_buffer`。无冲突。

### §15.8 跨卡交互（WM4 排水条款 × MC3 节流）—— 我的独立判断：**比 §15.8 评的更低**

现象属实：`_fill_initial_mode` 的 while 里有 `or self._staging_full()`，而门 B 可以拒绝落地，
于是队列被抽干、窗口塌到 1。但我给出 §15.8 **没有给的两条**理由，认为它不构成重启风险：

1. **它现在是自限的**。改前之所以是灾难，是因为缓冲永远不掉、条件**恒真**；改后该条件只可能
   在**真有 ≥8 GiB 可回收 assets** 时为真，而落地（最多被推迟 `LAND_MIN_INTERVAL_SECONDS`）
   **必然**回收它们。所以塌缩上界是 300 s 然后自己结束，不是稳态。
2. **它只可能发生在「等待代价为零」的那一格**。
   `_staging_full() = assets ≥ 8 GiB ∨ free < 4 GiB`，而
   `_land_cadence_ready() = (25 组 ∧ 300 s) ∨ free < 4 GiB`——**两个门共用同一个泄压阀**。
   所以只要挂载点是真的紧张，**两个门同时打开**，根本不会塌缩；剩下的唯一情形是
   「有 8 GiB assets 但空间充裕」，那正是推迟 300 s 毫无代价的情形。

叠加 §15.8 自己算的可达性（需 24,000 组/时 = 最佳实测 7.6 倍），我**不把它列为重启风险**。
§15.8 建议的重启后同盯两个数（`landing.checkpoints` 增速 与 组/小时）仍然值得做，成本近零。

**一条 §15.8 时点上还看不到的新交互，我这里补掉**：MirrorScrub-5 × MC-3 节流。
刷洗每 25 次 checkpoint 一发，而节流保证 checkpoint 间隔 ≥ 300 s，所以刷洗最密也就
25 × 300 s ≈ 125 分钟一次，单次 ≈ 418 MB / 98 MB/s ≈ 4.3 s。**可忽略**。
另核实 `_atomic_copy` 的临时文件是 `target.with_name(name + ".tmp")`，落在**镜像目录（NFS）**
而不是 tmpfs，所以刷洗**不吃一个字节的 tmpfs**，§W-五 的 24 GiB 预算不受影响。

---

## 16.6 ⑥ 变异与回归 —— **pass**

**变异**（任务卡要 3 个，我做了 4 个；全部在影子包上，工作树 md5 前后一致）：

| # | 变异 | 结果 | 杀它的用例 |
|---|---|---|---|
| M-A | `_trim_prefetch_buffer` 改扣 `_staged_bytes` | **FAIL 4** | 3 个 `PrefetchCeilingTests` + **我的第三场景** |
| M-B | `_staged_size()` 重新把缓冲加回去（**原 bug**） | **FAIL 7 + ERR 8** | 覆盖面最广 |
| M-C | `scrub` 挪到四条防御**之前** | **FAIL 4** | 含 `…keeps_its_own_reason` |
| M-D | 缺 state 不再刷洗（退回 Cadence-3 裸追加） | **FAIL 2** | 两个改名/改期望的用例 |

**逐文件复核**（`/home/bc/envs/databuild/bin/python`）：

| 文件 | 例数 | 结果 |
|---|---|---|
| `test_mirror_cadence` | 26 | OK |
| `test_land_watermark` | 18 | OK |
| `test_source_reuse` | 40 | OK |
| `test_sam3_drain_ledger` | 21 | OK |
| `test_land_integration` | 19 | OK |
| `test_sam3_ready_index` | 6 | OK |
| `test_sam3_batch_fallback` | 2 | OK |
| **七个文件合跑** | **132** | **132/132 OK** |

**全套 `discover`**：**Ran 514，failures 0，errors 31**。31 条我按异常类型聚类逐条核对：
`30 × FileNotFoundError: /home/bc/VeraRetouch/databuild.example.toml` +
`1 × ModuleNotFoundError: No module named 'uvicorn'`——与 §14.6 / §15.6 的已知名单
**逐条相同**，**本轮两卡 0 个新增失败**。（510 → 514 是 MirrorScrub-5 的 4 个新例。）

唯一改动的既有测试仍然只有 `test_land_integration.py` 的 `manifest["prefetch"]` 全等断言
（随 `bytes` / `evicted` 两个新字段更新，**仍是全等而非子集**，改法正确）。
我另确认全仓对该对象没有第二处全等断言，且 `.mirror_state.json` 的 `checkpoints` 新键
不破坏 `test_land_integration` 既有的 `state["version"]` 与 `files` 断言。

---

## 16.7 ⑦ 协议红线扫描 —— **pass（全部无涉）**

| 红线 | 涉及 |
|---|---|
| AUC 作空间场判据 / IoU / 边界 F1 / 中心先验 | 无涉（`rg -in 'auc\|soft.?iou\|boundary.?f1\|center.?prior'` 三文件零命中） |
| 空间场可视化（逐图 min-max / 色标 / 叠图 resize） | 无涉（零命中） |
| G 初始化 = 0 / σ 裸 exp / s 轴平滑正则 / 逐像素算子 | 无涉（纯 IO 记账） |
| checkpoint 选择禁用 val loss | 无涉（此处 checkpoint 是落地检查点，非模型权重） |
| s 缓存消费契约（域声明） | 无涉 |
| 长任务提交纪律（noclobber / `pgrep` 判活） | 无涉：三个文件零 `pgrep`/`nohup`/`setsid`/`pkill`；我自己只提交过一个前台可控的后台 discover，且**用 `rm -f` 开路**（本轮确实撞到一次 `file exists`，已按纪律处理，见 §16.8） |
| NFS 纪律 | **遵守**：只 `ls` 了 `/mnt/nfs-ro` 两次，未写 `/mnt/nfs`、未用也不需要 `nfsx` |
| 数据纪律（split / eval100 / low-confidence） | 无涉 |
| 配置红线（两卡声称不动 config/TOML） | **核实无涉**：`config.py` 与 `databuild.example.toml` 对 `watermark/land_min/mirror_scrub/prefetch_buffer/free_bytes_floor/mirror_verify` 全部零命中，五个常量都在 `agent.py` 顶部 |

---

## 16.8 ⚠ 两条过程记录

1. **`noclobber` 又咬了一次（本项目第 N 次）**：我做函数级对拍时循环里用 `> $SP/a.txt`，
   第二轮起全部 `file exists` 且**重定向整条失败**，于是后面几个函数 diff 出来的是
   第一个函数的内容——差点据此写出错误结论。按 CLAUDE.md 的纪律改成 `rm -f` 开路后正常。
   这条纪律对**审阅自己的中间文件**同样适用，不只对提交长任务。
2. **工作树并发改写，本轮升级成删除**：见本文件顶部的事故说明。`agent.py` 本身**全程未变**
   （17:17 与 17:40 两次 md5 均为 `909a04fc…`），所以 §16 的全部实验对上线版本有效；
   被删的是审阅底账与 NOTES。

---

## 16.9 nit 清单（6 条，均不阻塞）

| # | 位置 | 内容 |
|---|---|---|
| **N-W1** | `_trim_prefetch_buffer` docstring | 「nothing is writing into the directory」对**写者**成立（已验证），但**上一个 chunk 的在途源**其副本不在 `keep` 里，渲染线程可能正在读它。实际无害（POSIX unlink + `OSError` 回落），且它们是次热的、几乎轮不到被淘汰。建议 docstring 把「无写者」与「读者可被安全淘汰」分开写一句，否则后人会以为 `keep` 覆盖了所有在读的东西 |
| **N-W2** | `_trim_prefetch_buffer` 淘汰序 vs 停泊语义 | mtime 最冷优先与「给 drain 留副本」**结构性相冲**：停泊副本按构造恒比在渲染的 chunk 冷，必被优先淘汰；9 GiB 停泊对 3 GiB 上限 ⇒ 最老约 2/3 必然回落归档读。NOTES §W-七 待决策二已如实写明，此处只是补一句「这是结构冲突不是命中率抖动」，调 4 GiB 只缓解不消除 |
| **N-W3** | `_clean_orphan_assets` 的 `self._staged_bytes -= self._staged.pop(path)` | 在 `_staged_lock` **之外**改计数器与字典。**是既有行为，本卡未加重**（与 HEAD 逐字相同），但 WM-4 恰恰是让这个计数器变成源窗口生死攸关的量的那张卡，顺手收进锁里更稳妥 |
| **N-W4** | `_prefetch_counts` 的 `evicted` | 文件已消失的条目也计入 `evicted`（`os.unlink` 吞 `FileNotFoundError` 后照样 `evicted += 1`）。只影响诊断口径，且 `test_a_copy_that_vanished_underneath_still_gives_its_bytes_back` 就是按这个语义写的。纯文字 |
| **N-S1** | `_read_mirror_state` 的 `except (OSError, ValueError)` | 「文件不存在」（迁移，应整拷）与「瞬时读失败」（NFS 抖动）合并处理，后者现在要付 400 MB。方向是对的（吵闹优于静默），但建议注释里点明这是**有意**把瞬时故障也算作「位置未知」，或把 `FileNotFoundError` 单列 |
| **N-S2** | NOTES §S-六 关于陈旧 `.tmp` 的建议 | 「仍建议主 agent 择机清」是**多余动作**：`_atomic_copy` 的 tmp 名就是 `groups.jsonl.tmp`，第一次刷洗会 `open("wb")` 截断它、再 `os.replace` 掉，自动消失。另 §15.9 的 **N-M6（`tail_bytes` 未排除 `bool`）本卡未处理**，但刷洗把它的最坏存活期从「永远」压到了 25 次 checkpoint，风险已显著下降 |

**§15.9 的处置情况**：N-M1 **已闭环**（就是本卡）；N-M4 / N-M5 **由本卡顺带变成真命题**；
N-M6 未改但已被刷洗兜底；N-M2 / N-M3 / N-M7 / N-M8 / N-M9 是文字与注释，未见处理，仍不阻塞。

---

# **PASS。0 blocker。两卡可以带进重启窗口。**

> 重启后建议只盯三个数：`landing.checkpoints` 的增速、组/小时、以及
> `manifest["prefetch"]["bytes"] / ["evicted"]`（第三个是本轮新增的可见性，
> 缓冲是否真的稳在 3 GiB、淘汰是否在持续咬人，一眼就能看出来）。

---

## 16.10 附录：第三场景的测试源码（§16.3）

本轮为任务卡第 3 项新写的用例。**未落入工作树**（审阅不改实现），存档于此以便编码 agent
直接采纳进 `tests/test_land_watermark.py`；它独立杀掉变异 M-A。

```python
"""REVIEW §16 scenario 3: after the ceiling actually evicts, the window still stands.

The shipped pair proves the *classification* (bytes booked as buffer leave the
window open, booked as assets close it).  Neither of them ever runs a real
eviction.  This one does: the prod-l8 shape is seeded, ``_trim_prefetch_buffer``
is driven for real against a patched ceiling, and only then is the in-flight peak
measured — so it covers the one ordering the pair does not, namely that the trim
debits the buffer counter and leaves the mark's counter alone.
"""
from __future__ import annotations

import os
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from construct import agent

from tests.test_source_window_and_cgt import WindowFixture


class TrimmedBufferStillLeavesTheWindowOpenTests(WindowFixture):

    RESIDENT = agent.LAND_WATERMARK_BYTES + 1
    CEILING = 4096

    def test_a_real_eviction_brings_the_buffer_under_its_ceiling_and_the_window_holds(
        self,
    ) -> None:
        config, dependencies = self.build(window=5, tag="trimmed-buffer")
        lock = threading.Lock()
        filled = threading.Event()
        active: set[str] = set()
        leaders: set[str] = set()
        peak = 0
        observed: dict[str, object] = {}
        render = agent.CanonicalPipeline._render_source_buffered
        fill = agent.CanonicalPipeline._fill_initial_mode

        def seed(pipeline, mode, sources, target, **kwargs):
            if "before" not in observed:
                directory = pipeline.store.root / "prefetch"
                directory.mkdir(parents=True, exist_ok=True)
                pipeline._buffer_prefix = str(directory) + os.sep
                # A stub buffer: the fixture has no archive root, so the real one
                # is None and the trim would no-op.  Only ``directory`` and
                # ``path_for`` are ever reached from the code under test.
                pipeline._prefetch = SimpleNamespace(
                    directory=directory,
                    path_for=lambda source_path: directory / "never",
                    take=lambda: [],
                    submit=lambda paths: None,
                    close=lambda: None,
                    buffered=0,
                    errors=0,
                )
                # The prod-l8 shape: one oversized resident copy, 0 landable bytes.
                resident = directory / "resident"
                resident.touch()
                warm = directory / "warm"
                warm.write_bytes(b"\0" * 1024)
                os.utime(resident, (1_000_000.0, 1_000_000.0))
                os.utime(warm, (2_000_000.0, 2_000_000.0))
                with pipeline._staged_lock:
                    pipeline._staged[str(resident)] = self.RESIDENT
                    pipeline._staged[str(warm)] = 1024
                    pipeline._buffer_bytes += self.RESIDENT + 1024
                observed["before"] = (
                    pipeline._buffer_size(),
                    pipeline._staged_size(),
                    pipeline._staging_full(),
                )
                with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", self.CEILING):
                    observed["evicted"] = pipeline._trim_prefetch_buffer()
                observed["after"] = (
                    pipeline._buffer_size(),
                    pipeline._staged_size(),
                    pipeline._staging_full(),
                )
                observed["survivor"] = sorted(p.name for p in directory.iterdir())
            return fill(pipeline, mode, sources, target, **kwargs)

        def wrapper(pipeline, source, mode, *, selector_turn=None, **kwargs):
            nonlocal peak
            with lock:
                active.add(source.source_id)
                peak = max(peak, len(active))
                if len(active) >= 5:
                    filled.set()
                leader = mode not in leaders
                leaders.add(mode)
            try:
                if leader:
                    filled.wait(timeout=30)
                return render(pipeline, source, mode, selector_turn=selector_turn, **kwargs)
            finally:
                with lock:
                    active.discard(source.source_id)

        with mock.patch.object(
            agent.CanonicalPipeline, "_render_source_buffered", wrapper
        ), mock.patch.object(
            agent.CanonicalPipeline, "_fill_initial_mode", seed
        ), mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = agent.run(config, dependencies=dependencies)

        self.assertEqual(manifest["completed"]["groups"], self.TARGET)
        self.assertIn("before", observed, "the seed never ran")
        before_buffer, before_mark, before_full = observed["before"]
        after_buffer, after_mark, after_full = observed["after"]
        # Before: over the ceiling *and* over the water mark, yet not "full",
        # because none of it is landable.
        self.assertGreater(before_buffer, agent.LAND_WATERMARK_BYTES)
        self.assertEqual(before_mark, 0)
        self.assertFalse(before_full)
        # The eviction really happened and really came back under the ceiling.
        self.assertEqual(observed["evicted"], 1)
        self.assertLessEqual(after_buffer, self.CEILING)
        self.assertEqual(observed["survivor"], ["warm"])
        # And it debited the buffer only: the mark never moved off zero.
        self.assertEqual(after_mark, 0)
        self.assertFalse(after_full)
        # The window is still the configured five after all of that.
        self.assertGreaterEqual(peak, 5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
```

---

# 17. 终审：PIPE-PassInterleave-1（渲染/标注按轮流水化）—— 重启窗口唯一门禁

> 审阅人：实现审阅 subagent（独立，与编码 agent 无关）｜ 日期：2026-08-14
> 审阅对象：`dataset_build/src/construct/agent.py`（md5 `b9acdb08430e9e56707c9ab06e136801`，
> 10:48 取快照，11:14 复核**未变**）、`responses.py`（`65942a2b…`）、`state.py`（`28c89cab…`）、
> `tests/test_annotation_interleave.py`（27 例，`26a5c31e…`）
> 依据：NOTES §P-一~十、任务卡八项、本文件 §12/§15/§16
> 纪律：只读审阅；变异全部在影子包 `scratchpad/mut/construct/`（符号链接 + 三个真文件）上做，
> **工作树三个源文件全程一字未改**（md5 两次一致）；未碰运行中的 pid 1790282（实证仍在跑，
> etime 17:17，且其 manifest 里**没有** `annotation.inflight` 键 —— 即新代码确未生效）；
> 未读也未写 `trash/`；NFS **只读**走 `/mnt/nfs-ro`（一次 readdir + 一次 glob），
> 未写 `/mnt/nfs`、未用 `nfsx`；`/mnt/ramstage` 只读。

## 17.0 结论

# ✅ **PASS —— 0 blocker，可进重启窗口**

| 审阅项（对应任务卡） | 判定 | 依据 |
|---|---|---|
| ① land-unlink 竞态的结构性修复 | **pass** | §17.1，读码 + **反例穷举失败** + 我自搭的"在途批次跨越后续 checkpoint"实验（3 次落盘、8 次字节读全绿）；尾巴上界用现场 journal 复核 |
| ② 单写者收窄论证（逐消费者 + 游标折叠） | **pass** | §17.2，逐消费者复核；`_Sam3Ledger` 在 **16 线程标注 + 1 线程渲染并发追加 + 250 万次 refresh** 下与全表扫 oracle 逐源相等 |
| ③ `_next_rounds` 折叠等价 + `_attempt_count` 计费边界 | **pass（1 nit）** | §17.3，400 组随机 journal 全部相等；计费语义未动，成本已实测 |
| ④ 快照一致性 / 第五个读点 | **pass** | §17.4，全仓普查：无遗漏读点 |
| ⑤ 收尾语义 / 异常路径 / 孤儿线程 | **pass（2 nit）** | §17.5，`inflight` 结构性为 0；失败 build 无残留线程（我实测） |
| ⑥ resume | **pass** | §17.6，我自己写的三例夹具（含跨两次运行的**不重复计费**断言）全绿 |
| ⑦ 变异 5/8 复跑 + 27 例 + 全量回归 + 红线 | **pass** | §17.7，五个变异的失败数与 NOTES **逐个相同**；`Ran 541 / failures 0 / errors 31` 与名单逐条一致 |
| ⑧ 两个建议的独立意见 | —— | §17.8 |
| **合计** | **0 blocker / 7 nit** | 全部不阻塞 |

**一句话**：本卡最危险的那条（落盘 unlink 与标注读字节抢跑）确实被"让窗口不存在"而不是加锁解决了，
我按任务卡要求做了反例穷举，**没能构造出交接后字节还会动的路径**，并用一个把批次卡在 driver 里、
强行让渲染主线程再落盘两次的实验实证了这一点；单写者约定的收窄逐消费者成立，最脆的那条
（`_Sam3Ledger` 的游标假设）我用 17 线程压测钉死；批量折叠与逐任务 oracle 在 400 组随机
journal 上完全一致，且**没有碰计费**。**可以放重启窗口。**

最需要主 agent 知道的两条都不是 blocker：**(a) Ctrl-C 一次不再够**（§17.5 / N-P6）；
**(b) 重启后第一个边界会有一次数分钟的静默**（全量 catalog upsert + 4.8 万任务交接，§17.8）。

---

## 17.1 ① land-unlink 竞态 —— **pass**（反例穷举失败 + 实证）

### 判据是否真等价于「已落地」

读码链条（三处，缺一不可）：

1. `_land_groups:2042` 的 `pending` 要求**组内每个文件都存在**才进批；
2. `:2081` 两个 `land()` **都返回**（含各自的 verify）之后，`:2092` 才 `unlink(missing_ok=True)`；
3. `_annotation_bytes_settled:2221` 的判据是 `bool(after) and not Path(after).is_file()`。

所以 `after_path 不在本地` ⟸ `本组已落地`。任务卡要的是**反向**（⟹），我按"谁还能删这个文件"逐一穷举：

| 候选删除者 | 结论 |
|---|---|
| `_land_groups` 的 unlink | 正是我们要的那条，落地在前 ✅ |
| `_clean_orphan_assets:2274` | `referenced` 由**全部** `store.groups.values()` 的 `_group_assets` 构成（含已丢失组），一个仍在 journal 里的组的资产**永远不是孤儿** ✅ |
| `_trim_prefetch_buffer` / `_discard_prefetched` | 只动 buffer 目录，且 `_is_buffer` 前缀带尾分隔符；`_group_assets` 只含 `after_path`/`cgt_path`，二者不相交 ✅ |
| `_land_groups` 的 `shutil.rmtree(staging_root)` | 只删 `.land/` 下的**硬链接**，原始 inode 由 tar 成员持有 ✅ |
| 重启后 tmpfs 被清空 | `_run_phases:3647` 的 `_verify_group_assets` 在**任何 `_fill_mode` 之前**跑，判不出归档的组写 `group_assets_lost`，而 `pending_annotation_tasks:437` 显式跳过 `lost` ✅ |
| 槽位重填改写同名 JPEG | 发生在组入账**之前**（`_account_asset` 的注释即指此），入账后不再改写 ✅ |

**没有第七个删除者。** 结论：判据成立，但它是**推论**（「除了落地没人删」）而不是**实证**（「归档确实答得出」）。
这是 N-P2 的由来 —— 不阻塞，因为上表六条我逐条读码确认过。

### 我自己搭的实证：把批次卡在 driver 里，强行让渲染再落盘两次

不看它的测试，自搭夹具（scratchpad `my_inflight.py`，基于 `LandFixture`）：
标注 double 在 `drain(only=…)` 里**先挂起**，等一个由 `_land_groups` 包装器在"批次已交出之后"
才 set 的 Event，然后才逐任务读 `I_tar` **与** `I_in`（`prepare_task` 真正会读的两张图），
走 `read_bytes(path, db_path=…)` 的三级阶梯。

```
landings=3  reads_verified=8  problems=[]  status=complete  annotation.pending=0
```

即：**批次交出之后渲染主线程又落盘了，在途批次的 8 次字节读全部成功**。这正是"未来任何
checkpoint 都碰不到它"的那条主张，且是在真的发生了后续 checkpoint 的前提下测的
（`landed_after_handover.is_set()` 是硬断言，测不到就判失败，不会静默变成空测）。

顺带确认两条曾经是隐患的：
- **prefetch 副本被并发回收**：`tools/prefetch.py::_publish` 是 `tmp + os.replace`（原子），
  且 `archive_reader._local_or_prefetch` 把 buffer 上的**任何 OSError 当 miss** 回落归档。
  所以标注线程读 `I_in` 时渲染线程正在重取同一张源图（`max_source_uses=12` 下必然发生）**无害**。
- **sqlite 跨线程**：`ArchiveReader` 用 `check_same_thread=False` + 自带 `Lock`，
  `upsert` 走 copy-modify-rename、读者 `immutable=1` 钉住旧 inode、`_catalog_write_lock` 只锁写者。
  标注线程与渲染线程共用 `_SHARED` reader **安全**。

### 尾巴上界（8 GiB 水位 ⇒ ≤2,292 组）—— 用现场 journal 复核

读 `/mnt/ramstage/prod-l8-local400k-20260812/failures.jsonl`（45,433 行，只读）：

| 量 | 实测 |
|---|---|
| land checkpoint 数 | **1,790** |
| 累计落地组 | **59,511**（= manifest 的 completed.groups，即**全部**已落地） |
| 单次 checkpoint 最大落地组 | **2,321**（NOTES 记 2,292，build 又跑了一天） |
| p99 / 中位数 | **1,994 / 1** |
| 当前 tmpfs 上仍暂存的候选文件 | 13,119 个 ≈ **1,640 组**（= 此刻的"尾巴"） |

**尾巴 = 上次 checkpoint 以来渲染的组 = 下次 checkpoint 会落的组**，二者同分布，
所以上界就是上表的 2,321，**不是一整趟**。NOTES 的"一趟 3 万组里最多约 7% 延到下一轮"成立。

### 最后一轮的尾巴会不会永久滞留 —— 不会，且路径唯一

`_run_phases` 的收尾序列是死的：`_land_checkpoint(force=True)`（3666，**把尾巴全部落地**）→
`_close_annotation_driver()`（3674，join）→ **全量** `_refresh_catalog()`（3681）→
无 `only` 的 `drain()`（3682）→ `pending` 非零才 `PipelineError`。
`_drain_sam3_and_replacements` 里的替补组（`_fill_mode` 不开 `annotate_passes`）同样落在这条路上。
它们的测试形态就是 `test_a_still_staged_winner_waits_instead_of_being_handed_over`：
`batches == 0`（一个批次都没交出）而 `pending == 0`、sft 4 行 —— 收尾 drain 全接了。

---

## 17.2 ② 单写者收窄 —— **pass**（最脆的那条我压测了）

### 逐消费者复核（我读码，不采信 NOTES 的表）

先确认**写者**：`append_group` 的调用点只有 `agent.py:3031 / 3132`（渲染主线程 `_commit_source_result`
路径）与 `3908`（LegacyImportPipeline，不并发）；`append_sft` 只有 `responses.py:1999`（标注 worker）。
**groups 只有主线程写、sft 只有标注线程写、failures 两边都写** —— 这是全部前提。

| 消费者 | 读什么 | 我的复核 |
|---|---|---|
| `_Sam3Ledger.refresh` | `failures` 游标增量 | **见下，压测过** |
| `_landing_counts` | `stage == "landing"` | 标注行 `stage="annotation"`，不匹配；且 landing 行仍只由主线程写 ✅ |
| `_terminal_source_ids` | `terminal ∧ source_id ∧ stage ∈ {rendering, sam3_relabel}` | `ResponsesAnnotator._failure`（`responses.py:1912-1927`）的行**根本没有 `source_id` 键**，`.get()` 得 None，**双重不匹配** ✅ |
| `_pending_sam3_ids` | `error_code == "sam3_relabel_queued"` | 标注 error_code 取值集不含它 ✅ |
| `has_terminal_failure(task_id)` | 按 task_id | 标注 task_id 是 `stable_id("annotation",…)`，与渲染侧 id 前缀不同域 ✅ |
| `checkpoint_summaries` / `i_in_members` / `sft_winners` | landing 行的 message | 同上不受污染 ✅ |
| `mirror_artifacts` | 边写边读账本 | **见下，本卡确实引入了新情况**，但不变式仍成立 |
| resume `_load_unique` | 按 event_id 建索引 | 与顺序无关 ✅ |
| `_failure_counts` / `failures.by_code` | 计数 | 渲染期会开始出现标注类 code —— **预期的可见性变化**，NOTES 已自陈 ✅ |

### `_Sam3Ledger` 的游标假设 —— 我用 17 个线程压

任务卡点名要验的就是这条。自搭（scratchpad `my_ledger_race.py`）：
**16 个"标注 worker"线程**各追加 400 行 annotation 行（含随机 terminal）、
**1 个"渲染"线程**追加 1,200 行 sam3 行（20 个源、三种 event、随机 attempt），
主线程**在整个过程中不停 `ledger.refresh(store.failures)`**，`switchinterval` 压到 1e-6：

```
rows=7600  refreshes=2,557,500  sources_checked=20   → OK
```

断言是**逐源**与两条全表扫 oracle（`_sam3_attempts` / `_unconsumed_sam3_ready`）相等，
外加"从零重折一遍"与增量结果的 `attempts`/`maxima` 两个 dict 完全相等。**全部通过。**

为什么它必然成立（读码给理由，不只给实验）：`refresh` 先取 `end = len(failures)` 再切
`failures[self._pos:end]`，list 只增不减、切片是 GIL 内的一次 C 调用，所以切片窗口稳定；
折叠量是 **计数与 max**，对同一集合的**任意交错顺序**同值；标注行在第一层 `event_type`
过滤就被 skip。**交错不影响，缺的行下次 refresh 补上。**

### 镜像：本卡确实引入了「被镜像的文件正在被另一个线程追加」

先于本卡，`groups/failures.jsonl` 的追加与 `mirror_artifacts` **同在主线程**，不可能重叠；
现在 `sft.jsonl` 与 `failures.jsonl` 可以在 `_mirror_ledger` 拷贝期间增长。逐行看 `:718-733`：
增量段 `_copy_stream(reader, writer)` **不带 limit**、拷到 EOF，`written` 记实际字节，
`landed = target.stat()` 与 `committed + written` 一致 ⇒ 记录的 state 就是"本轮真实落地量"，
镜像仍是源的**字节前缀** —— 不变式没破。极端情况（record 被拷成半条）由
`restore_mirror` 的换行边界截断吃掉，下一轮的尾部比对会因源侧已补齐而 `prefix_diverged` → 整拷自愈。
**代价上界一次整拷，无静默错数据。** 追加速率（≤16 行/s × ~1 KB）远低于 NFS 98 MB/s，不存在追不上 EOF 的活锁。

---

## 17.3 ③ `_next_rounds` 与 `_next_round` 等价 + `_attempt_count` 边界 —— **pass**（1 nit）

### 随机 oracle 对拍（不看它的夹具，自己造）

scratchpad `fold_fuzz.py`：每个种子随机生成 1–12 个任务、1–4 轮、0–60 行 journal，
事件类型在 `{round_exhausted, terminal, attempt, skipped}` 里随机、**行序整体打乱**、
掺入不属于该批次的 outsider 任务、掺入 `stage="rendering"` 的行、`round` 随机取 `{1,2,3,4,None}`、
terminal 随机散布；再随机抽一个子集当 batch：

```
seeds: 400   mismatches: 0
```

**折叠与逐任务定义在 400 组随机 journal 上逐 key 相等。** 乱序 / 多轮 / terminal 交错三种情形全覆盖。

### 唯一一处不等价（不可达，记 nit N-P1）

`_round_done` 用 `row.get("round") == round_number`（严格相等），
`_next_rounds` 用 `int(row.get("round") or 0)`（强制转换）。实测：

| `round` 的值 | 折叠 | oracle |
|---|---|---|
| `"1"`（字符串） | `2` | `1` ← **不等价** |
| `"x"` | **ValueError** | `1` |
| `1.0` / `True` | `2` | `2`（等价） |

生产不可达：`ResponsesAnnotator._failure:1920` 恒写 `round: round_number`（`range` 出来的 int），
journal 是 JSON、int 往返不变形，且没有第二个写 `round_exhausted` 的人（全仓 `def drain(` 只有
`ResponsesAnnotator` 一个真实现）。但"参照定义"的口径应当写明只对 int 成立。

### 与 `_attempt_count` 的边界 —— 计费语义**确实没动**

`_attempt_count` 一行未改，仍是按任务的全表扫，仍在 `run_round` 入口与 drain 的异常分支各调一次。
它读 `store.failures`（list），并发 append 下 list 迭代良定义；过滤条件
（`task_id ∧ stage=="annotation" ∧ event_type=="attempt" ∧ round`）与渲染行、与**其他**任务的行
都不相交，且同一时刻只有一个 drain（driver 串行 + 收尾在 join 之后）⇒ **同一 task 的 attempt 行只有它自己在写**，
这正是实现方说"要逐条论证才敢换"的那条前提 —— 我这里顺手论证了，但同意**不在本卡改**。

成本我实测（现场 journal 45,433 行，只读）：

| 量 | 实测 / 外推 |
|---|---|
| `_attempt_count` 单次 | **14.98 ms** |
| 外推到 30 万行 | **98.9 ms/次** |
| × 4.8 万任务（第一轮） | **76 CPU-分钟 / 轮**，且在**与渲染抢 GIL 的 16 条线程上** |
| 同一 journal 下 `_next_rounds` 折叠（本卡新增） | 全批 **一次**遍历（§P-五 记 14.8 ms，与我 fold_fuzz 的量级一致） |

即：本卡把 `_next_round` 的 4 次/任务全表扫降到 1 次/轮，**剩下的 1 次/任务（`_attempt_count`）
现在是这条路径上唯一的平方项**。重启当天（journal 45k 行）它是 4.8 万 × 15 ms ≈ **12 CPU-分钟**，
可接受；journal 涨到 30 万行后是 76 分钟/轮。**支持实现方"单独排卡"的判断**，见 §17.8。

---

## 17.4 ④ 快照一致性 —— **pass**（全仓普查，无第五个读点）

`rg` 全仓找 `store.groups` / `store.sft` 的**裸**读点，逐个定性：

| 位置 | 读者线程 | 判定 |
|---|---|---|
| `agent.py:1291 / 1316`（`__init__`） | 主，driver 尚不存在 | ✅ |
| `:1413`（`_source_reuse_counts`） | 主 | ✅ groups 只有主线程写 |
| `:1462`（`_live_groups`）→ `_mode_groups` / `_land_groups` | 主 | ✅ 同上 |
| `:1918`（`_verify_group_assets`） | 主，且在 `_fill_mode` 之前 | ✅ |
| `:2276`（`_clean_orphan_assets`） | 主 | ✅ |
| `:2368 / 2398`（`len(store.groups)`） | 主 | ✅ `len` 非迭代 |
| `:3783 / 3799 / 3835 / 3899 / 3930-3937` | **LegacyImportPipeline**，无 driver | ✅ 不在本卡范围 |
| `projection.py:163-165 / 189 / 239` | 主，且在 `_run_phases:3695`，**join(3674) 与收尾 drain(3682) 之后** | ✅ |

换成快照的四处：`state.py` 的 `completed_sources / completed_source_uses /
completed_annotation_tasks / pending_annotation_tasks` + `agent.py:1627 / 1750 / 2128`
（manifest 的 sft 计数、`_annotation_counts`、`_winner_annotation_status`）。
**CanonicalPipeline 里已无一处裸读 `store.sft`**（`len(self.store.sft)` 全部落在 Legacy 段）。

方向也对：groups 的跨线程读者只有标注线程（经 `pending_annotation_tasks`），sft 的跨线程读者只有主线程。
`failures` 不快照的理由我认可（list + append-only + 几十个读点 + 收尾 30 万行）。

快照成本实测（现场 8,000 组样本外推）：`list(self.groups.values())` 在 **40 万组时 6.2 ms**，
持锁期间阻塞 append —— 可接受。

---

## 17.5 ⑤ 收尾语义 / 异常路径 / 孤儿线程 —— **pass**（2 nit）

- **`inflight == 0` 是结构性的而不是被检查的**：`_run_phases:3683` 只查 `pending`。
  但 `close()` 把 `None` 放在队列**末尾**，队列里的批次会被全部取出，
  且 `_loop` 的 `finally` **无条件**减 `_inflight`（错误后被丢弃的批次也走 finally）
  ⇒ join 返回后 `inflight` 恒 0。测试（`DriverTests` 三例 + 三处 manifest 断言）已覆盖。
  建议加一条零成本断言防未来有人把 join 挪走（N-P3）。
- **driver 异常不被吞**：我复跑了变异 M6（`submit` 不重抛 + `_run_phases` 不重抛），
  **3 例 FAIL**，与 NOTES 记的一致（§17.7）。
- **`_close_annotation_driver` 在异常退出路径上也被调用**：`execute()` 的 `finally`（3637）
  在三个 executor shutdown **之前**、`run()` 的 `with ArtifactStore` 关店**之前**，顺序正确
  （store 关早了会让 driver 的 append 全部 `StateError`）。
  我实测了失败 build：`test_a_failed_build_leaves_no_annotation_thread`（我写的），
  抛 `RuntimeError` 之后 `threading.enumerate()` 里**没有** `databuild-annotate` 残留。
- **新的运维锐边（N-P6）**：driver 是 `daemon=False` 且 `close()` 先排空队列。
  L8 第一个批次是 **4.8 万个任务**，一次 `drain(only=…)` 会跑满 3 轮。
  于是 **Ctrl-C 一次不够**：主线程进 `finally` → join → 等当前批次（可能数小时，且期间继续花钱）。
  第二次 Ctrl-C 会在 join 里抛出、`with` 关店、driver 的下一次 append 抛 `StateError` 从而快速收摊。
  改动前中断只需等 ≤16 个在途任务（`ThreadPoolExecutor.__exit__`）。**这条必须让运维知道。**

---

## 17.6 ⑥ resume —— **pass**（我自己写的夹具）

不复用它的 `ResumeTests`，自搭三例（scratchpad `my_resume.py`），全绿：

| 我的用例 | 断言 | 结果 |
|---|---|---|
| `test_partial_annotation_resumes_without_paying_twice` | 第一次运行每批只答第一条 → build 失败；重启后**跨两次运行**把所有被"报价"过的 task_id 汇总，断言 `len(all) == len(set(all))`，且重启后**从未**被提供第一次已完成的任务，最终 sft 8 行 / 8 个不同 `annotation_task_id` | **OK** |
| `test_rendered_but_never_annotated_is_requeued_on_restart` | 现场同形（4 组已渲、sft 0 行）：重启后 `CALLS[0] == (是批次, 8)`（**第一个边界就把全部积压交出**）、`CALLS[-1] == (收尾drain, 0)` | **OK** |
| `test_a_failed_build_leaves_no_annotation_thread` | 见 §17.5 | **OK** |

计费的关键在 `pending_annotation_tasks` 的 `done` 集合从 **sft 行 + terminal 事件**反推，
而 task_id 是 `stable_id("annotation", group_id, candidate_id, rank)`（确定性）⇒ 重启后同一任务同一 id、
已完成的不再 pending。`_annotation_submitted` 是**进程内**台账，只防同一次运行内的重复交付；
跨重启由 durable 状态防 —— 两层各司其职，我的用例把两层都测到了。

现场形状复核（只读 manifest + journal）：`completed.groups = 59,511`、
`landing` 累计 `winners = 48,160`、`annotation.pending = 48,160` —— **两者相等**，
即**积压任务 100% 已落地已 unlink**，重启后第一个边界全部通过 settled 门。NOTES §P-八 的判断成立
（它记的 46,337 是两天前的数）。

---

## 17.7 ⑦ 变异 / 回归 / 红线 —— **pass**

### 变异复跑（抽 5 个，我自己写补丁，跑在影子包上）

| # | 变异 | 我的结果 | NOTES 记的 |
|---|---|---|---|
| M1 | `_annotation_bytes_settled` 恒 True | **FAIL 2**（`…still_staged_winner_waits`、`…settled_test_follows_the_local_file`） | FAIL 2 ✅ |
| M4 | `_flush_annotation_batch` 挪到 `if final: return` 之后 | **FAIL 3**（`…single_pass_build_still_pipelines`、`…last_pass_error_surfaces`、`…landed_winner_is_handed_over`） | FAIL 3 ✅ |
| M5 | 去掉 `self._annotation_submitted |= batch` | **FAIL 5** | FAIL 5 ✅ |
| M6 | driver 的错误被吞（`submit` + `_run_phases` 都不重抛） | **FAIL 3**（`DriverFailureTests` 两例 + `…first_error_is_kept_and_re_raised`） | FAIL 3 ✅ |
| M7 | 边界处不做 `_refresh_catalog(incremental=True)` | **FAIL 1**（`…read_out_of_the_archive`） | FAIL 1 ✅ |

**五个的失败集合与 NOTES 逐个相同**，没有一个是"改了但测试照绿"。影子包基线 27/27 OK。

### 回归

| 跑法 | 结果 |
|---|---|
| `tests.test_annotation_interleave -v`（新，27 例） | **27/27 OK** |
| `test_mirror_cadence / test_land_watermark / test_source_reuse / test_sam3_drain_ledger / test_sam3_ready_index / test_sam3_batch_fallback / test_land_integration / test_canonical_orchestration` | **26 / 18 / 40 / 21 / 6 / 2 / 19 / 15，全 OK** |
| 全套 `discover` | **Ran 541，failures 0，errors 31** |

31 条噪音逐条核对：`30 × FileNotFoundError: databuild.example.toml` + `1 × ModuleNotFoundError: uvicorn`，
与 §15.6 / §14.6 的已知名单**完全一致**，本改动 **0 个新增失败**（总数 514 → 541 = 新增 27 例）。

既有测试的三处改动我逐条看过：`test_canonical_orchestration` 的 `FakeAnnotator` 加 `only` 支持
与那条 `manifest["phase"]` 从 `"annotation"` 改 `"rendering"` 的断言 —— **这是真实语义变化且是对的**
（中断现在发生在渲染相），该用例其余断言（只重标未完成的 3 个 / id 稳定 / 无重复）原样通过；
`test_land_integration` 的 double 只加 `only`，**判据一条未改**。

### 协议红线扫描

| 红线 | 涉及 |
|---|---|
| AUC 作空间场判据 / IoU / 边界 F1 / 中心先验 | 无涉。三个源文件 + 新测试 grep `auc|roc_` 仅命中 `responses.py:490` 一条**先于本卡**的 prompt 提示词注释（文本标注留出集，非空间场判据） |
| 空间场可视化（逐图 min-max / 色标 / 叠图 resize） | 无涉。`resize` 唯一命中是 `_encode_image` 的 LANCZOS 缩图（送模型的图，先于本卡） |
| G 初始化 / σ 裸 exp / s 轴正则 / 逐像素算子 / ckpt 选择 / s 归一化 / Δ 列 / IoU 当目标 / 干预对象 / 烘焙一致性 / attention eager | 无涉（纯编排与 IO） |
| s 缓存消费契约 | 无涉 |
| 长任务提交纪律（noclobber / `pgrep` 判活） | 无涉：新代码零 `pgrep`/`nohup`。**我自己**跑全量回归时用了后台提交，按纪律先 `rm -f` 日志、用 `ps -p <PID>` 实证存活，未用 `pgrep` |
| NFS 纪律 | **遵守**：只读 `/mnt/nfs-ro`（一次 readdir + 一次 glob），未写 `/mnt/nfs`、未起 `nfsx` |
| `trash/` | 未读未写 |
| 运行中的 build（pid 1790282） | **未碰**。其 manifest 无 `annotation.inflight` 键 ⇒ 新代码确未生效 |

---

## 17.8 ⑧ 对实现方两个建议的独立意见 + 重启操作提示

**建议一：`_attempt_count` 单独排卡（HOTFIX-AnnotAttemptIndex-1）—— 同意，但排在重启之后。**
理由是我实测的数字（§17.3）：重启当天 journal 45k 行，第一批 4.8 万任务的 `_attempt_count`
合计约 **12 CPU-分钟**（GIL 内），相对于要跑数小时的 4.8 万次 relay 往返可以忽略；
真正变贵是在 journal 涨到 30 万行之后（76 分钟/轮）。**本卡把 33 分钟/次 drain 降到毫秒级已经拿走了大头**，
剩下这块不值得为它推迟重启窗口。改的时候必须带上"本轮内该 task 的 attempt 行只有它自己在写"的
论证（我在 §17.3 给了半条：driver 串行 + 收尾在 join 之后 ⇒ 全局只有一个 drain）。

**建议二：relay 并发 16 → 8 —— 不同意先降，建议先看数。**
本机 48 核，当前 load 28.6（渲染独占）。标注侧新增的是每任务 2 张图的 PIL 解码 + LANCZOS + JPEG
（PIL 这三步大多释放 GIL，属于抢核不抢 GIL），16 路 relay 的**稳态**并发受限于外部端点的回包速率
而不是本地 CPU。先降到 8 等于把本卡的收益直接砍半。**判据建议**：重启后第一小时对比
`groups/h` 与重启前 baseline（现场 35 h / 59,511 组 ≈ 1,700 组/h），掉幅 >15% 再降并发。

**重启操作提示（不是缺陷，是必须提前知道的两条）**：

1. **第一个边界会静默数分钟。** `mix.local=1.0 ⇒ global_target=0`，`_fill_mode("global", (), 0)`
   空走一趟立刻触发第一个 `_flush_annotation_batch`，而此时 `_registered_datasets` 为空
   ⇒ `_refresh_catalog(incremental=True)` 退化为**全量**：现场已有 **2,556 个 batch**
   （groups 1,790 + sft 766，我从 `/mnt/nfs-ro` 数的），`_landed_datasets` 的 NFS glob
   实测**冷 14.8 s / 热 2.8 s**，`upsert` 还要把 **21.4 GiB** 的 `global.sqlite3`
   整份 copy-modify-rename（`_upsert_locked` 的整份拷贝**与命名批次数无关**，只要 `wanted` 非空就付）。
   **这几分钟里两张卡是空的**，属于设计内的一次性成本，不要误判成卡死。
2. **渲染期 manifest 的 `annotation.pending` 从此正常非零**，完成判据只在收尾 drain 之后成立；
   区分字段是新增的 `annotation.inflight`（在途任务数）与 `annotation.batches`（已交出的整趟数）。

---

## 17.9 nit 清单（7 条，均不阻塞）

| # | 位置 | 内容 |
|---|---|---|
| **N-P1** | `responses.py::_next_rounds:2123` | `int(row.get("round") or 0)` 与 `_round_done` 的 `== round_number` 在非 int `round` 上不等价：`"1"` → 折叠判为已用尽（oracle 判未用尽）、`"x"` → **ValueError**。生产不可达（`_failure` 恒写 int，全仓只有一个 drain 实现），但既然 `_next_round` 被声明为"参照定义"，等价性就该写明"仅对 int round 成立"，或在折叠里加同款容错 |
| **N-P2**（最值得处理） | `agent.py::_annotation_bytes_settled:2221` | 判据是「本地文件不在」⇒「已落地」，这是**推论**，前提是"除 `_land_groups` 外没有第二个删 `after_path` 的人"。我逐条穷举过六个候选删除者（§17.1），今天成立，但它没有写进代码，未来任何一个新的资产回收器都会静默地把这条推论变成假的、后果是**整批 winner 被判 terminal 且不可重试**。两种收法：(a) docstring 里点名 `_clean_orphan_assets` 的 `referenced` 集合是守卫；(b) 判据改成 `path_exists(after, db_path=…)`（把推论换成实证）——代价是每任务一次 sqlite 点查，L8 首个边界 4.8 万次 × 15 µs ≈ **0.7 s**，可接受 |
| **N-P3** | `agent.py:3683` | 收尾只查 `pending`，`inflight == 0` 由 join 结构性保证但无断言。加一句 `assert self._driver is None or self._driver.inflight == 0` 是零成本的防回归（防未来有人把 join 挪到 drain 之后） |
| **N-P4** | `_flush_annotation_batch` docstring + NOTES §P-七.3 | "costs one queue walk and no catalog copy" 属实（空批次在 `_refresh_catalog` **之前**就 return 了），但没写出有批次时的真实成本：一次 `_landed_datasets` 的 NFS glob（实测 2,556 batch：冷 **14.8 s** / 热 **2.8 s**，且随 batch 数线性增长）+ 一次 **21.4 GiB** 的 sqlite 整份拷贝。建议把这两个数写进 NOTES，否则下一个人会以为 `incremental=True` 把整份拷贝也省掉了 |
| **N-P5** | `responses.py::_encode_image:414` | `open_rgb(source)` **不带 `db_path`** ⇒ 恒读 `default_db()`；而登记走 `dependencies.catalog_db or default_db()`。生产 `catalog_db=None` 故一致，但这是"两个默认值恰好相同"而非一处真值；哪天有人给生产设了 `catalog_db`，标注会静默读错目录、**全部 winner 判 `annotation_image_invalid`（terminal）**。先于本卡存在，但本卡让"标注读归档"成为常态，值得顺手把 `db_path` 传下去 |
| **N-P6** | `_AnnotationDriver.close` / `execute` 的 finally | **Ctrl-C 一次不再够**：driver 非 daemon，`close()` 先排空队列再放哨兵，而 L8 第一个批次是 4.8 万任务的一次 `drain`，join 可能等数小时（期间继续花钱）。改动前中断只等 ≤16 个在途任务。最小修法：`close(discard=True)` 时先把队列里**尚未开始**的批次取空再放哨兵（在途那个仍要等，但至少有界）。**运维必须知道要按两次** |
| **N-P7**（范围外观察） | `_manifest:1691` 的 `len(self.store.pending_annotation_tasks())` | 实测：**59.5k 组 0.47 s / 外推 40 万组 3.19 s**，而每个 land checkpoint（有 mirror_root 时）都写一次 manifest —— 现场已 **1,790 次**（中位数每次只落 1 组）。这是**先于本卡**的平方项，本卡未加重（新增的 `inflight`/`batches` 是 O(1)），但流水化之后它压在与标注抢 GIL 的渲染主线程上。建议与 N-P1 同批排一张卡：把 `annotation.pending` 做成增量计数 |

---

# **PASS。0 blocker。PIPE-PassInterleave-1 可以带进重启窗口。**

> 唯一建议在重启**之前**顺手做的：**N-P3 的一行断言**（零成本）与 **N-P6 的运维告知**（不改代码也行，
> 但必须写进重启 runbook：Ctrl-C 要按两次）。其余五条都可以排到重启之后。

---

# 18. 终审：PIPE-StopFill-2（停渲标记 + 无 GPU 排干）—— L8 排干重启唯一门禁

> 审阅人：实现审阅 subagent（独立，与编码 agent 无关）｜ 日期：2026-08-14
> 审阅对象：`dataset_build/src/construct/agent.py`（md5 `55cd254591cb121efc72aeda4a4624ea`，
> 开工取快照、收工复核**未变**）、`tests/test_stop_fill.py`（40 例 / 3 subtest，`e69a17fc…`）、
> `tests/test_sam3_drain_ledger.py`（`4c384350…`）、`state.py`（`28c89cab…`，本卡未改）
> 依据：NOTES §S-一~八、任务卡七项、本文件 §12/§15/§16/§17
> 纪律：只读审阅；一切变异与自写夹具都在影子副本
> `scratchpad/wt/`（`rsync` 出的整树）与 `scratchpad/head_tree/`（`git archive HEAD` 出的干净树）上做，
> **工作树四个文件全程一字未改**（md5 首尾一致，见上）；未读也未写 `trash/`；
> NFS **一次也没碰**（既没读 `/mnt/nfs-ro` 也没写 `/mnt/nfs`，未起 `nfsx`）；
> `/mnt/ramstage/prod-l8-local400k-20260812` **只读**（manifest / 三个 journal 的行数与 error_code 直方图）；
> 无运行中的 build（`ps -eo pid,cmd` 仅两个 viewer 进程，无 `construct.agent`）。

## 18.0 结论

# ✅ **PASS —— 0 blocker，可立即以 CPU 任务重启排干**

| 审阅项（对应任务卡） | 判定 | 依据 |
|---|---|---|
| ① `_journal_once` 声明式偏离（判重键 / 相容性 / HEAD bug 亲手复现） | **pass（2 nit）** | §18.1，**干净 HEAD 上亲手复现了崩溃**，修后同一夹具全绿；6 个构造攻击我自己写并跑 |
| ② 无 GPU 证明（import 链 + 端到端 + 五加载点 + `is_initialized` 门） | **pass** | §18.2，我把 `sys.meta_path` 装了 torch 绊线，**整条排干链路在"import torch 即失败"的解释器里跑完** |
| ③ 断点位置（在途窗口排空 / 不半渲 / 截断 pass 仍交接） | **pass（1 nit）** | §18.3，**生产 `source_window=5` 下我自己复测**（既有测试只覆盖 window=2） |
| ④ SAM3 相跳过（队列保留 / 不花预算 / 收尾守卫） | **pass** | §18.4，我自搭三段式（停渲→重启→去标记）实测队列 1→1→0、attempt 0→0→2 |
| ⑤ 可逆性 | **pass** | §18.5，既有 3 例复跑 + 我加的两个"标记在 run() 读后被创建/删除"窗口用例 |
| ⑥ latch 语义（单调 / 计算-构造窗口） | **pass（1 nit）** | §18.6，两个方向的窗口我都写了夹具，答案都自洽 |
| ⑦ 9 变异抽 4 + 40 例 + 全量回归 + 31→30 溯源 + 红线 | **pass** | §18.7，四个变异失败数与 NOTES **逐个相同**；`552 passed / 30 failed`；31→30 已定位 |
| **合计** | **0 blocker / 6 nit** | 全部不阻塞 |

**一句话**：本卡最该被怀疑的是那条"顺手修的 HEAD 上的雷"——我没有采信 NOTES 的复现，
而是用 `git archive HEAD` 拉了一棵干净树、换上**真实时钟**自己写夹具，
**在没有本卡任何代码的 HEAD 上把崩溃复现了出来**（`run 1: complete_with_failures 2` →
`run 2 RAISED StateError: conflicting durable failure event`）；再把同一条轨迹放到本卡的树上，
连跑 5 次全绿。更要紧的是我在 L8 现场核出这个 bug **不是理论风险而是必经之路**：
`_record_shortfalls` 排在标注排干**之前**，而 L8 的 relay 现在带着 329 条
`untyped_stream_event` + 78 条 `annotation_round_exhausted`、49,741 条待标注只产出过 311 行 sft ——
**排干几乎一定会死一次以上，第二次重启就会撞上这颗雷**。所以这不是"超范围的顺手修"，
而是本卡交付物能不能用第二次的前提。**可以重启。**

---

## 18.1 ① `_journal_once` —— **pass**（2 nit）

### (a) 干净 HEAD 上的 bug：我自己复现的

不采信 NOTES 的复现。`git archive HEAD dataset_build | tar -x` 出一棵**不含本卡任何代码**的树
（HEAD 的 `_record_shortfalls:2404` 是 `stable_id("target", build_id, mode)`，直调 `_failure`），
自写夹具：`target=8`（local/global 各 4）配 **2** 个源必然短量，
并把 fixture 冻死的 `now` 换成 `datetime.now(timezone.utc)`（**这正是它一直藏着的原因**）：

```
RUN1 complete_with_failures 2
RUN1 failure codes: ['global_target_shortfall', 'local_target_shortfall']
RUN2 RAISED StateError conflicting durable failure event: failure_1c986cb7208a52a506115b76c0b35c81
```

机制与 NOTES 所述一致，我逐行核过：`_failure:1415` 的 `event_id` 由
`(build_id, event_type, stage, task_id, error_code, round, attempt, group_id, candidate_id, endpoint_id)`
构成，**不含 `message` 也不含 `timestamp`**；`state.py:337` 只对逐字节相同的行放行。
所以固定 `task_id` ⇒ 固定 `event_id` ⇒ 第二次写只差 `timestamp` ⇒ `existing != stored` ⇒ 抛。

修后（工作树版本）同一夹具：**三次连跑静默**（`failures.jsonl` 的 task_id 列表逐次相等），
且填得更多时会记新数字而不是被拒：

```
first : ['... 12 local groups, completed 2', '... 12 global groups, completed 0']
second: ['... 12 local ... completed 2', '... 12 local ... completed 6', '... 12 global ... completed 0']
```

我自己写的这两个夹具**不是空转**：把守卫去掉（变异 M5 = HEAD 行为）后两条**全部转红**。

### (b) 判重键的构成：6 个构造攻击（我自己写的 `test_zz_journal_once_attacks.py`，全跑）

| 攻击 | 期望 | 实测 |
|---|---|---|
| 同一 run 的 local + global shortfall | 两条不同行 | ✅ 两个不同 task_id，两个 error_code |
| shortfall 与 stop 事件混淆 | 永不同 | ✅ `stable_id` 把 kind 既进 payload 又做 id 前缀（`target_…` / `stop-fill_…`） |
| 重启但数字没变 | 静默 | ✅ 三次连跑 0 新增 |
| 重启且填得更多 | 记新数字，旧行留着 | ✅ 见上 |
| 停在不同位置的两次 stop | 两条 | ✅ `(at_start, local, global)` 就是身份 |
| shortfall 行污染源/SAM3 台账 | 不可见 | ✅ `source_id=None`，`_terminal_source_ids:1565` 要求 `source_id` 真值；`manifest.sources.terminal == 0` |

**唯一能把两个不同事件判成同一个的两条路径，我都构造出来了，都不阻塞**（见 nit N-S1 / N-S2）。

### (c) 与四项既有不变式的相容 —— 逐个核过

1. **`append_failure` 的逐字节判重**：`_journal_once` 命中时**直接 return**，根本走不到
   `append_failure`；未命中时写的是全新 `event_id`。两套判重不重叠。
   反过来它还**顺手修好了一类更隐蔽的冲突**：`local_target` 由 `allocate_sources` 用
   `inventory.eligible` 算，**不受 `_resume_config_differences` 保护**，所以两次 run 的
   `target` 可以不同而 `completed` 相同 —— 这时 `event_id` 相同、`message` 不同，
   HEAD 会抛，现在会静默。
2. **`event_id` 唯一性**：`ArtifactStore.__init__:219` 用 `_load_unique(…, "event_id", …)` 开库，
   一个 task_id 至多一行 ⇒ event_id 唯一，开库不会炸。
3. **`_Sam3Ledger` 游标**：`refresh:1224` 只折 `sam3_attempt/ready/ready_invalid` 三种
   `event_type`，本卡两种事件被 `continue` 掉；且 `state.py:342` 全仓**只 append 不改序**
   （我 grep 了 `self.failures` 的全部 6 处引用），游标假设不破。
4. **镜像前缀不变式**：本卡只在 journal 尾部追加，前缀字节零改动，`agent.py:655` 的
   "mirror is a byte-exact prefix" 成立。
5. **buffered 失败路径**：`_failure:1438` 会在 `self._source_context.failures` 非 None 时改走缓冲。
   该属性只在 `_render_source_buffered:3189` 设、`:3225` 删（worker 线程），
   而 `_stop_fill` / `_record_shortfalls` 的 6 个调用点（3396/3448/3518/3711/3799/3882）
   **全在主线程**，`getattr` 恒为 None ⇒ 判重看到的和写入的是同一个 list。

### (d) L8 现场的兼容性 —— 亲自核

`/mnt/ramstage/prod-l8-local400k-20260812/failures.jsonl`（46,258 行）的 error_code 直方图：
`visibility_rejected 36346 / sam3_relabel_queued 7702 / land_checkpoint 1790 / untyped_stream_event 329 /
annotation_round_exhausted 78 / network_error 5 / major_exhausted 3 / mirror_full_copy 3 / prose_violation 2`。
**没有任何 `*_target_shortfall` 行，也没有 `stop_fill_requested` 行** ⇒ task_id 形状变更零兼容问题，
与 NOTES 所述一致。另 `rg` 全仓确认 `target_*` 的唯一生产者是 `agent.py:3815`、
`stop-fill_*` 的唯一生产者是 `agent.py:3604`，无第三方消费方。

### (e) 这不是"超范围"，是必需 —— L8 轨迹实测

`_run_phases` 的次序是 fills → sam3（跳过）→ **`_record_shortfalls`** → land → release →
**annotation drain** → 收尾 `raise PipelineError("annotation queue remains unresolved")`。
即 **shortfall 写在排干之前**。我按 L8 真实形态写了 `test_zz_l8_trajectory.py`：
渲 3 组、标注全拒 → 打标记 → **连续 3 次停渲重启，每次都在排干处 `PipelineError`** →
第 4 次 relay 恢复 → 收工。

```
restart 1: shortfall rows=1 stop rows=1
restart 2: shortfall rows=1 stop rows=1
restart 3: shortfall rows=1 stop rows=1
final: complete_with_failures groups 3 sft 6 pending 0
```

同一个文件在 M5（去掉守卫）下 **`StateError: conflicting durable failure event`**。
结合 (d) 的 relay 健康度（49,741 待标注 / 历史只产出 311 行 sft / 329 次 untyped stream），
**"排干要重启不止一次"是预期而非意外**，这条修复必须带上。

---

## 18.2 ② 无 GPU 证明 —— **pass**

### (a) import 链：复跑通过

```
$ python -c "import construct.agent, construct.responses, construct.projection;
             from dataset_build.tools import archive_reader, land, global_catalog, prefetch;
             print('torch' in sys.modules)"
torch in sys.modules: False   (323 个模块)
```

我还多跑了一步 NOTES 没做的：`default_dependencies()` **真的调用一次**（它把
`LocalGpuOnlyRenderer.create` / `OneAlignScorer.create` / `_default_relabeler` 都装进 dataclass），
以及 `preflight_openai_sdk` 的导入 —— `torch in sys.modules` 仍为 **False**。
`awk` 扫 `construct/*.py` 的**行首** import：**module 级 torch import 一个都没有**，
6 处在 `visibility.py`、2 处在 `rendering.py`、1 处在 `agent.py:929`，全在函数体内。

### (b) 端到端：比 NOTES 更强的做法（我装了绊线）

NOTES 的子进程断言是 `torch.cuda.is_initialized() == False`。我改成**更强的命题**：
往 `sys.meta_path` 插一个 finder，**任何 `import torch` / `import torch.*` 直接抛 AssertionError**，
然后在这个解释器里跑完整条排干。分两个进程（phase1 正常渲染建积压 / phase2 全新解释器带绊线）：

```
phase2 status complete groups 3 sft 6 annotation_pending 0
phase2 stop_fill {"at_start": true, "gpu_resources_loaded": false, "marker": ".../STOP_FILL", "requested": true}
phase2 sam3      {"attempt_events": 0, "completed": 0, "expected": 0, "pending": 0, "terminal": 0}
phase3 status complete sft 6
torch tripwire: None
torch in sys.modules: False
OK
```

绊线本身经过验证有效：第一版我把它装在 phase1（正常渲染）上，它**立刻在
`visibility.py:926 prepare_torch_lab_reference` 抓到了 import**。所以 phase2 的 `None` 是真阴性。

### (c) 五个加载点逐个对照代码

| # | 加载点 | 代码 | 停渲去向 | 我的核实 |
|---|---|---|---|---|
| 1 | `renderer_factory` | `:531` = `LocalGpuOnlyRenderer.create` | `run():4252 if not stop_fill` 内 | ✅ 读码；测试把工厂换成"一被调用就 AssertionError"并断言 `.called is False` |
| 2 | `renderer.assert_ready()` | `:4255` | 同分支 | ✅ |
| 3 | `_load_scorer` | `:4256` → `:532/:542` OneAlign | 同分支 | ✅ 两个工厂都断言未被调用 |
| 3b | `_preflight_scorer` | `_load_scorer:1098` | 3 不调则不可达 | ✅ |
| 4 | `relabeler` | `:535` `_default_relabeler` | **全仓唯一调用点 `:3761`**，在 `_drain_sam3_and_replacements` 内 | ✅ `rg dependencies.relabeler` 只有 1 个真调用点 + 2 条注释 |
| 5 | `_empty_cuda_cache` | 定义 `:917`，调用点仅 `:3634`（`_release_heavy_resources`）与 `:3759`（SAM3 批次） | 前者在 `renderer is None and scorer is None` 时**先 return**，后者所在相被跳过 | ✅ 两个调用点在停渲下都不可达；`torch.cuda.is_initialized()` 门是纵深防御 |

`_empty_cuda_cache` 的门确实必要而非保险：它函数体第一行就是 `import torch`，
无门时**光是调用它就把 torch 拉进进程**。变异 M6（去门）杀死 1 例。

### (d) 仍会跑的 CPU/DB 依赖（运维须知，非缺陷）

`sdk_preflight` → `catalog_loader`（preset bank jsonl）→ `inventory_loader`（Postgres）
在 `stop_fill` 分支**之外**，停渲 build 照跑。理由成立（`local_target` 是 shortfall 的分母），
但意味着这个"CPU 任务"仍需 Postgres 与 preset bank 可达（见 N-S6）。

---

## 18.3 ③ 断点位置 —— **pass**（1 nit）

读码链条：`_fill_initial_mode:3518` 的 `if self._stop_fill(): break` 是
`for index, source in enumerate(sources)` 的**体首行**，位于"上一个源已提交"与"下一个源被提交"之间；
`break` 之后落到 `:3568` 的 `while inflight: finish_oldest()`，
与一趟正常结束**走同一段代码**，其中 `if not inflight: self._land_checkpoint()`。
`_fill_mode` 侧的次序是 `_flush_annotation_batch()` → `if self._stop_fill(): return`，
即被截断那趟的赢家**先交接再返回**。

**既有测试只覆盖 window=2**（`self.dependencies(...)` 的 `scorer_pool_factory is None`
⇒ `_source_window = min(2, gpu_concurrency)`），而生产是 `source_window=5`
（`databuild.prod-l8-local400k-20260812.toml:66`）。我自己补了 window=5 的复测
（`test_zz_boundary_and_latch.py`，把 `scorer_pool_factory` 设成非 None 才会走 `config.render.source_window`）：
20 源池、标记落在**第 1 次 commit** 上（在途最多的时刻），断言

* 落库组数 **== 已 commit 次数**（在途窗口一个不少地提交完，没有被取消的）；
* `>= 5`（确实是整个窗口，不是只剩 1 个）且 `< 20`（走真的停了）；
* 每组 **8 个 candidate + 有 winner**（没有半渲组）；
* `annotation.batches == 1 / inflight == 0 / pending == 0`，`sft == 2 × 组数`（截断趟仍交接）；
* `groups_assets_lost == 0`。

另加一例"标记落在最后一次 commit 上"：6/6 组、`pending 0`、`status complete`
（目标恰好达成，非 terminal 的 stop 事件不会把它变成 `complete_with_failures` —— 语义正确）。

变异 M1（去掉趟内 break）杀 2 例，与 NOTES 相同。

---

## 18.4 ④ SAM3 相跳过 —— **pass**

三道防线读码确认：`_run_phases:3882` 的 `if not self._stop_fill()`（整相不进）、
drain `while` 顶部 `:3711`（轮次之间的安全边界）、收尾 `:3799` 的
`if self._pending_sam3_ids() and not self._stop_fill(): raise`。

我自搭三段式（不看它的测试）：4 源、`border_from=2` 造出队列，标记在源入队瞬间落下，
然后**重启（标记仍在）**，最后**去掉标记再跑**：

```
run1 → queued 行落库，sam3 {pending: 1, attempt_events: 0}
run2 → sam3 {"attempt_events": 0, "completed": 0, "expected": 1, "pending": 1, "terminal": 0}   # 队列原样，预算未花
       且 relabeler / renderer / scorer / scorer_pool 四个 mock 全部 .called is False
run3 → sam3 {"attempt_events": 2, "expected": 2, "pending": 0, "terminal": 2}, relabeler called: True
```

即**跳过不是静默吞掉**：去掉标记后同一份状态里 drain 真的跑、真的花预算、真的收尾。
L8 现场的 7,702 条 `sam3_relabel_queued` 我在 `failures.jsonl` 里数到了（见 §18.1(d)），
与 manifest 的 `expected/pending = 7702` 一致。
变异 M4b（去掉轮次间检查）杀 1 例，与 NOTES 相同。

---

## 18.5 ⑤ 可逆性 —— **pass**

既有 `ReversibilityTests` 3 例复跑全绿（删标记重启 → 3 组填满、`gpu_resources_loaded` 回 True、
`effective_config` 两次逐键相等、开→关→开三次切换）。
我另加的两个窗口用例（§18.6）与 §18.4 的 run3 都从不同角度证了同一件事。
NOTES 声明式偏离 §S-五.2（填满后 status 仍是 `complete_with_failures`）我认同并复核过：
terminal 行是历史陈述，删 durable 行才能改，不该做。

---

## 18.6 ⑥ latch 语义 —— **pass**（1 nit）

**单调性**：`_stop_fill():3592` 一旦置 `_stop_fill_latched = True` 就不再 stat。
我抓住 `CanonicalPipeline` 实例，在 build 跑完后**删掉标记**再连问 5 次，
答案 `[True]*5`。半趟一个状态、半趟另一个状态的情形不存在。

**`run()` 计算与 pipeline 构造之间的窗口**，两个方向我都写了夹具：

| 场景 | 注入点 | 实测 manifest | 是否自洽 |
|---|---|---|---|
| 读到"无标记"→ 随后标记出现 | 包 `renderer_factory`，建完 renderer 再落标记 | `requested true / at_start false / gpu_resources_loaded true`，`groups 0`，`complete_with_failures` | ✅ 卡已经加载了，如实说；填充由趟内 poll 停住 |
| 读到"有标记"→ 随后标记被删 | 包 `agent.ArtifactStore`，建完 store 再 unlink | `requested true / at_start true / gpu_resources_loaded false`，`groups 0`，四个 GPU 工厂 `.called is False` | ✅ **没加载 renderer 就绝不能填**，显式传值正是为此 |

第二行就是"为什么传而不是让 pipeline 自己再读一遍"的实证：若 pipeline 重读，
它会认为可以渲染，而 `self.renderer is None` ⇒ 撞 `:2817` 的 `PipelineError`。
变异 M7（只在启动读、运行中不 poll）杀 **13** 例，与 NOTES 相同。

---

## 18.7 ⑦ 变异 / 回归 / 红线 —— **pass**

### 变异复跑（抽 4 个，锚点我自己写，跑在影子树上）

| # | 变异 | 我的结果 | NOTES 记的 |
|---|---|---|---|
| M1 | 去掉趟内边界（只在趟间查） | **FAIL 2**（`…cut_short_pass_still_hands_its_winners_over`、`…sources_already_in_flight_still_commit`） | 2 ✅ |
| M5 | 去掉 `_journal_once` 守卫（= HEAD 行为） | **FAIL 2**（`…a_short_build_can_be_restarted`、`…a_stopped_build_can_be_restarted_to_keep_draining`）+ 我自己的 3 个夹具全红 | 2 ✅ |
| M7 | 标记只在启动读、运行中不 poll | **FAIL 13**（名单见日志） | 13 ✅ |
| M4b | 去掉 drain 轮次之间的检查 | **FAIL 1**（`…marker_arriving_during_the_phase_retires_it_between_rounds`） | 1 ✅ |

四个的失败集合与 NOTES **逐个相同**，没有"改了但测试照绿"。影子树基线 40/40。
每次变异后都 `cp` 还原并核 md5（收工时影子树与工作树 agent.py 同为 `55cd2545…`）。

### 回归

| 跑法 | 结果 |
|---|---|
| `pytest dataset_build/tests/ -q -p no:randomly` | **552 passed / 30 failed / 7688 subtests passed**（107 s） |
| 30 条失败的构成 | **全部 60 处命中 `example.toml`**，分布 `test_canonical_foundation 11 / test_iaa_batch 9 / test_source_window_and_cgt 5 / test_winner_margin 5` —— 即 D1 的已知名单 |
| 点名套件（工作树上跑，非影子） | stop_fill **40** / interleave **27** / mirror **26** / watermark **18** / source_reuse **40** / sam3_drain_ledger **21** / sam3_ready_index **6**，全绿 |
| `python -m unittest discover -s dataset_build/tests -t .` | **Ran 581, errors=30** |

`552 − 40（新套件） = 512`，与 NOTES 的"改前基线 512/30"自洽。

### 31 → 30 的溯源（任务卡点名）——**是环境，不是口径，也不是本卡**

§17 记的是 `Ran 541 / errors 31 = 30 × example.toml + 1 × ModuleNotFoundError: uvicorn`。
我把两种跑法都跑了，现在 pytest 是 `30 failed`、unittest 是 `errors=30`，
**两种口径给出同一个数**，所以不是统计口径变了。少掉的那一条就是 uvicorn：

* 缺失者是 `test_canonical_responses.py:1855` 的 `BrokerResponsesTests`
  （`from dataset_build.core.broker.app import …` → `uvicorn`）；
* `/home/bc/.venvs/iaa437/pyvenv.cfg` 写着 `include-system-site-packages = true`，
  `uvicorn 0.41.0` 装在 `/home/bc/miniconda3/lib/python3.13/site-packages`（目录时间 **2026-02-20**）；
* 现在单跑 `BrokerResponsesTests` 是 **1 passed**。

结论：**该错误消失是解释器/site-packages 可见性造成的**（§17 那次多半用了看不到 system
site-packages 的解释器），与本卡无关；`Ran 541 → 581` 的 +40 恰是新套件，无新增失败。

### 协议红线扫描

| 红线 | 涉及 |
|---|---|
| AUC 作空间场判据 / IoU / 边界 F1 / 中心先验 | 无涉。`agent.py` 与 `test_stop_fill.py` grep `auc|roc_` **零命中**（本卡不产生任何指标） |
| 空间场可视化（逐图 min-max / 色标 / 叠图 resize） | 无涉，两文件 grep `min-?max` 零命中 |
| G 初始化 / σ 裸 exp / s 轴正则 / 逐像素算子 / ckpt 选择 / s 归一化 / Δ 列 / IoU 当目标 / 干预对象 / 烘焙一致性 / attention eager | 无涉（纯编排与 IO） |
| s 缓存消费契约 | 无涉 |
| 长任务提交纪律（noclobber / `pgrep` 判活） | 新代码零 `pgrep` / `nohup`。**我自己**后台跑全量回归时按纪律先 `rm -f` 日志、用 `ps -p <PID>` 实证存活，全程未用 `pgrep`（并且确实被 noclobber 咬了一次 `file exists`，按纪律停下改用 `rm -f` 重来） |
| NFS 纪律 | **一次也没碰**：既未读 `/mnt/nfs-ro`，也未写 `/mnt/nfs`，未起 `nfsx` |
| `trash/` | 未读未写 |
| 运行中的 build | **无**。`ps -eo pid,cmd` 只有两个 viewer 进程；L8 的 pid 已不在，`/mnt/ramstage/prod-l8-*` 全程只读 |
| TOML | 未改（本卡不加 config 字段，理由 §S-三.1 成立且我复核过 `_resume_config_differences` 确为逐路径精确比较） |
| 前七卡改动 | 一行未回退（工作树四文件 md5 首尾一致） |

---

## 18.8 L8 重启前的现场核对（我读到的真数字）

| 项 | 值 | 说明 |
|---|---|---|
| `output_root`（TOML:16） | `/mnt/ramstage/prod-l8-local400k-20260812` | **与标记所在目录逐字相同** ⇒ 标记放对了 |
| `STOP_FILL` | 0 字节，08-14 12:06 | 用**出厂谓词**实测：`_stop_fill_requested(...) → True` |
| groups / sft / annotation.pending / sam3.pending | 61,471 / 0 / 49,741 / 7,702 | 与 NOTES §S-一 逐项一致 |
| `failures.jsonl` | 46,258 行，无 `*_target_shortfall`、无 `stop_fill_requested` | task_id 形状变更零兼容问题 |
| `sft.jsonl` | **311 行**（manifest 11:51 写的 `completed.sft` 是 0，manifest 更旧） | 排干至今几乎没产出，见下 |
| 恢复守卫 | `run():4201-4213` 只比 `build_id` 与 `effective_config` | **没有"status=running 不许恢复"的守卫**，死进程留下的 `running` 不挡重启 ✅ |
| `restore_mirror` | 不触发（`manifest.json` 在位） | 不会去动 NFS 镜像 ✅ |

**两条给主 agent 的运维提醒（都不是本卡缺陷）**：

1. **排干很可能不是一次能跑完。** relay 侧现有 329 条 `untyped_stream_event` +
   78 条 `annotation_round_exhausted`，49,741 条待标注历史上只产出 311 行 sft。
   停渲**不豁免**标注完成门（`test_an_undrainable_backlog_is_still_an_error` 钉住了），
   所以排干收不了尾时进程会以 `PipelineError: annotation queue remains unresolved` 结束 ——
   **这是设计内的响亮失败，直接重启即可**（§18.1(e) 的三次重启实测），
   但如果 relay 本身坏着，重启多少次都只是重复失败，**建议重启前先确认 relay 健康**。
2. **重启后第一个动作是 `_verify_group_assets` 走 61,471 组**（NOTES 已提），
   期间没有输出；再叠上 §17.8 说的边界静默，头几分钟的安静是正常的。

---

## 18.9 nit 清单（6 条，均不阻塞）

* **N-S1（判重键过宽）**：`_journal_once` 只比 `task_id`，且扫的是**全部** failure 行。
  我构造了一条"同 task_id、不同 error_code"的行插在前面，shortfall 就**被静默吞掉**了
  （实测 `rows under the planted task_id: ['reviewer_probe']`）。
  生产不可达 —— `stable_id` 把 kind 同时写进 payload 与 id 前缀，全仓 `task_id=` 的
  14 处构造里 `target_*` / `stop-fill_*` 各只有一个生产者（`:3815` / `:3604`）。
  建议（重启后）：把 `event_type` 或 `error_code` 一起进判重条件，代价一行。
* **N-S2（`phase` 不在身份里）**：`(at_start, local, global)` 相同、`phase` 不同的第二次 stop
  **不落行**（我构造出来了：第二次在 `sam3_relabel` 相观察到，journal 仍只有第一次那条
  `"phase": "rendering"`）。丢的是非 terminal 的信息行，terminal shortfall 与 manifest 都不受影响。
* **N-S3（stop 事件的 message 是过了 `redact_text` 的 JSON）**：不保证可解析。
  测试 fixture 的 secrets 恰好是 `"u"` / `"p"`，于是实测得到
  `{"at_start": tr<redacted>e, …, "<redacted>hase": "rendering"}` —— 我的第一版夹具就是被这个 `json.loads` 打断的。
  生产 secrets 是长随机串，marker 路径与 phase 名不会被切，**当前无消费方解析它**（出厂测试用子串断言）。
  若日后要机读这条 message，需先把它挪出 `redact_text` 或改成结构化字段。
* **N-S4（无锁读共享 list）**：`_journal_once` 不持 store 锁就遍历 `self.store.failures`，
  而标注 driver 线程可能同时 `append`。CPython 下安全（list append 原子、该 list 全仓只 append
  不改序，`state.py` 6 处引用我逐个看过），但值得一行注释说明依赖的是这个不变式。
* **N-S5（窗口覆盖）**：出厂 40 例的在途窗口都是 2，生产是 5。我在 window=5 下复测通过
  （§18.3），建议把那一例并进套件，免得以后有人改窗口逻辑时失去覆盖。
* **N-S6（停渲仍依赖 Postgres / preset bank）**："CPU 任务"不等于"零外部依赖"：
  `catalog_loader` + `inventory_loader` 在 `stop_fill` 分支之外照跑（理由正当，`local_target`
  是 shortfall 的分母）。入队时要保证这两者可达。
  附带：`manifest.stop_fill.requested` 读的是闩，标记若在最后一次 `_stop_fill()` 调用之后
  （即进入 annotation 相之后）才落下，manifest 会写 `requested: false` —— 对那次 run 是诚实的，
  但运维若"先设标记再看 manifest"可能困惑，写进 runbook 即可。

---

# **PASS。0 blocker。PIPE-StopFill-2 可以立刻带上重启，按 CPU 任务入队。**

> 重启**之前**唯一值得先做的是**确认 relay 健康**（§18.8 提醒 1）——不是代码问题，
> 但它决定这次排干是一次跑完还是要重启若干次。6 条 nit 全部可以排到重启之后。
