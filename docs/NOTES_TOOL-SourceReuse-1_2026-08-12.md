> # ⚠⚠ 本文件不完整——四节丢失（2026-08-13 17:22 事故）
>
> 原件从未进入 git 索引，2026-08-13 17:22 随整个 `docs/` 树被并发进程删除。
> 以下内容由悬空 blob `30108ce82b3e9af8c660c494b4e2bdb32a05a3ef`（53,962 B / 804 行）恢复，
> **只到 `HOTFIX-Sam3Manifest-1` 为止**。
>
> **丢失且无法恢复的四节**：
> `HOTFIX-Sam3Drain-2`（§D-一~七）、`HOTFIX-MirrorCadence-3`（§M-一~八）、
> `HOTFIX-Watermark-4`（§W-一~八）、`HOTFIX-MirrorScrub-5`（§S-一~七）。
>
> 这四节的**结论与关键数字**在 `docs/reviews/REVIEW-impl-source-reuse.md` 的
> §14 / §15 / §16 里有大量引用与复核，可作二手依据；但设计论证原文已不存在。
> 若需重建，只能由编码 agent 重写。
>
> 详见 `docs/reviews/REVIEW-impl-source-reuse.md` 顶部的事故说明。

---

# NOTES — TOOL-SourceReuse-1：canonical databuild 源图重复采样开关

日期：2026-08-12 ｜ 实施者：编码 subagent ｜ 交付：`[sources] max_source_uses`（默认 1 = 现行为）

**未启动任何 build，未改任何现有 TOML，未碰 `/mnt/ramstage`、NFS、`trash/`。**

---

## 一、实施前核实记录（全部本机读码核实，无外部检索）

任务卡给的 6 条代码事实逐条复核，5 条属实、1 条需补充：

| # | 任务卡陈述 | 复核结果 |
|---|---|---|
| 1 | `sources.py:117-126` `_require_unique_source_ids` 同 id 出现两次即抛错 | 属实（现为 `:117-126`，改后行号不变） |
| 2 | `sources.py:458-492` 前 `target_groups` 个为初始配额，其余替补；`len(ordered) < target_groups` 时替补恒 0 | 属实。**补充一条任务卡没说、但决定了本次设计的事实**：池不足时 `prefix_labels` 是「先把 local 配额填满，再给 global」，`min(targets["local"], target_prefix)`（`:473`）。L7 的 `mix` 只要 `targets["local"] ≥ 28189`，**全部源都会被标成 local，global 一个都拿不到**。开关打开后必须改这个切分，否则重复采样只会把一个 mode 撑满。 |
| 3 | `agent.py` `_fill_initial_mode` 线性遍历 / prefetch 暖替补 / `replacement_used` / `replacement_capacity` | 属实 |
| 4 | resume 逐 key 精确比较 + `_RESUME_NEUTRAL_DEFAULTS` 机制 | 属实，已按该机制加名单项 |
| 5 | `config.py` sanitized_dict 走 `dataclasses.asdict` | 属实，新字段自动进 `effective_config`，无需改 `sanitized_dict` |
| 6 | group 身份派生需逐个排查 | 见下 §三 |

额外核实（本次设计的关键依据）：

- **`group_id = stable_id("group", build_id, source_id, mode, group_attempt)`**（`agent.py:1896`）。
  `group_attempt` 是同一次渲染内换 major 的重试计数，**不含"第几次使用该源"**，
  所以同源第二组会与第一组撞 `group_id` → `append_group` 抛
  `conflicting durable group record`。这是本改动**必须**动的唯一硬碰撞点。
  `candidate_id = stable_id("candidate", group_id, slot_id)`（`:1596`）、
  `after_path = candidates/<candidate_id>.jpg`（`:1607`）都从 `group_id` 派生，
  修好 `group_id` 即全部跟着修好。
- **preset 采样序列不依赖 `source_id`**（任务卡担心的"同源两组渲出完全相同的 8 候选"，
  **不会发生**，但理由和强度都要说准 —— 见下面的修正）：
  `CoverageSelector.begin_group` 选 major 用的是**全局覆盖计数器**
  `_major_success + _active_major` 的最小值（`presets.py:394-403`）；
  `_choose_minor`（`:600-608`）与 `_candidate_bag`（`:610-632`）的
  `_seeded_order` 种子是 `namespace + seed + major/minor + minimum`，
  **没有 `source_id`**。所以同源两组拿到什么由「前面几趟已经消耗掉哪些」决定，
  不需要改种子，只需要不让它们撞 id。
  唯一带 `source_id` 的是 `reservation_id`（`:414`），那是去重键，已加 `use_index`。

  > **措辞修正（2026-08-12，依据 REVIEW-impl-source-reuse B2）。**
  > 本节初稿写的是"同源两组的 preset **交集为空**、major 也不同，天然不同"。
  > **这句话是错的**，它只在交付自带的 2 major × 16 preset 玩具目录里成立
  > （一组吃掉一个 major 的 16 个里的 8 个，第二组不换 major 就凑不齐）。
  >
  > **选择器唯一真正保证的是"一个 group 内部的 8 个候选互异"**
  > （`GroupReservation.commit` 的 `len(set(preset_ids)) != 8` 检查，`presets.py:563-564`）。
  > **跨 group 没有任何守卫**：`_candidate_bag` 只是优先取"用得最少"的 preset，
  > 当一个 major 里所有 preset 的使用次数被拉平之后，它就会把该源已经渲过的 preset
  > 再发一次。成因是覆盖计数器是**全局**的（按 preset 计，不按 source 计），
  > 它没有、也不打算有"这个源已经用过哪些"的记忆。
  >
  > **真实 bank 上的实测**（口径与命令见 §七）：3522 preset / 10 major / 85 minor，
  > 按 L8 形状（26,460 源 × 12 趟 = 317,520 组、2,540,160 次抽取）：
  >
  > | 量 | 实测 |
  > |---|---|
  > | 整组 8 个 preset **完全重复**（任务卡字面要求） | **0** |
  > | 重复 `(source, preset)` 抽取 | **133,318 / 2,540,160 = 5.25%** |
  > | 同源两组的**最大** preset 交集 | **7 / 8** |
  > | 同源两组的平均交集 | 0.084 |
  > | 有任意交集的组对 | 87,079 / 1,746,360 = 5.0% |
  >
  > 另外 **`max_source_uses = 12 > 10 个 major`，鸽巢原理直接判死"同源各组 major 互不相同"**，
  > 这条不需要实验。
  >
  > 后果：global 模式下一次重复 `(source, preset)` = 同一张 `I_in` + 同一条 LUT
  > = **逐位相同的 `I_tar`**，只是换了 `candidate_id`/`sft_id`；
  > `sft_pack.py:69-100` 与 `q3vl/data/scan.py:39-76` 都**按 ID 去重、抓不到内容重复**。
  > local 模式还要同槽位才完全重合，概率更低但非零。
  >
  > 已做三件事让它不再是"假设"：
  > (a) manifest `sources.source_reuse` 每次落盘都带
  > `duplicate_source_preset_pairs` / `duplicate_source_preset_rate` / `max_pair_overlap`；
  > (b) 单测改名为 `test_the_repeat_pass_does_not_redraw_the_same_preset_set`，
  > 主断言降为任务卡真正要求的**"两组不是同一个集合"**，
  > 交集为空与 major 不同的断言保留但显式标注 `regime="2x16 toy catalog"`；
  > (c) 新增 `test_the_redundancy_columns_count_a_real_overlap`：
  > 用一个只有 8 个 preset 的退化目录，断言两列真的会动到 16 / 0.5 / 8，
  > 否则这两列等于什么都没测。
- **mask 种子只依赖 `source_id`**（`canonical_masks.py:245/256`、`pair_mask_slots` `:288`），
  所以同源两组的 7 张物理 mask 与 `mask_id` 完全相同 —— 这是**有意保留**的，理由见 §二.4。
- **SAM3 / terminal 相关 id 是 per-source 而非 per-group**
  （`stable_id("sam3", build_id, source_id)` `:1466/:2314`，`_terminal_source_ids` `:817`），
  保持用裸 `source_id`：主体掩膜是源的属性，一个源的掩膜救不回来，它的所有 use 都得死。

---

## 二、设计决策（逐条给理由）

### 1. 开关形态：`[sources] max_source_uses`（int，默认 1；0 = 不限）

选 int 不选 bool：任务卡给的 L8 目标 400,000 / 池约 37,800 ≈ 每源 10.6 次，
**"能重复"和"能重复多少次"是两个不同的决定**，bool 只能表达前者，而
`0`（不限）与 `11`（限一位数量级）在源被过度复用时的失败模式完全不同。
int 还让 `allocate_sources` 能算出「每个 mode 需要几个源」（`ceil(target/uses)`），
bool 做不到。上界 `MAX_SOURCE_USES_LIMIT = 1000` 纯粹是打错字的护栏。

放 `[sources]` 而不是 `[render]`：它描述的是**源池怎么用**，与
`source_window` / `gpu_concurrency` 这类调度宽度无关；且 `allocate_sources`
（sources.py）是第一个消费它的地方。

### 2. 轮转方式：外层多趟（pass），而不是在分配表里重复条目

任务卡要求"源池整体轮转（round-robin 全池再第二轮），不是同一源连续多组；
每轮内保持既有场景分层顺序"。两种实现：

- (A) 让 `allocation.local` 变成含重复条目的元组；
- (B) 分配表**完全不变**（每源仍只出现一次），改由 `_fill_mode` 对同一张表走多趟。

**选 (B)**。理由：
- (A) 会让 `_require_unique_source_ids`（红线要求保留）与分配表自相矛盾，
  还会波及 `by_id = {s.source_id: s for s in allocation.local}`（`agent.py:2428`）、
  `local_initial`（`:872`）等所有"分配表元素唯一"的隐含假设；
- (B) 天然满足"全池轮转 + 每轮保持分层顺序"（就是同一张有序表再走一遍）；
- (B) 让 `max_source_uses = 1` 退化成**一次** `_fill_initial_mode` 调用，
  与改动前逐字相同 —— 这正是红线要的"开关不开 = 现行为"。

### 3. `use_index` 的取得方式：主线程从 journal 现算，显式传参

`use_index = 本源已有的 durable group 数`。跳过判据从
`source_id in completed_sources()` 换成 `completed_source_uses()[sid] > use_index`
—— 在 `use_index = 0` 时这**就是**原来的成员判定（计数 >0 等价于在集合里）。

为什么不在 worker 线程里现算：`store.groups` 是普通 dict，
`_commit_source_result` 在主线程写它，worker 线程遍历会撞
`dictionary changed size during iteration`。现有代码所有
`completed_sources()` 调用点都在主线程，本改动保持这条不变，
所以 `use_index` 由 `_fill_initial_mode`（主线程）算好后显式传进
`_render_source_buffered`。

**为什么这个计数在 pass 内是精确的**：`_fill_initial_mode` 结尾
`while inflight: finish_oldest()` 会排空整趟的在飞源，所以第 k 趟开始时
第 0..k-1 趟的所有 group 都已落 journal。同一个源**不可能**同时有两个 use 在飞
—— 这顺带消掉了三类并发问题（`_write_cgt_once` 抢同一个 `.tmp`、
`duplicate active reservation`、prefetch 重复计账），因此没有额外加锁。

**修正（2026-08-12，依据 REVIEW-impl-source-reuse B1）**：
初稿把 `use_index` 叫作 "resume-stable cursor" —— 游标本身确实幂等可重建，
**但初稿没有任何地方用它决定从第几趟开始走**：`_fill_mode` 的 `use_index`
是函数局部、每次调用从 0 起。见 §二.3bis。

### 3bis. 收工护栏必须区分「游标推迟」与「池子拒绝」（B1 修复）

**初稿的病**（审阅 B1，两个复现脚本我都跑过、都复现）：`_fill_mode` 每次调用
`use_index = 0`，而"这一趟没产出就 return"的护栏分不清两种零产出：

| 零产出的原因 | 该怎么办 | 初稿怎么办 |
|---|---|---|
| 池子榨干（全 terminal / SAM3 卡住 / preset 用尽） | 收工 | 收工 ✅ |
| 这一趟的 group **早就存在**（重启、或 `_fill_mode` 被二次调用） | 走下一趟 | **收工 ❌** |

两条真实路径都会踩：
1. **中途重启**：崩溃后每个源都已有 group → 第 0 趟零产出 → 直接收工，
   `complete_with_failures` + shortfall，**不报错**。L8 口径 = 一次崩溃静默作废剩余全部趟数。
2. **SAM3 救回的源**：`_drain_sam3_and_replacements` 二次调用 `_fill_mode`，
   同样从 0 起 → 被救回的源只拿到 1 次使用。**不需要崩溃，一次干净的 L8 正跑就会发生。**

**修法（两个机制，各司其职，各有一条单测钉死）**：

- **正确性 = 游标感知的护栏**。`_fill_initial_mode` 现在返回
  **本趟因游标被跳过的源数**（`cursor_skips`）；terminal / SAM3-pending 的跳过
  **不计入**（那是拒绝，不是推迟）。护栏改成
  `零产出 且 cursor_skips == 0` 才收工。
  这个机制单独就足以保证正确，只是慢。
- **性能 = 从游标最小值起步**。`use_index` 初值取
  `min(completed_source_uses()[s] for s in pool)`，**再 clamp 到 `budget - 1`**。
  安全性是构造性的：任何低于该最小值的趟，**每一个**源都会被游标跳过，走了也是空走。
  它**只可能保守**（一个 count=0 的 terminal 源就会把起点拉回 0），不可能跳过该走的趟。

**为什么两个都留**：它们各自能单独修好那两个复现脚本，但**只有护栏在一般情况下正确**——
池子不均匀时（例如源 A 已有 3 组、源 B 永久 terminal 停在 0 组），
最小值被 B 拉回 0，第 0 趟零产出，此时只有护栏能让 A 继续走完它的预算。
反过来只留护栏则每次重启都要把已走完的趟重新空扫一遍。
两条单测分别钉死两个机制（见 §五「变异测试」的后两行：各自回退只挂各自那一条）。

> **那个 clamp 是自查出来的、必须有的**（不在审阅清单里，是修 B1 时新引入又自己发现的）：
> `completed_source_uses()` **把 lost group 也计入**（刻意的，group_id 已被占用），
> 而 `_mode_groups()` 过滤掉 lost。所以"资产全丢一组"会造成
> **游标已到 budget、但 live 数还不够 target** 的局面。此时无 clamp 的最小值起步会
> 打开一个预算不拥有的趟：
> - `max_source_uses = 12` → 某个源拿到第 13 组（`budget_exceeded` 会叫，但已经渲了）；
> - **`max_source_uses = 1` → 源拿到第 2 组，等于开关自己把自己打开了**，
>   直接违反"默认行为一字不变"这条红线。
>
> 为什么原来的 target 检查挡不住：`_fill_initial_mode` 第一步就 `break` 的前提是
> `completed >= target`，而 lost 恰好让 `completed < target`。
> 两条单测钉死：`test_a_lost_group_never_buys_a_source_an_extra_use`（默认档）与
> `test_a_lost_group_does_not_push_a_reuse_build_over_budget`（开档）——
> 把 clamp 那三行删掉，两条同时挂。

### 4. **不**改 mask 种子：同源多组共用同一套 mask / C_GT

考虑过给 mask 也加 `use_index`（同源不同 mask，多样性更高），**否决**。理由：
- `agent.py:186-199` `_write_cgt_once` 的注释是硬约束：C_GT 按 `mask_id` 复用，
  "任何让已渲染的源被重新 plan 的改动，必须先失效（删除）受影响的 C_GT 文件"。
  保持种子不变 → 重新 plan 出来的**像素逐位相同** → 该约束自动满足；
- `_sample_geometry` 有随机重试（`GEOMETRY_ATTEMPTS`），换种子意味着
  **第 1 趟成功的源可能在第 2 趟几何采样失败**，进而给一个"已经有 group 的源"
  排 SAM3 relabel。而 `_pending_sam3_ids()`（`:822`）会把有 group 的源过滤掉，
  于是这个 relabel 请求**静默消失**、`_drain_sam3_and_replacements` 判定无进展直接 break。
  不是死循环，但是一条静默的产能损失路径 —— 不值得为多样性换；
- 多样性由 preset 提供（**注意 §一 的措辞修正**：是"整组不重复"，
  不是"交集为空"；真实 bank 上 5.25% 的抽取会重复），
  用户裁定也是"source img × LUT 保证多样性"。
  给 mask 加 use_index 确实能补回一些多样性，但代价是上面两条静默失败路径，
  且 `_write_cgt_once` 的契约注释已按新事实重写（nit N3）明确写死
  "任何 per-pass 熵都会同时打破两条安全性论证"。

已加单测 `test_two_local_groups_of_one_source_land_their_shared_cgt`：
同源两组共用 7 张物理 mask，两组一起 land、共享 inode 被 hardlink 两次、
staging 原件被 unlink 两次，最终 `verify_dataset` 通过、32 个 candidate 全部可寻址。

### 5. `allocate_sources` 的池切分：只在开关打开时换分支

`max_source_uses == 1` 走**原封不动的老代码**（逐字保留，只是缩进进了 `if`）。
打开时走 `_reuse_prefix_labels`：
- `need[mode] = ceil(targets[mode] / uses)`（`uses == 0` 时用 `target_groups` 当上界）；
- `sum(need) <= 池` → 前缀就取 `need`，余下仍按 mix 当替补（沿用老语义）；
- 池连 `need` 都不够 → 按 mix 权重切整池，再用 `need` 封顶
  （防止 `mix.global = 0` 的 mode 白占源），多出来的还给能用的 mode。

这条修的正是 §一.2 里那个"全部变 local"的病。已加成对单测：
`test_the_default_still_starves_the_second_mode_on_a_scarce_pool`（关 = 老病照旧）
vs `test_reuse_splits_a_scarce_pool_so_both_modes_can_reach_their_target`。

### 6. prefetch 的 done 判据必须跟着改（否则是**静默**的吞吐塌方）

`_chunk_paths` 原来用 `completed_sources()` 过滤。开关打开后第 2 趟起
**每个源都已经有 group**，这个集合会把整池判成"已完成"，
`prefetch` 收到空列表 —— 渲染照跑，只是每张图退化成一次随机 archive pread，
**没有任何断言会失败**。已换成同一套 `> use_index` 判据，
并加了 `test_the_second_pass_still_renders_out_of_the_prefetch_buffer`
（在每次 `preprocess_source` 时快照 buffer 目录，断言该源的副本在里面）。
把这行改回旧写法，该测试立刻以 `not found in []` 失败 —— 已实测。

### 7. `replacement_capacity` / `replacement_used` 的语义

**不改公式**（改了会破坏 L6/L7 已落盘 manifest 的可比性），改为**显式标注**：
manifest 的 `sources` 节新增 `source_reuse`，并在两个老 key 旁写清
"开关打开后备用产能主要住在额外趟数里，池小于目标时这两个 key 会同时读 0
但目标仍然达成"。`source_reuse` 全部字段（都从 `store.groups` 一次遍历出，
与跳过判据同源，所以它是游标状态本身而不是第二意见）：

| 字段 | 含义 |
|---|---|
| `max_source_uses` | 生效的开关值 |
| `distinct_sources` / `groups` / `max_observed` | 用到的源数 / 组数 / 单源最高使用次数 |
| `budget_exceeded` | 游标不变式的自检位（nit N4）：`max_observed > budget` 即为真。当前不可能为真，写出来是为了让不变式**自己会叫**而不是只活在 docstring 里 |
| `uses_histogram` | 每源使用次数分布（开关关时恒为 `{"1": N}`） |
| `duplicate_source_preset_pairs` | 同源重复抽到的 preset 次数（B2） |
| `duplicate_source_preset_rate` | 上一项 / 总抽取数 |
| `max_pair_overlap` | 同源任意两组的最大 preset 交集，满分 8（B2） |

后三项是 B2 要求的"让 L8 决策者看得到代价"。它们**按每次 manifest 落盘重算**，
实测（本机、合成 317,520 组 / 26,460 源 / 12 趟的 L8 形状）**2.2–2.4 s**
（有交集与无交集两种数据分布分别测过；`max_pair_overlap` 命中 8 时二次项提前退出）。
L8 约 100 次 land checkpoint ⇒ 全程约 4 分钟，与 `_manifest` 里既有的
全量 candidate 遍历同量级。

### 8. group 记录新增 `source_use_index`，且**仅在非 0 时写**

第一趟的 journal 行形状与改动前逐字相同，读者可以把"缺失"当 0。
不这么做的话每一行都多一个 key，"默认行为一字不变"就不成立。

### 9. resume 豁免名单

`("sources", "max_source_uses"): 1` 已入 `_RESUME_NEUTRAL_DEFAULTS`。
论证：默认值 1 时 `allocate_sources` 走的是老分支、池只走一趟、所有派生 id
都不带 use 后缀 —— 老 manifest 描述的运行与这个默认产生的运行是同一个。
名单机制只豁免"老 manifest 缺该 key"，**不豁免**"老 manifest 写着 1、新 config 写 3"：
后者会照常报 `resume config differs`。两条都有单测
（`test_a_manifest_written_before_the_key_resumes_a_default_build` /
`test_an_aged_manifest_does_not_excuse_switching_reuse_on_mid_build`）。

---

## 三、影响面清单（逐文件：有影响 / 无影响 + 证据行号）

行号为改动后工作树。

### A. `dataset_build/src/construct/` 内（本次改动范围）

| 文件 | 影响 | 证据 / 处理 |
|---|---|---|
| `sources.py:117-126` `_require_unique_source_ids` | **无影响，且刻意保留** | 重复来自"多趟"，不是重复的 inventory 行；三种 `max_source_uses` 下都有单测断言它仍抛错 |
| `sources.py:498-527` `allocate_sources` | **有影响** | 开关打开时换分支（§二.5）；关闭时老代码逐字保留 |
| `agent.py:1896` `group_id` | **有影响（唯一硬碰撞点）** | 非首趟追加 `use_index`；首趟 id 逐位不变，有单测比对 `stable_id(...)` |
| `agent.py:1596` `candidate_id` / `:1607` `after_path` | **无需改动** | 从 `group_id` 派生，跟着唯一。单测断言 4 组共 32 个 candidate_id 互不相同 |
| `presets.py:414` `reservation_id` | **有影响** | 同源同 attempt 同 major 会撞；已加 `use_index`。注意 `_reservations` 在 commit/abandon 时会 pop，所以撞的是**同时在飞**的两个 —— §二.3 的排空不变式已排除，加后缀是纵深防御 |
| `presets.py:394-403 / 600-632` preset 抽取 | **无影响，且是多样性的来源** | 种子不含 `source_id`，由覆盖计数器驱动（§一） |
| `canonical_masks.py:176/245/256/288` mask 种子与 `mask_id` | **有意保持不变** | 同源多组共用 7 张物理 mask（§二.4） |
| `agent.py:1523` `_cgt_path` / `:1526` `_start_cgt_writes` / `:186` `_write_cgt_once` | **行为可接受，已测** | 第 2 趟若文件还在则复用（像素相同），若已 land 被 unlink 则重新编码。`save_cgt_png`（`rendering.py`）是 tmp + `os.replace`，且同源两 use 不并发（§二.3） |
| `agent.py:2179-2196` `_chunk_paths` | **有影响（静默）** | §二.6 |
| `agent.py:2318` `_fill_initial_mode` 跳过判据 | **有影响** | 换成 `> use_index`，`use_index=0` 时等价 |
| `agent.py:2112` `_render_source_buffered` 的 `render-source` task_id | **有影响** | 同源多趟会撞 task_id；非首趟追加 `use_index` |
| `agent.py:1466/2314` `stable_id("sam3", ...)`、`_terminal_source_ids`、`_pending_sam3_ids` | **有意保持裸 source_id** | 主体掩膜是源的属性（§一） |
| `agent.py:872-878` `replacement_used` / `:937-940` `replacement_capacity` | **数字仍自洽，语义变窄** | §二.7，已加 `source_reuse` 与旁注 |
| `state.py:290-307` `append_group` | **无影响** | 主键是 `group_id`，不是 `source_id`；`:263-273` `_load_unique` 同理 |
| `state.py:381-395` `pending_annotation_tasks` | **无影响** | `task_id = stable_id("annotation", group_id, candidate_id, rank)`，全部从 `group_id` 派生 |
| `projection.py:22-38` `canonical_groups` | **无影响** | `PRIMARY KEY(group_id)`；`source_id` 上只有普通 INDEX（`:37-38`），不是 UNIQUE |
| `projection.py:39-59` `canonical_candidates` | **无影响** | `UNIQUE(group_id, slot_index)`，按 group 而非 source |
| `legacy_import.py:183-184` | **无影响** | 走 LegacyImportPipeline，不经过 `_fill_mode` |

### B. `dataset_build/src/construct/` 外（独立 subagent 扫描 + 抽查复核）

**BREAKS：0 处。** 没有任何 construct 外的消费者会在同一 build 内
`source_id` 重复时报错、丢数据或静默去重。

**BEHAVIOR CHANGE（1 处，viewer 展示层）**

- `databuild_viewer/backend/repository.py:192-216`（`_related_failures`，JSONL 路径，
  匹配条件在 `:211-212`）、`:739-746`（Postgres `_where` 的 `related_failure` 片段
  `ff.group_id IS NULL AND ff.candidate_id IS NULL AND ff.source_id=g.source_id`，
  在 `:794-811` 用于 `failure_state` 过滤/facet）、`:910-916`（group 详情页同样的回退 join）：
  **孤儿 failure 事件**（无 `group_id`、无 `candidate_id`，例如源级 SAM3 失败）
  是靠 `source_id` 挂到 group 上的。同源多组时，一条源级失败会同时显示在
  **该源的每一个 group** 下，而不只是它真正对应的那个。
  不崩、不丢数据，纯展示归因偏移。`databuild_viewer/backend/test_app.py:116` 只断言
  SQL 文本存在，不覆盖这个场景，CI 抓不到。→ 列入 §四 待决策。

**NO IMPACT（要点，均有行号证据）**

- `dataset_build/tools/sft_pack.py:69-100` 按 `sft_id` 去重；`:118-159` 的
  `index_by_source_path` 按**候选自己的**输出路径建索引，且 `:280` 显式排除 `role == "in"`。
- `dataset_build/tools/global_catalog.py:69-75` `source_paths` 是
  `PRIMARY KEY(source_path)` + `INSERT OR REPLACE`（`:157-165`）：同一原始路径
  被 land 两次时后写覆盖先写。**这不是新问题** —— `:254-273` 的 docstring 与
  `tools/land.py:191-195` 都写明这是设计内的（一个 group 的 1-2 个 winner 已经共享
  同一份 `I_in`），且两份字节完全相同，`read_bytes` 解析到哪一份都正确。
  同源多组只是把这个已被接受的现象放大。
- `dataset_build/tools/archive_reader.py:155-167` `locate()` 单行 `fetchone()`：同上。
- `dataset_build/tools/prefetch.py:175` `dict.fromkeys` 已对请求列表去重。
- `dataset_build/tools/dataset_plan.py:740-757` 按精确渲染路径建表，不按 source。
- `dataset_build/tools/export_annot_review.py:37,51-60`、`review_annot_quality.py`、
  `reeval_*.py`（`reeval_iaa_pairs.py:172-196`、`reeval_direction_list.py:267-292` 等）
  一律按 `group_id` / `sft_id` 索引。
- `dataset_build/tools/shard_dataset.py`、`metric_bank.py`、`prod_watchdog.py:133-134`、
  `indexed_tar.py:586-588`（唯一性是 per-dataset `member`/`logical_path`）：无 source 键逻辑。
- `dataset_build/source_qa/db.py:16-33` `assets` 表 `PRIMARY KEY(asset_id)`：
  这是**建 group 之前**的原始素材 ingest 状态，与一个源后来产出几个 group 无关。
- **S/P split 纪律**（实际实现在 `tools/data_splits/`，不在 `source_qa/`）：
  `tools/data_splits/build_splits.py:69-96` + `vr_common.py:47-62` 的 S-split 是
  `source_id` 的纯函数，同源多组重新推导出**同一个** pool（`:84-85` 的
  `elif prev != pool` 永不触发），不会产生 `pool_conflicts`；
  `selfcheck.py:69-98` 的 `src_builds: dict[str, set[str]]` 本来就容忍 `source_id` 复现。
  **同源不会跨 split 泄漏。**
- `q3vl/data/splits.py:71-93` `source_groups` 在切分**之前**就按
  `source_id`（及 `i_in_path`）把行并成一个原子组（docstring `:74-77`），
  多组同源正是它设计要处理的情况；`:294-348` 的 `audit` 是该划分的子集，同样不会被拆开。
- `q3vl/data/pipeline.py:181-480`、`q3vl/data/scan.py:38-112`（去重键是 `sample_id`/`sft_id`）、
  `q3vl/whereb/attnprobe.py:63-94`（`fit_oof_split` 本就按 `source_image_id` 分组）、
  `q3vl/whereb/scripts/run_d0_7_replay.py:561-579`（已显式处理同源多样本）：均无影响。
- `tools/construct/splits.py:41-82` 只读冻结旁表，不扫 `groups.jsonl`。
- `tools/preset_viewer/app.py:183-219` 按 `(group_id, build_id)` 去重。
- `databuild_viewer` 前端 `Inspector.jsx:264,467,662` 的 React key 用
  `group_id`/`candidate_id`/`sft_id`，无重复 key 问题。

---

## 四、待主 agent 决策

> 以下三项两种做法都合理且影响后续，按纪律不静默拍板。均已采用保守默认继续。

### D1（**优先级最高，与本任务同源但独立**）`databuild.example.toml` 在 83da846 被误删，导致 **30 个既有单测在干净 HEAD 上就是红的**

- commit `83da846`「移除上一战役 VeraRetouch 0.5B 遗留文件（用户手动清理追认）」
  一次删掉 `LICENSE`、`utils.py`、`databuild.example.toml` 三个文件。
- 前两个确属上一战役遗留，但 **`databuild.example.toml` 是当前 canonical databuild
  的配置模板兼测试夹具**：它的 `[sources] subject_cache` 注释写的正是当前归档反查方案，
  且 `test_canonical_foundation.py` / `test_iaa_batch.py` /
  `test_source_window_and_cgt.py` / `test_winner_margin.py` 四个文件里
  **30 个 ConfigTests 用例直接 `shutil.copyfile(EXAMPLE, ...)`**。
- 实测：干净 worktree（HEAD=1018ef2）跑 `dataset_build/tests/` → **30 failed / 403 passed**，
  失败原因全是 `FileNotFoundError: .../databuild.example.toml`。
  把该文件从 `83da846^` 取回后 → **0 failed**。
- **保守默认**：本次**不**恢复该文件（超出任务卡范围，且它是用户手动删的仓库根文件），
  新增单测用自带的 `MINIMAL_TOML` 常量，不依赖它。
- **请裁定**：是否 `git checkout 83da846^ -- databuild.example.toml` 恢复？
  若恢复，还应在其 `[sources]` 段补上 `max_source_uses` 的注释与默认值
  （其余四个可选 key 都在该文件里有注释，本 key 现在没有落脚点）；
  同时 `test_source_reuse.py` 里的 `MINIMAL_TOML` 应改成读该文件。

### D2 渲染主循环的 O(groups × sources) 记账开销，L8 会放大约 100 倍（**先于本改动存在**）

- `_fill_initial_mode` 的每一次源迭代都要做若干次全量 group 扫描：
  `_mode_groups(mode)`（经 `_live_groups()`，2-3 次/迭代）+ 跳过判据 1 次。
- 本机实测 @200,000 groups：`_mode_groups` **43.2 ms**、
  `completed_sources()`（旧）**20.7 ms**、`completed_source_uses()`（新）**29.2 ms**。
  即本改动把每次迭代 ~110-150 ms 的既有开销增加约 8 ms（+6%），
  **主导项是我没动的 `_mode_groups`**。
- L8 量级估算：local 约 26,460 源 × 11 趟 ≈ 291k 次源迭代，
  平均 group 数约 140k → 纯记账 **约 8 小时**（L7 同口径约 5 分钟）。
  相对 400k 组 GPU 时间（周级）约 1.5%。
- **B1 修复对这笔账的影响（审阅 §9 要求量化）**：
  - **减**：跳过判据的 `completed_source_uses()` 已按审阅建议提到**每趟一次快照**
    （`_fill_initial_mode` 开头算一次，`_chunk_paths` 复用同一份）。
    精确性论证：`sources` 里每个 source_id 只出现一次
    （`_require_unique_source_ids`），所以一个源的计数只可能在它**自己那一步之后**
    变化，永远不会在读它的那一步之前变化。
    这让每次源迭代**比改动前还少一次 O(groups) 扫描**（旧代码是每次迭代重算
    `completed_sources()`，实测 20.7 ms @200k）。
  - **加**：轮转本身要多走 `(趟数-1) × 池` 次"空跳"迭代。L8 主调用
    ≈ 12 × 26,460 − 280,000 ≈ 3.7 万次多余迭代。
    「最小值起步」把重启后的空跳降到 0，但 `_drain_sam3_and_replacements`
    二次调用时若被救回的源计数很低，仍会从低趟起重走。
  - **残留主导项仍是我没动的 `_mode_groups`**（`completed = len(self._mode_groups(mode))`
    在每次源迭代都跑，实测 43.2 ms @200k）。
- **保守默认**：不动 `_mode_groups`。修它需要按 mode 的增量计数缓存
  （新状态 + lost-group 失效路径），属于未经审阅的优化，不该塞进"最小 diff"任务。
- **请裁定（审阅同意合并成一张卡）**：L8 开跑前是否单开一张卡做这个缓存？
  我的意见是**应该做**：B1 的修复让空跳迭代变成常规路径，
  而每次空跳仍要付一次全量 group 扫描。

### D3 viewer 的孤儿 failure 归因（§三.B 的唯一 BEHAVIOR CHANGE）

同源多组后，源级失败事件会挂在该源的每一个 group 下。
- **保守默认**：不动 viewer（本任务范围是 `construct/`；且它不影响任何落盘数据）。
- **请裁定**：是否给 viewer owner 单开一张卡，把 `source_id` 回退 join 收紧为
  「只挂到该源**最早**的 group」或「单列成 source 级事件」？

### D4（提示，不需裁定）`preset_inventory_exhausted` 在重复采样下是"一次失败、全部 use 作废"

`_render_source` 试遍所有 major 仍拿不到 8 个候选时会把源标 terminal
（`agent.py:2077-2081`），而 terminal 是 per-source 的。开关打开后，
第 7 趟的一次 exhausted 会让该源第 8 趟起也不再被使用。
现有语义如此，产能损失会被 `*_target_shortfall` 如实记录，
本次不改；若 L8 观察到该 error_code 计数偏高再说。
（补：真实 bank 的 dry run 里 **0 组被选择器拒绝**（§七），
所以 `preset_inventory_exhausted` 在 L8 只可能来自渲染/可见性失败，不来自 preset 不够。）

### D5（新，来自审阅 §5 风险提示）L8 数据上 Where 头的指令条件性判据必须重新标定

同一张 `I_in` 现在最多产出 12 个 group、共享**同一套 7 张 C_GT 区域池**，
即"同一张图、同一个区域池、12 条不同指令"。
- **不构成 split 泄漏**（S-split 是 `source_id` 纯函数，§三.B 已证）；
- **但训练集冗余度升到 12×/图**：CLAUDE.md 红线要求每个消融行必带
  `Δ_const/Δ_shuffle` 列，而在这个分布下"打乱指令仍能预测区域"的基线会被
  **结构性抬高**（图像→区域映射被重复了 12 次）。
- **建议写进 L8 预注册**：不得沿用 L7 数据上的 `Δ_shuffle` 门槛，须在新分布上重标。
- L_cube 监督从 `recipe.preset` 渲染、同源各组 preset 大部分不同 ⇒ 该路无耦合。

---

## 四bis、审阅 8 个 nit 的处置

| nit | 处置 |
|---|---|
| **N1** 默认路径多一次 `_mode_groups` 扫描 | **已改**。`_fill_mode` 先算 `final = budget and use_index+1 >= budget`，`final` 时 `before = None`、不做测量。`max_source_uses = 1` 现在是"一次 `_fill_initial_mode` 调用 + 零次额外扫描"，调用序列也与改动前一致 |
| **N2** `_try_ready_relabel` 每次 relabel 多一次 O(groups) 扫描 | **不改**，理由同审阅自己的定级：该值恒为 0（`_pending_sam3_ids()` 已 `.difference(completed_sources())`），而 relabel 循环里本来就有同量级的 `_pending_sam3_ids()` / `_mode_groups()`，属 D2 同一笔账。把它换成常量 0 会让"filter 若放宽就错"的隐患回来，注释已说明这点 |
| **N3** `_write_cgt_once` 契约注释落后于代码 | **已改**。注释现在列出**两条** re-plan 路径（SAM3 relabel / source reuse）及各自的安全性依据，并写死"给 mask plan 任何 per-pass 熵会同时打破两条论证" |
| **N4** `max_observed` 无硬校验 | **已改**。manifest `source_reuse.budget_exceeded` 布尔位，见 §二.7 表 |
| **N5** L8 REPORT 应出 `source_use_index × {scene, pool, mask_area 分位}` 表 | **采纳为 L8 实验设计建议**（非本卡代码）。`source_use_index` 已落在 group 行上，事后一次 group-by 即可 |
| **N6** 缺 B1 两条回归测试 | **已加**，见 §五 ④；另加两条分别钉死 B1 的两个机制 |
| **N7** 变异数字不可复现、"四处"只列三行 | **已改**，见 §五 变异表与其下的澄清框 |
| **N8** `MINIMAL_TOML` 与 example TOML 的关系 | **保持自带常量**（否则新测试会依赖一个当前不存在的文件）。测试文件顶部注释与 D1 都写明：example 一旦恢复，该常量应改为读该文件 |
| **N9** 「最小值起步」被永久 terminal 源钉死在 0（L7 有 898 个）| **已改**。min 现在剔除 `_terminal_source_ids()`。安全性：terminal 源在 walk 内部本来就被拒绝，剔除它不可能跳过一个该走的趟。单测 `test_a_terminal_source_does_not_drag_the_start_back_to_pass_zero`（前置断言钉死"预算 clamp 够不着"，所以只有 terminal 剔除能把起步推离 0；把该行改回去，这条立刻挂）|
| **N10** NOTES §一 行号 `presets.py:549-552` 指错 | **已改**为 `presets.py:563-564`（本机复核：`:544` 是 `accept`，`:554` 是 `commit`，`len(set(preset_ids)) != 8` 在 `:563`）|

---

## 五、单测与运行方式

新增 `dataset_build/tests/test_source_reuse.py`（**40 个用例 / 32 个 subtest**），
沿用仓库现有 `unittest.TestCase` + `OrchestrationFixture` / `LandFixture` 风格。

```bash
cd /home/bc/VeraRetouch
PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build:/home/bc/VeraRetouch/dataset_build/src \
  python -m pytest dataset_build/tests/ -q -p no:randomly
```

覆盖（对应任务卡三项要求）：

- **① 默认关 = 现行为**：`test_the_default_renders_each_source_once_and_writes_no_use_index`
  （3 源 3 组、无 `source_use_index` 键、`group_id` 逐位等于
  `stable_id("group", build_id, source_id, "global", 0)`）、
  `test_the_default_stops_at_the_pool_and_records_the_shortfall`、
  `test_the_default_still_starves_the_second_mode_on_a_scarce_pool`、
  `test_a_duplicated_inventory_row_is_still_rejected_with_reuse_on`
  （`_require_unique_source_ids` 在 uses ∈ {1,3,0} 下都仍抛错）。
- **② 开 = 同源多组 / preset 序列不同 / 轮转顺序正确**：
  `test_reuse_reaches_a_target_larger_than_the_pool`（4 组 / 2 源 / 32 个互异 candidate_id）、
  `test_the_repeat_pass_does_not_redraw_the_same_preset_set`
  （主断言 = **两组不是同一个集合**；交集为空与 major 不同的断言保留但标注
  `regime="2x16 toy catalog"`，docstring 里写明真实 bank 的 5.25% / 7-of-8）、
  `test_the_manifest_reports_preset_redundancy` 与
  `test_the_redundancy_columns_count_a_real_overlap`（B2 两列真的会动）、
  `test_the_pool_is_walked_round_robin_rather_than_source_by_source`（A,B,A,B ≠ A,A,B,B）、
  `test_the_first_pass_keeps_the_ids_a_build_without_reuse_would_write`、
  `test_the_budget_caps_the_walk`、`test_zero_walks_the_pool_until_the_target_is_met`。
- **③ resume 豁免**：`test_a_manifest_without_the_key_resumes_against_the_default`、
  `test_a_manifest_written_before_the_key_resumes_a_default_build`（真跑一遍、把
  manifest 里的 key 摘掉、再跑，断言 complete 且不重复渲染）、
  `test_an_aged_manifest_does_not_excuse_switching_reuse_on_mid_build`。
- **④ B1 的两条形状（审阅 nit N6，由审阅的两个复现脚本改写而来）**：
  `test_a_build_that_crashed_mid_reuse_finishes_its_remaining_passes`
  （在第 2 次 land checkpoint 注入崩溃 → 4 组 → 重启 → **6 组 / complete /
  shortfall 0 / `uses_histogram {"3": 2}` / use_index 序列 `[0,0,1,1,2,2]`**）、
  `test_a_source_rescued_by_the_sam3_drain_still_gets_its_budget`
  （border mask → sam3_queued → relabeler 修好 → **两个源各拿满 3 次**）。
  另加两条把 B1 的**两个机制分别**钉死（见变异表后两行）：
  `test_the_walk_resumes_past_a_source_that_can_never_render`（护栏）、
  `test_a_resumed_walk_does_not_re_walk_spent_passes`（最小值起步；
  spy 记录 `_fill_initial_mode` 收到的 `use_index`，断言 global 模式只被调用一次且是 `2`）、
  以及 `test_an_exhausted_pool_still_stops_instead_of_spinning` /
  `test_an_all_terminal_pool_retires_on_the_first_pass`（护栏没有变成"永不停"）。
- **额外**：`test_the_second_pass_still_renders_out_of_the_prefetch_buffer`（§二.6）、
  `test_two_local_groups_of_one_source_land_their_shared_cgt`（§二.4）、
  `test_a_completed_reuse_build_resumes_without_re_rendering`。

**变异测试（confirm the tests bite）** —— 逐处改回旧写法、实测，
`dataset_build/tests/test_source_reuse.py` 共 40 例：

| 改回 | 实测 |
|---|---|
| `_fill_mode` 的 `budget` 硬编码为 1 | **23 failed / 22 passed** |
| `group_id` 去掉 `use_index` 后缀 | **18 failed / 21 passed** |
| `_chunk_paths` 换回 `completed_sources()` 成员判定 | **2 failed**（`not found in []`，正是静默塌方的形状） |
| B1 只回退「最小值起步」 | **1 failed**：`test_a_resumed_walk_does_not_re_walk_spent_passes` |
| B1 只回退「游标感知护栏」 | **1 failed**：`test_the_walk_resumes_past_a_source_that_can_never_render` |
| B1 两个都回退 | **4 failed**（上面两条 + 崩溃重启 + SAM3 救回） |
| 删掉 `use_index` 的 `budget - 1` clamp | **2 failed**：两条 lost-group 用例 |
| min 里放回 terminal 源（N9 回退）| **1 failed**：`test_a_terminal_source_does_not_drag_the_start_back_to_pass_zero` |

> **nit N7 澄清**：初稿写的 11 / 8 / 2 是对**当时 29 例**测试集实测的，
> 审阅在同一份实现上重测得到 13 / 10 / 2 —— 差异来自审阅用影子包整树 cp 后
> 测试收集口径不同。上表是**当前 40 例**测试集在工作树上的实测值，
> 复现命令即 §五 开头那条 `pytest`（变异脚本见
> `/tmp/claude-1001/-home-bc-VeraRetouch/9e1ece54-*/scratchpad/`，改完即刻还原）。
> 另：初稿正文写"四处"但表里只有三行，已改为不写死数量。

**既有单测**：改后 `dataset_build/tests/` 的失败集合与干净 HEAD（worktree at HEAD）
的失败集合**逐条 diff 相同**（30 条，全部是 D1 的 `databuild.example.toml` 缺失）。
临时把该文件放回后：**443 passed, 0 failed**。

**唯一改到的既有测试文件**：`dataset_build/tests/test_source_window_and_cgt.py`，
两个 spy 从写死签名改成签名透明（`**kwargs` 透传）。两个 spy 分别只读
`pipeline._source_window` 和统计并发度，都不该钉死被测方法的关键字列表；
断言内容一字未动。

---

## 六、给 L8 的配置建议（不代改 TOML）

池约 37,800、目标 400,000 → 每源约 10.6 次。建议

```toml
[sources]
max_source_uses = 12   # 留一点余量；11 恰好卡满，任何 terminal 源都会造成 shortfall
```

不建议直接写 `0`（不限）：不限时若源池因 terminal / SAM3 大量减员，
剩下的少数源会被反复采样到很高次数而**没有任何上界告警**；
写死一个数则超出部分会如实落到 `*_target_shortfall`，
配合 manifest 的 `sources.source_reuse.uses_histogram` 一眼能看出分布是否倾斜。

**开跑前请先看 §七 的重复率**：`max_source_uses = 12` 的代价是
**5.25% 的候选是该源已经渲过的 preset**（global 模式下即逐位相同的 `I_tar`）。
若判定这个冗余不可接受，唯一的降低手段是调小 `max_source_uses`
（代价是达不到 400k），或扩 bank —— 抽取算法本身不在本次改动范围内。

---

## 七、真实 bank dry run（B2 ③）

只走 `PresetCatalog.load` + `allocate_sources` + `CoverageSelector`，
**不渲染、不打标、不写任何文件**；`draw_group()` 复刻 `_render_source` 对选择器的
全部调用（`begin_group` → 8 × `reserve_candidate`/`accept` → `commit`，
失败则换 major 重试，上限 `len(selector.majors)`）。

脚本已落库：**`dataset_build/tools/source_reuse_dryrun.py`**。

```
PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build:/home/bc/VeraRetouch/dataset_build/src \
  python -m dataset_build.tools.source_reuse_dryrun --sources 26460 --uses 12 --mode local
```

**bank 实测**（`/var/cache/veradata/preset_bank_full`）：
`presets=3522  majors=10  minors=85  formats={'lut': 3522}` ——
与审阅从 L7 manifest 读到的 `{"majors":10,"minors":85,"presets":3522}` 一致。

| 口径 | 200 源 × 12 趟 | **26,460 源 × 12 趟（L8 形状）** |
|---|---|---|
| groups | 2,400 | **317,520**（0 组被选择器拒绝） |
| `(source, preset)` 抽取 | 19,200 | **2,540,160** |
| `duplicate_source_preset_pairs` | 1,004 | **133,318** |
| **`duplicate_source_preset_rate`** | 5.23% | **5.25%** |
| **`max_pair_overlap`**（满分 8） | 5 | **7** |
| 整组 8 preset 完全重复 | 0 | **0** |
| 同源两组平均交集 | 0.084 | 0.084 |
| 有任意交集的组对 | 647 / 13,200 | 87,079 / 1,746,360（5.0%） |

耗时：catalog 载入 5.2 s（冷 cache 时 116 s），抽取 271 s。

**读法**：
1. 任务卡字面要求（"同源不同 group 必须渲出不同 preset 集合"）**成立**：0 次整组重复；
2. NOTES 初稿的"交集为空"**不成立**，最坏两组共享 7/8；
3. 5.25% 的候选是重复内容，**下游按 ID 去重抓不到**（§一 修正框内已列行号）；
4. 重复率对源池规模**不敏感**（200 源与 26,460 源都是 5.2%），
   说明它由 `每源趟数 × 8` 与 `每 major 的 preset 数` 的比值决定，不由池大小决定 ——
   所以想降只能动 `max_source_uses` 或 bank。

**口径限制（审阅 §12.5 方法学备注，已采纳写入脚本 docstring）**：dry run 对每个 slot
直接 `accept`，不走真实 `_render_source` 里可见性门失败后的
`reservation.reject` 重抽（`agent.py:1798`）。真实 build 每组会消耗更多 preset，
对重复率的影响方向不显然。**5.25% 应读作"同一抽取算法下的量级"，不是精确预测。**

---

## 八、分批 commit 提示（审阅 §10）

工作树里有**与本任务无关**的未提交改动，属另一条任务线（SAM3 / mask backfill 方向），
**不要与本改动同批 commit**：

| 文件 | 行数 | mtime | 与本卡的关系 |
|---|---|---|---|
| `dataset_build/core/responses_vlm.py` | +65 | 2026-08-12 10:01 | 无（不含 `max_source_uses` / `use_index`） |
| `dataset_build/source_qa/sam3_subject_instances.py` | +46 | 2026-08-12 10:10 | 无 |
| `dataset_build/tools/eval_subject_instance_selector.py` | +42 | 2026-08-12 09:53 | 无 |

本卡的完整文件清单（`git add` 时只取这些）：

```
dataset_build/src/construct/agent.py
dataset_build/src/construct/config.py
dataset_build/src/construct/presets.py
dataset_build/src/construct/sources.py
dataset_build/src/construct/state.py
dataset_build/tools/source_reuse_dryrun.py        (new, B2 dry run 工具)
dataset_build/tests/test_source_reuse.py          (new)
dataset_build/tests/test_source_window_and_cgt.py (2 个 spy 改签名透明)
docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md       (new)
```

---

# HOTFIX-Sam3Manifest-1 —— `_manifest` 的平方级 sam3 记账（2026-08-13）

> 本节与上文 SourceReuse 任务卡**无关**，只是共用 `agent.py`；追加于此是为了让
> `agent.py` 上所有未提交改动的来历在同一处可查。**可与 SourceReuse 分批 commit**，
> 文件清单见 §HOTFIX-五。

## HOTFIX-一、现象与实锤

L8（`prod-l8-local400k-20260812`，pid 587114）产出从 2,800 组/时塌到 120 组/时。
py-spy 主线程栈落在 `_manifest` 内的 `_unconsumed_sam3_ready`（`agent.py:900`）。
**以下行号一律按改后文件**；诊断当时 `_manifest` 在 `:912`，改后为 `:960`。

机制：`_manifest` 对每个 `sam3_relabel_queued` 的 source 问一次
`_unconsumed_sam3_ready`，而该函数**每次全表扫两遍** `self.store.failures`：

```python
sam3_completed_ids = {
    source_id for source_id in sam3_expected_ids
    if self._unconsumed_sam3_ready(source_id)   # 内部 2 × O(len(failures))
}
```

本机实测 L8 落盘现状：`failures.jsonl` **17,656 行**，其中
`sam3_relabel_queued` 的 distinct source **5,003 个**（任务卡写 4,841，期间又涨了）。
乘积 ≈ **1.77 亿次** dict 访问，**每次 land checkpoint 都重算一遍**。
`_manifest` 由 `_land_checkpoint`（`:1575`）调用，而 `_land_checkpoint` 在
`_fill_initial_mode.finish_oldest()`（`:2506`）里 inflight 排空时就触发 ——
主循环卡住期间渲染线程全部空等。journal 已记 **782 次 landed**。

## HOTFIX-二、改法（`agent.py`，+52 −1）

新增两个成员，`_unconsumed_sam3_ready` **本体一字未动**（别处单点调用仍走它，也正好
当等价性测试的 oracle）：

| 新成员 | 作用 |
|---|---|
| `_sam3_ready_maxima()` | **一遍**扫 `failures`，产出 `source_id -> (max sam3_ready attempt, max sam3_ready_invalid attempt)`；`None` = 该类事件不存在 |
| `_sam3_ready_beats_invalid(maxima)`（staticmethod） | 把上面的二元组翻译成原判据 |

`_manifest` 里只把谓词换成查表，集合的构造式本身不变。

**语义逐项对齐**（这是整个改动唯一有风险的地方，故逐条写明）：

| 原式 | 新式 | 说明 |
|---|---|---|
| `bool(ready)` | `ready is not None` | 「有没有 ready 事件」，与 attempt 取值无关 |
| `max(ready)` | 逐行 `max()` 累积 | 乱序 attempt 下二者同值；**不是**「最后一行胜出」 |
| `max(invalid or [0])` | `0 if invalid is None else invalid` | 无 invalid 事件时下限为 0 |
| `int(row.get("attempt") or 0)` | 同 | `attempt` 缺失 / `None` 一律读 0 |
| `row.get("source_id") == source_id` | 以 `str(source_id)` 建键 | `SourceRecord.source_id` 由 `sources.py:121` 强制为非空 `str`，journal 里也只写这个值，故 `str()` 归一不引入差异；`source_id` 为空的行**跳过不建键**（这类行原实现也永远匹配不上，因为查询侧的 id 必来自 `and row.get("source_id")` 过滤后的行） |

`None` 哨兵不是洁癖：`sam3_ready` 且 `attempt=None`（→0）、又无 invalid 时，原式算的是
`0 > 0` = **False**。若把「无 ready」也折成 0 就分不出这两种情形；若把「无 invalid」
折成 −1，`ready-zero` 那类源会被误判成已完成。两种折法都已写成变异体验证会被测试抓住。

## HOTFIX-三、等价性测试

新增 `dataset_build/tests/test_sam3_ready_index.py`（6 用例，其中 1 条 200 次随机 fuzz）。
夹具单条 journal 同时含：**乱序 attempt**（3 先于 2 落盘）、**ready 与 ready_invalid
交错**、**attempt 缺失 / `None`**、**ready 与 invalid 打平**（判据是严格 `>`）、
只有 invalid、只排队无事件、无 `source_id` 的行、未排队的源、重复 attempt 行。

断言口径：**逐 id 与保留下来的 `_unconsumed_sam3_ready` 对拍**，外加
`_manifest` 那条集合构造式的整体对拍，再加一条「夹具本身能区分」的防呆
（完成集恰为 `{interleaved, invalid-then-ready, duplicate-attempt}`，防止两侧一起全 False）。

```
$ PYTHONPATH=... python -m unittest tests.test_sam3_ready_index -v
Ran 6 tests in 0.011s — OK
```

变异体自证（测试真的会响）：

| 变异 | 结果 |
|---|---|
| 谓词 `>` 改 `>=` | FAIL（抓住） |
| 无 invalid 时下限取 −1 | FAIL（抓住） |
| 索引「最后一行胜出」而非取 max | FAIL（抓住） |

## HOTFIX-四、性能自证

L8 真实 `failures.jsonl` 只读载入后计时（`/home/bc/envs/databuild/bin/python`，
新路径取 20 次最优、旧路径取 3 次最优，两侧结果集断言相等）：

| 形状 | rows | queued sources | 旧 | 新 | 加速比 |
|---|---|---|---|---|---|
| L8 现状（原样） | 17,656 | 5,003 | **12,520 ms** | **1.96 ms** | **×6,381** |
| L8 + 合成 drain 事件（让新路径不是空索引） | 27,564 | 5,003 | 19,678 ms | 5.37 ms | ×3,663 |
| L8 ×3（逼近 400k 目标时的 journal 规模） | 52,968 | 5,003 | 38,328 ms | 4.72 ms | ×8,130 |

**读法**：现状下每次 land checkpoint 白扔 **12.5 秒**主循环时间，且随 journal
行数线性恶化（×3 那行 38 s）。782 次 checkpoint ≈ 2.7 小时纯空转，与
2,800 → 120 组/时的塌陷量级吻合。

## HOTFIX-五、冷路径清单（本次**未动**，逐条给理由）

同模式（每 source 一次全表扫）的调用点全部列出：

| 位置 | 函数 | 每次调用的扫描量 | 判定 |
|---|---|---|---|
| `_manifest:1001`（改前 `:950`） | `_unconsumed_sam3_ready` × 5,003 | 2 × O(F) each | **热路径，已改** |
| `_manifest:1091` | `_pending_sam3_ids` × 1 | O(F) + O(G) | 线性，**不动** |
| `_manifest:1070` | `_terminal_source_ids` × 1 | O(F) | 线性，**不动** |
| `_manifest:1092-1098` | `attempt_events` / `terminal` 计数 | 2 × O(F) | 线性，**不动** |
| `_fill_initial_mode:2476` | `_pending_sam3_ids` × 1 / pass | O(F) + O(G) | 每趟一次，**不动**（注释已声明这是 per-pass 快照） |
| `_fill_initial_mode:2475` | `_terminal_source_ids` × 1 / pass | O(F) | 同上，**不动** |
| `_drain_sam3_and_replacements:2624/2626/2636/2637/2647/2648/2651/2654` | `_sam3_attempts` / `_unconsumed_sam3_ready` **逐 pending 源** | 见下 | **平方级但本次不动 —— 见告警** |
| `LegacyImportPipeline._manifest:2866` | —— | 无 sam3 记账 | 不涉及 |

### ⚠ 告警：drain 阶段会再撞一次同一堵墙（本卡范围外，需主 agent 决策）

`_drain_sam3_and_replacements`（`:2614`）的 while 体内有四处**逐 pending 源全表扫**的
列表推导。用同一份 L8 journal（17,656 行 / 5,003 pending）实测**单次 while 迭代**：

| 片段 | 实测 |
|---|---|
| 循环 1：`_unconsumed_sam3_ready` per pending | 21.8 s |
| `exhausted` 列表：`_sam3_attempts` + `_unconsumed` per pending | 10.3 s |
| `candidates` 列表：同上 | 31.2 s |
| `next_attempt` + `batch`：2 × `_sam3_attempts` per candidate | 20.9 s |
| **单次 while 迭代合计** | **84.2 s** |

L8 目前 `sam3_ready` 事件数为 **0**（drain 还没跑到），所以这堵墙尚未撞上；一旦进入
drain，5,003 个 pending 源会让每轮 while 先烧 84 秒纯扫描，且 journal 每轮都在变长。

**本次不动的理由（不是遗漏）**：drain 的 while 体内 `_record_sam3_attempt` /
`_terminal_sam3` **会往 `failures` 追加行**，预建索引在迭代中途即失效 ——
照搬 `_manifest` 的快照做法会引入静默的陈旧读，性质比慢更坏。正确改法是在每轮
while 开头建一次索引、并在 batch 循环内对已改动的 source 定点失效（或直接改成
增量维护的 `dict`），属于**语义改动**，超出「热修 + diff 最小化」的授权范围。

建议：作为独立任务卡 HOTFIX-Sam3Drain-2 排在 L8 进入 drain 阶段**之前**。

## HOTFIX-六、测试与文件清单

```bash
cd /home/bc/VeraRetouch/dataset_build
PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build:/home/bc/VeraRetouch/dataset_build/src \
  /home/bc/envs/databuild/bin/python -m unittest discover -s tests -t . -p 'test_*.py'
```

（本机 `.venv` 与 `/home/bc/envs/databuild` 均无 pytest，故用 `unittest`；
测试全是 `unittest.TestCase`，行为一致。）

| 跑法 | 结果 |
|---|---|
| `tests.test_sam3_ready_index` | **6 / 6 OK** |
| `tests.test_source_reuse` | **40 / 40 OK** |
| 全套 `discover`（含本改动） | Ran **449**, errors **31** |
| 全套 `discover`（把本改动逐字回退后的同一工作树） | Ran 449, errors **36** |

差的 5 个正是回退后新测试找不到 `_sam3_ready_maxima` 而报错的那 5 条
（第 6 条只用 oracle，回退后仍过）。**31 个 error 为改动前既有**，全部是各
`*ConfigTests` 读已被 `83da846` 删除的 `databuild.example.toml`
（`FileNotFoundError: /home/bc/VeraRetouch/databuild.example.toml`），与本改动无关，
上文 §五 / N8 已记录同一现象。**本改动引入 0 个新失败。**

本卡文件清单：

```
dataset_build/src/construct/agent.py          (+52 −1，仅 §HOTFIX-二 两处)
dataset_build/tests/test_sam3_ready_index.py  (new)
docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md   (本节)
```

**未触碰**：运行中的 pid 587114、`/mnt/ramstage` 写路径（只读了
`failures.jsonl`）、任何 config、`trash/`；未启动任何 build。

---

# PIPE-PassInterleave-1 —— 渲染/标注按轮流水化（2026-08-14）

用户原话：「分 batch，一个 batch 所有图像各渲染一组，然后启动标注，第二组的渲染上卡」。

改动前 `_run_phases` 是严格串行的：`fill(渲染，多轮)` → `sam3_relabel` → `annotation`。
渲染相占满两张卡而 relay 全程空闲；标注相占满 relay 而两张卡全程空闲。
改动后**每走完源池的一趟（pass）就把该趟的标注任务批量交给一个后台标注线程**，
主线程立刻进入下一趟渲染；build 收尾时原 annotation 相退化为「排干剩余积压」。

**无新 config 字段，行为无条件生效。** 理由见 §P-七。

## P-一、实施前核实记录（全部本机读码 / 实测，无外部检索）

| # | 事实 | 核实方式与结果 |
|---|---|---|
| 1 | 渲染期所有 journal 写入都在主线程 | 属实。worker 走 `_render_source_buffered` → `_source_context.failures` 线程局部缓冲，主线程在 `_commit_source_result`（`agent.py`）按分配序 flush。这就是「单写者约定」的实体 |
| 2 | `ResponsesAnnotator` 本来就多线程写 store | 属实。`drain` 每轮开 `ThreadPoolExecutor(16)`，`_append_sft` / `_failure` 在 worker 线程直接调 `store.append_*`。**所以"标注多线程写"不是本卡引入的新事** |
| 3 | `ArtifactStore` 的四个写入口都在 `self._lock` 内 | 属实（`append_group/sft/failure`、`checkpoint`、`write_manifest`、`close`），RLock |
| 4 | 标注任务 id 是确定性的、幂等 | 属实：`stable_id("annotation", group_id, candidate_id, rank)`；`pending_annotation_tasks` 从 sft 行 + terminal 失败事件反推「已完成」，重启即可重建积压 |
| 5 | `_land_groups` 落盘后会 **unlink 掉暂存资产** | 属实（`for group in pending: for path in _group_assets: Path(path).unlink`）。这是本卡最大的正确性风险源，见 §P-三 |
| 6 | `land()` **不**写全局反查表 | 属实。全仓只有 `_refresh_catalog` 调 `upsert_catalog`，且只在 `_verify_group_assets`（有嫌疑组时）和 annotation 相之前各一次 |
| 7 | 现场实测（只读 `/mnt/ramstage`，未碰 pid 1790282） | `groups.jsonl` 57,234 组 / 1.11 GB；`failures.jsonl` 44,695 行；`sft.jsonl` **0 行**（标注未启）；`landing.checkpoints` **1,789**；`annotation.pending` **46,337**；归档已到 `groups/batch-1788` + `sft/batch-0764`；`/var/cache/veradata/global.sqlite3` **21.4 GiB** |

## P-二、改法总览（四个文件）

| 文件 | 改动 |
|---|---|
| `agent.py` | `QueueDrainer` 协议加 `only=`；新增 `_AnnotationDriver`（后台线程）；`CanonicalPipeline` 加 `_annotator` / `_driver` / `_annotation_submitted` / `_registered_datasets` 四个字段与 `_annotation_drainer` / `_annotation_bytes_settled` / `_flush_annotation_batch` / `_close_annotation_driver` 四个方法；`_fill_mode(..., annotate_passes=False)`；`_refresh_catalog(incremental=False)`；`_manifest` 的 `annotation` 块加 `inflight` / `batches`；`_manifest` 三处 sft 读改快照；`_run_phases` / `execute` 接线 |
| `responses.py` | `drain(*, max_workers=None, only=None)`；新增 `_next_rounds`（批量折叠，见 §P-五） |
| `state.py` | 新增 `group_records()` / `sft_records()` 锁内快照；`completed_sources` / `completed_source_uses` / `completed_annotation_tasks` / `pending_annotation_tasks` 改用之 |
| `tests/test_annotation_interleave.py` | 新增 27 例 |

触发点只有一处：`_fill_mode` 的 while 体内，`_fill_initial_mode` 返回之后、**三个 early return 之前**：

```python
cursor_skips = self._fill_initial_mode(mode, sources, target, use_index=use_index)
use_index += 1
if annotate_passes:
    self._flush_annotation_batch()
if final:
    return
```

放在 `if final` 之前是刻意的：`max_source_uses = 1` 时最后一趟就是唯一一趟，
放在后面等于「单趟 build 完全不流水」（变异 M4 就是这么做的，被 3 个测试杀掉）。

`_drain_sam3_and_replacements` 内部那次 `_fill_mode` **不开** `annotate_passes`：
那个 while 每救回一个源就重进一次，"一趟"只是几组替补，
每次都付一次目录刷新不划算；它们由收尾 drain 领走。

## P-三、⚠ 本卡的真正杀手：落盘 unlink 与标注读字节的竞争

任务卡把并行安全的重点放在 failures 单写者上。实际读码后，**更危险的是另一条**：

`_land_groups` 的最后一步是 `Path(after_path).unlink()`。而标注读 `I_tar` 走
`archive_reader.read_bytes` 的三级阶梯（本地文件 → prefetch → 归档），
**归档那级要求该 batch 已进全局反查表**，而 `land()` 自己不写反查表。
于是存在一个窗口：资产已被 unlink、batch 还没 upsert → `read_bytes` 抛 `KeyError`
→ `_encode_image` 抛 `AnnotationError("annotation_image_invalid", retryable=False)`
→ **terminal**。一次赶巧的 checkpoint 能把整批 winner 判死，且不可重试。

处理办法（不是加锁，是让窗口不存在）：

1. **`_annotation_bytes_settled`**：只交出 `after_path` **已经不在本地**的任务。
   `_land_groups` 的 `pending` 要求组内每个文件都存在才落盘，所以
   「文件已消失」⇔「已经落过盘」⇔「未来任何 checkpoint 都不会再动它」。
   归档是 append-only，字节从此不动。
2. **`_refresh_catalog(incremental=True)` 在交出批次之前跑**，把上次边界以来落的
   batch 全部登记进反查表。
3. 无 `archive_root` 的 build（所有旧测试、所有手搭 `PipelineDependencies`）
   从不落盘也从不 unlink，`_annotation_bytes_settled` 恒 True，全程可流水。

**代价与量化**：一趟渲染的"尾巴"（上次 cadence 落盘以来的组）本轮不交，等下一个
边界。实测 L8：57,234 组 / 1,789 次 checkpoint，落盘由 8 GiB 暂存水位驱动，
所以尾巴的上界是水位而不是一整趟 —— 现场最大一次 checkpoint 落 2,292 组，
即一趟 3 万组里最多约 7% 延到下一轮。这是与"整趟立即交付"的**唯一偏离**，
换来的是零 unlink 竞争窗口。

**为什么不改成"边界处强制 `_land_checkpoint(force=True)` 把整趟落干净"**（我先写了这版）：
它确实能让 100% 的一趟当场交付，但在小测试里凭空多切一个 batch，
`test_land_integration` 8 个用例（batch-0000 内容、`landing.checkpoints == 1`、
`annotation_status.datasets == 1`…）全部要改判据 —— 用一个标注改动去改写八个落盘
用例的语义，代价明显不划算，收益只有那 7%。已回退，仅保留在此备案。

## P-四、并行安全逐项排查（任务卡 §2）

先说结论：**「failures 单写者」这条约定被收窄、没有被打破，且不需要补锁或回主线程队列。**

### P-四.1 这条约定原本是干什么的

不是防数据竞争 —— `ArtifactStore._lock` 早就把四个写入口都串行化了，而且
`ResponsesAnnotator` 从第一天起就在 16 条 worker 线程上并发 `append_failure`。
它防的是**顺序**：`_render_source_buffered` 把 worker 的失败行缓冲进线程局部，
主线程在 `_commit_source_result` 里按**分配序**一次性 flush，
这样 failures.jsonl 里渲染行的相对顺序 = 源池顺序，而不是线程调度顺序。

### P-四.2 加了后台标注线程之后，逐个消费者复核

| 消费者 | 读什么 | 结论 |
|---|---|---|
| `_Sam3Ledger.refresh` | `store.failures` 游标增量折叠 | **不受影响**。只折 `sam3_*` 三种 event，标注行直接 skip；attempts 是计数、maxima 是 max，都与顺序无关；list append-only 且游标先取长度（其 docstring 已写明「并发追加无害」） |
| `_landing_counts.checkpoint_summaries` | `stage == "landing"` 行的**顺序** | **不受影响**：landing 行仍然只由主线程写 |
| `_terminal_source_ids` | `terminal and source_id and stage ∈ {rendering, sam3_relabel}` | **不受影响**：标注失败行 `stage="annotation"` 且**根本不带 `source_id` 字段**（`ResponsesAnnotator._failure` 的行结构里没有），双重不匹配 |
| `_pending_sam3_ids` | `error_code == "sam3_relabel_queued"` | 同上，不匹配 |
| `_failure_counts` / `failures.by_code` | 计数 | 渲染期 manifest 会开始出现标注类 code。**这是预期可见性变化**，不是错误 |
| `final_status` | `any(row["terminal"])` | 语义不变：标注 terminal 本来就让 build 收成 `complete_with_failures` |
| `mirror_artifacts` | 边写边读 ledger 文件 | **不受影响**：`_mirror_ledger` 的不变式本就是「镜像是源的字节前缀」，其 docstring 明写「对一个仍在被追加的文件做拷贝得到的是前缀而非末尾快照」；`_whole_records_limit` 在恢复侧按记录边界切 |
| `scan_jsonl` / `_load_unique`（resume） | 按 `event_id` 建索引 | 与顺序无关 |

### P-四.3 真正需要动的地方：dict 不能边插边遍历

这是加了第二个线程之后**新出现**的硬故障（不是理论风险）。CPython 的 `dict`
在迭代中被插入会抛 `RuntimeError: dictionary changed size during iteration`。
现在两个索引各自有了跨线程的读者：

| 索引 | 写者 | 跨线程读者 | 处理 |
|---|---|---|---|
| `store.groups` | 主渲染线程 | 标注线程的 `pending_annotation_tasks` / `completed_source_uses` / `completed_sources` | 全部改走 `group_records()`（锁内 `list(...)` 快照） |
| `store.sft` | 标注 worker 线程 | 主线程的 `_manifest` sft 计数、`_annotation_counts`、`_winner_annotation_status`；标注线程自己的 `completed_annotation_tasks` | 全部改走 `sft_records()` |
| `store.failures` | 两边都写 | 两边都读 | **不快照**。list 迭代器每步重读长度，并发 append 在 GIL 下良定义；而 failures 有几十个读点、L8 收尾时约 30 万行，每个读点复制一次是真实成本 |

快照只复制 **list**、不复制行对象（行入账后不再被修改，`append_group` 自己已经
deepcopy 过传入值）。`test_a_snapshot_is_a_copy_of_the_list_not_of_the_rows` 钉住这点。

**诚实说明一处**：`pending_annotation_tasks` 原本写的是
`sorted(self.groups.values(), key=...)`，而 `sorted()` 的物化在 CPython 里是一次
不释放 GIL 的 C 调用，所以那一行**恰好**不会炸。改成快照不是因为它今天会炸，
而是不想把正确性押在「`sorted` 碰巧是原子的」这个实现细节上。
真正会炸的是 `completed_source_uses` / `completed_annotation_tasks` /
`_annotation_counts` / `_winner_annotation_status` 这些 **Python 层 for 循环**，
变异 M2 / M3 各自复现了一次（见 §P-六）。

竞态测试把 `sys.setswitchinterval` 压到 `1e-6`：默认 5 ms 下一次遍历通常在一个时间片
内跑完，**根本采样不到**这个竞态 —— 这正是这类 bug 能活到生产、并且只在跑一周的
build 上现形的原因。测试里另有一条 `test_a_live_walk_of_the_same_index_really_would_break`
作为对照，证明所压的竞态确实存在，而不是工作量太轻。

### P-四.4 其余共享状态

| 状态 | 结论 |
|---|---|
| `_source_context`（`threading.local`） | 标注线程从不调 `CanonicalPipeline._failure`（它调的是 `store.append_failure`），线程局部缓冲无泄漏 |
| `store.checkpoint()` | 两边都调，都在 `_lock` 内；`_append_sft` 每行一次 fsync 是既有行为 |
| `_AnnotationDriver` 自己的计数 | `_inflight` / `_batches` / `_error` 全在自带 `Lock` 内；该锁**从不**在持有时进 store，不可能与 store 锁形成环 |
| `invalidate_shared()`（`_refresh_catalog` 内） | 只清模块级缓存 dict；已取到 reader 的线程继续用旧 `immutable=1` 快照，旧 db inode 被 open fd 钉住，`os.replace` 不影响它 |
| 双 drain 并发 | **结构上不可能**：driver 单线程串行；收尾 drain 在 `_close_annotation_driver()` join 之后才起。这条很重要 —— 同一 task 跑两遍会生成两行不同内容、同一 `sft_id` 的记录，`append_sft` 直接抛 `conflicting durable SFT record`，**而且是在钱已经花掉之后** |
| 批次不重叠 | `_annotation_submitted` 台账；变异 M5 删掉它后 5 个测试挂 |

### P-四.5 错误不许消失

`drain` 自己处理传输失败，能逃出来的都是结构性错误（store 冲突、KeyboardInterrupt）。
driver 记下**第一个**，在主线程的下一次 `submit` 或 `close` 后重抛。
`execute()` 的 `finally` 里只 join、**不重抛**（在 finally 里抛会顶掉正在上报的真异常）。

## P-五、附带修的一堵墙：`drain` 的每任务全表扫（不修则本卡无意义）

`drain` 每轮对**每个** pending 任务调一次 `_next_round`，而它内部是
`has_terminal_failure`（1 次全表扫）+ 最多 3 次 `_round_done`（各 1 次全表扫）。

拿**现场那份 L8 journal**（44,695 行）实测：

| | 实测 |
|---|---|
| `_next_round` 单任务 | **14.4 ms** |
| × 46,337 个 pending 任务 | **11.1 分钟 / 轮**，3 轮 ≈ **33 分钟 / 次 drain** |
| 新增 `_next_rounds` 批量折叠（同一 journal、同一批 46,337 任务） | **14.8 ms** |

而 journal 到 build 收尾会涨到约 30 万行（≈7 倍），即 33 min → **约 4 小时**。
这本是既有问题（收尾一次性 drain 同样会撞），但流水化把它变成**每轮边界都撞一次**，
且是在**和渲染抢 GIL 的线程上**烧 CPU。折叠之后一次 drain 的这部分从小时级降到毫秒级。

`_next_rounds` 与 `_next_round` 逐任务等价（一次遍历折出 terminal 集合与
`(task, round)` 的 round_exhausted 集合），`_next_round` 原样保留为**参照定义**，
`RoundFoldTests.test_the_fold_agrees_with_the_per_task_definition` 拿它当 oracle 逐个比对。
折叠**按轮重建**而非按 drain 重建，因为 `drain` 会在轮之间写 `round_exhausted`。

**未动的另一处**（明确记账，不是遗漏）：`run_round` 入口的 `_attempt_count` 仍是
每任务一次全表扫（14.4 ms 里约 1/4）。happy path 上另外两个扫描
（`_local_attempted` / `_draw_attempt_count`）都被短路：L8 `local_fallback = false`
且首个 attempt 时 `attempt(0) >= 4` 为假，`_local_rescue_due` 压根不被调用。
`_attempt_count` 直接关系到**每个任务还能重试几次 = 花多少钱**，把它换成快照要逐条论证
「本轮内该 task 的 attempt 行只有它自己在写」，属于计费语义改动，
不放进这张以流水化为题的卡。建议作为 HOTFIX-AnnotAttemptIndex-1 单独排。

## P-六、变异自证（8 个，全部被杀）

做法与 §HOTFIX-三 相同：把 `src/construct` 整棵拷进影子包，
`PYTHONPATH=<影子>:...` 前置覆盖，**工作树一字未改**。

| # | 变异 | 被杀于 |
|---|---|---|
| M1 | `_annotation_bytes_settled` 恒 True（取消落盘 settled 门） | `test_a_still_staged_winner_waits_instead_of_being_handed_over`、`test_the_settled_test_follows_the_local_file_not_the_journal`（2 fail） |
| M2 | `completed_source_uses` 改回遍历活的 `self.groups.values()` | `test_the_pending_queue_survives_a_concurrent_render_thread`（`RuntimeError: dictionary changed size during iteration`） |
| M3 | `completed_annotation_tasks` 改回遍历活的 `self.sft.values()` | `test_the_manifest_readers_survive_a_concurrent_annotation_thread` |
| M4 | 把 `_flush_annotation_batch()` 挪到 `if final: return` **之后** | `test_a_single_pass_build_still_pipelines`、`test_a_last_pass_error_surfaces_at_the_join`、`test_a_landed_winner_is_handed_over_and_read_out_of_the_archive`（3 fail） |
| M5 | 去掉 `self._annotation_submitted |= batch` | 5 fail，含 `test_every_pass_is_handed_over_at_its_own_boundary`、`test_the_closing_drain_owns_whatever_the_driver_did_not_finish` |
| M6 | driver 的错误被吞（`submit` 不重抛 + `_run_phases` 不重抛） | `DriverFailureTests` 两例 + `DriverTests.test_the_first_error_is_kept_and_re_raised_on_the_caller`（3 fail） |
| M7 | 边界处不做 `_refresh_catalog(incremental=True)` | `test_a_landed_winner_is_handed_over_and_read_out_of_the_archive`（标注读不到归档字节） |
| M8 | `_next_rounds` 折叠里漏掉 terminal 判定 | `RoundFoldTests.test_the_fold_agrees_with_the_per_task_definition` |

## P-七、待主 agent 决策 / 声明式偏离

1. **不做开关（任务卡已指定，此处补论证）**：`run()` 的 resume 会把
   `existing["effective_config"]` 与 `config.sanitized_dict()` 做**逐 key 精确比较**
   （`_resume_config_differences`），差一个 key 就 `StateError` 拒绝 resume。
   `sanitized_dict` 走 `dataclasses.asdict`，新增任何字段都会自动进 effective_config，
   于是**新开关 = L8 无法 resume**（除非再走 `_RESUME_NEUTRAL_DEFAULTS` 白名单，
   那是给"默认值等价于旧行为"的字段准备的，而本改动的默认值就是新行为）。
   故行为无条件生效。
2. **`manifest.phase` 在流水期取值**：**保持 `rendering`**。phase 命名的是
   「谁占着 GPU、resume 要重做哪一段」，relay 忙不忙不改变这两件事。
   后台工作体现在新增的 `annotation.inflight`（已交出且未结算的任务数）与
   `annotation.batches`（已交出的整趟数）。
   **副作用（需操作侧知悉）**：渲染期 manifest 的 `annotation.pending` 从此**正常非零**，
   它不再等价于"标注没做"；`inflight` 是区分二者的字段。完成判据仍读收尾 drain 之后的
   `pending`，那时 `inflight == 0`、两者一致。
3. **`_refresh_catalog` 新增 `incremental=`**：`_landed_datasets` 返回的是**本 build
   至今全部** batch，L8 现在已有 1,789 + 765 = 2,554 个；`upsert` 会把它们全部
   delete + 重建索引，并且**每次都整份拷贝 21.4 GiB 的 sqlite**。
   一趟一次全量 = 对整个 build 的索引做 12 次平方级重扫。故边界处只登记
   「本进程尚未登记过」的 batch；`_verify_group_assets` 与 annotation 相之前那两次
   **仍是全量**（后者也是 resume 时接管上一次运行所落 batch 的地方）。
   首个边界因台账为空而等价于全量，是每进程一次的固定成本。
4. **未修 `_attempt_count`**（见 §P-五末），建议单独排卡。
5. **CPU 争用（不是正确性问题，但重启后要看）**：标注的图像编码（PIL 解码 + LANCZOS
   缩放 + JPEG 编码，每任务 2 张）现在与 `postprocess_workers = 16` 同时跑。
   建议 L8 重启后头一小时对比 groups/h 与重启前的 baseline；若掉得明显，
   最小干预是把 relay 并发从 16 降到 8（改 TOML，需另开重启窗口）。**本卡不动 TOML。**

## P-八、resume 场景（任务卡 §4）

| 场景 | 行为 |
|---|---|
| 渲染中断、标注已交出若干批 | 已写 sft 行的任务不再 pending（`completed_annotation_tasks` 从 sft 行 + terminal 事件反推）；未完成的任务重启后仍 pending，**在下一个边界重新交出**。任务 id 确定性 ⇒ 不重复计费 |
| 标注中断（driver 抛错 / 进程被杀） | 同上。driver 的错误在主线程重抛 ⇒ build 失败而不是静默继续；重启后积压原样存在 |
| L8 这种「已渲 5.8 万组、标注一次没跑」的重启 | mix.local=1.0 ⇒ `global_target = 0` ⇒ `_fill_mode("global", (), 0)` 空走一趟然后**立刻触发第一个边界**：把 46,337 个积压任务里所有已落盘的（现场 `sft_winners = 46,337` 全部已落盘）一次交给 driver，随后渲染照常进入 local 的第 3 趟。**这就是任务卡要的"resume 后已渲组必须能被补标"** |
| 二次 resume（一切都已完成） | 每个边界都量到空 delta ⇒ driver 根本不被创建 ⇒ `annotation.batches == 0`，sft.jsonl 字节不变。`test_a_second_resume_re_annotates_nothing` 钉住 |
| 收尾 | `_close_annotation_driver()` join → 全量 `_refresh_catalog()` → 无 `only` 的 `drain()` 排干剩余积压 → `pending` 仍非零则照旧 `PipelineError("annotation queue remains unresolved")`。`complete` / `complete_with_failures` 判据一字未改 |

## P-九、测试

```bash
cd /home/bc/VeraRetouch/dataset_build
PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build/src:/home/bc/VeraRetouch/dataset_build \
  /home/bc/envs/databuild/bin/python -m unittest discover -s tests -t .
```

| 跑法 | 改动前 | 改动后 |
|---|---|---|
| 全套 `discover` | Ran **514**, failures **0**, errors **31** | Ran **541**, failures **0**, errors **31** |
| 31 条 error 构成 | 30 × `FileNotFoundError: databuild.example.toml` + 1 × `ModuleNotFoundError: uvicorn` | **逐条相同** |

点名回归（任务卡 §6）：

| 套件 | 结果 |
|---|---|
| `tests.test_mirror_cadence` | **26 / 26 OK** |
| `tests.test_land_watermark` | **18 / 18 OK** |
| `tests.test_source_reuse` | **40 / 40 OK** |
| `tests.test_sam3_drain_ledger` | **21 / 21 OK** |
| `tests.test_sam3_ready_index` | **6 / 6 OK** |
| `tests.test_sam3_batch_fallback` | **2 / 2 OK** |
| `tests.test_land_integration` | **19 / 19 OK**（判据一条未改，只改了 double 的 `drain` 签名） |
| `tests.test_annotation_interleave`（新） | **27 / 27 OK** |

新测试覆盖：每趟一批 / 最后一趟也流水 / 单趟 build 也流水 / 空 delta 不建批 /
任务不被交出两次 / 流水期 phase 仍是 rendering 且 `inflight` 非零 /
收尾 drain 接管 driver 没做完的 / 队列没排干仍然失败 / terminal 标注仍收成
`complete_with_failures` / resume 补标 / 二次 resume 零重标 /
暂存 winner 不被交出 / 已落盘 winner 交出且**从归档**读到字节 /
driver 串行 + inflight 记账 + 首错保留与重抛 + 错误之后的排队批次被丢弃 + close 幂等 /
groups 与 sft 两侧的并发遍历 + 一条证明竞态确实存在的对照 /
`_next_rounds` 与 `_next_round` 逐任务等价。

**既有测试只动了三处**：
- `test_canonical_orchestration.py`：`FakeAnnotator` 加 `only` 支持（拆出 `tasks()` 辅助），
  以及 `test_annotation_interrupt_resumes_pending_stable_task_ids_only` 的
  `manifest["phase"]` 断言 `"annotation"` → `"rendering"`。
  **这是真实语义变化**：标注现在由渲染趟边界驱动，该用例的中断因此发生在渲染相；
  收尾 drain 自身被中断仍然留在 `"annotation"`，由新套件的
  `test_a_last_pass_error_surfaces_at_the_join` 覆盖。该用例其余全部断言
  （只重标未完成的 3 个、任务 id 稳定、无重复）原样通过。
- `test_land_integration.py`：`PartialAnnotator.drain` 加 `only` 支持。**无判据改动。**

## P-十、红线扫描

| 红线 | 涉及 |
|---|---|
| AUC / 空间场判据 | 无涉（本卡不产生任何指标） |
| 可视化 min-max / 色标 / 叠图 resize | 无涉 |
| G 初始化、σ 参数化、s 轴正则、逐像素算子、ckpt 选择、s 归一化、消融 Δ 列、IoU 当目标、干预对象、烘焙一致性、attention eager | 无涉 |
| `trash/` | 未创建、未写入 |
| NFS | 只读 `/mnt/ramstage`（本地 tmpfs）与 `/var/cache`；**未写 `/mnt/nfs`**，未起 `nfsx` |
| 运行中的 build（pid 1790282） | 未碰。改动在下次重启生效 |
| TOML | 未改 |

本卡文件清单：

```
dataset_build/src/construct/agent.py                  (+~200)
dataset_build/src/construct/responses.py              (+~70)
dataset_build/src/construct/state.py                  (+~35 −4)
dataset_build/tests/test_annotation_interleave.py     (new, 27 例)
dataset_build/tests/test_canonical_orchestration.py   (double 的 only 支持 + 1 条 phase 断言)
dataset_build/tests/test_land_integration.py          (double 的 only 支持)
docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md           (本节)
```

---

# PIPE-StopFill-2 —— 停渲标记 + 无 GPU 排干模式（2026-08-14）

日期：2026-08-14 ｜ 实施者：编码 subagent ｜ 交付：`<output_root>/STOP_FILL` 标记文件

用户指令：L8 不再渲染新组（现 61,471 组，放弃 40 万目标），把 ~5 万组标注积压做完，
GPU 完全让给训练线。

**未启动任何 build，未改任何 TOML，未碰 `trash/`，未写 `/mnt/nfs`（只读了
`/mnt/ramstage/prod-l8-local400k-20260812/manifest.json` 一次取现场数字）。**

## S-一、实施前核实（全部本机读码 / 实测）

| 事实 | 核实方式 | 结论 |
|---|---|---|
| L8 进程 pid 3110570 在跑 | `ps -p 3110570` | **已不在**（本卡开工时已退出）。manifest 停在 `phase=rendering / status=running`，是死进程留下的陈述 |
| L8 现场 | 读 `manifest.json` | groups **61,471** / target 400,000；`annotation.pending` **49,741**；`sft` **0**；`sam3_relabel.pending` **7,702** |
| 标记落点 | `ls /mnt/ramstage/prod-l8-local400k-20260812/` | **`STOP_FILL` 已存在**（0 字节，08-14 12:06，非本卡创建）。路径与本卡实现完全一致，重启即生效 |
| resume 对 `effective_config` 逐路径精确比较 | `agent.py:_resume_config_differences` | 属实 → 这就是**不加 config 字段**的理由（同前七卡） |
| `self.renderer/self.scorer` 已是 `Optional` | `agent.py:1281-1282`（`_release_heavy_resources` 置 None）| 属实 → 传 `None` 不引入新状态 |
| 渲染守卫已存在 | `agent.py:2817` `if self.renderer is None or self.scorer is None: raise PipelineError` | 属实 → 万一走错路是**响亮失败**，不是静默降级 |

## S-二、无 GPU 论证：逐加载点清单

canonical build 全生命周期的 GPU 加载点**只有五个**，逐个给出停渲模式下的去向：

| # | 加载点 | 代码位置 | 真正碰卡的动作 | 停渲模式 |
|---|---|---|---|---|
| 1 | `dependencies.renderer_factory` = `LocalGpuOnlyRenderer.create` | `rendering.py:165` → `_preflight()` | `torch.zeros((1,), device="cuda:1")`——**这一次分配就是 CUDA context** | `run()` 的 `if not stop_fill` 分支内，**不调用** |
| 2 | `renderer.assert_ready()` | `rendering.py:188` | `torch.cuda.is_available()` | 同上，不调用 |
| 3 | `_load_scorer` → `OneAlignScorer.create` / `OneAlignScorerPool.create` | `canonical_qa.py:36 / 88` → `OneAlignRunner.load()` | cuda:0 上 1–2 份 OneAlign 权重 | 同上，不调用 |
| 3b | `_preflight_scorer` 的一次真实 forward | `agent.py:_preflight_scorer` | 同上 | 3 不调用则不可达 |
| 4 | `dependencies.relabeler` = `_default_relabeler` | `agent.py:487` `device="cuda:0"` | SAM3 860M 检测器 | **只从 `_drain_sam3_and_replacements` 调用**，该相被跳过 |
| 5 | `_empty_cuda_cache()` | `agent.py` | `torch.cuda.empty_cache()` | 两处调用点均被绕开，且函数本身加了 `torch.cuda.is_initialized()` 门 |

外加 `visibility.py` / `rendering.py` 里的 torch 调用（`visibility.py:688/753/806/926/962/1008`、
`rendering.py:392`）——**全部只从 `_render_source` / `_postprocess_candidate` 可达**，停渲后不可达。

**排干链路真的不碰 GPU 的正面证据**（不是"没找到"，是"跑出来的"）：

```
$ python -c "import sys; import construct.agent, construct.responses, construct.projection;
             from dataset_build.tools import archive_reader, land, global_catalog, prefetch;
             print('torch' in sys.modules)"
False
```

即 `construct.agent` 虽然 `from .rendering import ...` / `from .visibility import ...`，
但那两个模块的 `import torch` **全在函数体内**，模块级 import 链一次也不拉 torch。
排干侧读图走 `archive_reader`（tar + sqlite + PIL，见 `tools/archive_reader.py` 的 import 段），
`I_tar` / `I_in` 都是已落地归档，标注是 relay HTTP，landing 是 hardlink + 打包，
projection 是 Postgres——**任务卡担心的"隐藏 GPU 依赖"，逐条查完确认不存在**。

`_empty_cuda_cache` 的加门是**必要的而非保险**：停渲进程里 torch 可能一次都没被 import，
但 `_release_heavy_resources` 无条件调它就会把 torch 拉进来；`torch.cuda.is_initialized()`
是纯 Python 侧惰性初始化标志（不进 driver），所以加门后即便被调也不建 context。
另外 `_release_heavy_resources` 在 `renderer is None and scorer is None` 时直接 return，
连 `import torch` 都省了。

端到端实测（**子进程内**跑完一次停渲 build，避免同进程别的用例污染）：

```
complete_with_failures 0 False True
                       ^ groups  ^ torch.cuda.is_initialized()
```

> 用子进程而不是同进程断言，是因为本套件里确有用例合法地驱动真 torch 并把
> `torch.cuda.is_initialized()` 留成 True；"这个 build 进程碰没碰卡"的主张
> 必须在只跑过这个 build 的进程里做。

**仍然会跑的启动依赖（都是 CPU/DB，且不能省）**：`catalog_loader`（读 preset bank 的
jsonl，`presets.py` 无 torch）、`inventory_loader`（Postgres + 文件，`sources.py` 无 torch）。
不能省的原因：`allocate_sources` 要用 `inventory.eligible` 算出 `local_target`，
而 `local_target` 正是 shortfall 记账的分母；`_manifest` 要 `catalog.inventory_counts`。

## S-三、改法（`agent.py`，+250 −14）

1. **标记谓词**：`STOP_FILL_MARKER = "STOP_FILL"` + `_stop_fill_marker()` / `_stop_fill_requested()`。
   用 `exists()` 而非 `is_file()`——`touch` / `echo >` / `mkdir` 都算数，
   操作员停生产 build 时答案不该取决于他敲了哪条命令。
2. **`run()` 的无 GPU 分支**：`stop_fill = _stop_fill_requested(config.output_root)` 只算一次，
   `if not stop_fill:` 里才建 renderer/scorer，并把 `stop_fill=` **显式传给** `CanonicalPipeline`。
   > 为什么传而不是让 pipeline 自己再读一遍：两次读之间隔着 `sdk_preflight` /
   > catalog / inventory / ArtifactStore 打开（生产上是几十秒），期间落下的标记会造出
   > "pipeline 认为可以渲染、进程却没加载 renderer"的组合。传值把这类 bug 直接消掉。
3. **`_stop_fill()` 谓词（带闩）**：未闩上时 stat 一次标记；一旦为真就闩住，
   本进程内不再 stat、答案单调。
   - *为什么闩*：① 每源 poll 一次，26k 源的一趟 stat 才有意义，闩住后只 stat 到发现为止；
     ② **单调性**才是主因——半趟认为"停"、半趟认为"没停"的 build 处在两种状态之外。
     删标记因此在**下次启动**生效（GPU 也不可能中途"取消释放"），这正是可逆语义。
   - 转变瞬间写**一条**非 terminal 事件 `stop_fill_requested`（`event_type="stop_fill"`），
     message 里带 marker 路径 / `at_start` / phase / 两个 mode 的组数。
4. **趟内安全边界**（`_fill_initial_mode` 的 `for` 体首行）：发现即 `break`。
   优雅的关键是**在提交下一个源之前**跳出，而不是打断正在渲染的源：
   in-flight 窗口（生产 `source_window=5`）由循环尾部原有的 `while inflight: finish_oldest()`
   逐个提交完并做收尾 land checkpoint，与一趟正常结束**走同一段代码**。
   代价上界 = 一个窗口的渲染。
5. **趟间**（`_fill_mode`）：顶部一次（启动即停 ⇒ 连 `completed_source_uses()` 和
   `_terminal_source_ids()` 两次 O(groups) 扫描都不付），循环内**在
   `_flush_annotation_batch()` 之后**返回——被截断那趟的赢家仍然进流水线，
   是本 build 最后一批交给标注的任务。
6. **sam3_relabel 相**：`_run_phases` 里 `if not self._stop_fill(): self._drain_sam3_and_replacements()`。
   跳过而非空转，因为它**两半都要卡**：`relabeler` 是 cuda:0 上的 SAM3，
   救回的源紧接着走 `_try_ready_relabel → _render_source`（正是被停掉的活，
   而且会撞上 `render/QA resources are not loaded`）。
   队列**原样留着**：7,702 个源保留 `sam3_relabel_queued`，**不消耗任何 attempt 预算**，
   将来去掉标记的 build 原封不动接手。收尾的
   `raise PipelineError("SAM3 relabel queue remains unresolved")` 加了 `and not self._stop_fill()`
   ——那句话的意思是"drain 跑了但收不了尾"，而这里 drain 根本没跑。
   另在 drain 的 `while` 顶部加了一次检查（生产上 drain 本身是小时级，标记可能中途落下），
   轮次之间是它的安全边界。
7. **manifest 新增 `stop_fill` 块**：`requested` / `at_start` / `gpu_resources_loaded` / `marker`。
   `at_start=false` + `gpu_resources_loaded=true` 是"跑着跑着被停"的诚实描述。
   读闩而不调 `_stop_fill()`——**写 manifest 不该是往 journal 追加行的那个动作**。
8. **shortfall 记账不变**：`_record_shortfalls` 仍按配置目标算，L8 将落一条 terminal
   `local_target_shortfall`：`requested 400000 local groups, completed 61471`。
   `complete_with_failures` 因此由**一直以来那条规则**（有 terminal 行）达成，判定未改。

## S-四、⚠ 顺手挖出来的一颗 HEAD 上的雷：短量 build 无法重启

**现象**（在**干净 HEAD**、无本卡任何代码的影子树上复现，用真实时钟）：

```
run 1: complete_with_failures 3
run 2 RAISED: StateError conflicting durable failure event: failure_9bbc175ea474507645d4ec5d5638a7a1
```

**机制**：`ArtifactStore.append_failure` 只对**逐字节相同**的行去重
（`state.py:337 if existing != stored: raise`），而每行都带 `timestamp`。
`_record_shortfalls` 用固定 `task_id`，于是**任何记过 shortfall 的 build 再启动一次就死**，
死在做任何实事之前。之所以从没暴露：
① 测试 fixture 的 `now` 是冻结常量，两次跑出的行逐字节相同；
② 生产上 `_record_shortfalls` 只在渲染相**跑完**才可达，而 L8 从没跑完过。

**为什么本卡必须修**：停渲把它从"不可达"变成"必然"——停渲 build 第一次就记 shortfall，
而 49,741 条标注的排干是长任务，**几乎一定要重启**，第二次重启就撞上。

**改法**（`_journal_once`，两个调用点）：
- 把**合法变化的量折进 `task_id`**：shortfall 折进 `completed`，stop 事件折进
  `(at_start, local, global)`；
- 追加前查一次"这个 task_id 是否已有行"。
- 合起来的语义：**"这句话我们是不是已经原样说过"**，而不是"是不是说过话"。
  没变化的重启静默；真变了（比如去掉标记后又填了 4 组）就记新数字，
  而不是因为和旧行不一致被拒。

> `stable_id("target", build_id, mode)` → `stable_id("target", build_id, mode, completed)`
> 是 journal 格式变化。已 `rg` 确认**无任何消费方**读这个 task_id，
> 且 L8 现场还没有 shortfall 行，无兼容问题。

其余 durable 事件不需要这层保护，因为它们都写在 resume 不会重走的路径上：
terminal 源在渲染前就被拒、lost group 被 `lost_group_ids()` 跳过、
queued 的 sam3 源已在 `_pending_sam3_ids()` 里。
`_run_phases` 自己写的这两条，是**第一次真的会被重启走第二遍**的事件。

## S-五、待主 agent 决策 / 声明式偏离

1. **【已按保守默认执行，需追认】修了 S-四 那颗 HEAD 上的雷。**
   严格说超出"停渲标记"的范围，但不修则本卡交付的东西**第二次重启就废**。
   影响面：`_record_shortfalls` 的 `task_id` 形状变了（无消费方）。
2. **停渲后再去掉标记填满，status 仍是 `complete_with_failures`。**
   因为停渲那次写的 terminal shortfall 行是历史，不是可撤回的陈述。
   已由 `test_removing_the_marker_and_restarting_resumes_filling` 钉住并写进 docstring。
   要改成"填满即回 complete"就得删 durable 行，不做。
3. **停渲 build 仍会写 NFS**（landing / mirror / catalog refresh 照常）。
   任务卡的"让出显卡"已满足；"完全不碰 NFS"不是本卡目标，也做不到——排干本身要读归档。
4. **`sam3_relabel.pending = 7702` 会一直挂着**，manifest 如实报告。
   若日后要把这批彻底了结，需要一次**不带标记**的 build（会用 SAM3 + 渲染器）。

## S-六、单测（`dataset_build/tests/test_stop_fill.py`，**40 例 / 3 subtest**，新增）

```bash
cd /home/bc/VeraRetouch
PYTHONPATH=/home/bc/VeraRetouch:/home/bc/VeraRetouch/dataset_build:/home/bc/VeraRetouch/dataset_build/src \
  /home/bc/.venvs/iaa437/bin/python -m pytest dataset_build/tests/ -q -p no:randomly
```

分组（对应任务卡 §5 五项要求）：

- **标记谓词** `MarkerTests`（4）：不存在 / 根目录都还没建 / `touch`·`echo`·`mkdir` 三写法 / 路径常量。
- **启动即标记** `StartupStopFillTests`（8）：三个 GPU 工厂 + relabeler 全部 `mock` 成
  「一被调用就 AssertionError」并断言 `.called is False`；不走一趟
  （`_fill_initial_mode` 未被调用）；生命周期照常走到 projection；
  shortfall 数字与 message；stop 事件只一条且非 terminal；
  **积压排干**（先用 `RefusingAnnotator` 造出 3 组 0 标注的欠账，再带标记重启 → 6 条 sft / `complete`）；
  排干失败仍然报错（停渲**不豁免**标注完成门）。
- **sam3** `Sam3UnderStopFillTests`（6）：relabeler 未被调用 / 队列原样 / `attempt_events == 0`；
  相**根本没进**（`_drain_sam3_and_replacements` 未被调用）；轮次之间的中途停；
  收尾 `raise` 两个方向都钉（白盒直调）；未报成 drain 失败；去掉标记后队列被救回。
- **中途出现** `MidPassStopFillTests`（6）：窗口内的源全部提交完（5 源池 → 3 组，
  每组 8 candidate + 有 winner）；manifest 区分 `at_start`；被截断那趟仍交批
  （`annotation.batches == 1`）；第二个 mode 不开；下一趟不开；事件内容。
- **可逆** `ReversibilityTests`（3）：删标记重启即恢复填充；
  `effective_config` 两次逐键相等（= 为什么用文件不用 config 键）；开→关→开三次切换。
- **重启** `RestartWithAShortfallTests`（3）：S-四 的回归，**用会走的时钟**；
  连续 3 次停渲重启；填得更多时记新数字。
- **无 CUDA** `NoCudaContextTests`（5）：`is_initialized` 门两个方向；
  停渲 build 根本不到 allocator；**普通 build 仍然释放显卡**（守卫没把正常路径变成空操作）；
  排干链路模块不 import torch（子进程）；停渲 build 跑完 driver 仍未初始化（子进程）。
- **无标记时一切照旧** `UnchangedWithoutTheMarkerTests`（4）：正常填满 / 复用仍走满趟 /
  sam3 drain 仍跑仍收尾 / renderer 仍被建仍被 `bind_catalog` + `assert_ready`。

### 变异自证（9 处，**全部被杀**）

| 改回 | 实测（40 例） |
|---|---|
| M1 去掉趟内边界（只在趟间查） | **2 failed** |
| M2 `run()` 恒加载 renderer + scorer | **14 failed** |
| M3 停止返回挪到 `_flush_annotation_batch` **之前** | **1 failed** |
| M4 `_run_phases` 不跳过 sam3 相 | **1 failed** |
| M4b 去掉 drain 轮次之间的检查 | **1 failed** |
| M5 `_journal_once` 的守卫去掉（= HEAD 行为） | **2 failed** |
| M6 `_empty_cuda_cache` 去掉 `is_initialized` 门 | **1 failed** |
| M7 标记只在启动读、运行中不再 poll | **13 failed** |
| M8 收尾 sam3 `raise` 不再看标记 | **1 failed** |

> M1 / M4 初版变异**没被杀**，两处都补了测试才补上：M1 是变异锚点写错（插了死代码没删真检查）；
> M4 是 `_run_phases` 的守卫被 drain 内部守卫遮蔽（纵深防御生效），
> 补 `test_the_phase_is_not_entered_at_all`（结构断言：该方法未被调用）后才可分辨。
> M8 在整 build 里不可达（`_run_phases` 会先跳过该相），
> 改成白盒直调 `_drain_sam3_and_replacements` 两个方向才钉住。

### 全量回归

| | 通过 | 失败 |
|---|---|---|
| 改前基线 | 512 | 30 |
| 改后 | **552** | **30** |

失败集合与基线 **`diff` 逐条相同**（30 条全是 D1 的 `databuild.example.toml` 缺失）。
任务卡点名的各套件：interleave **27**、mirror **26**、watermark **18**、
source_reuse **40**、sam3_drain_ledger **21**、sam3_ready_index **6**、stop_fill **40** —— 全绿。

> 任务卡写的是"已知 31 错误"，本机实测基线是 **30**；改前改后都是 30，逐条一致。

**唯一改到的既有测试文件**：`dataset_build/tests/test_sam3_drain_ledger.py`。
`_ScriptedDrain`（假装成 pipeline 的差分测试替身，从不调 `__init__`）需要
`config.output_root` 与三个 `_stop_fill_*` 字段，否则 drain 里新增的检查 `AttributeError`。
给的是**真实状态**（一个没有标记的目录）而不是覆盖 `_stop_fill`，
所以这份差分比较跑的仍然是要上线的那段代码。断言一字未改。
另：该改动对**改前的 agent.py 也完全兼容**（用重建的 pre-change agent 跑该套件仍全绿）。

## S-七、给主 agent 的重启须知

标记已在位：`/mnt/ramstage/prod-l8-local400k-20260812/STOP_FILL`（0 字节，与实现契约一致）。
重启后这次 run 的预期轨迹：

1. 不建 renderer / 不加载 OneAlign / 不建 CUDA context —— **可按 CPU 任务入队，gpu1 卡+槽全让**；
2. `_verify_group_assets` 走一遍 61,471 组（**这是本次最长的 CPU 前置**，
   死进程留下的未落地资产会被记 `assets_lost`——既有行为，非本卡引入）；
3. 两个 `_fill_mode` 立即返回，写一条 `stop_fill_requested`；
4. sam3 相跳过，7,702 个 queued 源原样保留；
5. 一条 terminal `local_target_shortfall`：`requested 400000 local groups, completed 61471`；
6. 强制 land checkpoint（**会写 NFS**）→ 全量 `_refresh_catalog` → 收尾 drain 跑 **49,741** 条标注；
7. `_sync_annotation_status` 回写归档 metadata → projection → `complete_with_failures`。

第 6 步是长任务；**中途挂掉可以直接重启**（S-四 修的就是这个）。
要恢复渲染：`rm /mnt/ramstage/prod-l8-local400k-20260812/STOP_FILL` 后重启即可，
届时会重新加载渲染器与 OneAlign（需要归还 GPU 槽位）。

## S-八、红线扫描

| 红线 | 涉及 |
|---|---|
| AUC / 空间场判据 / 可视化 min-max / 色标 / 叠图 resize | 无涉（本卡不产生任何指标与图） |
| G 初始化、σ 参数化、s 轴正则、逐像素算子、ckpt 选择、s 归一化、消融 Δ 列、IoU 当目标、干预对象、烘焙一致性、attention eager | 无涉 |
| `trash/` | 未创建、未写入 |
| NFS | 只读 `/mnt/ramstage`（本地 tmpfs）；**未写 `/mnt/nfs`**，未起 `nfsx` |
| 运行中的 build | pid 3110570 本卡开工时已不在；`/mnt/ramstage/prod-l8-*` 只读了 manifest 一次 |
| TOML | 未改（本卡**不加任何 config 字段**，理由见 S-三.1） |
| 前七卡改动 | 一行未回退（重建 pre-change agent 跑基线得 512/30，与开工时逐条相同） |

本卡文件清单：

```
dataset_build/src/construct/agent.py                (+250 −14)
dataset_build/tests/test_stop_fill.py               (new, 40 例)
dataset_build/tests/test_sam3_drain_ledger.py       (替身补 4 个字段 + 2 个 import，断言未动)
docs/NOTES_TOOL-SourceReuse-1_2026-08-12.md         (本节)
```
