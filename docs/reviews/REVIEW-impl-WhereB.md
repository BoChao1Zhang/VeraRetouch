# REVIEW-impl-WhereB · Stage-Where-B MetaCanvas 独立实现审阅

**当前判决（第 6 轮 · 无人值守加固复审，2026-08-10，commits `d32ec60` + `244f770`）：BLOCKER 1 个 —— D-B17 的读路径改造漏掉了**主数据流**（records/images 仍走 hard 挂载），且启动断言会为此打绿灯。eval 防御 / D-B18 / EXEC-3 合并 / W1 溯源均 PASS。
**W03–W08 在 W-B1 修好前不得按「已受 nfs-ro 保护」的前提启动。** 判据侧（第 5 轮）结论不受影响。**

> 第 5 轮判决（2026-08-05 夜，`2c41e3c`）：BLOCKER 清零 —— Where-B **评测侧**完全合规，
> S6 选型与 REPORT 均解锁（配对 Δ 为报告列非 gate）。**该结论继续有效**，本轮问题在运行时基础设施，不在判据。

三个对抗构造由本审阅独立回归，全部通过：

| 构造 | 结果 |
|---|---|
| 零信息场（中心先验）· 同心 GT（F-B1 原案，raw 曾 **+0.7590**） | 校准后 **Δ = +0.0000** |
| 零信息场 · 面积悬殊 36 vs 324 / 等面积 | 校准后 **+0.0000 / +0.0000** |
| 真信号场（各自命中自己的 GT） | 校准后 **Δ = +0.6074** ﹥ 0.2 |
| n=2 pin | 校准后 **p = 1.000**（两单位 sign-flip 不可能显著，诚实） |

`§17.2` 与 `PENDING:200` 均改为「**报告列，非 gate**」并写明「`config.GATES` 恒为十行，
不含任何 `instruction_paired_*` 项」＋为何现在不能升为 gate；实测 GATES **10 行、无配对项** —— F-B2 清零。
N27（PENDING「十项 gate」）、N28（三条控制各出具名 drop 列 + `min_negative_control_drop` 取**最小**＝
「必须在三条上全部掉分才算跟随指令」的保守语义）、N30（删 `dim/light`、`warm/cool`、`rich/dull` 三对多义词，
**代价如实记账**：指令覆盖 98.5%→97.25%、temperature 轴 341→216、`<where>` 1.0%→0.5%；
本审阅实测 389/400、temp 216、2/400 与账面一致，对合 800 条仍 0 违例）、
N32（新增 `where_text_has_flippable` / `where_flippable_terms` 逐样本标记内部不自洽行）**全部 PASS**。
套件：`q3vl/whereb` **330 passed**；`q3vl/where + q3vl/tests + q3vl/what` **456 passed, 2 skipped**。

> 第 4 轮：BLOCKER 2（F-B1 / F-B2）—— **本轮已修复并独立验证**。
> 第 3 轮：CONDITIONAL，BLOCKER 1（A5-B1）—— 已关闭。第 2 轮：APPROVED。第 1 轮：BLOCKER 5。
> 五轮内容依次保留在下文，未做删改。

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-05 |
| 审阅类型 | 独立实现审阅（只读）。以协议为唯一规格来源，不信任 NOTES 的自述结论，逐条复现。 |
| 规格权威 | `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §1.1/§2.3/§3/§4.2/§5/§10.3/§11/§14 |
| 审阅对象 | `q3vl/whereb/`（16 模块 + 3 脚本 + 1 shell + 17 测试文件）、`experiments/Q3VL_metacanvas_where_what_20260804/where_b/`（NOTES.md、PREFLIGHT_WHERE_B_PENDING.md、mock_closed_loop.json、preflight_where_b_cpu.json） |
| 本次审阅修改的文件 | 仅本文件。未改动任何代码/数据/配置，未占用 GPU。 |
| 结论 | **5 BLOCKER / 13 NIT**。BLOCKER 全部可复现，均给出精确 file:line 与规格条文。 |

---

# 第 6 轮 · 无人值守加固复审（D-B17 / eval 防御 / D-B18 / EXEC-3，2026-08-10）

| 字段 | 值 |
|---|---|
| 审阅对象 | `d32ec60`（11 文件 +927/−25）、`244f770`（W1_provenance） |
| 方式 | 只读、CPU；**W01/W02 两卡训练全程未扰**；对 NFS 只读 `/mnt/nfs-ro`（soft），对 `/mnt/nfs` **零 IO**（关键证明用纯字符串分析完成） |
| 套件 | `q3vl/whereb` **382 passed** ✅；`q3vl/where + q3vl/tests + q3vl/what + q3vl/train` 527 passed / **5 failed** —— 已核实为**环境性**（见 6.6），非本次回归 |
| 结论 | **BLOCKER 1（W-B1）+ NIT 6**；其余四项全部 PASS |
| 审阅者操作披露 | 为判定 5 个失败是否为回归，我执行过一次 `git stash` / `git stash pop`（对比 `d32ec60~1`）。**工作树已完整还原**，与本会话开始时的 `git status` 快照一致。 |

## 6.1 BLOCKER W-B1 · D-B17 漏掉了**读取量最大的那条路径**，而断言会为它打绿灯

**结论**：`records` 与 `images` 的读取——即训练循环里 **100% 的流式 IO**——**仍然走 hard 挂载 `/mnt/nfs`**。
D-B17 实际改到 ro 的只有：索引文件本身（启动时读一次，几 MB）、以及 oracle/maskview/genwhere 三类已发布派生物。

**机制（三段，缺一不可）**：

1. 冻结 split 索引里 `shard` 字段记的是**绝对路径**。实测 `train.index.jsonl` 首行：
   ```
   image  shard=/mnt/nfs/bc/data/datasets/sft2seg-20260804/images/shards/shard-00005.tar
   record shard=/mnt/nfs/bc/data/datasets/sft2seg-20260804/records/shards/shard-00000.tar
   ```
2. `q3vl/train/shards.py:312-314` —— `path = Path(shard)`；**只有相对路径才拼 `shard_root`**：
   ```python
   path = Path(shard)
   if not path.is_absolute():
       path = self.shard_root / shard
   ```
3. `q3vl/whereb/data.py:156` —— `self.store = store or ShardStore("/", verify=verify)`，
   本来就**依赖**索引里的绝对路径。

**全量实测**（纯字符串复算 `_fd` 的解析逻辑，对 `/mnt/nfs` 零 IO）：

```
SPLIT_DIR (已改到 ro) = /mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/splits
train  : 159215 samples   image -> /mnt/nfs x159215   record -> /mnt/nfs x159215
V_where:    896 samples   image -> /mnt/nfs x896      record -> /mnt/nfs x896
```

**同一病灶的第二处**：`q3vl/where/maskdata.py:89` 的 `MaskResolver._store = ShardStore("/")`，
而 `root` 取自 `record["image"]["origin"]["root"]`（**烘进数据里**的 build 路径）。
于是 `open_dataset` 的 live-resolver 回退档（含 `catalog.sqlite3` 的打开）也留在 hard 挂载上。

**为什么这是 blocker 而不是 nit**：

- **敞口是全量且持续的**。每臂 1 epoch = 159,215 条 record + 159,215 张图；
  再加七块板 × 896 × 约 10 次 eval。NOTES §16.5 自己实测「瓶颈是 NFS 读不是 GPU」
  （W01 GPU util 仅 45%）——正说明这条路径**全程热**。D-B17 要消除的正是这个敞口。
- **断言会为它打绿灯**。`run_where_b.py:190` 调的是
  `assert_read_mount(SPLIT_DIR, WHERE_A_ORACLE_DIR, GENCTX_DIR)` —— 三条**都是已改到 ro 的**路径，
  必然全 ok，并写进 `run_setup.json` 的 `read_mount`。读的人会据此认为「本臂经 nfs-ro 读数据」。
- **NOTES §16.1 的读写分流表把它列为已覆盖**：
  「读｜`data.BUILD_ROOT` / `data.DATASET_ROOT` / `where.SFT2SEG_ROOT`(→`SPLIT_DIR`) / `where.BUILD_DATASET_ROOT`｜**`/mnt/nfs-ro`**」。
  对**索引文件**成立，对**字节**不成立。文档与实际不符，且没有任何地方标注这个例外。
- 这与本战役已栽过三次的失败类同构（第 1 轮 B1「preflight 报 PASS 但没跑」、
  第 3 轮 A5-B1「常量宣布了控制但产不出」、第 4 轮 F-B2「文档宣称 gate 代码没有」）：
  **保护措施存在、指示灯是绿的、被保护的东西没被保护**。

**公平记账 —— 这次改造真正生效的部分**：`PublishedStore.read` → `read_member(root, shard, …)`
（`q3vl/data/shardio.py:189`）是 `Path(root)/"shards"/f"{shard}.tar"`，**root 拼在前面**，
所以 oracle / maskview / genwhere 三类读取**确实**改到了 nfs-ro ✅。只是它们不是大头。

**建议修法**（(a) 最小、可测、一处收口）：

- **(a)** 给 `ShardStore` 加一个绝对路径前缀重写（`{"/mnt/nfs": "/mnt/nfs-ro"}`），
  由单一常量驱动，构造 `WhereBDataset` / `MaskResolver` 时传入；约 10 行 + 一条单测
  （断言解析结果落在 `NFS_RO_ROOT` 下）。
- (b) 重新发布索引，改记相对 shard 名 + `shard_root`——干净但要重跑发布。
- (c) bind-mount 覆盖——运维手段，不进代码，但无法被 `assert_read_mount` 证明。

**在修好之前**：要么修，要么把 §16.1 的表和 `run_setup.json` 的 `read_mount` 明确降级为
「仅覆盖索引与已发布派生物；records/images 仍在 hard 挂载」，别让 W03–W08 在错误前提下启动。

## 6.2 `assert_read_mount` 本身 —— PASS（函数正确，用错了参数）

| 要求 | 实现 | 判定 |
|---|---|---|
| 挂载存在 | 读 `/proc/mounts`，本地无 IO | PASS |
| **拒绝 hard 读挂载** | 选项里必须含 `soft`，否则 `ReadMountError` —— 这一条是本函数最有价值的部分：hard 的读挂载等于没改 | PASS |
| **深路径探活** | 对每个入参 `os.stat`，注释写明「挂载点自身由 dentry/属性缓存应答，探它等于没探」 | PASS |
| **不可能拖住调用方** | `threading.Thread(daemon=True)` + `join(timeout_s)`；`is_alive()` 则**丢弃线程**并抛错。这是对付 hard-mount D 状态的**唯一**正确姿势（`timeout` 救不了，因为要 `wait()`） | PASS |

nit：`if "soft" not in opts` 是对**整行**做子串判断（设备名或路径里含 "soft" 就会误过），
应取第 4 字段再按 `,` 切分后精确比对 → **N33**。

## 6.3 写路径确实未动 —— PASS

`WHERE_A_ROOT` / `MASKVIEW_DIR` / `ORACLE_DIR` / `BASIS_DIR`、`data.OUT_ROOT`、
`WHERE_B_ROOT` / `GENCTX_WRITE_DIR` 全部仍指 `/mnt/nfs`；
`make_generated_context.py` 的 `--out-root` 默认值已从 `GENCTX_DIR` 换成 `GENCTX_WRITE_DIR`
（消费侧读 ro、生产侧写 rw，两条常量分开）。**读写分流的设计本身是对的**——
问题只在 6.1 那条没被纳入分流。

## 6.4 eval 防御 —— PASS（两个 nit）

| 要求 | 实现 | 判定 |
|---|---|---|
| 异常不杀训练 | `_eval_and_record` 捕获 `BaseException` → 写状态文件 → `return`，主循环继续 | PASS |
| `KeyboardInterrupt` / `SystemExit` 例外 | 先 `isinstance` 判定后 `raise`；`finally: self.model.train()` 在重抛时**照常执行** | PASS |
| 不静默 | ① `EVAL_FAILED_step<N>.json`（含完整 traceback + 「TRAINING CONTINUED / 不可选型」说明）；② 日志大写 `!!! EVAL FAILED … !!!`；③ `state.eval_failures` | PASS |
| 「无 eval 记录不可选型」 | **结构性成立**：失败路径 `return` 前**从不** append 到 `state.checkpoints`，而 `best()` 只在 `state.checkpoints` 上取 max ⇒ 未测量的 checkpoint 根本进不了候选集。这是论证而非承诺 | PASS |

- **N34**：`state.eval_failures` **没有进交付物**。`run_where_b.py` 最后写的 `arm_<ARM>.json` 只含
  `setup / final / best_recorded / n_steps / format_stats`；`EVAL_FAILED_*.json` 只落在 `run_dir`。
  只读交付物的选型脚本看不出「这臂有几个 checkpoint 没被测量」。建议把 `eval_failures` 加进去。
- **N35**：**最后一次 eval 不在防护内**。`run_where_b.py:263` 的
  `final = evaluate_arm(...)` 在 `trainer.train()` **之后、trainer 之外**调用，没有 try/except。
  它抛错会在训练已经跑完 10 h 之后杀掉进程，且 `arm_<ARM>.json` 一个字都写不出来——
  正是这次防御要消除的那种损失。建议同样包起来（失败则写 `EVAL_FAILED_final.json` 并仍落 `arm_*.json`）。

## 6.5 D-B18 PID 口径 —— PASS（两个 nit）

- **`pgrep -P <wrapper> -n` 合规**：CLAUDE.md 禁的是 **`pgrep -f <pattern>`**（模式会匹配到正在
  grep 的那条 shell 自己）。`-P` 按**父 PID** 匹配，调用方 shell 不是 `$wrapper_pid` 的子进程，
  **结构上不可能自匹配**。脚本注释也明确写了「是 `-P` 不是 `-f`」。**无 `-f` 风险。**
- **`python_pid` 权威口径**：`job.marker` 现记 `python_pid`（权威）+ `wrapper_pid`（仅溯源）
  + `liveness_check=ps -p <python_pid>`，并保留 `pid=` 兼容旧读法。解析失败**回退 wrapper 并打 WARNING**，
  不静默把 wrapper 当作业。方向正确（此前是「wrapper 先退出 ⇒ 误报已死 ⇒ 监控去重启一个在跑的作业」）。
- **N36**：这次把原来的 `ps -p "$pid" -o pid,etime,cmd --no-headers` **删掉了**——
  那正是第 2 轮 N7 加上去、用来「PID 错了看得见」的那一行。现在没有任何地方打印被解析出的 PID 的命令行。
- **N37**：解析假定 `setsid` 只 fork 一层。若某版 `setsid` 不 fork，`pgrep -P wrapper` 为空 → WARNING + 回退（安全）；
  若 fork 两层，`$pid` 会是个瞬时中间进程，step 3 的 `ps -p "$pid"` 会对一个**健康**作业报 "died" 并 `return 1`。
  廉价护栏：解析后断言该 PID 的 `cmdline` 含 `run_where_b`（一并解决 N36）。

## 6.6 EXEC-3 oracle namespace 合并 —— PASS

| 项 | 复核 |
|---|---|
| `ORACLE_NAMESPACE = "s5"` + `cfg` 绑定 | **正确**。`make()` 闭包引用 `cfg.readout`，`cfg` 在 `make` **被调用前**赋值（定义在前、赋值在两次调用之前），Python 闭包按调用时解析 ⇒ 成立 |
| 静默错误的论证 | **成立且非平凡**：`<arm>/train` 不存在会**响亮**失败，而 `<arm>/V_where` **存在**且缺 `curve`/`cband_normalization` ⇒ 会**悄悄**用另一次 run 的 latent 当 eval 监督。两个 split 只有一个报错，正是最难发现的形态 |
| `assert_oracle_contract` | **正确，且正是 CLAUDE.md「s 缓存消费契约」要求的消费侧断言**：显式声明期望的 `cband_normalization` 与 z 网格（`CURVE_Z_LO/HI/N`），断言生数据确实住在里面；缺字段/域不符/网格不符全部 `SystemExit`；同时以「读得出 payload」证明 manifest complete ≠ 命名空间对 |
| `--oracle-root` 默认取 `WHERE_A_ORACLE_DIR` | 与 D-B17 正确叠加：该常量现指向 ro 镜像，而 `PublishedStore` 是 `root/shards/<shard>.tar` 拼接式读取 ⇒ **这条读路径确实走 nfs-ro** ✅ |
| `run_where_b.sh` 转发 `"${@:4}"` | 正确；注释记录了真实约束（一个 wave 两臂必须同 `--micro-batch`，否则 `len(sampler)`→`total_optimizer_steps`→LR schedule 全变，不再是 §11 的配对比较） |

**W1_provenance 溯源闭环 —— PASS**：`worktree_diff_committed_as = d32ec60` 指向真正含 fix 的 commit，
配合 `worktree_diff` 补丁 + `worktree_diff_sha256` + `worktree_dirty: true` + `why_dirty` 说明，
`run_setup.json.env.git_commit=0695e0a` 不描述运行代码这个缺口**已闭合**。
`oracle_namespace` 字段记的是 `/mnt/nfs/...`（W01/W02 实跑口径），与下一条注记一致，未粉饰。

**明知风险的记录完整性 —— PASS（有一处不足）**：`worktree_diff_committed_note` 明写
「W01/W02 still ran from the working tree … It also moves data READS to /mnt/nfs-ro (D-B17), **which W01/W02 do NOT have**」——
W01/W02 的敞口记录完整、诚实。**不足之处**是 6.1：文档同时让人以为 **W03–W08 已经被保护**，而主数据流并没有。

**关于 5 个测试失败（不是回归）**：`q3vl/train/tests/test_train_pipeline.py::TestFrozenHyperparameters`
的 5 条报 `"Your setup doesn't support bf16/gpu."` —— 我是以 `CUDA_VISIBLE_DEVICES=""` 跑的（避让两卡训练），
HF `TrainingArguments` 在冻结超参校验之前就先拒了 bf16。在 `d32ec60~1` 上**同样 5 条失败**，
故**与本次改动无关**；commit message 里的「534 passed」是有卡可见时的口径。

## 6.7 第 6 轮 NIT

| # | 内容 |
|---|---|
| N33 | `assert_read_mount` 的 `soft` 检查是整行子串判断，应取 `/proc/mounts` 第 4 字段按 `,` 精确比对 |
| N34 | `state.eval_failures` 未写进交付物 `arm_<ARM>.json`，只读交付物看不出有几个 checkpoint 未被测量 |
| N35 | `run_where_b.py:263` 的**最终** `evaluate_arm` 在 trainer 防护之外，抛错会在训练跑完后杀掉进程并丢掉 `arm_*.json` |
| N36 | `submit` 删掉了打印被解析 PID 命令行的那行（第 2 轮 N7 的产物），PID 错了不再「看得见」 |
| N37 | PID 解析假定 `setsid` 恰好 fork 一层；两层时 step 3 会对健康作业误报 died。护栏：断言 cmdline 含 `run_where_b` |
| N38 | `assert_oracle_contract` 每次启动读 8 条 payload；train split 上这是 8 次 ro 随机读，可接受，记录以备排期 |

## 6.8 第 6 轮判决

**BLOCKER 1（W-B1）+ NIT 6。**

- **判据侧（第 5 轮结论）不受影响**：本轮改动没有触碰 gate、loss、判据或评测口径。S6 选型与 REPORT 的解锁继续有效。
- **W03–W08 启动前置**：W-B1 必须先处置——要么按 (a) 加绝对路径前缀重写（约 10 行 + 一条单测），
  要么把 §16.1 的读写分流表与 `run_setup.json.read_mount` 明确降级说明。
  **不能让下一波在「已受 nfs-ro 保护」的错误前提下无人值守跑 6 × 10 h。**
- eval 防御、D-B18、EXEC-3 合并、W1 溯源四项**全部 PASS**，其中 `assert_oracle_contract`
  与 `assert_read_mount` 的 daemon-thread join deadline 是两处质量明显高于要求的实现。

**一句话**：这次加固的**方向和手艺都对**（拒绝式断言、深路径探活、不可能挂死的探测线程、
消费侧契约断言、PID 权威口径），唯独漏掉了最大的那条读路径——而且漏得**看不出来**，
因为断言只探被改过的三条。这正是本战役反复出现的那一类：指示灯绿着，被保护的东西没被保护。

---

# 第 4 轮 · 合并终审（A5-B1 关闭 + antonym 控制，2026-08-05 夜）

| 字段 | 值 |
|---|---|
| 审阅对象 | `779c7c5`（A5-B1 路线 (a) + 5 个 nit，12 文件 +584/−55）、`ea14653`（antonym 不变性控制，11 文件 +573/−33） |
| 方式 | 只读、CPU；**未碰 GPU0（Where-A 四臂）/ GPU1（forced_color genctx）**；六项关键结论全部**独立复算**，不采信自述 |
| 套件 | `q3vl/whereb` **315 passed**（295→315）；`q3vl/where + q3vl/tests + q3vl/what` **456 passed, 2 skipped** —— 与自述一致，无回归 |
| 结论 | A5-B1 **CLEARED**；antonym 控制 **PASS**；nit 处置 **全部 PASS**；**新发现 BLOCKER 2 个** |

## 4.1 A5-B1 · 三条负控制成为一等模式且真实可产出 —— **CLEARED**

| 复审点 | 结论 |
|---|---|
| 一等 `CONTEXT_MODES` | `context.py:52-53`：`(gt, generated, null, shuffled, irrelevant_words, fixed_phrase, antonym)` **七种**；`INSTRUCTION_OVERRIDE_MODES` 扩到四种，`__post_init__` 仍拒绝其余模式覆盖指令 | 
| `BatchBuilder` 分支 | `data.py:379-390` 三个新分支齐备 |
| `fixed_phrase` 选词 | 用红线**自己的反例** `"the main subject"`（曾拿 AUC 0.907 而 `AUC_target` 0.523）。用已知会骗人的那句做控制，是正确取材 |
| `irrelevant_words` | 固定词表 18 个无空间/摄影语义的具体名词 + **逐样本种子**（`Random(f"{seed}:{sample_id}")`）⇒ 同样本可复现、跨样本不同 |
| 三条控制同时换 instruction 与 `<where>` | 与 D-B15 一致（留着真实指令＝正确答案仍可达） |

**`test_batch_builder_has_a_branch_for_every_context_mode` 单独跑通**，其构造是**源码扫描**
（`f"== {const}" in src`），且**常量名从 `context` 模块反查而非硬编码**——加第七种模式时不会静默失效
（ea14653 的 commit message 说本次就是它先报的警，可信）。但源码扫描证明不了「分支能跑出东西」，
故本审阅补了**活体验证**：用 `FakeTokenizer` 直接过 `BatchBuilder.context_for` 打七种模式——

```
CONTEXT_MODES (7): ('gt','generated','null','shuffled','irrelevant_words','fixed_phrase','antonym')
  gt               ntok=5   instr_override=False  (sample's own)
  generated        ntok=3   instr_override=False  (sample's own)
  null             ntok=0   instr_override=False  (sample's own)
  shuffled         ntok=5   instr_override=True   'Cool the church and desaturate it.'
  irrelevant_words ntok=14  instr_override=True   'lantern thimble trombone granite envelope notebo…'
  fixed_phrase     ntok=5   instr_override=True   'the main subject'
  antonym          ntok=5   instr_override=True   'Make the jellyfish brighter and cooler.'
unknown mode -> rejected: unknown context mode 'made_up'
```

七种**全部真实产出**，未知模式仍被拒。第 3 轮 A5-B1「命名了但产不出」的病灶已消除。

### 交叉打分与 sign-flip 置换

`evaluate._instruction_paired`（`evaluate.py:144-176`）的构造**正确**：
`k` 取自样本自身 GT，`pred_bin` 只二值化**一次**，再分别对自 GT 与伙伴 GT 算 `hard_iou`——
比较的是**同一个场**在两张 GT 上的得分，不是两个场。三条跳过规则（无伙伴 / 几何不同 / 目标区域相同）齐备。

`paired_delta` 改 sign-flip 置换 + 加一修正，**实测**：全正差分下 `p = 9.999e-05` 恰等于 `1/(n_perm+1)`，
`p > 0` 成立 —— N23 要的「永不报 p=0」达成，且 `test` 字段记录了检验名。

### 「相反指令对 n=2 不可构造」的独立抽验

我用自己的谓词重跑 `V_where` 本地 400 条：

| 量 | §17.2 自述 | 本审阅实测 | 判定 |
|---|---:|---:|---|
| 同图不同指令的样本对 | 712 | **712** | **逐数一致** |
| 96 个多样本组是否全部含 ≥2 个不同 mask | 是 | **96/96** | **逐数一致** |
| 语义相反 | 147 | 676（我的谓词更松） | 谓词不同，非矛盾 |
| 同主体 + 方向相反 + 同 mask | 2 | 8（同上） | 谓词不同，非矛盾 |
| 不同主体 + 同 mask | 1 | 0 | 谓词不同 |

**承载结论的两项逐数复现**。而且方向一致：即便用我**明显更松**的谓词，字面对照也只有 8 对 / 400 条，
统计上依然不可用。⇒ **「字面对照在现有语料里不存在」这个结论独立成立**。
但 147 / 2 / 1 三个数依赖未写明的匹配谓词，别人复现不出来 → **N29**。

## 4.2 antonym 不变性控制 —— **PASS（六项全部独立复算）**

| 复审点 | 实测 | 判定 |
|---|---|---|
| **同步替换** | `warm the warmth` → `cool the coolness`（`warm` 未被 `cool` 规则回翻） | PASS |
| **最长优先** | `desaturated` → `saturated`（未被内含的 `saturated` 规则腐蚀）；`a saturated red` → `a desaturated red` | PASS |
| **词边界** | `unsaturated tones` → **原样不动** | PASS |
| **大小写保留** | `Brighter`→`Darker`、`DARKER`→`BRIGHTER` | PASS |
| **对合** | 真实语料 400 条 instruction + 400 条 `<where>`：**违例 0** | PASS |
| **一一映射** | `len(ANTONYM_MAP)=44 == 2×22`，模块导入期即断言 | PASS |
| **主体短语不变** | `antonym_context` 只翻 `instruction`，`<where>` ids 来自样本自身 `where_text`，**结构性不变**；单测 `test_antonym_context_flips_the_instruction_and_keeps_the_where_text` 另作断言 | PASS |
| **median \|Δ\| 口径 pin** | `antonym_invariance` 逐样本 join `gt`/`antonym` 两块板取 `|Δ|` 中位数；单测 `test_invariance_is_paired_per_sample_not_a_difference_of_medians` 专门排除「两个中位数相减」的错误口径（构造下正确答案 0.30、错误口径≈0） | PASS |
| **非 gate 定位** | `GATES` 实测 **10 行**，无 antonym 项；返回体带 `"note": "negative-control column, not a gate"`；`ANTONYM_INVARIANCE_MAX=0.05` 注释写明理由（不变性做硬 gate 会把 tie-break 罚得和真读颜色一样重） | PASS |
| **语料测量 98.5% / 1.0%** | 本审阅实测：instruction **394/400 = 98.5%**（luma 352 / temp 341 / sat 253）、`<where>` **4/400 = 1.0%**（全在 luma 轴） | **逐数一致** |
| **词表非生成** | 固定常量 + `table_digest()` sha256，`control_detail` 逐样本记 `flipped_terms` / `n_flipped` / digest | PASS |

词表本身覆盖三轴、每词只属一对。**这一节没有找到问题**——构造、口径、定位、数据全部对得上。

## 4.3 新发现 BLOCKER

### F-B1 · 方向性配对 Δ 存在**面积混杂**，零信息场可拿到 +0.76

`_instruction_paired` 的 docstring 与 §17.2 都写着「图像被固定，因此**图像显著性与中心先验成对抵消**」。
该断言**只在两张 GT 面积相当时成立**。机制：`k = gt_area_k(GT_A)` 把预测二值化成 `|GT_A|` 格，于是

```
IoU(pred_k, GT_B) <= |GT_A| / |GT_B|
```

当 `|GT_B| > |GT_A|` 时 `cross_iou` 被**机械压低**，`self_iou − cross_iou` 因此天然为正。
跳过规则只排除「几何不同」与「目标区域相同」，**不排除面积悬殊**。

**对抗性实测**（把「被测场」换成零参数中心先验，同图两张 GT）：

| 构造 | \|GT_A\| | \|GT_B\| | Δ | p |
|---|---:|---:|---:|---:|
| 等面积、位置不同 | 100 | 100 | **+0.0000** | 1.000 |
| 面积悬殊 | 36 | 324 | **+0.1343** | 1.000 |
| **同心**（圆心相同，只差面积） | 36 | 400 | **+0.7590** | 0.505 |

第三行是决定性的：两张 GT **中心重合**，位置信息完全相同，Δ 却是 **+0.7590**——
这整个数字都是面积效应，而产生它的场**不含任何指令信息**。
等面积那一行 Δ = 0.0000 说明「抵消」的机制本身是对的，**条件没写全**。

实测语料上 96/96 多样本组的成员 mask 各不相同，面积自然有差异，**该混杂在真实板上会生效**。
红线设立中心先验列的初衷正是「零参数基线不该赢」，这里零参数基线可以赢 0.76。

**建议修法（任一即可，都很便宜）**：
(a) **补一行中心先验校准**——用 `center_prior_field` 当场跑同一套 `_instruction_paired`，
把 `instruction_paired_center_prior` 与被测场并列输出（约 10 行，字段现成）；这最贴红线的既有做法。
(b) 配对时限制面积比（如 `0.5 <= |GT_A|/|GT_B| <= 2`），或改用面积归一化统计量。
**在修好之前，这一列不得作为「场跟随指令」的证据写进 REPORT。**

### F-B2 · §17.2 与 PENDING 把方向性配对 Δ 写成 gate，`config.GATES` 里没有它

`ea14653` 更新的两处文档：

- 协议 §17.2 的裁定表：`| **方向性配对 Δ**（主判据） | … | **Δ > 0**，sign-flip p | gate |`
- `PREFLIGHT_WHERE_B_PENDING.md:200`：`| 方向性配对 Δ | … | **主判据 / gate** |`

而实测 `config.GATES` **10 行**，逐行列出后**没有任何 `instruction_paired_*` 项**：

```
local_soft_iou_median / soft_iou_vs_oracle_ratio / local_soft_iou_p10 /
grid_boundary_f1_vs_oracle_ratio / center_prior_delta_hard_iou /
center_prior_delta_hard_iou_p / instruction_shuffle_iou_drop /
s_std_ratio_median / global_soft_iou / gt_generated_iou_gap
```

`instruction_shuffle_iou_drop` 是**另一个量**（generated 板与 shuffled 板的中位 IoU 之差），
不是同图配对 Δ。所以文档宣布的 gate 在代码里不存在。

**这与 A5-B1 是同一失败类**，且正是在关闭 A5-B1 的那次提交里被重新引入的——
上一轮的病灶是「常量宣布了控制、代码产不出」，这一轮是「文档宣布了 gate、代码不 gate」。

**建议**：先修 F-B1，再决定是否真的加这条 gate（**现在不能加**：零信息场能拿 +0.76，
加了等于给它开门）。若决定不加，把两处文档的「gate / 主判据」改为「报告列」，并说明理由。

## 4.4 nit 处置复核

| nit | 处置 | 判定 |
|---|---|---|
| **N23** p 值 | 改 sign-flip 置换 + 加一修正；CI 仍用 percentile bootstrap。实测 `p = 1/(n_perm+1)`，永不为 0 | **PASS** |
| **N24** viz `valid=None` | `color_scale` 新增 `allow_all_valid`，默认 `valid=None` **直接抛 `PerImageMinMaxError`**，异常文案写明「mode 被守住而 valid 没有，禁令就是一个参数之遥」。与 `mode` 完全对称 | **PASS** |
| **N25** 盲区进模板 | `ATTRIBUTION_NOTE["known_blind_spots"]` 两条 + `attribution_section` 渲染中文段落，数字（0.0000 vs 0.0357）与我实测一致，且点明「噪声在这一列上反而赢了位置错的紧凑场」 | **PASS** |
| **N20** 模块 docstring | `metrics.py:1-30` 重写为 A-5 后的**十行**表，并写明 AUC 已删、阈值化统一 top-k、指令条件性用配对差分 + 三条负控制 | **PASS** |
| **N21 / N22** 记录更正 | 两条都在 commit message 里明确更正（`evaluate_gates` 旧行为是走 `<=` 分支＝判据反转；random 是 0.0357/0.0050 不是 0.0000） | **PASS** |

## 4.5 七块上下文板的评测输出结构

`evaluate_arm(contexts=CONTEXT_MODES)` 默认跑**七块板**，每块一份 `summarise`（含 `instruction_paired`），
`per_context` 保留全部七块，`arm_metrics` 只从 `generated` 板读 gate 指标、拒绝求平均；
`strata` 逐板出四个分层；`antonym_invariance` 跨 `gt`/`antonym` 两块板逐样本 join；
`per_sample.jsonl` 每行带 `context` / `instruction_swapped` / `control_detail`；
`ATTRIBUTION.md` 随每份 `metrics.json` 落盘。**结构完整，判定 PASS。**

两点提醒：

- `context_deltas`（`metrics.py:571-585`）只派生 `null_context_gap` 与 `instruction_shuffle_iou_drop`，
  **`irrelevant_words` / `fixed_phrase` 有整块板却没有对应的具名 delta 列** → **N28**。
  红线的用法是「在 `fixed_phrase` 下与真实指令得分相同 ⇒ 读的是显著性」，这个比较应当是**算出来的列**，
  不该留给报告作者手算。
- **排期**：七块板 × 896 = 6,272 次编码/评测，每 500 步一次、约 10 次/臂 ≈ 62,720 次，
  相对 159,215 条训练前向是 **~39% 墙钟开销**（四块板时是 22.5%）。八臂四波，请把这部分单独计入 → **N31**。

## 4.6 §17.2 / NOTES / PENDING 终态一致性

| 检查 | 结论 |
|---|---|
| 开放项确已关闭 | **是**。§17.2 原来的「须主 agent 裁定」blockquote 已整段替换为「已裁定，开放项关闭」+ 两类控制对照表；PENDING S5.5 标题改为「**已完成，开放项已关闭**」 |
| 关闭理由是否如实 | **基本如实**：「字面对照在语料里不存在」有数据支撑（我复现了承载项），「antonym 以文本变换补上这一侧且不需要新数据产物」属实，「两类控制方向相反、各司其职」的论证正确 |
| 不如实之处 | **F-B2**（把配对 Δ 写成 gate）与 **N29**（147/2/1 的谓词未写明） |
| 遗留陈旧 | PENDING S6 首行仍写「§5.6 **九项** gate 的预注册判据 vs 实测数字并排表」——A-5 之后是**十项** → **N27** |

## 4.7 第 4 轮 NIT

| # | 内容 |
|---|---|
| N27 | `PREFLIGHT_WHERE_B_PENDING.md` S6 首行仍写「九项 gate」，A-5 之后是十项 |
| N28 | `context_deltas` 未给 `irrelevant_words` / `fixed_phrase` 派生具名 delta 列（两块板都跑了却要报告作者手算） |
| N29 | §17.2 / NOTES §12.1 的「语义相反 147 对 / 同主体同 mask 2 对 / 不同主体同 mask 1 对」依赖未写明的匹配谓词；我用更松的谓词得 676 / 8 / 0。承载结论不受影响，但数字不可复现 |
| N30 | 反义词表含多义词：`("dim","light")`、`("rich","dull")`、`("warm","cool")`。实测 `lighting a light` → `lighting a dim`（语法坏掉）。控制是**报告列**且失败方向才是信息，所以风险是「因文本变坏而假失败」；`control_detail.flipped_terms` 已给逐样本溯源，但**那 4 条 `<where>` 自身含可翻转词的样本没被标记**（此时指令翻了而主体描述没翻，配对内部不自洽） |
| N31 | 七块板使评测墙钟从训练前向的 22.5% 升到 ~39%，八臂四波请单独计入排期 |
| N32 | `control_detail["where_text_unchanged"] = True` 是硬编码字面量而非计算结果（结构上确实不变，属表述问题） |

## 4.8 第 4 轮判决

**A5-B1：CLEARED**（活体验证七模式全部可产出、交叉打分构造正确、sign-flip 加一实测、n=2 结论承载项逐数复现）。
**antonym 控制：PASS**（对合/最长优先/词边界大小写/主体不变/median 口径/非 gate/98.5%-1.0% 全部独立复算通过）。
**N20-N25 五项处置：全部 PASS。**

**新增 BLOCKER 2 个**，都在**报告列**而非 gate 上：

- **F-B1**：方向性配对 Δ 有面积混杂，零信息场同心构造下拿 +0.7590；
- **F-B2**：§17.2 与 PENDING 宣称它是 gate，`config.GATES` 十行里没有它。

**放行**：

| 阶段 | 判决 | 理由 |
|---|---|---|
| S3 preflight / S5 八臂训练 | **准许**（维持） | 本轮改动全在评测侧；§5.5 loss 仍一字未动（`losses.py` 两次提交均未被 diff 触及） |
| **S6 选型** | **解锁** | 十条 gate 逐条复核为面积稳健：中心先验双 gate 用**同一个 `k`、同一张 GT**，构造上就没有 F-B1 的混杂；lexicographic 四键也不读配对 Δ。**选出哪个 checkpoint 不受这两个 blocker 影响** |
| **S6 REPORT** | **不解锁** | F-B1 未修前，配对 Δ 那一列不得作为「跟随指令」的证据；F-B2 未修前，文档与代码对 gate 集合的陈述不一致 |

**一句话给主 agent**：这一轮的实现质量很高——A5-B1 是真关闭而不是补文档，antonym 控制的六个细节我逐个复算全对，
98.5%/1.0% 和 712/96 两组数逐数一致。两个新 blocker 都不难修（F-B1 加十行中心先验校准，F-B2 改两处文档措辞），
但**都必须在 REPORT 落笔前修**：F-B1 是「零参数基线能赢」这一类问题，正是这轮红线要根除的；
F-B2 是「文档宣称代码没做的检查」，本战役已经栽过两次（第 1 轮 B1、第 3 轮 A5-B1）。

---

# 第 3 轮 · amendment A-5 聚焦复审（2026-08-05 晚）

| 字段 | 值 |
|---|---|
| 触发 | 用户 2026-08-05 固化两条新红线：`CLAUDE.md`「AUC 全实验禁用」（L103-124）、「空间场可视化纪律」（L126-132），并入红线速查 L136 |
| 审阅对象 | commit `2c94347`（13 文件，+1114/−112）：`config.py` / `metrics.py` / `data.py` / `evaluate.py` / 新增 `viz.py` / 新增 `test_a5_criteria.py`+`test_viz.py` / 协议 §17.2 + §5.6 脚注 / NOTES §11 |
| 审阅范围 | 只审 A-5 diff；第 2 轮已 APPROVED 的部分不重审 |
| 方式 | 只读、CPU；**未碰 GPU0（Where-A 四臂）/ GPU1（forced_color genctx）**；判别力表由本审阅**独立重算**，不采信提交表格 |
| 结论 | **1 BLOCKER + 7 NIT**。红线的**空间场判据侧**落实到位且质量很高；缺口在红线的**指令条件性侧** |

## 3.0 复现的基线

| 项 | 结果 |
|---|---|
| `pytest q3vl/whereb/tests -q` | **279 passed / 36.5 s** ✅ 与 commit message 一致（268 → 279） |
| `pytest q3vl/where/tests q3vl/tests q3vl/what -q` | **456 passed, 2 skipped / 4:49** ✅ 无回归 |
| `git show --stat 2c94347` | `q3vl/whereb/losses.py` **不在改动清单里** —— 这是「loss 一字未动」最硬的证据 |

## 3.1 §5.6 gate 修订与红线逐条对照

| 红线条文 | A-5 实现 | 判定 |
|---|---|---|
| 「任何形式的 ROC-AUC 一律不得作为空间场的判据」「**新实验不得再产出该指标**」 | `GATES` 删行；`metrics.auc_target()` **整个函数删除**（不是留着不用）；`__all__`、`sample_metrics`、`summarise`、`ATTRIBUTION_NOTE`、`attribution_section` 全部清空。`rg -i auc q3vl/` 只剩注释/docstring 里的**说理**与断言其不存在的测试 | **PASS** |
| 「soft-IoU / hard-IoU：覆盖对不对。阈值化用**匹配 GT 面积的 top-k**，禁止逐场调阈值」 | `topk_mask()` + `gt_area_k()`；`TOPK_RULE="match_gt_area"`；pred / 中心先验 / oracle **三者共用同一个 k**（`sample_metrics` 里 `k` 只算一次） | **PASS** |
| 「grid 级边界 F1。**禁用像素级 3px 边界 F1**」 | gate 行改 `grid_boundary_f1_vs_oracle_ratio`；`grid_boundary_f1()` 在 `F_pre` 网格上、容差以**格**计（`GRID_BOUNDARY_TOL_CELLS=1`）；签名里**没有** `tol_px`（有单测钉住） | **PASS** |
| 「中心先验基线列，零参数 `-到画幅中心距离`，与被测场**同支撑、同阈值化规则**」 | `center_prior_field()` = `-sqrt(X²+Y²)`，坐标复用 `q3vl.where.phi.norm_coords`（真实宽高比，与 `geo5` 同一约定）；签名只有 `(grid_h, grid_w, device, dtype)` —— 单测断言它不可能依赖预测或 GT | **PASS** |
| 「任何『我们的场找到了主体』的主张，必须出示**配对 Δ 与 p 值**」 | 双 gate：`center_prior_delta_hard_iou > 0` 与 `center_prior_delta_hard_iou_p <= 0.05`；`paired_delta()` 出 delta / p / CI95，`summarise` 另出 boundary-F1 的同一组三元 | **PASS**（p 值算法见 N23） |
| 「指令条件性另用**配对差分（同图两条相反指令）** + **打乱指令 / 无关词 / 固定短语三条负控制**，不得用任何 AUC 变体代替」 | 只把三个名字写成常量 `INSTRUCTION_NEGATIVE_CONTROLS`；`irrelevant_words` / `fixed_phrase` **无生产者**，配对差分**未实现** | **BLOCKER A5-B1** |
| 字典序第 2 键 | `SELECTION_ORDER[1] == "grid_boundary_f1"` | **PASS** |
| gate 表整体 | 9 → 10 行；`test_the_gate_table_matches_amendment_a5` 用**全表字面比对**（不是逐条 in 判断）钉住，改任何一行都会红 | **PASS** |

### BLOCKER A5-B1 · 协议 §17.2 声明了一套代码产不出来的负控制框架

**红线原文**（`CLAUDE.md:122`）：

> 指令条件性另用配对差分（同图两条相反指令）+ 打乱指令 / 无关词 / 固定短语三条负控制，**不得用任何 AUC 变体代替**。

**协议 §17.2 写的**（`docs/METACANVAS_..._2026-08-04.md:995-998`，现在时、无 pending 标记）：

> **指令条件性**改用配对差分框架，不得用任何 AUC 变体代替：
> 同图相反指令的配对差分 + 三条负控制 `shuffled` / `irrelevant_words` / `fixed_phrase`
> （§5.4 原有的 `shuffled` 保留并并入此框架，gate 行不变）。

**代码实际有的**：

```
q3vl/whereb/config.py:224  INSTRUCTION_NEGATIVE_CONTROLS = ("shuffled", "irrelevant_words", "fixed_phrase")
q3vl/whereb/tests/test_a5_criteria.py:315  assert C.INSTRUCTION_NEGATIVE_CONTROLS == (...)
```

`rg` 全仓确认这两处是**仅有的**出现。同时：

- `context.py:41` `CONTEXT_MODES = (GT, GENERATED, NULL, SHUFFLED)` —— 仍是原来四种；
- `BatchBuilder.context_for`（`data.py:360-394`）没有 `irrelevant_words` / `fixed_phrase` 分支，传进去会走到 `raise ValueError(f"unknown context mode")`；
- 「同图两条相反指令的配对差分」在 `metrics.py` / `evaluate.py` 里**没有任何实现**（`paired_delta` 目前只用于中心先验列）；
- `INSTRUCTION_NEGATIVE_CONTROLS` 被 **0 个**生产代码消费。

**为什么这是 blocker 而不是 nit**：

1. 这是**第 1 轮 B1 的同一失败类**——权威文档断言了一项代码无法执行的检查。B1 当时的表现是
   preflight 打印 PASS 而检查从未构造；这次的表现是 §17.2 宣布负控制框架已就位而两条控制没有生产者。
   唯一的测试 `test_three_negative_controls_are_declared` 断言的是**常量等于它自己的字面值**，
   对「能不能产出」零覆盖——正是 A-5 作者自己批评过的「一个键之遥的默认值不算禁用」的同构问题。
2. 三条负控制是红线给**被删掉的 AUC** 指定的替代物之一。空间场那一半（IoU + grid BF1 + 中心先验）
   做得很扎实，但指令条件性这一半目前只有 1/3（`shuffled`），配对差分 0/1。
3. **没有任何地方标注它是待办**：协议 §17.2、NOTES §11、`PREFLIGHT_WHERE_B_PENDING.md`
   三处 `rg` 均无 pending / TODO / 待实现 字样。NOTES §11.1 第 5 行的措辞「**登记为常量**」
   是诚实的，但它被协议 §17.2 的现在时陈述盖过去了，而结果审阅 agent 读的是协议。

**影响面与排期（请主 agent 据此定优先级）**：三条负控制与配对差分**全部是评测侧**，
和 `null`/`shuffled` 同类，**不进训练目标**。因此：

- **不阻断 S1/S2/S3，也不阻断 S5 的 8 臂训练**；
- **阻断 S6 选型与 REPORT**：一旦用 §5.6 判据板冻结 checkpoint，就必须已经能产出 §17.2 声明的东西。
- 补做代价可控：在冻结的 checkpoint 上重跑 `evaluate_arm` 即可，不需要重训。

**修复二选一**（两条都可接受，但必须选一条并写进文档）：

- (a) 实现：给 `CONTEXT_MODES` 增两种模式（复用 B3 的 instruction 覆盖机制，替换文本分别为无关词 / 固定短语），
  并实现同图相反指令的配对差分（`paired_delta` 已现成）；
- (b) 缩范围：把 §17.2 那句改成「本轮只落地 `shuffled`；`irrelevant_words` / `fixed_phrase`
  与相反指令配对差分列为 S6 前置，责任人/排期 = X」，并在 `PREFLIGHT_WHERE_B_PENDING.md` 建条目。

## 3.2 判别力回归测试：构造是否忠实红线案例

**本审阅独立重算**（不采信提交里的表格；同一构造，自写脚本）：

```
gt cells k=100  gt perimeter=36
field    cells  perim  pix3px_BF1  grid_BF1  hard_IoU
good       100     36      0.0220    1.0000    0.6807
half        50     26      0.5827    0.6452    0.5000
random     100    100      0.0134    0.0357    0.0050
prior      100     44      0.0000    0.0000    0.0000

pixel: half/good = 26.5x   good/random = 1.65x
grid : good/random = 28.0x
red-line direction (random > prior) on pixel metric: 0.0134 > 0.0000 = True
3px tolerance in grid cells at spec-5: 0.188
```

**四位小数逐格复现**，周长也对得上。判定：

| 复审点 | 结论 |
|---|---|
| 是否忠实红线的 **0.0394 vs 0.0327 机制** | **PASS（且诚实）**。红线的机制是「像素级 3px BF1 主要在测边界**长度**，随机 top-k 因此高过中心先验」。本仓库构造复现了**同向**失败（random 0.0134 > prior 0.0000），并在 docstring 与 NOTES §11.2 里**明确标注**红线的具体数字来自 RO-9c 的设定、不是本仓库测出来的。没有把别人的数字冒充成自己的实测——这一点很重要，本项目有过编造数字的前科 |
| 构造是否有效 | **PASS**。GT 取**偏心**主体（`_square(9,12,5)` 在 32×48 网格上）——居中 GT 会让中心先验天然正确，那才是无效构造；`_fields()` 的 docstring 明写了这个理由。四个场**同 k=100**，符合 `TOPK_RULE` |
| 是否在**真实分辨率**上测 | **PASS**。32×48 `F_pre` 网格 → ×16 上采样到 512×768，正是 A-5 实际改变的那两个尺度。由此得出的「3px 容差 = **0.188 个网格格**」我已独立验证 |
| 断言强度 | **PASS**。`pix["half"] > 10 * pix["good"]`（实测 26.5×）、`pix["good"]/pix["random"] < 3.0`（实测 1.65×）、grid 侧 `good > half > random >= prior` 且 `good/random > 10`（实测 28×），hard-IoU 必须**同序**。不是只断言「新指标 > 0」这种空话 |
| 防退化 | **PASS，且写法值得推广**。失败信息写着「If it is genuinely gone, A-5's rationale needs re-checking rather than this test being deleted」——把「这条测试红了该怎么办」写进断言本身 |

### 「boundary F1 盲区」的自报是否如实呈现

红线要求三列缺一不可，A-5 自己给出了理由：boundary F1 单列有盲区。核查其呈现：

| 载体 | 呈现情况 |
|---|---|
| **NOTES §11.2** | **如实且正确**：「boundary F1 单独一列**无法**区分「紧凑但位置错」与「散点噪声」（displaced 与 prior 都是 0.0000）」，并接「这正是红线要求三列缺一不可的原因」。我实测 displaced = 0.0000/0.0000、prior = 0.0000/0.0000 —— **括号里的数字是对的** |
| **单测** | **有**：`test_no_single_column_is_sufficient_which_is_why_there_are_three` 断言 displaced 的两列都是 0，并把中心先验作为「零信息」参照 |
| **报告模板**（`attribution_section` / `ATTRIBUTION_NOTE`，随每份 `metrics.json` 落盘） | **缺**。模板强制了「每提一次 `local_soft_iou_median` 必须同时给出 `grid_boundary_f1` / `center_prior_delta_*` / shuffle 与 null 两个 delta」这条**操作性**规则，但没有写**为什么**——盲区本身没进模板。NOTES 是逐轮文档，`ATTRIBUTION_NOTE` 才是跟着每块板走的那个 → **N25** |
| **commit message** | **有一处事实错误**：写作「无法区分「紧凑但位置错」与「散点噪声」（两者都 0.0000）」，但 random（散点噪声）实测是 **0.0357 / 0.0050**，不是 0.0000。真正两者都 0.0000 的是 displaced 与 **中心先验**。NOTES 里是对的，commit message 里错了 → **N22** |

## 3.3 `evaluate_gates` 算子分发修复 —— PASS（且实际 bug 比自述更严重）

修复本身正确：显式 `ops` 字典 + 未知算子 `raise ValueError`（`metrics.py:454-466`）。

但 commit message 与 NOTES §11.1 都写「原来只认 `>=` / `<=`，**其余一律走 `>=` 分支**」。
旧代码是 `ok = v >= thr if op == ">=" else v <= thr`（`2c94347^:metrics.py:293`），
Python 求值为 `(v>=thr) if (op==">=") else (v<=thr)` —— `>` 落到的是 **`<=` 分支**，即**整个判据被反转**。
实测：

```
op=>  v=+0.0  thr=0.0   OLD passed=True    NEW passed=False
op=>  v=+0.5  thr=0.0   OLD passed=False   NEW passed=True
op=>  v=-0.5  thr=0.0   OLD passed=True    NEW passed=False
```

即：不修的话，**margin 为正的好臂会被判失败、margin 为负（比零参数基线还差）的臂会通过**。
自述说的「恰好等于中心先验会通过」只是三种后果里最轻的一种 → **N21**（记录准确性，不是代码问题）。

## 3.4 §5.5 loss 确实一字未动 —— PASS

| 证据层级 | 内容 |
|---|---|
| **最硬** | `git show --stat 2c94347` 里**没有** `q3vl/whereb/losses.py`。A-5 从未打开过这个文件 |
| 常量 | `test_config.py:42-47` 仍独立钉住 `(MASK_IOU_W, MASK_BCE_W, MASK_BF1_W) == (1.00, 0.25, 0.10)`、`STAGE1_WEIGHTS`、`STAGE2_WEIGHTS`、`CURVE_Z_*` —— 这些**不是** A-5 新加的，是原有覆盖被保留 |
| A-5 专门的守卫 | `test_the_loss_is_untouched_by_a5`：三个权重 + `BOUNDARY_TOL_PX==3` + `BOUNDARY_KERNEL==3` + `boundary_f1_loss` 的 `tol_px` 默认值 = 3 + **源码级**断言 `mask_loss` 里出现 `boundary_f1_loss` 且**不出现** `grid_boundary_f1` |
| 最有价值的一条 | `test_loss_and_criterion_are_different_functions_on_purpose`：在同一对 (pred, gt) 上实算，断言 `abs((1-loss) - criterion) > 1e-6`，失败信息是「the criterion collapsed onto the loss; A-5 requires them to differ」。**这条防的是「有人把判据也搬进 loss 让数字好看」**，是本 amendment 归因价值的守门员 |

**断言强度评估**：足够。源码级 `"grid_boundary_f1" not in src` 是弱断言（改名即绕过），
但它上面压着「文件根本没被 diff 触及」+「常量逐个钉死」+「数值上两者必须不同」三层，
组合起来防退化是够的。唯一还缺的是 `mask_loss` 在固定输入上的 golden 数值回归——建议补，但不是 nit 级问题。

## 3.5 viz.py 拒绝式纪律 —— PASS（一处不对称，N24）

| 红线条文 | 实现 | 判定 |
|---|---|---|
| 「禁止逐图 min-max 着色」 | `color_scale(mode="per_image_minmax")` → `raise PerImageMinMaxError`，异常信息带红线出处与 RO-9c 数据。**不是 warning、不是 fallback** | **PASS** |
| 「色标只取有效格」 | `mode="valid_cells"` 只取 `valid` 为 True 的格；无有效格时 `raise`。单测实证同一个场带 mask 上界 0.4、不带是 9.0 —— **22 倍差**，把红线的机制变成可复现的数字 | **PASS** |
| 「pad 格显式画白或打叉，不许静默填补」 | `pad_style ∈ {"white","hatch"}`，未知值 `raise`；`FieldRender` 同时返回 `n_valid` / `n_pad`，pad 数量强制可见 | **PASS** |
| 「叠图禁直接 resize，必须用严格逆映射」 | `grid_to_img` 要求 `out_h % grid_h == 0`，否则 `raise ValueError("not an integer multiple")`；返回整数 `y_edges`/`x_edges`；`overlay_grid_on_image` 用 `np.repeat` 整数倍展开。单测断言热格恰好占满自己的 4×4 块、**不向邻块渗透** | **PASS** |
| 「着色归着色、算数归算数；进入判据的数字一律用未归一化原始场」 | `raw_stats` 取自未归一化场（单测断言 `max==0.4` 而非 1.0），与 `vmin/vmax` 分开返回；`test_render_returns_no_criterion_number` 扫 `to_dict()∪raw_stats` 的键，禁止出现 `iou/f1/auc/gate/score` | **PASS** |

**N24（不对称）**：`mode` 被守住了，但 `valid` 没有——`color_scale(f, valid=None, mode="valid_cells")`
会退化成**对全场取 min-max**，也就是被禁的那个东西（单测 `test_color_scale_uses_valid_cells_only:40`
自己就演示了 `color_scale(f, None)[1] == 9.0`）。这正是作者自己提出的判据「一个键之遥的默认值不算禁用」，
只是没有对称地应用到 `valid` 上。**当前 Where-B 事实上无 pad**（`F_pre` 走真实宽高比 `grid_from_geometry`，
没有 `expand2square`），所以现在不会出错；但 `viz.py` 是给 §13 与 Stage-What 共用的模块。
建议把 `valid` 变成必填，或要求显式传 `valid=NO_PAD` 之类的哨兵。

**N26**：`viz.py` 目前**只有测试在调用**（`rg` 确认无生产调用点）。纪律已经**可用**，但还没有**作用在任何已产出的图**上——
§13 的可视化必须真的走这个模块，否则等于没落地。

## 3.6 A-5 文本（协议 §17.2）与红线的忠实性、不越界

| 检查 | 结论 |
|---|---|
| 引用红线是否准确 | **PASS**。三次误导的数字（0.907/0.523、0.836 vs 0.695–0.784、0.9469/0.9499/0.9481 vs 0.605/0.508/0.509、2.2 dB）与 `CLAUDE.md:109-111` **逐字一致**，没有加工 |
| 是否越界改了不该改的 | **PASS**。§17.2「不受影响的部分」明列 §5.5 loss、§5.1–5.4 结构与 context、§10.3 优化配置、§5.3 八臂矩阵不变；我逐项核对 `config.py`：`LEARNING_RATE`/`WEIGHT_DECAY`/`WARMUP_RATIO`/`EFFECTIVE_BATCH`/`EPOCHS`/`EVAL_STEPS`/`SAVE_STEPS`/`ARMS`/`STRUCTURES` 全部未动 |
| §5.6 原表是否就地修订且留痕 | **PASS**。表上方加了 blockquote 脚注指向 §17.2，表内 AUC 行删除、改行加粗，选择规则第 2 条标注「（A-5）」。**没有偷偷改数字不留痕** |
| 「loss 不变」是否明确声明 | **PASS，且给了理由**：「判据用 grid 级、loss 用像素级，二者**故意不同**且必须在 REPORT 里写清：这恰好让判据列不再是被直接优化的量（归因价值反而更高）」。这个论证是对的，且与 `ATTRIBUTION_NOTE` 的 `weakly_optimised` 项一致 |
| 是否有对红线的过度解读 | **PASS**。红线只禁 AUC 作**空间场判据**；A-5 删得更彻底（连函数都删），依据是红线原文「新实验不得再产出该指标」——有出处，不是自行加码 |
| 指令条件性一节 | **FAIL → A5-B1**。声明了代码产不出来的框架，且未标 pending（见 3.1） |
| 模块 docstring 同步 | **N20**：`metrics.py:1-20` 的模块 docstring 仍是 A-5 **之前**的 §5.6 表——第 8 行还写着 `AUC_target >= 0.80`、第 9/16 行还写「3px boundary F1」、开头还写「the **nine** Where gates」（现在是十条）。这是维护者最先读的那张表 |

## 3.7 第 3 轮新增 NIT

| # | 内容 |
|---|---|
| N20 | `metrics.py:1-25` 模块 docstring 仍是 A-5 前的 §5.6 表（含 `AUC_target` 行、3px 措辞、"nine gates"）。与 `config.GATES` 直接矛盾 |
| N21 | commit message / NOTES §11.1 把 `evaluate_gates` 的旧行为写成「其余走 `>=` 分支」，实际走 `<=` 分支即**判据反转**（实测见 3.3）。修复正确，记录不准 |
| N22 | commit message 写盲区是「紧凑但位置错」vs「散点噪声」两者都 0.0000；实测 random = 0.0357/0.0050。NOTES §11.2 写的（displaced 与 prior）才是对的 |
| N23 | `paired_delta` 的 p 值是**置信区间反演**（`2·min(#≤0, #>0)/B`），不是重心化 bootstrap 或符号翻转置换检验。配对设计下 sign-flip permutation 是教科书选择且更便宜、更严格。另：全正差分时会报 `p = 0.0` 而非 `< 1/2000` |
| N24 | `viz.color_scale(f, valid=None)` 静默退化为全场 min-max（= 被禁的那个）。`mode` 守住了、`valid` 没有——「一个键之遥」标准未对称应用。Where-B 当前无 pad 所以不出错，但 viz 是 §13/Stage-What 共用模块 |
| N25 | 「boundary F1 单列有盲区」只写在 NOTES §11.2，**没进** `ATTRIBUTION_NOTE`/`attribution_section`——而后者才是随每份 `metrics.json` 落盘、跟着板走的那块 |
| N26 | `viz.py` 无生产调用点（只有测试）。§13 出图必须真的路由到这个模块，否则纪律只存在于库里 |

## 3.8 第 3 轮判决

**A-5 遗留 BLOCKER 1 个（A5-B1）、NIT 7 个。**

- **红线的空间场判据侧：完全落实，质量高于要求。** AUC 连函数一起删、top-k 统一阈值化、
  grid 级 BF1 替换、中心先验双 gate（含 bootstrap p）、`evaluate_gates` 顺带修掉一个会**反转判据**的真 bug、
  判别力回归在真实分辨率上做且被我逐格复现、防退化断言写法优秀、可视化纪律是**拒绝式**而非警告式。
- **红线的指令条件性侧：只落了 1/3，且协议文本按已完成书写。** 这是 A5-B1。
- **§5.5 loss 一字未动**：由 `git diff` 无该文件 + 三层测试共同证明。

**放行**：S3 准许（不受影响）；**S5 八臂训练准许**（A5-B1 纯评测侧，不改训练目标）；
**S6 选型与 REPORT 不准许**，直到 A5-B1 按 (a) 实现 或 (b) 缩范围并建 pending 条目二选一落地。
若选 (b)，REPORT 必须写明「指令条件性只有 `shuffled` 一条负控制，红线要求的另两条与相反指令配对差分未做」，
不得让读者以为 §17.2 的框架已经跑过。

---

# 第 2 轮 · 聚焦复审（2026-08-05）

| 字段 | 值 |
|---|---|
| 审阅范围 | **只审修复 diff 与其单测**，不重审全量；第 1 轮已 PASS 的条目不再复述 |
| 代码基线 | `a2c2427`（`q3vl/` 首次入库，N13 因此关闭）+ 两处未提交增量（`q3vl/where/basis.py`、`q3vl/where/tests/test_fpre.py`） |
| 复现方式 | 只读；未占用 GPU；全部结论由本审阅**当场重跑**，不采信 NOTES 自述 |
| 结论 | **B1-B5 全部清除**，每一项都用对抗性复现验证（不只看单测通过） |

## 2.0 复现的基线数字

| 项 | 命令 | 结果 |
|---|---|---|
| Where-B 单测 | `pytest q3vl/whereb/tests -q` | **213 passed / 21.9 s** ✅ 与 NOTES §8.1 一致（原 170 + 新增 43） |
| Where-A + 共享套件 | `pytest q3vl/where/tests q3vl/tests -q` | **157 passed / 56.0 s** ✅（NOTES §8.4 写的 120 是旧数；commit message 写的 157 才是当前值 → **N15**） |
| Where-A 是否被动过 | `git diff -- q3vl/where/` | 只有两处：`basis.py` 的 docstring 公式笔误订正（正是本审阅第 1 轮指出的 `(w0+<phi,w>)*alpha` → `w0+alpha*<phi,w>`）与 `test_fpre.py` 删一个未使用变量。**零数值改动**，157 个测试全绿 → 可接受 |

## 2.1 B1 · preflight 真实接入 —— **CLEARED**

| 复审点 | 实现 | 判定 |
|---|---|---|
| `run_model_checks` 真实接入 | `preflight.py:480-541`：真造 processor/collator/`load_model`/`FrozenVLM`/`open_dataset`，逐个调用两个 check 并 `rep.add` | PASS |
| `REQUIRED_CHECKS` 7 项强制 | `preflight.py:82-95`；`missing` 属性 + `run_where_b_preflight:581-583` 对缺失项**补写 fail 行**（双保险） | PASS |
| `complete` 与 `skipped` 语义 | `preflight.py:117-129`：`ok` = 无 fail 且七项不缺（skip 计为"在场"）；`complete` = 七项**全 pass**。`to_dict` 输出 `complete/n_skip/skipped/missing_required` | PASS |
| 异常不被吞 | 加载失败 → 两个 id 各写一条 `fail` 并附 `error`（`:525-529`）；check 自身抛异常 → 同样是 `fail`（`:539-541`）。**没有任何路径能让必需检查"消失"** | PASS |
| S3 gate 改依 `complete` | `PREFLIGHT_WHERE_B_PENDING.md:32-33`「**S5 的放行条件绑定 `complete`，不是 `ok`**」、`:110`「验收标准：JSON 里 `complete == true`。只看屏幕上的 `preflight PASS` 不够」 | PASS |

**对抗性复现**（本审阅当场跑，非引用）：

```
CPU  ok: True   complete: False  n_skip: 2  missing: []
     skipped: ['WB-P7b-hidden-contract', 'WB-P8b-h-where-causal-independence']

--with-model + 故意给坏 checkpoint:
MODEL ok: False  complete: False  n_skip: 0
     WB-P7b-hidden-contract              -> fail | HFValidationError: Repo id must use ...
     WB-P8b-h-where-causal-independence  -> fail | HFValidationError: Repo id must use ...
```

第 1 轮同一条命令的输出是 `ok: True, n_pass: 5, n_fail: 0, n_skip: 0` 且两个 id **根本不出现**。
现在坏输入换来两条 `fail` + `ok=False`，**静默通过的路径已被堵死**。
`test_contracts_and_preflight_wiring.py:91-100` 用 `/nonexistent-model-dir` 把这条行为钉成回归测试。

## 2.2 B2 · `open_dataset` 工厂三态 —— **CLEARED**

| 复审点 | 实现 | 判定 |
|---|---|---|
| 三态工厂 | `data.py:251-299`：① 已发布 `MaskViewStore` 优先；② 构造失败（`FileNotFoundError`/非原子 `RuntimeError`）→ 回退实时 `MaskResolver`，并把原因写进 `info["maskview_unavailable"]`；③ `need_mask=False` 两者都不建 | PASS |
| mask 源前置校验 | `data.py:141-147`：`need_mask and include_local and 两个源都为 None` → **构造期 `ValueError`**，在任何 IO 之前。第 1 轮是"第 0 个样本才炸" | PASS |
| `need_mask=False` 时 local 必须 raise | `data.py:96-111` `mask_target_hi()`：local + `mask_loaded=False` → `RuntimeError("...must not be faked as all-ones")`；local + 应有 mask 却没有 → `RuntimeError("no GT mask")`；**只有 global 才返回全 1**（global 的全 1 是 §2.1 的定义，不是伪造） | PASS |
| 三个脚本真的走工厂 | `test_dataset_wiring.py:82-108` 用 **AST 扫描**三个脚本：必须出现 `open_dataset(...)` 调用，且不得存在不带 mask 源关键字的裸 `WhereBDataset(...)`。这比字符串 grep 强，也比"相信作者改了"强 | PASS |
| provenance 落盘 | `info` 进 `run_setup.json`（`run_where_b.py:151`）与两个作业的 `setup`，所以"这次用的是哪条 mask 源"是可审计的 | PASS |

**两份 smoke 日志的证据力评估**（关键：它们**支撑什么、不支撑什么**）：

| 日志 | 支撑的结论 | **不**支撑的结论 |
|---|---|---|
| `b2_make_oracle_latents_smoke.log`（4 样本，exit 0，9.8 s） | 走通了 ②**实时 resolver** 回退（`maskview_unavailable = FileNotFoundError: .../maskviews/V_where 未发布`）；`n_ok = {band:4, cband12:4}`、`n_rejected={}`；`manifest.status=complete`、`verify.ok=true`、`checksum_failures=[]` → **数据链路 + 原子发布 + 校验回读全通** | 不支撑任何拟合质量/墙钟结论（basis 是 scratchpad 里的合成件，`n_random=1, max_iter=20` 远低于生产的 6/120）；**也没有记录 `device`** → 无法审计是否碰了 GPU（**N14**） |
| `b2_make_generated_context_smoke.log`（2 样本，exit 0） | 走通了 ③`need_mask=False`（`mask_source: null`，不再解 `.cgt.png`）；`vlm.device = "cpu"` **明确记录未占 GPU**；`manifest.status=complete`、`store.index_rows=2` → 发布后能读回 | 不支撑生成质量：用的是 **base 未 SFT 权重 + `--max-new-tokens 8`**，所以 `format_failure_rate=1.0`、`starts_with_where_open_rate=0.0` |

**关于第二条的过度解读风险，NOTES §8.2 已经自己写明**：「这两个数只证明路径通，不是质量信号；真实数字要等 S2 用 `checkpoint-4976` 全量跑完」——
这个自我限定是**准确且必要**的，我确认它没有把 smoke 数字当成实验结果（CLAUDE.md §14 末段禁止的行为）。

**结论**：两份日志**足以**支撑"B2 的崩溃路径已修好、三态中的 ②③ 已在真实索引 + 真实 shard 上跑通"。
① **published_maskviews** 这一态只在单测里覆盖（Where-A 的 maskview shard 尚未发布，客观上跑不了），
S1 完成后第一次真实 `open_dataset` 会自动切到 ①，`run_setup.json` 的 `mask_source` 字段就是验收点 → 记为 **N16**。

## 2.3 B3 / 裁定 D-B15 · instruction + where 成对交换 —— **CLEARED（含端到端对抗验证）**

| 复审点 | 实现 | 判定 |
|---|---|---|
| 成对交换 | `context.py:183-202` `shuffled_context(tokenizer, partner_id, partner_where_text, partner_instruction)`，两半**取自同一 partner**；`data.py:386-390` 从 `by_id[partner]` 同时取 `where` 与 `instruction` | PASS |
| 覆盖真的进 prompt | `data.py:405` `_PromptShim(s, instruction=ctx.instruction)`；`_PromptShim.__init__:500` `instruction if instruction is not None else s.instruction` | PASS |
| `__post_init__` 强制 | `context.py:71-74`：**只有 `shuffled` 允许携带 instruction 覆盖**，其他三种模式带了就抛 `ValueError` → 不可能有第二条路径悄悄换指令 | PASS |
| 空指令被拒 | `context.py:194-198`：partner 指令为空/纯空白 → `ValueError` | PASS |
| `ShuffleIndex` 强制两半 | `context.py:217, 226-233` `REQUIRED_FIELDS = ("where", "instruction")`，缺任一半即 `ValueError`；`WhereBDataset.shuffle_records()`（`data.py:191-209`）同时带两半 | PASS |
| 可观测性 | `WhereContext.to_dict()` 出 `instruction_swapped`；`evaluate.py:98` 每行 `per_sample.jsonl` 带 `instruction_swapped` | PASS |

**对抗性端到端验证**（本轮新增：所有单测都停在 `WhereContext`/`_PromptShim` 层，
**没有一条**断言"换掉的指令真的改变了编码后的 prompt ids"。若 `Sft2SegCollator.build_prompt_text` 忽略入参，
整个 B3 修复就是装饰性的。用真实 processor + 真实 collator 实测）：

```
gt       ctx.instruction : None
shuffled ctx.instruction : Warm the church and the hillside behind it.
prompt len gt/shuffled   : 402 / 404
PROMPT IDS DIFFER        : True     (first differing position 389, 12 tokens differ)
decoded tail gt   : ...<|vision_end|>Make the jellyfish darker and cooler.<|im_end|>
decoded tail shuf : ...<|vision_end|>Warm the church and the hillside behind it.<|im_end|>
where-span ids differ    : True
```

指令与 `<where>` 正文**双双**换成了 partner 的。§5.6 第 6 行 gate 现在名副其实。
（建议把这段端到端断言补成常驻单测 → **N17**。）

## 2.4 B4 · autocast 守卫 —— **CLEARED（守卫下沉，且防退化设计正确）**

| 复审点 | 实现 | 判定 |
|---|---|---|
| 守卫内置而非调用点 | `fields.py:148-164` `no_autocast()`；被 `s_from_params:174`、`predict_fields:218`、`oracle_fields:259` **各自**包住。任何外层 autocast 都无法再降精度——这正是我第 1 轮要求的"守卫下沉"，比在 trainer 里加一层强 | PASS |
| 事后断言 | `s_from_params:179-183` 断言 `s.dtype == phi_dir.dtype`；`predict_fields:238-240` 断言**全部输出**同 dtype | PASS |
| 两路 `require_dtype` | 训练 `trainer.py:82` + 额外的 `s_low` 显式检查 `:84-88`；评测 `evaluate.py:84` 与 `:134`（oracle mask）。**两端都是 `torch.float32`**，"训练优化的函数 = gate 度量的函数"成为运行期断言 | PASS |
| 外层 autocast 仍在 | `trainer.py:205` 保留 `with self.autocast:`（§10.3 `precision: bf16` 仍对 connector 生效），`compute_batch:74` 再在内部关掉——**分层正确**：bf16 省显存的地方留着，解析链路强制 fp32 | PASS |
| `test_precision.py` 防退化 | **设计正确**：`:40-45` 先断言"没有守卫时 matmul 确实会被降成 bf16、elementwise 不会"——若 torch 将来改 autocast 策略，这条**先红**，而不是让守卫测试退化成恒真；`:99-100` 同理先断言"CBand12 在 σ 下界附近对 bf16 的敏感度 > 1e-3 仍然存在"，再断言实装路径不产生该差异 | PASS |
| 断言强度 | 用 `torch.equal`（逐位）而非 `allclose` | PASS |

**对抗性复现**：

```
dtype under outer autocast: torch.float32   bitwise equal to fp32 ref: True
m_hi dtypes: float32 / float32              bitwise equal: True
require_dtype enforced: "phi_dir is torch.bfloat16, the caller requires torch.float32"
```

第 1 轮实测的 `max|Δs| = 2.1e-2 → CBand12 max|Δm| = 0.42` 已归零。§5.3 的 Band vs CBand12 受控对比不再被精度偏置污染。

**残留**：`trainer.py:101` 的 `stats["s_dtype"]` 实际取的是 `losses[0].mask_term.dtype`（loss 的 dtype，不是 `s_low` 的）。
它是有效代理（`mask_term ← m_hi ← s_hi ← s_low`），且 `:84-88` 另有对 `s_low` 的直接断言，所以**覆盖没漏**，只是字段名误导 → **N18**。

## 2.5 B5 / 裁定 D-B16 · 辅助项分母 —— **CLEARED**

| 复审点 | 实现 | 判定 |
|---|---|---|
| `mean(mask) + sum(aux)/n_with_oracle` | `losses.py:300-311`：`mask_term` 全 batch 平均，`aux_term` 只对有 oracle 的样本求和后除以其个数。`SampleLoss` 把两项拆开存（`:205-210`）使这成为可能 | PASS |
| 默认不会误用 `"batch"` | 默认值来自 `config.py:141` `AUX_DENOMINATOR = "with_oracle"`；`aggregate` 的默认参数就是这个常量；`"batch"` 只能**显式**传入；未知值 `raise ValueError`（`:310`），不会静默退回 | PASS |
| `"batch"` 兼容档的定位 | docstring 明写「kept only so the effect can be reproduced」；`test_losses.py:278-280` 用它复现 0.25 稀释并断言 `aux_effective_scale == 0.25` —— 把旧 bug 变成一条**可复现的对照**而不是删掉了事 | PASS |
| 逐步日志字段 | `losses.py:314-319` 出 `n / n_with_oracle / oracle_fraction / aux_denominator / aux_effective_scale`；`trainer.py:233-236` 另写 `effective_aux_weights = 名义权重 × aux_effective_scale` 进 `steps.jsonl`。**未来任何稀释都会显式出现在日志里** | PASS |
| 归一化位置 | `aggregate` 在 `compute_batch` 的 `autocast(enabled=False)` 块内调用（`trainer.py:100`），loss 归约也是 fp32 | PASS |

**残留（不阻断，但请写进 REPORT 的"设置"节）**：`aux_effective_scale` 是**逐 micro-batch** 的，而 `steps.jsonl` 每个
optimizer step 只写最后一个 micro-batch 的 `stats`。micro_batch=4、local 占比 47.45% 时，
单个 micro-batch 全是 global 的概率约 `0.53^4 ≈ 7.9%`，那种 micro-batch 对 aux 零贡献。
于是**每个 optimizer step 的实效 aux 权重 ≈ 0.92 而不是恰好 1.0**（GAS=8 时，期望有 ~7.4/8 个 micro-batch 带 aux）。
这与 D-B16 的意图一致（远好过 0.47），但数字不是 1.000 → **N19**：建议在 accumulate 窗口上聚合 `aux_effective_scale` 后再写日志。

## 2.6 附带项复审

| 项 | 实现 | 判定 |
|---|---|---|
| `contracts.py` 常量上提（N1） | `contracts.py:30-40` 定义 `SEGMENT_HIDDEN_LAYER=-1` / `SEGMENT_HIDDEN_FINAL_NORM=True` / `SEGMENT_HIDDEN_RULING`（把 D-B2 裁定原文写进代码）；`config.py` 改为 re-export（单测断言 `is` 同一对象） | PASS |
| 重声明扫描测试 | `test_contracts_and_preflight_wiring.py:27-41`：遍历包内所有 `.py`（排除 `contracts.py` 与 tests），任何 `SEGMENT_HIDDEN_* =` 赋值即失败。**Stage-What 若自行声明会当场变红** —— 正是 N1 要的效果 | PASS |
| `assert_genctx_coverage`（N2） | `run_where_b.py:58-78`：启动时比对 `dataset.refs` 与 `genctx.sample_ids`，缺一即 `SystemExit` 并说明"没有 GT 回退是故意的（§5.4）"。`test_dataset_wiring.py:117-135` 覆盖两个方向 | PASS |
| N5 `render_mode` | `evaluate.py:103-113`：改为显式赋值（不再 `setdefault`）+ 值域断言，注释直接引用 N5 的失效模式（"row would vanish from **both** aggregates"） | PASS |
| N7 `run_where_b.sh` | `submit <gpu> <log> <marker-grep> <cmd...>`：① `rm -f` 日志与 `.pid`；② `export CUDA_VISIBLE_DEVICES` + `nvidia-smi` 打印目标卡状态，`ps -p $pid -o pid,etime,cmd` **把命令行打出来**（PID 错了看得见）；③ 改成 `until grep -q "$want" "$log"` 轮询实质内容，进程中途死亡或 600 s 无输出都 `return 1`；④ 三步全过才写 `job.marker`（含 gpu / 验证串 / 等待秒数）。usage 明示 `train W01 0 &` / `train W02 1 &` | PASS |
| `ATTRIBUTION_NOTE` 机制 | `metrics.py:191-222` 把 D-B1 的后果结构化（`directly_optimised` / `weakly_optimised` / `not_optimised` / `fair_comparisons` / `reporting_rule`），`arm_metrics:179` 让它**随每份 `metrics.json` 落盘**，另有 `attribution_section()` 渲染 REPORT 段落。这比"写进 REPORT 靠自觉"强得多——**直接实现了我第 1 轮 §四对 D-B1 的意见** | PASS |
| `null_context_gap` 新增 | `metrics.py:266-280` `context_deltas` 同时出 `null_context_gap` 与 `instruction_shuffle_iou_drop` → 红线"每个消融行必带 Δ_const/Δ_shuffle"两个都有了 | PASS |
| N3 逐样本 seed | `make_oracle_latents.py:133` `FitConfig(seed=args.seed + i)`，注释指明与 `calibrate.fit_sample` 的 `seed_offset=j` 对齐 | PASS |
| N4 `mask_low` 来源 | `make_oracle_latents.py:118-127`：优先读 Where-A 发布的 float `.masklow.npy`，回退才从 `mask_hi` 重算，并把 `mask_source` 写进每条 fit 记录 | PASS |
| N8 `format_stats` | `evaluate.py:176-179` 每次 `evaluate_arm` 开头清零 | PASS |
| N13 `q3vl/` 入库 | `a2c2427` 已提交，§13.1 的"config 快照 + git commit"对模型代码不再是空的 | PASS |
| §5.6 九项 gate 未被动过 | `config.py:167-187` 与协议表逐行比对，**阈值/方向/指标名全部未变** —— 修 blocker 没有顺手放松判据 | PASS |

## 2.7 第 2 轮遗留 NIT（8 项，全部不阻断）

| # | 状态 | 内容 |
|---|---|---|
| N6 | **沿用未改** | `instruction_shuffle_iou_drop = generated − shuffled` 仍混了两个轴。现在可接受：主榜=generated 是 §5.4「模型选择以 generated context 为主」的字面读法，且 `gt_generated_iou_gap ≤ 0.05` 这条 gate 把混淆量夹住了。建议 REPORT 里同时列 `gt − shuffled` 作为参照 |
| N10 | **沿用未改** | `AUC_target` 的定义（预测 mask 作 score、GT 在 0.5 二值化的 ROC AUC）仍是协议未定义、NOTES 未申报的量。实现本身正确（tie 平均秩、单类返回 `None`），只差一条决策记录 |
| N11 | **沿用未改** | `metrics.py:141` `if ... r.get("oracle_soft_iou")` 仍是真值判断，`oracle_soft_iou == 0.0` 的样本被静默排除出比值统计 |
| N12 | **沿用未改** | 仍无 checkpoint 滚动删除。4,975 步 / 每 500 步存一次 → 每臂 ~11 个；W07/W08 各 ~280 MB → 8 臂合计 ~20 GB。确认磁盘够即可 |
| N14 | **新增** | `make_oracle_latents.py` 的 `setup` 字典**不记录 `device`/`dtype`**，所以 `b2_make_oracle_latents_smoke.log` 无法被审计"是否碰了 GPU"（genctx 那份因为有 `vlm.device="cpu"` 就可以）。建议把 `args.device/args.dtype` 写进 `setup` |
| N15 | **新增** | NOTES §8.4 写「120 passed」，实测与 commit message 都是 **157**（Where-A 已被 WA-IMPL 扩充）。数字过期，改一下 |
| N16 | **新增** | `open_dataset` 三态里的 ①`published_maskviews` 只有单测覆盖，真实数据上从未走过（Where-A 的 maskview shard 未发布）。S1 完成后第一次跑要**核对 `run_setup.json` 的 `mask_source == "published_maskviews"`**，别默认它一定切过去了 |
| N17 | **新增** | B3 的端到端链路（`_PromptShim → collator.encode_one → build_prompt_text`）**没有常驻单测**；本轮由审阅手动验证通过。建议把"shuffled 的 prompt_ids ≠ 本人的 prompt_ids"补成测试，否则将来 collator 改了 prompt 构造不会有人发现 |
| N18 | **新增** | `trainer.py:101` 的 `stats["s_dtype"]` 取的是 `mask_term.dtype` 而非 `s_low.dtype`，字段名误导（覆盖没漏，`:84-88` 另有直接断言） |
| N19 | **新增** | `aux_effective_scale` 是逐 micro-batch 的，`steps.jsonl` 每步只写最后一个；实效 aux 权重约 0.92 而非 1.000（见 2.5）。建议在 accumulate 窗口上聚合后再记 |
| — | 关闭 | N1 / N2 / N3 / N4 / N5 / N7 / N8 / N13 已修复并有测试；N9 属数据量事实，无需改代码 |

## 2.8 第 2 轮判决

**遗留 BLOCKER 0 个。**

- **S3（GPU preflight）：准许。** B1 已使 `preflight --with-model` 真正执行 §14 项 7b/8b，坏输入换来 `fail` 而不是消失；
  B2 已使它依赖的 `open_dataset(need_mask=False)` 能构造出样本。
- **S1 / S2（oracle latent、generated context）：准许提交。** 两个脚本在真实索引 + 真实 shard 上跑通并原子发布、校验回读通过。
  提交前请按 PENDING 先 `--limit` 标定墙钟。
- **S5（8 个主臂）：准许，条件是三项前置在 GPU 上真实达成** —— 而这三项现在**都由代码自己把关**，不再靠人守：
  1. `preflight_where_b.json` 的 `complete == true`（不是屏幕上的 `PASS`，也不是 `ok`）；
  2. `assert_genctx_coverage` 在启动时通过（缺一条 genwhere 记录即 `SystemExit`）；
  3. `load_basis` 校验 `BA-3-Joint` 的 sha256 通过（Where-A 未定档就起不来）。

**给主 agent 的一句话**：五个 blocker 的修复**都不是最小补丁**——B1 补的是"缺失 = 失败"的语义而不只是加个调用，
B2 补的是唯一工厂 + 构造期校验 + AST 扫描而不只是补两个参数，B4 把守卫下沉进 `fields` 使得任何调用点都无法重新引入，
B5 把被否决的旧行为保留成可复现的对照档并把实效权重写进日志。防退化设计（先断言 bug 的前提仍然成立，再断言修复有效）
是本轮质量最高的部分。新增的 `ATTRIBUTION_NOTE` 把我第 1 轮对 D-B1 的意见变成了每份 `metrics.json` 都带的结构化字段，
超出了修 blocker 的要求。

---

# 第 1 轮 · 完整审阅记录（2026-08-05，历史留存）

## 复现环境与既有断言复核

| 项 | 命令 | 结果 |
|---|---|---|
| Where-B 单测 | `pytest q3vl/whereb/tests -q` | **170 passed / 19.45 s** ✅ 与 NOTES 一致 |
| Where-A + 共享套件回归 | `pytest q3vl/where/tests q3vl/tests -q` | **120 passed / 36.09 s** ✅ 与 NOTES 一致 |
| Where-A 未被篡改 | `q3vl/` 整目录未入 git，无 diff 可比；改以 mtime 排序 | `q3vl/where/*.py` **全部早于** `q3vl/whereb/*.py`，且 Where-A 120 个测试全绿 → 支持"无篡改"，但**这不是强证据**（见 N13） |
| 参数量表 | 手工重算 stream / bank / heads 逐项 | W01 34,838,745；W07 69,835,365；W08 69,851,781 —— **与 NOTES §3.3 逐位一致** ✅ |

参数量手算复核（确认 §5.2 的三档差异归因是真的）：
`ConnectorStream` = text_proj 1,311,232 + vision_proj 524,800 + pos 66,048 + 6×5,257,730 + norm_out 1,024 = **33,449,484**；
`bank(8×8)` = 32,896、`bank(16×16)` = 131,584；`heads(joint,band)` = 1,356,365、`heads(split,band)` = 2,673,229；
readout 差 = 512×(36−4)+(36−4) = **16,416**。W07 = 2×131,584 + 2×33,449,484 + 2,673,229 = 69,835,365 ✅。

---

## 一、BLOCKER

### B1 · `preflight --with-model` 从不执行 §14 项 7b/8b，却报 PASS（静默空转）

**规格**：§14 项 7「GT/generated/null/shuffled context 数据流无串线」、项 8「证明 `Q_where` 不读 `H_color`」；
`PREFLIGHT_WHERE_B_PENDING.md:98` 明写 `WB-P7b-hidden-contract` **「FAIL 则不许继续」**。

**实现**：`q3vl/whereb/preflight.py:415-444`

```python
    rep.add(check_context_flows(processor.tokenizer))
    rep.add(check_no_h_color(WhereBModel(arm_config("W01")).eval()))
    rep.add(check_no_target_leak())
    rep.add(check_zero_init_gates())
    rep.add(check_parameter_table(arms))
    if skip_model:
        for cid in ("WB-P7b-hidden-contract", "WB-P8b-h-where-causal-independence"):
            rep.add(Check(cid, "skip", {}, "--skip-model (needs a real forward)"))
```

`--with-model`（即 `run_where_b.sh preflight`）只做一件事：**把两个 skip 标记去掉**。driver 里没有任何路径去构造 VLM 并调用
`check_hidden_contract`（`preflight.py:287`）与 `check_h_where_causal_independence`（`preflight.py:251`）。
`rg` 全仓确认这两个函数**零调用者**（测试里也没有）。

复现（CPU，`skip_model=False`）：

```
ok: True  n_pass: 5  n_fail: 0  n_skip: 0
check ids: ['WB-P7-context-flows', 'WB-P8-no-h-color', 'WB-P9-no-target-leak',
            'WB-P-zero-init-gates', 'WB-P-param-table']
```

**危害**：操作员按 PENDING S3 跑 `run_where_b.sh preflight`，屏幕打印 `preflight PASS`、JSON 里 `n_skip=0`，
会得出"§14 项 7b/8b 已通过"的结论，而这两项**根本没跑**。这正是 CLAUDE.md 反复警告的静默失败类型
（"见到 PASS 不等于跑过"）。`PreflightReport.ok`（`preflight.py:86-88`）只检查"没有 fail"，缺失的检查不计入。

**修复**：`run_where_b_preflight` 增加 `else:` 分支，加载真实 checkpoint + processor + 一条真实样本，
调用两个函数并 `rep.add(...)`；同时把 `ok` 改成"必需检查 id 集合齐全 **且** 无 fail"。

---

### B2 · 两个硬前置作业（S1 oracle latent、S2 generated context）**当场崩溃，无法运行**

**规格**：§5.4「训练 batch 固定 50% teacher、50% generated」、§5.5 三个 oracle 辅助 loss；
PENDING 把它们列为 S1/S2，且 S5 不可跳过。

**实现**：

- `q3vl/whereb/scripts/make_generated_context.py:84` — `dataset = WhereBDataset(args.split, limit=args.limit)`
- `q3vl/whereb/scripts/make_oracle_latents.py:76` — `dataset = WhereBDataset(args.split, include_global=False, limit=args.limit)`

两处都**没有传 `maskviews=` 也没有传 `mask_resolver=`**。`WhereBDataset.__getitem__` 在 `data.py:173` 无条件调用
`self._mask(...)`，而 `data.py:198-201`：

```python
        raise RuntimeError(
            f"{sample_id} is a local sample but neither a published maskview store "
            "nor a live MaskResolver was provided"
        )
```

复现（真实 `V_where.index.jsonl` + 真实 shard，CPU，只读）：

```
len(ds) = 40
[0] RuntimeError: sft_00680d96738cd6077be81308b15e19b3 is a local sample but neither a
    published maskview store nor a live MaskResolver was provided
succeeded on 0 samples before the first failure
--- include_global=False (make_oracle_latents.py) ---
RuntimeError: sft_00680d96738cd6077be81308b15e19b3 is a local sample but ...
```

即：`make_oracle_latents.py` **第一个样本就死**；`make_generated_context.py` 在 train 段撞上第一个 local 样本时死
（train 段 47.45% 是 local，实测见下表）。这不是"权重没就绪"的伪失败——两个脚本的**数据构造本身**是错的。

顺带：`make_generated_context.py` 根本不需要 mask，`__getitem__` 却强制加载它；应当给 `WhereBDataset` 一个
`need_mask=False` 开关，或在 `generate_records` 里走一条不取 mask 的轻路径（169,215 个样本每个多解一次 `.cgt.png`
是纯浪费）。

**修复**：两处都传 `maskviews=MaskViewStore(WHERE_A_MASKVIEW_DIR / split)`（`run_where_b.py:102` 已经这么做了），
`make_generated_context.py` 另加 `need_mask=False`。**修完必须用 `--limit 8` 真跑一次**再提交全量。

---

### B3 · `shuffled` 只换 `<where>` 正文，**指令没有被交换**——与 §5.4/§5.6 的 gate 名字不符，且属静默拍板

**规格**：
§5.4「`shuffled` 在同一图像、同一局部层级内**交换 instruction/where context**，防止用图像主体显著性冒充指令理解」；
§5.6 gate 第 6 行「**instruction shuffle** 后 IoU 降幅 `>= 0.20`」。

**实现**：`q3vl/whereb/data.py:278-289`

```python
        elif mode == SHUFFLED:
            ...
            text = self.shuffle_index.by_id[partner]["where"]
            ctx = shuffled_context(self.tokenizer, partner, text)
```

只替换了 `<where>` 段的 token ids。而同一个 `build()` 里，prompt 仍来自样本自己：`data.py:302`

```python
            enc = self.collator.encode_one(_PromptShim(s))
```

`_PromptShim.__init__`（`data.py:390-396`）把 `self.instruction = s.instruction` —— **样本本人的指令**。
`Sft2SegCollator.encode_one`（`collator.py:138`）用 `build_prompt_text(sample.instruction)` 造 prompt。

**为什么这不是等价写法**：`Q_where` 只读 `H_where` + `F_pre`，指令**唯一**的入口就是 `H_where`
（因果注意力下 `<where>` 位置的 hidden 会 attend 到 prompt 里的指令）。所以留着真实指令 = 让正确答案仍然可达。
一个"完全靠正确指令、根本不看 `<where>` 推理"的模型，在这个 shuffled 下降幅接近 0，会**被这条 gate 误杀**；
反过来，一个靠图像显著性作弊的模型也降幅接近 0 ——**这条控制既抓不到作弊，也误伤好模型**，
和 §5.4 写的目的（"防止用图像主体显著性冒充指令理解"）正好相反。

**更严重的是这是一次静默拍板**。NOTES §四 D-B6 只讨论了分组键，D-B7 只讨论了"用 partner 的 GT 还是 generated 文本"，
**没有任何一条提到"prompt 里的指令换不换"**；而 D-B7 的行文写着「只换指令内容」（`NOTES.md:184`），
与代码**恰好相反**。CLAUDE.md 派工协议第 3 条：属于决策的必须写进 NOTES 待决策节，"不许静默拍板"。

**修复**：`_PromptShim` 增加一个 `instruction` 覆盖参数，`shuffled` 模式下同时替换 instruction 与 where 正文；
或由主 agent 明确裁定"只换 where 正文"并**把 §5.6 那一行 gate 改名/改判据**（改判据必须在 S5 之前，
看到结果后再改 = §10.3 末段禁止的行为）。

---

### B4 · bf16 autocast 漏进 `s_low`：训练用 bf16、评测用 fp32，且**对 CBand12 臂的伤害远大于 Band 臂**

**规格**：§5.3 的 8 臂是 4 结构 × 2 readout 的**受控对比**；§10.3 `precision: bf16`。
实现自己的规格声明（`trainer.py:26-30`）：

> "the connector runs under bf16 autocast, but **everything from `phi_dir` onwards is float32**.
> The guided filter divides by `var + 1e-3` and the readouts exponentiate; both lose too much in bf16."

`fields.py:168-171` 同样声明「Everything is computed in the dtype of `phi_dir`（float32 in training）」。

**实现与声明不符**：`trainer.py:187-190` 把整个 `compute_batch` 包进 autocast：

```python
                with self.autocast:
                    total, stats, _ = compute_batch(
                        self.model, batch, self.arm_cfg, weights
                    )
```

`compute_batch` 内部调用 `predict_fields` → `s_from_params`（`fields.py:152-153`）：

```python
    q = w0 + alpha * (phi_dir @ w_dir)
```

`@` 是 matmul，**autocast 的 bf16 白名单第一条**。实测（CPU autocast，与 CUDA 语义一致）：

```
avg_pool2d: float32   interpolate bilinear: float32   exp/sigmoid/tanh: float32
matmul: bfloat16      linear: bfloat16
```

即 guided filter 与 readout 确实是 fp32（作者的分析对），**但 `s_low` 本身是 bf16**——正好是他要保护的那个量。
在真实量级（`phi_dir ~ N(0,1)` 71 维、`‖w_dir‖=1`、`alpha=2`）上实测：

```
max|Δs| = 2.14e-2    mean|Δs| = 4.72e-3    std(s) = 1.51
cband12 (σ=0.03):  max|Δm| = 4.21e-1   mean|Δm| = 8.63e-3
band  (k≈8,h≈1):   max|Δm| = 8.91e-2
```

**三重危害**：

1. `evaluate_context`（`evaluate.py:40`）只有 `@torch.no_grad()`、**没有 autocast** → 评测的 `s` 是 fp32。
   于是「训练优化的 mask」与「gate 度量的 mask」不是同一个函数。
2. CBand12 的 `σ ∈ [0.025, 0.30]`（`where/config.py:66`），在 σ 下界附近 `Δs = 2e-2` 会把
   `exp(−0.5((z−μ)/σ)²)` 改变数倍 → **W02/W04/W06/W08 系统性受损，W01/W03/W05/W07 基本无感**。
   §5.3 要对比的正是这两组，这是一个直接污染主结论的偏置。
3. `L_s = Huber(s_pred/3, s*/3)`（`losses.py:160`）的监督噪声底也被抬到 ~5e-3。

**修复（一行）**：在 `compute_batch` 里、`model(**batch.inputs)` **之后**、逐样本循环**之前**，
包一层 `with torch.autocast(device_type=..., enabled=False):`；或在 `predict_fields` 入口处禁用 autocast。
并把 `evaluate_context` 与训练的精度口径显式对齐（两边都 fp32）。

---

### B5 · oracle 辅助 loss 被 global 样本稀释到名义权重的 ~0.47 倍——未在 NOTES 申报

**规格**：§5.5

```text
前 30% optimizer steps:  L_where = L_mask + 1.00 L_s + 1.00 L_curve + 0.10 L_dir
```

并解释「前段用 oracle latent 解决 `s/readout` 联合优化的**非辨识和早期坍缩**」。

**实现**：主 agent 已裁定 D-B5「global g1-g4 进训练、oracle 辅助关闭」，`data.py:362` 对 global 样本跳过 oracle：

```python
        if self.oracle is not None and not sample.is_global:
```

`sample_loss`（`losses.py:242-247`）在无 oracle 时**只**累加 `L_mask`；随后 `aggregate`（`losses.py:271`）：

```python
    total = torch.stack([l.total for l in losses]).mean()
```

对**全部**样本取平均。于是 `L_s/L_curve/L_dir` 的**实效权重 = 名义权重 × (带 oracle 的样本占比)**。

实测 split 组成（读 `train.index.jsonl` / `V_where.index.jsonl`）：

| split | n | global | local | **local 占比** |
|---|---:|---:|---:|---:|
| `train` | 159,215 | 83,671 | 75,544 | **47.45%** |
| `V_where` | 896 | 496 | 400 | 44.64% |

即 stage-1 的 `1.00 L_s + 1.00 L_curve` 实际是 `≈0.47 L_s + 0.47 L_curve`（再乘以 Where-A 的 fit 成功率），
**比 §5.5 给 stage-2 定的 0.25 更接近 stage-2 而不是 stage-1**。两段 schedule 的对比因此被压扁，
而 stage-1 存在的全部理由就是"用大权重的 oracle 监督压住早期坍缩"。

**这是一次未申报的静默决策**：NOTES D-B5 只写了"对 global 屏蔽三个辅助 loss"，
**没有指出屏蔽 + 全 batch 平均 = 名义权重被 batch 组成打折**。两种读法都成立：
(a) 现状：权重是"每 batch"的；(b) `sum(w·L_aux)/n_with_oracle`：权重是"每有 oracle 的样本"的，保持 §5.5 字面值。
必须由主 agent 裁定，且**必须在 S5 之前**——8 臂跑完再改等于全部重跑。

**最低要求**：无论裁定哪一种，`steps.jsonl` 必须记录每步的 `n_with_oracle / n`
（`aggregate` 已经算了 `stats["n_with_oracle"]`，只是没有换算成实效权重）。

---

## 二、逐项对照表（任务卡八个必审重点）

### 1. Connector 结构（§5.1）— **PASS**

| 条文 | 实现 | 判定 |
|---|---|---|
| 宽 512 | `config.py:44` `CONNECTOR_DIM = 512` | PASS |
| 6 pre-norm blocks | `config.py:45`；`connector.py:154` `ModuleList(... for _ in range(cfg.n_blocks))`；`ConnectorBlock` 全部 `norm_*` 在 attn/ffn 之前 | PASS |
| 8 heads / FFN 2048 | `config.py:46-47`；`connector.py:116-118` | PASS |
| 顺序 self → cross(H_where) → cross(F_pre) → FFN | `connector.py:128-137` 逐行即此顺序 | PASS |
| cross-attn residual gate zero-init，真 zero 且可学 | `connector.py:108,113` `nn.Parameter(torch.zeros(1))`；`preflight.check_zero_init_gates` 实测 W01/W08 初始输出对输入的 max diff = **0.0**；`build_optimizer` 把 gate 归入 no-decay 组但**仍在优化器里**（`trainer.py:93-98`） | PASS |
| H_where / F_pre 独立投影到 512 | `connector.py:151-152` `text_proj` / `vision_proj` 两个独立 `nn.Linear` | PASS |
| 输出头只产 w0/w_dir/alpha/rho，无 dense logits | `heads.py:87-142`：`AxisOutput`/`RhoOutput` 的输入都是 pooled `(B, dim)`，输出 `(B,73)`/`(B,n_rho)`；模块里不存在 canvas-token → 空间图的路径 | PASS |

补充确认（非任务卡点，但值得记）：`MultiheadAttention` 对全掩码行的处理（`connector.py:79-92`）——
把无有效 key 的行改成全有效再把输出置零，等价于"该分支不贡献"，避免 null 上下文 `0/0` → NaN 污染整 batch。
这是正确且非平凡的处理，有专门单测。

### 2. 四结构变体与参数量（§5.2/§5.3）— **PASS**

| 结构 | 规格 | 实现 | 判定 |
|---|---|---|---|
| MC8-Joint / MC16-Joint | 同一 attention pool + joint head | `config.py:63-64` `streams:1, pools:1`；`heads.py:160-164` 单 pool + 共享 trunk + 两个输出投影 | PASS |
| MC16-SplitHead | shared canvas，`w`/`rho` 独立 pool + 独立 head | `config.py:65` `streams:1, pools:2`；`heads.py:166-171` `pool_axis/pool_rho` + `trunk_axis/trunk_rho`；`model.py:118-119` 两条 canvas 指向同一 stream 输出 | PASS |
| MC16-DualCanvas | 两套独立 query bank/connector stream，只共享 frozen VLM | `config.py:66` `streams:2, pools:2`；`model.py:93-97` 两个 bank + 两个 `ConnectorStream`（投影/位置编码/6 block 全在 stream 内）；`w`-路与 `rho`-路无任何共享张量 | PASS |
| 8 臂 = 4×2 笛卡尔积 | §5.3 表 | `config.py:72-81` 逐行一致 | PASS |
| 参数量 W07/W08 ≈ 69.8M | — | 手工重算 = 69,835,365 / 69,851,781，**与 NOTES §3.3 逐位一致**；三档增量（+98,688 / +1,316,864 / +33,581,068）与结构差异吻合 | PASS |

### 3. Context 数据流（§5.4）— **B3 BLOCKER，其余 PASS**

| 条文 | 实现 | 判定 |
|---|---|---|
| 训练 batch 固定 50/50 | `context.py:231-274` `BalancedContextSampler`：micro-batch 内精确一半一半（奇数 micro-batch 直接抛错，`context.py:246-250`），因此对任何 GAS 都成立 | PASS |
| generated 缺闭合标签不回退 GT（**结构性证明**） | 三重：(a) `generated_context`（`context.py:123-161`）签名里**没有任何 GT 文本参数**，无值可回退；(b) `BatchBuilder.context_for`（`data.py:268-272`）在 `genctx is None` 时 `RuntimeError`，注释明写 "never fall back to GT"；(c) `GenContextStore.record` → `PublishedStore.read` 在样本缺失时 `KeyError`（`stores.py:73-74`），**不被任何 except 吞掉**（全包只有 2 处 `except Exception`，都在 `preflight.py` 的 env/键字探测里） | PASS |
| 固定边界截取 + 记录格式失败 | `context.py:139-161`：`</where>` 之前截断→`closed`；越界→`closed_over_boundary`+failure；无闭合→前 96 token+`no_close_tag`+failure；空→`empty`。`WHERE_CONTEXT_MAX_TOKENS=96` 由实测 2711 条 GT（local max 79 + 2 标签 = 81）导出，且 `gt_context` 对超界 GT **直接抛错而非截断**（`context.py:113-118`）——这条很对 | PASS |
| shuffled = 同图像同层级内 derangement | `context.py:178-219`：组内随机置换后循环移位 → 无不动点；单例组不跨图配对，记为 uncovered；`evaluate_context`（`evaluate.py:61-64`）跳过并计入 `n_skipped` | PASS（分组构造正确） |
| shuffled = **交换 instruction/where context** | 只换 where 正文，指令未换 | **B3 BLOCKER** |
| null 构造 | `context.py:164-165` 零 token；`WhereContext.__post_init__` 强制 null 不带 token；下游 `_pad_stack(..., min_len=1)` + 全 False mask → cross-attn 输出精确 0 | PASS |
| 四 context 分开报告，不混成均值 | `evaluate.py:153-163` 逐 context 各跑一遍全 split；`metrics.arm_metrics`（`metrics.py:160-189`）保留 `per_context` 并只从 `generated` 板读 gate 指标 | PASS |
| GT/generated 子批 loss 分别计算 | `losses.aggregate`（`losses.py:280-295`）按 `contexts` 分组给 `by_context` 的 loss 与全部标量；优化的标量仍是全 batch 均值（50/50 时等于两个子批均值的平均） | PASS |

### 4. `H_color` 隔离（§14 项 8）— **证明本身充分，但数值半从未执行（B1）**

| 半 | 实现 | 判定 |
|---|---|---|
| 签名白名单 | `preflight.py:211-224` 扫 `WhereBModel.forward` / `ConnectorStream.forward` / `ConnectorBlock.forward` / `LatentHeads.forward` 的参数名 | PASS |
| 标识符扫描 | `preflight.py:197-204` 用 `tokenize` 只取 `NAME` token（**剔除注释与字符串**，比裸 grep 强），扫 qwhere/connector/heads/model 四个模块 | PASS |
| 关键字被拒 | `preflight.py:239-247` 真调用 `model(..., h_color=...)` 断言 `TypeError` | PASS |
| 数值 bit-identical | `check_h_where_causal_independence`（`preflight.py:251-284`）逻辑正确：同图同指令同 `<where>`、只换 `<color>` 正文 → `H_where` 必须 `torch.equal` | **函数正确但零调用者（B1）** |

**绕过路径搜索结论**：我按四条可能的绕过路径逐一查过，未发现漏洞——
(a) `BatchBuilder` 造 `EncodeItem` 时 `where_ids=ctx.token_ids`，`_PromptShim.color_text="."` 只用于 `n_prompt_tokens`，
且 prompt 在第一个 `<where>` 之前就结束（`collator.py:139,145`），颜色正文进不了 prompt；
(b) `FrozenVLM.encode`（`hiddens.py:218`）切片 `hidden[i, n_p:n_p+n_w]`，右 padding，切片边界正确；
(c) `MODEL_INPUT_KEYS` 是硬编码五元组，`Batch.check_inputs`（`data.py:214-223`）拒绝任何额外键；
(d) `WhereBOutput.canvas_axis/canvas_rho` 虽然被导出（供 §6 的 `z_where`），但只来自 connector，不含颜色。
**唯一未被证明的是数值半——因为它从没跑过。**

### 5. Loss 逐符号（§5.5）— **PASS（数值口径见 B4，权重稀释见 B5）**

| 符号 | 规格 | 实现 | 判定 |
|---|---|---|---|
| `L_mask` 三项权重 | `1 / 0.25 / 0.10` | `config.py:115-117`，`losses.py:149` | PASS |
| `softIoU` | 未指定形式 | min/max（`losses.py:75`），D-B10 已申报，与 Where-A `soft_iou_minmax` 同定义 → loss/gate/oracle 比值三处同一个数 | PASS |
| `balanced_BCE` | 未定义 | 逐图 `w_pos=0.5/mean(t)`、`w_neg=0.5/(1-mean(t))`（`losses.py:91-95`）；t≡1 时负项恒 0，不会被 `w_neg` 放大 | PASS |
| `boundary_F1_loss_3px` | 未定义 | 逐符号照抄 arXiv:1905.07852（`losses.py:112-137`）：`pool(1-y,3)-(1-y)` → tol pool 7 → `1-2PR/(P+R)`。**出处已由实施者打开原文核实（V-B6），我复核了公式形状与论文一致** | PASS |
| — 边界退化 | — | `both_empty` 时返回 0 而非常数 1（`losses.py:136-137`），否则每个 global 样本白扛无梯度罚项 | PASS（正确的处理） |
| `L_s = Huber(s/3, s*/3)` | — | `losses.py:160` `F.huber_loss(s_pred/S_SCALE, s_star/S_SCALE)`，`S_SCALE=3.0` | PASS |
| `L_curve`，z=linspace(-3,3,257) | — | `config.py:129`；`losses.py:155-169`；`r_star` 由 `oracle_fields` 在**同一张 z 网格**上现算（`fields.py:218`） | PASS |
| `L_dir = 1 - cos` | — | `losses.py:172-175` | PASS |
| 两段 schedule 30%/70% | — | `losses.py:180-193`，切换点 `round(0.3·total)`，`step<boundary` 为 stage 1；权重组 `{1.00,1.00,0.10}` / `{0.25,0.25,0.05}`（`config.py:132-135`） | PASS |
| 「任何 loss 都同时在 GT/generated 子批上计算」 | — | `aggregate` 的 `by_context` | PASS |

补充确认：`w_dir` 的非辨识性没有制造问题——`w_dir_of` 不做符号规范化（`where/basis.py:31-32`），
但 `L_s` 直接对 `s*` 监督，符号被钉死；`L_dir` 只是 0.10/0.05 权重的辅助项。
`s_from_params`（`fields.py:152-153`）与 Where-A `basis.s_low`（`basis.py:118-119`）**逐符号一致**
（`q = w0 + alpha*(phi@w_dir)`，注意 `basis.py` 的模块 docstring 写成 `(w0+<phi,w>)*alpha` 是 Where-A 侧的文档笔误，代码是对的）。

### 6. generated-context 产出方案（§2.3 + §5.4 + D-B2）— **PASS（脚本本身见 B2）**

| 项 | 结论 |
|---|---|
| 缓存 token ids 重放同一 encode | **设计成立且是本包最好的一处**。`gencontext.py:1-30` 的论证正确：缓存 hidden 会让 teacher/generated 出自**两次不同的调用**，"同层同位置同归一化"就只能靠人守；缓存 ids 后两者共用 `FrozenVLM.encode`（`hiddens.py:157-223`），差别**只有 token ids**，口径一致变成结构性成立。`check_hidden_contract` 正是为此写的断言（可惜没跑，B1）。 |
| teacher/generated 口径一致 | `context_for` 的两条分支最终都汇入 `build()` 的同一个 `EncodeItem(prompt_ids=..., where_ids=ctx.token_ids)`，没有第二条路径。PASS |
| hidden 口径 = post-norm（D-B2） | `config.py:107,112` `WHERE_HIDDEN_LAYER=-1` + `WHERE_HIDDEN_FINAL_NORM=True`；`hiddens.py:203-204` `hidden = self.lm.norm(hidden)`。**代码与 D-B2 裁定、与 NOTES 文档三者一致**。且 teacher/generated 共用此函数 → 同口径。PASS（一个建议见 N1） |
| shard 契约符合性 | `publish_generated` → `q3vl.data.shardio.build_from_memory`（不压缩 tar、staging + fsync + 原子发布、sqlite catalog + `shard-*.idx.jsonl` 索引、checksum）。`GenContextStore` 读取前强制 `manifest.status == "complete"`（`stores.py:46-50`）→ 拒读非原子发布物。`test_stores.py` 用 Where-A **自己的 packer** 真打一套 shard 再读回，是真 interop 测试。PASS |
| shard 大小 | `GENCTX_SHARD_BYTES = 1 GiB`；genwhere 记录 ~500 B × 159k ≈ 80 MB → 只会有 1 个 shard，达不到 §2.3 的 1–4 GiB 目标。**这是数据量决定的，不是实现问题**，记为 N9 |
| 生成栈选择 | 用训练环境自己的 HF greedy，不用 vLLM。理由（另一套 torch/transformers、ids 可能不一致、给不出 hidden）**正确且保守**，且 V-B7 的可用性核实是真做过的。PASS |

### 7. Where-A 接口消费 — **PASS（两个 nit）**

| 项 | 结论 |
|---|---|
| import 复用无篡改 | `q3vl/where/` 全部 14 个 .py 的 mtime **早于** `q3vl/whereb/` 全部文件；Where-A 120 个测试全绿；`fields.py` / `stores.py` / `losses.py` 只 import 不改写。`make_oracle_latents.py` 明确说明"补 train 段而不编辑 `q3vl/where/`"，做法正确。PASS |
| oracle schema 一致性 | `OracleStore.latent`（`stores.py:102-122`）读 `payload["fits"][readout]["status"]/["latent"]` 并交给 **Where-A 自己的 `Latent.from_dict`** 反序列化。对照 `run_calibration.py:150-152` 的写入（`fit.to_dict()` 去掉 sample_id/meta/phi_diag）与 `FitResult.to_dict`（`oracle.py:220-235`），字段**逐项对得上**。PASS |
| 拒绝的拟合不填零 | `status != "ok"` → 返回 `None` → 该样本三个辅助 loss 被屏蔽而非零填（§10.2「不静默换成零向量」）。PASS |
| `s*` / `r*(z)` 现算不落盘 | `fields.oracle_fields` 用**同一张 `phi_dir`** 重算，杜绝"latent 是在旧 B 下拟的"这类静默错配。这是正确且非平凡的设计选择。PASS |
| `phi_dir_fast` 与 Where-A 逐位一致 | 有 `test_fields.py::test_phi_fast_matches_where_a_bit_for_bit`（3 种网格）钉死。PASS |
| interop mock 单测 schema 与实际产物结构一致 | `test_stores.py` 用 `pack_oracle`/`pack_maskviews`（Where-A 的 packer）真造 shard。**符合任务卡要求**。PASS |
| fit 复现口径 | `make_oracle_latents.py:108` 未传 per-sample `seed_offset`，而 Where-A `calibrate.fit_sample:121` 传 `seed_offset=j` → 见 N3 |
| `mask_low` 来源 | `make_oracle_latents.py:102-104` 从 `mask_hi` 重算而非直接读 Where-A 已发布的 `.masklow.npy` → 见 N4 |

### 8. 优化配置（§10.3）与 gate/选择（§5.6）— **PASS**

| §10.3 条目 | 值 | 实现 | 判定 |
|---|---|---|---|
| optimizer | AdamW | `trainer.py:99` | PASS |
| learning_rate | 2.0e-4 | `config.py:144` | PASS |
| weight_decay | 0.01 | `config.py:145`，`trainer.py:96` | PASS |
| warmup_ratio | 0.03 | `config.py:146`，`calibrate.make_scheduler:68` | PASS |
| scheduler | cosine | `calibrate.make_scheduler:73-76` | PASS |
| max_grad_norm | 1.0 | `trainer.py:197-198` | PASS |
| precision | bf16 | `trainer.py:150-154` | PASS（口径问题见 B4） |
| effective_batch_per_arm | 32 | `TrainConfig.grad_accum`（`config.py:272-278`）在 `32 % micro != 0` 时**抛错**而不是默默取整 | PASS |
| epochs | 1.0 | `BalancedContextSampler` 把数据集切成两个不相交的一半，每样本一个 epoch 只出现一次（D-B13） | PASS |
| eval_steps / save_steps | 500 / 500 | `config.py:151-152`，`trainer.py:220-223` | PASS |
| micro-batch 先探测再定 GAS | — | `probe_micro_batch`（`trainer.py:266-292`）在 {2,4,8} 上试，OOM 即停 | PASS |
| 单卡一臂 | — | `run_where_b.py --device cuda`；`run_where_b.sh` 不锁卡 → N7 |
| **checkpoint 选择禁用 val loss（红线）** | — | `WhereBTrainer.best`（`trainer.py:254-263`）按 `local_soft_iou_median` 选，`eval_loss` 只记录；单测构造了"eval_loss 单调下降但指标峰值在中间"的场景 | PASS |

§5.6 九项 gate：`config.py:161-171` 与协议表**逐行一致**（阈值、方向、指标名全对）。
`SELECTION_ORDER`（`config.py:173-180`）= 「median soft-IoU → boundary F1 → p10 → 参数量/显存/延迟」，与 §5.6 一致。
`lexicographic_best`（`metrics.py:215-245`）逻辑正确：缺失值一律排最后（`larger_is_better` 时取 `-v`，缺失取 `+inf`）；
**不过门不淘汰候选，只给最优者打 `WHERE-GATE-FAILED`**，与 §5.6 末段一致；`evaluate_gates` 对缺失指标判 fail 而非 pass（`metrics.py:198-200`）——方向正确。

---

## 三、NIT（不阻断，但建议在 S5 之前顺手处理）

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| N1 | `config.py:107-112` | D-B2 裁定的是"**全局统一口径**（含未来 Stage-What 的 `H_color`）"，但 `WHERE_HIDDEN_LAYER/WHERE_HIDDEN_FINAL_NORM` 定义在 `q3vl/whereb/config.py`，Stage-What 只能靠自觉 import 或重新声明 | 提到 `q3vl/train/constants.py` 或新建共享模块，让 Stage-What **无法**静默分叉 |
| N2 | `run_where_b.py:108-119` | 只查了 `oracle_coverage`（前 2000 条），**没查 genctx 覆盖率**。`BalancedContextSampler` 是先按索引分池、后取 genwhere 记录，一条缺失就会在训练几小时后炸 `KeyError` | 在 `setup` 里加 `assert set(generated_pool ids) <= genctx.sample_ids`，缺失即启动时失败 |
| N3 | `make_oracle_latents.py:108` | 未传 per-sample seed（Where-A 传 `seed_offset=j`），全 split 用同一个多起点种子；NOTES 却称"由产生 V_where 那批的同一套代码产生" | 传 `FitConfig(seed=base+i)`，与 `calibrate.fit_sample:120` 对齐 |
| N4 | `make_oracle_latents.py:102-104` | 从 `mask_hi` 重算 `mask_low`；若 `mask_hi` 来自已发布的 `.maskhi.png`（uint8），会先被量化到 1/255 再降采样，与 Where-A 的 float 路径有微差 | 直接读 `MaskViewStore.mask_low`（Where-A 已发布 `.masklow.npy`） |
| N5 | `evaluate.py:85-94` | `met.update({**{k: tgt["meta"].get(k) for k in STRATA_KEYS}})` 会把 `render_mode` 写成 `None`（键已存在），随后的 `met.setdefault("render_mode", ...)` **修不回来**；`summarise` 的 `is_local`/`is_global` 都会判 False → 该行**从 local 与 global 两个聚合里同时消失**。当前所有 record 都带 `render_mode`（实测），所以是潜伏 bug | 改成显式赋值 + `assert met["render_mode"] in ("local","global")` |
| N6 | `metrics.py:181-184` | `instruction_shuffle_iou_drop = generated_median − shuffled_median` 把"generated↔GT"与"本人↔partner"两个轴混在一起 | 改成 `gt_median − shuffled_median`（两边都是 GT 文本，只差 partner），或至少同时报两个版本 |
| N7 | `run_where_b.sh:24-35` | `submit` 不接受 GPU 参数、不设 `CUDA_VISIBLE_DEVICES`；按 usage 连开两臂会都落在 GPU 0。另：`( cd ... && nohup ... & echo $! )` 记录的可能是包装子 shell 的 PID 而非 python 的；step 3 的 `tail` 只**打印**日志、不**校验**是否有实质内容，空日志也会 return 0 | 加 `submit <gpu> <log> <cmd...>`；`ps -p $pid -o pid,cmd` 打印出来人工确认；step 3 改成 `grep -q '"arm"' "$log"` 之类的实质断言（D-20 第 3 步） |
| N8 | `data.py:256, 292` | `BatchBuilder.format_stats` 跨多次 `evaluate_arm` 调用**累加**；每 500 步一次 eval 复用同一个 `eval_builder`，于是 `metrics.json` 里的 `format_stats` 是累计值而非本次 eval 的 | eval 开始时 reset，或改成按 (step, mode) 分桶 |
| N9 | `config.py:197` | `GENCTX_SHARD_BYTES = 1 GiB`，但 genwhere 全量只有 ~80 MB → 单 shard，达不到 §2.3 的 1–4 GiB 目标 | 无需改代码，在报告里说明"数据量决定"即可 |
| N10 | `metrics.py:75-96` | `AUC_target` 定义（预测 mask 作为 score、GT 在 0.5 二值化的 ROC AUC）是**协议未定义**的量，NOTES §四没申报 | 补进 NOTES 的决策清单；实现本身（含 tie 平均秩、单类返回 `None`）是正确的 |
| N11 | `metrics.py:139-142` | `if r.get("oracle_soft_iou")` 用真值判断，`oracle_soft_iou == 0.0` 的样本会被静默排除出比值统计 | 改成 `is not None and > 0` 并单独计数被排除的样本 |
| N12 | `trainer.py:242-252` | 无滚动删除；159,215/32 ≈ 4,975 步 → 每臂 ~11 个 checkpoint。W07/W08 各 ~280 MB → 单臂 3.1 GB，8 臂 ~20 GB | 按 §10.4 的惯例保留 3 个 + 保护 0.5/1.0 epoch，或确认磁盘够 |
| N13 | 仓库层面 | `q3vl/` **整个目录未入 git**（`git status` 只显示 `?? q3vl/`），因此本次审阅无法用 diff 证明"Where-A 未被篡改"，只能靠 mtime + 测试全绿旁证 | 建议在 S5 之前把 `q3vl/` 入库并打 tag；否则 §13.1 要求的"config/ 快照 + git commit"对模型代码本身是空的 |

**排期数字（供主 agent 排 wave 用，非缺陷）**：train 159,215 样本 / effective batch 32 = **4,975 optimizer steps/臂**。
每 500 步一次四上下文全量 eval = 4 × 896 = 3,584 次编码，10 次 eval ≈ 35,840 次 → 约为训练前向量的 **22.5%**。
按 §11 一卡一臂、8 臂 4 个 wave，这部分开销不可忽略，`--eval-limit` 会违反 §5.4 的"每个 checkpoint 四上下文全报"，
建议按原样跑但把 eval 墙钟单独计入排期。

---

## 四、对主 agent 已有裁定的意见

| 裁定 | 我的意见 |
|---|---|
| **D-B1**（softIoU 进 loss 与选择，红线判旧战役语境） | **同意保留协议字面**，但请把一个后果写进 REPORT 的"设置"节：`local_soft_iou_median` 同时是**主损失的支配项**和**选择规则的第一顺位**，因此「median soft-IoU ≥ 0.75」这条 gate 实质上只在回答"训练收敛了吗"，**不是独立检验**。归因重量必须落到那些**没有**被直接优化的量上：`boundary_f1`（loss 里只有 0.10 权重）、`p10`、`auc_target`、以及 null/shuffled 两个 Δ。另：`soft_iou_vs_oracle_ratio` 仍然可信——分子分母都由同一目标优化，比值是公平比较。 |
| **D-B2**（post-norm，全局统一） | **同意**，实现与裁定、文档三者一致，且 teacher/generated 共用同一函数。唯一补充见 N1：常量放在 whereb 包内，Stage-What 有静默分叉的空间。 |
| **D-B5**（global 进训练、oracle 辅助关闭） | **同意 global 进训练**（否则 `global soft-IoU ≥ 0.98` 这条 gate 不可达）。但请注意 **B5**：这条裁定的**未申报副作用**是 §5.5 的 stage-1 辅助权重被打到 ~0.47 倍，而 stage-1 存在的全部理由就是"用大权重压住早期坍缩"。需要一次追加裁定（batch 平均 vs 有-oracle 样本平均）。 |
| **D-B11**（train 段 oracle latent 为硬前置，排进 GPU 关键路径） | **同意，且提高紧急度**：`make_oracle_latents.py` 目前**跑不起来**（B2）。请在排期前先要求实施者用 `--limit 8` 真跑一次通，再外推墙钟——PENDING S1 的"墙钟未知，必须先小规模实测"是对的，但前提是脚本能跑。 |
| **D-B3/B4/B6-B10/B12-B14**（保守默认） | 除 D-B6/D-B7 涉及的 shuffled 语义（**B3**）外，其余默认我都认为合理，无异议。特别地 D-B3（缓存 ids 而非 hidden）是本包最好的一个设计判断，D-B4（hi-res 算 L_mask）的理由（3px 容差在 32×48 网格上无意义）也站得住。 |

**新增待裁决项**（本次审阅提出，不在 NOTES §四里）：

1. **D-B15**：shuffled 是否同时交换 prompt 里的 instruction（B3）。若维持"只换 where 正文"，§5.6 第 6 行 gate 必须改名并重新预注册判据。
2. **D-B16**：oracle 辅助项的归一化分母是 batch 大小还是"有 oracle 的样本数"（B5）。

---

## 五、判决（**已被第 2 轮取代**，见本文件开头与「第 2 轮 · 聚焦复审」§2.8）

**BLOCKER 数量 5**（B1 preflight 空转 / B2 两个前置脚本崩溃 / B3 shuffled 未换指令 / B4 autocast 漏进 `s_low` / B5 辅助权重被稀释未申报）。

**是否准许进入 GPU preflight 与正式训练：否。**

- **S1/S2（oracle latent、generated context）**：B2 未清，脚本第一个样本即崩，**不得提交**。
- **S3（§14 项 7b/8b preflight）**：B1 未清，当前 `--with-model` 是空转并报 PASS，**跑了等于没跑**，不得据此放行。
- **S5（8 个主臂）**：B3/B4/B5 均属"跑完再改 = 全部重跑"的类别（预注册 gate 语义、受控对比的数值口径、预注册 loss 权重），**必须在开跑前清完**。

**放行条件**：B1、B2、B4 修复并附复现证据（B2 需 `--limit 8` 的真实跑通日志，B4 需给出 `s_low.dtype == torch.float32`
的训练期断言）；B3、B5 由主 agent 出裁定并落到代码与 NOTES；然后重跑 CPU preflight + 170 单测 + 真实 GPU preflight（含 7b/8b），
再进入 S1→S2→S3→S5。

**必须说明的正面结论**（避免以上 blocker 掩盖实现质量）：结构层（connector 四步顺序、zero-init gate、四结构装配、
参数量表、全掩码 softmax、只出全局参数无 dense logits）、loss 逐符号、九项 gate 与 lexicographic 选择、
禁回退 GT 的结构性设计、缓存 ids 重放同一 encode 的口径设计、以及与 Where-A 的 interop（用对方 packer 真打 shard）
**全部经得起逐条对照**，170 + 120 个测试真实全绿，参数量表可独立复算对上。
五个 blocker 集中在**"声明与实现不符"和"没跑过的路径"**这两类，不是设计错误。
