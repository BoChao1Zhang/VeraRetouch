# GPU 任务队列 · 本战役使用说明

> **队列系统本身已独立成仓库：`/home/bc/agent-gpu-queue`（QUEUE-2，2026-08-10）。**
> **通用手册（全部动词、事件日志、模板、自检）在那边的 `README.md`，本文只留本战役专属内容**，
> 免得同一件事写两处然后漂移。
>
> `tools/queue` 现在是指向该仓库的**目录 symlink**，所以旧路径
> `tools/queue/q`、`tools/queue/waves/where_b_arm.sh` 全部照常可用（已实测），
> 队列里在跑/在排的六条臂命令行未作任何改动。

---

## 0. 一条命令拿全景（会话断了三天也够用）

```bash
tools/queue/q status            # 人读 Markdown（约 0.5s）
tools/queue/q status --json     # 机读 JSON
tools/queue/q events --since 2h # 我不在的时候发生了什么
tools/queue/q doctor            # 有没有东西坏了
```

`q status` 给出：每卡当前任务 / pueue id / 载荷 PID / 已运行时长 / 阶段 / GPU 实测利用率、
排队列表（含 gate 满足情况与 `--desc`）、已完成与失败任务的 rc 与失败签名、
最近事件、以及一段 **Attention** 列表（卡空了、组被暂停、gate 卡住、任务"在跑"但 GPU 0%）。

**`q events` 是 QUEUE-2 新增的、给 agent 的收件箱**：没有任何机制能中断一个 Claude 会话去通知它
"W03 半夜挂了"，所以每次状态跃迁都追加一行 JSON，事后可按时间/按名字增量读。
所有退出路径都汇成唯一的 `end` 事件，`phase` 区分 `done` / `failed` / `start-failed` /
`gate-timeout` / `killed`。

```bash
# 今天所有失败
tools/queue/q events --all --json | jq 'select(.event=="end" and .rc != 0)'
# 半小时内有没有臂跑完
tools/queue/q events --since 30m --json | jq -r 'select(.event=="end") | "\(.name) \(.phase) rc=\(.rc)"'
```

> **改序会改 id**（`pueue switch` 交换的是两个任务的 id，实测）。所以**报告里一律写臂名不写 id**，
> 所有动词也都支持按名字寻址：`q cancel W07`、`q logs W05 200`。

---

## 1. 当前在队的六条臂（Where-B W2–W4）

| 卡 | 队列顺序 | 队头的产物 gate |
|---|---|---|
| gpu0 | W03 → W05 → W07 | `where_b/arm_W01.json` + checkpoint-4976 |
| gpu1 | W04 → W06 → W08 | `where_b/arm_W02.json` + checkpoint-4976 |

六臂均以 `MICRO_BATCH=8` 入队（与 EXEC-3 的 W01/W02 相同），八臂配置完全同构可比。

- 每卡一组、并行度 1，所以卡内先后天然成立，**不需要**声明依赖；
  某臂失败也**不会**挡住它后面的臂（一条臂死了不该赔上一整天的卡）。
- 队头的 gate 盯的是**产物**（`arm_W0x.json` 由 `run_where_b.py` 在最后写出），不是日志字符串。
  gate **等待**而非快速失败（上限 48h），所以 W1 一落地，卡在秒级换手。
- **gate 不碰 NFS**：oracle latent / maskview / genctx 都在 `/mnt/nfs`（hard 挂载），
  对 hard 挂载做 `test -e` 在服务端失联时会永久 D 状态且杀不掉。这些前置**没有被跳过**——
  `run_where_b.py` 自己会断言（`assert_genctx_coverage`、OracleStore coverage、`load_basis`），
  在作业内部失败，可见且可杀。

> **迁移遗留（会自愈，不用管）**：W03/W04 的包装器进程是 2026-08-10 11:49 起的，
> 跑的是 QUEUE-1 版 `qjob.sh`（进程持有的是迁移前那个 inode），
> 因此**这两条臂不会写事件**，只有状态 json。W05–W08 起在新包装器上，事件齐全。
> 迁移前的整份旧代码备份在 `/home/bc/data/queue/backup/tools-queue-preQUEUE2/`，
> W03/W04 结束后可删。

### 一个 wave 内的 micro batch

`run_where_b.sh` 写明：一个 wave 的两条臂必须钉同一个 `--micro-batch`，否则
`BalancedContextSampler` 切出的 `len(sampler)`、`total_optimizer_steps`、LR schedule 都变，
两条臂就不再是 protocol 11 要的配对比较。

`where_b_arm.sh` 的做法是**抢角色而不是派角色**：`mkdir` 锁目录是原子的，**先起来的那条臂**
成为 prober，probe 完把值发布到 `runs/.wave_<w>.micro_batch`，另一条读它并钉同一个数。
若 prober 启动阶段就死了，另一条臂**拒绝运行**（rc=80）而不是偷偷跑一个不配对的比较。
本次六臂显式钉了 8，整个协调机制被绕过。

重新入队某个 wave 前，**先清 `.wave_<w>.{prober.lock,micro_batch}`**（入队脚本已内置）：
被 kill 的臂会留下没人释放的锁，下次同一臂会退化成等一个永不 probe 的伙伴，
白等满 `--pair-timeout` 再 rc=80。**已实际踩到过一次。**

---

## 2. 排后续波次（What T/C 臂等）

照 `/home/bc/agent-gpu-queue/templates/wave.sh` 抄，一条波次约 12 行：

```bash
. /home/bc/agent-gpu-queue/wavelib.sh
wave_defaults --ready 'total_optimizer_steps' --ready-timeout 7200 \
              --gate "${CKPT}" --gate-timeout-hours 48 --poll 30
wave_arm T1 0 "${RUNS}/T1/train.log" --desc 'Stage-What T1' -- bash "${ARM}" T1
wave_arm T2 1 "${RUNS}/T2/train.log" --desc 'Stage-What T2' -- bash "${ARM}" T2
wave_submit "$@"     # --dry-run / --hold 白送
```

单条任务用 `q submit`：

```bash
tools/queue/q submit T1 0 /home/bc/data/runs/what/T1/train.log \
  --desc 'Stage-What T1' \
  --gate /home/bc/VeraRetouch/experiments/.../where_selection.json \
  --ready 'total_optimizer_steps' --ready-timeout 7200 \
  -- bash /home/bc/agent-gpu-queue/waves/what_arm.sh T1
```

> **载荷不要自己 setsid / nohup**。`run_where_b.sh train` 这类带自有 `submit` 的封装会把训练
> 甩进新会话：pueue 槽位瞬间释放（两卡看起来都空着），而且 `q cancel` 再也停不掉它。
> 队列下面直接 exec python 模块即可，D-20 由 `qjob.sh` 负责——
> `waves/where_b_arm.sh` 就是这么写的，照抄它（或 `templates/arm.sh`）。

---

## 3. 本战役的失败处置

```bash
tools/queue/q status                # Attention 段直接点名失败任务与签名
tools/queue/q logs <臂名> 200       # 看载荷日志
tools/queue/q events --name W05     # 这条臂的完整经过
```

| 现象 | 含义 | 处置 |
|---|---|---|
| `phase=gate-wait` 且长期不动 | 前置产物没落地 | 看 `gates[].ok` 哪个 false；上游若已判死，`q cancel` 让出卡 |
| `rc=78` | gate 超时 | 上游没产出。修好后 `q retry <臂名>` |
| `rc=79` | 起不来 / 起来了不出声 | 看日志前几十行；多半是数据/权重路径 |
| `rc=80` | wave 伙伴没发布 micro batch | 确认伙伴实际用值后，用 `MICRO_BATCH=N` 重新入队该臂 |
| `cuda_oom` 签名 | 显存不够 | 降 `--micro-batch` 重入队；**同 wave 两臂必须一起改** |
| `state=Running` 但 GPU 0% 且已久 | 卡死/空转 | Attention 会自动报；`q logs` 看最后输出 |
| `daemon_ok=false` | 守护进程没了 | `q daemon`；排队任务会恢复，**正在跑的那批已经死了**，需 `q retry` |

**拆队列前先 `q pause gpu0 gpu1`**：删掉队头会**立刻**把后面的臂放上卡（队列本来就该这样），
实测演练中 W05/W06 确实在删 W03/W04 的同一秒抢到卡并真的开跑。

---

## 4. 路径速查

| 东西 | 位置 |
|---|---|
| 队列系统（独立仓库，含通用手册 README.md） | `/home/bc/agent-gpu-queue`（= `tools/queue` symlink） |
| 波次脚本 | `/home/bc/agent-gpu-queue/waves/` |
| 新波次模板 | `/home/bc/agent-gpu-queue/templates/{wave,arm}.sh` |
| 每任务状态 json | `/home/bc/data/queue/status/<臂名>.json` |
| **事件日志** | `/home/bc/data/queue/events.jsonl` |
| 载荷日志 | Where-B 为 `/home/bc/data/runs/where_b/<ARM>/train.log` |
| `job.marker` | 载荷日志同目录 |
| pueue 状态/日志 | `~/.local/share/pueue/{state.json,task_logs/,log/}` |
| 迁移前旧代码备份 | `/home/bc/data/queue/backup/tools-queue-preQUEUE2/` |
| wave micro batch 锁与发布值 | `/home/bc/data/runs/where_b/.wave_<w>.{prober.lock,micro_batch}` |
