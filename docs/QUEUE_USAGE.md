# GPU 任务队列使用手册（QUEUE-1）

面向**主 agent**。目标：两张 H100 在队列非空时永不空转，且任何一个新的 Claude 会话
（哪怕上一个会话已经断了三天）执行**一条命令**就能拿到全景。

---

## 0. 一条命令拿全景

```bash
tools/queue/q status            # 人读 Markdown（0.4s）
tools/queue/q status --json     # 机读 JSON
```

输出包含：每卡当前任务 / pueue id / 载荷 PID / 已运行时长 / 阶段 / GPU 实测利用率、
排队列表（含前置 gate 满足情况）、已完成与失败任务的 rc 与失败签名、以及一段
**Attention** 列表（卡空了、组被暂停、gate 卡住、任务"在跑"但 GPU 0%）。

会话断了要恢复工作，只需要这一条命令。**不需要**去翻日志、翻 job.marker、翻 nvidia-smi。

---

## 1. 选型结论：为什么是 pueue，不是自写

调研了任务卡点名的四个候选，判据为「单机 / 轻量守护 / 结构化状态 / 可增删改序 / 失败可见 / GPU 可绑定」：

| 候选 | 结构化状态 | 增删改序 | 守护/脱离会话 | 结论 |
|---|---|---|---|---|
| **pueue 4.0.4** | `status --json` / `log --json` | `add/remove/switch/stash/enqueue/restart` 齐全 | `pueued -d`，状态落 `state.json`，重启后队列与分组完整恢复（已实测） | **采用** |
| task-spooler (tsp) | 只有纯文本表，无 JSON | 有 | 有 | 否决：状态非结构化，与「信息对 agent 友好」直接冲突 |
| GNU parallel | 无 | 无 | 无常驻查询接口 | 否决 |
| simple_gpu_scheduler | 无 | 无（stdin 喂命令，不可改序/删除） | 无 | 否决 |

选 pueue 还因为它比自写省掉的正是难写对的部分：崩溃安全的状态文件、分组并行度、
依赖、以及一个真正脱离终端的守护进程。**实测过的行为**（本机，v4.0.4）：

- 组 `gpu0`/`gpu1` 各 `parallel=1` → "两卡满载" 等价于 "两个组各有一个 running"；
- 某任务失败**不阻塞**同组后续任务（配置 `pause_group_on_failure: false`，已确认）；
- `--after` 依赖：前置失败时后继标记 `DependencyFailed` 并跳过，不会吊住队列；
- `pueue shutdown` 再起，**分组、并行度、排队任务全部恢复**。

安装物：两个静态 musl 二进制 `~/.local/bin/pueue`、`~/.local/bin/pueued`
（GitHub release v4.0.4，无动态依赖，未装任何系统包）。

### 本仓库补的三件事

pueue 不管的，`tools/queue/` 管：

1. **`q`** — 每卡一组、并行度 1；提交时用 `env -i` + 白名单清洗环境。
   （`pueue add` 会把提交 shell 的整个环境快照进任务并写进 `state.json`，
   本机那意味着十几个 API key 落盘并被 `status --json` 原样吐回来。已实测清洗后为 0 命中。）
2. **`qjob.sh`** — 每个载荷的包装器：产物 gate + D-20 + 失败签名扫描。
3. **`qstatus.py`** — 把 pueue 视角、qjob 视角、nvidia-smi 三方合并成上面那份全景。

---

## 2. 启动 / 修复守护进程

```bash
tools/queue/q daemon        # 幂等：已在跑就什么都不做
```

它会：拉起 `pueued`（`setsid nohup`，不随本终端/本会话死）、用 `ps -p` 实证存活
（**不用 pgrep**）、建好 `gpu0`/`gpu1` 两组并各设 `parallel=1` 且置为 running。

- 守护进程 socket 已固定到 `~/.local/share/pueue/`（改了 `shared.runtime_directory`）。
  默认的 `/run/user/1001` 是 tmpfs，且用户会话全部退出后可能被 systemd-logind 清掉
  —— 无人值守跨多天正好会踩到。开机自起也依赖这条（`@reboot` 时还没有登录会话）。

### 开机自起（已装，无需再操作）

```bash
tools/queue/install_autostart.sh            # 幂等安装 + 自验
tools/queue/install_autostart.sh --check    # 查看 linger / enabled / active
tools/queue/install_autostart.sh --remove
```

装的是 **systemd 用户服务**（`~/.config/systemd/user/pueued.service`）+ `linger`，**不是 crontab**：

> 本机 `/var/spool/cron/crontabs` 是 `drwx-wx--T root:crontab`，且 `crontabs/bc` 不归 `bc` 所有。
> `crontab -` 靠"写临时文件再 rename 覆盖"生效，粘滞位禁止覆盖非自己所有的文件，于是报
> `crontab: crontabs/bc: rename: Operation not permitted`，而本机 `sudo` 要密码 —— 无人值守装不上。
> `loginctl enable-linger bc` 则被 polkit 允许自助执行（实测 rc=0、`Linger=yes`），
> 用户级 unit 因此能在无人登录时随开机启动。
>
> 若日后仍想用 cron：`tools/queue/install_autostart.sh --cron-fallback` 会打印所需的那一条
> root 命令（`sudo chown bc:crontab /var/spool/cron/crontabs/bc`）与对应 crontab 行。

unit 取自 pueue v4.0.4 release 里的官方 `systemd.pueued.service`，只改两处：二进制路径指向
`~/.local/bin/pueued`（我们按用户装），以及 `Restart=on-failure` 取代上游的 `Restart=no`
—— 半夜挂掉的守护进程应该自己回来，而 `state.json` 是权威状态，重启无损。
**因此它比原计划的 `@reboot` 更强：不只覆盖重启，还覆盖守护进程中途死亡。**

安装脚本在有任务 `Running` 时**拒绝**接管（接管要先停旧守护进程，会杀掉在跑的训练），
只安装并 enable，留到下次重启生效；要立刻接管用 `FORCE_TAKEOVER=1`。

---

## 3. 状态契约

### Markdown（默认）

章节固定：`Attention` → `Cards` → `Queued` → `Finished` → `Log tails`。
`Log tails` 只对 **running** 和**最近 3 个失败**的任务展开，默认每个 3 行（`--tail N` 调整）。

### JSON（`--json`）

顶层键：

| 键 | 含义 |
|---|---|
| `daemon_ok` / `daemon_error` | 守护进程是否可达 |
| `warnings` | 与 Markdown 的 Attention 同源的字符串数组 |
| `groups` | `{gpu0: {parallel, status}, ...}` |
| `gpus` | `[{index, util_pct, mem_used_mib, mem_total_mib}]`（nvidia-smi 实测） |
| `tasks[]` | 见下 |

`tasks[]` 每项：`id`（pueue id，增删改序都用它）、`label`（= 臂名/任务名）、`group`、
`state`（Queued/Running/Done/Stashed）、`result`（Success/Failed/DependencyFailed/Killed）、
`rc`、`elapsed_s`、`phase`（qjob 阶段：`gate-wait`/`starting`/`running`/`done`/`failed`/`gate-timeout`）、
`pid`（**载荷真实 PID**，非包装器）、`log`（载荷自己的日志路径）、`note`、
`gates[]`（`{path, ok}`）、`failure_signatures[]`（`{signature, line}`）、`log_tail[]`（末 30 行）。

常用查询：

```bash
# 现在两张卡各在跑什么
tools/queue/q status --json | jq -r '.tasks[]|select(.state=="Running")|"\(.group) \(.label) \(.phase) pid=\(.pid)"'
# 有没有失败
tools/queue/q status --json | jq -r '.tasks[]|select(.result!=null and .result!="Success")|"\(.label) rc=\(.rc) \(.failure_signatures[]?.signature)"'
# 还剩几个没跑
tools/queue/q status --json | jq '[.tasks[]|select(.state=="Queued")]|length'
```

---

## 4. 入队

### 4.1 Where-B W2–W4（已入队）

```bash
tools/queue/waves/enqueue_where_b_w2_w4.sh --dry-run        # 先看要提交什么
MICRO_BATCH=8 tools/queue/waves/enqueue_where_b_w2_w4.sh    # 当前在用：钉死 micro batch=8
tools/queue/waves/enqueue_where_b_w2_w4.sh                  # 不钉，走 wave 内自动协调
```

> **现状**：六臂已按主 agent 裁定以 `MICRO_BATCH=8` 入队（与 EXEC-3 的 W01/W02 相同），
> 八臂配置完全同构可比，§4.2 的 wave 协调机制被整体绕过。

入队脚本会先**清掉本次要提交的 wave 的协调残留**（`.wave_<w>.prober.lock` / `.micro_batch`）。
这不是洁癖：被 kill 掉的臂会留下一把没人释放的锁，下次同一臂再跑就会退化成
"等一个永远不会 probe 的伙伴"，白等满 `--pair-timeout` 再以 rc=80 拒跑。**已实际踩到过一次。**

排布（依 protocol 的 Wave 表：奇数臂 GPU0、偶数臂 GPU1）：

| 卡 | 队列顺序 | 队头的产物 gate |
|---|---|---|
| gpu0 | W03 → W05 → W07 | `where_b/arm_W01.json` + checkpoint-4976 |
| gpu1 | W04 → W06 → W08 | `where_b/arm_W02.json` + checkpoint-4976 |

- 每卡一组、并行度 1，所以卡内先后天然成立，**不需要**声明依赖；
  某臂失败也**不会**挡住它后面的臂（这是要的行为：一条臂死了不该赔上一整天的卡）。
- 队头的 gate 盯的是**产物**（`arm_W0x.json` 由 `run_where_b.py` 在最后写出），不是日志字符串。
  gate **等待**而非快速失败（默认上限 48h），所以 W1 一落地，卡在秒级换手。
- **gate 不碰 NFS**：oracle latent / maskview / genctx 都在 `/mnt/nfs`（hard 挂载），
  对 hard 挂载做 `test -e` 在服务端失联时会永久 D 状态且杀不掉。这些前置**没有被跳过**——
  `run_where_b.py` 自己会断言（`assert_genctx_coverage`、OracleStore coverage、`load_basis`），
  在作业内部失败，可见且可杀。

### 4.2 一个 wave 内的 micro batch（重要）

`run_where_b.sh` 写明：**一个 wave 的两条臂必须钉同一个 `--micro-batch`**，否则
`BalancedContextSampler` 切出的 `len(sampler)`、`total_optimizer_steps`、LR schedule 都变，
两条臂就不再是 protocol 11 要的配对比较。

两条臂同时起在不同卡上，没法各自 probe 还碰巧一致。`where_b_arm.sh` 的做法是
**抢角色而不是派角色**：`mkdir` 锁目录是原子的，**先起来的那条臂**成为 prober，
probe 完把自己 `run_setup.json` 里的值发布到 `runs/.wave_<w>.micro_batch`，
另一条读它并钉同一个数。两卡进度怎么错开都不会死锁。

若 prober 启动阶段就死了、始终没发布，另一条臂**拒绝运行**（rc=80）而不是偷偷跑一个
不配对的比较——空一张卡可以补，悄悄作废的 wave 补不回来。队列会继续往下走。

> EXEC-3 的 W1 波用的是显式 `--micro-batch 8`。若要 W2–W4 沿用同一个数，
> 用 `MICRO_BATCH=8 tools/queue/waves/enqueue_where_b_w2_w4.sh`，这会完全绕开上面的协调机制。

### 4.3 通用入队（后续 Where 选型 / Stage-What T1–T4、C1–C2 / top-2 复跑）

```bash
tools/queue/q add <NAME> <GPU> <LOG> [选项] -- <命令...>
```

- `<NAME>`：任务名，同时是 pueue label 和状态文件名 `<QUEUE_HOME>/status/<NAME>.json`。**保持唯一**。
- `<GPU>`：`0` | `1` | `-`（`-` = 不占卡，进 `default` 组）。qjob 会据此设 `CUDA_VISIBLE_DEVICES`。
- `<LOG>`：载荷自己的日志路径（会被 `rm -f` 后重建）。

选项：

| 选项 | 作用 |
|---|---|
| `--gate <路径>` | 前置产物；可重复。**只用本地路径，禁止 /mnt/nfs** |
| `--gate-cmd '<shell>'` | 任意断言，退出 0 视为放行；可重复 |
| `--gate-timeout-hours N` | gate 等待上限（默认 24），超时 rc=78 并让出卡 |
| `--ready '<正则>'` | D-20 第 3 步：日志里出现它才算真起来 |
| `--ready-timeout S` | 起来的上限（默认 1800）；超时杀掉并判失败，绝不谎报成功 |
| `--poll S` | gate 轮询与状态刷新间隔（默认 60；载荷存活检测恒为 5s） |
| `--after <id>` | pueue 级依赖（跨卡时才需要；同卡靠队列顺序即可） |
| `--stashed` | 入队但挂起，之后 `q enqueue <id>` 放行 |

例：

```bash
tools/queue/q add T1 0 /home/bc/data/runs/what/T1/train.log \
  --gate /home/bc/VeraRetouch/experiments/.../where_selection.json \
  --ready 'total_optimizer_steps' --ready-timeout 7200 \
  -- bash tools/queue/waves/where_b_arm.sh ...   # 换成 What 的启动脚本
```

> **载荷不要自己 setsid / nohup**。`run_where_b.sh train` 这类带自有 `submit` 的
> 封装会把训练进程甩进新会话：pueue 的槽位会瞬间释放（两卡看起来都空着），
> 而且 `pueue kill` 再也停不掉它。队列下面直接调 python 模块即可，
> D-20 由 `qjob.sh` 负责——`tools/queue/waves/where_b_arm.sh` 就是这么写的，照抄它。

---

## 5. 调整队列

```bash
tools/queue/q ls                # pueue 原生表（含完整命令行）
tools/queue/q first <id>        # 插队到本卡队首
tools/queue/q swap <id> <id>    # 交换两个排队位置
tools/queue/q rm <id> [<id>...] # 删除排队任务
tools/queue/q kill <id>         # 停掉正在跑的（载荷随之被杀）
tools/queue/q retry <id>        # 原地重跑（失败臂改完就用它）
tools/queue/q stash <id>        # 暂时挂起  /  q enqueue <id> 放回
tools/queue/q pause  [gpu0]     # 暂停一张卡（不影响正在跑的）
tools/queue/q resume [gpu0]
tools/queue/q log <id> [N]      # tail 载荷自己的日志（不是包装器输出）
tools/queue/q clean             # 从列表里清掉已完成任务
tools/queue/q raw <任意 pueue 子命令>
```

**改序只对 `Queued` 生效**。已经 `Running`（含 `gate-wait`）的任务要让路，先 `q kill <id>` 再重新入队。

---

## 6. qjob 替你保证的东西

每个载荷都被 `tools/queue/qjob.sh` 包着，顺序是：

1. **产物 gate**：所有 `--gate` 路径存在（文件还需非空）且所有 `--gate-cmd` 退出 0，才继续；
   否则每 `--poll` 秒重试，并把 `phase=gate-wait` + 卡在哪个文件写进状态 json。
2. **D-20 ①** `rm -f` 日志（zsh 的 noclobber 会让 `> 已存在日志` 整条重定向失败，进程压根不起且外面看不出来）。
3. **D-20 ②** `ps -p $PID` 实证存活（**全程不用 pgrep**）。
4. **D-20 ③** 等 `--ready` 正则在日志里出现；进程中途死了或超时都判失败，**不谎报成功**。
5. **D-20 ④** 以上都过，才写 `job.marker`（就在日志同目录，含 PID / GPU / 完整命令 / 状态 json 路径）。
6. **收尾**：扫失败签名（`traceback` / `cuda_oom` / `nan_loss` / `killed` / `cuda_error` / `dataloader`），
   把 rc、命中的签名行、日志末 30 行写进状态 json。

载荷**不**被 setsid，因此始终留在 pueue 的进程组里 —— 这正是 `q kill` 能真杀掉训练的原因。
`q kill` / `pueued shutdown` 时 qjob 捕获信号，先 TERM 后 KILL 载荷，并把 `phase=killed` 落盘。

退出码约定：`78` = gate 到期未开（跳过，队列继续）、`79` = 起不来或起来了但一直不出声、
`80` = wave 伙伴没发布 micro batch（`where_b_arm.sh`）、其余 = 载荷自己的 rc。

---

## 7. 失败处置手册

```bash
tools/queue/q status                      # Attention 段会直接点名失败任务与签名
tools/queue/q log <id> 200                # 看载荷日志
```

| 现象 | 含义 | 处置 |
|---|---|---|
| `phase=gate-wait` 且长期不动 | 前置产物没落地 | 看 `gates[].ok` 哪个 false；若上游已判死，`q kill` 让出卡 |
| `rc=78` | gate 超时 | 上游没产出。修好后 `q retry <id>` |
| `rc=79` | 起不来 / 起来了不出声 | 看日志前几十行；多半是数据/权重路径 |
| `rc=80` | wave 伙伴没发布 micro batch | 确认伙伴实际用值后，用 `MICRO_BATCH=N` 重新入队该臂 |
| `cuda_oom` 签名 | 显存不够 | 降 `--micro-batch` 重入队；注意**同 wave 两臂必须一起改** |
| `state=Running` 但 GPU 0% 且已久 | 卡死/空转 | Attention 会自动报；`q log` 看最后输出 |
| `daemon_ok=false` | 守护进程没了 | `q daemon`；排队任务会恢复，**正在跑的那批已经死了**，需 `q retry` |

---

## 8. 路径速查

| 东西 | 位置 |
|---|---|
| CLI / 包装器 / 状态工具 | `tools/queue/{q,qjob.sh,qstatus.py}` |
| 波次入队脚本 | `tools/queue/waves/enqueue_where_b_w2_w4.sh`、`tools/queue/waves/where_b_arm.sh` |
| 每任务状态 json | `/home/bc/data/queue/status/<NAME>.json`（`QUEUE_HOME` 可改） |
| 载荷日志 | 由 `q add` 的 `<LOG>` 指定；Where-B 为 `/home/bc/data/runs/where_b/<ARM>/train.log` |
| `job.marker` | 载荷日志同目录 |
| pueue 状态/日志 | `~/.local/share/pueue/{state.json,task_logs/,log/}` |
| pueue 配置 | `~/.config/pueue/pueue.yml` |
| wave micro batch 锁与发布值 | `/home/bc/data/runs/where_b/.wave_<w>.{prober.lock,micro_batch}` |

---

## 9. 主 agent 裁定（2026-08-10，已执行）

1. **W2–W4 钉 `MICRO_BATCH=8`**（与 EXEC-3 的 W01/W02 相同），消除跨 wave 配置差异，
   八臂完全同构可比。✅ 六臂已重新入队，gate 未变（见 §4.1 表）。
2. **加开机自起。** ✅ 已装，但**改用 systemd 用户服务而非 crontab**——本机 cron 目录 root 属主 +
   粘滞位，无 sudo 密码装不上（详见 §2）。所得机制比 `@reboot` 更强：还覆盖守护进程中途死亡。
3. **后续波次脚本暂不预写**，等 W1 实测墙钟与 What preflight 就绪后按 `where_b_arm.sh` 模式补。

### 拆队列时的一个坑（已实测踩到）

`q rm` 掉队头会**立刻**把后面的臂放上卡（队列本来就该这样），所以拆队列前
**先 `q pause gpu0 gpu1`**，否则 W05/W06 会在你删 W03/W04 的同一秒抢到卡并真的开跑。
本次演练中它们确实起来了，5 秒内被杀停，EXEC-3 的 W01 未受影响。
