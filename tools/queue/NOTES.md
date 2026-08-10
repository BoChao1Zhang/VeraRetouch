# QUEUE-1 实施记录

## 一、在线核实记录（本任务用到的全部外部事实）

| 事实 | 来源 | 结论 |
|---|---|---|
| pueue 特性集（JSON 状态 / 分组并行 / 依赖 / 改序 / 编辑队列 / 暂停） | 打开 `github.com/Nukesor/pueue` README | 全部属实 |
| 分组命令语法 | 打开 wiki `Groups.md` 原文 | `pueue group add <g>` / `pueue parallel -g <g> N` / `pueue add -g <g>` |
| 最新版本与产物 | `api.github.com/repos/Nukesor/pueue/releases/latest` | v4.0.4，提供 `x86_64-unknown-linux-musl` 静态二进制 |
| task-spooler 可得性 | `apt-cache policy task-spooler` | jammy/universe 有 1.0.1，但需 sudo；且无 JSON 输出 |
| GNU parallel / simple_gpu_scheduler | `which` / `python -c import` | 本机均未安装 |

**未采信任何二手数字**：下面这些是本机实测而非文档转述——
静态链接（`file` 显示 statically linked）、`status --json` 形状、失败不阻塞后续、
`--after` 前置失败 → `DependencyFailed`、`shutdown` 后重启状态完整恢复、
socket 实际位于 `$XDG_RUNTIME_DIR`。

## 二、实测记录

| 项 | 结果 |
|---|---|
| 完整生命周期（入队→执行→完成/失败→可读→重排→取消） | 4 个 CPU 假任务跑通，未碰 GPU |
| 失败可见性 | `DEMO_FAIL` rc=1，签名 `traceback`，日志末行进状态 json |
| gate 等待→放行 | `DEMO_GATED` 等到上游产物落地后自动开跑并成功 |
| 失败不阻塞 | 同组后续任务照常执行 |
| 交接延迟（GPU 空转窗口） | `LAT_A` 结束 11:32:47.257 → `LAT_B` 开始 11:32:47.259，**2 ms** |
| 守护进程重启 | 分组、并行度、排队任务全部恢复 |
| 密钥泄漏 | `state.json` 与 `q status --json` 中 `sk-` / `xai-` / `jina_` / `tvly-` / `API_KEY` 命中数均为 **0** |
| `q status` 耗时 | 0.42 s（要求 ≤2 s） |
| wave micro batch 协调 | prober 发布 7 → follower 钉 7；伙伴不发布时 follower rc=80 拒跑 |

## 三、实施中发现并修掉的三个真 bug

1. **`GROUPS` 是 bash 内建数组**（当前用户的 unix 组）。`GROUPS=(gpu0 gpu1)` 被静默忽略，
   首版 `q daemon` 因此创建了名为 `1001` / `27` / `999` 的 pueue 组。已改名 `CARD_GROUPS`，
   并清掉了误建的组。
2. **Python 3.10 的 `datetime.fromisoformat` 解析不了 pueue 的 9 位纳秒时间戳**，
   导致整列 `elapsed` 静默显示 `-`。已截断到 6 位。
3. **首版 qjob 的监督循环 `sleep 60`**，载荷退出后槽位（以及卡）最多多占 60 秒。
   已改为 5 秒探活 / 60 秒才写状态，交接延迟降到毫秒级。

另修掉一个设计缺陷：初版 wave 配对是「指定某臂 probe、另一臂等」，当被指定的等待方
先到队头时会空等到超时。改为 `mkdir` 原子抢锁，**先起的臂 probe**，两卡任意错开都不死锁。

## 四、假设清单

1. Wave 划分 W1={W01,W02} … W4={W07,W08}、奇数臂 GPU0 / 偶数臂 GPU1 ——
   来自 protocol `METACANVAS_..._2026-08-04.md` 的 Wave 表（已读原文）。
2. 一条臂的完成产物是 `where_b/arm_<ARM>.json` —— 读 `run_where_b.py` 末尾确认，
   它在 `evaluate_arm` 之后写出，故存在即代表该臂真正跑完。
3. `--ready` 用 `total_optimizer_steps`，与 `run_where_b.sh` 原有 `submit` 的 gate 字符串一致。
4. 启动命令等价于 `run_where_b.sh train` 去掉其自有 `submit`（LD_LIBRARY_PATH、
   `/home/bc/envs/q3vl_sft/bin/python`、repo cwd、checkpoint-4976 全部照抄）。
   **不复用该 verb**，因为它 `nohup setsid` 会把训练甩出 pueue 的进程组。
5. gate 一律只用本地路径。NFS 前置由 `run_where_b.py` 自己断言（见 QUEUE_USAGE §4.1）。

## 五、主 agent 裁定与执行（2026-08-10）

1. **W2–W4 钉 `MICRO_BATCH=8`** —— 已执行。六臂重新入队，逐条核对
   `--micro-batch 8` 与 gate（W03→`arm_W01.json`、W04→`arm_W02.json`、六臂均含 checkpoint-4976）。
2. **开机自起** —— 已装，但**机制与原指令不同，理由如下**：
   `@reboot` crontab **装不上**。`/var/spool/cron/crontabs` 为 `drwx-wx--T root:crontab`
   且 `crontabs/bc` 非 `bc` 所有；`crontab -` 依赖 rename 覆盖，粘滞位禁止覆盖非自有文件，
   实测报 `crontab: crontabs/bc: rename: Operation not permitted`，而 `sudo` 要密码。
   改用 **systemd 用户服务 + linger**（`loginctl enable-linger` 自助执行被 polkit 允许，实测 rc=0）。
   unit 取自 pueue v4.0.4 官方 `systemd.pueued.service`，只改二进制路径与
   `Restart=on-failure`。**覆盖面比 `@reboot` 更大**：重启 + 守护进程中途死亡都能自愈。
   幂等由 `install_autostart.sh` 保证，且**装完自验** enabled/active/守护进程应答三项。
   cron 路线保留在 `--cron-fallback`（需一条 root 命令）。
3. **后续波次脚本** —— 按裁定暂不预写。

## 六、第二轮又逼出的两个问题（均已修）

1. **拆队列会瞬间放行后继臂**：`q rm` 掉队头后，W05/W06 在同一秒被提升为 Running
   并真的 exec 了训练（gate 只有 checkpoint，天然满足）。5 秒内杀停，EXEC-3 的 W01 未受影响。
   → 文档补「拆队列前先 `q pause` 两张卡」；本次重新入队即按此顺序执行。
2. **被 kill 的臂会留下 wave 锁**：`.wave_w3.prober.lock` 被夭折的 W05 占着，
   若不清理，下次 W05 会退化成等一个永不 probe 的伙伴，白等 2h 再 rc=80。
   → 入队脚本现在会先清掉本次所提交 wave 的 `.wave_<w>.{prober.lock,micro_batch}`。

另：`install_autostart.sh` 的第一版（cron 版）在 `crontab -` 已经失败的情况下照样打印
"installed:" —— 正是本项目反复吃亏的**假成功**。现版本三项实证（`is-enabled`、`is-active`、
守护进程实际应答）全过才报成功，任一不过即 dump `systemctl status` 并以非零退出。
