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

## 五、待主 agent 决策

1. **W2–W4 的 micro batch**：已按保守默认（每 wave 内自动协调）入队；
   EXEC-3 的 W1 用的是显式 `--micro-batch 8`。若需跨 wave 可比，
   请 `q rm` 掉六个再用 `MICRO_BATCH=8 ...enqueue_where_b_w2_w4.sh` 重入队。**未静默拍板。**
2. **开机自起**：现状重启后需手动 `q daemon`。是否加 `@reboot` crontab 一行。
3. **后续波次脚本**（Where 选型 / Stage-What T1–T4、C1–C2 / top-2 复跑）：
   入队通道已通用化，但各自的 harness 入口未定，未预写 `waves/` 脚本。
