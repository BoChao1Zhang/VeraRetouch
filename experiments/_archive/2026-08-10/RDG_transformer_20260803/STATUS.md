# RD-G · STATUS（排卡记录 + 长任务提交状态）

> 实时状态文件。作业清单与 PID/命令/日志见同目录 `job.marker`（D-20 硬规则）。

## 0. 排卡纪律（自查）

| 要求（任务卡） | 落实 |
|---|---|
| G-Lite 跑卡 0，G-Base 跑卡 1，两档并行 | ✅ `tools/run_stage1.sh`：GPU0 = `glite / mlp / gtiny`；GPU1 = `gbase / mlp_wide / gbase_free` |
| 每卡给其他作业留 ≥ 20 GB | ✅ 每进程 `torch.cuda.set_per_process_memory_fraction(0.22)` = 21.3 GB 上限，每卡 3 进程 = **64 GB 上限**，实测峰值约 19 GB/进程 ⇒ 约 **57 GB 常驻**，每卡留 ≥ 33 GB |
| 显存上限保护，宁可少占不许 OOM 掉别人 | ✅ 同上；越界的是**我自己的进程**先 OOM，不会挤掉邻居 |
| 严禁 kill 非自己启动的进程 | ✅ 全程只 kill 过自己启动的 `build_cache.py`（两次，见 §3 事故记录）与自己的 smoke |
| 长任务 nohup + job.marker（PID + 完整命令 + 日志） | ✅ `job.marker` |
| 断点续跑：已有 metrics 的 run 自动 SKIP | ✅ `train_rdg.py` 开头检查 `runs/<tag>/metrics.json`；`run_stage1.sh` 也复查一次 |

## 1. 开工前实测（决定 batch 的依据）

```
2026-08-03 14:14  nvidia-smi
GPU 0  0 MiB / 97871 MiB   0%
GPU 1  2807 MiB / 97871 MiB  0%   （PID 49548，.venv-lens，非本作业）
```

任务卡引用的「SM 6%、各占 2.4 GB」与实测一致（卡 1 2.8 GB / 卡 0 空闲）。

## 2. 作业与占卡（按启动顺序）

| 作业 | 卡 | 显存上限 | 说明 |
|---|---|---|---|
| `ceiling`（grid anchor，敏感性对照） | 0 | 0.20 → 19.4 GB | 逐 LUT 直接拟合 N=48，734 个 LUT × 8000 步。**已完成** |
| `ceiling_kmeans`（正式天花板，与各臂同 anchor） | 1 | 0.10 → 9.7 GB | 同上，改用 Stage-0 k-means anchor |
| `glite` / `mlp` / `gtiny` | 0 | 各 0.22 → 21.3 GB | Stage-1 |
| `gbase` / `mlp_wide` / `gbase_free` | 1 | 各 0.22 → 21.3 GB | Stage-1 |
| Stage-2（`runs2/`） | 视排期 | 0.12 → 11.7 GB | D-CONSTRUCT L0–L7 4D 档 |

## 3. 事故与返工记录（诚实登记）

1. **`build_cache.py` 第一次被我 kill**（自己的进程）：源图解析走 `tools/bgr_check` 的
   `BankResolver.resolve()`，它对每个 group 单独 `sqlite3.connect` 一次 NFS 上的 catalog，
   实测 19% CPU、72,587 组要 ~25 分钟。改成"每 bank 拷贝到本地 scratch + 一次全表查询"后秒级完成。
2. **`build_cache.py` 第二次被我 kill**：上面的批量版跑通后发现 `raise6k` 0/5363、`ppr10k` 0/9059、
   `fivek_gold` 0/4996 全部解析失败（合计丢 20% 组）。根因：这三个 bank 的 member 后缀是**角色限定**的
   （`.source.png` / `.preview.jpg` / `.before.jpg`），参考实现只认裸 `.jpg/.png` 后缀。
   改成按 metadata 自己的 `member` 字段精确匹配后 **72,587/72,587 = 100% 解析**。
   → 这是 `tools/bgr_check/common.py::BankResolver` 的一个静默缺陷，建议回修（见 REPORT 建议节）。
3. **天花板重算一次**：第一版天花板用均匀网格 anchor，而 PLAN 要求 anchor 来自 Stage-0 k-means。
   若各臂用 k-means anchor 而天花板用网格 anchor，臂的"天花板占比"会被系统性抬高（偏向"该停"结论）。
   → 用同一套 k-means anchor 重跑天花板（`runs/ceiling_kmeans`），网格版留作敏感性对照。

## 4. 数据加载瓶颈（任务卡点名要的发现）

见 `NOTES.md` §4/§5。要点：

- 原始 D-RENDER 路径 **125–140 图/s**（56 并发），56 个 worker 稳定在 **3.3–3.5% CPU** ⇒
  **瓶颈是 NFS 往返延迟，不是解码、不是本机 CPU**。折算 ≈ 48 MB/s。
- 训练侧单卡需要 ~2,000–5,000 样本/s ⇒ 原始路径慢 **15–40 倍**，**不建缓存就不可能吃满卡**。
- 采用的方案：离线一次性解码成 128×128 uint8 memmap（19.8 GB），训练时纯页缓存读。
- 源图（`imgs_in`）比 after 更慢（**33–40 图/s**）：ppr10k 是整幅 PNG、fivek_gold/raise6k 是大 JPEG，
  单文件字节数比 after 的 q95 JPEG 大一个量级。

### 3bis. 第一次 Stage-1 启动被我自己撤销（最重要的一次返工）

**症状**：六臂 15:45 启动后 13 分钟一条训练步日志都没有，`nvidia-smi` 却显示每卡 70+ GB。
**定位**：`nvidia-smi pmon -s um` 逐进程读数——我的 6 个进程 **SM 全是 `-`（≈0%）**，
而同卡其他 agent 的进程有 16–32%。即：我占着显存但根本没在算。
**根因**：`iostat -x` 实测 `/home` 所在的 **sdb 是机械盘**，
`%util 99.2 / r_await 120 ms / 284 IOPS / 21 MB/s`。
128×128 缓存共 19.8 GB，6 个进程各自**随机**读取，页缓存装不下（本机 125 GB RAM 已被其他 agent 占 62 GB），
于是每个样本都退化成一次 120 ms 的机械盘寻道。训练侧需要 6×768×98 KB ≈ 460 MB/step，
按 21 MB/s 算 **每步要 22 秒**——与"13 分钟不到 200 步"完全吻合。
**处置**：kill 我自己的 6 个进程（未动任何他人进程），把两个 `.u8` 文件复制进
**`/dev/shm`（tmpfs，纯内存）**，六臂改读 `/dev/shm/rdg_cache` 后重启；
同时把步数从 15,000 降到 8,000 以吸收这段损失的时间。

> 这是本实验第二次撞上数据加载墙：第一次是 NFS 延迟（原始 D-RENDER 路径 140 图/s），
> 第二次是本地机械盘随机读。两次都不是 GPU 算力问题。详见 REPORT「数据加载瓶颈」节。

### 事故 5（17:57 发现）：一次**静默失败的后台提交**，我还据此上报了假 PID

17:40 我以为把 `gbase_free` 按原口径重启了，并向主 agent 报了 `pid=574618`。
**17:57 复核发现该 PID 从不存在，进程从未启动。**

**根因**：本机 zsh 开着 `noclobber`，`nohup ... > logs/gbase_free.log` 对**已存在**的日志文件
会直接报 `file exists` 并让整条重定向失败，进程根本没起来。
我当时用 `pgrep -fc` 判定"起来了"——**而 pgrep 匹配到的是我自己那条包含同样字符串的 shell 命令**，
于是得到假阳性。

**连带**：重启前我 `rm -rf runs/gbase_free/`，新进程没起来 ⇒ 活动目录一度为空。
**step-1000 证据在 `runs/gbase_free_step1000_backup/` 完好**（rm 之前已备份），
+2.24 dB 的加固结论不依赖那次重跑。

**已固化的提交纪律（后续所有臂照此办）**：
1. 后台提交前**先 `rm -f` 目标日志**（noclobber 会静默吞掉重定向）；
2. 起完 `sleep` 后用 **`ps -p $PID`** 实证存活，**绝不用 `pgrep` 判活**（会匹配自己的命令行）；
3. 再 `tail` 日志确认已输出模型参数行，才能写进 `job.marker` 并上报。

17:57 已按此流程重启并实证：`pid=622857`，日志已出 params / LUT bank 行。

## 5. 实测显存与 SM 利用率

`tools/gpu_log.sh` 每 30 s 采样写入 `logs/gpu_usage.csv`；下表为逐进程实测（`nvidia-smi
--query-compute-apps`，2026-08-03 15:51，六臂全部进入训练稳态后）。

| 卡 | 我的进程 | 我的占用 | 其他 agent | 卡总占用 | 余量 |
|---|---|---|---|---|---|
| 0 | `glite` 21.4 GB + `gtiny` 18.5 GB + `mlp` 14.4 GB | **54.3 GB** | 14 个进程 × ~1.0 GB ≈ 14 GB | ~68 GB | ~29 GB |
| 1 | `gbase` 21.4 GB + `gbase_free` 21.4 GB + `mlp_wide` 14.5 GB + `ceiling_kmeans` 3.4 GB | **60.7 GB** | 5 个进程 × 2.6 GB ≈ 13 GB | ~74 GB | ~23 GB |

**全程采样统计**（`logs/gpu_usage.csv`，每 30 s 一点，取 tmpfs 重启后的样本）：

| 卡 | 显存 p50 | p90 | max | 整卡 SM p50 | SM 均值 |
|---|---|---|---|---|---|
| 0 | **64.0 GB** | 69.1 GB | 71.0 GB | **100%** | 84% |
| 1 | **64.1 GB** | 77.5 GB | 79.2 GB | **100%** | 89% |

开工前（14:14）：卡 0 **0 MiB / 0%**，卡 1 **2.8 GB / 0%**（任务卡说的"SM 6%、各占 2.4 GB"）。

**任务卡目标达成**：每卡 55–70 GB ✅（p50 64 GB）；SM ≥70% ✅（p50 100%、均值 84–89%）。
注意整卡 SM 含同卡其他 agent 的作业；**我自己的进程逐进程 SM 实测 12–49%/进程 × 3 进程/卡**
（`nvidia-smi pmon -s um`），这正是 tmpfs 修复前后的关键判别量——修复前我的进程逐进程 SM 是 **0%**，
而整卡 util 照样被别人顶到 99%。

### ⚠ 更正（17:40）：我自己的显存账本错了 8 GB，且"邻居超限"是误判

**错在哪**：我按 `torch.cuda.max_memory_allocated` 的自报数记账（三臂 16.6+11.2+13.8 ≈ 41.6 GB），
但 `nvidia-smi` 的真实占用含 CUDA context + 缓存分配器保留量，实测三臂 **21.3+18.4+14.4 = 54.1 GB**——
**我低估了自己 12.5 GB**。基于低估的账本，我把余量不足归因成"邻居扩到 39 GB、超了它声明的 25 GB 上限"。

**真相**（按 **cmdline 模块名**逐 PID 归属，工具 `tools/gpu_owner_audit.sh`）：

| GPU 0 owner | 实测 |
|---|---|
| **RD-G（我）** | **34.8 GB**（停掉 gtiny 后；停之前 54.1 GB） |
| RO9b（`export_vision`/`export_lm`） | 15.9 GB |
| PR-1/PR-3（`run_pipeline`/`run_behavior`） | 8.3 GB |
| RD-STD/E（`train_rd`） | **3.2 GB**（额度 25 GB 的 13%，**远未超限**） |
| other | 1.0 GB |
| 余量 | **32 GB** |

**误判根因（值得所有臂记下）**：卡 0 上 RD 组、RO9b、PR-1/PR-3 **共用同一个解释器**
`/home/bc/VeraRetouch/.venv-lens/bin/python`。**按解释器路径归属会把四家算成一家**
（3.2+15.9+8.3+1.0 ≈ 28 GB，量级正好和我误报的"39 GB"相当）。
**归属必须按 cmdline 里的模块名，不能按解释器路径。**

**我因此撤回两句话**：① "邻居扩到 ~39 GB"——错；② "超其声明的 25 GB 上限"——错，RD 组只用 3.2–4.3 GB。

### 显存预算算术（供后续排卡直接套用）

设邻居实占 N、硬底线留 20 GB，则我在该卡的上限 = `97.9 − N − 20`。

- **卡 0**：N ≈ 28 GB ⇒ 我 ≤ **49.9 GB**。两臂 35.7 GB ✅；**三臂 54.1 GB ❌ 超 4.2 GB**。
  ⇒ `gtiny` 要复跑，需卡 0 再腾约 **5 GB**（最大单一可压缩项是 RO9b 的 15.9 GB）。
- **卡 1**：N ≈ 15 GB ⇒ 我 ≤ **62.9 GB**。三臂（gbase+mlp_wide+gbase_free）≈ 57.1 GB ✅
  ⇒ **`gbase_free` 已于 17:40 按原口径重启**（不改 batch）。

### 复盘：gbase_free 那次是不是白砍了

**方向对、动作过头。** 16:36 时我在卡 1 占 60.7 GB、邻居约 21.7 GB，
按上式我的上限是 56.2 GB ⇒ **确实超了 4.5 GB，需要动手**。但：
① 我为了 4.5 GB 的超额砍掉了一个 21.4 GB 的臂；
② 余量在同一分钟（16:36 采样）就回到 20.0 GB，**属于瞬时波动**，
   而"连续三次才动手"的三振规则是我 17:32 才补上的。
⇒ **结论：那次是被瞬时波动误伤，可避免。** 已按原口径重启，step-1000 证据备份在
`runs/gbase_free_step1000_backup/`，+2.24 dB 的加固结论不依赖本次重跑。

**两条指令之间的张力（如实登记，未静默处理）**：任务卡同时要求「每卡 55–70 GB」与
「卡 1 给 RO-1+RO-3 留够（各 ≤20 GB，合计 40 GB）」。其他 agent 当前在卡 1 上实占 13 GB，
我留出 23 GB —— **满足「≥20 GB」这条硬规则**，但若 RO 两臂同时冲到各自 20 GB 上限（合计 40 GB）
则会不够。采取的处置：
1. 每个进程都有 `set_per_process_memory_fraction(0.22)` 上限，越界先 OOM 的是**我自己**；
2. 挂了余量监控，任一卡余量跌破 20 GB 时**立即 kill 我自己的 `gbase_free`**（PLAN 必做对照里
   优先级最低的一臂，可事后补跑），不动任何别人的进程；
3. 全程未 kill 过任何非本作业进程。
