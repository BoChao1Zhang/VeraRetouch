# S0-JOINT NOTES：核实记录 / 假设清单 / 待主 agent 决策

任务卡：S0-JOINT（Base SFT 联合 preflight，真实数据 × 两卡 ZeRO-3）。
规格：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` 全文（414 行）已读。
前置交付：`s0_preflight/model/`（S0-TRAIN）与 `s0_preflight/data/`（S0-DATA），二者均未被本任务覆盖。

---

## 一、实施前核实记录（全部本机实测，无一条来自记忆或二手检索）

本任务**没有使用任何新的外部事实**（无新 URL / 无新论文数字 / 无新仓库 API 约定）。
用到的外部事实只有一条，且是 S0-TRAIN 已核实并写死在代码里的上游 shard sha256
（`q3vl/train/modeling.py::UPSTREAM_SHARD_SHA256`），本任务的两次真实启动都重跑了该校验并 PASS。

需要"当场核实而不是假设"的都是**本机运行时行为**，逐条实测如下：

| # | 需要核实的事实 | 核实方式 | 结果 |
|---|---|---|---|
| V1 | 两卡 ZeRO-3 下 `from_pretrained` 之后参数处于什么状态 | 专用 probe，torchrun 2 卡实跑 | `emb.weight.shape=[0]`、`numel()=0`、`ds_shape=[151936,2560]`、`ds_numel=388956160`、`ds_status=NOT_AVAILABLE`、`zero3_enabled=true` |
| V2 | `sum(p.numel())` 在 ZeRO-3 下等于多少 | 同上 | **0**（对照 `sum(p.ds_numel)=4,437,815,808`） |
| V3 | trainer 实际算出的 `max_steps` | 生产 config 真启动，读日志 | **4976**，与 manifest 推导的 `ceil(159215/32)=4976` 一致 |
| V4 | 两卡稳态 s/step | 生产 config 跑 74 步 | **5.04 s/step**（区间 4.93–5.32） |
| V5 | 两卡峰值显存 | 3 s 间隔轮询 `nvidia-smi` | GPU0 **52,973 MiB** / GPU1 **52,355 MiB**（卡容量 97,871 MiB） |
| V6 | NFS / 本地盘写吞吐 | `dd` 单流 + 双流并发 | NFS **103 MB/s**（单流 104、双流聚合 103 —— 说明链路已饱和，加并发无用）；本地 `/home` **182–222 MB/s** |
| V7 | `/home` 是什么介质 | `lsblk -o NAME,ROTA` | LV 跨 `sda3`(SSD) + `sdb1`(**ROTA=1 机械盘 ST4000NM002A**)，故按机械盘对待 |
| V8 | 两卡 GPU 互联拓扑 | `nvidia-smi topo -m` | **PXB**（PCIe switch，**无 NVLink**）—— 解释了 ZeRO-3 参数 all-gather 为何是主要开销 |
| V9 | ZeRO-3 checkpoint 实际体量 | Phase B 真存盘 | 见 PREFLIGHT_JOINT.md §4 |
| V10 | `transformers` 的 checkpoint 保存/删除是否会进日志 | 读 Phase B 日志 | **不会**（默认 `log_level=passive` → transformers logger 停在 WARNING），见 F-3 |

---

## 二、假设清单（能自证的当场证掉）

| # | 假设 | 处理 |
|---|---|---|
| A1 | 单卡 preflight 通过 ⇒ 两卡 ZeRO-3 也通过 | **实测证伪**。ZeRO-3 的 `zero.Init` 让参数在 `from_pretrained` 之后就是"已分区且已释放"状态，单卡路径完全碰不到。两个缺陷见 PREFLIGHT_JOINT.md §6（F-1 崩溃、F-2 静默） |
| A2 | `emb.weight.shape[0]` 能读出词表行数 | **实测证伪**：ZeRO-3 下是 0。已改走 `ds_shape` |
| A3 | `param.numel()` 能用来做参数量审计 | **实测证伪**：ZeRO-3 下是 0，且"全 0"能让 §9.2/§9.3 的所有"必须为 0"断言**全部通过** —— 静默 |
| A4 | HF 的 `max_steps` 会按 `floor(len(dl)/GAS)` 少算 1 步 | **实测证伪**：实跑给出 4976，与 `ceil(N/32)` 相等，`resolve_protected_steps` 的 ±1 容差没被用到 |
| A5 | 重初始化的 4 行在多卡下会漂 | **实测证否**：`min_pairwise_distance=0.0016162680694833398`，与单卡 preflight 报告的 1.616e-3 **逐位相同**（噪声故意用 CPU generator，与 device / world_size 无关），并补了单测 `test_mean_init_rows_is_device_independent` |
| A6 | 首选 `4 x GAS4` 在真实数据上不会 OOM | **实测证实**：峰值 51.7 GiB / 95.6 GiB，无需退 `2 x GAS8` |
| A7 | NFS 写吞吐够用 | **实测证伪**：103 MB/s 封顶，且 checkpoint 写入期间同一挂载点的 `du` 会阻塞数分钟 —— 训练数据也在这个挂载点上 |

---

## 三、待主 agent 决策

### D-J1（**已裁定 2026-08-05：本地盘方案**）：正式训练的 `output_dir`

实测事实（不是估算）：

| 目标 | 写吞吐 | 单 checkpoint 用时 | 剩余空间 |
|---|---|---|---|
| `/mnt/nfs/bc/runs`（NFS，spec §9.12 的 durable 路径） | **103 MB/s** | 见 PREFLIGHT_JOINT.md §4 | 23 TB |
| `/home/bc/data/runs`（本地 LV，含机械盘） | **182 MB/s** | 约为 NFS 的 0.57 倍 | 1.5 TB |

两个理由让本地盘不只是"快 1.8 倍"：

1. **训练数据和 checkpoint 在同一个 NFS 挂载点上**。images shard 共 23.7 GB，每个 epoch 全量读一遍；
   checkpoint 写入期间 NFS 已被打满（实测该期间对同一目录的 `du -sb` 阻塞 >2 分钟），dataloader 会跟着停。
   本地盘输出把这条竞争彻底去掉。
2. NFS 上还有别的用户（120 T 已用 98 T），长时间打满链路不是只影响自己。

**主 agent 裁定（2026-08-05）：选 (b) 本地盘方案。** 裁定理由（原文）：训练图片与 checkpoint 同 NFS 挂载点、
存盘期间实测阻塞该挂载点 IO >60 s，存盘与数据读取的耦合风险不可接受；本地 1.5 TB 对常驻 339 GiB 充足。

落实：`output_dir = /home/bc/data/runs/q3vl_base_sft_20260804`；两个 protected checkpoint（2488 / 4976）
与最终交付 checkpoint 训练完成后由 `s0_base_sft/config/SYNC_PROTECTED_TO_NFS.sh` rsync 到
`/mnt/nfs/bc/runs/q3vl_base_sft_20260804`，并把两侧路径与 sha256 写入 `s0_base_sft/checkpoint_sync_record.json`。
`sft_base.yaml` / `sft_base_resolved.yaml` / `START_PRODUCTION_TRAINING.sh` 文件头均已注明本裁定。

备查——当时提交给主 agent 的两个选项：

- **(a) 维持 NFS**（当前默认）：一次跑完约 8.5–9.5 h，无额外运维步骤，checkpoint 落地即 durable。
- **(b) 本地盘 + 异步同步**：`output_dir=/home/bc/data/runs/q3vl_base_sft_20260804`，训练期间只写本地；
  两个 protected checkpoint（step 2488 / 4976）在写完后用后台 `rsync` 推到 NFS。
  省约 40–60 min，并消除 dataloader 与 checkpoint 的 IO 竞争；代价是**训练进行中 checkpoint 不是 durable**
  （机器挂了要重跑），且需要一个同步步骤。本地峰值占用 = `save_total_limit 3 + 2 protected` ≈ 5 份，1.5 TB 够。

### D-J2（**已采默认，需追认**）：Phase B smoke 的三处非规格偏离

为了在**不跑满 epoch** 的前提下闭合"eval → 存盘 → 轮转 → 中断 → resume"，Phase B 用
`Q3VL_ALLOW_NONSPEC=1` 覆盖了三个 spec 冻结值：`max_steps=8`（spec 无此项，但等效缩短 `num_train_epochs`）、
`eval_steps=4`、`save_steps=2`、`save_total_limit=1`。

- `save_total_limit=1` 是**故意**选的：它是 transformers 4.57.1 第二条删除路径的唯一触发条件
  （`trainer.py:2840`，S0-TRAIN 修的那个静默删除 protected checkpoint 的坑），
  在两卡 ZeRO-3 真实路径上验一次，比只靠单测强。
- **正式训练不设 `Q3VL_ALLOW_NONSPEC`**；Phase A 用的是**逐字生产 config**，只改了 `output_dir`。

### D-J3（**已裁定 2026-08-05：保持关闭**）：生成类诊断在正式训练中开不开

任务卡必做项 2 要求单验生成管线。已在单卡非 ZeRO 路径上跑通（PREFLIGHT_JOINT.md §5）。
但这**不等于**在两卡 ZeRO-3 下可用：ZeRO-3 的 `model.generate` 需要逐层 gather 参数，
而本任务的授权范围是短程 smoke，不足以给出可信的 ZeRO-3 生成耗时。

**主 agent 裁定（2026-08-05）：保持关闭**，正式 config `gen_diag_samples: 0`，
训练后在 protected checkpoint 上离线补。即 S0-TRAIN D-3 的选项 (a)：训练结束后**离线**批量生成，
再算 §8.3 的标签完整率 / 顺序正确率 / 非空率 / 旧七标签残留率。理由见 PREFLIGHT_JOINT.md §5 的实测耗时。

### D-J4（已采默认，随启动指令生效）：把 `log_level: info` 加进正式 config

见 PREFLIGHT_JOINT.md 缺陷 F-3：默认 `log_level=passive` 下，transformers 的
`Saving model checkpoint to ...` 与 `Deleting older checkpoint ...` 都是 `logger.info`，
**一条都不会进日志**。一个 7 小时、要写 11 个 checkpoint 的任务，如果存盘和删除都不可见：

- D-20 第 3 步"tail 日志确认已输出实质内容"在训练中段没有东西可看；
- S0-TRAIN 修的那个 bug 的性质就是"删除是静默的"，让它继续静默等于把同一个坑留一半。

**采用**：正式 config 增加 `log_level: info`。该项**不在** `FROZEN_HPARAMS` 里，不构成规格偏离。

### D-J5（继承 S0-TRAIN D-7，仍待确认）：Base SFT 的 eval 集 = `V_where`

任务卡已按"Base SFT 的 eval 集已裁定为 V_where"执行，本任务据此实跑（896 条，真实 eval 已跑通）。
此处只做记录，不重开决策。

---

## 四、纪律遵守记录

- **未启动正式训练**。GPU 只用于两段短程 smoke（Phase A 74 步后主动 kill；Phase B `max_steps=8`）
  与一次单卡生成管线单验。收到主 agent 的边界澄清后**立即停止** Phase A，并以 `ps -p` 逐 PID 实证退出、
  `nvidia-smi` 实证 0 MiB。
- **D-20 四步**全部走：`smoke_launch.sh` 先 `rm -f` 日志 → `sleep` 后 `ps -p $PID` 判活 →
  `grep` 日志确认实质输出 → 最后才写 `job.marker`。**全流程无 `pgrep`**；停止 Phase A 时也是
  先 `ps --ppid` 列出子进程、`kill -TERM` 后逐个 `ps -p` 验证退出。
- `smoke_launch.sh` 内置硬拒绝：`RUN_DIR` 命中 `q3vl_base_sft_20260804` 直接 `exit 2`，
  smoke 不可能写进正式输出路径。
- **未覆盖** `s0_preflight/data/` 与 `s0_preflight/model/` 的任何既有交付物；按主 agent 指令
  在 `PREFLIGHT_MODEL.md` / `preflight_model.json` 上**追加**带日期与出处的 addendum，原文一字未改。
- 被 NFS silly-rename 咬过一次：`rm -rf` 失败报 `Device or resource busy` + 残留 `.nfs00000000...` 文件。
  没有当噪声跳过——查清是自己的 `tail -f` 监控还开着句柄，先停监控再删。这与 D-20 记录的
  "stale shell 污染文件"是同一类根因。
- 清空 Phase A 首次失败的运行目录**之前**，先把失败日志备份到
  `logs/phaseA_attempt1_FAILED_zero3_embedding.log`（"清空重跑前必先备份已有产物"）。

---

## 五、主 agent 裁定与启动（2026-08-05）

独立实现审阅 blocker 已全部清零，主 agent 裁定 D-J1 = 本地盘、D-J3 = 关闭，指示删除 smoke 目录并启动正式训练。三项均已执行：

- smoke 目录 `/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/` 删除前逐项 `find` 核对，把三段的 `job.marker` / `run_setup.json` / `trainer_state.json` 补进交付目录 `logs/` 后才删除，已确认目录不存在；
- 正式训练按更新后的 `START_PRODUCTION_TRAINING.sh` 启动（三项硬拒绝检查保留，另加两项：RUN_DIR 必须等于 config 的 `output_dir`、本地卷余量 ≥400 GiB）；
- D-20 四步逐步执行并留证，两 rank PID 以 `ps -p` 实证存活，首个 loss 已落盘。

**本任务到此结束，训练监控移交主 agent。**
