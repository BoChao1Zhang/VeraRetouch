# STATUS — RD-STD / RD-E（+ 加挂 RD-A / RD-C / RD-D）

## 交付形态选择（主 agent 给了二选一）

加挂的 RD-A / RD-C / RD-D **并进本目录**，不另开 `experiments/RD_arms_20260803/`。
理由：五个臂共享**同一份像素缓存、同一个 seed、同一套 harness 调用、同一份可达性预检**；
并排成表的前提是配置逐字节相同，拆目录会引入「两套数据装配」的隐患。
REPORT 的判据表按臂加行（`arm_row` 字段标出 EXPERIMENTS_v3 的对应行号）。

## 冒烟（回合内，已通过）

- **RD CI 自检** `python -m model.glut_repro.ci_checks_rd`：**18 组 / 全 PASS**。
  含参数量对账（716 / 2530 / 2575 / 1100 / 1868 / 2636 / 780 / 14739 / 73695，
  **与 PLAN §2.1 各行标注逐位吻合**）、四线性插值对拍独立参考实现（3.6e-7）、
  gstd forward 对拍显式逐高斯求和（8.9e-8）、逐像素置换等变性（0.00）、
  GECO 对拍原文 Algorithm 1 四条、双 3D 对照臂的 s 无关性（严格 0）。
- **四臂 200–300 步冒烟**（L1/fixed）：g3d / gstd / lut3d / lut4d / ga / gc3 / gd 全部跑通。
- **单臂定标跑**（L1/fixed/gstd，2000 步 / px30000）：in-mask 41.86 dB（恒等 21.17）、
  Δ_const +14.27、Δ_shuffle +17.10 —— 链路通且判据线远在下方，据此把预算定为 8000 步（NOTES 决策 11）。
- **可达性预检**：L0–L5 fixed 已出（见 `ceiling/ceiling.json`），
  3D 对照臂 Δ_const*/Δ_shuffle* 严格为 0 的实现自检**逐档 True**。

## 长任务（已提交）

见 `job.marker`（PID + 完整启动命令 + 日志路径，D-20 硬规则）。

- **作业 1 可达性预检**：13 档已出 6 档后因 L6 manifest 的 list 型 `alpha_achieved`
  崩过一次（`data_construct.load_pairs` 只见过 L1/L4）。**未改 G3 的交付文件**，
  改为在 `data_rd.py` 里写了级别感知的 `load_pairs`，余下档随作业 3 续跑（已完成行自动 SKIP）。
- **作业 2 训练网格**：75 个 run，`WORKERS=16`，`STEPS=8000`。
  完成标志 `RD_GRID_DONE`；断点续跑：已有 metrics.json 的 run 自动 SKIP。

### 三次主动中止重提（诚实记录）

1. 14:47 首次提交：缓存预热**串行**，单档 ~5 min × 13 档 ≈ 65 min，GPU 全程空转。中止。
2. 14:54 二次提交：预热改 12 进程并行——但机器 **load 143 / 48 核**（同机另有 5 个 agent：
   RO-1 / RO-3 / RO-2 / RD-G / E2），12 个解码进程只是加剧抢核。中止。
3. 14:57 三次提交（**当前**）：预热降到 **4 进程并放到后台**，训练池**同时**启动，
   每个 worker 只等**自己那一份** `.npz`（`[grid] WAIT` 行每 60 s 打一次时间戳），
   第一档缓存就绪即开训。`WORKERS` 由 5 提到 **16**。

## 排卡（负责人要的实测数字）

**卡 0 only**（`CUDA_VISIBLE_DEVICES=0`），卡 1 全程未触碰。

| 时刻 | 我的 train_rd 进程数 | 我的 VRAM 合计 | 单进程均值 | 卡 0 上他人占用 | 卡 0 SM |
|---|---|---|---|---|---|
| 14:47（WORKERS=5，预热阶段） | 0 | 0 | — | 5.6 GB | 91%（RD-G 的 fit_ceiling，非我方） |
| 14:59（WORKERS=16，主网格稳态） | **16** | **13.9 GB** | **889 MiB** | 17.6 GB | **100%** |
| 17:5x（主网格已完，L0 补充网格 7 并行；**RD-G 主训已进卡 0**） | **6** | **5.68 GB** | 970 MiB | 卡 0 合计 69.4 / 97.9 GB | — |

**结论与取舍**：主 agent 给的红线是「全部臂合计 ≤ 25 GB」。16 worker 实测 13.9 GB，
**没有继续提到 24 的理由，且提上去有害**：
- SM 已经 100%，本作业是**发射受限（launch-bound）**的小模型（几 K 参数、bs 16384 点采样），
  再加进程主要是加内核发射争用，不是加吞吐；
- 卡 0 稍后要接 **RD-G 的 55–70 GB 主训**。当前 13.9（我） + 17.6（他人） = 31.5 GB，
  给 RD-G 留 65 GB；若提到 24 worker（外推 ~21 GB）就只剩 58 GB，会顶到 RD-G 的下沿。

故**定稿 WORKERS=16**。若 RD-G 起来后卡 0 吃紧，本网格可直接 `pkill` 再重启（全量断点续跑）。

同机 CPU 是真正的瓶颈（load 143 / 48 核），预热进程数因此压到 4；缓存建好后训练阶段
CPU 需求大幅下降（只剩 npz 读入与 24 张 val 图的重渲染）。

## 磁盘 I/O 核对（主 agent 转来 RD-G 的机械盘诊断，**已自行实测复核**）

RD-G 的诊断是真的，但**它对本任务缓存位置的假设不成立**，实测如下（`lsblk` / `df` / `iostat -x 1 3`）：

| 项 | 主 agent 转述 | 本任务实测 |
|---|---|---|
| 我的缓存位置 | `experiments/RD_std_e_20260803/cache/`（在 `/home`） | **错**。实际是 `/var/cache/veradata/rd/` → `ubuntu--vg-lv--root` → **`sda3`，`ROTA=0`（SSD）** |
| 机械盘 | `sdb` 是机械盘，`%util 99.2 / r_await 120 ms` | ✅ `sdb ROTA=1` 属实；但**当前实测 `%util` 6.34 / 9.90%、`r_await` 0.57 / 3.50 ms**，未饱和（RD-G 的 99% 是它自己那阵的负载） |
| 我卡住的根因 | 12 个预热进程往机械盘写 npz | **部分成立**：npz 写的是 SSD，但**读的源 PNG 在 `/home`**，而 `/home` 的 LV **横跨 `sda3`(SSD) + `sdb1`(机械)** ⇒ 200×3 张 PNG 的解码读确实会打到机械盘。这解释了预热慢，但不是"写盘卡死" |
| 当前瓶颈 | I/O | **CPU**：`iostat` 的 `avg-cpu` 为 **%user 96.15 / %iowait 0.00** |

**逐进程 SM（按主 agent 要求换成 `nvidia-smi pmon -s um`，不看整卡 util）**：
本任务的 L0 补充网格 5 个进程实测 **SM 12–16%**、各约 1.0–1.2 GB —— **在算，不是 SM=0% 的假忙**。
（同卡上 RD-G 的 `train_rdg` 为 21.3/14.4/18.4 GB，其中一个 SM 27–57%。）

**已采取的行动**：
1. 缓存搬 **tmpfs**：`/dev/shm/veradata_rd`（13 个 npz / 1.4 GB；内存 125 GB，余量充足），
   `config/run_grid.sh` 的 `CACHE` 默认值已改为它，并在启动时从 `/var/cache/veradata/rd`
   增量拷贝（`cp -n`）作为热身。
2. **约定（重要）**：`/dev/shm` 是 tmpfs，**重启即失**，因此**只放中间像素缓存**；
   一切交付物（`metrics.json` / `ckpt.pt` / `trace.jsonl` / REPORT / viz / ceiling）
   **仍写在 `experiments/RD_std_e_20260803/` 下**，且 `/var/cache/veradata/rd/` 保留一份持久副本。
3. **没有为此重启在跑的 L0 补充网格**：它只在启动时读一次 108 MB 的 npz（已在 page cache），
   之后全程 GPU 计算，重启只会白扔一小时进度而拿不到收益。
4. **WORKERS 的时序**：主 agent 提醒"搬 tmpfs 之前别把 WORKERS 提到 16"。
   实际时序上主网格**已在此诊断到达前跑完**（75/75，`RD_GRID_DONE`），
   WORKERS=16 全程实测 13.9 GB、SM 100%、无 I/O 停顿；**结论不受影响**。

## 基建缺陷核对：`BankResolver` 丢 20% 源 —— **与本任务无关**

`rg -n "BankResolver|bgr_check" model/glut_repro/ experiments/RD_std_e_20260803/config/` → **零命中**。
本任务的数据装配（`model/glut_repro/data_rd.py`）**不经过 bank 解析**：直接按 D-CONSTRUCT
manifest 里的相对路径 `Image.open(<root>/<split>/<level>/<uid>_{in,out,mask,maska,maskb,maskrender}.png)`。
覆盖率已逐级清点：**八级各 200 train / 24 val，全部读入成功，无缺样本**（§冒烟节的逐级校验）。
⇒ 本任务**不存在**静默少两成数据的风险；该缺陷影响的是走 bank 解析的实验（D-RENDER / bgr_check 系）。

## 完成情况（截至交付）

| 作业 | 状态 |
|---|---|
| 可达性预检 | ✅ 10 档全完成（8 级 `fixed` + 2 级 `mixed`），`ceiling/ceiling.json` |
| 主训练网格 | ✅ **75/75** 全完成，`RD_GRID_DONE`，已聚合进 `metrics.json` + 51 张图 |
| L0 收敛复核 | ✅ **7/7 全部跑完**（30000 步），`L0_LONG_DONE`。结论已定稿进 REPORT §3.6。
  判读经三次修订（8000 步"判死" → 10000 步我误写"差没收窄"→ 30000 步终值），**全部登记在报告里**。
  终值：RD-STD **−0.93**（判死线 −0.50）、RD-A −0.20、RD-D −0.76、**RD-C +1.82**、**RD-E −0.02**。
  重刷命令：`config/l0_table.py runs_l0long metrics.json` |

**RD-G 已进卡 0**（`train_rdg`，实测卡 0 合计 69.4/97.9 GB）。本任务当前只占 **5.68 GB**，
远低于 25 GB 红线；若 RD-G 需要更多，L0 补充网格可直接 `pkill -f "train_rd --level L0"`
再重启（断点续跑），**主网格 75 个 run 的结论不受影响**。

## 显存归属核对（2026-08-03 晚，主 agent 提示"你占了 39 GB"后实测）

**结论：不是我。实测本任务在卡 0 占 4,358 MiB = 4.26 GB，是 25 GB 额度的 17%。**

卡 0 合计 85,792 MiB 的逐 owner 分解（按 PID → cmdline 归属，命令见下）：

| owner | MiB | 进程数 | 说明 |
|---|---:|---:|---|
| **RD-G** (`train_rdg`) | **54,078** | 3 | glite 21,268 / gtiny 18,446 / mlp 14,364。**比主 agent 转述的 "~46 GB" 多 8 GB** |
| RO9b (`export_vision` / `export_lm`×3) | 17,122 | 4 | 另一 agent |
| `run_pipeline.py` | 5,232 | 2 | 另一 agent |
| `run_behavior.py` | 3,316 | 1 | 另一 agent |
| 其他 | 1,022 | 1 | — |
| **本任务** (`train_rd --level L0`) | **4,358** | 4 | L0 收敛复核剩余 4 臂（3 个对照臂已跑完退出） |

**误判的可能来源**：本任务与 RO9b / `run_pipeline` / `run_behavior` **共用同一个解释器路径**
`/home/bc/VeraRetouch/.venv-lens/bin/python`。若按 venv 路径而不是**模块名**归属，
这几家会被算成一家（4.36 + 17.12 + 5.23 + 3.32 + 1.02 ≈ 31 GB，量级与"39 GB"相当）。
**正确的归属命令**（按 cmdline 里的模块名，不是 venv 路径）：

```bash
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv,noheader,nounits \
| while IFS=, read pid mem uuid; do ps -o cmd= -p ${pid// /} ; done   # 逐 PID 看模块名
```

**处置**：本任务**未降并发**——4.26 GB 降到 0 只能腾出 4 GB，却会推迟 L0 终值；
真正的余量在上表前四行。**后续（同日）**：L0 复核 7/7 已自行跑完退出，
本任务在两张卡上的占用现已归零，那 4.26 GB 已自动释放，卡 0 降到 58.2 GB。

## 待办（下一棒）

1. ~~L0 补充网格刷新~~ ✅ 已完成并定稿。
2. 复核 `viz/ladder_delta_shuffle.png`：**RD-E 的停工门只在 `Δ*_shuffle ≥ 3 dB` 的档上宣读**，
   图里灰虚线就是那条前置条件线。
3. 单 seed 的局限（NOTES §4）：RD-A 的「与 RD-STD 差 < 0.5 dB → 换用」是**换主方案**级别的
   决定，实测差 0.19–0.68 dB 与 seed 噪声同量级，**单 seed 不可宣读**，需补 seed 1/2。
4. 烘焙一致性（E22）本卡未覆盖——红线级一等指标，RD 任何臂晋级前必须补。
5. REPORT §5 的 S1–S8 是给 EXPERIMENTS_v3 的具体改行建议，请阶段复盘时逐条处理。
