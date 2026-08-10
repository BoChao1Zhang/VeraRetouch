# genctx 两卡生产作业 · 启动卡片

**当前状态：已准备，未启动。** 等主 agent 裁定 checkpoint（2488 或 4976）后，一条命令触发。
背景、墙钟推算、发现的问题与待决策项见 `NOTES.md`。

```
genctx_dual.sh <ckpt>            两卡驱动（唯一入口）；每卡一个分片队列，收尾自动合并
genctx_forced_color_dual.sh      同一驱动，MODE=forced_color（What 的 C01/C02 专用）
  genctx_shard.py                  └ 单分片 runner（生产脚本原封不动，只切 ds.refs）
  merge_genctx.py                  └ 分片合并 + 断言 + 兼容软链 + 交付报告
dryrun/                          79 条 CPU-only 断言 + 真生产脚本的 CPU 实跑日志
logs/  reports/                  运行时产出
```

## 启动（D-20 四步，驱动本身也要走一遍）

```bash
cd /home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/genctx_jobs
CKPT=/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976   # ← 主 agent 裁定后替换

rm -f logs/driver_two_segment.log                                        # 1
nohup setsid bash genctx_dual.sh "$CKPT" > logs/driver_two_segment.log 2>&1 &
echo $! > logs/driver_two_segment.pid
ps -p "$(cat logs/driver_two_segment.pid)" -o pid,etime,cmd --no-headers # 2
grep -q PROBE-OK logs/driver_two_segment.log && tail -n 40 logs/driver_two_segment.log  # 3
# 4：驱动自己会给每个分片作业写 logs/<mode>_<split>_shard<i>of<n>.job.marker
```

驱动**自己**会在起任何进程之前拒绝：GPU 还忙（>2 GiB 占用）、checkpoint 不可用、
split 索引缺失、目标发布根已存在但不完整。`DRY_RUN=1` 只打印计划、不碰 GPU。

跑完 `two_segment` 之后（且 C01/C02 确实排上了）再跑：

```bash
rm -f logs/driver_forced_color.log
nohup setsid bash genctx_forced_color_dual.sh "$CKPT" > logs/driver_forced_color.log 2>&1 &
```

## 监控

```bash
grep -E 'PROBE-OK|START|RUNNING|END rc=|MERGE|DRIVER' logs/driver_two_segment.log
tail -f logs/two_segment_train_shard00of08.log        # 每 200 条一行 samples_per_s / eta_s
cat logs/*.rc                                          # 每个分片的退出码
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
```

**起来约 2 分钟后**第一行 `{"done": 200, ..., "samples_per_s": …, "eta_s": …}` 就是真实吞吐，
用它替换 `NOTES.md` §4 的推算（`ETA_小时 = 每卡样本数 / samples_per_s / 3600`）。

## 产出落点

| | 路径 |
|---|---|
| 分片（中间物） | `/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/_shards/<mode>/<split>/shard<i>of<n>/<leaf>/` |
| 合并后（Where-B 读） | `…/genwhere/<split>/` 、`…/genwhere/<split>-forced_color/` |
| 合并后（Stage-What 读） | `…/genwhere/<split>/two_segment` 、`…/genwhere/<split>/forced_color`（软链，见 NOTES CX-1） |
| 分片报告 | `reports/<mode>/<split>/shard<i>of<n>/genctx_<leaf>.json` |
| 交付报告 | `../genctx_<leaf>.json`（合并后写；已存在则写 `.merged.json`，不静默覆盖） |

## 旋钮（都有默认值）

`MODE` `SPLITS` `GPUS` `NUM_SHARDS`(8) `SMALL_NUM_SHARDS`(2) `SHARD_MODE`(interleave)
`BATCH_SIZE`(64) `MAX_NEW_TOKENS`(512) `LIMIT`(空) `DO_MERGE`(1) `DRY_RUN`(0)
`FORCE_BUSY_GPU`(0) `GPU_FREE_MIB`(2000) `PROBE_TIMEOUT`(900)

```bash
DRY_RUN=1 bash genctx_dual.sh "$CKPT"                 # 只看计划
LIMIT=256 NUM_SHARDS=2 DO_MERGE=0 bash genctx_dual.sh "$CKPT"   # 吞吐标定（256 条）
```

## 重跑 / 续跑

已完成的分片（`manifest.status == complete`）自动跳过，只补没跑完的。
**任何目标目录已存在但不完整时，脚本一律拒绝而不是删除**——按 CLAUDE.md 先挪走、再重跑。
标定跑（`LIMIT=…`）的分片必须先挪走，否则会被当成已完成的正式产物跳过。

---

## 本次运行（2026-08-05 12:51 起，错峰双卡）

启动时 GPU1 仍在跑 Where-A `bench_calibration`，双卡驱动被自己的 busy-GPU 门拦下
（留档 `logs/driver_two_segment.refused-gpu1-busy.log`，未起任何进程），改为两个单卡驱动
各取任务表的一半（`TASK_STRIDE=2` / `TASK_OFFSET=0|1`，切法与双卡驱动逐条相同）：

```
阶段一 B=64  (12:51-13:04)  V_where:0/1 + V_what:0/1 四个分片全部 rc=0，已完成，保留
阶段二 B=128 (13:07-)       train 八个分片重跑（主 agent 裁定，验证见 NOTES §4-quater）
  GPU0  driver pid 1986659  logs/driver_gpu0_b128.log   train:0,2,4,6
  GPU1  driver pid 1987066  logs/driver_gpu1_b128.log   train:1,3,5,7
```

> 停作业时注意：`kill` driver **不会**停 worker 子 shell，它会接着起下一个分片。
> 先杀 worker 子 shell，再杀 python，都按 PID + `ps -p` 复核（禁 pgrep）。

**两个驱动都 `DO_MERGE=0`**——谁都看不到对方的分片。12 个分片全部 rc=0 之后，
必须手工跑一次合并（见 NOTES.md §4-ter 的三行循环），它会顺带建 CX-1 软链、
写 CX-2 审计字段。

监控：

```bash
grep -E 'START|RUNNING|END rc=' logs/driver_gpu*.log | tail
cat logs/*.rc                                    # 每个分片的退出码，0 = 好
grep -o '"samples_per_s": [0-9.]*' logs/two_segment_train_shard0*.log | tail
```

---

## 状态（2026-08-05 20:36）

**`two_segment` 已完工并合并**——Where-B 八臂、Stage-What 十臂（T01–T08 + C03/C04）的 genctx 前置解除。

```
/mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/
  train/   V_where/   V_what/           ← 已发布，manifest complete，含 two_segment 软链
  交付报告：../genctx_train.json  ../genctx_V_where.json  ../genctx_V_what.json
```

**`forced_color` 进行中（单卡 GPU1，B=128，2026-08-05 20:30 起，ETA 08-06 ~08:25）。**
日志路径固定不再改名：

```bash
tail -f logs/driver_forced_color.log                      # driver（pid 见 logs/driver_forced_color.pid）
grep -E 'START|RUNNING|END rc=' logs/driver_forced_color.log
grep -o '"samples_per_s": [0-9.]*' logs/forced_color_train_shard0*.log | tail
cat logs/forced_color_*.rc                                # 每片退出码
```

10 片全 rc=0 后由主 agent 触发合并（命令见 NOTES.md §4-sexies）。
