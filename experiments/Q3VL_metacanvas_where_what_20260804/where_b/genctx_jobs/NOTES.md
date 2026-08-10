# EXEC-2 · generated-context 两卡生产作业 —— 准备记录（**未启动**）

任务：用选定的 Base SFT checkpoint 对全量样本自回归生成两段 reasoning
（`q3vl/whereb/scripts/make_generated_context.py`，schema `genwhere/2`），两张 H100 分片并行。
本目录只准备，**一行 GPU 都没碰**（全部 dry-run 都 `CUDA_VISIBLE_DEVICES=""`）。

引用文档节：`where_b/PREFLIGHT_WHERE_B_PENDING.md` §S2、`what/PREFLIGHT_WHAT_PENDING.md` WT-J10
与 §三-bis、`q3vl/whereb/{config,gencontext,data,stores,hiddens}.py`、
`q3vl/what/{config,stores}.py`、`q3vl/whereb/scripts/{make_generated_context.py,run_where_b.sh}`、
`q3vl/what/scripts/{run_what.py,evaluate_what.py}`、`s0_base_sft/chain/{joblib.sh,verify_checkpoints.py}`。

---

## 1 · 核实记录（全部本仓库/本机实测，未用外部检索）

| # | 事实 | 怎么核实的 |
|---|---|---|
| V1 | 生产脚本**没有**任何分片参数：只有 `--limit`（前 N 条），无 `--shard`/`--offset` | 读 `make_generated_context.py` 全文 + `--help` 实跑（`dryrun.log`） |
| V2 | `--report-dir` 下的文件名是 `genctx_<leaf>.json`，**leaf 不含分片号** | 同上，L165 |
| V3 | 发布是原子的且拒绝已存在目录（`_prepare_output` 用 `_lexists`） | `q3vl/data/shardio.py` L86 + `dataset_build/tools/indexed_tar.py` L655 |
| V4 | 消费方按 `manifest.json` + `indexes/shard-*.idx.jsonl` + `shards/*.tar` 定位，目录里多出的条目无害 | `q3vl/whereb/stores.py` `PublishedStore.__init__/read` |
| V5 | 样本量：train 159,215、V_where 896、V_what 897（`open_dataset` 无 build 剔除，等于索引行数） | 逐行解析三个 `*.index.jsonl`；`open_dataset` 实跑 n=896 |
| V6 | prompt 长度 p50 = **464** tok（visual p50 384），`total_tokens` p50 653、max 1142 | 索引自带 `n_visual_tokens`/`total_tokens`；`Sft2SegCollator.encode_one` 实跑 12 条取中位数 |
| V7 | CPU 侧每样本 **~31 ms** 且**不与 GPU 重叠**：`dataset[i]` 冷 24 ms（热 4.3 ms）+ `encode_one` 1.7 ms + `image_processor` 5.5 ms/图 | CPU 实测 48 条（V_where），见 §4 |
| V8 | 两卡是 H100 94 GB（97871 MiB），当前被 Base SFT 占 54 GB×2 | `nvidia-smi` |
| V9 | 模型 KV：36 层 × 8 kv-head × 128 head_dim × 2(KV) × 2 B = **144 KiB/token** | `/home/bc/data/models/Qwen3-VL-4B-Instruct/config.json` |
| V10 | GT 两段长度 p50 210 / p90 251 / p99 310 / **max 326** tok（64 条 V_where 实测） | `chain/verify_checkpoints.py` L703-705 注释；与 `whereb/config.py` 的 204/332 口径一致 |
| V11 | `run_where_b.py` 把 genctx 根**写死**成 `GENCTX_DIR/<split>`，无 CLI 覆盖 | 读 L128 与 argparse 段 |
| V12 | `run_what.py` / `evaluate_what.py` 用 `<root>/<split>/<mode>`，有 `--color-genctx` | 读 L223、L247、`evaluate_what.py` L179 |
| V13 | 生成作业把全部记录**攒在内存里最后一次性发布**（`records = list(...)` 后 `publish_generated`） | `make_generated_context.py` L143-150 |
| V14 | `generate_where` 是左 padding + KV cache + greedy，`prefix_ids` 用于 forced_color | `q3vl/whereb/hiddens.py` L227-268 |
| V15 | 消费侧**不校验** genctx 记录里的 `checkpoint` 与训练用的 SFT checkpoint 一致 | `assert_genctx_coverage` 只查覆盖；`ColorGenContextStore` 只查 schema/mode/覆盖 |
| V16 | **单步解码 ~20–27 ms**（B=1、`max_new=448`、FA2、bf16、H100）：单样本生成实测 8.1–12.5 s；checkpoint 加载 134.5 s | 链上 GPU0 验证作业的实时日志 `chain/logs/gpu0_ckpt_verify.log`（11:52） |
| V17 | 生成记录里的 `generated_ids` 含**整批统一宽度**的尾部 pad/eos；下游 `extract_segment` 先切 eos 再切 `</where>`，所以**上下文本身没被污染**，但 `n_generated_tokens` / summary 的 `generated_tokens` 是"该批的解码步数"而非样本自身长度 | 读 `hiddens.generate_where` L267-268 与 `context.extract_segment` L186-187 |

---

## 2 · 分片方案

生产脚本一行未改。`genctx_shard.py` 在**包自己的工厂** `open_dataset()` 建好数据集之后切
`ds.refs`，然后把控制权交回脚本自己的 `main()`——切片之后的一切（prompt、生成、记录、发布）
都是原作者的代码。

* **切法**：`interleave`（`refs[i::n]`，默认）。索引本身按 sample_id 哈希序，build 已充分打散
  （train 159,215 条里有 142,365 次相邻 build 变化），所以连续切也行；选交错是因为它对
  "两卡同时收工"更稳，且分片大小最多差 1。`contiguous` 保留作为可复现的备选。
* **强制隔离**：wrapper 自己给 `--out-root` 和 `--report-dir` 追加 `shard<i>of<n>`。
  - 不隔离 out-root：第二个分片在**跑完几小时之后**才因目录已存在而死（发布是原子的）；
  - 不隔离 report-dir：**8 个分片静默互相覆盖同一个 `genctx_train.json`**，最后留下一个
    看起来完整、其实只描述 1/8 工作量的报告——正是 Stage-What R5 事故那一类。
* **`--limit` 语义**：先对整个 split 生效，再切分片（`--limit 64 --num-shards 8` = 全 split 取 64 条、
  每片 8 条），标定用。
* **合并**：`merge_genctx.py` 把各分片的记录流回**原作者的** `publish_generated()` 重新发布成
  一个根。不手工拼 tar/index：那等于重写 manifest + 每分片摘要 + SQLite catalog。
  合并前逐项断言（见 §3），合并后用消费方自己的类读回。
* **分片数**：train 默认 8（每卡 4 段、每段 ~1 h），小 split 2。多于卡数是有意的——
  见 V13，作业中途被杀会**丢掉全部产出**，8 片把最坏损失压到 1/8，代价是多 6 次模型加载（~3%）。
  已完成的分片重跑时自动跳过（manifest `status: complete`）。

---

## 3 · 合并的断言（全部在发布任何字节之前）

1. 每个分片根存在、`manifest.status == complete`、索引可读；
2. 每个分片的 sample_id 集合 **恰好等于**它应该分到的那一片——不是"互不相交且并集覆盖"。
   两个分片各自用错 `--shard` 但恰好互补时，只有前者能抓住；
3. 并集 == `open_dataset(split)` 的样本集（同一个 build 过滤口径），不多不少；
4. 每条记录 `schema_version == genwhere/2`、`mode` == 本次要的、`checkpoint` **全 split 唯一**
   （半个 split 由 2488 生成、半个由 4976 生成，在 loss 里完全看不出来）。

合并后再用 `GenContextStore`（Where-B）与 `ColorGenContextStore`（Stage-What）读回，
写 `genctx_<leaf>.json`（已存在则写 `.merged.json`，`--force-report` 才覆盖并留 `.superseded`）。

---

## 4 · 墙钟预估（**模型推算，不是实测**）

单批耗时 = CPU 装配 + prefill + 解码步数 × 单步时延：

| 项 | 数值 | 来源 |
|---|---|---|
| CPU 装配 | B × 31 ms（**不与 GPU 重叠**，循环里是先取样本再 generate） | V7 实测 |
| prefill | B=64 时 ~0.8 s（文本 64×464 tok + 视觉 64×1536 patch） | 由 V6 推算 |
| 解码步数 S | HF `generate` 等**整批都结束**才停：正常终止时 S ≈ 批内最大长度 ≈ 330（V10 max 326+4 标签）；只要批里有一条不吐 EOS，S 就是 `max_new_tokens` = 512。B=64、不终止率 1% → 47% 的批会跑满，2% → 73% | V10 + `generate_where` 语义 |
| 单步时延 | **B=1 实测 20–27 ms**（V16）；B=64 估 25–35 ms（权重 8.8 GB + KV 8.6 GB 的带宽下限约 7 ms，其余是 HF 的 python/launch 开销，随 B 增长很慢） | V16 实测 + V8/V9 推算 |

| 场景 | 每样本 | train 分两卡 | + V_where/V_what |
|---|---|---|---|
| 乐观（S=420、单步 25 ms） | 0.20 s | 约 4.5 h | +6 min |
| 悲观（S=512、单步 35 ms） | 0.31 s | 约 7 h | +9 min |

### 4-bis · 上线后的实测（2026-08-05 12:54，第一个分片跑完）

**推算偏乐观，实测约慢 1.3–2 倍**，以实测为准：

| 量 | 实测 | 备注 |
|---|---|---|
| 吞吐 | **2.452 samples/s/卡**（B=64） | `reports/two_segment/V_where/shard00of02/genctx_V_where.json` |
| 每批 | 26.1 s / 64 条 | 448 条 / 7 批 / 182.7 s |
| 解码步数 | 259–279（p50 264） | 模型正常吐 EOS，**没有跑满 512**（CX-5 的悲观档没发生） |
| 单步 | **约 99 ms** | 比推算的 25–35 ms 慢 3 倍；CPU 装配只占 8%，主因是 HF `DynamicCache` 每步整块重拼 + python 开销 |
| 峰值显存 | **16.43 GiB / 94 GB** | B=128 完全放得下 |
| 每分片模型加载 | 20 s（页缓存热） | 比冷加载的 134 s 好得多 |
| 端到端（含加载/发布） | 213 s / 448 条 = 2.10 samples/s | 排期按这个算 |

**当时的 ETA**：每卡 4 个 train 分片 × 19,902 条 ≈ 9.1 h ⇒ 全量约 22:00 完成。
**已被 §4-quater 的 B=128 切换取代（新 ETA 18:10–18:20）。**

**质量（第一个分片 448 条，checkpoint-4976）全部满分**：
`format_failure_rate 0.0`、`starts_with_where_open_rate 1.0`、
`color_format_failure_rate 0.0`、`starts_with_color_open_rate 1.0`、
`both_segments_well_formed_rate 1.0`、`stop_reasons` 全 `closed`、`segments_overlap_rate 0.0`。
段长：`where` p50 **8** tok / p95 52 / max 60（远短于 GT 的 p50 41——V_where 55% 是 global 样本，
其 `<where>` 本来就极短）；`color` p50 173 / p95 222 / max 259，都在 96/384 的固定边界内。
`generated_tokens` p50 264 是"该批解码步数"（CX-6 口径，不是段长）。

* `two_segment`（train + V_where + V_what，161,008 条）：**两卡 ~4–7 h**；
* `forced_color`（train + V_what，160,112 条）：**再来一遍 ~4–7 h**；
* 单卡跑就是翻倍（7–13 h）。

**这不是"空档填充"级别的作业**，请主 agent 按此排期（§6 决策 D1）。

**怎么把估算换成实测（两个现成的读数）**：
1. GPU0 的 checkpoint 验证作业跑完后，`s0_base_sft/checkpoint_verification.json` 里
   `summary.generated_tokens` 分位数（S 的真值）与 `generation_seconds`（batch 8、max_new 448）；
   日志里还有每批 `generated (X.Xs this batch)`。
2. 本作业起来约 2 分钟后，分片日志的第一行 `{"done": 200, ..., "samples_per_s": x, "eta_s": y}`
   就是该卡的真实吞吐；`ETA_小时 = 每卡样本数 / samples_per_s / 3600`。
   两卡各自的 shard00 日志都要看。

**显存**：B=64 时估 ~25 GiB（权重 8.8 + KV 8.6 + 视觉激活/pixel_values 若干），94 GB 卡上很宽裕；
B=128 估 ~40 GiB 也进得去。默认取 **64**（保守），标定后可上调；作业报告里有 `peak_memory_gib`。

---

### 4-quater · B=128 验证与切换（主 agent 裁定，2026-08-05 13:00–13:07）

判据（主 agent 预注册）：**七项结构率全满分 且 `where_ids` 一致率 ≥ 0.90 → 切 B=128**。

方法：同一批 **train 头部 256 条**（`--limit 256 --num-shards 1`），先 B=64 再 B=128，
**同一张卡、同一 checkpoint、同样与生产作业争用**，逐样本比 ids。

| 项 | B=64 | B=128 | 判据 |
|---|---|---|---|
| `starts_with_where_open_rate` | 1.0 | **1.0** | 满分 |
| `starts_with_color_open_rate` | 1.0 | **1.0** | 满分 |
| `format_failure_rate` | 0.0 | **0.0** | 满分 |
| `color_format_failure_rate` | 0.0 | **0.0** | 满分 |
| `truncation_rate` | 0.0 | **0.0** | 满分 |
| `color_truncation_rate` | 0.0 | **0.0** | 满分 |
| `both_segments_well_formed_rate` | 1.0 | **1.0** | 满分 |
| （附）`segments_overlap_rate` | 0.0 | 0.0 | — |
| （附）`stop_reasons` | 全 `closed` | 全 `closed` | — |
| 段长 `where` / `color` | p50 8 / 171 | **完全相同** | — |
| 吞吐（争用下） | 1.707 /s | **2.69 /s（1.58×）** | — |
| 峰值显存 | 17.1 GiB | **25.8 GiB** | 94 GB 卡 |

**逐样本一致率（256 条相同样本）：`where_ids` 256/256 = 1.0000、`where_text` 1.0000、
`color_ids` 256/256 = 1.0000。** 即在 B=64↔B=128 这一对上，bf16 平票一次都没掷到不同面
（比 commit bdf93d3 测的 B=32 vs B=1 的 0.9688 还干净——那是与 B=1 比，跨度更大）。

⇒ **两项判据都过，执行切换**：13:04 停掉两卡 train 分片（**当时 0 个 train 分片已发布，
不丢任何产物**），13:07 以 B=128 重启全部 8 个 train 分片。
V_where / V_what 的 4 个 B=64 分片**已完成，不动**（记录自带 `gen.batch_size`，口径自述清晰）。

**切换后实测（两卡，无争用）：GPU0 4.37 /s、GPU1 4.48 /s ⇒ 相对 B=64 的 2.45 /s 提速 1.79×。**
每个 train 分片 eta ≈ 4,400–4,500 s（1.24 h），每卡 4 片 ⇒ **约 5.1 h，预计 18:10–18:20 完成**
（比 B=64 的 22:00 早约 3.8 h）。显存 33.8 / 37.6 GiB（nvidia-smi 口径，含分配器保留）。

**停作业时踩到并已记录的坑**：`kill` 掉 driver **不会**停下它的 worker 子 shell——worker 当时
正 `wait` 在 python 上，python 一死它就**接着起了队列里的下一个分片**（shard02/03）。
正确顺序是**先杀 worker 子 shell、再杀 python**（都按 PID，`ps -p` 复核，禁 pgrep）。
中止的 B=64 train 日志/marker 归档在 `logs/aborted_b64_train/`。

## 4-ter · 实际启动方式（2026-08-05 12:51 / 12:54，错峰起卡）

主 agent 裁定：checkpoint-4976、只跑 `two_segment`、B=64 放行。启动时 GPU0 空闲、
**GPU1 仍在跑 Where-A `bench_calibration`**，于是：

* 双卡驱动第一次提交在 12:49:23 被自己的 busy-GPU 门**正确拦下**（`GPU 1: 2255 MiB`，
  比我人工看到 0 MiB 晚 4 秒——正是这个门存在的理由）。日志留档为
  `logs/driver_two_segment.refused-gpu1-busy.log`，**一个进程都没起**；
* 给驱动加了 `TASK_STRIDE`/`TASK_OFFSET`（错峰起卡用）：两个单卡驱动各取任务表的一半，
  **切出来与双卡驱动分配的队列逐条相同**（已 DRY_RUN 双向验证）。
  两个驱动都 `DO_MERGE=0`——谁都看不到对方的分片，谁都不许合并；
* GPU0 12:51:11 起 slice 0/2（pid 1963934）；GPU1 挂一个等待器（`ps -p` 盯 bench PID +
  轮询显存，**禁 pgrep**），bench 一退就在 12:54:47 起 slice 1/2（pid 1964758）。

⇒ **全部 12 个分片跑完后，必须手工执行一次合并**（驱动不会自动做）：

```bash
cd /home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/genctx_jobs
for s in V_where V_what train; do
  n=2; [ "$s" = train ] && n=8
  /home/bc/envs/q3vl_sft/bin/python merge_genctx.py --split "$s" --mode two_segment \
    --shard-root /mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/_shards/two_segment/"$s" \
    --num-shards "$n" 2>&1 | tail -5
done
```

合并会顺带建好 CX-1 兼容软链并把 checkpoint 写进交付报告顶层（CX-2 审计字段）。

## 4-quinquies · two_segment 合并结果（2026-08-05 20:09–20:28，CPU）

12/12 分片 rc=0 后按 §4-ter 合并，三个 split 全部 `MERGE-OK`：

| split | n_samples | 覆盖 | manifest | shard 文件 | 大小 | 合并耗时 |
|---|---|---|---|---|---|---|
| `V_where` | 896/896 | 1.0 | complete | 1 | 6 MiB | 21 s |
| `V_what` | 897/897 | 1.0 | complete | 1 | 6 MiB | 13 s |
| `train` | **159,215/159,215** | 1.0 | complete | 2 | 1,116 MiB | 1,077 s |

四类断言全过：每个分片的 sample 集**恰好等于**它该分到的那一片（train 是 7×19,902 + 19,901）、
并集精确覆盖、schema 全 `genwhere/2`、mode 全 `two_segment`、
**checkpoint 全 split 唯一 = `/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976`**（CX-2 审计字段，
写在三份交付报告顶层）。

合并后全量统计（159,215 条 train）**依然全满分**：`format_failure 0.0`、`color_format_failure 0.0`、
`both_segments_well_formed 1.0`、`starts_with_where_open 1.0`、`starts_with_color_open 1.0`、
`truncation 0.0/0.0`、`segments_overlap 0.0`、`stop_reasons` 全 `closed`。
段长 `where` p50 8 / p95 52 / max 72，`color` p50 171 / p95 222 / **max 325**
（固定边界 96 / 384，都没碰到，即**一条都没被截断**）。

**CX-1 软链已建**（`<split>/two_segment -> ../<split>`，三个 split 各一条；
`forced_color` 那条报 `skipped_target_missing`，因为当时还没产出，合并 forced_color 时会补上）。

**两个真实消费方在生产产物上实测读通**（不是 mock）：

```
Where-B   run_where_b.assert_genctx_coverage   train 159215/159215 missing=0（V_where/V_what 同）
Stage-What ColorGenContextStore.assert_covers  train coverage=1.0，走的就是 CX-1 软链，
                                               probe schema=genwhere/2 mode=two_segment
```

⇒ **Where-B 八臂与 Stage-What 十臂（T01–T08 + C03/C04）的 genctx 前置已解除。**

## 4-sexies · forced_color 作业（2026-08-05 20:30 起，单卡 GPU1）

GPU0 被 Where-A `run_calibration --arm BA-1-Band` 四臂序列占用（**未触碰**），
故 forced_color 单卡跑：`BATCH_SIZE=128 GPUS=1 DO_MERGE=0`，10 个分片（V_what 2 + train 8）。
范围依 WT-J10 = `train` + `V_what`（Where-B 不读 forced-prefix 档，V_where 不产）。

**日志路径已固定，不会再改名**（B=128 那次改名让主 agent 监控失联半小时）：

| 用途 | 路径（都在 `genctx_jobs/`） |
|---|---|
| driver 日志 | `logs/driver_forced_color.log` |
| driver PID | `logs/driver_forced_color.pid`（= **2711342**） |
| 分片日志 | `logs/forced_color_<split>_shard<i>of<n>.log` |
| 分片 marker / rc | `logs/forced_color_<split>_shard<i>of<n>.job.marker` / `.rc` |
| 分片报告 | `reports/forced_color/<split>/shard<i>of<n>/genctx_<leaf>.json` |

实测：V_what 两片已 rc=0（897 条）；train shard00of08 **3.77 samples/s**、单片 eta 5,213 s。
⇒ 8 片 × 5,213 s ≈ **11.8 h，预计 2026-08-06 08:20–08:30 完成**（单卡；比 two_segment 的
4.4 /s 慢，是 forced-prefix 下 color 段自身的长度分布所致）。

**若 GPU0 的 Where-A 序列提前结束、想减半**：先按正确顺序停当前 driver
（**先杀 worker 子 shell、再杀 python**，按 PID + `ps -p`），再起两个
`TASK_STRIDE=2 TASK_OFFSET=0|1` 的 driver——**已发布的分片会自动跳过，不重跑**。

**合并命令（待主 agent 触发）**：

```bash
cd /home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/genctx_jobs
for s in V_what train; do
  n=2; [ "$s" = train ] && n=8
  /home/bc/envs/q3vl_sft/bin/python merge_genctx.py --split "$s" --mode forced_color \
    --shard-root /mnt/nfs/bc/data/datasets/where_b-20260805/genwhere/_shards/forced_color/"$s" \
    --num-shards "$n"
done
```

它会把 `<split>/forced_color -> ../<split>-forced_color` 这条 CX-1 软链补上
（two_segment 合并时它报的是 `skipped_target_missing`）。

## 5 · 发现的问题

### CX-1（阻塞级，路径口径冲突）——两个消费方对同一份产物的路径约定不一致

* Where-B：`GENCTX_DIR/<split>`，写死在 `run_where_b.py` L128，无 CLI 覆盖（V11）；
* Stage-What：`<root>/<split>/<mode>`，`run_what.py` L223 / `evaluate_what.py` L179（V12）。

即 `genwhere/train` 必须**同时**是"一个已发布数据集根"和"一个按 mode 分的目录"。
`PREFLIGHT_WHERE_B_PENDING.md` §S2 写的产物路径是前者，WT-J10 写的是"每 split × 每 mode 一套"，
双方都没发现对不上。

**本目录采用的解法**（零代码改动、零重复 GPU 开销）：真实发布放 Where-B 的位置，
再在里面放两个兼容软链——

```
genwhere/train/                     ← 真发布（two_segment），Where-B 直接读
genwhere/train/two_segment  -> ../train
genwhere/train/forced_color -> ../train-forced_color
genwhere/train-forced_color/        ← 真发布（forced_color）
```

消费方只看 `manifest.json`/`indexes/`/`shards/`（V4），多出来的软链是惰性的。
已在 dry-run 里用**真的** `ColorGenContextStore(link, mode=...)` + `assert_covers` 验证读得通，
且 `mode` 不匹配照样硬停。**备选**是给 `run_where_b.py` 加一个 `--genctx` 参数（一行），
但那是改 q3vl 既有文件，本任务卡禁止 → 见 §6 决策 D3。

### CX-2（静默级）——没有人校验 genctx 的 checkpoint 与训练用的 checkpoint 一致（V15）

`assert_genctx_coverage` 只查覆盖，`ColorGenContextStore` 只查 schema/mode/覆盖。
若 genctx 用 2488 生成、而 `run_where_b.py`/`run_what.py` 走默认的
`SFT_CHECKPOINT = .../checkpoint-4976`，则"生成上下文"的 ids 来自 A 模型、重放它的冻结 VLM 是
B 模型——**loss 里完全看不出来**，只会表现为"generated 档不如 gt 档"，和 s 缓存契约里
"第二种失败模式是静默的"同构。

本目录的对策：合并断言 checkpoint 全 split 唯一，并把它写进 `genctx_<leaf>.json` 的顶层字段，
作为可审计的凭据。**主 agent 裁定 2488 时，必须同时给 Where-B / What 的启动命令显式传同一个
`--checkpoint` / `--sft-checkpoint`。**

### CX-3（效率）——`max_new_tokens=512` 与批生成相乘

HF `generate` 等整批结束。不终止率 1–2% 时，B=64 下大多数批都会跑满 512 步，
而"正常"只需要 ~330 步。可选缓解见 §6 决策 D2。

### CX-5（**新出现，需主 agent 注意**）——左 padding 的批/单不一致

链上 GPU0 验证作业 11:52 的实时日志：

```
[verify] checkpoint-2488: LEFT-PADDING MISMATCH 1/4 -- falling back to batch_size=1
```

即同样 4 个样本，batch=8 左 padding 生成的文本与逐条生成的文本有 1/4 不同。贪心解码是
前缀确定性的，一旦某个位置两个 logit 接近、bf16/FA2 在不同 batch 形状下的规约顺序变了，
后面整段就分叉——这类差异本身不算 bug，但它意味着：

* **生成作业的产物依赖 batch 组成**。注意这与分片**无关**：生产脚本本来就按 `--batch-size`
  批量生成，任何 batch 边界的变化都会改变一部分记录。分片只是多了几条批边界。
* 验证作业为安全起见回退到 `batch_size=1`；**生产作业不能这么做**——B=1 实测每样本 8–12 s，
  160k 条要 400+ 小时。
* 因此本作业的立场必须**写明**：`genwhere/2` 是"该模型在给定 batch 配置下的贪心输出"，
  不是"逐条生成的唯一真值"。`job.marker` 与分片报告都记了 `batch_size`，驱动对一个 split
  的全部分片用同一个值。若主 agent 认为这条对论文口径有影响，请在 D5 里一并裁定。

**验证作业跑完后必看**（它是本作业唯一的真实先验）：
`s0_base_sft/checkpoint_verification.json` 的 `summary.generated_tokens` 分位数与
`truncated` / `terminated` 比例——**若被选中的 checkpoint 大量不吐 EOS，S 就恒等于
`max_new_tokens`，墙钟直接落到上表的悲观档**，D2 的取值也随之变得值钱。

### CX-6（口径 nit，不影响正确性）

`generate_where` 返回的是整批统一宽度的行，提前结束的样本尾部是 pad/eos（V17）。
下游 `extract_segment` 先切 eos，所以**上下文没被污染**；但记录里的 `n_generated_tokens`
与 summary 的 `generated_tokens` 分位数实际是"该批的解码步数"。读报告时不要把
`generated_tokens p50 ≈ 500` 误读成"模型在胡说"——真正的段长看 `where_tokens` / `color_tokens`。

### CX-4（已在本目录修复的坑）

* 用绝对路径跑脚本时 `import q3vl` 失败（`python /abs/x.py` 把**脚本目录**放 `sys.path`，
  `cd $REPO` 不管用，只有 `-m` 才加 cwd）。dry-run 抓到，两个脚本都显式插了仓库根；
* `publish_generated` 自己会调 `genwhere_payload`，传 payload 进去会 `TypeError`。dry-run 抓到。

---

## 6 · 待主 agent 决策

| id | 事项 | 保守默认（已采用） | 备选 |
|---|---|---|---|
| **D1** | 排期：这不是空档作业，`two_segment` 两卡 4–7 h，`forced_color` 再 4–7 h | 先只跑 `two_segment`（解锁全部 8 个 Where-B 臂 + 12 个 What 臂里的 10 个）；`forced_color` 只服务 C01/C02，等这两臂真的排上再跑 | 两个模式连着跑，占卡 8–14 h |
| **D2** | `MAX_NEW_TOKENS` | **512**（协议 `GEN_MAX_NEW_TOKENS`，`whereb/config.py` 有出处） | 416（仍 > GT max 330，不会截断合法输出，最坏情况省 ~19%）。属于改协议常量，不静默拍板 |
| **D3** | CX-1 的解法 | 兼容软链（零改动，已验证） | 给 `run_where_b.py` 加 `--genctx`（改 q3vl 既有文件，需实现审阅） |
| **D4** | checkpoint | 脚本参数化，等裁定 | 一旦选 2488，Where-B/What 启动命令必须显式同步（CX-2） |
| **D5** | `BATCH_SIZE` | 64 | 标定后按实测上调（96/128 显存都够）；改动只需重跑未完成的分片 |
| **D6** | 是否给 T_final / T_lut_unseen 也生成 | **不生成**。`evaluate_what.py` 默认 `--split V_what`，且 `--allow-split` 明确写着"打开 T_final 必须是有意为之（协议 12.4）" | 最终测试板需要时再补，届时是 ~4,700 条的小作业 |

---

## 7 · 交付物

| 文件 | 作用 |
|---|---|
| `genctx_shard.py` | 单分片 runner（切 `ds.refs` + 强制隔离 out-root/report-dir，其余透传） |
| `merge_genctx.py` | 分片合并 + 四类断言 + 兼容软链 + 交付报告 |
| `genctx_dual.sh` | 两卡驱动：每卡一个分片队列，**逐作业** D-20 四步，可续跑，收尾自动合并 |
| `genctx_forced_color_dual.sh` | 同一驱动，`MODE=forced_color`、`SPLITS="V_what train"` |
| `dryrun/run_dryrun.sh` `dryrun/test_shard_and_merge.py` | 79 条断言，CPU-only |
| `dryrun/dryrun.log` `dryrun/DRYRUN_RESULT.md` | dry-run 记录 |
| `dryrun/cpu_smoke_real_producer.log` | **真生产脚本**在 CPU 上跑通两个分片的实跑日志（`--limit 4`、base 权重、`--max-new-tokens 8`） |
| `README.md` | 启动命令卡片 + 监控 |
| `logs/` `reports/` | 运行时产出（现在是空的） |
