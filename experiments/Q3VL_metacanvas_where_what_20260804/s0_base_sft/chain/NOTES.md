# EXEC-1 · SFT 收尾链式启动 —— 实施记录

目标：Base SFT 一退出，两张 H100 立刻接上后续作业（GPU 零空闲），并在接手前先判定训练
是否真的正常结束。

---

## 1. 实施前核实记录

本任务**不引用任何外部 URL / 论文数字 / 第三方 API**，全部事实来自本机与本仓库，逐条实测如下
（命令与输出都在本次会话里跑过）。

| # | 事实 | 怎么核实的 | 结果 |
|---|---|---|---|
| 1 | 两个 rank 进程与命令行 | `ps -p 3395226,3395227 -o pid,ppid,etime,args` | 存活，cmdline 含 `-m q3vl.train.train_sft`，父进程 3395099 |
| 2 | 训练进度与 ETA | `tail train.log`；`job.marker` | 提交时 step 3663/4976，5.2 s/step；marker 的 ETA `2026-08-05T11:29` |
| 3 | `checkpoint-2488` 的实际文件清单 | `ls -la` | model×2 + index + tokenizer/processor/chat_template + `trainer_state.json` + `protected_checkpoint.json` + `global_step2488/`；磁盘占用 **56 GB**（含 ZeRO 优化器分片），权重本身 8.27 GiB |
| 4 | checkpoint 自带 eval 记录 | 解析 `checkpoint-2488/trainer_state.json` | 5 条 eval（step 500…2488），含 `eval_loss` 与 `eval_seg_*`；**不需要重跑 eval** |
| 5 | checkpoint 的 tokenizer 是否带四个特殊 token | `added_tokens.json` | `<where>`151669 `</where>`151670 `<color>`151671 `</color>`151672，齐 |
| 6 | checkpoint 的 `generation_config.json` | 直接读 | `do_sample: true, temperature 0.7, top_p 0.8, top_k 20` → **诊断必须显式压成贪心**，否则不可复现 |
| 7 | checkpoint 的 `config.json` 的 `use_cache` | `jq` | 顶层 `false`（训练期梯度检查点留下的），`text_config.use_cache` 为 `true` → 生成前把两处都置 True 并给 `generate(use_cache=True)` |
| 8 | `setup_model()` 会不会破坏 checkpoint | 读 `q3vl/train/modeling.py` + `tokens.py` | 会：`prepare_embeddings(reinit_new_rows=True)` **重置四行特殊 token 嵌入**。故验证脚本一律用裸 `from_pretrained`，不走 `setup_model` |
| 9 | V_where 评测集规模 | `ShardIndex.load` | 896 条（`wc -l` 一致） |
| 10 | GT assistant 目标长度分布（就是抽的那 64 条） | 用真实 processor + collator 逐条 `encode_one` | min 153 / p50 **210** / p90 251 / p99 310 / **max 326** tokens；prompt 311–546 → `max_new_tokens=448` 有 37% 余量 |
| 11 | 颜色段的结构 | `s0_preflight/data/metrics.json` 的 `color_is_six_bodies_in_order` + `constants.CANONICAL_COLOR_FIELDS` | 颜色段恰是 **6 个 body 换行拼接**，所以「预测行数是否为 6」比 token-F1 更有判别力，两个都报 |
| 12 | `train_sft.py` 结尾会不会写根级 `trainer_state.json` | 读源码结尾 | `trainer.save_model(); trainer.save_state()` → 会写；`SYNC_PROTECTED_TO_NFS.sh` 也在没有它时硬拒绝 |
| 13 | maskviews 默认是否包含 `winner_confidence=low` | 读 `extract_maskviews.py:74` 的 `exclude_low=not (args.include_low or not EXCLUDE_WINNER_CONFIDENCE_LOW)` + `config.EXCLUDE_WINNER_CONFIDENCE_LOW=False` | **默认已包含 low**，符合 D1；不需要也不应该加 `--include-low` |
| 14 | maskviews 目标目录是否已存在 | `ls /mnt/nfs/bc/data/datasets/where_a-20260805/` | 不存在 → 原子发布不会被"目录已存在"硬拒 |
| 15 | 五个 split 的 index 都在 | probe 里逐个 `split_index_path().is_file()` | train 159,215 / V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433 行 |
| 16 | `q3vl.where` 的三个入口 argparse | `--help` 实跑 | `preflight`（`--device --limit --checkpoint`）、`sweep_upsample`（`--limit --checkpoint --basis`）、`extract_maskviews`（`--split`）全部 rc=0 |
| 17 | `sqlite3` 在战役环境里的 CXXABI 问题 | 带 `LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib` 实跑 probe | `sqlite3=3.51.2` 正常 import；两个需要 build catalog 的作业都带上了这一行 |
| 18 | Qwen3-VL 的 mrope 是否正确处理左填充 | 读 `transformers/models/qwen3_vl/modeling_qwen3_vl.py::get_rope_index` | 逐样本 `input_ids[attention_mask[i]==1]` 后再算位置，左填充安全。**但仍然不假设**：见 §3 的一致性断言 |
| 19 | 本机 shell 与 bash 版本 | `bash --version` | bash 5.1.16（关联数组、`${arr[@]}` 空展开在 `set -u` 下都安全） |
| 20 | 磁盘容量 | `df -h` | 本地 1.2 T 可用、NFS 23 T 可用 —— 同步 ~120 GiB 有余量 |

---

## 2. 三个作业的定义与依据

| 作业 | 卡 | 做什么 | 依据 |
|---|---|---|---|
| `gpu0_ckpt_verify` | GPU0 | 对 checkpoint-2488 / 4976 各做：离线生成诊断（V_where 64 条，结构率 + 与 GT 的粗略对齐）+ 从 `trainer_state.json` 汇总 assistant-only eval_loss | 主 agent 裁定 **D-J3**（训练内生成诊断关闭，spec 8.3 结构率离线补） |
| `gpu1_wherea_s2` | GPU1 | Where-A **S2**：`q3vl.where.preflight --device cuda`（WA-P4b 冻结泄漏、WA-P4e F_pre/guided upsample 真实高分辨率通路、WA-P5 basis 条件数 + **L-BFGS GPU 吞吐实测**）→ `sweep_upsample`（D5 radius/eps 扫描，域 gate 字典序） | `where_a/PREFLIGHT_WHERE_A_PENDING.md` §S2 (a)(b) |
| `cpu_sync_maskviews` | CPU/IO | 先 `SYNC_PROTECTED_TO_NFS.sh`（protected checkpoint 落 NFS + 双侧 sha256），再 Where-A **S1** 五个 split 的 mask 视图打包 | 主 agent 裁定 **D-J1**；`PREFLIGHT_WHERE_A_PENDING.md` §S1 |

**为什么 S2 可以直接用 checkpoint-4976，不等 checkpoint 选择结论**：Base SFT 冻结了
`patch_embed` / `pos_embed` / 24 个 vision block（`q3vl/train/freeze.py`，`run_setup.json` 的
`frozen_groups` 实测确认），F_pre 因此对 SFT 不变；`q3vl/where/config.py` 的注释与
`WA-P4b` 检查本身就是为了把这句话钉死。**S2 反过来是这条推论的验证**，所以它必须跑，而不是
被推论豁免。

---

## 3. 实现上的几个硬点（都是能静默出错的地方）

1. **不能用 `setup_model` 载 checkpoint**（核实 #8）。用了就把训练出来的四行特殊 token 嵌入
   重新初始化掉，而生成诊断照样能跑出「结构率低」的数字 —— 静默失效。
2. **必须显式压成贪心**（核实 #6）。checkpoint 的 `generation_config` 是采样配置，不压就不可复现。
3. **必须显式开 KV cache**（核实 #7）。顶层 `use_cache=False` 是训练留下的，不改就是二次方解码。
4. **左填充不假设、要断言**：`--consistency-k 4` 条样本**同时**单条生成和放进 batch 生成，
   字符串必须逐字相同；不同就**自动整轮回退 `batch_size=1`** 并把回退记进 metrics。
   这是 mrope + 左填充唯一会静默出错的地方，而它的表现形式恰好是"模型好像没学会格式"。
5. **生成文本必须在第一个终止符处截断**再解析：HF 在序列结束后用 `pad_token_id`(151643) 填满
   整个 batch 宽度，不截断的话 `parse_two_segment` 会看到一堆 `<|endoftext|>`。
6. **`max_new_tokens` 由实测定档**（核实 #10），不是拍的：448 = GT 最长 326 + 余量。
   定得太小会把"截断率"这个信号变成"我自己设的上限"。
7. **红线：checkpoint 选择禁用 val loss**。eval_loss 只作为 book-keeping 出现在报告里，
   报告显式写明"本报告不做 checkpoint 选择"。

---

## 4. 假设与待确认清单

### 4.1 已自行核实、不需要主 agent 介入

- V_where 是本次 SFT 的 eval split（`sft_base.yaml` 的 `eval_index`），生成诊断用同一个 split
  没有引入新的数据接触面；`T_final` / `T_lut_unseen` 一次都没被指向。
- 抽样是 `random.Random(42).sample(range(896), 64)` 后排序，**两个 checkpoint 用完全相同的
  下标**，两列可比。下标已落进 `checkpoint_verification.json` 的 `settings.sample_indices`。
- maskviews 默认包含 low（核实 #13），与 D1 一致，脚本里不加 `--include-low`。

### 4.2 待主 agent 决策（已采保守默认继续，**没有静默拍板**）

| # | 决策点 | 两种做法 | 采用的保守默认 |
|---|---|---|---|
| **D-E1** | 完成判定要不要把「根级 `trainer_state.json` 存在」也算进去 | 任务卡只列了 checkpoint-4976 + global_step + 日志无异常 | **算进去 → 缺失即 CHAIN-ABORT**。缺它意味着 `save_state()` 没跑完，而 `SYNC_PROTECTED_TO_NFS.sh` 本来就会拒绝；早失败比派了一个注定失败的作业好 |
| **D-E2** | preflight 非 0 时还要不要跑 `sweep-upsample` | 串行短路 / 都跑 | **都跑，两个 rc 分开记**。`WA-P4e` 有可能合法地卡在预注册的 `S_OOD_FRAC_MAX` 门槛上，而 sweep 正是重定 D5 的工具；短路会浪费一整段 GPU 时间 |
| **D-E3** | 五个 split 的打包顺序 | 按 pending 文档的 train 优先 / 小 split 先做冒烟 | **V_where → V_what → T_final → T_lut_unseen → train**。循环本身无依赖；先花两分钟证明打包器能跑通，再投小时级的 train |
| **D-E4** | 生成诊断抽样量 | 任务卡要求 ≥64 | **正好 64**（墙钟见 §6，还有大量余量，主 agent 若要 128/256 直接改 `N_SAMPLES` 即可） |
| **D-E5** | 某个作业起不来时是否整链中止 | 中止 / 继续派其余作业 | **继续**。GPU 零空闲是本任务的第一目标，一个作业起不来不该连累另一张卡；`ALL-DISPATCHED` 行里带 `failed_to_start=` 计数 |
| **D-E6** | 报告要不要给出"选哪个 checkpoint" | 给 / 不给 | **不给**。红线禁止用 val loss 选，结构率两档若都满分也没有区分度；报告把两列并排摆出来，裁定交主 agent 结合下游指标 |

---

## 5. 测试记录（全部真实跑过，未占用 GPU，未写生产目录）

### 5.1 链逻辑 dry-run —— **43 passed / 0 failed**

`dryrun/run_dryrun.sh` 跑的是**真实的** `chain_after_sft.sh`：真实的等待循环、真实的完成判定、
真实的 D-20 四步派发、真实的监控循环。只替换了三样东西：rank 进程 → `mock_train_sft.sh`，
三个作业 → `stub_job.sh`（同一套 PROBE-OK / phase-rc / job-rc 协议），run 目录 → 假目录。

| case | 覆盖的分支 | 结果 |
|---|---|---|
| `happy` | 等待（rank 还活着）→ 判定通过 → 三个 STARTED → ALL-DISPATCHED → 逐 phase DONE → 三个 DONE → ALL-DONE，chain rc=0，job.marker 落盘 | 15/15 |
| `abort_step` | `global_step=3500 != 4976` → CHAIN-ABORT，**一个作业都没起**，rc=2 | 4/4 |
| `abort_traceback` | 日志尾部有 `Traceback` → ABORT | 3/3 |
| `abort_oom` | 日志尾部有 `CUDA out of memory` → ABORT | 3/3 |
| `abort_missing_ckpt` | checkpoint-4976 目录不存在 → ABORT | 3/3 |
| `abort_no_root_state` | 根级 `trainer_state.json` 缺失（D-E1）→ ABORT | 3/3 |
| `abort_missing_file` | checkpoint 少 `preprocessor_config.json` → ABORT | 3/3 |
| `job_failure` | gpu1 退出码 3 → 该作业 `FAILED rc=3` + 其 phase 也标 FAILED，另两个照常 DONE，`ALL-DONE done=2 failed=1`，chain rc=1 | 6/6 |
| `pid_reuse` | rank PID 指向一个**活着但不是训练**的进程 → cmdline 不匹配 → 判定为已退出并继续 | 3/3 |

完整记录：`dryrun/DRYRUN_RESULT.md`，每个 case 的 chain.log 在 `dryrun/cases/<case>/logs/chain.log`。

### 5.2 作业脚本自身的冒烟

| 测试 | 命令 | 结果 |
|---|---|---|
| `verify_checkpoints.py --help` | argparse 解析 | rc=0 |
| `verify_checkpoints.py --dry-run --steps 2488 --n 8`（无 GPU，写 scratch 目录） | 真实 shard index + ShardStore + `AutoProcessor.from_pretrained(checkpoint)` + `verify_single_token` + `Sft2SegCollator.encode_one` | rc=0：完整性 ok、`global_step=2488`、5 条 eval 记录、4 条样本编码成功（prompt 446–466 / where 8–53 / color 161–176 tokens） |
| `q3vl.where.preflight --help` / `sweep_upsample --help` / `extract_maskviews --help` | argparse | 三个都 rc=0 |
| `job_gpu0_ckpt_verify.sh`（`CHAIN_GPU0="" PROBE_ONLY=1`） | probe 走到底再 fail-closed | 打印 torch/transformers 版本与 checkpoint 权重 8.27 GiB，然后 `no CUDA device visible -- refusing`，rc=1，rc 文件正确落盘 |
| `job_gpu1_wherea_s2.sh`（同上，`CKPT=checkpoint-2488`） | sqlite3 + config 常量 + checkpoint + fail-closed | `sqlite3=3.51.2`、`S_DOMAIN=(-3.0,3.0) S_OOD_FRAC_MAX=0.01 GUIDED_PARAMS_PROVISIONAL=True`，然后 fail-closed，rc=1 |
| `job_gpu1_wherea_s2.sh`（默认 CKPT=4976，尚不存在） | 缺 checkpoint 的分支 | `checkpoint unusable`，rc=1 |
| `job_cpu_sync_maskviews.sh`（`SKIP_SYNC=1 SKIP_MASKVIEWS=1`） | probe | df/du + sqlite3 + 五个 split index 全部 `exists=True`，`PROBE-OK`，rc=0 |

### 5.3 测试过程中自己踩的两个坑（都已修，记在这里免得复发）

1. **`${VAR:-default}` 把"显式的空值"吃掉了。** 第一版 wrapper 写的是
   `CUDA_VISIBLE_DEVICES="${CHAIN_GPU0:-0}"`，于是我用 `CHAIN_GPU0=""` 做的"无 GPU 冒烟"
   实际拿到了 `CUDA_VISIBLE_DEVICES=0`，验证作业在训练还在跑的 GPU0 上真的起来了，跑了约 9 分钟
   才被 `timeout` 收掉。**事后核查：训练两个 rank 均存活（step 由 3663 推进到 4000+，
   `nvidia-smi` 只剩两个训练进程，显存回到 54070 MiB），真实交付目录
   `s0_base_sft/` 下没有产生任何 `checkpoint_verification*` 文件（脚本只在最后统一落盘）。**
   修法：`${CHAIN_GPU0-0}`（只在**未设置**时取默认）+ 新增 `PROBE_ONLY=1` 显式闸门，
   两道锁，冒烟测试再也够不到重活。
2. **`$( )` 里起的后台进程会把命令替换的管道按住。** dry-run 里
   `pids="$(start_mock_ranks 6)"` 一开始阻塞了整整 6 秒才返回，等 chain 起来时 mock rank
   已经死了，"等待中"这条分支根本没被覆盖（表现为 1 条断言 FAIL）。修法：后台进程
   `> /dev/null 2>&1 < /dev/null`，让它不再持有替换管道。**同源提醒**：链本身用
   `nohup ... > log 2>&1 &` 提交，三个作业各自重定向到自己的日志，因此没有这个问题。

---

## 6. 墙钟预估

| 作业 | 预估 | 依据 |
|---|---|---|
| `gpu0_ckpt_verify` | **8–15 分钟**（最坏 ~20） | 模型加载 8.27 GiB×2 次（S0-JOINT 实测同规模模型 68.5 s）+ 64 条×2 档生成。目标长度 p50 210 / max 326 token（实测），batch 8 贪心 + KV cache，空闲 H100 上 8 个 batch ≈ 1 分钟/档；左填充一致性检查另加 ~25 s/档。若一致性检查不通过而回退 `batch_size=1`，生成时间约 ×4，总计仍 <25 分钟。**任务卡要求 ≤40 分钟，满足。** |
| `gpu1_wherea_s2` | **30–60 分钟**（未实测，第一次上 GPU） | pending 文档给 preflight「约 5–10 分钟」（`--limit 32`）；`sweep-upsample` 是 3×3=9 组设置 × 24 条真实样本的交付分辨率评估，无先例数字。**L-BFGS 的 GPU 吞吐正是这一步要产出的量**（CPU 实测 1.816 s/拟合、外推 76.2 CPU-小时/臂），它决定 S5 是否要重裁 D3 |
| `cpu_sync_maskviews` | **sync 40–60 分钟；maskviews 1–4 小时**（长尾） | sync：实测 checkpoint-2488 占 56 GB（含 ZeRO 分片），两个 protected + 根级 ≈ 120 GiB，脚本按实测 103 MB/s 估 ~20 分钟 rsync，另加双侧 sha256（NFS 侧再读一遍 120 GiB）。maskviews：train split 159,215 行、过滤后约 75.5k 条，每条要解一张 `.cgt.png` 并投两个视图，**无实测吞吐**，按 20–100 ms/条外推 |

三个作业并行，**关键路径是 CPU/IO 那条**；两张 GPU 在 SFT 退出后 3 秒内就被占上。

---

## 7. 交给主 agent 的监控接口

- `chain.log`：`experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/chain/logs/chain.log`
- 状态行一律以 `CHAIN:` 开头，可直接 grep：
  - `CHAIN: <job> STARTED pid=<pid> log=<path>`
  - `CHAIN: <job>/<phase> DONE|FAILED rc=<n>`
  - `CHAIN: <job> DONE|FAILED rc=<n> elapsed_s=<n> log=<path>`
  - `CHAIN: ALL-DISPATCHED started=<n> failed_to_start=<n>`
  - `CHAIN: ALL-DONE done=<n> failed=<n>`
  - `CHAIN: CHAIN-ABORT reasons=<n>` + 每条 `CHAIN: CHAIN-ABORT reason: <...>`
- 其余诊断行以 `[chain <时间戳>]` 开头，不会与状态行混淆。
- 每个作业另有 `logs/<job>.job.marker`（pid + 完整命令 + 日志路径 + rc 文件路径 + phase 目录）
  与 `logs/<job>.rc`。
