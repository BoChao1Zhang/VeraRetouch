## 要验证的结论

如果这次联合 preflight 成立，我们就能说「Base SFT 的正式训练在真实数据 × 两卡 ZeRO-3 上没有**运行层**障碍：loss/梯度有限、显存有余量、eval 跑得通、checkpoint 存得下也读得回、里程碑步数由 manifest 现算而非硬编码」；失败就不能说，只能说单卡 mock 通过——而单卡 mock 恰恰**已经被证明不足以覆盖 ZeRO-3**（本报告 §6 的两个缺陷就是单卡全过、两卡才现形的）。

## 为什么需要验证它

`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` §9 的最后一句是「任一项失败时不得把 smoke 结果升级为正式训练」。这次是把 spec §9 里**只能在真实数据 + 真实并行度下取证**的那几项（项 10/11 的两卡版、项 12 的容量与路径、§8.1 的步数等式、§8.2 的 protected checkpoint）从"设计上应该对"变成"实测对"；不做就等于用一个 9 小时、写 600 GB 的任务去当第一次集成测试。

## 怎么验的

在 `/mnt/nfs/bc/data/datasets/sft2seg-20260804`（terminal manifest digest `9278d721…`，train `N_effective=159,215`，eval split `V_where` 896 条）上跑了三段**短程** 2×H100 ZeRO-3 smoke——Phase A 用**逐字生产 config**（只改 `output_dir`）跑 74 步测吞吐/显存/步数等式，Phase B 用缩短的 `max_steps=8` 闭合 eval + 存盘 + 保护性删除，Phase C 从 Phase B 的 protected checkpoint 恢复——外加一次单卡非 ZeRO 的生成管线单验；正式训练**未启动**。

---

# S0-JOINT 联合 preflight 报告

> 日期：2026-08-05　｜　机器可读输出：`metrics.json`　｜　假设与决策：`NOTES.md`
> 规格：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` §7 / §8 / §9
> 前置：`../model/PREFLIGHT_MODEL.md`（S0-TRAIN，已按主 agent 指令追加 ADDENDUM）、`../data/PREFLIGHT_DATA.md`（S0-DATA）
> **状态：正式训练未启动。** 三段 smoke 全部已停止，`nvidia-smi` 实证两卡 0 MiB、无残留 compute 进程。

## 0. 结论一览

| 任务卡必做项 | 内容 | 结果 |
|---|---|---|
| 1a | 两卡 ZeRO-3 真实数据短程 smoke：loss 有限且下降、梯度有限 | **PASS** |
| 1b | 两卡吞吐（实测 s/step）与峰值显存 | **PASS** — 5.078 s/step，52,973 MiB / 97,871 MiB |
| 1c | eval on `V_where` 跑通 | **PASS** — 896 条，50.8–56.3 s，`eval_loss` 2.598 → 2.170 |
| 1d | checkpoint 保存 → 中断 → resume 闭环 | **PASS** |
| 1e | protected checkpoint 步数逻辑（等式校验） | **PASS** — `ceil(159215/32)=4976`、`round(0.5×4976)=2488`，trainer 实算一致 |
| 2 | D-3 生成管线单验（单卡非 ZeRO，base 模型 3 样本） | **PASS**（管线可用；正式配置确认关闭） |
| 3 | 正式训练运行面规划（体量 / 容量 / 路径 / 时长）= spec §9.12 | **PASS**（本报告 §4，显式闭合审阅 B-4） |
| 4 | resolved config 落盘 + 启动脚本（不执行） | **PASS** |
| 5 | smoke 暴露的实现缺陷：修复 + 补单测 + 记录 | **2 个缺陷，均已修复**（本报告 §6），单测 68 → **75 passed** |

**overall: PASS（无 blocker）。D-J1 / D-J3 已由主 agent 于 2026-08-05 裁定，见 §10。**

## 1. 三段 smoke 的边界与授权

| 段 | config | 步数 | 目的 | 结束方式 |
|---|---|---|---|---|
| **Phase A** | **生产 `sft_base.yaml` 逐字**，只改 `output_dir` | 74（本可跑 4976） | 吞吐 / 显存 / loss / 步数等式 | 收到主 agent 边界澄清后**主动 kill**，逐 PID `ps -p` 实证退出 |
| **Phase B** | 同上 + `Q3VL_ALLOW_NONSPEC=1` 覆盖 `max_steps=8, eval_steps=4, save_steps=2, save_total_limit=1` | 8 | eval / 存盘 / 保护性删除 | 自然结束 |
| **Phase C** | 同 Phase B + `--resume_from_checkpoint <B/checkpoint-4> --log_level info` | 4→6 | resume 闭环 | 自然结束 |

- 三段的输出目录都在 `/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/`，**从未写入正式路径**
  `/mnt/nfs/bc/runs/q3vl_base_sft_20260804`；`smoke_launch.sh` 里有硬拒绝（命中该串直接 `exit 2`）。
- Phase A 的 `batch plan` 日志行 `'spec_violations': []`，Phase B/C 则逐条列出三处偏离——
  偏离在日志里是**显式可审**的，不是靠人记。
- Phase A 的 74 步只跑到 epoch 0.01，**没有产生任何可用于正式训练的状态**；正式训练必须从空目录重新开始。

## 2. 项 1a/1b：吞吐、显存、数值有限性（Phase A，生产 config）

出处：`logs/phaseA_2gpu_zero3_smoke.log`、`logs/phaseA_gpu_mem.csv`、`config/phaseA_run_setup.json`

| 指标 | 实测 |
|---|---|
| 稳态吞吐 | **5.078 s / optimizer step**（步 10→74 共 64 步；逐步 4–7 s，tqdm 1 s 分辨率） |
| 有效 global batch | 32 = 4/卡 × GAS 4 × world 2（`spec_violations: []`） |
| 峰值显存 | GPU0 **52,973 MiB** / GPU1 **52,355 MiB**（容量 97,871 MiB，**占用 54%**） |
| GPU 利用率 | 83% / 81% |
| 互联 | `nvidia-smi topo -m` = **PXB（PCIe switch，无 NVLink）** |
| 首步 | 16.09 s（含 warmup），第 2 步起即进入稳态 |

loss / grad_norm（`logging_steps=10`，步 10→70）：

| step | 10 | 20 | 30 | 40 | 50 | 60 | 70 |
|---|---:|---:|---:|---:|---:|---:|---:|
| loss | 3.4211 | 3.1362 | 2.6362 | 2.0339 | 1.6522 | 1.4589 | 1.3721 |
| grad_norm | 33.217 | 13.271 | 7.667 | 4.632 | 3.664 | 3.273 | 3.244 |
| lr | 6.00e-7 | 1.27e-6 | 1.93e-6 | 2.60e-6 | 3.27e-6 | 3.93e-6 | 4.60e-6 |

- **全程有限、无 NaN/Inf、单调下降**；lr 处于 warmup 线性段（`warmup_ratio 0.03 × 4976 ≈ 149` 步），与 spec §7.1 一致。
- 分段诊断也在跑（步 60：`where_acc 0.773 / color_acc 0.607 / eos_acc 1.000`），**不入总 loss**（spec §4.4）。

**结论**：spec §7.2 首选组合 `4 × GAS4` 在真实数据上成立，**无需**退到 `2 × GAS8`；PREFLIGHT_MODEL.md 项 11 的推荐不变。

## 3. 项 1c/1d/1e：eval、protected checkpoint、resume

### 3.1 步数等式（spec §8.1）

| 量 | 来源 | 值 |
|---|---|---|
| `N_effective` | terminal manifest（唯一权威） | **159,215** |
| `steps_per_epoch = ceil(N_effective / 32)` | 入口现算 | **4976** |
| trainer `state.max_steps` | HF 实算（生产 config，`num_train_epochs=1.0`） | **4976** |
| `half_epoch_step = round(0.5 × 4976)` | `trainer.py::resolve_protected_steps` | **2488** |
| `full_epoch_step` | 同上 | **4976** |

日志原文（Phase A）：

```
[protected-checkpoints] steps_per_epoch(from trainer state)=4976 -> half_epoch_step=2488, full_epoch_step=4976
```

两条独立路径（manifest 现算 vs trainer 状态）**给出同一个数**，`resolve_protected_steps` 的 ±1 容差没有被用到。
spec §8.1 的估算值 2645/5290 **一次也没有出现**。反向证据：Phase B/C 人为把 `max_steps` 改小后，
同一行立刻打出 `WARNING: manifest-derived ceil(N_effective/global_batch)=4976 differs from trainer max_steps=8` —— 该守卫是活的。

### 3.2 eval on `V_where`

| 轮次 | `eval_loss` | `eval_seg_assistant_loss` | 运行时间 | 吞吐 |
|---|---:|---:|---:|---:|
| Phase B step 4 | 2.5979745 | 2.6007838 | 56.33 s | 15.91 样本/s |
| Phase B step 8 | 2.1702332 | 2.1757313 | 52.09 s | 17.20 样本/s |
| Phase C step 6（resume 后） | 2.2815146 | — | 50.81 s | 17.63 样本/s |

- 896 条全量 eval，**一次未失败**；受监督 token 数逐轮一致（where 10,134 / color 83,207 / eos 896）。
- `eval_loss` 与自算的 `eval_seg_assistant_loss` 差 **0.003（0.1%）**。二者都是 **assistant-only**；
  差值来自平均口径：HF 的 `eval_loss` 是**逐 batch 取均值再平均**，分段累加器是**全局按 token 加权**。
  单卡 mock 里两者曾逐位相同，是因为那时只有一个 eval batch。**这不是 mask 差异**——
  受监督 token 计数三段相加 = 94,237，与 collator 的分段标记完全对应。

### 3.3 protected checkpoint 与两条删除路径（Phase B）

配置刻意选了 `save_total_limit=1` + `metric_for_best_model=eval_loss`：这是
transformers 4.57.1 **第二条删除路径**（`trainer.py:2840`，不经过 `_rotate_checkpoints`）的唯一触发条件，
也正是 S0-TRAIN 读源码才发现的那个静默删 protected checkpoint 的坑。

| 事实 | 值 |
|---|---|
| protected steps | `{4, 8}` |
| 实际写出的 checkpoint | 2, 4, 6, 8 |
| **训练结束后幸存** | **checkpoint-4, checkpoint-8**（两个 protected） |
| 被删除 | checkpoint-2, checkpoint-6（非 protected） |
| `best_model_checkpoint` | `.../phaseB/checkpoint-8` |
| `best_metric` | 2.1702332496643066 |
| 两个 protected 各带 `protected_checkpoint.json` | 是 |

**该修复此前只有单测背书，现在在真实 2 卡 ZeRO-3 路径上被正面验证。**

每个 protected checkpoint 的 sidecar 内容（spec §8.2 要求"带 tokenizer、processor、special-token map、训练状态和数据 manifest digest"）：

```json
{"global_step": 4, "protected": true, "milestone": "0.5_epoch",
 "special_token_ids": {"<where>": 151669, "</where>": 151670, "<color>": 151671, "</color>": 151672},
 "protected_steps": [4, 8], "best_metric": 2.5979745388031006,
 "data_manifest.digest": "9278d721bb1e234c0a6e7ad94f8f8ae1eab10844602a26a1cfb0e20cab7ac840",
 "data_manifest.n_effective_train": 159215,
 "freeze_report.total_params": 4437815808, "freeze_report.trainable_params": 4131573248}
```

### 3.4 resume 闭环（Phase C）

从 **Phase B 的 protected `checkpoint-4`** 恢复，输出写到一个**全新目录**（保住 Phase B 的证据）：

```
Attempting to resume from /mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/phaseB/checkpoint-4
  Continuing training from checkpoint, will skip to saved global_step
  Continuing training from epoch 0
  Continuing training from global step 4
  Will skip the first 0 epochs then the first 16 batches in the first epoch.
```

- **16 batches** = 4 步 × 4 GAS × 1 micro/rank — 数据跳过量与已完成步数**精确对应**；
- 恢复后 step 6 的 `eval_loss = 2.2815`，落在 Phase B 的 step 4（2.598）与 step 8（2.170）**之间**，
  说明恢复的是**优化器与模型状态**而不是重新初始化（重新初始化的话第一轮 eval_loss 会回到 6 量级）；
- checkpoint-6 正常写出，`training finished` 正常返回。

**ZeRO-3 checkpoint 从 NFS 恢复耗时：`02:56:19` → `03:05:24` = 545 s（约 9.1 min）**，
与 55.03 GiB / 103 MB/s 的读取上限一致。

## 4. 项 3 = spec §9.12：正式训练运行面（显式闭合审阅 B-4）

### 4.1 checkpoint 体量（实测，非估算）

一个 ZeRO-3 全状态 checkpoint = **59,084,057,355 B = 55.03 GiB**（26 个文件），逐项：

| 组成 | 字节 | 说明 |
|---|---:|---|
| `model-0000{1,2}-of-00002.safetensors` | 8,875,719,328 | 合并后的 bf16 全模型（`stage3_gather_16bit_weights_on_model_save`） |
| `global_stepN/bf16_zero_pp_rank_{0,1}_..._optim_states.pt` | 2 × 24,789,447,623 | ZeRO-3 分区的 fp32 master + Adam 两个矩 |
| `global_stepN/zero_pp_rank_{0,1}_..._model_states.pt` | 2 × ~306,703,900 | 含冻结参数分片与元数据 |
| tokenizer / processor / special-token map / chat template / config | ~15.9 MB | **spec §8.2 要求的随附物，逐个到位** |
| `trainer_state.json` / `scheduler.pt` / `rng_state_{0,1}.pth` / `latest` / `zero_to_fp32.py` | ~66 KB | 可 resume（§3.4 已实证） |
| `protected_checkpoint.json` | 7,411 | 本仓库自加的 sidecar |

### 4.2 写入吞吐与容量（实测）

| 目标 | `dd` 实测写吞吐 | 单 checkpoint 实测写入 | 剩余空间 |
|---|---:|---:|---:|
| `/mnt/nfs/bc/runs`（NFS4，durable） | 单流 104 MB/s，**双流聚合 103 MB/s** | **553.6 / 556.9 s**（≈ 106 MB/s，两次） | 23 TB |
| `/home/bc/data`（本地 LV，含机械盘 sdb） | 单流 185 MB/s（O_DIRECT 222） | 未实测（按 182 MB/s 折算 ≈ 325 s） | 1.5 TB |

**双流不比单流快** ⇒ 瓶颈是 NFS 链路本身（约 1 GbE 量级），不是 DeepSpeed 的写法，加并发无用。

正式训练的存盘点：`save_steps=500` 命中 500…4500 共 9 个，加 protected 2488 与 4976，**合计 11 次**。

| 项 | NFS | 本地盘 |
|---|---:|---:|
| 常驻峰值（`save_total_limit 3` + 2 protected，轮转瞬间取 6 份上界） | **330 GiB** | 同左 |
| 加上 `output_dir` 根目录的最终模型 8.27 GiB | **≈ 339 GiB** | 同左 |
| 全程写入总量 | **≈ 614 GiB** | 同左 |
| 占剩余空间 | 1.4% ✓ | 22% ✓ |

**容量 PASS（两个目标都够）**。

### 4.3 全程 1 epoch 时长外推（全部基于本次实测）

| 组成 | 计算 | NFS | 本地盘 |
|---|---|---:|---:|
| 训练计算 | 4976 × 5.078 s | 25,268 s | 25,268 s |
| eval | 11 × 53.1 s | 584 s | 584 s |
| checkpoint 写入 | 11 × 565 s / 11 × 333 s | 6,215 s | 3,663 s |
| 启动（shard sha256 + 模型加载 + DS init） | 实测 50–110 s | 120 s | 120 s |
| 收尾 `save_model` | 8.27 GiB | 84 s | 49 s |
| **合计** | | **32,271 s ≈ 9.0 h** | **29,684 s ≈ 8.2 h** |

**checkpoint 写入占 NFS 方案总时长的 19%。** 换到本地盘省约 **43 min**。

但真正的理由不是这 43 min，而是：**训练数据（images shard 23.7 GB）与 checkpoint 在同一个 NFS 挂载点上**。
实测在 checkpoint 写入期间，对同一目录的 `du -sb` / `ls` 会阻塞 **> 60 s**（本任务被卡过两次），
也就是说这 9.3 分钟里 dataloader 的 NFS 读也在抢同一条链路。上表的 25,268 s 计算时间是在**没有并发 checkpoint 写**的 Phase A 测的，
NFS 方案的真实总时长很可能**高于** 9.0 h。

**→ 主 agent 已裁定采用本地盘方案（D-J1），理由即上述 IO 耦合风险。** 见 §10 与 `NOTES.md`。

### 4.4 路径与 job.marker（spec §9.12 逐项）

| 项 | 值 |
|---|---|
| 训练期输出目录（**裁定后**） | `/home/bc/data/runs/q3vl_base_sft_20260804`（本地盘） |
| NFS durable 目录 | `/mnt/nfs/bc/runs/q3vl_base_sft_20260804`（训练后由 `SYNC_PROTECTED_TO_NFS.sh` 填充） |
| 训练日志 | `<output_dir>/train.log`（`launch_sft.sh` 先 `rm -f` 再重定向，绕开 zsh noclobber） |
| `job.marker` | `<output_dir>/job.marker`（PID + 完整启动命令 + 日志路径 + config + 时间 + host + CUDA_VISIBLE_DEVICES） |
| 启动期结构快照 | `<output_dir>/run_setup.json`（batch plan / special token id / embeddings / 冻结报告 / manifest / 架构 / 版本） |
| scratch 上限 | 数据经 `os.pread` 从 NFS 上的 indexed tar shard 随机读，**不解包成小文件**（spec §3.3）。唯一的本地写是 `output_dir`，上限 = 峰值常驻 339 GiB（启动脚本硬检查 ≥400 GiB 余量） |
| 数据只读挂载点 | `/mnt/nfs/bc/data/datasets/sft2seg-20260804`（records 550 MB + images 23.68 GB） |
| 启动脚本 | `../../s0_base_sft/config/START_PRODUCTION_TRAINING.sh`（**未执行**） |

## 5. 项 2：D-3 生成管线单验（单卡、非 ZeRO、base 模型）

出处：`logs/generation_pipeline_check.json`，脚本 `config/check_generation_pipeline.py`
（调用的是**发货代码本身** `Qwen3VLSFTTrainer.generation_diagnostics`，不是复刻实现）。

| 项 | 实测 |
|---|---|
| 样本 | `V_where` 前 3 条，`max_new_tokens=128`，greedy |
| 管线是否跑通 | **是**：prompt 切片（448/390/461 token，含图像占位）→ 图像张量 → `generate` → decode → `parse_two_segment` → 聚合，全程无异常 |
| 耗时 | 3 条共 23.25 s（6.60 / 6.66 / 6.64 s/条） |
| 峰值显存 | 8.46 GiB |
| 结构率 | 全部 **0.0**（`tag_completeness / order_accuracy / where_nonempty / color_nonempty / legacy_tag_leak / duplicate_*`） |

结构率为 0 是**预期**：base 模型从未见过这 4 个 special token，输出是普通对话文本
（例如 `"I can't directly modify the image to make the church and surroundings darker..."`）。
任务卡明确"生成质量不做要求"——这里要证的是管线可用，**已证**。
顺带说明解析器没有假阳性：`legacy_tag_leak_rate = 0`，即它没有把普通文本误判成旧七标签。

**正式配置确认关闭**：`sft_base.yaml` 的 `gen_diag_samples: 0`，脚本已断言
（`production_generation_diagnostics_disabled: true`）。

一条给未来开启者的实测依据（见 NOTES D-J3）：`load_model` 里设了 `model.config.use_cache = False`
（训练需要），本单验直接沿用，**没有 KV cache**。即便在最快的单卡非 ZeRO 路径上，128 token 也要 6.6 s/条；
按 `gen_diag_samples=32, max_new_tokens=512` 的默认配置外推，**单次 eval 的生成诊断就要 10 分钟以上**，
乘以 11 轮 eval ≈ 2 h，而 ZeRO-3 下还要逐层 gather 参数、只会更慢。**建议保持关闭、训练后离线补。**

## 6. 项 5：smoke 暴露的两个实现缺陷（均已修复）

两者的共同根因：`transformers` 在 ZeRO-3 生效时把 `from_pretrained` 包进 `deepspeed.zero.Init`，
所以 `setup_model` 拿到的模型**每个参数都已分区并释放**。单卡路径（先在 CPU 建模型再 `.to(device)`）
永远碰不到这个状态，因此 S0-TRAIN 的 68 个单测和单卡 preflight **全绿也不说明问题**。

两卡 probe 实测（`WORLD_SIZE=2`）：

```json
{"zero3_enabled": true, "emb_weight_shape": [0], "ds_shape": [151936, 2560],
 "ds_numel": 388956160, "ds_status": "ZeroParamStatus.NOT_AVAILABLE",
 "sum_numel_named_params": 0, "sum_ds_numel_named_params": 4437815808}
```

### F-1（崩溃型）`_mean_init_rows` 的 device 混用

`q3vl/train/tokens.py:97` 用 CPU generator 造噪声，直接乘以 CUDA 上的 `std`：

```python
noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32) * std * noise_scale
#       ^^^ CPU tensor                                                      ^^^ CUDA tensor
```

两个 rank 同时死在 `setup_model`：`RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`
（现场日志：`logs/phaseA_attempt1_FAILED_zero3_embedding.log`）。

**修复**：噪声仍用 **CPU generator**（故意的——这样抽出的值与 device / world_size 无关，
4 行的初值与单卡 preflight 验过的**逐位相同**），再显式 `.to(mean.device)`。
补单测 `test_mean_init_rows_on_cuda_weight`（有 GPU 时跑真 CUDA 张量）
与 `test_mean_init_rows_is_device_independent`。

**实证等价**：两卡实跑的 `min_pairwise_distance = 0.0016162680694833398`，
与 PREFLIGHT_MODEL.md 单卡报告的 `1.616e-3` **一致**。

### F-2（**静默型**，性质更严重）参数计数在 ZeRO-3 下恒为 0

`q3vl/train/freeze.py` 用 `param.numel()` 计数，ZeRO-3 下每个参数都是 0。后果两层：

1. **spec §9.2/§9.3 的审计表在正式训练里会全部打印 0**——而
   `assert_freeze_boundary` 的每一条断言都是「XX **必须为 0**」形式
   （`vision_blocks.trainable_params != 0` 之类），全 0 时**逐条通过**。
   也就是说：审计不但错，而且**永远不会报错**。
2. `prepare_embeddings` 把 `emb.weight.shape[0]` 读成 **0** ⇒ `need(151673) > rows_before(0)` ⇒
   触发 `resize_token_embeddings(151673, pad_to_multiple_of=64)` ⇒ 词表从 **151936 缩到 151680**，
   正好违反 S0-TRAIN 决策 D-4 要保护的「**永不缩表**」。

**修复**：
- 新增 `freeze.py::full_numel / full_shape`，优先读 `ds_numel` / `ds_shape`；
- `assert_freeze_boundary` **加正向非零断言**（`total_params <= 0` 直接报错，
  6 个关键子树的对应计数必须 > 0）——让"全 0 空转"这类空洞报告**从此不可能通过**；
- `prepare_embeddings` 改走 `full_shape`，行手术包进 `deepspeed.zero.GatheredParameters(modifier_rank=0)`；
- `assert_tied` 增加对象同一性判据（ZeRO-3 下两侧都是零长视图，`data_ptr()` 可能都是 0 而假通过）。

**修复后两卡实测与单卡口径逐字一致**（详见 PREFLIGHT_MODEL.md 的 ADDENDUM A2）：
总参 4,437,815,808｜可训练 4,131,573,248（93.0993%）｜冻结 306,242,560｜
`visual.blocks` 冻结 302,309,376｜`merger` 可训练 27,271,680｜3 个 deepstack 可训练 81,833,472｜
embedding 行数 151936（`resize_action: "kept"`）。

新增单测类 `TestZero3ParameterAccounting`（7 个）：`q3vl/train/tests/test_train_pipeline.py`。
**单测总数 68 → 75，全绿。**

### F-3（可观测性）checkpoint 的保存与删除**不进日志**

Phase B 全程没有一行 `Saving model checkpoint to ...`。原因：`TrainingArguments.log_level` 默认
`passive`，transformers 自己的 logger 停在 `WARNING`，而保存与删除都是 `logger.info`。

一个 9 小时、要写 11 个 checkpoint、要删掉其中几个的任务，如果**存盘与删除都不可见**：
D-20 第 3 步「tail 日志确认实质内容」在训练中段没有东西可看；而 S0-TRAIN 修的那个 bug
的性质恰恰就是"删除是静默的"——留一半静默等于没修完。

**修复**：`sft_base.yaml` 增加 `log_level: info`（**不在** `FROZEN_HPARAMS` 中，不构成规格偏离）。
Phase C 已用 `--log_level info` 实跑验证，日志中出现
`Saving model checkpoint to /mnt/nfs/.../checkpoint-6`。见 NOTES D-J4。

## 7. 项 4：resolved config 与启动脚本

| 交付物 | 路径 |
|---|---|
| resolved 生产 config | `../../s0_base_sft/config/sft_base_resolved.yaml` |
| 启动脚本（**未执行**） | `../../s0_base_sft/config/START_PRODUCTION_TRAINING.sh` |
| 实跑用的生产 config（启动器实际读的那个） | `q3vl/train/configs/sft_base.yaml` |
| DeepSpeed ZeRO-3 config | `q3vl/train/configs/ds_zero3.json` |

resolved config 里被钉死的关键量：
`seed 42 / data_seed 42`｜manifest digest `9278d721bb1e234c0a6e7ad94f8f8ae1eab10844602a26a1cfb0e20cab7ac840`｜
`N_effective 159,215`｜eval 集 `V_where`（896）｜`steps_per_epoch 4976`｜protected `2488 / 4976`｜
`4/卡 × GAS4 × 2 = 32`｜`lr 1e-5, wd 0, warmup 0.03, cosine, clip 1.0, bf16`｜`gen_diag_samples 0`。

`START_PRODUCTION_TRAINING.sh` 在真正启动前会硬性拒绝三种情况：GPU 上已有 compute 进程、
输出目录已存在 `checkpoint-*`（防止误用 smoke 状态）、terminal manifest 的 digest 或 `n_effective` 变了
（变了就必须重算步数，不能沿用本报告的 4976/2488）。

## 8. 复现命令

```bash
cd /home/bc/VeraRetouch
PYTHONPATH=. /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/train/tests -q     # 75 passed

# Phase A（生产 config 逐字，只改 output_dir；需手动 kill）
RUN_DIR=/mnt/nfs/bc/runs/<smoke>/phaseA MASTER_PORT=29517 \
  bash experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/joint/config/smoke_launch.sh

# Phase B（eval / 存盘 / 两条删除路径）
RUN_DIR=.../phaseB MASTER_PORT=29519 Q3VL_ALLOW_NONSPEC=1 \
  bash .../smoke_launch.sh --max_steps 8 --eval_steps 4 --save_steps 2 --save_total_limit 1

# Phase C（resume）
RUN_DIR=.../phaseC_resume MASTER_PORT=29521 Q3VL_ALLOW_NONSPEC=1 \
  bash .../smoke_launch.sh --max_steps 6 --eval_steps 6 --save_steps 6 --save_total_limit 1 \
       --log_level info --resume_from_checkpoint .../phaseB/checkpoint-4

# D-3 生成管线单验（单卡非 ZeRO）
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/bc/envs/q3vl_sft/bin/python \
  .../config/check_generation_pipeline.py --out .../logs/generation_pipeline_check.json --n 3
```

## 9. 遗留物与建议下一步

- **smoke 运行目录占用 183 GiB**：`/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/`
  （phaseA 40 KB、phaseB 119 GB、phaseC_resume 64 GB）。证据已全部提取到本交付目录的
  `logs/phaseB_collected.json` 与各段日志，**主 agent 确认后即可 `rm -rf` 整个 smoke 目录**。
  本任务不自行删除（"清空重跑前必先备份已有产物"）。
- **两项待决策**（`NOTES.md`）：D-J1（`output_dir` 用 NFS 还是本地盘 + 异步同步）、
  D-J3（生成类诊断是否在训练中开启，建议保持关闭 + 训练后离线补）。
  两者**都不阻塞**正式训练启动：默认值已经是可跑的。
- **建议下一步**：把本报告与 `PREFLIGHT_MODEL.md` 的 ADDENDUM 一并交独立实现审阅；
  blocker 清零后由主 agent 显式指令，用 `START_PRODUCTION_TRAINING.sh` 从**空的**
  `/mnt/nfs/bc/runs/q3vl_base_sft_20260804` 全新启动。

---

## 10. 主 agent 裁定与正式训练启动（2026-08-05）

独立实现审阅 blocker 清零后，主 agent 裁定：

| 决策 | 裁定 | 落实 |
|---|---|---|
| **D-J1** | **本地盘方案** —— 训练图片与 checkpoint 同 NFS 挂载点、存盘期间实测阻塞该挂载点 IO >60 s，耦合风险不可接受；本地 1.5 TB 对常驻 339 GiB 充足 | `output_dir = /home/bc/data/runs/q3vl_base_sft_20260804`；protected（2488 / 4976）与最终 checkpoint 训练后 rsync 到 `/mnt/nfs/bc/runs/q3vl_base_sft_20260804`，两侧路径与 sha256 记入 `s0_base_sft/checkpoint_sync_record.json` |
| **D-J3** | **保持关闭** | `gen_diag_samples: 0`；spec §8.3 结构率训练后在 protected checkpoint 上离线补 |

`sft_base.yaml` / `sft_base_resolved.yaml` / `START_PRODUCTION_TRAINING.sh` / `launch_sft.sh` 均已按裁定更新并在文件头注明。
启动脚本保留原三项硬拒绝，另加两项（D-J1 改路径后新增的失败面）：
**RUN_DIR 必须等于 config 的 `output_dir`**（`launch_sft.sh` 只用 RUN_DIR 决定日志位置，checkpoint 路径来自 config——不一致会把一次运行劈到两个文件系统上）、**本地卷余量 ≥ 400 GiB**。

smoke 目录 `/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/`（183 GiB）已删除；删除前把三段的 `job.marker` / `run_setup.json` / `trainer_state.json` 补进本交付目录 `logs/`。

### 正式训练已启动

| 项 | 值 |
|---|---|
| launcher PID | **3395099** |
| rank PID | **3395226**（GPU0）/ **3395227**（GPU1），均以 `ps -p` 实证存活 |
| 日志 | `/home/bc/data/runs/q3vl_base_sft_20260804/train.log` |
| `job.marker` | `/home/bc/data/runs/q3vl_base_sft_20260804/job.marker` |
| 启动时刻 | 2026-08-05 03:32:43（训练循环 03:32:52） |
| 预计完成 | 2026-08-05 **≈11:30**（±15 min） |
| 首批实测 | step 10 loss 3.421 / grad_norm 33.15；step 20 loss 3.136 / grad_norm 13.27；**4.90 s/step** |
| 复现性 | 与 Phase A（同 seed 42）的 3.4211 / 3.1362 逐位对齐 |

**训练监控自此移交主 agent。**
