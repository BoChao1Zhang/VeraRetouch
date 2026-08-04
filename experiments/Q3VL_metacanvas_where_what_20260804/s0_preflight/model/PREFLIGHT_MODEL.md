# S0-TRAIN 模型侧 preflight 报告

> 规格：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md`（§9 项 1–4 + 项 10/11）
> 日期：2026-08-05　｜　机器可读输出：`preflight_model.json`
> **状态：正式训练未启动**（任务卡纪律）。GPU 仅用于加载与 smoke。

## 结论一句话

Arm B 的四条结构性前提——权重完整、冻结边界正确、4 个 special token 单 token 且可 reload、
真实前反向数值有限——**全部实测通过**；两卡 ZeRO-3 的首选 batch 组合 `4 x GAS4` 在**最坏输入**
（4 张 512x2048 图 + 2044 token 序列）下峰值 42.2 GiB / 95.1 GiB，无需退到 `2 x GAS8`。

## 总览

| 规格项 | 内容 | 结果 |
|---|---|---|
| §9.1 | 权重 shard 完整性 + Transformers 版本 | **PASS** |
| §9.2 | frozen/trainable 逐名清单、参数量与比例 | **PASS** |
| §9.3 | Vision 无梯度、4 个 merger + Language 有梯度（实证） | **PASS**（含 gradient checkpointing 开/关两档） |
| §9.4 | 4 个 special token 的 ID / 单 token / reload 一致性 | **PASS** |
| §9.10 | 单 batch 前后显存、吞吐、loss/梯度有限性 | **PASS** |
| §9.11 | global batch 32 的最终 micro-batch/GAS 组合 | **PASS**（推荐 4 x GAS4） |
| 附加 | 数据管线 / loss mask 结构断言（mock shard） | **PASS** |
| §9.5–9.9, §9.12 | 七段→两段计数、split 交集、shard 随机读、图像/长度分布、NFS 路径 | 归属 S0-DATA，联合 preflight 出 |

**overall: PASS**

## 环境（本地实测，非文档抄写）

| 项 | 值 |
|---|---|
| Python 环境 | `/home/bc/envs/q3vl_sft`（`venv --system-site-packages`，叠在 conda env `llm_factory` 上） |
| Python / torch | 3.12.12 / 2.10.0+cu128（CUDA 12.8） |
| transformers | **4.57.1**（Qwen3-VL 支持自 4.57.0 起；base conda env 的 4.36.0 不可用） |
| deepspeed | **0.18.2**（本任务新装，见 NOTES 决策 D-1） |
| flash-attn / accelerate | 2.8.3 / 1.11.0 |
| GPU | 2 x NVIDIA H100，95.1 GiB/卡，驱动 570.195.03 |
| git commit | `bc888fe147af76f2f8ef0299c85de856b9cd4adf` |

模型 config 实测与 spec §12 完全一致：vision `depth=24, hidden=1024, patch=16, merge=2,
deepstack_visual_indexes=[5,11,17], out_hidden=2560`；text `36 层, hidden 2560,
vocab 151936, tie_word_embeddings=true`。

---

## 项 1：权重 shard 完整性 — PASS

上游 sha256 取自 HF 官方 API（`https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true`，
revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`，2026-08-04 当场拉取并写入
`q3vl/train/modeling.py::UPSTREAM_SHARD_SHA256`）。

| shard | 落盘字节 | 期望字节 | sha256 |
|---|---:|---:|---|
| `model-00001-of-00002.safetensors` | 4,967,229,296 | 4,967,229,296 ✓ | `30a01a05…3339a9` ✓ |
| `model-00002-of-00002.safetensors` | 3,908,490,048 | 3,908,490,048 ✓ | `046296a2…d002a6` ✓ |

- index 中 **713 个 tensor 全部可从对应 shard 读出**（safetensors header 扫描），无缺失；
- **无 `.aria2` / `.part` / `.incomplete` 残留**（spec §2.1 特别点名的风险，`.hfd/aria2c_urls.txt` 为 0 字节的收尾文件，不是下载状态文件）；
- index `total_size` ≤ 磁盘字节数，一致。

## 项 2：frozen / trainable 清单 — PASS

| 口径 | 参数量 | 占比 |
|---|---:|---:|
| 总计 | 4,437,815,808 | 100% |
| **可训练** | **4,131,573,248** | **93.0993%** |
| **冻结** | **306,242,560** | **6.9007%** |

| 子树 | tensors | 参数量 | 可训练 | 冻结 |
|---|---:|---:|---:|---:|
| `model.visual.blocks.*`（24 层） | 288 | 302,309,376 | 0 | **302,309,376** |
| `model.visual.patch_embed.*` | 2 | 1,573,888 | 0 | **1,573,888** |
| `model.visual.pos_embed.*` | 1 | 2,359,296 | 0 | **2,359,296** |
| `model.visual.merger.*`（主 merger） | 6 | 27,271,680 | **27,271,680** | 0 |
| `model.visual.deepstack_merger_list.*`（3 个） | 18 | 81,833,472 | **81,833,472** | 0 |
| `model.language_model.embed_tokens` | 1 | 388,956,160 | **388,956,160** | 0 |
| `model.language_model.layers.*`（36 层） | 396 | 3,633,509,376 | **3,633,509,376** | 0 |
| `model.language_model.norm` | 1 | 2,560 | **2,560** | 0 |
| `lm_head` | 0 | 0 | — | — |

- `lm_head.weight` 与 `embed_tokens.weight` **tied**（`_tied_weights_keys=['lm_head.weight']`，
  `named_parameters()` 去重后不单独出现；data_ptr 相同已断言）；
- spec §2.2 的警告已落实：**没有**使用 `model.visual.requires_grad_(False)`，
  规则按参数名逐条判定（`q3vl/train/freeze.py::is_trainable_param`），
  merger / deepstack merger 在 `model.visual` 子树内被显式豁免；
- 全模型无任何 `lora` / `adapter` 命名参数（非 PEFT 断言）。

`pos_embed` 归入冻结侧：它属于 vision tower 位置编码而非任何 merger，对应官方
`tune_mm_vision=False` 的语义。见 NOTES 决策 D-2。

## 项 3：梯度实证 — PASS（两档）

单卡真实前反向，mock batch（2 样本，102 个受监督 token），loss 6.2570。
**gradient checkpointing 关 / 开各跑一次**——冻结子树被 checkpoint 包裹时"梯度静默消失"
是这类配置最典型的坑，因此不作假设而是实测。

| 参数组 | 期望 | n_with_grad / n | grad L2（GC off） | grad L2（GC on） |
|---|---|---:|---:|---:|
| `visual.blocks` | 无梯度 | **0 / 288** | 0 | 0 |
| `visual.patch_embed` | 无梯度 | **0 / 2** | 0 | 0 |
| `visual.pos_embed` | 无梯度 | **0 / 1** | 0 | 0 |
| `visual.merger` | 有梯度 | **6 / 6** | 3.0848 | 3.0822 |
| `visual.deepstack_merger_list` | 有梯度 | **18 / 18** | 2.9405 | 2.9402 |
| `language_model.embed_tokens` | 有梯度 | **1 / 1** | 51.983 | 51.813 |
| `language_model.layers` | 有梯度 | **396 / 396** | 109.522 | 109.480 |
| `language_model.norm` | 有梯度 | **1 / 1** | 0.1908 | 0.1908 |

两档 loss 完全相同（6.256959915161133），梯度范数逐组吻合到小数点后 2–3 位（差异为
重算的浮点非确定性），**证明 gradient checkpointing 未切断 merger 梯度通路**。
所有梯度有限，无 NaN/Inf。

## 项 4：special token — PASS

| token | ID | 单 token | reload 后 |
|---|---:|---|---|
| `<where>` | **151669** | ✓ | 151669 ✓ |
| `</where>` | **151670** | ✓ | 151670 ✓ |
| `<color>` | **151671** | ✓ | 151671 ✓ |
| `</color>` | **151672** | ✓ | 151672 ✓ |

- 注册前 4 个字符串各被切成 3 个 token（如 `<where>` → `[27, 2870, 29]`），注册后各为 1 个；
- `tokenizer` 长度 151669 → **151673**；embedding 矩阵 **151936 行保持不变**（**不缩表**，见 D-4）；
- tied-weight 契约保持（input/output embedding 共享存储，已断言）；
- `save_pretrained` → `from_pretrained` 往返后 ID 与单 token 性重验通过；
  落盘含 `tokenizer.json / special_tokens_map.json / added_tokens.json / chat_template.jinja /
  preprocessor_config.json`；
- **新增行去简并**：checkpoint 里 151669+ 的 padding 行近乎全同（L2 均为 0.358–0.361，
  部分逐位相同），若直接使用，4 个 token 在第 0 步 logits 无法区分。已按
  HF `mean_resizing` 语义重初始化为「真实词表均值 + 逐维 std x 1e-3 噪声」，
  重初始化后最小两两距离 **1.616e-3 > 0**。

## 项 10：单卡单 batch smoke — PASS

模型加载 9.4 s。3 步真实 AdamW（lr 1e-5, wd 0, betas 0.9/0.999, eps 1e-8, clip 1.0），
gradient checkpointing 开，bf16 + flash_attention_2。

| 批次 | seq_len | 视觉 patch | peak allocated | peak reserved | 稳态 step | loss 轨迹 |
|---|---:|---:|---:|---:|---:|---|
| mock 混合尺寸 mb=4 | 1097 | 7,936 | **40.62 GiB** | 49.61 GiB | 0.819 s | 6.254 → 4.561 → 3.612 |
| **最坏输入** mb=4 | 2044 | 16,384 | **42.18 GiB** | 58.29 GiB | 1.707 s | 下降 |
| **最坏输入** mb=2 | 2044 | 8,192 | **40.53 GiB** | 47.80 GiB | 0.869 s | 下降 |

- loss 与梯度范数全程有限、为正、单调下降（3 步内 6.25 → 3.61，梯度范数 117.5 → 36.75）；
- 「最坏输入」= 4 张 512x2048（长边上限、宽高比恰 4:1、每图 1024 个视觉 token）
  且文本 target 撑到 2044 token，即本规格下合法样本的显存上界。

**显存口径说明（重要，不可直接外推）**：本 smoke 为单卡无 ZeRO-3，
`torch.optim.AdamW` 的矩与参数同 dtype，故常驻状态是 bf16 参数 8.3 GiB + bf16 梯度 8.3 GiB
+ 2 份 bf16 矩 16.5 GiB ≈ 33 GiB，其余为激活。正式 ZeRO-3 改为 fp32 master + fp32 矩但
**在 2 卡间切分**：`4.13e9 x 4 x 3 / 2 = 24.8 GiB`/卡，加上瞬时 bf16 gather 与
**此处实测的同一份激活开销**。可迁移的是激活数字，常驻状态数字不可迁移。

## 项 11：batch 组合 — PASS

| micro | GAS | world | 有效 global batch | 最坏输入是否通过 |
|---:|---:|---:|---:|---|
| **4** | **4** | 2 | **32** ✓ | ✓ 42.18 GiB |
| 2 | 8 | 2 | 32 ✓ | ✓ 40.53 GiB |

**推荐 `4 x GAS4`**（spec §7.2 首选）。最坏输入下峰值 42.2 GiB，距 95.1 GiB 有充裕余量，
即使按 ZeRO-3 常驻状态换算也不触顶，**无需启用 `2 x GAS8` 备选**。
`q3vl/train/args.py::validate_frozen_hyperparameters` 在启动时强制断言有效 global batch = 32
且组合只能取这两种之一。

## 附加：数据管线与 loss mask 结构断言 — PASS

在 mock indexed-tar-shard 上跑通全链路（真实 tokenizer + 真实 image processor）：

| 断言 | 结果 |
|---|---|
| 分段拼接 token 化 == 整串 token 化（BPE 未跨界合并） | ✓（329 vs 329，无分歧点） |
| image placeholder token **零个**被监督 | ✓ |
| 受监督位置 label == input_ids | ✓ |
| prompt / template / system 全部 `IGNORE_INDEX` | ✓ |
| 每行 `<where>` 段严格早于 `<color>` 段 | ✓ |
| 4 个 special token 每行各出现 1 次且均被监督 | ✓ |
| padding 位 label 全为 `IGNORE_INDEX` | ✓ |
| 视觉 token 数 == `(H/32) x (W/32)`，与 `image_grid_thw` 一致 | ✓ |
| 序列长度 ≤ 2048，超长**抛错不截断** | ✓ |

图像几何实测（spec §5）：`512x512 → 16x16 = 256` token；`512x1024 → 512`；
`512x2048 → 1024`；`640x512 → 320`。短边恒为 512、双边 32 对齐、比例误差 < 3.2%、
宽高比 > 4:1 直接进 rejection。

### 端到端训练闭环（mock 数据，单卡，24 train / 8 eval）

已用**真实入口** `python -m q3vl.train.train_sft --config ...` 跑完 1 个 epoch：

- `eval_loss = 4.554167` 与自算的 `eval_seg_assistant_loss = 4.554167` **逐位吻合**，
  实证主选择指标确为 **assistant-only**（spec §8.3）；
- Where/Color/EOS 分段 loss 与 token accuracy 正常记录（诊断用，不入总 loss）；
- 0.5 / 1.0 epoch **protected checkpoint 均落盘**，各带 tokenizer、processor、
  special-token map、chat template、training state 与 `protected_checkpoint.json`
  （内含 manifest digest、`N_effective`、special token ID、冻结报告）。

### 实现期发现并修复的一个真实缺陷（供实现审阅重点复核）

transformers 4.57.1 有**两条**删 checkpoint 的路径，只有一条经过 `_rotate_checkpoints`：
`_inner_training_loop` 收尾处还有一段

```python
if should_save and best_model_checkpoint is not None and save_total_limit == 1:
    for checkpoint in checkpoints_sorted:
        if not samefile(checkpoint, best_model_checkpoint): rmtree(checkpoint)
```

它**不调用** `_rotate_checkpoints`。首版只覆写了 `_rotate_checkpoints`，
mock 跑通后实测 **0.5 epoch 的 `checkpoint-2` 被静默删除**（日志无痕，因为该 INFO 走
transformers 自己的 logger）。修复改为覆写两条路径共同的取数口 `_sorted_checkpoints`
（在基类做完 best-checkpoint 索引记账之后再过滤），并补了覆盖**两条路径**的单测。
修复后同一配置（`save_total_limit=1` + `metric_for_best_model=eval_loss`）下
`checkpoint-2` 与 `checkpoint-3` 均存活。

> 本仓库当前配方用 `save_total_limit: 3`，不会触发 `== 1` 分支，但该陷阱是静默的，
> 且里程碑 checkpoint 是后续 Stage-Where / Stage-What 的共同基模，已按红线级别处理。

## 复现命令

```bash
cd /home/bc/VeraRetouch
PYTHONPATH=. /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/train/tests -q      # 68 passed

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/bc/envs/q3vl_sft/bin/python \
  -m q3vl.train.preflight_model \
  --out-dir experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/model \
  --device cuda:0 --micro-batch 4 --smoke-steps 3
```

正式训练启动脚本 `q3vl/train/scripts/launch_sft.sh` 已就绪（含 D-20 四步：先 `rm -f` 日志、
`ps -p $PID` 判活、`tail` 验实质输出、最后写 `job.marker`），**本任务未执行**。

## 与 S0-DATA 的接口状态

并行的 S0-DATA 任务在本任务收尾时已落地 `q3vl/data/`。已逐字段核对其
`pipeline.py::stage_manifest` 的**实际产物**（不是猜测），**schema 完全兼容**，
并补了 4 个用其真实行形状构造的 interop 单测：

- index 行 = 嵌套 `members`（`record`/`image`），成员字段
  `shard`(绝对路径)/`member`/`offset`/`length`/`size`/`sha256`（裸 hex）→ 消费侧全部命中；
- 记录字段 `where` / `color`，目标串 `<where>W</where><color>C</color>`
  与 `build_target_text` **逐字符相同**；
- manifest `counts.<split>.n_effective` + 顶层 `n_effective` + `digest` + `schema_version` 全部可读；
- 路径已写入 `configs/sft_base.yaml`：`splits/train.index.jsonl`、
  `manifest/terminal_manifest.json`。

修正一处口径：生产方每个 split 一个 index 文件、行内 `split` 是 `V_where` 这类名字而非
`"eval"`，入口原先按 `split="eval"` 过滤会把 eval 集滤空，已改为默认不过滤。

## 阻塞正式训练的前置条件

1. S0-DATA 完成数据落盘（截至本报告 `/mnt/nfs/bc/data/datasets/sft2seg-20260804/` 尚未产出实体）。
   入口在缺 `terminal_manifest` 时直接退出，**不会**用 spec 的 2645/5290 估算值兜底。
2. 联合 preflight 补齐 §9 项 5–9、12（七段→两段计数与抽样对照、三集合无交集、
   shard 随机读/checksum/resume、图像与序列长度分布、NFS 路径与 scratch 上限）。
3. NOTES.md 中 **D-3**（ZeRO-3 下生成类诊断开不开）、**D-5**（`eval_loss` 选择指标与
   CLAUDE.md 红线冲突）、**D-7**（Base SFT 的 eval 集用 `V_where` 还是别的）三项待主 agent 决策。

---

# ADDENDUM（2026-08-05，由 S0-JOINT 联合 preflight 追加，原文一字未改）

**数据来源**：全部取自 S0-JOINT 的两卡 ZeRO-3 真实数据 smoke，非本报告作者重新推算。
逐条出处：

| 证据 | 文件 |
|---|---|
| 两卡启动期结构审计（freeze / embeddings / batch plan / manifest） | `../joint/config/phaseA_run_setup.json`（= 运行目录里 `run_setup.json` 的原件拷贝） |
| Phase A 完整日志（生产 config 逐字，仅改 `output_dir`，74 步后主动 kill） | `../joint/logs/phaseA_2gpu_zero3_smoke.log` |
| Phase B 完整日志（eval / checkpoint / 保护性删除） | `../joint/logs/phaseB_2gpu_zero3_ckpt_eval.log` |
| 首次两卡启动的失败日志（本 addendum 记录的缺陷现场） | `../joint/logs/phaseA_attempt1_FAILED_zero3_embedding.log` |
| 结论与全部实测数字 | `../joint/PREFLIGHT_JOINT.md` |

## A1. 本报告的项 2 / 项 3 结论在两卡 ZeRO-3 下**一度不成立**

本报告的项 2（frozen/trainable 清单）与项 3（梯度实证）都是在**单卡、无 DeepSpeed**
路径下取得的。联合 preflight 首次用两卡启动时发现：

`transformers` 在 ZeRO-3 生效时会把 `from_pretrained` 包进 `deepspeed.zero.Init`，
于是模型返回时**每个参数都已分区并释放**。两卡实测（probe，`WORLD_SIZE=2`）：

```json
{"zero3_enabled": true, "emb_weight_shape": [0], "ds_shape": [151936, 2560],
 "ds_numel": 388956160, "ds_status": "ZeroParamStatus.NOT_AVAILABLE",
 "sum_numel_named_params": 0, "sum_ds_numel_named_params": 4437815808}
```

后果两条，第二条是静默的：

- **F-1（崩溃）** `tokens.py::_mean_init_rows` 用 CPU generator 造噪声乘以 CUDA 上的 `std`
  → `RuntimeError: Expected all tensors to be on the same device`，两个 rank 同时死在
  `setup_model`。单卡先在 CPU 建模型再 `.to(device)`，永远碰不到。
- **F-2（静默）** `freeze.py` 用 `param.numel()` 计数，ZeRO-3 下恒为 0。
  于是本报告项 2 的那张表在两卡下会**全部打印 0**，而 `assert_freeze_boundary` 的每一条
  「XX 必须为 0」断言**照样通过**——审计变成空转。同理 `prepare_embeddings` 会把
  `emb.weight.shape[0]` 读成 0，进而触发本不该发生的
  `resize_token_embeddings(151673, pad_to_multiple_of=64)`，把词表从 **151936 缩到 151680**，
  正好违反本报告 D-4 所要保护的「永不缩表」。

两者均已修复（`freeze.py::full_numel/full_shape` + 正向非零断言；`tokens.py::gathered`
用 `deepspeed.zero.GatheredParameters` 做行手术、噪声显式搬到 weight 所在 device），
并补了 7 个回归单测（`TestZero3ParameterAccounting`）。单测总数 68 → **75 passed**。

## A2. 修复后，两卡 ZeRO-3 的实测值与本报告单卡口径**逐字一致**

出处：`../joint/config/phaseA_run_setup.json`。

| 口径 | 本报告（单卡） | 两卡 ZeRO-3 实测 | 一致 |
|---|---:|---:|:--:|
| 总参数 | 4,437,815,808 | 4,437,815,808 | ✓ |
| 可训练 | 4,131,573,248（93.0993%） | 4,131,573,248（93.0993%） | ✓ |
| 冻结 | 306,242,560（6.9007%） | 306,242,560（6.9007%） | ✓ |
| `visual.blocks` 冻结 | 302,309,376 | 302,309,376 | ✓ |
| `visual.patch_embed` 冻结 | 1,573,888 | 1,573,888 | ✓ |
| `visual.pos_embed` 冻结 | 2,359,296 | 2,359,296 | ✓ |
| `visual.merger` 可训练 | 27,271,680 | 27,271,680 | ✓ |
| `deepstack_merger_list` 可训练 | 81,833,472 | 81,833,472 | ✓ |
| `language_model.*` 可训练 | 4,022,468,096 | 4,022,468,096 | ✓ |
| special token id | 151669–151672 | 151669–151672 | ✓ |
| embedding 行数 | 151936（不缩表） | 151936（`resize_action: "kept"`） | ✓ |
| 4 行重初始化最小两两距离 | 1.616e-3 | **0.0016162680694833398** | ✓ 逐位相同 |
| tied 契约 | 保持 | `tied_ok: true` | ✓ |

## A3. 项 10 / 项 11 的两卡实测（本报告的单卡数字**不可外推**，这里是可外推的那份）

本报告项 10 已自行声明「常驻状态数字不可迁移」。两卡 ZeRO-3 + 真实数据的实测替换值：

| 指标 | 两卡 ZeRO-3 实测 |
|---|---|
| 峰值显存 | GPU0 **52,973 MiB** / GPU1 **52,355 MiB**（卡容量 97,871 MiB，占用 54%） |
| 稳态吞吐 | **5.078 s/optimizer step**（步 10→74，global batch 32） |
| GPU 利用率 | 83% / 81% |
| batch 组合 | `4 x GAS4 x world 2 = 32` 实跑通过，**无需** `2 x GAS8` |
| 互联 | `nvidia-smi topo -m` = **PXB（PCIe，无 NVLink）** |
| loss（步 10→70） | 3.4211 → 3.1362 → 2.6362 → 2.0339 → 1.6522 → 1.4589 → 1.3721，全程有限 |
| grad_norm（同上） | 33.22 → 13.27 → 7.67 → 4.63 → 3.66 → 3.27 → 3.24，全程有限 |

**结论**：本报告项 11 推荐的 `4 x GAS4` 在真实数据两卡上成立，结论不变；
项 2/3/4 的**数值结论**成立，但其**单卡取证方式**不足以覆盖 ZeRO-3，已由 A1 的修复补齐。

## A4. 本报告末节那个 checkpoint 静默删除 bug 的两卡实证

本报告修复的 `_sorted_checkpoints` 双路径保护，此前只有单测。S0-JOINT 在
**两卡 ZeRO-3 + `save_total_limit=1` + `metric_for_best_model=eval_loss`** 下实跑了一次
（Phase B，8 步，存盘于 2/4/6/8，protected = {4, 8}）：

- 训练结束后**幸存的恰好是 checkpoint-4 与 checkpoint-8**（两个 protected），
  checkpoint-2 与 checkpoint-6 被 `trainer.py:2840` 那条路径删除；
- `trainer_state.best_model_checkpoint = .../checkpoint-8`，`best_metric = 2.1702332496643066`；
- 两个 protected checkpoint 各自带 `protected_checkpoint.json`（含 manifest digest、
  `N_effective`、special token id、冻结报告）。

即：该 bug 的修复在真实分布式路径上**被正面验证**，不再只有单测背书。
