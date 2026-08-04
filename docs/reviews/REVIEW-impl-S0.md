# REVIEW-impl-S0：Qwen3-VL Base SFT 实现审阅

> 审阅人：独立实现审阅 subagent（与实现者无共享上下文）
> 日期：2026-08-05
> 规格权威：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md`（全文 414 行，已全读）
> ＋ `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §2
> 审阅对象：`q3vl/`（data/ + train/ + tests/）、
> `experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/`、
> `/mnt/nfs/bc/data/datasets/sft2seg-20260804/`、`/home/bc/data/models/Qwen3-VL-4B-Instruct`
> 本次审阅**未修改任何代码或数据**，只跑只读检查、现有单测与独立复算脚本。

---

## 判决

**BLOCKER 数量 = 4；不准许进入正式训练。**

> **本节是初审判决，已被文末更新。请直接读文末的《最终判决（2026-08-05 02:12 更新）》。**
>
> 时间线：本报告主体针对 2026-08-05 01:57 之前的代码快照。01:53 一次并行的 2 卡 ZeRO-3
> 联合 smoke 在 `tokens.py:97` 崩溃，印证了 B-1；实现侧于 01:58-02:00 热修了
> `tokens.py` / `freeze.py`；02:01:53 第二次提交成功起跑；02:12 完成对热修 diff 的聚焦补审
> （**PASS，无新增 blocker**）。
>
> **更新后的状态：B-1 / B-2 / B-3 已关闭；剩余阻塞仅 B-4（§9 项 12 运行面交付，
> 由联合 preflight subagent 正在产出）与 PREFLIGHT_MODEL 的 ZeRO-3 证据回写。
> 代码侧不再有阻塞项。**

四个 blocker 集中在**同一个从未被执行过的代码路径**：`docs/…SPEC…§7.2` 冻结的
`world_size=2 + DeepSpeed ZeRO-3` 启动路径。preflight 的全部模型侧证据（§9 项 2/3/4/10/11）
都是在**单卡、无 DeepSpeed**下采集的；而 ZeRO-3 的 `zero.Init()` 会在 `from_pretrained`
阶段就把参数切片，使 `param.shape`/`param.numel()` 变成 0，从而让 `q3vl/train/tokens.py`
与 `q3vl/train/freeze.py` 走上另一条分支。

本次审阅期间（2026-08-05 01:53）有一个并行的联合 preflight 用**真实入口 + 真实 config**
起了 2 卡 ZeRO-3，**在 74 秒后崩溃**，崩溃点与本审阅独立推导出的位置逐字一致。该日志是
B-1 的直接证据（不属于被审交付物，但它执行的正是被审代码）。

数据侧（`q3vl/data/` + `sft2seg-20260804`）**未发现 blocker**：我用不导入 `q3vl.*` 的独立脚本
把七段→两段转换、四集合隔离、LUT reserve、图像契约、序列长度、shard 定位与 checksum
全部重算了一遍，**逐项与交付报告吻合**（证据见下文 §"独立复核记录"）。

---

## 一、BLOCKER

### B-1（hard，已实证）ZeRO-3 下训练入口直接崩溃：`q3vl/train/tokens.py:97`

**规格条文**：SPEC §7.2「`distributed: DeepSpeed ZeRO-3`、`world_size: 2`」为冻结项；
SPEC §9「在实现完成后、正式训练启动前，必须先产生一份 preflight 报告 …… 任一项失败时
不得把 smoke 结果升级为正式训练」。

**现场证据**：`/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/phaseA/train.log:24-62`

```
[rank0]:   File "/home/bc/VeraRetouch/q3vl/train/tokens.py", line 148, in prepare_embeddings
[rank0]:     info["reinit"] = _mean_init_rows(weight.data, target_rows, ref_upper, gen)
[rank0]:   File "/home/bc/VeraRetouch/q3vl/train/tokens.py", line 97, in _mean_init_rows
[rank0]:     noise = torch.randn(mean.shape, generator=generator, dtype=torch.float32) * std * noise_scale
[rank0]: RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!
```

**根因**：`q3vl/train/tokens.py:139` 建的是 CPU generator
（`torch.Generator(device="cpu").manual_seed(seed)`），`tokens.py:97` 的
`torch.randn(..., generator=generator)` 因而产出 **CPU** 张量，而 `std` 在 ZeRO-3 下位于
**CUDA**（`zero.Init` 的 `remote_device` 默认就是 GPU）。

**为什么 preflight 抓不到**：`q3vl/train/modeling.py:164` 的 `from_pretrained` 不传
`device_map`；单卡 preflight（`preflight_model.py`，无 DeepSpeed）下模型先落 **CPU**，
`std` 也在 CPU，两者同设备，恒通过。我已本机实证：只要 `TrainingArguments(deepspeed=…)`
被构造，`transformers.integrations.deepspeed.is_deepspeed_zero3_enabled()` 立刻为 True
（本机实测），而 `train_sft.py:137` 的 `parse_args()` 早于 `train_sft.py:165` 的 `setup_model()`，
因此 `modeling_utils.py:2300-2308` 必然进入 `zero.Init()` 分支。

**修复方向**（不由审阅指定实现）：让 noise 与参考张量同设备
（`torch.randn(..., device=weight.device, generator=<同设备 generator>)`，或先 `.cpu()` 再写回），
并**必须**用 2 卡 ZeRO-3 重跑一次入口证明它能起来。

---

### B-2（hard）ZeRO-3 下 `prepare_embeddings` 读到切片形状，会把词表**缩表**到 151680，与冻结决策 D-4 和 PREFLIGHT_MODEL.md 项 4 的结论相反

**规格条文**：SPEC §4.3「tokenizer 扩词表后同步 resize model embeddings；Qwen3-VL-4B 的
输入 embedding 与输出头共享时，保持 tied-weight 契约」；SPEC §9 项 4
「四个新 special token 的 ID、单 token 编码与 reload 一致性」。

**代码位置**：`q3vl/train/tokens.py:111`
```python
rows_before = int(emb.weight.shape[0])
```
`q3vl/train/tokens.py:119-125`
```python
if need > rows_before:
    model.resize_token_embeddings(need, pad_to_multiple_of=64, mean_resizing=True)
    info["resize_action"] = "grown"
else:
    info["resize_action"] = "kept"
```
`q3vl/train/tokens.py:14-18` 文档承诺：「Resize policy: **never shrink**」。

**实证**（本机跑，`zero.Init(config_dict_or_path=q3vl/train/configs/ds_zero3.json)`）：

```
param.shape under zero.Init  : (0,)
param.ds_shape               : (151936, 64)
shape[0] as read by prepare_embeddings: 0
```

于是 `need(151673) > rows_before(0)` 恒成立 → **必然**进入 `grown` 分支 →
`resize_token_embeddings(151673, pad_to_multiple_of=64)` 的实际目标是
`ceil(151673/64)*64 = 151680 < 151936`，即**缩表 256 行**，同时把
`config.vocab_size` 改成 151680。

**后果**：
1. `PREFLIGHT_MODEL.md` 项 4 写的「embedding 矩阵 **151936 行保持不变（不缩表）**」与
   `preflight_model.json.embeddings.resize_action = "kept"`、`embedding_rows_before = 151936`
   在正式配置下**均不成立**——交付的 §9 项 4 证据取自一条正式运行不会走的分支；
2. `NOTES.md`（model）决策 D-4「永不缩表」在正式配置下被自身代码推翻；
3. `info["embedding_rows_before"]` 会被写成 0 并落进 `run_setup.json` 与每个 checkpoint 的
   `protected_checkpoint.json`，成为后续 Stage-Where/What 的错误审计基线。

功能上 151672 < 151680，四个 token 仍可用，**但结论与证据不一致本身就是 blocker**：
本仓库的纪律是「任何结论必须先出示对应配置下的断言通过记录」。

**修复方向**：对 `hasattr(param, "ds_id")` 的参数改读 `ds_shape` / 用
`deepspeed.zero.GatheredParameters` 取真实形状后再判分支；修完必须在 2 卡 ZeRO-3 下
重新出示 §9 项 4 的四行证据（ID / 单 token / tied / reload）。

---

### B-3（hard）ZeRO-3 下 `FreezeReport` 的参数量全部塌成 0，SPEC §9 项 2/3 的数量断言在正式运行中**静默失效**

**规格条文**：SPEC §9 项 2「frozen/trainable 参数名、**参数量和比例**」；项 3
「证明 Vision blocks 无梯度、4 个 merger 与 Language 有梯度」。

**代码位置**：`q3vl/train/freeze.py:99`
```python
n = param.numel()
```
`q3vl/train/freeze.py:150-161`（结构断言全部基于上面那个 `n`）
```python
if st["vision_blocks"]["trainable_params"] != 0: raise ...
if st["vision_merger"]["frozen_params"] != 0:   raise ...
```

**实证**（同一 `zero.Init` 上下文）：

```
sum(p.numel()) under ZeRO-3 partitioning : 0
ds_numel of each param                   : [1215488, 64, 8]
```

即在正式配置下 `total_params = trainable_params = frozen_params = 0`，
`trainable_ratio = 0/1 = 0`。上面两条断言变成 `0 != 0` → 永远通过，**再也抓不住任何真实错误**；
`format_freeze_table` 打印全 0；写进 `run_setup.json` 与 checkpoint sidecar 的 `freeze_report`
同样全 0。

**注意**：冻结**行为本身是对的**——`freeze.py:30-35` 的 `is_trainable_param` 按参数名判定，
`freeze.py:141-146` 的 `requires_grad` 逐名校验在 ZeRO-3 下仍然有效，
`model.visual.requires_grad_(False)` 这条 SPEC §2.2 明令禁止的写法确实没有出现。
坏掉的是**审计与交付证据**，不是冻结边界。

**修复方向**：`numel()` 改为 `getattr(param, "ds_numel", param.numel())`，
并在 2 卡 ZeRO-3 下重新出一份非零的 frozen/trainable 清单。

---

### B-4（交付缺失）SPEC §9 项 12 在两份 preflight 报告里都不存在

**规格条文**：SPEC §9「必须先产生一份 preflight 报告，至少包含：…… 12. NFS durable
输出路径、scratch 容量上限、日志与 `job.marker` 路径。任一项失败时不得把 smoke 结果
升级为正式训练。」

**现状**：
- `preflight_model.json.deferred_items` 把 `"12": "owner=S0-DATA / joint preflight"`；
- `PREFLIGHT_MODEL.md:24` 同样标注「归属 S0-DATA，联合 preflight 出」；
- `PREFLIGHT_DATA.md` 的结论一览（§0）只覆盖 §9 项 **5/6/7/8/9** 与 METACANVAS §2.2，
  **全文没有项 12 的任何一节**（`rg "job.marker|scratch|item 12"` 在该文件命中 0 次）。

即项 12 被两边互相推给对方，最终无人交付。素材其实都在手边
（`launch_sft.sh` 已定义 `RUN_DIR` / `LOG` / `job.marker`；NOTES 记了 NFS 余量 23 TB；
我复核 `df -h /mnt/nfs` = 23T avail），补一节即可关闭，但在补齐前**规格意义上 preflight 不完整**。

补写时请一并覆盖：`save_total_limit=3` + 2 个 protected checkpoint ⇒ 磁盘上最多同时存在
**5 份** ZeRO-3 checkpoint（含 fp32 优化器态，单份约 55-60 GB，合计约 290 GB）。

---

## 二、NIT（不阻塞训练，但建议在启动前/启动时处理）

| # | 位置 | 内容 |
|---|---|---|
| N-1 | `q3vl/train/args.py:27`、`tokens.py:138` | `reinit_special_token_rows` 默认 True 且**无条件**执行。Base SFT 首训正确，但一旦 Stage-Where/What 把 `model_name_or_path` 指向本阶段的 protected checkpoint，这四行**训练好的** `<where>/<color>` embedding 会被静默重初始化。建议加一道「tokenizer 本来就已含这四个 token ⇒ 不重初始化」的门，或在下游 config 显式关掉。 |
| N-2 | ~~`q3vl/train/trainer.py:87-104`~~ | **已自行推翻，留档**：我起初按 `len(dataloader)//GAS` 推断 `max_steps` 会是 4975（比 manifest 的 `ceil(159215/32)=4976` 少 1）。2 卡 ZeRO-3 实跑证伪：日志 `steps_per_epoch(from trainer state)=4976 -> half_epoch_step=2488, full_epoch_step=4976`，与 manifest 口径**完全一致**，无告警也无偏差。无需处理。 |
| N-3 | — | `N_effective=159215` 为奇数，`DistributedSampler` 会补齐到 159216，**恰好 1 条样本在一个 epoch 内出现两次**。SPEC §3.2「每个合格样本在一个 epoch 内恰好出现一次」字面上被破坏 1 条，影响可忽略，但应在报告里点名而不是让人自己发现。 |
| N-4 | `q3vl/train/trainer.py:106-134` + `configs/sft_base.yaml:63` | protected checkpoint 被排除在轮转之外，所以磁盘上的 checkpoint 数会**超过** `save_total_limit`（最多 3+2=5）。这是设计意图，但需要写进项 12（见 B-4）。 |
| N-5 | `NOTES.md`（model）决策 D-5 | 「`eval_loss` **只做记账**、**不导致任何删除**」不准确。`transformers/trainer.py:3327-3341` 在 `metric_for_best_model` 非空且 `save_strategy=steps` 时**照样**设置 `state.best_model_checkpoint`（与 `load_best_model_at_end` 无关），而 `_sorted_checkpoints`（`trainer.py:4386-4393`）会把 best 挪到列表末尾以**免于滚动删除**。即 eval_loss 确实影响「哪些 checkpoint 活下来」。若主 agent 要的是"绝对惰性"，唯一干净的做法是 `metric_for_best_model: null`，把 eval_loss 当普通日志指标记。 |
| N-6 | `q3vl/data/bake.py:116-125` | `bake_one` 直接 resize 到 plan 给的 `(out_w,out_h)`，**没有**再断言解码 + `exif_transpose` 后的真实尺寸等于 header pass 的 `oriented_w/h`。若 header 探测出错（该 pass 自己就踩过一次 42 张误判），结果是**静默的非等比拉伸**。现有保护只有 24 条抽样保真度（我另独立抽 12 条复核，size 全等、RMSE ≤ 2.70，均为 JPEG 量化）。在 `bake_one` 里加一行零成本断言即可关掉这一类。 |
| N-7 | `q3vl/train/shards.py:361-362` | `__del__` 里 `close()`，dataloader worker 退出/解释器关闭时可能刷噪声。仅美观问题。 |
| N-8 | `PREFLIGHT_MODEL.md:221` | 报「68 passed」。审阅时点实测 `q3vl/train/tests` 68 passed、`q3vl/tests` 23 passed，全量 **91 passed**（我复跑；需 `CUDA_VISIBLE_DEVICES` 非空，否则 5 条用例因 `TrainingArguments` 拒绝 bf16/cpu 而失败——环境约束不是缺陷，但建议写进复现命令）。 |
| N-11 | 环境 | **`import torch` 之后再 `import sqlite3` 在本环境必然失败**：`ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version 'CXXABI_1.3.15' not found (required by …/libicui18n.so.78)`；反序 (`sqlite3` 先) 正常。可稳定复现。后果：`pytest q3vl/train/tests q3vl/tests` **collection 直接中断**，而 `pytest q3vl/tests q3vl/train/tests`（数据在前）98 passed。训练入口不受影响（不 import sqlite3），`q3vl.data.cli verify` 也不受影响（sqlite3 在 torch 之前加载），但任何「先起 torch 再读 shard catalog」的后续脚本都会踩。建议在 `launch_sft.sh` / 复现命令里固定 `LD_PRELOAD`/`LD_LIBRARY_PATH`，或在 `q3vl/__init__.py` 里先 `import sqlite3`。 |
| N-9 | 仓库状态 | `q3vl/` 至今**未进 git**（`git status` = `?? q3vl/`）。`config/env.json` 诚实地记录了 `"git_dirty_paths_q3vl": "?? q3vl/;"`，`git_commit.txt` 指向 base commit `bc888fe`——也就是说交付物的 commit 并不包含被审代码。正式训练前建议先提交，让 checkpoint sidecar 里的 commit 有意义。 |
| N-10 | 计划层面（非实现缺陷） | `T_lut_unseen` 最终只有 **433** 条 / **259** 个 LUT，代价是从训练集剔除 **8,371** 条（5.00%，预算上限打满）；296 个 reserve LUT 里有 **37 个一条样本也没换回来**（其 eval 样本落在 select 角色源上被弃用，即报告里的 142 条）。这是 METACANVAS §2.2 与 eval 池只占 1.96% 的结构性冲突，实现按纪律处理得当，但主 agent 需要知道「unseen LUT generalization」这一章最终只有 433 个样本的统计力。 |

---

## 三、逐项对照 SPEC / METACANVAS（pass 清单）

### 3.1 SPEC §2 基模与可训练边界

| 项 | 判定 | 证据 |
|---|---|---|
| 基模路径、shard 完整性 | **pass** | `modeling.py:47-140` 校验 index 里 713 个 tensor 全部可从 shard 读出、字节数与上游一致、`.aria2/.part/.incomplete` 为 0；上游 sha256 硬写在 `modeling.py:21-31` 并注明 revision `ebb281ec…`；入口 `train_sft.py:150-158` 每次启动重验（实测日志：`shard verification PASSED (2 shards)`） |
| Vision blocks / patch_embed 冻结 | **pass** | `freeze.py:30-35` 按名判定；`tests::TestFreezeRule` 12 个参数名参数化用例 |
| 主 merger + 3 个 deepstack merger 可训练 | **pass** | `constants.py:68-72` `VISUAL_TRAINABLE_PREFIXES` 在 `model.visual.` 子树内显式豁免；`freeze.py:128-139` 断言恰好 24 blocks / 3 deepstack / merger 存在 |
| **没有** `model.visual.requires_grad_(False)` 一刀切 | **pass** | 全仓 `rg` 无此调用；SPEC §2.2 点名的坑已避开 |
| Language + tied embedding 全参可训练 | **pass** | `freeze.py:34-35` 默认 True；`tokens.py:163-175` 断言 input/output embedding 共享存储 |
| 无 LoRA/PEFT | **pass** | `preflight_model.py:111-113` 断言无 `lora`/`adapter` 命名参数；全仓无 peft import |
| 参数量与比例 | **单卡 pass / ZeRO-3 见 B-3** | 单卡实测 4,131,573,248 / 4,437,815,808 = 93.0993% |

### 3.2 SPEC §3 数据范围与存储契约

| 项 | 判定 | 证据（本人独立复算） |
|---|---|---|
| 十个 build g1-g4 + l1-l6 | **pass** | `q3vl/data/config.py:27-38` 逐一列名；发布 index 的 build 分布覆盖全部 10 个 |
| split authority 只读冻结旁表 | **pass** | `config.py:45-48` 指向 `splits-20260803/`；manifest 记了三张表的 sha256 |
| `N_effective` 由 terminal manifest 决定 | **pass** | manifest `n_effective = 159215`；`train_sft.py:194-200` 若 manifest 与 index 长度不一致直接 `SystemExit`；`train_sft.py:179-183` 缺 manifest 直接退出，**不回退 2645/5290** |
| 全局 shuffle、每样本一 epoch 一次、不重采样 | **pass**（见 N-3） | HF 默认 sampler + `data_seed=42`，无 WeightedSampler / 无重复 |
| indexed-tar-shard 契约 | **pass** | 我用不经 `q3vl` 的裸 `seek/read` 按 index 的 `offset/length` 直读 26 个 record 成员，**sha256 全部吻合**，且 `size == length`（未压缩）；`shardio.py` 复用 `dataset_build.tools.indexed_tar` 的 writer/validator，staging 目录 + 单次 `rename` 发布，partial shard 不可能进 manifest |
| manifest digest 可复算 | **pass** | 我按 `pipeline.py:529-532` 的口径重算 digest，与文件内 `digest` **逐字符相同**（`9278d721…7ac840`）；5 个 split 的 `index_sha256` / `ids_sha256` 我逐个重算，全部命中 |

### 3.3 SPEC §4 两段式 reasoning（**必审重点 1**）

**我自行另抽了 26 个样本**（`train` 14 / `V_where` 3 / `V_what` 3 / `T_final` 3 / `T_lut_unseen` 3，
seed 20260805），用**自己手写的**七段解析器（tag 字符串从
`dataset_build/src/construct/responses.py:157-167` 手抄，不 import `q3vl.data.twoseg`）
直接对 `/mnt/nfs/bc/data/builds/*/sft.jsonl` 的原文重做一遍转换，再与从 tar 里裸读出的
发布 record 比对。**26/26 全部通过**，逐条检查：

- `where` 与 `region_scope` 正文逐字相等 —— pass
- `color` == 其余六段按 canonical 原序以 `\n` 连接 —— pass（顺序：`problem_lighting → problem_global_color → problem_specific_color → plan_lighting → plan_global_color → plan_specific_color`，与 SPEC §4.2 的固定顺序一致）
- `region_scope` 正文**未**被复制进 `color` —— pass
- 十四个旧 start/end 标签在 `where+color` 中**一个不剩** —— pass
- 收束文本：源语料实测 **0 条**存在，`has_closing_text` 全 false，**未补写**任何收束文本 —— pass
- `instruction` 逐字未改（且用的是长指令 `instruction`，非 `instruction_short`）—— pass
- `preset_path` / `task_type` / `winner_confidence` 与源行一致 —— pass

`q3vl/data/twoseg.py` 的解析器是**位置式**的（`twoseg.py:77-96`：把 14 个 tag 的出现序列
与 canonical 序列整体比对，并检查段间/段前无杂散文本），因此重复、缺失、乱序、未闭合
四种情况都能抓——这正是 SPEC §4.2 要求进 rejection report 而不是猜测性修复的四种。
`q3vl/tests/test_twoseg.py` 对这四种各有用例。

`<where>` 在 `<color>` 之前：由 `collator.py:118-122` `build_target_text` 与
`collator.py:141-157` 的拼接顺序结构性保证，我在真实 batch 上验过（见下）。

### 3.4 SPEC §4.3 special token（**必审重点 4**）

| 项 | 判定 | 证据 |
|---|---|---|
| 四个字符串各编码为 1 个 token | **pass** | 我本机用真实 tokenizer 复验：`{'<where>':151669,'</where>':151670,'<color>':151671,'</color>':151672}`，`tokens.py:50-69` 还额外断言 `decode([id]) == 字面量` 与四 id 互异 |
| 追加而非替换 additional_special_tokens | **pass** | `tokens.py:39-42` 显式 `replace_additional_special_tokens=False`（NOTES 已核实默认 True 会抹掉 `<\|image_pad\|>` 等原生 token） |
| tied-weight 契约 | **pass**（单卡） | `tokens.py:163-175` 比对 `data_ptr()` |
| resize | **B-2** | ZeRO-3 下会缩表，见上 |
| save/reload 一致性 | **pass**（单卡） | `tokens.py:178-187` `verify_reload`；`tests::TestSpecialTokens::test_reload_keeps_ids` 覆盖 |
| 四个 token 参与 loss | **pass** | 见下 §3.5 |

### 3.5 SPEC §4.4 Loss mask（**必审重点 2**）

我用**真实 processor + 真实发布数据**（50 条随机样本，其中 6 条组 batch）独立断言：

```
OK  no image placeholder supervised          （image_pad token 被监督数 = 0）
OK  supervised labels == input_ids
OK  everything unsupervised has SEG_IGNORE
OK  padding fully ignored
OK  no supervision before assistant turn     （第一个被监督 token 恰是 <where>）
OK  where strictly before color
OK  <where>/</where>/<color>/</color> 各恰好出现 1 次且均被监督
```

被监督区间解码出来是：
`'<where>global adjustment across the entire frame</where><color>…</color><|im_end|>\n'`；
未监督前缀的尾部是 `'…<|im_end|>\n<|im_start|>assistant\n'`。
即 system/user/chat-template/image placeholder 全部 `IGNORE_INDEX`，只监督两段 target
（外加 `<|im_end|>\n`，`collator.py:11-18` 有明确理由说明并可用 `supervise_eos` 关闭）。

**分段诊断不改变总 loss**：`trainer.py:170-182` 的 `compute_loss` 直接返回
`outputs.loss`，分段统计走 `torch.no_grad()` 的 `diagnostics.py:69-88`，`detach()` 后计算，
不参与反向。SPEC §4.4「记录 Where 与 Color 各自的 token loss 只用于诊断，不改变总 loss 权重」
**pass**。

**额外核查过的一个隐患（结论：无问题）**：自定义 `compute_loss` 没有把
`num_items_in_batch` 透传给模型。若 `Trainer.model_accepts_loss_kwargs` 为 True，
`transformers/trainer.py:4060-4064` 会**跳过** `loss / GAS`，等效学习率放大 4 倍。
实测 `Qwen3VLForConditionalGeneration.accepts_loss_kwargs = False`（类属性显式为 False），
因此 `num_items_in_batch is None`、`loss` 照常除以 GAS。**安全，但这是脆的**：
若将来换模型或换 transformers 版本，建议在 trainer 里显式 `self.model_accepts_loss_kwargs = False`
把它钉死。

### 3.6 SPEC §5 图像契约（**必审重点 6**）

我对 50 条发布样本**解码真实入库 JPEG**逐条断言，**50/50 通过**：
尺寸 == record plan、短边恒 512、双边 32 对齐、长边 ≤ 2048、`aspect_in ≤ 4`、
`vision_tokens == (H/32)*(W/32)`、32 对齐引入的宽高比误差 < 3.2%、
`plan_geometry(oriented_h, oriented_w)` 可复现同一组几何。

- EXIF：`headers.py:88-99, 126-127` 先读 orientation，5-8 转置后再算几何；实测语料含
  orientation 6（13 条）与 8（1,182 条）。`imageproc.py:141` 训练侧同样 `exif_transpose`。
- 不用 `max_pixels`：`imageproc.py:13-17` 明确说明并用 `do_resize=False`
  （`collator.py:197-199`）把几何权交给自己，`collator.py:200-202` 再与 processor 的
  `image_grid_thw` 交叉断言（`imageproc.py:164-175`）。SPEC §5 的告诫已落实。
- >4:1 过滤：`imageproc.py:84-86`；实测语料最大 aspect 3.207，**该分支计数为 0**——
  报告已明说「不是被裁出来的，是被 4:1 过滤器结构性保证的」，措辞准确。
- 长边 2048：`imageproc.py:96-102` 是断言而非裁剪，注释也说明了
  `aspect ≤ 4 且短边 512 ⇒ 长边 ≤ 2048`。逻辑成立。
- **短边 < 512 的 14.2%（23,045 条）按契约上采样**：SPEC §5 条 2 无下限例外，实现照做，
  报告单列 `upscaled` 计数与原始短边分布（min 256 / p05 360）。符合主 agent 裁定 D-8。
- 烘焙保真：我独立抽 12 条，从**生产 build 的原始成员**（sha256 校验通过）重新过一遍
  `prepare_image`，与入库图逐像素比：**size mismatch 0，RMSE mean 1.735 / max 2.704**，
  差异仅为 JPEG q95 4:4:4 量化。支持主 agent 裁定 D-1。（另见 N-6）

### 3.7 SPEC §6 序列长度

- 长度口径不是数据侧自造：`lengths.py:22-24` 直接 import 训练侧 collator 构造 prompt/target。
- **我独立复核了 50 条**：`Sft2SegCollator.encode_one` 的真实长度 == split index 的
  `total_tokens` == record 的 `tokens.total`，**50/50 完全相等**（交付报告只做了 16 条，
  本次扩到 50 条仍全中）。**必审重点 9 pass。**
- 全量分布：max 1142、p99 773，`>2048` 计数 0；`over_limit_remaining = 0`，
  `filtered_too_long = 0`。我从 5 个 split index 独立统计 `total_tokens`，
  `over 2048` 同样为 0。
- 超长不截断：`collator.py:159-160` 抛 `SequenceTooLong`，`tests::test_overlong_raises_not_truncates` 覆盖。
- 按原因分组的过滤数：`PREFLIGHT_DATA.md` §2 给了完整加减账
  （172,580 − 9 − 1,699 − 8,371 − 142 = 162,359），并**点名了所有为 0 的原因**
  （七段结构性拒绝 / 超长 / 宽高比 / 图像损坏 / index-checksum / split 不一致）。
  我按 5 个 split 的实际行数复算：918+433+897+896+159,215 = **162,359**，账对得上。

### 3.8 SPEC §7 优化器与分布式（**必审重点 8**）

`q3vl/train/configs/sft_base.yaml` 与 `q3vl/train/args.py:64-129` 逐项对照：

| SPEC §7 | 冻结值 | 配置 | 启动断言 |
|---|---|---|---|
| optimizer | `adamw_torch` | ✓ | `FROZEN_HPARAMS` |
| learning_rate | 1e-5 | ✓ | ✓ |
| weight_decay | 0.0 | ✓ | ✓ |
| betas / eps | 0.9 / 0.999 / 1e-8 | ✓ | ✓ |
| warmup_ratio | 0.03 | ✓ | ✓ |
| lr_scheduler_type | cosine | ✓ | ✓ |
| max_grad_norm | 1.0 | ✓ | ✓ |
| bf16 | true | ✓ | ✓ |
| ZeRO-3 | `ds_zero3.json` stage 3 | ✓ | — |
| attention | flash_attention_2 | ✓ | — |
| gradient_checkpointing | true | ✓（`use_reentrant=False`，`args.py:105-110`） | ✓ |
| global batch 32 | 4 × GAS4 × world 2 | ✓ | `args.py:155-169` 强制断言，且只允许 (4,4)/(2,8) |

- `ds_zero3.json` 不含 `optimizer` 段 ⇒ 用 HF 的 client optimizer（`adamw_torch`），与 SPEC 一致；
  `gradient_clipping: auto` / `train_batch_size: auto` 由 HF 填成 1.0 / 32；
  `stage3_gather_16bit_weights_on_model_save: true`（保存 checkpoint 必需）。**pass**
- merger 与 Language 同一学习率、无分层 LR：`args.py` 无 param group 定制，`Trainer` 默认单组。**pass**
- 逃生开关 `Q3VL_ALLOW_NONSPEC`（`args.py:134-142`）只降级为 warning，注释明确「real run must never set it」；
  `launch_sft.sh` 不设置它。**可接受**，但正式启动时请确认环境里没有这个变量。

### 3.9 SPEC §8 训练长度、checkpoint（**必审重点 5**）

**双删除路径修复：已核实两条路径确实都被拦住。**

我打开本机 `transformers 4.57.1` 源码逐行确认：
- 路径 A：`trainer.py:4396-4419` `_rotate_checkpoints` → 候选来自 `self._sorted_checkpoints`；
- 路径 B：`trainer.py:2836-2844` `_inner_training_loop` 收尾处的
  `save_total_limit == 1 and best_model_checkpoint is not None` 分支，**确实不经过**
  `_rotate_checkpoints`，但同样调用 `self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)`。

`q3vl/train/trainer.py:106-134` 覆写的正是两条路径**共同的取数口** `_sorted_checkpoints`，
且是在 `super()` 做完 best-checkpoint 索引记账**之后**再过滤（这一点很关键：基类
`trainer.py:4386-4393` 会 `list.index(best)`，先过滤会 `ValueError`；
`tests::test_protected_best_checkpoint_does_not_crash_sorting` 正是覆盖它）。
单测 `TestCheckpointProtection` 用**逐字复制的** 4.57.1:2841-2845 代码模拟路径 B，
并同时验证「protected 不被删」与「非 protected 仍被删」。**实现审阅结论：修复正确、覆盖充分。**

**步数来自 manifest 而非硬编码 2645/5290：pass。**
- `train_sft.py:179-183`：缺 `terminal_manifest` 直接 `SystemExit`，注释点名不得回退估算值；
- `train_sft.py:194-201`：`n_effective` 与 train index 长度不一致直接退出；实算
  `ceil(159215/32) = 4976`（我独立复算 manifest 得同值）；
- `trainer.py:87-104`：protected step 从 `state.max_steps` 推导，`resolve_protected_steps`
  的单测显式断言「除非数据这么说，否则 2645 不出现」；
- 实算 `max_steps = 4975`（见 N-2），half = 2488。**2645/5290 不可能出现在本次运行中。**
- 全仓 `rg "2645|5290"` 只命中文档与单测，不在任何运行时路径。

其余：`eval_steps/save_steps/save_total_limit = 500/500/3` ✓；`num_train_epochs = 1.0` ✓；
protected checkpoint 附带 tokenizer/processor/special-token map/training state
（`Trainer._save_checkpoint` 保存 `processing_class`）+ `protected_checkpoint.json` 侧车
（`trainer.py:148-167`，含 manifest digest、`N_effective`、special token id、freeze 报告）✓。

§8.3 辅助诊断：token accuracy / Where-Color 分段 loss 已实现（`diagnostics.py:24-88`）；
标签完整率 / 顺序正确率 / 非空率 / 旧七标签残留率需生成，`trainer.py:221-253` 已实现但
`gen_diag_samples: 0` 默认关闭（主 agent 裁定 D-3：离线在 protected checkpoint 上跑）。

### 3.10 METACANVAS §2 隔离 split（**必审重点 7**）

我**不经 `q3vl.data.splits`**，直接从 5 个发布 index 重算全部交集：

```
sample_id 两两交集      : 全部 0（10 对）
protocol group (source_image_id, lut_id, build) 两两交集 : 全部 0（10 对）
source_image_id 两两交集: 除 T_final|T_lut_unseen = 201 外全部 0
train ∩ T_lut_unseen 的 lut_id : 0        （train 3,149 个 LUT，T_lut_unseen 259 个）
reserve 的 296 个 lut_id ∩ train 的 lut_id : 0
T_lut_unseen 的 lut_id 全部落在 reserve 内 : 是（0 个例外）
train_removed = 8,371 = 4.995% ≤ 5% 预算
```

- **必审重点 7 pass**：`T_lut_unseen` 的 296 个 reserve LUT **确实一个也不在训练 manifest 里**，
  train × 四个 eval 集在 sample / source / group 三个层级全部零交集。
- `T_final|T_lut_unseen` 共享 201 个源是**设计如此**（两者同属 test 角色，靠 LUT 身份区分），
  与 METACANVAS §2.2「source image 不跨 train/select/test」三角色口径一致；
  select 角色（V_where ∪ V_what）与 test 角色的源集合交集为 0（我复核 = 0）。
- 落在 select 源上的 reserve-LUT 样本被**弃用**而非塞进选择集
  （`splits.py:272-275`，reason `reserved_lut_in_select_source`，142 条）——这条纪律很关键，
  否则 Where/What 的 checkpoint 会部分地在未见 LUT 上被选出来。**做法正确。**
- 划分完全确定性：`splits.py:46-47` 以 SHA-256 打破 tie，无 RNG；`_Union` 的合并按
  字典序取小根（`splits.py:63-68`）。可复现。
- `dedup_drop` 被正确识别为**删除名单**而非第三个平行集合（train∩dedup=1,667、
  eval∩dedup=32、三者并集 = 172,580），应用后三集合两两无交集，发布集不含任何 dedup id
  （`published_x_dedup = 0`）。SPEC §9 项 6 **pass**。

---

## 四、对主 agent 六项裁定的意见

| 裁定 | 我的意见 |
|---|---|
| **D-5(train)** assistant-only eval_loss 为主选择指标，折中为 `load_best_model_at_end=false` 只记账 | **同意裁定，但前提描述需修正**：`metric_for_best_model` 非空会让 `state.best_model_checkpoint` 被写入，并让该 checkpoint**免于滚动删除**（`transformers/trainer.py:4386-4393`），所以它不是"零影响"。实际风险很小（两个里程碑本就 protected、`save_total_limit=3`），但若主 agent 要的是字面意义的"只记账"，唯一干净做法是 `metric_for_best_model: null`。另：`eval_loss == assistant-only` 已被 mock 端到端逐位验证（4.554167 vs 自算 4.554167），我在代码层复核 loss mask 也确认 `eval_loss` 的分母只含 assistant target token。**不升 blocker。** |
| **D-7** Base SFT eval 集 = `V_where` | **同意。** 配置里确实只指向 `splits/V_where.index.jsonl`（`sft_base.yaml:26`），`T_final`/`T_lut_unseen`/`V_what` 未被任何配置引用（全仓 `rg` 确认）。`V_what` 对 Base SFT 零接触这一点会被后续 What 臂的审阅感谢。唯一提醒：Stage-Where 的选择集也是 `V_where`，Base SFT 已经在它上面报过 loss，严格说 Where 阶段的"选择集"不再是完全未接触的；但 Base SFT 不用它做任何选择（D-5 折中后），所以可接受。 |
| **D-1(data)** 按契约尺寸重编码 JPEG q95 4:4:4 入库 | **同意。** 关键前提「本战役全程无图像增广」成立（SPEC §5 是确定性变换，训练侧 `prepare_image` 对已达标尺寸是恒等）；每条 record 保留了原成员 `root/shard/offset/length/sha256` + `i_in_path`，我按此定位并 sha256 校验通过、重跑 `prepare_image` 与入库图逐像素比对通过。4:4:4 对配色数据集是正确取舍。**建议补 N-6 的零成本断言。** |
| **D-2(data)** `winner_confidence=low`（41%）保留在训练集 | **同意（以冻结 spec 为准）。** 实测 train 中 low 65,281 / 159,215 = 41.0%。实现把开关备好了（`cli.py:22-24 --drop-low-confidence`），每条 record 与每行 index 都带 `winner_confidence`，训练侧可随时二次过滤而无需重跑数据管线。**风险提示**：CLAUDE.md 那条纪律的原意是「low 不进评测 GT」，而当前四个 eval 集里 low 占比同样是 36%-44%（T_final 385/918、V_where 381/896）。若主 agent 只想放宽训练侧而不想放宽评测侧，需要在**评测阶段**另外定口径——这不是本次实现的缺陷，但现在不说，后面会变成"评测 GT 里有 40% 低置信"的被动。 |
| **D-8(data)** 短边 < 512 上采样 | **同意（契约明文如此）。** 但请把 `PREFLIGHT_DATA.md` §6 的提醒当真：23,045 条（14.2%）的高频细节是插值补出来的，**Where 阶段的边缘质量指标在这批图上天然偏乐观**。建议 Where 臂交付时把指标按 `upscaled` 分层报一次（record 里已有 `image.upscaled` 字段，零成本）。 |
| **D-3(train)** 生成类诊断训练中关闭、在 protected checkpoint 上离线跑 | **同意。** ZeRO-3 下 `model.generate` 需逐层 gather，训练中开销与风险都不划算；代码已完整实现且异常被捕获不会杀训练（`trainer.py:206-215`）。**唯一要求**：SPEC §8.3 把「标签完整率 / 顺序正确率 / 非空率 / 旧七标签残留率」列为辅助诊断，离线跑的结果必须进最终 REPORT.md，否则 §8.3 只交付了一半。 |

---

## 五、独立复核记录（本次审阅实际跑过的东西）

只读，未改动任何文件。

1. `q3vl/tests` + `q3vl/train/tests`：**91 passed**（`CUDA_VISIBLE_DEVICES=0`）。
2. 独立 split 审计脚本（不 import `q3vl`）：5 个 index 的两两 sample/group/source/lut 交集、
   manifest digest 重算、5×2 个 sha256 重算、reserve 集合关系、`total_tokens` 越限统计。
3. 独立七段→两段核对：自写解析器 + 裸 tar 定位读 + 原 `sft.jsonl` 对照，**26 条全通过**。
4. 真实 processor + 真实数据的 loss mask / 长度 / 图像契约断言：**50 条全通过**，
   6 条组真实 batch 验 mask。
5. 独立烘焙保真：12 条从原始 build 成员重放，size 全等、RMSE ≤ 2.704。
6. `transformers 4.57.1` 源码逐行核对两条 checkpoint 删除路径。
7. `deepspeed.zero.Init` 下的 `param.shape` / `sum(numel)` 实测（B-2 / B-3 的证据）。
8. `is_deepspeed_zero3_enabled()` 在 `SFTTrainingArguments` 构造后即为 True 的实测（B-1 触发条件）。
9. 读取并引用了并行联合 preflight 的崩溃日志
   `/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/phaseA/train.log`（B-1 的现场证据）。

---

## 六、放行条件

清掉 B-1 / B-2 / B-3 / B-4 后，**必须补一次「2 卡 ZeRO-3、真实 config、真实数据、跑到
至少一次 eval + 一次 save」的联合 smoke**，并在 PREFLIGHT_MODEL.md 里用该配置重新出示：

1. frozen/trainable 的**非零**参数量与比例（§9 项 2）；
2. Vision 无梯度 / 4 个 merger + Language 有梯度（§9 项 3，ZeRO-3 下需用 `ds_numel` 或
   `GatheredParameters` 取真实梯度）；
3. 四个 special token 的 ID / 单 token / tied / reload，**以及 resize 后 embedding 的实际行数**（§9 项 4）；
4. 单 step 显存与吞吐（§9 项 10，当前 42.18 GiB 是单卡无 ZeRO-3 的数字，
   报告自己也标了「常驻状态数字不可迁移」）；
5. §9 项 12 的完整一节。

在此之前，**不准许进入正式训练**。

---

## 附录：审阅期间代码被并行热修（必读）

本审阅的代码快照取自 2026-08-05 00:11-01:57 之间的文件状态（`q3vl/train/*.py` mtime 均在
23:43-00:22 区间）。**审阅进行到一半时，实现侧开始热修**：

| 文件 | mtime | 与本报告的关系 |
|---|---|---|
| `q3vl/train/freeze.py` | 01:58:17 | B-3 |
| `q3vl/train/tokens.py` | 01:58:56 | B-1 + B-2 |
| `q3vl/train/tests/test_train_pipeline.py` | 02:00:29 | 新增用例（全量 68 → 75） |

时间线：01:53 联合 preflight 起 2 卡 ZeRO-3 → 01:54:38 在 `tokens.py:97` 崩溃 →
01:58 起改这三个文件。热修方向与本报告的三条 blocker 一致：

- `tokens.py` 新增 `is_zero3_partitioned()`，`_mean_init_rows` 改为
  `noise.to(mean.device)`（B-1），`prepare_embeddings` 改用 `full_shape(emb.weight)`
  并加了「NEVER read `emb.weight.shape[0]` directly」的注释（B-2），
  reinit 包进 `gathered([weight], modifier_rank=0)`；
- `freeze.py` 新增 `full_numel()/full_shape()`（读 `ds_numel`/`ds_shape`）与一条
  「freeze report counted 0 parameters」的显式报错（B-3）。

02:01:53 第二次提交 2 卡 ZeRO-3（`job.marker` 已按 D-20 四步写出，`role` 标注
"S0-JOINT preflight smoke (not production training)"）。该次**起来了并且在跑**。
我在 02:0x 读取其产物，据此对三条 blocker 逐条复判：

| blocker | 复判 | ZeRO-3 实证（`phaseA/train.log` + `run_setup.json`） |
|---|---|---|
| **B-1** 入口崩溃 | **CLOSED** | 入口通过 `setup_model`，进入训练循环；step 1-29 的 loss `3.4211 → 3.1362`、`grad_norm 33.2 → 13.3`，全程有限 |
| **B-2** 缩表 | **CLOSED** | `embeddings: {"embedding_rows_before": 151936, "zero3_partitioned": true, "resize_action": "kept", "embedding_rows_after": 151936, "tied_ok": true, "min_pairwise_distance": 0.0016162680694833398}`——行数不变、tied 保持，且 `min_pairwise_distance` 与单卡 preflight **逐位相同**，证明四行 embedding 与已验证过的取值完全一致 |
| **B-3** 参数量塌成 0 | **CLOSED** | 日志 `total 4,437,815,808 / trainable 4,131,573,248 (93.0993%) / frozen 306,242,560 (6.9007%)`，与单卡 preflight 逐位一致；`run_setup.json.freeze` 同值 |
| **B-4** §9 项 12 缺失 | **OPEN** | 两份 preflight 报告均未补 |

顺带被这次实跑证实/证伪的几点（已回写进上文）：

- `steps_per_epoch = 4976`、`half_epoch_step = 2488`，与 manifest 的
  `ceil(159215/32)` **完全一致**，无告警（N-2 因此作废）；
- 分段诊断按预期只做记账：`train_seg_where_loss / where_acc / color_loss / color_acc /
  eos_loss / assistant_loss` 全部出现在日志里，而 `loss` 字段是模型自身的总 loss；
- accelerate 报 `Gradient accumulation steps mismatch: plugin 1 vs DeepSpeed 4, using DeepSpeed's`
  ——DeepSpeed 侧取 4，`train_batch_size` 由 `auto` 解析为 32，global batch 32 成立；
- 实测 ~5.1 s/step ⇒ 一个 epoch 约 **7.0 小时**（4976 步）。请注意这次"smoke"用的是
  完整 config，若不主动停它会跑满整个 epoch。

**仍需在放行前完成（更新后的放行条件）**：

1. **B-4**：补 SPEC §9 项 12 一节（NFS durable 输出路径、scratch 容量上限、日志与
   `job.marker` 路径，外加 protected + 轮转合计最多 5 份 ZeRO-3 checkpoint 的磁盘预算）。
2. **把上表这些 ZeRO-3 数字回写进 `PREFLIGHT_MODEL.md` / `preflight_model.json`**。
   目前两份文件里 §9 项 2/3/4/10/11 的数字**仍标注为单卡、无 DeepSpeed** 采集；
   证据已经存在（`run_setup.json`），但交付文档尚未引用它，规格意义上 §9 仍未闭合。
3. **对热修后的 `q3vl/train/tokens.py` 与 `q3vl/train/freeze.py` 做一次补充实现审阅**
   ——本报告的 pass 清单不覆盖 01:58 之后的版本。建议重点看：
   `gathered(..., modifier_rank=0)` 退出时的 rank-0 广播是否让四行 embedding 跨 rank
   严格一致；`full_numel/full_shape` 的 fallback 是否会在非 ZeRO-3 场景下改变既有数字
   （实测未变）；以及新加的「freeze report counted 0 parameters」报错是否有单测覆盖。
4. 处理或明确豁免上文 N-1（下游 checkpoint 会被重初始化）、N-5（`metric_for_best_model`
   实际影响 checkpoint 留存）、N-6（bake 缺一条零成本断言）、N-9（`q3vl/` 未进 git）、
   N-11（torch→sqlite3 的 `CXXABI` 冲突）。

**修订后的判决：BLOCKER 提出 4 条，其中 3 条已由并行热修 + 2 卡 ZeRO-3 实跑关闭，
B-4 仍未关闭；另有 2 项交付级要求（上面第 2、3 条）未完成。**
（上述第 3 条已于 02:08-02:12 完成，见下一节的聚焦补审。）

---

## 补审 2026-08-05 02:12：热修 diff 聚焦复审（`tokens.py` / `freeze.py`）

> 范围：仅审 01:58 热修引入的改动及其单测，不重审全量。
> 被审版本：`q3vl/train/tokens.py` md5 `7fa7b6adc1b6e78cce1ff4725c19364f`（mtime 01:58:56）、
> `q3vl/train/freeze.py` md5 `2f030ef9bc2ca78ff7cd86680a5f9d06`（mtime 01:58:17）、
> `q3vl/train/tests/test_train_pipeline.py`（mtime 02:00:29）。审阅期间未再变动。
>
> **补审结论：PASS，无新增 blocker。** 新增 3 条 nit（N-12/N-13/N-14）。

### 补审 P-1 — B-1 修法正确（设备不匹配）

`tokens.py:120-126`：噪声仍由 **CPU generator** 抽取，抽完再 `noise.to(mean.device)` 后
才与 `std` 相乘。这个顺序是对的，而且比"把 generator 换成 CUDA generator"更好——
**抽样值与设备/world size 无关**，因此四行 embedding 在单卡 preflight 与 2 卡 ZeRO-3 下
逐位相同。实证：两次运行的 `min_pairwise_distance` 均为
`0.0016162680694833398`（**逐位一致**）。

顺带修掉了同一函数下游的一个同类隐患：`tokens.py:194` 的 `torch.eye(...)` 现在带
`device=pair.device`，否则在 CUDA 权重上算 `min_pairwise_distance` 会以同样的方式崩。

单测：`test_mean_init_rows_on_cuda_weight`（GPU 上跑 bf16 权重 + CPU generator，
断言有限且四行互不相等）、`test_mean_init_rows_is_device_independent`（同种子两次调用
逐位相等）。**pass**

### 补审 P-2 — B-2 修法正确（缩表）

`freeze.py:57-62` 新增 `full_shape()`（优先 `ds_shape`），`tokens.py:144/162` 改用它，
并在 `tokens.py:139-143` 写下为什么不能直接读 `emb.weight.shape[0]` 的完整因果链。

我用**真实 `deepspeed.zero.Init`** 复核（单进程、真实 `ds_zero3.json`）：

```
is_zero3_partitioned : True
raw shape / numel    : (0,) 0
full_shape/full_numel: (151936, 8) 1215488   -> rows_before = 151936
=> need(151673) > rows_before? False -> resize_action == 'kept'
```

即 ZeRO-3 下**不再触发 resize**，缩表路径被彻底切断。2 卡实跑的
`run_setup.json.embeddings` 同样是 `{"embedding_rows_before":151936, "resize_action":"kept",
"embedding_rows_after":151936, "zero3_partitioned":true}`。

单测：`test_prepare_embeddings_does_not_resize_a_partitioned_matrix` 显式断言
`model.resize_calls == []`。**pass**

### 补审 P-3 — 改动的行手术**确实活过了 re-partition**（本节是本次补审最关键的一项）

单测里的 `_fake_gather` 是手写替身，**不能**证明真实 `GatheredParameters` 会把改动写回
分片；而 `min_pairwise_distance` 是在 gather 窗口**内**算的，同样不能证明。我因此做了两件事：

1. **源码核对** `deepspeed/runtime/zero/partition_parameters.py:2338-2348`：
   `GatheredParameters.__exit__` 在 `modifier_rank is None` 时走
   `partition(has_been_updated=False)`——**会丢弃修改**；只有传了 `modifier_rank` 才
   `dist.broadcast(p.data, src_rank)` 后 `partition(has_been_updated=True)`。
   `tokens.py:41/182` 传的正是 `modifier_rank=0`，**这个参数是承重的，不能随手改成 None**。
2. **实跑验证**（真实 `zero.Init` + 真实 `q3vl.train.tokens.gathered`）：

```
inside window shape        : (151936, 8)
after exit shape           : (0,)   (已重新分片)
rows CHANGED by reinit     : True
rows SURVIVED re-partition : True    max|inside-after| = 0.0
min pairwise distance after: 0.0031026615761220455      all finite: True
```

即改动写回分片后**逐位无损**。**pass**（建议把这条补成单测，见 N-14）

### 补审 P-4 — B-3 修法正确（参数量塌成 0）

`freeze.py:38-54` 新增 `full_numel()`（优先 `ds_numel`），`apply_arm_b_freeze:126` 改用它；
更重要的是 `assert_freeze_boundary:181-198` 补了**反空洞断言**：先要求
`report.total_params > 0`，再对 6 个子树各要求对应计数 **> 0**，最后才是原来那组
「必须为 0」的断言。这正好堵住我在 B-3 里指出的「全 0 满足所有 must-be-0 规则」。

回归检查：非 ZeRO-3 路径 `full_numel/full_shape` 退化为 `numel()/shape`，数字不变——
2 卡 ZeRO-3 实跑报出的 `4,437,815,808 / 4,131,573,248 (93.0993%) / 306,242,560 (6.9007%)`
与单卡 preflight **逐位相同**，无回归。

`vision_pos_embed / language_norm / lm_head` 被**刻意排除**在正向断言之外，是对的：
它们的存在性依模型而异，纳入会让断言变脆。

单测：`test_full_numel_uses_ds_numel`、`test_full_numel_plain_parameter`、
`test_vacuous_report_is_rejected`、`test_apply_freeze_counts_partitioned_params`
（用 `_FakeDSParam` 复刻 `shape==[0] / numel()==0 / ds_shape / ds_numel / ds_id` 的真实状态）。**pass**

### 补审 P-5 — 附带改动与卫生

- `assert_tied`（`tokens.py:211-214`）新增 `inp.weight is out.weight` 的对象同一性快速通道，
  理由（ZeRO-3 下两侧都是零长张量）成立且是**更强**的判据，不构成放宽。实跑 `tied_ok: true`。
- `gathered()`（`tokens.py:41-55`）对非 ZeRO-3 参数返回 `contextlib.nullcontext()`，
  `deepspeed` 是**函数内局部 import**。我实测：只 import `q3vl.train.*` 并走 nullcontext 分支后
  `'deepspeed' in sys.modules == False`——单卡/无 DeepSpeed 路径不会被拖上 deepspeed 依赖。**pass**
- 无循环 import（`freeze` 只依赖 `constants`，`tokens` 单向依赖 `freeze`）。
- `tokens.py:183` 的 `int(weight.shape[0]) < need` 是**窗口内**的事后断言，读裸 `shape` 正确，
  不是 B-2 的遗漏。
- 测试总数 68 → **75**（`q3vl/train/tests`），全量 `q3vl/tests + q3vl/train/tests` = **98 passed**
  （须数据目录在前，见 N-11）；新增 `TestZero3ParameterAccounting` 单独跑 **7 passed**。

### 补审新增 NIT

| # | 位置 | 内容 |
|---|---|---|
| N-12 | `q3vl/train/preflight_model.py:229, 238` | 仍直接读 `emb.weight.shape[0]`。该脚本按设计只跑单卡无 DeepSpeed，所以今天不是缺陷；但同包里已经有 `full_shape`，一旦有人在 ZeRO-3 config 下跑 preflight，会报 `embedding_rows: 0` 且 `ids_in_embedding_range: False`。建议统一改用 `full_shape`。 |
| N-13 | `q3vl/train/tokens.py:215-218` | `data_ptr()` 兜底在 ZeRO-3 下**不可靠**：我实测两个互不相同的 `torch.empty(0)` 的 `data_ptr()` 都是 `0`，因此比较必然相等。该分支只在"绑定真的断了"时才会被走到，而那正是它应该报错的时候 → **可能假通过**。建议两侧都是 ZeRO-3 参数时改比 `ds_id`。 |
| N-14 | `q3vl/train/tokens.py:182-195` | 两点：(a) `_mean_init_rows` 在 `modifier_rank=0` 的窗口里**在所有 rank 上都执行写入**，而 DeepSpeed 的文档用法是「只有该 rank 修改」。因为退出时会以 rank 0 广播、且 CPU generator 定种子使各 rank 本就算出同一结果，**当前无害**，但属于偏离文档契约，建议加注释或加 `if rank == 0` 守卫。(b) 单测只用手写 `_fake_gather` 替身，**没有**覆盖真实 `GatheredParameters` 的写回语义；`modifier_rank=0`→`None` 的一字之差会静默丢弃四行初始化。建议补一个 `skipif(deepspeed 不可用 / 无 GPU)` 的端到端小用例（我在 P-3 里跑的那段可以直接改造）。 |

### 补审后的整体状态

| blocker | 状态 |
|---|---|
| B-1 入口在 ZeRO-3 下崩溃 | **CLOSED**（修法经补审 + 实跑双重确认） |
| B-2 ZeRO-3 下缩表到 151680 | **CLOSED**（同上） |
| B-3 参数量塌成 0、断言空洞 | **CLOSED**（同上，且补了反空洞断言） |
| B-4 SPEC §9 项 12 未交付 | **OPEN**（据主 agent：联合 preflight subagent 正在产出） |
| 交付级：PREFLIGHT_MODEL 的 ZeRO-3 证据回写 | **OPEN** |
| 交付级：对热修代码补一次实现审阅 | **CLOSED（即本节）** |

---

## 最终判决（2026-08-05 02:12 更新）

**BLOCKER 累计提出 4 条，已关闭 3 条（B-1 / B-2 / B-3，经本节聚焦补审 + 2 卡 ZeRO-3
实跑双重确认）。剩余阻塞仅两项，且均为交付面而非代码面：**

1. **B-4**：SPEC §9 项 12（NFS durable 输出路径、scratch 容量上限、日志与 `job.marker`
   路径，含 protected + 轮转合计最多 5 份 ZeRO-3 checkpoint 的磁盘预算）——
   由联合 preflight subagent 正在产出；
2. **`PREFLIGHT_MODEL.md` / `preflight_model.json` 的 ZeRO-3 证据回写**——
   §9 项 2/3/4/10/11 目前仍标注为单卡、无 DeepSpeed 采集，而对应的 ZeRO-3 数字已经存在于
   `/mnt/nfs/bc/runs/q3vl_smoke_joint_20260805/phaseA/run_setup.json`，只是尚未被交付文档引用。

**这两项闭合后即可放行正式训练；实现（代码）侧本次审阅不再有阻塞项。**
N-1 / N-5 / N-6 / N-9 / N-11 / N-12 / N-13 / N-14 为非阻塞建议，请主 agent 决定处理或明确豁免。
