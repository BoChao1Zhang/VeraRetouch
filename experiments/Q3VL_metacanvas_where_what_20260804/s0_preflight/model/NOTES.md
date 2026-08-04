# S0-TRAIN NOTES：核实记录 / 假设清单 / 待主 agent 决策

任务卡：Wave S0 模型/训练部分。规格：`docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md`（全文已读）。
交叉参考：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §2.3 / §3 / §10.1 / §14。

---

## 一、实施前核实记录（全部本机/原始来源实测，无一条来自记忆或二手检索）

### 1.1 上游权重 checksum（唯一用到的外部事实）

`https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true` 当场拉取，
repo revision `ebb281ec70b05090aa6165b016eac8ec08e71b17`：

| 文件 | 上游 size | 上游 sha256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 4,967,229,296 | `30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9` |
| `model-00002-of-00002.safetensors` | 3,908,490,048 | `046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6` |

本地两个 shard 的字节数与 sha256 **均逐字节吻合**。该表写死在
`q3vl/train/modeling.py::UPSTREAM_SHARD_SHA256`，训练入口每次启动都会重验。
`.hfd/repo_metadata.json` 里**没有** checksum 字段（只有文件名列表），故必须走 API。

### 1.2 本地库版本自查（spec §9.1 要求）

| 环境 | transformers | 结论 |
|---|---|---|
| conda `base` | **4.36.0** | 太旧，无 Qwen3-VL |
| conda `jarvisart_rl` | 4.51.3 + deepspeed 0.15.4 + fa2 2.7.3 | 仍无 Qwen3-VL（需 ≥4.57.0） |
| conda `unsloth` | 4.57.3，无 flash-attn / deepspeed | 缺件 |
| conda `vllm` | 4.57.6，无 flash-attn / deepspeed | 缺件 |
| **conda `llm_factory`** | **4.57.1 + torch 2.10.0+cu128 + flash-attn 2.8.3 + accelerate 1.11.0** | 只缺 deepspeed |

Qwen3-VL 的 transformers 实现自 4.57.0 起提供；`llm_factory` 是唯一同时具备
Qwen3-VL、flash-attn 与匹配 torch 的环境。已实测 `Qwen3VLForConditionalGeneration`
/ `Qwen3VLProcessor` 可导入，`torch.cuda.device_count()==2`。

### 1.3 API 签名逐条本机验证（不依赖记忆）

| API | 实测结果 | 影响 |
|---|---|---|
| `TrainingArguments` | 有 `eval_strategy`，**无** `evaluation_strategy` | 配置用新名 |
| `Trainer.__init__` | 有 `processing_class`，无 `tokenizer` | 传 processor |
| `Trainer._save_checkpoint(self, model, trial)` | 位置参数，无 `**kwargs` | 覆写签名兼容 |
| `Trainer.log(self, logs, start_time=None)` | 两参 | 覆写带回退 |
| `Trainer._sorted_checkpoints(output_dir, checkpoint_prefix, use_mtime)` | 关键字 | 覆写用同名参数 |
| `PreTrainedTokenizerBase.add_special_tokens(..., replace_additional_special_tokens=True)` | 默认 **True 会替换**整个列表 | 必须显式传 `False` 追加，否则会抹掉 `<\|image_pad\|>` 等 13 个原生 token |
| `PreTrainedModel.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)` | 三参 | 见 D-4 |
| `Qwen2VLImageProcessorFast(do_resize=False)` | 走 `stacked_images.shape[-2:]`，`grid_h=H/16, grid_w=W/16` | 可用自控几何 |
| `Qwen3VLProcessor.__call__` 占位符展开 | `num_image_tokens = grid_thw.prod() // merge_size**2` | 手工展开口径与官方一致 |

### 1.4 模型结构实测（对照 spec §12）

`config.json` 与 meta-device 实例化的 `named_parameters()` 双向核对：

- vision：`depth=24`、`hidden_size=1024`、`patch_size=16`、`spatial_merge_size=2`、
  `deepstack_visual_indexes=[5,11,17]`、`out_hidden_size=2560` —— 与 spec §12 完全一致；
- text：36 层、hidden 2560、`vocab_size=151936`、`tie_word_embeddings=true`；
- 参数名前缀实测：`model.visual.blocks.*` / `model.visual.patch_embed.*` /
  `model.visual.pos_embed` / `model.visual.merger.*` /
  `model.visual.deepstack_merger_list.{0,1,2}.*` / `model.language_model.*`；
- `lm_head.weight` **不出现在** `named_parameters()`（tied，`_tied_weights_keys` 去重）。

### 1.5 图像几何实测（spec §5）

`do_resize=False` 下 LLM-facing token 数实测：`512x512→256`、`512x1024→512`、
`512x2048→1024`、`768x512→384`，即 `(H/32) x (W/32)`，与 spec §5 给的例子逐个吻合。

推论一条（已写进代码注释）：**宽高比 ≤ 4 且短边固定 512 ⇒ 长边必 ≤ 2048**，
所以 spec §5 的「长边上限 2048」在本实现里是断言而非独立裁剪步骤，
不存在"为塞进上限而非等比压缩"的路径。

### 1.6 tokenizer 现状实测

注册前 `len(tokenizer)=151669`（base vocab 151643 + 26 个 added token，最后一个是
`</think>`=151668）；4 个新 token 注册后为 151669–151672，`len=151673`。
embedding 矩阵 151936 行 ⇒ **词表比 tokenizer 长 263 行**，这是 Qwen 的对齐 padding。

---

## 二、假设清单（能自证的已当场证掉）

| # | 假设 | 处理 |
|---|---|---|
| A1 | gradient checkpointing 不会切断 merger 梯度 | **实测证掉**：GC 开/关两档 loss 相同、merger grad norm 3.082 vs 3.085。另已设 `use_reentrant=False` |
| A2 | 分段 token 化拼接 == 整串 token 化 | **实测断言**（`check_concat_equivalence`），并进单测 |
| A3 | `eval_loss` 就是 assistant-only loss | **实测证掉**：mock 端到端跑中 `eval_loss=4.554167` 与自算 `eval_seg_assistant_loss=4.554167` 逐位相同 |
| A4 | `save_total_limit` 只经 `_rotate_checkpoints` 删 checkpoint | **实测证伪**，见 PREFLIGHT_MODEL.md 末节；已修复 + 双路径单测 |
| A5 | padding 词表行可直接当新 token 用 | **实测证伪**：151669+ 各行近乎全同（L2 0.358–0.361），已重初始化 |
| A6 | 4 张最长图 + 2048 token 能放进 `4 x GAS4` | **实测证实**：42.18 GiB / 95.1 GiB |

---

## 三、待主 agent 决策（均已采保守默认继续，未静默拍板）

### D-1（已执行，可回滚）：新建 `/home/bc/envs/q3vl_sft` 而非改动共享 conda env

`llm_factory` 缺 deepspeed。两种做法都合理：直接 `pip install deepspeed` 进 `llm_factory`
（只新增 deepspeed/hjson/msgpack/py-cpuinfo，dry-run 确认不改动任何已有包），
或另建隔离环境。

**采用**：`python -m venv --system-site-packages /home/bc/envs/q3vl_sft`，只把
deepspeed 0.18.2 装进该 venv。**`llm_factory` 一个字节没动**，回滚 = `rm -rf` 该目录。
代价是多一个环境路径要记（已写进 launch 脚本与 config 快照）。
若主 agent 更希望统一进 `llm_factory`，一条 pip 命令即可。

### D-2（已采默认）：`model.visual.pos_embed` 归冻结侧

spec §2.2 表格逐字只写了「24 个 Vision blocks 及视觉 patch embedding」冻结，
未点名 `pos_embed`（2.36 M 参数）。官方开关语义 `tune_mm_vision=false` 的含义是
整个 vision tower 冻结、再单独打开 merger，据此 `pos_embed` 应冻结。

**采用**：冻结。影响面 2,359,296 / 4.44e9 = 0.053%。
若主 agent 认为应随 merger 一起训练，改 `constants.py::VISUAL_TRAINABLE_PREFIXES` 一行。

### D-3（**需决策**）：§8.3 生成类诊断在 ZeRO-3 下默认关闭

spec §8.3 要求辅助诊断含「标签完整率 / 固定顺序正确率 / 非空率 / 旧七标签残留率」，
这些必须**生成**才能算。代码已完整实现（`trainer.py::generation_diagnostics` +
`diagnostics.py::parse_two_segment`），单测覆盖解析与聚合。

问题：ZeRO-3 下 `model.generate` 需要 DeepSpeed 逐层 gather 参数，很慢，且**我无法在
不启动正式训练的前提下验证它**（任务卡禁止启动）。

**采用保守默认 `gen_diag_samples: 0`（关闭）**，异常已被捕获不会杀训练。
两个选项请主 agent 裁定：
- (a) 保持关闭，训练结束后对 protected checkpoint 离线批量生成算这些率 —— 更省、更稳，但拿不到训练中曲线；
- (b) 联合 preflight 时先在 2 卡 ZeRO-3 上验一次 `gen_diag_samples: 32`，通过再开。

### D-4（已采默认）：不缩词表，改为显式重初始化 4 行

spec §4.3 写「tokenizer 扩词表后同步 resize model embeddings」。字面执行
`resize_token_embeddings(151673)` 会把矩阵从 **151936 缩到 151673**——删掉 Qwen 自带的
对齐 padding，并让 vocab 变成非对齐的奇数，且这 4 个 id 本就已有行、并不需要扩容。

**采用**：`if len(tokenizer) > rows: resize` 否则保持 151936（永不缩表），
并按 HF `mean_resizing` 语义显式重初始化这 4 行（真实词表均值 + 逐维 std x 1e-3 噪声，
seed 固定 0），因为原 padding 行近乎全同（见 A5）。重初始化后最小两两距离 1.616e-3。
tied 契约在重初始化后重新断言通过。

### D-5（**需决策**）：`metric_for_best_model = eval_loss` 与 CLAUDE.md 红线冲突

- `CLAUDE.md` 红线速查：「**checkpoint 选择禁用 val loss**」；
- 本任务规格 §8.3：「主选择指标：`eval_loss`（仅 assistant target token）」。

两者直接冲突。红线成文在先，语境是渲染/LUT 类实验（val loss 与图像质量脱钩）；
本阶段是纯语言 SFT，assistant-only eval loss 是标准且被今日定档的规格显式冻结。

**采用保守默认**：配置 `metric_for_best_model: eval_loss` 但
**`load_best_model_at_end: false`**——即 eval_loss 只做记账（写进
`trainer_state.best_model_checkpoint`），**不自动挑选、不自动覆盖、不导致任何删除**
（protected checkpoint 已对两条删除路径免疫）。最终用哪个 checkpoint 进 Stage-Where
留给主 agent 显式决定。请主 agent 确认这个折中，或指定红线优先（则把
`metric_for_best_model` 置空）。

### D-6（**已收敛为已核实**）：与 S0-DATA 的 schema 对接

写代码时 `/mnt/nfs/bc/data/datasets/sft2seg-20260804/` 尚不存在、无 SCHEMA 文档，
故先按 spec §3.3 / METACANVAS §2.3 的契约字段实现 + 多布局兼容。**收尾时 S0-DATA 已落地
`q3vl/data/`，已逐字段核对其 `pipeline.py::stage_manifest` 的实际产物，结论：完全兼容**，
并补了 4 个用其**真实行形状**构造的 interop 单测（`TestS0DataInterop`）：

| 对接点 | 生产方实际产物 | 消费侧 |
|---|---|---|
| index 行布局 | 嵌套 `members`（`record` / `image` 两角色） | `_build_nested` ✓ |
| 成员字段 | `shard`(**绝对路径**)/`member`/`offset`/`length`/`size`/**`sha256`** | 别名表 `sha256→checksum`；绝对路径绕过 `shard_root` ✓ |
| checksum 形式 | 裸 hex（无 `sha256:` 前缀） | 按长度 64 判定为 sha256 ✓（并测了篡改必报错） |
| 记录字段 | `where` = region_scope 正文；`color` = 其余六段以 `\n` 连接 | `WHERE_KEYS`/`COLOR_KEYS` 命中 ✓ |
| 目标串拼装 | `<where>W</where><color>C</color>`，标签与正文间无换行 | 与 `build_target_text` **逐字符相同** ✓ |
| manifest | `counts.<split>.n_effective` + 顶层 `n_effective` + `digest` + `schema_version` | `TerminalManifest` 全部命中 ✓ |
| 文件位置 | `splits/<split>.index.jsonl`、`manifest/terminal_manifest.json` | 已写入 `configs/sft_base.yaml` ✓ |

一处已修正的口径：生产方**每个 split 一个 index 文件**，行内 `split` 字段是
`V_where` 这类名字而非 `"eval"`。入口原先硬编码按 `split="eval"` 过滤会把 eval 集滤空，
已改为默认不过滤（`train_split_name`/`eval_split_name` 仅在合并索引时才需要）。

保留的三层防御（仍然有效，用于未来 schema 漂移）：

1. **三种 index 布局自动识别**：嵌套 `members` 映射 / 每成员一行（按 `sample_id` 归并）/
   单成员扁平行（按扩展名推角色）；都不匹配时**报错并列出实际观察到的 key**，绝不静默降级；
2. **字段别名表** `shards.py::MEMBER_FIELD_ALIASES` 与 record 键别名
   （`instruction|prompt|...`、`where|where_text|...`、`color|color_text|...`）——
   名字对不上是改一行配置，不是打补丁；
3. **拒绝本地凑数**：七段→两段的重排属于 S0-DATA。若 record 只有 7 个 canonical 字段而无
   `where`/`color`，默认**直接报错**；本地拼装藏在 `allow_local_assembly=True` 后面，仅供 mock 单测用。
   理由：若在这里悄悄兜底，数据侧 schema 不一致会被掩盖成后期的质量退化。

### D-7（**需决策**）：Base SFT 的 eval 集用哪个 split

S0-DATA 按 METACANVAS §2.2 把 held-out 分成 **`V_where` / `V_what` / `T_final` / `T_lut_unseen`**
四个互斥集合，**没有**一个叫 `eval` 的集合。而 Base SFT 需要一个集合算 §8.3 的
assistant-only `eval_loss`。

硬约束：`T_final` 与 `T_lut_unseen` **绝对不能用**（METACANVAS §2.2：不参与任何模型/阈值选择）。
剩下 `V_where` / `V_what` / 两者并集。

**采用保守默认 `V_where`**，理由：两个 protected checkpoint 由**步数**（0.5/1.0 epoch）
预注册决定，与 eval_loss 无关；叠加 D-5 的 `load_best_model_at_end: false`，
本阶段 eval_loss 的选择权实际为零，因此只需避开测试集即可。选 `V_where` 而非并集，
是为了让 `V_what` 对 Base SFT 完全零接触，给后续 What 臂留最干净的选择集。

请主 agent 确认，或改用并集 / 指示 S0-DATA 另切一个 base-SFT 专用监控集。

---

## 四、纪律遵守记录

- **未启动正式训练**。GPU 仅用于：模型加载、单 batch 前反向 smoke、mock 数据的
  端到端闭环验证（24 样本 / 3 optimizer step / 单卡）。
- `launch_sft.sh` 内置 D-20 四步（先 `rm -f` 日志 → `ps -p $PID` 判活 → `tail` 验实质输出
  → 最后写 `job.marker`），**全脚本无 `pgrep`**。
- 期间确实被 noclobber 咬了一次（`head ... > 已存在文件` 报 `file exists`），当场查清、
  确认产物正确后才继续，未当噪声放过。
- 只新建 `q3vl/`（此前不存在）与本交付目录，**未改动工作区任何既有文件**，
  未触碰并行任务的 `q3vl/data/`。
- 临时产物写在 `/home/bc/data/tmp/q3vl_it`（scratchpad 所在卷曾被 9 GiB 级 checkpoint 写满，
  已改道大卷并清理）。

## 五、交付清单

```
q3vl/train/
  constants.py      # spec 冻结常量（token、图像、长度、batch、冻结前缀）
  freeze.py         # Arm B 冻结规则 + 结构断言 + 清单渲染
  tokens.py         # special token 注册/单 token 校验/不缩表 resize/重初始化/reload
  imageproc.py      # spec §5 几何（EXIF/短边512/32对齐/4:1过滤）+ 与 processor 交叉断言
  shards.py         # indexed-tar-shard 随机读 + checksum + 三布局 index + terminal manifest
  dataset.py        # 两段式记录读取（拒绝本地凑数）
  collator.py       # 占位符展开 + loss mask + 分段标记 + 超长报错不截断
  diagnostics.py    # 分段 token loss/acc（仅诊断）+ §8.3 生成结构率
  trainer.py        # protected checkpoint（覆盖两条删除路径）+ 分段诊断
  args.py           # 规格冻结超参 + 启动期断言
  modeling.py       # shard 完整性校验 + 模型/processor 构建
  train_sft.py      # 训练入口
  preflight_model.py# §9 项 1-4 + 项 10/11
  mock_shards.py    # 单测用 indexed tar shard 生成
  configs/          # sft_base.yaml + ds_zero3.json
  scripts/          # launch_sft.sh（D-20）
  tests/            # 68 passed
experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/model/
  PREFLIGHT_MODEL.md, preflight_model.json, NOTES.md, config/
```
