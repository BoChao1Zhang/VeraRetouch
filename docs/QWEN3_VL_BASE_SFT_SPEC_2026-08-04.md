# Qwen3-VL Base SFT 定档规格（2026-08-04）

> 状态：**DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED**  
> 版本：v1.0  
> 作用：冻结进入 Stage-Where / Stage-What 之前的 base-model SFT 训练口径。  
> 边界：本文不是运行记录，不表示数据转换、训练代码、checkpoint 或实验结果已经产生。

---

## 1. 目标与非目标

本阶段只做一件事：让 `Qwen3-VL-4B-Instruct` 在 VeraRetouch 的 global + local 指令数据上学习统一的两段式推理协议：

```text
I_in + instruction -> <where>...</where><color>...</color>
```

SFT checkpoint 是后续两阶段训练的共同基模：

1. Stage-Where 学习空间参数与 readout；
2. Stage-What 在 Where 条件下学习颜色变换和 LUT。

本阶段明确**不做**：

- 不训练 `Q_where`、`Q_color` 或 MetaCanvas；
- 不预测 mask、basis 系数、GLUT 参数或最终渲染图；
- 不使用 `I_tar` 作为模型输入；
- 不做 Arm A（全 Vision + Language 联合微调）；
- 不使用 LoRA、QLoRA、adapter-only 或其他参数高效微调；
- 不启动 Stage-Where / Stage-What 的联合训练。

---

## 2. 基模与可训练边界

### 2.1 基模

```text
model: Qwen3-VL-4B-Instruct
local_path: /home/bc/data/models/Qwen3-VL-4B-Instruct
```

运行前必须验证权重索引中的全部 shard 已完整落盘并通过大小/checksum 检查。当前目录出现过 `.aria2` 下载状态文件，不能仅凭目标文件名存在就认定模型下载完整。

### 2.2 唯一训练臂：官方 Arm B

采用 Qwen3-VL 官方 4B SFT 的模块边界：

| 模块 | 状态 |
|---|---|
| 24 个 Vision Transformer blocks 及视觉 patch embedding | 冻结 |
| 主 `vision.merger` | 全参数训练 |
| 3 个 `deepstack_merger_list` | 全参数训练 |
| Language model | 全参数训练 |
| 新增 special-token embeddings / tied output weights | 全参数训练 |

对应官方开关语义：

```text
tune_mm_vision = false
tune_mm_mlp    = true
tune_mm_llm    = true
```

注意：在 Transformers 的 Qwen3-VL 模块树中，merger 位于 `model.visual` 内。实现时不能用一次粗粒度的 `model.visual.requires_grad_(False)` 冻结整个子树，否则会把本应训练的主 merger 和 3 个 deepstack merger 一起冻结。必须按参数名逐组核对 trainable/frozen 清单和参数量。

这里的“全量微调”指 merger 与 Language 范围内的**全部参数**参与优化，不是 LoRA；Vision blocks 的冻结是本实验唯一预注册的结构边界。

---

## 3. 数据范围与采样

### 3.1 数据范围

使用全部已确认的 VeraRetouch SFT build：

```text
global: g1, g2, g3, g4
local:  l1, l2, l3, l4, l5, l6
```

split authority：

```text
/mnt/nfs/bc/data/datasets/sft/splits-20260803/train_sft_ids.txt
/mnt/nfs/bc/data/datasets/sft/splits-20260803/eval_sft_ids.txt
/mnt/nfs/bc/data/datasets/sft/splits-20260803/dedup_drop_sft_ids.txt
```

定档时的原始 split 数量：

```text
train: 169,260
eval:    3,320
```

实际训练样本数 `N_effective` 必须在图像、标签和长度验证全部完成后重新计算，并写入 terminal manifest；不能把上述原始数量当作过滤后的最终数量。

### 3.2 采样策略

- global 与 local 合并后做全局 shuffle；
- 每个合格样本在一个 epoch 内恰好出现一次；
- 不重采样，不额外强制 1:1 配平；
- 定档时原始比例约为 global 52.5%、local 47.5%；
- train/eval 按冻结的 `sft_id` split 读取，不按目录、mtime 或临时队列重新划分。

### 3.3 存储与读取契约

训练读取必须服从仓库的冷热分层与 indexed-tar-shard 契约：

- durable 样本和 terminal manifest 位于 NFS；
- 通过 index 的 `shard/member/offset/length/size/checksum` 随机定位成员；
- 不把全量图片解包成长期小文件目录；
- 本地仅允许容量有上限、可重建的 scratch/cache；
- 新建或重建数据视图时，partial shard 不得进入 terminal manifest；
- resume authority 来自 manifest 与 shard index，不来自目录 mtime。

两段式 reasoning 可以由已发布记录做确定性转换，但实现必须给出版本化 schema 和校验摘要；不得悄悄依赖一套未登记的小文件副本。

---

## 4. 输入、目标与两段式 reasoning

### 4.1 模型输入

```text
user input = I_in + instruction
```

- `instruction` 使用现有完整指令，不重新生成；
- `I_tar`、GT LUT、mask、recipe 和任何由目标图才能知道的信息都不进入输入；
- assistant target 使用现有 VeraRetouch reasoning 的确定性重排，不重新标注、不摘要、不改写正文。

### 4.2 原七段到两段的映射

现有真实 token 顺序为：

```text
problem_light
problem_globalcolor
problem_specificcolor
region_scope
plan_light
plan_globalcolor
plan_specificcolor
```

转换后的固定顺序为：

```text
<where>
原 region_scope 正文
</where>
<color>
原 problem_light 正文
原 problem_globalcolor 正文
原 problem_specificcolor 正文
原 plan_light 正文
原 plan_globalcolor 正文
原 plan_specificcolor 正文
原收束文本（若存在）
</color>
```

约束：

- `where` 必须在 `color` 前，使后续颜色推理显式依赖空间判断；
- 删除七种旧 start/end 标签，但保留六段非空间正文的原始先问题、后方案顺序；
- 不把 `region_scope` 正文复制进 `color`；
- 不补写原文中不存在的收束文本；
- 任一必需旧段缺失、重复、乱序或无法闭合时，样本进入 rejection report，不做猜测性修复。

代码字段名以当前 canonical parser 为准：`problem_lighting`、`problem_global_color`、`problem_specific_color`、`region_scope`、`plan_lighting`、`plan_global_color`、`plan_specific_color`。历史文档中的 `texture` 是过时简称，不能据此匹配真实 special token。

### 4.3 四个新 special token

注册为 tokenizer 的 `additional_special_tokens`：

```text
<where>
</where>
<color>
</color>
```

要求：

- 四个字符串各自编码为一个不可拆分 token；
- tokenizer 扩词表后同步 resize model embeddings；
- Qwen3-VL-4B 的输入 embedding 与输出头共享时，保持 tied-weight 契约；
- 四个 token 及两段正文均参与 assistant causal-SFT loss；
- tokenizer、special-token map、model config 必须与每个 checkpoint 一起保存；
- reload 后重新验证四个 token 的 ID 与单 token 编码性质。

### 4.4 Loss mask

采用标准 causal SFT：

- system/user/chat-template/image-placeholder token 的 label 为 ignore index；
- 只监督 assistant 的 `<where>...</where><color>...</color>`；
- 不增加单独的 Where loss、Color loss、mask loss、LUT loss 或渲染 loss；
- 记录 Where 与 Color 各自的 token loss 只用于诊断，不改变总 loss 权重。

---

## 5. 图像预处理

图像不是 `512x512`。冻结的预处理契约为：

1. 正确处理 EXIF orientation；
2. 等比例缩放，使短边为 512；
3. 不做正方形拉伸，不改变长宽比，不做 center crop；
4. Qwen3-VL 使用 `patch_size=16`、`spatial_merge_size=2`，输入高宽按 factor 32 对齐；
5. 长边上限为 2048；
6. 原始宽高比超过 4:1 的极端图直接过滤，不通过非等比压缩硬塞进上限；
7. 对齐后的实际高宽、`image_grid_thw`、视觉 token 数写入样本/批次诊断。

不能只设置 `max_pixels=262144` 来实现该契约。Qwen 的 `min_pixels/max_pixels` 是面积约束，不能保证宽图的短边为 512。实现必须显式验证预处理后的短边、比例误差和 32 对齐。

在该契约下，单图 LLM-facing 视觉 token 网格约为：

```text
(H / 32) x (W / 32)
```

例如 `512x512 -> 16x16 = 256 tokens`，`512x1024 -> 16x32 = 512 tokens`，长边上限处约为 1024 tokens。

---

## 6. 序列长度与拒绝策略

```text
model_max_length = 2048
```

- 长度统计必须基于最终 chat template、图像 placeholder 和两段式 target 的完整序列；
- 超过 2048 的样本直接过滤并进入 rejection report；
- 不截断 assistant target，不允许产生缺失 `</where>` 或 `</color>` 的监督样本；
- 训练前报告按原因分组的过滤数：图像缺失/损坏、宽高比、旧标签异常、序列超长、split 不一致、index/checksum 失败。

---

## 7. 优化与分布式训练配置

### 7.1 官方 Arm B 优化器口径

```yaml
optimizer: adamw_torch
learning_rate: 1.0e-5
weight_decay: 0.0
adam_beta1: 0.9
adam_beta2: 0.999
adam_epsilon: 1.0e-8
warmup_ratio: 0.03
lr_scheduler_type: cosine
max_grad_norm: 1.0
precision: bf16
```

merger 与 Language 使用相同学习率，不使用 vision/merger 分层学习率。

### 7.2 两卡配置

```yaml
world_size: 2
distributed: DeepSpeed ZeRO-3
attention: flash_attention_2
gradient_checkpointing: true
global_batch_size: 32
preferred_per_device_train_batch_size: 4
preferred_gradient_accumulation_steps: 4
```

启动正式训练前允许做一次单 batch 显存探测。若极端长图导致 OOM，只允许改为：

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

有效 global batch 必须保持 32。该探测是运行可行性校准，不是新增实验臂。

---

## 8. 训练长度、评测与 checkpoint

### 8.1 训练长度

```text
num_train_epochs = 1.0
```

必须保护两个里程碑 checkpoint：

- 0.5 epoch；
- 1.0 epoch。

按过滤前 `169,260 / global_batch 32` 估算：

```text
0.5 epoch: ~2,645 optimizer steps
1.0 epoch: ~5,290 optimizer steps
```

这两个数字只是定档时估算。正式 step 必须依据 terminal manifest 中的 `N_effective` 计算：

```text
steps_per_epoch = ceil(N_effective / 32)
half_epoch_step = round(0.5 * steps_per_epoch)
full_epoch_step = steps_per_epoch
```

不能在过滤完成前把 2645/5290 硬编码为 checkpoint authority。

### 8.2 普通评测与保存

```yaml
eval_steps: 500
save_steps: 500
save_total_limit: 3
```

0.5/1.0 epoch checkpoint 不受普通滚动清理影响。每个 protected checkpoint 必须带 tokenizer、processor、special-token map、训练状态和数据 manifest digest。

### 8.3 选择指标

主选择指标：

```text
eval_loss（仅 assistant target token）
```

辅助诊断：

- assistant token accuracy；
- `<where>...</where>` 与 `<color>...</color>` 标签完整率；
- 固定顺序正确率；
- Where/Color 非空率；
- Where token loss 与 Color token loss；
- 生成抽检中的旧七标签残留率；
- image/instruction shuffle 负控制只作为表征诊断，不用于 SFT loss。

自由文本 reasoning 不使用 exact match 作为主 checkpoint 指标。

---

## 9. 运行前强制校验

在实现完成后、正式训练启动前，必须先产生一份 preflight 报告，至少包含：

1. 模型 shard 完整性与本地 Transformers 版本；
2. frozen/trainable 参数名、参数量和比例；
3. 证明 Vision blocks 无梯度、4 个 merger 与 Language 有梯度；
4. 四个新 special token 的 ID、单 token 编码与 reload 一致性；
5. 七段到两段转换的全量计数、拒绝原因和抽样原文对照；
6. train/eval/dedup 三集合无交集；
7. indexed-shard 随机读取、checksum 和 resume 校验；
8. 图像短边、长边、宽高比、32 对齐和视觉 token 数分布；
9. 完整序列长度分布与 `>2048` 过滤数；
10. 单 batch 前后显存、吞吐、loss 有限性和梯度有限性；
11. global batch 32 的最终 micro-batch/GAS 组合；
12. NFS durable 输出路径、scratch 容量上限、日志与 `job.marker` 路径。

任一项失败时不得把 smoke 结果升级为正式训练。

---

## 10. 与后续两阶段训练的接口边界

Base SFT 只保证模型生成两段式语言 reasoning。它不自动证明：

- `<where>` hidden state 已经能预测高质量空间参数；
- `<color>` hidden state 已经能预测 33^3 LUT；
- Where 与 Color 已经实现因果解耦；
- Qwen3-VL 的视觉 basis 跨图具有稳定语义身份；
- 最终 renderer 已经可导出或达到 E2/RD-G 上界。

Stage-Where 与 Stage-What 必须另行冻结接口、监督目标、loss、参数边界和 gate。本文不替它们做决定。

---

## 11. 已冻结决策摘要

| 项目 | 决策 |
|---|---|
| 基模 | Qwen3-VL-4B-Instruct |
| SFT 臂 | 仅官方 Arm B |
| Vision | 冻结 |
| Merger + Language | 全参数训练 |
| PEFT | 不使用 |
| 数据 | g1-g4 + l1-l6，全局 shuffle，不重采样 |
| 输入 | `I_in + instruction` |
| 输出 | `where` 后接 `color` |
| reasoning | 原七段正文确定性重排，不重写 |
| special tokens | `<where>`, `</where>`, `<color>`, `</color>` |
| 图像 | 短边 512、等比例、32 对齐、长边最多 2048、>4:1 过滤 |
| 最大序列 | 2048；超长过滤，不截断 target |
| 优化器 | 官方 AdamW 口径，LR 1e-5，WD 0，3% warmup，cosine |
| 分布式 | 2 GPU，ZeRO-3，BF16，FA2，gradient checkpointing |
| batch | global 32；优先 4/GPU x GAS4，OOM 时 2/GPU x GAS8 |
| 时长 | 1 epoch，保护 0.5/1.0 checkpoint |
| 主选择指标 | assistant-only eval loss |
| 当前执行状态 | 未实现、未启动 |

---

## 12. 依据

- Qwen3-VL 官方 4B SFT：`qwen-vl-finetune/scripts/sft_qwen3_4b.sh`，其公开配置为 `tune_mm_vision=False`、`tune_mm_mlp=True`、`tune_mm_llm=True`、BF16、ZeRO-3、LR `1e-5`、WD `0`、warmup `0.03`、cosine、clip `1`、gradient checkpointing。
- Qwen3-VL Transformers 实现：主 merger 与 3 个 deepstack merger 输出 2560 维 LLM-facing 视觉 token；本 checkpoint 的 vision 配置为 24 blocks、hidden 1024、patch 16、merge 2、deepstack vision indexes `[5, 11, 17]`。
- 本仓库 canonical reasoning token 与字段映射：`dataset_build/src/construct/responses.py`。
- 数据字段与历史七段契约：`docs/_archive/2026-08-10/DATA_ASSETS_2026-08-02.md`；其中 `texture` 简称已由本文按真实 token 修正为 specific color。
- 当前实验事实入口：`docs/_archive/2026-08-10/EXPERIMENT_REGISTRY.md`。本文只冻结新 SFT 设计，不修改历史实验状态。

