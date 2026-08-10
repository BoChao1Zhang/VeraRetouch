# MCQ-L：全量 Local L1–L6 MetaCanvas 端到端训练

## 问题

验证 `I_in + instruction` 条件下，MetaCanvas 是否能同时给出：

1. instruction-conditioned 空间场（WHERE）；
2. 4D Gaussian GLUT 参数（WHAT）；
3. 经可导 `render4d` 后逼近真实 `I_tar`，同时保持 mask 外区域。

这不是 oracle-mask RD-G Stage-2。模型输入不含 `I_tar` 或 `.cgt`；`.cgt` 只作为训练标签与评测真值。

## 数据

- 六个 durable SFT build：`prod-l1..l6-local17k-*`。
- 只保留 `winner_confidence == normal`、`task_type == local`、有 instruction/source/target/cgt 的记录。
- split 权威仅为 `tools/data_splits/splits.sqlite3`。
- sweep 使用 S-train 内按 `source_id` 冻结出的 `fit/select`，两个 pool source-disjoint。
- 锁定配置后使用完整 S-train（`fit + select`）重训；S-val/S-test 不参与配置选择。
- 所有图像均通过 indexed tar ranged read；源图执行 `EXIF transpose -> LANCZOS 短边 1024`，再与归档 target/cgt 对齐。
- 派生 manifest v2 发布到 NFS durable root，带 JSONL digest、显式
  `shard/member/logical_path/offset/offset_data/length/size/checksum` SQLite index、
  源 batch manifest digest 和抽样 checksum/decode/alignment 验证。

## 模型

- VeraRetouch VLM + LoRA r32。
- 256 个 16x16 MetaCanvas query，读取 L11/L17/L23。
- 384-d connector，8 heads，FFN 1536，zero-init residual。
- spatial head：16x16 convolutional mask logits。
- parameter head：48 个 4D anchored Gaussian primitives + global affine；复用 RD-G `ParamHead4D/render4d`。

## 配置选择

两臂只改变同一结构的学习率尺度，保持 seed、样本顺序、batch、loss、steps、LoRA rank 完全相同：

| config | main lr | LoRA lr | warmup | wd |
|---|---:|---:|---:|---:|
| A | 2e-4 | 2e-5 | 200 | 0.05 |
| B | 1e-4 | 1e-5 | 200 | 0.05 |

两臂均训练 6,000 step、batch 8。`fit` 一次完整遍历为 4,671 个 full batches，
因此该预算覆盖全部 fit 记录后再进入第二个确定性 epoch；锁定配置后的完整
S-train 一次遍历为 5,199 个 full batches，同样保证全量覆盖。

checkpoint 不按 val loss 选择。训练中的 periodic checkpoint 使用 source-unique、
L1–L6 等额的 384 条 `select-core`；selection 依据为 ΔE00 p50/p90、mask
内/边界/外 PSNR、outside leakage、`delta_const`、`delta_shuffle` 和
AUC/AUC-shuffle，并对低于 3 dB 的 const/shuffle 施加 gate penalty；IoU 只作描述统计。
两套配置的最终选择使用完整 4,225 条 select，而不是 select-core。锁定配置后
用完整 S-train 重训，并只在 S-val 选 checkpoint，S-test 仅作终局一次评测。

## 训练损失

- 分层像素（mask 内 / 边界 / 外）Charbonnier reconstruction；
- `.cgt` BCE（不优化 IoU）；
- mask 外 preservation；
- 空间场 L1 sparse；
- renderer raw parameter prior；
- layer-route entropy floor；
- instruction condition dropout `p=0.15`。

## 交付

- 原子、可恢复 `latest.pt`（模型 trainable state、optimizer、scheduler、RNG、epoch/sample offset、manifest/config digest）。
- 非 loss 规则选出的 `best.pt`。
- 中文 `REPORT.md`、per-sample metrics。
- 最佳两个 checkpoint 的固定 20 样本联图。
- 最终 MetaCanvas memory 与 renderer MetaQuery feature 可视化。

## 全量结构矩阵

后续 renderer、条件输入和 MetaCanvas 读入/读出比较以
`EXPERIMENT_SCHEDULE.md` / `schedule.json` 为权威。所有 arm 均为相同 manifest、
seed、样本顺序和 6000-step 的 full run，不做低成本筛选；`run_full_matrix.py`
按每波两卡执行，并在终局自动生成中文报告与最佳两个结构的 20 样本/feature 可视化。
