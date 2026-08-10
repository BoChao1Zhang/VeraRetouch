# S0-TRAIN 收尾 · protected checkpoint 验证

## 要验证的结论
如果这个验证通过，我们就能说 **base SFT 的两个 protected checkpoint 真的学会了两段式`<where>`/`<color>` 输出格式**（标签完整、顺序固定、两段非空、旧七标签不再泄漏），后续 Where/What 各臂可以把它当作可用的读出起点；不通过就不能说 —— 训练 loss 下降只证明 teacher forcing 下的 token 概率，不证明自由生成时格式成立。

## 为什么需要验证它
主 agent 裁定 D-J3 把训练内的生成诊断关掉了（ZeRO-3 `generate` 未验证），spec 8.3 的结构率因此没有任何在线证据。整条 METACANVAS 链路（Where-B 的 `s` 读出、What 的配色）都建立在「模型自由生成时输出这两段」之上；不补这一步，审稿人问「你们的两段格式在推理时成立吗」时我们只有训练 loss 可拿。

## 怎么验的
离线单卡（无 DeepSpeed）加载两个 protected checkpoint，在 V_where 验证集上取**同一批** 64 条（seed=42）做贪心生成（左填充 + KV cache，max_new_tokens=448），跑 `parse_two_segment` 结构解析 + 与 GT 的粗略 token 对齐；eval_loss 直接读checkpoint 自带的 `trainer_state.json`，不重跑 eval。

---

## 1. 交付物与环境

- 生成时间：`2026-08-05T11:49:54+08:00`　主机：`h3c`
- git commit：`80a507892f1c5675da277b4a21ebe4815566cd8f`
- 环境：python `3.12.12` / torch `2.10.0+cu128` / transformers `4.57.1`
- 训练 run：`/home/bc/data/runs/q3vl_base_sft_20260804`
- 评测源：`/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits/V_where.index.jsonl`（split 代号 **V_where**，共 896 条，抽样 64 条）
- 采样的 dataset 下标（两个 checkpoint 完全相同）：`[6, 25, 27, 30, 32, 44, 80, 89]...`（共 64 个）

## 2. checkpoint 完整性

| 项 | checkpoint-2488 | checkpoint-4976 |
|---|---|---|
| 整体完整 | yes | yes |
| 权重分片与 index 一致 | yes | yes |
| 权重字节数 | 8875719328 | 8875719328 |
| 缺失文件 | （无） | （无） |
| global_step | 2488 | 4976 |
| epoch | 0.5001 | 1.0000 |

## 3. assistant-only eval_loss（book-keeping，**不得用于 checkpoint 选择**）

> CLAUDE.md 红线速查：**checkpoint 选择禁用 val loss**。下表只是把训练期已经算过的数字汇总在一处；本报告不据此推荐任何 checkpoint。选择依据请用第 4 节的结构率与下游任务指标。

| 指标 | checkpoint-2488 | checkpoint-4976 |
|---|---|---|
| `eval_loss` | 0.705572 | 0.664593 |
| `eval_seg_assistant_loss` | 0.703816 | 0.663505 |
| `eval_seg_assistant_acc` | 0.764922 | 0.776213 |
| `eval_seg_where_loss` | 0.257214 | 0.245148 |
| `eval_seg_where_acc` | 0.911091 | 0.912769 |
| `eval_seg_color_loss` | 0.765787 | 0.721603 |
| `eval_seg_color_acc` | 0.744589 | 0.757172 |
| `eval_seg_eos_loss` | 0.000034 | 0.000001 |
| `eval_seg_eos_acc` | 1.000000 | 1.000000 |

完整 eval 曲线（step → eval_loss）：

| step | epoch | eval_loss | assistant_acc | where_acc | color_acc |
|---|---|---|---|---|---|
| 500 | 0.1005 | 0.821242 | 0.7374 | 0.9033 | 0.7143 |
| 1000 | 0.2010 | 0.775346 | 0.7479 | 0.9013 | 0.7265 |
| 1500 | 0.3015 | 0.743435 | 0.7542 | 0.9070 | 0.7329 |
| 2000 | 0.4020 | 0.721686 | 0.7609 | 0.9083 | 0.7404 |
| 2488 | 0.5001 | 0.705572 | 0.7649 | 0.9111 | 0.7446 |
| 2500 | 0.5025 | 0.705965 | 0.7656 | 0.9100 | 0.7455 |
| 3000 | 0.6030 | 0.689796 | 0.7694 | 0.9105 | 0.7498 |
| 3500 | 0.7034 | 0.679215 | 0.7719 | 0.9098 | 0.7527 |
| 4000 | 0.8039 | 0.670233 | 0.7750 | 0.9122 | 0.7559 |
| 4500 | 0.9044 | 0.665426 | 0.7759 | 0.9124 | 0.7569 |
| 4976 | 1.0000 | 0.664593 | 0.7762 | 0.9128 | 0.7572 |

## 4. 离线生成诊断（spec 8.3 结构率）

预注册判据来自 spec 8.3 的结构性要求；这里并排给出实测。

| 指标 | 期望 | checkpoint-2488 | checkpoint-4976 |
|---|---|---|---|
| `tag_completeness` | = 1.00 | 1.0000 | 1.0000 |
| `order_accuracy` | = 1.00 | 1.0000 | 1.0000 |
| `where_nonempty_rate` | = 1.00 | 1.0000 | 1.0000 |
| `color_nonempty_rate` | = 1.00 | 1.0000 | 1.0000 |
| `legacy_tag_leak_rate` | = 0.00 | 0.0000 | 0.0000 |
| `duplicate_where_rate` | = 0.00 | 0.0000 | 0.0000 |
| `duplicate_color_rate` | = 0.00 | 0.0000 | 0.0000 |
| `where_copied_into_color_rate` | ≈ 0 | 0.0000 | 0.0000 |
| `eos_terminated_rate` | = 1.00 | 1.0000 | 1.0000 |
| `truncated_rate` | = 0.00 | 0.0000 | 0.0000 |

生成量与耗时：

| 项 | checkpoint-2488 | checkpoint-4976 |
|---|---|---|
| 生成 token 数 p50 | 200.0000 | 194.0000 |
| 生成 token 数 max | 253.0000 | 272.0000 |
| prompt token 数 p50 | 459.0000 | 459.0000 |
| batch size | 1 | 1 |
| 左填充一致性 | 0.2500 | 0.2500 |
| 模型加载秒 | 134.5000 | 12.3000 |
| 生成秒 | 635.6000 | 639.7000 |
| 显存峰值 GiB | 8.8000 | 8.8000 |

## 5. 与 GT 的粗略对齐（**不是 headline 指标**）

token 重叠只用来回答「生成的两段是不是在讲同一件事」，不能当质量指标：颜色段是六个固定 body 的拼接，所以行数统计比 F1 更有信息量。

| 指标 | checkpoint-2488 | checkpoint-4976 |
|---|---|---|
| where token-F1 p50 | 1.0000 | 1.0000 |
| where 完全一致率 | 0.5312 | 0.5312 |
| color token-F1 p50 | 0.6384 | 0.6324 |
| color 逐行 token-F1 p50 | 0.5091 | 0.5000 |
| color 预测行数 p50 | 6.0000 | 6.0000 |
| color 预测恰为 6 行的比例 | 1.0000 | 1.0000 |
| color GT 恰为 6 行的比例 | 1.0000 | 1.0000 |
| color 行数与 GT 一致率 | 1.0000 | 1.0000 |
| color 长度比 p50 | 0.9530 | 0.9539 |

## 6. 结论

- **checkpoint-2488**：结构率全部达标 = **是**；EOS 终止率 1.0000；截断率 0.0000。
- **checkpoint-4976**：结构率全部达标 = **是**；EOS 终止率 1.0000；截断率 0.0000。

> 本报告**不做 checkpoint 选择**：红线禁止用 val loss 选 checkpoint，而结构率若两档都满分则不构成区分度。选择应由主 agent 结合下游（Where-B / What）指标裁定。

## 7. 机器可读产物

- `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/checkpoint_verification.json`
- `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft/checkpoint_verification_samples.jsonl`（逐样本生成文本 + 解析结果）

