# genctx 阻塞诊断：左填充 batch 生成 vs B=1 贪心

日期：2026-08-05 ｜ checkpoint-4976 ｜ GPU0 ｜ 代码 `diag_batch_consistency.py` / `diag_segment_impact.py`
产物 `batch_consistency_trimmed.json` / `segment_impact.json` / `run*.log`

## 结论（一句话）

**(b) 数值边缘，且 mrope 被排除。** 每一次分歧处 top-1 与 top-2 的 logit 差是
**0 或 0.125（= 该量级下 bf16 的 1 个 ulp）**，而 batch 带来的扰动也正好 ~1 ulp ——
模型本身在那一步就是平票，两种配置把同一枚硬币掷向了不同面。
**genctx 可以按 B=64 开跑**；Where-B 真正消费的 `where_ids` 在 B=8 下 **100%** 与 B=1 逐位相同，
B=32 下 **96.9%**（唯一一条差异样本 token-F1 0.977）。

## 证据链

| 测试 | 结果 | 排除了什么 |
|---|---|---|
| T1 同一样本 ×4 | 4 行**逐位相同** | batch 机制本身没有 bug |
| T2 prefill logits（无生成） | pad=0 → max\|Δ\| **恰为 0.0**；pad=13 → 0.0；pad=71 → 0.234；pad=5 → 0.469。而 top1−top2 gap = **8～10** | 扰动只有几个 ulp，prefill 阶段**不可能翻转** argmax |
| **T3 padding 对照（判别性实验）** | 同一样本作**零填充**行 → 与 B=1 逐位相同；作**225 token 填充**行 → **同样逐位相同** | **mrope/position 正确**。若左填充下位置算错，225 个 pad 不可能复现 B=1 |
| T5b 零填充 batch（三条 prompt 均 456） | 仍然分歧（13、105 处） | **padding 不是原因**，batch 形状才是 |
| T4 逐 token | 每处分歧 `top1−top2 gap ∈ {0.0, 0.125}`，`max\|Δlogit\| ∈ [0.125, 0.156]` | 纯平票四舍五入 |

机理：bf16 GEMM 在不同 batch 形状下归约顺序不同 → logit 扰动 ~1 ulp → 模型平票处翻转 →
贪心把一次翻转放大成不同的后缀。**与左填充、与 Qwen3-VL 的 mrope 都无关**。

> 读源码的旁证（transformers 4.57.1 `modeling_qwen3_vl.py`）：`get_rope_index` 用
> `mrope_position_deltas = llm_positions.max()+1 - len(total_input_ids[i])`，其中
> `total_input_ids[i]` 是**含 padding 的整行**；解码时 `position = cache_position[0] + delta`，
> 首个解码步 `cache_position[0] = width`，恰好抵消。位置算术本身是对的——T3 是实证。

## 一处必须说明的自我更正

我第一版诊断复现出 ~1/4，但那是**我自己的比较有缺陷**：`generate` 会跑到 batch 里
**最后一行**结束，先结束的行后面跟着 pad/eos 填充；不截断就是在比填充。
按 EOS 截断后（这也正是 `build_record` 实际存的东西），T3 的「153 处分歧」变成 `null`（逐位相同）。
**原验证作业的 `generate_batch` 是有截断的**，所以它报的 1/4 是真实测量——
只不过「整条序列逐位相等」本身不是正确的验收口径。

## 影响（真正该看的数字，n=32，max_new_tokens=512）

| | B=8 | B=32 |
|---|---:|---:|
| 整条序列逐位相同 | 0.000 | 0.000 |
| **`where_ids` 逐位相同** | **1.0000** | **0.9688** |
| `color_ids` 逐位相同 | 0.2500 | 0.2812 |
| 分歧落在 where 段内 | **0** | **1** |
| 分歧只落在 color 段 | 24 | 22 |
| color token-F1（差异样本）p50 | 0.902 | 0.897 |
| 七项结构率（tag/order/nonempty/legacy/两段 format_failure） | **与 B=1 完全相同，全部满分** | **同左** |
| 相对 B=1 加速 | 5.4× | **16.6×** |

为什么 where 段几乎不受影响：它只有 ~41 token 且模型很确定（训练期 `where_acc` 0.913），
平票机会少；color 段 ~178 token 且是六段自由文本（`color_acc` 0.757），平票机会多得多。

## 建议口径（待主 agent 裁定）

**「genwhere/2 = 记录在案的 batch 配置下的贪心」在科学上可接受**，理由：

1. 产物是模型的**真实样本**，不是近似——每一步仍取 argmax；
2. **七项结构率与 B=1 完全一致且满分**，格式契约不受影响；
3. Where-B 的实际输入 `where_ids` 96.9–100% 逐位相同；
4. 差异只发生在模型**自己无所谓**的位置（gap ≤ 1 ulp）——把其中一面叫做"正确"没有依据；
5. B=1 需要 400+ 小时，不可行；「唯一可行的口径」也是一种科学事实。

**已落地的配套**：`build_record` 的 `gen` 块现在记录 `batch_size`
（`q3vl/whereb/gencontext.py`，单测 `test_record_declares_the_batch_size_it_was_produced_under`）——
贪心只在**给定 batch 形状**下确定，要逐位复现就必须知道当时的形状，所以让产物自述而不是留作隐含前提。

**必须一并说清的可复现性边界**：双卡分片 + B=64 时，每个 shard 的**最后一个不满批**形状不同；
只要分片边界与批次顺序不变（按索引确定），重跑即可复现。改分片数或改 batch size **会**改变 color 段。

## 等长分桶（顺带验证的结论：不是解药）

prompt 长度 313–538，48 条里有 **33 个不同长度**，最大同长组只有 3 条 →
**凑不出零填充的 B=64**。按长度排序可把 padding 从 5.3%→1.7%（B=4）、6.2%→3.7%（B=8）。
但 T5b 已证明**零填充也不能带来逐位一致**，所以分桶只是小幅吞吐优化，不能解决一致性。
考虑到 B=32 已有 16.6× 加速，不建议为此增加分片复杂度。

## 对 genctx 墙钟的影响

无负面影响：诊断支持按原计划 B=64 开跑。B=32 实测 16.6×，B=64 只会更快（显存充裕，
B=32 峰值远低于 97 GiB）。EXEC-2 估的 4.5–7h 前提成立。
