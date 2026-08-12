# PR-AMORT 长尾与分层归因（WEVAL 口径）

- split `V_where`，主上下文 `generated`，共同样本 **400**
- 口径：matched-area top-k IoU; normal_only tables are the reporting convention (winner_confidence=='normal')
- 深尾定义：各臂自己最差 **10%**（n=40 / 臂）
- 六维几何分类来自 `q3vl.whereb.analysis.taxonomy`，**只看 GT 几何、不看预测**，因此类别标签不可能是被评分对象的函数。

## 0. 两条上界（口径分别声明，不可混用、不可相加）

| 上界 | 适用臂 | 度量尺度 | 定义 | n | 中位 |
|---|---|---|---|---|---|
| **链路上界 chain** | P1, P3prime | delivery resolution | GT -> area_resize(H/16) -> guided_upsample -> top-k -> IoU vs delivery GT | 400 | **0.9875** |
| **基底上界 basis** | P1 | H/16 grid (same as the board) | published Where-A oracle latent decoded at H/16 | 400 | **0.9890** |

> P3' never touches Phi-71, so this bound does not apply to it and is not reported for it

**判读**：两条上界的中位都在 **0.99 附近**——`H/16 + guided upsample` 这条链路、以及 Phi-71 基底的可达性，**都不是当前的瓶颈**。这与 E5「D 能忠实复现交给它的任何场」同向，并且把它从「解码保真」推广到了「上界不设限」：**尾部不是被表达能力卡住的**。

## 1. 六维几何分层（normal-only）

### area

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| large | 92 | 0.756 | 0.742 | -0.014 | 0.608 | 0.528 |
| medium | 93 | 0.675 | 0.726 | +0.051 | 0.444 | 0.166 |
| small | 35 | 0.742 | 0.767 | +0.025 | 0.138 | 0.043 |
| tiny | 4 | 0.631 | 0.583 | -0.047 | 0.058 | 0.008 |

### boundary

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| compact | 169 | 0.698 | 0.737 | +0.039 | 0.549 | 0.336 |
| complex | 55 | 0.744 | 0.744 | +0.001 | 0.323 | 0.096 |

### position

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| center | 90 | 0.689 | 0.733 | +0.043 | 0.517 | 0.187 |
| edge | 39 | 0.744 | 0.778 | +0.034 | 0.127 | 0.138 |
| mid | 95 | 0.712 | 0.730 | +0.018 | 0.566 | 0.305 |

### softness

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| hard | 113 | 0.759 | 0.750 | -0.009 | 0.568 | 0.487 |
| soft | 111 | 0.641 | 0.730 | +0.089 | 0.454 | 0.164 |

### components

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| multi | 8 | 0.851 | 0.845 | -0.006 | 0.444 | 0.104 |
| single | 216 | 0.706 | 0.735 | +0.029 | 0.501 | 0.236 |

### topology

| 类 | n | P1 top-k IoU | P3' top-k IoU | P3'−P1 | 中心先验 | 随机地板 |
|---|---|---|---|---|---|---|
| solid | 224 | 0.710 | 0.742 | +0.032 | 0.486 | 0.225 |

## 2. normal / low 分层（数据纪律）

| 臂 | population | area 各档 n / top-k IoU |
|---|---|---|
| P1（经 Phi-71） | normal_only | large n=92 0.756 ｜ medium n=93 0.675 ｜ small n=35 0.742 ｜ tiny n=4 0.631 |
| P1（经 Phi-71） | pooled | large n=189 0.728 ｜ medium n=151 0.674 ｜ small n=52 0.741 ｜ tiny n=8 0.567 |
| P3'（不经 Phi） | normal_only | large n=92 0.742 ｜ medium n=93 0.726 ｜ small n=35 0.767 ｜ tiny n=4 0.583 |
| P3'（不经 Phi） | pooled | large n=189 0.742 ｜ medium n=151 0.760 ｜ small n=52 0.764 ｜ tiny n=8 0.567 |

> `low` 不进评测 GT（CLAUDE.md 数据纪律）。pooled 行仅供对照，**不得用于判据**：low 层 GT 面积更大 ⇒ 随机地板更高 ⇒ 每个 IoU 列都被系统性抬高。

## 3. family × area 交叉表（top-k IoU 中位 / 中心先验）

| family \| area | n | P1 | P3' | 中心先验 |
|---|---|---|---|---|
| band|large | 67 | 0.728 | 0.762 | 0.574 |
| band|medium | 41 | 0.567 | 0.545 | 0.394 |
| linear|large | 112 | 0.728 | 0.727 | 0.647 |
| radial|large | 6 | 0.662 | 0.758 | 0.590 |
| radial|medium | 71 | 0.658 | 0.782 | 0.473 |
| radial|small | 23 | 0.424 | 0.735 | 0.222 |
| semantic|large | 4 | 0.903 | 0.889 | 0.456 |
| semantic|medium | 39 | 0.848 | 0.850 | 0.422 |
| semantic|small | 29 | 0.784 | 0.767 | 0.131 |
| semantic|tiny | 8 | 0.567 | 0.567 | 0.000 |

## 4. 深尾归因（可救区 / 不可救区）

| 桶 | 性质 | P1 | P3' | 含义 |
|---|---|---|---|---|
| `chain_ceiling` | **不可救** | 1 | 2 | H/16 + guided upsample 在交付分辨率上表达不了该掩膜 |
| `basis_ceiling` | **不可救** | 1 | 0 | Phi-71 基底表达不了该掩膜（**仅 P1 适用**） |
| `context_missing` | 可救 | 3 | 0 | GT 上下文能做对、generated 做不对 ⇒ 信息丢在 `<where>` 文本，不在头 |
| `coverage_bias` | 可救 | 13 | 10 | 面积比系统性偏离 1 ⇒ 损失配重问题 |
| `underfit` | 可救 | 22 | 28 | 各上界都高、上下文也没问题 ⇒ 头单纯没学会 |

**不可救区占深尾**：P1 **2/40 = 5.0%**，P3' **2/40 = 5.0%**。
**可救区占深尾**：P1 38/40 = 95.0%，P3' 38/40 = 95.0%。

### 深尾构成（按 family 与几何类）

- **P1（经 Phi-71）**：family {'radial': 20, 'semantic': 4, 'band': 16}；area {'small': 15, 'medium': 21, 'tiny': 1, 'large': 3}；softness {'soft': 38, 'hard': 2}；面积偏向 {'under': 11, 'over': 29}
- **P3'（不经 Phi）**：family {'radial': 6, 'semantic': 7, 'band': 25, 'linear': 2}；area {'small': 7, 'medium': 25, 'tiny': 2, 'large': 6}；softness {'soft': 32, 'hard': 8}；面积偏向 {'over': 20, 'under': 20}

## 5. 综合判读

### 5.1 尾部**不是**表达能力问题（两臂同结论）

两条上界中位均 ≈0.99，深尾里 `chain_ceiling` + `basis_ceiling` 合计只占 **5.0%（2/40）**（两臂相同）。**95% 的深尾是可救的**，且主要是「没学会」与「配重偏了」这两类工程问题，不是分辨率或基底的物理上限。

### 5.2 Phi-71 的代价定位在**软边 + 过覆盖**（本节是 A/B 的机制解释）

- 软边分层：P1 `soft` 0.641 vs `hard` 0.759（Δ -0.118）；P3' `soft` 0.730 vs `hard` 0.750（Δ -0.020）。**P1 在软边上掉得多得多。**
- 深尾软边占比：P1 **38/40**，P3' 32/40。
- 深尾过覆盖占比：P1 **29/40**，P3' 20/40（P3' 过/欠基本对半）。
- P1 深尾的 family 集中在 `radial` + `band`，**零 `linear`**。

**机制解释**：`R(s;ρ)` 的 `band` 读出用一组**全局**参数（mu/h/k/pi）把标量场映射成掩膜，无法表达**空间上变化的软衰减**；遇到软边目标只能整体放宽 ⇒ 过覆盖。P3' 直出场 + 可学 gain 没有这个约束。这就是 A/B 里 **P1 − P3' = −0.0143 (p=6e-4)** 的来处：**代价不是均匀分布的，而是集中在软边 radial/band 这一族上**。

### 5.3 两臂都不在照抄中心先验

`position=edge` 档中心先验只有 0.127（它天然最不擅长的一档），而 P1 0.744、P3' 0.778 —— **两臂在先验最差的一档上反而最好**，是 M3 未复现的独立佐证。

### 5.4 建议下一步（按可救类型对应修法）

| 尾部类型 | 占比（P1 / P3'） | 对应修法 |
|---|---|---|
| `underfit` | 22/40 ｜ 28/40 | 加步数/容量：1200 步只是 ~0.9 个 epoch，两臂都远未收敛 |
| `coverage_bias` | 13/40 ｜ 10/40 | 面积带罚权重（当前 0.05）与 τ=0.15 需重标定；**P1 偏过覆盖，应非对称加罚** |
| `context_missing` | 3/40 ｜ 0/40 | 只在 P1 出现且仅 3 例；属 `<where>` 生成文本质量，不在头内 |
| 不可救 | 2/40 ｜ 2/40 | 无需投入——已达链路/基底上界 |

