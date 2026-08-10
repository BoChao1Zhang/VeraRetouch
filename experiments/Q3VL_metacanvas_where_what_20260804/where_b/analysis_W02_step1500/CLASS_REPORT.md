## 要验证的结论
如果这次分析成立，我们就能说 **W02 step 1500 的 Where 失败不是均匀的**：它集中在特定几何类别（哪一类由下表给出），且长尾样本的失败机制可以被归到 context / oracle 天花板 / s 场 / readout / upsample 这五个环节中的具体一个；失败就不能说——只能继续用一个 median soft-IoU 描述整臂，无法指出改哪里。

## 为什么需要验证它
论文要主张的是「语言条件的空间场能定位到指令所指的区域」。审稿人会问：**在什么样的区域上成立？**一个只在大面积、居中、实心区域上成立的场，与中心先验难以区分（红线：中心先验 AUC 0.836 曾跑赢全部六个 attention 读出）。同时，训练资源必须投在贡献最多差样本的那个环节上；没有归因表就只能靠猜。

## 怎么验的
读 `/home/bc/data/runs/where_b/W02/eval_step1500` 的 `per_sample.jsonl`（6232 行，7 个 context），用 Where-A 发布的 `V_where` GT mask（`.maskhi.png`，400 个 local 样本）逐样本算六个几何量并分档；每一档的指标由 `q3vl.whereb.metrics.summarise`（主榜同一个聚合器）在该档子集上重算；再对最差的 81 个样本按预注册决策树打失败机制标签。

---

## 1. 设置

| 项 | 值 |
| --- | ---: |
| arm | W02 |
| structure / readout | MC8-Joint / cband12 |
| checkpoint step | 1500 |
| eval 产物 | `/home/bc/data/runs/where_b/W02/eval_step1500` |
| split（数据代号） | `V_where`（DATA_ASSIGNMENT：S-val 源，永不进训练） |
| 主榜 context | generated |
| local / global 样本数 | 400 / 496 |
| 整臂 local softIoU 中位 | 0.511 |
| 整臂 grid hardIoU | 0.475 |
| 整臂 grid 边界 F1 | 0.306 |
| 中心先验 hardIoU / Δ / p | 0.518 / -0.026 / 0.0001 |
| field cache（GPU 重跑） | experiments/Q3VL_metacanvas_where_what_20260804/where_b/analysis_W02_step1500/fields |
| git commit | c1b4cdc94c0939f193bf3775ace1b51a48cf5723 |

> **判据纪律**：本报告不含任何 AUC 列（2026-08-05 红线）。所有二值化一律「匹配 GT 面积的 top-k」，每张表都带零参数中心先验列与配对 Δ、p 值。`n < 20` 的类别标 ⚠低置信——中位数不是发现。

## 2. 分类学定义

| 维度 | 定义（只看 GT，不看预测） | 分档 |
| --- | ---: | ---: |
| area | `area_frac = mean(mask_hi > 0.5)` | tiny < 0.05, small < 0.15, medium < 0.45, else large |
| components | 8 连通域计数，丢弃 < max(64 px, 2% 掩膜面积) 的碎片 | single = 1，multi ≥ 2 |
| topology | 填洞前后之差；洞需 ≥ max(64 px, 1% 填充面积) | holed / solid |
| boundary | `circularity = 4πA/P²`，P 用 marching-square 加权周长（圆 0.915、方 0.799、环 0.300 实测） | compact ≥ 0.5，否则 complex |
| position | 质心到画幅中心距离，short_side_unit 坐标（与中心先验场同一约定），除以角点距离归一 | center ≤ 0.15, edge > 0.3, 其余 mid |
| softness | `soft_frac = |{0.05 < m < 0.95}| / |{m > 0.05}|` | soft ≥ 0.4，否则 hard |

**global 样本（GT 全 1，496 个）不进任何几何分层**——它们不是「一种区域」，而是没有区域；混进去会把 496 行接近满分的样本灌进 large/solid/center 格。

### 2.1 类别分布与几何量分位

| 维度 | n | 分布 |
| --- | ---: | ---: |
| 面积占比（area_frac） | 400 | large=189（47%）, medium=151（38%）, small=52（13%）, tiny=8（2%） |
| 连通域数 | 400 | multi=16（4%）, single=384（96%） |
| 边界复杂度（circularity） | 400 | compact=305（76%）, complex=95（24%） |
| 位置（质心到画幅中心） | 400 | center=147（37%）, edge=76（19%）, mid=177（44%） |
| 软边比例 | 400 | hard=213（53%）, soft=187（47%） |
| region | 400 | center=305（76%）, left=25（6%）, lower=41（10%）, lower left=2（0%）, lower right=6（2%）, right=16（4%）, upper=5（1%） |
| winner_confidence（已有分层） | 400 | low=176（44%）, normal=224（56%） |
| image.upscaled（已有分层） | 400 | False=346（86%）, True=54（14%） |
| build（已有分层） | 400 | l1=75（19%）, l2=57（14%）, l3=53（13%）, l4=71（18%）, l5=75（19%）, l6=69（17%） |
| active primitive count（预测侧） | 400 | >=3=16（4%）, n/a-not-emitted=384（96%） |

| 几何量 | p10 | p50 | p90 |
| --- | ---: | ---: | ---: |
| area_frac | 0.104 | 0.409 | 0.764 |
| circularity | 0.345 | 0.680 | 0.832 |
| centroid_dist_rel | 0.059 | 0.181 | 0.370 |
| soft_frac | 0.194 | 0.369 | 0.649 |
| mask_max | 0.616 | 1.000 | 1.000 |
| largest_component_frac | 1.000 | 1.000 | 1.000 |

## 3. 各维度分层表（主榜 = generated context，右侧两列为 GT context 对照）

### 3.1 面积占比（area_frac）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| large | 189 | 0.639 | 0.467 | 0.614 | 0.218 | 0.616 | -0.012 | 0.1141 | 0.650 | 0.560 | -0.0005 | 0.704 | 0.064 |
| medium | 151 | 0.399 | 0.264 | 0.352 | 0.430 | 0.424 | -0.047 | 0.0001 | 0.429 | 0.513 | -0.0325 | 0.418 | 0.018 |
| small | 52 | 0.187 | 0.086 | 0.181 | 0.449 | 0.139 | -0.034 | 0.1254 | 0.218 | 0.578 | -0.0201 | 0.207 | 0.020 |
| tiny  ⚠低置信 | 8 | 0.051 | 0.021 | 0.273 | 0.800 | 0.000 | 0.107 | 0.0662 | 0.121 | 0.828 | -0.0048 | 0.051 | -0.000 |

最差类别：**small**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`tiny` 的中位更低（0.051，n=8 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.2 连通域数

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi  ⚠低置信 | 16 | 0.364 | 0.051 | 0.325 | 0.490 | 0.425 | -0.038 | 0.2505 | 0.429 | 0.665 | -0.0375 | 0.364 | -0.000 |
| single | 384 | 0.522 | 0.210 | 0.491 | 0.300 | 0.531 | -0.025 | 0.0001 | 0.539 | 0.551 | -0.0106 | 0.558 | 0.036 |

最差类别：**single**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`multi` 的中位更低（0.364，n=16 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.3 边界复杂度（circularity）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact | 305 | 0.587 | 0.277 | 0.553 | 0.281 | 0.571 | -0.025 | 0.0003 | 0.601 | 0.547 | -0.0077 | 0.627 | 0.039 |
| complex | 95 | 0.333 | 0.086 | 0.279 | 0.479 | 0.323 | -0.027 | 0.0165 | 0.361 | 0.576 | -0.0289 | 0.333 | -0.000 |

最差类别：**complex**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.4 位置（质心到画幅中心）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| center | 147 | 0.521 | 0.255 | 0.497 | 0.415 | 0.536 | -0.074 | 0.0001 | 0.536 | 0.553 | -0.0253 | 0.557 | 0.036 |
| edge | 76 | 0.291 | 0.088 | 0.215 | 0.277 | 0.175 | 0.028 | 0.0043 | 0.315 | 0.459 | -0.0008 | 0.295 | 0.003 |
| mid | 177 | 0.608 | 0.296 | 0.558 | 0.261 | 0.573 | -0.009 | 0.2000 | 0.618 | 0.601 | -0.0066 | 0.646 | 0.038 |

最差类别：**edge**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.5 软边比例

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard | 213 | 0.592 | 0.296 | 0.567 | 0.234 | 0.583 | -0.015 | 0.0210 | 0.604 | 0.576 | -0.0010 | 0.645 | 0.053 |
| soft | 187 | 0.393 | 0.139 | 0.355 | 0.381 | 0.454 | -0.037 | 0.0001 | 0.416 | 0.543 | -0.0184 | 0.423 | 0.030 |

最差类别：**soft**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.6 region

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| center | 305 | 0.587 | 0.291 | 0.546 | 0.313 | 0.574 | -0.045 | 0.0001 | 0.601 | 0.577 | -0.0173 | 0.627 | 0.039 |
| left | 25 | 0.384 | 0.088 | 0.296 | 0.235 | 0.340 | -0.009 | 0.6012 | 0.398 | 0.432 | -0.0000 | 0.385 | 0.001 |
| lower | 41 | 0.298 | 0.125 | 0.302 | 0.382 | 0.170 | 0.077 | 0.0001 | 0.330 | 0.571 | -0.0174 | 0.310 | 0.012 |
| lower left  ⚠低置信 | 2 | 0.081 | 0.081 | 0.035 | 0.170 | 0.009 | 0.054 | 0.5053 | 0.093 | 0.360 | -0.0028 | 0.080 | -0.001 |
| lower right  ⚠低置信 | 6 | 0.129 | 0.032 | 0.045 | 0.166 | 0.000 | 0.054 | 0.0633 | 0.176 | 0.194 | -0.0013 | 0.134 | 0.005 |
| right  ⚠低置信 | 16 | 0.279 | 0.074 | 0.235 | 0.278 | 0.259 | 0.012 | 0.5277 | 0.319 | 0.513 | 0.0058 | 0.284 | 0.004 |
| upper  ⚠低置信 | 5 | 0.356 | 0.111 | 0.282 | 0.215 | 0.321 | -0.033 | 0.0635 | 0.361 | 0.292 | 0.0012 | 0.306 | -0.049 |

最差类别：**lower**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`lower left` 的中位更低（0.081，n=2 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.7 winner_confidence（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 176 | 0.549 | 0.217 | 0.501 | 0.284 | 0.543 | -0.027 | 0.0007 | 0.574 | 0.585 | -0.0064 | 0.600 | 0.050 |
| normal | 224 | 0.482 | 0.185 | 0.455 | 0.335 | 0.486 | -0.024 | 0.0021 | 0.497 | 0.525 | -0.0134 | 0.501 | 0.019 |

最差类别：**normal**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.8 image.upscaled（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 346 | 0.492 | 0.186 | 0.461 | 0.304 | 0.515 | -0.028 | 0.0001 | 0.520 | 0.547 | -0.0091 | 0.523 | 0.031 |
| True | 54 | 0.579 | 0.236 | 0.527 | 0.346 | 0.526 | -0.013 | 0.3775 | 0.593 | 0.606 | -0.0187 | 0.586 | 0.007 |

最差类别：**False**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.9 build（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1 | 75 | 0.522 | 0.167 | 0.475 | 0.335 | 0.516 | -0.014 | 0.2348 | 0.553 | 0.553 | -0.0159 | 0.529 | 0.007 |
| l2 | 57 | 0.503 | 0.255 | 0.456 | 0.308 | 0.503 | -0.031 | 0.0547 | 0.526 | 0.512 | -0.0091 | 0.575 | 0.072 |
| l3 | 53 | 0.573 | 0.300 | 0.534 | 0.267 | 0.568 | -0.030 | 0.0484 | 0.583 | 0.575 | -0.0022 | 0.573 | 0.000 |
| l4 | 71 | 0.484 | 0.202 | 0.455 | 0.308 | 0.525 | -0.042 | 0.0094 | 0.510 | 0.507 | -0.0065 | 0.523 | 0.039 |
| l5 | 75 | 0.442 | 0.185 | 0.408 | 0.323 | 0.486 | -0.026 | 0.0406 | 0.465 | 0.596 | -0.0166 | 0.478 | 0.036 |
| l6 | 69 | 0.577 | 0.180 | 0.505 | 0.348 | 0.535 | -0.014 | 0.2422 | 0.584 | 0.530 | -0.0167 | 0.579 | 0.002 |

最差类别：**l5**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.10 active primitive count（预测侧）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| >=3  ⚠低置信 | 16 | 0.074 | 0.032 | 0.273 | 0.245 | 0.013 | 0.051 | 0.0754 | 0.121 | 0.683 | 0.0001 | 0.074 | -0.000 |
| n/a-not-emitted | 384 | 0.515 | 0.224 | 0.487 | 0.308 | 0.525 | -0.029 | 0.0001 | 0.532 | 0.553 | -0.0129 | 0.556 | 0.040 |

最差类别：**n/a-not-emitted**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`>=3` 的中位更低（0.074，n=16 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

## 4. 长尾归因

**长尾集合**（统计口径）= 主榜 context 下 softIoU < 0.3 的全部 local 样本，共 **81** 个（占 local 的 20.2%）；其中最差 12 个 + 每个维度最差类别的最差样本另出联图。机制标签是多标签；`primary` 按「越上游越优先」的固定顺序取一个。

| 失败机制 | primary 计数 | primary 占比 | 命中计数（多标签） | 命中占比 | 未测样本数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 面积失配（pred_mean vs gt_mean） | 54 | 66.7% | 78 | 96.3% | 0 |
| s 轴塌缩（std(s)/std(s*) 过小） | 11 | 13.6% | 13 | 16.0% | 0 |
| oracle 天花板低（basis 表达上界） | 10 | 12.3% | 10 | 12.3% | 0 |
| 未归因 | 3 | 3.7% | 3 | 3.7% | 0 |
| s 场方向错（w_dir 与 oracle 夹角大） | 2 | 2.5% | 6 | 7.4% | 69 |
| s 错（换 oracle s 就修好） | 1 | 1.2% | 4 | 4.9% | 69 |
| rho 错（换 oracle rho 就修好） | 0 | 0.0% | 1 | 1.2% | 69 |
| 打不过零参数中心先验 | 0 | 0.0% | 23 | 28.4% | 0 |

> `命中占比` 是多标签，列和会超过 100%；`primary 占比` 和为 100%。`未测样本数`= 该样本上这个机制**无法测**（没有 oracle、没跑 field cache、或该 readout 天生 n/a）——**不是**「不存在」。

同一套判定跑在**全部 local 样本**上（含好样本）作对照。注意好样本上的机制标签本身没有意义——一个 softIoU 0.9 的大面积样本当然会有近似常数的 s（`s_collapse`）——这一列只用来回答「长尾里的机制是不是尾部特有」：

| 失败机制 | primary 计数 | primary 占比 | 命中计数 | 命中占比 |
| --- | ---: | ---: | ---: | ---: |
| 未归因 | 146 | 36.5% | 146 | 36.5% |
| 面积失配（pred_mean vs gt_mean） | 111 | 27.8% | 143 | 35.8% |
| 打不过零参数中心先验 | 53 | 13.2% | 120 | 30.0% |
| generated context 质量（GT 好 / generated 差） | 39 | 9.8% | 39 | 9.8% |
| s 轴塌缩（std(s)/std(s*) 过小） | 37 | 9.2% | 50 | 12.5% |
| oracle 天花板低（basis 表达上界） | 10 | 2.5% | 10 | 2.5% |
| s 场方向错（w_dir 与 oracle 夹角大） | 3 | 0.8% | 7 | 1.8% |
| s 错（换 oracle s 就修好） | 1 | 0.2% | 4 | 1.0% |
| rho 错（换 oracle rho 就修好） | 0 | 0.0% | 1 | 0.2% |

**瓶颈结论**：长尾里 primary 占比最高的机制是 **面积失配（pred_mean vs gt_mean）**（66.7%）；未归因样本占 3.7%。

### 4.1 长尾样本清单

| sample_id | softIoU | hardIoU | 中心先验 | oracle | area/components/boundary/position/softness/region/winner_confidence/upscaled/build/active_primitive_bucket | primary 机制 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sft_9e2c10ec2fc4…` | 0.020 | 0.077 | 0.000 | 0.503 | tiny/single/complex/mid/soft/center/normal/False/l6/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_440542d09386…` | 0.021 | 0.000 | 0.000 | 0.567 | tiny/single/compact/edge/soft/right/low/False/l5/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_db86a642948d…` | 0.032 | 0.000 | 0.000 | 0.546 | tiny/multi/complex/edge/soft/lower right/low/False/l4/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_4fd87f21a752…` | 0.036 | 0.130 | 0.000 | 0.286 | tiny/multi/complex/mid/soft/center/normal/False/l5/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_9609c0ca0503…` | 0.042 | 0.000 | 0.000 | 0.715 | small/single/compact/edge/soft/right/normal/False/l1/>=3 | s 场方向错（w_dir 与 oracle 夹角大） |
| `sft_b2f508743028…` | 0.051 | 0.304 | 0.000 | 0.420 | tiny/multi/complex/mid/soft/lower/low/False/l5/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_b78fab3865c7…` | 0.071 | 0.289 | 0.115 | 0.441 | tiny/single/complex/mid/soft/lower/normal/False/l5/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_1d863dc54301…` | 0.074 | 0.304 | 0.132 | 0.503 | tiny/single/complex/mid/soft/lower/normal/False/l3/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_0e994a93216c…` | 0.074 | 0.000 | 0.000 | 0.922 | small/single/compact/edge/soft/right/low/False/l6/>=3 | s 场方向错（w_dir 与 oracle 夹角大） |
| `sft_35ec9c8dbc38…` | 0.077 | 0.273 | 0.273 | 0.684 | tiny/single/compact/center/soft/center/low/False/l1/>=3 | oracle 天花板低（basis 表达上界） |
| `sft_78f26362300a…` | 0.079 | 0.036 | 0.013 | 0.850 | small/single/complex/edge/hard/left/normal/False/l6/>=3 | 面积失配（pred_mean vs gt_mean） |
| `sft_cd79da4ed738…` | 0.080 | 0.053 | 0.000 | 0.837 | small/single/complex/edge/soft/lower/low/False/l6/>=3 | s 错（换 oracle s 就修好） |
| `sft_05b36cf36dab…` | 0.081 | 0.035 | 0.013 | 0.872 | small/single/complex/edge/hard/lower left/normal/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_bdca5c11f7e4…` | 0.086 | 0.026 | 0.013 | 0.850 | small/single/complex/edge/hard/left/normal/False/l6/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_3be5685578ee…` | 0.088 | 0.025 | 0.013 | 0.851 | small/single/complex/edge/hard/left/low/False/l5/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_690e6624b8e9…` | 0.094 | 0.094 | 0.009 | 0.403 | small/single/complex/edge/soft/lower left/low/False/l6/n/a-not-emitted | oracle 天花板低（basis 表达上界） |
| `sft_f49b47f3d915…` | 0.111 | 0.074 | 0.078 | 0.861 | small/single/compact/edge/hard/upper/normal/False/l2/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_9392222c7dc7…` | 0.112 | 0.033 | 0.000 | 0.735 | small/single/compact/edge/soft/lower right/low/False/l1/n/a-not-emitted | s 轴塌缩（std(s)/std(s*) 过小） |
| `sft_6ec6373df6d9…` | 0.125 | 0.088 | 0.000 | 0.761 | small/single/complex/edge/soft/lower/normal/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_80cc3e0911ed…` | 0.128 | 0.088 | 0.000 | 0.762 | small/single/complex/edge/soft/lower/normal/False/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_b434e56ede7e…` | 0.129 | 0.045 | 0.000 | 0.684 | small/single/compact/edge/soft/lower right/normal/False/l2/n/a-not-emitted | oracle 天花板低（basis 表达上界） |
| `sft_ced1f1b69e2d…` | 0.129 | 0.088 | 0.000 | 0.740 | small/single/complex/edge/soft/lower/normal/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_cc3aa5b91f70…` | 0.130 | 0.047 | 0.000 | 0.742 | small/single/compact/edge/soft/lower right/normal/False/l5/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_9210bd4bee69…` | 0.135 | 0.030 | 0.051 | 0.814 | small/multi/complex/edge/soft/lower/normal/False/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_6a27ddcd8e19…` | 0.135 | 0.055 | 0.000 | 0.770 | small/single/complex/edge/hard/lower/normal/True/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_6c212a9890cf…` | 0.139 | 0.097 | 0.000 | 0.760 | small/single/complex/edge/soft/lower/low/False/l5/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_12fb83d345b6…` | 0.155 | 0.202 | 0.167 | 0.889 | small/single/complex/edge/soft/lower/low/False/l6/n/a-not-emitted | s 轴塌缩（std(s)/std(s*) 过小） |
| `sft_8b59235a1c65…` | 0.157 | 0.043 | 0.385 | 0.788 | small/single/complex/center/soft/center/normal/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_0e7bd082004f…` | 0.167 | 0.197 | 0.532 | 0.891 | small/single/compact/center/soft/center/normal/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_941f07dcbc45…` | 0.171 | 0.181 | 0.532 | 0.842 | small/single/compact/center/soft/center/normal/False/l2/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_f9280808af9a…` | 0.175 | 0.181 | 0.518 | 0.904 | small/single/compact/center/soft/center/normal/False/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_9bb5598eeb22…` | 0.176 | 0.178 | 0.536 | 0.878 | small/single/compact/center/soft/center/normal/False/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_3e7eac89df42…` | 0.179 | 0.078 | 0.085 | 0.781 | medium/single/complex/edge/hard/left/low/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_30745d7326d2…` | 0.180 | 0.125 | 0.112 | 0.930 | small/single/compact/mid/soft/center/normal/False/l6/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_0d16e05f6811…` | 0.185 | 0.145 | 0.106 | 0.938 | small/single/compact/mid/soft/center/normal/False/l5/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_f23664eb8819…` | 0.186 | 0.114 | 0.073 | 0.957 | medium/single/complex/edge/soft/right/low/False/l1/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_1b4cd92929db…` | 0.187 | 0.119 | 0.021 | 0.928 | small/single/compact/edge/soft/lower/normal/False/l5/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_baf15c9c0c84…` | 0.198 | 0.159 | 0.476 | 0.921 | medium/single/compact/center/soft/center/normal/False/l1/n/a-not-emitted | s 轴塌缩（std(s)/std(s*) 过小） |
| `sft_34b2d6eb76bd…` | 0.200 | 0.333 | 0.533 | 0.917 | small/single/compact/center/soft/center/low/False/l4/n/a-not-emitted | 面积失配（pred_mean vs gt_mean） |
| `sft_87ce6b49e4ea…` | 0.202 | 0.215 | 0.096 | 0.960 | medium/single/compact/edge/soft/lower right/normal/True/l4/n/a-not-emitted | s 轴塌缩（std(s)/std(s*) 过小） |

联图见 `viz/`（85 张）。每张固定为 `I_in | GT mask | GT 叠图 | pred mask hi | pred mask low | s 叠图`，缺的格子画成 “not available” 而不是省略。

## 5. 结论与建议下一步

- 长尾里触发 `area_mismatch` 的 78 个样本中 **78 个是过覆盖**（pred_mean > gt_mean），预测/GT 面积比中位 **3.57**——场倾向于铺开成接近全局的掩膜，而不是缩到指令所指的区域。
- `area` 维度上最差类别是 **small**（softIoU 中位 0.187），最好类别 large（0.639），跨类差 0.453。
- `boundary` 维度上最差类别是 **complex**（softIoU 中位 0.333），最好类别 compact（0.587），跨类差 0.254。
- `position` 维度上最差类别是 **edge**（softIoU 中位 0.291），最好类别 mid（0.608），跨类差 0.317。
- `softness` 维度上最差类别是 **soft**（softIoU 中位 0.393），最好类别 hard（0.592），跨类差 0.198。
- `region` 维度上最差类别是 **lower**（softIoU 中位 0.298），最好类别 center（0.587），跨类差 0.290。
- `winner_confidence` 维度上最差类别是 **normal**（softIoU 中位 0.482），最好类别 low（0.549），跨类差 0.067。
- `upscaled` 维度上最差类别是 **False**（softIoU 中位 0.492），最好类别 True（0.579），跨类差 0.087。
- `build` 维度上最差类别是 **l5**（softIoU 中位 0.442），最好类别 l6（0.577），跨类差 0.135。
- 以下维度在本 split 上**只有一个类别**，表里不出现：`topology`——不是模型在这些维度上没差别，是数据里没有对比。
- 长尾的 primary 机制以 **area_mismatch** 为首（66.7%）；全体 local 样本上同一机制占 27.8% —— 这是**尾部特有**的机制（全体样本上稀有得多）。

## 6. 已知盲区

- **本工具只做单维度分层，不做全交叉**：六维全交叉是 144 格 / 400 个 local 样本。「small 且 multi」这种交互效应本报告读不出来，需要单独立项。
- **分类学只看 GT 几何**，不含语义（人 / 天空 / 建筑）。一个「所有小面积样本都是人脸」的混杂本报告识别不了。
- **归因的阈值是预注册的**（见 `config/thresholds.json`），但阈值附近的样本会在标签间抖动；`primary` 的顺序假定越上游的机制越该先修，这是一个工程判断，不是测量结果。
- **`s_direction` / `s_error` / `rho_error` / `single_primitive` 需要重跑 checkpoint**（`--checkpoint`）。没跑时它们不是 0，是 not_tested；把它们当 0 读会把瓶颈错误地推给 context。
- **`V_where` 的 400 个 `.cgt` 掩膜没有一个带洞、96% 单连通**，所以协议 §13 要求联图覆盖的「环形 / 多连通区域」在本 split 上基本取不到样本——这是数据的性质，不是本报告的遗漏。
- **local softIoU 是被直接优化的量**（`1 - softIoU` 是 L_mask 权重 1.00 的支配项），所以「哪一类最差」的排序里，它承担的是收敛度而不是独立证据；同一张表里的 grid 边界 F1 与中心先验 Δ 才是没被优化的列。详见同目录 eval 产物的 `ATTRIBUTION.md`。

> 分类学阈值的标定依据、核实记录、以及**待主 agent 决策的 6 项**，见 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/WEVAL1_NOTES.md`。
