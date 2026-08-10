## 要验证的结论
如果这次分析成立，我们就能说 **W01 step 1500 的 Where 失败不是均匀的**：它集中在特定几何类别（哪一类由下表给出），且长尾样本的失败机制可以被归到 context / oracle 天花板 / s 场 / readout / upsample 这五个环节中的具体一个；失败就不能说——只能继续用一个 median soft-IoU 描述整臂，无法指出改哪里。

## 为什么需要验证它
论文要主张的是「语言条件的空间场能定位到指令所指的区域」。审稿人会问：**在什么样的区域上成立？**一个只在大面积、居中、实心区域上成立的场，与中心先验难以区分（红线：中心先验 AUC 0.836 曾跑赢全部六个 attention 读出）。同时，训练资源必须投在贡献最多差样本的那个环节上；没有归因表就只能靠猜。

## 怎么验的
读 `/home/bc/data/runs/where_b/W01/eval_step1500` 的 `per_sample.jsonl`（6232 行，7 个 context），用 Where-A 发布的 `V_where` GT mask（`.maskhi.png`，400 个 local 样本）逐样本算六个几何量并分档；每一档的指标由 `q3vl.whereb.metrics.summarise`（主榜同一个聚合器）在该档子集上重算；再对最差的 89 个样本按预注册决策树打失败机制标签。

---

## 1. 设置

| 项 | 值 |
| --- | ---: |
| arm | W01 |
| structure / readout | MC8-Joint / band |
| checkpoint step | 1500 |
| eval 产物 | `/home/bc/data/runs/where_b/W01/eval_step1500` |
| split（数据代号） | `V_where`（DATA_ASSIGNMENT：S-val 源，永不进训练） |
| 主榜 context | generated |
| local / global 样本数 | 400 / 496 |
| 整臂 local softIoU 中位 | 0.486 |
| 整臂 grid hardIoU | 0.503 |
| 整臂 grid 边界 F1 | 0.298 |
| 中心先验 hardIoU / Δ / p | 0.518 / -0.012 / 0.0675 |
| field cache（GPU 重跑） | experiments/Q3VL_metacanvas_where_what_20260804/where_b/analysis_W01_step1500/fields |
| git commit | db20c1882f53481cf0795f59a449544069f2e1e1 |

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
| large | 189 | 0.630 | 0.465 | 0.607 | 0.203 | 0.616 | -0.025 | 0.0006 | 0.645 | 0.042 | -0.0006 | 0.690 | 0.061 |
| medium | 151 | 0.390 | 0.245 | 0.386 | 0.438 | 0.424 | -0.009 | 0.3671 | 0.420 | 0.484 | -0.0038 | 0.396 | 0.006 |
| small | 52 | 0.155 | 0.084 | 0.155 | 0.413 | 0.139 | 0.008 | 0.7834 | 0.181 | 0.489 | -0.0005 | 0.160 | 0.005 |
| tiny  ⚠低置信 | 8 | 0.044 | 0.023 | 0.191 | 0.738 | 0.000 | 0.126 | 0.0499 | 0.095 | 0.935 | -0.0001 | 0.043 | -0.001 |

最差类别：**small**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`tiny` 的中位更低（0.044，n=8 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.2 连通域数

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| multi  ⚠低置信 | 16 | 0.270 | 0.044 | 0.386 | 0.492 | 0.425 | -0.001 | 0.9797 | 0.376 | 0.649 | -0.0055 | 0.271 | 0.002 |
| single | 384 | 0.501 | 0.173 | 0.514 | 0.288 | 0.531 | -0.012 | 0.0618 | 0.522 | 0.405 | -0.0006 | 0.533 | 0.033 |

最差类别：**single**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`multi` 的中位更低（0.270，n=16 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.3 边界复杂度（circularity）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| compact | 305 | 0.585 | 0.232 | 0.550 | 0.259 | 0.571 | -0.021 | 0.0047 | 0.598 | 0.356 | -0.0007 | 0.612 | 0.027 |
| complex | 95 | 0.295 | 0.070 | 0.320 | 0.490 | 0.323 | 0.017 | 0.2452 | 0.340 | 0.490 | -0.0006 | 0.296 | 0.001 |

最差类别：**complex**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.4 位置（质心到画幅中心）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| center | 147 | 0.500 | 0.222 | 0.529 | 0.386 | 0.536 | -0.052 | 0.0001 | 0.518 | 0.466 | -0.0026 | 0.509 | 0.008 |
| edge | 76 | 0.270 | 0.092 | 0.293 | 0.256 | 0.175 | 0.052 | 0.0002 | 0.307 | 0.413 | 0.0001 | 0.274 | 0.004 |
| mid | 177 | 0.599 | 0.277 | 0.548 | 0.227 | 0.573 | -0.006 | 0.4423 | 0.628 | 0.378 | -0.0007 | 0.636 | 0.037 |

最差类别：**edge**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.5 软边比例

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hard | 213 | 0.596 | 0.256 | 0.550 | 0.216 | 0.583 | -0.021 | 0.0016 | 0.604 | 0.378 | -0.0007 | 0.633 | 0.038 |
| soft | 187 | 0.387 | 0.127 | 0.406 | 0.374 | 0.454 | -0.001 | 0.9274 | 0.404 | 0.461 | -0.0008 | 0.402 | 0.015 |

最差类别：**soft**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.6 region

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| center | 305 | 0.583 | 0.257 | 0.552 | 0.302 | 0.574 | -0.035 | 0.0001 | 0.597 | 0.405 | -0.0012 | 0.612 | 0.030 |
| left | 25 | 0.356 | 0.070 | 0.315 | 0.202 | 0.340 | 0.016 | 0.0848 | 0.384 | 0.484 | 0.0007 | 0.364 | 0.008 |
| lower | 41 | 0.275 | 0.103 | 0.331 | 0.491 | 0.170 | 0.129 | 0.0001 | 0.308 | 0.426 | -0.0009 | 0.277 | 0.002 |
| lower left  ⚠低置信 | 2 | 0.069 | 0.069 | 0.013 | 0.098 | 0.009 | 0.064 | 1.0000 | 0.081 | 0.457 | 0.0034 | 0.071 | 0.002 |
| lower right  ⚠低置信 | 6 | 0.131 | 0.044 | 0.051 | 0.207 | 0.000 | 0.055 | 0.0324 | 0.186 | 0.330 | -0.0008 | 0.139 | 0.008 |
| right  ⚠低置信 | 16 | 0.306 | 0.098 | 0.278 | 0.210 | 0.259 | -0.006 | 0.5370 | 0.349 | 0.484 | 0.0027 | 0.304 | -0.001 |
| upper  ⚠低置信 | 5 | 0.360 | 0.146 | 0.294 | 0.332 | 0.321 | -0.013 | 0.4322 | 0.376 | 0.088 | 0.0014 | 0.281 | -0.079 |

最差类别：**lower**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

注：`lower left` 的中位更低（0.069，n=2 < 20），因样本量不足未被选为最差类别，但**不能**据此说它更好。

### 3.7 winner_confidence（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| low | 176 | 0.531 | 0.222 | 0.529 | 0.288 | 0.543 | -0.010 | 0.2533 | 0.546 | 0.395 | -0.0006 | 0.597 | 0.066 |
| normal | 224 | 0.448 | 0.146 | 0.472 | 0.302 | 0.486 | -0.013 | 0.1517 | 0.469 | 0.426 | -0.0009 | 0.471 | 0.023 |

最差类别：**normal**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.8 image.upscaled（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| False | 346 | 0.471 | 0.164 | 0.488 | 0.292 | 0.515 | -0.014 | 0.0340 | 0.493 | 0.405 | -0.0006 | 0.508 | 0.037 |
| True | 54 | 0.570 | 0.231 | 0.547 | 0.307 | 0.526 | 0.006 | 0.7439 | 0.576 | 0.454 | -0.0021 | 0.541 | -0.029 |

最差类别：**False**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

### 3.9 build（已有分层）

| 类别 | n | softIoU med | softIoU p10 | hardIoU | 边界F1 | 中心先验 hardIoU | Δ vs 中心先验 | Δ 的 p | /oracle | std(s)/std(s*) | hi-lo 降幅 | GT ctx softIoU med | GT−gen gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1 | 75 | 0.503 | 0.127 | 0.505 | 0.314 | 0.516 | 0.003 | 0.8383 | 0.522 | 0.405 | -0.0012 | 0.533 | 0.030 |
| l2 | 57 | 0.493 | 0.231 | 0.484 | 0.348 | 0.503 | -0.002 | 0.9247 | 0.521 | 0.426 | -0.0006 | 0.494 | 0.002 |
| l3 | 53 | 0.558 | 0.297 | 0.536 | 0.231 | 0.568 | -0.034 | 0.0263 | 0.590 | 0.050 | -0.0006 | 0.581 | 0.023 |
| l4 | 71 | 0.480 | 0.173 | 0.503 | 0.269 | 0.525 | -0.033 | 0.0663 | 0.487 | 0.378 | -0.0002 | 0.509 | 0.029 |
| l5 | 75 | 0.443 | 0.140 | 0.460 | 0.336 | 0.486 | -0.006 | 0.6759 | 0.456 | 0.435 | -0.0011 | 0.455 | 0.012 |
| l6 | 69 | 0.534 | 0.154 | 0.519 | 0.267 | 0.535 | -0.004 | 0.7978 | 0.560 | 0.467 | -0.0009 | 0.531 | -0.002 |

最差类别：**l5**（按 local softIoU 中位，只在 n ≥ 20 的类别里选）

## 4. 长尾归因

**长尾集合（主榜口径）** = 主榜 context 下 softIoU < 0.3 的全部 local 样本，共 **89** 个（占 local 的 22.2%）。**辅助口径** = 最差 10%（n=40，截断在 softIoU 0.170）。

> 两个口径都报，因为它们回答的**不是同一个问题**（主 agent 裁定 2026-08-10）：绝对口径 0.30 跨 arm、跨 step 可比——臂变强，长尾就真的变小；分位口径样本数恒定，比的是失败的**构成**而不是数量。只看前者会把「尾巴变短」误读成「尾巴变好」，只看后者则永远有 10% 的样本可以叫长尾，臂再强也一样。

最差 12 个 + 每个维度最差类别的最差样本另出联图。机制标签是多标签；`primary` 按「越上游越优先」的固定顺序取一个——**这个顺序是工程判断（先修上游），不是测量结果**；换顺序只改 `primary` 列，命中列不受影响。

| 失败机制 | primary 计数 | primary 占比 | 命中计数（多标签） | 命中占比 | 辅助口径 primary 占比（最差 10%） | 未测样本数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 面积失配（pred_mean vs gt_mean） | 62 | 69.7% | 87 | 97.8% | 65.0% | 0 |
| s 轴塌缩（std(s)/std(s*) 过小） | 12 | 13.5% | 12 | 13.5% | 2.5% | 0 |
| oracle 天花板低（basis 表达上界） | 11 | 12.4% | 11 | 12.4% | 27.5% | 0 |
| 未归因 | 2 | 2.2% | 2 | 2.2% | 0.0% | 0 |
| s 场方向错（w_dir 与 oracle 夹角大） | 1 | 1.1% | 3 | 3.4% | 2.5% | 77 |
| s 错（换 oracle s 就修好） | 1 | 1.1% | 1 | 1.1% | 2.5% | 77 |
| 打不过零参数中心先验 | 0 | 0.0% | 29 | 32.6% | 0.0% | 0 |

> `命中占比` 是多标签，列和会超过 100%；`primary 占比` 和为 100%。`未测样本数`= 该样本上这个机制**无法测**（没有 oracle、没跑 field cache、或该 readout 天生 n/a）——**不是**「不存在」。

同一套判定跑在**全部 local 样本**上（含好样本）作对照。注意好样本上的机制标签本身没有意义——一个 softIoU 0.9 的大面积样本当然会有近似常数的 s（`s_collapse`）——这一列只用来回答「长尾里的机制是不是尾部特有」：

| 失败机制 | primary 计数 | primary 占比 | 命中计数 | 命中占比 |
| --- | ---: | ---: | ---: | ---: |
| s 轴塌缩（std(s)/std(s*) 过小） | 126 | 31.5% | 143 | 35.8% |
| 面积失配（pred_mean vs gt_mean） | 126 | 31.5% | 166 | 41.5% |
| 未归因 | 87 | 21.8% | 87 | 21.8% |
| generated context 质量（GT 好 / generated 差） | 38 | 9.5% | 38 | 9.5% |
| oracle 天花板低（basis 表达上界） | 11 | 2.8% | 11 | 2.8% |
| 打不过零参数中心先验 | 6 | 1.5% | 119 | 29.8% |
| s 场方向错（w_dir 与 oracle 夹角大） | 5 | 1.2% | 7 | 1.8% |
| s 错（换 oracle s 就修好） | 1 | 0.2% | 1 | 0.2% |

**瓶颈结论**：长尾里 primary 占比最高的机制是 **面积失配（pred_mean vs gt_mean）**（主榜口径 69.7%，辅助口径 65.0%）；未归因样本占 2.2%。

> `oracle_ceiling` 的阈值 0.7 是 **provisional**（主 agent 裁定 2026-08-10 暂用）：它没有标定依据，放松会把责任推给 basis、收紧则推给模型。任何依赖这一行的结论都必须同时写明这个阈值。

### 4.1 长尾样本清单

| sample_id | softIoU | hardIoU | 中心先验 | oracle | area/components/boundary/position/softness/region/winner_confidence/upscaled/build | primary 机制 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sft_9e2c10ec2fc4…` | 0.015 | 0.167 | 0.000 | 0.517 | tiny/single/complex/mid/soft/center/normal/False/l6 | oracle 天花板低（basis 表达上界） |
| `sft_440542d09386…` | 0.023 | 0.000 | 0.000 | 0.576 | tiny/single/compact/edge/soft/right/low/False/l5 | oracle 天花板低（basis 表达上界） |
| `sft_4fd87f21a752…` | 0.027 | 0.130 | 0.000 | 0.353 | tiny/multi/complex/mid/soft/center/normal/False/l5 | oracle 天花板低（basis 表达上界） |
| `sft_b2f508743028…` | 0.043 | 0.364 | 0.000 | 0.445 | tiny/multi/complex/mid/soft/lower/low/False/l5 | oracle 天花板低（basis 表达上界） |
| `sft_db86a642948d…` | 0.044 | 0.023 | 0.000 | 0.522 | tiny/multi/complex/edge/soft/lower right/low/False/l4 | oracle 天花板低（basis 表达上界） |
| `sft_b78fab3865c7…` | 0.055 | 0.318 | 0.115 | 0.540 | tiny/single/complex/mid/soft/lower/normal/False/l5 | oracle 天花板低（basis 表达上界） |
| `sft_9609c0ca0503…` | 0.057 | 0.000 | 0.000 | 0.787 | small/single/compact/edge/soft/right/normal/False/l1 | s 场方向错（w_dir 与 oracle 夹角大） |
| `sft_1d863dc54301…` | 0.057 | 0.333 | 0.132 | 0.479 | tiny/single/complex/mid/soft/lower/normal/False/l3 | oracle 天花板低（basis 表达上界） |
| `sft_35ec9c8dbc38…` | 0.058 | 0.191 | 0.273 | 0.610 | tiny/single/compact/center/soft/center/low/False/l1 | oracle 天花板低（basis 表达上界） |
| `sft_78f26362300a…` | 0.067 | 0.013 | 0.013 | 0.849 | small/single/complex/edge/hard/left/normal/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_bdca5c11f7e4…` | 0.069 | 0.013 | 0.013 | 0.849 | small/single/complex/edge/hard/left/normal/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_05b36cf36dab…` | 0.069 | 0.013 | 0.013 | 0.855 | small/single/complex/edge/hard/lower left/normal/False/l1 | s 错（换 oracle s 就修好） |
| `sft_3be5685578ee…` | 0.070 | 0.013 | 0.013 | 0.816 | small/single/complex/edge/hard/left/low/False/l5 | 面积失配（pred_mean vs gt_mean） |
| `sft_690e6624b8e9…` | 0.084 | 0.137 | 0.009 | 0.797 | small/single/complex/edge/soft/lower left/low/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_6a27ddcd8e19…` | 0.092 | 0.036 | 0.000 | 0.771 | small/single/complex/edge/hard/lower/normal/True/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_8b59235a1c65…` | 0.097 | 0.000 | 0.385 | 0.784 | small/single/complex/center/soft/center/normal/False/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_0e994a93216c…` | 0.098 | 0.000 | 0.000 | 0.921 | small/single/compact/edge/soft/right/low/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_ced1f1b69e2d…` | 0.103 | 0.374 | 0.000 | 0.738 | small/single/complex/edge/soft/lower/normal/False/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_80cc3e0911ed…` | 0.105 | 0.374 | 0.000 | 0.774 | small/single/complex/edge/soft/lower/normal/False/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_6ec6373df6d9…` | 0.107 | 0.388 | 0.000 | 0.658 | small/single/complex/edge/soft/lower/normal/False/l1 | oracle 天花板低（basis 表达上界） |
| `sft_941f07dcbc45…` | 0.107 | 0.133 | 0.532 | 0.894 | small/single/compact/center/soft/center/normal/False/l2 | 面积失配（pred_mean vs gt_mean） |
| `sft_6c212a9890cf…` | 0.107 | 0.388 | 0.000 | 0.487 | small/single/complex/edge/soft/lower/low/False/l5 | oracle 天花板低（basis 表达上界） |
| `sft_0e7bd082004f…` | 0.110 | 0.172 | 0.532 | 0.880 | small/single/compact/center/soft/center/normal/False/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_f9280808af9a…` | 0.112 | 0.133 | 0.518 | 0.882 | small/single/compact/center/soft/center/normal/False/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_9392222c7dc7…` | 0.127 | 0.051 | 0.000 | 0.670 | small/single/compact/edge/soft/lower right/low/False/l1 | oracle 天花板低（basis 表达上界） |
| `sft_9bb5598eeb22…` | 0.128 | 0.132 | 0.536 | 0.856 | small/single/compact/center/soft/center/normal/False/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_b434e56ede7e…` | 0.131 | 0.049 | 0.000 | 0.753 | small/single/compact/edge/soft/lower right/normal/False/l2 | 面积失配（pred_mean vs gt_mean） |
| `sft_cc3aa5b91f70…` | 0.133 | 0.056 | 0.000 | 0.716 | small/single/compact/edge/soft/lower right/normal/False/l5 | 面积失配（pred_mean vs gt_mean） |
| `sft_0d16e05f6811…` | 0.140 | 0.100 | 0.106 | 0.900 | small/single/compact/mid/soft/center/normal/False/l5 | 面积失配（pred_mean vs gt_mean） |
| `sft_30745d7326d2…` | 0.140 | 0.094 | 0.112 | 0.909 | small/single/compact/mid/soft/center/normal/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_b8b91f402f75…` | 0.145 | 0.103 | 0.368 | 0.818 | small/single/complex/center/soft/center/normal/False/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_f49b47f3d915…` | 0.146 | 0.120 | 0.078 | 0.856 | small/single/compact/edge/hard/upper/normal/False/l2 | 面积失配（pred_mean vs gt_mean） |
| `sft_cd79da4ed738…` | 0.147 | 0.145 | 0.000 | 0.845 | small/single/complex/edge/soft/lower/low/False/l6 | 面积失配（pred_mean vs gt_mean） |
| `sft_12fb83d345b6…` | 0.154 | 0.110 | 0.167 | 0.910 | small/single/complex/edge/soft/lower/low/False/l6 | s 轴塌缩（std(s)/std(s*) 过小） |
| `sft_c8c196c1a232…` | 0.155 | 0.384 | 0.104 | 0.907 | small/single/compact/edge/soft/lower/normal/False/l2 | 面积失配（pred_mean vs gt_mean） |
| `sft_4751eb7aeed3…` | 0.160 | 0.453 | 0.198 | 0.906 | small/single/compact/mid/soft/lower/normal/True/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_34b2d6eb76bd…` | 0.164 | 0.399 | 0.533 | 0.915 | small/single/compact/center/soft/center/low/False/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_3e7eac89df42…` | 0.165 | 0.074 | 0.085 | 0.782 | medium/single/complex/edge/hard/left/low/False/l1 | 面积失配（pred_mean vs gt_mean） |
| `sft_763bbca7d4e8…` | 0.168 | 0.396 | 0.521 | 0.928 | small/single/compact/center/soft/center/low/False/l4 | 面积失配（pred_mean vs gt_mean） |
| `sft_386090ac2077…` | 0.170 | 0.407 | 0.533 | 0.924 | small/single/compact/center/soft/center/low/False/l3 | 面积失配（pred_mean vs gt_mean） |

联图见 `viz/`（16 张）。每张固定为 `I_in | GT mask | GT 叠图 | pred mask hi | pred mask low | s 叠图`，缺的格子画成 “not available” 而不是省略。

## 5. 结论与建议下一步

- 长尾里触发 `area_mismatch` 的 87 个样本中 **87 个是过覆盖**（pred_mean > gt_mean），预测/GT 面积比中位 **3.69**——场倾向于铺开成接近全局的掩膜，而不是缩到指令所指的区域。
- `area` 维度上最差类别是 **small**（softIoU 中位 0.155），最好类别 large（0.630），跨类差 0.475。
- `boundary` 维度上最差类别是 **complex**（softIoU 中位 0.295），最好类别 compact（0.585），跨类差 0.290。
- `position` 维度上最差类别是 **edge**（softIoU 中位 0.270），最好类别 mid（0.599），跨类差 0.329。
- `softness` 维度上最差类别是 **soft**（softIoU 中位 0.387），最好类别 hard（0.596），跨类差 0.209。
- `region` 维度上最差类别是 **lower**（softIoU 中位 0.275），最好类别 center（0.583），跨类差 0.308。
- `winner_confidence` 维度上最差类别是 **normal**（softIoU 中位 0.448），最好类别 low（0.531），跨类差 0.083。
- `upscaled` 维度上最差类别是 **False**（softIoU 中位 0.471），最好类别 True（0.570），跨类差 0.099。
- `build` 维度上最差类别是 **l5**（softIoU 中位 0.443），最好类别 l3（0.558），跨类差 0.115。
- 以下维度在本 split 上**只有一个类别**，表里不出现：`topology`、`active_primitive_bucket`——不是模型在这些维度上没差别，是数据里没有对比。
- 长尾的 primary 机制以 **area_mismatch** 为首（69.7%）；全体 local 样本上同一机制占 31.5% —— 这是**尾部特有**的机制（全体样本上稀有得多）。
- 以下机制在**每一个**长尾样本上都无法测：`single_primitive`——瓶颈结论在这些机制上没有证据，不是它们被排除了。

## 6. 已知盲区

- **本工具只做单维度分层，不做全交叉**：六维全交叉是 144 格 / 400 个 local 样本。「small 且 multi」这种交互效应本报告读不出来，需要单独立项。
- **分类学只看 GT 几何**，不含语义（人 / 天空 / 建筑）。一个「所有小面积样本都是人脸」的混杂本报告识别不了。
- **归因的阈值是预注册的**（见 `config/thresholds.json`），但阈值附近的样本会在标签间抖动；`primary` 的顺序假定越上游的机制越该先修，这是一个工程判断，不是测量结果。
- **`s_direction` / `s_error` / `rho_error` / `single_primitive` 需要重跑 checkpoint**（`--checkpoint`）。没跑时它们不是 0，是 not_tested；把它们当 0 读会把瓶颈错误地推给 context。
- **环形（带洞）区域在评测数据里几乎不存在。** 2026-08-10 用同一套几何量普查了全部四个 eval split 的已发布 GT（`geometry_<split>.json`，CPU，只看 GT）：

| split | n(local) | 带洞 | 多连通 | max hole_frac |
| --- | ---: | ---: | ---: | ---: |
| V_where | 400 | 0 | 16 (4.0%) | 0.0061 |
| V_what | 408 | 0 | 14 (3.4%) | 0.0061 |
| T_final | 424 | 0 | 21 (5.0%) | 0.0042 |
| T_lut_unseen | 198 | **1** | 18 (9.1%) | 0.0617 |

  即：**1430 个掩膜里带洞的只有 1 个**（`sft_997b054f97c8c55e045b09f91b54545c`，在 T_lut_unseen，hole_frac 0.062 / circularity 0.139），`V_where`/`V_what`/`T_final` 三个 split **一个都没有**（其余样本的 hole_frac ≤ 0.006，是羽化边缘上的针孔，不构成拓扑洞）。所以协议 §13 「联图至少覆盖环形」这一条，在最终报告的 `T_final` 上**无法满足**——如实写明该类不存在，**不造样本**。**多连通可以满足**：`T_final` 自身有 21 个，最终 20 图从那里取即可。
- **local softIoU 是被直接优化的量**（`1 - softIoU` 是 L_mask 权重 1.00 的支配项），所以「哪一类最差」的排序里，它承担的是收敛度而不是独立证据；同一张表里的 grid 边界 F1 与中心先验 Δ 才是没被优化的列。详见同目录 eval 产物的 `ATTRIBUTION.md`。

> 分类学阈值的标定依据、核实记录、以及**待主 agent 决策的 6 项**，见 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/WEVAL1_NOTES.md`。
