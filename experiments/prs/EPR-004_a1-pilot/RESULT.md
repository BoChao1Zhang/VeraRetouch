几何码捕获率 macro-F1（shape / dir）: 无先例基线 → **shape 0.7467 / dir 0.0832**（shape+dir 合并 0.3781；n=500，无配对检验，非配对设计故 p 不适用）

## 判据表

| 判据 | 预注册 | 实测 | 结果 |
|---|---|---|---|
| R_json 解析失败率 | < 0.5% | **0.0%**（500/500 解码+解析成功） | **PASS** |
| R_trunc 截断率 | < 0.5% | **0.0%**（缓存 span 无截断标记） | **PASS** |
| 覆盖率 | ≥ 70% | **100%**（500/500 有 genctx 且有 vrmeta） | **PASS** |
| 弃权率（码全零） | 报告项 | **0.0%** | 报告 |
| capture_shape macro-F1 | 无预注册门 | **0.7467**（4 槽计分） | 报告 |
| capture_dir macro-F1 | 无预注册门 | **0.0832**（5 槽计分） | 报告 |
| capture_ext | 不可测（GT 无 extent） | 6 槽 GT 恒 0，未计分 | **N/A（按预注册）** |
| AUC | 禁用 | 未产出 | **PASS** |

## 指标明细

**shape 分组**（GT 未点亮的 `shape_oval` 已从宏平均剔除）

| 槽 | support | P | R | F1 |
|---|---|---|---|---|
| shape_semantic | 72 | 1.000 | 1.000 | **1.000** |
| shape_radial | 119 | 0.769 | 0.672 | 0.717 |
| shape_linear | 154 | 0.625 | 0.649 | 0.637 |
| shape_band | 155 | 0.495 | 0.877 | 0.633 |

**dir 分组**（`dir_edge/horizontal/vertical/diagonal` GT 恒 0，已剔除）

| 槽 | support | P | R | F1 |
|---|---|---|---|---|
| dir_center | 409 | 0.712 | **0.181** | 0.288 |
| dir_bottom | 54 | 0.059 | 0.074 | 0.066 |
| dir_left | 20 | 0.028 | 0.100 | 0.044 |
| dir_right | 19 | 0.011 | 0.053 | 0.018 |
| dir_top | 3 | 0.000 | 0.000 | 0.000 |

**集合级一致性**

| 量 | 值 |
|---|---|
| shape 集合完全一致 | **0.420** |
| GT 方向 ⊆ 解析方向 | 0.162 |
| 解析至少点亮 1 个方向 | 0.856 |
| 平均点亮方向槽数 | 2.23（GT 恒为 1.0） |
| 平均点亮总槽数 | 4.19（GT 2.01） |
| extent 槽激活率（解析侧） | 0.524（GT 侧无此字段） |

## GT 码本身的分布（同 500 样本，决定上界口径）

| 字段 | 分布 | 联合熵 |
|---|---|---|
| `slot_id` → shape | band 155 / linear 154 / radial 119 / semantic 72 | **1.943 bit** |
| `region` → direction | **center 409** / lower 49 / left 20 / right 14 / lower-right 5 / upper 3 | **1.006 bit** |
| 合计 | — | **≈2.95 bit/样本** |

## 口径声明

- 数据：local train，n=**500**，`seed=20260812` 无放回抽样，`exclude_low=True`，render_mode=local；
  **探针档**（全量为 42,752）。
- 文本：模型**自生成**的 `<where>` span（`GenContextStore(train)` 缓存），非 GT 文本。
- 度量：分组 macro-F1，阈 0.5，**GT 中无支撑的槽从宏平均剔除**（不计为满分）。
- 非配对设计（单臂对 GT 打分），故无配对 Δ 与置换 p。
- 无步数匹配问题（不涉及训练）。

## 交付物

- `metrics.json`（= `a1_pilot_stage1.json`）、`per_sample.jsonl`、`cases.json`（4 失败样例全文）
- 脚本：`q3vl/whereb/scripts/run_a1_pilot.py --stage 1`
