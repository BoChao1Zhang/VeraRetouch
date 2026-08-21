# NOTES · 任务卡 D1（P2 可见性指标族实现 + 历史回填 + 人工标注验证）

规格依据：`docs/DESIGN_databuild_v2_20260820.md` §2.8 S2.1 / S2.7。
本卡**不接线**：未改校准目标、未改任何门、未 bump revision、未改 `graph.py` / `render.py` / `models.py`。

## 1. 落盘文件

| 路径 | 内容 |
|---|---|
| `dataset_build/agent_loop/visibility.py` | 指标族实现（新模块，无调用方） |
| `dataset_build/tools/backfill_visibility_metrics.py` | `backfill` / `validate` 两个子命令 |
| `dataset_build/tests/test_visibility.py` | 14 个新单测 |
| `/home/bc/data/scratch/visibility_metrics/chains.jsonl` | 回填结果，1565 行 |
| `/home/bc/data/scratch/visibility_metrics/validation.json` | 验证报告 |
| `/home/bc/data/scratch/visibility_metrics/joined.jsonl` | 554 条 join 后的逐条记录 |

## 2. 预注册常量（`visibility.py` 顶部，不得逐图调）

```
VISIBILITY_CONTRACT   = "visibility-support-sampled-de00-v1"
VISIBILITY_TAU        = 0.05
EDGE_BAND             = [0.20, 0.80]
SAMPLE_BUDGET_MAX     = 4096      # 每个区域单次抽样上限
SAMPLE_PIXELS_IN      = 2048      # 支撑集内
SAMPLE_PIXELS_OUT     = 2048      # 支撑集外
SAMPLE_PIXELS_EDGE    = 1024      # 过渡带
DE_IN_QUANTILES       = (0.50, 0.90)
EDGE_STEP_QUANTILE    = 0.95
```

## 3. 假设与保守默认（未静默拍板，列此待用户裁决）

1. **用哪张 alpha**。任务卡写「applied_alpha mask」。实测：`applied_alpha` blob 在 iter2/3/4
   committed 叶子里缺 11 条，而 `render_record.input_json.mask_sha256_or_global`（原始 mask alpha）
   1565 条全在。且现行 `de_masked`（`render.py:181-184`）的权重就是**原始 alpha**，不是 applied。
   保守默认：**用原始 mask alpha** 做支撑集判定与加权，口径与 `de_masked` 对齐，
   使 `de_in` 与 `de_masked` 之间只差「采样集」这一项。`applied_alpha = alpha × local_strength`，
   `local_strength` 已作为列落盘，需要时可复算。
2. **每区域抽样条数**。任务卡写「全部在 ≤4096 采样点上算」。理解为**每个区域一次抽样 ≤4096**；
   实际取 in=2048 / out=2048 / edge=1024（随机地板另抽 in=2048 / out=2048）。
   若用户要求「全部区域合计 ≤4096」，改 `SAMPLE_PIXELS_*` 三个常量即可。
3. **随机地板掩膜形状**。S2.1 只写「同 support_frac、位置随机、seed 固定」，未指定形状。
   保守默认：**与整图同长宽比的轴对齐矩形**，面积 = `support_frac × H × W`，
   整体落在图内（面积精确，无裁剪损失），左上角由 `sha256(contract|branch_id|"floor")` 抽。
   若 `floor_de_in == 0`（随机框完全落在无编辑区），`de_in_over_floor` 记 `None`，
   同时给出配对差列 `de_in_minus_floor`（CLAUDE.md 要求任何定位主张出示配对 Δ）。
4. **`edge_de` 的「p95 单像素阶跃」**。落为 `edge_step_p95`：对过渡带每个采样像素，取其与
   右邻 / 下邻 ΔE00 之差绝对值的较大者，再取带内 p95。图像边界像素邻居 clamp 到自身。
5. **`de_in_p50/p90` 为不加权分位数**（加权分位数需插值口径，未预注册，故不用）。
   `de_out` / `edge_de` / 地板列均为**不加权均值**（该区域 alpha 权重无定义或恒 1）。
6. **iter2 的 global 父分支不在同 campaign**（`agent_branch` 里 iter2 只有 local 层）。
   因此 `global_after` 不走 `global_branch_id` 跨表 join，改走
   `render_record.input_json.input_image_sha256` —— 那是渲染时**实际喂进去的 before**，
   口径无歧义。
7. **问卷 CSV ↔ 轮次映射**由落盘 provenance 逐条核对（非猜测）：
   - `docs/assets/questionnaire/intentq.csv`（199 rated）↔ `intent_quality_200/item_key.json`
     ↔ campaign `local-v2-iter2`（`docs/assets/questionnaire/analysis.json` 的 `csv`/`item_key` 字段）
   - `intentq (1).csv`（156）↔ `intent_quality_v2/item_key.json` ↔ `local-v2-iter3`
     （`intent_quality_v2/analysis.json:csv` 逐字为 `docs/assets/questionnaire/intentq (1).csv`）
   - `intentq (2).csv`（200）↔ `intent_quality_v3/item_key.json` ↔ `local-v2-iter4`
     （`intent_quality_v3/analysis.json:csv` 逐字为 `docs/assets/questionnaire/intentq (2).csv`）
   合计 555 条已填评分。`intent_quality_recheck` 是复检轮，**未纳入**（任务卡只点名三轮 intent 问卷）。

## 4. blob 缺失

- 候选 blob 根（按序查找）：`/mnt/ramstage/agent_loop/local-v2-b15-smoke/blobs`、
  `/mnt/ramstage/agent_loop/local-v2-pilot/blobs`、`/mnt/ramstage/agent_loop/local-v2-b11-smoke/blobs`、
  `/home/bc/data/agent_loop/local-v2-pilot/blobs`、`/home/bc/data/agent_loop/a6-lane-smoke/blobs`。
- 1565 条 committed 叶子中 3 条 `final_after` blob 缺失 → `status = blob_missing`，仍写行、计数。
- 555 条标注里 1 条（r2/iter3）因此无指标 → `unmatched.r2_iter3:no_metrics = 1`，join 得 554。

## 5. 确定性

- 同一条链两次运行、且 `--workers` 从 12 改为 6，`chains.jsonl` **逐字节相同**（已实测）。
- 输出行按 `(campaign_id, branch_id)` 排序后写出，与 worker 完成顺序无关。

## 6. 复现命令

```bash
python3 dataset_build/tools/backfill_visibility_metrics.py backfill --workers 12 \
  --out /home/bc/data/scratch/visibility_metrics/chains.jsonl

python3 dataset_build/tools/backfill_visibility_metrics.py validate \
  --round "r1_iter2=docs/assets/lut_cluster_pilot_20260819/intent_quality_200/item_key.json=docs/assets/questionnaire/intentq.csv" \
  --round "r2_iter3=docs/assets/lut_cluster_pilot_20260819/intent_quality_v2/item_key.json=docs/assets/questionnaire/intentq (1).csv" \
  --round "r3_iter4=docs/assets/lut_cluster_pilot_20260819/intent_quality_v3/item_key.json=docs/assets/questionnaire/intentq (2).csv" \
  --out /home/bc/data/scratch/visibility_metrics/validation.json \
  --joined-out /home/bc/data/scratch/visibility_metrics/joined.jsonl
```

## 7. 待用户决定（本卡不做）

- 是否把该列族接进 `render.py` 的渲染记录 / `graph.py` 的 commit 排序键（S2.2 / S2.3）。
- `SAMPLE_PIXELS_*` 与随机地板形状是否改口径（见 §3.2 / §3.3）。
- 是否把 `intent_quality_recheck` 的 157 条也纳入验证。
