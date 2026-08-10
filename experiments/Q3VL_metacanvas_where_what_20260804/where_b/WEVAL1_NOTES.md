# WEVAL-1 实施记录（Where 专属分类评估与长尾瓶颈分析工具）

交付物：`q3vl/whereb/analysis/`（新子包）+ `analysis_W01_step1500/` / `analysis_W02_step1500/`。
本文件是任务卡要求的「实施前三件事」记录：读了哪些章节、核实了什么、哪些是假设、哪些要主 agent 拍板。

---

## 1. 读的文档章节（只读引用节，未读全文）

| 文档 | 章节 | 用到的东西 |
|---|---|---|
| `CLAUDE.md` | AUC 全实验禁用 / 空间场可视化纪律 / 红线速查 / s 缓存消费契约 / 长任务提交纪律 / NFS 访问纪律 | 全部当硬约束，见 §4 |
| `docs/METACANVAS_..._2026-08-04.md` | §5.6（gate 与选择规则）、§13（可视化与中文报告交付）、§13.1（必交文件） | 判据列的定义、联图必须覆盖「小/中/大 mask、环形/细边界/多连通区域」、至少 5 个最差案例分类 |
| `docs/DATA_ASSIGNMENT_2026-08-02.md` | S-split 定义、探针类实验一律 S-val | `V_where` 的性质：S-val 源，永不进训练 |
| 代码即规格 | `q3vl/whereb/metrics.py`、`evaluate.py`、`viz.py`、`fields.py`、`config.py` | 见 §2 |

§13 明确要求联图覆盖「小/中/大 mask」和「环形/细边界/多连通区域」，本工具的分类学正是这句话的可执行版本。

## 2. 核实记录（**没有使用任何检索引擎，没有引用任何外部 URL**）

本任务用到的外部事实只有一个——离散周长估计子的权重表——而它**不是引用来的，是实测标定的**（见下）。其余全部是仓库内事实，逐条在本机核实：

| 事实 | 怎么核实的 | 结果 |
|---|---|---|
| 周长估计子（marching-square 加权）正确 | 在合成圆（r=60）/方（120²）/环上直接量，与解析值比 | 圆 393.99 vs 2πr=376.99（**+4.5%**）；方 476.00 vs 480（−0.8%）。裸「裂缝计数」在圆上是 8R=+27%，各向异性大一个量级，故不用。三个数字钉进 `test_analysis_taxonomy.py` |
| circularity 的类别切点 | 用同一估计子在 400 个真实 GT 上量分布 | p10/p50/p90 = 0.345/0.680/0.832；切点 0.50 → complex 占 24% |
| `.cgt` GT mask 的幅值不是 1 | 读 400 个 `.maskhi.png` 量 max | **min 0.561 / p50 1.000**。故 (a) 二值化 0.5 安全，(b) 软边比例必须相对各自 max 量（否则量的是幅值不是边宽） |
| `V_where` 的 GT 有没有洞 | 400 个掩膜逐个填洞比对 | **0 个有洞**。`topology` 维度在本 split 上失效，报告显式声明「不是没差别，是数据里没有对比」 |
| `summarise()` 是主榜同一个聚合器 | 直接调用 `q3vl.whereb.metrics.summarise`，并写单测比对子集 | 分层数字与主榜同源，不可能漂 |
| soft-IoU 是 min/max 形式 | `config.SOFT_IOU_KIND == "minmax"`，本工具不自己算 IoU | 不涉及积形式 |
| top-k 匹配 GT 面积是唯一二值化规则 | 本工具不做任何二值化；单测反证「逐场 0.5 阈值」会给出不同（更高）的数 | `test_matched_area_topk_is_what_the_hard_iou_column_measures` |
| `s` 的值域 | `fields.s_from_params`：`s = S_SCALE*tanh(q/S_SCALE)`，`S_SCALE=3` | 可视化用**固定** (−3, 3) 色标，不做任何逐图归一化 |
| F_pre 网格与像素网格是整数倍 | `q3vl.where.fpre.grid_from_geometry`：`(H/16, W/16)`，且要求整除 | `grid_to_img` 的严格逆映射成立；且**没有 pad 格**（不走 expand2square），故 `allow_all_valid=True` 是有依据的声明而不是绕过 |
| `hi_lo_soft_iou_drop` 的符号 | `metrics.sample_metrics`：`grid_soft_iou − soft_iou` | 正 = hi 档更差，归因里按此读 |
| checkpoint 格式 | `trainer.save`：`{"model", "step", "basis_digest", ...}` | field cache 读它，并**断言 basis digest 与当前 basis 一致**，不一致直接报错 |
| oracle 掩膜的重算路径 | `evaluate._oracle_mask` | field cache 抄同一条路径（同 `phi_dir`、同一次 guided upsample、float32） |

## 3. 假设与待确认清单

### 3.1 自行核实后定下的（不需主 agent 拍板）

1. **global 样本不进几何分层**。它们 GT 全 1，496 行接近满分；混进任何几何格都会淹没 local 信号。报告显式写明。
2. **软边比例按各掩膜自身 max 归一后再量**。理由见 §2 第 3 行：`.cgt` 的幅值是候选置信度而不是几何。这是对 **GT 的分类**，不是对被打分场的归一化，不触碰红线；两种口径（相对 / 绝对）都存在 `geometry` 里，绝对口径就是 Where-A `maskmeta.mask_stats.frac_soft` 的定义。
3. **`pubio.py` 自带一个极小的 published-store 读取器**，不走 `q3vl.whereb.stores`。原因是实施期间 PERF-1 正在重写 `q3vl/train/shards.py` + `stores.py`，工作区一度处于 `resolve_read_path` 抛 `TypeError: 'function' object is not subscriptable` 的中间态（后已修好）。字节级读仍然委托给未被改动的 `q3vl.data.shardio.read_member`（含 sha256 校验），索引解析仍走 `ShardIndex`——没有重复实现任何格式。
4. **不改任何既有文件**（任务卡要求）。因此 `evaluate.py` 的 `active_primitive_bucket` 空洞（见 3.2-e）只被**报告**，没有被就地修掉。

### 3.2 待主 agent 决策（已采用保守默认继续，未静默拍板）

| # | 事项 | 现在采用的保守默认 | 备选与代价 |
|---|---|---|---|
| a | **面积分档切点** | 任务卡写的是 small<5% / medium<25% / large>25%；实测 279/400 落进 large，一个装 70% 样本的格回答不了「哪种区域难」。默认改为 **四档 (0.05, 0.15, 0.45)**，**保留 <5% 这条预注册边界作为独立的 `tiny` 类** | 若要严格按卡面两切点，`--taxonomy` 层面改一个常量即可；代价是 large 格失去分辨力 |
| b | **长尾集合的口径** | 统计口径 = 主榜 softIoU < **0.30** 的全部 local 样本（W01: 89 个 / 22.2%）；只有出图取最差 12 个 | 备选「按分位取 bottom 10%」。0.30 是预注册常数，随 arm 变强会自动收缩，分位口径则恒定 —— 两者哪个更适合跨 arm 比较，需要定 |
| c | **`oracle_ceiling` 阈值 0.70** | 该样本自己的 Where-A oracle softIoU < 0.70 才算「basis 表达上界」 | 更松（0.8）会把大量样本归给 basis 从而低估模型责任；更紧（0.6）反之。W01 长尾里它占 12.4%，对阈值不算敏感 |
| d | **归因优先级顺序（上游优先）** | `format_failure > oracle_ceiling > context_quality > s_direction > s_error > rho_error > s_collapse > single_primitive > upsample_collapse > area_mismatch > below_center_prior` | 这是工程判断不是测量结果，报告 §6 已声明。多标签命中率同时给出，改顺序只影响 `primary` 列 |
| e | **`active_primitive_bucket` 在 eval 侧是空的** | `metrics.active_primitive_count/_bucket` 有定义有单测，`config.EXTRA_STRATA_KEYS` 也列了，但 `evaluate.evaluate_context` **从未把它写进 per-sample 行**——所以每块已发布的板子里 `strata.active_primitive_bucket` 只有一格 `{"None": 896}`（W01/W02 step1500 实测如此）。本工具在 `--checkpoint` 模式下自己算（那是唯一有预测 `rho` 的地方），只覆盖被重跑的样本 | 建议派一个 1 行的修复：在 `evaluate.py` 的 `met.update({...})` 里加 `active_primitive_bucket`。**Where-A 继承来的单基元脆弱性风险目前在所有 Where-B 板子上都没有实际数据支撑** |
| f | **是否把本工具挂进 `run_where_b.py` 的 eval 回调** | 没挂。现在是离线工具，对任意已落盘 eval 目录可跑 | 挂进去可以每 500 步自动出分层报告，代价是训练进程里多一次 NFS 掩膜扫描（首轮 ~15 s，之后走缓存） |

## 4. 红线自查

| 红线 | 本工具的落实点 |
|---|---|
| AUC 全实验禁用 | 所有指标列由 `summarise()` 产出，它不产 AUC；`report.render_report` **主动拒绝**任何 key 含 `auc` 的表（单测 `test_report_refuses_an_auc_column`） |
| 空间场三列缺一不可 | 每张分层表都带 hardIoU + grid 边界 F1 + 中心先验列与配对 Δ、p；`render_report` 拒绝渲染缺中心先验列的表（单测） |
| top-k 匹配 GT 面积 | 本工具不做二值化，列直接来自按该规则算好的 per-sample 行；单测反证逐场阈值会给出不同的数 |
| 禁逐图 min-max 着色 | 掩膜固定 (0,1)、`s` 固定 (−3,+3)。**没有任何一处从被画的那张图里算色标端点** |
| 色标只取有效格 / pad 显式画出 | F_pre 网格无 pad 格（`grid_from_geometry` 直接由图像尺寸导出），以 `allow_all_valid=True` 显式声明，不是传 `valid=None` 蒙混 |
| 叠图禁直接 resize | 走 `viz.grid_to_img` 的整数最近邻严格逆映射；全包无 `resize` 调用 |
| 着色归着色、算数归算数 | `panels.py` 不产生任何指标，只接收调用方算好的数 |
| NFS 读走 nfs-ro | 掩膜与图像都读 `/mnt/nfs-ro`（`WHERE_A_MASKVIEW_DIR` / `rewrite_read_path`）；全程 `nfsx` 包裹；不写 NFS |
| sqlite3 必须在 torch 之前 | 入口 `run_analysis.py` 首行 `import sqlite3`；库模块里**没有**放这个 guard（放了会让库在 torch-first 进程里不可导入） |

## 5. 实施期间发现、但不属于本任务范围的问题

1. **`evaluate.py` 没有写 `active_primitive_bucket`**（见 3.2-e）。影响：Where-A 单基元脆弱性这条已知风险，在 Where-B 侧目前是**零证据**状态。
2. **PERF-1 与本任务并发**：期间 `q3vl/train/shards.py` 出现过 `_cache_state` 函数名与模块级备忘变量同名导致 `resolve_read_path` 返回函数对象、所有 published-store 读全挂的中间态。再次核查时已修复（本机 `/home/bc/data/shard_cache` 已有 23 个条目 / 29.6 GB）。记录在此仅为存档。
3. **`V_where` 的 `.cgt` 掩膜没有一个带洞**，且 96% 是单连通。§13 要求联图覆盖「环形 / 多连通」，**本 split 覆盖不了环形**；若那条交付要求要满足，需要从别的 split 取样或承认该类不存在。

## 6. 复现

```bash
# 无 GPU（约 30 s，首轮多 15 s 冷读掩膜）
nfsx 900 -- /home/bc/envs/q3vl_sft/bin/python -m q3vl.whereb.analysis.run_analysis \
    --eval-dir /home/bc/data/runs/where_b/W01/eval_step1500 --arm W01 --step 1500 \
    --out-dir experiments/Q3VL_metacanvas_where_what_20260804/where_b/analysis_W01_step1500 \
    --geometry-cache experiments/Q3VL_metacanvas_where_what_20260804/where_b/geometry_V_where.json

# 追加长尾样本的场（GPU，约 50 s，16 个样本 batch=2，不干扰训练吞吐）
... --checkpoint /home/bc/data/runs/where_b/W01/where_b_step1500.pt --device cuda --field-batch 2

# 单测（CPU）
/home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whereb/tests/test_analysis_taxonomy.py \
    q3vl/whereb/tests/test_analysis_attribution.py -q      # 47 passed
```

`geometry_V_where.json` 是**跨 arm 共享**的：几何只是 split 的属性，八个臂读同一份缓存。
