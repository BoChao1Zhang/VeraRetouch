# STATUS — PR-13（PR-1 颜色探针 + PR-3 空间探针 / H3 判决图）

> 状态文件按 D-20 维护：每个后台作业的 PID / 命令 / 日志见同目录 `job.marker`。

## 当前阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| 0 | 文档节阅读 + 在线核实 + 待决策清单 → `NOTES.md` | ✅ 完成（8 条方法学/公式已开原文核实，8 条待决策项已列） |
| 1 | 工具链 `tools/probe/{colorops,vlm_features,probes}.py` | ✅ 完成（冒烟通过：D=186944，159 位点，峰值显存 1.64 GB） |
| 2 | 采样 `config/pr13_{color,spatial,behavior}.json` | ✅ 完成（S-val：颜色 400 源 / 空间 214 源（G1 同批）/ 行为 100 源） |
| 3 | 抽取：core（颜色主 run 4160 行 + 空间 run 642 行） | ✅ 完成（91.0 / 2.7 min） |
| 4 | 抽取：extra（C3 / C4）+ c2（C2a / C2b） | ✅ 完成（81.6 min，含各自 spatial 对照） |
| 5 | 行为五档 + **修正版** matched probe（左右半均值之差） | ✅ 完成（529 行；第一版整段平均口径已作废） |
| 6 | 分析（PR-1 按属性分 3 片并行 + PR-3 + 行为） | ✅ 完成 |
| 7 | 因果（擦除 + steering） | ✅ 完成；**擦除操作检查未通过**，steering 幅度比 3.95× |
| 8 | 图（28 张，含 H3 判决图 png+pdf）+ 判据表 | ✅ 完成 |
| 9 | `REPORT.md` | ✅ 完成 |

## 算力占用声明

- 卡 0，`CUDA_VISIBLE_DEVICES=0`；三个抽取进程各约 1.6–2.3 GB，合计 **< 6 GB**（上限 10 GB）。
- 未启动 dataloader worker；线性代数走 GPU，未调用 sklearn。
- **未 kill 任何非本任务进程**。

## 交付物清单（完成后逐项打钩）

- [x] `REPORT.md`（强制三行 + 预注册 vs 实测并排表 + 结论 + 下一步）
- [x] `metrics.json`（判据表 + evidence_class）/ `metrics_raw.json`（全部原始数字）
- [x] `viz/H3_verdict.{png,pdf}`（论文主图）
- [x] `viz/layer_curves_color.png` / `viz/readout_bottleneck.png` / `viz/g1_crossref.png`
- [x] `viz/success_*` / `viz/failure_*`（颜色探针与空间探针各若干）
- [x] `config/`（env.json + 样本清单 + 采样报告 + 日志）
- [x] `NOTES.md`（已完成）
