# STATUS — RO-2 · logit lens 概率场 + A2 指令前置对照 · **已交付**

| 项 | 值 |
|---|---|
| 实验编号 | EXPERIMENTS_v3 §2.1 **RO-2**；A2 = 主 agent 本轮拍板的指令前置对照 |
| 分支 / commit | `lens-exp` / `metrics.json → git_commit`、`config/git_commit.txt` |
| 卡 / 显存 | **GPU 1**，实测峰值 reserved **3.48 GB**（预算 ≤12 GB；启动时卡1 已用 68.9/97.9 GB）。**未 kill 任何非本人进程** |
| 导出 | image_first 925/925 **0 error** 46.4 min；instr_first 925/925 **0 error** 53.0 min |
| 产物根 | `/home/bc/data/ro2_lens_20260803/`（3.9 GB）+ `/home/bc/data/ro2_lens_instrfirst_20260803/` |
| **RO-2 判定** | 形式 `promote=true`，**实质淘汰 → RO-5**（主 agent B1 裁定：以实质为准；A3 采纳判别比>1） |
| **A2 判定** | **prompt 顺序被排除**：指令确实进了 image token 表示（1−ρ ×146，p=2.3e-4），但 12 条读出判据全在 ±0.006 内不变 |
| **首位结论** | 词特异的空间信息在**进 LLM 之前**（mm_projector 输出）就存在，被 LLM 前向逐层抹掉（GT 掩膜 AUC 0.829 vs 无关词 0.523） |

## 阶段（全部完成）

- [x] 模型内部实测核实（**4 条与 DOSSIER §5.3 不一致**）+ 外部来源核实
- [x] image_first 全量导出 / 主分析 / 无关词对照 / 可视化 / scache
- [x] **scache 埋雷修复**：两个 arm 的 `meta.norm` 补 `domain` / `domain_p01_p99` / `median` /
      `domain_note` / `recommended_map`（线性）/ **`recommended_map_log`（重尾推荐档）** / `caveat`
- [x] **A2 指令前置**：预注册（NOTES §十，落地前写）→ 925 作业导出 → 主分析 → 无关词对照 → 配对 ρ 比较
- [x] **A2 行为退化对照**：24 源 × 两顺序 greedy 生成，合法性口径取自 `_generate` 源码
- [x] **主图**：`viz/word_specificity_erasure.png`（词特异性逐层抹除曲线 + 抹除机制分解 + A2 叠加）
- [x] REPORT.md（§十 A2 单独成节；结论已按主 agent 要求把 emb 发现提到首位）

## 交付清单

```
experiments/RO2_logitlens_20260803/
  REPORT.md                              强制三行 + 判定表 + §二词特异性 + §四刻度 + §十 A2
  metrics.json                           image_first 主判据（含 word_specificity / d31_final_reading）
  metrics_instrfirst.json                **A2** 同口径主判据
  metrics_supp_nosoftmax.json            无关词对照 × 概率/原始 logit × 三个集合
  metrics_supp_nosoftmax_instrfirst.json **A2** 同上
  metrics_a2_instrfirst.json             **A2** 配对 ρ 比较（含同顺序 dup/pad 数值地板）
  metrics_behavior.json                  **A2** 行为退化对照（24 源 × 两顺序，含逐源生成文本）
  NOTES.md                               实施前核实 8 条 + 待决策 A1–A5/B1–B2 + §十 A2 预注册 + 跑数后追记
  STATUS.md / job.marker
  analyze_ro2.py supp_nosoftmax.py compare_a2.py behavior_check.py
  ro2_viz.py ro2_toptokens.py fig_erasure.py write_scache.py
  config/  build_jobs.py ro2_jobs.json(925) ro2_jobs_report.json ro2_samesem.json
           introspect_model.json export_meta.json env.txt git_commit.txt
  logs/    export.log export_instrfirst.log analyze.log analyze_instrfirst.log
           supp.log supp_instrfirst.log compare_a2.log behavior.log viz*.log a2_chain.log
  viz/     word_specificity_erasure.png(主图) layer_curves.png scale_scatter.png
           toptokens_*(3) success_L0_*(6) failure_L0_*(6) success_emb_*(6) failure_emb_*(6)
tools/readout/ro2_logit_lens.py           导出工具（含 --prompt-order 与顺序守卫）
/var/cache/veradata/scache/ro2-lemb       **推荐接入**（437 条，唯一词特异读出点）
/var/cache/veradata/scache/ro2-l0         437 条（仅供复核判据数字）
```

## 待主 agent 决策（剩余）

NOTES §七 A1（同语义 200 图构造方法追认）/ A4（spatial 源补集词）
+ §9.4 B2（是否要求一组 ≥8 个无 referent 对照词）
+ 新增 C7/C8/C9（关闭 prompt 顺序解释线 / 指令条件性升格为训练目标 / 前置顺序若保留须复测合法率）
