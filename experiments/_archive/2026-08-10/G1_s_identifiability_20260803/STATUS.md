# STATUS — G1 s 可辨识性（W1c）· **全部批次已完成，终版已交付**

## 当前状态（2026-08-03 11:xx，重聚合 agent）

| 批次 | 目标 | 实际 | 状态 |
|---|---|---|---|
| 主批（`run_smoke30` + `run_full`） | 300 源 × 3 指令 = 900 | 49 + 851 = **900** | ✅ 完成 |
| 区域对立批（`run_regsmoke20` + `run_regfull`） | 214 源 × 2 = 428 | 8 + 420 = **428** | ✅ 完成 |
| 跨图错位对照（`run_shufctrl`） | 60 | **60** | ✅ 完成 |

跑批进程全部自然退出（`pgrep` 空，GPU0 显存 0 MiB）。**无需再跑任何 GPU 批次。**

## 终版交付（已产出，勿再手工重跑）

- `metrics.json` —— 全批聚合终版（n=300 主批 / 214 区域 / 60 错位）。
  **`gate = FAIL`**（主判据 ρ_region_opp 0.884，判据 <0.3，0/214 达标）。
  含 D-22 两项（`cross_token_rho` / `subject_auc`）、层选择重扫（`layer_scan` 逐层 +
  `band_scan` 10 个区间）、分层报告（`by_task_conf`）、`shuffle_control` 配对读法。
- `REPORT.md` —— 终版判定表 + 分层报告 + 层选择结论 + 建议下一步（B1–B5）。
- `NOTES.md` §九 —— 本轮重聚合的核实记录、掩膜数据源溯源、viz 命名诚实性修正、待决策 D9–D11。
- `viz/` 24 张 —— `success_*` / `failure_*`（主批各 6）、
  **`region_bestcase_*`**（区域对立最分离 6 例，**全部仍 FAIL**，故不叫 success）、
  `region_failure_*`（最不分离 6 例）。

## 复现命令（幂等；纯 CPU，约 10 min，不占 GPU）

```bash
cd /home/bc/VeraRetouch/experiments/G1_s_identifiability_20260803
/home/bc/VeraRetouch/.venv-lens/bin/python analyze_g1.py \
  --run-dirs run_smoke30 run_full run_regsmoke20 run_regfull run_shufctrl \
  --out . --viz-n 6
```

> ⚠️ **并发注意**：该命令会覆写本目录的 `metrics.json` 与 `viz/*.png`。
> 2026-08-03 11:09 观察到有外部会话用旧 STATUS 里的相对路径版本并发跑了一次，
> 与本 agent 的 nohup 批撞车（`viz/` 一度被清空）。再跑前先 `pgrep -af analyze_g1` 确认没有在跑。
> 依赖：`/var/cache/veradata/global.sqlite3`（掩膜索引）+ `/mnt/nfs/bc/data/datasets/cache/subject`
> （SAM3 主体掩膜银行）。银行不可达时加 `--no-subject` 降级跳过 D-22(b)，其余指标照算。

## 历史记录（存档）

- gflow 不可用 → 回退 nohup：`gbatch` job 2（exclusive 1 GPU）永远排不上，因两卡均被
  unmanaged 生产进程占用（GPU0 prod-g4；GPU1 prod-l5/l6 + W1a viz），已 gcancel。
- 跑批 PID（均已退出）：主批 3228455 ／ 错位对照 3232359 ／ 区域全量 3279260。
- 吞吐告警（02:3x）：t_gen 11→45 s/样本，归因 **CPU 过载**（load 65 / 48 核，
  prod-l5/l6 build agent + E1/A0/run_fit 工作池），非 GPU。已随 prod agent 退场回落，
  三批最终全部跑完。
- 02:50 曾产出 n=41 的中间版 metrics.json（`shuffle_control`/`region_opposition` = null，
  `gate = PENDING`）——**已被本轮终版覆写，勿再引用**。

## 终态（2026-08-03 11:2x，终版归档）

- 三批全部落地并自然退出：主批 900/900、区域对立批 428/428、跨图错位对照 60/60（**1388 次读出**）。
- 终版聚合完成：`metrics.json` **gate = FAIL**（D-30）；`REPORT.md` / `REVIEW-result.md` 终版已交付；
  `viz/` 27 张（含 3 张 `failure_instrblind_*` 指令无关性主图，区域批 12 张已改判据化命名）。
- **无在途任务**；GPU 0/1 本轮全程未占用（纯 CPU 分析侧）。`job.marker` 已标 finished。
