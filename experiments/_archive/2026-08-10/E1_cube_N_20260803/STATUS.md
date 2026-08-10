# E1 STATUS（2026-08-03）

## 长任务

- **进程**：nohup PID **3234243**（父 bash；子进程为 python3 -m model.glut_repro.run_e1）
- **日志**：`experiments/E1_cube_N_20260803/logs/e1_full.log`
- **内容**：400 个 minor 分层 LUT（config/e1_subset_400.txt，77/77 minor 覆盖）× N∈{8,16,24,32,48,64,96,128}，batch-luts=16（25 chunk/N × 8N = 200 chunk）
- **预计时长**：GT 缓存 ~20 min（首次，400×4M 点四面体，12 workers）+ 每 chunk 1–3 min（N 大者慢）≈ **共 6–9 h**
- **断点续跑**：`runs/main/per_fit.jsonl` append-only，重启同命令自动跳过已完成 (LUT,N) 对
- **检查命令**：
  ```
  tail -f experiments/E1_cube_N_20260803/logs/e1_full.log
  wc -l experiments/E1_cube_N_20260803/runs/main/per_fit.jsonl   # 进度 / 3200
  jq '.per_n, .r_of_n, .n_star' experiments/E1_cube_N_20260803/runs/main/metrics.json  # 每个 N 扫完后增量更新
  ps -p 3234243   # 存活；日志末尾 "E1_FULL_DONE" = 完成
  ```
- **完成后待办**（下一棒）：REPORT.md（p50/p90/p99/max 全表 + r(N) 饱和判定 + N\* + 门槛 p50<0.5/p90<1.0/p99<2.0 判读 + 尾部最差 5% 归因【PLAN 步骤 7】）、成功/失败 viz 按全量结果重出（viz/ 现为冒烟版；失败例用 `python -m model.glut_repro.viz e1-refit`）、NOTES §三待决策项（全量留出色复验 / 自然图 / 3,522 全量 3N 验证）报主 agent。

## 排卡

- 本任务用 **GPU0**（CUDA_VISIBLE_DEVICES=0）。提交时 GPU0 ~5GB 被其他 agent 占用，本任务峰值 <8GB（N=128 chunk），共存无压力。
- gflow 回退原因同 A0 STATUS（两卡 unmanaged 占用，gflow job PD 不调度；已实测后 cancel）。

## GT 缓存

`/var/cache/veradata/dcube/gtcache/e1/`（float16，~9.6GB，可删可重建）。

## 冒烟基线（详见 NOTES §四 与 smoke/）

N=8: p50 1.58 / p90 2.12；N=32: p50 0.75 / p90 0.95 —— N=32 已过 p90 门槛（子样口径、仅 5 LUT，勿外推）。
