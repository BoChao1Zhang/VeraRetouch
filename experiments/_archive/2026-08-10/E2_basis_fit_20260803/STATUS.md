# E2 全量长任务状态

- **进程**：`run_full.sh`（nohup 后台），**PID 3248254**，启动 2026-08-03（本回合）。
- **日志**：`experiments/E2_basis_fit_20260803/full_run.log`
- **链路**：prep_data.py（GPU0，CLIP 特征 300 图 + 掩膜 1020 张，约 12 min）
  → run_fit.py（36 CPU workers，约 4,440 个 L-BFGS 拟合，约 50–80 min）
  → analyze.py（判据表 + metrics.json + 全部 viz，约 5 min）。
- **预计总时长**：约 1.5 h。
- **完成标志**：出现文件 `FULL_RUN_DONE`；`metrics.json`（无 _smoke 后缀）+ `viz/*.png` 即为全量产物。
- **检查命令**：
  ```
  tail -5 experiments/E2_basis_fit_20260803/full_run.log
  ls experiments/E2_basis_fit_20260803/FULL_RUN_DONE 2>/dev/null && echo DONE
  jq .criteria experiments/E2_basis_fit_20260803/metrics.json   # 完成后
  ps -p 3248254   # 存活检查
  ```
- 冒烟规模（_smoke 后缀文件）已全链路验证通过；全量仅是规模放大，无代码差异。
- 完成后待办：用 metrics.json 全量数字替换 REPORT.md 中标注为「冒烟」的数字并定稿判定列。
