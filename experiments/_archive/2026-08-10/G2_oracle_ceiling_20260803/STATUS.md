# G2 oracle 天花板 — 运行状态

任务卡：G2 三档 Δ_ceil（解析法）+ 实验法互证 + D0-4 MLP 容量探针。
GPU 纪律：**只用卡 1**（CUDA_VISIBLE_DEVICES=1），启动前 nvidia-smi 实测 0 MiB / 0%。
marker：`job.marker`（PID / 完整启动命令 / 日志路径）。
驱动脚本：`run_all.sh`（六步顺序：construct → real → control → MLP 探针 → GLUT 互证 → viz → summarize）。

## 进度日志

- 2026-08-03 09:32:29  construct: start
- 2026-08-03 09:33:22  construct: rc=0
- 2026-08-03 09:33:22  real: start
- 2026-08-03 09:40:54  real: rc=0
- 2026-08-03 09:40:54  control: start
- 2026-08-03 09:48:20  control: rc=0
- 2026-08-03 09:48:20  mlp: start
- 2026-08-03 09:54:09  mlp: rc=0
- 2026-08-03 09:54:09  xcheck: start
- 2026-08-03 10:35:34  xcheck: rc=0 (输出完整 n=100)；**驱动 shell 在此后中断**，viz/summarize 改为直接补跑
- 2026-08-03 10:40  viz: rc=0（三档直方图 ×2 口径 + success/failure 各 3）
- 2026-08-03 10:41  summarize: rc=0 → metrics.json（8/8 判据全绿）
- **ALL DONE**（全部计算产物完整；xcheck 之后驱动 shell 被中断，viz/summarize 手工补跑，结果等价）
- 2026-08-03 11:10:39  [补批] strat_real: start
- 2026-08-03 11:20:42  [补批] strat_real: rc=0
- 2026-08-03 11:20:42  [补批] strat_control: start
- 2026-08-03 11:27:38  [补批] strat_control: rc=0
- 2026-08-03 11:27:38  [补批] viz_delta: start
- 2026-08-03 11:27:53  [补批] viz_delta: rc=0
- 2026-08-03 11:27:53  [补批] all: done
- **补批 ALL DONE**（D-34 覆盖面）：real 6×100=600（l1–l6 全覆盖）/ control 4×150=600（g1–g4 全覆盖）；
  O-3 缺的 3 张附录口径案例图已补；l5/l6 案例图 4 张新增。原结论未被推翻（7.969→8.037 dB）。
