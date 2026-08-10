# STATUS — RO-9b · RO-9 读出算子的四处修复

| 项 | 值 |
|---|---|
| 状态 | **已完成**（导出 ✅ / 分析 ✅ / viz+scache ✅ / REPORT ✅） |
| 卡 | **卡 0**（`CUDA_VISIBLE_DEVICES=0`），每进程 `torch.cuda.set_per_process_memory_fraction`；实测各进程峰值 4.4 GB，同时最多 4 个进程合计 ≤ **14.2 GB**（限额 15 GB） |
| 后台作业 | 见 `job.marker`（PID / 完整命令 / 日志路径）；全部 FINISHED，**0 error** |
| 前向次数 | LM 逐头栈 642（生成+teacher-forced，25 min）｜ 组 A 13 臂 × 214 = 2782（prefill，13 min）｜ 桥接臂 642（teacher-forced，0 error） |
| 分析 | 纯 CPU：主 26 min + 受限瀑布 17 min，`OMP_NUM_THREADS=4` |
| 未 kill 任何他人进程 | ✅ 全程只启停自己的 PID |
| 卡 0 竞争实测 | 单样本生成 78–93 s（3.4 tok/s），G1 空闲卡时 9.4 s（34 tok/s）⇒ 约 9× 竞争。生成改批量（gen-batch=8），等价性核验与影响量化见 `NOTES.md` §三 U1 与 `REPORT.md` §1.1 |
| scache arm | `/var/cache/veradata/scache/ro9b-fixed`（424 条，`meta.norm.domain=[0,1]` 显式写出，实测 >0 的格占 99.6%） |

最终结论以 `REPORT.md` 为准。
