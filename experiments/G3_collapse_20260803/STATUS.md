# STATUS — G3 塌陷通道验证（Gate D3）

## 冒烟（回合内，已通过）

- **4D CI 自检** `python -m model.glut_repro.ci_checks_4d`：10 组 / 全 PASS。
  含 log_density 对 `torch.distributions.MultivariateNormal` 逐点对拍（max err 1.4e-5）、
  weights 对 Eq.2 原式对拍（6e-7）、payload 对显式逐高斯求和对拍（2e-7）、
  σ_s→1e12 / 1e-6 两个极限均有限不 NaN、双臂契约（μ_s 冻结/可学、σ_s 有界/无界）。
- **200 步双臂冒烟**（任务卡要求）：`fixed`/`tiered`/`mixed` 三档 × naive/anchored 全部跑通。
- **分析链路干跑**：4 run 迷你网格 → metrics.json + 5 张曲线图 + 16 张渲染样例，全部产出。

## 长任务（已提交）

- **nohup PID：3834264**（09:57:11 提交），见 `job.marker`（含完整启动命令与日志路径）。
- 10 个 run，统一预算 25000 步；**预计 ~2.5–3 h**（单跑实测 26.5 步/s ≈ 16 min/run，
  共卡后可能变慢）。
- 完成标志：日志出现 `G3_FULL_DONE`（analyze_g3.py 已自动串在末尾）。
- 断点续跑：已有 metrics.json 的 run 自动 SKIP。

### 排卡

- 本任务用 **卡 1**（`CUDA_VISIBLE_DEVICES=1`），峰值显存 ~2 GB / 97 GB。
- 09:42 首次提交时卡 1 空闲；09:56 起卡 1 上出现**其他 agent 的 G2 作业**
  （pid 3832339 `tools/ceiling/fit_crosscheck.py`，5.4 GB）——**未触碰、共卡运行**。
  卡 0 上另有其他 agent 的 17 GB 任务，全程未碰。

### 一次主动中止重提（诚实记录）

首次提交（PID 3825287，09:42）跑到 fixed/naive/s0 的 19000 步时，自查发现
`build_probe_samples` 取的是 uid 排序后的**前 n 个**——而 uid 排序是
`L1_val_*` 全部在 `L4_val_*` 之前，故 `probe-n=8` 的逐 100 步探针集
**100% 是 L1（语义二值掩膜），一张 L4（几何软掩膜）都没有**。
这只影响轨迹图（final-n=48 覆盖全部 48 对，终值不受影响），但轨迹图正是本实验
的主交付物，遂改为**沿 uid 排序等距取样**（实测 8/12/48 三档均为 L1:L4 = 1:1），
清空 runs/ 全量重跑，保证 10 个 run 口径一致。旧日志留档为
`logs/g3_full.aborted-prefix-probe-bug.log`。

## 提交前已定的一件大事（详见 NOTES 决策 1）

任务卡指定的 D-CONSTRUCT L1+L4 **原样目标混了 40 种变换身份**（class|tier|sign），
而 s 只编码 where 不编码 which。实测该数据上**任何** f(x,s) 的天花板：
Δ_shuffle\* = **+0.32 dB 全图 / −2.73 dB 掩膜内**——即数据本身表达不了
「Δ_shuffle ≥ 3 dB」这个判据，两臂都会「因为无信号而判塌陷」。
因此主判读改用**同源同掩膜、目标按单一变换重渲染**的 `fixed` 档
（天花板 Δ_shuffle\* = **+14.9 dB**），并保留 `tiered`（+4 dB 量级）与
`mixed`（任务卡字面档）两档做梯度对照。三档一起跑，结论按三档并排读。

## 待办（长任务完成后，下一棒）

1. `grep G3_FULL_DONE logs/g3_full.log` 确认，然后把 metrics.json 的数字填进
   REPORT.md 判定表中的 `【待全量】` 占位。
2. 复核 `viz/sigma_s_trajectory.png`（论文附录图，无论结果如何都要进附录）。
3. 若 naive 臂在 `fixed`/`tiered` 上 σ_s 未发散且 Δ_shuffle ≥ 3 dB ——
   **不要**直接判「R-2/R-3 可降级」：结论只对「s 收益足够大」的任务成立，
   必须连 `mixed` 档一起读（见 REPORT 结论节的三档并排表）。
