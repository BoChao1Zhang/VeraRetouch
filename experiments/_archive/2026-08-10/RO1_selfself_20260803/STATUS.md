# STATUS · RO-1（self-self 零训练读出）

- **状态**：**DONE**（2026-08-03 15:44 完成，wall-clock 78 min）
- **实验编号**：EXPERIMENTS_v3 §2.1 RO-1
- **Gate 判定**：**PROMOTE** —— 最优 ClearCLIP+伪影三件套，median AUC **0.939**（L1, n=23）/
  **0.930**（SAM3 主体掩膜, n=214）；预注册晋级线 ≥0.80、淘汰线 <0.75
- **三算子**：ClearCLIP 0.930 > NACLIP 0.916 > SCLIP 0.902（**弱排序，CI 重叠**）；
  未改造 CLIP 对照 **0.214**（系统性反相关）
- **符号检查**：三算子全 PASS（逐样本 AUC>0.5 占比 0.925–1.000），**无任何事后翻转**
- **指令条件性**：Δ_shuffle 配对中位 +0.220（p=5.5e-17，胜率 0.73）—— **正面证据**，
  与 RO-9（配对 Δ=+0.0005, p=0.121）形成直接对比
- **卡**：仅卡 1，显存峰值 **1.72 GB**（限额 20 GB）；`job.marker` 有 PID/命令/日志
- **scache arm**：`/var/cache/veradata/scache/ro1-clearclip-l11`（237 条，32×32 fp16，全局固定仿射）

## 给主 agent 的三个必读

1. `REPORT.md` §二（判据并排表）+ §十（结论）+ §十一（8 条建议，含 **C2：建议改 RO-4 判据**）
2. `NOTES.md` §三 —— **与 IMPL_DOSSIER 不符的 5 条**（gem_torch API bug / gem_depth 语义 /
   min-max 可关 / open_clip pin 需收紧到 ==2.24.0 / pkg_resources）
3. `NOTES.md` §五 —— **6 条待决策**，其中 U-RO1-2（短名词 vs 长描述，差 +0.02~+0.05）
   与 U-RO1-6（GEM 是否入榜，需独立 venv）需要拍板
