# STATUS — RO-3 · 全层全头 text→image attention 扫描

| 项 | 值 |
|---|---|
| 状态 | **完成**（主实验 + 两条主 agent 追加 + 两条补件全部落地） |
| 裁决 | **晋级**：融合 out-of-fold AUC_target **0.830** ≥ 0.75，最佳层 **L11（中层）**；末两层未触发淘汰 |
| 追加①（逐头差分场） | **(b) 读法错确证**：L11H5 AUC_diff = 0.930（SAM3）/ 0.840（`.cgt` 主列）/ 0.922（构造 GT），同区域对照 0.522，FWER p = 0.000 |
| 追加②（AUC 三列） | `.cgt` 主列 0.8405 / D-CONSTRUCT 0.9217 / SAM3 对照 0.9246；软边阈值 0.3–0.7 稳定 |
| 补件①（D-CONSTRUCT S-train） | **1600/1600**，两个 arm 已落盘，RO-W 侧独立复算键 **1600/1600 命中，0 miss** |
| 补件②（U3 prefill vs 生成位） | `instr` 池**等价**（Δ ≤ 0.0008，空间 ρ 0.9999）→ RO-W 用 prefill；`gl` 池生成位略强但**仍在噪声地板内**（0.5908 < 零分布 q95 0.596），与 RO-9b 的 0.5719 一致 → **三家口径已对齐** |
| 卡 | 主实验卡 1、补件① 卡 0、补件② 卡 1；显存峰值 **≤1.46 GB**（限额 20 GB） |
| 提交纪律 | 补件两个作业全程 `rm -f` 日志 → `ps -p $PID` 实证 → `tail` 实质输出 → 才写 marker；**全程未用 `pgrep` 判活/判忙**（本次收尾时 `cat > STATUS.md` 又撞了一次 noclobber `file exists`，已按 D-20 先 `rm -f` 重写，如实登记） |
| 未 kill 任何他人进程 | ✅ 只启停自己的 PID |

## 交付物

```
experiments/RO3_layerhead_scan_20260803/
  REPORT.md            # 强制三行 + 判据并排表 + 三列 GT + §十二 U3 + §十三 补件
  NOTES.md             # 实施前实测核实 + 决策 + 待决策 U1–U7（U3 已关闭）+ 返工登记 + 补件记录
  STATUS.md  metrics.json   # metrics 已并入 4 个附录 json
  metrics_diffield.json  metrics_gt3col.json  metrics_genpos.json
  analyze_ro3.py  analyze_diffield.py  analyze_gt3col.py
  probe_genpos.py        # U3 探针（prefill 位 vs 生成位）
  write_construct_arms.py  viz_diffield_cases.py  ro3_viz.py
  config/verify_env.py|.json          # **实测** eager vs sdpa / 24 层 14 头 / token id
  config/build_jobs.py                # G1 协议批（零重新采样）
  config/build_jobs_construct.py      # D-CONSTRUCT S-train（import RO-W 模板）
  config/ro3_jobs.json  ro3_jobs_construct.json  ro3_construct_{val,train}.json
  config/finalize_scache.py           # arm 级 domain 元数据
  config/scache_arms_construct.json  config/auc_equiv.json  config/diffield_raw.npz
  logs/  viz/  job.marker

tools/readout/ro3_layerhead_scan.py
/var/cache/veradata/ro3_stacks_20260803/          4.4 GB（G1 协议批 1708）
/home/bc/data/ro3_stacks_construct_20260803/      4.2 GB（D-CONSTRUCT S-train 1600）
/var/cache/veradata/ro3_analysis_cache.pkl        重载阶段缓存
/var/cache/veradata/scache/ro3-l11-h5/            772   + _ARM_INFO.json
/var/cache/veradata/scache/ro3-fused/             772   + _ARM_INFO.json
/var/cache/veradata/scache/ro3-l11-h5-construct/  1600  + _ARM_INFO.json
/var/cache/veradata/scache/ro3-fused-construct/   1600  + _ARM_INFO.json
```
