# RO-W 状态

实验：**RO-W**（新增臂，VLM → 14 维基系数读出）｜ 卡 1 ｜ 2026-08-03

## 管线与状态

| 阶段 | 脚本 | 产物 | 状态 |
|---|---|---|---|
| 0a 基底通道 | `run_prep.sh` → `prep_feats.py` | `/home/bc/data/row_basis_20260803/feats_{train,val}/<uid>.npz`（L,S,e1..e6 + 掩膜 + 原图）、`meta_*.jsonl`（含 CLIP 锚点 class5） | ✅ 完成（train 1600 / val 192，标志 `PREP_FEATS_DONE`） |
| 0a′ 读出斜率校准 | `calib_readout.py` | `config/calib_readout.json` | ✅ 完成，**g₀ = 6.0**（预注册默认被数据支持，见 NOTES D1） |
| 0b 离线 w* 拟合 | `run_fit_labels.sh` → `fit_labels_gpu.py` + `fit_labels.py`（CPU L-BFGS，逐掩膜取优，见 NOTES D12） | `labels_{train,val}.jsonl`、`labels_val_cpu.jsonl`、`phi64_{train,val}.npz` | ✅ 完成（2,000 唯一键；**S-train 离线上界 soft-IoU 0.940**；L0 上 α 精确 = 0，24/24 满足 α<1e−2） |
| 0b′ 拟合器对拍 | `fit_labels_gpu.py --validate` | `config/fitter_validation_val*.json` | ✅ v1 已出（中位 −0.021 vs CPU L-BFGS）；v2（轴对齐重启版）待重跑 |
| 1 VLM 隐状态缓存（retouch tap，负对照） | `run_cache_vlm.sh` → `cache_vlm.py` | `vlm_{train,val}/` | ✅ 完成（train 5,200 + val 624） |
| 1b **instruction token 缓存（主档）** | `run_cache_vlm2.sh` → `cache_vlm2.py` | `vlm2_{train,val}/`（含 `instr_pool` / `instr_tok`；val 另含 `nonoun` 档） | ✅ 完成（train 1,600 + val 816，标志 `CACHE_VLM2_DONE`） |
| 2 读出头训练（**负对照 tap = retouch token**） | `run_train_all.sh` → `train_head.py --feat retouch` | `sweep/*`、`runs/<sup>-<head>-L<层>/` | 🔄 运行中 |
| 2b 读出头训练（**主档 tap = instruction token**） | `run_train_instr.sh` → `train_head.py --feat instr` | `sweep_instr/*`、`runs/<sup>-<head>-instr-L<层>/` | ⏸ 等 1b + 2 |
| 3 评测 | `evaluate.py` | `metrics.json` | ⏸ |
| 4 图 | `viz.py` | `viz/success_* / failure_* / w_components_boxplot.png / shuffle_pair_* / alpha_by_level.png` | ⏸ |
| 5 scache 导出 | `export_scache.py` | `/var/cache/veradata/scache/row-<sup>-<head>/` | ⏸ |
| 6 报告 | `make_report.py` | `REPORT.md`（全部数字由 `metrics.json` 生成，无手工录入） | ⏸ |

## 中途设计变更（2026-08-03 傍晚，主 agent 转达 RO-3 判决）

任务卡原设计「取 `<retouch_light>` 隐状态 → 14 维」被 RO-3 证伪（该 token 池在全部
336 个 (层,头) 上 AUC_target ≤ 0.5542，低于置换零分布 q95 = 0.596）。**主档改为
instruction 文本 token 的表示**；原设计全部保留为**负对照行**。另新增 `nonoun`
（无区域名词）指令档单列。细节与限定见 `NOTES.md` D13/D14。

## D4 主档切换（2026-08-03 晚，主 agent 拍板）

| 档 | 数据 | 状态 |
|---|---|---|
| **主档** | **D-SFT-L**：真实用户指令 + `.cgt.png` 逐候选区域掩膜 | 📋 样本清单已建并核实：`sftl_samples.jsonl`，**S-train 3,302 / S-val 219**（源 2,724 / 178），join 5/5 seek-read 通过，`.in.jpg`↔`.cgt.png` 长宽比一致 40/40。管线脚本可整套复用，只换样本源 |
| 诊断档 | D-CONSTRUCT L0–L7 | ✅ 保留：能力上界 0.940 + **α=0 构造性退化**（这份数据独有） |

⚠ journal 里的 `I_in` 路径 **5/5 已失效**（指向已清理的 ramstage），取图必须走 shard 的
`.in.jpg` 成员——D-25 的一个具体实例，已写进 NOTES D18。

## 复现

```bash
cd experiments/ROW_basis_coeff_20260803
./run_prep.sh                                   # 0a  (GPU1, 5 shard, ~50 min)
python calib_readout.py --gs 2,4,6,8,12         # 0a′ (CPU, TRAIN only)
./run_fit_labels.sh                             # 0b  (GPU1, ~50 min)
./run_cache_vlm.sh                              # 1   (GPU1, 3 shard, ~90 min)
./run_train_all.sh                              # 2+3+4
python export_scache.py                         # 5
python report_tables.py > /tmp/tables.md         # 6
```

## 交付物检查命令

```bash
jq '.arms | to_entries[] | {arm:.key, s:.value.summary}' metrics.json
jq '.arms[].shuffle_paired.shufin | {d_iou_med, d_iou_ci95}' metrics.json
jq '.offline_upper_bound.all_iou_med, .auc_luma_med' metrics.json
ls viz/success_* viz/failure_* viz/w_components_boxplot.png
cat config/scache_arm.json
```

## 纪律

- 卡 1，每进程 `set_per_process_memory_fraction` 0.05–0.12（≤ 11.7 GB），实测占用远低于
  15 GB 上限；**未 kill 任何非本任务进程**。
- 长任务全部 nohup + `job.marker`（PID / 完整命令 / 日志 / 完成标志 / 断点续跑规则）。
- split：只读 T4_construct manifest 里 T1 冻结旁表给出的 S-train / S-val，**无 ad-hoc 切分**；
  模型选择用的是 S-train **内部按 source 切**的 inner 折，**S-val 在最终评测前从未被看过**。
