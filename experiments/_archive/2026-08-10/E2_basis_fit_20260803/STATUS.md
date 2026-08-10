# E2 状态

## 第一回合（全量拟合）：**已完成**（2026-08-03 03:24）

- `run_full.sh`（nohup，PID 3248254）→ 完成标志 `FULL_RUN_DONE`，日志 `full_run.log`。
- 链路：prep_data.py（GPU0，CLIP 特征 300 图 + 掩膜 1020 张）→ run_fit.py（36 workers，4,344 拟合，3,938 s）
  → analyze.py（metrics.json + viz）。`n_fit_errors=0`。
- 冒烟产物保留为 `*_smoke.*`。

## 第二回合（补件轮：缺口 A + 缺口 B）：**已完成**

依据 `REVIEW-result.md` §2.3 末（缺口 B：`gauss` 路径 `KeyError('mu')`）与 §2.4（缺口 A：
受约束 s 轴能否合成平顶）。预注册判据写在 `NOTES.md` 的「补件轮」节（**先写判据后跑**）。

| 任务 | 脚本 | 产物 | 状态 |
|---|---|---|---|
| 缺口 B：`gauss` 全量补数 | `run_supplement.sh` shard 1 | `results_gauss.jsonl`（1,196 fits） | 完成 |
| 缺口 A（mask 级）：约束 s 轴 | `run_supplement.sh` shard 2 | `results_constrained.jsonl`（2,400 fits） | 完成 |
| A-4 预算对照 + A-5 σ_max/M 扫描 | `run_supplement.sh` shard 3 | `results_sweep.jsonl` | 完成 |
| A-2（响应级）+ A-5 响应级扫描 | `axis_response.py`（8 workers） | `metrics_axis_response.json`、`viz/constrained_axis_response.png` | 完成 |
| 汇总 | `analyze.py` | `metrics.json`（6,217 结果 / 0 报错）、`viz/*.png` | 完成 |

**结果一句话**：缺口 A **闭合且为正面结论**（约束 s 轴 mask 级 0.9754 vs 无约束 0.9753，几何-only 0.9901 反超 0.9766；
响应级平顶偏差 1.4e-05）；缺口 B **修复**（`gauss` 1,196 拟合 0 报错，环形 0.7781 复现「0.78」）。
预注册判据 A-1/A-2/A-3/B 全过，**A-4（预算反混淆）未过门**（+0.0105 vs 门 0.01，n=8，方向有利，已如实报出）。

- 长任务纪律（D-20）：`job.marker` 记录 PID / 完整启动命令 / 日志路径 / 完成标志 / **两次重排的理由**；
  完成标志 = `SUPPLEMENT_FITS_DONE`；日志 `supplement_run.log`、`axis_response.log`、
  `cheap_block.log`、`unnorm_block.log`。
- 资源：**CPU only**（≤40/48 核），**全程未使用 GPU**；只 kill 过本任务自己启动的
  `python3 run_fit.py`（anchored pattern），已核实其它 agent 的 RO1/RO2/RO3/RDG 作业未受影响。
- **机器争用**：其它 agent 的作业把 load 顶到 90–140，同一个约束档拟合空载 97 s / 本轮 330–2000 s。
  因此对**非主判据**的格降 n（清单见 REPORT §8 与 NOTES）；环形主判据格保持最大 n。
- 未新增任何数据：复用第一回合的 `/var/cache/veradata/e2_basis_20260803`（S-val 派生，split_rule =
  T1 旁表 `splits.sqlite3`），因此 split 纪律与第一回合一致，无新增暴露面。


## 第三回合（wave 3 收尾轮）：**已完成**（2026-08-03 19:33）

任务卡三件事。预注册 W-1..W-4 写在 `NOTES.md`「收尾轮 wave 3」节（**先写后跑**）。

| 任务 | 脚本 | 产物 | 状态 |
|---|---|---|---|
| ① β 可达区间随 M（解析 + 1,500 组数值核验 + 响应级 4M×2cfg×n=200） | `beta_reach.py` + `beta_domain_addendum.py` | `metrics_beta_M.json` | 完成 |
| ① 掩膜级 M 扫描（M=6/8 main14、M=6/8/10 geo6，各 n=60） | `run_wave3.sh` block 1/2/4/5/6 | `results_m{6,8}main.jsonl`、`results_m{6,8,10}geo.jsonl` | 完成 |
| ② A-4 正式复跑（n=40 / 300 步） | `run_wave3.sh` block 3 | `results_a4rerun.jsonl` | 完成 |
| ② A-4 预算匹配对照（无约束档 300 步，**本轮新增**） | `run_wave3b.sh` | `results_a4free.jsonl` | 完成 |
| ③ 两处表述订正 | — | REPORT §14 订正声明 R-1/R-2（+R-3） | 完成 |
| 汇总 | `wave3_analyze.py` | `metrics_wave3.json`、`viz/beta_reach_vs_M.png`、`metrics.json` 的 `wave3` 键 | 完成 |

**结果一句话**：
① **M=6（域 [−3,3]）在等不透明度读法下画不出 44.5% 的环**（β_min 8.00 > 实需下限 6.16），
但掩膜级仍过 A-1 两条门（0.9716 / 0.9777，掉幅 −0.0022 / −0.0002，0% 掉幅 >0.03）——
**「行，但余量为零」**（35% 的环 σ 顶死上界）。M=8/10/12 解析与实测同时给正面证据。
**真正的口径冲突是「域」不是 M**：RD-G Stage-2 实装是 **K=6 @ [0,1]**（Δμ/σ_max=0.67，比 M=12@[−3,3] 还宽松），
**代码一行都不用改**；危险的只有「把 K=6 搬到 [−3,3]」。
② **A-4 仍未过门**（n=40：+0.01269，CI [+0.00995,+0.01642]，门 <0.01），但新增的预算匹配对照
把结论完全倒向 A-1：无约束档 300 步只涨 +0.00008（120 步已收敛），
**同预算 300 步下约束档反超无约束档 +0.01085**（CI [+0.00968,+0.01479]）。
③ 订正 R-1（语义族缺格 → 按需求判定不适用）、R-2（C-4 改写为条件形式，
RD_std_e 数字逐条回读核实**完全一致**）。

- 长任务纪律（D-20）：`job.marker` 两条新记录（shell PID **655043** / **763580**、完整启动命令、
  日志 `wave3_run.log` / `wave3b_run.log`、提交前 `rm -f` 日志、`ps -p` 实证存活、
  完成标志 `WAVE3_FITS_DONE` / `WAVE3B_FITS_DONE`）。
- 资源：**CPU only（28/24 workers ≤ 48 核），全程未使用 GPU**；**未 kill 任何进程**（本轮无重启）。
- **只增不改**：第二回合的拟合结果文件一字未动；两处表述订正写成 REPORT §14 的显式「订正声明」，
  正文改动处留 `【订正 R-n】` 标记。`run_fit.py` 只新增两个 CONFIGS 条目。
- 未新增任何数据：仍复用 `/var/cache/veradata/e2_basis_20260803`（S-val 派生，T1 旁表 split）。

## 复现

```
python3 prep_data.py               # 第一回合，GPU（CLIP 特征 + 掩膜）
python3 run_fit.py --workers 36    # 第一回合，results.jsonl
./run_supplement.sh                # 第二回合三个 shard（CPU，约 2 h）
python3 axis_response.py --workers 8
python3 analyze.py                 # metrics.json + 全部 viz

# 第三回合（wave 3）
./run_wave3.sh                     # 掩膜级 M 扫描 + A-4 复跑（CPU，约 80 min）
./run_wave3b.sh                    # A-4 预算匹配对照（CPU，约 2 min）
python3 beta_reach.py --n 200 --workers 8
python3 beta_domain_addendum.py
python3 wave3_analyze.py           # metrics_wave3.json + viz/beta_reach_vs_M.png
```

## 交付物检查命令

```
jq '.criteria, .criteria_supplement_gapAB' metrics.json
jq '.criteria_matrix_main14.ring' metrics.json
ls viz/success_* viz/failure_*
jq '.wave3.beta_reach_per_M_domain_pm3_sigma_0.025_0.30' metrics.json
jq '.wave3.A4_rerun_n40, .wave3.A4_budget_matched_control' metrics.json
```
