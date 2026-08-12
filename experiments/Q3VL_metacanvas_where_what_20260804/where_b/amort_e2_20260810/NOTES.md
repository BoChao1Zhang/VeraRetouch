# E2 · 核实记录 / 假设 / 待决策

## 一、实测核实

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| A1 | oracle 在**序列化前**规范化 | `q3vl/where/oracle.py:395` `latent = canonicalize(Latent(...))`,其后才 `to_dict()` | 落盘 `w_raw` 本身即规范 |
| A2 | 训练侧不做二次规范化,但也不需要 | `q3vl/whereb/stores.py:171` `Latent.from_dict` 读 `w_raw` → `q3vl/whereb/fields.py:268` `w_dir_of(latent.w_raw)`;`basis.py:31` 的 `w_dir_of` 只做**保号**归一化 | 符号**按构造继承**,不是巧合 |
| A3 | 全量实证 | `q3vl/whereb/scripts/check_wdir_canon.py`,train 75,544(band)/75,543(cband12)+ V_where 400,两 readout | **100% 一致**,max 偏差 5.55e-16 |
| A4 | `canonicalize` 翻的是 `(w0, w_raw, rho)` 三者并镜像 readout | `basis.py:140-160` + `readout.py:179` `mirror_params` | E1 NOTES 写的 `(w_dir, α)→(−w_dir, −α)` 表述略有出入(α 不变),但群阶为 2 的结论不变 |
| A5 | **Tikhonov 不能加在 `w_raw` 上** | `oracle.py:302` `_forward`: `w_dir = w_dir_of(raw["w_raw"])` —— `w_raw` 的**模长是纯规范自由度**,mask 完全看不见 | 罚 `w_raw` 等于罚一个不可观测量。正确对象是 `w_eff = α·w_dir`,而 `‖w_eff‖ = α`,故惩罚项 = `λα²` |
| A6 | `λα²` 确实是真 ridge | 把单位方向的一部分花在 Φ 的近零方向上会缩小 `‖Φ w_dir‖`,为维持 q 的尺度必须抬高 α,惩罚项随即计费 | 即 `σ²/(σ²+λ)` 收缩在本参数化下的写法。默认 `ridge_lambda=0.0`,已发布 oracle 口径不变 |
| A7 | 改动不破坏既有行为 | `pytest q3vl/where/tests/test_oracle.py test_fitpool.py test_basis.py` | **46 passed** |
| A8 | 有效条件数复现 | 本卡复算已发布 latent 得 8.414e5 | 与 E1 的 8.356e5 一致(E1 的 Λ 含 α²,本卡的 Λ 对应 `w_eff`,差一个 α² 常数) |

## 二、决策(保守默认)

| # | 决策 | 取值 | 理由 |
|---|---|---|---|
| D1 | 惩罚项加在 `w_eff` 而非 `w_raw` | `λ·α²` | 见 A5/A6。加在 `w_raw` 上会得到一个「什么都没发生」的假阴性,是本卡最容易踩的坑 |
| D2 | λ 按 E1 谱定标而非盲扫 | `λ₀ = median(λ_max)/(2·target_cond)` | 加 `λ‖w_eff‖²` 给 GN Hessian 贡献 `2λI`,故 `cond ≈ λ_max/(2λ)`。整臂常量,非逐图 |
| D3 | 记录的 `loss` 仍是**纯目标值**,不含惩罚项 | 只在 closure 里加罚 | 否则 `reject_loss` 阈值与 `start_losses` 的含义会漂移,已发布口径不可比 |
| D4 | 只跑 band | 与 E1/E5/W01 一致 | cband12 可复跑;**趋势不会反转**(惩罚项与 readout 无关),故未跑 |
| D5 | 验收门未过即**不进 GPU 阶段** | 停在 stage A | 任务卡明文「先验收…再往下」。硬扛下去等于用一个已知被毁的 oracle 去训练 |

## 三、待主 agent 决策

### A.(新发现,便宜且影响 P1)`L_dir` 的符号不连续

§0b 实测:**3.92%** 的 train 目标坐在「间距<1.05 且次大系数反号」的活跃翻转边界上
(放宽到间距<1.20 则为 **13.9%**)。这些样本上,输入的无穷小扰动会翻转全部 71 维目标符号,
`L_dir = 1 − cos` 从 0 跳到 2。

候选修法:① `L_dir` 改 `1 − |cos|`(符号不变);② 取消 `L_dir`,监督全部搬输出空间。
**P1/P3 本就走输出空间,②自动成立**,所以本条实际只影响「是否还要保留 W 系那种 w 空间辅助损失」。
**保守默认:P1/P3 不设 `L_dir`,本条隐患随之消失;不回头改 W01/W02。**

### B. 是否接受「M2 修复路线关闭」这一改判

E1 把 M2 抬为主嫌并建议 E2 为第一优先。本卡的结论是:**M2 的诊断成立,但 M2 的修复方案被证伪** ——
病态与 0.97 天花板在 Phi-71 上是结构性绑定的。需主 agent 确认是否据此关闭该路线
(并相应下调 E1 建议 4「基底正交化/白化」的优先级,理由见 REPORT §4-1)。

### C. E4 是否还跑

E1 已弱化 M1 前提(满秩、零近零特征值);本卡进一步显示「难回归」的成因是基底-天花板绑定
而非多模态。**保守默认:E4 保持排在最后、可延后**,不主动申请槽位。请裁定是否直接取消。

## 四、其它

- stage A 运行 **8.9 分钟**(400 样本 × 6 档 λ × 18 起点 / 40 workers),纯 CPU,**未占 GPU 队列**。
- 第 0 项普查运行 **约 2 分钟**(151,487 条 latent 读取 + 校验)。
- 产物:`metrics_stageA.json`、`config/item0_wdir_canon_audit.json`、`config/run_setup_stageA.json`、
  `logs/e2_stageA.log`。
- 代码改动:`q3vl/where/config.py`(新增 `FitConfig.ridge_lambda`,**默认 0.0**)、
  `q3vl/where/oracle.py`(closure 内条件性加罚)、
  新增 `q3vl/whereb/scripts/check_wdir_canon.py`、`q3vl/whereb/scripts/run_amort_e2.py`。
- 本卡无 viz:stage A 的产出是一条 λ-IoU-cond 扫描曲线,没有逐样本空间场可画;
  失败画像由 E3/E5 的 viz 承担。**若审阅要求 viz,请裁定**(可画 λ 扫描曲线图,但那是曲线不是案例)。
