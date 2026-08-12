# E1 · 核实记录 / 假设 / 待决策

## 一、实测核实(全部打开代码/磁盘验证,零转述)

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| A1 | **F_pre 缓存不存在** | `ls /mnt/nfs-ro/.../where_a-20260805/` 只有 `basis/ maskviews/ oracle/`;本地 `find -iname '*fpre*'` 零命中;`q3vl/whereb/data.py:334` 明文「no persisted feature cache」 | **任务卡「E1 纯 CPU」与 REQUIREMENTS「F_pre 缓存现成」均不成立**;已补一次视觉塔前向(400 图 57 s)落本地缓存 |
| A2 | **Where-A oracle 不是 WLS,链路里没有 Λ** | 读 `oracle.py:311-351`:目标 `objective_value` = `1 − soft_iou_minmax`,**逐像素等权**;优化器多起点 L-BFGS(band 18 起点),float64;`_lsq_start` 里的 OLS 只是**起点生成器**且无权 | 处方文档「逐图加权最小二乘」是理想化。本卡据此把 Λ 重新定义为 **Gauss-Newton 权重**(见 A3),并在 REPORT §0 显式声明 |
| A3 | `AᵀΛA` 的正确形式 | `_forward`:`q=w0+α(Φw_dir)`, `s=3tanh(q/3)`, `m=readout(s,ρ)`。m 对 w_dir 只经 `Φw_dir` ⇒ GN Hessian = `Φᵀ diag[(dm/ds·sech²(q/3)·α)²] Φ` | 该式为本卡「有效条件数」的定义;`dm/ds` 用 autograd 取 |
| A4 | oracle 记录 schema | 实测 dump:`fits.band.latent.{w_raw[71], w_dir[71], w0, alpha, rho_raw, sign_index}`;`start_losses[18]`、`best_start` | `w_raw` 是字面 w\*,`w_dir` 是单位化方向,**两者本卡都报** |
| A5 | 命名空间陷阱 | `oracle/BA-3-Joint/` 下有 **两套** V_where:非 s5 档缺 `curve` 与 `cband_normalization`;`whereb/config.py:307` 明文警告静默读错 | 本卡用 **s5** 档(400 条),正确 |
| A6 | 基底 B | `basis/BA-3-Joint/B.npy` = float32 (64,1024);B(·) 的实现在 `projector.py::BasisProjector`(**不在 basis.py**) | 直接用已发布 B.npy,与 projector 等价 |
| A7 | 独立复核 | 另一路 CPU fp32 端到端重跑得 `phi_gram_cond=3.33e5`、oracle 解码 IoU 复现残差 ~1e-5 | 与本卡结构条件数中位 3.01e5 一致 |

## 二、决策(保守默认)

| # | 决策 | 取值 | 理由 |
|---|---|---|---|
| D1 | Λ 取 GN 权重而非「guided 权重」 | `(dm/ds·sech²(q/3)·α)²` @ oracle 解 | 链路里不存在 guided 权重(A2);guided upsample 是 fit **之后**才作用在 s 上的后处理,fit 阶段看不到 guide。GN 权重是「oracle 实际求解的那个问题」的唯一正确 Hessian |
| D2 | 同时报结构/有效两个条件数 | 分开列,禁混用 | 二者差 1 个数量级且含义不同(基底自身 vs 实际求解问题) |
| D3 | readout 取 `band` | BA-3-Joint 两个 readout 都有,本卡用 band | band 的 ρ 只有 4 维、GN 权重解析清楚;cband12 可复跑,预期同量级。**如需 cband12 复核请裁定** |
| D4 | 近零阈值 | 相对阈 `λ < 1e-8·λ_max` | 绝对阈无意义(λ_max 跨样本差多个数量级) |

## 三、待主 agent 决策

### A.(高优先,零成本)`w_dir` 符号规范化的训练侧一致性

`latent.sign_index` 表明 `w_dir` 是方向且符号被**事后**规范化 —— `(w_dir, α) → (−w_dir, −α)` 是解集的
**精确 2 重对称**。若 W01/W02 训练时读的目标**没有逐字施加同一规范化**,回归目标里就混着人为双峰,
这**本身足以解释塌缩**,且修复成本是一行。**建议在 E2 之前先核对**(读一次训练侧的目标构造代码即可)。

### B. E2 的 λ 扫描定标方式

建议按**有效条件数**定标(令 cond 降到 ~1e3 量级)而不是盲扫 λ。需主 agent 确认是否接受该定标口径。

### C. E1 是否补 cband12 档

本卡只跑 band。cband12 的 ρ 有 36 维、GN 权重同样可解析,成本 ~1 分钟。**默认不跑**,请裁定。

## 四、其它

- 缓存 `/home/bc/data/runs/where_b/amort_cache_20260810`(879 MB,400 样本):`semantic_low(P,64)`、
  `img_low(3,gh,gw)`、`merger_out(n_img,2560)`。**F_pre(P,1024)本体故意不存**(只有 `B@F_pre` 进 phi,
  存全量大 16 倍且无消费方)。E5 直接复用此缓存,**无需再碰 GPU**。
- 本卡纯 CPU 运行 **11 秒**(400 样本),不占队列。
