# G3 — 塌陷通道验证（Gate D3）+ D0-5 朴素 4D Δ_shuffle

**判决对象**：朴素 4D 扩展在**纯重建损失**下，是否真的走 PLAN §1.2 那条零代价逃逸通道。
**判决影响**：R-2（μ_s 锚定 + σ_s 有界）与 R-3（多样性 hinge）是否第一天必上。

> 状态：长任务已提交（`job.marker`，PID 3825287，卡 1，10 run ≈ 2.5 h）。
> 本文件中标 `【待全量】` 的格子由 `analyze_g3.py` 跑完后回填；
> 已有数字（天花板预检、CI、200 步冒烟）均为**本回合实测**。

---

## 1. 设置

| 项 | 值 |
|---|---|
| 渲染器 | 单个**全局** 4D GLUT，N=32 高斯，`model/glut_repro/model4d_naive.py`（3D 版 `model.py` 未改动） |
| 寻址 | μ∈R⁴，Σ 为 **4×4 全 Cholesky**，坐标序 **(s,r,g,b)**（s 首位 ⇒ L[0,0] 就是边缘 σ_s，见 NOTES 决策 2） |
| 载荷 | 不变：M_i∈R^{3×3}、b_i∈R³、全局 G∈R^{3×3}、g∈R³（只作用于 RGB） |
| 权重 | 完整归一化加权和（Eq.2 含 ε 分母），CI 对拍原式 max err 6e-7 |
| 初始化 | μ_c 均匀格、σ=0.15 各向同性、o=1、M=I、b=0、**G=0、g=0** ⇒ f(x,s)=x（红线：G 初始化 0 不是 I） |
| 损失 | **纯 L_rec（L1）**，无任何 s 轴正则（红线：s 轴禁平滑正则） |
| 数据 | D-CONSTRUCT `sanity` **L1+L4**，S-train 400 对 / S-val 48 对（`tools/data_splits` T1 旁表纪律） |
| oracle s | GT 掩膜 →**PIL BOX 面积加权 32×32**（与 `tools/scache/oracle.py` 同配方）→ **双线性**回原分辨率（不用 guided，NOTES 决策 3） |
| 预算 | 25000 步 / bs 16384 / Adam lr 2e-3 cosine→1%；**两臂逐字节同预算** |
| checkpoint | 不做选择，一律取最后一步（红线：checkpoint 选择禁用 val loss） |

**两条臂**（除 s 轴参数化外完全相同）：

| 臂 | μ_s | σ_s | 参数量(N=32) |
|---|---|---|---|
| **naive**（被测） | **可学，全部初始化 0.5** —— 正好落在退化集 D 上的对称鞍点（PLAN §1.2 第 1 条），故意的 | `exp(raw)`，**自由无界**（故意违反「σ 禁裸 exp」红线，NOTES 决策 4） | 876 |
| **anchored**（R-2 最小版） | **不可学**，K=6 均匀网格（i mod 6） | `0.025 + 0.275·sigmoid(raw)`，有界 [0.025, 0.30] | **844** |

> anchored 臂的 844 与 `PLAN_v2` §2 表 R-2 行标注的 **844** 逐位吻合——
> 这是「参数化确实照 R-2 规格实现」的独立佐证。

## 2. 三档数据（**这是本次实验最重要的方法学修订，见 NOTES 决策 1**）

任务卡指定的 L1+L4 **原样目标混了 40 种变换身份**（class|tier|sign），而 oracle s
只编码 **where**、不编码 **which**。于是任何 f(x,s) 的最优解都是条件均值 E[y|x,s]，
而变换符号近似对称 ⇒ 该均值退回近恒等。**结论：原样档表达不了「Δ_shuffle≥3 dB」这个判据。**

**天花板预检**（16³ RGB × 8 s 分箱条件均值，S-train 拟合 / S-val 评测，harness 口径）：

| 数据档 | s 的信息量 | 恒等基线(全图) | 天花板 f(x,s) | **Δ_shuffle\*（该档可达上限）** |
|---|---|---|---|---|
| `mixed`（任务卡字面档） | s 几乎无用 | 40.21 dB | 36.54 dB | **+0.32 dB**（掩膜内 **−2.73 dB**） |
| `tiered`（class+sign 固定，幅度随对变） | s 部分够用 | — | — | **≈ +4 dB**（200 步冒烟实测 +4.0） |
| `fixed`（exposure +0.60 EV 单一变换） | s 完全够用 | 31.46 dB | 43.66 dB | **+14.90 dB** |

三档**同源图、同掩膜、同 `T.composite` 原语**，只改目标怎么渲染；`fixed`/`tiered`
是把目标按固定/分档变换重渲染，`mixed` 是原样出厂目标。
**主判读用 `fixed`，三档并排读**——因为「零代价」和「零收益」必须区分开：
- naive 在 **fixed/tiered** 上也塌陷 ⇒ 逃逸通道是真·零代价 ⇒ **R-2+R-3 必上**；
- 只有 **mixed** 上塌陷 ⇒ 那是任务无信号，**不能**据此判 R-2/R-3 必上，也不能判可降级。

## 3. 预注册判据 vs 实测

### 3.1 主判据（Δ_shuffle，S-val 48 对，harness `collapse_probes`）

判据（任务卡 / `EXPERIMENTS_v3` Gate D3）：**<0.3 dB = 塌陷复现 → R-2+R-3 必上**；
**≥3 dB = 防塌陷可降级为可选**。

| 数据档 | 臂 | 该档天花板 Δ_shuffle\* | **实测 Δ_shuffle** | 实测 Δ_const | 判定 |
|---|---|---|---|---|---|
| fixed | naive | +14.90 | 【待全量】 | 【待全量】 | 【待全量】 |
| fixed | anchored | +14.90 | 【待全量】 | 【待全量】 | 【待全量】 |
| tiered | naive | ≈+4 | 【待全量】 | 【待全量】 | 【待全量】 |
| tiered | anchored | ≈+4 | 【待全量】 | 【待全量】 | 【待全量】 |
| mixed | naive | +0.32 | 【待全量】 | 【待全量】 | 判据在本档**不可用**（见 §2） |
| mixed | anchored | +0.32 | 【待全量】 | 【待全量】 | 判据在本档**不可用** |

**200 步冒烟的方向性读数**（不是结论，只说明链路通且三档确实分层）：
fixed naive **+13.06** / anchored **+13.94**；tiered naive **+4.00** / anchored **+4.23**；
mixed naive **−0.21**。

### 3.2 副判据

| 指标 | 红线 | fixed naive | fixed anchored | mixed naive | 出处 |
|---|---|---|---|---|---|
| Δ_const | <0.05 dB 且 loss 仍降 = 已塌陷 | 【待全量】 | 【待全量】 | 【待全量】 | PLAN §3 M1 |
| σ_s 贴上界比例 | >80% = 红线 | 【待全量】 | 【待全量】 | 【待全量】 | PLAN §3 M3 |
| σ_s 是否单调发散 | 发散 + sensitivity→0 = 通道证实 | 【待全量】 | 结构上不可能（有界） | 【待全量】 | PLAN §5 Gate D3 |
| s-sensitivity | →0 = s 轴被关掉 | 【待全量】 | 【待全量】 | 【待全量】 | 任务卡 |
| std(μ_s) | <γ_μ=0.15 则 R-3 hinge 会常年激活 | 【待全量】 | 固定 0.34（K=6 网格） | 【待全量】 | PLAN §2 R-3 |

### 3.3 实现自检（本回合已全过）

`python -m model.glut_repro.ci_checks_4d` — 10 组全 PASS，关键几条：

| 检查 | 结果 |
|---|---|
| `sqrt((LLᵀ)[0,0]) == sigma_s()`（σ_s 确实是边缘标准差） | max err **0.0** |
| s–RGB 交叉协方差非零且不可因子分解（R-2 要求） | max \|Cov(s,rgb)\| 0.033 ✅ |
| `log_density` vs `MultivariateNormal.log_prob` | max err **1.4e-5** |
| `weights` vs Eq.2 原式 `p·o/(Σp·o+ε)` | max err **6.1e-7** |
| forward vs 显式逐高斯求和 Σᵢwᵢ(Mᵢx+bᵢ)+Gx+g | max err **2.1e-7** |
| σ_s=1e12（逃逸通道终点）不 NaN，w→0，输出有限 | ✅ max w 7e-6 |
| init 严格等于闭式 x·(1−ε/(Σp·o+ε))，与恒等差 ≤0.37/255 灰阶 | ✅ |
| anchored 臂 σ_s 在 raw∈[−50,50] 下恒在 [0.025,0.30] | ✅ |
| anchored 臂 μ_s 是 buffer 无梯度、K=6 锚点全用上 | ✅ |

## 4. 交付物

```
experiments/G3_collapse_20260803/
  REPORT.md      本文件
  NOTES.md       核实记录 + 6 条待决策（决策 1 需主 agent 裁决）
  STATUS.md      冒烟 / 长任务 / 排卡
  job.marker     PID 3825287 + 完整启动命令 + 日志路径（D-20 新规）
  metrics.json   【待全量】analyze_g3.py 产出
  analyze_g3.py  聚合 + 出图
  config/        run_all.sh（全网格）+ env.json（git commit / 库版本 / 全超参）
  logs/          g3_full.log
  runs/<ds>_<arm>_s<seed>/   trace.jsonl（每 100 步探针）+ metrics.json + ckpt.pt
  viz/
    sigma_s_trajectory.png   ← **双臂 σ_s 轨迹并排图，无论结果如何都进论文附录**
    s_sensitivity.png        s-sensitivity 曲线
    delta_shuffle.png        Δ_shuffle 轨迹（含 0.3 / 3 dB 判据线）
    delta_const.png          Δ_const 轨迹
    mu_s_spread.png          std(μ_s) 轨迹（R-3 统计量）
    success_*.png failure_*.png   渲染样例（输入/GT/预测/s 场/误差图 五联）
```

## 5. 结论

【待全量】按 §3.1 三档并排填写。回填时必须同时回答三问：

1. **naive 臂的 σ_s 在 fixed/tiered 上是否单调发散？** 发散且 sensitivity→0 ⇒
   零代价逃逸通道证实 ⇒ R-2+R-3 必上，朴素版全部作废（PLAN §6 失败分支树 Gate D3 行）。
2. **naive 与 anchored 的 Δ_shuffle / in-mask PSNR 差多少？** 若 anchored 明显更好，
   即使 naive 没塌陷，R-2 仍有净收益，不该降级。
3. **mixed 档的 Δ_shuffle<0.3 dB 能否当作塌陷证据？** 依 §2 的天花板测量：**不能**。

## 6. 建议下一步（指向 EXPERIMENTS_v3 的具体行）

【待全量】候选：

- 若通道证实 → `EXPERIMENTS_v3` **RD-STD 行**保持 R-1+R-2+R-3+R-7+R-10 组合不变；
  并把「朴素 4D」从所有对照臂里删掉（它会 NaN-free 地退化成全局仿射，不构成基线）。
- 若通道未证实（naive 在 fixed/tiered 上都不塌陷）→ **不要**直接降级 R-2/R-3。
  建议改为在 `EXPERIMENTS_v3` 加一行 **RD-NAIVE 同预算对照臂**，
  在真实 s（RO-1/RO-9 读出，而非 oracle 掩膜）上重判——因为本实验的 s∈[0,1]
  只有 where、没有 what，是真实 s 信息量的**下界**情形。
- 无论哪种 → 建议把 NOTES 决策 5（harness `_mean_s_null` 对 2-D s 场按最后一维当通道）
  记进 harness 已知口径坑，避免后续实验静默踩中。
