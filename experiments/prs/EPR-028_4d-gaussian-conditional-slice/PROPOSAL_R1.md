# EPR-028 R1 · 4D 高斯条件切片（稳定版规格）

状态：R1，2026-08-16 主 agent 依外部审查意见修订。**本文是唯一实施依据。**

与 `PROPOSAL.md`（R0，939 行）的关系：

| R0 的节 | R1 处置 |
|---|---|
| 跨臂冻结口径块、§1.2 数据计数、§4 外部事实核验（4DGS / GLUT / demo 行号与 sha256） | **保留，本文引用不重抄** |
| §1.1 单变量声明、§2.2 伪代码、§2.3 参数表、§3.1 公式、§3.2 损失、§3.3 优化器、§3.4 数值纪律、§3.5 五模式表 | **全部作废，由本文 §2–§8 替代** |
| headline 图像形成式（冻结块那一行） | **作废**，由本文 §5 替代 —— 见 §0.1 |

---

## 0. 修订依据（逐条对应审查意见）

### 0.1 P0 · double-α：R0 的 headline 在监督 α²

R0 同时规定内部目标 `f(x,s) ≈ (1−s)x + s·L(x)` 与外层混合
`Î = (1−α)I + α·f(I, s=α)`。代入理想预测：

```
f(I,α)     = I + α[L(I) − I]
Î          = (1−α)I + α{I + α[L(I) − I]} = I + α²[L(I) − I]
I*(GT)     = I + α[L(I) − I]
Î − I*     = −α(1−α)[L(I) − I]
```

`α=0.5` 时误差 = 完整 LUT residual 的 **25%**。函数值训练误差可以降到 0，
`s=0`/`s=1` 端点正常，而 local headline 与 `E_band` 永远不好，加步数无效。

修订：形成式按 arm 定义（§5）。公平比较由**同 GT、同样本、同指标、同步数**保证，
不由「所有臂共用同一个外层 compositor」保证。

### 0.2 P0 · R0 主臂一次改了七个变量

3D→4D 载体、Cholesky→scale+双四元数、1068→1404 生成维、目标换式、新增 marginal gate、
新增 `s` 采样、外层继续消费 α —— 七项同时变。训练好或坏都不可归因。
修订：四级 arm（§3），论文问题落在 **A3 − A2** 这一个差分上。

### 0.3 P0 · 双四元数 + Schur 换成条件参数化（§4.1）；bf16 inverse/det + identity fallback 禁用（§7）

### 0.4 P0 · R0 的「对数域」实际不在对数域

R0 伪代码 `p = exp(logp); w = p·o/(Σ p·o + ε)`：`exp` 之后仍会下溢，
`weight_underflow_frac` 只能计数不能解决。修订见 §4.3。

### 0.5 P1

| 项 | R0 | R1 |
|---|---|---|
| marginal 归一化 | 默认 `full`（含 `−log τ − ½log 2π`） | 默认 **peak-normalized**（§4.2）；full 降为附录消融 |
| τ | `exp` 无界 | **有界** `τ = 0.1 + 0.9·σ(r)`（§4.2） |
| carrier 与 VLM conditioner | 同时从零训练 | **分离**：Experiment C（oracle embedding）先过，才准进 Experiment Z（§6） |
| `s` 采样 | `s ~ U(0,1)` 独立 | **paired anchors** `{0, 1, u, 1−u}`（§8.1） |
| `L_m`（单调正则） | 主臂消融行 | **删除**（与生成律冲突，§8.2），换 `R_line` |
| `17⁴` 格 | 每训练步全量 | 训练用 256 随机 4D 点，`17⁴` 只在评测（§8.3） |
| opacity 初始化 | `logit(0.99)` | **0.5**（logit 0），`R_sparse` 前 10% 步关（§8.4） |
| 执行顺序 | 直接上全量 | **G0→G4 五道 gate**（§9） |

### 0.6 研究问题重述

4DGS 的前三维是物理空间，joint 4D rotation 表达物体运动；其 Table 3 的
No-4DRot 对照结论**不能直接搬**到本任务。本数据集的生成律是已知的精确可分解式

```
F*(x, p) = x + α(p)·[L(x) − x]
```

—— 一个只依赖 RGB 的 LUT residual × 一个只依赖位置的标量。所以本实验问的是：

> 在真值可精确分解为 `mask × LUT residual` 的数据上，联合 RGB–s 高斯耦合是否
> 仍带来可测收益？若不能，cross term 是否收缩到零，或只带来优化不稳定？

三种结果都可读：A3 > A2 / A3 ≈ A2 且 β→0 / A3 < A2。

`s` 全文称 **mask-strength scalar（条件标量）**，不称空间坐标：模型不消费像素 `(u,v)`，
空间结构由 where 分支的场提供。

---

## 1. 数据

沿用 EPR-030 口径层 `q3vl/whatb/caliber.py`，不新建第二套。

| 项 | 值 |
|---|---|
| 训练集 | `--data v2seg+l8`，`split=train` 且 `winner_confidence=normal`，**n = 119,828**（sft2seg 93,934 + L8 25,894） |
| 选型集 | V_what normal-only **n = 567** |
| 终测 | T_final n=533 / T_lut_unseen n=252（normal-only），**本 EPR 不碰** |
| LUT 池 | **本轮实测澄清口径**：train 全体（含 low）uniq lut_id = **3,149**；**normal-only = 3,081**；`v2seg+l8` = **3,166**。runner 按实测记录，不硬编码 |
| 切分 | sha1 规则族，不新建 |

V_where / V_what / T_final 永不进训练。

## 2. 模型（伪代码）

### 2.1 共用前向骨架（A1–A3）

```python
# theta: 每条 LUT 一组参数，由 generator(cond) 生成
# x: (B, P, 3) 颜色点   s: (B, P) 条件标量
def f(x, s, theta):
    mu_x, mu_s, C_chol, beta, tau, o_logit, M, b, G, g = theta   # §4.1
    if arm == "A2":  beta = zeros_like(beta)                     # 唯一差别
    ds     = s[..., None] - mu_s                                 # (B,P,N)
    mu_cs  = mu_x[:, None] + beta[:, None] * ds[..., None]       # (B,P,N,3) §4.1
    log_p  = log_gauss3(x, mu_cs, C_chol)                        # §4.3, FP32
    log_g  = -0.5 * (ds / tau) ** 2                              # §4.2 peak-norm
    log_a  = log_p + logsigmoid(o_logit) + log_g
    log_Z  = logaddexp(logsumexp(log_a, dim=-1), log(EPS))       # §4.3
    w      = exp(log_a - log_Z)                                  # (B,P,N)
    local  = (w[..., None] * (M @ x + b)).sum(-2)                # GLUT Eq.3
    glob   = clamp01(G @ x + g)                                  # GLUT Eq.4, 双裁
    return clamp01(glob + local)
```

A1 不走上式，走 §3 的显式门控式；A0 走标准 3D GLUT。

### 2.2 条件来源

| 阶段 | cond | 维 |
|---|---|---|
| Experiment C（G1–G3） | 可学习 `E[lut_id]`，3,149 × 64 | 64 |
| Experiment Z（G4） | `π(z_color)`，z 来自冻结基座 `<seg_color>` hidden | 64 |

生成器骨架不变（GLUT App A.2 的 MLP 形状，`q3vl/whatb/generator.py`）。

## 3. 四级 arm（+ 一个附录臂）

| Arm | 定义 | 隔离的问题 | per-prim | N | 生成维 |
|---|---|---|---|---|---|
| **A0** 3D-LUT | `T(x) ≈ L(x)`，图像端显式混合 | 原始强基线 | 22 | 48 | 1068 |
| **A1** 3D-ExplicitGate | `f(x,s) = x + s[T(x) − x]` | 只引入 `(x,s)` 目标与采样 | 22 | 48 | 1068 |
| **A2** 4D-BlockDiag | §2.1 且 **β ≡ 0**（生成但不接线） | Gaussian s-gating | 27 | 48 | 1308 |
| **A3** 4D-Joint | §2.1，β 参与前向 | joint RGB–s coupling | 27 | 48 | 1308 |
| **A4** 双四元数（附录） | R0 的 scale+双四元数+Schur | 保真度对照 | 29 | 48 | 1404 |

- **A2 与 A3 生成维相同、初始化相同、优化器相同**，唯一差别是 β 是否进前向。
  β 零初始化 ⇒ A3 的 step 0 与 A2 逐位相同（G0 单测钉死）。
- 论文问题 = **A3 − A2**。A0/A1 是量级参照，不是论文问题。
- 另报 **生成维匹配对照**：A0/A1 `N=48, dim=1068` 对 **A3 `N=39, dim=1065`**（39×27+12）。
- **A4 不作为主臂**，只在 A3 通过 G2 之后作为附录行跑一次。

## 4. 数学公式

### 4.1 条件参数化（替换双四元数 + Schur）

每个基元直接生成：条件协方差的 Cholesky 因子 `L_C`（3×3 下三角，diag 走 softplus）、
回归斜率 `β ∈ R³`、s 轴尺度 `τ > 0`、均值 `(μ_x ∈ R³, μ_s ∈ R)`。隐含的 4D 协方差是

```
Σ = [[ C + τ²ββᵀ ,  τ²β ],
     [   τ²βᵀ    ,   τ² ]]        C = L_C L_Cᵀ
```

条件分布**直接读出，不做任何 Schur 减法**：

```
μ_{x|s} = μ_x + β(s − μ_s)
Σ_{x|s} = C
```

**表达力无损**：任意 SPD 的 4D `Σ` 都可反解为
`τ² = Σ₄₄`，`β = Σ_{1:3,4}/Σ₄₄`，`C = Σ_{1:3,1:3} − Σ_{1:3,4}Σ₄₄⁻¹Σ_{4,1:3}`。
G0 用 float64 随机 SPD 矩阵逐点对拍（判据 §9-G0）。

代价被删掉的：运行时 Schur 减法、除以可能极小的 `Σ₄₄`、四元数单位化、
`q`/`−q` 双覆盖、左右四元数 gauge、paper/code 的 sign 与 axis-flip 之争。

### 4.2 s 门（marginal）

```
g_i(s) = exp[ −½ ((s − μ_{s,i}) / τ_i)² ]          # peak-normalized，无 −log τ − ½log2π
τ_i    = τ_min + (τ_max − τ_min)·σ(r_i)           # τ_min = 0.1, τ_max = 1.0
```

理由：完整一维高斯密度的常数项 `−log τ − ½log 2π` 逐基元不同，
在 mixture 归一化中**不抵消**；缩小 τ 会在 `s ≈ μ_s` 处抬高该基元权重，
路径是 τ collapse → 极窄 gate → dead primitives → seed 方差 / 梯度尖峰。
4DGS 官方实现 `gaussian_model.py:242` 的 `get_marginal_t` 也把 `/sqrt(2π σ)` 注释掉了。
**完整 normalized marginal 降为附录消融，只在主臂过 G2 之后跑。**

### 4.3 对数域前向

```
log p_i(x|s) = −½ [ ‖L_C⁻¹(x − μ_{x|s})‖² + 2Σ_k log L_C[k,k] + 3 log 2π ]
log a_i      = log p_i(x|s) + log σ(o_i) + log g_i(s)
log Z        = logaddexp( logsumexp_i log a_i ,  log ε )        ε = 1e-6
w_i          = exp( log a_i − log Z )
```

`log Z` 这一式**精确保留** GLUT Eq.2 的 `+ε`（`Σ_i a_i + ε`），同时全程不把小概率
先压成 0。`L_C⁻¹(·)` 走 triangular solve，不构造逆矩阵。

新增遥测（替代 `weight_underflow_frac`，后者保留但不再是主诊断）：

```
null_mass = ε / (Σ_i a_i + ε) = exp( log ε − log Z )
```

—— 直接量「所有高斯都失去覆盖、退化到 global 分支」的程度。

### 4.4 损失

主臂（沿用 EPR-030 已冻结的 pure L1 口径，`--loss-level 1`）：

```
L = L_rec + λ_line · R_line
L_rec   = ‖ f(x,s) − y* ‖₁ ,     y* = (1−s)x + s·L_ℓ(x)
R_line  = ‖ f(x,s) − [(1−s)·f(x,0) + s·f(x,1)] ‖₁
```

`λ_line` 预注册 **0.1**（消融行：0 / 0.1 / 1.0）。`R_line` 零额外前向成本 ——
§8.1 的 paired anchors 每个颜色本来就含 `s=0` 与 `s=1`。

**`L_m`（沿 s 轴单调正则）删除**：目标满足 `∂y*_c/∂s = L_c(x) − x_c`，
LUT 在某通道暗化时该导数为负是合法的，惩罚负差分等于惩罚正确目标。

`L_hc`（λ=10）、`R_sparse`（λ=0.001）、`L_s4d`、`L_img` 全部默认关，各自是消融行。
主 agent 裁定：本战役 EPR-030 已实证坏 loss 配方的机制链，oracle 阶段一律 pure L1 起步。

## 5. 图像形成式（按 arm，不共用外层 compositor）

```
A0        :  Î = I + S ⊙ [T(I) − I]
A1/A2/A3  :  Î = f(I, S)
```

A1 的 `f(I,S) = I + S⊙[T(I)−I]` 与 A0 在形式上相同，A2/A3 不是。
`S` 的来源、重采样算子（`q3vl/where/upsample.py:54-62` 的 `area_resize`）、
`E_in/E_band/E_out` 的三分层掩码（GT α，短边 512）**照 R0 冻结块不变**。

R0 那条外层 GT α 混合被删的第二个后果：它让 `S=0` 区域的输出被强制等于输入，
`E_out` 被系统性压低，**看不见 A2/A3 在 `s≈0` 处的颜色泄漏** —— 而那正是本实验要测的。

## 6. carrier 与 conditioner 分离

R0 同时从零训练 `π(z_color)`、生成器、高斯几何、opacity/gate、局部/全局仿射。
collapse 时无法区分四种成因。修订为两个实验：

**Experiment C（carrier oracle）** —— cond = 可学习 `E[lut_id] ∈ R⁶⁴`，跑 A0–A3。
回答：carrier 是否可优化、joint coupling 是否有效。

**Experiment Z（VLM conditioner）** —— 从 C 的稳定 checkpoint 出发，分级解冻：

| 行 | 内容 |
|---|---|
| Z0 | LUT-ID oracle（= C 的 A3 板，参照行） |
| Z1 | 只训 `π(z_color)`，generator 冻结 |
| Z2 | 解冻 color/global heads |
| Z3 | 0.1× lr 解冻 geometry |

必报：oracle gap、**同一 LUT 不同 record 的参数方差**（`z_color` 的 within-LUT 方差）。

**硬门：oracle carrier 未过 G2 时，禁止提交任何 Experiment Z 作业。**

## 7. 数值纪律（硬）

| 允许 bf16 | 强制 FP32 |
|---|---|
| VLM cache、投影层、generator MLP | covariance 激活、Cholesky、Mahalanobis、logdet、高斯权重、RGB→Lab |

- 用 `torch.linalg.cholesky_ex` + triangular solve；**禁止显式 inverse / det**。
- `cholesky_ex.info != 0`、NaN、Inf、非有限梯度 ⇒ **立即终止 run**（非零 rc）。
- **禁止 identity fallback**、禁止 determinant floor。错误必须暴露，不得吞掉后继续产出
  不可解释的 checkpoint。
- 因此 R0 的 `slice_det_fallback` / `sigma44_floor_hits` / `schur_pd_violations` /
  `logdet_floor_hits` 四个计数列**全部消失**（它们计的是被删掉的代码路径）；
  新列见 §10。

## 8. 采样与训练细节

### 8.1 paired anchors 采样器

保持 EPR-030 冻结的色批规模 **2,097,152 色/步**，把结构换成：

```
256 LUT × 2048 colors × 4 s-anchors = 2,097,152
每个颜色:  s ∈ {0, 1, u, 1−u},   u ~ U(0,1)
```

每步都保证：identity 端点、LUT 端点、两个互补中间点、同一颜色上的低方差路径监督。
**hard mining 以颜色组为单位**：选中一个颜色就保留它全部四个 s，不得单独丢端点。

G1/G2 的规模见 §9。

### 8.2 见 §4.4（`L_m` 删除、`R_line` 替代）

### 8.3 `17⁴` 只在评测

`17⁴ = 83,521`；`B=32, N=48` 下每步约 1.28 亿个 Gaussian-point 组合，
乘 117,440 步 ≈ 1.5×10¹³，尚未计入 Cholesky、仿射与反传。修订：

- 训练期的 4D 平滑正则（若开）每样本随机 **256** 个 4D base points 做有限差分；
- 完整 `17⁴` 只用于评测；
- `L_img`（若开）每样本采 **512–1024** 像素，或独立 `B_img ≤ 4`；
- **不在 B=256 上做完整短边 512 的图像 loss。**

### 8.4 初始化与调度

| 项 | 值 |
|---|---|
| opacity | `logit(0.5) = 0`（R0 的 `logit(0.99)` 已近饱和，与 entropy 正则同向推端点） |
| `L_C` | diag = 0.15（各向同性，与 3D GLUT 一致），off-diag = 0 |
| `β` | **0**（⇒ A3 step 0 逐位等于 A2） |
| `μ_s` | U(−0.1, 1.1)（照 4DGS） |
| `τ` | `r` 初始化使 `τ ≈ 0.447`（= √0.2，照 4DGS `dist_t`） |
| `μ_x` | 均匀格点（`uniform_grid_positions`） |
| 生成器各头末层 | 零初始化（step 0 恒等，proposition 2 保留） |
| `R_sparse`（仅 loss_level≥3） | 前 10% 步权重 0，随后线性 ramp 到 0.001 |

**优化器（§0.5 之外未改）**：Adam、`--base-lr 1e-3`、cosine、无 warmup、wd=0。
lr 依据：EPR-030 §3.1 在 2,097,152 色/步上实测 —— lr 1.6e-2 七行七死，
lr {1e-3, 4e-3, 3e-4} 零死。**待用户裁定**：是否加 StatLUT 原配方的
`AdamW wd=0.05 + 5-epoch 线性 warmup`（未自行改，NOTES 记录）。

## 9. 执行顺序（五道 gate）

| Gate | 内容 | 通过条件 |
|---|---|---|
| **G0** | float64 条件参数化对拍、log-domain、gradcheck、axis-order 单测（CPU） | 数学逐点对拍通过；step 0 严格 identity；A3(β=0) 与 A2 逐位相同 |
| **G1** | 单 LUT overfit，无 VLM、无 mining。1 LUT × 2048 colors × 4 anchors = 8,192 色/步，2,000 步 | 全部 s 上误差下降；零 NaN / 零 cholesky failure |
| **G2** | 32 LUT oracle，A1/A2/A3 × 3 seed。32 × 512 × 4 = 65,536 色/步，4,000 步 | 三臂稳定；seed 方差可接受 |
| **G3** | 全 3,149 LUT oracle，A0–A3，全口径 2,097,152 色/步 × 18,760 步 | carrier 主板完成 |
| **G4** | Z1→Z3 VLM conditioner | oracle gap、三负控制、within-LUT 方差可解释 |

故障定位表：

| 现象 | 指向 |
|---|---|
| A1 就不稳 | 公共 GLUT forward 或生成器 |
| A1 稳、A2 不稳 | s-gate / τ / coverage |
| A2 稳、A3 不稳 | β / cross coupling |
| oracle 稳、VLM 不稳 | condition mapping |
| 函数值好、图像 local 差 | field 或图像形成 |

**G1 与 G2 的板作为 G3 提交的 gate 文件串起来**（用户裁定：G0+G1+G2 一次全挂，
中间不等人；失败按 gate 自动不往下走）。

### 9.1 oracle 阶段该读哪个数字

G1–G3 的条件是 `E[lut_id]`（见 GT lut_id），所以它们的 `headline_normal_only`
**本质是 `B4_oracle` 的一个可学习近似，不可与任何 arm 的 headline 并排读**，
在板上标 `oracle_reference=true`。G1–G3 的判据是：

| 层 | 读什么 |
|---|---|
| G1 / G2 | LUT 格上的函数值误差（`grid_de00_mean`，分 s ∈ {0, 0.25, 0.5, 0.75, 1} 五列）+ §10 的稳定性遥测 |
| G3 | 同上，加 `E_in / E_band / E_out` 三分层（`E_out` 是 A2/A3 在 `s≈0` 处颜色泄漏的直接读数），加 `beta_absmean` |
| G4 | 才是 headline + 五基线 + 三负控制的完整板 |

G1/G2 **不需要** z 缓存、图像、`pred_field`，只需要预设库 `luts.npz`；
它们不出 published 板（`published=false`），只出 `metrics.json` 供下一道 gate 判读。

## 10. 遥测与判据

判据键不动：`headline_normal_only`（取 `.contexts.all.headline_normal_only`，
**禁用顶层 pooled `.baselines`**）、五条平凡基线、三负控制、12 个预注册键
+ **每键运行时断言**。AUC 禁用；checkpoint 选择禁 val loss；禁逐图 min-max；
跨行比较必须步数匹配。

`steps.jsonl` 首行必须出现的新列（替代 §7 删掉的四个）：

| 列 | 含义 |
|---|---|
| `null_mass_mean` / `null_mass_p99` | §4.3，退化到 global 分支的程度 |
| `cholesky_info_nonzero` | `cholesky_ex.info != 0` 的个数（>0 即终止，所以正常恒为 0） |
| `tau_p05` / `tau_p50` / `tau_p95` | τ collapse 监测 |
| `beta_absmean`（仅 A3） | cross term 是否收缩到零 —— 论文问题的直接读数 |
| `opacity_p05` / `opacity_p50` / `opacity_p95` | dead primitives |
| `n_pairs_s` | 每步 `(x,s)` 对数，钉死 §8.1 的采样结构 |
| `gnorm` | 尖峰即 NaN 前兆（本战役 10+ 条 NaN 全部此模式） |

采信纪律（EPR-030 §3.3 的洞，本 EPR 必须修）：**每次 quick eval 都复查
`L_rec` 是否为 NaN**，不只首次。NaN 预测的 ΔE00 算出来是 0.0 而非 NaN，
守卫不复查就会发布 `headline = 0.0` 的假板。

结果表规则：测什么指标只展示该指标数字；消融只用叠加式写法；**禁下任何结论、
禁一切揣测性表述**。

## 11. 待用户裁定（未静默拍板）

1. §8.4 的 warmup / wd（StatLUT 原配方 vs 本战役冻结的 Adam/无 warmup/wd=0）。
2. A4（双四元数保真度附录）是否要跑 —— 目前排在 A3 过 G2 之后，未提交。
3. `λ_line` 预注册值 0.1 是拍的（R0 无此项），是否要先用 G1 扫 {0, 0.1, 1.0}。

---

## 12. 实测结果（2026-08-16）

### 12.1 G0（CPU 单测）

`test_g4d_r1.py` **43 passed**（float64 SPD 等价性对拍、log-domain 对拍、axis-order、
gradcheck、step0 恒等、A2≡A3@step0 逐位、参数量 1068/1308/1065、禁 fallback、
A1 端点逐位、`R_line`≈0）。
`test_g4d.py`（runner 层，重写）**45 passed**。`q3vl/whatb/tests` 全量 **751 passed / 0 failed**。

### 12.2 G1（单 LUT overfit，A1，oracle，2,000 步，8,192 色/步）

LUT `rcp_0001f94f36ca4af3`。17³ 网格 ΔE00：

| s | 0 | 0.25 | 0.50 | 0.75 | 1.00 | mean |
|---|---|---|---|---|---|---|
| G1 | 0 | 0.12694013118743 | 0.25580412149429 | 0.38837644457817 | 0.52611875534057 | **0.25944789052009** |

`L_rec` 0.0393543615937233 → 0.0017211547819897532；`R_line` 恒 0；
`cholesky_info_nonzero` 0；`opacity_p50` 0.46781861782073975；`void_reason` null；
`published` false / `oracle_reference` true。

### 12.3 G2（32 LUT oracle，4,000 步，65,536 色/步，seed 20260810）

17³ 网格 ΔE00，A2 与 A3 的唯一差别是 β 是否进前向（生成维、初始化、优化器全同）：

| 列 | A2 | A3 | A3 − A2 |
|---|---|---|---|
| `grid_s0` | 0.07854729978135 | 0.10217790596653 | +0.02363060618518 |
| `grid_s25` | 0.22328902687877 | 0.23578127915971 | +0.01249225228094 |
| `grid_s50` | 0.39327506464906 | 0.38912830338813 | −0.00414676126093 |
| `grid_s75` | 0.58722608443350 | 0.57930135540664 | −0.00792472902685 |
| `grid_s100` | 0.84688038658351 | 0.82318751141429 | −0.02369287516922 |
| `grid_de00_mean` | 0.42584357246524 | 0.42591527106706 | **+0.00007169860182** |

末步遥测：

| 列 | A2 | A3 |
|---|---|---|
| `L_rec` | 0.003578596515581 | 0.003614491317421 |
| `R_line` | 0.001542291836813 | 0.001708222320303 |
| `gnorm` | 0.022057628259062 | 0.018463995307683 |
| `null_mass_mean` | 5.271238592285954e-07 | 2.1429987384635751e-07 |
| `cholesky_info_nonzero` | 0 | 0 |
| `tau_p50` | 0.350431114435195 | 0.375874340534210 |
| `opacity_p50` | 0.471223264932632 | 0.474144160747528 |
| `beta_absmean` | null（未接线） | 0.093917034566402 |

**上表只有 1 个 seed。** §9 的 G2 预注册是 A1/A2/A3 × 3 seed；A1 三条与
A2/A3 各两个补充 seed（20260811 / 20260812）已于同日提交（pueue 259–265），未出数。

### 12.4 与 EPR-031 O0 的配对（同 LUT、同 carrier、同步数、同色批、同 seed）

`rcp_0001f94f36ca4af3`，A1，2,000 步，8,192 色/步；step 0 的 `L_rec` 两行逐位相同
（0.0393543615937233）。唯一差别：O0 是每 LUT 一组独立参数（无 generator），
G1 是 oracle embedding → generator → carrier 参数。

| s | O0（无 generator） | G1（有 generator） |
|---|---|---|
| 0 | 0 | 0 |
| 0.25 | 0.16326683759689 | 0.12694013118743 |
| 0.50 | 0.33029332756996 | 0.25580412149429 |
| 0.75 | 0.50457954406738 | 0.38837644457817 |
| 1.00 | 0.68922787904739 | 0.52611875534057 |
| mean | 0.33747351765632 | 0.25944789052009 |

### 12.5 显存实测

`nvidia-smi --query-compute-apps`：G2/A2 **1452 MiB**、G2/A3 **1452 MiB**（65,536 色/步）。
G3（2,097,152 色/步）的 `--mem-peak` **待冒烟实测，禁外推**。

### 12.6 G2 的 3-seed 复现（§9 预注册的 A1/A2/A3 × 3 seed）

同 32 LUT 池、4,000 步、65,536 色/步；`grid_de00_mean.mean`（n=160 = 32 LUT × 5 个 s 档）：

| 臂 | seed 20260810 | seed 20260811 | seed 20260812 |
|---|---|---|---|
| A1 | 0.29623845543247 | 0.36402021839749 | 0.28294252341148 |
| A2 | 0.42584357246524 | 0.42314014041331 | 0.41191064943559 |
| A3 | 0.42591527106706 | 0.41691124178469 | 0.39934478057548 |

| 臂 | n_seed | mean | std | 极差 |
|---|---|---|---|---|
| A1 | 3 | 0.31440039908048 | 0.04348321864658 | 0.08107769498602 |
| A2 | 3 | 0.42029812077138 | 0.00738846349191 | 0.01393292302964 |
| A3 | 3 | 0.41405709780908 | 0.01351322850454 | 0.02657049049158 |

逐 seed 配对的 `A3 − A2`：

| seed | A3 − A2 |
|---|---|
| 20260810 | +0.00007169860182 |
| 20260811 | −0.00622889862862 |
| 20260812 | −0.01256586886011 |

`A3 − A2` 三个配对值的 mean = **−0.00624102296230**、std = **0.00631879245492**
（|mean| 与 std 同量级）。A1 的 3-seed 极差 0.08107769498601、std 0.04348321864658；
A2 的 3-seed std 0.00738846349192；A3 的 3-seed std 0.01351322850454。
变异系数：A1 13.83%、A2 1.76%、A3 3.26%。

三臂全程 `cholesky_info_nonzero` = 0、`void_reason` = null。
