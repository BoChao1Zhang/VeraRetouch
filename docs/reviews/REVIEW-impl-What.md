# REVIEW-impl-What：Stage-What 实现审阅

- 审阅人：独立实现审阅 subagent（与实现者无共享上下文）
- 日期：2026-08-05
- 规格权威：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`
  §2.2 / §2.3 / §6 / §7 / §8 / §9 / §10.4 / §12 / §14
- 审阅对象：`q3vl/what/`（19 模块 + 11 测试文件 + 3 作业脚本），
  `experiments/Q3VL_metacanvas_where_what_20260804/what/`（NOTES.md、
  PREFLIGHT_WHAT_PENDING.md、config/arm_matrix.json、config/mock_e2e.json、
  preflight/preflight_what.json）
- git commit：`fef9f93`
- 审阅方式：全量读码 + 独立复跑。未修改任何代码/数据，未使用 GPU
  （Base SFT rank PID 3395226/3395227 全程未受干扰）。

## 复跑记录（审阅人自己执行的，不采信实现者报数）

| 动作 | 环境 | 结果 |
|---|---|---|
| `pytest q3vl/what/tests -q` | `/home/bc/envs/q3vl_sft/bin/python`（3.12.12 / torch 2.10.0+cu128） | **155 passed, 3 skipped**（skip = 缺 `skimage` / `colour` / `dataset_build.src.construct.rendering` 的 libstdc++ 问题） |
| 同上 3 个 skip 的对拍 | `/home/bc/miniconda3/bin/python` | **全部 pass**（20 passed，含 colour-science 1e-9、dataset_build CPU oracle 1e-5、skimage Lab） |
| `python -m q3vl.what.preflight --limit 400` | 战役环境 | **11 pass / 0 fail / 0 skip**，`ok=true, complete=true`，与 `PREFLIGHT_WHAT_PENDING.md` 声称的 11/11 逐项一致 |
| `check_lut_unseen_disjoint()` 全量 index 扫描 | 战役环境 | **pass**：`T_lut_unseen` 259 LUT，与 train(3149)/V_where(530)/V_what(531)/T_final(577) 交集全 0 |
| `check_gt_lut_resolves(limit=400)` | 战役环境 | **pass**：5 个 split 各 400 条，`n_missing = 0`（LUT 数 310/288/304/251/298） |
| D-W1 独立重推（自写公式，不调用 `q3vl.what.gaussians`） | 战役环境 | `Σ_i q_i ∈ [0.9999975, 1.0]`；字面公式 `mean T(x)/x = 1.99999988`、`max|T−x| = 0.99984`；实现的 `residual_zero` `max|T−x| = 2.4e-6` |
| `main_board` gate 行为构造性验证 | 战役环境 | 见 B-1，gate 未生效 |

---

## 一、D-W1 amendment 审定（最高优先，单列）

### (a) 数学上确认字面公式确实产生 2x —— **确认**

协议 §7.6 同时给出两句话：

1. 「全局与局部 affine 都采用 identity-centered residual 参数化」；
2. `T_pred(x) = clamp((I + ΔG) x + b_g + Σ_i q_i(x)·(M_i x + b_i), 0, 1)`。

关键在 `q_i` 是**归一化**权重。实现的归一化式（与 GLUT 一致）为
`q_i = o_i g_i N_i / (Σ_j o_j g_j N_j + ε)`，因此 `Σ_i q_i ≤ 1`，且在锚点覆盖良好时
`Σ_i q_i → 1`（我实测 min 0.9999975 / max 1.0，ε=1e-6 只会让它**小于**1，永远不会大于 1）。

于是当 `M_i = I + ΔM_i` 且 `ΔG = ΔM_i = b_g = b_i = 0` 时：

```
T(x) = (I)·x + Σ_i q_i(x)·(I·x) = x + (Σ_i q_i)·x ≈ 2x
```

我用**自己写的公式**（不经过 `q3vl.what.gaussians.render`）复算：`mean T(x)/x = 1.99999988`，
不 clamp 时 `max|T−x| = 0.99984`（在 x→1 处）。这正是战役红线点名的
「全局仿射 G 初始化 = 0 不是 I（否则 f(x)=2x）」。**实现者的发现成立，不是误读。**

补充一个实现者没写出来的必要条件：这个失败是**零初始化**特有的。协议之所以能写出这个公式
而不自觉，是因为它没有同时固定「输出头零初始化」。而输出头必须零初始化——否则第一步的
`T_pred` 就是一个随机 LUT，`L_bake`（权重 0.10）会在训练开始就把一个随机函数烘进 33³ 格点。
所以「identity-centered 局部 affine」+「零初始化头」+「归一化权重」三者是**联立不可能**的，
必须改其中一条。

### (b) 实现采用的修正是否正确且最小 —— **正确，且是最小改动**

实现取 `GLOBAL_AFFINE_MODE = "residual_zero"`：

```
T_pred(x) = clamp( ΔG·x + b_g + Σ_i q_i(x)·(M_i x + b_i), 0, 1 ),   M_i = I + ΔM_i, ΔG|_{init} = 0
```

**正确性**：我把 `q3vl/what/gaussians.py::render` 与 `model/glut_repro/model_rdg.py::render`
逐行对读，除命名（`gate`↔`existence`、`c`↔`b_g`）与 §7.6 新加的 `clamp` 外**逐项相同**：
同一个前代换 Mahalanobis、同一个 log-max 外提、同一个 `eps_term = exp(clamp(-m, max=80))·1e-6`
放在分母、同一个 `Σ_i w_i(M_i x + b_i) = (Σ_i w_i M_i)x + Σ_i w_i b_i` 展开。
而 `model_rdg.py::render` 已由 `ci_checks_rdg.check_render_matches_batched_glut` 对拍
`BatchedGLUT`（GLUT Eq.1-3）到 <1e-5。所以「等于 GLUT 官方渲染核」在本仓库是**测试**不是主张。
零初始化恒等我复跑得 `max|T−x| = 2.4e-6`（交付 preflight 记 1.49e-6，同量级）。

**最小性**：只有一项改变——`decode_global` 里 `G = 0.1·z` 而不是 `I + 0.1·z`。局部 affine 仍是
§7.6 要求的 identity-centered，混合权重、归一化 ε、SPD 构造、clamp 全部不动。
保留 `identity_centered` 分支**只**为让 2x 可被测量（`WT-P-zero-init-identity` 的 detail
里同时记了 `literal_formula_unclamped_ratio = 2.0000`）。这是正确做法：把红线证据写成数字而
不是注释里的论证。

### (c) 修正是否改变协议的科学问题 —— **否，且可证明**

两种写法的**函数族完全相同**：令 `G' = G + I`，则
`{x ↦ (I+ΔG)x + b_g + Σ q_i(M_i x + b_i)}` 与 `{x ↦ ΔG'x + b_g + Σ q_i(M_i x + b_i)}`
在 `ΔG, ΔG' ∈ R^{3×3}` 上取遍同一集合。差别**只在参数化的原点落在哪个函数上**。
由于 decoder head 的第二个 Linear 零初始化，那个原点恰好就是初始化点。

因此 amendment 改的是**初始化**，不是**假设类**：
- 表达容量不变；
- 每样本参数量不变（FG 1116 / SB 684，§7.4/§7.5 的数字都不动）；
- §12 的任何指标定义不动；
- §15 的五个论文问题一个都不动。

**审定结论：D-W1 修正应予批准，并以 amendment 形式写回协议 §7.6。**
理由不是「实现方便」，而是字面公式与本战役红线直接冲突且已被实测证伪；不改写协议的话，
每一轮实现审阅都会在同一处撞墙，而且任何后来者照字面重写渲染器都会复现 f(x)=2x。

### amendment 文本建议（可直接粘进协议）

在 §7.6 末尾，把公式段替换为：

> Gaussian 中心用 sigmoid 限定在 RGB cube；协方差由 Cholesky 构造并对角 softplus（带正下界
> `σ_lo = 0.02`），保证 SPD；opacity/existence 使用 sigmoid。
>
> **局部** affine 采用 identity-centered residual 参数化 `M_i = I + ΔM_i`；**全局** affine 采用
> **纯残差**参数化，`G` 零初始化（**不是** `I`）。令归一化权重为 `q_i(x)`（`Σ_i q_i ≤ 1`，
> 分母含 `ε = 1e-6`），则：
>
> ```
> T_pred(x) = clamp( G x + b_g + Σ_i q_i(x) * (M_i x + b_i), 0, 1 )
> ```
>
> **为什么不是 `(I + ΔG)x`**：`q_i` 归一化意味着 `Σ_i q_i(x)·M_i x ≈ x` 已经在混合项里给出了
> 恒等；若全局项再写成 `(I + ΔG)x`，零初始化时 `T(x) = 2x`——正是战役红线点名的失败
> （实测 `mean T(x)/x = 2.0000`，见 `preflight_what.json::WT-P-zero-init-identity`）。
> 本式与 `model/glut_repro/model_rdg.py::render`（已 CI 对拍 `BatchedGLUT` / GLUT Eq.1-3 到
> 1e-5）逐项一致，零初始化恒等实测 `max|T(x)−x| ≤ 2e-6`。
>
> 该改写是**参数化原点**的改写，不是假设类的改写：`G' = G + I` 的重参数化使两式的可达函数集
> 完全相同，表达容量、每样本参数量（FG 1116 / SB 684）与 §12 全部指标定义不变。
>
> 所有输出均可在固定 33³ RGB lattice 上求值并烘焙成标准 LUT。生产评测一律使用与交付一致的
> tetrahedral interpolation 回读。

并在协议末尾 changelog 追加：

> **2026-08-05 · amendment A-1（D-W1）**：§7.6 的 `T_pred` 公式由
> `(I + ΔG)x + b_g + Σ q_i(M_i x + b_i)` 改写为 `G x + b_g + Σ q_i(M_i x + b_i)`，`G` 零初始化。
> 依据：字面式在零初始化时给出 `f(x) = 2x`（实测比值 2.0000），与战役红线冲突；改写为
> `G' = G + I` 的重参数化，函数族不变、科学问题不变。由独立实现审阅
> （`docs/reviews/REVIEW-impl-What.md` §一）审定。

---

## 二、逐项 pass / blocker / nit

### 2.1 §9 Loss 逐符号（任务卡重点 2）

| 项 | 判定 | 依据 |
|---|---|---|
| `L_func` 2048 点、50/50 uniform/natural | **pass** | `queries.py`：uniform 固定 8³ strata × 2 抖动点 = 1024，`UNIFORM_SEED` 单次抽取后全局共享；natural 1024 对 `(seed, sample_id)` 用 sha256 确定。总 loss 对 2048 点取平均，故 50/50 由构造保证而非加权 |
| Charbonnier | **pass** | `sqrt(d²+ε²)`，ε=1e-3（D-W8 已声明） |
| uniform / natural 分别报告 | **pass** | `loss_func` 每步写 `L_func_uniform` / `L_func_natural`；`evaluate.sample_row` 也按 kind 分列 |
| `L_hc` 归一化 chroma 加权 hue cosine | **pass** | `normalised_chroma` 除以**全局常量** `√2`，非逐图/逐 batch 最大值（红线「s 禁逐图归一化」的同类纪律）；hue 差用 `(a1a2+b1b2)/(C1C2)` 避开 atan2 分支切割 |
| Lab 归一化的单位契约 | **pass** | `colorspace.py` 只暴露两种 Lab，`srgb_to_lab_norm = (L/100, a/128, b/128)` 是唯一允许进 loss 与 `v_i` 的形式。preflight 实测放大倍数 L=100.0、a/b=128.0，`lab_norm_abs_max = 0.977` |
| `grad_norm(L_func)` vs `grad_norm(10·L_hc)` 记录 | **pass** | `losses.grad_norm_ratio` 用真实两次 backward；trainer 每 50 步（含 step 0）写进 `steps.jsonl`。这是能在第一步抓住 A0 那次 229 倍事故的唯一手段 |
| `R_sparse` | **pass** | `0.5·(mean be(opacity) + mean be(existence))`，与协议 `mean(be(o)+be(e))/2` 等价 |
| `L_style_cos` | **pass** | `1 − cos(z_style, z_gt)` |
| `L_style_dist` | **BLOCKER B-3** | 见下 |
| `L_var` / `L_cov` | **pass** | `L_var = mean relu(1 − std_d)`（VICReg 式 `std = sqrt(var+1e-4)`）、`L_cov = Σ off-diag² / d`，与协议字面一致 |
| stop-gradient FIFO queue（单卡臂 → 该分支为操作分支） | **pass** | `StyleQueue` 固定 256 / 暖机 32，只 push `detach()`，类里没有任何逐臂旋钮（§9.3 末句「队列内容和更新规则对所有 arm 固定」被结构性满足）。queue 在 loss 之后 push，当前 batch 不会自己统计自己两次 |
| `L_bake` 同批查询点、bake_33 + tetra 回读、可微 | **pass** | `bake_readback(params, cfg, x)` 用同一 `x`；`bake` → `tetra_lookup` 全程可微，`compute_batch` 复用同一个 `t_read` 供 loss 与 `bake_metrics`，不会出现「loss 用一份、指标用另一份」 |
| 权重 1.00/10.00/0.001/0.05/0.05/0.02/0.10 | **pass** | `config.LOSS_WEIGHTS` 逐值，测试逐值断言，`arm_matrix.json` 也落盘 |
| `I_tar` 不进 `L_what` | **pass** | `compute_loss` 签名里没有任何图像参数；`WhatDataset.load_target_image` 的唯一调用方在 evaluate 路径 |

### 2.2 SRHT `z_gt`（任务卡重点 3）

**pass**。`Φ = sqrt(n/k)·S·H·D`：`D` 为 ±1 对角、`H` 为归一化 Walsh-Hadamard（FWHT 后除 `sqrt(pad)`）、
`S` 为 `randperm(pad)[:k]` 的无放回行采样、`scale = sqrt(pad/k)`。这是标准 JL 构造，
`E‖Φv‖² = ‖v‖²` 成立。实测（我复跑）：跨实例逐位相同（`reconstruction_max_abs_diff = 0.0`）、
距离比 mean 1.00006 / 区间 [0.935, 1.073]、`z_gt` 单位范数。17³ grid 与 `lattice_points` 同序
（R 最慢），`u(T)` 对恒等 LUT 严格为 0。`mean_train_u` 只用 train 的 `lut_id`——
`scripts/make_zgt_center.py` 把这条写死在 `TRAIN_SPLIT` 常量与 `center_report.json` 里，
并在模块 docstring 里说明「若用全语料求中心，§2.2 的 unseen-LUT 说法作废」。
`SRHT` 刻意不是 `nn.Module`，无法被误收进 optimizer group。`digest()` 存在，可跨机核对。

一处残余风险（已被 digest 覆盖，故不列 nit）：`torch.randperm` 的实现在 torch 大版本间理论上
可变，此时 `digest` 会变，因此会被发现而不是静默漂移。

### 2.3 §7.3–7.5 结构（任务卡重点 4）

| 项 | 判定 | 依据 |
|---|---|---|
| 2 seed + 4 refinement | **pass** | `BackendConfig.seed_blocks=2 / refine_blocks=4`；`_BaseGenerator.forward` 的次序就是 `run_seed → seed_geometry → pool_fn → run_refine → heads` |
| 每 block 512 / 8 heads / FFN 2048 / pre-norm | **pass** | `SlotBlock`：ModLN(pre) → self-attn → gated cross(M_color) → gated cross(WC) → gated v_i → ModLN → FFN |
| provisional geometry → aligned pooling → refinement 的梯度通路 | **pass（附 N-6）** | `pool_fn(geom)` 里没有 detach，梯度确实能回到 provisional head。实现者自己指出并测试了零初始化门的后果：第 0 步 `gate_v = 0` 使 `∂loss/∂v = 0`，此时有梯度的是 `gate_v` 本身；门离开 0 后通路打开。测试拆成两条（`test_the_v_gate_receives_gradient_at_step_zero` / `test_gradient_reaches_the_provisional_geometry_through_the_pooling`）而不是把断言放宽——这是正确处理 |
| zero-init gated residual 恒等断言 | **pass** | 三个 gate（color / wc / v）都是零初始化标量；`SlotHeads.w2`、`b2`、`global_head[-1]` 全零初始化，故 `z_prim = z_glob = 0` → `T(x) = x`（实测 1.49e-6） |
| ModLN | **pass** | `LayerNorm(elementwise_affine=False)` + 零初始化 `Linear(1024, 2·512)`，初始即普通 LayerNorm；每子层独立一份（RD-G / PLAN 1.4 先例） |
| 48 独立 head + global head | **pass** | 批量参数 `(48, in, out)` 的 einsum，不共享任何权重；测试断言 head i 的输出只依赖 slot i。global head 读 `[mean_i h_i, P(z_style)]` |
| FG/SB 参数量差 0.013% | **pass（复算通过）** | 我手算：`in_dim = 512+128 = 640`；FG 增量 = 48·(640·128+128+128·23+23) + (640·128+128+128·12+12) + 48·(512·9+9) = 4,385,932；SB 增量 = 32093·b + 1116，b=137 → 4,397,857；差 **11,925**。与 `arm_matrix.json` 的六对 `abs_diff = 11925`、`rel_diff = 1.27e-4 ~ 1.31e-4` 完全一致。preflight WT-W4 在战役环境复跑亦得 61,394,716 / 61,406,641 |
| `solve_sb_bottleneck` 的搜索正确性 | **pass** | 目标函数在 b 上严格单增，`elif value > target: break` 的提前退出在 b=138 才触发（b=137 时 err=11,925 < b=136 的 20,168），不会漏掉最优 |
| SB shared geometry：4×4×3 anchors 初始化 | **pass** | `anchor_points` 取**cell 中心** `(i+0.5)/n`——因为 `mu` 过 sigmoid，0.0/1.0 没有有限前像。`GeometryBank.raw` 零初始化 → `mu = anchor`、`σ = σ_init` |
| SB shared geometry 独立 lr | **pass** | `_group_of("generator.geometry.raw") → "geometry" → 5e-5`；`_no_decay` 命中 `"geometry" in name` → 无 WD |
| FG/SB 同深同宽同 query 数 | **pass** | 两者共用同一个 `SlotBackend(cfg)`，唯一差别是 head 的 `n_out`(23/14)、bottleneck 与 provisional head / GeometryBank |

### 2.4 参数域与 D-W2（任务卡重点 5）

**pass**。`mu = sigmoid(z + logit(anchor))`（闭 cube，且 z=0 时回到锚点，而不是 48 个中心叠在 0.5）；
`σ = 0.02 + softplus(raw + softplus_inv(0.18))`，`raw=0` 时 σ=0.20，严格正下界保证 SPD 与条件数；
`off` 只缩放不约束（下三角，SPD 只需对角为正）；opacity/existence 走 sigmoid，偏置 −2 / +4 沿用
RD-G（恒等 CI 依赖 gate 是**公因子**、在归一化器里消去——这条实现者引用得准确，
`render` 里 `og = opacity * existence` 确实在分子而非载荷上）；输出 `clamp(0,1)`。
preflight WT-P12 实测 `mu_in_cube=True`、`σ ∈ [0.031, 1.418]`、`Σq_i ∈ [0.9999965, 1.0000002]`、
两组 raw 的梯度全有限。

D-W2 的裁定（`softplus_floor`）已落实为默认；`bounded_sigmoid`（RD-G 已验证的 `0.02+0.48·sigmoid`）
保留为一常量之隔的开关，两条路径都有域测试。**主 agent 已裁定 softplus_floor，实现与裁定一致。**
提醒结果审阅：softplus 无上界，实测已出现 σ=1.42（远大于 RGB cube 的边长 1），
此时该 Gaussian 近似均匀覆盖全 cube、退化为一个额外的全局 affine 分支。这不是 bug，
但「48 个基元有多少个塌成全局项」应当作为 §12.3 的一个读数记录（见 N-6 的同类建议）。

### 2.5 WC 接口 §6 四种 + C01–C04（任务卡重点 6）

| 项 | 判定 |
|---|---|
| WC-0/1/2/3 的 token 集与 `mask_pool` 与 §6 表逐行一致 | **pass**（`config.WC_INTERFACES`；WC-1/WC-3/ORACLE 开 mask_pool，WC-0/WC-2/NOWHERE 不开） |
| `F_roi` / `F_bg` 公式 | **pass**（`roi_bg_pool` 与 §6 逐符号一致，含 padding 处理） |
| D-W4（WC-0 不吃 `m_pred`） | **模型输入侧 pass，loss 侧 BLOCKER B-5** |
| D-W5（oracle `z_where` 用 `OracleLatentEncoder(w*, ρ*)`） | **pass**。把**预测的** query state 喂给 oracle 臂会让上界不成其为上界；实现拒绝这么做，并在 `WCEncoder.facts()["z_where_from"]` 记 `"oracle_latent"` |
| D-W6（C01/C02 图像指标仍用冻结 `m_pred` 合成） | **pass（附 N-5 标签问题）**。`evaluate.py` 模块 docstring 写明理由，每行记 `composite_mask` |
| oracle 输入只进 ceiling control 的隔离 | **pass**。`ARMS` 里只有 C03/C04 是 `"oracle"`；`preflight.check_no_target_leak` 断言 `{oracle arms} == {C03, C04}`；`evaluate.main_board` 用 `is_ceiling` 把两臂剔出主榜，`ceiling_board` 单列 |
| `Q_color` 不读 `H_where` | **pass**。`ColorStack.forward(h_color, h_color_mask)` 签名封闭并被 preflight 断言；`color.py` / `attention.py` 的 AST 标识符里不存在含 `where` 的名字（`torch.where` 按**限定名**放行，故裸 `where` 局部变量仍会被抓）。`attention.py` 刻意重写而不 import Where-B 的 connector，正是为了让这个证明是结构性的 |

### 2.6 `T_gt` 管线（任务卡重点 7）

**pass**。

- `preset_path` → `dataset_build.lut_io.load_lut` → **一次**转置 `(2,1,0,3)` 到 `table[r,g,b]`，
  之后全链路不再记轴序。`test_cube_round_trip_axis_order` 用真实 `.cube` 回读——考虑到
  `render_backend` 2026-07-17 修过一次 `f(B,G,R)` 轴序 bug，这个测试是必要的而不是装饰。
- **按原生网格求值**：`GtLutTable` 不重采样，`_corner_setup` 用 `K = table.shape[1]`，
  16/25/32/33/64/65 各尺寸都按各自的 K 插值。
- **拒绝 npy33 缓存**：全包 grep `npy33` / `dcube` 零命中；NOTES V-W12 给了拒绝理由
  （用重采样到 33³ 的缓存会把 32³/64³ 原生表的重采样误差注入监督目标）。这条判断正确。
- **D-W3（三线性监督 / 四面体并排）**：`GT_LUT_INTERP = "trilinear"`，与 `I_tar` 的生成方式
  （`grid_sample(mode="bilinear", align_corners=True, padding_mode="border")` 及其 CPU oracle）
  一致；四面体口径由 §12.1 的并排指标与 `L_bake` 始终报告。**主 agent 已裁定三线性，实现一致。**
  `z_gt` 与 `t_gt` 用同一个 `gt_interp`，不会出现两套定义。
- **`T_lut_unseen` 的 lut_id 隔离强制**：`WT-W2` 是**全量 index 扫描**（不是抽样），我独立复跑
  得 259 / 0 / 0 / 0 / 0。§2.2 的「unseen LUT generalization」说法成立。
- 无法解析 `T_gt` 的样本走 `LutBank._load` 的 `KeyError`，错误信息明确写「必须拒绝样本，
  不得用伪造目标训练」——符合「失败样本进入显式 rejection，不静默换成零向量」的纪律。

### 2.7 跨实现对拍的证据力（任务卡重点 8）—— **pass，且不是自证**

我逐条确认这三组对拍的**对照方是外部或生产代码**，不是本包自己的另一份拷贝：

| 对拍 | 对照方 | 是否外部 | 我复跑结果 |
|---|---|---|---|
| 三线性 | `dataset_build.src.construct.rendering.apply_lut_cpu_oracle` | **是**——这就是生产 `I_tar` 的那份算术；测试还刻意把本包的 `[r,g,b]` 表转置成 `[b,g,r]` 再喂给对照方，等于同时验证了轴序 | pass（base 环境，<1e-5） |
| 四面体 | `colour.algebra.table_interpolation_tetrahedral` | **是**——第三方库 colour-science | pass（base 环境，<1e-9，double 精度） |
| 四面体 | `model/glut_repro/model_rdg.py::tetra_lookup` | 同仓库，但该实现已由 `ci_checks_rdg` 独立对拍 colour-science | pass（<1e-6） |
| Lab / ΔE00 | `model/glut_repro::srgb_to_lab` / `delta_e00`，以及 `skimage.color.rgb2lab` | skimage 为外部 | pass（glut_repro <1e-4；skimage 差 ~0.015 Lab 单位，NOTES V-W9 已定位为**白点约定**差异而非精度，选择与本仓库既有实现一致——这个归因我核对过，`_XN = 0.3127/0.3290` 是色度导出值，skimage 用 ASTM 表值） |

我另外独立验证了四面体的 6 个 case 覆盖全部 6 种大小序、两两互斥，且每个 case 的四个顶点与
权重 `(1−a, a−b, b−c, c)` 均为标准 Kasson 分解——即使没有外部库也站得住。

三个 skip 只发生在战役环境（缺 `skimage`/`colour`，`dataset_build.src.construct.rendering`
因 libstdc++ 版本 import 不了）。生产路径只用 `dataset_build.lut_io`（纯 numpy），不受影响。
**结论：对拍证据力充分。**

### 2.8 §12 评测与 gate 脚本、§12.4 字典序、hidden 口径（任务卡重点 9）

| 项 | 判定 |
|---|---|
| §12.1 指标齐备（MAE/RMSE/PSNR、ΔE00 mean/median/p90/p95、hue/chroma、analytic vs baked、out-of-range、non-finite） | **pass** |
| bake gate 三条阈值常量 | **pass**（`BAKE_GATE = 1e-4 / 5e-4 / 0`，逐值断言） |
| bake gate 的聚合口径 | **pass**（`bake_err_p99` 取全 arm 的 **max**、`bake_non_finite` 取 max，不是平均掩盖） |
| §12.2 分区（内部/3px 边界带/外部）与分层 | **pass**（`boundary_band` 用 max_pool 膨胀−腐蚀；`STRATA_KEYS` 覆盖 build/render_mode/winner_confidence/upscaled/mask_area/L-level） |
| §12.2 LPIPS | **pass（诚实缺项）**——无后端时报 `nan` 并显式列在 `PREFLIGHT_WHAT_PENDING.md` 的 `WT-G7`，不做静默替代 |
| §12.4 只在 `V_what` 选择 | **pass**——`main_board` 对非白名单 split 抛 `PermissionError`，「我们偷看了 T_final」必须是一次刻意的传参 |
| §12.4 字典序键与方向 | **pass**（`SELECTION_ORDER` 五键、全部 smaller-is-better） |
| §12.4 step 1（gate 作为选择前置） | **BLOCKER B-1** |
| 每 arm 只有一个 checkpoint 进跨臂排名 + top-2 必须是两个不同配置 | **pass**（`best_per_arm` + `top2_distinct_arms`） |
| checkpoint 选择禁用 val loss（红线） | **pass**——`WhatTrainer.best` 只按 `evaluate` 产出的榜单键排序，docstring 明写「never eval_loss」 |
| hidden 口径 import `q3vl/whereb/contracts.py` | **pass**——`config.py` 由 `from q3vl.whereb.contracts import ...` 引入并改名 `COLOR_HIDDEN_*`，无重声明；`WhatVLM.__init__` 的默认值取自该常量。whereb 侧扫描测试的 root 是 `q3vl/whereb`（不覆盖 what），本包自带了同型的 what-root 扫描测试，两者都过（见 N-13） |

### 2.9 12 臂 arm_matrix 与 §8 表逐行对照（任务卡重点 10）—— **pass**

| §8 表 | `config.ARMS` / `arm_matrix.json` | 一致 |
|---|---|---|
| T01 WC-0 + FG48 | `("WC-0","FG48","predicted")` | ✓ |
| T02 WC-1 + FG48 | ✓ | ✓ |
| T03 WC-2 + FG48 | ✓ | ✓ |
| T04 WC-3 + FG48 | ✓ | ✓ |
| T05 WC-0 + SB48 | ✓ | ✓ |
| T06 WC-1 + SB48 | ✓ | ✓ |
| T07 WC-2 + SB48 | ✓ | ✓ |
| T08 WC-3 + SB48 | ✓ | ✓ |
| C01 NoWhere-FG48 | `("NOWHERE","FG48","none")`，`where_prefix=False`、`mask_pool=False` | ✓ |
| C02 NoWhere-SB48 | 同上 SB | ✓ |
| C03 OracleWhere-FG48 | `("ORACLE","FG48","oracle")`，`is_ceiling=True`，六 token 全给 | ✓ |
| C04 OracleWhere-SB48 | 同上 SB | ✓ |

12 臂全部可构造并前向（preflight `WT-P-arm-matrix` pass，我复跑确认）。
`4 WC × 2 generator + 4 control = 12`，无减配、无混用。

### 2.10 §10.4 优化器 —— **pass**

三组 lr（backend/head 1e-4、geometry 5e-5）、WD 0.01、warmup 0.03、cosine、clip 1.0、bf16、
effective batch 32（`grad_accum()` 在 micro 不整除时直接抛错）、1 epoch、eval/save 500、
keep 3、保护 0.5/1.0 epoch —— 全部逐值落实并有测试。`_no_decay` 覆盖 geometry / bias /
LayerNorm / **ModLN**（`".mod_"` 匹配到 `mod_self`/`mod_q_color`/`mod_q_wc`/`mod_ffn` 的
`proj.weight`），逐参数断言存在。

---

## 三、BLOCKER 清单（6 项）

### B-1 · §12.4 step 1 的 gate 根本没有参与选择
**位置**：`q3vl/what/evaluate.py::main_board`（L191-201）、`lexicographic_best`。
**事实**：`gate_pass` 只被写进行里和 `any_gate_failed`，既不参与「每臂选一个 checkpoint」，
也不参与跨臂排序。我构造性验证：

```
两臂：T01 gate_pass=False / dE00=1.0，T02 gate_pass=True / dE00=2.0
→ ranked = [('T01', False), ('T02', True)]，top2 第一名是 gate 未过的臂
同一臂：step500 gate_pass=False / dE00=1.0，step1000 gate_pass=True / dE00=1.5
→ 进榜的是 step500（gate 未过）
```

§12.4 明写「**先**满足 bake gate、finite gate 和 instruction-shuffle 正向依赖」。这是过滤器，
不是标签。考虑到 R1（bake gate 可能所有臂都过不了），正确形态是：把 `gate_pass` 作为
字典序的**首键**（过 gate 的一律排在未过的前面），全部未过时仍出排名但整榜打
`WHAT-GATE-FAILED` 标记——与 §5.6 给 Where 的 `WHERE-GATE-FAILED` 同构，也正好落实
主 agent「gate 不得事后放宽，不过则按 §15 分阶段报告」的裁定。
**修法**：`main_board` 的两处排序键前置 `not r.get("gate_pass")`；board 级加 `tag`。

### B-2 · 滚动删除会删掉将被选中的 checkpoint
**位置**：`q3vl/what/trainer.py::_roll`（L310-316）。
**事实**：`keep_last=3`，`save_steps=500`，约 4975 步 → 约 9 次普通保存，前 6 次被 `unlink`。
`_eval_and_record` 只把指标写进 `state.checkpoints`，不引用 checkpoint 文件，也不把当前最优
标记为 protected。于是若某臂的 `V_what` 最优出现在 step 500–3000，选择时文件已不存在。
§10.4 明写「被选中和保护的 checkpoint 不受滚动删除影响」；§12.4「同一 arm 的多个 step 只保留
最佳 checkpoint 进入跨 arm 排名」也预设文件还在。Where-B 的 trainer 根本没有滚动删除，
这个风险是 Stage-What 新引入的。
**修法**：`_eval_and_record` 之后按 §12.4 键重排 `state.checkpoints`，把当前最优对应的
`saved` 条目置 `protected=True`（并把被它替下的还原为普通）；或直接 `keep_last=None`
（9 个 checkpoint × 约 0.4 GB ≈ 3.5 GB/臂，代价可接受）。

### B-3 · `L_style_dist` 的 `d_func(T_i, T_j)` 被替换成 `d(z_gt_i, z_gt_j)`，且未声明
**位置**：`q3vl/what/losses.py::loss_style_dist`（L127-146）。
**事实**：协议 §9.3 写 `L_style_dist = Huber(d_style(i,j), d_func(T_i, T_j))`。实现两侧都用
**L2 归一化后**的码：`d_func := ‖ẑ_gt_i − ẑ_gt_j‖`。因为 `z_gt = L2Norm(SRHT(u − mean))`，
归一化把 `u(T) = T(x) − x` 的**幅度**信息全部丢掉：两个只差强度（比如同一 look 的 50% 版本）
的 preset，其 `d_func` 在这个替换下为 0，而真实函数距离很大。本仓库语料是 Lightroom preset，
强度变体是常见的。
叠加 `L_style_cos` 也只管方向，结果是 **`L_what` 里没有任何一项监督 `z_style` 的编辑幅度**。
这不是笔误级别的自由度：它改变了 §15 问题 5（连续函数码能否泛化）的答案含义。
而且它**不在** NOTES 的 D-W1…D-W8 待决策清单里，属于对冻结公式的**未声明**替换；
§9.5 禁止事后改 loss，所以必须现在定。
**修法（二选一，需主 agent 裁定）**：
(a) `d_func := ‖u_i − u_j‖ / c`，`c` 为 train 集上 `‖u_i − u_j‖` 的 RMS，与 `mean_train_u`
一起在 `make_zgt_center.py` 里算好并发布（一个常量，不逐 batch、不逐图，不触红线）；
`d_style := ‖ẑ_style_i − ẑ_style_j‖`。
(b) 明确批准「用归一化码距离作为 `d_func`」，写进 amendment，并在 REPORT 中声明
`z_style` 不携带幅度信息。
我倾向 (a)：它才是协议字面，并且顺带解掉 B-4。

### B-4 · §12.3 的防坍缩诊断与它要检验的 loss 同源（循环论证）
**位置**：`q3vl/what/metrics.py::style_diagnostics`（L230-237）。
**事实**：`z_dist_spearman_vs_gt` 用的正是 `L_style_dist` 直接优化的那组 `z_gt` 成对距离。
§12.3 要求的是「`z_style` pairwise distance 与 **GT LUT function distance** 的 Spearman 相关」。
用被优化的目标当独立证据，这个诊断在 0.05 权重下仍然会偏高，无法回答「码是否真的编码了函数」。
**修法**：与 B-3 同一处修——把真实 `u` 距离（或其发布的成对尺度）带进 target dict，
Spearman 对它算；两者都报也可以，但独立那个必须存在。

### B-5 · C01/C02 与 T01/T05 有**两处**差别，与已裁定的 D-W4 矛盾
**位置**：`q3vl/what/data.py::WhatBatchBuilder.query_points`（L357-364）+ `_where_signals`（L436-440）。
**事实**：D-W4（主 agent 已按保守默认采纳）声明「T01/T05 与 C01/C02 的**唯一**差别就是语言
序列里还有没有 `<where>` 段」。模型输入侧确实做到了。但 §9.1 的 natural 半区采样：

```python
w = None if sample.is_global else m_hi          # data.py:360
```

对 `where_source == "none"` 的 C01/C02，`WhereSignals(source="none")` 的 `m_hi is None`，
于是 local 样本也退回**全图**采样；而 T01/T05（`where_source="predicted"`）拿到 `m_hi`，
按 `m_pred` 加权采样。两臂的 **loss 查询色分布不同**，这是第二处差别，且未声明。
C03/C04 同理（按 **GT mask** 加权），使 ceiling 臂的目标分布又是第三种。
**修法（需主 agent 裁定）**：§9.1 的 `m_pred` 是**监督侧**的冻结量，与 `T_gt` 同性质，不是
模型输入；§9.5 又说「该配方是所有 12 个 What 臂的统一起点」。因此正解是
(a) 让 12 臂**都**用同一个冻结 Where 的 `m_pred` 加权 natural 半区（C01/C02 也照常跑
`WhereRunner`，只是其输出不进模型）；备选 (b) 保留现状但登记为新决策项 D-W9，并在每条
per-sample 记录里写 `natural_weighting` 字段。我倾向 (a)——它才让 C01/C02 是干净的下界。

### B-6 · Where checkpoint digest 校验被写进文档，但代码里不存在
**位置**：`q3vl/what/scripts/run_what.py`（docstring L6-10 与 L100-110）；NOTES §七同款描述。
**事实**：docstring 写「its digest goes into `run_setup.json` and a mismatch with a previous
arm's record is a hard stop」。实际只记录了 `path` / `where_arm` / `step` / `basis_digest`
（**basis** 的 digest，不是 Where checkpoint 的），也没有任何跨臂比对。§6 要求
「Where checkpoint 对所有 What arm 完全相同且冻结」——12 个臂跨 4 个 wave 分批起跑，
这正是最容易出错的地方，而现在没有任何机制会发现。
**修法**：`sha256` checkpoint 文件写进 `run_setup.json`；启动时扫描 `RUN_ROOT/*/run_setup.json`，
digest 不一致即 `SystemExit`。

---

## 三-bis · 审阅过程中由审阅人造成的一次事故（自我披露）

审阅初期我执行了 `python -m q3vl.what.preflight --no-data`，**没有注意到该脚本 `--out` 的
默认值就是交付目录** `experiments/.../what/preflight/`。这次运行把交付的
`preflight_what.json` 覆盖成了一份 `--no-data` 的产物（9 pass / 2 skip / `complete: false`）。
我一度据此写下一条「文档声称 11/11、落盘却是 9+2」的 blocker——**那条是我自己造成的，
不是实现者的缺陷**，现已撤销。

发现方式：文件 mtime 07:55:36 与同目录其它交付物 07:38–07:44 相差 11 分钟，恰好落在我第一次
运行的时刻。

处置：我在战役环境按 `PREFLIGHT_WHAT_PENDING.md` 记录的原命令 `--limit 400` 重跑并覆盖回去，
现在文件是 `ok=true / complete=true / 11 pass / 0 fail / 0 skip`，`WT-W1` 与 `WT-W2` 的 detail
与文档声称的数字逐项一致。**实现者的 preflight 声明经核实为真。**

由此得到一条对实现者的改进建议（列为 N-16）：**preflight 脚本的 `--out` 默认值不该是交付目录**。
任何人跑一次带默认参数的 preflight 就会静默覆盖交付证据，而覆盖后的文件看起来完全正常
（`ok: true`），只有 `complete` 字段和 `skipped` 列表会变——这正是本战役 s 缓存契约里
说的「第二种失败模式是静默的」。建议默认写到 run_dir 或要求显式 `--out`，
并在写入前对已存在的文件做 `complete` 降级检查（从 complete=true 覆盖成 complete=false 应拒绝或备份）。

---

## 四、NIT 清单（16 项）

- **N-1**：`config/arm_matrix.json` 与 `config/mock_e2e.json` 的环境戳是 `python 3.13.5 /
  torch 2.6.0+cu124`（base conda），不是战役环境（3.12.12 / 2.10.0+cu128）——与主 agent 裁定
  R3（正式作业一律 q3vl_sft）不符。参数量我在战役环境复跑得同值（61,394,716 / 61,406,641），
  所以没有错数字，但两份交付物的 provenance 应重新生成。
- **N-2**：`colorspace.hue_cos_diff` 的 docstring 说灰点「yields 0」；实际返回 1.0
  （`cos = 0/(ε·ε) = 0`）。复合的 `L_hc` 仍≈0（因为 `normalised_chroma ≈ 7e-7`），loss 正确，
  注释错误。改注释即可。
- **N-3**：`pooling.aligned_pool` 每次前向都做 `torch.isfinite(v).all()` 断言并把 6 个
  `int()/float()` 统计取回主机——每 micro-batch 一次 device sync。改成按 `log_every` 周期检查。
- **N-4**：`WhereSignals.m_hi` 标注 `torch.Tensor | None`，两个生产者（`WhereRunner.signals`、
  `_oracle_signals`）都返回 `list[Tensor]`（各图网格不同，无法 stack）。标注应改。
- **N-5**：`evaluate.sample_row` 的 `composite_mask` 只有 `"predicted"` / `"ones"` 两值，
  于是 C03/C04 的 **GT mask** 会被记成 `"predicted"`。D-W6 让这个字段成为审计凭据，标签必须准。
  加一个显式的 `mask_source` 入参。
- **N-6**：FG48 用 **provisional** geometry 做 aligned pooling，用 **refined** geometry 渲染，
  两者之间没有任何约束项。这是 §7.4 的字面读法，实现无误；但训练后两套 geometry 可能分道扬镳，
  届时「每个 Gaussian 对应的真实颜色分布」（§7.2）就不再成立。建议把
  `mean‖mu_prov − mu_refined‖` 与 `σ` 的同类差写进 `steps.jsonl`，供结果审阅归因。
  同处建议：记录 σ 越过 cube 边长（>1.0）的基元数——softplus 无上界，preflight 已见 σ=1.42。
- **N-7**：`arm_metrics["lut_de00_p90"]` 实为「每样本 p90 的跨样本中位数」。§12.4 的
  「LUT function CIEDE2000 p90」更自然读作汇总分布的 p90。二选一并写进 REPORT，或两者都报。
- **N-8**：`instruction_shuffle_delta` 比较两组聚合中位数，不是**配对**差。同一批样本本来就
  两次都跑，改成配对差是免费的，且严格更强。
- **N-9**：§12.3 的 **image-shuffle** 负控制、**WC-1/2/3 相对 WC-0 的 paired improvement**、
  以及 §10.4/§12.4/§13.1 的 **paired bootstrap 95% CI** 都未实现。image-shuffle 已列
  `WT-J4`；后两项**没有出现在任何 pending 清单里**。请补进 `PREFLIGHT_WHAT_PENDING.md`。
- **N-10**：§12.1/§13 要求最终图像同时给 analytic render 与 **33³ baked render**；
  `sample_row` 只渲染 analytic（bake 只在 LUT 函数层面对比）。交付前需补，且当前不在 pending 清单。
- **N-11**：渲染器参数（`mu/sigma/off/M/b/G/b_g`）是在 bf16 autocast 区内解码的，
  `compute_batch` 才 `.float()`。docstring 的「renderer 之后全 float32」成立，但**喂给** renderer
  的参数带 bf16 舍入（相对约 4e-3）。这对 bake gate 无影响（analytic 与 baked 用同一份参数），
  属标准混合精度；但 `WT-G6` 应把 `out.params` 与 `aligned_pool` 内 Mahalanobis 的实测 dtype
  一并记录，不要只查外层 autocast 泄漏。
- **N-12**：`StyleQueue` 把 256×1024 保存在 CPU，每个 micro-batch `.to(device)` 一次
  （约 1 MB/step）。放在 device 上即可。
- **N-13**：hidden 契约的重声明扫描存在两份（whereb-root 与 what-root），各自只覆盖自己的包，
  未来第三个阶段仍会漏。建议把扫描提到 `q3vl/` 根的共享 helper。
- **N-14**：`Batch.check_inputs` 只校验键名，不校验 `where.source` 与 `cfg.where_source` 一致。
  主臂被塞进 oracle `WhereSignals` 不会报错（现有 sanctioned 路径不会发生，但 §8.2 值得一行断言）。
- **N-15**：`lexicographic_best` 平局取第一个参数，于是 `main_board` 的每臂循环在完全平局时
  偏向**较晚**的 checkpoint。无害，但应显式写明。
- **N-16**：`q3vl/what/preflight.py::main` 的 `--out` 默认值是交付目录
  `REPORT_DIR / "preflight"`。任何人跑一次默认参数的 preflight（尤其带 `--no-data`）就会
  **静默覆盖**交付证据，且覆盖后的文件仍然 `ok: true`，只有 `complete` 与 `skipped` 会变
  ——审阅人本人就踩了这个坑（见 §三-bis）。建议默认写 run_dir 或强制显式 `--out`，
  并在覆盖前拒绝 `complete: true → false` 的降级（或先备份）。

---

## 五、值得表扬的做法（供后续任务卡复用）

1. **红线冲突被做成可测量的数字而不是注释里的论证**：`identity_centered` 分支保留下来，
   专门用于让 `T(x)=2x` 出现在 `preflight_what.json` 的 detail 里。
2. **「证明」是结构性的而不是约定性的**：§14.8 用 AST 标识符扫描 + 封闭 forward 签名，
   并且 `attention.py` 刻意重写而非 import Where-B 的 connector——正是为了让「一个 import 边
   之外就有 `h_where`」这件事不成立。
3. **失败模式被分成两条测试而不是放宽一条**：零初始化门导致第 0 步梯度不经 pooling 回 geometry，
   实现者没有把断言改松，而是拆成「gate 本身有梯度」与「门打开后通路成立」两条。
4. **拒绝 npy33 缓存**：宁可按原生网格（16/25/32/33/64/65）求值，也不让重采样误差进监督目标。
   这个判断直接关系到 §12.1 的 1e-4 bake gate 是否有意义。
5. **R1 被预先写成「这是实验结果，不是实现缺陷」**：随机参数下 48-Gaussian 的
   analytic→33³→tetra 回读 MAE 2.19e-4 / p99 4.01e-3（gate 的 2.2 倍 / 8 倍），
   同时给出仿射函数 4.8e-7、格点 0.0 的对照，证明插值器本身无损。这是正确的归因方式。

---

## 六、判决

**BLOCKER 数量：6**（B-1 … B-6）。**NIT：16**。
（初稿曾有第 7 条 blocker，经查是审阅人自己覆盖交付文件所致，已撤销并复原——见 §三-bis。）

**是否准许进入 GPU preflight 与正式训练（Where 定档后）：**

- **GPU preflight `WT-G1`–`WT-G8`：准许**（在两卡从 Base SFT 释放后）。
  6 个 blocker 没有一个会改变 `WT-G1`–`WT-G8` 所测量的对象（`H_color` 切片契约、
  `where_prefix` 的序列差、`F_pre` 形状、冻结 Where 接入、显存/吞吐、bf16 数值、LPIPS、延迟），
  提前跑掉可以给排期解压。建议把 **N-11** 的 dtype 记录并入 `WT-G6`。
- **正式训练（12 臂任何一臂）：不准许**，直到 B-1 … B-6 全部清零。理由分三档：
  - **B-3 / B-5 必须在第一个臂起跑前定档**——§9.5 明禁「看到主实验结果后改 Loss 再只重跑失败臂」，
    这两项都在 loss / 控制臂定义里，跑完再改等于全部作废；
  - **B-2 / B-6 会在跑的过程中静默毁证据**（删掉将被选中的 checkpoint、放跑不同 Where checkpoint
    的臂），事后无法补救；
  - **B-1 / B-4 是交付/选择的正确性问题**，可以与训练并行修，但必须在 `V_what` 选择发生前完成。
- B-3、B-5 需要**主 agent 裁定**（两者都是「两种做法都合理、影响后续」的决策，
  我在各自条目里给了倾向与理由）；B-1、B-2、B-4、B-6 是纯实现修复，无需裁定。
- 另需主 agent 走完的动作：把 **D-W1 amendment（本文 §一末的文本）** 追加进协议 §7.6 与
  changelog，此后 `GLOBAL_AFFINE_MODE = "residual_zero"` 即为规格，不再是「实现偏离」。

清完 B-1 … B-6 后**无需重做**已通过的 11 项 CPU preflight（现已在战役环境复跑并落盘，
`complete: true`）与 158 个单测中与这些 blocker 无关的部分；但涉及 loss 的测试（B-3/B-4）
必须重跑并重新落盘。

> **本节判决已被 2026-08-05 的聚焦复审取代，见下方 §七。**

---

# 聚焦复审（2026-08-05，commit `cf17089`）

- 范围：只审修复 diff 与其单测 + 协议 amendment A-1/A-2/A-3 的文本忠实性。
- 方式：只读；未用 GPU（Base SFT rank PID 3395226/3395227 全程 54 GiB 正常运行）；
  复跑用 `/home/bc/envs/q3vl_sft/bin/python`。
- 复跑：`pytest q3vl/what/tests -q` → **188 passed, 3 skipped**（skip 仍是该环境缺
  `skimage`/`colour`/`sqlite3`，base 环境下全过），与实现者报数一致。

## 七、六个 BLOCKER 的复审结论

### B-1 · §12.4 step 1 变成真过滤器 —— **pass**

`main_board` 现在先按 `gate_pass` 切成 `passed` / `failed`：`_best_per_arm(passed)` 决定
每臂代表，`_rank` 只排过 gate 的；没有任何过 gate checkpoint 的臂被单列进 `gate_failed`
（按其自身最好的失败 step），不与主榜混排。我用初审那两个反例复跑，行为已翻转：

- 两臂（T01 gate 未过 / dE00 1.0，T02 过 / dE00 2.0）→ `ranked` 只剩 T02，T01 落 `gate_failed`；
- 同臂两 step（step500 未过 / 更好，step1000 过）→ 进榜的是 step1000。

全臂未过时：`ranked = []`、`top2 = []`、`selection_possible: false`、整榜 `tag =
WHAT-GATE-FAILED`，另出 `diagnostic_ranked` 供调试。这正是 §5.6 给 Where 的形状转置过来，
也正好落实「gate 不得事后放宽，不过则按 §15 分阶段报告」的裁定——**选择被禁用，诊断被保留**，
两者分在不同键上，不会被误读成一次选择。

`WhatTrainer.best` 同步收紧（`gated = [c for c in scored if c.get("gate_pass", True)]`），
`test_b2_best_skips_gate_failing_checkpoints` 覆盖。N-15 的平局方向也一并明确
（`_best_per_arm` 保留先到者 = 较早 step，docstring 写明理由）。

### B-2 · 滚动删除不再吃掉将被选中的 checkpoint —— **pass**

三处结构性改动，缺一不可，都到位了：

1. `_roll` 不再把已删条目移出 `state.saved`，而是留 `deleted: True`。这是后面两点能成立的前提；
2. `save()` 的顺序改成 **append → `_protect_best()` → `_roll()`**，即先给新文件挂上保护再滚动；
   `_eval_and_record()` 末尾也调 `_protect_best()`。**两条路径都算**，符合任务卡要求；
3. `_protect_best` 用两个**分离**的标志：`protected`（0.5/1.0 epoch 与 final，永不清）与
   `best_protected`（随最优移动，全局只留一份）。

「文件未写」vs「已被删」的区分**严密**：
- `entries` 非空 ⟺ `save()` 曾为该 step 追加过记录（且记录永不被移除）；
- `entries` 空 → eval 先于 save 的良性时序，静默（否则每次 eval 都告警，告警就没人看了）；
- `entries` 非空但 `alive` 空 → 文件确实被删过，进 `lost_best_steps` 显式告警。

`TrainState.lost_best_steps` 有 docstring 写明「必须保持为空，非空即 §10.4 被违反」。
`keep_last=None` 直接短路 `_roll`。回归测试 7 条，含「eval_steps==save_steps」与
「save_steps<eval_steps」两种交错（S0-TRAIN 双删除路径的教训被显式引用）。

单调性我另行确认过：`best()` 取全历史最小，一旦 B 胜过 A，A 不可能再变回最优，
所以「只留一份 `best_protected`」不会丢掉未来还会用到的文件。

### B-3 / amendment A-2 · `d_func` 恢复协议字面 —— **pass**

- `pairwise_d_func(u_gt, C) = ‖u_i − u_j‖₂ / C`，`u_gt` 是**原始**
  `flatten(T(x) − x)`（未过 SRHT、未 L2 归一化）；`data.py::targets_for` 现在同时产出
  `u_gt` 与 `z_gt`，两者来自同一次 `table.apply(17³ grid)`。
- **闭式 `C` 的正确性：我独立验算，逐位吻合。** 推导
  `Σ_{i<j}‖u_i−u_j‖² = ½Σ_{i,j}(‖u_i‖²+‖u_j‖²−2⟨u_i,u_j⟩) = N Σ‖u_i‖² − ‖Σu_i‖²`，
  除以 `N(N−1)/2` 即
  `C² = 2(N Σ‖u_i‖² − ‖Σu_i‖²)/(N(N−1))`，**恰为全部 `N(N−1)/2` 对的均方距离**，
  `C` 即成对距离的 RMS。数值复核（vs 暴力枚举）：

  | N | D | brute RMS | 闭式 | 相对误差 |
  |---:|---:|---|---|---|
  | 5 | 7 | 6.444604774796 | 6.444604774796 | 0 |
  | 37 | 129 | 28.771785411032 | 28.771785411032 | 1.2e-16 |
  | 200 | 64 | 21.150010035765 | 21.150010035765 | 0 |

  另验证了 docstring 声称的「中心化在差分里抵消」（最大差 1.8e-15），所以
  `make_zgt_center.py` 用同一次遍历的 `sum_u` / `sum_sq` 求 `C` 是精确的，无抽样、无二次开销。
- **拒绝 `C ≤ 0` 的三道闸**：`srht.pairwise_rms` 在 `n<2` 或均方 ≤0 时抛错；
  `WhatBatchBuilder.__init__` 在 `d_func_scale` 缺失/非正时抛错；`WhatTrainer.__init__` 从
  builder 读出后再查一次并抛错；`loss_style_dist` 自己还有一道。
- **`d_func_scale` 必填无默认**：`compute_batch(..., *, d_func_scale: float, ...)` 是
  keyword-only **且无默认值**，任何调用点想省略都会 `TypeError`；trainer 从 builder 读而不是
  另收一个参数，使 loss 与 batch 不可能对 `C` 有分歧。构造正确。
- 反例测试 `test_d_func_is_the_raw_u_distance_not_the_normalised_code` 用「同一 look 的
  0.2 与 1.0 强度」造出 `u₂ = 5u₁`，断言原始距离看得见（>0.5‖u₂‖）而 `z_gt` 距离 <1e-4。
  这正是 B-3 描述的失效模式，判别力充分。

### B-4 · §12.3 Spearman 用原始函数距离 —— **pass（并经我实测确认有双向动态范围）**

`style_diagnostics` 现在给两个数：`z_dist_spearman_vs_func`（对 **原始** `‖u_i−u_j‖`，
协议 §12.3 要的那个）与 `z_dist_spearman_vs_zgt`（方向-only，并列保留、明确命名、
docstring 写明「不是那个诊断」）。

实现者的反例测试只断言 `vs_func < 0.9`，看起来很松，所以我自己量了它的**判别力**：

| 探针 | `vs_zgt` | `vs_func` |
|---|---:|---:|
| 幅度盲码（测试的反例，20 个种子） | 1.000 | **0.110 – 0.329**（均值 0.205） |
| 幅度写在**范数**里的码 | 1.000 | 0.189 |
| 幅度写在**方向**里的球面等距嵌入（K=2×/10×/100×） | 0.189 | **0.996 / 0.9999 / 1.0000** |

即：真正把函数幅度编进**方向**（也就是 `L_style_dist` 唯一能奖励的那种编码）的码，
`vs_func` 可达 1.00；幅度盲码只有 0.2。**诊断的动态范围是满的，不是一个恒低的数**。
`< 0.9` 的阈值虽宽，但实测落在 0.33 以下，留了三倍余量，不会误判。

顺带确认了一件容易被误读的事：`d_style` 两处（loss 与诊断）都用 **L2 归一化后**的
`z_style`，所以「把幅度放进范数」是无效策略——码必须把幅度编进方向。这与
`L_style_cos` 也只管方向是自洽的，且 A-2 的 `C`（使 `d_func` 的 RMS 为 1）正好把目标压进
`d_style ∈ [0,2]` 的可达区间。见 N-19 的一条配套建议。

### B-5 / amendment A-3 · 12 臂统一 natural 采样 —— **pass**

- `_frozen_where(...)` **无条件**运行冻结 Where（`where_runner is None` 直接 `RuntimeError`，
  明写「没有 fallback」），`_model_signals(...)` 才按 `where_source` 决定模型输入侧
  （`none` → 空 `WhereSignals`；`oracle` → GT mask + oracle latent；否则 → frozen）。
  **供给侧与输入侧被彻底分开**，这是 A-3 的正确结构。
- `query_points` 三分支：global → `global_uniform`；local 且有 `m_hi` → `frozen_m_pred`；
  **local 且无 `m_hi` → `RuntimeError`**（不再静默退回全图）。`natural_weighting` 逐样本写进
  target 与 per-sample 记录，可审计。
- C03/C04 的查询点用的是 **frozen `m_pred`**（`frozen.m_hi`），不是 GT mask——与 A-3 一致，
  ceiling 臂的 oracle 只进模型输入。这点容易做错，实现做对了。
- **C01/C02 的短前向不引入 `<color>` 泄漏**：`sup_items` 构造为
  `prompt_ids + gt_where_ids`、`color_ids=[]`，故 `h_color` 切片为空、`h_where` 干净；
  模型侧的 `encoded` 则是 `prompt + color`（无 `<where>`）。两条序列各取所需，互不污染。
  索引对齐我核过：`sup_items` 与 `items` 同循环同序，且 `not where_prefix` 时每个样本都建，
  故 `sup_encoded[i]` 与 `samples[i]` 对应；`where_prefix` 时 `sup_encoded is encoded`。
  另外 `_frozen_where` 复用 `encoded` 的 `f_pre`（而非 `sup_encoded` 的），这是**更强**的
  选择：保证 Where 模型与 What 模型看到逐位相同的 `F_pre`。
- 回归测试含「六个臂（T01/T04/C01/C02/C03/C04）产生逐位相同的查询点」，直接钉死 A-3 的语义。

### B-6 · Where checkpoint provenance —— **pass**

新增 `q3vl/what/provenance.py`，三条硬停路径齐备且顺序正确（在任何昂贵操作之前）：

1. 任一既有 `run_setup.json` **不可读/损坏** → `WhereProvenanceError`
   （「崩在写 provenance 之前的那次 run，正是没人能担保其 checkpoint 的那次」）；
2. 已有臂**声明过** digest 而本臂拿不出 → 停（「丢了 provenance 不比 provenance 冲突轻」）；
3. digest **不一致** → 停，并列出每个冲突臂的 digest 与路径。

`run_what.py` 对**所有 12 臂**强制 `--where-checkpoint`（A-3 之后 C01/C02 也被冻结 mask 条件化，
不再豁免），先 `file_sha256` 再 `assert_where_consistency`，结果写进 `run_setup.json.where`
（含 `used_for: "model input + supervision mask"` / `"supervision mask only (amendment A-3)"`）。
`collect_where_digests` 用 `exclude_arm` 排除自己的上一次 run，避免自我冲突。7 条回归测试
覆盖全部路径，含「第一个臂无可冲突对象」与「no-where 控制臂不豁免」。

## 八、amendment A-1/A-2/A-3 文本审定

| 条 | 与我 §一末建议的一致性 | 是否越界 |
|---|---|---|
| **A-1** | **忠实**。§7.6 替换段、「为什么不是 (I+ΔG)x」、`G' = G + I` 重参数化证明、四条不变量（容量/参数量/§12 指标/§15 问题）、实现指针（`model_rdg::render` + `ci_checks_rdg`）逐条都在。**新增**了一张两方独立复算对照表（实现者 vs 审阅人），是加强不是改动 | **否** |
| **A-2** | **忠实且更完整**。我建议的是「`d_func = ‖u_i−u_j‖/c`，`c` 为 train 集 RMS，与 `mean_train_u` 一起发布」——A-2 就是这个，并补上了闭式与「禁逐 batch 归一」的红线措辞。它顺带**定义**了协议原本未定义的 `d_style`（L2 归一化后的欧氏距离）——补全未定义符号属必要，不属越界 | **否** |
| **A-3** | **忠实且取了更强的一支**。我给了 (a) 统一冻结 `m_pred` 与 (b) 登记 `natural_weighting` 字段两个选项，A-3 **两个都取**：既统一采样，又要求逐样本记字段。并正确推导出「C03/C04 的 oracle 只进输入、loss 分布与主臂相同」 | **否** |

三条都只改了它们各自要改的东西：**没有动任何 loss 权重、没有动 §8 的 12 臂、没有动 §12 的
指标定义或 gate 阈值、没有动 §10.4 的任何超参**。我逐条比对过 §7.4/§7.5 的 1116/684、
§9.5 的七个权重、§12.1 的 1e-4/5e-4、§12.4 的字典序五键——全部未变。

**一条编辑体例上的 nit（N-17）**：§7.6 正文仍原样保留 `(I + DeltaG) x` 的公式块，只在其**后**
加了「修订说明」脚注。脚注醒目、紧邻、指向明确，可以接受；但任何人 grep
`T_pred(x) = clamp(` 首先命中的仍是被取代的式子。建议在代码块**内部**加一行
`# SUPERSEDED by amendment A-1 (2026-08-05)`，使公式本身自带标记。

## 九、N-16 与 N-13

- **N-16 · pass**。`--out` 默认值从交付目录改为 `RUN_ROOT / "preflight"`（scratch）；
  新增 `write_report(..., force=False)`：当**已存在**报告 `complete: true` 而新报告
  `complete: false` 时**拒绝写入**并给出解释，`--force` 才允许且先备份为
  `preflight_what.json.superseded`。两条回归测试（拒绝 + 默认值不是交付目录）。
  这条修得比我建议的更彻底——我只提了改默认值，实现把「静默降级」也堵上了。
- **N-13 · 确认当前状态无一致性风险**（维持不上提，记可选债务）：
  `q3vl/what/config.py` 由 `from q3vl.whereb.contracts import ...` 引入并改名
  `COLOR_HIDDEN_*`，测试以 **`is` 身份**（不是 `==`）断言其与 `SEGMENT_HIDDEN_*` 为同一对象，
  `WhatVLM.__init__` 的默认值同样以身份断言。两份扫描测试（whereb-root、what-root）合起来
  覆盖两个包的全部非测试 `.py`，且 what 侧的扫描名单额外包含 `COLOR_HIDDEN_*` 两个别名。
  what → whereb 的依赖方向本就大量存在（`whereb.heads` / `stores` / `config` / `model`），
  不引入新耦合。**结论：两阶段现状下漂移风险为零**；债务只在「未来出现第三个阶段」时兑现。

## 十、新增发现

### NF-1（**BLOCKER**）· Stage-What 全程用 GT `<where>` / `<color>` 文本，且此决策从未声明

**这是我初审漏掉的**（不是本次修复引入的回归），是在为 B-5 重读 batch 构造时发现的。

`WhatBatchBuilder.build` 的 `where_ids` / `color_ids` 全部来自
`collator.encode_one(s)`，即 record 里的 **GT** `where` / `color` 文本。Stage-What 的
**训练与评测**因此 100% teacher-forced：`Q_color` 读的是 GT `<color>` 推理，
冻结 Where 读的是 GT `<where>` 推理。包内没有任何生成路径（grep `generated` / `gencontext`
零命中）。

而 Where-B 对同一问题有**完全相反**的纪律：`q3vl/whereb/gencontext.py` +
`scripts/make_generated_context.py` 产出并发布 `.genwhere.json`，`whereb/trainer.py` 按
`["gt", "nd"] * (mb//2)` 做 50/50，`run_where_b.py` 断言逐样本覆盖并明写
**"there is no GT fallback"**——正是 §5.4 要求的。

为什么这是 blocker 而不是 nit：

1. **§0 的核心主张失去证据**。「最终网络只接收 `I_in + instruction`」在 Stage-What 的任何一个
   交付数字里都没有被检验过；§15 问题 3/4/5 的答案全部条件在 GT 推理文本上。
2. **A-3 恰好抬高了赌注**。冻结 Where 的 `m_pred` 现在是全部 12 臂的**监督**掩膜，而那个
   `m_pred` 是由 GT `<where>` 文本算出来的——包括按构造永远看不到 `<where>` 的 C01/C02。
   （A-3 本身没错：它比较的是**输入**通道，这正是 §8.2 要的；但整块板子是 teacher-forced。）
3. **train/test 失配无法事后补救**。若 12 臂在 100% teacher 上训完，再改用 generated 上下文
   评测，掉分是必然的，正确动作只能是**重训 12 臂**。Where-B 用 50/50 训练就是为了避免这件事。
4. **它是一条未声明的决策**。NOTES §三「待主 agent 决策」里没有它（grep `context` / `teacher` /
   `generated` 在 NOTES 与 PENDING 中零命中），违反「属于决策的写入待决策节，不许静默拍板」。

**需要主 agent 裁定**，可选项：
- (a) 与 Where-B 对齐：训练 50/50 teacher/generated，选择以 generated 为主，四种上下文
  （GT / generated / null / shuffled）分开报告（§5.4 的形状）。代价是要把
  `make_generated_context.py` 扩到同时产出 `<color>` 段（Base SFT 本来就一次生成
  `<where>...</where><color>...</color>`，多半只是把已生成的字符串多留一段）。
- (b) 明确裁定 Stage-What 用 teacher context，并**在协议里写下来**（新 amendment A-4）+
  在 REPORT 与 §15 的结论措辞里声明「Stage-What 的全部数字条件在 GT 推理文本上」，
  同时把 generated-context 复评列为 `T_final` 之后的独立一次性附加实验。

我倾向 (a)：(b) 会让 §15 的问题 3/4/5 都带一个无法在本轮消除的限定词，而这正是
「不能用最终平均图像指标掩盖…」那条纪律要防的。但两种做法都合理、都影响后续，
所以是**决策**不是缺陷，交主 agent。

### 新增 NIT

- **N-17**：§7.6 正文的公式块本身未带 SUPERSEDED 标记（见 §八）。
- **N-18**：`WhatTrainer.best` 的 `pool = gated or scored` 在**全部 checkpoint 都未过 gate**
  时回退到全集，于是返回一个 gate 未过的行；这对 `_protect_best`（保住文件供 §15 诊断）
  是对的，但 `best()` 同时也是「哪个 checkpoint 赢了」的公开访问器，而 `main_board` 在同样
  情形下是 `selection_possible: false`。两者语义不同却同名同形。建议 `best()` 额外返回
  `gate_fallback: True`，或 docstring 明写「回退时返回的不是选择」。
- **N-19**：`c.get("gate_pass", True)` 在 eval 报告缺该键时默认**通过**。生产路径的
  `arm_metrics` 永远会写这个键，所以目前是理论洞；但这正是 Where-B 审阅 blocker B1
  （「没跑的检查不得报 PASS」）的同型。建议缺键即计入一个 `n_gate_unknown` 并在 setup 里报。
- **N-20**：`make_zgt_center.py` 发布了 `C`，但没发布 `d_func` 的分布尾部。由于
  `d_style = ‖ẑ_i − ẑ_j‖ ≤ 2` 而 `d_func` 无上界，`d_func > 2` 的那部分对是**结构上不可达**的
  （Huber 只是把它们的梯度压成线性）。建议同一次遍历里抽样估一个 `d_func` 的 p99/max 并落盘，
  让结果审阅知道有多少比例的对落在可达区间之外。
- **N-21**：`config/arm_matrix.json`、`config/mock_e2e.json`、`preflight_what.json` 的环境戳
  已按 N-1 改成战役环境（3.12.12 / torch 2.10.0+cu128）✓，但 `git_commit` 仍是 `fef9f93`
  （生成于 `cf17089` 提交之前）。建议提交后重新生成，或改记 `git describe --dirty`。

## 十一、最终判决（取代 §六）

**修复复审：B-1 … B-6 全部 pass，A-1/A-2/A-3 文本忠实且未越界，N-16 pass，
N-13 现状确认无一致性风险。原 6 个 BLOCKER 清零。**

**新增 BLOCKER 1 项：NF-1（Stage-What 全程 teacher-forced 推理文本，决策未声明）。
新增 NIT 5 项：N-17 … N-21。**

**是否准许进入正式训练（Where 定档后）：**

- **GPU preflight `WT-G1`–`WT-G8`：准许**（两卡释放后即可）。NF-1 不改变这八项测量的对象。
  建议在 `WT-G2` 里顺带把「GT vs generated 两条序列的 `H_color`」一起测出来——那正是
  NF-1 选项 (a) 需要的第一个数字。
- **正式训练（12 臂任何一臂）：仍不准许**，唯一未清项是 **NF-1**，且它需要的是**主 agent
  裁定**而不是更多实现工作。理由与 B-3/B-5 同档：它决定 12 臂**训练时**看到什么上下文，
  跑完再改只能重训全部 12 臂。
- 一旦 NF-1 裁定落地（选 (a) 则需扩 `make_generated_context.py` 并加 50/50 采样；
  选 (b) 则需写 amendment A-4 + REPORT 措辞约束），**即可开跑**。N-17 … N-21 全部为
  非阻塞，可在训练期间并行处理。

> **本节判决已被 2026-08-05 的第三轮聚焦审阅取代，见下方 §十二起。**

对实现者这一轮的评价：六个 blocker 全部是**结构性**修复而非打补丁——B-1 把标签变成过滤器、
B-2 把两个保护标志分离并覆盖两条时序、B-3 用闭式常量取代了抽样、B-5 把「供给侧」与
「输入侧」拆成两个方法、B-6 新建了一个只负责 provenance 的模块。回归测试尽量复用了我给的
构造性反例（`test_b1_a_gate_failing_arm_cannot_top_the_board` 的 docstring 直接写
「the reviewer's counter-example, verbatim」），这使「修复前会失败」是可验证的而不是自述的。
N-16 修得比我提的更彻底。

---

# 第三轮聚焦审阅（2026-08-05，commits `ba9963e` / `2eb706c` / `e002262` / `5977dc5`）

- 范围：只审 A-4 相关 diff（含与 WB-IMPL `dc6944c` 的接口对齐）。
- 方式：只读；未用 GPU（Base SFT 收尾中，两卡未受干扰）。
- 复跑（战役环境 `/home/bc/envs/q3vl_sft/bin/python`）：
  - `pytest q3vl/what/tests -q` → **221 passed, 8 skipped**（8 skip = 5 个需
    `PublishedStore`/shardio 的 store 契约测试 + 3 个缺 `colour`/`skimage`/`dataset_build`
    的跨实现对拍），与上报一致；
  - `python -m q3vl.what.preflight --limit 400`（写到 scratch，**未碰交付目录**）→
    **12 pass / 0 fail / 0 skip、`complete: true`**，与交付的 `preflight_what.json` 逐项一致。

## 十二、逐项复审

### 12.1 A-4 amendment 文本 —— **pass**

**四点齐全**：① 训练 50/50（明确锁在 **micro**-batch 上，因而对任意梯度累积倍数的 effective
batch 都成立；缺闭合标签不回退 GT，与 §5.4 逐字同构；缓存 **token ids** 而非 hidden，使
「同层、同位置、同归一化」成为构造性事实）；② 评测分报，每 checkpoint 每 context 一行，
永不混成均值；③ generated 主榜，teacher 榜并列，两者之差是「对 GT 推理文本的依赖程度」；
④ 控制臂 forced prefix，且明写「生成模式由 `where_prefix` 推导，非硬编码」。
另含上游依赖（`genwhere/2`）与 token 边界（384，附 3745 条抽样的 min/p50/p95/p99/max）。

**「恢复而非改变可回答性」的论证成立**。§0 写的是「最终网络只接收 `I_in + instruction`」，
§15 的问题 3/4/5 也都以此为前提——它们**本来就要求** generated 语境下的数字。100% teacher-forced
不是协议的一个选项，而是协议未声明处被实现默认掉的一个缺省。因此 A-4 是把本就该有的口径补上，
不是引入新口径。这个论证我认为是准确的，不是修辞。

**不变量声明准确**——我直接 import config 逐值核对，全部与 A-4 声称的一致：

| 声称不变 | 实测 |
|---|---|
| §9.5 七个权重 | `1.0 / 10.0 / 0.001 / 0.05 / 0.05 / 0.02 / 0.1` ✓ |
| §12.1 bake gate | `1e-4 / 5e-4 / 0` ✓ |
| §12.4 字典序五键 | `local_image_de00_median, lut_de00_p90, boundary_de00_median, n_trainable_params, latency_ms` ✓ |
| §8 十二臂 | `len(ARMS) == 12` ✓ |
| §7.4/§7.5 每样本参数量 | `1116 / 684` ✓ |

一处措辞需要读者注意（不构成 blocker）：A-4 说「§12.4 的字典序五键不变」——**字面成立**，
但选择的**候选集合**确实变了（只有 generated-context 行进榜）。这一点 A-4 第 3 点已明说，
两处合起来没有误导。

顺带确认 N-17 已修：§7.6 的公式块**内部**现在带
`# SUPERSEDED by amendment A-1 (2026-08-05)` 三行行内标记，grep 到公式即看到标记。

### 12.2 训练 50/50 与「无 GT 回退」的结构性证明 —— **pass**

- **复用 whereb 的 `BalancedContextSampler` 正确**：`q3vl/what/context.py` 直接 import
  Where-B 的 sampler 与 `FormatStats`，不是重实现。`iter_modes` 在 yield 前**断言**
  Where-B 的 `GT`/`GENERATED` 字符串与本包的 `CONTEXT_GT`/`CONTEXT_GENERATED` 相同，
  任一侧改名即 `AssertionError` 而不是「一批 teacher context 被标成 generated」。
  奇数 micro-batch 在 sampler 构造时就抛 `ValueError`（`micro_batch % 2`），
  WT-P7 对 mb ∈ {2,4,8} 实测每批恰好一半 teacher，对 mb=3 实测被拒。
- **`generated_color_context` 签名无 GT 入参**：实测参数集为
  `{sample_id, generated_ids, close_id, text, max_tokens, eos_id, genctx_mode}`。
  WT-P7 断言 `color_text` / `sample` / `record` / `tokenizer` 均不在其中——**审的是签名，
  不是读实现体**，符合任务卡要求。`data.py::color_context` 的 generated 分支也确实
  只传 `rec["color_ids"]`，`sample.color_text` 在该分支不可达。
- 无闭合标签 → 截断 + `format_failure=True` + `stop_reason="no_close_tag"`；
  闭合但超界 → `closed_over_boundary`；空 → `empty`。四条都有测试。
- teacher 侧超界**报错而非截断**（`gt_color_context` 抛 `ValueError`，并说明「边界是对语料的
  断言，不是截断路径」），全语料校验列为 `WT-J9`。这个方向是对的：截断 teacher context
  会静默改变监督。

### 12.3 评测分报与 generated 主榜 —— **库层 pass，接线 BLOCKER（见 NF-2）**

库层实现正确：
- `arm_metrics(..., context=SELECTION_CONTEXT)` 把 context 写进行里，未知 context 直接
  `ValueError`；
- `main_board(..., context=...)` 先**拒收无 `context` 标签的行**（明写「A-4 之前的行不得被
  静默当作 generated」），再按 context 过滤，然后才是第二轮已通过的 gate 过滤 + 字典序；
- `ceiling_board` 也按 context 过滤，C03/C04 不会跨 context 混进来；
- `context_report` 按 `(arm, step)` 配对两种 context，在**主选择键**上给 `gap = generated − teacher`，
  并统计 `n_pairs`。
- 关键设计正确：**disjoint 只发生在训练 sampler，评测侧没有任何 disjoint 划分**，
  所以「同一批 `V_what` 样本各跑两遍」在库层是可实现的，`gap` 是配对量。

### 12.4 C01/C02 的 forced prefix 由 `where_prefix` 推导 —— **pass**

`config.genctx_mode_of(arm)` 的实现是

```python
return (GENCTX_MODE_FORCED_COLOR if not WC_INTERFACES[ARMS[arm][0]]["where_prefix"]
        else GENCTX_MODE_WITH_WHERE)
```

即**从 WC 接口表的 `where_prefix` 推导**，不是 `{"C01","C02"}` 硬编码。WT-P7 反过来验证
推导结果恰为 `{C01, C02}`（实测 `genctx_mode_by_arm` 里 T01–T08/C03/C04 全是 `two_segment`），
且有一条测试专门盯「模式跟随 `where_prefix` 而非臂名」。这样将来新增一个 no-where 臂会自动
落到 forced 档，不需要有人记得改列表。消费侧还逐样本断言 `rec["mode"] == cfg.genctx_mode`
（`data.py` 与 `ColorGenContextStore.record` 两道），错档记录无法被静默使用。

### 12.5 接口对齐（`e002262`）与 WB-IMPL `dc6944c` 的逐字段核对 —— **pass**

我把消费侧与 producer 侧并排核对：

| 共享量 | producer（`q3vl/whereb/config.py`） | consumer（`q3vl/what/config.py`） |
|---|---|---|
| mode 词表 | `GENCTX_MODES = ("two_segment", "forced_color")` | **import**，并在 import 处断言集合相等 |
| schema id | `SCHEMA_GENCTX = "q3vl.where_b.genwhere/2"` | **import** as `SCHEMA_COLOR_GENCTX` |
| `<color>` 边界 | `COLOR_CONTEXT_MAX_TOKENS = 384` | **import** |
| 生成预算 | `GEN_MAX_NEW_TOKENS = 512` | **import** as `GEN_COLOR_MAX_NEW_TOKENS` |

四个共享量**全部 import、无一重声明**，运行时实测
`('two_segment', 'forced_color') / 'q3vl.where_b.genwhere/2' / 384`。可读标识符
`GENCTX_MODE_WITH_WHERE` / `_FORCED_COLOR` 保留但**取 producer 的值**——命名归消费侧、
取值归 producer，这个分工是对的。

字段逐一核对（producer `q3vl/whereb/gencontext.py` 的 payload vs consumer 的
`REQUIRED_FIELDS` + `summary()`）：`sample_id` / `schema_version` / `mode` / `color_ids` /
`color_text` / `color_stop_reason` / `color_format_failure` / `color_truncated` —— **八个全部对得上**。

自曝的分歧值得记一笔：消费侧原本自拟 `("with_where_prefix", "forced_color_prefix")`，
与 producer 实际的 `("two_segment", "forced_color")` 不同，**会让每一条真实记录都被
`ColorGenContextStore.record` 拒收**。这是「先写消费侧契约、再对齐 producer」这条路线本该
暴露出来的东西，而它确实被暴露了——因为契约是硬断言而不是 `.get()` 默认值。

`ColorGenContextStore` 的四条契约（缺字段 / schema 非 v2 / mode 不匹配 / 覆盖不全）
全部是硬停，`assert_covers` 在**任何昂贵操作之前**对整个 split 跑一次。这与 Where-B
`run_where_b.py` 的 `assert_genctx_coverage` 同构。

### 12.6 自曝的 NOTES 静默未落盘（`5977dc5`）—— **pass**

我直接 grep 交付文件确认三节**现已真实存在**：`### D-W9`（第 141 行）、`### D-W10`（163）、
`### D-W11`（177），内容与 A-2/A-3/A-4 一一对应，D-W11 还含与 WB-IMPL 的接口对齐记录。

事故本身（用带全角括号的锚点 `### D-W8（未在协议中固定…）` 去 `str.replace`，
实际标题无括号，替换失败原样返回而脚本仍打印 "ok"）是**实现者自查发现并主动上报**的，
修法是把锚点改成硬断言（锚点不存在即 `AssertionError`），并逐条 grep 复核本轮所有文档改动
确实落盘。这与我自己在 §三-bis 记的那次是同一类失败（「第二种失败模式是静默的」），
处理方式也一致：披露 + 复原 + 加断言。**该处理正确，不留 blocker。**

### 12.7 R6 guard（sqlite3-before-torch）—— **pass**

R6 我独立复现，是真的：

```
import torch; import sqlite3   -> ImportError
import sqlite3; import torch   -> OK
```

三个入口脚本（`run_what.py` / `pack_gt_luts.py` / `make_zgt_center.py`）都在
`from __future__` 之后、**任何其它 import 之前**放了 `import sqlite3  # noqa: F401`，
并附了解释性注释。AST 单测取每个模块的**首次** import 行号并断言
`lines["sqlite3"] < lines["torch"]`，同时断言 guard 存在（「这条测试是用来阻止有人把它整理掉的」）——
这正是这类 guard 最需要的那种测试。

我另行确认了一件容易误判的事：`q3vl/what/preflight.py` **没有** guard，但它**不需要**——
其数据检查走 `q3vl.train.shards.ShardStore`，不在 `q3vl.data.shardio → sqlite3` 那条链上；
我用 `python -m q3vl.what.preflight --limit 400`（torch 先于一切被 import）实跑，
`WT-W1`/`WT-W2` 均 **pass**。真正踩链的是 `PublishedStore`（`ColorGenContextStore` 的基类），
而它只在 `run_what.py` 里被构造，那里有 guard。**覆盖面正确，无遗漏。**

实现者同时上报「`q3vl/whereb/stores.py` 在同一条链上，`run_where_b.py` 有同样暴露，
越出边界故未改，已写入 PENDING R6 待主 agent 决策」——**这个处理是对的**：发现越界问题
应上报而不是顺手改别人的包。请主 agent 把它转给 WB-IMPL。

### 12.8 互斥样本池的读数警告 —— **pass**

NOTES 第 260 行起写得准确且完整：

> **这两列不可跨 context 比大小**：`BalancedContextSampler` 把数据集**一次性**切成互斥的
> teacher / generated 两池（这正是「1 epoch = 每样本只被看见一种 context」的实现方式），
> 所以两列跑的是**不同的样本子集**…… 可比的是**各自的趋势**——两列都单调下降。
> 真实的 "generated 比 teacher 差多少" 要由 §12 的评测双榜（同一批 `V_what` 样本各跑两遍）
> 给出，`evaluate.context_report` 的 `gap` 就是那个数，不是这里。

三件事都说到了：为什么不可比、什么可比、真正的数在哪里。mock 表里
`L_func` 已按 teacher/generated 分列，且记录了「全部步 `n_gt == n_generated`、
`teacher_fraction` 取值集合 = {0.5}」。

**评测侧确实正确承接**（库层）：`BalancedContextSampler` 只被 trainer 使用，
`arm_metrics` / `main_board` / `context_report` 都不做任何 disjoint 划分，
`context_report` 按 `(arm, step)` 配对——所以 `gap` 在设计上是配对量，不重复训练侧的混淆。
唯一的问题是这条路径目前没有任何生产入口在调用（NF-2）。

## 十三、新增发现

### NF-2（**BLOCKER**）· 生产路径没有接线评测：`eval_fn` 未挂，无评测入口脚本

`q3vl/what/scripts/` 只有三个脚本（`run_what.py` / `pack_gt_luts.py` / `make_zgt_center.py`）。
`run_what.py` 构造 trainer 时**不传 `eval_fn`**：

```python
trainer = WhatTrainer(model, builder, dataset, cfg, tcfg, run_dir=run_dir,
                      device=args.device)          # eval_fn 缺省为 None
```

全仓库 grep `arm_metrics|main_board|context_report`，**除 `evaluate.py` 自身与测试外零命中**。
三个后果，第二个最严重：

1. **§10.4 的 `eval_steps: 500` 在生产路径上未实现**——不会有任何 in-loop 评测。
2. **第二轮通过的 B-2 修复在生产路径上是失效的**。`eval_fn is None` ⇒
   `state.checkpoints` 恒空 ⇒ `best()` 返回 `None` ⇒ `_protect_best()` 清空所有标志后直接
   return ⇒ 没有任何 checkpoint 被 `best_protected` ⇒ `_roll()` 按 `keep_last=3` 照常删。
   约 4975 步、`save_steps=500` 下有约 9 次普通保存，**前 6 次仍会被删掉**——正是 B-2 被提出的
   那个失效模式。B-2 的 7 条回归测试全部注入了 `_EvalStub`，所以它们全绿而生产无保护。
   **这一条我在第二轮漏了**：我验证了机制，没有验证接线。与 NF-1 同类的疏漏，记在我账上。
3. **A-4 的第 2、3 点没有生产者**。`arm_metrics(context=...)` / `main_board(context=...)` /
   `context_report` 实现正确、测试充分，但没有任何代码在真实数据上调用它们；
   NOTES 指向的「`context_report` 的 `gap` 就是那个数」目前还没有产出它的路径。

**这不在任何 pending 清单里**。`WT-J4`/`J5`/`J6`/`J7` 都是对评测的**增补**，
全部预设「评测 harness 已存在」；没有一条说「evaluate 尚无生产入口」。

**修法（两条都可，需主 agent 选一条并落盘）**：
- (a) **接线 in-loop 评测**：给 `run_what.py` 加一个 `eval_fn`，在 `V_what`（或其固定子集）上
  按两种 context 各跑一遍，产出两行 `arm_metrics`，写 `eval.jsonl`。这同时激活 B-2 的保护、
  落实 §10.4 的 `eval_steps`、并让 A-4 的 `gap` 每 500 步就有读数。代价是每 500 步一次评测的
  时间开销（可用固定子集控制）。
- (b) **离线评测 + 关闭滚动删除**：`run_what.py` 传 `keep_last=None`（约 9 × 0.37 GB ≈ 3.4 GB/臂、
  12 臂约 41 GB，NFS 可承受），训练只保存不评测；另写一个 `evaluate_what.py`，训练后对全部
  存活 checkpoint × 两种 context 出 `arm_metrics`，再 `main_board` + `context_report`。
  这条更省训练时间，但 §10.4 的 `eval_steps: 500` 需要在 REPORT 里显式声明为「离线等价执行」。

**无论选哪条，都必须在第一个臂起跑前落地**：选 (a) 需要改 runner；选 (b) 至少需要**现在**
就把 `keep_last=None` 写进 `TrainConfig` 的生产取值，否则前 6 个 checkpoint 在第一个臂跑完时
就已经不存在了，事后无法补救。

### 新增 NIT

- **N-22**：`generated_color_context` 的 `text` 形参是一个**无约束的自由字符串**。
  WT-P7 的禁用名单是 `color_text/sample/record/tokenizer`，不含 `text`。它只流向
  `ColorContext.text`（元数据），不进 `token_ids`，所以**条件化路径**的无-GT-回退证明成立；
  但一个写错的调用点可以把 GT 文本塞进标着 `mode="generated"` 的记录里，污染 per-sample 日志
  与 `color_text` 溯源。建议把 `text` 也纳入 WT-P7 的说明（明确「结构性证明覆盖的是 token 路径」），
  或让 `data.py` 只从 store 记录取该字段（现已如此，但没有断言钉住）。
- **N-23**：`ColorGenContextStore.assert_covers` 只查**存在性**，不查
  `mode`/`schema_version`（那两条在 `record()` 里逐样本查）。于是「整个 split 都是错档」
  这一情形要到第一个 batch 才暴露，而不是在启动前的覆盖检查里。建议 `assert_covers` 顺带抽
  1 条 `record()`，把三条契约都提前到启动前。
- **N-24**：`WT-J9`（`<color>` 边界全语料校验）目前排在正式开跑**之后**的作业里，
  但 `gt_color_context` 超界是**抛错**，意味着一条超界样本会让某个臂在训练中途崩。
  边界校验是纯读 record 的轻量扫描，建议提到开跑前与 `WT-J3` 合并做一次。

## 十四、最终判决（取代 §六 与 §十一）

**A-4 相关全部审阅项 pass**：amendment 文本（四点齐全、论证成立、不变量经实测核对）、
训练 50/50（sampler 复用正确、签名级无-GT-回退证明成立）、评测分报与 generated 主榜（库层）、
C01/C02 forced prefix 由 `where_prefix` 推导、接口对齐（四个共享量全 import、八个字段全对齐）、
NOTES 三节已真实落盘、R6 guard（三入口 + AST 单测，覆盖面经实测确认无遗漏）、
互斥样本池警告已写入且评测双榜设计正确承接。

**BLOCKER 累计：1 项（NF-2，本轮新增；其中第 2 点是我第二轮的漏审）。**
**NIT 新增 3 项（N-22 … N-24），历史 N-9/N-10/N-11 等仍在 PENDING 中跟踪。**

**是否准许正式训练（Where 定档后）：**

- **不准许**，唯一未清项是 **NF-2**。它需要一个**主 agent 的选择**（in-loop 评测 vs
  离线评测 + `keep_last=None`）加一处不大的实现改动。
- 之所以仍判为 blocker 而不是 nit：NF-2 的第 2 点会**在跑的过程中静默删除将被选中的
  checkpoint**，与 B-2 完全同型且事后不可补救；第 3 点则意味着 A-4 刚刚定档的 generated 主榜
  在跑完之后没有任何产出路径。两者都属于「跑完再修等于重跑」。
- 其余前置条件不变且已就绪：CPU preflight 12/12（`complete: true`，我独立复跑一致）、
  221 单测通过、协议 amendment A-1…A-4 齐备。仍需等待的外部依赖是
  **Where-B 定档一个冻结 checkpoint**、**`WT-J10`（WB-IMPL 的 `genwhere/2` 生成作业，两种 mode）**、
  以及两卡释放后的 `WT-G1`–`WT-G8`（其中 `WT-G2` 已按我上轮建议并入「GT vs generated 两条序列的
  `H_color`」测量）。
- **GPU preflight `WT-G1`–`WT-G8`：仍准许**先跑，NF-2 不影响其测量对象。

对本轮的评价：A-4 是一次范围明确、边界克制的落地——`context.py` 复用 Where-B 的 sampler 而不是
重实现，共享量全部 import 而不是重声明，越界发现（whereb 的 R6 暴露）上报而不顺手改。
两次自曝（接口词表分歧、NOTES 未落盘）都由实现者自己发现并主动上报，这比审阅者抓到更有价值。
唯一的系统性缺口是 NF-2：三轮下来，**被审的一直是库，没有人审过"谁来调用这个库"**——
这也是我连续两轮没抓到它的原因。
