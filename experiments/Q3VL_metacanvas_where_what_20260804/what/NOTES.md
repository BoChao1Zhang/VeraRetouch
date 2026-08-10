# Stage-What 实现说明（WHAT-IMPL）

- 任务卡：WHAT-IMPL（Waves T1-T4 / C1-C2 的实现前置）
- 权威文档：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md`
  §0 / §2.3 / §5 / §6 / §7 / §8 / §9 / §10.4 / §12 / §13 / §14
- 代码：`q3vl/what/`（新包，未改动 `q3vl/data`、`q3vl/train`、`q3vl/where`、`q3vl/whereb` 任何文件）
- 状态：**实现完成，CPU 级验证通过，未启动任何正式训练，全程未用 GPU**
- 环境：战役环境是 `/home/bc/envs/q3vl_sft/bin/python`（transformers 4.57.1 / torch 2.10.0+cu128）；
  base conda 环境是 transformers 4.36.0，**Where-B 现有测试在 base 下会因为 Qwen3-VL 类不存在而报错**（与本次实现无关，
  `git status` 证实未改动 `q3vl/whereb/` 任何文件）。本包 157/158 个测试在两个环境下都通过（q3vl_sft 下 3 个跨实现对拍因缺
  `colour`/`skimage`/`sqlite3` 而 skip）。

---

## 一、实施前核实记录（全部当场打开原始来源核实，未采信检索结论）

| 编号 | 事实 | 核实方式与结果 |
|---|---|---|
| V-W1 | **`T_gt` 的来源**：`sft2seg` 每条 record 自带 `preset_path`，指向真实 `.cube` / `.3dl` 文件 | 直接从 `/mnt/nfs/bc/data/datasets/sft2seg-20260804` 读 record。示例：`preset_path=/home/bc/data/datasets/recipes/quandian/quandian_003374.cube`。**不需要额外渲染，不需要从 `recipe.preset` 重建** |
| V-W2 | `lut_id -> preset_path` 是 1:1 | 跨 train / V_what / T_lut_unseen 抽样 481 个不同 `lut_id`：**0 冲突，481/481 文件存在**；扩展名分布 `.cube` 474 / `.3dl` 7（1.5%），两者 `dataset_build.lut_io.load_lut` 都支持 |
| V-W3 | LUT 语料规模 | `find /home/bc/data/datasets/recipes -name '*.cube'` = **7046 个文件 / 8.79 GiB 文本**；`LUT_3D_SIZE` 实测取值 **16 / 25 / 32 / 33 / 64 / 65**（非统一尺寸，代码必须按原生网格求值） |
| V-W4 | 各 split 的 `lut_id` 隔离 | 直接扫 5 个 `*.index.jsonl`：train 3149 个 LUT / 159,215 样本；V_where 530/896；V_what 531/897；T_final 577/918；**T_lut_unseen 259 个 LUT / 433 样本，与 train / V_where / V_what / T_final 的交集全部为 0**。§2.2 的 "unseen LUT generalization" 说法成立 |
| V-W5 | **`I_tar` 是用什么插值渲染的** | 读源码：`dataset_build/core/render_backend.py::_render_cube_gpu` 用 `F.grid_sample(mode="bilinear", padding_mode="border", align_corners=True)`，CPU oracle `dataset_build/src/construct/rendering.py::apply_lut_cpu_oracle` 是同一算术的展开式，都在 gamma sRGB 空间、按 `[DOMAIN_MIN, DOMAIN_MAX]` 归一化后 clamp 到 [0,1]。合成式是 `composite_srgb(before, edited, alpha)`，即 `I_out = I_in + m (T(I_in) - I_in)` |
| V-W6 | `.cube` 的轴序 | `dataset_build.lut_io.load_lut` 返回 `grid[b, g, r]`；`tools/cube/cubelib` 与 `model/glut_repro` 用 `table[r, g, b]`。本包在边界处**只转置一次**，并用真实 `.cube` 文件做了轴序回读测试（`test_cube_round_trip_axis_order`）——`render_backend` 2026-07-17 修过一次 `f(B,G,R)` 的轴序 bug，说明这不是假想风险 |
| V-W7 | 四面体插值实现 | 本包 `tetra_lookup` 与 **`model/glut_repro/model_rdg.py::tetra_lookup`**（该实现已由 `ci_checks_rdg` 对拍 colour-science）逐位一致（1e-6），并在 base 环境下**直接对拍了 `colour.algebra.table_interpolation_tetrahedral`**（1e-9） |
| V-W8 | 三线性实现 | 与 `dataset_build.src.construct.rendering.apply_lut_cpu_oracle`（生产 `I_tar` 的那份算术）对拍通过（1e-5） |
| V-W9 | CIELAB / CIEDE2000 | 与 `model/glut_repro/model_rdg.py` 的 `srgb_to_lab` / `delta_e00` 对拍（1e-4）；与 `skimage.color.rgb2lab` 差 ~0.015 Lab 单位，原因**不是精度**而是白点约定（skimage 用 ASTM `(0.95047,1,1.08883)`，本仓库沿用 `(0.3127,0.3290)` 色度导出值）。选择与本仓库既有实现一致 |
| V-W10 | hidden 口径 | `q3vl/whereb/contracts.py` 的 `SEGMENT_HIDDEN_LAYER=-1` / `SEGMENT_HIDDEN_FINAL_NORM=True`（裁定 D-B2）**由 import 引入，未重声明**；本包另加了与 Where-B 同型的"全包重声明扫描"测试 |
| V-W11 | GPU 占用 | `nvidia-smi` 实测两卡各 54 GiB / 87-97% 利用率（Base SFT 在跑）。**本次全程未提交任何 GPU 作业，未执行任何重 IO 作业**（仅在两个小 split 上做过 1330 条 record 的 dry-run） |
| V-W12 | 已有 npy33 缓存 | `/var/cache/veradata/dcube/npy33` 有 7083 个文件，但那是**重采样到 33³ 的规范化缓存**。用它当 `T_gt` 会把 32³/64³ 原生表的重采样误差注入监督目标，因此**不采用**，改为按原生网格求值 |

## 二、逐符号落实对照（§7 / §9）

| 协议条款 | 实现位置 | 备注 |
|---|---|---|
| §7.1 16 个 `Q_color` 只 cross-attend `H_color`，`M_color∈R^[16×512]` | `color.py::ColorStack` | forward 签名只有 `(h_color, h_color_mask)`；模块内**不存在**任何含 `where` 的标识符（AST 扫描断言） |
| §7.1 `z_style = LN(MLP(AttentionPool(M_color))) ∈ R^1024` | `color.py::StyleHead` | `Z_STYLE_DIM == ZGT_DIM == 1024` 有断言 |
| §7.1 `u(T)=flatten(T(x)−x)` on 固定 17³ grid | `srht.py::u_of_table` / `identity_grid` | 恒等 LUT 的 `u` 严格为 0 |
| §7.1 `z_gt = L2Norm(SRHT_1024(u−mean_train_u))` | `srht.py::SRHT` / `encode_z_gt` | `Φ = sqrt(n/k)·S·H·D`，FWHT 与显式 Hadamard 矩阵对拍；**跨实例逐位确定**；范数保持均值误差 <2%，成对距离比 ∈ [0.75,1.25] |
| §7.2 `a_i(p)=m_pred(p)·N(I_in(p);μ_i,Σ_i)` | `pooling.py::aligned_pool` | 全对数域（`softmax(log m + log N)`），与朴素式在良态区对拍 |
| §7.2 `v_i` 拼 `[RGB,Lab,V(F_pre)]` + `log(Σa)` + valid bit | 同上，`POOL_FEATURE_DIM=264` | Lab 用**无量纲**形式 |
| §7.2 无有效像素回退 masked global pool 不产 NaN | 同上 | 有专门测试：σ 压到下界 + 像素全在角落 → 触发回退，结果与 masked global pool 逐值相等且全有限 |
| §7.3 48 slots + WC tokens → 2 seed → geometry → pooling → 4 refine → 48 独立 head + 1 global head | `backend.py` + `generator.py` | 每 block 宽 512 / 8 heads / FFN 2048 / pre-norm；slots 自注意力、cross-attend 完整 `M_color`、接收 `v_i` 与 WC tokens、`z_style` ModLN、zero-init gated residual |
| §7.4 FG48 每样本 `48×23+12=1116` | `config.FG_TOTAL_PER_SAMPLE` + 测试断言 | 参数分组 μ3/Cholesky6/opacity1/existence1/M9/b3 |
| §7.4 梯度可穿过 pooling 回 geometry | `generator.py` 无 detach + 专门测试 | 见下方"零初始化门"的说明 |
| §7.5 SB48 跨样本共享 μ/Σ，4×4×3 anchors 初始化，每样本 14/primitive | `gaussians.py::GeometryBank` + `PRIM_LAYOUT_SB` | 测试断言同一 batch 内 SB 的 μ 完全相同、FG 的不同 |
| §7.5 FG/SB 总参数差 ≤2% | `generator.py::solve_sb_bottleneck` | 实测 **0.013%**（见下表） |
| §7.6 μ sigmoid 限 RGB cube / Σ Cholesky+对角 softplus SPD / opacity·existence sigmoid | `gaussians.py::decode_*` | 极端 raw 值下仍在闭 cube 内；σ 有正下界 |
| §7.6 `T_pred = clamp(...)` + 33³ 烘焙 + 四面体回读 | `gaussians.py::render`/`bake` + `lut.py::tetra_lookup` | 见下方 D-W1 |
| §9.1 2048 点（1024 固定分层 uniform + 1024 natural，local 按 frozen `m_pred` 加权） | `queries.py` | uniform 集固定且每个 8³ stratum 恰好 2 点；natural 对 `(seed, sample_id)` 确定 |
| §9.1 两半分别报告 | `losses.py::loss_func` | 每个训练 step 都写 `L_func_uniform` / `L_func_natural` |
| §9.2 `L_hc = mean[normalize(C_gt)(1-cos Δh)]`，Lab 归一化 | `losses.py::loss_hue_chroma` + `colorspace.py` | `normalize(C)` 用**全局常量 √2**，非逐图/逐 batch 最大值；hue 差用 `(a1a2+b1b2)/(C1C2)` 避开 atan2 分支切割 |
| §9.2 单独记录 `grad_norm(L_func)` 与 `grad_norm(10·L_hc)` 之比 | `losses.py::grad_norm_ratio` + trainer 定期写日志 | mock 实测比值 7.3–11.7（见第五节） |
| §9.3 `R_sparse` binary entropy / `L_style_cos` / `L_style_dist` / `L_var` / `L_cov` | `losses.py` | `L_style_dist` 两侧都用 L2 归一化后的欧氏距离，量纲一致 |
| §9.3 stop-gradient FIFO statistics queue，内容与更新规则对所有臂固定 | `losses.py::StyleQueue` | 队列固定 256、暖机 32、只 push `detach().cpu()`，无逐臂旋钮 |
| §9.4 `L_bake = Charbonnier(T_tetra(bake_33(T_pred),x) − T_pred(x))` 同批查询点 | `losses.py::bake_readback` / `loss_bake` | 全程可微 |
| §9.5 权重 1.00/10.00/0.001/0.05/0.05/0.02/0.10 | `config.LOSS_WEIGHTS` + 测试逐值断言 | |
| §9.5 `I_tar` 不进 `L_what` | `compute_loss` 签名无任何图像参数（测试断言） | `WhatDataset.load_target_image` 只在 evaluate 路径存在 |
| §10.4 三组 lr / WD 例外 / warmup / cosine / clip / bf16 / batch 32 / 1 epoch / 500 / 保护 0.5·1.0 | `trainer.py::param_groups` / `WhatTrainer` | geometry、bias、LayerNorm、**ModLN** 全部无 WD（测试逐参数断言） |
| §12 bake gate / LUT 指标 / 图像分区分层 / 负控制 / §12.4 字典序 | `metrics.py` / `evaluate.py` | `main_board` 拒绝非 `V_what` 的 split，并把 C03/C04 排除在主榜外 |

## 三、待主 agent 决策（已按保守默认继续，未静默拍板）

### D-W1（**最重要**）§7.6 的字面公式在零初始化时给出 `f(x)=2x`，与红线冲突

协议 §7.6 同时写了两句：

1. 「全局与局部 affine 都采用 identity-centered residual 参数化」
2. `T_pred(x) = clamp((I + ΔG) x + b_g + Σ_i q_i(x)(M_i x + b_i), 0, 1)`

因为 `q_i` 是归一化权重（`Σ_i q_i = 1`），若 `M_i = I + ΔM_i` 且 `ΔG=ΔM=b=0`，字面公式给出 `T(x) = x + x = 2x`——正是本战役红线点名的
「全局仿射 G 初始化 = **0** 不是 I（否则 f(x)=2x）」。

**已实测**（写进 preflight `WT-P-zero-init-identity` 的 detail）：字面读法下不 clamp 时 `mean(T(x)/x) = 2.000`，最大绝对误差 1.0；
clamp 后最大误差 0.5。

**保守默认**：`GLOBAL_AFFINE_MODE = "residual_zero"`，即
`T_pred(x) = clamp(ΔG x + b_g + Σ_i q_i(x)(M_i x + b_i), 0, 1)`，`M_i = I + ΔM_i`，`ΔG` 零初始化。
这同时满足红线、给出精确的零初始化恒等（实测最大误差 <1e-4），并且**逐项等于**
`model/glut_repro/model_rdg.py::render`（该实现已与 `BatchedGLUT` CI 对拍到 1e-5，即 GLUT Eq.1-3）。
`"identity_centered"` 分支保留，只为让 2x 可被**展示**而不是被断言。

**已裁定（2026-08-05）**：独立实现审阅（`docs/reviews/REVIEW-impl-What.md` §一）审定批准，主 agent 已把 amendment 写回协议
（`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §17 changelog 的 **A-1**，§7.6 处加了指向它的脚注）。
自此 `GLOBAL_AFFINE_MODE = "residual_zero"` 是**规格**，不再是「实现偏离」。审阅独立复算的数字与实现者的一致：
`Σq_i ∈ [0.9999975, 1.0]`、字面式 `mean T(x)/x = 1.99999988`、改写式 `max|T−x| = 2.4e-6`。

### D-W2 σ 参数化：协议说 softplus，红线说有界 sigmoid

§7.6 明写「协方差由 Cholesky 构造并**对角 softplus**」；红线速查写「σ 参数化禁裸 exp（**有界 sigmoid**）」。
softplus 不是裸 exp，所以两者不直接矛盾，但红线的括号给的是另一种做法。

**保守默认**：`SIGMA_PARAM = "softplus_floor"`，即 `diag = 0.02 + softplus(raw + c)`，`c` 使 `raw=0` 时 `diag = 0.20`。
下界 0.02 保证 SPD 与条件数（与 RD-G 的 `SIGMA_LO` 同值）。`"bounded_sigmoid"`（RD-G 已验证的 `0.02 + 0.48·sigmoid`）
是一个常量之隔的开关，两条路径都有域测试。**请主 agent 明确采用哪一支**——这会影响 σ 的可达上界（softplus 无上界，
sigmoid 上界 0.50），进而影响单个 Gaussian 能覆盖多大的颜色区域。

### D-W3 `T_gt` 用哪种插值定义

`.cube` 只是一张表，它代表哪个函数取决于插值器。`I_tar` 是用**三线性**渲染的（V-W5），而交付回读用**四面体**（§7.6/§12.1）。
若用四面体定义 `L_func` 的目标、却用三线性产生的 `I_tar` 做最终图像指标，会引入系统性偏置。

**保守默认**：`GT_LUT_INTERP = "trilinear"`（与 `I_tar` 的生成方式一致），四面体口径始终作为并排诊断报告（§12.1 本来就要求）。
一行常量即可切换。

### D-W4 `WC-0` 与严格 no-where 的唯一差别

§6 说 `WC-0` 是「H_color + 全局视觉池化；**无显式 Where 输出**」，§8.2 说 `C01/C02` 去掉「`<where>` prefix、所有 Where 输出
和 mask-conditioned pooling」。因此 `WC-0` 也**不能**吃 `m_pred`：`mask_pool` 只对 `WC-1`/`WC-3`/`ORACLE` 打开。

**行文修正（2026-08-05，因 REVIEW-impl-What B-5）**：初版这里写的「`T01/T05` 与 `C01/C02` 的唯一差别就是语言序列里还有没有
`<where>` 段」当时**只在模型输入侧成立**，loss 侧不成立——§9.1 的 natural 半区采样当时对 `where_source="none"` 的臂退回了全图
采样。裁定 D-W10（协议 amendment A-3）之后这句话才在 loss 侧**成立**：12 臂的 natural 半区一律用同一个冻结 Where 的 `m_pred` 加权，
`where_source` 只影响模型输入侧。审阅抓得对：这是一处未声明的第二处差别。

**第二次行文修正（因 NF-1 / amendment A-4）**：`<where>` 段的有无现在还决定了**该臂的 generated `<color>` 是怎么生成的**——
`C01`/`C02` 用 forced `<color>` prefix（prompt 无 `<where>`），其余臂用 with-`<where>`-prefix 生成。所以准确的表述是：
**`T01/T05` 与 `C01/C02` 的差别自始至终只有一件事——语言序列里有没有 `<where>` 段——而这件事同时决定了模型输入侧的
`H_color` 与 generated context 的生成方式**。两者是同一个决定的两个面，不是两处差别。

### D-W5 `C03/C04` 的 oracle `z_where`

§6 把 `z_where` 定义为 `AttentionPool(Q_axis, Q_readout)`，但 oracle 臂没有 MetaCanvas query state。
把**预测的** query state 喂给一个叫 "oracle" 的臂会让上界不成其为上界。

**保守默认**：`OracleLatentEncoder(w*, rho*) → 512` 作为 oracle 臂的 `z_where`，并在 arm facts 里记 `z_where_from="oracle_latent"`。
这两臂本来就不进主榜（§8.2），但口径必须写清楚。

### D-W6 `C01/C02` 的最终图像指标用哪个 mask 合成

§12.2 的 `I_out = I_in + m(T(I_in) − I_in)` 需要一个 `m`。**保守默认**：no-where 臂在评测时仍用**冻结 Where 的 `m_pred`** 合成，
理由是这两臂界定的是「Stage-What 拿不到 Where **输入**时的下界」，换渲染器会把控制变量和 mask 消融混在一起。
每行 per-sample 记录里都写 `composite_mask` 字段。

### D-W7 Aligned pooling 在哪个网格上做

§7.2 的 `v_i` 里有 `V(F_pre(p))`，`F_pre` 只在 `H/16 × W/16` 网格上有定义。**保守默认**：pooling 在 `F_pre` 网格上做
（`I_in` area-downsample 到同网格、`m_pred` 用低分辨率 readout `m_low`）；§9.1 的 natural 查询色在**全分辨率**上按
guided-upsample 的 `m_hi` 采。两处用途不同，差别写进 `POOL_SPACE` 常量而不是含糊过去。

### D-W9（已裁定，协议 amendment A-2）`L_style_dist` 的 `d_func` 口径

**问题**（REVIEW-impl-What B-3）：初版把协议字面的 `d_func(T_i,T_j)` 换成了 `‖ẑ_gt_i − ẑ_gt_j‖`（L2 归一化后的 SRHT 码距离），
且**未列入待决策清单**。因为 `z_gt = L2Norm(SRHT(u − mean))` 丢掉 `u` 的幅度，同一 look 的强度变体之间目标距离为 0；叠加
`L_style_cos` 也只管方向，结果是 `L_what` 里**没有任何一项监督 `z_style` 的编辑幅度**。本仓库语料是 Lightroom preset，
强度变体是常见的。§9.5 禁止事后改 loss，所以必须在第一个臂起跑前定。

**主 agent 裁定**：恢复协议字面，`d_func(T_i,T_j) = ‖u(T_i) − u(T_j)‖₂ / C`，`d_style = ‖ẑ_style_i − ẑ_style_j‖`，
`C` 是 **train LUT 集上预计算的固定整体常量**，与 `mean_train_u` 在同一次遍历中算出并随之发布，**禁逐 batch 归一**。

**实现**：`losses.pairwise_d_func` / `loss_style_dist(z_style, u_gt, C)`；targets 携带原始 `u_gt`（14739 维，
micro-batch 4 时 236 KB）；`C` 的闭式使一次遍历就能精确覆盖全部 `N(N−1)/2` 对，无需抽样：

```
C² = 2 ( N·Σᵢ‖uᵢ‖² − ‖Σᵢuᵢ‖² ) / ( N(N−1) )
```

`srht.pairwise_rms` 实现该式；`make_zgt_center.py` 在算 `mean_train_u` 的同一趟里累加 `Σu` 与 `Σ‖u‖²` 并把
`d_func_scale` 写进 `zgt_center.npz`。builder 与 trainer 都在 `C ≤ 0` 时**拒绝启动**（`compute_batch` 的
`d_func_scale` 是必填 keyword，无默认值可退回逐 batch）。mock 集上实测 `C = 35.21`；接入后 `L_style_dist`
在 24 步 mock 里从 0.518 降到 0.084（T01），即该项确实在被优化。

### D-W10（已裁定，协议 amendment A-3）12 臂统一的 natural 采样 mask

**问题**（REVIEW-impl-What B-5）：见上面 D-W4 的行文修正——`where_source="none"` 的 C01/C02 当时在 loss 侧退回了全图采样，
是一处未声明的第二处差别。

**主 agent 裁定**：§9.1 的 `m_pred` 是**监督侧冻结量**，与 `T_gt` 同性质，不是模型输入。12 臂（含 C01–C04）的 loss 查询点
natural 半区**一律**用同一个冻结 Where checkpoint 的 `m_pred` 加权；`where_source` 只影响模型输入侧。

**实现**：`WhatBatchBuilder.build` 对每个臂都跑冻结 Where（`_frozen_where`），再由 `_model_signals` 决定其输出是否进模型。
`where_prefix=False` 的 C01/C02 因此多一次**较短的**监督前向（`prompt + <where>`，无 color body）——12 臂里 2 个臂付一次额外
前向，代价换来的是 12 臂优化同一个 loss。每条 target 与 per-sample 记录写 `natural_weighting`（`frozen_m_pred` /
`global_uniform`，后者只用于 global 样本），使该性质可被审计；local 样本拿不到冻结 mask 时**直接报错**，不静默退回全图。
已加回归测试：六个代表臂（T01/T04/C01/C02/C03/C04）对同一批样本产生**逐位相同的查询点**。

### D-W11（已裁定，协议 amendment A-4）Stage-What 的 `<color>` context

**问题**（REVIEW-impl-What 复审 NF-1，**审阅者自承是初审漏掉的**）：初版实现的 `color_ids` / `where_ids` 全部来自 record 的
**GT** 文本，即训练与评测 100% teacher-forced，且**此决策从未写进本节**——违反「属于决策的写入待决策节，不许静默拍板」。
§0 的「最终网络只接收 `I_in + instruction`」因此在 Stage-What 的任何交付数字里都没被检验过；§15 问题 3/4/5 的答案全部
条件在 GT 推理文本上；且跑完再改只能重训 12 臂。

**主 agent 裁定**：采纳审阅者选项 (a)，与 Where-B §5.4 对齐。四点：

1. **训练 50/50**：每个 micro-batch 固定一半 teacher（GT `<color>` hidden）、一半 generated（Base SFT 自回归生成的
   `<color>` hidden，由 token ids 重放）。generated 缺闭合标签**不回退 GT**。
2. **评测分报**：GT 与 generated 两种 context 分开报告，每个 checkpoint 每 context 一行，永不混成均值。
3. **generated 主榜**：`V_what` 选择读 generated-context 榜；teacher 榜并列，两者之差是「对 GT 推理文本的依赖程度」。
4. **控制臂 forced prefix**：`C01`/`C02` 的 generated context 必须由「无 `<where>` prompt + `<color>` forced prefix」
   生成，否则 where 推理会经 token ids 回流到严格 no-where 控制臂。生成模式由 `where_prefix` **推导**，非硬编码列表。

**实现**：新增 `q3vl/what/context.py`（`ColorContext` / `gt_color_context` / `generated_color_context`，复用 Where-B 的
`BalancedContextSampler` 与 `FormatStats` 以保证两阶段口径一致）与 `q3vl/what/stores.py`（`ColorGenContextStore`）。
**无 GT 回退是结构性的**：`generated_color_context` 的签名里根本没有 GT 文本这个入参（preflight `WT-P7` 对**签名**做断言，
不是读实现体）。

**上游依赖与接口对齐**：generated context 由 Where-B 的生成作业产出（WB-IMPL 已提交 `dc6944c`，schema
`q3vl.where_b.genwhere/2` = v1 只增不改 + `<color>` 段 + `mode`）。**三个共享量由本包 import 而非重声明**——
`COLOR_CONTEXT_MAX_TOKENS`(384)、`GENCTX_MODES`、`SCHEMA_GENCTX` 全部来自 `q3vl.whereb.config`，并在 import 处断言
mode 词表一致（producer 用 `two_segment` / `forced_color`；本包保留可读的 `GENCTX_MODE_WITH_WHERE` /
`GENCTX_MODE_FORCED_COLOR` 作**标识符**，值取 producer 的）。边界 384 是两侧**各自独立**测出的同一个数
（本包据 3745 条抽样：min 108 / p50 178 / p95 246 / p99 285 / max 324，+两标签 +18% 余量）。teacher 侧超界**报错**
而非截断；全语料校验列为 `WT-J9`。消费侧逐条断言：缺字段、schema 非 v2、`mode` 不匹配、覆盖不全，四种都硬停。

### D-W12（已裁定，NF-2 路线 (a)+离线互补）在线评测 vs 离线选择

**问题**（REVIEW-impl-What 三审 NF-2，**审阅者自承第 2 点是二审漏审的**）：`run_what.py` 构造 trainer 时不传 `eval_fn`，
于是三件事同时不成立——§10.4 的 `eval_steps: 500` 在生产路径上未实现；**B-2 的 checkpoint 保护在生产上是失效的**
（无 eval 报告 ⇒ `best()` 返回 None ⇒ 无人被标 `best_protected` ⇒ 滚动删除照常吃掉前 6 个）；amendment A-4 的双榜
没有生产者。B-2 的 7 条回归测试全部注入 `_EvalStub`，所以全绿而生产无保护。

**主 agent 裁定**：路线 (a) + 离线互补。两条**不是同一个测量**，命名上必须分清：

| | 在线（`evalloop.make_eval_fn`） | 离线（`scripts/evaluate_what.py`） |
|---|---|---|
| 数据 | `V_what` 的**固定确定性子集**（256 条） | **完整** `V_what` |
| 指标 | LUT function 级（§9.1 的 2048 查询点）+ bake gate | §12.1 全套 + §12.2 图像指标（渲染 `I_out` 对 `I_tar`、三分区、分层）+ §12.3 |
| 频率 | 每 `eval_steps`(=500) 步 | 训练后一次 |
| 用途 | 决定**哪些 checkpoint 文件活下来** | 决定**哪个 checkpoint 赢**（`main_board` 的输入） |
| 主键 | `ONLINE_SELECTION_KEY = "local_lut_de00_median"`（代理） | `local_image_de00_median`（§12.4 主键） |

**代理与主键必须同向**：两者都是 local 样本、generated context、CIEDE2000、越小越好。**不同向的代理会让滚动删除
丢掉离线榜后来想要的那个文件——那就是 B-2 换了一层再犯一次**。在线报告里**故意不写** `local_image_de00_median`
（这一趟不渲染图像），一个名字与别处含义不同的键比一个缺失的键更坏。

**子集**：256 条，按 `(build, render_mode, mask_area_bin)` 分层、最大余数比例分配，组内按
`sha256(seed|sample_id)` 排序取——**跨进程/跨机器/跨 Python 可复现**（`random`/`torch.randperm` 只在单进程内够用）。
清单落 `run_dir/eval_subset.json`，其 digest 进 `run_setup.json` 的 `config_digest`，所以「在线代理是在哪 256 条上算的」
是这次 run 身份的一部分。mask 面积来自 Where-A 已发布的低分辨率 maskview（40×32 的小数组，897 条读取代价可忽略）；
**maskviews 未发布时 manifest 记 `mask_area_used: false` 并退回两键分层**，不静默少一层。

**墙钟实测与预估**：见第十一节。

### D-W8 未在协议中固定、已按惯例取值的次要常量

Color connector 深度 6（与 §5.1 的 Where connector 同型，只去掉 `F_pre` cross-attn）｜ Charbonnier ε=1e-3 ｜
Huber δ=1.0 ｜ uniform 采样 8³ strata × 2 点 ｜ SRHT seed = 20260804（战役 seed，已公开写进 config）｜
FIFO queue 256 / 暖机 32 ｜ 梯度比值记录周期 50 步 ｜ head bottleneck FG=128（SB 由求解器给出 137）｜
`z_style` head 投影 128 ｜ `log(Σa_i)` 特征 clamp ±40 后 /10 ｜ instruction-shuffle 最小裕度 0.25 ΔE00（预注册）。

## 四、参数量表（12 臂）

完整机器可读版：`config/arm_matrix.json`。

| Arm | WC | Generator | Where 源 | 可训练参数 | 每样本 LUT 参数 | head bottleneck |
|---|---|---|---|---:|---:|---:|
| T01 | WC-0 | FG48 | predicted | 91,364,898 | 1116 | 128 |
| T02 | WC-1 | FG48 | predicted | 92,415,522 | 1116 | 128 |
| T03 | WC-2 | FG48 | predicted | 92,739,106 | 1116 | 128 |
| T04 | WC-3 | FG48 | predicted | 93,789,730 | 1116 | 128 |
| T05 | WC-0 | SB48 | predicted | 91,376,823 | 684 | 137 |
| T06 | WC-1 | SB48 | predicted | 92,427,447 | 684 | 137 |
| T07 | WC-2 | SB48 | predicted | 92,751,031 | 684 | 137 |
| T08 | WC-3 | SB48 | predicted | 93,801,655 | 684 | 137 |
| C01 | NoWhere | FG48 | none | 91,364,898 | 1116 | 128 |
| C02 | NoWhere | SB48 | none | 91,376,823 | 684 | 137 |
| C03 | OracleWhere | FG48 | oracle | 94,109,730 | 1116 | 128 |
| C04 | OracleWhere | SB48 | oracle | 94,121,655 | 684 | 137 |

**FG/SB 配对差值**（§7.5 要求 ≤2%）：六对全部 **11,925 参数 = 0.013%**。

参数量的大头是 ModLN：`z_style` 是 1024 维、backend 宽 512，每个子层一份 `Linear(1024, 1024)` = 1.05M，
每 block 4 份 → 4.2M，6 blocks → 25.2M。这是 §7.3「ModLN 调制」+「每子层独立参数」（PLAN 1.4 / RD-G 先例）的直接后果，
不是实现膨胀。若主 agent 认为 90M 的 adapter 过大，唯一无损的削减点是 ModLN 投影共享或先把 `z_style` 降维再喂 ModLN——
**这属于改结构，需要主 agent 决策，本次未做**。

## 五、mock 端到端闭环数字

设置：结构完全保真（48 slots / 16 color queries / 2 seed + 4 refinement blocks / 完整参数布局 / 完整 §9.5 loss），
仅把宽度降到 64 以便 CPU 上跑；8 个 mock 样本、micro-batch 2（= 每 4 步一轮），共 24 步 6 轮。
mock 的 GT 是**已知的仿射 LUT** 烘到真实 17³ 表上，`H_color` 是该 LUT 自身参数的固定线性嵌入——
所以"loss 下降"等价于"`H_color → z_style → 48 Gaussians → T_pred` 这条路真的携带信号"。
mock 学习率用 1e-3（协议的 1e-4 在 24 步内只移动 ~0.5%，信噪比不足以当断言）；协议学习率由
`test_optimizer_groups_match_protocol_10_4` 单独断言。完整数字：`config/mock_e2e.json`。

**（2026-08-05 第二次重跑：接入 amendment A-2/A-3/**A-4** 之后，战役环境
`/home/bc/envs/q3vl_sft/bin/python`（3.12.12 / torch 2.10.0+cu128）。A-4 生效后每步都是 50% teacher / 50% generated，
下表的 loss 是两种 context 混合后的批均值。）**

| 逐轮均值 | T01 (WC-0+FG48) | T08 (WC-3+SB48) |
|---|---|---|
| total loss | 1.28141 → **1.15990** | 1.27762 → **1.13450** |
| `L_func`（teacher 半区） | 0.20071 → **0.19217** | 0.20027 → **0.18667** |
| `L_func`（generated 半区） | 0.14839 → **0.14509** | 0.14865 → **0.14466** |
| `L_bake` | ≈0.001（Charbonnier ε 的数值地板） | 同 |
| 每步 50/50 | ✔ 全部步 `n_gt == n_generated`，`teacher_fraction` 取值集合 = {0.5} | 同 |

**这两列不可跨 context 比大小**：`BalancedContextSampler` 把数据集**一次性**切成互斥的 teacher / generated 两池
（这正是「1 epoch = 每样本只被看见一种 context」的实现方式），所以两列跑的是**不同的样本子集**，
mock 里两池的 LUT 难度本来就不同。可比的是**各自的趋势**——两列都单调下降。
真实的 "generated 比 teacher 差多少" 要由 §12 的评测双榜（同一批 `V_what` 样本各跑两遍）给出，
`evaluate.context_report` 的 `gap` 就是那个数，不是这里。

mock 集上的 A-2 常量 `C = 35.2095`。`L_style_dist` 从 0.518 降到 0.084/0.312，说明新口径下该项确实在被优化——
旧口径（归一化码距离）下它只能约束方向，与 `L_style_cos` 冗余。

**`grad_norm(10·L_hc) / grad_norm(L_func)`**：T01 首测 11.70、末测 9.32（区间 9.17–11.70）；
T08 首测 11.67、末测 7.28。权重是 10，两项量级同为 O(0.1)，所以比值 ~10 是**设计值**，
不是 A0 那种 229 倍的单位错误。这条比值每 50 步写进 `steps.jsonl`。

`bake_non_finite` 全程 0；参数域（μ∈cube、σ>0、SPD、opacity/existence∈(0,1)）在训练后仍全部成立。

**两个需要结果审阅注意的读数**：
1. `n_active_mean`（opacity>0.5 且 existence>0.5 的 primitive 数）在初期恒为 **0**，因为 opacity 初值是
   `sigmoid(z−2)≈0.12`（GLUT/RD-G 先例，恒等初始化依赖 gate 是公因子）。这是设计而非坍缩，**但 §12.3 的
   "48 个 existence/opacity 激活数分布"在训练早期读 0 是正常的**，不要误判。
2. `z_effective_rank` 在 micro-batch=2 时上界就是 1，必须在 ≥32 个样本的评测集上算才有意义。
   同理 §12.3 的 `z_dist_spearman_vs_func` 需要 ≥3 个样本才会出现在日志里（micro-batch=2 时缺席是正常的）。
3. **SB48 的 `geom_drift_mu` 恒为 0.0000，FG48 的在 24 步后是 0.0371**（新增读数，见审阅 N-6）。这正是两臂的结构差别：
   SB 的 provisional 与 refined geometry 是同一个张量，FG 的不是，而协议 §7.4 对两者不加任何约束项。若 FG 的 drift 在正式
   训练中变大，§7.2 的「每个 Gaussian 对应真实颜色分布」就不再成立——请结果审阅盯这一列。

## 六、零初始化门的一个真实后果（实现审阅请注意）

§7.3 要求 zero-init gated residual。后果是：**第 0 步 `v_i`（aligned pooling 的结果）对 refinement blocks 完全无影响**，
所以第 0 步没有梯度经 pooling 回到 provisional geometry。有梯度的是 `gate_v` 本身，门一旦离开 0，通路就打开。
这与 Where-B connector 的行为一致，测试里分成了两条（`test_the_v_gate_receives_gradient_at_step_zero` 与
`test_gradient_reaches_the_provisional_geometry_through_the_pooling`），不是把断言放宽混过去。

## 七、数据派生物作业（已写好，**未执行**）

| 脚本 | 作用 | 状态 |
|---|---|---|
| `q3vl/what/scripts/pack_gt_luts.py` | 把 ~3.4k 个 `.cube`/`.3dl` 打成 §2.3 indexed-tar shards（`<lut_id>.lut.npy` float32 + `.lutmeta.json` 含源文件 sha256） | **未执行**；已在 V_what+T_lut_unseen 上 `--dry-run` 验证：790 个 lut_id、0 冲突、0 缺失 |
| `q3vl/what/scripts/make_zgt_center.py` | 只在 **train** 的 lut_id 上算 `mean_train_u`，再给全部 split 的 LUT 出 `z_gt` | **未执行** |
| `q3vl/what/scripts/run_what.py` | 单臂 runner（preflight 前置、Where checkpoint digest 校验、`run_setup.json`） | **未执行** |

float32 而非 float16：`.cube` 是 6 位小数，§12.1 的 bake gate 是 1e-4 RGB，而 float16 在 1.0 附近的分辨率约 1e-3——
用 float16 存 GT 会把 gate 压到自己 ground truth 的噪声以下。

## 八、测试清单

`q3vl/what/tests/`，共 **254 个测试**（base 环境全过；战役环境 221 过 / 8 skip）。

战役环境的 8 个 skip 分两类，都不是缺陷：
- 4 个跨实现对拍缺 `colour` / `skimage` / `dataset_build.src.construct.rendering`（后者因 R6 的同一个
  libstdc++ 问题）——生产路径只用 `dataset_build.lut_io`（纯 numpy），不受影响；
- 4 个 published-shard store 契约测试需要 `q3vl.data.shardio`（触 sqlite3），在 pytest 进程里 torch 已先加载，
  故按 R6 跳过。**生产入口脚本已加 sqlite3-before-torch guard 并有 AST 单测钉住**，实测三个脚本在战役环境下
  连同两个 store 一起 import 正常。

- `test_review_blockers.py`（29 个）逐条钉住初审的六个 blocker，尽量复用审阅人自己给的构造性反例，
  **每一条在修复前的代码上都会失败**（不是「跑一遍已修好的路径」）。
- `test_a4_color_context.py`（35 个）钉住 amendment A-4：无 GT 回退（对**签名**断言，不是读实现体）、
  截断/EOS/空生成的四种 stop_reason、teacher 侧超界报错、micro-batch 2/4/8/32 全部恰好 50/50、奇数被拒、
  两阶段 mode 字符串一致、一个样本每 epoch 只被看见一种 context、arm→生成模式由 `where_prefix` **推导**而非硬编码列表、
  published store 的四条契约（v1 拒收 / mode 不匹配 / 未知 mode / 覆盖不全）、builder 的两条分支与三种拒绝、
  trainer 无 genctx 时拒训、mock 跑批逐步记录 50/50 与两个 context 的 `L_func`、
  **选择只读 generated 榜**（teacher 行再好也选不上）、无 context 标签的行被拒、`context_report` 的 gap。

| 文件 | 覆盖 |
|---|---|
| `test_srht.py` (7) | FWHT 对拍显式 Hadamard；跨实例逐位确定；范数/距离保持实测；`z_gt` 单位范数；中心平移生效 |
| `test_lut.py` (10) | 真实 `.cube` 轴序回读；恒等 LUT；三线性对拍 dataset_build CPU oracle；四面体对拍 glut_repro 与 colour-science；格点精确；仿射精确；`DOMAIN_MIN/MAX`；非有限值拒绝；LutBank LRU |
| `test_gaussians.py` (17) | anchors 是 4×4×3 cell center；参数布局 23/14/1116/684；μ 闭 cube；σ 有下界且**线性增长**（非 exp）；SPD 特征值；log-density 手算对照；权重归一化；**零初始化恒等 <1e-4**；**字面公式 = 2x**；clamp；各参数组梯度非零；GeometryBank 从 anchors 起步；bake 与 render 在格点一致 |
| `test_colorspace.py` (10) | 对拍 glut_repro / skimage；ΔE00 自比 0；**229 倍单位放大实测**（a/b 恰 128×、L 恰 100×）；hue-cos 等价 atan2 形式且无分支切割；灰点不产 NaN；`normalize(C)` 用全局常量 |
| `test_pooling.py` (10) | 与朴素式对拍；特征宽度 264；**无有效像素回退逐值等于 masked global pool**；全 padding 不 NaN；mask 条件化生效并被记录；梯度到 μ 与 Cholesky；ROI/BG 公式；全局池化尊重 padding |
| `test_attention_color_backend.py` (12) | 与 Where-B MHA 逐位一致；全掩码行返回精确 0；**零初始化门使 `M_color` 与语言无关**；`ColorStack.forward` 签名封闭；ModLN 初始即普通 LayerNorm；48 head **互不串扰**（梯度隔离）；head 零初始化；参数量对拍解析式 |
| `test_generator_and_matrix.py` (13) | FG/SB 形状与每样本预算；**FG geometry 逐样本、SB 跨样本共享**；`gate_v` 第 0 步有梯度；门打开后梯度穿过 pooling 回 geometry；SB geometry 独立梯度；**四个 WC 配对全部 ≤2%**；两臂同深同宽同 query 数；12 臂矩阵 = 4WC×2gen+4；WC 表符合 §6；loss 权重逐值 |
| `test_queries_losses_metrics.py` (27) | uniform 集固定且每 stratum 恰 2 点；natural 跟随 mask、对 seed 确定、零 mask 回退；Charbonnier；两半分别报告；`L_hc` 手算对照 + 灰点不贡献；梯度比值；binary entropy 手算；style 队列 FIFO 有界且 stop-grad；var/cov 惩罚坍缩、暖机前静默；仿射函数 `L_bake≈0`；bake 可微；总 loss = 手算加权和；bake gate 三条阈值；SSIM/PSNR；合成式；分区覆盖率和为 1；effective rank / Spearman；激活分布；字典序选择 |
| `test_model_and_wc.py` (28) | 六种 WC token 数；`WC-0` 不吃 mask；缺信号即报错；oracle 臂走 `OracleLatentEncoder`；**12 臂全部可构造并前向**；forward 签名 == 白名单；`WhereSignals` 无禁字段；batch 拒绝非白名单输入 |
| `test_e2e_mock.py` (9) | T01/T08 逐轮 loss 与三个监督项同时下降；全程有限；两半分别报告；梯度比值有限；参数域训练后仍成立；**门确实会打开**；优化器分组符合 §10.4；ModLN/LayerNorm 无 WD；setup 记录完整；`compute_batch` 可复现 |
| `test_preflight_and_evaluate.py` (16) | `Z_STYLE_DIM == ZGT_DIM`；§10.4 与 §12.1 常量逐值；9 条 preflight 单检；零初始化行记录 2x 测量；**缺失必检 = 失败而非静默通过**；标识符扫描对 `torch.where` 不误报、对 `h_where` 必报；**主榜排除 ceiling 臂**；每臂只留一个 checkpoint；**选择拒绝 T_final/T_lut_unseen/V_where**；gate 失败传播；**Stage-What 未重声明 hidden 契约**（全包扫描） |

## 九、未完成 / 待 GPU 的部分

见同目录 `PREFLIGHT_WHAT_PENDING.md`。


---

## 十、对 REVIEW-impl-What 的逐条处置（2026-08-05）

审阅判定 6 BLOCKER / 16 NIT。以下为处置记录；测试列指钉住该修复的测试名。

| 项 | 处置 | 测试 |
|---|---|---|
| **B-1** gate 未参与选择 | `evaluate.main_board` 把 §12.4 step 1 变成**过滤器**：gate 未过的臂/step 不进 `ranked`，单列 `gate_failed` 表；全部未过时 `ranked` 为空、出 `diagnostic_ranked` 并整榜打 `WHAT-GATE-FAILED`（§5.6 同构）。`WhatTrainer.best` 同样跳过 gate 未过的 checkpoint | `test_b1_*`（4） |
| **B-2** 滚动删除会删掉将被选中的 checkpoint | 新增 `best_protected` 标志与 `_protect_best()`，在**每次 save 之后与每次 eval 之后**都重算（两条路径都覆盖，这正是 S0-TRAIN 双删除路径的教训）；`_roll` 尊重两种保护；被删条目保留在 `state.saved` 里标 `deleted` 以保全 checkpoint index；新增 `lost_best_steps` 作为「保护失效」的显式告警；`keep_last=None` 可整体关闭滚动 | `test_b2_*`（6，含两种 eval/save 交错顺序） |
| **B-3** `d_func` 被换成归一化码距离 | 见 D-W9 / 协议 amendment A-2 | `test_d_func_is_the_raw_u_distance_not_the_normalised_code`、`test_style_dist_matches_a_hand_computed_huber` |
| **B-4** §12.3 诊断与优化项同源 | `style_diagnostics` 的头号读数改为对**原始 `‖uᵢ−uⱼ‖`** 的 Spearman（`z_dist_spearman_vs_func`）；方向-only 的 `z_dist_spearman_vs_zgt` 并列保留但不作为唯一读数 | `test_the_function_spearman_sees_magnitude_that_the_zgt_spearman_misses`（构造一个「方向全对、幅度全错」的码：vs_zgt > 0.99 而 vs_func < 0.9） |
| **B-5** C01/C02 与 T01/T05 有两处差别 | 见 D-W10 / 协议 amendment A-3 | `test_b5_*`（5，含「六个臂产生逐位相同查询点」） |
| **NF-2** 生产路径没有接线评测 | `run_what.py` 构造并传入 `eval_fn`（固定 256 条确定性分层子集、双 context、LUT function 级）；新增 `evalloop.py` 与离线全量入口 `scripts/evaluate_what.py`；`best()` 改用 `TrainConfig.selection_key`（在线代理），与 §12.4 主键**同向**。回归测试**不再依赖 `_EvalStub`**：既有对 `run_what.py` 调用点的 AST 断言（`eval_fn` 是否真传、边界门是否在 trainer 之前），也有**真实 `make_eval_fn` 驱动真实 trainer** 验证 B-2 保护确实生效 | `test_nf2_eval_wiring.py`（25） |
| **N-24** 边界校验排在开跑之后 | 提升为**开跑前硬前置**：`scripts/scan_color_boundary.py`（纯读 record）+ `boundary.require_color_boundary_scan`；缺失 / schema 过期 / 边界不符 / split 未覆盖 / 有超界样本 / 有缺字段记录，六种都硬停 | 同上（7 条） |
| **N-22/N-23** | `text` 明确为元数据（结构性证明覆盖的是 token 路径），WT-P7 增断言「`data.py` 的 generated 分支只从 record 取 `text`，且分支内不提 GT 文本」；`assert_covers` 顺带抽一条 `record()`，把 schema/mode 两条契约从「第一个 batch」提前到「启动前」 | WT-P7 + 2 条 |
| **N-18/N-19** | `best()` 返回 `gate_fallback`（全员未过 gate 时是诊断而非选择，与 `main_board` 的 `selection_possible: false` 同形）与 `n_gate_unknown`（缺 `gate_pass` 键的计数不再隐形） | 1 条 |
| **B-6** Where checkpoint digest 校验只在文档里 | 新增 `q3vl/what/provenance.py`：`file_sha256` + `assert_where_consistency` 扫描 `RUN_ROOT/*/run_setup.json`，digest 不一致、**已宣称却缺失**、或 setup 不可读，三种情况都 `WhereProvenanceError` 硬停；`run_what.py` 在任何昂贵操作之前调用它 | `test_b6_*`（6） |

NIT 处置：**已修** N-1（两份交付物改在战役环境重新生成）、N-2（灰点 docstring 改正为「返回 1.0，但被 `normalize(C_gt)≈7e-7`
压掉」）、N-3（pooling 的 finite 断言与 6 个标量统计改为按日志周期开启，`collect_stats=False` 时跳过 device sync）、
N-4（`m_hi` 标注改 `list[Tensor] | None`）、N-5（`sample_row` 增 `mask_source` 显式入参，C03/C04 不再被标成 `"predicted"`）、
N-6（`steps.jsonl` 新增 `geom_drift_mu` / `geom_drift_sigma` / `n_sigma_over_cube` / `sigma_max`）、N-7（p90 两种口径都报，
选择键仍是逐样本 p90 的中位数，另出 `lut_de00_p90_of_sample_means`）、N-8（instruction-shuffle 改**配对**中位差，
不配对值并列保留因为预注册阈值是按它定的）、N-12（StyleQueue 不再往返主机）、N-14（`check_inputs(expect_source=...)`
断言臂与 `WhereSignals.source` 一致）、N-15（平局取更早的 checkpoint，写进 `_best_per_arm` docstring）、
N-16（preflight `--out` 默认改到 `RUN_ROOT/preflight`，并拒绝把 `complete: true` 覆盖成 `false`，`--force` 时先备份）。

**已补进 pending 清单**：N-9（image-shuffle 负控制、WC-1/2/3 相对 WC-0 的 paired improvement、paired bootstrap 95% CI）、
N-10（33³ baked render 的最终图像指标与可视化）、N-11（bf16 下 `out.params` 与 pooling 内 Mahalanobis 的实测 dtype 并入 `WT-G6`）。

**未处理并说明理由**：N-13（把 hidden 契约重声明扫描提到 `q3vl/` 根的共享 helper）——需要在 `q3vl/` 根新建模块，
超出「不改 data/train/where/whereb」的边界之外还会牵动 Where-B 的既有测试，留给主 agent 决定是否单开一张卡。


---

## 十一、在线评测的墙钟（NF-2 要求给出）

**已实测（CPU，生产宽度，8 线程，T04 即最重的 WC-3）**，每样本：

| 环节 | ms/sample |
|---|---:|
| WhatModel forward | 22.2 |
| render 2048 查询点 | 0.6 |
| **bake 33³** | **15.1** |
| tetra 回读 | 1.4 |
| What 侧合计 | **39.4** |

一次 eval = 256 条 × 2 context = 512 次样本前向 → **CPU 上 What 侧 20.2 s**。

**GPU 预估**（H100，bf16）。主导项不是 What 侧而是**冻结 VLM 前向**（每样本每 context 一次）：

- 由正在跑的 Base SFT 实测反推：global batch 32、两卡 ZeRO-3、**fwd+bwd** 5.1 s/step ⇒ ~160 ms/sample。
  纯前向、单卡、无 ZeRO 通信、无反向约为其 1/4–1/6 ⇒ **30–50 ms/sample**。
- 512 次样本前向 × ~40 ms ≈ **20 s**；Where 前向与 phi/guided upsample 约 +15% ≈ 3 s；
  What 侧在 H100 上约 1–3 ms/sample ⇒ ~1 s。
- **一次 eval ≈ 25 s**。

**开销占比**：`eval_steps=500`、约 4975 步 ⇒ 每臂约 10 次 eval ≈ **4 分钟**。臂本身的步时同样由 VLM 前向主导
（32 样本 × ~40 ms ≈ 1.3 s/step ⇒ 约 1.8 h/臂），故在线评测的开销约 **0.2–0.4%**。这正是「LUT function 级、
不渲染全图、固定子集」三条限制换来的；若改成整个 `V_what`(897) 且渲染全图，同样的 10 次 eval 会变成小时级。

**注意**：VLM 那一项是**从 Base SFT 的训练步时外推的**，不是直接测的。已列为 `WT-G9`，两卡释放后与
`WT-G1`–`WT-G8` 一起实测确认；`make_eval_fn` 每次都把 `eval_seconds` 写进 `eval.jsonl`，所以第一次真实 eval
之后这个数就不再是估计。C01/C02 另付一次**较短的**监督前向（`prompt + <where>`，无 color body），已计入。

---

## 十二、EXEC-4 执行记录（2026-08-10，C 波数据前置 + GPU preflight + 启动）

主 agent 任务卡 EXEC-4：Where-B 主臂被叫停、新 Where 方案待定，但 What 的四个控制臂不依赖 Where 预测，
提前跑 C 波把两卡用起来。本节记录**实测数字**、**新发现的问题**与**待主 agent 决策项**。

### 12.1 WT-J 数据前置作业（全部完成）

| 作业 | 产物 | 实测 |
|---|---|---|
| `WT-J9` `<color>` 边界全语料扫描 | `preflight/color_boundary_scan.json` | **162,359 条**（train 159,215 / V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433），`tokens.color` 缺字段 **0**，超界 **0**。全语料 max = **361**（+2 标签 = 363），边界 384 ⇒ **headroom 21**。抽样期的 max 是 324，全量比它高 37，**余量从 58 掉到 21**——边界仍成立但不再宽裕（见待决策 D-EXEC4-3）。5.5 s |
| `WT-J1` GT-LUT 打包 | `/mnt/nfs/bc/data/datasets/what-20260805/gtluts/` | **3,408 个 LUT**（train 3,149 / V_where 530 / V_what 531 / T_final 577 / T_lut_unseen 259），`lut_id→preset_path` 冲突 **0**、缺文件 **0**（这同时是 `WT-J3` 的全量版本，`collect_lut_paths` 读的就是全部 5 个 split 的全部 record）。6,816 个成员 / 3 个 shard / 2.4 GB / `status: complete`。440 s |
| `WT-J2` `mean_train_u` + `C` | `/mnt/nfs/bc/data/datasets/what-20260805/zgt/` | 中心只用 **3,149 个 train lut_id**；`d_func_scale C = 29.4131`（A-2 闭式）；`zgt.jsonl` **3,408 行**；`center_sha256 = f682a512…`、`center_abs_mean = 0.0837`；SRHT digest `87857cc5…`。64.9 s |
| `WT-J10`（**本次补完的上游件**） | `genwhere/train-forced_color`、`genwhere/V_what-forced_color` + 兼容软链 | 生成早已跑完（8+2 个分片，rc 全 0，2026-08-06 10:55 结束），但**合并从未执行**——`genwhere/<split>/forced_color` 根本不存在，C01/C02 直接开跑会在 `ColorGenContextStore` 构造时就拒绝。本次合并：train **159,215/159,215**（coverage 1.0，色段 format failure **0%**、截断 **0%**、全部 `closed`、`color_tokens` max 331）、V_what **897/897**（max 290）。166.7 s + 20 s |

`WT-J4`–`WT-J8` 属于评测/选择期作业，与 C 波开跑无关，未动。

### 12.2 `WT-G` GPU preflight（`preflight/preflight_what_gpu.json`，7 pass / 1 skip / 1 warn / **0 fail**）

新脚本 `q3vl/what/scripts/preflight_gpu.py`（全部走生产类，不复制管线）。

| id | 结论 | 实测 |
|---|---|---|
| `WT-G1` | pass | hook 与 `output_hidden_states` 逐位相同（max abs diff **0.0**）；layer `-1`、final RMSNorm **已施加**；`<color>` 切片 174 token 与 `n_color_tokens` 对齐；`h_color` (174, 2560)，序列 455+8+174=637 |
| `WT-G2` | pass | `where_prefix` 只改序列：637→629，`h_where` 由 8 token 变空，**`F_pre` 逐位相同**，`<color>` token 数不变、hidden 相对差 **0.0927**（这就是 T01 与 C01 之间那条残余通道的直接读数）。GT vs generated：174 vs 190 token，前缀重合处 hidden 相对差 **0.437** |
| `WT-G3` | pass | 768×512 → `F_pre` 48×32（stride 16.0/16.0），宽高比逐位一致，`rgb_low` 同网格 |
| `WT-G4` | **skip** | 无冻结 Where checkpoint。C 波恰好是唯一不消费它的四个臂；**T01–T08 开跑前必须补做** |
| `WT-G5` | pass | micro-batch 2/4/8/16 全部装得下，peak 10.1 / 10.2 / 11.1 / **12.9 GiB**（H100 95 GiB）。取 **16**，`grad_accum = 2`，effective batch 32 |
| `WT-G6` | pass | `loss` 与 `t_pred`、`bake`、`render` 全 float32，无外层 autocast 泄漏；**N-11 实测**：autocast 区内 `out.params` 里 `mu/sigma/off/opacity/existence/M/b` 是 float32，**`G` 与 `b_g` 是 bfloat16**（`compute_batch` 之后统一 `.float()`）；`aligned_pool` 的 Mahalanobis 项实测 **float32** |
| `WT-G7` | **warn** | 环境里 `lpips` / `torchmetrics` 都**没装**。只影响离线图像指标（§12.2 的 LPIPS 列），在线 eval 是 LUT function 级、不渲图，**不阻塞训练**；`image_metrics` 报 `nan` 而非静默替代 |
| `WT-G8` | pass | 33³ 烘焙 **0.26 ms/sample**（batch 16 共 4.1 ms），VLM 单样本前向 88.4 ms ⇒ 烘焙占 **4.6%** |
| `WT-G9` | pass | 实测训练 **0.0985 s/sample**（mb=16）；256 条 × 2 context × 0.6 ⇒ 一次 eval 外推 **≈ 30 s**，权威数字是 `eval.jsonl` 的 `eval_seconds`（smoke 实测 8 条子集 2.16 s） |

### 12.3 D-EXEC4：A-3 统一 natural 采样在 C 波的**声明式**偏离

**主 agent 裁定**（任务卡 EXEC-4 第 3 项）：没有冻结 Where checkpoint，因此

- `C03/C04`：natural 半区用它们**本就拿到的 GT mask** 加权（`--natural-mask-source oracle_gt_mask`，per-sample 记 `gt_mask`）；
- `C01/C02`：全图采样（`--natural-mask-source global_uniform`，per-sample 记 **`global_uniform_declared`**，与 global 样本天然的 `global_uniform` 区分开）。

**落码方式（关键：不是回退，是拒绝 + 显式声明）**：
`frozen_m_pred` 仍然是默认值，且**没有 WhereRunner 时直接抛错**；`run_what.py` 在缺 `--where-checkpoint` 时
① 对 `where_source="predicted"` 的臂**硬停**（主臂不得走这条路），② 对控制臂**要求显式给出** `--natural-mask-source`，
③ 把偏离写进 `run_setup.json.deviation`、`builder.natural_mask_source`、**以及 `config_digest` 的输入材料**
（换了监督 mask 的 run 不共享 digest）。新增 5 个回归测试。

**代价（已知并接受）**：C 臂的 loss 查询色分布与将来的 T 臂不同。新 Where 定档、D-W10 重新校准后若要求统一口径，
**C 波重跑**。

### 12.4 本次发现的三个真问题（都会让 C 波在没修之前跑不起来或跑出错东西）

1. **oracle 路径写错**：`run_what.py` / `evaluate_what.py` 用的是 `<oracle>/<split>`，而 Where-A 的实际发布布局是
   `<oracle>/<basis_arm>/<namespace>/<split>`（= `oracle/BA-3-Joint/s5/<split>`，`q3vl.whereb.config` 自己的注释与
   `run_where_b.py` 都是这么读的）。原路径解析到**不存在的目录**，C03/C04 必崩。已改为按 `BASIS_ARM`/`ORACLE_NAMESPACE`
   拼路径并可用 `--oracle-root` 覆盖。
2. **评测侧 oracle store 用错 split**：`run_what.py` 把 **train** 的 `OracleStore` 传给了 `V_what` 的 eval builder。
   store 按 sample_id 索引且只含本 split，C03/C04 会在**第 500 步的第一次 eval** 才崩。已为 eval split 单独构造。
3. **oracle 覆盖率 = 恰好所有 local 样本，global 样本一个都没有**（实测：train 75,544/75,544 local 有 fit、
   83,671 个 global 一个都没有；V_what 408/408 vs 489）。原实现遇到没有 fit 的样本**抛错**（"必须拒绝样本，不许编造"），
   即 C03/C04 会在第一个 global 样本上崩。这是**决策**，见下。

### 12.5 待主 agent 决策（已按保守默认继续，未静默拍板）

**D-EXEC4-1 · `C03/C04` 遇到没有 oracle latent 的样本怎么办**
一次 global 编辑没有 ROI，Where-A 因此没有为任何 global 样本拟合 latent。三条路：
(a) **拒绝样本** → C03/C04 的训练population 变成「只有 local」，与另外 10 臂不同population，主榜与分层报告都不可比；
(b) **保持population，给常量空 latent**（本次默认，`--oracle-missing-latent null_global`）：mask 用**诚实的全 1**
（`_oracle_signals` 本就如此），`w_vec`/`rho_vec` 取全零常量，逐样本计数写进 `run_facts_final.json` 的
`oracle_latent_stats`。语义上是「对 global 编辑，oracle 没有额外信息可说」——这本身是真的；
(c) 给 global 样本单独一个可学习的 null token（改结构，需重跑参数量匹配）。
**采用 (b)**。若主 agent认为 ceiling 臂必须只在 local 上定义，则应改判为 (a) 并同时规定主榜只读 local 分层。

**D-EXEC4-2 · `C03/C04` 用哪个 readout 的 oracle latent**
`oracle/BA-3-Joint/s5` 同时发布了 `band` 与 `cband12` 两套 fit（抽样 300 条 local，两套 status 全 `ok`）。
`rho` 维度不同（`band` 是 4 个标量，`cband12` 是 3×12），因此它决定 `rho` token 投影的大小与臂的参数量。
**默认取 `cband12`**（`ArmConfig.where_readout` 的默认值、W02 的 readout、amendment A-WhereA-1 §A3 专门定档了它的归一化约定）。
风险：将来冻结的新 Where 若用 `band`，C03/C04 与 T 臂的 rho 维度不一致；ceiling 臂本就不进主榜，但**分层对照时要注明**。

**D-EXEC4-3 · `<color>` 边界 384 的余量只剩 21 token**
全语料 max 361（抽样期 324）。teacher 侧超界是**抛错**不是截断，所以余量小意味着「未来任何新增/重建的 build
只要多写 22 个 token 就会让某个臂中途崩」。当前语料是安全的（超界 0），但如果还会再产数据，建议把边界提到 448。
**默认不动**（改边界会改 `COLOR_CONTEXT_MAX_TOKENS`，而它是 Where-B 与 Stage-What 共用的常量，动它要两侧同时重发）。

**D-EXEC4-4 · LPIPS 后端未安装**
`WT-G7` warn。离线评测（§12.2）要 LPIPS，环境里没有 `lpips` 也没有 `torchmetrics`。不阻塞 C 波训练，但
**在 `evaluate_what.py` 出主榜之前必须装**，否则该列全是 `nan`。装哪一个（`lpips` 官方 AlexNet/VGG 权重需要联网下载）
属于环境决策。

**D-EXEC4-5 · Where-B 入口脚本的 R6 guard**
NOTES R6 里遗留的那条越界项仍未处置（`run_where_b.py` / `make_generated_context.py` / `make_oracle_latents.py`
是否也加 `import sqlite3` 前置）。本次没有改 `q3vl/whereb/`。合并脚本 `merge_genctx.py` 本来就有该 guard。

### 12.6 顺带修的一个上游脆弱点（`merge_genctx.py`）

合并作业**写 `/mnt/nfs`（rw, nfs4.1）、读回却走 `/mnt/nfs-ro`（soft, nfs3）**，而刚创建的目录在 ro 客户端的
dentry 缓存里还不存在（acdirmax 默认 60 s）。第一次跑 V_what 时因此在第 4 步 `ENOENT: shards/shard-00000.tar` 失败——
**发布本身是完整的，失败的是它自己的校验**。已加 `_await_read_mirror()`：发布后只用 `os.stat` 在**软挂载**上轮询
（不可能挂死、不写任何东西），全部可见才继续，超时 300 s 则报错并提示「把发布挪走重跑」。
失败的那次发布已按纪律**挪走而非删除**：`genwhere/.V_what-forced_color.aborted-readback-20260810`。

### 12.7 C 波的提交（gpu-queue）

`waves/what_c_arm.sh`（payload，`exec` 不 nohup）+ `waves/enqueue_what_c_wave.sh`（波次）。
C1 = C01(gpu0) + C02(gpu1)；C2 = C03(gpu0) + C04(gpu1)，同卡排在后面 **且** gate 在前一臂的 `what_final.pt`
（本地盘，不是 `/mnt/nfs`）。四臂共用 `--micro-batch 16`，因此 `total_optimizer_steps` 与 LR 计划完全一致。
`--keep-last none`（审阅 N-26）。

**提交前清掉的一个雷**：`q resume gpu0` 会把 Where-B 停摆时留在暂停队列里的 **W03** 立刻放到卡上
（它 8 小时来一直是 `starting`、GPU 0%）。已 `q cancel W03`。

### 12.8 PERF-1 shard cache 对 What 的适用性（任务卡第 5 项：确认即可）

**确认成立，且自动生效**：`WhatDataset` 用的是 `q3vl.train.shards.ShardStore`，因此 What 的每一次 record / image
读取都经过 `resolve_read_path`，同时拿到 ① `/mnt/nfs` → `/mnt/nfs-ro` 的读路径重写、② 本地 shard cache。
不需要任何接线改动。

但 PERF-1 那次预热是按 **Where-B 的五个数据根** 做的（23 条目 / 28 GB），**不含 Stage-What 的新产物**。
本次补预热了 9 个 shard / 3.52 GiB（155–162 MB/s，sha256 全部 ok，缓存现为 32 条目 / 32 GB）：
`what-20260805/gtluts`（3）、`genwhere/train-forced_color`（2）、`genwhere/V_what-forced_color`（1）、
`genwhere/V_what`（1）、`where_a/maskviews/V_what`（1）、`where_a/oracle/BA-3-Joint/s5/V_what`（1）。

**时序注意**：cache manifest 是**进程内 memoise 一次**的，C01/C02 于 20:11 启动、预热 20:23 才完成，
所以**这两个臂全程不吃新缓存**（gtluts 与 forced_color 仍走 nfs-ro 随机读），C03/C04 才吃得到。
只影响墙钟，不影响数字（字节相同且逐成员 sha256 校验）；报告 C 波耗时时要提这一条，
否则会把 C1/C2 两波的步时差误读成臂之间的差别。

### 12.9 主 agent 对 §12.5 五项的裁定与落实（2026-08-10）

| 项 | 裁定 | 落实 |
|---|---|---|
| **D-EXEC4-1** oracle 缺 latent | **认可 `null_global`**（global 样本的 oracle where 语义上就是全图，全 1 mask 正确；逐样本计数保持） | 已是默认；计数写在 `run_facts_final.json` 的 `oracle_latent_stats`（`fitted` / `null_global`），`run_setup.json` 里是**声明值**（跑之前恒为 0），两份文件故意分开 |
| **D-EXEC4-2** oracle readout | **认可 `cband12`**（Where-A 结论其稳定占优）；**风险入档** | 见下面「风险 RK-1」，同时写进 `docs/reviews/REVIEW-impl-What.md` 的放行条件清单 |
| **D-EXEC4-3** `<color>` 边界 | **记录；任何新 build 前 384→448** | 注释写在常量旁：`q3vl/whereb/config.py::COLOR_CONTEXT_MAX_TOKENS`（产方，权威）与 `q3vl/what/config.py` 的 import 处（消费方）。**本次不改数值**——它是两 stage 共用常量，改它要两侧同时重发 generated context |
| **D-EXEC4-4** LPIPS | **现在就装** | 已装：`lpips 0.1.4`（`/home/bc/envs/q3vl_sft`），backbone torchvision 0.25.0+cu128。实测 `LPIPS(同一张图)=0.0`、`LPIPS(随机两张)=0.1255`；AlexNet 线性权重（6,009 B）**随包发布**，评测时不需要联网。落盘 `preflight/wt_g7_lpips_closure.json`（`WT-G7` 由 warn 转 pass）。**原 `preflight_what_gpu.json` 不改**——它当时说的是真话，静默改写发布过的 preflight 正是 R5 事故的形态 |
| **D-EXEC4-5** whereb R6 guard | **记录（whereb 暂停使用，新方案时一并修）** | 本次未改 `q3vl/whereb/` 的任何入口脚本。`merge_genctx.py` 本来就有该 guard，所以合并作业不受影响 |

**LPIPS 接线的一个未闭合点（留给离线评测任务）**：`lpips.LPIPS` 吃 `[-1,1]`，而 `q3vl.what.metrics.image_metrics`
传的是 `[0,1]`。`evaluate_what.py` 目前传 `lpips_fn=None`（列报 `nan`），**接线的人必须在注入处做 `x*2-1`**，
否则会得到一个安静的错数。已写进 closure 记录的 `input_convention` 字段。

### 风险 RK-1 —— `C03/C04` 的 oracle readout 与将来的 Where 可能不同档

C03/C04 用 `cband12` 的 Where-A oracle latent（`rho` = 3×12 = 36 维）。若新 Where 方案最终冻结在 **`band`**
（`rho` = 4 个标量），则：

- C03/C04 的 `rho` token 投影维度与 T01–T08 不同 ⇒ 两者的可训练参数量不再严格可比（`WCEncoder.proj["rho"]`
  是 `Linear(n_rho, 512)`，36 vs 4 相差 16k 参数，占 ~0.02%，**对 §12.4 的 `n_trainable_params` 键几乎无影响**）；
- 更实质的是**语义**：ceiling 臂的「上界」是在 cband12 的场参数化下定义的，而主臂在 band 下。
  C03/C04 本就**永不进主榜**（§8.2），但**分层对照与「离上界还有多远」这类叙述必须注明档位不同**。

处置：不阻塞 C 波；已同步写进 `REVIEW-impl-What` 的放行条件清单，供结果审阅逐条对照。

---

## 十三、任务卡 EXEC-5 —— C 臂离线评测入队（2026-08-10）

### 13.1 实施前核实记录（不是检索来的，是打开原始来源读的）

| 要核实的事 | 怎么核实的 | 结论 |
|---|---|---|
| `evaluate_what.py` 是否已接 LPIPS | 读源码 + `git show 80a5078 -- q3vl/what/scripts/evaluate_what.py` | **已接且已 `x*2-1`**（2026-08-05 的 NF-2 提交里就有）。`wt_g7_lpips_closure.json` 的 `still_open: "currently passes lpips_fn=None"` 是**过期陈述**。本次把 lambda 提成具名 `lpips_closure()` 并补单测（缺 `x*2-1` 不会崩，只会安静地量一对半对比度的图） |
| LPIPS 是否真的装了 | `python -c "import lpips"` 于 `/home/bc/envs/q3vl_sft` | 装了；`lpips_closure` 实测 identical=0.0、random=0.1031 |
| `V_what` 规模与掩膜可得性 | 真机 `open_dataset("V_what", need_mask=True)` | **897** 样本；`mask_source = published_maskviews`（不需要回落到 live resolver） |
| generated `<color>` context 覆盖 | `ColorGenContextStore.assert_covers` 两个 mode | `two_segment` / `forced_color` **各覆盖全部 897** |
| C03/C04 的 oracle latent | `OracleStore.latent(..., "cband12")` 抽 24 条 | 14 命中 / 10 缺失，缺的都是 global 样本 ⇒ 必须 `--oracle-missing-latent null_global`（已写进 wave 脚本） |
| 在线 eval 实测墙钟（`WT-G9` 闭环） | `C01/eval.jsonl` 第一条 | **57.23 s** / 256 样本 × 2 context ⇒ 0.112 s per (sample, context)，纯 LUT 指标 |
| C 波训练节奏 | `C01/train.log` step 600 `elapsed_s=2011.6` | 3.35 s/step ⇒ 4975 步 ≈ **4.6 h**；C1 约 00:50 完成，C2 约 05:30 完成 |

### 13.2 入队的四个任务

`waves/what_c_eval.sh`（单臂载荷）+ `waves/enqueue_what_c_eval.sh`（wave）。
每臂一次 `evaluate_what`，读 `what_final.pt`，完整 `V_what` 双 context，
`--full-grid --lpips`，输出到 `/home/bc/data/runs/what/evaluate/<ARM>/`。

| 任务 | 卡 | gate | 排在谁后面 |
|---|---|---|---|
| `eval_C01` | gpu0 | `C01/what_final.pt` | C01 → C03 |
| `eval_C02` | gpu1 | `C02/what_final.pt` | C02 → C04 |
| `eval_C03` | gpu0 | `C03/what_final.pt` | eval_C01 |
| `eval_C04` | gpu1 | `C04/what_final.pt` | eval_C02 |

组内 FIFO 保证顺序，gate 保证「前一臂没死」。`--ready 'progress'`：`evaluate_what` 每
32 个样本打一行进度，**打在真的算完之后**，所以 ready 不会因为「python 起来了」就误判成功。

### 13.3 本次修掉的三个会让评测直接失败或静默错的问题

1. **`--where-checkpoint` 曾是 required** ⇒ C 波（D-EXEC4，无冻结 Where）根本跑不了。
   现在按 `run_what.py` 同一套规则放行：`predicted` 臂仍拒绝；无 checkpoint 时
   `--where-readout` 与 `--natural-mask-source` 必须显式，且**与该 checkpoint 目录里的
   `run_setup.json` 交叉校验**，不一致就硬停（natural 半边的查询色分布不同 = 指标不可比）。
2. **builder 从未收到 `natural_mask_source` / `oracle_missing_latent`** ⇒ C03/C04 会在第一个
   global 样本上以 `ORACLE_MISSING_REJECT` 抛出。已接线。
3. **`I_tar` 读错了对象（最严重）**。`WhatDataset.load_target_image` 读的是 `image.baked`，
   而 `q3vl/data/bake.py` 写得很清楚：`baked` 是 **`I_in` 的契约尺寸副本**
   （`verify.check_bake_fidelity` 拿它跟 `prepare_image(原图)` 逐像素比）。实测该 locator 与
   dataset 已经加载的 `image` 成员 **shard/offset/sha256 完全相同** ⇒ 若它「能跑」，
   §12.2 全套图像指标就是拿预测结果去比**输入图自己**，量的是「你改得多小」。
   真正的 `I_tar` 是 build 里的渲染候选图 —— `.jpg` 成员，和 `.in.*`（`I_in`）、`.cgt.png`（GT mask）
   同批发布；这条 provenance `q3vl/where/maskdata.py` 的模块文档 2026-08-05 就写明了。
   **它没有静默出错，是因为它连类型都不对**（dict 传给了要 `MemberRef` 的 `ShardStore.read`），
   所以一读就崩 —— 这是运气，不是设计。
   已改：`MaskResolver` 接受 `suffix`（默认仍是 `.cgt.png`）并新增 `read_bytes`；
   `load_target_image` 走同一套 catalog 查 `.jpg`，过同一个 `prepare_image`，
   **断言与 `I_in` 同形而不是 resize**。
   **真机验证（13 个 V_what 样本）**：`I_tar ≠ I_in`（0/13 相同）；local 样本
   掩膜内平均 |Δ| 是掩膜外的 **6–10 倍**（例：0.1058 内 vs 0.0100 外），正是局部编辑该有的样子。

### 13.4 待主 agent 决策

| # | 事项 | 我采用的保守默认 | 为什么需要你裁 |
|---|---|---|---|
| **DQ-1** | **§12.2 的合成掩膜**（deviation **D-EXEC5**）。协议要求四臂都用**冻结 Where 掩膜**合成 `I_out`，好让 C01/C02 是干净的下界而不是「掩膜消融」。D-EXEC4 下没有冻结掩膜。 | `--composite-mask gt`：四臂**统一用 GT 掩膜**（唯一对每个臂都存在的掩膜；C01/C02 从未把它当输入，C03/C04 本来就是它）。`--composite-mask` **无默认值**，不给就拒跑；每行都记 `composite_mask` | 另一个选项 `model` 会让 C01/C02 用全 1、C03/C04 用 GT，**两臂的渲染器不同**，等于把控制臂和掩膜消融混在一起。我认为 `gt` 更贴协议原意，但这是判据口径，应由你确认 |
| **DQ-2** | **每臂只评 `what_final.pt`** | 是（单 checkpoint）。`keep-last none` 下每臂另有 9 个 500 步 checkpoint | §12.4 的选择规则允许跨 step 选。全扫 10 个 checkpoint × 2 context × 897 样本 ≈ 10 倍机时。建议：先看 `eval.jsonl` 的在线曲线，若终点不是最优再补跑指定 step |
| **DQ-3** | 四个任务各自出一份**单臂** `main_board` | 是 | 跨臂主榜需要把四份 `candidates_V_what.jsonl` 合并后再跑一次 `main_board`（纯 CPU，几秒）。**尚未实现合并脚本**，等四个任务跑完再做 |
| **DQ-4** | `I_tar` 取 build 的 `.jpg` 渲染图 | 是 | 备选是「用 GT LUT + GT mask 现场重渲」。`.jpg` 是 QA 与 winner 选择**当时真正评分的那张图**，是数据集自己的真值；重渲只是它的近似，且会让掩膜外误差恒等于 0（分区指标退化）。实测 `.jpg` 的掩膜外误差 ~0.006–0.010（JPEG 重编码 + 软边），**分区不退化** |

### 13.5 未做 / 已知缺口

- **未做 GPU 端到端 smoke**：两卡在跑 C 波，任务卡明令勿扰。VLM 那一半由 C 波本身在证；
  非 VLM 的全部环节（数据集、掩膜、genctx、oracle、gtluts、zgt、LPIPS、`I_tar`、12.2 分区）
  已在 CPU 上用**真实数据**跑通（见 13.1 / 13.3）。
- `WT-J7`（33³ baked render 的第二组图像指标与并排图）仍未实现，§13 联图交付前要补。
- `WT-J4`（image-shuffle 批次构造）未实现 ⇒ §12.3 的 image-shuffle 一列这次仍缺；
  instruction-shuffle 的配对差分已在 `arm_metrics` 里。
