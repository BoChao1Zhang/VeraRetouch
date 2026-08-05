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

`q3vl/what/tests/`，共 **229 个测试**（base 环境全过；战役环境 221 过 / 8 skip）。

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
