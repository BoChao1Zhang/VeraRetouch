# A0 GLUT 复现 · NOTES（实施前核实 + 假设 + 待决策）

日期：2026-08-03　任务卡：W1a（EXPERIMENTS_v3 W1 行 A0；IMPL_DOSSIER §二）

## 一、实施前核实记录

| # | 事实 | 来源 | 结论 |
|---|---|---|---|
| 1 | 复现规格全套（Eq.1-3 / 22N+12 参数 / Adam 1e-3 cosine 20ep bs1024 / Hald 128³ 颜色空间 split / 锚点 45.47±0.3 dB, ΔE00 0.41） | IMPL_DOSSIER §2.2–2.4（引用节，已读） | 按规格实现 |
| 2 | 两处论文矛盾的裁决：Cholesky 对角 **log/exp（init=log 0.15）**、opacity **raw+clamp（init=1.0）** | IMPL_DOSSIER §2.4 | 照办；CI 有断言 |
| 3 | mv-lab/nilut 仓库真实存在（GitHub API 一手核实，非附录 B 内 URL）；`dataset/cube-files/` 只放出 **LUT01.cube**（33³ Resolve 方言），全集在 Kaggle 登录墙后 | api.github.com/repos/mv-lab/nilut + raw 下载实测 | 冒烟只能拿到 1 个 NILUT LUT |
| 4 | 本地 64³ .cube 共 447 个（parse_report orig_size=64），其中全部冒烟/全量选出的 75 个均 used_in_prod、非近恒等、md5 去重 | tooling-wave1/cube/parse + inventory + near_identity_stats | A0 语料来源 |
| 5 | Hald 生成器与两路 LUT 应用器（grid_sample / tetrahedral）已过 identity 对拍与 IM 交叉验证 | tools/cube NOTES §5.4 + selfcheck 冻结产物 | 直接复用 cubelib |
| 6 | GLUT 官方仓库空壳、mv-lab/GLUT 系编造 | IMPL_DOSSIER §2.1 + 附录 B 警示 | 从零复现成立 |

## 二、假设与决策（保守默认已采用，未静默拍板的列入 §三）

1. **GLUT 原版初始化照抄**（μ 均匀网格、Σ iso σ=0.15、o=1.0、局部与全局仿射均 I+0 → f(x)=2x @init）。任务卡裁决：对齐锚点用原版；我方 G=0 改动属 RD 臂。CI `check_glut_original_init_is_2x` 固化该语义。
2. **Loss 作用在未 clamp 的 f(x) 上**（论文未写明；若在 clamp 后算 L1，2x 初始化下 f>1 区域梯度全零）。留 `--loss-on-clamped` 开关做对照。
3. **非立方数 N 的 μ 网格**：取 k=⌈N^{1/3}⌉ 网格后按 linspace 均匀抽 N 点（论文 Uniform 45.47 vs Random 45.45，差异 <0.05 dB，任何确定性近似均可）。
4. **75 LUT 批内并行训练**（一个 BatchedGLUT 实例、独立参数/GT/采样流）＝统计上等价逐个训练；冒烟已验 batched-vs-single 一致性（CI `check_batched_equals_single`）。
5. **PSNR 双口径都报**（float MSE 与 round×255）；FiveK 的 round 口径红线不直接适用 Hald，但双报以便对齐。
6. **GT 路径**：原生分辨率 .cube → colour 四面体插值（GT 口径与工具链一致）；float16 缓存（量化噪声 ~5e-4，对 45 dB 量级无影响）。
7. torch Lab（L_hc 用）与 cubelib colour Lab 偏差 0.021（IEC 圆整矩阵 vs colour 色度推导矩阵，与 tools/cube NOTES §5.3 的 skimage 偏差同源同量级）；L_hc 是损失项非评测口径，CI 阈值 0.05。

## 三、待主 agent 决策

1. **75-LUT 集合的锚点可比性**：GLUT 原 75 个 64³ 文件不可得（官方仓库空壳）。保守默认 = 本地 447 个 64³ 生产 cube 确定性抽 75（seed 0，清单 config/a0_luts_75.txt）。锚点 45.47±0.3 是 GLUT 自有语料上的数：**在不同语料上 ±0.3 判据只能作方向参照**。冒烟 2-LUT 已见 48.9 dB / ΔE00 0.25 —— 高于锚点、无欠拟合迹象；若全量 75 均值显著高于 45.8，判读应为「本语料较 GLUT 75 平滑」而非「复现超越」。是否需要另寻更接近 GLUT 语料（如购买/申请 NILUT Kaggle 全集）请主 agent 拍板。
2. **NILUT 7-LUT 冒烟降级为 1+1**：Kaggle 登录墙。若需要严格 7-LUT 冒烟，需提供 Kaggle 凭据。
3. 自然图协议（S-test 100 张替代 FiveK #4501–4600）本轮未跑（DATA_ASSIGNMENT E1 行的自然图列），建议长任务出数后作为第二阶段。

## 四、冒烟结果（2026-08-03，2 LUT × rec 臂 × 20ep，评测 2^21 留出色子样）

| LUT | PSNR(float) | PSNR(8bit) | ΔE00 mean | p99 |
|---|---|---|---|---|
| nilut__LUT01 | 49.02 | 48.27 | 0.251 | 1.64 |
| e18__e18_000902 | 48.78 | 48.05 | 0.245 | 1.57 |

loss 0.476 → 0.00237（41k 步 430s，B=2）。曲线单调下降，无发散。CI 自检 10/10 PASS。

---

# 附录 A：锚点缺口归因轮次（2026-08-03，第二棒）

任务卡：A0 锚点缺口归因（full 臂 bug 猎捕 + rec 臂缺口三假设）。

## A.1 实施前核实记录（本轮新增）

| # | 事实 | 核实方式 | 结论 |
|---|---|---|---|
| 1 | GLUT 附加项净增益 +0.11 dB（45.36→45.47）、hard mining 45.13→45.47、λ_hc=10、R_sparse=0.001 | IMPL_DOSSIER §2.2 Loss 段与训练配方段（引用节，重读） | full 臂 −9.2 dB 与论文矛盾 → 必是实现缺陷，成立 |
| 2 | 论文 A.1 初始化措辞「identity 矩阵 + 零 bias」同时涵盖 M_i,b_i 与 G,g | IMPL_DOSSIER §2.2 初始化段 | 本仓按字面实现 → f(x)=2x @init；本轮加 `g0` 臂做对照，未改 A0 主口径 |
| 3 | opacity 实现决定 = raw 参数 + clamp[0,1]（论文只说 ∈[0,1]） | IMPL_DOSSIER §2.4 矛盾表第 2 行 | 本轮实测其后果（单向陷门），见 A.3 |
| 4 | torch `clamp` 反向在 `x<min` 或 `x>max` 处梯度为 0，边界值本身梯度为 1 | 本地实测（`torch.tensor([1.0001]).clamp(0,1).backward()` → grad 0） | opacity 越界后永久冻结，成立 |
| 5 | GLUT 300 个 CC cube 中 7-LUT 子集取自 NILUT | IMPL_DOSSIER §2.2 数据构造段 | 用本地唯一公开 NILUT cube（LUT01）作语料平滑度参照点 |
| 6 | 本仓 A0 语料 75 个全部来自本地生产 preset（e18 62 个 / quandian 13 个），无任何 CC 电影模拟 LUT | `cut -f1 config/a0_luts_75.txt` 实查 | 语料口径差异（假设 c）有事实基础 |

**未做在线检索**：本轮所有结论均来自本地实测 + 已核实的 IMPL_DOSSIER 引用节，没有引入新的外部 URL/数字，因此无新的编造风险面。GLUT 论文原文本身仍不可得（官方仓库空壳，§2.1），λ_hc 的单位口径无法从原始来源核实——这一条记在 A.4 待决策。

## A.2 本轮新增代码

| 文件 | 作用 |
|---|---|
| `model/glut_repro/ablate_a0.py` | 逐项隔离训练器（配对臂 + 逐项 loss/梯度范数/alive_frac 诊断）；`l_hc_stable`（chroma 下限）与 `SigmoidOpacityGLUT`（有界 opacity）也在此 |
| `model/glut_repro/run_e1_on_a0.py` | 把 E1 过拟合引擎接到 A0 语料/GT/留出色上（假设 a 的对照） |
| `experiments/.../analysis/probe_lhc_grad.py` | L_hc 梯度尺度与奇点的数值证据 |
| `experiments/.../analysis/corpus_difficulty.py` | 逐 LUT 难度特征 + 与实测 PSNR 的回归（假设 c） |
| `experiments/.../analysis/make_viz.py` | 成功/失败 + full 臂塌陷可视化 |
| `experiments/.../analysis/aggregate.py` | 顶层 metrics.json 汇总 |

**未改动 A0 主口径**：`model.py` / `losses.py` / `train_a0.py` / `run_a0.py` 一行未动，`runs/rec`、`runs/full` 的数字仍是原始复现结果。所有修复都在 ablation 臂里做对照，等主 agent 拍板后再决定是否回写主实现。

## A.3 假设与决策（本轮）

1. **ablation 用 9 个分层 LUT 而非全 75**：按 rec 臂 PSNR 排序取 9 分位点（config/ablate_luts_9.txt），跨 36.2–54.7 dB 全区间；同种子配对，逐项差值比绝对值可靠。全 75 复跑留给主 agent 拍板后的定案轮。
2. **评测口径统一为 2^21 留出色子样**（种子 12345），与 E1 一致；`runs/rec|full` 的原始数字是全量 14.68M 留出色。两者在 rec 臂上相差 <0.05 dB（子样是均匀随机），但**跨表比较时以同口径列为准**。
3. **`l_hc_stable` 的 chroma 下限取 1.0 Lab 单位**（≈1 JND）：低于该值的色本来就无可辨识色相，权重设 0 而非发散。这是我方设计，不是论文规格。
4. **λ_hc 校准值 0.006** 由 `analysis/lhc_grad_probe.json` 的梯度比反推（收敛态 10·L_hc_stable / L_rec = 184×，取 11% 目标）。这是「让附加项成为扰动而非目标」的工程判据，不是论文数字。
5. **sigmoid opacity init logit = 4.0**（o=0.982），最接近论文的 1.0 又保留梯度；符合本项目红线「σ 参数化禁裸 exp（有界 sigmoid）」的同类精神。

## A.4 待主 agent 决策（本轮新增）

1. **λ_hc 的单位口径无法核实**：论文 L=L_rec+10·L_hc，但我方 L_rec 在 RGB[0,1]、L_hc 在绝对 CIELab（a,b ~ O(100)），实测梯度比 229×（收敛态）。可能的还原：(i) 论文 L_rec 在 0–255 尺度（则比值 0.9，恰好平衡）；(ii) 论文 Lab 归一化到 [0,1]；(iii) λ 就是 10 但他们的实现有别的归一化。**三种都合理且都无法从原始来源确认**（GLUT 仓库空壳）。保守默认已采用「梯度校准 λ」并在 ablation 里给了 λ=1 与 λ=0.006 两个对照点，是否把某一档写回主实现请主 agent 拍板。
2. **A0 是否换语料**：假设 c 成立（见 REPORT §4.3）。选项：(a) 保持本地 75 个生产 preset，把锚点判据从「45.47±0.3」改成「同语料下 rec 臂 vs E1 引擎一致 + 容量曲线符合论文 N 标度」；(b) 申请/购买 NILUT Kaggle 全集重建更接近 GLUT 的 75 个。**（a）成本为 0 且已有全部数据，（b）才能真正对 45.47**。本轮按 (a) 给出替代判据建议，最终取舍请主 agent 拍板。
3. **opacity 参数化是否回写主实现**：`opac_sig` 臂的实测收益见 REPORT §3.4。若收益显著，建议在 RD 臂统一采用有界 sigmoid，A0 复现臂保留 raw+clamp 以维持「照抄论文」的语义。
4. **自然图协议仍未跑**（沿用上一棒的 §三.3）：本轮 viz 用了 3 张 S 源自然图做定性对比，但没有 100 张的定量表。
