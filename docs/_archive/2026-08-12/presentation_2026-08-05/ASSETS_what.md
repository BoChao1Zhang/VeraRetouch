# 配图素材清单 · What / 渲染器线（P18–P23）

> **用法**：每页给「已有素材（绝对路径，已 `ls` 核过并逐张看图确认内容）」「需生成（写清怎么生成）」「需自绘（写清画什么）」。
> **本文不生成任何图。**
> 配套的数字与出处见同目录 `MODELCARDS_what.md`。

## 总览

| 页 | 现成可直接用 | 需生成 | 需自绘 |
|---|---:|---:|---:|
| P18 G2 · 3D 不够 | **3** | 0 | 0 |
| P19 RD-ORACLE 容量阶梯 | **1**（有两处必须加注） | **2** | 0 |
| P20 RD-G 生成器横评 | **2** | **3** | **1** |
| P21 颜色崩塌 | **1** | **1** | 0 |
| P22 三阶段分离训练 | 0 | 0 | **1** |
| P23 实验矩阵 | 0 | **1**（纯表，可直接排版） | 0 |
| **合计（P18–P23）** | **7** | **7** | **2** |
| *附录 A · 备询素材（E1/E1b/A0，不上主页面）* | *6* | *0* | *0* |

---

# P18 · G2：3D 颜色算子为什么不够

**这一页要讲的一句话**：真实局部编辑里，**同一个 RGB 值在不同区域需要不同的输出**，所以 3D LUT（只按颜色寻址）原理上够不着；自家真实数据上这块「够不着的量」是 **8.04 dB**，而真·全局编辑的对照只有 **0.04 dB**。

## ① 已有 · 主图（镜像对，一页两张，左右并排）

**这是全套素材里最强的一张页面**，因为两张图是**同题材、相反结局**，直接把「Δ_ceil 量的不是题材、不是编辑幅度、不是掩膜大小，而是掩膜内外的输入颜色分布是否可分」讲透。

| | 路径 | 内容（已逐张看过） |
|---|---|---|
| **左：3D 够不着** | `/home/bc/VeraRetouch/experiments/G2_oracle_ceiling_20260803/viz/strat_l5l6/success_0_5395229490.png` | 六联图。前景向日葵黄→奶白，**背景整片花田是同一种黄**。第 5 格「3D 天花板残差×5」**铺满整个花田与天空**，第 6 格「4D 天花板残差×5」几乎全黑。标题栏印着 `Δ_ceil(arm)=16.47 dB / Δ_nested=15.23 / Δ_donor=0.55 / PSNR 3D=22.76→4D_arm=39.23 / s̄=0.177` |
| **右：3D 就够了** | `/home/bc/VeraRetouch/experiments/G2_oracle_ceiling_20260803/viz/failure_1_6a466726da.png` | 同为向日葵，黄→粉，但**花长在纯蓝天上、在 RGB 空间自成一簇**，3D LUT 靠颜色就能精确寻址。第 5、6 格残差**几乎一模一样**。标题栏 `Δ_ceil(arm)=0.19 dB / PSNR 3D=37.66→4D_arm=37.85 / s̄=0.214` |

**六联图列序（两张相同，讲图时直接念）**：`I_in | I_tar | s = C_GT | |I_tar − I_in|×5 | 3D 天花板残差×5 | 4D 天花板残差×5`（`experiments/G2_oracle_ceiling_20260803/REPORT.md:586`）。

**讲图建议**：只讲第 5 格和第 6 格的差别。左图「3D 解释不掉的那部分恰好长在掩膜里」，右图「3D 解释掉了，4D 无处可用」。这同时**主动交代了方法边界**（`REPORT.md:609` 把这类归为「掩膜与颜色簇重合」，明写「属方法边界，必须在论文里主动报告，否则会被审稿人当作 cherry-picking 打掉」）。

## ② 已有 · 分布图（真实档 vs 全局对照档）

**路径**：`/home/bc/VeraRetouch/experiments/G2_oracle_ceiling_20260803/viz/hist_delta_arm_three_tracks.png`

三联直方图，横轴 Δ_ceil (delta_arm) dB，纵轴 images，每格带**红色中位线**与**绿色虚线判据线**：

| 子图 | 内容 |
|---|---|
| 左 | D-CONSTRUCT(S-val) n=192，median **9.03 dB**，判据线 8 dB。形状是「L0/L5 两个零点对照堆在 0，其余散在 3–50」 |
| 中 | D-SFT-L(S-val,normal) **真实档** n=600，median **7.97 dB**，判据线 1 dB。单峰、分布在 0–17.5，**几乎整条分布都在判据线右边** |
| 右 | D-SFT-G(S-val) **对照档** n=600，median **0.04 dB**，判据线 0。**一根极窄的针，全部落在 ±0.5 dB 内** |

**⚠ 两处必须加注（否则口径对不上）**：
1. 这张图画的是**原批次**（真实档 l1–l4，中位 **7.97**）。汇报口径按 REPORT 自己的引用纪律用的是**补批 8.04 dB**（6 build 全覆盖、等权，`REPORT.md:371-373`）。**建议 PPT 上写「7.97（图中，原批次）/ 8.04（补批，正式引用）」**，或在图注里一句带过。
2. 同目录还有 `viz/strat_l5l6/hist_delta_arm_three_tracks.png`，但它中间那格**只画了新覆盖的 l5/l6 两个 build（n=200，median 7.99）**，不是补批全量 600。**它不能替代上图。**「补批 600 组的分布图」在交付里**不存在**。

## ③ 已有 · 备用（若 mentor 追问「这不是解析上界虚高吗」）

`/home/bc/VeraRetouch/experiments/G2_oracle_ceiling_20260803/viz/strat_l5l6/success_1_02df3d94fc.png`
—— 低调黑白感人像，输入颜色几乎全贴在灰轴上，带内带外像素颜色分布**高度重叠**；**s̄ = 0.705（大掩膜）却拿到 15.80 dB**，是「Δ_ceil 与掩膜面积基本无关（Spearman 0.093）」的直观反例。

> **不需要为 P18 生成任何新图。** 若时间允许，L7（边界横切同色区，Δ_arm 19.21 dB）是最锋利的一条论据，但**交付里没有 L7 的案例图**，只有 §七 阶梯表里的一行数字——建议**口头讲，不配图**。

---

# P19 · RD-ORACLE：oracle s 下的容量阶梯

**这一页要讲的一句话**：给一根干净的 oracle s，**渲染器这一端容量足够**（L1–L4 相对同 N 3D 高出 **+20.6 ~ +23.5 dB**）；但有**两个缺口**——**L6 单标量 s 到顶**、**L0 全局退化未过门**。

## ① 已有 · 主图（阶梯曲线）

**路径**：`/home/bc/VeraRetouch/experiments/RD_std_e_20260803/viz/ladder_in_mask.png`

已逐格看过。标题 `in-mask PSNR (S-val, fixed)`，x = L0…L7 分类轴，y = 20–60 dB。**9 条线**：7 条彩色臂（`g3d`/`lut3d` 两条对照 + `gstd`/`ga`/`gc3`/`gd`/`lut4d` 五条 s 臂）+ **黑虚线 identity** + **灰点线 measured ceiling**。

**能不能看出两个缺口？**

| 缺口 | 在这张图上 | 判断 |
|---|---|---|
| **L6 单轴不足** | **能，而且很直观**。L6 处 s 臂（~27 dB）与 3D 对照（~22.5 dB）的间距只有 ~5 dB，**而 L1–L4 是 ~20 dB**；更关键的是 **5 条 s 臂在 L6 完全重叠成一条线**——参数量从 780 到 73,695（**94.5×**），in-mask 只在 **27.01–27.53（0.52 dB）**内浮动。这就是「加容量不管用」的视觉证据 | ✅ 直接可用 |
| **L0 全局退化** | **不能**。y 轴跨 40 dB，L0 处七条线全挤在 55–59，RD-STD 的退化幅度在这个尺度上**不可读** | ❌ 需要补一张 |

**⚠ 这张图必须加的三条注**（不加就是口径错误）：

1. **L0 数据点是 8000 步的旧数字**（生成脚本 `analyze_rd.py:230` 只读 `runs/`，不读 `runs_l0long/`）。图上 RD-STD 画的是 **−2.02 dB**，而**定稿结论是 30000 步的 −0.93 dB**。
2. 图上**没有** `gc1`(K=1) / `gc5`(K=5) 两个臂（`analyze_rd.py:228` 的臂列表不含它们），也**不含** `mixed` 与 `full` 两个侧网格。
3. 图上只有 `fixed` × `c32` 档（`analyze_rd.py:230` 的过滤条件）。

## ② 需生成 · L0 全局退化放大图（补上第二个缺口）

**为什么必须生成**：现有三张阶梯图**没有任何一张画 L0 收敛复核**；30000 步的定稿结论只以表格形式存在于 `experiments/RD_std_e_20260803/REPORT.md:238-246`。

**怎么生成**：横向条形图或点图，x = 相对同族 3D 对照的 dB 差，**画一条 0 线 + 一条 −0.05 过线 + 一条 −0.50 死线**。数据（全部来自 `experiments/RD_std_e_20260803/runs_l0long/fixed_c32_L0_<arm>_s0/metrics.json :: metrics.psnr_full`，差值对同族 3D 对照现算）：

| 臂 | 参数 | 8000 步 vs 3D | **30000 步 vs 3D（定稿）** | 判定 |
|---|---:|---:|---:|---|
| **RD-STD** `gstd` | 2,530 | −2.02 | **−0.93** | **DEAD**（未过 −0.50 死线） |
| RD-D `gd` | 780 | −2.68 | **−0.76** | **DEAD** |
| RD-A `ga` | 2,575 | −1.47 | **−0.20** | MARGINAL |
| **RD-E** `lut4d` | 73,695 | −0.03 | **−0.02** | **PASS** |
| **RD-C K=3** `gc3` | 1,868 | +0.49 | **+1.82** | **PASS（反超 3D）** |

**这张图要读出的结论**（写进图注）：**代价来自「s 进寻址」，不是来自「有 s 轴」**——RD-E（四线性沿 s 代数退化）与 RD-C（几何 s-free、s 只进载荷）都不付代价，而三个把 s 放进高斯寻址的臂全部付（`REPORT.md:315-318`）。

## ③ 需生成 ·（可选）L6 的「加容量不管用」条形图

现有折线图在 L6 处五条线叠成一条，**看得出重叠但读不出参数量跨度**。若要把这一刀讲死，建议单独一张条形图：

x = 五个 s 臂按参数量升序（`gd` 780 → `gc3` 1,868 → `gstd` 2,530 → `ga` 2,575 → `lut4d` 73,695），y = L6 in-mask PSNR，**再叠一条 +8 dB 判据线（相对同 N 3D 的 22.34 dB，即 30.34 dB）**。数据（`experiments/RD_std_e_20260803/metrics.json :: rows[50..56]`）：

| 臂 | 参数 | L6 in-mask | vs 同族 3D |
|---|---:|---:|---:|
| `gd` RD-D | 780 | 27.37 | +5.04 |
| `gc3` RD-C K=3 | 1,868 | **27.53（最好）** | +5.19 |
| `gstd` RD-STD | 2,530 | 27.31 | +4.97 |
| `ga` RD-A | 2,575 | 27.32 | +4.98 |
| `lut4d` RD-E | **73,695** | **27.01（最差）** | +4.48 |
| *3D-G 对照* | 716 | 22.34 | — |
| *3D-LUT 对照* | 14,739 | 22.53 | — |

**一句话图注**：参数量放大 **94 倍**、机制从高斯换成 4D LUT，in-mask 只在 **0.52 dB** 内浮动，且全部卡在 +5 dB 一线（门 +8）——**瓶颈不是渲染器容量，是单个标量 s 无法区分两张重叠掩膜**。

## ④ 已有 · 备用案例面板（若 mentor 要看图像）

`/home/bc/VeraRetouch/experiments/RD_std_e_20260803/viz/` 下 24 张 `success_*` + 24 张 `failure_*`。

- 命名：`{success|failure}_{level}_{arm}_{uid}.png`，`uid` 形如 `L1_val_0000`（所以级别在文件名里出现两次，如 `failure_L1_ga_L1_val_0000.png`）。
- 覆盖：**只有 4 个臂**（`gstd`/`lut4d`/`ga`/`gd`）× **只有 3 个级**（L1/L4/L7）× 各 2 张。**没有 L6 的案例图**——这是 P19 想配图时的一个硬缺口。
- **是五联图，不是六联**（`analyze_rd.py:194`）：`输入 | 预测({arm}) | GT | oracle s（viridis, 固定 0–1） | |err|（magma, 固定 0–0.15, 带 colorbar）`。
- 选样是**排名不是眼选**：按「in-mask PSNR 相对恒等基线的增益」升序，最低 2 张 = failure、最高 2 张 = success（`analyze_rd.py:187-192`）。

## ⑤ 不要用的图

`viz/ladder_delta_const.png` 与 `viz/ladder_delta_shuffle.png` 是红线负控制图，**对 mentor 的结构追问没有信息量**，且同样带 L0 旧数字问题。留作备询即可。

---

# P20 · RD-G：生成器 MLP vs transformer（六臂）

**这一页要讲的一句话**：**同一个渲染核心（1116 个数，一个都没多）、同一份数据、同一套配方**下，把「条件 → 渲染器参数」这段映射从 CGLUT 式小 MLP 换成 transformer，收敛后 ΔE00 p50 **9.06 → 2.67（−70.5%）**；而**把 MLP 加宽到同参数量只到 7.12**——**买到的是结构，不是参数量**。

## ① 需自绘 · 六臂结构对比图（这一页的主图，mentor 会盯着看）

**必须自绘**，交付里没有任何结构图。建议一页横向排布，**左边一根共享的竖轴，右边六个并列的方块**：

```
        ┌──────────────────────────────────────────┐
条件    │  (I_in, after)  128×128×6                │
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
共享    │ PairTokenizer  220,736 参数（六臂逐位相同）│  ❄ 不是变量
        │ Conv 4×4 → 2×2 → 2×2（总 16×）           │
        │ ⇒ 8×8 = 64 patch token (d=256) + style(128)│
        └────────────────────┬─────────────────────┘
                             ▼
   ★ 唯一变量：生成器 ★   ← 六个并列方块，见下
                             ▼
        ┌──────────────────────────────────────────┐
共享    │ 零初始化输出头 ParamHead（单 Linear，全零）│
        │ mu=anchor+⅓tanh · σ=0.02+0.48·sigmoid    │
        │ M=I+0.1z · o=σ(z−2) · g=σ(z+4) · G=0.1z  │
        └────────────────────┬─────────────────────┘
                             ▼
        ┌──────────────────────────────────────────┐
共享    │ 渲染器 GLUT N=48：23N+12 = 1116 个数      │  ❄ 六臂完全相同
        │ = 「头重脚轻」的那只脚                    │
        └──────────────────────────────────────────┘
```

**中间六个方块的画法**（方块宽度按参数量对数比例，一眼看出容量阶梯）：

| 方块 | 标题 | 块内画什么 | 参数 |
|---|---|---|---|
| 1 | `mlp` **CGLUT 基线** | 一条竖直链：`Linear(256→64)` → 3×`Linear(128)` → 四个并列 head（mu/col/chol/op）+ glob head。**用灰色，视觉上明显最小** | **251,292** |
| 2 | `gtiny` | 2 个 decoder block，**两层都带 cross**，d=192，6 heads | **1,561,445** |
| 3 | **`glite` G-Lite** | 4 个 block，**第 1、3 层带 cross**（另两层只有 self+FFN），d=256，8 heads | **4,447,846** |
| 4 | `mlp_wide` **容量对齐对照** | **与方块 1 完全相同的形状，只是每个 Linear 加宽到 880**。**用与方块 1 相同的灰色**，并用一条虚线框把它和方块 3 圈在一起标注「同参数量」 | **4,933,244** |
| 5 | **`gbase` G-Base** | 6 个 block，**第 1、3、5 层带 cross**，d=384，8 heads | **14,056,806** |
| 6 | `gbase_free` | 与方块 5 **同一个方块**，只在输出头那一层打红叉：`mu 自由 / σ 裸 exp / G = I` | 同上（**参数量逐位相同**） |

**transformer 方块内部要画出来的五个要件**（画在方块 3 或 5 的放大插图里就够，不用六个都画）：
1. **N + 1 = 49 个 query**（48 个 primitive query + 1 个 global query），画成一列小方格，最后一格用不同颜色；
2. **K/V = 64 patch token + 4 个 register token**，register 用不同颜色标出并注 `Darcet et al. arXiv:2309.16588`；
3. **learnable Fourier PE 只加在 key 上**（画一个 `⊕` 在 K/V 那一侧，**不在 query 侧**）；
4. **ModLN 由 style code 驱动，每子层独立**（从 style code 拉三根箭头到 n1/n2/n3）；
5. **energy routing**：从第 {3,4,6} 层各引一条线到一个 `softmax` 融合节点，并注「每 tap 另有 aux head，带熵下界」。

**这张图上必须显式写死的两个数**（mentor 的第一问和最后一问）：
- 顶部：「六臂**共享** tokenizer 220,736 参数」
- 底部：「六臂**共享** 渲染 payload **1116**（= 48×23 + 12），**一个数都没多**」

## ② 需生成 · 收敛后 ΔE00 主图

**为什么必须生成**：交付里**没有任何一张 RD-G 的曲线/柱状图**，`viz/` 全是逐样本六联面板。

**建议画两联（一页）**：

**左：训练轨迹折线**（x = step 1000…8000，y = val_img ΔE00 p50，六条线）。数据来自 `experiments/RDG_transformer_20260803/runs/<arm>/history.json`，已逐 step 读出：

| step | mlp | mlp_wide | gtiny | glite | gbase | gbase_free |
|---|---|---|---|---|---|---|
| 1000 | 11.303 | 11.043 | 9.440 | 8.807 | 8.452 | 11.969 |
| 2000 | 10.548 | 10.280 | 7.471 | 7.110 | 6.075 | 8.984 |
| 3000 | 10.014 | 9.084 | 6.576 | 5.482 | 4.636 | 6.354 |
| 4000 | 9.579 | 8.174 | **5.968（止）** | 4.676 | 3.882 | 5.131 |
| 5000 | 9.397 | 7.823 | — | 4.232 | 3.295 | 4.274 |
| 6000 | 9.102 | 7.356 | — | 3.896 | 2.906 | 3.434 |
| 7000 | 9.074 | 7.146 | — | 3.695 | 2.709 | 3.176 |
| **8000** | **9.057** | **7.116** | — | **3.610** | **2.670** | **3.062** |

**这张折线的读法（写进图注）**：MLP 族在 step 6000 后已走平（9.102→9.057，四步只降 0.5%），transformer 族**还在降**（gbase 2.906→2.670，降 8.1%）。**这条比任何门槛都硬**，也直接回答「是不是只是大模型收敛快」。
**两条必须标的注**：`gtiny` 在 step 4000 停（腾显存给邻居臂，不降 batch 换空间以免混口径）；`gbase_free` 是无加固对照，不是容量点。

**右：收敛后柱状图**（x = 生成器参数量对数轴，y = val_img ΔE00 p50 或 PSNR，**MLP 族与 transformer 族用两种颜色**，并把 `glite` 与 `mlp_wide` 用一条虚线连起来标 **「同参数量，差 5.556 dB / ΔE00 −49.3%」**）。数据（`experiments/RDG_transformer_20260803/runs/<arm>/metrics.json`）：

| 臂 | 生成器参数 | val_img p50 | PSNR | 天花板占比(dB) |
|---|---:|---:|---:|---:|
| `mlp` | 251,292 | 9.057 | 19.563 | 20.0% |
| `gtiny`(4000) | 1,561,445 | 5.968 | 22.868 | 31.0% |
| `mlp_wide` | 4,933,244 | 7.116 | 21.720 | 27.2% |
| `glite` | 4,447,846 | 3.610 | 27.276 | 45.6% |
| `gbase` | 14,056,806 | **2.670** | **30.024** | **54.8%** |
| *天花板* | — | 0.519 | 43.650 | 100% |
| *恒等* | 0 | 17.847 | 13.530 | 0% |

**建议在柱状图上加一条「天花板 43.65 dB」的横线**，把「G-Base 只走完 54.8%、还有 45.2% 没取」画出来——这是「该继续加大」的正面证据，也是「MLP 已达天花板 90% → 停 RD-G」这条死刑条款未触发的可视化。

## ③ 需生成 · 33³ 烘焙回读误差图

**为什么必须生成**：`bake_check_converged.json` 有数字但没有图；而「烘焙一致性从第一天当一等指标」是战役红线，mentor 若问「交付物真的是标准 3D LUT 吗」，这是唯一的答案。

**怎么画**：对数纵轴的分组柱状图，x = 四个臂（`mlp`/`mlp_wide`/`glite`/`gbase`），每组三根（17³ / 33³ / 65³ 的 `vs 直接 p50`），**再画一条 PLAN 的 `ΔE < 2` 门线**（会高出数据两个半数量级，正是重点）。数据（`experiments/RDG_transformer_20260803/bake_check_converged.json`，n=256，全部 step 8000）：

| 臂 | 直接 ΔE00 p50 | 17³ | **33³ p50** | 33³ p99 | 33³ max | 65³ | 33³ 净代价 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp` | 9.776 | 0.0840 | **0.0213** | 0.0554 | 0.0655 | 0.0055 | +0.0017 |
| `mlp_wide` | 7.644 | 0.0955 | **0.0249** | 0.0593 | 0.0646 | 0.0063 | +0.0008 |
| `glite` | 3.907 | 0.1141 | **0.0296** | 0.0669 | 0.0683 | 0.0075 | **−0.0060** |
| `gbase` | 2.943 | 0.1204 | **0.0312** | 0.0757 | 0.0835 | 0.0079 | +0.0003 |

**图注**：33³ 导出 + 四面体回读的代价 p50 ≤ 0.031、max ≤ 0.084，**净代价 ≈ 0**（G-Lite 甚至微负）。**⚠ 交付里没有 `gtiny` 与 `gbase_free` 的收敛后烘焙数字（「缺」），图上只画四臂并注明。**

## ④ 已有 · 逐样本面板（备询用，不必上主页面）

`/home/bc/VeraRetouch/experiments/RDG_transformer_20260803/viz/` 共 51 张 png + 4 个 `manifest_*.json`（`gbase`/`glite`/`mlp`/`mlp_wide` 各一套；**`gtiny`/`gbase_free` 没有**）。

- **六联图列序**（已看图确认）：`I_in | after=GT | f_θ(I_in) | dE00 map (image) | GT cube slice b=0.5 | predicted cube slice b=0.5`。**最后两格是 cube 切片，这是本套面板的价值所在**——直接看生成器吐出的 LUT 长什么样。
- 命名：`{success|failure}_{arm}_{pool}_{row}.png`，`pool ∈ {val_img, val_lut}`。
- 选样是**排名不是眼选**（`tools/make_viz.py::rank_pool`，按逐样本 cube ΔE00 排序取最好 2 / 最差 2–3）。
- **推荐用于「典型失败」的一张**：`/home/bc/VeraRetouch/experiments/RDG_transformer_20260803/viz/failure_gbase_val_lut_326997.png`（`cube dE00 20.981`，preset `quandian__quandian_003358`）。已看图：GT cube 切片是奶白暖调，预测切片明显**偏青绿**——即「未见 LUT 上大幅度 preset 欠拟合 / 色调偏移」，与「泛化间隙随容量放大」是同一件事。
- **对照用**：`/home/bc/VeraRetouch/experiments/RDG_transformer_20260803/viz/success_gbase_val_lut_326098.png`（`cube dE00 0.685`，`bake dE00 0.016`）。

## ⑤ 这一页必须口头说的两句（否则被查到就炸）

1. **本页所有数字来自 `metrics_converged.json` / `runs/*/metrics.json`（step 8000），不是 `REPORT.md`**——REPORT 通篇是 step 2000/4000 的旧口径，项目文档自己也登记了这一点（`docs/EXPERIMENT_RESULTS_CURRENT.md:42-43`）。
2. **Stage-2（有 s 的 4D 档）一个数字都没有**（`runs2/` 目录不存在，调度脚本在 `stage1 done=5/6` 上报错卡住）。**所以「transformer 的优势在有 s 时是否延续」完全未测**。

---

# P21 · 颜色崩塌的证据

**这一页要讲的一句话**：读出侧 12 个臂里，**只要给了指令，颜色幅度就稳定卡在 GT 的 20–32%**；**拿掉指令就塌到 6%**。因果链是干净的（指令确实在起作用），但**幅度只恢复三分之一**——这就是「瓶颈在读出侧不在渲染器」的最直接读数（渲染器那端 RD-G 的方差比已经 0.967）。

## ① 需生成 · var_ratio 柱状图（主图）

**为什么必须生成**：交付里**没有任何 var_ratio 的图**，只有 json 里的标量。

**怎么画**：横向条形图，**y 轴按 var_ratio 降序排列臂名**，x 轴 0→1.0。**画一条 `GT = 1.0` 的参考线**，**用不同颜色区分三组**：① 有指令的主臂/结构臂（蓝）② **因果控制臂（红）** ③ 训练器的 `var_ratio < 0.05` 死刑线（灰色阴影带）。

统一口径：完整 `select` 池 **n = 4,225**，取 `final.select.var_ratio`。

| 臂 | 组 | 变的是什么 | **var_ratio** | json 路径 |
|---|---|---|---:|---|
| *GT 参照* | — | 定义上 | **1.000** | — |
| `condition_short` | 蓝 | 短指令 | 0.3204 | `experiments/MCQ_full_local_l1l6_20260804/runs/condition_short/metrics.json :: final.select.var_ratio` |
| **`config_a`**（anchor，lr 2e-4） | 蓝 | — | **0.3175** | `…/runs/config_a/metrics.json :: final.select.var_ratio` |
| `renderer_lut4d` | 蓝 | 换 17³×5 四线性 context LUT | 0.2775 | `…/runs/renderer_lut4d/…` |
| `config_b`（lr 1e-4） | 蓝 | 学习率 | 0.2753 | `…/runs/config_b/…` |
| `condition_instruction_only` | 蓝 | **图像置零**，只给指令 | 0.2285 | `…/runs/condition_instruction_only/…` |
| `basis_geo_range8_full` | 蓝 | **监督形式改为 8 维全局 basis 系数** | 0.2256 | `experiments/MCQ_basis_where_l1l6_20260804/runs/basis_geo_range8_full/metrics.json :: final.select.var_ratio` |
| `renderer_gaussian3d_alpha` | 蓝 | 3D 高斯 LUT + alpha 合成 | 0.2048 | `experiments/MCQ_full_local_l1l6_20260804/runs/renderer_gaussian3d_alpha/…` |
| `basis_vlm14_full` | 蓝 | 8D 几何 + 6D 语义 basis | 0.2008 | `experiments/MCQ_basis_where_l1l6_20260804/runs/basis_vlm14_full/…` |
| **`condition_fixed_shuffle`** | **红** | **固定的错指令** | **0.0670** | `…/runs/condition_fixed_shuffle/…` |
| **`condition_image_only`** | **红** | **中性指令** | **0.0632** | `…/runs/condition_image_only/…` |

**⚠ 两个臂无数字（写「缺」，不要画）**：`interaction_instruction_bilinear` 与 `spatial_metaquery8` 只有 `metrics_partial.json`，没有 `metrics.json`。

**图注要写的三条**：
1. **指令是颜色幅度的唯一来源**：拿掉指令 0.32 → **0.063**，同时 `delta_shuffle_db` 变成 0.000 / −0.0002（构造上无指令可 shuffle）。
2. **但给足指令也只有 0.32**：**听了，但不敢改**。
3. **换渲染器、换监督形式、换学习率都救不回来**：10 个有指令的臂全部落在 **0.20–0.32** 的窄带。

**⚠ 一个必须防的误读**：**这个 `var_ratio` 与 RD-G 的 `var_ratio` 不是同一个量**，两个 0.317 / 0.967 **绝对不能并排画在同一张图里**。定义差别见 `MODELCARDS_what.md` §5.1（MCQ = 每图编辑增量的空间方差之比；RD-G = 输出色在样本间的方差之比）。

## ② 已有 · 视觉佐证（强烈建议配在柱状图旁）

**路径**：`/home/bc/VeraRetouch/experiments/MCQ_full_local_l1l6_20260804/viz/config_ab_compare/compare20_p01.png`

已看图。5 行样本 × 7 列：`输入 | 目标 | A 最终渲染 | B 最终渲染 | GT 区域 | A 预测区域 | B 预测区域`。

**它把 0.317 这个数字直接画成了图**：第 1 行目标是**很强的蓝色调**，A/B 渲染只是**淡淡一层**；第 3 行目标是明显的暖调降饱和，A/B 几乎**没动**。同时右边三列显示**区域预测其实相当准**（GT 区域与 A/B 预测区域形状对得上）——**「位置找得到，颜色改不动」一图说完**。

同目录另有 `compare20_overview.png`（20 张总览）与 `compare20_p02/p03/p04.png`。**用 p01 就够**。

## ③ 备用数字（若 mentor 追问「那 delta_shuffle 呢」）

`config_a` 的 `delta_shuffle_db_p50 = 1.901 dB`，**预注册的因果门是 3 dB，所有臂全部未过**（训练器对 <3 dB 的臂加 `10·(3−x)` 惩罚，`train.py:267-269`）。项目自己的说法：「只能说 A 的 instruction 响应更强，**不能据此宣称已经充分听懂指令颜色**」（`experiments/MCQ_full_local_l1l6_20260804/AB_VISUALIZATION.md:29`）。

---

# P22 · 下一轮：三阶段分离训练（**需自绘**）

**这一页要讲的一句话**：既然「渲染器容量够、瓶颈在读出侧」，下一轮就**把 Where 与 What 彻底拆开、分三阶段冻结着训**，用来**直接检验此前的颜色崩塌是不是来自空间 query 与颜色 query 的梯度冲突**（这句是协议原文的动机，`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md:146`）。

## 需自绘 · 三阶段流程图

交付里**没有任何示意图**，但协议 §0 与 §7.3 各给了一段可直接转框图的 ASCII。建议**一页四行，从上到下**，每行一个阶段，**左侧统一用 ❄（冻结）/ 🔥（训练）图标**。

### 第 1 行：Base SFT

```
❄ 24 个 Vision blocks
🔥 主 merger + 3 个 deepstack mergers + Language + 4 个 special tokens
输入 I_in + instruction  ──▶  输出 <where>…</where><color>…</color>
```
- 基模 **Qwen3-VL-4B-Instruct**；trainable **4,131,573,248** / total **4,437,815,808**（冻结 306,242,560 = 24 个 vision blocks + patch embed，trainable 占 **93.1%**）
- **不用 LoRA**（协议明写不用任何 PEFT）
- 出处：协议 `:141`；参数量 `experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/model/preflight_model.json :: 252-255`

### 第 2 行：Stage-Where（A 校准 + B MetaCanvas）

```
❄ 整个 SFT VLM
🔥 Where-A：shared basis projector B (1024→64)          — 只有 65,536 参数
🔥 Where-B：Q_where + Where connector + readout/output heads — 34.8M ~ 69.9M
输出 全局 basis 参数 (w, ρ) ─▶ s(p)=Φ(I,p)·w ─▶ m(p)=R(s(p); ρ)
```
- Where connector：**宽 512、6 个 pre-norm block、8 heads、FFN 2048**
- **高分辨率边缘来自 `F_pre` 的 H/16×W/16 网格 + 原图 guided upsample，不是来自 query 数**（协议 `:185`）——**这句要画在图上**，否则会被问「8×8 的 canvas 怎么可能出细边界」
- **只对标量 `s_low` 做一次 edge-aware guided upsample**（禁止先上采样 64 通道再组合）

### 第 3 行：Stage-What

```
❄ 整个 SFT VLM + 完整 Where checkpoint
🔥 独立 Q_color（16 个）+ style connector + 48-slot Transformer backend + 48 个 heads
   （SB48 另训练跨样本共享的 geometry）
输出 连续 z_style ∈ R^1024 ─▶ 48-Gaussian LUT 函数 T_pred ─▶ 标准 33³ LUT
```
Backend 内部（画成第 3 行的放大插图，协议 `:418-425` 的原文流程）：
```
48 learned slot embeddings + WC tokens
  ─▶ 2 个 seed Transformer block
  ─▶ provisional / fixed Gaussian geometry
  ─▶ Gaussian-aligned pooling v_i        ← 梯度可穿回 geometry
  ─▶ 4 个 refinement Transformer block
  ─▶ 48 个独立 decoder head + 1 个 global head
```
每 block：**宽 512、8 heads、FFN 2048、pre-norm**，共 **2+4 = 6 层**。

### 第 4 行：最终合成

```
I_out(p) = I_in(p) + m_pred(p) · ( T_pred(I_in(p)) − I_in(p) )
```

### **这张图最重要的部分：三条「断开的箭头」**

用**红色打叉的虚线箭头**画出三条被**故意切断**的梯度路径，并在旁边写一句动机：

| 断开的路径 | 出处 |
|---|---|
| **What ⇏ Where**（What 不向 Where 反传） | 协议 `:146` |
| **Where ⇏ VLM**（Where 不向 VLM 反传） | 协议 `:146` |
| **Q_color ⇏ H_where 且 Q_where ⇏ H_color**（两套 query bank / attention pool / 输出头**完全不共享**） | 协议 `:146, :382` |

图右侧写动机原文：**「这样可以直接检验此前颜色崩塌是否来自空间/颜色 query 的梯度冲突」**（协议 `:146`）。

### 图上还要标的四个数（mentor 必问）

| 标注 | 值 |
|---|---|
| 推理期输入 | **只有 `I_in + instruction`**；`I_tar` / GT mask / GT LUT / oracle latent 一律不进推理 |
| 每样本生成的 LUT 参数 | **FG48 = 48×23 + 12 = 1116**（**与 RD-G 完全相同**）；**SB48 = 48×14 + 12 = 684** |
| 两个生成器参数对齐 | **FG 61,394,716 vs SB 61,406,641，相对差 0.019%**（靠调 head bottleneck 128/137，容差 2%） |
| 零初始化恒等 | 残差参数化实测 max abs err **1.49e-6**；**协议字面公式实测 `T(x)/x = 2.000`，已由 amendment A-1 改为纯残差 `G x`（G 零初始化）** |

> **这最后一条值得单独讲 20 秒**：协议原文写的 `(I + ΔG)x + Σ q_i(M_i x + b_i)` 与归一化权重 `Σq_i ≈ 1` 联立，零初始化时给出 `f(x) = 2x`——正命中战役红线「全局仿射 G 初始化 = 0 不是 I」。这个 bug 是在 preflight 里被实测抓出来的（双方独立复算：`mean T(x)/x = 2.0000` / `1.99999988`），改成纯残差后 max|T−x| 从 >0.9 降到 **1.49e-6**。**这是「红线清单不是洁癖」的现成案例。**

### Where→What 之间实际传的三样东西（若画得下，标在第 2→3 行的箭头上）

```
F_roi   = Σ_p m_pred(p)·F_pre(p) / (Σ_p m_pred(p) + eps)
F_bg    = Σ_p (1−m_pred(p))·F_pre(p) / (Σ_p (1−m_pred(p)) + eps)
z_where = AttentionPool(Q_axis, Q_readout)
```
四种 WC 接口就是这三样的四种组合：`WC-0` 一样都不给、`WC-1` 给 `m_pred/F_roi/F_bg`、`WC-2` 给 `z_where, w, ρ`、`WC-3` 全给。

---

# P23 · 实验矩阵（29 个臂）

**这一页要讲的一句话**：下一轮共 **29 个完整训练臂**，分四段；**每一臂都完整训练，不做短跑筛选、不做中途淘汰**。

## 需生成 · 一张表即可（不需要画图，直接排版）

**⚠ 先纠正一个易错点**：**协议 §8 只有 12 个臂**（8 主 + 4 控制）。**29 这个数字来自协议 §16「实验规模摘要」六行相加**，汇报时**不要说「§8 有 29 个臂」**。

### 表 A · 29 臂的分解（协议 `:829-837`）

| 段 | 臂数 | 选择集 |
|---|---:|---|
| Base SFT | **1** | 原 SFT eval / 语言指标 |
| Where-A basis calibration | **4** | `V_where` oracle 指标 |
| Where-B MetaCanvas | **8** | `V_where` generated-context 指标 |
| **Stage-What 主矩阵** | **8** | `V_what` |
| Stage-What 控制臂 | **4** | `V_what`，**不进主榜** |
| Top-2 稳定性复跑（各 2 个额外 seed） | **4** | `V_what`，配置已冻结 |
| 最终测试 | 0 | `T_final` 与 `T_lut_unseen` **各只打开一次** |
| **合计** | **29** | |

（协议 §11 排期表只排到 **25** 臂 = 1+4+8+8+4；Top-2 的 4 个复跑排在主矩阵之后，29 = 25 + 4。）

### 表 B · Stage-What 12 臂（本线的主体，带实测参数量）

| Arm | Where 接口 | Generator | where_source | 可训练参数 |
|---|---|---|---|---:|
| T01 | WC-0 ColorOnly | FG48 | predicted | 91,364,898 |
| T02 | WC-1 MaskPool | FG48 | predicted | 92,415,522 |
| T03 | WC-2 QueryState | FG48 | predicted | 92,739,106 |
| T04 | WC-3 FullWhere | FG48 | predicted | 93,789,730 |
| T05 | WC-0 | SB48 | predicted | 91,376,823 |
| T06 | WC-1 | SB48 | predicted | 92,427,447 |
| T07 | WC-2 | SB48 | predicted | 92,751,031 |
| T08 | WC-3 | SB48 | predicted | 93,801,655 |
| **C01** | NoWhere | FG48 | **none** | 91,364,898 |
| **C02** | NoWhere | SB48 | **none** | 91,376,823 |
| **C03** | OracleWhere | FG48 | **oracle** | 94,109,730 |
| **C04** | OracleWhere | SB48 | **oracle** | 94,121,655 |

数据源：`experiments/Q3VL_metacanvas_where_what_20260804/what/config/arm_matrix.json`（`n_arms: 12`）。
**四个 WC 接口的含义**：`WC-0` 只给颜色 reasoning + 全局视觉池化 / `WC-1` 加 dense region pooling / `WC-2` 加全局空间 latent / `WC-3` 两者都给。**主报告必须给 WC × generator 的 interaction，不是两个边际排名**（协议 `:509`）。

### 表 C · Where-B 8 臂（若这一页放得下；否则口头带过）

| Arm | MetaCanvas 结构 | queries | Readout | 可训练参数 |
|---|---|---:|---|---:|
| W01 | MC8-Joint (8×8) | 64 | R-Band | 34,838,745 |
| W02 | MC8-Joint | 64 | R-CBand12 | 34,855,161 |
| W03 | MC16-Joint (16×16) | 256 | R-Band | 34,937,433 |
| W04 | MC16-Joint | 256 | R-CBand12 | 34,953,849 |
| W05 | MC16-SplitHead | 256 | R-Band | 36,254,297 |
| W06 | MC16-SplitHead | 256 | R-CBand12 | 36,270,713 |
| W07 | MC16-DualCanvas | 256 | R-Band | **69,835,365** |
| W08 | MC16-DualCanvas | 256 | R-CBand12 | **69,851,781** |

数据源：`experiments/Q3VL_metacanvas_where_what_20260804/where_b/preflight_where_b_cpu.json`。

### 这一页必须主动说的三条（mentor 一定会问）

1. **「29 臂」不在 §8。** §8 只有 12 个（8 主 + 4 控制），29 是 §16 六行相加。
2. **参数量只在 What 的两个生成器之间对齐**（FG48 vs SB48，相对差 0.019%，容差 2%）。**Where-B 的 8 个臂参数量相差近 2 倍**（W01 34.8M vs W07 69.8M），协议也**没有要求**对齐——参数量在 Where 的选择规则里只排**第 4 顺位**。**和外部竞品基线的参数量对齐：没有做**（本轮已按定档取消 MetaQuery baseline）。
3. **这两份文档头部都写着 `DESIGN FROZEN / NOT IMPLEMENTED / NOT STARTED`**——**是方案冻结，不是结果**。已经落盘的只有 CPU preflight（Stage-What 12/12 pass），GPU 侧还有 9 项待办（`WT-G1`…`WT-G9`），且 `T_lut_unseen` 能否成立还是条件式的（若现有 split 不满足，报告只能写 held-out sample，不能写 unseen LUT generalization）。

---

# 附录 A · 备询素材（不上主页面，但 mentor 追问时能立刻调出）

这几张我也逐张看过，内容已核实，**都是现成的**。

| 追问 | 现成图（绝对路径） | 画的是什么 |
|---|---|---|
| **「N=48 是怎么定的？为什么不是 32/64？」** | `/home/bc/VeraRetouch/experiments/E1b_svd_20260803/viz/rank_vs_de00.png` | 单幅 log-log 折线。x = SVD rank（1…512），y = 跨 LUT 的 per-LUT mean ΔE00 分位（p50/p90/p99 三条）。**灰色竖向阴影 = 预注册拐点窗 32–64**，两条水平门线 `p90<1.0` / `p99<2.0`。**读法**：三条曲线在 log-log 上近乎直线，**阴影窗内看不到任何斜率变化（无肘部）**；p90 直到 **r=384** 才压到 1.0 以下。⚠ **图上没有泛化（留出字典）曲线**，读者会照 384 做设计决策——审阅点名要求补，未补 |
| **「能量占比 99% 不就够了吗？」** | `/home/bc/VeraRetouch/experiments/E1b_svd_20260803/viz/singular_spectrum.png` | 双联。左 = 奇异谱 log-log（σ₁≈1.12e4 → σ₂≈1.56e3 一个陡崖，之后长长的近幂律缓降到 index≈2000，最后在 3.5e3 处垂直跌落）；右 = 累计能量（**r=1 就已经 0.946**，r≈11 到 0.99，r≈100 后完全压平）。**这是「能量口径低估感知维度」的图形版** |
| **「低秩失败长什么样？」** | `/home/bc/VeraRetouch/experiments/E1b_svd_20260803/viz/failure_r48.png`（配 `success_r48.png`） | 3×3：`original hald \| recon r=48 \| ΔE00 map`。失败三例（mean ΔE00 9.76/9.48/8.75）分别是**橙-黑硬色阶分离 / 红-黑+品红蓝色块 / 近黑白单色**，重建后浑浊串色，ΔE00 colorbar 飙到 23/28/33。成功三例（0.55–0.57）都是平滑饱和的彩虹渐变。**归因**：极端风格化 look（近单色/双色调/硬色阶）离主子空间最远——与 E1 尾部的「青橙/复古/胶片/暗调」是同一现象 |
| **「你们复现 GLUT 了吗？差多少？」** | `/home/bc/VeraRetouch/experiments/A0_glut_repro_20260803/viz/fig_task2_hypAB.png` | 三联。**中间那格最有用**：x=N（log2 8→128），y=PSNR，**三条近乎平行的曲线**——蓝（本语料 64³ 生产 preset）/ 橙（E1 400 个 33³ 生产 preset）/ 黑（GLUT 论文语料），红点线 = anchor 45.47。**一眼结论：曲线形状（模型类）对，整体电平（语料）差 3–4 dB。** 左格是「换优化引擎也落在同一条对角线上」（否决「配方不足」），右格是「20→40→60 ep 仍在上升未触锚点」 |
| **「那 45.47 到底为什么达不到？」** | `/home/bc/VeraRetouch/experiments/A0_glut_repro_20260803/viz/fig_task2_hypC.png` | 三联散点，共用 y = 实测 rec PSNR，x 分别是 `affine_psnr`(r=+0.81) / `curv_mean`(r=−0.70) / `lip_p999`(r=−0.66)，带 OLS 线、anchor 红线，以及**橙五角星 = NILUT LUT01（GLUT 自述来源家族）的回归位置 + 红十字 = 其实测 49.02 dB**。**核心图**：本语料的难度分布整体把 rec 压在 45.47 之下，而 GLUT 来源家族那个参照点坐在我方语料的平滑极端且实测远超锚点 |
| **「红线（G 初始化 = 0）到底值多少？」** | `/home/bc/VeraRetouch/experiments/A0_glut_repro_20260803/viz/fig_task1_arms.png` | 三联。**中间那格是 L_hc 单位失配的图形证据**：6 条梯度范数曲线（对数轴）**分成两簇**，`hc`/`full` 在 ~1–3 且尖峰密布，其余四条压在 ~0.02–0.03 —— 差 2–3 个数量级，与 229×/869× 的数值一致。左格是 6 个臂的 Δ 柱状图（hc **−10.73**、full **−9.59**、hc_cal **−0.12**、mining **+0.66**）。右格显示**没有 opacity 塌缩**（`frac opacity<0.05` 全为 0） |

**⚠ 两条使用限制**：
- `E1_cube_N_20260803/viz/` 的两张（`success_smoke_e18_000042_N32.png` / `failure_smoke_e18_000089_N8.png`）**是 5-LUT 冒烟产物（02:00），不是全量最差 5% 的可视化**，已被审阅打回。**要用请先重出。**（内容本身很直观：N=32 误差图近黑，N=8 误差图满屏规则竖条带 = 高斯覆盖不足的结构性残差。）
- A0 若要讲「G 初始化 = 0 值 +2.86 dB」这条（`ablate/ablate_20ep.json :: results.g0` = **45.491 dB** vs 同批 rec 42.627），**必须同时说这是 9-LUT 分层子集**（该子集的 rec 基线比全 75 高 0.71 dB），**不能说成「复现达标」**。这个数字**没有进任何 REPORT**。

---

# 附录 B · 整份 PPT 建议主动交代的四处「诚实点」

放在最后一页或口头带过，比被追问出来好：

| # | 诚实点 | 为什么必须说 |
|---|---|---|
| 1 | **RD-G 的 REPORT.md 通篇是 step 2000/4000 的旧口径**，本次全部数字改引 `metrics_converged.json` / `runs/*/metrics.json`（step 8000）。项目文档自己已登记该缺口（`docs/EXPERIMENT_RESULTS_CURRENT.md:42-43`） | 任何人打开 REPORT 都会看到不一样的数 |
| 2 | **RD-G Stage-2（有 s 的 4D 档）一个数字都没有** | 「transformer 的优势在有 s 时是否延续」是本线最大的未知 |
| 3 | **加固对照（`gbase_free`）收敛后从 +2.24 dB 降到 +1.685 dB（未见源）/ +0.319 dB（未见 LUT）**，落在 PLAN 的「0.5–2 dB」灰区。诚实说法：**加固买的是训练稳定性与收敛速度，不是最终上限** | 这是 REPORT / 聚合文件 / 逐臂真值三者唯一不一致处 |
| 4 | **RD-ORACLE 全程只有 PSNR，没有任何 ΔE00**；且 L0 结论有 8000 步（−2.02）与 30000 步（**−0.93，定稿**）两套数，三张阶梯图画的都是旧的 | 任何 RD-ORACLE 的 ΔE00 数字都是错的；L0 图上的数与结论对不上 |
