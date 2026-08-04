# WA-IMPL · Stage-Where-A basis calibration — 实施记录

日期：2026-08-05 ｜ 任务卡：WA-IMPL ｜ 代码：`q3vl/where/` ｜ 测试：`q3vl/where/tests/`
引用：`docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §0/§1/§2.3/§3/§4/§10.2/§14

状态：**实现完成 + CPU 级验证通过；GPU 级 preflight 与正式校准未启动**（两张 H100 全程被 Base SFT 占用，PID 3395226/3395227）。

---

## 一、实施前核实记录（凡是被写进代码的外部事实，都在这里给出核实方式）

| # | 事实 | 核实方式 | 结果 |
|---|---|---|---|
| V1 | `vision_config.hidden_size = 1024`、`patch_size = 16`、`spatial_merge_size = 2`、`depth = 24`、`deepstack_visual_indexes = [5,11,17]` | 直接读 `/home/bc/data/models/Qwen3-VL-4B-Instruct/config.json` | 全部一致，`F_pre ∈ R^[B×H/16×W/16×1024]` 成立 |
| V2 | **merger 前 token 顺序不是行主序**，而是 `(grid_h//2, grid_w//2, 2, 2)` | 读 transformers 4.57.1 `Qwen2VLImageProcessorFast` 的 `permute(0,1,4,7,5,8,3,2,6,9)` + `Qwen3VLVisionModel.fast_pos_embed_interpolate` 的 `view(t,h//m,m,w//m,m,-1).permute(0,1,3,2,4,5)`；再用**真实 processor** 喂一张每个 16×16 patch 编码自身 (row,col) 的图，反查布局 | 一致。`q3vl/where/fpre.py:unshuffle_to_grid` 为其逆；`tests/test_fpre.py::test_unshuffle_matches_the_real_processor` 同时断言"朴素 reshape 会失败"，防止测试本身失效 |
| V3 | Base SFT 冻结 `patch_embed` / `pos_embed` / 24 个 vision blocks | 读 `q3vl/train/freeze.py` 与 `q3vl/train/constants.py:VISUAL_TRAINABLE_PREFIXES`（只含 merger + deepstack mergers） | **推论：`F_pre` 对 Base SFT 完全不变**。这允许在 base 权重上做 CPU 级数值验证，也允许 4 个臂共用同一次视觉前向。已写成 preflight 项 `WA-P4b`，checkpoint 出现后必须实测 |
| V4 | GT mask 来源 | 见下节"二、mask 数据来源结论" | `.cgt.png`，100% 可定位、可读 |
| V5 | `Qwen3VLVisionRotaryEmbedding.inv_freq` 是 **non-persistent buffer** | 读 modeling 源码 `register_buffer(..., persistent=False)` | 因此 `meta` + `to_empty()` 装载路径会把 `inv_freq` 留成未初始化内存（**静默错误的旋转位置**）。已改为在目标 device 上正常构造再 `load_state_dict`，并在返回前断言所有 buffer 有限 |
| V6 | guided upsample 算法出处 | 打开 https://arxiv.org/abs/1505.00996 核实：标题 *Fast Guided Filter*，作者 Kaiming He, Jian Sun，摘要为 O(N)→O(N/s²) 的子采样加速 | 编号与内容一致，可引用；低分辨率估 `(a,b)` → 双线性上采样 → 用原分辨率 guide 施加，即该文的做法 |
| V7 | E2 已验证的 readout 形式与超参 | 读 `experiments/E2_basis_fit_20260803/e2lib.py`（本仓库既有实现） | `cband_norm`、`bandpass`、`σ∈[0.025,0.30]` 有界 sigmoid、`μ` 固定网格为 buffer、`k∈[1,40]`、L=Rec.709 luma / S=HSV saturation、短边归一坐标、lstsq 残差化——全部与本实现对齐 |
| V8 | split 权威表与本地样本量 | 数 `/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits/*.index.jsonl` | train 159,215（local 75,544：normal 42,752 / low 32,792）；V_where 896（local 400：normal 224 / low 176）；V_what 897（local 408）；T_final 918（local 424）；T_lut_unseen 433（local 198） |
| V9 | 训练 env 的 `sqlite3` 坏了 | `/home/bc/envs/q3vl_sft/bin/python -c "import sqlite3"` → `CXXABI_1.3.15 not found` | 解法：`export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib`。已写进 `scripts/run_where_a.sh`。**mask 定位依赖 build 的 `catalog.sqlite3`，不设这个变量整条 Where-A 数据路径起不来** |

---

## 二、mask 数据来源结论（任务卡第 7 项）

**链路**：`sft2seg` record → `image.origin.root`（原 build 的 batch 目录）+ `source_sample_id` → 该 batch 的 `indexes/catalog.sqlite3` 里 `suffix='.cgt.png'` 的成员 → `shards/<shard>.tar` 定位读 + sha256 校验。

**格式**（450 个样本抽样实测，报告见 `mask_validation_V_where.json` / `mask_validation_train.json`）：

> 下表每一个数都可以在归档 JSON 里逐字段查到（审阅 N-9 指出旧版 NOTES 引的是一次更小的
> 预跑、与归档不符；现已全部对齐到 `--limit 200` / `--limit 250` 这两份归档）。

| 项 | V_where（200，仅 normal） | train（250，含 low） |
|---|---|---|
| 定位成功率 | 100.0% | 100.0% |
| 读取+校验成功率 | 100.0% | 100.0% |
| PIL 模式 | 全部 `L`（uint8 单通道） | 全部 `L` |
| 与 EXIF 校正图宽高比一致 | 100.0%（相对误差中位 2.54e-5，最大 3.66e-4） | 100.0%（中位 1.51e-4，最大 3.91e-4） |
| 退化（低分辨率视图 std<1e-4） | 0.0% | 0.0% |
| 软边像素占比（0.04<v<0.96）中位 | 0.2334 | 0.2741 |
| `image.upscaled` 占比 | 15.0% | 10.4% |
| EXIF orientation | 199×1, 1×8 | 249×1, 1×8 |
| `winner_confidence` | normal 200 | normal 148 / low 102 |
| 定位来源 | catalog 100%，jsonl 回退 0 | 同 |

**对齐性验证（关键，格式检查抓不到"看起来对但其实错"的 mask）**：把 `I_in`（源图）与 `I_tar`（`.jpg` 渲染件）都重采样到 mask 分辨率，算 `|I_tar−I_in|` 的逐像素均值 `d`：

- `corr(d, mask)`：V_where 中位 **0.787**（最小 0.313）；train 中位 **0.783**（最小 0.349）
- 编辑能量落在 mask 内的比例 `Σ(d·m)/Σd`：V_where 中位 **0.715**，train 中位 **0.680**

> 独立审阅用**不 import `q3vl.where.*`** 的脚本另抽 48 个样本复算（含逐字节比对 USTAR 头、
> 左右/上下翻转负控制、以及 `I_in` 改用模型真正吃进去的 baked 512 短边图），得 corr 中位
> 0.793 / 0.818，与上表同量级；并额外确认 `members.image` 是 `I_in` 而非 `I_tar`
> （`MAE(baked, I_in)` 比 `MAE(baked, I_tar)` 小 5–40 倍，**无 I_tar 泄漏**）。

相关系数达不到 1 是预期的：LUT 对不同颜色的改动幅度本来就不同，`d` 的幅度受内容调制。这两个数足以判定 `.cgt.png` 就是该样本被编辑的区域掩膜，与 `CLAUDE.md` 数据纪律里"`C_GT` = 逐候选区域掩膜（软边单通道），不是配色真值"完全吻合。

**分辨率链**：`.cgt.png` 是 build 的渲染分辨率（短边 1024，原始宽高比）→ 面积平均到 spec-5 网格 `(out_h, out_w)` 得 `mask_hi` → 再面积平均到 `F_pre` 网格 `(out_h/16, out_w/16)` 得 `mask_low`（拟合就在这一层做）。

---

## 三、CPU 级验证结果

### 3.1 单元测试：**142 个全部通过**（`q3vl/where/tests/`，用时 48 s；初审前 97，一审后 134，复审后 142）

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_readout.py` | 21 | Band/CBand12 公式逐符号对照手算；极端 raw 下 `h>0`、`k∈[1,40]`、`σ∈[0.025,0.30]`、`o,c∈(0,1)`（禁裸 exp 的实证）；μ 网格是固定对称 linspace(−3,3,12)；polarity `π b+(1−π)(1−b)` 互补；**mirror 恒等式 `m(−z;mirror(ρ))≡m(z;ρ)` 到 1e-12**（logsumexp 下重测）；mirror 是对合；批量广播；**新增 B-4**：logsumexp 与 eps 形式在良态区差 <1e-9、**旧式在 σ 下界处的塌陷数字被钉死**、`o` 下溢到 0.0 不产生 NaN、越域仍在 [0,1] |
| `test_basis.py` | 15 | `s=3tanh(q/3)` 与手算一致；`|s|<3`；`‖w_dir‖=1`；`α=softplus>0`；**符号规则**；**canonicalize 不动一个像素**（12 组随机 latent × 2 readout，1e-12）且 `s→−s` 精确；幂等；`w_raw=0` 不被静默翻转；JSON 往返 |
| `test_phi.py` | 15 | P2 = Legendre；坐标按真实宽高比；geo5 列序与行主序；L/S 已知颜色取值；残差化后最大相关 <1e-6；语义块零均值单位方差；梯度流回语义块；常数图不炸；形状校验 |
| `test_upsample.py` | 15 | **多通道抛 `ChannelOrderError`**；常数场被保持；边缘窄于 bilinear；box 均值对照手算；梯度可回传；**新增 N-2**：batch 轴折叠也抛；**新增 B-4**：越域被实测到并在 clamp *前*上报、clamp 不动域内值、域默认 = 生产方不变量 |
| `test_fpre.py` | 5 | `grid=(H/16,W/16)`；unshuffle/shuffle 往返；**unshuffle ≠ 朴素 reshape**；**对着真实 image processor 验布局**（并断言朴素 reshape 会失败） |
| `test_oracle.py` | 17 | 目标函数；两 readout 都能从植入 latent 复原（soft-IoU>0.95）；latent 已规范化；同 seed 确定；不可拟合目标进 rejection；常数 mask 打 flag；起点数（band 12 / cband 6）；fit report 可序列化；**新增 B-1**：`all_starts_failed` 返回 `latent=None`（零向量路径已删）、`usable` 是唯一闸门、`rejection_row` 紧凑且可 JSONL、`evaluate_latent` 出 s 域报告 |
| `test_calibrate.py` | 24 | 四臂表；projector 同 seed 同 B、正交误差 <1e-5；BA-3 一步闭环；BA-1/2 只训自己的 readout；**BA-0 永不训 B**；warmup+cosine；checkpoint 往返；**新增 B-1**：被拒拟合不进 loss/不进梯度、全拒时 `grad_norm is None` 且权重逐位不变、被拒样本不进天花板聚合；**新增 B-2**：`fit_rows` 逐样本、`sample_every` 采健康样本、`rejection_summary`；**新增 B-4**：`evaluate` 出交付分辨率档 + s 域；半供 hi 分支即报错；**新增 D1**：headline=normal 而 low 单独分层；**新增 B-6**：内外层目标一致；**新增 N-11**：百分位口径；**新增 B-3**：按 eligible 步数建表能退火到 0、**用未过滤数则复现"永不退火"** |
| `test_packing.py` | 5 | 成员名符合 `webdataset_basename_v1`；manifest `status=complete`/未压缩/样本数；随机读 + sha256；index 行字段齐全；basis 落盘与 digest；空输入不留半成品（原子发布） |
| `test_maskdata.py` | 11 | 只保留 local l1–l6；**D1 后默认保留 low**（`exclude_low=True` 仍可关）；缺定位符/几何即拒；mask 视图形状与面积平均；软边保持；退化判定；宽高比；**live 定位+校验读 5 个真实 `.cgt.png`**；**新增 N-3**：`MaskViewStore` 往返 + 缺样本干净回退 + 空目录报错 |
| `test_scripts.py` | 7 | **新增（复审）**：`r*(z)` 网格 = 协议 §5.5 的 `linspace(-3,3,257)` 且 `curve_of` 在声明约定下逐元素一致；`cband_normalization` 两处都发布；越域门槛被两个消费方使用且**过滤先于排序**；resolver 抛错时 `prepare` 记 rejection 而非中断；`count_eligible` 自称上界且驱动器条件性 `SystemExit` |
| `test_preflight.py` | 7 | **新增**：边界检查与上采样顺序检查确实 PASS 且证据字段在；缺 checkpoint → skip；**N-5b**：坏 checkpoint → `fail` 而非异常、驱动器异常时 JSON 仍落盘且带 `WA-P0-preflight-driver=fail`；`--skip-model` 列全 8 个检查 id |

### 3.2 mock 端到端闭环（BA-3-Joint，8 步 × batch 4，D2 裁定后内外层同为 soft-IoU）

```
step loss(1-softIoU) grad_norm  lr        fit_loss                 rej  rows
 1   0.246911        1.709822   1.0e-4    band .2369 / cband .2569  0    8
 2   0.231691        2.041931   9.5e-5    band .2374 / cband .2260  0    0
 3   0.216595        1.842472   8.1e-5    band .2353 / cband .1979  0    0
 4   0.230780        2.121190   6.1e-5    band .2290 / cband .2326  0    0
 5   0.218254        2.405507   3.9e-5    band .2127 / cband .2238  0    8
 6   0.220179        1.882475   1.9e-5    band .2416 / cband .1988  0    0
 7   0.219326        1.920263   5.0e-6    band .2383 / cband .2004  0    0
 8   0.234019        1.601298   0.0       band .2425 / cband .2255  0    0
‖ΔB‖_F = 0.0645  (‖B‖_F = 8.0)   全部有限   rejection_summary: 64 fits / 0 rejected
evaluate(4 个带 hi 分支的新 mock)：
  band     low median 0.7583   hi median 0.3818   越域 1.30%  raw s [−4.805, +3.926]
  cband12  low median 0.7603   hi median 0.4027   越域 1.46%  raw s [−2.564, +4.518]
```

外层 loss 现在与内层 fit_loss 同量级（同一个泛函），这正是 B-6 要的一致性。
`rows` 列是落进 `fit_rejections.jsonl` 的行数：非 `ok` 的全落，另外每 4 步采一批健康样本。
**mock 的 hi 档低是 fixture 造成的**：mock 的 `guide_hi` 是纯随机噪声，guided filter 无边可循；
真实数据上的落差只有 0.018（band）/ 0.074（cband12），见 3.3。

### 3.3 真实数据上的 CPU 级 preflight（4 个 V_where 本地样本，float32，seeded-orthogonal B = BA-0 起点）

归档：`preflight_where_a_cpu_dryrun.json`（**下面每个数都能在该文件里查到**）。

```
[PASS] WA-P6a-readout-bounds
[PASS] WA-P4c-upsample-order
[PASS] WA-P4a-fpre-geometry      grid 32x48 <- 512x768，宽高比 1.5 一致
                                 空间相干性  正确 unshuffle 0.01733  vs  朴素 reshape 0.00745
[PASS] WA-P4d-position-encoding  独立复算 pos_embed 双线性插值，最大相对误差 1.85e-7  (新增, N-6)
[PASS] WA-P5-basis-conditioning  残差化后最大相关 7.60e-7；design Gram cond 中位 27.9 / 最大 95.7
                                 phi Gram cond 中位 8.57e4 / 最大 1.18e5；死通道 0
                                 oracle 拟合成功率 band 100% / cband12 100%
                                 headline low soft-IoU (n=2)  band p10 0.840 / p90 0.965
                                                              cband12 p10 0.861 / p90 0.974
                                 吞吐 1.816 s/拟合 (CPU, 8 次) -> 43.1 / 76.2 CPU-小时每臂  (N-15)
[PASS] WA-P4e-highres-path       headline hi soft-IoU (n=2)  band p10 0.822 / p90 0.960
                                                             cband12 p10 0.787 / p90 0.969
                                 low->hi 中位落差  band 0.0177 / cband12 0.0738
                                 s 越域(clamp 前, n=4 样本)  band 中位 0.000 / p90 0.542% / 均值 0.140%
                                                            cband12 全 0
                                 raw s 域  band [−4.577, +2.569]   cband12 [−0.455, +2.306]  (B-4)
                                 预注册门槛 frac_out_of_domain 中位 ≤ 1%  -> 两个 readout 都通过 (N-20)
[PASS] WA-P6b-latent-invariants  8/8 latent 规范；‖w_dir‖ 误差 1.70e-13；min α = 0.187
[SKIP] WA-P4b-fpre-sft-invariance  checkpoint 尚不存在
```

> **headline 的 n=2**（4 个样本里只有 2 个 `normal`）：最近秩约定下 k=2 时 `p10 == median`，
> 上面刻意只列 p10/p90 并标 n，免得被读成"分布很紧"。这些数只能当**未校准 basis 的量级参考**，
> 不是任何结论。JSON 里 `headline_n_low/hi` 与 `percentiles.n` 都带着 n（N-23）。

vision tower 单独加载：415.3M 参数，CPU float32 加载 3.0 s，512×768 前向 ~1 s/图。

> 注意口径（审阅 N-7/N-8，已写进 JSON）：WA-P5 测的是**未校准的 BA-0 起点**，不是校准后的 B；
> 条件数报的是 **Gram** 的（= 矩阵条件数的平方），1e10 阈值相当于 `phi` 本身的 1e5。
> 因此上面的 soft-IoU 只能读作"**未校准 basis 的下界**"，不是 Where-A 的天花板结论。

---

## 四、决策清单（D1/D2/D3 已由主 agent 裁定，2026-08-05）

已核实、无需决策的：V1–V9 全部（见上表）。

### 已裁定（代码已落地）

**D1 · `winner_confidence=low` 进 Where-A —— 裁定：训练放开，headline 收紧。**
`winner_confidence` 排的是"**哪个候选胜出**"，Where-A 的标签是"**候选区域在哪**"，两者正交；
抽样实测 low 与 normal 在定位/格式/宽高比/对齐相关性上无法区分（train 250 抽样含 102 个 low，
全部 100% 通过；独立审阅另抽 23 个 low 得同样结论）。
- 落地：`EXCLUDE_WINNER_CONFIDENCE_LOW = False`（训练用满 **local train 75,544**，不再丢 43%），
  `HEADLINE_WINNER_CONFIDENCE = ("normal",)`（V_where headline 只用 224 个 normal，
  `low` 由 `by_winner_confidence_*` 单独成层）。
- **边界**：这个放宽**只对 Where-A 的 basis 校准生效**，不得顺势流进 Where-B / What 的评测 GT。

**D2 · 校准目标函数 —— 裁定：内外层统一为 `soft_iou_minmax`。**
"IoU 禁当优化目标"属旧战役语境（协议 §5.5 本轮明文以 softIoU 作 Where 主 loss）。
更关键的是内外层不一致会让 D3 的论证失效（见下）。
- 落地：`FIT_OBJECTIVE = CALIB_OBJECTIVE = "soft_iou_minmax"`。
- 后果：外层 loss 与内层 fit_loss 现在同量级（3.2 的 mock 数字可见），两者本就都逐步记录。

**D3 · 校准时的 latent 来源 —— 裁定：维持"每 batch 用当前 B 现拟 latent"，但理由换成成立的那个。**
(a) 离线 latent 表在 `B` 走第一步之后就失配，1 epoch 内每个 latent 只更新一次等于没更新；
(b) 每个样本一次访问、当场用当前 `B` 拟合，符合"1 epoch over all eligible local train samples"。
- **正确性依据（修订）**：包络定理——`λ*(B) = argmin_λ f(B,λ)` 处 `∂f/∂λ = 0`，故
  `d/dB f(B,λ*(B)) = ∂f/∂B`。**这一条只在内外层是同一个 `f` 时成立**，D2 裁定后才真正成立。
  （审阅 B-6：此前内层 `1−softIoU`、外层 MSE，被丢掉的隐式项 `(∂g/∂λ)(dλ*/dB)` 是 O(1)，
  当时的梯度不属于任何良定义的双层目标。）
- **墙钟**：CPU 实测 **1.816 s/拟合**（8 次，`n_random=3, max_iter=80`，1536 点）→
  **43.1 h**（42,752×2）/ **76.2 h**（75,544×2）每臂。
  **GPU 数字是 S2 的必交输出**；若 GPU 上没有大幅下降，D3 需要重裁。

**D5 · guided filter 的 radius / eps —— 裁定：先例作废，S2 现扫现定。**
原来的"E2 先例"不成立：E2 的 `guided_filter` 是**全分辨率逐 basis 通道** `r=32`
（`experiments/E2_basis_fit_20260803/prep_data.py:149`），正是 §4.2 现在明令禁止的顺序。
- 落地：`GUIDED_PARAMS_PROVISIONAL = True` 标记当前 `radius_low=2, eps=1e-3` 为临时值；
  `scripts/sweep_upsample.py` 扫 3×3 组合，S2 定档后写回 `config.py` 并翻标记。
- **选择规则是字典序，不是"IoU 最高者胜"**（复审 N-20）：先过预注册门槛
  `S_OOD_FRAC_MAX = 0.01`（每样本越域像素比例的**中位数** ≤ 1%），再在合格集里比交付分辨率
  soft-IoU。理由：越域比例一高，clamp 就不再是"修边界"而是"决定掩膜"，那种配置的 IoU 是
  拿 clamp 换来的。若一个都不合格，脚本推荐越域最小者并把 `selection_rule` 写成
  `NO_SETTING_PASSED_THE_DOMAIN_GATE_...`，绝不静默回退到最高 IoU。
- **门槛本身是 provisional**：现在的 1% 来自 4 个真实样本的 CPU 实测（band 中位 0.000、
  p90 0.542%、均值 0.140%；cband12 全 0），S2 扫参时与 D5 一起定档。
- **N-27**：sweep 默认用未校准的 seeded B；S4 结束后须用 `--basis .../BA-3-Joint/B.npy` 复跑确认
  推荐值仍成立，再把 `GUIDED_PARAMS_PROVISIONAL` 翻 `False`。已写进脚本输出的 `note` 与 PREFLIGHT S6。

### 仍待主 agent 确认（保守默认继续）

**D4 · oracle 拟合的分辨率。**
默认在 `F_pre` 网格（32×48 = 1536 点）上拟合；**天花板现在同时在交付分辨率上报**
（`headline_hi`，审阅 B-4 要求）。直接在 512×768（393k 点）上拟合会让逐图 L-BFGS 涨约 250 倍，不建议。

**D6 · Band 的 `h` 参数化。**
协议只要求 `h>0`。默认沿用 E2 的 `h = 0.02 + 2.48·sigmoid(h_raw)`（有界，`h>0` 成立，`s∈(−3,3)` 时 2.5 已经是全通带）。另一种是 `softplus(h_raw)`（无界）。选有界是因为红线要求 σ 类参数用有界 sigmoid，且无界 h 在 L-BFGS 里容易跑到梯度消失区。

**D7 · BA-3-Joint 两个 readout 的损失权重。**
协议没给。默认 0.5/0.5（`JOINT_READOUT_WEIGHTS`）。

**D8 · 退化 mask 是否剔除。**
默认 `DROP_DEGENERATE_MASKS = False`——只统计不剔除，因为剔除会改变"上界"被测量的样本总体。450 个抽样里退化率 0.0%，所以这一项目前不影响任何数字。

**D9 · `F_pre` 是否落盘缓存。**
不缓存。75,544 张 × 1536 × 1024 × 2 B ≈ 236 GB，且 §2.3 明确"若不持久化大体积 `F_pre`，训练时直接复用冻结 Qwen3-VL 的同一次视觉前向"。副作用：四个臂各自重跑一次视觉前向。若主 agent 想省这笔，可以缓存 `B(F_pre)`（64 维，14.8 GB）——但那样 B 就不能被训练了，只对 BA-0 成立。
（**mask 视图**是另一回事，已按 N-3 落地缓存：`MaskViewStore` 让四个臂 + latent 作业只解码一次 `.cgt.png`。）

**D10 · CBand12 归一化方式（新，由 B-4 引出）。**
协议写 `m = Σ c_i g_i/(Σ g_i + eps)`。字面实现会在 σ 取下界时于中心之间与端点之外**恒等于 0**
（`Σ g_i ≈ 6e-27 < eps=1e-9`）。默认改用 `logsumexp`（同一公式的 `eps→0` 极限，`softmax(log g_i)` 稳定求值），
良态区与原式差 <1e-9，端点外返回最近基元的 `c_i`。原式保留为 `normalization="eps"` 且单测把
**旧式的塌陷数字钉死**。若主 agent 认为必须逐字实现协议公式，一行可切回——但那条路径的
静默失效已被本次审阅实测确认。

---

## 四之二、对 `REVIEW-impl-WhereA` 的逐项回应（2026-08-05）

审阅判 6 个 BLOCKER，公式层全部 pass。逐项修复如下（均附回归测试）：

| # | 修复 | 关键改动 | 回归测试 |
|---|---|---|---|
| **B-1** | 被拒拟合不再驱动 `B`、不再进天花板 | `all_starts_failed` 返回 `latent=None`（**字面零向量路径已删**）；新增 `FitResult.usable` 作唯一闸门；`step()` 对非 `ok` 直接 `continue`；`evaluate()` 只在 `ok` 子集上聚合，被拒集合单出 `n_rejected` + `reject_reasons` | `test_oracle`: `all_starts_failed` 无 latent / `usable` 语义；`test_calibrate`: 被拒不进 loss、全拒时 `grad_norm is None` 且权重逐位不变、被拒不进 ceiling |
| **B-2** | 校准 epoch 的 rejection report 落盘 | `step()` 返回 `fit_rows`（非 `ok` 全落 + 每 N 步采健康样本）；`run_calibration.py` 写 `fit_rejections.jsonl`；`Calibrator.rejection_summary()` 进 checkpoint 与 `schedule.json` | `test_calibrate`: 逐样本行字段齐全且可 JSONL、`sample_every` 行为、summary 计数 |
| **B-3** | 调度基于 eligibility 过滤后的真实步数 | `WhereADataSource.count_eligible()`（只读 record，结果缓存）；`total_steps = ceil(n_eligible/batch)`；跑完**断言 `step_count == total_steps`**，不等即 `SystemExit`；`schedule.json` 落 planned/actual/warmup/final_lr | `test_calibrate`: 按 eligible 建表能退火到 <1e-4，**用未过滤数则复现"结尾 LR 停在峰值 30%+"** |
| **B-4** | 高分辨率通路进生产 + 两个静默失效封堵 | `WhereASample.mask_hi/guide_hi` + `WhereADataSource(attach_hi=True)`；`evaluate()` 出 `headline_hi`；`guided_upsample` 返回 clamp 前域报告并 clamp 回 `S_DOMAIN=(−3,3)`；CBand12 改 `logsumexp`；新增 `WA-P4e` preflight 与 `sweep_upsample.py` | `test_upsample`: 越域实测 + clamp 不动域内值 + 域默认；`test_readout`: logsumexp≡eps（良态）且**旧式塌陷被钉死**；`test_calibrate`: `evaluate` 出 hi 档与 s 域 |
| **B-5** | train split oracle latent 进正式流程 | 新增 `scripts/make_oracle_latents.py`（读冻结的 `B.npy`，basis 不存在即拒跑；产 §2.3 shards，含 `r*(z)` 在固定 121 点 z 网格的采样供 §5.5 `L_curve`）；写进 `run_where_a.sh` 的 `oracle-latents` 步与 PREFLIGHT 文档 S5 | 脚本 argparse + import 校验；墙钟待 GPU 实测 |
| **B-6** | 包络定理论证恢复成立 | `CALIB_OBJECTIVE` 改 `soft_iou_minmax`（= `FIT_OBJECTIVE`）；`calibrate.py` docstring 重写，明说"只有内外层同一个 `f` 时才成立"，D3 理由随之更换 | `test_calibrate`: 内外层目标一致性断言 |

择要修的 nit：

| # | 修复 |
|---|---|
| N-1 | `basis.py` docstring 的 `s_low` 公式改正为 `3*tanh((w0 + alpha*<phi,w_dir>)/3)` |
| N-2 | `guided_upsample` 增加 batch/guide 数量一致断言，堵死"把 64 通道折进 batch 轴"的绕过 |
| N-3 | `MaskViewStore` + `WhereADataSource(maskview_root=...)`：发布的 mask 视图现在真的被消费（四臂 + latent 作业只解码一次 `.cgt.png`） |
| N-4 | 在线 pipeline 与打包作业统一：宽高比不符**两边都丢弃**并记 rejection，population 不再分叉 |
| N-5 | 删掉未使用的 `n_tokens` 形参；`WA-P4b` 与驱动器双层 try/except，**JSON 一定落盘**（fail-closed） |
| N-6 | 新增 `WA-P4d`：独立复算 `pos_embed` 的双线性插值并与 `fast_pos_embed_interpolate` + unshuffle 逐元素比（真实数据最大相对误差 1.85e-7），不再只有相干性这个间接代理 |
| N-7/N-8 | preflight JSON 显式标注 `projector_stage`（BA-0 起点，非校准后）与 `condition_number_basis`（Gram，= 矩阵条件数平方） |
| N-9 | NOTES 全部数字重新对齐到归档 JSON（本次的 3.1/3.2/3.3 与第二节表格都可逐字段查证） |
| N-10 | `FIT_N_RANDOM` 注释改正为 `n_random+3` 个种子（band ×2 极性 → 18 起点） |
| N-11 | 百分位口径固定为最近秩 `idx = clip(ceil(q*k)−1, 0, k−1)`，抽出 `percentiles()` 单一实现并单测钉死 |
| N-12 | `pack_oracle` 目标已存在时**硬报错**（与 `extract_maskviews` 一致），不再静默跳过发布 |
| N-13 | `run_calibration` 的 evaluate 改为流式消费生成器，不再一次性物化全部 V_where 样本 |
| N-14 | basis 元数据走裸文件 + digest 的**显式豁免**写进 PREFLIGHT S6 清单，要求进 REPORT |
| N-15 | 内层 L-BFGS 吞吐进 `WA-P5.throughput`（CPU 1.697 s/拟合 → 40.3/71.2 h 每臂），GPU 数列为 S2 必交输出 |
| N-16 | D5 先例作废，`GUIDED_PARAMS_PROVISIONAL=True` + `sweep_upsample.py` 升级为 S2 硬性输出 |

**顺带修的一处稳健性问题**（B-1 的测试暴露）：`build_starts` 的 lsq / radial informed 起点在
退化 `phi` 上会让 LAPACK 抛 `LinAlgError`，原实现会把整次多起点拟合一起带走。现已 try/except
降级到随机起点——informed 起点是便利，不是依赖。

### 复审第二轮（2026-08-05 晚）：B-7 + 6 项 nit

复审判原 6 项全部关闭、准入 S2/S4，新增 B-7（仅阻塞 S5）。本轮修复：

| # | 修复 | 关键改动 | 回归测试 |
|---|---|---|---|
| **B-7** | `r*(z)` 的 z 网格 121 → **257**，与协议 §5.5 `z = linspace(-3,3,257)` 一致 | `CURVE_Z_N = 257` 进 `config.py`（注明不是可调项），`make_oracle_latents.CURVE_Z` 由 `S_DOMAIN` + 该常量生成 | `test_scripts`: 网格长度/端点/等距逐项断言；`curve_of` 输出 257 点且与 `apply_readout` 在同一约定下逐元素相等 |
| **N-25** | 载荷声明 CBand 归一化约定 | 每样本 `oracle.json` 加 `cband_normalization`，run report 同步；Where-B 必须用同一约定重算 `R(z;ρ_pred)`，否则 `L_curve` 两侧不是同一个函数 | `test_scripts`: 两处字段都在，且值为 `logsumexp` |
| **N-18** | objective 一致性变成运行期护栏 | `Calibrator.__init__` 对 `cfg.objective != cfg.inner_fit.objective` 直接 `ValueError`；`fit_sample` 校验传入的 `fit_cfg.objective` | `test_calibrate`: 两个方向的错配都被拒，匹配的组合仍可构造 |
| **N-19** | 坏样本不再拖垮整个 epoch | `prepare()` 把 mask IO 异常转成 `self.rejections`（reason=`mask_io`）；`count_eligible` docstring 明示是**上界**并返回 `is_upper_bound`；`run_calibration` 的终局断言改成"缺口能被 `source.rejections` 逐条解释 → warning + 落 `schedule.json`；解释不了才 `SystemExit`" | `test_scripts`: resolver 抛错时 `prepare` 返回 None 且记一行 rejection；驱动器含 `gap_explained` / 条件 `SystemExit` |
| **N-20** | 越域门槛预注册 + 字典序选择 | `S_OOD_FRAC_MAX = 0.01`；`evaluate` 出每样本越域比例的百分位；`WA-P4e` 对中位数判 fail（默认配置下**能**失败了）；`sweep_upsample` 先过门槛再比 IoU，全不合格时明示 | `test_calibrate`: 域报告带百分位且 `max` 对得上；`test_scripts`: 门槛值 + 两个消费方都用它 + 过滤先于排序 |
| **N-21** | informed 起点降级不再无声 | 起点构造后统一做有限性检查（`_radial_start` 在 NaN `phi` 上不抛异常、只返回 NaN，这才是真正的漏网点）；丢弃时 `flags += ["informed_start_unavailable"]`，可被 `rejection_summary().flag_counts` 聚合 | `test_oracle`: 正常 phi 丢 0 个、NaN phi 丢 2 个且随机起点全部有限、fit 结果带该 flag |
| **N-22/N-23** | NOTES 数字重新对齐 + headline 标 n | 吞吐改为 JSON 现值（1.816 s/拟合 → 43.1 / 76.2 h）；`percentiles()` 一律带 `n`，preflight 另出 `headline_n_low/hi` | `test_calibrate`: `percentiles` 带 n、k=2 时 `p10==median` 被显式钉住；headline 行的 n 可查 |
| **N-24** | "B 永不被动" 变成强制 | `Calibrator.freeze_projector()`（清 optimizer/scheduler + `requires_grad_(False)`）；`make_oracle_latents` 调用后 assert，`phi_for` 包进 `no_grad`，收尾比对 `B` digest 未变 | `test_calibrate`: freeze 后 optimizer 为 None、step 无梯度、digest 不变 |
| **N-26** | 等价性单测覆盖到边界 | σ ∈ {0.30, 0.20, **0.10**} 参数化；另加一条"σ=0.05 时两式确实分离且是 eps 式在掉"的方向性断言，防止等价性测试将来变成空转 | `test_readout` |
| **N-27** | 记入待办 | sweep 的 `note` 与 PREFLIGHT S6 都写明：S4 后须用校准后的 B 复跑一次再翻 `GUIDED_PARAMS_PROVISIONAL` | — |

---

## 五、代码地图

```
q3vl/where/
  config.py     所有冻结常量 + 三个 dataclass 配置（PhiConfig / FitConfig / CalibConfig）
  fpre.py       §4.1  F_pre：forward hook、merge-order unshuffle、只装 vision tower 的加载器
  phi.py        §4.2  geo5 / L,S / 残差化 + 标准化 / phi_dir(71)
  basis.py      §4.2  s=3tanh(q/3)、w_dir、alpha、符号规范化（保掩膜不变）
  readout.py    §4.3  R-Band、R-CBand12、mirror、边界报告
  upsample.py   §4.2  guided upsample（多通道即抛错）+ combine_then_upsample
  oracle.py     §4.4/§10.2  多起点 L-BFGS float64 + rejection/fit report + 双分辨率评测
  projector.py  §4.4  B: 1024→64，seeded orthogonal
  calibrate.py  §4.4  四臂、双层 step、evaluate、AdamW+warmup+cosine
  maskdata.py   GT mask 定位（catalog→jsonl 回退）、加载、两档视图、eligibility
  pipeline.py   split → 图像 → F_pre → WhereASample（唯一碰真实模型和真实 shard 的地方）
  packing.py    §2.3  mask 视图 / oracle latent 的 indexed-tar 发布 + 随机读校验
  preflight.py  §14 项 4/5/6（+ 4b：F_pre 对 SFT 不变、4d：位置编码、4e：高分辨率通路）
  scripts/
    validate_masks.py      ✅ 已跑（450 样本）
    extract_maskviews.py   ⏸ 全量 IO 作业，未跑
    sweep_upsample.py      ⏸ D5 定档，S2 硬性输出，未跑
    run_calibration.py     ⏸ 单臂全量校准，未跑
    make_oracle_latents.py ⏸ train split oracle latent（Where-B 前置），未跑
    run_where_a.sh         ⏸ 提交入口，内置 D-20 四步
  tests/                   142 个用例，全绿（+ test_scripts.py：待跑作业的接口契约）
```

---

## 六、红线自查

| 红线 | 本实现 |
|---|---|
| σ 参数化禁裸 exp（有界 sigmoid） | `bounded_sigmoid` 是唯一通路，`σ∈[0.025,0.30]`、`k∈[1,40]`、`h∈(0.02,2.50)`；`test_readout` 在 raw=±1e6 下实测边界 |
| s 轴禁平滑正则 | 全代码无任何 s 轴正则项；损失只有掩膜项 |
| 逐像素算子禁 (x,y)/邻域/MLP/排序 | 这条约束的是**渲染器**的逐像素算子；Where-A 的 `phi_dir` 按协议 §4.2 显式包含 geo5，是 basis 定义本身，不是渲染算子 |
| s 禁逐图 min-max/softmax 归一化 | 没有。`s = 3tanh(q/3)` 是全局有界映射；逐图标准化只作用在**输入特征** L/S 与语义块上（§4.2 明文要求），且这正是让**全局** `w` 跨图可用的前提 |
| IoU 禁当优化目标 | **主 agent 2026-08-05 裁定该红线属旧战役语境**（协议 §5.5 本轮明文以 softIoU 作 Where 主 loss），内外层统一为 `soft_iou_minmax`。见 D2 / B-6 |
| s 场落在锚点外 / 被 clamp 成 0（s 缓存消费契约） | 生产方声明 `S_DOMAIN=(−3,3)`；消费方每次上采样都返回 clamp **前**的 `raw_min/raw_max/frac_out_of_domain` 并逐层上报；clamp 是**整臂常量**、非逐图。CBand12 的分母塌陷（另一种"s 轴没了"）改 logsumexp 修掉并被单测钉死 |
| checkpoint 选择禁用 val loss | Where-A 不做 checkpoint 选择；选择集 V_where 上报的是 oracle 指标 |
| 每个消融行必带 Δ_const/Δ_shuffle | 属于 Where-B/What 的消融表，不在本任务范围；BA-0-Fixed 本身就是本阶段的 no-calibration 对照 |
| 烘焙一致性一等指标 | 属于 Stage-What |
| attention 导出必须 eager | 本阶段不导出 attention；`F_pre` 取的是 block 输出，与 attention 实现无关（默认 sdpa，可切 eager） |

## 七、给实现审阅的提醒（最容易出错的三处）

1. **merge-order unshuffle**（`fpre.py`）：写错不会报错，只会把图像打散成 2×2 块。已用真实 processor 钉死，并有"朴素 reshape 必须失败"的反向断言。
2. **`inv_freq` 非持久 buffer**（V5）：`meta` + `to_empty()` 的常见加载写法会静默毁掉旋转位置编码。当前实现改为正常构造 + `load_state_dict`，并在返回前检查全部 buffer 有限。
3. **符号规范化必须同时翻 `w0` 和镜像 readout**：只翻 `w_dir` 会改掩膜。`canonicalize` 翻 `(w_raw, w0)` 并 `mirror(ρ)`，12 组随机 latent × 2 readout 实测掩膜差 <1e-12。
