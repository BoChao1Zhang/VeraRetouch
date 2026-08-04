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

| 项 | V_where（200，仅 normal） | train（250，含 low） |
|---|---|---|
| 定位成功率 | 100.0% | 100.0% |
| 读取+校验成功率 | 100.0% | 100.0% |
| PIL 模式 | 全部 `L`（uint8 单通道） | 全部 `L` |
| 与 EXIF 校正图宽高比一致 | 100.0%（相对误差中位数 6.3e-5，最大 3.7e-4） | 100.0% |
| 退化（低分辨率视图 std<1e-4） | 0.0% | 0.0% |
| 软边像素占比（0.04<v<0.96） | 中位 0.204，p95 0.893 | 同量级 |
| `image.upscaled` 占比 | 16.7% | 10.4% |
| EXIF orientation | 199×1, 1×8 | 249×1, 1×8 |
| 定位来源 | catalog 100%，jsonl 回退 0 | 同 |

**对齐性验证（关键，格式检查抓不到"看起来对但其实错"的 mask）**：把 `I_in`（源图）与 `I_tar`（`.jpg` 渲染件）都重采样到 mask 分辨率，算 `|I_tar−I_in|` 的逐像素均值 `d`：

- `corr(d, mask)`：V_where 中位 **0.787**（均值 0.749，最小 0.313，p95 0.932）；train 中位 **0.783**（最小 0.349）
- 编辑能量落在 mask 内的比例 `Σ(d·m)/Σd`：V_where 中位 **0.715**，train 中位 **0.680**

相关系数达不到 1 是预期的：LUT 对不同颜色的改动幅度本来就不同，`d` 的幅度受内容调制。这两个数足以判定 `.cgt.png` 就是该样本被编辑的区域掩膜，与 `CLAUDE.md` 数据纪律里"`C_GT` = 逐候选区域掩膜（软边单通道），不是配色真值"完全吻合。

**分辨率链**：`.cgt.png` 是 build 的渲染分辨率（短边 1024，原始宽高比）→ 面积平均到 spec-5 网格 `(out_h, out_w)` 得 `mask_hi` → 再面积平均到 `F_pre` 网格 `(out_h/16, out_w/16)` 得 `mask_low`（拟合就在这一层做）。

---

## 三、CPU 级验证结果

### 3.1 单元测试：**97 个全部通过**（`q3vl/where/tests/`，用时 35 s）

| 文件 | 数量 | 覆盖 |
|---|---:|---|
| `test_readout.py` | 15 | Band/CBand12 公式逐符号对照手算；极端 raw 下 `h>0`、`k∈[1,40]`、`σ∈[0.025,0.30]`、`o,c∈(0,1)`（禁裸 exp 的实证）；μ 网格是固定对称 linspace(−3,3,12)；polarity `π b+(1−π)(1−b)` 互补；**mirror 恒等式 `m(−z;mirror(ρ))≡m(z;ρ)` 到 1e-12**；mirror 是对合；批量参数广播与逐样本一致 |
| `test_basis.py` | 15 | `s=3tanh(q/3)` 与手算一致；`|s|<3`；`‖w_dir‖=1`；`α=softplus>0`；**符号规则**（绝对值最大系数为正）；**canonicalize 不动一个像素**（12 组随机 latent，两种 readout，1e-12）且 `s→−s` 精确；幂等；`w_raw=0` 不被静默翻转；JSON 往返 |
| `test_phi.py` | 15 | P2 = Legendre；坐标按真实宽高比（短边 [−1,1]，长边 [−AR,AR]，方图仍是方的）；geo5 列序与行主序；L/S 在已知颜色上的取值；残差化后与 `[1,geo5,L,S]` 的最大相关 <1e-6；语义块零均值单位方差；梯度确实流回语义块（B 的唯一通路）；常数图不炸；形状校验 |
| `test_upsample.py` | 10 | **多通道输入直接抛 `ChannelOrderError`**（禁"先上采样 64 通道再组合"）；常数场被保持；边缘过渡宽度显著窄于 bilinear（且 ≤ 半个低分辨率格）；box 均值对照手算；`combine_then_upsample` 只吐标量；梯度可回传 |
| `test_fpre.py` | 5 | `grid = (H/16, W/16)`；unshuffle/shuffle 往返；**unshuffle ≠ 朴素 reshape**；**对着真实 image processor 验布局**（并断言朴素 reshape 会失败） |
| `test_oracle.py` | 13 | 目标函数；两种 readout 都能从植入 latent 复原（soft-IoU>0.95）；返回 latent 已规范化；同 seed 完全确定；**不可拟合目标进 rejection 而非置零**；常数 mask 打 flag；多起点数目正确（band 12 / cband 6）；fit report 可序列化；低/高分辨率双档评测 |
| `test_calibrate.py` | 11 | 四臂表与协议 §4.4 一致；projector 同 seed 同 B、正交误差 <1e-5、cond=1.0；**BA-3 一步闭环**（loss 有限、两个 readout 都在、B 确实移动、每 (样本×readout) 一行 fit report）；BA-1/BA-2 只训自己的 readout；**BA-0 永不训 B**（无 optimizer、requires_grad 全 False、权重逐位不变）；evaluate 出 ceiling；warmup+cosine 调度；形状校验；checkpoint 往返 |
| `test_packing.py` | 5 | 成员名符合 `webdataset_basename_v1`、三成员共享 key；打包后 manifest `status=complete`/未压缩/样本数正确；随机读 + sha256 校验通过；index 行含 shard/member/offset/length/size/sha256/schema_version；basis 落盘与 digest；空输入不会留下半成品目录（原子发布） |
| `test_maskdata.py` | 9 | 只保留 local l1–l6；默认排除 `winner_confidence=low`（可关）；缺定位符/几何即拒；mask 视图形状与面积平均；软边保持；退化判定；宽高比校验；**live 定位+校验读 5 个真实 `.cgt.png`**；5 个 split 索引存在 |

### 3.2 mock 端到端闭环（BA-3-Joint，8 步 × batch 4）

```
step loss(MSE)  grad_norm   lr        fit_loss(1-softIoU)      rejected
 1   0.024863   0.183968    1.0e-4    band .2369 / cband .2569   0
 2   0.020239   0.098008    9.5e-5    band .2313 / cband .2371   0
 3   0.019231   0.071206    8.1e-5    band .2431 / cband .1907   0
 4   0.026461   0.172712    6.1e-5    band .2239 / cband .2335   0
 5   0.021922   0.069757    3.9e-5    band .2391 / cband .2318   0
 6   0.017208   0.087963    1.9e-5    band .2461 / cband .1958   0
 7   0.018865   0.092281    5.0e-6    band .2424 / cband .2290   0
 8   0.022736   0.072557    0.0       band .2321 / cband .2350   0
‖ΔB‖_F = 0.0671   (‖B‖_F = 8.0)      全部有限   拒绝拟合 0/16
evaluate(4 个新 mock)：median soft-IoU  band 0.777 / cband12 0.781，fit 成功率 1.0/1.0
```

### 3.3 真实数据上的 CPU 级 preflight（4 个 V_where 本地样本，float32，seeded-orthogonal B，即 BA-0 起点）

```
[PASS] WA-P6a-readout-bounds
[PASS] WA-P4c-upsample-order
[PASS] WA-P4a-fpre-geometry      grid 32x48 <- 512x768，宽高比 1.5 一致
                                 空间相干性  正确 unshuffle 0.0209  vs  朴素 reshape 0.0089
[PASS] WA-P5-basis-conditioning  残差化后最大相关 8.1e-7；design Gram cond 中位 17.8 / 最大 35.0
                                 phi Gram cond 中位 8.6e4 / 最大 1.5e5；死通道 0
                                 oracle 拟合成功率 band 100% / cband12 100%
                                 median soft-IoU  band 0.965 / cband12 0.963  （未校准 B！）
[PASS] WA-P6b-latent-invariants  8/8 latent 规范；‖w_dir‖ 误差 2.7e-13；min α = 0.102
[SKIP] WA-P4b-fpre-sft-invariance  checkpoint 尚不存在
```

vision tower 单独加载：415.3M 参数，CPU float32 加载 3.0 s，512×768 前向 ~1 s/图。

---

## 四、假设与待确认清单（可自行核实的已当场核实）

已核实、无需决策的：V1–V9 全部（见上表）。

以下是**两种做法都合理、且影响后续**的，一律采**保守默认**继续，**未静默拍板**：

### 待主 agent 决策

**D1（最重要）· `winner_confidence=low` 是否进 Where-A。**
`CLAUDE.md` 数据纪律写"low 不进 SFT 主训与评测 GT"。但 Where-A 的标签是**候选区域掩膜**，与"哪个候选胜出"这件事无关——mask 的正确性不受 winner 排名影响，且抽样实测 low 样本的 mask 定位/格式/对齐质量与 normal 无差别（train 250 抽样含 102 个 low，全部 100% 通过）。
- 保守默认（**当前代码**）：`EXCLUDE_WINNER_CONFIDENCE_LOW = True`，headline 只用 normal。代价：本地 train 从 75,544 掉到 42,752（**−43%**），V_where 本地选择集从 400 掉到 224。
- 建议选项：校准训练放开 low（`--include-low`），V_where 的 headline 指标仍只用 normal，low 单独一层报告。
- 无论选哪个，`low` 层都会被单独统计（`run_calibration.py` 的 `strata.winner_confidence`）。

**D2 · projector 校准的目标函数。**
PLAN v2 L205 与 E2 的逐图 oracle 拟合用的是 `1 − soft-IoU_minmax`；红线写"IoU 禁当优化目标"。两者的适用面不同：oracle 拟合是**表达力上界的测量**（不产生被训练的参数），而 B 是**被训练的模型参数**，且 soft-IoU 正是 Where-A 在 V_where 上的选择指标——拿选择指标直接反传 B 属于"对着评测指标训练"。
- 保守默认（**当前代码**）：`FIT_OBJECTIVE = "soft_iou_minmax"`（拟合）、`CALIB_OBJECTIVE = "mse"`（训 B）。
- 代价：内外层目标不同，双层优化不是严格一致的（内层的 argmin 不是外层损失的 argmin）。若主 agent 认为红线只针对场/渲染器，把 `CALIB_OBJECTIVE` 改成 `"soft_iou_minmax"` 是一行的事，两个数在每次 step 里都被记下来（`loss` 与 `fit_loss_per_readout`），事后可对比。

**D3 · 校准时的 latent 来源：逐 batch 内层拟合 vs 离线 latent 表。**
§3 表格写"训练：shared basis projector B **与离线逐图 oracle latent**"，§4.4 写"逐图用多起点 L-BFGS 拟合 oracle，同时校准共享 B"。两种读法：(a) 先用 B_init 离线拟一张 latent 表，之后固定；(b) 每个 batch 用**当前** B 现拟 latent，再对 B 走一步。
- 保守默认（**当前代码**）：(b)。理由：(a) 的 latent 是在 B_init 下最优的，B 一动就失配，1 epoch 内每个 latent 只能被更新一次，等于没更新；(b) 每个样本一次访问、当场用当前 B 拟合，符合"1 epoch over all eligible local train samples"，且用包络定理（内层取到最优时 latent 的一阶项为 0）保证固定 latent 的梯度就是对的。
- 代价：内层 L-BFGS 进了训练循环（默认 `n_random=2, max_iter=40`，实测 1536 点约 2 s/图/readout on CPU，GPU 上会快但仍是主要开销之一）。若主 agent 要 (a)，`Calibrator.fit_sample` 是唯一需要换的接口。
- **本项直接决定 Where-A 的墙钟时间**，请优先裁决。

**D4 · oracle 拟合的分辨率。**
协议没写在哪一层拟合。默认在 `F_pre` 网格（32×48 = 1536 点）上拟合，`evaluate_latent` 同时给低分辨率与"经一次 guided upsample 后的原分辨率"两档指标。若要求直接在 512×768（393k 点）上拟合，逐图 L-BFGS 的成本会涨约 250 倍，不建议。

**D5 · guided filter 的 radius / eps。**
协议只说"一次 edge-aware guided upsample"。默认 `radius_low=2`（低分辨率格，等价原分辨率 32 px）、`eps=1e-3`（E2 `guided_filter` 默认，guide 为 [0,1] luma）。guide 取 Rec.709 luma 单通道；也可以用 RGB 三通道 guided filter（更贵、边缘更准）。这个参数会直接影响 Where 的边缘指标，建议在 preflight 阶段扫 2–3 个值再定档。

**D6 · Band 的 `h` 参数化。**
协议只要求 `h>0`。默认沿用 E2 的 `h = 0.02 + 2.48·sigmoid(h_raw)`（有界，`h>0` 成立，`s∈(−3,3)` 时 2.5 已经是全通带）。另一种是 `softplus(h_raw)`（无界）。选有界是因为红线要求 σ 类参数用有界 sigmoid，且无界 h 在 L-BFGS 里容易跑到梯度消失区。

**D7 · BA-3-Joint 两个 readout 的损失权重。**
协议没给。默认 0.5/0.5（`JOINT_READOUT_WEIGHTS`）。

**D8 · 退化 mask 是否剔除。**
默认 `DROP_DEGENERATE_MASKS = False`——只统计不剔除，因为剔除会改变"上界"被测量的样本总体。450 个抽样里退化率 0.0%，所以这一项目前不影响任何数字。

**D9 · `F_pre` 是否落盘缓存。**
不缓存。75,544 张 × 1536 × 1024 × 2 B ≈ 236 GB，且 §2.3 明确"若不持久化大体积 `F_pre`，训练时直接复用冻结 Qwen3-VL 的同一次视觉前向"。副作用：四个臂各自重跑一次视觉前向。若主 agent 想省这笔，可以缓存 `B(F_pre)`（64 维，14.8 GB）——但那样 B 就不能被训练了，只对 BA-0 成立。

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
  preflight.py  §14 项 4/5/6（+ 4b：F_pre 对 SFT 不变）
  scripts/
    validate_masks.py     ✅ 已跑（450 样本）
    extract_maskviews.py  ⏸ 全量 IO 作业，未跑
    run_calibration.py    ⏸ 单臂全量校准，未跑
    run_where_a.sh        ⏸ 提交入口，内置 D-20 四步
  tests/                  97 个用例，全绿
```

---

## 六、红线自查

| 红线 | 本实现 |
|---|---|
| σ 参数化禁裸 exp（有界 sigmoid） | `bounded_sigmoid` 是唯一通路，`σ∈[0.025,0.30]`、`k∈[1,40]`、`h∈(0.02,2.50)`；`test_readout` 在 raw=±1e6 下实测边界 |
| s 轴禁平滑正则 | 全代码无任何 s 轴正则项；损失只有掩膜项 |
| 逐像素算子禁 (x,y)/邻域/MLP/排序 | 这条约束的是**渲染器**的逐像素算子；Where-A 的 `phi_dir` 按协议 §4.2 显式包含 geo5，是 basis 定义本身，不是渲染算子 |
| s 禁逐图 min-max/softmax 归一化 | 没有。`s = 3tanh(q/3)` 是全局有界映射；逐图标准化只作用在**输入特征** L/S 与语义块上（§4.2 明文要求），且这正是让**全局** `w` 跨图可用的前提 |
| IoU 禁当优化目标 | 训练参数（B）的目标默认 MSE；soft-IoU 只用于逐图 oracle 拟合（表达力测量，PLAN L205 / E2 先例）与报告。见 D2 |
| checkpoint 选择禁用 val loss | Where-A 不做 checkpoint 选择；选择集 V_where 上报的是 oracle 指标 |
| 每个消融行必带 Δ_const/Δ_shuffle | 属于 Where-B/What 的消融表，不在本任务范围；BA-0-Fixed 本身就是本阶段的 no-calibration 对照 |
| 烘焙一致性一等指标 | 属于 Stage-What |
| attention 导出必须 eager | 本阶段不导出 attention；`F_pre` 取的是 block 输出，与 attention 实现无关（默认 sdpa，可切 eager） |

## 七、给实现审阅的提醒（最容易出错的三处）

1. **merge-order unshuffle**（`fpre.py`）：写错不会报错，只会把图像打散成 2×2 块。已用真实 processor 钉死，并有"朴素 reshape 必须失败"的反向断言。
2. **`inv_freq` 非持久 buffer**（V5）：`meta` + `to_empty()` 的常见加载写法会静默毁掉旋转位置编码。当前实现改为正常构造 + `load_state_dict`，并在返回前检查全部 buffer 有限。
3. **符号规范化必须同时翻 `w0` 和镜像 readout**：只翻 `w_dir` 会改掩膜。`canonicalize` 翻 `(w_raw, w0)` 并 `mirror(ρ)`，12 组随机 latent × 2 readout 实测掩膜差 <1e-12。
