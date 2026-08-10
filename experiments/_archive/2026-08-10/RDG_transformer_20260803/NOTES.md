# RD-G · NOTES（实施前核实记录 + 假设清单 + 待决策项）

实验号：**RD-G**（EXPERIMENTS_v3 §3.1）｜日期：2026-08-03｜卡：GPU 0（G-Lite 系）+ GPU 1（G-Base 系）
分支：`lens-exp`｜解释器：`/home/bc/miniconda3/bin/python`（torch 2.6.0+cu124, colour-science 0.4.7, numpy 2.4.6, sklearn 1.7.2）

---

## 0. 读过的文档节（只读引用节）

| 文档 | 节 | 取到的东西 |
|---|---|---|
| `EXPERIMENTS_v3_2026-08-02.md` | §0、§3.1 RD-G 行、Changelog 全部 | 判据（ΔE00 p50 降≥15% 且方差比>0.6；MLP 达天花板 90% → 停）；F5 gate 放行 = RD-G Stage-1 解锁；「PASS 必须区分未触发死刑 vs 正面证据」 |
| `PLAN_v2_local-retouch_2026-07-31.md` | §1.1、§1.2、§1.4 全套、§1.6、§3 第一级/第二级（含 Stage-1 配方与 M1–M9） | 生成器规格（G-Lite/G-Base 宽深、token 布局、ModLN、learnable Fourier 加 key、register token、energy routing）、输出头逐行公式、Stage-1 优化器配方、烘焙-导出损失预算、N\* 门槛 |
| `DATA_ASSIGNMENT_2026-08-02.md` | §1.1/§1.2、§2（D-RENDER / D-CONSTRUCT / D-CUBE / D-HALD）、§3.2 RD-G 两行、§4 行动项 A/B/G/H | D-RENDER 实测 1,144,000 对；low/abstain 明文允许进渲染器预训练；C_GT 不是配色真值、L_cube 从 `recipe.preset` 渲染；S/P split 旁表只读 |
| `IMPL_DOSSIER_2026-08-02.md` | §2.2、§2.4、附录 B | GLUT Eq.1-3、22N+12、A.1 初始化、CGLUT §3.2 生成器形状（64 维 embedding + 3 Linear shared encoder + 逐类 head）、两处矛盾裁决、锚点数字 |
| `model/glut_repro/` | `model.py`、`model4d_naive.py`、`losses.py`、`fit_e1.py` | 现成渲染核心与 4D 实现；E1 直接拟合引擎的 σ 退火 / 密度控制配方 |
| `experiments/A0_glut_repro_20260803/REPORT.md` | §0–§2 | `10·L_hc` 是 −10.73 dB 元凶（量纲错）；`R_sparse` 把 opacity 推向 1 不是 0；mining +0.66 dB |
| `experiments/E1_cube_N_20260803/REVIEW-result.md` | 全 | **N\* = 48**（p50 门被漏读，32 未过门）；N=48 逐 LUT 直接拟合 p50 0.467 / p90 0.785 / PSNR 44.33 |
| `experiments/tooling-wave1/bgr_check/REPORT.md` | 全 | F5 判定 (a)：`axis_order:"bgr"` 只是内存轴序，after 图语义与 RGB 一致，**RD-G Stage-1 可直接用 after 图**；tetra vs tri 中位差 0.009 ΔE00 |
| `experiments/tooling-wave1/data_splits/REPORT.md` | §2/§6 | split 旁表 seed `verasplit-v1`；D-RENDER 1,144,000 逐 build 对账 |

---

## 1. 在线核实记录（附录 B 之外的外部事实，全部打开原始来源）

| # | 事实 | 来源（已 fetch） | 结论 |
|---|---|---|---|
| 1 | register token 的出处与作者 | `https://arxiv.org/abs/2309.16588` | ✅ 真实：Darcet, Oquab, Mairal, Bojanowski, *Vision Transformers Need Registers*；摘要明写「providing additional tokens to the input sequence」。**摘要页不含默认 register 数**，故本实验用 PLAN §1.4 自己写的 **4 个**，不向该论文归因任何具体数字 |
| 2 | GLUT arXiv 号是否为真 | `https://arxiv.org/abs/2605.19889` | ✅ 真实且对题：*GLUT: 3D Gaussian Lookup Table for Continuous Color Transformation*，摘要含 CGLUT 条件生成器与风格插值 |

**没有引入任何需要外部核实的架构细节**：本实验**不用** RoPE / RMSNorm / SwiGLU / QK-norm / muP。
理由（预注册）：PLAN §1.4 指定的是 pre-LN + ModLN + cross-attn + learnable Fourier PE + register token，
这些全部在权威文档内；额外引入需要外部超参出处的组件，只会给一个「测生成器容量」的实验增加不可归因的自由度，
且本项目检索引擎有编造前科。因此 transformer 用 **标准 pre-LN decoder + GELU MLP（mlp_ratio=4）**，
`torch.nn.MultiheadAttention`，无外部数字。

**优化器配方不需要外部核实**：AdamW β(0.9,0.95) / wd 0.05 / lr 4e-4 / warmup 2000 / clip 1.0 / bf16
逐字来自 `PLAN §3 第二级 · Stage 1`（内部权威文档），非外部论文。

---

## 2. 实现前的自查 CI（红线，实测数字）

`ci_checks_rdg.py`（可复跑）：

| # | 检查 | 判据 | 实测 |
|---|---|---|---|
| 1 | `model_rdg.render()` 与现成渲染核心 `model.BatchedGLUT.forward` 同参数下等值 | max abs diff < 1e-5 | **2.38e-7** ✅（"接到同一个渲染核心" 是测试不是声称） |
| 2 | 零初始化输出头 ⇒ f(x)=x（PLAN §1.4 的 CI：33³ 全格点 max ΔE00 < 1e-4） | <1e-4 | **8.87e-5** ✅ |
| 3 | 全局仿射 G 初始化 = 0（红线，禁 I） | ‖G‖=0 | **0.0** ✅ |
| 4 | σ 参数化有界（禁裸 exp） | σ ∈ [0.02, 0.50] | 零初始化处恒 **0.26**；范围 [0.02,0.50] ✅ |
| 5 | `tetra_lookup` 与 colour-science `table_interpolation_tetrahedral` 一致 | <1e-6 | **1.19e-7** ✅ |
| 6 | `delta_e00` 与 colour-science `delta_E_CIE2000` 一致 | 相对误差 <1e-3 | max abs **0.0187**，相对均值 **4.2e-4** ✅（fp32 精度尘埃） |

参数量（实测，`RDGModel.n_params()`）：

| 臂 | tokenizer（共享） | **generator** | 合计 |
|---|---|---|---|
| `mlp`（CGLUT §3.2，width 128 = 论文 Large） | 220,736 | **251,292** | 472,028 |
| `mlp_wide`（同结构加宽，容量对齐 G-Lite） | 220,736 | **4,933,244** | 5,153,980 |
| `gtiny`（d=192,L=2） | 220,736 | **1,561,445** | 1,782,181 |
| `glite`（d=256,L=4,cross@{1,3}，PLAN ≈5M） | 220,736 | **4,447,846** | 4,668,582 |
| `gbase`（d=384,L=6,cross@{1,3,5}，PLAN ≈15M） | 220,736 | **14,056,806** | 14,277,542 |

渲染器 payload 恒为 **23N+12 = 1,116**（N=48）——即"头重脚轻"里的"脚"，五个臂完全相同。

---

## 3. 假设与设计决定（已自行核实/预注册者）

1. **N=48**，不是 PLAN §1.4 字面的 32。依据：E1 结果审阅推翻初报（p50 门被漏读），N\*=48。
2. **D-RENDER 只取 g 线（global）候选**。g1/g2/g3 全量扫描实测：候选 100% `format="lut"` + `render_mode="global"`，
   即 after 图 = 整幅套 preset 的 .cube，`recipe.preset` 才是合法的 L_cube 目标。l 线是掩膜合成图，
   拿来当 cube 监督会把目标系统性地拉回恒等。**置信度不筛**（normal/low/abstain/null 全收，DATA_ASSIGNMENT §1.2 明文允许）。
   实测入册：600,000 行，`no_split_src=0 / no_npy33=0 / not_lut=0`。
3. **条件 = (I_in, after) 6 通道叠放**，不是 PLAN 字面的「输入=LUT 作用后的图」单流。
   理由：只给 after 时 LUT 不可辨识（同一张 after 可由无数 (源图, LUT) 对产生），
   容量比较会退化成"猜风格先验"而不是"生成参数"。after-only 档列为待决策 #1 的对照。
4. **L_cube 目标 = T2 的 33³ 重采样表**（`/var/cache/veradata/dcube/npy33`），三线性读表。
   三线性是生产渲染器自己的插值器；F5 实测 tetra 与 tri 中位差 0.009 ΔE00（远低于归档 JPEG 底噪）。
   烘焙一致性那一项**故意改用四面体**回读，因为宿主用四面体（PLAN §1.6）。
5. **色彩口径**：主口径 = 立方体内均匀随机色（与 E1、与本实验天花板同轴）；
   副口径 = 该样本 I_in 自身像素色（ENNELUT 教训，PLAN 第一级第 2 条）。训练一半一半。
6. **不加 1D 前置曲线**。PLAN §1.4 的输出头清单本身就没有曲线头；加了会同时改渲染器，
   把"生成器容量"和"渲染器容量"混在一起。曲线是 RD-I 的活。→ token 布局相应只留 48 primitive + 1 global。
7. **PLAN §1.4 的 5 个 mask token 不实例化**：掩膜基底不在 Stage-1 cube 目标里，
   实例化只会得到 5 个无监督死 token。
8. **存在门 g_i 放进混合分子（与 opacity 同处），不是 payload 门**。这是被 PLAN 自己的 CI 逼出来的：
   零初始化处 g_i ≡ sigmoid(4)=0.982，公共因子在归一化里对消 ⇒ f(x)=x 成立；
   若做 payload 门则 f(x)=0.982x，ΔE00≈1，直接违反 §1.4 的 `max ΔE00 < 1e-4`。
   代价：o 与 g 在数学上冗余（见待决策 #4）。
9. **不引入任何感知/色相损失**。A0 实测 `10·L_hc` = −10.73 dB（CIELab 绝对值 vs RGB[0,1] 量纲不匹配，
   梯度差 229×）；DOSSIER 自己给的附加损失总收益只有 +0.11 dB。任何将来要加的感知项必须先过量纲对拍。
10. **checkpoint 选择禁用 val loss**（红线）：用 val ΔE00 p50，方差比 <0.30 一票否决。
11. **γ_μ / s 轴相关**：Stage-1 无 s 轴，R-3 hinge 与 s 轴平滑正则均不适用（红线在此**空满足**，
    不算"正面证据"）。主 agent 的 γ_μ 相对量修订（`0.4 × std(anchor_grid)`）记录在案，
    留给 Stage-2 / RD-STD 系使用。
12. **s-sensitivity 不作判据**（主 agent 中途更正）：本实验主判据是 Δ_const / Δ_shuffle 与 ΔE00 分位。
13. **天花板自己算、脚本落盘**：`tools/fit_ceiling.py`（本目录）。不复用 `RD_std_e_20260803/ceiling/` 的产物——
    那份是 D-CONSTRUCT 图像域 4D LUT 分箱天花板，与本实验的 cube 空间生成器天花板不是同一个量。
    本实验的天花板定义：**同一套 `ParamHead` + 同一个 `render`，把 z 从"预测"改成"逐 LUT 自由优化"**，
    即容量无穷、无泛化负担的 oracle 生成器。天花板拿不到的是渲染器容量，臂拿不到的是生成器容量。
14. **探针/子集取样一律等距或无放回随机，绝不取前 n 个**（G3 教训）。LUT 选择用 `rng.choice` 无放回后排序。
15. **Δ_const / Δ_shuffle 在生成器语境下的定义**（每行必带）：
    `Δ_const = PSNR(真条件) − PSNR(condition-dropout 学到的常量条件)`；
    `Δ_shuffle = PSNR(真条件) − PSNR(跨样本置换条件)`。
    塌陷成"一张平均 LUT"的生成器在这两列上都是 0，这正是红线要抓的东西。
16. **判据用相对量**（主 agent 中途更正三）：晋级判据里的 ΔE00 p50 降幅是**相对 MLP 基线**的相对量，
    天花板占比也是相对量；不使用任何绝对 dB 阈值。

---

## 4. 实测基建数字（诚实记录，含瓶颈）

| 项 | 实测 |
|---|---|
| D-RENDER g 线可用对 | **600,000**（manifest 全量，split 覆盖 100%） |
| 本实验取样 | train(S-train×P-train) 320,000 / val_img(S-val×P-train) 6,000 / val_lut(S-val,test×P-val,test) 4,690 |
| 源图可回取率 | **72,587 / 72,587 = 100%** |
| 缓存分辨率 | 128×128（BOX 面积平均；JPEG `draft` 做 DCT 域预降采样） |
| 缓存体积 | after 16.25 GB + in 3.57 GB ≈ 19.8 GB（可全量进 page cache，机器 125 GB RAM） |
| **原始路径吞吐（瓶颈实测）** | **125–140 图/s**（56 进程并行，NFS ranged read + JPEG 解码）。**这就是"不建缓存就吃不满卡"的实证**：按训练需要的 ~5,000 样本/s 计，原始路径慢 **≈40×**。见 §5 |
| 瓶颈定位（关键） | 56 个解码 worker 稳定停在 **3.3–3.5% CPU**，宿主 load average 138–151（48 核，与其他 agent 共享）。worker 几乎全程在等 I/O ⇒ **瓶颈是 NFS 往返延迟，不是 JPEG 解码，也不是本机 CPU**。折算带宽 ≈ 140 图/s × 350 KB ≈ **48 MB/s**——远低于链路能力，是"每样本一次 open+seek+read"的延迟墙 |

---

## 5. 数据加载瓶颈结论（任务卡点名要的发现）——**撞了两堵墙**

### 墙一：NFS 延迟（原始 D-RENDER 路径）

- 单样本 = 1 次 NFS tar ranged read（≈350 KB）+ 1 次短边 1024 JPEG 解码。
  56 并行实测 **125–140 图/s**，worker 稳定 3.3–3.5% CPU（等 I/O，不是解码不动）。
- 训练侧一张 H100 在本配方下要 **≈1,700–2,300 样本/s** ⇒ 原始路径慢 **12–18 倍**。
- 处置：一次性离线解码成 128×128 uint8 memmap（19.8 GB，一次 ~50 分钟）。

### 墙二：本地机械盘随机读（离线缓存放在 `/home` 上）

**这堵墙比第一堵更隐蔽，也更致命**：缓存建好后第一次启动六臂，13 分钟一条训练步日志都没有，
显存却已经涨到每卡 70 GB。`nvidia-smi pmon -s um` 逐进程读数给出真相——

```
GPU 0  290797 (glite)  SM  -    fb 21352 MiB     <- 我的进程，0% SM
GPU 0  291011 (mlp)    SM  -    fb 14436 MiB
GPU 0  288197 (别人)   SM 16%   fb  1078 MiB
```

`iostat -x`：`/home` 所在 **sdb 是机械盘**，`%util 99.2 / r_await 120 ms / 284 IOPS / 21 MB/s`。
6 个进程各自随机读 19.8 GB 缓存，本机 125 GB RAM 已被其他 agent 占去 62 GB，页缓存装不下，
于是每个样本退化成一次 120 ms 寻道。训练需要 6×768×98 KB ≈ **460 MB/step**，
按 21 MB/s 算 **22 秒/步**——与实测完全吻合。

**处置：把缓存整个搬进 `/dev/shm`（tmpfs，纯内存）**，六臂改读 `/dev/shm/rdg_cache`。

| | 迁移前（`/home`，机械盘） | 迁移后（`/dev/shm`） |
|---|---|---|
| 我的进程 SM 利用率 | **0%**（`pmon` 读数 `-`） | **12–49%/进程**，两卡合计 99–100% |
| `data_wait` 占比 | ≈100%（200 步跑不完 13 分钟） | **2.2–4.8%** |
| 吞吐 | <0.07 it/s | **1.0–2.3 it/s**（768 样本/步 ⇒ 750–1,800 样本/s/进程） |

### 结论与可复用建议

1. **在这台机器上，任何"边训边解码"的数据管线都不可行**：NFS 与本地机械盘两条路都比 GPU 慢一个量级以上。
2. **离线缓存必须落在 tmpfs（或 NVMe），不能落在 `/home`**。这一条建议直接推广给后续所有 RD/RO 臂。
3. 若坚持在线路径，可行组合（未采用，备查）：① NVIDIA DALI / nvJPEG 把解码搬上 GPU；
   ② 把 shards 预取到本地 NVMe；③ `draft()` 让 libjpeg 在 DCT 域直接出 1/8 尺寸。
4. **诊断手法值得沉淀**：`nvidia-smi` 的整卡 util 会被同卡其他作业"顶满"而掩盖自己 0% 的事实，
   必须用 `nvidia-smi pmon -s um` 看**逐进程 SM**，再用 `iostat -x` 定位到设备。

---

## 5bis. RO-3 真实 s 档的接入前置（2026-08-03 18:0x，一手核实后写入）

主 agent 通报 RO-3 判出可用读出（L11 第 5 头，差分场 AUC 0.9298），
scache arm 在 `/var/cache/veradata/scache/ro3-fused/`。**我打开原始 `_ARM_INFO.json` 核实了以下数字**：

| 项 | 实测（arm 自述 + 我方复算） |
|---|---|
| `domain_global` | **[−3.2559, 8.8047]**——**不在 [0,1]** |
| 分位 | p1 −1.025 / p50 **−0.0974** / p99 1.536 / p100 8.805 |
| `frac_gt_zero` | 0.4116（**58.8% 的格为负**） |
| `no_per_image_norm` | true（与"s 禁逐图归一化"红线一致） |
| arm 自带 WARNING | ro9 臂已实测踩中：默认 `clamp=(0,1)` 消费后 >0 的格仅 **0.93%**，**且不报错** |

**我方复算（193/772 条）**：逐图 frac<0 中位 **0.613**、最大 0.879；
默认 `clamp(0,1)` 后存活格中位 **38.7%**、最差条目 **12.1%** ⇒ **默认路径会静默丢掉约六成的场**。
按 arm 自己的 `consume_recipe`（p1/p99 winsorize 后线性映射）处理后：全格非零，range 严格 [0,1]。

### 这个坑对本实验的具体威胁（比 ro9 更隐蔽）

我的 Stage-2 4D 档里 `mu_s` 是**冻结在 [0,1] 上的 K=6 网格**、`sigma_s ∈ [0.025,0.30]`。
**两种失败模式，性质不同**：

1. **场落在锚点之外** → 所有高斯密度塌到 0，归一化退回 ε，渲染器只剩全局仿射分支。
   实测 RO-3 生数据的"孤儿格"占比中位仅 **1.6%**（最差条目 25.4%）——**这条大多数时候抓不住**。
2. **场被 clamp 到 0** → 值仍然**落在锚点域内**，孤儿检查完全沉默，但 s 轴已经没了。
   **这才是 ro9 真正踩的那个坑。**

### 已落地的两道守卫（`train_rdg2.py`，CI 26/26 覆盖）

- `assert_s_matches_declared_domain(raw, lo, hi)`——**消费方必须声明域，且生数据必须真的住在里面**。
  RO-3 生数据按 [0,1] 消费时实测抛错：*"60.3% of cells are outside it (raw range
  [−1.655, 5.023], 55.6% below)"*。oracle 掩膜声明 [0,1] 且确实在 [0,1] ⇒ 静默通过，不误伤。
- `assert_s_in_anchor_domain(s, mu_s)`——补抓失败模式 1，训练**第 0 步**就跑，不浪费整轮。
- `load_s_arm_recipe()` / `s_to_unit()`——按 arm 自述的 recipe 归一化，**全局常数，无逐图归一化**。
- CI 新增 3 条：生数据按 [0,1] 消费必须抛错 / 量化"默认 clamp 会清零多少格"（实测 55.6%）/
  过 recipe 后必须是合法 [0,1] 场。另把 CI 的随机源 seed 住——
  之前 ΔE00 容差检查用了无种子随机数，连续两次跑出 25/26 与 26/26，**已修**。

**结论**：真实 s 档现在是"接上去就会响"，不是"接上去悄悄坏"。是否加这一档等主 agent 看表 B 后定。

## 6. 待主 agent 决策（保守默认已执行，未静默拍板）

1. **条件形态**：本实验主档用 **(I_in, after) 对**；DATA_ASSIGNMENT §3.2 还要求「输入 = I_in + 风格指令」的
   第二档对照。后者需要文本编码器（preset bank 里有 `text_emb.vlm_plain.npz`），会引入第二个变量。
   **保守默认**：本轮只跑图像对档，把 after-only 与 文本条件档列为 RD-G 后续 wave。
   影响：PLAN 字面的「输入=LUT 作用后的图」单流没有被直接检验。
2. **L_param / L_prequery 缺席**：PLAN Stage-1 损失表含 `L_param`(0.3) 与 `L_prequery`(0.2)，
   两者都要 Stage-0 逐 LUT 金标准参数当回归目标。**E1 只落盘了指标（`per_fit.jsonl` 无参数）**，
   目标不存在。**保守默认**：本轮不带这两项，只用 `L_cube` + 每层 aux + `L_prior`。
   代价：置换不变的 cube 损失少了 query 直接监督（Mask2Former 的「可学但不监督=没改」风险）。
   **补救**：本实验的 `fit_ceiling.py` 已经把 P-train 384 个 LUT 的拟合参数落盘，
   下一轮可以直接当 `L_param` 目标——请示是否值得再开一轮。
3. **μ anchor 来源**：PLAN 写「anchor 来自 Stage-0 k-means」。本轮由 `fit_ceiling.py` 现算
   （P-train 384 LUT 拟合 μ 的 k-means），若该文件缺席则回退 GLUT A.1 的均匀网格。
   两者对五个臂完全相同，不影响臂间比较，但会影响与 E1 绝对数字的可比性。
4. **o 与 g 数学冗余**：见 §3.8。两者都乘在混合分子上，在当前损失下不可辨识。
   **保守默认**：两者都保留（规格忠实度），报告里注明冗余。
   备选：合并为一个门，或把 g 改成 payload 门并放弃 §1.4 的恒等 CI。请拍板。
5. **mmart_ppr10k 池 326 源疑似落 PPR10K 官方 val 段**（DATA_ASSIGNMENT 行动项 H，未拍板）。
   本实验是 cube 空间的渲染器预训练，与 PPR10K 官方口径打榜无关，**保守默认：按冻结 split 表原样使用，不额外剔除**，
   在此登记以免日后 E20 追责。
6. **判据「方差比」无成文定义**。本实验预注册两种读法并都报：
   - `var_ratio`（主）= 预测变换在样本间的方差 / 目标变换在样本间的方差，同一组色上算。
     塌陷成"一张平均 LUT"时 → 0。这条直接对应 PLAN「L1 最低 = 最保守平均 LUT」的担忧，故取为主。
   - `var_explained`（副）= 1 − MSE / Var_between（R² 式）。
   若主 agent 认定原意是后者，报告里的数字可直接换列，不需重跑。
7. **Stage-2（D-CONSTRUCT）范围**：任务卡把 D-CONSTRUCT 定为"主实验"，DATA_ASSIGNMENT §3.2 把
   RD-G Stage-2 定为 D-SFT-G + D-SFT-L。两处不一致。**保守默认：按任务卡走 D-CONSTRUCT**，
   并把 Stage-1（D-RENDER，容量-收益曲线的来源）作为同等重要的一半报出。
