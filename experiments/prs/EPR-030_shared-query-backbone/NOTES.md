# EPR-030 NOTES —— 假设与待决策（保守默认已写死在代码与提案里，**未静默拍板**）

每条给：任务卡怎么写的 / 冲突或缺口在哪 / **本轮采用的保守默认** / 备选 / 影响面 / 谁来定。

---

## 1.【必须裁定】全局仿射 `G` 锚在 0 还是锚在 I

- **任务卡原文**：「全局 query 的 12 维：ΔG(9) bias=0（`G = I + ΔG`）、g(3) bias=0。」
- **同一张卡的另一条要求**：「运行时断言 …… 断言 `f(x) == x`（在 1e-6 容差内）且返回的
  `GlutParams` 各字段与 `SharedGeometry` 的初值逐元素相等。」
- **冲突（算术事实，不是判断）**：载体的 Eq.5 是
  `f(x) = Σ_i w_i (M_i x + b_i) + clamp(G x + g, 0, 1)`，末端再 clamp。
  取 `M_i = I`、`b = 0`、`Σ_i w_i → 1` 时局部支路给出 `x`；若同时 `G = I`、`g = 0`，
  全局支路再给一个 `x` ⇒ `f(x) = clamp(2x) ≠ x`。**两条要求不可能同时成立。**
  同一事实在 EPR-029 §3.2 的 NOVEL 注里写过，`GlutParams.identity`（`glut.py:315-345`）
  也把 `g_matrix` 写成 0。
- **本轮保守默认**：`G = ΔG`（bias 0，**不锚在 I**）⇒ step0 恒等成立，实测
  `max|f(x) − x| = 4.172e-07`，8 个参数字段与 `SharedGeometry` 初值偏差全 0。
- **备选（已实现，未启用）**：`--g-residual` ⇒ `G = I + ΔG`。该档下 step0 是 `f = 2x`，
  `assert_step0_identity()` 会**主动 raise**（测试
  `test_the_global_query_is_anchored_at_zero_not_at_the_identity` 并排断言了两档的行为）。
- **影响面**：只影响本臂 step0 的取值与「step0 headline == B0_identity」这条见证；不影响判据定义。
- **请用户拍板**：保留默认（G 锚 0），还是按卡面文字改成 `I + ΔG` 并同时撤掉恒等断言。

## 2.【必须裁定】直通 clamp 的默认作用范围与模块级默认值

- **任务卡原文**：「加一个开关 `clamp_grad: "st" | "hard"`，默认 **"st"**」。
- **两个未写明的点**，本轮各取一个保守默认：
  1. **作用在哪几个 clamp 上**：`--clamp two` 有两处硬 clamp（全局分支 `Gx+g`、末端整体）。
     卡面只点名了全局分支的零梯度。**本轮把 `st` 同时作用在两处**（同一个 `_clamp01` 帮助函数），
     forward 两处都逐位不变。备选 = 只改全局分支、末端仍硬裁。
  2. **`glut.py` 的模块级默认**：卡面说默认 st。若把**模块默认**设成 st，
     EPR-024~029 五个既有臂的**梯度行为会被追溯改变**（forward 不变，但它们若重跑就不是同一个
     优化问题）。**本轮把模块默认留在 `hard`，只在 EPR-030 的入口 `--clamp-grad` 默认 st**，
     即「本臂默认 st」这一条按卡面执行，既有臂一字不动。
- **影响面**：(1) 决定饱和点上末端 clamp 是否也回传梯度；(2) 决定跨臂旧数字是否需要重跑。
- **请用户拍板**：(1) 两处 / 只全局分支；(2) 模块默认留 hard（当前）/ 改 st 并重跑五臂。

## 3.【必须裁定】0.1× 低 lr 组映射到哪些参数 —— **含一条由实测改掉的默认值**

- CGLUT App A.1 把 0.1× 给 "style embeddings and shared geometry parameters"。本臂**两者都没有**
  这个名字，所以映射属 NOVEL（与 EPR-029 §3.5 同性质）。
- **本轮的映射（两处）**：
  1. `q_emb` + `PE_R/PE_G/PE_B`（**30,720** 参数）—— 唯一按 μ 网格索引、持有几何先验的参数；
  2. `head_gauss` + `head_global`（**17,442** 参数）—— Bias-HyperInit 下**头的 bias 就是
     shared geometry**（μ 网格残差基、σ=0.15、o logit +4、M=I），正是 EPR-029 里叫 `theta_base`
     并放进 0.1× 组的那个物件。
  其余 **20,230,656** 参数走基础 lr 1e-3。
- **第 2 条不是先验偏好，是实测后改的默认**（第一版把头留在 1e-3，CPU 冒烟预演时炸了）。
  协议：`scratchpad/probe_epr030.py`，CPU、真 z 真 bank、fp32、B=32×Q=256、seed 20260810、
  d=512/L=6、cosine T_max=300、`max_grad_norm=1.0`；探针 = train 前 16 条在 9³ 网格上的
  `measure_degeneracy`；`f ≡ identity` 时 `L_rec = 0.1756`。

  | step | 头 **1.0×**：L_rec / gnorm / point_std / cross_std | 头 **0.1×**：同四列 |
  |---|---|---|
  | 0 | 0.1897 / 2.85 / 0.0447 / 2.84e-4 | 0.1897 / 2.85 / 0.2997 / 4.84e-4 |
  | 10 | 0.4849 / 137.79 / 0.1289 / 6.31e-3 | 0.2349 / 46.67 / 0.2155 / 1.45e-3 |
  | 20 | 0.4901 / 252.39 / 0.1585 / 3.70e-3 | 0.1702 / 47.87 / 0.2118 / 4.98e-3 |
  | 30 | 0.5776 / 195.80 / 0.4215 / 5.82e-3 | 0.1666 / 16.44 / 0.2333 / 2.86e-3 |
  | 40 | 0.3510 / 74.34 / 0.2277 / 7.57e-3 | 0.1573 / 18.60 / 0.2269 / 4.42e-3 |
  | 50 | 0.3336 / 56.01 / 0.1913 / 1.32e-2 | 0.1651 / 28.49 / 0.2128 / 4.14e-3 |
  | 60 | 0.3629 / 131.82 / **0.0000** / **0.000e+00** | 0.1594 / 11.05 / 0.2633 / 1.11e-2 |
  | 70 | 0.3437 / 93.42 / 0.4068 / 6.00e-3 | 0.2393 / 87.09 / 0.3152 / 8.79e-3 |
  | 80 | **NaN** | 0.2587 / 111.13 / 0.2354 / 1.28e-2 |
  | 90–160 | **NaN**（每个探针点） | L_rec 0.1417–0.1683、gnorm 7.80–28.26、point_std 0.2159–0.2589、cross_std 1.01e-2–3.48e-2 |

  同协议 `--backbone mlp` 对照跑满 300 步：s200 L_rec 0.1580 / gnorm 0.288 / cross_std 9.04e-3；
  s299 L_rec 0.1420 / gnorm 0.105 / cross_std 1.03e-2。
  **落盘默认（头 0.1×）复跑 250 步确认**（`--tag shipped`，cosine T_max=250）：
  s230 L_rec 0.1385 / gnorm 8.37 / point_std 0.2458 / cross_std 5.79e-2；
  s240 0.1357 / 5.02 / 0.2438 / 5.79e-2；s249 0.1523 / 9.34 / 0.2443 / 5.76e-2；
  `l0_calls = 250`（= 步数，L0 每步都被调用）；墙钟 96.2 s / 250 步（CPU 10 线程）。
- **备选（都已实现，各是一条消融行）**：`--qdec-head-lr-scale 1.0`（上表左列）、
  `--qdec-prior-lr-scale 1.0`。
- **另外两条没有采用的候选**（属**用户级裁定**，本轮**未动**）：StatLUT 自己的配方是
  AdamW `wd=0.05` + **5-epoch 线性 warmup**，本战役的冻结优化器口径（Adam / lr 1e-3 cosine /
  无 warmup / wd=0）来自 CGLUT App A.1、是给 0.45 M 的 MLP 写的。把 20.28 M 的 transformer
  放在这套口径下是否需要 warmup / wd，**请用户裁定**；本轮一律照冻结口径，不混用另一篇的配方。
- 逐组参数量与 lr 都落盘在 `run_setup.json` 的 `epr030.optimizer_groups`。

## 4.【已按纪律处理，供备案】`--backbone mlp` 对照行的初始化档

- 卡面：「mlp 档走现有 `CGLUTGenerator`，用于对照」。现有 EPR-024 主臂的构型是
  `mode="full", m_residual=False, zero_init_last=False`（PyTorch 默认初始化，**step0 不是恒等**）。
- **保守默认**：`mlp` 档**逐字沿用 EPR-024 的构型**，即该行与 EPR-024 只差 loss 与 clamp 梯度两处。
- **备选**：给 mlp 档也套恒等锚定（`m_residual=True, zero_init_last=True`），使两档 step0 相同；
  但那样 μ 会被零初始化到 0（不再是规则网格），几何与 EPR-024 不同，对照行就同时改了两处。
- **后果（写进结果表脚）**：主行与 `mlp` 行的 **step0 不同**（主行是恒等，mlp 行不是）。

## 5.【已按纪律处理，供备案】step0 恒等断言的两处容差口径（**第二处是 GPU 冒烟打出来的**）

- 卡面：容差 1e-6。
- **算术事实**：Eq.2 的分母有 `+ε`（ε=1e-6），所以 `Σ_i w_i = 1 − ε/(Σ_j p_j o_j + ε) < 1` **恒成立**，
  即**载体自己**的恒等点也只在这个比值内成立。N=48 时实测偏差 4.172e-07（在 1e-6 内）；
  N 很小时同一个 ε 摊到更少的高斯上，偏差变大（N=6 实测 1.03e-04）。
- **保守默认**：断言的地板取 `max(1e-6, 载体在恒等点自身的偏差)`，并把两者**分列落盘**
  （`step0_maxabs_f_minus_id` 与 `carrier_eps_dev`）。同时**加了一条更强的断言**：decoder 的
  step0 前向必须与「它自己 bias 所编码的参数集」的前向 `torch.equal` 逐位相等。
- 主构型（N=48）两个值相同、都在 1e-6 内，所以这条口径在主构型上**不放宽任何东西**
  （GPU 实测 3.576e-07，CPU 实测 4.172e-07）。
- **第二处容差（GPU 冒烟第一次 rc=1 打出来的）**：断言里「头的 bias 编码的参数集 ==
  `GlutParams.identity` 的 SharedGeometry 初值」这一条，参考量会在**当前设备上重算**
  `uniform_grid_positions`，而 `mu_base` 是构造时在 CPU 上算好再 `.to(cuda)` 的。
  `(i+0.5)/3` 的 CPU 与 CUDA 除法差 **1 个 float32 ULP = 2⁻²⁴ = 5.960e-08**（4×4×3 网格
  B 轴的 0.5 与 0.8333 两个格心）。CPU 上两边同源、差 0，所以这条只在 GPU 上暴露 ——
  首次 GPU 冒烟正是死在这里（23 秒，训练前）。
  **保守默认**：该项的地板改为一个 float32 ULP，实测偏差逐条落盘
  （`shared_geometry_dev_*`，GPU 实测：`mu` 5.960e-08、其余六项 0）；
  「模型吐出的参数 == 头 bias 编码的参数集」那一条**仍然要求恰为 0**（同一批张量，无重算，
  GPU 实测 8 项全 0）。补了一条回归测试：把 σ 挪 1e-3、把一个格点挪 1e-4，两种**真**错误
  都仍然 raise —— 地板没有把真错误一起吞掉。
- **备选**：把 `mu_base` 与参考量都固定在 CPU 上比较（要在断言里做一次 `.cpu()`，与
  「判据路径禁 `.cpu()`」的仓库习惯相抵），或干脆不比较 `mu`（会削弱断言）。

## 6.【已按纪律处理，供备案】判据表没有登记 `EPR-030` 这个 arm 名

- `q3vl/whatb/criteria.py:102-110` 的 `ARM_AXES` 只登记 EPR-024~029。
- **保守默认**：**不改共用表**，本臂在 `assert_publishable(..., axes=("P1",))` 显式传轴，
  `required_criteria` 因此走显式分支。required 表 = 12 个公共键 + P1 四列，与 EPR-024 逐字相同。
- 备选：给 `ARM_AXES` 加一行 `"EPR-030": ("P1",)`（一行增量，但动的是六臂共用文件）。

## 7.【已按纪律处理，供备案】复用 runner 的方式

- 卡面：「**复用** `run_carrier_arm.py` 的数据管线 …… 不要复制粘贴一份新的（能 import 就 import，
  必要时把公共部分抽出来）」。
- **本轮做法**：给 `run_carrier_arm.py` 的六个函数加 `arm=<模块>` 形参，**默认值就是 EPR-024 的
  carrier 模块**，函数体内 `A.xxx → arm.xxx`。EPR-030 的入口用
  `R.main(argv, arm=sys.modules[__name__])` 复用整条管线（数据、z 缓存断言、colorspan 断言、
  mining、quick eval + 退化守卫、判据出板、发布门、checkpoint 选优）。
- **副作用（已核对）**：`run_carrier_arm.py` 的 sha256 变了，EPR-024 已落盘的 `run_setup.json`
  里记的是**旧** sha256。既有测试 `test_carrier.py` 的三条 runner 测试（`evaluate_and_publish`
  签名、旗标面、选优只读 headline）**一行未改仍全过**。
- 备选：把公共部分抽成 `q3vl/whatb/arms/_pipeline.py` 再由两个 runner 各自 import
  （改动面更大，且会让 EPR-024 的 `run_carrier_arm.py` sha256 同样变化）。

## 8.【已按纪律处理，供备案】`--loss-level` 与 `--loss` 的关系

- EPR-024 的梯级旗标 `--loss-level {1,2,3,4}` 决定 `λ_hc` / `λ_sparse` / `λ_img`。
  本臂的损失由 `--loss` 与三个 `--lambda-*` 决定，两套会互相覆盖。
- **保守默认**：本臂把 `--loss-level` **钉死在 1**（在 `add_arguments` 里把该 action 的
  `default` 与 `choices` 都改成 `[1]`，传别的值 `Epr030Config` 直接 `ValueError`），
  `λ_hc / λ_sparse` 改由 `Epr030Config` 的同名 property 从 `--lambda-*` 取。
  `--loss-level 4`（图像项）在本臂**显式禁用**（`train_step` 收到 images/alphas 即 raise）。

## 9.【已按纪律处理，供备案】冒烟档首次 quick eval 的步数

- 本臂 step0 **结构上就是恒等**（`identity_dev ≈ 0`、`cross_std = 0`），退化守卫的两条地板在极早期
  必然贴地 —— 与 EPR-026/027 各赔过一轮的坑同源，但成因相反（那两个是先塌后长，本臂是从恒等出发）。
- **CPU 实测（真 z 真 bank、fp32、B=32×Q=256、6 步）**：第 3 步的首次 quick eval **已过三条地板**
  （产物走到了出板阶段）。
- **保守默认**：冒烟取 `--max-steps 20 --eval-every 10`（首次 quick eval 在第 10 步）。
  **GPU 冒烟实测该步的守卫 ok**：`point_std` 0.2802 / `identity_dev` 0.0682 /
  `cross_std` 1.5111e-3（地板 1e-3 / 1e-3 / 1e-4）。
  **阈值、判据、守卫实现一字未动**，只挪冒烟档的 binding 步数。全量档在 step 2936。
- **CPU 预演实测（落盘默认、同一套冒烟旗标、rc=0）**：`quick_eval@step10` 守卫 **ok** ——
  `point_std` 0.2796（地板 1e-3）/ `identity_dev` 0.0706（1e-3）/ `cross_std` 1.4824e-3（1e-4）；
  16 个 required 判据键全部 n>0；`l0_calls = 10`。

## 10.【已知瑕疵，未改共用代码】退化守卫的打印里 `arm` 写着 EPR-024

- `A.quick_eval`（EPR-024 的实现，六臂共用）把 `extra={"arm": ARM, ...}` 里的 `ARM` 写死成
  `"EPR-024"`，本臂复用它，所以守卫的落盘记录 / 失败打印里 `extra.arm` 显示 `EPR-024`。
- **保守默认**：**不改共用文件**（改它要动五个既有臂的行为面）。板上的 `arm` / `arm_name`
  字段是对的（`EPR-030` / `QDEC`），只有守卫 `extra` 这一处显示旧名。
- 备选：给 `A.quick_eval` 加一个 `extra_tags` 形参（一行增量，但动共用文件）。

## 11.【待观测，非决策】主构型的显存与每步墙钟仍未逐项实测

- 提交时 `--mem-peak 12` 是**申报值不是实测值**。冒烟跑通了但 runner 目前**不落盘**
  `torch.cuda.max_memory_allocated`，所以显存峰值仍是申报值；全量档开跑前建议加一列。
- 已有的墙钟事实（冒烟，gpu0）：`ready` 用了 40 s（读 159,215 行 train index + 256 条
  colorspan 对拍 + 四份 z 缓存断言），**整个作业 elapsed 74 s**（20 步训练 + 两次 quick eval +
  一整块 48 条的 gated board）。**每步墙钟没有单独计时**，不从 74 s 里倒推。

## 12.【已作废并重做】L8 新数据不接 → **2026-08-16 已接入**

- 旧卡面：「L8 新数据本次不接」。**新卡面（2026-08-16）改为接入**，本条作废，实施记录如下。
- 已落地：`--data {v2seg,v2seg+l8}`，入口默认 **`v2seg+l8`**；n 93,934 → **119,828**；
  步/epoch 2,936 → **3,745**；总步数 117,440 → **149,800**。
- 实现落点：`q3vl/whatb/splits.py`（`load_l8_index` / `train_normal_rows` / `train_normal_n` /
  `train_source_facts` / `color_texts_of` / `load_index_cached`）、
  `q3vl/whatb/zcache.py`（`MultiZCache` + `resolve_leaf` 的 `<root>/<context>/` 回退）、
  `q3vl/whatb/scripts/run_carrier_arm.py`（`open_z_caches(extra_sources=...)`、总体合并、
  两条运行时断言）、`q3vl/whatb/arms/carrier.py`（`CarrierConfig.data`）、
  `q3vl/whatb/scripts/run_epr030_arm.py`（`--data`、`train_n` 实测传入）。
- **没有写死 119,828**：`TRAIN_NORMAL_N = 93934` 仍是冻结块第 1 条、仅作 v2seg 那一路的断言；
  L8 那一路对 `l8_train.manifest.report.json` 自己的 `n_manifest` / `counts.usable_final_normal`
  断言；合并数是 `len()` 出来的。

## 13.【已按纪律处理，供备案】L8 的 `where` / `color` 不在本 EPR 里重新转换

- 卡面 E 要求「转换用 `q3vl/data/twoseg.py::convert()`，不要新写一份」。
- 盘上事实：`l8_train.manifest.jsonl` 的 `where` / `color` 两字段**就是**缓存生产者
  （`zcache_l8/_src/build_manifest_l8.py`）用 `q3vl.data.twoseg.convert` 转好的，z 缓存
  `meta.dataset.where_color_source` 逐字记录了这一点；`scan_batch` / `geometry_for` /
  `lengths` / `prepare_image` 同样是生产阶段调用的。
- **保守默认**：本轮**直接读 manifest 的这两个字段**，不在训练侧再调一次 `convert()`。
  再转一次会产生同一段 reasoning 的第二次转换结果，而 z 已经是按第一次的结果读出的——
  「两次转换是文本缓存与 z 缓存漂移的标准成因」（该缓存生产者自己的 D7 决策同因）。
- 备选（未采）：训练侧重跑 `convert()` 并与 manifest 逐条对拍。代价是每次启动多读一遍
  49,629 行 `sft.jsonl`（NFS），且对拍不过时也只能停手——与直接读同一份的效果相同。

## 14.【已改，供备案】`--eval-every` 的默认从 2936 改成 0（= 一个 epoch）

- 旧默认是**写死的 2936**。接入 L8 后一个 epoch 是 3,745 步，写死的 2936 会让首次 quick eval
  （退化守卫的 binding 时机）落在 0.78 个 epoch 上，且随每次加数据继续漂。
- 改法：`--eval-every 0` ⇒ 运行时取 `cfg.steps_per_epoch`。对 `--data v2seg` 与
  EPR-024 的 carrier 臂**行为不变**（`ceil(93934/32)` 仍是 2936）。
- 队列载荷 `waves/whatb_epr024_029_arm.sh` 的 QDEC 全量档已同步改成 `--eval-every 0`，
  其余五个臂的分支**一行未动**。

## 15.【待决策，保守默认已执行】L8 的 `winner_confidence=low` 20,235 条只进缓存不进训练

- L8 z 缓存里 **46,129** 行（normal 25,894 + low 20,235）都在，训练总体只取 normal 25,894。
  `MultiZCache` 因此持有 140,063 个 z，其中 20,235 个在本臂永不被取用（内存 mmap，代价是磁盘常驻）。
- 保守默认：**照战役数据纪律，low 不进主训**（`winner_confidence=low` 不进主训与评测 GT）。
- 待用户裁定的备选：把 low 也放进 whatb 主训（Where-B 侧 `EXCLUDE_WINNER_CONFIDENCE_LOW=False`
  是**放**的，两个消费者本来就不同口径）。若采纳，n 会变成 140,063，步/epoch 变成 4,377。
  **本轮不动。**

## 16.【待观测，非决策】B3 桶池与 `Lib_tr` 仍只定义在 sft2seg train index 上

- `bucket_pools` 与 `_library_ids` 读的是**整份 sft2seg train index**（含 low，159,215 行），
  与训练总体无关；L8 没有 `minor` 字段，也没有 record shard。
- 保守默认：**不动**。五条平凡基线不含被训练的参数，动它们的总体会让已出板的
  B0/B1/B2/B3/B4 五个数字全部重算，而这五个数字本轮明确「不作废」。
- 代价（实测数字，不作结论）：L8 normal 的 2,785 个 `lut_id` 中 **2,727** 个也出现在 sft2seg
  train index（3,149 个）里，**58** 个只在 L8 出现，未进入 `Lib_tr` 的抽样总体。
  B4_oracle 用样本自己的 `lut_id`，不受影响。

## 17.【待决策，未拍板】放开色批后的**生产档 `--batch-split` 取值**

- 已做：`BATCH_SPLITS` 放开（`assert b*q == COLORS_PER_STEP` 删除，改为按档记录
  `BATCH_SPLIT_COLORS[档] = B×Q`，并保留「`32x256` / `64x128` 仍恰为 8192」这条断言），
  17 档在 1 × H100-95.58 GiB 上各跑 20 步实测（表见 PROPOSAL「跨臂冻结口径的变更清单」）。
  **17 档全部未 OOM。**
- **未做（不许静默拍板）**：生产档取哪一个。任务卡要的是两套候选，不是一个值：
  - 单进程/卡 ≥70 GiB：`512x28672` → reserved **77.36 GiB**（95.58 GiB 卡余量 18.2 GiB），
    5.008 s/步，235 步/epoch，40 ep = 9,400 步 ≈ **13.1 h**；
    更高两档 `512x32768`（88.29 GiB，14.45 h）/ `1024x16384`（88.70 GiB，8.76 h）余量仅 ~7 GiB。
  - 两进程/卡各 ~35 GiB：`256x24576`（33.36 GiB，1.983 s/步，469 步/epoch，18,760 步 ≈ 10.3 h）
    或 `512x12288`（33.55 GiB，2.809 s/步，235 步/epoch，9,400 步 ≈ 7.3 h）。
- 队列侧的既有约束（事实，不作结论）：`qjob.sh` 的显存准入是
  `used < Q_MEM_ADMIT_GB(65)` 且 `used + --mem-peak <= Q_MEM_CAP_GB(80)`。
  两进程 ~34 GiB 档在默认阈值下可共存；单进程 77.36 GiB 档实际占用超过 80 GiB 的申报上限，
  只能以 `--mem-peak 80` 在空卡上准入，或改 `Q_MEM_CAP_GB`。
  另有一条既有运维纪律与「两进程/卡」相冲突：CLAUDE.md「同卡禁双训练臂（破坏步数匹配）」。
- 主机 RAM（事实）：标定探针自身峰值 RSS 1.71–2.25 GB（不含 z 缓存与评测图像）；
  任务卡给的真实入口是 7.94 GB/进程，4 进程 ≈ 31.8 GB / 125 GB。
- 本轮**未提交任何训练作业**；正式矩阵由主 agent 编排。

---

## 已完成的核实（外部事实，逐条打开过原始来源）

| 事实 | 核实方式 | 结果 |
|---|---|---|
| NILUT 的 L1 与注释 | `curl raw.githubusercontent.com/mv-lab/nilut/main/fit.py` | HTTP 200，6,194 B；`loss = torch.mean(torch.abs(model_output - ground_truth)) # more stable than L2` |
| Text-to-LoRA 的 Bias-HyperInit | `curl .../SakanaAI/text-to-lora/main/src/hyper_llm_modulator/hyper_modulator.py` | HTTP 200，39,354 B；`:535 nn.init.zeros_(head.weight)`、`:541 head.bias.copy_(torch.cat(init_bias))` |
| VeraRetouch Retouch Renderer = 2,577,795 | `curl` 两份原文件后**逐层复算** | HTTP 200 / 200；166,659 + 2,409,344 + 1,792 = **2,577,795**（与任务卡给的数一致） |
| 本臂各档参数量 | `sum(p.numel())` 实测 + 闭式对拍 | 逐档相等（`test_every_ablation_rung_counts_itself_correctly`） |

---

## 18.【已实施，供备案】把 EPR-025~029 五臂接到 EPR-030 的口径上（2026-08-16 下午）

不是新的结构改动，是重跑准备。**五个臂各自的实验变量一行未改**
（AFFONLY 共享几何 / INTERPC `λ_int·L_interp` / IDGATE 恒等锚定强度门 `u` /
G4D 4D 高斯 Schur 补切片 / QDUAL 双条件解码器），判据表、三负控制、
12 个预注册键、退化守卫阈值、Adam / cosine / 无 warmup / wd=0 均未动。

### 18.1 共用层：`q3vl/whatb/caliber.py`（新增，唯一实现）

四项口径写在一处，五个 runner 调用，**没有第二份**：

| 口径 | 来源（未复制，import） |
|---|---|
| 色批 `BATCH_SPLITS` / `FROZEN_BATCH_SPLITS` / `BASE_LR` | `q3vl.whatb.arms.carrier`（未改动该文件） |
| `--data` 与实测 n | `q3vl.whatb.splits.train_normal_rows` / `DATA_SOURCES` |
| L8 并集的 z | `run_carrier_arm.open_z_caches(..., extra_sources=...)` → `zcache.MultiZCache`（**EPR-030 在跑的同一条调用**） |
| 纯 L1 的运行时断言 | `q3vl.whatb.losses_l0.assert_l0_pure` |
| λ 阶梯 | `effective_lambda_hc/sparse` = `carrier.py:347`/`:351` 原式 |

`assert_steps_per_epoch(value, n_train, B)` 是五个臂共用的运行时断言；
`horizon_record(...)` 是五个臂写进各自 `run_setup.json` 的 `caliber` 块，
内部先跑那条断言再返回。

**未改动任何被在跑作业冻结 sha256 的文件**（`arms/carrier.py` /
`scripts/run_carrier_arm.py` / `run_epr030_arm.py` / `zcache.py` / `evaldata.py` /
`colorimetry.py` / `qdecoder.py` / `losses_l0.py` / `glut.py`；mtime 全部早于本轮，
逐个核对过）。`splits.py` 也未改（`train_normal_rows` / `color_texts_of` /
`L8_ZCACHE_ROOT` 已具备所需全部能力）。

### 18.2 逐臂改动与新增/放开的旗标

| 臂 | 纯 L1 | `--batch-split` | `--base-lr` | L8 数据 | `--dry-run` |
|---|---|---|---|---|---|
| AFFONLY | `--loss-level` choices `[3,4]` → `[1,2,3,4]`；`loss_terms` 改吃 `cfg.lambda_{hc,sparse}_effective` | 新增 `--batch-split`（共用表），`--batch-samples/--queries` 降为显式覆盖（default `None`） | 已有 | 新增 `--data` / `--zcache-root-l8` | 新增 |
| INTERPC | 已有（`arms/interpc.py:859-862` 本来就按 `loss_level` 门控） | choices `("32x256","64x128")` → `K.batch_split_choices()` | 新增 `--base-lr`（与 `--lr` 同 dest） | 新增 | 新增 |
| IDGATE | `--loss-level 1` 已在 choices；`glut_loss` 改吃 `_effective` | 写死的 `("32x256","8x64","2x16")` → `K.batch_split_choices(SMOKE_BATCH_SPLITS)`（17 档 + 两个 smoke 档） | 新增 `--base-lr`（同 dest `lr`） | 新增 | 已有 |
| G4D | 新增 `--loss-level {1,3,4}`（default `None` = 旧派生 `4 if w_img else 3`）+ `effective_lambdas(args)` | 自由字符串改走 `K.parse_batch_split` | 新增 `--base-lr`（同 dest `lr`） | 新增（`ConditionStore(data=, zcache_root_l8=)`） | 新增 |
| QDUAL | 硬 `!= 3` 退出 → `not in (1,3)`；`qdual_losses(..., lambda_hc=, lambda_sparse=)` | 新增 `--batch-split`（`_Parser` 子类在 `parse_args` 里解析成 B/Q） | 新增 `--base-lr`（同 dest `lr`） | 新增（`ZStore(data=, zcache_root_l8=)`）；**旧 `--data {zcache,synthetic}` 更名为 `--z-source`** | 新增 |

其他被一并接上的点：

- 三处硬编码的 n 改成实测：INTERPC `config_from_args` 的 `-(-93934 // b)`、
  `interpc.as_dict()["train_normal_n"]`、AFFONLY `--total-steps` 默认 `117440` → `0`
  （= `epochs × ceil(n/B)`）。
- IDGATE `assert_frozen_organisation` 由「必须等于 EPR-024 的七个数」改成
  「只断言算术」（`colors_per_step == B*Q`、`steps_per_epoch == ceil(n/B)`、
  `total_steps == spe*epochs`），EPR-024 的七个数改成**记录**
  （`epr024_frozen` / `differs_from_epr024` / `step_matched_to_epr024`）。
  这与 `arms/carrier.py` 2026-08-16 早上放开 `BATCH_SPLITS` 的处理同形。
- INTERPC `InterpcConfig.__post_init__` 里的 `B*Q != 8192 → raise` 删除，改为记录
  （同上）；`batch_split` 在 AFFONLY / INTERPC / IDGATE 改成派生属性 `f"{B}x{Q}"`，
  名字与数对不上的可能性被消掉。
- AFFONLY 的 colorspan 取文本改走 `splits.color_texts_of`（L8 行的 `<color>` 在
  manifest 里内联，没有 record shard；原来的 `read_record` 在 L8 行上 `KeyError`）。
- IDGATE 的 B3 桶池过滤成「有 record shard 的行」（L8 行不出桶，原来会 `KeyError`）。
- 五个臂的 `run_setup.json` 都新增 `caliber` 块，含
  `data / data_sources / train_normal_n_measured / batch_split / batch_samples /
  queries_per_sample / colours_per_step / batch_split_step_matched_to_epr024 /
  steps_per_epoch / total_steps / base_lr / epochs / loss_level / pure_l1 /
  lambda_*_effective / assert_l0_pure / train_source_facts / z_cache_train`。

### 18.3 `--dry-run` 打印（五臂，同一条命令族）

`--batch-split 256x8192 --data v2seg+l8 --base-lr 1e-3 --loss-level 1`：

| 臂 | n_train | colours_per_step | steps_per_epoch | total_steps |
|---|---|---|---|---|
| AFFONLY | 119,828 | 2,097,152 | 469 | 18,760 |
| INTERPC | 119,828 | 2,097,152 | 469 | 18,760 |
| IDGATE | 119,828 | 2,097,152 | 469 | 18,760 |
| G4D | 119,828 | 2,097,152 | 469 | 18,760 |
| QDUAL | 119,828 | 2,097,152 | 469 | 18,760 |

n 是实测的：sft2seg train normal 93,934 + L8 normal 25,894 = 119,828
（`splits.train_source_facts`，每个源对自己的盘上声明断言；合并数只取 `len()`）。
AFFONLY / IDGATE / G4D / QDUAL 的 dry-run 走到了真实 z 缓存，`MultiZCache` 的
两个成员各自过 `assert_belongs_to`：
`zcache_v2seg/train.generated.none.zcache.pt`（n=93,934）+
`zcache_l8/generated/l8_train__none`（n=46,129），并集 140,063 个 sample_id。
INTERPC 的 dry-run 落在开缓存之前（只测种群与横轴）。

### 18.4 `32x256` / `64x128` 行为逐位不变的证据

`experiments/prs/EPR-030_shared-query-backbone/caliber_bit_identity.py`
（产物 `caliber_bit_identity_20260816.json`）：每个臂的 CPU 冒烟路径同一 seed 跑两遍，
A = 现在的代码，B = 用 monkeypatch 把 2026-08-16 之前的表达式装回去
（未门控的 `cfg.lambda_hc` / `cfg.lambda_sparse` + 字面 `text.split("x")` 解析），
再逐行比 `steps.jsonl`。**墙钟列**（`wall_ms_per_step` / `wall_s` / `wall_time_s`）
排除在比较之外（是计时器，不是结果），其余每一列都进 sha256。

| 臂/档 | 行数 | sha256(数值列) 新 = 旧 |
|---|---|---|
| affonly/32x256 | 2 | `89ffb1ca89b41950…` ✓ |
| affonly/64x128 | 2 | `67151576ec600f5c…` ✓ |
| interpc/32x256 | 8 | `63c819f9f7016df4…` ✓ |
| interpc/64x128 | 8 | `63c819f9f7016df4…` ✓ |
| idgate/32x256 | 3 | `9769ecc64c7b3556…` ✓ |
| idgate/64x128 | 3 | `6257288c27af76df…` ✓ |
| g4d/32x256 | 3 | `5ed43f80ef24abb8…` ✓ |
| g4d/64x128 | 3 | `96511ee60d5560bf…` ✓ |
| qdual/32x256 | 4 | `3bf27c11aefa64ac…` ✓ |
| qdual/64x128 | 4 | `6d699feaf24d7e28…` ✓ |

（INTERPC 两档 sha 相同，因为 `--stage self-test` 自己把 B×Q 覆写成 4×2048
`run_interpc_arm.py:1185`；那一档只证明本轮改动没动它的 loss 数字，不证明分档差异。）

算术侧的同一件事在 `q3vl/whatb/tests/test_caliber.py` 里另立断言：
`parse_batch_split(name) == tuple(int(v) for v in name.split("x"))`（全 17 档）、
两个冻结档在五个臂上仍是 `(B,Q)=(32,256)/(64,128)`、`B*Q=8192`、
`step_matched_to_epr024 is True`、`λ_hc=10.0`、`λ_sparse=0.001`、
`steps_per_epoch/total_steps = 2936/117440` 与 `1468/58720`。

### 18.5 pytest

`CUDA_VISIBLE_DEVICES="" /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests -q`
→ **720 passed**（改动前基线 701 passed）。

| 文件 | 通过数 |
|---|---|
| `test_affonly.py` | 65（原 63，+2 新增） |
| `test_interpc.py` | 65（原 64，改写 1 条 + 新增 1 条） |
| `test_idgate.py` | 79（原 77，改写 1 条 + 新增 3 条） |
| `test_g4d.py` | 57（不变） |
| `test_qdual.py` | 78（原 76，改写 1 条 + 新增 2 条） |
| `test_caliber.py` | 12（新增文件） |

改写的三条旧断言（旧口径已被 EPR-030 放开，原样保留会与新口径直接冲突）：
`test_interpc.py::test_batch_split_must_keep_8192_colours_per_step`、
`test_idgate.py::test_frozen_organisation_rejects_an_unmatched_split`、
`test_qdual.py::test_loss_level_4_is_refused_...`。三条都改成「记录而非拒绝」，
并各自补了「算术仍然拒绝」的对照断言。

### 18.6 复现命令行（`q submit` payload 形式，本轮**未提交任何作业**）

共用前缀：

```
PY=/home/bc/envs/q3vl_sft/bin/python
CKPT=/home/bc/data/runs/q3vl_base_sft_v2seg_20260814/checkpoint-4976
BANK=/var/cache/veradata/preset_bank_full
RUNS=/home/bc/data/runs/what_b
ZROOT=/home/bc/data/runs/whatb/zcache_v2seg
ZROOT_L8=/home/bc/data/runs/whatb/zcache_l8
EVALB=/home/bc/data/caches/whatb_eval_20260815/V_what
CAL="--batch-split 256x8192 --data v2seg+l8 --zcache-root-l8 $ZROOT_L8 --base-lr 1e-3 --loss-level 1"
```

EPR-025 AFFONLY：

```
q submit E030C_AFFONLY gpu0 /home/bc/data/logs/whatb-e030c-affonly-20260816.log \
  --desc "EPR-025 新口径 256x8192(2.10M 色/步), 18,760 步" --mem-peak 40 \
  --gate $CKPT --gate $ZROOT_L8/generated/l8_train__none/z.npy \
  --ready whatb_first_step --ready-timeout 7200 -- \
  env LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib \
      PYTHONPATH=/home/bc/VeraRetouch \
  $PY -u -m q3vl.whatb.scripts.run_affonly_arm \
    --run-dir $RUNS/whatb_AFFONLY_e030c --device cuda:0 \
    --base-checkpoint $CKPT --bank-dir $BANK \
    --readout seg_color --cond-dim 64 --n-gauss 48 --gen-width 128 \
    --share geo_opacity --global-affine affine --zero-init-heads --clamp two \
    --z-source cache --z-cache $ZROOT $CAL \
    --total-steps 0 --epochs 40 --eval-every 469 --log-every 50 \
    --quick-n 32 --interp-pairs 120 --ipb-paths 20 --repeats 8 --lib-rows 2500
```

EPR-026 INTERPC（三段，`--stage setup` → `train` → `eval`，`$CAL` 每段都带）：

```
$PY -u -m q3vl.whatb.scripts.run_interpc_arm --out-root $RUNS --run-name whatb_INTERPC_e030c \
  --checkpoint $CKPT --bank-dir $BANK --z-cache $ZROOT --device cuda:0 \
  --readout seg_color --context generated --cond-dim 64 --n-gauss 48 --gen-width 128 \
  --gen-mode full --clamp two --lut-resample none \
  --interp-weight 1.0 --interp-alpha beta0.5 --interp-where post_pi \
  --interp-target gt_mix --interp-stage joint $CAL --stage setup
# 同前缀 --stage train --epochs 40 --eval-every 469 --select-samples 64
# 同前缀 --stage eval  --eval-pairs-per-source all --ipb-k 20 --publish
```

EPR-027 IDGATE：

```
$PY -u -m q3vl.whatb.scripts.run_idgate_arm \
  --run-dir $RUNS/whatb_IDGATE_e030c --device cuda:0 --precision bf16 \
  --checkpoint $CKPT --bank-dir $BANK --split train --eval-split V_what \
  --context generated --z-cache $ZROOT --eval-bundle $EVALB \
  --n-gauss 48 --cond-dim 64 --gen-width 128 --clamp two --lut-resample none \
  --gate --gate-u-source sample --gate-u-dist uniform01 --gate-clamp after \
  --gate-p-end 0.2 --gate-ku 4 --gate-lambda-sign fixed --gate-null-prompt keep \
  --gate-field-src gt $CAL \
  --epochs 40 --eval-every 469 --quick-eval-n 64 \
  --interp-pairs 64 --strength-n 64 --library-size 1137 --n-repeats 8
```

EPR-028 G4D：

```
$PY -u -m q3vl.whatb.scripts.run_g4d_arm \
  --out $RUNS/whatb_G4D_e030c --device cuda:0 --precision bf16 \
  --base-ckpt $CKPT --lut-bank $BANK --z-cache $ZROOT \
  --train-split train --eval-split V_what \
  --carrier glut4d --glut4d-mode joint --glut4d-field gt --glut4d-n 48 \
  --glut4d-marg-norm full --glut4d-rot-sign paper \
  --clamp two --cond-dim 64 --gen-width 128 --readout seg_color \
  --lut-resample none --w-img 0 --alpha-s 0 --alpha-m 0 $CAL \
  --epochs 40 --quick-eval-every 469 --quick-eval-n 32 --lib-sample 1137
```

EPR-029 QDUAL（注意 `--z-source`，见 18.7）：

```
$PY -u -m q3vl.whatb.scripts.run_qdual_arm \
  --out-root $RUNS --run-name whatb_QDUAL_e030c --device cuda:0 --precision bf16 \
  --base-checkpoint $CKPT --bank-dir $BANK \
  --z-source zcache --zcache-dir $ZROOT \
  --train-split train --eval-split V_what \
  --bucket-pool-cache $RUNS/bucket_pools_train.json \
  --rung c --n-gauss 48 --decoder-layers 4 --decoder-width 256 --decoder-heads 8 \
  --z-expand proj --z-expand-k 4 --field-source m_low --field-kind gt \
  --field-grid 32x48 --zero-init-head --clamp two --readout seg_color $CAL \
  --epochs 40 --total-steps 0 --quick-eval-every 469 --eval-every 469 \
  --quick-eval-n 32 --log-every 50 --lib-sample 2500 --baseline-repeats 8
```

### 18.7【待决策，未拍板】三件必须由主 agent 裁定的事

1. **`--base-lr` 的取值。** 上面的命令写的是 `carrier.py:404` 的默认 `1e-3`。
   在跑的 EPR-030 作业里两种值都有（2026-08-16 13:2x `q ls` 的三条 Running：
   `--base-lr 1.6e-2` 与 `--base-lr 1e-3` 并存）。
   「与 EPR-030 相同的口径」指哪一个，本轮不替主 agent 定。

2. **wave 载荷 `/home/bc/agent-gpu-queue/waves/whatb_epr024_029_arm.sh` 需要同步改。**
   它把旧口径写死在里面（`--batch-split 32x256`、`--loss-level 3`、
   AFFONLY `--total-steps 117440`、QDUAL `--batch-samples 32 --queries 256
   --total-steps 117440`、五臂 `--eval-every 2936`）。其中 **QDUAL 那一档现在会
   直接报错退出**：脚本里的 `--data zcache` 已更名为 `--z-source zcache`
   （argparse 会打 `invalid choice: 'zcache' (choose from v2seg, v2seg+l8)`，
   提交即失败，不会静默跑错）。
   **本轮没有改这个文件**：两条 EPR-030 作业（`E031_MLP_NOL8` / `E031_MLP_LR1E3`）
   此刻正在执行同一个 bash 脚本，运行中改 bash 脚本会破坏正在执行的进程。
   等两卡空出来再改。

3. **`--eval-every` / `--quick-eval-every` 从 2936 改成 469 的口径。**
   上面的命令按「一个 epoch」换算成 469（B=256）。若要与 EPR-024 板做步数匹配的
   配对 Δ，评测点的步号也要匹配，这是主 agent 的编排决定。

### 18.8【如实上报，未自行改结构】新色批下五臂的结构性观察（**未做任何显存实测**）

本轮禁跑 GPU 作业，下面只有形状与计数，**没有一个 GiB 数字**：

- **IDGATE**：`n_pairs_per_step = colours_per_step × K_u`。`--gate-ku 4`（该臂主档）
  在 `256x8192` 上是 **8,388,608** 个 `(x, u)` 对/步，是其余四臂的 4 倍。
  这是该臂自己的实验变量（`gate_ku`），未动，dry-run 的 `frozen_organisation`
  里已如实记下。
- **INTERPC**：`pairs_per_step` 默认 `= batch_samples`（`arms/interpc.py:321`），
  在 B=256 上是 256 对/步；插值流与 fit 流各自 Q=8192 色，
  即 `--interp-weight > 0` 时每步再多约 2,097,152 色的前向 + 两次 LUT 求值
  （`values_a` / `values_b`）。同样是该臂自己的变量，未动。
- **G4D**：`--alpha-s` / `--alpha-m` 的 4D 正则在 `17^4` 格上算，与 B×Q 无关；
  主档两者为 0，不进图。
- **AFFONLY / QDUAL**：未发现与色批耦合的额外形状。
- 已知的既有阻塞（与本轮无关）：G4D 与 QDUAL 的出板 required 表含 `field_pred`，
  盘上仍没有 where 臂的 `m_pix` 产物 —— 本轮 G4D 的 CPU 冒烟就是死在这一步
  （`CriterionNotComputed: 'field_pred'`，steps.jsonl 已完整落盘）。本轮没有改它。
  （wave 头注释里那条「INTERPC 退化守卫固定在第 49 步」已经过期：
  `run_interpc_arm.py:767` 现在是 `if not guard_done and (due or step + 1 == total)`，
  已经挂在首次 quick eval 上，与其余五臂同口径。）

---

## DATA-P45-ENABLE（2026-08-18）：数据集口径化（p45 过滤后索引接线）

### 落点

`q3vl/whatb/splits.py` 引入 `DatasetVersion` / `DATASET_VERSIONS`，两个口径：
`v20260804`（原始发布索引，train normal 93,934）与 `cut-p45`（默认，80,269）。
两者指向**同一批 shard**（`SFT2SEG_SHARD_ROOT = /mnt/nfs-ro/bc/data/datasets/sft2seg-20260804`），
索引行的 `members` 路径是绝对路径，所以过滤后的目录只有 `splits/`。
sha1 切分规则族未动，无一行换 split。

- `TRAIN_NORMAL_N` 不再是断言值，改为 `DATASET_VERSIONS["v20260804"].train_normal_n`
  （冻结块第 1 条的历史记录，供各 runner 的 `frozen_block` 记录使用）。
- `train_normal_rows()` 的断言改为按 `root` 解析口径、对该口径 `normal_n[split]` 断言；
  **未注册的 root 直接 raise**（没有实测 n 就没有可断言的对象）。
- 旗标：`--dataset-version {v20260804,cut-p45}`，默认 `cut-p45`，在
  `caliber.add_caliber_arguments()` 注册，八个 runner 同一拼法；
  `caliber.apply_dataset_version(args)` 在每个 `main()` 最开头调用（早于任何索引读取）。
- `run_setup.json`：`caliber.horizon_record()` 加 `dataset_version`（口径名 / root /
  每 split 的 n 与 normal n / 排除清单 sha256 / 规则文字）+ `l8_manifest`。

### 待决策（保守默认继续，未擅自拍板）

1. **只注册了两个口径**。盘上还有 `-bandcut60` / `-maskcut` / `-cut-p50` / `-cut-p75`，
   没有实测 n，因此没有注册；`--dataset-root` 指向它们会以
   `not a registered dataset version` 报错而不是静默跑。要用哪一个，把实测 n
   填进 `DATASET_VERSIONS` 即可。**未改动这些目录**。
2. **口径是进程级全局**（`use_dataset_version`），读过索引之后再切会 raise
   （测试可以 `force=True`）。选择理由：把 root 显式穿到 ~15 个 `load_index` 调用点
   会把改动面放大数倍；代价是它是可变全局态，靠 "读过就锁死" + run_setup 落盘挡住漂移。
3. **`MultiZCache.mean()` 按各 member 的缓存长度加权**，缓存仍是旧的 93,934 条全量。
   `--cond-trainmean` 这一路的均值因此仍是旧全量的均值，不是 p45 子集的均值。
   本轮**没有改**（不在任务范围），若要用 `--cond-trainmean` 出板需先决定。
4. **各 runner 的 `frozen_block` / `FROZEN` 记录仍钉在 `v20260804`**（93,934 / 2,936 /
   117,440），因为已出的板都是那个口径；`measured.train_matches_frozen` 在 p45 下为
   `false`，是如实记录不是失败。`run_g4d_arm.DEGENERACY_BINDING_MIN_STEPS`
   （= `FROZEN["steps_per_epoch"]` = 2936）因此行为不变。
5. **L8 不在剔除范围**：`l8_train.manifest.jsonl` 无任何 sft2seg 口径参与，两个口径下
   L8 normal 均为 25,894（dry-run 实测）。

### 本轮发现的两条必须由人决定的事

6. **`q3vl/whatb/jetlut/run.py` 的 LUT 池会跟着默认口径变**：它调用
   `train_lut_ids()` / `eval_only_lut_ids()` 且不传 `root`，默认口径切到 `cut-p45`
   后，`train_full` 由 **3,149 → 3,103** 个 `lut_id`，`t_lut_unseen` 池由
   **259 → 232**，`held_out`（bank 减 train）随之变大。**本轮没有改 jetlut**
   （EPR-032 的两个 sweep 作业正在跑，"进程启动后禁改源码"）。若 EPR-032 要继续用
   原池，需要在 `jetlut/run.py` 显式传 `root=DATASET_VERSIONS["v20260804"].root`。
7. **本轮改源码时队列并非空的**：任务卡写"两卡空闲、队列为空"，动手前 `pueue status`
   实测 279 条全 Done；随后 11:16:38 另有 agent 提交了 `JET_SWEEP_P1` / `JET_SWEEP_P2`
   （EPR-032 jetlut，GPU 0/1）。本轮对 `q3vl/whatb/**` 的写入发生在 11:49-11:51。
   已核实：这两个作业是单进程 `exec python -m q3vl.whatb.jetlut.run`，它依赖的
   `codec/lutcode.py` / `splits.py` 在 11:16 启动时就已 import 进内存，源码改动不影响
   在跑的进程；但这仍然是一次违反"进程启动后禁改源码"的时序，如实记录。

---

## DATA-P45-FIX（2026-08-18，清独立审阅 BLOCKED 的四个消费点）

### 改了什么

| 文件 | 改动 |
|---|---|
| `q3vl/whatb/caliber.py` | `--dataset-version` 改用 `_RecordExplicit` action，记录"用户是否真的敲了这个旗标"；`apply_dataset_version` 只在**显式给了 version 且与 root 不一致**时才判冲突 |
| `q3vl/whatb/scripts/build_zcache.py` | 注册 `--dataset-version`（走 `K.add_caliber_arguments`，同拼法/同 choices/同默认）；`apply_dataset_version` 在第一次 `load_index` 之前；`setup` 记 `dataset_version` / `dataset_root` / `dataset_version_facts` / `n_index_rows`；同样三个字段写进缓存 `meta.json` |
| `q3vl/whatb/scripts/run_carrier_arm.py` | 新增 `assert_zcache_dataset_version()`；`open_z_caches` 对每个 sft2seg split 的缓存（含 `MultiZCache` 每个 member）执行，结果落进 `run_setup.json.z_caches.*.dataset_version_check` |
| `q3vl/whatb/scripts/build_eval_bundle.py` | 注册 `--dataset-version` 并在 `load_index` 前 apply；口径三字段进 `plan`（因此 `--dry-run` 也打印）→ 经 `**plan` 进 `meta.json` |
| `tools/visualize_what_global.py` | 注册 `--dataset-version`，`render()` 先 `use_dataset_version` 再读索引；metadata 记口径 + `n_pool` |
| `tools/visualize_joint_where_what.py` | 同上；`manifest.json` 加 `dataset_version` / `dataset_root` / `dataset_version_facts` |
| `q3vl/whatb/arms/affonly.py` | `total_steps` 默认改为哨兵 `0`，`__post_init__` 由**本 config 自己的** `train_n`/`batch_samples`/`epochs` 派生；显式非零值（`--total-steps N` 冒烟路径）照旧生效 |
| `q3vl/whatb/tests/test_caliber.py` | 新增 12 条：默认口径、显式切换、读后切换拒绝、未注册 root/未注册名拒绝、`--dataset-root` 单独给、root+version 一致/冲突、四个入口旗标拼法一致、affonly 派生 |
| `q3vl/whatb/tests/test_zcache.py` | 新增 2 条：口径不匹配 `SystemExit`、无口径字段 `unknown` + stderr 告警、非 sft2seg split 不检 |

### 待决策 / 未做（不静默拍板）

8. **B1（`q3vl/whatb/jetlut/run.py`）本轮仍未修**。任务卡的条件是"若 `JET_SWEEP_P2`
   已结束再改"。实测：动手前 `JET_SWEEP_P2`（pueue 309, pid 3555801）在跑；收尾时它
   已 `Success`，但 **12:50:50 又起了两个新的 jetlut 作业**——pueue 311 `JET_SYNTH`
   （pid 3799490）与 pueue 312 `JET_SWEEP_P1_M7`（pid 3799630），二者都在跑
   `python -m q3vl.whatb.jetlut.run`，其中 `JET_SWEEP_P1_M7` 正是 `sweep --pool held_out`
   （即 B1 所述池定义 902/948 的那条路）。按"进程启动后禁改源码"，本轮不改该文件。
   NOTES 第 6 条的结论不变：要继续用原池必须显式传 `v20260804`。
9. **两个 visualize 工具的默认口径保持不变（`cut-p45`，与八个 runner 同）**。核实：
   `docs/assets/joint_*` 九个目录的 `manifest.generated_at` 全部 ≤ 2026-08-18 10:22:23，
   而默认口径翻转（`splits.py` mtime）是 11:49:27 —— 因此**已有的所有 joint 板都是在
   `v20260804` 下出的**，其 manifest 里没有口径字段（本轮新加的三个字段是空的）。
   现在要重画它们必须显式 `--dataset-version v20260804`。
   本轮**没有改这两个工具的默认值**（不改变默认行为），是否应把 viz 工具的默认钉回
   `v20260804` 以让旧板"零参数可复现"——留给人决定。**未改动 `docs/assets/**`。**
10. **老 z 缓存（`/home/bc/data/runs/whatb/zcache_v2seg/*.zcache.pt`）没有口径字段**，
    新的消费侧闸门把它们判为 `unknown` 并向 stderr 响亮告警、不当成匹配、不阻断。
    要变成硬失败需要先重建缓存（`build_zcache.py --dataset-version <口径>`），
    或人工确认这批 `.pt` 是哪个口径 —— 留给人决定。
11. **`assert_zcache_dataset_version` 只检 `split in splits.SPLITS` 的缓存**；
    `l8_train` 等非 sft2seg split 不检（它们的行来自自己的 manifest，无 sft2seg 口径）。
12. NOTES 第 3 条（`MultiZCache.mean()` 仍是旧全量均值）、审阅 NOTE 3
    （`excluded_sha256` 是常量、运行时不核盘）、NOTE 6（`test_g4d.py:52` 的挂载门）
    本轮**未动**，不在任务卡范围。
