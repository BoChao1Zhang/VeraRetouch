# RO-W · 实施前核实记录 / 假设清单 / 待主 agent 决策

实验编号 **RO-W**（新增臂，VLM → 14 维基系数读出）｜ 日期 2026-08-03 ｜ 编码 subagent
git commit（起始）：`bc888fe147af76f2f8ef0299c85de856b9cd4adf`

---

## 一、读过的文档节（只读引用节）

| 文档 | 节 | 取用的事实 |
|---|---|---|
| `docs/PLAN_v2_local-retouch_2026-07-31.md` | §1.3 掩膜基底（L58–L75） | `q = w₀ + α·(w_dir·φ_dir)`、`‖w_dir‖=1`、`α≥0`、`s = 3·tanh(q/3)`；φ = [1] ⊕ [x,y,P₂(x),P₂(y),xy]（Legendre，中心化/除**短边**）⊕ [L,S] ⊕ [e₁..e₆]；**α=0 精确退化全局**、α 初始≈0；语义基逐图标准化 + 对几何块残差正交化；禁 softmax 竞争 |
| 同上 | §1.5 第 1.5 级（L198–L213） | 全局常数类判据 `α<1e−2 且 std<0.01`；L-BFGS + soft-IoU 拟合口径；Gram 条件数目标 <10 |
| 同上 | §红线速查（L464） | 禁逐图归一化 s ｜ 基底禁 softmax ｜ IoU 禁当优化目标 |
| `docs/EXPERIMENTS_v3_2026-08-02.md` | §2.1 RO 臂表 | RO-1..RO-9 分工；RO-2 的**刻度标准差 <0.15 且 AUC≥0.70**；RO-5 三档监督（GT/蒸馏/端到端）；每臂交付 = s 缓存 + 质量报告（AUC / 跨图刻度方差 / 符号）+ 接 RD-STD 的 L2 |
| 同上 | §3.2 MB-1..MB-5 | MB-1 = 单轴 14 维基底本身的表达力（离线），RO-W 是它的**读出端**：MB-1 问"基底画不画得出"，RO-W 问"VLM 说不说得出这 14 个数" |
| 同上 | Changelog 全部（3 条） | 2026-08-04 条：G1 Gate D1 **FAIL**（ρ_region_opp 0.884、shuffle 0.895 反超同义 0.842）；**被证伪的是读法不是假设**；表述纪律 D-31：判据 PASS 必须区分"未触发死刑"与"获得正面证据" |
| 同上 | 2026-08-03 深夜条 | **shuffle 前置判别为硬规则**：跨图打乱指令的对照批缺席则任何正向解读不成立 |
| `docs/DATA_ASSIGNMENT_2026-08-02.md` | §1.1/§1.2 split 纪律 | S-split 只读 T1 冻结旁表；禁 ad-hoc 切分 |
| 同上 | §2 D-CONSTRUCT 条目 | train 套 = S-train 源生成、val 套 = S-val 源生成，两套独立 |
| 同上 | §3.3 RO 系数据行 | 零训练臂统一评分集；RO-5 训练档 = D-CONSTRUCT(S-train) GT 掩膜 |
| `experiments/E2_basis_fit_20260803/STATUS.md` + `job.marker` | 全文 | **读取前确认了产物新鲜度**：第一回合 `results.jsonl`（4,344 fits，`FULL_RUN_DONE` 2026-08-03 03:24）已定稿；补件轮（`results_gauss/constrained/sweep.jsonl`）**正在被另一 agent 重跑**（15:12 与 15:52 两次重排），因此 RO-W **只引用第一回合的 main14/monotone 结果与 e2lib.py 的基底实现**，不引用任何补件轮数字 |
| `experiments/G1_s_identifiability_20260803/REPORT.md` | AUC 口径节（L143–L169、L200–L232） | AUC = **归一化 Mann-Whitney U**，**必须同时报 luma 基线**（不报基线无法排除"主体恰好比背景亮"）；G1 实测 AUC_light 0.664（local 段 0.690、L20–22 区间 0.825）、AUC_luma 0.475 |
| `experiments/RO9_layer_verdict_20260804/REPORT.md` | 判据量节 | `n` = 指令 A 的场把 A 区（正）与 B 区（负）分开的 AUC，阈值 **0.65**；RO-9 建议把 `n` 提升为 RO 系通用判据 |
| `tools/readout/ro9_gl_attention.py` | 全文 | 模型加载（`VeraRetouchForCausalLLM_Unified`，`attn_implementation="eager"` 硬守卫）、`<retouch_light>` id **151646**、prompt 构造（`data/infer_dataset.py:141` 同款 style 模板）、image span 定位（`model.lens_track_spans`）、24 层 / 256 image token / hidden 896 |
| `tools/scache/README.md`+`api.py` | 全文 | arm 目录约定 `<root>/<arm>/<img_id>__<instr_hash>.npy` + `.meta.json`，meta 必含 img_id/instr_hash/shape/dtype/layer/norm/created_at/arm/arm_version |

### 关于「AUC_target」口径的一处澄清（重要）

任务卡要求「必报 AUC 与 **AUC_target**（G1 口径）」。**全仓库 grep 不到 `AUC_target` 这个标识符**
（`rg -rn "AUC_target|auc_target"` 命中 0 处代码 / 0 处文档）。G1/RO-9 里与之对应的两个量是：

1. `AUC_light`：s 场对**目标掩膜**的归一化 Mann-Whitney U（+ luma 基线）——本报告记作 **AUC**；
2. `n`（RO-9 定义、RO-9 建议升为 RO 系通用判据）：**指令所指区域 vs 另一区域**的判别 AUC，
   阈值 0.65——这才是"指令条件性"的判据量。本报告记作 **AUC_target**，在 **L6（唯一带
   `maska`/`maskb` 双区域的级）** 上按 A\B vs B\A 计算。

若主 agent 的 AUC_target 另有所指，请指出，评测脚本 `evaluate.py` 里这一项是独立函数
（`R.auc_pool`），改口径是一行改动。

---

## 二、在线核实记录（DOSSIER 附录 B 之外的外部事实）

本臂用到的外部事实很少（模型内部结构全部由本仓库代码核实，不引用外部数字）。唯一引用外部
文献的地方是「VLM 输出结构化低维参数」的既有做法，逐条打开原始来源核实：

| 断言 | 来源 | 核实结果 |
|---|---|---|
| LISA 用扩充词表的 `<SEG>` token，以「embedding-as-mask」范式从 LLM 隐状态出分割 | arXiv **2308.00692** | ✅ 打开原文摘要核实：`"We expand the original vocabulary with a <SEG> token and propose the embedding-as-mask paradigm to unlock the segmentation capability."` 标题 *LISA: Reasoning Segmentation via Large Language Model* 一致 |
| 存在「控制 token 隐状态 → 轻量 MLP → 连续数值」的成文范式（不是我方独创） | arXiv **2511.11239**（GEODE） | ✅ 打开原文核实：标题 *Beyond Flatlands: Unlocking Spatial Intelligence by Decoupling 3D Reasoning from Numerical Regression*，作者 Zhongbin Guo 等，2025-11-14 提交（v2 11-18）；确有 **Direct Regression Head (DRH)**，`"'Embedding-as-Value' paradigm which routes specialized control tokens to a lightweight MLP for precise, continuous regression of scalars and 3D bounding boxes"`。**RO-W 的回归头 = 同一范式，输出换成 14 维基系数** |
| 「VLM 里的定位信号不在常用的固定条件层，而在随 prompt 变化的中间层」 | arXiv **2607.06445** | ✅ 打开原文核实：*Analysis-by-Proxy: Localization Signals in VLMs Operating as Condition Encoders*（Baron/Dorfman/Paiss/Cohen-Or/Patashnik，2026-07-07，ICML 2026 MI Workshop spotlight）；结论 `"the localization signal does not reliably propagate to the predefined layer configurations commonly used for conditioning"`，且位置 `"vary depending on the input prompt"`。**直接影响 RO-W 的设计：层号不能拍死，必须扫层**（见 D6） |

检索卫生：本轮 WebSearch 返回的其余 arXiv 号（2605.27737 / 2604.01206 / 2606.14703 / 2606.28127 /
2605.21642）**未打开核实，因此一律未使用、未引用**。

---

## 三、与任务卡的三处偏差（必须先说的坏消息）

### ① 任务卡说的「E2 拟合产物 w*」在本臂的训练集上**覆盖率 = 0%**

任务卡写「训练：`T4_construct/sanity/train/L0..L7` + **对应的 E2 拟合产物 w\***」。
实际核对 `experiments/E2_basis_fit_20260803/results.jsonl`（4,344 行）：

- E2 的 `mask_id` 全部形如 `semantic_0062` / `linear_0003` / `ring_0121` / `constant_0009`；
- 其 `feat_key` 形如 `sem_batch-0003_000454_candidate_...` / `pool_012_...`；
- 掩膜来源是 **E2 自己 `prep_data.py` 现造的**：4 个几何合成族（linear / radial_ell / ring /
  wedge，各 200）+ 常数 20 + 从 D-MASKBANK（S-val 源，`slot_id=semantic-*` 的 `.cgt.png`）
  抽的 200 个语义掩膜，统一 512²。
- **和 T4_construct 的 L0–L7 样本没有一个 uid / mask 是同一个东西**（T4 的掩膜是 L0 全一、
  L1–L3 语义、L4 几何四族、L5 错配、L6 双重叠、L7 同色区横切，且各自绑定具体源图）。

⇒ **不存在「对应的 E2 拟合产物 w\*」**。处置（保守默认，已执行）：**用 E2 的同一套代码
（`e2lib.py`，逐字复用基底/正交化/L-BFGS/soft-IoU）在 T4_construct 的 train+val 掩膜上重新
离线拟合 w\***，作为 (a) 档的标签与离线上界。E2 的 0.975 是**在 E2 自己的掩膜集上**的数字，
本报告把它当**外部参照**而不是本臂的上界；本臂的上界是我们自己重拟合出来的那条线。

### ② 任务卡说「预测 14 维 w」，但 E2 的读出头有 2 个自由参数

E2 的 monotone 读出是 `m = σ(g·s + b)`，(g, b) **逐掩膜自由拟合**（实测 g 中位 6.88）。
若 RO-W 也让 (g, b) 逐图自由，**跨图刻度这条主张当场作废**——每张图都可以自选 s 的刻度。
所以 RO-W 把读出**固定为全局共享的 `m = σ(g₀·s)`，g₀ 是一个常数**，VLM 只输出 14 个数。
g₀ 的取值是一个**校准量**，见 D1。

### ③ T4_construct **没有自然语言指令**

manifest 里只有 `transform`（类别/符号/幅度）、`mask_kind`、`geom_params`、`class5` 需自算。
没有指令就没有 shuffle 对照，整条判据链断掉。处置见 D4（模板化生成，确定性可复现）。

---

## 四、待主 agent 决策（两种做法都合理、影响后续；已采用保守默认继续）

### D1 · 固定读出的斜率 g₀（**已按预注册程序在 TRAIN 上校准，不看 val**）

- 两难：g₀ 太小 → 要画硬边掩膜只能让 ψ 饱和，实测 α 冲到 **1.27e4**，s 场退化成二值蜡纸、
  端到端梯度死在 tanh 尾巴上；g₀ 太大 → 过渡带比渲染器自己的 s 轴分辨率还窄
  （M=12 个原语均布 [−3,3]，格距 0.545；sigmoid 10–90% 宽度 = 4.4/g₀），读出变成渲染器
  画不出来的东西。
- 保守默认：**g₀ = 6.0**（过渡带 0.73 ≈ 1.3 个格距；且与 E2 自由拟合的 g 中位 6.88 接近，
  固定它的代价小）。校准全过程与各档数字见 `config/calib_readout.json`，只用 TRAIN 样本
  （每级 4 张 × 8 级 = 24 张，逐档跑「固定读出」与「E2 自由 (g,b) 读出」两次拟合）：

| g₀ | 固定读出 soft-IoU | 自由 (g,b) soft-IoU | **gap 中位** | gap p90 | α 中位 | **α p90** | α(L0) 中位 |
|---|---|---|---|---|---|---|---|
| 2 | 0.8786 | 0.9127 | 0.0213 | 0.0657 | 73.5 | 7673 | 5.0e−4 |
| 4 | 0.9122 | 0.9127 | 0.0012 | 0.0119 | 24.9 | 11083 | 2.5e−4 |
| **6** | **0.9129** | 0.9127 | **0.0000** | **0.0086** | 15.9 | **450** | 1.7e−4 |
| 8 | 0.9130 | 0.9127 | 0.0001 | 0.0084 | 13.6 | 2110 | 1.3e−4 |
| 12 | 0.9200 | 0.9127 | −0.0000 | 0.0105 | 9.8 | 2710 | 8.3e−5 |

  **预注册的 g₀=6.0 被校准数据支持**：g₀≥4 起「固定读出」相对「逐图自由 (g,b)」的代价
  已经归零（中位 gap 0.000，p90 0.009），而 g₀=6 的 α 尾巴最短（p90 = 450，其余档 2000–11000），
  回归条件数最好。**注意这条只是「未触发死刑」**：它说明固定读出不比自由读出差，
  不说明固定读出是"对的"。
- 请主 agent 确认：是否接受"读出固定"这一设定本身。若主 agent 认为读出应当也由 VLM 输出
  （即 16 个数而非 14 个），跨图刻度节需要改判据口径，请明示。

### D2 · w* 标签的重拟合（见上文 ③①）

保守默认：**重拟合**，并在报告里把 E2 的 0.975 标注为"E2 自己掩膜集上的数字，非本臂上界"。
替代做法是把本臂的训练集换成 E2 的掩膜集——但那样就不是 D-CONSTRUCT，也没有 L0–L7 阶梯、
没有 L6 双区域（AUC_target 就没地方算），故未采用。

### D3 · 不做生成（prefill-only 读出）

RO-9 是 `generate` 到 512 token 再 teacher-forced 回读，实测 40–60 s/样本；RO-W 需要
~5,800 次前向（3 个指令档 + L6 双区域档 × train/val），按 RO-9 的做法要 **60+ 小时**。
保守默认：**把三个 retouch token 直接接在 assistant 轮头之后做单次 teacher-forced prefill**
（≈0.3 s/样本）。这正是 RO-9 自己的 `fallback` 分支（其 D6），只是从"偶尔用"变成"总是用"。
风险：token 的隐状态不再处在模型自己生成的上下文里。若主 agent 认为这会实质改变表征，
需要重跑一个 generate 版的小对照（~200 样本，2–3 h）——**未做，请拍板**。

### D4 · 指令是模板生成的，不是人写的

- L0：`Please {verb} the whole photo.`
- L1/L2/L3/L5（语义）：`Please {verb} {noun}, and leave the rest of the photo alone.`
  其中 `{noun}` 来自 **CLIP 锚点 argmax**（sky/skin/foliage/water/architecture/subject，
  与 E2 `prep_data.py` 的 `class5` 判定同一套代码）。
- L4/L6/L7（几何）：位置词取 3×3 九宫格 + 大小档 + 形状名（`geom_params` 反推），
  **刻意用粗粒度词**（"top-left 的小圆块"），不写坐标数字，否则任务退化成解析数字串。
- 影响：几何级上的成功**只证明"读指令→出几何系数"**，不证明视觉 grounding；语义级上的
  成功才需要把词落到图上。报告里分开写，不混。
- 请主 agent 决定：是否要把 D-SFT-L 的真实指令接进来重跑（更贵，需要重建 T4 样本↔build
  候选的对应关系；T4 manifest 里有 `provenance.sample_id`，理论上可回取 `vrmeta.region`，
  但 `vrmeta` 里只有 `region: "lower"` 这种粗标签，**没有完整指令句**）。

### D5 · 第二种读出头选了 cross-attention query token，而不是"复用 3-latent 改低维"

理由：3-latent 是 **what 线**（颜色）的载体，另有实验在优化它；在 where 线上复用同一结构会
把两条线的结论混在一起。cross-attention query 的动机是可证的：拟合 Legendre 系数 = 求空间
矩，而 `attention 权重 × 位置特征` 正好能算加权空间矩，所以这个头**在原理上够得着**离线
L-BFGS 的解；而 mlp 头对 image token 只做 mean-pool，**空间布局信息被抹掉**，它给出的任何
几何系数只可能来自指令文本——两个头因此构成一条干净的消融轴（"读出需不需要看清位置"）。

### D6 · 读出层的选择

外部核实的 arXiv 2607.06445 明确说定位信号**不在常用固定层**且**随 prompt 变化**。
保守默认：缓存 5 个层（8/11/14/20/22）的全 256 token + 全 24 层的 pooled 与 token 隐状态，
**用 TRAIN 上的表现选层**（不看 val）。若主 agent 要求全 24 层全 token 缓存，磁盘 ~45 GB，
可以做但要先批预算。

### D7 · checkpoint 选择

红线禁用 val loss。保守默认：**用 val AUC（排序类指标，非训练目标、非 IoU）选点**，同时
`train.json` 里保留 last-epoch 数字，报告两者并排。若主 agent 认为 val AUC 也算"用 val 调
参"，改成固定 epoch 预算是一行改动。

### D9 · 拟合器换成 GPU 批量 Adam（**这是一处实现替换，必须审阅**）

`fit_labels.py` 是 E2 `e2lib.fit_mask` 的逐字复用（LSQ 起点 + centroid-radial 起点 +
随机重启 + torch L-BFGS strong-Wolfe + α 小者优先的 tie-break），但在这台机器上实测
**~120 s/掩膜**（load ~130/48 核，另有四个 agent 的作业），2,240 个掩膜要 **~9 小时**。
本臂的标签只是「离线上界 + (a) 档的回归目标」，不是科学结论本身，因此改用
`fit_labels_gpu.py`：**同一目标函数（soft-IoU）、同一参数化、同一起点构造**，
把所有掩膜一起放在 GPU 上用 Adam 批量优化（6 个重启，500 步，128² 拟合格）。
实测 **~1.4 s/掩膜**。

两处必须说明的实现细节：
1. **tie-break 照抄 E2**：先取 loss 最小，再在 loss 落后 ≤1e−3 的重启里取 ‖v‖ 最小者；
2. **α=0 精确可达（PLAN 1.3 硬要求）**：在选定方向上做 γ∈[0,1] 的 21 点收缩线搜索，
   取「loss 仍在最优 1e−3 以内」的最小 γ。全局掩膜（L0）上 γ 直接落到 0，
   **实测 L0 的 α 精确等于 0，soft-IoU = 1.0000，24/24 满足 α<1e−2**；真实掩膜上 γ 停在 1。

`--validate` 会在若干掩膜上同时跑 CPU L-BFGS 参照并把差值写进
`config/fitter_validation_*.json`；**一个没被对拍过的重实现不算证据**，这条数字进 REPORT。

### D10 · 标签的 α 上限（`--alpha-cap 100`）

train 上 w\* 的 α 尾巴很长（p50 = 11.2，p90 = 144，**p99 = 1.2e4，max 3.4e4**）：
这些是需要极硬边的掩膜，固定 g₀ 下只能靠把 ψ 推到饱和来实现。直接拿来当回归目标会让
标准化后的 target 差出三个数量级，梯度全被少数样本占住。保守默认：**把标签的 ‖v‖ 截到 100**
（实测代价：离线 soft-IoU 中位 0.9085 → 0.8957，**−0.0128**，13.8% 的样本被截）。
截断只作用于**标签与输出标准化**，**评测用的离线上界不截**。各档代价：

| cap | 离线 soft-IoU 中位 | Δ | 受影响样本 |
|---|---|---|---|
| 不截 | 0.9085 | — | 0% |
| 500 | 0.8992 | −0.0093 | 10.2% |
| 200 | 0.8965 | −0.0120 | 11.9% |
| **100** | **0.8957** | **−0.0128** | 13.8% |
| 50 | 0.8937 | −0.0148 | 16.6% |
| 30 | 0.8886 | −0.0199 | 21.7% |

### D11 · 标签文件里有 224 个重复 (uid, mask_key)（**必须报告的操作事故**）

拟合任务重启时，一个旧的 `run_fit_labels.sh` shell 没被前一次 `pkill` 干掉，
**与新 shell 并发跑了一段 train 拟合**，两边都往 `labels_train.jsonl` 追加。
后果与处置：

- 实测 **1,056 行 / 832 个唯一键 / 224 个键重复**；**0 行损坏**（逐行 flush 写入是原子的，
  已用 `json.loads` 逐行校验）。`labels_val.jsonl` 240 行 **0 重复**，未受影响。
- 两轮拟合用的是同一目标函数与同一参数化，差别只在优化步数（500 vs 900 步/相）。
- 处置：`train_head.load_labels` 改为**按 soft-IoU 取更好的一行**（等价于"多一轮重启"，
  与拟合器自己跨重启的 tie-break 同一规则），**与写入顺序无关**（不是 last-write-wins）。
- 旧 shell 还**提前 `touch` 了 `FIT_LABELS_DONE`**；已发现并清除，链式脚本当时仍在等
  VLM 缓存，没有被误触发。教训写进 job.marker。

### D12 · 标签实际上是「CPU L-BFGS 与 GPU 批量拟合逐掩膜取优」的并集（**第二起操作事故，但结果是好的**）

承 D11：那个没被杀干净的旧 shell 不止提前 touch 了完成标志，它还在后台**跑满了一小时的
CPU L-BFGS 版拟合**（`fit_labels.py --split train --workers 8`，即 E2 `e2lib.fit_mask`
的逐字复用版），持续往 `labels_train.jsonl` 追加。发现时已产出大量行。

**处置与后果（不是掩盖，是记录）**：
- `load_labels` 的合并规则是**逐 (uid, mask_key) 取 soft-IoU 更高的一行**，与写入顺序无关，
  所以结果是良定义的：**两种优化器的逐掩膜取优**。
- 实测最终 2,000 个唯一键里，**1,668 个由 CPU L-BFGS 胜出、332 个由 GPU 批量拟合胜出**
  ——即 CPU L-BFGS 确实更强（这与 D9 的对拍结论一致：GPU 版中位低 0.021）。
- **离线上界因此从 GPU-only 的 0.9085 提升到 0.940**（S-train 逐级：L0 1.000 / L1 0.882 /
  L2 0.900 / L3 0.856 / L4 0.996 / L5 0.884 / L6 0.938 / L7 1.000）。
- 发现后**已 kill 该 CPU 作业**（只 kill 自己的 `fit_labels.py --split train`），
  并**用同样的 CPU L-BFGS 补跑了 val 侧 240 个掩膜**（`labels_val_cpu.jsonl`），
  使 train / val 两侧的标签质量口径一致；GPU-only 版本保留为 `labels_val_gpuonly.jsonl`。
- 另一处连带问题：`run_chain.sh` 的等待条件用了
  `! pgrep -f "cache_vlm.py --split train"`，而我自己开的若干个**等待用 shell 的命令行里
  含有这个字符串**，被 pgrep 自匹配，导致链式脚本多等了约 40 分钟。已 kill 那些等待 shell。
  教训：守卫用的 pgrep 模式必须锚定（`pgrep -f '^python.*cache_vlm'`）或改用 pidfile。

### D13 · **读出口径按 RO-3 判决重做**（主 agent 2026-08-03 傍晚下发，任务卡原设计被推翻）

任务卡原文是「取某层的 `<retouch_light>` 隐状态 → 线性/MLP 头 → 14 维」。RO-3 的实测把这条否掉：

| 读法 | AUC_target | ρ_region_opp | 差分场 AUC |
|---|---|---|---|
| `<retouch_light>` 当 query（RO-9 canonical） | 0.479–0.526 | 0.884 | 0.5129 |
| **L11 第 5 头 + instruction 文本 token 当 query** | **0.7660** | **0.171** | **0.9298** |

`gl` 池在**全部 336 个 (层,头)** 上 max AUC_target = 0.5542，**低于置换零分布 q95（0.596）**。

**处置**：
1. **主档换成 instruction 文本 token 的表示**（`cache_vlm2.py` 新缓存 `instr_pool`(24,896)
   与 `instr_tok`(5,40,896)），`train_head.py --feat instr`；层扫描限定在已缓存的
   {8,11,14,20,22}，RO-3 的最佳层 L11 在内。
2. **原 retouch-token 档全部保留为负对照行**（`--feat retouch`，即任务卡原设计）。
   早期信号已经支持 RO-3：retouch 档的层扫描里 inner-val AUC 在 ep1（≈常数 w 初始化）
   最高、随训练**下降**（L2: best 0.859@ep1 → last 0.603），即那条 tap 上没有可学的空间信息。
3. **新增 `nonoun` 指令档**（同一编辑动词、**完全不含区域名词**，如「give the photo a
   moodier feel」），单列 soft-IoU / AUC / AUC_target。对标：RO-1 的 CLIP 在无名词上 AUC
   不掉（0.907 vs 0.930）但 **AUC_target 塌到 0.523**——这一格决定 VLM 在 where 线上
   是否可被免费 CLIP 替代。
   ⚠ **限定**：D-CONSTRUCT 的掩膜（尤其 L4/L6/L7 随机几何）与图像内容无关，
   **无名词指令在原理上无法确定区域**，所以这一格在本数据上只能读作
   「模型有多依赖区域词」，**不能**读作「无名词指令下模型仍能对」。后者需要
   D-SFT-G 风格指令 + 真实区域 GT，属下一步。
4. **Δ_shuffle 的对标值**（RO-3 给出）：L11H5 单头 +0.0733、融合 OOF +0.1079。
   RO-2 另外证明了「把 Instruction 移到 <image> 之前」这条路无效（12 条判据全在 ±0.006
   内），并提出**指令条件性可能是要「训出来」而非「读出来」**——RO-W 是第一个真正在训练
   这件事的臂，因此 **Δ_shuffle 在本臂是验收项，不是对照项**。

### D14 · 蒸馏档 (d) 仍然跑不了（不是因为没等，是因为 key 空间不匹配）

主 agent 指出 `/var/cache/veradata/scache/ro3-fused/`（772 条）可直接当蒸馏目标。**实际核对
后不成立**：该 arm 的 772 条里只有 **44 条**是 `L1_val_*`（D-CONSTRUCT val 的一小部分），
其余是 G1 的源级 key（`ppr10k_*` 等）+ 各自的指令哈希。RO-W 需要的是
**S-train 的 1,600 条 D-CONSTRUCT 条目**，RO-3 没有产出。
⇒ (d) 档代码已就位（`--sup distill --distill-dir`），**要么请 RO-3 在 D-CONSTRUCT S-train 上
补跑读出，要么这一档留到下一轮**。域坑已记：`ro3-fused` 域 [−3.256, 8.805]、58.8% 为负，
`upsample.py` 默认 `clamp=(0,1)` 会静默吃掉负值。

### D15 · 输出头的初始化必须是「中性 w=0」，不能是「最优常数 w」（实测坑）

原设计把输出层的 shift 设成**训练集上拟合出的最优常数 w**，理由是「零初始化的头正好
等于 Δ_const 基线，Δ_const 因此可直接解读」。**实测这会让端到端档直接死掉**：那个常数
基线的掩膜几乎处处为 0（in-mask m ≈ 0.05），于是 `σ(g₀·s)` 一开始就贴在饱和轨上，
BCE 梯度只能靠把 α 顶上去——实测 **1 个 epoch 内 α 冲到 220、场塌成常数、inner-val
AUC = 0.500**。

改成 **shift = 0（中性：w=0 → α=0 → s≡0 → m≡0.5）**后，同一配置立刻开始学：
inner-val soft-IoU **0.254 → 0.423（ep10）→ 0.490（ep40）→ 0.498（ep50）**，
AUC **0.647 → 0.720 → 0.762 → 0.769**。`--init {neutral,const}` 已做成开关，
默认 neutral，`const` 保留并在 docstring 里写清它为什么会坏。
Δ_const 仍然独立计算（单个最优共享 w 在 S-val 上的 soft-IoU），不依赖初始化。

### D16 · 初始化修订带来的混杂已就地清理

D15 的中性初始化改动落在**层扫描跑到一半的时候**，因此有 7 个扫描项是旧（const）初始化
下训的。这会污染层选择（IoU 列尤其不可比）。处置：
- 旧初始化的 7 个 run 全部移到 `_init_confound/`（**保留不删**，可核）；
- `redo_sweep.sh` 用中性初始化把它们**逐个重跑**，层选择只在同初始化的结果里做；
- **主网格（`runs/`）全部在改动之后跑，不受影响**。
- 修订前的匹配初始化对照（同为 neutral、同层 L20）：
  **instruction tap 0.8871 vs retouch tap 0.8305**，方向与 RO-3 一致，混杂不改变结论。

### D17 · **给主 agent 的两处更正**（我上一轮报出的数字，其中一条已被自己的新数据推翻）

主 agent 打算引用的三条结果里，**两条需要改**：

**(a) 「预注册 0.85 在 L3 上高于 oracle」——这条现在是错的，请勿写进 EXPERIMENTS_v3。**
那是我用 **GPU-only 拟合**（离线上界 0.9085）时报的 L3 = 0.8149/0.856。
CPU L-BFGS 并入后（D12，上界升到 0.940），**两个 split 上没有任何一级低于 0.85**：

| split | ALL | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 |
|---|---|---|---|---|---|---|---|---|---|
| S-train (n=1600) | 0.9400 | 1.000 | 0.882 | 0.900 | **0.856** | 0.996 | 0.884 | 0.938 | 1.000 |
| S-val (n=192) | 0.9436 | 1.000 | 0.908 | 0.909 | **0.883** | 0.986 | 0.897 | 0.919 | 0.999 |

**实质建议仍然成立**（判据应相对各级 oracle 表述），但依据要换成
「L3 的余量只有 **0.006**，判据几乎贴着上界」，**不是**「判据不可达」。

**(b) 「α=0 精确退化」在 train 上精确，在 val 上不精确。**
- S-train：**200/200 α 精确等于 0.0**，soft-IoU 全部 1.0000；
- S-val：**0/24 精确为 0**，但 **24/24 < 1e-2**（max 1.667e-4），soft-IoU 全部 1.0000。

主 agent 原话「200/200 train、24/24 val < 1e-2, soft-IoU 1.0000」**是准确的**；
只要不把它简写成「val 上也精确为 0」即可。

**(c) 第一条（RO-3 复现）现已在同初始化下核实，可以引用。** 清理掉初始化混杂后的
逐层 inner-val AUC（全部 neutral init）：

| tap | L2 | L5 | L8 | L11 | L14 | L20 | L22 | L23 |
|---|---|---|---|---|---|---|---|---|
| **instruction** | — | — | 0.8751 | **0.9002** | 0.8947 | 0.8871 | 0.8870 | — |
| retouch | 0.8273 | 0.8242 | — | — | — | 0.8305 | 0.8399 | 0.8399 |

instruction tap **在 L11 达峰**（RO-3 的层），且在 AUC-最优 checkpoint 上的 soft-IoU
**0.306–0.359 vs retouch 的 0.000–0.181**。

### D18 · D4 已执行：D-SFT-L 主档数据就绪并验证

`build_sftl.py` 打通并核实了 journal 链路（与 RO-3 同一条）：

- 连接：`journal/<build>/sft.jsonl`（`sft_id` → **真实指令** `instruction` /
  `instruction_short` / `winner_confidence` / `local.subject` / `local.region`）
  × `batch/metadata.jsonl`（`sft_id` ↔ `sample_id`）
  × `batch/indexes/*.idx.jsonl`（→ shard/offset）× shard seek 直读 `.cgt.png`。
- 规模：6 个 `prod-l*` build × 前 4 batch，39,864 候选 →
  **3,521 条（S-train 3,302 / S-val 219）**，源多样性 **2,724 / 178**。
- 漏斗：非语义槽 31,941（几何槽不带真实区域掩膜）、`winner_confidence=low`
  **3,082 条按纪律排除**、无 journal 行 1,130、S-test 189（不取）。
- 核实：随机 5 条 seek-read `.cgt.png` 全部成功（真实软掩膜，mean 0.044–0.430）。

**⚠ 一处数据陷阱（D-25 的实例）**：journal 里记的 `I_in` 路径 **5/5 都已不存在**
（指向 `/mnt/ramstage/...` 等已清理的暂存区）。所以**不能用 journal 的 I_in 路径取图**，
必须走 shard 里的 `.in.jpg` 成员（`SemanticBank.load_image` 那条路，T4/E2 都用它）。
已加对齐核验：随机 40 条，`.in.jpg` 经 `exif_transpose` 后与 `.cgt.png`
**长宽比一致 40/40**；掩膜覆盖率中位 0.119，90% 落在 E2 的可用带 [0.02, 0.85]。

**下一轮 D-SFT-L 主档的执行顺序**（脚本全部可复用，只换样本源）：
`prep_feats.py` → `fit_labels_gpu.py`（+ CPU 对拍）→ `cache_vlm2.py`
（真实 `instruction`；shuffle 对照直接用**同 build 内跨图打乱真实指令**，
比模板 shuffle 更强；无名词档改用 D-SFT-G 的风格指令）→ `train_head.py --feat instr`。
**这一轮不需要再造指令**，因此 D4 里「模板从 GT 派生」这条限定在下一轮自动消失。

### D8 · 监督第四档（蒸馏）已预留、未跑

按主 agent 补充指令：`train_head.py --sup distill --distill-dir <path>`，监督源做成配置项，
**没有写死**。等 G1b / RO-D 的 DiffLMM scache arm 落地后直接接。当前交付的是 (a)(b)(c) 三档。

---

## 五、假设清单（能自行核实的已当场核实）

| # | 假设 | 核实方式 | 结果 |
|---|---|---|---|
| A1 | `<retouch_light>` token id = 151646 | `ro9_gl_attention.py` 里的断言 + 实跑 | ✅ 断言通过 |
| A2 | 模型 24 层、hidden 896、image token 256（16×16） | 实跑 `hidden_states` 形状 + `lens_image_spans` 断言 | ✅ |
| A3 | 非方图走 `expand2square` **黑边 pad** 再 resize 1024²（不是 center crop） | `llava/mm_utils.py:186` + RO-9 `luma_to_grid` 注释 | ✅，token 网格坐标按此换算（`cache_vlm.token_xy`），padding 格子用 `valid16` 屏蔽 |
| A4 | T4_construct 的 split 来自 T1 冻结旁表 | manifest 每行 `split_rule: t1_side_table:splits.sqlite3(rows=33652)` | ✅ 未做任何 ad-hoc 切分 |
| A5 | L6 有 `maska`/`maskb` 两张区域掩膜（AUC_target 的落点） | `ls sanity/train/L6` + manifest `files` | ✅ 200(train)/24(val) 组齐备 |
| A6 | 固定读出 g₀ 下 α=0 仍精确可达（退化性不被破坏） | L0 样本离线拟合实测 α = 5e-4，soft-IoU 0.9975 | ✅ |
| A7 | E2 的补件轮不会污染我引用的数字 | 只读 `e2lib.py` 与第一回合结论；`results_constrained.jsonl` 等一律未读 | ✅ |

---

## 六、资源与纪律

- 卡：**CUDA_VISIBLE_DEVICES=1**，每进程 `set_per_process_memory_fraction`（prep 0.06、
  VLM 缓存 0.12、训练 0.10），峰值远低于 15 GB 上限。
- **未 kill 任何非本任务进程**（RO-1/RO-2/RO-3/RD-G/E2 补件轮全部存活）。
- CPU：机器 load ~120–155 / 48 核，本臂 dataloader/worker 一律 ≤8，且不用 torch 多线程
  （`OMP_NUM_THREADS=1`、`torch.set_num_threads(1)`）。
- 长任务：`job.marker` 记 PID / 完整启动命令 / 日志 / 完成标志；已有产物的 run 自动 SKIP。
