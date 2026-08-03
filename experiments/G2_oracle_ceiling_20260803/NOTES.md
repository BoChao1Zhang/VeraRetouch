# G2 — 实施前核实记录 / 假设清单 / 待主 agent 决策

## 〇 读过的文档章节（只读引用节）

| 文档 | 节 | 取到的东西 |
|---|---|---|
| `CLAUDE.md` | 全文 | 交付物规范、红线速查、数据纪律、长任务 + job.marker |
| `docs/EXPERIMENTS_v3_2026-08-02.md` | L17 (H5)、L128/158 (分支树)、L176 (W1 排期) | H5 判死线 `Δ_ceil<3dB 或实际<1dB`；分支树 `G2 oracle <0.4dB → 停/换数据`；W1「三个数出齐」 |
| `docs/PLAN_v2_local-retouch_2026-07-31.md` | §4.1 差分分层法 Step1–7、§6 D0 行、L456 | 33³ 条件均值 / n<20 并 17³ / 17³×8 s 桶 / Δ_ceil 定义 / 退化子集 Δ<0.1dB / 带宽 Spearman>0.8 / 「今天三个数」Δ_ceil≥8dB、MLP 探针≥45dB |
| `docs/DATA_ASSIGNMENT_2026-08-02.md` | §3.1 G2 行、L39–41 (D-RENDER/D-SFT-G/D-SFT-L)、L29 | 三档定义、②−③ = H5 自家数据预登记数字、S/P split 纪律 |

## 一 在线/本地核实（外部事实一律追到原始来源）

本任务**没有引用任何外部 URL / 论文数字**，全部依赖仓库内已落盘资产。对这些资产逐一回到**源码或原始数据**核实（不采信文档转述）：

1. **shard 成员布局**（`sft/prod-l*/batch-*/indexes/shard-*.idx.jsonl` 直读）：
   `.in.jpg`（VLM 预览图）/ `.jpg`（归档 after = I_tar）/ `.cgt.png`（C_GT 掩膜）/ `.vrmeta.json`。
   无压缩 ustar，`offset_data + length` 可 seek 直读（沿用 `tools/scache/oracle.py` 与
   `tools/bgr_check/common.py` 两处已评审实现的同一约定）。
2. **`.in.jpg` 不能当 I_in（关键坑，实测发现）**：抽样 6 个候选，`.in.jpg` 尺寸为
   (1000,667)/(683,1024)/(2048,3640)/(540,360)…，而 `.jpg` 与 `.cgt.png` 恒为**短边 1024**
   的渲染分辨率。二者逐像素**不可比**。若强行 crop/resize 对齐，实测 |I_in−I_tar| 均值达 75/255
   （纯错位）。正确做法从 `tools/bgr_check/common.py:preprocess_source_bytes` 读出：
   生产渲染输入 = `ImageOps.exif_transpose → RGB → LANCZOS 缩到短边 1024`。
   本实验按此从 **img 银行**重建 I_in，实测三方尺寸 100% 一致（见 `loader.py`）。
3. **img 银行解析**：复用 `bgr_check.common.BankResolver`（full_path 优先、unsplash 必须命中
   `unsplash_work`），journal 里的本机路径已失效（`/home/bc/data/datasets/...` 实测 False）。
4. **S-split**：`s_split()` 与 `tools/data_splits/vr_common.py` 逐字节同式
   （`sha1("verasplit-v1:"+source_id)[:8] %100`，0-89/90-94/95-99）。旁表
   `tools/data_splits/splits.sqlite3`（33,652 源）已核对为同一函数的物化，**未做任何 ad-hoc 切分**。
5. **D-SFT-L/G 行定义**：来自 journal 归档 `sft.jsonl`（含 `winner_confidence` / `instruction` /
   `local.C_GT`），只取 `normal`；`low` 依数据纪律不进评测 GT。
6. **PSNR 口径**：`round(x*255)` 后 `10log10(255²/MSE)`，与 `tools/harness/metrics.py:psnr_full`
   同式（本模块内联 torch 版以走 GPU，`selfcheck` 对拍）。
7. **GLUT 引擎**：`model/glut_repro/model.py:BatchedGLUT.forward` 接受 `(B,P,3)`，
   **每个 batch 元素可有各自的像素集**——互证实验据此把 7 个臂放进一次前向；
   初始化复用 `fit_e1.weighted_kmeans_init` + `init_e1_residual`（G=I、M=0，非红线里的 G=I 禁令场景：
   那条红线针对 RD 臂的 4D 模型，A0/E1 复现臂按 GLUT 原文，见 `model.py` 注释）。
8. **JPEG 底噪参照**：`experiments/tooling-wave1/bgr_check/REPORT.md` §5.1 实测
   after 相对理想 float 渲染的噪声 = JPEG q95 底噪 ΔE00 p50≈0.76 / p99≈4.1。
   这是真实档 4D 天花板的物理上限来源之一（构造集是 PNG 无损，不受此限）。

## 二 方法学发现（**改变了估计器，必须记录**）

### 发现 1：PLAN §4.1 字面口径的「逐箱条件均值」有一个 ~41 dB 的硬地板，在 D-CONSTRUCT 上完全失效

33³ 等宽箱宽 = 1/33 ≈ 7.7 个 8bit 级 → 箱内色差自身贡献 MSE ≈ 7.7²/12 ≈ 4.9 →
PSNR 上限 ≈ 41 dB。实测 L0（全局编辑档）`PSNR_3D = 41.41 dB`，与理论地板吻合到 0.01 dB。

而 D-CONSTRUCT sanity 的编辑幅度偏小：**恒等映射（什么都不做）本身就有 37–41 dB**
（L3 41.23 / L4 41.61 / L5 40.75）。于是常数估计器的 `PSNR_3D` 反而**低于恒等**
（L3 39.92 < 41.23，L4 39.06 < 41.61），Δ_ceil 被压到 0.1–2.2 dB —— 这是估计器的量化地板，
不是数据的性质。用它去判 ①「Δ_ceil ≥ 8 dB」等于用一把最小刻度 41 dB 的尺子量 8 dB 的差。

**对策**：加一档**逐箱最小二乘仿射**估计器（单元内 `ŷ = A x + b`，12 自由度）。
局部线性正是三线性插值 3D LUT 在单个格子里的行为，因此它是**更贴近真实 3D LUT 能力**的天花板；
地板抬到 L0 实测 **63.43 dB**。两套估计器的数字都报（`delta_*` vs `delta_*_constest`）。

### 发现 2：单一支持度门（n≥60）会把小掩膜的 s 细分整个否掉

D-CONSTRUCT L1 实测掩膜只占 **2.4%** 像素（`alpha_achieved=0.024`）。
在「颜色单元 × s 桶」上，掩膜内子单元几乎都不到 60 像素 → 全部退回父预测。
同一样本用硬两臂 oracle（mask>0.5 直接分两组做仿射）得 **84.22 dB**，
而单档门的 4D 天花板只有 **54.06 dB** —— 差 30 dB 全是估计器自己丢的。

**对策**：三档细化（`_refine`）：`n≥60` 自己的仿射 → `n≥20` 父仿射+常数偏置（3 自由度）
→ 否则沿用父预测。三档都是父模型类的**子集扩张**，SSE 单调不增，Δ_ceil ≥ 0 仍然被构造保证。

### 发现 3：两种 4D 分区形态给出的数字差很多，两个都报

- **嵌套式 `delta`**：在 3D 分区**内部**按 s 桶细分（PLAN 字面）。父分区的颜色分辨率会牵制细分。
- **臂式 `delta_arm`**：每个 s 桶**各自独立**建颜色分层（= N 张 3D LUT 按 s 查表，
  正是 4D LUT 的实现形态，也正是实验法互证那条臂）。小掩膜下颜色分层会自动退到更粗层级，
  不受父分区牵制。

实测 D-CONSTRUCT L1：嵌套 5.58 dB vs 臂式 19.89 dB。**headline 用臂式**（它对应可实现的架构），
嵌套式作保守下界一并报。

### 发现 4：交叉拟合列（`delta_cv` / `delta_arm_cv`）在小掩膜上系统性偏保守

二折交叉拟合把每个单元的可用样本砍半，小掩膜单元同时踩到支持度门 → 细化被否 → 该列趋近 0。
它在**大掩膜真实档**上正常工作（实测真实档 `delta_cv` 7.48 vs in-sample 8.62，只低 1.1 dB），
在 D-CONSTRUCT 小掩膜档上不可用。

**因此有限样本膨胀的主控不是 CV 列，而是三个零点对照**（都跑在同一估计器上）：
`Δ_const`（s≡1，构造上恒为 0，作实现自检）、`Δ_donor`（移植别的图的掩膜 = Δ_shuffle）、
D-CONSTRUCT **L0**（真全局编辑）与 **L5**（`mismatch_semantic_gt`，掩膜故意与编辑区错配）。
L0 实测 0.00、L5 实测 0.16 dB —— L5 的分区细度与 L1/L2 完全同级却拿不到增益，
**这直接证明 in-sample 数字不是分区自由度堆出来的**。

## 三 实现自检（`tools/ceiling/selfcheck.py`，11/11 PASS）

| 检查 | 语义 | 实测 |
|---|---|---|
| `global_lut_delta_small` | 纯全局 3D LUT 编辑 + 随机 s → Δ≈0 | 0.054 |
| `masked_two_lut_delta_large` | 掩膜内外两个不同 LUT + s=掩膜 → Δ 大 | 15.19 |
| `masked_two_lut_delta_cv_large` | 同上，交叉拟合列也大 | 10.09 |
| `masked_two_lut_wrong_s_delta_small` | 同一编辑换成无关 s → Δ 塌回 | 0.064 |
| `delta_nonneg` | Δ_ceil ≥ 0 恒成立 | 6/6 |
| `identity_edit_delta_zero` | 恒等编辑 → Δ≈0 | 0.052 |
| `delta_const_exactly_zero` | s≡1 时 4D 分区 ≡ 3D 分区 → **严格 0** | 0.000e+00 |
| `s_buckets_degenerate/uniform` | 分位桶退化/满桶行为 | 1 / 8 |
| `moran_local_gt_noise` | 成簇残差 Moran's I ≫ 白噪 | 0.481 vs −0.002 |
| `blur_drop_noise_high` | 白噪残差 blur_drop > 0.6 | 0.889 vs 0.170 |

`delta_const_exactly_zero` 抓到过两个真实 bug：(a) 层级回退时用「整箱均值」而非「本单元成员均值」，
4D 侧会白捡差额；(b) CV 路径只给 4D 侧加偏置档，把「支持度门更松」误算成 s 的增益（L0 虚高 2.2 dB）。
两个都已修（见 `delta_ceil.py` 内注）。

## 四 假设与待确认清单

**已自行核实、按核实结果执行的：**
- A1 I_in 重建管线（见 §一.2）——已对 6 个 pool 抽验尺寸 100% 一致。
- A2 S-split 只读旁表口径——已与 `vr_common.py` 逐式对照。
- A3 PSNR round×255——已与 `harness/metrics.py` 对照。
- A4 `.jpg` 就是 I_tar（不是别的候选）——由 `sample_id` 前缀 = candidate_id 唯一确定。

**决策项（两种做法都合理、影响后续），已采保守默认并在此登记：**

### 待主 agent 决策 #1：headline 估计器用「逐箱仿射」而不是 PLAN 字面的「逐箱条件均值」
- 保守默认（本次采用）：**两套都跑、两套都报**，REPORT 主表用仿射（`delta_arm`），
  附 PLAN 字面口径列（`delta_arm_constest` / `psnr_*_constest`）。
- 理由见 §二发现 1：字面口径的地板在 D-CONSTRUCT 上低于恒等映射，判据 ① 用它无法成立也无法证伪。
- 影响面：若主 agent 坚持字面口径，则判据 ① 的 8 dB 阈值需要连同估计器一起改写
  （EXPERIMENTS_v3 L401 / PLAN L456），否则该判据在**任何**数据上都不可达。

### 待主 agent 决策 #2：D-SFT-G 对照档的 s 用「移植掩膜」
- D-SFT-G 没有 C_GT。若直接取 s≡1，Δ_ceil 恒为 0（构造使然），**测不出方法的虚高**。
- 保守默认（本次采用）：从 l 系 S-val 取 64 张 C_GT 组成移植池（面积 2%–98% 过滤），
  按索引轮转贴到 g 图上（最近邻重采样，不引入新灰度）。这样 s 的空间/取值统计与真实档同级，
  但编辑本身是真·全局 → 任何非零 Δ_ceil 全部是有限样本膨胀。
- 备选：s 用随机高斯场 / SLIC 分割。移植掩膜更贴近真实档统计，故选它。

### 待主 agent 决策 #3：PLAN §4.1 Step 5 的两个混淆过滤器「只报不筛」
- Moran's I < 0.3 丢弃、blur_drop > 0.6 丢弃这两条是为 FiveK/PPR10K 这类**来源不明**的配对设计的。
- 自家渲染数据的编辑形态由 recipe 完全已知（LUT + 掩膜合成），不存在「高频编辑/非色彩算子」混淆。
- 保守默认（本次采用）：**两个统计逐图算出来写进 per_image，但不据此丢样本**；
  REPORT 里报分布（实测真实档 Moran's I 中位 0.985、blur_drop 中位 0.016，
  与「真局部色彩编辑」的预期完全一致，没有一张需要被筛）。

### 待主 agent 决策 #4：像素取整数 stride 抽样，不做插值 resize
- PLAN §4.1 Step 1 写「sRGB 512 长边」。插值 resize 会破坏逐像素对应
  （`avg(f(x)) ≠ f(avg(x))`，同时污染 3D 与 4D 天花板）。
- 保守默认（本次采用）：**原生分辨率全像素**（短边 1024，实测 1.0–1.9 M 像素/图），
  超过上限才按整数 stride 抽同一网格坐标。像素越多、分箱统计越稳，且完全无重采样污染。

### 待主 agent 决策 #5：配准检查按任务卡跳过
- 任务卡明写「配准检查跳过——自家渲染对天然对齐」。已按此执行；
  I_in 由源图经与生产**同一条** LANCZOS 管线重建，几何上逐像素同源，无配准自由度。

## 五 GPU 与长任务纪律

- 启动前 `nvidia-smi` 实测：卡 0 = 17,055 MiB / 100%（他人在途，未触碰），**卡 1 = 0 MiB / 0%**。
- 全程 `CUDA_VISIBLE_DEVICES=1`，单卡串行六步。
- `job.marker` 含 PID / 完整启动命令 / 全部日志路径；`STATUS.md` 每步 start 与 rc 落盘。
- 回合内只跑冒烟（construct 全量 20 s、real 20 张、xcheck 3 张、MLP 1 样本 1500 步）；
  全量走 nohup。

---

## 六 补批实施记录（2026-08-03，D-34 覆盖面）

### 6.1 读过的文档章节
- `CLAUDE.md`（全文：角色分工 / REPORT.md 强制前三行 / 交付物规范 / 红线速查）
- `docs/DECISIONS_2026-08-03.md` **§七 D-30..D-35**（本次任务由 **D-34** 派生；D-31 的措辞纪律
  「区分『未触发死刑』与『获得正面证据』」已在 §6.2′ 的写法上遵守 —— 报的是「原结论未被推翻」，
  不是「补批证明了 s 有用」）
- 本实验原 `REPORT.md` 的 〇（O-2/O-3）、§三、§6.1–6.4、§十、§十一、附录 B

### 6.2 本次改了什么（**只有抽样**）
`tools/ceiling/run_analytic.py`：
1. `iter_indexed()` 新增 `per_build` 参数 —— 按 build 分层均衡取样（build 内按 candidate_id
   确定序，build 间按名字升序）。`per_build=None` 时行为与原口径**完全一致**（保留原路径）。
2. `aggregate()` 新增 `by_build` 块（中位 + p10/p90 + `frac_ge_1db` 等）—— **纯追加**，
   不改任何已有字段的计算。
3. CLI 新增 `--per-build` / `--tag`；`run()` 的输出文件名加 `tag` 后缀，
   agg 里追记 `sampling` / `per_build_cap` 两个自述字段。

**没有碰**：`delta_ceil.py`（估计器）、`summarize.py`（判据）、`loader.py`、`viz.py`、
`selfcheck.py`、seed、donor 池构造。

### 6.3 零改动的验证方式（**不是靠声明，是靠对拍**）
两批在真实档上天然有 **374 组重叠样本**（原批次 = 索引前 600；补批 = 每 build 前 100）。
逐样本对拍结果：

| 键 | max\|Δ\| | 逐位一致 |
|---|---|---|
| `delta_arm`（headline）| **0.00e+00** | **374/374** |
| `psnr_3d` / `psnr_4d_arm` / `psnr_id` | **0.00e+00** | **374/374** |
| `delta`（嵌套式） | 8.08e-07 | 370/374 |
| `delta_alt`（17³ 起点） | 7.22e-06 | 372/374 |

嵌套式两列的 1e-6 量级差来自 GPU `index_add_` 原子累加顺序（float32 求和不满足结合律），
与 O-5a 已登记的「真实档 Δ_const ≤1.68e-6 非严格 0」是同一个物理原因。**不改变任何一位有效数字。**

### 6.4 假设与待确认清单

**当场核实的（不留悬念）**
1. 索引里 6 个 l build 各有 157/167/202/179/171/185 条、4 个 g build 各有 296/323/315/341 条
   → 真实档 100/build、对照档 150/build 都取得满，无需「不足则取满」的降级分支（分支仍实现了）。
2. 原批次 `per_image_real.jsonl` 的 build 分布确为 157/167/202/74（l1–l4），
   `per_image_control.jsonl` 为 296/304（g1–g2）—— 与 O-2 的记账逐数字吻合。
3. HEAD = `0e5d04a7…`，与原批次 `config/config.json` 记录的 `git_commit` **同一个 commit**，
   两批代码基线一致（差异只有本次对 `run_analytic.py` 的抽样改动，已快照进
   `config/run_analytic.strat.py`）。

**待主 agent 决策（已采用保守默认继续，未静默拍板）**

- **决策 #6：headline 该引哪一批的数字？**
  两批都过判据 ②③⑤，方向一致（7.969 → 8.037 dB）。
  - 保守默认（本报告采用）：**原批次结论原样保留不覆盖**，补批作为独立一节并列，
    §三 判据表里两批数字并排列出。
  - 建议（需主 agent 拍板）：对外（论文/汇报）引**补批 8.04 dB**——它是 6 build 全覆盖、
    每 build 等权的中位数，抗「抽样偏差」质疑；脚注原批次 7.97 dB。
  - **注意两批的加权口径不同**：原批次按索引可用量加权（l3 占 33.7%、l5/l6 占 0%），
    补批按 build 等权。这不是同一个总体的两次抽样，是**两个略微不同的估计量**，
    §6.2′ 末尾的「两批口径差异」表已如实标注。

- **决策 #7：要不要把补批扩成全量普查？**
  现在两批各 600 组，仍是索引子集（真实档 600/1061、对照档 600/1275）。
  跑全量真实档 1061 组约需 18 min、对照档 1275 组约 15 min，成本不高。
  保守默认：**不跑**——分 build（6/6 与 4/4，极差 1.14 / 0.018 dB）与分池（5/5）两条稳健性
  都已全覆盖且高度一致，全量普查的边际信息很低。若主 agent 认为「600/1061 是子集」
  这条仍会被审稿人抓，可随时补跑（脚本已就位：`run_strat.sh` 把 `PB_REAL` 调到 ≥202 即可）。

- **决策 #8：`--limit` 顺序截断这个坑要不要工具层封死？**
  本项目所有索引都按 build 顺序拼接，任何 `--limit` 都会让尾部 build 零样本。
  保守默认：**只在报告 §十.建议 1 写纪律**，未改动其它工具。
  建议：在 `run_analytic.py` 里让 `--limit` 在检测到多 build 索引时**打印显式警告**
  （或直接要求二选一），把这个坑变成不可能再犯的错误 —— 与 D-25 对 `.in.jpg` 加启动断言同思路。

### 6.5 GPU 与长任务纪律
- 启动前 `nvidia-smi`：卡 0 = 15100MiB / 79–100%（4 个在途进程），
  卡 1 = 1773MiB / 3–5%（G3 训练 PID 3900089，`train_g3 --arm anchored`）。
  → **选卡 1**（与原批次同卡，且负载远低于卡 0）；显存充裕，实测共卡后卡 1 峰值 4001MiB / 41%。
- marker：`job_strat.marker`（PID 3943059 / 完整启动命令 / 日志路径 / 输出清单 / kill 方式）。
- 进度写 `STATUS.md`，补批行统一带 `[补批]` 前缀，与原批次行不混淆。
- 实际耗时 17 min（real 10 min + control 7 min）+ viz 15 s，全程 rc=0。
