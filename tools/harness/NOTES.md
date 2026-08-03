# NOTES — T3 统一评测 harness（INF-1）

日期：2026-08-02　实施者：编码 subagent（任务卡 T3）

## 一、实施前核实记录（外部事实）

均在本机环境（/home/bc/miniconda3/bin/python3）**直接 import + inspect 核实**，不依赖检索引擎：

| 事实 | 核实方式 | 结果 |
|---|---|---|
| skimage 版本 | `skimage.__version__` | 0.25.2 |
| `rgb2lab` 签名与默认光源 | `inspect.signature` | `(rgb, illuminant='D65', observer='2', *, channel_axis=-1)` — 默认即 D65，符合 IMPL_DOSSIER §4.2-8 的口径 |
| `deltaE_ciede2000` 签名 | `inspect.signature` | `(lab1, lab2, kL=1, kC=1, kH=1, *, channel_axis=-1)` |
| `structural_similarity` 签名 | `inspect.signature` | `(im1, im2, *, win_size=None, gradient=False, data_range=None, channel_axis=None, gaussian_weights=False, full=False, **kwargs)` |
| lpips 已安装 | `import lpips` | 可用；`LPIPS(net='alex', version='0.1')`（无 `__version__` 属性） |
| scipy `binomtest` / `distance_transform_edt` | import 核实 | scipy 1.16.3 均可用 |
| numpy / torch / pillow | import 核实 | 2.4.6 / 2.6.0+cu124 / 12.2.0 |
| 落盘数据结构 | `tar tf` + idx.jsonl 抽查 mini30-v52-20260730 | 每样本三件套 `{sid}.in.jpg`（输入）/ `{sid}.jpg`（输出 GT）/ `{sid}.vrmeta.json`，部分样本带 `{sid}.cgt.png`（单通道软掩膜，任务卡已核实=逐候选区域掩膜） |
| ΔE00 参考实现 | selfcheck 内独立实现 CIEDE2000 公式（Sharma 2005 标准公式结构，纯代数，不引外部数字）与 skimage 对拍 | 见 selfcheck 结果 |

新装依赖：**无**（全部用环境已有库）。

## 二、口径锁定（引权威文档）

- **全图 PSNR**：`round(x*255)` 后算 MSE，`10*log10(255²/mse)`，逐图（batch=1）再平均 —— IMPL_DOSSIER §4.1 条 6、INF-4。
- **masked PSNR 三分**：掩膜内 / 边界带 ±k px（k 可配）/ 掩膜外 —— PLAN §3 M7、EXPERIMENTS_v3 INF-1。量化口径与全图 PSNR 一致（同一把尺）。
- **ΔE00**：skimage `rgb2lab`（D65）→ `deltaE_ciede2000`，与 IMPL_DOSSIER §4.2-8「纯 python 复刻 ΔE 时 rgb2lab 用 D65 sRGB（skimage 一致）」对齐。
- **Δ_const**：PSNR(s) − PSNR(s_∅)（PLAN §3 M1）；**Δ_shuffle**：PSNR(s) − PSNR(跨图置换 s)（PLAN §3 M2）。
- **配对统计**：逐图配对 ΔPSNR + bootstrap 95% CI + 符号检验 —— PLAN §3「配对 ΔPSNR+bootstrap CI」。
- **σ_s 分布统计**：M3（σ_s 贴上界比例，红线 >80%）。
- **红线遵守**：harness 对 s 不做任何逐图归一化（min-max/softmax 均禁）；leaderboard 每行强制 Δ_const/Δ_shuffle 列，缺失标 N/A 并 stderr 警告。

## 三、假设清单（已自行核实/可自行拍板的实现细节）

1. **图像值域**：所有 API 接收 float RGB ∈[0,1]（HxWx3）或 uint8（自动 /255）。数据集 jpg 用 PIL 读取为 RGB。
2. **软掩膜二值化**：三分区先按 `mask >= thr`（默认 0.5）二值化再算距离带。C_GT 是软边单通道，阈值可配。
3. **三分区互斥**：`in_core = 掩膜内且距边界 > k`；`band = 距二值边界 ≤ k 的双侧像素`；`out_core = 掩膜外且距边界 > k`。三区不交、并集=全图（M7 表述"内/边界带/外"未明说是否互斥，取互斥版避免边界像素重复计入掩膜内——效应量口径更干净）。距离用 `scipy.ndimage.distance_transform_edt`（欧氏）。
4. **PSNR 上限**：mse=0 时 PSNR=∞；数值输出以 `cap_db`（默认 100.0）截断，JSON 可序列化。空区域 → NaN，聚合用 nanmean 并报有效图数。
5. **SSIM 参数**：`gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255`，作用在 round×255 后的值上——即标准 Wang 2004 的 11×11 高斯窗配置（与 DOSSIER 竞品表 "skimage win11 gaussian" 同构）。
6. **LPIPS**：懒加载（函数内 import + 模块级缓存），net='alex'，输入变换到 [-1,1]。首次调用可能触发 torchvision 权重下载。
7. **Δ_shuffle 的置换**：seeded 随机置换后再整体轮移 1 位，保证无不动点（derangement）；n=1 时拒绝并报错。
8. **σ_s「贴上界」判定**：σ ≥ σ_max − ε，ε = edge_frac×(σ_max−σ_min)，edge_frac 默认 0.01（M3 未给 ε，取窄容差保守值，可配）。
9. **T5 之外**：任务卡只要 σ_s 分布统计器"吃参数字典"，实现为对 dict 里指定 key（默认 `sigma_s`）的数组做统计。

## 四、待主 agent 决策（保守默认已采用，不阻塞）

1. **边界带默认宽度 k**：PLAN/EXPERIMENTS 均未给数值。默认 `band_px=3`（360p–480p 量级下羽化过渡的典型宽度），**全 API 可配**。若后续统一口径请在 EXPERIMENTS_v3 固定一个 k 并全局沿用（k 改变会改变三分 PSNR 的可比性）。
2. **Δ_const 的 s_∅ 取法**：PLAN §3 里 s_∅ 是模型 condition-dropout 学出的常量，属模型内部件；harness 评测器无法拿到。保守默认：`s_null` 参数留接口（模型方可传入自己的 s_∅），不传时用**全评测集 s 的逐通道均值**广播成常量场。两种口径数值会不同，榜单里 `delta_const_mode` 字段记录用的哪种。
3. **烘焙一致性（四面体插值回读）**：INF-1 原文含此件，但任务卡 T3 的实现要求（5 个文件）未列。按任务卡范围执行，**本交付不含**烘焙一致性评测器；若需要请另开任务卡（依赖 INF-3 的 .cube 工具链）。
4. **metrics.json schema**：文档未定义机器可读 schema。已在 README 定义最小 schema（leaderboard 依此解析），若主 agent 有既定 schema 请告知后改 leaderboard 的 key 映射（单文件 <30 行处）。
5. **LPIPS 真实数据自检**：若本机无 torchvision alexnet 缓存且无外网，LPIPS 冒烟测试会跳过（selfcheck 中标注 SKIPPED），不影响其余指标。实测结果见 REPORT.md。

## 五、冒烟实测补记（2026-08-02）

- mini30-v52 shard-00000 实测：108 members / 30 样本，24 个带 `.cgt.png`，其中仅 8 个
  三件齐全（16 个带 cgt 但**无 `.in.jpg`**）——smoke_real.py 只取三件齐全者；下游消费
  shard 时勿假设每样本必有输入图。
- LPIPS alex 权重本机已缓存，可正常出数（torchvision 仅发弃用警告）。
- 三分区在"大面积软渐变掩膜"样本上会退化成对角切分（见 failure viz），band 统计意义变弱
  ——已写进 REPORT.md 建议（正式实验按掩膜面积分层报告）。

## 六、范围与红线自查

- 只新建文件：`tools/harness/*` 与 `experiments/tooling-wave1/harness/*`，未动仓库既有代码。
- s 不做逐图归一化；leaderboard 强制 Δ_const/Δ_shuffle 列；PSNR 口径 round×255；ΔE00 D65。

## 七、wave-1.5 修复记录（2026-08-03，F2 清 B2）

依据：REVIEW-impl-wave1 T3-B2、DECISIONS_2026-08-03 §三 B2/F2、EXPERIMENTS_v3 §1 INF-1。

1. **新增 `bake_consistency.py`（INF-1 烘焙一致性评测器，清 blocker B2）**
   - 接口 `bake_consistency(render_fn, natural_images=None)`，`render_fn(rgb)->rgb` 为任意
     可微渲染器的逐点求值（`(...,3)` float RGB ∈[0,1]）。
   - 流程：均匀 33³ 格点采样渲染 → 组装 (33,33,33,3) canonical LUT（逐点采样天然保持
     `identity_table` 的 [r,g,b] 索引约定）→ **colour 四面体插值回读**——复用
     `tools/cube/cubelib.apply_lut_tetrahedral`（即 hald.py `--method tetrahedral` 的同一应用器，
     未重写；不 import hald 本体以避免其 `sys.path.insert(0, cube)` 遮蔽 harness 同名模块）。
   - 评测面：**128³ 留出色** = {1,3,...,255}³（全通道奇数 ⇒ 与 GLUT 训练色 {0,2,...,254}³
     严格互斥，整体属于 T2 hald eval 留出集），排成 1024×2048×3；外加可选自然图列表。
   - 指标：直渲 vs 烘焙回读的逐像素 ΔE00（skimage rgb2lab D65 → deltaE_ciede2000，与
     metrics.delta_e00 同链）p50/p90/p99（附 mean/max）、PSNR 差（round×255 口径
     `metrics.psnr_full(direct, baked)`）、最大逐通道绝对误差。
   - **判据常量写死**：`DE00_P99_PASS = 0.5`，每个评测面 ΔE00 p99 < 0.5 为过，
     总判 `passed` = 全部评测面过（红线「烘焙一致性从第一天当一等指标」）。
2. **metrics.py Pyright 修复（仅类型收窄，口径零改动）**
   - L94（现 L96-97）：`distance_transform_edt` 默认参数下恒返回 ndarray，scipy 存根的
     tuple/None 分支不可达 → `typing.cast(np.ndarray, ...)`。
   - L150：`structural_similarity` 在 full=False/gradient=False（默认）下返回标量，
     tuple 分支不可达 → 赋中间变量后 `cast(float, ...)`。
3. **smoke_real.py Pyright 清扫（模块 0 报错要求）**：`tf.extractfile` 可空 → assert 收窄；
   `Image.BILINEAR`（存根未知）→ 类型化别名 `Image.Resampling.BILINEAR`（×4，运行时同值）。
   修后重跑 smoke_real（默认 n=6）恢复交付产物，exit 0。
4. **selfcheck 新增 §8 两例（8 项检查）**
   - 恒等渲染器：烘焙一致性 **ΔE00 max = 0（tol 1e-12）**、max_abs_err = 0、PSNR 触 cap
     ——恒等 LUT（节点值 i/32 为二进制精确 dyadic）的四面体回读逐位精确。
   - gamma(γ=2) 渲染器：烘焙误差为小的非零值且**闭式可预估**——四面体插值对逐通道可分
     函数等价于逐通道 1D 线性插值（方案对线性函数精确 ⇒ x1 面权重和=分数坐标），
     f=x² (f''=2) 的单元内误差 e(x)=(x−x0)(x0+h−x)，h=1/32；对留出色格独立算
     解析 max=2.44137e-4，实测 2.44129e-4（差 7.5e-9 = LUT float32 舍入），tol 1e-6 过；
     ΔE00 p99=0.030 ∈ (0, 0.5) 过判据；附带覆盖 natural-image 路径（随机 64×64）。
5. **自检结果**：selfcheck **50 passed, 0 failed（原 42 + 新 8），exit 0**；
   `pyright tools/harness/` **0 errors**（唯一豁免：`from cubelib import ...` 跨目录运行时
   import 定点 `pyright: ignore[reportMissingImports]`，沿用 tools/cube 的定点豁免惯例，
   不引入 repo 级 pyrightconfig 以免扰动并行模块基线）。
6. **遗留（非本卡范围）**：烘焙一致性尚未并入 metrics.json schema v1 / leaderboard 列
   （B2 修复指引提及「可增列」，schema 改动按 D-11 需走版本号，待主 agent 排卡）；
   s 条件渲染器需按固定 s 切片包装成 `render_fn(rgb)` 后逐切片调用（接口按 F2 任务卡）。
