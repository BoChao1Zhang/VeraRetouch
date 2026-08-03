# VeraRetouch 实现档案（implementation dossier）

依据 12 路实现级调研（所有 URL 均经一手核实；标注"论文声称"处无代码佐证）。给马上写代码的人。

---

## 一、复用资产总表

按项目动工顺序排列。接入成本：**当天** = clone/pip 后当日出结果；**一周** = 涉及编译/环境锁/重排目录/自写胶水。

| # | 组件 | 官方仓库（已核实） | 拿来做什么 | 成本 | 已知坑 |
|---|------|------|------|------|------|
| 1 | FiveK 480p 成品数据包 | github.com/HuiZeng/Image-Adaptive-3DLUT（数据在 GoogleDrive folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a，百度盘码 5fyk） | 训练/评测主数据：8bit sRGB 输入 + 16bit XYZ 输入 + expertC 目标，4500/500 split | 当天 | split 靠 train_input.txt/train_label.txt/test.txt（随数据包，repo 内无）；XYZ 分支 dataloader 保持 BGR 不转 RGB；PSNR 必须 round(x*255) 后算 |
| 2 | Prepare_FiveK.md（DNG→480p 官方步骤） | github.com/ImCharlesY/AdaInt（SepLUT 同款） | 需要全分辨率/自定义导出时的唯一成文流程 | 一周（Lightroom 手工） | 输入取 collection 'InputZeroed with ExpertC WhiteBalance'，JPEG q=100 sRGB 短边 480 |
| 3 | PPR10K 360p tif 包 + masks_360p + calculate_metrics.m | github.com/csjliang/PPR10K | 人像数据 + HRP/GLC 指标唯一官方实现 | 一周（91GB 下载） | 度量脚本假定结果 .png / GT .tif 同名、group 数写死 325；训练 mask 权重 5、评测背景权重 0.5，两处口径不同；文件名必须保持 `<groupid>_<idx>` |
| 4 | colour-science（.cube 读写 + 四面体插值） | github.com/colour-science/colour（colour/io/luts/iridas_cube.py） | 4000-cube 语料的解析器与 GT 渲染器 | 当天 | .cube 行序 R-fastest，reshape order='F'；LUT3D 表 shape (S,S,S,3) 索引 [r][g][b]，与 torch (3,S,S,S) 系相反 |
| 5 | pillow-lut + ImageMagick hald: | github.com/homm/pillow-lut-tools | Hald↔cube 互转、快速批处理、identity 生成 | 当天 | 忽略 DOMAIN_MIN/MAX（HDR log LUT 必须走 colour）；Hald 严禁 JPEG |
| 6 | AceTone 工具链：convert_luts.py / select_luts.py / dataset/lut3d.py | github.com/martian422/AceTone | 任意 .cube→32³ npy 统一化；PCA+KMeans 聚类抽代表 LUT；apply_lut_batch（grid_sample 5D）+ ΔE76/94/2000 | 当天 | **apply_lut 内有 `lut[..., ::-1]` BGR 翻转约定**，接自家渲染器必翻色；`get_path` 未定义（issue #3），需自补 |
| 7 | NamedCurves 工程模板 + utils/deltaE.py + mit5k_ids_filepath/ | github.com/davidserra9/namedcurves | GLUT 复现的训练脚手架（omegaconf 反射构建 optim/sched/criterion）+ 与 GLUT 同组的 ΔE00/ΔEab 口径 + FiveK 三种 split 清单 | 当天 | upe config 的 test split 误指 images_train.txt；yaml 里作者本机绝对路径需全换；torch 1.12 老 |
| 8 | trilinear_cpp CUDA 扩展 / grid_sample 平替 | HuiZeng 仓库（PPR10K/CLUT/ICELUT/NLUT 同源） | 3D LUT 可微渲染 | 当天（grid_sample）；编译版半天-一周 | 必须先 `import torch` 再 `import trilinear`；CUDA 版本失配 undefined symbol；issue #14 官方认可 grid_sample 平替——现代 torch 直接用 grid_sample |
| 9 | SA-LUT clut4d.py + quadrilinear_cpp | github.com/Ry3nG/SA-LUT | 4D LUT 对照臂：参数化 num_context_bins 的最干净实现 + 4D 插值 CUDA（与 4DLUT 同血统） | 一周 | CUDA 扩展不支持 batch 内不同 LUT（per-sample for 循环）；trainer 默认 dim=33 会被 yaml 覆盖成 17；训练数据管线未发布，只能借结构不能重训 |
| 10 | AceTone VQ-VAE tokenizer（model/acetone-vqvae-d64.pt + vq.py） | 同 #6 | 32³ LUT ↔ 64 token（K=256/D=64），VLM→LUT token 路线起点，权重直接在仓库 | 当天 | vq.py 仅依赖 torch 可单文件搬走；eval_vqvae.py 用合成正弦渐变图而非论文所述 Adobe-5K 图 |
| 11 | InstantRetouch torch_layers.py（Biliteral_Grid_Joint + slicing + Guide）+ hist_loss.py + loss.py | github.com/OpenImagingLab/InstantRetouch | bilateral grid 渲染对照组（纯 PyTorch grid_sample 无 CUDA kernel）；可微 YUV 直方图损失；IP2P 条件 DMD/VSD 蒸馏参考实现 | 当天（渲染栈）/一周（蒸馏链） | 权重/数据/属性库全未放出；grid_res 是结构超参（16 vs 部署 32），ckpt 不通用；权重调度列表用 `eval()` 解析 |
| 12 | RSFNet renderer_arch.py 8 滤镜可微原语 + HF 权重/数据 | github.com/Vicky0522/RSFNet（权重 hf.co/Vicky0522/RSFNet-models，数据 hf.co/datasets/Vicky0522/MIT-Adobe5k-for-RSFNet） | 掩膜基底 E2 最接近的完整参照：Y=X+ΣΣ(F−X)⊙M 并行 delta 合成 + SegRecolorHead 双分支 | 当天（抄公式）/一周（重训） | rgb2lab 是自编译 CUDA 算子→换 kornia；rsfnet_arch.py 顶部 `set_detect_anomaly(True)` 忘关，先删；输入 [-1,1]/renderer 内部 [0,1] 换算是最常见转写错误 |
| 13 | Harmonizer filter.py 6 白盒滤镜 + CascadeArgumentRegressor | github.com/ZHKKKe/Harmonizer | 参数域统一 [-1,1] 的纯 PyTorch(+kornia) 滤镜库；60 行序贯参数回归头 | 当天 | 滤镜级联应用（非并行），restore/adjust 互为反序；Contrast/Temperature 用全局均值→高分辨率执行有统计漂移；CC BY-NC-SA |
| 14 | DeepLPF model.py 三滤镜（Cubic/Graduated/Elliptical） | github.com/sjmoran/deeplpf-image-enhancement | 几何参数化掩膜锚点 + 三个预训练权重随仓库 | 当天 | mask_scale6 存在 b3 抄错 bug（以代码为准）；x/y 轴命名与常规相反（x 是 H 方向）；最终输出与原图残差相加，转写常漏 |
| 15 | CSRNet_arch.py（76 行）+ csrnet.pth | github.com/hejingwenhejingwen/CSRNet | 零编译最低成本纯 PyTorch baseline，GFM 条件注入模块 | 当天 | 数据是作者自制预处理包（非 Zeng 480p，跨论文对比注明）；yml 里 dataroot 必改 |
| 16 | AdaInt/SepLUT 权重 + annfiles + ailut/seplut CUDA 算子 | github.com/ImCharlesY/AdaInt、/SepLUT | LUT baseline 对比 + 非均匀采样/1D+3D 级联算子 | 一周 | 环境锁死 Py3.7.10+PT1.8.1+CUDA10.2+GCC7.5；AdaInt smooth_factor=0、SepLUT monotonicity=0（与 Zeng 不同，勿抄错） |
| 17 | CLUT/ICELUT/NLUT 低秩 CLUT 实现 | github.com/Xian-Bei/CLUT、Stephen0808/ICELUT、semchan/NLUT | LUT 低秩压缩参数化（NLUT 的 class CLUT 可单独拿走）；ICELUT transfer2LUT.py 网络→纯查表端侧转换 | 当天 | 默认 loss 是 l1+cos 非 MSE；ipdb 依赖残留；数据目录要重排成 input_train/target_train |
| 18 | UltraEdit data_generation.py 掩膜链 | github.com/HaozheZhao/UltraEdit | 「指令+edit_object→GroundingDINO(0.3/0.25)+SAM ViT-H→三重掩膜质检→带掩膜图对」唯一开箱链路 | 一周 | 代码质量差（replace 分支 num_idx 未定义、目录名 traning/、ckpt 路径拼错 .pthc）；inpaint 分支须环境变量 INPAINTING=True；LLM 指令生成部分未开源 |
| 19 | IP2P dataset_creation + metrics/clip_similarity.py | github.com/timothybrooks/instruct-pix2pix | 700 条人写种子三元组 + CLIP 三重过滤器（0.2/0.2/0.7 阈值事实标准） | 当天（过滤器） | GPT-3 finetune API 已下线需换开源 LLM；clip_dir 0.2 阈值会误杀色彩微调样本（方向信号弱）——影调语料需下调或改用 SSIM/DINOv2 上限过滤 |
| 20 | OmniEdit-Filtered-1.2M + VIEScore 配方 | hf.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M；github.com/TIGER-AI-Lab/VIEScore | attribute-modification 子集直筛做色彩编辑参考；「GPT-4o VIEScore→蒸馏 InternVL2→≥9 分门」质量门配方（附录含完整 prompt） | 当天（数据） | 官方仓库零代码；HF task 列 6 类与论文 7 类口径不一致，先 value_counts |
| 21 | AnyEdit color_alter/tone 子集 + 逐类型 pre/post filter | github.com/DCDmllm/AnyEdit（数据 hf.co/datasets/Bin1117/AnyEdit） | 唯一显式带 color_alter 标签的指令语料；schema 直接采纳 | 当天 | 检索引擎会编造 AnyEdit/AnyEdit 假链接；全局影调类无掩膜字段 |
| 22 | MagicBrush（人工掩膜+多轮指令） | github.com/OSU-NLP-Group/MagicBrush（hf.co/datasets/osunlp/MagicBrush） | 「指令→掩膜」监督 GT；事实标准评测集 | 当天 | test 走密码 zip（MagicBrush）且 canary 禁止入训练集；掩膜是手画 free-form 不贴轮廓 |
| 23 | GEM（pip gem_torch） | github.com/WalBouss/GEM | training-free 读出最快路径：6 行出第一张 s 图 | 当天（2 步） | open_clip>2.24 签名变动会错位，pin ≤2.24；输出是 min-max 热图非 softmax |
| 24 | SCLIP / NACLIP | github.com/wangf3014/SCLIP、sinahmr/NACLIP | CSA / 邻域高斯注意力读出 + mmseg 评测线 | 当天（单图自写 15 行）/评测线一周 | mm 栈锁版 mmcv==2.0.1/mmseg==1.1.1/yapf==0.40.1/numpy==1.26；均无 demo.py |
| 25 | ClearCLIP / ProxyCLIP | github.com/mc-lan/ClearCLIP、/ProxyCLIP | 带 demo.py 的读出 + DINO proxy attention 注入 | 当天（4 步） | vendored open_clip 与 pip 版同名串包（repo 根目录置 sys.path 最前）；S-Lab License 非商用 |
| 26 | FeatUp / LoftUp / JAFAR 特征上采样 | github.com/mhamilton723/FeatUp、andrehuang/loftup、PaulCouairon/JAFAR | s 图上采样：FeatUp hub 'maskclip' 是拿高分辨率 MaskCLIP 特征最短路径；LoftUp 覆盖 SigLIP2；JAFAR 唯一有 DINOv3 权重 | 当天（hub 两行） | FeatUp use_norm 须与 ckpt 匹配（maskclip 只有 no_norm）；JAFAR 权重按 timm 精确模型名命名 |
| 27 | kornia.filters.guided_blur | kornia 官方 | s 图边缘细化（subsample>1 即 Fast GF，可微） | 当天 | eps 是归一化域量级 1e-4~1e-2，照搬 OpenCV 0-255 域 eps 会过平滑；需 kornia ≥0.7.x |
| 28 | READ SasP 两函数 | github.com/rui-qian/READ（model/READ.py） | E11 点读出：compute_similarity_map + similarity_map_to_points，零参数 ~150 行，不依赖 SAM | 当天-一周 | 超参在源码 837-839/764 行注释切换，不在命令行；24×24 粗图直接当 mask 的质量无验证 |
| 29 | UGround MasP + tools/simi_loss.py | github.com/rui-qian/UGround | E12 soft mask 读出（裸 einsum 点积）+ 高斯软化 GT（ksize=31, σ=7）BCE/Dice 离线打分工具（吃 .npy，不需要 SAM） | 当天 | **PPM RL 选层是死代码**（mode2/3/4 提前 return mode1），官方脚本也用 --mode=1——不要实现 RL 部分 |
| 30 | concept-erasure / repeng / nnsight / tuned-lens | EleutherAI/concept-erasure、vgel/repeng、ndif-team/nnsight、AlignmentResearch/tuned-lens | 见第六节 | 当天（推理侧）/一周（VLM tuned lens 训练） | 见第六节 |
| 31 | VideoColorGrading（LUT 残差扩散） | github.com/seunghyuns98/VideoColorGrading | 16³ LUT↔64×64×3 图像↔残差扩散编解码 + LUT 外推增广 a∈[-2,2]；权重与处理好数据均有 Drive 链接 | 一周 | yaml 覆盖脚本默认（实际 lr=1e-5 fp32，别看脚本默认 3e-5）；Movie_3K_Step2.py 硬编码 LUT 路径要手改 |
| 32 | Hist2Style 数据配方（无代码） | github.com/dgalor/hist2style（代码 Adobe 审批中） | D 层合成语料模板：LLM prompt 库 → FLUX.1 Kontext dev 批量编辑 → VGG19 cosine>0.5 过滤 | 配方参考 | 只有正文+补充材料的数值配方，盯 repo 等放码 |

---

## 二、GLUT 基座档案

### 2.1 代码现状

- 官方仓库 **github.com/CVC-Color/glut 是占位壳**（GitHub API 核实）：Apache-2.0，2026-05-18 创建后零提交，全仓 3 个文件，README 共 6 字节（"# GLUT"）。issue #1 催码（2026-07-20）零回复。项目页 color.cvc.uab.cat/glut/ 无响应，Wayback 无快照。
- 搜索引擎给出的 github.com/mv-lab/GLUT 是**编造的**（API 404）。
- 结论：**按 no_code 规划，从论文（arXiv:2605.19889v1）从零复现**。工程模板直接用同组的 davidserra9/namedcurves（换掉 models/ 与 data/ 即可）。

### 2.2 从零复现规格书

一作 Danna Xue（不是 Serrano-Lozano）。以下标注均为论文章节/表号（arXiv v1 无稳定页码，用 section id）。

**模型参数化（§3.1）**，N 个 Gaussian，总参数 22N+12（N=32→716，N=64→1420）：

| 参数 | 形状/域 | 说明 |
|---|---|---|
| μ_i | R³ | RGB 空间中心 |
| Σ_i = L_i L_iᵀ | 6 参/个 | 下三角 Cholesky；对角正性 §3.1 写 Softplus（矛盾见 2.4） |
| o_i | [0,1] | opacity，激活函数全文未写明 |
| M_i, b_i | R³ˣ³, R³ | 逐 Gaussian 局部仿射 |
| G, g | R³ˣ³, R³ | 每 LUT 一个全局仿射 |

**前向（§3.1, Eq.1-3）**：
```
p_i(x) = (2π)^(-3/2) |Σ_i|^(-1/2) exp(-½ d_i(x))     # 完整归一化高斯密度，d_i = Mahalanobis
w_i(x) = p_i(x)·o_i / (Σ_j p_j(x)·o_j + ε),  ε = 1e-6
f(x)   = Σ_i w_i(x)·(M_i x + b_i) + (G x + g)         # 残差/全局分支不可省！
output = clamp(f(x), 0, 1)
```
消融（§B.4）：**w/o Global w/o Residual 崩到 40.54 dB（−4.93）——这是全模型最大的结构性因素，复现优先级第一**。w/o weight norm 45.17、w/o opacity 45.43、w/o Global(G,g) 45.28。局部变换选 Affine（45.47/716p）是 RGB 常量（42.58）与 SH-3（46.78/1868p）间的折中。Gaussian 数量：8→37.01, 16→41.50, 32→45.47, 64→48.42, 128→50.31。

**Loss（§4.1）**：`L = L_rec + 10·L_hc + 0.001·R_sparse`
- L_rec = ‖ŷ−y‖₁
- L_hc：转 CIELab，C=√(a²+b²)，h=(a/C, b/C)，L_hc = C·(1−⟨ĥ,h⟩)（目标 chroma 加权 hue 余弦距离）
- R_sparse = −(1/N)Σ[o_i·log(o_i+ε) + (1−o_i)·log(1−o_i+ε)]（opacity 二元熵推向 0/1）
- 注意：loss 附加项总增益仅 +0.11 dB（45.36→45.47），换数据域时 λ 不鲁棒，别在这上面花时间。

**初始化（附录 A.1）**：
- μ：规则网格均匀铺满 [0,1]³（Uniform 45.47 vs Random 45.45，几乎无差）
- Σ：各向同性 σ=0.15，"via logarithmic Cholesky parameters"
- opacity：1.0；仿射：identity 矩阵 + 零 bias（措辞涵盖 M_i,b_i 与 G,g）

**训练配方（§4.1 + A.1）**：
- PyTorch，Adam，cosine annealing 全程，base lr 1e-3
- 单 GLUT：20 epochs，bs=1024；CGLUT：40 epochs，bs=8192
- CGLUT 差分 lr：style embedding 与 shared geometry 参数 0.1×（=1e-4），generator 1e-3
- Hard sample mining：epoch 5→20，按 L1 error 最高样本 mining 比例 10%→40% 线性升（增益 45.13→45.47）
- 硬件：单卡 RTX 4090 24GB；**训练 wall-clock 论文未给**（估算：单 LUT 分钟级）

**数据构造（§4.1）**：300 个 CC .cube（225 个 33³/32³ + 75 个 64³；7-LUT 子集取自 NILUT）。训练样本 = 均匀采 128³ 个 8-bit 颜色（=1024×2048×3 Hald 图），**其余 256³−128³ 颜色留作测试（=3584×4096×3 Hald）——split 是颜色空间层面，不是图像层面**。对 Hald 套 LUT 得 GT 对。

**CGLUT（§3.2 + A.2）**：embedding E∈R^{L×64}；shared encoder = 3 Linear（hidden 128 Large / 64 Small）+ ReLU；每类参数一个 head（mean head = 2 Linear，local color head = 3 Linear，global affine head 输出 12）。两模式：Full Generation vs Shared Geometry（表格 † 标注；μ/Σ 换成跨 style 共享 nn.Parameter）。Full 重建好（50.76 vs 49.35 dB），Shared Geo 的 embedding 插值 blending 更平滑（47.95 vs 47.56）。blending = embedding 线性插值后过 generator。

**Editing（§3.3）**：给 (c_in,c_out)：δ=c_out−f(c_in)；按 w_i(c_in) 取 top-K；α_k=w_k/Σw_j；仅 `b_k ← b_k + s·α_k·δ`（M_k 不动）。

**Sparse activation（§B.5）**：推理期先按欧氏距离 ‖x−μ_i‖ 取 **top 50%** 候选（比例策略，无固定阈值），仅对子集算精确 Mahalanobis，其余权重置 0，计算量约 −30%。训练不剪。Figure 10 有完整 tradeoff 曲线但数值只在图里。

**CUDA 推理（附录 A.3，无代码佐证）**：单 kernel 融合权重计算+局部变换+加权混合；block=(32,32)，第一维一像素/线程，第二维分摊 N 个 Gaussian；全局共享的 Gaussian 参数每 forward 预计算一次缓存 shared memory。训练路径保留 PyTorch autograd。**先纯 PyTorch 对齐精度，CUDA kernel 只在需要 FPS 数字时写。**

**评测协议（§4.1 + B.1）**：
- Hald：留出颜色上 PSNR / ΔE00 / ΔE76
- 自然图：FiveK **images #4501–#4600 共 100 张**，GT=传统 .cube 处理图，报 PSNR/SSIM/LPIPS/ΔE00/ΔE76
- 效率：GFLOPs @512²；FPS @512²/720p/4K，100 张 MIT5K 平均；压缩比 = .pth/.cube
- 对齐锚点：**GLUT-32 @75-LUT Hald ≈ 45.47 dB / ΔE00 0.41 / 0.49 GFLOPs**；GLUT-64 48.42；225-LUT GLUT-32 50.42；自然图 GLUT-32 45.92/0.997/0.003
- baseline 口径：NILUT/CNILUT 是**延长训练版**（10K/60K iters），复刻对比需同设置

### 2.3 复现顺序

1. 用 namedcurves 脚手架 + colour-science 写 Hald 生成器（np.mgrid 128³ → 1024×2048×3）
2. 纯 PyTorch 单 GLUT（几百行）：Eq.1-3 + A.1 超参，先只开 L_rec，验证残差/全局分支
3. 对齐 GLUT-32@75-LUT ≈ 45.5 dB（用 namedcurves utils/deltaE.py 保证 ΔE 口径与同组一致）
4. 加 L_hc/R_sparse/hard mining（预期总增益 <0.4 dB，不达标别恋战）
5. 扩 CGLUT → Editing → sparse culling → CUDA kernel

### 2.4 论文内部矛盾与实现决定

| 矛盾 | 论文两处说法 | 实现决定 |
|---|---|---|
| Cholesky 对角激活 | §3.1 Softplus vs A.1 "logarithmic Cholesky parameters" | **采 log/exp 参数化**：init = log(0.15) ≈ −1.897，实现最简；留 Softplus 版（init = softplus⁻¹(0.15) ≈ −1.7346）做一次对照消融 |
| opacity 激活 | §3.1 只说 ∈[0,1]，init=1.0 恰是边界（sigmoid 参数化则 logit=+∞ 不可行） | **raw 参数 + clamp[0,1]**（推断）；R_sparse 里的 ε 已保护 log(0) |
| 密度数值范围 | 完整归一化系数 1/√((2π)³|Σ|)，σ 小时可达 10² 量级 | 密度计算强制 fp32（即使整体 AMP）；与 ε=1e-6 搭配防 fp16 溢出 |

---

## 三、竞品训练配方对照表

| 项目 | InstantRetouch (CVPR26) | AceTone | SA-LUT (ICCV25) | 4D LUT (TIP23) | RSFNet (ICCV23) | DeepLPF (CVPR20) |
|---|---|---|---|---|---|---|
| **数据规模与来源** | ~200K 三元组（论文声称，未放出）：公开+网图经 MUSIQ/LAION 双阈值过滤，photo-finishing 管线退化，GroundingDINO+SAM2 局部掩膜，Qwen2.5-VL-72B 指令 | VQ: 10K licensed .cube；预训练 MSCOCO×LUT 在线组对；SFT Adobe-5K/PPR-10K jsonl；RL rl-8k；AceTone-800K 不放出 | vlog/RGB 图 + .cube 对，yaml 路径全空（**未发布**）；需先训 Style2VLog 网络 | FiveK MIT-Adobe5k-UPE：input=InputAsShotZero PNG，GT=Export_C_512，4500/500 | FiveK（zeroed_with_expertC / zeroed_as_shot，HF 有成品）+ PPR10K 360p；mask 变体另配 saliency/palette/semantic 掩膜 | FiveK Adobe-DPE 协议（split 是 best-guess txt），Expert C |
| **预处理** | teacher 256px crop+flip；蒸馏两阶段 512×512 直接 resize（改纵横比），无增广 | LUT 统一 32³ npy；VLM 输入 max_pixels 50176；LUT 在线增广 noise0.1/shift0.2/gamma0.8-1.2/warp0.1 | output_resolution 256；context 内部 512×512；lut_augment=true | 随机比例裁剪 0.6–1.0 + hflip 0.5 + 仅输入 brightness/saturation 0.8–1.2 | gt_size 256 crop + random_resize 0.5–1.0 + hflip；Normalize(0.5,0.5) → [-1,1] | normaliser=1，ToTensor，无颜色增广；batch=1 全尺寸 |
| **优化器/lr** | 全程 5e-5：teacher（AdamW wd1e-2）、Stage1/2（AdamW wd1e-7, eps1e-8, AMP） | VQ AdamW 3e-4（论文 2e-4，矛盾）；预训练 5e-5 cosine warmup 0.01；SFT 2e-5；GRPO trl 默认+KL β=0.01 | gen AdamW 1e-4 wd0.01，disc 1e-5；warmup 5ep 线性→cosine 到 1e-7；手动优化 clip 0.5；seed 3407 | Adam 1e-4 (0.9,0.999)，无 schedule | Adam 2e-4 wd0 (0.9,0.99)；CosineAnnealingRestartLR periods [250k]×4，eta_min 1e-7 | Adam 1e-4 (0.9,0.999)，无 schedule 无 wd |
| **bs** | teacher 2×ga4；Stage1/2 bs=2 单卡 | VQ 8；预训练全局 256（8×4×8）；SFT 全局 512；GRPO 4×2×8 | 1/GPU × 2 GPU DDP | 1 | 16（单 GPU） | 1（bs>1 需 --crop_size） |
| **步数** | teacher 20K steps；Stage1 1 epoch；Stage2 9 epochs | VQ 500 ep；预训练 2 ep；SFT 2 ep（论文说 1，矛盾）；RL 步数 unknown | max_epochs 300（发布 ckpt epoch 100 / step 4127466） | 1000 epochs | 1M iters | 上限 100000 ep，早停（发布 ckpt epoch 424），valid_every 25 |
| **loss 及权重** | Stage1: l_diff 0.2(MSE) + l_vsd 0.03(DMD, t∈U(140,750)) + l_lpips_diff 0.02 + l_clip_cont 0.01(τ=0.07)；Stage2 9-ep 日程: SmoothL1 [0.2→0.05] + lpips [0.04→0.01] + lpips_bila_diff [0.1→0.2] + penalty(warmup) + LapReg [2e-6→1e-7] + hist 0.5 | VQ: MSE + 1e-2·vq(β=0.25, EMA 0.99)；LM: CE；GRPO: ΔE 奖励 1/(max(ΔE,2)−1)（解析失败罚 ΔE=20）+ Pref 秩归一 + DeQA/5 | tv 1 + mn 1 + LPIPS(vgg) 10 + LAB 100 + hist 1 + GAN(BCE, 条件判别) 1/1 | MSE + 1e-4·(weights_norm+TV) + 10·mn（tv 不含 context 轴，mn 含） | L1 1.0（+ DiceLoss 1.0 监督 mask 分支，仅 saliency/palette/semantic 变体） | CIELAB L1 + 1e-3·(1−MS-SSIM(L))，window 5，MS-SSIM 权重归一化非标准（作者自认） |
| **GPU 时数** | unknown（runtime 测于 8×4090） | VQ 8卡~7h、VLM 8卡~3天（论文声称，卡型 unknown） | unknown | unknown | unknown（num_gpu=1） | unknown |
| **评测协议** | iRetouch 451 对（未公开）；fidelity=灰度+histmatch 后 SSIM/CW-SSIM/DISTS/GMSD；GPT-4o SC/PQ（Step1X-Edit 协议）；延迟 720p–4K 端到端（65ms@720p，**68ms@4K**）；推理 bila_grid_res=32（≠训练脚本默认 16） | AceTone-Bench Transfer 1024（HF: Vivre/AceTone-Bench-Transfer）+ PST-50；ΔE2000/LPIPS/PSNR/ColorSim(ONNX)；生成 do_sample T=0.01，缺 token 用**第一个** token 补齐到 64 | PST50（HF: zrgong/PST50）：LPIPS/PSNR/SSIM/H-Corr；视频 16 FPS | UPE test 500 @512px；PSNR=round×255 后 MSE；SSIM=skimage win11 gaussian；**发布代码 context 置零，论文数值不可按仓库复现** | FiveK test.txt / PPR10K a/b/c；calculate_psnr/ssim，crop_border=2；>720px 输入内部降到 512 再插回 | Adobe-DPE test 全分辨率逐图 PSNR/SSIM；bundled ckpt 23.90/0.911；split 是官方自认 best-guess |

---

## 四、数据管线规格

### 4.1 FiveK 480p 标准管线（checklist）

1. [ ] 从 GoogleDrive folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a 下载 480p 包（含 split txt）
2. [ ] 目录：`input/JPG/480p/*.jpg`、`input/PNG/480p_16bits_XYZ_WB/*.png`（cv2.imread(-1)，**保持 BGR**）、`expertC/JPG/480p/*.jpg`
3. [ ] split：train_input.txt + train_label.txt 合并 = 4500 训练；test.txt = 500 测试；GT 一律 expertC
4. [ ] 增广（sRGB 分支，抄 Zeng datasets.py）：H/W 独立随机 crop 比例 U(0.6,1.0) → hflip p=0.5 → 仅输入 brightness U(0.8,1.2) + saturation U(0.8,1.2)；XYZ 分支：brightness U(0.6,1.4)、无 saturation
5. [ ] 渲染算子：现代 torch 直接 `F.grid_sample`（Zeng issue #14 官方认可），别碰 trilinear_cpp 编译
6. [ ] PSNR 实现锁死：`torch.round(x*255)` 后 MSE，`10*log10(255²/mse)`，逐图平均，batch=1；论文级指标另跑 average_psnr_ssim.m
7. [ ] 480p 训练的模型可直接上 4K（README 声明）；跑效率数字用包内 10 张全分辨率图

### 4.2 PPR10K 含掩膜管线（checklist）

1. [ ] 只下 `train_val_images_tif_360p`（91GB）+ `masks_360p`（56MB），**不要碰 313GB raw**；GoogleDrive folders/1kB2OSAGy8uc0xUXaMKoPB0HMSc-rkrLW 或百度盘（码 mrwn）
2. [ ] 目录：`train/source_aug/*.tif`（16bit，含 5 版 XMP 增广）、`train/target_a|b|c/*.tif`（8bit）、`train/masks/*.png`（二值）、`val/...` 同构
3. [ ] split：前 8875 训练 / 后 2286 验证（按文件序）；expert 硬编码在 datasets.py `self.retoucher='a'`，换 expert 改源码
4. [ ] 配对约定：增广文件名靠 `img_name.split('_')` 回落到原始 target/mask——**自定义命名会静默错配**
5. [ ] 读图：16bit tiff cv2.imread(-1) 后显式 `[:, :, [2,1,0]]` 转 RGB；mask 按 `mask>0` 判定
6. [ ] 增广（datasets.py）：随机 crop 0.6–1.0 → resized_crop 448×448 → hflip 0.5，mask 同步
7. [ ] HRP：训练 loss 权重人像=5/背景=1（`weights[mask>0]=5`，对 fake*w vs real*w 做 MSE）；评测 calculate_metrics.m 里人像=1/背景=0.5——**两处口径不同勿混**
8. [ ] 指标必须走 MATLAB `calculate_metrics(source_dir, target_dir, mask_dir)`：结果 .png、GT .tif、mask imresize 对齐；GLC 按文件名下划线前 group id 聚 325 组（写死，只适用官方 val split）；纯 python 复刻 ΔEab 时 rgb2lab 用 D65 sRGB（skimage 一致，OpenCV 8-bit Lab 不一致）
9. [ ] 可选增广扩展：utils/data_augment_get_xmps.py（Temperature/Tint/Exposure/Highlights/Contrast/Saturation 六属性随机 XMP）+ Lightroom 批导出

### 4.3 4000-cube 语料管线（含 .cube 解析与 Hald 评测）

1. [ ] 收集 CC 授权 .cube（GLUT 论文未给清单，自建；NILUT Kaggle 数据集、HYouTube 400 LUT、各 CC 滤镜站为起点）
2. [ ] 解析：`colour.io.read_LUT_IridasCube`——行序 **R 变最快、B 最慢**，reshape order='F'；DOMAIN_MIN/MAX 非 0–1 的 log LUT 必须走 colour（pillow-lut 忽略 domain）
3. [ ] 统一化：AceTone `useful_tools/convert_luts.py` 任意尺寸→32³ npy+cube；**注意其 apply_lut 有 `lut[..., ::-1]` BGR 翻转约定，接自家渲染器前先对拍 identity LUT**
4. [ ] 去冗/抽代表：`select_luts.py`（PCA+KMeans）；按 AceTone 作者建议先规则滤掉单色/黑白 LUT
5. [ ] Hald 训练对生成（GLUT 协议）：`np.mgrid` 均匀采 128³ 颜色 → reshape 1024×2048×3 → 逐 LUT 套用（colour `table_interpolation_tetrahedral` 做 GT，训练内用 torch grid_sample）→ 图对
6. [ ] Hald 评测集：其余 256³−128³ 颜色 → 3584×4096×3；报 PSNR/ΔE00/ΔE76（ΔE 用 namedcurves utils/deltaE.py 对齐同组口径）——**这是颜色空间 split，不是图像 split**
7. [ ] 自然图评测：FiveK #4501–4600 共 100 张，传统 .cube 结果为 GT
8. [ ] 对拍验证：ImageMagick `magick hald:8 hald.png` + `-hald-clut` 输出 vs 自家渲染逐像素对比（发现行序/domain 错误）；identity LUT apply 必须恒等
9. [ ] Hald 规范红线：level N = 每通道 N² 采样（level 8→64³，图 512²）；存储 PNG-only 严禁 JPEG；编码 R 最快、左上黑右下白（相对 GL 纹理上下颠倒）
10. [ ] 蒸馏 GUI 调色为 LUT 时：只能捕获逐像素全局变换，**必须关掉锐化/降噪/局部遮罩/暗角**

### 4.4 合成局部编辑语料管线（借 IP2P/UltraEdit/Hist2Style 部件）

1. [ ] **指令层（抄 IP2P 配方）**：人写 ~700 条色彩/影调种子三元组（caption→指令→编辑后 caption，参考 human-written-prompts.jsonl 格式）→ 开源 LLM few-shot 批量扩展（GPT-3 finetune API 已死）→ 沿用 Moderation + caption 相同即弃 + 双去重
2. [ ] **指令 schema**：直接采纳 AnyEdit 格式（edit / edited object / input / output / edit_type / image_file / edited_file），edit_type 含 color_alter / tone
3. [ ] **掩膜层（搬 UltraEdit data_generation.py）**：edit_object → GroundingDINO（SwinB, box 0.3 / text 0.25）→ SAM ViT-H（sam_vit_h_4b8939.pth）→ torch.max 合并多实例；质检三件套：`check_mask_size` 面积占比 (0.01, 0.9)、`find_contours_number` 轮廓数 >500 弃、三种掩膜变体（SAM 精细 <150 轮廓 / 凸包轮廓 150–500 / bbox）；`edit_object='NONE'` → 全白掩膜（全局影调）
4. [ ] **图对层（我方优势，跳过生成模型）**：参数化色彩变换（LUT/曲线/HSL/白盒滤镜——可直接用 Harmonizer filter.py 或 RSFNet render() 原语）在掩膜内**确定性**施加 → 像素级完美 GT，无 P2P/inpaint 噪声；掩膜边缘用 kornia guided_blur 或 InstantRetouch 式高斯羽化软化
5. [ ] **质量门**：CLIP 方向过滤降级为 sanity check（**方向分对微弱色彩变化不敏感，0.2 阈值误杀**）；改用 UltraEdit 扩展版 ClipSimilarity 的 SSIM/DINOv2 做"变化过小不可感知"的下限剔除；高价值子集可选 OmniEdit 配方（VIEScore GPT-4o ≥9 门，附录含 prompt 模板）或 HQ-Edit metrics/eval.py（Alignment/Coherence）
6. [ ] **风格语料模板（抄 Hist2Style 数值配方）**：LLM 一次性生成风格名+描述库 → FLUX.1 Kontext [dev] 本地批量编辑（25K 图 × ~67 变体/图，同 prompt 跨图共享保风格一致）→ VGG19 特征 cosine >0.5 过滤（1.7M→1.1M）→ 同内容双变体互为(输入, 风格+GT)；loss 在 VGG 感知空间（MSE + sorted-1D-Wasserstein，Algorithm 1）
7. [ ] **反向指令**：色彩变换天然可逆，抄 HQ-Edit 的 edit/inverse_edit 双字段设计，一对生成
8. [ ] 现成语料补充：OmniEdit-1.2M 按 task 筛 attribute modification；AnyEdit 筛 color_alter/tone；MagicBrush 的人工 mask_img 做「指令→掩膜」监督 GT

---

## 五、读出侧接入手册

### 5.1 training-free 稠密读出（clone → 第一张 s 图）

**GEM（首选，2 步）**
1. `pip install gem_torch`（pin `open_clip_torch<=2.24`，新版 create_model 签名变动会错位）
2. README snippet 6 行：`create_gem_model(...)` → `gem_model(image, text)` → [B, num_prompt, W, H]
- 坑：openai B/16 权重本身带 quickgelu，模型名别加 -quickgelu 后缀（那是 metaclip 的）；输出是 min-max 热图非 softmax，做分割走 README threshold（≈0.85）流程
- 可调：gem_depth=7（最后 7 层换 SelfSelfAttention）、ss_attn_iter、ss_attn_temp

**SCLIP（3 步单图 / 5 步评测线）**
- 单图：clone → `pip install ftfy regex` + torch → 自写 ~15 行：`CLIP.encode_image(image, return_all=True, csa=True)` @ text_embeds.T
- 改造点：`clip/model.py` 的 `VisionTransformer.custom_attn`——CSA = `softmax(q@qᵀ·scale) + softmax(k@kᵀ·scale)`，仅最后一层 resblock 生效
- 评测线：mim install mmcv==2.0.1 mmengine==0.8.4 mmsegmentation==1.1.1 + yapf==0.40.1（锁版，新卡需源码编译 mmcv）→ 改 configs/cfg_DATASET.py 的 data_root → `python eval.py`；关键配置 logit_scale=65, prob_thd=0.1

**NACLIP（4-5 步）**
- 同 SCLIP 骨架；改造点 `clip/model.py` 的 `set_params(arch, attn_strategy, gaussian_std)`，naclip 分支 = `k@kᵀ·scale + omega`（高斯邻域加性偏置，`gaussian_window`+`get_attention_addition` 约 40 行可独立搬走，含变分辨率缓存）
- 主结果命令：`bash test_all.sh reduced naclip 5 on {gpu} {log}`；额外锁 numpy==1.26
- 消融白送：attn_strategy ∈ {naclip, nonly, kk, csa, vanilla}

**ClearCLIP（4 步）**
- conda py3.10 → `pip install -r requirements.txt` → `python demo.py`
- 改造在 vendored `open_clip/transformer.py`（去残差 + self-self attn + 丢 FFN，调用签名 `encode_image(img, model_type, ignore_residual)`）
- 坑：vendored open_clip 与 pip 版同名——repo 根目录必须在 sys.path 最前；S-Lab License

**ProxyCLIP（4 步，dino 分支免下权重）**
- clone → conda → pip -r → `python demo.py`（DINO 走 torch.hub 自动拉）
- 机制：`encode_image(img.half(), external_feats=VFM特征, beta=1.2, gamma=3.0)`；VFM 可换 SAM/DINO/DINOv2/MAE（configs/base_config.py `vfm_model`）
- 双归一化别搞错：CLIP 输入 CLIP mean/std，VFM 输入先 UnNormalize 再 ImageNet mean/std；全 fp16，老卡溢出改 fp32

**上采样 + 细化（组合拳收尾）**
- `torch.hub.load('mhamilton723/FeatUp', 'maskclip', use_norm=False)`（maskclip 只有 no_norm）或 `torch.hub.load('andrehuang/loftup', 'loftup_clip'/'loftup_siglip2', pretrained=True)`；DINOv3 只有 JAFAR 有权重
- 最后 `kornia.filters.guided_blur(guidance=RGB原图, input=s图, kernel_size, eps=1e-4~1e-2, subsample=4~8)`——eps 是归一化域量级，s 图低频 subsample 开大无损

### 5.2 SasP / MasP（[SEG] 系读出）

**SasP（E11，抄 READ）**
- 从 `model/READ.py` 整段拷 `compute_similarity_map`（CLIP_Surgery 加权点积：prob 温度 ×2 softmax → w=prob/mean → feats 加权求和）+ `similarity_map_to_points`（576→24×24→down_sample=2→min-max→t_pos=0.8/t_neg=0.2 取正负点，num_points=30，Discrete_to_Continuous 精化），共 ~150 行，零参数
- 输入 = [SEG] token 的最后一层 hidden state（**未过** text_hidden_fcs）+ 576 个图像 token hidden states
- 超参位置：model/READ.py 837-839 行（num_points/t_pos/t_neg）、764 行（down_sample），注释切换，不在命令行
- 计算上不依赖 SAM；但 24×24 图直接当 mask 的质量无验证，需自测

**MasP（E12，抄 UGround）**
- `model/UGround.py`：`compute_similarity = torch.einsum("sd, sid -> si", seg_token_embeds, seg_image_token_embeds)`（裸点积，比 SasP 更薄）→ `get_similarity_map`：min-max → 24×24 → bilinear 336 → 按宽高比裁剪 → resize 原尺寸（这份可直接当输出）
- 监督：`SimiLoss` 高斯软化 GT heatmap（ksize=31, σ=7.0，pos_weight 自动取 neg/pos 比）的 BCE+Dice——**tools/simi_loss.py 是独立离线工具**（吃 layer{N}/{id}.npy + json GT），零改动用来给我方渲染器读出图打分，不需要 SAM
- **红线：released PPM.py 的 RL 选层是死代码**（mode2/3/4 算完立即 return mode1），官方脚本 `--mode=1 --eval_legacy`——只实现「最后一层 + soft mask」即官方发布行为
- 两者共享同一张点积相似度图，E11→E12 增量成本很小

### 5.3 Qwen 系 VLM 适配点

- 模块路径（新版 transformers）：`model.model.visual.blocks[i]`（视觉塔）/ `model.model.language_model.layers[i]`（LLM）/ 终端 `model.model.language_model.norm`（RMSNorm）/ `model.lm_head`；旧版是 `model.model.layers`，hook 代码按安装版本探测
- 视觉 token 定位：`vision_start=151652 / vision_end=151653 / image_token=151655`；hidden_states[0] 已是 merge 后序列
- logit lens 正确写法：`lm_head(language_model.norm(hidden_states[l]))`——**必须先过 RMSNorm**；tie 配置逐 config 核实：2.5-VL-3B tied / 7B untied / Qwen3-VL-8B untied（tied 时早层有输入自反射偏置）
- Qwen3-VL 陷阱：`deepstack_visual_indexes [8,16,24]`——视觉特征在这些层二次注入，lens 读数突变，解释时必须标注；config 是 text_config 嵌套
- SasP/MasP 接 Qwen：patch 网格不是 24×24（patch14/16 + 2×2 merge），num_patches 动态算（LISA 的 255 pad 硬编码不能抄）

### 5.4 attention 导出 × FlashAttention 互斥

- **新版 transformers：FA2 与 SDPA 在 output_attentions=True 时都只 warning + 返回 None，不回退**（旧版 SDPA 才自动回退 eager——同代码换环境结果不同）
- 唯一可靠姿势：`from_pretrained(..., attn_implementation="eager")` + `output_attentions=True`；或整体 FA2 加载、仅分析步 `model.set_attn_implementation("eager")`
- 视觉塔：`Qwen2_5_VLVisionAttention.forward` 内 `attn_output, _ = attention_interface(...)` **明文丢弃权重**——视觉 attention map 要 eager 下在 `visual.blocks[i].attn` 挂 hook 用 q,k 自行重算 softmax，且处理 window attention 的 cu_seqlens 分段与 `fullatt_block_indexes [7,15,23,31]` 全注意力层
- 显存 O(L·H·N²)，图像 token 多时按层/按 head 选择性导出；跨后端对比 attention 熵等指标须固定后端

---

## 六、探针工具链

### 选型结论

| 需求 | 选型 | 理由/替代 |
|---|---|---|
| 逐层线性探针 | **自实现**（sklearn LogisticRegression/SGDClassifier + torch MLP） | 无"官方最佳实践仓库"；配方抄 Hewitt/Voita：epochs 40 / CE / lr 1e-3 / wd 0 / bs 20-40；control task 要点仅两条——type-level 固定随机标签（映射持久化！）+ 探针超参与真实任务严格一致，selectivity = 真实 acc − control acc |
| MDL / online codelength | **自实现 ~50 行** | 官方实现本质是 config 里 `inds: 0.001,0.002,...,0.5,1` 的外层循环：逐段训练、累计下一段 −log p，首段按 uniform code 计费（易漏）；对数据顺序敏感，固定 shuffle seed + 多 seed 平均；不要跑 2020 年原环境 |
| 线性概念擦除 | **concept-erasure（LEACE，pip）** | 一次闭式解、保证所有线性分类器失效、损伤最小；`LeaceFitter.update()` 流式喂大激活集，拟合用 float64 再 cast 回模型 dtype。INLP 的 `src/debias.py`（numpy+scipy+sklearn 零依赖）留作对照——注意 INLP 每轮只删 1 维、要几十~几百轮、min_accuracy 设 majority 基线 |
| activation steering | **repeng（pip）** | 层路径后缀匹配 `"model.layers"` 大概率直接命中 Qwen2.5-VL 语言层（`model.language_model.layers` 后缀吻合，源码推断未官方测试）；不行则 `model.repeng_layers = model.model.language_model.layers` 一行兜底。**ControlModel 就地改模型结构——先做完所有 hook/探针采集再包装**；MoE 不支持；只测过 left padding。学术引用挂 RepE（andyzoujm/representation-engineering） |
| VLM 全链路访问/干预 | **nnsight** | TransformerLens 确认零 VLM 支持（README 无一字多模态）。nnsight `NNsight(hf_model)` 按 module 路径寻址 output、直接赋值干预、`.source` 截 attention_interface 调用；坑：必须按执行顺序访问、值要 `.save()`、多 invoke 输入传第一个 invoke |
| tuned lens | **tuned-lens（pip）+ 自写 VLM 训练循环** | 推理端：给 `tuned_lens/model_surgery.py` 的 `get_final_norm`/`get_transformer_layers` isinstance 白名单加 Qwen2_5_VLTextModel → `.norm`/`.layers` 两个 elif，~10 行。训练端 CLI 绑死 AutoModelForCausalLM + the_pile 纯文本，VLM 必须自写（图文 batch + processor + KL 对齐末层 logits）；配方照抄：250 步 × 2^18 tokens，SGD nesterov lr_scale 0.1（等效 lr≈1.0）/wd 1e-3，linear schedule，translator 全零初始化残差参数化 `h + T(h)`；**纯文本训的 lens 在视觉 token 位置外推不可信，训练数据必须含图像 token 真实分布**；Qwen-VL 无现成预训练 lens，必训 |
| logit lens | 自实现 3 行 | 见 5.3；视觉塔侧可视化范式参考 ViT-Prisma 的 Emoji Logit Lens notebook |

### 安装清单

```bash
pip install concept-erasure repeng nnsight tuned-lens        # 探针/干预四件套
pip install kornia colour-science pillow_lut gem_torch       # 渲染/读出
pip install "open_clip_torch<=2.24"                          # GEM 兼容
pip install git+https://github.com/mhamilton723/FeatUp       # 不在 PyPI
# scikit-learn / scipy 常规；INLP debias.py 直接拷文件
```

---

## 七、信息缺口

### 无代码、只能按论文自写

| 项目 | 缺什么 | 行动 |
|---|---|---|
| **GLUT** | 代码、权重、.cube 来源清单、训练 wall-clock、Figure 10 sparse tradeoff 数值（只在图中） | 按第二节规格书复现；订阅 CVC-Color/glut issue #1；邮件 Danna Xue / Javier Vazquez-Corral（CVC/UAB）问三件事：Cholesky 激活到底哪个、opacity 参数化、.cube 清单 |
| SA-3DLUT | 全部（含 UNet 两头具体结构） | 只当论文层设计参考；loss 权重 1e-4/10/0.005/0.05 可抄 |
| DY-LUT | 全部；共享 encoder 层数只在图 2 | preprint 未评审，仅抄超参思路（双 lr 分组、βMN=2.0） |
| StatLUT / MRStyle / LumiVideo | 全部；MRStyle 连 loss 权重都没给 | 只搬设计（2304 维统计特征、identity-Query Seq2Seq、teacher 自蒸馏三元组配方） |
| Hist2Style | 代码在 Adobe 审批中 | 盯 dgalor/hist2style；配方数值已抄全可先动工 |
| iRetouch 500 对 | 多轮检索不可得（无仓库无 HF，issue #4 索要无回复） | **管线不依赖它**；替代：FiveK 任取 500 子集或 PPR10K val |

### 有代码但关键件缺失

- **InstantRetouch**：权重私有（HF 401）、200K 数据、iRetouch、属性库、评测脚本全缺；**三处论文-代码不一致**（joint 训练 vs `--only_bila`；VSD 三段课程 vs 单区间 (140,750)；EMA 声称 vs 无）——复刻按脚本为准；teacher base model 未写明（推断 timbrooks/instruct-pix2pix，需自验）
- **AceTone**：800K 数据 + 10k LUT 库确定不放（版权）；只有 PST Preview 权重（无 IGG/RL）；`get_path` 未定义（open issue #3）需自补；超参三处论文-代码矛盾（VQ lr/bs、vq_weight、epoch 数）——以脚本为准、按论文数值扫一遍
- **SA-LUT**：训练数据管线 + Style2VLog 前置模型未发布，第三方无法重训主模型
- **WaveLUT**：训练代码"upcoming update"至今未出，loss/优化器不可核
- **4DLUT**：发布代码 context 置零（issue #1/#5 无答复），论文数值按仓库不可复现——对照臂用 SA-LUT 的 clut4d.py 自实现
- **NILUT**：CNILUT 多风格训练代码"released soon"状态，只能从 notebook 复原
- **HQ-Edit**：数据构造代码永远的"Code Refactoring"占位符
- **OmniEdit**：inpainting 专家权重、EditNet、InternVL2 打分器全未发布
- **MagicBrush**：无采集工具（DALL·E 2 众包）
- **JAFAR**：config 目录两次抓取失败，训练超参未核实（论文附录有）
- **F-LMM**：SAMWrapper 内部可训练范围未核实；transformers==4.39.1 硬 pin

### 需自行试错验证

1. GLUT：log vs Softplus Cholesky 两种参数化对结果的影响（论文没做这个消融）
2. GLUT：opacity raw+clamp 推断是否成立（init=1.0 反证 sigmoid 不可行，但 clamp 的梯度死区需观察）
3. Qwen2.5-VL 视觉塔 attention 经 `_can_record_outputs` 是否真能导出（forward 明文丢弃权重，**需实测**；备选方案 hook 重算已给出）
4. repeng 后缀匹配在 Qwen2.5-VL 上是否唯一命中（未官方测试）
5. SasP 相似度图脱离 SAM 直接当 mask 的质量（READ 从未这么用）
6. 24×24/自家 patch 网格的 s 图 + guided_blur 细化链在色彩编辑掩膜上的边缘贴合度
7. tuned lens 在视觉 token 位置的训练数据配比（论文与库均未处理该 setting）
8. CLIP 方向过滤对微弱色彩编辑的实际误杀率（决定合成语料质量门的阈值）

### 检索卫生警示

本轮调研中搜索引擎**编造过**：mv-lab/GLUT、HuiZeng/3D-LUT（作为 SA-3DLUT 仓库）、megvii-research/WaveLUT、AnyEdit/AnyEdit、HDRNet 复刻的假 star 数、Gehler-Shi "2025 re-release"。**任何本档案之外的新 URL/数字必须打开核实后再引用。**
---

## 附录 B：来源链接总表（全部经 web_fetch 一手核实；2026-08-02）

### 官方代码仓库
| 项目 | 链接 | 状态 |
|---|---|---|
| GLUT | https://github.com/CVC-Color/glut | ⚠️ 占位壳（零提交）；~~github.com/mv-lab/GLUT~~ 为检索引擎编造 |
| NamedCurves（GLUT 同组模板） | https://github.com/davidserra9/namedcurves | 可用 |
| InstantRetouch | https://github.com/OpenImagingLab/InstantRetouch | 代码可用；权重/数据未放 |
| AceTone | https://github.com/martian422/AceTone | 含 VQ-VAE 权重 |
| SA-LUT | https://github.com/Ry3nG/SA-LUT | clut4d.py + quadrilinear CUDA |
| Zeng 3D LUT | https://github.com/HuiZeng/Image-Adaptive-3DLUT | FiveK 480p 数据入口 |
| AdaInt / SepLUT | https://github.com/ImCharlesY/AdaInt ｜ https://github.com/ImCharlesY/SepLUT | 环境锁老 |
| CLUT-Net | https://github.com/Xian-Bei/CLUT | — |
| ICELUT | https://github.com/Stephen0808/ICELUT | — |
| NLUT | https://github.com/semchan/NLUT | class CLUT 可单独拿 |
| CSRNet | https://github.com/hejingwenhejingwen/CSRNet | 零编译 |
| PPR10K | https://github.com/csjliang/PPR10K | 数据+官方指标 |
| RSFNet | https://github.com/Vicky0522/RSFNet | 权重 https://hf.co/Vicky0522/RSFNet-models ｜ 数据 https://hf.co/datasets/Vicky0522/MIT-Adobe5k-for-RSFNet |
| Harmonizer | https://github.com/ZHKKKe/Harmonizer | CC BY-NC-SA |
| DeepLPF | https://github.com/sjmoran/deeplpf-image-enhancement | — |
| Exposure | https://github.com/yuanming-hu/exposure | filters.py |
| VideoColorGrading | https://github.com/seunghyuns98/VideoColorGrading | — |
| Hist2Style | https://github.com/dgalor/hist2style | 代码 Adobe 审批中 |
| InstructPix2Pix | https://github.com/timothybrooks/instruct-pix2pix | 过滤器可用 |
| UltraEdit | https://github.com/HaozheZhao/UltraEdit | 掩膜链 |
| MagicBrush | https://github.com/OSU-NLP-Group/MagicBrush ｜ https://hf.co/datasets/osunlp/MagicBrush | — |
| OmniEdit | https://hf.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M ｜ VIEScore https://github.com/TIGER-AI-Lab/VIEScore | 官方仓库零代码 |
| AnyEdit | https://github.com/DCDmllm/AnyEdit ｜ https://hf.co/datasets/Bin1117/AnyEdit | — |
| GEM | https://github.com/WalBouss/GEM（pip gem_torch） | 当天 |
| SCLIP | https://github.com/wangf3014/SCLIP | — |
| NACLIP | https://github.com/sinahmr/NACLIP | — |
| ClearCLIP | https://github.com/mc-lan/ClearCLIP | S-Lab License |
| ProxyCLIP | https://github.com/mc-lan/ProxyCLIP | — |
| FeatUp | https://github.com/mhamilton723/FeatUp | torch.hub |
| LoftUp | https://github.com/andrehuang/loftup | SigLIP2 权重 |
| JAFAR | https://github.com/PaulCouairon/JAFAR | 唯一 DINOv3 权重 |
| READ (SasP) | https://github.com/rui-qian/READ | model/READ.py |
| UGround (MasP) | https://github.com/rui-qian/UGround | PPM RL 是死代码 |
| colour-science | https://github.com/colour-science/colour | .cube 解析 |
| pillow-lut | https://github.com/homm/pillow-lut-tools | 忽略 DOMAIN |
| concept-erasure | https://github.com/EleutherAI/concept-erasure | LEACE |
| repeng | https://github.com/vgel/repeng | steering |
| nnsight | https://github.com/ndif-team/nnsight | VLM 干预 |
| tuned-lens | https://github.com/AlignmentResearch/tuned-lens | 需自写 VLM 训练 |
| INLP | https://github.com/shauli-ravfogel/nullspace_projection | debias.py 拷文件 |
| RepE（steering 学术引用） | https://github.com/andyzoujm/representation-engineering | — |

### 数据与基准
| 资产 | 链接 |
|---|---|
| FiveK 480p 成品包 | Zeng 仓库 README → GoogleDrive folders/1Y1Rv3uGiJkP6CIrNTSKxPn1p-WFAc48a（百度盘码 5fyk） |
| PPR10K 360p+masks | PPR10K 仓库 README → GoogleDrive folders/1kB2OSAGy8uc0xUXaMKoPB0HMSc-rkrLW（百度盘码 mrwn） |
| AceTone-Bench-Transfer | https://hf.co/datasets/Vivre/AceTone-Bench-Transfer |
| PST50 | https://hf.co/datasets/zrgong/PST50 |
| MIT-Adobe FiveK 原始 | https://data.csail.mit.edu/graphics/fivek/ |
| Cube+ | https://ipg.fer.hr/ipg/resources/color_constancy （NUS-8 同页系列） |
| iRetouch 451 对 | ❌ 不可得（无仓库无 HF，issue #4 无回复） |

> 编造警示（勿引）：mv-lab/GLUT、HuiZeng/3D-LUT（作为 SA-3DLUT）、megvii-research/WaveLUT、AnyEdit/AnyEdit、假 star 数、Gehler-Shi "2025 re-release"。本表之外的新 URL 使用前必须打开核实。
