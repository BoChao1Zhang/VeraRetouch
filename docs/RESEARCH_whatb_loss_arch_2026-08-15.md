# whatb 分支调研报告：条件化颜色变换的 loss 配方与 query-based 生成结构

日期：2026-08-15
范围：what 侧「指令 → 全局颜色变换（GLUT）」的 Q1（loss）与 Q2（结构）两个问题
本文件只陈述「某工作写的是什么」「某结构是怎么接的」「对应关系是什么」。不含结论、不含推荐。

---

## 1. 交付说明

### 1.1 本轮回答的两个问题

- **Q1（loss）**：指令→LUT 是一对多映射；同时本项目多项 loss 量纲失衡（10·L_hc 等效约 325×L_rec）。调研「这类条件化颜色变换生成任务，已发表工作的 loss 写的是什么」。
- **Q2（结构）**：当前生成器是「单个 pooled z ∈ R^2560 → MLP → 22N+12 个参数」。调研以 transformer 为主的结构，以及本仓库 where 分支 ST_LANG 结构向 GLUT 的机制对照。

### 1.2 方法

八个专题并行调研。检索用 grok_search MCP（`web_search`），事实核实一律用 `web_fetch` 或 `curl` 拉取原始字节（arXiv PDF 经 `pdftotext -layout` 抽取；GitHub 代码经 `raw.githubusercontent.com`；arXiv 存在性经 `export.arxiv.org/api/query`；GitHub 仓库存在性经 `api.github.com`）。每条 finding 标注 `opened=true/false` 与逐字摘录。随后对全部 findings 做一轮引文核查（逐条重新打开原始来源比对 quote 与数值）。

本报告中出现的全部权重数值、公式、文件路径，均来自 `opened=true` 的条目。

### 1.3 核查阶段剔除/修正的引文（BAD 列表，逐条）

**BAD-1｜AceTone（arXiv:2604.00530）GRPO 颜色奖励公式（已修正）**
原 finding 的 quote 写作 `The reward is calculated as max(2,1∆E)−1`，并附推测「疑为 max(2,ΔE)^{-1}」。核查判定：该推测是错的。原文（§3.3 Reinforcing AceTone with GRPO）为分式 **r_color = 1 / (max(2, ΔE(I_gt, L_pred(I))) − 1)**；pdftotext 抽取形式是 `1 max(2, ∆E)−1`（分子 1、分母 max(2,ΔE)−1）。按原式 ΔE<2 时 reward = 1/(2−1) = 1 取到最大值，与原文 "the maximum value is given when ∆E < 2" 一致。同一批 findings 中 `retrieval-classify-then-refine` 专题写的 `1/(max(2,ΔE)−1)` 是正确的，两条原本自相矛盾。**（核查修正）** 本报告一律采用 `1/(max(2,ΔE)−1)`。该条其余内容（β=0.25、K=256、EMA、Adam lr 2e-4 / batch 64 / 500 epoch、Qwen2.5-VL-3B 冻结视觉编码器、ΔE 用 CIEDE2000）核对无误。
来源：https://arxiv.org/pdf/2604.00530

**BAD-2｜CogACT（arXiv:2411.19650v1）N 的取值（已修正）**
原 finding 写「context length = 1 + N，默认 N=16」。原文逐字为：`In practice, the number of predicted future actions is set to a small value (N=15 by default), leading to a context length of N+2=17 for the action module.` 论文的 N=15 指未来步数，context length 写作 N+2=17。**（核查修正）** 本报告写作：默认 N=15，context length = N+2 = 17，展开为 1 个 cognition token + 16 个 noisy action token。
来源：https://arxiv.org/html/2411.19650v1

**BAD-3｜Splatter Image 输出通道数（已修正）**
原 finding 写「最后一层换成 1×1 卷积输出 12+k_c 通道」，与同条列出的 `split_dimensions=[1,3,1,3,4,3]`（合计 15）自相矛盾。核查 `scene/gaussian_predictor.py` 的 `get_splits_and_inits`：`with_offset` 分支 split_dimensions=[1,3,1,3,4,3]=15（depth 1 + xyz_offset 3 + opacity 1 + scale 3 + rotation 4 + rgb 3），`without_offset` 分支才是 [1,1,3,4,3]=12；`configs/default_config.yaml` 中 `network_with_offset: true`、`network_without_offset: false`。SH 部分 `sh_num_rgb = ((max_sh_degree+1)^2 − 1)*3`，默认 `max_sh_degree: 1` → 9 通道。**（核查修正）** 默认走 15 + 9 = 24 通道；「12 + k_c」只对应被关闭的 without_offset 变体。
来源：https://raw.githubusercontent.com/szymanowiczs/splatter-image/main/scene/gaussian_predictor.py

### 1.4 检索引擎编造记录（调研过程中自查、未采信、不进本报告正文）

以下条目由 grok `web_search` 给出，经 arXiv API / GitHub API / arXiv 全站检索核实为不存在或张冠李戴，全部未写入 findings，亦不作为任何方法细节的来源：

| 编造内容 | 核实方式与结果 |
|---|---|
| `Text2LUT: Text-driven 3D LUT Generation via CLIP` | arXiv API 全文检索 n=0；arXiv 全站检索返回 "produced no results"（https://arxiv.org/search/?searchtype=all&query=Text2LUT） |
| `InstructColor: Instruction-guided 3D LUT for Color Grading` | arXiv API 全文检索 n=0 |
| `arXiv:2303.12345 Differentiable Color Difference Loss for Image Enhancement` | 不存在（占位式伪造） |
| `arXiv:2303.12345 DiffRetouch: Using Diffusion Models for Real-World Image Retouching` | 同一伪造 ID 的第二次使用；真实条目为 arXiv:2407.03757、标题 `DiffRetouch: Using Diffusion to Retouch on the Shoulder of Experts` |
| `arXiv:2208.04567` / `arXiv:2109.03456`（soft-CIEDE2000） | 不存在 |
| GitHub `serkansulun/differentiable-color-loss` | GitHub API 404 |
| GitHub `xichenpan/MetaQueries` | GitHub API 404；真实仓库为 `facebookresearch/metaquery` |
| `LUT-GAN: Learning 3D Lookup Tables for Photo Retouching` | 未能核实，未打开、未采信 |
| 把 GVGEN 标为 `arXiv:2405.17894` | 该 ID 实为一篇 VLM 越狱论文；GVGEN 真实 ID 为 2403.12957 |
| 把 Gamba 的 GambaFormer 安到 GaussianAnything 头上 | GaussianAnything（2411.08033）实为 point-cloud latent flow matching + VAE |
| 把 Chang et al. "Principled Weight Initialization for Hypernetworks" 标为 `arXiv:1910.03999` | 该 ID 实为一篇液晶相变论文（Ferroelectric-ferroelastic phase transition in a nematic liquid crystal） |
| 「CLIP-LUT / Text2LUT / InstructColor 均无 GitHub 仓库」 | 未再核实；CLIP-LUT 的真实条目是 arXiv:2311.03943 `CLIP Guided Image-perceptive Prompt Learning for Image Enhancement`（其方法自称 CLIP-LUT） |

---

## 2. 问题抽象

### 2.1 项目现状（只列数字）

任务：冻结 Qwen3-VL-4B 在 `<seg_color>` token 上读出单个 hidden state z ∈ R^2560 → 生成 f: [0,1]^3 → [0,1]^3。参数化 GLUT，N=48 个 3D 色域高斯（μ 3 + Cholesky 6 + opacity 1 + 局部仿射 M 9 + b 3）+ 全局仿射 G 9 + g 3，共 22N+12 = 1068 维。监督：B=32 个 LUT × Q=256 个查询色 = 8192 色，在函数值空间比 f(x) 与 L_l(x)。训练集 93,934 条（normal-only）。

headline ΔE00（越小越好）：

| 列 | CARRIER 臂 | IDGATE 臂 |
|---|---|---|
| 本臂 headline | 7.895 | 10.149 |
| B0 恒等（什么都不做） | 8.293 | 8.293 |
| B1 训练集平均 LUT（不看指令） | 7.632 | 7.715 |
| B2 库内随机抽一个 LUT | 10.099 | 9.971 |
| B3 按 (major,minor) 类别桶检索 | 6.155 | 6.064 |
| B4 库内最优（oracle） | 0.825 | 2.827 |

其它已测数字：
- CARRIER：L_rec 0.5596 → 0.1770（117,399 步）；L_hc 25.42 → 2.106（−92%）。
- 恒等变换 f≡identity 时 L_rec = 0.1756。
- IDGATE：三个负控制 delta 全为 0。
- AFFONLY：head_color 末层隐层 ReLU 全局死亡率 0.469(step0) → 0.938(s500) → 1.0000(s1300)；之后只剩末层 bias，32 个样本预测逐位相同，梯度精确为 0。
- 当前 loss：L_total = L_rec + 10·L_hc + 0.001·R_sparse；L_rec = ‖f(x) − L_l(x)‖₁（RGB，对颜色与通道取均值）；L_hc = mean(C·(1 − cos Δhue))，C 为未归一化 CIELab chroma，实测 mean C = 32.5 ⇒ 10·L_hc 等效约 325×L_rec。
- 梯度裁剪：pre-clip 范数 ~13，阈值 1.0/0.5/0.1 使 100% 的步触发；Adam 对均匀缩放免疫。
- 已实证有效的唯一一档：c = c / c.detach().mean()。

### 2.2 Q1 抽象

- 同一句「暖一点」在库里对应多个 LUT；L1 的最优解是条件中位数，L2 的最优解是条件均值。
- 当前 headline 7.895 vs B1（训练集平均 LUT）7.632。
- 多项 loss 的等效权重比为 325 : 1 : 0.001（L_hc : L_rec : R_sparse 的等效量级）。

### 2.3 Q2 抽象

- 条件通路：z(2560) → LayerNorm+Linear → u(64) → 3 层 128 宽 MLP encoder → 每组一个 2~3 层 MLP head（head_color 3 层输出 12N；head_mu / head_cov / head_opacity / head_global 各 2 层）。
- 已观测：唯一条件通路 head_color 的末层隐层 ReLU 全局死亡率达到 1.0000，不可逆。
- ST_LANG（本仓库 where 分支，q3vl/whereb/amort/uniq4.py + uniq4b.py）：K=8 个新 token id（embedding 可训练，经 forward hook 写入，resize 出来的行本身冻结）→ 取其最后一层 hidden（2560 维）作为 query → Mask2Former 式 RefineLayer（pre-norm；masked cross-attention → query self-attention → FFN；两个 attention 的 out_proj 与 FFN 末层零初始化）→ 变体叠加 language-only LoRA（正则 `^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj)$`）。

---

## 3. Q1：loss 调研

### 3.1 同族工作的 loss 配方对照表

一行一篇。「监督位置」区分图像空间 / LUT 参数空间 / 函数值空间。

| 工作 | 重建项 | 色彩空间 / 监督位置 | 感知项 | 方向/色相项 | 正则项 | 各项权重 | 来源（文件路径 / URL） |
|---|---|---|---|---|---|---|---|
| Image-Adaptive 3D LUT (TPAMI 2020) | MSE | sRGB 图像空间 | 无 | 无 | weights_norm=mean(pred²)、TV_3D、单调性 mean(ReLU(dif)) | `loss = mse + 1e-4*(weights_norm + tv_cons) + 10.0*mn_cons` | `image_adaptive_lut_train_paired.py:33-34,50,133,222`；`models.py:340-359`｜https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/image_adaptive_lut_train_paired.py |
| 同上（论文正文） | Lmse Eq.(7) | sRGB | 无（ΔE* 仅评测） | 无 | Rs（L2 距离）、Rm | λs=0.0001、λm=10，网格 {0,1e-5,1e-4,1e-3,1e-2,1e-1}×{0.1,0,1.0,10,100,1000} | https://arxiv.org/pdf/2009.14468 |
| CLUT-Net (ACM MM 2022) | l1 = (fake−expert).abs().mean() | sRGB 图像空间 | 无 | `cos = (1 − cosine_similarity(fake, expert, dim=1)).mean()`，dim=1 为 RGB 通道维 | TVMN 默认 `--tvmn=False` 关闭 | l1 权重 1、cos 权重 1（`sum(loss_ls).backward()` 无缩放）；开 tvmn 时 lambda_smooth=1e-4、lambda_mn=10.0 | `utils/losses.py:10-11,24-25`；`parameters.py:21`；`train.py:48,85-87`；`models.py:221-262`｜https://raw.githubusercontent.com/Xian-Bei/CLUT/main/utils/losses.py |
| AdaInt (CVPR 2022) | MSELoss | sRGB 图像空间 | 无 | 无 | sparse=mean(weights²)、smooth(TV)、monotonicity | sparse_factor=0.0001、smooth_factor=0（关）、monotonicity_factor=10.0、recons loss_weight=1.0 | `adaint/configs/fivekrgb.py:17-20`；`adaint/model.py:359-369`｜https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/configs/fivekrgb.py |
| AdaInt（论文 Eq.6） | MSE | sRGB | 无（ΔEab 仅评测） | 无 | Ls、Lm | `L = L_r + 0.0001×L_s + 10×L_m` | https://arxiv.org/pdf/2204.13983 |
| SepLUT (ECCV 2022) | MSELoss（正文写明不加其它约束） | sRGB 图像空间 | 无 | 无 | 仅 sparse | sparse_factor=0.0001、smooth_factor=0、monotonicity_factor=0 | `seplut/configs/fivekrgb.py:18-21`｜https://arxiv.org/pdf/2207.08351 |
| 4D LUT (arXiv:2209.01749) | Lr = 平方误差 Eq.(16) | 图像空间 | 无 | 无 | L2-norm 平滑（作用于 4D LUT 元素与编码器输出系数）、单调性 | αs=0.0001、αm=10 | https://arxiv.org/pdf/2209.01749 |
| NILUT / CNILUT (AAAI 2024) | L1，Eq.(6) `L = Σ‖Φ(x_i) − φ(x_i)‖₁` | **RGB 函数值空间**（在颜色集合 X 上比 LUT 函数值） | 无（Lab ΔE 仅评测，`utils.deltae_dist` 为 Lab 欧氏距离） | 无 | 无 | 单项，无权重；CNILUT 每步累加 3/5 个 condition 的 L1 项 | `fit.py:77-78`（注释 `# more stable than L2`）；`utils.py:86-96`｜https://raw.githubusercontent.com/mv-lab/nilut/main/fit.py |
| Neural Preset (CVPR 2023) | Lrec Eq.(7) = ‖Y_i−I_i‖₁ + ‖Y_j−I_j‖₁ | sRGB 图像空间 | 无 | 无 | Lcon Eq.(5) = ‖Z_i − Z_j‖₂（归一化色彩风格空间一致性） | `L = Lrec + λ·Lcon`，λ=10 | https://arxiv.org/pdf/2303.13511v2（官方仓库仅 `src/metric/*`，无训练代码） |
| CLIP-LUT (arXiv:2311.03943) | L_MSE | 图像空间 | L_perceptual（CLIP 图文余弦过 softmax）、L_SSIM | 无 | 无 | `L_total = L_MSE + 0.4*L_perceptual + 0.4*L_SSIM`；正文写明 0.4 是为把量纲拉平 | https://arxiv.org/pdf/2311.03943 |
| AceTone (arXiv:2604.00530) | ① tokenizer：voxel-wise MSE `L_rec=‖L−L̂‖₂²`（**LUT 参数空间**）② VLM：自回归 CE Eq.(3) | ①LUT 体素 ②token 序列 | 训练阶段刻意不用对抗/感知 loss，对齐推迟到 RL | 无 | L_commit=‖e_L − sg(ê_L)‖₂² | β=0.25 常数；官方代码另有外层 `vq_weight=1e-2`；GRPO reward = 1/(max(2,ΔE)−1) + DeQA 美学分（核查修正） | https://arxiv.org/pdf/2604.00530；`train_vq.py`｜https://raw.githubusercontent.com/martian422/AceTone/open-source-ready/train_vq.py |
| StatLUT (arXiv:2607.08227) | L_LUT = SmoothL1(LUT_pred, LUT_gt)（参数空间）+ L_img = ‖I_pred−I_gt‖₁ + L_SSIM（图像空间） | **两处同时监督** | SSIM（含在 L_img 内） | 无 | L_mono、L_tv | λ_lut=1.0、λ_img=0.5、λ_mono=5.0、λ_tv=0.0001；文本分支 λ_L=1.0、λ_ab=1.2、λ_M=1.5，权重掩码 W = 1 + α·H̄_ab^gt | https://arxiv.org/pdf/2607.08227 |
| FlowLUT (arXiv:2509.23608) | L_MSE 逐像素 | 图像空间 | LPIPS | 无 | 无 | `L_total = L_MSE + 0.1·L_LPIPS` | https://arxiv.org/pdf/2509.23608 |
| PPR10K (CVPR 2021) | L_HC = ‖W_I⊙Î − W_I⊙Y‖₂²（人像区 w=5、背景 w=1） | sRGB 图像空间 | 无 | 无 | L_GLC = ‖Î_CO1 − Î_CO2‖₂²（两次 crop+扰动一致性） | `L = L_HC + λ·L_GLC`，λ=1 | https://arxiv.org/pdf/2105.09180 |
| DualBLN | MSELoss（PPR10K 场景 mask>0 处权重 5） | sRGB 图像空间 | 无 | 无 | weights_norm、tv_cons、mn_cons（5 个 LUT 相加） | 1e-4 / 10.0，与 3DLUT 逐字同构 | `code/train_lut_bilinear_pooling_effres.py:38-40,59,205-215,225`｜https://raw.githubusercontent.com/120326/DualBLN/master/code/train_lut_bilinear_pooling_effres.py |
| TSFlow (arXiv:2207.05430) | L_Retouch Eq.(11) = Σ_{t=1..3}‖Ŷ(t) − Y‖₁ | 图像空间 | 无 | 无 | L_NLL Eq.(10)（对一维 style 向量 s 的归一化流最大似然） | `L = L_Retouch + λ·L_NLL`，λ=1 | https://arxiv.org/pdf/2207.05430v2 |
| NamedCurves (ECCV 2024) | α‖y−ŷ_b‖₂ + ‖y−ŷ‖₂ | 图像空间 | (1 − SSIM(y, ŷ)) | 无 | 无 | α=0.5（附录表 5 消融 α∈{0,0.5,1}）；checkpoint 用验证集 ΔE00 而非 val loss | https://arxiv.org/pdf/2407.09892 |
| CURL (ICPR 2020) | L_rgb = ω_rgb‖Î−I‖₁；L_lab = ω_lab‖Lab(Î)−Lab(I)‖₁ | RGB + Lab + 锥形 HSV 图像空间 | MS-SSIM（在 L 通道） | ① RGB 余弦项 ② L_hsv = ω_hsv(‖ŜV̂cosĤ − SVcosH‖₁ + ‖ŜV̂sinĤ − SVsinH‖₁)，幅值 S·V ∈[0,1] | L_reg（曲线段斜率二阶差） | 官方代码写死：`(rgb + cos_rgb + l1 + hsv + 10*ssim + 1e-6*grad_reg)/6`；论文报 25.45→27.09 dB（加 RGB 余弦项） | `model.py` CURLLoss.forward｜https://raw.githubusercontent.com/sjmoran/curl-image-enhancement/master/model.py |
| StarEnhancer (ICCV 2021) | L_E = ‖Lab(I_b) − Lab(F(...))‖₁ | **CIELab 图像空间** | 无 | 无 | 风格分类器 L_S（L2 归一化 + 缩放 s 的 softmax CE） | 未给出跨项数值 | https://ar5iv.labs.arxiv.org/html/2107.12898 |
| Deep Preset (WACV 2021) | L_MSE | 图像空间 | LPIPS | 无 | L_p = ‖P−P̂‖₁（69 维 preset 参数回归）、L_pp（同 preset 不同图的 latent L1） | `α,β,γ,η = 1, 0.5, 0.01, 1` | https://ar5iv.labs.arxiv.org/html/2007.10701 |
| GLARE (ECCV 2024) | Stage I L1；Stage III L1 | 图像空间 | perceptual、SSIM、semantic | 无 | L_code = ‖sg(z_nl)−z_q‖² + β‖sg(z_q)−z_nl‖² | λ_adv=0.0005、λ_code=1、λ_per=0.01、λ_ssim=0.2、λ_sem=0.1；Stage III `L1 + 0.2·SSIM + 0.01·perceptual` | https://arxiv.org/html/2407.12431v1 |
| InstantRetouch (arXiv:2602.17044) | 纯 L1 `L_recon = E‖D_φ(x, E_θ(x,y)) − y‖₁` | 图像空间（解码器是逐像素 MLP ℝ³→ℝ³） | 无 | 无 | 无 | 单项；文中记 KL/GAN/LPIPS 三种辅助项分别得 PSNR 20.84 / 30.64 / 31.66 | https://arxiv.org/html/2602.17044v1 |
| RAG 颜色复原 (arXiv:2608.08211) | L_L1 | Lab 的 ab 通道（预测残差 Δâb） | perceptual、SSIM | 无 | L_hist（256-bin 高斯直方图 CDF 的 MSE 近似 EMD），`L_hist = ½(L_hist^GT + L_hist^ref)` | λ_hist=2.0、λ_L1=0.5、λ_SSIM=0.1、λ_perc=0.02 | https://arxiv.org/html/2608.08211v1 |
| DiffRetouch (ACM MM 2024) | L_rec Eq.(4) = ‖ε_pred,t − ε‖² + β‖D_t − X_0‖²（latent + pixel 两处） | latent + 图像空间 | 无 | 无 | L_cl（InfoNCE 形式，作用在 colorfulness/contrast/color temperature/brightness 四个标量属性分数上） | λ=1、β=0.01、τ=0.1；Adam lr 1e-6 | https://arxiv.org/html/2407.03757v1 |
| MRStyle / TRStyle (arXiv:2409.05250) | L_teach = MSE(Y, I_g) | 图像空间 | 无 | 无 | 无（IRStyle 侧另有 L_content / L_style / L_hist） | 只训 mapper，其余冻结；数据构造阶段把一对多变成一对一 | https://arxiv.org/html/2409.05250v1 |
| VeraRetouch (arXiv:2604.27375v2) | L1（图像） | 图像空间 | 无 | 无 | CE（token ids） | `L_total = α·L_CE^text + L_1^img`，α 数值正文未给 | https://arxiv.org/html/2604.27375v2 |
| InstructIR (ECCV 2024) | L1 | 图像空间 | 无 | 无 | L_ce（7 类退化意图分类） | `L = L1 + L_ce`（系数均为 1） | https://arxiv.org/html/2401.16468v2；`models/instructir.py` |

按上表的横向事实：

1. **监督位置**。在函数值/参数空间直接监督的只有三家：NILUT（RGB 函数值 L1，与本项目 B×Q 采样口径同构）、AceTone（voxel MSE）、StatLUT（SmoothL1）。StatLUT 是唯一两边同时挂的（λ_lut=1.0 + λ_img=0.5）。其余全部只在图像空间监督。
2. **重建项形态**。图像空间 L2/MSE：3DLUT、AdaInt、SepLUT、4D LUT、DualBLN、PPR10K、FlowLUT、NamedCurves。图像空间 L1：CLUT-Net、Neural Preset、TSFlow、StatLUT 的 L_img、InstantRetouch、InstructIR、VeraRetouch。函数值 L1：NILUT。SmoothL1：StatLUT。CIELab 空间的重建项只有 StarEnhancer 与 CURL 的 L_lab 分项（CURL 代码在算 Lab-L1 之前把 Lab 值 clamp 到 [0,1]）。
3. **ΔE 的位置**。本轮打开的全部来源中，ΔE00/ΔE76 只出现在两处：(a) 评测指标（3DLUT 的 4E*、AdaInt/SepLUT 的 ΔE_ab、NILUT 的 deltae_dist、CLIP-LUT、NamedCurves 的 ΔE00、FlowLUT 的 CIEDE、AceTone 的 CIEDE2000）；(b) RL reward（AceTone，1/(max(2,ΔE)−1)）。**本轮没有找到把完整 ΔE00 当可微训练 loss 的已发表工作或公开实现**（限定：不代表不存在，见 §5）。
4. **可微感知项**只见到 LPIPS（FlowLUT 0.1；Deep Preset 0.5；GLARE 0.01）与 SSIM（CLIP-LUT 0.4；NamedCurves 系数 1；StatLUT 含在 λ_img 内；GLARE 0.2）。
5. **方向/色相类项**只有三处：CLUT-Net 的通道余弦（权重 1，无 chroma 加权、无归一化）、CURL 的锥形 HSV L1（幅值 S·V ∈ [0,1]）、CURL 的 RGB 余弦项。本项目 L_hc 的 C 是未归一化 CIELab chroma（mean C=32.5）且外挂 10×。
6. **正则权重的稳定值**：TV/smooth = 1e-4，monotonicity = 10，融合权重 L2（sparse）= 1e-4。发布代码里的实际开关：3DLUT 三项全开；DualBLN 三项全开；AdaInt sparse 开 / TV 关 / mono 开；SepLUT sparse 开 / TV 关 / mono 关；CLUT-Net 默认全关；StatLUT mono=5.0、tv=1e-4。本项目 R_sparse（opacity 稀疏，0.001）在这一族里没有直接对应物；最接近的是对 LUT 融合权重的 mean(w²)（1e-4）。

### 3.2 量纲与自动平衡

#### 3.2.1 色相项的原始出处与量纲

- Sharma, Wu, Dalal (2005)《The CIEDE2000 Color-Difference Formula: Implementation Notes...》，Color Research and Application 30(1):21-30：CIE1976/CIE1994 也可用度量色相差 ΔH 表达，但只依赖 ΔH² 而与 ΔH 符号无关，所以没有 CIEDE2000 的不连续。原文 Eq.(11) 为 `ΔH = 2√(C'₁C'₂)·sin(Δh'/2)`，平方后 `ΔH² = 4·C'₁C'₂·sin²(Δh'/2) = 2·C'₁C'₂·(1 − cos Δh')` —— 即「chroma 乘 (1−cos Δhue)」形式的原始出处。原公式中该项随后被除以 `(k_H·S_H)²`，其中 Eq.(20) `S_H = 1 + 0.015·C̄'·T`。
  https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf
- 对照：本项目 L_hc = mean(C·(1−cos Δhue)) 是**一次幂**、分子侧放未归一化 CIELab chroma、**没有 S_H 分母**；CIEDE2000 在分子放 C'₁C'₂ 的同时在分母放随 C̄' 线性增长的 S_H。CURL 的对应项幅值为 S·V ∈ [0,1]（有界）。
  https://raw.githubusercontent.com/sjmoran/curl-image-enhancement/master/model.py

#### 3.2.2 ΔE00 直接作为 loss 的不连续性（原文陈述）

Sharma 2005 原文：`the discontinuities do preclude the use of the formula in analysis based on Taylor series approximations and in design techniques using gradient based optimization, that not only require continuity of the function but also continuity of the first derivative.` 量级：两色相距 5 个 CIELAB ΔE*ab 单位以内时，ΔE00 的不连续跳变小于 0.2734；相距 1 个单位时小于 0.0119。主要不连续来自平均色相 h̄' 的 180° 跳变，经 T 与 Δθ 传入 S_H。
https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf

官方 MATLAB 参考实现 `deltaE2000.m` 中可见的不可微点：`find(Cpprod == 0)` 的零色度分支硬置 `dhp=0`；`dhp - 2*pi*(dhp > pi)` 与 `+ 2*pi*(dhp < -pi)` 的布尔回绕；`hp - (abs(hpstd-hpsample) > pi)*pi` 的布尔 180° 修正；最外层 `sqrt`。
https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/deltaE2000.m

#### 3.2.3 自动平衡的三条已核实路径

| 方法 | 公式 | 归一化性质 | 超参 / 数值 | 来源 |
|---|---|---|---|---|
| GradNorm (ICML 2018) | `G_W^(i)(t) = ‖∇_W w_i(t)L_i(t)‖₂`；`L̃_i = L_i(t)/L_i(0)`；`r_i = L̃_i / E_task[L̃_i]`；目标 `G_W^(i) ↦ Ḡ_W(t)×[r_i]^α`；权重更新用独立 L1：`L_grad = Σ_i |G_W^(i) − Ḡ_W×[r_i]^α|₁`，目标项 detach，只对 w_i 求导 | 每步后重归一化使 `Σ_i w_i = T`，与全局 lr 解耦 | α=1.5（NYUv2）/ 0.12（玩具）；w 的更新 lr 0.025，Adam | http://proceedings.mlr.press/v80/chen18a/chen18a.pdf |
| Uncertainty weighting (CVPR 2018) | `L = (1/2σ₁²)L₁ + (1/2σ₂²)L₂ + log σ₁σ₂`；实现回归 `s := log σ²` 避免除零 | 不归一化；原文报告 CityScapes 三任务末期有效权重比 43 : 0.16 : 1（初始 500 步 23 : 0.22 : 1），并用 power-law 退火 lr 抵消权重整体上升 | log σ² 初值 −2.0 至 5.0 均约 100 步内收敛到同一解 | https://ar5iv.labs.arxiv.org/html/1705.07115；可运行实现 `mtan/im2im_pred/utils.py`（`loss = Σ 1/(2*exp(logsigma[i]))*L_i + logsigma[i]/2`，logsigma 初值 [-0.5,-0.5,-0.5]） |
| DWA（MTAN 官方实现） | `w_k = avg_cost[t−1,k]/avg_cost[t−2,k]`；`λ_k = K·exp(w_k/T)/Σ_j exp(w_j/T)`，K=3 | Σλ = K | 默认 T=2.0；前两个 epoch 权重全为 1.0 | https://raw.githubusercontent.com/lorenmt/mtan/master/im2im_pred/utils.py |
| CoV-Weighting (WACV 2021) | `ℓ_t = L_t / μ_{L_{t−1}}`（分母为该项到 t−1 的观测均值，Welford 在线更新）；`α_it = (1/z_t)·σ_{ℓ_it}/μ_{ℓ_it}`，`z_t = Σ_i c_{ℓ_it}` | `Σ_i α_it = 1`，原文写明「以与学习率解耦」 | 官方代码：`L0 = running_mean_L; l = L/L0; ls = running_std_l/running_mean_l; alphas = ls/torch.sum(ls)`；KITTI 32 项 loss 的 ARD：等权 0.0923、手调 0.0944、CoV 0.0912、GradNorm 0.1152、Uncertainty 1.0712、Multi-objective 6.3400 | https://ar5iv.labs.arxiv.org/html/2009.01717；`rickgroen/cov-weighting · losses/covweighting_loss.py` |

与本项目的形式对应：本项目已实证有效的 `c = c / c.detach().mean()` 在形式上是 CoV 的 loss-ratio 那一步（除以该量的均值），差别在于作用对象——本项目除的是 **batch 内 chroma 权重张量的均值（在 loss 内部）**，CoV 除的是 **该 loss 标量的历史 running mean（在 loss 外部）**。三种自动平衡方法改的都是各项的相对系数（非均匀缩放），与本项目已实证「梯度裁剪对 Adam 无效（均匀缩放免疫）」的作用面不同。

CLIP-LUT 是本轮打开的 LUT 族工作里唯一把「量纲对齐」写进论文正文的：`To balance the composition of Ltotal, we set the weight of Lperceptual to 0.4, as it is too large compared to LMSE and LSSIM.`（https://arxiv.org/pdf/2311.03943）StatLUT 的做法是四项各给一个经验常数（1.0 / 0.5 / 5.0 / 0.0001）；3DLUT 系是把权重当超参在两个离散集合上扫。

#### 3.2.4 外部工作里多项 loss 的权重比量级（供与 325:1 对照）

DETR bbox 5 / giou 2 / eos 0.1；Mask2Former COCO panoptic mask 5.0 / dice 5.0 / class 2.0 / no-object 0.1（默认 config MASK_WEIGHT 20.0 : NO_OBJECT_WEIGHT 0.1）；LISA txt 1.0 / bce 2.0 / dice 0.5；pix2pix `L_cGAN + 100·L_L1`；SRGAN `l_X + 1e-3·l_Gen`；MGIE `L_ins + 0.5·L_edit`；InstructIR `L1 + L_ce`（系数均 1）；GS-LRM λ_perceptual=0.5；Gamba λ_mask=1、λ_LPIPS=0.5、λ_rdist 0.1→0；TGS λ_m=1、λ_s=1、λ_l=2、λ_c=10、λ_e=0；Splatter Image 用凸组合 `lambda_l12 = 1 − lambda_lpips`（两项和恒为 1）。

### 3.3 一对多与均值坍塌

#### 3.3.1 现象出处（逐字）

- Zhang, Isola, Efros《Colorful Image Colorization》（ECCV 2016）：`If an object can take on a set of distinct ab values, the optimal solution to the Euclidean loss will be the mean of the set. In color prediction, this averaging effect favors grayish, desaturated results. Additionally, if the set of plausible colorizations is non-convex, the solution will in fact be out of the set, giving implausible results.` https://ar5iv.labs.arxiv.org/html/1603.08511
- pix2pix（Isola et al.）：`L1 will be minimized by choosing the median of the conditional probability density function over possible colors.` https://ar5iv.labs.arxiv.org/html/1611.07004
- SRGAN（Ledig et al.）：`minimizing MSE encourages finding pixel-wise averages of plausible solutions`。https://arxiv.org/pdf/1609.04802v5
- Bishop《Mixture Density Networks》（NCRG/94/004, 1994）：在无限数据极限下 sum-of-squares error 与 cross-entropy error 的最优解都是 `f_k(x,w*) = ⟨t_k | x⟩`（条件平均）；摘要点名多值映射：`the average of several correct target values is not necessarily itself a correct value`。https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
- InDI（TMLR 2023）逐字使用「regression to the mean」这一命名：`In the case p=2, the minimum mean-squared error (MMSE) optimal solution is the conditional expectation: x_MMSE(y)=E[x|y] ... A similar statement is true for other p≠2 in which case the mean is replaced by another aggregation operator (e.g. median for L1).` https://ar5iv.labs.arxiv.org/html/2303.11435
- Deep3DBox（MultiBin）：`It is known that using the L2 loss is not a good fit for many complex multi-modal regression problems. The L2 loss encourages the network to minimize to average loss across all modes` https://arxiv.org/pdf/1612.00496v2
- DiffRetouch（ACM MM 2024，image retouching 场景）：`current retouching methods mostly adopt deterministic models, which not only neglects the style diversity in the expert-retouched results and tends to learn an average style during training` https://arxiv.org/html/2407.03757v1
- OpenVLA-OFT（自述）：`L1 regression may help smoothen out noise in training demonstrations by encouraging the policy to learn the median mode` https://arxiv.org/html/2502.19645v1

#### 3.3.2 解法族（含数值）

**(a) 分类离散化**

| 做法 | 具体写法 | 数值 | 来源 |
|---|---|---|---|
| ab 空间量化 + 多项 CE + 类再平衡 + annealed-mean | grid size 10，保留 in-gamut 的 Q=313 个 bin；GT 用近邻高斯核 soft-encode；`w ∝ ((1−λ)p̃ + λ/Q)^{-1}`，E[w]=1；解码 `H(Z)=E[f_T(Z)]` | 论文写 5 近邻、λ=1/2、σ=5、T=0.38；官方 caffe 层写 `self.NN = 10`、`self.sigma = 5.`、`gamma=.5`、`alpha=1.`；再平衡在 **backward 里乘梯度**（forward 恒等直通，源码注释：`this was bad, would mess up the gradients going up`） | https://arxiv.org/pdf/1603.08511v5；https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/caffe_traininglayers.py |
| 标量 → 直方图 CE（Two-Hot / HL-Gauss） | Two-Hot：`p_i=(目标−z_i)/(z_{i+1}−z_i)`；HL-Gauss：`p_i = F_Y(z_i+ς/2) − F_Y(z_i−ς/2)`，`Y ∼ N(μ=目标, σ²)`，ς=(v_max−v_min)/m | 推荐调 σ/ς；`Unless specified otherwise, we set σ/ς=0.75`（质量分到约 6 个位置） | https://arxiv.org/html/2403.03950v1 |
| 角度 bin + bin 内残差（MultiBin） | 每 bin 输出 (置信度 c_i, cos Δθ_i, sin Δθ_i)，共 3n 个输出；`Lθ = Lconf + w×Lloc`，`Lloc = −(1/n_θ*)Σ cos(θ*−c_i−Δθ_i)`；总 `L = α×Ldims + Lθ` | bin 数消融（1 bin 等价 L2）：KITTI OS 0.89(1)→0.98(2)→0.97(4)→0.97(8)→0.96(16)；Pascal3D+ Acc π/6 0.65(1)→0.72(2)→0.78(4)→0.81(8)→0.77(16)。w 与 α 数值论文未给 | https://arxiv.org/pdf/1612.00496v2 |
| 动作维度离散成 256 bin + 覆写词表 token | bin 边界取训练集 1%/99% 分位数；覆写 Llama tokenizer 最低频 256 个 token；只在 action token 上算 CE | 256 bin | https://arxiv.org/html/2406.09246v3；https://ar5iv.labs.arxiv.org/html/2307.15818 |
| 位置头做成分类式期望 | [−0.5,0.5] 每轴离散成 21 个格点 c_j，linear 出 21 维 logits，softmax 后取加权期望 `y_i = Σ P(Q_ij)·c_j` | 21 格点 | https://arxiv.org/html/2403.18795v3 |

注：Bishop 的推导给出「sum-of-squares 与 cross-entropy 两种 error 的最优解都等于 ⟨t_k|x⟩」；Colorful Colorization 亦独立写到 `taking the mean after performing classification suffers from some of the same issues as optimizing for a Euclidean loss`，其对策是 annealed-mean 而非取均值。

**(b) 判别式**

| 工作 | 目标函数 | 权重 | 判别器输入 | 来源 |
|---|---|---|---|---|
| pix2pix | `G* = arg min_G max_D L_cGAN(G,D) + λ L_L1(G)` | 官方实现 `--lambda_L1 default=100.0`（`loss_G_L1 = criterionL1(fake_B, real_B) * lambda_L1`） | PatchGAN（70×70）；D 额外 concat 条件输入 `fake_AB = cat((real_A, fake_B),1)`；1×1 PixelGAN 变体对空间锐度无影响、增加 colorfulness。消融 FCN-score（Cityscapes labels→photo）：L1 0.42/0.15/0.11，L1+cGAN 0.66/0.23/0.17，L1+GAN 0.64/0.20/0.15。原文另记去掉条件后 `generator collapsed into producing nearly the exact same output regardless of input photograph` | https://ar5iv.labs.arxiv.org/html/1611.07004；`models/pix2pix_model.py` |
| SRGAN | `l^SR = l^SR_X + 10^{-3} l^SR_Gen` | 内容项 1、对抗项 1e-3 | 图像判别器；实际最小化 −log D | https://arxiv.org/pdf/1609.04802v5 |

**(c) 对比**

| 工作 | 目标函数 | 正/负样本构造 | 权重 | 来源 |
|---|---|---|---|---|
| CUT PatchNCE | `L_PatchNCE = E Σ_l Σ_s ℓ(ẑ_l^s, z_l^s, z_l^{S∖s})`；总 `L_GAN + λ_X L_PatchNCE(X) + λ_Y L_PatchNCE(Y)` | 负样本取自同一张输入图内的其他 patch；(N+1) 路 CE | τ=0.07，num_patches=256，nce_layers='0,4,8,12,16'；CUT 模式 λ_NCE=1.0、FastCUT 模式 10.0；不让 PatchNCE 梯度回传到 decoder 时 FID 444.2 | https://ar5iv.labs.arxiv.org/html/2007.15651；`models/patchnce.py` |
| DiffRetouch L_cl | `L_cl = Σ_{i|c_i≠0} −log[ e^{−|s_i−s_i^+|/τ} / (e^{−|s_i−s_i^+|/τ} + e^{−|s_i−s_i^-|/τ}) ]` | 作用在四个标量属性分数（colorfulness / contrast / color temperature / brightness）上；正样本 = 同条件换噪声 ε′，负样本 = 条件取反 c⁻ = −c | λ=1、β=0.01、τ=0.1；未加 L_cl 时原文记 `adjusting the coefficients related to contrast and color temperature has little effect on the final result` | https://arxiv.org/html/2407.03757v1 |
| Deep Preset PPL | `L_pp = (1/N)Σ‖F_{Z'} − F_Z‖₁`，Z' 是同一 preset 修过的另一张随机图 | 同标签一致性 | 权重 1（与 L_MSE 的 1、LPIPS 的 0.5、preset 回归的 0.01 并列）。消融：不带 PPL 在「preset 预测再回放」路 H-Corr 0.6360 / PSNR 21.86 / LPIPS 0.1120；带 PPL 0.6231 / 21.48 / 0.1197；生成器直接出图路，带 PPL 0.6749 / 22.33 / 0.1011，不带 0.6687 / 22.17 / 0.1041 | https://ar5iv.labs.arxiv.org/html/2007.10701 |

**(d) MDN**

Bishop 1994：`p(t|x) = Σ_{i=1}^{m} α_i(x) φ_i(t|x)`，φ_i 为球形高斯；α_i 由 softmax 输出、σ_i = exp(z^σ_i)、μ_ik = z^μ_ik 直接输出；网络输出总数 (c+2)·m（常规网络为 c）；损失 `E_q = −ln{Σ_i α_i(x^q) φ_i(t^q|x^q)}`；报告写明该式与 Jacobs et al. 的 competing local experts 形式等价；优化用 BFGS。
https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf

**(e) 参数空间生成**

| 工作 | 目标函数 | 数值 | 来源 |
|---|---|---|---|
| Flow Matching / CFM (ICLR 2023) | `L_CFM = E_{t,q(x1),p_t(x|x1)}‖v_t(x) − u_t(x|x1)‖²`（Eq.9）；定理 2 `∇_θ L_FM = ∇_θ L_CFM`。OT 路径 Eq.(20-22)：`μ_t(x)=t·x1`，`σ_t(x)=1−(1−σ_min)t`，`u_t(x|x1) = (x1 − (1−σ_min)x)/(1 − (1−σ_min)t)`，`ψ_t(x) = (1−(1−σ_min)t)x + t·x1`；代入后 `L_CFM = E‖v_t(ψ_t(x0)) − (x1 − (1−σ_min)x0)‖²` | ImageNet 64×64：FM w/ OT NLL 3.31 / FID 14.45 / NFE 138；DDPM 基线 3.32 / 17.36 / 264（同架构同超参） | https://arxiv.org/pdf/2210.02747v2 |
| TSFlow | `L_NLL = −log p_z(F(s;X,θ_F)) − log|det J_F(s)|`（Eq.10）；`L = L_Retouch + λ·L_NLL`，推理 `s = F^{-1}(z;X)`，z~N(0,I) | λ=1，n=3 | https://arxiv.org/pdf/2207.05430v2 |
| InDI | `x_t=(1−t)x+t·y`；`min_θ E‖F_θ(x_t,t) − x‖_p`（Eq.4）；带噪版 Eq.9；推理 `x̂_{t−δ} = (δ/t)F_θ(x̂_t,t) + (1−δ/t)x̂_t` | 玩具例：迭代回归收敛到四个模式之一，regression to the mean 始终是所有模式的加权平均 | https://ar5iv.labs.arxiv.org/html/2303.11435 |
| DiffRetouch | latent 扩散，**变换参数**（16×16×8×12 的 affine bilateral grid）是 U-Net 每步的额外输出，切片后仿射直接作用在输入图像上 | β=0.01（pixel 项） | https://arxiv.org/html/2407.03757v1 |

**(f) 离散 token + 自回归 CE**

AceTone：3D-conv VQ-VAE 把 3×32³ 连续 LUT 压成 4×4×4=64 个离散 token，K=256、embedding_dim=64、EMA 更新、β=0.25；tokenizer 目标 `L = L_rec(voxel MSE) + β·L_commit`；官方 `train_vq.py` 的实际写法是 `loss = F.mse_loss(x_rec, x) + vq_weight * loss_vq`，`vq_weight` 默认 1e-2，AdamW lr=3e-4 betas=(0.9,0.95) wd=1e-4 grad clip 1.0，500 epoch。第二阶段把 256 个 LUT token 加进 Qwen2.5-VL-3B 词表（README：`Qwen models have about 300 unused token slots, so for an extended vocabulary size of 256, you can directly modify the tokenizer configuration without editing the LM itself.`），用 next-token CE 训练；第三阶段 GRPO。LUT 库：约 10,000 商用 `.cube` + PPR-10K 导出约 34,000，PCA+K-means 聚成 8,192 个。tokenizer 在 held-out 1024 个 LUT 上 PSNR 37.5 dB、ΔE=1.38。
https://arxiv.org/html/2604.00530v1；https://raw.githubusercontent.com/martian422/AceTone/open-source-ready/train_vq.py

VQ-VAE 原始出处：`L = log p(x|zq(x)) + ‖sg[ze(x)] − e‖₂² + β‖ze(x) − sg[e]‖₂²`（Eq.3）；`We found the resulting algorithm to be quite robust to β, as the results did not vary for values of β ranging from 0.1 to 2.0. We use β = 0.25 in all our experiments`；附录 EMA 字典更新 γ=0.99。https://arxiv.org/pdf/1711.00937v2

CodeFormer 的 code-level 两项：`l_feat_encoder = mean((quant_feat_gt.detach()−lq_feat)**2) * feat_loss_weight`（feat_loss_weight=1.0）与 `cross_entropy_loss = F.cross_entropy(logits.permute(0,2,1), idx_gt) * entropy_loss_weight`（entropy_loss_weight=0.5）；Stage I codebook 1024 项、维度 256，image-level L1 1.0 + LPIPS 1.0 + hinge GAN 1.0。
https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/models/codeformer_idx_model.py

**(g) 数据构造侧消除一对多**

MRStyle / TRStyle：先用 ChatGPT 生成风格文本 T_s → Stable Diffusion 生成风格图 I_s → 已训好的 IRStyle 由 (I_c, I_s) 出 I_g 作为 GT，构成三元组 (I_c, T_s, I_g)；损失就是 Y 与 I_g 的 MSE（L_teach），只训 mapper 其余冻结。https://arxiv.org/html/2409.05250v1

### 3.4 检索式与 classify-then-refine

| 工作 | 检索/分类键 | 聚合方式 | 下游 | 权重与数值 | 来源 |
|---|---|---|---|---|---|
| InstantRetouch | 冻结 SigLIP-v2 内容 embedding 的余弦相似度 | top-K 后 softmax 加权平均风格 latent：`w_i = exp(s_i/τ)/Σexp(s_j/τ)`，`z_q = Σ w_i z_i` | 逐像素条件 MLP ℝ³→ℝ³（3→128→[256,512,3]，ReLU，末层 sigmoid），2048 维风格 latent 投影后逐层**相加**注入 | τ=0.1、K=3；纯 L1 训练；条件注入对比中「简单相加」被记为高于 adaLN 与 cross-attention；「用全部参考」被记为差于「用检索到的少量相关参考」 | https://arxiv.org/html/2602.17044v1 |
| RAG 颜色复原 (2608.08211) | VGG19 relu3_4 特征的全局均值/方差各建一个 L2-归一化 IndexFlatIP（FAISS 双索引，α=β=0.5，Top-k=1），知识库 800 张 DIV2K | 取检索图 Lab 的 ab 通道算 spatial-preserving 直方图，GAP 成 512 维颜色向量，经 AdaIN 注入 | 只预测颜色残差 `âb = clamp(ab + Δâb, −1, 1)`，loss 作用在加回后的结果上 | λ_hist=2.0 / λ_L1=0.5 / λ_SSIM=0.1 / λ_perc=0.02；`L_hist = ½(L_hist^GT + L_hist^ref)`；残差 vs 直接预测：LOLv1 直接预测数值略高，LOLv2-Real 直接预测 19.68 dB（低 1.69 dB）、LOLv2-Synthetic 19.77 dB | https://arxiv.org/html/2608.08211v1 |
| StarEnhancer | 风格分类器（L2 归一化 embedding 与无 bias 全连接权重、带缩放 s 的 softmax CE） | 类原型：该风格 n 张图的 L2 归一化 embedding 求平均再归一化 | mapping network → L 组 (μ,σ) → Dual AdaIN 注入曲线编码器（15 组曲线控制点，输入通道 {r,g,b,x,y} × 输出通道 {r,g,b}）；L_E 在 CIELab 空间 L1 | 用训练集子集生成更多风格 embedding 以避免只用类中心 | https://ar5iv.labs.arxiv.org/html/2107.12898 |
| GLARE | VQGAN codebook（1000 个码、每码维度 3） | 不用 token 分类，而用条件可逆归一化流（I-LNF）把 NL 特征分布映到简单分布，NLL 训练；推理反向采样后最近邻查表 | AFT 融合模块 | 消融把 I-LNF 换成「Transformer 直接预测 code index」作为对照项；去掉 I-LNF 后 PSNR −1.16 dB、SSIM −0.017；去掉 codebook 后平均 PSNR −2.0 dB、SSIM −0.024 | https://arxiv.org/html/2407.12431v1 |
| Image-Adaptive 3D LUT | CNN classifier 预测 3 个标量权重 | 对 3 个基 LUT 线性组合（LUT0 恒等初始化 `Generator3DLUT_identity()`，LUT1/LUT2 零初始化 `Generator3DLUT_zero()`） | 直接查表 | MSE + 1e-4·(weights_norm + tv) + 10·mn | https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/image_adaptive_lut_train_paired.py |
| AceTone | 无检索；LUT 直接 tokenize 后自回归生成 | — | — | 见 §3.3(f) | https://arxiv.org/html/2604.00530v1 |
| PromptIR | 图像特征 GAP 后 Linear + softmax | 在可学基元库上加权：`prompt_param` 形状 (1, prompt_len=5, prompt_dim, prompt_size, prompt_size)，加权求和后 bilinear 插值到 (H,W) 再过 3×3 conv，与 decoder 特征 concat | — | prompt_len=5 | https://raw.githubusercontent.com/va1shn9v/PromptIR/main/net/model.py |

与本项目的对应位置：B3 用的 (major,minor) 类别桶（ΔE00 6.155 / 6.064）在形式上就是 StarEnhancer 的「风格类标签」；vrmeta 里已有该标签。检索基底 + 残差的形态出自 2608.08211，其残差加在前端输出的 ab 通道上、分布级直方图项权重 4× 于逐点 L1 项。

### 3.5 候选 loss 方案表（并列陈述，不排序、不推荐）

「冻结口径」指本项目现有约束：VLM 冻结、`<seg_color>` 单向量读出、函数值空间 B=32×Q=256 监督、normal-only、评测 headline 用 ΔE00 与五条基线。

#### A 组：量纲 / 权重（7 个）

| # | 公式 | 需要改哪里 | 来源依据 | 与冻结口径的冲突点 |
|---|---|---|---|---|
| A1 | `L_hc = mean( (c/c.detach().mean()) · (1 − cos Δhue) )` | loss 内部：chroma 权重张量除以 batch 均值 | 本项目已实证的唯一有效档；形式对应 CoV-Weighting 的 loss-ratio 步（https://ar5iv.labs.arxiv.org/html/2009.01717） | 无（不动结构、不动读出口径）。CoV 的原始作用对象是 loss 标量的历史均值而非张量内权重，二者叠加行为无外部文献覆盖 |
| A2 | `L_hc = mean( 2·C'₁C'₂·(1 − cos Δh') / (k_H·S_H)² )`，`S_H = 1 + 0.015·C̄'·T` | 给色相项加 S_H 分母；由一次幂改为 CIEDE2000 的 ΔH'² 形态 | Sharma 2005 Eq.(11)(20)（https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf） | 需要在 loss 里算 h̄'、T、Δθ；原文写明 h̄' 有 180° 跳变（不连续 ≤0.2734 @ 5 ΔE*ab） |
| A3 | `L_hue = ω(‖ŜV̂cosĤ − SVcosH‖₁ + ‖ŜV̂sinĤ − SVsinH‖₁)`（锥形 HSV，幅值 S·V∈[0,1]） | 把 CIELab chroma 换成 HSV 的 S·V | CURL（https://ar5iv.labs.arxiv.org/html/1911.13175；官方 `model.py` 权重写死为 1，全式 `(rgb + cos_rgb + l1 + hsv + 10*ssim + 1e-6*grad_reg)/6`） | 色彩空间从 Lab 换成 HSV；CURL 代码在 Lab-L1 前把 Lab clamp 到 [0,1] |
| A4 | `L_cos = mean(1 − cosine_similarity(f(x), L_l(x), dim=通道))`，权重 1 | 去掉 chroma 加权，权重从 10 改为 1 | CLUT-Net（`utils/losses.py:24-25`，与 l1 并列权重各 1，`sum(loss_ls).backward()`） | 无 chroma 加权即失去「高饱和处更重」这一性质 |
| A5 | GradNorm：`L_grad = Σ_i |G_W^(i) − Ḡ_W×[r_i]^α|₁`，目标项 detach，每步后 Σw_i = T | 需要在共享层（对应 encoder 输出层）对每项单独 backward 取范数；加一个只更新 w_i 的优化器 | http://proceedings.mlr.press/v80/chen18a/chen18a.pdf（无官方开源实现，论文只写用 TensorFlow v1.2.1） | 每步多次 backward，步数匹配（U4）口径下的等价步数需重定义；需按 Eq.(1)(2) 自己实现并加运行时断言 |
| A6 | Uncertainty weighting：`L = Σ_i 1/(2 exp(s_i))·L_i + s_i/2`，`s_i = log σ_i²` | 加 3 个 nn.Parameter | https://ar5iv.labs.arxiv.org/html/1705.07115；可运行实现 `mtan/im2im_pred/utils.py` | 原文自述末期权重整体上升等价抬高全局 lr，需 power-law 退火 lr 抵消；CoV 论文在 KITTI 单任务多 loss 上报该方法 ARD=1.0712（等权 0.0923） |
| A7 | DWA / CoV：`λ_k = K·exp(w_k/T)/Σ exp(w_j/T)`（T=2.0）；或 `α_i = (σ_{ℓ_i}/μ_{ℓ_i})/Σ_j(·)`，Σα=1 | loss 外部：维护每项 running mean/std，逐步重算系数 | `lorenmt/mtan · im2im_pred/utils.py`；`rickgroen/cov-weighting · losses/covweighting_loss.py` | Σα=1 会同时改变总 loss 尺度，与现有 lr 配置耦合 |

#### B 组：一对多（8 个）

| # | 公式 | 需要改哪里 | 来源依据 | 与冻结口径的冲突点 |
|---|---|---|---|---|
| B1 | HL-Gauss 逐通道标量直方图 CE：`p_i = F_Y(z_i+ς/2) − F_Y(z_i−ς/2)`，`Y∼N(μ=L_l(x)_c, σ²)`，σ/ς=0.75 | f(x) 的每个通道输出从 1 个标量改为 m 维 logits；GLUT 末端渲染需给出 logits 而非颜色值 | https://arxiv.org/html/2403.03950v1 | GLUT 的 f(x) 是由高斯基元解析求值得到的连续量，改成 logits 需要在 GLUT 之后再挂一层离散化头；[v_min,v_max]、ς 需自定 |
| B2 | 3D RGB 分箱多项 CE + 类再平衡 + annealed-mean：`w ∝ ((1−λ)p̃ + λ/Q)^{-1}`，`H(Z)=E[f_T(Z)]` | 输出色空间量化；GT soft-encode；再平衡在 backward 里乘梯度 | https://arxiv.org/pdf/1603.08511v5；官方 caffe 层（NN=10、sigma=5.、gamma=.5、alpha=1.） | 原文只在 2D ab 上量化得 313 bin；3D RGB 的 bin 数、稀疏性、近邻数/核宽在任何已核实来源里都没有给出 |
| B3 | MDN：`E = −ln Σ_i α_i φ_i(t|x)`，α softmax、σ=exp(z^σ) | 输出从 22N+12 变成 m×(22N+12) + m 个 α + m 个 σ | https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf | 输出规模 ×m；推理端如何从混合分布出单个 f 未在原文覆盖 |
| B4 | 参数空间 CFM：`L_CFM = E‖v_t(ψ_t(x0)) − (x1 − (1−σ_min)x0)‖²`，x1 = 1068 维 GLUT 参数向量，条件为 z | 生成器改为向量场网络 + 多步积分推理 | https://arxiv.org/pdf/2210.02747v2 | GLUT 的 N=48 个基元可置换，同一个 f 对应 48! 个参数向量，x1 需要一个规范排列；本轮未找到处理「置换不变参数集合的条件生成」的做法 |
| B5 | VQ tokenizer + 自回归 CE：`L=L_rec+β L_commit`（β=0.25，外层 vq_weight=1e-2）→ `L_gen=−Σ_t log p_θ(z_t|z_<t, I, c)` | 需要先训一个把 GLUT 参数或 f 的 32³ 采样张量量化成 token 的 tokenizer；VLM 需自回归吐 token | https://arxiv.org/html/2604.00530v1；https://raw.githubusercontent.com/martian422/AceTone/open-source-ready/train_vq.py；VQ-VAE https://arxiv.org/pdf/1711.00937v2 | 与「冻结 VLM + 单个 `<seg_color>` hidden 一次性读出」不兼容（AceTone 是扩词表 + 调 MLP connector + 调语言模型）；量化对象若取 32³ 采样则丢掉 GLUT 参数化 |
| B6 | cGAN 项：`min_G max_D L_cGAN + λ L_L1` | 加判别器；D 需吃 (条件, 输出) 对 | pix2pix λ=100（`--lambda_L1 default=100.0`）；SRGAN 1e-3 | 两篇的 D 都吃图像；本项目只有 8192 个 (查询色, 输出色) 对、无空间结构，最接近的先例只有 pix2pix 的 1×1 PixelGAN |
| B7 | 对比项：`L_cl = Σ −log[e^{−|s−s⁺|/τ}/(e^{−|s−s⁺|/τ}+e^{−|s−s⁻|/τ}])`，τ=0.1，λ=1 | 需要定义若干标量属性分数 s（DiffRetouch 用 colorfulness/contrast/color temperature/brightness），并构造 c⁻ 分支 | https://arxiv.org/html/2407.03757v1；CUT PatchNCE τ=0.07、num_patches=256 | 本项目现有三负控制（shuffle/无关词/固定短语）是评测口径；把它写进训练 loss 后，同一构造不能再当作独立负控制 |
| B8 | style-NLL：`L = L_Retouch + λ·L_NLL`，λ=1，推理 `s=F^{-1}(z;X)`，z~N(0,I) | 加一个对低维 style 向量的可逆流 | https://arxiv.org/pdf/2207.05430v2 | 推理需采样，headline ΔE00 的单点评测口径需重新定义（采样几次、取哪一个） |

#### C 组：监督位置 / 结构性（4 个）

| # | 公式 | 需要改哪里 | 来源依据 | 与冻结口径的冲突点 |
|---|---|---|---|---|
| C1 | 纯 L1，去掉色相项：`L = Σ_i‖Φ(x_i) − φ(x_i)‖₁` | 删 L_hc 与其 10× 权重 | NILUT Eq.(6)（与本项目 B×Q 采样口径同构）；`fit.py:77-78` 注释 `# more stable than L2` | 无结构冲突；NILUT 一个模型只拟合一个 LUT，CNILUT 用 3/5 个 one-hot condition 并累加 3/5 个 L1 项，条件维度与注入方式与本项目单个 2560 维 pooled 向量不同 |
| C2 | 函数值 + 图像空间双监督：`L = λ_lut·SmoothL1(参数/函数值) + λ_img·(‖I_pred−I_gt‖₁ + SSIM) + λ_mono + λ_tv` | 需要在训练中渲染图像 | StatLUT λ_lut=1.0、λ_img=0.5、λ_mono=5.0、λ_tv=0.0001（https://arxiv.org/pdf/2607.08227） | 训练侧需引入图像渲染路径（当前只有 8192 色的函数值监督） |
| C3 | 检索基底 + 残差 + 分布项：`f = f_base + Δ`，`L = 0.5·L1 + 2.0·L_hist + 0.1·SSIM + 0.02·L_perc`，`L_hist = ½(L_hist^GT + L_hist^ref)` | 生成器只出残差；基底来自 B3 桶检索 | https://arxiv.org/html/2608.08211v1（残差 vs 直接预测：LOLv2-Real 直接预测低 1.69 dB） | 残差加在 GLUT 参数空间（μ/Cholesky/仿射非线性、不可直接相加）还是函数值空间（f_base(x)+Δ(x)），任一篇打开的文献都没有直接讨论 |
| C4 | 同标签一致性：`L_pp = (1/N)Σ‖F_{Z'} − F_Z‖₁`，权重 1 | 每步额外取「同一指令/同一 LUT 在另一张图上」的样本，拉齐条件向量或预测参数 | Deep Preset（α,β,γ,η = 1, 0.5, 0.01, 1） | 需要按 LUT id 或指令 id 组 batch，与当前 B=32 随机采样的组批方式不同 |

合计 **19 个候选 loss 方案**。

---

## 4. Q2：结构调研

### 4.1 query-based 解码头的出处链

| 工作 | query 从哪来 | 数量 | cross-attention 的 memory | 层内顺序 | 零初始化 | 损失权重 | 来源 |
|---|---|---|---|---|---|---|---|
| DETR | 可学习位置编码（learnt positional encodings），在每层 attention 输入上相加；内容向量 `tgt = torch.zeros_like(query_embed)` | N=100 | CNN 特征 1×1 降维 + 展平 d×HW，再过 6 层 encoder | 官方默认 post-norm（`--pre_norm` 为 `action='store_true'`，默认关闭）；forward_post = self-attn → norm1 → cross-attn → norm2 → FFN → norm3 | **无**。`for p in self.parameters(): if p.dim() > 1: nn.init.xavier_uniform_(p)` | bbox 5 / giou 2 / eos 0.1 / cost_class 1 | https://ar5iv.labs.arxiv.org/html/2005.12872；`models/transformer.py`、`main.py` |
| Mask2Former | 两个 nn.Embedding：`query_feat`（可学习 query 内容）+ `query_embed`（可学习 query p.e.）；进 decoder 前就直接监督 | 100 | pixel decoder 的 1/32、1/16、1/8 三尺度 round robin（`src[level_index]`） | `# attention: cross-attention first` → cross-attn(masked) → self-attn → FFN；论文附录表 12(b)：MaskFormer=SA-CA-FFN，Mask2Former=MA-SA-FFN。官方 config `PRE_NORM: False`（post-norm） | **无**，三个 layer 类的 `_reset_parameters` 均为 xavier_uniform_ | COCO panoptic：MASK 5.0 / DICE 5.0 / CLASS 2.0 / NO_OBJECT 0.1；默认 config MASK 20.0 / DICE 1.0 / CLASS 1.0 / NO_OBJECT 0.1 | https://ar5iv.labs.arxiv.org/html/2112.01527；`mask2former_transformer_decoder.py`、`config.py`、`maskformer2_R50_bs16_50ep.yaml` |
| Mask2Former masked attention 细节 | — | — | `attn_mask = (attn_mask.sigmoid().flatten(2)... < 0.5).bool()` 并 `.detach()`；True = 禁止注意；守卫 `attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False`（整行全屏蔽时改为全部放开）；第 0 层的 mask 由进 decoder 前的 query features 直接预测 | — | — | — | 同上（论文 Eq.(2)(5) 的 LaTeX 被 ar5iv 剥离，阈值与 −inf 写法只从官方代码取到） |
| MetaQueries | 往词表加 N 个新 token id：`resize_token_embeddings(num_embeddings + num_metaqueries + 2)` + `add_special_tokens(['<begin_of_img>','<end_of_img>'] + [f'<img{i}>'])`；原有词表行用 grad 钩子冻住 `grad[:self.num_embeddings].zero_()`；`lm_head = nn.Identity()`，切 BOI/EOI 之间的行 | 默认 256（消融 64 / 512） | 下游**没有 cross-attention**：connector = 24 层 Qwen2Encoder（为 connector 开双向注意力）+ Linear + GELU + Linear + RMSNorm(权重初值 sqrt(5.5))；扩散模型再来 cross-attend 这 N 行 | 冻结 MLLM 内**保持 causal masking**：`we continue to use causal masking for the entire sequence rather than specifically enabling full attention for Q` | 未提 | 训练目标 = 扩散模型原本的 denoising / flow 目标 | https://arxiv.org/html/2504.06256v1；`facebookresearch/metaquery · models/model.py`、`configs/qwen2p5vl3b_sana.yaml` |
| LISA | 词表扩一个 `<SEG>`；取该 token 的 LLM 最后一层 embedding | 每个分割目标 1 个 | SAM image embeddings（以 sparse prompt 进 SAM 的 two-way decoder） | — | **无** | λ_txt=1.0、λ_mask=1.0、λ_bce=2.0、λ_dice=0.5 | https://ar5iv.labs.arxiv.org/html/2308.00692；`model/LISA.py` |
| PixelLM | segmentation codebook `C_seg = {c_n^l}`，n=1..N、l=1..L（L 个视觉尺度）；N>1 时线性投影 φ 融合成一个 h^l | N×L | CLIP 多尺度 f_img；上一尺度 mask 调制下一尺度特征 `f'_img^l = f_img^l ⊙ (σ(m^{l+1}) + 1)` | — | 未提 | `L = L_txt + λ_ref L_ref + λ_dice L_dice`，`A_i = α 当 Σ_k M̂_ki ≥ 2 否则 1` | https://ar5iv.labs.arxiv.org/html/2312.02228 |

MetaQueries 的 query 数量标度（LLaVA-OneVision-0.5B + Sana-0.6B 512，25M pairs 4 epochs；MJHQ-30K FID / GenEval / DPG）：LLM last layer embedding 7.49 / 0.55 / 78.41；Random queries N=64 → 8.59 / 0.35 / 54.81；Learnable queries N=64 → 7.43 / 0.56 / 75.35；N=512 → 7.34 / 0.56 / 78.43。connector 架构：Enc-Proj 24 层 896 维 316M → 7.43 / 0.56 / 75.35；Proj-Enc 24 层 2304 维 2046M → 7.41 / 0.51 / 73.75。冻结与否：Frozen MLLM + frozen DiT 7.43/0.56/75.35，MLLM tuning 7.75/0.58/78.97，E2E tuning 6.28/0.61/79.39。摘要逐字：`this transfer is effective even when the MLLM backbone remains frozen`。
https://arxiv.org/html/2504.06256v1

PixelLM 的 query 数标度：N=1 对应无 token fusion，N 增到 2 和 3 分别带来 cIoU +0.9%~2.1%、gIoU +1.3%~2.1%，再增收益很小；codebook 从 N×L 退化成 N（跨尺度共享）会变差。https://ar5iv.labs.arxiv.org/html/2312.02228

### 4.2 LLM hidden → 连续参数的非 MLP 读出

| 工作 | 读出结构 | 条件形态 | 训练目标 | 数值 | 来源 |
|---|---|---|---|---|---|
| π0 | 两套权重（PaliGemma 主干 width=2048/depth=18/mlp_dim=16384；action expert width=1024/mlp_dim=4096/~300M），只在 self-attention 层交互，动作 token 之间全双向 | 整条 token 序列 | 条件 flow matching：`τ~Beta((s−τ)/s;1.5,1)`，`A_t^τ=τA_t+(1−τ)ε`，回归 `u=A_t−ε`；代码 `return jnp.mean(jnp.square(v_t − u_t), axis=-1)`，无加权项、无辅助 loss | 推理 10 步积分；`time = beta(1.5,1)*0.999+0.001` | https://arxiv.org/html/2410.24164v4；`src/openpi/models/pi0.py` |
| Octo | learned readout token（只读不被读，等同 BERT [CLS]）→ MAPHead（multi-head attention pooling，用可学 query 对 token 组做一次 cross-attention）或 mean pooling → 3 层 MLP（hidden 256，残差 + LayerNorm）扩散 head | readout token 组 | DDPM，cosine schedule，20 步；并排比较 diffusion / MSE(L2) / 256-bin 离散 CE | `MSEActionHead` 与 `L1ActionHead` 默认 `use_map=True`；`ContinuousActionHead` 默认 `use_map=False`；`DiscreteActionHead` 支持 `token_per="action_dim_and_action_horizon"` | https://arxiv.org/html/2405.12213v2；`octo/model/components/action_heads.py` |
| OpenVLA | 无回归 head；逐维离散 256 bin（边界取训练集 1%/99% 分位数），覆写 Llama tokenizer 最低频 256 个 token | — | next-token CE（只在 action token 上） | 256 bin | https://arxiv.org/html/2406.09246v3 |
| RT-2 | 动作拼成字符串（如 `1 128 91 241 5 101 127`）作为 VLM 微调目标；解码时限制输出词表 | — | CE | 256 bin 均匀离散 | https://ar5iv.labs.arxiv.org/html/2307.15818 |
| OpenVLA-OFT | causal mask 换 bidirectional，decoder 输入插入 empty action embeddings（彼此只靠位置编码区分），一次前向并行出整个 chunk；连续动作由 MLP 头从最后一层 hidden 映射 | LLM 序列 | L1 回归 / diffusion 两种变体 | 论文写「4 层 ReLU MLP」，代码为 `MLPResNet(num_blocks=2)`（LayerNorm→Linear→ReLU→2×pre-LN 残差 FFN→LayerNorm→Linear）；LIBERO 平均 76.5%→97.1%；continuous 比 discrete +5%（绝对）；L1 与 diffusion 原文写 `comparable`；diffusion 变体 50 步去噪、latency ×3 | https://arxiv.org/html/2502.19645v1；`prismatic/models/action_heads.py` |
| CogACT | LLaMA-2 序列末尾加一个 learnable cognition token，其输出特征作为唯一条件 → DiT；cognition feature 与 noisy action 一起作为 transformer 输入 token | 单向量 | MSE(预测噪声, 真噪声) | 默认 **N=15（未来步数），context length = N+2 = 17**（核查修正）；同参数量对照（平均成功率）：MLP(3层,3M)=50.6、MLP(7层,89M)=52.5、DiT-Small(13M)=58.5、DiT-Base(89M)=62.5、DiT-Large(308M)=64.8 | https://arxiv.org/html/2411.19650v1 |
| MGIE | MLLM 生成的 expressive instruction 后追加 N=8 个 `[IMG]` token（word embedding 可训练）；edit head 𝒯 = 4 层 Transformer，输入是 `[IMG]` 的 word embedding e 与最后一层 hidden h 之和，带 L=77 个 learnable query embedding，输出 77×768 扩散条件 | K 个 token | `L_all = L_ins + 0.5·L_edit` | `[IMG]` 数量少会显著掉点，>4 之后趋于持平 | https://arxiv.org/html/2309.17102v2 |
| LISA | 单个 `<SEG>` hidden → 2 层 MLP（`Linear(in,in) → ReLU → Linear(in,out) → Dropout(0.0)`，in=4096、out=256）→ 作为 SAM 的 sparse prompt（`text_embeds=pred_embeddings[i].unsqueeze(1)`）→ SAM transformer mask_decoder | 单向量 | CE + BCE + Dice | 论文写通道 `[256, 4096, 4096]`，代码走向为 4096 → 4096 → 256（书写顺序相反） | `model/LISA.py` |
| InstructIR | 冻结 BGE-micro-v2 句向量 + 100K 参数线性投影 + l2 归一化；注入方式是逐通道 sigmoid 门控（task routing）：`ℱ′_c = Block(ℱ_c ⊙ m_c) + ℱ_c`，`m_c = σ(W_c·e)`；ICB 另带 zero-init 的 gamma/beta | 单向量 | `L = L1 + L_ce` | 系数均为 1 | https://arxiv.org/html/2401.16468v2；`models/instructir.py` |
| VeraRetouch (arXiv:2604.27375v2) | FastVLM-0.5B 上设 3 个 special retouch token（light / global color / specific color），取最后一层 hidden，过三层 bottleneck MLP「Retouch Adaptor」得三组解耦 control latent；latent 以加性注入方式喂进纯 MLP 的逐像素 Retouch Renderer | 3 个 token | `L_total = α·L_CE^text + L_1^img` | 用 binary mask M_l/M_gc/M_sc 随机屏蔽单个 latent 以强制解耦；消融并排给出 latents-pred（0.061 / 24.11 / 0.905 / 0.057 / 0.042）与直接预测 LightRoom 离散参数（param-pred） | https://arxiv.org/html/2604.27375v2 |
| InstructPix2Pix | 文本走 SD 原有 UNet cross-attention（机制未改）；图像条件把 ℰ(c_I) 与 z_t 在第一层卷积通道拼接，新增通道权重 zero-init | token 序列 | latent diffusion L2 | 推理两套 CFG 尺度 s_T∈5–10、s_I∈1–1.5 | https://arxiv.org/html/2211.09800v2 |

### 4.3 N 基元 ↔ N query 的前馈生成器（3DGS 方向）

#### 4.3.1 query 形态的三种接法

| 工作 | query 形态 | 数量 | 条件注入点 | 结构 | 来源 |
|---|---|---|---|---|---|
| AGG | 纯可学习 query（= 一组可学习 position embedding），每个 query 对应一个 3D 高斯；进 transformer 前与 DINOv2 的全局 `[CLS]` **相加** | 4096 / 16384 | `[CLS]` 相加 + cross-attention 读 DINOv2 patch 特征 | block = cross-attn → self-attn → MLP（夹 LayerNorm 与 GeLU）；末层 MLP head 解出 3 维位置；纹理由另一个 transformer 出 triplane，由几何分支的位置去 query | https://arxiv.org/html/2401.04099v1 |
| TGS | 2048 个可学习 positional embedding，每 token 当一个点 | 2048 | cross-attention 读图像 token | 6 层，每 block = self-attn + cross-attn + FFN，hidden 512；其余属性由 triplane 插值特征 + 局部图像特征经 MLP φ_g 一次性输出 | https://arxiv.org/html/2312.09147v1 |
| Gamba | `G = S + E`：S 是（参考图 + Plücker ray 拼 9 通道、经 p=8 大卷积、按 4 种预定义扫描序展开成长度 L 的序列），E 是可学习 per-Gaussian embedding | L=16384 | camera token 与 DINOv2 图像 token 每层 Prepend/Drop 注入序列头部 | 共享 shallow MLP → 每属性一条独立 linear | https://arxiv.org/html/2403.18795v3 |
| GS-LRM / Splatter Image / LGM | 不用 query：从 pixel-aligned 特征 reshape | — | — | GS-LRM 24 层 pre-LN transformer（宽 1024、16 头、MLP 4096），输出侧只用一个 Linear → R^{p²·q}，q=12 | https://arxiv.org/html/2404.19702v1 |

两条与「纯可学习 query」直接相关的对照数字（自变量同时含「有没有 pixel-aligned 捷径」）：
- TGS 消融：`3DG`（直接从 latent token 解码原生可泛化 3D 高斯）PSNR 18.56 / SSIM 0.80 / LPIPS 0.20；Triplane-NeRF 21.85/0.84/0.15；Triplane-Gaussian 23.15/0.87/0.13。原文：`The native generalizable 3D Gaussians, which are directly decoded from latent tokens like points, exhibit the poorest performance according to all metrics.`
- Gamba 消融：Full 23.81/0.88/0.12；w/o Prepending 22.57/0.85/0.10；w/o additive 3DGS tokens（即 `G = E`，论文点名这就是 AGG 与 TGS 的做法）20.35/0.79/0.16；w/o radial mask constraint 12.72/0.58/0.47。
- Splatter Image 消融：把结构化输出换成「全连接、无结构输出」（w/o image）PSNR 22.25→20.60、LPIPS 0.115→0.152。
- AGG 消融：Full 28.54/0.87/0.1426；w/o Super Resolution 27.59/0.87/0.1772；w/o Texture Field（强迫几何 predictor 同时出所有属性）27.09/0.85/0.1910。

#### 4.3.2 参数头激活与初始化（事实）

| 参数组 | 写法 | 出处 |
|---|---|---|
| opacity | `σ(G_opacity − 2.0)`，σ(−2.0)≈0.1 | GS-LRM https://arxiv.org/html/2404.19702v1 |
| opacity | `torch.sigmoid`，末层 bias `opacity_bias = −2.0`、gain `opacity_scale = 1.0` | Splatter Image `configs/default_config.yaml` |
| opacity | `torch.sigmoid` | LGM `core/models.py` |
| opacity | linear + Sigmoid | Gamba |
| scale | `min{exp(G_scale − 2.3), 0.3}`，exp(−2.3)≈0.1；0.3 上限的理由原文写作「不裁剪时高斯退化成长线」 | GS-LRM |
| scale | `torch.exp`，末层 bias `log(scale_bias)`（scale_bias=0.02）、gain `scale_scale = 0.003` | Splatter Image |
| scale | `0.1 * F.softplus(x)`；原文写明是为稳定训练而故意偏离原版 3DGS，使初期高斯靠近场景中心 | LGM |
| scale | linear + Softplus，无约束 | Gamba |
| position | `x.clamp(-1, 1)` | LGM |
| position | 分类式期望：[−0.5,0.5] 每轴 21 个格点 c_j，softmax 后 `y_i = Σ P(Q_ij)·c_j` | Gamba |
| rotation | 预测未归一化四元数再 L2 归一化（`F.normalize`） | GS-LRM / Splatter Image / LGM |
| rgb | `0.5*tanh(x)+0.5`（代码注释 `may use sigmoid if train again`） | LGM |
| 全局初始化 | 全模型 Linear 与 LayerNorm 都不用 bias 项、权重初始化 N(0, 0.02²)，因此初始输出零均值，常数 bias 直接等于目标初值 | GS-LRM |
| 逐组初始化 | `split_dimensions=[1,3,1,3,4,3]`（with_offset 默认，合计 15，再加 SH 9 = 24 通道，**核查修正**）；逐组 `nn.init.xavier_uniform_(weight[组切片], gain=s)` + `nn.init.constant_(bias[组切片], b)`；`scale_inits=[depth_scale, xyz_scale, opacity_scale, scale_scale, 1.0, 5.0]`，`bias_inits=[depth_bias, xyz_bias, opacity_bias, np.log(scale_bias), 0.0, 0.0]` | Splatter Image `scene/gaussian_predictor.py` |
| 冻结不稳定参数组 | scale 与 rotation 完全冻结为常数（各向同性）：4096 高斯时 scale=0.03、16384 时 0.01，rotation 固定 [1,0,0,0]。原文：`we find these attributes extremely unstable during the amortized optimization process` | AGG |
| 范围正则 | `scaling>20` 取 `mean(scaling)*0.1`；`scaling<1e-5` 取 `mean(−log(scaling))*0.1`（仅 hydrants/teddybears 类别开启） | Splatter Image `train_network.py` |
| 外挂几何先验 loss 并退火到 0 | `λ_rdist` 初始 0.1、前 10 epoch 内衰减到 0；去掉它出现「所有 opacity α 变成 0、高斯完全不可见、PSNR 12.72」 | Gamba |

#### 4.3.3 固定基元数下的退化处理

- GVGEN：固定 N³=32,768 个高斯与 32×32×32 网格点双射；`μ = p + Δμ`，只对 offset 回传梯度；Candidate Pool Strategy——被 prune 的点不删除而进候选池变成 deactivated（不参与前反向），densify 时从池中按「距待致密点在 ε_offsets 内的最近点」取回激活；结束时把池中剩余点全部放回再精修。`L_offsets = Mean(ReLU(|Δμ − ε_offsets|))`。拟合消融：Full 30.122/0.963/0.038；w/o CPS 29.677/0.958/0.049；w/o offsets（μ=p 完全固定）27.140/0.936/0.084。生成消融：Full 35.03/0.9872/0.0236；w/o L_3D 35.21/0.9846/0.0268；w/o L_2D 29.55/0.9654/0.0444。https://arxiv.org/html/2403.12957v2
- GaussianCube：保留原版 pruning，只对 densification 加约束（候选采样 `min(N_max − N_c, N_d)`、clone 与 split 拆成交替独立步骤、剪掉 α<ε 的点）；拟合结束后用 α=0 的高斯补齐到 N_max；再用 Optimal Transport（Jonker-Volgenant，按四段近似求解）分配到 N×N×N 体素格，得 32×32×32×14 的结构化张量。N_max=32,768，C=14。https://arxiv.org/html/2403.19655v3

#### 4.3.4 监督位置（Q1 的旁证）

四个前馈工作的主 loss 都在渲染/函数值空间：MSE 或 L1（+ LPIPS 或 VGG perceptual）。参数空间约束只在两处出现：(i) AGG 的 warmup——因为高斯是无序集合，原文写 `Due to these Gaussians being stored in random orders as sets, we cannot use L1 reconstruction loss`，改用 Chamfer 最近邻匹配后在参数空间对 location/opacity/color 做 L1；(ii) GVGEN 生成阶段的 λ_3D·L_3D（消融数字见上）。

### 4.4 条件注入与稳定化

| 机制 | 公式 / 代码 | 零初始化加在哪 | 数值 | 来源 |
|---|---|---|---|---|
| adaLN-Zero (DiT) | `modulate(x, shift, scale) = x*(1 + scale) + shift`；`x = x + gate * attn(modulate(norm(x), shift, scale))`；条件走 `nn.Sequential(nn.SiLU(), nn.Linear(hidden, 6*hidden))` chunk 成 6 组 | 只在 adaLN_modulation 的最后一个 Linear（weight+bias 置 0）与 final_layer 输出 Linear（weight+bias 置 0）；主干 Linear 全部 xavier_uniform，norm1/norm2 用 `elementwise_affine=False` | 论文写明先例是 Goyal et al. 零初始化残差块最后一个 BN 的 scale，以及 diffusion U-Net 零初始化残差前最后一层卷积 | https://ar5iv.labs.arxiv.org/html/2212.09748；`facebookresearch/DiT · models.py:19,113-121,207-216` |
| FiLM | `FiLM(F_{i,c}|γ,β) = γ·F_{i,c} + β`；官方代码 `return (gammas * x) + betas` | 无零初始化；`gamma_option='linear'`、`gamma_baseline=1`，`out + gamma_shift`（网络输出接近 0 时 γ≈1、β≈0 为恒等） | 实测 γ ∈ [−15,19]、β ∈ [−9,16]；36% 的 γ 为负、76% 的 β 为负；消融 β:=0 得 96.9、γ:=1 得 95.9、γ:=σ(γ) 95.9、γ:=tanh(γ) 96.3、γ:=exp(γ) 96.3（完整 97.7） | https://ar5iv.labs.arxiv.org/html/1709.07871；`vr/models/filmed_net.py:18-26`、`vr/models/film_gen.py:30-31,160-161,181-183` |
| ControlNet zero-conv | `y_c = F(x;Θ) + Z(F(x + Z(c;Θz1);Θc);Θz2)`；`zero_module` 把 module 全部 parameters `.zero_()` | 只加在把 trainable copy 输出接回主干的连接层（`self.zero_convs`、`middle_block_out`）与 `input_hint_block` 最后一层；可训练副本内部不零初始化 | 补充材料逐式：W=0,B=0 时 `∂Z/∂B_i = 1`、`∂Z/∂I_{p,i} = ΣW = 0`、`∂Z/∂W_{i,j} = I_{p,j} ≠ 0`；原文写明前提 `As long as the feature I is non-zero, the weight W will be optimized into a non-zero matrix in the first gradient descent iteration` | https://ar5iv.labs.arxiv.org/html/2302.05543v1；https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_Adding_Conditional_Control_ICCV_2023_supplemental.pdf；`ldm/modules/diffusionmodules/util.py:177-183`、`cldm/cldm.py:281-282` |
| ReZero | `x_{i+1} = x_i + α_i F(x_i)`，α 初始 0。玩具模型：`w ← w − λ L α x0 (1+αw)^{L−1} ∂_x C`（含因子 α）；`α ← −λ L w x0 ∂_x C`（不含 α 因子） | 门控标量 | 原文：`Initially the gradients for all parameters defining F vanish, but dynamically evolve to suitable values during initial stages of training`；可训 10000 层 FC、>100 层 Transformer；12 层 Transformer 到 1.2 BPB 快 56%，ResNet 快 32% | https://arxiv.org/pdf/2003.04887 |
| Goyal et al. | 残差块最后一个 BN 的 γ 初始化为 0 | 残差分支末端的乘性缩放，不是分支内部权重 | — | https://arxiv.org/pdf/1706.02677 |
| Flamingo tanh-gating | 新层输出乘 `tanh(α)` 后加回残差流，α 为逐层可学习标量、初始 0 | 门控标量 | 消融 row (iii)：去掉 0-initialized tanh gating 总分下降 4.2%，并记录到训练不稳定 | https://arxiv.org/pdf/2204.14198 |
| GLIGEN | `v = v + β·tanh(γ)·TS(SelfAttn([v, h^e]))`，γ 可学习标量初始 0，β 全程为 1 | 门控标量；attention 的 q/k/v 投影不零初始化 | — | https://arxiv.org/pdf/2301.07093 |
| LoRA | `h = W0 x + BAx`；A 随机高斯、B 零；ΔW x 乘 α/r | 低秩通路最外侧的 B | — | https://arxiv.org/pdf/2106.09685v2 |
| LoRA 初始化端的差异 | Init[A]（A 随机、B 零）最大学习率标度 `γ[η] = −1/2`；Init[B] 为 `−1`；Init[B] 下不出现内部不稳定，但大宽度极限下 B 欠训练，其极限动力学与「B 不训练、只训练 A」相同 | — | — | https://arxiv.org/pdf/2406.08447 |
| HyperNetworks（static） | 每层 layer embedding z^j ∈ R^{Nz}，两层线性 g(z^j) 生成卷积核；`a_i^j = W_i z^j + B_i`，`K_i^j = ⟨W_out, a_i^j⟩ + B_out` | HyperLSTM 附录 A.2.3：前两式权重初始化为零、bias 初始化为 1；第三式权重 std=0.01 正态；`W_bz` 零；`W_hz`/`W_xz` 常数 0.1/Nz（沿用 Recurrent BN 把 scaling 初始化成 0.1 而非 1.0） | — | https://arxiv.org/pdf/1609.09106v4 |
| MIP（Magnitude Invariant Parametrizations） | 在（全连接堆叠 / ϕ(x)=max(αx,0)+min(βx,0) / bias 初始化为 0）三条前提下 `x_i^{(1)} = ϕ(W_i γ + b) = γ·ϕ(W_i) ∝ γ`，归纳得 `‖θ‖₂ ∝ ‖γ‖`、`Var(θ) ∝ ‖γ‖²`。改动：(a) `E_L2(γ)=[cos(γπ/2), sin(γπ/2)]` 使 ‖E_L2(γ)‖ 恒为 1；(b) 输出改残差 `θ = θ0 + h(E_L2(γ);ω)`，θ0 为独立可学习参数、按主网络常规初始化 | — | hypernetwork 权重 ω 用 Kaiming fan-out、bias 置零 | https://arxiv.org/pdf/2304.07645v2 |
| hyperfan init | weight 型输出头 `var_in = c_relu/(c_bias·m_fan_in·fan_in·input_variance)`、`var_out = c_relu/(m_fan_out·fan_in·input_variance)`，'harmonic' 取调和均值；bias 型 `var_out = max(0, c_relu(1 − m_fan_in/m_fan_out)/(fan_in·input_variance))` | — | 代码注释记：hypernetwork 直接用 Xavier/Kaiming 时，生成的主网络权重方差等于输入 embedding 的方差 | `chrhenning/hypnettorch · hypnettorch/hnets/mlp_hnet.py:340-375,608-640,734-760`（原始论文 OpenReview H1lma24tPB 的 PDF 被 challenge 拦截，未打开） |
| Bias-HyperInit | Weight-HyperInit：`W_{:,i} := φ_i ∼ f(φ)`、`b := 0`。Bias-HyperInit：`W_{i,j} := 0 ∀i,j`、`b := φ_shared ∼ f(φ)`，于是 `φ_init = Wx + b = φ_shared`（与条件 x 无关，等于一份按标准方案采样的参数） | 输出头 weight 全零、bias 非零 | 论文报告在 meta-RL 上对 Kaiming、Orthogonal、Normc 三种默认初始化都观察到失败 | https://proceedings.mlr.press/v205/beck23a/beck23a.pdf |
| Text-to-LoRA | 任务文本 embedding（gte 1024D 或 Mistral 4096D）→ 线性压到 64D，与 32D module embedding、32D layer embedding 拼接 → mlp0 + 若干 residual MLP block（pre-LayerNorm → Linear → SiLU → Dropout(0.05) → Linear → SiLU，外层 `x + mlp(x)`）→ 线性 head 输出 LoRA 的 A/B | 用 Bias-HyperInit：`nn.init.zeros_(layer.weight)` + `head.bias.copy_(torch.cat(init_bias))`，init_bias 来自 `get_init_peft_weights`（A head 为 U(−1/d, 1/d)、B head 全零）；shared head 情形 bias 再除以 sqrt(2) 或 sqrt(r) 以「match the gradient scale」 | — | https://arxiv.org/pdf/2506.06105；`src/hyper_llm_modulator/hyper_modulator.py:494-501,532-554,195-215` |
| PromptIR | `prompt_param` 形状 (1, prompt_len=5, prompt_dim, prompt_size, prompt_size)；`emb = x.mean(dim=(-2,-1))`；`prompt_weights = F.softmax(self.linear_layer(emb), dim=1)`；加权求和后 bilinear 插值到 (H,W) 再过 3×3 conv，与 decoder 特征 concat | — | prompt_len=5；权重来源是图像特征而非文本 | `net/model.py` |
| Dying ReLU | 定义 BD（born dead）= 初始化后整网退化为常函数。Theorem 3.4：若参数初始化落在 BD 集合内，则对任意损失函数、任意基于梯度的方法，网络被优化成一个常数函数。相位划分：训练前已死（BD）与训练后死；相位 1 蕴含相位 2，反之不成立 | — | 数值例：10 层、宽 2 的 ReLU 网络拟合 f(x)=\|x\|，1000 次独立实验中 >90% 塌缩为常函数；对策为 RAI（随机非对称初始化），BD 概率上界 `P(J) ≤ 1 − Π_{ℓ=1}^{L−1}(1 − (1/2 − γ_ℓ)^{N_ℓ})` | https://arxiv.org/pdf/1903.06733v3 |
| GELU / SiLU | `GELU(x) = x·Φ(x)`；把 Φ 换成 Logistic CDF 得 `SiLU = x·σ(x)`。原文对 ReLU 的刻画：`weights inputs by their value, rather than gates inputs by their sign` | — | 中位错误率：CIFAR-10 九层卷积 GELU 7.89% / ReLU 8.16% / ELU 8.41%；CIFAR-100 WRN-40-4 20.74% / 21.77% / 22.98%；TIMIT 29.3% / 29.5% / 29.6%；Twitter POS 12.57% / 12.67% / 12.91% | https://ar5iv.labs.arxiv.org/html/1606.08415 |

零初始化「加在哪」的三种已核实形态（与本项目 head_color 的对照）：
- (a) **门控系数**（DiT gate、Goyal 的 BN γ、ReZero α、Flamingo tanh(α)、GLIGEN γ）：被门控的分支内部权重是常规随机初始化，step0 输出等于 baseline，分支内部前向仍在算非零激活。
- (b) **并联支路最外侧一层的权重（含 bias）**（ControlNet zero-conv、LoRA 的 B）：该支路上游（trainable copy、LoRA 的 A）保持随机初始化并接收非零输入。
- (c) **输出头 weight 全零但 bias 非零**（Bias-HyperInit、Text-to-LoRA）：bias 直接拷贝主网络（或 LoRA）的标准初始化样本，step0 生成的是一份合法随机初始化参数、而不是零参数。
- 本项目 head_color 末层是「零初始化末层 + 上游 ReLU 隐层」；上游隐层被观测到 ReLU 全局死亡率 0.469→1.0000。ControlNet 补充材料的推导前提是 I ≠ 0（`∂Z/∂W = I`）；当上游 ReLU 全死时 I ≡ 0（常数），`∂Z/∂W = 0`。Dying ReLU 的 Theorem 3.4 描述的是「参数落入 BD 集合后，任意梯度方法只把网络优化成常数函数」。
- DiT 的调制写成 `x*(1+scale)+shift`、FiLM 官方代码把 γ 加常数 `gamma_baseline=1`，两者都用「偏移到恒等」而不是「输出为零」来实现 step0 恒等。

### 4.5 ST_LANG → GLUT 适配的机制对照表

（ST_LANG 侧为本地代码事实：q3vl/whereb/amort/uniq4.py + uniq4b.py；GLUT 侧只列对应关系与开放问题。）

| ST_LANG 组件 | 本地机制 | GLUT 侧候选对应物 | 外部同型出处（含 URL） | 开放问题 |
|---|---|---|---|---|
| K=8 个新 token id，embedding 可训练（经 embedding forward hook 写入，resize 出来的行本身冻结） | 追加在 prompt 推理段之后 | ① K=48 个 token，一个 token 对应一个高斯基元；② K 个 token 按参数组切分（μ/cov/opacity/color/global 各一组）；③ K 为纯容量旋钮，与 48 无关 | MetaQueries：resize + `add_special_tokens` + `grad[:num_embeddings].zero_()` 钩子（https://raw.githubusercontent.com/facebookresearch/metaquery/main/models/model.py）；MGIE N=8 个 `[IMG]`（https://arxiv.org/html/2309.17102v2）；VeraRetouch 3 个 special retouch token 按参数组分（https://arxiv.org/html/2604.27375v2） | ST_LANG 用 forward hook 写入、MetaQueries 用 grad 钩子清零原有行；两者在 weight decay、优化器状态、tied lm_head 存在时对「哪些行真的被更新」是否等价，外部来源没有讨论。GLUT 的 48 个高斯在函数值上可置换，「一个 token 一个基元」的绑定是否需要规范排序，无先例 |
| VLM 自身 36 层做聚合（K 个 token 的最后一层 hidden 直接作为下游 query 状态） | `UniQ4Head._query_states`：`qh = h_where[0, -rows:, :]` 后过 `q_proj_in` | 同样取 K 个 token 的最后一层 hidden 作为 GLUT decoder 的 query | MetaQueries（把 `lm_head` 换成 `nn.Identity()`，切 BOI/EOI 之间的行；冻结 MLLM 内保持 causal mask，未为 Q 开双向）；LISA 单个 `<SEG>` hidden；PixelLM N×L 个 codebook token 线性融合成 L 个；CogACT 单个 cognition token | ST_LANG 的 K 个 token 之间能否互相看到（attention mask 构造）需读本地 uniq4b.py 确认，外部来源无法回答。MetaQueries 明写沿用 causal mask |
| masked cross-attention（用上一步预测场限制每个 query 能读什么） | Mask2Former 式；本地 memory 为空间特征场 | memory 候选：① 每步 Q=256 个采样查询色 x 的编码（此时 attn_mask 需由「每个高斯基元对这些色的 responsibility」给出）；② 真值 LUT 在若干固定色域锚点上的采样；③ z 的多 token 展开；④ 图像的色彩统计（Lab 直方图/GAP 向量） | Mask2Former：`attn_mask = (attn_mask.sigmoid().flatten(2)... < 0.5).bool()` 且 `.detach()`，True=禁止；守卫 `attn_mask[全行被屏蔽] = False`（https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py）。RAG 颜色复原用「检索图 Lab 的 ab 直方图 GAP 成 512 维」（https://arxiv.org/html/2608.08211v1）；StatLUT 用 Lab 统计直方图特征（https://arxiv.org/pdf/2607.08227） | **本轮打开的全部来源中，没有任何一篇把 query-based decoder 的 cross-attention 用在色域/颜色变换上**。DETR/M2F/LISA/PixelLM 的 memory 一律是空间图像特征；MetaQueries 下游没有 cross-attention。「色域 cross-attention 读什么」属于本项目外推 |
| 两个 attention 的 out_proj 与 FFN 末层零初始化（保证 step0 与 baseline 逐位相同） | 权重矩阵整体零初始化 | 同样零初始化 GLUT decoder 的 out_proj / FFN 末层；或改为 (a) 门控标量 / (b) Bias-HyperInit（weight 零、bias 拷贝各参数组的目标初值） | DETR/Mask2Former/LISA/PixelLM 官方代码**都没有** out_proj 零初始化（一律 xavier_uniform_）；DETR 唯一的零是 `tgt = torch.zeros_like(query_embed)`，而 M2F 把这一项换成可学习 Embedding。零初始化的三种外部形态见 §4.4 | ReZero 的梯度结构（Eq.6/Eq.8）针对的是**标量门**；**权重矩阵整体零初始化**时分支内上游权重在 step 0 是否拿到零梯度、需多少步脱离，原文未覆盖，未找到直接文献。ControlNet 的推导前提 I ≠ 0 在上游 ReLU 全死时不成立 |
| pre-norm（有意偏离 Mask2Former 默认） | 本地 docstring 记录：post-norm 下 LayerNorm(q+0) ≠ q，零初始化无法构成恒等 | 同 | 官方 Mask2Former `PRE_NORM: False`、官方 DETR `--pre_norm` 默认关闭（两处已核实）；本地偏离与该事实一致 | 没有任何外部来源为 query decoder 选 pre-norm 而非 post-norm 给出理由；该偏离只由「step0 逐位等价」这一本地约束驱动，该约束在五篇（DETR/M2F/MetaQueries/LISA/PixelLM）里都不存在 |
| language-only LoRA（`^(?!.*visual).*\.(q_proj\|k_proj\|v_proj\|o_proj)$`） | 变体 ST_LANG 叠加 | 同样正则；或不加（全冻结） | LoRA A 随机 / B 零（https://arxiv.org/pdf/2106.09685v2）；初始化端差异（https://arxiv.org/pdf/2406.08447）；LISA 用 LoRA 调 LLM、冻结 vision backbone；MetaQueries 冻结 MLLM 时 FID 7.43 / GenEval 0.56 / DPG 75.35，MLLM tuning 7.75 / 0.58 / 78.97，E2E 6.28 / 0.61 / 79.39 | 「LoRA 是否与冻结口径的评测可比性冲突」由本项目自定，无外部依据 |
| **没有直接对应物的组件** | | | | |
| 「上一步预测场」这一物件 | ST_LANG 的 attn_mask 来自上一层的空间 mask 预测 | GLUT 没有 per-primitive 目标（只有聚合函数值 L1），48 个高斯在函数值下可置换 | DETR/M2F 用 Hungarian 匹配把无序集合对齐到 GT 集合；AGG 用 Chamfer 最近邻匹配后才能在参数空间算 L1 | 五篇 query-decoder 工作里没有任何一篇为「无 per-slot 目标、只有聚合函数值监督」的情形定义匹配代价 |
| 多尺度 memory（M2F 的 1/32、1/16、1/8 round robin） | 本地沿用空间多尺度 | 色彩空间没有「尺度」的对应物；最接近的候选是「查询色采样密度」或「色域分块粒度」 | PixelLM 用 L 个视觉尺度 → N×L 个 codebook token | 无先例 |
| pixel-aligned 捷径 | where 侧存在（query ↔ 空间位置） | z 是单向量，不存在与 48 个基元同构的输入域 | TGS 3DG 消融 18.56 vs 23.15；Gamba G=E 消融 20.35 vs 23.81；Splatter Image w/o image 22.25→20.60 | 这三个对照的自变量同时含「有没有 pixel-aligned 捷径」，无法从中分离「query 结构本身」的贡献 |

### 4.6 候选结构方案表（并列陈述，不排序、不推荐）

| # | 结构 | 需要改哪里 | 来源依据 | 与冻结口径的冲突点 |
|---|---|---|---|---|
| S1 | ST_LANG 直搬：K 个新 token（embedding 可训练）→ VLM 36 层聚合 → 最后一层 hidden 作 query → RefineLayer（pre-norm，masked cross-attn → self-attn → FFN，out_proj/FFN 末层零初始化）；memory 取 Q=256 个查询色的编码 | prompt 追加 K 个 token；写 GLUT 版 RefineLayer 与 attn_mask 生成器 | 本地 uniq4.py/uniq4b.py；MetaQueries token 写入机制；Mask2Former 的 attn_mask 二值化与全行守卫 | VLM 保持冻结（token embedding 可训练属于既有 ST_LANG 口径）。attn_mask 需要「每个基元对每个查询色的 responsibility」这一预测量，无外部先例 |
| S2 | K 个 token 按参数组切分（μ/cov/opacity/color/global 各一组或各一个） | prompt token 分组 + 每组一个读出头 | VeraRetouch 3 个 special retouch token（light / global color / specific color），用 binary mask 随机屏蔽单个 latent 以强制解耦（https://arxiv.org/html/2604.27375v2） | 组数与 22N+12 的切分方式由本项目自定；VeraRetouch 未给出 latent 维度与 α 数值 |
| S3 | N=48 个可学习 query 放在下游 decoder，z 作为全局 token 与每个 query **相加** | 下游 decoder 自带 nn.Embedding；z 经 LayerNorm+Linear 后广播相加 | AGG（query = 可学习 pos-emb 与 DINOv2 `[CLS]` 相加；block = cross-attn → self-attn → MLP）（https://arxiv.org/html/2401.04099v1）；TGS point decoder 2048 个 token（https://arxiv.org/html/2312.09147v1） | AGG/TGS 的 cross-attention memory 是图像 patch 特征；GLUT 侧 memory 待定（见 §4.5） |
| S4 | query + 锚点编码相加：`G = S + E`，S = 48 个预设色域锚点（μ 初始网格）的编码，E = 可学习 per-primitive embedding | 需要预设 48 个色域锚点 | Gamba（G=S+E；w/o additive tokens 即 G=E 时 PSNR 20.35 vs Full 23.81）（https://arxiv.org/html/2403.18795v3） | Gamba 的 S 来自 pixel-aligned 扫描序，GLUT 的锚点由人工设定，两者性质不同 |
| S5 | attention pooling 读出：用可学 query 对 token 组做一次 cross-attention（MAPHead），替代 mean pooling / 单 token 读出 | 在 `<seg_color>` 附近取一组 token 而非单个 | Octo（`MSEActionHead`/`L1ActionHead` 默认 `use_map=True`）（https://raw.githubusercontent.com/octo-models/octo/main/octo/model/components/action_heads.py） | 需要改「单个 `<seg_color>` hidden 读出」这一现有接口（改为读一组 token） |
| S6 | 把 48（或 22N+12 分组）个 empty param embeddings 插进 VLM 自身，causal mask 换 bidirectional，一次前向并行出全部 slot | 改 VLM 的 attention mask | OpenVLA-OFT（empty action embeddings，彼此只靠位置编码区分；LIBERO 76.5%→97.1%；continuous 比 discrete +5%）（https://arxiv.org/html/2502.19645v1） | 改 causal mask 属于改动冻结 VLM 的前向行为；MetaQueries 明写它**不**为 query 开双向（沿用 causal） |
| S7 | DiT / adaLN-Zero 参数生成器：z（+ 若干 token）作为条件，N 个 slot 作为 DiT 输入 token，`x*(1+scale)+shift` 调制、gate 零初始化 | 换掉 MLP encoder + 各 head | DiT（https://ar5iv.labs.arxiv.org/html/2212.09748；`facebookresearch/DiT · models.py`）；CogACT 同参数量对照 MLP(7层,89M)=52.5 vs DiT-Base(89M)=62.5（https://arxiv.org/html/2411.19650v1，核查修正后 N=15 / context 17） | CogACT 的 DiT 是扩散去噪器（生成式）；若只做确定性回归，该对照的自变量不止「MLP vs DiT」 |
| S8 | FiLM / 逐通道 sigmoid 门控注入：`ℱ′ = Block(ℱ ⊙ σ(W·e)) + ℱ`，另带 zero-init 的 gamma/beta | 条件注入方式从「z→u 拼进 MLP」改为门控调制 | InstructIR ICB（https://raw.githubusercontent.com/mv-lab/InstructIR/main/models/instructir.py）；FiLM 官方 `gamma_baseline=1`；InstantRetouch 记「简单相加」高于 adaLN 与 cross-attention | FiLM 实测 γ∈[−15,19]、36% 为负；本项目条件通路已观测 ReLU 全死，门控与 ReLU 的组合行为无外部覆盖 |
| S9 | HyperNetwork + Bias-HyperInit / MIP 残差：`θ = θ0 + h(E_L2(z);ω)`，输出头 weight 零、bias 拷贝各参数组目标初值 | 输出头初始化与残差形式 | Bias-HyperInit（https://proceedings.mlr.press/v205/beck23a/beck23a.pdf）；Text-to-LoRA（`nn.init.zeros_(layer.weight)` + `head.bias.copy_(init_bias)`，backbone 用 pre-LayerNorm + SiLU + 残差）；MIP `Var(θ) ∝ ‖γ‖²`（https://arxiv.org/pdf/2304.07645v2） | GLUT 的 μ/Cholesky/opacity/M,b/G,g 没有「Kaiming/Xavier 分布」这样的标准 f，bias 该拷贝什么值属本项目自定，无外部先例。MIP 的推导前提之一是 hypernetwork 无 normalization 层；本项目 z 先过 LayerNorm |
| S10 | 基元库 softmax 加权：持有一个可学基元库，权重由 z 经 Linear + softmax 给出 | 生成器从「回归 48 组参数」改为「在库上加权」 | PromptIR（prompt_len=5，权重 = softmax(Linear(GAP 特征))）；Image-Adaptive 3D LUT（3 个基 LUT，LUT0 恒等初始化、LUT1/2 零初始化，classifier 出 3 个标量） | PromptIR 与 3DLUT 的权重都来自图像特征而非文本；库容量与 B3 桶检索（ΔE00 6.155/6.064）的关系需自定 |
| S11 | 参数头激活与初始化改造（可与任一结构叠加）：opacity `σ(x−2.0)`；Cholesky 对角 `min{exp(x−2.3), 0.3}` 或 `0.1·softplus(x)` 或 `exp` + bias=log(0.02)/gain=0.003；μ 用 21 格点 softmax 期望；全模型去 bias、权重 N(0,0.02²) | 只改各 head 的末层 | GS-LRM（https://arxiv.org/html/2404.19702v1）；Splatter Image `get_splits_and_inits`（逐组 gain/bias，**默认 24 通道，核查修正**）；LGM `core/models.py`；Gamba 21 格点 | Gamba 的 21 格点期望本身仍是加权平均，原文未测其在一对多下的行为（无 argmax/采样对照） |
| S12 | connector 式：K 个 token hidden → 24 层双向 self-attention encoder（不带 cross-attention）→ 投影到 GLUT 参数 | 下游换成 encoder + 投影 | MetaQueries Enc-Proj 24 层 896 维 316M → FID 7.43 / GenEval 0.56 / DPG 75.35；Proj-Enc 24 层 2304 维 2046M → 7.41 / 0.51 / 73.75 | MetaQueries 下游是大型扩散模型；下游换成 1068 个标量的小解码器时的行为无来源覆盖 |
| S13 | language-only LoRA 的保留 / 移除（与 S1~S12 正交） | 正则 `^(?!.*visual).*\.(q_proj\|k_proj\|v_proj\|o_proj)$` 的开关 | MetaQueries 三档：Frozen 7.43/0.56/75.35、MLLM tuning 7.75/0.58/78.97、E2E 6.28/0.61/79.39；LISA 用 LoRA 调 LLM、冻结 vision backbone | 与「冻结 VLM」这一口径的边界由本项目自定 |

合计 **13 个候选结构方案**。

---

## 5. 开放问题清单（需要用户裁定）

### 5.1 Q1 侧

1. **ΔE00 作为训练 loss**：本轮在已核实来源中没有找到任何把完整 ΔE00 当可微训练 loss 的已发表工作或公开实现（GitHub 搜 `ciede2000+language:python` 的 34 个仓库中无一是可微训练 loss；检索引擎给出的三个 soft-CIEDE2000 arXiv 号与一个仓库经核实全部不存在）。是否再用 Google Scholar / Semantic Scholar 全文检索确认一轮。
2. **L_hc 的第三处出处**：「chroma·(1−cos Δhue)」这一确切写法（一次幂、未归一化 CIELab chroma、无 S_H 分母）没有找到独立出处。最接近的两个是 CIEDE2000 的 ΔH²（带 S_H 分母）与 CURL 的锥形 HSV（幅值 S·V∈[0,1]）。是否需要继续找。
3. **batch 内 chroma 均值归一 vs CoV 的 loss 标量历史均值归一**：作用对象不同（张量内权重 vs 标量项系数），二者是否叠加、叠加后 Adam 的等价缩放行为如何，无外部文献覆盖，需自己做消融。
4. **3D RGB 分箱的参数**：Colorful Colorization 只在 2D ab 上量化得 313 bin。3D RGB 上等效 grid 的 bin 数、稀疏性、soft-encoding 的近邻数与核宽，在任何已核实来源中都没有给出。另：论文写 5 近邻、官方 caffe 层写 `NN = 10`，若照搬需先决定采用哪一个。
5. **无空间结构的条件判别器**：pix2pix 与 SRGAN 的 D 都吃图像。对「只有 8192 个 (查询色, 输出色) 对、完全无空间结构」的情形，D 该吃什么输入、什么感受野，无对应设置（最接近只有 pix2pix 的 1×1 PixelGAN）。
6. **置换不变参数集合的条件生成**：GLUT 的 48 个基元在函数值上可置换，作为 CFM/扩散的生成目标时缺一个规范排序；本轮未找到处理该情形的做法与 loss 写法。
7. **检索基底 + 残差加在哪一层**：残差加在 GLUT 参数空间（μ/Cholesky/仿射非线性、不可直接相加）还是函数值空间（f_base(x)+Δ(x)），任一篇打开的文献都没有直接讨论。
8. **B5（VQ token + CE）与冻结口径的兼容性**：AceTone 的 CE 监督是 next-token 自回归，需要 VLM 自回归吐 token（扩词表 + 调 MLP connector + 调语言模型）。这与「冻结 VLM + 单个 `<seg_color>` hidden 一次性读出」的现有约束不兼容；是否接受该改动需裁定。
9. **量化对象**：量化 GLUT 参数本身（有排列不变性）还是量化 f 在 32³ 网格上的采样（与 AceTone 完全一致但丢掉 GLUT 参数化），未找到直接可比的先例。
10. **纯 LUT-id 分类**：本轮未找到「把预测目标直接设为库内 LUT 的 id」的公开工作（最接近是 AceTone 的 256 路 codebook token 与 Zeng 的固定基 LUT 连续权重）。
11. **推理端单点评测**：B3（MDN）、B4（CFM）、B8（style-NLL）都会产出分布或需要采样；headline ΔE00 的单点评测口径（采样几次、取哪一个、是否算 oracle）需重新定义。

### 5.2 Q2 侧

12. **色域 cross-attention 的 memory**：已核实的全部 query-decoder 工作里，memory 一律是空间图像特征（DETR/M2F/LISA/PixelLM）或没有 cross-attention（MetaQueries）。GLUT 侧 memory 取什么（查询色编码 / 锚点采样 / z 多 token 展开 / Lab 直方图）无先例，属外推。
13. **无 per-slot 目标下的 attn_mask**：M2F 的 masked attention 需要「上一层的 mask 预测」；GLUT 只有聚合函数值 L1，没有 per-primitive 目标。要用 responsibility 当 mask，其定义与是否 detach 无先例。
14. **N 个 query ↔ N 个基元的绑定是否需要匹配代价**：DETR/M2F 用 Hungarian 对齐无序集合，AGG 用 Chamfer 匹配后才能在参数空间算 L1；GLUT 在函数值监督下是否需要、以及如何定义匹配代价，五篇里无覆盖。
15. **权重矩阵整体零初始化的梯度行为**：ReZero 的 Eq.(6)/(8) 针对标量门；ControlNet 的推导前提是 I ≠ 0。ST_LANG 用的是 out_proj 权重矩阵整体零初始化，且本项目已观测上游 ReLU 全死（I ≡ 0）。这一组合无直接文献。
16. **ST_LANG 的 forward hook 写入 vs MetaQueries 的 grad 钩子清零**：在 weight decay、优化器状态、tied lm_head 存在时是否等价，外部无讨论。
17. **K 个 token 之间的 attention mask**：MetaQueries 明写沿用 causal（不为 Q 开双向）；ST_LANG 的 K 个 token 之间能否互看需读本地 uniq4b.py 确认。
18. **前馈生成器内部有无「基元复活」机制**：GVGEN 的 Candidate Pool 与 GaussianCube 的 densification-constrained fitting 都发生在逐物体优化阶段，不在前馈生成器里；本次检索未找到前馈内的等价机制（Gamba 用的是外挂先验 loss 并退火到 0）。
19. **冻结形状、只留仿射**：AGG 冻结 scale/rotation 后外观由独立 triplane 纹理场承担；GLUT 若冻结 Cholesky，表达力必须由 M,b 承担——本批文献无「冻结形状、只留仿射」的对照数字。
20. **条件通路激活死亡的诊断**：本轮打开的全部来源中，没有任何一篇报告过条件通路 MLP 的 ReLU 死亡率统计或塌缩分析（CogACT 的 MLP vs DiT 对照只给成功率）。

### 5.3 引文与核实状态相关

21. **未打开正文的来源**（不得据此写方法细节）：CLUT-Net 的 ACM MM 2022 论文正文（DOI 10.1145/3503161.3547879，非 arXiv、需订阅）；CURL supplementary（ω 的具体数值是否列出未核实）；Chang et al.《Principled Weight Initialization for Hypernetworks》原文 PDF（OpenReview `/pdf` 与 `/attachment` 均返回 challenge，API 403）；PixelLM 附录 A（L、N、α、λ_ref、λ_dice 数值与 Alg.1 被 ar5iv 截断）；GILL(arXiv:2306.00008)；三篇学习式图像压缩（2406.13709 / 2306.17460 / 2401.17246，只看到标题）；GLARE 官方仓库；OpenVeraTeam/VeraRetouch 仓库；majumderb/rezero。
22. **论文↔代码不一致（已记录，未向作者确认）**：AdaInt 论文 Eq.(6) 把 0.0001 标为 smoothness 权重 L_s，官方 config `smooth_factor=0`（TV 关闭）、0.0001 挂在 `sparse_factor`；OpenVLA-OFT 论文写「4 层 ReLU MLP」、代码是 `MLPResNet(num_blocks=2)`；LISA 论文写通道 `[256, 4096, 4096]`、代码走向 4096→4096→256；Colorful Colorization 论文写 5 近邻、代码 `NN=10`。
23. **无代码交叉验证的数值**：4D LUT（无官方实现，αs=0.0001 / αm=10 仅来自正文；扫描区间原文排版为 `{0, e−5, e−4, e−3, e−2, e−1}`，疑为 1e−5 系列，未能从原文确认）；Neural Preset（官方仓库只有 `src/metric/`，λ=10 与 Lrec/Lcon 仅论文来源）；AceTone / StatLUT / InstantRetouch / RAG 颜色复原 / LumiVideo 等 2026 年新预印本的同行评审状态与代码状态未核实（AceTone 有官方仓库与 CVPR 2026 标注）。
24. **Mask2Former 公式原文**：Eq.(2)(5) 的 LaTeX 被 ar5iv 剥离（`M_{l-1}(x,y) = 0 / −inf` 的显式写法与二值化阈值数值），0.5 阈值与 True=禁止的语义只从官方代码取到；如需论文侧逐字公式须另取 CVPR 2022 PDF。
25. **Sharma 2005 符号丢失**：PDF 提取把 Δ / ΔE*ab 等符号丢成空白（如 "within 5 CIELAB E* ab units" 实为 "5 CIELAB ΔE*ab units"）；公式编号与数字（0.2734 / 0.0119 / Eq.11 / Eq.20）与原 PDF 一致，写进正式文档前需再用 PDF 阅读器核对符号。
26. **DualBLN 仓库归属**：`120326/DualBLN` 的 README 全文未出现 "DualBLN"/ACCV/论文引用字样；仓库名与 ACCV 2022 论文的对应关系未从仓库内独立证据核实（代码事实本身是直接读文件得到的）。
27. **Splatter Image 第二阶段 LPIPS 微调的 λ_lpips 数值**取不到（`configs/experiment/lpips_*.yaml` 抓取失败），目前只核到 `default_config.yaml` 的 `lambda_lpips: 0.0` 与代码里的凸组合。
28. **GLARE 消融 Table 5 的具体数值行**未被完整取到（I-LNF vs Transformer 直接预测 code index 的差值未核实）。
29. **InstantRetouch 纯 L1 的 PSNR 绝对值**正文未给出，因此「20.84 / 30.64 / 31.66」这三个辅助项数字的对比差值无法从原文核实。
30. **Deep3DBox 的 w 与 α 数值**论文未给出（只给了 bin 数消融）。

---

## 6. 来源清单

### 6.1 已打开（按专题分组）

#### 6.1.1 LUT / 颜色变换的 loss 配方

- https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/image_adaptive_lut_train_paired.py
- https://raw.githubusercontent.com/HuiZeng/Image-Adaptive-3DLUT/master/models.py
- https://arxiv.org/pdf/2009.14468
- https://raw.githubusercontent.com/Xian-Bei/CLUT/main/utils/losses.py
- https://raw.githubusercontent.com/Xian-Bei/CLUT/main/parameters.py
- https://raw.githubusercontent.com/Xian-Bei/CLUT/main/train.py
- https://raw.githubusercontent.com/Xian-Bei/CLUT/main/models.py
- https://raw.githubusercontent.com/Xian-Bei/CLUT/main/README.md
- https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/model.py
- https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/configs/fivekrgb.py
- https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/configs/fivekxyz.py
- https://raw.githubusercontent.com/ImCharlesY/AdaInt/main/adaint/configs/ppr10k.py
- https://arxiv.org/pdf/2204.13983
- https://raw.githubusercontent.com/ImCharlesY/SepLUT/main/seplut/model.py
- https://raw.githubusercontent.com/ImCharlesY/SepLUT/main/seplut/configs/fivekrgb.py
- https://raw.githubusercontent.com/ImCharlesY/SepLUT/main/seplut/modules/lut.py
- https://arxiv.org/pdf/2207.08351
- https://arxiv.org/pdf/2209.01749
- https://arxiv.org/abs/2306.11920 / https://arxiv.org/pdf/2306.11920v3 / https://ar5iv.labs.arxiv.org/html/2306.11920
- https://raw.githubusercontent.com/mv-lab/nilut/main/fit.py
- https://raw.githubusercontent.com/mv-lab/nilut/main/utils.py
- https://raw.githubusercontent.com/mv-lab/nilut/main/nilut.ipynb
- https://raw.githubusercontent.com/mv-lab/nilut/main/nilut-multiblend.ipynb
- https://arxiv.org/pdf/2303.13511v2
- https://raw.githubusercontent.com/ZHKKKe/NeuralPreset/main/README.md
- https://arxiv.org/pdf/2311.03943
- https://arxiv.org/pdf/2604.00530 / https://arxiv.org/html/2604.00530v1
- https://arxiv.org/pdf/2607.08227
- https://arxiv.org/pdf/2509.23608
- https://arxiv.org/pdf/2105.09180
- https://arxiv.org/pdf/2207.05430v2
- https://arxiv.org/pdf/2407.09892
- https://raw.githubusercontent.com/120326/DualBLN/master/code/train_lut_bilinear_pooling_effres.py
- https://raw.githubusercontent.com/120326/DualBLN/master/README.md
- arXiv API 存在性核实：`all:"Text2LUT"` / `all:"CLIP-LUT"` / `all:"InstructColor"` / `abs:"3D LUT" AND abs:language`
- https://api.github.com/repos/SSRHeart/TSFlow
- https://api.github.com/search/repositories?q=CLUT-Net+in:name,description,readme

#### 6.1.2 可微色差与多 loss 自动平衡

- https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/
- https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/ciede2000noteCRNA.pdf
- https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/deltaE2000.m
- https://arxiv.org/abs/1911.13175 / https://ar5iv.labs.arxiv.org/html/1911.13175
- https://github.com/sjmoran/CURL
- https://raw.githubusercontent.com/sjmoran/curl-image-enhancement/master/model.py
- https://ar5iv.labs.arxiv.org/html/1603.08511
- https://arxiv.org/abs/1711.02257 / https://ar5iv.labs.arxiv.org/html/1711.02257 / http://proceedings.mlr.press/v80/chen18a/chen18a.pdf
- https://ar5iv.labs.arxiv.org/html/1705.07115
- https://raw.githubusercontent.com/lorenmt/mtan/master/im2im_pred/utils.py
- https://raw.githubusercontent.com/lorenmt/mtan/master/im2im_pred/model_segnet_mtan.py
- https://arxiv.org/abs/2009.01717 / https://ar5iv.labs.arxiv.org/html/2009.01717
- https://raw.githubusercontent.com/rickgroen/cov-weighting/main/losses/covweighting_loss.py
- https://arxiv.org/abs/1903.06733 / https://arxiv.org/pdf/1903.06733v3
- https://ar5iv.labs.arxiv.org/html/1606.08415
- https://ar5iv.labs.arxiv.org/html/2003.04887 / https://arxiv.org/pdf/2003.04887
- https://api.github.com/repos/serkansulun/differentiable-color-loss（404 核实）
- https://api.github.com/search/repositories?q=ciede2000+language:python
- http://export.arxiv.org/api/query?search_query=abs:%22CIEDE2000%22&max_results=40

#### 6.1.3 一对多与均值坍塌

- https://ar5iv.labs.arxiv.org/html/1603.08511 / https://arxiv.org/pdf/1603.08511v5
- https://raw.githubusercontent.com/richzhang/colorization/caffe/resources/caffe_traininglayers.py
- https://ar5iv.labs.arxiv.org/html/1611.07004
- https://raw.githubusercontent.com/junyanz/pytorch-CycleGAN-and-pix2pix/master/models/pix2pix_model.py
- https://ar5iv.labs.arxiv.org/html/1609.04802 / https://arxiv.org/pdf/1609.04802v5
- https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf
- https://ar5iv.labs.arxiv.org/html/2007.15651
- https://raw.githubusercontent.com/taesungp/contrastive-unpaired-translation/master/models/patchnce.py
- https://raw.githubusercontent.com/taesungp/contrastive-unpaired-translation/master/models/cut_model.py
- https://arxiv.org/html/2407.03757v1（+ arXiv API `ti:"DiffRetouch"` 核实）
- https://ar5iv.labs.arxiv.org/html/2303.11435
- https://arxiv.org/abs/2403.03950 / https://arxiv.org/html/2403.03950v1
- https://ar5iv.labs.arxiv.org/html/2210.02747 / https://arxiv.org/pdf/2210.02747v2
- https://arxiv.org/html/2409.05250v1
- http://export.arxiv.org/api/query?search_query=all:%22regression%20to%20the%20mean%22%20AND%20cat:cs.CV...
- http://export.arxiv.org/api/query?search_query=abs:%22lookup+table%22+AND+abs:%22text%22+AND+cat:cs.CV...

#### 6.1.4 检索式与 classify-then-refine

- https://arxiv.org/abs/1711.00937 / https://arxiv.org/pdf/1711.00937v2 / https://ar5iv.labs.arxiv.org/html/1711.00937
- https://arxiv.org/abs/2206.11253 / https://ar5iv.labs.arxiv.org/html/2206.11253
- https://raw.githubusercontent.com/sczhou/CodeFormer/master/options/CodeFormer_stage2.yml
- https://raw.githubusercontent.com/sczhou/CodeFormer/master/options/VQGAN_512_ds32_nearest_stage1.yml
- https://raw.githubusercontent.com/sczhou/CodeFormer/master/basicsr/models/codeformer_idx_model.py
- https://github.com/martian422/AceTone
- https://raw.githubusercontent.com/martian422/AceTone/open-source-ready/train_vq.py
- https://arxiv.org/pdf/1612.00496v2
- https://arxiv.org/html/2407.12431v1
- https://arxiv.org/html/2602.17044v1
- https://arxiv.org/html/2608.08211v1
- https://ar5iv.labs.arxiv.org/html/2107.12898
- https://ar5iv.labs.arxiv.org/html/2007.10701
- arXiv API 检索：`abs:"3D LUT" AND (text OR language OR prompt)`、`abs:"preset" AND (retouch OR color style)`、`(retrieval-augmented OR retrieve) AND (color transfer OR image enhancement OR retouching OR color grading)`、`ti:"StarEnhancer"`

#### 6.1.5 query-based 解码头出处链

- https://arxiv.org/abs/2005.12872 / https://ar5iv.labs.arxiv.org/html/2005.12872
- https://raw.githubusercontent.com/facebookresearch/detr/main/models/transformer.py
- https://raw.githubusercontent.com/facebookresearch/detr/main/main.py
- https://arxiv.org/abs/2112.01527 / https://ar5iv.labs.arxiv.org/html/2112.01527
- https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/modeling/transformer_decoder/mask2former_transformer_decoder.py
- https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/mask2former/config.py
- https://raw.githubusercontent.com/facebookresearch/Mask2Former/main/configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml
- https://arxiv.org/abs/2504.06256 / https://arxiv.org/html/2504.06256v1
- https://raw.githubusercontent.com/facebookresearch/metaquery/main/models/model.py
- https://raw.githubusercontent.com/facebookresearch/metaquery/main/configs/qwen2p5vl3b_sana.yaml
- https://api.github.com/repos/facebookresearch/metaquery（存在）/ https://api.github.com/repos/xichenpan/MetaQueries（404）
- https://arxiv.org/abs/2308.00692 / https://ar5iv.labs.arxiv.org/html/2308.00692
- https://raw.githubusercontent.com/dvlab-research/LISA/main/model/LISA.py
- https://arxiv.org/abs/2312.02228 / https://ar5iv.labs.arxiv.org/html/2312.02228

#### 6.1.6 LLM hidden → 连续参数

- https://arxiv.org/abs/2410.24164 / https://arxiv.org/html/2410.24164v4
- https://raw.githubusercontent.com/Physical-Intelligence/openpi/main/src/openpi/models/pi0.py
- https://arxiv.org/html/2405.12213v2
- https://raw.githubusercontent.com/octo-models/octo/main/octo/model/components/action_heads.py
- https://arxiv.org/html/2406.09246v3
- https://ar5iv.labs.arxiv.org/html/2307.15818
- https://arxiv.org/html/2502.19645v1
- https://raw.githubusercontent.com/moojink/openvla-oft/main/prismatic/models/action_heads.py
- https://arxiv.org/html/2411.19650v1
- https://arxiv.org/html/2309.17102v2
- https://arxiv.org/html/2401.16468v2
- https://raw.githubusercontent.com/mv-lab/InstructIR/main/models/instructir.py
- https://raw.githubusercontent.com/va1shn9v/PromptIR/main/net/model.py
- https://arxiv.org/html/2211.09800v2
- https://arxiv.org/abs/2604.27375 / https://arxiv.org/html/2604.27375v2
- arXiv 全站检索（返回 no results）：https://arxiv.org/search/?searchtype=all&query=Text2LUT 、`language-guided 3D LUT image retouching`、`"look-up table" text-guided color`、`instruction photo retouching parameters`

#### 6.1.7 N 基元 ↔ N query 的前馈生成器

- https://arxiv.org/abs/2312.13150 / https://ar5iv.labs.arxiv.org/html/2312.13150
- https://raw.githubusercontent.com/szymanowiczs/splatter-image/main/scene/gaussian_predictor.py
- https://raw.githubusercontent.com/szymanowiczs/splatter-image/main/train_network.py
- https://raw.githubusercontent.com/szymanowiczs/splatter-image/main/configs/default_config.yaml
- https://raw.githubusercontent.com/szymanowiczs/splatter-image/main/README.md
- https://arxiv.org/abs/2404.19702 / https://arxiv.org/html/2404.19702v1
- https://arxiv.org/html/2402.05054v1
- https://raw.githubusercontent.com/3DTopia/LGM/main/core/models.py
- https://raw.githubusercontent.com/3DTopia/LGM/main/core/gs.py
- https://arxiv.org/html/2312.09147v1
- https://arxiv.org/html/2403.12957v2
- https://arxiv.org/html/2403.19655v3
- https://arxiv.org/html/2403.18795v3
- https://arxiv.org/html/2401.04099v1
- arXiv API 核实：`id_list=2405.17894,2403.18795,2411.08033`、`ti:"GVGEN"`、`ti:"Amortized Generative 3D Gaussians"`

#### 6.1.8 条件注入与稳定化

- https://arxiv.org/abs/2212.09748 / https://ar5iv.labs.arxiv.org/html/2212.09748
- https://raw.githubusercontent.com/facebookresearch/DiT/main/models.py
- https://ar5iv.labs.arxiv.org/html/1709.07871
- https://raw.githubusercontent.com/ethanjperez/film/master/vr/models/filmed_net.py
- https://raw.githubusercontent.com/ethanjperez/film/master/vr/models/film_gen.py
- https://ar5iv.labs.arxiv.org/html/2302.05543v1 / https://arxiv.org/html/2302.05543v3
- https://openaccess.thecvf.com/content/ICCV2023/supplemental/Zhang_Adding_Conditional_Control_ICCV_2023_supplemental.pdf
- https://raw.githubusercontent.com/lllyasviel/ControlNet/main/ldm/modules/diffusionmodules/util.py
- https://raw.githubusercontent.com/lllyasviel/ControlNet/main/cldm/cldm.py
- https://arxiv.org/pdf/1609.09106v4
- https://arxiv.org/pdf/2304.07645v2
- https://raw.githubusercontent.com/chrhenning/hypnettorch/master/hypnettorch/hnets/mlp_hnet.py
- https://proceedings.mlr.press/v205/beck23a/beck23a.pdf
- https://arxiv.org/abs/2506.06105 / https://arxiv.org/pdf/2506.06105
- https://raw.githubusercontent.com/SakanaAI/text-to-lora/main/src/hyper_llm_modulator/hyper_modulator.py
- https://arxiv.org/pdf/2106.09685v2
- https://arxiv.org/pdf/2406.08447
- https://arxiv.org/pdf/1706.02677
- https://arxiv.org/pdf/2204.14198
- https://arxiv.org/pdf/2301.07093
- https://arxiv.org/pdf/1903.06733v3

### 6.2 出现过但未打开正文（不得据此引用细节）

| URL / 标识 | 状态 |
|---|---|
| DOI 10.1145/3503161.3547879（CLUT-Net, ACM MM 2022 正文） | 非 arXiv、需订阅，未打开。本报告中 CLUT-Net 的 loss 事实全部来自官方仓库代码 |
| https://sjmoran.github.io/pdfs/CURL_supplementary.pdf | 未打开；ω 的具体数值是否列出未核实 |
| https://openreview.net/forum?id=H1lma24tPB（Chang et al., Principled Weight Initialization for Hypernetworks） | `/pdf` 与 `/attachment` 均返回 challenge 页，API 403。hyperfan-in/out 公式只有 `hypnettorch` 实现代码与 Beck et al. 转述两个二手来源 |
| https://github.com/LowLevelAI/GLARE | 论文给出的仓库，未打开文件 |
| https://github.com/OpenVeraTeam/VeraRetouch | 未打开；α 数值与 Retouch Adaptor 隐层维度未核实 |
| majumderb/rezero | ReZero 论文脚注给出的代码地址，未打开 |
| ZHKKKe/NeuralPreset | 仅打开 README；仓库只发布 `src/metric/*`，无训练代码 |
| arXiv:2306.00008（GILL） | 未打开；MGIE 与 MetaQueries 均引用其为 learnable token 设计来源 |
| arXiv:2406.13709 / 2306.17460 / 2401.17246（学习式图像压缩中的 CIEDE2000） | 只看到标题，未打开正文；是否把 ΔE 放进训练 loss 未核实 |
| PixelLM 附录 A | ar5iv 抓取截断，未核实 |
| Mask2Former 论文 Eq.(2)(5) 的 LaTeX | 被 ar5iv 剥离，未取到；如需须另取 CVPR 2022 PDF |
| https://arxiv.org/abs/2307.06949 | 在 conditioning-stability 专题的 citations 中出现，findings 未据其写任何事实 |
| `configs/experiment/lpips_*.yaml`（Splatter Image 第二阶段） | 抓取失败 |
| GLARE Table 5 具体数值行 | 未完整取到 |
