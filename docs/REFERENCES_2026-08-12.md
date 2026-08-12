# VERIFIED 参考文献总表(2026-08-12 合并)

> **本文档是四份调研报告 VERIFIED 参考文献表的合并留档。** 2026-08-12 文档收敛时,四份调研报告
> (`RESEARCH_unified-field-prediction` / `RESEARCH_analytic-edge-quality` / `RESEARCH_ceiling-push` /
> `RESEARCH_geometry-extraction-arch`)的**结论**已被 `EXPERIMENT_INDEX` 的 EPR-H 系 RETRO 与
> `PROPOSAL_geometry-injection` 吸收,报告正文移入 `trash/docs/`(禁读);但其**参考文献表**
> 当时未被任何保留文档引用,故按清理规则抽出合并于此,**唯一存续副本**。
>
> **性质**:文献底账,不是方案、不是判据、不是现行结论。**任何设计主张不得引用本文档为依据**
> ——设计依据一律走 `docs/EXPERIMENT_INDEX.md`(实验结论)与 `docs/PROPOSAL_geometry-injection_2026-08-11.md`
> (在途 EPR-001~003 判据出处)。本文档只回答「某篇文献的出处/链接是什么」。
>
> **可信性**:各表条目在原调研阶段均标注 VERIFIED(原始来源已打开核实)。本项目检索引擎有编造
> 前科(假仓库、假 arXiv 号),各表末尾的「备注」记录了已剔除的编造条目,**予以保留**。
> 转引前仍建议按 CLAUDE.md 派工协议第 2 条重新打开原始来源。
>
> **规模**:四表合计 207 条唯一 arXiv 链接(另含 openaccess / 官方博客 / 项目页若干)。
>
> **另**:`PROPOSAL_geometry-injection_2026-08-11.md` §7 自带**代码证据索引**(亲读仓库文件路径)
> 与该提案实际消费的 12 篇论文清单,二者不重复——那一份是「本轮在用的」,本文档是「调研全量底账」。

---

## A. 统一场预测框架(源:RESEARCH_unified-field-prediction_2026-08-10 §7)

> 原文档已移入 trash;三案 Gate 0 结论见 EXPERIMENT_INDEX EPR-H17/H18/H19/H20。

以下全部条目在调研阶段经原始来源核实(VERIFIED);按调研方向分组,重复条目只列一次并标注复用方向。

### 7.1 field-generative(场空间条件生成)

| 名称 | 出处 | arXiv | 在本文档中的角色 |
|---|---|---|---|
| Pix2Seq-D | ICCV 2023 | [2210.06366](https://arxiv.org/abs/2210.06366) | 一对多映射交给条件分布多模式而非架构分支(B §3.1) |
| Ambiguous-MedSeg | CVPR 2023 | [2304.04745](https://arxiv.org/abs/2304.04745) | 似然目标不塌向平均掩膜、复现模式频率(A/B 约束 5) |
| Marigold | CVPR 2024 | [2312.02145](https://arxiv.org/abs/2312.02145) | 74k 小数据可训先例;冻结解码器须先验证重建上界(B Gate 0) |
| E2E-FT | WACV 2025 Oral | [2409.11355](https://arxiv.org/abs/2409.11355) | 单步确定化只在近单峰任务成立的警告(A §2.8 / B 约束 5) |
| DepthFM | AAAI 2025 | [2403.13788](https://arxiv.org/abs/2403.13788) | flow matching 直线传输少步可行;采样方差=免费不确定性(B) |
| Lotus | ICLR-era 2024→2025 | [2409.18124](https://arxiv.org/abs/2409.18124) | x0-参数化优于噪声预测(B §3.1) |
| Diffusion in Infinite Dimensions | AISTATS 2023 | [2212.00886](https://arxiv.org/abs/2212.00886) | 函数空间似然视角(背景) |
| HyperDiffusion | ICCV 2023 | [2303.17015](https://arxiv.org/abs/2303.17015) | 多盆地是未对齐参数化的产物;共享初始化对齐(A C_ε / B 约束 2)(与 inr-hyper 复用) |
| Shap-E | OpenAI 2023 | [2305.02463](https://arxiv.org/abs/2305.02463) | 规范潜码的两阶段模板;B 论证其 encoder 可被闭式投影替代 |
| InstructDiffusion | CVPR 2024 | [2309.03895](https://arxiv.org/abs/2309.03895) | 指令 cross-attention 免路由注入通道(B 约束 6) |
| Shortcut Models | ICLR 2025 | [2410.12557](https://arxiv.org/abs/2410.12557) | 少步方案储备(未进主方案) |
| DiffEdit | ICLR 2023(奠基) | [2210.11427](https://arxiv.org/abs/2210.11427) | 零训练双 prompt 差分,第四基线(A/B 基线排) |

### 7.2 hier-latent(离散+连续潜变量)

| 名称 | 出处 | arXiv | 角色 |
|---|---|---|---|
| FSQ | ICLR 2024 | [2309.15505](https://arxiv.org/abs/2309.15505) | 结构性消除码本塌缩(C 后备路线背景) |
| ReinMax | NeurIPS 2023 Oral | [2304.08612](https://arxiv.org/abs/2304.08612) | 二阶直通估计;C 修订后不再需要(记录) |
| Rotation Trick | 2024 | [2410.06424](https://arxiv.org/abs/2410.06424) | VQ 梯度修复(C 的 HiMTok 后备背景) |
| DLT | ICCV 2023 | [2303.03755](https://arxiv.org/abs/2303.03755) | 离散-连续联合扩散同构体(A 背景) |
| LayoutDM | CVPR 2023 | [2303.08137](https://arxiv.org/abs/2303.08137) | 推理期 logit 约束注入(备选机制) |
| MAR / Diffusion Loss | NeurIPS 2024 Spotlight | [2406.11838](https://arxiv.org/abs/2406.11838) | 连续 token 逐 token 分布建模;采样纪律(C 推理)(与 program-token 复用) |
| GIVT | ECCV 2024 | [2312.02116](https://arxiv.org/abs/2312.02116) | GMM 头=离散选择涌现;mode-seeking 纪律(C)(与 program-token 复用) |
| AiT | ICCV 2023 | [2301.02229](https://arxiv.org/abs/2301.02229) | token 瓶颈承载场输出;暴露偏差已知问题(C 风险 6) |
| HNC-CAD | ICML 2023 | [2307.00149](https://arxiv.org/abs/2307.00149) | 层次码事后语义对齐验证方法论(A §2.6) |
| Bayesian Flow Networks | 2023 | [2308.07037](https://arxiv.org/abs/2308.07037) | 离散+连续统一损失(理论储备) |
| Neural Path Representation | SIGGRAPH 2024 | [2405.10317](https://arxiv.org/abs/2405.10317) | 轮廓作为「参数化家族之一」的思想(A §2.6) |

### 7.3 sbi-posterior(摊销后验估计)

| 名称 | 出处 | arXiv | 角色 |
|---|---|---|---|
| FMPE | NeurIPS 2023 | [2305.17161](https://arxiv.org/abs/2305.17161) | proper scoring 的条件分布估计,A 的连续层似然;密度精确评估(A 推理) |
| Simformer | ICML 2024 | [2404.09636](https://arxiv.org/abs/2404.09636) | 单 transformer 单 diffusion 目标的 token 化组织(A §2.1) |
| CMPE | NeurIPS 2024 | [2312.05440](https://arxiv.org/abs/2312.05440) | 低维后验 1–2 步采样(A 延伸路线) |
| Hierarchical NSBI | TMLR 2024 | [2306.12584](https://arxiv.org/abs/2306.12584) | q(z|x)·q(w|z,x) 单一 NLL 分解(A §2.1);稀有家族欠拟合失效项 |
| Sourcerer | 2024 | [2402.07808](https://arxiv.org/abs/2402.07808) | 「参数空间分布+输出空间损失」同构先例;最大熵消解不可辨识(A 约束 3) |
| DINGO-BNS | Nature 639, 2025 | [2407.09602](https://arxiv.org/abs/2407.09602) | 生产级 NPE 旗舰;嵌入自动学成后验充分统计量(A 约束 6) |
| NPE-PF | 2025 | [2504.17660](https://arxiv.org/abs/2504.17660) | 零训练 in-context 后验探针(A Step 0) |
| RoPE | 2024/2025 | [2405.08719](https://arxiv.org/abs/2405.08719) | 金标小集的正确消费方式:事后校准(A/B/C 分布移位) |
| sbi reloaded | 2024/2025 | [2411.17337](https://arxiv.org/abs/2411.17337) | NPE 标准工作流与诊断配套(A) |
| TARP | ICML 2023 | [2302.03026](https://arxiv.org/abs/2302.03026) | 覆盖检验充要性;塌缩机械检出(A E3 / B #9) |

### 7.4 program-token(结构化 token 化)

| 名称 | 出处 | arXiv | 角色 |
|---|---|---|---|
| PolyFormer | CVPR 2023 | [2302.07387](https://arxiv.org/abs/2302.07387) | AR 逐步化规避整体回归病态(C 约束 2) |
| HiMTok | ICCV 2025 | [2503.13026](https://arxiv.org/abs/2503.13026) | 层次掩膜 token 后备(C E0 后备) |
| Text4Seg | ICLR 2025 | [2410.09855](https://arxiv.org/abs/2410.09855) | 序列域大掩膜捷径风险(C 文法设计的规避对象) |
| Stop Regressing | ICML 2024 | [2403.03950](https://arxiv.org/abs/2403.03950) | 回归换分类的可扩展性质变;HL-Gauss 软直方图(C 损失) |
| FAST | Physical Intelligence 2025 | [2501.09747](https://arxiv.org/abs/2501.09747) | 先去相关再量化;逐维简单分箱失败实证(C 词表) |
| MeshGPT | CVPR 2024 | [2311.15475](https://arxiv.org/abs/2311.15475) | 小词表结构化解码 vs 高维直出;数万样本量级先例(C) |
| IconShop | SIGGRAPH Asia 2023 | [2304.14400](https://arxiv.org/abs/2304.14400) | 命令 token 决定槽位;唯一可解码序列(C 文法) |
| StrokeNUWA | 2024 | [2401.17093](https://arxiv.org/abs/2401.17093) | 语义块级词表粒度证据(C) |
| LLM4SVG | CVPR 2025 | [2412.11102](https://arxiv.org/abs/2412.11102) | 数字串 token 化幻觉证据与修复(C 约束 6) |
| VisProg | CVPR 2023 Best Paper | [2211.11559](https://arxiv.org/abs/2211.11559) | 语法约束解码思想(C 文法定位) |

### 7.5 inr-hyper(神经场/超网络潜空间)

| 名称 | 出处 | arXiv | 角色 |
|---|---|---|---|
| mNIF | NeurIPS 2023 | [2310.19464](https://arxiv.org/abs/2310.19464) | 共享基+低维系数联合学习→系数空间生成(A 理论定位) |
| Spatial Functa | 2023 | [2302.03130](https://arxiv.org/abs/2302.03130) | 全局低维 modulation 装不下复杂空间结构的硬负结果(A 头号风险) |
| Equivariant Weight Spaces | ICML 2023 | [2301.12780](https://arxiv.org/abs/2301.12780) | x↦w\* 非单值函数的理论诊断,审稿人级引用(A/B 约束 2) |
| DPF | ICLR 2023 | [2303.00165](https://arxiv.org/abs/2303.00165) | 场空间迭代去噪可行,修正约束 4 过度解读(B 约束 4;B 回退路线) |
| Functional Diffusion | CVPR 2024 | [2311.15435](https://arxiv.org/abs/2311.15435) | 函数空间条件扩散工程成熟(B 约束 4) |
| INFD | CVPR 2024 | [2406.07480](https://arxiv.org/abs/2406.07480) | 「先获得好潜空间」成败关键;两阶段耦合松的教训(B §3.1) |
| GINR-IPC | CVPR 2023 highlight | [2211.13223](https://arxiv.org/abs/2211.13223) | 低秩调制收缩自由度(备选机制) |
| Locality-Aware GINR | NeurIPS 2023 | [2310.05624](https://arxiv.org/abs/2310.05624) | 局部 latent token 配方(A/B 轮廓升级路径) |
| Attention Beats Concatenation | TMLR 2023 | [2209.10684](https://arxiv.org/abs/2209.10684) | 条件注入机制系统对比:attention 最优(A/B/C 条件编码) |
| CORAL | NeurIPS 2023 | [2306.07266](https://arxiv.org/abs/2306.07266) | 良态潜空间里回归未必死(A E5 改口预案) |
| AROMA | NeurIPS 2024 | [2406.02176](https://arxiv.org/abs/2406.02176) | amortized encoder 潜 token + 扩散式训练在多峰场景的收益(A/B 升级路径) |

### 7.6 frozen-probe(冻结模型探针与场重组)

| 名称 | 出处 | arXiv | 角色 |
|---|---|---|---|
| Talk2DINO | ICCV 2025 | [2411.19331](https://arxiv.org/abs/2411.19331) | 文本嵌入映射进冻结视觉几何(条件注入共识;C 约束 6) |
| CLIP-DINOiser | 2023/2024 | [2312.12359](https://arxiv.org/abs/2312.12359) | 极小参数亲和整理(备选预整理) |
| ProxyCLIP | ECCV 2024 | [2408.04883](https://arxiv.org/abs/2408.04883) | 零参数 proxy attention 平滑相似度场(C §4.5) |
| LPOSS | 2025 | [2503.19777](https://arxiv.org/abs/2503.19777) | 凸标签传播唯一解,εI 同族(C 解释器轮廓支路) |
| CaR | CVPR 2024 | [2312.07661](https://arxiv.org/abs/2312.07661) | 冻结打分器不动点迭代替代逆映射(备选) |
| FeatUp | ICLR 2024 | [2403.10516](https://arxiv.org/abs/2403.10516) | 图像引导上采样的自监督预训练信号(D 上采样支路背景) |
| LoftUp | 2025 | [2504.14032](https://arxiv.org/abs/2504.14032) | 坐标查询式上采样 + SAM 伪 GT(升级储备) |
| Trident | 2024 | [2411.09219](https://arxiv.org/abs/2411.09219) | 场→提示→精化桥接组织方式(参考,不照搬) |
| LaVG | ECCV 2024 | [2408.04961](https://arxiv.org/abs/2408.04961) | Ncut 段支撑上的离散选择;where⊥which 分解(C 轮廓家族) |
| F-LMM | 2024 | [2406.05821](https://arxiv.org/abs/2406.05821) | attention 阴性结论的模型特异性限定;多源弱信号+小翻译头结构(背景) |
| PinPoint | 2026 预印本 | [2605.26689](https://arxiv.org/abs/2605.26689) | 瓶颈在粗信号→掩膜接口而非 grounding 的同构诊断(背景) |
| Early Semantic Grounding in IIE | 2026 预印本 | [2605.13122](https://arxiv.org/abs/2605.13122) | 指令编辑模型早期表征含 where 信号,候选第四特征源(C 约束 6;全案储备) |

---

## B. 解析族边缘质量(源:RESEARCH_analytic-edge-quality_2026-08-11 §六)

> 原文档已移入 trash;脏边机理与门控修复结论见 EXPERIMENT_INDEX EPR-H13/H14,结构损失包裁决见 EPR-H23。

(六路调研均标注 VERIFIED、原始来源已核实;重复出现的文献只列一次,「路线」列标注全部出现处。缩写:SC=shape-constraints,FC=freq-control,AG=adaptive-guidance,MP=manifold-projection,CO=conditional-operator,SS=structural-supervision)

### 6.1 形状/几何约束
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| Star Shape Prior in FCN for Skin Lesion Segmentation | MICCAI 2018 | https://arxiv.org/abs/1806.08437 | SC |
| Learning Euler's Elastica Model for Medical Image Segmentation | arXiv 2011.00526 | https://arxiv.org/abs/2011.00526 | SC, SS |
| Learning Active Contour Models for Medical Image Segmentation | CVPR 2019 | https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Learning_Active_Contour_Models_for_Medical_Image_Segmentation_CVPR_2019_paper.html | SC |
| On Regularized Losses for Weakly-supervised CNN Segmentation | ECCV 2018 | https://arxiv.org/abs/1803.09569 | SC |
| Implicit Geometric Regularization for Learning Shapes (IGR) | ICML 2020 | https://arxiv.org/abs/2002.10099 | SC, SS |
| Input Convex Neural Networks (ICNN/PICNN) | ICML 2017 | https://arxiv.org/abs/1609.07152 | SC |
| Constrained Monotonic Neural Networks | ICML 2023 | https://arxiv.org/abs/2205.11775 | SC |
| Lagrangian Optimization via Log-Barrier Extensions | EUSIPCO 2022 | https://arxiv.org/abs/1904.04205 | SC |
| clDice — Topology-Preserving Loss for Tubular Segmentation | CVPR 2021 | https://arxiv.org/abs/2003.07311 | SC |
| Convex Shape Prior for DCNN-based Eye Fundus Segmentation | arXiv 2005.07476 | https://arxiv.org/abs/2005.07476 | SC |
| Anatomically Constrained Neural Networks (ACNN) | IEEE TMI 2017 | https://arxiv.org/abs/1705.08302 | SC, MP |

### 6.2 频率控制
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| BACON: Band-limited Coordinate Networks | CVPR 2022 | https://arxiv.org/abs/2112.04645 | FC |
| BANF: Band-limited Neural Fields | CVPR 2024 | https://theialab.github.io/banf | FC |
| LPTN: Laplacian Pyramid Translation Network | CVPR 2021 | https://github.com/csjliang/LPTN | FC |
| FreeU: Free Lunch in Diffusion U-Net | CVPR 2024 | https://huggingface.co/papers/2309.11497 | FC |
| SAPE: Spatially-Adaptive Progressive Encoding | NeurIPS 2021 | https://huggingface.co/papers/2104.09125 | FC |
| Mip-NeRF: Multiscale Anti-Aliasing NeRF | ICCV 2021 | https://jonbarron.info/mipnerf/ | FC |
| Neural Fourier Filter Bank | CVPR 2023 | https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Neural_Fourier_Filter_Bank_CVPR_2023_paper.html | FC |
| FreeNeRF: Free Frequency Regularization | CVPR 2023 | https://openaccess.thecvf.com/content/CVPR2023/html/Yang_FreeNeRF_Improving_Few-Shot_Neural_Rendering_With_Free_Frequency_Regularization_CVPR_2023_paper.html | FC |
| Fourier Features Let Networks Learn High Frequency Functions | NeurIPS 2020 | https://bmild.github.io/fourfeat/ | FC |
| On the Spectral Bias of Neural Networks | ICML 2019 | https://proceedings.mlr.press/v97/rahaman19a.html | FC |
| Alias-Free GAN (StyleGAN3) | NeurIPS 2021 | https://arxiv.org/abs/2106.12423 | FC |

### 6.3 引导自适应
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| Joint Bilateral Upsampling (Kopf et al.) | SIGGRAPH 2007 | https://johanneskopf.de/publications/jbu/ | AG |
| Guided Depth Map Super-resolution: A Survey | ACM CSUR 2023 | https://arxiv.org/abs/2302.09598 | AG |
| Mutual-Structure for Joint Filtering | ICCV 2015 | http://www.cse.cuhk.edu.hk/leojia/projects/mutualstructure/index.html | AG |
| MSF with Embedded Edge Inconsistency Measurement | IEEE TIP 2018 | https://pubmed.ncbi.nlm.nih.gov/29993634/ | AG |
| Joint Image Filtering with Deep Convolutional Networks (DJF/DJFR) | TPAMI | https://arxiv.org/abs/1710.04200 | AG |
| SVLRM: Spatially Variant Linear Representation Models | CVPR 2019 / TPAMI 2022 | https://pubmed.ncbi.nlm.nih.gov/34357863/ | AG |
| Unsharp Mask Guided Filtering (UMGF) | IEEE TIP 2021 | https://arxiv.org/abs/2106.01428 | AG |
| Fast End-to-End Trainable Guided Filter (DGF) | CVPR 2018 | https://arxiv.org/abs/1803.05619 | AG, CO |
| Pixel-Adaptive Convolutional Neural Networks (PAC) | CVPR 2019 | https://arxiv.org/abs/1904.05373 | AG, CO |
| Deformable Kernel Networks (DKN) | IJCV 2021 | https://arxiv.org/abs/1910.08373 | AG |
| The Fast Bilateral Solver (FBS) | ECCV 2016 | https://arxiv.org/abs/1511.03296 | AG |

### 6.4 流形投影
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| DC3: Learning Method for Optimization with Hard Constraints | ICLR 2021 | https://arxiv.org/abs/2104.12225 | MP |
| HardNet: Hard-Constrained NN with Universal Approximation | arXiv 2410.10807 | https://arxiv.org/abs/2410.10807 | MP |
| OptNet: Differentiable Optimization as a Layer | ICML 2017 | https://arxiv.org/abs/1703.00443 | MP |
| Differentiable Convex Optimization Layers (cvxpylayers) | NeurIPS 2019 | https://arxiv.org/abs/1910.12430 | MP |
| Deep Declarative Networks (DDN) | TPAMI 2022 | https://arxiv.org/abs/1909.04866 | MP |
| Fast Differentiable Sorting and Ranking (PAV/isotonic) | ICML 2020 | https://arxiv.org/abs/2002.08871 | MP |
| DSAC — Differentiable RANSAC | CVPR 2017 | https://arxiv.org/abs/1611.05705 | MP |
| Deep Hough Transform for Semantic Line Detection | TPAMI 2021 | https://arxiv.org/abs/2003.04676 | MP |
| Post-DAE: Anatomical Priors via Denoising Autoencoders | MICCAI 2019 | https://arxiv.org/abs/1906.02343 | MP |
| Enforcing Analytic Constraints in NN Emulating Physical Systems | PRL 126, 098302 (2021) | https://arxiv.org/abs/1909.00912 | MP |
| RAYEN: Imposition of Hard Convex Constraints | arXiv 2307.08336 | https://arxiv.org/abs/2307.08336 | MP |

### 6.5 条件化算子
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| FADE: Fusing Assets of Decoder and Encoder for Upsampling | ECCV 2022 | https://arxiv.org/abs/2207.10392 | CO |
| DySample: Learning to Upsample by Learning to Sample | ICCV 2023 | https://arxiv.org/abs/2308.15085 | CO |
| CARAFE: Content-Aware ReAssembly of FEatures | ICCV 2019 | https://arxiv.org/abs/1905.02188 | CO |
| FiLM: Visual Reasoning with a General Conditioning Layer | AAAI 2018 | https://arxiv.org/abs/1709.07871 | CO |
| CondConv: Conditionally Parameterized Convolutions | NeurIPS 2019 | https://arxiv.org/abs/1904.04971 | CO |
| Dynamic Convolution: Attention over Convolution Kernels | CVPR 2020 | https://arxiv.org/abs/1912.03458 | CO |
| ASTMT: Attentive Single-Tasking of Multiple Tasks | CVPR 2019 | https://arxiv.org/abs/1904.08918 | CO |
| Mod-Squad: MoE as Modular Multi-Task Learners | arXiv 2212.08066 | https://arxiv.org/abs/2212.08066 | CO |

### 6.6 结构化监督
| 文献 | 出处 | URL | 路线 |
|---|---|---|---|
| SIREN: Implicit Neural Representations with Periodic Activations | NeurIPS 2020 | https://arxiv.org/abs/2006.09661 | SS |
| MiDaS: Robust Monocular Depth Estimation(多尺度梯度匹配) | TPAMI 2020 | https://arxiv.org/abs/1907.01341 | SS |
| BGMv2: Real-Time High-Resolution Background Matting | CVPR 2021 | https://arxiv.org/abs/2012.07810 | SS |
| VNL: Virtual Normal for Depth Prediction | ICCV 2019 | https://arxiv.org/abs/1907.12209 | SS |
| Focal Frequency Loss (FFL) | ICCV 2021 | https://arxiv.org/abs/2012.12821 | SS |
| Boundary Loss for Highly Unbalanced Segmentation | MIDL 2019 / MedIA 2021 | https://arxiv.org/abs/1812.07032 | SS |
| Shape-Aware Segmentation by Predicting SDM | AAAI 2020 | https://arxiv.org/abs/1912.03849 | SS |
| How to Incorporate Monotonicity in Deep Networks | NeurIPS 2019 WS | https://arxiv.org/abs/1909.10662 | SS |
| Unifying Distillation and Privileged Information | ICLR 2016 | https://arxiv.org/abs/1511.03643 | SS |
| Structured Knowledge Distillation for Dense Prediction | CVPR 2019 / TPAMI | https://arxiv.org/abs/1903.04197 | SS |

> 备注:structural-supervision 路线曾核实剔除一条编造引文(arXiv:2003.04625 实为量子物理论文,检索引擎为 elastica 主题编造)——与项目「检索引擎有编造前科」纪律一致,本表仅收已核实条目。

---

## C. 冲击拟合上界(源:RESEARCH_ceiling-push_2026-08-11 §六)

> 原文档已移入 trash;D0 板与 P0 探针结论见 EXPERIMENT_INDEX EPR-H26(引用须连带 S15(a) / 裁决 K7)。

### (ii) predictability-ceiling
| 文献 | 出处 | 链接 |
|---|---|---|
| Direct Bayes Error Estimation (Ishida et al.) | ICLR 2023 notable-top-5% | https://arxiv.org/abs/2202.00395 |
| Label Convergence (Tschirschwitz & Rodehorst) | WACV 2025 | https://arxiv.org/abs/2409.09412 |
| GDE: Assessing Generalization via Disagreement (Jiang et al.) | ICLR 2022 | https://arxiv.org/abs/2106.13799 |
| Chinchilla (Hoffmann et al.) | 2022, 机制锚点 | https://arxiv.org/abs/2203.15556 |
| Observational Scaling Laws (Ruan et al.) | NeurIPS 2024 | https://arxiv.org/abs/2405.10938 |
| Tyche: Stochastic In-Context Segmentation (Rakic et al.) | CVPR 2024 | https://arxiv.org/abs/2401.13650 |
| β-NLL (Seitzer et al.) | ICLR 2022 | https://arxiv.org/abs/2203.09168 |
| Benchmarking Uncertainty Disentanglement (Mucsányi et al.) | NeurIPS 2024 | https://arxiv.org/abs/2402.19460 |
| Rethinking Aleatoric and Epistemic Uncertainty (Bickford Smith et al.) | ICML 2025 | https://arxiv.org/abs/2412.20892 |
| V-Usable Information / PVI (Ethayarajh et al.) | ICML 2022 | https://arxiv.org/abs/2110.08420 |
| VariErr NLI (Weber-Genzel et al.) | ACL 2024 main | https://arxiv.org/abs/2403.01931 |

### (i) language-spatial
| 文献 | 出处 | 链接 |
|---|---|---|
| ARO / bags-of-words (Yuksekgonul et al.) | ICLR 2023 Oral | https://arxiv.org/abs/2210.01936 |
| What'sUp (Kamath et al.) | EMNLP 2023 | https://arxiv.org/abs/2310.19785 |
| LAVT | CVPR 2022 | https://arxiv.org/abs/2112.02244 |
| EVF-SAM | 2024 | https://arxiv.org/abs/2406.20076 |
| SmartEdit | CVPR 2024 | https://arxiv.org/abs/2312.06739 |
| Ranni | CVPR 2024 | https://arxiv.org/abs/2311.17002 |
| RPG | ICML 2024 | https://arxiv.org/abs/2401.11708 |
| VoxPoser | CoRL 2023 | https://arxiv.org/abs/2307.05973 |
| RoboPoint | CoRL 2024 | https://arxiv.org/abs/2406.10721 |
| RoboSpatial | CVPR 2025 Oral | https://arxiv.org/abs/2411.16537 |
| COMFORT | ICLR 2025 Oral | https://arxiv.org/abs/2410.17385 |
| SPRIGHT | ECCV 2024 | https://arxiv.org/abs/2404.01197 |

### (i) conditioning-arch
| 文献 | 出处 | 链接 |
|---|---|---|
| SD3 / MM-DiT | 2024 | https://arxiv.org/abs/2403.03206 |
| DiT / adaLN-Zero | 2022 | https://ar5iv.labs.arxiv.org/html/2212.09748 |
| IP-Adapter | 2023 | https://ar5iv.labs.arxiv.org/html/2308.06721 |
| OminiControl | 2024 | https://arxiv.org/abs/2411.15098 |
| MagNet: Mask Grounding for RIS | CVPR 2024 | https://arxiv.org/html/2312.12198v2 |
| F-LMM | 2024 | https://arxiv.org/html/2406.05821v3 |
| PSALM | ECCV 2024 | https://arxiv.org/abs/2403.14598 |
| ControlNet | 2023, 机制锚点 | https://arxiv.org/abs/2302.05543 |

### (iv) frozen-features
| 文献 | 出处 | 链接 |
|---|---|---|
| AM-RADIO | CVPR 2024 | https://arxiv.org/abs/2312.06709 |
| Probe3D | CVPR 2024 | https://arxiv.org/abs/2404.08636 |
| DINOv2 (lin.4 协议) | 2023 | https://arxiv.org/abs/2304.07193 |
| DINOv3 | 2025 | https://arxiv.org/abs/2508.10104 |
| FeatUp | ICLR 2024 | https://arxiv.org/abs/2403.10516 |
| LoftUp | 2025 | https://arxiv.org/abs/2504.14032 |
| AnyUp | 2025 | https://arxiv.org/abs/2510.12764 |
| DVT | ECCV 2024 | https://arxiv.org/abs/2401.02957 |
| Registers | ICLR 2024 | https://arxiv.org/abs/2309.16588 |
| NeCo | 2024 | https://arxiv.org/abs/2408.11054 |
| Talk2DINO | 2025 | https://arxiv.org/abs/2411.19331 |
| ProxyCLIP | ECCV 2024 | https://arxiv.org/abs/2408.04883 |

### (iii) tail-learning
| 文献 | 出处 | 链接 |
|---|---|---|
| RHO-LOSS | ICML 2022 | https://arxiv.org/abs/2206.07137 |
| InfoBatch | ICLR 2024 | https://arxiv.org/abs/2303.04947 |
| JTT | ICML 2021 | https://arxiv.org/abs/2107.09044 |
| When Do Curricula Work | ICLR 2021 | https://arxiv.org/abs/2012.03107 |
| Beyond Neural Scaling Laws | NeurIPS 2022 Outstanding | https://arxiv.org/abs/2206.14486 |
| ConR | ICLR 2024 | https://arxiv.org/abs/2309.06651 |
| Balanced MSE | CVPR 2022 | https://arxiv.org/abs/2203.16427 |
| FreeMask | NeurIPS 2023 | https://arxiv.org/abs/2310.15160 |
| Feedback-guided Data Synthesis | 2023/2024 | https://arxiv.org/abs/2310.00158 |
| Learning To Simulate | ICLR 2019, 机制锚点 | https://arxiv.org/abs/1810.02513 |
| MEDOE | 2023 | https://arxiv.org/abs/2308.08213 |
| Frequency-based Matcher for LTSS | TMM 2024 | https://arxiv.org/abs/2406.03917 |

### test-time
| 文献 | 出处 | 链接 |
|---|---|---|
| Marigold | CVPR 2024 | https://arxiv.org/abs/2312.02145 |
| Scaling Diffusion for Perceptual Tasks | 2024 | https://arxiv.org/abs/2411.08034 |
| Inference-Time Scaling beyond Denoising Steps | 2025 | https://arxiv.org/abs/2501.09732 |
| SAM(多头 min-loss + IoU 头 + stability score) | 2023 | https://arxiv.org/abs/2304.02643 |
| EvanySeg | 2024 | https://arxiv.org/abs/2409.14874 |
| Decoding-Format Audit of Consistency Selection | 2026 | https://arxiv.org/abs/2608.01207 |
| Soft Dice Confidence (SDC) | 2024 | https://arxiv.org/abs/2402.10665 |
| Rethinking Uncertainty: Estimation to Decision | 2026 | https://arxiv.org/abs/2604.13262 |
| S³-TTA | 2023 | https://arxiv.org/abs/2310.16783 |
| SAMRefiner | ICLR 2025 | https://arxiv.org/abs/2502.06756 |
| ADE-CoT | 2026 | https://arxiv.org/abs/2603.00141 |
| TTA Aleatoric Uncertainty (Wang et al.) | 2018, 机制锚点 | https://arxiv.org/abs/1807.07356 |

---

## D. 低维几何码提取与注入架构(源:RESEARCH_geometry-extraction-arch_2026-08-11 §6)

> 原文档已移入 trash;该调研的选型结论已落进 `PROPOSAL_geometry-injection_2026-08-11.md`
> (三臂 A1/B1/C 与 PCH 共享注入模块),在途实验 EPR-001~004。

### A 符号解析式
| # | 工作 | 来源 |
|---|---|---|
| 1 | JSONSchemaBench (Guidance+EPFL/MSR) | https://arxiv.org/abs/2501.10868 |
| 2 | XGrammar (MLSys 2025) | https://arxiv.org/abs/2411.15100 |
| 3 | OpenAI Structured Outputs（生产数字） | https://openai.com/index/introducing-structured-outputs-in-the-api/ |
| 4 | StructuredRAG (Weaviate) | https://arxiv.org/abs/2408.11061 |
| 5 | Let Me Speak Freely? (EMNLP 2024 industry) | https://arxiv.org/abs/2408.02442 |
| 6 | Say What You Mean（dottxt 复现反驳） | https://blog.dottxt.co/say-what-you-mean.html |
| 7 | Grammar-Aligned Decoding (NeurIPS 2024) | https://arxiv.org/abs/2405.21047 |
| 8 | Know Your Limits: Abstention Survey (TACL 2024) | https://arxiv.org/abs/2407.18418 |
| 9 | Does Prompt Formatting Have Any Impact? (Microsoft) | https://arxiv.org/abs/2411.10541 |
| 10 | BAML Schema-Aligned Parsing (BoundaryML) | https://www.boundaryml.com/blog/schema-aligned-parsing |
| 11 | NuExtract (NuMind) | https://web.archive.org/web/2024/https://numind.ai/blog/nuextract-a-foundation-model-for-structured-extraction |

### B readout / seg-token
| # | 工作 | 来源 |
|---|---|---|
| 12 | LISA (CVPR 2024) | https://arxiv.org/abs/2308.00692 |
| 13 | PixelLM (CVPR 2024) | https://arxiv.org/abs/2312.02228 |
| 14 | GLaMM (CVPR 2024) | https://arxiv.org/abs/2311.03356 |
| 15 | PSALM (ECCV 2024) | https://arxiv.org/abs/2403.14598 |
| 16 | OMG-LLaVA (NeurIPS 2024) | https://arxiv.org/abs/2406.19389 |
| 17 | Sa2VA (2025) | https://arxiv.org/abs/2501.04001 |
| 18 | READ: How <SEG> Token Works (CVPR 2025) | https://arxiv.org/abs/2412.17741 |
| 19 | F-LMM: Grounding Frozen LMMs | https://arxiv.org/abs/2406.05821 |
| 20 | When Do Prompting/Prefix-Tuning Work? (ICLR 2024, Petrov) | https://arxiv.org/abs/2310.19698 |
| 21 | Universality and Limitations of Prompt Tuning (NeurIPS 2023, 补充边界) | https://arxiv.org/abs/2305.18787 |
| 22 | Vision Transformers Need Registers (ICLR 2024) | https://arxiv.org/abs/2309.16588 |
| 23 | ViTs Don't Need Trained Registers (2025) | https://arxiv.org/abs/2506.08010 |
| 24 | Readout Guidance (CVPR 2024) | https://arxiv.org/abs/2312.02150 |

### C query 桥接
| # | 工作 | 来源 |
|---|---|---|
| 25 | **MetaQueries** (Meta+NYU, 2025) | https://arxiv.org/abs/2504.06256 |
| 26 | OpenUni（开源复现） | https://arxiv.org/abs/2505.23661 |
| 27 | MQT-LLaVA (NeurIPS 2024) | https://arxiv.org/abs/2405.19315 |
| 28 | DeCo（double abstraction 诊断） | https://arxiv.org/abs/2405.20985 |
| 29 | Bifrost-1 (2025-08) | https://arxiv.org/abs/2508.05954 |
| 30 | Honeybee (CVPR 2024) | https://arxiv.org/abs/2312.06742 |

### D 预训练 decoder 对齐
| # | 工作 | 来源 |
|---|---|---|
| 31 | EVF-SAM（冻结消融 82.9/83.3/83.7 vs 21.2） | https://arxiv.org/abs/2406.20076 |
| 32 | SAMWISE (CVPR 2025) | https://arxiv.org/abs/2411.17646 |
| 33 | SAM4MLLM (ECCV 2024) | https://arxiv.org/abs/2409.10542 |
| 34 | Matting Anything (MAM) | https://arxiv.org/abs/2306.05399 |
| 35 | Grounded SAM | https://arxiv.org/abs/2401.14159 |

### E 组装
| # | 工作 | 来源 |
|---|---|---|
| 36 | Segment Anything（三轨接口,源码级核实） | https://arxiv.org/abs/2304.02643 |
| 37 | SurgicalSAM (AAAI 2024) | https://arxiv.org/abs/2308.08746 |
| 38 | SEEM (NeurIPS 2023) | https://arxiv.org/abs/2304.06718 |
| 39 | AnyControl (ECCV 2024;检索引擎初给编造号,已纠正) | https://arxiv.org/abs/2406.18958 |
| 40 | Uni-ControlNet (NeurIPS 2023) | https://arxiv.org/abs/2305.16322 |
| 41 | Compose and Conquer (ICLR 2024) | https://arxiv.org/abs/2401.09048 |
| 42 | Composer | https://arxiv.org/abs/2302.09778 |
| 43 | SPADE (CVPR 2019,背景证据) | https://arxiv.org/abs/1903.07291 |
| 44 | Text4Seg (ICLR 2025) | https://arxiv.org/abs/2410.09855 |
| 45 | MM1 (Apple) | https://arxiv.org/abs/2403.09611 |

> 备注：检索引擎在调研过程中多次编造（假 arXiv 号、零来源条目）,上表全部条目经原始来源打开核实;F-LMM/LISA/PSALM/Sa2VA/EVF-SAM/SAM4MLLM 在多节复用,表中只列一次。

---

*合并留档于 2026-08-12 文档收敛(唯一入口 = `docs/EXPERIMENT_INDEX.md`)。四份源文档正文见 `trash/docs/`(禁读);本表为其参考文献表的唯一存续副本,内容逐字未改,仅补各节来源抬头。*
