# 解析族边缘质量:失败机理诊断、方法谱系与设计处方

**文档编号**:RESEARCH_analytic-edge-quality_2026-08-11
**性质**:纯文献综合 + 设计处方。输入仅为六路文献调研(shape-constraints / freq-control / adaptive-guidance / manifold-projection / conditional-operator / structural-supervision)与缺口审查报告;**未读取任何本地实验产物**。所有引用文献均为 VERIFIED(原始来源已核实),见 §六。
**用途**:作为「解析族边缘脏」问题的方法裁决底账,供主 agent 排期归因实验与修复臂。

---

## 一、问题设定复述

**任务**:条件 dense 场预测。给定图像 + 文本指令 + 冻结视觉特征,输出软场 y ∈ [0,1]^{H×W}。

**输出的两族混合结构**(GT 来自两个生成家族,家族标签**训练期可得**):

| 家族 | 占比 | 性质 |
|---|---|---|
| **解析族** | ~83% | 低维参数渲染的解析场(线性渐变、椭圆衰减、条带)。全局光滑、沿某方向单调、等值线为直线/椭圆等解析曲线、**与图像内容统计独立**(等值线在图像里没有对应边缘) |
| **语义族** | ~17% | 贴合物体轮廓的掩膜,边界与图像边缘强相关 |

**现行模型**:共享轻量 conv 头逐格输出 1/16 粗场 → 图像引导上采样(joint bilateral / guided filter 类**固定算子**)到全分辨率。输出空间监督(BCE + 边界罚 + 面积罚)。

**实测症状**:整体中位 IoU 0.74 vs 零参数基线 0.49,已显著超基线;但**两族边缘质量严重不对称**——语义族边界干净贴合;解析族边缘脏:空间噪声、等值线扭曲毛糙、光滑渐变里被刻入无关图像纹理。深尾:软边解析场上模型只能全局变宽(过覆盖),不能表达干净的空间变化衰减。

**怀疑机理(本文档裁决对象,可多因并存)**:
- (a) 逐格自由输出头无「光滑/单调/解析形状」结构归纳偏置,逐像素监督不罚高频噪声;
- (b) 图像引导上采样对解析族是污染源——引导算子假设「输出边界与图像边缘对齐」,对语义族成立、对解析族恰好相反;
- (c) 逐像素 BCE 对「等值线是否解析」不敏感,边界罚只惩罚位置不惩罚形状光滑性。

**已确立约束(所有方案必须相容)**:
1. **低维参数直接回归判死**(参数坐标条件数 ~10⁶,两次训练塌缩实证)——不许退回「回归渲染参数」;
2. **输出空间监督是既定裁定**(损失在场上);
3. **显式 if-else 双路径渲染被否**——统一框架;家族差异由条件/涌现/软机制表达(家族标签可当训练信号;推理期路由若存在必须是模型内软机制或已证 100% 可靠的廉价判别);
4. **资源**:数万样本、单卡数小时训练、推理低开销;
5. **冻结视觉特征不可微调**。

---

## 二、失败机理诊断与归因实验

### 2.0 「脏边」的量化定义(预注册指标,全文档共用)

现行判据表缺「解析级干净」的量化列(缺口 M6),先定义;全部指标遵守项目红线:**禁用任何 AUC 变体**;空间场常规列(soft-IoU / grid 级边界 F1 / 中心先验基线列)照旧保留;可视化禁逐图 min-max。

| 指标 | 定义 | 说明 |
|---|---|---|
| **E_HF**(高频超额能量) | Σ_{(u,v)∈B_hi}\|F(y)\|² − Σ_{B_hi}\|F(y_GT)\|²,报相对 GT 总能量的比值 | B_hi = 覆盖 GT 解析场整臂 99.5% 频谱能量的最小低频盘之外;**整臂常量标定,禁逐图** |
| **κ̃**(等值线窄带曲率) | \|div(∇y/(\|∇y\|+ε))\| 在窄带 \|∇y\|>τ 内的中位数,报 κ̃(y)/κ̃(y_GT) 比值 | 线性族 GT 的 κ≡0,椭圆族有解析值;τ 整臂标定 |
| **MVR**(单调违反率) | 沿 GT 方向场 d(线性/条带取常向量,椭圆取径向)相邻格对中 ⟨∇y,d⟩<0 的占比 | GT 参数为训练期特权信息,仅诊断/训练用 |
| **AFR**(族最佳拟合残差) | 对解析族最佳拟合成员(网格初始化 + LM 局部精化)的 RMSE;报中位与 P90(深尾) | 拟合器**只做诊断,不进训练**,避免裁定 1 争议 |
| 常规列 | soft-IoU / hard-IoU(匹配 GT 面积 top-k 阈值化)、grid 级边界 F1、中心先验基线列 | 按项目红线不可缺 |

**「脏边总量」**定义为模型输出相对 GT 的 E_HF(记 D_total),归因实验把 D_total 分解到 (a)(b)(c)。

### 2.1 机理 (a):头无结构偏置 + 逐像素监督不罚高频

**文献支撑**(判定:**成立,但有保留**):
- Rahaman et al.(ICML 2019, spectral bias):谱偏置只是**学习速度排序不是终态保证**——低频先学 ≠ 高频不学,长训后高频噪声必然回流;且条件输入越丰富、高频越易学(图像特征耦合进输出的理论侧写)。
- Tancik et al.(Fourier Features, NeurIPS 2020):NTK 视角——无频率结构的头缺「带宽旋钮」,单一全局带宽无法同时最优两族;σ 过高实证拟合出高频噪声,恰是本症状的机制预言。
- StyleGAN3(NeurIPS 2021):非带限算子让高频信息(纹理/栅格坐标)泄进本应光滑的输出——「纹理粘在坐标上」与「纹理被刻进解析渐变」同型。

**审计保留意见**:三条全部出自 freq-control 一路(无跨路交叉),且全部来自 coordinate-MLP / 生成模型设定,对「conv 头 + BCE + 高维冻结条件输入」只有**定性外推效力**。(a) 与 (c) 相互混杂、与 (b) 未解耦——脏边可能大部分是 (b),(a)(c) 只是残余。**必须由 §2.4 归因实验裁决占比。**

### 2.2 机理 (b):图像引导上采样是解析族污染源

**文献支撑**(判定:**裁决成立,全案最强**,跨三路独立交叉):
- **症状有标准名称:texture-copying artifact**。GDSR 综述(ACM CSUR 2023)机理定式化:guided filter 局部线性模型强制 ∇y = a_k·∇G(窗内系数恒定),凡 guide 有而目标无的边**必被复制进输出**;解析族与图像统计独立,是该失效的极端情形——失效是**系统性而非偶发**。
- Kopf JBU 原文(SIGGRAPH 2007)核心假设:「guide 的不连续应保留进上采样解」——对解析族**整体为假**,假设层定位。
- Mutual-Structure(ICCV 2015):「两图可能存在完全不同的边,把所有模式都传给 target 会引入显著误差」——原理级陈述。
- FADE(ECCV 2022)三臂对照:上采样核由 decoder(低分内容)生成 → 赢 region 任务;由 encoder(图像细节)生成 → 赢 detail 任务;**二者不可兼得,须显式融合**——与两族不对称完全同构。
- DySample(ICCV 2023)实测:guided 上采样「边界更锐但内部区域出错」(bIoU 高 / mIoU 低解离)——正是解析族光滑内部被刻纹理的发表级实证。

**修复文献最充分**(四类,见 §三.3/.5):不一致度量门控、逐像素线性系数、单系数结构转移量、可学核/引导图。

### 2.3 机理 (c):逐像素 BCE 对等值线解析性不敏感

**文献支撑**(判定:**修复富余、诊断薄**——诊断主要是调研者推理,不是文献裁决):
- FFL(ICCV 2021):空间域损失有频谱缺口(泛化论断);
- Structured KD(CVPR 2019):逐像素损失缺结构统计(泛化论断);
- Boundary loss(MIDL 2019)动机:区域损失对边界几何不敏感(泛化论断)。

三条**无一直接研究「解析等值线不可分辨性」**。另有底座疑点(缺口 M11):BCE 对软 GT 的梯度权重本身未审——matting 域软场标配是 L1/Charbonnier,没人问底座对不对。**(c) 必须由 DX-4 损失盲区审计给出量化裁决。**

### 2.4 廉价归因实验电池(全部零训练/近零成本,修复臂排期前必须跑完)

| 编号 | 裁决对象 | 操作 | 成本 |
|---|---|---|---|
| DX-1 | (b) 算子单独归因 | GT 解析场→按训练同规则降到 1/16→走现行引导上采样→与 GT 比 | eval 脚本,<1 GPU·h |
| DX-2 | (b) 贡献占比 | eval 期解析族换 bilinear 上采样,其余不动,比换前后 | 同上 |
| DX-3 | (a)+(c) 残余 | 直接量 1/16 粗场干净度(logit 与 y 空间各一遍) | 同上 |
| DX-4 | (c) 损失盲区 | 反事实输出对上量各损失项的分辨力 | CPU 可跑 |
| DX-5 | 软路由地基 | 家族可分性线性探针(S-val 源) | <1 GPU·h |
| DX-6 | 条件算子上界 | oracle 家族标签硬路由的收益上界 | eval 脚本 |

**DX-1(判 b,模型完全不参与)**:GT 粗场走同一引导上采样算子。指标:输出的 E_HF、κ̃ 比值、soft-IoU。
预注册:若 E_HF(GT粗场→引导上采样) ≥ **0.5 × D_total** → (b) 判为主因,上采样槽改造(P1)最高优先;若 ≤ **0.1 × D_total** → **(b) 主因证伪**,重心转 (a)(c)。

**DX-2(判 b,消融换算子)**:换 bilinear 后脏边下降比例 δ_b = 1 − E_HF(bilinear)/E_HF(现行),即 (b) 贡献占比点估计。
预注册:δ_b ≥ **60%** → P1 优先;δ_b ≤ **20%** → P1 降级,结构损失(P2)升首位。注意:换 bilinear 只是不放大、不清洁——粗场本身脏的部分要与 DX-3 联合读数。

**DX-3(判 a+c,粗场直检)**:粗场的高频能量占比、MVR、AFR;对照 = **GT 降采样粗场的同指标(噪声地板)**。
预注册:粗场 MVR > 地板 + **3 个百分点**,或 AFR > **2×** 地板 → (a)(c) 有实质贡献;全部指标落在地板 **1.2×** 内 → (a)(c) 贡献证伪,脏边几乎全归 (b)。

**DX-4(判 c,损失盲区审计)**:构造反事实对:GT vs GT+典型伪影(按失败画像注入:等值线扭曲、图像纹理刻入、全局变宽),伪影幅度用 E_HF 标定到与实测 D_total 同量级。测现行每个损失项(BCE、边界罚、面积罚)的 Δloss。
预注册:某伪影的 Δloss/loss < **5%**(batch 统计噪声内)→ (c) 对该伪影**量化成立**(损失不可见);Δloss ≥ **20%** → 监督本可罚它,脏边另有优化侧原因(表达偏置或权重失衡),修复应偏架构/算子而非加损失。

**DX-5(家族可分性探针——全部软路由方案的公共地基,审计指出从未被测)**:冻结特征池化 ⊕ 指令嵌入 → 线性/两层 MLP 探针预测家族标签。
预注册:held-out acc ≥ **99.5%** 且「解析被误判语义」方向错误率 ≤ **0.5%**(该方向 = 最坏情形:解析样本走贴边路径)→「廉价可靠判别」达标,门类方案放行;acc ∈ [95%, 99.5%) → 只许软门 + 损失兜底,禁任何硬路由;acc < **95%** → 全部 gate 类方案降权,优先无标签机制(mutual-structure 门控 / Post-DAE / 两族通用损失)。

**DX-6(oracle 路由上界)**:eval 期按 GT 家族标签硬路由(解析→bilinear/带限,语义→现行引导),给全部条件算子方案定收益上界。
预注册:oracle 路由使解析族中位 soft-IoU **+≥0.05** 且 κ̃ 比值降 ≥ **50%**、语义族指标不动 → 条件算子路线 headroom 充分;oracle 收益 < 症状的一半 → 粗场质量是瓶颈,必须上 P2/P3。

### 2.5 归因读数 → 处方分配规则

```
δ_b 高(≥60%) 且 粗场干净(DX-3 落地板 1.2× 内)  → P1 主打,P2 轻量叠加
δ_b 中(20–60%) 或 粗场亦脏                      → P2 先行(含底座损失审计),P1 并行
DX-4 显示损失全盲 + 粗场脏                       → (c) 实锤,P2 的结构项按盲区定制
深尾「全局变宽」在 P1+P2 后仍不动               → 上 P3((s,w) 重参数化)
P1+P2+P3 后解析族仍达不到解析级                 → 申请裁定 1 边界裁决,上 P4(解算-渲染)
```

---

## 三、方法谱系地图(六方向)

### 3.1 shape-constraints(形状/几何约束:损失级 / 重参数化级 / 架构级)

**适用**:等值线规整、单调性、凸性有三层可微实现,均与五条约束相容;结构损失只在训练期起作用、烘进权重,推理零开销。「按族选择性施约束」是文献常规(star-shape 只加病灶类,log-barrier 逐类尺寸界)。
**失效**:硬单调架构曾因激活选型系统性欠拟合(ICML23 修复);曲率/TV 权重过大 → 面积收缩与阶梯化,须窄带计损 + 相对 GT 的超额形式;三罚(超额 TV + eikonal + 曲率)权重强耦合;对语义族凹形细节全部有害,**必须族门控**。
**成本**:损失级实现各半天—1 天,推理零增量;架构级(PICNN/单调 MLP)实现 2—4 天,推理小增量。

| 机制 | 层级 | 一句话 | 关键失效 |
|---|---|---|---|
| Star-shape 径向单调 hinge | 损失 | 沿 GT 中心/方向射线罚单调翻转 max(0, 符号·Δy) | 只罚翻转不罚小幅高频;凹形物体是错误先验,禁全体施加 |
| Euler elastica 曲率罚 | 损失 | (a+bκ²)\|∇y\|,κ=div(∇y/\|∇y\|),直接罚等值线曲率+长度 | \|∇y\|→0 病态须 ε+窄带;二阶梯度方差大;建议粗场施加 |
| AC/TV 长度项(超额形式) | 损失 | \|TV(y)−TV(y_GT)\| 压高频毛边 | 裸 TV 极小值=分段常数(阶梯化),不用超额形式必翻车 |
| 双边亲和损失(ECCV18) | 损失 | 「贴图像边」先验从算子挪成**仅语义族**的训练损失 | 贴边从「保证」退化为「学到的偏好」,需 grid 边界 F1 盯住 |
| (s,w) 重参数化 + eikonal(IGR) | 重参数化 | y=σ((s−s0)/w),s 加 (\|∇s\|−1)² → 等值线自动平行等距光滑 | 坏临界解、初始化敏感;语义族平坦区 \|∇s\|=1 是错误约束;w 必须有界 sigmoid |
| PICNN 坐标-凸分支 | 架构 | 对画布坐标凸 → 等值线硬保证嵌套凸曲线(三类解析场精确可表达) | 非负权优化慢;软门坍缩成事实 if-else 需监督+熵正则;坐标只许从凸路径进 |
| 单调 MLP(ICML23 修复版) | 架构 | 对投影坐标 t=dᵀ(u,v) 硬单调 | 方向 d 预测错则整场错(单点故障) |
| 拟凹采样 hinge(凸形状先验) | 损失 | 采样 (p,q,m) 罚 max(0, min(y_p,y_q)−y_m) | 采样方差;**注意:三个解析子族超水平集全部是凸集,可全族统一施加**(审计纠正原条目) |
| soft-opening 罚(clDice 型) | 损失 | 迭代 min/max pooling,罚 \|y−open(y)\| 削孤立噪点 | 结构元须小于最窄条带;max/min 梯度稀疏 |
| log-barrier 约束退火 | 施加方式 | 结构统计量设上界,barrier 温度递增逼近硬约束 | 需可行初始;阈值须 GT 整臂标定(禁逐图) |
| ACNN 形状 AE 隐空间罚 | 学习先验 | GT 场上训 AE,罚 ‖E(y)−E(y_GT)‖²,两族统一免路由 | 低通,不管细部锐度;λ 过大拉向流形均值 |

### 3.2 freq-control(频率控制:带限参数化 / 频率门控)

**适用**:「解析族脏边」是典型混叠/高频泄漏问题。带限输出参数化(BACON/BANF)给**构造级**光滑保证——监督带宽不必匹配输出带宽,全分辨率 GT 监督低带宽出口,它自动拟合 GT 低通部分;频率门控(FreeU/LPTN/SAPE/Mip-NeRF IPE)把「高频只留给需要它的样本/位置」做成条件软机制。
**失效**:**带限 ≠ 解析**——窄条带/小软边宽的合法高频会被固定低通截断,深尾恰在此,单用必在深尾失效,须与条件带宽(Mip-NeRF σ 回归)或形状路线组合;谱偏置不可依赖(隐式偏置只是速度排序);四条核心文献(BACON/BANF/SAPE/NFFB)是单信号 test-time 拟合设定,摊销成条件前馈头属系统性外推。
**成本**:BANF 型带限上采样 = 换插值核,near-free;带限出口/子带结构 1—3 天;推理零—小增量。

| 机制 | 一句话 | 关键失效 |
|---|---|---|
| BACON 带限出口 | 谱有解析上界的多出口头,解析族走低带宽出口 | 带宽构造期冻结,不能逐样本连续调 |
| BANF 粗网格+带限插值 | 粗场→带限核插值本身就是低通,直接替换引导上采样 | 线性核低通不严格;sinc 有 Gibbs;带宽档位离散 |
| LPTN Laplacian 金字塔+掩膜 | 低频基场+「图像高频×模型掩膜」逐层调制,闭式重建 | 掩膜不压死则纹理照样刻入(须家族标签监督掩膜) |
| FreeU base+s·Δ 分解 | y=base(带限上采样)+s·(guided−bilinear),s 由条件预测 | 原文全局手调标量,移植属结构类比需消融 |
| SAPE 逐位置频率门(摊销) | 前馈预测 1/16 逐格带宽图 b(x) 调制高频分支 | 门自身需低通(放粗分辨率),否则门引入高频噪声 |
| Mip-NeRF IPE 连续带宽 | exp(−σ²ω²/2) 闭式衰减,σ 由指令+特征回归 | σ 预测错=带宽错,须校准/负控制;需先加 Fourier 特征 |
| FreeNeRF 频率课程 | 训练期从低频逐步放开高频(一行实现) | 纯时间课程,单用无法表达两族差异 |
| NFFB 子带出口 | 2—3 个子带出口,族条件门按子带施加 | 子带绑定是软约束,无硬保证 |
| StyleGAN3 带限算子 | 解析支所有算子低通包夹,「不需要的信息无法泄入」 | 有实测开销;只能用在解析支,全网带限伤语义贴边 |

### 3.3 adaptive-guidance(引导自适应:让引导强度成为双方的函数)

**适用**:全部解法收敛到同一原则——**引导强度必须是 (guide, 目标/输出) 双方的函数,而非只看 guide**。四类机制:①互结构/不一致度量当权重(零学习成本,推理期免标签的天然软路由:解析族互结构≈0→引导自动关断);②逐像素线性系数 y=a(x)·G+b(x)(SVLRM,a→0 即脱钩);③单系数结构转移量 y=LP(coarse↑)+α(x)·HF_guide(UMGF);④可学核/引导图(PAC/DKN/DGF)。
**失效**:83/17 失衡致门控塌缩;粗场过糊低估语义族互结构→贴边受损;SVLRM 双系数**已被 UMGF 文献内判次优**(halo 根源);DGF 的引导变换是任务级共享函数,两族混训须补条件注入;FBS 默认形态平滑项永远沿 reference 边——**改造前对解析族是纯污染方向不可用**;PAC/DKN 全分辨率开销高,需低分辨率+轻量精修折中。
**成本**:不一致度量门控 1 天;UMGF 型改造 2—3 天;PAC/DKN 3—5 天且推理增量最大。

| 机制 | 一句话 | 关键失效 |
|---|---|---|
| JBU(现行原型) | 诊断对象:核心假设对解析族整体为假 | texture copying 系统性 |
| GDSR 综述 | 诊断出处:∇Y=a_k∇G 机理定式化 + 解法谱系 | 深度图仍有边,本问题更极端 |
| Mutual-Structure 门控 | 逐像素局部结构相似度乘性门控 range 核,免标签 | 粗场糊/错位时低估互结构 |
| 边不一致度量(TIP18) | w(x)=exp(−inc(x)/σ),零参数基线 | 解析族「无边可比」时退化为单边检测 |
| DJF/DJFR 双分支融合 | 学习版互结构,只转移双方一致的结构 | 黑盒无显式 α,失衡下偏解析行为 |
| SVLRM y=a(x)G+b(x) | a→0 即与 guide 脱钩 | 双系数联合估计次优(halo);b 无光滑约束时噪声漏进 |
| UMGF 单系数 α | y=LP+α·HF_guide,α=0 即纯光滑路径 | LP 截止频率过低抹掉条带真台阶 |
| DGF 可学引导图 | 引导图=T(I,F_coarse),解析样本学出近平坦引导 | T 只看图像则原样污染;g 塌常数则语义丢贴边 |
| PAC 可学引导嵌入 | 引导空间可学,f 平坦→核退化为普通平滑卷积 | 无监督不保证学平;计算重 |
| DKN 可变形核 | 逐像素 (offset,weight),不一致处权重收回自身邻域 | 17% 语义样本可能欠拟合 offset;显存/延迟高 |
| FBS 可微求解器 | 数据项置信 c(x) + 亲和混合 λ(x) 由网络预测 | 默认平滑方向对解析族纯污染,必须加亲和混合 |

### 3.4 manifold-projection(自由场 → 可微投影到解析族)

**适用**:与「回归渲染参数」有文献级形式区分——投影层内族参数是 argmin 隐变量,外层损失仍在输出空间,梯度经隐函数定理只依赖最优点局部几何(DDN/OptNet),**不暴露参数坐标的全局条件数**;HardNet 定理证明附加投影层保留万能逼近能力。DSAC 同构实证:dense 证据+最小子集解算+概率化选择 > 直接回归;**soft-average 会塌缩,须概率选择训练 + argmax 推理**。额外红利:投影产出的解析参数可任意分辨率解析渲染——**解析族从而完全绕开引导上采样**。
**失效**:梯度病态集中于 argmin 非唯一/Hessian 奇异(两个等好渐变方向)、isotonic 块内梯度被平均、QP active-set 跳变;OptNet/cvxpylayers 逐样本解 QP **只能当粗场原型/因果验证工具,不能进最终方案**(裁定 4);「等值线为椭圆族」整体非凸,凸投影只保必要性质;**「解算/检测低维几何量」是否踩裁定 1 属项目决策,不许由文献静默拍板**。
**成本**:isotonic/PAV 逐射线投影 O(n log n) 1—2 天;DSAC 型解算-渲染每原语单独写解算器,1 周+;Post-DAE 2 天。

| 机制 | 一句话 | 关键失效 |
|---|---|---|
| DDN 隐式梯度 | argmin 层理论支柱:反传只依赖最优点局部几何 | 最优点非唯一/Hessian 奇异时梯度错乱 |
| HardNet 闭式投影层 | 仿射/凸约束硬满足 + 万能逼近定理 | 只覆盖仿射/凸;约束数≤输出维 |
| OptNet / cvxpylayers | QP/锥程序层,粗场「单调+光滑+box」最近点投影 | 立方复杂度,仅原型;active-set 跳变 |
| RAYEN 射线缩回 | 闭式无迭代永远可行,最快硬可行层 | 非最近点投影(向锚点收缩有偏);仅凸集 |
| DC3 完成+校正 | k 步 ∇violation 校正推进约束集,渐近可行 | k 小残留违反;非凸下震荡 |
| isotonic/PAV 逐射线投影 | 沿方向硬单调化,精确 Jacobian,O(n log n) | 块内梯度被平均;方向错则主动破坏正确场;阶梯化须配平滑 |
| DSAC 解算+软选择 | 采样点闭式解算参数→多假设打分→概率选择 | 策略梯度方差大;分数接近时推理抖动;**踩裁定 1 边界待裁决** |
| Deep Hough 参数空间读出 | 固定积分变换汇聚证据,峰值检测代替回归 | 离散化精度;椭圆需广义 Hough 开销升 |
| Post-DAE 学习投影 | GT+症状化退化训 DAE,流形上近恒等=内建软路由 | 退化分布域差;「语义族近恒等」是假设须实测 |
| PRL 零空间闭式 | 线性**等式**约束硬内建 | 解析族几乎无线性等式,**实际可用面近零**(审计纠正) |

### 3.5 conditional-operator(条件化输出算子:同一算子、条件化参数)

**适用**:FADE 三臂对照裁决机理 (b) 同构成立并给统一软融合方案;三条配方:①FiLM/任务条件 SE(ASTMT)注入家族/指令嵌入(零推理开销必做基线);②CondConv/DynamicConv 专家核凸组合(sigmoid 路由或 softmax+温度 30→1 退火,**已证可稳定端到端训练**,路由自发双峰=软机制自己逼近双路径而不违反统一框架);③DySample「恒等=bilinear」offset 采样,offset 幅度门由条件生成——解析族门→0 即严格退化为纯双线性。家族标签只进训练:直接 BCE 监督 gate,或 Mod-Squad 式最大化 MI(族;专家)(**其 P(E) 熵项显式防 83/17 饿死少数族专家**)。
**失效**:软门+族标签 BCE → 门双峰化(CondConv 实证)→ 事实 if-else,裁定 3 合规与否取决于门错误率(**从未被测,即 DX-5**);门错误代价不对称(解析走贴边路径=最坏);softmax 原始温度直训=塌缩(DynamicConv 实证,与本项目两次塌缩病态同类);FiLM/SE 纯通道级无空间分辨率,不能表达图内局部混合;ASTMT 原文推理期需任务 ID,须换预测后验软加权。
**成本**:FiLM 半天;DySample 替换 1—2 天;CondConv/MoE 2—4 天;推理全部近零增量(DySample 6.2ms vs bilinear 1.6ms @256×120×120)。

| 机制 | 一句话 | 关键失效 |
|---|---|---|
| FADE 门控加性细化 | F_up=F_img·G+F_main·(1−G),G=sigmoid(decoder 1×1) | 两族特征不可分则门失效;DySample 实测其 gating 部分任务不稳 |
| DySample 条件 offset | 零 offset 严格=bilinear;幅度门条件生成 | 贴边上限低于 guided 类(bIoU 差 1.7–2.2),语义族或需补偿 |
| CARAFE 内容核 | 核只由粗场内容生成,不看图像,region 端点参照 | detail 任务明确劣于 encoder 引导,单用语义退化 |
| FiLM | 条件生成 (γ,β) 通道调制,最低成本基线 | 无空间分辨率;条件源须含指令/家族信号 |
| CondConv | 输出核=Σαᵢ·Wᵢ,sigmoid 逐例路由 | 专家未必按家族分化;整图一套核 |
| DynamicConv | softmax(z/τ) 路由,τ=30→1 退火防塌缩 | 不退火=塌缩单专家(实证);凸包限表达 |
| ASTMT 任务条件 SE+adapter | 同一 backbone 两种相反偏好干净切换;梯度对抗防 83% 淹没 17% | 原文需任务 ID;通道级同 FiLM |
| Mod-Squad MI 路由 | −I(族;专家) 塑造路由,稀疏分工且不饿死少数族 | MI 靠 batch 内统计,小 batch/极不平衡噪声大 |

### 3.6 structural-supervision(结构化监督:把「解析」写成输出空间损失)

**适用**:全部为输出空间损失(裁定 2 正面相容),家族标签只做 loss mask(裁定 3 相容),推理零开销(裁定 4);「GT 是解析渲染」提供特权信息:精确 ∇y_GT(SIREN 证明导数监督良定)、GT 方向场 d(逐点单调 hinge)、GT 等值线距离图(boundary loss 型)、GT 高频带能量≈0(静态频域罚);generalized distillation/LUPI 是「特权信息只在训练期」的正统理论框架。
**失效**:共性——结构项权重过大压平场、吃掉语义族细节,**所有结构损失必须按族门控或条件加权**;‖∇y‖≈0 与 y 饱和区需 ε 稳定化(eikonal 须在 logit/势空间施加);boundary loss 单用塌缩须与区域损失 ramp 混合;二阶项(曲率)放大差分噪声建议粗场施加;纯导数监督有全局偏移不定性须混 0 阶项。
**成本**:每项半天—1 天;权重扫描 2—4 组;推理零。

| 机制 | 一句话 | 关键失效 |
|---|---|---|
| MiDaS 多尺度残差梯度匹配 | Σ_k\|∇(y−y_GT)\|@4 尺度,两族通用免门控 | 只罚残差一阶结构;GT 错位处推糊 |
| BGMv2 Sobel 梯度 L1 | 软 alpha 场梯度监督,产品级标配,与本设定同型 | 一阶项对等值线形状无显式约束 |
| SIREN 型导数回归 | 对 ∇y 直接回归解析渲染的 ∇y_GT | 偏移不定性;粗场差分噪声;语义族无解析梯度须门控 |
| 逐点方向单调 hinge | Σ max(0,−⟨∇y,d⟩),d 取 GT 特权方向场 | 常数场也满足单调→须与 0 阶项平衡 |
| VNL 型三点组长程罚 | 线性场任意三点共面→罚偏离仿射插值,O(K) 采样 | 仅线性族;语义族完全不成立;采样方差 |
| 静态高频带能量罚(FFL 退化版) | 罚 Σ_{B_hi}\|F(y)\|²(解析 GT 该带≈0) | DFT 全局性定位差;语义族高频合法须门控 |
| 多水平 boundary loss | GT 等值线 {y=c} 距离图罚水平集位移 | 单用塌缩;逐水平预计算距离图;二值掩膜文献移植软场未验证 |
| SDM 势场参数化 | 头输出势 u,固定 σ(u/τ) 转 y,光滑在 u 空间表达 | 两损失平衡不稳;τ 须条件化 |
| 成对差分蒸馏(Structured KD) | 远距点对 y(p)−y(q)=⟨d,p−q⟩ 罚,长程等值线敏感 | O(N²) 须采样;对语义族无益须降权 |
| generalized distillation | 教师吃特权(标签+参数),学生统一头蒸馏,推理无路由 | 教师误差蒸给学生;λ/温度敏感 |

---

## 四、设计处方(按证据强度排序)

### 4.0 收敛主模板与互斥槽(组合前必读)

**四路独立收敛的主模板**(可信度最高):

> **y = 低频/内容基路径(家族中立带限上采样) + 条件门控的图像高频加性残差**
> UMGF(LP+α·HF) ≡ FADE(gate 加性细化) ≡ LPTN(金字塔+掩膜) ≡ FreeU(base+s·Δ) ≡ BANF(带限基+残差)

损失级结构罚(族门控、训练期)与可选投影兜底在其上**正交叠加**。

**同槽互斥(二选一,混用即冗余或冲突)**:
1. **上采样槽**:{加性门控主模板} vs {DySample offset 重采样} vs {SVLRM (a,b)}——SVLRM 已被文献内判次优;DySample 是 warping 与加性分解不同构,叠加冗余;
2. **解析支表达槽**:{PICNN 凸} vs {单调 MLP} vs {BACON 带限出口} vs {isotonic 投影}——多硬机制叠加会过约束,交集可能排除合法 GT(高曲率小椭圆、窄条带);
3. **通路级替代**:{解算-渲染(P4)} vs {场路径改造(P1–P3)}——互为替代不是叠加,两个都做是预算浪费。

**损失施加空间必须先裁定**:eikonal 在 y 饱和区失效须在 s/logit 空间施加,曲率罚在 y 空间——**组合前先裁定输出参数化(直接 y vs (s,w)),再选损失集**;这是排期前的决策项。

---

### P1 「带限基 + 门控引导残差」——证据最强(机理 (b) 裁决成立 + 四路模板收敛)

- **机制一句话**:上采样重写为 y = U_bl(coarse) + α(x)·Δ_guide,其中 U_bl 为家族中立带限插值(BANF 型),Δ_guide = guided(coarse) − U_bl(coarse),门 α 由粗场特征 ⊕ 指令嵌入在 1/16 分辨率预测(门放粗分辨率防门噪声);家族标签仅训练期给 α 加 BCE 辅助监督并线性衰减到 0。
- **约束相容性**:裁定 1 ✓(无任何参数回归);裁定 2 ✓(损失全在场上);裁定 3——α 为模型内软门,放行前提 = DX-5 探针通过;附 Mod-Squad 式 MI 项或按族重采样防 83/17 塌缩,门错误代价不对称单列报告。裁定 4 ✓(推理增量 <10%);裁定 5 ✓。
- **语义族无损论证**:α→1 时恢复现行引导行为(上界=现状);另按 ECCV18 把贴边先验做成**仅语义族**的双边亲和损失烘进权重,双保险;护栏指标 = grid 边界 F1。
- **最小验证实验**:一次训练(数小时)。预注册:解析族 E_HF ↓≥**70%**、κ̃ 比值 ≤**1.5**、MVR ≤地板+**1 个百分点**;语义族 grid 边界 F1 Δ ≥ **−0.005**、soft-IoU Δ ≥ **−0.01**。**证伪**:语义护栏任一破 → 回退引导支权重;解析族 E_HF 降幅 <30% → 主模板对本任务失效,重心转 P2/P3。
- **成本**:实现 2—3 天,训练 1 次,推理近零增量。

### P2 「结构损失包」——训练期修复,与 P1 正交可叠,证据次强(MiDaS/BGMv2 产品级 + LUPI 框架)

- **机制一句话**:底座损失审计(BCE→L1/Charbonnier 对软 GT 的对照,缺口 M11)+ 两族通用多尺度残差梯度匹配 + 解析族门控项{方向单调 hinge(GT 特权方向)、超额 TV、窄带 elastica 曲率(粗场施加)、静态高频带能量罚}。
- **约束相容性**:裁定 1 ✓;裁定 2 ✓(全部输出空间损失);裁定 3 ✓(家族标签只是 loss mask,推理无任何路由);裁定 4 ✓(推理零开销)。
- **语义族无损论证**:通用项(梯度匹配)对语义族本来有利(BGMv2 同型任务实证);族门控项语义族权重恒为 0,结构上不可能伤语义族。
- **最小验证实验**:逐项消融,每行必带 Δ_const/Δ_shuffle 列(项目红线)。预注册:粗场 MVR 降至地板+**1 个百分点**内、解析族深尾 AFR(P90)↓≥**40%**。**证伪**:某项使语义族 grid F1 掉 >**0.01** 或解析族指标改善 <**10%** → 该项撤下;三罚联调两轮仍震荡 → 换 log-barrier 退火,再失败则该子包放弃。
- **成本**:每项半天—1 天,权重扫描 2—4 组训练;推理零。风险:三罚权重强耦合(audit 隐藏冲突 #2)。

### P3 「(s,w) 重参数化 + eikonal」——直接打深尾「只能全局变宽」,证据中等(IGR+SDM,摊销外推)

- **机制一句话**:统一头改输出势场 s 与有界宽度 w(sigmoid 参数化,呼应「σ 禁裸 exp」红线),y = σ((s−s0)/w);解析族在 s 上加 eikonal 罚 (\|∇s\|−1)² 与方向监督,软边宽度被 w 显式化——「干净的空间变化衰减」成为可表达量。
- **约束相容性**:裁定 1 ✓(s 是 dense 场不是低维参数);裁定 2 ✓(主监督仍在 y 上,s 上仅正则);裁定 3 ✓(族差异由 (s,w) 形态**涌现**——语义族 w→小自然退化近二值掩膜,无门无路由);裁定 4 ✓。
- **语义族无损论证**:w 小时 σ 陡峭、等值线由 s 零水平集决定,贴边继续由上采样/亲和损失负责;eikonal 对语义族减权或不加。
- **最小验证实验**:预注册:深尾软边样本(GT ramp 宽 > P75)的 AFR ↓≥**50%**、「全局变宽」失败模式占比减半;语义族 soft-IoU Δ ≥ **−0.02**。**证伪**:s 空间训练两次发散 → 撤(呼应本项目训练塌缩前科);深尾指标不动 → 深尾问题不在参数化,在表达/监督。
- **成本**:改头+损失 2—3 天;风险:eikonal 坏临界解、初始化敏感(IGR 明言)。

### P4 「解算-渲染支路」——上限最高,但踩裁定 1 边界,**需主 agent 裁决后才许排期**

- **机制一句话**:解析族把粗场当 dense 证据:最小子集闭式解算 / 可微 Hough 峰值读出族参数 → 多假设打分、概率选择训练、argmax 推理 → **任意分辨率解析渲染,完全绕开上采样污染源**;语义支原样保留,混合门同 P1。
- **约束相容性**:裁定 1 ——**边界游走**:「解算/检测」非「回归」有 DDN 文献级区分(梯度只依赖最优点局部几何,不暴露 10⁶ 条件数坐标),DSAC 实证解算+选择优于直接回归;但「病态判死」的边界(全参数联合回归 vs 任何低维几何量估计)是项目裁定,**决策项,不许静默拍板**。裁定 2 ✓(损失在渲染出的场上);裁定 3 同 P1 软门;裁定 4:推理为一次解算+渲染,低开销。
- **语义族无损论证**:语义支完全不动。
- **最小验证实验**:先 offline 零训练——GT 粗场+注入噪声测解算器精度。预注册:方向误差 <**2°**、偏移 <**1 格**的样本占比 ≥**95%**;**证伪**:解算成功率 <90%,或多假设选择训练 loss 方差不可控(DSAC 已警告),或 soft-average 塌缩复现 → 撤。
- **成本**:每个解析原语单独写闭式解算器,1 周+;工程最重,但解析级干净是**构造保证**(唯一能严格达到「解析级」的路线)。

### P5 「Post-DAE 学习式投影兜底」——耦合最低,任何阶段可旁路接入

- **机制一句话**:用己方渲染器无限量合成(干净场,症状化退化)对——退化按实测失败画像定制(高频噪声、等值线扭曲、纹理刻入、全局变宽)——训练轻量 DAE 接在输出后;流形上近恒等 = 单算子内建软路由。
- **约束相容性**:裁定 1 ✓(AE 隐坐标是重建目标塑造的良态参数化,仅正则/后处理);裁定 2 ✓;裁定 3 ✓(无门无标签);裁定 4:推理 +1 次轻量前向。
- **语义族无损论证**:「语义族近恒等」**是假设不是保证,必须实测**——语义族过 DAE 后 grid F1 Δ ≥ **−0.003** 才许上线;接入位置须在带限支后(audit 冲突 #6:接在引导上采样后会磨贴边,接在解析支后与带限部分冗余)。
- **最小验证实验**:预注册:解析族深尾 E_HF ↓≥**50%**;**证伪**:语义贴边磨损超阈,或退化分布与真实模型误差域差导致投影失真(盲审 20 例可视化,按项目可视化纪律出图)。
- **成本**:数据合成+小模型 2 天。

### 4.6 执行序(诊断先行,硬规则)

```
第 0 步(1–2 天,全部零训练):DX-1 → DX-2 → DX-3 → DX-4 → DX-5 → DX-6
       产出:δ_b(b 占比)、粗场干净度读数、损失盲区表、家族可分性 acc、oracle 上界
第 1 步:按 §2.5 分配规则选主修复臂(默认预期:δ_b 高 → P1 主打 + P2 轻量叠加)
第 2 步:P1/P2 消融(每行 Δ_const/Δ_shuffle;新指标 E_HF/κ̃/MVR/AFR 与常规三列并排)
第 3 步:深尾不动 → P3;仍不达解析级 → 提请裁定 1 边界裁决 → P4
兜底:P5 可在任何阶段旁路验证,但不得在 P1 落地前接在引导上采样之后
```

### 4.7 隐藏冲突与公共风险(全部候选共享,排期时逐条对号)

1. **软门+族标签 BCE → 门双峰化 → 事实 if-else**(CondConv 实证):裁定 3 合规取决于门错误率,DX-5 是放行门;门错误代价不对称(解析→贴边路径=最坏)。
2. **裸 TV/长度项压扁渐变、isotonic 阶梯化 vs 平滑罚互拉**:须「超额形式 + ε-强单调斜率下界」同时上;三罚权重强耦合,log-barrier 退火为对策,阈值 GT 整臂标定。
3. **带限 ≠ 解析**:窄条带/小软边宽合法高频被固定低通截断——freq-control 单用恰在深尾失效,须配条件带宽或形状路线。
4. **损失施加空间不统一**:先裁输出参数化再选损失集(§4.0)。
5. **硬投影层与结构损失梯度干扰**(PAV 块内平均、QP active-set 跳变):安全组合=「损失进训练、投影进推理」;两者都进训练需专门消融。
6. **Post-DAE 接入位置**(见 P5)。
7. **双 schedule 错配**:频率课程(FreeNeRF)与路由温度退火(DynamicConv)叠加时,高频解锁若晚于路由定型,语义族专家学不到贴边。
8. **83/17 失衡**是所有 router/gate/MoE 的公共风险;Mod-Squad MI 损失或按族重采样是通用解(重采样与「家族标签只进训练」相容)。

---

## 五、缺口如实清单

### 5.1 诊断强度保留意见
- **(a) 诊断无跨调研路线交叉**:三条支撑全部出自 freq-control 一路,且全部来自 coordinate-MLP/生成模型设定,对「conv 头 + BCE + 高维冻结条件输入」只有定性外推效力。IGR、elastica、ACNN、PAC、DGF 各在两路重复出现——按论文口径「独立支撑数」达标,按调研路线交叉口径 **(a) 未达标**。
- **(c) 的诊断主要是调研者推理,不是文献裁决**:三条支撑全是泛化论断,无一直接研究「解析等值线不可分辨性」。DX-4 是补此缺口的实验。
- (a)(c) 相互混杂、与 (b) 未解耦;六路均未主动提出归因判别设计(§2.4 电池为缺口审查补出)。

### 5.2 决策项(待主 agent 裁决,不许静默拍板)
1. **裁定 1 边界**:单调 MLP 的方向 d 预测、Deep Hough 的 (角度,偏移) 读出、DSAC 的采样点解算——三者都在产出低维渲染参数,只是用「解算/检测/软选择」代替「回归」。文献级区分存在(DDN),但边界归属是项目裁定。P4 的排期以此为前提。
2. **输出参数化裁定**:直接 y vs (s,w) 势场——决定可用损失集(eikonal 须在 s/logit 空间)。
3. **门可靠性标准**:裁定 3 的「已证 100% 可靠」如何操作化(DX-5 的 99.5%/0.5% 阈值是本文档提案,需追认)。

### 5.3 设定外推风险(交付时须当新设计预注册消融)
- boundary loss / clDice / star-shape 全部为**二值掩膜**设计,对宽 ramp 软场的多水平集改写、opening 罚、射线单调 hinge 均无文献验证;
- BACON/BANF/SAPE/NFFB 为**单信号 test-time 拟合**文献,摊销成条件前馈头属系统性外推(整类风险);
- FreeU 结论解剖自生成式 U-Net,移植属结构类比;
- OptNet/cvxpylayers 显式降级为「粗场因果验证工具」,不进最终方案;FBS 默认形态对解析族是纯污染方向,改造前不可用;
- PRL 零空间闭式实际可用面近零(解析族无线性等式);Convex Shape Prior 原条目的子族细分警告方向错误——三个子族超水平集全部是凸集,拟凹 hinge 可全族统一施加,只需排除语义族。

### 5.4 缺失文献线(M1–M11,建议查询词见缺口审查原文)
| # | 缺失线 | 为什么重要 |
|---|---|---|
| M1 | TGV(二阶广义全变差) | 分段仿射先验精确匹配线性渐变;是「TV 阶梯化」问题的标准答案(调研自造的「超额形式」的正规替代) |
| M2 | HDRNet / bilateral-grid 系 | 低分系数网格+引导 slicing 与现行架构同构且出自修图域 |
| M3 | UMNN 积分参数化单调 | 沿方向预测非负导数再累积——最便宜硬单调,免权重符号约束欠拟合 |
| M4 | PointRend / 隐式边界细化 | 语义族贴边的「无图像引导滤波」实现——把污染源整体移除,主模板之外唯一根治路线 |
| M5 | LIIF / 任意分辨率隐式解码 | 家族中立连续上采样器,基路径候选 |
| M6 | 软场评测指标(matting Gradient/Connectivity 误差) | 「解析级干净」量化判据;§2.0 指标为本文档自定义,需与成熟指标对齐 |
| M7 | 语言→空间几何 grounding | 渐变方向由指令决定,六路无一条处理文本→空间方向条件化 |
| M8 | SPADE 空间条件归一化 | FiLM 纯通道级短板的现成补丁(图内局部混合需求) |
| M9 | 选择性预测 / 门校准 | 裁定 3「100% 可靠廉价判别」分支需要校准与拒识文献支撑 |
| M10 | PnP / RED 学习式近端算子 | Post-DAE 的成熟谱系,近恒等性与收敛性有理论 |
| M11 | 软目标逐点损失选型 | 现行 BCE 对软 GT 的底座合理性未审(matting 域标配 L1/Charbonnier) |

**补文献优先级**:先 M1/M4/M6(直接改变设计与判据),M2/M3 随主模板选型定,其余按需。

### 5.5 公共未验证前提
所有软路由方案(门 α、σ 预测、router、家族后验)隐含「冻结特征+指令可判家族」——**没有任何一条 finding 或实验测过**。家族大概率由指令文本决定(渐变方向由指令决定),若成立则路由可近 100% 可靠。一个线性探针即可测(DX-5),**应最先做**。

---

## 六、VERIFIED 参考文献表

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
