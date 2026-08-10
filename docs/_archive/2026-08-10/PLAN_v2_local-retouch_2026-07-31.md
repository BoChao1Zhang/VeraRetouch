# 局部精修研究计划 v2.1（2026-07-31）

> **方法底稿 / 预注册快照。** H1、H3、RO-9、N*=32 等初始假设已有后续正反结果，
> 不应从本文件推断当前结论。当前结论见 [`EXPERIMENT_RESULTS_CURRENT.md`](EXPERIMENT_RESULTS_CURRENT.md)，
> 逐实验出处见 [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)。

> 来源：8 轮外部文献调研（现状 / 对抗核验 / 双边泼溅 / GLUT-4D / 执行期 / 掩膜基底 / 颜色探针 / transformer 渲染器），全程未读本仓库历史。
> v2.1 相对 v2：加入完整备选方案库（渲染器 R-1..R-11、读出 D-0..D-10）、消融总表、失败分支树、论文外支线。
> 所有 arXiv 编号经 agent 打开 abs 页核验，但**写论文前仍须逐条用 export.arxiv.org API 复核**（本次调研抓到检索引擎双向错误：编造论文 + 武断否定真实论文，见 §9）。

---

## 0. 假设清单与三条腿

| 编号 | 假设 | 证伪条件 |
|---|---|---|
| **H1** | 颜色信息在 VLM 表征里，语言/latent 读出通路丢了（机制候选按证据强度：**control-latent 带宽/有效秩不足** > 末层退化为全局池化 > connector drop-off > 注意力同质化 > sink/register 伪影） | 探针 ≈ 行为回答；或 LoRA 2k 图即抹平（只能写 pathway-training deficit） |
| **H2** | 空间信息在表征里，读出口没有承载通道 | 修复伪影后读出图仍与随机场无异 |
| **H3** | 存在层 L，颜色探针与空间探针同时接近各自最优（统一读出层） | 两峰层号相差 > 1/4 深度 |
| **H4** | 4D 渲染器容量足以表达局部编辑，且退化为全图 | 容量阶梯 L0–L4 任一判死 |
| **H5** | 局部子集上 4D 相对 3D 有显著增益 | Δ_ceil < 3 dB 或实际 < 1 dB |

**因果链**（三条腿是同一条信息通路的三个环节，不是三个并列卖点）：
```
编码（有信息）──腿A：探针证明──▶ 传输（读出口太窄）──腿B：换读出口+渲染器──▶ 落地（可测量）──腿C：分区指标
```

**story 三段式**（四六级词汇）：
> Finding. The model already holds what it needs. A small probe reads colour, and place, out of its hidden states, while its own words get them wrong.
> Diagnosis. The read-out path has no room. Three global latents carry one colour change for the whole image, so the rest is dropped.
> Fix. We widen the path. A short weight vector picks from a basis of shape and meaning, and the colour table gains one more axis.

---

## 1. 架构定稿

### 1.1 渲染器（逐像素算子，极轻，可烘焙）

```
f(x) = Σᵢ ŵᵢ(x)·(Mᵢ·curve(x) + bᵢ) + G·curve(x) + c        Σŵ=1
```
- N=32 个各向异性高斯（3D 起步；4D 后 μ∈R⁴、6 个 Givens 角——4D 无四元数，SO(4)=6 自由度，保留逐轴裁剪语义；不用 Cholesky 10 参：混掉"尺寸/朝向"）。
- **+3×17 点单调 1D 前置曲线**（`cumsum(softmax)`，+51 参；对症 cell utilization 仅 5.53%；烘焙时与高斯精确复合仍出单张 .cube。SepLUT 证据：47.2K 带曲线 > 593.5K 纯 LUT）。
- **+存在门 gᵢ**（Stage-2 加 L1 稀疏后剪枝；Long-LRM λ=0.1 → 有效原语 ~40%）。
- payload：3D **836** / 4D **≈967** 个数；上限 N=64（≈1600）。
- **运行时**：参数一变即重烘焙 33³ LUT（比 4K 直接求值便宜 58×），4K 永远只查表。
- **硬边界**：逐像素算子禁 (x,y)/邻域、禁 MLP、禁排序/α-blend（GaussianImage 去排序 +0.8 dB）、σ 必须有界、s 须可折回逐点函数或"N 张 LUT+guide"。
- **容量证据**（为何不加大逐像素算子）：CSRNet×41=+0.2dB、AdaInt×7.6=+0.05dB、SepLUT×5.6=+0.06dB、NeurOp×2=+0.03dB；CSRNet 加 condition network=+3.22dB。

### 1.2 s 轴与 R1 塌陷的精确数学（勿再用"σ_s→∞"这个不完整描述）

块对角下 `w_i = o_i p_i^c(x) p_i^s(s) / Σ_j o_j p_j^c(x) p_j^s(s)`。
**退化集 D = {所有 o_i>0 的高斯共享同一 s 边缘分布}** ⟺ 所有高斯共享同一 (μ_s, σ_s)。三条后果：
1. σ_s→∞ 只是 D 的无穷远端；**μ_s 全相等在有限参数处可达**，且照抄 GLUT 各向同性初始化并把 μ_s 全设 0.5，**初始点正好落在 D 上**（对称鞍点）。
2. "只裁剪 σ_max"是错的——必须 **μ_s 锚定（R-2）与跨高斯多样性（R-3）成对使用**。
3. 沿 σ→∞ 方向输出空间惩罚的梯度以 O(1/σ³) 衰减救不动——必须在参数空间下手，或让 s 只出现在分子（R-1/4/5/6/8）。
- **ControlNet 零初始化是把 s 关掉（梯度非零可长回）；归一化对消是把 s 除掉（梯度恒零回不来）**——两者本质不同。
- **MoE load-balancing loss 治不了 R1**（塌陷时负载完全均衡，loss 恰好最优、零梯度）。正确替代：s 分桶互信息 `I(bucket_s; i)`（B=8，系数 1e-2）+ router z-loss（c_z=1e-3；**勿粗暴 clamp logits**，ST-MoE 实测 clamp 掉质量 −4.206 vs z-loss −1.741）。

s 定义：**寻址坐标（选哪套局部仿射），非混合强度**。必答题：固定 s 扫 RGB，展示不同 s 切片变换**形状**不同（否则"3D LUT+alpha 图"一句打死）。s 轴 M=12 个高斯、μ 均匀 [−3,3] **固定**、σ_s 有界 sigmoid；**s 轴禁平滑正则**（=推向塌陷），加单调性+切片可分性下界。

### 1.3 掩膜基底（s 的生产端；最终版含双轴）

```
单轴 v1:  q(p) = w₀ + α·(w_dir·φ_dir(p)),  ‖w_dir‖=1, α≥0,  s = 3·tanh(q/3)
双轴 v2（gate 后）: s₁ = 3·tanh((w₀¹+α¹·w_dir¹·φ_geo)/3)      # 5 维几何
                    s₂ = 3·tanh((w₀²+α²·w_dir²·φ_[range+sem])/3) # 2+6 维
                    高斯 (s₁,s₂) 块全协方差：轴对齐=交集，旋转=加性；严格包含单轴
φ = [1] ⊕ [x,y,P₂(x),P₂(y),xy]（Legendre 正交，中心化/除短边）⊕ [L,S] ⊕ [e₁..e₆]
```
- **2 阶定稿**（ψ 单调 ⇒ 等值线族只由 q 决定；6 阶渐晕多项式是 r 的单调函数被 s 轴高斯吸收）。三次 4 项 flag 消融；色相 sin/cos 2 维 flag（14→16）。
- α=0 **精确退化全局**（球面约束会让退化不可达）；α 初始 ≈0。
- **带通读出红利**：环形/径向带、语义∩几何交集（加性 logit 即可）、语义区内连续渐变（PerTouch/RSFNet 做不到——卖点）。
- **表达不了（limitation）**：单支楔形（对顶领结）→ 收窄为 linear/radial/elliptical/hyperbolic-band；重叠软掩膜（单轴秩上限）→ 双轴或操作序列；弯曲边界 → 语义基的活。
- 语义基：**固定文本投影锚定身份**（sky/skin/foliage/water/architecture/subject）+ 轻 adapter 联训；NMF/谱分解只当离线诊断；拼接前逐图标准化 + 对几何基残差正交化。
- 防塌陷：文本 emb 离线 Gram-Schmidt → VICReg variance/covariance（像素维）→ **禁 softmax 竞争**（掩膜须可重叠）→ |w_dir| 使用率均衡（"基饿死">"塌同图"）。诊断：effective rank / 两两 IoU / 使用率直方图。
- 边界：guided filter 作用在 **k 个基通道**上（非最终 s）。
- 可编辑性给闭式反解（A 特征分解 → 椭圆中心/主轴/羽化环）。
- **多步操作绝不串联**（RSFNet：20 组随机顺序近半失败；串联还毁可编辑性——第 t 步 w 以前序结果为条件，用户改第 2 步则第 3 步语义失效；梯度穿 T 次查表且钳位饱和——Exposure/JarvisArt 被迫用 RL 就是旁证）。**s 轴上的 M=12 个高斯本身就是 12 张带通掩膜**，共享 s 场、自动软划分、并行线性合成——多掩膜机制是内置的，不需要 k 个独立 w。

### 1.4 生成器（transformer decoder）

Token：primitive query 32 + global 1 + curve 1 + mask 5 = 39；K/V = patch token + **4 个 register token（第一版前必做）**；文本 ≤32 走独立 cross-attn；**不压 token**（占 VLM 前向 ~0.5%）。

| 方案 | 规格 | 用途 |
|---|---|---|
| G-Lite ≈5M | d=256,L=4,cross@{1,3} | 第一周 baseline + 无加固对照 |
| **G-Base ≈15M** | d=384,L=6,cross@{1,3,5},文本@{2,4,6} | 主方案 |
| G-Comp ≈18M | +slot-axis softmax（attn map 即掩膜 w）+软掩膜偏置+Group DETR K=4 | 仅塌陷时启用 |

取层 {L/2,3L/4,L} energy-routing（entropy 下界防塌末层）；位置编码 learnable Fourier 加 key；s 进 decoder 用 **E2**（(x,y,s̄) 三维位置编码，零 token 零参），E3 ALiBi 偏置留 v1.5（前 2 层强制全通+温度退火防自锁）；全局 style 走 **ModLN 每子层独立参数**（可插值性是功能，删=回退）；全局量走调制、空间量走 cross-attn，不混。

输出头（单 Linear 零初始化，禁堆 MLP）：
```python
μ_i = anchor_i + r_i*tanh(zμ)          # anchor 来自 Stage-0 k-means
s_i = 0.02 + 0.48*sigmoid(zs)          # ★禁裸 exp（GRM: sigmoid vs exp = +3.08 dB）★
M_i = I + 0.1*zM; b_i = 0.1*zb; o_i = sigmoid(zo−2); g_i = sigmoid(zg+4)
G = 0.1*zG; c = 0.1*zc                 # ★G 初始 0 非 I，否则 f(x)=2x★
# CI: max ΔE00(f(x),x) < 1e-4 over 33³
```
主损失放 **LUT 立方体空间**（置换不变，绕开匈牙利匹配）；代价是丢一半去重压力 → 锚点初始化 + query 直接监督（Mask2Former：可学但不监督=没改）补。

### 1.5 统一读出头（腿 A 兑现）

废"末层→3 latent"。第 L 层出三路：`P_col`（→ModLN 色彩条件）、`P_sem`（patch→6 语义基）、`P_w`（→基底权重）。消融行"L=末层且只留 P_col"= 现状 VeraRetouch，天然存在。

**S4 CGLUT 条件源替换三步**（改点精确）：
1. 可学查表 `E∈R^{L×64}` → `e = Proj(VLM_embed(instr, img))`（Linear→64 维，49K；下游一字不改）。副产品：**可泛化到未见指令**（原版做不到）。
2. 新增 s 轴 head（R-1/R-5 路线 +8K；R-2 路线 +20K）。
3. **混合模式（论文没试过的中间档）**：RGB 几何走 Shared Geometry（跨指令共享→服务跨图刻度），s 轴参数+payload 走 Full Generation（服务 PSNR）。GLUT Table 8 只试过两个极端（44.02 vs 49.84），中间档是干净增量。

### 1.6 可导出的唯一活口（措辞照抄，其余全是雷）

一手核实：.cube 规范只许 1D/3D 表且关键字白名单直接报错；OCIO 全部 fileformats 无一 4D 读写器；Vulkan 无 4D 纹理；ICC 的 4D 是设备色料通道（偷换语境一戳就破）；4D LUT 论文零篇幅谈导出。
> **可以写**：训练与推理在 4D (s,RGB) 连续空间；**交付物是标准 3D LUT**——s 固化后沿第四轴切片、重采样均匀 33³、导出 .cube/CLF（引 SMPTE ST 2136-1，勿引 S-2014-006）。保留空间自适应时交付"N 张 3D LUT + 引导图"由 OFX/DCTL 合成——**运行时交付，非标准格式交换**。
> **绝不说**：「导出 4D LUT」「导出 N 维表」「ICC 支持 4D」。
额外两雷：各向异性高斯烘焙必须重采样均匀网格（损失单独量化上报）；训练用高斯加权和 vs 宿主用四面体插值——**烘焙评测必须含"目标插值算子回读"**。

---

## 2. 备选方案库

### 2.1 渲染器候选 R-1..R-11（全表）

| # | 名称 | 机制 | 解决 | 参数(N=32) | 难度 | 失败判据 |
|---|---|---|---|---|---|---|
| **R-1** | 零初始化 s-门控载荷调制 | `f_i=(1+γ_i(s))⊙(M_ix+b_i)+β_i(s)`；Fourier(16)→共享MLP→⊙z_i(16)→[γ;β]，输出层全零 init；**w_i 不动** | R1 结构性免疫 + 退化保证 | 2.4K | 低 | L1 in-mask <+3dB；或 Var_s[γ]<1e-4（调制学成常数） |
| **R-2** | 锚定 4D 高斯 | μ_s 不可学 K=6 网格；σ_s∈[0.025,0.30] sigmoid；**保留 s–RGB 交叉协方差**（不可因子分解） | R1 结构性封堵 | 844 | 低 | Δ_shuffle→0；或 L0 掉>0.05dB |
| **R-3** | 参数空间多样性 hinge | 沿**高斯 index 维**：`relu(γ_μ−std(μ_s))+relu(γ_λ−std(logλ))`，γ_μ=0.15 | R1（补 R-2 缺口） | 0 | 低 | hinge 常年 100% 激活；或几何分散但载荷全同→转 R-4 |
| **R-4** | 低秩 s 依赖载荷分解 | 几何只建 RGB；`M_i(s)=M_i⁰+Σ_k β_k(s)M_i^k`，β_k 为 s 上 B 样条基，k≥1 零 init | R1+表达力上限 | K=3→1868 | 中 | K 1→3→5 提升<0.3dB；或 train/val 差>0.5dB |
| **R-5** | DoRA 式分解，s 只调 magnitude | `M_i=m_i·V_i/‖V_i‖`；V 与 s 无关，`m_i(s)=m_i⁰(1+Δ_i(s))` 零 init。**副产品：m_i 直接读出"模型认为效应量多大"** | R1+可解释 | +1.5K | 低 | 与 R-1 同预算：R-1 高>2dB→方向必须随 s 变（此结论本身有价值）；差<0.5dB→选 R-5 |
| **R-6** | RGB-边缘归一化 + 未归一化 s 门 | 分母只对 RGB 求和；`g_i(s)=exp(−(s−μ)²/2σ²)` **去前置常数**。s 依赖数学上不可能被对消 | R1（唯一公式级证明） | 780 | 低 | L0 掉>0.6dB（GLUT 消融上限 0.30 的两倍）→回退 |
| **R-7** | GECO 拉格朗日自动配权 | Δ_shuffle 可微代理为约束，β 自动调 | 化解"强制敏感 vs 优雅退化"矛盾 | ~10 行 | 低 | β 钉上界仍不满足→架构问题回 R-2/R-8 |
| **R-8** | **s 轴软分桶多基底**（PSNR 最可能赢） | 取消 σ_s：B=6 软基 π_b(s)，B 组载荷共享 RGB 几何，**s 只进分子** | R1 从根消失+退化自然可达 | 2316 | 中 | B 组载荷两两距离<5%（问题在 R2/R3 非 R1）；B=8 带状伪影 |
| **R-9** | 逐图单调 s-warp（AdaInt 式） | 图内 CDF + 小头预测 K=8 锚点 cumulative-softmax 单调重标定 | **R2** | 1–2K | 低 | 掩膜内 s 中位数跨图方差不降（排序问题非刻度问题） |
| **R-10** | 掩膜监督+参数扰动 loss | (a) `BCE(Φ(s),mask)` 构造样本；(b) s 扰动要求输出变；(c) 惩罚 ∂f/∂s≈0 | R1+R2，**性价比最高** | ~0 | 低 | 加了仍 σ_s 增大 s 响应趋零→塌陷结构性，换 R-6/R-8 |
| **R-11** | **4D quadrilinear LUT 对照臂（必做控制实验）** | s 做第 4 格点轴 17³×5，TV 正则 λ→∞ 退化 3D | **归因**：s 信息 vs 高斯参数化分开 | 73.7K | 低 | **相对 3D <0.1dB 且 in-mask <0.3dB → 立即停渲染器全部工作，转修 s/数据** |

**组合建议**：第一版 = **R-1+R-2+R-3+R-7+R-10**（≈2.5K 参数）。**R-1 vs R-5 同预算对照是核心科学问题**（"s 要改变换方向还是只改强度"）。R-8 等 s 通路确认有效后启动。**R-11 与 R-9 分别是 H5 与 R2 的判决器，第一周必跑**。

### 2.2 读出候选 D-0..D-10（按"当天可跑"→"需训练"）

| # | 名称 | 机制要点 | 训练 | 失败判据 |
|---|---|---|---|---|
| **D-0** | 伪影三件套（公共前置必做） | test-time registers 承接高范数激活；norm>median+3·MAD 剔除插值；16px 周期功率谱检查→DVT | 零 | outlier 不降且 s 场无差别→此骨干无此病，跳过 |
| **D-1** | self-self 读出（SCLIP/NACLIP/ClearCLIP 三选一 A/B） | 末 attn block 换 qqᵀ/kkᵀ；与指令名词 cos；减全数据集固定背景 prompt 均值。**必做符号 sanity check**（raw attention 可能前景冷背景热）；**归一化用全数据集固定仿射，绝不逐图 min-max/softmax** | 零 | AUC<0.75；或 in-mask 提升<0.3dB |
| **D-2** | logit lens 概率场 | 第 L 层 hidden→final LN→LM head 取目标名词列。**词表概率天然跨图绝对刻度**（R2 最干净解）；层偏深 | 零 | 同语义 200 图中位数标准差>0.15；AUC<0.70 |
| **D-3** | guided-filter 上采样（配套必做） | 32×32→4K 先 guided filter（零参）；不够 JAFAR/LoftUp；最后 FeatUp | 零 | 边界 PSNR 回收<0.05dB |
| **D-4** | VFM proxy attention（ProxyCLIP/Trident/CorrCLIP） | DINO 相似度替换 CLIP 末层 attention | 零 | AUC 提升<0.02→回退 D-1 省算力 |
| **D-5** | LMM 中层 text→image attention + **系统扫层** | **pre-softmax logit**（sink 是 softmax 归一化的产物、不进 value 计算——这正是 R2 的机理来源）；200 张构造样本扫全层全头按 AUC 选 top-k；**先验搜索区间=中层**（FastV：深层视觉注意力已稀疏） | 零 | 最好单头 AUC<0.70；或最佳层落最后两层（读到的是答案聚合非空间场） |
| **D-6** | 冻结 VLM + 轻量探针头 | 最优层 hidden→1×1 conv(→1)+sigmoid（1–4K）或 v1 66–262K；监督三档：GT 掩膜 / D-1/D-2 蒸馏 / 端到端重建 loss | 训探针 | 留出类别 AUC<0.70（背类别）；或端到端后 s 方差<0.05（R1 在探针侧） |
| **D-7** | **学习式 context encoder + VLM 蒸馏初始化（最强 baseline，最可能赢 PSNR）** | conv encoder 出 c(x)；`s=α·s_VLM+(1−α)c` 或蒸馏正则；s_VLM 离线缓存 32×32（2KB/图，训练零 VLM 成本） | 训 encoder 0.1–1M | **决定性消融：随机 init vs VLM 蒸馏 init 差<0.1dB → VLM 读出对 PSNR 无贡献，卖点改可控性** |
| **D-8** | 指令差分 relevance map（离线伪标签厂） | InstructPix2Pix 带指令/空指令去噪差范数。**语义同构**："该被改多少"而非"是不是天空"；差分自动消逐图基线 | 零（离线贵） | AUC<0.80（不如零成本 D-1）；单图>5s |
| **D-9** | Talk2DINO / dino.txt（2026 线） | DINOv3 patch + CLIP 文本 mapping（MB 级） | 训 mapping | AUC 未超 D-1 最优 +0.03 |
| **D-10** | `<EDIT>` token + LoRA + soft logit mask | 扩词表 + LoRA 微调；相似度用未过 softmax 的 logit；L 层可学凸组合。质量上限最高，最后做 | LoRA 4–8M | LoRA 后 AUC 未超 D-1 +0.03→砍掉 |

**三条硬结论**：① pre-softmax 优先（MaskCLIP 干脆 Attn=I）；② attention 读出取**中层**、logit lens 取**深层**，两路分别扫（UGround 用 RL 选层=「固定末层」已被公开否定；具体层号无文献，必须自己扫）；③ **绝不逐图归一化 s**——那是在亲手制造 R2。

第一版读出 = **D-0 + D-1 + D-3 + R-9**（零训练零参数一次 ViT 前向）。

---

## 3. 实验梯（由简到难：核对 → cube → 图像 → VLM）

### 第〇级：开工核对与三个 gate（半天–一天，零训练）

**代码核对 C1–C3（30 分钟）**：
- C1 GLUT 权重有无归一化分母（论文 Eq.2 有/项目页无，**以代码为准**）。无分母 → R-2/R-6 降级可选。
- C2 密度 p_i 有无 `1/√((2π)^d|Σ|)` 前置常数（R-6 的门必须去掉它，否则退化方向反了）。
- C3 初始化改 `G=I,g=0,M=0,b=0`（GLUT 原版 M=I 叠加 G=I 会得 f=2x）。

**Gate D1 · s 可辨识性体检（半天，最便宜最致命，第一个做）**
200–500 图 × 3 指令（2 同义 + 1 对立），冻结 VLM 读出 s。
看：ρ_syn（同义相关）/ ρ_opp / ρ_Y（与亮度相关）/ outlier token 占比 / 跨图组间组内变异比。
**过**：ρ_syn>0.7 且 ρ_opp<0.3 且 |ρ_Y|<0.5。**死**：ρ_syn<0.5；或 |ρ_Y|>0.8（**s 是亮度的马甲**——3D LUT 已隐含亮度，第四维零信息）。

**Gate D2 · oracle 上界（半天）**
不用 VLM：GT 掩膜构造完美 s（3 档），3 份 3D GLUT 按 s 查表 vs (a) 单张 3D (b) 亮度当 s 的伪 4D。
**过**：oracle-4D ≥ +1.0 dB。**死**：<0.4 dB（≈4D LUT 的 0.36）——**s 完美也没空间，停项目**。

**Gate D3 · 塌陷通道验证（半天，玩具规模）**
几百对数据 + 32 高斯 4D，纯重建 loss 训 1h，看 σ_s 轨迹与 s-sensitivity。
σ_s 单调发散 + sensitivity→0 = **零代价逃逸通道被证实**，不上约束不进主训练。

### 第一级：cube 拟合（无图像无 VLM；Stage 0，1–3 天 + ~40 GPU-h）

目的：测 N\* / 金标准 / 锚点 / condition 宽度。**文献空白：无人报过跨 LUT 误差分位**。

1. 4000 .cube 重采样 33³，剔近恒等（ΔE00<0.2）。
2. **训练采样 128³ 均匀格**；评测双口径 = 留出色 + 自然图像像素（ENNELUT：只按自然色训，全 Hald 崩 36.95→25.23）。
3. 逐 LUT Adam 直接过拟合 ~800 参数：μ 由 |f(x)−x| 加权 k-means 初始化，σ 大→小退火（对抗局部支撑的局部极小）。
4. **密度控制必做**：每 500 步死原语重投放到 ΔE 残差最大体素；否则有效原语只剩 15–20，**把优化问题误判成容量问题**。有效原语<0.8N → 先修优化器。
5. 扫 N∈{8,16,24,32,48,64,96,128}：先 400 分层子集（胶片/电影/人像/分离色调/bleach-bypass）全扫，再全量验 3 个 N。
6. **报跨 LUT p50/p90/p99/max**。饱和判据：r(N)=p90(N)/p90(2N)，>1.25 未饱和 <1.15 饱和；**N\*=最小满足 p90<1.0 且 p99<2.0**；N\*>64 → 不加 N 走尾部归因。
7. **尾部归因**：最差 5% 分类；锐转折主导 → A/B `N=32` vs `N=32+1D曲线` vs `N=64`（预期曲线赢）。
8. 产出：N\*、4000×N 金标准、锚点(μ̄,r)、参数边缘分布、死原语统计。
9. 副实验：4000×107811 截断 SVD → 秩-ΔE 曲线定 condition 宽度（三条间接证据预测 32–64）。

**门槛：p50<0.5 / p90<1.0 / p99<2.0，不过停在这里改渲染器。** 同时判定：CGLUT-MLP 已达天花板 90% → transformer 收益上限<10%，预算全转 s 轴+掩膜+曲线。
文献锚点（GLUT Table 9, 75×64³）：N=8→37.01 / 16→41.50 / 32→**45.47** / 64→48.42 / 128→50.31 dB。

### 第 1.5 级：掩膜基底离线拟合（单卡几小时，与第一级并行）

每类 200 张 512²：线性渐变 / 径向椭圆 / 束状楔形 / **环形（专项）** / 语义五类（Mask2Former/SAM on FiveK）/ 全局常数。
固定 φ（14 维 Legendre + 语义残差化），L-BFGS 优化 (w₀,α,w_dir)，损失 soft-IoU，**单调版与带通版都跑**：

| 掩膜类 | 单调 | 带通 | 说明 |
|---|---|---|---|
| 线性渐变 | ≥0.98 | — | 达不到=实现 bug |
| 径向/椭圆 | ≥0.97 | — | 精确可表达 |
| **环形** | ≤0.40 | **≥0.90** | **这对数字=「高斯轴买到了什么」的量化证据，单独成图** |
| 束状 | 0.55–0.70 | 无改善 | **低值是预期不是 bug**（领结一半）；<0.50 才是实现问题。产出=量化后的 limitation |
| 语义五类 | ≥0.85 (k=6) | — | <0.75 → k=8 重跑或查文本基共线 |
| 全局常数 | α<1e−2 且 std<0.01 | — | 退化性检验 |

零成本三条消融曲线：φ 截到 [1,x,y]（=复现 Exposure）量化二次净收益；二次 vs +cubic 四项（**文献真空，本身可发表一节**）；6 / 6+2range / 6+2+k 边际贡献。
条件数检查：单项式 vs Legendre 的 Gram 条件数（目标<10）；残差化前后块外 Frobenius（<0.05）。
**决策规则**：前两类拿不到 0.97 → 设计不往下走；环形对比拿不到 0.9 vs 0.4 → 5D 方案与"s 是坐标轴"的 novelty 都重估。

### 第二级：图像域，oracle s（有图像无 VLM；Stage 1–2）

**合成数据（权威数值表，2026-08-03 内联定稿）**：长边 1024，sRGB [0,1]，`O=(1−m)T₀(I)+m·T₁(I)`。

| 几何族 | 参数采样 |
|---|---|
| Radial | 中心 (cx,cy)~U(0.2,0.8)·(W,H)；r0~U(0.10,0.40)·D；r1=r0+U(0.15,0.60)·D（D=对角线）；`m=smoothstep(r0,r1,dist)` |
| Linear | θ~U(0,2π)；偏移 t0~U(0.2,0.8)；过渡宽度 w~U(0.05,0.50) |
| Elliptical | 半轴 a,b~U(0.10,0.50)·D；旋转 φ~U(0,π)；羽化 f~U(0.02,0.20)·D |
| Vignette | Radial 特例：r0=0.5·D，r1=1.0·D |

| 变换 | 四档幅度 |
|---|---|
| Exposure（线性光域标量缩放） | ΔEV ∈ {0.15, 0.30, 0.60, 1.20} stop |
| White balance（对角阵 diag(1+δ,1,1−δ)） | δ ∈ {0.03, 0.06, 0.12, 0.24} |
| Saturation（YUV chroma 缩放） | k ∈ {0.70, 0.85, 1.15, 1.40} |
| Hue rotation（YUV 平面旋转） | {5°, 10°, 20°, 40°} |
| Tone curve（gamma + S-curve） | γ ∈ {0.80, 0.90, 1.10, 1.25} |
| 困难对照 | 通道独立、3 结点随机非单调分段线性 |

语义掩膜：**自家 l 系 C_GT（slot_id=semantic-\* 过滤，T4 已实现）**为主，PPR10K masks/FiveK×分割为辅；羽化 σ∈{0,2,8,24}px；面积 α∈{5,15,40,70}%。T0=恒等变换（追认 T4 默认）。

**容量阶梯**（s=GT 直喂，隔离渲染器容量与读出质量）：

| 级 | 内容 | 判据 |
|---|---|---|
| T1 | 单样本 overfit | ≥45dB 过（GLUT 45.47/GaussianImage 44.08 校准）；<40 表示/优化缺陷 |
| L0 | 全局 | 与 3D 差<0.05dB；>0.5 判死 |
| L1 | 二值语义 | 距天花板<2dB；in-mask vs 同 N 3D **≥+8dB**，<+3 结构死 |
| L2 | L1 换 VLM 实际 s | 比 L1 低 ≤3dB（**L1−L2=读出代价=R2 效应量**） |
| L3 | 软 matte | 同 L1 |
| L4 | 几何渐变 | 同 L1（**s 轴上几何是难例**：连续统需 ≥3–5 模态；语义双峰极易——难度顺序与图像空间直觉相反） |
| L5 | 径向掩膜×语义 s 错配 | 预期 FAIL（负控制） |
| L6 | 两重叠软掩膜 | 预期 FAIL（秩上限→触发双轴 gate） |
| L7 | 边界横切同色区 | 预期 FAIL（**通过=偷到 (x,y)，是 bug**） |
| T5 | s 轴频率扫描 sin(ωξ) | ω_cut 对应 ≥6 模态，反推 μ_s 网格密度 |

**训练配方**：
- Stage 1（冻 VLM，~1 周）：4000 cube×8–16 图；**输入=LUT 作用后的图**；Loss = `L_cube`(1.0)+`L_param`(0.3)+`L_prequery`(0.2)+`L_prior`(0.01)+每层 aux；AdamW β(0.9,0.95) wd0.05 lr4e-4 warmup2000 clip1.0 bf16；VGG 非 LPIPS。判据：留出 cube p50<1.5/p90<3.0；**方差比>0.6**；死原语<10%。**必做对照：无加固自由回归版**（<0.5dB → 色彩域温和全面简化；≥2dB → 确认加固）。
- Stage 2（VLM 仍冻，2–3 周）：新模块 zero-init gate 接入；**L1/Charbonnier 非 L2**（median mode + 天然滤噪）；+ΔE00+VGG(0.5)+权重 L1 稀疏(0.1)+**采样必含 128³ 均匀格**；存在门课程（前 60% 冻 1，解冻后**误差先升是预期**）；**checkpoint 禁 val loss**（L1 最低=最保守平均 LUT），用 ΔE00 分位+方差比+人评。
- **Condition dropout 必装**（p=0.15 换可学常量 s_∅）：(a) 显式训练退化分支；(b) 免费给出 Δ_const 塌陷探针；(c) 推理端 `y_w = y(s_∅)+w·(y(s)−y(s_∅))` 放大局部效应（仅消融）。
- hard mining **改按掩膜内/外分层采样**（否则 90% 背景像素稀释梯度）。
- 课程：前 3–5k 步 σ_s 冻在 σ_min~2σ_min 强制局部；反塌陷项 cyclical 退火（3–4 周期）。**预期 sudden convergence**（ControlNet 现象），勿过早判死。
- GLUT 自身消融：L_hc+R_sparse 合计只值 0.11dB——**主战场是 PSNR 时先关掉，调参预算留给 s 轴**。

**监控 M1–M9（每 epoch，写进 tensorboard；Δ_const/Δ_shuffle 是唯一一票否决，每个消融行必带）**：

| ID | 指标 | 红线 |
|---|---|---|
| M1 | Δ_const = PSNR(s)−PSNR(s_∅) | <0.05dB 且 loss 仍降 = 已塌陷 |
| M2 | Δ_shuffle | <0.3dB 一票否决 |
| M3 | σ_s 贴上界比例 | >80% |
| M4 | s 边缘分布两两散度 min | <0.05 |
| M5 | I(bucket_s; gaussian) | 上升但 M1 不动=指标被刷 |
| M6 | mean\|∂f/∂s\| | 单调趋 0 |
| M7 | masked PSNR 三分（内/边界带/外） | 边界带低>5dB 且羽化不改善=高斯表示在 s 轴无法表达锐边界，**真实局限写进论文** |
| M8 | clamp 触发像素比例 | >5% |
| M9 | R-8 专用：B 组载荷两两距离 | <5% |

**效应量解析放大**（为何主指标必须 masked PSNR）：掩膜占比 α 时 in-mask/overall 放大 (1−α)/α；α=0.15 → **5.67×=7.5dB**；实证 PA-LUT 全图 +0.24 vs 人像区 +1.24。**主指标=掩膜内+边界带 PSNR，全图 PSNR 降为附录；配对 ΔPSNR+bootstrap CI。**

**烘焙-导出损失预算（W3）**：N_s∈{2,3,5,9} 切片→均匀 33³→.cube→**四面体插值回读**→4096² Hald 上 PSNR+ΔE，损失拆项（切片/重采样/插值失配/量化）。过：ΔE<2 且 <0.3dB。死：ΔE>4 或需 N_s≥17 → 可导出叙事改运行时插件交付。

**负控制（W4）**：随机 s / 置换 s / 亮度 s / 加噪 s。过：随机置换明显掉点且加噪掉幅<0.8dB（对齐 DY-LUT 0.763）。死：随机 s 掉点<0.1dB=轴是装饰品。

### 第三级：VLM（探针 → 读出 → 端到端）

**腿 A 颜色探针（1 周，不重训）**：
- **全部配对 delta**（同图已知扰动，探两版之差）。属性：A1 ΔCCT(mired) / A2 ΔWB / A3 ΔEV / A4 Δ对比斜率（像素基线预期打平，几乎零信息量）+ **A5 肤色记忆色偏差 / A6 gray-world 陷阱光源色温（命门：需语义才算对；Cube+/NUS-8 有 GT illuminant）**。
- 探针阶梯 P0→P4 报 Pareto 曲线；指标 R²/MAE + **MDL** + selectivity；**取点 layer×module 双扫**（ViT 每 block 三位点 / connector / LLM 每层三位置 / **control latent 单独一路** / **sink-register token 单独一路**）。
- 对照 C1 标签置换 / C2 随机骨干 / C3 token 置换 / C4 纯文本+灰度 / **C5 像素统计上界基线（A5/A6 上相对提升 ≥20% 才算 VLM 增量）**。
- 行为侧 5 档取最好：数值 / 多选 / CoT / **B4 softmax-logit 期望**（最难打败，本身即"换读出口"最便宜版）/ self-consistency。
- **预注册阈值**：selectivity≥0.25 且 R²≥0.60（LLM 末层 last-token 也成立）；行为 MAE≥2× 探针（CI 下界>1.5×；CCT 探针≤15 mired 行为≥30）；A5/A6 对 C5 ≥20%。
- **因果闭环**：INLP 投影掉色温方向 + activation steering 剂量-反应曲线（Spearman≥0.9，reverse/random 双对照）。**干预对象=第 L 层整段 image token 序列，不是 last token**（VLM 里 last-token 干预无效应，image tokens 携带几乎全部因果影响——按 LLM 惯例做会假阴性）。

**质疑预案 A–E（审稿攻防表，逐条备好）**：
- A「探针在算不是模型在编码」→ C1–C5 + MDL 学习曲线（真信息小样本区就领先）+ 跨分布零样本迁移（真实→合成+ColorChecker，R² 保留≥70%）+ 因果闭环。
- B「线性探针容量大」→ Pareto 全曲线；主张只下在 P1(≤10k)；同时引 Hewitt&Liang + Pimentel + Zhu&Rudzicz 三边，站哪派都打不动；正则用 wd 不用 dropout。
- C「你探的是视觉编码器不是 LLM」→ 画完整逐层曲线 pixel→ViT→connector→LLM→latent→输出，**R² 在哪一段塌就是结论本身**；已知两个反例（adapter drop-off 后 LLM 层恢复；Qwen2.5-VL 的 LLM 层反而提升探针）→ 只测两端会得反结论，必须全程。
- D「涨点来自渲染器容量不是捞回信息」（**最致命，一定被问**）→ D1 随机同维噪声条件对照；D2 2×2 消融；D3 **同骨干同渲染器三路 condition**：(a) AceTone 式离散 token 头 (b) 末层 pooled (c) 第 L 层中间特征——不做这个，"读出通路丢信息"就是未证伪口号；D4 按属性分报 ΔE/CCT/EV 不只 PSNR。
- E「ColorBench 说 LM 更重要」→ 区分色彩物理量估计（编码器可线性读）与色彩语义推理（ColorBench 主测）；其附录 K 本就站我方。

**sink/读出诊断流程 S0–S8（顺序不可换，S0 是闸门）**：
S0 读出上限探针（ViT 输出 R²<0.6 → 编码器盲区，停 sink 分析走 MoF）→ S1 token 范数分布（决定性判据：极值跨图恒定=固定偏置非信息）→ S2 注意力质量分布挑 image-centric heads → S3 跨图平均图分诊（规则网格→DVT；高范数落背景→test-time registers）→ **S4 因果验证禁止只用置零**（四组：置零/跨图替换/均值/噪声；register 置零 −36.6pp 而三种替换 ~1pp——只做置零必假阳性；(b) 也掉=sink 存了 image-specific 信息→策略从剔除改**显式读出**）→ **S4.5 sink token 色度探针（最有卖点一步）**：sink/CLS/patch-mean 三方对比，sink 上 R² 最高 → 全局色彩信息正存放在 CLS/mean 会削弱的子空间 → **直接把 sink token 接到 condition 端**，这是"有道理"到"可发表"的最短路径 → S5 逐层探针曲线 + tuned lens（勿 logit lens，brittle；中层达峰后跌=读出通路丢信息直接证据）→ **S6 control latent 容量检查（最可能的真凶）**：有效秩/participation ratio/类内噪声比 + **latent 数量 sweep(1/4/16/64) 画"预算 vs 可达 ΔE"曲线**（4 个就饱和→瓶颈在优化不在带宽）→ S7 模态竞争（真图≈无图→text inertia→PAI）→ S8 位置衰减（RoPE decay→CCA）。
修复清单按代价：读出层（logit 期望/离散 levels/结构化 token 头/峰值层/多层融合）→ 注意力重分配（VAR/PAI）→ 结构性（test-time registers/PH-Reg/DVT）→ 架构（sigmoid attention）。
纪律：三类失败先分清——**编码器没编码（MoF）/ 注意力砸偏（B/C 类修复）/ 读出头形式错（A 类修复）**；修图 VLM 最可能第三种（最便宜也最少被检查）。

**腿 A' 空间侧**：B2 空间探针逐层扫 + D-5 全层全头 AUC 扫描；B3 **H3 判决图**（颜色/空间两曲线同图，峰距≤1/4 深度=统一读出层成立）；B4 **有名词/无名词指令对比**（GL token vs CLIP vs GEM；无名词半集 CLIP 结构性崩塌而 GL token 有结构 → VLM 不可替代性被证明；不崩→退回"w 决策不可替代"论证）。

**读出接入与端到端**：读出层 L 由 H3 定；统一读出头接 G-Base；先 S-a（s=g(RGB) 可折回单张 .cube）后 gate 决定 S-b（掩膜基底 s，交付 N 张 LUT+guide）；S-b gate：oracle 真值 mask 下 4D vs 3D <0.3dB 不做。L2−L1 报读出代价。**Stage 3 解冻 VLM 默认不做**（先 oracle 诊断 gap；若做：LP-FT、1/10 lr、**优先解冻前几层**（corruption-type shift）、连续损失 stop-grad + 离散 CE 辅助防遗忘）。

---

## 4. 评测集构造

### 4.1 分层法：残差分层 → **差分分层**（残差高有四个混淆源：噪声/高频编辑/JPEG/非色彩算子，会系统性过选）

```
Step 1 预处理：sRGB 512 长边；ECC/SIFT 配准检查，残差超阈丢弃
Step 2 3D 天花板：33³ 直方图分箱 + **逐箱最小二乘仿射**（局部线性 = 三线性 3D LUT 的格内真实行为；D-24 裁定的权威口径。字面的「逐箱条件均值」低估 3D LUT 能力——33³ 箱宽 ≈7.7 个 8bit 级导致 PSNR 硬地板 ≈41 dB，改仿射后地板 63.43 dB。三道控制臂证明不虚高：全局对照 0.040 dB / L5 错配 0.16 dB / Δ_const=0）；n<20 的 bin 并到 17³
Step 3 4D 天花板：oracle s 三来源——构造集 GT / PPR10K 人像掩膜 / FiveK 用 SLIC-50 或 SAM region 分段常数场（该粒度分割式语义轴的上界代理）；17³×8 s 桶
Step 4 Δ_ceil = PSNR_4D_ceil − PSNR_3D_ceil；top 20% = 局部子集，bottom 20% = 全局子集（验收退化，要求 Δ<0.1dB）
Step 5 混淆过滤：(a) 残差图 Moran's I < 0.3 丢弃（真局部编辑空间成簇，噪声空间白）；(b) 3×3 blur 后 MSE 降>60% 丢弃（细节/噪声非色彩编辑）
Step 6 带宽稳定性：33³ 与 17³ 双算，Spearman > 0.8
Step 7 报告：FiveK vs PPR10K 的 Δ_ceil 直方图。文献预测 FiveK 中位数 < 0.3dB —— 若如此，这本身就是 R3 成立的定量判决，主战场移构造集+PPR10K
```

### 4.2 三层交付 + D 层

| 层 | 内容 | 用途 |
|---|---|---|
| A 构造集 | L0–L7 每级 ≥2000 张含 GT | 唯一能测容量；退化子集保留验收 |
| B PPR10K | 人像掩膜真实集 | 报 **HRP** 非全图 PSNR；**GLC 当 R2 免费探针**（s 刻度漂移时同组一致性先于 PSNR 报警） |
| C 分层子集 | FiveK top-20% | 真实数据不掉点；**别指望大数字**（历史局部 context 只 +0.5dB / 4D LUT +0.36dB） |
| D 合成语料（可选，R3 根本解法） | 用修图 VLM 大规模合成"空间变化色彩编辑"语料（Hist2Style/InstantRetouch 做法，须引它们） | 让局部编辑成为训练分布主导成分 |

**泄漏指标**：`‖(Î−X)⊙(1−R_gt)‖`，**R_gt 必须人工标注意图区域**（compose-through-mask 下模型自身 M 的泄漏恒 0）。定义借 GIE-Bench non-targeted preservation。
**三层评价协议**：L1 渲染层（唯一否决权）/ L2 语义加权（PPR10K HRP 式给指标加权，不给场打分）/ L3 场诊断（IoU 只作描述统计并引 RSFNet"对齐度高渲染反而差"防误读）。
**效率口径 amortized**：一次读出 + N 次零成本重渲染，显式拆开报（Kim et al. ICCV25 警告：s 场生成开销远超 GLUT 本体 0.49 GFLOPs，不拆开报"compact"卖点会被打掉）。

---

## 5. 消融总表（论文主表 + 附录全集）

**主表（死线级，缺一被拒）**：
1. **三档轴消融**：无 s / 端到端自学 s（R-11 或 D-7 随机 init）/ VLM s——缺中档必被问"随便一根轴行不行"。
2. **D-7 决定性消融**：context encoder 随机 init vs VLM 蒸馏 init（<0.1dB → 卖点从 PSNR 改可控性）。
3. **读出口消融**：L=末层+3 latent（=VeraRetouch 现状）vs 第 L 层统一读出。
4. 每行必带 **Δ_const / Δ_shuffle** 列。
5. 退化验收：全局子集与 3D 基线差 <0.05dB。
6. **方向性证明**：固定 s 扫 RGB，不同 s 切片变换形状不同（打"3D LUT+alpha"）。

**附录全集（按模块）**：
- 基底：φ 截断 [1,x,y]（=Exposure）/ 二次 vs +cubic / 6 vs 6+range vs 6+range+sem / k∈{4,6,8} / Legendre vs 单项式条件数 / 残差正交化开关 / guided filter 在基通道 vs 在 s / ψ 固定 vs 可学 / w 规范分解 vs 朴素回归。
- 渲染器：R-1 vs R-5（**核心科学问题：s 改方向还是只改强度**）/ R-2 交叉协方差 vs 块对角 / 1D 曲线开关 / 存在门开关 / N∈{16,32,64} / 单轴 vs 双轴 5D / S-a vs S-b / R-8 vs R-1+R-2。
- 生成器：G-Lite vs G-Base / 无加固自由回归 vs 加固 / 稠密 cross-attn 屏蔽（<0.15dB=装饰品删 15M）/ ModLN 屏蔽（风格插值跳变=可插值性丢失）/ register token 开关 / 三层 routing vs 末层 / E2 vs E3 / WTA M=4 vs 单头。
- 读出：D-1 三算子 A/B / pre- vs post-softmax / 中层 vs 末层 / D-0 伪影修复前后 / 逐图归一化 vs 全局 CDF vs R-9 warp（四档 A/B，W2）。
- 探针：P0–P4 Pareto / C1–C5 全对照 / 行为 B1–B5 全档。
- 训练：L1 vs L2 vs Charbonnier / 匹配-free vs 匈牙利（可选验证）/ 存在门课程开关 / condition dropout p∈{0,0.15,0.3} / hard-mining 分层 vs 原版。

---

## 6. 失败分支树（每个 gate 断了走哪）

```
Gate D1 s 可辨识性 ─死→ s 是亮度马甲/对同义都不稳 → 换读出（D-2 logit lens 绝对刻度 → D-6 探针）→ 仍死 → s 只能学（D-7 纯学习式），VLM 叙事降为初始化
Gate D2 oracle 上界 ─死→ 完美 s 也 <0.4dB → 数据里没有局部信号 → 转构造集+PPR10K+D 层合成语料重造数据；仍死 → 停 4D，转 Laplacian 高频分支（§7）
Gate D3 塌陷通道 ─证实→ R-2+R-3 必上；朴素版全部作废
Stage 0 cube 拟合 ─p90 不达标→ 尾部归因：锐转折 → +1D 曲线；仍不行 → N=64；仍不行 → 换基元（R-8 软分桶 / 4D quadrilinear / bilateral grid 系）
    └─ CGLUT-MLP 已达天花板 90% → transformer 改造停，预算转 s 轴
第 1.5 级 基底拟合 ─前两类 <0.97→ 实现 bug（正交化/归一化）；环形对比不成立 → 5D 与 novelty 主张重估
L1 in-mask <+3dB ─→ R-8 软分桶；L4 差 >5dB ─→ 几何头补救（DeepLPF 式解析参数头 ~50K，s_geo 第二轴）
R-11 对照臂 <0.3dB ─→ 立即停渲染器全部工作，修 s / 修数据
L2−L1 >5dB ─→ R2 是主瓶颈，主攻 D-6/R-9；R-9 后跨图方差不降 → D-2 → D-6
D-5 扫层最佳层在最后两层 ─→ attention 读出不可用，走 logit lens / 探针
H1 证伪（探针≈行为）─→ 颜色腿删，只留空间腿，方法不受影响
H1 只在 LoRA 后成立 ─→ 措辞改 pathway-training deficit，禁写表征坍缩
S4.5 sink 探针 ≈ 随机 ─→ sink 假说证伪，叙事改用 S6（latent 带宽）
S6 latent sweep 4 个就饱和 ─→ 瓶颈在优化/对齐不在带宽，重写 Diagnosis 段
H3 两峰不重合 ─→ 双层读出，故事弱一档方法不变
D-7 消融 <0.1dB ─→ PSNR 卖点放弃，全文改可控性/可解释性/零样本指令泛化（W1 的可控性指标顶上：指令跟随分/pointing game/同义一致性，对标 SA-LUT H-Corr +10% 量级）
PPR10K HRP <0.2dB ─→ 真实数据无效，回构造集主战场并诚实说明
```

---

## 7. 论文外支线（高杠杆，不受论文叙事约束）

1. **Laplacian 高频分支**（LLF-LUT++ 路线，TPAMI 2025，arXiv:2510.11613）：HDR+ 上 +2.64dB，而 LUT 家族内换结构历史只 +0.1–0.5dB。**逐像素色彩映射对高频细节有硬天花板；要把 PSNR 跑上去，加这条分支的性价比高于任何渲染器替换**。与 4D 化正交，第二周并行。失败判据 <0.3dB（也说明构造样本过于理想化、纯色彩无高频差）。
2. **4 维 culling 换血**：欧氏预筛在 4 维失效（量纲不同+各向异性）。换 N-D Gaussians 的 LSH 随机投影（k=4–8 投影，3σ 保守剔除，~3× 提速）。**副产品：s 轴剔除比例恒 0 = σ_s 涨爆的免费 R1 诊断量**。
3. **涌现分析（原始发现的归宿）**：pre-SFT（Qwen3-VL 基座）vs post-SFT 的读出图对比 + 指令条件性检验（同图 4 条指向不同区域的指令）+ DiffLMM 图降级为动机图。若零 mask 监督下涌现坐实，独立成 analysis 一节：READ/UGround 全在 BCE+DICE 下训，「无空间监督涌现」无人占。
4. **S4.5 sink token 色度探针**：若 sink 上 R² 最高，"全局色彩信息存放在 CLS/mean 会削弱的子空间"可独立成短文（interpretability venue）。
5. **可移植性实验**（(s,RGB) 变换的跨图迁移）：同一组高斯搬到新图重算 s；LUT 风格插值在 s 轴上的表现；对标 StatLUT 的 spatially-agnostic 主张。
6. **视频延伸**：全局 LUT 换 s 轴后时序一致性是否保住（LumiVideo/NLUT 的"全局=解析保证"一旦空间可变即失效且无人处理——是下一篇的口子）。
7. **D 层合成语料放大**：用修图 VLM 造空间变化编辑语料（Hist2Style 范式），把局部编辑从长尾变主导——R3 的根本解法，也可反哺 VeraRetouch 主线。
8. **R-8 软分桶多基底**：即使主线用 R-1/R-2，R-8 作为"取消塌陷通道而非拉锯"的结构性方案值得单独探索（预期 L1–L4 +2–5dB）。

---

## 8. 排期（gate 式，含当天三个数）

**当天必须产出**：`Δ_ceil = X dB / MLP 探针(4→256×4→3 单样本 overfit) = Y dB / 朴素 4D 的 Δ_shuffle = Z dB`——这三个数决定后两周全部方向。

| 时段 | 实验 | 通过判据 | 失败走向 |
|---|---|---|---|
| D0 (8h) | C1–C3 核对 / 合成生成器 / Tier-0 Δ_ceil / MLP 探针 / 朴素 4D Δ_shuffle | Δ_ceil≥8 / 探针≥45 / — | 见 §6 分支树 |
| D1 | **R-11 对照臂** L0–L4 | in-mask ≥+3dB | <0.3dB 停渲染器 |
| D1–D2 | R-1+R-2+R-3+R-7 组合，oracle s，L0–L7 全跑 | §3 判据全过 | L1 弱→R-8；L4 弱→几何头 |
| D2 | T5 频率扫描 | ω_cut≥6 模态 | 加 N/K 或 R-8 |
| D3 | **R-1 vs R-5 同预算** + masked PSNR 评测器 | 差<0.5dB 选 R-5 | R-1 高>2dB=方向必须随 s 变 |
| D4 | T3 退化验证（100% 全局数据） | ±0.05dB | 查零初始化 |
| D4–D5 | 读出 D-0+D-1 三算子 A/B + D-3 | AUC≥0.80 | <0.75→D-4 |
| D5 | L2（VLM s） | L2−L1≤3dB | >5dB 第二周主攻 D-6/R-9 |
| D6–D7 | 分层脚本 FiveK+PPR10K | PPR10K 人像区 Δ_ceil 显著 | 也<0.5dB→语义轴在真实数据不成立，重审题 |
| D7 | R-9 s-warp 四档归一化 A/B（W2） | 跨图方差降≥30% | 不降→D-2 |
| D8 | D-2 logit lens | 刻度稳定性<0.15 | ≥0.15→D-6 |
| D8–D9 | D-5 全层全头扫描 | AUC≥0.75 且中层 | 末两层→不可用 |
| D9–D10 | D-6 探针三档监督 | 留出类 AUC≥0.70 | s 方差<0.05→加 R-10 扰动 |
| D10–D11 | R-8（若未贴天花板） | +2–5dB；M9>5% | M9<5%=问题在 R2/R3 |
| D11–D12 | S4 CGLUT 条件源替换 | 全局掉≤2dB；未见指令>最近邻 | 掉>2→保留可学残差 emb；无泛化=VLM emb 只是检索 |
| D12–D13 | PPR10K HRP+GLC | ≥+0.3dB CI 不跨 0 | <0.2 回构造集 |
| D13–D14 | **D-7 决定性消融** | ≥0.3dB | <0.1→卖点改可控性 |
| 并行 W2+ | 颜色探针 B1 全套 + sink 诊断 S0–S8 + H3 判决图 + 有名词/无名词 | 预注册阈值 | §6 分支树 |
| 并行 W2+ | Laplacian 支线 / W3 烘焙预算 / W4 负控制 | ΔE<2 | 改运行时交付 |

---

## 9. 撞车、措辞红线与引用卫生

### 9.1 撞车定位（前两篇必读全文）

| 风险 | 论文 | 区分点 |
|---|---|---|
| 极高 | InstantRetouch (CVPR26, 2606.05071) | 显式可编辑 836 参基元 vs 隐式网格；s=冻结 VLM 语义读出可干预 vs 端到端 guidance map。**必造它必败切片**：同亮度异材质/多人单人/发丝背景。对齐 iRetouch 指标协议 |
| 极高 | PerTouch (AAAI26, 2511.12998) | 它 VLM 利用≈0（soft hint 进 ControlNet）、diffusion、SAM1。引它消融原话 "leading to spillover effects" 而全文无分区度量=问题存在的证人 |
| 高 | 4D LUT (2209.01749) | **"给 LUT 加第 4 轴"绝不能当新颖性（2022 做完）**；新颖性=轴来自冻结 VLM、指令条件、可控可解释，须 D-7 数字支撑 |
| 高 | SA-LUT (2506.13465) | 同路线 2025 版；它证明"自建 benchmark 当贡献"是被接受的做法 |
| 中 | Hist2Style / INRetouch / NamedCurves(+) / AceTone / Kim ICCV25 | INRetouch 的原话可作动机句但重比须公平设置；NamedCurves 的 "mimic spatial editing" 措辞=GLUT 组已意识到纯色域做不出空间编辑（positioning 强论据，但他们可能在推进后续，**窗口以月计**）；AceTone 证明 VLM→参数接口 CVPR 级可行但全局、其 Limitation 自认局部不行=我们的口子；Kim=效率必须拆开报的警告 |
| — | READ/UGround（读出写 method 不写贡献；SasP/MasP 接我们渲染器当 baseline）；MetaCanvas（差分=下游低 4–6 量级）；GLUT（上游，引用不主张）；Exposure/DeepLPF/RSFNet（掩膜是 lerp 权重 vs 我们的查表坐标轴）；StatLUT（正面回应：归一化加权和结构上无几何形变，列主对比）；Seeing or Knowing 2607.26326（语义色类别+重建 vs 连续物理量+探针+驱动算子） | |

### 9.2 定位陈述（一句话版）

> 我们不主张"给 LUT 加一根轴"新颖（4D LUT 已完成）；我们主张：**这根轴可由冻结的 instruction-conditioned VLM 稠密读出，使色彩变换的空间局部性可指令控制、可解释、可干预**，且在 ~836 个显式参数（vs 4D LUT 924.4K）内做到。为此必须回答三个无人系统处理的问题：归一化高斯混合在新增轴上的**塌陷通道**及结构性封堵、外生标量场的**跨图刻度对齐**、**局部编辑效应量的可测量性**。

### 9.3 禁写清单

首次高斯基元色彩变换（GLUT）/ 首次参数化可微掩膜（Exposure+DeepLPF）/ VLM 生成 LUT（AceTone，其语料含 ~10k licensed cube，**我们的 4000 cube 只能当测量工具不能当新颖点**）/ transformer 生成 LUT（StatLUT）/ 重生成器+轻算子防漂移（InstantRetouch）/ 首次 token 相似度读出（READ）/ "semantic lattice"词组（GCPR19 占名）/ 首次 LUT 第 4 轴（4D LUT/SA-LUT）/ 首个 VLM 局部修图（PerTouch/InstantRetouch）/ 只报全图 PSNR / 无 Δ_shuffle 声称用到语义轴 / 无退化子集声称优雅退化 / 用 load-balancing 论证解决 R1 / "限制 σ_max 消除塌陷"（须同时说 μ_s 锚定+多样性）/ 效率单次延迟口径 / "sink"当机制主张 / BlindTest 当反方（v6 是友军）/ 「导出 4D LUT」。

### 9.4 引用卫生（双向错误都出现过；写作前逐条 export.arxiv.org 复核）

**编造（勿引）**：Token2LUT、LUT-LLM(调色)、AniLUT、"Conic Soft Masks"、"1200 cube PCA"、ICELUT 28.47dB、"SepLUT 4–8 基饱和"、"GS-LRM learnable queries"、DETR 塌陷具体数字、DiffLMM 假标题、Text2LUT、Image-GS 的两个假 ID。
**曾被武断否定但真实**：ENNELUT(2412.15438, AAAI25)、UniLIP(2507.23278)、OminiControl(2411.15098)、JarvisArt、PerTouch。
**ID/归属纠错**：4D LUT=2209.01749｜NILUT=2306.11920｜SA-3DLUT=2108.08697｜READ=2412.17741｜UGround=2510.03853(ICML26)｜Group DETR=2207.13085｜GaussianImage=2403.08551｜GS-LRM=2404.19702｜pixelSplat=2312.12337｜Makansi=1906.03631｜AttentionLut=2401.01569｜CLUT-Net=DOI 10.1145/3503161.3547879｜CLIP Surgery=2304.05653(非 CVPR)｜LLF-LUT=NeurIPS23 2310.17190(非 ICCV)｜Harmonizer=ECCV22｜LIRA=ICCV25 且无相似度图模块｜GLUT/MetaCanvas 无会议｜PPR10K 指标名 PSNR^HC/ΔE^HC/M_GLC｜InstantRetouch 一名两指（2606.05071 vs 2602.17044）。
**未核实标记 `[待核]`**：GLUT 的 G,g 初始化；DeepLPF 椭圆式两路转写不一致（逐篇开 CVF 原文）；Shorten & Murray-Smith 1996；D-5 具体层号。

---

## 10. 速查卡

```
今天三个数：Δ_ceil ≥8dB ｜ MLP 探针 ≥45dB ｜ 朴素 4D Δ_shuffle ≥3dB
三个 gate：D1 s 可辨识（ρ_syn>0.7, |ρ_Y|<0.5）｜ D2 oracle ≥1.0dB ｜ D3 塌陷轨迹
第一版渲染器 = R-1+R-2+R-3+R-7+R-10（≈2.5K 参）
第一版读出   = D-0+D-1+D-3+R-9（零训练）
主指标 = 掩膜内/边界带/掩膜外 PSNR 三分 + 配对 CI；必带列 Δ_const/Δ_shuffle
一票否决 = Δ_shuffle < 0.3dB
止损点：R-11 <0.3dB → 停渲染器 ｜ D-7 <0.1dB → 卖点改可控性 ｜ D2 <0.4dB → 停项目
红线：G init 0 非 I ｜ σ 禁裸 exp ｜ s 轴禁平滑正则 ｜ 逐像素禁 (x,y) ｜ checkpoint 禁 val loss ｜
      禁逐图归一化 s ｜ 基底禁 softmax 竞争 ｜ IoU 禁当优化目标 ｜ 干预对象=image tokens 非 last token
```
