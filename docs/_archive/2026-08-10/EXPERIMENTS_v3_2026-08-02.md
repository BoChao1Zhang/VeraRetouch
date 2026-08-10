# 实验总案 v3：双创新点 · 多臂并行 · 排期（2026-08-02）

> **本文是协议与设计史，不是当前状态表。** 末尾 changelog 保留实验发生时的判断，其中部分已被
> 后续产物修订。当前逐实验 `what / where / status / gate` 以
> [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) 为准；当前综合结论见
> [`EXPERIMENT_RESULTS_CURRENT.md`](EXPERIMENT_RESULTS_CURRENT.md)。

> 取代 `DECISION_TREE_2026-07-31.md` 的单主线结构（该文档保留作 gate 判据参考）。方法细节与文献判据见 `PLAN_v2_local-retouch_2026-07-31.md`；论文全集见 `SURVEY_papers_2026-07-31.md`；**实现档案 `IMPL_DOSSIER_*.md` 由第 9 轮调研生成后回填**（本文标 ⟦档案⟧ 处）。
> 前提：训练资源充足（按 ≥16 卡并行编排）。原则从"单主线+止损"改为：**设计空间摊开成并行臂 → 统一 harness 计分 → 每周淘汰会**。

---

## 0. 两个核心创新点与实验总览

| | 创新点 ① 统一读出 | 创新点 ② 语义轴渲染器 |
|---|---|---|
| 主张 | VLM 表征里同时有 what（颜色）与 where（空间），现有 3-latent 读出口太窄；换一个"第 L 层 → 色彩条件 + 空间基 + 基底权重"的宽读出口 | 色彩变换的查表定义域加一根语义轴 s：GLUT 高斯升 4D + 掩膜基底生产 s；空间可变但参数仍是 LUT 量级、可烘焙导出 |
| 实验臂 | 读出臂 RO-0..RO-9 + 探针臂 PR-1..PR-5 | 渲染器臂 RD-STD..RD-H + 基底臂 MB-1..MB-5 |
| 科学支撑 | probe-vs-behavior gap（H1/H2/H3） | 容量阶梯 + 塌陷数学（H4/H5） |
| 决定性消融 | E21（D-7 蒸馏 init vs 随机 init） | 三档轴消融（无 s / 自学 s / VLM s） |

**多臂并行的关键解耦机制（INF-5）**：每个读出臂把 s 场**离线导出成缓存**（32×32，约 2KB/图）。渲染器臂训练时只读缓存、零 VLM 成本 → 读出臂 × 渲染器臂可以完全独立排卡，交叉组合只是换一个缓存目录。

---

## 1. 共享基础设施（W1 建完，所有臂插拔）

| 编号 | 内容 | 说明 |
|---|---|---|
| INF-1 | **统一评测 harness** | 输入接口 `(render_fn, s_source)`；输出：masked PSNR 三分（掩膜内/**边界带预注册 ±3 px**（k∈{1,3,8} 留附录消融）/外）、ΔE00、LPIPS、**Δ_const / Δ_shuffle**（每行必带）、烘焙一致性（四面体插值回读）、配对 bootstrap CI。跑完自动进总榜。 |
| INF-2 | 构造数据生成器 | L0–L7 八级 ×（幅度 4 档 × 羽化 4 档 × 面积 4 档），每级 ≥2000 张；几何/语义/全局三族；GT 掩膜全保留。参数见 PLAN §3。 |
| INF-3 | 4000-cube 语料 + Stage-0 金标准 | E1 的产出：N\*、逐 cube 金标准参数、k-means 锚点、参数边缘分布。**实现**：colour-science `read_LUT_IridasCube`（R 变最快、reshape order='F'）+ AceTone `convert_luts.py` 统一 32³（**其 apply_lut 有 BGR 翻转约定，先对拍 identity**）；Hald 按 GLUT 协议 128³ 训 / 留出色测（**颜色空间 split 非图像 split**）；ImageMagick `-hald-clut` 对拍验行序（DOSSIER §4.3） |
| INF-4 | FiveK/PPR10K 管线 + Δ_ceil 分层 | FiveK 用 Zeng 官方 480p 成品包（GoogleDrive 含 split txt；**PSNR 必须 round×255 后算**）；PPR10K 只下 360p tif+masks（91GB+56MB），指标必须走官方 `calculate_metrics.m`（**训练权重 5/1 与评测权重 1/0.5 两处口径不同勿混**）；差分分层切局部/全局子集（PLAN §4.1；DOSSIER §4.1–4.2） |
| INF-5 | **s 缓存服务** | 每臂一个目录：`s_cache/{arm}/{img_id}_{instr_hash}.npy` + 元数据（层号/归一化参数）。含 oracle 目录（GT 掩膜）。 |
| INF-6 | 监控面板 | M1–M9 全量（PLAN §3），tensorboard 模板，红线自动报警。 |
| INF-7 | 探针工具链 | 选型定稿：探针+MDL **自实现**（~50 行，配方抄 Hewitt/Voita）；擦除用 **concept-erasure（LEACE）**、steering 用 **repeng**（ControlModel 就地改结构——**先采集后包装**）、全链路干预用 **nnsight**（TransformerLens 零 VLM 支持）、tuned lens 需自写 VLM 训练循环（纯文本 lens 在视觉 token 上不可信）。安装清单见 DOSSIER §六 |

---

## 2. 创新点 ①：读出臂（RO）与探针臂（PR）

### 2.1 读出臂 RO-0..RO-9

每臂交付三样：**s 缓存目录 + 质量报告（AUC、跨图刻度方差、符号检查）+ 接入 RD-STD 后的 L2 成绩**。统一在构造集 L1/L4 + 200 图指令三元组（G1 协议）上打分。

| 臂 | 方案 | 训练 | 要验证的问题 | 数据 | 改动/实现 | 晋级判据 | 淘汰去向 |
|---|---|---|---|---|---|---|---|
| **RO-0** | oracle（GT 掩膜） | 无 | 上界参照 + 隔离渲染器容量 | INF-2 | 无 | 永不淘汰（永久对照） | — |
| **RO-1** | self-self 三算子（SCLIP/NACLIP/ClearCLIP）+ 伪影三件套 + guided filter | 零 | 零训练读出的天花板在哪？三种改法哪家强？ | L1 图 + GT 掩膜 | **最快路径 GEM：`pip install gem_torch`（pin open_clip≤2.24）6 行出第一张图**；SCLIP 改 `clip/model.py::custom_attn`（CSA 仅末层）、NACLIP 改 `set_params`（kkᵀ+高斯邻域偏置，40 行可独立搬）、ClearCLIP/ProxyCLIP 自带 demo.py（vendored open_clip 须置 sys.path 最前）；**符号 sanity check；禁逐图归一化**；收尾 FeatUp('maskclip', use_norm=False)/LoftUp + `kornia.guided_blur`（eps 1e-4~1e-2 归一化域）（DOSSIER §5.1） | AUC≥0.80 | AUC<0.75 → RO-4 |
| **RO-2** | logit lens 概率场 | 零 | 词表概率能否给出**跨图绝对刻度**（R2 最干净解）？ | 同上 + 200 同语义图 | `lm_head(language_model.norm(h_l))`——**必须先过 RMSNorm**；tie 逐 config 核（2.5-VL-3B tied / 7B untied / Qwen3-VL-8B untied）；**Qwen3-VL 的 deepstack [8,16,24] 层视觉二次注入会使 lens 读数突变，解释时必须标注**；视觉 token 定位用 id 151652/151653/151655（DOSSIER §5.3） | 刻度标准差<0.15 且 AUC≥0.70 | → RO-5 |
| **RO-3** | LMM 中层 text→image attention 全层全头扫描 | 零 | 哪一层哪些头有空间场？pre-softmax 能否绕开 sink？ | 200 张构造样本 | prefill 缓存全层全头；**pre-softmax logit**；top-k 可学凸组合。**FA 互斥**：新版 transformers 下 FA2/SDPA 遇 output_attentions 只 warning 返回 None **不回退**——必须 `attn_implementation="eager"` 加载或分析步 `set_attn_implementation("eager")`；视觉塔 forward **明文丢弃权重**，要在 `visual.blocks[i].attn` 挂 hook 用 q,k 自重算（处理 window attention 的 cu_seqlens 与 fullatt_block_indexes [7,15,23,31]）（DOSSIER §5.4） | 融合 AUC≥0.75 且最佳层在中层 | 最佳层在末两层 → 弃（读到的是答案聚合） |
| **RO-4** | DINO proxy（ProxyCLIP/Trident 档） | 零 | 边界质量是不是瓶颈？外挂 DINO 值不值一次前向？ | 同 RO-1 | RO-1 之上加 DINOv2/v3 相似度替换 attention | 比 RO-1 AUC +≥0.02 且 in-mask +≥0.1dB | 回退 RO-1 省算力 |
| **RO-5** | 轻量探针头（1×1 conv 1–4K 参 / v1 66–262K） | 训探针 | 学习式读出的性价比；三档监督（GT/蒸馏/端到端）哪档够？ | 构造集 + RO-8 伪标签 | 最优层 hidden→conv 头；输出限 [0,1] | 留出类别 AUC≥0.70 且 L2−L1≤2dB | 端到端后 s 方差<0.05 → 加 R-10 扰动 loss |
| **RO-6** | context encoder + VLM 蒸馏 init（D-7，**最强 baseline**） | 训 encoder 0.1–1M | VLM 读出对 PSNR 到底有没有贡献（决定性消融 E21 的载体） | 全部 | conv encoder 出 c(x)；`s=α·s_VLM+(1−α)c`；s_VLM 读缓存 | 与 RO-5 打平或更好 | **随机 init 差<0.1dB → 全项目卖点改可控性** |
| **RO-7** | `<EDIT>` token + LoRA + soft logit mask | LoRA 4–8M | 微调能否把上限再抬一截？ | 构造集少量 | 扩词表 + L 层相似度可学凸组合（MasP 式） | AUC 超 RO-1 最优 +0.03 | 砍掉（微调没带来信息） |
| **RO-8** | 指令差分 relevance map（离线伪标签厂） | 零（离线贵） | "该被改多少"的同构信号能否当监督源？ | 构造集全量 | InstructPix2Pix 带/空指令去噪差；**警告：IP2P 的 CLIP 方向过滤 0.2 阈值会误杀微弱色彩编辑**，质量门改 SSIM/DINOv2 下限剔除（DOSSIER §4.4） | AUC≥0.80 | <0.80（不如免费的 RO-1）→ 弃 |
| **RO-9** | **GL token 原生读出**（我们的 special token，最初的发现） | 零 | SFT 涌现的注意力经伪影修复后能否直接用？与 RO-1/RO-3 谁强？ | G1 协议三元组 + L1 | GL token→image token attention，D-0 三件套修复，pre-softmax | 与 RO-1 打平即有故事（emergent 一节）；更强则升主线 | 弱于 RO-1 → 降级为 analysis，不进方法 |

**读出臂内部横评实验**（全臂就位后一次跑）：
- **RO-X1 有名词/无名词分离**（原 P4）：无名词半集上 RO-9/RO-3（VLM 系）vs RO-1/RO-4（CLIP 系）的结构性分离——VLM 不可替代性的证明实验。
- **RO-X2 归一化四档 A/B**（原 E13）：逐图 min-max / 逐图分位 / 全局 CDF / 可训单调映射，对每个晋级臂各跑一遍（归一化方式是臂的属性，不是全局常量）。
- **RO-X3 SasP/MasP 插件对照**：READ/UGround 接到 RD-STD 上当外部 baseline。接入成本已核：SasP = `model/READ.py` 两函数 ~150 行零参数不依赖 SAM（超参在源码 837-839/764 行注释里切换）；MasP = 裸 einsum 点积 + min-max 上采样，`tools/simi_loss.py` 可零改动给任何读出图打分。**红线：UGround 的 PPM RL 选层是死代码（mode2/3/4 提前 return mode1），官方自己也用 --mode=1，勿实现 RL 部分**；接 Qwen 时 patch 网格动态算，勿抄 LISA 的 24×24/255 pad 硬编码（DOSSIER §5.2/5.3）。

### 2.2 探针臂 PR-1..PR-5（科学支撑，不占主线 GPU）

| 臂 | 内容 | 要验证 | 判据（预注册） |
|---|---|---|---|
| PR-1 | 颜色探针逐层扫描（配对 delta；A1–A4 + 命门 A5/A6；C1–C5 对照；行为五档） | H1 | selectivity≥0.25、R²≥0.60、行为 MAE≥2×、A5/A6 对 C5 ≥20% |
| PR-2 | sink/读出诊断 S0–S8（含 S4 四组替换、**S4.5 sink token 色度探针**、S6 latent 数量 sweep 1/4/16/64） | H1 的机制归因 | 见 PLAN §3；S4.5 阳性 → sink token 直接接 condition 端（新臂 RO-10） |
| PR-3 | 空间探针逐层 + H3 判决图 | H2/H3 | 两峰层距 ≤1/4 深度 |
| PR-4 | 因果闭环：INLP + steering 剂量反应（对象=**整段 image token**） | 探针结论的因果性 | Spearman≥0.9 单调 |
| PR-5 | pre/post-SFT 对照 + 指令条件性（同图 4 指令）——**涌现主张的正式检验** | RO-9 的 emergent 叙事 | SFT 后显著优于 base 且随指令变 |

---

## 3. 创新点 ②：渲染器臂（RD）与基底臂（MB）

### 3.1 渲染器臂（全部在 **RO-0 oracle 缓存**下训练打分——先比容量，再换真 s）

| 臂 | 方案 | 要验证的问题 | 改动 | 参数量 | 晋级判据 | 淘汰去向 |
|---|---|---|---|---|---|---|
| **RD-STD** | R-1+R-2+R-3+R-7+R-10 组合（主方案） | 组合拳能否爬完 L0–L7？ | 载荷零初始化门控 + μ_s 锚定 + 多样性 hinge + GECO + 掩膜监督 | ≈2.5K | L 阶梯全过 + Δ_shuffle≥3dB | L1 弱→RD-B；L4 弱→MB 几何头 |
| **RD-A** | R-5 magnitude-only | **核心科学问题：s 要改变换方向还是只改强度？** | DoRA 分解，s 只调幅值 | +1.5K | 与 RD-STD 差<0.5dB → 换用（更省更稳可解释） | R-1 高>2dB=方向必须随 s 变（结论本身可发表） |
| **RD-B** | R-8 软分桶多基底 | 取消塌陷通道（而非拉锯）能多赚几个 dB？ | B=6 软基 π_b(s)，s 只进分子 | 2316 | L1–L4 比 RD-STD +≥1dB | 载荷两两距离<5% → 问题在 R2/R3 |
| **RD-C** | R-4 低秩载荷 K∈{1,3,5} | s 依赖需要几阶自由度？ | B 样条基载荷分解 | 1.1–2.6K | K 增益曲线拐点 | 增益<0.3dB → 秩不是瓶颈 |
| **RD-D** | R-6 未归一化 s 门 | 公式级"对消不可能"值不值 0.3dB 的代价？ | 分母只含 RGB | 780 | L0 掉幅<0.6dB | 回退完整归一化 |
| **RD-E** | R-11 四线性 4D LUT 对照臂 | s 轴信息量的干净读数（归因） | 17³×5 + TV | 73.7K | **<0.3dB → 全 RD 停工，转 RO/数据** | — |
| **RD-F** | 双轴 5D (s_geo, s_sem) + 2×2 全协方差 | 区域码碰撞（L6）是否值得第二根轴？ | 基底双路 + 协方差块 | +~10% | L6 从 FAIL 变 PASS 且单轴 oracle 差距≥1.5dB | 不达 → 单轴定稿 |
| **RD-G** | transformer 生成器 G-Lite→G-Base（vs CGLUT-MLP 生成器） | 生成器容量值多少 dB？无加固自由回归行不行？ | PLAN §1.4 全套；输出头零初始化 | 5M/15M | 比 MLP 生成器 ΔE00 p50 降≥15% 且方差比>0.6 | MLP 已达天花板 90% → 停，预算转 s 轴 |
| **RD-H** | bilateral grid + 语义 guide（plan-B，InstantRetouch 形态） | 若高斯系全败，网格系能否兜底？ | 低分辨率仿射网格，guide=s | ~InstantRetouch 量级 | 仅在 RD-STD/B 判死后启动 | — |
| **RD-I** | N × 曲线扫描 | 图像域上 N∈{16,32,64} × 1D 曲线开关 | — | — | 附录消融 | — |

**渲染器臂内部横评**：T5 频率扫描（每臂跑，s 轴模态数需求可能不同）、T3 退化验证（每个晋级臂必过）、E22 烘焙预算（每个晋级臂必过——**烘焙一致性是一等指标**）、E23 负控制。

### 3.2 基底臂 MB-1..MB-5（s 生产端，离线 L-BFGS 为主，极便宜）

| 臂 | 内容 | 要验证 | 判据 |
|---|---|---|---|
| MB-1 | 单轴 14 维基底（主）：5 几何 Legendre + 2 range + 6 语义 + ψ=3tanh(q/3) + w 规范分解 | 四类掩膜画不画得出 | 线性/径向≥0.97；环形带通 0.9 vs 单调 0.4（**核心证据图**）；语义≥0.85 |
| MB-2 | 双轴（配 RD-F） | 交集与重叠掩膜 | L6 类拟合显著改善 |
| MB-3 | 三次基 flag | 三次到底买不买得到东西（文献真空，可独立成节） | soft-IoU 边际 |
| MB-4 | 语义基来源三选：文本投影+adapter / VLM token k 维投影 / DINO 通道 | 谁的基跨图身份最稳、渲染最好 | 基身份漂移 + 下游 in-mask |
| MB-5 | guided filter 位置：k 基通道 vs 最终 s；上采样器三档（GF/JAFAR/FeatUp） | 边界质量修在哪一层最省 | 边界带 PSNR 回收 |

### 3.3 交叉矩阵（W4 起）

不做全交叉（9×9），用锚定设计：
1. **固定 RD-STD，扫全部晋级 RO**（≤9 行）——读出臂的最终排名以此为准（L2 成绩）。
2. **固定最佳 RO，扫全部晋级 RD**（≤8 行）——渲染器排名在真 s 下复核（oracle 下的排名可能翻转）。
3. 重点交叉单元（预算内加测）：RO-9×RD-STD（emergent 故事线）、RO-6×RD-B（PSNR 冲榜线）、RO-5×RD-F（双轴上限线）、RO-3×RD-A（最便宜组合线）。

---

## 4. 端到端与打榜（联合阶段）

| 编号 | 实验 | 说明 |
|---|---|---|
| E19 | CGLUT 条件源替换（Proj(VLM emb) + Shared-Geometry/Full-Generation 混合档） | 未见指令泛化 vs 最近邻检索 |
| E20 | PPR10K HRP+GLC / FiveK 局部子集打榜 | 配对 CI。**协议变更：iRetouch 451 对确认不可得**（无仓库无 HF，issue 索要无回复）——不再对齐它的数据，改为**复刻它的指标实现**（fidelity=灰度+histmatch 后 SSIM/CW-SSIM/DISTS/GMSD；GPT-4o SC/PQ 走 Step1X-Edit 协议）跑在我方评测集上；外部基准补 **AceTone-Bench-Transfer（HF: Vivre/AceTone-Bench-Transfer）与 PST50（HF: zrgong/PST50）**，两者公开可下 |
| E21 | D-7 决定性消融（在 RO-6 上执行） | PSNR 叙事的最终审判 |
| E24 | D 层语料合成 + 主模型重训 | 部件已核齐（DOSSIER §4.4）：指令层抄 IP2P 700 条种子三元组格式 + 开源 LLM 扩展；schema 用 AnyEdit（含 color_alter/tone 类型）；掩膜层搬 UltraEdit 链（GroundingDINO 0.3/0.25 → SAM ViT-H → 三重质检）；**图对层是我方优势——参数化变换在掩膜内确定性施加，像素级完美 GT，跳过生成模型**（可直接用 Harmonizer filter.py / RSFNet render() 原语）；质量门弃 CLIP 方向分（误杀微弱色彩编辑）改 SSIM/DINOv2 下限；风格语料模板抄 Hist2Style 数值配方（LLM 风格库 → FLUX.1 Kontext 批量 → VGG19 cos>0.5）；现成补充：OmniEdit-1.2M attribute 子集、AnyEdit color_alter、MagicBrush 人工掩膜 |
| E25 | Laplacian 高频支线 | 与全部 RD 臂正交，独立排卡 |
| E26 | user study + VLM-as-judge 可信度校验 | 可控性指标包 |

---

## 5. 总图（多臂版）

```mermaid
flowchart TD
    subgraph W1["W1 · 基建 + 守门（零训练）"]
        INF["INF-1..7 统一 harness / 数据 / s 缓存 / 监控 / 探针工具链"]
        G["G0–G3 四个守门<br/>+ E1 cube 扫 N + E2 基底拟合"]
    end

    subgraph TA["Track A · 渲染器臂（oracle s，互相独立排卡）"]
        RDSTD["RD-STD 组合拳"]
        RDA["RD-A 只调强度"]
        RDB["RD-B 软分桶"]
        RDC["RD-C 低秩载荷"]
        RDD["RD-D 未归一化门"]
        RDE["RD-E 4D LUT 对照臂"]
        RDF["RD-F 双轴 5D"]
        RDG["RD-G transformer 生成器"]
    end

    subgraph TB["Track B · 读出臂（产出 s 缓存，互相独立）"]
        RO1["RO-1 self-self"]
        RO2["RO-2 logit lens"]
        RO3["RO-3 中层 attention"]
        RO4["RO-4 DINO proxy"]
        RO5["RO-5 探针头"]
        RO6["RO-6 context+蒸馏"]
        RO7["RO-7 EDIT token LoRA"]
        RO9["RO-9 GL token 原生"]
    end

    subgraph TC["Track C · 探针臂（不占主线 GPU）"]
        PR1["PR-1 颜色探针"] --> PR2["PR-2 sink 诊断 S0–S8"] --> PR3["PR-3 H3 判决图"] --> PR4["PR-4 因果闭环"] --> PR5["PR-5 涌现检验"]
    end

    INF --> TA & TB & TC
    G -->|"G2 oracle <0.4dB"| DEAD1["停/换数据"]
    G --> TA
    TA -->|"RD-E <0.3dB"| DEAD2["全 RD 停工"]
    TA --> SYNC1{{"W3 末淘汰会：RD 晋级 ≤4 臂"}}
    TB --> SYNC2{{"W3 末淘汰会：RO 晋级 ≤5 臂"}}
    SYNC1 & SYNC2 --> CROSS["W4–5 · 交叉矩阵<br/>RD-STD×全RO ｜ 最佳RO×全RD ｜ 4 个重点单元"]
    CROSS --> SYNC3{{"W5 末定主配置"}}
    SYNC3 --> FINAL["W6–8 · E19 条件替换 → E20 打榜 → E21 终审判<br/>E24 语料重训 ∥ E25 Laplacian ∥ E26 user study"]
    PR2 -->|"S4.5 阳性"| RO10["新臂 RO-10：sink token 直连 condition"]
    RO10 --> CROSS
```

---

## 6. 排期（8 周，≥16 卡）

| 周 | Track A 渲染器 | Track B 读出 | Track C 探针 | Track D 数据/评测 | 里程碑（周五淘汰会） |
|---|---|---|---|---|---|
| **W1** | **A0 从零复现 GLUT（2–3 天，对齐 45.5 dB，先全局+残差分支）**；G0/G3 核对与玩具验证；E1 cube 扫 N（8 卡）；输出头 CI | G1 可辨识性；RO-1（GEM 当天出图）/RO-2/RO-3/RO-9 零训练臂开跑（2 卡） | PR-1 数据构造（配对 delta 生成） | INF-1..7 全部建完（FiveK 480p 包 + PPR10K 360p 下载先行，91GB）；E2 基底拟合（CPU/1 卡） | **A0 对齐 45.5±0.3 dB**（不达标查行序/残差分支）；三个数出齐（Δ_ceil/MLP 探针/朴素 Δ_shuffle）；N\* 定；G1–G3 判定 |
| **W2** | RD-STD + RD-E 对照臂过 L0–L7（4 卡）；T5 频率扫描 | RO-1..RO-4、RO-9 全部出 s 缓存与 AUC 榜；RO-X2 归一化四档 | PR-1 逐层扫描跑完 | Δ_ceil 分层脚本跑 FiveK/PPR10K | **RD-E gate**；零训练读出榜首确定；H1 初判 |
| **W3** | RD-A/B/C/D/F 并行（各 2 卡）；T3/E23 每臂 | RO-5/RO-6 训练（各 1 卡）；RO-8 伪标签厂离线 | PR-2 sink 诊断 S0–S8；PR-3 H3 图 | E22 烘焙预算；PPR10K 掩膜管线验收 | **双淘汰会：RD 晋级 ≤4，RO 晋级 ≤5**；R-1 vs R-5 结论；S4.5 是否开 RO-10 |
| **W4** | 交叉矩阵第 1 轮：RD-STD × 全部晋级 RO（每格 1 卡） | RO-7 LoRA；RO-X1 有名词/无名词；RO-X3 SasP/MasP 对照 | PR-4 因果闭环 | E14 局部子集冻结版本 | 读出臂最终排名（按 L2）；VLM 不可替代性判定 |
| **W5** | 交叉矩阵第 2 轮：最佳 RO × 全部晋级 RD + 4 个重点单元 | 晋级臂精调 | PR-5 涌现检验（pre/post-SFT） | E24 语料合成管线试产 | **主配置定稿**；emergent 叙事去留 |
| **W6** | E19 条件源替换；主配置上真实数据 | RO-10（若开） | 探针结果写作 | E24 语料量产 | 全局任务掉幅 ≤2dB 验收；未见指令泛化判定 |
| **W7** | E20 打榜（FiveK 局部子集/PPR10K/iRetouch 协议）；E25 Laplacian 支线 | — | — | E26 user study 发放 | **E21 终审判**（PSNR vs 可控性叙事二选一） |
| **W8** | 全消融表回填（§5 主表+附录）；失败臂的负结果整理（R-1 vs R-5、三次基、双轴——负结果也是章节） | — | — | user study 回收分析 | 论文素材冻结 |

**排卡估算**：W3 峰值 = RD 5 臂×2 + RO 2 臂×1 + 探针 1 + 机动 2 ≈ **14–15 卡**；W4–5 交叉矩阵每格单卡短跑（构造集小模型，单格 <1 天）。cube/基底/探针臂几乎不占卡。

**淘汰会规则**：每臂带着统一 harness 的榜单来；判据用各臂表里的预注册数字；被淘汰的臂**保留负结果记录**（R-1 vs R-5、三次基、双轴这三个负结果本身是论文章节）。

---

## 7. 实现档案回填完成（全文见 `IMPL_DOSSIER_2026-08-02.md`），四条改变计划的事实

1. **GLUT 官方仓库是空壳**（CVC-Color/glut 零提交，README 6 字节；mv-lab/GLUT 是检索引擎编造的）。→ **W1 新增任务 A0：从零复现 GLUT**，规格书已抄全（DOSSIER §二：全部公式/超参/初始化/矛盾裁决）。对齐锚点 **GLUT-32@75-LUT ≈ 45.47 dB / ΔE00 0.41**；**复现优先级第一的部件是全局+残差分支**（消融显示去掉直接 −4.93 dB）；工程模板用同组 namedcurves 脚手架（换 models/ 与 data/ 即可）；训练顺序：先只开 L_rec 对齐 45.5，再加 L_hc/R_sparse/hard-mining（合计只值 +0.4 dB，不达标别恋战）。两处论文矛盾的实现决定：Cholesky 对角采 log/exp（init=log 0.15）、opacity 用 raw+clamp（init=1.0 反证 sigmoid 不可行）。
2. **iRetouch 451 对不可得** → E20 协议已改（复刻指标跑自家集 + AceTone-Bench-Transfer/PST50 两个公开外部基准）。
3. **4D LUT 官方代码 context 被置零**（issue 无答复），论文数值按仓库不可复现 → **RD-E 对照臂改用 SA-LUT 的 `clut4d.py` + quadrilinear_cpp 自实现**（参数化 num_context_bins，最干净；CUDA 扩展不支持 batch 内异 LUT，per-sample 循环）。SA-LUT 自身训练数据未发布，只能借结构。
4. **AceTone 的 VQ-VAE 权重就在仓库里**（acetone-vqvae-d64.pt，vq.py 仅依赖 torch 可单文件搬走）→ 质疑 D 的三路 condition 消融之 (a)（离散 token 头）接入成本降为当天；注意其 apply_lut 的 BGR 翻转约定。

其余回填均已就地写入各臂表格（INF-3/4/7、RO-1/2/3/8、RO-X3、E20、E24）。**检索卫生**：本轮又抓到 6 个编造仓库链接（DOSSIER §七末），本档案之外的任何新 URL 必须打开核实后再用。

---

## Changelog

- **2026-08-03（W1 batch-1 首轮结果驱动）**：① G1 判据修订——初版"反义=方向词取反（同区域）"实现走样，ρ_opp 高是"s 只编码 where"的预期行为而非失败；主判据改为**区域对立**（指向不同区域的指令对，ρ_region_opp<0.3），方向对立降级为对照组（详见 DATA_ASSIGNMENT §3.1 G1 行）。② E1b 预注册预测被推翻：线性基底达 E1 门需 r≈384（预测 32–64），能量口径严重低估感知维度；**E1 的 N\* 读数升级为关键判决**——若高斯 N=32 达门（冒烟子样 p90=0.95 已接近），则"非线性基元 vs 线性基底 ≈12× 容量效率"本身成为论文素材。③ F5 bgr gate 放行，RD-G Stage-1 解锁。④ INF 基建就绪宣告（B1/B2/B3 全 CLOSED）。
- **2026-08-03 深夜（G1 中途审阅驱动）**：① RO-9 晋级判据增加**shuffle 前置判别**——shufctrl 批（跨图打乱指令的 ρ 对照）必跑，缺席则不能排除"s 不依赖指令"，方向对立 ρ 高的正向解读不成立。② G1 补充批协议：每源复用已缓存 syn_a 场、只加 1 条异区域指令（~300 前向）；task_type×winner_confidence 分层必报。③ G1 判定顺延（共卡实测吞吐 40–60s/样本，ETA 8–12h）；层选择待主判据数据。④ 中途体检：ρ_Y=0.247 →「s 非亮度马甲」PASS。

- **2026-08-04（G1 终判驱动的 RO 重排）**：Gate D1 **FAIL**——ρ_region_opp 0.884（n_below_0.3=0/214）、shuffle 对照 0.895 反超同义 0.842，`<retouch_light>` canonical 读出是**指令无关的主体显著场**。**RO 臂优先级改为**：第一梯队 RO-1(self-self) → RO-3(全层全头扫描) → RO-2(logit lens) → RO-5(探针头)；**RO-9 降为待判**（早期层 AUC 判别中：L0 ρ_region_opp=0.045 但需 AUC 区分"随指令变"与"纯噪声"）；RO-6/7/8 顺延。**注意：被证伪的是读法不是假设**——H2（VLM 里有 where）未证伪，G2 的 7.97 dB 局部信号与读出方式无关，渲染器线不受影响。同时更正表述纪律：判据"PASS"必须区分**未触发死刑**与**获得正面证据**（ρ_Y 属前者）。

---

- **2026-08-03（W1 batch-2：G3 收尾 + G1 归因终审 + RO 首梯队 + RD-G）**

  **【判据修订，立即生效】**
  1. **UX-2 — RO 系主判据从 AUC 改为 AUC_target（缺一不判）**。依据：RO-X1 实测，一条对 214 张图**完全相同**的短语 `"the main subject"` 拿到 AUC **0.907**，与逐源长描述 0.930 **统计上不可区分**（配对 Δ=+0.009，p=0.133）。AUC 高只说明"找到了主体"，不说明"听懂了指令"。所有 RO 臂的解读按新判据。
  2. **RD-E 的「Δ_shuffle<0.3dB → 全 RD 停工」加前置条件**（§3.1 line 82，同步影响 line 160/177）：仅当该档 **Δ\*_shuffle（天花板）≥ 3 dB** 时该门有效；天花板 < 1 dB 的档**不得用于 RD-E gate**，标注 N/A。依据：G3 实测 mixed 档天花板仅 +0.32 dB，任何模型都读不出，会**误触发全表最重判决**。
  3. **RD 系晋级判据改为相对量**：`Δ_shuffle ≥ max(3 dB, 0.5·Δ*_shuffle(该档))`。依据：同一份 4D 代码在三档数据上读出 +18.1 / +6.0 / +0.59 dB，差异全部来自数据天花板而非模型。
  4. **γ_μ 改为相对量 `0.4·std(anchor_grid)`**（PLAN line 125）。依据：0.15 是按 s∈[0,1] 定的（占网格 std 0.351 的 43%），而 PLAN §1.2 的 s∈[−3,3] 上网格 std≈2.11，同一个 0.15 只剩 **7.1%**，R-3 hinge 会**静默空转**。
  5. **`s-sensitivity` 不得单独当塌陷判据**。反例：fixed 档 Δ_shuffle=+18.1 而 sens=0.00172，tiered 档 Δ_shuffle=+6.0 却 sens=0.00452，**序是反的**。
  6. **⚑ 评测 GT 口径修订（影响全部 RO 臂）**：此前 AUC 的 GT 用的是 **SAM3 主体掩膜**，测的是"能否找到主体"而非"能否找到指令指定的编辑区域"。今后 AUC **一律三列并排**：`.cgt.png`（D-SFT-L 自带逐候选区域掩膜，主列）／ D-CONSTRUCT 构造 GT（诊断列，可按几何/语义/全局分层）／ SAM3 主体掩膜（对照列，用于量化"多少分是光靠找主体拿到的"，基线已钉死 AUC 0.907 / AUC_target 0.523）。**不依赖 GT 的指标**（ρ_region_opp、Δ_shuffle、跨图刻度、噪声地板）不受影响。历史数字按负责人决定**只重算关键几个**（RO-1 与 RO-9 的头条对比、RO-X1 的无名词分离）。

  **【实验判决】**
  7. **G3 / Gate D3 — 塌陷通道「未证实，亦未否证」**。fixed 档 Δ_shuffle naive +18.08 / anchored +18.13（n=3），tiered +6.04 / +5.85，mixed 两臂 inconclusive（数据天花板 +0.32 dB 所致，**非渲染器失败**）。σ_s 全档未发散（`frac_ge_r2_max=0`）。**明确不支持「R-2/R-3 可删」**：(a) lr cosine 退火到 2e-5，"25k 步没发散"有一部分是**日程**不是梯度；(b) mixed 档 σ_s 单调升且未收敛；(c) 玩具规模 ≠ RD-STD。R-2 在该规模**也没赚**（配对差 +0.050 dB，落在 seed 噪声内）。**新增 RD-NAIVE 同预算对照臂**（插 §3.1 line 82 后），在主训规模 + 真实 s 下跑，作为"加固必要性"那节的唯一实证支撑。附：PLAN §6 line 363 只有「证实→…」一支，**「未证实」这支计划里不存在**，需补。
  8. **G1 归因终审 — 代码无缺陷，归因为 (b) 读法错**。四视角审查 + 从 1388 个 npz 全量独立重算（与 metrics.json 吻合到小数点后四位），**零 `bug-invalidates-result` 发现，FAIL 判决不需重跑**。
     - **⚑「shuffle 反超同义」是假象**：0.895/0.842 是 n=60 配对子集（抽签 p=0.0097）且 syn_b 并非同义句；同 37 源同口径重算 ρ_region_opp **0.8905** vs ρ_shuf **0.9093**，配对差 −0.0153，**p=0.571**。真实读数是"三条件并列"。REPORT.md:52 的「负控制被击穿」判词依据的不等号数据里从不存在（NOTES §六预注册写的是"≈"不是"<"），**该行必改**；同步影响 EXPERIMENTS_v3:207 与 EXPERIMENT_INDEX:89。
     - **⚑ 架构级硬上限**：`ro9_gl_attention.py:242` 把 image token 拼在指令**之前**，因果掩码下 image token 的 key 与指令**逐比特无关**，全部指令条件性只能挤过一个 64 维 query（`s = Kᵀq`）。RO-2 从数值侧独立证实：跨图打乱指令 ρ=**0.99995**，而 bit 相同输入 ρ=**1.000000**、句尾加废话 ρ=**0.999934** —— **换指令的差异全部是 bf16 数值噪声**。
     - **「共模淹没/sink」在 head-mean 口径被证伪**：区域对立差分场 AUC **0.5129**，而同区域对照（零对比度基线）**0.5293**（更高），spatial 子类 **0.4929**。**限定：仅 head-mean 口径**——逐头场在 `:378` `raw.mean(axis=2)` 落盘前即丢失（单头方差占比最高 33%，多数头与 head-mean 相关 <0.9），RO-3 正在补逐头版本，那是该解释唯一的翻盘机会。
     - **(a) 假设错在 G1 里从未被检验**：G1 只测了一个读出算子，对 H2 无发言权。**唯一能正面回答 (a) 的形态是 RO-5 探针头**。
     - **新增设计缺陷**：相似度判据缺**噪声地板标定**——全 1388 次读出最小 ρ = **0.585**，而门设在 0.3，**判据可能从设计上不可达**。旁证：`cross_token_rho` light~colormixer **0.809**，换整个 query token 的场变化与换指令同量级甚至更小 ⇒ 该通道动态范围已耗尽。**今后所有相似度判据必须先测地板。**
     - **档案单一真值源**：ρ_Y 在三处并存 0.247 / 0.203 / 0.177，**终值 0.177**（`metrics.json`），0.203 是 n=41 中途快照（EXPERIMENT_INDEX:47 需改）。引用 G1 数字一律以 metrics.json 为准。
  9. **RO-1 — PROMOTE**。ClearCLIP(+A3+A1+A2) SAM3 **0.930** [0.901,0.945] / L1 **0.939**（门 0.80）。三算子 ClearCLIP 0.930 > NACLIP 0.916 > SCLIP 0.902，但 CI 重叠、短名词口径下排名翻转，**不构成可靠排序**——真结论是「**改末层 attention 值 +0.70 AUC，怎么改只值 0.02**」（未改造 vanilla CLIP 仅 **0.214** 且系统性反相关）。16×16 同口径 RO-1 **0.941** vs RO-9 canonical **0.648**／最好层 0.834；Δ_shuffle RO-1 **+0.220**（p=5.5e-17）vs RO-9 **+0.0005**（p=0.121）。**§2.1 RO-1 行必须补限定**：0.930 里 0.907 是**无指令也能拿到的**，指令净增量 +0.009 不显著。RO-4 判据改为「在 RO-1 失败的 18% 子集上 +≥0.05」（子集已落盘 `RO1.../config/ro4_failure_subset.json`，45 源，RO-1 基线 median 0.663）。
  10. **RO-2 — 实质淘汰 → RO-5**。形式 PASS（AUC_target 0.7081 ≥ 0.70；刻度 std 0.0748 < 0.15）但**判别比 0.188 < 1 FAIL**：跨图漂移是主体对比度的 **5.3 倍**，且 std<0.15 在 **25/25 个读出点全过**——该判据无筛选力（A3 采纳：刻度判据改判别比 > 1）。**跨图绝对刻度不成立**。
      - **⚑ 本轮最有价值的发现**：唯一**词特异**的读出点是 `emb`（**mm_projector 输出，进 LLM 之前**）——D-CONSTRUCT GT 掩膜集 AUC **0.829** vs 无关对照词 `calculator` **0.523**（Δ +0.215）；而 **LLM 前向逐层把词特异性抹掉**（emb +0.126 → L1 −0.059 → L8–L19 −0.07…−0.16）。「信息在里面但读出坏了」由此精确化为：**词特异的空间信息在进 LLM 之前就存在，是 LLM 的前向把它抹掉的**。逐层抹除曲线建议作论文主图。
      - **自我证伪范例**：L0 上无关词 `calculator` 的 AUC（**0.7306**）**高于**目标词（0.7081），故那条 0.708 定性为"与词无关的图/底先验，非目标词寻址"。无此对照则 0.708 会被当作正面证据写进论文。
      - `tie_word_embeddings = True`（实测，checkpoint 无 `lm_head.weight`）；DOSSIER §5.3 有 4 条在本项目不成立（无 `language_model` 路径、视觉 token 不能用 151652/153/155 定位、无 deepstack、**`hidden_states` 末项已过 `model.norm`，照抄会双重归一化**）。
  11. **RO-X1（CLIP 侧半场）— 预注册的负面结论触发**。`X_deictic`（无名词固定短语）0.907 vs `N_desc` 0.930，配对 Δ **+0.009，p=0.133** ⇒ **不分离**；辅判据 AUC_target(X_deictic vs 补集) = **0.523** ⇒ 触发。**结论不是"CLIP 能替代 VLM"，而是"AUC 在这里会骗人"**：CLIP 丧失的是**区域可控性**（要求"除主体以外的一切"时场仍压在主体上，AUC 0.887 而 AUC_target 0.523，与 RO-9 被判死时的 0.479–0.526 **同档**）。**⇒ 论文里 VLM 的立足点应写作「可控性」而非「定位精度」。** 最有力一例：`N_desc` AUC **0.084**（场压在铁丝网上）vs `X_deictic` **0.939**——**给了名词全错，不给名词全对**。
      - **DATA_ASSIGNMENT §3.3 订正**：「D-SFT-G 风格指令 = 无名词半集」**不成立**——两个独立检测器实测 style 半集具体名词命中率 0.884/0.942，**严格无名词子集仅 2/86 源**。但 style 半集仍显结构性分离：ρ_shuf local 0.073 vs style 0.536；Δ_shuffle local +0.049(p=4.7e-4) vs style **−0.023(p=0.885, 胜率 0.419)**——**与 G1 里 RO-9 的签名几乎逐字相同**。
      - **免费先验**：`X_deictic` 一条固定短语、零指令、AUC 0.907，优于 RO-9 建议保留的 L20–22 先验（0.825）。RO-5 初始化 / RO-8 伪标签种子优先用它。
      - **已登记短板**：真实无名词语料仅 2 条，`X_*` 全为构造式短语，不代表真实用户语言（语料是否重造待拍板）。
  12. **RD-G — G-Lite / G-Base 双双晋级；淘汰条款未触发**（MLP 仅达天花板 **15.3%**）。ΔE00 p50：MLP 251K **10.548** ／ MLP-Wide 4.93M **10.280**（降 **2.5% ✗**）／ G-Tiny 1.56M 7.471（29.2%）／ **G-Lite 4.45M 7.110（32.6% ✓）** ／ **G-Base 14.06M 6.075（42.4% ✓，方差比 0.932）**。
      - **⚑ 决定性对照 = 参数效率（措辞已于 step 4000 更正，勿引旧版）**：MLP-Wide 与 G-Lite 参数量对齐（4.93M vs 4.45M）。**@2000 读数**（旧）MLP-Wide 降 2.5%、G-Lite 降 32.6%，曾被表述为"加宽 MLP 几乎无用"——**该表述不成立**：**@4000** MLP-Wide 已升至 **14.7%**（逼近 15% 门）。**正确表述**：同等训练预算下加宽 MLP 的**参数效率**远低于换 transformer——**G-Tiny 用 1/3 参数量（1.56M vs 4.93M），@4000 仍比 MLP-Wide 好 27.0%（5.968 vs 8.174）**。差距是结构性的，但**绝对差值随步数变化，任何跨臂绝对差必须标注读数步**。配对 bootstrap 5 个 CI 全不跨 0（G-Base [+3.65,+4.57]，@2000），口径对 transformer 不利（MLP 多训 50–100% 步数）。这仍然回应"0.6B VLM 配几 K 渲染器头重脚轻"的批评，但论据是参数效率而非"加宽无效"。
      - **容量-收益上凸，应继续加大**：transformer 每 10× 参数 +1.63 → +2.24 dB，MLP 族仅 +0.37 dB/decade；G-Base 才走完天花板 31%。**建议加 RD-G3 档 d=512/L=8 ≈35M**。保留意见：读数在 2000/8000 步，上凸可能部分来自"大模型学得快"，8000 步须复核。
      - **加固红线值 2.24 dB**：G-Base 20.231 vs 无加固自由回归 17.994 ⇒ PLAN「≥2 dB 确认加固」成立，**μ 锚定 / σ 有界 / G 初始化=0** 三条红线由此坐实。
      - **烘焙一致性六臂全 PASS**：33³ 导出 + 四面体回读 ΔE00 p50 **0.019–0.026**、p99 ≤0.066，净代价≈0（17³ max 0.285，65³ p50 0.006）⇒ **PLAN §1.6「交付物是标准 3D LUT」在数值上成立**。

  **【基建缺陷，影响全部在跑与后续实验】**
  13. **`/home` 是机械盘**（`sdb`：%util 99.2 / r_await 120 ms / 284 IOPS / 21 MB/s）。缓存落 `/home` 时六臂 **13 分钟零训练步而显存已满 70 GB**。**离线缓存必须落 tmpfs（`/dev/shm`）或 NVMe**，搬迁后 data_wait 从 ~100% → 2.2–4.8%，吞吐 <0.07 → **1.0–2.3 it/s**。**诊断必须用 `nvidia-smi pmon -s um` 看逐进程 SM**——整卡 util 会被邻居顶满，掩盖自己 SM=0% 的真相。
  14. **`tools/bgr_check/common.py::BankResolver` 静默丢 20% 的组**（14,422/72,587）：`ppr10k`/`raise6k`/`fivek_gold` 三个 bank 的 member 后缀是角色限定的 `.source.png`/`.preview.jpg`/`.before.jpg`，解析失败**且不报错**。修法见 `tools/build_cache.py::resolve_sources`。**用过 BankResolver 的臂须复核源覆盖率。**
  15. **scache 埋雷**：`ro9` 臂写入 [−12,+1.2] 原始 logit 但 `meta.norm` 无 `domain` 字段，而消费端 `tools/scache/upsample.py:53` 默认 `clamp=(0.0,1.0)` ⇒ 下游按 README 直接消费会拿到**几乎全 0 的场且不报错**（实测 2776 条目中 >0 的格仅 ~1%）。**所有读出臂写 arm 时必须显式写 `meta.norm.domain`**（RO-1 已补 `[-3.3164, 4.0430]` + `domain_note`）。
  16. **`_mean_s_null` 必须显式传 `s_null`**（按最后一维当通道，G3 靠 `provided_constant` 规避）。**探针取样须等距**（uid 排序会让某一级全占前缀，G3 因此重跑）。

  **【DOSSIER 附录 B 订正队列】** GEM `create_model_and_transforms` 参数错位（须直接用 `create_gem_model`）｜`gem_depth=7` 实为最后 **6** 层｜**GEM `forward(normalize=True)` 默认输出 min-max 热图，照字面用即踩「禁逐图归一化」红线**｜open_clip 须 pin `==2.24.0`（v3.0.0 起 `create_model` 第 3 位参插入 `load_weights`）｜`pkg_resources` 弃用｜§5.3 的 4 条 Qwen2.5-VL 写法在本项目不成立（见 10）。
