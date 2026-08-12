# 冲击拟合上界:条件 dense 场预测 0.76 → ? 的诊断与修复调研

- 日期:2026-08-11
- 性质:纯调研/设计文档。撰写时**未读取任何本地文件**,全部外部事实仅来自文末 VERIFIED 参考表;全部本项目数字来自任务简报。
- 用途:为「收益递减区」的下一阶段提供 (a) 区分四个误差来源贡献占比的**预注册诊断实验包**,(b) 六方向方法谱系,(c) 「诊断结果 → 修复选择」决策树,(d) 缺口清单。
- 与项目红线的关系:本文所有判据均使用 soft-IoU / hard-IoU + grid 级边界 F1 + 中心先验基线列,**不使用任何 AUC 变体**;所有消融臂带 Δ_const / Δ_shuffle 列;可视化遵守空间场可视化纪律。

---

## 一、问题设定复述

**任务**:条件 dense 场预测。输入 = 图像 + 自然语言编辑指令 + 冻结多模态基础模型中间特征(单层 patch 特征 H/16×1024 + 词-区域相似度先验场 + 池化文本 hidden 的 FiLM 调制);~4M 参数轻量 conv 头直出软场;输出空间监督为既定裁定。

**现状**:验证集中位 IoU **0.76**,显著超过零参数中心先验 0.49(配对 +0.26),已治愈「输出被中心先验支配」的历史病;逐样本 oracle 拟合上界 **0.97+**,链路上界 **0.99**;继续训练边际收益缩小。

**输出结构**:GT 场两层生成——离散家族(线性渐变 / 椭圆衰减 / 条带 / 语义轮廓)+ 家族低维参数(中心 / 方向 / 范围)渲染。指令名词决定「哪个物体」(相似度场已供给);指令其余部分**与构图**共同决定家族与几何参数。

**已排除瓶颈(实证,本文不再调研,并作为全部新臂的约束)**:
1. 解码器/链路表达力(oracle 0.97–0.99);
2. 上采样边缘质量(曲率污染修复后 IoU 零变化——边缘不是 IoU 瓶颈);
3. 朴素结构正则(曲率/单调性损失实测 −0.027 IoU);
4. 低维参数**回归输出**路线(条件数 1e6,两次塌缩);
5. 参数量本身(35M 大头以更坏机制失败;4M 头无容量不足直接证据)。

**四个候选误差来源**:
- (i) **条件信息瓶颈**:决定场几何(方向/侧向/范围)的低维语言信息未被有效提取/注入。现有注入只有相似度场(只带物体位置)+ pooled FiLM(全局调制);换主体配对差分仅 +0.04,条件化浅。
- (ii) **目标固有随机性**:GT 几何参数部分由标注/构造期随机或主观选择决定,y|x 存在 aleatoric 上界——**从未被测量**。
- (iii) **尾部欠学习**:误差集中于小/偏心/复杂样本长尾,均匀采样训练学不好尾部。
- (iv) **特征供给不足**:单层冻结特征 + 单一相似度场缺空间信息(多层/多模型/更高分辨率特征未用)。

**约束**:冻结基础模型不可微调;单卡数小时训练量级;数万训练样本;推理低开销;输出空间监督既定。

**核心概念裁定(Bickford Smith, 2412.20892)**:「可达上界」定义为**实际评测损失下的 Bayes 风险** min_f E[1−softIoU(f(x),y)],所有诊断读数统一折算成 IoU 量纲。**0.97 的逐样本 oracle 拟合不是条件可达上界**——oracle 看到了 y,估的是链路容量;真正的条件上界 U 必须由 §2 的 (ii) 测量线给出。表观差距 0.21 = (0.97−U) 的不可约份额 + (U−0.76) 的可修份额。

---

## 二、诊断实验包(核心交付)

### 2.0 归因对账协议(先于一切实验,主 agent 层面裁定)

六路调研各给了读数,但归因**不可加**:oracle 参数注入 Δ(记 (i))与特征交换 Δ(记 (iv))有重叠(几何信息可从任一侧补入);best-of-k 差含 (ii)+择优器损失;缩放外推 C 混淆 (ii)/(iv)。预注册以下规则:

1. **统一量纲与 split**:全部读数报 IoU(中位数 + P10/P25/P75),同一 S-val 源 split;分层(面积×偏心×家族)双列报告,防「尾部修好头部退化」被中位数掩盖。
2. **重叠归属**:(i)×(iv) 用 2×2 因子设计(§B2-1/B2-3 合并跑 base / +oracle条件 / +强特征 / +both 四臂),主效应各记各的,交互项单列 "(i∩iv)" 行,**不双计**。
3. **(ii) 的主判据是直接测量**(重放/矛盾对/best-of-k 三件套),缩放外推降级为辅助证据(理由见 §2.4 B3-3)。
4. **跨候选修复只记一次**:Talk2DINO/ProxyCLIP/EVF-SAM 式修复同时利好 (i)(iv),归因表规定其收益记入 (iv)(特征空间侧),(i) 行只记条件通路干预。
5. **oracle 注入类诊断必配负控制**(打乱参数 / 常量参数),否则「IoU 跳升」无法区分「信息通了」与「多一条输入通道的容量效应」——与项目 Δ_const/Δ_shuffle 纪律同构。
6. **不确定性纪律(Mucsányi, 2402.19460)**:禁止把 ensemble 方差 / MC-dropout / 证据头单一数字标成「aleatoric 份额」写进 REPORT——实测全部不解耦。正确姿势:seed 分歧只当 epistemic 读数,生成器重放/矛盾对离散度只当 aleatoric 读数,独立测量后对账。

### 2.1 D0 批:零训练诊断(第 0 天,全部标 [零训练])

| 编号 | 名称 | 做法 | 预注册判据 / 证伪数字 | 成本 |
|---|---|---|---|---|
| **D0-1** | **重放可行性核实**(整条 (ii) 直接测量线的闸门) | 审计 GT 构造管线:随机性住在哪(家族平票 tie-break?参数抖动?人工预设选择?);固定 (图,指令) 能否重放生成器得到不同合法 GT | 二值判定:**可重放** → D0-7 开;**不可重放**(固定种子单次落盘、随机性在上游人工选择) → (ii) 主判据退回 D0-8 矛盾对 + B1-2 best-of-k | 半天,零训练 |
| **D0-2** | **参考系审计**(COMFORT/RoboSpatial 移植) | 审计 GT 构造程序:left/near the edge 等方向词构造期参考系是否唯一一致;指令模板是否混用 ego/object-centric | 若混用:按参考系分桶重测 IoU,**桶间 Δ ≥ 0.05** → (ii) 中可归因于参考系歧义的份额实锤,修复在 GT 侧(指令规范化补参考系槽位);唯一一致 → 此项排除,(i) 嫌疑加重 | 半天,零训练 |
| **D0-3** | **SDC 难度分 + 风险-覆盖曲线**(2402.10665) | 对现有 0.76 模型软场 ŷ 与其面积匹配 top-k 二值化 M 算 2Σ(ŷ·M)/(Σŷ+ΣM),一次前向全量排序,画 risk-coverage | **最低 SDC 的 25% 样本承载 ≥ 60% 总 IoU 损失** → 尾部集中确认,(iii) 立案;损失近均匀分布 → (iii) 降权 | 一次前向,零训练 |
| **D0-4** | **特征伪影检查**(Registers/DVT) | VLM 单层特征范数直方图 + PCA 可视化,检出高范数 sink/artifact 格(项目已见 pad/sink 涌现,同源) | 高范数格占比 ≥ 3% 且空间分布网格状/pad 集中 → (iv) 内立「去噪/显式排除」子臂(量级参照 DVT +1~2 mIoU);无伪影 → 关闭该子臂 | 半天,零训练 |
| **D0-5** | **空间语言覆盖率审计**(SPRIGHT 动机移植) | 统计数万条指令中显式方向/侧向/范围词的覆盖率与多样性,按 GT 几何自由度交叉 | 某几何自由度(如方向)的语言证据覆盖率 **< 30%** → (i) 的病根部分在数据不在结构,数据增密臂立案;覆盖充分 → 病根在通路 | 1 小时脚本,零训练 |
| **D0-6** | **ProxyCLIP 式先验增强试探**((i)/(iv) 区分器,2408.04883) | 取 DINOv2/v3 patch 亲和度矩阵 A,把现有相似度场 s 换成 A·s(一次矩阵乘),直接喂现有已训头(不重训)评测 | ΔIoU ≥ +0.02 → 相似度场毛病是**空间破碎**((iv) 侧,VFM 亲和度可补);Δ≈0 → 毛病在「选错物体/缺几何词」((i) 侧) | 当天,零训练 |
| **D0-7** | **生成器重放 Bayes-IoU 上界**((ii) 主测量之一;Ishida 2202.00395 同构 + TTA-aleatoric 1807.07356 谱系) | D0-1 通过后:对 S-val 每样本重放生成器 k=8 次(仅重采随机/主观自由度,其余固定),计算重放场两两 IoU 的均值,得逐样本 aleatoric 天花板;按指令类型/家族聚合 | 全集均值 **U_replay ≤ 0.85** → (ii) 主导(吃掉表观差距 0.21 的一半以上),继续压均值 IoU 不值得;**0.85–0.93** → (ii) 实质存在,须与修复并行处理;**≥ 0.95** → (ii) 基本排除,可修份额 ≈ 0.2 仍在 | 纯渲染批跑,零训练;**四个候选里最便宜的决定性测量** |
| **D0-8** | **矛盾对挖掘 + 小规模人工裁定**(Label Convergence 2409.09412 + VariErr 2403.01931) | 冻结特征对 (图,指令) 联合嵌入近邻检索,挖「条件近重复、GT 几何显著不同」对(目标 ≥ 200 对);对内 GT 互算 IoU 得经验上界带;对残差最大的 200–500 样本人工三选一裁定:GT 唯一合理 / 模型场同样合理 / GT 是错的 | **0.76 落入矛盾对上界带 [q25,q75] 内** → (ii) 主导判定(LVIS 教训:表观 30 mAP 空间实则为 0);「模型场同样合理」率即 aleatoric 人工下界;「GT 错」率 ≥ 10% → 构造管线修复立案。警示:**不许用模型置信度或 LLM 自动代替人工裁定**(VariErr 实证自动方法不可靠) | 检索零训练;人工半天(需懂 GT 构造规范者) |

### 2.2 B1 批:一组 3–5 seed 训练,三个读数复用同一批 checkpoint

| 编号 | 名称 | 做法 | 预注册判据 / 证伪数字 | 成本 |
|---|---|---|---|---|
| **B1-1** | **GDE 种子分歧**(2106.13799;**只当定性读数**——等式定理在分类误差上成立,soft-IoU 无理论保证) | 同配置 5 seed 各训一遍 4M 头;算 seed–seed 配对 soft-IoU 矩阵与各 seed–GT IoU;逐样本分「都会/都错/意见分裂」三层 | seed 间配对 IoU **≥ 0.92** 且对 GT 停在 0.76 → 残差是共享偏置或 aleatoric,**继续训练/加 ensemble/调 lr 均无效**,转 (i)/(iv)/(ii);seed 间距 ≥ 0.5×(对 GT 距) 即配对 IoU **≤ 0.88** → epistemic 未榨干,训练侧仍有肉;「意见分裂」层样本清单交 (iii) 线 | 5×数小时(可过夜),一次性 |
| **B1-2** | **best-of-k / oracle-of-N 分解**(Tyche 2401.13650 + SAM 2304.02643 min-loss 多头) | 4M 头改 K=8 输出通道,winner-takes-all(仅最优候选回传)重训一次;或先用 B1-1 的 5 seed 当免费 N 假设。逐样本取事后最优(oracle-of-N),同时训 SAM 式 IoU 回归分支自选 | **E[best-of-8] − E[single] ≥ 0.10** → GT 多峰实锤(指令欠定几何),0.76 是单点输出的**结构性上界**,与 D0-7/D0-8 交叉对账;**≤ 0.03** → 多峰性排除,残差另有来源。oracle-of-N→0.97 的剩余差 = 系统性偏差,归 (i)/(iv)。「oracle 自选差」= 择优器损失,单列 | 1 次重训(K 头版)+ 复用 B1-1;近零推理增量 |
| **B1-3** | **假设散布 × 残差回归** | 用 B1-1/B1-2 的假设间逐像素方差图当 aleatoric 代理场,对逐样本残差 (1−IoU) 做回归 | 散布能解释的方差份额 R² ≥ 0.3 → 该份额记 (ii) 候选(仍须 D0-7/D0-8 确认,勿单独下结论——Mucsányi 红线);R² ≈ 0 → 残差是系统性的 | 复用,零新训练 |

### 2.3 B2 批:干预式诊断(4–6 次头训练,可两夜排完)

| 编号 | 名称 | 做法 | 预注册判据 / 证伪数字 | 成本 |
|---|---|---|---|---|
| **B2-1** | **oracle 几何参数注入 + 负控制**((i) 天花板测量;Ranni 2311.17002 思路的诊断化) | 把 GT 家族+参数按解析式渲染成 1–2 张几何先验通道(沿方向向量的有符号坐标场 / 锚物质心径向场),与相似度场并列输入,重训头。**必配两个负控制臂**:打乱参数(换成另一样本的)与常量参数 | **IoU_oracle ≥ 0.90 且 Δ_shuffle ≤ +0.02** → (i) 通路瓶颈实锤,且解析注入路线的天花板 ≈ IoU_oracle;**IoU_oracle ≤ 0.82** → 几何参数即便全给也补不上,主嫌疑转 (iv)/(ii);Δ_shuffle 大 → 容量效应污染,判读作废重设计。注意:参数在此是**输入条件**,与已排除 #4(参数作回归输出)不冲突 | 3 臂 × 数小时 |
| **B2-2** | **PVI 指令置空差分**(V-usable information 2110.08420) | 同族同协议训两个头:完整条件 vs 指令置空(只留相似度场);逐样本 ΔNLL = 指令的可用条件信息;再在强特征臂(B2-3)上重算 | 全集 PVI 中位数 ≈ 0 → 指令信息根本没被用((i) 实锤,与 +0.04 配对差分互证);PVI 在强特征下上升 → 原残差部分是 (iv);逐样本 PVI≈0 且 D0-8 判「GT 唯一合理」→ 该样本疑标注错,送人工 | 2 次头训练 |
| **B2-3** | **特征供给消融矩阵**((iv) 主测量;AM-RADIO 2312.06709 / Probe3D 2404.08636 / DINOv2 lin.4 协议) | 同一 4M 头 × 同预算 × 同步数,臂:{VLM 单层(基线) / VLM 多层 concat(等距 4 层,各 1×1 投影同维) / +DINOv3 拼接 / 仅 DINOv3 / +AnyUp 上采样先导}。**公平性协议**:全臂 1×1 投影到同通道数,输入统计量各自标准化(整臂常量,禁逐图) | 最优臂 **ΔIoU ≥ +0.03** → (iv) 确认,占比 = Δ/可修份额;全臂 **≤ +0.01** → (iv) 排除。分辨率臂(AnyUp)单独判:无收益即整体关闭上采样方向(与已排除 #2 一致,预期低)。与 B2-1 合并为 2×2 因子(base/+oracle条件/+DINOv3/+both),交互项记 (i∩iv) | 4–5 臂 × 数小时,单卡一夜 |
| **B2-4** | **逐词最小对配对差分**(What'sUp 2310.19785 协议移植) | GT 程序渲染 → 零成本构造「同图 + 方向词反转」「同图 + 范围词替换」「同图 + 家族暗示词替换」测试对,现有模型直接评,报逐词类配对 Δ + 打乱指令/无关词/固定短语三条负控制 | 方向词反转配对 **Δ ≤ 0.02** → 方向信息未进通路(与换主体 +0.04 并列成「名词/方向/范围三路条件化深度」画像);哪类词 Δ 低,修复就打哪类 | 零训练(纯评测) |
| **B2-5** | **pooled vs token 线性探针**(ARO 2210.01936 诊断半边) | 从冻结 VLM 分别取 pooled hidden 与逐 token 隐层(扫中间层),训线性探针解码方向(8 分类)/侧向/范围/家族 | pooled 探针差、**token 探针 acc ≥ 75%** → 信息在池化步丢失 → 修复走「头内 token 级注入」分支;**token 探针也 < 60%** → 冻结骨干文本表征本身不含几何词信息(What'sUp 负结果的本地版) → (i) 修复只剩显式解析路线;此判据同时是 §四 决策树 (i) 分支的**预注册分岔数字** | 线性探针,分钟级训练 |
| **B2-6** | **masked-token grounding 探针**(MagNet 2312.12198 辅助任务当探针用) | 从空间特征 + GT 场预测被 mask 的指令词(方向词/名词分列) | 被遮方向词预测 acc 显著低于名词 → 方向信息没进空间特征,佐证 B2-5 | 1 次轻训练 |

### 2.4 B3 批:分型与辅助

| 编号 | 名称 | 做法 | 预注册判据 / 证伪数字 | 成本 |
|---|---|---|---|---|
| **B3-1** | **RHO-LOSS reducible loss 分型**((ii)/(iii) 边界的现成区分器,2206.07137) | 训练集外留小 holdout 训同构 4M 头;全训练集逐样本 reducible loss = L_train − L_holdout | 高 loss 且 reducible ≈ 0 → 不可学(归 (ii) 候选池);高 loss 且 reducible 高 → 尾部欠学习(归 (iii),进重采样池);两池占比直接进对账表 | 1 次头训练 + 1 次全量前向 |
| **B3-2** | **β-NLL 逐像素 σ 头**(2203.09168) | 冻结特征上并联 σ 头,β≈0.5(**禁裸 NLL**——梯度被方差反向加权的自证放弃病理);held-out 上 variance scaling 校准 | Spearman(σ̂, 1−IoU) ≥ 0.4 且跨 seed 稳定 → σ̂ 可用作 (iii) 重加权的免噪权重(高 σ̂ 样本不 upweight,与欠学习区分);不稳定 → 弃用 | 1 次头训练 |
| **B3-3** | **缩放外推(降级为辅助)**(Chinchilla 2203.15556 同构 + 观测式缩放律 2405.10938) | {1M,2M,4M,8M} 头 × {25%,50%,100%} 数据的小网格拟合 IoU=C−A·N^−α−B·D^−β;或复用既有 arm checkpoint 按观测式缩放律联合拟合共享渐近 C | **预注册可用性门**:C 的 95% CI 宽 ≤ 0.04 才允许判读(9–15 个带噪点拟 5 参数饱和曲线,CI 大概率盖住 0.78–0.90 整个判决区间——预期不可判);**35M 臂剔除**(以更坏机制失败,非容量点,入拟合即污染);判读(若可用):C≈0.78–0.82 → (ii)/(iv) 主导;C≥0.9 → (iii) 主导 | 若跑全网格 ≈ 十余个数小时 run;建议只做复用版 |

### 2.5 对账表模板(REPORT 强制格式)

```
表观差距 0.21 = [0.97 − U] 不可约份额        ← D0-7 / D0-8 / B1-2 三件套,取交叉一致值
             + [U − 0.76] 可修份额,拆:
               (i)   主效应   ← B2-1 oracle 注入 Δ(扣除负控制)
               (iv)  主效应   ← B2-3 最优特征臂 Δ
               (i∩iv) 交互    ← 2×2 因子交互项(不双计)
               (iii)          ← B3-1 reducible-高池 × D0-3 集中度
               择优器/训练方差 ← B1-2 oracle-自选差 + B1-1 seed 距
               残余(未解释)   ← 如实报,不摊派
判读一致性检查:三个 (ii) 读数(重放 U_replay、矛盾对带、best-of-k 差)须量级互洽;
不洽时以重放为准(最干净),矛盾对次之,best-of-k 最后(含择优器损失混入)。
```

**执行序(成本递增,前序门控后序)**:D0 全部(1–2 天,零训练)→ B1(一夜)→ B2(两夜)→ B3(按需)。**修复臂全部押后到对账表落定**。

---

## 三、方法谱系(六方向)

> 增益量级标注可信度:**[A]** 可直接引用(条件相近);**[B]** 方向可信、数字打对折以上;**[C]** 高折扣或不可搬。

### 3.1 predictability-ceiling((ii) 测量与建模)

| 工作 | 适用条件 | 实证增益/读数 | 成本 |
|---|---|---|---|
| Ishida 2202.00395(直接 Bayes error 估计,ICLR'23 top-5%) | GT 生成器可重放(D0-1 通过) | 先例:ViT 级模型已逼近估计 Bayes error——「0.76 即上界」是可检验命题 [A,机制] | 纯推理,零训练 |
| Label Convergence 2409.09412(WACV'25) | 数据集内存在近重复条件对 | LVIS 上界带 62.6–67.5 mAP,SOTA 已在带内——表观空间可以为 0 [A,方法论] | 检索+互算 IoU |
| GDE 2106.13799(ICLR'22) | ensemble 大体校准;**soft-IoU 上仅定性** | 分类上分歧率≈测试误差(数个百分点内)[C→定性用] | 3–5 次重训 |
| Chinchilla 2203.15556 + 观测式缩放律 2405.10938 | 幂律小网格拟合稳定、CI 达标 | 把边际递减变成带渐近线曲线 [B;本设定预期不可判,降级] | 十余 run 或复用 |
| Tyche 2401.13650(CVPR'24)/ SAM 2304.02643 多头 min-loss + IoU 头 | GT 确有多解(先用 oracle-of-K 检验) | Tyche:20 个未见任务 best-candidate 追平专训基线;SAM:oracle 选择显著高于自选——收益大半卡在择优器 [A,机制] | K 头改造,近零增量 |
| β-NLL 2203.09168(ICLR'22) | σ̂ 需 held-out 校准 | 修复裸 NLL 的自证放弃病理 [A,机制] | 1 次头训练 |
| Mucsányi 2402.19460(NeurIPS'24)/ Bickford Smith 2412.20892(ICML'25) | 方法论红线 | 负结果:现成 aleatoric/epistemic 分解全部不解耦;熵类代理常测非所标 [A,红线] | 零 |
| VariErr 2403.01931(ACL'24) | 需懂 GT 规范的裁定者 | 500 items 裁定即足以下结论级判断;自动方法(含 LLM)不可靠 [A] | 人工半天 |
| PVI 2110.08420(ICML'22) | 两头同族同协议 | 逐样本可用信息差分,兼查标注错 [A,机制] | 2–3 次头训练 |

### 3.2 language-spatial((i) 机制与语言侧修复)

| 工作 | 适用条件 | 实证增益 | 成本 |
|---|---|---|---|
| ARO 2210.01936(ICLR'23 Oral) | 诊断+硬负例训练 | pooled 表征系统性丢关系/词序;NegCLIP 硬负例显著修复组合任务 [A,机制] | 探针分钟级 |
| What'sUp 2310.19785(EMNLP'23) | 最小对协议移植 | VLM 方位理解 56% vs 人类 99%;**负结果:浅层加权/直接微调修不好**——若 token 隐层本就没有,修复必须走解析注入 [A,警示] | 零训练评测 |
| SmartEdit 2312.06739(CVPR'24) | 条件源可换成「图文前向后」的指令 token 隐层 | 归因:CLIP pooled 条件化是编辑失败主因;少量复杂指令数据即可激发 [B] | 头内 BIM 式双向块 ~0.5M |
| Ranni 2311.17002 / RPG 2401.11708 / VoxPoser 2307.05973 | 「语言→结构化槽位→dense 先验通道」;**参数作输入非输出,与已排除 #4 不冲突** | 结构化中间表示在数量/绑定/空间组合上显著优于端到端 [B];**解析准确率必须先测 200 条,禁引「通常>95%」这类无出处数字** | 离线解析;**部署需蒸馏轻解析头(缺文献)** |
| RoboPoint 2406.10721(CoRL'24) | 辅助点/向量头;**与已排除 #4 有结构相似性,必须带塌缩监控 + 小权重上限 + kill 判据** | +21.8%(任务不同,仅量级参考)[C] | <0.1M 辅助头 |
| RoboSpatial 2411.16537(CVPR'25 Oral)/ COMFORT 2410.17385(ICLR'25 Oral) | GT 程序可审计 | 参考系歧义是真实 y\|x 多解源;9 个 SOTA VLM 在歧义参考系下一致性差 [A,诊断] | 零训练审计 |
| SPRIGHT 2404.01197(ECCV'24) | 空间词覆盖率审计后 | 0.25% 空间密集数据 → +22% 空间正确率;<500 样本微调达 spatial SOTA——数据侧修复样本效率极高 [C→定性:样本效率高] | LLM 改写 + 渲染器回验 |

### 3.3 conditioning-arch((i) 注入结构修复)

| 工作 | 适用条件 | 实证增益 | 成本 |
|---|---|---|---|
| EVF-SAM 2406.20076 | 条件形态选型证据 | text-only 池化 65.1 → 晚期 concat 70.2 → token 级早融合 83.7 cIoU;**微调了 BEiT-3,冻结约束下是不可达上界** [C,引用必须标注] | — |
| SD3/MM-DiT 2403.03206 | 文本以 token 序列供给 | 固定文本表征喂 cross-attn 非最优;双向 token 混流全程占优 [B,序关系] | 双向块(可共享权重) |
| DiT adaLN-Zero 2212.09748 | 条件是全局低维量(家族选择) | adaLN-Zero FID 约为 in-context 一半且省算力——**FiLM 族对全局离散条件最优,对空间几何结构上不可表达**;零初始化是稳定关键 [A,机制] | 零增量 |
| IP-Adapter 2308.06721 | 多条件流解耦注入 | 22M 达全量微调级;consat 进同一注意力显著劣于解耦 K/V [B] | 每层 2 矩阵 |
| OminiControl 2411.15098 | 主干有注意力块 | 0.1% 参数同时承载空间/语义条件;位置编码决定条件被当空间还是语义用 [B] | conv 头先插轻注意力块 |
| MagNet 2312.12198(CVPR'24) | 词级辅助监督;**−0.027 前科 → 必带 kill 判据** | 即插即用 +2.48 oIoU,开销 +4.76%;增益在复杂/罕见表述尾部最大 [B] | 辅助头轻量 |
| F-LMM 2406.05821 | eager attention 可导出(项目已有纪律) | 全层×全头 attention 堆栈喂 tiny U-Net 达 RES 竞争力;**全层堆叠>任何单层,晚层最差,几何在中早层** [A,与本设定同构最强] | 通道数增大,头仍 tiny |
| PSALM 2403.14598(ECCV'24) | 只取接口思想;**禁「场=少数因子重构」强分解**(与 #4 病灶结构相似) | 多 token 联合解码全面超单 [SEG] 池化路线 [B] | K 个 query token |
| ControlNet 2302.05543 | 分工注入锚点 | 空间条件→逐位置加法、语义条件→注意力、连接处 zero-init 不掉点 [A,机制] | 零风险叠加 |

### 3.4 frozen-features((iv) 特征供给修复)

| 工作 | 适用条件 | 实证增益 | 成本 |
|---|---|---|---|
| AM-RADIO 2312.06709(CVPR'24) | 旁挂或换骨干 | ADE20k 线性探针:CLIP 系 35–41 vs DINOv2-g 48.7 vs RADIO-H 51.3——语言对齐特征 dense 差距 8–11 mIoU [A,探针口径] | 特征缓存 |
| Probe3D 2404.08636(CVPR'24) | 诊断协议(**探针须 DPT 式多层,单层线性会低估**) | 语言监督模型深度/法向探针垫底 [A,定性;**不覆盖带 LLM 解码器的 VLM——证据缺口**] | 探针级 |
| DINOv2 lin.4 协议 2304.07193 | 中间层可访问 | 深度 lin.4 一致优于 lin.1;ADE20k lin→+ms 47.7→53.1(多层+分辨率+TTA 三因素混合)[A,协议] | ~0 训练增量 |
| DINOv3 2508.10104 | 旁挂第二特征流默认选择(2025 起) | dense 基准超 DINOv2 与 AM-RADIO [B,自报] | 一次特征缓存 |
| FeatUp 2403.10516 / LoftUp 2504.14032 / AnyUp 2510.12764 | **门控在尾部诊断之后**(已排除 #2:边缘非瓶颈,头条数字在本设定不可信) | LoftUp 比 bilinear +5~9 mIoU(探针口径)[C 本设定];AnyUp 免逐骨干训练,当零成本先导 | AnyUp 零训练先导 |
| DVT 2401.02957(ECCV'24)/ Registers 2309.16588(ICLR'24) | D0-4 查出伪影才开 | 去噪 +1~2 mIoU(已带 registers 模型仍 +0.86/+1.12)[A] | 10k 样本轻训练 |
| NeCo 2408.11054 | **给 VLM 特征训 NeCo 是无实证研究赌注** | DINOv2 上 +5.5~7.2 mIoU,19 GPU 时 [C 移植] | 单卡一夜 |
| Talk2DINO 2411.19331 / ProxyCLIP 2408.04883 | 相似度场改在空间最强特征空间算;**无监督 OVS 口径,有监督头可能已自行补偿,净增益缩水** | Talk2DINO 46.3 vs ProxyCLIP 42.3;ProxyCLIP 零训练 +4.1 [B] | 小投影 / 零训练 |

### 3.5 tail-learning((iii) 修复)

| 工作 | 适用条件 | 实证增益 | 成本 |
|---|---|---|---|
| RHO-LOSS 2206.07137(ICML'22) | 兼诊断分型(§B3-1) | Clothing-1M 少 18× 步数 +2%;近收敛模型上价值主要在分型 [A] | 1 holdout 头 |
| InfoBatch 2303.04947(ICLR'24) | 倾斜训练不退化头部的技术答案 | ADE20K 分割 20–40% 成本无损(梯度期望无偏重标定)[A,有分割直接证据] | 即插即用 |
| JTT 2107.09044(ICML'21) | 错误集上权;λ/q 需小 val 调 | worst-group 差距补 75%;平均精度通常轻降 [B] | 2 阶段重训 |
| When Do Curricula Work 2012.03107(ICLR'21) | **反证锚点** | 标准设置难度排序无增益;仅预算受限或标签噪声下「易先」有效;anti-curriculum 全灭 [A,警示] | — |
| Beyond Scaling Laws 2206.14486(NeurIPS'22 Outstanding) | 倾斜方向 = 数据量级的函数 | 数据稀缺保易、充裕保难;**数万样本属中间区,方向必须 A/B 实测,选错显著损性能** [A,警示] | 3 臂 A/B |
| ConR 2309.06651(ICLR'24)/ Balanced MSE 2203.16427(CVPR'22) | 特征正则(不动输出头);**Balanced MSE 辅助回归探针废弃**(修标签不平衡,不修条件数 1e6 病态,大概率复现塌缩) | ConR few-shot 区间 1–5% [B];前提「参数分布确有可识别头尾」需先验证 | 辅助损失,必带 kill |
| FreeMask 2310.15160(NeurIPS'23)/ Feedback-guided 2310.00158 | **前提缺口:渲染器只能渲染场,不能合成图像**;重采样 (指令,家族,参数) 须先建「与构图相容」一致性检查器,否则批量制造标注噪声并污染 (ii) 测量 | FreeMask +3.3 mIoU;尾类 +4~5% [B,数字不可直接搬] | 渲染近零 + 检查器待建 |
| Learning to Simulate 1810.02513(ICLR'19) | 渲染器采样分布当策略,外环优化 val IoU | 框架级;自动规避过度倾斜(reward 是全体 val)[B,锚点] | 外环 5–10 轮内环数小时 |
| MEDOE 2308.08213 / Frequency-based Matcher 2406.03917 | 轻量专家头 / 评测纪律 | ≤+1.78 mIoU;**分层双指标进判据表是必须移植的纪律** [B/A-纪律] | 2–3 个 ~1M 专家头 |

### 3.6 test-time(测试期,与四候选正交)

| 工作 | 适用条件 | 实证增益 | 成本 |
|---|---|---|---|
| Marigold 2312.02145(CVPR'24) | 头侧多假设 + 免 GT 中值合并;**假设对齐步在 area-matched top-k 下是空操作,不移植** | ensemble 相对 3–8%(确定性头的多样性更弱,再打折)[B] | N 个 4M 头前向近零 |
| 扩散感知缩放 2411.08034 | 预算重分配思想(假设数×每假设精度配比实测) | 定性一致改善 [B] | 扫描级 |
| Inference-time scaling 2501.09732 | best-of-N 框架;**先 oracle verifier 测上限再谈无 GT 择优;单一 verifier 会被 hack,用集成** | oracle/监督 verifier 下大幅单调改善;错配反向退化 [A,框架] | — |
| SAM IoU 头 + stability score 2304.02643 | 训练期 GT 免费的质量回归 | oracle 显著高于自选——择优器质量是主损耗 [A] | IoU 分支近零 |
| EvanySeg 2409.14874 | 外置质量评估器,可评任意来源假设 | 可行性验证级;期望增益 = oracle 差 × 择优器保真度 [B] | 小回归器 |
| 一致性择优审计 2608.01207(2026) | **红线:一切择优臂必配同预算共识聚合(中值/多数投票)对照臂** | 负结果:扰动一致性择优对配平多数投票净增益全基准不显著 [A,负结果] | — |
| SDC 2402.10665 | 免训练逐样本难度分 | 选择性预测近最优;25% deferral 移除高达 80% 误差(Rethinking UQ 2604.13262 口径)[A] | 一次前向 |
| ADE-CoT 2603.00141(2026) | 难度感知假设预算路由;任务特化 verifier(候选场×相似度场重合度 / 方向词×场梯度一致性) | 同预算优于 Best-of-N 且 >2x 加速;通用 MLLM 打分早期不可靠(负发现)[B] | 路由逻辑 |
| S³-TTA 2310.16783 / SAMRefiner 2502.06756 | 输入级 TTA 只对难样本、只用预筛增强;SAMRefiner 仅轮廓家族「支撑对贴合差」尾部 | +3.4%/+1.3%(个位数方差削减性质,不修 (i))[B];SAMRefiner 定性 [C] | 骨干重跑,最贵,末位 |

---

## 四、设计处方:诊断结果 → 修复选择决策树

```
D0+B1 落定后,按对账表最大份额进入对应分支;份额相近时并行最小验证臂,禁全面开工。

[根] U(条件可达上界,D0-7/D0-8/B1-2 交叉)
 ├─ U ≤ 0.85 ──────────────→ 分支 (ii):接受上界,转向
 ├─ U ≥ 0.95 且 B2-1 oracle 注入 ≥0.90 ─→ 分支 (i)
 │        └─ 其内再按 B2-5 探针分岔:token 探针 ≥75% → (i-a) 头内注入
 │                                    token 探针 <60% → (i-b) 解析注入
 ├─ B2-3 特征矩阵 Δ ≥ +0.03 ─→ 分支 (iv)(与 (i) 交互项按 2×2 因子归账)
 └─ D0-3 尾部集中 + B3-1 reducible-高池大 ─→ 分支 (iii)
 测试期层对全分支正交,最后叠加。
```

### 分支 (ii):固有随机性主导 → 停止压均值,三条出路

1. **生成器侧确定化(最便宜,工程路径,无文献但机制自明)**:D0-1/D0-2 已定位随机自由度(平票 tie-break、参数抖动、参考系混用)→ 在源头消掉 y|x 熵:tie-break 定则化、参考系槽位规范化、抖动固定;重渲染 GT,重训单头。**最小验证**:重测 U_replay 应 ≥ 0.95,单头 IoU 应向新 U 移动 ≥ +0.05;不动则确定化没打中随机源,回 D0-8 找。
2. **多假设输出**:K=4–8 头 + winner-takes-all(SAM/Tyche),SAM 式 IoU 回归分支自选,产品侧可出 top-k 候选。**最小验证**:B1-2 已给 best-of-k 差;上线判据 = 自选实现 ≥ 60% 的 oracle 差,且配共识聚合对照臂(2026 审计红线)。
3. **重设基线**:REPORT 判据表把 0.97 换成 U,性能主张一律相对 U 报告(Bickford Smith 依据);已达 U 的子桶宣布关闭。
- 备选(缺文献,见 §五):条件隐变量场头(Probabilistic U-Net / PHiSeg / SSN 线)——比 K 头 WTA 更省参数的 dense 多峰标准解,立臂前须补文献核实。

### 分支 (i):条件通路主导

**(i-a) 头内 token 级注入**(B2-5 判 token 隐层含几何信息时):
- 处方:conv 头前插 1–2 层 grid→text-token cross-attn(F-LMM 证据:文本-图 attention 堆栈信息在中早层,全层>单层);多条件流解耦 K/V(IP-Adapter),共享 Q,输出相加;**全部新增块 zero-init 门控**(ControlNet/adaLN-Zero);**双粒度分工**:家族选择走 FiLM(DiT 证据:全局低维条件 FiLM 最优),几何词走 token cross-attn(SD3 证据:固定表征单向注入非最优,预算允许则做 1–2 轮双向更新);可选 MagNet masked-token grounding 辅助头。
- **最小验证(1 次训练)**:方向词反转配对 Δ 从 ≤0.04 升到 **≥ 0.10** 且总 IoU **≥ +0.02**;辅助损失单独消融,Δ < +0.01 即 kill(−0.027 前科)。
- **诚实预期**:EVF-SAM 65.1→83.7 含条件编码器微调,冻结+只训头内新增块的增益**无 verified 先例**,预期打大折;这是研究赌注不是移植。
- 数据侧并行小臂(D0-5 判语言覆盖稀疏时):SPRIGHT 式空间增密改写,渲染器回验改写不改变 GT 语义;判据同上。

**(i-b) 显式解析注入**(token 隐层本身不含几何信息时,唯一完全绕过嵌入瓶颈的路线):
- 处方:LLM 把指令译成微型几何 DSL {family, direction, side, extent, anchor}(Ranni/VoxPoser/RPG 谱系),按家族解析式渲染 1–2 张先验通道,与相似度场并列输入;**参数是输入条件,不是回归输出,与已排除 #4 不冲突**。
- **前置(硬门)**:先测 200 条解析准确率——**禁止引用任何「LLM 槽位抽取通常 >95%」类无出处数字**(本项目有检索编造前科)。净增益期望 ≈ B2-1 的 Δ_oracle × 实测解析准确率。
- **部署合规**:推理期 LLM 调用违反低开销约束;必须 LLM 当教师、蒸馏轻量解析头(数万样本可训)——此环节**无文献支撑**,自担设计(见 §五)。
- **最小验证**:解析准确率 ≥ 门槛(预注册 90%)后,parsed-params 臂对 base 的 ΔIoU ≥ 0.5×Δ_oracle 才立项蒸馏。

### 分支 (iii):尾部欠学习主导

- **前置 0**:strata 定义协议(无 verified 支撑,自定并预注册):面积/偏心/家族参数按分位数分桶,每桶 ≥ 300 样本保 CI;REPORT 强制 head/tail 分列 + 中位数与 P10/P25 并报(Frequency-based Matcher 纪律)。
- **前置 1(合成臂硬门)**:建「指令-构图相容性检查器」——GT 渲染器只能渲染场不能合成图像,在既有图像上重采样 (指令,家族,参数) 可能产出与构图不一致的监督;检查器不落地,合成臂不开工(否则批量制造标注噪声,反向污染 (ii) 测量)。
- 处方(按摩擦递增):
  1. **InfoBatch 反向用法**:剪低 loss 样本 + 无偏重标定,省下迭代喂 B3-1 的 reducible-高池;头部不退化有梯度期望无偏保证。**最小验证**:tail 桶 IoU +≥0.03 且 head 桶 Δ ≥ −0.01。
  2. **JTT 快臂**:最低 20% IoU 样本 λ 上权重训,λ∈[5,50] 小网格。同判据。
  3. **倾斜方向 A/B**(Beyond Scaling Laws:方向是数据量级函数,数万样本属中间区):易先 / 难先 / 均匀三臂,各数小时。
  4. **失败驱动程序化重采样**(检查器落地后,本设定特权:GT 可程序生成):按 per-stratum (1−IoU) 或 reducible loss 定 (指令,家族,参数) 过采样比例(FreeMask/Feedback-guided 两纪律:靠近真实支撑 + 保持多样 + 质量过滤);进阶版 Learning-to-Simulate 外环直接优化全体 val IoU。**最小验证**:tail +≥0.05 且 head 不降。
  5. ConR 特征正则 / MEDOE 轻专家头:末位,各带 kill 判据。

### 分支 (iv):特征供给主导

- 处方(按成本递增,收益已在 B2-3 量化,直接采纳最优臂):
  1. **多层 concat**(lin.4 协议,等距 4 层 + 1×1 投影):改动最小,先行。
  2. **旁挂 DINOv3 特征流** + **Talk2DINO 式投影**把相似度先验改在 DINOv2/v3 空间计算(此臂收益按 §2.0 规则只记 (iv),不再在 (i) 重复申报)。
  3. **DVT 去噪 / 高范数格显式排除**(D0-4 门控;与项目 pad 格纪律同构,扩展到 sink 格)。
  4. **上采样**(双重门控:AnyUp 零成本先导有收益 + 误差确实集中小/偏心目标)。
- **最小验证**:每臂独立 ΔIoU ≥ +0.02 保留;叠加臂报边际 Δ 防重复计数。
- 诚实预期:全部 dense 探针数字来自 CLIP/DINO 系;「带 LLM 解码器的 VLM 中层特征」无探针文献(§五),B2-3 就是本项目自己补的这块证据。

### 测试期层(正交,最后叠加)

1. 质量回归头 best-of-k 重排(SAM 头 / EvanySeg 外置):**必配同预算中值/投票共识对照臂**;verifier 用任务特化信号(候选场×相似度场重合度、方向词×场梯度一致性——顺带补 (i))。
2. SDC 难度路由:易样本 K=1、难样本 K 大(ADE-CoT);交互产品有人在环出口 → deferral(25% deferral 移除约 80% 误差量级)。
3. 输入级 TTA 末位:只对 SDC 难样本、只用预筛 1–2 个增强(S³-TTA),骨干重跑最贵。

---

## 五、缺口清单(如实)

**修复证据的条件不匹配(最重要)**
1. (i) 的全部 token 级融合增益来自**可微调条件编码器**(EVF-SAM 微调 BEiT-3、LAVT 微调视觉编码器、SmartEdit 训 MLLM);「冻结 VLM + 只训头内新增块」下的增益**无一条 verified 实证**——(i-a) 是赌注,B2-5/B2-4 探针是它的止损线。
2. (iv) 的 dense 探针证据全部来自 CLIP/DINO 系;Probe3D 的「语言监督」= CLIP 式对比模型,**不覆盖 Qwen-VL 类带 LLM 解码器架构**——本项目特征源恰是后者,B2-3 须自建证据。
3. (i-b) 解析路线的部署环节(LLM 教师 → 轻量解析头蒸馏)无文献支撑;「离线解析推理零开销」的声明对新指令不成立。

**文献线整条缺失(待补检索,查询词)**
4. 多标注/条件隐变量分割线((ii) 修复主力缺口):`probabilistic U-Net aleatoric segmentation`、`stochastic segmentation networks multi-rater`、`PHiSeg`、`inter-rater variability segmentation 2024`。
5. 编辑域参数化蒙版直接先验(**全局最大盲区**:GT 家族 = Lightroom 局部调整原语,可能有直接同题工作):`parametric mask prediction local photo adjustment`、`graduated filter radial filter automatic placement learning`、`language-driven image editing local mask GIER LDIE`、`learning by planning global image editing T2ONet`。
6. VLM(带 LLM 解码器)特征 dense 探针:`MLLM patch features dense prediction probing 2025`、`LLaVA Qwen-VL visual features segmentation linear probe`。
7. 方向的稠密回归参数化(若上辅助几何监督):`dense orientation field estimation von Mises loss`、`circular regression deep learning`。
8. 程度/范围词接地(现有证据只覆盖方向词):`scalar adjective grounding intensity`、`vague quantifier grounding image editing strength`。
9. 指令→布局解析准确率实证:`LLM spatial layout generation faithfulness benchmark 2024`。
10. 文本 token 取层依据:`intermediate layers better representations LLM probing 2024`。
11. 免费集成:`model soups weight averaging dense prediction`、`SWA segmentation`;多选学习机制锚点:`multiple choice learning Lee 2016 WTA`。

**方法内部缺口**
12. (ii) 修复偏薄:除多假设输出外只有「清洗 GT/软目标」一句话;「众数场蒸馏」零文献;生成器侧确定化是工程主张非文献结论。
13. (iii) strata 分桶边界、每桶样本量与统计功效无 verified 支撑;合成臂的「指令-构图相容性检查器」六路均未处理,本文将其设为硬门。
14. 缩放外推 C 的可分辨性存疑(CI 大概率盖住判决区间),已降级;35M 臂剔除拟合。
15. GDE 等式在 soft-IoU 上无理论保证,只作定性;Balanced MSE 辅助回归探针**废弃**(修标签不平衡不修条件数病态,大概率复现塌缩误判)。
16. RoboPoint 式端点监督头与已排除 #4 边界模糊,无 verified 证据支撑「图像空间端点比抽象参数良态」;若立臂必须带塌缩监控、权重上限、预注册中止判据。
17. C 档数字重申:EVF-SAM 83.7(含微调)、SPRIGHT +22% / RoboPoint +21.8%(跨任务口径)、VoxPoser「解析>95%」(无出处)、NeCo 数字(测在 DINOv2 非 VLM)、Ishida 逼近 Bayes(前提是软标签可得)——引用必须带条件标注。

---

## 六、VERIFIED 参考文献表

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

*本文档由调研 subagent 于 2026-08-11 生成;所有预注册数字为设计值,须经主 agent 裁定后写入 EXPERIMENTS 计划文档方生效。*
