# MCQ-E2E 最终报告：MetaCanvas query 的 what / where 读出

实验日期：2026-08-03  
可视化与中文报告更新：2026-08-04

## 一、结论摘要

四条严格配对实验均已完成，正式目录下都有终局 `metrics.json`，实验根目录存在
`ALL_DONE`。结果是**混合结论**，不支持“带空间结构的 MetaCanvas 能同时提供更优
what 与 where 读出”这一强主张。

- **WHAT / LUT：**16×16 MetaCanvas 在 held-out preset 上优于同预算的 8-query
  MetaQuery：ΔE00 p50 为 `9.500 vs 10.121`，p90 为 `18.715 vs 20.623`。
  两者都没有坍缩，并且都对 instruction shuffle 敏感，说明接口确实携带了语义信号；
  但两者都远未达到预注册的高保真门槛（p50 `< 1.5`、p90 `< 3.0`）。因此这只是
  MetaCanvas 的弱结构性证据，不是一个已经成功收敛的高精度 LUT 预测器。
- **WHERE / w14：**两者都能较好读取普通前景/背景位置（AUC `0.891/0.902`），
  但都没有通过更困难的同图 A/B 条件区域门槛（AUC_target
  `0.562/0.604 < 0.65`）。关键条件指标和 shuffle 敏感性上，MetaQuery 反而优于
  MetaCanvas。因此，本实验没有证明空间 canvas 对 instruction-conditioned where
  是必要的或更优的。
- **联合结论：**当前结果支持“部分 what + 普通 where 可读”，但不支持“稳健、同时、
  由 instruction 决定的 what + where”。

## 二、实验契约

由于 G1--G4 没有 mask 标注，两个任务采用相互独立的监督轨道：

1. G1--G4：`I_in + instruction -> 48-primitive GLUT`，测试 what；
2. D-CONSTRUCT：`I_in + instruction -> w14`，测试 where；其中 L6 为同图分别构造
   `rega`/`regb` 指令和 `maska`/`maskb` 目标。

基础 VeraRetouch VLM 冻结；训练 rank-32 LoRA、可学习 query、层路由、connector 与任务
readout。这里的“端到端”是从最终任务损失反向传播到全部新增模块和 LoRA，而不是对
VLM 全参数微调。

- **MetaCanvas 臂：**256 个带可学习 Fourier 2D 位置的 query；读取第 11、17、23 层，
  经路由加权后与对应的 16×16 image patch token 融合。
- **MetaQuery 对照臂：**8 个一维 query；没有二维坐标，也没有 connector 内的
  patch-wise image-token 融合。
- 优化器、LoRA、任务解码器、数据和训练步数预算保持一致。
- 两臂都使用独立但同 seed 的 sampler generator（`20260803 + 17`），保证样本顺序
  配对。该控制很重要，因为 LUT 训练只看到 `18,000 / 45,724` 行，即一个 loader
  epoch 的 `39.37%`。

## 三、实际数据范围

目标是验证 3,500--4,000 个 LUT，但本次 G1--G4 normal-confidence manifest 实际只有
2,947 个唯一 preset，因此本报告不作“已验证 4,000-LUT 泛化”的经验性声明。

| 数据池 | 行数 | 唯一 preset 数 |
|---|---:|---:|
| train | 45,724 | 2,815 |
| val_img | 2,635 | 1,144 |
| val_lut | 91 | 59 |
| test_lut | 126 | 59 |
| 全部保留数据 | 48,576 | 2,947 |

Manifest SHA256：
`089e9e89af23b239ff38848f628856d377b25348f3799f6c306a97b9a893747a`。
源图通过 indexed-tar reference 随机读取，没有长期展开为小文件目录。

## 四、正式配对指标

### 4.1 WHAT：held-out `test_lut`

| 指标 | MetaCanvas | MetaQuery | Canvas − Query |
|---|---:|---:|---:|
| 最佳 step | 3,000 | 2,000 | -- |
| ΔE00 p50 | **9.500** | 10.121 | -0.621 |
| ΔE00 p90 | **18.715** | 20.623 | -1.908 |
| ΔE00 p99 | **24.405** | 25.498 | -1.093 |
| PSNR（dB） | **18.740** | 18.539 | +0.202 |
| variance ratio | 0.833 | **0.837** | -0.004 |
| instruction shuffle 降幅（dB） | 3.707 | **3.735** | -0.028 |
| cross-image shuffle 降幅（dB） | **3.874** | 3.822 | +0.051 |
| bake33 ΔE00 p50 | 0.01205 | **0.01067** | +0.00138 |
| bake33 ΔE00 p99 | 0.03116 | **0.02663** | +0.00453 |
| query 平均 cosine | -0.00281 | -0.02851 | -- |
| query effective rank | 101.43 / 256 | 2.31 / 8 | -- |

门槛判定：

- `variance_ratio > 0.6` 的非坍缩门：两者均通过；
- Canvas 的 P-test 色差低于同预算 MetaQuery：通过；
- 绝对精度 `ΔE00 p50 < 1.5, p90 < 3.0`：两者均失败；
- 标准 33³ cube 导出/回读近无损：两者均通过。

bake33 结果只说明预测出的紧凑 GLUT 可以被高保真烘焙为标准 33³ LUT，并不消除预测
GLUT 与目标 LUT 之间约 9.5 的 ΔE00 误差。

### 4.2 WHERE：source-disjoint `val`

| 指标 | MetaCanvas | MetaQuery | Canvas − Query |
|---|---:|---:|---:|
| 最佳 step | 2,500 | 2,500 | -- |
| soft-IoU p50 | **0.355** | 0.345 | +0.010 |
| 普通 mask AUC p50 | 0.891 | **0.902** | -0.011 |
| L6 AUC_target p50 | 0.562 | **0.604** | -0.042 |
| w14 RMSE p50 | **7.5886** | 7.5890 | -0.0004 |
| instruction shuffle IoU 降幅 | 0.00333 | **0.00445** | -0.00113 |
| query 平均 cosine | -0.00337 | -0.11984 | -- |
| query effective rank | 102.21 / 256 | 3.88 / 8 | -- |

完整的 selection `inner` pool 有 410 个样本；其中 L6 条件结果为 Canvas `0.613`、
MetaQuery `0.649`。后者仍略低于预注册的 0.65 门槛，并且在 source-disjoint val 上下降
到 `0.604`。Val 中有 24 条 `rega` 和 24 条 `regb`，其中 46 条的条件 AUC 有定义。

`AUC_target` 把“只属于目标区域的像素”与“同图另一个候选区域的像素”直接比较；普通
AUC 只把目标区域与一般背景比较。两者之间的大幅落差说明：高普通 AUC 不能被写成
成功的 instruction-conditioned where。

门槛判定：

- 普通空间信息可读：是；
- source-disjoint val 上 `AUC_target >= 0.65`：两者均失败；
- shuffle 降幅高于机器零：通过，但幅度很小；
- query 复制/完全坍缩：未观察到；
- MetaCanvas 在核心 where 指标上优于 MetaQuery：失败。

## 五、两个最佳 checkpoint 的效果图

为避免视觉 cherry-picking，主图中的 WHAT 案例从 `test_lut` 按 `sft_id` 排序后等距选
4 条；WHERE 案例从 source-disjoint val 的 L6 UID 排序后等距选 3 个，并对每个 UID
同时展示 `rega` 与 `regb`。第 5.3 节进一步扩展为每个任务 20 张源图。所有选择过程都
不读取逐样本效果指标。

| 任务 | 展示 checkpoint | 选择理由 | SHA256 前 12 位 |
|---|---|---|---|
| WHAT | `lut_canvas@3000` | held-out LUT 色差最优 | `2a017521d61e` |
| WHERE | `w14_metaquery@2500` | 核心条件 AUC_target 最优 | `43f5b8e1c4a8` |

### 5.1 WHAT 最佳效果：`lut_canvas@3000`

每行依次为输入图、目标 LUT 输出、预测 GLUT 输出和绝对 RGB 误差。图内 ΔE00/PSNR 在
该展示图的整图像素上计算；报告第 4.1 节的正式指标则在统一色立方采样上计算，二者不可
混用。

![WHAT 最佳 checkpoint 的 LUT 效果](viz/best_what_lut_canvas.png)

从固定样本可见，模型能产生与指令大方向一致、非恒等的全局色彩变化，但仍存在明显的
色偏、亮度或局部色彩响应误差；这与“相对优于 MetaQuery、绝对精度仍失败”的量化结论
一致。

### 5.2 WHERE 最佳效果：`w14_metaquery@2500`

每个 UID 连续两行分别为 `rega` 和 `regb`，依次展示输入图、目标区域、同图另一候选区、
预测 mask 与叠加图。图内同时给出普通 AUC 和直接区分 A/B 的 `AUC_target`。

![WHERE 最佳 checkpoint 的 mask 效果](viz/best_where_w14_metaquery.png)

固定样本直观呈现了主要失败模式：预测通常能落在大致空间位置，因此普通 AUC 较高；但
响应较宽、且同图 A/B 切换不稳定，所以一些 `rega` 的 `AUC_target` 仍接近 0.5。该图
不能被解读为稳健的条件区域选择成功。

### 5.3 每个任务 20 张稳定样本联图

WHAT 补充图从 126 条 `test_lut` 中按排序位置等距选择 20 张。每个小格内部依次为输入、
目标 LUT 输出、预测 GLUT 输出和统一色标下的 RGB 误差。

![WHAT 的 20 张稳定样本总览联图](viz/gallery20_what_lut_canvas_overview.png)

WHAT 高分辨率分页细图：
[第 1 页](viz/gallery20_what_lut_canvas_p01.png) ·
[第 2 页](viz/gallery20_what_lut_canvas_p02.png) ·
[第 3 页](viz/gallery20_what_lut_canvas_p03.png) ·
[第 4 页](viz/gallery20_what_lut_canvas_p04.png)。

WHERE 补充图从 24 个 source-disjoint L6 UID 中等距选择 20 张源图；每张同时展示 A/B
两条指令，共 40 次模型读出。青色表示 GT，红色表示预测。`AUC_target` 的 A/B 数值直接
写在每格标题中；如果某一侧没有可用的目标独占/另一候选区域像素池，则如实标记为
`n/a`，不纳入条件 AUC 聚合。

![WHERE 的 20 张 L6 源图总览联图](viz/gallery20_where_w14_metaquery_overview.png)

WHERE 高分辨率分页细图：
[第 1 页](viz/gallery20_where_w14_metaquery_p01.png) ·
[第 2 页](viz/gallery20_where_w14_metaquery_p02.png) ·
[第 3 页](viz/gallery20_where_w14_metaquery_p03.png) ·
[第 4 页](viz/gallery20_where_w14_metaquery_p04.png)。

20 张选样、checkpoint hash 和逐案例指标分别记录在
`viz/gallery20_what.json` 与 `viz/gallery20_where.json`。

## 六、MetaCanvas / MetaQuery 最终 feature 可视化

### 6.1 可视化对象与公平口径

这里的“最终 feature”严格指 `ex["canvas"]`：第 11、17、23 层经 softmax route 加权，
再通过 connector 的 `out_norm` 后、任务 readout 前的 memory。它不是原始 vision
patch feature。

- MetaCanvas：`[B, 256, 384]`，可恢复为 16×16 token 网格；展示 token PCA 的
  PC1/PC2、去 token 均值后的 L2 norm 和 256×256 centered-cosine Gram。
- MetaQuery：`[B, 8, 384]`，8 个 token 没有二维坐标；展示真实的 8×384
  token/channel feature、PC 得分、norm 和 8×8 Gram，**不把它伪装成二维空间图**。
- 跨结构 PCA：每个样本先对 token mean-pool 成 `[384]`，再把两臂等样本数联合 PCA。
  这样不会让 256-token Canvas 相对 8-token MetaQuery 获得 32 倍统计权重。

需要特别注意：MetaCanvas 的空间 PC 图同时包含 VLM query 状态、patch 融合和可学习
Fourier 2D 位置编码。出现连续空间结构只说明最终表示保留了二维组织，不能单独证明
某个 PC 就是语义区域、也不能证明该空间组织被任务 readout 有效使用。跨结构 PCA 中的
明显分离同样只表明两种接口形成了不同的表示族，不等价于精度优劣。

### 6.2 WHAT 最终 feature

两臂都使用第 5.1 节相同的 4 个稳定样本。token 细图固定使用排序后的第一个样本；底部
联合 PCA 对 4 个样本等权。

![WHAT 的 MetaCanvas 与 MetaQuery 最终 feature](viz/feature_what_canvas_vs_metaquery.png)

量化上，MetaCanvas 的 effective rank 为 `101.43 / 256`，MetaQuery 为
`2.31 / 8`。因此 Canvas 确实形成了更高维、空间组织更丰富的 memory；但结合高绝对
色差，这还不能写成“高维 feature 已被成功解码为精确 LUT”。

### 6.3 WHERE 最终 feature

两臂都使用第 5.2 节相同的 3 个 L6 UID、共 6 条 `rega/regb` 输入。token 细图固定使用
`L6_val_0000/rega`。

![WHERE 的 MetaCanvas 与 MetaQuery 最终 feature](viz/feature_where_canvas_vs_metaquery.png)

量化上，MetaCanvas 的 effective rank 为 `102.21 / 256`，MetaQuery 为
`3.88 / 8`。Canvas 的二维 feature 没有发生简单复制坍缩；但它在核心
`AUC_target` 上仍落后于 8-query 对照，说明“保留二维结构”和“成功使用 instruction
选择同图目标区域”是两件不同的事。

可视化脚本与可审计选样/逐案例指标分别保存在 `visualize_best.py`、
`viz/what_visualization.json` 和 `viz/where_visualization.json`。

## 七、“GLUT 参数”具体指什么

这里预测的是紧凑 Gaussian LUT 的结构化参数，而不是 33³ dense cube 的逐格值。
48 个 primitive 时，raw head 输出：

`48 × 23 个 primitive 值 + 12 个全局值 = 1,116 个值`。

`ParamHead` 再把 raw 值转换为带边界的 `mu / sigma / covariance offset / opacity /
gate / local affine / global affine` 等渲染参数。最终色彩变换可以被采样成标准
33×33×33 RGB LUT，即 107,811 个标量表项。因此，较强的 Transformer 可以保留在
训练或服务器侧负责条件参数生成，而交付端仍然是普通、轻量的 33³ LUT。

## 八、建议的贡献表述

结合独立的 RD-G Stage-1 Transformer-vs-MLP 结果与本实验的导出回读结果，推荐写成：

> 我们发现，系统的主要容量瓶颈位于条件参数生成，而非 LUT 执行。相较浅层 MLP，
> Transformer 参数生成器能够更有效地把图像—指令上下文映射为紧凑的 48-primitive、
> 1,116-value Gaussian-LUT 表示；该预测变换又可近无损烘焙为标准 33³ LUT。由此，
> 表达能力较强的条件推理与轻量、标准化的部署后端可以解耦。

论文中使用“更有效”或“更强”时，必须紧邻 RD-G Stage-1 的 Transformer-vs-MLP
定量表。本次 MCQ 实验单独不足以支持“必须使用 Transformer”：MetaCanvas 的 LUT
优势较小，且空间 query 臂在条件 where 上输给了 8-query 对照。

## 九、backend 是否支持泛化到 4,000 LUT

需要区分两个问题：

1. **backend / LUT 库容量：原则上支持。**渲染器固定输出 1,116 个参数，没有
   4,000-way 的类别专用 head；preset 库变大不会改变输出维度。近零 bake33 回读误差
   说明紧凑输出可以稳定交付为标准 LUT。
2. **生成器对 4,000 个 preset 的泛化：尚未验证。**本次只有 2,947 个唯一 preset，
   训练涉及 2,815 个，held-out test 只有 59 个；而且 held-out 色差仍远高于绝对门槛。
   要证明 4,000-LUT 泛化，需要先建立完整的 4,000-preset authority，做严格的
   preset-disjoint 划分、均衡训练曝光，并在该划分上通过绝对色差指标。

因此安全表述是：“backend 与类别数无关，兼容 4,000-LUT 库”，不能写成“模型已经
证明能泛化到 4,000 个 LUT”。

## 十、有效性修复与排除项

正式终局实验前发现并修复了以下问题：

- query 插在每个样本有效 multimodal prefix 之后、batch padding 之前，并通过显式 span
  读取；
- 构造新 connector/readout 前恢复 Llava `disable_torch_init` 对 Linear/LayerNorm 的
  全局 monkey patch；
- 使用 eager attention 与 finite guard，在首个非有限值处立即失败；
- D-CONSTRUCT NPZ 在 DataLoader fork 前完整 materialize，避免共享 ZipFile offset
  race；
- w14 checkpoint selection 使用完整 410 样本 inner pool，不再使用只有一组 L6 A/B
  的前缀；
- 两条结构臂使用相同的显式 sampler 序列。

NaN smoke、无效配置和早期未配对运行保留在 `_smoke*`、`_invalid_*` 与
`_pilot_unpaired_*` 下供审计，但本报告的所有数字与图片都不使用这些结果。

## 十一、结果直接导出的下一步实验

- **WHAT：**改用 preset-balanced schedule，至少覆盖一个完整 G1--G4 epoch；先建立
  真正的 4,000-preset authority 和 preset-disjoint split，再讨论库规模泛化。
- **WHERE：**增加同图 A-vs-B 的直接 ranking/contrastive loss，并对 canvas slot 或
  query route 加空间监督。当前全局 w14 head 可以获得高普通 AUC，同时丢掉 MetaCanvas
  真正要验证的 instruction-conditioned 区域区别。
