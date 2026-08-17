# ICLR 2026 投稿准备报告：Instruction-Guided Local Retouching

**版本**：2026-08-16 21:55 CST  
**范围**：当前 `where` / `what` 双分支、公开评测构造与后训练计划  
**阅读约定**：本报告按 9 页版式组织。图优先于表；运行中作业仅作状态快照，不作为论文结果。训练数据不公开，构造代码、配置和公开测试协议计划公开。

---

## 第 1 页：任务、主张与贡献

### 任务

输入一张图像 (I) 和自然语言指令 (q)，目标是完成**语言约束的局部色彩/色调精修**：既要执行正确的颜色变化，也要把变化约束在指令所指定的连续空间范围内，并保持非目标区域不变。

当前系统将输出写为：

\[
\hat I=(1-\alpha)\odot I+\alpha\odot f(I).
\]

其中，(alpha\in[0,1]^{H\times W}) 是 `where` 产生的连续编辑场，(f) 是 `what` 产生的颜色函数。模型输入只有图像和文字；掩膜、几何参数和 LUT 是训练或评测监督，不是推理输入。

### 贡献表述（投稿叙事）

1. **Instruction-guided local retouching 框架。** 提出一个由 VLM 驱动的局部色彩/色调精修框架：将“编辑在哪里”和“如何改颜色”因子化为连续空间场与颜色变换，并以可微合成式联结，使两部分既可独立监督，也可端到端优化。重点不是把传统全局修图器套上文字，而是学习指令中的**范围-动作绑定**。

2. **可复现的局部精修数据构造协议。** 从真实图像、真实预设和闭式连续几何场构造指令、空间场、颜色操作及其 provenance。该协议提供 `where` 与 `what` 的参数空间真值和同图反事实配对。训练数据本身不发布；构造流程、代码和配置计划发布。

3. **反事实优先的公开评测协议。** 公开派生测试集和构造方法，分别衡量范围定位、目标区域编辑质量/非目标保持，以及同图改变范围或动作后的指令敏感性。协议包含 shuffle、irrelevant words、fixed phrase、antonym 与 subject-swap 等负控制，避免把“无条件平均修图”误判为指令理解。

### 贡献边界

GLUT/CGLUT 是 `what` 的颜色函数载体，不是本工作的贡献。我们也不使用“首个”或“无需外部编辑器”作为卖点；后者只是在与基于工具执行的工作比较时的实现差异。

<div style="page-break-after: always;"></div>

## 第 2 页：当前框架

```text
image I + instruction q
          |
          v
 frozen Qwen3-VL-4B + task SFT
          |
    +-----+---------------------+
    |                           |
<seg_where>                 <seg_color>
    |                           |
    v                           v
continuous field alpha      z in R^2560 -> GLUT parameters theta
    |                           |
    +------------ differentiable composition ------------+
                              |
                              v
              I_hat = (1-alpha) I + alpha f_theta(I)
```

`where` 的读出是一个指令条件的连续场，不等同于二值语义分割。当前最有效的读出路径是 special token 写入结合语言侧 LoRA：多个 query 产生候选场，再由选择头选择一个输出。这个机制是当前框架的一部分，而非单独的任务贡献。

`what` 使用 48 个 3D 色域高斯的 GLUT。每个高斯包含颜色中心、Cholesky、透明度及局部仿射，另加全局仿射，共 (22N+12=1068) 个参数。`<seg_color>` 的 (z\in\mathbb{R}^{2560}) 经解码器映射为这些参数。当前 E030/E031 主线均冻结 VLM 并缓存 (z)。

### 与相邻工作的任务差异

| 工作 | 核心任务 | 局部范围是否由当前指令预测 | 与本工作的关键差异 |
|---|---|---:|---|
| VeraRetouch | 多任务、全局照片精修 | 否 | 有语言计划和可微渲染，但颜色/光照 latent 作用于整图；不学习范围-动作绑定。 |
| AceTone | 指令条件全局色彩分级 | 否 | 用 VQ LUT token 与自回归预测全局 grade；本工作将连续颜色函数与空间场组合。 |
| JarvisArt / JarvisEvo | 工具驱动的照片编辑 | 有局部操作声明 | 通过 Lightroom 工作流执行；公开评测未给出指令特异的连续范围真值和反事实范围协议。 |
| GLUT / CGLUT | 全局颜色映射表示 | 否 | 是本工作的函数载体来源；原始条件是闭集 embedding，不承担语言局部精修任务。 |
| PPR10K / RSFNet | 人像优先或区域特异自动修图 | 否 | PPR10K 以人像区权重训练全局模型；RSFNet 自动产生区域滤镜，均不处理语言范围-动作绑定。 |

PPR10K 的“局部”是人像区损失加权（人像权重 5、背景权重 1），不是对每条指令预测编辑范围。RSFNet（常被误写为 RFSNet）使用 attention 与区域特异滤镜，但主实验主要报告整图 PSNR/SSIM/DeltaE 和可编辑性示例，没有语言条件范围 GT、定位 headline 或同图反事实测试。因而它们适合做问题边界参照，不应与本任务直接拼接数值排名。

<div style="page-break-after: always;"></div>

## 第 3 页：`where` 要解决什么，以及已测试什么

`where` 要回答的不是“图中有什么”，而是“这条指令要求**哪些像素以多大强度接受该编辑**”。当前已测问题按优先级为：

- **选择问题**：K=8 的候选场中，best-of-K 为 0.8072，而当前选择器输出为 0.7571，`sel_is_best=0.128`。已测 headroom 约 0.050，是最大缺口。
- **缩放问题**：K8 续训到 3500 步后为 0.7571，低于现役 M0 的 0.79095（配对 (-0.0362)，(p=10^{-4})）。`ST_LANG_CONT@3500 vs M0` 仍未跑成，是关键决策实验。
- **边界与几何问题**：监督网格为 (32\times48)，约每格 16 px；尤其是斜向 `band` 场常退化成 radial 或近均匀场。**band 退化属于 `where`，不是 `what`。**
- **评测问题**：当前 headline 是已知 GT 面积的 matched-area top-k IoU，适于诊断，但不适合作为对外可比主指标；固定阈值列尚未实现。

为这些问题已迁移并测试 LISA/SEG token 到 SAM decoder（SEGSAM）、SAM/SAM2 解码器（SAMDEC）、Mask2Former masked cross-attention、K-Net、DETR/Mask2Former 一对多 query、PointRend、LIIF、ViTMatte 与 CondInst。它们不是纸面比较：每条路线均已在同一 V_where 设置下出数或被门控判负。

### 指标说明

**top-k IoU 越高越好。** 对每个样本，令 k 等于 GT 编辑区域面积，再将预测连续场的 top-k 像素二值化，与 GT top-k 像素计算交并比。它衡量预测范围是否吻合；但因为 k 使用了 GT 面积，它是内部诊断指标，投稿前必须补充固定阈值/非 oracle 范围指标。

<div style="page-break-after: always;"></div>

## 第 4 页：`where` 可视化诊断，重点是 band

![新的 source-unique band 反例：左侧保留 instruction 和 VLM 生成的 `<where>` reasoning；右侧为输入、GT alpha、ST_LANG、SEGSAM、MATTE。](assets/where_selected_rows_20260817/where_band_counterexample_b15f0d1e99fa.png)

**图 1：band 家族的连续编辑场反例。** 图内严格只含一个样本行：左侧矩形框保留原始 instruction 与模型生成的 `<where>`，右侧五列为输入、GT alpha 与三种读出。所有场图固定色标 (0\ldots1)，不是逐图 min-max。该样本的 GT 是斜向条带，而三种方法都集中到图像内容附近，图内 fresh replay 的 IoU 为 `0.268 / 0.295 / 0.367`（ST_LANG / SEGSAM / MATTE）。这些逐例数字来自本次使用 v2seg 生成 context 的重推理，只用于图像诊断；下表的正式 board 数字仍来自各原始 run。

全局结果与图像诊断一致：

| 路线 | V_where top-k IoU | 读法 |
|---|---:|---|
| ST（token 写入 + visual LoRA，双种子） | 0.7739 / 0.7659 | 当前该分支的最佳 ST 结果。 |
| ST_LANG（本图对应 run） | 0.7455 | 图 1 的参考读出；`fix`/`irrelevant` 负控制均有显著变化。 |
| SEGSAM / MATTE / SAMDEC / LIIF | 0.7208 / 0.7005 / 0.6938 / 0.6418 | 均未超过 ST_LANG；后两者还出现门控或指令控制问题。 |
| PRND / CONDINST | 0.1717 / 0.1692 | 常量场退化，低于随机 top-k 地板 0.2254。 |

还有两个不能回避的结论。第一，`best-of-K=0.8072` 明确说明候选集合尚有信息，而选择器没有取到。第二，现役 M0 的 0.79095 高于上述 ST 结果，当前不能以 `where` 宣称已经超越 M0 或达到 0.85 的可用线。

> 图像 provenance 注意：本页与补页均由 `visualize_where_selected_rows.py` 从 `V_where` 新选 source-unique 样本重新推理生成，不再使用旧总览图的裁剪面板。选择规则、原始 instruction、生成 `<where>`、运行权重与 fresh replay IoU 都保存在同目录 `manifest.json`。

<div style="page-break-after: always;"></div>

## 第 5 页：`what` 要解决什么，已完成的实证

`what` 解决的是：在文字条件下预测一个足够准确、非塌缩的颜色函数 (f)。当前它先在**全局边界**下验证，即选择 `style` 样本且 (alpha=1)；这隔离了颜色函数能力，不能替代端到端局部精修评估。

主要问题与已得到的证据如下。

- 早期 CARRIER（DeltaE00 7.895）甚至没有超过训练库平均 LUT（7.6323）；IDGATE 为 10.149，说明“能运行”不等于使用了条件信息。
- 旧 hue/chroma 项未归一化，`10*L_hc` 实际约为重建项的 325 倍；删除失衡项后，纯函数值 L1 成为当前可靠基线。
- 一条旧条件 MLP 的 ReLU 死亡率从 step 0 的 0.469 升至 step 1300 的 1.000，输出逐位相同；硬 clamp 还使饱和点梯度为零。
- 文字到 LUT 是一对多映射，因而必须越过“按风格桶检索”的无条件/弱条件下界，同时保持负控制差异。

![两个 style 样例的 what 分支全局边界可视化：输入、由数据生成 law 得到的 LUT 目标、E030 MLP 预测与逐像素 DeltaE00。](assets/what_global_20260816/what_global_e030_mlp_group_03.png)

**图 2：`what` 的全局边界效果。** 两例均为 `style`，所以图中整幅图均由 `what` 处理。预测已经能跟随复古压彩和近单色方向，但误差图仍显示纹理、高光和局部色域中的残余差异。这是颜色函数拟合质量的可视化，不是 `where` 的成功证据；其余六个样例见文末可视化补页。

### 指标说明与完成结果

**DeltaE00 越低越好。** 它在 CIE Lab 感知颜色空间中度量预测与目标颜色差异；本报告的 `what` headline 是对每个样本合成图的平均 DeltaE00。

已完成的 `E030_P4_MLP`（447,212 参数、旧 `32x256=8192` 色/步口径）为 **4.7338**，独立 seed 为 **4.6254**；优于恒等 8.2926、训练库平均 7.6323、桶检索 6.1553（相对桶检索 (-1.4215)，Wilcoxon (p=3.5\times10^{-28})），库内 oracle 为 0.8253。PSNR 诊断为 28.0184。三种文字负控制均产生非零差异，说明该结果不是单纯的固定 LUT。

<div style="page-break-after: always;"></div>

## 第 6 页：`what` 正在跑什么

![E031 运行中快照：MLP 条件维度与 L8 数据消融。](assets/what_global_20260816/what_e031_running_mlp.png)

![E031 运行中快照：query decoder 稳定性扫描。](assets/what_global_20260816/what_e031_running_qdec.png)

**图 3：新色批口径的运行曲线。** 新设置是 `256x8192=2,097,152` 色/步，40 epoch 共 18,760 step；与第 5 页旧色批结果不可直接比较。曲线纵轴是固定 selection subset 上的 DeltaE00 快照，仅用于检查训练是否稳定和决定哪些 run 值得完成；不是 full-board 结果，也不是 checkpoint 选择结论。

截至本报告截点，以下作业仍在运行：

- `E031_MLP_LR1E3`：新口径 MLP 对照，最近 step 12,663，快照 4.333。
- `E031_MLP_CD256_L3` / `CD512_L3`：缩小条件维度，最近 step 17,353，快照 4.135 / 4.098。
- `E031_MLP_NOL8_L3`：去掉 L8 的数据消融，最近 step 11,256，快照 4.100；只回答数据轴，不能直接和含 L8 的总量比较。
- `E031_QDEC_LR4E3`：退化 memory-row=1 的稳定对照，step 12,663，快照 5.806。
- `E031_QDEC_MEM4_LR3E4`：唯一仍在运行的非退化 attention 路径，step 10,318，快照 6.693。
- `EPR028R1_G3_A1/A2`：全 LUT oracle 的 4D Gaussian 条件切片，运行中；A3 排队。这一组先回答 GLUT 载体在 oracle 条件下的表达能力，而不回答语言条件化。

已有稳定性结论：`qdec-mem-rows=1` 的 cross-attention 数学上退化为与 query 无关的单 key 映射，因此不能据此声称“MLP 优于 transformer”。memory=2 曾在约 step 5,699 NaN，memory=4 仅在降低 lr 后仍存活，尚无一条非退化 attention 完整出板。所有作业结束后必须检查 `steps.jsonl` 是否出现 `L_rec=null`；该守卫漏洞曾让 NaN run 被错误发布为 0.0 headline。

<div style="page-break-after: always;"></div>

## 第 7 页：公开测试集与反事实协议

### 公开数据的角色

计划以 MIT-Adobe FiveK、PPR10K 等公开源数据构造**派生测试基准**，最终使用哪些来源及 split 需完成数据污染审核后确定。公开内容是测试集/测试构造与协议；训练数据不公开，训练构造方法公开。

FiveK 与 PPR10K 只有 before/after 图像对，没有 LUT 真值，也没有自然语言指令。因此不能把它们直接塞进当前 `what` 监督并称为 instruction following。

```text
before/after pair
     |
     +--> fit a GLUT theta_i on paired pixels --> what supervision / carrier test
     |
     +--> fit a conventional 3D LUT ----------> independent control
     |
     +--> synthesize edit instruction + matched counterfactuals
                                                       |
                                      public derived benchmark (three axes)
```

### 处理方案

对每个图像对拟合一个 GLUT：

\[
\min_{\theta_i}\ \|f_{\theta_i}(I_i^{in})-I_i^{out}\|_1,
\]

采样应来自图像实际像素颜色分布，而不是只在均匀 RGB 网格采样。产出的 1068 维参数可作为 `what` 监督。并行拟合标准 33³ 3D LUT（带 monotonicity/TV 正则）作为表达残差和同类方法的对照。

三道硬门：

1. 报告拟合 DeltaE00 的完整分布，并按预先确定的残差阈值筛除不适合“全局 LUT”表示的对；局部 dodge/burn 残差可转为 `where` 的候选样本，而不伪装为 `what` GT。
2. 审核 PPR10K 与现有 MMArt/PPR 来源的重合风险。在处理方案定稿前，不报告 PPR10K 公开评测数字。
3. 对公开集自动修图只报告其作为颜色载体测试；若构造指令，就必须同步构造 shuffle、irrelevant、fixed phrase、antonym 和 subject-swap，才能作为指令条件证据。

公开评测的三个主轴是：范围定位；目标区编辑/非目标保持；同图改变范围或动作后的反事实响应。任何单一 PSNR、SSIM 或整图 DeltaE 表都不能替代这三轴。

<div style="page-break-after: always;"></div>

## 第 8 页：后训练方案与结果纪律

当前 `what` 优化函数值 L1，`where` 优化 (32\times48) 网格 BCE；二者尚未直接优化图像空间 PSNR 或 LPIPS。后训练先做监督式联合微调：冻结 VLM，训练 conditioner 与 `where` 头，在端到端合成图上保留函数约束并加入图像约束：

\[
\mathcal L=
1.0\mathcal L_{fn}+0.5\|\hat I-I^*\|_1+0.1\,\mathrm{LPIPS}(\hat I,I^*).
\]

其中 (mathcal L_{fn}) 保留对 LUT/GLUT 函数值的监督，防止图像损失将颜色函数推向训练集平均编辑。起点采用 StatLUT 的函数值与图像双监督形制，加入 FlowLUT 常用的 0.1 LPIPS 权重；这些只是起始配方，必须做权重扫描而不是当作论文结论。

后训练的评测纪律：

- 预注册的指令条件板与图像质量板并存。前者保留 DeltaE00、top-k IoU、五个平凡基线、三类负控制与运行时断言；后者新增 PSNR、LPIPS、SSIM。后者不能覆盖前者。
- 每个后训练 checkpoint 都重跑三类文字负控制及范围/action counterfactual。若图像损失改善 PSNR/LPIPS 但降低条件差异，应报告为退化而不是“更好”。
- 只有在三轴公开协议稳定后，才考虑偏好/RL 后训练。reward 必须同时奖励编辑质量、非目标保持和指令反事实响应；否则会偏向无条件“更好看”的平均修图。

<div style="page-break-after: always;"></div>

## 第 9 页：投稿前的执行优先级与风险

### 近期执行顺序

1. **清理现有证据。** 修 `where` 总览图标题，给全部已完成 `where` 臂补固定阈值/非 oracle 范围指标；修复 NaN 守卫，使任何 `L_rec=null` run 无法发布。
2. **完成两个 where 决战。** 优先攻选择器的 0.050 gap；然后跑 `ST_LANG_CONT@3500 vs M0`。在此之前，不能声称统一 query 头的缩放有效。
3. **收敛 what 主干。** 收齐 E031 与 G3 任务，按完整 V_what board、负控制、NaN 守卫和独立 seed 判决；不将运行曲线写进论文主结论。
4. **落地公开派生基准。** 先完成 FiveK/PPR10K 的 GLUT/3D LUT 拟合残差审计、PPR 污染处置和指令反事实构造，再决定公开源与 split。
5. **做端到端后训练。** 用第 8 页的 Stage-1 起点逐项消融，并检查是否以条件性换取了图像质量。

### 当前投稿风险，必须如实写入内部清单

- `where` headline 的 top-k IoU 使用 GT 面积，外部不可比；固定阈值列尚缺失。
- 当前视觉 LoRA ST 的 0.7739 不高于现役 M0 的 0.79095；`band` 是明确的失败家族。
- `what` 新色批尚无完整结果；旧 E030 的 4.7338 不能与 E031 曲线横比。
- 非退化 attention 没有稳定完整 run；不能作 MLP/Transformer 的优劣结论。
- FiveK/PPR10K 缺失 LUT 与指令，且 PPR10K 存在待处置的数据重合风险。

### 参考工作（用于论文 related work 的起点）

VeraRetouch（2026）；AceTone（2026）；JarvisArt（arXiv:2506.17612）；JarvisEvo（arXiv:2511.23002）；GLUT/CGLUT（2026）；PPR10K（CVPR 2021）；RSFNet（ICCV 2023，arXiv:2303.08682）；以及 LISA、SAM/SAM2、Mask2Former、K-Net、PointRend、LIIF、ViTMatte、CondInst、NILUT、StatLUT。

---

## PPT 可视化补页（不计入上述 9 页内容）

每张 PNG 只展示**一个** source-unique 样本：左侧为 instruction 和 VLM 生成的 `<where>` reasoning 矩形框；右侧固定为输入、GT alpha、ST_LANG、SEGSAM、MATTE 五个视觉面板。所有场图色标固定为 0 到 1。下列逐例 IoU 是 fresh v2seg-context replay，用于读图，不替代正文已完成实验的正式 board。

<div style="page-break-after: always;"></div>

### W1：radial 正例

![radial 正例 1。](assets/where_selected_rows_20260817/where_radial_positive_0111c1f2ff0f.png)

![radial 正例 2。](assets/where_selected_rows_20260817/where_radial_positive_08ee43843a61.png)

两例的 ST_LANG / SEGSAM / MATTE IoU 分别为 `0.961 / 0.869 / 0.904` 与 `0.880 / 0.874 / 0.891`。它们说明当前读出并非普遍失效。

<div style="page-break-after: always;"></div>

### W2：band 正例

![band 正例。](assets/where_selected_rows_20260817/where_band_positive_dd657c8358cd.png)

第 4 页已经展示 source-unique 的斜条带反例（`0.268 / 0.295 / 0.367`）；本附页只保留另一张 source-unique 正例，三种读出为 `0.922 / 0.855 / 0.788`。二者共同展示现有读出对定向、非主体中心几何的真实缺口和组内波动，而不会在 PPT 中重复同一张图。

<div style="page-break-after: always;"></div>

### W3：linear 几何

![linear 正例 1。](assets/where_selected_rows_20260817/where_linear_positive_fb79ad8ee9ab.png)

![linear 正例 2。](assets/where_selected_rows_20260817/where_linear_positive_4535cbd74909.png)

两例分别为 `0.935 / 0.882 / 0.747` 与 `0.865 / 0.899 / 0.792`。它们展示出三种读出在宽线性范围下的不同边界形状。

<div style="page-break-after: always;"></div>

### W4：semantic 正例与边界差异

![semantic 正例 1。](assets/where_selected_rows_20260817/where_semantic_positive_160ff1f79956.png)

![semantic 正例 2。](assets/where_selected_rows_20260817/where_semantic_positive_28cc4b720c67.png)

两例分别为 `0.849 / 0.906 / 0.897` 与 `0.741 / 0.895 / 0.810`。SEGSAM 与 MATTE 在这两个语义边界上更强，而 ST_LANG 在几何正例上更突出；这提示读出之间存在互补性，不能把任何一头写成普适最优。

<div style="page-break-after: always;"></div>

### C1：`what` 全局边界样例，暗调与冷灰

![what 全局边界样例组 1。](assets/what_global_20260816/what_global_e030_mlp_group_01.png)

<div style="page-break-after: always;"></div>

### C2：`what` 全局边界样例，冷绿与青橙

![what 全局边界样例组 2。](assets/what_global_20260816/what_global_e030_mlp_group_02.png)

<div style="page-break-after: always;"></div>

### C3：`what` 全局边界样例，复古压彩与近单色

![what 全局边界样例组 3。](assets/what_global_20260816/what_global_e030_mlp_group_03.png)

<div style="page-break-after: always;"></div>

### C4：`what` 全局边界样例，银黑白与高键冷调

![what 全局边界样例组 4。](assets/what_global_20260816/what_global_e030_mlp_group_04.png)

上述八例均为 α=1 的 `style` 样例：图的目的仅是呈现 `what` 对全局颜色函数的预测质量、强编辑方向与误差位置。它们不被用来证明局部范围预测。

<div style="page-break-after: always;"></div>

### R1：E031 MLP 运行曲线

![E031 MLP 运行曲线。](assets/what_global_20260816/what_e031_running_mlp.png)

<div style="page-break-after: always;"></div>

### R2：E031 query decoder 运行曲线

![E031 query decoder 运行曲线。](assets/what_global_20260816/what_e031_running_qdec.png)

两张曲线均是运行中 selection-subset 快照，不计入正式结果；它们用于 PPT 中解释“为何当前优先收敛 MLP 控制组、同时单独处理非退化 attention 的稳定性”。

---

## 产物与复现备注

- `where` 的当前 PPT 图由 [visualize_where_selected_rows.py](../tools/visualize_where_selected_rows.py) 重推理生成；选择、文本、source 去重、权重与逐例结果见 [manifest.json](assets/where_selected_rows_20260817/manifest.json)。旧的 [where_arms_overlay.png](assets/where_arms_20260815/where_arms_overlay.png) 保留为历史总览，不再作为 PPT 裁图来源。
- `what` 全局边界样例由 [visualize_what_global.py](../tools/visualize_what_global.py) 重建；样例元数据同目录 JSON。
- E031 运行曲线由 [visualize_what_progress.py](../tools/visualize_what_progress.py) 从运行日志生成。
- 所有已完成数值的主来源：[ICLR_SUBMISSION_BRIEF_2026-08-16.md](ICLR_SUBMISSION_BRIEF_2026-08-16.md)。
