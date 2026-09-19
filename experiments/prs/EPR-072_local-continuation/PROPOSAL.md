# EPR-072-LOCAL：当前状态蒙版的六阶段续训与style回放

状态：方案就绪，尚未启动训练。沿用用户指定编号；已有 `EPR-072_mmart-ppr10k` 是数据任务，保留其目录与产物。本任务使用唯一标签 `EPR072LOCAL` 和独立输出根，不能覆盖原EPR-072。

## 1. 目标与基线

在EPR-071 R连续码预测上续训六阶段local，使每步动作接近记录的prev，并用global/style回放保持已有style能力。预测仍为绝对颜色码，执行仍为 `z + beta*(F(z;W)-z)`，不切换到残差码监督，不引入推理时bank检索。

默认初始化：验证集选择的R best@800，完整继承base、LoRA、8个readout token、颜色头和固定scaler；这份checkpoint已通过smoke。R final@6250作为已完成的独立对照，记录其完整400分数，但不按full400表现反向选择训练起点。

| R final@6250 | L1×100 | L2×1000 | PSNR | ΔE00 |
|---|---:|---:|---:|---:|
| val50（用户展示值） | 16.23 | 44.94 | 14.57 | 17.03 |
| full400 | 8.41 | 14.91 | 20.42 | 9.88 |

val50 A/B L1：用户展示18.02/14.43。原始聚合分别18.0217905/14.4383443，val50 PSNR=14.5757946；用户文本采用截断显示，复现实验保留原始数值。主表仅用full400的四项，不混入val50/A/B或旧checkpoint的SSIM/条件分数。

## 2. beta的确定：训练与推理共用同一规则

当前实现事实：EPR-071单段为全1支持；旧chain训练用参考图构造时记录的beta（或beta/s）；推理接口可选择input/current，默认input不能当作current。新任务必须显式固定`alpha_on=current`并记录到config。

- Global：beta=1。
- Hue / Shadows / Midtones / Highlights：在**本步输入状态**上调用同一解析支持函数，固定生产参数；教师强制时输入是记录z_k，自由滚动时是预测current。
- Subject：来自本步图像及区域文字的冻结、版本固定的where头。GT主体蒙版仅作诊断对照；不得缺省成全零或全1后声称启用了Subject。
- `beta`在本任务表示纯support，颜色强度吸收到W；执行时不再乘旧calib.s，亦不使用旧默认0.5。训练构造的状态仍保持原记录，不擅自修改。
- 计算支持时使用固定输入范围约定；如果解析函数对输入clamp到[0,1]，训练与推理必须一致，记录发生clamp的比例。颜色执行中间状态是否clamp也必须单独配置，不能继承隐式默认。

## 3. 监督码：必须对新支持重新求解

换成current支持后，禁止复用reference-mask监督码，亦禁止直接使用旧残差码。
对记录输入z_k、记录目标prev_k及新beta_k：

`H = beta_k * phi(z_k)`
`T = (prev_k - z_k) + beta_k*z_k`
`W*_k^T = solve(H^T H + lambda I, H^T T)`。

调用已验证的`mixed_codes.solve_support_code`，ridge=1e-2，保持trace缩放与下限。码监督和执行器使用同一beta。按几何/支持策略/scaler/目标状态/正则化版本给缓存建立标识；旧reference-mask、旧strength-scaled及residual包不能混读。

新支持不一定覆盖记录变化，需报告：恢复L1、相对分半稳定性、支持覆盖度及零支持区域中存在目标变化的比例。先做100条独立local链验证，不沿用global测试成绩作为六段验收。过滤失败保留原因，不将目标改成solver重建图掩盖误差。

## 4. 六阶段模型与索引

保留现有冻结几何、780维码与EPR-071 scaler。连续R读出头共享；将8个readout token扩展为6组，每组沿用同一初始化并采用零初始化的阶段embedding，先验证单组/Global输出能复现初始化checkpoint。

必须从冻结构造配置读取`step_order`，按真实退化顺序反向执行，不按论文文字硬编码另一套槽位顺序。每条记录保存`construction_slot -> inference_move -> semantic_kind`映射。

原D2及若干trainer只训练颜色slot1..5，Subject/where slot0跳过。新任务要求6个阶段都明确：若Subject是可表示的局部颜色动作，纳入码/像素监督；若原记录含几何变化且无法由颜色执行器表达，必须标明并筛除/另行构造，不能仅将slot0放入loss列表。

## 5. 训练批次与损失

两个队列按**optimizer update**严格1:1交替：global/style replay、六段local。一个update内所有梯度累积micro-batch来自同一队列，避免batch组成随累积步数漂移。

Global/style replay复用EPR-071已确认的目标口径与指令/CoT来源，包括其中原名local但已按全图LUT监督的条目，不在本轮静默改其标签。每个batch执行beta=1，使用原R目标：pixel MAE + 0.1 standardized-code MSE + 0.1 InfoNCE，温度0.07。

Local batch默认教师强制：

`L_local = mean_k [ MAE(R_beta(z_k,W_hat_k), prev_k) + 0.1*MSE(q_hat_k,q*_k) ]`。

Local InfoNCE初始设0；只有按该阶段恢复方向及新beta重新分配了可靠bank正样本，才做开启0.1的对照。禁止拿正向退化LUT id充当恢复动作标签。

每一步只监督到自己的prev，不同时对clean加同等损失。最后一步prev=clean时自然得到终点监督。后续可单独消融增加自由滚动clean损失；不得把一次单步预测直接与clean比较来训练整条链的每一步。

局部像素损失采用全图MAE并按有效阶段平均；同时记录支持内误差与支持外漂移。全零支持且目标不变是no-op，跳过码损失，不能制造“唯一可恢复代码”的标签。

建议pilot：200 updates，micro=1、有效batch=16、LoRA lr=1e-5、head/token lr=1e-4、clip=1；几何、scaler、视觉编码器和where头冻结。沿用AdamW与bf16计算/fp32主权重。两队列采样、每阶段梯度、动作幅度和direction cosine均留痕。pilot后再确定长训步数，不沿用6250当作已经批准的local预算。

## 6. style保留与评测

开训前对初始化checkpoint冻结style验证集指标与val50指标。每100 updates检查同一style验证集、local教师强制单步、local自由滚动终点与各阶段幅值；full400仅在预先确定的checkpoint上做终评，不用来挑checkpoint。

默认style回退门：style验证集L1相对初始化增加超过5%且连续两次评测出现时，停止并保留已通过门槛的checkpoint；这是本任务的预设门槛，不是已经观察到的结果。若触发，优先增加style replay比重或降低local学习率，再做单因素试验。

local模型选择以style门通过为前提，比较local验证集自由滚动误差，不用纯teacher-forced loss代替部署表现。分别报告current支持、旧reference支持/GT支持对照，避免将蒙版偏差误归因于颜色读出。

## 7. 数据隔离与启动门

- 100张ArtEdit恢复smoke产物及所有用benchmark GT求出的码只能评测，不能加入local训练/原型扩充。
- 禁止不经检查复用`mixed_data.local_sources()`中的ArtEdit oracle增广路径；显式训练白名单与基准ID/原图去重守卫必须先通过。
- 初始化R@800 smoke复现，Global单组参数移植误差检查，current-mask同规则检查，六槽映射检查，绝对码语义检查，空掩码处理，恢复方向正样本检查全部通过才开训。
- 100条local验收必须包含soft mask、零支持、强度变化、Subject以及不易恢复样本；报告拒绝率，不挑成功案例。
- 新启动入口需保存完整argv、checkpoint/scaler/geometry/where/source哈希；输出用独立`EPR072LOCAL`目录。

本任务卡给出实现与运行规格；现有`select_train.py`只消费single行，不能直接换个编号当作已经支持六阶段。当前未启动GPU训练、未重建全量current-mask local监督码。
