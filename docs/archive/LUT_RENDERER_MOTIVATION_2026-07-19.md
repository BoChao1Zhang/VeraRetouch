> **实施规范说明（2026-07-19）**
>
> 本文只保留研究动机、假设推导与后续方向。当前实验唯一具有规范效力的实施公约是
> [`LUT_RENDERER_EXPERIMENT_PROTOCOL.md`](LUT_RENDERER_EXPERIMENT_PROTOCOL.md)。
> 若本文中的旧建议、数据规模、训练阶段、loss、baseline 或指标与实施公约冲突，
> 一律以实施公约为准。

对，这样的研究主线就比“把 VeraRetouch 的三个 latent 换成 CGLUT latent”更独立、更完整：

[
\boxed{
\text{一个指令条件化 Query Token}
\rightarrow
\begin{cases}
\text{Where：空间注意力图}\
\text{What：调色风格表示}
\end{cases}
\rightarrow
\text{Attention-conditioned 4D Gaussian LUT}
}
]

而且它恰好针对 VeraRetouch 公开承认的局限：当前 Retouch Renderer 本质上是全局逐像素 RGB 映射，缺乏 pixel-wise local retouching，论文也把引入像素级 mask 机制列为未来工作。([arXiv][1])

先纠正一个术语：VeraRetouch 论文中的 L、GC、SC 三个 token 是 **special retouch tokens**，它们最后一层的 hidden feature 经 Retouch Adaptor 生成三个控制 latent；公开结构并没有把它们设计成受 mask 监督、显式对 vision tokens 做 cross-attention 的空间 query。([arXiv][1])

因此，你提出的单 Query Token 不是简单减少 token 数量，而是改变 token 的功能：

* VeraRetouch token：读取全局调色控制参数；
* 你的 Query Token：同时决定 **编辑哪里** 和 **应用什么颜色变换**。

---

# 一、整体结构：一个 Query Token 同时输出 Where 和 What

建议增加一个明确的特殊 token：

[
\texttt{<LOCAL_EDIT>}
]

VLM 根据图像和编辑指令生成该 token 的 hidden state：

[
h_q=\operatorname{VLM}(I,t)_{\texttt{<LOCAL_EDIT>}}.
]

然后不要直接读取 VLM 内部已有的 self-attention map，而是在 VLM 后面增加一个显式、可监督的 Query-to-Vision Cross-Attention 模块：

[
V={v_1,\ldots,v_P}
]

表示 Vision Encoder 输出的空间 patch tokens。计算：

[
s_p=
\frac{
(W_qh_q)^\top(W_kv_p)
}{
\sqrt d
}.
]

这个相似度同时产生两种输出。

## 1. Where：空间注意力

不要直接用 softmax attention 当 mask。softmax 满足：

[
\sum_p A_p=1,
]

会导致目标区域面积较大时，每个位置的概率被摊薄，也不适合多个不连通区域。

更合适的是使用可校准的 sigmoid map：

[
a_p=
\sigma\left(
\frac{s_p-\tau}{T}
\right),
]

其中：

* (a_p\in[0,1]) 是第四维 context；
* (T) 是温度；
* (\tau) 是可学习阈值。

训练时用精确 mask 监督 (a_p)，推理时不需要用户提供 mask。

## 2. What：调色风格

同一个 Query Token 对视觉信息进行聚合：

[
\bar v
======

\sum_p
\operatorname{softmax}(s)_p W_vv_p,
]

然后通过另一个 projection head 预测 CGLUT style code：

[
z=
A_{\text{style}}([h_q,\bar v]).
]

所以同一个 Query Token 产生：

[
\boxed{
a(x,y)=\text{编辑区域}
}
]

以及：

[
\boxed{
z=\text{需要应用的 LUT 风格}
}
]

结构可以概括为：

```text
Input Image ──→ Vision Encoder ──→ Vision Tokens V ─────────────┐
                                                                │
Instruction ──→ VLM ──→ <LOCAL_EDIT> hidden state hq            │
                              │                                  │
                              └──── Explicit Cross-Attention ────┘
                                      │                 │
                                      │                 └─ Attention map a(x,y)
                                      │
                                      └─ Query feature → Style code z
                                                        │
                                                        ▼
                                                      CGLUT
                                                        │
                                                 3D color transform
                                                        │
RGB + attention a(x,y) ──────────────→ 4D Gaussian LUT ─┘
```

这种“双头单 Query”比让一个混合 latent 同时隐式承担风格和空间信息更稳定：

* attention logits 表示 where；
* query feature 表示 what；
* 两者共享语义推理，但输出结构解耦。

---

# 二、第四维应该是 Attention，而不是空间坐标

定义每个像素：

[
x_p=(R_p,G_p,B_p),
\qquad
a_p\in[0,1].
]

新的颜色函数为：

[
F_z:
[0,1]^4
\rightarrow
[0,1]^3,
]

[
y_p=F_z(R_p,G_p,B_p,a_p).
]

第四维不是 (x) 坐标或 (y) 坐标，而是 Query Token 对该像素的编辑置信度：

[
\boxed{
(R,G,B,A)\rightarrow(R',G',B')
}
]

这样，即使两个位置的 RGB 完全相同：

[
x_{p_1}=x_{p_2},
]

只要：

[
a_{p_1}\neq a_{p_2},
]

它们就可以得到不同输出：

[
F(x_{p_1},a_{p_1})
\neq
F(x_{p_2},a_{p_2}).
]

因此，4D GLUT 本身仍然不存储空间位置；空间信息由随位置变化的 attention map 提供。

这与已有 4D LUT 的 RGBC 思路在概念上相似：已有工作把 RGB 与 context map 拼接，通过 4D LUT 实现同色像素的内容相关变换。([arXiv][2])

需要特别注意：SA-LUT 已经使用 content-style cross-attention 生成 context map，再输入 4D LUT。因此，单纯提出“把 attention 作为 LUT 第四维”本身不足以构成完整创新。([arXiv][3])

你的区别必须建立在以下几点上：

1. instruction-conditioned VLM Query，而不是 reference-style feature；
2. 精确 mask 只用于监督，推理时只使用 Query attention；
3. 从现有 3D GLUT 进行结构化 3D→4D inflation；
4. 连续 Gaussian 4D 表示，而不是规则 4D 网格；
5. 只需要独立的 3D LUT 数据和 mask 数据，不要求成对的 local-retouch GT；
6. 专门评价 unseen region × unseen style 的组合泛化。

## 暂缓问题：XMP 空间算子与第四维语义（2026-07-17）

**状态：暂缓。首轮实验不声称解决 XMP 空间算子，待 renderer-only 对比完成后重新评估。**

当前首轮实验的 renderer 预训练核心数据固定为：

* 4000 个严格解析的原生 3D LUT；
* 581 个已经过照片 ΔE 验收的 baked LUT。

当前 r5 local 的 XMP 成片不作为严格 LUT GT。含未标定空间算子的 XMP 也不混入
首轮核心数据，避免把空间不可表示误差当成普通 LUT 标签噪声。

这里存在一个尚未解决的标定鸿沟：普通 HALD 中每个 RGB 只在一个固定空间位置出现，
因此只能观测到 RGB 与位置绑定后的结果，无法区分颜色映射与暗角等空间效应。增加同布局
HALD 的数量不能消除这个不可识别性；至少需要让相同 RGB 在多个位置/context 下重复出现。

可能的后续标定路线（尚未定案）：

1. 将 XMP 的空间键归零，先标定 color-only 3D transform `T_z(x)`；
2. 使用完整 XMP 渲染 repeated/permuted HALD、多位置重复色块或纯色画布；
3. 从 full XMP 与 color-only 结果之差拟合空间 residual；
4. 对暗角使用由 center/amount/feather/roundness 计算的径向场 `c_op(x,y)` 作为
   operator context，并约束 `F_z(x, 0) = T_z(x)`；
5. 在独立布局和自然照片上验证，不能只在用于提取的 HALD 布局上报告误差。

单标量 `RGB+C` 的可表达边界也需要保留：暗角和几何渐变可由径向场或 alpha 场表示；
Clarity、Texture、Sharpen、Dehaze、NR 等依赖邻域或整图统计，单个 context scalar
通常只能近似；Grain 还包含随机、分辨率和整图依赖，静态 4D LUT 不能通用复现。

更关键的是，本文当前把第四维定义为 local-edit attention `a_edit`，而 XMP 暗角需要的
是 operator context `c_op`。两者不是同一语义。候选方案包括：

* 4D LUT 使用 `c_op`，local attention 继续作为 renderer 外部的 endpoint blend；
* 同时把 `c_op` 与 `a_edit` 放入 renderer，形成实际的 5D 映射；
* 使用 3D/4D color renderer 加独立的小型 spatial residual network。

在恢复该问题前，不把 VLM attention、XMP operator field 和 local mask 合并成同一个
第四维，也不宣称 4D LUT 已覆盖 XMP 的全部空间算子。

---

# 三、不建议直接学习一个完全自由的 4D GLUT

最直接的实现是把 GLUT 的三维高斯扩展为四维高斯：

[
\tilde \mu_i\in\mathbb R^4,
\qquad
\tilde \Sigma_i\in\mathbb R^{4\times4}.
]

输入为：

[
u=[x,a]=[R,G,B,A].
]

高斯权重：

[
p_i(u)
======

\frac{
1
}{
\sqrt{(2\pi)^4|\tilde\Sigma_i|}
}
\exp
\left[
-\frac12
(u-\tilde\mu_i)^\top
\tilde\Sigma_i^{-1}
(u-\tilde\mu_i)
\right].
]

局部变换可以写为：

[
f_i(x,a)
========

M_ix+c_ia+b_i.
]

然后混合得到输出。

这个定义在数学上成立，但对于你的数据条件——只有 3D LUT 和 mask——会存在严重的欠约束问题。

3D LUT 只能给出：

[
F_z(x,1)=L_z(x),
]

mask 外区域只能给出：

[
F_z(x,0)=x.
]

但是中间的：

[
F_z(x,a),\qquad a\in(0,1)
]

并没有唯一答案。如果直接让一个高容量 4D Gaussian 函数自由学习，很容易出现：

* 中间 attention 值颜色不连续；
* LUT 色彩方向发生反转；
* attention 轻微变化引起大幅颜色跳变；
* 在 mask 边缘产生 halo；
* 破坏预训练 3D LUT 的全色域精度。

所以第一版不应直接从头学习任意：

[
(R,G,B,A)\rightarrow RGB.
]

---

# 四、推荐：Endpoint-Preserving Attention-Gated 4D GLUT

更稳妥的结构是把问题分解成：

1. 3D CGLUT 负责“应用什么调色”；
2. 4D Gaussian gate 负责“这个像素应该应用多少”。

设 CGLUT 输出的完整全局颜色变换为：

[
T_z(x).
]

定义 LUT residual：

[
\Delta_z(x)=T_z(x)-x.
]

最终的 local edit 为：

[
F_z(x,a)
========

x+
g_z(x,a)\Delta_z(x),
]

其中：

[
g_z(x,a)\in[0,1].
]

## 一个非常合适的约束参数化

令 4D Gaussian 网络预测：

[
r_z(x,a)\in\mathbb R.
]

然后定义：

[
g_z(x,a)
========

a+
a(1-a)\tanh(r_z(x,a)).
]

这个形式有三个关键性质。

### Attention 为 0 时严格保持原图

[
g_z(x,0)=0,
]

因此：

[
F_z(x,0)=x.
]

### Attention 为 1 时严格恢复原始 3D LUT

[
g_z(x,1)=1,
]

因此：

[
F_z(x,1)=T_z(x).
]

### 中间值始终合法

由于：

[
\tanh(r)\in[-1,1],
]

所以：

[
a^2
\leq
g_z(x,a)
\leq
2a-a^2,
]

从而：

[
0\leq g_z(x,a)\leq1.
]

这意味着 4D 模型不会随意改变 LUT 的作用方向，而是在原图与目标 LUT 之间进行受约束的、RGB-aware 的非线性调节。

更重要的是，当初始化：

[
r_z(x,a)=0
]

时：

[
g_z(x,a)=a.
]

模型一开始就等价于最稳定的 alpha blend：

[
F_z(x,a)
========

x+a(T_z(x)-x).
]

之后 4D Gaussian 只学习如何纠正 attention 的不准确性，而不是重新学习整个颜色映射。

---

# 五、4D Gaussian 在这里真正解决的是什么

一个非常强的基线其实是：

[
Y_p
===

I_p+
a_p
\left(
T_z(I_p)-I_p
\right).
]

也就是：

> Query attention + 普通 3D CGLUT + alpha blending。

所以必须回答：为什么还需要 4D GLUT？

答案不能只是“为了增加一维”，而应是：

> Query attention 是空间置信度，不是精确 alpha。4D Gaussian LUT 学习 (RGB) 与 attention 的联合校准，减少 attention 错误导致的颜色泄漏。

例如 Query 想编辑天空：

* 蓝色天空像素 attention 为 0.55；
* 白色云层 attention 为 0.45；
* 蓝色衣服被误激活为 0.50；
* 建筑边缘产生 0.20 的模糊响应。

单纯 alpha blend 会机械地对所有像素应用对应强度。

4D gate 学的是：

[
g(x,a)
\approx
P(M=1\mid RGB=x,\text{attention}=a).
]

它可以学习到：

* 天空蓝 + 中等 attention：提高 gate；
* 肤色或衣服颜色 + 偶然高 attention：抑制 gate；
* 边界低置信度：根据颜色连续性平滑处理；
* 高饱和目标颜色：避免过度变换。

因此：

[
\boxed{
\text{Attention 提供空间线索，RGB 提供颜色兼容性，4D Gaussian 联合决定最终编辑强度}
}
]

当然，它仍然不能解决“同色天空和同色衣服具有完全相同 attention”这种情况。真正的空间区分仍由 Query attention 承担。

---

# 六、如何从 3D GLUT 初始化为 4D GLUT

GLUT 本身用高斯的 RGB 均值和协方差描述颜色空间中的局部区域，CGLUT 再通过 style embedding 生成这些参数。

建议不要重新从随机 4D 高斯训练，而是直接从 3D CGLUT inflation。

对于已有 3D Gaussian：

[
\mu_i^{RGB}\in\mathbb R^3,
\qquad
\Sigma_i^{RGB}\in\mathbb R^{3\times3},
]

扩展为：

[
\tilde\mu_i=
\begin{bmatrix}
\mu_i^{RGB}\
\nu_i
\end{bmatrix},
]

[
\tilde\Sigma_i=
\begin{bmatrix}
\Sigma_i^{RGB} & 0\
0 & \sigma_{a,i}^2
\end{bmatrix}.
]

这里：

* (\nu_i) 是 attention 轴上的中心；
* (\sigma_{a,i}) 控制高斯对 attention 值的响应范围。

第一版建议使用 block-diagonal covariance，而不是完整 (4\times4) covariance。这样：

* 可以完整复用 3D GLUT 的 RGB 几何；
* attention 维度单独学习；
* 参数更少；
* 更容易稳定训练；
* 不会轻易破坏已有 LUT 表示。

可以为每个 RGB Gaussian 设置两个 attention anchor：

[
\nu_{i,0}=0,
\qquad
\nu_{i,1}=1.
]

分别对应：

* Off primitive：不执行颜色变换；
* On primitive：执行目标 CGLUT 变换。

或者设置三个 anchor：

[
\nu\in{0,0.5,1},
]

使模型专门学习：

* background；
* uncertain boundary；
* foreground。

后续可以把 RGB-attention cross covariance 作为增强版本或消融实验，而不是第一版默认结构。

---

# 七、只有 3D LUT 数据和 Mask 数据就可以训练

这是你的方案中很强的一点：

> LUT 和 mask 不需要天然配对。

给定：

* 任意自然图像 (I)；
* 任意局部区域 mask (M)；
* 任意 3D LUT (L_k)；

可以直接合成 local-retouch target：

[
Y_p
===

(1-M_p)I_p
+
M_pL_k(I_p).
]

也就是：

[
Y
=

I+
M\odot(L_k(I)-I).
]

因此可以构造组合数据：

[
\mathcal D_{\text{image-mask}}
\times
\mathcal D_{\text{LUT}}.
]

假设有：

* 1000 个 LUT；
* 100 万个带 mask 图像；

理论上可以产生大量不同的：

[
\text{region}\times\text{style}
]

组合，而不需要人工制作一一对应的 local-retouch 图像对。

这使得模型天然适合做 compositional OOD：

* 训练见过 sky；
* 训练见过 warm-film LUT；
* 但从未见过 warm-film × sky；
* 测试时要求只给 sky 加 warm-film。

---

# 八、Mask 数据还必须带有定位语义

这里有一个不可绕开的可识别性问题。

如果训练样本只有：

[
(I,M)
]

却没有说明“为什么选择这个 mask”或“用户想编辑什么区域”，那么同一张图中可能存在多个合法 mask，单个 Query Token 无法知道应该预测哪一个。

因此至少需要以下一种信息：

* semantic mask 类别，例如 sky、person、foreground、water；
* 文本区域描述，例如 “the sky in the upper half”；
* referring expression；
* object class；
* point 或 box prompt；
* 自动修图任务中明确的“需要修复区域”标签。

对于 instruction-guided local edit，可以生成模板：

```text
Apply warm cinematic color grading to the sky.
Make only the subject brighter.
Increase the saturation of the flowers.
Apply the LUT to the background, preserving the person.
```

Query Token 必须由这段 instruction 条件化。

如果没有区域描述，一个固定 Query Token 最多学到“显著区域”或“数据集中最常出现的 mask”，而不能实现用户可控的任意局部编辑。

对于 Auto-Retouch，普通语义分割 mask 也不够，因为它只告诉模型“哪里是天空”，没有告诉模型“天空是否需要被修改”。Auto-local retouch 还需要：

* 缺陷区域标注；
* 编辑前后差异 mask；
* 或一个判断区域是否需要调整的 aesthetic/local reward。

---

# 九、训练时不能只使用二值 Mask

训练 mask 通常是：

[
M_p\in{0,1},
]

但推理时 attention 一定是连续的：

[
a_p\in[0,1].
]

如果训练只看到 0 和 1，模型在：

[
a\in(0,1)
]

上的行为依然是 OOD。

因此需要主动构造 soft-mask supervision。

可以对精确 mask 做：

* Gaussian blur；
* distance transform；
* 随机 erosion/dilation；
* 不同温度的 sigmoid；
* 边缘噪声；
* patch-level downsample/upsample；
* 随机漏检和误激活。

得到：

[
\tilde M_p\in[0,1].
]

对应 target：

[
\tilde Y_p
==========

I_p+
\tilde M_p
\left(
L_k(I_p)-I_p
\right).
]

这样第四维 attention 轴的整个区间都得到监督。

特别应强化：

* (a\approx0)：背景泄漏；
* (a\approx0.5)：边界过渡；
* (a\approx1)：目标区域 LUT fidelity。

---

# 十、推荐分阶段训练

## 阶段一：预训练 3D CGLUT

只使用 LUT 数据，在完整 RGB cube 上训练：

[
z_k
\rightarrow
T_{z_k}(x)
\approx
L_k(x).
]

这一阶段确保：

* full-gamut color fidelity；
* style latent 插值；
* CGLUT 几何稳定；
* 无需自然图像和 mask。

建议优先使用 Shared Geometry 或 Hybrid Geometry，因为不同 LUT 的 Gaussian 区域保持对应，更利于 style interpolation。GLUT 的实验也表明 Shared Geometry 的 blending 更稳定，但拟合精度低于 Full Generation。

## 阶段二：训练单 Query Token 定位

冻结 CGLUT。

输入：

[
(I,t)
]

预测：

[
a=\operatorname{QueryAttn}(I,t).
]

使用精确 mask：

[
\mathcal L_{\text{mask}}
========================

\lambda_{\text{BCE}}\operatorname{BCE}(a,M)
+
\lambda_{\text{Dice}}(1-\operatorname{Dice}(a,M)).
]

还可以加入：

[
\mathcal L_{\text{boundary}},
\qquad
\mathcal L_{\text{equivariance}}.
]

其中 equivariance 要求对图像做 crop、flip、resize 后，attention map 做相同几何变换。

## 阶段三：使用 GT soft mask 训练 4D Gaussian gate

先使用：

[
a=\tilde M
]

而不是预测 attention，确保 4D Renderer 本身学对。

同时冻结或基本冻结 3D CGLUT，只训练：

* attention 轴 Gaussian 参数；
* gate coefficients；
* 少量 adaptor。

## 阶段四：Scheduled Mask Replacement

逐渐从 GT mask 过渡到预测 attention：

[
a_{\text{train}}
================

\beta\tilde M
+
(1-\beta)\hat a,
]

其中：

[
\beta:1\rightarrow0.
]

最终训练阶段：

[
a_{\text{train}}=\hat a.
]

精确 mask 只参与 loss，不再进入 forward renderer。

这一步非常重要，否则会出现：

* 训练时完美 mask；
* 推理时 noisy attention；
* Renderer 对 attention 错误毫无鲁棒性。

## 阶段五：小学习率联合微调

联合训练：

* VLM / Query adaptor；
* attention module；
* 4D Gaussian gate；
* CGLUT parameter generator。

但 CGLUT 使用较小学习率，避免 image-space local training 破坏 full-gamut LUT 表示。

---

# 十一、完整损失设计

可以使用：

[
\mathcal L
==========

\lambda_{\text{mask}}\mathcal L_{\text{mask}}
+
\lambda_{\text{in}}\mathcal L_{\text{inside}}
+
\lambda_{\text{out}}\mathcal L_{\text{outside}}
+
\lambda_{\text{end}}\mathcal L_{\text{endpoint}}
+
\lambda_{\text{soft}}\mathcal L_{\text{soft}}
+
\lambda_{\text{boundary}}\mathcal L_{\text{boundary}}
+
\lambda_{\text{eq}}\mathcal L_{\text{equiv}}.
]

## 区域内 LUT fidelity

[
\mathcal L_{\text{inside}}
==========================

\frac{
\sum_pM_p
\Delta E_{00}
\left(
\hat Y_p,L_k(I_p)
\right)
}{
\sum_pM_p+\epsilon
}.
]

## 区域外泄漏

[
\mathcal L_{\text{outside}}
===========================

\frac{
\sum_p(1-M_p)
|
\hat Y_p-I_p
|_1
}{
\sum_p(1-M_p)+\epsilon
}.
]

这个指标比整图 PSNR 更重要，因为 local edit 最核心的问题是：

> 目标区域是否编辑正确，非目标区域是否完全不受影响。

## 4D endpoint constraint

在随机 RGB 点上监督：

[
\mathcal L_{\text{endpoint}}
============================

\mathbb E_x
\left[
|F_z(x,0)-x|_1
+
|F_z(x,1)-L_z(x)|_1
\right].
]

## Attention 中间值监督

随机采样：

[
a\sim U(0,1),
]

构造：

[
Y_a=x+a(L_z(x)-x),
]

监督：

[
\mathcal L_{\text{soft}}
========================

\mathbb E_{x,a}
|
F_z(x,a)-Y_a
|_1.
]

这使模型即使在没有自然图像 mask 的情况下，也可以在完整 RGB×Attention 空间接受监督。

---

# 十二、这套结构如何实现 OOD

最有说服力的不是笼统说“支持 OOD”，而是明确拆分测试。

## 1. Unseen region × style composition

训练时故意去掉部分组合：

```text
sky × warm-film           不出现
person × faded-film       不出现
water × teal-orange       不出现
```

但分别保留：

* sky 与其他 LUT；
* warm-film 与其他区域。

测试模型是否能组合。

这种能力来自明确解耦：

[
\text{Where}=a(x,y)
]

和：

[
\text{What}=T_z.
]

## 2. Unseen semantic region

训练不包含某类区域，例如 snow 或 neon sign，测试 Query Token 的开放词汇定位能力。

这部分主要依赖预训练 VLM 的语义能力和 Query mask supervision，而不是 4D LUT 本身。

## 3. Unseen LUT style

使用：

[
z_\alpha=(1-\alpha)z_1+\alpha z_2
]

或 prototype mixture 生成训练集中不存在的中间风格。

CGLUT 负责 style OOD/interpolation，Query attention 负责 spatial OOD。

## 4. Noisy-attention OOD

人工加入：

* false positive；
* false negative；
* 边界模糊；
* attention temperature shift；
* patch-level aliasing。

比较普通 alpha blend 与 4D Gaussian gate 的鲁棒性。

这实际上最能证明第四维 Gaussian 建模是否必要。

---

# 十三、更丰富的插值变成两个独立控制轴

原始 CGLUT 主要支持 style embedding 插值：

[
z(\alpha)
=========

(1-\alpha)z_1+\alpha z_2.
]

引入 attention 第四维后，可以同时控制：

### 风格轴

[
\alpha:\quad
\text{Style A}\rightarrow\text{Style B}.
]

### 空间强度轴

[
s:\quad
a_p^{(s)}=s,a_p.
]

最终：

[
Y_p(\alpha,s)
=============

F_{z(\alpha)}
\left(
I_p,
s,a_p
\right).
]

因此可以实现：

* 只给天空应用 30% cinematic；
* 逐渐从 warm style 过渡到 teal-orange；
* 保持背景不变，只调整目标区域；
* 连续改变局部编辑范围和强度。

这比 VeraRetouch 的全局 L/GC/SC latent 插值多了一条明确的 spatial control axis。

---

# 十四、最关键的基线和消融
必须加入这个非常强的基线：
[
\boxed{
\text{Query Attention}
+
\text{3D CGLUT}
+
\text{Alpha Blending}
}
]
即：
[
Y=I+a\odot(T_z(I)-I).
]

如果 4D Gaussian LUT 不能在以下方面显著优于它：

* outside-mask leakage；
* boundary halo；
* noisy attention robustness；
* unseen region-style composition；
* color fidelity；
那么 reviewer 会合理地认为第四维 GLUT 没有必要。
建议实验至少包括：

| 方法                                                 | 空间控制            | 颜色执行              |
| -------------------------------------------------- | --------------- | ----------------- |
| 3D CGLUT + GT mask                                 | Oracle mask     | Alpha blend       |
| 3D CGLUT + predicted attention                     | Query attention | Alpha blend       |
| Grid 4D LUT                                        | Query attention | 规则网格              |
| Full 4D Gaussian LUT                               | Query attention | 自由 4D 映射          |
| Endpoint-preserving 4D GLUT                        | Query attention | 受约束 Gaussian gate |
| Endpoint-preserving 4D GLUT + soft-mask curriculum | Query attention | 完整方法              |
核心指标：
* Attention IoU / Dice；
* Inside-mask (\Delta E_{00})；
* Outside-mask leakage (\Delta E_{00})；
* Boundary-band error；
* P95 / maximum color error；
* Style fidelity；
* unseen region × style accuracy；
* FPS、参数量和显存。
---

[1]: https://arxiv.org/html/2604.27375v1 "VeraRetouch: A Lightweight Fully Differentiable Framework for Multi-Task Reasoning Photo Retouching"
[2]: https://arxiv.org/abs/2209.01749?utm_source=chatgpt.com "4D LUT: Learnable Context-Aware 4D Lookup Table for Image Enhancement"
[3]: https://arxiv.org/abs/2506.13465?utm_source=chatgpt.com "SA-LUT: Spatial Adaptive 4D Look-Up Table for Photorealistic Style Transfer"
