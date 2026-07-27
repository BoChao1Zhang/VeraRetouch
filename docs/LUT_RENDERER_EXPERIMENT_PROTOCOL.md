# VeraRetouch LUT Renderer 对比实验实施公约

> 状态：**CANONICAL / 唯一实施规范**  
> 版本：`v1.0`  
> 生效日期：`2026-07-19`  
> 适用范围：本仓库首轮 LUT-only renderer、gate 与冻结 VLM 对齐实验

## 0. 文档权威与变更规则

1. 本文是当前实验唯一具有规范效力的实施公约。
2. `docs/test.md` 只保留研究动机、假设推导与后续方向，不再作为实现参数来源。
3. 本文与聊天记录、临时笔记、旧配置或论文默认值冲突时，以本文为准。
4. 已确定的数据生产 pipeline 不属于本实验的修改范围。本实验只消费其产物。
5. 任何会改变数据边界、split、模型结构、loss、训练预算、主指标或统计结论的调整，
   必须先修改本文、提升版本号并记录原因，之后才能继续运行。
6. 不允许在看到 test 结果后静默改变配置、选择分支或重新定义主指标。

本文中的 `MUST`、`MUST NOT`、`SHOULD` 分别表示必须、禁止和推荐。

---

## 1. 实验目标

首轮实验只回答 renderer 与 attention gate 的问题，不训练新的空间 Query Token。

### 1.1 核心问题

在相同 LUT 数据、相同冻结 VLM hidden states、相同 full-image SFT 数据与相同
attention corruption 下，比较：

1. Shared Geometry 3D CGLUT + raw alpha blend；
2. Shared Geometry 3D CGLUT + learned 1D alpha calibrator；
3. Shared Geometry 3D CGLUT + endpoint-preserving 4D Gaussian gate；
4. VeraRetouch native `RetouchHead + ConditionalMLPDecoder` + external raw alpha。

### 1.2 两类结果必须分开

实验必须输出两个独立结果表，不能混成一个排名：

1. **Controlled Gate Ablation**
   - 方法 1、2、3 使用完全相同且冻结的 CGLUT checkpoint；
   - 只比较 raw alpha、1D calibration 与 RGB-attention 4D calibration；
   - 1D 与 4D correction branch 做 trainable-parameter matching，并单独报告 FLOPs。
2. **Native-Capacity Complete System**
   - 比较完成冻结 VLM SFT 后的端到端系统；
   - 允许各原生 renderer 使用其已确认的容量和低学习率联合微调；
   - 必须同时报告参数量、FLOPs、延迟与显存。

### 1.3 预注册结论

- 核心主张是：4D gate 是否比 raw alpha 和 trainable-parameter matched 1D calibrator
  更能抵抗
  noisy attention，并降低局部编辑泄漏和边界误差。
- “完整系统优于 VeraRetouch”是独立结论，不能由 controlled ablation 代替。
- CGLUT 优化器 A/B 是预注册敏感性实验，不允许根据 test 结果挑选更有利的分支。

---

## 2. 明确不在本轮范围内的内容

以下问题全部暂缓，不得混入首轮训练数据或论文主张：

1. XMP 暗角、Clarity、Texture、Sharpen、Dehaze、NR、Grain 等空间或邻域算子；
2. 从普通 HALD 中反演 XMP 空间 operator context；
3. 将 local attention 与 XMP operator context 合并为同一个第四维；
4. 新的 instruction-conditioned spatial Query Token 或 VLM mask predictor；
5. full 4D free-form Gaussian mapping、RGB-attention cross covariance；
6. CGLUT 扩容、Full Generation CGLUT 或非论文原始 generator；
7. 修改当前数据 build pipeline、重新设计 instruction 或 taxonomy；
8. 使用 XMP target、XMP instruction 或无法由 3D LUT 严格表示的成片作为 GT。

本轮允许使用 r5 的 `C_GT` mask bank，但不使用 r5/XMP 的颜色 target 或 instruction。

---

## 3. 已接受的实验假设

本实验在下列假设成立的前提下执行，不在首轮重新证明：

1. `4000 native LUT + 581 baked LUT` 对大类与小类风格具有足够覆盖。
2. baked LUT 是近似 LUT，允许包含通过既定验收后的标签噪声。
3. 每条 SFT instruction 由 VLM 独立生成，文本不同，并且只对应一个 LUT。
4. 数据 pipeline 与数据库中的 preset 信息已经确定，不再改造。
5. LUT 轴序修复已经完成；本实验只接受 canonical axis contract。

如果结果暴露明显的 style coverage 或 baked noise 问题，应记录为假设失败，不能在同一次
test 运行中临时替换数据。

---

## 4. 数据资产

### 4.1 核心 LUT 集

核心集合固定为下列经数据快照审计得到的资产；P0 必须重新生成 manifest 并逐项核对，
任何数量差异都必须阻断运行：

| 来源 | 数量 | 格点 |
|---|---:|---|
| native `.cube` | 4000 | 原生尺寸 |
| accepted baked `.cube` | 581 | 全部 `64^3` |
| 合计 | 4581 | 不统一重采样 |

广义 bank 中额外存在的 51 个 `.3dl` 不属于本轮核心 4581 LUT；如需加入，必须升级协议。

native `.cube` 的原生尺寸分布为：

| N | LUT 数量 |
|---:|---:|
| 16 | 18 |
| 17 | 76 |
| 21 | 2 |
| 25 | 55 |
| 32 | 2832 |
| 33 | 560 |
| 40 | 5 |
| 64 | 400 |
| 65 | 52 |

### 4.2 LUT axis contract

所有 loader、训练 target 和测试工具必须遵守：

- LUT tensor shape：`[N, N, N, 3]`；
- tensor 索引：`lut[b, g, r]`；
- 对应输入坐标：`x = [r, g, b]`；
- `.cube` 行序：red-fastest；
- 非单位 domain 使用文件中的 `DOMAIN_MIN/DOMAIN_MAX` 生成原生输入坐标；
- 输出为 RGB，范围按既定 parser 归一到 `[0,1]`。

实现前必须用 identity cube 与 axis probe 做 golden test。任何 R/B 交换都属于阻断错误。

### 4.3 LUT full-grid supervision

LUT 训练禁止随机采样 RGB 点。每次使用某个 LUT 时：

1. 枚举其原生 `N^3` lattice；
2. 所有格点都参与 loss；
3. 允许为显存而 chunk，但不得丢点；
4. 先在单个 LUT 内对 `N^3` 点取均值，再在 style 间取均值；
5. `64^3` LUT 不得因格点更多而获得更高的 style 权重；
6. 不得把不同 N 的 LUT padding 成伪格点，也不得统一重采样为 `32^3` 或 `64^3`。

一个 optimizer update 逻辑上包含 16 个 style。实现可以按 N 分组、逐 style 或逐 chunk
累计梯度，但最终 loss 必须等价于 16 个 per-style mean 的平均。

### 4.4 Mask bank

复用 r5 的完整 `C_GT` mask bank，训练 mask 类型保持自然比例：

| mask 类型 | 训练比例 |
|---|---:|
| gradient | 73.0% |
| circular gradient | 19.4% |
| semantic | 7.6% |

干净 soft mask 内部的连续梯度是 GT，不是 corruption，不得二值化。

### 4.5 SFT 数据规模

| split | 数量 | 用途 |
|---|---:|---|
| train | 150,000 | frozen-VLM adaptor/renderer SFT |
| validation | 10,000 | checkpoint 与超参数选择 |
| seen-style test | 10,000 | 已见 LUT、未见 source/instruction/mask |
| unseen-style test | 10,000 | 完全未参与 renderer/SFT 的 LUT |

---

## 5. Metadata 与可复现性

### 5.1 数据库访问

PostgreSQL 只允许在 dataset construction 阶段使用：

1. 构建时一次性 bulk join preset、taxonomy、source 与 split 信息；
2. 导出版本化、不可变 metadata snapshot；
3. 训练启动时每个 rank 将 snapshot 一次性加载为内存 dictionary；
4. `Dataset.__getitem__` 内禁止查询 PostgreSQL。

### 5.2 Snapshot 最低字段

每条训练或评测记录至少必须可关联：

- `sample_id`
- `lut_content_hash`
- `preset_id`
- LUT `path/fmt/grid_size/domain`
- `source_asset_id`
- `source_cluster`
- `split_version`
- `taxonomy_major/taxonomy_minor`
- versioned 24D probe descriptor 与 descriptor version
- `mask_id/mask_type`
- `instruction`
- target 与 source 路径

训练配置中必须记录 snapshot 文件 hash，不允许只记录数据库查询语句。

### 5.3 Instruction contract

- instruction pipeline 已固定，不再修改；
- 每条 instruction 必须唯一对应一个 `lut_content_hash`；
- instruction 文本不作为 LUT 唯一身份键；
- `content_hash` 是精确 LUT 身份，taxonomy 与语言描述只提供语义。

### 5.4 Random seed contract

正式 run 固定三个 base seeds：

```text
1701
1702
1703
```

所有阶段 seed 由以下 tuple 的 SHA256 确定性派生：

`(protocol_version, base_seed, stage, method, cglut_branch, purpose)`

其中 `purpose` 至少区分 initialization、style order、image order、mask corruption、
DataLoader worker 与 bootstrap。A/B 在同一 base seed 下必须共享 initialization、
style order、image order 与 corruption seed，只允许 optimizer 配置不同。

---

## 6. Split 与泄漏边界

### 6.1 双重 grouped split

LUT 与 source 必须独立 grouped split：

- LUT：按 `lut_content_hash` 分组，并在 `(native|baked, taxonomy_major)` strata 内
  用 SHA256 排序后按 largest-remainder 分配 80/10/10；
- source：按 `source_cluster` 分组，用独立 SHA256 salt 排序后按
  largest-remainder 分配 80/10/10；
- split 由版本化 manifest 固化，所有方法共享。

固定 salts：

```text
lut_split_salt    = lut-renderer-protocol-v1-lut
source_split_salt = lut-renderer-protocol-v1-source
```

若 4581 个 LUT 的 content hash 全部唯一，目标文件数为 train 3665、validation 458、
test 458；若存在相同 content hash，必须按 group 分配并在 manifest 中记录实际文件数和
group 数。任何 group 跨 split 都是 P0 阻断错误。

### 6.2 Strict unseen-style 定义

40-epoch renderer calibration 与 150k SFT 只能使用 train LUT。

validation LUT 与 unseen-test LUT 的以下信息禁止进入梯度：

- `L_GT` full-grid target；
- trainable preset embedding；
- Vera teacher prototype；
- renderer calibration image pair；
- SFT sample。

held-out LUT 的 `L_GT` 只用于 validation/test 指标。其 Vera teacher prototype 可以作为
oracle analysis 计算，但不得作为被测模型输入或训练 target。

### 6.3 各 split 语义

- validation 共 10k，固定拆成：
  - `val_seen` 5k：train LUT + source-validation；
  - `val_unseen` 5k：validation LUT + source-validation；
- source-validation 与 source-test 必须完全不相交；
- seen-style test：train LUT + held-out source；
- unseen-style test：held-out test LUT + held-out source；
- test manifest 在所有配置锁定后只读使用。

`val_seen` 用于 gate checkpoint selection；SFT checkpoint selection 同时使用
`val_seen` 与 `val_unseen`，但只使用 output-space objective，不需要 held-out style
representation target。

---

## 7. 图像、颜色与 target contract

### 7.1 图像预处理

数据资产生成与实验训练视图必须区分：

- 既有 target 资产按 pipeline 约定由短边 1024 的渲染输入生成；
- 本协议禁止为了训练而用短边 512 重新执行 Lightroom/local render；
- 训练视图从已经落盘的 source 与 C_GT 派生；
- 对 resize 后的 source 重新应用确定性的 `L_GT`，再按第 7.3 节在 sRGB 中合成 target；
- 既有 target image 只用于 provenance/QA，不作为训练 target 的二次压缩来源。

所有 image-space 方法使用相同的完整训练视图：

1. 保留宽高比；
2. 短边缩放到 512；
3. 长边上限 1024；
4. 不做 random/center crop；
5. 按宽高比分桶；
6. batch 内 padding；
7. padding 区域不参与任何 loss 或 metric。

RGB image 下采样固定使用 Lanczos；soft mask/C_GT resize 固定使用 bilinear；
padding 不改变原图有效区域。解码后统一转换为 RGB sRGB。

所有训练 target 与 metric 在 float sRGB `[0,1]` 中计算。不得在 loss 前进行 8-bit
量化。保存可视化时可以量化，但不能回读量化结果做指标。

### 7.2 模型输入域

metric、target 合成与 CGLUT 输入均使用 float sRGB `[0,1]`。

Vera official renderer 的模型输入按仓库 inference contract 转换：

\[
x_{\mathrm{Vera}}=2x_{\mathrm{sRGB}}-1.
\]

- Vera natural input、HALD lattice 与 reference before/after 都转换到 `[-1,1]`；
- Vera decoder 输出经 sigmoid/clip 解释为 sRGB `[0,1]`；
- loss 与 metric 始终比较 `[0,1]` 输出和 `[0,1]` GT；
- CGLUT 不执行该 `[-1,1]` 归一化。

### 7.3 Local GT 合成

给定 source image `I`、clean soft mask `m` 与 LUT `L`：

\[
Y = I + m\odot(L(I)-I).
\]

- 合成在 sRGB 域执行；
- `m=0` 的像素必须保持 source；
- `m=1` 的像素必须等于 LUT render；
- soft transition 是合法 GT。

### 7.4 Reference 与 target source

Vera reference pair 与 renderer target image 必须来自不同 `source_cluster`。同一 LUT 的
teacher references 也不得与该训练 step 的 target source 重合。

---

## 8. Attention corruption

### 8.1 训练混合

- 20%：clean mask 直接作为 attention；
- 80%：online-corrupted attention；
- corrupted 部分按 40% light、40% medium 分配；
- 每个 corrupted sample 只选择 shift、boundary、coarse-grid、FP/FN 中一个 operator，
  四类等概率；
- severe 与 combined 只用于 held-out test。

### 8.2 Corruption 强度

设有效图像短边为 `S`：

- shift：x/y translation 独立从 `[-sS,sS]` 均匀采样；
- boundary：以随机正负号执行 radius=`bS` 的 dilation/erosion；
- coarse grid：将 attention 的短轴降到表中 grid size，保持宽高比后 bilinear 上采样；
- FP/FN：以 `m>=0.5` 划分 foreground/background，各自按表中比例注入 false negative
  与 false positive，再用原 soft mask 的局部值保持连续边界。

corruption 只改变输入 attention `a`，clean soft target `m` 永远不变。manifest 必须记录
实际像素值、operator 与随机 seed。

| 等级 | shift | boundary | coarse grid | FP/FN |
|---|---:|---:|---:|---:|
| light | 1.5% | 1% | 32 | 5% |
| medium | 4% | 3% | 16 | 15% |
| severe | 8% | 6% | 8 | 30% |

`severe` bank 对四个单 operator 等量覆盖；`combined` 在同一 attention 上按固定顺序
shift -> boundary -> coarse-grid -> FP/FN 组合全部 severe 级 operator。

### 8.3 固定评测 bank

- validation corruption bank 预生成并版本化，分布为 20% clean / 80% light-or-medium；
- severe 与 combined test 各为每个样本生成 3 个固定 realization；
- 所有方法复用相同 attention tensor，不允许方法内部重新采样。

---

## 9. 方法定义

### 9.1 公共颜色 residual

对于 CGLUT 方法：

\[
T_z(x)=\text{CGLUT}(x,z),\qquad
\Delta_z(x)=T_z(x)-x.
\]

最终 local renderer 统一写为：

\[
F_z(x,a)=x+g(x,a)\Delta_z(x).
\]

### 9.2 M0：CGLUT + Raw Alpha

\[
g_{\mathrm{raw}}(a)=a.
\]

该方法无可训练 gate。

### 9.3 M1：CGLUT + 1D Calibrator

\[
r_{\mathrm{1D}}(a)=
\operatorname{MLP}_{1\rightarrow8\rightarrow8\rightarrow1}(a),
\]

中间激活为 SiLU，总参数 97。最终层权重和 bias 全零初始化：

\[
g_{\mathrm{1D}}(a)=
a+a(1-a)\tanh(r_{\mathrm{1D}}(a)).
\]

1D calibrator：

- style-independent；
- 不接收 RGB、style、图像或 mask 类别；
- 与 4D gate 使用相同训练数据、loss、steps、optimizer 和 checkpoint 规则；
- 97 vs 96 只表示 trainable-parameter matching，不表示函数族或 FLOPs 完全等价。

### 9.4 M2：Endpoint-Preserving 4D Gaussian Gate

RGB geometry 继承已训练 CGLUT 的 32 个 Gaussian，并在 gate 训练期间冻结。

attention 轴固定为：

\[
\nu\in\{0,0.5,1\},\qquad \sigma_a=0.25.
\]

要求：

- block-diagonal covariance；
- 无 RGB-attention cross covariance；
- attention centers 与 `sigma_a` 固定；
- 每个 RGB Gaussian × attention anchor 只有一个可训练 scalar coefficient；
- correction branch 共 `32×3=96` 个参数；
- coefficients 全零初始化；
- gate style-independent。

令 `q_i(x)` 为冻结 RGB Gaussian density，attention density 为：

\[
q_j(a)=
\exp\left(-\frac{(a-\nu_j)^2}{2\sigma_a^2}\right).
\]

归一化 4D 权重：

\[
w_{ij}(x,a)=
\frac{q_i(x)q_j(a)}
{\sum_{u,v}q_u(x)q_v(a)+10^{-6}}.
\]

\[
r_{\mathrm{4D}}(x,a)=
\sum_{i=1}^{32}\sum_{j=1}^{3}w_{ij}(x,a)c_{ij}.
\]

\[
g_{\mathrm{4D}}(x,a)=
a+a(1-a)\tanh(r_{\mathrm{4D}}(x,a)).
\]

该参数化必须严格保证：

\[
F_z(x,0)=x,\qquad F_z(x,1)=T_z(x).
\]

### 9.5 M3：VeraRetouch Native Renderer + Raw Alpha

使用官方预训练：

- Retouch Encoder；
- `RetouchHead`；
- `ConditionalMLPDecoder`。

native style renderer：

\[
R_z(x)=\operatorname{ConditionalMLPDecoder}(x,z).
\]

local 输出使用 external raw alpha：

\[
F_z^{\mathrm{Vera}}(x,a)
=x+a(R_z(x)-x).
\]

Vera 的三个 control aspect 全部启用，control mask 固定为 `[1,1,1]`。

本轮不把 Vera decoder 误称为空间 renderer。它仍是 global per-pixel ColorMLP；空间控制
来自外部 attention。

---

## 10. CGLUT 规范

### 10.1 架构来源

首轮使用 GLUT 论文的 **Large Shared Geometry CGLUT-32**。作者官方 runnable code
截至协议生效日尚未发布，因此实现必须以论文与 supplement 为准，并加入本协议的测试。

### 10.2 架构

- Gaussian 数量：`N=32`；
- 每个 train LUT 一个 64D style embedding；
- shared encoder：`64->128->128->128`，每个中间层后 ReLU；
- opacity head：`128->128->N`，中间 ReLU；
- local color head：`128->128->128->12N`，中间 ReLU；
- global affine head：`128->128->12`，中间 ReLU；
- Gaussian means 与 covariances 在 style 间共享并可训练；
- style generator 不生成 means/covariances；
- local transform 为每 Gaussian 的 `3×3 matrix + 3D bias`；
- 每 style 一个 `3×3 global matrix + 3D bias`。

数值参数化固定为：

- covariance 使用 lower-triangular Cholesky factor；
- Cholesky diagonal 使用 log/exp 参数化并设置最小值 `1e-4`；
- opacity 使用 sigmoid 映射到 `(0,1)`；
- 论文中的 opacity=1 用 `1-1e-4` 作为有限 logit 初始化；
- epsilon 固定为 `1e-6`。

严格 unseen 协议只为 train LUT 建立 embedding。按论文 Shared Geometry dagger 表
外推，shared generator+geometry 约 16.3 万参数；若 train LUT group 数约为 3665，
embedding table 约 23.5 万参数，完整 CGLUT 预计约 39.7 万参数。最终精确值必须由
实例化后的参数统计脚本写入报告；若为全部 4581 LUT 建表则约为 45.6 万，但该做法违反
strict unseen 协议。

### 10.3 初始化

以下是本协议已锁定的实验初始化，不是 GLUT 论文的 regular-grid initialization：

- 8 个 RGB cube corners；
- 24 个固定 Sobol centers；
- center seed 写入配置并在所有 A/B/seed 之间共享；
- covariance isotropic，`sigma=0.15`；
- opacity 的语义初始化为 1，数值上使用 `1-1e-4`；
- local/global affine 初始化为 identity matrix + zero bias。

### 10.4 CGLUT loss

逐格点误差定义为 `L1_p=||y_hat_p-y_p||_1`；其 dense 聚合由第 10.5 节唯一规定。

在 D65 CIELab 中：

\[
C=\sqrt{a^2+b^2},\qquad
\mathbf h=(a,b)/\max(C,10^{-6}),
\]

\[
\mathcal L_{\mathrm{hc}}=
\operatorname{mean}
\left[
C_{\mathrm{GT}}
\left(1-\langle \hat{\mathbf h},\mathbf h_{\mathrm{GT}}\rangle\right)
\right].
\]

Opacity binary entropy：

\[
\mathcal R_{\mathrm{opacity}}=
-\frac1N\sum_i
\left[
o_i\log(o_i+10^{-6})+
(1-o_i)\log(1-o_i+10^{-6})
\right].
\]

\[
\mathcal L_{\mathrm{CGLUT}}=
\mathcal L_{\mathrm{rec,dense}}
+10\mathcal L_{\mathrm{hc}}
+10^{-3}\mathcal R_{\mathrm{opacity}}
+10^{-4}\|e_z\|_2^2.
\]

`1e-4 ||e_z||^2` 是本协议新增的 embedding regularizer，不是 GLUT 论文原式。
Lab conversion 必须固定实现、白点与 sRGB transfer function，并由 golden values 测试。

### 10.5 Dense hard mining

所有 full-grid 点始终参与 base loss。设 `q(e)`：

- epoch `<5`：不启用额外 hard term；
- epoch 5：top 10%；
- epoch 5 到 20：线性增加到 top 40%；
- epoch `>20`：保持 top 40%。

epoch 按 1-based 编号。精确比例：

\[
q(e)=0.10+0.30\frac{e-5}{15},\qquad 5\le e\le20.
\]

每个 LUT 的 hard count 为 `k=max(1,ceil(q(e)*N^3))`。先将每个 RGB 点的三通道 L1
求和为 scalar，再按 error 降序、canonical `[b,g,r]` flat index 升序稳定排序解决平票。

\[
\mathcal L_{\mathrm{rec,dense}}=
\begin{cases}
\operatorname{mean}(L1_{\mathrm{all}}), & e<5,\\
\frac12\left[
\operatorname{mean}(L1_{\mathrm{all}})
+
\operatorname{mean}(L1_{\mathrm{top}\text{-}q(e)})
\right], & e\ge5.
\end{cases}
\]

top-q 必须在每个 LUT 内独立计算，不能让高动态 LUT 垄断跨 style hard samples。
每个 LUT 先得到一个 `L_rec,dense`，再与该 LUT 的 hue/chroma、opacity 和 embedding
regularizer 组成 `L_CGLUT`；一个 update 的最终 loss 是 16 个 per-style loss 的等权
平均。

### 10.6 优化器 A/B

两支必须完整训练并进入下游 SFT，不能用 test 选择。

| 分支 | generator/head LR | embedding LR | geometry LR | warmup | cosine floor |
|---|---:|---:|---:|---:|---:|
| A：paper-optimizer branch | `1e-3` | `1e-4` | `1e-4` | 0 | 0 |
| B：optimizer sensitivity | `1e-3` | `1e-3` | `3e-4` | 500 steps | `1e-5` |

公共设置：

- Adam；
- 40 coverage epochs；
- 16 styles / optimizer update；
- full native grids；
- 每个 coverage epoch 的 multiset 精确包含每个 train LUT 一次；
- 令 `n_rr=floor(0.8*N_train)`，`n_uniform=N_train-n_rr`；
- 用 Bresenham-style interleave 生成 `n_rr` 个 round-robin slots 与 `n_uniform` 个
  uniform slots，使任意 prefix 尽量接近 80/20；
- round-robin slot 按派生 seed 确定 major cycle，每个 major 内按 hash-shuffled
  queue 从尚未使用 LUT 取样；
- uniform slot 从全局 hash-shuffled 尚未使用队列取第一个 LUT；
- 某 major 队列耗尽时跳过，不允许为凑比例重复 LUT；
- batch 内 LUT 必须唯一，最后不足 16 的 batch 按实际 style 数归一化；
- 每个 epoch 记录实际 round-robin/uniform decision counts 与 per-major exposure；
- 相同 initialization、manifest、style order 与 random seeds；
- epoch 40 checkpoint 是 calibration checkpoint；
- 中间 checkpoint 只用于诊断，不使用 test 选择。

optimizer updates：

\[
S_{\mathrm{CGLUT}}
=40\left\lceil\frac{N_{\mathrm{LUT,train}}}{16}\right\rceil.
\]

严格 80% LUT split 下预计约 9.2k updates，最终以冻结 manifest 的实际 train LUT 数为准。

### 10.7 N ablation

主实验完成后必须运行 `N=16` 与 `N=64` ablation：

- 其余架构、数据与训练协议不变；
- 单独报告参数量和计算量；
- 由于 4D coefficients 分别变为 48/192，不再声称与 97 参数 1D calibrator
  容量匹配；
- N ablation 不替代主 `N=32` 结论。

---

## 11. Vera preset representation 与 renderer calibration

### 11.1 Teacher prototype

每个 train LUT 使用：

- 1 个 HALD before/after reference pair；
- 4 个来自不同 source cluster 的 natural before/after reference pairs。

官方 reference encoder 永久冻结。每对 reference 输出一个 raw 2688D latent：

\[
\bar z_k=
\frac15\sum_{j=1}^{5}z_{k,j}.
\]

### 11.2 Trainable preset latent

为每个 train LUT 建立 trainable `z_k∈R^2688`：

- 用 `bar_z_k` 初始化；
- full-grid calibration 时与 decoder 联合训练；
- latent consistency：

\[
\mathcal L_{\mathrm{latent}}
=\operatorname{SmoothL1}(z_k,\bar z_k).
\]

calibration 后的 `z_k` 冻结，并作为 frozen-VLM RetouchHead 的 representation target。

validation/test LUT 不得建立参与训练的 `z_k`。

### 11.3 Calibration loss

每个 style occurrence 使用：

- 该 LUT 的完整原生 lattice；
- 一张与 teacher references source cluster 不同的完整 natural image；
- control mask `[1,1,1]`。

P0 必须生成 `vera_calibration_pairs.jsonl`，显式列出
`(epoch, lut_content_hash, target_source_cluster, target_source_path)`。每个 epoch 内
同一 LUT 只出现一次；target source 由 train-source pool 按派生 seed 循环选择，禁止与
五个 teacher reference clusters 重合。A/B/CGLUT 不读取该 Vera-only image manifest。

\[
\mathcal L_{\mathrm{VeraCal}}
=
\mathcal L_{\mathrm{image-L1}}
+
\mathcal L_{\mathrm{grid-L1}}
+10\mathcal L_{\mathrm{hc}}
+0.1\mathcal L_{\mathrm{latent}}.
\]

其中 `L_hc=0.5*(L_hc,image+L_hc,grid)`。image 与 grid loss 都先 per-style mean，
再对 16 styles 平均。

### 11.4 Calibration 优化

- 完整 Vera/VLM checkpoint 固定为
  `Gyh68/VeraRetouch@0cccf3ef2ff16fd1f8b7867ab5cb3e725142696e`；
- reference Encoder/Renderer checkpoint 固定为
  `Gyh68/VeraRetouch.Encoder_Renderer@9b50d4c8241a3d811f4cccddf6024a12498c7a97`；
- P0 下载后必须记录每个实际 weight file 的 SHA256；
- native decoder 与 RetouchHead 从完整 Vera checkpoint 加载；
- frozen reference encoder 从 Encoder/Renderer checkpoint 加载；
- loader 必须断言 RetouchHead/decoder/encoder 关键 state_dict 无 missing/unexpected keys；
- 不额外重跑论文的 200k image-pair pretraining；
- reference encoder 冻结；
- trainable：preset latent 与 native decoder；
- 40 coverage epochs；
- global style batch 16；
- 每个 epoch 每个 train LUT 恰好一次，末批不得 drop 或重复，并按实际 style 数归一化；
- optimizer updates：
  \[
  S_{\mathrm{VeraCal}}
  =40\left\lceil\frac{N_{\mathrm{LUT,train}}}{16}\right\rceil;
  \]
- AdamW，`lr=1e-4`；
- cosine decay，无 warmup；
- epoch 40 checkpoint 进入 SFT。

repo 实现的 native decoder 配置必须为：

- `ConditionalMLPDecoder`；
- `latent_dim=896`，实际 condition width 为 `3×896=2688`；
- hidden dims `[128,256,512]`；
- `cond_method=add`；
- final activation `sigmoid`。

完整 checkpoint 与独立 Encoder/Renderer checkpoint 中的 decoder 参数若不一致，M3
必须使用完整 Vera checkpoint 的 native decoder；独立 checkpoint 只提供 frozen
reference encoder。差异必须记录，禁止静默覆盖。

论文的 200k/batch16/lr1e-4 只记录为官方 checkpoint provenance，不计入本实验适配
steps，也不能作为 Vera-only 额外训练。

---

## 12. Gate/Calibrator 预训练

### 12.1 共享条件

M0/M1/M2 使用：

- 同一个 CGLUT A 或 B calibration checkpoint；
- 完全相同的 full images、LUT、clean masks 与 corrupted attention；
- CGLUT 完全冻结；
- 相同 sample order、corruption seed 与 preprocessing；
- global image batch 16；
- full-image loss，不做任何像素子采样。

A 与 B 分支分别训练 gate，不共享训练后的 gate checkpoint。

gate 训练遍历冻结的 150k train manifest；每个 base seed 对 manifest 做确定性
epoch shuffle，30k steps 期间循环使用。VLM cache token failure 不影响 gate 训练，
因为 gate 阶段不读取 VLM hidden states。

### 12.2 Loss

clean GT 为 `m`，输入 attention 为 `a`：

\[
\mathcal L_{\mathrm{gate}}
=\operatorname{SmoothL1}(g(x,a),m).
\]

Charbonnier：

\[
\rho(d)=\sqrt{d^2+10^{-6}}.
\]

\[
\mathcal L_{\mathrm{render}}
=\operatorname{mean}\rho(\hat Y-Y).
\]

boundary weight：

\[
w_b=4m(1-m).
\]

对于 binary semantic mask，额外使用 5px morphology boundary band；最终 boundary
weight 是 soft weight 与 binary band 的并集。

\[
\mathcal L_{\mathrm{boundary}}
=
\frac{\sum w_b\rho(\hat Y-Y)}
{\sum w_b+10^{-6}}.
\]

\[
\mathcal L=
\mathcal L_{\mathrm{gate}}
+
\mathcal L_{\mathrm{render}}
+
0.5\mathcal L_{\mathrm{boundary}}.
\]

M0 raw alpha 无训练，仅在相同数据上直接评测。

### 12.3 优化

M1 与 M2：

- 30,000 steps；
- AdamW；
- `lr=1e-3`；
- `betas=(0.9,0.999)`；
- `weight_decay=0`；
- gradient clipping `1.0`；
- 500-step linear warmup；
- cosine decay 到 `1e-5`；
- BF16 forward/backward，endpoint unit test 使用 FP32。

### 12.4 Checkpoint

- 每 500 steps 在固定 `val_seen` manifest 与 validation corruption bank 上评估；
- validation 必须遍历完整 `val_seen`，按 image 先聚合，再按 mask type 和 taxonomy
  major macro-average；
- 候选范围为 step 5k 到 30k；
- 选择最低：

\[
\mathcal L_{\mathrm{val}}
=
\mathcal L_{\mathrm{gate}}
+
\mathcal L_{\mathrm{render}}
+
0.5\mathcal L_{\mathrm{boundary}};
\]

- 不允许为 1D/4D 使用不同的选择指标；
- objective 相同到 `1e-8` 时选择更早 checkpoint；
- 任一 NaN/Inf checkpoint 直接失效。

---

## 13. Frozen VLM hidden-state cache

### 13.1 生成方式

使用官方冻结 VLM 做真实 autoregressive generation：

- 禁止 teacher forcing；
- 读取最后层 `h_L/h_GC/h_SC`；
- 按官方逻辑记录三个 special token 的 generation position；
- cache dtype 为 BF16；
- 三个 896D hidden 拼接后为 2688D。

### 13.2 Cache key

cache key/manifest 必须绑定：

- `sample_id`
- VLM checkpoint SHA/hash
- tokenizer hash/version
- prompt template version
- generation config hash
- image preprocessing version
- special-token ids
- code commit

预计 150k train hidden cache 约 0.8 GB；validation/test 也必须使用同一协议单独缓存。

### 13.3 异常 generation

- valid generation 必须恰好包含一个 L、一个 GC、一个 SC critical token；
- 缺失或重复任一 critical token 都记为 generation failure；
- cache audit 先生成冻结的 `train_valid_hidden_manifest`；
- train：缺少任一 critical token 的样本不得进入 adaptor loss，并记录 failure；
- 50k SFT 只循环 `train_valid_hidden_manifest`，所有方法共享同一有效样本顺序；
- validation/test：不得删除失败样本；
- validation/test failure 使用 deterministic identity output，并单独报告 generation
  failure rate；
- 所有方法共享同一个 cache，因此 generation failure 不得按方法重新计算。

---

## 14. Frozen-VLM adaptor

### 14.1 公共输入

\[
h=[h_L;h_{GC};h_{SC}]\in\mathbb R^{2688}.
\]

四个系统都使用本仓库 Vera implementation/config 中的 `RetouchHead`：

\[
2688\rightarrow1344\rightarrow1344\rightarrow2688
\]

结构为 Linear、LayerNorm、GELU、Linear、LayerNorm、GELU、Linear，并从第 11.4 节
固定的完整 Vera checkpoint 初始化。`RetouchHead` 与 2688/1344 维度属于仓库实现事实，
不是 VeraRetouch 论文正文披露的结构。

### 14.2 CGLUT projection

M0/M1/M2 在 RetouchHead 后增加：

\[
\operatorname{Linear}(2688,64).
\]

- Xavier 初始化；
- 三个 CGLUT 方法共享相同初始化；
- 输出对齐冻结的 64D train-LUT embedding。

Vera 直接使用 2688D RetouchHead 输出对齐冻结的 `z_k`。

### 14.3 Representation loss

\[
\mathcal L_{\mathrm{repr}}
=
1-\cos(\hat z,z_k)
+0.1\operatorname{SmoothL1}(\hat z,z_k).
\]

目标 embedding/prototype 必须 stop-gradient。

### 14.4 Capacity accounting

按当前仓库结构的静态参数公式，预期值为：

| 模块 | 参数量 |
|---|---:|
| Vera RetouchHead | 9,042,432 |
| CGLUT `2688->64` projection | 172,096 |
| Vera ConditionalMLPDecoder | 2,577,795 |
| Shared CGLUT generator+geometry | 162,892 |
| train-LUT embedding table | `64*N_LUT,train` |
| 1D calibrator | 97 |
| 4D gate, N=32 | 96 |

若 `N_LUT,train=3665`：

- CGLUT train-time renderer 含 embedding table约 397,452 参数；
- CGLUT complete training system约 9,611,980 参数，再加对应 gate；
- unseen inference 不需要 preset lookup table，deployed CGLUT complete system约
  9,377,420 参数，再加对应 gate；
- Vera RetouchHead+decoder约 11,620,227 参数。

以上均为 protocol expectation。正式报告必须用实际实例化脚本统计，并同时区分
train-time total、trainable 与 deployed parameters；数值不符必须先解释，不能改口径。

---

## 15. 150k Full-Image SFT

### 15.1 公共预算

- global batch size：16；
- total steps：50,000；
- 约 5.3 data epochs；
- 前 10,000 steps：只训练 RetouchHead 与 CGLUT projection；
- 后 40,000 steps：低学习率解冻 renderer；
- AdamW；
- adaptor/projection LR：`5e-5`；
- renderer/gate LR：`5e-6`；
- adaptor 从 step 0 开始独立 warmup 1,000 steps，之后 cosine decay；
- renderer/gate 在 step 10k 解冻后独立 warmup 1,000 steps，之后在剩余 39k steps
  上 cosine decay；
- 所有方法共享 sample order、mask、attention 和 VLM cache。

SFT sampler 对 `train_valid_hidden_manifest` 做确定性 epoch shuffle，不做 test-aware
重采样。image/repr loss 保持 sample-weighted；只有 full-grid replay 对重复 LUT 去重。

### 15.2 后 40k 解冻范围

Vera：

- 解冻 `ConditionalMLPDecoder`；
- reference encoder 永久冻结；
- teacher prototype 与 calibrated preset `z_k` 永久冻结。

CGLUT：

- 解冻 parameter generator；
- 解冻 shared geometry；
- 解冻 global/local color heads；
- calibrated train-LUT embedding table 永久冻结。

Gate：

- M1 解冻 1D calibrator；
- M2 解冻 4D coefficients；
- M0 无 gate 参数。

### 15.3 Full-image forward

每个 step 使用完整图像与 corrupted attention，target 为第 7.3 节定义的 `Y`。

CGLUT 方法：

\[
\hat Y=x+g(x,a)(T_{\hat z}(x)-x).
\]

Vera：

\[
\hat Y=x+a(R_{\hat z}(x)-x).
\]

### 15.4 Full-grid replay

每第 10 个 SFT step，对当前 batch 的 train LUT 加入一次 full-grid auxiliary：

- 使用当前 sample 的 adaptor-predicted latent；
- 不使用 preset lookup 作为 renderer 输入；
- 对 batch 中重复的 `lut_content_hash` 去重；同一 LUT 的 replay latent 定义为该 batch
  内所有对应 sample predicted latents 的算术平均；
- 每个 LUT 的全部原生 `N^3` 点参与；
- 先对每个 LUT 的格点取均值，再对 unique LUT 等权平均；
- target 是该 LUT 的原生 `L_GT`；
- replay 使用第 15.5 节的 `L_grid-L1+10*L_hc`，不使用 CGLUT calibration 的
  hard-mining term；
- `a=1`；
- 与当步 full-image loss 联合反传；
- 仍只执行一次 optimizer update。

50k SFT 约产生 5k full-grid replay steps；按 nominal train split 约为每个 train LUT
额外 22 次 full-grid exposure。由于 image batch 可能包含重复 LUT，该数值只是 nominal
estimate；实际 per-LUT replay counts 必须写入日志和报告。

replay step 上 `lambda_grid=1`，不再除以 10；因此跨全部 SFT steps 的预期有效权重约为
0.1。这是预注册的稀疏 replay 权重，不得在运行后重标定。

### 15.5 SFT loss

公共：

\[
\mathcal L_{\mathrm{SFT}}
=
\mathcal L_{\mathrm{repr}}
+
\mathcal L_{\mathrm{render}}
+
0.5\mathcal L_{\mathrm{boundary}}
+
\mathbb 1_{\mathrm{gate}}\mathcal L_{\mathrm{gate}}
+
\mathbb 1_{t\bmod10=0}\mathcal L_{\mathrm{grid}}.
\]

其中：

\[
\mathcal L_{\mathrm{grid}}
=
\mathcal L_{\mathrm{grid-L1}}
+10\mathcal L_{\mathrm{hc}}.
\]

Vera/raw-alpha 没有 learned gate 时，`L_gate=0`。

- image/repr/gate/boundary loss 先以样本为单位归一化，再做 batch mean；
- full-grid replay 先以 unique LUT 为单位归一化，再做 style mean；
- batch 中 LUT 重复不改变 full-grid style 权重。

### 15.6 SFT checkpoint

- 每 1,000 steps 在固定 validation manifest 和 corruption bank 上评估；
- 候选范围 step 10k 到 50k；
- 不 early stop；
- checkpoint objective 不包含 `L_repr`，因为 `val_unseen` 没有合法 representation
  target；
- 定义：

\[
\mathcal L_{\mathrm{ckpt}}
=\frac12\mathcal L_{\mathrm{out}}(\mathrm{val\_seen})
+\frac12\mathcal L_{\mathrm{out}}(\mathrm{val\_unseen}),
\]

\[
\mathcal L_{\mathrm{out}}
=\mathcal L_{\mathrm{render}}
+0.5\mathcal L_{\mathrm{boundary}}
+\mathbb 1_{\mathrm{gate}}\mathcal L_{\mathrm{gate}}.
\]

- 两个 validation panel 均按 mask type 与 taxonomy major macro-average；
- 以同一 `L_ckpt` 最低值选 checkpoint；
- test set 不参与选择；
- CGLUT A/B 各自保留完整 checkpoint，不做 test-time branch selection；
- objective 相同到 `1e-8` 时选择更早 checkpoint；
- 任一 NaN/Inf checkpoint 直接失效。

---

## 16. 评测轨道

### 16.1 Renderer calibration

报告：

- train-LUT full-grid fidelity；
- held-out natural image fidelity；
- CGLUT A/B；
- Vera calibrated renderer；
- `N=16/32/64` ablation。

此轨道不用于宣称 unseen-style，因为 held-out LUT 没有 calibrated embedding。

### 16.2 Controlled Gate Ablation

使用：

- seen-style test；
- frozen CGLUT；
- calibrated GT style embedding；
- clean、light、medium、severe、combined attention；
- raw alpha、1D、4D。

该表回答 gate 本身是否必要。

### 16.3 Native-Capacity Complete System

使用 frozen VLM autoregressive cache 与 SFT checkpoint，分别报告：

- seen-style test；
- unseen-style test；
- clean oracle；
- severe；
- combined。

该表包含：

- CGLUT-A raw/1D/4D；
- CGLUT-B raw/1D/4D；
- Vera native + raw alpha。

A 是 paper-optimizer 主分支，B 是敏感性分支。报告必须同时展示，不能只展示较好者。
若结论方向不同，必须写为 optimizer-dependent。

---

## 17. 指标

### 17.1 主指标：Balanced Local Delta E 2000

\[
E_{\mathrm{bal}}
=\frac{
E_{\mathrm{inside}}+
E_{\mathrm{outside}}+
E_{\mathrm{boundary}}
}{3}.
\]

- inside：`m>=0.95`，与完整 LUT output `L(I)` 比较；
- outside：`m<=0.05`，与 source `I` 比较；
- boundary：与 composite GT `Y` 比较；
- soft mask boundary 使用 `4m(1-m)`；
- binary semantic 使用 5px morphology band。

空 region 不伪造 0；该样本从对应 component 排除并报告有效样本数。
region 是否有效只由 GT mask 决定，并对所有方法使用同一固定有效样本集合。

主分数：

- 在 severe + combined 上计算；
- 每个 severity 内先平均同一样本的 3 个固定 corruption realizations；
- severe 与 combined 再按 1:1 等权；
- 然后在每个 `(mask_type, taxonomy_major)` cell 内做 sample mean；
- gradient/circular/semantic 等权 macro-average；
- LUT major taxonomy 等权 macro-average；
- 不能按训练中的 73/19.4/7.6 比例做 headline metric。

generation failure 的 deterministic identity output 必须进入上述 headline 聚合；另外可以
报告仅成功 generation 的 conditional metric，但后者不能替代主结果。

### 17.2 VeraRetouch 对齐指标

完整、去除 padding 的图像必须报告：

- L1，越低越好；
- PSNR，越高越好；
- SSIM，越高越好；
- LPIPS，越低越好；
- DISTS，越低越好；
- GMSD，越低越好；
- TD，越低越好。

L1/PSNR 同时报告 whole image 与 inside/outside/boundary region。
SSIM/LPIPS/DISTS/GMSD/TD 只报告完整图像，不构造非标准 masked structural/perceptual
metric。

### 17.3 指标实现

- 颜色空间：float sRGB `[0,1]`；
- output 先 clamp 到 `[0,1]`；
- metric 前不做 8-bit quantization；
- PSNR：`data_range=1`；
- LPIPS：AlexNet `v0.1`，输入映射到 `[-1,1]`；
- padding 全部剔除；
- 每样本只有唯一 `L_GT`，不采用 FiveK 多专家取最大值规则；
- 上一条是本实验针对唯一 LUT GT 的明确改动，不是 VeraRetouch/FiveK 原指标协议；
- Delta E 使用固定 D65 sRGB-to-Lab 实现；
- metric library 与版本写入环境 lock。

### 17.4 Gate 与 endpoint diagnostics

另外报告：

- gate SmoothL1/MAE；
- Brier score；
- attention calibration curve；
- full-grid mean/P95/max RGB error；
- full-grid mean/P95 Delta E 2000；
- endpoint max error at `a=0` and `a=1`；
- generation failure rate。

---

## 18. 统计协议

### 18.1 重复

- 每个方法、CGLUT A/B 分支使用 3 个独立训练 seeds；
- severe 与 combined 各有每样本 3 个固定 corruption realizations；
- 所有 paired comparison 使用完全相同的样本与 corruption。

### 18.2 统计单位

统计单位为：

`(training_seed, lut_content_hash, source_cluster, mask_id, sample_id)`

禁止把像素当作独立样本计算显著性。

每个 sample 的 3 个 corruption realizations 先在 severity 内平均，severe/combined 再
等权平均，因此 realization 不作为独立统计样本。三个 training seeds 的 seed-level
mean 与 standard deviation 必须单独报告。

### 18.3 主比较

主比较：

1. 4D vs raw alpha；
2. 4D vs 1D calibrator。

使用 stratified paired multiway-cluster bootstrap：

- 先固定 `(mask_type, taxonomy_major)` strata；
- 外层重采样 training seed；
- 在每个 stratum 内对 LUT hash 与 source cluster 做 two-way cluster resampling；
- 在被选中的 LUT/source intersection 内重采样 `mask_id/sample_id`；
- 每次 replicate 重新计算 cell mean，再对 mask type 与 taxonomy major 等权 macro；
- 10,000 bootstrap replicates；
- 报告 absolute difference、relative improvement、95% CI；
- 两个主比较使用 Holm correction。

同一 replicate 中所有方法必须使用同一 cluster/sample 权重与同一 failure 状态。

PSNR/L1/SSIM/LPIPS 等为预注册 secondary metrics，报告 CI，但不能替换主指标结论。

---

## 19. 成功标准

### 19.1 “4D gate 有必要”成立

必须同时满足：

1. severe+combined 上，4D 的 `E_bal` 相对 raw alpha 至少降低 5%；
2. severe+combined 上，4D 的 `E_bal` 相对 1D 至少降低 5%；
3. 对
   \[
   R=(E_{\mathrm{baseline}}-E_{\mathrm{4D}})/E_{\mathrm{baseline}}
   \]
   计算 Holm-adjusted simultaneous 95% CI；两个比较的 lower bound 都必须至少为 5%；
4. clean/oracle 相对 raw alpha：
   - `E_bal` degradation 95% CI upper bound 不超过 1%；
   - PSNR drop 95% CI upper bound 不超过 0.1 dB；
   - LPIPS increase 95% CI upper bound 不超过 0.002；
5. FP32 endpoint unit test 的 max RGB error 不超过 `1e-6`。

若 endpoint test 失败，视为实现错误，不能进入性能比较。

### 19.2 “完整系统优于 VeraRetouch”成立

该主比较固定为 **CGLUT-A 4D complete system vs Vera native**，在 severe 与 combined
各 3 realization 先按第 17.1 节聚合。只有当：

- `E_bal` relative improvement point estimate 至少为 5%；
- relative improvement 的 95% CI lower bound 至少为 5%；

才允许使用 superiority 表述。否则必须报告 trade-off 或 statistically indistinguishable。

主结论以 CGLUT-A paper-optimizer branch 为预注册主分支。CGLUT-B 用于判断结论是否依赖优化器；
如果 A/B 方向不一致，必须显式报告敏感性，不能用 B 替换 A。

---

## 20. 效率评测

每个完整系统必须报告：

- trainable/total parameter count；
- renderer-only 与 complete-system 参数量；
- BF16/FP32 checkpoint size；
- FLOPs 或 MACs；
- peak training VRAM；
- peak inference VRAM；
- latency median/P95；
- throughput/FPS。

Controlled Gate Ablation 还必须单独报告 raw/1D/4D gate-only FLOPs 与 latency，避免把
“参数量接近”表述为“计算量完全匹配”。

统一 benchmark：

- 同一 GPU、driver、PyTorch 与 CUDA；
- batch size 1；
- `512×512` RGB image；
- BF16；
- 50 次 warmup；
- 200 次 timed runs；
- `torch.cuda.synchronize()` 包围计时；
- CGLUT parameter generation 与 renderer forward 都计入 complete latency；
- VLM latency与 cached-renderer latency分开报告；
- 如使用优化 kernel，PyTorch reference 与 optimized kernel 都必须给出一致性误差。

---

## 21. 必须产出的实验工件

每次正式 run 必须保存：

1. protocol version；
2. git commit 与 dirty diff hash；
3. environment lock；
4. LUT/source split manifests；
5. metadata snapshot 与 hash；
6. C_GT/mask manifest；
7. corruption manifests；
8. VLM cache manifest；
9. 完整 config；
10. random seeds；
11. upstream checkpoint repository revisions、weight-file SHA256 与训练 checkpoints；
12. validation selection log；
13. per-sample metrics；
14. aggregate metrics；
15. bootstrap replicates 或其可重算输入；
16. failure/NaN/OOM/invalid-generation ledger。

推荐目录：

```text
outputs/lut_renderer_protocol_v1/
  manifests/
  metadata/
  caches/
  configs/
  cglut_a/
  cglut_b/
  vera/
  gates/
  sft/
  metrics/
  reports/
```

禁止只保存 aggregate CSV 而丢失 per-sample 结果。

---

## 22. 执行顺序与阻断门

### P0：冻结输入

- 生成 LUT/source split；
- 校验 content hash；
- 核对 4000 native、581 baked、grid-size distribution 与文件存在性；
- 导出 metadata snapshot；
- 固化 150k train、5k val_seen、5k val_unseen、10k seen-test、10k unseen-test
  sample manifests；
- 生成 `vera_calibration_pairs.jsonl`；
- 生成 corruption manifests；
- 下载并校验第 11.4 节两个 Hugging Face revisions，记录实际 weight SHA256；
- 运行 axis/identity/domain golden tests；
- 写出 `protocol_lock.json`，包含所有实际 group/sample 数、manifest hashes、checkpoint
  hashes、三个 base seeds 与软件版本。

阻断条件：任何 LUT/source group 跨 split、validation-source 与 test-source 重合、
LUT 缺失、数量审计失败、checkpoint key/hash 异常或 axis test 失败。

### P1：CGLUT calibration

- 训练 A/B；
- 运行 full-grid fidelity；
- 保存 epoch 40 checkpoints；
- 运行 N=16/64 ablation。

阻断条件：NaN、opacity 越界、identity 初始化异常、A/B 未共享数据顺序。

### P2：Vera calibration

- 生成 1 HALD + 4 natural teacher latents；
- 训练 preset latents + decoder；
- 冻结 calibrated preset targets。

阻断条件：reference/target source cluster 重合、teacher 数量不足、control mask 非 `[1,1,1]`。

### P3：Gate training

- 对 CGLUT A/B 分别训练 1D/4D；
- raw alpha 直接建 baseline；
- 完成 endpoint tests；
- 按固定 validation objective 选 checkpoint。

阻断条件：1D 参数量不为 97、4D 主配置不为 96、endpoint error 超阈值。

### P4：VLM cache

- 真实 autoregressive generation；
- 缓存三个 token hidden；
- 固化 checkpoint/prompt/tokenizer/generation hashes；
- 固化 `train_valid_hidden_manifest`，所有 SFT 方法共享；
- 输出 failure ledger。

阻断条件：缓存 key 不完整、teacher forcing、不同方法使用不同 cache。

### P5：50k SFT

- A/B 与 Vera 使用同一 sample order；
- 10k adaptor-only；
- 40k joint tuning；
- 1:10 full-grid replay；
- 固定 validation checkpoint selection。

阻断条件：test 数据进入训练、preset target 未冻结、DB 出现在 hot path。

### P6：评测与报告

- controlled gate table；
- native complete-system table；
- seen/unseen；
- clean/severe/combined；
- efficiency；
- paired multiway-cluster bootstrap；
- success-threshold 判定。

---

## 23. 报告必须包含的表

1. 数据与 split 审计表；
2. CGLUT A/B full-grid fidelity；
3. N=16/32/64 capacity ablation；
4. Controlled Gate Ablation；
5. Complete System seen-style；
6. Complete System unseen-style；
7. clean/noisy/severe/combined robustness；
8. PSNR/L1/SSIM/LPIPS/DISTS/GMSD/TD；
9. endpoint 与 P95/max color error；
10. 参数量/FLOPs/VRAM/latency；
11. bootstrap effect size 与 CI；
12. failure ledger。

所有表必须同时给出 mean、有效样本数和 95% CI。仅给最佳 seed 不合规。

---

## 24. 已知风险

1. Shared Geometry 在 LUT 数增大时拟合能力可能弱于 Full Generation；
2. 581 baked LUT 含近似噪声；
3. semantic mask 仅占训练 7.6%，必须依靠 macro evaluation 暴露弱点；
4. VLM instruction 到完全 unseen LUT 的 exact mapping 可能不可识别；
5. official GLUT runnable code 尚未发布，实现需依赖论文规格；
6. 4D gate 只能利用 RGB 与 attention，不能区分 RGB/attention 完全相同的不同位置；
7. full-image + full-grid 训练计算量较大，chunking 必须保持数学等价；
8. CGLUT A/B 可能产生不同结论，必须如实报告 optimizer sensitivity。

风险出现时优先记录和解释，不得在 test 后改变协议掩盖。

---

## 25. 后续但非本轮事项

本轮报告完成后，才允许讨论：

- CGLUT generator 扩容；
- Full Generation CGLUT；
- 4D gate cross covariance；
- XMP repeated/permuted HALD 标定；
- operator context 与 edit attention 的 5D/双路径建模；
- instruction-conditioned spatial Query Token；
- predicted attention 的端到端训练；
- 更大的 semantic-mask 数据比例。

---

## 26. 参考

1. VeraRetouch: *A Lightweight Fully Differentiable Framework for Multi-Task
   Reasoning Photo Retouching*, arXiv:2604.27375v1.
2. GLUT: *3D Gaussian Lookup Table for Continuous Color Transformation*,
   arXiv:2605.19889v1.
3. 研究动机与 XMP 暂缓说明：`docs/test.md`。
4. LUT parser 与 canonical axis：`dataset_build/recipes.py`。
5. Vera native renderer：`model/colormlp_v2.py`。
6. Vera three-token readout：`llava/model/VeraRetouch.py`。
