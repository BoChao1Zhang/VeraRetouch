# RO-1 · NOTES —— 实施前核实记录 / 假设清单 / 待主 agent 决策

实验编号：EXPERIMENTS_v3 §2.1 **RO-1**（self-self 零训练读出）｜日期 2026-08-03｜分支 `lens-exp`

---

## 〇、读了哪些文档节（任务卡第 1 件事）

| 文档 | 读的节 | 取到的约束 |
|---|---|---|
| `docs/EXPERIMENTS_v3_2026-08-02.md` | §2.1 表的 RO-0 / RO-1 行、表下 RO-X1/X2/X3 说明、Changelog 末三条（2026-08-03 深夜 / 2026-08-04 RO 重排） | 判据 AUC≥0.80 晋级、<0.75 → RO-4；三算子逐个报；伪影三件套+guided filter 逐档消融；符号 sanity check；**禁逐图归一化**（归一化留给 RO-X2）；RO-1 是所有 VLM 读出臂的基准线 |
| `docs/IMPL_DOSSIER_2026-08-02.md` | §5.1 全节 + 附录 B 官方代码仓库表 | GEM/SCLIP/NACLIP/ClearCLIP/ProxyCLIP 的接入路径、FeatUp/LoftUp 收尾、`kornia.guided_blur` eps 归一化域 |
| `docs/DATA_ASSIGNMENT_2026-08-02.md` | §3.3 表（第 83 行 RO-1/2/3/4/9 行）+ §1.1/§1.2 split 纪律 | 统一评分集 = **D-CONSTRUCT L1(S-val) GT 掩膜 + D-SFT-L(S-val) SAM3 掩膜 + G1 三元组**；只读 T1 冻结旁表，禁 ad-hoc 切分 |
| `experiments/G1_s_identifiability_20260803/REPORT.md` | 全文 | AUC 口径（归一化 Mann-Whitney U）、16×16 网格 + 覆盖≥0.5 判正、luma 基线必报、D-31 措辞纪律 |
| `experiments/RO9_layer_verdict_20260804/REPORT.md` | §二、§五建议 4 | **AUC_target 定义**：混池 {AUC(s_a,M), AUC(s_b,¬M)}，利用 AUC(s,¬M)=1−AUC(s,M)，与指令无关 ⟺ 中位数塌到 0.5。建议升为 RO 系通用判据 |
| `tools/scache/README.md` | 全文 | arm 目录约定、meta 必含字段、`norm` 字段、**无逐图归一化**红线、`upsample.py` 的 guided filter 签名已核 |
| `tools/harness/` | `metrics.py` / `stats.py` / `README.md` | 见下方 §四「复用与不复用」 |

---

## 一、在线核实记录（任务卡第 2 件事）

原则：`IMPL_DOSSIER 附录 B` 之外的 URL / 签名 / 超参一律打开原始来源。以下每条都**打开了源码**
（clone 后本地读文件，不是搜索摘要），并记下 commit。

### 1.1 三个官方仓库（全部 clone 后逐行读）

| 仓库 | commit | 核到的关键实现 |
|---|---|---|
| `github.com/wangf3014/SCLIP` | `3608360267b6130c1ef18090d7289f17c771cb90` | `clip/model.py:283-313` `custom_attn(..., csa=False)`：`csa=True` 时 `softmax(q@qᵀ·scale) + softmax(k@kᵀ·scale)`。`VisionTransformer.forward:249-251` **仅末层**：`x = x + custom_attn(blk.attn, blk.ln_1(x), csa=csa)` 然后 `x = x + blk.mlp(blk.ln_2(x))`（**残差与 FFN 都保留**）。`clip_segmentor.py:63` `encode_image(img, return_all=True, csa=True)` |
| `github.com/sinahmr/NACLIP` | `0cac3a651f753b11315ea799cfbfabf79da1bd76` | `clip/model.py:87-92` `set_params(arch, attn_strategy, gaussian_std)`，断言 `arch∈{reduced,vanilla}`、`attn_strategy∈{naclip,nonly,kk,csa,vanilla}`。`custom_attn:178+` naclip 分支 = `bmm(k,kᵀ)·scale + omega` 再 softmax；`omega` 由 `gaussian_window(2H-1,2W-1,std)` + `get_attention_addition`（`F.conv2d` 的 same padding 卷出逐位置高斯邻域偏置，含 CLS 行/列补零）生成，按 `n_patches` 缓存。`naclip.py:21` 默认 `arch='reduced', attn_strategy='naclip', gaussian_std=5.`；README:54 主结果 `bash test_all.sh reduced naclip 5 on {gpu} {log}` |
| `github.com/mc-lan/ClearCLIP` | `ad68a404d55d48d27330b93554eb64a234ff717f` | vendored `open_clip/transformer.py:614-616` `model_type=='ClearCLIP'` → `softmax(bmm(q,qᵀ)·scale)`（**只有 qq，没有 kk**）；`:520-528` `ignore_residual=True` + `last_n_layers=1` → `output = custom_attn(...)`，**丢残差、丢 FFN**。`demo.py` 存在且默认 `model_type='ClearCLIP', ignore_residual=True`；`clearclip_segmentor.py:31` `create_model('ViT-B/16', pretrained='openai', precision='fp16')` |

**⇒ 与 DOSSIER §5.1 的一致性**：SCLIP 的 CSA 描述、NACLIP 的 `set_params` / kkᵀ+高斯邻域偏置 /
「40 行可独立搬」、ClearCLIP 自带 demo.py 与 vendored open_clip 需置 sys.path 最前 —— **全部属实**。

### 1.2 `gem_torch`（DOSSIER 称「最快路径，6 行出第一张图」）

下载 `gem_torch-1.0.1-py3-none-any.whl`（PyPI 官方源；国内镜像 403 需 `-i https://pypi.org/simple`）
并解包读源码。**三处与 DOSSIER 表述有出入，见 §三「与 DOSSIER 不符之处」。**

### 1.3 `open_clip` 版本 pin（DOSSIER：`open_clip_torch<=2.24`）

打开两个 tag 的 `src/open_clip/factory.py`：

- `v2.24.0`：`create_model(model_name, pretrained, precision, device, jit, force_quick_gelu, ...)`
- `v3.3.0`：`create_model(model_name, pretrained, **load_weights**, precision, device, jit, ...)`
  —— 第 3 个位置参数变成了 `load_weights: bool`

`gem/gem.py::create_gem_model` 用**位置参数**转调 `open_clip.create_model(model_name, pretrained,
precision, device, ...)`，在 ≥3.0 上 `precision`（字符串）会落到 `load_weights`（bool）。
**⇒ DOSSIER 的 `<=2.24` pin 到 2026-08-03 仍然成立且必要**（当前 PyPI 最新 3.3.0）。

### 1.4 `kornia.filters.guided_blur`

kornia **0.8.2**（本机已装）。签名 `guided_blur(guidance, input, kernel_size, eps,
border_type='reflect', subsample=1)`，eps 为归一化域量级 —— 与 `tools/scache/upsample.py`
的模块 docstring 里已核实的记录一致，本实验直接复用该模块，未重新实现。

### 1.5 FeatUp `maskclip` + `use_norm=False`

`raw.githubusercontent.com/mhamilton723/FeatUp/main/hubconf.py`：
```python
def maskclip(pretrained=True, use_norm=True):
    assert not use_norm, "MaskCLIP only supports unnormed model"
```
**⇒ DOSSIER 的 `torch.hub.load('mhamilton723/FeatUp', 'maskclip', use_norm=False)` 属实**
（`use_norm=False` 不是可选项而是**强制**）。本实验**未使用** FeatUp，理由见 §二 A4。

### 1.6 测试时寄存器（D-0 三件套第一件）

DOSSIER / PLAN 只写了「test-time registers 承接高范数激活」，没给出处，属必须核实项。
核到：**arXiv 2506.08010，*Vision Transformers Don't Need Trained Registers*，NeurIPS'25 Spotlight**，
官方实现 `github.com/nickjiang2378/test-time-registers` commit `860df43515c8d8e9e90952af25a46c26e4469570`。
逐文件读到实现要点并复刻：

- `shared/algorithms.py::find_register_neurons`：用末层 patch 范数 > 阈值定位 outlier token，
  统计各层 `mlp.gelu` 输出在这些位置的平均 |激活|，取 top-k 作为 register neuron；
- `shared/hook_fn.py::activate_on_registers`：追加 token 的这些神经元置为全序列的 `sign_max`，
  图像 patch 的这些神经元置 0（`normal_values='zero'`）；
- `clip/clip/transformer.py:805-822`：额外 token = **零嵌入**，加在位置编码之后、`ln_pre` 之前；
- `configs/openai_clip_base.yaml`（**正好是 ViT-B-16 / openai，与本实验同骨干**）：
  `register_norm_threshold: 30`、`top_k: 10`、`highest_layer: 5`、`detect_outliers_layer: -1`；
  notebook 默认 `num_registers=1`、`scale=1`、`normal_values='zero'`、`apply_sparsity_filter=True`。

本实验按这套官方超参实现（`tools/readout/ro1_selfself.py::find_register_neurons /
RegisterIntervention`），**唯一改写是把官方的 NLD 索引改成本 fork 的 LND 布局**。

### 1.7 权重

OpenAI CLIP ViT-B/16 从官方 URL 下载，`sha256 =
5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`，与 URL 内嵌的期望哈希
（`clip.py::_download` 的校验口径）一致。落位 `/home/bc/data/models/openai_clip/ViT-B-16.pt`。

---

## 二、方法学决策（能自己核实的已核实，属决策的见 §五）

### A. 为什么把三个算子实现在**一份**骨干上，而不是分别 clone 三个仓库跑

任务卡问的是「三家谁强」。分别跑三个仓库会**同时**换掉 open_clip 版本（ClearCLIP 用 vendored
open_clip，SCLIP/NACLIP 用 openai-CLIP fork）、精度（ClearCLIP fp16 / NACLIP fp32）、
输入短边（336 vs 448）、mmseg 版本 —— 三个数字不可比，得不出「谁强」的结论。

做法：把 NACLIP 的 `clip/`（本身就是 OpenAI CLIP 的 fork，且**已内建与 SCLIP 逐字相同的 `csa` 分支**）
vendored 进 `tools/readout/clip_naclip/`，只加 **一个** `attn_strategy='clearclip'` 分支
（逐字抄自 ClearCLIP `transformer.py:614-616`），于是：

| 臂 | arch | attn_strategy | 等价于 |
|---|---|---|---|
| SCLIP | `vanilla`（末层保残差+FFN） | `csa` | SCLIP 官方 `forward:249-251` |
| NACLIP | `reduced`（末层丢残差丢 FFN） | `naclip`, std=5 | NACLIP 官方默认 |
| ClearCLIP | `reduced` | `clearclip` | ClearCLIP 官方 `ignore_residual=True, last_n_layers=1` |
| vanilla（对照） | `vanilla` | `vanilla` | 未改造 CLIP |

改动清单与逐行出处写在 `tools/readout/clip_naclip/VENDOR.md`，可 `diff` 回原仓库核对。

### B. 读出定义与「禁逐图归一化」的落实

`s = cos(patch_feat, text_emb)`，其中 `patch_feat` 是三个官方 segmentor 用的同一个量
（`encode_image(return_all=True)` → 丢 CLS → L2 归一），`text_emb` 是 **80 条
`openai_imagenet_template` 集成**（NACLIP `naclip.py:30-42` 的逐字口径：逐模板 encode →
L2 归一 → 求均值 → 再 L2 归一）。

- **没有 min-max、没有 softmax、没有分位数、没有逐图 z-score**。
- AUC 对逐图单调变换不变，所以本臂的判据数字与归一化档无关；归一化只影响跨图刻度，
  是 **RO-X2** 的题目。
- scache 落盘用**全数据集固定仿射** `z=(cos−μ)/σ`，μ/σ 是**整臂两个常量**（写进每条 meta，
  `per_image: false`，并给出反算公式）。这不是逐图归一化。

### C. 后处理阶梯（每档一行，预注册在 `run_ro1.py::RUNGS`）

| 档 | 内容 | 出处 |
|---|---|---|
| `base` | 无后处理 | — |
| `+A1_outlier` | 高范数 outlier 剔除 + 4 邻域插值（`norm > median + 3·MAD`） | PLAN §2.2 D-0 第二件 |
| `+A1+A2_notch` | 再加 patch 网格周期性陷波 | D-0 第三件 |
| `+A3_ttreg` | 测试时寄存器（前向期干预，1 个寄存器） | D-0 第一件（arXiv 2506.08010） |
| `+A3+A1` / `+A3+A1+A2` | 三件套叠加 | — |
| 全分辨率块：`bilinear` vs `guided` | D-3 收尾（`kornia.guided_blur`，eps=1e-4，subsample=4） | PLAN §2.2 D-3 |

**A1 的 outlier 判据取「末层之前」的 hidden 范数**：三个算子只改最后一个 block，所以
末层之前的 hidden 完全相同 —— 高范数 outlier 是「骨干×图像」的属性，不是 attention 手术的产物，
这样三条阶梯的 A1 掩膜完全一致，Δ 才可比。

**A2 的口径**：s 场已在 patch 网格上，原图的 16px 周期 = patch 网格上的 **1 patch 周期 = Nyquist**，
所以诊断量 = 棋盘格分量功率 / 同环带中位功率（`nyquist_ratio`），阈值 2.0 记为阳性。
D-0 原文的「阳性 → DVT」不做：DVT 需要逐图优化，不属于「零训练一次前向」的 RO-1 定义域，
若诊断阳性会在 REPORT 里明确记为**未做的补救**。

### A4. 为什么没做 FeatUp / LoftUp

D-3 的原文顺序是「32×32→4K 先 guided filter（零参）；不够再 JAFAR/LoftUp；最后 FeatUp」。
本实验的 s 场原生分辨率已是 **28×42 起**（448 短边 / patch16），不是 32×32 的粗场；
先跑零参的 guided filter 看回收多少，够就不上学习式上采样器。若 REPORT 里 guided 档
相对 bilinear 的增量为负或可忽略，才有理由再上 FeatUp —— 那是 RO-1 的后续小实验，不是本轮判据。
（FeatUp 的调用签名已核实，见 §1.5，随时可接。）

### D. GEM 没有作为第四个算子入榜

理由是**环境风险**，不是技术障碍：`gem_torch` 依赖 `open_clip_torch`（且必须 pin ≤2.24），
而 open_clip 2.24 的依赖里带 `timm`，本机 conda 环境现装 `timm 0.4.12`，
**卡 0 / 卡 1 上有其他 agent 的作业**，升 timm 有可能打断别人。GEM 的机制
（最后 `depth-1` 层全部换 SelfSelfAttention）与本实验的三算子（**只改末层**）不同档，
不放进同一张表反而更干净。核实结果仍如实记在 §三。

---

## 三、在线核实中发现的**与 DOSSIER 不符 / 需修订**之处（重要）

> 以下三条都是打开 `gem_torch-1.0.1` 源码后发现的，`open_clip` 那条是打开两个 tag 的 factory.py 发现的。

1. **`gem.create_model_and_transforms` 在 1.0.1 里是坏的**（位置参数错位）。
   `gem/gem.py` 里 `create_gem_model` 的签名是
   `(model_name, pretrained, gem_depth, ss_attn_iter, ss_attn_temp, precision, device, ...)`，
   而 `create_model_and_transforms` 用
   `create_gem_model(model_name, pretrained, gem_depth, precision, device, jit, ...)`
   **按位置**转调 —— `precision` 落到 `ss_attn_iter`、`device` 落到 `ss_attn_temp`、
   `jit` 落到 `precision`。**DOSSIER §5.1 的「README snippet 6 行」若走 `create_model_and_transforms`
   会踩这个雷**；必须直接用 `create_gem_model(...)` + `get_gem_img_transform(...)`。
2. **`gem_depth=7` 不是「最后 7 层」而是「最后 6 层」**。
   `gem_wrapper.py::apply_gem` 是 `for i in range(1, self.depth)`，即 `depth-1` 个 block。
   DOSSIER §5.1 「gem_depth=7（最后 7 层换 SelfSelfAttention）」应改为「最后 `gem_depth−1` 层」。
3. **GEM 的 min-max 是可关的**（DOSSIER 只说「输出是 min-max 热图非 softmax」，读起来像不可关）。
   `GEMWrapper.forward(image, text, normalize=True, ...)`，传 `normalize=False` 即得未归一化的
   余弦图。**这条对本项目是红线级信息**：按 DOSSIER 字面理解直接用 GEM 输出 = 逐图 min-max = 踩红线；
   实际只要 `normalize=False`。建议 DOSSIER §5.1 补一句。
4. **`open_clip` 的 pin 现在更紧了**：`v3.0.0` 起 `create_model` 第 3 个位置参数插入了
   `load_weights: bool`。DOSSIER 写的「新版 create_model 签名变动会错位」属实，
   建议把 pin 从「≤2.24」写成硬性 `open_clip_torch==2.24.0`（2.26–2.32 未逐个核，只核了 2.24 与 3.3）。
5. **`clip.py` 依赖 `pkg_resources`**（`from pkg_resources import packaging`）。三个仓库的
   openai-CLIP fork 都有这一行；Python 3.13 + 新 setuptools 下 `pkg_resources` 已被标记弃用，
   本机仍可 import（未触发失败）。若将来报错，改成 `from packaging import version` 即可。
   记录在此供后续臂（RO-4 ProxyCLIP 同样是这套 fork）参考。

**没有发现编造**：附录 B 里 SCLIP / NACLIP / ClearCLIP / GEM / FeatUp / test-time-registers
六个仓库地址全部真实可 clone，commit 已记录在案。

---

## 四、复用与不复用

- **复用**：`tools/scache/api.py`（arm 写入）、`tools/scache/upsample.py`（guided filter，
  签名已在该模块核实过 kornia 0.8.2 源码）、G1 `analyze_g1.py::roc_auc` 的 AUC 口径
  （归一化 Mann-Whitney U，并列取平均秩）与 16×16 网格 + 覆盖≥0.5 判正的标签口径。
- **不复用 `tools/harness/metrics.py`**：那套是**渲染臂**的图像质量度量（PSNR/ΔE00/边界带/
  烘焙一致性），RO-1 产出的是 s 场不是重建图，接不上。`tools/harness/stats.py` 的 bootstrap
  思路照搬（本实验自带 `boot_ci`，2000 次，seed 固定），没有重造 harness。

---

## 五、待主 agent 决策（采用保守默认继续，未静默拍板）

| # | 决策点 | 两种做法 | **本轮采用的保守默认** |
|---|---|---|---|
| **U-RO1-1** | **gate 判在哪个评分集上** | EXPERIMENTS_v3 RO-1 行写的数据是「L1 图 + GT 掩膜」（S-val 只有 **23** 个可用样本，功效极低）；DATA_ASSIGNMENT §3.3 L83 写的是「统一评分集」（L1 + SAM3 主体掩膜 + G1 三元组，n=214+23） | **两个评分集都报，gate 规则预注册为**：两集均 ≥0.80 → 晋级；两集均 <0.75 → 转 RO-4；**否则判「分裂」交主 agent 裁决**。不擅自只取一个集下结论 |
| **U-RO1-2** | **目标短语用长描述还是短名词** | l 系 journal 同时有 `subject.description`（长句）与 `subject.name`（单词）。CLIP 的 80 模板是为**类名**设计的 | 预注册主口径 = **长描述**（与 SFT 指令实际措辞一致，是 VLM 臂真正会拿到的文本）；短名词作为并列列全程同报。若两者差距大，主 agent 需拍板 RO 系统一口径 |
| **U-RO1-3** | **输入短边尺度** | SCLIP/NACLIP 官方 336、ClearCLIP 官方 448；混用则三家不可比 | 主协议统一 **448**（三家都支持，且是分辨率更高的一档），**336 全量复跑一遍作稳健性行**。若两个尺度上排名翻转，需主 agent 决定 RO 系统一尺度 |
| **U-RO1-4** | **AUC_target 的 `reg_b` 文本** | G1 的 `reg_b` 是「the background」/「the right side of the image」这类**空间词**。CLIP 对「the right side of the image」几乎无定位能力，spatial 子集（n=49）的 AUC_target 会被 prompt 质量而非方法本身拖低 | 主口径限 **`region_b_kind='background'`（n=165）**，spatial 单列并注明「该子集的低分不可归因于读出方法」。同时用**跨图错位（shuffle）**作为独立于 prompt 质量的指令条件性证据 |
| **U-RO1-5** | **A2 诊断阳性后是否上 DVT** | D-0 原文写「16px 周期功率谱检查 → DVT」。DVT 需逐图优化，与「零训练一次前向」的 RO-1 定义冲突 | 本轮**只做诊断 + 零成本陷波**，不上 DVT；若诊断强阳性且陷波有正收益，在 REPORT 的「建议下一步」里提出，不擅自扩范围 |
| **U-RO1-6** | **GEM 是否入榜** | GEM 需要 `open_clip_torch<=2.24`，会连带升级共享 conda 环境里的 `timm`（现 0.4.12），**卡 0/卡 1 上有其他 agent 作业** | **本轮不入榜**，只登记核实结果（§三）。若主 agent 认为 GEM 必须入榜，建议开独立 venv 单跑，不动共享环境 |

---

## 六、红线自查

| 红线 | 本实验的落实 |
|---|---|
| **s 禁逐图 min-max/softmax/分位数归一化** | 读出唯一口径 = 原始余弦；scache 用整臂两个常量的固定仿射（meta `per_image: false`）。全分辨率块里 guided filter 前的 `[0,1]` 缩放是**逐图仿射**，只为满足 kornia 的 eps 归一化域语义，**是单调变换、不改 AUC**，且该数值不写进 scache —— 已在代码注释与 REPORT 里标注 |
| **每个消融行必带 Δ_const / Δ_shuffle 列** | 每个 (算子 × 档) 都报 `delta_const`（相对常数场 0.5）、`delta_shuffle`（相对跨图错位短语，含配对 Wilcoxon）、`delta_fixedprompt`（相对全体共用的固定短语）、`delta_luma`（相对亮度基线） |
| **符号 sanity check，禁事后翻转** | 符号约定 `s = +cos(...)` 写死在代码里，判据「逐样本 AUC>0.5 的比例 ≥0.90 且中位 >0.5」预注册；**代码里没有任何 flip 分支** |
| **IoU 禁当优化目标** | 本臂零训练，无优化目标；IoU 未出现 |
| **禁 ad-hoc 切分** | 评分集两个来源都是既有冻结产物：T4 的 `sanity/val`（split 由 T1 旁表逐条背书）与 G1 的 `g1_region_opp.json`（S-val 214 源，**复用不重采**）。register neuron 的发现集用 **S-train** 图，不碰 val |
| **表述纪律 D-31** | REPORT 的每条结论都标了「未触发死刑」还是「获得正面证据」 |
| **不 kill 别人的进程 / 卡 1 / ≤20GB** | 全程只用 `CUDA_VISIBLE_DEVICES=1`；显存峰值见 `config/env.json`；长任务 nohup + `job.marker` |

---

## 七、跑完后补记（2026-08-03 15:45）

### 7.1 实现自检（跑数前做的，留证）

- **AUC 实现**：`run_ro1.fast_auc` 对 `sklearn.metrics.roc_auc_score` 校验，
  200 个随机用例（刻意 round 到 2 位制造大量并列）最大差 **1.1e-16**。
- **guided filter 前的 [0,1] 缩放不改 AUC**：同一张图，用原尺度余弦 vs 用 [0,1] 缩放后的
  余弦分别过 `guided_blur`，两路输出 Pearson **0.99999999926**，AUC 差 **1.7e-7**。
  数学上也成立：guided filter 对输入 `p` 是线性算子（`a`、`b` 都是 `p` 的线性泛函），
  eps 只作用在 guidance 域。故 §六红线表里那条注记是可核的，不是口头保证。
- **vendored fork 的改动可核**：`config/vendored_model_py.diff`（104 行）是
  `tools/readout/clip_naclip/model.py` 相对 NACLIP@`0cac3a6` 的**全部**差异；
  `clip.py` / `simple_tokenizer.py` / `__init__.py` 经 `cmp` 与上游**逐字节相同**。

### 7.2 D-0 三件套的实测诊断（决定了哪几件该做）

- **A1（高范数 outlier）：病存在**。末层前 hidden 范数中位约 14、最大约 96，
  `median+3·MAD` 判出 **8.0–8.3% 的 patch** 是 outlier。A1 在三算子上一致 **+0.009~+0.012**。
- **A3（测试时寄存器）：病存在，药只对一个算子有效**。register neuron 检测在
  **100/100 张 S-train 图**上都命中超阈 token（阈值 30，官方 `configs/openai_clip_base.yaml`），
  发现 `{L5:[924,2256,1541,112,2562], L4:[1606,447,722], L3:[803,2884]}`；
  **用 20 张与 100 张图检出结果完全相同**（说明 top-10 很稳，不是采样噪声）。
  效果：ClearCLIP **+0.015**、NACLIP +0.005、SCLIP **0.000/负**。
- **A2（16px 周期性）：病不存在**。预注册阳性线 `nyquist_ratio > 2.0`，实测 0.52–0.75。
  按 D-0 自身的失败判据「此骨干无此病，跳过」，**A2 判为不适用**，DVT **未做**
  （对应 §五 U-RO1-5 的保守默认，未擅自扩范围）。

### 7.3 §五待决策项的实测材料（供主 agent 拍板时看数）

| 决策项 | 实测材料 |
|---|---|
| **U-RO1-1**（gate 判在哪个集） | 两集都过（0.939 / 0.930），**未触发「分裂」分支**，本轮无需裁决；但 L1 集 n=23、CI [0.832,0.966] 很宽，后续若靠 L1 单独下结论要先扩样本 |
| **U-RO1-2**（长描述 vs 短名词） | 短名词**一致更好**：SAM3 集 naclip **+0.050**、clearclip **+0.031**、sclip +0.022。用短名词时 NACLIP 0.962 ≈ ClearCLIP 0.961（**排名从"ClearCLIP 领先"变成打平**）。**这条会影响 RO 系的排名结论，需要拍板** |
| **U-RO1-3**（输入尺度） | 336 vs 448 的 Δ 为 −0.009~+0.009，**排名不翻转**。保守默认（统一 448）无风险，可追认 |
| **U-RO1-4**（AUC_target 的 comp 文本） | `AUC(s_comp, M)` 中位 0.48–0.69（理想应远 <0.5）→ CLIP 对「the background」「the right side of the image」定位弱，确认是 prompt 可定位性问题。**保守默认（主口径限 background 子集 + 用 shuffle 作主证据）成立** |
| **U-RO1-5**（A2 阳性后是否上 DVT） | 诊断**阴性**（0.52–0.75 << 2.0），本项自动落空，无需拍板 |
| **U-RO1-6**（GEM 是否入榜） | 仍待拍板。补充材料：三算子彼此只差 0.02，而 GEM 改的是最后 `gem_depth−1` 层（不同档），入榜价值主要在「多层 self-self 是否比单层更好」这个独立问题上 |

### 7.4 一条没做但审阅可能会问的

**没有跑 SCLIP/NACLIP/ClearCLIP 官方 benchmark（VOC/COCO）复现验收**：
本实验是把三家的 attention 手术搬到同一骨干上比 AUC，没有在 mmseg + VOC21 上跑出
各自论文的 mIoU 来证明"搬对了"。替代证据是：① 三段 attention 代码逐字对照官方源码
（§1.1，commit 已记）；② vendored fork 的 diff 只有 3 处、其余逐字节相同（§7.1）；
③ 未改造 CLIP 的 AUC 0.214（与三篇论文共同描述的"原始 CLIP 稠密图反相关"现象一致），
说明骨干与读出管线本身是对的。若审阅认为需要硬复现，跑 VOC21 需要 mmseg==1.1.1 +
mmcv==2.0.1 的锁版环境（DOSSIER §5.1），属独立任务卡。


---

## 八、主 agent 定案与交办的落地（2026-08-03，RO-1 交付之后）

| 事项 | 定案 | 本目录里的落地 |
|---|---|---|
| **U-RO1-2**（长描述 vs 短名词） | **主协议用长描述，短名词作对照**。理由：SFT 指令里出现的就是长描述，主协议必须与下游真实使用一致 | REPORT §七表 H 已改写，并补上「短名词口径下排名翻转、CI 重叠、三算子差异不构成可靠排序」的措辞；§〇 抬头加了「改末层值 +0.70、怎么改只值 0.02」的引述块 |
| **C2**（RO-4 判据） | **采纳**：RO-4 改为「在 RO-1 失败的那 18% 子集上 median AUC +≥0.05」 | `config/ro4_failure_subset.json`：45 源（SAM3 41 + L1 4，判据 `auc_desc<0.75`，最优档口径），逐源含 `auc_desc / auc_shuffle / area / region_b_kind / conf / pool / img_hw / auc_luma`；**RO-1 在该子集上的基线 = median 0.663（SAM3 段）、0.637（全体）**，RO-4 直接对表 |
| **scache `domain`** | 补显式 `domain` 字段坐实约定 | 已对 237 条 meta 原子重写：`norm.domain = [-3.3164, 4.0430]`（整臂 z 实测值域，**不是 [0,1]**），并加 `domain_note` 明写「`upsample_s` 默认 `clamp=(0.0,1.0)` 会把本臂的 z 截断，必须显式传 `clamp=None` 或先反算回余弦域」。主 agent 指出的 `ro9` 臂埋雷在本臂上同样成立（z 值域确实超出 [0,1]），现已显式化 |
| **DOSSIER 第 3 条（GEM min-max）** | 主 agent 记为**红线级**并收进档案 | 无需本目录改动；`NOTES §三` 保持原文 |
| **RO-X1 CLIP 侧半场** | 追加任务 | 单独交付 `experiments/ROX1_clipside_20260803/` |

### 8.1 `ro4_failure_subset.json` 的读法（给 RO-4 的接手说明）

- `rows[*].key` 与 RO-1 scache arm 的 `img_id` **同键**，可直接 `SCache.read(key, ihash)` 取到 RO-1 的 s 场做对照。
- 失败模式**统一**是「强纹理/强语义竞争背景吃掉主体」（REPORT §九的失败归因），
  所以这批样本正是 DINO 的语义-边界先验该发力的地方；若 RO-4 在这批上都拿不到 +0.05，
  「外挂 DINO 值一次额外前向」就没有立足点。
- 该子集**不是**独立留出集：它是按 RO-1 自己的表现挑出来的（选择性偏差）。
  RO-4 报数时必须**同时**报全集，防止只在这批上过拟合参数。
