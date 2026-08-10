# RD-G 方法详解：把 GLUT 渲染器的参数生成端换成 transformer

**实验目录**：`experiments/RDG_transformer_20260803/`（下文简称 `E/`）
**源码权威位置**：`/home/bc/VeraRetouch/model/glut_repro/`；`E/config/` 是交付快照。
实测比对：`model_rdg.py` / `train_rdg.py` / `data_rdg.py` 三个文件的快照与线上模块**逐字节相同**（`diff -q` 无输出）；`ci_checks_rdg.py` 与 `train_rdg2.py` 的线上版本比快照新（差异见 §7.4）。本文所有行号引用 `E/config/model_rdg.py`，与 `model/glut_repro/model_rdg.py` 等价。

**本文纪律**：每个数字后给出处（`文件:行` 或 `文件 :: json 键路径`）。追不到的写「**缺**」，全文不做估算。凡本文自己重跑得到的数字，标注「本文复算」。

---

## 1. 这个实验在回答什么问题

### 1.1 一句可证伪的问题

> **在渲染核心、tokenizer、数据、优化配方、天花板全部固定不变的前提下，把「条件 → 渲染器参数」这一段映射从 CGLUT 式 25 万参数 MLP 换成 4.4M/14M transformer，能买到多少 dB / ΔE00？这个收益是来自结构，还是仅仅来自参数更多？**

可证伪在两个方向：

- 如果 transformer 相对 MLP 基线的 ΔE00 p50 降幅 <15%，或方差比 ≤0.6 → 这条线不晋级（预注册判据，`docs/EXPERIMENTS_v3_2026-08-02.md:89`）。
- 如果**把 MLP 加宽到同参数量后追平 transformer** → 收益来自参数量而非结构，方法节里的 transformer 生成器就是无依据的堆砌。这就是 `mlp_wide` 臂存在的全部意义（§3.3）。

另有一条预注册的**死刑条款**：若 MLP 生成器已经达到渲染器天花板的 90%，则 RD-G 停工、预算转 s 轴（同上 `EXPERIMENTS_v3:89` 的「淘汰去向」列）。

### 1.2 「头重脚轻」的具体含义

批评是这样的：整套系统的前端（VLM + 生成器）会是几十亿到千万参数量级，而它最终吐出的**渲染器载荷只有 1,116 个数**，之后所有像素都由这 1,116 个数决定。看上去投入与产出严重不匹配 —— 「脚」太小，「头」白堆。

这个批评能否被正面回答，取决于一件事：**在「脚」一个参数都不加的前提下，把「头」做大是否真的能把这 1,116 个数生成得更好**。所以本实验的实验设计核心是：

- 渲染器载荷 **恒为 `23N+12 = 1116`（N=48）**，六个臂逐位相同（`model_rdg.py:111-113`；实测见 `E/config/ci_checks_rdg.json :: results[15].detail.payload = 1116`）；
- 图像 tokenizer 恒为同一个模块、**220,736 参数**，六臂逐位相同（`ci_checks_rdg.json :: results[15].detail.counts.<arm>.tokenizer`；该 json 覆盖 `ARMS` 里的五个 arm 名，`gbase_free` 的参数量见 `E/runs/gbase_free/metrics.json :: params`，与 `gbase` 逐位相同）；
- **唯一变量**是中间那段生成器：0.25M / 1.56M / 4.45M / 4.93M / 14.06M。

因此「transformer 值多少 dB」这个问题在本实验里是良定义的：它就是同一条流水线上换掉中间一段的差值。

---

## 2. 被测系统的完整数据流

```
条件 (I_in, after)  (B, 6, 128, 128)  ∈[0,1]
        │
        │  PairTokenizer            (model_rdg.py:274-308)   220,736 参数，六臂共享同一实现
        │  Conv 4×4/s4 → GELU → Conv 2×2/s2 → GELU → Conv 2×2/s2  (总下采样 16×)
        ↓
   tok (B, 64, 256)   patch token，8×8 网格
   sty (B, 128)       style code = tok.mean(1) 过两层 MLP
        │
        │  ★ 生成器 ★              唯一变量
        │  MLPGen (model_rdg.py:315-355) 或 TransformerGen (model_rdg.py:429-488)
        ↓
   z_prim (B, 48, 23)   逐 primitive 的原始 head 输出
   z_glob (B, 12)       全局的原始 head 输出
        │
        │  ParamHead                (model_rdg.py:162-215)   0 个可训练参数（只有 anchors buffer）
        │  逐条参数化约束（§5）
        ↓
   p = {mu, sigma, off, opacity, gate, M, b, G, c}   合计 1,116 个数
        │
        │  render(p, x)             (model_rdg.py:65-108)    0 参数，纯函数
        ↓
   f_θ(x)  (B, P, 3)
```

### 2.1 逐段的张量形状、参数量、是否共享

| 段 | 输入 | 输出 | 参数量 | 六臂共享 | 出处 |
|---|---|---|---|---|---|
| `PairTokenizer.stem` | (B,6,128,128) | (B,256,8,8) | 170,432 | 是（实现相同，权重各自训练） | `model_rdg.py:286-289` |
| `PairTokenizer.norm` LayerNorm(256) | (B,64,256) | (B,64,256) | 512 | 是 | `:290` |
| `PairTokenizer.to_style` | (B,256) | (B,128) | 49,408 | 是 | `:291-292` |
| `null_tokens` (1,1,256) + `null_style` (1,128) | — | — | 384 | 是 | `:294-295` |
| **tokenizer 合计** | | | **220,736** | 是 | 实测 `ci_checks_rdg.json :: results[15].detail.counts.mlp.tokenizer` |
| 生成器 | tok, sty, grid | z_prim (B,48,23), z_glob (B,12) | **0.25M–14.06M** | **否，唯一变量** | §3 |
| `ParamHead` | z_prim, z_glob | 参数 dict | **0**（`anchors` 是 buffer，`model_rdg.py:184`） | 是 | `:162-215` |
| `render` | p, x (B,P,3) | (B,P,3) | 0 | 是 | `:65-108` |

逐层参数量由本文按定义重算并与实测逐位对上（本文复算，见 §3.1 与 §3.5 的分解表）。

### 2.2 1,116 个参数分别是什么

`N_PRIM_OUT = 23`（`model_rdg.py:57`）、`N_GLOB_OUT = 12`（`:58`），`n_render_params(n) = 23n + 12`（`:111-113`）。

**逐 primitive 的 23 个数**（切片见 `ParamHead.forward`，`model_rdg.py:189-191`）：

| 切片 | 个数 | 量 | 在 `render` 里的作用（行号） |
|---|---|---|---|
| `z_prim[...,0:3]` | 3 | `mu_i` 高斯中心（RGB 立方体里的位置） | `diff = x − mu`，`:89` |
| `z_prim[...,3:6]` | 3 | `sigma_i` Cholesky 对角（三个主轴尺度） | `z0,z1,z2` 的分母，`:91-93`；`log_det`，`:95` |
| `z_prim[...,6:9]` | 3 | `off_i` Cholesky 次对角（L10, L20, L21，各向异性/相关性） | `:92-93` 的前向代入 |
| `z_prim[...,9]` | 1 | `o_i` opacity | 混合分子 `og`，`:97` |
| `z_prim[...,10]` | 1 | `g_i` 存在门（PLAN §1.1 加的那一个，GLUT 原版没有） | 同上，`:97` |
| `z_prim[...,11:20]` | 9 | `M_i` 逐 primitive 3×3 局部仿射矩阵 | `:103` |
| `z_prim[...,20:23]` | 3 | `b_i` 逐 primitive 平移 | `:105` |
| | **23** | | |

**全局的 12 个数**：

| 切片 | 个数 | 量 | 行 |
|---|---|---|---|
| `z_glob[:,0:9]` | 9 | `G` 全局仿射 3×3 | `:106` |
| `z_glob[:,9:12]` | 3 | `c` 全局平移 | `:107` |

`23 × 48 + 12 = 1116`。GLUT 原版是 `22N+12`；**多出来的那一个就是存在门 `g_i`**（`model_rdg.py:112` 的注释；对照证据：E1 的 N=48 逐 LUT 直接拟合记的是 **1068** 个参数 = `22×48+12`，`experiments/E1_cube_N_20260803/REVIEW-result.md:20`）。汇报时这两个数不要混。

### 2.3 渲染函数本身（六臂完全相同，且是被测出来的）

`render()`（`model_rdg.py:65-108`）是 GLUT Eq.1-3 的函数式写法：

1. 白化：`z0 = d0/σ0`，`z1 = (d1 − L10·z0)/σ1`，`z2 = (d2 − L20·z0 − L21·z1)/σ2`（前向代入，`:91-93`）——即对下三角 Cholesky 的三角解，代替 `torch.linalg.solve_triangular`。
2. 对数密度 `ld = −1.5·log(2π) − 0.5·log|Σ| − 0.5·maha`（`:96`）。
3. 权重 `w_i = o_i·g_i·exp(ld_i − m) / (Σ_j o_j·g_j·exp(ld_j − m) + e^{−m}·ε)`，`m = max_i ld_i`（`:97-101`），`ε = EPS_W = 1e-6`（`:52`）。把 log-max 提出来是为了 **σ 塌陷时 w→0 而不是 NaN**（`:76-77` 的注释）。
4. 输出 `f(x) = (Σ_i w_i M_i) x + Σ_i w_i b_i + G x + c`（`:103-108`）。

**渲染核心不是重写的，这是一条测试而不是一句声称**：`render()` 与现成的 `model/glut_repro/model.py::BatchedGLUT.forward` 在相同参数下 **max abs diff = 2.98e-7**（`ci_checks_rdg.json :: results[0].detail`，检查代码 `ci_checks_rdg.py:32-50`）。

**一个必须记住的退化模式**：如果所有高斯密度在某个 `x` 处全部下溢，`num→0`，`w→0`，混合分支消失，`f(x) = Gx + c` —— **渲染器只剩全局仿射**。Stage-1 没有 s 轴所以这条不会被触发，但它正是仓库 `CLAUDE.md`「s 缓存消费契约」里第一种失败模式的机制来源。

---

## 3. 六个臂的结构（本文重点）

臂表定义在 `model_rdg.py:495-502`：

```python
ARMS = {
    "mlp":      ("mlp", dict(width=128)),
    "mlp_wide": ("mlp", dict(width=880)),
    "gtiny":    ("tf",  dict(dim=192, depth=2, heads=6, cross_at=(1, 2))),
    "glite":    ("tf",  dict(dim=256, depth=4, heads=8, cross_at=(1, 3))),
    "gbase":    ("tf",  dict(dim=384, depth=6, heads=8, cross_at=(1, 3, 5))),
}
```

`gbase_free` 不是一个 arm 名，而是 `--arm gbase --free`（`E/tools/run_stage1.sh:60`；`job.marker` 记录的启动命令可查）。

### 3.0 六臂总表

| 臂 | 族 | d / width | L | heads | cross-attn 层（1-based） | routing 层（1-based） | 生成器参数 | 全模型参数 | 角色 |
|---|---|---|---|---|---|---|---|---|---|
| `mlp` | CGLUT §3.2 MLP | width 128 | 3-Linear trunk | — | — | — | **251,292** | 472,028 | 同口径基线 |
| `gtiny` | transformer | 192 | 2 | 6 | {1,2}（全部层） | {1,2} | **1,561,445** | 1,782,181 | 容量曲线低端 |
| `mlp_wide` | CGLUT §3.2 MLP | width 880 | 3-Linear trunk | — | — | — | **4,933,244** | 5,153,980 | **容量对齐对照** |
| `glite` | **G-Lite** | 256 | 4 | 8 | {1,3} | {2,3,4} | **4,447,846** | 4,668,582 | PLAN §1.4 ≈5M 档 |
| `gbase` | **G-Base** | 384 | 6 | 8 | {1,3,5} | {3,4,6} | **14,056,806** | 14,277,542 | PLAN §1.4 ≈15M 档，主方案 |
| `gbase_free` | 同 `gbase` | 384 | 6 | 8 | {1,3,5} | {3,4,6} | **14,056,806** | 14,277,542 | **无加固自由回归对照** |

参数量出处：实测 `ci_checks_rdg.json :: results[15].detail.counts.<arm>`；`gbase_free` 与 `gbase` 逐位相同（同一个 `--arm gbase`，只多一个 `--free`，而 `ParamHead` 的参数量是 0）。本文另在 CPU 上重新实例化五个臂逐层核对，**全部对上**（本文复算）。

`cross_at` 是 **1-based**，源码里 `cross_set = {c - 1 for c in cross_at}` 转 0-based（`model_rdg.py:450`）。

**routing 层不是手写的，是按 depth 自动算的**（`model_rdg.py:454-457`）：

```python
route_at = tuple(sorted({max(1, depth // 2), max(1, (3 * depth) // 4), depth}))
```

- L=2 → `{1, 1, 2}` 去重后 **{1, 2}**（只有 2 个 tap）
- L=4 → **{2, 3, 4}**
- L=6 → `{3, 4, 6}`（`3*6//4 = 4`）→ **{3, 4, 6}**

### 3.1 `mlp` —— CGLUT §3.2 形状，width 128

结构（`model_rdg.py:315-355`）：

```
tok.mean(1) (B,256) → to_code Linear(256→64)              # 64 维 style code
  → trunk: [Linear(64→W) ReLU] [Linear(W→W) ReLU] [Linear(W→W) ReLU]   # 3-Linear shared encoder
  → 五个逐类 head：
      mu_head   Linear(W→W) ReLU zero_linear(W→48*3)
      col_head  Linear(W→W) ReLU Linear(W→W) ReLU zero_linear(W→48*12)
      chol_head zero_linear(W→48*6)
      op_head   zero_linear(W→48*2)
      glob_head zero_linear(W→12)
  → z_prim = cat([mu(3), chol(6), op(2), col(12)]) = 23   # 与 ParamHead 的切片顺序一致
```

`W=128` 复现论文的 "Large" 档（`model_rdg.py:19-20` 的注释）。**与 CGLUT 原文唯一的改动**：原文的条件是逐 style 的可学查表 `E ∈ R^{L×64}`（无法泛化到未见 style），这里改成从图像对算出的 64 维 code（`model_rdg.py:316-322`，对应 PLAN §1.5 第 1 步「E → Proj(embed)」）。下游一字不改。

逐层参数量（本文复算，与实测 251,292 逐位对上）：

| 层 | 计算 | 参数 |
|---|---|---|
| `to_code` Linear(256→64) | 256·64+64 | 16,448 |
| `trunk` Linear(64→128) + 2×Linear(128→128) | 8,320 + 2×16,512 | 41,344 |
| `mu_head` Linear(128→128) + zero(128→144) | 16,512 + 18,576 | 35,088 |
| `col_head` 2×Linear(128→128) + zero(128→576) | 33,024 + 74,304 | 107,328 |
| `chol_head` zero(128→288) | | 37,152 |
| `op_head` zero(128→96) | | 12,384 |
| `glob_head` zero(128→12) | | 1,548 |
| **合计** | | **251,292** ✅ |

注意 `mlp` 臂的 `forward` 只返回一个 tap（`model_rdg.py:355` 返回 `[(z_prim, glob)], None`），因此**它没有 aux 损失、没有 routing 熵项**（训练侧 `train_rdg.py:329-341`：`len(plist)-1 = 0` ⇒ aux 恒为 0；`extra is None` ⇒ 不加熵项）。这是 MLP 与 transformer 在损失上的一处结构性不对称，**如实登记**：两族并非逐项同配方，transformer 多了两项辅助损失。这两项的单独消融 —— **缺**。

### 3.2 `mlp_wide` —— 同结构，width 880

`MLPGen` 里 `width` 是唯一的 knob（`model_rdg.py:325-328`：`self.W = int(width)`，所有 head 的隐层宽度都取 `W`）。`width=880` 使生成器参数达 **4,933,244**，比 `glite` 的 4,447,846 **高 10.9%**（本文复算：4933244/4447846 = 1.109）。逐层分解（本文复算，逐位对上）：

| 层 | 参数 |
|---|---|
| `to_code` | 16,448 |
| `trunk` Linear(64→880)+2×Linear(880→880) | 1,607,760 |
| `mu_head` | 902,144 |
| `col_head` | 2,058,016 |
| `chol_head` | 253,728 |
| `op_head` | 84,576 |
| `glob_head` | 10,572 |
| **合计** | **4,933,244** ✅ |

**880 这个具体数字是怎么搜出来的 —— 缺**（源码只给结果，docstring 只写 "widened to the transformer's budget"，`model_rdg.py:20-21`）。

### 3.3 `mlp_wide` 存在的意义（本实验能否成立的关键）

审稿人/mentor 必问的一句是「你换了个大 20 倍的东西，是不是只是参数多了」。`mlp_wide` 就是为这一句准备的：它与 `mlp` **逐行同结构**、与 `glite` **同参数量级（甚至更大 10.9%）**，并且与全部六臂共享同一个 tokenizer、同一份数据、同一套配方、同一个渲染核心、同一条天花板。于是「参数量」这一维被**钉死**，剩下的差就只能归给结构。

收敛后（两臂都是 step 8000）的实测（本文复算自 `E/runs/glite/metrics.json` 与 `E/runs/mlp_wide/metrics.json`）：

| 档 | `glite` 4.45M | `mlp_wide` 4.93M | 差 |
|---|---|---|---|
| 未见源 ΔE00 p50 | 3.6097 | 7.1161 | **−49.27%** |
| 未见源 PSNR | 27.2762 | 21.7201 | **+5.5560 dB** |
| 未见 LUT ΔE00 p50 | 5.5340 | 8.4220 | **−34.29%** |
| 未见 LUT PSNR | 22.5908 | 19.5756 | **+3.0152 dB** |

**这条对照不依赖任何门槛线的位置**。REPORT.md 自己撤回过一句依赖门槛线的表述（§12.1a，`E/REPORT.md:439-455`：「MLP-Wide 只降 14.67%，差 0.33 个百分点没过 15% 门」——那是 step-4000 的切片，step 5000 起就翻了）。同参数量对照没有这个毛病：跨四个检查点单调扩大（REPORT.md:459-464 的 20.3 → 30.8 → 39.7 → 42.8%，step 8000 时为 49.27%，本文复算）。

### 3.4 `gbase_free` 存在的意义（PLAN 明写的必做对照）

PLAN §3「第二级 / Stage 1」原文：「**必做对照：无加固自由回归版**（<0.5dB → 色彩域温和全面简化；≥2dB → 确认加固）」（`docs/PLAN_v2_local-retouch_2026-07-31.md:258`）。

`--free` 把 §5 的三件加固**同时**关掉（`model_rdg.py:193-202`）：

| 量 | 加固档 | `--free` 档 |
|---|---|---|
| `mu` | `anchor + (1/3)·tanh(z)` | `z`（无锚定、无界） |
| `sigma` | `0.02 + 0.48·sigmoid(z)` | **`exp(z).clamp(1e-3, 10)`（裸 exp）** |
| `gate` | `sigmoid(z+4)` | 恒为 1（`torch.ones_like`） |
| `G` | `0.1·z` ⇒ 初始 **0** | **`z + I` ⇒ 初始 I（反模式）** |

生成器参数量**完全相同**，因此这是一个纯参数化对照、零参数量差。

**注意**：三件事是**一起**改的，所以本实验**不能**把收益拆到单条约束上。μ 锚定 / σ 有界 / G 初始 0 的**逐条**消融 —— **缺**。

`--free` 是不是真的关掉了约束，也有 CI 钉着：`ci_checks_rdg.json :: results[4].detail = 10.0`（free 档 sigma 在 logit=3 时能冲到 clamp 上界 10.0，检查代码 `ci_checks_rdg.py:75-80`）。

### 3.5 `gtiny` / `glite` / `gbase` —— transformer 族的逐层分解

三个臂共用 `TransformerGen`（`model_rdg.py:429-488`），只差 `dim/depth/heads/cross_at`。

**G-Base（d=384, L=6, heads=8, cross@{1,3,5}）逐层**（本文复算，与实测 14,056,806 逐位对上）：

| 组件 | 计算 | 参数 | 行 |
|---|---|---|---|
| `query` (1, N+1=49, 384) | 49×384 | 18,816 | `:445` |
| `register` (1, 4, 384) | 4×384 | 1,536 | `:449` |
| `route_logit` (3,) | | 3 | `:458` |
| `kv_proj` Linear(256→384) | 98,304+384 | 98,688 | `:446` |
| `kv_norm` LayerNorm(384) | | 768 | `:447` |
| `pe` Linear(2→32, no bias) + Linear(64→384) | 64 + 24,960 | 25,024 | `:389-393` |
| 3 × cross layer（i=0,2,4） | 每层 ModLN×3 (3×99,072) + self-attn 591,360 + cross-attn 591,360 + FFN 1,181,568 = 2,661,504 | 7,984,512 | `:404-426` |
| 3 × non-cross layer（i=1,3,5） | 每层 ModLN×2 + self-attn + FFN = 1,971,072 | 5,913,216 | 同上 |
| `out_norm` LayerNorm(384) | | 768 | `:459` |
| `prim_head` `zero_linear(384→23)` | 8,832+23 | 8,855 | `:460-461` |
| `glob_head` `zero_linear(384→12)` | 4,608+12 | 4,620 | `:462` |
| **合计** | | **14,056,806** ✅ | |

单层内部（`DecoderLayer.__init__`，`model_rdg.py:404-417`，d=384）：

| 子模块 | 参数 |
|---|---|
| `ModLN` = LayerNorm(elementwise_affine=False) + Linear(128→768) | 99,072 |
| `nn.MultiheadAttention(384, 8)`：in_proj 3×384² + 1,152，out_proj 384²+384 | 591,360 |
| FFN Linear(384→1536) + GELU + Linear(1536→384)（mlp_ratio=4） | 1,181,568 |

**cross-attn 只在指定层，非指定层根本不建这两个模块**（`model_rdg.py:410-414`），所以每关掉一层 cross 就少 `ModLN + MultiheadAttention = 690,432` 个参数。

**G-Lite（d=256, L=4, cross@{1,3}）**：query 12,544 + register 1,024 + route_logit 3 = 13,571；kv_proj 65,792；kv_norm 512；pe 16,704；2×cross layer 2,500,096 + 2×non-cross 1,841,664 = 4,341,760；out_norm 512；prim_head 5,911；glob_head 3,084 ⇒ **4,447,846** ✅（本文复算）

**G-Tiny（d=192, L=2, heads=6, cross@{1,2}）**：query 9,408 + register 768 + route_logit 2 = 10,178；kv_proj 49,344；kv_norm 384；pe 12,544；2×cross layer 1,481,856；out_norm 384；prim_head 4,439；glob_head 2,316 ⇒ **1,561,445** ✅（本文复算）

注意 `gtiny` 的 `route_logit` 只有 **2** 个元素（L=2 时 route_at 去重成 {1,2}）——这对熵下界有影响，见 §4.5。

---

## 4. transformer 生成器的具体设计

`TransformerGen.forward`（`model_rdg.py:472-488`）的完整流程：

```python
kv = kv_proj(tok)                                   # (B,64,256) → (B,64,d)     :475
kv = kv + pe(gh, gw, ...).unsqueeze(0)              # 位置编码只加在 kv 上       :476
kv = cat([kv, register.expand(B,-1,-1)], dim=1)     # (B, 64+4, d)               :477
kv = kv_norm(kv)                                    #                            :478
q  = query.expand(B,-1,-1)                          # (B, 49, d)                 :479
taps = []
for i, layer in enumerate(layers):                  #                            :481-484
    q = layer(q, kv, sty)                           # self-attn → [cross] → FFN，全部 ModLN 前置
    if (i+1) in route_at: taps.append(q)
wts   = softmax(route_logit)                        #                            :485
fused = Σ w_k · tap_k                               #                            :486
outs  = [head(t) for t in taps]                     # 每 tap 一个 aux 头          :487
return [head(fused)] + outs, route_entropy()        #                            :488
```

`_head`（`:464-466`）：`h = out_norm(h)`；`prim_head(h[:, :N])` → (B,48,23)，`glob_head(h[:, N])` → (B,12)。

### 4.1 48 个 primitive query + 1 个 global query

`self.n_q = self.N + 1`（`:444`），`query = nn.Parameter(randn(1, n_q, d) * 0.02)`（`:445`）。

- **前 48 个 query** 一一对应 48 个高斯 primitive：`h[:, :48]` 送进 `prim_head`，每个出 23 个数（`:466`）。
- **第 49 个 query（global query）**：`h[:, 48]` 送进 `glob_head`，出 12 个数，即全局仿射 `G` 与平移 `c`（`:466`）。

这条布局对应 PLAN §1.4 的「全局量走调制、空间量走 cross-attn，不混」（`PLAN:91`）：全局仿射由一个专门的 query 承载，不与 primitive query 争抢。

**PLAN 字面写的是 32 个 primitive query**（`PLAN:83`）。本实验用 **48**，依据是 E1 的结果审阅把 `N* = 48` 推翻了初报的 32（`experiments/E1_cube_N_20260803/REVIEW-result.md:20, 44-48`：N* 的定义是「最小满足 p90<1.0 且 p99<2.0」，N=32 未过门；N=48 的 p50/p90/p99 = 0.467/0.785/1.264、PSNR 44.33、alive 0.993）。

主损失放在 **LUT 立方体空间**（置换不变），这是 PLAN 的刻意选择（`PLAN:101`：「置换不变，绕开匈牙利匹配」）。代价是**没有对 query 的直接监督** —— PLAN 自己也点了这个代价并要求用「锚点初始化 + query 直接监督」补，本实验只做到了前一半（后一半即 `L_param`/`L_prequery`，见 §6.5）。

### 4.2 K/V 侧：patch token + 4 个 register token

`register = nn.Parameter(randn(1, 4, d) * 0.02)`（`:449`），在**位置编码之后**拼到 kv 序列尾部（`:476-477`），因此 **register token 不带任何位置编码**，随后与 patch token 一起过 `kv_norm`（`:478`）。

出处：Darcet, Oquab, Mairal, Bojanowski, *Vision Transformers Need Registers*，**arXiv:2309.16588**。本实验的 NOTES 明确记载已打开原始来源核实，且记了一条克制的说明：**摘要页不含默认 register 数量**，所以「4 个」这个数字来自 PLAN §1.4 自己（`PLAN:83`「4 个 register token（第一版前必做）」），**不向该论文归因任何具体数字**（`E/NOTES.md:28`）。

`n_register=4` 是 `TransformerGen.__init__` 的默认值（`:441`），六个 transformer 臂都没有覆盖它。**register token 的开关消融 —— 缺**（PLAN §5 的消融清单里列了这一行，`PLAN:355`，本实验未做）。

### 4.3 learnable Fourier 位置编码只加在 key 上

实现（`LearnableFourierPE`，`model_rdg.py:381-401`）：

```python
freq = nn.Linear(2, 32, bias=False);  init.normal_(freq.weight, std=8.0)     # 频率可学
out  = nn.Linear(64, dim)
# forward:
pos  = 归一化到 [-1,1] 的 (gx, gy) 网格，(gh*gw, 2)
a    = freq(pos)                       # (T, 32)
return out(cat([sin(a), cos(a)], -1))  # (T, dim)
```

调用点只有一处：`kv = kv + self.pe(...)`（`:476`）。**query 侧没有任何位置编码**。

- **规格出处**：PLAN §1.4 逐字写「位置编码 learnable Fourier 加 key」（`PLAN:91`）。
- **机制后果（本文按源码推）**：query 是 48+1 个自由学到的向量，本身不携带任何图像位置身份；空间信息进入 query 的**唯一**通道是 cross-attn 对带位置标签的 key 做加权（`:424`）。这与 PLAN 同一句里的「全局量走调制、空间量走 cross-attn，不混」是一致的：query 是「要生成哪个色彩 primitive」的槽位，不是「图像上哪个位置」的槽位；给 query 加空间 PE 会强行给每个色彩 primitive 指派一个空间身份。
- 更深一层的「为什么必须是 key-only 而不是两侧都加」的实验依据 —— **缺**（本实验未做该消融）。

### 4.4 ModLN：每子层独立参数，由 style code 驱动

```python
class ModLN(nn.Module):                                     # model_rdg.py:362-378
    norm = LayerNorm(dim, elementwise_affine=False)         # :371  自身无 affine 参数
    proj = Linear(style=128 → 2*dim);  weight、bias 全零初始化   # :372-374
    def forward(x, sty):
        scale, shift = proj(sty).unsqueeze(1).chunk(2, -1)   # (B,1,d) 各一份
        return norm(x) * (1.0 + scale) + shift               # :378
```

**具体怎么调制**：`sty` 是 (B,128) 的全局 style code（tokenizer 的 `to_style` 出，`model_rdg.py:302`）。每个 `ModLN` 用自己**独立**的 `proj` 把它映成一对 (scale, shift)，在 token 维广播，对该子层输入的 LayerNorm 输出做 FiLM 仿射。

**「每子层独立参数」在代码里是这样落实的**：`DecoderLayer` 里建了 **三个**互相独立的 `ModLN` —— `n1`（self-attn 前）、`n2`（cross-attn 前，仅 cross 层有）、`n3`（FFN 前）（`model_rdg.py:408, 412, 415`）。所以 G-Base 六层共 **15 个** `ModLN`（3 个 cross 层各 3 个 + 3 个非 cross 层各 2 个），每个都有自己的 `Linear(128→768)`（99,072 参数），合计 1,486,080 参数，占生成器的 10.6%（本文复算）。

**零初始化的意义**：`proj` 权重与 bias 全零 ⇒ 训练第 0 步 `scale = shift = 0` ⇒ `ModLN` 退化成一个普通的无 affine LayerNorm。也就是说 style 调制在训练开始时是**关闭**的，模型自己决定要不要打开它。

规格出处：PLAN §1.4「全局 style 走 **ModLN 每子层独立参数**（可插值性是功能，删=回退）」（`PLAN:91`）。「可插值性」的实测验证（在两个 style code 之间插值、看输出 LUT 是否平滑过渡）—— **缺**。

### 4.5 energy routing 与熵下界

**做什么**：不只用最后一层的 query 输出，而是在 `route_at` 指定的三个深度各取一次 tap，用一个可学的 softmax 权重把它们融合（`model_rdg.py:480-486`）：

```python
taps  = [q_after_layer_k  for k in route_at]
wts   = softmax(route_logit)            # route_logit 初始全 0 ⇒ 初始等权
fused = Σ_k wts[k] · taps[k]
```

融合后的 `fused` 走主头，**每个 tap 另外各走一个 aux 头**（`:487`），共享同一套 `out_norm/prim_head/glob_head` 权重。训练侧主损失只用 `plist[0]`（fused），其余 tap 进 aux 损失，权重 0.2（`train_rdg.py:329-338`）。

**熵下界解决什么问题**：如果不加约束，`route_logit` 的 softmax 会塌到最后一层（这是 routing 的已知退化——最后一层的表示总是「最成熟」的，梯度最容易把权重全推给它），三个 tap 就退化成「只用末层」，routing 与 aux 头全部变成装饰。PLAN §1.4 原文：「取层 {L/2,3L/4,L} energy-routing（**entropy 下界防塌末层**）」（`PLAN:91`）。

实现（`train_rdg.py:339-341`）：

```python
if extra is not None:                       # extra = route_entropy()
    hmin = 0.5 * math.log(3.0)              # = 0.5493
    loss = loss + args.w_route * torch.relu(hmin - extra)     # w_route = 0.01
```

`route_entropy()`（`model_rdg.py:468-470`）= `−Σ p log p`，`p = softmax(route_logit)`。

**一处必须如实说的细节**：`hmin` 硬编码为 `0.5·ln 3 = 0.5493`，即「三个 tap 的一半熵」。但 `gtiny`（L=2）只有 **2** 个 tap，其最大可能熵是 `ln 2 = 0.6931`。所以对 `gtiny` 而言这个下界要求它的两 tap 分布熵不低于最大熵的 79%——**可满足但显著更紧**，而对 L=4/L=6（3 个 tap，最大熵 `ln 3 = 1.0986`）只要求 50%。这不是 bug（不会导致不可行），但**三个 transformer 臂的 routing 正则强度并不等价**。训练结束时各臂的 `route_logit` 实际取值 —— **缺**（未落盘，`metrics.json` 不含该量；只在 `best.pt` 的 state_dict 里）。

「三层 routing vs 末层」的消融 —— **缺**（PLAN §5 消融清单有这一行，`PLAN:355`，本实验未做）。

### 4.6 未实例化的部分及原因

PLAN §1.4 的 token 清单是「primitive query 32 + global 1 + curve 1 + mask 5 = 39」（`PLAN:83`），另有「文本 ≤32 走独立 cross-attn」，G-Base 规格里还有「文本@{2,4,6}」（`PLAN:87`）。本实验实例化的只有 **48 primitive + 1 global**。

| 未建的东西 | 原因 | 出处 |
|---|---|---|
| **5 个 mask token** | 掩膜基底不在 Stage-1 的 cube 目标里，建了就是 5 个无监督死 token | `model_rdg.py:432-435`；`E/NOTES.md:86-87`（决定 7） |
| **1 个 1D 曲线 token** | PLAN §1.4 的输出头清单本身就没有曲线头；加曲线会同时改**渲染器**，把「生成器容量」和「渲染器容量」混在一起。曲线是 RD-I 的活 | `E/NOTES.md:84-85`（决定 6） |
| **文本 cross-attn 分支（G-Base 的文本@{2,4,6}）** | 本轮条件只有图像对，没有文本条件；文本档列为待决策 #1 的后续 wave | `E/NOTES.md:213-216`（待决策 1） |
| **s 轴进 decoder 的 E2 位置编码** | Stage-1 没有 s 轴 | `model_rdg.py:39`；`PLAN:91` |
| **G-Comp（slot-axis softmax + Group DETR K=4）** | PLAN 写明「仅塌陷时启用」，本轮未观察到 query 塌陷（死原语比例见 §8） | `PLAN:87`；`E/REPORT.md:356-357` |

### 4.7 条件形态：为什么是 (I_in, after) 6 通道而不是 PLAN 字面的单流

PLAN §3 Stage 1 字面写「输入=LUT 作用后的图」（`PLAN:258`），即单流 after。本实验改成 `(I_in, after)` 6 通道叠放（`PairTokenizer(in_ch=6)`，`model_rdg.py:283`；训练侧拼接在 `train_rdg.py:307-309`）。

理由（`E/NOTES.md:76-78`，决定 3）：只给 after 时 **LUT 不可辨识** —— 同一张 after 可以由无数组 (源图, LUT) 产生，容量比较会退化成「猜风格先验」而不是「生成参数」。after-only 单流档列为待决策 #1 的对照，**本轮未跑 —— 缺**。

这是一处**偏离权威文档字面**的设计决定，NOTES 里已按协议登记为待主 agent 决策项而非静默拍板（`E/NOTES.md:213-216`）。

### 4.8 condition dropout 的落点

`PairTokenizer.forward(img, drop)`（`model_rdg.py:297-308`）：当 `drop[b]` 为真时，该样本的 **整段 patch token 换成可学常量 `null_tokens`、style code 换成可学常量 `null_style`**（`:303-307`）。训练时 `drop = rand(B) < 0.15`（`train_rdg.py:315`）。

这个可学常量不只是训练技巧，它同时是**评测里 Δ_const 的定义所依赖的对象**：Δ_const = PSNR(真条件) − PSNR(把整个 batch 都置为 null 条件)（`train_rdg.py:130-135, 175-176`）。见 §7.1。

---

## 5. 输出头的参数化约束（红线所在，逐条）

全部在 `ParamHead.forward` 的加固分支（`model_rdg.py:203-212`），逐行照抄 PLAN §1.4 的代码块（`PLAN:95-99`）。输出头是 **单个零初始化 Linear**（`zero_linear`，`model_rdg.py:262-267`：weight 与 bias 都 `init.zeros_`），PLAN 明确「禁堆 MLP」（`PLAN:95`）。

| 量 | 参数化 | 零初始化时的值 | 行 |
|---|---|---|---|
| `mu_i` | `anchor_i + r·tanh(z)`，`r = 1/3` | `anchor_i` | `:204`（r 的默认值 `:178`） |
| `sigma_i` | **`0.02 + 0.48·sigmoid(z)`** | 0.26 | `:205`（常量 `:55-56`） |
| `off_i` | `0.1·z` | 0 | `:206` |
| `o_i` | `sigmoid(z − 2)` | 0.1192 | `:207` |
| `g_i` | `sigmoid(z + 4)` | 0.9820 | `:208` |
| `M_i` | `I + 0.1·z` | `I` | `:209` |
| `b_i` | `0.1·z` | 0 | `:210` |
| **`G`** | **`0.1·z`** ⇒ 初始 **0，不是 I** | **0** | `:211` |
| `c` | `0.1·z` | 0 | `:212` |

零初始化时的取值由本文在 CPU 上重新算过：`sigma = 0.2600`、`o = 0.1192`、`g = 0.9820`、`‖G‖ = 0.0`（本文复算）。

### 5.1 `G = 0.1z`（初始 0，不是 I）—— 防的是 f(x) = 2x

这是全仓红线里字面写出的一条（`CLAUDE.md`「红线速查」；`PLAN:98`「G 初始 0 非 I，否则 f(x)=2x」）。

**机制是纯算术，来自 `render` 的最后一行**（`model_rdg.py:108`：`return mix + glob`）：

- 零初始化时 `M_i = I`、`b_i = 0`，而 `w` 是归一化过的（Σw_i ≈ 1，`:101`），所以 **`mix = (Σ w_i I) x + 0 = x`**。混合分支**自己就已经复现了恒等**。
- 如果 `G` 再被初始化成 `I`、`c=0`，那么 `glob = x`，输出 `f(x) = x + x = 2x`。

本文实测验证（本文复算，CPU，4096 个随机色）：

| 档 | `f(x)/x` 中位 | max abs(f(x) − x) | max abs(f(x) − 2x) | 与恒等的 max ΔE00 |
|---|---|---|---|---|
| 加固（`G = 0.1z`） | **1.0000** | **0.0000** | 1.0000 | **0.0001** |
| `--free`（`G = z + I`） | **2.0000** | 1.0000 | **0.0000** | **43.08** |

也就是说反模式下模型在第 0 步就偏离恒等 **ΔE00 = 43**（JND ≈ 1），整个训练前期都在把这个 2× 的偏差纠回来。

### 5.2 `sigma = 0.02 + 0.48·sigmoid(z)`（禁裸 exp）—— 防两个方向的发散

裸 `exp` 在两侧都无界：

- **σ → 0**：`log_det → −∞`、`maha → +∞`，密度要么爆掉要么下溢。本实现把 log-max 提出来（`model_rdg.py:98-101`），把「爆 NaN」换成「w → 0」（`:76-77` 的注释），代价是那个 primitive 直接失效。
- **σ → 大**：每个高斯覆盖整个立方体，`w` 趋于均匀，混合分支退化成一个全局仿射，48 个 primitive 的表达力全部作废。

有界 sigmoid 把 σ 钉在 **[0.02, 0.50]**。CI 用极端 logit（±50）实测夹在 **[0.0200, 0.5000]**（`ci_checks_rdg.json :: results[3].detail`，检查代码 `ci_checks_rdg.py:67-73`）。对照臂用裸 exp，实测能冲到 clamp 上界 **10.0**（`results[4].detail`）。

数值出处：PLAN §1.4 在这一行旁边引了「GRM: sigmoid vs exp = +3.08 dB」（`PLAN:93`）。**该 GRM 原始论文本文未打开核实 —— 缺（外部核实）**；本实验自身没有单独对这一条做消融（三件加固一起改，见 §3.4）。

### 5.3 `mu = anchor + (1/3)·tanh(z)` —— 防 primitive 漂移/塌堆

`tanh` 把每个 primitive 的中心锁在自己 anchor 周围半径 1/3 的方块内，因此：primitive 不会全部漂到立方体外（漂出去 = 该 primitive 对任何合法颜色都没有密度，等于死掉），也不会全部塌到同一处。

anchor 来自 **Stage-0 k-means**（PLAN §1.4「anchor 来自 Stage-0 k-means」，`PLAN:96`）。本实验的具体来源是 `fit_ceiling.py` 的副产物：对 384 个 P-train LUT 逐 LUT 直接拟合得到的 μ 做 k-means（`E/tools/fit_ceiling.py:191-198`）。

**这里有一个容易看错的文件关系，必须写清**：

- 各臂训练用的 anchors 是 `--anchors` 的默认值 `E/runs/ceiling/anchors.npy`（`train_rdg.py:218-219`；六臂的 `metrics.json :: config.anchors` 逐个确认都是这个路径）。
- 正式天花板 `runs/ceiling_kmeans` 用的是 `E/config/anchors_stage0_kmeans.npy`（`E/config/ceiling_kmeans.json :: config.anchors`）。
- **这两个文件 md5 相同**（`578140f068eb087d37f92f82bf45cf22`，本文复算）⇒ **臂与正式天花板确实同 anchor**，REPORT 的说法成立。
- `runs/ceiling_kmeans/anchors.npy` 是**第二代** k-means（在已经用一代 anchor 拟合出的 μ 上再聚一次），md5 不同、**没有任何东西消费它**，是一个副产物。

`ParamHead` 在 `anchors=None` 时回退到 `uniform_grid_mu(N)`（`model_rdg.py:183`）。

### 5.4 `M = I + 0.1·z`、`b = 0.1·z`、`off = 0.1·z`、`c = 0.1·z`

`M` 是 identity-centered residual：零初始化时 `M = I`，这是 §5.1 里「mix 自己就等于 x」的前提。`0.1` 这个缩放对所有这些量都一样，作用是把零初始化点附近的 Jacobian 压小十倍，让 head 的输出 `z` 需要走比较大的距离才能造成大的参数变化（即在零点附近是一个「慢启动」）。**`0.1` 这个具体系数的来源除了 PLAN 逐字之外 —— 缺**。

### 5.5 `o = sigmoid(z−2)`、`g = sigmoid(z+4)` —— 以及为什么门必须乘在混合分子上

两者在 `render` 里进入同一个位置：`og = (opacity * gate).unsqueeze(-1)`，乘在混合分子上（`model_rdg.py:97`）。

**这个放置位置是被恒等 CI 逼出来的**（`model_rdg.py:80-84` 的注释）：零初始化时所有 `g_i ≡ sigmoid(4) = 0.982`，作为**公共因子**在归一化的分子分母里对消，`f(x) = x` 成立；如果做成 **payload 门**（即 `f_i = g_i·(M_i x + b_i)`），公共因子不再对消，`f(x) = 0.982x`，ΔE00 ≈ 1，直接违反 PLAN §1.4 的 `max ΔE00 < 1e-4`。

`o` 初始偏小（0.119）、`g` 初始偏大（0.982），**这两个偏置的具体数值（−2 / +4）逐字来自 PLAN §1.4（`PLAN:97`），本文追不到更深的推导 —— 缺**。

**已登记的冗余**：`o` 与 `g` 都乘在同一处，在当前损失下数学上不可辨识。保守默认是两者都保留（规格忠实度），并在报告里注明冗余；备选（合并为一个门，或把 g 改成 payload 门并放弃恒等 CI）留给主 agent 拍板（`E/NOTES.md:226-228`，待决策 4）。**这个冗余的定量代价 —— 缺**。

另有一个副作用值得记住：`dead_prim_frac` 的定义是 `o·g ≤ 0.05` 的 primitive 比例（`train_rdg.py:126, 185-186`）。`--free` 档里 `g ≡ 1`、`o = sigmoid(z)`，几乎不可能压到 0.05 以下，所以 **`gbase_free` 的死原语比例恒为 0（实测 `runs/gbase_free/metrics.json :: val_img.dead_prim_frac = 0`）—— 这一列对 free 臂没有信息量**，不要拿它跟其他臂并排解读。

### 5.6 零初始化恒等自检：33³ 全格点 max ΔE00 = 8.87e-5

PLAN §1.4 自带的 CI 是 `max ΔE00(f(x), x) < 1e-4 over 33³`（`PLAN:99`）。

- 实测 **8.870669e-05 < 1e-4** PASS（`ci_checks_rdg.json :: results[1].detail`，检查代码 `ci_checks_rdg.py:52-61`）。
- 五个臂各自实例化后再测一遍，全部同值 8.870669e-05（`results[10..14]`，代码 `ci_checks_rdg.py:139-150`）——同值是预期的：零初始化的头对所有臂给出完全相同的 z=0。

**一个 CI 没覆盖到的角落，本文补测了**：`ci_checks_rdg.py:53` 与 `:142` 建 `ParamHead(48)` / `RDGModel(arm, 48)` 时都**没传 anchors**，所以恒等 CI 是在**均匀网格 anchor** 上跑的，而训练实际用的是 Stage-0 k-means anchor。恒等性质原则上是 anchor 相关的（若某处 `x` 落在所有高斯的有效支撑之外，`w→0`、`mix→0`、`f(x)=0≠x`）。本文用训练实际使用的 `config/anchors_stage0_kmeans.npy` 重跑了这条检查：

| anchor | 33³ 全格点 max ΔE00 | 均值 |
|---|---|---|
| 均匀网格（CI 用的） | 8.713e-05 | 2.943e-05 |
| **Stage-0 k-means（训练实际用的）** | **8.732e-05** | 2.918e-05 |

（本文复算，CPU fp32；与落盘的 8.87e-5 差在 fp32/设备精度尘埃量级。）**结论：恒等在训练实际使用的 anchor 上同样成立**，但这条覆盖是本文补的，原 CI 里没有。

---

## 6. 训练配方

驱动脚本 `E/config/train_rdg.py`（= `model/glut_repro/train_rdg.py`）。配方逐字来自 PLAN §3「第二级 / Stage 1」（`PLAN:258`）。

### 6.1 优化器与调度

| 项 | 值 | 出处 |
|---|---|---|
| 优化器 | AdamW，β=(0.9, 0.95) | `train_rdg.py:266-268` |
| weight decay | 0.05，**但 ndim≤1 的参数、`query`、`register`、`null_*` 不衰减**（分两个 param group） | `train_rdg.py:262-268` |
| lr | 4e-4 | `train_rdg.py:208`；实测 `runs/*/metrics.json :: config.lr = 0.0004` |
| warmup | 线性 2000 步 | `train_rdg.py:210, 271-272` |
| warmup 后 | 余弦退火到 `steps`，最低降到峰值的 1% | `train_rdg.py:273-274`（`0.5(1+cos)·0.99 + 0.01`） |
| grad clip | 全局范数 1.0 | `train_rdg.py:211, 344` |
| 精度 | bf16 autocast，**只包 `params_from_image`**；`render` 与损失在 fp32 | `train_rdg.py:298, 324-325` |
| batch | 768 | `run_stage1.sh:21`；实测 `metrics.json :: config.bs = 768` |
| 步数 | 8000（首次提交是 15000，因 I/O 事故重启时降到 8000） | `run_stage1.sh:20`；`E/STATUS.md:73` |
| seed | 20260803（全局 torch/numpy），评测色另有 seed 777 | `train_rdg.py:225, 241-242`；`E/config/env.txt` |
| TF32 | matmul 与 cudnn 都开 | `train_rdg.py:243-244` |

### 6.2 condition dropout

`p = 0.15`，drop 到**可学常量**（不是 0，不是均值）：`null_tokens` / `null_style`（`train_rdg.py:212, 315`；`model_rdg.py:294-295, 303-307`）。

### 6.3 损失

```python
loss = L_cube(1.0) + 0.2 · aux/ntap + 0.01 · L_prior + 0.01 · relu(0.5·ln3 − H_route)
```

| 项 | 权重 | 定义 | 行 |
|---|---|---|---|
| `L_cube` | 1.0 | `(render(p_fused, x) - y).abs().mean()`，L1，x 是 4096 个采样色 | `train_rdg.py:331` |
| 每 tap aux | 0.2（再除以 tap 数） | 每个 routing tap 的头单独算一次 L1，**只用前 1024 个色**（`p_aux`） | `train_rdg.py:213, 327-333, 337` |
| `L_prior` | 0.01 | `z_prim².mean() + z_glob².mean()`，**只对主 tap**（`plist[:1]`） | `train_rdg.py:214, 334-336` |
| 路由熵下界 | 0.01 | `relu(0.5·ln3 − H)`，`H` = softmax(route_logit) 的熵 | `train_rdg.py:215, 339-341` |

`y` 由 `tri_lookup_bank` 从常驻显存的全语料 LUT bank 里三线性读出（`train_rdg.py:321-322`；`data_rdg.py:54-64`）。bank 尺寸 3,513 × 33³ × 3 fp32 ≈ **1.4 GB**（`train_rdg.py:279-281` 打印实际值；`n_presets` 见 `E/config/cache_report.json :: n_presets = 3513`）。

### 6.4 色彩采样规则（一半均匀、一半自身像素）

```python
xu = torch.rand(B, 2048, 3)          # 立方体内均匀随机色
xn = D.image_colors(src, 2048, g)    # 该样本 I_in 自身的像素色（放回抽样）
x  = torch.cat([xu, xn], 1)          # 每步 4096 个色
```
（`train_rdg.py:316-320`；`image_colors` 定义在 `data_rdg.py:100-106`。）

**两半各自的理由**（`data_rdg.py:14-19`；`E/NOTES.md:82-83` 决定 5）：

- **立方体均匀色**：PLAN 第一级要求「采样必含均匀格」。同时这是 E1 与本实验天花板使用的口径，只有同轴才能算「天花板占比」。缺了它，模型在图像里罕见的立方体角落上会完全失控，而这些角落**在烘焙成 33³ .cube 时是必须填的格点**。
- **样本自身像素色**：ENNELUT 的教训——只按立方体均匀色训会低估自然图上的表现，反之只按自然色训会在全 Hald 上崩。
- **一半一半**：让两条轨都不饿死（`data_rdg.py:18-19`）。

实测结果证明这两个口径确实差得远：G-Base 未见 LUT 档自然色 p50 **1.4931** vs 均匀色 p50 **5.2316**，差 **3.50×**（`runs/gbase/metrics.json :: val_lut.de00_nat_p50 / val_lut.de00_p50`，本文复算比值）。**论文报数必须两个口径都给。**

### 6.5 `L_param` / `L_prequery` 为什么没用：目标不存在

PLAN §3 Stage 1 的损失表是 `L_cube(1.0) + L_param(0.3) + L_prequery(0.2) + L_prior(0.01) + 每层 aux`（`PLAN:258`）。本实验**只有** `L_cube` + aux + `L_prior` + 熵项。

原因不是嫌麻烦，而是**回归目标不存在**：`L_param` 与 `L_prequery` 都需要 Stage-0 逐 LUT 的**金标准参数**当目标，而 E1 只把**指标**落了盘（`per_fit.jsonl` 里没有参数）（`E/REPORT.md:61-63`；`E/NOTES.md:217-222` 待决策 2）。

**代价（NOTES 自己写的）**：主损失放在置换不变的 cube 空间，本来就少了对 query 的去重压力，PLAN 要求用「锚点初始化 + query 直接监督」两条一起补（`PLAN:101`，援引 Mask2Former 的「可学但不监督=没改」）。本实验只做到了锚点初始化那一半。

**补救已就位**：本实验的 `fit_ceiling.py` 已经把 384 个 P-train LUT 的逐 LUT 拟合结果落盘（`runs/ceiling_kmeans/per_lut_p_train.npz`），下一轮可以直接当 `L_param` 目标（`E/REPORT.md:344-346`）。

### 6.6 其他明确不加的东西

- **任何感知/色相损失**：A0 实测 `10·L_hc` 是 **−10.73 dB**，根因是 CIELab 绝对值与 RGB[0,1] 的量纲不匹配（梯度差约 229×）；DOSSIER 给的整个附加损失包的收益只有 +0.11 dB。本轮不碰（`train_rdg.py:20-24`；`E/NOTES.md:92-93`）。
- **s 轴平滑正则**：Stage-1 没有 s 轴，红线在此**空满足**，不算正面证据（`train_rdg.py:25-27`；`E/NOTES.md:95-97`）。CI 用 AST 去掉 docstring 后扫描 `smooth/tv(/total_variation/laplacian`，无命中（`ci_checks_rdg.py:117-133`；`ci_checks_rdg.json :: results[8]`）。去 docstring 这一步是必要的——本仓的写作习惯是在注释里**点名**被禁的东西来解释它为什么不在，朴素子串扫描会把文档写得最好的文件全部误报。

### 6.7 checkpoint 选择（红线：禁用 val loss）

```python
score = r_i["de00_p50"] if r_i["var_ratio"] > 0.30 else 1e9      # train_rdg.py:370
```

即：**用未见源池的 ΔE00 p50 选，方差比 < 0.30 一票否决**（`train_rdg.py:369-375`）。CI 逐字比对源码里这一行是否存在（`ci_checks_rdg.py:135-137`；`ci_checks_rdg.json :: results[9].detail`）。

红线的理由（PLAN 的原话）是「L1 最低 = 最保守平均 LUT」：用 val loss 选点会系统性地选中那个塌陷成一张平均 LUT 的 checkpoint。方差比否决就是专门抓这件事的（§7.1）。

---

## 7. 评测口径（三个坑）

### 7.0 评测器本身

`Evaluator`（`train_rdg.py:69-194`）在两个池上各跑一次，**完全确定性**：

| 项 | 值 | 行 |
|---|---|---|
| 未见源池 `val_img` | S-val 源 × P-train 预设，每次评前 1024 行 | `train_rdg.py:248`（`--eval-n 1024`，见 `run_stage1.sh:36`） |
| 未见 LUT 池 `val_lut` | S-val/S-test 源 × P-val/P-test 预设，前 1024 行 | `train_rdg.py:249` |
| 主口径色 | 每样本 **8192** 个立方体均匀随机色，seed 777+i | `train_rdg.py:79, 108`；`data_rdg.py:109-114` |
| 副口径色 | 每样本 **4096** 个该样本 I_in 的像素色，独立 generator | `train_rdg.py:79, 111-113` |
| 评测频率 | 每 1000 步 | `run_stage1.sh:36` |
| 目标 | LUT bank 的三线性读表 | `train_rdg.py:110`；`data_rdg.py:47-52` |
| 烘焙回读 | 只在 `step == steps` 跑；最终 metrics 重载 best.pt 后用 `bake_max=512` 再跑一次 | `train_rdg.py:358-359, 398-399` |

**每一行都带的两个红线列**（生成器语境下的定义，`E/NOTES.md:104-107`）：

- `Δ_const` = PSNR(真条件) − PSNR(**condition-dropout 学到的那个常量条件**)（`train_rdg.py:130-135, 175-176`）
- `Δ_shuffle` = PSNR(真条件) − PSNR(**跨样本置换条件**，目标不变、只换别人的图)（`train_rdg.py:137-144, 177-178`），置换是固定 seed 777 的一次性 permutation（`train_rdg.py:88-89`）

塌陷成「一张平均 LUT」的生成器在这两列上都是 0。

`var_ratio`（主）= 预测变换在**样本间**的方差 / 目标变换在样本间的方差，在同一组色上算（`train_rdg.py:157-162`）。`var_explained`（副）= `1 − MSE/Var_between`（R² 式，`:163`）。两种读法都预注册并都报（`E/NOTES.md:232-236`，待决策 6）。

> ⚠ **这个 `var_ratio` 与其他实验（如 MCQ）的同名量不是同一个东西**，切勿并排引用。

### 7.1 坑一：REPORT.md 全文没有任何 step-8000 数字

本文自己核过一遍（`rg -n "8000" E/REPORT.md`，13 处命中）：**每一处 8000 都出现在「ETA / 待复核 / 计划」的语境里，没有一处是实测读数**。REPORT 的表格是 step 2000（§4/§5）与 step 4000（§12，自称「对外引用口径」）。

REPORT.md:411-415 承诺过一张「表 B（§13）step 8000 收敛表」，**§13 这一节不存在**（本文核对 REPORT.md 全部标题：1, 1.1–1.3, 2, 3, 4, 4.1–4.5, 5, 表1–表5, 6, 9, 10, 7, 8, 11, 11.1, 12, 12.1, 12.1a, 12.1b, 12.2–12.4，止于 §12.4）。

**因此「9.06 → 2.67」这个头条数字来自 `E/metrics_converged.json`，不是 REPORT**：

- `metrics_converged.json :: rows[0].de00_p50 = 9.057435989379883`（mlp / val_img / step 8000）
- `metrics_converged.json :: rows[8].de00_p50 = 2.6700026988983154`（gbase / val_img / step 8000）

项目级文档自己也登记了这一点：`docs/EXPERIMENT_RESULTS_CURRENT.md:41-42`「RD-G 报告仍混有不同 step 的旧口径，需用 `metrics_converged.json` 重写报告」。

### 7.2 坑二：`metrics_converged.json` 自己有一行过期

`metrics_converged.json :: rows[10].selected_step = 2000`（gbase_free），而 `E/runs/gbase_free/metrics.json :: selected_step = 8000`，`history` 完整到 8000 步。

**根因可以逐步复现**：

1. `E/tools/finalize.sh:17-25` 的等待循环只等 **4 个臂**（`glite gbase mlp mlp_wide`）落 `metrics.json`；
2. 这 4 个臂最晚的是 `runs/gbase/metrics.json`，mtime **18:45:06**，循环随即 break；
3. `finalize.sh:31` 立刻跑 `aggregate.py`，产出 `metrics_converged.json`，mtime **18:45:43**；
4. 而 `runs/gbase_free/metrics.json` 的 mtime 是 **19:38:53**，比聚合文件晚 **53 分钟**；
5. `aggregate.py:44-47` 在 `metrics.json` 不存在时回落到 `metrics_partial.json`，当时那份 partial 的 best 停在 step 2000（`metrics_converged.json :: runs.gbase_free.wall_sec = 2576.6`，与 17:57 重启 + 2576 s = 18:40 的那次 eval 落盘吻合）；
6. `finalize.sh:28` 同一时刻检查 `runs/gbase_free/metrics.json` 是否存在——不存在，所以 **`bake_check_converged.json` 与 `paired_ci_converged.json` 都不含 `gbase_free`**（也不含 `gtiny`，它压根没有 `metrics.json`）；
7. 之后**没有人重跑过 `aggregate.py`**，所以那一行至今是 step 2000。

顺带一提，`finalize.sh:20` 与 `run_stage2_after.sh:20` 都用了 `pgrep -fc` 判活——正是 `CLAUDE.md`「长任务提交纪律」点名禁止的用法。后者的后果见 §8.5。

### 7.3 坑三：加固对照的结论必须降级

`metrics_converged.json :: verdicts._hardening_control.measured_db = 9.886`、`verdict = "confirmed"`。

**这个数字是混口径的，不可引用**：`aggregate.py:198-206` 直接相减两臂的 `val_img.psnr_mean`，而此刻 `gbase` 是 step 8000（30.0244）、`gbase_free` 是 step 2000（20.1379），30.0244 − 20.1379 = 9.886。**拿收敛臂减未收敛臂。**

同 step 的三种合法比法（本文全部复算自 `runs/gbase*/metrics.json` 与其 `history`）：

| 口径 | G-Base（加固） | 无加固自由回归 | 差 | PLAN 判读规则（`PLAN:258`） |
|---|---|---|---|---|
| **同 step 1000**（REPORT §4.3 的口径，`REPORT.md:138-148`） | 20.2309 dB | 17.9943 dB | **+2.2366 dB** | ≥2 dB ⇒ 确认加固 |
| **同 step 8000，未见源** | **30.0244 dB** | **28.3391 dB** | **+1.6853 dB** | 落在 0.5–2 dB 的**灰区**，未达「确认」线 |
| **同 step 8000，未见 LUT** | 23.0151 dB | 22.6966 dB | **+0.3185 dB** | **靠近「<0.5 dB ⇒ 温和全面简化」一侧** |
| 混口径（8000 vs 2000） | 30.0244 | 20.1379 | +9.886 | **不可引用** |

ΔE00 口径同向：同 step 8000 未见源 2.6700 vs 3.0616（加固好 12.79%）、未见 LUT 5.2316 vs 5.4254（好 3.57%）（本文复算）。

**降级后的诚实说法**：加固三件套（μ 锚定 / σ 有界 / **G 初始 0**）在**训练早期**价值很大（step 1000 就是 +2.24 dB，无加固臂那时 ΔE00 11.97 vs 8.45），**到 8000 步收敛后收缩到 +1.69 dB（未见源）/ +0.32 dB（未见 LUT）**。也就是说它买的主要是**训练稳定性与收敛速度**，而不是最终上限。REPORT §4.3 与 §10 里「加固确认（≥2 dB）」这句话，在收敛口径下**不再成立**。

这条不影响任何一条晋级判据（晋级判据是「相对 MLP 基线的降幅」与「方差比」，与加固对照无关）。

### 7.4 顺带一处口径不一致：CI 的 21/21 vs 26/26

`E/REPORT.md:323` 写「`ci_checks_rdg.py`，21/21 PASS」。本文核对：

- `E/config/ci_checks_rdg.py`（交付快照）确实产出 **21** 条检查（本文按 `check(` 调用逐条数）；
- 但落盘的 `E/config/ci_checks_rdg.json` 有 **26** 条、`n_fail = 0`（`jq '.results|length'`）；
- 线上 `model/glut_repro/ci_checks_rdg.py` 比快照多了 5 条 s-cache 域检查（`diff` 显示新增 `:200-247`），而 CI 的输出路径是硬编码到 `config/ci_checks_rdg.json` 的（`ci_checks_rdg.py:204-206`），所以线上版覆盖了快照版的输出。

线上版另外还修了一件事：`np.random.seed(0)` 被补上（新增的 `:30-33`），注释写明「无种子的抽样让 ΔE00 容差检查在连续两次运行里给出 25/26 与 26/26」。这与 `E/NOTES.md:207` 记录一致。

**结论：CI 是 26/26 PASS，REPORT 的「21/21」是旧快照口径。** 交付快照 `config/ci_checks_rdg.py` 与它自己产出的 json **不同源** —— 这是一处交付快照的不一致。

### 7.5 天花板：定义、算法、以及为什么必须自己重算

**定义**（`E/tools/fit_ceiling.py:8-14`）：生成器把图像对映射到 `23N+12` 个原始 head 输出 `z`。天花板 = 把**同一批 z 从「预测」改成「逐 LUT 自由优化」** —— 同一套 `ParamHead`、同一个 `render`、同一套 anchor、同一套色彩口径、同一份 ΔE00 代码。即一个容量无穷、无泛化负担的 oracle 生成器。

> **天花板拿不到的是渲染器容量，臂拿不到的才是生成器容量。**

**算法**（`fit_ceiling.py:46-108`）：

1. `z_prim` 初始化为 0，`z_glob` 用**闭式最小二乘热启动**：对 20,000 个随机色做 `lstsq` 拟合 (G,c)，再按 head 的 `G = 0.1z` 反推回 z（`:60-67`）。热启动只会**抬高**天花板，因此只会让臂的判定更保守（`:56-59` 的注释）。
2. Adam 直接优化 `(z_prim, z_glob)`，lr 3e-2、余弦退火到 1%、**8000 步**、每步 4096 个随机色、L1 损失（`:71-85`）。
3. 评测：16,384 个 seed-777 的均匀色，同一份 `delta_e00`（`:92-103`）。
4. LUT 选择来自 manifest 的 P-split，**不在此处重新切分**；用 `rng.choice` 无放回后排序，不取前 n 个（`:135-148`，对应 G3 的教训 `E/NOTES.md:103`）。

**实测**（`E/runs/ceiling_kmeans/ceiling.json`，等价键路径 `metrics_converged.json :: ceiling.*`）：

| 版本 | anchor | P-train p50 | p90 | p99 | PSNR | P-val p50 | PSNR | alive |
|---|---|---|---|---|---|---|---|---|
| `runs/ceiling_kmeans`（**正式**，与各臂同 anchor） | Stage-0 k-means | **0.5191** | 0.8714 | 1.2447 | **43.6504** | **0.5205** | 43.6896 | 0.6897 |
| `runs/ceiling`（敏感性对照） | GLUT A.1 均匀网格 | 0.5488 | 0.8867 | 1.1709 | 43.4663 | 0.5612 | 43.3415 | 0.6113 |
| *恒等基线* | — | 14.0780 | — | — | 14.8161 | 13.6661 | 15.1932 | — |

LUT 数：P-train **384**、P-val/test **350**（`ceiling.json :: p_train.n_luts / p_val.n_luts`）。

三点诚实说明（`E/REPORT.md:101-108`）：

1. 本实验的天花板比 E1 的逐 LUT 直接拟合差约 11%（0.5191 vs E1 的 0.467，`experiments/E1_cube_N_20260803/REVIEW-result.md:20`）。原因是**参数化不同**：本实验的 μ 被 anchor±1/3 约束、σ 被夹在 [0.02,0.50]（这是 PLAN 要求的加固），E1 是无约束自由拟合，且每 LUT 只有 1068 个参数（`22N+12`，无存在门）。**必须用本实验这版**，否则臂与天花板不同参数化，「天花板占比」没有意义。
2. **天花板偏低会让「MLP 已达 90%」这条死刑判据更容易触发** —— 即天花板的选择偏向**淘汰**方向，对本实验想证明的结论是保守的。
3. `alive_frac` 0.69 低于 E1 的 0.8 门，但两处 alive 定义不同（本实验：`o·g > 0.05`；E1：概率质量份额），不能直接比。

**天花板占比的算法**（`aggregate.py:127-140`）：

```
ceiling_frac_db  = (PSNR_arm − PSNR_identity_同池) / (PSNR_ceiling − PSNR_identity_同池)
ceiling_frac_de00 = (ΔE00_identity_同池 − ΔE00_arm) / (ΔE00_identity_同池 − ΔE00_ceiling)
```

**一处口径细节要说明**：分子里的 `PSNR_identity` 取自**臂自己的评测池**（1024 张图 × 8192 色，`train_rdg.py:124`），分母里的 `PSNR_ceiling` 取自**天花板自己的 LUT 集合**（384/350 个 LUT × 16384 色）。两者不是同一批样本。`val_img` 用 `ceiling.p_train`、`val_lut` 用 `ceiling.p_val` 配对（`aggregate.py:128`），P-split 对齐，但样本集不同。**这个口径差带来的偏差量 —— 缺**（未做敏感性分析）。

### 7.6 烘焙回读：33³ 导出 + 四面体

红线「烘焙一致性从第一天当一等指标」的落实（`E/tools/bake_check.py`；训练内的版本 `train_rdg.py:147-152`）：

1. `bake_cube(p, size)` 在 `size³` 的均匀格上求值生成的渲染器，clamp 到 [0,1]（`model_rdg.py:543-562`）；
2. 用 **`tetra_lookup`（四面体插值）** 把这张表读回来（`model_rdg.py:565-606`），**而不是**训练时用的三线性 —— 因为 `.cube` 宿主用四面体（PLAN §1.6）；
3. 报两个数：`bake_vs_direct`（纯导出损失）与 `bake_vs_gt`（.cube 消费者实际看到的）。

两条独立的一致性钉子：

- `tetra_lookup` vs colour-science 的 `table_interpolation_tetrahedral`：**1.19e-7**（`ci_checks_rdg.json :: results[5].detail`）
- `delta_e00` vs colour-science 的 `delta_E_CIE2000`：max abs **0.01465**、相对 **3.28e-4**（`results[6].detail`，判据是相对误差 <1e-3）

评测里也有一处形式与实质的错位要说明：训练侧的 `L_cube` 目标用**三线性**读 33³ 表（`data_rdg.py:7-12`），理由是三线性是生产渲染器自己的插值器，F5 实测 tetra 与 tri 的中位差只有 0.009 ΔE00，低于归档 JPEG 底噪（`E/NOTES.md:79-81`）。

---

## 8. 结果与边界

### 8.1 收敛口径主表

数据源：`E/runs/<arm>/metrics.json`（`gtiny` 只有 `metrics_partial.json`）。色彩口径 = 立方体均匀色 8192 点。

**未见源（`val_img`，S-val 源 × P-train 预设）**：

| 臂 | 生成器参数 | step | ΔE00 p50 | p90 | p99 | 自然色 p50 | PSNR | vs MLP 降幅 | 方差比 | Δ_const | Δ_shuffle | 死原语 | 天花板占比(dB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `mlp` | 251,292 | 8000 | 9.0574 | 15.2502 | 21.7080 | 3.7934 | 19.5627 | — | 0.8473 | 4.60 | 6.10 | 4.00% | **20.0%** |
| `gtiny` | 1,561,445 | **4000** ⚠ | 5.9680 | 10.5812 | 15.5316 | 2.0668 | 22.8675 | 34.11% | 0.8971 | 7.92 | 9.81 | 0.95% | 31.0% |
| `mlp_wide` | 4,933,244 | 8000 | 7.1161 | 13.4220 | 19.1722 | 2.8717 | 21.7201 | 21.43% | 0.8827 | 6.76 | 8.53 | 1.67% | 27.2% |
| **`glite`** | 4,447,846 | 8000 | **3.6097** | 7.2998 | 12.8300 | 1.3875 | 27.2762 | **60.15%** | 0.9505 | 12.30 | 14.64 | 1.78% | **45.6%** |
| **`gbase`** | 14,056,806 | 8000 | **2.6700** | 5.4246 | 11.5641 | 1.1726 | **30.0244** | **70.52%** | **0.9668** | 15.06 | 17.48 | 2.23% | **54.8%** |
| `gbase_free` | 14,056,806 | 8000 | 3.0616 | 6.6291 | 13.1288 | 1.3982 | 28.3391 | 66.20% | 0.9600 | 13.36 | 15.76 | 0（见 §5.5） | 49.2% |
| *天花板* | — | — | 0.5191 | 0.8714 | 1.2447 | — | 43.6504 | — | — | — | — | — | 100% |
| *恒等* | 0 | — | 17.8469 | — | — | — | 13.5298 | — | 0 | 0 | 0 | — | 0% |

**未见 LUT（`val_lut`，S-val/test 源 × P-val/test 预设）**：

| 臂 | ΔE00 p50 | p90 | 自然色 p50 | PSNR | vs MLP 降幅 | 方差比 | Δ_const | Δ_shuffle | 天花板占比(dB) |
|---|---|---|---|---|---|---|---|---|---|
| `mlp` | 9.0500 | 14.4929 | 3.7662 | 19.0133 | — | 0.8519 | 3.43 | 4.80 | 14.8% |
| `gtiny`(4000) | 6.5319 | 10.8706 | 1.9554 | 21.4832 | 27.82% | 0.8886 | 5.78 | 7.60 | 23.4% |
| `mlp_wide` | 8.4220 | 13.5103 | 3.2939 | 19.5756 | **6.94%** | 0.8724 | 4.00 | 5.50 | 16.8% |
| **`glite`** | 5.5340 | 10.6343 | 1.5404 | 22.5908 | **38.85%** | 0.9369 | 7.00 | 9.00 | 27.2% |
| **`gbase`** | 5.2316 | 11.3914 | 1.4931 | 23.0151 | **42.19%** | 0.9534 | 7.46 | 9.51 | 28.6% |
| `gbase_free` | 5.4254 | 11.0150 | 1.7244 | 22.6966 | 40.05% | 0.9496 | 7.19 | 9.18 | 27.5% |
| *天花板* | 0.5205 | 0.8891 | — | 43.6896 | — | — | — | — | 100% |

（降幅、天花板占比、PSNR 增益均为本文按 `aggregate.py:124-140` 的公式复算，与 `metrics_converged.json :: rows[*]` 对上，`gbase_free` 除外——那一行是过期口径。）

### 8.2 训练轨迹（这张表回答「是不是只是大模型收敛快」）

`val_img` ΔE00 p50，逐 step 读自 `runs/<arm>/metrics.json :: history[*].val_img.de00_p50`：

| step | mlp | mlp_wide | gtiny | glite | gbase | gbase_free |
|---|---|---|---|---|---|---|
| 1000 | 11.3034 | 11.0428 | 9.4399 | 8.8066 | 8.4523 | 11.9690 |
| 2000 | 10.5475 | 10.2801 | 7.4710 | 7.1099 | 6.0753 | 8.9838 |
| 3000 | 10.0144 | 9.0839 | 6.5764 | 5.4817 | 4.6360 | 6.3542 |
| 4000 | 9.5792 | 8.1737 | **5.9680（止）** | 4.6763 | 3.8822 | 5.1314 |
| 5000 | 9.3973 | 7.8230 | — | 4.2319 | 3.2946 | 4.2745 |
| 6000 | 9.1021 | 7.3556 | — | 3.8959 | 2.9063 | 3.4341 |
| 7000 | 9.0745 | 7.1461 | — | 3.6950 | 2.7095 | 3.1763 |
| **8000** | **9.0574** | **7.1161** | — | **3.6097** | **2.6700** | **3.0616** |

**MLP 族在 step 6000 后基本走平**（`mlp` 9.1021 → 9.0574，四步只降 0.49%），**transformer 族还在降**（`glite` 3.8959 → 3.6097 降 7.3%；`gbase` 2.9063 → 2.6700 降 8.1%）（本文复算）。所以「大模型只是收敛快」这个替代解释在收敛口径下不成立。

### 8.3 派生结论

| 结论 | 数字 | 算法 |
|---|---|---|
| **同参数量下架构差距**（关掉「是不是只是参数多了」的那句） | `glite` 4.45M **27.2762 dB** vs `mlp_wide` 4.93M **21.7201 dB** = **+5.5560 dB / ΔE00 −49.27%**；未见 LUT 档 +3.0152 dB / −34.29% | 直接相减（本文复算） |
| 族内斜率（未见源，dB/decade） | MLP 族 0.25M→4.93M **+1.669**；transformer 族 4.45M→14.06M **+5.499** | `ΔPSNR / log10(参数比)`（本文复算） |
| 天花板占比 | MLP **20.0%** → G-Base **54.8%**，仍有 **45.2%** 未取 | `aggregate.py:136`；`verdicts._kill_switch.fired = false` |
| **淘汰条款「MLP 已达天花板 90%」** | 实测 **20.03%**，**未触发** | `metrics_converged.json :: verdicts._kill_switch.measured_ceiling_frac_db = 0.2003` |

> **`gtiny` → `glite` 那一段斜率不要算**：`gtiny` 停在 step 4000、`glite` 是 8000，混口径。本文只给同 step 8000 的 `glite → gbase` 段。

### 8.4 配对 bootstrap 与烘焙

**配对 bootstrap**（`E/paired_ci_converged.json`；5000 次重抽，n=512，量 = `MLP p50 − 该臂 p50`，正值 = 该臂更好；工具 `E/tools/paired_ci.py`，逐样本 ΔE00 在**同一批行、同一顺序、同一组色**上重算）：

| 臂 | val_img 点估计 [CI95] | val_lut 点估计 [CI95] | P(该臂更好) |
|---|---|---|---|
| `mlp_wide` | +1.9987 [+1.5094, +2.4502] | +0.7533 [+0.4313, +1.0710] | 1.000 |
| `glite` | +5.5176 [+5.0731, +6.0028] | +3.5151 [+3.1491, +3.9514] | 1.000 |
| `gbase` | +6.4912 [+6.0559, +7.0054] | +3.8315 [+3.3280, +4.2636] | 1.000 |

六个 CI 全部不跨 0。**`gtiny` 与 `gbase_free` 的收敛口径 CI —— 缺**（§7.2 的第 6 步）。

**烘焙预算**（`E/bake_check_converged.json`，n=256，val_img，8192 色，四臂 step 8000）：

| 臂 | 直接 ΔE00 p50 | 17³ vs 直接 | **33³ vs 直接 p50** | 33³ p99 | 33³ max | 65³ p50 | 33³ 净代价 | 门 ΔE<2 |
|---|---|---|---|---|---|---|---|---|
| `mlp` | 9.7757 | 0.0840 | **0.0213** | 0.0554 | 0.0655 | 0.0055 | +0.0017 | PASS |
| `mlp_wide` | 7.6440 | 0.0955 | **0.0249** | 0.0593 | 0.0646 | 0.0063 | +0.0008 | PASS |
| `glite` | 3.9066 | 0.1141 | **0.0296** | 0.0669 | 0.0683 | 0.0075 | −0.0060 | PASS |
| `gbase` | 2.9430 | 0.1204 | **0.0312** | 0.0757 | 0.0835 | 0.0079 | +0.0003 | PASS |

「交付物是标准 3D LUT」这条叙事在数值上成立：33³ 导出 + 四面体回读的代价 p50 ≤ 0.031、max ≤ 0.084，净代价 ≈ 0。**注意 33³ 绝对误差随容量单调上升**（0.0213 → 0.0312）——模型越强、cube 越不平坦——但仍比 JND(≈1) 小一个半数量级。

**缺的部分**：`gtiny` 的烘焙数字（任何格点、任何 step，其 `metrics_partial.json :: val_img.bake_de00_vs_direct_p50 = null`）；`gbase_free` 的 17³/65³ 档（`bake_check_converged.json` 不含该臂；它自己的 `metrics.json` 只有 33³ 一档：`val_img.bake_de00_vs_direct_p50 = 0.0204`）。另外 `train_rdg.py:192` 把 `bake_psnr_penalty_db` 硬写成 `None` —— **该字段全实验为缺**。

### 8.5 Stage-2（有 s 的那一半）：一个数字都没有

**`E/runs2/` 目录不存在**（本文实测 `ls`：`No such file or directory`）。

代码是就绪的：`train_rdg2.py` + `ParamHead4D`（`model_rdg.py:221-259`）+ `render4d`（`:120-155`），并且过了 CI：

| 检查 | 实测 | 出处 |
|---|---|---|
| `render4d` vs `model4d_naive.GLUT4D(anchored)` | 1.19e-7 | `ci_checks_rdg.json :: results[16].detail` |
| 4D 零初始化恒等（17³ × 5 个 s 值） | **1.1005e-3**，判据写的是 <1e-2 **PASS**，但**超 PLAN 字面的 1e-4 阈值 11 倍** | `results[17].detail` |
| `mu_s` 是冻结的 K=6 buffer，不是生成量 | std = 0.3452，`requires_grad=False` | `results[18]` |
| `sigma_s` 有界 [0.025, 0.30] | [0.0250, 0.3000] | `results[19].detail` |

**为什么一个数字都没有 —— 根因可逐行定位**：调度脚本 `E/tools/run_stage2_after.sh` 第 20 行

```bash
live=$(pgrep -fc "train_rdg --arm" 2>/dev/null || echo 0)
```

`pgrep -fc` 在无匹配时**既打印 `0` 又返回非零退出码**，于是 `|| echo 0` 再打一个 `0`，`$live` 变成两行 `"0\n0"`；第 23 行的 `[ "$live" -eq 0 ]` 因此报 `integer expression expected` 并永远为假。同时第 16-19 行等的是 **6 个** `metrics.json`，而 `gtiny` 永远不会有（停在 step 4000），所以 `n` 卡在 **5/6**。

实测：`logs/stage2_chain.log` 有 **11,059** 行，最后一条时间戳 **2026-08-05T16:56:28**，内容是 `stage1 done=5/6 live=0` + 两行脚本报错；调度进程 **PID 379823 仍在运行，已空转 2 天 00:33:59**（本文实测 `ps -eo pid,etime,cmd`）。

（另有一条更早的证据：`logs/s2_mlp.log` 只有 3 行，是与 Stage-1 同跑时的一次试跑，实测 **0.31 it/s**，随后改为串行等待——这就是 `run_stage2_after.sh` 取代 `run_stage2.sh` 的原因，`job.marker` 的 `[stage2_after]` 块有记录。）

**后果（必须主动交代）**：

> **「transformer 生成器的优势在有 s 轴时是否延续」，本实验完全没有测。** RD-G 的全部结论都限定在 Stage-1 的全局 cube 任务上。

项目级登记同此：`docs/EXPERIMENT_REGISTRY.md:36`「有 s 的 Stage-2 尚无审阅结论」。

### 8.6 其他边界

| 项 | 状态 | 出处 |
|---|---|---|
| `gtiny` 只到 step 4000 | 卡 0 余量跌破 20 GB 硬底线，作者停掉自己优先级最低的一臂；**不降 batch 换空间**（混口径的点比缺点更糟）。实测：日志末行是 `5000/8000` 的训练打印，`EVAL 5000` 从未出现 ⇒ 在 step-5000 的 eval 中途被停 | `E/REPORT.md:417-422`；`logs/gtiny.log` 尾部 |
| **泛化间隙随容量放大** | 收敛后 `gbase` 未见源 2.6700 vs 未见 LUT 5.2316（**+95.9%**）；`glite` +53.3%；`mlp` **−0.1%**（未见 LUT 反而略好）。**容量买到的收益里有一部分是「记住训练 preset」** | 本文复算自 §8.1 两表 |
| 两个色彩口径差距 | `gbase` 未见 LUT：自然色 1.4931 vs 均匀色 5.2316（**3.50×**）。cube 全域指标显著低估自然图观感 | `runs/gbase/metrics.json :: val_lut.*` |
| PLAN Stage-1 的收敛门全部未达 | p50<1.5 / p90<3.0：最好的 `gbase` 未见 LUT 是 5.2316 / 11.3914 | `metrics_converged.json :: verdicts.gbase.plan_stage1_cube_p50_lt_1p5` |
| 4D 零初始化恒等超阈值 | 3D 档 8.87e-5 过 1e-4；4D 档 1.10e-3，超 11 倍（ε=1e-6 在立方体远角占比变大）。ΔE00 1e-3 感知上是零（JND≈1），但如实登记 | `ci_checks_rdg.json :: results[17]` |
| 未做 | after-only 单流档、文本条件档、`L_param`/`L_prequery`、register token 开关、三层 routing vs 末层、ModLN 屏蔽、稠密 cross-attn 屏蔽 | `E/REPORT.md:293`；`E/NOTES.md` §6；`PLAN:355` |
| `mlp_wide` 的可视化是 step-4000 的 | `viz/manifest_mlp_wide.json` mtime 08-03 16:32，收敛后的 viz 只重跑了 `gbase/mlp/glite`（`finalize.sh:42`）—— **`mlp_wide` / `gtiny` / `gbase_free` 的收敛口径 viz 缺** | 本文实测 `ls -l viz/` |

### 8.7 失败案例的选法与归因

选样规则是**排名，不是眼选**（`E/tools/make_viz.py:8-10, 112-120`）：对每个验证池按逐样本 cube ΔE00 排序，取最好的 2 张当 success、最差的 2–3 张当 failure；`per_sample.npz` 缺失时当场重算（`:89-109`）。「找不到失败案例」在这套流程里不可表达。

面板列（`make_viz.py:147-149`）：`I_in | after=GT | f_θ(I_in) | 逐像素 ΔE00 图 | GT cube 切片 b=0.5 | 预测 cube 切片 b=0.5`。

`gbase` 收敛后的最差三例（`viz/manifest_gbase.json`）：cube ΔE00 分别 **17.82 / 20.98 / 22.09**（preset `quandian_003361 / 003358 / 005563`），最好两例 **0.685 / 0.764**。注意同样这三张的**烘焙误差全在 0.022–0.043** —— 即失败不是烘焙造成的，是生成器本身没拟合上那些 preset。

---

## 9. 复现入口

### 9.1 环境（`E/config/env.txt`）

```
python  /home/bc/miniconda3/bin/python  3.13.5
torch   2.6.0+cu124   cuda 12.4
numpy 2.4.6 | colour-science 0.4.7 | scikit-learn 1.7.2 | pillow 12.2.0
gpu     2 × NVIDIA H100 (97,871 MiB)
git     bc888fe147af76f2f8ef0299c85de856b9cd4adf  branch lens-exp  (dirty 72 files)
seed    global 20260803 | eval colours 777
```

（`E/NOTES.md:4` 另记了同一套依赖版本，但没记 python 版本；python 版本以 `env.txt` 为准。`git_dirty_files=72` 说明快照时工作区不干净，严格复现需以 `E/config/` 的快照文件为准，而不是当前 HEAD。）

### 9.2 数据

| 用途 | 数据集 | 规模 | split | 出处 |
|---|---|---|---|---|
| Stage-1 训练 | **D-RENDER**（g 线，build `prod-g1/g2/g3-global25k`） | **320,000** 对 | S-train × P-train | `build_cache.py:49`（`N_TRAIN`）；`E/config/manifest_stage1_summary.json :: builds` |
| 未见源验证 `val_img` | D-RENDER | **6,000** 对（每次评前 1024） | S-val × P-train | `build_cache.py:50`（`N_VAL_IMG`） |
| 未见 LUT 验证 `val_lut` | D-RENDER | **4,690** 对（每次评前 1024） | S-val/test × P-val/test | 全取（cap = 1e9，`build_cache.py:67`）；总行数反推 330,690 − 320,000 − 6,000 |
| `L_cube` 目标 | D-CUBE 的 33³ 重采样 | **3,513** 个 preset | 跟 P-split | `data_rdg.py:29`（`/var/cache/veradata/dcube/npy33`）；`cache_report.json :: n_presets` |
| 天花板 / anchor | D-CUBE | P-train **384** + P-val/test **350** | 只读 P-split | `ceiling.json :: p_train.n_luts / p_val.n_luts` |
| Stage-2（未跑） | D-CONSTRUCT | train 1,600 / val 192 | S-train / S-val | `train_rdg2.py:57-59`；**无结果** |

manifest 阶段实测：**600,000** 行入册，`no_split_src=0 / no_npy33=0 / not_lut=0 / no_split_preset=0`（`E/config/manifest_stage1_summary.json :: stats`）。缓存实测：**330,690 行 / 72,587 组 / R=128 / bad=0**（`E/config/cache_report.json`）。

`winner_confidence` **不筛**（normal 180,440 / low 114,288 / abstain 180,168 / null 125,104，`manifest_stage1_summary.json :: confidence`）—— 这是 `DATA_ASSIGNMENT §1.2` 对渲染器预训练开的明文例外。

**为什么只取 g 线**：g1/g2/g3 全量扫描实测候选 100% 是 `format="lut"` + `render_mode="global"`，after 图 = 整幅套 preset 的 `.cube`，`recipe.preset` 才是合法的 `L_cube` 目标；l 线是掩膜合成图，拿来当 cube 监督会把目标系统性拉回恒等（`E/NOTES.md:72-74`）。

### 9.3 缓存构建

```bash
python experiments/RDG_transformer_20260803/tools/build_cache.py --workers 48
```

产物（`E/cache/`）：`imgs_after.u8` (330,690, 128,128,3) uint8 ≈ **16.25 GB**、`imgs_in.u8` (72,587, 128,128,3) ≈ **3.57 GB**、`index.npz`、`rows.jsonl`、`manifest_stage1.jsonl`、`cache_report.json`。

分辨率口径：`ImageOps.exif_transpose` → 单次 `Image.resize((128,128), BOX)`（面积平均，保色彩统计），并用 `draft()` 让 libjpeg 在 DCT 域先做 2×/4× 降采样（`build_cache.py:251-261`）。宽高比是**压扁**不是裁剪，理由是裁剪会丢掉色彩质量（`:18-20`）。

**必须做的一步（否则跑不动）**：把 `cache/` 整个复制进 **`/dev/shm`（tmpfs）**，训练用 `--cache /dev/shm/rdg_cache`（`run_stage1.sh:24`；`env.txt :: cache`）。原因见 §9.6。

### 9.4 天花板 + Stage-0 anchor

```bash
# 第一遍：均匀网格 anchor（同时产出 Stage-0 k-means anchors.npy）
python experiments/RDG_transformer_20260803/tools/fit_ceiling.py \
    --device cuda:0 --mem-frac 0.20 --steps 8000 --chunk 128
# 把 runs/ceiling/anchors.npy 拷成 config/anchors_stage0_kmeans.npy，再跑正式天花板
python -u experiments/RDG_transformer_20260803/tools/fit_ceiling.py \
    --device cuda:1 --mem-frac 0.10 --steps 8000 --chunk 128 \
    --anchors <config/anchors_stage0_kmeans.npy> --out runs/ceiling_kmeans
```
（命令逐字来自 `E/job.marker` 的 `[ceiling]` / `[ceiling_kmeans]` 块。）

耗时（`job.marker` 的 `started` 与产物 mtime 之差，本文复算）：`ceiling` 14:52:56 → 15:23:27 = **30 分 31 秒**；`ceiling_kmeans` 15:25:11 → 16:04:17 = **39 分 6 秒**。显存上限分别是 `mem_frac` 0.20 / 0.10（≈19.4 / 9.7 GB），**实际峰值 —— 缺**。

### 9.5 Stage-1 六臂

```bash
bash experiments/RDG_transformer_20260803/tools/run_stage1.sh
```

展开后每臂的命令（`run_stage1.sh:34-37`，实录见 `job.marker`）：

```bash
python -u -m model.glut_repro.train_rdg \
  --arm <mlp|mlp_wide|gtiny|glite|gbase> --tag <name> \
  --device cuda:<0|1> --mem-frac 0.22 --steps 8000 --bs 768 \
  --p-uni 2048 --p-nat 2048 --p-aux 1024 \
  --eval-every 1000 --eval-n 1024 --workers 3 \
  --cache /dev/shm/rdg_cache [--free]
```

排卡：GPU0 = `glite / mlp / gtiny`，GPU1 = `gbase / mlp_wide / gbase_free`（`run_stage1.sh:53-60`）。每进程 `set_per_process_memory_fraction(0.22)` ≈ 21.3 GB 上限，越界先 OOM 的是自己（`train_rdg.py:240`）。**三臂同卡是有意的**：步是启动开销主导而非 FLOP 主导（实测 gbase 在 B=256 时 188 ms/step、B=1024 时 359 ms/step，4 倍工作量只用 1.9 倍时间），共驻能把空闲 SM 换成吞吐（`run_stage1.sh:5-8`）。

**逐臂实测耗时 / 吞吐 / 显存**（`runs/<arm>/metrics.json :: wall_sec / peak_mem_gb / data_wait_frac`；it/s 取各自日志末行）：

| 臂 | wall_sec | 折合 | it/s（末行） | `torch` 峰值显存 | data_wait |
|---|---|---|---|---|---|
| `mlp` | 6,609.5 | 1 h 50 m | 1.21 | 11.22 GB | 7.7% |
| `mlp_wide` | 6,522.1 | 1 h 49 m | 1.23 | 11.29 GB | 7.1% |
| `gtiny`（止于 5000 步中途） | 4,646.7 | 1 h 17 m | 0.93 | 13.78 GB | 1.2% |
| `glite` | 7,495.9 | 2 h 05 m | 1.07 | 16.55 GB | 1.0% |
| `gbase` | 9,593.7 | 2 h 40 m | 0.89 | 19.26 GB | 1.0% |
| `gbase_free` | 6,163.8 | 1 h 43 m | 1.30 | 19.26 GB | 0.7% |

（这些 wall_sec 是**六臂共驻**下的墙钟，不是独占单卡的时间。`torch.cuda.max_memory_allocated` 的自报数**比 `nvidia-smi` 的真实占用少约 12.5 GB/三臂**，因为不含 CUDA context 与缓存分配器保留量——作者在 `E/STATUS.md:123-127` 记过这次记账错误。逐进程实测占用见 `STATUS.md:104-107`：约 14.4–21.4 GB/进程。）

### 9.6 数据加载：这台机器上必须知道的两堵墙

这两条是本实验最可复用的基建结论（`E/NOTES.md:113-167`；`E/STATUS.md:50-76`）：

**墙一：NFS 往返延迟。** 原始 D-RENDER 路径（每样本一次 NFS tar ranged read ≈350 KB + 一次短边 1024 JPEG 解码），56 进程并行实测 **125–140 图/s**，而 56 个解码 worker 稳定停在 **3.3–3.5% CPU** ⇒ 瓶颈是 I/O 等待，不是解码、不是 CPU。折算 ≈48 MB/s。训练侧一张 H100 在本配方下要 ≈1,700–2,300 样本/s ⇒ 原始路径慢 **12–18 倍**。

**墙二：本地机械盘随机读（更隐蔽）。** 缓存建好后第一次启动六臂，**13 分钟一条训练步日志都没有**，显存却已涨到每卡 70 GB。`nvidia-smi` 的整卡 util 被同卡其他作业顶到 99%，完全看不出问题；必须用 **`nvidia-smi pmon -s um` 看逐进程 SM** 才看到自己的进程是 **0%**。`iostat -x` 定位到 `/home` 所在的 sdb 是机械盘：`%util 99.2 / r_await 120 ms / 284 IOPS / 21 MB/s`。6 个进程各自随机读 19.8 GB 缓存、页缓存装不下（125 GB RAM 已被其他 agent 占 62 GB），每个样本退化成一次 120 ms 寻道；训练需要 6×768×98 KB ≈ 460 MB/step，按 21 MB/s 算 **22 秒/步**，与实测吻合。

**处置与效果**（`E/NOTES.md:152-158`）：

| | `/home`（机械盘） | `/dev/shm`（tmpfs） |
|---|---|---|
| 我的进程逐进程 SM | **0%** | 12–49%/进程，两卡合计 99–100% |
| `data_wait` 占比 | ≈100% | **0.7–7.7%**（实测见 §9.5 表） |
| 吞吐 | <0.07 it/s | **0.89–1.30 it/s** |

> **可复用建议：在这台机器上，离线缓存必须落 tmpfs 或 NVMe，不能落 `/home`；诊断 GPU 空转必须用 `nvidia-smi pmon -s um` 看逐进程 SM。**

顺带一条工具缺陷（对所有臂有效）：`tools/bgr_check/common.py::BankResolver` 对 `ppr10k / raise6k / fivek_gold` 三个 bank 静默解析失败（它们的 member 后缀是角色限定的 `.source.png` / `.preview.jpg` / `.before.jpg`，参考实现只认裸 `.jpg/.png`），实测会丢掉 **20% 的组（14,422/72,587）**。本实验改成按 metadata 自己的 `member` 字段精确匹配后 **72,587/72,587 = 100% 解析**（`build_cache.py:130-138`；`E/STATUS.md:40-45`）。**凡用过 `BankResolver` 的臂都应复核自己的源覆盖率。**

### 9.7 聚合、烘焙、CI、可视化

```bash
# 收敛口径聚合（不带 --at-step 就读每臂自己的 best）
python experiments/RDG_transformer_20260803/tools/aggregate.py \
    --out experiments/RDG_transformer_20260803/metrics_converged.json
# 共同步口径（例：全部读 step 4000）
python .../tools/aggregate.py --at-step 4000 --out .../metrics_step4000.json
# 烘焙预算
python -u .../tools/bake_check.py --runs mlp mlp_wide glite gbase gbase_free \
    --device cuda:0 --mem-frac 0.10 --n 256 --out .../bake_check_converged.json
# 配对 bootstrap
python -u .../tools/paired_ci.py --arms mlp mlp_wide glite gbase gbase_free \
    --device cuda:1 --mem-frac 0.10 --n 512 --out .../paired_ci_converged.json
# 红线 CI（26 条）
python -m model.glut_repro.ci_checks_rdg cuda:0
# 可视化
python -u .../tools/make_viz.py --run gbase --device cuda:0 --mem-frac 0.08
```

**要修复本文 §7.2 的过期行，只需要重跑第一条 `aggregate.py`**（`gbase_free/metrics.json` 现在已经在位），并把 `bake_check.py` / `paired_ci.py` 的 `--runs/--arms` 补上 `gtiny gbase_free` 重跑一次。`finalize.sh:16-25` 的等待条件与第 20 行的 `pgrep -fc` 应一并修掉。

---

## 附录 A：本文追不到的数字清单（「缺」汇总，共 22 处）

| # | 缺什么 | 为什么缺 | 正文 |
|---|---|---|---|
| 1 | **Stage-2（4D、有 s 轴）的全部实测数字** | `runs2/` 不存在；调度脚本卡死 | §8.5 |
| 2 | `gtiny` 的 step-8000 全部数字 | 停在 step 4000 | §8.1/§8.6 |
| 3 | `gtiny` 的烘焙数字（任何格点、任何 step） | `metrics_partial.json` 的 bake 字段为 `null` | §8.4 |
| 4 | `gbase_free` 的 17³/65³ 烘焙数字 | `bake_check_converged.json` 不含该臂 | §7.2 第 6 步 |
| 5 | `gtiny` / `gbase_free` 的收敛口径配对 bootstrap CI | 同上 | §8.4 |
| 6 | `bake_psnr_penalty_db` 字段（全实验） | `train_rdg.py:192` 硬写 `None` | §8.4 |
| 7 | μ 锚定 / σ 有界 / G 初始 0 的**逐条**消融 | `--free` 三件一起改 | §3.4 |
| 8 | `o` 与 `g` 冗余的定量代价 | 未做（NOTES 待决策 4） | §5.5 |
| 9 | `mlp_wide` 的 `width=880` 是怎么搜出来的 | 源码只给结果 | §3.2 |
| 10 | 「GRM: sigmoid vs exp = +3.08 dB」的原始论文核实 | PLAN 内部引用，本文未打开外部来源 | §5.2 |
| 11 | 各臂训练结束时的 `route_logit` 实际取值 / routing 权重分布 | 未落盘，只在 `best.pt` 的 state_dict 里 | §4.5 |
| 12 | register token 开关、三层 routing vs 末层、ModLN 屏蔽、稠密 cross-attn 屏蔽的消融 | PLAN §5 列了，本实验未做 | §4.2/§4.5/§8.6 |
| 13 | ModLN「可插值性」的实测验证 | 未做 | §4.4 |
| 14 | after-only 单流条件档、文本条件档 | 待决策 #1，未跑 | §4.7 |
| 15 | `L_param` / `L_prequery` 的对照 | 目标当时不存在；现已具备 | §6.5 |
| 16 | 天花板两次运行的实际显存峰值 | `ceiling.json` 只记 `mem_frac` 上限 | §9.4 |
| 17 | 天花板占比里「臂的 identity 池」与「天花板的 LUT 集」不同源带来的偏差量 | 未做敏感性分析 | §7.5 末 |
| 18 | **每 tap aux 损失 + routing 熵项的单独消融** | MLP 族天然没有这两项，两族并非逐项同配方，但代价未量化 | §3.1 |
| 19 | 「位置编码为什么必须 key-only，而不是两侧都加」的实验依据 | 只有 PLAN 的规格文字，本实验未做该消融 | §4.3 |
| 20 | `M/b/off/G/c` 统一乘 `0.1` 这个系数的来源 | PLAN 逐字，无更深推导 | §5.4 |
| 21 | `o = sigmoid(z−2)`、`g = sigmoid(z+4)` 里 −2 / +4 的来源 | 同上 | §5.5 |
| 22 | `mlp_wide` / `gtiny` / `gbase_free` 的**收敛口径**可视化面板 | `finalize.sh:42` 只重跑了 `gbase/mlp/glite` | §8.6 |

## 附录 B：本文自己重跑过的项（复算记录）

| 项 | 方法 | 结果 |
|---|---|---|
| 五臂逐层参数量 | CPU 实例化 `RDGModel(arm, 48)`，逐子模块 `numel()` 求和 | 与 `ci_checks_rdg.json :: results[15].detail.counts` **逐位相同** |
| 零初始化恒等（**用训练实际的 k-means anchor**） | `ParamHead(48, anchors=config/anchors_stage0_kmeans.npy)`，33³ 全格点 | max ΔE00 **8.732e-05 < 1e-4**（均匀网格档 8.713e-05） |
| `f(x)=2x` 反模式 | 零初始化下比较加固档与 `--free` 档 | 加固：`f(x)/x` 中位 1.0000、max\|f−x\|=0.0000；free：中位 **2.0000**、max\|f−2x\|=**0.0000**、与恒等的 max ΔE00 = **43.08** |
| anchor 文件同一性 | `md5sum` | `runs/ceiling/anchors.npy` == `config/anchors_stage0_kmeans.npy`（`578140f0…`）；`runs/ceiling_kmeans/anchors.npy` 不同且无人消费 |
| 交付快照 vs 线上模块 | `diff -q` | `model_rdg.py` / `train_rdg.py` / `data_rdg.py` 逐字节相同；`ci_checks_rdg.py` / `train_rdg2.py` 线上更新 |
| CI 条数 | `jq '.results \| length'` + 逐条数 `check(` 调用 | json 26 条，快照脚本 21 条（§7.4） |
| REPORT 有无 step-8000 读数 | `rg -n "8000" REPORT.md` 逐条读上下文 + 列全部标题 | 13 处命中全在 ETA/计划语境；承诺的 §13 不存在 |
| Stage-2 调度器状态 | `ls runs2`、`ps -eo pid,etime,cmd`、读 `logs/stage2_chain.log` | 目录不存在；PID 379823 空转 2 天 00:33:59；日志 11,059 行止于 `done=5/6` + 脚本报错 |
| 全部派生数字（降幅 / 天花板占比 / 斜率 / 加固差 / 泛化间隙 / 色彩口径比） | 按 `aggregate.py:124-140` 的公式从 `runs/<arm>/metrics.json` 重算 | 与 `metrics_converged.json :: rows[*]` 一致（`gbase_free` 除外，那一行是过期口径） |
