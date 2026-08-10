# NOTES — G3 塌陷通道验证（Gate D3）

## 〇 实施前核实记录（CLAUDE.md 派工协议第 2 条）

本任务卡引用的都是**仓库内文档与仓库内资产**，没有引入任何外部 URL / 论文数字 /
超参出处，因此不触发「在线核实」要求。改为对**所有被引用的仓库内事实逐条落地核实**
（本项目检索引擎有编造前科，仓库内断言同样不默认可信）：

| 断言（来源） | 核实方式 | 结果 |
|---|---|---|
| Gate D3 = 「几百对 + 32 高斯 4D + 纯重建 loss 训 1h，看 σ_s 与 s-sensitivity」 | 读 `PLAN_v2` §5 第〇级 | ✅ 原文一致，N=32 采纳 |
| 退化集 D = 「所有 o_i>0 的高斯共享同一 s 边缘分布」，μ_s 全 0.5 = 鞍点 | 读 `PLAN_v2` §1.2 | ✅；据此把 s 放 Cholesky 首位（见决策 2） |
| R-2 = 「μ_s 不可学 K=6 网格；σ_s∈[0.025,0.30] sigmoid；保留 s–RGB 交叉协方差」，**844 参数** | 读 `PLAN_v2` §2 表 R-2 行 | ✅ 且**本实现 N=32 anchored 臂实测 844 参数，与表中数字逐位吻合**（naive 臂 876 = 844+32 个可学 μ_s）——反向印证参数化选对了 |
| R-3 γ_μ=0.15（std(μ_s) hinge） | 读 `PLAN_v2` §2 表 R-3 行 | ✅ 只作为统计量上报，**不进 loss**（本实验是纯 L_rec） |
| Δ_shuffle 判据 <0.3 dB / ≥3 dB | 任务卡 + `PLAN_v2` §3 M2 | ✅ |
| Δ_const/Δ_shuffle 由 `tools/harness/collapse_probes` 提供，置换保证无不动点 | 读源码 `delta_shuffle` | ✅ derangement 实现正确（`np.roll(order,1)` + 反向散射） |
| 掩膜下采样口径 = PIL BOX 面积加权到 32×32 | 读 `tools/scache/README.md` + `oracle.py` | ✅ 本实验逐字复用同一配方 |
| D-CONSTRUCT sanity = 8 级 ×200/×24，manifest 全参数 | `ls` + `jq` 清点 | ✅ train 每级 200、val 每级 24；L1=semantic_binary、L4=geom_*；**feather_px=0 故两级掩膜实际都是二值**（下采样后才有软边） |
| 合成口径 O=(1−m)I+m·T1(I) | 读 `tools/construct/transforms.py:composite` | ✅ 重渲染直接调用同一函数，非重写 |
| 红线「G 初始化 = 0 不是 I」 | `CLAUDE.md` 红线速查 | ✅ 双臂 G=0，f(x,s)=x at init |
| 红线「s 轴禁平滑正则」 | 同上 | ✅ loss 只有 L1，`train_g3.py` 单行可查 |
| 红线「checkpoint 选择禁用 val loss」 | 同上 | ✅ 不做 checkpoint 选择，一律取最后一步 |
| 红线「每个消融行必带 Δ_const/Δ_shuffle 列」 | 同上 | ✅ metrics.json 每行都带 |

## 一 待主 agent 决策（保守默认已采用，可推翻）

### 决策 1（**最重要**）：任务卡指定的数据档表达不了任务卡指定的判据

**发现**：D-CONSTRUCT L1+L4 原样目标含 **40 种不同变换身份**（class|tier|sign，
每种仅 4–9 对），而 s = 掩膜下采样**只编码 where、不编码 which**。于是任何
f(x,s) 的最优解只能是条件均值 E[y|x,s]，而符号近似对称 ⇒ 该均值退回近恒等。

**实测天花板**（16³ RGB×8 s 分箱条件均值，S-train 拟合 / S-val 评测，
`tools/harness/metrics` 口径）：

| 数据档 | 恒等基线(全图) | 天花板 f(x,s) | Δ_shuffle\*（可达上限） | 掩膜内 Δ_shuffle\* |
|---|---|---|---|---|
| **mixed**（任务卡字面） | 40.21 | 36.54 | **+0.32 dB** | **−2.73 dB** |
| **fixed**（exposure +0.60 EV） | 31.46 | 43.66 | **+14.90 dB** | +13.50 dB |
| tiered（exposure +，幅度随对变化） | — | — | +4 dB 量级（200 步冒烟实测 +4.0） | — |

即：**在 mixed 档上，两臂无论塌陷与否都会读出 Δ_shuffle<0.3 dB**，
「塌陷复现」与「任务无信号」不可区分，Gate D3 会得出一个无效的正面结论。

**保守默认（已采用）**：不删不改任何原始数据；主判读改用 `fixed` 档 =
**同源图、同掩膜、同 `T.composite` 原语**，只把目标按**单一固定变换**重渲染
（`data_construct.FIXED_SPEC`）。同时保留 `tiered`（部分可辨识）与 `mixed`
（任务卡字面档）一起跑，三档按 s 收益从高到低并排读。这样：
- 若 naive 臂在 fixed/tiered 上也塌陷 ⇒ 逃逸通道是**真·零代价**，R-2+R-3 必上；
- 若只在 mixed 上塌陷 ⇒ 那是**零收益**不是零代价，不能据此判 R-2/R-3 必上，
  也不能据此判可降级。

**请主 agent 裁决**：是否接受用重渲染的 `fixed` 档作为 Gate D3 主判读。
若坚持只认原样 mixed 档，则 Gate D3 在现有数据上**无法判定**，需要先补一批
单一变换的 D-CONSTRUCT（`tools/construct/generate.py` 已有全部原语，~3 min/200 对）。

### 决策 2：4D Cholesky 的坐标顺序放 (s, r, g, b) 而非 (r, g, b, s)

Σ=LLᵀ 时，s 放**首位**才有 Var(s)=L[0,0]²，即 L[0,0] **就是**边缘 σ_s；
放末位则 Var(s)=L[3,0]²+L[3,1]²+L[3,2]²+L[3,3]²，此时「σ_s 有界 sigmoid」
既没有界住 PLAN §1.2 定义退化集 D 所用的那个量，画出来的 σ_s 轨迹也不是那个量。
本实现取 s 首位；CI 检查 1 用 `sqrt((LLᵀ)[0,0]) == sigma_s()` 逐点钉住。
第 0 列的三个次对角元 (L[1,0],L[2,0],L[3,0]) 即 s–RGB 交叉协方差，
**保留、不可因子分解**（R-2 明文要求），CI 检查 1 同时验证其非零。

### 决策 3：s 上采样用**双线性**，不用 guided filter

`tools/scache/upsample.py` 的生产路径是 guided_blur（guide=原图）。本实验故意不用：
guided 上采样会把**输入图像本身**注入 s，Δ_const/Δ_shuffle 会因为「s 里混进了图像
内容」而虚高，与「模型是否真的用了 s 轴」无关。训练与评测走同一条双线性路径。
（生产渲染臂仍应按 scache 口径用 guided；这里只为 Gate D3 的因果干净。）

### 决策 4：naive 臂**故意**违反「σ 参数化禁裸 exp」红线

红线针对生产臂。本实验的被测对象就是「照抄 GLUT 各向同性初始化直接升 4D」的
朴素实现，裸 exp + μ_s 全 0.5 正是它的定义。对照臂（anchored）合规。
**两臂除 s 轴参数化外完全相同**（同 init、同 payload、同优化器、同预算、同 seed 集）。

### 决策 5：Δ_const 的 s_∅ 显式传入常量场，不用 harness 默认

`collapse_probes._mean_s_null` 对 `ndim>=2` 的 s 按**最后一维当通道**求均值：
对 (32,32) 的 s 场，它给出的是「逐列均值构成的剖面」而不是常量场。
PLAN §3 M1 要的是常量 s_∅，故本实验显式传 `np.full((32,32), mean_s)`。
**建议主 agent 把这条记进 harness 的已知口径坑**（不改 harness，避免影响在途实验）。

### 决策 6：预算与 seed 分配

任务卡说「1–2 小时（卡 1）」。实测 14 ms/step + 2.1 s/probe，取 25000 步 ≈ 15 min/run。
主档 fixed 给 3 个 seed（判读要看跨 seed 稳定性），tiered/mixed 各 1 个 seed，
共 10 run ≈ 2.5 h。若主 agent 认为超预算，可只保留 fixed 的 3 seed（6 run ≈ 1.5 h）。

## 二 实现要点（审阅时重点看这几条）

1. `model/glut_repro/model.py`（3D / A0 锚点）**一个字节没动**，4D 是新文件
   `model4d_naive.py`，只复用其 `uniform_grid_mu`。
2. 权重用「逐像素 log-max 外提」的等价形式算，
   `w_i = o_i e^{l_i−m} / (Σ_j o_j e^{l_j−m} + ε e^{−m})`——与 Eq.2 原式**数值恒等**
   （CI 对拍 6e-7），但 σ_s→∞（逃逸通道本身）时不会 inf/NaN，而是正确地退到 w→0。
   这一条很关键：如果朴素实现在这里 NaN 了，实验会「因为炸了而看不到塌陷」。
3. Eq.2 的 ε 分母使得 init 时 f(x,s)=x·(1−ε/(Σp·o+ε)) 而**非**严格 x，
   实测偏差 ≤0.37/255 灰阶。CI 检查 5 钉的是这个**闭式**而不是一个手调容差。
4. σ_s→∞ 时模型退化为「只剩全局仿射 Gx+g」——这正是逃逸通道的终点形态，
   渲染样例里可直接看到（掩膜内外同色）。
5. 性能：一次训练步 14 ms。曾误判瓶颈两次（先怀疑 4×4 三角求解、再怀疑逐步
   `.item()` 同步），实测都不是主因，真正的开销是**每 100 步的整图 probe**
   （probe-n=8 时 2.1 s/次）。三处优化（前代替换 solve_triangular、payload 先缩并
   高斯维、去掉逐步同步）合计把 wallclock 从 55 → 14 ms/step，已保留。

## 三 已知局限（结果审阅请一并看）

- **玩具规模**：单一全局渲染器 + N=32 + 400 对，按 PLAN 的 Gate D3 口径。
  不能外推到 RD-STD 的全量训练。
- **s 只有 where 没有 what**：这是 oracle 掩膜 s 的固有性质，也是决策 1 的根源。
  真实 s（VLM 读出 / 掩膜基底）值域 [−3,3] 且连续，信息量更大；本实验的
  s∈[0,1] 是其**下界**情形。K=6 锚点网格因此铺在 [0,1] 而非 PLAN §1.2 的 [−3,3]。
- **Δ_shuffle 的置换粒度**是整张 32×32 s 场（跨图），与 harness 口径一致。
- 未跑 R-3（多样性 hinge）本身，只把 std(μ_s) 当统计量上报——任务卡要的对照臂
  是「R-2 最小版」，R-3 需要改 loss，超出本卡范围。

---

## 四 收尾回合（回填 REPORT，2026-08-03 下午）核实记录

**本回合不训练、不占 GPU、不改 `runs/`**（`runs/` 全程只读；重出图与重跑聚合均
`CUDA_VISIBLE_DEVICES="" --device cpu`）。

### 4.1 读过的文档章节（CLAUDE.md 派工协议第 1 条）

| 文档 | 章节 | 用途 |
|---|---|---|
| `EXPERIMENTS_v3` | §3.1 渲染器臂表（line 77 RD-STD / line 82 RD-E）、line 160 失败分支图、line 177 W2 里程碑 | §7 的行级修改建议 |
| `PLAN_v2` | §1.2（退化集 D 的数学）、§2.1 表 R-2/R-3 行（line 124/125）、§5 Gate D3、§6 失败分支树 | 判据出处与 §4.2 的命题拆分 |

### 4.2 本回合逐条核实（不接受任何转述，全部落地重算）

| 断言 | 核实方式 | 结果 |
|---|---|---|
| 10 个 run 全部跑完 | `grep G3_FULL_DONE logs/g3_full.log` + 逐 run `G3_RUN_DONE` + 10 份 metrics.json/ckpt.pt | ✅ |
| 聚合 metrics.json 与逐 run / 日志一致 | 三处对拍 6 行数字 | ✅ 逐位吻合 |
| 聚合可复现 | CPU 重跑 `analyze_g3.py` | ✅ 6 行数字逐位复现 |
| 探针 L1:L4 均衡（STATUS 记载的 bug 已修） | 独立复算 `build_probe_samples` 的 `np.linspace` 取样 | ✅ n=8→4:4，n=12→6:6，n=48→24:24 |
| Δ_shuffle 的负控制真的是无不动点置换 | 读 `collapse_probes.delta_shuffle` | ✅ seeded permutation + `np.roll(order,1)` |
| σ_s / std(μ_s) / sens 的轨迹形状 | 从 10 份 trace.jsonl 抽 step 0/2500/5000/12500/25000 逐 run 打表 | ✅ 见 REPORT §3.2 |
| 「天花板」是不是天花板 | 拿实测 psnr_full / Δ_shuffle 与 43.66 / 36.54 / +14.90 / +0.32 对比 | ❌ **三处被越过**，且脚本不在交付里 → REPORT §2 已改口径（见下） |

### 4.3 本回合的三处主动修改（供实现审阅核对）

1. **REPORT §2 的「天花板 / 该档可达上限」改称「分箱 LUT 参考值」**，并加显式更正框。
   触发原因：实测 fixed 的 psnr_full 46.7–46.9 > LUT 的 43.66、Δ_shuffle +18.1 > +14.90，
   mixed naive +0.585 > +0.32——**三处都越过了那个"上限"**。原措辞会让后续实验
   把它当成 gate 的分母（§7 建议 2/3 恰好要这么用），必须先把口径说对。
2. **`analyze_g3.py::fig_sigma_s` 三处可读性修复**（图例压字 / 对数轴只有一个刻度 /
   缺初值参考线与 seed 计数）。这是论文附录图，按附录标准修。
3. **`delta_shuffle.png` / `delta_const.png` 标题加口径注**：轨迹是 8 对探针集、
   结论表是 48 对，低信号档两者差 >1 dB（mixed/naive: trace +2.357 vs final +0.585）。
   不加注就会被交叉读成相反结论。

### 4.4 待主 agent 决策（本回合新增，保守默认已采用）

- **决策 7：Gate D3 的判词**。保守默认写成「**通道未证实，但也未被否证**」，
  并在 REPORT §4.2 把「梯度不会自发走进通道」与「加固可以删掉」拆成两个命题、
  逐条标注 G3 支持哪个。**另一种合理做法**是直接判 Gate D3 `PASS`（Δ_shuffle 三档
  两档 ≥3 dB），但那会让「R-2/R-3 必上」这条主张失去实证支撑而**无人察觉**，
  故不采用。请主 agent 确认判词。
- **决策 8：γ_μ 是否改为相对量**（REPORT §7 建议 5）。这条要改 `PLAN_v2` §2.1
  R-3 行（权威文档），**不在本卡权限内**，只出建议；但它直接影响卡 0 上在跑的
  RD-STD——若其 s 值域是 [−3,3] 而 γ_μ 仍写死 0.15，R-3 现在就是空操作。
- **决策 9：前置估计脚本是否补进仓库**（REPORT §7 建议 7）。建议 2/3 要把
  Δ\*_shuffle 升级成 gate 的一部分，那就必须先让它可复现；本回合没有补写
  （不在任务卡范围，且会引入新的未审阅代码）。
