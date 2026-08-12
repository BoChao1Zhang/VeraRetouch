# 交接：几何注入实施（零上下文接力者自足版）

> 写于 2026-08-12。读者假设：**零上下文**，没参与过前面的战役。
> 本文件自足：读完它 + 引用的 metrics.json 就能接着干，不需要回溯对话。
> 前一版 `HANDOFF_geometry-injection_20260811.md` 仍在，本文件是**更新且更完整**的版本，以本文件为准。

---

## 0. 三十秒版

- **病灶已定位**：不是数据、不是 reasoning 质量，而是**条件注入通路**。
  几何信息在 `<where>` 里（覆盖 79%、词级准确 82%），但把**真值文本**喂给头只值 **+0.0007~+0.0087 IoU**。
- **当前最好成绩**：P3'_CONT **0.7622**（normal-only, generated），已过 0.75 目标。
- **下一步**：把几何码注入头。**B2 系（broadcast 下界形态）三臂全部因一个 bug 崩了（§2.2），修法已给出**。
  **PCH 模块（正式形态）已实现并冒烟通过，A1 臂未开工。**
- ⚠️ **gpu0 当前空闲**（2026-08-12 交接时），队列为空。接力者第一件事应是修 §2.2 的 bug 并重提 B2 三臂。

---

## 1. 实施状态快照

### 1.1 解析器（已完成，可直接用）

`q3vl/whereb/amort/geomparse.py`，**21 维多热码**（`GEOM_DIM=21`）。

- `geom_features(where_text)` —— 从 `<where>` 的 `edit scope:` 子句解析（**部署路径**，~82% 词级质量）。
- `geom_features_from_vrmeta(slot_id, region)` —— 从构造侧 `.vrmeta.json` 直取（**真值，零生成误差**）。
- `shuffle_features(v, rng)` —— 负控制，**保持激活位数**（否则「几何错了」会和「输入少了」混淆）。

**对表产物**：`amort_dx_20260811/geom_vocab_crosscheck.json`
—— 生产模板 v4a（`responses.py::_geometry_words`）的全部产出词 → 槽位映射，**未覆盖词 = 0**。

**已修的三个 bug（都会静默污染码，不要退回）**：

| bug | 后果 |
|---|---|
| token 正则 `[a-z\-]+` | `lower-right corner` 成一个 token、匹配不到任何槽 ⇒ **角落方向全丢** |
| 无 q=3 中桶 | `moderately wide` / `moderate` 无处可去 ⇒ extent 系统性低读 |
| 短语未优先 | `moderately wide` 同时点亮 `ext_large`（三桶应互斥）；对角轴描述词 `diagonal, running from the upper left down to the lower right` 喷出 4 个假方向位 |

⚠️ **GT 码不是解析码的严格超集**：vrmeta 有 `slot_id`（shape）和 `region`（direction），**没有 extent**。
故 GT 码在 9 槽精确、6 个 extent 槽恒 0。比较时必须记住这条不对称。

### 1.2 PCH 注入模块（已完成，已冒烟，**不要重写**）

`q3vl/whereb/amort/pch.py`。原型 token × dense 特征 cross-attn + hypernetwork 调制。

- `PCHConfig.full()` → **3,855,296** 参数（提案 Full 预算 3.96M）
- `PCHConfig.lite()` → **501,536** 参数（提案 Lite 预算 0.52M）
- 冒烟已验证：**零初始化输出**（step 0 精确 no-op，故加到 resumed checkpoint 上不会破坏它）；
  训练几步后残差非零；**空码 ⇒ 残差恒为 0**（优雅回退真的生效，不是装饰）。
- 用法：`residual = pch(feat, code)`，**调用方自己做加法**（注入点留在调用处可见）。

**PCH 相对 broadcast 的两个理由**（写进 REPORT 时可直接引用）：
① 空间选择性——attention 让每个格子取它需要的原型，`broad band, horizontal` 可以在沿带/跨带方向上表现不同；
② 组合性——hypernetwork 调制共享原型库而非查表，没见过的槽组合会插值而不是掉出表外。

### 1.3 A1 臂：**未开工**

A1 = 修复版解析器（21 维码）→ PCH 注入 + 两遍式 forced-prefix `<geom>` 改造 + **500 样本试点门**。

可复用基建：
- **forced prefix 生成**已存在：`q3vl/whereb/hiddens.py::FrozenVLM.generate_where(..., prefix_ids=...)`
  （Stage-What 的 C01/C02 用它强制 `<color>` 开头）。
- **加新上下文模式的唯一入口**：`q3vl/whereb/amort/data.py::AmortBatchBuilder.context_for`。
- **注入接线已通**（broadcast 形态）：`AmortModel(geom_inject=True)` +
  `AmortBatchBuilder(geom_inject=..., geom_source="parsed"|"vrmeta", geom_shuffle=...)`。
  A1 只需把 broadcast 换成 PCH 调用点。
- **纪律**：500 样本试点门先跑、过门才全臂。

---

## 2. 队列与在途作业

### 2.1 已落盘（接力者只需读 metrics，不要重跑）

判据口径统一为 **matched-area top-k IoU、normal-only**（`winner_confidence=="normal"`，n=224）。
路径前缀 `/home/bc/data/runs/where_b/<run>/eval_final/metrics.json`。

| run | 是什么 | 关键数字 |
|---|---|---|
| `amort_P3prime_20260810` | P3' 基线（1200 步） | **0.7417** |
| `amort_P1_20260810` | P1（经 Phi-71） | **0.7095** |
| `amort_P1_pooled_20260811` | 池化对照（同参数量） | **0.6222**；`corr(中心先验)−corr(GT)` **+0.1226**（M3 复活） |
| `amort_P3prime_cont_20260811` | P3' 续训 | **0.7622**（配对 +0.0254，p=1e-4）**← 当前最好** |
| `amort_P3prime_cont2_20260811` | 二次续训 | 已落盘，**未读**（接力者读它判断续训是否还在爬） |
| `amort_SHAPE3_A_20260811` | eikonal 距离场重参数化 | 已落盘，**未读** |
| `amort_SHAPE3_B_20260811` | 同容量普通头 + 最强 P2 结构损失（形状对照） | 已落盘，**未读** |
| `amort_P3prime_p2struct*/p2w*` | P2 结构损失包四臂 | 0.7522 / 0.7486 / 0.7477 —— **全部低于 0.7622，已定案不进配方** |

其它已落盘产物：
- `experiments/.../where_b/amort_dx_20260811/` —— DX-1..6（脏边归因）
- `experiments/.../where_b/probe_gated_upsample_20260811/` —— 门控上采样 A/B（含 REPORT + viz）
- `experiments/.../where_b/ceilpush_d0_20260811/` —— D0-3/5/8a + `d0_7/`（U_replay）+ `p0_hidden_probe/`
- `experiments/.../where_b/analysis_longtail/`（在 `amort_p3prime_20260810/` 内）—— 六维分层 + 长尾归因

### 2.2 ⚠️ B2 系三臂**全部失败**——bug 已定位，修法如下

`B2_GTCODE` / `B2_PARSED` / `B2_SHUFFLE` 三个作业**都在启动后约 2 分钟崩溃**，
`q status` 里是 `Failed rc=1 traceback`。

**根因**：`--geom-inject` 会给 conv 塔的 stem 增加 21 个输入通道
（`AmortModel.__init__` 里 `extra_ch += geom_dim`），而 `--resume` 加载的 P3' checkpoint
的 `geo.tower.stem.weight` 是**窄的**，于是：

```
RuntimeError: Error(s) in loading state_dict for AmortModel:
  ...size mismatch for geo.tower.stem.weight...
```

崩溃点：`q3vl/whereb/scripts/run_amort_arm.py:291`，`model.load_state_dict(sd["model"])`。

**修法（二选一，推荐 B）**：

- **A（快）**：`load_state_dict(sd["model"], strict=False)`，然后**把 stem 的新增通道权重显式置零**
  ——新通道零初始化 ⇒ 恢复瞬间模型与原 checkpoint **逐位等价**，几何通道从零开始学。
  不置零会让 resumed 模型一上来就被随机权重扰动，等于白费续训。
- **B（更干净，且是提案正式形态）**：**改用 PCH**。PCH 是**残差旁路模块**、输出零初始化，
  **完全不改 stem 的输入通道**，所以 resume 天然不冲突。这也是提案 §D-1 要的形态。
  ⇒ 建议接力者跳过 broadcast 版，直接上 PCH，把 B2 三档（GT 码 / 解析码 / shuffle）
  用 PCH 重跑一遍——一次拿到 M1 门的正式数字。

**三档的语义（不要混）**：

| 档 | `--geom-source` | 是什么 |
|---|---|---|
| GT 码 | `vrmeta` | 真值码，**go/no-go 上界**。它不动 ⇒ 整个几何注入方向死 |
| 解析码 | `parsed` | 部署路径（~82% 质量） |
| shuffle | `parsed --geom-shuffle` | 负控制，**保持激活位数** ⇒ 隔离「几何内容有用」与「多了通道有用」 |

---

## 3. 判读规则汇编

### 3.1 M1 门层级（**不要混用两条线**）

| 门 | 注入形态 | 判据 |
|---|---|---|
| B2（broadcast，**下界形态**） | 常量通道 | **≥ +0.005** ⇒ 方向坐实，进提案 P1 阶段；**不过 ⇒ 不判死** |
| M1（**PCH，正式形态**） | 原型 token cross-attn + hypernet | **< +0.008 ⇒ 三臂全停**；**≥ +0.015 ⇒ 晋级** |

**下界不否定上界**：broadcast 阴性只说明「最弱形态不行」，不说明方向错。

### 3.2 C（query 桥）臂立不立

规则：**解析臂捕获 Δ_GT 的比值 ≥ 0.7 ⇒ 不立 C 臂**（解析已经吃到大头，端到端可学提取不值一臂）。
比值 = (解析码 Δ) / (GT 码 Δ)。**B1（attention 读出）臂已砍**（依据见 §3.5）。

### 3.3 SHAPE3 判据

预注册：**SHAPE3_A 的形状残差显著低于 SHAPE3_B，且 IoU 不降 >0.01 ⇒ 层 3 采纳**。
- 形状残差 = `q3vl/whereb/edgequal.py::shape_residual`
  = `1 − IoU(pred, 该族解析形状的最小二乘最佳拟合)`；GT 自拟合 ≈ 0 是对照列。
  已验证判别力：真椭圆 0.047 / 噪声椭圆 0.040 / **光滑但非椭圆的团 0.268**。
- **背景**：用户肉眼判定「门控后场光滑了但形状还是不对」——光滑度与形状规整性是两回事，
  SHAPE3 就是为这条观察立的臂。

### 3.4 CONT2 读法

CONT（2541 步，撞 4h 墙钟而非步数上限）把 0.7417 推到 **0.7622**，曲线**未平**。
CONT2 读法：**若继续显著上升 ⇒ 续训仍是最便宜的推进手段，注入臂的增益要在此基线上算**；
若已平 ⇒ 续训见顶，注入是唯一剩下的路。**注意基线随之改变**：所有注入臂的 Δ 必须对**最新**基线算。

### 3.5 已定的方向裁定（依据数字，接力者不必复验）

| 裁定 | 依据 |
|---|---|
| 病灶 = 条件注入通路 | gt vs generated 配对 Δ = **+0.0007 / +0.0087 / +0.0053**，三臂全在 0.01 带下；而 `<where>` 对 shape 覆盖 79%、词级准确 82% |
| **B1（attention 读出）砍掉** | P0 探针：generated 档最佳层 shape acc **0.7374 < 0.82**；36 层曲线只涨 **+0.0101**；gt 档 0.9848 是**退化读数**（layer0 静态 embedding 已 0.9646 = 读回自己的输入） |
| **P2 结构损失包不进配方** | 三臂 0.7522/0.7486/0.7477 全部低于 0.7622 |
| **门控上采样 B 档进配方** | κ̃ 3.398→0.080（−97.6%，p=1e-4），IoU Δ=−0.0000（p=0.99）；语义族关引导掉 bF1 0.107（p=1e-4） |
| **池化是独立病因** | 同参数量对照：0.7095 → 0.6222（配对 +0.0856，p=1e-4），且 **M3 复活**（corr 差 +0.1226） |
| (ii) 固有随机性实质存在 | D0-7：U_replay **0.8040** 全体 / **0.7596** 几何族 |

**关键论证（提案的立论基础）**：`<where>` 文本是**掩膜的下游产物**
（`responses.py:1443` 把 `edit_geometry_hint` 烘进生成 prompt），所以它**泄露了实际抽到的那一次几何**。
U_replay 只是「从 (图, 指令) 预测」的上界；**注入若成功可以超过 U_replay**。

**另一条硬事实**：连续量（角度/宽度/覆盖）在**生成期就被 q=3 量化成词**
（`_geometry_words`，生产变体 `PROMPT_VARIANT="v4a"`：narrow/moderately wide/broad；tight/moderate/large）。
⇒ **不存在可供任何文本路线回收的连续残差**，21 维码相对文本已接近无损。

### 3.6 判据参照值（normal-only，V_where local n=224）

随机地板 **0.2254** ｜ 中心先验 **0.4853** ｜ W01 generated **0.4874** ｜ W01 gt **0.5109**。
⚠️ 全项目早期引用的 pooled 值（0.2582 / 0.5088 / 0.5333）含 **44% 的 `low` 样本**，
会系统性抬高每个 IoU 列，**不得用作判据**。

---

## 4. 工程军规（全部是这两天付过学费的，每条都不报错）

1. **队列作业 fd 软限是 1024**（继承 `pueued`），交互 shell 是 1,048,576。
   同一份代码手跑 0 失败、队列里 **83.7%** 失败。**入口处必须把 `RLIMIT_NOFILE` 软限抬到硬限。**
2. **`CUDA_VISIBLE_DEVICES` 由 `qjob.sh:265` 设置**，卡内逻辑序号恒为 **0**。
   payload 必须写 `--device cuda:0`；写 `cuda:$GPU` 在 1 号卡必炸、在 0 号卡「碰巧」通过。
3. **`MaskResolver` 非线程安全**（sqlite 句柄）⇒ `threading.local()` 每线程一个。
4. **任何「静默降级」路径必须配覆盖率硬守卫**（如 `except: return "unknown"` 要配 `>5% 即 raise`），
   否则环境问题会伪装成数据问题。失败若对类别**无偏**，幸存直方图看起来完全正常。
5. **NaN 守卫**：曲率类损失 `grad/|grad|` 的 `eps=1e-6` 会在局部平坦处爆炸，
   已改 `1e-3` + 曲率 clamp；训练器已有守卫（跳步 + 计数 + 超 50 步中止）。
   **一次 NaN 会静默毁掉整臂并照常产出像样的板**（三臂曾落在逐位相同的 0.1692）。
6. **先落盘再跑可选阶段**，可选阶段一律 try/except。DX 电池被最后一步连炸两次，
   前面五个已完成实验的结果全丢。
7. **取消必须绑定补位**：任何 cancel/完成事件后自查两卡待跑队列非空，为空立即补预备作业。
   自查：`q status | grep -cE "\| gpu0 \| Queued"`。
8. **禁止 shell 循环批量提交队列作业**：`set -- $spec` 拆参失败会让 `q` 误解析成**取消**动作
   （我因此误杀过一个正在跑的作业）。一次一条、逐条核对返回行。
9. **提交后看 `q status` 实际落位**，不要假设它排进了队列——`pueue` 组并行度被改成 2 时，
   `q submit` 会**直接开跑**而不排队（四个训练同时抢卡）。核并行度：`pueue group`。
10. **按名寻址队列**：`pueue switch` 会交换两个任务的 **id**，id 会在你手里漂移；名字稳定。
11. **重放 GT 必须换 `build_id` 或 `output_root`**：`mask_id` 不含 seed 且 `_write_cgt_once`
    见文件存在即返回 ⇒ **静默复用旧 `.cgt.png` 字节**、同时写入新几何。
12. **日志禁跨轮 append**：混两轮的 `steps.jsonl` 里 `head` 是死跑、`tail` 是活跑，中间无标记
    （已加 `_rotate`，move-aside 不删）。
13. **在错误环境里做的全量验证毫无价值，甚至有害**——它给「已修复」的假信号。
    要验证会在队列/后台出现的失败，先**复现该环境的约束**（fd 软限、`CUDA_VISIBLE_DEVICES`、cwd）。
14. **每处修复都要有一个能证伪它的观测量**，并在冒烟里核对该观测量本身。
    「代码改了 + 不报错」不是证据——我有三次「修好了但没生效」（clamp 写错位置被覆盖、
    p 值读错 key 整列变 `n/a`、读到上一轮的 stale 日志行）。
15. **`q events` 只记录「谁在何时做了什么」，不记录「为什么」**。多主体共用队列时，
    时间相关性**不是**因果证据（我曾把主 agent 执行用户裁定的取消误判成自己的 bug）。
16. **判据纪律**：禁 AUC；阈值化一律匹配 GT 面积 top-k；配对差分同图内 + 置换 p 值；
    色标固定 0..1 禁逐图 min-max；叠图用 `grid_to_img` 严格逆映射；
    checkpoint 选择禁用 val loss（硬门全过才可选，无人过门就如实报「无可选 checkpoint」）。
17. **κ̃ 度量两处必须保持**：τ **按家族**标定（单一 pooled τ 会让 linear 全族窄带为空、
    κ̃ 恒 NaN 而**静默从中位数消失**）；窄带一律取自 **GT**，A/B 在同一批像素上比较
    （否则更光滑的一档因带为空得 NaN，与预期读数**相反**）。

---

## 5. 代码地图 `q3vl/whereb/amort/`

| 文件 | 职责 |
|---|---|
| `model.py` | `AmortModel`：三种臂（`P1` / `P3prime` / `SHAPE3`）+ 语义头 + 路由 + 门控上采样开关 |
| `heads.py` | 卷积塔、FiLM、`CoeffHead`（空间读出 w）、`PooledCoeffHead`（池化对照）、`P3PrimeHead`、`ShapeDistHead`（距离场）、`SemanticHead` |
| `pch.py` | **PCH 注入模块**（原型 token cross-attn + hypernet，Full/Lite，零初始化 + 空码回退） |
| `geomparse.py` | 21 维几何码：文本解析 / vrmeta 真值 / shuffle 负控制 |
| `losses.py` | 五项预注册 loss + P2 结构项（curv/mono，已定案不用）+ `eikonal`（SHAPE3 用） |
| `data.py` | `AmortBatchBuilder`：一次前向出 F_pre/merger/H_where，相似度场、几何码、配对 partner、家族标签、上下文模式 |
| `trainer.py` | 训练循环、早警列、NaN 守卫、日志 rotate、`best()`（硬门 + 禁 val loss） |
| `evaluate.py` | 六上下文评测板、分层、`m_sem`、E3 证伪列、硬门 |
| `simfield.py` | 相似度场 + 整臂常量归一化 + 消费侧域断言（s 缓存契约） |
| `viz.py` | success/failure 五联图（固定色标、严格逆映射） |
| `tests/` | `test_separation_guard.py`（U1 回归，3 条） |

**同级相关**：`q3vl/whereb/edgequal.py`（E_HF / κ̃ / MVR / AFR / `shape_residual`）、
`q3vl/whereb/scripts/`（`run_amort_arm.py` 训练入口、`amort_board.py` 判据表、
`amort_deliver.py` 交付组装、`run_amort_dx.py`、`probe_gated_upsample.py`、
`run_ceilpush_d0.py`、`run_d0_7_replay.py`、`run_p0_hidden_probe.py`、`viz_*`）。

**payload**：`/home/bc/agent-gpu-queue/waves/amort_arm.sh`
（`AMORT_RUN_NAME` / `AMORT_MAX_STEPS` / `AMORT_MAX_HOURS` 可覆盖；已钉 `--device cuda:0` 与 `--attn eager`）。

---

## 6. 接力者建议的第一批动作

1. **读三份未读的板**：`amort_P3prime_cont2_20260811`、`amort_SHAPE3_A/B_20260811`
   （按 §3.3/§3.4 判读；SHAPE3 用 `shape_residual`，不要只看 IoU）。
2. **修 §2.2 的 resume bug**，推荐直接走 PCH（方案 B），用 PCH 重跑 B2 三档 ⇒ 一次拿到 M1 正式数字。
3. **gpu0 当前空闲**，队列为空 —— 按军规 7 立即补位。
4. A1（两遍式 `<geom>`）在 M1 门过了之后再开；**500 样本试点门先跑**。
