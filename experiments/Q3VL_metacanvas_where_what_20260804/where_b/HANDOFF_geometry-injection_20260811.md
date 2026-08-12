# 交接：几何注入提案实施（PCH + A1）

> 写于 2026-08-11，交接原因：上下文预算（~78 万 tokens）不足以安全完成 PCH + A1 全量实施。
> 按 `amort_p1_20260810/NOTES.md` 先例，把**裁定快照 / 已完成状态 / 队列 / 地雷**一次交清。
> **本文件是工作交接，不是实验交付**；实验交付仍按 REPORT 三行 + metrics + viz 规范。

---

## 0. 一句话状态

**PCH 注入模块已实现并冒烟通过（见 §3），A1 臂（两遍式 forced-prefix `<geom>` 改造 + 500 样本试点门）未开工**，
留给接力 agent。解析器已修好并对表（§2），B2 系四臂在队列里跑前置数字（§4）。

---

## 1. 裁定快照（接力 agent 必须先读这一节）

### 1.1 已定的方向裁定

| # | 裁定 | 依据数字 |
|---|---|---|
| 病灶 = **条件注入通路**，不是数据、不是 reasoning 质量 | 已定 | 读数 1：gt-context vs generated-context 配对 Δ = **+0.0007 / +0.0087 / +0.0053**（三臂全在 0.01 判据带以下）；读数 2：`<where>` 对 shape 的覆盖 **79%**、词级准确 **82%** |
| **B1（attention 读出）砍掉** | 已定 | P0 探针：generated 档最佳层 shape acc **0.7374 < 0.82**；曲线扁平（embedding→layer36 仅 **+0.0101**）；gt 档 0.9848 是**退化读数**（layer0 静态 embedding 已 0.9646，读回自己的输入） |
| **P2 结构损失包不进配方** | 已定 | NaN 修复后三臂复活但全部低于基线：0.7522 / 0.7486 / 0.7477 vs **0.7622** |
| **门控上采样 B 档已进配方** | 已定 | κ̃ 3.398→0.080（−97.6%，p=1e-4），IoU Δ=−0.0000（p=0.99）；语义族负对照掉 bF1 0.107（p=1e-4） |
| **(ii) 固有随机性实质存在** | 已定 | D0-7：U_replay **0.8040** 全体 / **0.7596** 几何族（≤0.85 带） |
| 唯一被证实推动 IoU 的手段 = **续训** | 已定 | 0.7417 → **0.7622**（配对 +0.0254，p=1e-4） |

### 1.2 关键机制事实（决定提案为什么这么设计）

1. **几何住在 `<where>`，指令按设计不含几何**——`responses.py::_GEOMETRY_WORDING` 原文要求
   「Declare that subject and that reach in region_scope. Everywhere else … **never by its shape,
   its direction, or any number from this hint**」。故 D0-5 测到的「指令 shape 覆盖 0.3%」**不是缺陷**。
2. **连续量在生成期就被 q=3 量化成词**（`_geometry_words`，生产变体 `PROMPT_VARIANT="v4a"`）：
   band 宽度 → narrow / moderately wide / broad；radial 覆盖 → tight / moderate / large；角度 → axis_bucket。
   **推论：不存在可供任何文本路线回收的连续残差**——21 维码相对文本已接近无损。
3. **`<where>` 文本是掩膜的下游产物**（`responses.py:1443` 把 `edit_geometry_hint` 烘进 prompt）。
   故 D0-7 的 U_replay 只是「从 (图, 指令) 预测」的上界；`<where>` **泄露了实际抽到的那一次**，
   注入若成功**可以超过 U_replay**。这是提案的核心论证。

### 1.3 门的层级（不要混用）

| 门 | 注入形态 | 线 |
|---|---|---|
| 我在跑的 **B2_GTCODE** | broadcast 通道（**下界形态**） | **≥ +0.005** → 方向坐实，直接进提案 P1 阶段；不过 **不判死** |
| 提案正式 **M1** | **PCH**（原型 token cross-attn + hypernetwork） | **< +0.008** 三臂全停 / **≥ +0.015** 晋级 |

**B2_PARSED 阳性（Δ≥+0.02）= 方向坐实且仍有注入升级空间；阴性 ≠ 方向判死**（下界不否定上界）。

---

## 2. 已完成：解析器修复（21 维码）

`q3vl/whereb/amort/geomparse.py`。**三个 bug 已修**，全部会静默污染注入码：

| bug | 后果 | 修法 |
|---|---|---|
| token 正则 `[a-z\-]+` | 生成器输出连字符方位（`lower-right corner`），整体成一个 token、**匹配不到任何槽** ⇒ 角落方向全丢 | 改 `[a-z]+` |
| 无 q=3 中桶 | `moderately wide` / `moderate` 无处可去 | 新增 `ext_moderate` |
| `moderately wide` 同时点亮 `ext_large` | 含 "wide"；q=3 三桶应互斥 | 短语优先规则 |
| 对角轴描述词 | 「diagonal, running from the upper left down to the lower right」含四个方位词，但它是**一个朝向**不是四个方向 ⇒ 每个对角样本喷 4 个假方向位 | 短语优先 → 只点 `dir_diagonal` |

- **`GEOM_DIM` 20 → 21**（新增 `ext_moderate`）。所有臂从同一份代码建模型，维度自洽。
- **对表产物**：`amort_dx_20260811/geom_vocab_crosscheck.json` —— v4a 全部产出词
  （两套 gauge、4 个 axis bucket、8 个 compass bucket、shape 短语、anchor）→ 槽位映射，
  **未覆盖词 = 0**。
- **GT 码**（`geom_features_from_vrmeta`）：`.vrmeta.json` 的 `slot_id`→shape、`region`→direction，
  **无需解析文本、零生成误差**。⚠️ **vrmeta 不记录 extent**，故 GT 码在 9 个槽精确、6 个 extent 槽恒 0
  ——**GT 码不是解析码的严格超集**，比较时必须记住这条不对称。

---

## 3. 已完成：PCH 共享注入模块

见 `q3vl/whereb/amort/pch.py`（本次实现，已冒烟）。规格与提案 §D-1 对齐：
原型 token × dense 特征 cross-attn + hypernetwork 调制，Full / Lite 两档，含低置信优雅回退。
**接力 agent 直接复用，不要重写。** 用法与参数量见该文件 docstring 与 §5 的冒烟命令。

---

## 4. 未开工：A1 臂（交接重点）

**A1 = 修复版解析器（21 维码）→ PCH 注入 + 两遍式 forced-prefix `<geom>` 改造 + 500 样本试点门。**

已有的可复用基建：
- **forced prefix 生成**：`q3vl/whereb/hiddens.py::FrozenVLM.generate_where(..., prefix_ids=...)`
  已实现「强制前缀」语义（Stage-What 的 C01/C02 用它强制 `<color>` 开头）。两遍式改造应基于它。
- **上下文模式**：`q3vl/whereb/amort/data.py::AmortBatchBuilder.context_for` 是加新模式的唯一入口
  （现有 gt / generated / shuffled / antonym / fixed_phrase / irrelevant_words / null / foreign）。
- **生成上下文存储**：`GenContextStore`（`q3vl/whereb/stores.py`），train 全量 159,215 行、V_where 896 行。
- **注入接线**：`AmortModel(geom_inject=True)` + `AmortBatchBuilder(geom_inject=..., geom_source=...)`
  已经通了（broadcast 形态）。A1 只需把 broadcast 换成 PCH 调用点。

**试点门纪律**：500 样本试点先跑、过门才全臂——不要图省事直接全量。

---

## 5. 队列与复现

- 提案作业排在 **B2 系之后**（它们是前置数字）、**Stage-What 之前**。
- 满载纪律：**任何取消/完成事件后自查两卡待跑队列非空**，为空立即补预备作业。
  自查：`q status | grep -cE "\| gpu0 \| Queued"`（gpu1 同）。
- 提交纪律：**禁止用 shell 循环批量提交**（`set -- $spec` 拆参失败会让 `q` 误解析成取消动作，
  我因此误杀过一个作业）；一次一条、逐条核对返回行；提交后**看 `q status` 实际落位**，
  不要假设它排进了队列（`pueue` 组并行度被改成 2 时，`q submit` 会直接开跑而不排队）。

冒烟命令模板（把 `--geom-inject` 换成 PCH 开关即可）：

```bash
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
/home/bc/envs/q3vl_sft/bin/python -m q3vl.whereb.scripts.run_amort_arm \
  --arm P3prime --geom-inject --device cuda:0 \
  --train-limit 32 --eval-limit 12 --quick-eval-limit 6 --norm-samples 8 \
  --micro-batch 4 --effective-batch 8 --max-steps 3 --save-steps 3 --eval-steps 3 \
  --max-hours 0.15 --run-name smoke_x --out-root /tmp/.../scratchpad/smoke
```

---

## 6. 地雷清单（都不报错，都咬过人）

1. **队列作业的 fd 软限是 1024**（继承 `pueued`），交互 shell 是 1,048,576。
   同一份代码手跑 0 失败、队列里 83.7% 失败。**入口处必须抬 `RLIMIT_NOFILE` 软限到硬限。**
2. **`CUDA_VISIBLE_DEVICES` 由 `qjob.sh:265` 设置**，卡内逻辑序号恒为 **0**。
   payload 里必须 `--device cuda:0`，写 `cuda:$GPU` 在 1 号卡上必炸（0 号卡会「碰巧」通过）。
3. **`MaskResolver` 非线程安全**（sqlite 句柄），必须 `threading.local()` 每线程一个。
4. **重放 GT 必须换 `build_id` 或 `output_root`**：`mask_id` 不含 seed 且 `_write_cgt_once`
   见文件存在即返回 ⇒ 静默复用旧 `.cgt.png` 字节、同时写入新几何。
5. **曲率类损失的 `eps=1e-6` 会 NaN**（局部平坦处 `grad/|grad|` 爆炸），已改 `1e-3` + 曲率 clamp；
   训练器已加 NaN 守卫（跳步 + 计数 + 超 50 步中止）。一次 NaN 会静默毁掉整臂并**照常产出像样的板**。
6. **日志禁跨轮 append**：`steps.jsonl` 混两轮时 `head` 是死跑、`tail` 是活跑，中间无标记。
   已加 `_rotate`（move-aside，不删）。
7. **可选阶段必须最后跑且不得毁掉已完成结果**：DX 电池被 DX-5 连炸两次，
   五个已完成实验的结果全丢。现已改为「先落盘，再跑可选项，可选项 try/except」。
8. **判据口径**：主列 = **matched-area top-k IoU、normal-only**（`winner_confidence=="normal"`）。
   pooled 口径含 44% low，会系统性抬高每个 IoU 列。参照值（normal-only）：
   随机地板 **0.2254**、中心先验 **0.4853**、W01 generated **0.4874**、W01 gt **0.5109**。
9. **κ̃ 度量两处已修**：τ 按家族标定（单一 pooled τ 会让 linear 全族窄带为空、κ̃ 恒 NaN 而静默消失）；
   窄带一律取自 **GT**，A/B 在同一批像素上比较（否则更光滑的一档因带为空得 NaN，与预期读数相反）。
10. **`q events` 只记录「谁在何时做了什么」，不记录「为什么」**。多主体共用队列时，
    时间相关性**不是**因果证据——我曾把主 agent 执行用户裁定的一次取消误判成自己的 bug。

---

## 7. 待接力 agent 决策（不许静默拍板）

- **C（query 桥）臂是否立**：规则已定——解析臂捕获 Δ_GT 的比值 **≥0.7** 则不立。数字未到。
- **A1 两遍式生成的成本**：两遍前向会让训练期每步成本上升，需先量一遍（`generate_where` 的实测耗时）
  再决定是否只在评测期两遍、训练期用缓存。
