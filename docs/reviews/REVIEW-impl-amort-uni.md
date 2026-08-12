# 实现审阅 · PR-AMORT / unified-field / E1 attention 探针

> 审阅人：实现审阅 subagent（独立，与编码 agent 无关）
> 日期：2026-08-11 ｜ 分支：`lens-exp` ｜ 依据：工作区未 commit 的 diff（`git status` 见 §7）
> 权威文档：`WHERE_HEAD_REDESIGN_PROPOSAL/DELTA_2026-08-10.md`、
> `RESEARCH_amortized-oracle-fitting_2026-08-10.md`、`RESEARCH_unified-field-prediction_2026-08-10.md`、
> `experiments/.../amort_p1_20260810/NOTES.md §7`、`.../config/PREREGISTRATION.md`
>
> **【2026-08-12 编者注，不改本审阅记录正文】** 上行所列前三份文档均已不在 `docs/`：
> `WHERE_HEAD_REDESIGN_PROPOSAL/DELTA` 与 `RESEARCH_amortized-oracle-fitting` 由用户于 2026-08-11
> 手动清理（未留档）；`RESEARCH_unified-field-prediction` 于 2026-08-12 移入 `trash/docs/`（**禁读**），
> 其 VERIFIED 参考文献表存续于 `docs/REFERENCES_2026-08-12.md`。本审阅所审对象的**实验结论**见
> `docs/EXPERIMENT_INDEX.md` EPR-H06–H12（amort 系）与 EPR-H17–H20（unified 三案 Gate 0）；
> 唯一入口 = `docs/EXPERIMENT_INDEX.md`。本文档保留原文，作为「当时按什么规格审的」历史记录。

---

## 0. 结论速览

| 区块 | blocker | nit | 状态 |
|---|---|---|---|
| **P1/P3' 双臂**（训练循环 / 五项 loss / merger hook / checkpoint 门） | **7**（**5 项 URGENT**） | 10 | **不得开跑** |
| **FAFM / unifield Gate 0** | **9**（**2 项 URGENT**） | 15 | **不得开跑** |
| amort 诊断 E1/E2/E3/E5（已落盘） | 4 | 17 | 回溯，但 2 条动摇 NOTES §7 的交接结论 |
| E1 attention 探针（方案 B，已作废） | 3 | 11 | 回溯，1 条影响移交 C 案的资产 |
| merger 并列 hook（`hiddens.py` / `fpre.py`） | **0** | **0** | **pass** |
| **合计** | **23** | **53** | |

**如果只看一条**：`U1`（换主体分离项）——它把 NOTES §7.2 明令禁止的
「为 antonym 造分离损失」从 `ShuffleIndex` 侧绕了回来，实测 15.2% 的配对不可满足、
65.2% 的配对根本没换主体。**如果只看两条**：再加 `U5`（FAFM 的 Λ 度量项被欠权 16 倍，
本探针的核心机制近乎没开）。

**三个队列作业受影响，全部尚未上卡**（`q status` 2026-08-11T01:37）：
`AMORT_P1`(id 8, gpu0)、`AMORT_P3PRIME`(id 9, gpu1)、`FAFM_PROBE`(id 10, gpu0)，
均排在 C01/C02 之后。**本轮 blocker 全部可在开跑前修掉，零重跑成本**——这是把审阅
排在此刻的全部价值，请在 C01/C02 结束前处理。

全仓测试 `pytest q3vl/whereb/tests/ -q` → **508 passed**（96.9s）。
但 `q3vl/whereb/amort/` 与 `q3vl/whereb/fafm.py` **无任何单测覆盖**（`rg -l amort q3vl/*/tests/` 为空）。

---

## 1. URGENT ——会污染待跑臂的 blocker

### U1 【BLOCKER·URGENT】换主体配对分离项在实测 15.2% 的配对上不可满足，且其立论对 65.2% 的配对不成立

**位置**：`q3vl/whereb/amort/losses.py:142-153`（`paired_separation`）、
`q3vl/whereb/amort/data.py:274-289`（`AmortBatchBuilder._partner_gt_low`，**无任何配对合法性守卫**）

> 行号按 2026-08-11 01:5x 的工作区。编码 agent 在本次审阅期间仍在改
> `amort/data.py` 与 `run_amort_arm.py`（MaskResolver 线程安全、`RLIMIT_NOFILE`），
> 故以下一律**同时给出符号名**，行号漂移时按符号定位。

**规格**：NOTES §7.2 ——「`antonym` 保持只报不训……**禁止为 antonym 构造任何分离/互补损失**
—— 那会逼掩膜依赖颜色词，破坏一个当前通过的控制……这条是本轮最接近踩坑的地方，务必照办。」

**实现**：`L_sep = relu(margin − (d(m, gt_partner) − d(m, gt_own)))`，`margin = 0.05`，
`d` = 有效格上的平均绝对差，权重 `w.sep = 0.30`。partner 由
`ShuffleIndex`（`context.py:395`，按 `SHUFFLE_GROUP_KEYS = ("source_image_id", "render_mode")`
分组做 derangement）给出，**取到就用，不检查两个 GT 是否真的不同、也不检查主体是否真的换了**。

**实测**（V_where local，`ShuffleIndex(seed=0)`，16×16 网格，n=382 对；
复现脚本见 §8）：

| 量 | 值 |
|---|---|
| `d(gt_own, gt_partner)` 分位 | p0 **0.0000** ／ p5 0.0033 ／ p10 0.0259 ／ p50 0.2562 |
| **`frac(d < sep_margin=0.05)`** | **0.1518（58/382）** |
| `frac(d < 0.01)` | 0.0654（25/382） |
| `frac(d < 0.001)` | 0.0236（9/382，其中数对逐位为 0.0000） |
| **`frac(配对共享主体名词)`** | **0.6518** |

**两个独立的错误**：

1. **不可满足的 hinge**。最优解 `m → gt_own` 时可达到的优势上限恰为 `d(gt_own, gt_partner)`。
   凡 `d(gt_own, gt_partner) < margin` 的样本（**15.2%**），即使预测完全正确 hinge 仍激活，
   在两个 GT 相异的那少数格上留下一个不随收敛衰减的常幅 L1 梯度（权重 0.30）。
   这不是「训练慢」，是**在正确答案处仍被推离**。

2. **立论错误，且正是被明令禁止的那条**。`losses.py:36-39` 写道「只有真正解析出指令指称
   *哪一个* 主体才能降低此项」——**对 65.2% 的配对这是假的，它们指称同一个主体**。
   实测最接近的四对逐字如下（同一主体 + 两条颜色/影调方向不同的指令）：

   ```
   d=0.0000  own: "...transform the plant beside the pond into a dark, high-contrast..."
             par: "...brighten the plant beside the pond, give its texture stronger contrast..."
   d=0.0000  own: "...make the lizard noticeably brighter and more contrasty..."
             par: "...refine the lizard by making it moderately brighter, cooling its colour..."
   d=0.0000  own: "...give the elephant a bold, darker treatment..."
             par: "Warm and enrich the elephant in the center..."
   ```

   在这些配对上施加分离压力，**就是给 antonym 造分离损失**——NOTES §7.2 明文禁止的那件事，
   只是绕道 `ShuffleIndex` 进来的。而本臂自己的硬门 `gate.antonym_ok`
   （`evaluate.py:316`，`|Δ| ≤ 0.05`）正是检验这条的控制：**该 loss 项可以把臂训练成
   通不过自己的预注册门**。

**最小修法**（`ForeignIndex` 已经实现了同类守卫，照抄即可）：
`data.py:188`（`ForeignIndex.foreign_for`）对 foreign 候选做
`self.nouns.get(cid, set()) & own_nouns` → 拒绝；`_partner_gt_low` 缺同一守卫。建议两条一起加：
(a) partner 的主体名词集与自身**不相交**；(b) `d(gt_own, gt_partner) >= sep_margin`，
否则该样本本 step 不计 sep 项。并把 sep 项的实际覆盖率
（现有 `frac_with_partner`，`losses.py:238` 之外再加 `frac_with_usable_partner`）打进 `steps.jsonl`。

> 注：预注册失败画像 §3-4 已经预留了「分离项打不动 ⇒ 三元组 margin 形式无效」的解释口子。
> 按当前实现，即使该口子被触发也**无法区分**「形式无效」与「配对本身有 15% 是退化的」。

---

### U2 【BLOCKER·URGENT】两个预注册消融开关被解析后从未使用，消融行会静默跑成主臂配置

**位置**：`q3vl/whereb/scripts/run_amort_arm.py:100`（`--no-film`）、`:102`（`--train-context`）

**证据**：
```
$ rg -n "args\.(no_film|train_context|no_sim_field|center_prior_channel|no_semantic_head)" \
      q3vl/whereb/scripts/run_amort_arm.py
223:        use_sim_field=not args.no_sim_field,
224:        use_center_prior_channel=args.center_prior_channel,
225:        with_semantic=not args.no_semantic_head,
232:                  use_center_prior_channel=args.center_prior_channel)
```
`args.no_film` 与 `args.train_context` **零引用**。`AmortModel.__init__`
（`amort/model.py:34-52`）根本没有 film 开关形参；训练 context 在
`trainer.py:264-267` 由 `fake_prob` / `teacher_fraction` 硬编码生成。

**后果**：`PREREGISTRATION.md §4` 注册的三行消融——「−FiLM」、
「shuffle 指令训练天花板 `--train-context shuffled`」、「固定短语训练天花板
`--train-context fixed_phrase`」——跑出来会是**与主臂逐位相同的配置**，
却被当作消融结果发布。这是最坏的一类静默失败：数字完全合理，只是测的不是那件事。

**修法**：`AmortModel` 加 `use_film`（`ConvTower(film_from=...)` / `SemanticHead.film` 置空），
`AmortTrainer` 接受固定 context 模式；或者——如果本轮不打算跑这三行——**把两个 flag 删掉**，
不要留在 `--help` 里。

---

### U3 【BLOCKER·URGENT】checkpoint 选择整条链路未接线，预注册硬门不门任何东西

**位置**：`q3vl/whereb/amort/trainer.py:344-348`（`best()`）、
`q3vl/whereb/scripts/run_amort_arm.py:290-297`（终评）

**规格**：`PREREGISTRATION.md §2.1`「硬门（checkpoint 选择的前置，**三条全过才可选**）……
选优量 = median top-k IoU（local）。禁用 val loss。」

**两处问题**：

1. **`best()` 从未被调用**。`rg -n "\.best\(" q3vl/` 只命中 `q3vl/what/trainer.py:457` 与若干
   测试；`run_amort_arm.py` 里没有。`trainer.train()` 末尾 `self.save("final")`，
   随后终评（`:295`）直接在 **model 的最终态**上跑。也就是说交付板 = 最后一步的产物，
   与三条硬门无关；`eval.jsonl` 里逐 500 步的 `gate_pass` 记录**无人消费**。
2. **`best()` 自身把硬门软化**：
   ```python
   347:  pool = ok or self.state.checkpoints     # 无人过门 → 退回全体里选最好
   ```
   与预注册「三条全过才可选」直接冲突，也与本模块自己的 docstring
   （`trainer.py:17-19`「a checkpoint with no eval record is simply not selectable」）冲突。
   无人过门的诚实结果是「本臂无可选 checkpoint」，不是「在不合格里挑一个最好的」。

叠加 `max_hours=3.6` 墙钟停机（`trainer.py:251`），交付的是「墙钟到点那一刻恰好是什么」，
可能停在 cosine 中段。**红线「checkpoint 选择禁用 val loss」本身没有违反**（确实没读 val loss），
违反的是预注册的选择规程。

**修法**：终评前调用 `trainer.best()`，取不到（无 checkpoint 过门）就如实上报
「no selectable checkpoint」并按预注册判该臂未过门；同时把 `:347` 的 `or` 兜底删掉。

---

### U4 【BLOCKER·URGENT】P1 vs P3' 的 A/B 只对齐了预算，没有对齐步数

**位置**：`trainer.py:181-182`（`total_steps` 由数据量算，两臂相同）、`:251`（墙钟停机）、
`:282`（`scheduler.step()` 每 optimizer step 一次）；
`waves/amort_arm.sh` 两臂除 `ARM`/`GPU` 外参数逐项相同（seed 20260810、data、loss、`--max-hours 3.6`）——
**这部分是对称的，已核**。

**问题**：达成步数取决于吞吐。P1 的前向多一段
`predict_fields`（`model.py:104-112`，float32 + 关 autocast + 逐样本 `phi_dir @ w_dir` 矩阵乘），
P3' 是纯卷积路径（`model.py:122-131`）。若 P1 更慢，同一墙钟下它**步数更少、且停在 cosine
调度的更早位置**（`total_steps` 两臂相同，故 LR 轨迹按步对齐而非按时间对齐）。

`PREREGISTRATION.md §2.4` 声称「两臂同 seed、同数据、同 loss、**同预算**（`--max-hours` 相同硬墙）」，
并据此把差异**全部归因于 Phi 层**。等预算 ≠ 等训练量：若两臂步数不同，
「Phi 是资产/负债」的结论被吞吐混淆，而这正是这对臂唯一要回答的问题。

**修法（廉价）**：`save_steps=500` 两臂都存了盘，预注册改为「在两臂共同达到的最大 500 步
checkpoint 上做配对比较」；或加 `--max-steps` 并取一个两臂都能到的值。
无论选哪个，`stopped_reason` / `steps_trained` 已经在 board 里（`run_amort_arm.py:299-300`），
**终评时必须核对两臂是否相等，不等就不许直接比**。

---

### U5 【BLOCKER·URGENT】FAFM 的 Λ 度量项被欠权约 16 倍：全分辨率常数套在四分之一分辨率算子上

**位置**：`q3vl/whereb/fafm.py:45`（`SIGMA_MAX_SQ_ARM = 256.0`）、`:195`（消费）、
`q3vl/whereb/scripts/dump_fafm_cache.py:52`（`GUIDE_DIV = 4`）、
`q3vl/whereb/scripts/run_fafm_probe.py:160-174`（`make_apply_A(guide_q, ...)`）、`:230`

**链路（逐行核实）**：
```python
# fafm.py:41-45  常数的出处，注释自陈
#: Arm-wide constant ||A_I||_2^2, the median largest eigenvalue of A_I^T A_I
#: measured over 1000 S-train samples in uni_gate0 case B.
SIGMA_MAX_SQ_ARM = 256.0
# fafm.py:195
term_f = (apply_A(r) ** 2).flatten(1).sum(dim=1) / SIGMA_MAX_SQ_ARM
# run_fafm_probe.py:230
apply_A = make_apply_A(b["guide_q"], gh, gw)     # guide_q 存于 H/4（GUIDE_DIV = 4）
```
Gate 0 case B 的 `eig_max` 是在**全分辨率** guide 上测的：
`caseB_fafm/metrics.json` `spectrum.eig_max` median **256.034**、min **256.0002**——
从下方紧贴 256，正是「每个 16×16 粗格对应 256 个全分辨率像素」这个比值（常数模被 guided
upsample 保持，故 λ_max 恰等于每格输出像素数）。而训练里的算子从 `H/16` 升到 `H/4`，
比值是 **16**，不是 256。

**后果**：`Λ = λI + (1−λ)AᵀA/‖A‖²` 的第二项谱范数实际约 **0.0625** 而非 1。
`lambda_mix = 0.5` 时，§3.1 立为全案要点的「任务对齐度量」只贡献了约 **6%** 的预期权重——
本探针的核心机制近乎没开。**这是最值得在开跑前修的一条**。

**修法**：二选一——(a) 除以四分之一分辨率下重测的常数（约 16）；
(b) 在 `guide_q` 上重跑 case B 的 `eig_max` 普查并更新常数。无论哪条，
`SIGMA_MAX_SQ_ARM` 的 docstring 必须写明它绑定的**分辨率**，而不只是「arm-wide」。

---

### U6 【BLOCKER·URGENT】FAFM 判据 4 私自放松预注册阈值，且反向指令配对差分根本没实现

**位置**：`q3vl/whereb/scripts/run_fafm_probe.py:590-594`

**规格** §3.6 判据 4：「指令条件性：同图**反向指令**配对差分 ｜ 差分效应 **≥ 3× 全部负控制**，p<0.01」

**实现**：
```python
"pass": bool(all(d["delta"] > 0 and d["p_value"] < 0.01 for d in sel["deltas"].values())),
"line": "paired diff vs all three negatives > 0 with p<0.01",
```
`3×` 倍数消失，门槛变成 `> 0`；且全文件没有任何**反向/antonym**指令构造
（项目自带 `q3vl/whereb/context.py:340 antonym_context` 未被引用）。
`fafm_probe_20260811/NOTES.md §6` 却写「九条判据与 CFG 选择规则**逐条照搬**文档 §3.6」，
随后以放松后的形式复述判据 4 ——**预注册阈值被改，同时被描述为逐字照搬**。

---

### U7 【BLOCKER·URGENT】`winner_confidence == "low"` 未被排除：44% 的评测population 违反数据纪律，且同批样本进了 P1/P3' 训练

**位置**：`q3vl/whereb/scripts/run_amort_arm.py:167-168`（两处 `open_dataset(..., need_mask=True)`
**均未传 `exclude_low`**）、`q3vl/whereb/data.py:141`（`exclude_low: bool = False` 默认值）、
`q3vl/whereb/data.py:171`（真正的过滤点，无人触发）

**规格**：CLAUDE.md 数据纪律 ——「`winner_confidence=low` **不进 SFT 主训与评测 GT**；
可进渲染器预训练（D-RENDER）与伪标签。」
`q3vl/where/config.py:196-203` 更明确：low 的放宽是「**Where-A-only**：must not flow into
Where-B / What evaluation GT」，headline 只用 `normal`，low 单列成层。

**实测**（`amort_cache_20260810` 的 manifest，V_where local 400）：
```
Counter({'normal': 224, 'low': 176})        # 176/400 = 44.0%
```

**两个后果**：

1. **对已交付的四张诊断卡**：pooled 数字被 low 层系统性抬高（low 层 GT 面积更大 ⇒
   随机地板更高 ⇒ 每个 IoU 列都被灌水）。按 E5 的 `per_sample` 行回接 manifest：

   | 层 | n | `dot__full_softiou16` 中位 | 中心先验 | GT 面积 | 随机地板 |
   |---|---|---|---|---|---|
   | normal | 224 | **0.4055** | 0.4853 | 0.368 | 0.2254 |
   | low | 176 | 0.4823 | 0.5422 | 0.507 | 0.3391 |
   | pooled（**已上报**） | 400 | 0.4478 | 0.5088 | 0.410 | 0.2582 |

   E5 的注册门是 `≤0.40 ⇒ 从零训练`：**规定口径（normal-only）是 0.4055，离触发门只差 0.006**，
   而上报的 pooled 0.4478 稳稳落在死区里。E3 暴露更重——low 意味着**被选中的候选区域本身不确定**，
   `corr(输出, GT)` 因而按构造被压低，而 E3 的 M3 判决恰恰是 `corr(中心先验) > corr(GT)`。

2. **对两条待跑臂（这才是 URGENT 的原因）**：`train_idx` / `eval_idx`
   （`run_amort_arm.py:172-173` 附近）只按 `render_mode == "local"` 过滤，
   **low 样本既进训练、又进 V_where 评测 GT** —— 直接违反数据纪律的两条禁令。
   `amort/evaluate.py:154` 已经逐样本记录了 `winner_confidence`，但
   `summarise_rows` 从不按它分层，`strata` 只有 area/family/head。

**修法**：训练侧 `open_dataset(args.train_split, need_mask=True, exclude_low=True)`；
评测侧保留全量但**把 `winner_confidence` 加进 `strata`，headline 报 normal-only**，
预注册的 0.5088 / 0.2582 / 0.550 三个参照数也必须按同一口径重算——
否则本臂的「> 0.550 才算赢」是拿 normal-only 的分数去比 pooled 的门槛。

---

## 2. P1/P3' 双臂 · 其余 blocker 与 nit

### B1 【BLOCKER】一个样本没有 partner，整个 chunk（8 条）被判 uncovered

**位置**：`q3vl/whereb/amort/evaluate.py:105-113`

```python
try:
    inputs = builder.build(samples, [mode] * len(samples))
except KeyError:
    for s in samples:                       # <-- 整批，不是出错的那一条
        rows.append({... "uncovered": True})
    continue
```
`context_for(mode="shuffled")` 在**第一条**无 partner 的样本上抛 `KeyError`
（`data.py:342`，`AmortBatchBuilder.context_for`），`build` 随即整批失败。

**实测规模**：V_where 的 `ShuffleIndex(seed=0)` coverage = 0.9554（896 行），
其中 local 400 条里 **18 条无 partner**。`batch_size=8` 下最多有 **~144 条**被记为 uncovered，
而真正没有 partner 的只有 18 条。

**后果**：`swap_subject_delta`（`evaluate.py:264-266`）——NOTES §7.2 立为**一等训练信号与早警列**、
预注册 §2.1 立为硬门之一的那一列——是在一个**按索引相邻性挑出来的 ~2/3 子集**上算的，
`gate.swap_delta_ok` 亦随之。

**修法**：逐样本 build，或先过滤掉本 chunk 中无 partner 的样本再 build。

### B2 【BLOCKER】`SimFieldNorm.assert_in_domain` 不 assert

**位置**：`q3vl/whereb/amort/simfield.py:180-194`

docstring 自陈「Not a clamp: clamping is the silent failure the s-cache contract calls out……
**This reports and raises instead.**」——实际只在**非有限值**时 raise（`:191-192`）；
`frac_outside_tol` 算出来后**只 return，从不 enforce**。

s 缓存消费契约要求的「消费方必须声明自己期望的域，并**断言**生数据确实住在里面」，
在此**有其形无其力**。调用点 `data.py:298`（`_sim_field`）只是
`self.domain_reports.append(self.norm.assert_in_domain(raw))`，返回值从不检查。
缓解项：`AmortBatchBuilder.facts()` 把 `max_frac_outside_tol` 汇总进 `run_setup.json`，
事后可见——但「可见于一个没人读的 JSON 字段」正是契约里「第二种失败是静默的」所指。

**修法**：`frac_outside_tol` 超过预注册阈值即 raise（或至少打 `EARLY_WARNING`）。
写入侧（`fit_norm` 落 `raw_min/max/p001/p999` + `attn_implementation` + `checkpoint`）
与 `assert_compatible`（kernel/ckpt 不符即 raise，`:166-178`）**做得很好，保留**。

### P1/P3' nit

| # | 位置 | 内容 |
|---|---|---|
| N1 | `trainer.py:241` | `order = self.rng.permutation(...)` 立即被 `:243` 覆盖；死代码，但它消耗了 RNG 状态（给定 seed 仍确定，只是误导） |
| N2 | `evaluate.py:326` | `board["local_soft_iou_median"] = board["topk_iou_median"]`——名为 soft_iou、装的是 top-k hard IoU。`:322-324` 有注释说明，但对下游读者是地雷 |
| N3 | `evaluate.py:250,259` vs `run_amort_arm.py:295` | 在线 quick eval 的 `main` 是 `gt`，终评的 `main` 是 `generated`；在线门与终板打的不是同一个条件 |
| N4 | `run_amort_arm.py:250` | `quick_idx = eval_idx[:200]` 是索引前缀子集，非 sha1 规则族。且 quick eval 与终板同一 split——一旦 U3 修好接上 `best()`，就变成在报告集上选模型 |
| N5 | `amort/viz.py:116-117` | 注释「there are no pad cells in this representation; **assert it**」，实际是 `valid = torch.ones(...)` 的**假设**。结论本身为真（逐样本原生网格），但注释过度声称 |
| N6 | `amort/viz.py:112` | 裸 `except Exception: m_shuf = None` 吞掉换主体面板的一切错误 |
| N7 | `amort/heads.py:47` | `inv_bounded_sigmoid` / `param_shapes` 导入未用 |
| N8 | `amort/losses.py:224-227` | `aggregate` 把 foreign 样本（各项恰为 0）算进 `L_bce` 等的均值，日志值被按 `fake_prob` 稀释 ~15%。仅影响日志 |
| N9 | `amort/losses.py:215-217` vs `trainer.py:219` | 前者说面积比触发器「inside the first 20% of steps」可见，后者 `_warn` 恰好**在前 20% 内不报警**。文档自相矛盾 |
| N10 | `q3vl/whereb/amort/`、`q3vl/whereb/fafm.py` | 无任何单测。五项 loss 是预注册形式，至少 U1 的退化配对该有一条回归测试 |

### P1/P3' 已核实通过（pass）

- **merger 并列 hook（重点怀疑区）**：`hiddens.py` 用 `ExitStack` 把 `FPreHook`/`MergerHook`/
  `LastLayerHook` 挂在**同一次前向**（diff 已核，无第二次视觉前向，protocol 2.3 安全）；
  `want_merger=False` 默认关闭，既有 Where-B 臂逐位不变；`:220-225` 断言 merged grid
  恰为 F_pre grid 的一半。`fpre.py:MergerHook.split` 的行主序推理有据（`Qwen3VLVisionPatchMerger`
  折叠连续 `m*m` token），并显式说明**不得**调 `unshuffle_to_grid`；`:158-161` 有 token 数对账断言。
  **零额外前向、无静默回退，pass。**
- **IoU / dice 从未作为优化目标**：五项 loss 全部为 BCE_soft / SDF / 面积带 / 空掩膜 / 平均绝对差三元组
  （`losses.py:106-153`），`iou_as_target: False` 亦落进 `config/loss_preregistration.json`。
- **antonym 无任何损失项**：`losses.py:13-22` 明文说明，全仓 grep 无 antonym 损失。
  （**但 U1 从 `ShuffleIndex` 侧绕回来了，见上。**）
- **先验场只作条件输入**：中心先验通道默认关（`model.py:48`）、是消融开关；
  相似度场只进 `_extra` 通道，从不做目标或择优量。
- **禁 AUC**：`amort/` 全包无 AUC 计算。判据列为 matched-area top-k IoU + grid 边界 F1 +
  中心先验列 + 随机地板 `a/(2−a)` + area/family 分层（`evaluate.py:172-232`），齐。
- **无坐标通道/位置编码**：`heads.py` 全程 `padding_mode="reflect"`，无 coord-conv、无 PE
  （`heads.py:15-21` 引 DELTA §5.6）——这是对 E3 实测「corr(输出,中心先验) 0.64 > corr(输出,GT) 0.47」
  的结构级免疫，做得对。
- **可视化纪律**：`amort/viz.py` 全部面板固定 `FIXED = (0.0, 1.0)` 色标，叠图走
  `overlay_grid_on_image` → `grid_to_img` 严格逆映射 + 整数倍 `np.repeat`，无 resize。
- **归一化用整臂常量**：`fit_norm`（`simfield.py:208-229`）在 256 个训练样本的**全部格**上取
  median/MAD，逐图归一化零出现。
- **只训 local**：`run_amort_arm.py:148-149` 按 `render_mode == "local"` 过滤，global 不进训练。
- **V_where 不进训练**：train/eval 两个 dataset 分别 open，无交叉。
- **`oracle.py` 的 `ridge_lambda`**：默认 `0.0`，且注释写明「published-oracle setting，must stay
  the default」；惩罚落在 `w_eff = alpha·w_dir` 而非 `w_raw`（后者是纯规范自由度）——数学正确。

---

## 3. amort 诊断 E1/E2/E3/E5（已落盘，回溯）

> 四张卡均已交付。本节 blocker 不阻塞在跑作业，但 **E3-a 与 E5-a 直接影响 NOTES §7 的两条交接结论**
> （M3 定案、P3 证伪），因而影响正在排队的 P1/P3' 的立论基础。

### 焦点项裁决（对应任务卡）

| 项 | 裁决 |
|---|---|
| (a) E1 与附录 A 的数学 | **对**。`s = S·tanh(q/S) ⇒ ds/dq = sech²(q/S)`，`dq/dw_dir = α·Φ`，故 `J = (dm/ds)·sech²·α·Φ`、`JᵀJ = ΦᵀΛΦ`，与实现逐项一致；`dm/ds` 由 autograd 在逐元素 readout 上取。与规格「逐图 WLS」的偏离已如实登记 |
| (b) E2 是否真重算 | **是**。经 `FitPool` 带 `ridge_lambda` 重跑完整多起点 L-BFGS，惩罚项是本轮新增且**位置正确**（罚在 `w_eff = α·w_dir`，不是规范自由的 `w_raw`）；`mult=0` 复现已发布天花板（0.9748 vs 0.9737），是正确的对照。问题在缺 `status`/`flags` 过滤，不在复用 |
| (c) E3 勘误后的 `antonym_invariance_pass` | **语义正确**。小 `\|gt − antonym\|` 记为 PASS、verdict 里 M4 已撤、正确的 M4 探针改挂到换主体档（`conditioning_responds_to_subject_swap`）。REPORT §2(iii) 的数字走的是勘误无关的代码路径，**可复现**。残留两条见 nit（阈值口径借用、交付 JSON 手改） |
| (d) E5 作为「照抄先验」判别器 | **判别的那一半站得住**：`fit_iou_vs_prior`(0.92/0.92/0.98) 与 `decode_minus_prior` 配对 Wilcoxon(+0.005/0.000/0.000) 干净地把「D 丢信息」与「先验才是瓶颈」分开，且有零信息对照臂 + `a/(2−a)` 地板 + area 分层。**§3.2 结论成立**。但 **§3.3 的 P3 那一半不成立**，见 E5-a |

### E3-a 【BLOCKER】grid 边界 F1 被 import、被传参、从不调用

`run_amort_e3.py:120`（import）、`:244`（作为最后一个位置参数传入）、`:293`（出现在签名里），
而函数体（`:294-395`）**无任何调用**。E3 的判据块（`:347-351`）只发
`softiou_output` + `softiou_centre_prior` + corr 列。红线：「空间场一律用这三列，**缺一不可**」。
E3 REPORT §2(ii) 那一行也确实没有边界 F1。（E5 做对了。）
那个死参数就是「本来要做、后来掉了」的物证。
**修法**：`:343-346` 的 `iou_g`/`iou_c` 循环里已经握着二值化掩膜，补两次调用即可。

### E5-a 【BLOCKER】`ridge_lsq` 不是 P3 那一层；「P3 预注册证伪已触发」的结论不成立

`run_amort_e5.py:346` + `:356-362`。两处互相叠加的错误规格：

1. **链接函数错**。`:346` 解的是 `A c ≈ logit(prior)`，即假定解码是 `m = sigmoid(q)`。
   真实解码是 `s = 3·tanh(q/3)` 之后 `apply_readout("band", s, rho)`，而
   `_band_apply`（`q3vl/where/readout.py:99-106`，**已逐行核实**）是一个**凸包/bump**：
   ```python
   b = torch.sigmoid(k * (z - mu + h)) - torch.sigmoid(k * (z - mu - h))
   return pi * b + (1.0 - pi) * (1.0 - b)
   ```
   它对 `s` **非单调**，且在已发布的 latent 上实际是**随 q 递减**的。实测复现
   （3 个样本，直接取 `amort_cache_20260810` + 已发布 band latent）：
   ```
   sft_00680d9673  corr(logit(m), q) = -0.9284   m(s) 在 s∈[-3,3] 上单调? False
   sft_0116bd2d61  corr(logit(m), q) = -0.7031   单调? True（μ=-5.14，落在 s 域外）
   sft_01f6c38e06  corr(logit(m), q) = -0.9839   单调? False
   ```
   **负相关意味着闭式解把 `q` 往反方向推**。另外 `logit(prior.clamp(1e-3,1-1e-3))` 跨度 ±6.9，
   而 `s` 在 ±3 饱和，目标域一半以上不可达。
2. **ρ 是借来的**。`:356-362` 用 `lat.rho` 构造 ridge latent，而那个 ρ 是多起点 L-BFGS
   **为另一个 `w`**（`full_fit` 解）拟合出来的。脚本 docstring（`:39-41`）声称
   「the closed form … with an **arm-constant** `rho`」「this is literally the layer P3 embeds」——
   两句都不成立。`rho_pool`（`:343`）——本该产出那个整臂常量的收集器——是**死代码，从未被读**。
3. 另：规格的 P3 层是 `w = (AᵀΛA + εI)⁻¹AᵀΛ ŷ`，E5 把 Λ 整个丢了（`:348 G = A.T @ A`），无说明。

**后果**：E5 REPORT §3.3 据以宣告 P3 被证伪的 −14.8 / −8.9 / −29.2 点
`full_fit → ridge_lsq` 落差，按此证据**最可能是链接函数 + ρ 失配的产物**，
不是闭式投影的性质。而 NOTES §7.3 正是引这一条来砍掉闭式投影、改立 P3'。
**这不否定 P3' 作为一条臂的价值**（直出场本身合理），但它否定
「闭式投影已被证伪」这句话的证据地位——C 案/What 侧若要复用该结论，必须先修。
**修法**：经真实 readout 反演（或在 `s` 域对 band 的逆做回归）、配一个**真正整臂常量**的 ρ，
再重读 §3.3。

### E5-b 【BLOCKER】E1 交付的 `metrics.json` 含仓库里无代码可产出的 `w_raw` 块

`run_amort_e1.py:192-203` 的 `oracle_w_star` 只发
`{n, dim, per_dim_std, per_dim_mean_abs, per_dim_std_summary, mean_norm, note, alpha, w0}`，
全部由 `w_dirs`（`:135`, `:170`）算出。而交付的
`amort_e1_20260810/metrics.json:261,407,408` 另外带着
`w_raw_per_dim_std` / `w_raw_per_dim_mean` / `w_raw_cv_frac_gt1` / `note_w_raw`；
全仓 `rg` 这些键**只在那个 JSON 文件里出现**。`run_amort_e1.py` 甚至从不从 oracle 记录里读
`w_raw`（`:109` 只读 `lat["w_dir"]`）。

这不是装饰数字：E1 REPORT §1 的头条行（`w_raw` 逐维 std **1.146**、|mean| 0.108、
**CV 中位 10.66**、100% 维度 CV>1）与「没有一个维度的跨样本均值能盖过它自己的散布」
全靠它们，且 `run_amort_e2.py:41-43` 引它作为 E2 整个 CV 列的动机。
按存下的列表回算，数值**内部自洽**（CV 中位 10.6598、frac>1 = 1.0），所以数字大概率是真的——
但**被审阅的脚本无法复现被交付的产物**，且 E1 `NOTES.md` 没有披露第二次运行。

**加重情节**：`q3vl/whereb/amort/` 与 `q3vl/whereb/scripts/run_amort_*.py`
**全部未进 git**（逐个 `git ls-files --error-unmatch` 均失败），而每张卡都把
`git rev-parse HEAD` 当 provenance 记下（`run_amort_e2.py:230`、`e3.py:265`、`e5.py:482`，
外加四份手写的 `config/git_commit.txt`）。**记下的 commit 复现不了其中任何一次运行。**
（`run_amort_arm.py::_provenance` 已经把 dirty flag + 变更文件表 + amort 包内容 sha256 一并记下，
是正确范式——四张诊断卡应回填同样的东西。）

### E5-c 【BLOCKER】E2 的验收门把 Where-A 协议判为 rejected 的拟合算了进去

`run_amort_e2.py:174` 只按 `fr.latent is None` 过滤。而
`FitResult.usable`（`q3vl/where/oracle.py:257-260`）是 `status == "ok" and latent is not None`，
`fit_latent` 在 `loss > FIT_REJECT_LOSS = 0.90` 时置 `status = "rejected"`（即 softIoU < 0.10）。
交付的 `metrics_stageA.json` 里 `mult_10` 的 `oracle_soft_iou.min = 0.0118`（loss 0.988）、
`mult_3` min 0.0402——这些行是 `rejected`，却在 `n: 400` 的聚合里，而门
`gate_iou_ge_0.95` 就是从该聚合读的。另 `alpha.min` 在 `mult_10` 低到 **1.5e-43**
（vs `FIT_REJECT_ALPHA = 1e-6`），`fit_latent:410-411` 会置 `alpha_collapsed`，E2 从不读 `fr.flags`。
`oracle.py:258-260` 的 docstring 明写只有 `usable` 的拟合「may drive a gradient or **a ceiling number**」。
基于中位数的判决大概率仍成立，但门是在混合 population 上算的，且**无拒绝率普查**（协议 10.2 要求）。

### amort 诊断 nit（18 条，按重要性）

| # | 位置 | 内容 |
|---|---|---|
| NA1 | `run_amort_e3.py:387-388` | `antonym_invariance_pass` 语义对、**口径借用**：预注册量是逐样本配对的 median `\|Δ grid_hard_iou\|` 配 `ANTONYM_INVARIANCE_MAX = 0.05`（`metrics.py:284,322`；`config.py:250`），而 `:388` 把同一个 0.05 套在 `median(mean\|场差\|)/整臂 mean\|场\|` 上——不同的量、不同的单位，且用 `<` 而注册写的是 `<=`。两种读法都以很大裕度通过（0.0031/0.0059 vs 0.05；board 的 `median_abs_delta = 0.0`），**无结论移动**；建议改名为 `antonym_field_delta_below_5pct` 并单独引 board 列 |
| NA2 | `amort_e3_20260810/metrics.json:121,234,239` | 交付件相对代码被**手工编辑**：带 `M4_no_conditioning__RETRACTED: true` 与 `ERRATUM_2026-08-10` 块，而现脚本两者都不发、且会发两个 JSON 里没有的键。重跑会产出**不同的文件**。`NOTES.md:50-51` 已披露，REPORT §2(iii) 的数字也确实来自未改动的路径——建议重跑（9 分钟，无需排队）让产物与代码对齐 |
| NA3 | `run_amort_e3.py:331-351` | 与中心先验列**无配对 Δ + p 值**，只有 `agg()` 汇总和一个不带检验的配对差 `corr_centre_minus_corr_gt`。红线要求任何中心先验比较必须出示配对 Δ 与 p；而 REPORT §2(ii)/§4-2 恰恰在主张「比中心先验高 +0.033」并提议把 `corr(out,centre) > corr(out,GT)` 立为 P1 门。`cc/cg/iou_g/iou_c` 已逐样本对齐，`attnprobe.paired_wilcoxon`（E5 用得对）只差一行 import |
| NA4 | `run_amort_e3.py:345-346`、`run_amort_e5.py:329-333` | 名为「soft-IoU」的列其实是 hard IoU：`soft_iou_value(topk_mask(f,k), (gt>0.5))` 两侧都是二值，恒等于 `hard_iou`。E5 自己的 JSON 就是证据：`full_fit_softiou16 = 0.44778502035` vs `full_fit_hardiou16 = 0.44778502069`。E5 另有真软列（`..._unthresholded`，中位 0.368）但 REPORT 头条用的是阈值化那个；**E3 则完全没有软列**。三条强制列里有两条塌成了一条 |
| NA5 | `run_amort_e5.py:278` | 逐图标准化，与本文件 `:213-214` 自称的整臂口径矛盾：`sigmoid(gain*(cp16 − cp16.mean())/(cp16.std()+EPS))`。危害有限（中心先验是网格形状的确定性函数，不含图像内容，没有跨样本变异被毁），但它字面上是一个逐图 squash，出现在一个写着「Explicitly not per image (s-cache contract red line)」的脚本里 |
| NA6 | `run_amort_e2.py:145` vs `run_amort_e1.py:116,128` | 两个不同的量都叫「effective cond」：E1 用 `effective_gram_cond`（先丢零范数列再对子 Gram 取 `cond`），E2 用全 71 维上的 `λ_max/最小正特征值`，而其**ridge 后**的数（`:191-194`）又换成 `ev.max()/ev.min()`。E2 REPORT §1 把 8.414e5 vs 8.36e5 的差解释为「E1 的 Λ 含 α²」——**这个解释是错的**：α² 是逐样本标量乘 Λ，在条件数里**精确抵消**。真因是上述两种口径 + 样本过滤不同（`e1:106` `fit.get("usable")` vs `e2:125` `status != "ok"`） |
| NA7 | `run_amort_e2.py:152-153` | λ 标定从不与其目标核对：`lambda_0 = lmax_med/(2·target_cond)` 假定 `λ_max` 重拟合后不变，但 α 在惩罚下从中位 2.30 塌到 0.057，于是 `mult=1` 实际达成的 cond 是 **6.67e4**，是所述 1e3 目标的 **66 倍**，而 JSON 里 `target_cond: 1000.0` 就挨着它、无任何限定语 |
| NA8 | `run_amort_e2.py:213-222` | `published` 块里没有 `oracle_soft_iou`。本卡自己的风险陈述是「先在 oracle 侧验收」，`mult_0`（中位 0.9748）**是**对 E1 已发布 0.9737 的有效复现对照，但只读 `metrics_stageA.json` 的人做不了这个比较 |
| NA9 | `run_amort_e1.py:154-166` | `cond_full` 被丢弃（`phi.py:156-171` 特意保留可能无穷的全值「for transparency」），且 `structural_dropped`/`effective_dropped` 从不汇总。全谱 `n_near_zero_eig`（`:130-133`，在未丢列的 71×71 `H` 上算）盖住了缺口，故判决安全——但注意两条门腿按构造近乎冗余：`near_zero_rel = 1e-8` 时 `n_near_zero > 0` 基本等价于 `cond > 1e8`，故 `no_near_zero` 永远否决不了一个 `cond < 1e2` 的通过 |
| NA10 | `run_amort_e1.py:178-180` | 过度声称：`what_was_measured.effective` 说这是「the fit the oracle actually solved」的 GN Hessian，但 `AᵀΛA`（`Λ = (dm/ds·sech²·α)²`）是 `m` 上**平方误差**损失的 GN Hessian，而 oracle 最小化的是 `1 − soft_iou_minmax`（`config.py:165`），其 GN Hessian 残差权重不同。紧邻的 `caveat` 字段（`:181-184`）与 NOTES A2 是诚实的，这一行不是 |
| NA11 | `run_amort_e1.py` | 无 viz、无 config/env 快照：脚本只写一个 JSON（`:231`），`amort_e1_20260810/viz/` **空目录（0 文件）**，`config/{environment.json,git_commit.txt}` 是手工拼的。交付物规范要求 `viz/failure_*` 与含 seed + git commit + 环境的 `config/` 快照；另外三个脚本都写了 `config/run_setup.json` |
| NA12 | `dump_amort_cache.py:104-122` | 缓存 manifest **不声明 domain、不记 dtype、不记注意力 kernel**。`merger_out` 存成 float16（`:107`）而不记范围；实测域 `merger_out [-11.69, +14.06]`、`semantic_low [-270.5, +235.6]`、`img_low [0,1]`——全未记录，四个消费者也全都不断言。kernel 按项目自己的分析是要紧的（`amort/simfield.py:21-28` 记录 eager-vs-sdpa 的 merger 漂移 rel-max 0.12），`:64` 硬编码了 `eager` 却不记。实际风险低（无 fp16 溢出、全程 eager），但写入侧的「meta.norm 必填 domain」未实现 |
| NA13 | `run_amort_e5.py:456-457` | 只要任一臂缺一个键，配对差分块就被**静默跳过**（`if v.size == len(rows_out)`），会不留痕迹地丢掉强制的中心先验配对 Δ。本次恰好全过（400/400 `ok`），但一次 rejected 拟合就能抹掉那条红线列 |
| NA14 | `run_amort_e5.py:466-471` | `best_arm_median_softiou16` 在包含 `centre_prior` 的 `ARMS` 上取 max，即零参数对照可以被报成「最佳臂」（实际就是：0.5163 vs dot 的 0.4478）。门的布尔量正确地只看 `dot`，但这个汇总字段招人误读 |
| NA15 | `run_amort_e3.py:146` | 未记 `shuffle_coverage`。E3 用 `ShuffleIndex(ds.shuffle_records(), seed=0)`，与 `run_where_b.py:299` 一致；但 E3 把已发布 board 的行折进它的 (iii) 读数，而 board 侧存了 `shuffle_coverage`、E3 的 `metrics.json` 什么都不存——两者的 partner 映射是否同一，从产物上**无法验证** |
| NA16 | `run_amort_e3.py:433`、`run_amort_e5.py:549` | 中心先验可视化面板**自动缩放**：其它每个掩膜/场面板都钉了 `vmin=0, vmax=1`，唯独 `imshow(center_prior_field(...))` 不传上下限，matplotlib 于是对一个空间场做逐图 min-max。实质无害（逐网格形状确定），但与 `:401-402`/`:516-517` 的 docstring 相矛盾 |
| NA17 | E3 | 无自有 Δ_const 列：`iii_conditioning` 只带 `gt_vs_shuffled`。REPORT §3 的 `null`/`fixed_phrase`/`irrelevant_words` 是从已发布 board 借的，§2(ii) 的随机地板列是从 E5 借的。可论证（同一批 400 样本），但 E3 的 `metrics.json` 对「每个消融行必带 Δ_const/Δ_shuffle」而言**不自足** |
| NA18 | `amort/simfield.py:180-194` | 与 §2-B2 同一条（`assert_in_domain` 不 assert），此处独立复现，不重复计数 |

### amort 诊断已核实通过（pass）

- **禁 AUC**：六个文件 `rg -ni 'auc|roc'` 只命中 `AutoProcessor` / `subprocess` 的子串。
  无 AUC 计算、无 AUC 排序、无 AUC 上报；E5 docstring `:49`「No AUC anywhere」属实。
- **IoU 非优化目标（带书面豁免）**：E2(`:163-170`)/E5(`:290-293,:303-307`) 确实在
  `objective = soft_iou_minmax` 上跑 L-BFGS——但那是从 Where-A oracle 原样继承的 `FIT_OBJECTIVE`，
  其豁免在 `q3vl/where/config.py:157-166` 有书面裁定（「protocol 5.5 makes soft-IoU the Where
  main loss outright」）。两个脚本都没覆盖它，改了反而破坏与已发布 0.97 天花板的可比性。
  E5 的拟合目标是**先验**、绝非 GT（`:290-293`），GT 只作度量进入。
- **场/s 无逐图归一化**（除 NA5 一行）：E5 先扫一遍全臂所有格算 center/scale（`:213-224`）
  再当常量用（`:226-228`）；E3 的相对 L1 分母是整臂均值（`:366`）；E3 的 `corr()` 按构造尺度不变。
- **checkpoint 选择不用 val loss**：E3 载入 `where_b_final.pt`（`:178`）即末态，无选择步；
  六个文件里无任何 `best_*_loss`/`val_loss` 选择。
- **antonym 只报不训（重点 grep）**：amort 包内 `antonym` 只出现在
  (a) context 构造器、(b) 上报的不变性列、(c) `run_amort_arm.py:283` 的
  「reported only; never a loss term (E3 erratum)」provenance 串、(d) `losses.py:13-22` 的常驻说明。
  训练循环的 mode 选择只可能产出 `foreign`/`generated`/`gt`——**`antonym` 从损失路径不可达**。
  分离项用的是换主体 partner，正是勘误裁定要提拔的那个。（**但见 U1：换主体 partner 自身没做守卫。**）
- **先验场不作优化/择优目标**：E5 按注册设计拟合**到**先验（规格 §4 E5 行明写），
  结果只当零训练基线/`w₀`，从不作择优判据；全仓无 `‖D(w) − prior‖` 形式的选择规则。
- **切分用 sha1 规则族**：六个文件 `rg 'random.shuffle|train_test_split|np.random.(shuffle|permutation)'`
  **零命中**；四张卡全部经 `open_dataset` 消费已发布的 `V_where`。路径上唯一的 shuffle 是
  `ShuffleIndex`（分组内带 seed 的 derangement，是控制配对不是切分），E3 用 `seed=0` 与
  `run_where_b.py:299` 一致。`--limit` 取未打乱前缀，但四次交付都用了全量 400。
- **V_where 不进训练**：六个文件都没有可学头的训练循环——E2/E5 是逐图 L-BFGS latent 拟合
  （无跨样本共享参数），E3 是 `torch.no_grad()` 纯前向，E1 是纯线性代数。
- **pad 格**：`dump_amort_cache.py:84` 用 `do_resize=False` + `grid_from_geometry`，
  网格是原生长宽比（512×768 → 32×48），该路径**无 `expand2square` 填充**，无 pad 格可排除。
- **叠图不用 resize**：两个 viz 函数都不把场叠回原图，所有面板是独立场渲染；
  `to_common`（`run_amort_e3.py:78-87`）是场到场的面积重采样、用于跨样本统计，
  docstring 明确它不是 viz 路径。

---

## 4. FAFM / unifield Gate 0 · 其余 blocker 与 nit

> 三案 Gate 0 已交付（`uni_gate0_20260811/`），FAFM 探针（`fafm_probe_20260811/`）**在队列中未开跑**。
> U5/U6 见 §1。

### 焦点项裁决（对应任务卡 (a)–(e)）

| 项 | 裁决 |
|---|---|
| (a) `c*` 构造 | **部分**：公式与 §3.2 一致，但被求值的算子有三种分辨率并存（见 U5 与 N-F1） |
| (b) 九条判据 | 九条全部实现、无一硬编码为 pass；但 #4 阈值被放松（U6）、#6 分母口径错（B3） |
| (c) `0.8949` 分母 | **已接线**（不只出现在散文里），但**接到了错的口径**——见 B3 |
| (d) posterior 按形状分组修复 | **完整**。`run_fafm_probe.py:462-474`（`:278-281` 同构）逐 shape group 全量处理，`data.batch` 按构造即形状同质，该路径上无 `try/except: continue`、无形状守卫、无未计数过滤 |
| (e) CFG 规则 | **pass**。dropout 率、无条件 token 构造、guidance 施加点三项与 §3.3/§3.6 逐条相符 |

### B3 【BLOCKER】判据 6 用全分辨率软场天花板去归一化 grid top-k 软 IoU

`run_fafm_probe.py:53-56, 607-621, 677`。`GATE0_FAMILY_CEILING["semantic"] = 0.8949`
取自 Gate 0 的 `softiou_hi_by_family`（**未阈值化的全分辨率软场**列），而判据 6 的 `measured`
是 `by_group(rows_a, "family", "soft_iou")`，其定义为
`soft_iou_value(topk_mask(pred16, kk), gt16b)`（`:359`，**grid 级、二值化、面积匹配 top-k**）。
两者是不同的量。按交付的 `caseB_fafm/metrics.json`，判据 6 实际所在列的 Gate 0 天花板是：

| family | `softiou_hi`（0.8949 的出处） | `grid_softiou`（#6 实际测的列） |
|---|---|---|
| linear / band / radial | 0.9990 / 0.9971 / 0.9937 | 1.0000 / 1.0000 / 1.0000 |
| **semantic** | **0.8949** | **0.9888** |

故 `frac_of_ceiling`(`:617`) 与 `neck_tax = 1.0 - ceil`(`:618`) 对**每个 family 都是错的**。
semantic 在判据 6 口径下的 neck tax 是 **0.0112，不是 0.1051**——
「替冻结算子征收的 0.105 税买单」这条协调裁定的叙事，在它被并排打印的那一列上**量级反了**。
`:677` 的 `"gate0_semantic_ceiling": 0.8949` 同病。
（过线判定 `beats_centre_prior`(`:620`) 不受影响，与 §3.6「每个家族 A ≥ B」一致。）

### B4 【BLOCKER】消费侧无 `c*` 域断言

§3.2 末句：「值域纪律：`c*` 的整臂实测值域必须落盘声明（本项目 s 缓存消费契约）；
**消费侧断言后才进训练**。」

写入侧合规：`dump_fafm_cache.py:265-276` 落 `domain` / `p01,p50,p99` / `frac_below_0` + 消费建议；
`run_uni_gate0_b.py:254-268` 同。**消费侧不合规**：`FAFMData.load`(`:102-114`) 读出 `cstar`
直接喂 `train_arm`；`:704` 的 `"cstar_domain_train": train.manifest.get("cstar_domain")`
只是把声明**抄进** `metrics.json`，从未检查张量真的住在里面。这正是契约点名的第二种静默失败。

### B5 【BLOCKER】三条负控制显著弱于项目正规版，且其中两条塌成同一条

`run_fafm_probe.py:63-65, 284-291`：
```python
FIXED_PHRASE = "edit the image"
IRRELEVANT   = "keyboard volcano penguin spreadsheet"
roll = instrs[1:] + instrs[:1]; instrs = roll        # :286
```
对照 `q3vl/whereb/context.py`：`FIXED_PHRASE_TEXT = "the main subject"`(`:62`)——
**刻意选的是红线自己的那个已知强基线**；`irrelevant_words_context`(`:303`) 逐样本
从 `IRRELEVANT_VOCAB` 抽 12 词并按样本 seed 变化，**正是为了不做成常量串**
（「a single constant string is the *other* control」）；`ShuffleIndex`(`:395`)
是分组 derangement 且如实上报 uncovered。探针把三条全换掉：固定短语丢了主体先验（更好打）、
`irrelevant_words` 变成**第二条常量串**、`shuffled` 变成无守卫的批内 rotation
（形状组大小为 1、或相邻两条指令雷同时，静默产生一条空负控制）。
**判据 4 是本卡唯一的条件性证据，而它整个立在这三条上。**

### B6 【BLOCKER】消融臂无 Δ_const / Δ_shuffle 列

`run_fafm_probe.py:661-673`。`summary()` 只出
`soft_iou / hard_iou / gbf1 / area_ratio / by_family / by_area_stratum`。
`arms` 表（`A_fafm` / `B_centre_prior` / `C_regression` / `D_no_text` / `F_k1` / `E_*`）
就是一张消融表，而只有臂 A 有配对差分（且只在 `crit["4"]` 内），C/D/F 两列皆无。
红线速查：「每个消融行必带 Δ_const/Δ_shuffle 列」。

### B7 【BLOCKER】§3.5 的三条相似度场负控制完全未实现

§3.5：「配套负控制预注册：**打乱 S 通道、置零 S、错图 S** 三条」。
代码只有训练期 `p_drop_sim = 0.1`（`fafm.py:59`，施加于 `run_fafm_probe.py:224-227`），
评测期**没有任何 S 负控制臂**——`eval_arm` 的 `negative` 形参只改写指令（`:285-291`）。
S 是携带强主体/覆盖先验的 concat 通道，故「场跟的是指令而不是 S」目前**没有任何控制支撑**。

### B8 【BLOCKER】探针零可视化

`run_fafm_probe.py:492` 声明 `--n-viz`（default 6），此后**再无引用**（已核：`args.n_viz`
全文件仅此一处）；`:499` 建了 `viz/` 目录，无任何写入。
CLAUDE.md 交付物规范要求 `viz/success_*` 与 `viz/failure_*`，并写明
「没有失败案例 = 没找够，审阅直接打回」。三张 Gate-0 卡都有可用的 `_write_viz`
（`run_uni_gate0_a.py:651`、`_b.py:309`、`_c.py:713`），探针独缺。

### B9 【BLOCKER】`run_uni_gate0_a.py` 变量遮蔽，E0a 探针幅度对 256 个样本中的 255 个是错的

`run_uni_gate0_a.py:363, 371-372, 390, 405`：
```python
363:  scale = float(torch.median(w_eff_all.norm(dim=-1)))   # 整臂常量 = 2.143
371:  w1 = (w1 / w1.norm() * scale).to(dev)                 # 样本循环内
390:      scale = max(float((v[2] - v[3]).norm()), 1e-30)   # 在 stage 循环里遮蔽了它
405:  "w_scale_used": scale,                                # 报的是最后一次被污染的值
```
第 1 个样本之后，`scale` 是上一张图的 `‖m[2]−m[3]‖`。交付的
`caseA_chnpe/metrics.json` 里 `E0a.w_scale_used = 38.648`，而
`arm_constants.w_eff_norm_median = 2.143`——**18 倍过驱动**。
`D_I` 只在 `s = S_SCALE·tanh(q/S_SCALE)` 之前是仿射的，残差幅度随尺度变，故非纯装饰：
样本 0（唯一按正确的 2.143 探的）`s_rel = 0.2109`，恰等于 `E0a.stage_s_after_tanh.min`，
而总体中位是 0.9053。**E0a 的 `fail` 结论（→「线性分支永久退役」）大概率仍成立**
（0.757 ≫ 0.01），但 REPORT 里那个数不是预注册所定义的测量，**引用前必须重跑 E0a**。

### FAFM / Gate 0 nit（15 条，按重要性）

| # | 位置 | 内容 |
|---|---|---|
| NF1 | `dump_fafm_cache.py:149-163` | `c*` 在**半分辨率**算子上解（`SOLVE_DIV = 2`），`RIDGE_EPS` 仍为 `1e-3`；`AᵀA` 随输出像素数缩放，故 ε/Gram 比实际紧了 ~4×。等价性有实测（manifest `reduced_solve_equivalence` mean 1.21e-4），但 ε 未随之改标，文档亦未说明 |
| NF2 | `run_uni_gate0_c.py:580-581,666,671-673` | 轮廓门读的是 `softiou_hi_best_variant = max(两个解释器变体)`，即**逐样本按 soft-IoU 择优**——择优路径上的 IoU（红线 2 边缘）。本轮实际效应恰为零（`best_variant.mean` 与 `nolposs.mean` 逐位相同），故不判 blocker；但**不得带进两变体会互换名次的卡** |
| NF3 | `run_fafm_probe.py:402-421,638-639` | 判据 7 的「最近 `c*` 邻」实现为逐场 L2 归一化后的**余弦**相似度，丢掉幅度；参照库取 manifest 前 3000 行（索引序）而非随机抽 |
| NF4 | `run_fafm_probe.py:654-659` | 判据 8 只记 k=4 下 median soft-IoU vs N。§3.6 要的是步数-**多样性**曲线，其读法（「N=1 与 N=8 无差 ⇒ 多峰假设弱」）说的是多样性不是精度；没有任何多样性统计量落盘，该判据按现输出**读不出来** |
| NF5 | 全局 | §3.3/§3.6 的「另报」诊断缺失：R2（CNF 精确对数密度 + Hutchinson 散度择最大密度）与 **R1/R2 分歧率**、R1-medoid vs oracle-best-of-16 落差、ambiguity↔IoU 相关、逐 family 的 `c*` 批内有效出现率、family 平衡重采样开关臂。现只有 R1（`fafm.py:239`）与 `ambiguity` 汇总（`:673`） |
| NF6 | `run_fafm_probe.py:231-240` | 臂 C 用 `lambda_metric_loss` 训练，§3.6 写的是「一次性 **L2** 回归」。用同度量可论证更公平，但偏差未登记 |
| NF7 | `q3vl/whereb/metrics.py:233` | `paired_delta` 是配对均值的符号翻转置换检验；§3.6 #1 写「配对 **Wilcoxon**」。项目修正 A-6 已钉死置换检验，属既定家规——但 REPORT 该写明，而不是照抄规格的词 |
| NF8 | `run_fafm_probe.py:597-600` | `sem_a` 只按 `family == "semantic"` 过滤，`sem_b` 另加 `sample_id in semb`；`rows_b` 覆盖全集时无害，但两个均值可能算在不同集合上，无守卫无计数 |
| NF9 | 多处 | 未计数的 skip（只 print，不进 `metrics.json`）：`run_uni_gate0_a.py:292-293`、`run_uni_gate0_c.py:470-471`、`:554-556`（静默丢掉 GEOMETRIC∪CONTOUR 之外的 family，如 `unknown`）、`run_uni_gate0_b.py:114-116/125-126`。`dump_fafm_cache.py:241-244` 是好范式（`n_fail` 计数并写进 manifest） |
| NF10 | `fafm_probe_20260811/NOTES.md §4 F3` | 半分辨率解的等价性引「差均值 8.6e-5、最大 2.4e-4」；交付的 eval-cache manifest 是 mean **1.21e-4**、max **1.67e-3**(n=16)，`dump_fafm_cache.py:23` 又是第三对数（5e-5 / 2.8e-4）。`8.649e-5` 实为 `caseC_fpd/metrics.json` 的 `E0.quantisation_cost.mean`——**取错了数** |
| NF11 | `run_fafm_probe.py:558-560` | 臂 F 是朴素 `k=1` 抽样；§3.6 叫它「K=1 **低温**」。`sample_fafm` 无温度旋钮，故该臂测的是「抽一个」而非「低温抽一个」 |
| NF12 | `q3vl/whereb/data.py:141` | `exclude_low` 默认 `False` 且无卡传它，故 `winner_confidence == "low"` 同时进 FAFM 训练缓存与 V_where 评测 GT，违反数据纪律「low 不进 SFT 主训与评测 GT」。**全项目性问题，非本轮引入**，但它在本探针的判据路径上 |
| NF13 | `fafm.py:239-263` | `select_mode` 是最大邻域启发式（取最大的 `sim[i] >= tau` 集合再取 medoid），不是 §3.3 所称的凝聚聚类。K=16 下大概率够用，REPORT 该写明 |
| NF14 | `run_fafm_probe.py:226-233`（经 `fafm.py:219-233`） | CFG 无条件分支只丢 `E_T` 保留 `S`，而训练期 `E_T` 与 `S` 各自独立以 p=0.1 丢弃。可论证（CFG 是对指令的），但未登记 |
| NF15 | `run_fafm_probe.py:375-384` / `:683-707` | `_Sub` 是死代码，其 docstring 描述的是按形状分组修复已经消除的行为，会误导下一位读者；另：无顶层 `verdict` 字段（三张 Gate-0 卡都有），读者得自己从 `n_criteria_passed` 推 |

### FAFM / Gate 0 已核实通过（pass）

- **禁 AUC**：七个文件 `rg -i 'auc|roc'` 只命中「AUC 被禁」的说明文字。判据行齐备：
  面积匹配 top-k 的 soft/hard IoU、**grid 级**边界 F1（`metrics.py:198`，像素级 3px 明确拒绝）、
  同支撑同 top-k 的中心先验列、`a/(2−a)` 随机地板、area 分层。
- **IoU 非优化目标**：FAFM 训练只有 `lambda_metric_loss`；case A 是 L2 + proximal；
  case C 是 L2 拟合、L2 选段、L2 选羽化槽。（唯一残留在 case C 的**报告**路径，见 NF2。）
- **场/s 无逐图归一化**：`c*` 全程不 clamp 不 min-max；`SIGMA_MAX_SQ_ARM` 形式上是整臂常量
  （**值错，见 U5，但形式对**）；case A 的 `theta0/w0/rho` 取整臂 medoid/median。
- **checkpoint 选择不用 val loss**：`train_arm` 跑固定 `--steps` 的 OneCycle 后返回末态，
  无 eval-loss 分支、无 `best_*` 跟踪。
- **先验场不作优化/择优目标**：中心先验在 grid 上按同一 top-k 规则直接打分，
  且刻意**不**经 `A_I` 往返（`:294-321`，理由在 `:295-299`）。
- **可视化纪律**：三张 Gate-0 卡掩膜固定 `vmin=0, vmax=1`；`c*` 用**整臂**对称上下限
  （池化 99.5 分位，`run_uni_gate0_b.py:324-325`）；无逐图 min-max；粗场单独成板
  用 `interpolation="nearest"`，无 resize 叠图。此处无 pad 格（真实长宽比 `F_pre` 网格，非 `expand2square`）。
- **切分**：全部经 `open_dataset(split)` 读已发布的 `SPLIT_DIR/<split>.index.jsonl`，无 ad-hoc 切分；
  split 内的抽样是带 seed 的 `rng.choice`（为探针成本），不是切分。
- **V_where 不进训练**：磁盘核实——`fafm_cache_vwhere_20260811/manifest.json` 为 `"split": "V_where"`, n=400，
  是 `--eval-cache`；`--train-cache` 由 `--split train` 导出。
- **长任务纪律**：七个文件无 `nohup`/`setsid`/`pgrep`/后台 `&`；
  `waves/fafm_probe.sh` 直接 `exec` python，把日志截断 / `ps -p` 判活 / `job.marker` 明确交给 `qjob.sh`。
  两份 amort `job.marker` 亦逐条对齐 D-20 四步并显式声明不用 `pgrep`——**做得好**。
- **`GuidedOp` 值得保留的做法**：算子内强制 `clamp_domain=False` 并把 clip 单独暴露
  （`unifield.py:132-135, 171-173`），这正是让 `A_I` 成为诚实线性映射的关键；
  伴随算子是精确 VJP，且带 `<Ac,r> == <c,Aᵀr>` 自检断言（`:154-183`）。
- **Gate-0 case B 阈值**与 §3.6 逐字相符（`run_uni_gate0_b.py:48-50`）。

---

## 5. E1 attention 探针（方案 B）· 回溯审阅

> **处置说明**：DELTA §七已于 2026-08-10 终裁**方案 B 作废、不许复活**，本节 blocker
> 因此不阻塞任何在跑/待跑作业。它们仍然重要，因为 §七随终裁把**四项资产移交 C 案**，
> 其中资产 1（P4）恰好落在 blocker E1-a 的射程内。

### E1-a 【BLOCKER】固定短语 Δ_const 列不在主判据表内

`run_attn_probe.py:57` `--arms` 默认 `"gt,shuffled"`，发布的导出**从未跑 `fixed_phrase`**；
`analyze_attn_probe.py`（`:29, 216-231`）没有任何 fixed-phrase 代码路径。
PROPOSAL §4 P-W3 明写「三件套负控制：shuffle 指令场(P3)、固定短语场(P4)、中心先验列(P2)
——三列**全部内建为判据**，不是附录」；红线亦要求每行必带 Δ_const。
实测核对：`metrics.json` 任何 pool 下都无 fixed/const 键；交付的 `REPORT.md:329` 自己写的是
「三件套负控制(shuffled / 中心先验)」——**只列了三分之二**。

**为什么它现在仍要紧**：该格后来由 `analyze_attn_diff.py` 补测，而该脚本
**自陈 `FAMILY_STATUS: OUT OF FAMILY`**（`:6-9`，已逐字核实），p 值未校正，
且明确声明「Nothing here can promote the raw arm」。
DELTA §七资产 1 —— **本战役第一条通过的登记判据**（`where_content` 小目标子集 vs 固定短语
Δ=+0.131, p=7.7e-5）——正是这个家族外脚本的产物。
**建议**：C 案立项文档引用该资产时必须标注「家族外补测、p 未校正」，
或在 C 案里把 P4 放进一个预注册家族重跑。这是本节唯一有前向后果的一条。

### E1-b 【BLOCKER】P5 边界 F1 行缺预注册的随机 top-k 守卫列

主 pool 记录只发 `grid_boundary_f1` / `..._center_prior` / `boundary_f1_vs_center_prior`。
PROPOSAL §4 P-W3 P5：「**附随机 top-k 守卫列**（同支撑同 k）：守卫列若不显著低于被测场，
边界 F1 结论作废重析」。守卫只存在于家族外补丁（`analyze_attn_diff.py:193`）。
补测值守卫 0.661 vs 场 0.787，结论**碰巧**成立；但发布当时该行按卡自己的规则是不可解读的，
而 `where_content` 的 `boundary_f1_vs_center_prior` 只有 Δ=+0.0120 (p=0.0201)——
守卫是承重的，不是装饰。

### E1-c 【BLOCKER】`oracle_ceiling` 与它并排打印的场分数支撑不同，且键名指向错误的那个

`analyze_attn_probe.py:132-138`（逐行核实）：
```python
o = np.zeros(c["n_img"]); cand = np.flatnonzero(v)
o[cand[np.argsort(-gtv[v])[:kk]]] = 1.0
ceiling[sid] = float(np.minimum(o, gtv).sum() / (np.maximum(o, gtv).sum() + 1e-8))
results["oracle_ceiling_under_valid_mask"] = _agg(list(ceiling.values()))
```
min/max 跑遍**全部** `n_img` 格，故落在 sink 上的 GT 质量进了分母不进分子——这是
**不排除（mask-cost）口径**。而 `field_scores`（`attnprobe.py:158-172`，已核）把 `pred` 与
`gt_b` 都限制到 `[:, v]`，即每个判据数字都是**仅有效格**口径。两个分母并排发布，
键名却写的是错的那个。后果实测：**每个 pool 都有 189 个 OOF 样本中的 3 个满足
`grid_soft_iou_center_prior > oracle_ceiling`**——零参数先验打赢了自己的「天花板」。
`viz_attn_probe.py:162` 把这个不可比的数与 soft-IoU 三元组印在同一张图题里。
同类误标见 `analyze_attn_sink.py:118,122`、`analyze_sink_b3.py:81-87`。

### E1 nit（11 条）

| # | 位置 | 内容 |
|---|---|---|
| NE1 | `analyze_attn_probe.py:237-238` | 裁定 4 的家族形状实为 **3 池 × 仅 raw 臂**两个 Westfall-Young 家族，非「4 组合(2 臂 × 2 池)同一 max-stat 零分布」。DELTA §六「只导 raw 臂单份场」授权了这一取舍，且差分臂的家族外身份是**保守**处置（永不能晋级）；登记备查：**当前代码下任何含差分臂的组合都不可晋级**，战役若重启必须在预注册家族内重跑，不许打补丁 |
| NE2 | `analyze_attn_probe.py:226,237` | 双侧检验配单侧门。`paired_wilcoxon` 用 `alternative="two-sided"`，`max_stat_fwer` 取 `abs(signed_rank_z)`。统计上保守（`:246` 的 `delta_median >= P2_GATE` 挡住误晋级），但**显著为负**会读成「显著」：`where_special` 发的是 `P2_delta_median=-0.0398, P2_p_fwer=5.0e-5, P2_pass=false` |
| NE3 | `run_attn_probe.py:230-231` | fp16 存储未进 `meta.norm`。实测导出中 0.3–0.6% 的场格被冲成恰好 0，最小非零 5.96e-8（fp16 次正规下限）；而 `write_attn_norm_meta.py:54-79` 写的是 `normalisation_applied: "none"`，消费者无从得知缓存的有效分辨率 |
| NE4 | `write_attn_norm_meta.py:41-52` vs `:83-90` | 消费侧断言是**循环的**：域按被断言的那批文件的 min/max 来定，`assert_domain` 按构造不可能失败，`all_passed: true` 不构成证据。另 `max_frac_saturated: 0.4516` 是假警报——下界声明为 `0.0` 而原始注意力非负，`np.isclose(f, 0.0, atol=1e-6)` 把 45% 的合法小值记成「饱和」 |
| NE5 | 全局 | 判据路径上无 `assert_domain`。`rg 'assert_domain' q3vl/whereb/` 只命中定义（`attnread.py:412`）、那份循环记录、和测试；**产出全部已发布数字的 `analyze_attn_probe.py` 从不断言输入的域** |
| NE6 | `viz_attn_probe.py:145` | 六张场板里有四张（GT、pred top-k vs GT、中心先验 top-k、sink mask）走 `render_field(..., allow_all_valid=True)`，即色标取遍**全部**格、sink 格未画白。`viz.py` 自己的守卫写明该 flag「only if this field genuinely has no pad cells」。最要紧的两张（注意力场与其 shuffled 孪生，`:148-150`）做对了（`valid=V` + 共享 `fixed` 色标），故属辅板纪律松动 |
| NE7 | `attnread.py:263` | sink profile 非查询无关：`profile_rows = np.arange(img_end + 1, len(ids))` 含 `<where>` 段行，即被探的查询池自己参与定义了哪些格被排除。合取规则（`:452-471`）有缓解，docstring 亦有论证，但缓解不等于 profile 独立 |
| NE8 | `analyze_sink_b3.py:125-126` | 注册门下的未注册旋钮：`thr = np.median(rate[rate > 0]); maps[s] = rate > thr` 是逐形状自适应阈值，直接决定 `:157-158` 的 `REGISTERED_GATE: 0.8` 跨分辨率 Jaccard 判定 |
| NE9 | `analyze_attn_diff.py:216-224` | `"frac_cells_negative": None`。`domain_note` 断言差分场「CROSSES ZERO by construction」，而唯一能逐格证明它的字段留空（`domain_raw_diff_field` 倒是填了 `[-3.036, 8.984]`） |
| NE10 | `attnprobe.py:196-212` | `head_norm_constants` 不按 `n_cells` 加权：`s = a.mean(axis=-1)` 后对样本取平均，16×16 与 16×24 两种形状对一个逐格统计量等权贡献。对单头数字无影响（top-k 单调不变），但会移动 8 头凸组合场 |
| NE11 | 多处 | 死代码：`analyze_attn_diff.py:199` `keep` 未用、`:148` `.transpose(0,1,2)` 是空操作、`analyze_sink_b3.py:102` `k0 = max(1, 0)` 未用、`analyze_attn_probe.py:63` 导入 `topk_mask` 从不调用 |

### E1 已核实通过（pass，逐条）

- **禁 AUC**：12 个文件无一处计算 AUC；`tests/test_a5_criteria.py:39-44` 甚至断言产出者**已被删除**。
- **eager 导出无静默回退**：`run_attn_probe.py:95` 显式传 `attn_implementation="eager"`
  （注意 `q3vl/train/modeling.py:153` 的默认是 `flash_attention_2`，故这是**真覆盖**）；
  `attnread.py:100-105` 载入期拒绝其它 impl；`:332-337` 逐层在 `attn_weights is None` 时抛错；
  该路径无任何 `except`/回退分支；`tests/test_attnread.py:151-158` 覆盖。运行快照确认 `"eager"`。
- **叠图无 resize**：`viz.py:200-206` 走 `grid_to_img` 整数边界 + 整数倍 `np.repeat`。
  12 个文件里 `resize|interpolate` 唯一命中是 `attnprobe.py:121-128` 的 `area_resize`
  （GT 从 out/16 到 out/32 的精确 2×2 面积均值，形状非整 2 倍即抛错）。
- **禁逐图 min-max**：`viz.py:60-104` 的 `mode="per_image_minmax"` 直接抛
  `PerImageMinMaxError`，`valid=None` 亦抛（除非显式 `allow_all_valid=True`）。
  对比板用**两场并集的有效格**算共享 `fixed` 色标（`viz_attn_probe.py:127-128`）——
  这正是「两板两色标」陷阱的正确解法。
- **判据用未归一化原始场，着色与算数分开**：`FieldRender.raw_stats` 取自未归一化场（`viz.py:170-176`）。
- **pad/sink 显式排除并单列**：`field_scores` 把预测与 GT 同时限制到 valid（sink 格既不能被选中、
  也不能充当覆盖，已数值验证）；`sink_frac` 逐样本发布并汇总；另有专门的 P-W1 普查
  （质量占比 / argmax 落 sink 率 / GT 质量损失 / k 扫描）。
- **整臂常量**：`head_norm_constants` 在整个 fit 折上池化；唯一的逐图因子 `n_cells` 是确定性几何常量
  （`tests/test_attnread.py:230-241` 断言）。sink **阈值**虽逐图，但它只从支撑里剔除格、不改任何值。
- **裁定 2（双池、零额外前向）**：每个 (sample, arm) 只有一次 `model(...)`（`:213-219`），
  所有池都是那一个 `stacked` 张量的切片（`:222-230`）。实现为**三**池
  （`where_special`/`where_content`/`instr_text`，裁定 2 的超集），结果前预注册；
  `where_close` 被刻意排除在 FWER 家族外以堵事后复活。
- **裁定 5（Δ≥+0.10 且 p<0.01，灰区删除）**：`:30` `P2_GATE, P3_GATE, P_ALPHA, P1_REF = 0.10, 0.08, 0.01, 0.45`；
  `:246-247` 对 **FWER 校正后**的 p 取合取；`:254` 把 P1 降级为 `P1_above_reference` 报告项。
  `rg -i 'gray|grey|灰区|amber'` 全无命中——无灰区残留分支。
- **裁定 4（真 max-stat 联合零分布）**：`max_stat_fwer`（`attnprobe.py:329-337`）每次置换只抽**一个**
  符号向量并施加于全部 key，保持跨池相关性。数值验证：三个相同 key → `p_fwer` 相同（零校正代价）；
  一信号两噪声 → 5.0e-4 / 0.9995 / 0.897。发布的 `p_fwer` 下限为 1/(20000+1)=4.99975e-5，符合预期。
- **面积匹配 top-k**：`attnprobe.py:158-159` 的 k 取自 valid 内的 GT 面积，对每个场（含先验）同一规则，
  无逐场阈值。中心先验经**同一个** `field_scores` 与**同一个** `mask2d` 闭包打分。
- **IoU 非优化目标 / 选择不用 val loss**：`train_learnable_head` 用有效格上的软目标
  `binary_cross_entropy_with_logits`，checkpoint 按 **fit 折** BCE 选。
- **<50k 参数融合头**：1,154（GatedLinearHead）与 26,753（HeadStackCNN），测试断言。
- **切分用 sha1 规则族**：`fit_oof_split` 用 `sha1(f"verasplit-v1:{source_id}")`，
  与旁表自陈的 `meta.split_seed='verasplit-v1'` 一致；按 `source_image_id` 分组，绝不拆开同一 source。
  旁表确实缺 448/448 分区（落后 3 个 build），该偏差在 `NOTES.md §四A` 作为**待决项上报**，未静默取用。

---

## 6. 红线核对总表

| 红线 | P1/P3' | FAFM/Gate0 | E1 探针 | amort 诊断 |
|---|---|---|---|---|
| 禁 AUC（任何空间场判据） | pass | pass | pass | pass |
| 场/s 禁逐图 min-max / softmax | pass | pass（形式；U5 值错） | pass | pass（除 NA5 一行） |
| IoU 禁当优化目标 | pass | pass（NF2 报告路径边缘） | pass | pass（Where-A oracle 书面豁免） |
| checkpoint 选择禁 val loss | pass（但选择未接线 → U3） | pass | pass | pass（无选择步） |
| attention 导出必须 eager | n/a | n/a | **pass（强）** | n/a |
| 色标只取 valid 格、叠图禁 resize | pass | pass（探针零 viz → B8） | pass（辅板 NE6） | pass（NA16 中心先验板自缩放） |
| 缓存 meta.norm 域声明 + 消费端断言 | 声明 pass / **断言 B2** | 声明 pass / **断言 B4** | 声明循环 NE4 / 判据路径无断言 NE5 | **写入侧未实现 NA12** |
| 每消融行必带 Δ_const/Δ_shuffle | Δ_shuffle pass；Δ_const 由 fixed_phrase 承担 | **B6** | **E1-a** | E3 不自足 NA17 |
| antonym 只报不训 | 无损失项 pass；**但 U1 从 ShuffleIndex 绕回** | n/a | n/a | pass（损失路径不可达） |
| 先验场禁当优化/择优目标 | pass | pass | pass | pass |
| V_where 禁进训练 | pass | pass（磁盘核实） | n/a | pass（无可学头） |
| low 置信度禁进主训与评测 GT | **U7** | NF12（同因） | n/a | **U7（同一条，44%）** |
| 切分用 sha1 规则族无 ad-hoc | pass（N4 为子集非切分） | pass | pass | pass（零命中） |
| 长任务 D-20 纪律 | pass（job.marker 逐条对齐，显式弃用 pgrep） | pass | pass | n/a |

---

## 7. 审阅范围

```
新增：q3vl/whereb/{attnread,attnprobe,unifield,fafm}.py
      q3vl/whereb/amort/{data,evaluate,heads,losses,model,simfield,trainer,viz}.py
      q3vl/whereb/scripts/ 22 个新脚本
      q3vl/whereb/tests/test_attnread.py
修改：q3vl/where/{config,fpre,oracle}.py、q3vl/whereb/{hiddens,viz}.py
```
测试：`pytest q3vl/whereb/tests/ -q` → 508 passed（96.9s）。
`pytest q3vl/whereb/tests/test_attnread.py -q` → 30 passed。

## 8. Blocker 复现件

U1 的复现脚本（只读，不改仓库）：
```
/tmp/claude-1001/-home-bc-VeraRetouch/f25f01e7-86bf-444e-91bf-bd8160450b32/scratchpad/check_partner.py
```
运行：`LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib:$LD_LIBRARY_PATH \
/home/bc/envs/q3vl_sft/bin/python <上述路径>`
它打开 V_where、按 `ShuffleIndex(seed=0)` 取 partner、在 16×16 上报
`d(gt_own, gt_partner)` 的分位与 `frac(d < margin)`，并打印最退化的若干对指令原文。
建议把它的核心断言收进 `q3vl/whereb/tests/` 作为回归测试。
