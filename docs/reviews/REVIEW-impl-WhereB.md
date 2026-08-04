# REVIEW-impl-WhereB · Stage-Where-B MetaCanvas 独立实现审阅

**判决：BLOCKER 数量 5；不准许进入 GPU preflight（S3）与正式训练（S5）。**

| 字段 | 值 |
|---|---|
| 日期 | 2026-08-05 |
| 审阅类型 | 独立实现审阅（只读）。以协议为唯一规格来源，不信任 NOTES 的自述结论，逐条复现。 |
| 规格权威 | `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` §1.1/§2.3/§3/§4.2/§5/§10.3/§11/§14 |
| 审阅对象 | `q3vl/whereb/`（16 模块 + 3 脚本 + 1 shell + 17 测试文件）、`experiments/Q3VL_metacanvas_where_what_20260804/where_b/`（NOTES.md、PREFLIGHT_WHERE_B_PENDING.md、mock_closed_loop.json、preflight_where_b_cpu.json） |
| 本次审阅修改的文件 | 仅本文件。未改动任何代码/数据/配置，未占用 GPU。 |
| 结论 | **5 BLOCKER / 13 NIT**。BLOCKER 全部可复现，均给出精确 file:line 与规格条文。 |

## 复现环境与既有断言复核

| 项 | 命令 | 结果 |
|---|---|---|
| Where-B 单测 | `pytest q3vl/whereb/tests -q` | **170 passed / 19.45 s** ✅ 与 NOTES 一致 |
| Where-A + 共享套件回归 | `pytest q3vl/where/tests q3vl/tests -q` | **120 passed / 36.09 s** ✅ 与 NOTES 一致 |
| Where-A 未被篡改 | `q3vl/` 整目录未入 git，无 diff 可比；改以 mtime 排序 | `q3vl/where/*.py` **全部早于** `q3vl/whereb/*.py`，且 Where-A 120 个测试全绿 → 支持"无篡改"，但**这不是强证据**（见 N13） |
| 参数量表 | 手工重算 stream / bank / heads 逐项 | W01 34,838,745；W07 69,835,365；W08 69,851,781 —— **与 NOTES §3.3 逐位一致** ✅ |

参数量手算复核（确认 §5.2 的三档差异归因是真的）：
`ConnectorStream` = text_proj 1,311,232 + vision_proj 524,800 + pos 66,048 + 6×5,257,730 + norm_out 1,024 = **33,449,484**；
`bank(8×8)` = 32,896、`bank(16×16)` = 131,584；`heads(joint,band)` = 1,356,365、`heads(split,band)` = 2,673,229；
readout 差 = 512×(36−4)+(36−4) = **16,416**。W07 = 2×131,584 + 2×33,449,484 + 2,673,229 = 69,835,365 ✅。

---

## 一、BLOCKER

### B1 · `preflight --with-model` 从不执行 §14 项 7b/8b，却报 PASS（静默空转）

**规格**：§14 项 7「GT/generated/null/shuffled context 数据流无串线」、项 8「证明 `Q_where` 不读 `H_color`」；
`PREFLIGHT_WHERE_B_PENDING.md:98` 明写 `WB-P7b-hidden-contract` **「FAIL 则不许继续」**。

**实现**：`q3vl/whereb/preflight.py:415-444`

```python
    rep.add(check_context_flows(processor.tokenizer))
    rep.add(check_no_h_color(WhereBModel(arm_config("W01")).eval()))
    rep.add(check_no_target_leak())
    rep.add(check_zero_init_gates())
    rep.add(check_parameter_table(arms))
    if skip_model:
        for cid in ("WB-P7b-hidden-contract", "WB-P8b-h-where-causal-independence"):
            rep.add(Check(cid, "skip", {}, "--skip-model (needs a real forward)"))
```

`--with-model`（即 `run_where_b.sh preflight`）只做一件事：**把两个 skip 标记去掉**。driver 里没有任何路径去构造 VLM 并调用
`check_hidden_contract`（`preflight.py:287`）与 `check_h_where_causal_independence`（`preflight.py:251`）。
`rg` 全仓确认这两个函数**零调用者**（测试里也没有）。

复现（CPU，`skip_model=False`）：

```
ok: True  n_pass: 5  n_fail: 0  n_skip: 0
check ids: ['WB-P7-context-flows', 'WB-P8-no-h-color', 'WB-P9-no-target-leak',
            'WB-P-zero-init-gates', 'WB-P-param-table']
```

**危害**：操作员按 PENDING S3 跑 `run_where_b.sh preflight`，屏幕打印 `preflight PASS`、JSON 里 `n_skip=0`，
会得出"§14 项 7b/8b 已通过"的结论，而这两项**根本没跑**。这正是 CLAUDE.md 反复警告的静默失败类型
（"见到 PASS 不等于跑过"）。`PreflightReport.ok`（`preflight.py:86-88`）只检查"没有 fail"，缺失的检查不计入。

**修复**：`run_where_b_preflight` 增加 `else:` 分支，加载真实 checkpoint + processor + 一条真实样本，
调用两个函数并 `rep.add(...)`；同时把 `ok` 改成"必需检查 id 集合齐全 **且** 无 fail"。

---

### B2 · 两个硬前置作业（S1 oracle latent、S2 generated context）**当场崩溃，无法运行**

**规格**：§5.4「训练 batch 固定 50% teacher、50% generated」、§5.5 三个 oracle 辅助 loss；
PENDING 把它们列为 S1/S2，且 S5 不可跳过。

**实现**：

- `q3vl/whereb/scripts/make_generated_context.py:84` — `dataset = WhereBDataset(args.split, limit=args.limit)`
- `q3vl/whereb/scripts/make_oracle_latents.py:76` — `dataset = WhereBDataset(args.split, include_global=False, limit=args.limit)`

两处都**没有传 `maskviews=` 也没有传 `mask_resolver=`**。`WhereBDataset.__getitem__` 在 `data.py:173` 无条件调用
`self._mask(...)`，而 `data.py:198-201`：

```python
        raise RuntimeError(
            f"{sample_id} is a local sample but neither a published maskview store "
            "nor a live MaskResolver was provided"
        )
```

复现（真实 `V_where.index.jsonl` + 真实 shard，CPU，只读）：

```
len(ds) = 40
[0] RuntimeError: sft_00680d96738cd6077be81308b15e19b3 is a local sample but neither a
    published maskview store nor a live MaskResolver was provided
succeeded on 0 samples before the first failure
--- include_global=False (make_oracle_latents.py) ---
RuntimeError: sft_00680d96738cd6077be81308b15e19b3 is a local sample but ...
```

即：`make_oracle_latents.py` **第一个样本就死**；`make_generated_context.py` 在 train 段撞上第一个 local 样本时死
（train 段 47.45% 是 local，实测见下表）。这不是"权重没就绪"的伪失败——两个脚本的**数据构造本身**是错的。

顺带：`make_generated_context.py` 根本不需要 mask，`__getitem__` 却强制加载它；应当给 `WhereBDataset` 一个
`need_mask=False` 开关，或在 `generate_records` 里走一条不取 mask 的轻路径（169,215 个样本每个多解一次 `.cgt.png`
是纯浪费）。

**修复**：两处都传 `maskviews=MaskViewStore(WHERE_A_MASKVIEW_DIR / split)`（`run_where_b.py:102` 已经这么做了），
`make_generated_context.py` 另加 `need_mask=False`。**修完必须用 `--limit 8` 真跑一次**再提交全量。

---

### B3 · `shuffled` 只换 `<where>` 正文，**指令没有被交换**——与 §5.4/§5.6 的 gate 名字不符，且属静默拍板

**规格**：
§5.4「`shuffled` 在同一图像、同一局部层级内**交换 instruction/where context**，防止用图像主体显著性冒充指令理解」；
§5.6 gate 第 6 行「**instruction shuffle** 后 IoU 降幅 `>= 0.20`」。

**实现**：`q3vl/whereb/data.py:278-289`

```python
        elif mode == SHUFFLED:
            ...
            text = self.shuffle_index.by_id[partner]["where"]
            ctx = shuffled_context(self.tokenizer, partner, text)
```

只替换了 `<where>` 段的 token ids。而同一个 `build()` 里，prompt 仍来自样本自己：`data.py:302`

```python
            enc = self.collator.encode_one(_PromptShim(s))
```

`_PromptShim.__init__`（`data.py:390-396`）把 `self.instruction = s.instruction` —— **样本本人的指令**。
`Sft2SegCollator.encode_one`（`collator.py:138`）用 `build_prompt_text(sample.instruction)` 造 prompt。

**为什么这不是等价写法**：`Q_where` 只读 `H_where` + `F_pre`，指令**唯一**的入口就是 `H_where`
（因果注意力下 `<where>` 位置的 hidden 会 attend 到 prompt 里的指令）。所以留着真实指令 = 让正确答案仍然可达。
一个"完全靠正确指令、根本不看 `<where>` 推理"的模型，在这个 shuffled 下降幅接近 0，会**被这条 gate 误杀**；
反过来，一个靠图像显著性作弊的模型也降幅接近 0 ——**这条控制既抓不到作弊，也误伤好模型**，
和 §5.4 写的目的（"防止用图像主体显著性冒充指令理解"）正好相反。

**更严重的是这是一次静默拍板**。NOTES §四 D-B6 只讨论了分组键，D-B7 只讨论了"用 partner 的 GT 还是 generated 文本"，
**没有任何一条提到"prompt 里的指令换不换"**；而 D-B7 的行文写着「只换指令内容」（`NOTES.md:184`），
与代码**恰好相反**。CLAUDE.md 派工协议第 3 条：属于决策的必须写进 NOTES 待决策节，"不许静默拍板"。

**修复**：`_PromptShim` 增加一个 `instruction` 覆盖参数，`shuffled` 模式下同时替换 instruction 与 where 正文；
或由主 agent 明确裁定"只换 where 正文"并**把 §5.6 那一行 gate 改名/改判据**（改判据必须在 S5 之前，
看到结果后再改 = §10.3 末段禁止的行为）。

---

### B4 · bf16 autocast 漏进 `s_low`：训练用 bf16、评测用 fp32，且**对 CBand12 臂的伤害远大于 Band 臂**

**规格**：§5.3 的 8 臂是 4 结构 × 2 readout 的**受控对比**；§10.3 `precision: bf16`。
实现自己的规格声明（`trainer.py:26-30`）：

> "the connector runs under bf16 autocast, but **everything from `phi_dir` onwards is float32**.
> The guided filter divides by `var + 1e-3` and the readouts exponentiate; both lose too much in bf16."

`fields.py:168-171` 同样声明「Everything is computed in the dtype of `phi_dir`（float32 in training）」。

**实现与声明不符**：`trainer.py:187-190` 把整个 `compute_batch` 包进 autocast：

```python
                with self.autocast:
                    total, stats, _ = compute_batch(
                        self.model, batch, self.arm_cfg, weights
                    )
```

`compute_batch` 内部调用 `predict_fields` → `s_from_params`（`fields.py:152-153`）：

```python
    q = w0 + alpha * (phi_dir @ w_dir)
```

`@` 是 matmul，**autocast 的 bf16 白名单第一条**。实测（CPU autocast，与 CUDA 语义一致）：

```
avg_pool2d: float32   interpolate bilinear: float32   exp/sigmoid/tanh: float32
matmul: bfloat16      linear: bfloat16
```

即 guided filter 与 readout 确实是 fp32（作者的分析对），**但 `s_low` 本身是 bf16**——正好是他要保护的那个量。
在真实量级（`phi_dir ~ N(0,1)` 71 维、`‖w_dir‖=1`、`alpha=2`）上实测：

```
max|Δs| = 2.14e-2    mean|Δs| = 4.72e-3    std(s) = 1.51
cband12 (σ=0.03):  max|Δm| = 4.21e-1   mean|Δm| = 8.63e-3
band  (k≈8,h≈1):   max|Δm| = 8.91e-2
```

**三重危害**：

1. `evaluate_context`（`evaluate.py:40`）只有 `@torch.no_grad()`、**没有 autocast** → 评测的 `s` 是 fp32。
   于是「训练优化的 mask」与「gate 度量的 mask」不是同一个函数。
2. CBand12 的 `σ ∈ [0.025, 0.30]`（`where/config.py:66`），在 σ 下界附近 `Δs = 2e-2` 会把
   `exp(−0.5((z−μ)/σ)²)` 改变数倍 → **W02/W04/W06/W08 系统性受损，W01/W03/W05/W07 基本无感**。
   §5.3 要对比的正是这两组，这是一个直接污染主结论的偏置。
3. `L_s = Huber(s_pred/3, s*/3)`（`losses.py:160`）的监督噪声底也被抬到 ~5e-3。

**修复（一行）**：在 `compute_batch` 里、`model(**batch.inputs)` **之后**、逐样本循环**之前**，
包一层 `with torch.autocast(device_type=..., enabled=False):`；或在 `predict_fields` 入口处禁用 autocast。
并把 `evaluate_context` 与训练的精度口径显式对齐（两边都 fp32）。

---

### B5 · oracle 辅助 loss 被 global 样本稀释到名义权重的 ~0.47 倍——未在 NOTES 申报

**规格**：§5.5

```text
前 30% optimizer steps:  L_where = L_mask + 1.00 L_s + 1.00 L_curve + 0.10 L_dir
```

并解释「前段用 oracle latent 解决 `s/readout` 联合优化的**非辨识和早期坍缩**」。

**实现**：主 agent 已裁定 D-B5「global g1-g4 进训练、oracle 辅助关闭」，`data.py:362` 对 global 样本跳过 oracle：

```python
        if self.oracle is not None and not sample.is_global:
```

`sample_loss`（`losses.py:242-247`）在无 oracle 时**只**累加 `L_mask`；随后 `aggregate`（`losses.py:271`）：

```python
    total = torch.stack([l.total for l in losses]).mean()
```

对**全部**样本取平均。于是 `L_s/L_curve/L_dir` 的**实效权重 = 名义权重 × (带 oracle 的样本占比)**。

实测 split 组成（读 `train.index.jsonl` / `V_where.index.jsonl`）：

| split | n | global | local | **local 占比** |
|---|---:|---:|---:|---:|
| `train` | 159,215 | 83,671 | 75,544 | **47.45%** |
| `V_where` | 896 | 496 | 400 | 44.64% |

即 stage-1 的 `1.00 L_s + 1.00 L_curve` 实际是 `≈0.47 L_s + 0.47 L_curve`（再乘以 Where-A 的 fit 成功率），
**比 §5.5 给 stage-2 定的 0.25 更接近 stage-2 而不是 stage-1**。两段 schedule 的对比因此被压扁，
而 stage-1 存在的全部理由就是"用大权重的 oracle 监督压住早期坍缩"。

**这是一次未申报的静默决策**：NOTES D-B5 只写了"对 global 屏蔽三个辅助 loss"，
**没有指出屏蔽 + 全 batch 平均 = 名义权重被 batch 组成打折**。两种读法都成立：
(a) 现状：权重是"每 batch"的；(b) `sum(w·L_aux)/n_with_oracle`：权重是"每有 oracle 的样本"的，保持 §5.5 字面值。
必须由主 agent 裁定，且**必须在 S5 之前**——8 臂跑完再改等于全部重跑。

**最低要求**：无论裁定哪一种，`steps.jsonl` 必须记录每步的 `n_with_oracle / n`
（`aggregate` 已经算了 `stats["n_with_oracle"]`，只是没有换算成实效权重）。

---

## 二、逐项对照表（任务卡八个必审重点）

### 1. Connector 结构（§5.1）— **PASS**

| 条文 | 实现 | 判定 |
|---|---|---|
| 宽 512 | `config.py:44` `CONNECTOR_DIM = 512` | PASS |
| 6 pre-norm blocks | `config.py:45`；`connector.py:154` `ModuleList(... for _ in range(cfg.n_blocks))`；`ConnectorBlock` 全部 `norm_*` 在 attn/ffn 之前 | PASS |
| 8 heads / FFN 2048 | `config.py:46-47`；`connector.py:116-118` | PASS |
| 顺序 self → cross(H_where) → cross(F_pre) → FFN | `connector.py:128-137` 逐行即此顺序 | PASS |
| cross-attn residual gate zero-init，真 zero 且可学 | `connector.py:108,113` `nn.Parameter(torch.zeros(1))`；`preflight.check_zero_init_gates` 实测 W01/W08 初始输出对输入的 max diff = **0.0**；`build_optimizer` 把 gate 归入 no-decay 组但**仍在优化器里**（`trainer.py:93-98`） | PASS |
| H_where / F_pre 独立投影到 512 | `connector.py:151-152` `text_proj` / `vision_proj` 两个独立 `nn.Linear` | PASS |
| 输出头只产 w0/w_dir/alpha/rho，无 dense logits | `heads.py:87-142`：`AxisOutput`/`RhoOutput` 的输入都是 pooled `(B, dim)`，输出 `(B,73)`/`(B,n_rho)`；模块里不存在 canvas-token → 空间图的路径 | PASS |

补充确认（非任务卡点，但值得记）：`MultiheadAttention` 对全掩码行的处理（`connector.py:79-92`）——
把无有效 key 的行改成全有效再把输出置零，等价于"该分支不贡献"，避免 null 上下文 `0/0` → NaN 污染整 batch。
这是正确且非平凡的处理，有专门单测。

### 2. 四结构变体与参数量（§5.2/§5.3）— **PASS**

| 结构 | 规格 | 实现 | 判定 |
|---|---|---|---|
| MC8-Joint / MC16-Joint | 同一 attention pool + joint head | `config.py:63-64` `streams:1, pools:1`；`heads.py:160-164` 单 pool + 共享 trunk + 两个输出投影 | PASS |
| MC16-SplitHead | shared canvas，`w`/`rho` 独立 pool + 独立 head | `config.py:65` `streams:1, pools:2`；`heads.py:166-171` `pool_axis/pool_rho` + `trunk_axis/trunk_rho`；`model.py:118-119` 两条 canvas 指向同一 stream 输出 | PASS |
| MC16-DualCanvas | 两套独立 query bank/connector stream，只共享 frozen VLM | `config.py:66` `streams:2, pools:2`；`model.py:93-97` 两个 bank + 两个 `ConnectorStream`（投影/位置编码/6 block 全在 stream 内）；`w`-路与 `rho`-路无任何共享张量 | PASS |
| 8 臂 = 4×2 笛卡尔积 | §5.3 表 | `config.py:72-81` 逐行一致 | PASS |
| 参数量 W07/W08 ≈ 69.8M | — | 手工重算 = 69,835,365 / 69,851,781，**与 NOTES §3.3 逐位一致**；三档增量（+98,688 / +1,316,864 / +33,581,068）与结构差异吻合 | PASS |

### 3. Context 数据流（§5.4）— **B3 BLOCKER，其余 PASS**

| 条文 | 实现 | 判定 |
|---|---|---|
| 训练 batch 固定 50/50 | `context.py:231-274` `BalancedContextSampler`：micro-batch 内精确一半一半（奇数 micro-batch 直接抛错，`context.py:246-250`），因此对任何 GAS 都成立 | PASS |
| generated 缺闭合标签不回退 GT（**结构性证明**） | 三重：(a) `generated_context`（`context.py:123-161`）签名里**没有任何 GT 文本参数**，无值可回退；(b) `BatchBuilder.context_for`（`data.py:268-272`）在 `genctx is None` 时 `RuntimeError`，注释明写 "never fall back to GT"；(c) `GenContextStore.record` → `PublishedStore.read` 在样本缺失时 `KeyError`（`stores.py:73-74`），**不被任何 except 吞掉**（全包只有 2 处 `except Exception`，都在 `preflight.py` 的 env/键字探测里） | PASS |
| 固定边界截取 + 记录格式失败 | `context.py:139-161`：`</where>` 之前截断→`closed`；越界→`closed_over_boundary`+failure；无闭合→前 96 token+`no_close_tag`+failure；空→`empty`。`WHERE_CONTEXT_MAX_TOKENS=96` 由实测 2711 条 GT（local max 79 + 2 标签 = 81）导出，且 `gt_context` 对超界 GT **直接抛错而非截断**（`context.py:113-118`）——这条很对 | PASS |
| shuffled = 同图像同层级内 derangement | `context.py:178-219`：组内随机置换后循环移位 → 无不动点；单例组不跨图配对，记为 uncovered；`evaluate_context`（`evaluate.py:61-64`）跳过并计入 `n_skipped` | PASS（分组构造正确） |
| shuffled = **交换 instruction/where context** | 只换 where 正文，指令未换 | **B3 BLOCKER** |
| null 构造 | `context.py:164-165` 零 token；`WhereContext.__post_init__` 强制 null 不带 token；下游 `_pad_stack(..., min_len=1)` + 全 False mask → cross-attn 输出精确 0 | PASS |
| 四 context 分开报告，不混成均值 | `evaluate.py:153-163` 逐 context 各跑一遍全 split；`metrics.arm_metrics`（`metrics.py:160-189`）保留 `per_context` 并只从 `generated` 板读 gate 指标 | PASS |
| GT/generated 子批 loss 分别计算 | `losses.aggregate`（`losses.py:280-295`）按 `contexts` 分组给 `by_context` 的 loss 与全部标量；优化的标量仍是全 batch 均值（50/50 时等于两个子批均值的平均） | PASS |

### 4. `H_color` 隔离（§14 项 8）— **证明本身充分，但数值半从未执行（B1）**

| 半 | 实现 | 判定 |
|---|---|---|
| 签名白名单 | `preflight.py:211-224` 扫 `WhereBModel.forward` / `ConnectorStream.forward` / `ConnectorBlock.forward` / `LatentHeads.forward` 的参数名 | PASS |
| 标识符扫描 | `preflight.py:197-204` 用 `tokenize` 只取 `NAME` token（**剔除注释与字符串**，比裸 grep 强），扫 qwhere/connector/heads/model 四个模块 | PASS |
| 关键字被拒 | `preflight.py:239-247` 真调用 `model(..., h_color=...)` 断言 `TypeError` | PASS |
| 数值 bit-identical | `check_h_where_causal_independence`（`preflight.py:251-284`）逻辑正确：同图同指令同 `<where>`、只换 `<color>` 正文 → `H_where` 必须 `torch.equal` | **函数正确但零调用者（B1）** |

**绕过路径搜索结论**：我按四条可能的绕过路径逐一查过，未发现漏洞——
(a) `BatchBuilder` 造 `EncodeItem` 时 `where_ids=ctx.token_ids`，`_PromptShim.color_text="."` 只用于 `n_prompt_tokens`，
且 prompt 在第一个 `<where>` 之前就结束（`collator.py:139,145`），颜色正文进不了 prompt；
(b) `FrozenVLM.encode`（`hiddens.py:218`）切片 `hidden[i, n_p:n_p+n_w]`，右 padding，切片边界正确；
(c) `MODEL_INPUT_KEYS` 是硬编码五元组，`Batch.check_inputs`（`data.py:214-223`）拒绝任何额外键；
(d) `WhereBOutput.canvas_axis/canvas_rho` 虽然被导出（供 §6 的 `z_where`），但只来自 connector，不含颜色。
**唯一未被证明的是数值半——因为它从没跑过。**

### 5. Loss 逐符号（§5.5）— **PASS（数值口径见 B4，权重稀释见 B5）**

| 符号 | 规格 | 实现 | 判定 |
|---|---|---|---|
| `L_mask` 三项权重 | `1 / 0.25 / 0.10` | `config.py:115-117`，`losses.py:149` | PASS |
| `softIoU` | 未指定形式 | min/max（`losses.py:75`），D-B10 已申报，与 Where-A `soft_iou_minmax` 同定义 → loss/gate/oracle 比值三处同一个数 | PASS |
| `balanced_BCE` | 未定义 | 逐图 `w_pos=0.5/mean(t)`、`w_neg=0.5/(1-mean(t))`（`losses.py:91-95`）；t≡1 时负项恒 0，不会被 `w_neg` 放大 | PASS |
| `boundary_F1_loss_3px` | 未定义 | 逐符号照抄 arXiv:1905.07852（`losses.py:112-137`）：`pool(1-y,3)-(1-y)` → tol pool 7 → `1-2PR/(P+R)`。**出处已由实施者打开原文核实（V-B6），我复核了公式形状与论文一致** | PASS |
| — 边界退化 | — | `both_empty` 时返回 0 而非常数 1（`losses.py:136-137`），否则每个 global 样本白扛无梯度罚项 | PASS（正确的处理） |
| `L_s = Huber(s/3, s*/3)` | — | `losses.py:160` `F.huber_loss(s_pred/S_SCALE, s_star/S_SCALE)`，`S_SCALE=3.0` | PASS |
| `L_curve`，z=linspace(-3,3,257) | — | `config.py:129`；`losses.py:155-169`；`r_star` 由 `oracle_fields` 在**同一张 z 网格**上现算（`fields.py:218`） | PASS |
| `L_dir = 1 - cos` | — | `losses.py:172-175` | PASS |
| 两段 schedule 30%/70% | — | `losses.py:180-193`，切换点 `round(0.3·total)`，`step<boundary` 为 stage 1；权重组 `{1.00,1.00,0.10}` / `{0.25,0.25,0.05}`（`config.py:132-135`） | PASS |
| 「任何 loss 都同时在 GT/generated 子批上计算」 | — | `aggregate` 的 `by_context` | PASS |

补充确认：`w_dir` 的非辨识性没有制造问题——`w_dir_of` 不做符号规范化（`where/basis.py:31-32`），
但 `L_s` 直接对 `s*` 监督，符号被钉死；`L_dir` 只是 0.10/0.05 权重的辅助项。
`s_from_params`（`fields.py:152-153`）与 Where-A `basis.s_low`（`basis.py:118-119`）**逐符号一致**
（`q = w0 + alpha*(phi@w_dir)`，注意 `basis.py` 的模块 docstring 写成 `(w0+<phi,w>)*alpha` 是 Where-A 侧的文档笔误，代码是对的）。

### 6. generated-context 产出方案（§2.3 + §5.4 + D-B2）— **PASS（脚本本身见 B2）**

| 项 | 结论 |
|---|---|
| 缓存 token ids 重放同一 encode | **设计成立且是本包最好的一处**。`gencontext.py:1-30` 的论证正确：缓存 hidden 会让 teacher/generated 出自**两次不同的调用**，"同层同位置同归一化"就只能靠人守；缓存 ids 后两者共用 `FrozenVLM.encode`（`hiddens.py:157-223`），差别**只有 token ids**，口径一致变成结构性成立。`check_hidden_contract` 正是为此写的断言（可惜没跑，B1）。 |
| teacher/generated 口径一致 | `context_for` 的两条分支最终都汇入 `build()` 的同一个 `EncodeItem(prompt_ids=..., where_ids=ctx.token_ids)`，没有第二条路径。PASS |
| hidden 口径 = post-norm（D-B2） | `config.py:107,112` `WHERE_HIDDEN_LAYER=-1` + `WHERE_HIDDEN_FINAL_NORM=True`；`hiddens.py:203-204` `hidden = self.lm.norm(hidden)`。**代码与 D-B2 裁定、与 NOTES 文档三者一致**。且 teacher/generated 共用此函数 → 同口径。PASS（一个建议见 N1） |
| shard 契约符合性 | `publish_generated` → `q3vl.data.shardio.build_from_memory`（不压缩 tar、staging + fsync + 原子发布、sqlite catalog + `shard-*.idx.jsonl` 索引、checksum）。`GenContextStore` 读取前强制 `manifest.status == "complete"`（`stores.py:46-50`）→ 拒读非原子发布物。`test_stores.py` 用 Where-A **自己的 packer** 真打一套 shard 再读回，是真 interop 测试。PASS |
| shard 大小 | `GENCTX_SHARD_BYTES = 1 GiB`；genwhere 记录 ~500 B × 159k ≈ 80 MB → 只会有 1 个 shard，达不到 §2.3 的 1–4 GiB 目标。**这是数据量决定的，不是实现问题**，记为 N9 |
| 生成栈选择 | 用训练环境自己的 HF greedy，不用 vLLM。理由（另一套 torch/transformers、ids 可能不一致、给不出 hidden）**正确且保守**，且 V-B7 的可用性核实是真做过的。PASS |

### 7. Where-A 接口消费 — **PASS（两个 nit）**

| 项 | 结论 |
|---|---|
| import 复用无篡改 | `q3vl/where/` 全部 14 个 .py 的 mtime **早于** `q3vl/whereb/` 全部文件；Where-A 120 个测试全绿；`fields.py` / `stores.py` / `losses.py` 只 import 不改写。`make_oracle_latents.py` 明确说明"补 train 段而不编辑 `q3vl/where/`"，做法正确。PASS |
| oracle schema 一致性 | `OracleStore.latent`（`stores.py:102-122`）读 `payload["fits"][readout]["status"]/["latent"]` 并交给 **Where-A 自己的 `Latent.from_dict`** 反序列化。对照 `run_calibration.py:150-152` 的写入（`fit.to_dict()` 去掉 sample_id/meta/phi_diag）与 `FitResult.to_dict`（`oracle.py:220-235`），字段**逐项对得上**。PASS |
| 拒绝的拟合不填零 | `status != "ok"` → 返回 `None` → 该样本三个辅助 loss 被屏蔽而非零填（§10.2「不静默换成零向量」）。PASS |
| `s*` / `r*(z)` 现算不落盘 | `fields.oracle_fields` 用**同一张 `phi_dir`** 重算，杜绝"latent 是在旧 B 下拟的"这类静默错配。这是正确且非平凡的设计选择。PASS |
| `phi_dir_fast` 与 Where-A 逐位一致 | 有 `test_fields.py::test_phi_fast_matches_where_a_bit_for_bit`（3 种网格）钉死。PASS |
| interop mock 单测 schema 与实际产物结构一致 | `test_stores.py` 用 `pack_oracle`/`pack_maskviews`（Where-A 的 packer）真造 shard。**符合任务卡要求**。PASS |
| fit 复现口径 | `make_oracle_latents.py:108` 未传 per-sample `seed_offset`，而 Where-A `calibrate.fit_sample:121` 传 `seed_offset=j` → 见 N3 |
| `mask_low` 来源 | `make_oracle_latents.py:102-104` 从 `mask_hi` 重算而非直接读 Where-A 已发布的 `.masklow.npy` → 见 N4 |

### 8. 优化配置（§10.3）与 gate/选择（§5.6）— **PASS**

| §10.3 条目 | 值 | 实现 | 判定 |
|---|---|---|---|
| optimizer | AdamW | `trainer.py:99` | PASS |
| learning_rate | 2.0e-4 | `config.py:144` | PASS |
| weight_decay | 0.01 | `config.py:145`，`trainer.py:96` | PASS |
| warmup_ratio | 0.03 | `config.py:146`，`calibrate.make_scheduler:68` | PASS |
| scheduler | cosine | `calibrate.make_scheduler:73-76` | PASS |
| max_grad_norm | 1.0 | `trainer.py:197-198` | PASS |
| precision | bf16 | `trainer.py:150-154` | PASS（口径问题见 B4） |
| effective_batch_per_arm | 32 | `TrainConfig.grad_accum`（`config.py:272-278`）在 `32 % micro != 0` 时**抛错**而不是默默取整 | PASS |
| epochs | 1.0 | `BalancedContextSampler` 把数据集切成两个不相交的一半，每样本一个 epoch 只出现一次（D-B13） | PASS |
| eval_steps / save_steps | 500 / 500 | `config.py:151-152`，`trainer.py:220-223` | PASS |
| micro-batch 先探测再定 GAS | — | `probe_micro_batch`（`trainer.py:266-292`）在 {2,4,8} 上试，OOM 即停 | PASS |
| 单卡一臂 | — | `run_where_b.py --device cuda`；`run_where_b.sh` 不锁卡 → N7 |
| **checkpoint 选择禁用 val loss（红线）** | — | `WhereBTrainer.best`（`trainer.py:254-263`）按 `local_soft_iou_median` 选，`eval_loss` 只记录；单测构造了"eval_loss 单调下降但指标峰值在中间"的场景 | PASS |

§5.6 九项 gate：`config.py:161-171` 与协议表**逐行一致**（阈值、方向、指标名全对）。
`SELECTION_ORDER`（`config.py:173-180`）= 「median soft-IoU → boundary F1 → p10 → 参数量/显存/延迟」，与 §5.6 一致。
`lexicographic_best`（`metrics.py:215-245`）逻辑正确：缺失值一律排最后（`larger_is_better` 时取 `-v`，缺失取 `+inf`）；
**不过门不淘汰候选，只给最优者打 `WHERE-GATE-FAILED`**，与 §5.6 末段一致；`evaluate_gates` 对缺失指标判 fail 而非 pass（`metrics.py:198-200`）——方向正确。

---

## 三、NIT（不阻断，但建议在 S5 之前顺手处理）

| # | 位置 | 问题 | 建议 |
|---|---|---|---|
| N1 | `config.py:107-112` | D-B2 裁定的是"**全局统一口径**（含未来 Stage-What 的 `H_color`）"，但 `WHERE_HIDDEN_LAYER/WHERE_HIDDEN_FINAL_NORM` 定义在 `q3vl/whereb/config.py`，Stage-What 只能靠自觉 import 或重新声明 | 提到 `q3vl/train/constants.py` 或新建共享模块，让 Stage-What **无法**静默分叉 |
| N2 | `run_where_b.py:108-119` | 只查了 `oracle_coverage`（前 2000 条），**没查 genctx 覆盖率**。`BalancedContextSampler` 是先按索引分池、后取 genwhere 记录，一条缺失就会在训练几小时后炸 `KeyError` | 在 `setup` 里加 `assert set(generated_pool ids) <= genctx.sample_ids`，缺失即启动时失败 |
| N3 | `make_oracle_latents.py:108` | 未传 per-sample seed（Where-A 传 `seed_offset=j`），全 split 用同一个多起点种子；NOTES 却称"由产生 V_where 那批的同一套代码产生" | 传 `FitConfig(seed=base+i)`，与 `calibrate.fit_sample:120` 对齐 |
| N4 | `make_oracle_latents.py:102-104` | 从 `mask_hi` 重算 `mask_low`；若 `mask_hi` 来自已发布的 `.maskhi.png`（uint8），会先被量化到 1/255 再降采样，与 Where-A 的 float 路径有微差 | 直接读 `MaskViewStore.mask_low`（Where-A 已发布 `.masklow.npy`） |
| N5 | `evaluate.py:85-94` | `met.update({**{k: tgt["meta"].get(k) for k in STRATA_KEYS}})` 会把 `render_mode` 写成 `None`（键已存在），随后的 `met.setdefault("render_mode", ...)` **修不回来**；`summarise` 的 `is_local`/`is_global` 都会判 False → 该行**从 local 与 global 两个聚合里同时消失**。当前所有 record 都带 `render_mode`（实测），所以是潜伏 bug | 改成显式赋值 + `assert met["render_mode"] in ("local","global")` |
| N6 | `metrics.py:181-184` | `instruction_shuffle_iou_drop = generated_median − shuffled_median` 把"generated↔GT"与"本人↔partner"两个轴混在一起 | 改成 `gt_median − shuffled_median`（两边都是 GT 文本，只差 partner），或至少同时报两个版本 |
| N7 | `run_where_b.sh:24-35` | `submit` 不接受 GPU 参数、不设 `CUDA_VISIBLE_DEVICES`；按 usage 连开两臂会都落在 GPU 0。另：`( cd ... && nohup ... & echo $! )` 记录的可能是包装子 shell 的 PID 而非 python 的；step 3 的 `tail` 只**打印**日志、不**校验**是否有实质内容，空日志也会 return 0 | 加 `submit <gpu> <log> <cmd...>`；`ps -p $pid -o pid,cmd` 打印出来人工确认；step 3 改成 `grep -q '"arm"' "$log"` 之类的实质断言（D-20 第 3 步） |
| N8 | `data.py:256, 292` | `BatchBuilder.format_stats` 跨多次 `evaluate_arm` 调用**累加**；每 500 步一次 eval 复用同一个 `eval_builder`，于是 `metrics.json` 里的 `format_stats` 是累计值而非本次 eval 的 | eval 开始时 reset，或改成按 (step, mode) 分桶 |
| N9 | `config.py:197` | `GENCTX_SHARD_BYTES = 1 GiB`，但 genwhere 全量只有 ~80 MB → 单 shard，达不到 §2.3 的 1–4 GiB 目标 | 无需改代码，在报告里说明"数据量决定"即可 |
| N10 | `metrics.py:75-96` | `AUC_target` 定义（预测 mask 作为 score、GT 在 0.5 二值化的 ROC AUC）是**协议未定义**的量，NOTES §四没申报 | 补进 NOTES 的决策清单；实现本身（含 tie 平均秩、单类返回 `None`）是正确的 |
| N11 | `metrics.py:139-142` | `if r.get("oracle_soft_iou")` 用真值判断，`oracle_soft_iou == 0.0` 的样本会被静默排除出比值统计 | 改成 `is not None and > 0` 并单独计数被排除的样本 |
| N12 | `trainer.py:242-252` | 无滚动删除；159,215/32 ≈ 4,975 步 → 每臂 ~11 个 checkpoint。W07/W08 各 ~280 MB → 单臂 3.1 GB，8 臂 ~20 GB | 按 §10.4 的惯例保留 3 个 + 保护 0.5/1.0 epoch，或确认磁盘够 |
| N13 | 仓库层面 | `q3vl/` **整个目录未入 git**（`git status` 只显示 `?? q3vl/`），因此本次审阅无法用 diff 证明"Where-A 未被篡改"，只能靠 mtime + 测试全绿旁证 | 建议在 S5 之前把 `q3vl/` 入库并打 tag；否则 §13.1 要求的"config/ 快照 + git commit"对模型代码本身是空的 |

**排期数字（供主 agent 排 wave 用，非缺陷）**：train 159,215 样本 / effective batch 32 = **4,975 optimizer steps/臂**。
每 500 步一次四上下文全量 eval = 4 × 896 = 3,584 次编码，10 次 eval ≈ 35,840 次 → 约为训练前向量的 **22.5%**。
按 §11 一卡一臂、8 臂 4 个 wave，这部分开销不可忽略，`--eval-limit` 会违反 §5.4 的"每个 checkpoint 四上下文全报"，
建议按原样跑但把 eval 墙钟单独计入排期。

---

## 四、对主 agent 已有裁定的意见

| 裁定 | 我的意见 |
|---|---|
| **D-B1**（softIoU 进 loss 与选择，红线判旧战役语境） | **同意保留协议字面**，但请把一个后果写进 REPORT 的"设置"节：`local_soft_iou_median` 同时是**主损失的支配项**和**选择规则的第一顺位**，因此「median soft-IoU ≥ 0.75」这条 gate 实质上只在回答"训练收敛了吗"，**不是独立检验**。归因重量必须落到那些**没有**被直接优化的量上：`boundary_f1`（loss 里只有 0.10 权重）、`p10`、`auc_target`、以及 null/shuffled 两个 Δ。另：`soft_iou_vs_oracle_ratio` 仍然可信——分子分母都由同一目标优化，比值是公平比较。 |
| **D-B2**（post-norm，全局统一） | **同意**，实现与裁定、文档三者一致，且 teacher/generated 共用同一函数。唯一补充见 N1：常量放在 whereb 包内，Stage-What 有静默分叉的空间。 |
| **D-B5**（global 进训练、oracle 辅助关闭） | **同意 global 进训练**（否则 `global soft-IoU ≥ 0.98` 这条 gate 不可达）。但请注意 **B5**：这条裁定的**未申报副作用**是 §5.5 的 stage-1 辅助权重被打到 ~0.47 倍，而 stage-1 存在的全部理由就是"用大权重压住早期坍缩"。需要一次追加裁定（batch 平均 vs 有-oracle 样本平均）。 |
| **D-B11**（train 段 oracle latent 为硬前置，排进 GPU 关键路径） | **同意，且提高紧急度**：`make_oracle_latents.py` 目前**跑不起来**（B2）。请在排期前先要求实施者用 `--limit 8` 真跑一次通，再外推墙钟——PENDING S1 的"墙钟未知，必须先小规模实测"是对的，但前提是脚本能跑。 |
| **D-B3/B4/B6-B10/B12-B14**（保守默认） | 除 D-B6/D-B7 涉及的 shuffled 语义（**B3**）外，其余默认我都认为合理，无异议。特别地 D-B3（缓存 ids 而非 hidden）是本包最好的一个设计判断，D-B4（hi-res 算 L_mask）的理由（3px 容差在 32×48 网格上无意义）也站得住。 |

**新增待裁决项**（本次审阅提出，不在 NOTES §四里）：

1. **D-B15**：shuffled 是否同时交换 prompt 里的 instruction（B3）。若维持"只换 where 正文"，§5.6 第 6 行 gate 必须改名并重新预注册判据。
2. **D-B16**：oracle 辅助项的归一化分母是 batch 大小还是"有 oracle 的样本数"（B5）。

---

## 五、判决

**BLOCKER 数量 5**（B1 preflight 空转 / B2 两个前置脚本崩溃 / B3 shuffled 未换指令 / B4 autocast 漏进 `s_low` / B5 辅助权重被稀释未申报）。

**是否准许进入 GPU preflight 与正式训练：否。**

- **S1/S2（oracle latent、generated context）**：B2 未清，脚本第一个样本即崩，**不得提交**。
- **S3（§14 项 7b/8b preflight）**：B1 未清，当前 `--with-model` 是空转并报 PASS，**跑了等于没跑**，不得据此放行。
- **S5（8 个主臂）**：B3/B4/B5 均属"跑完再改 = 全部重跑"的类别（预注册 gate 语义、受控对比的数值口径、预注册 loss 权重），**必须在开跑前清完**。

**放行条件**：B1、B2、B4 修复并附复现证据（B2 需 `--limit 8` 的真实跑通日志，B4 需给出 `s_low.dtype == torch.float32`
的训练期断言）；B3、B5 由主 agent 出裁定并落到代码与 NOTES；然后重跑 CPU preflight + 170 单测 + 真实 GPU preflight（含 7b/8b），
再进入 S1→S2→S3→S5。

**必须说明的正面结论**（避免以上 blocker 掩盖实现质量）：结构层（connector 四步顺序、zero-init gate、四结构装配、
参数量表、全掩码 softmax、只出全局参数无 dense logits）、loss 逐符号、九项 gate 与 lexicographic 选择、
禁回退 GT 的结构性设计、缓存 ids 重放同一 encode 的口径设计、以及与 Where-A 的 interop（用对方 packer 真打 shard）
**全部经得起逐条对照**，170 + 120 个测试真实全绿，参数量表可独立复算对上。
五个 blocker 集中在**"声明与实现不符"和"没跑过的路径"**这两类，不是设计错误。
