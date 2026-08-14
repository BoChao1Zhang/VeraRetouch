# IMPL_NOTES · EPR-012 ~ EPR-016 五特性实现（执行 subagent，2026-08-13）

范围：`q3vl/` 下的实现落地。五个特性互相独立、默认全关；全关时训练/推理路径 = ST_LANG baseline。
本文件只记录**假设、与提案有出入之处、待决策项**，不含解读与预测。

参考代码在线核实（2026-08-13 当日打开 raw.githubusercontent.com 原文，逐条确认后才写代码）：

| 来源 | 核实到的事实 | 与本实现的关系 |
|---|---|---|
| SAM `segment_anything/modeling/mask_decoder.py` | `MLP` 类：`h = [hidden_dim]*(num_layers-1)`，`zip([input_dim]+h, h+[output_dim])`，forward 前 n-1 层 ReLU，`sigmoid_output` 时末端 sigmoid | `uniq.SelMLP` 逐行同形 |
| SAM2 `training/loss_fns.py` | `actual_ious = area_i / clamp(area_u, min=1.0)`；`use_l1_loss` 分支 L1/MSE；`supervise_all_iou=True → loss_multiiou.mean(dim=-1)`，False → 只取 best_loss_inds | L1 默认 + 全候选 mean 默认，两个 ablation 旗标各对应一支 |
| Mask2Former `mask2former_transformer_decoder.py` | 全空行重置 `attn_mask[torch.where(attn_mask.sum(-1)==attn_mask.shape[-1])] = False`；层内顺序 cross→self→FFN；`(sigmoid < 0.5).bool()` + `.detach()`；self-attn `q = k = with_pos_embed(tgt, query_pos)`、v = tgt | `RefineLayer` 与 `_run_refine` 同形（norm 位置的偏离见 D2） |
| K-Net `knet/kernel_updator.py` | 四个 Linear + `input_norm_in`（input_gate）/`norm_in`（update_gate）/`norm_out`（param_out）/`input_norm_out`（input_out`）；`gate_norm` 仅在 `gate_norm_act` 时使用 | `KernelUpdate` 用四个独立 LayerNorm，未实现 `gate_norm_act`（K-Net 默认关） |

---

## A. 与提案有出入 / 需要裁决的实现决定（保守默认继续，未静默改设计）

**D1 · EPR-014：提案内部自相矛盾（结构行 post-norm vs 初始化行 step0 等价）。**
「模块内部结构」第 (11)(12) 项写 `MultiheadAttention + LN`、`FFN + LN`（K-Net 原文的 post-norm
`obj_feat = self.attention_norm(...)`），但「初始化」行要求零初始化出口后 `q^s ≡ q^0`、`s^s ≡ s^0` = baseline。
post-norm 下 `LayerNorm(q + 0) ≠ q`，两条要求不可兼得。**采纳初始化行**（任务卡把 step0 等价列为硬要求，
且 EPR-013 提案对同一取舍已显式写明「选 pre 是为了 step0 等价」）：`KernelUpdate` 的 attention 与 FFN
子层改 pre-norm。KernelUpdator 内部四个 LayerNorm 位置与 K-Net 完全一致，未动。
→ **待人类确认**：是否接受 EPR-014 也走 pre-norm。

**D2 · EPR-013 / EPR-014：query 间 self-attention 的 value 不加 query_pos。**
M2F `SelfAttentionLayer` 是 `q = k = tgt + query_pos`、`v = tgt`（已在线核实）。`RefineLayer` 照此实现；
`KernelUpdate` 的 query 间 attention K-Net 本无 query_pos，故不加。

**D3 · EPR-015：dropout 后只剩 1 个假设时的 δ̂。**
提案给了三个 NOVEL 细节（winner 在 kept 内取 / ε 按 kept 摊 / 全 drop 回退），未覆盖「kept 只剩 1 个」。
本实现取 `δ̂_winner = 1.0`（没有输家可摊 ε），而不是 `1−ε` 后丢掉 ε 那份质量。ε 质量守恒（Σδ̂ = 1）。
hdrop 默认 0，主臂 hdrop=0.01 下 K=8 全 drop 概率 ~1e-16，此分支实际不会触发。

**D4 · EPR-012：`--uniq-iou-stability` 实现成 float 阈值而非 bare flag。**
提案「入口旗标」行写成开关；实现成 `--uniq-iou-stability FLOAT`（默认 0.0 = 关，SAM2 默认值 0.98 需显式给出），
这样阈值本身进 run_config、可做敏感性行。delta 固定 0.05（SAM2 `mask_decoder.py` L28）暂未开旗标。
回退实现挂在 `UniQHead.select_index`（`model.py` 与 `uniq4.py` 两个消费点共用一条选择规则）。

**D5 · EPR-012：IoU 目标口径按提案的 NOVEL 适配，不用 SAM2 的 `logits>0`。**
目标 = `hard_iou(topk_mask(mask_of(s_k), k_area), topk_mask(gt, k_area))`，`k_area = gt_area_k(gt)`，
与 `evaluate.py:_uniq_row` 的 `uniq_query_ious` 同一组函数（已写成断言测试
`test_iou_target_matches_evaluate_uniq_query_ious`）。这是判据红线（面积匹配 top-k）要求的偏离，提案已标 NOVEL。

**D6 · EPR-013 + EPR-016 组合时 `query_pos` 的形状。**
`query_pos` 是 (K, ch)；EPR-016 训练态行数为 (1+m)·K。实现取 `query_pos.repeat(1+m, 1)`（各辅助组复用同一套
位置嵌入）。提案未覆盖此组合。两特性各自单开时不受影响。

**D7 · EPR-016：辅助 embedding 种子取 20260813，暴露为 `--uniq4-aux-seed`。**
提案只写「辅助组用独立固定种子，落 run_config」，未给数值。取值与正式组（20260812）不同即可，已进
`uniq4b_setup.json` 与 `model.facts()`。

**D8 · 统一 `aux_supervision` 通道，替代提案里的两个不同键名。**
EPR-013 提案要 `s_all_aux`（列表）、EPR-014 提案要 `s_all_stages`（含最终场的列表）。实现统一成一个
`out["aux_supervision"] = [{"s_all", "cls_logits"?, "sel_logits"?, "weight", "tag"}, ...]`，trainer 只有一个循环。
语义与两份提案一致：受监督场 = 中间场 + 主场；EPR-013 的中间场不带 cls/sel（只挂最终层），EPR-014 每 stage
带自己的 cls/sel。terms 后缀 `_ref{i}` / `_st{i}`，对应 M2F `criterion.py` 的 `k + f"_{i}"`。

**D9 · EPR-014 的入口用 wrapper-of-wrapper，不复制 seam 代码。**
提案说「新建 `run_uniq5_arm.py`，照 `run_uniq4b_arm.py:16-61` 的 wrapper 模式」。实现里 `run_uniq5_arm.py`
只解析 `--uniq5-*`，通过 `run_uniq4b_arm.EXTRA_HEAD_CLS/EXTRA_HEAD_KWARGS/EXTRA_SETUP` 三个模块级变量把
`UniQ5Head` 装进同一条 seam，再 `return run_uniq4b_arm.main(rest)`。sha256 冻结记录里同时含 uniq4.py /
uniq4b wrapper / uniq5.py / uniq5 wrapper 四个哈希。

**D10 · 头的构造参数走 `VARIANT4["head_cls"] / VARIANT4["head_kwargs"]`（NOVEL 管道）。**
提案各自说「进 VARIANT4 并传入 UniQ4Head」。实现给出一个通用的 kwargs 字典而不是逐个字段，
`AmortModelV4` 只有一处 `head_cls(..., **head_kwargs)`。默认 `head_cls=None`（= UniQ4Head）、`head_kwargs={}`，
默认路径与改动前逐字相同。

**D11 · `LossWeights.to_dict()` 返回类型从 `dict[str, float]` 放宽到 `dict[str, Any]`。**
新增两个 bool 字段（`uniq_iou_winner_only` / `uniq_iou_mse`）。落盘（`trainer.setup()["loss_weights"]` 与
`loss_preregistration.json`）自动带出全部新字段，无需另接线。

**D12 · EPR-012 的 `w_iou` 与 `uniq_sel` 的联动只在 `--uniq-iou-head` 打开时发生。**
`--uniq-iou-weight` 单独给值而不给 `--uniq-iou-head` 时不生效（避免「MLP 头没换却回归 sigmoid 前 logit」的
半开状态）。`--uniq-iou-keep-ce` 时 `uniq_sel` 保持 0.05，两项并联。

---

## B. 明确未实现（消融表里有、旗标行里没有的项）

以下消融行目前没有命令行开关，需要时再补，不影响主行：

1. EPR-015「松弛只作用于 BCE 项 vs 作用于全部 mask 拟合项」——当前固定为提案主行口径
   （作用域 = bce + sdf + area，sep 只回传赢家，fake/cls/sel 不动）。
2. EPR-016「辅助组不带空场 loss」——当前辅助组恒走 `is_fake` 分支（= 提案默认）。
3. EPR-016「辅助组 token 与正式组共享 embedding」——当前恒为独立 `aux_embed`（= 提案默认）。
4. EPR-014 `stage_loss_weights` 未开旗标，恒为 `[1.0]*S`（K-Net `knet_s3_r50_fpn.py:69` 默认）。
5. EPR-012 stability 的 `delta` 未开旗标，恒为 SAM2 默认 0.05。
6. EPR-013 mask annealing 的 `poly_power` 未开旗标，恒为 0.9（EoMT `mask_classification_panoptic.py:33`）。

## C. 等价性的验证方式

用**冻结的参考实现**而非 `git stash` / 读 HEAD：`q3vl/whereb/amort/tests/test_epr012_016.py` 顶部的
`_ref_wta` 与 `_ref_head_forward` 是改动前 `uniq_wta_loss` 汇聚段与 `UniQ4Head.forward` 的逐行转写，
断言 `torch.equal`（不是 allclose）。理由：git 基线一旦被提交就漂移，测试会失效；冻结转写让等价性断言
永久有效，且审阅者能直接对着两段代码看差异。EPR-015 的短路分支（`uniq_eps == 0 and uniq_hdrop == 0` 时
原样走 `total = per[j].total`）与「hdrop=0 不抽任何 RNG」都各有一条断言。

## D. 显存探针的偏离记录

探针用 ST_LANG 形态、全部新旗标关闭，但为控制时长偏离了正式配方三处，**只影响耗时不影响每步显存**：
`--norm-samples 64`（正式 256，no-grad 逐样本前向）、`--eval-limit 8`、`--quick-eval-limit 8`。
步数 `--max-steps 20`，`--effective-batch 32` 固定，mb8 → 累积 4，mb16 → 累积 2。
两个口径分别取 `torch.cuda.max_memory_allocated()`（在 `AmortTrainer.train` 入口 reset、训练循环结束时读）
与训练全程每 0.5s 采样的 `nvidia-smi memory.used` 峰值，落在各自 run 目录的 `mem_probe.json`。

实测（cuda:0 独占，gpu1 的 databuild 未触碰；两个 run 目录保留）：

| 臂 | 累积 | torch max_alloc | torch max_reserved | nvidia-smi 峰值 | 20 步训练墙钟 |
|---|---|---|---|---|---|
| `probe_memprobe_mb8` | 4 | **11.824 GB** | 14.293 GB | **15.337 GB** (15705 MiB) | 123.8 s |
| `probe_memprobe_mb16` | 2 | **15.097 GB** | 24.881 GB | **25.923 GB** (26545 MiB) | 112.7 s |

mb16 未 OOM。两个 run 的 `config/uniq4b_setup.json` 记为 `refine_layers 0 / aux_groups 0 /
head_cls UniQ4Head / head_kwargs {}`，`config/loss_preregistration.json` 记为
`uniq_iou 0.0 / uniq_eps 0.0 / uniq_hdrop 0.0 / uniq_iou_winner_only false / uniq_iou_mse false`
——即探针确实跑在「五个特性全关」的形态上；steps.jsonl 首行无任何 `_ref*/_st*/_aux*/uniq_iou` 列。

## E. 实现审阅 blocker 修复（2026-08-13）与其中的保守取舍

B1/B2/B3/B4/B5/B6 + S1/S4 已按审阅给的修法落地（改动文件见下节汇总）。三处审阅原文未指定、
按保守默认处理并在此记录，未静默拍板：

1. **B1 的标签集合尊重两个消融开关**：`deep_supervision_tags()` 在 `stage_supervision=False`
   （EPR-014 消融）与 `refine_aux_loss=False`（EPR-013 `--uniq4-refine-no-aux-loss`）时返回空列表。
   这两行是**提案里预注册的消融**，本就不产出中间列；若照字面按 `n_stages`/`n_refine_layers`
   强求列存在，会把合法消融臂判死。`aux_groups` 没有对应开关，恒要求 `L_*_aux{g}`。
2. **B1 在 `--eval-only` 路径跳过**：eval-only 不训练、本 run 没有自己的 `steps.jsonl`，
   板上记 `deep_supervision_check.skipped = "eval_only"`，不拿别的 run 的日志顶替。
   非 eval-only 路径下日志缺失/为空 → `_first_step_row` 返回 None → 断言 raise（缺日志不算通过）。
3. **B5 的 aux_groups 断言不按「首步」而是每个训练 micro-batch 都查**，且**不以
   `model.training` 为前置条件**：它要抓的失败正是「模型停在 eval 模式继续训练」，
   若只在 train 模式下检查就恰好对该场景失明。`compute_micro_batch` 仅由训练循环调用，
   开销是一次 dict 查找。

改动清单（仅这 7 个文件）：`amort/evaluate.py`（B1：`deep_supervision_tags` +
`assert_criteria_ran(head_facts=, steps_row=)`）、`amort/losses.py`（B2：`aggregate` 汇总
`uniq_*` stats 为 `_mean/_median/_n`）、`amort/trainer.py`（B5 断言）、`amort/uniq4.py`
（B6：退火放行加 `and self.training`）、`scripts/run_amort_arm.py`（B3 拆
`iou_as_field_target`/`iou_as_selection_target`；B5 `try/finally` 恢复 train；B1 出板路径接线 +
`_first_step_row`）、`scripts/run_uniq4b_arm.py`（S4：setup 记 head_kwargs 实际值，原始 CLI 旗标
另存 `cli_flags`）、`amort/tests/test_epr012_016.py`（B4/S1 改强 + 5 条新测试）。
