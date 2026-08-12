RUNNING

# EPR-001 · PCH 注入 · GT 码上界档（M1 go/no-go）

> 本文档是在途实验的**事后补件**：判据在 2026-08-12 09:20 提交作业时就已冻结
> （提案 `docs/PROPOSAL_geometry-injection_2026-08-11.md` §1.3 / §4 + HANDOFF §3.1），
> 本文只是把当时的预注册快照搬进新规范的格式，**没有任何判据被重新设计或放宽**。

## 1. 目标指标与基线

- **目标指标**：headline **mIoU**（V_where local，**normal-only**，面积匹配 top-k soft-IoU）
- **基线值**：**0.7909**
- **基线出处**：`/home/bc/data/runs/where_b/amort_P3prime_cont2_20260811/eval_final/metrics.json`
  的 `.topk_iou_median_normal_only`（= 0.79095，n=224，产出它的权重是 `.checkpoint_selection.selected == 3500`）
- **本臂 M0 就是这块板本身**：基座冻结，只训注入器 ⇒ Δ 里不可能混进续训收益。

## 2. 预注册数字（冻结）

| 门 | 数字 | 触发处置 |
|---|---|---|
| Gate-0 晋级 | Δ ≥ **+0.015** | 三臂全线开跑 |
| 灰区 | +0.008 ≤ Δ < +0.015 | 允许**一次** tap 结构迭代，判据数字不许改 |
| 证伪 | Δ < **+0.008** | 20 维码路线整体证伪，**三臂全停** |
| G4 回退健全 | 空码档 \|Δ\| ≤ 0.003 | 违反 = blocker，全部 Δ 无效 |

## 3. 假设一句话

成功则能说：**构造侧 GT 几何码（形状+方向）注入冻结 dense 头，能把 mIoU 推过
(image, instruction) 的重放上界**；失败则不能说 reasoning 里的几何信息可被注入利用
——注意失败只否定**这 21 维码**这一形态，不否定文本里其他几何信息（见 EPR-004 的发现）。

## 4. 输入

- 训练：local train **42,752**（`exclude_low=True`，render_mode=local）**全量**
- 评测：V_where local **400** → headline **normal-only n=224**
- 切分：沿用 sha1(sample_id) 规则族的 selection/holdout 半区（202/198）
- 几何码来源：`.vrmeta.json` 的 `slot_id`+`region` → `geom_features_from_vrmeta`（21 维，**零生成误差**）

## 5. 与基线的唯一差异

**只改一件事**：在几何头卷积塔的**倒数第二层特征**上加一个零初始化的 PCH 残差
（`residual = pch(codes, geom_code)`），其余一切（数据、loss 五项、上采样门控、
checkpoint 三硬门、seed）与基线逐字相同。基座**全部冻结**，唯一可训参数 = PCH **3,855,296**。

> 不用 broadcast 形态：它把 stem 输入通道 1025→1046，无法 resume（已炸掉三个作业），
> 且即使把新通道置零也做不到逐位一致（1046 与 1025 通道卷积规约顺序不同，实测 7e-9）。
> PCH 逐位一致，已单测。

## 6. 判据表（含运行时断言）

| 列 | 规则 | 运行时断言 |
|---|---|---|
| mIoU（主） | 面积匹配 top-k，normal-only，禁逐场调阈值 | `headline_normal_only.n == 224` |
| grid 边界 F1 | 禁像素级 3px 版 | 随主表落盘 |
| 中心先验列 | 零参数 −中心距离场，同支撑同阈值化 | 每行必报，`baselines` 与 headline 同口径 |
| 随机地板 | 同上 | 同上 |
| Δ_shuffle | EPR-003 提供（同码位数、槽打乱） | 独立臂 |
| Δ_const | **本轮缺**（见 §8 回退/缺口） | — |
| 配对检验 | 同 sample_id 配对 + 10,000 次 sign-flip 置换 | `paired_delta` |
| **AUC** | **禁**（红线） | 板内 `auc` 键数必须为 0 |
| 注入器活性 | 残差不得恒为 0（否则测的是基线） | 训练日志 `grad_norm > 0`；空码档残差必须 == 0 |

## 7. 资源预算与回退

- 预算：**2.5 GPU·h**（gpu0，max_steps 1500 / max_hours 2.5，实排 1336 步）
- 显存：实测 20.2 GiB，单卡独占
- 回退预案：Δ 落灰区 → 按 D-2 做**一次** tap 结构迭代（tap A/B 单开）；
  Δ < +0.008 且 EPR-004 显示解析码与 GT 码**信息不同源**（已发生）→ 结论限定为
  「vrmeta 码无效」，不外推到文本几何。

## 8. 已知缺口（诚实记录，不改判据）

1. **Δ_const（M2b 常量码臂）本轮没排**——三臂只覆盖 GT/解析/shuffle。
   shuffle 保持激活位数，能隔离「几何内容 vs 多了通道」，但**不能**隔离
   「码是常量时注入器纯粹靠增容涨分」。补臂建议列入后续 EPR。
2. **PCH 只注入几何头**，语义族样本（17.5%，路由到 m_sem）不受影响 ⇒ headline Δ 偏保守。
   板上同时报几何族子集 Δ。
3. 实现版 PCH 与提案 §2.2 字面张量流不同（预算与关键性质对齐，非 TwoWayTransformer 双抽头），
   `GeoCode` 契约的 `c_cont`/`conf` 未落地，conf 由码密度代理。

## 9. 状态流转

`PROPOSED(补) → APPROVED(主 agent 2026-08-12 追认三项 deviation) → RUNNING(09:20 提交) → SETTLED`

## Changelog

- 2026-08-12：建档（在途补件）。基线由 0.7622 更正为 **0.7909**（CONT2 落盘且更高，
  HANDOFF §3.4 要求对最新基线算 Δ），resume 权重相应改为 cont2 step3500。
