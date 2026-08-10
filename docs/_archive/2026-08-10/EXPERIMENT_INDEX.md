# 实验文档入口

> 从这里开始读。旧版把“计划、运行状态、初报、终审修订”放在同一页，导致同一个实验出现多个
> 相互冲突的结论。现在当前事实只从注册表进入，历史文档保留但不再充当状态页。

## 1. 三份必读文档

| 你要回答的问题 | 文档 |
|---|---|
| **现在到底跑完了什么，证据在哪？** | [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)：按 module，逐项给 `what / where / status / gate` |
| **目前好的结论、坏的结论、仍未知的是什么？** | [`EXPERIMENT_RESULTS_CURRENT.md`](EXPERIMENT_RESULTS_CURRENT.md)：按 render、what、where 汇总，并把 paper list 映射到下一步 |
| **实验原本怎么设计、判据是什么？** | [`EXPERIMENTS_v3_2026-08-02.md`](EXPERIMENTS_v3_2026-08-02.md)：协议与设计史，不是当前结果表 |

## 2. 一分钟状态

| module | 已完成到哪里 | 还缺什么 |
|---|---|---|
| **Render** | G2 证明真实数据有约 8.04 dB 局部信号；E2 受约束 s 轴闭环；RD oracle 容量在 L1–L4 强通过 | RD-STD 的 L0 全局退化未过；L6 单轴不足；RD-G 有 s 的 Stage-2 未终判 |
| **What** | A1–A5 颜色/曝光在多层可线性解码，最终 latent 仍保留大部分信息 | A6 色恒常在 connector 处断；通用 what 读出不是当前瓶颈 |
| **Where** | RO-3/PR13 证明 instruction-side 单头、融合或双线性读出可拿到 0.83–0.92 AUC_target | 免费 GL attention 与普通主体 AUC 不可用；RO-W 真实端到端读出未完成 |
| **E2E / Eval** | 两端分别有证据；G2 给出分区评测方法 | 尚无完整“真实指令→where→renderer”主榜、未见指令/LUT和 user study |

## 3. 当前总判词

```text
what：大部分低层量已经在，且像素统计不输 VLM；不值得另造一条昂贵通路。
where：信息也在，但免费 attention 读法错；应训练 instruction-side / bilinear 小头。
render：oracle s 下容量强且可烘焙；真实 readout 接入仍是最后一公里。
```

## 4. 其他文档各自只负责什么

| 文档 | 责任边界 |
|---|---|
| [`PLAN_v2_local-retouch_2026-07-31.md`](PLAN_v2_local-retouch_2026-07-31.md) | 方法假设、候选结构、失败分支；其中 H1/H3 等已被后续结果部分推翻 |
| [`DATA_ASSIGNMENT_2026-08-02.md`](DATA_ASSIGNMENT_2026-08-02.md) | 数据代号与 split；数据假设若被实测推翻必须在原处更正 |
| [`DECISIONS_2026-08-03.md`](DECISIONS_2026-08-03.md) | 决策事件日志；不承担当前数值真值 |
| [`HANDOFF_2026-08-04.md`](HANDOFF_2026-08-04.md) | 当时的交接快照；其中“在跑/待跑”已被后续实验超过 |
| [`SURVEY_papers_2026-07-31.md`](SURVEY_papers_2026-07-31.md) | 论文摘要底账；针对当前失败的方案映射已整理进当前结果文档 |
| [`reviews/`](reviews/) | 独立审阅与历史修订依据；不直接覆盖机器结果 |
| `experiments/<ID>/REPORT.md` | 单实验解释与边界 |
| `experiments/<ID>/metrics.json` | 单实验机器数值真值 |

## 5. 写新结果时的最小模板

```markdown
ID / module:
what: 一句可证伪问题
where: protocol / code / artifact / metrics
status: planned | running | completed | invalidated
gate: PASS | FAIL | MIXED | INCONCLUSIVE | N/A
result: 带口径、样本量、step/seed 的数字
conclusion: 能支持什么
limitations: 不能支持什么
supersedes: 替代了哪条旧结论
```

同一个数字只在注册表维护当前版本；计划、decision、handoff 只链接，不再复制。
