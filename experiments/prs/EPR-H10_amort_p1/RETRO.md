# EPR-H10 amort_p1(Phi-71 前馈臂)

状态: MERGED(S5 出处;探针档,量级待全量复现)
目标指标: Phi-71 中间层在前馈链路是资产还是负债(与 P3′ 步数匹配 A/B)。硬门(预注册):面积比 ∈[0.8,1.5]、换主体 Δ>0、antonym |Δ|≤0.05;M3 证伪列 corr(先验)−corr(GT) <0 才可晋级。
输入: train 池 normal-only 42,752;**1,200 步 × batch 32 = 38,400 次呈现 ≈0.90 epoch(<1 epoch,探针)**;eval V_where local 400/400(headline normal-only n=224,per_sample 2,400 行)。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/amort_p1_20260810/`(BOARD.md、metrics.json、per_sample.jsonl;无 REPORT.md,板式交付)
指标变化: top-k IoU 中位(normal-only)**0.7095**(W01 0.4874 / 中心先验 0.4853 / 地板 0.2254);vs 中心先验配对 Δ +0.2268(p=1e-4);corr(先验)−corr(GT) = **−0.1366**(M3 未复现);硬门全 PASS;holdout 半区 0.7045。**配对 A/B:P1 − P3′ = −0.0143(p=0.0006)**。
结论一句话: 71 维码前馈链路可用且远胜 W01,但显著劣于不经 Phi 的自由粗场(P3′)——Phi-71 在前馈链路是负债(S5);两臂均 <1 epoch,负债量级待收敛后复现。
冲突/修正记录: 无;rider——探针 <1 epoch,S5 的「负债」方向可信、量级待全量复现。
