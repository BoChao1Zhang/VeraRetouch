# EPR-H20 fafm_probe(B 案生成式探针臂)

状态: MERGED(S6;**详裁待完成**;判据⑨按 K15 改「不可判」)
目标指标: 九条预注册判据(§3.6 逐条锁定):①vs 中心先验 Δ≥+0.05 p<0.01(证伪<+0.03)②vs 回归 ③面积 W1 减半 ④指令条件性 ≥3× 全部负控制 ⑤bF1 ⑥分家族 ⑦家族模式召回 ≥70% ⑧步数-多样性(记录性)⑨c 空间 TARP ≤0.10;CFG 规则预注册。
输入: 训练 S-train **20,000/≈70,000**(3,000 步 × batch 48,探针档,主 agent 裁定 3 批准);eval V_where local 400/400。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/fafm_probe_20260811/{metrics.json,NOTES.md,config,viz}`(无 REPORT.md)
指标变化: 板记法(K15 修正):**4 过 / 3 FAIL / 1 不可判 / 1 记录性**——过:①+0.0999 p=1e-4、⑤bF1 +0.156(semantic +0.381)、⑥分家族全过先验、⑦召回 0.86;FAIL:②−0.1418、③W1 0.0602>0.5×0.0947、④反指令效应=0(负控制 0.031–0.036 反而显著);不可判:⑨TARP(判 FAIL 但 `max_abs_deviation` 落盘 null);记录性:⑧;CFG 选 g=1。A_fafm soft-IoU 中位 **0.6098**(n=400)——低于 P3′ 同 universe 档。
结论一句话: 生成式路线判据近半过、指令条件性与分布刻画两条核心 FAIL,暂居 P3′ 下风;探针数字按预注册**不得当全量臂结果引用**(`metrics.json::train.scale_deviation`)。
冲突/修正记录: K15:判据⑨降级为「不可判」(`max_abs_deviation` 落盘 null 但判 FAIL),总板由「4/8 过」改记「4 过 / 3 FAIL(②③④)/ 1 不可判(⑨)/ 1 记录性(⑧)」,详裁与补跑前须先修导出;详裁在 WHERE_STATE §四挂起未完成;实现期修掉 `collect_posterior` 静默丢样本与逐 K 串行两个自身 bug(NOTES §5)。
