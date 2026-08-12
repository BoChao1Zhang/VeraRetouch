# EPR-H22 amort_P3prime_cont2(续训第二段,现行基线)

状态: MERGED(**现行基线 M0 = 0.79095**,S15a)
目标指标: 在 CONT 0.7622 之上继续验证「underfit 仍是主桶」;NONE-PREREG(续训无独立晋级门;其结果反过来把注入臂 M1 门基线更新为 0.7909)。
输入: 同 P3′ 池 42,752;续训至 3,753/6,000 步(`stopped_reason=max_hours`);eval V_where local 400/400,headline normal-only n=224。
输出: `/home/bc/data/runs/where_b/amort_P3prime_cont2_20260811/eval_final/{metrics.json,per_sample.jsonl}`(无 experiments/ 交付目录);读数板 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/geoinj_summary_20260812/SCOREBOARD.md`
指标变化: 0.76224 → **0.79095**(配对 Δ vs CONT **+0.0140**, p=1e-4;vs P3′(1200) +0.0394, p=1e-4;bF1 +0.0204, p=3e-4);Δ vs 中心先验 +0.3018(p=1e-4);corr(先验)−corr(GT)=−0.222;holdout 0.7959;三硬门全过;在线曲线到 step3500 仍在涨(0.8273)。
结论一句话: 续训仍在爬且已越过几何族 U_replay 0.7596 朝全体 0.8040 走——S13「已到天花板」被部分证伪(S15a),0.79095 成为一切注入臂的新基线。
冲突/修正记录: ① headline 出自 **step3500 checkpoint**(`checkpoint_selection.selected=3500`),非 3753 final——**resume 必须用 `amort_step3500.pt`**;② K6:顶层 `.baselines` 为 pooled 值的坑同 EPR-H12(normal-only Δ 一律取 `.contexts.*.headline_normal_only`);③ EPR-006_cont3 已立项接续。
