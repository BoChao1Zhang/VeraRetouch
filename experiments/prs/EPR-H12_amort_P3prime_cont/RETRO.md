# EPR-H12 amort_P3prime_cont(P3′ 续训主榜,2541 步)

状态: SUPERSEDED(基线地位被 EPR-H22 CONT2 的 0.79095 取代;继承链 0.7417 → 0.7622 → 0.79095)
目标指标: normal-only 面积匹配 top-k IoU 中位,把 P3′(1200) 的 0.7417 推过 **0.75 门**(预注册门 0.75;三硬门 面积比/换主体 Δ/antonym 沿用)。
输入: train local 排 low 后 42,752/42,752(≈1.90 epoch);eval V_where local 400/400,headline normal-only n=224;续训 2541/4000 步(`stopped_reason=max_hours`,墙钟 4.1h 截断)。
输出: `/home/bc/data/runs/where_b/amort_P3prime_cont_20260811/eval_final/{metrics.json,per_sample.jsonl}`(**无 experiments/ 交付目录**,LEDGER 缺口 1;引用必须指向 runs/ 路径)
指标变化: 0.7417 → **0.76224**(配对 Δ=+0.0254, p=1e-4);Δ vs 中心先验(normal-only 0.48565)= +0.2877 (p=1e-4);corr(先验)−corr(GT)=−0.197;holdout 半区 0.7595;三硬门全过(area_ratio 0.991 / swap Δ +0.0453 p=1e-4 / antonym 中位 0.0064)。
结论一句话: 续训是当时唯一被证实推动 IoU 的手段,0.75 门已过,但曲线未平、0.7622 是下界——该基线地位已被 CONT2 的 0.7909 取代。
冲突/修正记录: ① K6(强制口径规则):metrics.json 顶层 `.baselines.center_prior=0.5170` 是 **pooled 全 400** 值,normal-only 基线只在 `.contexts.*.headline_normal_only`(0.48565),混用会使每个 Δ 少算约 0.031;② 仓库内 `amort_p3prime_20260810/metrics.json` 仍是 1200 步旧值 0.7417,读者易拿错。
