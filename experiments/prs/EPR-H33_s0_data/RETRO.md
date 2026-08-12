# EPR-H33 S0-DATA 数据交付校验(LEDGER 未覆盖,建议补录)

状态: MERGED
目标指标: 验证 sft2seg-20260804 交付「同一份可复推数据 + T_lut_unseen LUT 身份零泄漏」;判据门:spec §9 项 5–9 + §2.2 隔离全 PASS(预注册于 SFT spec)。
输入: 十个生产 build 全部 **172,580** 条 SFT 行 → N_effective train **159,215**(拒绝 10,221 + LUT reserve 移除 8,371);V_where 896 / V_what 897 / T_final 918 / T_lut_unseen 433——**全量**。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/data/`(PREFLIGHT_DATA.md、metrics.json、manifest、conversion_samples.md;manifest digest `9278d721…`)
指标变化: 校验列全 PASS:unseen LUT × {train, V_where, V_what, T_final} 交集全 **0**(259 unseen vs 3,149 train LUT);三集合无交集 0;bake fidelity all_pass(差异仅 JPEG q95 量化);length/shard/resume 全过。
结论一句话: 全战役的分母权威数(159,215/896/897/918/433)与「unseen LUT generalization 成立的前提」都出自这份交付,后续所有臂(含 LEDGER 〇节口径表)引用的就是它。
冲突/修正记录: 定档期的 169,260 是过滤前原始数,非最终 N_effective——metrics 已明记,防止后人误引;LEDGER 现无本条行,建议补入(裁决上报事项 3)。
