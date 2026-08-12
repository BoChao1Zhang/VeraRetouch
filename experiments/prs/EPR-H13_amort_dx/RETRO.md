# EPR-H13 amort_dx(脏边机理判别)

状态: MERGED(S7/S8)
目标指标: 判别几何族脏边机理——高频能量注入(E_HF/D_total 口径)vs 引导上采样曲率污染(κ̃ 口径);NONE-PREREG(诊断卡,无晋级门)。
输入: V_where local 400/400(analytic 320 + semantic 80),零训练,generated 上下文。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/amort_dx_20260811/{metrics.json,per_sample_dx.jsonl,geom_vocab_crosscheck.json}`(无 REPORT.md)
指标变化: D_total 全为负(P3′ analytic −0.00316 / P1 −0.00373)⇒ E_HF 口径证伪废弃;κ̃_guided P3′ analytic 3.398(P1 5.134)显著超界;DX-4 loss 审计:contour_warp 扰动下 SDF 边界罚仅占总损失 0.44%、面积带 0%(两项 blind <5%),BCE 独担 70.5%;DX-6 oracle 路由 IoU 代价 ≤5e-4。附:DX-5 子探针 RuntimeError「0 usable samples」(need_mask bug)未产出。
结论一句话: 脏边 = 引导上采样的曲率污染而非高频能量,E_HF 口径废弃、κ̃ 口径确立;SDF 边界罚与面积带罚在当前权重下惰性(S8)。
冲突/修正记录: DX-5 挂重跑(入口文档 §四);K14:κ̃ 绝对值(3.398/5.134)须带 EPR-H14 的两处度量缺陷说明(τ 按家族标定、窄带取自 GT;方向与量级由 16 样本修正后度量复现 2.26→0.088),论文表格前须全量重算;K13:S8 引「SDF 边界罚 ≤1.4%」与本卡 0.44% 为不同口径(contour_warp 扰动口径),结论(惰性)同向,引用须带口径限定。
