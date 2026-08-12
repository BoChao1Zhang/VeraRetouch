# EPR-H08 amort_e3(W01/W02 死因终裁:M3 vs M4)

状态: MERGED(勘误后版本;S4 出处)
目标指标: 判别 M3(照抄几何先验)vs M4(指令没进网络)。M3 条件(登记):corr(输出, 中心先验) > corr(输出, GT);antonym 预注册为**不变性负控制**,|Δ|≤0.05 为 PASS。
输入: V_where local 400/400 逐臂(W01/W02 冻结 checkpoint,只前向;shuffled 档 382 条);全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/amort_e3_20260810/`(REPORT.md 带勘误横幅;metrics.json 的 verdict 字段显式标 `M4_no_conditioning__RETRACTED: true`)
指标变化: **M3 坐实(两臂)**:corr(输出,先验) 0.6430/0.6077 > corr(输出,GT) 0.4694/0.5251;corr(输出,P-W5 语义场) 仅 0.129/0.121;输出跨样本自相似 0.525 = GT(0.233)的 2.26 倍。条件性(正确探针):shuffled −0.145、fixed_phrase −0.230、irrelevant −0.238(指令确实在条件化);antonym |Δ| 相对幅度 0.31%/0.59% ⇒ 不变性控制 PASS。gt 0.550 只比零参数中心先验 0.517 高 **+0.033**。
结论一句话: W01/W02 死因 = M3(输出被中心先验支配,指令进来了但空间收益只有 +0.033);M4「无条件化」不成立;antonym 永远只做不变性负控制、禁止做训练信号。
冲突/修正记录: K1(裁决闭环):初稿 M4 结论 RETRACTED(2026-08-10 交付后自查)——初版把 `|gt−antonym|≈0` 读成「指令没进网络」,方向读反;antonym 只翻颜色方向词、保留主体,掩膜不动才是 PASS。勘误横幅在 REPORT §0,metrics verdict 字段同步标注;viz 尾注「M4 的肉眼版本」是初稿残留措辞,以勘误横幅为准。
