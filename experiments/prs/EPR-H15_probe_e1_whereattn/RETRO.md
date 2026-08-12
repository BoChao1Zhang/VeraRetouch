# EPR-H15 probe_e1_whereattn(attention 通路终裁)

状态: MERGED(S1、S10;方案 B/attention 差分路线因此死亡)
目标指标: `<where>`/instr attention 是否为指令条件定位场。预注册:P1 参考线 0.45;P2 vs 中心先验 Δ≥+0.10 且 p<0.01;P3 Δ_shuffle≥+0.08 且 p<0.01;P4 instr-vs-fixed ≥+0.05(FWER 置换)。
输入: V_where 896 → local 400;fit/OOF 按 source_image_id 切 193/189(18 条无 shuffle partner 排除);sink 普查 n=400。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/probe_e1_whereattn_20260810/{REPORT.md,metrics.json,pw1_sink_survey.json,b1–b3 补件,REVIEW-result.md,viz/}`
指标变化: 三池 P1 中位 0.4244/0.4335/0.4218 vs 中心先验 **0.4666**(P2 Δ 全为负 −0.029~−0.049,p_FWER≤5e-4,反向显著);P3 Δ_shuffle≈+0.002 全灭;唯一过门 = P4 物体性闸门(where_content 小目标 Δ=+0.1309, p=7.7e-5);差分臂补件同判(比 raw 好 +0.009~+0.039 但仍全踩线)。sink 普查:合取规则 9.90% 格 / 45.4% 图上质量 / 96.3% argmax,图像块首 token 400/400 皆 sink,oracle 天花板 0.787→0.729。
结论一句话: attention 通路对「有没有指代」有反应、对「指代哪一个」没有——无指令条件定位,读出路线关闭;sink 合取排除规则定稿。
冲突/修正记录: K2(裁决闭环):B3 补件三处更正、原数字 RETRACTED(单臂/合取规则数字分列;0.791 为误写,正确 0.7866;「sink 近似均匀」限中位成立);P-W1 登记线 Jaccard≥0.8 判 FAIL 但语义为「sink 不在刻板位置」,规则本身跨分辨率稳定。
