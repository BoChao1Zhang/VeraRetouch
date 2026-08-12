# EPR-H01 where_a(BA-0/1/2/3 basis 校准)

状态: MERGED
目标指标: Phi-71 basis 对 local 掩膜的表达上界(oracle 天花板)+ 四臂校准收益归因。判据门(预注册):oracle 拟合成功率 ≥90%、残差化正交误差 <1e-4、phi Gram 条件数 <1e10、s 越域比例 ≤1%、§14 preflight 全 PASS;basis 表达力本身 NONE-PREREG(描述性)。
输入: train local l1–l6 75,544/75,544(全量,含 low);V_where local 400/400(每臂 n_ok=400, n_rejected=0);headline = normal-only n=224。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_a/`(REPORT.md、metrics.json、calibration_BA-{0,1,2,3}.json、oracle_latents/、REVIEW-result.md)
指标变化: 判据五项全过(成功率 100%、正交 7.6e-7、cond 中位 8.87e4、越域中位 0、preflight 8/8)。天花板(hi 档 cband12 中位):BA-0 0.9714 → BA-3 0.9724(BA-2 0.9736 最高但按预注册不改选);校准收益在尾部:band hi p10 0.7843(BA-0)→ 0.8258–0.8308(BA-1/2/3)。F_pre 对 SFT 不变(max_abs_weight_diff=0)。
结论一句话: 71 维 basis + F_pre 的表达上界够用(hi 中位 ≈0.97),BA-3-Joint 按预注册成为下游 oracle 基座;校准收益是二阶的、集中在 p10 尾部;已知风险 = 单活跃基元样本(30.8%)的上采样脆弱性(裁定:评测侧按 active primitive count 分层,不改训练)。
冲突/修正记录: 结果审阅(2026-08-06)初判「科学结论成立、交付形式不合规 + 3 blocker」,REPORT/metrics/viz/config 后补齐;审阅另勘误 D5 小样本(24 例)的「cband12 落差为负」在全量上不成立(hi<low 普遍,落差 −0.003~−0.008,量级不影响使用)。
