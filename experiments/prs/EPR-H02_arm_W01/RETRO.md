# EPR-H02 arm_W01(MC8-Joint + band 终局)

状态: MERGED(负结果,支撑 S4;MetaCanvas query→71 维回归路线入死亡清单)
目标指标: Where-B W01 臂 10 门 gate 板,主门 local_soft_iou_median ≥0.75、soft_iou_vs_oracle_ratio ≥0.85、p10 ≥0.55、bF1/oracle ≥0.75、shuffle drop ≥0.2、s_std_ratio ≥0.6 等(预注册)。
输入: train 159,215(sampler 实际用 159,208,teacher/generated 各半);eval V_where 896/896(local 400);全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/arm_W01.json`(setup + final + best_recorded + gate 板)
指标变化: final local_soft_iou_median **0.5565**(门 0.75,FAIL);vs oracle ratio 0.579(门 0.85,FAIL);p10 0.236(FAIL);shuffle drop 0.145(门 0.2,FAIL);s_std_ratio 0.386(门 0.6,FAIL)。gate **4/10 过,总判 WHERE-GATE-FAILED**。best_recorded 0.5608。
结论一句话: MC8-Joint+band 回归 71 维 latent 的路线终局失败——形式面(format/global/gt-gen gap)全好,空间面全塌,死因后由 amort_e3 终裁为 M3(输出被中心先验支配,normal-only 对中心先验仅 +0.0017)。
冲突/修正记录: LEDGER 写「mIoU 0.557」= final 0.5565 四舍五入,一致;train 记 159,215,sampler 实用 159,208(丢 7 条,不改结论)。
