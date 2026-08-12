# EPR-H03 arm_W02(MC8-Joint + cband12 终局)

状态: MERGED(负结果,与 W01 同构失败,支撑 S4)
目标指标: 同 W01 的 10 门 gate 板(预注册同版)。
输入: 同 W01(train 159,215;eval V_where 896/896);全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/arm_W02.json`
指标变化: final local_soft_iou_median **0.5617**(门 0.75,FAIL);vs oracle ratio 0.582(FAIL);p10 0.296(FAIL);shuffle drop 0.126(FAIL);s_std_ratio 0.449(FAIL)。gate **4/10 过,WHERE-GATE-FAILED**。
结论一句话: 换 readout(cband12)不改变死法——与 W01 同构失败(0.5617 vs 0.5565),坐实病根不在 readout 而在「产生场的方式」,两臂一并进死亡路线清单。
冲突/修正记录: 无(LEDGER 0.562 与 metrics 一致)。
