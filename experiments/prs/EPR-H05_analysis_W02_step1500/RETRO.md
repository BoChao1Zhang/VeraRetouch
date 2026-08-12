# EPR-H05 analysis_W02_step1500(失败画像)

状态: MERGED(同 EPR-H04,被 S4/S9 引用)
目标指标: 同 EPR-H04,对象换 W02 step1500(NONE-PREREG 分析卡)。
输入: V_where local 400/400;全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/analysis_W02_step1500/`
指标变化: 整臂 local softIoU 中位 0.5115,中心先验 Δ hardIoU **−0.0256(p=1e-4,显著输给零参数先验)**。长尾 n=81(20.2%):primary **area_mismatch 54/81 = 66.7%**(命中 96.3%,辅助口径 57.5%),78/78 过覆盖,面积比中位 3.57。最差类别与 W01 同构(small/complex/edge/soft/lower)。
结论一句话: W02 失败画像与 W01 逐类同构(过覆盖面积失配为主因),确认失败模式是路线级而非 readout 级。
冲突/修正记录: K12(裁决,已独立复核):LEDGER 写「同上(69.8%)」与 metrics 不符——per_class_metrics.json 实读 **0.6667**(69.8% 在交付内无出处,疑似与 W01 的 69.7% 串行);以 metrics 的 **66.7%** 为准,LEDGER 须勘误。
