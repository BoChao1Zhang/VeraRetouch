# EPR-H19 uni_gate0/caseC(FPD Gate 0)

状态: MERGED(实验为有效负结果;**方案 FPD REJECTED 入死亡清单**;几何段诊断工具移交 B 案)
目标指标: 预注册:E0 几何家族 ≥0.92;轮廓(oracle 段选择)≥0.80(**证伪线 <0.70**);箱宽场空间校准 ≤0.02。
输入: V_where local 397/400(3 条丢弃;实质接近全量);零训练。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/uni_gate0_20260811/caseC_fpd/{REPORT.md,metrics.json,viz 24,config,logs}`
指标变化: 几何家族 mean **0.9484** PASS(band 0.9546 / linear 0.9851 / radial 0.9253);轮廓 **0.5722 < 0.70 触发证伪** FAIL(阶梯定位:最佳单段 mean 0.4529 → 段 OR 0.5812 → 上采样后主管线 0.3567——挂在**段支撑**,非选择/传播/量化);箱宽 **0.0095** PASS;量化代价中位 4.7e-7、mean 8.6e-5;`verdict: fail`。
结论一句话: 程序化表示对几何原语成立、对语义轮廓证伪(Ncut 段支撑装不下语义轮廓)——C 案不开臂,几何段保留为零参数可解释家族诊断工具移交 B 案复用。
冲突/修正记录: K9(状态词裁决):两分片 MERGED vs REJECTED 冲突,裁定实验 MERGED、方案 REJECTED 入死亡清单;口径注记(吸收自 what-misc 重复条目):判据行 0.5722 取的是最有利宽口径 `contour_softiou_hi_nolposs`(oracle 段选择)的 mean,主管线 `contour_softiou_hi` mean 仅 **0.3567**——两口径均低于证伪线 0.70,结论不受影响,引用时须写明口径;本条为去重合并版。
