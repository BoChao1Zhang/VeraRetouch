# EPR-H17 uni_gate0/caseA(CH-NPE Gate 0)

状态: MERGED(实验为有效负结果;**方案 CH-NPE REJECTED 入死亡清单**;E0b 单条结论保留)
目标指标: 规范化算子前置门。预注册:E0a 线性性残差 ≤1%;E0b 规范化码稳定性中位放大 ≤50;E0c 重建上界全体 ≥0.85、轮廓 ≥0.75(**证伪线 <0.60**)。
输入: V_where local 400(E0b/E0c n=400;**E0a 判据段 n=256——探针档**,S6 风险清单);零训练。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/uni_gate0_20260811/caseA_chnpe/{REPORT.md,metrics.json,viz 12,config,logs}` + 总表 `uni_gate0_20260811/REPORT.md`
指标变化: E0a **0.5697** FAIL(线性分支按预注册永久退役);E0b 中位放大 **1.76–7.70** PASS(对照未正则 8e5);E0c(A71,判据档)全体 **0.7052** FAIL、轮廓 **0.5279 < 0.60 触发证伪**;A71p 全体 0.9314 但轮廓 **0.1957**(多起点 0.782 → C_ε 0.196,规范化自身崩塌);metrics `verdict: fail`。
结论一句话: 轮廓表示证伪、A 案不开臂——A71 是 71 维码装不下(上界 0.7421),A71p 是规范化本身崩塌,两种参数化各挂在不同环节,独立身份未证成;唯有「ε 正则驯服病态」(E0b,≤7.7 vs 8e5)保留引用。
冲突/修正记录: K3(勘误链):E0a 曾因探针幅度遮蔽(实现审阅 B9)虚报 0.9624 PASS,RETRACTED,重跑 0.5697 为准,FAIL 判定不变、E0b/E0c 逐位复现;K9(状态词裁决):两分片 MERGED vs REJECTED 冲突,裁定实验 MERGED(预注册证伪线触发 = 有效负结果)、方案 REJECTED 入死亡清单;本条为与 what-misc 重复条目的去重合并版(吸收 A71p 细节);E0a 判据段 n=256/400 探针档,方向可信、量级待全量(S6)。
