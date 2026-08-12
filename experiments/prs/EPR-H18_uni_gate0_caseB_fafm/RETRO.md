# EPR-H18 uni_gate0/caseB(FAFM Gate 0)

状态: MERGED(B 案唯一晋级,进 fafm_probe)
目标指标: 预注册:G0a 叠加线性性残差 ≤1%;G0b 颈部重建上界全体 ≥0.93、分家族 ≥0.88。
输入: S-train local **1,000/75,544(1.3%)**——探针档(S6 风险清单首位;G0a n=256);零训练闭式岭解。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/uni_gate0_20260811/caseB_fafm/{REPORT.md,metrics.json,viz 12,config,logs}`
指标变化: G0a worst 中位残差 **4.67e-16** PASS(U_I 严格线性 ⇒ 固定引导线性化 A_I 免实现);G0b 全体中位 **0.9967** PASS;分家族 linear 0.9990 / band 0.9971 / radial 0.9937 / semantic **0.8949**(门 0.88,富余仅 +0.015,p10 0.7269);`verdict: pass`。
结论一句话: 颈部线性 + 自由粗场上界两门全过,「自由粗场 0.895 ≫ 71 维码 0.586」的排序是本轮最强单一发现(独立支撑 S5);B 案获准开探针臂——semantic 族边缘通过,探针须按「天花板/实测/占比」三列读。
冲突/修正记录: 1.3% 训练数据探针档,预注册风险已写入 LEDGER §四(三案排序可能随全量翻转);探针数字不得当全量臂结果引用(scale_deviation 预注册);K9 去重:与 what-misc 重复条目状态一致,直接合并;下游 fafm_probe(EPR-H20)4/8 判据过是本 Gate 的下游而非修正。
