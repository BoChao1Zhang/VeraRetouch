# EPR-H04 analysis_W01_step1500(失败画像)

状态: MERGED(失败画像被 S4/S9 引用)
目标指标: W01 step1500 失败的几何分层画像 + 长尾机制归因(预注册决策树打标;分析卡,主门 NONE-PREREG)。
输入: V_where local 400/400(per_sample 6,232 行 × 7 context);全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/analysis_W01_step1500/`(CLASS_REPORT.md、per_class_metrics.json、tail_samples.jsonl、fields/、viz/)
指标变化: 整臂(generated)local softIoU 中位 0.486,中心先验 Δ hardIoU −0.012(p=0.067,打不过零参数先验)。长尾(softIoU<0.3)n=89(22.2%):primary 机制 **area_mismatch 62/89 = 69.7%**(命中口径 97.8%,辅助口径 65.0%),其中 **87/87 全部是过覆盖**,pred/GT 面积比中位 **3.69**;s_collapse 13.5%、oracle_ceiling 12.4%。最差类别:small(0.155)/complex(0.295)/edge(0.270)/soft(0.387)/lower(0.275)。
结论一句话: W01 的失败不均匀——主因是面积失配式过覆盖(场铺开成近全局掩膜而不缩到指令区域),集中在小面积、复杂边界、离心区域;这是「输出被中心先验支配」的画像级证据。
冲突/修正记录: K12(裁决,已独立复核):LEDGER 写「area_mismatch 为主因(43.1%)」与 metrics 不符——per_class_metrics.json 的 `bottleneck_primary_share` 实读 **0.6966**,任何口径(primary 69.7% / 命中 97.8% / decile 65.0% / 全体 primary 31.5% / 全体命中 41.5%)都不是 43.1%,该数在交付内无出处;以 metrics 的 **69.7%(primary,主榜口径)** 为准,LEDGER 须勘误。
