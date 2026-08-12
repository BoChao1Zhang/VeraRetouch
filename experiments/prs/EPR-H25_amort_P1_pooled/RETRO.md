# EPR-H25 POOLED(amort_P1_pooled,池化对照)

状态: MERGED(M5 终裁证据;支撑 S13 的 pooled-FiLM 瓶颈判断)
目标指标: 空间条件通道换成 pooled 池化后损失多少定位能力(与 P1 同步数配对);NONE-PREREG(对照板,无晋级门)。
输入: 1,200/1,200 步(与 P1 步数匹配);eval V_where local 400/400,normal-only n=224。
输出: `/home/bc/data/runs/where_b/amort_P1_pooled_20260811/eval_final/{metrics.json,per_sample.jsonl}`(无 experiments/ 交付);配对数见 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/geoinj_summary_20260812/` §2
指标变化: **0.62219** vs P1 0.70951,配对 Δ = **−0.0856**(p=1e-4);corr(先验)−corr(GT) = **+0.123**——全板 11 个 run 中唯一为正(输出被中心先验支配的 W01 型病态方向);硬门虽全过。
结论一句话: 把空间通道池化掉直接损失 0.086 IoU 且把模型推回「先验支配」病态——空间 dense 通道是 P3′ 配方的必要件,pooled 表征装不下几何。
冲突/修正记录: 无。
