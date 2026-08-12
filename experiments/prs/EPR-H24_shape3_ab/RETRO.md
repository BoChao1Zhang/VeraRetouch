# EPR-H24 SHAPE3_A/B(eikonal 重参数化架构探针 + 同容量对照)

状态: DISPUTED(判据未接线,不可裁;待 EPR-005_shape3-evalfix 补跑后改裁;**单列上报主 agent**)
目标指标: 层-3 裁决:A(eikonal)形状残差 `shape_residual` 显著低于同容量对照 B(预注册主判据;明确声明不只看 IoU)。
输入: 各 1,200/1,200 步(A 选 step1200,B 选 step800);eval V_where local 400/400,normal-only n=224。
输出: `/home/bc/data/runs/where_b/amort_SHAPE3_{A,B}_20260811/eval_final/`;`amort_SHAPE3_{A,B}_evalfix_20260812/` **均为空目录**(补跑未落盘);裁决依据 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/geoinj_summary_20260812/SCOREBOARD.md §3.3`
指标变化: (仅 IoU 侧,非预注册主判据)A 0.73467 / B 0.75414 vs 基线 0.74172;A vs B −0.0057 (p=0.016)、A vs 基线 −0.0040 (p=0.029)、B vs 基线 +0.0017 (p=0.34);bF1 A vs B +0.0005 (p=0.90);两臂硬门全过。
结论一句话: 预注册主判据 `shape_residual` 定义了、导出了、**零调用**(WEVAL-1 型「没接线」第三次再现),两臂被误用 IoU 判读——「A 形状残差低于 B」既未证实也未证伪,层-3 结论暂缺。
冲突/修正记录: K11:任何层-3 形状裁决 RETRACTED;IoU 侧读数仅可作描述性引用(「eikonal 不涨 IoU 且略输对照」,禁引申);次要:SHAPE3_B 的 `metrics.json::arm` 误记为 `"P3prime"`;两个 `*_evalfix_20260812/` 目录为空,补跑未落盘;S15(c) 定为不可裁,待 EPR-005 补跑。
