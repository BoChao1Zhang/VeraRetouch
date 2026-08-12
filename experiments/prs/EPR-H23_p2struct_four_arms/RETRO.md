# EPR-H23 P2STRUCT 系四臂(p2struct .05 / p2struct_hi .15 / p2w 5× / p2w_hi 10×)

状态: MERGED(不采纳;结论经 S15(b) 改判为 null)
目标指标: 结构损失包(curv/mono)与惰性 loss 重标定(SDF/面积带加权)能否推 IoU;权重预注册({p2struct,p2w}×{base,hi} 消融矩阵);判据 = 对匹配基线的配对 Δ。
输入: 从 P3′(1200) 起各再训 500/500 步;eval V_where local 400/400,normal-only n=224。
输出: `/home/bc/data/runs/where_b/amort_P3prime_{p2struct,p2struct_hi,p2w,p2w_hi}_20260811/eval_final/`(无 experiments/ 交付,LEDGER 缺口 2);配对复算见 `experiments/Q3VL_metacanvas_where_what_20260804/where_b/geoinj_summary_20260812/paired_deltas.txt`
指标变化: 四臂 0.75221 / 0.74945 / 0.74864 / 0.74773;对**步数匹配基线** P3′(1200)=0.74172 的配对 Δ = +0.0014 (p=0.50) / +0.0011 (p=0.59) / −0.0020 (p=0.34) / −0.0013 (p=0.55)——四臂全部统计学 null;硬门全过、NaN bug 修复后 nonfinite=0。
结论一句话: 结构损失包在该权重族里什么也没买到(null),而同样算力用于续训能买 +0.02——不进配方,理由固定为「无增益」而非「费 IoU」。
冲突/修正记录: K4:S14 初判「费 IoU」(−0.0100~−0.0145)出自四臂对 2541 步 CONT 的**步数错配比较**(违反 U4 规则),错配版结论 RETRACTED,S15(b) 改判 null;「不采纳」决定不变;WHERE_STATE §二 S8 更新块与 S14 原文须挂 S15(b) rider(回改清单项);HANDOFF §2.1 曾漏列 p2struct_hi 的 0.74945。
