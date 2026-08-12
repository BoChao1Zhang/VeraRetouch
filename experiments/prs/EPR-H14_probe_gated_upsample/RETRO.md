# EPR-H14 probe_gated_upsample(家族门控上采样)

状态: MERGED(S7,B 档已进 P3′ 推理配方)
目标指标: 干预验证脏边机理 (b):几何族关引导 κ̃ 显著下降、IoU 不降、语义族关引导显著掉分(三判据预注册,见 REPORT §1)。
输入: V_where local 400/400(analytic 320 + semantic 80),P3′ step1200 checkpoint 纯推理,同一前向三档(A 引导全开/B 门控/C 全关)。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/where_b/probe_gated_upsample_20260811/{REPORT.md,metrics.json,per_sample.jsonl,viz_ab16/}`
指标变化: 几何族 κ̃ 3.398 → **0.080**(配对 Δ=−73.29, p=1e-4);几何族 IoU 0.7405 → 0.7405(Δ=−4.0e-6, p=0.987,零代价);语义族 C 档 bF1 −0.1069(p=1e-4)、IoU −0.0118 ⇒ 引导对语义族有真实价值;类型词路由与 GT family 一致 100%(400/400)。
结论一句话: 门控修复零训练零 IoU 代价治好几何族曲率污染,语义族保留引导(bF1 +0.107 价值),已默认进配方——但它不涨 top-k IoU,涨分靠续训。
冲突/修正记录: K14:κ̃ 度量修两处(τ 按家族标定、窄带一律取自 GT);主表 κ̃ 数字来自修正前聚合,方向与量级由 16 样本修正后度量复现(2.26→0.088),**论文表格前需全量重算 κ̃ 列**。
