# EPR-H07 amort_e2(Tikhonov 凸化)

状态: MERGED(S3:M2 修复路线关闭)
目标指标: 验证「加正则就能把 w\* 变成好目标」。判据门(预注册):重算后 oracle 解码 IoU **≥0.95** 才开 GPU 重训阶段。
输入: V_where local 400/400(n=400,band);另符号普查 train 75,544+75,543 + V_where 400;全量。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/amort_e2_20260810/`(**metrics_stageA.json**,无 GPU 阶段故无主 metrics.json)
指标变化: λ 六档扫描全 FAIL:λ₀/10 就把 oracle IoU 0.9748→**0.8640**(cond 仅降到 2.0e5);λ=10λ₀ 时 cond 5.6e3 但 IoU **0.4874**(天花板损失一半);且 **CV>1 维度占比在全部 λ 上恒为 100%**——目标从未变得可回归。第 0 项:w_dir 符号规范化训练/落盘两侧 100% 一致(max 偏差 5.6e-16),不是病因。
结论一句话: 凸化与天花板结构性绑定——病态是 oracle 0.97 表达力的价钱,付 IoU 也买不来可回归性,M2 修复路线(含基底白化)关闭,GPU 重训阶段按判据不开。
冲突/修正记录: 无撤回。新发现(0b 项,阳性):符号规范化自身引入翻转边界,3.9%(间距<1.05)–13.9%(<1.20)的训练目标坐在上面(L_dir 可从 0 跳 2);修法随 P1 输出空间监督自动消失。
