# EPR-H30 C04(OracleWhere-SB48,ceiling 臂)

状态: REJECTED(无终态且按裁定不重跑;**非假设否决**)
目标指标: 同 C03(SB48 变体);判据门同 C 波(预注册)。
输入: train 截断——steps.jsonl 实读 **3,443/4,975 步**(3h42m)后 rc=1 崩溃;在线 eval 6 次(至 step 3000)——**探针(训练截断,无终态)**。
输出: `/home/bc/data/runs/what/C04/`(step500–3000 + epoch0.5 共 7 个 checkpoint,**无 what_final.pt**;eval.jsonl 6 行;train.log 含崩溃栈)
指标变化: 在线(256 子集,generated):local_lut_de00_median 13.060(step500)→ 10.832(step3000),gate_pass 全程 False(step3000 bake_err_p99 5.773e-3 ✗)。
结论一句话: 崩于单一样本 `sft_ef61ee381c35f9dc5fd29e611402686b` 缺 cband12 oracle fit(train 覆盖 75,543/75,544;presence≠coverage 是唯一实质教训),无可用终态结论;根因修复(`--oracle-uncovered fail/drop`)已合入但从未真机运行。
冲突/修正记录: LEDGER 写「~3,000–3,440 步」,实读 steps.jsonl 为 **3,443** 步,以实读为准(LEDGER 勘误项)。
