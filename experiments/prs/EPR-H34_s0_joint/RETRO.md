# EPR-H34 S0-JOINT Base SFT 联合 preflight(LEDGER 未覆盖,建议补录)

状态: MERGED
目标指标: 验证 Base SFT 在真实数据 × 2×H100 ZeRO-3 上无运行层障碍;判据门:任务卡 1a–1e + 生成管线单验全 PASS(spec §9「任一失败不得升级为正式训练」)。
输入: 三段短程 smoke(Phase A 74 步生产 config 逐字 / Phase B max_steps=8 / Phase C resume)+ eval V_where 896——**探针(短程 smoke,设计如此)**。
输出: `experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/joint/`(PREFLIGHT_JOINT.md、metrics.json、logs、config)
指标变化: overall **PASS**:5.078 s/step、峰值显存 52,973/97,871 MiB(54.1%)、loss 3.42→1.37 单调有限、eval_loss 2.598→2.170、步数等式 ceil(159215/32)=4976 与 trainer 实算一致、checkpoint 存/删/恢复闭环;2 个仅两卡才现形的缺陷修复(单测 68→75)。
结论一句话: Base SFT 正式训练的放行证据,并坐实「单卡 mock 不足以覆盖 ZeRO-3」;half/full epoch protected 步数(2488/4976)由 manifest 现算而非硬编码。
冲突/修正记录: spec 预估步数 2645/5290 未被采用(以 manifest 实算 2488/4976 为准),metrics 有显式标记;LEDGER 现无本条行,建议补入(裁决上报事项 3)。
