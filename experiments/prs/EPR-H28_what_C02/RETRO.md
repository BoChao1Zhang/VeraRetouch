# EPR-H28 C02(NoWhere-SB48,What 控制臂)

状态: MERGED
目标指标: 同 C01(SB48 生成器变体);判据门同上(预注册)。
输入: train 159,200/159,215 呈现(4,975 步,1 epoch);离线 V_what **gt 897/897 + generated 897/897**——**全量**(唯一端到端跑完的 What 臂)。
输出: `/home/bc/data/runs/what/C02/`(what_final.pt、eval.jsonl 10 行、run_facts_final.json)+ `/home/bc/data/runs/what/evaluate/C02/`(evaluate_V_what.json、candidates_V_what.jsonl 2 行、双 context per-sample 各 897 行)
指标变化: 离线(evaluate_V_what.json,generated 主榜):local_image_de00_median **2.3024**、lut_de00_p90 **20.8425**、boundary_de00_median **4.1707**,gate_pass **false**(bake_err_p99 4.969e-3 ✗ vs 5e-4;bake_mae_mean 9.472e-5 ✓);gt 档 2.3174/21.1401/4.2073;context gap(generated−gt)= **−0.0150**;main_board `selection_possible=false`,tag WHAT-GATE-FAILED。在线 local_lut_de00_median 12.604→11.269。
结论一句话: 唯一端到端(训练+全量双 context 离线板)完成的 What 臂,仍 gate FAILED,卡点是系统性的 bake_err_p99(两臂同超约 10 倍,**What 重启前须独立诊断,硬前置**);读自己生成的 `<color>` 不比读 GT 差(gap −0.015)。
冲突/修正记录: LEDGER 引 2.302/20.84/4.17 与 metrics 一致;同受 D-EXEC5 偏离约束(见 EPR-H27 ③)。
