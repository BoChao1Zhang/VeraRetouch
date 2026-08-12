# EPR-H32 C 波在线 eval(全臂)

状态: MERGED
目标指标: 训练期监控 + checkpoint 存活选择;代理键 ONLINE_SELECTION_KEY = local_lut_de00_median(generated context,与离线主键 §12.4 同向,预注册于 D-W12 裁定);bake gate 同 §12.1。
输入: V_what 固定确定性分层子集 **256/897**(按 (build, render_mode, mask_area_bin) 分层、sha256 排序,subset_digest 计入 config_digest)——**探针(256/897)**。
输出: 各臂 `eval.jsonl`(C01 10 行 / C02 10 行 / C04 6 行)+ `eval_subset.json`(各 run 目录内,`/home/bc/data/runs/what/{C01,C02,C04}/`)
指标变化: 终点 local_lut_de00_median:C01 **9.903** / C02 **11.269** / C04(step3000)**10.832**;三臂全程 gate_pass=False;eval_seconds 实测 55.9–118.0 s/次(WT-G9 预估 ≈30 s 闭环后的权威数)。
结论一句话: 在线监控链路成立且代理键与离线主键同向,曲线单调改善但所有 C 臂全程 bake gate 未过——「训练在进步」与「gate 过不了」是两件事,后者是系统性卡点。
冲突/修正记录: 无(LEDGER「训练期监控用」定位与实读一致)。
