# EPR-H27 C01(NoWhere-FG48,What 控制臂)

状态: MERGED
目标指标: Stage-What 无 Where 输入时的下界参照(控制臂,永不进主榜);判据门(预注册,协议 §12.1/§12.4):bake_mae_mean ≤1e-4、bake_err_p99 ≤5e-4、bake/lut_non_finite = 0,主键 local_image_de00_median(generated context 主榜)。
输入: train 159,200/159,215 呈现(4,975 步 × batch 32,1 epoch,50/50 teacher/generated);在线 eval V_what 分层子集 256/897;离线 V_what:gt 897/897,generated 中断于 480/897(被 kill)——**探针(离线 generated 不完整)**。
输出: `/home/bc/data/runs/what/C01/`(what_final.pt、eval.jsonl 10 行、run_facts_final.json、steps.jsonl 至 step 4975)+ `/home/bc/data/runs/what/evaluate/C01/`(per_sample_C01_step4975_gt.jsonl 897 行;**无 evaluate_V_what.json**;generated per-sample 未落盘,仅 evaluate.log 进度记录至 480/897)
指标变化: 无基线(首批 What 臂)。在线(256 子集,generated):local_lut_de00_median 12.717(step500)→ 9.903(step4975),gate_pass 全程 False(final bake_mae_mean 1.410e-4 ✗、bake_err_p99 4.793e-3 ✗)。离线 gt 档(NOTES §14.3 从 897 行 per-sample 按 arm_metrics 口径重算,非官方 json):local_image_de00_median 2.0572、lut_de00_p90 18.7128、boundary_de00_median 3.6409、img_psnr 28.728、img_lpips 0.0412。
结论一句话: 训练端到端跑完,但 bake gate 双项未过(bake_err_p99 超阈约 10 倍,与 C02 同病、系统性),离线板不完整,C01 优于 C02 只能当排序线索,不能当结论。
冲突/修正记录: ① 离线官方汇总 json 不存在,gt 三数是重算值(rider);② 离线重算 bake_mae_mean 1.4201e-4 与在线 1.410e-4 是两个口径(897 全量 gt vs 256 子集 generated),都超 1e-4;③ 全部图像指标在偏离 D-EXEC5 下产出(composite_mask=gt、natural_mask_source=global_uniform),新 Where 冻结后须按 frozen m_pred 口径重评(DQ-1 裁定:重评即可,不必重训)。
