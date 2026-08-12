# EPR-H11 amort_p3prime(P3′ 首版,1200 步)

状态: SUPERSEDED(配方 MERGED 为主榜;本档数字被续训取代,继承链终点 = EPR-H22 的 0.79095)
目标指标: 不经 Phi-71 的自由粗场直出配方(F_pre ⊕ 相似度 dense ⊕ FiLM → conv 塔)首验。硬门同 P1(面积比 / 换主体 Δ / antonym);M3 证伪列 <0。
输入: 同 P1 训练池(42,752 normal-only,**1,200 步 ≈0.90 epoch,探针**);eval V_where local 400/400(per_sample 2,400 行 = 400×6 指令模式)。
输出: `/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/where_b/amort_p3prime_20260810/`(BOARD.md、metrics.json、analysis_longtail/、viz_random16/)
指标变化: top-k IoU 中位(normal-only)**0.7417**(pooled 0.7489;holdout 半区 0.7281,硬门全过);vs 中心先验 Δ +0.2624(p=1e-4);corr(先验)−corr(GT) = −0.1692;m_sem 语义头 0.8197(bF1 0.9597);硬门全 PASS。
结论一句话: P3′ 配方一次性越过 W01 全部病灶(M3 反向、硬门全过、小目标带不塌),成为主榜配方;**0.7417 已被同配方续训取代**(CONT 0.7622 → CONT2 0.79095,S15a),引用主榜必须指向 runs/ 路径的 cont2 档而非本目录。
冲突/修正记录: 本档无勘误;取数陷阱(LEDGER 交付缺口 1):仓库内本目录 metrics.json 仍是 1,200 步的 0.7417,主榜数字只存在于 `/home/bc/data/runs/where_b/amort_P3prime_cont2_20260811/eval_final/metrics.json`(现行基线 0.79095),读者拿本目录数字会拿到旧值。
