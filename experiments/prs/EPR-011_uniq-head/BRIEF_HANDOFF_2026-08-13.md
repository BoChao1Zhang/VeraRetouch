# EPR-011 交接简报(2026-08-13 17:00,前任主 agent 上下文耗尽前落盘)

> 接手路径:INDEX → 本文件 → PROPOSAL(判据,逐波冻结)/RESULT(数字,只报不解读)。
> 可用线 = **0.85**(用户口径);现役最优 = **M0 0.79095**(P3'+续训@3500,headline 口径);
> 本 EPR 十九臂没有任何配置超过 M0。缺口 ≈ 0.06,且见下文——这不是调参能填的缺口。

## 一、实验简报(三天,~19 臂,全部预注册后出数)

**冻结档(VLM 不动)**:统一 query 头(K 假设 + WTA + 选择头)与 P3' 在 1200 步打平
(全臂 0.73-0.75);一切正因子——K=8(+0.0082)、解析码读出(+0.0052)、像素注意力(+0.0039)
——**两两组合全部为负或零(交互 −0.006~−0.011,三个独立样本)**:它们在竞争同一份信息。
坐标基/presence/带宽/attention-map 读出全部无增益;M3 中心先验病零复发。

**缩放档**:统一头**不随训练缩放**——K8@3500 = 0.7571 vs M0 0.79095(配对 −0.0362, p=1e-4,
判负)。但其 **best-of-8 = 0.8072 > M0**,选择头只有 12.8% 命中率——**5 分落在选择机制上,
这是全战役最大的已测余量**。

**微调档(LoRA)**:增益**专属**「in-context token 写入 × LoRA」组合:
ST(多 special token+LoRA)= 0.7739/0.7659(双种子,vs 1200 步锚点 +0.017/+0.018,均 p<0.01,
**唯一晋级且复现的配置**);同 LoRA 配桥式读出 = **负**(−0.0054, p=0.0015);配 P3' 头 = **零**
(+0.0002)。机制结论:**必须给模型写入通道,微调才兑现**(LISA 前提的因子级证明)。
2×2 交互 +0.0225。

**工程判决**:视觉塔 LoRA 与冻结 sim-norm 契约结构性不相容(三次域断言处决,死亡速率 =
头对 sim 通道依赖度:P3' step130 / 桥 step400 / in-context 满 1200 但续训 step~1720 死)。
语言侧-only 是现契约下唯一合法 LoRA 形态,已验证零漂移。

**在跑(接手时)**:ST_LANG2(干净 2×2 收口,~17:45)、ST_LANG_R32(rank 旋钮,~19:45,gpu0)、
ST_LANG_K16(token 数旋钮,~19:45,gpu1)。等待器已挂,出板收割进 RESULT。

## 二、瓶颈分析(为什么 0.85 不是调参可达)

1. **信息可达性(K17 终裁后重写,2026-08-13 晚)**:**U_replay 已整体撤回(K17)**——
   几何参数系 subject mask 确定性导出(`dataset_build/src/construct/subject_geom.py`,
   `_mask_pca` 质心/主轴/角度 + 面积自适应 margin),D0-7 重放把确定性量当自由量重抽,
   测出的「歧义」是协议自造的;U_BAYES 提案同机件作废。**当前没有任何有效的天花板测量,
   且方向转为乐观:几何族 (image,instruction)→mask 基本是确定性映射 ⇒ 0.85 没有已测硬界阻挡。**
   接替首要行动改为两件:(a) **确定性验证**(同 subject mask 重跑导出应逐位复现 .cgt,CPU 级);
   (b) **ORACLE-SUBJECT 上界**(喂 GT subject mask 测几何导出上限,把「找主体」与「导几何」
   的缺口拆开——m_sem 0.820 vs oracle 的差距会直接指出主攻方向)。
   连带解读修正:「GT 是分布抽样」对几何族失效,WTA/best-of-K 的收益解释由「歧义建模」
   软化为「集成/优化效应」(数字全部有效,解读改);reasoning 注入线(P0 GT 腿 +0.1111
   p=4.8e-7)与数据升级(L8)仍是候选路,但不再有「必要条件」的地位。
2. **选择瓶颈(最大可测余量,~5 分)**:多假设场已集体持有 0.807,选择头是随机水平。
   未试:更强选择监督权重、两阶段 rerank、场级置信估计。教训:FQ(按族指派)判负——
   假设的多样性不按 family 组织,别再走这条。
3. **缩放瓶颈**:ST 线 1200 步斜率全场最陡(quick-eval 尾段 +0.025/500 步)但 3500 步
   续训从未成功测到(视觉 LoRA 被漂移锁死)。**ST_LANG_CONT@3500 对 M0 是下一个决战,
   唯一无漂移路线**,等今晚三板选出最优旋钮(r16/32 × K8/16)后立项。
4. **契约瓶颈(工程,可解但有代价)**:sim-norm 冻结契约禁视觉微调。解锁三选一:
   norm 随训重拟(破坏跨臂可比,需预注册协议)/sim 通道走基模型双前向(算力 ×1.5)/
   直接消融 sim 通道价值(一臂可测,若其价值已被 LoRA 吸收则契约整个退役)。
5. **口径风险(审稿人视角,勿忘)**:headline 是 GT 面积 oracle top-k(外部不可比,系统性
   偏高);0.85 若是产品可用线,**先确认口径**——固定阈值列至今未实现(审稿意见 W1),
   真实可用缺口可能大于 0.06。

## 三、下一个 agent 的建议行动序(按信息价值/成本排序)

1. 收今晚三板(等待器在;判据在 PROPOSAL wave-8);若 R32/K16 有增益,带最优旋钮立项
   **ST_LANG_CONT@3500 vs M0**(resume 链已验证,epochs=2.0 防钳制,域列必报)。
2. **选择头攻坚**(最大余量):--uniq-sel-weight 扫描是零代码起点;rerank 需新文件。
3. **固定阈值列**(半天,eval-only 重扫已有 per_sample 即可)——0.85 讨论的前提。
4. reasoning 注入与 PCH 战役(EPR-009/010,原编排会话失联)协同——B2 线的 token 级
   注入接口本 EPR 已建好(UniQ 系),缺的是把 reasoning hidden 接进来当第三通道。
5. 运维契约(全部已制度化,踩过的坑别再踩):新文件+seam 模式(冻结源码零改动,
   uniq.py→uniq4b.py 谱系);显存准入门已内建 q(--mem-peak,65 准入/80 帽);
   域断言是朋友不是障碍;cancel 释放下一任务上卡;sed 克隆 wrapper 会咬人,
   干跑 --help 测不到工厂路径;gpu1 的 L8/组暂停归 databuild 会话管(SendMessage 通道已断,
   协调走 agent-gpu-queue/NOTES.md)。

## 四、关键文件地图

- 判据:`PROPOSAL.md`(逐波冻结,含 AMD 与 wave-4~8 追加);数字:`RESULT.md`;
- 代码:`q3vl/whereb/amort/uniq{,2,3,3b,3c,4,4b}.py` + `scripts/run_uniq*_arm.py` +
  `waves/uniq*_arm.sh`(全部新文件,基础设施零改动;测试 `tests/test_uniq*.py` 全绿);
- 运行:`/home/bc/data/runs/where_b/amort_UNIQ*_2026081{2,3}/`(board=eval_final/metrics.json);
- 锚点 per-sample:`EPR-005_shape3-evalfix/per_sample_amort_P3prime1200_evalfix_20260812.jsonl`;
  M0 per-sample:`runs/where_b/amort_P3prime_cont2_20260811/eval_final/per_sample.jsonl`。
