# 实验 PR 索引(**唯一入口**,新实验在此登记)

> **这是本仓库的唯一文档入口。** 任何 agent 接手工作先读本文档;其余一切文档经由下方「文档地图」到达,
> 不再有第二入口。规范见 `docs/EXPERIMENT_PR_SPEC_2026-08-12.md`;历史实验(规范前)已回填为下方
> **EPR-H 系**表格。
> 当前目标指标:**headline mIoU(V_where local,normal-only,面积匹配 top-k soft-IoU)**,基线 **0.7909**(CONT2)。

## 文档地图(2026-08-12 收敛后的全部到达路径)

| 文档 | 用途 | 何时读 |
|---|---|---|
| `docs/EXPERIMENT_INDEX.md`(本文档) | **唯一入口**:在途 EPR 表 + 历史 EPR-H01–H34 表 + 冲突裁决记录 K1–K16 + 本地图 | 每次接手,先读 |
| `docs/EXPERIMENT_PR_SPEC_2026-08-12.md` | **流程规范**:一实验=一 PR;PROPOSAL/RESULT/ANALYSIS 三文档的强制字段与角色分工 | 开新实验前 |
| `docs/WHERE_STATE_2026-08-11.md` | **发现文档**(非入口):当前最优配方 §一、机制结论 S1–S15 §二、死亡路线清单 §三 | 要引用「已确立的机制」或确认某路线是否已判死时 |
| `docs/PROPOSAL_geometry-injection_2026-08-11.md` | **在途 EPR-001~004 的判据出处**:PCH 共享注入模块规格、三臂 A1/B1/C 定义、go/no-go 门、§7 代码证据索引 | 执行或分析 PCH / A1 相关 EPR 时 |
| `docs/REFERENCES_2026-08-12.md` | **文献底账**:四份已归档调研报告的 VERIFIED 参考文献表合并(207 条唯一 arXiv),唯一存续副本 | 查某篇文献出处;**不得当设计依据** |
| `docs/reviews/REVIEW-impl-*.md`(5 份) | **实现审阅记录**:S0 / WhereA / WhereB / What / amort-uni 的逐项 blocker/nit,被 EPR-H 系 RETRO 引用 | 追溯某实现为何这样写、某 blocker 是否清了 |
| `experiments/prs/EPR-*/` | 每个实验的交付:在途为 PROPOSAL+RESULT+ANALYSIS,历史为单文件 `RETRO.md` | 需要实验的实读数字与 per-sample 证据时 |
| `docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` | **What 战役权威文档**:Base SFT 规格(EPR-H33/H34 的判据出处) | 做 What 侧工作时 |
| `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` | **What 战役权威文档**(带作废横幅:**架构部分已作废**,死亡路线 1) | 做 What 侧工作时,仅取未作废章节 |
| `docs/_archive/2026-08-10/` | 上一轮 local-retouch 战役的五份权威文档(归档制度,原样不动;清单见该目录 README) | 需要上一轮方法/判据/数据集代号/文献底账时 |
| `trash/` | **禁读**(红线级):已作废且形似现行方案的材料,仅供人类留档 | 永不 |

历史归档:`docs/_archive/`(2026-08-10 上一战役五份权威文档;2026-08-12 汇报快照/七月文档/论文摘要)

## 在途实验(EPR 系,新实验在此登记)

| EPR | 标题 | 目标指标与门 | 状态 | 指标变化 | 一句话结论 |
|---|---|---|---|---|---|
| EPR-001 | PCH 注入 · GT 码上界档 | mIoU ≥+0.015 晋级 / <+0.008 三臂全停 | RUNNING | — | — |
| EPR-002 | PCH 注入 · 解析码档(82%) | 同上,主对照 shuffle | RUNNING | — | — |
| EPR-003 | PCH 注入 · shuffle 负控制 | 应≈0(守卫) | RUNNING | — | — |
| EPR-004 | A1 两遍式 forced-prefix 试点门 | 500 样本试点过门 | RUNNING | — | — |
| EPR-005 | SHAPE3 补跑 eval(shape_residual 接线) | 形状残差 A<B 且 mIoU 不降 | QUEUED | — | — |
| EPR-006 | CONT3 续训(曲线未平) | mIoU vs 0.7909,收敛判定 | PROPOSED | — | — |

## 历史实验(回填,EPR-H 系,2026-08-12 裁决后落盘)

> 单文件 RETRO 在 `experiments/prs/EPR-Hxx_<slug>/RETRO.md`(历史实验不拆三文件)。编号沿用旧
> EXPERIMENT_LEDGER 顺序,Where(H01–H26)先、What(H27–H34)后。
> **本表已完整取代 `EXPERIMENT_LEDGER_2026-08-11.md`(2026-08-12 移入 `trash/docs/`,禁读)**——
> 下文各行提到的「LEDGER 勘误/补录」均因该文档退役而**关闭**,以本表实读数字为唯一口径。
> 状态词规范(裁决 K10):**REJECTED** = 未运行/无终态除名(非假设否决);**RETRACTED** = 已发表结论的撤回;负结果实验一律 MERGED、其方案入死亡清单。
> 强制口径规则(K6):normal-only Δ 一律取 `.contexts.*.headline_normal_only`,禁用顶层 pooled `.baselines`(混用少算约 0.031)。

| EPR-H | 实验 | 目标指标 | 指标变化 | 状态 | 一句话结论 |
|---|---|---|---|---|---|
| H01 | where_a(BA-0/1/2/3 basis 校准) | oracle 天花板 + 校准判据五项(成功率≥90% 等,预注册) | 判据全过;hi 中位 0.9714→0.9724(BA-3);尾部 p10 0.784→0.826+ | MERGED | 71 维 basis 表达上界够用(≈0.97),BA-3-Joint 成下游 oracle 基座;校准收益二阶、在尾部 |
| H02 | arm_W01(MC8-Joint+band) | 10 门 gate 板,主门 mIoU ≥0.75 | final 0.5565,gate 4/10,WHERE-GATE-FAILED | MERGED | 回归 71 维 latent 路线终局失败,死因 M3(中心先验支配),入死亡清单 |
| H03 | arm_W02(MC8-Joint+cband12) | 同 W01 10 门 | final 0.5617,gate 4/10,WHERE-GATE-FAILED | MERGED | 换 readout 不改死法,病根在「产生场的方式」 |
| H04 | analysis_W01_step1500 | 失败画像(NONE-PREREG 分析卡) | 长尾 22.2%;primary area_mismatch **69.7%**(0.6966 实读);过覆盖面积比中位 3.69 | MERGED | 失败主因 = 面积失配式过覆盖,集中小面积/复杂边界/离心区;LEDGER 43.1% 须勘误 |
| H05 | analysis_W02_step1500 | 同 H04 | 长尾 20.2%;primary **66.7%**(0.6667 实读);Δ hardIoU −0.0256(p=1e-4 输给先验) | MERGED | 与 W01 逐类同构,失败模式是路线级而非 readout 级;LEDGER 69.8% 须勘误 |
| H06 | amort_e1(条件数/多盆地) | M1 判别门:cond<1e2 且无近零谱(预注册) | cond 中位 8.36e5(超门 4 个数量级);CV>1 维度 100% | MERGED | w\* 不可回归:非多模态(M1 不成立)而是病态+路径依赖(M2) |
| H07 | amort_e2(Tikhonov 凸化) | 重算后 oracle IoU ≥0.95 才开 GPU(预注册) | λ 六档全 FAIL;λ₀/10 即 0.9748→0.8640;CV>1 恒 100% | MERGED | 凸化与天花板结构性绑定,M2 修复路线关闭,GPU 阶段不开 |
| H08 | amort_e3(死因终裁 M3 vs M4) | corr(输出,先验)>corr(输出,GT) ⇒ M3;antonym 为不变性负控制 | M3 坐实:0.6430/0.6077 > 0.4694/0.5251;gt 仅 +0.033 vs 先验 | MERGED | 死因 = M3(先验支配);初版 M4 结论 RETRACTED(方向读反,勘误横幅在 REPORT §0) |
| H09 | amort_e5(闭式 ridge/先验投影) | 双门 ≥0.70/≤0.40;P3 证伪线落差 >3 点 | 双门空档(0.395–0.516);ridge 投影落差 −8.9~−29.2 点触证伪 | MERGED | w 空间做什么都没用,战场只剩「产生强过中心先验的场」;−29.2 档引用须注「E5 复核未结案」 |
| H10 | amort_p1(Phi-71 前馈臂) | 硬门三项 + M3 证伪列 <0(预注册) | 0.7095(硬门全过,M3 反向);P1−P3′ = −0.0143(p=0.0006) | MERGED | Phi-71 在前馈链路是负债(S5);探针 <1 epoch,量级待全量复现 |
| H11 | amort_p3prime(P3′ 首版 1200 步) | 硬门同 P1 | 0.7417(硬门全过,M3 反向) | SUPERSEDED | 见勘误:数字被续训取代(0.7417→0.7622→**0.79095**),主榜引用指向 runs/ cont2 档;仓库内本目录 metrics 是旧值陷阱 |
| H12 | amort_P3prime_cont(2541 步) | 0.75 门(预注册) | 0.7417→0.76224(Δ+0.0254, p=1e-4),过门 | SUPERSEDED | 见勘误:基线地位被 CONT2 0.79095 取代;顶层 pooled 基线坑(K6);无 experiments/ 交付目录 |
| H13 | amort_dx(脏边机理判别) | E_HF vs κ̃ 口径判别(NONE-PREREG 诊断卡) | D_total 全负(E_HF 废弃);κ̃ 3.398/5.134 超界;SDF 罚仅 0.44% | MERGED | 脏边 = 引导上采样曲率污染;κ̃ 绝对值须带 H14 度量修正说明;DX-5 挂重跑 |
| H14 | probe_gated_upsample | 三判据:几何族 κ̃ 降、IoU 不降、语义族掉分(预注册) | κ̃ 3.398→0.080(p=1e-4);IoU Δ=−4e-6(p=0.987);语义族 bF1 −0.107 | MERGED | 门控修复零代价治好几何族曲率污染,已进配方;论文前须全量重算 κ̃ 列 |
| H15 | probe_e1_whereattn(attention 终裁) | P1–P4 四门(预注册,FWER) | P2 Δ 全负(−0.029~−0.049 反向显著);P3 全灭;唯 P4 过(+0.1309) | MERGED | attention 对「有没有指代」有反应、对「指代哪一个」没有,读出路线关闭;B3 原数字 RETRACTED |
| H16 | probe_pw5_fpresim | 词特异 Δ>0 且 p<0.05(登记) | Δ+0.0211(p=1.4e-7)但 vs 中心先验 −0.118 | MERGED | 词特异定位真实存在但绝对定位打不过零参数先验——可读「哪个词」、读不出「在哪」 |
| H17 | uni_gate0/caseA(CH-NPE) | E0a≤1%、E0b≤50、E0c≥0.85/轮廓≥0.75(证伪<0.60) | E0a 0.5697 FAIL;E0c 轮廓 0.5279 触证伪;A71p 轮廓 0.1957 | MERGED | 实验 = 有效负结果;**方案 CH-NPE REJECTED 入死亡清单**;E0b「ε 正则驯服病态」单条保留 |
| H18 | uni_gate0/caseB(FAFM) | G0a≤1%、G0b≥0.93/分家族≥0.88(预注册) | G0a 4.67e-16;G0b 0.9967(semantic 0.8949 边缘过) | MERGED | 三案唯一晋级;「自由粗场 0.895 ≫ 71 维码 0.586」为本轮最强单一发现;1.3% 探针档,排序可能随全量翻转 |
| H19 | uni_gate0/caseC(FPD) | 几何≥0.92、轮廓≥0.80(证伪<0.70)、箱宽≤0.02 | 几何 0.9484 PASS;轮廓 0.5722(宽口径;主管线 0.3567)触证伪 | MERGED | 实验 = 有效负结果;**方案 FPD REJECTED 入死亡清单**;几何段留作零参数诊断工具 |
| H20 | fafm_probe(B 案生成式探针) | 九条预注册判据 | 4 过 / 3 FAIL(②③④)/ 1 不可判(⑨)/ 1 记录性(⑧);A_fafm 0.6098 | MERGED | 生成式暂居 P3′ 下风,核心两条(指令条件性/分布刻画)FAIL;探针数字不得当全量引用;详裁挂起 |
| H21 | amort_e4(2 头 WTA) | NONE-PREREG(仅有定义) | ——(零产物) | REJECTED | 从未运行,账面除名;不构成对 2 头 WTA 想法本身的证伪(状态词由 RETRACTED 统一改 REJECTED,K10) |
| H22 | amort_P3prime_cont2 | NONE-PREREG(续训;反向更新注入臂门基线) | 0.76224→**0.79095**(Δ+0.0140, p=1e-4);step3500 在线 0.8273 仍在涨 | MERGED | **现行基线 M0 = 0.79095**;「已到天花板」被部分证伪(S15a);resume 必须用 `amort_step3500.pt` |
| H23 | P2STRUCT 系四臂 | 权重预注册消融矩阵;判据 = 对匹配基线配对 Δ | 四臂 vs 步数匹配基线 0.74172:Δ 全 null(p=0.34–0.59) | MERGED | 结构损失包无增益(null),不进配方;S14「费 IoU」系步数错配,已按 S15(b) 改判 RETRACTED |
| H24 | SHAPE3_A/B(eikonal 探针) | 预注册主判据 `shape_residual` A<B(明确不只看 IoU) | 主判据零调用;IoU 侧 A 0.7347 / B 0.7541(仅描述性) | **DISPUTED** | 判据「定义了、导出了、没接线」,层-3 结论暂缺;待 EPR-005 补跑改裁(见文末待裁定节) |
| H25 | amort_P1_pooled(池化对照) | NONE-PREREG(对照板) | 0.6222 vs P1 0.7095(Δ−0.0856, p=1e-4);corr 差 +0.123 唯一为正 | MERGED | 空间 dense 通道是配方必要件——池化掉直接损失 0.086 IoU 并推回先验支配病态 |
| H26 | ceilpush_d0 系(D0 板 + P0 探针) | D0-7 预注册档位等(preregistered_thresholds.json) | U_replay mean 0.8040/几何族 0.7596;P0 GT 配对 0.9848 vs 0.8737(p=4.8e-7);generated 配对 Δ−0.0404 **p=0.115 不显著** | MERGED | 可预测性上界 ≈0.80、GT 几何经 reasoning 段泄露 ⇒ B2 注入方向;**引用必须连带 S15(a)**;P0 generated 腿降级为方向性 |
| H27 | C01(NoWhere-FG48) | bake gate(≤1e-4/≤5e-4)+ 主键 de00(预注册) | 在线 12.717→9.903;gate 全程 False(p99 超阈 ~10 倍);离线 gt 档 2.0572(重算值) | MERGED | 训练跑完但 bake gate 未过、离线板不完整;C01 优于 C02 只是排序线索;须按 frozen m_pred 重评 |
| H28 | C02(NoWhere-SB48) | 同 C01 | 离线 generated 2.3024/20.8425/4.1707,WHAT-GATE-FAILED;context gap −0.015 | MERGED | 唯一端到端完成的 What 臂仍 gate FAILED;bake_err_p99 系统性超阈是 What 重启硬前置 |
| H29 | C03(OracleWhere-FG48) | 同 C 波(预注册) | ——(零产物) | REJECTED | 从未启动(队列缺陷 + What 停摆裁定);非对 ceiling 假设的否决 |
| H30 | C04(OracleWhere-SB48) | 同 C 波 | 3,443/4,975 步崩溃(实读);在线 13.060→10.832 | REJECTED | 崩于单样本缺 oracle fit;presence≠coverage 是唯一实质教训;修复已合入未真机运行 |
| H31 | T01–T08(What 主臂矩阵) | bake gate + 字典序主榜(预注册) | ——(零产物) | REJECTED | 从未运行,阻塞于 Where 冻结未定稿;判据未废,开跑前两项硬前置 |
| H32 | C 波在线 eval | 代理键 local_lut_de00_median(预注册 D-W12) | C01 9.903 / C02 11.269 / C04 10.832;gate 全程 False | MERGED | 在线监控链路成立、代理键与离线主键同向;「训练在进步」≠「gate 能过」 |
| H33 | S0-DATA 数据交付校验 | spec §9 项 5–9 + §2.2 隔离全 PASS(预注册) | unseen LUT 交集全 0;N_effective 159,215 | MERGED | 全战役分母权威数与 unseen LUT 隔离前提的唯一出处(已录入本表,LEDGER 补录项关闭) |
| H34 | S0-JOINT 联合 preflight | 任务卡 1a–1e 全 PASS(spec §9) | overall PASS;5.078 s/step、显存 54.1%、resume 闭环 | MERGED | Base SFT 放行证据;坐实「单卡 mock 不足以覆盖 ZeRO-3」(已录入本表,LEDGER 补录项关闭) |

## 冲突裁决记录(2026-08-12)

> 完整裁决见各 RETRO.md 的「冲突/修正记录」。此处为精简版,按裁决规则序。

**规则 1(勘误/撤回,已闭环)**
- K1:amort_e3 初版 M4「指令没进网络」RETRACTED(方向读反),勘误版 M3 MERGED(H08)。
- K2:probe_e1 B3 补件原数字 RETRACTED(0.791→0.7866 等),更正版 MERGED(H15)。
- K3:caseA E0a 0.9624 RETRACTED(探针幅度遮蔽,B9),重跑 0.5697 FAIL 为准(H17)。

**规则 2(比较口径违规)**
- K4:S14「P2 结构损失费 IoU」出自步数错配比较,RETRACTED;S15(b) 改判 null,「不采纳」理由固定为「无增益」(H23)。S8/S14 原文须挂 S15(b) rider。
- K5:S2「`<where>` 几何不可信」限定为**逐字口径**;词级 82% / position 覆盖 0.8675 版 MERGED,是 B2 注入依据。
- K6(强制口径规则):normal-only Δ 一律取 `.contexts.*.headline_normal_only`(0.48565),禁用顶层 pooled `.baselines`(0.5170),混用少算约 0.031(H12/H22)。
- K7:P0 探针「0.9848 vs 0.82」为非配对基线混接,配对文本基线 **0.8737**(Δ+0.1111,p=4.77e-7)为准;**新发现**:generated 配对 Δ−0.0404,p=0.115 不显著——S13「注入应走解析文本」的 generated 腿降级为方向性,GT 腿坚实(H26;取数 `ceilpush_d0_20260811/p0_hidden_probe/paired_tests.json`)。

**规则 3(被更强实验推翻)**
- K8:S13「已到天花板/续训关闭」被 CONT2 0.79095(+0.0140,p=1e-4)部分 SUPERSEDED——限几何族口径;U_replay 测量不撤回,引用 H26 必须连带 S15(a)。主榜继承链 0.7417(H11)→0.7622(H12)→**0.79095(H22,现行基线 M0)**。WHERE_STATE §一/§四 的 0.7622 须回改。

**规则 4(已判死方案)**
- K9:caseA/caseC 状态词冲突裁定——**实验 MERGED**(预注册证伪线触发 = 有效负结果),**方案(CH-NPE/FPD)REJECTED 入死亡清单**;重复条目去重并入 H17/H18/H19,细节已吸收(A71p 0.9314/0.1957;caseC 宽口径 0.5722 vs 主管线 0.3567)。
- K10:状态词统一——REJECTED = 未运行/无终态除名;RETRACTED = 已发表结论撤回。amort_e4 由 RETRACTED 改 REJECTED(H21)。

**附:metrics-vs-文档冲突(以 metrics 为准)**
- K12:LEDGER 43.1%/69.8% → 实读 **69.7%(0.6966)/66.7%(0.6667)**(H04/H05,`analysis_W0{1,2}_step1500/per_class_metrics.json`,已复核),LEDGER 须勘误;另 C04 步数 → 实读 3,443。
- K13:S8「SDF ≤1.4%」与 H13 的 0.44% 为不同口径(contour_warp 扰动),结论同向,引用带口径限定。
- K14:H13 κ̃ 绝对值须带 H14 两处度量缺陷说明;论文表格前全量重算 κ̃ 列。
- K15:fafm_probe 判据⑨改「不可判」(落盘 null),板记法改「4 过/3 FAIL/1 不可判/1 记录性」(H20)。
- K16:D0-3 判据数引 worst25% 的 28.7%(worst-10% 13.2% 为另一切点)(H26)。

**LEDGER 勘误/补录清单(已关闭)**:原 LEDGER 于 2026-08-12 退役移入 `trash/docs/`,本表 H01–H34 已含全部补录项(S0-DATA、S0-JOINT、ceilpush_d0、POOLED、P2STRUCT、CONT、CONT2、SHAPE3),勘误值(69.7%/66.7%,K12;C04 步数 3,443)已写入对应行——**不再需要回改任何外部文档**。
**WHERE_STATE 回改清单**:§一/§四 0.7622→0.7909(S15a);S8/S14 挂 S15(b) rider(K4);S2 加逐字口径限定(K5);S13 P0 generated 腿降级(K7)。

## 待主 agent 裁定(DISPUTED)

| EPR-H | 实验 | 争议 | 依赖 |
|---|---|---|---|
| H24 | SHAPE3_A/B | 预注册主判据 `shape_residual` 零调用(WEVAL-1 型「没接线」第三次再现),两臂被误用 IoU 判读;层-3 形状裁决 RETRACTED,IoU 侧仅可描述性引用;`amort_SHAPE3_{A,B}_evalfix_20260812/` 为空目录、补跑未落盘;SHAPE3_B `metrics.json::arm` 误记 `"P3prime"` | 待 EPR-005_shape3-evalfix 补跑后改裁;补跑前层-3 结论位置留空 |
