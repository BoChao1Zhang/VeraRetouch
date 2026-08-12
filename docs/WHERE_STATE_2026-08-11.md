# Where 战役 · 发现文档(2026-08-11;2026-08-12 降级为非入口)

> **本文档是 where 侧的「发现文档」——机制结论(S 条)与死亡路线的权威落点,不是入口。**
> **唯一入口 = `docs/EXPERIMENT_INDEX.md`**(实验总览 + 文档地图),任何 agent 接手 where 相关
> 工作从 INDEX 进,再由它链到本文档。
> 分工:**实验发生了什么** → INDEX(EPR 表 + EPR-H 系 RETRO);**由此确立了什么机制** → 本文档 §二 S 条。
> 冲突时:S 条口径以本文档为准,实验编号/状态/裁决记录以 INDEX 为准。
> `trash/` 目录下的内容一律**禁止阅读**(误导性/已作废材料,只为人类留档)。

---

## 一、当前最优配方(P3' + 门控,全部判据背书)

**架构**:F_pre(H/16×1024,冻结)⊕ 软化相似度场 dense 通道 ⊕ FiLM(pooled `<where>` hidden)
→ 轻 conv 塔(~3.8M 参数)直出粗场 ŷ(H/16)→ **家族门控上采样**(semantic 族保留图像引导、
几何族低通;路由 = 类型词规则,实测 400/400)→ m_pred。副头:语义头 m_sem(~1-3M,subject 类
`.cgt` 直接监督)。VLM 全冻结,无 Phi-71。训练:local 75,544 排 low 后 42,752;loss 五项
(BCE_soft 主 + SDF 边界 + 面积带 + 假指令 + 换主体分离[仅异主体对,31.4% 覆盖]);
checkpoint 三硬门(面积比/换主体 Δ/antonym 不变性)。

**成绩(V_where,normal-only,面积匹配 top-k soft-IoU)**:

| 指标 | 值 | 参照 |
|---|---|---|
| **主榜中位(CONT2,step3500 在线 0.8273 仍在涨)** | **0.79095 — 现行基线 M0**(继承链 0.7417→0.7622→0.79095,K8) | 中心先验 0.4853 / W01 0.4874 / 地板 0.2254;目标 0.75 已达 |
| 配对 Δ vs 1200 步基线 | +0.0254 (p=1e-4);M3 证伪加强到 −0.1974 | underfit 假设(S9)证实,仍有余量 |
| 配对 Δ vs 中心先验 | **+0.2624 (p=1e-4)** | W01 仅 +0.0017 |
| corr(先验)−corr(GT) | **−0.169**(M3 治愈) | W01 为正(病态) |
| holdout 半区 | 0.7281,硬门全过 | 非 selection 幻觉 |
| 小目标带 area<0.15 | **0.7493** | 先验 0.131(历史全方法失败带) |
| 语义头 m_sem | 0.820(bF1 0.96) | 相似度场 0.324 |
| 门控后几何族曲率 κ̃ | **0.08**(修复前 3.40) | IoU 零代价(p=0.99) |

## 二、有效机制结论(编号引用,出处在括号)

- **S1** attention 通路无指令条件定位:`<where>`/instr/special 三池全踩线,raw 与差分臂同判;
  唯一通过判据是「物体性闸门」P4(probe_e1_whereattn + REVIEW-result,含 E1b 补件)。
- **S2** F_pre × 名词相似度有真实词特异定位(找物体),但 GT 是编辑衰减区≠物体掩膜
  (probe_pw5)。`<where>` 生成文本:名词可信、空间几何不可信(local 逐字 0/30);
  「token-F1 p50=1.0」是 global 常量句(34/34)的算术产物,**不是**定位证据(s0 分层复算)。
- **S3** w\*(Phi-71 oracle 系数)不可回归:有效条件数中位 8.4e5、多盆地(86% 单一重启命中)、
  逐维 CV 全 >1(amort_e1);Tikhonov 凸化与天花板结构性绑定,全 λ 不过 0.95 验收
  (amort_e2);闭式 ridge 投影落差 9–29 点(amort_e5)。**参数回归/凸投影路线永久关闭。**
- **S4** W01/W02 死因终裁 = M3(输出被中心先验支配,corr 0.64;normal-only 下仅 +0.0017);
  M4「无条件化」已勘误撤回——antonym 是不变性控制且通过(amort_e3 勘误版)。
- **S5** Phi-71 在前馈链路是负债:配对步数匹配 A/B −0.0143(p=6e-4);Gate0 上界同向
  (自由粗场 0.895 ≫ 71 维码 0.586,semantic 族);What 侧 (w\*,ρ\*) 接口维护成本已被测量。
- **S6** 三套「统一框架」方案 Gate 裁决:B(FAFM)两门过、探针 4/8 判据(生成式暂居 P3' 下风,
  详裁待完成);A(CH-NPE)与 C(FPD)触发预注册证伪线作废(uni_gate0 + fafm_probe)。
- **S7** 脏边机理 = 引导上采样的**曲率污染**(κ̃ 口径 δ_b=97.6%),非高频能量(D_total 为负,
  E_HF 口径废弃);家族门控修复零成本已采纳;semantic 族引导价值 bF1 +0.107,保留
  (amort_dx + probe_gated_upsample)。
- **S8** 粗场独立脏(AFR 2–3× 地板);DX-4 审计:SDF 边界罚 ≤1.4%、面积带罚 ≈0% 贡献
  ——**两项 loss 当前权重下惰性**。
  **【2026-08-11 更新·定案】P2 结构损失包实测不进配方**:NaN 数值 bug 修复后三臂全部复活
  (`nonfinite_tensors=0`、硬门全过),但**全部低于 0.7622 基线**——
  curv/mono 0.05 **0.7522**(−0.0100)、shaped 5× **0.7486**(−0.0136)、
  shaped 10× **0.7477**(−0.0145)。**三个独立数据点同向 ⇒ 结构损失费 IoU,不采纳。**
  「推 IoU 的主选项是 P2」这句**作废**;当前唯一被证实推动 IoU 的是**续训**
  (0.7417→0.7622,p=1e-4)。惰性 loss 项的重标定已随 P2 一并否决(shaped 5×/10× 即该实验)。
- **S9** 长尾 95% 可救(underfit 主桶 + coverage_bias);链路天花板 0.9875 / basis 0.9890
  只解释 5%(analysis_longtail)。
- **S10** sink 排除规则定稿:合取规则(9.90% 格),跨分辨率稳定;图像块首 token 400/400 皆 sink
  (pw1_sink_survey,在 probe_e1 交付内)。
- **S11** 数据事实:`.cgt` 由 `raster_geometry` 参数化渲染,mask_type 分布 linear 30.8 /
  radial 26.2 / band 25.5 / **semantic 17.5**(真轮廓,非椭圆近似);global GT 是字面常量句;
  `winner_confidence=low` 占 V_where 44%,**评测 headline 一律 normal-only**(U7 裁定)。
- **S12** 运维教训(全部已制度化):pueue fd 限 1024、CUDA_VISIBLE_DEVICES 重映射、sqlite
  跨线程、进程启动后禁改源码(sha256 冻结)、cancel 必须绑 backfill、禁 shell 循环提交队列。
- **S13(2026-08-11 晚,战略合成)** 可预测性上界已实测:U_replay=0.8040 全体 / **0.7596 几何族**
  (D0-7 生成器重放 k=8,发布 GT 即重放分布一次抽样 corr 0.887)——**模型 0.7622 已打到
  (image, instruction) 的信息论天花板**。但 GT 几何按设计泄露在 `<where>` reasoning 段
  (79% 样本点名形状、词级 82% 正确——「逐字 0/30」是苛刻口径造成的误读),而 pooled FiLM
  把它扔掉(teacher-forced 完美 reasoning 只动 +0.0007~0.0087)。**唯一超越天花板的路 =
  `<where>` 几何短语的解析/token 级注入(B2)**;加容量、加训练、改指令三条路全部关闭。
  Caveat:U_replay 是保守上界(点估计优于抽样);band 族地板最低(0.69/P10 0.455)。
- **S14** 结构损失包定案:修 NaN 后三臂 0.7477-0.7522,全部低于 0.7622 基线——**费 IoU,
  不进配方**(与 S13 自洽:逼形状规整=偏离后验均值);形状规整性另由 SHAPE3 架构级探针裁决。
- **S15(2026-08-12,SCOREBOARD 修正,零上下文接力 agent 复核)** 三条修正:
  (a) **S13 部分证伪**:CONT2 续训到 0.79095(+0.0140,p=1e-4)且曲线仍在爬——headline 已越过
  几何族 U_replay 0.7596 朝全体 0.8040 走;「已到天花板」限几何族口径,续训收益未尽,
  **M1 门基线更新为 0.7909**;
  (b) **S14 措辞修正**:P2「费 IoU」出自步数错配比较(违反 U4 规则);对匹配基线四臂统计学
  null(p=0.34-0.59)——不采纳的决定不变,理由改为「无增益」;
  (c) **SHAPE3 不可裁**:预注册主判据 shape_residual 定义了、导出了、**零调用**(WEVAL-1 型
  「没接线」再现),两臂被误用 IoU 判读——待补跑 eval 后方可裁,层 3 结论暂缺。
  另:B2 加载修复经 bit-exactness 测试;broadcast 形式经证不可能 bit-identical(浮点归约序),
  独立支持「跳过 broadcast 直上 PCH」;PCH 三档(GT 码/解析/shuffle)frozen-base 在跑。

## 三、死亡路线清单(禁止复活,除非新证据推翻对应 S 条)

| 路线 | 死因 | 证据 |
|---|---|---|
| MetaCanvas query→71 维回归(W01/W02) | M3 + S3 | 用户终裁 + amort 诊断 |
| attention 差分读出(方案 B / PR-ATT1) | S1,4 组合全踩线 | probe_e1 交付 + E1b 补件 + REVIEW-result |
| 固定短语共模差分 / M1 序列均值减法 | S1 + RO9c(上一战役) | E1b 补件 |
| 参数回归、Tikhonov 凸化、闭式 ridge 投影 | S3 | E1/E2/E5 |
| CH-NPE(A 案)、FPD(C 案) | 轮廓表示证伪 | uni_gate0 |
| 纯几何原语头(砍语义表达) | S11(semantic 是真轮廓) | 用户否决 + mask_type 普查 |
| AUC 判据、E_HF 脏边口径 | 红线 / S7 | CLAUDE.md + amort_dx |

## 四、在途(2026-08-11 14:30)

- P3' 续训已结算:**0.7622,0.75 门已过**(墙钟截断于 2541/4000 步,续训仍有收益空间)。
- 双卡:P2STRUCT_A/B(结构损失探针)+ P2STRUCT_W/W2(惰性 loss 重标定,权重预注册)在跑/排队。
- 待裁读:FAFM 4/8 详情;POOLED 池化对照板(M5 终裁);DX-5 重跑(need_mask bug)。
- 议题:三案后备与「低维结构化潜码是否续投」议题已消化进 `docs/EXPERIMENT_INDEX.md`
  (EPR-H17/H18/H19 结论 + 死亡清单)与 `docs/PROPOSAL_geometry-injection_2026-08-11.md`
  (选型落点);另有审阅遗留(E5 复核、P4 统计修正)。

## 五、文档与实验状态表(清理依据)

> **2026-08-12 收敛后本节不再是文档地图。** 现行文档地图在 `docs/EXPERIMENT_INDEX.md` 顶部
> 「文档地图」小节(INDEX 之下的全部到达路径);本节只保留清理的**历史勘误记录**。

**2026-08-12 收敛移动(七份 → `trash/docs/`,禁读)**:四份调研报告
(RESEARCH_unified-field-prediction / _analytic-edge-quality / _ceiling-push / _geometry-extraction-arch)
——结论已被 EPR-H 系 RETRO 与 `PROPOSAL_geometry-injection_2026-08-11.md` 吸收,
其 VERIFIED 参考文献表抽出合并为 **`docs/REFERENCES_2026-08-12.md`**(唯一存续副本);
CAMPAIGN_PROGRESS_BRIEF_2026-08-10(过时快照);plan/PROPOSAL_REVISION_AGENDA_2026-08-11
(议题已消化进 INDEX/EPR);EXPERIMENT_LEDGER_2026-08-11(已被 INDEX 的 EPR-H01–H34 表完整取代)。
另:**RESEARCH_amortized-oracle-fitting_2026-08-10.md 早已不在仓库**(与下段三份同批,用户手动
清理),其结论见 §二 S3/S5 与 INDEX EPR-H06–H12。

**文件状态勘误(2026-08-11 清理时发现)**:WHERE_HEAD_REDESIGN_PROPOSAL、_DELTA 与
WHERE_HEAD_REQUIREMENTS_2026-08-10.md 三份文档已不在仓库——**用户手动清理**(前两份从未
commit;REQUIREMENTS 曾 commit,清理 agent 一度按旧 ACTIVE 表恢复,经用户追认后已重新删除)。
**三份均已从上方 ACTIVE 表移除**;其有效结论已收拢进本文档 §二各 S 条(需求侧 → S2/S11,
方案 B 终裁 → S1/S4),**不再依赖这三份文档**;方案 B 的终裁证据以 probe_e1 交付
(REPORT+E1b+REVIEW-result)为准。清理复核结论:where_b
下四项候选(PREFLIGHT/mock_closed_loop/micro_batch_probe/genctx_jobs)均有勘误横幅或被
ACTIVE 文档按行号引用,**全部保留,零移动**;METACANVAS_..._PROTOCOL_2026-08-04.md(死亡
路线 1 的架构规格)加作废横幅处理。QUEUE_USAGE.md 系用户手动删除(内容已进入 gpu-queue
skill),追认;REVIEW-impl-amort-uni.md 待下次 commit 纳入(用户裁定)。

**实验总账**:哪些实验做了、实跑多少数据(实读数,非 REPORT 宣称)、结论多硬,见
`docs/EXPERIMENT_INDEX.md` 的 **EPR-H01–H34 表**(2026-08-12 起取代 EXPERIMENT_LEDGER,
逐行给出目标指标 / 实测指标变化 / 状态 / 一句话结论,并附「冲突裁决记录」K1–K16)。
**探针风险仍适用**:建立在探针(非全量)上的 S 条为 **S6 / S5 / S9**,连同 §一主榜的步数截断
——引用这几条时须一并声明该可信性档位;逐实验的样本数实读见对应
`experiments/prs/EPR-Hxx_*/RETRO.md`。

**trash 纪律**:trash/ 根目录 README.md 声明「本目录内容已作废且可能误导,任何 agent 禁止
读取;人类留档用」。CLAUDE.md 载有该禁令与入口指针(入口 = `docs/EXPERIMENT_INDEX.md`)。
