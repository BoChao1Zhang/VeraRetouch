# 实验总账表(2026-08-11)

> **本表是「哪些实验做了、用了多少数据、结论多硬」的唯一账本。** 入口文档为
> `docs/WHERE_STATE_2026-08-11.md`(§二 S 条为结论出处),本表与之冲突时以入口文档的**结论**为准、
> 以本表的**数字**为准(本表数字一律从各实验的 `metrics.json` / `config/run_setup.json` /
> `per_sample.jsonl` 行数实读,**不抄 REPORT 宣称值**)。

## 〇、全量口径(分母)

权威 split 表来自 `/mnt/nfs/bc/data/datasets/sft2seg-20260804/splits/*.index.jsonl`
(实数记录见 `experiments/.../where_a/NOTES.md` 第 21 行 V8):

| split | 总量 | local | 其中 normal | 其中 low |
|---|---|---|---|---|
| train | 159,215 | 75,544 | 42,752 | 32,792 |
| **V_where** | **896** | **400** | 224 | 176 |
| V_what | 897 | 408 | — | — |
| T_final | 918 | 424 | — | — |
| T_lut_unseen | 433 | 198 | — | — |

**关键口径说明(避免把全量误读成子集)**:V_where 的 896 条里 496 条是 **global**(常量句、
GT 恒为全 1),空间判据对其无定义,故 where 侧空间实验的**全量universe = local 400**。
本表中 `n=400` 一律是**全量**,不是抽样。headline 取 **normal-only(224)** 是 U7 裁定的口径选择
(见 S11),不按子集计。

## 一、可信性标记规则(用户裁定)

| 标记 | 含义 |
|---|---|
| **全量** | 训练与评测都跑在其声称的全量数据集上 |
| **探针(n=X/全量 Y)** | 抽样 / 子集 / 截断。结论**在机制判别与方向上有效**,但**量级待全量复现**——不是否定 |

## 二、Where 侧

| 实验 | 输入(数据集 + 实际 n) | 输出 | 结论 | 可信性 |
|---|---|---|---|---|
| `where_a`(BA-0/1/2/3 校准) | train **75,544/75,544**;V_where local **400/400**(每臂 `n_ok=400, n_rejected=0`) | F_pre 提取管线 + oracle latent + 四臂校准表 | BA-3-Joint 胜出,成为下游 oracle 基座;F_pre 对 SFT 不变 | **全量** |
| `arm_W01.json` | train **159,215/159,215**;eval V_where **896/896** | W01 终局逐样本结果 | MC8-Joint+Band 失败,mIoU 0.557、gate 全 FAILED | **全量** |
| `arm_W02.json` | 同上 | W02 终局逐样本结果 | 同上,0.562;与 W01 同构失败 | **全量** |
| `analysis_W01_step1500` | V_where local **400/400** | 分类失败画像 + 长尾样本表 | area_mismatch 为主因(43.1%);失败画像支撑 S4 | **全量** |
| `analysis_W02_step1500` | V_where local **400/400** | 同上 | 同上(69.8%) | **全量** |
| `amort_e1`(条件数/多盆地) | V_where local **400/400**,零训练 | 病态诊断 metrics | w\* 有效条件数中位 8.4e5、86% 单一重启命中、逐维 CV 全 >1 ⇒ **不可回归** | **全量** |
| `amort_e2`(Tikhonov) | V_where local **400/400**(`n=400`) | λ 扫描 stageA 表 | 凸化与天花板结构性绑定,全 λ 不过 0.95 验收 | **全量** |
| `amort_e3`(W01/W02 死因) | V_where local **400/400**(逐臂) | 四问诊断 + antonym 勘误 | M3(输出被中心先验支配)坐实;**M4「无条件化」勘误撤回** | **全量** |
| `amort_e5`(闭式 ridge) | V_where local **400/400**(`per_sample` 400 行) | 投影落差表 | 闭式 ridge 投影落差 9–29 点,路线关闭 | **全量** |
| `amort_p1`(Phi-71 前馈) | train 池 42,752;**1,200 步 × batch 32 = 38,400 次呈现 ≈ 0.90 epoch**;eval **400/400** | P1 臂 checkpoint + 逐样本板 | 71 维码前馈链路可用但劣于自由粗场 | **探针(训练 38,400/42,752 次呈现,<1 epoch)** |
| `amort_p3prime`(1200 步) | 同上池;**1,200 步 ≈ 0.90 epoch**;eval **400/400**(2,400 行 = 400×6 指令模式) | P3' 首版 checkpoint + 板 | normal-only 中位 **0.7417** | **探针(训练 <1 epoch)** |
| **`amort_P3prime_cont`**(主榜) | 同池 **42,752/42,752**(family 直方图精确求和);**2,541/4,000 步**(4h 墙钟截断,≈1.90 epoch);eval **400/400** | 主榜 checkpoint + `eval_final/`;**⚠ 无 experiments/ 交付目录** | **normal-only 0.7622,0.75 门已过**;三硬门全过 | **全量(数据 42,752/42,752;步数截断 2,541/4,000,曲线未平)** |
| `amort_dx`(脏边机理) | V_where local **400/400**(`n=400`) | κ̃ / D_total 口径对照 | 脏边 = 引导上采样的**曲率污染**,非高频能量;E_HF 口径废弃 | **全量**(子探针 DX-5 报错 0 可用样本,需重跑) |
| `probe_gated_upsample` | V_where local **400/400** | 家族门控上采样对照 | 门控修复 κ̃ 3.40→0.08,IoU 零代价;semantic 族引导 bF1 +0.107 | **全量** |
| `probe_e1_whereattn` | V_where **896**→拟合 193 / OOF 189 / 排除 18(无 shuffle 配对)= local **400** | attention 读出四池 + sink 普查 | **attention 通路无指令条件定位**;唯一过判据是物体性闸门 P4 | **全量** |
| `probe_pw5_fpresim` | V_where local **400/400** | F_pre×名词相似度场 | 有真实词特异定位,但 GT 是编辑衰减区 ≠ 物体掩膜 | **全量** |
| `uni_gate0/caseA`(CH-NPE) | V_where local 400;**判据段 E0a 用 256** | A 案 Gate0 metrics | `verdict: fail` —— 轮廓表示证伪,A 案作废 | **探针(E0a 256/400)** |
| `uni_gate0/caseB`(FAFM) | train local **1,000/75,544** | B 案 Gate0 metrics | `verdict: pass` —— 两门过,进 fafm_probe | **探针(1,000/75,544 = 1.3%)** |
| `uni_gate0/caseC`(FPD) | V_where local **397/400**(3 条丢弃) | C 案 Gate0 metrics | `verdict: fail` —— C 案作废 | **探针(397/400,实质接近全量)** |
| `fafm_probe`(B 案详探) | 训练 **20,000/≈70,000**;eval **400/400** | FAFM 探针板 | 4/8 判据过,生成式路线暂居 P3' 下风,详裁待完成 | **探针(训练 20,000/≈70,000)** |
| ~~`amort_e4`~~(2 头 WTA) | —— | **零产物**(空目录) | 从未运行;已移入 `trash/experiments/` | **未运行** |

## 三、What 侧

Where-B 冻结 checkpoint 未定稿,**T01–T08 八臂全部只有定义、无运行**;已跑的只有 C 波四臂。

| 实验 | 输入(数据集 + 实际 n) | 输出 | 结论 | 可信性 |
|---|---|---|---|---|
| `C01` | train **159,200/159,215**(1 epoch,4,975 步 × batch 32);在线 eval 子集 256;离线 V_what:**gt 897/897 完成、generated 仅 480/897 被杀** | checkpoint + gt 逐样本板 | 训练跑完但 **gate FAILED**;离线板**不完整**,无 `evaluate_V_what.json` | **探针(离线 generated 480/897)** |
| `C02` | train **159,200/159,215**;离线 V_what **gt 897/897 + generated 897/897** | checkpoint + `evaluate_V_what.json` + 候选表 | **唯一端到端跑完的 What 臂**;仍 **gate FAILED**(`local_image_de00_median` 2.302、`lut_de00_p90` 20.84、`boundary_de00_median` 4.17) | **全量** |
| `C03` | —— | **零产物**(`runs/what/C03` 与 `evaluate/C03` 均空) | 从未启动 | **未运行** |
| `C04` | train 截断于 **~3,000–3,440/4,975 步**(单样本 KeyError 崩溃) | 仅中途 checkpoint,无 `what_final.pt` | 崩溃,无可用终态结论 | **探针(训练截断,无终态)** |
| `T01`–`T08` | —— | **零产物** | 仅定义;阻塞于 Where-B 冻结 checkpoint | **未运行** |
| C 波在线 eval(全臂) | V_what 分层子集 **256/897** | `eval.jsonl` 逐步曲线 | 训练期监控用 | **探针(256/897)** |

## 四、建立在探针上的 S 条清单(风险清单)

以下 S 条的**主要证据来自探针档**。它们的**方向**可信,**量级**待全量复现;推翻它们不需要新机制,
只需要同一实验的全量版给出相反量级。

| S 条 | 内容 | 依赖的探针 | 风险 |
|---|---|---|---|
| **S6** | 三套统一框架 Gate 裁决:B 两门过、A/C 作废 | `uni_gate0/caseB` **1,000/75,544(1.3%)**;`caseA` E0a **256/400**;`fafm_probe` 训练 **20,000/≈70,000** | **最高**。A/C 的「作废」与 B 的「晋级」都建立在 ≤1.3% 训练数据或 256 样本判据段上,三案排序可能随全量翻转。**该风险是预注册的**:主 agent 2026-08-11 裁定 3 已写明「批准探针级 20k……**探针数字不得当全量臂结果引用**」,并落进 `metrics.json::train.scale_deviation`(见 `fafm_probe_20260811/NOTES.md:50`) |
| **S5** | Phi-71 在前馈链路是负债(配对 A/B −0.0143) | `amort_p1` 与 `amort_p3prime` **均 <1 epoch(38,400/42,752)** | **中**。两臂步数匹配,配对差分内部一致;但两者都未收敛,负债量级可能随收敛变化。Gate0 上界证据(自由粗场 0.895 ≫ 71 维码 0.586)为独立同向支撑 |
| **S9** | 长尾 95% 可救(underfit 主桶) | `analysis_longtail` 仅取 **80 条尾部/400** | **中**。尾部定义本身依赖 1200 步探针档的排序,而该档已被 cont 档超越 |
| §一 主榜配方 | P3' + 门控,0.7622 | `amort_P3prime_cont` 数据全量但**步数截断 2,541/4,000,曲线未平** | **低-中**。数据覆盖完整(1.90 epoch),但入口文档自述「续训仍有收益空间」,0.7622 是**下界**而非终值 |

**不在风险清单上(全量背书,可放心引用)**:S1(probe_e1 全 896/400)、S2(pw5 400/400)、
S3(e1/e2/e5 全 400/400)、S4(W01/W02 全量 159,215+896,e3 400/400)、S7(dx + gated_upsample
全 400/400)、S10(sink 普查 400/400)、S11(全量普查)。

## 五、交付缺口(不影响结论,影响可追溯性)

1. **`amort_P3prime_cont` 无 `experiments/` 交付目录**——主榜数字 0.7622 只存在于
   `/home/bc/data/runs/where_b/amort_P3prime_cont_20260811/eval_final/metrics.json`,
   仓库内的 `amort_p3prime_20260810/metrics.json` 仍是 1,200 步的 0.7417。**引用主榜数字时
   必须指向 runs/ 路径**,否则读者会拿到旧值。
2. `runs/where_b/` 下另有 **12 个无交付目录的运行**(P2STRUCT 四臂、五个 P3' 消融
   `abl_{nosim,fixedctx,shufctx,nofilm,cprior}`、`sep060`、`P1_pooled`、`P1_cont`),
   其中消融臂是每个消融行必带的 Δ_const/Δ_shuffle 列的来源,建议补交付。
3. `amort_dx` 的 DX-5 子探针报错(0 可用样本),入口文档 §四已挂「DX-5 重跑」。
