# 决策日志（wave-1 审阅后，2026-08-03）

> 来源：`docs/reviews/REVIEW-impl-wave1.md` 的 22 项待决策 + 3 blocker。常规项由主 agent 按保守默认裁定并在此记录；**标 ⚑ 的需要用户拍板**，未拍板前按建议值执行但不进任何对外口径。

## 一、升级给用户的决策（⚑）

| # | 事项 | 建议 | 状态 |
|---|---|---|---|
| ⚑U1 | **mmart_ppr10k 池 326 源疑似落 PPR10K 官方 val 段**（gid≥1356；此前 §4-A 只覆盖主池） | 326 源全部隔离出训练（标 quarantine）；E20 报官方口径时剔除对应 val 图并在论文注明。已训 build 中含这些源的行在 E20 相关训练里剔除重训成本极低（<1%） | 待拍板，先按建议隔离 |
| ⚑U2 | l6 完成后整体划 held-out build（行动项 D） | 采纳 | 待拍板 |
| ⚑U3 | 补齐 478 cube（tooling-wave1/cube/inventory/supplement_proposal.txt，bucket 均衡 + 未覆盖 minor 优先） | 采纳清单；新增 preset 一律标 **P-extra**：只进 E1 容量测量，永不进条件化训练（保持 P-split 冻结） | 待拍板 |
| ⚑U4 | null 组（未注释终态）247,920 对计入 D-RENDER | **计入**（渲染确定性与标注状态无关），仅用于 Stage-1 预训练，不进任何评测 | 已按默认执行，告知 |

## 二、主 agent 裁定的常规项（生效）

| # | 决策 | 裁定 |
|---|---|---|
| D-01 | P-split 28 个 n<10 小层无 val/test 名额 | 接受（层内全 train；对账时这些 minor 不参与未见-LUT 评测） |
| D-02 | npy33 / hald / s_cache 落盘位置 | 维持 `/var/cache/veradata/dcube/` 与项目内 `s_cache/`（本地快盘）；每日 rsync 到 NFS `datasets/derived/` 备份 |
| D-03 | .3dl 按 OCIO 3DMESH 约定解析（43 个） | 追认 |
| D-04 | T0 = 恒等变换 | 追认，已写入 PLAN §3 权威表 |
| D-05 | 语义掩膜羽化用高斯 σ、几何用 smoothstep 半宽 | 接受口径差异，manifest 里 `feather_kind` 字段区分，评测按 kind 分层报 |
| D-06 | 幅度档数值 | 以 PLAN §3 2026-08-03 内联表为唯一权威（悬空引用已修复） |
| D-07 | winner_confidence=low 的**源图**准入 D-CONSTRUCT | 准入（置信度是标注属性，源图与之无关） |
| D-08 | sanity 批 JPEG q95 / 正式生产 PNG | 追认；正式生成一律 `--img-format png` |
| D-09 | 边界带 band_px=3 | 已预注册进 EXPERIMENTS_v3 INF-1（k∈{1,3,8} 附录消融） |
| D-10 | Δ_const 的 s_∅ 双轨 | 默认=评测集均值常量场；模型自带 s_∅ 走 `s_null=` 参数，报告须写明用了哪轨 |
| D-11 | metrics.json schema | 以 T3 README 的 schema v1 冻结；改 schema 走版本号 |
| D-12 | L6 双掩膜进 harness 形态 | npz 双通道（mask1, mask2），harness 侧按 L6 专用评测器读 |
| D-13 | oracle 缓存 img_id=candidate_id、instr_hash=md5(instruction)[:12]、无指令用 `noinstr` | 追认 |
| D-14 | oracle IoU 判据口径 | 0.5 阈值二值化后 IoU（T5 实测 0.9695 达标）追认 |
| D-15 | DATA_ASSIGNMENT「行动项 B→E」笔误 | 已修复 |
| D-16 | D-RENDER 规模数 | 文档已更新为实测 1,144,000 对（normal/low/abstain/null = 315,952/217,512/362,616/247,920） |

## 三、blocker 修复安排（wave-1.5，已派工）

| blocker | 内容 | 修复 |
|---|---|---|
| B1 | T1 P-split 层内位置法在 preset 集合变化时翻转既有归属；build_splits.py 重跑无声整表重写 | F1：既有归属冻结表 + 增量模式（新 preset 只能进新增名额）+ 重跑需 `--force` 且打印 diff |
| B2 | INF-1 规格的烘焙一致性评测器缺件 | F2：harness 补 `bake_consistency.py`（渲染器→33³ 采样→四面体回读→ΔE00 分位），RD/E22 前必须就位 |
| B3 | T4 内联 split 与 T1 冻结表不同构（81.2% 一致，train 集 11.8% 污染） | F3：T4 挂 T1 sqlite 重生成全部 sanity 批与 val 批；内联规则仅留 fallback 且启动时与旁表校验一致才允许用 |
| — | 各工具 Pyright 报错清扫（metrics.py/oracle.py/generate.py/selfcheck.py 等） | 并入 F1–F3 对应模块；F4 扫尾 |
| — | **跨工具风险：生产渲染器 axis_order=bgr** | F5：D-RENDER after 图 vs colour 四面体渲染对拍 100 例，判定 BGR 约定对 L_cube 监督的影响与修正方式 |

## 四、W1 batch-1 后新增（2026-08-03 晚）

| # | 事项 | 建议/裁定 | 状态 |
|---|---|---|---|
| ⚑U5 | **GL token 身份确认**：词表无字面 "GL" token；候选为 `<retouch_light>`(151646) / `<retouch_color&temp>` / `<retouch_colormixer>` 三个 special token。W1c 保守默认取 `<retouch_light>`，另两个同前向顺带导出对照 | **✅ 用户已定案（2026-08-04）：`<retouch_light>`（id 151646）**——即当初 DiffLMM 可视化所用 token。RO-9 主叙事挂 light；另两 token 的导出保留为附录对照（含 D-22 的跨 token ρ 与掩膜 AUC） | **CLOSED** |
| D-17 | G1 判据修订（区域对立为主判据，方向对立降对照） | 主 agent 裁定，已写入 DATA_ASSIGNMENT 与 EXPERIMENTS_v3 changelog；**需补跑区域对立指令批**（G1 补充批，等 300 源主批完成后追加） | 生效 |
| D-18 | checkpoint 事实记录：当前实验用原作者 5 月发布权重（本项目自己的 SFT checkpoint 尚不存在）；PR-5 的 pre/post-SFT 对照中 "post-SFT" 即此权重，"pre-SFT" 用其 base（llava_qwen2 0.5B 底座） | 记录在案 | 生效 |
| D-19 | E1b 结论采纳：条件通道宽度的"32–64 维够"预测作废；condition 宽度决策改为**等 E1 全量 N\* 落地后**按"高斯参数量 × SVD 曲线"联合判断 | 生效 | — |

## 五、G1 中途审阅后新增（2026-08-03 深夜）

| # | 事项 | 裁定 |
|---|---|---|
| D-20 | **Monitor 假阳性教训**：pgrep ERE 里误用 `\|` 转义导致 G1 进程从未被匹配。修正版 monitor 已换（逐模式独立 pgrep）。新纪律：**长任务启动时必须在交付目录写 `job.marker`（含 PID+启动命令），monitor 与复核一律按 marker+PID 双查，不单靠进程名模式** | 生效，写入 CLAUDE.md 长任务纪律的下一次修订 |
| D-21 | G1 交付形态修复项（审阅提出）：① analyze_g1.py 判据同步 D-17（gate 输出 FAIL→PENDING）；② REPORT 判定表换新口径；③ 删 02:05 陈旧 allcells 口径 viz | 已派修复 agent |
| D-22 | ⚑U5 数据侧补强（审阅建议）：区域对立批**顺带零成本**补两个度量——跨 token 空间两两 ρ + 三 token 各自对 SAM3/GT 掩膜的 AUC，回答"哪个 token 最像主体感知" | 已写入补充批任务的后续要求 |
| D-23 | G1 实测吞吐 40–60s/样本（四任务共两卡），ETA 8–12h；**W1 里程碑 G1 判定顺延**；层选择等主判据数据重扫（勿用 n=15 syn 曲线定层） | 生效 |
| 快照 | 中途体检可信数字（n=15）：**ρ_Y=0.247（24 层 max 0.27）→「s 不是亮度马甲」PASS**；ρ_syn=0.872 方向 PASS（style 任务低尾待分层）；方向对立 ρ=0.883 按对照组口径为正向证据但需 shufctrl 批排除"s 不依赖指令"；ρ_region_opp 待补充批 → **Gate D1 = PENDING** | 记录 |

## 六、G2/G3 守门结果后的裁定（2026-08-04）

| # | 事项 | 裁定 |
|---|---|---|
| **D-24** | **Δ_ceil 估计器口径变更**（G2 提出）：PLAN §4.1 字面的「逐箱条件均值」在 33³ 分箱下 PSNR 硬地板 ≈41 dB，而 D-CONSTRUCT 的恒等映射本身就有 37–41 dB → PSNR_3D 反低于恒等，**8 dB 阈值在任何数据上都不可达**。改用**逐箱最小二乘仿射**（局部线性 = 三线性 3D LUT 的格内真实行为），地板抬到 63.43 dB。 | **采纳仿射估计器为权威口径**，字面口径数字并列保留进附录。**理由不是为了达标**：三道控制臂证明新口径不虚高——对照档（全局线）0.040 dB、L5 故意错配掩膜 0.16 dB（同自由度的 L1/L2 是 19.9/23.5）、Δ_const 严格 0。估计器错的是它低估了 3D LUT 的能力，不是阈值定高了。PLAN §4.1 已改写 |
| **D-25** | **数据坑（影响所有后续实验）**：`.in.jpg` **不是 I_in**，是尺寸任意的 VLM 预览图；after `.jpg` 与 `.cgt.png` 恒为短边 1024 渲染分辨率。强行对齐得到 \|I_in−I_tar\| 均值 75/255 的纯错位。 | **正确取法：从 img 银行按生产同一管线重建**（exif_transpose → LANCZOS 短边 1024）。已写入 DATA_ASSIGNMENT §1.2 硬规则 |
| **D-26** | **Gate D3 主判读档位**（G3 提出）：任务卡指定的 D-CONSTRUCT L1+L4 混了 40 种变换身份，而 oracle s 只编码 *where* 不编码 *which*；天花板预检显示该档上 Δ_shuffle 上限仅 +0.32 dB —— **数据本身表达不了「≥3 dB」判据**，两臂都会被误判为塌陷。 | **采纳 `fixed` 档（同源图/同掩膜/同原语、目标按单一变换重渲染，天花板 +14.90 dB）为 Gate D3 主判读**，`tiered`(+4 dB) 与 `mixed`(字面档) 保留为梯度对照。理由：判据必须在数据可表达的范围内才有意义；且**「零代价」与「零收益」必须区分**——只有 naive 在有收益的档上也塌陷，才证明逃逸通道是真·零代价 |
| **D-27** | G3 方向性读数（**非最终结论**，全量 25000 步在跑）：naive 臂在 fixed 档 Δ_shuffle **+13.06→+18.53**、σ_s q50 0.1418（初值 0.15，**未发散**）、sensitivity 0.0018 → **朴素 4D 没有塌陷** | 记录。**不得据此降级 R-2/R-3**：本实验 s∈[0,1] 只有 where 没有 what，是真实 s 信息量的**下界**情形；真实 s 携带更多信息时逃逸通道的相对代价可能不同。等全量 + 真实 s 档复核 |
| **D-28** | harness 已知口径坑：`_mean_s_null` 对 2-D s 场按最后一维当通道处理，给出的不是常量场 | 记入 harness README 已知坑；Δ_const 用 2-D s 场时必须显式传 `s_null=` |
| **D-29** | G2 的 REPORT.md 因 subagent 工具限制未落盘（正文在 workflow 返回里），其余交付物齐全 | 已派补写；**协议补丁**：任务卡须显式授权 REPORT.md 写入路径 |

## 七、G1 终判与 RO 重排（2026-08-04）

| # | 事项 | 裁定 |
|---|---|---|
| **D-30** | **Gate D1（G1）判 FAIL**。终版数据：ρ_region_opp 中位 **0.884**（判据<0.3，**n_below_0.3 = 0/214**）；**shuffle 对照 0.895 > 同义 0.842**（配对差 −0.015，win rate 0.40）。→ `<retouch_light>` token 在 canonical 读出（pre-softmax, L8–15 均值, D-0 修复）下**基本不依赖指令**，是一张逐图的主体显著场 | **RO-9 不得作为主叙事**。被证伪的是"GL token 这个读法能拿到 where"，**不是**"VLM 里有 where"（H2 未证伪） |
| **D-31** | 措辞更正入档：**ρ_Y 低只排除"s 是亮度的线性/单调翻版"这一种失败模式，不构成"s 有用"的正面证据**（纯噪声场同样通过）。此前"最致命死刑线已排除 ✅"的表述过度解读，已在 EXPERIMENT_INDEX 与本档更正 | 生效。**今后任何"某判据 PASS"的表述必须区分「未触发死刑」与「获得正面证据」** |
| **D-32** | RO-9 的最后一条线索：分层曲线显示早期层 ρ_region_opp 显著更低（L0 **0.045** / L4 0.270 / L3 0.426，而 L8–15 为 0.92–0.99）。低相关有两解——真随指令变 or 纯噪声。**判别只需一个数：那些层的 s 对目标区域的 AUC** | 已派 RO9-L 判别任务。AUC 显著>0.5 且区域对立分离 → RO-9 是**层选错**，救回；AUC≈0.5 → RO-9 作为 where 读出**判死**，降为 analysis |
| **D-33** | **RO 臂优先级重排**：RO-1(self-self) / RO-3(全层全头扫描) / RO-2(logit lens) / RO-5(探针头) **升为第一梯队**；RO-9 降为待判；RO-6/RO-7/RO-8 顺延 | 已写入 EXPERIMENTS_v3 |
| **D-34** | **G2 覆盖面记账（报告 O-2）**：`--limit 600` 按索引顺序截断而非分层抽样 → 真实档只覆盖 l1/l2/l3/l4(74/179)，**l5/l6 零样本**；对照档只有 g1/g2，**g3/g4 零样本**。分池稳健性完整、分 build 稳健性只覆盖一半 | 结论不变（97.8% 的图 ≥1 dB、Spearman(掩膜面积,增益)=0.081），但**补一轮分层抽样覆盖 l5/l6/g3/g4** 以关掉这个口子。已派 |
| **D-35** | G2 报告其余 4 处出入（判据①分母口径两版并列均过 / viz 缺 3 张 / NOTES 若干开发期数字 / Δ_const 真实档为 1.7e-6 浮点噪声非严格 0）已在报告开头「落盘核对声明」如实登记 | 追认 |
