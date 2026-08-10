# 数据分配表：实验 × 训练/验证数据（2026-08-02）

> 配套：`EXPERIMENTS_v3_2026-08-02.md`（实验定义）、`IMPL_DOSSIER_2026-08-02.md`（外部数据管线）、数据资产说明（2026-08-02 晚版）。
> 本文回答：每个实验用哪些数据训练、哪些数据验证，以及切分纪律。

## 0. 资产对计划的三个升级（先说结论）

1. **l 系 build 就是真实局部编辑三元组**（I_in + 指令 + SAM3 掩膜/region + 赢家渲染 + 已知 preset）。v3 里原计划"只能靠合成数据 + PPR10K"的局部实验（G2 真实档、容量阶梯真实档、E14 局部子集、E20 打榜、RO 评分），全部升级为**自家真实数据为主、外部数据为辅**。全领域没有第二家有这个东西（PerTouch 是 SAM 合成打分、InstantRetouch 语料未放出）。
2. **groups.jsonl 的 8 候选是渲染确定性数据**：每组 8 个 (I_in, preset, after) 对，**不依赖标注质量**（渲染是机械的）→ 完成 build 已有 ≈100 万渲染对，是 RD-G Stage-1"抄参数"预训练的金矿，把 DOSSIER 里"4000 cube × 8–16 张标准图 = 3–6 万样本"的方案直接放大 20 倍且换成真实分布。低置信组、甚至弃权组的渲染对都能用。
3. **RO-X1 有名词/无名词对比的数据是现成的**：l 系指令天然带 region 名词，g 系风格指令天然无名词（"胶片感"类）——不用另造。

## 1. 全局切分纪律（所有实验共用一张表）

### 1.1 三级 split

| Split | 定义 | 规模 | 用途 |
|---|---|---|---|
| **S-split（源级，最高优先）** | `hash(source_id) mod 100`：0–89 = **S-train** / 90–94 = **S-val** / 95–99 = **S-test**。跨 build 全局生效（同源在多个 build 出现，split 跟源走） | ≈30.3k / 1.7k / 1.7k 源 | 一切图像侧实验的训练/验证/测试边界 |
| **P-split（preset 级）** | 3,522 preset 按 taxonomy minor 分层：90% = **P-train** / 5% = **P-val** / 5% = **P-test**。**P-test 永不进任何条件化训练**（含 CGLUT/生成器/E19） | ≈3,170 / 176 / 176 | 未见 LUT 泛化（E19/E20） |
| **I-split（指令级）** | 跟随 S-split 的行；另建 **I-novel**：模板外新写指令 ~200 条（含无名词类） | — | 未见指令泛化（E19） |

**落地要求**：S/P split 物化成两张旁表（`source_id→split`、`preset_id→split`），固定 hash seed，入库 `global.sqlite3` 旁或独立 sqlite；**所有实验只读这两张表，禁止各自 ad-hoc 切分**。

### 1.2 硬规则

- `eval100-annotqa-20260727` 永不进训练（已有政策，重申）。
- **winner_confidence=low（≈33%）**：不进 SFT 主训与任何评测 GT；**可进**渲染器/生成器预训练（D-RENDER，渲染确定性与置信度无关）与伪标签挖掘。
- 战前小 build（fresh*/wp* 系）只做 QA 夹具与调试，不进正式训练/评测。
- 评测集一律 normal-confidence + S-test 源；固化后加版本号冻结（进 INF-1 harness）。
- **PPR10K 官方 val 无污染（已核实，见 §4-A）**：训练只用了官方 train 段（索引 1–8871），官方 val（≥8875）零命中，E20 可直接报官方口径。
- 探针类实验（PR 系）一律用 **S-val 源**——避免"探针在背训练图"的质疑。
- **⚠ `.in.jpg` 不是 I_in（D-25，2026-08-04 实测）**：它是尺寸任意的 VLM 预览图；after `.jpg` 与 `.cgt.png` 恒为短边 1024 渲染分辨率。直接三方对齐会得到 |I_in−I_tar| 均值 75/255 的纯错位。**I_in 必须从 img 银行按生产同一管线重建**（exif_transpose → LANCZOS 短边 1024）。

## 2. 派生数据集定义（实验表引用的代号）

| 代号 | 内容 | 构成与规模 | Split 归属 |
|---|---|---|---|
| **D-CUBE** | preset .cube 全量 | 3,522 个生产 preset（磁盘另有 7,083 个可解析 .cube；补齐至 ~4000 的候选清单见 tooling-wave1/cube/inventory，行动项 E） | E1 逐 LUT 过拟合不涉泛化 → 全量可用；报告仍按 P-split 分层报尾部 |
| **D-HALD** | 每 preset 的 Hald 训测对 | 128³ 训色 / 256³−128³ 留出色（**颜色空间 split，非图像 split**，GLUT 协议） | 跟 D-CUBE |
| **D-RENDER** | 完成 build 全部候选渲染对 (I_in, preset_id, after[, mask]) | **实测 1,144,000 对**（T1 全量核实：normal 315,952 / low 217,512 / abstain 362,616 / null 247,920；l 系另有 544,000 张 .cgt 掩膜齐备；I_in 经 img 银行 100% 可回取）；l 系对带 mask/region | 按 S-split 切；置信度不限 |
| **D-SFT-G** | global 线 SFT 行 (I_in, instruction, I_tar, recipe.preset) | 完成 68,085 行（g1 22,803 + g2 22,370 + g3 22,912）；normal ≈ 45.6k；g4 在途续入 | S-split × P-split |
| **D-SFT-L** | local 线 SFT 行 (I_in, instruction+region, mask, I_tar, preset) | 完成 40,482 行（l1 13,978 + l2 12,814 + l3 13,690）；normal ≈ 27.1k；l4–l6 在途 | 同上 |
| **D-CONSTRUCT** | INF-2 合成八级阶梯（L0–L7） | 源图取自家源池（几何/语义/全局三族 × 幅度/羽化/面积档，每级 ≥2000）；语义掩膜用 **SAM3 subject cache** | 训练用 S-train 源、验证用 S-val 源生成两套 |
| **D-VERABENCH-L / -G** | 固化自建评测集（腿 C 的 benchmark 雏形） | S-test 源 × normal-confidence 完成组：local ≈1.5k 组（含掩膜与 preset GT）、global ≈2k 组；掩膜边界人工抽检 10% | 只评不训，版本冻结 |
| **D-PROBE** | PR 系配对 delta 样本 | S-val 源施加已知扰动（ΔCCT/ΔWB/ΔEV/Δ对比）；**RAISE 池优先**（最接近原片、色彩中性）；A5 用 korean/ppr10k 人像池 + PPR10K 掩膜取肤区 | S-val only |
| **D-MASKBANK** | 语义掩膜库 | SAM3 subject cache（全部有主体源）+ PPR10K masks_360p + l 系 region 描述 | 跟源的 S-split |
| **外部** | FiveK 480p（Zeng 包）、PPR10K 官方 val、Cube+/NUS-8（光源 GT）、AceTone-Bench-Transfer、PST50、MagicBrush（人工掩膜） | 见 DOSSIER §4 | 外部自带 split |

## 3. 实验 × 数据分配总表

### 3.1 守门与基建

| 实验 | 训练数据 | 验证/评测数据 | 备注 |
|---|---|---|---|
| G0 代码核对 | — | — | 无数据 |
| G1 s 可辨识性 | —（零训练） | **S-val 源 300 张 × 4 指令**（2026-08-03 修订）：同义对 = `instruction` vs `instruction_short`；**区域对立 = 指向不同区域的两条指令（从 D-SFT-L region 构造），ρ_region_opp<0.3 为主判据**；方向对立（同区域方向词取反）**降级为对照组**——若 s 只编码 where，方向对立 ρ 高是预期行为不判死 | 池按 unsplash/awards/ppr10k 各 1/3。修订原因：初版把"反义"实现为方向词取反（同区域），冒烟 ρ_opp=0.883 被误读为 FAIL；where 场对 what 方向不敏感恰是"s=寻址坐标"假设的正向证据，须换区域对立才构成真检验 |
| G2 oracle 天花板 | —（解析/直接拟合） | ① D-CONSTRUCT(S-val)；② **真实档：D-SFT-L(S-val) 的 (I_in, I_tar, mask) 用 SAM3 掩膜当 oracle s 算 Δ_ceil**；③ 对照 D-SFT-G(S-val) 应得 Δ_ceil≈0 | ②③ 之差就是"自家数据里局部信号强度"的直接测量——**H5 在自家数据上的预登记数字** |
| G3 塌陷通道 | D-CONSTRUCT L1 几百对（S-train） | 同源 held-out | 玩具规模 |
| E1 cube 扫 N | **D-CUBE 全量**（N 扫描先用 400 个按 major/minor 分层子集） | D-HALD 留出色 + 自然图（S-test 源 100 张，替代 GLUT 的 FiveK #4501–4600） | 逐 LUT 过拟合无泛化问题；A0 复现锚点仍按 GLUT 原协议对 45.5 dB |
| E1b SVD | D-CUBE 全量 | — | — |
| E2 基底拟合 | —（L-BFGS 逐掩膜） | 几何族合成 + **D-MASKBANK 语义五类**（S-val 源）+ 全局常数 | 语义类从 SAM3 cache 按主体类别分桶抽 200/类 |
| INF-2 构造集 | 生成自 S-train 源 | 生成自 S-val 源（两套独立） | 变换参数分布抄 PLAN §3 |
| INF-4 分层 | — | FiveK 480p 全量 + PPR10K 官方 val + **D-SFT-L/G 完成组** | E14 的 Δ_ceil 直方图三个来源各画一份 |

### 3.2 渲染器臂（RD）

| 实验 | 训练数据 | 验证数据 | 备注 |
|---|---|---|---|
| RD-STD / RD-A / RD-B / RD-C / RD-D / RD-F | **oracle 档**：D-CONSTRUCT(S-train)，s=GT 掩膜；**真实档**：D-SFT-L(S-train, normal)，s=SAM3 掩膜 | D-CONSTRUCT(S-val) 全阶梯 + D-VERABENCH-L | 两档都跑：oracle 档比容量、真实档比落地；每行必带 Δ_const/Δ_shuffle |
| RD-E 4D LUT 对照臂 | 同上 oracle 档 | 同上 | 用 SA-LUT clut4d.py 自实现 |
| RD-G 生成器 Stage-1 | **D-RENDER(S-train，全置信度，≈90 万对)** + D-HALD（P-train）；监督 = recipe.preset 的 GT LUT 参数（L_cube）+ Stage-0 金标准锚位 | D-RENDER(S-val) + D-HALD(P-val) | **这是资产带来的最大升级**：真实 (图, LUT) 对替代合成方案，规模 ×20；输入=渲染后图（模拟"用户想要这个效果"）与输入=I_in+风格指令两种条件各训一版对照 |
| RD-G Stage-2 | D-SFT-G + D-SFT-L（S-train, normal, P-train） | S-val 行 + **P-test 未见 LUT 组** | checkpoint 禁 val loss，用 ΔE00 分位+方差比 |
| RD-I N/曲线扫描 | 同 RD-STD oracle 档 | 同 | 附录消融 |
| T3 退化验证 | **D-SFT-G only**（100% 全局） | D-SFT-G(S-val) vs 无 s 基线 | 差 <0.05 dB |
| E22 烘焙预算 | — | **D-CUBE(P-val)** 切片导出回读 | 四面体插值回读 |
| E23 负控制 | — | RD 验证集同套，s 换随机/置换/亮度/加噪 | — |

### 3.3 读出臂（RO）

| 实验 | 训练数据 | 验证/评分数据 | 备注 |
|---|---|---|---|
| RO-0 oracle | — | D-MASKBANK | 永久对照 |
| RO-1/2/3/4/9（零训练） | — | **统一评分集**：D-CONSTRUCT L1(S-val) GT 掩膜 + D-SFT-L(S-val) SAM3 掩膜 + G1 三元组（AUC/刻度方差/符号） | RO-9 另跑 pre-SFT 基座对照（PR-5 共用） |
| RO-5 探针头 | D-CONSTRUCT(S-train) GT 掩膜 + RO-8 伪标签(S-train) + D-1/D-2 蒸馏 | **留出语义类**（SAM3 主体类别 leave-out：训练排除 2–3 类，测未见类 AUC）+ S-val | 三档监督叠加消融 |
| RO-6 context encoder | D-SFT-G/L(S-train, normal) 端到端重建 + s_VLM 缓存蒸馏 | S-val 行；**E21 同数据只换 init** | s_VLM 离线缓存（INF-5） |
| RO-7 EDIT token LoRA | D-CONSTRUCT(S-train) 2k + D-SFT-L(S-train) 5k 子集 | RO 统一评分集 | 最后做 |
| RO-8 伪标签厂 | —（离线推理 S-train 源） | 与 GT 掩膜对拍 AUC | 产物喂 RO-5 |
| RO-X1 有名词/无名词 | — | **名词半集 = D-SFT-L(S-val) 指令 200 条；无名词半集 = D-SFT-G(S-val) 风格指令 200 条**（人工核一遍确无名词） | 数据现成，不另造 |
| RO-X2 归一化四档 | — | 每晋级臂在 RO 统一评分集重跑 | — |
| RO-X3 SasP/MasP | — | RO 统一评分集 | 外部 baseline |

### 3.4 探针臂（PR）

| 实验 | 训练数据 | 验证数据 | 备注 |
|---|---|---|---|
| PR-1 颜色探针 | 探针拟合：D-PROBE(S-val 源生成) 的 train 折 | D-PROBE test 折（按源再切折，杜绝同源跨折）+ **C5 像素基线同折** | A1–A4 自家可发；**A6 必须外部 Cube+/NUS-8**（自家无光源 GT）；A5 用人像池+PPR10K 掩膜取肤区 |
| PR-2 sink 诊断 | — | S-val 源 500 张（跨图平均图须覆盖不同色温——按池混采）+ D-PROBE | S6 的 latent sweep 用 VeraRetouch 自身 checkpoint |
| PR-3 H3 判决图 | — | PR-1 数据 + 空间探针（D-MASKBANK S-val） | 双曲线同图 |
| PR-4 因果闭环 | — | D-PROBE 子集 | 干预对象=image tokens |
| PR-5 涌现检验 | — | S-val 源 × g/l 两类指令；**pre-SFT 基座 vs SFT 后同图同指令** | 与 RO-9 共用推理产物 |

### 3.5 端到端与打榜

| 实验 | 训练数据 | 验证/评测数据 | 备注 |
|---|---|---|---|
| E19 条件源替换 | D-SFT-G(S-train, normal, **P-train**) | ① 全局回归检验：S-val；② **未见指令：I-novel + S-val 新指令**；③ **未见 LUT：P-test 组**（"最近邻已见 preset"作对照） | 泛化主张全靠 ②③ |
| E20 打榜 | —（用定稿模型） | **D-VERABENCH-L/G（主榜）** + PPR10K 官方 val（先过行动项 A）+ FiveK 局部子集（E14 切）+ AceTone-Bench-Transfer + PST50 | InstantRetouch 指标复刻跑在 D-VERABENCH 上 |
| E21 决定性消融 | RO-6 同数据，仅换 init | 同 E20 主榜 | — |
| E24 D 层语料 | S-train 源 + 参数化变换确定性施加（DOSSIER §4.4；掩膜用 D-MASKBANK）+ OmniEdit/AnyEdit 色彩子集补充 | 混入后在 D-VERABENCH 复测不退化 | **与现有 build 管线同构，可直接作为 l7+/g5+ 的新 build 类型进生产** |
| E25 Laplacian 支线 | D-SFT-G/L(S-train) | S-val | 与 RD 正交 |
| E26 user study | — | D-VERABENCH 分层抽样 ~200 组 | 掩膜/指令随图展示 |

## 4. 行动项与状态（2026-08-02 晚复核后）

| # | 事项 | 状态 |
|---|---|---|
| **A** | PPR10K 官方 val 交集检查 | ✅ **已核实干净**：journal-archive 全量扫描，`ppr10k/source` 用到 4,310 张不同源（15,605 组占用），文件索引范围 1–8871，**官方 val 段（≥8875）零命中**。E20 可直接报官方口径 |
| **B** | C_GT（.cgt.png）内容确认 | ✅ **已核实：= 逐候选的区域掩膜**（单通道 1024×1536，98.8% 像素为 0/255、软边过渡 217 级灰度）。"配色真值"命名有误导，实为主体/区域掩膜的候选级落盘。**结论**：RD-G Stage-1 的 L_cube 监督从 `recipe.preset` 的 LUT 文件渲染；C_GT 掩膜**直接编入 D-MASKBANK 当 oracle s**（全分辨率、逐候选、软边现成，比 subject_cache 更贴训练样本） |
| **C** | S/P split 旁表物化 + 固定 seed，入 harness | ⏳ 进 tooling wave-1（T1） |
| **D** | l6 完成后整体划为 held-out build（源级+build 级双保险），作最终打榜 fresh 集 | 待拍板（建议采纳） |
| **E** | 3,522 与"4000 cube"差额 preset 盘点补入 D-CUBE | ⏳ 进 tooling wave-1（T2） |
| **F** | D-VERABENCH 固化（S-test × normal + 掩膜 10% 抽检 + 版本号） | ⏳ tooling wave-1 之后（T3 产出 harness 后执行） |
| **G** | 弃权组渲染产物是否留在归档 | ✅ **已核实：全部落盘可用**。弃权组（margin<1.0）渲染产物 100% 在 shards（JPEG 完整性抽验通过）；D-RENDER 实测 **1,144,000 对** |
| **H** | **mmart_ppr10k 池 326 源疑似落 PPR10K 官方 val 段（gid≥1356）**——此前 §4-A 结论只覆盖 ppr10k/source 主池 | ⚠️ **新发现，待处置**（建议：326 源全部隔离出训练，E20 报官方口径时剔除对应 val 图并注明；见 DECISIONS） |
