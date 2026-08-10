# 实验注册表：module / what / where / status

> **当前实验事实的唯一人工入口。** 本表记录“做了什么、证据在哪里、当前能说什么”。
> 原始数值仍以各实验目录的 `metrics.json` 为机器真值，方法与预注册判据见
> [`EXPERIMENTS_v3_2026-08-02.md`](EXPERIMENTS_v3_2026-08-02.md)。
> 本表不把设计臂、运行状态、gate 判决混成一件事。

**范围**：本表覆盖本轮 local-retouch 的 A/E/G/RD/RO/PR 核心实验。`tooling-wave1` 属基础设施验收，
`docs/iaa_benchmark/` 属更早的 IAA 评测战役，`experiments/lut_renderer_pilot/` 只有旧协议、没有可引用
结果；三者不混入本轮结果状态。

## 1. 如何读这张表

- **module**：`render`（渲染器/容量）、`readout`（VLM 空间读出）、`probe`（表征探针）、
  `basis`（s 场基底）、`eval`（评测/天花板）、`e2e`（读出接渲染器）。
- **what**：一句可证伪的问题，不写愿景。
- **where**：实验目录是结果证据；`REPORT.md` 是解释；`metrics.json` 是数值真值；
  `STATUS.md` 只说明执行状态，不能替代结果。
- **status**：`planned` / `running` / `completed` / `invalidated`。实验跑完但 gate FAIL，仍是
  `completed`；只有实现错误、协议被替代或证据作废才是 `invalidated`。
- **gate**：`PASS` / `FAIL` / `MIXED` / `INCONCLUSIVE` / `N/A`，与 status 分开。
- **证据等级**：`A` = 有 metrics + report + 可追产物；`B` = 有机器结果但报告/状态未同步；
  `C` = 仅协议、计划或外部依赖，不能当完成结果。

## 2. Render：渲染器、容量与基底

| ID / module | what | where | status / gate | 当前结果与边界 |
|---|---|---|---|---|
| **A0** · render | 本地 GLUT 复现是否足以充当后续比较锚点？ | [`experiments/A0_glut_repro_20260803/`](../experiments/A0_glut_repro_20260803/)：`REPORT.md`、`metrics.json`、`runs/rec_60ep/` | `completed` / **REVISE** · A | 原论文 45.47±0.3 dB 不适合作本语料 gate。full 臂 −9.2 dB 由 `10·L_hc` 单位失配造成，修复后约 −0.12 dB；rec 缺口主要是步数不足和语料更难。旧 full 数字已作废。 |
| **E1** · render | 单张专业 LUT 至少需要多少高斯基元？ | [`experiments/E1_cube_N_20260803/`](../experiments/E1_cube_N_20260803/)：`runs/main/metrics.json`、`runs/main/per_fit.jsonl`、`STATUS.md`；终审见 [`REVIEW-result-W1batch1.md`](reviews/REVIEW-result-W1batch1.md) | `completed`（REPORT/viz 未收口） / **PASS: N*=48** · B | 三门 `p50<0.5 / p90<1 / p99<2` 下 N=32 未过，当前结论为 **N*=48**。不能再引用 N*=32。 |
| **E1b** · render | LUT 库是否能被 32–64 维低秩条件表示？ | [`experiments/E1b_svd_20260803/`](../experiments/E1b_svd_20260803/)：`REPORT.md`、`metrics.json`；留出复核见结果审阅 | `completed` / **FAIL** · B | 32–64 维预测被推翻。`r≈384` 只是在字典与评测双重 in-sample 的记忆口径；留出泛化需 **r≥1536**。因此“高斯约 12× 更高效”已撤回。 |
| **E2** · basis | 14 维 what/where 基底能否表达常用掩膜；受约束 s 轴能否真正合成平顶带通？ | [`experiments/E2_basis_fit_20260803/`](../experiments/E2_basis_fit_20260803/)：`REPORT.md`、`metrics.json`、`metrics_axis_response.json`、`viz/` | `completed` / **PASS（主判据）** · A | 线性/径向约 0.987/0.986，语义 0.873；薄环单调 0.342、带通 0.975。补件后受约束 M=12 轴 mask 级 0.9754，闭合了旧审阅指出的关键前提。单调薄环失败是解析必然；A-4 优化预算控制增益 0.0105、略超 `<0.01`，两项都必须随主结论保留。 |
| **G2** · eval/render | 自家真实局部数据是否真的需要第 4 维 s？ | [`experiments/G2_oracle_ceiling_20260803/`](../experiments/G2_oracle_ceiling_20260803/)：`REPORT.md`、`metrics.json`、分层补批 | `completed` / **PASS** · A | 分层补批后的真实档 Δceil 中位 **8.037 dB**；6/6 个 l build 均为强正值；全局对照约 0.040 dB。证明“数据里有局部信号”，不证明真实 VLM 已能提供 s。 |
| **G3** · render | 朴素 4D 参数化是否会沿 σs 逃逸通道自发塌回 3D？ | [`experiments/G3_collapse_20260803/`](../experiments/G3_collapse_20260803/)：`REPORT.md`、`metrics.json`、10 run 轨迹 | `completed` / **INCONCLUSIVE** · A | fixed/tiered 未塌陷（Δshuffle 18.08/6.04 dB）；mixed 中 σs↑、s-sensitivity↓，但信号本身很弱。结论是“通道未证实、亦未否证”，**不能**据此删 R-2/R-3。 |
| **RD-ORACLE**（RD-STD/RD-A/C/D/E）· render | 给定干净 oracle s，轻量 4D 算子能否表达局部编辑并通过负控制？ | [`experiments/RD_std_e_20260803/`](../experiments/RD_std_e_20260803/)：`REPORT.md`、`metrics.json`、75/75 runs | `completed` / **MIXED** · A | RD-STD 在 L1–L4 相对同 N 3D 的 in-mask 增益为 **+20.6 至 +23.5 dB**；错配 L5 按预期失败；单轴 L6 只 +4.97 dB，支持多轴需求；L0 全局退化仍 −0.93 dB（未过门）；L7 意外通过。全分辨率 s 明显优于 32×32 s，空间分辨率是现实瓶颈。 |
| **RD-G** · render/e2e | 参数生成器从 CGLUT 式 MLP 换成 transformer 是否更有效；优势是否延续到 s 轴？ | [`experiments/RDG_transformer_20260803/`](../experiments/RDG_transformer_20260803/)：`metrics_converged.json`（最新机器结果）、`REPORT.md`（仍以早期 step 为主）、`runs2/` | `running`（Stage-1 完成，Stage-2 未形成终判） / **Stage-1 PROMOTE** · B | 最新 Stage-1 机器结果中 G-Base 的 ΔE00 p50 2.67、MLP 9.06，降幅 70.5%；G-Lite 3.61，降幅 60.2%。但报告仍混有 step-2000/4000 口径，且 **有 s 的 Stage-2 尚无审阅结论**；现阶段只能说 transformer 参数效率很有希望。 |

## 3. VLM readout：what 与 where

### 3.1 Where：空间定位与指令可控性

| ID / module | what | where | status / gate | 当前结果与边界 |
|---|---|---|---|---|
| **G1 / RO-9** · readout | `<retouch_light>`→image attention 是否随“编辑哪一区域”改变？ | [`experiments/G1_s_identifiability_20260803/`](../experiments/G1_s_identifiability_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **FAIL** · A | ρregion-opp=0.884，0/214 过 `<0.3`；AUC_target 全层约 0.48–0.53。它是主体显著性，不是指令条件 where。旧“shuffle 0.895 反超同义 0.842”是不同子集造成的假象；正确说法是三条件不可分。 |
| **RO9-L** · readout | RO-9 是否只是层选错了？ | [`experiments/RO9_layer_verdict_20260804/`](../experiments/RO9_layer_verdict_20260804/)：`REPORT.md`、`metrics.json` | `completed` / **FAIL** · A | 24 层没有一层达到指令目标定位门；早层低相关来自不可复现场，不是被救回的指令信号。 |
| **RO9b** · readout | 修层、head 聚合和 self-self 算子能否救回 RO-9？ | [`experiments/RO9b_readout_fix_20260803/`](../experiments/RO9b_readout_fix_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **定位 PASS；可控性 FAIL** · A | 主体定位 AUC 可从 0.661 提到约 0.920–0.935，说明旧 0.648 有算子问题；但 instruction AUC_target 最高仅约 0.572，修好定位仍没有指令依赖。 |
| **RO-1** · readout | 零训练 CLIP self-self 是否是可靠的 where 基线？ | [`experiments/RO1_selfself_20260803/`](../experiments/RO1_selfself_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **定位 PROMOTE；可控性未过** · A | ClearCLIP 主体 AUC 0.930–0.939；但固定无名词短语本身已有 0.907，净增量 +0.009 不显著。它是强主体先验，不是充分的区域可控性证据。 |
| **RO-X1** · readout/eval | 去掉名词是否让 CLIP 的区域能力崩掉，从而证明 VLM 不可替代？ | [`experiments/ROX1_clipside_20260803/`](../experiments/ROX1_clipside_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **FAIL（原假设）** · A | 无名词固定短语仍有 AUC 0.907，但 AUC_target 0.523；证明普通 AUC 会把主体先验误当指令理解。原“global/style 指令天然无名词”的数据切分也被实测否决。 |
| **RO-2** · readout | logit lens 能否给出跨图可比的绝对刻度 s？ | [`experiments/RO2_logitlens_20260803/`](../experiments/RO2_logitlens_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **FAIL** · A | 跨图绝对刻度不成立，判别比 0.188。重要正结果在 `emb`：进入 LLM 前目标词 AUC 0.829、无关词 0.523；LLM 前向逐层抹掉词特异性。 |
| **RO-3** · readout | 单头 text→image attention 或融合能否恢复指令条件 where？ | [`experiments/RO3_layerhead_scan_20260803/`](../experiments/RO3_layerhead_scan_20260803/)：`REPORT.md`、`metrics.json` | `completed`（`STATUS.md` 陈旧） / **PROMOTE** · B | OOF 融合 AUC_target 0.830；L11H5 差分场对 `.cgt` AUC 0.840，零对比度对照 0.522。说明 G1 失败主要是 query token 与 head-mean 读法错误。generated-position 未重跑、构造 S-val 仅 22，是现有限定。 |
| **G1b / RO-D** · readout | DiffLMM attend-and-segment 是否能靠去 sink 救回 G1？ | [`experiments/G1b_difflmm_20260803/`](../experiments/G1b_difflmm_20260803/)：`REPORT.md`、`metrics.json` | `completed` / **sink 假说 FAIL；query 诊断成立** · A | 改归一化/去 sink 本身不能救回；真正决定因素是“谁作为看过指令的 query”。结果支持训练 text-side/bilinear readout，不支持继续调 canonical GL attention。 |
| **RO-W** · readout/e2e | VLM 是否能直接预测 E2 的 14 维基系数并在真实 D-SFT-L 上泛化？ | [`experiments/ROW_basis_coeff_20260803/`](../experiments/ROW_basis_coeff_20260803/)：`STATUS.md`、`NOTES.md` | `running` / **PENDING** · C | 离线 w* 上界与特征缓存已完成；instruction-token 主档训练、真实 D-SFT-L 评测、metrics、viz、scache、REPORT 尚未完成。它是当前最直接的“where 接 renderer”缺口。 |
| **MCQ-L** · readout/e2e | MetaCanvas 能否在全量 normal-confidence D-SFT-L L1–L6 上同时读出 instruction-conditioned where 与 4D GLUT 参数？ | [`experiments/MCQ_full_local_l1l6_20260804/`](../experiments/MCQ_full_local_l1l6_20260804/)：`NOTES.md`、`runs/` | `running` / **PENDING** · C | 输入严格为 `I_in + instruction`；`.cgt` 监督空间场、`I_tar` 监督可导 4D renderer。先以 source-disjoint inner selection 比较同结构两组优化配置，再锁定配置在完整 S-train 重训；不把 oracle-mask RD-G Stage-2 当成端到端结果。 |

### 3.2 What：颜色、曝光与低层视觉信息

| ID / module | what | where | status / gate | 当前结果与边界 |
|---|---|---|---|---|
| **PR-1 / PR-3（PR13）** · probe | what/where 是否同时存在；是否分处不同层；颜色是否被 3-latent 读出口丢掉？ | [`experiments/PR13_probe_whatwhere_20260803/`](../experiments/PR13_probe_whatwhere_20260803/)：`REPORT.md`、`metrics.json`、`metrics_raw.json` | `completed` / **MIXED** · A | A1–A5 颜色 R²=0.89–0.93，但像素统计 5/6 更强；A6 仅 0.134，视觉塔到 connector 0.44→0.018。where token probe AUC 0.953；双线性交互 AUC_target 0.917，shuffle 后约 0.54。what/where 曲线都近乎全层平坦，不能再写“分处不同层”。 |
| **PR13 因果补件** · probe | 可解码颜色方向是否被下游实际使用？ | 同上 `REPORT.md` §三 | `completed` / **PARTIAL** · A | steering 幅度比随机方向 3.95×，提供部分因果证据；rank-1 擦除操作检查失败，不能据此宣称因果闭环。 |

## 4. 尚未形成结果的设计臂

下列名字出现在计划里，但当前工作区没有足以支持“已完成”的结果目录或终态产物：

| module | 设计臂 | 当前判断 |
|---|---|---|
| readout | RO-4、RO-5、RO-6、RO-7、RO-8 | `planned`。RO-W/RO-3/PR13 已改变其中若干设计前提，开跑前应先改协议。 |
| probe | PR-2、PR-4、PR-5 | `planned`。PR13 已覆盖 PR-1/PR-3 的大部分问题，但多秩擦除与 pre/post-SFT 涌现仍未做。 |
| e2e | E19、E20、E21、E24、E25、E26 | `planned`。没有端到端主榜、未见指令/LUT 泛化或 user study 结果。 |
| eval | D-VERABENCH 固化、E14 分层 | `planned/running`。G2 给出评测方法学和信号强度，不等于 benchmark 已发布。 |

## 5. 跨实验依赖：现在真正完成到哪一层

```text
真实数据有局部信号（G2 PASS）
        │
        ├─ oracle s 下渲染器有容量（E2 + RD-ORACLE PASS/MIXED）
        │       └─ 标准 LUT 烘焙已验证（RD-G Stage-1）
        │
        └─ VLM 里有 where（RO-3 / PR13）
                ├─ 免费 GL attention 读法失败（G1 / RO-9 FAIL）
                └─ 可训练 instruction-side / bilinear 读法可行（RO-3 / PR13）
                        └─ 真正接到 renderer 并在真实局部数据上验证（RO-W / RD-G Stage-2）【未完成】
```

因此目前可以证明“**两端分别可行**”，还不能证明“**端到端链路已经成立**”。

## 6. Where 的可复现性边界

“产物在工作区可查”不等于“只 clone 仓库即可完整重跑”。当前至少有以下外部依赖：

| 实验族 | 仓库外依赖 | 影响 |
|---|---|---|
| E1/E1b | `/var/cache/veradata/dcube/` 的 GT/npy33 缓存 | 可读现有 metrics；重跑需重建或恢复 cache |
| G1/RO/PR | `/home/bc/data/models/VeraRetouch`、`/var/cache/veradata/scache/`、主体 mask bank | 模型、原始 stacks/scache 不在 git；报告与聚合结果仍在实验目录 |
| RO-W | `/home/bc/data/row_basis_20260803/` | 当前训练和评测尚依赖外部特征缓存 |
| RD-G | `/dev/shm`/本地 cache 与外部 D-RENDER 资产 | 训练缓存可重建但不耐久；当前机器结果在实验目录 |

所以 `where` 至少要同时写清“最终证据路径”和“重跑所需外部资产”。后续 REPORT 应把外部路径、
数据版本/digest、构建命令放进 manifest，避免只留下机器本地绝对路径。

## 7. 更新规则

1. 新实验先在本表登记 `planned`，再创建目录；不允许先跑完再补 ID。
2. `completed` 必须至少有 `metrics.json`、`REPORT.md` 和复现入口；缺一项就在状态后注明交付缺口。
3. 当前数值只改本表与 [`EXPERIMENT_RESULTS_CURRENT.md`](EXPERIMENT_RESULTS_CURRENT.md)；
   `PLAN / EXPERIMENTS_v3 / DECISIONS / HANDOFF` 只保留协议、事件和历史快照。
4. 被推翻的数字不静默删除：在原报告保留历史，在本表只保留当前判词并写明 superseded 关系。
5. 所有定位指标必须同时报目标条件和负控制，至少包括 `AUC_target`、shuffle/无关词、固定 deictic 主体先验；
   普通主体 AUC 不能单独支持“听懂了指令”。
