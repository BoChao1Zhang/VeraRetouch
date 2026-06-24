# Source-QA 三套 LLM-QA 问卷 + 规则清洗器设计

> 日期：2026-06-17 ｜ 适用：`dataset_build/source_qa` 的图像与预设 LLM-QA
> 需求来源：`docs/source_qa/LLM_QA_REDESIGN_REQUIREMENTS_2026-06-17.md`（方法论 P1–P6、§6 校准、§7 验收）
> 关联：`SOURCE_QA_REVIEW_2026-06-15.md`、`DATASET_QUALITY_AUDIT_2026-06-13.md`
> 产出方式：三套设计各经「设计 → 对抗评审 → 折叠修订」，每套题面/清洗器均按**真实数据现实与代码**（非需求文档示例）重新设计；本文为汇编 + 跨流程共享约定。
> 修订：2026-06-17 增补 §1.1 **题号反作弊设计**（题号去语义 + 固定乱序 + 异质谓词 R）；流程 1 把不触发的 **safety 维替换为「过度后期破坏」缺陷**（并入 `defect_count`，0..5→0..6）。
> **2026-06-18 实现 + pilot R1–R4 修订（见下 §0' 实现修订记录；代码为最终事实源）。**

---

## 0'. 实现修订记录（pilot R1–R4，2026-06-18）

> 本节记录设计落地（`config.py` + `qa_clean.py` + `test_qa_clean.py` + `qa_runner.py` + `pilot.py` + `db.py` 迁移）
> 后，在**固定 200 图 pilot**（`dataset_build/source_qa/pilot/`）上经 4 轮真实 35B 推理 + read-only 子代理对抗
> 审阅（Review-40）迭代出的、**据真实数据回调**的增量。下文各章原伪码保留为设计依据；**实际运行以代码为准**。
> 全程：硬 drop 关闭（DEFECT_HARD_DROP/PRO/INTENT/KAPPA_PASS=False）、未放宽任何验收门、SCRAMBLE_SEED 不变。
> 完整走势/证据见 `pilot/FINAL_REPORT.md` 与 `pilot/changelog.md`。

**题面修订（位置码/POS_MAP/SCRAMBLE_SEED 均不变，仅文本）**
- **ANCHOR(IMQ)/T1(aes)/T1·T1b(preset)/H1·H2(aes)** 去方位词「上方/上面」——R1 实测 ANCHOR yes=0.11（模型看得到图
  REAL_F=0.97 却把「上方有图片」答 0，"上方"被读成画面上部区域），改去歧义后 yes→1.00（§1.3/§2.3/§3.3 题面）。
- **CLEAN_F** 去「画面干净」前缀（被读成「画面不杂乱」致繁杂干净图误判）；只问叠加物，与 CLEAN_R 严格互补（§1.3）。
- IMQ system prompt 增「检查四角/边缘半透明水印、版权文字、字幕、拼贴分割线」以降 CLEAN_R 欠检（§1.4）。
- AES system prompt 强化「多数维度写 0、很少全维出众」反 merit 饱和（§2.4）。

**清洗器语义修订（`qa_clean.py`，确认 by Review-40 看图）**
- **软对矛盾仅计 F=1∧R=1**（IMQ 软对 + AES 软对）：both-0 = 该维度「诚实中庸」（既不明显好也不明显坏），非逻辑矛盾。
  R1 实测 AES 全部 78 条软矛盾皆 both-0、子代理看图确认皆诚实中庸。此改既消假矛盾又解 merit 饱和
  （merit 众数集中 0.50→0.30）。acquiescence 仍由 honesty 硬门(aes H1=1/H2=0)+REAL 硬对+ANCHOR(imq) 兜住。
  **CONTRA_TOL 不变（IMQ/AES/preset=1/1/0）**——仅纠正「中庸=矛盾」类别错误，非放松容差。（修订 §1.6 门4、§2.6 门5）
- **CLEAN 硬对 → 软对**（与 INTACT 同）：R3 实测 IMQ 硬矛盾 23 中 20 为 CLEAN both-0（模型 768px 欠检淡水印/拼贴，
  CLEAN_F=0 但 CLEAN_R=0=诚实欠检，非真不一致）；CLEAN_F=0 仍由 verdict 路由 review(invalid:CLEAN)。
  IMQ 硬对仅留 **REAL**（§1.6 hard=["REAL"]）。
- **FACE_GT 只信 Haar 正检**：cv2 Haar 漏检率高（R1 trap:face 10 例中 8 例 mff=0 实为侧脸/小脸/遮挡真脸）。
  改为 mff>IMQ_FACE_MIN 才强制 FACE_GT=1；mff≤min（漏检 FN 高发）→ skip 不硬剔（§1.6 门3）。
- **解析 dup 容错**：同码重复**同一比特**为良性（模型常见），折叠之；仅同码**冲突比特**判 parse:dup（§1.2/§1.6 门1）。

**阈值回调（据 pilot 真实分布，§1.8）**
- **NOISE_SIGMA_DROP_ABOVE 18→800**：实测 noise_sigma 中位 232/p90 839（非 0..25 sigma 量纲），旧 18 触发率 99.5%
  使 reconcile 把全部 keep 误降 review；回调到 ≈p90 后 IMQ verdict 恢复 keep 130/review 65 可分。

**未跑/未达（详见 FINAL_REPORT §3）**：流程 3 preset 整轮未跑（LR farm 离线 + 6 探针需 mean_luma 等列未算，
清洗器+13 单测已就绪待真渲染）；缺陷/负向题 [3%,97%] 在策展 pilot 结构性不可达（需 degraded 队列/全库分层）；
LANDSCAPE/T2 待 EXIF P-1、COLOR 待 is_bw_img P-2；REAL 硬矛盾 2.5% 与 PRO/INTENT 待金标 κ≥0.6 才开硬 drop。

---

## 0. 执行摘要

把现行「让 35B VLM 直接打分 → 直接采信」（全库 `B_quality` 98.9% 判 3、与 `musiq` 矛盾、门形同虚设）改造成**三套独立的二元判断题问卷**，每套都遵循同一方法：**只问真/假二元题 → 每个关键判断配正向 F + 逻辑反向 R（F⊕R 互斥）+ 确定性陷阱/锚点 → 强规则清洗器先裁定"这次回答可不可信" → 不可信不采信（重问一次仍不可信 → review，永不进自动 keep/drop）**。可区分质量用**缺陷/优点命中计数**替代坍缩的 0–3 分级。**题号对模型不透明、F/R 配对在固定乱序里拉远（§1.1）**，使一致性自检无法被"逻辑反推"绕过。

| # | 流程 | 对象 | 角色 | 可区分信号 | 是否可硬 drop |
|---|---|---|---|---|---|
| 1 | **摄影图像质量 QA** | 单张源图（1 次 vLLM 调用） | 技术质量门：有效性 + 技术缺陷（含过度后期破坏） | `defect_count`(0..6) | 可（validity/face 经 κ 标定后） |
| 2 | **摄影图像审美 QA（通用结构角度）** | 单张源图（独立第 2 次 vLLM 调用） | 跨题材通用结构性审美 | `merit_frac`(0..1) | **否**（仅 keep 优先级排序，永不 drop） |
| 3 | **Preset QA** | 单 preset × 6 固定探针的真实 before/after 渲染对 | 预设 look 专业/可学/连贯 | `coherence_score`(0..1) | PRO/INTENT 经 κ 后可（preset 永不自动 keep，all-pass→review） |

**与现行的关键差异**：① 分级 → 二元 + F⊕R 一致性自检；② 直接采信 → 先过可信度门，不可信→review；③ 单一 `B_quality` → 可区分的缺陷/优点计数；④ 质量与审美**拆成两次独立调用**（审美只作软排序，符合"高质量图退化再还原"的数据哲学）；⑤ preset 弃用全为 None 的 `style` 标签，重构为"跨 6 场景看 look 是否专业/可学/连贯 + 确定性配对指标当陷阱真值"；⑥ **题号去语义化 + 固定乱序**，杜绝模型靠"我答了 F 就反推 R"的作弊式自洽。

**主要风险**（详见 §4）：可信≠正确（任何题进硬 drop 须经 §6 金标 κ≥0.6）；两项 iqa 前置改造（EXIF 旋正、`is_bw_img`）未落地前部分陷阱降级失效；tad66k ~800px 低分辨率主池的题面退化与池间偏置；LUT 与真实 LR 渲染量纲差异；审美信号相对库内已有 `aesthetic`/`aesthetic_vlm` 的增量 ROI 需 pilot 证明。

---

## 1. 跨流程共享约定（三套统一，先读这一节）

三套是**三次相互独立的 vLLM 调用 + 三个独立清洗器**，各自 system prompt + 题库固定（命中 vLLM prefix-cache），但共用同一套机制与确定性信号。

### 1.1 题号反作弊设计（题号去语义 + 固定乱序 + 异质谓词 R）

**要解决的漏洞。** F⊕R 一致性自检的全部价值在于"模型独立地对两个逻辑相反的陈述各自看图判断"。但若正向题 `A1` 紧跟反向题 `A2`、且题号暴露二者是同维度一对、R 又是 F 的字面否定，模型可以**不再看图**，纯靠"我 A1 答 1，那 A2 按逻辑答 0"满足 F⊕R——一致性变成恒真的格式逻辑，丧失"证明看了图/没幻觉/没敷衍"的诊断力（P3 失效）。三重泄漏：① 同字母（同维度）② 相邻序号（明示配对）③ R = F 的字面否定（可逻辑反推）。

**四条设计规则（三套通用）：**

- **R1 题号去语义。** 模型看到的题号只是**固定的"位置码"（2 位数字 `01..NN`）**，不含任何维度/极性/配对信息。维度、claim、极性（F/R）、配对、真值源**只存在于清洗器内部的一张固定映射表**（`POS_MAP`，问卷的单一事实来源）。各流程 §x.3 的"题库表"即该内部映射表的**审计视图**（带 claim 审计 ID 便于人读），**模型永远看不到它**。
- **R2 固定乱序、配对拉远。** 用一个**固定置换**（由 `SCRAMBLE_SEED` 决定，每流程一份）把题目打散呈现：每对 F/R 的两题间隔 **≥ ⌈N/3⌉ 且绝不相邻**，不同维度与陷阱题交错。固定单一乱序**不破 prefix-cache**；35B 是无状态推理、跨样本学不到"位置码 03 与 19 是一对"。**只有重问（reask）时才每样本重洗题序**（罕见、不吃缓存，额外破除模式化）。
- **R3 R 写成"反方向的正向证据"（异质谓词）。** 让 R 必须独立看图才能答，而非 F 的字面否定。例：REAL_F=`真实相机照片`，REAL_R 不写"不是真实照片"，而写`画面带有界面/截图元素(状态栏/光标/菜单/播放控件)、印刷海报排版、3D/CG 渲染感或 AI 异常细节`——找具体破绽，不能由 F 逻辑反推。**隐藏配对（R1+R2）已是主防线；异质谓词是对"字面否定最易被反推"题（尤其有效性 R）的叠加硬化。**
- **R4 保留 ID 锚定 + 审计可读。** 输出仍是 `<位置码><bit>`、按位置码解析（漏答/串行不错位，需求文档要的对齐鲁棒性保留）；清洗器先把位置码经 `POS_MAP` 还原成 claim+极性，再按 claim 判 F⊕R。维度/极性对**模型不透明、对人（审计/Web UI）透明**两全。

**worked example（流程 1 片段）：** 内部 `POS_MAP`（审计视图）把 claim 散布到不相邻的位置——

| 位置码 | 展示位 | 维度(内部) | claim | 极性 |
|---|---|---|---|---|
| `03` | 3 | validity | REAL | F |
| `07` | 7 | defect | SHARP | F |
| `11` | 11 | trap | LANDSCAPE | trap |
| `19` | 19 | validity | REAL | R |
| `22` | 22 | defect | SHARP | R |

模型只看到无语义的 `01 02 … NN` 共 N 题、REAL 的正反隔了 16 位且谓词不同 → 逻辑对账既远又难，不如老实看图；输出 `031 070 110 190 221 …`，清洗器按 `POS_MAP` 把 `03`/`19` 归到 REAL 再判 F⊕R。

### 1.2 ID 锚定输出格式与解析（位置码）

模型侧：system prompt 按**固定乱序**列出 `01..NN` 每个位置码对应的题面（无维度提示、无配对提示）。逐题输出「位置码紧贴答案」（位置码 `03` 答真=`031`，答假=`030`），题间单空格，按位置码升序，每码恰一次。**每个答案自带位置码 → 漏答/串行/多答都不会整体错位**。

- 解析：`re.findall(r'(\d{2})([01])', s)` → `{pos: bit}`；要求恰好覆盖本次全部位置码集合、每码唯一、无未知码 → 否则 `reliable=False, reason=parse:{empty|missing|dup|unknown}`。
- 三套统一用**纯数字 2 位位置码**（preset 取消旧 `T1b` 字母后缀特例，反向锚点也只是另一个位置码）。
- 成本 ~3 字符/答案，比定长位串多几十 token，换来消除位错幻觉 + 隐藏配对（方案 A）。定长位串（方案 B）会重新暴露"位置=题号"且无法隐藏配对，仅留作受控 A/B（§4）。

### 1.3 共享清洗器骨架（确定性，三套复用）

```python
def clean(raw, asset, pos_map, expected_pos, cfg):
    # pos_map: {位置码 -> (dim, claim, polarity in {F,R,trap,anchor}, pair_id, truth_source)}
    out = {"reliable": True, "reason": None, "contradiction_count": 0, "trap_fail": None}
    # 门1 解析门: 解析 位置码→bit, 要求恰好=expected_pos, 每码唯一, 无未知码, 非空
    #            否则 reliable=False, reason=parse:{empty|missing|dup|unknown}
    # 门1.5 还原: 用 pos_map 把 {位置码:bit} 还原成 {claim: {F:bit, R:bit}} 与 {trap/anchor:bit}
    # 门2 锚点门: ANCHOR 必=1 (preset 另含反向锚点 claim, 必=0 且与正锚互斥) 否则 reason=anchor
    # 门3 确定性陷阱门(硬): 逐 trap 与"我方非-VLM 信号"比对(见 1.4); 不符→reason=trap:<claim>
    #            信号为 None 时 skip 该陷阱并计数 trap_skipped(落库监控, 防静默失效)
    # 门4 正反矛盾门: 按 claim 判 F⊕R —— 硬对违反即 contradiction_hard; 软对统计违反数, >CONTRA_TOL→contradiction_soft
    # 门5 裁定: 以上全过 → reliable=True, 计算可区分信号(defect_count/merit_frac/逐探针)
    # 重问: reliable=False → 重问 1 次(temp=0, 重洗题序); 仍 False → review, 永不自动 keep/drop
    return out
```

三套差异仅在：`POS_MAP`/期望位置码集、硬/软对划分、陷阱真值源、可区分信号定义、判级映射。

### 1.4 统一的"确定性信号 → 陷阱真值源"对照表

陷阱/锚点的真值**只能**取自我方非-VLM 信号或构造常量（P4/P6）。下表汇总三套用到的真值源与现状缺口：

| 信号（精确列名） | 来源 | 现状/缺口 | 用于陷阱 |
|---|---|---|---|
| 构造常量 =1 / =0 | 题库写死 | 恒可用 | 质量 ANCHOR、审美 ANCHOR、preset 正/反锚点、审美 H_honesty |
| `assets.width/height` → `width>height` | `iqa.py` 原生 decode | ⚠ index 内为 NULL，仅 iqa 后有值；且 **decode 未 `exif_transpose`**，朝向语义可能与送审图不一致 | 质量 LANDSCAPE、审美 LANDSCAPE（**见 §1.5 前置 P-1**） |
| `assets.max_face_frac` | `iqa.py` cv2 Haar（portrait 池） | 有漏检噪声 | 质量 FACE_GT/no_face、审美 FACE/`M_moment` 发题口径 |
| `assets.is_bw_img` | 需新增（饱和度+色度双条件检测） | ⚠ **当前不存在**，图像侧从未计算（`is_bw` 仅 PRESET 填，`ingest.py:125`） | 质量 COLOR、审美 COLOR（**见 §1.5 前置 P-2**） |
| `preset_previews.paired_metrics.delta_e2000_mean` | `paired_metrics.py` | 已算（每探针） | preset CHANGED（近似 no-op）、order=0 no-op 门 |
| `preset_previews.paired_metrics.ssim` | `paired_metrics.py` | 已算 | preset DESTRUCT（结构破坏） |
| `preset_previews.paired_metrics.clip_pct / noop_score / hist_emd_ab` | `paired_metrics.py` | 已算 | preset order=0 门、一致性旁证 |

### 1.5 两项共享的 iqa 前置改造（阻塞项，影响多套）

> 这两项不是某一套私有，而是质量与审美**共用**的确定性真值前提，应统一在 iqa 侧一次做掉。

- **P-1 EXIF 旋正**：`iqa.py` 的 decode 当前无 `ImageOps.exif_transpose` → 落库 `width/height` 是原始像素栅格朝向，而 VLM 收到的是 EXIF 旋正后的显示朝向（手机/相机竖拍 JPEG 常带 orientation 6/8）。这会让**质量与审美的 LANDSCAPE 陷阱**的横/竖真值在 portrait 池系统性打错。修复：decode 后 `exif_transpose`，重算 `width/height/megapixels/longedge`，重跑受影响图。落地前置 `HAS_EXIF_FIXED=False`，LANDSCAPE 降级为软监控（不入硬门、仅记 `trap_fail`/`aes_soft_trap_fail`）。
- **P-2 `is_bw_img` 检测**：iqa 增廉价"HSV-S 均值低 **AND** Lab a*/b* 色度方差低"**双条件**（复用 `paired_metrics` 的 `rgb2lab`，防青橙/褪色胶片/ppr10k 低饱和成片误判为黑白），落新列 `assets.is_bw_img`（与 PRESET 的 `is_bw` 分开）。供**质量与审美的 COLOR 陷阱**取真值。启用前必须在已知黑白/彩色（含低饱和成片）子集核验**假阳率 < 2%** 方置 `ENABLE_T_COLOR/HAS_ISBW_IMG=True`。
- **执行顺序**：三套图像 QA（质量/审美）**必须在 iqa 之后跑**，否则 `width/height/max_face_frac/is_bw_img` 为 NULL 使陷阱静默失效（清洗器对 None 信号 skip 并计 `trap_skipped` 监控）。

### 1.6 跨流程需对齐的常量

- **`FACE_MIN` 口径不一致需对齐**：质量流程取 `0.012`（小脸合影也判"有脸"，配 no_face/FACE_GT），审美流程取 `0.04`（对齐 `GATE.min_face_frac`，决定是否发 `M_moment`/FACE）。二者用途不同 → **命名空间化为 `IMQ_FACE_MIN` 与 `AES_FACE_MIN`**，各按 portrait 池 `max_face_frac` 直方图单独定标，勿共用一个常量。
- **`CONTRA_TOL` 是每流程私有**：质量 `1`、审美 `1`、preset `0`（COH 单对，0 容差）。
- **`is_bw_img` 单列单检测器**：质量与审美共用同一列与同一 §1.5 P-2 检测，勿各写一份。
- **`SCRAMBLE_SEED` 每流程一份**：决定该流程题目的固定乱序置换（满足 §1.1 R2 的间隔约束）；三套各自固定、互不影响，便于独立重洗。
- **κ 闸门统一语义**（§6）：任何题进硬 drop 前须逐题 Cohen's κ≥0.60；0.4≤κ<0.6 仅 review；κ<0.4 重写。对应开关：质量 `KAPPA_PASS{REAL,CLEAN,FACE}`/`DEFECT_HARD_DROP`、preset `PRO_DROP_ENABLED`/`INTENT_DROP_ENABLED`，**默认全 False**（标定前一律 review，不 drop）。

### 1.7 统一 schema / 落库迁移（汇总）

- `llm_qa`：复用，逐题落库 `questionnaire∈{'IMQ','aes','preset'}`、**`item=claim 审计 ID`（如 `REAL_F`，非模型看到的位置码）**、`answer∈{0,1}`、`raw=`模型原始位置码串；**新增列 `probe_id TEXT`**（preset 6 探针同题区分，唯一键 `(asset_id,questionnaire,probe_id,item)`）。
- `assets` 新增列（`db.py` 的 `ADD COLUMN IF NOT EXISTS` 迁移）：质量 `defect_count / qa_reliable / qa_unreliable_reason / qa_contra_count / reduced_anchor`；审美 `merit_count / merit_n / merit_frac / aes_sort_key / aes_keep_vote / aes_reliable / aes_reason / aes_contradiction_count / aes_trap_fail / aes_soft_trap_fail`；共享 `is_bw_img`；preset 探针选取 `mean_luma / highlight_frac / shadow_frac / saturation_mean`。
- 新表：`llm_qa_runs`（图像整组可信度）、`preset_qa_runs`（preset 级投票汇总，`pass_c` 恒 0/1 不写 NULL）；`preset_previews` 增 `probe_reliable / probe_reason / trap_fail`。
- `config.py`：新增 `QUESTIONNAIRE_IMQ / AES_QUESTIONNAIRE / PRESET_QA`（含各自 `POS_MAP` + `SCRAMBLE_SEED`，替换图像侧 `QUESTIONNAIRE_A/B` 与 `QUESTIONNAIRE_C`）+ 三套常量；保留旧列兼容但从 gate 决策移除/降权坍缩的 `B_quality`。

---
## 流程 1：摄影图像质量 QA（技术质量门 — validity + 技术缺陷（含过度后期破坏），不含审美）

### 1.1 目标与数据依据

把"让 35B VLM 直接打 `B_quality` 0–3 分"（全库 98.9% 判 3、与 `musiq` 矛盾、门形同虚设）替换为**纯二元判断题 + F⊕R 正反一致性自检 + 确定性陷阱锚定**的技术质量门。本门只裁决技术 / 有效性质量，审美归流程 2。validity（REAL/CLEAN/INTACT）的正向题为 0 → 候选硬 drop（受 §6 金标 κ 约束）；**6 道独立缺陷题**命中数 = 可区分的 `defect_count(0..6)`，替代坍缩的 `B_quality`。

> **本次修订两点**：① **删除 safety 维**——本语料是合理开源的摄影/调色素材，NSFW/血腥/暴力几乎不触发（S1 yes>99%、S2<1%，退化题、白占题位还易被审计误判失效）；把这两道题位换成更有区分力的 **OVERCOOK（过度后期破坏）缺陷对**，并入 `defect_count`。② **题号反作弊**：题号去语义 + 固定乱序 + 异质谓词 R，见共享约定 §1.1（本流程是其落地范例，§1.3 给出完整 non-portrait 位置码映射）。

数据依据：
- **唯一可锚场景是 `is_portrait_pool`（独立布尔列，带 `max_face_frac` 真值）**；scene 文本标签取自 tag_cache（`ingest.py:72`，非严格二元），与本技术门无关——故删除需求文档的 SCENE_F/R，不依赖 scene 字段。
- **`tad66k`（~800px / ~0.4MP）是结构性低分辨率主池**：UPSC 题只问"放大插值/糊边痕迹"这一可视语义，绝不问"分辨率够不够高"，分辨率绝对值的门交给 `gate.py`（`megapixels/longedge`）。
- **已精修成品也是优质源**：system prompt 显式指示"已调色/已修图不扣分"。**OVERCOOK 只针对*破坏性*痕迹**（halo/死白涂抹/塑料感/HDR 脏渲染/色阶断裂），与"看起来已调色"严格区分，不与该前提冲突。
- **`width/height` 在 index 里是 NULL（`config.py:61`），仅 `iqa.py` 解码后写入** → llm_qa 必须在 iqa 之后跑。
- **图像 `is_bw` 未计算**（仅 PRESET 由 `ingest.py:125` 填），故 T_COLOR 默认关闭。
- 可用确定性信号（精确列名）：`is_portrait_pool, width, height, megapixels, musiq, clipiqa, niqe, brisque, sharpness, noise_sigma, max_face_frac`。

**本次评审折叠的修订（一句话列点）：**
- 显式声明 **llm_qa 必须在 iqa 之后**，否则 `width/height` NULL 使 LANDSCAPE 静默失效；清洗器统计 skip 比例落库监控，全 skip 的样本标 `reduced_anchor`（折叠 blocker①）。
- T_COLOR 补 `is_bw_img` 改用 **"S 均值低 AND lab a*/b* 色度方差低"双条件**，启用前在已知子集核验假阳率 < 2%（折叠 major②）。
- INTACT 硬对降为 **soft + 容差**，并把 INTACT_R 重写为 INTACT_F 的单一否定命题去掉异质 OR 项（折叠 major③）。
- `no_usable_face` 从硬 drop **降为 review**，直到 FACE 题过金标 κ≥0.60（折叠 major④）。
- `defect_count` 硬 drop 加 **`DEFECT_HARD_DROP`（默认 False）**开关，标定前只 review（折叠 minor⑥）。
- 验收显式声明 **validity 题（A*）为硬门、豁免 [3%,97%] 退化判定**，可信度由 F⊕R 矛盾率≈0 + 金标 κ 保证（折叠 minor⑦）。
- gate 取交集写成确定性 **`reconcile`** 函数：仅 reliable 且 verdict=keep 但 NR-IQA 软尾命中 → 降 review，reason=`iqa_conflict`（折叠 minor⑧）。
- COMP / UPSC / OVERCOOK **按池分层验 yes 率**，防 tad66k 池间偏置抬高 `defect_count`（折叠 degeneracy 风险⑨）。

### 1.2 维度

> 注：下表"维度字母"仅为**内部审计**（`POS_MAP` 里的 claim 分组）；模型实际只看到无语义的两位**位置码**（§1.1 R1、§1.3 范例）。

| 维度(内部) | 名称 | 角色 | 说明 |
|---|---|---|---|
| **validity** | 有效性 | 候选硬 drop（受 κ 约束） | 真实自然照片 / 无叠加污染 / 内容完整非碎片测试图。 |
| **defect** | 技术缺陷 | `defect_count` 加项 | 失焦 / 噪点 / 压缩 / 上采样 / 曝光死区 / **过度后期破坏(OVERCOOK)** 共 6 类，软对、不单独 drop。 |
| **subject** | 主体 | review 信号 | 弱主体→review；FACE（仅 portrait）。 |
| **trap** | 陷阱/锚点 | 可信度门（确定性真值） | ANCHOR(恒1) / LANDSCAPE / FACE_GT / COLOR(默认关)。 |

> `defect_count = SHARP_R + NOISE_R + COMP_R + UPSC_R + EXPO_R + OVERCOOK_R`，取值域 **0..6**。

### 1.3 题库（内部 `POS_MAP` / 审计视图）

> **下表是清洗器内部的 claim 定义（审计用，带可读的 claim 审计 ID）；模型永远看不到它。** 模型看到的是据本表 + `SCRAMBLE_SEED` 生成的**乱序两位位置码**（见表后"展示乱序"）。`item` 落库用 claim 审计 ID（如 `REAL_F`）。

base claim 集合（恒出）：`REAL/CLEAN/INTACT(F,R)` + `SHARP/NOISE/COMP/UPSC/EXPO/OVERCOOK(F,R)` + `SUBJ(F,R)` + `ANCHOR/LANDSCAPE`；`is_portrait_pool` 时并入 `FACE(F,R)/FACE_GT`；`ENABLE_T_COLOR 且 is_bw_img 非空`时并入 `COLOR`。

| claim 审计 ID | 维度 | 极性 | 配对(硬/软) | 题面（中文，最终版） | 好图答案 | 真值源 / portrait_only |
|---|---|---|---|---|---|---|
| `REAL_F` | validity | F | REAL · 硬 | 这是相机拍摄或扫描得到的真实自然照片。 | 1 | vlm |
| `REAL_R` | validity | R | REAL · 硬 | （异质谓词）画面带有非自然来源的破绽：界面/截图元素（状态栏、光标、菜单栏、播放控件）、印刷海报排版、3D/CG 渲染感，或 AI 生成的异常细节（畸形手指/文字/拼接）。 | 0 | vlm |
| `CLEAN_F` | validity | F | CLEAN · 硬 | 画面干净，没有叠加在照片上的水印、台标/平台 logo、文字条、加框、二维码或拼贴分割线。 | 1 | vlm |
| `CLEAN_R` | validity | R | CLEAN · 硬 | （异质谓词）画面上能找到具体的叠加物：水印/台标、文字条/字幕、明显边框、二维码，或多图拼贴的分割线。 | 0 | vlm |
| `INTACT_F` | validity | F | INTACT · **软** | 这是一张内容完整的正常照片，不是被极端裁切的图像碎片、纯色块或测试图/色卡。 | 1 | vlm |
| `INTACT_R` | validity | R | INTACT · **软** | 上一句不成立：这张图**不**是内容完整的正常照片（即它是极端裁切碎片、纯色块或测试图/色卡）。 | 0 | vlm（INTACT_R 为 INTACT_F 的单一否定；配对已被乱序隐藏，故保留严格否定） |
| `SHARP_F` | defect | F | SHARP · 软 | 画面主体清晰、对焦准确。 | 1 | vlm |
| `SHARP_R` | defect | R | SHARP · 软 | 画面主体失焦、脱焦或整体发虚，看不清细节。 | 0 | vlm |
| `NOISE_F` | defect | F | NOISE · 软 | 画面纯净，没有明显的噪点或颗粒感。 | 1 | vlm |
| `NOISE_R` | defect | R | NOISE · 软 | 画面有明显的噪点、彩色噪声或颗粒感（尤其暗部/天空）。 | 0 | vlm |
| `COMP_F` | defect | F | COMP · 软 | 画面没有可见的 JPEG 压缩方块、阶梯状色带或块状色噪。 | 1 | vlm |
| `COMP_R` | defect | R | COMP · 软 | 画面有可见的 JPEG 压缩方块、平滑区域的阶梯色带或块状色噪。 | 0 | vlm |
| `UPSC_F` | defect | F | UPSC · 软 | 画面看起来是原生清晰度，没有被放大插值后的糊边、锯齿或过度平滑塑料感。（**注：原生分辨率低/小图本身不算放大痕迹**） | 1 | vlm |
| `UPSC_R` | defect | R | UPSC · 软 | 画面有被低分辨率放大插值的痕迹：边缘糊化、锯齿或过度平滑的塑料感。 | 0 | vlm |
| `EXPO_F` | defect | F | EXPO · 软 | 曝光基本正常，没有大面积纯白死白或纯黑死黑的丢失细节区域。 | 1 | vlm |
| `EXPO_R` | defect | R | EXPO · 软 | 画面有大面积过曝纯白或欠曝纯黑的死区，完全丢失细节。 | 0 | vlm |
| `OVERCOOK_F` | defect | F | OVERCOOK · 软 | 后期克制自然：没有过度处理的破坏痕迹（无 halo 描边/亮边、无高光涂抹丢层次、无塑料感过度磨皮、无 HDR 脏渲染、无色阶断裂）。 | 1 | vlm |
| `OVERCOOK_R` | defect | R | OVERCOOK · 软 | （异质谓词）画面有过度后期的破坏痕迹：高反差边缘出现 halo/亮边、皮肤或天空被过度平滑成塑料感、高光被涂抹丢层次，或出现 HDR 脏渲染/色调分离(posterize)。 | 0 | vlm |
| `SUBJ_F` | subject | F | SUBJ · 软 | 画面有明确的主体或清晰的构图意图。 | 1 | vlm |
| `SUBJ_R` | subject | R | SUBJ · 软 | 画面空洞、没有明确主体，或基本只是纯文字/截图式内容。 | 0 | vlm |
| `FACE_F` | subject | F | FACE · 软 | 画面中有清晰、可用的人脸（五官清楚、不模糊、不塑料）。 | 1 | vlm / **portrait_only** |
| `FACE_R` | subject | R | FACE · 软 | 画面中没有清晰可用的人脸（无人脸，或人脸严重模糊/塑料感/被遮挡）。 | 0 | vlm / **portrait_only** |
| `ANCHOR` | trap | 锚点 | — | 上方确实有一张可见的图片。 | 1 | **恒为 1**（≠1 即模型未看图/故障） |
| `LANDSCAPE` | trap | 陷阱 | — | 这张图是横向构图（宽度大于高度）。 | 占位 | `width > height`（清洗器按列算；仅 iqa 后有值；**见 §1.5 前置 P-1 EXIF**） |
| `FACE_GT` | trap | 陷阱 | — | 画面中能看到人的脸。 | 占位 | `max_face_frac > IMQ_FACE_MIN`（仅 portrait；cv2 Haar） / **portrait_only** |
| `COLOR` | trap | 陷阱 | — | 这是一张彩色照片（不是黑白/单色照片）。 | 占位 | `not is_bw_img`（默认关闭；启用前补双条件并核验假阳率 < 2%，见 §1.5 P-2） |

**展示乱序（位置码 → claim，non-portrait 范例，N=22）。** 由 `SCRAMBLE_SEED` 固定生成，约束：每对 F/R 间隔 ≥ ⌈22/3⌉=8 且不相邻、维度交错；模型只看到这 22 个位置码与对应题面（无维度/极性标注）：

```
01 REAL_F   02 SHARP_F  03 NOISE_F  04 ANCHOR   05 CLEAN_F
06 COMP_F   07 INTACT_F 08 UPSC_F   09 EXPO_F   10 SUBJ_F
11 OVERCOOK_F  12 REAL_R  13 NOISE_R  14 LANDSCAPE  15 SHARP_R
16 COMP_R   17 CLEAN_R  18 UPSC_R   19 INTACT_R 20 OVERCOOK_R
21 EXPO_R   22 SUBJ_R
```
> 校验：REAL 01↔12(隔11)、SHARP 02↔15(13)、NOISE 03↔13(10)、CLEAN 05↔17(12)、COMP 06↔16(10)、INTACT 07↔19(12)、UPSC 08↔18(10)、EXPO 09↔21(12)、SUBJ 10↔22(12)、OVERCOOK 11↔20(9)，全部 ≥8 且不相邻；ANCHOR=04、LANDSCAPE=14 为单题陷阱。

**portrait 范例（N=25，加 `FACE_F/FACE_R/FACE_GT`，间隔约束 ≥⌈25/3⌉=9）：**

```
01 REAL_F   02 SHARP_F  03 NOISE_F  04 ANCHOR   05 CLEAN_F
06 COMP_F   07 INTACT_F 08 UPSC_F   09 FACE_GT  10 EXPO_F
11 SUBJ_F   12 OVERCOOK_F 13 FACE_F 14 LANDSCAPE 15 REAL_R
16 NOISE_R  17 SHARP_R  18 CLEAN_R  19 COMP_R   20 INTACT_R
21 UPSC_R   22 EXPO_R   23 SUBJ_R   24 OVERCOOK_R 25 FACE_R
```
> 校验：REAL 01↔15(14)、SHARP 02↔17(15)、NOISE 03↔16(13)、CLEAN 05↔18(13)、COMP 06↔19(13)、INTACT 07↔20(13)、UPSC 08↔21(13)、EXPO 10↔22(12)、SUBJ 11↔23(12)、OVERCOOK 12↔24(12)、FACE 13↔25(12)，全部 ≥9 且不相邻；ANCHOR=04/FACE_GT=09/LANDSCAPE=14 为单题陷阱。启用 COLOR（默认关）按各自 `SCRAMBLE_SEED` 再插入一道 COLOR 单题陷阱 → N=23/26。

### 1.4 System prompt（固定可缓存）

```
你是严格的照片技术质检员。只依据所给这一张图片作答；不解释、不推理、不输出多余文字。
你只判断"技术与有效性质量"，不评价美感、风格或调色好坏。注意：已精修/已调色的成片是正常照片，不要因为"看起来已调色/已修图"就判为非自然或扣画质——只看真实技术缺陷与破坏性痕迹。原生分辨率低的小图本身不是"放大插值痕迹"。
下面是若干二元判断题，每题有一个两位题号（如 03）。逐题判断该陈述对这张图是否成立：成立=1，不成立=0。
【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这张图独立判断，不要根据别的题的答案来推断本题。每题必须作答，拿不准时给最可能的判断，不得留空、不得答"不确定"。
题目（按题号）：
01：这是相机拍摄或扫描得到的真实自然照片。
02：画面主体清晰、对焦准确。
…（按 §1.3 展示乱序，逐题列出 01..N 的题面，不带任何维度/正反标注）…
作答格式：每题输出"题号紧跟答案"（03 成立写 031，不成立写 030），各题用一个空格分隔，按题号从小到大，每题号只出现一次。只输出这一串，不要其它任何内容。
```

### 1.5 输出格式（位置码锚定）

每题输出"位置码紧跟答案"：两位位置码 + 答案（1/0）紧贴，题间单空格，按位置码升序，每码恰一次。

real 示例串（non-portrait，一张合格的彩色横向成片，按 §1.3 展示乱序，好图：F 全 1、R/缺陷全 0、ANCHOR=1、LANDSCAPE=1）：
```
011 021 031 041 051 061 071 081 091 101 111 120 130 141 150 160 170 180 190 200 210 220
```
> 对照展示乱序：位置 12=REAL_R→0、15=SHARP_R→0、19=INTACT_R→0、20=OVERCOOK_R→0…（R 题/缺陷题命中=1 才表示该缺陷存在）；14=LANDSCAPE→1（横图）。清洗器按 `IMQ_POS_MAP` 把每个位置码还原成 claim 后判 F⊕R。

解析正则：`re.findall(r'(\d{2})([01])', s)` → `dict{pos:bit}`。解析前 `s = raw.strip()`，容忍多余空格/换行。

### 1.6 清洗器（有序门，可直接转 Python）

```python
def clean(raw, asset, pos_map, cfg):
    """pos_map: {位置码 -> (claim, polarity)}; asset 列: is_portrait_pool, width, height, max_face_frac, is_bw_img"""
    out = {"reliable": True, "reason": None, "contradiction_count": 0,
           "trap_fail": None, "trap_skipped": 0, "reduced_anchor": False,
           "defect_count": None, "reask_count": 0, "raw": raw}
    expected = set(pos_map)                      # 本次应发的位置码集合(随 portrait/COLOR 动态)

    # ===== 门 1: parse_gate 解析门 =====
    pairs = re.findall(r'(\d{2})([01])', (raw or "").strip())
    pos = [p[0] for p in pairs]
    bit = {p[0]: int(p[1]) for p in pairs}
    if not pairs:                       return _fail(out, "parse:empty")
    if len(pos) != len(set(pos)):       return _fail(out, "parse:dup")
    if set(pos) - expected:             return _fail(out, "parse:unknown")
    if expected - set(pos):             return _fail(out, "parse:missing")

    # ===== 门 1.5: 还原 位置码 → claim/极性 =====
    A = {}                                       # A[claim][F|R] = bit ; A[trapclaim]=bit
    for p, b in bit.items():
        claim, pol = pos_map[p]
        if pol in ("trap", "anchor"): A[claim] = b
        else:                         A.setdefault(claim, {})[pol] = b

    # ===== 门 2: anchor_gate 锚点门 =====
    if A.get("ANCHOR") != 1:           return _fail(out, "anchor")

    # ===== 门 3: trap_gate 确定性陷阱门 =====
    w, h = asset["width"], asset["height"]
    if w is None or h is None:         out["trap_skipped"] += 1
    elif w != h:
        if A["LANDSCAPE"] != (1 if w > h else 0):  return _fail(out, "trap:landscape")
    if asset["is_portrait_pool"]:
        mff = asset["max_face_frac"]
        if mff is None:                out["trap_skipped"] += 1
        elif A["FACE_GT"] != (1 if mff > cfg.IMQ_FACE_MIN else 0):
            return _fail(out, "trap:face")
    if "COLOR" in expected_claims(pos_map):
        if A["COLOR"] != (0 if asset["is_bw_img"] else 1):  return _fail(out, "trap:color")
    if out["trap_skipped"] and not _has_live_trap(pos_map, asset):
        out["reduced_anchor"] = True

    # ===== 门 4: contradiction_gate 正反矛盾门（按 claim 判 F⊕R）=====
    hard = ["REAL", "CLEAN"]                      # INTACT 已降 soft
    soft = ["INTACT","SHARP","NOISE","COMP","UPSC","EXPO","OVERCOOK","SUBJ"]
    if asset["is_portrait_pool"]: soft.append("FACE")
    hc = sum(1 for c in hard if A[c]["F"] + A[c]["R"] != 1)
    sc = sum(1 for c in soft if A[c]["F"] + A[c]["R"] != 1)
    out["contradiction_count"] = hc + sc
    if hc > 0:                         return _fail(out, "contradiction_hard", A)
    if sc > cfg.CONTRA_TOL:            return _fail(out, "contradiction_soft", A)

    # ===== 门 5: verdict_gate 裁定门 =====
    out["A"] = A
    out["defect_count"] = (A["SHARP"]["R"] + A["NOISE"]["R"] + A["COMP"]["R"]
                           + A["UPSC"]["R"] + A["EXPO"]["R"] + A["OVERCOOK"]["R"])   # 0..6
    return out

def _fail(out, reason, A=None):
    out["reliable"] = False; out["reason"] = reason
    if reason.startswith("trap:"): out["trap_fail"] = reason.split(":",1)[1]
    if A is not None: out["A"] = A
    out["defect_count"] = None
    return out
```

重问策略：任一门 `reliable=False` 时按 `REASK_MAX=1` 重问一次（temp=0、**重洗题序**）；仍 `False` → `auto_verdict='review'`, `status='qa_unreliable'`，**永不进自动 keep/drop**。

### 1.7 可区分信号与判级映射

**`defect_count` 定义**：`SHARP_R+NOISE_R+COMP_R+UPSC_R+EXPO_R+OVERCOOK_R`，取值域 **0..6** 整数。每加项是一道独立反向缺陷题命中（=1 表示该缺陷存在）。**仅当 `reliable=True` 才有效，不可信样本 `defect_count = NULL`**。6 道缺陷相互独立且各受 `[3%,97%]` 约束，命中数天然在 0..6 上有分布，替代坍缩的 `B_quality`。`weak_subject = (SUBJ_F==0)` 为辅助弱主体信号（→review）。

```python
def verdict(A, is_portrait, max_face_frac, defect_count, cfg):
    # 仅在 reliable=True 时调用；否则一律 review。
    # 1) 有效性候选硬 drop —— 仅 κ≥DROP_KAPPA_MIN 的题进硬 drop，否则降 review。
    #    REAL/CLEAN 受 §6 κ 约束；INTACT 已降 soft 不单独硬 drop。（safety 维已删除）
    for claim, kpass in (("REAL", cfg.KAPPA_PASS["REAL"]),
                         ("CLEAN", cfg.KAPPA_PASS["CLEAN"])):
        if A[claim]["F"] == 0:
            return ("drop", f"invalid:{claim}") if kpass else ("review", f"invalid:{claim}:unkappa")

    # 2) portrait 无脸 —— 降为 review，FACE 过 κ 前不进硬 drop。
    if is_portrait and A.get("FACE", {}).get("F", 1) == 0:
        if cfg.KAPPA_PASS["FACE"] and max_face_frac is not None and max_face_frac > cfg.IMQ_FACE_MIN:
            return ("drop", "no_usable_face")
        return ("review", "face_uncertain")

    # 3) 可区分缺陷计数 —— DEFECT_HARD_DROP 前缺陷只 review。
    weak_subject = (A["SUBJ"]["F"] == 0)
    if defect_count >= cfg.DEF_DROP and cfg.DEFECT_HARD_DROP:
        return ("drop", "quality_defects")
    if defect_count >= cfg.DEF_REVIEW or weak_subject or defect_count >= cfg.DEF_DROP:
        return ("review", "quality_or_subject")
    return ("keep", None)


def reconcile(ans_verdict, reason, iqa, cfg):
    # 与 gate.py 确定性信号取交集。drop 方向 gate 先行（llm_qa.py:308 已排除 auto_verdict='drop'）；
    # 此处只处理 keep→冲突降级。
    v, r = ans_verdict, reason
    if v == "keep":
        if (iqa["musiq"] is not None and iqa["musiq"] < cfg.MUSIQ_DROP_BELOW) \
           or (iqa["niqe"] is not None and iqa["niqe"] > cfg.NIQE_CONFLICT_ABOVE) \
           or (iqa["noise_sigma"] is not None and iqa["noise_sigma"] > cfg.NOISE_SIGMA_DROP_ABOVE):
            return ("review", "iqa_conflict")
    return (v, r)
```

`auto_verdict` 永不破坏性删除，仅建议；Web UI 人工终裁。

### 1.8 config 常量

| 名 | 默认 | 含义 |
|---|---|---|
| `IMQ_QUESTIONNAIRE_TAG` | `"IMQ"` | `llm_qa.questionnaire` 取值，区分图像技术质量门。 |
| `IMQ_POS_MAP` / `IMQ_SCRAMBLE_SEED` | 见 §1.3 | 位置码→claim 固定映射 + 生成乱序的种子（满足 F/R 间隔约束）；模型侧单一事实来源。 |
| `IMQ_FACE_MIN` | `0.012` | FACE_GT / no_face 的 `max_face_frac` 阈值；保守取小避免小脸合影误判无脸（与审美 `AES_FACE_MIN` 分开命名）。 |
| `DEF_DROP` | `3` | `defect_count ≥` 此值 → drop（仅 `DEFECT_HARD_DROP=True` 时）；0..6 区间，标定后再调。 |
| `DEF_REVIEW` | `2` | `defect_count ≥` 此值（且 < DEF_DROP）或 weak_subject → review。 |
| `CONTRA_TOL` | `1` | 软对允许的最大矛盾数（INTACT 现也计入软对）。 |
| `REASK_MAX` | `1` | 不可信重问上限（重问重洗题序）；仍不可信 → review。 |
| `VLLM_IMAGE_LONGEDGE` | `768` | 送判图片长边下采样（沿用 config:47）；陷阱失败率高时上调。 |
| `DEFECT_HARD_DROP` | `False` | 缺陷计数是否准硬 drop；金标证明缺陷组合 κ≥0.60 才置 True。 |
| `ENABLE_T_COLOR` | `False` | 是否启用 COLOR 陷阱；需先补 `is_bw_img` 双条件并核验假阳率 < 2%。 |
| `ISBW_SAT_MEAN_MAX` | `10.0` | `is_bw_img` 条件一：HSV 的 S 通道（0..255）均值 < 此阈。 |
| `ISBW_AB_VAR_MAX` | `35.0` | `is_bw_img` 条件二：lab a*/b* 色度方差 < 此阈（双条件均满足才判黑白；`rgb2lab` 复用 `paired_metrics`）。 |
| `MUSIQ_DROP_BELOW` | `30.0` | reconcile 软尾：keep 但 `musiq <` 此值 → 降 review（对齐 gate `musiq_drop_below`）。 |
| `NIQE_CONFLICT_ABOVE` | `9.0` | reconcile 软尾：`niqe >` 此值视为冲突。 |
| `NOISE_SIGMA_DROP_ABOVE` | `18.0` | reconcile 软尾：`noise_sigma >` 此值视为冲突（对齐 gate）。 |
| `DROP_KAPPA_MIN` | `0.60` | §6：逐题 Cohen's κ ≥ 此值才允许该题进硬 drop；0.4≤κ<0.6 仅 review；κ<0.4 重写。 |
| `KAPPA_PASS` | `{REAL:F,CLEAN:F,FACE:F}` | 各题是否已过 κ 标定（标定前全 False，对应题降 review）。**safety 已删除，不在此列。** |

### 1.9 可区分性验收指标（§7）

| 指标 | 目标 |
|---|---|
| 每道缺陷题（`SHARP/NOISE/COMP/UPSC/EXPO/OVERCOOK` 的 F&R）全库 yes 率 | 落 `[3%,97%]`；超出即退化需重写。 |
| `defect_count(0..6)` 分布 | 非单点堆积；keep 集中 0/1、review 在 2、drop 在 ≥3 上可分离。 |
| **COMP/UPSC/OVERCOOK 按池分层 yes 率** | 分 tad66k / 其它池单独验；UPSC 在 ~800px 池不得系统性抬高（防"原生小图"误读为"放大痕迹"）；OVERCOOK 防把"风格化成片"误判破坏。 |
| 首问不可信率 | 期望 < 15%；过高 = 题面/prompt 需改。 |
| 硬正反矛盾率（REAL/CLEAN F⊕R） | ≈0。 |
| **INTACT 软对违反率** | 监控；持续偏高则进一步收紧 INTACT_R 题面。 |
| 陷阱失败率 LANDSCAPE | < 3%；同时监控 skip 比例（`trap_skipped`），skip 高 = llm_qa 跑在 iqa 前。 |
| 陷阱失败率 FACE_GT（portrait） | < 8%，与 Haar `max_face_frac` 高一致。 |
| **COLOR 假阳率**（启用时） | `is_bw_img` 在已知彩色子集（含青橙/褪色/ppr10k 低饱和成片）假阳 < 2% 为硬前置门；之后失败率 < 2%。 |
| **作弊检测（反作弊新增）** | 监控 F⊕R 软对违反率随题序乱化是否上升——若隐藏配对后矛盾率显著升高，说明此前的"低矛盾"部分来自逻辑反推而非看图；持续逼近 0 才说明模型在认真看图。 |
| tad66k 主池 keep 率 | 不被结构性低分辨率系统性判 drop。 |
| 硬 drop 题逐题 Cohen's κ（300–500 金标） | REAL/CLEAN/FACE κ ≥ 0.60 才进硬 drop，否则降 review。 |

> **validity 豁免声明**：REAL/CLEAN/INTACT 为有效性硬门，本语料下 yes 率天然偏态（预计 REAL_F > 97%、INTACT_F > 97%），**豁免 `[3%,97%]` 退化判定**；其可信度不靠 yes 率分布，而由 **F⊕R 硬对矛盾率≈0 + 金标 κ** 保证。审计 degeneracy 时不得把 validity 高 yes 率误标为失效题。（safety 维已删除，不再出现 S* 偏态题。）

### 1.10 schema / 落库改动

- **`config.py`**：新增 `QUESTIONNAIRE_IMQ`（`POS_MAP`：claim 审计 ID→题面+polarity+pair+portrait_only）+ `IMQ_SCRAMBLE_SEED` 替代 `QUESTIONNAIRE_A/B` 在图像侧的用途；新增 §1.8 全部常量；`IMQ_QUESTIONNAIRE_TAG='IMQ'`。**删除 safety 题、新增 OVERCOOK 题。**
- **`iqa.py`**：补廉价 `is_bw_img` 检测（HSV S 均值 < `ISBW_SAT_MEAN_MAX` **AND** `rgb2lab` 的 a*/b* 色度方差 < `ISBW_AB_VAR_MAX`），写入 `assets.is_bw_img`（新列，与 PRESET 的 `is_bw` 分开）；其余信号已算。**强约束：llm_qa 必须在 iqa 之后跑。**
- **`llm_qa.py`**：把现有 `_yn/_allyes/pass_b` 推导替换为 §1.6 的 5 门清洗器（含位置码→claim 还原）+ 重问逻辑；解析 `re.findall(r'(\d{2})([01])', s)`。逐题落库 `llm_qa(questionnaire='IMQ', item=claim 审计 ID, answer∈{0,1})`，`raw` 存模型原始位置码串。
- **新表 `llm_qa_runs`**（或并入 assets）：`asset_id, questionnaire, reliable, reason, contradiction_count, trap_fail, trap_skipped, reduced_anchor, reask_count, defect_count, raw, model, run_id`。
- **`assets` 新列**：`defect_count INTEGER`、`qa_reliable INTEGER`、`qa_unreliable_reason TEXT`、`qa_contra_count INTEGER`、`is_bw_img INTEGER`、`reduced_anchor INTEGER`。保留 `pass_a/pass_b`、`b_quality/b_comp` 兼容，但从 gate 决策移除/降权。
- **`webapp/app.py`**：Web UI 经 `POS_MAP` 把位置码翻成 claim 审计 ID 展示 `reliable/reason/defect_count/contradiction_count/trap_skipped/reduced_anchor`；不再展示坍缩的 `B_quality`。

### 1.11 遗留问题

- **金标 κ 标定**：REAL/CLEAN/FACE 各题在 300–500 金标上跑 Cohen's κ；κ≥0.60 才把对应 `KAPPA_PASS` 置 True，否则维持 review。`no_usable_face`、`DEFECT_HARD_DROP` 均待此标定。
- **OVERCOOK 题面定标**：`OVERCOOK_R` 与"已精修成品是优质源"的边界需在 pilot 上核——监控 OVERCOOK_R 在已知优质调色成片（GREYSKY/ppr10k）上的假阳率，过高则收紧到"明显 halo/posterize/塑料"等最硬破坏项。
- **`is_bw_img` 双条件阈值**：`ISBW_SAT_MEAN_MAX=10` + `ISBW_AB_VAR_MAX=35` 需在已知子集核验假阳率 < 2% 后才置 `ENABLE_T_COLOR=True`。
- **`IMQ_FACE_MIN=0.012`**：先跑 portrait 池 `max_face_frac` 直方图；确认池内存在足量 `< IMQ_FACE_MIN` 样本方使 FACE_GT 有区分力，并在金标上验 `no_usable_face` 精确率。
- **执行顺序**：监控 `trap_skipped` / `reduced_anchor` 比例，确认 llm_qa 确在 iqa 之后跑。
- **乱序种子**：`IMQ_SCRAMBLE_SEED` 固定后不要随意改（改了即 prompt 变、prefix-cache 失效、需重跑）；portrait/COLOR 各态各自一份固定序。
- **COMP/UPSC 池间偏置**：按池分层 yes 率，必要时放宽 COMP_R 题面或强化 UPSC 的"小图≠放大痕迹" prompt 指引。

---
## 流程 2：摄影图像审美 QA（通用结构角度）— 软信号 merit_frac（排序-only，永不改判级）

> flow_id: `flow2_aesthetic`

### 2.1 目标与数据依据（软信号、永不 drop）

**目标。** 在任意题材 / 分辨率 / 已精修成品的源图上，用二元判断题 + F⊕R 一致性自检，抽取**跨风格通用的结构性审美优点**，并产出一个可信、可区分的连续软信号 `merit_frac`。该信号**仅**用于 keep 优先级**排序**（一个独立排序键，不进任何判级 good 计数器），**永不 drop、永不把任何样本从 review/None 翻成 keep**。审美无外部金标，故只锚定**模型一定看得见的几何 / 存在事实**做硬陷阱，抓 acquiescence（敷衍附和）与幻觉；一切内容判断由 VLM 做，但只有过全部清洗门（判为可信）后才采信。

它是**独立的第二次 vLLM 调用**，与质量门（流程 1）完全分开，题库固定、可命中 prefix-cache。

**数据依据（据真实 schema + `iqa.py`/`gate.py` 源码核验，而非需求文档示例）。**

1. **scene 标签退化 → 完全不依赖 scene。** `scene` 实际仅 `any`（≈78%）/ `portrait`（≈22%），细粒度场景退化。故审美题**完全不依赖 scene 标签**，全部写成跨题材通用的结构观察；portrait 仅追加一对人脸表情题 `M_moment`（M3/M4）与人脸存在陷阱 T4，且二者发题口径统一为 `max_face_frac>AES_FACE_MIN`（实有人脸），而非仅 `is_portrait_pool`——避免宠物 / 物品误入 portrait 池时被问神态而产生稳定假矛盾（评审 major-5）。
2. **preset style 不涉及。** `preset style=None` 在图像流程不涉及。
3. **tad66k 已策展成品池的双重退化风险。** 最大池 tad66k 是 web-scraped ~800px / ~0.4MP 的已策展成品图，存在两重风险：① 不因分辨率 / 已调色扣审美分（prompt 明确），技术缺陷归流程 1 正交；② 已策展 + "不扣分"会让朴素正向题 yes 率 >97%、merit 向满分坍缩，沦为第二个 `B_quality`。故本版**把每个软对的题面从"有没有优点"改写为"是否达到明显高于随手拍的较高水准"**以拉开判别面（评审 major-3），并把 200 图 pilot 的 per-题 yes 率 + `merit_frac` 直方图设为**全库重跑前的硬 gate**。
4. **EXIF 朝向隐患（blocker）。** 审美无确定性金标，唯一可锚的确定性信号是模型一定看得见的几何 / 存在事实；但核验 `iqa.py` L61-64 的 decode **无 `ImageOps.exif_transpose`**：写库的 `width`/`height` 是原始像素栅格朝向。手机 / 相机竖拍 portrait 的 JPEG 常带 EXIF orientation=6/8 → 库里 `width>height`（横），而 VLM 收到的是已旋正的竖图 → 把 `T_LANDSCAPE` 作硬陷阱会在 portrait 池系统性把诚实回答打成不可信（评审 blocker）。本版采纳修复 (a)：`schema_changes` 要求 iqa decode 后做 `exif_transpose`，重算 `width`/`height`/`megapixels`/`longedge` 使其 = 显示朝向（与送审图、与 musiq 等一致），受影响图重跑。在该修复落地前 `HAS_EXIF_FIXED=False`，T2 **降级为软监控信号**（不入硬陷阱门、不置 `reliable=False`，仅记 `trap_fail` 供观测）。**（与流程 1 共享，见 §1.5 P-1）**
5. **is_bw_img 缺列（major-4）。** `is_bw` 列在 schema 但**图像侧未计算**（`ingest.py:125` 仅 PRESET 填）→ 颜色陷阱 T3 当前无真值源。本版把 `is_bw_img` 检测从"可选"升级为**非 portrait 池抗 acquiescence 的必做依赖**：非 portrait 在 EXIF 修复前若仅 T2 一道又因 EXIF 降级，硬陷阱会归零、盲猜全过概率 = 1，merit 可被"全 1 / 全 0"敷衍刷满；补 `is_bw_img` 后 T3 入硬门，给非 portrait 提供至少一道不依赖朝向的硬陷阱。**（与流程 1 共享同一列与同一检测，见 §1.5 P-2）**
6. **极性打散的诚实性软陷阱（major-4 第 2 点）。** 评审指出"全 1 / 全 0 敷衍 → 所有 F⊕R 软对 viol=0"的盲区：本版额外引入一对**极性打散的诚实性软陷阱** H1/H2（措辞一正一反，但好图都应一真一假），使天然"全 1 / 全 0"必然触发其矛盾，补强 acquiescence 检出。
7. **与流程 1 正交。** 流程 1 管真实 / 安全 / 技术缺陷（可硬 drop）；本流程只问构图 / 光影 / 色彩 / 层次 / 主体 / 整洁这些结构性审美优点在不在（软信号，永不 drop、永不改判级）。

### 2.2 维度

| key | 维度 | 理由 |
| --- | --- | --- |
| **K** | 构图与取景 / 画面平衡 | 跨风格通用的结构骨架：主体落点 / 留白 / 线条引导 / 边缘干净 / 横竖比是否经过经营，还是随手抓拍的失衡 / 歪斜 / 贴边切割。题面已从"有没有构图"提升为"是否明显优于随手拍"以拉开判别面（评审 major-3），不依赖题材或 scene 标签。 |
| **L** | 光线质量与影调 | 讲究的光（方向性 / 层次 / 明暗张力）是结构性审美核心，与技术曝光（死白死黑 → 流程 1）正交：这里问光线是否塑造了立体与氛围、影调过渡是否丰富，而非有无过曝缺陷。 |
| **C** | 色彩协调 | 配色是否有自觉的整体调性（呼应或克制），而非脏浊偏色；R 收敛到单一最具判别力的"严重偏色 / 发脏"（评审 minor）。BW-aware：黑白 / 单色图改问"影调是否统一克制"而非"颜色"，由 `is_bw_img` 决定 C 维题面分支（评审 missing_pairs C 维），避免黑白图在颜色题上产生不可预期取值。 |
| **D** | 空间层次 / 景深 / 立体感 / 主体背景分离 | 前中后景层次、虚实景深、主体与背景分离，是"立体不平板"的结构来源；与 K（怎么摆）、M（视觉中心）互补但角度不同（D 看纵深与分离）。低分辨率 tad66k 上模型对景深判别弱，题面强调"明显的"纵深 / 分离以避免"看不清 → 保守答优点"推高 yes 率（评审 degeneracy D3）。 |
| **M** | 主体显著性 / 视觉中心 / 决定性瞬间或表情 | 是否有强视觉中心、主体抓住注意力；`M_moment`（M3/M4）仅在确有人脸（`max_face_frac>AES_FACE_MIN`）时追加"神态 / 瞬间到位"，与 K（怎么摆）分离。发题口径与 T4 统一以消除无脸图的稳定假矛盾（评审 major-5）。 |
| **N** | 整洁度（无杂乱干扰 / 无割裂元素） | 画面是否干净、无抢眼干扰物 / 割裂构图的突兀元素。与流程 1 的"叠加物（水印 / 台标）"不同——N 评**自然画面内的视觉杂乱**这一审美维度，非有效性污染。 |
| **H** | 诚实性软陷阱（极性打散，抓全 1 / 全 0 敷衍） | 评审 major-4 指出：一个全 F=1 / 全 R=0 的 acquiescence 模型会让所有 F⊕R 软对 viol=0、锚点 T1 也答 1，仅需 T2 蒙对即整份 reliable 且 merit 满分。H 维引入一对**措辞一正一反、但好图应答为 (H1=1, H2=0) 的诚实性配对**：天然"全 1"会让本应为 0 的那项答错 → 暴露；"全 0"同理。它不是 F⊕R 对，而是"好图固定取值对"，清洗器据固定期望比对（见 `H_honesty` 门）。 |
| **T** | 陷阱 / 锚点（确定性一致性自检） | 审美无外部金标，用模型一定看得见的几何 / 存在事实做硬陷阱。`T_ANCHOR`（T1，恒 1）抓未看图 / 故障；`T_FACE`（T4，portrait 实有脸）抓存在性幻觉；`T_COLOR`（T3，`not is_bw_img`）在 `is_bw_img` 补齐后给非 portrait 一道不依赖朝向的硬陷阱；`T_LANDSCAPE`（T2，`width>height`）因 iqa decode 无 `exif_transpose` 而朝向语义不对齐，在 EXIF 修复（`HAS_EXIF_FIXED`）前**降级为软监控、不入硬门**（评审 blocker）。 |

### 2.3 题库

> **题号说明（反作弊）**：下表 K/L/C/D/M/N/H/T 题号为**内部审计 ID**（`AES_POS_MAP` 的 claim）；模型实际看到的是据 `AES_POS_MAP`+`AES_SCRAMBLE_SEED` 生成的**固定乱序两位位置码**（共享约定 §1.1/1.2），每对 F/R 在乱序里拉远、不相邻、维度交错，清洗器经 POS_MAP 还原 claim 再判 F⊕R。审美 R 题面本就是异质的具体观察（非 F 的字面否定），天然满足 §1.1 R3。
> 极性：F = 正向命中优点，R = 反向缺陷，陷阱 = 确定性硬 / 软陷阱，锚点 = 恒真锚点。配对 kind：硬 = 确定性陷阱真值（无 F⊕R 对偶），软 = F⊕R 计 merit 的软对，none = 单题（陷阱 / 锚点 / 诚实对）。

| ID | 维度 | 极性 | 配对（kind / pair_id） | 题面 | 好图答案 | 真值源 |
| --- | --- | --- | --- | --- | --- | --- |
| **K1** | K | F | 软 / K_compose | 构图明显经过经营、优于随手拍：主体落点 / 留白 / 线条引导有自觉安排，而非仅居中端正。 | 1 | vlm |
| **K2** | K | R | 软 / K_compose | 构图随意失衡：主体贴边或被边缘切割、画面明显歪斜或重心失衡。 | 0 | vlm |
| **K3** | K | F | 软 / K_frame | 取景边缘干净利落：四边与角落没有多余杂物或半截割裂的元素。 | 1 | vlm |
| **K4** | K | R | 软 / K_frame | 取景潦草局促：边缘塞进多余杂物或被切一半的元素。 | 0 | vlm |
| **L1** | L | F | 软 / L_light | 光线明显讲究：有清晰的方向感，光影塑造出立体与氛围（非平光直照）。 | 1 | vlm |
| **L2** | L | R | 软 / L_light | 照明平板无方向感：光线平铺呆滞，画面发闷。 | 0 | vlm |
| **L3** | L | F | 软 / L_tone | 影调层次丰富：高光到阴影的明暗过渡顺滑且有细腻层次。 | 1 | vlm |
| **L4** | L | R | 软 / L_tone | 影调灰平一团：明暗缺乏层次、整体发灰扁平。 | 0 | vlm |
| **C1** | C | F | 软 / C_harmony | 色彩有自觉的整体调性：配色和谐统一、有呼应或克制（黑白 / 单色图：影调统一克制、调性鲜明）。 | 1 | vlm |
| **C2** | C | R | 软 / C_harmony | 色彩严重偏色发脏：整体明显发浊 / 脏污或偏色失调（黑白 / 单色图：影调脏浊不统一）。 | 0 | vlm |
| **D1** | D | F | 软 / D_depth | 有明显的空间纵深：前中后景或虚实关系清楚，画面立体不平板。 | 1 | vlm |
| **D2** | D | R | 软 / D_depth | 画面平板无纵深：前后景糊成一片、缺乏立体感。 | 0 | vlm |
| **D3** | D | F | 软 / D_sep | 主体与背景明显分离：主体清楚地从背景中凸显出来。 | 1 | vlm |
| **D4** | D | R | 软 / D_sep | 主体淹没于背景：主体与背景粘连难分、无法凸显。 | 0 | vlm |
| **M1** | M | F | 软 / M_focus | 有强而明确的视觉中心：一眼锁定主体，注意力被有效引导。 | 1 | vlm |
| **M2** | M | R | 软 / M_focus | 无视觉中心：画面空洞或元素平均散乱，看不出重点。 | 0 | vlm |
| **M3** | M | F | 软 / M_moment（portrait_only） | 人物神态 / 瞬间到位：表情或动作自然有感染力（决定性瞬间）。 | 1 | vlm |
| **M4** | M | R | 软 / M_moment（portrait_only） | 人物神态僵硬尴尬：表情 / 动作呆板别扭、瞬间没抓住。 | 0 | vlm |
| **N1** | N | F | 软 / N_clean | 画面整洁：背景干净不杂乱，没有抢眼的干扰物。 | 1 | vlm |
| **N2** | N | R | 软 / N_clean | 画面杂乱：背景堆满杂物或有抢眼干扰物，注意力被分散。 | 0 | vlm |
| **H1** | H | 陷阱 | none / H_honesty | 上方是一张照片（不是纯文字截图或纯色块）。 | 1 | 诚实性软陷阱（固定期望 H1=1）：真值 = 1；与 H2 共同构成"极性打散的好图固定取值对"（非 F⊕R）。全 1 / 全 0 敷衍模型在 H2 上必答错 → 暴露（见 `H_honesty` 门，评审 major-4）。 |
| **H2** | H | 陷阱 | none / H_honesty | 上方画面是完全空白、没有任何可见内容。 | 0 | 诚实性软陷阱（固定期望 H2=0）：真值 = 0；与 H1 极性相反。一个对所有题"全答 1"的敷衍模型会把 H2 也答成 1≠0 → 被 `H_honesty` 门抓出；"全答 0"则 H1=0≠1 被抓。truth_source 为构造常量，非 VLM 语义。 |
| **T1** | T | 锚点 | none | 上方确实存在一张可见的图片。 | 1 | 常量恒为 1（`T_ANCHOR`）；=0 即模型未看图 / 故障 → 整组不可信。 |
| **T2** | T | 陷阱 | none | 这张图是横向的（宽大于高）。 | 1（示例） | `assets.width>assets.height`（`T_LANDSCAPE`）。⚠ `iqa.py` decode 无 `ImageOps.exif_transpose` → width/height 是原始像素栅格朝向，与 VLM 所见 EXIF 旋正显示朝向可能不一致（portrait 竖拍 JPEG 常带 orientation 6/8）。故 `HAS_EXIF_FIXED=False` 时 T2 **降级软监控，不入硬门、不置 reliable=False，仅记 trap_fail**；iqa 加 `exif_transpose` 重算后（`HAS_EXIF_FIXED=True`）方升级为硬陷阱。good_answer 为示例，运行时真值 = 逐图 `width>height`。 |
| **T3** | T | 陷阱 | none | 这是一张彩色照片（不是黑白 / 单色）。 | 1（示例） | 真值 = `not assets.is_bw_img`。⚠ `is_bw_img` 列**当前不存在**（`assets.is_bw` 仅 PRESET 填，`ingest.py:125`；图像侧从未计算）。本版把 iqa 增廉价饱和度 → `is_bw_img` 检测列为**非 portrait 必做依赖**（评审 major-4）：`HAS_ISBW_IMG=True` 后 T3 发题并入硬门，为非 portrait 提供不依赖朝向的硬陷阱；未补则不发、不判（非 portrait 抗 acquiescence 退化，见遗留问题）。 |
| **T4** | T | 陷阱 | none（portrait_only） | 画面中能看到人脸。 | 1 | `assets.max_face_frac>AES_FACE_MIN`（portrait 池 cv2 Haar；硬陷阱，仅在 `max_face_frac>AES_FACE_MIN` 即确有人脸时发题，与 M3/M4 同口径）。cv2 Haar 有漏检，真值侧有噪声，监控 Haar 假阴率。 |

**展示乱序（位置码 → claim，反作弊）。** 由 `AES_POS_MAP`+`AES_SCRAMBLE_SEED` 固定生成，每对 F/R（含诚实对 H1/H2）间隔 ≥⌈N/3⌉ 且不相邻、维度交错；模型只看到 `01..NN` 与对应题面（无维度/极性）。

非 portrait 基线（无脸/无 isbw，N=22，⌈22/3⌉=8）：
```
01 K1  02 L1  03 C1  04 T1(ANCHOR)  05 K3   06 L3  07 D1  08 M1
09 N1  10 D3  11 H1  12 K2  13 L2  14 T2(LANDSCAPE)  15 C2
16 K4  17 L4  18 D2  19 M2  20 N2  21 D4  22 H2
```
> 校验：K_compose 01↔12、K_frame 05↔16、L_light 02↔13、L_tone 06↔17、C_harmony 03↔15、D_depth 07↔18、D_sep 10↔21、M_focus 08↔19、N_clean 09↔20、H_honesty 11↔22，均 ≥11 不相邻；T1=04、T2=14 单题陷阱。

portrait（有脸/无 isbw，N=25，加 M_moment(M3/M4)/FACE(T4)，⌈25/3⌉=9）：
```
01 K1  02 L1  03 C1  04 T1  05 K3  06 L3  07 D1  08 M1  09 T4(FACE)
10 N1  11 D3  12 M3  13 H1  14 K2  15 L2  16 T2  17 C2  18 K4
19 L4  20 D2  21 M2  22 N2  23 D4  24 M4  25 H2
```
> 校验：K_compose 01↔14、L_light 02↔15、C_harmony 03↔17、K_frame 05↔18、L_tone 06↔19、D_depth 07↔20、M_focus 08↔21、N_clean 10↔22、D_sep 11↔23、M_moment 12↔24、H_honesty 13↔25，均 12~14 全 ≥9 不相邻；T1=04/T4=09/T2=16 单题陷阱。`HAS_ISBW_IMG=True` 时按各自 `AES_SCRAMBLE_SEED` 在中段插入 COLOR(T3) 单题 → N=23/26。

### 2.4 System prompt

```
你是严谨的摄影审美评审。只依据所给这一张图片作答；不解释、不推理、不输出任何多余文字。
下面是固定的若干道判断题，每题有一个两位题号（如 03）。逐题判真假：陈述为真写 1，为假写 0。
评的是【画面结构性审美】（构图取景、光线影调、色彩协调、空间层次、主体显著、画面整洁），与清晰度/噪点/压缩等技术质量无关；不要因为图片分辨率低或看起来已调色而扣分。
注意：很多题问的不是"有没有"，而是"是否达到明显高于随手拍的较高水准"——只有确实出彩才写 1，平庸或仅及格写 0；请严格区分，不要一律给好评。
【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这张图独立判断，不要根据别的题的答案来推断本题。每题都必须答；看不准时给出你最可能的判断，不得留空、不得答"不确定"。
作答格式：把【题号和答案紧贴】写出（03 为真写 031，为假写 030），题与题之间空一格，按题号从小到大，每个题号恰好出现一次。
只输出这一串题号答案，不要任何其它文字。
```

### 2.5 输出格式

**反作弊位置码方案（共享约定 §1.1/1.2）**：模型逐题输出"位置码 + 答案"紧贴，位置码 = 两位数字、答案 = 1/0 紧贴其后，题间空格，按位置码升序，每码恰一次。解析 `re.findall(r'(\d{2})([01])', s)` → `{pos:bit}`，再经 `AES_POS_MAP` 还原 claim/极性。**下方示例串与 §2.6 伪码中的 `K1/T1…` 为审计 ID 形式（便于阅读）；线上 prompt 实际发的是固定乱序的位置码。**

应发位置码集合由发题逻辑**动态确定**（has_face / HAS_ISBW_IMG 决定 `AES_POS_MAP` 的活动子集，对应下方 `EXPECTED`），清洗器据该集合做解析门。

按 §2.3 的展示乱序，一张合格的彩色横图（LANDSCAPE 横=1）位置码好例串：

- **非 portrait 基线（无脸/无 isbw，N=22）**：
  ```
  011 021 031 041 051 061 071 081 091 101 111 120 130 141 150 160 170 180 190 200 210 220
  ```
- **portrait（有脸/无 isbw，N=25，加 M_moment/FACE）**：
  ```
  011 021 031 041 051 061 071 081 091 101 111 121 131 140 150 161 170 180 190 200 210 220 230 240 250
  ```

注：`HAS_ISBW_IMG=True` 时按各自 POS_MAP 在中段插入 COLOR(T3) 一道 → N=23/26（种子重生成）；LANDSCAPE(T2) 始终在串中（软监控或硬陷阱视 `HAS_EXIF_FIXED`）。

**解析正则：** `pairs = re.findall(r'([A-Z]\d)([01])', s)` → `dict{ID:bit}`；容忍多余空格 / 换行。缺 ID / 重复 ID / 未知 ID（不在本次 `EXPECTED`）→ `reliable=False`（见解析门）。

### 2.6 清洗器（有序门伪码）

```
# 反作弊: 先按 AES_POS_MAP 把模型输出的两位位置码 → claim/极性; 下文 ans[...] 均指还原后的 claim (共享约定 §1.1/1.3)
# 全局开关: AES_FACE_MIN, HAS_EXIF_FIXED, HAS_ISBW_IMG, CONTRA_TOL, AES_TEMPERATURE, AES_RETRY

# ── 门 1: 解析门 parse ─────────────────────────────────────────
has_face = (assets.max_face_frac is not None and assets.max_face_frac > AES_FACE_MIN)
BASE     = {K1,K2,K3,K4, L1,L2,L3,L4, C1,C2, D1,D2,D3,D4, M1,M2, N1,N2, H1,H2, T1,T2}
EXPECTED = BASE
         ∪ ({M3,M4,T4} if has_face else {})
         ∪ ({T3}       if HAS_ISBW_IMG else {})
EXPECTED_POS = {pos for pos,c in AES_POS_MAP.items() if c.claim in EXPECTED}   # 本次应发的两位位置码集
pairs = re.findall(r'(\d{2})([01])', s)                 # 模型输出两位位置码
pos   = [p[0] for p in pairs]
ans   = {AES_POS_MAP[p].claim: int(b) for p,b in pairs} # 还原→以 claim(K1/T1/..)为键, 供门2-6
if set(pos) != EXPECTED_POS  or  len(pos) != len(EXPECTED_POS)  # 缺/重/未知位置码
   or  not pairs:                                       # 空
       FAIL → reliable=False
              reason ∈ {parse_missing, parse_dup, parse_unknown, parse_empty}
              重问 1 次 (temp=AES_TEMPERATURE, 可打乱题序);
              仍失败 → merit_frac=NULL, 不投票, auto_verdict 不变(维持/置 review)

# ── 门 2: 锚点门 anchor ────────────────────────────────────────
if ans['T1'] != 1:                  # T_ANCHOR 恒真; 未看图/输出空白会给 0
    FAIL → reliable=False; reason=anchor; 重问 → 仍败 → merit_frac=NULL

# ── 门 3: 诚实性软陷阱门 H_honesty (硬, 构造真值, 抓全1/全0敷衍) ──
expect_H1 = 1   # '是照片' 好图必 1
expect_H2 = 0   # '完全空白' 好图必 0, 极性相反
if ans['H1'] != 1 or ans['H2'] != 0:
    FAIL → reliable=False; reason=honesty; 重问 → 仍败 → merit_frac=NULL
# 全答 1 的敷衍模型 H2=1≠0 被抓; 全答 0 则 H1=0≠1 被抓
# —— F⊕R 软对 'viol=0 盲区' 之外的独立诚实性硬门(评审 major-4)

# ── 门 4: 确定性陷阱门 trap (硬, 与我方信号比对) ───────────────
# T4 人脸 (仅 has_face 时已发):
if has_face and ans['T4'] != 1:     # 发题前提即 max_face_frac>AES_FACE_MIN
    FAIL → reliable=False; reason=trap:T4
# T3 颜色 (仅 HAS_ISBW_IMG 时发; 非 portrait 抗 acquiescence 主力):
if HAS_ISBW_IMG:
    expect_T3 = 0 if assets.is_bw_img else 1
    if ans['T3'] != expect_T3:
        FAIL → reliable=False; reason=trap:T3
# T2 朝向 (仅 HAS_EXIF_FIXED=True 才入硬门):
if HAS_EXIF_FIXED:
    expect_T2 = 1 if assets.width > assets.height else 0
    if ans['T2'] != expect_T2:
        FAIL → reliable=False; reason=trap:T2
else:
    # HAS_EXIF_FIXED=False: T2 不判 reliable, 仅软监控记录(评审 blocker)
    aes_soft_trap_fail = 1 if ans['T2'] != (assets.width > assets.height) else 0

# ── 门 5: 正反矛盾门 contradiction (全软对 F⊕R) ────────────────
SOFT_PAIRS = [(K1,K2),(K3,K4),(L1,L2),(L3,L4),(C1,C2),
              (D1,D2),(D3,D4),(M1,M2),(N1,N2)]
           + ([(M3,M4)] if has_face else [])
viol = sum(1 for (f,r) in SOFT_PAIRS if ans[f] == ans[r])   # F⊕R 被违反(同值=矛盾)
aes_contradiction_count = viol                              # 落库
if viol > CONTRA_TOL:
    FAIL → reliable=False; reason=contradiction_soft; 重问 → 仍败 → merit_frac=NULL(转 review)
# H1/H2 不是 F⊕R 对, 不计入此门, 由门 3 单独判

# ── 门 6: 裁定 reliable ────────────────────────────────────────
reliable = (门1 ∧ 门2 ∧ 门3 ∧ 门4 ∧ 门5 全过)
if reliable:
    计算 merit_frac (见 §2.7)
    写入独立 keep 排序键 aes_sort_key = merit_frac
    置 aes_keep_vote                       # 仅供观测/可选独立软栏, 不进 gate.py good 计数器
else:
    merit_frac = NULL; 不计、不投票、不入排序; auto_verdict 维持/置 review
# 任何情况下都【不产生 drop、不把 review/None 翻成 keep】
```

### 2.7 可区分信号（merit_count / merit_frac）与软信号用法

**merit_count 定义。** `merit_count` = 干净命中的**结构性审美优点**对数，仅 `reliable=True` 时有定义。计法只用通过一致性的 F⊕R 软对，**一对至多 1 分，去重抗单边噪声**：对每个软对 (F, R)，**当且仅当 `ans[F]==1` 且 `ans[R]==0`**（干净命中该优点）记 1；其余（0/0、1/1[已被门 5 限 `viol≤CONTRA_TOL`]、0/1）记 0。

参与计分的软对：

```
MERIT_PAIRS = [K_compose(K1,K2), K_frame(K3,K4),
               L_light(L1,L2),  L_tone(L3,L4),
               C_harmony(C1,C2),
               D_depth(D1,D2),  D_sep(D3,D4),
               M_focus(M1,M2),  N_clean(N1,N2)]        # 无脸 N=9
有脸追加      M_moment(M3,M4)                            # → N=10
```

H / T 陷阱对不计入 merit。

**统一软信号 merit_frac（评审 minor）。** `merit_frac = merit_count / merit_n ∈ [0,1]`。跨 portrait / 非 portrait 与有脸 / 无脸的题数差全部用 `merit_frac` 消除——keep 排序与金标校准**只用 `merit_frac`**，弃用绝对阈值 `MERIT_KEEP_MIN[_PORTRAIT]`。

**反退化（评审 major-3）。** 题面已从"有没有优点"改写为"是否明显优于随手拍 / 达到较高水准"，system prompt 额外强调"平庸写 0"，拉开判别面以抗 tad66k 已策展池 + 不扣分指令导致的 F 全 yes、merit 向满分坍缩。落地把 200 图 pilot 的**每 F/R 题 yes 率 + `merit_frac` 直方图**设为全库重跑前的硬 gate（见 §2.9 / §2.11）。分布性目标：`merit_frac` 在 [0,1] 上非单点堆积（不得 >50% 落单值、不得贴满分），形成可用于 keep 排序的连续谱。

**软信号用法（排序-only，永不 drop / 永不改判级；用户拍板 + 评审 major-2）。**

```
if not aes_reliable:
    merit_frac   = None
    aes_sort_key = None
    # 不投票、不入排序; 若该图当前无其它判级,
    # 由编排置/保持 auto_verdict='review' (reason=qa_aes_unreliable)
    return                                   # 永不改 keep/drop

# reliable:
merit_frac    = merit_count / merit_n        # ∈[0,1], 跨群可比
aes_sort_key  = merit_frac                   # 仅写入 keep 优先级【排序列】
aes_keep_vote = 1 if merit_frac >= MERIT_KEEP_FRAC else 0   # 默认 0.65; 仅独立观测软栏

update_asset_fields(merit_count, merit_n, merit_frac,
                    aes_sort_key, aes_keep_vote,
                    aes_reliable=True, aes_reason=None, ...)
```

与既有判级的关系（严守排序-only，评审 major-2）：

- **不覆盖任何 drop：** 流程 1 / IQA 已 drop 的图，本信号无效（审美不能救回）。
- **绝不进 `gate.py` 的 `verdict_relative` good 计数器**（该计数器 `good >= keep_min_good` 会把 borderline review 翻成 keep，见 `gate.py:118-128`）；`aes_sort_key` 只写一个独立 keep 优先级排序列，不参与任何 verdict 翻转。
- **低 merit_frac 绝不触发 drop / review 升级**——审美低只是排序靠后。
- **回归断言：** 加入 `aes_sort_key` / `aes_keep_vote` 后，全样本的 keep / review / drop verdict 逐一不变（见 §2.9 "drop 影响" + "verdict 不变" 断言）。

**可信度产出列（落库）：**

| 列 | 类型 | 含义 |
| --- | --- | --- |
| `aes_reliable` | BOOLEAN | 本次审美问卷是否通过清洗器全部门。 |
| `aes_reason` | TEXT | 不可信原因：`parse_missing` / `parse_dup` / `parse_unknown` / `parse_empty` / `anchor` / `honesty` / `trap:T2` / `trap:T3` / `trap:T4` / `contradiction_soft` / `null`（可信）。 |
| `aes_contradiction_count` | INTEGER | 门 5 软对违反数 viol（F==R 的对数），0 为最佳。 |
| `aes_trap_fail` | TEXT | 触发的硬陷阱 id 列表（honesty / T3 / T4，及 `HAS_EXIF_FIXED` 时的 T2），空为通过。 |
| `aes_soft_trap_fail` | INTEGER | T2 软监控位：`HAS_EXIF_FIXED=False` 时 `(ans[T2]!=(width>height))` 指示（1/0），评估 EXIF 隐患与朝向蒙对率，不进 reliable。 |
| `merit_count` | INTEGER, nullable | reliable 时干净命中的结构性审美优点对数（0..N），unreliable 为 NULL。 |
| `merit_n` | INTEGER | 本次应计入 merit 的软对数 N（无脸 = 9，有脸 = 10），供归一化。 |
| `merit_frac` | DOUBLE, nullable | `merit_count/merit_n ∈ [0,1]`，跨 portrait / 有脸-无脸可比的统一软信号；keep 排序与金标校准均在此列。 |
| `aes_sort_key` | DOUBLE, nullable | 写入 keep 优先级排序的独立列（= merit_frac），**不进 `gate.py` `verdict_relative` 的 good 计数器**（评审 major-2）。 |
| `aes_keep_vote` | INTEGER | 可选独立观测软栏（`merit_frac >= MERIT_KEEP_FRAC ? 1 : 0`）；仅观测 / 可选聚合，默认不参与判级翻转。 |
| `aes_raw` | TEXT | 模型原始输出串，复跑 / 审计用。 |
| `aes_retry` | BOOLEAN | 是否触发过重问。 |

### 2.8 config 常量

| 常量 | 默认 | 含义 |
| --- | --- | --- |
| `AES_QUESTIONNAIRE` | 见 §2.3 questions（K/L/C/D/M/N 各 F⊕R 软对 + H1/H2 诚实性 + T1/T4，T2/T3 视开关） | 流程 2 审美题库，固定单一真值源，llm_qa 审美 runner 与 Web UI 共享；固定以命中 prefix-cache。 |
| `AES_SYSTEM_PROMPT` | 见 §2.4 system_prompt | 固定可缓存 system prompt；含"平庸写 0"反退化指令。 |
| `VLLM_IMAGE_LONGEDGE` | 768（复用现值） | 送审图片长边下采样；审美与流程 1 可共用同一下采样图以省编码；该图为 EXIF 旋正后的显示朝向。 |
| `CONTRA_TOL` | 1 | 门 5 软对违反容差；`viol>CONTRA_TOL` 即 `reliable=False`。9~10 对中允许至多 1 对 F==R。 |
| `AES_FACE_MIN` | 0.04（对齐 `config.py:129` GATE.min_face_frac） | 统一口径：`max_face_frac>AES_FACE_MIN` 才判"确有人脸"→ 发 M3/M4/T4 且 T4 真值 = 1；无脸图不发这三题（评审 major-5）。与流程 1 `IMQ_FACE_MIN` 分开命名（用途不同）。 |
| `MERIT_KEEP_FRAC` | 0.65 | `merit_frac >=` 此值记 `aes_keep_vote=1`（仅独立观测软栏）；统一用 frac 消除题数差（评审 minor）；金标（在 merit_frac 上）校准后再调。 |
| `AES_RETRY` | 1 | `reliable=False` 时重问次数；仍不可信 → `merit_frac=NULL` → review。 |
| `AES_TEMPERATURE` | 0.1（复用现值） | 审美调用温度；低温稳一致性。 |
| `AES_QUESTIONNAIRE_TAG` | `'aes'` | `llm_qa.questionnaire` 取值，区别于现 A/B/C/caption 与流程 1；item = claim 审计 ID，answer ∈ {0,1}。 |
| `AES_POS_MAP` / `AES_SCRAMBLE_SEED` | 见 §2.3 | 位置码→claim 固定映射 + 乱序种子（每对 F/R 拉远、维度交错）；模型侧单一事实来源（反作弊 §1.1）。 |
| `HAS_EXIF_FIXED` | False | 全局开关：iqa decode 是否已加 `ImageOps.exif_transpose` 使 width/height = 显示朝向。False → T2 降为软监控（不入硬门、不置 reliable=False，仅记 `aes_soft_trap_fail`）；True → T2 入硬陷阱门（评审 blocker）。**与流程 1 共享，见 §1.5 P-1。** |
| `HAS_ISBW_IMG` | False | 全局开关：iqa 是否已为图像计算 `is_bw_img`（饱和度检测）。False → T3 不发题、不入硬门（非 portrait 抗 acquiescence 退化，见 §2.11）；True → T3 发题入硬门，给非 portrait 一道不依赖朝向的硬陷阱（评审 major-4）。亦驱动 C 维 BW-aware 题面分支。**与流程 1 共享，见 §1.5 P-2。** |

### 2.9 可区分性验收指标

| 指标 | 目标 |
| --- | --- |
| 每道 F/R 审美题全库 yes 率（§5 可区分性硬验收） | 落在 [3%, 97%]；超出退化带即该题失效、重写题面。**pilot 硬 gate：** 200 图 pilot 任一 F/R 出退化带即阻断全库重跑（对照 `B_quality` 98.9% 坍缩）。 |
| `merit_frac` 分布 | 非单点堆积、不得贴满分；在 [0,1] 上有可用区分度（理想近钟形 / 有尾），不得 >50% 集中单值。**pilot 硬 gate：** 直方图 >50% 落单值即阻断全库、强制改题（评审 major-3）。 |
| 首问不可信率 `aes_reliable=False`（首问） | 可监控；过高（经验 >25%）说明题面 / 格式 / 陷阱需改。 |
| 诚实性陷阱失败率（`H_honesty`：H1!=1 或 H2!=0） | 低；偏高直接说明大量"未看图 / 全 1 或全 0 敷衍"被抓（P4 acquiescence 检出有效性），同时反查是否题面把模型逼成模式化。 |
| 硬陷阱失败率（T3 颜色 / T4 人脸；`HAS_EXIF_FIXED` 后含 T2 朝向） | 低（经验 <5%）；`HAS_ISBW_IMG` 前非 portrait 仅靠 `H_honesty` 抗 acquiescence，必须监控其失败率作为唯一防线。 |
| T2 软监控失败率 `aes_soft_trap_fail`（`HAS_EXIF_FIXED=False` 时） | 观测 EXIF 隐患规模与朝向蒙对率；若 portrait 池显著高于非 portrait，印证 EXIF 旋转问题，推动尽快上 `exif_transpose` 修复。 |
| 软对矛盾率（门 5 `viol>0` 占比） | 受控；整体均值低且 `contradiction_count` 有分布；持续高说明题对不够互斥，改写 R。 |
| `merit_frac` 与既有 `aesthetic_vlm` / `aesthetic` 的秩相关 + 增量决策门 | Spearman 约 0.3~0.7；**落地决策门（评审 minor）：** 若 `merit_frac` vs `aesthetic_vlm` 的 Spearman>0.7 或对 keep 排序的增量区分（AUC/NDCG）不显著，则不全库铺开，退回只用现有两个审美分——本流程 ROI 须在 pilot 证明。 |
| drop 影响 + verdict 不变断言 | 恒为 0 —— 审美在任何 reliable 取值下都不得产生 drop；且回归断言：加入 `aes_sort_key` / `aes_keep_vote` 后全样本 keep / review / drop verdict 逐一不变（评审 major-2）。 |
| 无脸 portrait 池图的 `M_moment`/T4 发题率 | 应为 0（无脸不发 M3/M4/T4）；断言无脸 portrait 池图不出现 M3/M4 稳定矛盾（评审 major-5）。 |

### 2.10 schema / 落库改动

1. **`llm_qa`：** 新增 `questionnaire` 取值 `'aes'`（逐题落库，item = claim 审计 ID（如 `K1`，非模型看到的位置码），answer ∈ {0,1}，`raw` 存原始位置码串，rationale 留空省 token）；与现 A/B/C/caption 并存，复用现有 `add_qa()`。
2. **`assets` 新增列：** `merit_count INTEGER`（nullable）、`merit_n INTEGER`、`merit_frac DOUBLE PRECISION`（nullable）、`aes_sort_key DOUBLE PRECISION`（nullable）、`aes_reliable BOOLEAN`、`aes_reason TEXT`、`aes_contradiction_count INTEGER`、`aes_trap_fail TEXT`、`aes_soft_trap_fail INTEGER`、`aes_keep_vote INTEGER` —— 经 `db.py` 的迁移列表（`db.py:239` 风格）追加，用 `db.py:386` 的 `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` 迁移。
3. **`config.py`：** 新增 `AES_QUESTIONNAIRE` / `AES_SYSTEM_PROMPT` / `CONTRA_TOL` / `AES_FACE_MIN` / `MERIT_KEEP_FRAC` / `AES_RETRY` / `AES_QUESTIONNAIRE_TAG` / `AES_TEMPERATURE` / `HAS_EXIF_FIXED` / `HAS_ISBW_IMG`（见 §2.8）；不动现 `QUESTIONNAIRE_A/B/C`。
4. **新模块** `dataset_build/source_qa/aesthetic_qa.py`（或 `llm_qa.py` 内 `run_aesthetic`）：独立第二次 vLLM 调用 + 审美清洗器（6 门）；与流程 1 解耦，各自 prefix-cache。
5. **【blocker 修复，必做，与流程 1 共享 §1.5 P-1】iqa.py decode 加 EXIF 旋正：** `_tensor` 里 `im = Image.open(path); im.load()`；在 `convert('RGB')` 之前 / 之后 `im = ImageOps.exif_transpose(im)`；取 `native_w, native_h = im.size` 使 `width`/`height`/`megapixels`/`longedge` = 显示朝向（与送审图、与 musiq 等一致）；对已写库的受影响图重跑 iqa（尤其 portrait 池竖拍 JPEG）；完成后置 `HAS_EXIF_FIXED=True`，T2 升级为硬陷阱。
6. **【major-4 修复，非 portrait 必做依赖，与流程 1 共享 §1.5 P-2】iqa.py 增廉价 `is_bw_img` 检测：** decode 后下采样，计 HSV/Lab 平均饱和度或 R/G/B 通道间最大均差，低于阈值判 `is_bw_img=1`，落 `assets.is_bw_img`（图像侧，区别于现仅 PRESET 的 `is_bw`）；新增列 `assets.is_bw_img INTEGER`（同迁移方式）；置 `HAS_ISBW_IMG=True` 后 T3 入硬门，给非 portrait 一道不依赖朝向的硬陷阱，并驱动 C 维 BW-aware 题面分支。
7. **`gate.py` / 编排：** 把 `aes_sort_key`（= merit_frac）接入**独立 keep 优先级排序列**，**明确不进 `verdict_relative` 的 good 计数器**（防 review→keep 翻转，评审 major-2）；断言审美永不触发 drop 且不改任何 verdict；`aes_keep_vote` 仅作可选独立观测软栏。

### 2.11 遗留问题

1. **[已从可选升为硬依赖] `is_bw_img`（T3）：** 非 portrait（≈78% 主体量）在 `HAS_ISBW_IMG=False` 且 T2 因 EXIF 降级时，硬陷阱会归零，只剩 `H_honesty` 一道软诚实陷阱抗 acquiescence——薄弱。故落地优先级里 `iqa.is_bw_img` 列为非 portrait 必做；但仍开放：饱和度阈值如何定（低饱和成品图 vs 真黑白的边界）、是否需 Lab a/b 通道而非 HSV 以抗偏色？
2. **[blocker 修复路径选择]** 已采纳 (a) iqa 加 `exif_transpose` 重算 width/height；但需确认：重跑全库 iqa 的成本与 `native_w/h` 改写是否影响下游已用旧 `megapixels`/`longedge` 的门（gate 分辨率门）？若重算改变了某些图的 `longedge` 判定，需联动复核 gate 分辨率结论。在 `HAS_EXIF_FIXED` 落地前 T2 软监控期间，`aes_soft_trap_fail` 的 portrait / 非 portrait 分布是验证 EXIF 假设的直接证据。
3. **[H_honesty 有效性]** H1/H2 是构造的极性打散诚实对，能抓"全 1 / 全 0"敷衍；但一个"聪明的敷衍"模型（读懂 H2 是反向措辞、其余全答优点）仍可绕过。是否需要在 K/L/C/D/M/N 软对里也随机打散少量 R 的措辞极性（部分"反向措辞但好图应 = 1"）以让纯模式化必触矛盾？代价是题面更绕、可能升高诚实模型的偶发矛盾。pilot 观测 `H_honesty` 失败率与软对矛盾率联动后定。
4. **[反退化校准]** 题面已提升判别面 + system 强调"平庸写 0"，但 tad66k 策展池上 F 仍可能 yes>97%。`merit_frac` 直方图与 per-题 yes 率是 pilot 硬 gate；若多道 F 仍退化，是否进一步把 F 改成"相对水准三选一里的最高档二元化"（更难全 yes）？需与"保持二元 P1"权衡。
5. **[ROI 决策门]** 库内已有 `aesthetic`（laion）+ `aesthetic_vlm` 两审美分。`merit_frac` 增量价值须在 pilot 用 Spearman + keep 排序增量区分证明；若与 `aesthetic_vlm` 秩相关>0.7 即无新增，退回不铺开。开放：增量区分用什么标的（人工金标 keep 排序？下游训练收益代理？）。
6. **[merit_frac 金标校准]** 虽审美永不进硬 drop，但 `MERIT_KEEP_FRAC` 作排序 / 可选软栏阈值也宜用金标分布校准（§6 思路）；是否需 200-500 张审美金标做 `merit_frac` 与人审 keep 优先级的相关，而非仅看分布？
7. **[cv2 Haar 假阴]** T4 真值依赖 cv2 Haar（漏检率不低）：真有脸但 Haar 漏检 → 该图归入"无脸"不发 M3/M4/T4，损失人脸维度但不产生假矛盾（比之前"发题后矛盾"更安全）；仍开放是否换更稳的人脸检测器以提高 portrait 召回。
8. **[CONTRA_TOL 松紧]** =1 对 9~10 软对是否过松 / 过严，pilot 看软对矛盾率分布定；`H_honesty` 独立成硬门后，软对容差可略松以容诚实边界图。

---
## 流程 3 — Preset QA（单 preset · 6 固定确定性探针 · 单探针 before/after 二元判级 + 一致性自检 + 跨探针投票）

### 3.1 目标与数据依据（含 style=None 的重构说明 + 本次折叠了哪些评审修订）

**目标。** 对单个 preset，在 6 个固定的 before/after 真实渲染对上，逐探针用二元 F/R 判断题 + 确定性陷阱裁定该 preset 的 look 是否（a）专业非破坏 **PRO**、（b）有明确可学的编辑意图 **INTENT**、（c）在 before→after 方向上连贯一致非随机 **COH**。每探针先过确定性 no-op 门，再过清洗器（解析→锚点→确定性陷阱→F⊕R 矛盾）裁定 `reliable`；仅 `reliable` 探针参与逐题跨探针多数投票；输出跨探针 `coherence_score` 作可区分排序信号。**preset 永不自动 keep**：all-pass→`review`，任一 `unreliable` 探针不得拉高通过率。

**三处由数据现实强制的重构。**

1. **style=None → COH（彻底去 style 依赖）。** 预设 `style` 字段全为 None，原 C2「after 与声称 style 一致」无真值可锚，**删除**。重构为 COH：VLM 只比较 after 相对 before 的**内部连贯性**（全局色调/影调走向是否统一 vs 局部失控/随机），**不引用任何 style 或 scene 标签**。评审确认此方向正确（strengths #1），但指出原草稿 COH 仍在退化风险（degeneracy #4）——本次把 COH 收紧为 0 容差软对并明确其为软排序信号，详见 3.3/3.7。

2. **scene 只有 any/portrait → 探针改用我方确定性图像信号选取（折叠 blocker #1 / trap_signal_gaps #1、#4）。** `ingest.py:72` 把 image scene 固定为 `t.get('scene') or 'any'`，实测仅 `any(~78%)`/`portrait(~22%)`；`resolve_probes(preset_qa.py:238)` 按 `scene` 去重根本挑不出「天空/美食/建筑/夜景」。原草稿把「6 探针覆盖 6 类色调挑战」当既成事实，使 `coherence_score` 语义悬空。**本次修订**：probe_set 不再依赖伪 scene，改用**我方可确定性计算的图像信号**（`is_portrait_pool`、`max_face_frac`、新增 `mean_luma` / `highlight_frac` / `shadow_frac` / `saturation_mean`）来构造色调挑战覆盖（详见 3.2、3.11 新增检测）。

3. **真值源严格限定 paired_metrics + after_iqa（P6）。** 本流程唯一可锚的确定性信号为 `preset_previews.paired_metrics` 已算列：`delta_e2000_mean / delta_e2000_p95 / ssim / hist_emd_L / hist_emd_ab / clip_pct / noop_score`，以及 `after_iqa.musiq / clipiqa+`。图像级 `is_bw/orientation` 陷阱在 preset QA 不适用（同探针 before/after 同朝向同色域，无区分力）。

**本次折叠的评审修订一览（blocker/major 全落正文，minor 落正文或验收）：**

| 评审条目 | 严重度 | 折叠到本设计的处理 | 落点 |
|---|---|---|---|
| probe scene 无真值 | blocker | 探针改确定性图像信号选取 + 诚实降级表述 | 3.2 / 3.11 |
| T3(DESTRUCT) clip_pct 退化恒 0 | blocker | T3 真值改 `ssim < τ_ssim`（结构破坏），并加分布硬验收 | 3.3 / 3.6 / 3.9 / 3.10 |
| T2(CHANGED) 过 no-op 门后恒 1 | major | T2 **反转**为「近乎一模一样」，真值 `ΔE < τ_noop_low`，0/1 混合 | 3.3 / 3.6 / 3.10 |
| T1 anchor 单题挡不住全-yes | major | 新增反向锚点 T1b（good=0）构 F⊕R 互斥 | 3.3 / 3.6 |
| INTENT 非真二分 | major | 重写 I2 为 I1 严格否定，「极端怪异」并入 PRO(P2) | 3.3 |
| pass_c=None / 无 probe_id 列 | major | pass_c 恒 0/1；llm_qa 加 `probe_id`；新表 `preset_qa_runs` | 3.5 / 3.11 |
| LUT vs LR ΔE 量纲不可比 | major | engine-aware 阈值 + order=0 以 `noop_score` 为主判 | 3.6 / 3.8 / 3.9 |
| 768px 细缺陷不可辨 | minor | 题面删 banding/halo，交确定性陷阱；验收加分辨率回退 | 3.3 / 3.10 |
| κ 未标定不得硬 drop | minor | `PRO_DROP_ENABLED/INTENT_DROP_ENABLED` 默认 False，κ-gate 显式开关 | 3.8 / 3.9 |
| 无差别重问浪费调用 | minor | 重问按 reason 分流（parse/anchor 重问；trap/contra 改条件或不重问） | 3.6 |

### 3.2 6 固定探针集（确定性信号选取）

探针**不再**声称伪 scene。改由 `resolve_probes` 在我方确定性图像信号列上选取，使 6 张图各命中一类色调挑战；`probe_id` 稳定 → 渲染缓存命中。若 3.11 的廉价图像检测未就绪，则**诚实降级**：6 探针仅保证「6 张不同的高美学图、内容多样」，`coherence_score` 只反映「look 在 6 张内容上的稳定性」，不得宣称命中具体色调类别。

| slot | 确定性选取条件（我方可算列） | 覆盖的色调/区域挑战 | 为何入选 |
|---|---|---|---|
| 1 | `is_portrait_pool=1` 且 `max_face_frac∈[0.08,0.45]` | 肤色 / 人脸区域 | 肤色最易翻车（偏色/塑料感/吹高光在脸上暴露）；唯一真实可锚的「场景」。local-mask 人像预设的人脸 mask 区在此被检验。 |
| 2 | `highlight_frac` Top + `mean_luma` 高 | 大面积高光 / 吹白风险 | 检验 T_DESTRUCT（高光涂抹/结构破坏）与冷高光偏色；与 slot6 暗部为主形成两端覆盖。天空 mask 预设在此命中。 |
| 3 | `saturation_mean` Top | 高饱和 / 白平衡与饱和控制 | 暖色/绿植高饱和检验过饱和塑料感与白平衡；与中性槽互补。 |
| 4 | `saturation_mean` 最低（近中性，非纯 BW） | 中性基线 / 偏色放大镜 | 低饱和内容是 look 偏色的放大镜（原本无色偏，after 任何色偏一目了然）。 |
| 5 | `shadow_frac` 中等 + `max_face_frac` 中等（混合内容） | 中等暗部 / 暗部提亮过渡 | 检验阴影提亮是否塌陷/噪点放大、白平衡在内容上的连贯性（COH 主战场）。 |
| 6 | `shadow_frac` Top + `mean_luma` 低 | 大面积暗部 / 死黑塌陷 | 检验 T_DESTRUCT（死黑塌陷/暗部 posterize）与暗部提亮的结构破坏；高光点检验高光保护。 |

> 选取实现：`resolve_probes` 在 `images` 上先按 `aesthetic DESC` 取候选池，再用上述确定性列各取分位极值/匹配，去重落 6 个稳定 `asset_id`。其 6 类色调覆盖**需人工一次性确认**（见 3.12），确认前 `coherence_score` 按降级语义解读。

### 3.3 维度与题库（最终题面）

固定 9 题（含反向锚点 `T1b` 共 10 行），全 preset 通用、无 portrait 分支 → 每探针题集恒定，system prompt 可 prefix-cache。`good_answer` 指「好 after」应给的答案；陷阱/锚点 `truth_source` 为确定性，**逐探针**计算。

> **题号说明（反作弊，共享约定 §1.1/1.2）**：下表 `P1/P2/I1/I2/H1/H2/T1/T1b/T2/T3` 为**内部审计 ID**（`PRESET_POS_MAP` 的 claim）；模型实际看到的是据 `PRESET_POS_MAP`+`PRESET_SCRAMBLE_SEED` 生成的**固定乱序两位位置码**，PRO/INTENT/COH 的 F/R 与正/反锚点（T1/T1b）均拉远不相邻、维度交错。`T1b` 不再是字母后缀特例，只是又一个普通位置码对应的反向锚点 claim。清洗器经 POS_MAP 还原 claim 再判 F⊕R。

| ID | 维度 | 极性 | 配对(kind) | 题面（最终版，中文，VLM 仅看 before/after 两图） | 好图答案 | 真值源 |
|---|---|---|---|---|---|---|
| P1 | PRO | F | PRO(hard) | after 是专业、克制的修饰：整体影调与色彩自然，没有明显的**整体过曝死白、大面积死黑、明显塑料感、或严重整体偏色**。 | 1 | vlm |
| P2 | PRO | R | PRO(hard) | after 存在破坏性问题：**大面积过曝死白或死黑、明显塑料感、严重整体偏色、或像被极端/怪异地处理过**（之一或多项）。 | 0 | vlm |
| I1 | INTENT | F | INTENT(hard) | after 相对 before 有明确、可学习的编辑方向（可清楚看出在做调色/影调/对比的处理）。 | 1 | vlm |
| I2 | INTENT | R | INTENT(hard) | after 几乎等于 before、看不出实质编辑（近似 no-op）。 | 0 | vlm |
| H1 | COH | F | COH(soft·0容差) | after 相对 before 的调整是一个连贯一致的整体 look：色调与影调走向统一，像同一套专业预设作用于全图。 | 1 | vlm |
| H2 | COH | R | COH(soft·0容差) | after 的调整随机或局部失控：不同区域色调走向互相矛盾、有突兀的局部色块/断裂，不像统一的 look。 | 0 | vlm |
| T1 | TRAP·锚点 | anchor(F) | ANCHOR(hard) | 上面确实给了我**两张**图（before 和 after）。 | 1 | 恒为 1（每探针均提交两张图） |
| T1b | TRAP·锚点 | anchor(R) | ANCHOR(hard) | 上面只给了我**一张**图。 | 0 | 恒为 0（反向锚点，与 T1 构 F⊕R） |
| T2 | TRAP | trap | — | after 与 before **几乎一模一样**、看不出肉眼可辨的差异。 | 由真值定 | `delta_e2000_mean < τ_noop_low` → 1，否则 0（逐探针） |
| T3 | TRAP | trap | — | after 出现明显的**结构破坏 / 色阶断裂 / 细节涂抹涂糊**（不是自然的调色）。 | 由真值定 | `ssim < τ_ssim` → 1，否则 0（逐探针） |

**关键修订说明：**

- **INTENT 重写为真二分（major）。** I1='有明确可学编辑方向'，I2 收窄为 I1 的**严格逻辑否定**='近似 no-op'。原草稿 I2 的「或极端怪异」是独立第三维，会让「中等编辑但方向不够明确」的温和 preset（I1=0∧I2=0 语义自洽）被硬 F⊕R 误剔。「极端怪异」并入 PRO 的 P2（破坏项），P2 题面已含「像被极端/怪异处理过」。
- **T2 反转（major / degeneracy #2）。** 原 T2='有明显差异'(good=1) 在 order=0 no-op 门后恒 1 → 退化。改为 T2='几乎一模一样'，真值 `ΔE < τ_noop_low(2.5)`。order=0 仅拦 `mean<τ_noop_mean(1.5)`，单探针仍可能落 1.5~2.5 轻变化 → t2_truth 有 0/1 混合，恢复 acquiescence 检测力（硬验收见 3.10）。
- **T3 改 SSIM（blocker / degeneracy #1）。** `clip_pct>0.02` 在 768px + 温和语料下真值全库 <3% 恒 0 → 退化。改真值 `ssim < τ_ssim`：结构破坏/posterize 比「新增极端纯黑白」常见、分布更均衡，且题面从「死白死黑」改为更可被 768px 辨认的「结构破坏/色阶断裂/细节涂抹」。
- **T1b 反向锚点（major / missing_pairs #1）。** 单一 good=1 的 T1 挡不住「对 F 题答 1、R 题答 0」的纯 acquiescence。新增 T1b='只给了一张图'(good=0)，与 T1 构成 F⊕R 互斥锚点对：全-yes 串在锚点对上立即矛盾被抓。
- **PRO 题面删 banding/halo（minor / trap_signal_gaps #3）。** 768px + ~800px tad66k 主池下 banding/halo/细塑料感 VLM 不可辨。P1/P2 只保留 768px 真能判的整体过曝、大面积死黑、明显塑料感、严重整体偏色；细结构破坏交确定性 T3(ssim)。
- **COH 0 容差软对（missing_pairs #2 / degeneracy #4）。** COH 是 `coherence_score` 主输入却几乎无单探针校验。改 `CONTRA_TOL=0`（H1==H2 即记 1 次违反，>0 即 `contradiction_soft`），但 COH **永不进硬 drop**，仅作软排序与单探针剔除。

**展示乱序（位置码 → claim，反作弊，N=10，⌈10/3⌉=4）。** 由 `PRESET_POS_MAP`+`PRESET_SCRAMBLE_SEED` 固定生成；PRO/INTENT/COH 的 F/R 与正/反锚点（T1/T1b）均拉远不相邻：
```
01 P1(PRO_F)   02 I1(INTENT_F)   03 T2(CHANGED)    04 H1(COH_F)   05 T1(ANCHOR_POS)
06 P2(PRO_R)   07 T3(DESTRUCT)   08 I2(INTENT_R)   09 H2(COH_R)   10 T1b(ANCHOR_NEG)
```
> 校验：PRO 01↔06(5)、INTENT 02↔08(6)、COH 04↔09(5)、ANCHOR 05↔10(5)，全部 ≥4 且不相邻；CHANGED=03、DESTRUCT=07 为单题陷阱。每探针题集恒定（无 portrait 分支），故全 preset 共用此一份固定乱序，system prompt 可 prefix-cache。

### 3.4 System prompt（固定可缓存，双图 before/after）

```
你是严格的修图预设质检员。下面给你两张图：第 1 张是原图(before)，第 2 张是对其应用某预设后的真实成品(after)。
仅依据这两张图作答，不解释、不推理、不输出多余文字。

共有 10 道判断题，每题有一个两位题号(如 03)。逐题判断该陈述对这两张图是否为真：为真=1，为假=0。
【重要】题目之间彼此独立、没有配对或正反关系；每一题都要重新看这两张图独立判断，不要根据别的题的答案来推断本题。必须每题都答；看不准也要给出最可能的判断，不得留空。

题目（注：线上 prompt 按 PRESET_POS_MAP 乱序、用两位位置码逐题列出；PRO/INTENT/COH 的 F/R 与正/反锚点均拉远不相邻；下面按维度顺序列出仅为审计可读）：
P1：after 是专业、克制的修饰：整体影调与色彩自然，没有明显的整体过曝死白、大面积死黑、明显塑料感、或严重整体偏色。
P2：after 存在破坏性问题：大面积过曝死白或死黑、明显塑料感、严重整体偏色、或像被极端/怪异地处理过（之一或多项）。
I1：after 相对 before 有明确、可学习的编辑方向（可清楚看出在做调色/影调/对比的处理）。
I2：after 几乎等于 before、看不出实质编辑（近似 no-op）。
H1：after 相对 before 的调整是一个连贯一致的整体 look：色调与影调走向统一，像同一套专业预设作用于全图。
H2：after 的调整随机或局部失控：不同区域色调走向互相矛盾、有突兀的局部色块/断裂，不像统一的 look。
T1：上面确实给了我两张图(before 和 after)。
T1b：上面只给了我一张图。
T2：after 与 before 几乎一模一样、看不出肉眼可辨的差异。
T3：after 出现明显的结构破坏 / 色阶断裂 / 细节涂抹涂糊（不是自然的调色）。

作答格式：每题写出"题号+答案"并紧贴(题号 03 答真写 031，答假写 030)，题间用一个空格，
按题号从小到大，每个题号恰好出现一次。只输出这一串，不要任何其它文字。
示例：011 020 030 041 050 …(两位位置码紧贴 1/0)
```

### 3.5 输出格式（ID 锚定 + 示例串 + 解析正则）

单一可缓存格式（共享约定 §1.1/1.2）：每探针逐题输出「两位位置码+答案」紧贴，题间一个空格，按位置码升序，每码恰一次。题号去语义：模型看不到 P/I/H/T，10 道题（含正/反锚点）由 `PRESET_POS_MAP`+`PRESET_SCRAMBLE_SEED` 散布到不相邻的位置码。**下方 P1/T1b… 为审计 ID，仅便于阅读；线上发的是固定乱序的位置码。**

- **EXPECTED_POS** = 本探针应发的 10 个两位位置码集合（经 `PRESET_POS_MAP` 一一对应 claim `P1,P2,I1,I2,H1,H2,T1,T1b,T2,T3`；`T1b` 现只是又一个普通位置码对应的反向锚点 claim）。
- **好例（审计 ID 视角）**：PRO 专业 P1=1,P2=0；确有编辑 I1=1,I2=0；连贯 H1=1,H2=0；正锚 T1=1、反锚 T1b=0；明显变化 T2(CHANGED=近似一样)=0；无破坏 T3=0。对应位置码好例串（按 §3.3 展示乱序）：`011 021 030 041 051 060 070 080 090 100`。
- **坏例 A**（近-no-op 探针上乱答"有编辑"）：实际 ΔE 落 no-op 带而 T2(CHANGED) 答 0 → `trap:T2`；I1=1∧I2=0 与真值冲突另由投票稀释。
- **坏例 B**（纯 acquiescence）：正锚 T1 与反锚 T1b 同答（T1==T1b）→ `contradiction_anchor`。隐藏配对后这是抓"对所有题统一答 1/0"的主力。
- **解析**（统一数字位置码，取消旧 `T1b` 字母后缀特例）：

```python
pairs = re.findall(r'(\d{2})([01])', s)            # [(pos, bit), ...]
bit   = {p: int(b) for p, b in pairs}
ok_parse = (set(bit) == EXPECTED_POS and len(pairs) == len(EXPECTED_POS))   # 无缺/重/未知；唯一
d = restore_claims(bit, PRESET_POS_MAP)            # 位置码 → {'P1':..,'T1b':..,'T2':..,..}; 下游 §3.6 不变
# 失败细分：missing / dup / unknown
```

落库 `raw=` 原始位置码串（每探针一行，见 3.11）。

### 3.6 单探针清洗器（有序门，伪码引用精确字段名）

```python
# 入参: probe(含 probe_id, paired_metrics dict, after_iqa dict, engine), preset(含 engine)
# 阈值按 engine 选 (LUT vs param), 见 3.9
def clean_probe(probe, preset, cfg):
    pm  = probe.paired_metrics          # {'delta_e2000_mean','delta_e2000_p95','ssim',
                                        #  'hist_emd_L','hist_emd_ab','clip_pct','noop_score'}
    eng = preset.engine                 # 'param'(LR) | 'lut'(numpy trilinear)
    tau_change   = cfg.TAU_CHANGE_LUT   if eng=='lut' else cfg.TAU_CHANGE      # T 文案用; T2 用 noop_low
    tau_noop_low = cfg.TAU_NOOP_LOW_LUT if eng=='lut' else cfg.TAU_NOOP_LOW
    tau_ssim     = cfg.TAU_SSIM         # 结构破坏阈, engine 无关(ssim 已归一)

    # —— 门1: 解析门（两位位置码解析 + POS_MAP 还原→{claim:bit}, 见 3.5）——
    d, ok, reason = parse_pos(probe.raw, cfg.PRESET_POS_MAP)   # reason∈missing|dup|unknown; d 以 claim(P1/T1b/..) 为键
    if not ok:
        return mark(probe, reliable=False, reason=f'parse:{reason}')

    # —— 门2: 锚点门 (F⊕R 互斥, 抓未看图/全-yes acquiescence) ——
    if d['T1'] != 1 or d['T1b'] != 0:
        return mark(probe, reliable=False, reason='anchor')
    if d['T1'] == d['T1b']:                        # 同答即锚点矛盾(纯 acquiescence)
        return mark(probe, reliable=False, reason='contradiction_anchor')

    # —— 门3: 确定性陷阱门 (硬, 真值=本探针 paired_metrics) ——
    t2_truth = 1 if pm['delta_e2000_mean'] <  tau_noop_low else 0   # T2 反转: 近似=真
    t3_truth = 1 if pm['ssim']             <  tau_ssim     else 0   # T3: 结构破坏=真
    if d['T2'] != t2_truth:
        return mark(probe, reliable=False, reason='trap:T2')        # ΔE 显示有/无差异却反答
    if d['T3'] != t3_truth:
        return mark(probe, reliable=False, reason='trap:T3')        # ssim 显示破坏却反答

    # —— 门4: 正反矛盾门 ——
    if d['P1'] == d['P2']:                          # PRO 硬对必 F⊕R
        return mark(probe, reliable=False, reason='contradiction_hard:PRO')
    if d['I1'] == d['I2']:                          # INTENT 硬对必 F⊕R (I2 已是 I1 严格否定)
        return mark(probe, reliable=False, reason='contradiction_hard:INTENT')
    coh_viol = 1 if d['H1'] == d['H2'] else 0       # COH 软对, CONTRA_TOL=0
    if coh_viol > cfg.CONTRA_TOL:
        return mark(probe, reliable=False, reason='contradiction_soft:COH')

    return mark(probe, reliable=True, reason=None)


# —— 门5: reliable 汇总 + 按 reason 分流的重问 (折叠 minor: 重问策略分流) ——
REASK_REASONS = {'parse', 'anchor'}                 # 仅格式抖动类值得 temp=0 重问
def settle_probe(probe, preset, cfg):
    clean_probe(probe, preset, cfg)
    if probe.reliable:
        return
    base = probe.reason.split(':')[0]
    if base in REASK_REASONS and cfg.REASK_MAX >= 1:
        probe.raw = reask(probe, temp=0.0)          # 确定性重问, 题序不变
        clean_probe(probe, preset, cfg)
    elif base in ('trap', 'contradiction', 'contradiction_anchor', 'contradiction_hard',
                  'contradiction_soft'):
        # 模型在 temp≈0 下确定性答错: 重问需改输入条件才有意义
        if cfg.REASK_MAX >= 1:
            probe.raw = reask(probe, temp=0.3, shuffle=True)   # 打乱题序+轻提温
            clean_probe(probe, preset, cfg)
        # 仍 False → 不再消耗调用
    if not probe.reliable:
        probe.status = 'qa_unreliable'              # 落库, 退出投票
```

> order=3 的硬剔除前提：T2/T3 真值已在 200-preset 试跑上验证落 `[3%,97%]`（3.10）。未通过验证前，该陷阱**降级为软标记**（记 `trap_fail` 但不剔除探针），防止把退化信号当硬门无差别放行。

### 3.7 跨探针聚合（仅 reliable 探针参与）

```python
def aggregate(preset, cfg):
    R = [p for p in preset.probes if p.reliable]
    if len(R) < cfg.MIN_RELIABLE_PROBES:            # 3
        return None                                 # → 3.8 裁 review(insufficient_reliable_probes)

    # 逐题(维)在 reliable 探针上算"好答案"通过率
    pro_pass_rate    = mean(1 if (p.d['P1']==1 and p.d['P2']==0) else 0 for p in R)
    intent_pass_rate = mean(1 if (p.d['I1']==1 and p.d['I2']==0) else 0 for p in R)
    coh_pass_rate    = mean(1 if (p.d['H1']==1 and p.d['H2']==0) else 0 for p in R)

    vote_PRO    = 1 if pro_pass_rate    >= cfg.VOTE_THRESH     else 0   # 0.5
    vote_INTENT = 1 if intent_pass_rate >= cfg.VOTE_THRESH     else 0   # 0.5
    vote_COH    = 1 if coh_pass_rate    >= cfg.COH_VOTE_THRESH else 0   # 0.66 (look 须多数场景稳定)

    # 一致性分(连续, 粒度 1/len(R)): look 在多内容上的稳定性
    coherence_score = cfg.W_PRO * pro_pass_rate + cfg.W_COH * coh_pass_rate   # w_pro=w_coh=0.5
    # 确定性旁证(纯 paired_metrics, 不靠 VLM): 色彩偏移方向在不同内容上是否发散
    emd_ab = [p.paired_metrics['hist_emd_ab'] for p in R]
    edit_direction_dispersion = stdev(emd_ab) / (mean(emd_ab) + 1e-6)

    return dict(R=R, pro_pass_rate=pro_pass_rate, intent_pass_rate=intent_pass_rate,
                coh_pass_rate=coh_pass_rate, vote_PRO=vote_PRO, vote_INTENT=vote_INTENT,
                vote_COH=vote_COH, coherence_score=coherence_score,
                edit_direction_dispersion=edit_direction_dispersion,
                reliable_probe_count=len(R))
```

- **仅 reliable 探针参与**：被解析/锚点/陷阱/矛盾门剔除或重问后仍 False 的探针**不污染**投票（抗单探针幻觉）。
- **一致性分定义**：`coherence_score = 0.5·pro_pass_rate + 0.5·coh_pass_rate ∈ [0,1]`，粒度 `1/len(R)`（6 探针时取值 {0, 1/6, …, 1}）。=1 表 look 在全部 reliable 内容上专业且连贯；中间值表 look 在某些内容翻车（确定性旁证 `edit_direction_dispersion` 越大越不像统一 look）。**降级语义下**它反映「6 张多样内容上的稳定性」，而非「6 类色调挑战」——见 3.2/3.12。
- **可信探针数下限**：`len(R) < MIN_RELIABLE_PROBES(3)` → 不投票、不裁决，直接 `review(insufficient_reliable_probes)`。任何 unreliable 探针**不计入** pass_rate 分母，故不会拉高通过。

### 3.8 判级映射（all-pass→review，preset 永不自动 keep；near_noop→drop）

```python
def map_verdict(preset, cfg):
    # —— order=0: preset 级前置确定性 no-op 门 (不调 VLM), engine-aware ——
    eng       = preset.engine
    tau_noop  = cfg.TAU_NOOP_MEAN_LUT if eng=='lut' else cfg.TAU_NOOP_MEAN   # 1.5 / 2.5(LUT)
    previews  = preset.previews
    mean_de   = mean(pm['delta_e2000_mean'] for pm in previews)
    all_noop  = all(pm['noop_score'] == 1 for pm in previews)               # ΔE<1.5 且 ssim>0.99
    # 以 noop_score 为主判(对 LUT 更稳), mean_de 阈仅作 engine-aware 辅助
    if all_noop or mean_de < tau_noop:
        set_fields(preset, pass_c=0, auto_verdict='drop', verdict_reason='near_noop')
        return                                                              # 跳过全部 VLM 探针

    agg = aggregate(preset, cfg)
    if agg is None:
        set_fields(preset, pass_c=0, auto_verdict='review',                 # pass_c 恒 0/1, 不用 None
                   verdict_reason='insufficient_reliable_probes')
        return

    # —— 硬否决 (PRO/INTENT): κ-gate 默认关, 未标定前一律 review(pending_kappa) ——
    if not agg['vote_PRO']:
        v = 'drop' if cfg.PRO_DROP_ENABLED else 'review'
        set_fields(preset, pass_c=0, auto_verdict=v,
                   verdict_reason='not_professional' if cfg.PRO_DROP_ENABLED else 'not_professional:pending_kappa')
        return
    if not agg['vote_INTENT']:
        v = 'drop' if cfg.INTENT_DROP_ENABLED else 'review'
        set_fields(preset, pass_c=0, auto_verdict=v,
                   verdict_reason='no_intent' if cfg.INTENT_DROP_ENABLED else 'no_intent:pending_kappa')
        return
    # —— COH: 软信号, 永不硬 drop ——
    if not agg['vote_COH']:
        set_fields(preset, pass_c=0, auto_verdict='review', verdict_reason='incoherent_look')
        return
    # —— all-pass: preset 永不自动 keep, 仅 review 待人工 Web UI 终裁 ——
    set_fields(preset, pass_c=1, auto_verdict='review', verdict_reason='all_pass')
    # local-mask preset(has_local_mask) 仍走既有 preset_needs_local_render/review 分支, 不变。
```

要点：（1）**pass_c 恒 0/1**（不足时 `auto_verdict='review'` 表达，绝不写 None，避免污染 `apply.py`/Web UI）。（2）**κ-gate 显式开关**：`PRO_DROP_ENABLED/INTENT_DROP_ENABLED` 默认 `False`，标定前 PRO/INTENT 不过映射 `review` 而非 `drop`，`verdict_reason` 带 `:pending_kappa`，杜绝下游误把 `drop` 当可批量删除。（3）**preset 永不自动 keep**：all-pass 也只到 `review`。（4）order=0 以 `noop_score` 为主判 + engine-aware `tau_noop`，避免温和 LUT 被系统性误杀。

### 3.9 config 常量

| 名 | 默认 | 含义 |
|---|---|---|
| `PRESET_PROBE_COUNT` | 6 | 固定探针数（已在 config.py）；`probe_id` 稳定→渲染缓存命中 |
| `PRESET_POS_MAP` / `PRESET_SCRAMBLE_SEED` | 见 §3.3/§3.5 | 位置码→claim 固定映射 + 乱序种子（PRO/INTENT/COH 的 F/R 与正/反锚点拉远、维度交错）；模型侧单一事实来源（反作弊 §1.1） |
| `EXPECTED_POS` | 10 个两位位置码 | 解析门期望完整位置码集（经 POS_MAP 对应 P1/P2/I1/I2/H1/H2/T1/T1b/T2/T3） |
| `TAU_NOOP_MEAN` | 1.5 | param(LR) order=0 前置 no-op 门：mean ΔE < 此值 → near_noop drop（沿用 noop_score ΔE<1.5 口径） |
| `TAU_NOOP_MEAN_LUT` | 2.5 | **LUT** 前置 no-op 门阈（LUT 全局映射 ΔE 普遍偏小，单阈会误杀；先在 LUT 子集跑 ΔE 分布回调） |
| `TAU_NOOP_LOW` | 2.5 | param T2 真值：探针 `delta_e2000_mean < 此值` → t2_truth=1（高于 1.5 前置门，留 1.5~2.5 轻变化使 0/1 混合） |
| `TAU_NOOP_LOW_LUT` | 3.5 | **LUT** T2 真值阈（同 LUT 量纲修正） |
| `TAU_CHANGE` / `TAU_CHANGE_LUT` | 3.0 / 4.0 | 仅供文案/监控参考的「明显变化」线（engine-aware）；T2 真值实际用 `TAU_NOOP_LOW*` |
| `TAU_SSIM` | 0.90 | T3 真值：探针 `ssim < 此值` → t3_truth=1（结构破坏/posterize）；**须在 200-preset 试跑验证 t3_truth∈[3%,97%] 后回调** |
| `CONTRA_TOL` | 0 | COH 软对违反容差：H1==H2 即 1 次违反，>0 才判 contradiction_soft（收紧为 0，恢复单探针一致性校验） |
| `MIN_RELIABLE_PROBES` | 3 | 投票所需最少 reliable 探针数；不足→review(insufficient_reliable_probes)，永不裁 drop/keep |
| `VOTE_THRESH` | 0.5 | PRO/INTENT 跨探针多数投票阈 |
| `COH_VOTE_THRESH` | 0.66 | COH 投票阈（2/3，更严：防只在 2/6 内容连贯的脆弱 look 蒙混） |
| `W_PRO` / `W_COH` | 0.5 / 0.5 | coherence_score 权重 |
| `REASK_MAX` | 1 | 单探针重问次数；parse/anchor temp=0、trap/contra 打乱题序+temp=0.3，仍 False 退出投票 |
| `PRO_DROP_ENABLED` | False | κ-gate：PRO 维 κ≥0.6 标定通过后才置 True，方允许 PRO 不过映射 auto_verdict='drop' |
| `INTENT_DROP_ENABLED` | False | 同上，INTENT 维 κ-gate |

### 3.10 可区分性验收指标（§7）

| 指标 | 目标 |
|---|---|
| 每道 VLM 二元题（P1/P2/I1/I2/H1/H2）全 preset×探针 yes 率 | 落 `[3%,97%]`；超出即该题退化需重写（防 C2 式坍缩） |
| **T2 真值多样性（硬验收，折叠 major）** | 进入 VLM 的探针中 `t2_truth` 的 1 占比 `≥10%` 且 `≤90%`；不满足 → T2 不得作 order=3 硬门，仅记 trap_fail |
| **T3 真值分布（硬验收，折叠 blocker）** | 全库 `t3_truth∈[3%,97%]`；不满足 → 先在 200-preset 跑 `1-ssim` 直方图回调 `TAU_SSIM`，回调前 T3 不作硬门 |
| coherence_score 分布 | 非单点堆积；在 {0..1} 上有分布，且与人工金标 keep/drop 单调相关 |
| 首问不可信探针率 | 可监控；>30% 说明题面/提示词/分辨率需改（优先**提高送判分辨率**——after 本身 1600px staged——而非降模型） |
| 硬正反矛盾率（PRO/INTENT F⊕R） | 接近 0；INTENT 重写为真二分后温和 preset 不应再被误剔 |
| 陷阱失败率（T2/T3 与真值不符） | 低；偏高说明模型未真看图，换分辨率/模型 |
| 锚点矛盾率（T1/T1b 同答） | 接近 0；偏高=纯 acquiescence 模型，必须换模型/改提示 |
| judge_kappa（PRO/INTENT 进硬 drop 前提） | 300-500 张人工金标逐题 Cohen's κ；PRO/INTENT κ≥0.6 才置 `*_DROP_ENABLED=True`；0.4≤κ<0.6 仅 review；κ<0.4 重写。COH 永不进硬 drop |
| all-pass→review 占比 vs review/drop 占比 | 均非极端；**preset 自动 keep 占比恒为 0** |
| near_noop 命中率（分 engine） | param 与 LUT 分开看；LUT 命中率不得系统性高于 param（否则 `TAU_NOOP_MEAN_LUT` 误杀，回调） |

### 3.11 schema / 落库改动

**新增确定性图像检测（probe 选取依赖，折叠 blocker / trap_signal_gaps #4）。** 在 ingest/特征阶段对 `images` 计算并落列（廉价、768px 工作图上算）：

```sql
ALTER TABLE assets ADD COLUMN mean_luma       DOUBLE PRECISION;  -- 平均亮度(0..1)
ALTER TABLE assets ADD COLUMN highlight_frac  DOUBLE PRECISION;  -- 高光占比(>=0.95 像素比)
ALTER TABLE assets ADD COLUMN shadow_frac     DOUBLE PRECISION;  -- 暗部占比(<=0.05 像素比)
ALTER TABLE assets ADD COLUMN saturation_mean DOUBLE PRECISION;  -- 平均饱和度(HSV S 均值, 复用流程1/2 is_bw_img 同一检测)
```
`resolve_probes(preset_qa.py:238)` 改：候选按 `aesthetic DESC` 取池，再按上述列在 6 slot 上取分位极值/匹配（替换 `scene` 去重）。**若不新增这些列 → probe_set 降级为「6 张高美学多样图」，3.2/3.7 按降级语义解读。**

**llm_qa 加 probe 维度（折叠 major：消除 6 探针同 ID 互相覆盖）。**
```sql
ALTER TABLE llm_qa ADD COLUMN probe_id TEXT;            -- 沿用 db.py:245 ALTER 风格
-- 唯一键: (asset_id, questionnaire, probe_id, item)
```
`db.add_qa` 增 `probe_id` 参数；`questionnaire='preset'`；**`item=` claim 审计 ID（`P1/I1/T1b…`，非模型看到的位置码）**；`answer∈{0,1}`；`raw=` 该探针原始位置码串。每 preset 落 10 题×6 探针（T1b 计入）各行由 `probe_id` 区分，不再覆盖。

**preset_previews 加逐探针清洗结果。**
```sql
ALTER TABLE preset_previews ADD COLUMN probe_reliable INTEGER;  -- 0/1
ALTER TABLE preset_previews ADD COLUMN probe_reason   TEXT;     -- parse:* | anchor | contradiction_* | trap:T2 | trap:T3 | NULL
ALTER TABLE preset_previews ADD COLUMN trap_fail       TEXT;     -- 记 T2/T3 软标记(降级期不剔除仍记录)
```

**preset 级汇总新表（pass_c 不污染、连续值有列可落）。**
```sql
CREATE TABLE IF NOT EXISTS preset_qa_runs (
  asset_id              TEXT,
  questionnaire         TEXT DEFAULT 'preset',
  pro_pass_rate         DOUBLE PRECISION, intent_pass_rate DOUBLE PRECISION, coh_pass_rate DOUBLE PRECISION,
  coherence_score       DOUBLE PRECISION, edit_direction_dispersion DOUBLE PRECISION,
  vote_pro INTEGER, vote_intent INTEGER, vote_coh INTEGER,
  reliable_probe_count  INTEGER, total_probe_count INTEGER,
  near_noop             INTEGER,           -- 前置 ΔE 门命中
  pass_c                INTEGER,           -- 恒 0/1, 绝不 NULL
  auto_verdict          TEXT,              -- drop | review | needs_local_render
  verdict_reason        TEXT,              -- ...:pending_kappa 等
  created_at            TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (asset_id, questionnaire)
);
```
`assets` 表：`pass_c` 保留 0/1，`auto_verdict` 复用现值域（drop|review|needs_local_render）。

**config.py / preset_qa.py。** 用本设计 `PRESET_QA`（system_prompt + probe_set 6 slot 确定性选取条件 + 10 题二元题库 + ID 方案）替换 `QUESTIONNAIRE_C`，删除依赖 style 的 C2；新增 3.9 全部常量。`preset_qa.py`：`_questionnaire_c` 改按探针单图对发新 system prompt（去掉 style/scene_affinity 注入），输出 ID 串；新增 `_clean_probe`/`settle_probe`/`aggregate`/`map_verdict`；`run_stage2` 的 c_results 聚合段替换为投票裁定。

### 3.12 遗留问题

1. **probe 6 类色调覆盖需一次性人工确认。** 即便新增 `mean_luma/highlight_frac/shadow_frac/saturation_mean`，分位极值选出的 6 张是否真命中肤色/高光/暗部/中性/饱和 5 类挑战仍需人工核对一次；确认前 `coherence_score` 一律按降级语义（「内容多样性下的稳定性」）解读，不得在产物文案中宣称具体色调类别。
2. **TAU_SSIM 与 TAU_NOOP_LOW 是起始猜值。** 必须先在 200-preset 试跑出 `1-ssim` 与 `delta_e2000_mean` 直方图，回调使 t3_truth∈[3%,97%]、t2_truth 的 1 占比∈[10%,90%]；未达标前 T2/T3 仅软标记不硬剔（3.6/3.10）。
3. **LUT 专属阈仍待数据定标。** `TAU_NOOP_MEAN_LUT/TAU_NOOP_LOW_LUT/TAU_CHANGE_LUT` 默认值是工程猜测，需在 LUT 子集单独跑 ΔE 分布，并分 engine 监控 near_noop 命中率确认 LUT 未被系统性误杀。
4. **COH 仅作软排序。** COH 永不进硬 drop、κ 校准前永远 review；是否单独为 COH 标金标（还是仅作 coherence_score 排序信号）取决于人工队列对 incoherent_look 的处置频率——用户已拍板 preset 永不自动 keep，COH 主要服务排序。
5. **insufficient_reliable_probes 积压风险。** 6 探针下若某极端 look 系统性使 3+ 探针不可信而全落 review，需监控 `insufficient_reliable_probes` 占比；过高则考虑提高送判分辨率或单独排查该类 look，而非放宽 MIN_RELIABLE_PROBES。
6. **768px 仍是 PRO 判力上限。** 已从题面删 banding/halo 并把结构破坏交确定性 T3(ssim)，但「明显塑料感/严重整体偏色」仍由 VLM 判；若 P1/P2 yes 率仍落退化带，优先把 probe 送判分辨率提到 after 的 1600px staged 档（验收已列），分辨率换不动再换模型。

---
---

## 4. 落地优先级与风险

### 4.1 落地优先级（与需求文档 §10 对齐，三套统一推进）

1. **离线纯规则单测（无 GPU）**：三套清洗器对构造串单测——正常串、缺/重/未知位置码、锚点失败、各陷阱不符、硬/软矛盾、反向锚点解析、preset `probe_id` 多探针不覆盖。**反作弊校验**：断言每套 `POS_MAP` 乱序满足"每对 F/R 间隔 ≥ ⌈N/3⌉ 且不相邻、维度交错"，位置码解析 `(\d{2})([01])` + POS_MAP 还原后 F⊕R 按 claim 判定正确。验证门逻辑确定性可复现。
2. **iqa 前置改造（§1.5）**：先做 P-1 `exif_transpose` + P-2 `is_bw_img` 双条件检测，重跑受影响 iqa；核验 `is_bw_img` 假阳率 < 2% 再置 `ENABLE_T_COLOR/HAS_ISBW_IMG=True`；确认三套图像 QA 跑在 iqa 之后。
3. **小样本 pilot（硬 gate）**：质量/审美各 ~200 图（混 color/bw、横/竖、portrait/非、含 tad66k 低分辨率），preset ~200 个 × 6 探针。量：**每 F/R 题全库 yes 率 ∈ [3%,97%]**（任一超带即阻断全库重跑、强制改题，对照 `B_quality` 坍缩）、`defect_count`/`merit_frac`/`coherence_score` 分布非单点堆积、首问不可信率、硬正反矛盾率≈0、陷阱失败率、preset `T2/T3` 真值分布∈带内、`τ_change/τ_ssim/τ_noop`(分 engine)回调。
4. **人工金标 κ（300–500 张/预设，分层）**：逐题 Cohen's κ。κ≥0.60 才把对应 `KAPPA_PASS/*_DROP_ENABLED` 置 True 准其进硬 drop；0.4≤κ<0.6 仅 review；κ<0.4 移除/重写。陷阱真值用确定性信号，无需人工。
5. **全库重跑 → §7 验收 → 闭环**：达标后 `apply.py` 写 cleaned 索引、`config.yaml` 指过去替换 gate（坍缩的 `B_quality/B_comp` 从 gate 移除/降权）。
6. **A/B（可选）**：方案 A（ID 锚定）vs 方案 B（定长位串）线上对比可信率/矛盾率/陷阱失败率/输出 token 成本，择优固化。

### 4.2 风险登记

| 风险 | 说明 | 缓解 |
|---|---|---|
| 可信≠正确 | 一致性自检只保证模型自洽，不保证正确；坏模型可能"自洽地错" | 任何题进硬 drop 须经 §6 金标 κ≥0.6；`*_DROP_ENABLED/KAPPA_PASS` 默认 False，标定前一律 review |
| 陷阱前置未落地 | `HAS_EXIF_FIXED/HAS_ISBW_IMG=False` 时 `T2/T_COLOR` 降级，硬陷阱减少→盲猜全过概率升高 | §1.5 优先做 iqa 改造；审美在颜色陷阱缺位时靠 `H_honesty` 诚实软陷阱补抗 acquiescence；监控 `trap_skipped/reduced_anchor` |
| tad66k 题面退化/池间偏置 | ~800px 低分辨率主池：UPSC 误读"原生小图"为"放大痕迹"、审美 F 题在策展池 yes>97% | 题面显式"原生小图≠放大痕迹""平庸写 0"；COMP/UPSC/审美 F 题**按池分层验 yes 率**；pilot 直方图硬 gate |
| LUT vs 真实 LR 量纲 | LUT(numpy trilinear) ΔE 普遍偏小，单阈会系统性误杀 | preset 阈值 engine-aware（`*_LUT` 变体）；order=0 以 `noop_score` 为主判；分 engine 监控 near_noop 命中率 |
| 审美 ROI 不确定 | 库内已有 `aesthetic`(laion)+`aesthetic_vlm`，`merit_frac` 可能无增量 | pilot 用 Spearman + keep 排序增量区分（AUC/NDCG）做 ROI 决策门：与 `aesthetic_vlm` 秩相关>0.7 或增量不显著则不铺开 |
| preset 可信探针不足积压 | 极端 look 系统性使 ≥3 探针不可信 → 全落 `insufficient_reliable_probes` review | 监控该 reason 占比；优先提高送审分辨率（after 有 1600px staged 档）而非放宽 `MIN_RELIABLE_PROBES` |
| 输出过大致失败（工程） | 单次产出超大结构化 JSON 易撞传输层 socket 断 | 线上 runner 输出仅 ~40–50 token 的 ID 串，天然规避；本文设计阶段已改用分节产出 |

### 4.3 验收口径（§7 汇总，三套通用 + 各自专项）

通用：每 F/R 题 yes 率∈[3%,97%]（有效性硬门豁免，见流程 1 §1.9；safety 维已删除）；可区分信号有分布；首问不可信率可监控；硬正反矛盾率≈0（题号去语义+乱序后矛盾率才真实反映"是否看图"）；陷阱失败率低；与确定性信号高一致。各流程专项指标见对应章节的"可区分性验收指标"小节。

---

*本文三套设计（流程 1/2/3）的题库题面与清洗器伪码均可直接据以实现；标注为"需 pilot 试跑定标"的阈值（`τ_*`、`FACE_MIN`、`DEF_*`、`MERIT_KEEP_FRAC`、`CONTRA_TOL`、κ 开关）须在 §4.1 第 3–4 步用真实数据回调后再固化。实现前建议先按 §4.1 第 1 步对三套清洗器做离线单测。*
