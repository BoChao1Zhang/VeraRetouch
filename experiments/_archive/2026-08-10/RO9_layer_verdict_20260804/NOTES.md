# NOTES — RO9-L 早期层 AUC 判别（D-32）

## 〇、落盘核对声明

本报告所有数字来自 `metrics.json`（本目录，`analyze_layers.py` 一次跑出，seed 20260804）。
下列出入已如实登记，不做粉饰：

1. **214 → 212**：`src_b6debb1de3bd1261` / `src_61a87d742f415b8d` 两源的 SAM3 主体掩膜下采样到
   16×16 后主体格占比 = 0.000（主体过小），AUC 无定义（正类为空），跳过。全部 AUC 口径为
   **n=212**；ρ_syn / ρ_shuf 口径分别为主批 n=300 / 错位批 n=60（与 G1 一致）。
2. **早期层 AUC 曲线并非本实验首次算出**：G1 的 `metrics.json.subject_auc.layer_auc_median_curve`
   已有一条逐层 AUC（reg_a 单条件对主体掩膜）。本实验的**新增量**是判据量本身 ——
   目标区域 AUC（reg_a→主体 / reg_b→**补集**）、配对 ΔAUC、空间自相关/有效秩及其零模型、
   逐案例通过率。G1 那条旧曲线（L22=0.834 等）在本报告中对应 `auc_rega_subject_light` 列，
   数值一致（L22 0.834 / L11 0.812 / L20 0.799），**不是**判据量，读法见 REPORT §四(1)。
3. **提示词长度假说是 null 结果**：见下 §三，图与文件名已按 null 命名
   （`viz/diag_prompt_length_NULL.png`），未写成「已证实是长度伪影」。
4. **viz 案例两端同筛**：成功/失败两端一律只在 `0.04 ≤ 主体面积16 ≤ 0.60` 的 172/212 源里挑
   （16×16 上主体过小/过大时 AUC 方差爆炸），不是只对成功端设门槛。规则与入选名单落在
   `metrics.json.viz_case_selection`。

## 一、实施前核实记录

- 读了任务卡引用的两处：`CLAUDE.md`（交付物规范 / REPORT.md 强制前三行 / 红线速查）、
  `docs/DECISIONS_2026-08-03.md` §七（D-30..D-35）。D-32 原文把判别条件写成「那些层的 s 对
  目标区域的 AUC」，本实验按此实现。
- **零额外前向**（任务卡硬要求）：确认所有 npz 已在盘 —— `run_regfull/stacks` 420 个
  （214 源 × 2 指令，含 6 个冒烟重叠），加上主批/冒烟/错位批共 1388 个 npz 可索引。
  未加载模型权重，未占用 GPU。`nvidia-smi` 于开工时查过：卡 0 有 4 个生产 python 进程
  （100% util），卡 1 有 1 个 `.venv-lens` 进程；**本任务两张卡都没用**。
- **未走长任务纪律**：全程 CPU，单次全量 wall-clock **107 s**（`config/run_analyze.log`），
  远低于 20 min 门槛，故不建 `job.marker` / `STATUS.md`。
- 掩膜银行 loader 直接复用 `experiments/G1_s_identifiability_20260803/analyze_g1.py` 的
  `SubjectMaskBank` 与 `luma_valid_from_image`（import 而非拷贝），保证与 G1 判据同口径：
  短边 512 bilinear → expand2square 黑边 pad → 面积均值下采样 16×16 → `≥0.5` 二值化，
  只在 valid（非 pad）格上评分。
- 复核过 `valid16` 口径：`run_regfull` 的 npz 内 `valid16` 与从源图重算的结果逐格一致
  （抽样 agree=1.0）；本脚本仍一律从源图重算，与 G1「旧批 luma16 是 center-crop 口径」的
  处置保持一致。

## 二、方法学决策（自行核实后采用，非拍板）

1. **「目标区域 AUC」的定义**。任务卡写「reg_a 指令用主体掩膜、reg_b 指令用其补集」。
   注意恒等式 `AUC(s, ¬M) = 1 − AUC(s, M)`，所以 reg_b 条件的目标 AUC = `1 − AUC(s_b, M)`。
   两条件混池取中位数得 `AUC_target`。**这正是关键**：若 s 与指令无关，则
   `AUC(s_a,M) ≈ AUC(s_b,M) ≈ 0.65`，混池后一半是 0.65、一半是 0.35，中位数必然回到 ≈0.5。
   因此 AUC_target ≈ 0.5 **不是** 「s 没有空间结构」，而是 「s 的空间结构不随指令改指」。
   同时报配对量 `ΔAUC = AUC(s_a,M) − AUC(s_b,M)`（逐图配对，免疫逐图主体先验/面积差异）。
2. **region_b_kind 分层**。`g1_region_opp.json` 里 165 源的 reg_b 字面是「the background」
   （＝主体补集，标签语义严格成立），另 47 源是「左半/右半」等 spatial 说法（与主体补集
   只是近似）。故设 background 子集为**主判据口径**、全 212 源为并列口径，两个都算、都报，
   救回判据取**两者中更宽松的一个**（`max`），杜绝事后挑口径。分层是在看过
   `region_b_kind` 计数、未看过分层后 AUC 数值时定的。
3. **白噪声判别的两个量**。
   - Moran's I（rook 4-邻，只在 valid 格间连边），白噪声期望 `E[I] = −1/(n−1) ≈ −0.006`；
   - 有效秩 `erank = exp(−Σ p_k log p_k)`，`p = σ_k/Σσ`，取 16×16 场（valid 减均值、
     invalid 置 0）的奇异值谱。
   两者各配一条**同支撑空间置换零模型**（valid 格内随机重排，32 次/场/层），
   而不是用教科书的解析期望 —— 这样 pad 形状、样本量、数值范围全部自动匹配。
4. **不进判据的探索项**：提示词长度诊断（§三）与跨案例通过率，只报不判。

## 三、待主 agent 决策 / 需要记账的两件事

1. **早期层低 ρ 的成因仍未定论（本实验只排除了一个解释）**。开工时的工作假说是
   「reg_a 是长主体描述、reg_b 是短『the background』，token 数差中位 12，GL token 绝对位置
   随之移动 → 早期层 attention logit 被 RoPE 位置项主导」。**实测否定**：
   `Spearman(ρ_region_opp, |Δtoken数|)` 在 L0 = **−0.036**、L4 = **−0.028**（≈0，散点无趋势），
   反倒是 canonical 层 L11 = −0.128（但那里 ρ 已饱和在 0.99，无解释力）。
   已按 null 结果落盘（`viz/diag_prompt_length_NULL.png`）。
   **实际观察到的是**：L0/L3/L4 的 s 场呈明显**行状条带**（见任一 `*_sfield_*.png` 的
   L0/L3/L4 两行），Moran's I 0.45/0.39/0.23 显著高于零模型 −0.007，erank 6.0/8.0/7.9
   低于零模型 9.0 —— 即**空间上有结构，但结构是行方向的条带，不是物体形状**；同时
   ρ_syn(L0)=0.025 说明换个同义说法这条带就全变。
   → **建议**：这不影响 RO-9 的判死（判据挂在 AUC 上，24/24 层不过线），但若 RO-3
   （全层全头扫描）要解释早期层，条带的来源（head-mean 前的单头结构？位置编码？
   D-0 插值残留？）值得单独一行。本实验**不下结论**。
2. **RO-9 的降级用途要不要保留**。本实验给出正面证据的只有一件事：s 是**图像驱动的主体
   显著场**（`AUC(s_a, M)` 在 L22 = 0.834、L11 = 0.812、canonical L8–15 = 0.648，
   luma 基线 0.465），且**与指令无关**（reg_b 条件同样 0.822/0.813/0.642）。
   按 D-32 的措辞这属于「降为 analysis 素材」。是否要把它写进论文（例如作为
   「VLM 的 special token 自发对齐主体」的观察）还是彻底弃用，属决策，留主 agent。
   **保守默认**：只入 analysis，不进任何主张 where 能力的论证链。

## 四、红线自查（对照 CLAUDE.md 速查表）

- 本实验不训练、不改渲染器，涉及的红线只有两条，均已守住：
  - **「s 禁逐图 min-max/softmax 归一化」**：全程用原始 pre-softmax head-mean logit；
    Pearson 与 AUC 都是仿射/单调不变量，未做任何逐图归一化。
  - **「VLM 干预对象 = 整段 image tokens 非 last token」**：本实验只读不干预；读出的是
    GL token 对**整段 256 个 image token** 的 attention logit（沿用 `ro9_gl_attention.py`）。
- **「每个消融行必带 Δ_const/Δ_shuffle 列」**：本实验非消融表，但等价对照已给全 ——
  常量基线 = AUC 0.5 与 luma 基线 0.465（图 A 虚线）；shuffle 对照 = `rho_shuf` 逐层曲线
  （图 C 灰线），且 `rho_shuf ≈ rho_syn ≈ rho_region_opp` 逐层成立。
- attention 导出为 eager/pre-softmax：由 `ro9_gl_attention.py` 在 G1 阶段保证，本实验只读 npz。

## 五、复现

```bash
cd /home/bc/VeraRetouch
.venv-lens/bin/python experiments/RO9_layer_verdict_20260804/analyze_layers.py \
  --n-perm 32 --viz-n 4 --out experiments/RO9_layer_verdict_20260804
```

约 107 s，纯 CPU，不需要 GPU、不加载模型权重。环境与 commit 见 `config/env.json`。
