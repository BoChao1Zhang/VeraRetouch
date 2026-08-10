# RO-X1 · CLIP 侧半场 · NOTES

追加任务（主 agent 2026-08-03 派工），承 RO-1 的结果而来。实验编号 EXPERIMENTS_v3 §2.1 表下 **RO-X1**。

## 〇、读了哪些文档节

- `docs/EXPERIMENTS_v3_2026-08-02.md` §2.1 表下 **RO-X1** 说明（原文：「无名词半集上 RO-9/RO-3（VLM 系）
  vs RO-1/RO-4（CLIP 系）的结构性分离——**VLM 不可替代性的证明实验**」）
- `docs/DATA_ASSIGNMENT_2026-08-02.md` §3.3 **RO-X1 行**（「名词半集 = D-SFT-L(S-val) 指令 200 条；
  无名词半集 = D-SFT-G(S-val) 风格指令 200 条（人工核一遍确无名词）」）—— **本实验证否了这一行**，见 REPORT §三
- `experiments/RO1_selfself_20260803/REPORT.md`（沿用最优配置与全部评分口径）
- `experiments/G1_s_identifiability_20260803/REPORT.md`（ρ_syn / ρ_shuf 的口径与 RO-9 的对照数字）

## 一、无需新的在线核实

本实验**没有引入任何新的外部事实**：算子、权重、超参、评分口径全部继承 RO-1（其核实记录见
`experiments/RO1_selfself_20260803/NOTES.md` §一，含三个官方仓库的 commit、arXiv 2506.08010、
open_clip pin、FeatUp 签名）。唯一新增的第三方件是 **NLTK 3.9.4 的 POS tagger**，只用于
Arm A 的**切分验证**（不参与任何判据），且已用正则分词绕开 `punkt_tab` 数据依赖（见 §四）。

## 二、设计上的两个关键选择（写在跑数前）

### 2.1 为什么必须做 Arm B（构造式无名词短语）

DATA_ASSIGNMENT 假设「D-SFT-G 风格指令 = 无名词」。**跑之前我不知道这条成不成立**，
所以预注册了作废条件（§一的判据表 ④）并同时准备了 Arm B。结果作废条件触发
（严格无名词真实指令仅 2/86），**Arm B 成了唯一有效的判据臂** —— 这正是预注册的价值。

### 2.2 为什么主口径的无名词短语取 `"the main subject"`

它是 5 条无名词短语里**对 CLIP 最有利**的一条（指向区域，只是没有具体名词）。
**用最有利的一条得到「不分离」，负结论才最强**；反之若挑 `X_tonal`（0.605）当主口径，
就会得到一个被口径选择制造出来的"分离"。这条选择在跑数前写死在 `PRIMARY_NOUNFREE`。
**但它同时是本实验最大的口径风险**，故列入 §五待决策 UX-1。

### 2.3 GT 与文本的对应关系（防止无意义比较）

Arm B 的**全部**条件都在**同一张 GT（SAM3 主体掩膜）**上评分，因为它们**都指向同一个区域**
（主体）或其补集。这样 ΔAUC 才是"只换措辞"的净效应。
Arm A 的 style 半集**不满足**这个前提（全局指令没有局部 GT），故该半集的 AUC 只作描述，
承重量改用不依赖 GT 正确性的 ρ_shuf / sep。

## 三、红线自查

| 红线 | 落实 |
|---|---|
| s 禁逐图归一化 | 继承 RO-1：唯一口径 = 原始余弦；本实验不写 scache，无归一化环节 |
| 每个消融行必带 Δ_const / Δ_shuffle | `N_shuf`（跨图错位名词）与 `R_shuf`（跨图错位真实指令）两列全程同报；**无名词族的 Δ_shuffle 在定义上恒为 0**（文本对所有图相同），这一点已在 REPORT §二第 3 点显式写明，不是遗漏 |
| 符号 sanity check | 继承 RO-1（三算子全 PASS）；本实验另报每条件的 `frac_gt_half`，无名词族全部 >0.5，无需翻转 |
| 禁 ad-hoc 切分 | 切分依据是 journal 的结构化字段 `task_type`，源清单直接复用 `g1_samples.json`（S-val 300 源）与 RO-1 的 `scoring_set.json`，**未重新采样** |
| 不 kill 别人进程 / 卡 1 / ≤20GB | 仅 `CUDA_VISIBLE_DEVICES=1`，显存峰值 **1.01 GB**；nohup + `job.marker` |
| 表述纪律 D-31 | REPORT §五逐条标注「未获证据/被证否」与「获得正面证据」 |

## 四、跑挂记录（如实登记）

**第一次启动（pid 351781）崩溃**：`nltk.word_tokenize` 需要 `punkt_tab` 数据包，本机没有，
在 D2 检测器处抛 `LookupError`。日志片段：

```
Please use the NLTK Downloader to obtain the resource:
>>> nltk.download('punkt_tab')
Attempted to load 'tokenizers/punkt_tab/english/'
```

**处理**：没有去下载数据包（增加一个运行期外部依赖不划算），改为用本文件已有的正则
`_TOK = re.compile(r"[A-Za-z][A-Za-z\-']*")` 自行分词 —— `nltk.pos_tag` 只需要 token 列表。
同时借这次修改补上了**比较级误标守卫**：POS tagger 会把摄影语境里的 "cooler"/"darker"/"warmer"
误标成 NN，规则改为「NN/NNS ∧ 不在停用表 ∧（不像比较级 ∨ 在 D1 词表里）」
（"flower"/"water"/"tower" 这类真名词都在 D1 词表里，不会被误杀）。
第二次启动（pid 361160）跑通，wall-clock 554s。

## 五、待主 agent 决策

| # | 决策点 | 保守默认 | 影响 |
|---|---|---|---|
| **UX-1** | **主口径的无名词短语取哪一条** | 取**对 CLIP 最有利**的 `X_deictic`（"the main subject"，AUC 0.907）→ 得到"不分离"的**负结论** | 若改取 `X_intent`（0.773）或 `X_tonal`（0.605），主判据①②会翻成"分离"。**结论的方向取决于这一条**，必须由主 agent 拍板并写进 EXPERIMENTS_v3，不能事后挑 |
| **UX-2** | **RO 系主判据是否从 AUC 改成 AUC_target** | 本实验只提出建议（X1），未擅自改判据 | 本实验给出了 AUC 会骗人的直接反例（AUC 0.907 / AUC_target 0.523）。改判据会影响所有已跑与在跑的 RO 臂的解读 |
| **UX-3** | **RO-X1 的无名词语料换成什么** | 本轮用构造式 5 条固定短语；未去人工筛 D-SFT-G | DATA_ASSIGNMENT §3.3 该行已被证否（严格无名词 2/86）。选项：① 构造式（零成本、可控、但非真实用户语言）② 人工筛真实语料（贵、量未知）③ 另造无名词指令语料（需 LLM 改写 + 人工核） |
| **UX-4** | **style 半集的 AUC 该不该进任何汇总表** | 本轮只作描述、不承重，并在 REPORT §七显式标注"GT 不是正确答案" | 若主 agent 认为该半集需要正式判据，得先定义全局指令的 GT（例如"全图均匀"的偏离度），那是新的度量设计 |

## 六、可复现性

- `run_rox1.py` 一个文件跑完全部；预注册常量在文件头（`PREREG` / `NOUNFREE` / `PRIMARY_NOUNFREE`）。
- seed 20260803（错位捐赠者的确定性错排；**只在同半集内错排**，避免把 style 的文本扣到 local 上）。
- register neurons **直接复用 RO-1 在 100 张 S-train 图上的检出结果**（写死在脚本里，
  见 `metrics.json.prereg` 与 RO-1 的 `metrics.json.register_neurons`），不重跑检测。
- `config/split_examples.json` 存了 style/local 各若干条真实指令与其 D1/D2 命中词，切分可逐条核。
