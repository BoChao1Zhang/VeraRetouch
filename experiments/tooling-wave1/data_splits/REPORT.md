# REPORT — T1：S/P split 旁表物化 + 数据核查三件套

- 日期：2026-08-03（初版 2026-08-02 晚，l4 归档后全量重生成）
- 依据：DATA_ASSIGNMENT §1（行动项 C）、§4 行动项 A/G；实现见 `tools/data_splits/`
- 环境：python 3.13.5（/home/bc/miniconda3，仅标准库）；git commit `0e5d04a`（lens-exp）
- Split seed（冻结）：`verasplit-v1`；规则全文见 `tools/data_splits/README.md`

## 1. 目标与设置

把 DATA_ASSIGNMENT §1.1 的 S-split（源级）与 P-split（preset 级）物化为只读旁表
（`tools/data_splits/splits.sqlite3` + 同名 CSV），并固化两项核查：PPR10K 无污染复核、
行动项 G（弃权组渲染产物可用性 + D-RENDER 规模数）。

**输入**：journal 归档内全部完成 build（判据 = 归档存在 groups.jsonl），共 **7 个**：
g1/g2/g3（各 25,000 组）+ l1/l2/l3/l4（各 17,000 组）= 143,000 组 × 8 候选。
注：l4 归档出现于 2026-08-02 23:55（本报告初版旁表生成之后），已全量重生成纳入；
g4、l5、l6 无 journal 归档（在途），不纳入，续入后重跑 `build_splits.py` 即可。

## 2. 预注册判据 vs 实测

| 判据（任务卡） | 实测 | 结果 |
|---|---|---|
| split 表覆盖全部完成 build 的 source_id 与 preset_id | 33,652/33,652 源、3,522/3,522 preset，missing=0（selfcheck 独立重扫） | ✅ |
| 同 source 跨 build 恒同 split（抽查 100 例） | 多 build 共现源池 30,768 个，抽 100 例逐 build 独立重算，0 不一致（构造上纯函数） | ✅ |
| 统计报告落盘 | 本文件 + `stats_sources_pool_split.csv` + `stats_presets_minor_split.csv` + `ppr10k_verify.{json,md}` + `action_g_report.{json,md}` + `build_splits_summary.json` | ✅ |
| S 比例 ≈90/5/5 | train/val/test = 30,229/1,723/1,700 = 89.8%/5.1%/5.1% | ✅ |
| P 比例 ≈90/5/5（minor 分层） | 3,172/175/175（DATA_ASSIGNMENT 预估 ≈3,170/176/176） | ✅ |
| PPR10K 已知结论复核（索引 1–8871，≥8875 零命中） | 4,311 不同源，索引 1–8871，≥8875 命中 **0**，exit 0 | ✅ |
| 存储 split == 独立重算（S 与 P 两表） | mismatches=0（selfcheck 7/7 PASS） | ✅ |

## 3. 源 pool × split（S-split，源数量）

| pool | train | val | test | total |
|---|---|---|---|---|
| awards | 5,979 | 331 | 347 | 6,657 |
| fivek_gold | 2,092 | 117 | 129 | 2,338 |
| greysky | 78 | 5 | 3 | 86 |
| korean | 1,794 | 95 | 80 | 1,969 |
| mmart_ppr10k | 3,285 | 195 | 194 | 3,674 |
| ppr10k | 3,897 | 208 | 206 | 4,311 |
| quandian | 1,918 | 113 | 111 | 2,142 |
| raise6k | 2,207 | 118 | 121 | 2,446 |
| unsplash | 8,979 | 541 | 509 | 10,029 |
| **TOTAL** | **30,229** | **1,723** | **1,700** | **33,652** |

无 pool 冲突（同 source_id 跨 build 池归属恒同，conflicts=0）；无 `other` 池残留。

## 4. preset minor × split（P-split）

全表 77 个 minor 层见 `stats_presets_minor_split.csv`（TOTAL 3,172/175/175）。要点：

- 层内规则：`preset_id` 升序，层尾 `floor(n*0.05+0.5)` 个 = test，再前同数 = val；
  `preset_id → minor` 绑定全量扫描 0 冲突。
- 头部层（青橙胶片_00 = 363、青橙胶片_01 = 270、低饱和复古_03 = 200）val/test 各 18/14/10。
- **28 个 minor 层 n<10，无 val/test 名额**（如 高亮复古_09、高饱和暖调_06 各 n=1）——
  P-val/P-test 覆盖 49/77 层；E19/E20 报「未见 LUT」分层指标时空层无样本，已在
  NOTES.md 待决策 #3 提请（保守默认维持现规则）。

## 5. PPR10K 无污染复核（行动项 A 固化）

- `ppr10k/source`：4,311 不同源、17,706 组占用；文件索引 **1–8871**；
  官方 val 段（≥8875，README "train with the first 8,875 files"）命中 **0** → **干净**。
- **新发现（原 §4-A 未覆盖，提请主 agent 裁决）**：`mmart_ppr10k` 池 3,674 源
  （MMArt-PPR10k 基于 PPR10K 原图，目录 `<gid>_<pid>`，gid 1–1680），其中 gid≥1356
  （论文口径 train 组数边界，advisory、未逐字核实）**326 源**疑似落 PPR10K 官方 val 组段。
  若 E20 报官方口径，建议将 mmart_ppr10k 源从相关训练口径剔除或强制 S-train（待拍板）。

## 6. 行动项 G：弃权组渲染产物可用性 + D-RENDER 规模

**结论：弃权组（winner_margin<1.0，未出 SFT 行）的渲染产物全部落盘可用。**

- 语义核实（l3 全量）：abstain 组 margin ∈ [0, 0.977] 全 <1.0；low ∈ [1.001, 1.978]；
  abstain/null 组 group_id 在 sft.jsonl 命中 **0**（前提成立）。
- 落盘核实：`datasets/groups/<build>` shards 索引全扫，journal 内 143,000 组 × 8 =
  **1,144,000 候选 after .jpg 100% 落盘**（l 系另 100% 带逐候选 `.cgt.png` 掩膜，544,000 张）；
  物理抽验 tar 内 JPEG 完整（SOI/EOI）、vrmeta 回链 group/source/preset/confidence 一致。
- I_in 可回取：33,652 源 100% 命中 NFS img 银行（9 池全部，逐池抽样人工复核 27/27）。
- **可用渲染对总数（D-RENDER 规模数）= 1,144,000**，按置信度：
  normal 315,952 / low 217,512 / **abstain 362,616** / unannotated(null) 247,920。
  DATA_ASSIGNMENT §2 的 "≈100 万对" 预估偏保守：journal 组恒 8 候选，渲染损耗
  （visibility_rejected 等 ~74k 事件）发生在进组之前，failures.jsonl 只解释「为何某些
  preset 尝试没进组」，不扣减已进组候选。
- null（未注释终态）组默认计入 D-RENDER（渲染确定性与置信度无关）；若须剔除，
  扣 247,920 对 → 896,080 对（分项数已在 `action_g_report.json`）。

## 7. 结论与建议

1. 旁表已物化并通过全部预注册判据；下游实验一律 `SELECT split FROM sources/presets`，
   禁止 ad-hoc 切分（DATA_ASSIGNMENT §1.1 落地要求达成）。
2. g4/l5/l6 归档后重跑 `build_splits.py` + `selfcheck.py` 即可增量续入（纯函数 S-split
   保证既有源 split 不变；P-split 若出现新 preset 会引起层内重排，建议届时 diff 报告）。
3. 待主 agent 决策三项（NOTES.md）：mmart_ppr10k 疑似官方 val 重叠处置；null 组是否
   进 D-RENDER；P-split 小层无 val/test 是否接受。
