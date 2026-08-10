# 归档清单 · 2026-08-10（ARCH-1）

> **这里是上一轮 local-retouch 战役（A/E/G/RD/RO/PR + tooling wave-1）的全部实验文档与实验目录。**
> 归档只是**移动路径**，任何文件内容、结论、数字都未改动。当前战役是
> **Q3VL MetaCanvas Where/What**，其文档仍留在 `docs/` 根与 `docs/reviews/` 下，不在本清单内。
>
> 本文件同时是 `experiments/_archive/2026-08-10/` 的清单——文档和实验目录一起归档，
> 只维护这一份索引。

## 0. 当前战役保留了什么（**不在**本归档内）

| 保留路径 | 说明 |
|---|---|
| `docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` | Q3VL base SFT 规格（当前战役权威文档） |
| `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` | Where/What 最终实验协议（当前战役权威文档） |
| `docs/QUEUE_USAGE.md` | GPU 队列使用说明 |
| `docs/reviews/REVIEW-impl-{S0,WhereA,WhereB,What}.md` | 当前战役四份实现审阅 |
| `experiments/Q3VL_metacanvas_where_what_20260804/` | 当前战役实验目录 |
| `docs/archive/`、`docs/assets/`、`docs/presentation_2026-08-05/` | 更早的历史归档 / 图片资产 / 两周汇报，本次未动 |

## 1. 文档：原路径 → 新路径

### 1.1 战役权威文档（原 `docs/` 根，git mv 保历史）

| 原路径 | 新路径 | 一句话说明 |
|---|---|---|
| `docs/PLAN_v2_local-retouch_2026-07-31.md` | `docs/_archive/2026-08-10/PLAN_v2_local-retouch_2026-07-31.md` | 上一轮方法、判据、文献依据与红线总纲 |
| `docs/EXPERIMENTS_v3_2026-08-02.md` | `docs/_archive/2026-08-10/EXPERIMENTS_v3_2026-08-02.md` | 实验臂定义、晋级/淘汰判据、排期（活文档，含 changelog） |
| `docs/DATA_ASSIGNMENT_2026-08-02.md` | `docs/_archive/2026-08-10/DATA_ASSIGNMENT_2026-08-02.md` | 数据集代号、split 纪律、实验×数据表 |
| `docs/IMPL_DOSSIER_2026-08-02.md` | `docs/_archive/2026-08-10/IMPL_DOSSIER_2026-08-02.md` | 复现规格、竞品配方、接入手册，**附录 B 已核实链接表** |
| `docs/SURVEY_papers_2026-07-31.md` | `docs/_archive/2026-08-10/SURVEY_papers_2026-07-31.md` | 579 篇论文底账（官方摘要） |
| `docs/DATA_ASSETS_2026-08-02.md` | `docs/_archive/2026-08-10/DATA_ASSETS_2026-08-02.md` | 数据字段与历史七段契约 |
| `docs/DECISIONS_2026-08-03.md` | `docs/_archive/2026-08-10/DECISIONS_2026-08-03.md` | 上一轮决策记录 |
| `docs/HANDOFF_2026-08-04.md` | `docs/_archive/2026-08-10/HANDOFF_2026-08-04.md` | 上一轮交接说明 |
| `docs/EXPERIMENT_INDEX.md` | `docs/_archive/2026-08-10/EXPERIMENT_INDEX.md` | 上一轮实验文档入口页（指向注册表 / 结果页 / 协议） |

### 1.2 结果视图（原 `docs/` 根，**未入 git**，随目录移动）

| 原路径 | 新路径 | 一句话说明 |
|---|---|---|
| `docs/EXPERIMENT_REGISTRY.md` | `docs/_archive/2026-08-10/EXPERIMENT_REGISTRY.md` | 上一轮实验注册表：逐项 `what / where / status / gate` |
| `docs/EXPERIMENT_RESULTS_CURRENT.md` | `docs/_archive/2026-08-10/EXPERIMENT_RESULTS_CURRENT.md` | 上一轮结果叙事视图：什么成立、什么被证伪 |

> 文件名里的 “CURRENT” 指的是**上一轮**战役当时的现状，不是本仓库当前状态。

### 1.3 方法长文（原 `docs/methods/`，**未入 git**，随目录移动）

| 原路径 | 新路径 | 一句话说明 |
|---|---|---|
| `docs/methods/METHOD_E2_basis_fit.md` | `docs/_archive/2026-08-10/methods/METHOD_E2_basis_fit.md` | E2 基底拟合方法长文 |
| `docs/methods/METHOD_RDG_transformer.md` | `docs/_archive/2026-08-10/methods/METHOD_RDG_transformer.md` | RD-G transformer 参数生成端方法长文 |

### 1.4 旧审阅（原 `docs/reviews/`，git mv 保历史）

| 原路径 | 新路径 | 一句话说明 |
|---|---|---|
| `docs/reviews/REVIEW-impl-wave1.md` | `docs/_archive/2026-08-10/reviews/REVIEW-impl-wave1.md` | tooling wave-1 工具实现审阅 |
| `docs/reviews/REVIEW-result-W1batch1.md` | `docs/_archive/2026-08-10/reviews/REVIEW-result-W1batch1.md` | wave-1 batch1 结果审阅 |
| `docs/reviews/REVIEW-impl-dbv-nfs-1.md` | `docs/_archive/2026-08-10/reviews/REVIEW-impl-dbv-nfs-1.md` | databuild_viewer NFS 改造实现审阅（Round 3 APPROVED） |

> `REVIEW-impl-dbv-nfs-1.md` 归属存疑：它审的是 `databuild_viewer/`，与 Q3VL 战役无关，
> 但 `databuild_viewer/` 目前**仍有未提交的在做改动**。若该工作线仍在推进，
> `git mv docs/_archive/2026-08-10/reviews/REVIEW-impl-dbv-nfs-1.md docs/reviews/` 即可取回。

### 1.5 IAA 评测战役（更早一轮，git mv 保历史）

| 原路径 | 新路径 | 一句话说明 |
|---|---|---|
| `docs/iaa_benchmark/README.md` | `docs/_archive/2026-08-10/iaa_benchmark/README.md` | IAA 评测战役说明 |
| `docs/iaa_benchmark/HANDOFF_EVAL_PROMPT_2026-07-07.md` | `docs/_archive/2026-08-10/iaa_benchmark/HANDOFF_EVAL_PROMPT_2026-07-07.md` | IAA 评测 prompt 交接 |
| `docs/iaa_benchmark/IAA_FINAL_EVAL_REPORT_2026-07-08.md` | `docs/_archive/2026-08-10/iaa_benchmark/IAA_FINAL_EVAL_REPORT_2026-07-08.md` | IAA 最终评测报告 |

## 2. 实验目录：`experiments/<name>` → `experiments/_archive/2026-08-10/<name>`

24 个目录，目录名全部保持不变。

### 2.1 Render / 容量 / 基底

| 目录 | 一句话说明 |
|---|---|
| `A0_glut_repro_20260803` | GLUT 复现锚点缺口归因 |
| `E1_cube_N_20260803` | cube 扫 N：3D LUT 网格分辨率对 ΔE00 的影响 |
| `E1b_svd_20260803` | 3,522 生产 preset 的截断 SVD 秩–ΔE00 曲线 |
| `E2_basis_fit_20260803` | 14 维单轴基底拟合；薄环掩膜的沿-s 平顶带通响应 |
| `G2_oracle_ceiling_20260803` | oracle s 作为第四根轴接入逐像素色彩算子的天花板 |
| `G3_collapse_20260803` | 朴素 4D 高斯在纯重建损失下自发走进退化集 D |
| `RD_std_e_20260803` | 给定干净语义轴 s，LUT 量级逐像素算子的局部编辑容量 |
| `RDG_transformer_20260803` | 参数生成端：CGLUT 式 MLP → 5M/15M transformer |
| `lut_renderer_pilot` | LUT renderer 在线标定 pilot（只有旧协议，无可引用结果） |

### 2.2 Where 读出 / 探针

| 目录 | 一句话说明 |
|---|---|
| `G1_s_identifiability_20260803` | 冻结 VLM 中 `<retouch_light>` → image token 注意力的 s 可辨识性 |
| `G1b_difflmm_20260803` | G1 的 DiffLMM 读法复检：是读法问题还是信息不存在 |
| `RO1_selfself_20260803` | 零训练 CLIP self-self 稠密读出 |
| `RO2_logitlens_20260803` | logit lens：image token 隐状态投词表取目标语义词概率当空间场 |
| `RO3_layerhead_scan_20260803` | 逐 (层, 头) 扫描 instruction 文本 token → image token 注意力 |
| `RO9_layer_verdict_20260804` | GL token 读出拿不到 where 是否只是层选错了 |
| `RO9b_readout_fix_20260803` | RO-9 的低 AUC 是读出算子问题还是 VLM 里没有 where |
| `RO9c_subject_repro_20260805` | 三个 retouch special token 经 DiffLMM 序列均值减法（M1）的主体复现 |
| `ROW_basis_coeff_20260803` | VLM 直接吐 14 个基系数、由固定基底合成空间场 |
| `ROX1_clipside_20260803` | CLIP 是否只做「名词→区域」对齐（去可定位名词的负控制） |
| `PR13_probe_whatwhere_20260803` | VLM 内部是否同时存在 what（颜色）与 where（空间）信息 |

### 2.3 MetaCanvas query（MCQ）端到端

| 目录 | 一句话说明 |
|---|---|
| `MCQ_e2e_whatwhere_20260803` | MetaCanvas query 的 what / where 读出端到端 |
| `MCQ_full_local_l1l6_20260804` | 全量 Local L1–L6 MetaCanvas 端到端训练 |
| `MCQ_basis_where_l1l6_20260804` | L1–L6 上 basis-where 两臂（`basis_vlm14` / `basis_geo_range8`）全量对照 |

### 2.4 基础设施验收

| 目录 | 一句话说明 |
|---|---|
| `tooling-wave1` | wave-1 工具验收：`data_splits`(T1) / `cube`(T2) / `harness`(T3/INF-1) / `T4_construct`(INF-2) / `scache`(T5/INF-5) / `bgr_check`(F5) |

## 3. git 处理方式（重要）

归档目录里**只有文档类小文件留在 git**，共 **216** 个（`REPORT/NOTES/STATUS/*.md`、
`metrics.json`、`config/` 快照、分析脚本 `*.py`/`*.sh`、`*.txt`/`*.csv`/`*.marker`），
它们随 `git mv` 保留历史。

**241** 个日志与大产物（`*.png` / `*.log` / `*.pt` / `*.npy` / `*.npz` / `*.jsonl`）已
`git rm --cached` 移出 index——**文件仍在磁盘上**，只是不再入 git。仓库根 `.gitignore` 里
对应的规则是：

```gitignore
experiments/_archive/**
!experiments/_archive/**/
!experiments/_archive/**/*.md
```

`.gitignore` 只作用于未跟踪文件，所以仍在 index 里的那 216 个文件不受影响。
原本就未 track 的 14 个实验目录（`G1b` / `MCQ_*` / `PR13` / `RDG` / `RD_std_e` /
`RO1` / `RO2` / `RO3` / `RO9b` / `RO9c` / `ROW` / `ROX1`）整体随目录移动，未加入 git。

## 4. 已知仍指向旧路径的引用（**本次未改**）

以下文件属于任务卡明确划定的「不动」范围（代码包 `tools/` / `model/` / `q3vl/` /
`databuild_viewer/`），因此其中的旧路径字符串保持原样。取回或重跑这些旧工具时需要照本清单换路径。

**运行期硬编码路径（重跑旧工具会实际报错）**

| 文件 | 旧路径 |
|---|---|
| `model/glut_repro/data.py:39,41` | `experiments/tooling-wave1/cube/{inventory/dcube_manifest.jsonl, parse/parse_report.jsonl}` |
| `tools/ceiling/mlp_probe.py:25`、`tools/ceiling/run_analytic.py:34` | `experiments/tooling-wave1/T4_construct/sanity` |
| `tools/harness/smoke_real.py:36` | `experiments/tooling-wave1/harness` |
| `tools/data_splits/vr_common.py:31` | `experiments/tooling-wave1/data_splits` |

**注释 / 文档字符串里的引用（不影响运行）**

`q3vl/where/config.py:70`、`q3vl/where/scripts/sweep_upsample.py:5`（→ `E2_basis_fit_20260803`）；
`tools/readout/{ro2_logit_lens,ro3_layerhead_scan,ro9b_readout_fix,ro9c_m1_teacherforced}.py`、
`tools/probe/{colorops,vlm_features}.py`、`tools/{construct,cube,data_splits,harness,scache}/NOTES.md`、
`tools/data_splits/README.md`、`tools/harness/bake_consistency.py`、`tools/data_splits/build_splits.py`、
`model/glut_repro/__init__.py`、`databuild_viewer/NOTES.md`。

归档目录内部（各实验自己的 `NOTES.md` / `REPORT.md` / `config/*.json`）互相引用的旧路径
也一律未改——归档件保持原样，不做二次编辑。

## 5. 已同步更新的引用（本次改了路径字符串，未改语义）

| 文件 | 改了什么 |
|---|---|
| `CLAUDE.md` | 「权威文档」表五行 → `docs/_archive/2026-08-10/`，并补一句说明当前战役权威文档是哪两份 |
| `docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md` | `DATA_ASSETS_2026-08-02.md`、`EXPERIMENT_REGISTRY.md` 两条 |
| `docs/METACANVAS_WHERE_WHAT_FINAL_EXPERIMENT_PROTOCOL_2026-08-04.md` | §1.3 的 `E2_basis_fit_20260803`、`RDG_transformer_20260803` 两条 |
| `docs/reviews/REVIEW-impl-WhereA.md` | `E2_basis_fit_20260803/{e2lib.py,prep_data.py}` 两条 |
| `experiments/Q3VL_metacanvas_where_what_20260804/what/PREFLIGHT_WHAT_PENDING.md` | `RDG_transformer_20260803/tools/bake_check.py` 一条 |
| `experiments/Q3VL_metacanvas_where_what_20260804/where_a/{PREFLIGHT_WHERE_A_PENDING.md,NOTES.md}` | `E2_basis_fit_20260803` 三条 |
