# 归档清单 · 2026-08-12（ARCH-2）

2026-08-12 归档:对外汇报快照,数字止于 0.7622 基线,后续修正见 `docs/EXPERIMENT_INDEX.md`。

> 归档只是**移动路径**,任何文件内容、结论、数字都未改动(逐文件 md5 校验,239/239 一致)。

## 原路径 → 新路径

| 原路径 | 新路径 | 内容 | 数量 |
|---|---|---|---|
| `docs/presentation_2026-08-05/` | `docs/_archive/2026-08-12/presentation_2026-08-05/` | 两周进展汇报快照(BIWEEKLY_REPORT / ASSETS_* / MODELCARDS_* / ROW_FIGURE_MAP + figs*/assets*) | 208 文件 |
| `docs/archive/`(小写) | `docs/_archive/2026-08-12/july-docs/` | 七月历史文档(DATABUILD 三份 / DECISION_TREE / IAA / LUT_RENDERER / QA_ANALYSIS) | 7 文件 |
| `docs/plan/paper_digests/` | `docs/_archive/2026-08-12/paper_digests/` | 论文摘要卡 P0_*/P1_*_digest.md | 24 文件 |

移动后 `docs/archive/` 与 `docs/plan/` 变空,已删除空目录本身(仅空目录,零文件删除)。

## 引用旧路径的换算

归档内文档(七月文档之间的互引、presentation 内脚本的绝对路径、`docs/_archive/2026-08-10/`
下已冻结的审阅与结果文档)**保持原样未改**,其中出现的 `docs/archive/…`、`docs/plan/paper_digests/…`、
`docs/presentation_2026-08-05/…` 一律按上表换算到新路径。仓库内**现行**文档的指向已就地修正。

## 汇报快照的数字口径(重要)

`presentation_2026-08-05/` 是 2026-08-05 的对外快照,主榜数字止于 **0.7622**(EPR-H12,
amort_P3prime_cont)。此后主榜继承链已推进到 **0.79095**(EPR-H22,CONT2,现行基线 M0)。
引用该快照内任何 where 侧数字前,须先对照 `docs/EXPERIMENT_INDEX.md` 的 EPR-H 表与冲突裁决
记录 K1–K16(尤其 K6 强制口径规则、K8 主榜继承链)。
