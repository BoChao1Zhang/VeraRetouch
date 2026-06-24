# Preset Clean + Tag — 数据分析 (2026-06-24)

全量 preset 清洗(6 探针真实 LR/numpy 渲染 + 10 题位置码问卷 + qa_clean 投票)+ 功能 tag
(逐探针 LAB 指标 + vLLM 命名)结果。**所有结果已入库 assets 表**(见末尾"DB 字段")。

## 1. preset 总览(9943 非 dup)

| status | 数量 | 含义 |
|---|---|---|
| **preset_meta_pass** | **8027** | 通过 stage1 + 已渲染+判级(clean 完成) |
| preset_meta_fail | 1621 | stage1 剔除(参数失常/重复/近 no-op/黑白/技术图) |
| preset_meta_local | 189 | 局部 mask 预设(需带 mask 的真实 LR 路径,**未清**) |
| preset_render_failed | 62 | 真正渲染不出(源/格式问题) |
| preset_needs_local_render | 44 | 同 local-mask,待真实 LR |

→ **可清洗的全局预设 8027 已全部完成**(param 3818 + lut 4209)。

## 2. Clean verdict 分布(8027 已判)

| verdict | 数量 | 占比 |
|---|---|---|
| **all_pass** | **7778** | **96.9%** |
| not_professional:pending_kappa | 235 | 2.9% |
| near_noop | 11 | 0.1% |
| insufficient_reliable_probes | 3 | <0.1% |

- `pass_c=1`: 7778(送 review);`pass_c=0`: 249(near_noop 11 drop + not_professional 235 review + insufficient 3)。
- **preset 永不自动 keep**(设计):全部 pass_c=1 进人工 review;唯一硬 drop = near_noop(11)。
- 维度投票通过率:**PRO 97.1% / INTENT 100% / COH ~100%**。PRO 未过的 235 个=最激进 LUT(pro<0.5),
  因 PRO/INTENT/COH 硬 drop 仍 κ-gated → 全部落 review(not_professional:pending_kappa)。

### 问卷可靠性(强)
| reliable 探针/6 | preset 数 |
|---|---|
| 6/6 | 7935 (98.9%) |
| 5/6 | 67 |
| ≤4/6 | 14 |
| (near_noop 未聚合) | 11 |

→ 6 探针问卷在全量上 **98.9% 满可靠**,与 pilot R2/R3 的 100% 一致,问卷+清洗器稳健。

## 3. 功能 tag(8027,vLLM 命名)

**grade_family**(确定性):stylized 6182 / teal_orange 818 / vintage_film 471 / clean_natural 292 / bw 264。

**确定性轴分布**(色温/色罩读中性探针,饱和用相对 chroma):
| 轴 | 分布 |
|---|---|
| temperature | cool 3984 / neutral 3083 / warm 960 |
| tint | neutral 5983 / green 1430 / magenta 614 |
| saturation | muted 5181 / neutral 1891 / vibrant 691 / bw 264 |
| contrast | neutral 4271 / punchy 2311 / flat 1445 |
| tone | neutral 4649 / lifted 1732 / crushed 1646 |
| exposure | neutral 4211 / high_key 1949 / low_key 1867 |

→ 这批预设库偏 **冷调(50%)、去饱和(65%)** —— 以电影暗调/胶片 look 为主,与 pilot 一致。

**语义区分度**:`caption` **8027/8027 = 100% 唯一**;`per_probe` ≈100% 唯一;name 993 unique(粗浏览)。
最大同名类(327)内部 caption 全唯一 → **同类内部可区分**(逐探针 Δ 指纹)。

## 4. 探针覆盖

param ≥6 LAB 探针:**3818**(早期 2272 render_failed 重渲补回);余无预览=stage1 剔除/local-mask/真渲不出,非缺口。

## 5. DB 字段(assets 表,均可 SQL 查询/分层)

**clean verdict**:`status`、`auto_verdict`、`pass_c`、`preset_clean_verdict`(verdict_reason)、
`preset_pro_rate`/`preset_intent_rate`/`preset_coh_rate`、`preset_coherence`、
`preset_vote_pro`/`_intent`/`_coh`、`preset_reliable_probes`、`preset_edit_dispersion`、`preset_near_noop`。
**功能 tag**:`preset_look_name`、`preset_caption`(唯一)、`preset_grade_family`、`preset_axes`(JSON)、
`preset_per_probe`(JSON)、`preset_tag_metrics`(JSON)。
**渲染明细**:`preset_previews` 表(每 preset×探针 before/after 路径 + paired_metrics)。

落库脚本(幂等):`source_qa/_persist_preset_clean.py`(verdict+status)、`_persist_preset_tags.py`(tag);
原始逐探针 jsonl:`pilot/round_10/preset_raw.jsonl`(8027)、`preset_tags.jsonl`(8027)。

## 6. 未竟
- 189 local-mask + 44 needs-local(233):需带 mask 的真实 LR 渲染路径,未清。
- 62 preset_render_failed:真渲不出(源/格式),保留。
- PRO/INTENT/COH 进硬 drop 待金标 κ≥0.6(现全 review)。
- client catalog auto-reset 多 bug:见 `lrc_scripts/docs/CATALOG_AUTORESET_ISSUES_2026-06-23.md`。
