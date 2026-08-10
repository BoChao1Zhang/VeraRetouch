"""CLASS_REPORT.md -- the Chinese per-class / long-tail report.

Rendering only.  Every number is looked up from the structures
:mod:`tables` and :mod:`attribution` produced; nothing is recomputed here, so the
markdown and ``per_class_metrics.json`` cannot disagree.

Two things this module refuses to render, both red lines:

* a metric table without its centre-prior columns -- the zero-parameter baseline
  travels with every spatial-field number, so a table that lost it is a bug, not
  a formatting choice;
* any column whose key contains ``auc``.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .tables import LOW_CONFIDENCE_N, REPORT_COLUMNS

__all__ = ["render_report", "markdown_table"]

_DIM_TITLE = {
    "area": "面积占比（area_frac）",
    "components": "连通域数",
    "topology": "拓扑（是否带洞）",
    "boundary": "边界复杂度（circularity）",
    "position": "位置（质心到画幅中心）",
    "softness": "软边比例",
    "winner_confidence": "winner_confidence（已有分层）",
    "upscaled": "image.upscaled（已有分层）",
    "build": "build（已有分层）",
    "active_primitive_bucket": "active primitive count（预测侧）",
}

_MECH_ZH = {
    "format_failure": "生成 context 格式失败",
    "oracle_ceiling": "oracle 天花板低（basis 表达上界）",
    "context_quality": "generated context 质量（GT 好 / generated 差）",
    "s_direction": "s 场方向错（w_dir 与 oracle 夹角大）",
    "s_error": "s 错（换 oracle s 就修好）",
    "rho_error": "rho 错（换 oracle rho 就修好）",
    "s_collapse": "s 轴塌缩（std(s)/std(s*) 过小）",
    "single_primitive": "单基元脆弱（CBand12 仅 1 个 primitive 开启）",
    "upsample_collapse": "guided upsample 塌陷（low 好 hi 差）",
    "area_mismatch": "面积失配（pred_mean vs gt_mean）",
    "below_center_prior": "打不过零参数中心先验",
    "unexplained": "未归因",
}


def _fmt(v: Any, digits: int) -> str:
    if v is None:
        return "n/a"
    if digits == 0:
        return str(int(v))
    return f"{float(v):.{digits}f}"


def markdown_table(rows: Sequence[Sequence[str]], header: Sequence[str]) -> str:
    align = ["---"] + ["---:"] * (len(header) - 1)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _class_rows(table: Mapping[str, Mapping[str, Any]],
                gt_table: Mapping[str, Mapping[str, Any]] | None) -> list[list[str]]:
    rows = []
    for cls, s in table.items():
        cells = [cls + ("  ⚠低置信" if s.get("low_confidence") else "")]
        for key, _hdr, digits in REPORT_COLUMNS:
            cells.append(_fmt(s.get(key), digits))
        if gt_table is not None:
            g = gt_table.get(cls, {})
            gt_med = g.get("local_soft_iou_median")
            med = s.get("local_soft_iou_median")
            cells.append(_fmt(gt_med, 3))
            cells.append("n/a" if (gt_med is None or med is None)
                         else _fmt(float(gt_med) - float(med), 3))
        rows.append(cells)
    return rows


def _table_header(with_gt: bool) -> list[str]:
    hdr = ["类别"] + [h for _k, h, _d in REPORT_COLUMNS]
    if with_gt:
        hdr += ["GT ctx softIoU med", "GT−gen gap"]
    return hdr


def render_report(payload: Mapping[str, Any]) -> str:
    """Assemble CLASS_REPORT.md from the analysis payload."""
    meta = payload["meta"]
    dims: list[str] = payload["dimensions"]
    tables = payload["per_class"]["generated"]
    gt_tables = payload["per_class"].get("gt", {})
    dist = payload["class_distribution"]
    geom_stats = payload.get("geometry_stats", {})
    tail = payload["tail"]
    summ = payload["tail_summary"]
    dec = payload.get("tail_summary_decile", {}) or {}
    pop = payload.get("population_summary", {})
    overall = payload["overall"]["generated"]

    for dim, tbl in tables.items():
        for cls, s in tbl.items():
            if "center_prior_hard_iou" not in s:
                raise AssertionError(
                    f"{dim}/{cls}: the centre-prior column is missing; a spatial-"
                    "field table without its zero-parameter baseline is a red-line "
                    "violation (CLAUDE.md 2026-08-05)"
                )
            bad = [k for k in s if "auc" in k.lower()]
            if bad:
                raise AssertionError(f"{dim}/{cls}: AUC column(s) {bad} -- banned")

    L: list[str] = []
    A = L.append

    A("## 要验证的结论")
    A(f"如果这次分析成立，我们就能说 **{meta['arm']} step {meta['step']} 的 Where 失败不是均匀的**："
      "它集中在特定几何类别（哪一类由下表给出），且长尾样本的失败机制可以被归到 "
      "context / oracle 天花板 / s 场 / readout / upsample 这五个环节中的具体一个；"
      "失败就不能说——只能继续用一个 median soft-IoU 描述整臂，无法指出改哪里。")
    A("")
    A("## 为什么需要验证它")
    A("论文要主张的是「语言条件的空间场能定位到指令所指的区域」。审稿人会问：**在什么样的区域上成立？**"
      "一个只在大面积、居中、实心区域上成立的场，与中心先验难以区分（红线：中心先验 AUC 0.836 曾跑赢全部六个 attention 读出）。"
      "同时，训练资源必须投在贡献最多差样本的那个环节上；没有归因表就只能靠猜。")
    A("")
    A("## 怎么验的")
    A(f"读 `{meta['eval_dir']}` 的 `per_sample.jsonl`（{meta['n_rows']} 行，{meta['n_contexts']} 个 context），"
      f"用 Where-A 发布的 `{meta['split']}` GT mask（`.maskhi.png`，{meta['n_masked']} 个 local 样本）"
      "逐样本算六个几何量并分档；每一档的指标由 `q3vl.whereb.metrics.summarise`（主榜同一个聚合器）在该档子集上重算；"
      f"再对最差的 {summ['n_tail']} 个样本按预注册决策树打失败机制标签。")
    A("")
    A("---")
    A("")
    A("## 1. 设置")
    A("")
    A(markdown_table([
        ["arm", meta["arm"]],
        ["structure / readout", f"{meta.get('structure')} / {meta.get('readout')}"],
        ["checkpoint step", str(meta["step"])],
        ["eval 产物", f"`{meta['eval_dir']}`"],
        ["split（数据代号）", f"`{meta['split']}`（DATA_ASSIGNMENT：S-val 源，永不进训练）"],
        ["主榜 context", meta["main_context"]],
        ["local / global 样本数", f"{overall.get('n_local')} / {overall.get('n_global')}"],
        ["整臂 local softIoU 中位", _fmt(overall.get("local_soft_iou_median"), 3)],
        ["整臂 grid hardIoU", _fmt(overall.get("grid_hard_iou"), 3)],
        ["整臂 grid 边界 F1", _fmt(overall.get("grid_boundary_f1"), 3)],
        ["中心先验 hardIoU / Δ / p", "{} / {} / {}".format(
            _fmt(overall.get("center_prior_hard_iou"), 3),
            _fmt(overall.get("center_prior_delta_hard_iou"), 3),
            _fmt(overall.get("center_prior_delta_hard_iou_p"), 4))],
        ["field cache（GPU 重跑）", meta.get("field_cache") or "未启用（三个机制记为 not_tested）"],
        ["git commit", meta.get("git_commit", "n/a")],
    ], ["项", "值"]))
    A("")
    A("> **判据纪律**：本报告不含任何 AUC 列（2026-08-05 红线）。所有二值化一律「匹配 GT 面积的 top-k」，"
      "每张表都带零参数中心先验列与配对 Δ、p 值。`n < %d` 的类别标 ⚠低置信——中位数不是发现。" % LOW_CONFIDENCE_N)
    A("")
    A("## 2. 分类学定义")
    A("")
    cuts = list(meta["taxonomy"]["area_cuts"])
    A(markdown_table([
        ["area", "`area_frac = mean(mask_hi > 0.5)`",
         "tiny < {0}, small < {1}, medium < {2}, else large".format(*cuts)],
        ["components", "8 连通域计数，丢弃 < max(%d px, %.0f%% 掩膜面积) 的碎片" % (
            meta["taxonomy"]["min_component_px"], 100 * meta["taxonomy"]["min_component_frac"]),
         "single = 1，multi ≥ 2"],
        ["topology", "填洞前后之差；洞需 ≥ max(%d px, %.0f%% 填充面积)" % (
            meta["taxonomy"]["min_hole_px"], 100 * meta["taxonomy"]["min_hole_frac"]),
         "holed / solid"],
        ["boundary", "`circularity = 4πA/P²`，P 用 marching-square 加权周长（圆 0.915、方 0.799、环 0.300 实测）",
         f"compact ≥ {meta['taxonomy']['compact_min']}，否则 complex"],
        ["position", "质心到画幅中心距离，short_side_unit 坐标（与中心先验场同一约定），除以角点距离归一",
         f"center ≤ {meta['taxonomy']['center_max']}, edge > {meta['taxonomy']['edge_min']}, 其余 mid"],
        ["softness", "`soft_frac = |{%.2f < m < %.2f}| / |{m > %.2f}|`" % (
            meta["taxonomy"]["soft_lo"], meta["taxonomy"]["soft_hi"], meta["taxonomy"]["soft_lo"]),
         f"soft ≥ {meta['taxonomy']['soft_frac_min']}，否则 hard"],
    ], ["维度", "定义（只看 GT，不看预测）", "分档"]))
    A("")
    A("**global 样本（GT 全 1，%d 个）不进任何几何分层**——它们不是「一种区域」，而是没有区域；"
      "混进去会把 496 行接近满分的样本灌进 large/solid/center 格。" % (overall.get("n_global") or 0))
    A("")
    A("### 2.1 类别分布与几何量分位")
    A("")
    rows = []
    for dim in dims:
        counts = dist.get(dim, {})
        total = sum(counts.values()) or 1
        cells = ", ".join(f"{k}={v}（{100*v/total:.0f}%）" for k, v in counts.items())
        rows.append([_DIM_TITLE.get(dim, dim), str(total), cells])
    A(markdown_table(rows, ["维度", "n", "分布"]))
    A("")
    if geom_stats:
        A(markdown_table(
            [[k, _fmt(v.get("p10"), 3), _fmt(v.get("p50"), 3), _fmt(v.get("p90"), 3)]
             for k, v in geom_stats.items()],
            ["几何量", "p10", "p50", "p90"]))
        A("")

    A("## 3. 各维度分层表（主榜 = %s context，右侧两列为 GT context 对照）" % meta["main_context"])
    A("")
    for dim in dims:
        A(f"### 3.{dims.index(dim) + 1} {_DIM_TITLE.get(dim, dim)}")
        A("")
        A(markdown_table(_class_rows(tables[dim], gt_tables.get(dim)),
                         _table_header(bool(gt_tables))))
        A("")
        worst = payload["worst_class"].get(dim)
        if worst:
            A(f"最差类别：**{worst}**（按 local softIoU 中位，只在 n ≥ {LOW_CONFIDENCE_N} 的类别里选）")
            note = payload.get("worst_class_note", {}).get(dim)
            if note:
                A("")
                A(note)
            A("")

    A("## 4. 长尾归因")
    A("")
    A(f"**长尾集合（主榜口径）** = 主榜 context 下 softIoU < {meta.get('tail_cut')} 的全部 local 样本，"
      f"共 **{summ['n_tail']}** 个（占 local 的 {100*summ['n_tail']/max(1, overall.get('n_local') or 1):.1f}%）。"
      f"**辅助口径** = 最差 {100*float(meta.get('tail_decile') or 0.1):.0f}%（n={dec.get('n_tail')}，"
      f"截断在 softIoU {_fmt(meta.get('tail_decile_cut'), 3)}）。")
    A("")
    A("> 两个口径都报，因为它们回答的**不是同一个问题**（主 agent 裁定 2026-08-10）："
      "绝对口径 0.30 跨 arm、跨 step 可比——臂变强，长尾就真的变小；"
      "分位口径样本数恒定，比的是失败的**构成**而不是数量。只看前者会把「尾巴变短」误读成「尾巴变好」，"
      "只看后者则永远有 10% 的样本可以叫长尾，臂再强也一样。")
    A("")
    A(f"最差 {meta.get('n_viz_samples')} 个 + 每个维度最差类别的最差样本另出联图。"
      "机制标签是多标签；`primary` 按「越上游越优先」的固定顺序取一个——"
      "**这个顺序是工程判断（先修上游），不是测量结果**；换顺序只改 `primary` 列，命中列不受影响。")
    A("")
    rows = []
    for m, share in sorted(summ["primary_share"].items(), key=lambda kv: -kv[1]):
        if (summ["counts"][m] == 0 and summ["primary_counts"][m] == 0
                and dec.get("counts", {}).get(m, 0) == 0):
            continue
        rows.append([
            _MECH_ZH.get(m, m),
            str(summ["primary_counts"][m]), f"{100*share:.1f}%",
            str(summ["counts"][m]), f"{100*summ['share'][m]:.1f}%",
            f"{100*dec.get('primary_share', {}).get(m, 0):.1f}%",
            str(summ["not_tested_counts"].get(m, 0)),
        ])
    A(markdown_table(rows, ["失败机制", "primary 计数", "primary 占比",
                            "命中计数（多标签）", "命中占比",
                            "辅助口径 primary 占比（最差 10%）", "未测样本数"]))
    A("")
    A("> `命中占比` 是多标签，列和会超过 100%；`primary 占比` 和为 100%。"
      "`未测样本数`= 该样本上这个机制**无法测**（没有 oracle、没跑 field cache、或该 readout 天生 n/a）——"
      "**不是**「不存在」。")
    A("")
    if pop:
        A("同一套判定跑在**全部 local 样本**上（含好样本）作对照。"
          "注意好样本上的机制标签本身没有意义——一个 softIoU 0.9 的大面积样本"
          "当然会有近似常数的 s（`s_collapse`）——这一列只用来回答"
          "「长尾里的机制是不是尾部特有」：")
        A("")
        rows = [[_MECH_ZH.get(m, m), str(pop["primary_counts"][m]),
                 f"{100*pop['primary_share'][m]:.1f}%", str(pop["counts"][m]),
                 f"{100*pop['share'][m]:.1f}%"]
                for m in sorted(pop["primary_share"], key=lambda k: -pop["primary_share"][k])
                if pop["counts"][m] or pop["primary_counts"][m]]
        A(markdown_table(rows, ["失败机制", "primary 计数", "primary 占比",
                                "命中计数", "命中占比"]))
        A("")
    A(f"**瓶颈结论**：长尾里 primary 占比最高的机制是 **{_MECH_ZH.get(summ['bottleneck'], summ['bottleneck'])}"
      f"**（主榜口径 {100*summ['bottleneck_primary_share']:.1f}%，"
      f"辅助口径 {100*dec.get('primary_share', {}).get(summ['bottleneck'], 0):.1f}%）；"
      f"未归因样本占 {100*summ['unexplained_share']:.1f}%。")
    A("")
    A(f"> `oracle_ceiling` 的阈值 {meta['thresholds']['oracle_ceiling']} 是 **provisional**"
      "（主 agent 裁定 2026-08-10 暂用）：它没有标定依据，放松会把责任推给 basis、收紧则推给模型。"
      "任何依赖这一行的结论都必须同时写明这个阈值。")
    A("")

    A("### 4.1 长尾样本清单")
    A("")
    rows = []
    for t in tail[: min(len(tail), 40)]:
        rows.append([
            f"`{t['sample_id'][:16]}…`",
            _fmt(t["row"].get("soft_iou"), 3),
            _fmt(t["row"].get("grid_hard_iou"), 3),
            _fmt(t["row"].get("center_prior_hard_iou"), 3),
            _fmt(t["row"].get("oracle_soft_iou"), 3),
            "/".join(t["labels"].get(d, "?") for d in dims),
            _MECH_ZH.get(t["attribution"]["primary"], t["attribution"]["primary"]),
        ])
    A(markdown_table(rows, ["sample_id", "softIoU", "hardIoU", "中心先验",
                            "oracle", "/".join(dims), "primary 机制"]))
    A("")
    A(f"联图见 `viz/`（{meta.get('n_panels', 0)} 张）。每张固定为 "
      "`I_in | GT mask | GT 叠图 | pred mask hi | pred mask low | s 叠图`，"
      "缺的格子画成 “not available” 而不是省略。")
    A("")

    A("## 5. 结论与建议下一步")
    A("")
    for line in payload.get("conclusions", []):
        A(f"- {line}")
    A("")
    A("## 6. 已知盲区")
    A("")
    A("- **本工具只做单维度分层，不做全交叉**：六维全交叉是 144 格 / %d 个 local 样本。"
      "「small 且 multi」这种交互效应本报告读不出来，需要单独立项。" % (overall.get("n_local") or 0))
    A("- **分类学只看 GT 几何**，不含语义（人 / 天空 / 建筑）。一个「所有小面积样本都是人脸」的混杂本报告识别不了。")
    A("- **归因的阈值是预注册的**（见 `config/thresholds.json`），但阈值附近的样本会在标签间抖动；"
      "`primary` 的顺序假定越上游的机制越该先修，这是一个工程判断，不是测量结果。")
    A("- **`s_direction` / `s_error` / `rho_error` / `single_primitive` 需要重跑 checkpoint**（`--checkpoint`）。"
      "没跑时它们不是 0，是 not_tested；把它们当 0 读会把瓶颈错误地推给 context。")
    A("- **环形（带洞）区域在评测数据里几乎不存在。** 2026-08-10 用同一套几何量普查了全部四个 eval split 的"
      "已发布 GT（`geometry_<split>.json`，CPU，只看 GT）：")
    A("")
    A(markdown_table([
        ["V_where", "400", "0", "16 (4.0%)", "0.0061"],
        ["V_what", "408", "0", "14 (3.4%)", "0.0061"],
        ["T_final", "424", "0", "21 (5.0%)", "0.0042"],
        ["T_lut_unseen", "198", "**1**", "18 (9.1%)", "0.0617"],
    ], ["split", "n(local)", "带洞", "多连通", "max hole_frac"]))
    A("")
    A("  即：**1430 个掩膜里带洞的只有 1 个**（`sft_997b054f97c8c55e045b09f91b54545c`，在 T_lut_unseen，"
      "hole_frac 0.062 / circularity 0.139），`V_where`/`V_what`/`T_final` 三个 split **一个都没有**"
      "（其余样本的 hole_frac ≤ 0.006，是羽化边缘上的针孔，不构成拓扑洞）。"
      "所以协议 §13 「联图至少覆盖环形」这一条，在最终报告的 `T_final` 上**无法满足**——"
      "如实写明该类不存在，**不造样本**。**多连通可以满足**：`T_final` 自身有 21 个，最终 20 图从那里取即可。")
    A("- **local softIoU 是被直接优化的量**（`1 - softIoU` 是 L_mask 权重 1.00 的支配项），"
      "所以「哪一类最差」的排序里，它承担的是收敛度而不是独立证据；同一张表里的 grid 边界 F1 与中心先验 Δ 才是没被优化的列。"
      "详见同目录 eval 产物的 `ATTRIBUTION.md`。")
    A("")
    A("> 分类学阈值的标定依据、核实记录、以及**待主 agent 决策的 6 项**，见 "
      "`experiments/Q3VL_metacanvas_where_what_20260804/where_b/WEVAL1_NOTES.md`。")
    return "\n".join(L) + "\n"
