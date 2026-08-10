#!/usr/bin/env python
"""WEVAL-1 entry point: per-class evaluation + long-tail bottleneck analysis.

Runs against **any** arm's eval directory, at any step -- the tool takes
``per_sample.jsonl`` plus the split's published GT masks and needs nothing that
is specific to W01.  A checkpoint is optional and only unlocks the three
field-level mechanisms and the prediction panels.

Usage::

    # artefact-only (no GPU, ~30 s incl. the cold mask read)
    python -m q3vl.whereb.analysis.run_analysis \\
        --eval-dir /home/bc/data/runs/where_b/W01/eval_step1500 \\
        --arm W01 --step 1500 \\
        --out-dir experiments/.../where_b/analysis_W01_step1500

    # with a re-run of the tail samples' fields (needs a free GPU)
    ... --checkpoint /home/bc/data/runs/where_b/W01/where_b_step1500.pt --device cuda

The geometry pass is cached (``--geometry-cache``): it is a property of the
split, so all eight arms share one cache file.
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# Campaign bug R6 (2026-08-05): torch loads a libstdc++ that shadows the one
# `_sqlite3`'s dependency chain needs, so a torch-first process can never open a
# published shard afterwards.  This job reads published shards (maskviews, and
# the split index for the panels), so the guard has to be here, in the entry
# point, before anything drags torch in.
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

MAIN_CONTEXT = "generated"
CONTROL_CONTEXT = "gt"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, cwd=Path(__file__).resolve().parents[3],
                              timeout=10).stdout.strip()
    except Exception:                                            # pragma: no cover
        return "unknown"


def _env_facts() -> dict[str, Any]:
    import numpy

    facts = {"python": sys.version.split()[0], "numpy": numpy.__version__,
             "argv": sys.argv}
    try:
        import torch

        facts["torch"] = torch.__version__
    except Exception:                                            # pragma: no cover
        facts["torch"] = None
    try:
        import scipy

        facts["scipy"] = scipy.__version__
    except Exception:                                            # pragma: no cover
        facts["scipy"] = None
    return facts


def _tail_ids(rows: Sequence[Mapping[str, Any]], k: int) -> list[str]:
    ranked = sorted((r for r in rows if r.get("soft_iou") is not None),
                    key=lambda r: float(r["soft_iou"]))
    return [str(r["sample_id"]) for r in ranked[:k]]


def _best_ids(rows: Sequence[Mapping[str, Any]], k: int) -> list[str]:
    ranked = sorted((r for r in rows if r.get("soft_iou") is not None),
                    key=lambda r: -float(r["soft_iou"]))
    return [str(r["sample_id"]) for r in ranked[:k]]


def main() -> int:
    from .attribution import AnalysisThresholds, attribute_sample, mechanism_summary
    from .classify import (
        EXTRA_DIMENSIONS, dimension_activity, extra_labels, geometry_quantiles,
        labels_from_geometry, load_geometry_cache, merge_labels, scan_geometry,
    )
    from .report import render_report
    from .tables import (
        class_distribution, load_per_sample, overall_table, per_class_tables,
        rows_by_context, worst_class, worst_class_note,
    )
    from .taxonomy import DIMENSIONS, TaxonomyConfig

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", required=True,
                    help="a checkpoint's eval directory (holds per_sample.jsonl)")
    ap.add_argument("--arm", default=None, help="default: read from metrics.json")
    ap.add_argument("--step", default=None, help="default: parsed from --eval-dir")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--maskview-root", default=None,
                    help="default: WHERE_A_MASKVIEW_DIR/<split>")
    ap.add_argument("--geometry-cache", default=None,
                    help="shared across arms; recomputed when absent")
    ap.add_argument("--main-context", default=MAIN_CONTEXT)
    ap.add_argument("--tail-k", type=int, default=12)
    ap.add_argument("--success-k", type=int, default=4)
    ap.add_argument("--no-viz", action="store_true")
    ap.add_argument("--verify-checksums", action="store_true")
    # optional GPU pass
    ap.add_argument("--checkpoint", default=None,
                    help="re-run this checkpoint on the tail samples to unlock "
                         "s_direction / s_error / rho_error / single_primitive "
                         "and the prediction panels")
    ap.add_argument("--field-cache", default=None,
                    help="reuse an existing field cache directory instead of "
                         "re-running the checkpoint")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--field-batch", type=int, default=2)
    ap.add_argument("--attn", default="flash_attention_2")
    args = ap.parse_args()

    t_start = time.time()
    eval_dir = Path(args.eval_dir)
    per_sample = eval_dir / "per_sample.jsonl"
    if not per_sample.exists():
        raise SystemExit(f"{per_sample} does not exist")
    metrics_path = eval_dir / "metrics.json"
    board = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    arm = args.arm or board.get("arm") or "unknown"
    step = args.step or "".join(c for c in eval_dir.name if c.isdigit()) or "unknown"
    out_dir = Path(args.out_dir or (eval_dir.parent / f"analysis_{arm}_step{step}"))
    (out_dir / "config").mkdir(parents=True, exist_ok=True)

    print(f"[weval1] arm={arm} step={step} eval_dir={eval_dir}", flush=True)

    rows = load_per_sample(per_sample)
    by_ctx = rows_by_context(rows)
    if args.main_context not in by_ctx:
        raise SystemExit(f"context {args.main_context!r} not in {sorted(by_ctx)}")
    main_rows = by_ctx[args.main_context]
    local_main = [r for r in main_rows if r.get("render_mode") == "local"]
    local_ids = [str(r["sample_id"]) for r in local_main]
    gt_rows = {str(r["sample_id"]): r for r in by_ctx.get(CONTROL_CONTEXT, [])}
    print(f"[weval1] {len(rows)} rows, {len(by_ctx)} contexts, "
          f"{len(local_main)} local samples on the main board", flush=True)

    # --- geometry ---------------------------------------------------------
    cfg = TaxonomyConfig()
    cache_path = Path(args.geometry_cache) if args.geometry_cache else \
        out_dir / "config" / f"geometry_{args.split}.json"
    scan = load_geometry_cache(cache_path)
    if scan is None or set(scan.get("geometry", {})) < set(local_ids):
        from q3vl.whereb.config import WHERE_A_MASKVIEW_DIR

        root = Path(args.maskview_root or (Path(WHERE_A_MASKVIEW_DIR) / args.split))
        print(f"[weval1] scanning GT masks under {root}", flush=True)
        scan = scan_geometry(local_ids, maskview_root=root, cfg=cfg,
                             verify=args.verify_checksums)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(scan, ensure_ascii=False), encoding="utf-8")
    else:
        print(f"[weval1] geometry cache hit: {cache_path}", flush=True)
    geometry = scan["geometry"]
    geo_labels = labels_from_geometry(geometry, cfg)
    extra = extra_labels(local_main, scan.get("extra", {}))
    labels = merge_labels(geo_labels, extra)
    dims = list(DIMENSIONS) + list(EXTRA_DIMENSIONS)
    gt_local = [r for r in by_ctx.get(CONTROL_CONTEXT, [])
                if r.get("render_mode") == "local"]

    def build_tables() -> tuple[dict, dict, dict, dict, list[str]]:
        act = dimension_activity(labels, dims)
        active = [d for d in dims if act[d]["active"]]
        pc = {args.main_context: per_class_tables(local_main, labels, active),
              CONTROL_CONTEXT: per_class_tables(gt_local, labels, active)}
        w = {d: worst_class(pc[args.main_context][d]) for d in active}
        note = {d: worst_class_note(pc[args.main_context][d], w[d]) for d in active}
        return act, pc, w, note, active

    # Pass 1 picks which samples deserve a figure; the field cache may then add
    # the ``active_primitive_bucket`` labels the eval never emitted, so the
    # tables are rebuilt afterwards.  Two cheap passes over 400 rows beats a
    # stratum that silently reports "None: 400".
    activity, per_class, worst, worst_note, active_dims = build_tables()

    # --- the long tail -----------------------------------------------------
    # Two different sets, and conflating them is how a tail analysis turns into
    # anecdote: the *statistical* tail (everything under the pre-registered
    # soft-IoU cut) is what the mechanism shares are computed on, and the worst
    # ``tail_k`` are the ones that get a figure drawn.
    thresholds = AnalysisThresholds()
    by_id = {str(r["sample_id"]): r for r in local_main}
    ranked_local = sorted((r for r in local_main if r.get("soft_iou") is not None),
                          key=lambda r: float(r["soft_iou"]))
    tail_set_ids = [str(r["sample_id"]) for r in ranked_local
                    if float(r["soft_iou"]) < thresholds.tail_soft_iou]
    # Main-agent ruling (2026-08-10): report BOTH cuts.  The absolute cut is
    # comparable across arms (a stronger arm has a smaller tail); the bottom
    # decile always holds the same number of samples, so it compares *shapes* of
    # failure rather than amounts.  Reading either one alone gets a question
    # wrong: "did the tail shrink" needs the absolute cut, "did the tail change
    # character" needs the decile.
    n_dec = max(1, int(round(thresholds.tail_decile * len(ranked_local))))
    decile_ids = [str(r["sample_id"]) for r in ranked_local[:n_dec]]
    decile_cut = (float(ranked_local[n_dec - 1]["soft_iou"]) if ranked_local else None)
    panel_ids = _tail_ids(local_main, args.tail_k)
    # one extra sample from each dimension's worst class, so the panels cover the
    # classes the tables just accused rather than only the global bottom
    for dim, cls in worst.items():
        members = [sid for sid in local_ids
                   if labels.get(sid, {}).get(dim) == cls and sid in by_id]
        members.sort(key=lambda s: float(by_id[s].get("soft_iou") or 1.0))
        for sid in members[:1]:
            if sid not in panel_ids:
                panel_ids.append(sid)
    tail_ids = panel_ids                       # what the field cache has to cover
    success_ids = _best_ids(local_main, args.success_k)

    fields: dict[str, Any] = {}
    field_cache_dir = None
    if args.field_cache:
        field_cache_dir = Path(args.field_cache)
    elif args.checkpoint:
        from .fieldcache import build_field_cache

        field_cache_dir = out_dir / "fields"
        meta = build_field_cache(
            tail_ids + success_ids, out_dir=field_cache_dir,
            checkpoint=Path(args.checkpoint), arm=arm, split=args.split,
            context=args.main_context, device=args.device,
            attn=args.attn, batch_size=args.field_batch)
        print(f"[weval1] field cache: {json.dumps(meta.get('n_written'))} samples",
              flush=True)
    if field_cache_dir is not None and field_cache_dir.exists():
        from .fieldcache import FieldCache

        fc = FieldCache(field_cache_dir)
        fields = fc.scalars
        # the predicted rho is the only place the active-primitive count exists;
        # fill the stratum the eval leaves empty
        filled = 0
        for sid, row in fields.items():
            if row.get("active_primitives") is not None:
                from q3vl.whereb.metrics import active_primitive_bucket

                labels.setdefault(sid, {})["active_primitive_bucket"] = \
                    active_primitive_bucket(int(row["active_primitives"]))
                filled += 1
        if filled:
            print(f"[weval1] active_primitive_bucket filled for {filled} samples; "
                  "rebuilding tables", flush=True)
            activity, per_class, worst, worst_note, active_dims = build_tables()

    # --- attribution -------------------------------------------------------
    labelled_all = []
    for r in local_main:
        sid = str(r["sample_id"])
        a = attribute_sample(r, gt_row=gt_rows.get(sid), fields=fields.get(sid),
                             thresholds=thresholds)
        a["sample_id"] = sid
        labelled_all.append(a)
    by_attr = {a["sample_id"]: a for a in labelled_all}
    tail = [{
        "sample_id": sid,
        "rank": i,
        "in_viz": sid in panel_ids,
        "in_decile": sid in decile_ids,
        "labels": labels.get(sid, {}),
        "row": by_id[sid],
        "attribution": by_attr[sid],
        "fields": fields.get(sid),
    } for i, sid in enumerate(dict.fromkeys(tail_set_ids + decile_ids + panel_ids))
        if sid in by_id]
    tail_summary = mechanism_summary([t["attribution"] for t in tail])
    decile_summary = mechanism_summary([by_attr[s] for s in decile_ids if s in by_attr])
    population_summary = mechanism_summary(labelled_all)

    # --- panels ------------------------------------------------------------
    n_panels = 0
    if not args.no_viz:
        n_panels = _render_panels(
            out_dir / "viz", tail=tail, success_ids=success_ids, by_id=by_id,
            labels=labels, by_attr=by_attr, split=args.split,
            maskview_root=args.maskview_root,
            field_cache_dir=field_cache_dir)

    # --- write -------------------------------------------------------------
    payload: dict[str, Any] = {
        "meta": {
            "arm": arm, "step": step, "split": args.split,
            "eval_dir": str(eval_dir), "main_context": args.main_context,
            "structure": board.get("structure"), "readout": board.get("readout"),
            "n_rows": len(rows), "n_contexts": len(by_ctx),
            "n_masked": len(geometry), "n_panels": n_panels,
            "tail_cut": thresholds.tail_soft_iou,
            "tail_decile": thresholds.tail_decile,
            "tail_decile_cut": decile_cut,
            "n_viz_samples": len(panel_ids),
            "taxonomy": cfg.to_dict(),
            "thresholds": thresholds.to_dict(),
            "field_cache": (str(field_cache_dir) if fields else None),
            "git_commit": _git_commit(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "dimensions": active_dims,
        "dimension_activity": activity,
        "class_distribution": class_distribution(labels, dims),
        "geometry_stats": geometry_quantiles(geometry),
        "overall": {args.main_context: overall_table(main_rows),
                    CONTROL_CONTEXT: overall_table(by_ctx.get(CONTROL_CONTEXT, []))},
        "per_class": per_class,
        "worst_class": worst,
        "worst_class_note": worst_note,
        "tail": tail,
        "tail_summary": tail_summary,
        "tail_summary_decile": decile_summary,
        "population_summary": population_summary,
        "conclusions": _conclusions(per_class[args.main_context], worst, activity,
                                    tail_summary, population_summary, geometry,
                                    tail),
        "mask_scan_facts": scan.get("facts"),
    }
    (out_dir / "per_class_metrics.json").write_text(
        json.dumps({k: v for k, v in payload.items() if k != "tail"},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    with (out_dir / "tail_samples.jsonl").open("w", encoding="utf-8") as fh:
        for t in tail:
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    (out_dir / "config" / "thresholds.json").write_text(
        json.dumps({"taxonomy": cfg.to_dict(),
                    "attribution": thresholds.to_dict()}, indent=2), encoding="utf-8")
    (out_dir / "config" / "provenance.json").write_text(
        json.dumps({"env": _env_facts(), "git_commit": payload["meta"]["git_commit"],
                    "eval_dir": str(eval_dir), "board_arm": board.get("arm"),
                    "mask_scan": scan.get("facts"),
                    "seconds": round(time.time() - t_start, 1)},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "CLASS_REPORT.md").write_text(render_report(payload), encoding="utf-8")
    print(f"[weval1] wrote {out_dir}/CLASS_REPORT.md  ({time.time() - t_start:.1f}s)",
          flush=True)
    return 0


def _conclusions(tables, worst, activity, tail_summary, pop_summary, geometry,
                 tail=()) -> list[str]:
    """Statements the numbers support, written so a wrong one is falsifiable."""
    import statistics

    out: list[str] = []
    fired = [t["attribution"]["evidence"] for t in tail
             if "area_mismatch" in t["attribution"]["mechanisms"]]
    ratios = [e["area_ratio"] for e in fired if e.get("area_ratio")]
    if ratios:
        over = sum(1 for e in fired if e.get("area_direction") == "over")
        out.append(
            f"长尾里触发 `area_mismatch` 的 {len(fired)} 个样本中 **{over} 个是过覆盖**"
            f"（pred_mean > gt_mean），预测/GT 面积比中位 **{statistics.median(ratios):.2f}**"
            "——场倾向于铺开成接近全局的掩膜，而不是缩到指令所指的区域。")
    for dim, cls in worst.items():
        tbl = tables.get(dim, {})
        if cls is None or cls not in tbl:
            continue
        vals = [(k, v.get("local_soft_iou_median")) for k, v in tbl.items()
                if v.get("local_soft_iou_median") is not None]
        if len(vals) < 2:
            continue
        best = max(vals, key=lambda kv: kv[1])
        worst_v = dict(vals)[cls]
        spread = best[1] - worst_v
        if spread >= 0.05:
            out.append(
                f"`{dim}` 维度上最差类别是 **{cls}**（softIoU 中位 {worst_v:.3f}），"
                f"最好类别 {best[0]}（{best[1]:.3f}），跨类差 {spread:.3f}。")
    inactive = [d for d, a in activity.items() if not a["active"]]
    if inactive:
        out.append("以下维度在本 split 上**只有一个类别**，表里不出现："
                   + "、".join(f"`{d}`" for d in inactive)
                   + "——不是模型在这些维度上没差别，是数据里没有对比。")
    b = tail_summary.get("bottleneck")
    if b:
        tail_share = tail_summary["bottleneck_primary_share"]
        pop_share = pop_summary["primary_share"].get(b, 0.0)
        verdict = ("这是**全局**问题，不是尾部特有" if pop_share >= 0.6 * tail_share
                   else "这是**尾部特有**的机制（全体样本上稀有得多）")
        out.append(
            f"长尾的 primary 机制以 **{b}** 为首（{100*tail_share:.1f}%）；"
            f"全体 local 样本上同一机制占 {100*pop_share:.1f}% —— {verdict}。")
    nt = tail_summary.get("not_tested_counts", {})
    blocked = [k for k, v in nt.items() if v >= tail_summary["n_tail"]]
    if blocked:
        out.append("以下机制在**每一个**长尾样本上都无法测："
                   + "、".join(f"`{k}`" for k in sorted(blocked))
                   + "——瓶颈结论在这些机制上没有证据，不是它们被排除了。")
    return out


def _render_panels(viz_dir, *, tail, success_ids, by_id, labels, by_attr, split,
                   maskview_root, field_cache_dir) -> int:
    """Composite panels for the tail (and a few successes, so it is not a
    failure-only gallery -- protocol §13 asks for both)."""
    from q3vl.whereb.config import SPLIT_DIR, WHERE_A_MASKVIEW_DIR

    from .panels import panel_caption, render_tail_panel
    from .pubio import MaskViews, SplitImages

    viz_dir.mkdir(parents=True, exist_ok=True)
    mv = MaskViews(Path(maskview_root or (Path(WHERE_A_MASKVIEW_DIR) / split)),
                   verify=False)
    try:
        images = SplitImages(Path(SPLIT_DIR) / f"{split}.index.jsonl")
    except Exception as exc:                                     # pragma: no cover
        print(f"[weval1] images unavailable ({type(exc).__name__}: {exc}); "
              "panels will show the GT only", flush=True)
        images = None
    cache = None
    if field_cache_dir is not None and Path(field_cache_dir).exists():
        from .fieldcache import FieldCache

        cache = FieldCache(field_cache_dir)

    n = 0
    # only the samples the report says get a figure: the statistical tail is 89
    # rows on W01 and drawing all of them would produce 73 panels whose
    # prediction tiles are empty (the field cache covers the selected ones)
    jobs = [("failure", t["sample_id"], t["rank"]) for t in tail if t.get("in_viz")]
    jobs += [("success", sid, i) for i, sid in enumerate(success_ids)]
    for kind, sid, rank in jobs:
        row = by_id.get(sid)
        if row is None:
            continue
        try:
            img = images.image_array(sid) if images is not None else None
        except Exception as exc:
            print(f"[weval1] {sid}: image read failed ({exc})", flush=True)
            img = None
        gt = mv.mask_hi(sid) if mv.has(sid, mv.HI) else None
        arrays = cache.arrays(sid) if (cache is not None and sid in cache) else {}
        attribution = by_attr.get(sid, {"primary": "n/a", "mechanisms": [],
                                        "evidence": {}})
        title = (f"[{kind}#{rank}] {sid}   softIoU="
                 f"{float(row.get('soft_iou') or 0):.3f}")
        caption = panel_caption(row, labels.get(sid, {}), attribution)
        render_tail_panel(
            viz_dir / f"{kind}_{rank:02d}_{sid}.png",
            image=img, mask_gt_hi=gt,
            mask_pred_hi=arrays.get("m_hi"), mask_pred_low=arrays.get("m_low"),
            s_low=arrays.get("s_low"), s_star_low=arrays.get("s_star_low"),
            title=title, caption=caption)
        n += 1
    return n


if __name__ == "__main__":
    raise SystemExit(main())
