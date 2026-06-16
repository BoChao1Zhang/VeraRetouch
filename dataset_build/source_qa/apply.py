"""Close the loop: materialize QA verdicts into CLEANED indexes the build reads.

The build (run.load_plan_inputs) reads source_index.jsonl / recipe_index.jsonl
directly and never opens qa.db, so without this step every keep/drop/dedup
decision is a dead end. This writes:
  * source_index.cleaned.jsonl  (drops bad/duplicate images; tags survivors)
  * recipe_index.qa.jsonl       (drops bad/duplicate presets; flags local-edit)
then config.yaml storage.{source,recipe}_index is pointed at them (the build
already honors those keys -> zero hot-path change). Re-uses registry._AtomicWriter
for atomic, auditable writes.

Drop rules (image OR preset), in order of authority:
  1. human final_decision='drop'                         -> drop
  2. human final_decision in (keep,hold)                 -> keep/keep+flag (never auto-override a human)
  3. dup_of non-head:
       - auto_verdict='drop' (true cross-file duplicate)  -> drop
       - else (ppr10k same-file expert sibling)           -> INHERIT the head's verdict (fan-out)
  4. auto_verdict='drop'                                  -> drop
  5. preset status='preset_meta_fail'                     -> drop
  6. no QA row at all                                     -> drop if fail_closed else keep
Survivors carry meta: qa_verdict, dup_of, dup_cluster, split, and (presets)
qa_local_edit / qa_ai_mask / qa_render_engine.

Run: python -m dataset_build.source_qa.apply [--out-root DIR] [--hold-policy exclude|keep]
                                             [--keep-unqaed] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, Optional

from . import config, db


def _read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def _load_assets(conn, asset_type: str) -> Dict[str, dict]:
    rows = conn.execute(
        "SELECT asset_id, final_decision, auto_verdict, dup_of, dup_cluster, split, "
        "status, has_local_mask, has_ai_mask, pass_a, pass_b FROM assets WHERE asset_type=?",
        (asset_type,)).fetchall()
    return {r["asset_id"]: dict(r) for r in rows}


def _effective_verdict(a: Optional[dict], by_id: Dict[str, dict], fail_closed: bool) -> tuple:
    """Returns (keep: bool, reason). a is the asset row (or None if not QA'd)."""
    if a is None:
        return (not fail_closed, "no-qa-row")
    if a["final_decision"] == "drop":
        return False, "human:drop"
    if a["final_decision"] == "keep":
        return True, "human:keep"
    # dup handling (head verdict fan-out for same-file siblings)
    if a["dup_of"]:
        if a["auto_verdict"] == "drop":
            return False, "dup:drop"
        head = by_id.get(a["dup_of"])
        if head is not None:
            keep, why = _effective_verdict(head, by_id, fail_closed)
            return keep, f"inherit({why})"
        return (not fail_closed, "dup:head-missing")
    if a["auto_verdict"] == "drop":
        return False, "auto:drop"
    if a["status"] == "preset_meta_fail":
        return False, "preset_meta_fail"
    # final_decision hold or auto_verdict review/keep/None -> keep (flag review in meta)
    return True, (a["final_decision"] or a["auto_verdict"] or "kept")


def _id_guard(index_path: str, by_id: Dict[str, dict], id_key: str) -> float:
    """Fraction of qa.db asset_ids that still resolve in the current index."""
    idx_ids = {rec.get(id_key) for rec in _read_jsonl(index_path)}
    if not by_id:
        return 1.0
    hit = sum(1 for aid in by_id if aid in idx_ids)
    return hit / len(by_id)


def export_images(conn, src_index: str, out_path: str, hold_policy: str,
                  fail_closed: bool, dry_run: bool) -> dict:
    from dataset_build.registry import _AtomicWriter
    by_id = _load_assets(conn, "image")
    cov = _id_guard(src_index, by_id, "source_id")
    if cov < 0.99:
        raise SystemExit(f"[apply] ABORT images: only {cov:.1%} of qa.db image ids resolve in "
                         f"{src_index} (id drift / wrong index generation). Re-run ingest first.")
    kept = dropped = flagged = 0
    w = None if dry_run else _AtomicWriter(out_path)
    if w:
        w.__enter__()
    try:
        for rec in _read_jsonl(src_index):
            a = by_id.get(rec.get("source_id"))
            keep, why = _effective_verdict(a, by_id, fail_closed)
            if a is not None and a["final_decision"] == "hold" and hold_policy == "exclude":
                keep, why = False, "hold:excluded"
            if not keep:
                dropped += 1
                continue
            if a is not None:
                rec["qa_verdict"] = a["auto_verdict"] or a["final_decision"]
                rec["qa_status"] = a["status"]
                rec["dup_of"] = a["dup_of"]
                rec["dup_cluster"] = a["dup_cluster"]
                rec["split"] = a["split"]
                if a["dup_of"]:
                    rec["qa_fanout_from"] = a["dup_of"]
                if (a["auto_verdict"] == "review") or (a["final_decision"] == "hold"):
                    rec["qa_review"] = 1
                    flagged += 1
            kept += 1
            if w:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if w:
            w.__exit__(None, None, None)
    except Exception:
        if w:
            w.__exit__(*sys.exc_info())
        raise
    return {"kept": kept, "dropped": dropped, "review_flagged": flagged, "id_coverage": round(cov, 4)}


def export_recipes(conn, rcp_index: str, out_path: str, fail_closed: bool, dry_run: bool) -> dict:
    from dataset_build.registry import _AtomicWriter
    by_id = _load_assets(conn, "preset")
    # render engine per preset (best-fidelity preview, if any)
    eng = {r["asset_id"]: r["render_engine"] for r in conn.execute(
        "SELECT asset_id, render_engine FROM preset_previews WHERE render_engine IS NOT NULL")}
    cov = _id_guard(rcp_index, by_id, "recipe_id")
    if cov < 0.99:
        raise SystemExit(f"[apply] ABORT recipes: only {cov:.1%} of qa.db recipe ids resolve in "
                         f"{rcp_index}. Re-run ingest first.")
    kept = dropped = local = 0
    w = None if dry_run else _AtomicWriter(out_path)
    if w:
        w.__enter__()
    try:
        for rec in _read_jsonl(rcp_index):
            a = by_id.get(rec.get("recipe_id"))
            keep, why = _effective_verdict(a, by_id, fail_closed)
            if not keep:
                dropped += 1
                continue
            if a is not None:
                rec["qa_verdict"] = a["auto_verdict"] or a["final_decision"]
                rec["qa_local_edit"] = 1 if a["has_local_mask"] else 0
                rec["qa_ai_mask"] = 1 if a.get("has_ai_mask") else 0
                rec["qa_render_engine"] = eng.get(a["asset_id"])
                if a["has_local_mask"]:
                    local += 1
            kept += 1
            if w:
                w.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if w:
            w.__exit__(None, None, None)
    except Exception:
        if w:
            w.__exit__(*sys.exc_info())
        raise
    return {"kept": kept, "dropped": dropped, "local_edit_flagged": local, "id_coverage": round(cov, 4)}


def run(out_root: Optional[str] = None, hold_policy: str = "exclude",
        fail_closed: bool = True, dry_run: bool = False) -> dict:
    conn = db.connect()
    run_id = db.start_run(conn, "apply", {"hold_policy": hold_policy, "fail_closed": fail_closed,
                                          "dry_run": dry_run})
    img = export_images(conn, config.SOURCE_INDEX, config.SOURCE_INDEX_CLEANED,
                        hold_policy, fail_closed, dry_run)
    rcp = export_recipes(conn, config.RECIPE_INDEX, config.RECIPE_INDEX_CLEANED, fail_closed, dry_run)
    out = {"images": img, "recipes": rcp,
           "source_index_cleaned": None if dry_run else config.SOURCE_INDEX_CLEANED,
           "recipe_index_qa": None if dry_run else config.RECIPE_INDEX_CLEANED}
    db.finish_run(conn, run_id, out)
    conn.close()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if not dry_run:
        print("\n[apply] point the build at the cleaned indexes in dataset_build/config.yaml:\n"
              "  storage:\n"
              "    source_index: source_index.cleaned.jsonl\n"
              "    recipe_index: recipe_index.qa.jsonl", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-policy", choices=["exclude", "keep"], default="exclude",
                    help="how to treat human 'hold' decisions (default exclude)")
    ap.add_argument("--keep-unqaed", action="store_true",
                    help="keep assets with no QA row (default: fail-closed = drop them)")
    ap.add_argument("--dry-run", action="store_true", help="report counts, write nothing")
    args = ap.parse_args()
    run(hold_policy=args.hold_policy, fail_closed=not args.keep_unqaed, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
