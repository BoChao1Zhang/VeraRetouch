"""Persist full preset-clean verdict detail (round_<r>/preset_raw.jsonl) into assets.
run() only wrote pass_c + auto_verdict; this backfills the rich verdict fields and
corrects stale status (presets in the jsonl were rendered+judged → not render_failed).
    python -m dataset_build.source_qa._persist_preset_clean [round_dir]
"""
from __future__ import annotations
import json, os, sys
from . import db

COLS = [
    ("preset_clean_verdict", "TEXT"),       # verdict_reason: all_pass/near_noop/not_professional...
    ("preset_pro_rate", "DOUBLE PRECISION"),
    ("preset_intent_rate", "DOUBLE PRECISION"),
    ("preset_coh_rate", "DOUBLE PRECISION"),
    ("preset_coherence", "DOUBLE PRECISION"),
    ("preset_vote_pro", "INTEGER"),
    ("preset_vote_intent", "INTEGER"),
    ("preset_vote_coh", "INTEGER"),
    ("preset_reliable_probes", "INTEGER"),
    ("preset_edit_dispersion", "DOUBLE PRECISION"),
    ("preset_near_noop", "INTEGER"),
]


def main(rdir="round_10"):
    path = os.path.join(os.path.dirname(__file__), "pilot", rdir, "preset_raw.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    conn = db.connect()
    for col, typ in COLS:
        conn.execute(f"ALTER TABLE assets ADD COLUMN IF NOT EXISTS {col} {typ}")
    conn.commit()
    n = status_fixed = 0
    judged_ids = []
    for r in recs:
        aid = r.get("asset_id"); v = r.get("verdict") or {}
        if not aid:
            continue
        judged_ids.append(aid)
        conn.execute(
            "UPDATE assets SET pass_c=%s, auto_verdict=%s, preset_clean_verdict=%s, "
            "preset_pro_rate=%s, preset_intent_rate=%s, preset_coh_rate=%s, preset_coherence=%s, "
            "preset_vote_pro=%s, preset_vote_intent=%s, preset_vote_coh=%s, "
            "preset_reliable_probes=%s, preset_edit_dispersion=%s, preset_near_noop=%s WHERE asset_id=%s",
            (v.get("pass_c"), v.get("auto_verdict"), v.get("verdict_reason"),
             v.get("pro_pass_rate"), v.get("intent_pass_rate"), v.get("coh_pass_rate"),
             v.get("coherence_score"), v.get("vote_pro"), v.get("vote_intent"), v.get("vote_coh"),
             v.get("reliable_probe_count"), v.get("edit_direction_dispersion"), v.get("near_noop"), aid))
        n += 1
        if n % 1000 == 0:
            conn.commit(); print(f"[persist-clean] {n}", file=sys.stderr)
    conn.commit()
    # status fix: jsonl 内的都已渲染+判级 → 把残留 preset_render_failed 订正为 preset_meta_pass
    res = conn.execute(
        "UPDATE assets SET status='preset_meta_pass' WHERE asset_id = ANY(%s) "
        "AND status='preset_render_failed'", (judged_ids,))
    status_fixed = res.rowcount if hasattr(res, "rowcount") else 0
    conn.commit()
    got = conn.execute("SELECT count(*) c FROM assets WHERE preset_clean_verdict IS NOT NULL").fetchone()["c"]
    print(f"[persist-clean] verdict persisted: {n}; status render_failed→meta_pass: {status_fixed}; "
          f"assets with preset_clean_verdict: {got}")
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "round_10")
