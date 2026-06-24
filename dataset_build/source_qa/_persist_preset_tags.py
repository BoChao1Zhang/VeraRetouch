"""Persist preset tag round results (round_<r>/preset_tags.jsonl) into assets.
tag_round only writes jsonl; this backfills the DB columns so tags are queryable.
    python -m dataset_build.source_qa._persist_preset_tags [round_dir]
"""
from __future__ import annotations
import json, os, sys
from . import db

COLS = [
    ("preset_look_name", "TEXT"),      # vlm name (粗浏览)
    ("preset_caption", "TEXT"),        # vlm caption (唯一细描)
    ("preset_grade_family", "TEXT"),   # det family
    ("preset_per_probe", "TEXT"),      # vlm per_probe JSON
    ("preset_axes", "TEXT"),           # det axes JSON (temp/tint/sat/contrast/tone/exposure)
    ("preset_tag_metrics", "TEXT"),    # det LAB metrics JSON
]


def main(rdir="round_10"):
    path = os.path.join(os.path.dirname(__file__), "pilot", rdir, "preset_tags.jsonl")
    recs = [json.loads(l) for l in open(path) if l.strip()]
    conn = db.connect()
    for col, typ in COLS:
        conn.execute(f"ALTER TABLE assets ADD COLUMN IF NOT EXISTS {col} {typ}")
    conn.commit()
    n = 0
    for r in recs:
        aid = r.get("asset_id")
        if not aid:
            continue
        axes = {k: r.get(k) for k in ("temperature", "tint", "saturation", "contrast", "tone", "exposure")}
        conn.execute(
            "UPDATE assets SET preset_look_name=%s, preset_caption=%s, preset_grade_family=%s, "
            "preset_per_probe=%s, preset_axes=%s, preset_tag_metrics=%s WHERE asset_id=%s",
            (r.get("vlm_name"), r.get("vlm_caption"), r.get("grade_family"),
             json.dumps(r.get("vlm_per_probe"), ensure_ascii=False) if r.get("vlm_per_probe") else None,
             json.dumps(axes, ensure_ascii=False),
             json.dumps(r.get("metrics"), ensure_ascii=False) if r.get("metrics") else None,
             aid))
        n += 1
        if n % 1000 == 0:
            conn.commit(); print(f"[persist] {n}", file=sys.stderr)
    conn.commit()
    got = conn.execute("SELECT count(*) c FROM assets WHERE preset_caption IS NOT NULL").fetchone()["c"]
    print(f"[persist] updated {n} from jsonl; assets with preset_caption now: {got}")
    conn.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "round_10")
