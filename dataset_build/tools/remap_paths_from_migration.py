#!/usr/bin/env python3
"""Remap source_qa DB asset paths after the 2026-06-22 dataset migration.

The migration (tools/migrate_datasets.py) renamed every source/recipe file into
~/data/datasets/<corpus>/ and left an authoritative src->dst ledger in
~/data/datasets/_migration/manifests/*.jsonl. The source_qa Postgres still holds
the OLD absolute paths, so every preset (and some images) now points at a moved
file. This walks the ledgers, builds src->dst, and UPDATEs assets.path where the
DB path matches a ledger src and the dst exists. Idempotent.

    python -m dataset_build.tools.remap_paths_from_migration [--apply]
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

MIG = Path("/home/bc/data/datasets/_migration/manifests")


def build_map() -> dict:
    m = {}
    for mf in sorted(MIG.glob("*.jsonl")):
        with open(mf, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for a in rec["assets"]:
                    src = a["src"]
                    if src.startswith("tar://"):
                        continue
                    m[src] = a["dst"]
    return m


def main(apply: bool) -> int:
    from dataset_build.source_qa import db
    src2dst = build_map()
    print(f"[remap] ledger entries: {len(src2dst)}")
    conn = db.connect()
    rows = conn.execute("SELECT asset_id, path, asset_type FROM assets").fetchall()
    hits = miss_dst = 0
    updates = []
    for r in rows:
        dst = src2dst.get(r["path"])
        if dst is None:
            continue
        if not os.path.exists(dst):
            miss_dst += 1
            continue
        hits += 1
        updates.append((dst, r["asset_id"]))
    print(f"[remap] DB rows matched by old path: {hits}  (dst-missing skipped: {miss_dst})")
    by_type = {}
    if apply and updates:
        for dst, aid in updates:
            conn.execute("UPDATE assets SET path=%s WHERE asset_id=%s", (dst, aid))
        conn.commit()
        print(f"[remap] APPLIED {len(updates)} path updates")
    elif not apply:
        print("[remap] dry-run (pass --apply to write)")
    # quick post-check on presets
    pe = conn.execute("SELECT path FROM assets WHERE asset_type='preset' "
                      "AND status IN ('preset_meta_pass','preset_meta_local') "
                      "AND dup_of IS NULL").fetchall()
    ok = sum(1 for r in pe if os.path.exists(r["path"]))
    print(f"[remap] stage1-pass presets with existing file now: {ok}/{len(pe)}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main("--apply" in sys.argv))
