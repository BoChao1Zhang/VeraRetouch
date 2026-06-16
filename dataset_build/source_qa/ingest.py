"""Ingest source_index.jsonl (images) + recipe_index.jsonl (presets) + tag_cache
into the QA SQLite, creating one `assets` row + an `ingest` provenance event per
asset. Idempotent (upsert). aesthetic_vlm / aesthetic_model from tag_cache are
seeded as iqa_scores so they show up alongside NR-IQA in the UI.

Run:  python -m dataset_build.source_qa.ingest [--limit N] [--images-only|--presets-only]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional

from . import config, db


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def load_tagcache_by_source_id(tag_dir: str) -> Dict[str, dict]:
    """Walk tag_cache once -> {source_id: tags.json dict}. ~106k small files."""
    out: Dict[str, dict] = {}
    if not os.path.isdir(tag_dir):
        return out
    n = 0
    for entry in os.scandir(tag_dir):
        if not entry.is_dir():
            continue
        tp = os.path.join(entry.path, "tags.json")
        try:
            with open(tp, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        sid = d.get("source_id")
        if sid:
            out[sid] = d
        n += 1
        if n % 20000 == 0:
            print(f"  ...tag_cache scanned {n}", file=sys.stderr)
    return out


def _f(v):
    try:
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def ingest_images(conn, run_id: str, tags: Dict[str, dict], limit: Optional[int] = None) -> int:
    n = 0
    for rec in _iter_jsonl(config.SOURCE_INDEX):
        sid = rec.get("source_id")
        if not sid:
            continue
        t = tags.get(sid, {})
        aesthetic = _f(t.get("aesthetic"))
        aesthetic_vlm = _f(t.get("aesthetic_vlm"))
        aesthetic_model = _f(t.get("aesthetic_model"))
        scene = (t.get("scene") or rec.get("scene") or "any")
        row = {
            "asset_id": sid,
            "asset_type": "image",
            "corpus": rec.get("corpus"),
            "path": rec.get("path"),
            "scene": scene,
            "style": (t.get("style") or None),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "bytes_size": rec.get("bytes_size"),
            "is_portrait_pool": 1 if rec.get("is_portrait_pool") else 0,
            "status": "pending",
            "aesthetic": aesthetic,
            "aesthetic_vlm": aesthetic_vlm,
            "meta_json": json.dumps({"index": rec, "tag_cache": t or None}, ensure_ascii=False),
        }
        db.upsert_asset(conn, row)
        seeded = {}
        if aesthetic_model is not None:
            seeded["aesthetic_model"] = aesthetic_model
        if aesthetic_vlm is not None:
            seeded["aesthetic_vlm"] = aesthetic_vlm
        if seeded:
            db.add_scores(conn, sid, seeded, run_id=run_id, model_version="tag_cache")
        db.log_event(conn, sid, "ingest", "ok",
                     {"corpus": rec.get("corpus"), "has_tagcache": bool(t)}, run_id)
        n += 1
        if limit and n >= limit:
            break
        if n % 5000 == 0:
            conn.commit()
            print(f"  images ingested {n}", file=sys.stderr)
    conn.commit()
    return n


def ingest_presets(conn, run_id: str, limit: Optional[int] = None) -> int:
    n = 0
    for rec in _iter_jsonl(config.RECIPE_INDEX):
        rid = rec.get("recipe_id")
        if not rid:
            continue
        row = {
            "asset_id": rid,
            "asset_type": "preset",
            "corpus": rec.get("pack_id"),
            "path": rec.get("path"),
            "style": rec.get("style"),
            "kind": rec.get("kind"),
            "fmt": rec.get("fmt"),
            "pack_id": rec.get("pack_id"),
            "scene_affinity": rec.get("scene_affinity"),
            "is_bw": 1 if rec.get("is_bw") else 0,
            "is_technical": 1 if rec.get("is_technical") else 0,
            "has_local_mask": 1 if rec.get("has_local_mask") else 0,
            "has_ai_mask": 1 if rec.get("has_ai_mask") or rec.get("qa_ai_mask") else 0,
            "lut_size": rec.get("lut_size"),
            "status": "pending",
            "meta_json": json.dumps({"index": rec}, ensure_ascii=False),
        }
        db.upsert_asset(conn, row)
        db.log_event(conn, rid, "ingest", "ok",
                     {"pack_id": rec.get("pack_id"), "kind": rec.get("kind"), "fmt": rec.get("fmt")}, run_id)
        n += 1
        if limit and n >= limit:
            break
        if n % 5000 == 0:
            conn.commit()
    conn.commit()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--images-only", action="store_true")
    ap.add_argument("--presets-only", action="store_true")
    args = ap.parse_args()

    db.init_db()
    conn = db.connect()
    run_id = db.start_run(conn, "ingest", {"limit": args.limit})

    ni = npp = 0
    if not args.presets_only:
        print("loading tag_cache (source_id -> tags) ...", file=sys.stderr)
        tags = load_tagcache_by_source_id(config.TAG_CACHE_DIR)
        print(f"  tag_cache entries: {len(tags)}", file=sys.stderr)
        ni = ingest_images(conn, run_id, tags, args.limit)
    if not args.images_only:
        npp = ingest_presets(conn, run_id, args.limit)

    db.finish_run(conn, run_id, {"images": ni, "presets": npp})
    print(json.dumps({"images": ni, "presets": npp, "run_id": run_id}))
    conn.close()


if __name__ == "__main__":
    main()
