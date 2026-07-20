"""Idempotently ingest source images needed by caption and instance SAM3."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from . import config, db


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def load_tagcache_by_source_id(tag_dir: str) -> dict[str, dict]:
    result: dict[str, dict] = {}
    if not os.path.isdir(tag_dir):
        return result
    for index, entry in enumerate(os.scandir(tag_dir), 1):
        if not entry.is_dir():
            continue
        try:
            with open(os.path.join(entry.path, "tags.json"), encoding="utf-8") as handle:
                tags = json.load(handle)
        except (OSError, ValueError):
            continue
        source_id = tags.get("source_id")
        if source_id:
            result[str(source_id)] = tags
        if index % 20000 == 0:
            print(f"tag cache scanned: {index}", file=sys.stderr)
    return result


def _number(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def ingest_images(conn, run_id: str, tags: dict[str, dict], limit: Optional[int] = None) -> int:
    count = 0
    for record in _iter_jsonl(config.SOURCE_INDEX):
        source_id = record.get("source_id")
        source_path = record.get("path")
        if not source_id or not source_path:
            continue
        metadata = tags.get(str(source_id), {})
        row = {
            "asset_id": source_id,
            "asset_type": "image",
            "corpus": record.get("corpus"),
            "path": source_path,
            "scene": metadata.get("scene") or record.get("scene") or "unknown",
            "style": metadata.get("style") or None,
            "width": record.get("width"),
            "height": record.get("height"),
            "bytes_size": record.get("bytes_size"),
            "is_portrait_pool": 1 if record.get("is_portrait_pool") else 0,
            "status": "pending",
            "aesthetic": _number(metadata.get("aesthetic")),
            "aesthetic_vlm": _number(metadata.get("aesthetic_vlm")),
            "meta_json": json.dumps(
                {"index": record, "tag_cache": metadata or None}, ensure_ascii=False
            ),
        }
        db.upsert_asset(conn, row)
        db.log_event(
            conn,
            str(source_id),
            "ingest",
            "ok",
            {"corpus": record.get("corpus"), "has_tagcache": bool(metadata)},
            run_id,
        )
        count += 1
        if limit and count >= limit:
            break
        if count % 5000 == 0:
            conn.commit()
    conn.commit()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    db.init_db()
    connection = db.connect()
    run_id = db.start_run(connection, "source_ingest", {"limit": args.limit})
    tags = load_tagcache_by_source_id(config.TAG_CACHE_DIR)
    count = ingest_images(connection, run_id, tags, args.limit)
    db.finish_run(connection, run_id, {"images": count})
    connection.close()
    print(json.dumps({"images": count, "run_id": run_id}))


if __name__ == "__main__":
    main()
