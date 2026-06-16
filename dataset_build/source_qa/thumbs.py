"""Batch thumbnail warmer. The web /thumb endpoint generates on first view, which
makes the first gallery load slow (decodes ~60 full source images). This pre-bakes
256px JPEG thumbs into THUMB_DIR so the gallery is instant.

Run: python -m dataset_build.source_qa.thumbs [--corpus C] [--limit N] [--size 256] [--workers 8]
"""
from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from . import config, db


def _make(args_tuple) -> int:
    asset_id, path, size = args_tuple
    cache = os.path.join(config.THUMB_DIR, f"{asset_id}_{size}.jpg")
    if os.path.exists(cache):
        return 0
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        im = Image.open(path); im.load(); im = im.convert("RGB")
        im.thumbnail((size, size))
        im.save(cache, "JPEG", quality=82)
        return 1
    except Exception:
        return -1


def run(corpus=None, limit=None, size=256, workers=8) -> dict:
    os.makedirs(config.THUMB_DIR, exist_ok=True)
    conn = db.connect()
    where = ["asset_type='image'"]
    params = []
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    sql = f"SELECT asset_id, path FROM assets WHERE {' AND '.join(where)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    print(f"[thumbs] {len(rows)} images, size={size}, workers={workers}", file=sys.stderr)
    ok = fail = skip = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_make, [(x["asset_id"], x["path"], size) for x in rows])):
            if r == 1: ok += 1
            elif r == 0: skip += 1
            else: fail += 1
            if (i + 1) % 2000 == 0:
                print(f"[thumbs] {i+1}/{len(rows)} ok={ok} skip={skip} fail={fail}", file=sys.stderr)
    print({"ok": ok, "skip": skip, "fail": fail})
    return {"ok": ok, "skip": skip, "fail": fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run(corpus=args.corpus, limit=args.limit, size=args.size, workers=args.workers)


if __name__ == "__main__":
    main()
