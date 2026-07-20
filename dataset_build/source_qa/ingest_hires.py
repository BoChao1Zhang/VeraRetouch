"""Ingest high-resolution images into the retained PostgreSQL source inventory.

Files whose short edge meets ``--min-side`` are upserted as pending assets. PARA
metadata may provide a scene stratum; other corpora use the explicit unknown bucket.
Canonical eligibility is determined only by the instance-SAM3 cache and decodability,
never by historical source-QA scores or verdicts.

用法:
    python -m dataset_build.source_qa.ingest_hires --dir <图像目录> --corpus <名> [--min-side 720]
        [--scene-csv PARA-Images.csv]   # PARA 官方标注（imageName,sceneCategory 列）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys

from PIL import Image

from . import db

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}
# PARA sceneCategory -> retained scene strata; unmatched values use the unknown bucket.
PARA_SCENE_MAP = {
    "portrait": "portrait", "scene": "landscape", "food": "food",
    "stilllife": "still_life", "still life": "still_life",
    "building": "architecture", "architecture": "architecture",
    "nightscene": "night", "night scene": "night",
    "indoor": "still_life", "animal": "any", "plant": "any",
}


def _stable_id(path: str, size: int) -> str:
    h = hashlib.sha1(f"{path} {size}".encode("utf-8", "surrogatepass"))
    return f"src_{h.hexdigest()[:16]}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--min-side", type=int, default=720)
    ap.add_argument("--scene-csv", default=None)
    args = ap.parse_args()

    scene_by_name: dict = {}
    if args.scene_csv:
        with open(args.scene_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("imageName") or row.get("image_name") or ""
                cat = (row.get("sceneCategory") or row.get("semantic") or "").strip().lower()
                if name:
                    scene_by_name[name] = PARA_SCENE_MAP.get(cat, "any")

    db.init_db()
    conn = db.connect()
    stats = {"seen": 0, "small": 0, "bad": 0, "ingested": 0}
    for root, _dirs, files in os.walk(args.dir):
        for fn in sorted(files):
            if os.path.splitext(fn)[1].lower() not in EXTS:
                continue
            stats["seen"] += 1
            p = os.path.abspath(os.path.join(root, fn))
            try:
                sz = os.path.getsize(p)
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:  # noqa: BLE001 - 坏图跳过
                stats["bad"] += 1
                continue
            if min(w, h) < args.min_side:
                stats["small"] += 1
                continue
            db.upsert_asset(conn, {
                "asset_id": _stable_id(p, sz),
                "asset_type": "image",
                "corpus": args.corpus,
                "path": p,
                "scene": scene_by_name.get(fn, "any"),
                "style": None,
                "width": w, "height": h,
                "bytes_size": sz,
                "is_portrait_pool": 0,
                "status": "pending",
                "meta_json": json.dumps({"ingest": "ingest_hires", "min_side": args.min_side}),
            })
            stats["ingested"] += 1
            if stats["ingested"] % 2000 == 0:
                conn.commit()
                print(f"  {stats['ingested']} ingested...", file=sys.stderr, flush=True)
    conn.commit()
    conn.close()
    print(f"[ingest_hires] corpus={args.corpus} {json.dumps(stats)}")


if __name__ == "__main__":
    main()
