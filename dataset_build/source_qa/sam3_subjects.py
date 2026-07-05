"""SAM3 主体 mask precompute：消费 source_captions.subjects（caption_subjects.py 产出），
对每张源图的每个主体 en 名跑 SAM3 text-prompt 分割，PNG 写入 sam3_cache（与
mask_cache.CachedMasker 共享 key），几何记录入库 sam3_masks（未检出也落 NULL 行，保证可续跑）。

在 veraretouch-unified:dev 容器内跑（SAM3 = transformers 5.7 内置，依赖已预装）：
    docker run --rm --runtime nvidia --gpus '"device=0"' --network host \
        -v /home/bc/VeraRetouch:/work -v /home/bc/data:/home/bc/data -w /work \
        --entrypoint python3 veraretouch-unified:dev \
        -m dataset_build.source_qa.sam3_subjects [--limit N] [--shard 0/1]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from dataset_build.mask_cache import compute_regions, concept_slug, path_key
from . import db

_CACHE = "/home/bc/data/datasets/vera_directionA_1M/sam3_cache"


def _save_png(mask: np.ndarray, out_path: str) -> None:
    from PIL import Image
    a = (np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0) * 255.0).round().astype("uint8")
    tmp = out_path + ".tmp"
    Image.fromarray(a, mode="L").save(tmp, format="PNG", compress_level=1)
    os.replace(tmp, out_path)


def _pending(conn, shard_i: int, shard_n: int, limit: int) -> list:
    """(asset_id, path, [(concept, slug)]) still missing masks for this shard."""
    rows = conn.execute(
        "SELECT c.asset_id, s.path, c.subjects FROM source_captions c "
        "JOIN assets s USING(asset_id) ORDER BY c.asset_id").fetchall()
    done = {(r[0], r[1]) for r in conn.execute("SELECT asset_id, slug FROM sam3_masks").fetchall()}

    todo = []
    for r in rows:
        if shard_n > 1 and int(path_key(r["path"]), 16) % shard_n != shard_i:
            continue
        if not os.path.exists(r["path"]):
            continue
        pend, seen = [], set()
        for s in json.loads(r["subjects"] or "[]"):
            slug = concept_slug(s["en"])
            if slug in seen or (r["asset_id"], slug) in done:
                continue
            seen.add(slug)
            pend.append((s["en"], slug))
        if pend:
            todo.append((r["asset_id"], r["path"], pend))
        if limit and len(todo) >= limit:
            break
    return todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="0/1", help="i/n stable path-hash partition")
    ap.add_argument("--cache-dir", default=_CACHE)
    ap.add_argument("--follow", type=int, default=0, metavar="SECS",
                    help="stream mode: re-poll source_captions every SECS for newly captioned "
                         "sources until 3 consecutive empty polls — lets this GPU lane run "
                         "CONCURRENTLY with caption_subjects (vlm lane) instead of after it")
    a = ap.parse_args()
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))

    db.init_db()
    conn = db.connect()
    run_id = db.start_run(conn, "sam3_subjects", {"shard": a.shard, "follow": a.follow})
    print(f"[sam3] shard {shard_i}/{shard_n} (run {run_id}, follow={a.follow}s)", flush=True)

    masker = None
    n_img = n_mask = n_miss = n_fail = 0
    empty_polls = 0
    budget = a.limit
    while True:
        todo = _pending(conn, shard_i, shard_n, budget)
        if not todo:
            if not a.follow:
                break
            empty_polls += 1
            if empty_polls >= 3:
                break
            import time
            time.sleep(a.follow)
            continue
        empty_polls = 0
        if masker is None:
            from dataset_build.masking import Sam3Masker   # lazy: monetgpt_sam3 only
            masker = Sam3Masker()
        print(f"[sam3] batch: {len(todo)} images pending", flush=True)
        for aid, path, pend in todo:
            concepts = [c for c, _ in pend]
            try:
                cmaps = masker.masks(path, concepts)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"[sam3] fail {path}: {e}", file=sys.stderr)
                db.log_event(conn, aid, "sam3_subjects", "error", detail=str(e)[:300], run_id=run_id)
                conn.commit()
                continue
            d = os.path.join(a.cache_dir, path_key(path))
            os.makedirs(d, exist_ok=True)
            regions = compute_regions(cmaps)
            for concept, slug in pend:
                m = cmaps.get(concept)
                geo = regions.get(concept)
                png = None
                if m is not None and geo:
                    png = os.path.join(d, slug + ".png")
                    _save_png(m, png)
                    n_mask += 1
                else:
                    n_miss += 1
                conn.execute(
                    "INSERT INTO sam3_masks(asset_id, concept, slug, png_path, area, bbox, centroid, run_id) "
                    "VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(asset_id, slug) DO UPDATE SET "
                    "concept=EXCLUDED.concept, png_path=EXCLUDED.png_path, area=EXCLUDED.area, "
                    "bbox=EXCLUDED.bbox, centroid=EXCLUDED.centroid, run_id=EXCLUDED.run_id",
                    (aid, concept, slug, png,
                     geo["area"] if geo else None,
                     json.dumps(geo["bbox"]) if geo else None,
                     json.dumps(geo["centroid"]) if geo else None, run_id))
            conn.commit()
            n_img += 1
            if n_img % 100 == 0:
                print(f"[sam3] img={n_img} masks={n_mask} miss={n_miss} fail={n_fail}", flush=True)
        if budget:
            budget -= len(todo)
            if budget <= 0:
                break
        if not a.follow:
            break
    db.finish_run(conn, run_id, {"img": n_img, "masks": n_mask, "miss": n_miss, "fail": n_fail})
    conn.close()
    print(f"[sam3] DONE img={n_img} masks={n_mask} miss={n_miss} fail={n_fail}", flush=True)


if __name__ == "__main__":
    main()
