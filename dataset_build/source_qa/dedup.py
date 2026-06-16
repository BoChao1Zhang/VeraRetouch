"""Source-image deduplication (runs AFTER ingest, BEFORE iqa/llm_qa so the
expensive NR-IQA + 35B judge never pay for redundant pixels).

Two tiers, both from established libraries (no hand-rolled hashing/search):
  * EXACT  : sha256 of the decoded RGB pixels (defeats re-encode / metadata noise,
             and collapses the ppr10k a/b/c triples that share one source file).
  * NEAR   : 64-bit DCT perceptual hash via `imagehash.phash`, clustered with a
             BK-tree (`pybktree`) at Hamming <= DEDUP_PHASH_HAMMING -> O(n).

Clustering = union-find over exact + near edges. Each cluster keeps a HEAD
(highest megapixels, then aesthetic_vlm, then bytes_size); non-heads get
`dup_of=head` + `dup_cluster=head`. Two kinds of non-head:
  * different-path duplicate  -> auto_verdict='drop' (a truly redundant source).
  * SAME-path sibling (ppr10k a/b/c: 3 expert TARGETS sharing one source file)
    -> NOT dropped. It is judged once on the head and the verdict is fanned out
    to the siblings at apply time (apply.py), so we save 2/3 of the IQA + 35B
    work without discarding two-thirds of a legitimate expert-target corpus.

iqa / llm_qa / gate only process `dup_of IS NULL` (heads); apply.py inherits the
head verdict for same-path siblings. dup_cluster also drives cluster-level
train/eval split assignment (leakage prevention) downstream.

Run: python -m dataset_build.source_qa.dedup [--limit N] [--corpus C]
                                              [--pixel-only] [--phash-hamming 6]
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from typing import Dict, List, Optional

from . import config, db


# --------------------------------------------------------------------------- #
# pass 1: decode -> pixel sha256 + pHash (+ native dimensions), resumable
# --------------------------------------------------------------------------- #
def _hash_one(path: str):
    import numpy as np
    from PIL import Image
    import imagehash
    Image.MAX_IMAGE_PIXELS = None
    im = Image.open(path); im.load(); im = im.convert("RGB")
    w, h = im.size
    arr = np.asarray(im, dtype="uint8")
    px = hashlib.sha256(arr.tobytes()).hexdigest()
    ph = imagehash.phash(im)                       # 64-bit DCT pHash
    ph_hex = str(ph)
    ph_int = int(ph_hex, 16)
    return px, ph_hex, ph_int, w, h


def compute_hashes(conn, run_id: str, limit: Optional[int], corpus: Optional[str],
                   workers: int = 16) -> int:
    where = ["asset_type='image'", "pixel_sha256 IS NULL"]
    params: list = []
    if corpus:
        where.append("corpus=?"); params.append(corpus)
    sql = f"SELECT asset_id, path FROM assets WHERE {' AND '.join(where)}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    rows = conn.execute(sql, params).fetchall()
    print(f"[dedup] hashing {len(rows)} images (workers={workers})", file=sys.stderr)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def work(r):
        try:
            px, ph_hex, _, w, h = _hash_one(r["path"])
            return r["asset_id"], (px, ph_hex, w, h), None
        except Exception as e:
            return r["asset_id"], None, str(e)[:200]

    # decode/hash in parallel (I/O + C-level PIL releases the GIL); DB writes stay
    # serialized in this thread for psycopg-connection safety.
    n_ok = n_err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, r) for r in rows]
        for i, fut in enumerate(as_completed(futs)):
            aid, res, err = fut.result()
            if err:
                db.log_event(conn, aid, "dedup", "error", {"err": err}, run_id)
                n_err += 1
            else:
                px, ph_hex, w, h = res
                db.update_asset_fields(conn, aid, pixel_sha256=px, phash=ph_hex,
                                       width=w, height=h, megapixels=round(w * h / 1e6, 4))
                n_ok += 1
            if (i + 1) % 2000 == 0:
                conn.commit()
                print(f"[dedup] hashed {i+1}/{len(rows)} ok={n_ok} err={n_err}", file=sys.stderr)
    conn.commit()
    return n_ok


# --------------------------------------------------------------------------- #
# pass 2: cluster (union-find over exact + near edges) and assign heads
# --------------------------------------------------------------------------- #
class _UF:
    def __init__(self):
        self.p: Dict[str, str] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:        # path compression
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def cluster(conn, run_id: str, near: bool, phash_hamming: int) -> dict:
    rows = conn.execute(
        "SELECT asset_id, corpus, path, pixel_sha256, phash, megapixels, aesthetic_vlm, bytes_size "
        "FROM assets WHERE asset_type='image' AND pixel_sha256 IS NOT NULL").fetchall()
    info = {r["asset_id"]: r for r in rows}
    uf = _UF()
    for aid in info:
        uf.find(aid)

    # exact edges: same decoded-pixel sha256
    by_px: Dict[str, List[str]] = defaultdict(list)
    for aid, r in info.items():
        by_px[r["pixel_sha256"]].append(aid)
    for ids in by_px.values():
        for o in ids[1:]:
            uf.union(ids[0], o)
    exact_groups = sum(1 for ids in by_px.values() if len(ids) > 1)

    # near edges: pHash Hamming <= threshold via BK-tree over unique hashes
    near_pairs = 0
    if near:
        import pybktree
        by_ph: Dict[int, List[str]] = defaultdict(list)
        for aid, r in info.items():
            if r["phash"]:
                by_ph[int(r["phash"], 16)].append(aid)
        uniq = list(by_ph)
        tree = pybktree.BKTree(pybktree.hamming_distance, uniq)
        for h in uniq:
            for dist, h2 in tree.find(h, phash_hamming):
                if h2 != h:
                    uf.union(by_ph[h][0], by_ph[h2][0])
                    near_pairs += 1
        # also union assets that share an identical phash bucket
        for ids in by_ph.values():
            for o in ids[1:]:
                uf.union(ids[0], o)

    # gather clusters
    clusters: Dict[str, List[str]] = defaultdict(list)
    for aid in info:
        clusters[uf.find(aid)].append(aid)

    def _key(aid):
        r = info[aid]
        return (r["megapixels"] or -1.0, r["aesthetic_vlm"] or -1.0, r["bytes_size"] or -1)

    n_clusters = n_dup = n_dropped = n_fanout = 0
    for members in clusters.values():
        if len(members) == 1:
            continue
        n_clusters += 1
        head = max(members, key=_key)
        head_px = info[head]["pixel_sha256"]
        head_path = info[head]["path"]
        for m in members:
            if m == head:
                db.update_asset_fields(conn, m, dup_cluster=head)
                continue
            same_file = (info[m]["pixel_sha256"] == head_px and info[m]["path"] == head_path)
            if same_file:
                # ppr10k-style expert target sibling: keep, fan-out verdict at apply
                db.update_asset_fields(conn, m, dup_of=head, dup_cluster=head)
                n_fanout += 1
            else:
                db.mark_dup(conn, m, head)               # dup_of + dup_cluster + auto 'drop'
                n_dropped += 1
            db.log_event(conn, m, "dedup", "ok",
                         {"dup_of": head, "kind": "fanout" if same_file else "drop"}, run_id)
            n_dup += 1
    conn.commit()

    # cross-corpus exact-collision report (quantifies the unmeasured DEDUP-4 risk)
    cross = defaultdict(int)
    for ids in by_px.values():
        corpora = sorted({info[a]["corpus"] for a in ids})
        if len(corpora) > 1:
            cross[" x ".join(corpora)] += 1
    summary = {"clusters": n_clusters, "dup_members": n_dup, "dropped": n_dropped,
               "fanout_siblings": n_fanout, "exact_groups": exact_groups,
               "near_pairs": near_pairs, "cross_corpus_exact": dict(cross)}
    return summary


def run(limit: Optional[int] = None, corpus: Optional[str] = None,
        near: bool = True, phash_hamming: Optional[int] = None, pixel_only: bool = False,
        workers: int = 16) -> dict:
    phash_hamming = phash_hamming if phash_hamming is not None else config.DEDUP_PHASH_HAMMING
    conn = db.connect()
    run_id = db.start_run(conn, "dedup", {"near": near and not pixel_only, "hamming": phash_hamming})
    n_hashed = compute_hashes(conn, run_id, limit, corpus, workers=workers)
    summary = cluster(conn, run_id, near=(near and not pixel_only), phash_hamming=phash_hamming)
    summary["hashed"] = n_hashed
    db.finish_run(conn, run_id, summary)
    conn.close()
    import json
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--pixel-only", action="store_true", help="tier-1 exact dedup only (skip pHash)")
    ap.add_argument("--phash-hamming", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    run(limit=args.limit, corpus=args.corpus, pixel_only=args.pixel_only,
        phash_hamming=args.phash_hamming, workers=args.workers)


if __name__ == "__main__":
    main()
