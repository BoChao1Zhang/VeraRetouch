#!/usr/bin/env python
"""Mirror everything a Where-B arm streams onto the local disk (PERF-1).

Five published datasets feed one arm, all of them read at random over a shuffled
1-epoch permutation, i.e. always cold:

    sft2seg-20260804/records      0.62 GiB   the record JSON per sample
    sft2seg-20260804/images      22.16 GiB   the spec-5 JPEG per sample
    where_a/maskviews/<split>     1.45 GiB   the GT mask views
    where_a/oracle/<arm>/s5/<sp>  2.09 GiB   the Where-A latents
    where_b/genwhere/<split>      1.21 GiB   the generated <where> contexts
                                 -------
                                 27.5 GiB    ~4.5 min at the measured 106 MB/s

Run it once before the queue starts; the arms pick the cache up automatically
(``q3vl.train.shards.resolve_read_path``) and record what they used in
``run_setup.json`` under ``shard_cache``.

Usage (D-20: rm -f the log first, then verify with ``ps -p <PID>``):
    rm -f /home/bc/data/runs/where_b/warm_cache.log
    nohup python -m q3vl.whereb.scripts.warm_shard_cache \
        > /home/bc/data/runs/where_b/warm_cache.log 2>&1 &

The cache is rebuildable scratch: it holds byte-for-byte copies of immutable
published shards, verified against the ``tar_sha256`` their producers published,
and every member read out of it is still sha256-checked individually.  Deleting
the cache root costs one re-run and nothing else.
"""

from __future__ import annotations

import sqlite3  # noqa: F401  (import order guard, see run_where_b)

import argparse
import json
from pathlib import Path

from q3vl.train.shardcache import (
    DEFAULT_MAX_BYTES,
    FREE_SPACE_FLOOR_BYTES,
    cache_root_from_env,
    plan_dataset,
    preload,
    warm,
)
from q3vl.where.config import SFT2SEG_ROOT
from q3vl.whereb.config import (
    BASIS_ARM,
    GENCTX_DIR,
    ORACLE_NAMESPACE,
    WHERE_A_MASKVIEW_DIR,
    WHERE_A_ORACLE_DIR,
)


def sources(splits: list[str], basis_arm: str, namespace: str) -> list[Path]:
    """Every published root a Where-B arm streams, for the given splits."""
    roots = [SFT2SEG_ROOT / "records", SFT2SEG_ROOT / "images"]
    for split in splits:
        roots.append(Path(WHERE_A_MASKVIEW_DIR) / split)
        roots.append(Path(WHERE_A_ORACLE_DIR) / basis_arm / namespace / split)
        roots.append(Path(GENCTX_DIR) / split)
    return roots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", action="append", default=None,
                    help="repeatable; default: train and V_where")
    ap.add_argument("--basis-arm", default=BASIS_ARM)
    ap.add_argument("--oracle-namespace", default=ORACLE_NAMESPACE)
    ap.add_argument("--root", default=None, help="cache root")
    ap.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    ap.add_argument("--free-floor", type=int, default=FREE_SPACE_FLOOR_BYTES)
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-preload", action="store_true",
                    help="skip the page-cache preload that follows the copy")
    args = ap.parse_args(argv)

    splits = args.split or ["train", "V_where"]
    roots = sources(splits, args.basis_arm, args.oracle_namespace)
    entries = []
    for r in roots:
        got = plan_dataset(r)
        print(f"[plan] {r}: {len(got)} shards, "
              f"{sum(e.bytes for e in got) / 2**30:.2f} GiB", flush=True)
        entries += got
    # one dataset may be listed twice (same split twice); keep the first
    seen, uniq = set(), []
    for e in entries:
        if e.key not in seen:
            seen.add(e.key)
            uniq.append(e)

    root = Path(args.root) if args.root else cache_root_from_env()
    print(json.dumps({"cache_root": str(root), "splits": splits,
                      "n_shards": len(uniq),
                      "gib": round(sum(e.bytes for e in uniq) / 2**30, 2)},
                     indent=2), flush=True)
    if args.plan_only:
        return 0
    warm(uniq, root, max_bytes=args.max_bytes, free_floor=args.free_floor,
         force=args.force)
    # A run that copied nothing (the cache was already complete) has touched
    # nothing either, and cold pages cost ~100 ms per micro-batch.  Preloading is
    # the difference between "the cache exists" and "the cache is in RAM".
    if not args.no_preload:
        preload(root)
    return 0


if __name__ == "__main__":                                     # pragma: no cover
    raise SystemExit(main())
