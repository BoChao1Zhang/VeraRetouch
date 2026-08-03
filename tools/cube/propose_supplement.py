#!/usr/bin/env python3
"""T2 action item E: propose the supplement list that tops D-CUBE up to ~4000.

Candidate pool = on-disk LUTs that are (a) not used in production, (b) parse-ok,
(c) not byte- or content-identical to a used preset (md5), (d) not near-identity
per selfcheck. The proposal picks --deficit of them deterministically (sorted by
id, round-robin over buckets so quandian/e18 both contribute); final adoption is
the orchestrator's decision — the full pool is emitted alongside.

Usage:
  python3 propose_supplement.py --manifest dcube_manifest.jsonl \
      --parse-report parse_report.jsonl --nearident near_identity_stats.jsonl \
      --deficit 478 --out-dir DIR
"""

from __future__ import annotations

import argparse
import collections
import json
import os


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--parse-report", required=True)
    ap.add_argument("--nearident", required=True)
    ap.add_argument("--deficit", type=int, default=478)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    parse_ok = {r["id"] for r in map(json.loads, open(args.parse_report))
                if r["ok"]}
    near_ident = {r["id"] for r in map(json.loads, open(args.nearident))
                  if r["near_identity"]}

    pool, excluded = [], collections.Counter()
    md5_seen = set()
    rows = [json.loads(l) for l in open(args.manifest)]
    for r in rows:
        if r["used_in_prod"]:
            md5_seen.add(r["md5"])
    for r in sorted(rows, key=lambda x: x["id"]):
        if r["used_in_prod"]:
            continue
        if r["id"] not in parse_ok:
            excluded["parse_failed"] += 1
            continue
        if r["dup_of_used"] or r["md5"] in md5_seen:
            excluded["dup_of_used_or_pool"] += 1
            continue
        if r["id"] in near_ident:
            excluded["near_identity"] += 1
            continue
        md5_seen.add(r["md5"])  # also dedupe within the pool itself
        pool.append(r)

    by_bucket = collections.defaultdict(list)
    for r in pool:
        by_bucket[r["bucket"]].append(r)
    buckets = sorted(by_bucket)
    picked, idx = [], {b: 0 for b in buckets}
    while len(picked) < args.deficit:
        progressed = False
        for b in buckets:
            if idx[b] < len(by_bucket[b]) and len(picked) < args.deficit:
                picked.append(by_bucket[b][idx[b]])
                idx[b] += 1
                progressed = True
        if not progressed:
            break

    with open(os.path.join(args.out_dir, "supplement_pool.txt"), "w") as f:
        f.write("\n".join(r["path"] for r in pool) + ("\n" if pool else ""))
    with open(os.path.join(args.out_dir, "supplement_proposal.txt"), "w") as f:
        f.write("\n".join(r["path"] for r in picked) + ("\n" if picked else ""))
    summary = {
        "pool_size": len(pool),
        "excluded": dict(excluded),
        "proposed": len(picked),
        "proposed_by_bucket": dict(collections.Counter(r["bucket"] for r in picked)),
        "target_total": 3522 + len(picked),
    }
    with open(os.path.join(args.out_dir, "supplement_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
