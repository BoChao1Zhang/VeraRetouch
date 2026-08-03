#!/usr/bin/env python3
"""PPR10K contamination re-verification (DATA_ASSIGNMENT §4-A, frozen as a tool).

Known conclusion to reproduce: builds only used ppr10k/source flat indices
1-8871; official val territory (index >= 8875, i.e. beyond the first 8,875
train files per the official README) has zero hits.

Additionally audits the mmart_ppr10k pool (MMArt-PPR10k is built on PPR10K
raw photos; sample dirs are named <group_id>_<photo_id>): reports the group-id
distribution and hits above the paper-derived train-group boundary (1356).
That pool was NOT covered by the original §4-A verification.

Outputs: experiments/tooling-wave1/data_splits/ppr10k_verify.{json,md}
Exit code: 0 iff the ppr10k/source check is clean (mmart findings are
advisory and do not affect the exit code).

Usage: python3 verify_ppr10k.py [--limit N]
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vr_common as C  # noqa: E402

IDX_RE = re.compile(r"(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    builds = C.completed_builds()
    ppr_sources: dict[str, int] = {}          # basename -> flat index
    ppr_group_occupancy = collections.Counter()   # basename -> n groups
    mmart_sources: dict[str, int] = {}        # dirname -> group id
    mmart_group_occupancy = collections.Counter()
    per_build = {}

    for b in builds:
        n_ppr = n_mmart = 0
        for g in C.iter_groups(b, limit=args.limit):
            sp = g.get("source_path", "")
            pool = C.pool_of(sp)
            if pool == "ppr10k":
                base = os.path.basename(sp)
                # last digit run: basenames look like "ppr10k_008871.jpg" and the
                # pool name itself contains the digits "10"
                runs = IDX_RE.findall(os.path.splitext(base)[0])
                idx = int(runs[-1]) if runs else -1
                ppr_sources[base] = idx
                ppr_group_occupancy[base] += 1
                n_ppr += 1
            elif pool == "mmart_ppr10k":
                d = os.path.basename(os.path.dirname(sp))
                m = re.match(r"(\d+)_(\d+)$", d)
                gid = int(m.group(1)) if m else -1
                mmart_sources[d] = gid
                mmart_group_occupancy[d] += 1
                n_mmart += 1
        per_build[b] = {"ppr10k_groups": n_ppr, "mmart_groups": n_mmart}

    idxs = sorted(ppr_sources.values())
    n_ge_boundary = sum(1 for i in idxs if i >= C.PPR10K_VAL_FIRST_INDEX)
    unparsed = sum(1 for i in idxs if i < 0)
    clean = (n_ge_boundary == 0 and unparsed == 0 and len(idxs) > 0)

    gids = sorted(mmart_sources.values())
    gid_hist = collections.Counter(g // 100 * 100 for g in gids)
    n_gid_suspect = sum(1 for g in gids if g >= C.PPR10K_TRAIN_GROUPS_PAPER)

    result = {
        "builds": per_build,
        "ppr10k": {
            "distinct_sources": len(ppr_sources),
            "total_group_occupancy": sum(ppr_group_occupancy.values()),
            "index_min": idxs[0] if idxs else None,
            "index_max": idxs[-1] if idxs else None,
            "official_val_first_index": C.PPR10K_VAL_FIRST_INDEX,
            "hits_at_or_above_val_boundary": n_ge_boundary,
            "unparsed_basenames": unparsed,
            "clean": clean,
        },
        "mmart_ppr10k": {
            "distinct_sources": len(mmart_sources),
            "total_group_occupancy": sum(mmart_group_occupancy.values()),
            "group_id_min": gids[0] if gids else None,
            "group_id_max": gids[-1] if gids else None,
            "paper_train_group_boundary": C.PPR10K_TRAIN_GROUPS_PAPER,
            "sources_with_gid_ge_boundary": n_gid_suspect,
            "gid_histogram_by_100": {str(k): v for k, v in sorted(gid_hist.items())},
            "note": "MMArt-PPR10k is built on PPR10K raw photos; official val = "
                    "last 2,286 files (files ordered by group), so high group ids "
                    "overlap official val. Boundary 1356 is paper-derived "
                    "(1,356 train groups), not re-verified verbatim - advisory.",
        },
        "provenance": {
            "official_split_quote": "train with the first 8,875 files and validate "
                                     "with the last 2286 files (github.com/csjliang/PPR10K README, "
                                     "fetched 2026-08-02)",
            "totals_quote": "1,681 groups and 11,161 photos (arXiv 2105.09180 abstract)",
        },
        "limit": args.limit,
    }

    os.makedirs(C.REPORT_DIR, exist_ok=True)
    suffix = ".sample" if args.limit is not None else ""
    with open(os.path.join(C.REPORT_DIR, f"ppr10k_verify{suffix}.json"), "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    md = []
    md.append("# PPR10K 无污染复核报告\n")
    md.append(f"- 完成 build：{', '.join(builds)}")
    p = result["ppr10k"]
    md.append(f"- `ppr10k/source` 不同源：**{p['distinct_sources']}**，组占用 **{p['total_group_occupancy']}**")
    md.append(f"- 文件索引范围：**{p['index_min']}–{p['index_max']}**（官方 train = 前 8,875 个文件）")
    md.append(f"- 索引 ≥{C.PPR10K_VAL_FIRST_INDEX}（官方 val 段）命中：**{p['hits_at_or_above_val_boundary']}**")
    md.append(f"- 结论：**{'干净，无官方 val 污染' if p['clean'] else '不干净或数据异常，需人工排查'}**\n")
    m = result["mmart_ppr10k"]
    md.append("## 附：mmart_ppr10k 池（原 §4-A 结论未覆盖，提请裁决）\n")
    md.append(f"- 不同源 {m['distinct_sources']}，组占用 {m['total_group_occupancy']}，group id 范围 "
              f"{m['group_id_min']}–{m['group_id_max']}")
    md.append(f"- group id ≥{C.PPR10K_TRAIN_GROUPS_PAPER}（论文口径 val 组段，advisory）：**{m['sources_with_gid_ge_boundary']}** 源")
    md.append(f"- {m['note']}")
    with open(os.path.join(C.REPORT_DIR, f"ppr10k_verify{suffix}.md"), "w") as f:
        f.write("\n".join(md) + "\n")

    print(json.dumps(result["ppr10k"], ensure_ascii=False, indent=2))
    print("mmart_ppr10k advisory:", json.dumps(
        {k: m[k] for k in ("distinct_sources", "group_id_max", "sources_with_gid_ge_boundary")},
        ensure_ascii=False))
    return 0 if clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
