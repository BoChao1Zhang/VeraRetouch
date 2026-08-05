#!/usr/bin/env python
"""N-24 -- scan the whole corpus for the ``<color>`` token boundary.  NOT RUN YET.

Amendment A-4 fixed ``COLOR_CONTEXT_MAX_TOKENS = 384`` from a **3,745-record
sample** (max 324).  ``gt_color_context`` raises past the boundary, so one
over-long record in the remaining 165k would kill an arm mid-epoch, hours in,
with eleven arms queued behind it.  The review moved this from the post-run job
list (``WT-J9``) to a pre-run hard gate; :func:`q3vl.what.boundary.
require_color_boundary_scan` is the gate and this script produces what it reads.

Cost: one record read per sample, no images, no tokenisation (the sft2seg build
already wrote ``tokens.color`` with the same tokeniser the collator uses).
IO-bound on NFS; ~173k records across the five splits.

Usage (D-20: rm -f the log first, then verify with ``ps -p <PID>``, never pgrep):
    rm -f /home/bc/data/runs/what/scan_color_boundary.log
    nohup python -m q3vl.what.scripts.scan_color_boundary \
        > /home/bc/data/runs/what/scan_color_boundary.log 2>&1 &
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# See q3vl/what/scripts/run_what.py for the measurement; the store chain reaches
# sqlite3 and torch poisons it if it goes first.
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
from pathlib import Path

from q3vl.what.boundary import (
    DEFAULT_REPORT_PATH,
    SCANNED_SPLITS,
    merge_report,
    scan_split,
)
from q3vl.what.config import COLOR_CONTEXT_MAX_TOKENS
from q3vl.what.data import WhatDataset


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", nargs="+", default=list(SCANNED_SPLITS))
    ap.add_argument("--out", default=str(DEFAULT_REPORT_PATH))
    ap.add_argument("--boundary", type=int, default=COLOR_CONTEXT_MAX_TOKENS)
    ap.add_argument("--verify", default="none",
                    help="shard checksum mode; 'none' is enough for a record scan")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    per_split = []
    for split in args.splits:
        ds = WhatDataset(split, need_mask=False, verify=args.verify, limit=args.limit)
        row = scan_split(ds, split=split, boundary=args.boundary)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        per_split.append(row)

    report = merge_report(per_split, boundary=args.boundary)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "splits"},
                     ensure_ascii=False, indent=1), flush=True)
    if not report["ok"]:
        # a failing scan is a result, not a crash: the report is written either
        # way so the offending sample ids are on disk for the fix.
        print("SCAN FAILED: see over_boundary in the report", flush=True)
        return 1
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
