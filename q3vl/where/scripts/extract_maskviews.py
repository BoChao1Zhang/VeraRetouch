#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  Publish the Where-A GT mask views as shards.

Reads every eligible local sample of a split, resolves its ``.cgt.png`` in the
original build batch, projects it onto the two Where-A views (spec-5 grid and
F_pre grid) and publishes them as indexed tar shards under
``WHERE_A_ROOT/maskviews/<split>`` (protocol 2.3).

Why it is a separate job: the projection is pure IO + PIL and needs no GPU, so
it can run once and be reused by all four arms and by every later Where-B run;
re-decoding a 1024-short-side PNG 4x per epoch would not.

**Do not start this while Base SFT is training** -- it reads the same NFS build
tree and the same local scratch the trainer is streaming from.  Run order:

    1. Base SFT finishes;
    2. this job (train, V_where, V_what, T_final, T_lut_unseen);
    3. `python -m q3vl.where.preflight` (protocol 14 items 4/5/6);
    4. `run_calibration.py` per arm.

Usage:
    python -m q3vl.where.scripts.extract_maskviews --split train
    python -m q3vl.where.scripts.extract_maskviews --split V_where
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


from q3vl.train.shards import ShardIndex, ShardStore
from q3vl.where.config import (
    EXCLUDE_WINNER_CONFIDENCE_LOW, LOCAL_BUILDS, MASKVIEW_DIR, REPORT_DIR,
)
from q3vl.where.fpre import grid_from_geometry
from q3vl.where.maskdata import (
    MaskResolver, aspect_check, eligibility, mask_stats, mask_views, split_index_path,
)
from q3vl.where.packing import pack_maskviews, verify_published


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--include-low", action="store_true")
    ap.add_argument("--verify", default="checksum", choices=["checksum", "length", "none"])
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else MASKVIEW_DIR / args.split
    if out_root.exists():
        raise SystemExit(f"{out_root} already exists; publication is atomic, "
                         f"move the old one aside instead of overwriting it")

    store = ShardStore("/", verify=args.verify)
    resolver = MaskResolver(verify=args.verify)
    index = ShardIndex.load(split_index_path(args.split))
    rejections: list[dict] = []
    stats: list[dict] = []
    t0 = time.time()

    def rows():
        n = 0
        for ref in index.samples:
            if args.limit is not None and n >= args.limit:
                break
            if ref.meta.get("build") not in LOCAL_BUILDS:
                continue
            record = json.loads(store.read(ref.members["record"]).decode("utf-8"))
            ok, reason = eligibility(
                record, exclude_low=not (args.include_low or not EXCLUDE_WINNER_CONFIDENCE_LOW)
            )
            if not ok:
                rejections.append({"sample_id": record["sample_id"], "reason": reason})
                continue
            try:
                mref = resolver.resolve(record)
                mask = resolver.load(mref)
            except Exception as exc:                   # noqa: BLE001
                rejections.append({"sample_id": record["sample_id"], "reason": "mask_io",
                                   "error": f"{type(exc).__name__}: {exc}"})
                continue
            img = record["image"]
            ac = aspect_check(record, mask)
            if not ac["ok"]:
                rejections.append({"sample_id": record["sample_id"],
                                   "reason": "aspect_mismatch", "detail": ac})
                continue
            gh, gw = grid_from_geometry(img["out_h"], img["out_w"])
            hi, low = mask_views(mask, img["out_h"], img["out_w"], gh, gw)
            st = mask_stats(low)
            stats.append({"sample_id": record["sample_id"], **st,
                          "build": record["build"],
                          "winner_confidence": record.get("winner_confidence"),
                          "upscaled": img.get("upscaled")})
            meta = {
                "build": record["build"], "build_id": record.get("build_id"),
                "source_sample_id": record["source_sample_id"],
                "source_image_id": record.get("source_image_id"),
                "lut_id": record.get("lut_id"), "region": record.get("region"),
                "winner_confidence": record.get("winner_confidence"),
                "upscaled": img.get("upscaled"), "split": args.split,
                "grid": [gh, gw], "out": [img["out_h"], img["out_w"]],
                "mask_raw_shape": list(mask.shape),
                "mask_locator": mref.to_dict(), "mask_stats": st,
                "aspect": ac,
            }
            n += 1
            yield record["sample_id"], low, hi, meta

    manifest = pack_maskviews(rows(), out_root, source_label=f"where_a.maskviews/{args.split}")
    report = {
        "split": args.split,
        "out_root": str(out_root),
        "elapsed_s": round(time.time() - t0, 1),
        "manifest": {k: manifest[k] for k in
                     ("dataset_id", "sample_count", "member_count", "shard_count",
                      "payload_bytes", "status")},
        "n_rejected": len(rejections),
        "rejection_reasons": {r: sum(1 for x in rejections if x["reason"] == r)
                              for r in {x["reason"] for x in rejections}},
        "degenerate": sum(1 for s in stats if s["degenerate"]),
        "upscaled": sum(1 for s in stats if s["upscaled"]),
        "by_build": {b: sum(1 for s in stats if s["build"] == b) for b in LOCAL_BUILDS},
        "verify": verify_published(out_root, n_random=64),
    }
    rep_dir = REPORT_DIR / "maskviews"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{args.split}.report.json").write_text(json.dumps(report, indent=2))
    with (rep_dir / f"{args.split}.rejections.jsonl").open("w") as fh:
        for r in rejections:
            fh.write(json.dumps(r) + "\n")
    with (rep_dir / f"{args.split}.mask_stats.jsonl").open("w") as fh:
        for s in stats:
            fh.write(json.dumps(s) + "\n")
    print(json.dumps(report, indent=2))
    store.close()
    resolver.close()
    return 0 if report["verify"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
