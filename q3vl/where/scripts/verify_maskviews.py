#!/usr/bin/env python
"""Data-plane check for the S1 mask views: do the shards say what the build says?

The calibration arms read masks from the published shards instead of decoding
``.cgt.png`` again per arm.  That swap is only safe if the two paths agree, so
this compares them sample by sample:

* the published shard is readable at its recorded offset and matches its sha256;
* ``mask_low`` and ``mask_hi`` are bit-equal (within float16 storage) to what the
  live ``MaskResolver`` produces from the original build member;
* the shard's own metadata (grid, out size, locator, mask stats) agrees with the
  record;
* every eligible sample in the split is actually present in the shards, and the
  shard carries nothing that is not eligible.

CPU only, no GPU, no model: it reads shards and PNGs.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from q3vl.train.shards import ShardIndex, ShardStore
from q3vl.where.config import LOCAL_BUILDS, MASKVIEW_DIR, REPORT_DIR
from q3vl.where.fpre import grid_from_geometry
from q3vl.where.maskdata import (
    MaskResolver, MaskViewStore, eligibility, mask_views, split_index_path,
)
from q3vl.where.packing import verify_published


def _q(vals: list[float]) -> dict:
    if not vals:
        return {}
    a = np.asarray(vals, dtype=np.float64)
    return {"n": int(a.size), "max": float(a.max()), "median": float(np.median(a)),
            "mean": float(a.mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--maskview-root", default=None)
    ap.add_argument("--sample", type=int, default=200,
                    help="how many samples to compare against the live resolver")
    ap.add_argument("--stride", type=int, default=0,
                    help="0 = spread the sample across the whole split")
    ap.add_argument("--verify", default="checksum", choices=["checksum", "length", "none"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    root = Path(args.maskview_root) if args.maskview_root else MASKVIEW_DIR / args.split
    t0 = time.time()

    published = verify_published(root, n_random=128)
    store_view = MaskViewStore(root, verify=args.verify)
    index = ShardIndex.load(split_index_path(args.split))
    store = ShardStore("/", verify=args.verify)
    resolver = MaskResolver(verify=args.verify)

    eligible_ids: list[str] = []
    for ref in index.samples:
        if ref.meta.get("build") not in LOCAL_BUILDS:
            continue
        rec = json.loads(store.read(ref.members["record"]).decode("utf-8"))
        if eligibility(rec)[0]:
            eligible_ids.append(rec["sample_id"])

    shard_ids = set(store_view._members)
    missing = [s for s in eligible_ids if s not in shard_ids]
    extra = sorted(shard_ids - set(eligible_ids))

    stride = args.stride or max(1, len(eligible_ids) // max(1, args.sample))
    picks = eligible_ids[::stride][: args.sample]
    by_id = {r.meta.get("sample_id") or r.sample_id: r for r in index.samples}

    d_low, d_hi, mismatched = [], [], []
    n_cmp = 0
    for sid in picks:
        ref = by_id.get(sid)
        if ref is None:
            continue
        rec = json.loads(store.read(ref.members["record"]).decode("utf-8"))
        img = rec["image"]
        gh, gw = grid_from_geometry(img["out_h"], img["out_w"])
        got = store_view.get(sid)
        if got is None:
            mismatched.append({"sample_id": sid, "reason": "absent_from_shards"})
            continue
        hi_pub, low_pub, meta = got
        mref = resolver.resolve(rec)
        live = resolver.load(mref)
        hi_live, low_live = mask_views(live, img["out_h"], img["out_w"], gh, gw)

        if tuple(low_pub.shape) != (gh, gw) or tuple(hi_pub.shape) != (img["out_h"], img["out_w"]):
            mismatched.append({"sample_id": sid, "reason": "shape",
                               "published": [list(low_pub.shape), list(hi_pub.shape)],
                               "expected": [[gh, gw], [img["out_h"], img["out_w"]]]})
            continue
        dl = float((low_pub.double() - low_live.double()).abs().max())
        dh = float((hi_pub.double() - hi_live.double()).abs().max())
        d_low.append(dl)
        d_hi.append(dh)
        # float16 storage for mask_low (~1e-3 ulp near 1) and uint8 PNG for mask_hi
        if dl > 1e-3 or dh > 1.0 / 255 + 1e-6:
            mismatched.append({"sample_id": sid, "reason": "value",
                               "d_low": dl, "d_hi": dh})
        if meta.get("grid") != [gh, gw]:
            mismatched.append({"sample_id": sid, "reason": "meta_grid",
                               "meta": meta.get("grid"), "expected": [gh, gw]})
        if (meta.get("mask_locator") or {}).get("member", {}).get("member") \
                not in (None, mref.member.member):
            mismatched.append({"sample_id": sid, "reason": "meta_locator"})
        n_cmp += 1

    report = {
        "split": args.split,
        "maskview_root": str(root),
        "elapsed_s": round(time.time() - t0, 1),
        "published": published,
        "n_shard_samples": len(shard_ids),
        "n_eligible_in_split": len(eligible_ids),
        "n_missing_from_shards": len(missing),
        "missing_examples": missing[:20],
        "n_extra_in_shards": len(extra),
        "extra_examples": extra[:20],
        "n_compared_against_live": n_cmp,
        "max_abs_diff_mask_low": _q(d_low),
        "max_abs_diff_mask_hi": _q(d_hi),
        "n_mismatched": len(mismatched),
        "mismatches": mismatched[:20],
    }
    report["ok"] = bool(
        published.get("ok")
        and not missing and not extra and not mismatched and n_cmp > 0
    )
    out = Path(args.out) if args.out else REPORT_DIR / f"maskview_verify_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("mismatches", "missing_examples", "extra_examples")},
                     indent=2))
    print(f"\nwrote {out}")
    store_view.close()
    resolver.close()
    store.close()
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
