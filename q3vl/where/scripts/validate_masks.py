#!/usr/bin/env python
"""Sampled validation of the Where-A GT mask source (task card item 7).

Answers, from the published data only:

1. can every eligible local sample's ``.cgt.png`` be located from its
   ``sft2seg`` record (``image.origin.root`` + ``source_sample_id``)?
2. is it a single-channel soft mask, at the build's render resolution and at the
   EXIF-oriented image's aspect ratio?
3. does it actually mark the edited region -- i.e. is ``|I_tar - I_in|``
   concentrated where the mask is high?  (This is the check that catches a
   *wrong but plausible* mask; the format checks do not.)

Read-only.  ``--limit`` caps the number of samples so it can run while the two
GPUs are busy with Base SFT.

Usage:
    python -m q3vl.where.scripts.validate_masks --split V_where --limit 200 \
        --pixel-check 60 --out report.json
"""

from __future__ import annotations

import argparse
import collections
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from q3vl.train.shards import MemberRef, ShardIndex, ShardStore
from q3vl.where.config import LOCAL_BUILDS, REPORT_DIR
from q3vl.where.fpre import grid_from_geometry
from q3vl.where.maskdata import (
    MaskResolver, aspect_check, eligibility, mask_stats, mask_views, split_index_path,
)


def _pct(x, n):
    return round(100.0 * x / max(1, n), 2)


def _quantiles(v):
    if not v:
        return {}
    a = np.asarray(v, dtype=np.float64)
    return {
        "n": int(a.size), "mean": float(a.mean()), "p05": float(np.quantile(a, 0.05)),
        "median": float(np.median(a)), "p95": float(np.quantile(a, 0.95)),
        "min": float(a.min()), "max": float(a.max()),
    }


def _sibling(resolver: MaskResolver, root: str, src: str, suffixes: tuple[str, ...]):
    cat = resolver._catalog(root)
    if cat is None:
        return None
    for suf in suffixes:
        row = cat.execute(
            "select shard, member, offset_data, size, sha256 from members "
            "where sample_id=? and suffix=?", (src, suf),
        ).fetchone()
        if row:
            shard, member, offset, size, sha = row
            return MemberRef(shard=str(Path(root) / "shards" / f"{shard}.tar"),
                             member=member, offset=int(offset), length=int(size),
                             size=int(size), checksum=f"sha256:{sha}")
    return None


def _load_rgb(data: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0


def _resize_to(a: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    if a.shape[:2] == tuple(hw):
        return a
    t = torch.from_numpy(a).permute(2, 0, 1)[None]
    t = torch.nn.functional.interpolate(t, size=tuple(hw), mode="bilinear",
                                        align_corners=False)
    return t[0].permute(1, 2, 0).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--pixel-check", type=int, default=60,
                    help="how many samples also decode I_in/I_tar (heavier IO)")
    ap.add_argument("--include-low", action="store_true",
                    help="do not filter winner_confidence=low")
    ap.add_argument("--verify", default="checksum", choices=["checksum", "length", "none"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    store = ShardStore("/", verify=args.verify)
    resolver = MaskResolver(verify=args.verify)
    index = ShardIndex.load(split_index_path(args.split))

    counters: collections.Counter = collections.Counter()
    by_build: collections.Counter = collections.Counter()
    conf: collections.Counter = collections.Counter()
    exif: collections.Counter = collections.Counter()
    modes: collections.Counter = collections.Counter()
    frac_soft, mask_mean, aspect_err, low_std = [], [], [], []
    inside_ratio, corr_vals = [], []
    failures: list[dict] = []
    seen = 0
    t0 = time.time()

    for ref in index.samples:
        if seen >= args.limit:
            break
        if ref.meta.get("build") not in LOCAL_BUILDS:
            continue
        record = json.loads(store.read(ref.members["record"]).decode("utf-8"))
        ok, reason = eligibility(record, exclude_low=not args.include_low)
        if not ok:
            counters[f"skip_{reason}"] += 1
            continue
        seen += 1
        by_build[record["build"]] += 1
        conf[record.get("winner_confidence")] += 1
        img_meta = record["image"]
        exif[img_meta.get("exif_orientation")] += 1
        counters["upscaled"] += int(bool(img_meta.get("upscaled")))

        try:
            mref = resolver.resolve(record)
        except Exception as exc:                       # noqa: BLE001
            counters["resolve_failed"] += 1
            failures.append({"sample_id": record["sample_id"], "stage": "resolve",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        counters["resolved"] += 1
        try:
            mask = resolver.load(mref)
        except Exception as exc:                       # noqa: BLE001
            counters["load_failed"] += 1
            failures.append({"sample_id": record["sample_id"], "stage": "load",
                             "error": f"{type(exc).__name__}: {exc}"})
            continue
        counters["loaded"] += 1
        modes["L_uint8"] += 1

        ac = aspect_check(record, mask)
        aspect_err.append(ac["rel_error"])
        counters["aspect_ok"] += int(ac["ok"])
        gh, gw = grid_from_geometry(img_meta["out_h"], img_meta["out_w"])
        hi, low = mask_views(mask, img_meta["out_h"], img_meta["out_w"], gh, gw)
        st = mask_stats(low)
        counters["degenerate"] += int(st["degenerate"])
        frac_soft.append(float(((mask > 0.04) & (mask < 0.96)).mean()))
        mask_mean.append(st["mean"])
        low_std.append(st["std"])
        if float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
            counters["value_range_violation"] += 1

        if counters["loaded"] <= args.pixel_check:
            i_tar = _sibling(resolver, mref.root, mref.source_sample_id, (".jpg",))
            i_in = MemberRef(
                shard=str(Path(img_meta["origin"]["root"]) / "shards"
                          / f"{img_meta['origin']['shard']}.tar"),
                member=img_meta["origin"]["member"], offset=img_meta["origin"]["offset"],
                length=img_meta["origin"]["length"], size=img_meta["origin"]["length"],
                checksum=f"sha256:{img_meta['origin']['sha256']}",
            )
            try:
                # I_in is the *source* image (full resolution); I_tar and the
                # mask are the build's render (short side 1024).  Compare all
                # three on the mask's own grid.
                a = _resize_to(_load_rgb(store.read(i_in)), mask.shape)
                b = _resize_to(_load_rgb(store.read(i_tar)), mask.shape)
                if a.shape != b.shape:
                    counters["pixel_shape_mismatch"] += 1
                else:
                    d = np.abs(a - b).mean(-1)
                    m = mask
                    tot = d.sum() + 1e-9
                    inside_ratio.append(float((d * m).sum() / tot))
                    dv, mv = d.reshape(-1), m.reshape(-1)
                    if dv.std() > 1e-8 and mv.std() > 1e-8:
                        corr_vals.append(float(np.corrcoef(dv, mv)[0, 1]))
                    counters["pixel_checked"] += 1
            except Exception as exc:                   # noqa: BLE001
                counters["pixel_check_failed"] += 1
                failures.append({"sample_id": record["sample_id"], "stage": "pixel",
                                 "error": f"{type(exc).__name__}: {exc}"})

    report = {
        "split": args.split,
        "n_examined": seen,
        "elapsed_s": round(time.time() - t0, 1),
        "counters": dict(counters),
        "by_build": dict(by_build),
        "winner_confidence": {str(k): v for k, v in conf.items()},
        "exif_orientation": {str(k): v for k, v in exif.items()},
        "pil_modes": dict(modes),
        "resolve_rate_pct": _pct(counters["resolved"], seen),
        "load_rate_pct": _pct(counters["loaded"], seen),
        "aspect_ok_pct": _pct(counters["aspect_ok"], counters["loaded"]),
        "upscaled_pct": _pct(counters["upscaled"], seen),
        "degenerate_pct": _pct(counters["degenerate"], counters["loaded"]),
        "locator_source": {"catalog_hits": resolver.n_catalog_hits,
                           "jsonl_hits": resolver.n_jsonl_hits},
        "frac_soft_pixels": _quantiles(frac_soft),
        "mask_low_mean": _quantiles(mask_mean),
        "mask_low_std": _quantiles(low_std),
        "aspect_rel_error": _quantiles(aspect_err),
        "edit_energy_inside_mask": _quantiles(inside_ratio),
        "corr_absdiff_vs_mask": _quantiles(corr_vals),
        "failures": failures[:50],
        "checksum_verified_reads": store.n_checksum_verified + resolver._store.n_checksum_verified,
    }
    out = Path(args.out) if args.out else REPORT_DIR / f"mask_validation_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    print(f"\nwrote {out}")
    store.close()
    resolver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
