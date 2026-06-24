#!/usr/bin/env python3
"""Mine local-mask templates from the real Lightroom preset corpus.

WS-B Step 1. The old random radial/linear masks (_degrade_mask) had no aesthetic
grounding. Instead we mine how *real* presets use local masks: parse every
preset's CorrectionMasks geometry + its Local*2012 edit vector, then bucket by
(mask_type, position) and summarise each bucket with the median + IQR of its
geometry and edit direction. The result (template_bank.json) gives build-time
anchoring real anchors ("circular over the sky -> darken + lift highlights")
that get pinned to each image's SAM3 region and perturbed.

Run:
  python -m dataset_build.tools.mine_mask_templates \
      --recipe-root /home/bc/data/datasets/recipes \
      --out /home/bc/data/datasets/vera_directionA_1M/mask_templates/template_bank.json

ponytail: rule-based bucketing (not KMeans) — the corpus is ~1.3k masks, small
enough that fixed position/angle bins are more stable and interpretable.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

from dataset_build.recipes import parse_local_masks

# Local-edit param -> aspect (mirrors config.cgt L/GC/SC split). Keys are the
# crs:Local* attribute names parse_local_masks returns.
_ASPECT_OF: Dict[str, str] = {
    "LocalExposure2012": "L", "LocalContrast2012": "L", "LocalHighlights2012": "L",
    "LocalShadows2012": "L", "LocalWhites2012": "L", "LocalBlacks2012": "L",
    "LocalClarity2012": "L", "LocalDehaze": "L",
    "LocalTemperature": "GC", "LocalTint": "GC",
    "LocalToningHue": "SC", "LocalToningSaturation": "SC", "LocalSaturation": "SC",
}

# Rough native scale per param so aspect dominance isn't hijacked by units:
# ToningHue is a 0-360 *angle*, not a magnitude; everything else is ~[-1,1].
_PARAM_SCALE: Dict[str, float] = {"LocalToningHue": 180.0}


def _f(geom: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(geom.get(key, default))
    except (TypeError, ValueError):
        return default


def _norm_geom(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Project a raw mask to a size-independent geometry, or None if inert."""
    g = rec["geom"]
    mt = rec["mask_type"]
    if mt == "circulargradient":
        t, l, b, r = _f(g, "Top"), _f(g, "Left"), _f(g, "Bottom"), _f(g, "Right")
        rx, ry = (r - l) / 2.0, (b - t) / 2.0
        if rx <= 0 and ry <= 0:
            return None  # degenerate/placeholder mask
        return {
            "shape": "radial",
            "center": [(l + r) / 2.0, (t + b) / 2.0],
            "radius": [abs(rx), abs(ry)],
            "angle": _f(g, "Angle"),
            "feather": _f(g, "Feather") / 100.0,
            "flipped": str(g.get("Flipped", "false")).lower() == "true",
        }
    if mt == "gradient":
        zx, zy, fx, fy = _f(g, "ZeroX"), _f(g, "ZeroY"), _f(g, "FullX"), _f(g, "FullY")
        dx, dy = fx - zx, fy - zy
        width = math.hypot(dx, dy)
        if width <= 1e-6:
            return None
        return {
            "shape": "linear",
            "center": [(zx + fx) / 2.0, (zy + fy) / 2.0],
            "angle": math.atan2(dy, dx),
            "width": width,
        }
    # AI / image / paint masks: no usable geometry, semantic only
    return {"shape": "semantic"}


def _bucket_id(ng: Dict[str, Any], rec: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    """Return (bucket_id, anchor_concept)."""
    if rec["is_ai"] or ng["shape"] == "semantic":
        label = (rec["what"] or "ai").rsplit("/", 1)[-1].lower() or "ai"
        concept = {"sky": "sky", "subject": "subject", "person": "person",
                   "people": "person", "background": "background"}.get(label)
        return f"ai_{label}", concept
    cx, cy = ng["center"]
    xbin = "left" if cx < 0.4 else "right" if cx > 0.6 else "ctr"
    ybin = "top" if cy < 0.4 else "bot" if cy > 0.6 else "mid"
    if ng["shape"] == "radial":
        bid = f"circ_{ybin}_{xbin}"
    else:  # linear: bucket by orientation
        a = ng["angle"] % math.pi
        ob = ("h" if a < math.pi / 8 or a > 7 * math.pi / 8
              else "v" if 3 * math.pi / 8 < a < 5 * math.pi / 8
              else "d1" if a < math.pi / 2 else "d2")
        bid = f"grad_{ob}"
    # coarse semantic anchor from position: top -> sky, central -> subject
    concept = "sky" if cy < 0.4 else ("subject" if 0.35 <= cx <= 0.65 else None)
    return bid, concept


def _stats(vals: List[float]) -> Dict[str, float]:
    import numpy as np
    a = np.asarray(vals, dtype="float64")
    return {"median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75))}


def mine(paths: List[str]) -> Dict[str, Any]:
    buckets: Dict[str, Dict[str, Any]] = {}
    geom_acc: Dict[str, collections.defaultdict] = {}
    param_acc: Dict[str, collections.defaultdict] = {}
    n_masks = n_used = n_ai = 0
    mtype = collections.Counter()
    ai_usage: collections.Counter = collections.Counter()

    for p in paths:
        for rec in parse_local_masks(p):
            n_masks += 1
            mtype[rec["mask_type"] or "(none)"] += 1
            if rec["is_ai"]:
                n_ai += 1
                ai_usage[(rec["what"] or "ai").rsplit("/", 1)[-1].lower()] += 1
            if not rec["local_params"]:
                continue  # inert mask, no edit -> not a template
            ng = _norm_geom(rec)
            if ng is None:
                continue
            n_used += 1
            bid, concept = _bucket_id(ng, rec)
            b = buckets.setdefault(bid, {"id": bid, "shape": ng["shape"], "n": 0,
                                         "anchor_concept": concept,
                                         "mask_type": rec["mask_type"]})
            b["n"] += 1
            ga = geom_acc.setdefault(bid, collections.defaultdict(list))
            for k, v in ng.items():
                if isinstance(v, (int, float)):
                    ga[k].append(float(v))
                elif isinstance(v, list):
                    for i, vi in enumerate(v):
                        ga[f"{k}{i}"].append(float(vi))
            pa = param_acc.setdefault(bid, collections.defaultdict(list))
            for k, v in rec["local_params"].items():
                pa[k].append(float(v))

    templates = []
    for bid, b in buckets.items():
        ga, pa = geom_acc[bid], param_acc[bid]
        geom_med = {k: _stats(v)["median"] for k, v in ga.items()}
        geom_iqr = {k: [_stats(v)["q1"], _stats(v)["q3"]] for k, v in ga.items()}
        lp_med = {k: _stats(v)["median"] for k, v in pa.items()}
        lp_iqr = {k: [_stats(v)["q1"], _stats(v)["q3"]] for k, v in pa.items()}
        aspect_mag: Dict[str, float] = collections.defaultdict(float)
        for k, m in lp_med.items():
            aspect_mag[_ASPECT_OF.get(k, "L")] += abs(m) / _PARAM_SCALE.get(k, 1.0)
        dom = max(aspect_mag, key=lambda k: aspect_mag[k]) if aspect_mag else "L"
        templates.append({
            "id": bid, "shape": b["shape"], "mask_type": b["mask_type"],
            "n": b["n"], "weight": round(b["n"] / max(1, n_used), 4),
            "anchor_concept": b["anchor_concept"], "aspect_dominant": dom,
            "geom_median": geom_med, "geom_iqr": geom_iqr,
            "local_params_median": lp_med, "local_params_iqr": lp_iqr,
        })
    templates.sort(key=lambda t: -t["n"])
    return {
        "version": "v1", "n_presets": len(paths),
        "n_masks": n_masks, "n_template_masks": n_used, "n_ai_masks": n_ai,
        "mask_type_dist": dict(mtype), "ai_mask_usage": dict(ai_usage),
        "templates": templates,
    }


def _scan(recipe_root: str) -> List[str]:
    out = []
    for dp, _, fns in os.walk(recipe_root):
        for fn in fns:
            if fn.lower().endswith(".xmp"):
                out.append(os.path.join(dp, fn))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe-root", default="/home/bc/data/datasets/recipes")
    ap.add_argument("--out", default="/home/bc/data/datasets/vera_directionA_1M/"
                                     "mask_templates/template_bank.json")
    ap.add_argument("--report", action="store_true", help="print analysis to stdout")
    args = ap.parse_args()

    paths = _scan(args.recipe_root)
    bank = mine(paths)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(bank, f, indent=2, ensure_ascii=False)

    print(f"scanned {len(paths)} xmp -> {bank['n_masks']} masks "
          f"({bank['n_template_masks']} usable, {bank['n_ai_masks']} AI) "
          f"-> {len(bank['templates'])} templates -> {args.out}")
    if args.report:
        print("mask_type_dist:", bank["mask_type_dist"])
        print("ai_mask_usage:", bank["ai_mask_usage"])
        print("top templates (id | n | anchor | aspect | key edit):")
        for t in bank["templates"][:15]:
            top_edit = sorted(t["local_params_median"].items(),
                              key=lambda kv: -abs(kv[1]))[:2]
            edit = ", ".join(f"{k}={v:+.2f}" for k, v in top_edit)
            print(f"  {t['id']:16s} n={t['n']:4d} {str(t['anchor_concept']):9s} "
                  f"{t['aspect_dominant']:3s} | {edit}")


if __name__ == "__main__":
    main()
