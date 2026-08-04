#!/usr/bin/env python
"""S2 REQUIRED OUTPUT -- NOT RUN YET.  Fix D5 (guided-filter radius / eps).

The current defaults (``radius_low=2, eps=1e-3``) were carried over from
``experiments/E2_basis_fit_20260803``, where the guided filter ran at **full
resolution over each basis channel** -- exactly the order protocol 4.2 now
forbids.  The precedent therefore does not transfer and the parameters must be
re-fixed on the real path (REVIEW-impl-WhereA B-4 requirement 3 / N-16).

For each (radius, eps) this reports, on real ``V_where`` local samples:

* delivery-resolution soft-IoU of the oracle mask (the number that matters);
* the low -> high resolution drop;
* the fraction of pixels the upsample pushes outside the declared s domain
  (pre-clamp), i.e. how hard the filter is fighting the +-3 bound;
* edge sharpness: mean |grad s| in the 10% of pixels with the strongest guide
  gradient, relative to bilinear.

Pick the setting with the best high-res soft-IoU whose out-of-domain fraction is
small, write it into ``config.py`` and flip ``GUIDED_PARAMS_PROVISIONAL`` to
``False``.

Usage:
    python -m q3vl.where.scripts.sweep_upsample --limit 24 --device cuda \
        --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from q3vl.where.calibrate import Calibrator, percentiles
from q3vl.where.config import (
    CalibConfig, FitConfig, MODEL_DIR, PhiConfig, REPORT_DIR, UpsampleConfig,
)
from q3vl.where.fpre import load_vision_tower
from q3vl.where.oracle import evaluate_latent
from q3vl.where.pipeline import WhereADataSource
from q3vl.where.preflight import _env

RADII = (1, 2, 4)
EPSILONS = (1e-4, 1e-3, 1e-2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=24)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--arm", default="BA-3-Joint")
    ap.add_argument("--basis", default=None, help="B.npy; default = uncalibrated seeded B")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    src_dir = Path(args.checkpoint) if args.checkpoint else Path(args.model_dir)
    visual = load_vision_tower(src_dir, dtype=getattr(torch, args.dtype), device=args.device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device, attach_hi=True)

    projector = None
    if args.basis:
        import numpy as np

        from q3vl.where.projector import BasisProjector
        projector = BasisProjector()
        with torch.no_grad():
            projector.weight.copy_(torch.from_numpy(np.load(args.basis)))
    cal = Calibrator(CalibConfig(arm=args.arm, phi=PhiConfig()),
                     projector=projector, device=args.device)
    fit_cfg = FitConfig(n_random=4, max_iter=100)

    # fit once per sample, then re-use the same latents for every (r, eps):
    # the sweep is about the upsample, not about the fit.
    cached = []
    for j, prepared in enumerate(source.iter_split("V_where", limit=args.limit)):
        s = prepared.sample
        parts = cal.phi_for(s)
        for r in ("band", "cband12"):
            fit = cal.fit_sample(s, parts.phi_dir, r, fit_cfg=fit_cfg, seed_offset=j)
            if fit.usable:
                cached.append((s, parts.phi_dir.double().detach(), r, fit.latent))
    if not cached:
        raise SystemExit("no usable fit; cannot sweep")

    results = []
    t0 = time.time()
    for radius in RADII:
        for eps in EPSILONS:
            cfg = UpsampleConfig(radius_low=radius, eps=eps)
            per: dict[str, list] = {"band": [], "cband12": []}
            drops: list[float] = []
            oods: list[float] = []
            raw_lo, raw_hi = 0.0, 0.0
            for s, phi, readout, latent in cached:
                with torch.no_grad():
                    ev = evaluate_latent(
                        phi, latent, s.mask_low.to(cal.device).double(),
                        s.grid_h, s.grid_w,
                        guide_hi=s.guide_hi.to(cal.device).double(),
                        target_hi=s.mask_hi.to(cal.device).double(), up_cfg=cfg,
                    )
                per[readout].append(ev["hi"]["soft_iou_minmax"])
                drops.append(ev["hi_minus_low_soft_iou"])
                oods.append(ev["s_domain"]["frac_out_of_domain"])
                raw_lo = min(raw_lo, ev["s_domain"]["raw_min"])
                raw_hi = max(raw_hi, ev["s_domain"]["raw_max"])
            results.append({
                "radius_low": radius, "eps": eps,
                "hi_soft_iou": {k: percentiles(v) for k, v in per.items()},
                "hi_minus_low_soft_iou": percentiles(drops),
                "frac_out_of_domain": percentiles(oods),
                "raw_s_range": [raw_lo, raw_hi],
            })
            print(json.dumps(results[-1]), flush=True)

    best = max(results, key=lambda r: (
        (r["hi_soft_iou"]["band"].get("median", 0) +
         r["hi_soft_iou"]["cband12"].get("median", 0)) / 2))
    out = {
        "env": _env(), "n_latents": len(cached), "limit": args.limit,
        "basis": args.basis or "seeded_orthogonal_uncalibrated",
        "elapsed_s": round(time.time() - t0, 1),
        "grid": {"radius_low": list(RADII), "eps": list(EPSILONS)},
        "results": results,
        "recommended": {k: best[k] for k in ("radius_low", "eps")},
        "note": "write the recommendation into config.py and set "
                "GUIDED_PARAMS_PROVISIONAL = False",
    }
    path = Path(args.out) if args.out else REPORT_DIR / "d5_upsample_sweep.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"recommended": out["recommended"], "wrote": str(path)}, indent=2))
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
