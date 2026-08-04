#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  Per-image oracle latents on the TRAIN split.

Protocol 4.4: "the per-image oracle parameters serve only as **supervision** and
as a ceiling"; protocol 5.5 then spends them::

    L_s     = Huber(s_pred / 3, s* / 3)
    L_curve = mean_z |R(z; rho_pred) - r*(z)|
    L_dir   = 1 - cos(w_dir_pred, w_dir*)

all three weighted 1.00 for the first 30% of Where-B's steps.  They need
``s*, r*(z), w_dir*`` on the **training** samples, which the calibration run only
ever produced for ``V_where`` -- Where-B would have stalled on day one and the
most expensive step in Where-A (the full inner L-BFGS pass) would have had to be
repeated at the worst possible moment (REVIEW-impl-WhereA B-5).

Position in the pipeline: **after** S4 freezes ``BA-3-Joint``'s ``B``, **before**
Where-B training. ``B`` is loaded from the published basis and never touched.

Output (protocol 2.3 indexed tar shards), per sample:
    ``<sample_id>.oracle.json``  both readouts' canonical ``w*, rho*``, the fit
    report, the low- and high-resolution metrics, and ``r*(z)`` sampled on a
    fixed z grid so ``L_curve`` needs no readout code at Where-B training time.

Usage:
    python -m q3vl.where.scripts.make_oracle_latents \
        --arm BA-3-Joint --split train \
        --basis /mnt/nfs/bc/data/datasets/where_a-20260805/basis/BA-3-Joint/B.npy \
        --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from q3vl.where.basis import Latent
from q3vl.where.calibrate import Calibrator, percentiles
from q3vl.where.config import (
    ARMS, BASIS_DIR, CBAND_MU_HI, CBAND_MU_LO, MODEL_DIR, ORACLE_DIR, REPORT_DIR,
    CalibConfig, FitConfig, PhiConfig, UpsampleConfig,
)
from q3vl.where.fpre import fpre_facts, load_vision_tower
from q3vl.where.oracle import evaluate_latent
from q3vl.where.packing import pack_oracle, verify_published
from q3vl.where.pipeline import WhereADataSource
from q3vl.where.preflight import _env
from q3vl.where.projector import BasisProjector
from q3vl.where.readout import apply_readout

# r*(z) is sampled on this fixed grid so Where-B's L_curve is a plain vector
# difference; the grid spans the full s domain including both end centres.
CURVE_Z = np.linspace(CBAND_MU_LO, CBAND_MU_HI, 121)


def curve_of(latent: Latent) -> list[float]:
    z = torch.tensor(CURVE_Z, dtype=torch.float64)
    with torch.no_grad():
        return apply_readout(latent.readout, z, latent.rho).tolist()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="BA-3-Joint", choices=list(ARMS))
    ap.add_argument("--split", default="train")
    ap.add_argument("--basis", default=None, help="B.npy; defaults to BASIS_DIR/<arm>/B.npy")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--maskview-root", default=None)
    ap.add_argument("--exclude-low", action="store_true")
    ap.add_argument("--attach-hi", action="store_true",
                    help="also record delivery-resolution metrics (slower)")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args()

    out_root = Path(args.out_root) if args.out_root else ORACLE_DIR / args.arm / args.split
    if out_root.exists():
        raise SystemExit(f"{out_root} already exists; move it aside (do not delete) and re-run")

    basis_path = Path(args.basis) if args.basis else BASIS_DIR / args.arm / "B.npy"
    if not basis_path.exists():
        raise SystemExit(
            f"{basis_path} not found. This job runs AFTER the arm's calibration has "
            f"frozen B (S4); it must never fit latents against an uncalibrated basis."
        )
    weights = np.load(basis_path)
    projector = BasisProjector()
    with torch.no_grad():
        projector.weight.copy_(torch.from_numpy(weights))
    projector.requires_grad_(False)

    src_dir = Path(args.checkpoint) if args.checkpoint else Path(args.model_dir)
    visual = load_vision_tower(src_dir, dtype=getattr(torch, args.dtype), device=args.device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device,
                              exclude_low=args.exclude_low,
                              maskview_root=args.maskview_root,
                              attach_hi=args.attach_hi)

    cfg = CalibConfig(arm=args.arm, phi=PhiConfig(), upsample=UpsampleConfig())
    cal = Calibrator(cfg, projector=projector, device=args.device)
    fit_cfg = FitConfig(n_random=6, max_iter=120)      # full multi-start, protocol 4.4

    stats = {r: [] for r in ("band", "cband12")}
    rejections: list[dict] = []
    n_done = 0
    t0 = time.time()

    def rows():
        nonlocal n_done
        for j, prepared in enumerate(source.iter_split(args.split, limit=args.limit)):
            sample = prepared.sample
            parts = cal.phi_for(sample)
            fits: dict[str, dict] = {}
            for r in ("band", "cband12"):
                fit = cal.fit_sample(sample, parts.phi_dir, r, fit_cfg=fit_cfg, seed_offset=j)
                if not fit.usable:
                    rejections.append(fit.rejection_row(
                        sample_id=sample.sample_id, build=sample.meta.get("build"),
                        winner_confidence=sample.meta.get("winner_confidence")))
                    continue
                with torch.no_grad():
                    ev = evaluate_latent(
                        parts.phi_dir.double(), fit.latent,
                        sample.mask_low.to(cal.device).double(),
                        sample.grid_h, sample.grid_w,
                        guide_hi=(sample.guide_hi.to(cal.device).double()
                                  if sample.has_hi else None),
                        target_hi=(sample.mask_hi.to(cal.device).double()
                                   if sample.has_hi else None),
                        up_cfg=cfg.upsample,
                    )
                stats[r].append(ev["low"]["soft_iou_minmax"])
                d = fit.to_dict()
                d["eval"] = ev
                # r*(z) on the fixed grid: protocol 5.5's L_curve reads this
                d["curve_z"] = CURVE_Z.tolist()
                d["curve"] = curve_of(fit.latent)
                fits[r] = d
            if not fits:
                continue
            n_done += 1
            if n_done % args.log_every == 0:
                print(json.dumps({"n": n_done, "elapsed_s": round(time.time() - t0, 1),
                                  "n_rejected": len(rejections)}), flush=True)
            yield sample.sample_id, fits, sample.meta

    manifest = pack_oracle(rows(), out_root,
                           source_label=f"where_a.oracle/{args.arm}/{args.split}")
    report = {
        "arm": args.arm, "split": args.split, "basis": str(basis_path),
        "basis_sha256_of_npy": __import__("hashlib").sha256(basis_path.read_bytes()).hexdigest(),
        "env": _env(), "fpre": fpre_facts(visual),
        "n_samples": n_done, "n_rejected_fits": len(rejections),
        "elapsed_s": round(time.time() - t0, 1),
        "soft_iou_low": {r: percentiles(v) for r, v in stats.items()},
        "manifest": {k: manifest[k] for k in
                     ("dataset_id", "sample_count", "member_count", "shard_count", "status")},
        "verify": verify_published(out_root, n_random=64),
        "curve_z_grid": {"lo": CBAND_MU_LO, "hi": CBAND_MU_HI, "n": len(CURVE_Z)},
    }
    rep_dir = REPORT_DIR / "oracle_latents"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{args.arm}_{args.split}.report.json").write_text(json.dumps(report, indent=2))
    with (rep_dir / f"{args.arm}_{args.split}.rejections.jsonl").open("w") as fh:
        for r in rejections:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    source.close()
    return 0 if report["verify"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
