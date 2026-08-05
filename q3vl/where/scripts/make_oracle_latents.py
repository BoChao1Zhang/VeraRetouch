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

# D12: pin BLAS to one thread before numpy/torch load (see fitpool.py).
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from q3vl.where.basis import Latent
from q3vl.where.calibrate import Calibrator, percentiles
from q3vl.where.config import (
    ARMS, BASIS_DIR, CBAND_NORMALIZATION, CURVE_Z_N, MODEL_DIR, ORACLE_DIR,
    REPORT_DIR, S_DOMAIN, CalibConfig, FitConfig, PhiConfig, UpsampleConfig,
)
from q3vl.where.fitpool import FitPool, pin_single_thread
from q3vl.where.fpre import fpre_facts, load_vision_tower
from q3vl.where.oracle import evaluate_latent
from q3vl.where.packing import pack_oracle, verify_published
from q3vl.where.pipeline import WhereADataSource, prefetch
from q3vl.where.preflight import _env
from q3vl.where.projector import BasisProjector
from q3vl.where.readout import apply_readout

# Protocol 5.5 pins the grid: `z = linspace(-3, 3, 257)`.  Where-B consumes r*(z)
# as a vector, so any other length forces it to interpolate (error injected into
# a supervision target) or to deviate from 5.5 (REVIEW-impl-WhereA B-7).
CURVE_Z = np.linspace(S_DOMAIN[0], S_DOMAIN[1], CURVE_Z_N)


def curve_of(latent: Latent) -> list[float]:
    """r*(z) under the *declared* CBand normalisation.

    The convention is published alongside the samples: Where-B must recompute
    ``R(z; rho_pred)`` the same way, or the two sides of ``L_curve`` are not the
    same function (REVIEW-impl-WhereA N-25 / D10).

    Evaluated on CPU in float64 regardless of where the latent currently lives.
    The fit runs on CPU but the latent is moved to the calibrator's device for
    the evaluation, so a curve built from a bare ``torch.tensor(CURVE_Z)`` mixed
    devices and raised -- caught on the first S5 split, 20 s in.  This is a
    serialisation artifact: CPU float64 is its canonical form.
    """
    z = torch.tensor(CURVE_Z, dtype=torch.float64)
    rho = {k: v.detach().double().cpu() for k, v in latent.rho.items()}
    with torch.no_grad():
        return apply_readout(latent.readout, z, rho).tolist()


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
    ap.add_argument("--workers", type=int, default=32,
                    help="CPU fit-pool workers (0 = serial); D12")
    ap.add_argument("--chunk", type=int, default=32,
                    help="samples per pool dispatch (tasks are per-sample)")
    ap.add_argument("--prefetch", type=int, default=64,
                    help="samples prepared ahead of the fits (0 = off)")
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
    pin_single_thread()
    pool = FitPool(n_workers=args.workers or None)
    pool.warmup()
    # N-24: "B is never touched" must be enforced, not merely intended.  This
    # job never calls cal.step(), but the Calibrator would happily have built an
    # optimiser for an arm with readouts; freeze it for real and prove it.
    cal.freeze_projector()
    assert cal.optimizer is None and not cal.trains_projector
    assert not any(p.requires_grad for p in cal.projector.parameters())
    b_digest_before = cal.projector.digest()
    fit_cfg = FitConfig(n_random=6, max_iter=120)      # full multi-start, protocol 4.4

    stats = {r: [] for r in ("band", "cband12")}
    rejections: list[dict] = []
    n_done = 0
    t0 = time.time()

    def rows():
        """Chunked, pool-backed.  The first S5 attempt built the pool and then
        fitted through `cal.fit_sample`, i.e. one fit at a time in-process: at
        ~2.5 s per full-config fit that is ~100 h for train's 151,088 fits.  The
        fits go through `run_fits` like the calibration loop's do; seed_offset is
        still the global sample index, so the numbers are unchanged."""
        nonlocal n_done
        stream = source.iter_split(args.split, limit=args.limit)
        if args.prefetch:
            stream = prefetch(stream, depth=args.prefetch)
        j = -1
        for group in cal._chunks((p.sample for p in stream), args.chunk):
            parts_by_id = {}
            for sample in group:
                with torch.no_grad():       # no graph: B is frozen (N-24)
                    parts_by_id[sample.sample_id] = cal.phi_for(sample)
            cal._phi_cache = parts_by_id
            base = j + 1
            tasks = []
            for k, sample in enumerate(group):
                tasks.extend(cal.fit_tasks([sample], ("band", "cband12"),
                                           fit_cfg=fit_cfg, step=0))
                tasks[-1].seed_offset = base + k
            group_fits = cal.run_fits(tasks, pool)
            cal._phi_cache = {}

            for sample in group:
                j += 1
                parts = parts_by_id[sample.sample_id]
                fits: dict[str, dict] = {}
                for r in ("band", "cband12"):
                    fit = group_fits[sample.sample_id][r]
                    if not fit.usable:
                        rejections.append(fit.rejection_row(
                            sample_id=sample.sample_id, build=sample.meta.get("build"),
                            winner_confidence=sample.meta.get("winner_confidence")))
                        continue
                    if fit.latent is not None:
                        fit.latent = fit.latent.to(cal.device)
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
                    # r*(z) on protocol 5.5's own grid, plus the convention it was
                    # sampled under -- Where-B must use the same one (N-25)
                    d["curve_z"] = CURVE_Z.tolist()
                    d["curve"] = curve_of(fit.latent)
                    d["cband_normalization"] = CBAND_NORMALIZATION
                    fits[r] = d
                if not fits:
                    continue
                n_done += 1
                if n_done % args.log_every == 0:
                    rate = (time.time() - t0) / max(1, n_done)
                    print(json.dumps({
                        "n": n_done, "elapsed_s": round(time.time() - t0, 1),
                        "s_per_sample": round(rate, 3),
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
        # the two conventions Where-B must match to make L_curve well defined
        "curve_z_grid": {"lo": S_DOMAIN[0], "hi": S_DOMAIN[1], "n": len(CURVE_Z),
                         "source": "protocol 5.5: z = linspace(-3, 3, 257)"},
        "cband_normalization": CBAND_NORMALIZATION,
        "b_digest": b_digest_before,
        "b_unchanged": cal.projector.digest() == b_digest_before,
    }
    if not report["b_unchanged"]:
        raise SystemExit("B changed during latent generation; it must be frozen")
    rep_dir = REPORT_DIR / "oracle_latents"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"{args.arm}_{args.split}.report.json").write_text(json.dumps(report, indent=2))
    with (rep_dir / f"{args.arm}_{args.split}.rejections.jsonl").open("w") as fh:
        for r in rejections:
            fh.write(json.dumps(r) + "\n")
    report["fit_pool"] = pool.facts()
    print(json.dumps(report, indent=2), flush=True)
    pool.close()
    source.close()
    return 0 if report["verify"]["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
