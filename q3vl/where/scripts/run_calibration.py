#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  One Where-A basis-calibration arm end to end.

Protocol 4.4 + 10.2:

    for each eligible local *train* sample (1 epoch):
        F_pre  <- frozen SFT vision tower (bf16 forward)
        phi    <- B(F_pre) residualised + standardised, with geo5 / L / S
        latent <- multi-start L-BFGS in float64, B held fixed   (inner)
        B      <- one AdamW step through phi with the latent fixed   (outer)
    then: freeze B, refit Band and CBand latents on V_where, publish
          the oracle latents and the basis.

`BA-0-Fixed` skips the outer step entirely; it exists to answer "how much of the
ceiling is already there before any calibration?".

Requires one GPU.  **Do not start while Base SFT holds both cards.**

Usage (D-20: rm -f the log first, then verify with `ps -p <PID>`, never pgrep):
    rm -f /home/bc/data/runs/where_a/BA-3-Joint/train.log
    nohup python -m q3vl.where.scripts.run_calibration \
        --arm BA-3-Joint --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976 \
        > /home/bc/data/runs/where_a/BA-3-Joint/train.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from q3vl.where.calibrate import Calibrator, percentiles
from q3vl.where.config import (
    ARMS, BASIS_DIR, CalibConfig, FitConfig, MODEL_DIR, ORACLE_DIR, PhiConfig,
    REPORT_DIR, UpsampleConfig,
)
from q3vl.where.fpre import fpre_facts, load_vision_tower
from q3vl.where.packing import pack_oracle, verify_published, write_basis
from q3vl.where.pipeline import WhereADataSource
from q3vl.where.preflight import _env


def _batched(it, n):
    batch = []
    for x in it:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--checkpoint", default=None,
                    help="Base SFT checkpoint; defaults to the base model")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--train-limit", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--out-root", default="/home/bc/data/runs/where_a")
    ap.add_argument("--exclude-low", action="store_true",
                    help="drop winner_confidence=low from the TRAINING population "
                         "(D1 default is to keep it; the headline stays normal-only)")
    ap.add_argument("--maskview-root", default=None,
                    help="published mask views (scripts/extract_maskviews.py); "
                         "avoids re-decoding every .cgt.png per arm")
    ap.add_argument("--sample-every", type=int, default=200,
                    help="also record one accepted fit every N steps")
    ap.add_argument("--count-cache", default=None,
                    help="JSON produced by a previous eligible-count pass")
    args = ap.parse_args()

    run_dir = Path(args.out_root) / args.arm
    run_dir.mkdir(parents=True, exist_ok=True)
    src_dir = Path(args.checkpoint) if args.checkpoint else Path(args.model_dir)

    visual = load_vision_tower(src_dir, dtype=getattr(torch, args.dtype), device=args.device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device,
                              exclude_low=args.exclude_low,
                              maskview_root=args.maskview_root)

    cfg = CalibConfig(arm=args.arm, batch_size=args.batch_size,
                      phi=PhiConfig(), upsample=UpsampleConfig())

    # --- LR schedule: built from the ACTUAL number of steps -----------------
    # REVIEW-impl-WhereA B-3: counting `build in l1..l6` without running
    # eligibility() stretched warmup to 5.3% and stopped cosine at 42% of peak,
    # so the arms would have been compared at different points on the schedule.
    count_path = Path(args.count_cache) if args.count_cache else run_dir / "eligible_count.json"
    if count_path.exists():
        counts = json.loads(count_path.read_text())
    else:
        print("counting eligible train samples (records only, no GPU)...", flush=True)
        counts = source.count_eligible("train")
        count_path.parent.mkdir(parents=True, exist_ok=True)
        count_path.write_text(json.dumps(counts, indent=2))
    n_eligible = counts["n_eligible"]
    n_planned = min(n_eligible, args.train_limit) if args.train_limit else n_eligible
    total_steps = max(1, math.ceil(n_planned / args.batch_size))
    cal = Calibrator(cfg, total_steps=total_steps, device=args.device)

    setup = {
        "arm": args.arm, "env": _env(), "fpre": fpre_facts(visual),
        "weights": str(src_dir),
        "eligible_count": counts,
        "n_planned_samples": n_planned,
        "total_steps": total_steps,
        "warmup_steps": cal.warmup_steps,
        "maskview_root": args.maskview_root,
        "config": {
            "projector_lr": cfg.projector_lr, "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio, "scheduler": cfg.scheduler,
            "epochs": cfg.epochs, "objective": cfg.objective,
            "batch_size": cfg.batch_size, "seed": cfg.seed,
            "inner_fit": cfg.inner_fit.__dict__, "phi": cfg.phi.__dict__,
            "upsample": cfg.upsample.__dict__,
        },
        "projector_init": cal.projector.facts(),
    }
    (run_dir / "run_setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps(setup, indent=2), flush=True)

    # --- 1 epoch over the eligible local train samples ---------------------
    t0 = time.time()
    log_path = run_dir / "steps.jsonl"
    rej_path = run_dir / "fit_rejections.jsonl"
    n_seen = 0
    rej_before = len(source.rejections)
    with log_path.open("a") as log, rej_path.open("a") as rej:
        stream = (p.sample for p in source.iter_split("train", limit=args.train_limit))
        for batch in _batched(stream, args.batch_size):
            out = cal.step(batch, sample_every=args.sample_every)
            n_seen += len(batch)
            # B-2: every non-ok fit lands on disk, sample-level, as it happens
            for row in out.pop("fit_rows"):
                rej.write(json.dumps(row) + "\n")
            out["elapsed_s"] = round(time.time() - t0, 1)
            log.write(json.dumps(out) + "\n")
            if cal.step_count % args.log_every == 0:
                log.flush()
                rej.flush()
                print(json.dumps(out), flush=True)
            if cal.step_count % args.save_every == 0:
                torch.save(cal.state(), run_dir / f"projector_step{cal.step_count}.pt")
    torch.save(cal.state(), run_dir / "projector_final.pt")

    # B-3: the schedule must have actually finished.  N-19: a gap that the
    # data source can explain sample-by-sample (a mask that would not read, an
    # aspect mismatch) is a warning with the evidence written down -- refusing to
    # continue *after* a whole GPU epoch would throw away the epoch as well.
    # A gap nothing accounts for is still fatal.
    late_drops = source.rejections[rej_before:]
    gap = n_planned - n_seen
    explained = len(late_drops)
    schedule = {
        "planned_total_steps": total_steps, "actual_steps": cal.step_count,
        "warmup_steps": cal.warmup_steps, "n_planned_samples": n_planned,
        "n_seen_samples": n_seen, "sample_gap": gap,
        "late_drops": explained,
        "late_drop_reasons": {r: sum(1 for d in late_drops if d["reason"] == r)
                              for r in {d["reason"] for d in late_drops}},
        "late_drop_examples": late_drops[:20],
        "gap_explained": gap == explained,
        "final_lr": cal.optimizer.param_groups[0]["lr"] if cal.optimizer else 0.0,
        "rejections": cal.rejection_summary(),
    }
    (run_dir / "schedule.json").write_text(json.dumps(schedule, indent=2))
    print(json.dumps({k: v for k, v in schedule.items()
                      if k != "late_drop_examples"}, indent=2), flush=True)
    if gap != explained:
        raise SystemExit(
            f"schedule mismatch: planned {total_steps} steps from {n_planned} "
            f"eligible samples but ran {cal.step_count} over {n_seen} samples; "
            f"{gap} missing and only {explained} explained by source.rejections. "
            f"The cosine never finished and the shortfall is unaccounted for. "
            f"Delete {count_path} and re-count."
        )
    if gap:
        print(f"WARNING: {gap} sample(s) dropped after the eligibility count "
              f"({schedule['late_drop_reasons']}); the schedule ran "
              f"{cal.step_count}/{total_steps} steps, final lr "
              f"{schedule['final_lr']:.3e}. See schedule.json.", flush=True)

    # --- refit with B frozen, on the selection set (protocol 4.4) ----------
    # attach_hi: the ceiling that goes in the report is measured at delivery
    # resolution, through the one guided upsample (B-4).
    source.attach_hi = True
    final_fit = FitConfig(n_random=6, max_iter=120)
    report = cal.evaluate(
        (p.sample for p in source.iter_split("V_where", limit=args.eval_limit)),
        readouts=("band", "cband12"), fit_cfg=final_fit, record_fits=True,
    )
    rows = report.pop("rows")

    def by(key: str) -> dict[str, Any]:
        groups: dict[str, list] = {}
        for r in rows:
            if r.get("status") != "ok" or not r.get("eval"):
                continue
            groups.setdefault(str(r["meta"].get(key)), []).append(r)
        out: dict[str, Any] = {}
        for k, v in groups.items():
            lo = percentiles([x["eval"]["low"]["soft_iou_minmax"] for x in v])
            hi_vals = [x["eval"]["hi"]["soft_iou_minmax"] for x in v if x["eval"].get("hi")]
            out[k] = {"n": len(v), "low": lo, "hi": percentiles(hi_vals)}
        return out

    # `upscaled` matters because ~15% of local samples were upsampled to reach
    # short side 512: their GT edges are interpolated, so edge metrics read high.
    report["strata"] = {"upscaled": by("upscaled"),
                        "winner_confidence": by("winner_confidence"),
                        "build": by("build")}
    report["elapsed_s"] = round(time.time() - t0, 1)
    (run_dir / "eval_V_where.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "strata"}, indent=2), flush=True)

    # --- publish oracle latents + the calibrated basis ---------------------
    fits: dict[str, dict] = {}
    metas: dict[str, dict] = {}
    for r in rows:
        fits.setdefault(r["sample_id"], {})[r["readout"]] = {
            k: v for k, v in r.items() if k not in ("meta", "phi_diag", "sample_id")
        }
        metas[r["sample_id"]] = r["meta"]
    oracle_root = ORACLE_DIR / args.arm / "V_where"
    # N-12: publication is atomic; an existing root is an operator decision, not
    # something to silently skip (CLAUDE.md: back up before re-running).
    if oracle_root.exists():
        raise SystemExit(
            f"{oracle_root} already exists. Move it aside (do not delete: "
            f"'back up before clearing' is a standing rule) and re-run."
        )
    pack_oracle(((sid, fits[sid], metas[sid]) for sid in sorted(fits)), oracle_root,
                source_label=f"where_a.oracle/{args.arm}/V_where")
    print(json.dumps(verify_published(oracle_root, n_random=32), indent=2), flush=True)

    basis_meta = write_basis(BASIS_DIR / args.arm, args.arm, cal.projector, {
        "setup": setup, "eval_summary": report["per_readout"],
        "oracle_shards": str(oracle_root),
    })
    (REPORT_DIR / f"calibration_{args.arm}.json").write_text(json.dumps({
        "setup": setup, "eval": report, "basis": basis_meta,
    }, indent=2))
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
