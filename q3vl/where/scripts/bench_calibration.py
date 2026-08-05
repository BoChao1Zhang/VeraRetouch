#!/usr/bin/env python
"""Measure the real calibration step: GPU F_pre -> CPU pool fits -> B backward.

The S4 schedule rests on one number -- seconds per step -- and it cannot be
inferred from the per-fit timing alone, because the GPU vision forward and the
CPU fit pool are different resources with different scaling.  So this runs the
actual `Calibrator.step` loop against the real checkpoint and the real masks and
reports the breakdown.

Usage:
    python -m q3vl.where.scripts.bench_calibration \\
        --steps 20 --batch-size 32 --workers 38 \\
        --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976
"""

from __future__ import annotations

# The BLAS thread pin has to happen before numpy/torch load, otherwise the
# *_NUM_THREADS variables are read too late (fitpool falls back to a runtime
# ctypes call, but doing it here is the clean path).
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from q3vl.where.calibrate import Calibrator, percentiles  # noqa: E402
from q3vl.where.config import (  # noqa: E402
    CalibConfig, MODEL_DIR, PhiConfig, REPORT_DIR, UpsampleConfig,
)
from q3vl.where.fitpool import FitPool, pin_single_thread  # noqa: E402
from q3vl.where.fpre import load_vision_tower  # noqa: E402
from q3vl.where.pipeline import WhereADataSource, prefetch  # noqa: E402
from q3vl.where.preflight import _env  # noqa: E402


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
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=38)
    ap.add_argument("--arm", default="BA-3-Joint")
    ap.add_argument("--split", default="train")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--maskview-root", default=None)
    ap.add_argument("--eligible", type=int, default=42752,
                    help="eligible sample count used for the wall-clock projection")
    ap.add_argument("--warmup-steps", type=int, default=2,
                    help="steps excluded from the timing (pool start, CUDA autotune)")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="overlap the data path with the step (batches ahead)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    pinned = pin_single_thread()
    src_dir = Path(args.checkpoint) if args.checkpoint else Path(args.model_dir)
    visual = load_vision_tower(src_dir, dtype=getattr(torch, args.dtype), device=args.device)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device,
                              maskview_root=args.maskview_root)

    cfg = CalibConfig(arm=args.arm, batch_size=args.batch_size,
                      phi=PhiConfig(), upsample=UpsampleConfig())
    total_steps = max(1, math.ceil(args.eligible / args.batch_size))
    cal = Calibrator(cfg, total_steps=total_steps, device=args.device)

    pool = FitPool(n_workers=args.workers)
    pool.warmup()

    rows: list[dict] = []
    t_data_total = 0.0
    t0_all = time.perf_counter()
    batches = _batched(source.iter_split(args.split), args.batch_size)
    if args.prefetch:
        batches = prefetch(batches, depth=args.prefetch)
    n_done = 0
    checked = None
    while n_done < args.steps:
        # the data cost lives in pulling from the generator (shard read, JPEG
        # decode, mask decode, spec-5 resize, vision forward), so the clock has
        # to wrap `next()` -- timing only the list comprehension after the pull
        # reports 0.0 and hides the real bottleneck.
        t_data = time.perf_counter()
        try:
            prepared = next(batches)
        except StopIteration:
            break
        batch = [p.sample for p in prepared]
        t_data = time.perf_counter() - t_data

        if checked is None:
            # one real fit, worker vs parent, before anything is timed: a broken
            # worker fails every start and reports it as an ordinary rejection
            checked = pool.self_check(cal.fit_tasks(batch[:1], cal.readouts, step=0)[0])
            print(json.dumps({"pool_self_check": checked}), flush=True)

        t_step = time.perf_counter()
        out = cal.step(batch, pool=pool)
        t_step = time.perf_counter() - t_step
        out.pop("fit_rows", None)
        out["t_step_s"] = round(t_step, 3)
        out["t_data_s"] = round(t_data, 3)
        rows.append(out)
        t_data_total += t_data
        n_done += 1
        print(json.dumps({k: out[k] for k in
                          ("step", "loss", "t_step_s", "t_data_s", "timing_s",
                           "n_rejected_fits")}), flush=True)

    elapsed = time.perf_counter() - t0_all
    timed = rows[args.warmup_steps:] or rows
    step_s = [r["t_step_s"] for r in timed]
    parts = {k: percentiles([r["timing_s"][k] for r in timed])
             for k in ("phi_gpu", "fit", "outer", "backward")}
    med = percentiles(step_s)["median"]
    data_med = percentiles([r["t_data_s"] for r in timed])["median"]
    per_step_total = med + data_med
    # sanity: the two measured phases must account for the wall clock, or
    # something untimed is eating the run
    wall_per_step = elapsed / max(1, n_done)

    report = {
        "env": _env(),
        "arm": args.arm, "split": args.split,
        "weights": str(src_dir),
        "batch_size": args.batch_size,
        "n_workers": args.workers,
        "prefetch_depth": args.prefetch,
        "openblas_pinned": pinned,
        "pool": pool.facts(),
        "pool_self_check": checked,
        "n_steps_run": n_done,
        "n_steps_timed": len(timed),
        "warmup_steps_excluded": args.warmup_steps,
        "elapsed_s": round(elapsed, 1),
        "fits_per_step": args.batch_size * len(cal.readouts),
        "step_s": percentiles(step_s),
        "data_s": percentiles([r["t_data_s"] for r in timed]),
        "phase_s": parts,
        "s_per_fit_effective": round(med / (args.batch_size * len(cal.readouts)), 4),
        "wall_s_per_step": round(wall_per_step, 3),
        "unaccounted_s_per_step": round(wall_per_step - per_step_total, 3),
        "bottleneck": "data" if data_med > med else "fit",
        "projection": {
            "eligible_samples": args.eligible,
            "total_steps": total_steps,
            "hours_per_arm_step_only": round(med * total_steps / 3600, 2),
            "hours_per_arm_with_data": round(per_step_total * total_steps / 3600, 2),
            "hours_per_arm_wall": round(wall_per_step * total_steps / 3600, 2),
            "hours_per_arm_if_data_overlapped": round(
                max(med, data_med) * total_steps / 3600, 2),
        },
        "rows": rows,
    }
    out_path = Path(args.out) if args.out else REPORT_DIR / "calibration_throughput.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    print(f"\nwrote {out_path}")
    pool.close()
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
