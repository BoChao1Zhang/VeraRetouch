"""A0 driver: train B LUTs with the GLUT recipe, eval on full held-out colors.

Usage (from repo root):
  python -m model.glut_repro.run_a0 --lut-list <txt: id<TAB>path per line> \
      --arm rec --outdir experiments/A0_glut_repro_20260803/runs/rec \
      [--epochs 20] [--n 32] [--chunk-size 25] [--smoke]

Writes per-LUT metrics jsonl + checkpoint per chunk + summary metrics.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from model.glut_repro import data  # noqa: E402
from model.glut_repro.train_a0 import (  # noqa: E402
    A0Trainer, de00_worker, psnr_8bit, psnr_float)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-list", required=True,
                    help="txt file: <lut_id>\\t<cube_path> per line")
    ap.add_argument("--arm", choices=["rec", "full"], default="rec")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk-size", type=int, default=25,
                    help="LUTs trained simultaneously per BatchedGLUT")
    ap.add_argument("--gt-workers", type=int, default=8)
    ap.add_argument("--de-workers", type=int, default=12)
    ap.add_argument("--loss-on-clamped", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="eval on the evalsub subsample instead of full")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    luts = []
    with open(args.lut_list) as f:
        for line in f:
            if line.strip():
                lid, path = line.rstrip("\n").split("\t")
                luts.append({"id": lid, "path": path})
    print(f"[a0] {len(luts)} LUTs, arm={args.arm}, N={args.n}", flush=True)

    # 1) GT cache (parallel CPU)
    eval_part = "evalsub" if args.smoke else "eval"
    jobs = [("a0", l["id"], l["path"], ("train", eval_part)) for l in luts]
    t0 = time.time()
    errs = [e for _, e in data.build_gt_parallel(jobs, args.gt_workers) if e]
    if errs:
        raise RuntimeError(f"GT build failures: {errs}")
    print(f"[a0] GT cache ready in {time.time()-t0:.0f}s", flush=True)

    colors_train = data.train_colors()
    colors_eval = data.eval_colors()
    if args.smoke:
        colors_eval = colors_eval[data.evalsub_indices()]

    rows = []
    for c0 in range(0, len(luts), args.chunk_size):
        chunk = luts[c0:c0 + args.chunk_size]
        ids = [l["id"] for l in chunk]
        gts = np.stack([np.load(data._cache_path("a0", lid, "train"))
                        for lid in ids])
        tr = A0Trainer(colors_train, gts, args.n, arm=args.arm,
                       seed=args.seed + c0, epochs=args.epochs, bs=args.bs,
                       lr=args.lr, loss_on_clamped=args.loss_on_clamped)
        t0 = time.time()
        tr.train()
        print(f"[a0] chunk {c0//args.chunk_size}: trained {len(ids)} LUTs "
              f"in {time.time()-t0:.0f}s, final loss "
              f"{tr.log[-1]['loss']:.5f}", flush=True)
        tr.save(os.path.join(args.outdir, f"ckpt_chunk{c0:03d}.pt"), ids,
                {"args": vars(args)})
        # eval
        pred = tr.predict_colors(colors_eval)
        pred_dir = os.path.join(args.outdir, "pred_tmp")
        os.makedirs(pred_dir, exist_ok=True)
        de_jobs = []
        for k, lid in enumerate(ids):
            pp = os.path.join(pred_dir, f"{lid}.npy")
            np.save(pp, pred[k].astype(np.float16))
            de_jobs.append((pp, data._cache_path("a0", lid, eval_part),
                            slice(None)))
        from multiprocessing import Pool
        with Pool(args.de_workers) as pool:
            de_stats = pool.map(de00_worker, de_jobs)
        for k, lid in enumerate(ids):
            gt = np.load(data._cache_path("a0", lid, eval_part)) \
                .astype(np.float32)
            row = {"lut_id": lid, "arm": args.arm, "n": args.n,
                   "psnr_float": psnr_float(pred[k], gt),
                   "psnr_8bit": psnr_8bit(pred[k], gt),
                   "de00": de_stats[k]}
            rows.append(row)
            print(json.dumps(row), flush=True)
        for lid in ids:
            os.remove(os.path.join(pred_dir, f"{lid}.npy"))
        with open(os.path.join(args.outdir, "per_lut.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    psnrs = np.array([r["psnr_float"] for r in rows])
    de_means = np.array([r["de00"]["mean"] for r in rows])
    summary = {
        "exp": "A0_glut_repro", "arm": args.arm, "n_gaussians": args.n,
        "n_luts": len(rows), "eval_part": eval_part,
        "anchor": {"psnr_db": 45.47, "tolerance": 0.3, "de00": 0.41,
                   "note": "GLUT-32@75-LUT, paper's own 64^3 subset "
                           "(files not public; ours is a local 64^3 sample)"},
        "psnr_float_mean": float(psnrs.mean()),
        "psnr_float_std": float(psnrs.std()),
        "psnr_8bit_mean": float(np.mean([r["psnr_8bit"] for r in rows])),
        "de00_mean_over_luts": float(de_means.mean()),
        "de00_p50": float(np.percentile(de_means, 50)),
        "de00_p90": float(np.percentile(de_means, 90)),
        "psnr_p10": float(np.percentile(psnrs, 10)),
        "psnr_min": float(psnrs.min()),
        "anchor_pass": bool(abs(float(psnrs.mean()) - 45.47) <= 0.3),
        "git_commit": git_commit(), "args": vars(args),
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
