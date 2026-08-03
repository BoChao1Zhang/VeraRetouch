"""Hypothesis (a): run the E1 direct-overfit engine on the SAME A0 LUT set.

Same LUTs (a0 list), same GT path (native 64^3 cube -> colour tetrahedral,
`a0` GT cache), same held-out colour split, same N.  The only thing that
changes versus `run_a0 --arm rec` is the OPTIMIZATION RECIPE:

              A0 rec (GLUT paper)         E1 engine
  init        uniform-grid mu, Sigma      |f(x)-x|-weighted k-means mu,
              iso 0.15, M=I, G=I          per-cluster residual b, M=0, G=I
              -> f(x) = 2x @ init         -> f(x) = x  @ init
  sigma       free                        annealed floor 0.30 -> 0.02 -> 0.005
  density     none                        dead-primitive relocation / 500 steps
  batch/lr    1024 / 1e-3, 20 ep (41k st) 8192 / 5e-3 cosine, `--steps` steps

If E1 reaches ~45 dB where A0-rec sits at 41.9, the 22N+12 representation is
NOT the bottleneck and the gap is an optimization-recipe gap.

Eval口径: fixed 2^21 held-out colour subsample (data.evalsub_indices()), the
same one the ablation runner uses, so all three tables are comparable.
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
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from model.glut_repro import data  # noqa: E402
from model.glut_repro.fit_e1 import E1Fitter  # noqa: E402
from model.glut_repro.train_a0 import de00_worker, psnr_8bit, psnr_float  # noqa: E402


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lut-list", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--batch-luts", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--de-workers", type=int, default=8)
    ap.add_argument("--tag", default="e1_on_a0")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    luts = []
    with open(args.lut_list) as f:
        for line in f:
            if line.strip():
                lid, path = line.rstrip("\n").split("\t")
                luts.append({"id": lid, "path": path})
    print(f"[e1a0] {len(luts)} LUTs, N={args.n}, steps={args.steps}", flush=True)

    jobs = [("a0", l["id"], l["path"], ("train", "eval")) for l in luts]
    errs = [e for _, e in data.build_gt_parallel(jobs, 8) if e]
    if errs:
        raise RuntimeError(f"GT build failures: {errs}")

    colors_train = data.train_colors()
    sub = data.evalsub_indices()
    colors_eval = data.eval_colors()[sub]

    gt_dir = os.path.join(args.outdir, "gt_tmp")
    os.makedirs(gt_dir, exist_ok=True)
    rows = []
    for c0 in range(0, len(luts), args.batch_luts):
        chunk = luts[c0:c0 + args.batch_luts]
        ids = [l["id"] for l in chunk]
        gts = np.stack([np.load(data._cache_path("a0", lid, "train"))
                        for lid in ids])
        t0 = time.time()
        fit = E1Fitter(colors_train, gts, args.n, seed=args.seed + c0,
                       steps=args.steps, bs=args.bs, lr=args.lr)
        stats = fit.fit()
        print(f"[e1a0] chunk {c0}: fit {len(ids)} LUTs in {time.time()-t0:.0f}s "
              f"loss {fit.log[-1]['loss']:.5f} alive {stats['alive_frac']}",
              flush=True)
        pred = fit.predict_colors(colors_eval)
        pred_dir = os.path.join(args.outdir, f"pred_tmp_{args.tag}")
        os.makedirs(pred_dir, exist_ok=True)
        de_jobs, gt_np = [], []
        for k, lid in enumerate(ids):
            mm = np.load(data._cache_path("a0", lid, "eval"), mmap_mode="r")
            g = np.asarray(mm[sub])
            gt_np.append(g)
            gp = os.path.join(gt_dir, f"{lid}.gt.npy")
            if not os.path.exists(gp):          # atomic: concurrent runs share gt_tmp
                tmp = f"{gp}.{os.getpid()}.tmp.npy"
                np.save(tmp, g)
                os.replace(tmp, gp)
            pp = os.path.join(pred_dir, f"{lid}.npy")
            np.save(pp, pred[k].astype(np.float16))
            de_jobs.append((pp, gp, slice(None)))
        from multiprocessing import Pool
        with Pool(args.de_workers) as pool:
            de_stats = pool.map(de00_worker, de_jobs)
        for k, lid in enumerate(ids):
            g = gt_np[k].astype(np.float32)
            rows.append({
                "lut_id": lid, "n": args.n, "steps": args.steps,
                "psnr_float": psnr_float(pred[k], g),
                "psnr_8bit": psnr_8bit(pred[k], g),
                "de00_mean": de_stats[k]["mean"],
                "de00_p99": de_stats[k]["p99"],
                "alive_frac": stats["alive_frac"][k],
                "final_loss": fit.log[-1]["loss"]})
            os.remove(os.path.join(pred_dir, f"{lid}.npy"))
            print(json.dumps(rows[-1]), flush=True)
        with open(os.path.join(args.outdir, f"{args.tag}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    psnrs = np.array([r["psnr_float"] for r in rows])
    des = np.array([r["de00_mean"] for r in rows])
    summary = {
        "exp": "A0_gap_hypA_e1_engine", "n_gaussians": args.n,
        "steps": args.steps, "n_luts": len(rows), "eval_part": "evalsub_2^21",
        "psnr_float_mean": float(psnrs.mean()),
        "psnr_float_std": float(psnrs.std()),
        "psnr_8bit_mean": float(np.mean([r["psnr_8bit"] for r in rows])),
        "psnr_min": float(psnrs.min()),
        "psnr_p10": float(np.percentile(psnrs, 10)),
        "de00_mean_over_luts": float(des.mean()),
        "de00_p50": float(np.percentile(des, 50)),
        "de00_p90": float(np.percentile(des, 90)),
        "alive_frac_mean": float(np.mean([r["alive_frac"] for r in rows])),
        "anchor": {"psnr_db": 45.47, "de00": 0.41},
        "git_commit": git_commit(), "args": vars(args),
    }
    with open(os.path.join(args.outdir, f"{args.tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)
    print("E1_ON_A0_DONE", flush=True)


if __name__ == "__main__":
    main()
