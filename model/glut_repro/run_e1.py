"""E1 driver: per-LUT direct-fit N sweep on the stratified D-CUBE subset.

Usage (from repo root):
  python -m model.glut_repro.run_e1 --subset <txt: id<TAB>npy33<TAB>minor> \
      --n-list 8,16,24,32,48,64,96,128 \
      --outdir experiments/E1_cube_N_20260803/runs/main [--steps 3000] \
      [--batch-luts 16]

Per (LUT, N) row -> per_fit.jsonl; aggregation (cross-LUT p50/p90/p99/max of
per-LUT mean DeltaE00, r(N)=p90(N)/p90(2N), N*) -> metrics.json.
Saturation criteria (PLAN "第一级" step 6, preregistered):
  r(N) > 1.25 unsaturated, < 1.15 saturated
  N*   = min N with p90 < 1.0 and p99 < 2.0   (thresholds on the cross-LUT
         distribution of per-LUT MEAN DeltaE00 over held-out colors)
  gate : p50 < 0.5 / p90 < 1.0 / p99 < 2.0, else stop and fix the renderer
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
from model.glut_repro.fit_e1 import E1Fitter  # noqa: E402
from model.glut_repro.train_a0 import (  # noqa: E402
    de00_worker, psnr_8bit, psnr_float)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def aggregate(rows: list[dict], n_list: list[int]) -> dict:
    agg: dict = {"per_n": {}, "r_of_n": {}, "n_star": None}
    for n in n_list:
        de = np.array([r["de00"]["mean"] for r in rows if r["n"] == n])
        if de.size == 0:
            continue
        agg["per_n"][str(n)] = {
            "n_luts": int(de.size),
            "p50": float(np.percentile(de, 50)),
            "p90": float(np.percentile(de, 90)),
            "p99": float(np.percentile(de, 99)),
            "max": float(de.max()),
            "mean": float(de.mean()),
            "psnr_mean": float(np.mean(
                [r["psnr_float"] for r in rows if r["n"] == n])),
            "alive_frac_mean": float(np.mean(
                [r["alive_frac"] for r in rows if r["n"] == n])),
        }
    for n in n_list:
        if str(n) in agg["per_n"] and str(2 * n) in agg["per_n"]:
            p90n = agg["per_n"][str(n)]["p90"]
            p902n = agg["per_n"][str(2 * n)]["p90"]
            r = p90n / max(p902n, 1e-9)
            agg["r_of_n"][str(n)] = {
                "r": float(r),
                "verdict": ("unsaturated" if r > 1.25 else
                            "saturated" if r < 1.15 else "transition")}
    for n in sorted(n_list):
        pn = agg["per_n"].get(str(n))
        if pn and pn["p90"] < 1.0 and pn["p99"] < 2.0:
            agg["n_star"] = n
            break
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", required=True,
                    help="txt: <lut_id>\\t<npy33_path>\\t<minor> per line")
    ap.add_argument("--n-list", default="8,16,24,32,48,64,96,128")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-luts", type=int, default=16)
    ap.add_argument("--gt-workers", type=int, default=12)
    ap.add_argument("--de-workers", type=int, default=12)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    n_list = [int(s) for s in args.n_list.split(",")]
    luts = []
    with open(args.subset) as f:
        for line in f:
            if line.strip():
                lid, npy, minor = line.rstrip("\n").split("\t")
                luts.append({"id": lid, "npy": npy, "minor": minor})
    print(f"[e1] {len(luts)} LUTs x N in {n_list}", flush=True)

    # 1) GT cache
    jobs = [("e1", l["id"], l["npy"], ("train", "evalsub")) for l in luts]
    t0 = time.time()
    errs = [e for _, e in data.build_gt_parallel(jobs, args.gt_workers) if e]
    if errs:
        raise RuntimeError(f"GT build failures: {errs}")
    print(f"[e1] GT cache ready in {time.time()-t0:.0f}s", flush=True)

    colors_train = data.train_colors()
    colors_evalsub = data.eval_colors()[data.evalsub_indices()]

    per_fit_path = os.path.join(args.outdir, "per_fit.jsonl")
    done: set[tuple[str, int]] = set()
    rows: list[dict] = []
    if os.path.exists(per_fit_path):          # resume support
        with open(per_fit_path) as f:
            for line in f:
                r = json.loads(line)
                rows.append(r)
                done.add((r["lut_id"], r["n"]))
        print(f"[e1] resume: {len(rows)} fits already done", flush=True)

    pred_dir = os.path.join(args.outdir, "pred_tmp")
    os.makedirs(pred_dir, exist_ok=True)
    for n in n_list:
        todo = [l for l in luts if (l["id"], n) not in done]
        for c0 in range(0, len(todo), args.batch_luts):
            chunk = todo[c0:c0 + args.batch_luts]
            ids = [l["id"] for l in chunk]
            gts = np.stack([np.load(data._cache_path("e1", lid, "train"))
                            for lid in ids])
            t0 = time.time()
            fitter = E1Fitter(colors_train, gts, n, seed=args.seed,
                              steps=args.steps, bs=args.bs, lr=args.lr)
            stats = fitter.fit()
            pred = fitter.predict_colors(colors_evalsub)
            de_jobs = []
            for k, lid in enumerate(ids):
                pp = os.path.join(pred_dir, f"{lid}_{n}.npy")
                np.save(pp, pred[k].astype(np.float16))
                de_jobs.append(
                    (pp, data._cache_path("e1", lid, "evalsub"), slice(None)))
            from multiprocessing import Pool
            with Pool(min(args.de_workers, len(ids))) as pool:
                de_stats = pool.map(de00_worker, de_jobs)
            with open(per_fit_path, "a") as f:
                for k, lid in enumerate(ids):
                    gt = np.load(data._cache_path("e1", lid, "evalsub")) \
                        .astype(np.float32)
                    row = {"lut_id": lid, "n": n,
                           "minor": chunk[k]["minor"],
                           "de00": de_stats[k],
                           "psnr_float": psnr_float(pred[k], gt),
                           "psnr_8bit": psnr_8bit(pred[k], gt),
                           "alive_frac": stats["alive_frac"][k],
                           "n_relocated_total": stats["n_relocated_total"],
                           "final_loss": fitter.log[-1]["loss"]}
                    rows.append(row)
                    f.write(json.dumps(row) + "\n")
            for lid in ids:
                os.remove(os.path.join(pred_dir, f"{lid}_{n}.npy"))
            print(f"[e1] N={n} chunk {c0//args.batch_luts}: {len(ids)} LUTs "
                  f"in {time.time()-t0:.0f}s "
                  f"de00_mean={np.mean([d['mean'] for d in de_stats]):.3f}",
                  flush=True)
        # aggregate incrementally after each N
        agg = aggregate(rows, n_list)
        summary = {"exp": "E1_cube_N", "n_list": n_list,
                   "n_luts": len(luts), "steps": args.steps,
                   "criteria": {"p50": 0.5, "p90": 1.0, "p99": 2.0,
                                "r_unsaturated": 1.25, "r_saturated": 1.15},
                   **agg, "git_commit": git_commit(), "args": vars(args)}
        with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
            json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
