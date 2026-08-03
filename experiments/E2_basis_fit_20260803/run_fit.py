"""E2 fitting: per-mask L-BFGS over configs x readouts (multiprocessing).

Configs (dir-feature columns; constant handled by w0):
  main14    geo5 + range2 + sem6          readouts: monotone + bandpass, all
  lin3      [x, y] (Exposure reproduction) monotone, geometric families
  cubic18   geo5 + cubic4 + range2 + sem6  monotone, geometric families
  dims6     geo5                           monotone, semantic masks
  dims8     geo5 + range2                  monotone, semantic masks

Fit at stride 2 (256^2-equivalent), evaluate soft-IoU at full 512-res.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

EXP = Path("/home/bc/VeraRetouch/experiments/E2_basis_fit_20260803")
CACHE = Path("/var/cache/veradata/e2_basis_20260803")
sys.path.insert(0, str(EXP))

GEO_FAMS = ["linear", "radial_ell", "ring", "wedge"]
CONFIGS = {
    "main14": {"cols": "full", "readouts": ["monotone", "bandpass"],
               "families": GEO_FAMS + ["constant", "semantic"]},
    "lin3": {"cols": "lin", "readouts": ["monotone"], "families": GEO_FAMS},
    "cubic18": {"cols": "cubic", "readouts": ["monotone"],
                "families": GEO_FAMS},
    "dims6": {"cols": "geo", "readouts": ["monotone"],
              "families": ["semantic"]},
    "dims8": {"cols": "geo_range", "readouts": ["monotone"],
              "families": ["semantic"]},
    # geometry-only ring control: no image channels to exploit -> clean
    # monotone-vs-bandpass expressiveness comparison
    "geo6_ring": {"cols": "geo", "readouts": ["monotone", "bandpass"],
                  "families": ["ring"]},
}

_worker_state: dict = {}


def _feat_cache(key: str, suffix: str):
    if key in _worker_state.setdefault("feats", {}):
        return _worker_state["feats"][key]
    z = np.load(CACHE / f"feats{suffix}" / f"{key}.npz")
    d = {"L": z["L"].astype(np.float64), "S": z["S"].astype(np.float64),
         "e": z["e"].astype(np.float64)}
    cache = _worker_state["feats"]
    if len(cache) > 12:
        cache.clear()
    cache[key] = d
    return d


def build_phi(h, w, stride, cols, ch):
    import e2lib
    if cols == "lin":
        X, Y = e2lib.norm_coords(h, w, stride)
        return np.stack([X.reshape(-1), Y.reshape(-1)], axis=1)
    if cols == "geo":
        return e2lib.geo_features(h, w, stride)
    geo = e2lib.geo_features(h, w, stride, cubic=(cols == "cubic"))
    L = ch["L"][::stride, ::stride].reshape(-1, 1)
    S = ch["S"][::stride, ::stride].reshape(-1, 1)
    if cols == "geo_range":
        return np.concatenate([geo, L, S], axis=1)
    e = ch["e"][::stride, ::stride].reshape(-1, 6)
    return np.concatenate([geo, L, S, e], axis=1)


def fit_task(task):
    import torch
    torch.set_num_threads(1)
    import e2lib
    suffix = task["suffix"]
    row = task["row"]
    z = np.load(CACHE / f"masks{suffix}" / f"{row['mask_id']}.npz")
    mask = z["mask"].astype(np.float64)
    h, w = mask.shape
    ch = _feat_cache(row["feat_key"], suffix)
    stride = task["stride"]
    Phi_fit = build_phi(h, w, stride, task["cols"], ch)
    Phi_eval = build_phi(h, w, 1, task["cols"], ch)
    t_fit = mask[::stride, ::stride].reshape(-1)
    t_eval = mask.reshape(-1)
    extra = (e2lib.radial_starts(Phi_fit, t_fit, task["readout"])
             if task["cols"] != "lin" else [])
    t0 = time.time()
    try:
        res = e2lib.fit_mask(Phi_fit, t_fit, Phi_eval, t_eval,
                             task["readout"],
                             seed=abs(hash(row["mask_id"])) % (2 ** 31),
                             n_random=task["n_random"],
                             max_iter=task["max_iter"],
                             extra_starts=extra)
    except Exception as exc:  # noqa: BLE001
        return {"mask_id": row["mask_id"], "config": task["config"],
                "readout": task["readout"], "error": repr(exc)}
    res.update({
        "mask_id": row["mask_id"], "family": row["family"],
        "config": task["config"], "readout": task["readout"],
        "class5": row.get("class5"), "slot_id": row.get("slot_id"),
        "feat_key": row["feat_key"], "n_dirs": Phi_fit.shape[1],
        "target_mean": float(mask.mean()),
        "fit_seconds": round(time.time() - t0, 2),
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    rows = [json.loads(line) for line in open(CACHE / f"masks{suffix}.jsonl")]
    tasks = []
    for cname in args.configs:
        cfg = CONFIGS[cname]
        for row in rows:
            if row["family"] not in cfg["families"]:
                continue
            for ro in cfg["readouts"]:
                tasks.append({"row": row, "config": cname,
                              "cols": cfg["cols"], "readout": ro,
                              "stride": args.stride,
                              "max_iter": args.max_iter,
                              "n_random": args.n_random,
                              "suffix": suffix})
    print(f"tasks: {len(tasks)}", flush=True)
    t0 = time.time()

    out_path = EXP / f"results{suffix}.jsonl"
    import multiprocessing as mp
    with mp.get_context("fork").Pool(args.workers) as pool, \
            open(out_path, "w") as fh:
        for i, res in enumerate(pool.imap_unordered(fit_task, tasks,
                                                    chunksize=4)):
            fh.write(json.dumps(res) + "\n")
            if (i + 1) % 200 == 0 or i + 1 == len(tasks):
                fh.flush()
                print(f"{i + 1}/{len(tasks)} elapsed={time.time() - t0:.0f}s",
                      flush=True)

    cfg = {"git_commit": subprocess.run(
        ["git", "-C", "/home/bc/VeraRetouch", "rev-parse", "HEAD"],
        capture_output=True, text=True).stdout.strip(),
        "argv": sys.argv, "n_tasks": len(tasks),
        "stride": args.stride, "max_iter": args.max_iter,
        "n_random": args.n_random,
        "wall_seconds": round(time.time() - t0, 1)}
    with open(EXP / "config" / f"fit{suffix}.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(json.dumps(cfg), flush=True)


if __name__ == "__main__":
    main()
