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

    # ---------------- supplement round (2026-08-03) -----------------------
    # gap B: the delivered `gauss` path (strict single Gaussian, amplitude 1)
    "main14_gauss": {"cols": "full", "readouts": ["gauss"], "as": "main14",
                     "families": GEO_FAMS + ["constant", "semantic"]},
    "geo6_ring_gauss": {"cols": "geo", "readouts": ["gauss"],
                        "as": "geo6_ring", "families": ["ring"]},
    # gap A: the renderer's own constrained s axis (M=12, mu fixed, sigma
    # bounded) -- can it synthesize the flat top the 0.975 depends on?
    "main14_cband": {"cols": "full", "readouts": ["cband_norm"],
                     "as": "main14", "families": GEO_FAMS},
    "main14_cgauss": {"cols": "full", "readouts": ["cgauss"],
                      "as": "main14", "families": GEO_FAMS},
    "main14_cband_unnorm": {"cols": "full", "readouts": ["cband_unnorm"],
                            "as": "main14", "families": ["ring"]},
    "geo6_ring_cband": {"cols": "geo",
                        "readouts": ["cband_norm", "cband_unnorm"],
                        "as": "geo6_ring", "families": ["ring"]},
    "geo6_ring_cgauss": {"cols": "geo", "readouts": ["cgauss"],
                         "as": "geo6_ring", "families": ["ring"]},
    # ring-only variants used by the A-4 budget control and the A-5 sweeps
    "main14_cband_ring": {"cols": "full", "readouts": ["cband_norm"],
                          "as": "main14", "families": ["ring"]},
    "geo6_ring_cband_norm": {"cols": "geo", "readouts": ["cband_norm"],
                             "as": "geo6_ring", "families": ["ring"]},
    # the three non-ring families are ~430 s/fit under the constrained band
    # (vs ~90 s for rings), so they run at reduced n for the completeness
    # cells of the criteria matrix -- see REPORT "reduced n" note
    "main14_cband_geo3": {"cols": "full", "readouts": ["cband_norm"],
                          "as": "main14",
                          "families": ["linear", "radial_ell", "wedge"]},
    # ---------------- wave 3 (2026-08-03) ---------------------------------
    # budget-matched control for A-4: the UNCONSTRAINED band-pass arm at the
    # same max_iter as the constrained arm, so the A-1 drop can be read at a
    # matched budget instead of only at 120 steps.
    "main14_band_ring": {"cols": "full", "readouts": ["bandpass"],
                         "as": "main14", "families": ["ring"]},
    "geo6_band_ring": {"cols": "geo", "readouts": ["bandpass"],
                       "as": "geo6_ring", "families": ["ring"]},
    # geometry-only wedge control: separates "quadratic geometric terms" from
    # "image/semantic channels" as the source of wedge 0.942 (REVIEW-result
    # section 3.1 asked for this; round 1 only had the ring geo-only control)
    "geo6_wedge": {"cols": "geo", "readouts": ["monotone"],
                   "as": "geo6", "families": ["wedge"]},
}

_worker_state: dict = {}
_WARM: dict = {}          # (config_label, mask_id) -> prior band-pass row


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
    axis = task.get("axis")
    if task["readout"] in ("cband_norm", "cband_unnorm"):
        prior = _WARM.get((task["config"], row["mask_id"]))
        for n_on in (1, 3):
            ws = e2lib.band_warm_start(prior, axis, n_on=n_on)
            if ws is not None:
                extra = extra + [ws]
    t0 = time.time()
    try:
        res = e2lib.fit_mask(Phi_fit, t_fit, Phi_eval, t_eval,
                             task["readout"],
                             seed=abs(hash(row["mask_id"])) % (2 ** 31),
                             n_random=task["n_random"],
                             max_iter=task["max_iter"],
                             extra_starts=extra, axis=axis)
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
        "max_iter": task["max_iter"], "arm": task.get("arm", "main"),
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
    ap.add_argument("--out-tag", default="",
                    help="shard suffix: results<smoke><tag>.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="max masks per family (0 = all)")
    ap.add_argument("--arm", default="main",
                    help="label written into every row (sweep bookkeeping)")
    ap.add_argument("--axis-m", type=int, default=0)
    ap.add_argument("--axis-sig-lo", type=float, default=0.0)
    ap.add_argument("--axis-sig-hi", type=float, default=0.0)
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import e2lib
    axis = dict(e2lib.CONSTRAINED_AXIS)
    if args.axis_m:
        axis["M"] = args.axis_m
    if args.axis_sig_lo:
        axis["sig_lo"] = args.axis_sig_lo
    if args.axis_sig_hi:
        axis["sig_hi"] = args.axis_sig_hi

    # warm starts for the constrained readouts: reuse the *unconstrained*
    # band-pass solution of the same mask from the main shard (fairness -- we
    # are asking about expressiveness, not optimizer luck).  Loaded in the
    # parent so fork children share it copy-on-write.
    main_shard = EXP / f"results{suffix}.jsonl"
    if main_shard.exists():
        for line in open(main_shard):
            r = json.loads(line)
            if r.get("readout") == "bandpass" and "error" not in r:
                _WARM[(r["config"], r["mask_id"])] = r
    print(f"warm-start rows: {len(_WARM)}", flush=True)

    rows = [json.loads(line) for line in open(CACHE / f"masks{suffix}.jsonl")]
    tasks = []
    for cname in args.configs:
        cfg = CONFIGS[cname]
        label = cfg.get("as", cname)
        seen: dict = {}
        for row in rows:
            fam = row["family"]
            if fam not in cfg["families"]:
                continue
            seen[fam] = seen.get(fam, 0) + 1
            if args.limit and seen[fam] > args.limit:
                continue
            for ro in cfg["readouts"]:
                tasks.append({"row": row, "config": label,
                              "cols": cfg["cols"], "readout": ro,
                              "stride": args.stride,
                              "max_iter": args.max_iter,
                              "n_random": args.n_random,
                              "arm": args.arm,
                              "axis": axis, "suffix": suffix})
    print(f"tasks: {len(tasks)}", flush=True)
    t0 = time.time()

    out_path = EXP / f"results{suffix}{args.out_tag}.jsonl"
    import multiprocessing as mp
    # chunksize 4 idles most workers when there are few tasks (a shard of 8
    # tasks on 8 workers would run on 2 workers), and each constrained fit is
    # minutes long -- so only chunk when there is enough work to amortize it.
    chunk = 4 if len(tasks) >= args.workers * 8 else 1
    with mp.get_context("fork").Pool(args.workers) as pool, \
            open(out_path, "w") as fh:
        for i, res in enumerate(pool.imap_unordered(fit_task, tasks,
                                                    chunksize=chunk)):
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
        "n_random": args.n_random, "arm": args.arm,
        "constrained_axis": axis, "n_warm_start_rows": len(_WARM),
        "wall_seconds": round(time.time() - t0, 1)}
    with open(EXP / "config" / f"fit{suffix}{args.out_tag}.json", "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(json.dumps(cfg), flush=True)


if __name__ == "__main__":
    main()
