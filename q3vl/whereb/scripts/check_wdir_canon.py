"""Item-0 audit: is the ``w_dir`` sign canonicalisation applied *identically*
on the training-target read path as it was on the Where-A oracle write path?

Why this matters (amort E1 NOTES §三.A): ``canonicalize`` in
:mod:`q3vl.where.basis` kills the exact 2-fold symmetry
``(w_raw, w0, rho) -> (-w_raw, -w0, mirror(rho))`` which leaves the decoded mask
bit-identical.  If the published latents are canonical but the trainer rebuilt
``w_dir`` from something else (or vice versa), the ``L_dir = 1 - cos`` target
carries an artificial bimodality that alone can explain the W01/W02 collapse.

The audit walks the *published* store the arms actually read
(``setup.datasets.train.oracle.root`` of ``arm_W01.json`` / ``arm_W02.json``)
and, per sample, compares three things:

  A. the on-disk ``latent.canonical`` flag written by ``Latent.to_dict``;
  B. the sign rule evaluated on the on-disk ``latent.w_dir`` array;
  C. the sign rule evaluated on ``w_dir_of(w_raw)`` -- i.e. *exactly* what
     :func:`q3vl.whereb.fields.oracle_fields` recomputes at train time, through
     the very same call the trainer makes.

Agreement of all three at 100% is the only thing that clears the hypothesis.

Usage::

    python -m q3vl.whereb.scripts.check_wdir_canon --limit 4000
    python -m q3vl.whereb.scripts.check_wdir_canon --limit 0   # full split
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from q3vl.where.basis import Latent, sign_index, w_dir_of
from q3vl.whereb.stores import OracleStore

DEFAULT_ROOTS = {
    "train": "/mnt/nfs-ro/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/train",
    "V_where": "/mnt/nfs-ro/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/V_where",
}
READOUTS = ("band", "cband12")


def audit(root: str, readout: str, limit: int, seed: int) -> dict:
    store = OracleStore(root, verify=False)
    ids = sorted(store.sample_ids)
    if limit and limit < len(ids):
        random.Random(seed).shuffle(ids)
        ids = sorted(ids[:limit])

    n = 0
    n_flag_canon = 0          # A: on-disk "canonical" flag true
    n_disk_wdir_pos = 0       # B: sign rule holds on the stored w_dir array
    n_train_wdir_pos = 0      # C: sign rule holds on w_dir_of(w_raw)
    n_bc_agree = 0            # B and C pick the same sign_index and same sign
    n_abc_agree = 0
    max_wdir_gap = 0.0        # ||stored w_dir - w_dir_of(w_raw)||_inf
    violators: list[str] = []
    # canonicalisation-induced discontinuity: the sign rule pivots on *which*
    # coefficient is largest in absolute value.  When |w|_(1) ~ |w|_(2) the
    # pivot is a coin toss, and an arbitrarily small input change flips the
    # sign of all 71 target coordinates.  Killing the 2-fold ambiguity trades
    # it for a boundary discontinuity; this measures how close the targets sit
    # to that boundary.
    margins: list[float] = []     # |w|_(1) / |w|_(2)
    pivots: list[int] = []        # sign_index
    live_flip: list[bool] = []    # runner-up carries the opposite sign
    t0 = time.time()

    for sid in ids:
        fits = store.payload(sid).get("fits") or {}
        fit = fits.get(readout)
        if fit is None or fit.get("status") != "ok":
            continue
        lat_d = fit.get("latent")
        if lat_d is None:
            continue
        n += 1

        flag = bool(lat_d.get("canonical"))
        wd_disk = torch.tensor(lat_d["w_dir"], dtype=torch.float64)
        # exactly the trainer's path: Latent.from_dict -> oracle_fields ->
        # w_dir_of(latent.w_raw)
        lat = Latent.from_dict(lat_d, dtype=torch.float64)
        wd_train = w_dir_of(lat.w_raw)

        disk_pos = bool(wd_disk[sign_index(wd_disk)] > 0)
        train_pos = bool(wd_train[sign_index(wd_train)] > 0)
        gap = float((wd_disk - wd_train).abs().max())
        max_wdir_gap = max(max_wdir_gap, gap)

        n_flag_canon += flag
        n_disk_wdir_pos += disk_pos
        n_train_wdir_pos += train_pos
        order = wd_train.abs().sort(descending=True).indices
        absw = wd_train.abs()[order]
        margins.append(float(absw[0] / (absw[1] + 1e-30)))
        pivots.append(int(sign_index(wd_train)))
        # A near-tie only *bites* when the runner-up carries the opposite sign:
        # the vector is canonicalised so w[pivot_1] > 0, so if w[pivot_2] < 0 an
        # infinitesimal perturbation that makes |w_2| > |w_1| forces a flip of
        # all 71 coordinates to restore the sign rule.  Same-sign runner-up
        # crosses the boundary continuously.
        live_flip.append(bool(wd_train[order[1]] < 0))

        same = (sign_index(wd_disk) == sign_index(wd_train)) and (disk_pos == train_pos)
        n_bc_agree += same
        if flag == disk_pos == train_pos and same:
            n_abc_agree += 1
        elif len(violators) < 20:
            violators.append(sid)

    store.close()
    mg = np.asarray(margins) if margins else np.zeros(0)
    lf = np.asarray(live_flip, dtype=bool) if live_flip else np.zeros(0, dtype=bool)
    pv = np.asarray(pivots) if pivots else np.zeros(0, dtype=int)
    margin_stats = {}
    if mg.size:
        margin_stats = {
            "pivot_margin_p05": float(np.percentile(mg, 5)),
            "pivot_margin_median": float(np.median(mg)),
            "pivot_margin_p95": float(np.percentile(mg, 95)),
            "frac_margin_below_1.05": float((mg < 1.05).mean()),
            "frac_margin_below_1.20": float((mg < 1.20).mean()),
            "frac_margin_below_2.00": float((mg < 2.0).mean()),
            "frac_runner_up_opposite_sign": float(lf.mean()),
            "frac_live_flip_boundary_margin<1.05": float((lf & (mg < 1.05)).mean()),
            "frac_live_flip_boundary_margin<1.20": float((lf & (mg < 1.20)).mean()),
            "n_distinct_pivots": int(np.unique(pv).size),
            "pivot_top3_share": [
                float(c / pv.size)
                for c in sorted(np.bincount(pv, minlength=71).tolist(), reverse=True)[:3]
            ],
        }
    return {
        **margin_stats,
        "root": root,
        "readout": readout,
        "n_ids_scanned": len(ids),
        "n_with_ok_fit": n,
        "n_flag_canonical": n_flag_canon,
        "n_disk_wdir_sign_ok": n_disk_wdir_pos,
        "n_train_wdir_sign_ok": n_train_wdir_pos,
        "n_disk_train_agree": n_bc_agree,
        "n_all_three_agree": n_abc_agree,
        "frac_all_three_agree": (n_abc_agree / n) if n else None,
        "max_abs_gap_disk_vs_train_wdir": max_wdir_gap,
        "violator_head": violators,
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=4000,
                    help="0 = whole split")
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--splits", nargs="*", default=["train", "V_where"])
    ap.add_argument("--readouts", nargs="*", default=list(READOUTS))
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    results = []
    for split in args.splits:
        for readout in args.readouts:
            r = audit(DEFAULT_ROOTS[split], readout, args.limit, args.seed)
            r["split"] = split
            results.append(r)
            print(json.dumps(r, ensure_ascii=False))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
