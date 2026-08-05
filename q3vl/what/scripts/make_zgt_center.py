"""Compute ``mean_train_u`` and publish every LUT's frozen ``z_gt`` (protocol 7.1).

    u(T) = flatten(T(x) - x),  x a fixed 17^3 RGB grid
    z_gt = L2Norm(SRHT_1024(u(T) - mean_train_u))

    "The SRHT uses a fixed public seed, **the centre is computed on the train LUTs
     only**, and no LUT lookup is learned."

"Train LUTs only" is the whole point of this being a separate job: if the centre
were computed over the corpus it would carry information from ``V_what``,
``T_final`` and above all ``T_lut_unseen`` into every training target, and the
unseen-LUT claim of protocol 2.2 would be quietly false.  The centre is therefore
taken over the ``lut_id`` set of the *train* split alone, and this job records
which ids went into it.

``z_gt`` itself is computed for every LUT in every split (the evaluation splits
need it too) but *always* with the train centre.

Outputs (protocol 2.3: durable, checksummed, atomically published)
-----------------------------------------------------------------
``zgt_center.npz``   ``mean_u`` (14739,) float32, ``d_func_scale`` (amendment A-2),
                     ``n_train_lut``, srht facts
``zgt.jsonl``        one row per lut_id: ``z_gt`` (1024 float32, base64), split set
``center_report.json``  provenance: split digests, id counts, elapsed

NOT EXECUTED.  It needs the GT-LUT shards (or the raw ``.cube`` corpus) and is
IO-bound; scheduling is the main agent's call.

Usage
-----
    python -m q3vl.what.scripts.make_zgt_center --gtluts <shard root> \
        --out /mnt/nfs/bc/data/datasets/what-20260805/zgt
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from q3vl.what.config import GTLUT_DIR, SCHEMA_ZGT, ZGT_DIR, ZGT_GRID
from q3vl.what.data import WhatDataset
from q3vl.what.lut import LutBank
from q3vl.what.srht import default_srht, encode_z_gt, identity_grid, pairwise_rms, u_of_table

PRODUCER = "q3vl.what.scripts.make_zgt_center/1"
TRAIN_SPLIT = "train"
EVAL_SPLITS = ("V_where", "V_what", "T_final", "T_lut_unseen")


def _bank(gtluts: Path | None, path_map: dict[str, str]) -> LutBank:
    if gtluts is not None and (Path(gtluts) / "manifest.json").exists():
        from q3vl.whereb.stores import PublishedStore

        return LutBank(store=PublishedStore(gtluts), path_map=path_map, capacity=8)
    return LutBank(path_map=path_map, capacity=8)


def split_lut_ids(split: str, verify: str = "checksum") -> dict[str, str]:
    ds = WhatDataset(split, need_mask=False, verify=verify)
    return ds.lut_path_map()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gtluts", default=str(GTLUT_DIR))
    ap.add_argument("--out", default=str(ZGT_DIR))
    ap.add_argument("--interp", default="trilinear",
                    choices=("trilinear", "tetrahedral"))
    ap.add_argument("--verify", default="checksum")
    args = ap.parse_args()

    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_map = split_lut_ids(TRAIN_SPLIT, args.verify)
    all_map = dict(train_map)
    membership: dict[str, list[str]] = {k: [TRAIN_SPLIT] for k in train_map}
    for split in EVAL_SPLITS:
        m = split_lut_ids(split, args.verify)
        for k, v in m.items():
            all_map.setdefault(k, v)
            membership.setdefault(k, []).append(split)

    bank = _bank(Path(args.gtluts), all_map)
    grid = identity_grid(ZGT_GRID)
    srht = default_srht()

    # --- pass 1: the centre AND the d_func scale, over the TRAIN ids only ----
    # Amendment A-2's C is the RMS of ||u_i - u_j|| over all N(N-1)/2 train pairs.
    # It needs no second pass and no sampling: sum_i u_i and sum_i ||u_i||^2 are
    # enough (see q3vl.what.srht.pairwise_rms), and both accumulate here.
    total = torch.zeros(ZGT_GRID ** 3 * 3, dtype=torch.float64)
    sum_sq = 0.0
    n_train = len(train_map)
    for i, lid in enumerate(sorted(train_map)):
        with torch.no_grad():
            u = u_of_table(bank.get(lid).apply(grid, args.interp)).double()
        total += u
        sum_sq += float(u.pow(2).sum())
        if (i + 1) % 500 == 0:
            print(f"centre {i + 1}/{n_train}", flush=True)
    center = (total / max(1, n_train)).float()
    d_func_scale = pairwise_rms(n_train, total, sum_sq)

    np.savez(out / "zgt_center.npz", mean_u=center.numpy(),
             d_func_scale=np.float64(d_func_scale),
             n_train_lut=np.int64(n_train),
             srht=json.dumps(srht.facts()))

    # --- pass 2: z_gt for every LUT, always with the train centre -----------
    rows = 0
    with (out / "zgt.jsonl").open("w", encoding="utf-8") as fh:
        for lid in sorted(all_map):
            with torch.no_grad():
                z = encode_z_gt(bank.get(lid).apply(grid, args.interp), center, srht)
            fh.write(json.dumps({
                "schema_version": SCHEMA_ZGT, "lut_id": lid,
                "splits": membership[lid],
                "in_center": TRAIN_SPLIT in membership[lid],
                "z_gt_b64": base64.b64encode(
                    z.numpy().astype(np.float32).tobytes()).decode("ascii"),
            }) + "\n")
            rows += 1

    report: dict[str, Any] = {
        "producer": PRODUCER,
        "interp": args.interp,
        "n_train_lut_in_center": n_train,
        "d_func_scale": d_func_scale,
        "d_func_scale_definition": (
            "amendment A-2: RMS of ||u_i - u_j|| over all N(N-1)/2 train LUT "
            "pairs, closed form from sum_i u_i and sum_i ||u_i||^2"),
        "n_lut_total": len(all_map),
        "n_rows": rows,
        "center_sha256": hashlib.sha256(center.numpy().tobytes()).hexdigest(),
        "center_abs_mean": float(center.abs().mean()),
        "srht": srht.facts(),
        "lut_bank": bank.facts(),
        "elapsed_s": round(time.time() - t0, 1),
        "note": ("the centre uses TRAIN lut_ids only; every split's z_gt uses that "
                 "same centre, which is what keeps T_lut_unseen unseen"),
    }
    (out / "center_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    raise SystemExit(main())
