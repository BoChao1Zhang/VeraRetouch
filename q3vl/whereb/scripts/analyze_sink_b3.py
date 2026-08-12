"""E1b / B3: settle P-W1's registered completion standard on ONE stated rule.

Three口径 problems the review found in the published REPORT:

1. the registered standard is "sink set Jaccard >= 0.8 **across resolution
   strata**", and the delivery reported a ``sink_frac`` range instead.  Cell
   indices cannot transfer between a 16x24 and a 24x16 grid, so this computes the
   Jaccard on **normalised sink positions** binned to a common grid -- the
   registered quantity, in the only coordinate system where it is defined;
2. the REPORT's table quoted the **single-arm** rule (11.98% of cells) while the
   rule that actually entered the contract is the **conjunctive** one (9.90%);
3. the no-exclusion oracle ceiling was quoted as 0.791, which is not reproducible
   from either JSON -- the value is 0.787.

Everything here is recomputed from the exported profiles so the REPORT can quote
one rule with one set of numbers.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

NBIN = 8


def _agg(v) -> dict[str, Any] | None:
    a = np.asarray([x for x in v if x is not None], dtype=np.float64)
    if not a.size:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=3.0)
    args = ap.parse_args(argv)

    from q3vl.whereb.attnread import sink_mask_conjunctive, sink_mask_from_profile

    exp = Path(args.export)
    rows = [json.loads(l) for l in (exp / "samples.jsonl").read_text().splitlines() if l.strip()]
    by = defaultdict(dict)
    for r in rows:
        by[r["sample_id"]][r["arm"]] = r

    single, conj = [], []
    occ: dict[tuple[int, int], np.ndarray] = {}
    cnt: dict[tuple[int, int], int] = defaultdict(int)

    for sid, arms in by.items():
        if "gt" not in arms or "shuffled" not in arms:
            continue
        dg = np.load(exp / "fields" / f"{sid}__gt.npz")
        dsh = np.load(exp / "fields" / f"{sid}__shuffled.npz")
        pg = dg["col_profile"].astype(np.float64)
        psh = dsh["col_profile"].astype(np.float64)
        mg, msh = pg.mean(axis=(0, 1)), psh.mean(axis=(0, 1))
        gh, gw = arms["gt"]["grid_h"], arms["gt"]["grid_w"]
        n = mg.size
        gt = np.load(exp / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)

        s1 = sink_mask_from_profile(mg, k=args.k)
        sc = sink_mask_conjunctive(mg, msh, k=args.k)
        am = pg.reshape(-1, n).argmax(axis=1)

        def stats(mask):
            v = ~mask
            kk = max(1, min(int((gt[v] > 0.5).sum()), int(v.sum())))
            o = np.zeros(n)
            cand = np.flatnonzero(v)
            o[cand[np.argsort(-gt[v])[:kk]]] = 1.0
            ceil = float(np.minimum(o, gt).sum() / (np.maximum(o, gt).sum() + 1e-8))
            return {
                "sink_frac": float(mask.mean()),
                "mass_share": float(mg[mask].sum() / mg.sum()) if mask.any() else 0.0,
                "argmax_in_sink": float(mask[am].mean()),
                "gt_mass_in_sink": float(gt[mask].sum() / gt.sum()) if gt.sum() else 0.0,
                "ceiling": ceil,
            }

        single.append(stats(s1))
        conj.append(stats(sc))

        # normalised sink occupancy on a common NBIN x NBIN grid
        ys, xs = np.nonzero(sc.reshape(gh, gw))
        m = occ.setdefault((gh, gw), np.zeros((NBIN, NBIN)))
        for y, x in zip(ys, xs):
            by_ = min(int(y / gh * NBIN), NBIN - 1)
            bx_ = min(int(x / gw * NBIN), NBIN - 1)
            m[by_, bx_] += 1
        cnt[(gh, gw)] += 1

    k0 = max(1, 0)
    # no-exclusion ceiling, recomputed
    noex = []
    for sid, arms in by.items():
        if "gt" not in arms:
            continue
        p = exp / "gt" / f"{sid}.npy"
        if not p.exists():
            continue
        g = np.load(p).astype(np.float64).reshape(-1)
        kk = max(1, int((g > 0.5).sum()))
        o = np.zeros(g.size)
        o[np.argsort(-g)[:kk]] = 1.0
        noex.append(float(np.minimum(o, g).sum() / (np.maximum(o, g).sum() + 1e-8)))

    def col(rs, key):
        return _agg([r[key] for r in rs])

    # cross-resolution Jaccard on normalised occupancy, shapes with n >= 10
    shapes = [s for s in occ if cnt[s] >= 10]
    maps = {}
    for s in shapes:
        rate = occ[s] / cnt[s]
        thr = np.median(rate[rate > 0]) if (rate > 0).any() else 0.0
        maps[s] = rate > thr
    jac = {}
    vals = []
    for i, a in enumerate(shapes):
        for b in shapes[i + 1:]:
            inter = int((maps[a] & maps[b]).sum())
            union = int((maps[a] | maps[b]).sum())
            j = inter / union if union else 1.0
            jac[f"{a[0]}x{a[1]} vs {b[0]}x{b[1]}"] = j
            vals.append(j)

    out = {
        "card": "PR-ATT1-E1b / B3 -- P-W1 口径统一",
        "n_samples": len(conj),
        "mad_k": args.k,
        "RULE_IN_CONTRACT": "conjunctive (gt AND shuffled above median+k*MAD)",
        "single_arm_rule": {k: col(single, k) for k in
                            ("sink_frac", "mass_share", "argmax_in_sink",
                             "gt_mass_in_sink", "ceiling")},
        "conjunctive_rule": {k: col(conj, k) for k in
                             ("sink_frac", "mass_share", "argmax_in_sink",
                              "gt_mass_in_sink", "ceiling")},
        "oracle_ceiling_no_exclusion": _agg(noex),
        "cross_resolution_jaccard": {
            "definition": (f"normalised sink occupancy binned to {NBIN}x{NBIN}, "
                           "thresholded at each shape's own median positive rate; "
                           "Jaccard between shapes with n>=10 samples"),
            "shapes": [f"{a}x{b}" for a, b in shapes],
            "pairwise": jac,
            "median": float(np.median(vals)) if vals else None,
            "min": float(np.min(vals)) if vals else None,
            "REGISTERED_GATE": 0.8,
            "passes": bool(vals and float(np.min(vals)) >= 0.8),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"n = {out['n_samples']}")
    for name in ("single_arm_rule", "conjunctive_rule"):
        r = out[name]
        print(f"\n{name}:")
        for k, v in r.items():
            print(f"   {k:18s} median {v['median']:.4f}  mean {v['mean']:.4f}")
    print(f"\noracle ceiling NO exclusion: median "
          f"{out['oracle_ceiling_no_exclusion']['median']:.4f}")
    cj = out["cross_resolution_jaccard"]
    print(f"\ncross-resolution Jaccard: median {cj['median']:.3f} min {cj['min']:.3f} "
          f"gate {cj['REGISTERED_GATE']} -> {'PASS' if cj['passes'] else 'FAIL'}")
    for k, v in sorted(cj["pairwise"].items(), key=lambda kv: kv[1]):
        print(f"   {k:22s} {v:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
