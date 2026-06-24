"""Analyze r2_calib.json file(s): per-saturation-band mean best_q with bootstrap CIs, the
saturation-conditional selector (vlemb on colorful / random on muted) and its best threshold,
and per-source winners. De-risk tooling for the R2 selector decision (small-n -> show CIs).

CLI:  python -m construct._r2_analyze f1.json [f2.json ...]
"""
from __future__ import annotations

import json
import sys

import numpy as np


def _boot(xs, reps=2000, seed=0):
    if not xs:
        return (None, None, None)
    a = np.asarray(xs, "float32"); rng = np.random.default_rng(seed)
    bs = rng.choice(a, size=(reps, len(a)), replace=True).mean(1)
    return round(float(a.mean()), 2), round(float(np.percentile(bs, 5)), 2), round(float(np.percentile(bs, 95)), 2)


def load(files):
    out = []
    for f in files:
        out += json.load(open(f))["results"]
    return out


def main():
    R = load(sys.argv[1:])
    print(f"n={len(R)} sources\n")
    ARMS = tuple(R[0]["best_q"].keys()) if R else ()
    bands = [("low <0.15", lambda s: s < 0.15), ("mid 0.15-0.30", lambda s: 0.15 <= s < 0.30),
             ("hi >=0.30", lambda s: s >= 0.30), ("ALL", lambda s: True)]
    for name, f in bands:
        g = [r for r in R if f(r["sat"])]
        if not g:
            continue
        print(f"== {name} (n={len(g)}) ==  mean best_q [90% CI]")
        for a in ARMS:
            xs = [r["best_q"][a] for r in g if r["best_q"].get(a, -99) > -90]
            m, lo, hi = _boot(xs)
            print(f"   {a:9s} {m:+.2f}  [{lo:+.2f},{hi:+.2f}]" if m is not None else f"   {a:9s}  -")
    # saturation-conditional selector: vlemb if sat>=thr else random; sweep thr
    print("\nsaturation-conditional (vlemb if sat>=thr else random):")
    for thr in (0.15, 0.20, 0.25, 0.30):
        qs = [r["best_q"]["vlemb"] if r["sat"] >= thr else r["best_q"]["random"] for r in R
              if r["best_q"]["vlemb"] > -90 and r["best_q"]["random"] > -90]
        m, lo, hi = _boot(qs)
        print(f"   thr={thr}: {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    # pure single-selector baselines for comparison
    for a in ("vlemb", "random"):
        xs = [r["best_q"][a] for r in R if r["best_q"].get(a, -99) > -90]
        m, lo, hi = _boot(xs)
        print(f"   pure {a}: {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    # per-source winner
    from collections import Counter
    w = Counter()
    for r in R:
        bq = {a: r["best_q"][a] for a in ARMS if r["best_q"].get(a, -99) > -90}
        if bq:
            w[max(bq, key=bq.get)] += 1
    print("\nper-source winner:", dict(w))
    # paired vlemb-minus-random by band (the key contrast)
    print("\nvlemb - random (paired) by band:")
    for name, f in bands:
        g = [r for r in R if f(r["sat"]) and r["best_q"]["vlemb"] > -90 and r["best_q"]["random"] > -90]
        d = [r["best_q"]["vlemb"] - r["best_q"]["random"] for r in g]
        m, lo, hi = _boot(d)
        if m is not None:
            print(f"   {name:14s} Δ={m:+.2f} [{lo:+.2f},{hi:+.2f}]  (n={len(g)})")


if __name__ == "__main__":
    main()
