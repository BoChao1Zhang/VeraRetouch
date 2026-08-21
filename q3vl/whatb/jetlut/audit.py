"""EPR-032 solver-floor audit: how much L1 is still on the table at 2,000 iters?

    .venv-lens/bin/python -m q3vl.whatb.jetlut.audit --out DIR

The A-nested tolerance and every "p2 beats p1 by X%" statement rest on the
premise that both orders are solved to far better than X.  The rho scan measured
that at m=4 with 16 LUTs; this measures it at the *worst* configuration actually
run (coarse m, where r_pri is largest) on the real pool, by re-solving at 4x the
iteration budget and reporting the relative L1 improvement.

Reported, never used to change a headline number: the ladder rows stay as they
were solved.  This only says how big the floor is.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from q3vl.whatb.codec.lutcode import open_bank
from q3vl.whatb.jetlut.core import Atlas, admm_lad, design_matrix, n_dynamic_params
from q3vl.whatb.jetlut.run import fit_colours, pools, residuals

#: the corners of the ladder: coarsest (worst r_pri) and the two matched-budget
#: rows the G-main hull actually leans on.
CONFIGS = ((3, 1), (3, 2), (4, 2), (7, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/prs/"
                                     "EPR-032_jetlut-canonical-color-field/out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--c-star", type=float, default=0.7)
    ap.add_argument("--n-lut", type=int, default=256)
    ap.add_argument("--base-iter", type=int, default=2000)
    ap.add_argument("--long-mult", type=int, default=4)
    args = ap.parse_args()

    dev, dt = torch.device(args.device), torch.float64
    bank = open_bank()
    held, _, _ = pools(None)
    ids = held[:args.n_lut]
    x = fit_colours(dt).to(dev)
    r = residuals(bank, ids, x, dev, dt)

    rows = []
    for m, p in CONFIGS:
        f = design_matrix(x, Atlas(m=m, c=args.c_star), p)
        t0 = time.time()
        a = admm_lad(f, r, max_iter=args.base_iter)
        b = admm_lad(f, r, max_iter=args.base_iter * args.long_mult)
        l1a, l1b = float(a["primal"].sum()), float(b["primal"].sum())
        rows.append({
            "m": m, "p": p, "P_dyn": n_dynamic_params(m, p),
            "iters_base": args.base_iter,
            "iters_long": args.base_iter * args.long_mult,
            "l1_base": l1a, "l1_long": l1b,
            "rel_floor": (l1a - l1b) / l1b if l1b else None,
            "gap_base": float(a["gap"].max()), "gap_long": float(b["gap"].max()),
            "r_pri_base": a["r_pri"], "r_pri_long": b["r_pri"],
            "seconds": time.time() - t0,
        })
        q = rows[-1]
        print(f"m={m} p={p} P={q['P_dyn']:5d}  L1 {l1a:.6f} -> {l1b:.6f}  "
              f"rel_floor={q['rel_floor']:.2e}  gap {q['gap_base']:.1e} -> "
              f"{q['gap_long']:.1e}  {q['seconds']:.0f}s", flush=True)
        del f
        torch.cuda.empty_cache()

    doc = {"n_lut": len(ids), "c_star": args.c_star, "rows": rows,
           "worst_rel_floor": max((q["rel_floor"] or 0.0) for q in rows)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "solver_audit.json").write_text(
        json.dumps(doc, indent=1, sort_keys=True))
    print(f"\nworst relative solver floor: {doc['worst_rel_floor']:.2e}")


if __name__ == "__main__":
    main()
