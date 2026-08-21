"""EPR-032 §8 criteria, wired as a runnable gate over the ladder artefacts.

    .venv-lens/bin/python -m q3vl.whatb.jetlut.gate --out DIR

Exits non-zero when G-wire fails.  G-wire is a *wiring* assertion, not a result:
p=1 is a strict subspace of p=2 on the same atlas and the same fit colours, so
the L1 optimum can only go down.  A violation means the solver did not converge,
the design matrix is miswired, or the two orders were fitted under different
conventions -- never "p2 is worse".

Deliberately a separate module from ``run.py``: the ladder jobs are already
running and the campaign rule is that source is frozen once a process is up.
The criteria live on the artefacts, which is where a board is read from anyway.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

#: PROPOSAL §8.1 A-nested; the tolerance is bound to the measured solver primal
#: accuracy (rho=100 / 2000 iters -> ~1e-6 relative, two orders of headroom).
NESTED_TOL = 1e-4
#: PROPOSAL §8.2 G-main
G_MAIN_DROP = 0.20
G_MAIN_MIN_POINTS = 3
#: PROPOSAL §4.1 as revised by the measured rho scan: the certificate converges
#: an order of magnitude slower than the primal, so this is telemetry, not a gate.
GAP_REPORT_THRESHOLD = 1e-2


def _key(row: dict, field: str = "p95", clamp: str = "pre_clamp",
         grid: str = "fitgrid") -> float:
    return float(row[f"{grid}_{clamp}"][field])


def check_nested(rows: list[dict]) -> list[dict]:
    """A-nested / G-wire: same m, same fit set, pre-clamp -> E*_p2 <= E*_p1."""
    by = {(r["m"], r["p"]): r for r in rows}
    out = []
    for (m, p), r in sorted(by.items()):
        if p != 2 or (m, 1) not in by:
            continue
        e1 = by[(m, 1)]["l1_fit_mean_per_point"]
        e2 = r["l1_fit_mean_per_point"]
        out.append({"m": m, "l1_p1": e1, "l1_p2": e2,
                    "ratio": e2 / e1 if e1 else math.nan,
                    "ok": bool(e2 <= e1 * (1.0 + NESTED_TOL))})
    return out


def _interp_log(points: list[tuple[float, float]], x: float) -> float | None:
    """Linear interpolation of ``y`` on ``log P``.  None outside the hull."""
    pts = sorted(points)
    if not pts or x < pts[0][0] or x > pts[-1][0]:
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))
            return y0 + t * (y1 - y0)
    return None


def check_g_main(rows: list[dict], field: str = "p95", grid: str = "fitgrid"
                 ) -> dict:
    """G-main: relative drop of p2 vs log-P-interpolated p1, on shared budgets."""
    p1 = [(r["P_dyn"], _key(r, field, grid=grid)) for r in rows if r["p"] == 1]
    p2 = [(r["P_dyn"], _key(r, field, grid=grid)) for r in rows if r["p"] == 2]
    pts = []
    for pdyn, y2 in sorted(p2):
        y1 = _interp_log(p1, pdyn)
        if y1 is None:
            continue
        pts.append({"P_dyn": pdyn, "p1_interp": y1, "p2": y2,
                    "rel_drop": (y1 - y2) / y1 if y1 else math.nan})
    n_pass = sum(1 for q in pts if q["rel_drop"] >= G_MAIN_DROP)
    same_sign = all(q["rel_drop"] > 0 for q in pts) if pts else False
    return {"field": field, "grid": grid, "points": pts,
            "n_points_in_hull": len(pts), "n_points_ge_threshold": n_pass,
            "all_same_sign": same_sign,
            "threshold": G_MAIN_DROP, "min_points": G_MAIN_MIN_POINTS,
            "PASS": bool(n_pass >= G_MAIN_MIN_POINTS and same_sign)}


def check_g_shape(rows: list[dict]) -> dict:
    """G-shape: is the p2 gain concentrated in the theory-predicted strata?

    Reported per stratum as the relative p95 drop at each matched m, so a gain
    spread evenly over the whole cube is visible as such rather than hidden.
    """
    by = {(r["m"], r["p"]): r for r in rows}
    out: dict[str, list[dict]] = {}
    for (m, p), r in sorted(by.items()):
        if p != 2 or (m, 1) not in by or "fitgrid_strata" not in r:
            continue
        s1 = by[(m, 1)].get("fitgrid_strata", {})
        for name, st2 in r["fitgrid_strata"].items():
            if name not in s1:
                continue
            a, b = float(s1[name]["p95"]), float(st2["p95"])
            out.setdefault(name, []).append(
                {"m": m, "p1_p95": a, "p2_p95": b,
                 "rel_drop": (a - b) / a if a else math.nan})
    return out


def solver_health(rows: list[dict]) -> dict:
    worst = max(rows, key=lambda r: r["gap_max"]) if rows else {}
    return {
        "n_rows": len(rows),
        "gap_max_over_rows": max((r["gap_max"] for r in rows), default=None),
        "gap_max_row": {k: worst.get(k) for k in ("m", "p", "P_dyn")},
        "n_rows_gap_above_threshold": sum(
            1 for r in rows if r["gap_max"] > GAP_REPORT_THRESHOLD),
        "gap_report_threshold": GAP_REPORT_THRESHOLD,
        "dual_feas_max": max((r["dual_feas"] for r in rows), default=None),
        "r_pri_max": max((r["r_pri"] for r in rows), default=None),
        "tie_gap_rel": [r["tie_gap_rel"] for r in rows if "tie_gap_rel" in r],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/prs/"
                                     "EPR-032_jetlut-canonical-color-field/out")
    ap.add_argument("--pools", nargs="+",
                    default=["held_out", "t_lut_unseen", "train"])
    ap.add_argument("--tag", default="", help="artefact suffix, e.g. '_g17'")
    ap.add_argument("--report", default="gate.json")
    args = ap.parse_args()

    report: dict[str, dict] = {}
    failed = False
    for pool in args.pools:
        path = Path(args.out) / f"ladder_{pool}{args.tag}.json"
        if not path.exists():
            print(f"[skip] {path} absent")
            continue
        doc = json.loads(path.read_text())
        rows = doc["rows"]
        nested = check_nested(rows)
        rep = {
            "c_star": doc.get("c_star"), "n_lut": doc.get("n_lut"),
            "lut_ids_sha": doc.get("lut_ids_sha"),
            "G_wire_nested": nested,
            "G_wire_PASS": all(q["ok"] for q in nested) if nested else None,
            "G_main_p95_fitgrid": check_g_main(rows, "p95", "fitgrid"),
            "G_main_p95_heldgrid": check_g_main(rows, "p95", "heldgrid"),
            "G_main_p99_fitgrid": check_g_main(rows, "p99", "fitgrid"),
            "G_main_mean_fitgrid": check_g_main(rows, "mean", "fitgrid"),
            "G_shape": check_g_shape(rows),
            "solver": solver_health(rows),
        }
        report[pool] = rep

        print(f"\n=== {pool}  (c*={rep['c_star']}, n_lut={rep['n_lut']}) ===")
        for q in nested:
            flag = "ok " if q["ok"] else "FAIL"
            print(f"  G-wire m={q['m']}: L1/pt p1={q['l1_p1']:.6f} "
                  f"p2={q['l1_p2']:.6f} ratio={q['ratio']:.6f}  {flag}")
        if nested and not all(q["ok"] for q in nested):
            failed = True
        for name in ("G_main_p95_fitgrid", "G_main_p95_heldgrid"):
            g = rep[name]
            print(f"  {name}: PASS={g['PASS']} "
                  f"({g['n_points_ge_threshold']}/{g['n_points_in_hull']} "
                  f">= {g['threshold']:.0%}, same_sign={g['all_same_sign']})")
            for q in g["points"]:
                print(f"    P={q['P_dyn']:5d}  p1_interp={q['p1_interp']:.4f} "
                      f"p2={q['p2']:.4f}  drop={q['rel_drop']:+.2%}")
        s = rep["solver"]
        print(f"  solver: gap_max={s['gap_max_over_rows']:.2e} at "
              f"{s['gap_max_row']}, rows above {GAP_REPORT_THRESHOLD:g}: "
              f"{s['n_rows_gap_above_threshold']}/{s['n_rows']}, "
              f"dual_feas_max={s['dual_feas_max']:.1e}, "
              f"r_pri_max={s['r_pri_max']:.1e}, tie={s['tie_gap_rel']}")

    (Path(args.out) / args.report).write_text(
        json.dumps(report, indent=1, sort_keys=True))
    if failed:
        print("\nG-wire VIOLATED -- nested subspace inequality broken; this is a "
              "solver/wiring failure, not a result (PROPOSAL §8.1).",
              file=sys.stderr)
        sys.exit(1)
    print(f"\n{args.report} written")


if __name__ == "__main__":
    main()
