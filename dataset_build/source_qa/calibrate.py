"""Distribution-driven, PER-CORPUS IAA thresholds.

Hardcoded global cutoffs over-drop the hard corpora and under-drop the easy ones
(tad66k vs ppr10k vs greysky have very different baseline IAA distributions). This
computes, per (corpus, metric), the empirical percentiles and derives:
  * drop_value : the bad-tail cutoff at `drop_pctile` (default 10) — a value past it
                 (in the metric's bad direction) casts a "bad" vote.
  * keep_value : p50 — a value on the good side casts a "keep" vote.
Writes them to gate_thresholds (+ a global '*' row as fallback). `gate.py --mode
relative` then votes across metrics per corpus instead of using fixed numbers.

Run: python -m dataset_build.source_qa.calibrate [--drop-pctile 10] [--min-n 200]
"""
from __future__ import annotations

import argparse
import sys
from typing import Dict, List

from . import config, db


def _pct(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return float("nan")
    n = len(sorted_vals)
    i = min(n - 1, max(0, int(round(p / 100.0 * (n - 1)))))
    return float(sorted_vals[i])


METRICS = ["iaa_mixed", "artimuse", "charm"]


def run(drop_pctile: float = 10.0, min_n: int = 200) -> dict:
    conn = db.connect()          # gate_thresholds is created by db.init_db()
    run_id = db.start_run(conn, "calibrate", {"drop_pctile": drop_pctile, "min_n": min_n})

    # gather values per (corpus, metric) and global
    rows = conn.execute(
        "SELECT a.corpus AS corpus, s.metric AS metric, s.value AS value "
        "FROM iqa_scores s JOIN assets a ON a.asset_id=s.asset_id "
        "WHERE a.asset_type='image' AND s.value IS NOT NULL AND s.metric IN "
        f"({','.join(['?'] * len(METRICS))})",
        METRICS,
    ).fetchall()
    buckets: Dict[tuple, List[float]] = {}
    for r in rows:
        buckets.setdefault((r["corpus"], r["metric"]), []).append(r["value"])
        buckets.setdefault(("*", r["metric"]), []).append(r["value"])

    written = skipped = 0
    summary = {}
    conn.execute("DELETE FROM gate_thresholds")
    for (corpus, metric), vals in sorted(buckets.items()):
        # min_n now actually enforced: a per-corpus bucket too small to estimate a
        # stable p10 tail is skipped (the relative gate falls back to the '*' row).
        if corpus != "*" and len(vals) < min_n:
            skipped += 1
            continue
        vals.sort()
        n = len(vals)
        higher = 1 if config.IQA_HIGHER_BETTER.get(metric, True) else 0
        p = {q: _pct(vals, q) for q in (2, 5, 10, 25, 50, 75, 90, 95)}
        if higher:
            drop_value = _pct(vals, drop_pctile)          # below this -> bad
        else:
            drop_value = _pct(vals, 100 - drop_pctile)    # above this -> bad
        keep_value = p[50]
        conn.execute(
            "INSERT INTO gate_thresholds(corpus,metric,direction,n,drop_pctile,drop_value,"
            "keep_value,p02,p05,p10,p25,p50,p75,p90,p95,updated_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,now()) "
            "ON CONFLICT(corpus,metric) DO UPDATE SET direction=EXCLUDED.direction,"
            "n=EXCLUDED.n,drop_pctile=EXCLUDED.drop_pctile,drop_value=EXCLUDED.drop_value,"
            "keep_value=EXCLUDED.keep_value,p02=EXCLUDED.p02,p05=EXCLUDED.p05,p10=EXCLUDED.p10,"
            "p25=EXCLUDED.p25,p50=EXCLUDED.p50,p75=EXCLUDED.p75,p90=EXCLUDED.p90,"
            "p95=EXCLUDED.p95,updated_at=now()",
            (corpus, metric, higher, n, drop_pctile, drop_value, keep_value,
             p[2], p[5], p[10], p[25], p[50], p[75], p[90], p[95]))
        written += 1
        if corpus == "*":
            summary[metric] = {"n": n, "drop_value": round(drop_value, 3),
                               "keep_value": round(keep_value, 3), "dir": "↑" if higher else "↓"}
    conn.commit()
    db.finish_run(conn, run_id, {"written": written, "skipped": skipped, "global": summary})
    conn.close()
    print(f"[calibrate] wrote {written} thresholds (skipped {skipped} small buckets, min_n={min_n}) "
          f"@ drop_pctile={drop_pctile}", file=sys.stderr)
    import json
    print(json.dumps({"global": summary}, ensure_ascii=False, indent=2))
    return {"written": written, "global": summary}


def load_thresholds(conn) -> Dict[str, Dict[str, dict]]:
    """{corpus: {metric: {direction, drop_value, keep_value, ...}}}"""
    out: Dict[str, Dict[str, dict]] = {}
    for r in conn.execute("SELECT * FROM gate_thresholds"):
        out.setdefault(r["corpus"], {})[r["metric"]] = dict(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-pctile", type=float, default=10.0)
    ap.add_argument("--min-n", type=int, default=200)
    args = ap.parse_args()
    run(drop_pctile=args.drop_pctile, min_n=args.min_n)


if __name__ == "__main__":
    main()
