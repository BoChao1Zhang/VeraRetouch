"""Cross-run paired deltas on the normal-only headline column.

The landed metrics.json files carry a paired delta against the CENTRE PRIOR
only; comparing one arm to another has to be redone from per_sample.jsonl,
matched by sample_id.  Column: `hard_iou` == the matched-area top-k IoU the
boards call `topk_iou` (verified identical), context `generated`,
winner_confidence == "normal".
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "/home/bc/VeraRetouch")
from q3vl.whereb.metrics import paired_delta

RUNS = Path("/home/bc/data/runs/where_b")


def rows(run, ctx="generated"):
    out = {}
    p = RUNS / run / "eval_final" / "per_sample.jsonl"
    for line in p.open():
        r = json.loads(line)
        if r.get("mode") != ctx or r.get("uncovered") or r.get("is_fake"):
            continue
        if r.get("winner_confidence") != "normal":
            continue
        out[r["sample_id"]] = r
    return out


def delta(a_run, b_run, col="hard_iou"):
    a, b = rows(a_run), rows(b_run)
    ids = sorted(set(a) & set(b))
    d = paired_delta([a[i][col] for i in ids], [b[i][col] for i in ids])
    return {"n_pairs": len(ids), "delta": d["delta"], "p": d["p_value"],
            "ci": d.get("ci95")}


PAIRS = [
    # (label, arm, baseline)
    ("CONT vs P3'(1200)  [published +0.0254]", "amort_P3prime_cont_20260811",
     "amort_P3prime_20260810"),
    ("P1 vs POOLED       [published +0.0856]", "amort_P1_20260810",
     "amort_P1_pooled_20260811"),
    ("CONT2 vs CONT", "amort_P3prime_cont2_20260811", "amort_P3prime_cont_20260811"),
    ("CONT2 vs P3'(1200)", "amort_P3prime_cont2_20260811", "amort_P3prime_20260810"),
    ("SHAPE3_A vs SHAPE3_B", "amort_SHAPE3_A_20260811", "amort_SHAPE3_B_20260811"),
    ("SHAPE3_A vs P3'(1200)", "amort_SHAPE3_A_20260811", "amort_P3prime_20260810"),
    ("SHAPE3_B vs P3'(1200)", "amort_SHAPE3_B_20260811", "amort_P3prime_20260810"),
    ("p2struct(.05) vs P3'(1200)", "amort_P3prime_p2struct_20260811",
     "amort_P3prime_20260810"),
    ("p2struct_hi(.15) vs P3'(1200)", "amort_P3prime_p2struct_hi_20260811",
     "amort_P3prime_20260810"),
    ("p2w(5x) vs P3'(1200)", "amort_P3prime_p2w_20260811", "amort_P3prime_20260810"),
    ("p2w_hi(10x) vs P3'(1200)", "amort_P3prime_p2w_hi_20260811",
     "amort_P3prime_20260810"),
    ("p2struct(.05) vs CONT", "amort_P3prime_p2struct_20260811",
     "amort_P3prime_cont_20260811"),
]

if __name__ == "__main__":
    for label, a, b in PAIRS:
        try:
            d = delta(a, b)
            print(f"{label:42s} n={d['n_pairs']:3d}  delta={d['delta']:+.4f}  "
                  f"p={d['p']:.2e}")
        except Exception as exc:                  # noqa: BLE001
            print(f"{label:42s} FAILED {exc}")
    # boundary F1 on the same pairing, for the shape column
    print()
    for label, a, b in PAIRS[2:7]:
        d = delta(a, b, col="grid_boundary_f1")
        print(f"bF1 {label:38s} n={d['n_pairs']:3d}  delta={d['delta']:+.4f}  "
              f"p={d['p']:.2e}")
