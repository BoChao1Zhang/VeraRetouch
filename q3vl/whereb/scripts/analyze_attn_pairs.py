"""B2: does the shuffle partner actually point somewhere else?

``Delta_shuffle ~= 0`` only means "the field ignores the instruction" **if the
partner instruction refers to a different region**.  The partner is drawn from
the same ``(source_image_id, render_mode)`` group -- the same photograph -- so if
group members share a target, a perfectly instruction-following field would also
score ``Delta_shuffle ~= 0``.  That is the one false-negative path the original
delivery left open (review blocker B2).

This closes it with no GPU: for every OOF pair it measures how far apart the two
GT regions actually are, then re-runs P3 on the subset where they are genuinely
separated.  If P3 stays at zero there, the partner-overlap explanation is dead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    from q3vl.whereb.attnprobe import paired_wilcoxon
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset

    exp = Path(args.export)
    metrics = json.loads(Path(args.metrics).read_text())

    ds, _ = open_dataset("V_where", need_mask=False)
    shuffle_rows = ds.shuffle_records()
    sidx = ShuffleIndex(shuffle_rows, seed=args.seed)

    def gt_of(sid: str) -> np.ndarray | None:
        p = exp / "gt" / f"{sid}.npy"
        return np.load(p).astype(np.float64) if p.exists() else None

    def centroid(g: np.ndarray) -> tuple[float, float] | None:
        tot = g.sum()
        if tot <= 0:
            return None
        ys, xs = np.mgrid[0:g.shape[0], 0:g.shape[1]]
        # normalised so grids of different shapes are comparable
        return (float((g * ys).sum() / tot / max(g.shape[0] - 1, 1)),
                float((g * xs).sum() / tot / max(g.shape[1] - 1, 1)))

    def soft_iou(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + 1e-8))

    pool_rows = {p: {r["sample_id"]: r for r in v["per_sample"]}
                 for p, v in metrics["pools"].items()}
    any_pool = next(iter(pool_rows.values()))

    pairs: list[dict[str, Any]] = []
    for sid in any_pool:
        partner = sidx.partner_of(sid)
        if partner is None:
            continue
        g_self, g_part = gt_of(sid), gt_of(partner)
        if g_self is None or g_part is None:
            continue
        if g_self.shape != g_part.shape:
            # same source image, so the grids should match; if they do not the
            # pair cannot be compared cell-wise and is reported separately
            pairs.append({"sample_id": sid, "partner": partner,
                          "shape_mismatch": True})
            continue
        c1, c2 = centroid(g_self), centroid(g_part)
        dist = (float(np.hypot(c1[0] - c2[0], c1[1] - c2[1]))
                if c1 and c2 else None)
        pairs.append({
            "sample_id": sid, "partner": partner, "shape_mismatch": False,
            "pair_gt_soft_iou": soft_iou(g_self, g_part),
            "pair_centroid_dist": dist,
            "self_area": float(g_self.mean()), "partner_area": float(g_part.mean()),
            "identical": bool(np.allclose(g_self, g_part, atol=1e-6)),
        })

    ok = [p for p in pairs if not p["shape_mismatch"]]
    ious = np.array([p["pair_gt_soft_iou"] for p in ok])
    dists = np.array([p["pair_centroid_dist"] for p in ok
                      if p["pair_centroid_dist"] is not None])

    def agg(a):
        a = np.asarray(a, dtype=np.float64)
        if not a.size:
            return None
        return {"n": int(a.size), "mean": float(a.mean()),
                "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                "p25": float(np.percentile(a, 25)), "p75": float(np.percentile(a, 75)),
                "p90": float(np.percentile(a, 90))}

    out: dict[str, Any] = {
        "card": "PR-ATT1-E1b / B2 -- shuffle partner separation",
        "question": ("is Delta_shuffle ~= 0 a false negative caused by the partner "
                     "pointing at the same region?"),
        "n_pairs": len(ok),
        "n_shape_mismatch": sum(1 for p in pairs if p["shape_mismatch"]),
        "n_identical_gt": int(sum(1 for p in ok if p["identical"])),
        "pair_gt_soft_iou": agg(ious),
        "pair_centroid_dist": agg(dists),
        "separation_buckets": {},
        "P3_by_separation": {},
    }

    # how many pairs are genuinely separated?
    for lo, hi in [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 1.01)]:
        sel = [p for p in ok if lo <= p["pair_gt_soft_iou"] < hi]
        out["separation_buckets"][f"[{lo},{hi})"] = len(sel)

    # THE test: re-run P3 on the separated subset, per pool
    for thresh in (0.3, 0.1):
        keep = {p["sample_id"] for p in ok if p["pair_gt_soft_iou"] < thresh}
        entry: dict[str, Any] = {"n_kept": len(keep), "rule": f"pair_gt_soft_iou < {thresh}"}
        for pool, rows in pool_rows.items():
            sub = [r for sid, r in rows.items() if sid in keep]
            if len(sub) < 10:
                entry[pool] = {"n": len(sub), "note": "too few pairs to test"}
                continue
            a = np.array([r["grid_soft_iou_gt"] for r in sub])
            b = np.array([r["grid_soft_iou_shuffled"] for r in sub])
            c = np.array([r["grid_soft_iou_center_prior"] for r in sub])
            entry[pool] = {
                "n": len(sub),
                "P3_vs_shuffled": paired_wilcoxon(a, b),
                "P2_vs_center_prior": paired_wilcoxon(a, c),
                "median_gt": float(np.median(a)), "median_shuffled": float(np.median(b)),
            }
        out["P3_by_separation"][f"iou_lt_{thresh}"] = entry

    out["pairs"] = ok
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"pairs: {len(ok)}  identical GT: {out['n_identical_gt']}")
    print(f"pair GT soft-IoU: median {out['pair_gt_soft_iou']['median']:.4f} "
          f"p25 {out['pair_gt_soft_iou']['p25']:.4f} p75 {out['pair_gt_soft_iou']['p75']:.4f}")
    print(f"pair centroid dist: median {out['pair_centroid_dist']['median']:.4f}")
    print(f"separation buckets: {out['separation_buckets']}")
    for k, e in out["P3_by_separation"].items():
        print(f"\n--- {k} (n={e['n_kept']})")
        for pool in pool_rows:
            v = e.get(pool, {})
            if "P3_vs_shuffled" in v:
                w = v["P3_vs_shuffled"]
                print(f"   {pool:15s} n={v['n']:3d} gt={v['median_gt']:.4f} "
                      f"shuf={v['median_shuffled']:.4f} "
                      f"D3={w['delta_median']:+.4f} p={w['p_value']:.3g}")
            else:
                print(f"   {pool:15s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
