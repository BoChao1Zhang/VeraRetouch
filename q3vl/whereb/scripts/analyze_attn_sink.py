"""P-W1: sink / artefact survey -> the whole-arm ``valid_mask`` rule.

The premise card, not a post-processing step (PROPOSAL 2.2 step 4, finding one).
Qwen3-VL is native-resolution, so there are no ``expand2square`` pad cells to
exclude -- but an attention sink is a different animal from a pad cell, and the
question of whether high-mass image cells exist and are instruction-invariant has
to be answered before any field is scored.

Three things are measured, all on the raw exported profiles:

1. **mass concentration** -- what fraction of the image-directed attention mass
   the sink cells carry, and how often the per-head argmax lands on one.  RO-9c's
   pad cells carried 53-74% of the mass and 93% of the argmax; the number here is
   the Qwen3-VL analogue.
2. **instruction invariance** -- the sink criterion's second half.  A cell that
   is high because the instruction points at it is signal; a cell that is high
   under a *shuffled* instruction too is a sink.  Measured as the Jaccard overlap
   of the sink sets between the ``gt`` and ``shuffled`` arms of the same image.
3. **cross-resolution stability** -- P-W1's completion standard.  Sink structure
   is compared across merged-grid shapes; the card says to bucket the rule by
   resolution if Jaccard < 0.8.

Because sink identity is a *position within one image*, "stability across
resolutions" cannot mean "the same cell index"; it is measured on the structural
descriptors that do transfer: sink count, mass share, and normalised position.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union else 1.0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=float, default=3.0, help="MAD multiplier")
    ap.add_argument("--k-grid", default="2.0,2.5,3.0,4.0,5.0")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    from q3vl.whereb.attnread import sink_mask_from_profile

    exp = Path(args.export)
    rows = [json.loads(l) for l in (exp / "samples.jsonl").read_text().splitlines() if l.strip()]
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_sample[r["sample_id"]][r["arm"]] = r
    sids = sorted(by_sample)
    if args.limit:
        sids = sids[: args.limit]

    ks = sorted({float(x) for x in args.k_grid.split(",")} | {float(args.k)})
    per_sample: list[dict[str, Any]] = []
    k_sweep: dict[float, list[float]] = {k: [] for k in ks}
    k_ceiling: dict[float, list[float]] = {k: [] for k in ks}
    k_gt_lost: dict[float, list[float]] = {k: [] for k in ks}
    k_argmax: dict[float, list[float]] = {k: [] for k in ks}
    k_jac: dict[float, list[float]] = {k: [] for k in ks}
    by_shape: dict[tuple[int, int], list[dict]] = defaultdict(list)

    for sid in sids:
        arms = by_sample[sid]
        if "gt" not in arms:
            continue
        d = np.load(exp / "fields" / f"{sid}__gt.npz")
        prof = d["col_profile"].astype(np.float64)          # (L, H, n_img)
        gh, gw = arms["gt"]["grid_h"], arms["gt"]["grid_w"]
        mean_prof = prof.mean(axis=(0, 1))                   # (n_img,)
        n_img = mean_prof.size

        sink = sink_mask_from_profile(mean_prof, k=args.k)
        # mass share carried by sink cells, over the raw (un-normalised) profile
        mass_share = float(mean_prof[sink].sum() / mean_prof.sum()) if sink.any() else 0.0
        # per (layer, head) argmax landing on a sink cell
        argmax_cells = prof.reshape(-1, n_img).argmax(axis=1)
        argmax_in_sink = float(sink[argmax_cells].mean())

        rec: dict[str, Any] = {
            "sample_id": sid, "grid": [gh, gw], "n_img": n_img,
            "n_sink": int(sink.sum()), "sink_frac": float(sink.mean()),
            "sink_mass_share": mass_share,
            "argmax_in_sink_rate": argmax_in_sink,
            "profile_max_over_median": float(mean_prof.max() / np.median(mean_prof)),
        }

        # --- what does excluding these cells COST? --------------------------
        # Excluding a cell removes it from the top-k support, so if sinks sit on
        # the target the rule silently caps the achievable IoU.  The oracle
        # ceiling below is the honest upper bound the criteria are measured
        # against: the best mask any field could produce under this valid mask.
        gt_path = exp / "gt" / f"{sid}.npy"
        if gt_path.exists():
            gt = np.load(gt_path).astype(np.float64).reshape(-1)
            for k in ks:
                sk = sink_mask_from_profile(mean_prof, k=k)
                vk = ~sk
                gt_lost = float(gt[sk].sum() / gt.sum()) if gt.sum() > 0 else 0.0
                kk = max(1, min(int((gt[vk] > 0.5).sum()), int(vk.sum())))
                oracle = np.zeros(n_img)
                cand = np.flatnonzero(vk)
                pick = cand[np.argsort(-gt[vk])[:kk]]
                oracle[pick] = 1.0
                inter = np.minimum(oracle, gt).sum()
                union = np.maximum(oracle, gt).sum()
                k_ceiling[k].append(float(inter / (union + 1e-8)))
                k_gt_lost[k].append(gt_lost)
                if k == args.k:
                    rec["gt_mass_in_sink"] = gt_lost
                    rec["oracle_ceiling_under_valid"] = float(inter / (union + 1e-8))
            # the same ceiling with NO exclusion, to price the rule
            k0 = max(1, int((gt > 0.5).sum()))
            o0 = np.zeros(n_img)
            o0[np.argsort(-gt)[:k0]] = 1.0
            rec["oracle_ceiling_no_exclusion"] = float(
                np.minimum(o0, gt).sum() / (np.maximum(o0, gt).sum() + 1e-8))

        for k in ks:
            sk = sink_mask_from_profile(mean_prof, k=k)
            k_sweep[k].append(float(sk.mean()))
            am = prof.reshape(-1, n_img).argmax(axis=1)
            k_argmax[k].append(float(sk[am].mean()))
            if "shuffled" in arms:
                ds2 = np.load(exp / "fields" / f"{sid}__shuffled.npz")
                ps2 = ds2["col_profile"].astype(np.float64).mean(axis=(0, 1))
                k_jac[k].append(_jaccard(sk, sink_mask_from_profile(ps2, k=k)))

        if sink.any():
            ys, xs = np.nonzero(sink.reshape(gh, gw))
            rec["sink_pos_norm"] = [[float(y / max(gh - 1, 1)), float(x / max(gw - 1, 1))]
                                    for y, x in zip(ys, xs)]
            rec["sink_first_cell"] = bool(sink[0])

        # instruction invariance: same sinks under a shuffled instruction?
        if "shuffled" in arms:
            ds = np.load(exp / "fields" / f"{sid}__shuffled.npz")
            ps = ds["col_profile"].astype(np.float64).mean(axis=(0, 1))
            sink_s = sink_mask_from_profile(ps, k=args.k)
            rec["sink_jaccard_gt_vs_shuffled"] = _jaccard(sink, sink_s)
            rec["n_sink_shuffled"] = int(sink_s.sum())
            order_gt = np.argsort(-mean_prof)[: max(int(sink.sum()), 5)]
            order_sh = np.argsort(-ps)[: max(int(sink.sum()), 5)]
            rec["top_cell_rank_overlap"] = float(
                len(set(order_gt.tolist()) & set(order_sh.tolist())) / len(order_gt)
            )
        per_sample.append(rec)
        by_shape[(gh, gw)].append(rec)

    def agg(vals: list[float]) -> dict[str, float] | None:
        a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
        if not a.size:
            return None
        return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
                "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
                "min": float(a.min()), "max": float(a.max())}

    shape_summary = {}
    for shape, recs in sorted(by_shape.items()):
        shape_summary[f"{shape[0]}x{shape[1]}"] = {
            "n_samples": len(recs),
            "sink_frac": agg([r["sink_frac"] for r in recs]),
            "sink_mass_share": agg([r["sink_mass_share"] for r in recs]),
            "argmax_in_sink_rate": agg([r["argmax_in_sink_rate"] for r in recs]),
            "sink_jaccard_gt_vs_shuffled": agg(
                [r.get("sink_jaccard_gt_vs_shuffled") for r in recs
                 if r.get("sink_jaccard_gt_vs_shuffled") is not None]),
        }

    # cross-resolution stability of the STRUCTURE (index sets cannot transfer)
    fracs = {s: v["sink_frac"]["median"] for s, v in shape_summary.items()
             if v["sink_frac"]}
    masses = {s: v["sink_mass_share"]["median"] for s, v in shape_summary.items()
              if v["sink_mass_share"]}
    jac = [r["sink_jaccard_gt_vs_shuffled"] for r in per_sample
           if r.get("sink_jaccard_gt_vs_shuffled") is not None]

    out = {
        "card": "P-W1",
        "export": str(exp),
        "n_samples": len(per_sample),
        "mad_k": args.k,
        "rule": ("valid_mask = NOT (mean-over-layers-and-heads column profile, "
                 "taken over post-image text query rows, > median + k*MAD). "
                 "Whole-arm constant rule; per-image threshold; no field value is "
                 "rescaled and no cell is interpolated."),
        "headline": {
            "sink_frac": agg([r["sink_frac"] for r in per_sample]),
            "sink_mass_share": agg([r["sink_mass_share"] for r in per_sample]),
            "argmax_in_sink_rate": agg([r["argmax_in_sink_rate"] for r in per_sample]),
            "profile_max_over_median": agg(
                [r["profile_max_over_median"] for r in per_sample]),
            "instruction_invariance_jaccard": agg(jac),
            "first_image_cell_is_sink_rate": float(np.mean(
                [r.get("sink_first_cell", False) for r in per_sample])),
        },
        "k_sweep": {
            str(k): {
                "sink_frac": agg(k_sweep[k]),
                "argmax_in_sink_rate": agg(k_argmax[k]),
                "instruction_invariance_jaccard": agg(k_jac[k]),
                "gt_mass_lost_to_sink": agg(k_gt_lost[k]),
                "oracle_ceiling_under_valid": agg(k_ceiling[k]),
            } for k in ks
        },
        "ceiling_cost": {
            "oracle_ceiling_no_exclusion": agg(
                [r["oracle_ceiling_no_exclusion"] for r in per_sample
                 if "oracle_ceiling_no_exclusion" in r]),
            "note": ("the price of the valid mask: an excluded cell cannot be "
                     "selected by the matched-area top-k, so GT mass sitting on a "
                     "sink is unreachable for every field including the oracle"),
        },
        "by_resolution": shape_summary,
        "cross_resolution_stability": {
            "sink_frac_by_shape": fracs,
            "sink_frac_spread": (max(fracs.values()) - min(fracs.values())) if fracs else None,
            "mass_share_by_shape": masses,
            "mass_share_spread": (max(masses.values()) - min(masses.values()))
            if masses else None,
        },
        "per_sample": per_sample,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    h = out["headline"]
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_sample", "by_resolution")}, indent=2))
    print(f"\nsink cells: median {h['sink_frac']['median']:.3%} of the grid, "
          f"carrying median {h['sink_mass_share']['median']:.1%} of the image mass; "
          f"argmax lands on one {h['argmax_in_sink_rate']['median']:.1%} of the time")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
