"""P-W3 raw arm: head scan, the preregistered criteria, and the learnability probe.

Reads the exported per-head fields and produces ``metrics.json``.  The criteria
are the ones registered in the task card, evaluated **on the OOF fold only**,
**on the local subset only**, with head selection done **on the fit fold only**:

* **P1** median grid soft-IoU of the top-8 convex field  (report item, 0.45 line)
* **P2** paired Δ vs the zero-parameter centre prior     (>= +0.10, p < 0.01)
* **P3** paired Δ vs the shuffled-instruction arm        (>= +0.08, p < 0.01)

P2 and P3 are corrected across the three query pools with a Westfall-Young
max-statistic sign-flip permutation, so "one pool out of three cleared the bar"
cannot be bought by looking three times.

No AUC is computed anywhere in this file, by construction.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PRIMARY_POOLS = ("where_special", "where_content", "instr_text")
P2_GATE, P3_GATE, P_ALPHA, P1_REF = 0.10, 0.08, 0.01, 0.45


def _agg(vals) -> dict[str, Any] | None:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if not a.size:
        return None
    return {
        "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
        "frac_below_0.2": float((a < 0.2).mean()),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sink-k", type=float, default=3.0)
    ap.add_argument("--top-k-heads", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260810)
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--skip-learnable", action="store_true")
    args = ap.parse_args(argv)

    from q3vl.whereb.attnread import sink_mask_conjunctive, sink_mask_from_profile
    from q3vl.whereb.attnprobe import (
        EXCLUDE_FIRST_LAYERS, GatedLinearHead, HeadStackCNN, combine_heads,
        field_scores, head_norm_constants, max_stat_fwer, paired_wilcoxon,
        train_learnable_head,
    )
    from q3vl.whereb.metrics import center_prior_field, grid_boundary_f1, hard_iou, topk_mask

    exp = Path(args.export)
    rows = [json.loads(l) for l in (exp / "samples.jsonl").read_text().splitlines() if l.strip()]
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_sample[r["sample_id"]][r["arm"]] = r

    # only samples that have BOTH arms can enter the paired P3 test
    usable = sorted(s for s, a in by_sample.items() if "gt" in a and "shuffled" in a)
    gt_only = sorted(s for s, a in by_sample.items() if "gt" in a and "shuffled" not in a)
    print(f"samples with both arms: {len(usable)}  (gt-only, excluded: {len(gt_only)})")

    geom = {}
    if args.geometry and Path(args.geometry).exists():
        geom = json.load(open(args.geometry)).get("geometry", {})

    # ---- load everything once -------------------------------------------
    cache: dict[str, dict[str, Any]] = {}
    for sid in usable:
        meta = by_sample[sid]["gt"]
        gh, gw = meta["grid_h"], meta["grid_w"]
        dg = np.load(exp / "fields" / f"{sid}__gt.npz")
        dsh = np.load(exp / "fields" / f"{sid}__shuffled.npz")
        pg = dg["col_profile"].astype(np.float64).mean(axis=(0, 1))
        psh = dsh["col_profile"].astype(np.float64).mean(axis=(0, 1))
        sink = sink_mask_conjunctive(pg, psh, k=args.sink_k)
        valid = ~sink
        gt = np.load(exp / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)
        if valid.sum() < 4 or (gt[valid] > 0.5).sum() < 1:
            continue
        cache[sid] = {
            "fold": meta["fold"], "gh": gh, "gw": gw, "n_img": gh * gw,
            "valid": valid, "gt": gt,
            "gt_arrays": {p: dg[f"field_{p}"].astype(np.float64) for p in PRIMARY_POOLS
                          if f"field_{p}" in dg},
            "sh_arrays": {p: dsh[f"field_{p}"].astype(np.float64) for p in PRIMARY_POOLS
                          if f"field_{p}" in dsh},
            "sink_frac": float(sink.mean()),
            "region": meta.get("region"),
            "source_image_id": meta.get("source_image_id"),
        }
    fit = [s for s in cache if cache[s]["fold"] == "fit"]
    oof = [s for s in cache if cache[s]["fold"] == "oof"]
    print(f"usable local samples: fit {len(fit)}, oof {len(oof)}")

    n_layers, n_heads = next(iter(cache.values()))["gt_arrays"][PRIMARY_POOLS[0]].shape[:2]
    layer_ok = np.zeros(n_layers, dtype=bool)
    layer_ok[EXCLUDE_FIRST_LAYERS:] = True

    results: dict[str, Any] = {
        "card": "PR-ATT1-E1 / P-W3 raw arm",
        "export": str(exp),
        "n_fit": len(fit), "n_oof": len(oof),
        "n_excluded_no_shuffle_partner": len(gt_only),
        "sink_k": args.sink_k,
        "sink_rule": "conjunctive: above median+k*MAD under BOTH gt and shuffled",
        "exclude_first_layers": EXCLUDE_FIRST_LAYERS,
        "top_k_heads": args.top_k_heads,
        "grid": {"n_layers": n_layers, "n_heads": n_heads},
        "sink_frac": _agg([cache[s]["sink_frac"] for s in cache]),
        "pools": {},
    }

    # ---- centre prior + oracle ceiling, per sample -----------------------
    prior_cache, ceiling = {}, {}
    for sid, c in cache.items():
        pf = center_prior_field(c["gh"], c["gw"]).numpy().astype(np.float64).reshape(-1)
        prior_cache[sid] = pf
        gtv, v = c["gt"], c["valid"]
        kk = max(1, min(int((gtv[v] > 0.5).sum()), int(v.sum())))
        o = np.zeros(c["n_img"])
        cand = np.flatnonzero(v)
        o[cand[np.argsort(-gtv[v])[:kk]]] = 1.0
        ceiling[sid] = float(np.minimum(o, gtv).sum() / (np.maximum(o, gtv).sum() + 1e-8))
    results["oracle_ceiling_under_valid_mask"] = _agg(list(ceiling.values()))

    prior_oof = [float(field_scores(prior_cache[s][None, None, :], cache[s]["gt"],
                                    cache[s]["valid"])[0, 0]) for s in oof]
    results["center_prior_oof"] = _agg(prior_oof)

    # ---- per-pool: scan on fit, evaluate on OOF --------------------------
    diffs_p2: dict[str, np.ndarray] = {}
    diffs_p3: dict[str, np.ndarray] = {}

    for pool in PRIMARY_POOLS:
        if pool not in next(iter(cache.values()))["gt_arrays"]:
            continue
        # --- fit fold: per-head median grid soft-IoU
        per_head = np.zeros((len(fit), n_layers, n_heads))
        for i, sid in enumerate(fit):
            c = cache[sid]
            per_head[i] = field_scores(c["gt_arrays"][pool], c["gt"], c["valid"])
        fit_median = np.median(per_head, axis=0)
        masked = np.where(layer_ok[:, None], fit_median, -np.inf)
        flat = masked.reshape(-1)
        order = np.argsort(-flat)[: args.top_k_heads]
        idx = [(int(o // n_heads), int(o % n_heads)) for o in order]
        weights = [float(masked[l, h]) for l, h in idx]

        mu, sigma = head_norm_constants(
            [cache[s]["gt_arrays"][pool] for s in fit], [cache[s]["n_img"] for s in fit]
        )

        def combined(sid: str, arm: str) -> np.ndarray:
            c = cache[sid]
            arr = c["gt_arrays" if arm == "gt" else "sh_arrays"][pool]
            return combine_heads(arr, idx, weights, mu, sigma, c["n_img"])

        oof_gt, oof_sh, oof_prior, oof_bf1, oof_pbf1, oof_hard = [], [], [], [], [], []
        per_sample_rows = []
        for sid in oof:
            c = cache[sid]
            f_gt = combined(sid, "gt")
            f_sh = combined(sid, "shuffled")
            s_gt = float(field_scores(f_gt[None, None, :], c["gt"], c["valid"])[0, 0])
            s_sh = float(field_scores(f_sh[None, None, :], c["gt"], c["valid"])[0, 0])
            s_pr = float(field_scores(prior_cache[sid][None, None, :], c["gt"],
                                      c["valid"])[0, 0])
            oof_gt.append(s_gt); oof_sh.append(s_sh); oof_prior.append(s_pr)

            # boundary F1 + hard IoU on the same top-k masks, 2D
            gh, gw, v = c["gh"], c["gw"], c["valid"]
            kk = max(1, min(int((c["gt"][v] > 0.5).sum()), int(v.sum())))

            def mask2d(f):
                z = np.where(v, f, -np.inf)
                m = np.zeros(c["n_img"])
                m[np.argsort(-z)[:kk]] = 1.0
                return torch.from_numpy(m.reshape(gh, gw))

            gtb = torch.from_numpy(((c["gt"] > 0.5) & v).astype(np.float64).reshape(gh, gw))
            mg, mp = mask2d(f_gt), mask2d(prior_cache[sid])
            oof_bf1.append(grid_boundary_f1(mg, gtb))
            oof_pbf1.append(grid_boundary_f1(mp, gtb))
            oof_hard.append(hard_iou(mg, gtb))
            g = geom.get(sid, {})
            per_sample_rows.append({
                "sample_id": sid, "pool": pool,
                "grid_soft_iou_gt": s_gt, "grid_soft_iou_shuffled": s_sh,
                "grid_soft_iou_center_prior": s_pr,
                "grid_hard_iou_gt": oof_hard[-1],
                "grid_boundary_f1_gt": oof_bf1[-1],
                "grid_boundary_f1_center_prior": oof_pbf1[-1],
                "oracle_ceiling": ceiling[sid],
                "area_frac": g.get("area_frac"), "centroid_dist": g.get("centroid_dist"),
                "region": c["region"],
            })

        a_gt = np.asarray(oof_gt); a_sh = np.asarray(oof_sh); a_pr = np.asarray(oof_prior)
        diffs_p2[pool] = a_gt - a_pr
        diffs_p3[pool] = a_gt - a_sh

        results["pools"][pool] = {
            "selected_heads": [{"layer": l, "head": h, "fit_median_soft_iou": w}
                               for (l, h), w in zip(idx, weights)],
            "fit_best_head_median": float(masked.max()),
            "P1_oof_grid_soft_iou": _agg(oof_gt),
            "shuffled_oof_grid_soft_iou": _agg(oof_sh),
            "center_prior_oof_grid_soft_iou": _agg(oof_prior),
            "grid_boundary_f1": _agg(oof_bf1),
            "grid_boundary_f1_center_prior": _agg(oof_pbf1),
            "grid_hard_iou": _agg(oof_hard),
            "P2_vs_center_prior": paired_wilcoxon(a_gt, a_pr),
            "P3_vs_shuffled": paired_wilcoxon(a_gt, a_sh),
            "boundary_f1_vs_center_prior": paired_wilcoxon(
                np.asarray(oof_bf1), np.asarray(oof_pbf1)),
            "per_sample": per_sample_rows,
        }
        print(f"  {pool:15s} P1={np.median(oof_gt):.4f} "
              f"prior={np.median(oof_prior):.4f} shuf={np.median(oof_sh):.4f} "
              f"dP2={np.median(a_gt-a_pr):+.4f} dP3={np.median(a_gt-a_sh):+.4f}")

    # ---- FWER across the three pools ------------------------------------
    results["P2_fwer"] = max_stat_fwer(diffs_p2, n_perm=args.n_perm, seed=args.seed)
    results["P3_fwer"] = max_stat_fwer(diffs_p3, n_perm=args.n_perm, seed=args.seed)

    verdicts = {}
    for pool in results["pools"]:
        p2 = results["pools"][pool]["P2_vs_center_prior"]
        p3 = results["pools"][pool]["P3_vs_shuffled"]
        p2f = results["P2_fwer"]["p_fwer"].get(pool)
        p3f = results["P3_fwer"]["p_fwer"].get(pool)
        ok2 = (p2["delta_median"] >= P2_GATE) and (p2f is not None and p2f < P_ALPHA)
        ok3 = (p3["delta_median"] >= P3_GATE) and (p3f is not None and p3f < P_ALPHA)
        verdicts[pool] = {
            "P2_delta_median": p2["delta_median"], "P2_p_raw": p2["p_value"],
            "P2_p_fwer": p2f, "P2_pass": bool(ok2),
            "P3_delta_median": p3["delta_median"], "P3_p_raw": p3["p_value"],
            "P3_p_fwer": p3f, "P3_pass": bool(ok3),
            "P1_median": results["pools"][pool]["P1_oof_grid_soft_iou"]["median"],
            "P1_above_reference": bool(
                results["pools"][pool]["P1_oof_grid_soft_iou"]["median"] >= P1_REF),
            "promoted": bool(ok2 and ok3),
        }
    results["verdicts"] = verdicts
    results["gates"] = {"P2": P2_GATE, "P3": P3_GATE, "alpha": P_ALPHA,
                        "P1_reference_only": P1_REF}
    results["any_pool_promoted"] = any(v["promoted"] for v in verdicts.values())

    # ---- learnability up-probe ------------------------------------------
    if not args.skip_learnable:
        results["learnability"] = {}
        pool = "where_content" if "where_content" in results["pools"] else PRIMARY_POOLS[0]
        mu, sigma = head_norm_constants(
            [cache[s]["gt_arrays"][pool] for s in fit], [cache[s]["n_img"] for s in fit])

        def zstack(sid, arm="gt"):
            c = cache[sid]
            a = c["gt_arrays" if arm == "gt" else "sh_arrays"][pool] * c["n_img"]
            return torch.from_numpy(((a - mu[..., None]) / sigma[..., None])).float()

        fit_items = [(zstack(s), torch.from_numpy(cache[s]["gt"]).float(),
                      torch.from_numpy(cache[s]["valid"])) for s in fit]

        gl = GatedLinearHead(n_layers, n_heads)
        info = train_learnable_head(gl, fit_items, epochs=25, lr=5e-2, seed=args.seed)
        scores, scores_sh = [], []
        with torch.no_grad():
            for sid in oof:
                c = cache[sid]
                for arm, sink_list in (("gt", scores), ("shuffled", scores_sh)):
                    f = gl(zstack(sid, arm)).numpy().astype(np.float64)
                    sink_list.append(
                        float(field_scores(f[None, None, :], c["gt"], c["valid"])[0, 0]))
        results["learnability"]["gated_linear"] = {
            **info, "oof_grid_soft_iou": _agg(scores),
            "oof_grid_soft_iou_shuffled": _agg(scores_sh),
            "P2_vs_center_prior": paired_wilcoxon(np.asarray(scores), np.asarray(prior_oof)),
            # THE control: a head trained on GT masks can learn a generic blob and
            # score well without reading the instruction at all (finding four).
            # Delta_shuffle is the only thing that separates the two stories.
            "P3_vs_shuffled": paired_wilcoxon(np.asarray(scores), np.asarray(scores_sh)),
        }
        print(f"  gated_linear ({info['n_params']} params): "
              f"OOF median {np.median(scores):.4f} "
              f"(shuffled {np.median(scores_sh):.4f}, "
              f"d_shuf {np.median(np.asarray(scores)-np.asarray(scores_sh)):+.4f})")

        # top-128 heads by fit median, stacked as channels
        per_head = np.zeros((len(fit), n_layers, n_heads))
        for i, sid in enumerate(fit):
            c = cache[sid]
            per_head[i] = field_scores(c["gt_arrays"][pool], c["gt"], c["valid"])
        fm = np.where(layer_ok[:, None], np.median(per_head, axis=0), -np.inf)
        top128 = np.argsort(-fm.reshape(-1))[:128]
        sel = [(int(o // n_heads), int(o % n_heads)) for o in top128]

        def cnn_in(sid, arm="gt"):
            c = cache[sid]
            z = zstack(sid, arm).numpy()
            st = np.stack([z[l, h].reshape(c["gh"], c["gw"]) for l, h in sel])
            return torch.from_numpy(st).float()

        cnn_fit = [(cnn_in(s), torch.from_numpy(
            cache[s]["gt"].reshape(cache[s]["gh"], cache[s]["gw"])).float(),
            torch.from_numpy(cache[s]["valid"].reshape(cache[s]["gh"], cache[s]["gw"])))
            for s in fit]
        cnn = HeadStackCNN(128, 64, 32)
        info2 = train_learnable_head(cnn, cnn_fit, epochs=25, lr=3e-3, seed=args.seed)
        scores2, scores2_sh = [], []
        with torch.no_grad():
            for sid in oof:
                c = cache[sid]
                for arm, sink_list in (("gt", scores2), ("shuffled", scores2_sh)):
                    f = cnn(cnn_in(sid, arm)).numpy().astype(np.float64).reshape(-1)
                    sink_list.append(
                        float(field_scores(f[None, None, :], c["gt"], c["valid"])[0, 0]))
        results["learnability"]["head_stack_cnn"] = {
            **info2, "n_heads_stacked": len(sel), "oof_grid_soft_iou": _agg(scores2),
            "oof_grid_soft_iou_shuffled": _agg(scores2_sh),
            "P2_vs_center_prior": paired_wilcoxon(np.asarray(scores2), np.asarray(prior_oof)),
            "P3_vs_shuffled": paired_wilcoxon(np.asarray(scores2), np.asarray(scores2_sh)),
        }
        print(f"  head_stack_cnn ({info2['n_params']} params): "
              f"OOF median {np.median(scores2):.4f} "
              f"(shuffled {np.median(scores2_sh):.4f}, "
              f"d_shuf {np.median(np.asarray(scores2)-np.asarray(scores2_sh)):+.4f})")

    # ---- stratification --------------------------------------------------
    best = max(results["pools"], key=lambda p:
               results["pools"][p]["P1_oof_grid_soft_iou"]["median"])
    results["best_pool"] = best
    strata: dict[str, Any] = {}
    prs = results["pools"][best]["per_sample"]
    for name, key, cuts in [("area_frac", "area_frac", [0.05, 0.15, 0.45]),
                            ("centroid_dist", "centroid_dist", [0.1, 0.2, 0.3])]:
        buckets: dict[str, list] = defaultdict(list)
        for r in prs:
            v = r.get(key)
            if v is None:
                buckets["unknown"].append(r)
                continue
            lab = f"<{cuts[0]}"
            for i, c in enumerate(cuts):
                if v >= c:
                    lab = f">={c}" if i == len(cuts) - 1 else f"[{c},{cuts[i+1]})"
            buckets[lab].append(r)
        strata[name] = {
            lab: {"n": len(rs),
                  "gt": _agg([r["grid_soft_iou_gt"] for r in rs]),
                  "center_prior": _agg([r["grid_soft_iou_center_prior"] for r in rs]),
                  "delta_median": float(np.median(
                      [r["grid_soft_iou_gt"] - r["grid_soft_iou_center_prior"]
                       for r in rs])) if rs else None}
            for lab, rs in sorted(buckets.items())
        }
    results["strata"] = strata

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== verdicts ===")
    for p, v in verdicts.items():
        print(f"  {p:15s} P1={v['P1_median']:.4f} "
              f"P2 Δ={v['P2_delta_median']:+.4f} p_fwer={v['P2_p_fwer']:.2g} {'PASS' if v['P2_pass'] else 'FAIL'} | "
              f"P3 Δ={v['P3_delta_median']:+.4f} p_fwer={v['P3_p_fwer']:.2g} {'PASS' if v['P3_pass'] else 'FAIL'}")
    print(f"\nany pool promoted: {results['any_pool_promoted']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
