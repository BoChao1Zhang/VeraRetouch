"""E1b / B1: the fixed-phrase common-mode arm, P4, P5's guard column, random floor.

**Family status: OUT OF FAMILY.** Everything this script produces is a one-shot
falsification patch, run after the three-pool result was published.  Its p-values
are **uncorrected** and are deliberately *not* merged into the published
Westfall-Young null over ``{where_special, where_content, instr_text}`` -- adding
tests to a family after seeing its outcome is exactly the post-hoc family
expansion DELTA ruling 4 forbids.  Nothing here can promote the raw arm; it exists
to close the registered P4/P5 cells and to replace an inference ("instr - fixed is
about zero, because Delta_shuffle is about zero") with a measurement.

Four things get produced:

* **diff arm** ``D = S(instr) - S(fixed)`` through the same P1/P2/P3 machinery;
* **P4** paired ``instr`` vs ``fixed`` field, on the registered eccentric /
  small-target subsets;
* **P5 guard** the random top-k column the boundary-F1 row needs to be readable
  at all (same support, same k);
* **random floor** ``a/(2-a)``, the expected soft-IoU of a random field under
  matched-area top-k -- the column the review found missing and which turns out
  to sit *above* the measured field on half the evaluation set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

POOLS = ("where_special", "where_content", "instr_text")


def _agg(vals) -> dict[str, Any] | None:
    a = np.asarray([v for v in vals if v is not None], dtype=np.float64)
    if not a.size:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
            "frac_below_0.2": float((a < 0.2).mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--export-fixed", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--sink-k", type=float, default=3.0)
    ap.add_argument("--top-k-heads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    from q3vl.whereb.attnread import sink_mask_conjunctive
    from q3vl.whereb.attnprobe import (
        EXCLUDE_FIRST_LAYERS, combine_heads, field_scores, head_norm_constants,
        paired_wilcoxon,
    )
    from q3vl.whereb.metrics import center_prior_field, grid_boundary_f1

    exp, expf = Path(args.export), Path(args.export_fixed)
    published = json.loads(Path(args.metrics).read_text())
    geom = {}
    if args.geometry and Path(args.geometry).exists():
        geom = json.load(open(args.geometry)).get("geometry", {})

    rows = [json.loads(l) for l in (exp / "samples.jsonl").read_text().splitlines() if l.strip()]
    meta = {}
    for r in rows:
        meta.setdefault(r["sample_id"], {})[r["arm"]] = r

    rng = np.random.default_rng(args.seed)
    cache: dict[str, dict[str, Any]] = {}
    for sid, arms in meta.items():
        if "gt" not in arms or "shuffled" not in arms:
            continue
        fp = expf / "fields" / f"{sid}__fixed_phrase.npz"
        if not fp.exists():
            continue
        dg = np.load(exp / "fields" / f"{sid}__gt.npz")
        dsh = np.load(exp / "fields" / f"{sid}__shuffled.npz")
        df = np.load(fp)
        pg = dg["col_profile"].astype(np.float64).mean(axis=(0, 1))
        psh = dsh["col_profile"].astype(np.float64).mean(axis=(0, 1))
        # SAME valid mask as the published run, so every number is comparable
        valid = ~sink_mask_conjunctive(pg, psh, k=args.sink_k)
        gt = np.load(exp / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)
        if valid.sum() < 4 or (gt[valid] > 0.5).sum() < 1:
            continue
        gh, gw = arms["gt"]["grid_h"], arms["gt"]["grid_w"]
        cache[sid] = {
            "fold": arms["gt"]["fold"], "gh": gh, "gw": gw, "n_img": gh * gw,
            "valid": valid, "gt": gt,
            "A_gt": {p: dg[f"field_{p}"].astype(np.float64) for p in POOLS},
            "A_sh": {p: dsh[f"field_{p}"].astype(np.float64) for p in POOLS},
            "A_fx": {p: df[f"field_{p}"].astype(np.float64) for p in POOLS},
        }
    fit = [s for s in cache if cache[s]["fold"] == "fit"]
    oof = [s for s in cache if cache[s]["fold"] == "oof"]
    print(f"diff-arm usable: fit {len(fit)}, oof {len(oof)}")

    n_layers, n_heads = cache[fit[0]]["A_gt"][POOLS[0]].shape[:2]
    layer_ok = np.zeros(n_layers, dtype=bool)
    layer_ok[EXCLUDE_FIRST_LAYERS:] = True

    out: dict[str, Any] = {
        "card": "PR-ATT1-E1b / B1 -- fixed_phrase common-mode arm",
        "FAMILY_STATUS": "OUT OF FAMILY -- one-shot falsification patch",
        "p_value_discipline": (
            "All p-values below are UNCORRECTED and are NOT merged into the "
            "published 3-pool Westfall-Young null. Expanding a family after "
            "seeing its outcome is forbidden (DELTA ruling 4). These numbers "
            "close registered cells P4/P5 and cannot promote the raw arm."),
        "n_fit": len(fit), "n_oof": len(oof),
        "pools": {},
    }

    # ---------- random floor a/(2-a) and the centre prior ----------
    floors, priors_s, areas = [], [], []
    prior_cache = {}
    for sid in oof:
        c = cache[sid]
        v, g = c["valid"], c["gt"]
        a = float((g[v] > 0.5).sum() / v.sum())
        areas.append(a)
        floors.append(a / (2 - a) if a < 1 else 1.0)
        pf = center_prior_field(c["gh"], c["gw"]).numpy().astype(np.float64).reshape(-1)
        prior_cache[sid] = pf
        priors_s.append(float(field_scores(pf[None, None, :], g, v)[0, 0]))
    out["random_floor"] = {
        "formula": "E[soft-IoU] = a/(2-a) under matched-area top-k, a = GT area fraction",
        "note": ("the column the published REPORT lacked; it is the honest zero "
                 "point for every soft-IoU in this campaign"),
        "area_frac": _agg(areas), "floor": _agg(floors),
        "center_prior": _agg(priors_s),
    }
    # empirical check of the closed form, so it is not taken on faith
    emp = []
    for sid in oof[:60]:
        c = cache[sid]
        v, g = c["valid"], c["gt"]
        r = rng.standard_normal((8, c["n_img"]))
        emp.extend(field_scores(r[:, None, :].transpose(0, 1, 2), g, v).reshape(-1).tolist())
    out["random_floor"]["empirical_random_field"] = _agg(emp)

    # ---------- per pool: diff arm + P4 ----------
    for pool in POOLS:
        # diff field on the RAW attention scale, before any head normalisation
        def dfield(sid, a="gt"):
            c = cache[sid]
            return c["A_gt" if a == "gt" else "A_sh"][pool] - c["A_fx"][pool]

        per_head = np.zeros((len(fit), n_layers, n_heads))
        for i, sid in enumerate(fit):
            c = cache[sid]
            per_head[i] = field_scores(dfield(sid), c["gt"], c["valid"])
        fm = np.where(layer_ok[:, None], np.median(per_head, axis=0), -np.inf)
        order = np.argsort(-fm.reshape(-1))[: args.top_k_heads]
        idx = [(int(o // n_heads), int(o % n_heads)) for o in order]
        w = [float(fm[l, h]) for l, h in idx]
        mu, sg = head_norm_constants([dfield(s) for s in fit],
                                     [cache[s]["n_img"] for s in fit])

        d_gt, d_sh, pr, bf1, bf1_prior, bf1_rand = [], [], [], [], [], []
        raw_lo, raw_hi = [], []
        for sid in oof:
            c = cache[sid]
            v, g = c["valid"], c["gt"]
            f = combine_heads(dfield(sid, "gt"), idx, w, mu, sg, c["n_img"])
            fs = combine_heads(dfield(sid, "sh"), idx, w, mu, sg, c["n_img"])
            raw_lo.append(float(f[v].min())); raw_hi.append(float(f[v].max()))
            d_gt.append(float(field_scores(f[None, None, :], g, v)[0, 0]))
            d_sh.append(float(field_scores(fs[None, None, :], g, v)[0, 0]))
            pr.append(float(field_scores(prior_cache[sid][None, None, :], g, v)[0, 0]))

            gh, gw = c["gh"], c["gw"]
            kk = max(1, min(int((g[v] > 0.5).sum()), int(v.sum())))

            def m2(x):
                z = np.where(v, x, -np.inf)
                m = np.zeros(c["n_img"]); m[np.argsort(-z)[:kk]] = 1.0
                return torch.from_numpy(m.reshape(gh, gw))

            gtb = torch.from_numpy(((g > 0.5) & v).astype(np.float64).reshape(gh, gw))
            bf1.append(grid_boundary_f1(m2(f), gtb))
            bf1_prior.append(grid_boundary_f1(m2(prior_cache[sid]), gtb))
            # P5 GUARD: same support, same k, random values (red line)
            bf1_rand.append(grid_boundary_f1(m2(rng.standard_normal(c["n_img"])), gtb))

        a_gt, a_sh, a_pr = map(np.asarray, (d_gt, d_sh, pr))
        pub = published["pools"][pool]
        pub_gt = np.asarray([r["grid_soft_iou_gt"] for r in pub["per_sample"]])
        pub_ids = [r["sample_id"] for r in pub["per_sample"]]
        keep = [i for i, s in enumerate(pub_ids) if s in set(oof)]
        order_map = {s: i for i, s in enumerate(pub_ids)}
        pub_aligned = np.asarray([pub_gt[order_map[s]] for s in oof])

        out["pools"][pool] = {
            "selected_heads_diff": [{"layer": l, "head": h, "fit_median": ww}
                                    for (l, h), ww in zip(idx, w)],
            "P1_diff_oof": _agg(d_gt),
            "P2_diff_vs_center_prior": paired_wilcoxon(a_gt, a_pr),
            "P3_diff_vs_shuffleddiff": paired_wilcoxon(a_gt, a_sh),
            "diff_vs_published_raw_arm": paired_wilcoxon(a_gt, pub_aligned),
            "P5_boundary_f1": {
                "field": _agg(bf1), "center_prior": _agg(bf1_prior),
                "RANDOM_TOPK_GUARD": _agg(bf1_rand),
                "vs_center_prior": paired_wilcoxon(np.asarray(bf1), np.asarray(bf1_prior)),
                "vs_random_guard": paired_wilcoxon(np.asarray(bf1), np.asarray(bf1_rand)),
            },
            "meta_norm": {
                "domain_raw_diff_field": [float(np.min(raw_lo)), float(np.max(raw_hi))],
                "domain_note": ("D = S(instr) - S(fixed) on the arm-constant z scale; "
                                "CROSSES ZERO by construction -- any consumer that "
                                "clamps to (0,1) silently deletes the negative half "
                                "(s-cache contract)"),
                "clamp_applied": False,
                "frac_cells_negative": None,
            },
        }
        print(f"  [diff] {pool:15s} P1={np.median(d_gt):.4f} prior={np.median(pr):.4f} "
              f"dP2={np.median(a_gt-a_pr):+.4f} dP3={np.median(a_gt-a_sh):+.4f} "
              f"vs_raw={np.median(a_gt-pub_aligned):+.4f}")

    # ---------- P4: instr field vs fixed field (registered cell) ----------
    p4: dict[str, Any] = {"definition": ("paired grid soft-IoU, instr field vs fixed-phrase "
                                         "field, SAME top-8 heads selected on the gt arm")}
    for pool in POOLS:
        heads = [(h["layer"], h["head"])
                 for h in published["pools"][pool]["selected_heads"]]
        ws = [h["fit_median_soft_iou"]
              for h in published["pools"][pool]["selected_heads"]]
        mu, sg = head_norm_constants([cache[s]["A_gt"][pool] for s in fit],
                                     [cache[s]["n_img"] for s in fit])
        gi, fi, sub = [], [], []
        for sid in oof:
            c = cache[sid]
            v, g = c["valid"], c["gt"]
            fg = combine_heads(c["A_gt"][pool], heads, ws, mu, sg, c["n_img"])
            ff = combine_heads(c["A_fx"][pool], heads, ws, mu, sg, c["n_img"])
            gi.append(float(field_scores(fg[None, None, :], g, v)[0, 0]))
            fi.append(float(field_scores(ff[None, None, :], g, v)[0, 0]))
            gg = geom.get(sid, {})
            sub.append((gg.get("area_frac"), gg.get("centroid_dist")))
        a, b = np.asarray(gi), np.asarray(fi)
        entry = {"all": paired_wilcoxon(a, b),
                 "median_instr": float(np.median(a)), "median_fixed": float(np.median(b))}
        # registered subsets: small target / eccentric
        small = [i for i, (ar, _) in enumerate(sub) if ar is not None and ar < 0.15]
        ecc = [i for i, (_, cd) in enumerate(sub) if cd is not None and cd >= 0.3]
        if len(small) >= 10:
            entry["small_target_area_lt_0.15"] = {
                **paired_wilcoxon(a[small], b[small]), "n_sub": len(small)}
        if len(ecc) >= 10:
            entry["eccentric_centroid_ge_0.3"] = {
                **paired_wilcoxon(a[ecc], b[ecc]), "n_sub": len(ecc)}
        p4[pool] = entry
        print(f"  [P4]   {pool:15s} instr={np.median(a):.4f} fixed={np.median(b):.4f} "
              f"D={entry['all']['delta_median']:+.4f} p={entry['all']['p_value']:.3g}")
    out["P4_instr_vs_fixed"] = p4

    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nrandom floor: median {out['random_floor']['floor']['median']:.4f} "
          f"(empirical {out['random_floor']['empirical_random_field']['median']:.4f}) "
          f"vs centre prior {out['random_floor']['center_prior']['median']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
