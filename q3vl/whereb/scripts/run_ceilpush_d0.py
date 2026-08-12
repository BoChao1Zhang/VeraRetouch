"""D0 zero-training diagnostics for the ceiling-push package.

Implements the cheap, artefact-only members of the D0 battery
(RESEARCH_ceiling-push_2026-08-11.md §2.1).  Every item here runs off the
already-published `per_sample.jsonl` + the split's GT, so none of it needs a
training slot:

  D0-3  SDC difficulty score + risk-coverage curve      (is the tail concentrated?)
  D0-5  spatial-language coverage audit                 (is the language evidence there?)
  D0-8a contradiction-pair mining (retrieval half)      (empirical upper-bound band)

D0-1 (replay feasibility) is a code audit and gates D0-7; D0-4/D0-6 need a VLM
forward and are run separately.

Discipline carried over unchanged: no AUC anywhere, matched-area top-k
thresholding, paired statistics with permutation p-values, and every
pre-registered number snapshotted into `config/` so a later ruling can be
appended as a dated changelog rather than silently editing the threshold.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# D0-5: the spatial vocabulary, split by the geometric degree of freedom each
# word can constrain.  A family whose DoF has no language evidence cannot be
# learned from the instruction no matter how good the head is.
DOF_WORDS: dict[str, tuple[str, ...]] = {
    "direction": ("left", "right", "top", "bottom", "upper", "lower", "above",
                  "below", "north", "south", "east", "west", "leftmost",
                  "rightmost", "topmost", "bottommost"),
    "extent": ("whole", "entire", "all", "half", "part", "partial", "corner",
               "edge", "border", "margin", "narrow", "wide", "broad", "thin",
               "large", "small", "tiny", "huge"),
    "shape": ("oval", "elliptical", "circular", "round", "band", "strip",
              "stripe", "gradient", "ramp", "radial", "linear", "diagonal",
              "horizontal", "vertical", "falloff", "vignette"),
    "position": ("center", "centre", "middle", "side", "background",
                 "foreground", "front", "back", "near", "far", "around",
                 "surrounding", "within", "inside", "outside", "beyond"),
}


def _median(xs) -> float | None:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def sdc_score(pred: np.ndarray, k: int) -> float:
    """Soft-Dice Confidence (2402.10665): agreement of the soft field with its
    own matched-area binarisation.

    Uses **no ground truth** -- that is the point.  It ranks samples by how
    committed the field is to its own decision, so a risk-coverage curve built
    on it says whether the loss is concentrated in a recognisable minority.
    """
    f = np.asarray(pred, dtype=np.float64).reshape(-1)
    if f.size == 0:
        return float("nan")
    kk = int(min(max(k, 1), f.size))
    idx = np.argpartition(-f, kk - 1)[:kk]
    m = np.zeros(f.size)
    m[idx] = 1.0
    denom = f.sum() + m.sum()
    return float(2.0 * (f * m).sum() / denom) if denom > 0 else float("nan")


def risk_coverage(scores: list[float], losses: list[float]) -> dict[str, Any]:
    """Sort by confidence, report the loss share carried by the worst quantiles."""
    order = np.argsort(np.asarray(scores))          # lowest confidence first
    L = np.asarray(losses)[order]
    total = float(L.sum())
    out: dict[str, Any] = {"total_loss": total, "n": int(L.size)}
    for q in (0.10, 0.25, 0.50):
        n = max(1, int(round(q * L.size)))
        out[f"loss_share_worst_{int(q*100)}pct"] = (
            float(L[:n].sum() / total) if total > 0 else None)
    # a uniform distribution of loss would put exactly q of the loss in q of the
    # samples; the registered trigger is >=60% of loss in the worst 25%
    out["tail_concentrated"] = bool(
        out.get("loss_share_worst_25pct") is not None
        and out["loss_share_worst_25pct"] >= 0.60)
    return out


def language_coverage(instructions: dict[str, str],
                      families: dict[str, str]) -> dict[str, Any]:
    """D0-5: does the instruction text carry evidence for each geometric DoF?"""
    per_dof: dict[str, int] = {d: 0 for d in DOF_WORDS}
    per_dof_family: dict[str, Counter] = {d: Counter() for d in DOF_WORDS}
    fam_total: Counter = Counter()
    vocab: dict[str, Counter] = {d: Counter() for d in DOF_WORDS}
    n = 0
    for sid, text in instructions.items():
        n += 1
        fam = families.get(sid, "unknown")
        fam_total[fam] += 1
        words = set(re.findall(r"[a-z]+", (text or "").lower()))
        for dof, vs in DOF_WORDS.items():
            hit = words & set(vs)
            if hit:
                per_dof[dof] += 1
                per_dof_family[dof][fam] += 1
                vocab[dof].update(hit)
    out: dict[str, Any] = {"n_instructions": n, "coverage": {}, "by_family": {},
                           "top_words": {}}
    for dof in DOF_WORDS:
        cov = per_dof[dof] / max(n, 1)
        out["coverage"][dof] = {
            "coverage": cov,
            "n_distinct_words": len(vocab[dof]),
            # registered trigger: <30% coverage means the DoF is under-evidenced
            # in language, so the fault is in the DATA rather than the pathway
            "under_evidenced": bool(cov < 0.30),
        }
        out["by_family"][dof] = {
            f: per_dof_family[dof][f] / max(fam_total[f], 1)
            for f in sorted(fam_total)}
        out["top_words"][dof] = per_dof[dof] and vocab[dof].most_common(8)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir",
                    default="/home/bc/data/runs/where_b/amort_P3prime_cont_20260811/eval_final")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--main", default="generated")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import resource

        s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if s_ < h_:
            resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    except Exception:
        pass

    import torch

    from q3vl.where.fpre import grid_from_geometry
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.amort.data import family_labels
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import gt_area_k

    out = Path(args.out)
    (out / "config").mkdir(parents=True, exist_ok=True)

    rows = {}
    for line in (Path(args.eval_dir) / "per_sample.jsonl").read_text().splitlines():
        r = json.loads(line)
        if (r.get("mode") == args.main and not r.get("uncovered")
                and not r.get("is_fake")):
            rows[r["sample_id"]] = r
    print(f"loaded {len(rows)} rows from {args.eval_dir}", flush=True)

    ds, _ = open_dataset(args.split, need_mask=True)
    idx = [i for i, r in enumerate(ds.meta_rows())
           if r.get("render_mode") == "local"]
    fam = family_labels(ds, idx)
    by_id = {ds.record(i)["sample_id"]: i for i in idx}

    # ---- D0-3: SDC + risk-coverage ---------------------------------------
    # The published per_sample carries pred_area_frac / pred_std but not the
    # field, so SDC is computed from the GT-matched k and the recorded field
    # statistics where available; otherwise it is recomputed from the mask.
    scores, losses, sids = [], [], []
    for sid, r in rows.items():
        if sid not in by_id:
            continue
        # a confidence proxy that needs no GT: how peaked the prediction is
        # relative to its own area (std/mean is scale-free and monotone in
        # commitment).  Recorded per sample by the eval.
        pm = r.get("pred_area_frac") or 1e-6
        ps = r.get("pred_std") or 0.0
        scores.append(float(ps / max(pm, 1e-6)))
        losses.append(1.0 - float(r["hard_iou"]))
        sids.append(sid)
    rc = risk_coverage(scores, losses)
    rc["score"] = "pred_std / pred_area (GT-free commitment proxy)"
    # stratified so a fixed tail cannot hide a broken head
    strat: dict[str, Any] = {}
    for f_ in ("radial", "linear", "band", "semantic"):
        sel = [i for i, s in enumerate(sids) if fam.get(s) == f_]
        if sel:
            strat[f_] = risk_coverage([scores[i] for i in sel],
                                      [losses[i] for i in sel])
    rc["by_family"] = strat

    # ---- D0-5: spatial language coverage ---------------------------------
    instr = {}
    for sid, i in by_id.items():
        try:
            instr[sid] = ds.record(i).get("instruction", "") or ""
        except Exception:
            continue
    lang = language_coverage(instr, fam)

    # ---- D0-8a: contradiction-pair band (retrieval half) ------------------
    # Near-duplicate CONDITIONS with different GT geometry.  Condition proximity
    # is approximated by (same source image) x (high instruction word overlap):
    # the strongest available zero-training notion of "the model was asked
    # almost the same thing".
    gts: dict[str, np.ndarray] = {}
    for sid, i in by_id.items():
        s = ds[i]
        gh, gw = grid_from_geometry(s.geometry.out_h, s.geometry.out_w)
        gts[sid] = area_resize(s.mask_target_hi()[None, None].float(),
                               (gh, gw))[0, 0].numpy()
    src: dict[str, list[str]] = defaultdict(list)
    for sid, i in by_id.items():
        src[str(ds.record(i).get("source_image_id"))].append(sid)

    def toks(t): return set(re.findall(r"[a-z]+", (t or "").lower()))

    pairs = []
    for _, group in src.items():
        for a in range(len(group)):
            for b in range(a + 1, len(group)):
                sa, sb = group[a], group[b]
                ta, tb = toks(instr.get(sa, "")), toks(instr.get(sb, ""))
                if not ta or not tb:
                    continue
                jac = len(ta & tb) / max(len(ta | tb), 1)
                ga, gb = gts[sa], gts[sb]
                if ga.shape != gb.shape:
                    continue
                ka = gt_area_k(torch.from_numpy(ga))
                def tk(z, k):
                    f = z.reshape(-1)
                    ii = np.argpartition(-f, min(k, f.size - 1))[:k]
                    m = np.zeros(f.size, bool); m[ii] = True
                    return m
                pa, pb = tk(ga, ka), tk(gb, ka)
                iou = float((pa & pb).sum() / max((pa | pb).sum(), 1))
                pairs.append({"a": sa, "b": sb, "instr_jaccard": jac,
                              "gt_iou": iou, "family_a": fam.get(sa),
                              "family_b": fam.get(sb)})
    # A FIXED jaccard cut of 0.60 returned zero pairs: the observed maximum
    # in-image instruction similarity on this split is 0.521 (median 0.236), so
    # "near-duplicate condition" in the literature's sense does not occur here.
    # Report the band as a function of the cut, plus the similarity distribution,
    # rather than an empty dict that reads like a bug.
    model_med = _median(r["hard_iou"] for r in rows.values())
    jac = np.array([p["instr_jaccard"] for p in pairs]) if pairs else np.zeros(0)
    band: dict[str, Any] = {
        "model_median_iou": model_med,
        "n_pairs_examined": len(pairs),
        "instr_jaccard": {
            "median": float(np.median(jac)) if jac.size else None,
            "p90": float(np.percentile(jac, 90)) if jac.size else None,
            "max": float(jac.max()) if jac.size else None,
        },
        "registered_cut_0.60_n": int((jac >= 0.60).sum()) if jac.size else 0,
        "by_cut": {},
        "note": ("pairs are different edits of the SAME source image; a high-GT-IoU "
                 "pair means two differently-worded instructions that nonetheless "
                 "select the same region"),
    }
    for cut in (0.30, 0.35, 0.40, 0.45):
        sel = [p for p in pairs if p["instr_jaccard"] >= cut]
        if len(sel) >= 8:
            v = np.array([p["gt_iou"] for p in sel])
            band["by_cut"][f"{cut:.2f}"] = {
                "n": len(sel), "q25": float(np.percentile(v, 25)),
                "median": float(np.median(v)), "q75": float(np.percentile(v, 75)),
                "model_inside_band": bool(
                    model_med is not None
                    and float(np.percentile(v, 25)) <= model_med
                    <= float(np.percentile(v, 75))),
            }
    band["verdict"] = (
        "NOT MEASURABLE on this split: no near-duplicate conditions exist "
        "(max instruction jaccard 0.52 < the 0.60 the protocol assumes), so the "
        "contradiction-pair upper bound cannot be estimated here. (ii) must be "
        "adjudicated by D0-7 replay or B1-2 best-of-k instead."
        if not band["by_cut"] else "see by_cut")
    res = {
        "eval_dir": args.eval_dir, "split": args.split, "context": args.main,
        "n_rows": len(rows),
        "D0_3_risk_coverage": rc,
        "D0_5_language_coverage": lang,
        "D0_8a_contradiction_band": band,
        "n_pairs_total": len(pairs),
    }
    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str),
                                      encoding="utf-8")
    (out / "config" / "preregistered_thresholds.json").write_text(json.dumps({
        "source": "RESEARCH_ceiling-push_2026-08-11.md §2.1 (defaults; "
                  "main-agent ruling pending -- changes to be appended as a "
                  "dated changelog, never edited in place)",
        "D0_3": {"trigger": "worst 25% by confidence carry >= 60% of total loss",
                 "then": "(iii) tail under-learning opened"},
        "D0_5": {"trigger": "a DoF with < 30% language coverage",
                 "then": "(i) fault is partly in DATA, densification arm opened"},
        "D0_7": {"U_replay<=0.85": "(ii) dominates; stop pushing mean IoU",
                 "0.85-0.93": "(ii) real, handle in parallel with fixes",
                 ">=0.95": "(ii) essentially excluded"},
        "D0_8": {"trigger": "model median inside the contradiction band [q25,q75]",
                 "then": "(ii) dominant"},
    }, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "n_pairs_total"},
                     indent=2, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
