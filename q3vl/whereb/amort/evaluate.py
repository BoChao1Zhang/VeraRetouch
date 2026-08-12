"""PR-AMORT evaluation -- every column the protocol makes mandatory, and no AUC.

The criteria table for a spatial field is fixed by the campaign's red lines and
is reproduced here in full, because a partial table is what let three separate
wrong conclusions through earlier in this project:

* **the matched-area top-k IoU** as the primary column.  This is the published
  convention and it is not optional: thresholding both fields to a top-k of the
  GT's area and taking the IoU is what produces the numbers every prior card
  quotes -- centre prior **0.5088**, random floor **0.2582**, W01 **0.550**,
  oracle **0.9737**.  Verified on 2026-08-11 by recomputing all three published
  baselines from the raw masks and matching them to four decimals.  The raw
  (unthresholded) soft-IoU is a *different number* -- 0.4548 for the same centre
  prior -- and is reported next to it as ``soft_iou_raw``, never in its place;
* **grid-level boundary F1** (the pixel-level 3px variant is banned as a
  criterion: a random top-k scores 0.0394 on it, above the centre prior's 0.0327,
  because it mostly measures boundary length);
* the **centre-prior column**, zero-parameter, same support and same
  thresholding rule -- E3 showed both trained arms were explained better by it
  than by GT, so no claim survives without it;
* the **random floor** ``a/(2-a)``, which on the ``area>=0.45`` stratum is
  already 0.527 and makes half the set nearly undiscriminable;
* **area strata** and **mask_type strata**;
* the **swap-subject paired delta** (the real conditionality probe) and the
  **antonym invariance** column, reported with a |Delta| <= 0.05 threshold and
  never trained on.

Additionally, and specific to this campaign, every board carries the E3
falsification column: ``corr(pred, centre prior)`` vs ``corr(pred, GT)``.  NOTES
§7.1 registers it as a promotion gate -- an arm for which the former still
exceeds the latter has the same disease as W01/W02 and does not advance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from q3vl.whereb.metrics import (center_prior_field, grid_boundary_f1, gt_area_k,
                                 hard_iou, paired_delta, percentile, soft_iou_value,
                                 topk_mask)

__all__ = ["random_floor", "field_corr", "center_prior_unit", "evaluate_context",
           "evaluate_arm", "AREA_BINS", "summarise_rows"]

#: E1 review R5 / E5 §3.4.  The last bin is reported but flagged: its random
#: floor is already 0.527 and it holds ~47% of V_where.
AREA_BINS: tuple[tuple[float, float], ...] = ((0.0, 0.15), (0.15, 0.30),
                                              (0.30, 0.45), (0.45, 1.01))


def center_prior_unit(grid_h: int, grid_w: int, device=None) -> torch.Tensor:
    """The centre prior mapped analytically onto ``[0, 1]``.

    ``center_prior_field`` returns ``-distance-to-centre``, which is **negative
    everywhere**.  Feeding it straight into a soft-IoU yields a meaningless
    number (measured: -5.63 where a soft-IoU must be in [0,1]) -- the top-k
    hard-IoU and boundary-F1 columns are immune because thresholding is
    invariant to any monotone map, which is exactly why the bug is invisible
    unless the soft column is looked at.

    The rescaling is a deterministic function of ``(grid_h, grid_w)`` alone --
    it touches no image content -- so it is not the per-image min-max the
    visualisation red line forbids; that rule exists because a data-dependent
    denominator can be captured by outliers or pad cells, and neither exists
    here.
    """
    cp = center_prior_field(grid_h, grid_w, device=device)
    lo, hi = cp.min(), cp.max()
    return (cp - lo) / (hi - lo + 1e-12)


def random_floor(area_frac: float) -> float:
    """Expected soft-IoU of a random top-k of the matched size: ``a/(2-a)``."""
    a = float(min(max(area_frac, 0.0), 1.0))
    return 1.0 if a >= 1.0 else a / (2.0 - a)


def field_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    x = a.detach().float().reshape(-1)
    y = b.detach().float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    d = float(x.norm() * y.norm())
    return float((x @ y) / d) if d > 1e-12 else 0.0


@torch.no_grad()
def evaluate_context(
    model, builder, dataset, indices: Sequence[int], mode: str,
    *, batch_size: int = 8, want_hi: bool = False, limit: int | None = None,
    progress: bool = False,
) -> list[dict[str, Any]]:
    """One context mode over the eval indices -> per-sample rows."""
    model.eval()
    idx = list(indices)[: limit or len(indices)]
    rows: list[dict[str, Any]] = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start:start + batch_size]
        samples = [dataset[i] for i in chunk]
        # Review B1: build the chunk, but fall back to per-sample on KeyError.
        # `context_for(mode="shuffled")` raises on the FIRST partnerless sample,
        # which used to mark the whole chunk uncovered -- turning 18 genuinely
        # partnerless V_where samples into as many as ~144 "uncovered" rows and
        # computing `swap_subject_delta` (a registered hard gate) on a subset
        # selected by index adjacency.
        try:
            inputs = builder.build(samples, [mode] * len(samples))
        except KeyError:
            inputs = []
            for s in samples:
                try:
                    inputs.extend(builder.build([s], [mode]))
                except KeyError:
                    rows.append({"sample_id": s.sample_id, "mode": mode,
                                 "uncovered": True})
        for x in inputs:
            cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
            if x.route_semantic and model.sem is not None:
                out = model.forward_sem(x.feat, cond, sim=x.sim, center=x.center,
                                        geom=x.geom,
                                        guide_hi=x.guide_hi if want_hi else None)
                head = "semantic"
            else:
                out = model.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                        center=x.center, geom=x.geom,
                                        guide_hi=x.guide_hi if want_hi else None,
                                        grid_h=x.grid_h, grid_w=x.grid_w)
                head = "geometry"
            m = out["m_low"].float()
            gt = x.gt_low.float()
            gh, gw = x.grid_h, x.grid_w
            k = gt_area_k(gt)
            area_frac = float((gt > 0.5).float().mean())
            cp = center_prior_unit(gh, gw, device=m.device)
            pred_k = topk_mask(m, k)
            gt_k = topk_mask(gt, k)
            cp_k = topk_mask(cp, k)
            rows.append({
                "sample_id": x.sample_id, "mode": mode, "head": head,
                "family": x.family, "uncovered": False,
                "grid_h": gh, "grid_w": gw,
                "soft_iou": soft_iou_value(m, gt),
                "hard_iou": hard_iou(pred_k, gt_k),
                "grid_boundary_f1": grid_boundary_f1(pred_k, gt_k),
                "center_prior_soft_iou": soft_iou_value(cp, gt),
                "center_prior_hard_iou": hard_iou(cp_k, gt_k),
                "center_prior_boundary_f1": grid_boundary_f1(cp_k, gt_k),
                "random_floor": random_floor(area_frac),
                "gt_area_frac": area_frac,
                "pred_area_frac": float(m.mean()),
                "gt_mean": float(gt.mean()),
                "pred_std": float(m.std()), "gt_std": float(gt.std()),
                # the E3 falsification pair
                "corr_pred_center": field_corr(m, cp),
                "corr_pred_gt": field_corr(m, gt),
                "is_fake": x.is_fake,
                "winner_confidence": x.meta.get("winner_confidence"),
                "build": x.meta.get("build"),
                # SHAPE3's pre-registered criterion.  It is "is this the right
                # SHAPE", which `hard_iou` cannot answer: a smooth blob and a
                # proper ellipse can score the same IoU.  The GT column is the
                # control that makes the number readable (it is ~0 by
                # construction), so both are computed or neither is.
                **_shape_residual_row(m, gt, x.family, k),
            })
        if progress and (start // max(1, batch_size)) % 10 == 0:
            print(f"  [{mode}] {start + len(chunk)}/{len(idx)}", flush=True)
    return rows


#: families `best_fit_analytic` can fit; `semantic` has no analytic member, so
#: a residual against one would be meaningless rather than merely large.
_SHAPE_FAMILIES = ("radial", "linear", "band")


def _shape_residual_row(m, gt, family: str, k: int) -> dict[str, Any]:
    """``shape_residual`` for one sample, plus its GT self-fit control.

    Wired 2026-08-12.  It had been defined in `edgequal.py` since the SHAPE3
    arms were designed and **never called anywhere** -- both arms ran, landed,
    and were read out on IoU alone, so the criterion the experiment existed to
    test was silently absent from its own board.  See `assert_criteria_ran`.
    """
    if family not in _SHAPE_FAMILIES:
        return {"shape_residual": None, "shape_residual_gt": None}
    from q3vl.whereb.edgequal import shape_residual

    y = m.detach().cpu().numpy()
    g = gt.detach().cpu().numpy()
    try:
        return {"shape_residual": shape_residual(y, family, k=k),
                "shape_residual_gt": shape_residual(g, family, k=k)}
    except Exception as exc:                                    # noqa: BLE001
        # never let an optional column take the board down (D-20 lesson 6),
        # but never let it vanish silently either
        return {"shape_residual": None, "shape_residual_gt": None,
                "shape_residual_error": repr(exc)}


def assert_criteria_ran(board: dict[str, Any], arm: str) -> dict[str, Any]:
    """Fail loudly when an arm's pre-registered criterion was never computed.

    Third occurrence of the same class of error in this campaign ("defined,
    measured, not wired"): a criterion function exists, the experiment cites it,
    and nothing calls it -- so the board looks complete and the arm is judged on
    whatever column *did* get computed.  A criterion that is not asserted at
    runtime is a criterion that can quietly not exist.
    """
    required = {"SHAPE3": ["shape_residual"]}.get(arm, [])
    report = {"arm": arm, "required": required, "computed": {}}
    for name in required:
        col = board.get("criteria_columns", {}).get(name, {})
        n = int(col.get("n", 0))
        report["computed"][name] = n
        if n == 0:
            raise AssertionError(
                f"arm {arm} pre-registers '{name}' and the board carries 0 "
                "values for it; refusing to publish a board that cannot "
                "adjudicate its own experiment")
    return report


def _agg(xs: Iterable[float]) -> dict[str, Any]:
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not v:
        return {"n": 0}
    return {"n": len(v), "mean": float(np.mean(v)), "median": float(np.median(v)),
            "p10": percentile(v, 10), "p25": percentile(v, 25),
            "p75": percentile(v, 75), "p90": percentile(v, 90),
            "min": float(np.min(v)), "max": float(np.max(v))}


def summarise_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    live = [r for r in rows if not r.get("uncovered") and not r.get("is_fake")]
    if not live:
        return {"n": 0, "n_uncovered": sum(1 for r in rows if r.get("uncovered"))}
    out: dict[str, Any] = {
        "n": len(live),
        "n_uncovered": sum(1 for r in rows if r.get("uncovered")),
        # primary, published convention (matched-area top-k)
        "topk_iou": _agg(r["hard_iou"] for r in live),
        "soft_iou_raw": _agg(r["soft_iou"] for r in live),
        "grid_boundary_f1": _agg(r["grid_boundary_f1"] for r in live),
        "center_prior_topk_iou": _agg(r["center_prior_hard_iou"] for r in live),
        "center_prior_soft_iou_raw": _agg(r["center_prior_soft_iou"] for r in live),
        "center_prior_boundary_f1": _agg(r["center_prior_boundary_f1"] for r in live),
        "random_floor": _agg(r["random_floor"] for r in live),
        "area_ratio": _agg(r["pred_area_frac"] / max(r["gt_mean"], 1e-6) for r in live),
        "std_ratio": _agg(r["pred_std"] / max(r["gt_std"], 1e-6) for r in live),
        "corr_pred_center": _agg(r["corr_pred_center"] for r in live),
        "corr_pred_gt": _agg(r["corr_pred_gt"] for r in live),
    }
    # paired, same-sample: the only honest way to compare against the prior
    # paired and in the published convention, so the delta is comparable with
    # every number the earlier cards quote
    out["delta_vs_center_prior"] = paired_delta(
        [r["hard_iou"] for r in live], [r["center_prior_hard_iou"] for r in live])
    out["delta_vs_random_floor"] = paired_delta(
        [r["hard_iou"] for r in live], [r["random_floor"] for r in live])
    # the E3 promotion gate, as a paired difference
    out["corr_center_minus_corr_gt"] = paired_delta(
        [r["corr_pred_center"] for r in live], [r["corr_pred_gt"] for r in live])
    out["m3_disease_present"] = bool(
        out["corr_center_minus_corr_gt"]["delta"] > 0)

    # strata
    strata: dict[str, Any] = {"area": {}, "family": {}, "head": {},
                              "winner_confidence": {}}
    for lo, hi in AREA_BINS:
        sub = [r for r in live if lo <= r["gt_area_frac"] < hi]
        if sub:
            strata["area"][f"{lo:.2f}-{hi:.2f}"] = {
                "n": len(sub),
                "topk_iou_median": float(np.median([r["hard_iou"] for r in sub])),
                "center_prior_median": float(np.median(
                    [r["center_prior_hard_iou"] for r in sub])),
                "random_floor_median": float(np.median([r["random_floor"] for r in sub])),
                "boundary_f1_median": float(np.median(
                    [r["grid_boundary_f1"] for r in sub])),
            }
    for key in ("family", "head", "winner_confidence"):
        for val in sorted({str(r.get(key)) for r in live}):
            sub = [r for r in live if str(r.get(key)) == val]
            strata[key][val] = {
                "n": len(sub),
                "topk_iou_median": float(np.median([r["hard_iou"] for r in sub])),
                "center_prior_median": float(np.median(
                    [r["center_prior_hard_iou"] for r in sub])),
                "random_floor_median": float(np.median([r["random_floor"] for r in sub])),
                "boundary_f1_median": float(np.median(
                    [r["grid_boundary_f1"] for r in sub])),
            }
    out["strata"] = strata

    # Review U7 / data discipline: `winner_confidence == "low"` must not enter
    # evaluation GT.  It is 44% of V_where local (176/400) and it inflates every
    # IoU column at once, because low samples have larger GT area and therefore a
    # higher random floor.  The headline is therefore normal-only; the pooled
    # number stays visible beside it so the two can never be silently swapped.
    normal = [r for r in live if r.get("winner_confidence") == "normal"]
    if normal:
        out["headline_normal_only"] = {
            "n": len(normal),
            "topk_iou_median": float(np.median([r["hard_iou"] for r in normal])),
            "center_prior_median": float(np.median(
                [r["center_prior_hard_iou"] for r in normal])),
            "random_floor_median": float(np.median([r["random_floor"] for r in normal])),
            "boundary_f1_median": float(np.median(
                [r["grid_boundary_f1"] for r in normal])),
            "delta_vs_center_prior": paired_delta(
                [r["hard_iou"] for r in normal],
                [r["center_prior_hard_iou"] for r in normal]),
            "corr_center_minus_corr_gt": paired_delta(
                [r["corr_pred_center"] for r in normal],
                [r["corr_pred_gt"] for r in normal]),
        }
        out["headline_convention"] = (
            "normal-only (winner_confidence=='normal'); pooled shown alongside "
            "but is NOT the reporting convention (data discipline)")
    return out


def _paired_by_id(a: Sequence[dict], b: Sequence[dict], key: str = "hard_iou"):
    ma = {r["sample_id"]: r for r in a if not r.get("uncovered")}
    mb = {r["sample_id"]: r for r in b if not r.get("uncovered")}
    ids = sorted(set(ma) & set(mb))
    return ([ma[i][key] for i in ids], [mb[i][key] for i in ids], ids)


def evaluate_arm(
    model, builder, dataset, indices: Sequence[int],
    *, contexts: Sequence[str] = ("gt", "generated", "shuffled", "antonym",
                                  "fixed_phrase", "irrelevant_words"),
    batch_size: int = 8, want_hi: bool = False, limit: int | None = None,
    out_dir: Path | None = None, quick: bool = False, progress: bool = False,
) -> dict[str, Any]:
    if quick:
        contexts = ("gt", "shuffled", "antonym")
    per_context: dict[str, list[dict[str, Any]]] = {}
    for mode in contexts:
        per_context[mode] = evaluate_context(
            model, builder, dataset, indices, mode, batch_size=batch_size,
            want_hi=want_hi, limit=limit, progress=progress)

    board: dict[str, Any] = {"contexts": {m: summarise_rows(r)
                                          for m, r in per_context.items()}}
    main = "generated" if "generated" in per_context else "gt"
    board["main_context"] = main
    ref = per_context[main]

    # swap-subject: the registered first-class training signal and gate
    if "shuffled" in per_context:
        a, b, ids = _paired_by_id(ref, per_context["shuffled"])
        board["swap_subject_delta"] = {**paired_delta(a, b), "n_pairs": len(ids)}
    # antonym: invariance control, reported only, |Delta| <= 0.05
    if "antonym" in per_context:
        a, b, ids = _paired_by_id(ref, per_context["antonym"])
        d = [abs(x - y) for x, y in zip(a, b)]
        board["antonym_invariance"] = {
            "median_abs_delta": float(np.median(d)) if d else None,
            "max_abs_delta": float(np.max(d)) if d else None,
            "threshold": 0.05, "n_pairs": len(ids),
            "within_threshold": bool(d and float(np.median(d)) <= 0.05),
            "note": "negative control, never trained on (E3 erratum)",
        }
    for neg in ("fixed_phrase", "irrelevant_words"):
        if neg in per_context:
            a, b, ids = _paired_by_id(ref, per_context[neg])
            board[f"delta_vs_{neg}"] = {**paired_delta(a, b), "n_pairs": len(ids)}

    # the m_sem first-class deliverable: the semantic subset, on its own
    sem_rows = [r for r in ref if r.get("family") == "semantic"
                and not r.get("uncovered")]
    if sem_rows:
        board["m_sem"] = {
            "n": len(sem_rows),
            "topk_iou_median": float(np.median([r["hard_iou"] for r in sem_rows])),
            "soft_iou_raw_median": float(np.median([r["soft_iou"] for r in sem_rows])),
            "boundary_f1_median": float(np.median(
                [r["grid_boundary_f1"] for r in sem_rows])),
            "center_prior_median": float(np.median(
                [r["center_prior_hard_iou"] for r in sem_rows])),
            "random_floor_median": float(np.median(
                [r["random_floor"] for r in sem_rows])),
            "routed_to_semantic_head": float(np.mean(
                [r["head"] == "semantic" for r in sem_rows])),
        }
    # pre-registered criterion columns, on the analytic families only (the
    # families that HAVE an analytic member to be fitted against), with the GT
    # self-fit alongside every entry -- the residual is unreadable without it.
    live_ref = [r for r in ref if not r.get("uncovered") and not r.get("is_fake")]
    shp = [r for r in live_ref if r.get("shape_residual") is not None]
    board["criteria_columns"] = {
        "shape_residual": {
            **_agg(r["shape_residual"] for r in shp),
            "gt_control": _agg(r["shape_residual_gt"] for r in shp),
            "by_family": {
                fam: {**_agg(r["shape_residual"] for r in shp
                             if r["family"] == fam),
                      "gt_control_median": (
                          float(np.median([r["shape_residual_gt"] for r in shp
                                           if r["family"] == fam]))
                          if any(r["family"] == fam for r in shp) else None)}
                for fam in _SHAPE_FAMILIES},
            "n_errors": sum(1 for r in live_ref
                            if r.get("shape_residual_error")),
            "note": ("1 - IoU(pred, best-fit member of the sample's own "
                     "analytic family); semantic family excluded (no analytic "
                     "member exists to fit)"),
        },
    }

    # routing confusion, measured not assumed
    board["routing"] = {
        "n": len(ref),
        "frac_routed_semantic": float(np.mean([r.get("head") == "semantic"
                                               for r in ref])) if ref else 0.0,
        "confusion": _routing_confusion(ref),
    }

    # gates (hard, then selection on median soft-IoU; never a val loss)
    s = board["contexts"][main]
    ar = s.get("area_ratio", {}).get("median")
    gate = {
        "area_ratio_median": ar,
        "area_ratio_ok": bool(ar is not None and 0.8 <= ar <= 1.5),
        "swap_delta": board.get("swap_subject_delta", {}).get("delta"),
        "swap_delta_ok": bool(board.get("swap_subject_delta", {}).get("delta", 0) > 0),
        "antonym_ok": bool(board.get("antonym_invariance", {}).get("within_threshold",
                                                                  True)),
    }
    gate["pass"] = bool(gate["area_ratio_ok"] and gate["swap_delta_ok"]
                        and gate["antonym_ok"])
    board["gate"] = gate
    # NOTE the convention: this key keeps its historical name so the selection
    # code and the published boards line up, but it carries the matched-area
    # top-k IoU -- the number 0.5088 / 0.550 / 0.9737 are all quoted in.
    board["topk_iou_median"] = s.get("topk_iou", {}).get("median")
    board["topk_iou_median_pooled"] = board["topk_iou_median"]
    # selection and headline both use the normal-only figure when available
    hn = s.get("headline_normal_only")
    if hn:
        board["topk_iou_median_normal_only"] = hn["topk_iou_median"]
        board["topk_iou_median"] = hn["topk_iou_median"]
    board["local_soft_iou_median"] = board["topk_iou_median"]
    board["soft_iou_raw_median"] = s.get("soft_iou_raw", {}).get("median")
    board["baselines"] = {"center_prior": s.get("center_prior_topk_iou", {}).get("median"),
                          "random_floor": s.get("random_floor", {}).get("median"),
                          "published_center_prior_V_where": 0.5088,
                          "published_random_floor_V_where": 0.2582,
                          "published_oracle_ceiling": 0.9737,
                          "published_W01_generated": 0.550,
                          "convention": "matched-area top-k IoU"}
    board["gate_pass"] = gate["pass"]
    # runtime assertion, not a convention: an arm whose pre-registered criterion
    # produced no values must not be able to publish a board at all
    board["criteria_assertion"] = assert_criteria_ran(
        board, getattr(model, "arm", "?"))

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "metrics.json").write_text(json.dumps(board, indent=2),
                                              encoding="utf-8")
        with (out_dir / "per_sample.jsonl").open("w") as fh:
            for mode, rs in per_context.items():
                for r in rs:
                    fh.write(json.dumps(r) + "\n")
        board["_rows"] = per_context
    else:
        board["_rows"] = per_context
    return board


def _routing_confusion(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for r in rows:
        if r.get("uncovered"):
            continue
        fam = str(r.get("family"))
        head = str(r.get("head"))
        out.setdefault(fam, {}).setdefault(head, 0)
        out[fam][head] += 1
    return out
