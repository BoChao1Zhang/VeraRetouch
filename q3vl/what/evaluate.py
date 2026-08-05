"""Protocol 12 -- evaluation, gates and the ``V_what`` selection rule.

    12.4  "Select checkpoints only on ``V_what``:
           1. first satisfy the bake gate, the finite gate and the positive
              instruction-shuffle dependence;
           2. primary ordering: local final-image median CIEDE2000;
           3. secondary: LUT function CIEDE2000 p90;
           4. then boundary-band CIEDE2000, parameter count and latency.
           Only one checkpoint per arm enters the cross-arm ranking, and the
           final top-2 must be two different configurations rather than two
           adjacent steps of the same arm."

Two protocol rules are enforced by this module rather than by convention:

* the oracle-where arms ``C03``/``C04`` are *excluded from the main board*
  (protocol 8.2: "oracle inputs exist only in the ceiling control"); they are
  reported next to it as a ceiling;
* ``T_final`` and ``T_lut_unseen`` are opened once, after every selection is
  frozen.  :func:`main_board` refuses a split it was not given permission for,
  so "we peeked at T_final" has to be a deliberate argument, not a default.

The final composite of protocol 12.2 uses the frozen Where mask even for the
strict no-where arms ``C01``/``C02``.  Those controls bound what Stage-*What*
can do without Where *input*; giving them a different renderer would confound the
control with a mask ablation.  The choice is recorded in every row as
``composite_mask``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import (
    BAKE_GATE,
    CEILING_ARM_IDS,
    CONTEXT_MODES,
    FINITE_GATE,
    SELECTION_CONTEXT,
    GATE_FAILED_TAG,
    INSTRUCTION_SHUFFLE_MIN_DELTA,
    SELECTION_ORDER,
    STRATA_KEYS,
    BAKE_SIZE,
)
from .gaussians import bake, render
from .lut import lattice_points, tetra_lookup
from .metrics import (
    bake_metrics,
    compose_image,
    evaluate_gates,
    image_metrics,
    lexicographic_best,
    lut_metrics,
    percentile,
)

__all__ = ["sample_row", "summarise", "strata_report", "arm_metrics", "main_board",
           "write_per_sample", "ceiling_board", "context_report"]


@torch.no_grad()
def sample_row(params_i: Mapping[str, torch.Tensor], target: Mapping[str, Any],
               *, lut_cfg, table=None, gt_interp: str = "trilinear",
               i_in: torch.Tensor | None = None,
               i_tar: torch.Tensor | None = None,
               mask: torch.Tensor | None = None, mask_source: str | None = None,
               full_grid: bool = False, lpips_fn=None) -> dict[str, Any]:
    """All of protocol 12.1/12.2 for one sample.  ``params_i`` is batch size 1."""
    x = target["x"].unsqueeze(0)
    t_pred = render(params_i, x, lut_cfg)
    row: dict[str, Any] = {
        "sample_id": target["sample_id"], "lut_id": target.get("lut_id"),
        **{k: v for k, v in (target.get("meta") or {}).items() if k in STRATA_KEYS
           or k in ("build", "render_mode", "winner_confidence")},
    }
    kind = target["query_kind"]
    for i, name in enumerate(("uniform", "natural")):
        sel = kind == i
        if bool(sel.any()):
            row.update(lut_metrics(t_pred[0][sel], target["t_gt"][sel], f"{name}_"))
    row.update(lut_metrics(t_pred[0], target["t_gt"], "lut_"))

    cube = bake(params_i, lut_cfg, BAKE_SIZE)
    row.update(bake_metrics(t_pred, tetra_lookup(cube, x)))
    if full_grid and table is not None:
        pts = lattice_points(BAKE_SIZE, x.device, x.dtype).unsqueeze(0)
        row.update(lut_metrics(render(params_i, pts, lut_cfg)[0],
                               table.apply(pts[0], gt_interp), "grid33_"))

    if i_in is not None and i_tar is not None:
        m = mask if mask is not None else torch.ones(i_in.shape[-2:], device=i_in.device)
        t_img = render(params_i, i_in.reshape(3, -1).t().unsqueeze(0), lut_cfg)
        t_img = t_img[0].t().reshape(i_in.shape)
        i_out = compose_image(i_in, t_img, m)
        row.update({f"img_{k}": v for k, v in
                    image_metrics(i_out, i_tar, m, lpips_fn=lpips_fn).items()})
        # review nit N-5: D-W6 makes this field an audit record, so it may not
        # guess.  C03/C04 composite with the *GT* mask and would have been logged
        # as "predicted" by a two-valued inference.
        if mask_source is None:
            mask_source = "predicted" if mask is not None else "ones"
        row["composite_mask"] = mask_source
    return row


def summarise(rows: Sequence[Mapping[str, Any]], prefix: str = "") -> dict[str, Any]:
    """Mean / median / p90 of every numeric column, plus the counts."""
    if not rows:
        return {f"{prefix}n": 0}
    keys = sorted({k for r in rows for k in r
                   if isinstance(r.get(k), (int, float)) and not isinstance(r.get(k), bool)})
    out: dict[str, Any] = {f"{prefix}n": len(rows)}
    for k in keys:
        vals = torch.tensor([float(r[k]) for r in rows if k in r], dtype=torch.float64)
        vals = vals[torch.isfinite(vals)]
        if not vals.numel():
            continue
        out[f"{prefix}{k}_mean"] = float(vals.mean())
        out[f"{prefix}{k}_median"] = percentile(vals, 50)
        out[f"{prefix}{k}_p90"] = percentile(vals, 90)
    return out


def strata_report(rows: Sequence[Mapping[str, Any]],
                  keys: Sequence[str] = STRATA_KEYS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in keys:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            if k in r:
                groups.setdefault(str(r[k]), []).append(r)
        if groups:
            out[k] = {g: summarise(v) for g, v in sorted(groups.items())}
    return out


def arm_metrics(rows: Sequence[Mapping[str, Any]], *, arm: str, step: int,
                n_trainable: int, latency_ms: float | None = None,
                shuffle_rows: Sequence[Mapping[str, Any]] | None = None,
                context: str = SELECTION_CONTEXT) -> dict[str, Any]:
    """One checkpoint's board row: gates, the selection keys and the strata.

    ``context`` is amendment A-4's teacher/generated dimension.  A checkpoint
    produces **one row per context**; they are never averaged together, and
    :func:`main_board` selects on the generated one.  Reporting a single blended
    number would hide exactly the quantity A-4 exists to expose -- how much the
    arm loses when it has to read its own colour reasoning instead of the GT one.
    """
    if context not in CONTEXT_MODES:
        raise ValueError(f"unknown context {context!r}; have {CONTEXT_MODES}")
    local = [r for r in rows if r.get("render_mode") == "local"]
    glob = [r for r in rows if r.get("render_mode") == "global"]
    m: dict[str, Any] = {
        "arm": arm, "step": step, "context": context,
        "n_trainable_params": n_trainable,
        "latency_ms": latency_ms, "n": len(rows),
        "is_ceiling": arm in CEILING_ARM_IDS,
    }
    m.update(summarise(rows))
    m.update(summarise(local, "local_"))
    m.update(summarise(glob, "global_"))
    # protocol 12.4's ordering keys, named exactly as SELECTION_ORDER expects
    m["local_image_de00_median"] = m.get("local_img_de00_median_median")
    # review nit N-7: "LUT function CIEDE2000 p90" has two readings.  Both are
    # reported; the selection key is the per-sample-p90 median, and the
    # pooled-distribution p90 sits next to it under an unambiguous name.
    m["lut_de00_p90"] = m.get("lut_de00_p90_median")
    m["lut_de00_p90_of_sample_means"] = m.get("lut_de00_mean_p90")
    m["boundary_de00_median"] = m.get("local_img_boundary_de00_median_median")
    # the bake gate reads the *worst* sample, not the average of the averages
    for key, agg in (("bake_mae_mean", "mean"), ("bake_err_p99", "max"),
                     ("bake_non_finite", "max")):
        vals = [float(r[key]) for r in rows if key in r]
        if vals:
            m[key] = sum(vals) / len(vals) if agg == "mean" else max(vals)
    nf = [float(r["lut_non_finite"]) for r in rows if "lut_non_finite" in r]
    m["lut_non_finite"] = max(nf) if nf else 0.0

    gates = evaluate_gates(m, BAKE_GATE + FINITE_GATE)
    m["gates"] = gates
    if shuffle_rows is not None:
        # review nit N-8: the same samples are evaluated twice, so the paired
        # median of the per-sample differences is free and strictly stronger than
        # the difference of two aggregate medians.  The unpaired number is kept
        # beside it because that is what the pre-registered threshold was set on.
        by_id = {r["sample_id"]: r for r in shuffle_rows}
        pairs = [float(by_id[r["sample_id"]]["lut_de00_mean"]) - float(r["lut_de00_mean"])
                 for r in local
                 if r.get("sample_id") in by_id and "lut_de00_mean" in r
                 and "lut_de00_mean" in by_id[r["sample_id"]]]
        base = summarise(local).get("lut_de00_mean_median")
        shuf = summarise([r for r in shuffle_rows
                          if r.get("render_mode") == "local"]).get("lut_de00_mean_median")
        unpaired = (shuf - base) if (base is not None and shuf is not None) else None
        paired = percentile(torch.tensor(pairs), 50) if pairs else None
        m["instruction_shuffle_delta_unpaired"] = unpaired
        m["instruction_shuffle_n_paired"] = len(pairs)
        delta = paired if paired is not None else unpaired
        m["instruction_shuffle_delta"] = delta
        m["instruction_shuffle_pass"] = bool(
            delta is not None and delta >= INSTRUCTION_SHUFFLE_MIN_DELTA)
        gates["all_pass"] = gates["all_pass"] and m["instruction_shuffle_pass"]
    m["gate_pass"] = gates["all_pass"]
    if not gates["all_pass"]:
        m["tag"] = GATE_FAILED_TAG
    m["strata"] = strata_report(rows)
    return m


def _best_per_arm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Protocol 12.4: "only one checkpoint per arm enters the cross-arm ranking".

    Ties keep the *first* row seen, i.e. the earlier step when ``rows`` arrive in
    step order.  Stated rather than left to chance (review nit N-15): an earlier
    checkpoint at identical metrics is the cheaper and the more conservative pick.
    """
    best: dict[str, Mapping[str, Any]] = {}
    for r in rows:
        cur = best.get(r["arm"])
        if cur is None:
            best[r["arm"]] = r
        elif lexicographic_best([cur, r], SELECTION_ORDER) is r:
            best[r["arm"]] = r
    return best


def _rank(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(rows, key=lambda r: tuple(
        (-float(r[k]) if big else float(r[k]))
        if r.get(k) is not None else float("inf")
        for k, big in SELECTION_ORDER))


def _summary(r: Mapping[str, Any]) -> dict[str, Any]:
    return {"arm": r["arm"], "step": r["step"], "context": r.get("context"),
            **{k: r.get(k) for k, _ in SELECTION_ORDER},
            "gate_pass": r.get("gate_pass"),
            "instruction_shuffle_pass": r.get("instruction_shuffle_pass")}


def main_board(candidates: Iterable[Mapping[str, Any]], *, split: str,
               allow: Sequence[str] = ("V_what",),
               context: str = SELECTION_CONTEXT) -> dict[str, Any]:
    """Protocol 12.4 selection: **gate first, then the lexicographic order**.

        "1. first satisfy the bake gate, the finite gate and the positive
            instruction-shuffle dependence;
         2. primary ordering: local final-image median CIEDE2000; ..."

    Step 1 is a **filter, not a label** (review blocker B-1).  A checkpoint that
    fails the gate cannot be selected, so it does not appear in ``ranked`` at all;
    it goes into ``gate_failed`` instead.  Before this was enforced, a gate-failing
    arm with a better CIEDE2000 came out on top of the board while still carrying
    ``gate_pass: False`` in its own row -- the failure was recorded and ignored in
    the same object.

    When *no* checkpoint passes, the board follows protocol 5.6's shape for the
    Where gate: a ``diagnostic_ranked`` list is still produced so the run can be
    debugged, ``ranked`` stays empty so nothing can be selected by accident, and
    the whole board is tagged ``WHAT-GATE-FAILED``.  Protocol 15 then applies:
    report by the stage that failed, do not relax the gate after the fact.
    """
    if split not in allow:
        raise PermissionError(
            f"selection may only run on {list(allow)}; {split!r} was requested. "
            "T_final and T_lut_unseen are opened once, after every selection is "
            "frozen (protocol 12.4)."
        )
    if context not in CONTEXT_MODES:
        raise ValueError(f"unknown context {context!r}; have {CONTEXT_MODES}")
    rows = [c for c in candidates]
    # amendment A-4 item 3: selection reads the generated-context board.  A row
    # that carries no context at all predates A-4 and is refused rather than
    # silently treated as generated.
    unlabelled = [r for r in rows if r.get("context") is None]
    if unlabelled:
        raise ValueError(
            f"{len(unlabelled)} candidate rows carry no 'context' field; "
            "amendment A-4 requires every board row to declare teacher or "
            "generated context (arm_metrics writes it)."
        )
    board = [r for r in rows
             if not r.get("is_ceiling") and r.get("context") == context]
    passed = [r for r in board if r.get("gate_pass")]
    failed = [r for r in board if not r.get("gate_pass")]

    best_pass = _best_per_arm(passed)
    ranked = _rank(best_pass.values())
    # an arm with no gate-passing checkpoint is listed once, by its own best step
    best_fail = {a: r for a, r in _best_per_arm(failed).items() if a not in best_pass}
    gate_failed = _rank(best_fail.values())

    top2 = ranked[:2]
    out: dict[str, Any] = {
        "split": split,
        "context": context,
        "n_candidates": len(rows),
        "n_candidates_in_context": len(board),
        "n_checkpoints_gate_pass": len(passed),
        "n_checkpoints_gate_failed": len(failed),
        "n_arms": len(best_pass),
        "n_arms_gate_failed": len(best_fail),
        "ranked": [_summary(r) for r in ranked],
        "gate_failed": [_summary(r) for r in gate_failed],
        "top2": [{"arm": r["arm"], "step": r["step"]} for r in top2],
        "top2_distinct_arms": len({r["arm"] for r in top2}) == len(top2),
        "any_gate_failed": bool(failed),
        "selection_possible": bool(ranked),
    }
    if not ranked:
        # protocol 5.6's shape, transposed to What: still produce an ordering for
        # diagnosis, but refuse to call any of it a selection.
        out["diagnostic_ranked"] = [_summary(r) for r in _rank(_best_per_arm(board).values())]
        out["tag"] = GATE_FAILED_TAG
        out["top2"] = []
        out["top2_distinct_arms"] = False
    return out


def ceiling_board(candidates: Iterable[Mapping[str, Any]],
                  context: str = SELECTION_CONTEXT) -> list[dict[str, Any]]:
    """``C03``/``C04`` reported *beside* the board, never inside it."""
    return [{"arm": c["arm"], "step": c["step"], "context": c.get("context"),
             **{k: c.get(k) for k, _ in SELECTION_ORDER}}
            for c in candidates
            if c.get("is_ceiling") and c.get("context") == context]


def context_report(candidates: Iterable[Mapping[str, Any]],
                   metric: str | None = None) -> dict[str, Any]:
    """Amendment A-4 item 2: the two contexts side by side, per arm.

    ``gap`` is generated minus teacher on the primary selection key.  A large
    positive gap is the honest reading of "this arm depends on GT colour
    reasoning"; a gap near zero is what protocol 0's claim needs.  Neither is
    visible if the two contexts are averaged, which is why they never are.
    """
    rows = list(candidates)
    by: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault((r["arm"], r["step"]), {})[r.get("context")] = r
    key = metric or SELECTION_ORDER[0][0]
    out = []
    for (arm, step), ctxs in sorted(by.items()):
        gt, gen = ctxs.get("gt"), ctxs.get("generated")
        row: dict[str, Any] = {"arm": arm, "step": step,
                               "gt": gt.get(key) if gt else None,
                               "generated": gen.get(key) if gen else None,
                               "metric": key}
        if gt is not None and gen is not None and gt.get(key) is not None \
                and gen.get(key) is not None:
            row["gap"] = float(gen[key]) - float(gt[key])
        out.append(row)
    return {"metric": key, "rows": out,
            "n_pairs": sum(1 for r in out if r.get("gap") is not None)}


def write_per_sample(rows: Iterable[Mapping[str, Any]], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    return path
