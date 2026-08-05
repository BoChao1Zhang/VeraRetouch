"""Protocol 5.6 -- the nine Where gates and the lexicographic selection rule.

| metric                                  | gate     |
|-----------------------------------------|---------:|
| local `.cgt` median soft-IoU            | `>= 0.75` |
| soft-IoU relative to the per-image oracle | `>= 85%` |
| local soft-IoU p10                      | `>= 0.55` |
| `AUC_target`                            | `>= 0.80` |
| 3px boundary F1 / oracle boundary F1    | `>= 75%` |
| IoU drop after instruction shuffle      | `>= 0.20` |
| median `std(s_pred) / std(s*)`          | `>= 0.60` |
| global mask soft-IoU                    | `>= 0.98` |
| GT vs generated context IoU gap         | `<= 0.05` |

    "Once the gates pass, the single frozen Where checkpoint is selected in this
     order: generated-context local median soft-IoU; 3px boundary F1; p10
     soft-IoU; parameter count, peak memory and latency.  If no arm passes every
     gate, the lexicographic best is still selected for diagnosis but must be
     tagged ``WHERE-GATE-FAILED``, and the downstream What results may not claim
     that the complete method holds."

All of these are computed on ``V_where`` with the **generated** context as the
main board (protocol 5.4), which is why :func:`arm_metrics` takes a per-context
mapping and refuses to average the contexts together.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import (
    GATES,
    GATE_FAILED_TAG,
    GRID_BOUNDARY_TOL_CELLS,
    SELECTION_ORDER,
)
from .losses import _EPS, boundary_map, soft_iou

__all__ = ["percentile", "soft_iou_value", "boundary_f1", "topk_mask",
           "gt_area_k", "center_prior_field", "hard_iou", "grid_boundary_f1",
           "paired_delta", "sample_metrics", "summarise", "arm_metrics",
           "evaluate_gates", "lexicographic_best", "ATTRIBUTION_NOTE",
           "attribution_section", "context_deltas"]


def percentile(xs: Sequence[float], p: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    if p <= 0:
        return s[0]
    if p >= 1:
        return s[-1]
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def soft_iou_value(m: torch.Tensor, t: torch.Tensor) -> float:
    return float(soft_iou(m.reshape(-1).double(), t.reshape(-1).double()))


def boundary_f1(m: torch.Tensor, t: torch.Tensor, kernel: int = 3, tol_px: int = 3) -> float:
    """The *metric* (not the loss): 2PR/(P+R) with a 3px tolerance."""
    import torch.nn.functional as F

    def _as4(x):
        return x.reshape(1, 1, *x.shape[-2:]).double()

    m4, t4 = _as4(m), _as4(t)
    bp, bt = boundary_map(m4, kernel), boundary_map(t4, kernel)
    ext = 2 * tol_px + 1
    pool = lambda x: F.max_pool2d(x, kernel_size=ext, stride=1, padding=ext // 2)  # noqa: E731
    sp, st = float(bp.sum()), float(bt.sum())
    if sp < 1e-3 and st < 1e-3:
        return 1.0                       # no boundary on either side = agreement
    prec = float((bp * pool(bt)).sum()) / (sp + _EPS)
    rec = float((bt * pool(bp)).sum()) / (st + _EPS)
    return 2 * prec * rec / (prec + rec + _EPS)


def topk_mask(field: torch.Tensor, k: int) -> torch.Tensor:
    """Binarise a field by taking its top ``k`` cells.

    Red line (CLAUDE.md 2026-08-05): thresholding is ALWAYS "top-k matching the
    GT area", never a per-field tuned threshold.  A tuned threshold lets a field
    buy coverage it did not earn, and makes two fields incomparable.
    """
    flat = field.reshape(-1)
    k = int(max(0, min(k, flat.numel())))
    out = torch.zeros_like(flat)
    if k:
        out[torch.topk(flat, k).indices] = 1.0
    return out.reshape(field.shape)


def gt_area_k(m_gt: torch.Tensor, threshold: float = 0.5) -> int:
    """``k`` = the number of cells the GT region occupies (the matched area)."""
    return int((m_gt.reshape(-1) > threshold).sum())


def center_prior_field(grid_h: int, grid_w: int, *, device=None,
                       dtype=torch.float32) -> torch.Tensor:
    """The zero-parameter baseline: ``-distance to the frame centre``.

    Red line: every spatial field must be reported next to this, on the SAME
    support and under the SAME top-k rule.  Measured precedent (RO-9c supplement
    B/D): this field scores AUC 0.836 and beats all six RO-9c attention readouts
    (0.695-0.784) while containing no information whatsoever -- which is why AUC
    is banned and why any "our field found the subject" claim has to show a
    paired delta against this column.

    Coordinates use the same true-aspect-ratio convention as the rest of the
    stage (``q3vl.where.phi.norm_coords``), so the distance is isotropic in
    image space rather than in cell index space.
    """
    from q3vl.where.phi import norm_coords

    X, Y = norm_coords(grid_h, grid_w, device=device, dtype=dtype)
    return -torch.sqrt(X ** 2 + Y ** 2)


def hard_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """IoU of two binary masks."""
    p = pred_mask.reshape(-1) > 0.5
    g = gt_mask.reshape(-1) > 0.5
    inter = float((p & g).sum())
    union = float((p | g).sum())
    return inter / union if union else 1.0


def grid_boundary_f1(pred_mask: torch.Tensor, gt_mask: torch.Tensor,
                     tol_cells: int = GRID_BOUNDARY_TOL_CELLS) -> float:
    """Boundary F1 on the **grid**, on binary masks, with a cell-level tolerance.

    Red line: the pixel-level 3px boundary F1 is banned as a criterion.  On the
    upsampled pixel grid a random top-k field scores 0.0394 against a centre
    prior's 0.0327 -- it mostly measures boundary *length*, so a shredded field
    with lots of edge wins.  On the ``F_pre`` grid, with masks binarised by the
    matched-area top-k rule, there is no length to farm: both fields own exactly
    ``k`` cells, and the score is about where those cells are.

    ``L_mask`` still uses the pixel-level 3px term (protocol 5.5) -- amendment
    A-5 changes the *criterion*, not the optimisation target (protocol 9.5/10.4
    forbid changing a loss after the fact).
    """
    import torch.nn.functional as F

    def edges(mask: torch.Tensor) -> torch.Tensor:
        m = (mask.reshape(1, 1, *mask.shape[-2:]) > 0.5).float()
        inv = 1.0 - m
        return ((F.max_pool2d(inv, 3, stride=1, padding=1) - inv) * m).clamp(0, 1)

    bp, bt = edges(pred_mask), edges(gt_mask)
    sp, st = float(bp.sum()), float(bt.sum())
    if sp < 0.5 and st < 0.5:
        return 1.0
    if sp < 0.5 or st < 0.5:
        return 0.0
    ext = 2 * tol_cells + 1
    pool = lambda x: F.max_pool2d(x, ext, stride=1, padding=ext // 2)  # noqa: E731
    prec = float((bp * pool(bt)).sum()) / sp
    rec = float((bt * pool(bp)).sum()) / st
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def paired_delta(a: Sequence[float], b: Sequence[float], *, n_boot: int = 2000,
                 seed: int = 0) -> dict[str, Any]:
    """Paired mean ``a - b`` with a bootstrap CI and a two-sided p-value.

    Red line: "any claim that the field found the subject must show a paired
    delta and a p-value" -- against the centre prior, on the same samples.
    """
    import random as _random

    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "delta": None, "p_value": None, "ci95": None}
    diffs = [x - y for x, y in pairs]
    obs = sum(diffs) / n
    rng = _random.Random(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    # two-sided bootstrap p: how often the resampled mean crosses zero
    n_le = sum(1 for v in boots if v <= 0.0)
    p = 2.0 * min(n_le, n_boot - n_le) / n_boot
    return {"n": n, "delta": obs, "p_value": min(1.0, p), "ci95": [lo, hi],
            "n_boot": n_boot}


def sample_metrics(
    m_pred: torch.Tensor,
    m_gt: torch.Tensor,
    *,
    s_pred: torch.Tensor | None = None,
    s_star: torch.Tensor | None = None,
    m_oracle: torch.Tensor | None = None,
    grid_pred: torch.Tensor | None = None,
    grid_gt: torch.Tensor | None = None,
    grid_oracle: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Per-sample criteria columns (amendment A-5).

    Three mandatory columns for any spatial field, none optional:
      1. soft-IoU / hard-IoU  -- is the coverage right
      2. grid boundary F1     -- does the shape follow
      3. centre-prior baseline on the same support and the same top-k rule

    ``AUC_target`` is **not produced**: red line, 2026-08-05.  It was fooled
    three separate ways (subject prior 0.907 vs AUC_target 0.523; a zero-
    parameter centre prior at 0.836 beating every RO-9c readout; and MCQ-L where
    AUC moved 0.947/0.950/0.948 while soft-IoU moved 0.605/0.508/0.509).

    ``grid_*`` arguments carry the ``F_pre``-grid fields, which is where the
    grid-level and centre-prior columns are computed.  The hi-res soft-IoU stays
    because it is the coverage number the gate is written against.
    """
    out: dict[str, Any] = {
        "soft_iou": soft_iou_value(m_pred, m_gt),
        "pred_mean": float(m_pred.mean()),
        "pred_std": float(m_pred.std(unbiased=False)),
        "gt_mean": float(m_gt.mean()),
    }
    if s_pred is not None and s_star is not None:
        out["s_std_ratio"] = float(
            s_pred.std(unbiased=False) / (s_star.std(unbiased=False) + _EPS)
        )
    if m_oracle is not None:
        out["oracle_soft_iou"] = soft_iou_value(m_oracle, m_gt)

    if grid_pred is not None and grid_gt is not None:
        gp = grid_pred.reshape(grid_gt.shape)
        k = gt_area_k(grid_gt)
        gt_bin = (grid_gt > 0.5).float()
        pred_bin = topk_mask(gp, k)
        prior = center_prior_field(*grid_gt.shape[-2:], device=gp.device,
                                   dtype=torch.float32).to(gp.dtype)
        prior_bin = topk_mask(prior, k)
        out.update({
            "grid_k": k,
            "grid_soft_iou": soft_iou_value(gp, grid_gt),
            "grid_hard_iou": hard_iou(pred_bin, gt_bin),
            "grid_boundary_f1": grid_boundary_f1(pred_bin, gt_bin),
            # the zero-parameter baseline, same support, same k
            "center_prior_hard_iou": hard_iou(prior_bin, gt_bin),
            "center_prior_boundary_f1": grid_boundary_f1(prior_bin, gt_bin),
        })
        if grid_oracle is not None:
            og = topk_mask(grid_oracle.reshape(grid_gt.shape), k)
            out["oracle_grid_hard_iou"] = hard_iou(og, gt_bin)
            out["oracle_grid_boundary_f1"] = grid_boundary_f1(og, gt_bin)
    return out


def summarise(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample rows, split into local / global as protocol 12.2 asks.

    Amendment A-5: no AUC column; grid-level boundary F1 replaces the pixel-level
    3px one as a criterion; the centre-prior baseline is reported next to every
    field column, with a paired delta and a p-value.
    """
    rows = list(rows)
    def pick(key, pred=lambda r: True):
        return [float(r[key]) for r in rows if pred(r) and r.get(key) is not None]

    is_local = lambda r: r.get("render_mode", "local") == "local"   # noqa: E731
    is_global = lambda r: r.get("render_mode") == "global"          # noqa: E731

    loc = pick("soft_iou", is_local)
    glo = pick("soft_iou", is_global)
    gbf = pick("grid_boundary_f1", is_local)
    ghi = pick("grid_hard_iou", is_local)
    ratio = [float(r["soft_iou"]) / float(r["oracle_soft_iou"])
             for r in rows if is_local(r) and r.get("oracle_soft_iou")]
    bratio = [float(r["grid_boundary_f1"]) / float(r["oracle_grid_boundary_f1"])
              for r in rows if is_local(r) and r.get("oracle_grid_boundary_f1")]
    sratio = pick("s_std_ratio", is_local)

    # paired against the centre prior, on the samples that have both
    local_rows = [r for r in rows if is_local(r)
                  and r.get("grid_hard_iou") is not None
                  and r.get("center_prior_hard_iou") is not None]
    d_iou = paired_delta([r["grid_hard_iou"] for r in local_rows],
                         [r["center_prior_hard_iou"] for r in local_rows])
    d_bf1 = paired_delta([r["grid_boundary_f1"] for r in local_rows],
                         [r["center_prior_boundary_f1"] for r in local_rows])
    return {
        "n": len(rows), "n_local": sum(1 for r in rows if is_local(r)),
        "n_global": sum(1 for r in rows if is_global(r)),
        "local_soft_iou_median": percentile(loc, 0.5),
        "local_soft_iou_mean": (sum(loc) / len(loc)) if loc else None,
        "local_soft_iou_p10": percentile(loc, 0.10),
        "local_soft_iou_p90": percentile(loc, 0.90),
        "global_soft_iou": percentile(glo, 0.5),
        "grid_boundary_f1": percentile(gbf, 0.5),
        "grid_hard_iou": percentile(ghi, 0.5),
        "soft_iou_vs_oracle_ratio": percentile(ratio, 0.5),
        "grid_boundary_f1_vs_oracle_ratio": percentile(bratio, 0.5),
        "s_std_ratio_median": percentile(sratio, 0.5),
        # --- the mandatory centre-prior column (amendment A-5) --------------
        "center_prior_hard_iou": percentile(
            pick("center_prior_hard_iou", is_local), 0.5),
        "center_prior_boundary_f1": percentile(
            pick("center_prior_boundary_f1", is_local), 0.5),
        "center_prior_delta_hard_iou": d_iou["delta"],
        "center_prior_delta_hard_iou_p": d_iou["p_value"],
        "center_prior_delta_hard_iou_ci95": d_iou["ci95"],
        "center_prior_delta_boundary_f1": d_bf1["delta"],
        "center_prior_delta_boundary_f1_p": d_bf1["p_value"],
        "center_prior_delta_boundary_f1_ci95": d_bf1["ci95"],
    }


def arm_metrics(
    per_context: Mapping[str, Mapping[str, Any]],
    *,
    main_context: str = "generated",
    resources: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten the four context summaries into the gate namespace.

    The GT/generated gap and the instruction-shuffle drop are the only two
    metrics that *span* contexts; everything else is read off the main board.
    """
    if main_context not in per_context:
        raise KeyError(f"main context {main_context!r} missing from {sorted(per_context)}")
    main = dict(per_context[main_context])
    out = dict(main)
    out.update(context_deltas(per_context, main_context))
    out["per_context"] = {k: dict(v) for k, v in per_context.items()}
    out["main_context"] = main_context
    out["attribution"] = ATTRIBUTION_NOTE
    if resources:
        out.update({k: v for k, v in resources.items()})
    return out


#: Ruling D-B1 + the reviewer's addendum.  ``1 - softIoU`` is the dominant term
#: of the optimised loss AND the first key of the §5.6 selection order, so
#: ``local_soft_iou_median`` is not an independent test -- passing its gate says
#: "training converged", nothing more.  Attribution weight has to fall on the
#: quantities that are NOT directly optimised.  This block is emitted into every
#: ``metrics.json`` so no report can quietly forget it.
ATTRIBUTION_NOTE: dict[str, Any] = {
    "ruling": "D-B1 (2026-08-05) + REVIEW-impl-WhereB §四",
    "directly_optimised": {
        "local_soft_iou_median": (
            "1 - softIoU is the weight-1.00 term of L_mask and the first key of "
            "SELECTION_ORDER; its gate (>= 0.75) tests convergence, not a claim"
        ),
        "local_soft_iou_p10": "same objective, tail of the same distribution",
    },
    "weakly_optimised": {
        "grid_boundary_f1": "L_mask has a PIXEL-level 3px term at weight 0.10; the\n            criterion is the GRID-level one, which nothing optimises directly",
        "s_std_ratio_median": "constrained by L_s at 1.00 -> 0.25 across stages",
    },
    "not_optimised": {
        "center_prior_delta_hard_iou": "the zero-parameter baseline; nothing\n            optimises the margin over it",
        "instruction_shuffle_iou_drop": "causal control; nothing optimises it",
        "null_context_gap": "causal control; nothing optimises it",
        "gt_generated_iou_gap": "cross-context consistency; nothing optimises it",
    },
    "fair_comparisons": {
        "soft_iou_vs_oracle_ratio": (
            "numerator and denominator are optimised for the same objective, so "
            "the ratio remains a fair headroom measure"
        ),
    },
    "reporting_rule": (
        "REPORT.md must carry the attribution weight on boundary_f1 / "
        "local_soft_iou_p10 / grid_boundary_f1 / the centre-prior paired delta "
        "and the null- and shuffled-context "
        "deltas, and must NOT present local_soft_iou_median as independent "
        "evidence that Where-B works."
    ),
}


def attribution_section(metrics: Mapping[str, Any]) -> str:
    """The mandatory REPORT.md section, rendered from the arm's own numbers.

    Emitted next to ``metrics.json`` by :func:`q3vl.whereb.evaluate.evaluate_arm`
    so the D-B1 consequence travels with every board instead of depending on
    whoever writes the report remembering it.
    """
    def fmt(key: str) -> str:
        v = metrics.get(key)
        return "n/a" if v is None else f"{float(v):.4f}"

    rows = [
        ("local_soft_iou_median", "1.00 (dominant)", "convergence only -- NOT independent evidence"),
        ("local_soft_iou_p10", "1.00 (same objective)", "tail of the same distribution"),
        ("grid_boundary_f1", "0 (the criterion is grid-level; L_mask's 3px term is pixel-level)",
         "PRIMARY attribution: does the shape follow?"),
        ("s_std_ratio_median", "via L_s, 1.00 -> 0.25", "collapse probe on the s axis"),
        ("center_prior_hard_iou", "0", "zero-parameter baseline (-distance to centre)"),
        ("center_prior_delta_hard_iou", "0", "PRIMARY attribution: margin over the baseline"),
        ("center_prior_delta_hard_iou_p", "0", "PRIMARY: the paired p-value that margin needs"),
        ("instruction_shuffle_iou_drop", "0", "PRIMARY attribution: instruction dependence"),
        ("null_context_gap", "0", "PRIMARY attribution: what the language context adds"),
        ("gt_generated_iou_gap", "0", "consistency between the two training contexts"),
        ("soft_iou_vs_oracle_ratio", "same objective on both sides", "fair headroom measure"),
    ]
    body = "\n".join(
        f"| `{k}` | {fmt(k)} | {w} | {role} |" for k, w, role in rows
    )
    return (
        f"## 归因纪律（arm {metrics.get('arm', '?')}，主榜 = {metrics.get('main_context')} 上下文）\n\n"
        "**AUC 已按 2026-08-05 红线全实验禁用**（本表不含任何 AUC 列）。\n\n"
        "**`local_soft_iou_median` 不是独立检验。** 裁定 D-B1 保留了协议 §5.5 的字面写法，\n"
        "于是 `1 - softIoU` 同时是优化损失里权重 **1.00** 的支配项、以及 §5.6 选择规则的\n"
        "**第一顺位**。因此「local median soft-IoU >= 0.75」这条 gate 实质上只回答\n"
        "「训练收敛了吗」，**不能**作为「Where-B 成立」的证据。归因重量必须落在\n"
        "**没有被直接优化**的量上。\n\n"
        "| 指标 | 实测 | 在 loss 里的权重 | 报告中的角色 |\n"
        "|---|---:|---|---|\n"
        f"{body}\n\n"
        "写作要求：结论段落里每出现一次 `local_soft_iou_median`，必须同时给出\n"
        "`grid_boundary_f1` / `center_prior_delta_*` / instruction-shuffle 与 null 两个 delta 的对应数字；\n"
        "不得只用前者作结。\n"
    )


def context_deltas(per_context: Mapping[str, Mapping[str, Any]],
                   main_context: str = "generated") -> dict[str, Any]:
    """The causal-control deltas: main - null and main - shuffled."""
    main = per_context.get(main_context, {})
    out: dict[str, Any] = {}
    base = main.get("local_soft_iou_median")
    for other, name in (("null", "null_context_gap"),
                        ("shuffled", "instruction_shuffle_iou_drop")):
        v = per_context.get(other, {}).get("local_soft_iou_median")
        if base is not None and v is not None:
            out[name] = float(base) - float(v)
    gt = per_context.get("gt", {}).get("local_soft_iou_median")
    if base is not None and gt is not None:
        out["gt_generated_iou_gap"] = abs(float(gt) - float(base))
    return out


def evaluate_gates(metrics: Mapping[str, Any],
                   gates: Sequence[tuple[str, str, float]] = GATES) -> dict[str, Any]:
    rows = []
    for key, op, thr in gates:
        val = metrics.get(key)
        if val is None:
            rows.append({"metric": key, "op": op, "threshold": thr, "value": None,
                         "passed": False, "reason": "missing"})
            continue
        v = float(val)
        # A-5 added a strict ">" gate (the centre-prior margin must be positive,
        # not merely non-negative).  Dispatch explicitly and reject anything
        # unknown: the old two-way branch silently treated ">" as ">=", which
        # would have let a field that exactly ties the zero-parameter baseline
        # pass the one gate that exists to catch it.
        ops = {">=": lambda: v >= thr, ">": lambda: v > thr,
               "<=": lambda: v <= thr, "<": lambda: v < thr}
        if op not in ops:
            raise ValueError(f"unknown gate operator {op!r} for {key!r}")
        ok = ops[op]()
        rows.append({"metric": key, "op": op, "threshold": thr, "value": v,
                     "passed": bool(ok), "reason": ""})
    passed = all(r["passed"] for r in rows)
    return {
        "passed": passed,
        "n_passed": sum(r["passed"] for r in rows),
        "n_gates": len(rows),
        "rows": rows,
        "tag": None if passed else GATE_FAILED_TAG,
    }


def lexicographic_best(
    candidates: Sequence[Mapping[str, Any]],
    order: Sequence[tuple[str, bool]] = SELECTION_ORDER,
) -> dict[str, Any]:
    """Protocol 5.6 selection.  Gate failures do not remove a candidate -- they
    tag the winner -- because the protocol still wants a best-of for diagnosis."""
    if not candidates:
        raise ValueError("no candidates")

    def key(c: Mapping[str, Any]):
        out = []
        for name, larger_is_better in order:
            v = c.get(name)
            if v is None:
                out.append(float("inf"))          # missing sorts last either way
            else:
                out.append(-float(v) if larger_is_better else float(v))
        return tuple(out)

    ranked = sorted(candidates, key=key)
    best = dict(ranked[0])
    gate = evaluate_gates(best)
    any_passed = any(evaluate_gates(c)["passed"] for c in candidates)
    return {
        "best": best,
        "ranking": [c.get("arm") for c in ranked],
        "gate": gate,
        "tag": None if gate["passed"] else GATE_FAILED_TAG,
        "any_candidate_passed_all_gates": any_passed,
        "order": [list(o) for o in order],
    }
