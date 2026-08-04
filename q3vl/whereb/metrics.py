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

from .config import GATES, GATE_FAILED_TAG, SELECTION_ORDER
from .losses import _EPS, boundary_map, soft_iou

__all__ = ["percentile", "soft_iou_value", "boundary_f1", "auc_target",
           "sample_metrics", "summarise", "arm_metrics", "evaluate_gates",
           "lexicographic_best", "ATTRIBUTION_NOTE", "attribution_section",
           "context_deltas"]


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


def auc_target(m: torch.Tensor, t: torch.Tensor, threshold: float = 0.5) -> float | None:
    """ROC AUC of the predicted mask as a score for the binarised GT region.

    ``None`` when the GT is entirely inside or entirely outside the region (every
    global sample), because AUC is undefined with one class.
    """
    scores = m.reshape(-1).double()
    labels = (t.reshape(-1) > threshold).double()
    n_pos = float(labels.sum())
    n_neg = float(labels.numel() - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None
    order = torch.argsort(scores)
    ranks = torch.empty_like(scores)
    ranks[order] = torch.arange(1, scores.numel() + 1, dtype=scores.dtype,
                                device=scores.device)
    # average ranks over ties so a constant prediction scores exactly 0.5
    uniq, inv, counts = torch.unique(scores, return_inverse=True, return_counts=True)
    rank_sum = torch.zeros_like(uniq).scatter_add_(0, inv, ranks)
    ranks = (rank_sum / counts)[inv]
    r_pos = float((ranks * labels).sum())
    return (r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def sample_metrics(
    m_pred: torch.Tensor,
    m_gt: torch.Tensor,
    *,
    s_pred: torch.Tensor | None = None,
    s_star: torch.Tensor | None = None,
    m_oracle: torch.Tensor | None = None,
    tol_px: int = 3,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "soft_iou": soft_iou_value(m_pred, m_gt),
        "boundary_f1": boundary_f1(m_pred, m_gt, tol_px=tol_px),
        "auc_target": auc_target(m_pred, m_gt),
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
        out["oracle_boundary_f1"] = boundary_f1(m_oracle, m_gt, tol_px=tol_px)
    return out


def summarise(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-sample rows, split into local / global as protocol 12.2 asks."""
    rows = list(rows)
    def pick(key, pred=lambda r: True):
        return [float(r[key]) for r in rows if pred(r) and r.get(key) is not None]

    is_local = lambda r: r.get("render_mode", "local") == "local"   # noqa: E731
    is_global = lambda r: r.get("render_mode") == "global"          # noqa: E731

    loc = pick("soft_iou", is_local)
    glo = pick("soft_iou", is_global)
    bf = pick("boundary_f1", is_local)
    auc = pick("auc_target", is_local)
    ratio = [float(r["soft_iou"]) / float(r["oracle_soft_iou"])
             for r in rows if is_local(r) and r.get("oracle_soft_iou")]
    bratio = [float(r["boundary_f1"]) / float(r["oracle_boundary_f1"])
              for r in rows if is_local(r) and r.get("oracle_boundary_f1")]
    sratio = pick("s_std_ratio", is_local)
    return {
        "n": len(rows), "n_local": sum(1 for r in rows if is_local(r)),
        "n_global": sum(1 for r in rows if is_global(r)),
        "local_soft_iou_median": percentile(loc, 0.5),
        "local_soft_iou_mean": (sum(loc) / len(loc)) if loc else None,
        "local_soft_iou_p10": percentile(loc, 0.10),
        "local_soft_iou_p90": percentile(loc, 0.90),
        "global_soft_iou": percentile(glo, 0.5),
        "boundary_f1": percentile(bf, 0.5),
        "auc_target": percentile(auc, 0.5),
        "soft_iou_vs_oracle_ratio": percentile(ratio, 0.5),
        "boundary_f1_vs_oracle_ratio": percentile(bratio, 0.5),
        "s_std_ratio_median": percentile(sratio, 0.5),
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
        "boundary_f1": "enters L_mask at weight 0.10 only -> largely independent",
        "s_std_ratio_median": "constrained by L_s at 1.00 -> 0.25 across stages",
    },
    "not_optimised": {
        "auc_target": "no term in L touches ranking quality",
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
        "local_soft_iou_p10 / auc_target and the null- and shuffled-context "
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
        ("boundary_f1", "0.10 (weak)", "PRIMARY attribution: is the edge real?"),
        ("s_std_ratio_median", "via L_s, 1.00 -> 0.25", "collapse probe on the s axis"),
        ("auc_target", "0", "PRIMARY attribution: ranking quality, nothing optimises it"),
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
        "**`local_soft_iou_median` 不是独立检验。** 裁定 D-B1 保留了协议 §5.5 的字面写法，\n"
        "于是 `1 - softIoU` 同时是优化损失里权重 **1.00** 的支配项、以及 §5.6 选择规则的\n"
        "**第一顺位**。因此「local median soft-IoU >= 0.75」这条 gate 实质上只回答\n"
        "「训练收敛了吗」，**不能**作为「Where-B 成立」的证据。归因重量必须落在\n"
        "**没有被直接优化**的量上。\n\n"
        "| 指标 | 实测 | 在 loss 里的权重 | 报告中的角色 |\n"
        "|---|---:|---|---|\n"
        f"{body}\n\n"
        "写作要求：结论段落里每出现一次 `local_soft_iou_median`，必须同时给出\n"
        "`boundary_f1` / `auc_target` / instruction-shuffle 与 null 两个 delta 的对应数字；\n"
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
        ok = v >= thr if op == ">=" else v <= thr
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
