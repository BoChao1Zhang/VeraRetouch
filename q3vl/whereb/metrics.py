"""Protocol 5.6 as revised by amendment A-5 -- the Where gates and the
lexicographic selection rule.

| metric                                    | gate      |
|-------------------------------------------|----------:|
| local `.cgt` median soft-IoU              | `>= 0.75` |
| soft-IoU relative to the per-image oracle | `>= 85%`  |
| local soft-IoU p10                        | `>= 0.55` |
| **grid-level** boundary F1 / oracle       | `>= 75%`  |
| centre-prior paired delta (hard-IoU)      | `> 0`     |
| that delta's paired p-value               | `<= 0.05` |
| IoU drop after instruction shuffle        | `>= 0.20` |
| median `std(s_pred) / std(s*)`            | `>= 0.60` |
| global mask soft-IoU                      | `>= 0.98` |
| GT vs generated context IoU gap           | `<= 0.05` |

Ten gates.  Amendment A-5 (2026-08-05, user red lines in CLAUDE.md) deleted the
`AUC_target >= 0.80` row and does not produce the metric at all; replaced the
pixel-level "3px boundary F1" criterion with a **grid-level** one; and added the
zero-parameter centre-prior baseline with a paired delta and p-value.  The §5.5
**loss is untouched** -- it keeps its pixel-level 3px term, and that divergence
is deliberate: it leaves the criterion something training does not optimise.

Thresholding is always **top-k matching the GT area**, never a per-field tuned
threshold.  Instruction conditionality is measured by the same-image paired
difference plus three negative controls (shuffled / irrelevant words / fixed
phrase), never by any AUC variant.

    "Once the gates pass, the single frozen Where checkpoint is selected in this
     order: generated-context local median soft-IoU; grid boundary F1; p10
     soft-IoU; parameter count, peak memory and latency.  If no arm passes every
     gate, the lexicographic best is still selected for diagnosis but must be
     tagged ``WHERE-GATE-FAILED``."
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import torch

from .config import (
    GATES,
    GATE_FAILED_TAG,
    ACTIVE_PRIMITIVE_ON_THRESHOLD,
    ANTONYM_INVARIANCE_MAX,
    GRID_BOUNDARY_TOL_CELLS,
    SOFT_IOU_KIND,
    SELECTION_ORDER,
)
from .losses import _EPS, boundary_map, soft_iou

__all__ = ["percentile", "soft_iou_value", "boundary_f1", "topk_mask",
           "gt_area_k", "center_prior_field", "hard_iou", "grid_boundary_f1",
           "paired_delta", "instruction_paired_delta", "antonym_invariance",
           "active_primitive_count", "active_primitive_bucket",
           "sample_metrics", "summarise", "arm_metrics",
           "evaluate_gates", "lexicographic_best", "ATTRIBUTION_NOTE",
           "attribution_section", "context_deltas"]


def percentile(xs: Sequence[float], p: float) -> float | None:
    """**nearest-rank** order statistic (amendment A-6), i.e. the equivalent of
    ``numpy.quantile(..., method="nearest")`` -- NOT an interpolated quantile.
    n=400 median is the 201st sorted element; the two differ by ~4e-4 here, which
    matters only because the gates are hard thresholds."""
    if not xs:
        return None
    s = sorted(xs)
    if p <= 0:
        return s[0]
    if p >= 1:
        return s[-1]
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def soft_iou_value(m: torch.Tensor, t: torch.Tensor) -> float:
    """``sum(min)/sum(max)`` -- **the min/max form; the product form is banned.**

    Where-A result review B-1: the product form
    ``sum(p*g) / sum(p + g - p*g)`` correlates **0.955 with the softness of the
    GT mask** and **-0.003 with fit quality**, and its ceiling on a *perfect*
    prediction of a soft GT has median **0.786** -- below the 0.75 gate's
    headroom.  A §5.6 "soft-IoU >= 0.75" gate evaluated in that form would
    therefore mostly be measuring how soft the GT happens to be, which is
    exactly the failure mode that got AUC banned.  Measured here: a perfect
    prediction of a soft GT scores 1.0000 in min/max and 0.5054 in product.

    ``soft_iou`` still exposes ``kind="prod"`` because Where-A reports both for
    provenance, but nothing on the criteria path may use it -- asserted by
    ``tests/test_soft_iou_form.py``.
    """
    return float(soft_iou(m.reshape(-1).double(), t.reshape(-1).double(),
                          SOFT_IOU_KIND))


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


def active_primitive_count(readout: str, rho: Mapping[str, Any]) -> int | None:
    """How many CBand12 primitives are actually switched on (``c > 0.5``).

    Where-A's failure analysis found a systematic fragility in the guided
    upsample's hi tier for **single-active-primitive** fits: 30.8% of samples,
    median hi-vs-low drop **+0.012** against **+0.006** for fits with >= 3
    primitives, and all three collapse cases were single-primitive narrow-band
    extrapolations.  Where-B can inherit it, because it predicts the same rho.

    ``None`` for ``R-Band``: a single learnable band-pass has exactly one
    primitive by construction, so the count carries no information and is
    reported as ``n/a`` rather than as a misleading ``1``.
    """
    if readout != "cband12":
        return None
    from q3vl.where.readout import cband_params

    c = cband_params({k: torch.as_tensor(v) for k, v in rho.items()})["c"]
    return int((c > ACTIVE_PRIMITIVE_ON_THRESHOLD).sum())


def active_primitive_bucket(count: int | None) -> str:
    """``"n/a"`` | ``"1"`` | ``"2"`` | ``">=3"`` -- the reporting strata."""
    if count is None:
        return "n/a"
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    return ">=3"


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


def paired_delta(a: Sequence[float], b: Sequence[float], *, n_perm: int = 10000,
                 seed: int = 0) -> dict[str, Any]:
    """Paired mean ``a - b`` with a **sign-flip permutation** p-value and a CI.

    Under the paired null the two labels are exchangeable within a pair, so the
    sign of each difference is equally likely to be + or -.  Enumerating (or
    sampling) sign flips is the textbook exact test for this design, and it is
    both cheaper and stricter than inverting a bootstrap CI, which was the
    previous implementation (review nit N23): CI inversion also reported a hard
    ``p = 0.0`` for an all-positive difference set instead of the honest
    ``<= 1/(n_perm + 1)`` bound that a permutation test gives.

    ``p`` is the two-sided add-one-corrected proportion of sign-flipped means at
    least as extreme as the observed one, so it can never be exactly zero.

    Amendment A-6 pins the counts, because A-5's text ("bootstrap, 2000
    resamples") named neither the right test nor the right number: the p-value is
    **10,000 sign-flip permutations** (hence the observed floor 1/10001), and the
    2,000 resamples are the *bootstrap CI* below, a different quantity.
    """
    import random as _random

    pairs = [(float(x), float(y)) for x, y in zip(a, b)
             if x is not None and y is not None]
    n = len(pairs)
    if n == 0:
        return {"n": 0, "delta": None, "p_value": None, "ci95": None,
                "test": "sign_flip_permutation"}
    diffs = [x - y for x, y in pairs]
    obs = sum(diffs) / n
    rng = _random.Random(seed)

    n_extreme = 0
    for _ in range(n_perm):
        m = sum(d if rng.random() < 0.5 else -d for d in diffs) / n
        if abs(m) >= abs(obs) - 1e-12:
            n_extreme += 1
    p = (n_extreme + 1) / (n_perm + 1)          # add-one: never exactly 0

    # a percentile bootstrap CI is still the natural interval for the estimate
    boots = []
    for _ in range(2000):
        boots.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {"n": n, "delta": obs, "p_value": min(1.0, p),
            "ci95": [boots[49], boots[1949]], "n_perm": n_perm,
            "test": "sign_flip_permutation"}


def antonym_invariance(
    rows: Sequence[Mapping[str, Any]], *, reference: str = "gt",
    control: str = "antonym", key: str = "grid_hard_iou", seed: int = 0,
) -> dict[str, Any]:
    """Invariance under a colour-direction flip (main-agent ruling on S5.5).

    Joins the reference board and the antonym board **per sample** and reports
    the median ``|delta|`` -- a paired quantity, which is what the pre-registered
    ``|delta| <= 0.05`` threshold is stated against.  Where's mask is a function
    of the subject, so flipping ``darker`` <-> ``brighter`` must not move it; a
    field that does move is reading colour words.

    This is a **reported negative-control column, not a gate**: the ruling is
    explicit about that, and a hard gate on an invariance would be the wrong
    instrument anyway (it would punish a field for a tie-break as harshly as for
    genuinely reading colour).
    """
    by: dict[str, dict[str, Mapping[str, Any]]] = {}
    for r in rows:
        ctx = str(r.get("context"))
        if ctx in (reference, control) and r.get(key) is not None:
            by.setdefault(str(r.get("sample_id")), {})[ctx] = r
    paired = [(v[reference], v[control]) for v in by.values()
              if reference in v and control in v]
    if not paired:
        return {"n": 0, "median_abs_delta": None, "max_abs_delta": None,
                "signed_delta": None, "threshold": ANTONYM_INVARIANCE_MAX,
                "within_threshold": None}
    deltas = [float(a[key]) - float(b[key]) for a, b in paired]
    abs_d = sorted(abs(d) for d in deltas)
    med = abs_d[len(abs_d) // 2]
    signed = paired_delta([float(a[key]) for a, _ in paired],
                          [float(b[key]) for _, b in paired], seed=seed)
    return {
        "n": len(paired), "key": key,
        "median_abs_delta": med,
        "p90_abs_delta": abs_d[min(len(abs_d) - 1, int(0.9 * (len(abs_d) - 1)))],
        "max_abs_delta": abs_d[-1],
        "signed_delta": signed["delta"], "signed_p_value": signed["p_value"],
        "threshold": ANTONYM_INVARIANCE_MAX,
        "within_threshold": med <= ANTONYM_INVARIANCE_MAX,
        "note": "negative-control column, not a gate (main-agent ruling)",
    }


def instruction_paired_delta(
    rows: Sequence[Mapping[str, Any]], *, seed: int = 0,
) -> dict[str, Any]:
    """Paired difference over two instructions on the **same image** (A-5).

    For a pair ``(A, B)`` drawn from one ``source_image_id`` with different
    instructions and different GT regions::

        d_A = IoU(field_A, GT_A) - IoU(field_A, GT_B)
        d_B = IoU(field_B, GT_B) - IoU(field_B, GT_A)

    Both terms hold the *image* fixed, so image salience and the centre prior
    cancel by construction -- which is exactly what the shuffled control cannot
    guarantee, since it draws its partner from the same image but scores against
    only one GT.  ``delta > 0`` with a small p means the field follows the
    instruction rather than the picture.

    Each row must carry ``sample_id``, ``source_image_id``, ``instruction``,
    ``self_iou`` and ``cross_iou`` (the latter two computed by the caller, which
    is the only place that holds the predicted fields).
    """
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault(str(r.get("source_image_id")), []).append(r)
    self_v, cross_v, n_groups = [], [], 0
    for members in groups.values():
        used = [m for m in members if m.get("cross_iou") is not None]
        if len(used) < 2:
            continue
        n_groups += 1
        for m in used:
            self_v.append(float(m["self_iou"]))
            cross_v.append(float(m["cross_iou"]))
    out = paired_delta(self_v, cross_v, seed=seed)
    out["n_groups"] = n_groups
    out["n_samples"] = len(self_v)
    return out


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
        # The user's 2026-08-10 ruling: with the Where-A oracle verified at a
        # 0.97 ceiling, day-to-day progress is "how close is the prediction to
        # the oracle", as a DIRECT value -- the ratio hides whether both terms
        # moved.  Same min/max form as every other soft-IoU here (footnote 1).
        out["soft_iou_vs_oracle"] = soft_iou_value(m_pred, m_oracle)

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
        # Where-A's fragility is measured as the hi-vs-low degradation, so the
        # same quantity is reported here and stratified by primitive count.
        out["hi_lo_soft_iou_drop"] = out["grid_soft_iou"] - out["soft_iou"]
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
    # amendment A-6: the ">=85% of oracle" gate's denominator is each sample's
    # OWN delivery-tier (hi) ceiling -- a per-image paired ratio, not the low-tier
    # corpus constant the protocol footnote used to name.  `oracle_soft_iou` is
    # computed against `mask_hi` in `sample_metrics`, so the tier is hi by
    # construction; keep it that way, a tier swap here changes the gate silently.
    ratio = [float(r["soft_iou"]) / float(r["oracle_soft_iou"])
             for r in rows if is_local(r) and r.get("oracle_soft_iou")]
    bratio = [float(r["grid_boundary_f1"]) / float(r["oracle_grid_boundary_f1"])
              for r in rows if is_local(r) and r.get("oracle_grid_boundary_f1")]
    sratio = pick("s_std_ratio", is_local)
    hilo = pick("hi_lo_soft_iou_drop", is_local)

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
        # direct value (the main monitoring reading) next to the ratio
        "soft_iou_vs_oracle": percentile(
            pick("soft_iou_vs_oracle", is_local), 0.5),
        "soft_iou_vs_oracle_ratio": percentile(ratio, 0.5),
        "grid_boundary_f1_vs_oracle_ratio": percentile(bratio, 0.5),
        "s_std_ratio_median": percentile(sratio, 0.5),
        "hi_lo_soft_iou_drop_median": percentile(hilo, 0.5),
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
    "known_blind_spots": {
        "boundary_f1_alone": (
            "grid boundary F1 cannot separate 'compact but in the wrong place' "
            "from 'scattered noise': measured on an off-centre GT, a displaced "
            "compact field and the centre prior both score 0.0000 while "
            "scattered noise scores 0.0357. The coverage column (hard-IoU) and "
            "the centre-prior column are what close that gap -- no single column "
            "is a criterion, which is why the red line demands all three."
        ),
        "single_active_primitive_fragility": (
            "KNOWN RISK, inherited from Where-A (REVIEW-result + failure "
            "analysis): fits with exactly one active CBand12 primitive (c > 0.5) "
            "are systematically fragile in the guided upsample's hi tier -- "
            "30.8% of samples, median hi-vs-low drop +0.012 vs +0.006 at >= 3 "
            "primitives, and all three collapse cases were single-primitive "
            "narrow-band extrapolations. Hi-tier columns are therefore "
            "stratified by active primitive count (1 / 2 / >=3); R-Band is n/a "
            "since one band-pass is one primitive by construction. Reported "
            "only -- no loss, gate or training change (main-agent ruling)."
        ),
        "antonym_invariance": (
            "small |delta| is the PASS here, not a large one -- this control is "
            "the mirror image of the directional paired difference. A field that "
            "moves when only darker<->brighter flips is reading colour words it "
            "should not; reported, never gated (main-agent ruling)."
        ),
        "instruction_conditionality": (
            "no single context board proves instruction following. The three "
            "negative controls (shuffled / irrelevant_words / fixed_phrase) and "
            "the same-image paired difference are read together; a field that "
            "scores the same under fixed_phrase as under the real instruction is "
            "reading image salience."
        ),
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
        "不得只用前者作结。\n\n"
        "### 已知盲区（每块板都必须带着走，不是 NOTES 里的一次性说明）\n\n"
        "- **单看 grid boundary F1 会漏判**：它分不出「紧凑但位置错」与「散点噪声」。\n"
        "  实测（偏心 GT）：位移的紧凑场与中心先验都是 **0.0000**，而散点噪声是 **0.0357**——\n"
        "  也就是说噪声在这一列上反而『赢』了那个位置错的紧凑场。补位的是覆盖列（hard-IoU）\n"
        "  与中心先验列：**三列缺一不可**，任何一列单独都会被骗。\n"
        "- **单看任何一块上下文板都证明不了指令跟随**：三条负控制\n"
        "  （`shuffled` / `irrelevant_words` / `fixed_phrase`）与同图配对差分要**合起来读**。\n"
        "  一个在 `fixed_phrase` 下与真实指令得分相同的场，读的是图像显著性而不是指令——\n"
        "  红线里那句对全体样本相同的 `\"the main subject\"` 拿到 AUC 0.907 就是这么来的。\n"
    )


def context_deltas(per_context: Mapping[str, Mapping[str, Any]],
                   main_context: str = "generated") -> dict[str, Any]:
    """Named delta columns for every negative-control board (review N28).

    The red line's usage is "a field that scores the same under ``fixed_phrase``
    as under the real instruction is reading image salience".  That comparison
    has to be a **computed column**, not something the report author works out by
    hand from two boards -- which is what `irrelevant_words` and `fixed_phrase`
    were reduced to before this.
    """
    main = per_context.get(main_context, {})
    out: dict[str, Any] = {}
    base = main.get("local_soft_iou_median")
    #: control board -> the name of its drop column
    controls = {
        "null": "null_context_gap",
        "shuffled": "instruction_shuffle_iou_drop",
        "irrelevant_words": "irrelevant_words_iou_drop",
        "fixed_phrase": "fixed_phrase_iou_drop",
        "antonym": "antonym_iou_drop",
    }
    for other, name in controls.items():
        v = per_context.get(other, {}).get("local_soft_iou_median")
        if base is not None and v is not None:
            out[name] = float(base) - float(v)
    gt = per_context.get("gt", {}).get("local_soft_iou_median")
    if base is not None and gt is not None:
        out["gt_generated_iou_gap"] = abs(float(gt) - float(base))
    # the smallest drop across the three negative controls is the binding one:
    # a field only demonstrates instruction dependence if it loses on ALL of them
    drops = [out[controls[c]] for c in ("shuffled", "irrelevant_words", "fixed_phrase")
             if controls[c] in out]
    if drops:
        out["min_negative_control_drop"] = min(drops)
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
