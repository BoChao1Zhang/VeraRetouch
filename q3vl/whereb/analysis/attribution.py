"""Which stage of the pipeline produced this bad sample.

The Where chain has five places a local prediction can die, and the eval board
reports none of them separately::

    instruction --> <where> context --> H_where --> (w0, w_dir, alpha)  --> s_low
                                                 \\-> rho                    |
                                                                 guided upsample
                                                                            |
                                                       m = R(s; rho) <------+

So a sample with soft-IoU 0.11 could be (a) a generated context that says the
wrong thing, (b) a region the frozen Where-A basis cannot express at all -- its
own oracle is bad -- (c) an ``s`` field pointing somewhere else, (d) an ``s``
field that is right while ``rho`` slices it at the wrong level, or (e) a low tier
that was right and a guided upsample that lost it.  Those five call for five
different fixes, and "median soft-IoU 0.49" calls for none.

Each mechanism below is a **predicate over quantities that already exist**, so
the labelling is reproducible from the delivered artefacts:

============================  =========================================  ========
mechanism                     evidence                                   needs
============================  =========================================  ========
``format_failure``            the row's own ``format_failure``           eval
``context_quality``           GT-context IoU minus generated-context IoU eval
``oracle_ceiling``            ``oracle_soft_iou`` of this sample          eval
``s_collapse``                ``s_std_ratio``                            eval
``upsample_collapse``         ``hi_lo_soft_iou_drop``                    eval
``area_mismatch``             ``pred_mean`` vs ``gt_mean``               eval
``below_center_prior``        ``center_prior_hard_iou`` - ``grid_hard_iou`` eval
``s_direction``               ``cos(w_dir, w_dir*)``                     fields
``s_error``                   IoU(oracle s, predicted rho) - IoU(pred)   fields
``rho_error``                 IoU(predicted s, oracle rho) - IoU(pred)   fields
``single_primitive``          CBand12 active count == 1 and hi < low     fields
============================  =========================================  ========

"needs eval" = computable from ``per_sample.jsonl`` alone; "needs fields" = only
available when :mod:`q3vl.whereb.analysis.fieldcache` has re-run a checkpoint.
The tool reports which set it had, and a mechanism it could not test is recorded
as ``not_tested`` rather than as ``absent`` -- the difference matters when the
summary is read as "what is the training bottleneck".

Labels are **multi-label** (a collapsed ``s`` and a bad context are not exclusive)
and a single ``primary`` is assigned by :data:`PRIORITY`, which orders the
mechanisms by how far upstream they sit: fixing a downstream symptom while the
upstream cause stands is the failure mode this ordering exists to prevent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

__all__ = ["AnalysisThresholds", "MECHANISMS", "PRIORITY", "FIELD_MECHANISMS",
           "attribute_sample", "mechanism_summary"]

MECHANISMS: tuple[str, ...] = (
    "format_failure", "oracle_ceiling", "context_quality", "s_direction",
    "s_error", "rho_error", "s_collapse", "upsample_collapse", "area_mismatch",
    "below_center_prior", "single_primitive", "unexplained",
)

#: mechanisms that can only be tested with a re-run checkpoint's fields
FIELD_MECHANISMS: frozenset[str] = frozenset(
    {"s_direction", "s_error", "rho_error", "single_primitive"}
)

#: upstream first: the primary label is the most upstream mechanism that fired
PRIORITY: tuple[str, ...] = (
    "format_failure",       # the input never arrived intact
    "oracle_ceiling",       # the target is not reachable through this basis
    "context_quality",      # the language conditioning is what differs
    "s_direction",          # the field points elsewhere
    "s_error",              # swapping in the oracle s repairs the mask
    "rho_error",            # swapping in the oracle rho repairs the mask
    "s_collapse",           # the field has no contrast left to point with
    "single_primitive",     # known Where-A fragility, hi tier only
    "upsample_collapse",    # the low tier was fine
    "area_mismatch",        # right place, wrong extent
    "below_center_prior",   # a zero-parameter baseline does better
    "unexplained",
)


@dataclass(frozen=True)
class AnalysisThresholds:
    """Pre-registered cut-offs.  Dumped verbatim into ``config/``."""

    #: a sample is in the long tail when its main-context soft-IoU is below this
    #: OR it is among the worst ``tail_k`` (whichever the caller asks for)
    tail_soft_iou: float = 0.30
    #: GT-context minus generated-context IoU that counts as a context problem,
    #: and how good the GT context has to be for the comparison to mean anything
    context_gap: float = 0.10
    context_gt_ok: float = 0.50
    #: the per-image Where-A oracle is itself this bad -> the basis is the cap
    oracle_ceiling: float = 0.70
    #: std(s_pred)/std(s*) below this is a collapsed axis
    s_collapse: float = 0.20
    #: low-tier soft-IoU minus hi-tier soft-IoU above this is an upsample loss
    upsample_drop: float = 0.05
    #: |pred_mean - gt_mean| / gt_mean above this is an extent error
    area_ratio_gap: float = 0.50
    #: centre prior beats the field's hard-IoU by at least this much
    center_prior_margin: float = 0.05
    #: cos(w_dir, w_dir*) below this is a wrong direction
    w_dir_cos: float = 0.30
    #: an oracle-component swap that lifts hi-tier soft-IoU by this much
    #: identifies which component was broken
    swap_gain: float = 0.15

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _f(row: Mapping[str, Any], key: str) -> float | None:
    v = row.get(key)
    return None if v is None else float(v)


def attribute_sample(
    main: Mapping[str, Any],
    *,
    gt_row: Mapping[str, Any] | None = None,
    fields: Mapping[str, Any] | None = None,
    thresholds: AnalysisThresholds | None = None,
) -> dict[str, Any]:
    """Label one sample.

    ``main`` is the row of the main context (generated), ``gt_row`` the same
    sample's GT-context row, ``fields`` the optional per-sample record produced
    by :mod:`fieldcache` (``w_dir_cos``, ``iou_oracle_s_pred_rho``,
    ``iou_pred_s_oracle_rho``, ``active_primitives``).
    """
    t = thresholds or AnalysisThresholds()
    ev: dict[str, Any] = {}
    hit: list[str] = []
    not_tested: list[str] = []

    if bool(main.get("format_failure")):
        hit.append("format_failure")
    ev["format_failure"] = bool(main.get("format_failure"))

    iou = _f(main, "soft_iou")
    ev["soft_iou"] = iou

    oracle = _f(main, "oracle_soft_iou")
    ev["oracle_soft_iou"] = oracle
    if oracle is None:
        not_tested.append("oracle_ceiling")
    elif oracle < t.oracle_ceiling:
        hit.append("oracle_ceiling")

    if gt_row is None:
        not_tested.append("context_quality")
        ev["context_gap"] = None
    else:
        gt_iou = _f(gt_row, "soft_iou")
        gap = None if (gt_iou is None or iou is None) else gt_iou - iou
        ev["gt_soft_iou"] = gt_iou
        ev["context_gap"] = gap
        if (gap is not None and gt_iou is not None
                and gap >= t.context_gap and gt_iou >= t.context_gt_ok):
            hit.append("context_quality")

    sratio = _f(main, "s_std_ratio")
    ev["s_std_ratio"] = sratio
    if sratio is None:
        not_tested.append("s_collapse")
    elif sratio < t.s_collapse:
        hit.append("s_collapse")

    drop = _f(main, "hi_lo_soft_iou_drop")
    ev["hi_lo_soft_iou_drop"] = drop
    if drop is None:
        not_tested.append("upsample_collapse")
    elif drop >= t.upsample_drop:
        hit.append("upsample_collapse")

    pm, gm = _f(main, "pred_mean"), _f(main, "gt_mean")
    ratio_gap = None if (pm is None or not gm) else abs(pm - gm) / gm
    ev["pred_mean"], ev["gt_mean"], ev["area_ratio_gap"] = pm, gm, ratio_gap
    # the sign is the actionable half: over-covering (the field defaults towards
    # a global mask) and under-covering call for opposite fixes, and a symmetric
    # |gap| column hides which one is happening
    ev["area_ratio"] = None if (pm is None or not gm) else pm / gm
    ev["area_direction"] = (None if ratio_gap is None
                            else ("over" if pm > gm else "under"))
    if ratio_gap is None:
        not_tested.append("area_mismatch")
    elif ratio_gap >= t.area_ratio_gap:
        hit.append("area_mismatch")

    ghi, chi = _f(main, "grid_hard_iou"), _f(main, "center_prior_hard_iou")
    margin = None if (ghi is None or chi is None) else chi - ghi
    ev["grid_hard_iou"], ev["center_prior_hard_iou"] = ghi, chi
    ev["center_prior_margin"] = margin
    if margin is None:
        not_tested.append("below_center_prior")
    elif margin >= t.center_prior_margin:
        hit.append("below_center_prior")

    # --- the three that need a re-run checkpoint ---------------------------
    if not fields:
        not_tested.extend(["s_direction", "s_error", "rho_error", "single_primitive"])
    else:
        cos = _f(fields, "w_dir_cos")
        ev["w_dir_cos"] = cos
        if cos is None:
            not_tested.append("s_direction")
        elif cos < t.w_dir_cos:
            hit.append("s_direction")

        base = _f(fields, "iou_pred") if fields.get("iou_pred") is not None else iou
        for name, key in (("s_error", "iou_oracle_s_pred_rho"),
                          ("rho_error", "iou_pred_s_oracle_rho")):
            swapped = _f(fields, key)
            ev[key] = swapped
            gain = None if (swapped is None or base is None) else swapped - base
            ev[f"{name}_gain"] = gain
            if gain is None:
                not_tested.append(name)
            elif gain >= t.swap_gain:
                hit.append(name)

        n_prim = fields.get("active_primitives")
        ev["active_primitives"] = n_prim
        if n_prim is None:
            not_tested.append("single_primitive")     # band readout: n/a by design
        elif int(n_prim) <= 1 and (drop is not None and drop > 0):
            hit.append("single_primitive")

    if not hit:
        hit.append("unexplained")
    primary = next(m for m in PRIORITY if m in hit)
    return {
        "mechanisms": [m for m in PRIORITY if m in hit],
        "primary": primary,
        "not_tested": sorted(set(not_tested)),
        "evidence": ev,
    }


def mechanism_summary(labelled: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Share of the tail each mechanism explains, and the bottleneck verdict.

    Two views, deliberately both: ``share`` counts every sample a mechanism fired
    on (multi-label, so the column sums above 1) and ``primary_share`` counts only
    the samples where it is the most upstream cause (sums to 1).  Reading only the
    first over-credits downstream symptoms -- an ``s`` that points elsewhere also
    makes the extent wrong -- and reading only the second hides how widespread a
    co-occurring mechanism is.
    """
    n = len(labelled)
    counts = {m: 0 for m in MECHANISMS}
    primary = {m: 0 for m in MECHANISMS}
    untested = {m: 0 for m in MECHANISMS}
    for row in labelled:
        for m in row.get("mechanisms", []):
            counts[m] = counts.get(m, 0) + 1
        primary[row["primary"]] = primary.get(row["primary"], 0) + 1
        for m in row.get("not_tested", []):
            untested[m] = untested.get(m, 0) + 1
    order = sorted(MECHANISMS, key=lambda m: (-primary[m], -counts[m], m))
    top = next((m for m in order if primary[m] > 0 and m != "unexplained"), None)
    return {
        "n_tail": n,
        "counts": counts,
        "share": {m: (counts[m] / n if n else 0.0) for m in MECHANISMS},
        "primary_counts": primary,
        "primary_share": {m: (primary[m] / n if n else 0.0) for m in MECHANISMS},
        "not_tested_counts": untested,
        "bottleneck": top,
        "bottleneck_primary_share": (primary[top] / n if (top and n) else 0.0),
        "unexplained_share": (primary["unexplained"] / n if n else 0.0),
        "note": (
            "share is multi-label and sums above 1; primary_share sums to 1 and "
            "uses the upstream-first PRIORITY order. A mechanism listed in "
            "not_tested_counts was not measurable on that sample (no oracle, no "
            "field cache, or n/a for this readout) -- it is NOT evidence of "
            "absence."
        ),
    }
