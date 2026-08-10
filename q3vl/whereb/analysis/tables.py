"""Per-class metric tables.

One rule governs this module: **every number is produced by
:func:`q3vl.whereb.metrics.summarise`**, the aggregator the main board already
uses, applied to a subset of the same ``per_sample.jsonl`` rows.  Re-deriving a
median here would let a per-class number drift from the headline number it is
supposed to decompose, and the reader would have no way to tell which one moved.

Consequences that come for free from that choice:

* no AUC column can appear -- ``summarise`` does not produce one (red line);
* the centre-prior baseline, its paired delta and the delta's permutation p-value
  travel with every single table, because ``summarise`` emits them;
* thresholding is the matched-area top-k rule, because the columns were computed
  under it upstream.

The tables are **marginal**, one dimension at a time.  ``n`` is carried in every
row and any class with fewer than :data:`LOW_CONFIDENCE_N` samples is flagged --
a median over 6 samples is a data point, not a finding.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["LOW_CONFIDENCE_N", "REPORT_COLUMNS", "load_per_sample",
           "rows_by_context", "per_class_tables", "overall_table"]

#: below this a class row is reported but marked low-confidence
LOW_CONFIDENCE_N = 20

#: (key in summarise() output, column header, digits).  Order is the report order.
REPORT_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("n_local", "n", 0),
    ("local_soft_iou_median", "softIoU med", 3),
    ("local_soft_iou_p10", "softIoU p10", 3),
    ("grid_hard_iou", "hardIoU", 3),
    ("grid_boundary_f1", "边界F1", 3),
    ("center_prior_hard_iou", "中心先验 hardIoU", 3),
    ("center_prior_delta_hard_iou", "Δ vs 中心先验", 3),
    ("center_prior_delta_hard_iou_p", "Δ 的 p", 4),
    ("soft_iou_vs_oracle_ratio", "/oracle", 3),
    ("s_std_ratio_median", "std(s)/std(s*)", 3),
    ("hi_lo_soft_iou_drop_median", "hi-lo 降幅", 4),
)


def load_per_sample(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def rows_by_context(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(str(r.get("context")), []).append(dict(r))
    return out


def _summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from q3vl.whereb.metrics import summarise

    return summarise(rows)


def overall_table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The un-stratified row, so a class can be read against its own board."""
    s = _summarise(rows)
    s["n"] = len(rows)
    s["low_confidence"] = s.get("n_local", 0) < LOW_CONFIDENCE_N
    return s


def per_class_tables(
    rows: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, str]],
    dimensions: Sequence[str],
    *,
    include_unlabelled: bool = False,
) -> dict[str, dict[str, dict[str, Any]]]:
    """``{dimension: {class: summarise(rows of that class)}}``.

    ``labels`` maps ``sample_id -> {dimension: class}``.  A row whose sample has
    no label (global samples when the caller filtered them out, or a sample whose
    mask could not be read) lands in ``"unlabelled"`` and is reported there rather
    than being dropped silently -- a missing mask must not shrink a denominator
    without saying so.
    """
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for dim in dimensions:
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for r in rows:
            lab = labels.get(str(r.get("sample_id")))
            cls = "unlabelled" if lab is None else str(lab.get(dim, "unlabelled"))
            if cls == "unlabelled" and not include_unlabelled:
                continue
            groups.setdefault(cls, []).append(r)
        table: dict[str, dict[str, Any]] = {}
        for cls, members in sorted(groups.items()):
            s = _summarise(members)
            s["n"] = len(members)
            s["low_confidence"] = s.get("n_local", 0) < LOW_CONFIDENCE_N
            table[cls] = s
        out[dim] = table
    return out


def worst_class(table: Mapping[str, Mapping[str, Any]], *,
                key: str = "local_soft_iou_median",
                min_n: int = LOW_CONFIDENCE_N) -> str | None:
    """The lowest-scoring class of one dimension, ignoring low-confidence cells.

    A class with 6 samples can hold the worst median by accident; the tail is
    supposed to point at a *kind* of region, so the pick is made among classes
    that carry enough samples to mean something.  If no class clears ``min_n``
    the pick falls back to all classes and :func:`worst_class_note` says so.
    """
    cand = {k: v for k, v in table.items()
            if v.get(key) is not None and v.get("n_local", 0) >= min_n}
    if not cand:
        cand = {k: v for k, v in table.items() if v.get(key) is not None}
    if not cand:
        return None
    return min(cand, key=lambda k: float(cand[k][key]))


def worst_class_note(table: Mapping[str, Mapping[str, Any]], picked: str | None, *,
                     key: str = "local_soft_iou_median") -> str | None:
    """Name the class that would have won if low-confidence cells counted.

    Without this the report reads "worst class: single (0.501)" on a dimension
    where ``multi`` scored 0.270 -- technically the documented rule, and exactly
    the kind of silently-dropped comparison this campaign has been bitten by.
    """
    vals = {k: float(v[key]) for k, v in table.items() if v.get(key) is not None}
    if not vals or picked is None:
        return None
    low = min(vals, key=lambda k: vals[k])
    if low == picked:
        return None
    n = table[low].get("n_local", 0)
    return (f"注：`{low}` 的中位更低（{vals[low]:.3f}，n={n} < {LOW_CONFIDENCE_N}），"
            f"因样本量不足未被选为最差类别，但**不能**据此说它更好。")


def class_distribution(labels: Mapping[str, Mapping[str, str]],
                       dimensions: Sequence[str]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for dim in dimensions:
        counts: dict[str, int] = {}
        for lab in labels.values():
            counts[str(lab.get(dim))] = counts.get(str(lab.get(dim)), 0) + 1
        out[dim] = dict(sorted(counts.items()))
    return out
