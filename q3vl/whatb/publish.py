"""The publication gate: nothing reaches ``metrics.json`` without these checks.

One entry point, :func:`assert_publishable`, wired in front of the write of
``metrics.json`` at **every** call site (quick eval, holdout, final board).  It
checks three things and fetches its own data for all of them:

1. **the training side ran** -- the FIRST row of ``steps.jsonl`` carries every
   pre-registered loss column.  The row is resolved three tiers deep by
   :func:`q3vl.whatb.guards.resolve_first_step_row` (caller -> disk -> in-process
   witness), because the two call sites of the where-side equivalent disagreed
   about their arguments and the arm hook read ``steps_row=None`` as "the loss
   never ran" (SEGSAM and PRND, 2026-08-15).  *"Nobody handed me a row"*
   (:class:`~q3vl.whatb.guards.StepsRowUnavailable`) and *"the row has no loss
   columns"* (:class:`~q3vl.whatb.guards.LossColumnsMissing`) are different
   exceptions, and neither is a pass.

2. **the criteria ran** -- every pre-registered key of
   :func:`q3vl.whatb.criteria.required_criteria` has ``n > 0``.

3. **the degenerate-solution guard ran** -- ``q3vl.whatb.guards`` records a
   witness every time :func:`~q3vl.whatb.guards.assert_transform_not_degenerate`
   executes, and a *published* board whose process never fired it is refused
   (:class:`~q3vl.whatb.guards.DegeneracyCheckNotRun`).  Without this, four of
   the six arms could reach ``metrics.json`` with the guard skipped
   (``--eval-only`` / ``--no-train`` / ``--quick-eval-every 0``) -- "the check
   exists" is not "the check ran", which is how PRND and CONDINST spent 2.6
   GPU-hours on a constant field.  EPR-026 had this check privately; it is
   shared now.

4. **the headline exists on the published board** -- and on an interim board its
   absence is recorded together with the subset's ``winner_confidence``
   histogram, while an interim board that *does* contain normal rows and still
   has no ``headline_normal_only`` raises: then the two disagree, which is the
   ``evaluate.py:483-484`` trap ("the key only exists when the subset happened
   to contain a normal row").

The steps-row machinery itself lives in :mod:`q3vl.whatb.guards` -- one copy, six
arms.  This module is the board half.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import criteria as _criteria
from .guards import (
    FIRST_STEP_COLUMNS,
    DegeneracyCheckNotRun,
    LossColumnsMissing,
    StepsRowUnavailable,
    assert_first_step_columns,
    degeneracy_check_ran,
    record_step_witness,
    resolve_first_step_row,
)

__all__ = [
    "FIRST_STEP_COLUMNS",
    "DegeneracyCheckNotRun",
    "LossColumnsMissing",
    "StepsRowUnavailable",
    "HeadlineMissing",
    "record_step_witness",
    "resolve_first_step_row",
    "assert_first_step_columns",
    "step_columns_for",
    "is_publication_board",
    "assert_publishable",
]


class HeadlineMissing(AssertionError):
    """The board has no ``headline_normal_only`` where one is required."""


def step_columns_for(loss_level: int = 3, *, extra: Iterable[str] = ()
                     ) -> tuple[str, ...]:
    """The first-row columns this configuration promises.

    §4.H: ``L_rec`` / ``L_hc`` / ``L_sparse`` / ``n_colors`` /
    ``n_luts_in_batch`` / ``mining_ratio`` / ``n_hc_masked``, plus ``L_img`` at
    ``--loss-level 4``.  ``extra`` is for an arm whose own experiment variable
    has a column of its own (``L_interp`` for EPR-026, ``L_gate`` for EPR-027):
    an arm whose variable has no first-row column can lose it silently.
    """
    cols = list(FIRST_STEP_COLUMNS)
    if int(loss_level) >= 4:
        cols.append("L_img")
    cols += [c for c in extra if c not in cols]
    return tuple(cols)


def is_publication_board(board: Mapping[str, Any]) -> bool:
    """Is this the final board, or an interim (quick-eval / holdout) one?

    The published board is the one built on the full evaluation split; an arm
    marks it by setting ``board["published"] = True`` or by evaluating the whole
    of ``V_what`` (``n_rows >= n_expected``).  Interim boards are exempt from the
    headline-presence hard check and get the recorded-absence treatment instead.
    """
    if "published" in board:
        return bool(board["published"])
    return not bool(board.get("quick", False))


def assert_publishable(board: Mapping[str, Any], arm: str, *,
                       steps_row: Mapping[str, Any] | None = None,
                       steps_path: Any = None,
                       eval_only: bool = False,
                       loss_level: int = 3,
                       extra_step_columns: Iterable[str] = (),
                       axes: Sequence[str] | None = None,
                       required: Sequence[str] | None = None,
                       require_degeneracy_check: bool = True,
                       ) -> dict[str, Any]:
    """Refuse a board that cannot show its own experiment ran.  Returns a report.

    ``eval_only`` skips only the training-side check (there are no steps), never
    the criteria check.  Raises
    :class:`~q3vl.whatb.guards.StepsRowUnavailable`,
    :class:`~q3vl.whatb.guards.LossColumnsMissing`,
    :class:`~q3vl.whatb.guards.DegeneracyCheckNotRun`,
    :class:`~q3vl.whatb.criteria.CriterionNotComputed` or
    :class:`HeadlineMissing` -- five distinct failures, never one generic one.

    ``require_degeneracy_check=False`` is the deliberate escape hatch (a board
    assembled from another process's checkpoints); it is recorded in the report,
    never silent.
    """
    report: dict[str, Any] = {"arm": arm,
                              "published_board": is_publication_board(board)}

    # (1) the training side
    if eval_only:
        report["steps"] = {"skipped": "eval_only (no training steps)"}
    else:
        want = step_columns_for(loss_level, extra=extra_step_columns)
        row, source = assert_first_step_columns(want, steps_row=steps_row,
                                                steps_path=steps_path)
        report["steps"] = {"source": source, "required": list(want),
                           "values": {c: row[c] for c in want}}

    # (2) the degeneracy guard really ran in this process
    state = degeneracy_check_ran()
    report["degeneracy"] = state
    if state is None and require_degeneracy_check and report["published_board"]:
        raise DegeneracyCheckNotRun(
            f"arm {arm}: the degenerate-solution guard never ran in this "
            "process (q3vl.whatb.guards.assert_transform_not_degenerate).  A "
            "board published without it cannot tell a trained transform from a "
            "constant one -- which is how PRND and CONDINST spent 2.6 GPU-hours "
            "on a std-0 field.  Run the quick eval at least once, or pass "
            "require_degeneracy_check=False and say why on the artefact.")
    if state is None:
        report["degeneracy"] = {"ran": False, "required": bool(
            require_degeneracy_check and report["published_board"])}

    # (3) the criteria
    report["criteria"] = _criteria.assert_criteria_ran(board, arm, axes=axes,
                                                       required=required)

    # (4) the headline, with the interim-board caveat
    ctx_all = ((board.get("contexts") or {}).get("all") or {})
    hn = ctx_all.get("headline_normal_only") or {}
    n_hn = int(hn.get("n", 0) or 0)
    n_normal = int(board.get("n_normal", n_hn) or 0)
    report["headline"] = {"n": n_hn, "n_normal_rows": n_normal,
                          "n_low_excluded": board.get("n_low_excluded")}
    if n_hn == 0:
        if report["published_board"]:
            raise HeadlineMissing(
                f"arm {arm}: the published board has no "
                ".contexts.all.headline_normal_only.  Checkpoint selection and "
                "every cross-arm delta read exactly that key; the pooled figure "
                "may not stand in for it (it under-counts by ~0.031).")
        if n_normal > 0:
            raise HeadlineMissing(
                f"arm {arm}: the interim board carries {n_normal} normal rows "
                "but no headline_normal_only -- the board and its own subset "
                "disagree, which means the column was not computed rather than "
                "not applicable")
        report["headline"]["note"] = (
            "interim board with no normal rows: absence recorded, not raised "
            "(quick eval can draw an all-low subset)")
    return report
