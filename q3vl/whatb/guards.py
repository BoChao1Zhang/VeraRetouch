"""Run-time guards the six whatb arms share.  Two failures, structurally closed.

1. **The assertion-contract failure** (where side, 2026-08-15, SEGSAM and PRND
   one each).  ``assert_criteria_ran`` had two call sites with different
   signatures; the arm hook received ``steps_row=None`` from the one that gates
   ``metrics.json`` and reported "the loss never ran".  Here the board-time
   assertion **fetches its own data**, three tiers deep -- caller -> the first
   line of ``steps.jsonl`` on disk -> an in-process witness recorded by the
   first micro-batch -- and *"nobody handed me a row"* and *"the row is missing
   the loss columns"* raise **different exceptions**
   (:class:`StepsRowUnavailable` vs :class:`LossColumnsMissing`).

2. **The constant-field failure** (where side, PRND and CONDINST).  1200 steps
   and 2.6 GPU-hours before anyone noticed the head emitted a constant field:
   std 0, zero correlation with GT, IoU carried entirely by top-k tie-breaks.
   :func:`assert_transform_not_degenerate` runs at the **first quick eval** and
   exits the process on any of three conditions -- the transform is flat across
   query colours, or it is the identity, or it is the same for every sample.

Everything is computed on the tensor's own device: a ``.cpu()`` round trip
before a comparison is how the where side moved an IoU by 0.296 (CUDA and CPU
``topk`` break ties differently).
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

__all__ = [
    "DegeneracyThresholds",
    "DegeneracyReport",
    "DegenerateTransform",
    "measure_degeneracy",
    "assert_transform_not_degenerate",
    "DegeneracyCheckNotRun",
    "degeneracy_check_ran",
    "record_degeneracy_check",
    "clear_degeneracy_check",
    "StepsRowUnavailable",
    "LossColumnsMissing",
    "record_step_witness",
    "clear_step_witness",
    "resolve_first_step_row",
    "assert_first_step_columns",
    "FIRST_STEP_COLUMNS",
]


# --------------------------------------------------------------------------- #
# 2. degenerate-solution guard
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DegeneracyThresholds:
    """Deliberately small floors.  Anything above them is "not obviously dead".

    ``point_std``
        Mean over samples and channels of ``std_over_queries(f(x))``.  A
        constant transform (one colour for every input) sits at 0.
    ``identity_dev``
        Mean ``|f(x) - x|``.  Zero means the arm learned nothing but the
        residual anchor -- which is exactly step 0 for a zero-initialised head,
        so this check is for the FIRST QUICK EVAL, not for step 0.
    ``cross_std``
        Mean over queries and channels of ``std_over_samples(f_i(x))``.  Zero
        means one transform for every instruction -- the N3 "constant" control
        would score identically to the arm, and the board could still look fine.

    Values are written into ``run_setup.json`` so the board records what bar
    was used.  They are floors for "alive", never targets.
    """

    point_std: float = 1e-3
    identity_dev: float = 1e-3
    cross_std: float = 1e-4

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class DegeneracyReport:
    """Measured values next to the floors, plus which checks failed."""

    point_std: float
    identity_dev: float
    cross_std: float
    n_samples: int
    n_queries: int
    thresholds: DegeneracyThresholds
    failures: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "point_std": self.point_std,
            "identity_dev": self.identity_dev,
            "cross_std": self.cross_std,
            "n_samples": self.n_samples,
            "n_queries": self.n_queries,
            "thresholds": self.thresholds.as_dict(),
            "failures": list(self.failures),
        }


class DegenerateTransform(RuntimeError):
    """Raised by :func:`assert_transform_not_degenerate` when ``exit_process`` is off."""

    def __init__(self, report: DegeneracyReport) -> None:
        super().__init__("degenerate transform: " + ", ".join(report.failures))
        self.report = report


def measure_degeneracy(
    y: Tensor,
    x: Tensor,
    *,
    thresholds: DegeneracyThresholds = DegeneracyThresholds(),
) -> DegeneracyReport:
    """Three scalars from ``f(x)`` -- all reductions stay on ``y``'s device.

    ``y`` and ``x`` are ``(B, P, 3)``: ``B`` samples (distinct conditions),
    ``P`` query colours.  ``x`` may be ``(P, 3)`` when every sample shares the
    query set.
    """
    if y.dim() != 3 or y.shape[-1] != 3:
        raise ValueError(f"expected f(x) of shape (B, P, 3), got {tuple(y.shape)}")
    xr = x.to(device=y.device, dtype=y.dtype)
    if xr.dim() == 2:
        xr = xr.unsqueeze(0).expand_as(y)
    if xr.shape != y.shape:
        raise ValueError(f"query shape {tuple(xr.shape)} != output shape {tuple(y.shape)}")
    b, p = int(y.shape[0]), int(y.shape[1])

    point_std = y.std(dim=1, unbiased=False).mean() if p > 1 else y.new_zeros(())
    identity_dev = (y - xr).abs().mean()
    cross_std = y.std(dim=0, unbiased=False).mean() if b > 1 else y.new_full((), float("nan"))

    ps, idv, cs = float(point_std), float(identity_dev), float(cross_std)
    failures: list[str] = []
    if not (ps > thresholds.point_std):
        failures.append(
            f"flat across query colours: mean std_over_queries = {ps:.3e} <= {thresholds.point_std:.1e}"
        )
    if not (idv > thresholds.identity_dev):
        failures.append(
            f"transform is the identity: mean |f(x)-x| = {idv:.3e} <= {thresholds.identity_dev:.1e}"
        )
    if b > 1 and not (cs > thresholds.cross_std):
        failures.append(
            f"one transform for every sample: mean std_over_samples = {cs:.3e} <= {thresholds.cross_std:.1e}"
        )
    return DegeneracyReport(ps, idv, cs, b, p, thresholds, tuple(failures))


def assert_transform_not_degenerate(
    y: Tensor,
    x: Tensor,
    *,
    thresholds: DegeneracyThresholds = DegeneracyThresholds(),
    where: str = "quick_eval",
    exit_process: bool = True,
    extra: Mapping[str, Any] | None = None,
) -> DegeneracyReport:
    """Stop the run at the first quick eval if ``f`` is a degenerate solution.

    Prints the three measured numbers next to their floors and, when
    ``exit_process`` (the default the runners use), leaves via ``SystemExit(2)``
    -- a training loop must not be able to swallow this as a warning.  Tests
    pass ``exit_process=False`` and catch :class:`DegenerateTransform`.
    """
    report = measure_degeneracy(y, x, thresholds=thresholds)
    record_degeneracy_check(report, where=where, extra=extra)
    if report.ok:
        return report
    lines = [
        f"[whatb.guards] DEGENERATE TRANSFORM at {where}",
        f"  samples B = {report.n_samples}, queries P = {report.n_queries}",
        f"  std over queries  = {report.point_std:.6e}   floor {thresholds.point_std:.1e}",
        f"  mean |f(x) - x|   = {report.identity_dev:.6e}   floor {thresholds.identity_dev:.1e}",
        f"  std over samples  = {report.cross_std:.6e}   floor {thresholds.cross_std:.1e}",
    ]
    lines += [f"  FAIL: {f}" for f in report.failures]
    if extra:
        lines += [f"  {k} = {v}" for k, v in extra.items()]
    text = "\n".join(lines)
    if not exit_process:
        print(text, file=sys.stderr, flush=True)
        raise DegenerateTransform(report)
    print(text, file=sys.stderr, flush=True)
    raise SystemExit(2)


class DegeneracyCheckNotRun(AssertionError):
    """The degenerate-solution guard never executed in this process.

    "The check exists" and "the check ran" are indistinguishable from the
    artefact unless something records the second one.  PRND and CONDINST burned
    2.6 GPU-hours on a constant field for exactly that reason, and four of the
    six whatb arms could still reach a board with the guard skipped
    (``--eval-only`` / ``--no-train`` / ``--quick-eval-every 0``).  EPR-026 had
    its own copy of this class; this is the shared one, checked by
    :func:`q3vl.whatb.publish.assert_publishable` for every arm.
    """


_DEGENERACY_WITNESS: dict[str, Any] = {}


def record_degeneracy_check(report: "DegeneracyReport", *, where: str = "",
                            extra: Mapping[str, Any] | None = None) -> None:
    """Called by :func:`assert_transform_not_degenerate` -- pass or fail."""
    _DEGENERACY_WITNESS.clear()
    _DEGENERACY_WITNESS.update({"where": where, "ok": bool(report.ok),
                                **report.as_dict()})
    if extra:
        _DEGENERACY_WITNESS["extra"] = dict(extra)


def clear_degeneracy_check() -> None:
    """Drop the in-process record (tests, and a runner re-entering training)."""
    _DEGENERACY_WITNESS.clear()


def degeneracy_check_ran() -> dict[str, Any] | None:
    """The witness, or ``None`` if the guard never ran in this process."""
    return dict(_DEGENERACY_WITNESS) or None


# --------------------------------------------------------------------------- #
# 1. first-step-row contract
# --------------------------------------------------------------------------- #
#: The columns the frozen block requires on the FIRST line of ``steps.jsonl``
#: (``docs/HANDOFF_whatb_2026-08-15.md`` section 4.H).  ``--loss-level 4`` adds
#: ``L_img``; the caller passes the augmented tuple.
FIRST_STEP_COLUMNS: tuple[str, ...] = (
    "L_rec",
    "L_hc",
    "L_sparse",
    "n_colors",
    "n_luts_in_batch",
    "mining_ratio",
    "n_hc_masked",
)


class StepsRowUnavailable(RuntimeError):
    """No first-step row anywhere: caller silent, disk empty, no witness.

    This is the *plumbing* failure -- "nobody handed me the row".  It is NOT a
    pass and it is NOT the same thing as :class:`LossColumnsMissing`; conflating
    the two is what made SEGSAM and PRND report a loss that had in fact run.
    """


class LossColumnsMissing(RuntimeError):
    """A row was found and it is missing required loss columns.

    This is the *substantive* failure -- "the loss really did not run", or ran
    without publishing its pre-registered columns.
    """


_WITNESS: dict[str, Any] = {}


def record_step_witness(row: Mapping[str, Any]) -> None:
    """Called by the first micro-batch: the in-process third tier.

    Needed because a quick eval can land before the trainer flushed
    ``steps.jsonl``; without it the disk tier is empty and a healthy run looks
    like a plumbing failure.
    """
    _WITNESS.clear()
    _WITNESS.update(dict(row))


def clear_step_witness() -> None:
    """Drop the in-process witness (tests, and runs that re-enter the loop)."""
    _WITNESS.clear()


def _first_line_row(path: Any) -> dict[str, Any] | None:
    try:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    return json.loads(line)
    except (OSError, ValueError):
        return None
    return None


def resolve_first_step_row(
    steps_row: Mapping[str, Any] | None = None, *, steps_path: Any = None
) -> tuple[dict[str, Any] | None, str]:
    """``(row, source)`` with ``source`` in ``caller`` / ``disk`` / ``witness`` / ``unavailable``.

    Tiers, most authoritative first.  Deliberately usable from a call site that
    knows nothing but the run directory -- that is the whole point.
    """
    if steps_row is not None:
        return dict(steps_row), "caller"
    if steps_path is not None:
        row = _first_line_row(steps_path)
        if row is not None:
            return row, "disk"
    if _WITNESS:
        return dict(_WITNESS), "witness"
    return None, "unavailable"


def assert_first_step_columns(
    required: Sequence[str] | Iterable[str] = FIRST_STEP_COLUMNS,
    *,
    steps_row: Mapping[str, Any] | None = None,
    steps_path: Any = None,
) -> tuple[dict[str, Any], str]:
    """Assert the first training step published every pre-registered column.

    Raises :class:`StepsRowUnavailable` when all three tiers came up empty and
    :class:`LossColumnsMissing` when a row exists but lacks columns.  Returns
    ``(row, source)`` so the caller can record which tier answered.
    """
    required = tuple(required)
    row, source = resolve_first_step_row(steps_row, steps_path=steps_path)
    if row is None:
        raise StepsRowUnavailable(
            "no first-step row from any tier (caller / "
            f"{steps_path!r} / in-process witness); this is a wiring failure, not a pass"
        )
    missing = [c for c in required if c not in row or row[c] is None]
    if missing:
        raise LossColumnsMissing(
            f"first step row (source={source}) is missing {missing}; present keys: {sorted(row)}"
        )
    return row, source
