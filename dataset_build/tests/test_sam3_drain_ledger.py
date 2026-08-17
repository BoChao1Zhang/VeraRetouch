"""``_drain_sam3_and_replacements``: the same drain, without the second scan.

The drain asked ``_sam3_attempts`` and ``_unconsumed_sam3_ready`` about every
pending source, four comprehensions per ``while`` iteration, and each of those
helpers answers with a full scan of ``failures``.  At L8 shape — 5,003 queued
sources against a 17,656-row journal — one iteration measured 84.2 s
(21.8 + 10.3 + 31.2 + 20.9) with the render GPUs idle behind it.  ``_Sam3Ledger``
folds both quantities in one pass and keeps them current incrementally.

Incrementally, not once per round, and that is the whole risk: **the loop writes
to the journal it is reading**.  ``_record_sam3_attempt`` appends ``sam3_attempt``
and ``sam3_ready`` rows, ``_try_ready_relabel`` appends ``sam3_ready_invalid``,
``_terminal_sam3`` appends terminal rows, and a source can go ready and then
consumed *inside one loop body*.  An index built at the top of the round would
answer with the pre-append state — faster and silently wrong, which is worse
than slow.  Every test below is therefore about staleness, in two layers:

1. ``_Sam3Ledger`` against the untouched per-source scans, fed the journal in
   arbitrary slices so a refresh boundary falls in every possible place;
2. the whole drain against ``_oracle_drain`` — the pre-hotfix body, kept here
   verbatim — run over the same scripted pipeline, comparing the journals row
   for row rather than just the final group count.
"""
from __future__ import annotations

import dataclasses
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

from construct import agent
from construct.agent import CanonicalPipeline, _Sam3Ledger


def _stub(rows: list[dict[str, Any]]) -> Any:
    """The only attribute the per-source scans touch is ``store.failures``."""
    return SimpleNamespace(store=SimpleNamespace(failures=list(rows)))


def _scan_attempts(rows: list[dict[str, Any]], source_id: str) -> int:
    return CanonicalPipeline._sam3_attempts(_stub(rows), source_id)


def _scan_ready(rows: list[dict[str, Any]], source_id: str) -> bool:
    return CanonicalPipeline._unconsumed_sam3_ready(_stub(rows), source_id)


def _row(
    source_id: str | None,
    event_type: str,
    attempt: int | None,
    *,
    stage: str = "sam3_relabel",
) -> dict[str, Any]:
    row: dict[str, Any] = {"event_type": event_type, "stage": stage}
    if source_id is not None:
        row["source_id"] = source_id
    row["attempt"] = attempt
    return row


# Every shape the fold can get wrong, in one journal: attempts out of order,
# ready/invalid interleaved, a missing ``attempt`` (``None`` -> 0), ties, counted
# events wearing the wrong stage, and rows the ledger must walk past.
FIXTURE: list[dict[str, Any]] = [
    _row("descending", "sam3_ready", 3),
    _row("descending", "sam3_ready_invalid", 3),
    _row("descending", "sam3_ready", 2),
    _row("descending", "sam3_attempt", 3),
    _row("descending", "sam3_attempt", 2),
    _row("interleaved", "sam3_ready", 1),
    _row("interleaved", "sam3_ready_invalid", 1),
    _row("interleaved", "sam3_attempt", 1),
    _row("interleaved", "sam3_ready", 2),
    _row("interleaved", "sam3_ready_invalid", 2),
    _row("interleaved", "sam3_attempt", 2),
    _row("interleaved", "sam3_ready", 3),
    # ``attempt`` absent entirely and ``attempt: None`` both read as 0.
    {"event_type": "sam3_ready", "stage": "sam3_relabel", "source_id": "no-attempt"},
    _row("no-attempt", "sam3_ready", None),
    _row("tie", "sam3_ready", 2),
    _row("tie", "sam3_ready_invalid", 2),
    _row("invalid-only", "sam3_ready_invalid", 4),
    _row("ready-then-invalid", "sam3_ready", 1),
    _row("ready-then-invalid", "sam3_ready_invalid", 2),
    _row("invalid-then-ready", "sam3_ready_invalid", 1),
    _row("invalid-then-ready", "sam3_ready", 2),
    _row("ready-zero", "sam3_ready", 0),
    _row("ready-zero", "sam3_ready_invalid", None),
    _row("duplicate-attempt", "sam3_ready", 2),
    _row("duplicate-attempt", "sam3_ready", 2),
    _row("duplicate-attempt", "sam3_ready_invalid", 1),
    _row("duplicate-attempt", "sam3_attempt", 2),
    _row("duplicate-attempt", "sam3_attempt", 2),
    # The stage asymmetry the two scans really have: ``_sam3_attempts`` demands
    # ``stage == "sam3_relabel"``, ``_unconsumed_sam3_ready`` never looks.
    _row("wrong-stage", "sam3_attempt", 1, stage="rendering"),
    _row("wrong-stage", "sam3_ready", 1, stage="rendering"),
    # Noise: unrelated events, a row with no source id, an unqueued source.
    {"event_type": "attempt", "stage": "rendering", "source_id": "tie", "attempt": 40},
    {"event_type": "queued", "stage": "sam3_relabel", "source_id": "tie",
     "error_code": "sam3_relabel_queued", "attempt": None},
    {"event_type": "landed", "stage": "landing", "message": "{}", "attempt": None},
    _row(None, "sam3_ready", 7),
    _row(None, "sam3_attempt", 7),
    _row("not-queued", "sam3_ready", 5),
]

FIXTURE_IDS = sorted(
    {str(row["source_id"]) for row in FIXTURE if row.get("source_id")}
    | {"absent-from-journal"}
)


class Sam3LedgerEquivalenceTests(unittest.TestCase):
    """Layer 1: the fold against the scans, at every refresh boundary."""

    def assert_agrees(self, ledger: _Sam3Ledger, rows: list[dict[str, Any]]) -> None:
        for source_id in FIXTURE_IDS:
            with self.subTest(source_id=source_id):
                self.assertEqual(ledger.attempts_of(source_id), _scan_attempts(rows, source_id))
                self.assertEqual(ledger.unconsumed_ready(source_id), _scan_ready(rows, source_id))

    def test_one_shot_matches_the_scans(self) -> None:
        self.assert_agrees(_Sam3Ledger().refresh(FIXTURE), FIXTURE)

    def test_the_fixture_separates_ready_from_consumed(self) -> None:
        """Guard against both implementations agreeing on an all-False answer."""
        ready = {source_id for source_id in FIXTURE_IDS if _scan_ready(FIXTURE, source_id)}
        self.assertEqual(
            ready,
            {"interleaved", "invalid-then-ready", "duplicate-attempt", "not-queued",
             "wrong-stage"},
        )
        self.assertEqual(_scan_attempts(FIXTURE, "duplicate-attempt"), 2)
        self.assertEqual(_scan_attempts(FIXTURE, "wrong-stage"), 0)

    def test_every_refresh_boundary_gives_the_same_answers(self) -> None:
        """Grow the journal one row at a time; the ledger tracks it exactly."""
        ledger = _Sam3Ledger()
        live: list[dict[str, Any]] = []
        for row in FIXTURE:
            live.append(row)
            ledger.refresh(live)
            self.assert_agrees(ledger, live)

    def test_refreshing_without_new_rows_changes_nothing(self) -> None:
        """The drain calls ``refresh`` per read; repeats must not double count."""
        live = list(FIXTURE)
        ledger = _Sam3Ledger().refresh(live)
        before = (dict(ledger.attempts), dict(ledger.maxima))
        for _ in range(5):
            ledger.refresh(live)
        self.assertEqual((ledger.attempts, ledger.maxima), before)

    def test_a_verdict_flips_the_moment_the_consuming_row_lands(self) -> None:
        """The within-one-body flip the batch loop performs, in miniature."""
        live: list[dict[str, Any]] = []
        ledger = _Sam3Ledger()
        self.assertFalse(ledger.refresh(live).unconsumed_ready("s"))
        live.append(_row("s", "sam3_attempt", 1))
        self.assertEqual(ledger.refresh(live).attempts_of("s"), 1)
        live.append(_row("s", "sam3_ready", 1))
        self.assertTrue(ledger.refresh(live).unconsumed_ready("s"))
        live.append(_row("s", "sam3_ready_invalid", 1))
        self.assertFalse(ledger.refresh(live).unconsumed_ready("s"))
        live.append(_row("s", "sam3_ready", 2))
        self.assertTrue(ledger.refresh(live).unconsumed_ready("s"))

    def test_a_replaced_or_truncated_list_rebuilds(self) -> None:
        ledger = _Sam3Ledger().refresh(list(FIXTURE))
        self.assertEqual(ledger.refresh([]).attempts, {})
        self.assertFalse(ledger.refresh([]).unconsumed_ready("interleaved"))
        other = [_row("only", "sam3_attempt", 1)]
        self.assertEqual(ledger.refresh(other).attempts, {"only": 1})
        self.assertEqual(ledger.refresh(other[:0]).attempts, {})

    def test_the_manifest_reading_is_the_same_fold(self) -> None:
        """``_sam3_ready_maxima`` now delegates here; its callers must not move."""
        maxima = CanonicalPipeline._sam3_ready_maxima(_stub(FIXTURE))
        self.assertEqual(maxima, _Sam3Ledger().refresh(FIXTURE).maxima)
        for source_id in FIXTURE_IDS:
            with self.subTest(source_id=source_id):
                self.assertEqual(
                    CanonicalPipeline._sam3_ready_beats_invalid(maxima.get(source_id)),
                    _scan_ready(FIXTURE, source_id),
                )

    def test_random_journals_agree_under_random_slicing(self) -> None:
        rng = random.Random(20260813)
        events = ["sam3_ready", "sam3_ready_invalid", "sam3_attempt", "queued", "attempt"]
        for trial in range(200):
            ids = [f"s{index}" for index in range(5)]
            rows = [
                _row(
                    rng.choice(ids + [None]),
                    rng.choice(events),
                    rng.choice([None, 0, 1, 2, 3]),
                    stage=rng.choice(["sam3_relabel", "rendering"]),
                )
                for _ in range(rng.randint(0, 30))
            ]
            ledger = _Sam3Ledger()
            live: list[dict[str, Any]] = []
            cursor = 0
            while cursor < len(rows) or not live:
                step = rng.randint(1, 4)
                live.extend(rows[cursor:cursor + step])
                cursor += step
                ledger.refresh(live)
                for source_id in ids:
                    with self.subTest(trial=trial, source_id=source_id):
                        self.assertEqual(
                            ledger.attempts_of(source_id), _scan_attempts(live, source_id)
                        )
                        self.assertEqual(
                            ledger.unconsumed_ready(source_id), _scan_ready(live, source_id)
                        )
                if cursor >= len(rows):
                    break


# --------------------------------------------------------------------------
# Layer 2: the drain itself, against the body it replaced.
# --------------------------------------------------------------------------


def _oracle_drain(self: Any) -> None:
    """``_drain_sam3_and_replacements`` exactly as it stood before the hotfix.

    Kept verbatim (only ``refresh_source_record`` / ``_empty_cuda_cache`` are
    module-qualified so the tests can patch them) so the comparison below is
    against the shipped behaviour rather than against a paraphrase of it.
    """
    by_id = {source.source_id: source for source in self.allocation.local}
    target = self.allocation.local_target
    max_attempts = self.config.masks.sam3_relabel_attempts
    while len(self._mode_groups("local")) < target:
        pending_ids = self._pending_sam3_ids()
        pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]

        made_progress = False
        for source in list(pending):
            if not self._unconsumed_sam3_ready(source.source_id):
                continue
            attempt = self._sam3_attempts(source.source_id)
            status = self._try_ready_relabel(source, attempt)
            made_progress = True
            if status == "retry" and attempt >= max_attempts:
                self._terminal_sam3(source, "ready relabel still violates canonical geometry")

        pending_ids = self._pending_sam3_ids()
        pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
        exhausted = [
            source for source in pending
            if self._sam3_attempts(source.source_id) >= max_attempts
            and not self._unconsumed_sam3_ready(source.source_id)
        ]
        for source in exhausted:
            self._terminal_sam3(source, "SAM3 relabel attempt budget exhausted")
            made_progress = True

        pending_ids = self._pending_sam3_ids()
        pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
        candidates = [
            source for source in pending
            if self._sam3_attempts(source.source_id) < max_attempts
            and not self._unconsumed_sam3_ready(source.source_id)
        ]
        if candidates:
            next_attempt = min(self._sam3_attempts(row.source_id) + 1 for row in candidates)
            batch = [
                row for row in candidates
                if self._sam3_attempts(row.source_id) + 1 == next_attempt
            ]
            agent._empty_cuda_cache()
            try:
                statuses = self.dependencies.relabeler(batch, self.config, next_attempt)
            except Exception as exc:  # noqa: BLE001
                statuses = {source.source_id: f"relabel_exception:{type(exc).__name__}:{exc}"
                            for source in batch}
            for source in batch:
                status = str(statuses.get(source.source_id) or "missing_relabel_status")
                self._record_sam3_attempt(
                    source, next_attempt, f"sam3_{status}", status
                )
                refreshed, reason = agent.refresh_source_record(source)
                if refreshed is None:
                    if next_attempt >= max_attempts:
                        self._terminal_sam3(source, f"{status}; integrity={reason}")
                    continue
                self._record_sam3_attempt(
                    source, next_attempt, "sam3_relabel_ready", status,
                    event_type="sam3_ready",
                )
                result = self._try_ready_relabel(source, next_attempt)
                if result == "retry" and next_attempt >= max_attempts:
                    self._terminal_sam3(source, "relabel cannot satisfy canonical geometry")
            made_progress = True

        pending_count = len(self._pending_sam3_ids())
        before = len(self._mode_groups("local"))
        if before + pending_count < target:
            self._fill_mode("local", self.allocation.local, target)
            made_progress = made_progress or len(self._mode_groups("local")) > before \
                or len(self._pending_sam3_ids()) > pending_count
        if not made_progress:
            break

    if self._pending_sam3_ids():
        raise agent.PipelineError("SAM3 relabel queue remains unresolved")


class _DrainLoopGuard(Exception):
    """The scripted pipeline refusing to spin forever — an outcome, not a bug.

    Some scripts (a resumed ``sam3_ready`` whose geometry keeps failing at an
    attempt below the one the ready row carries) make the *original* loop spin:
    ``made_progress`` stays true and nothing consumes the ready row.  That is
    pre-existing behaviour and not this hotfix's business, so the guard is
    compared like any other outcome — both bodies must spin the same way.
    """


@dataclasses.dataclass(frozen=True)
class _Plan:
    """One source's scripted behaviour, indexed by attempt (last entry repeats).

    ``relabel`` is what the SAM3 stub reports, ``refresh`` whether
    ``refresh_source_record`` then accepts the source, and ``outcome`` what
    ``_try_ready_relabel`` makes of it.  ``queued`` false means the source is not
    in the journal when the drain starts; ``preready`` seeds an unconsumed
    ``sam3_ready`` row, the state a crashed build resumes into.
    """

    relabel: tuple[str, ...] = ("ready",)
    refresh: tuple[bool, ...] = (True,)
    outcome: tuple[str, ...] = ("completed",)
    queued: bool = True
    preready: int | None = None
    preattempts: int = 0

    @staticmethod
    def _at(values: tuple[Any, ...], attempt: int) -> Any:
        return values[min(max(attempt - 1, 0), len(values) - 1)]


class _ScriptedDrain(CanonicalPipeline):
    """A pipeline stripped to what the drain touches, driven by ``_Plan``s.

    Subclassed rather than stubbed so ``_pending_sam3_ids`` and
    ``_terminal_source_ids`` — the accounting the drain interleaves with the two
    ledger readings — are the real ones.  The journal rows carry the fields the
    real ``_failure`` writes for these events and nothing else.
    """

    MAX_LOOKS = 60

    def __init__(self, plans: dict[str, _Plan], target: int, max_attempts: int) -> None:
        self.plans = plans
        self.groups: list[str] = []
        self.trace: list[str] = []
        self.looks = 0
        self.last_attempt: dict[str, int] = {}
        sources = [SimpleNamespace(source_id=source_id) for source_id in sorted(plans)]
        rows: list[dict[str, Any]] = []
        for source in sources:
            plan = plans[source.source_id]
            if plan.queued:
                rows.append({
                    "event_type": "queued", "stage": "sam3_relabel",
                    "error_code": "sam3_relabel_queued", "source_id": source.source_id,
                    "attempt": None, "terminal": False,
                })
            for attempt in range(1, plan.preattempts + 1):
                rows.append(_row(source.source_id, "sam3_attempt", attempt))
                self.last_attempt[source.source_id] = attempt
            if plan.preready is not None:
                rows.append(_row(source.source_id, "sam3_ready", plan.preready))
        self.store = SimpleNamespace(
            failures=rows, completed_sources=lambda: {row for row in self.groups}
        )
        self.allocation = SimpleNamespace(local=sources, local_target=target)
        # ``output_root`` is real state rather than an override of ``_stop_fill``:
        # the drain now asks whether the build has been told to stop, and giving
        # it a root with no marker keeps the answer False through the real
        # predicate, so this comparison still exercises the code that ships.
        # A stop-fill drain is a different subject and lives in test_stop_fill.py.
        self.config = SimpleNamespace(
            masks=SimpleNamespace(sam3_relabel_attempts=max_attempts),
            output_root=Path(tempfile.gettempdir()) / "scripted-drain-no-stop-fill",
        )
        self._stop_fill_latched = False
        self._stop_fill_at_start = False
        self._stop_fill_recorded = False
        self.dependencies = SimpleNamespace(relabeler=self._relabel)

    def _mode_groups(self, mode: str) -> list[dict[str, Any]]:
        self.looks += 1
        if self.looks > self.MAX_LOOKS:
            raise _DrainLoopGuard("scripted drain did not terminate")
        return [{"group_id": group_id} for group_id in self.groups]

    def refresh_source(self, source: Any) -> tuple[Any, str]:
        """Stands in for ``refresh_source_record``, scripted by attempt.

        The drain calls it immediately after recording the attempt, so the
        attempt number is the one this pipeline just journalled — which keeps
        the stub a pure function of the script rather than of call order, and
        therefore identical across the two bodies under comparison.
        """
        attempt = self.last_attempt.get(source.source_id, 1)
        ok = bool(_Plan._at(self.plans[source.source_id].refresh, attempt))
        return (source, "ok") if ok else (None, "subject mask missing")

    def _relabel(self, batch: list[Any], _config: Any, attempt: int) -> dict[str, str]:
        self.trace.append(
            f"relabel@{attempt}:{','.join(source.source_id for source in batch)}"
        )
        return {
            source.source_id: _Plan._at(self.plans[source.source_id].relabel, attempt)
            for source in batch
        }

    def _record_sam3_attempt(
        self, source: Any, attempt: int, code: str, message: object,
        *, event_type: str = "sam3_attempt",
    ) -> None:
        self.trace.append(f"{event_type}@{attempt}:{source.source_id}:{code}")
        if event_type == "sam3_attempt":
            self.last_attempt[source.source_id] = attempt
        self.store.failures.append({
            "event_type": event_type, "stage": "sam3_relabel",
            "source_id": source.source_id, "attempt": attempt,
            "error_code": code, "terminal": False,
        })

    def _terminal_sam3(self, source: Any, message: object) -> None:
        self.trace.append(f"terminal:{source.source_id}:{message}")
        self.store.failures.append({
            "event_type": "terminal", "stage": "sam3_relabel",
            "source_id": source.source_id, "attempt": None,
            "error_code": "sam3_relabel_failed", "terminal": True,
        })

    def _try_ready_relabel(self, source: Any, attempt: int) -> str:
        """The real one's journal contract: a retry consumes the ready row."""
        outcome = str(_Plan._at(self.plans[source.source_id].outcome, attempt))
        self.trace.append(f"ready_relabel@{attempt}:{source.source_id}->{outcome}")
        if outcome == "completed":
            self.groups.append(source.source_id)
        elif outcome == "retry":
            self._record_sam3_attempt(
                source, attempt, "sam3_geometry_failed", "mask_failed:",
                event_type="sam3_ready_invalid",
            )
        else:
            self._terminal_sam3(source, "render terminal")
        return outcome

    def _fill_mode(self, mode: str, records: list[Any], target: int) -> int:
        """No replacement pool: the drain's last branch must still be exercised."""
        self.trace.append(f"fill:{mode}:{len(self.groups)}")
        return 0

    def journal(self) -> list[dict[str, Any]]:
        return list(self.store.failures)


def _run_one(
    body: Callable[[Any], None], plans: dict[str, _Plan], target: int, max_attempts: int
) -> Any:
    """One body over a fresh scripted pipeline, with its own dependency stubs."""
    pipeline = _ScriptedDrain(plans, target, max_attempts)
    error: Exception | None = None
    # ``_empty_cuda_cache`` would import torch for nothing, and the real
    # ``refresh_source_record`` wants a populated cache directory on disk.
    with mock.patch.object(agent, "_empty_cuda_cache", lambda: None), \
            mock.patch.object(agent, "refresh_source_record", pipeline.refresh_source):
        try:
            body(pipeline)
        except (agent.PipelineError, _DrainLoopGuard) as exc:
            error = exc
    return SimpleNamespace(pipeline=pipeline, error=error)


class _ScriptedDrainCase(unittest.TestCase):
    def assert_identical(self, plans: dict[str, _Plan], target: int, max_attempts: int) -> Any:
        old = _run_one(_oracle_drain, plans, target, max_attempts)
        new = _run_one(
            CanonicalPipeline._drain_sam3_and_replacements, plans, target, max_attempts
        )
        self.assertEqual(new.pipeline.trace, old.pipeline.trace)
        self.assertEqual(new.pipeline.journal(), old.pipeline.journal())
        self.assertEqual(new.pipeline.groups, old.pipeline.groups)
        self.assertEqual(type(new.error), type(old.error))
        self.assertEqual(str(new.error or ""), str(old.error or ""))
        # Rounds too, not just the outcome: a ledger refreshed once per round
        # instead of once per read reaches the same end state by spending an
        # extra round on work it could not see yet, and that is the failure
        # mode this whole file exists to catch.
        self.assertEqual(new.pipeline.looks, old.pipeline.looks)
        return new.pipeline


class DrainEquivalenceTests(_ScriptedDrainCase):
    """Layer 2: the drain, row for row, against the body it replaced."""

    def test_a_plain_relabel_round(self) -> None:
        pipeline = self.assert_identical(
            {f"s{index}": _Plan() for index in range(4)}, target=4, max_attempts=2
        )
        self.assertEqual(len(pipeline.groups), 4)
        self.assertIn("relabel@1:s0,s1,s2,s3", pipeline.trace)

    def test_a_ready_consumed_inside_the_same_loop_body(self) -> None:
        """The case a per-round index gets wrong: ready then invalid at one attempt.

        ``s_flip`` is handed back ready at attempt 1, fails geometry (which
        writes ``sam3_ready_invalid`` at the same attempt) and must therefore
        still be a *candidate* for attempt 2 rather than a ready source waiting
        to be retried — a stale index would see the ready row and neither.
        """
        pipeline = self.assert_identical(
            {
                "s_flip": _Plan(
                    relabel=("ready", "ready"), refresh=(True, True),
                    outcome=("retry", "completed"),
                ),
                "s_ok": _Plan(),
            },
            target=2, max_attempts=3,
        )
        self.assertEqual(sorted(pipeline.groups), ["s_flip", "s_ok"])
        self.assertIn("ready_relabel@2:s_flip->completed", pipeline.trace)

    def test_a_source_that_never_becomes_ready_exhausts_its_budget(self) -> None:
        pipeline = self.assert_identical(
            {
                "s_bad": _Plan(relabel=("no_instance",), refresh=(False,)),
                "s_ok": _Plan(),
            },
            target=2, max_attempts=2,
        )
        self.assertEqual(pipeline.groups, ["s_ok"])
        self.assertIn("terminal:s_bad:no_instance; integrity=subject mask missing",
                      pipeline.trace)

    def test_a_resumed_journal_with_an_unconsumed_ready_row(self) -> None:
        """The first branch of the loop — a ready row from a previous process."""
        pipeline = self.assert_identical(
            {
                "s_pre": _Plan(preready=1, outcome=("completed",)),
                "s_new": _Plan(),
            },
            target=2, max_attempts=2,
        )
        self.assertEqual(pipeline.trace[0], "ready_relabel@0:s_pre->completed")

    def test_a_resumed_ready_row_is_consumed_and_retried_in_one_round(self) -> None:
        """The read-after-write inside a single round, and the reason for it.

        A build that died between journalling ``sam3_ready`` at attempt 1 and
        checking the geometry resumes with an unconsumed ready row *and* an
        attempt already spent.  The drain's first branch consumes it — writing
        ``sam3_ready_invalid`` at that same attempt — and the ``candidates``
        comprehension two branches later must already see the source as not
        ready, so its next relabel goes out in this round.  Read the ledger once
        per round instead and the source sits out a round: same end state, one
        wasted iteration, and the ``made_progress`` bookkeeping around it moves.
        """
        pipeline = self.assert_identical(
            {
                "s_resumed": _Plan(
                    preattempts=1, preready=1,
                    relabel=("ready", "ready"), refresh=(True, True),
                    outcome=("retry", "completed"),
                ),
            },
            target=1, max_attempts=3,
        )
        self.assertEqual(pipeline.groups, ["s_resumed"])
        self.assertEqual(
            [line for line in pipeline.trace if line.startswith(("ready_relabel", "relabel"))],
            ["ready_relabel@1:s_resumed->retry", "relabel@2:s_resumed",
             "ready_relabel@2:s_resumed->completed"],
        )
        # Everything above happened in one round: the ``while`` test, the
        # ``before`` read in the fill branch, and the ``while`` test that ends
        # the loop are the only three times the drain counts local groups.
        self.assertEqual(pipeline.looks, 3)

    def test_a_resumed_ready_row_next_to_a_fresh_source(self) -> None:
        """The same round, with a second source moving ``next_attempt`` under it."""
        self.assert_identical(
            {
                "s_resumed": _Plan(
                    preattempts=2, preready=2,
                    relabel=("ready",), refresh=(True,),
                    outcome=("retry", "retry", "completed"),
                ),
                "s_fresh": _Plan(outcome=("retry", "completed")),
            },
            target=2, max_attempts=4,
        )

    def test_a_preready_source_that_keeps_failing_goes_terminal(self) -> None:
        """Attempt counting and the ready branch disagreeing is the point here."""
        self.assert_identical(
            {"s_pre": _Plan(preready=3, outcome=("retry",), relabel=("ready",),
                            refresh=(True,))},
            target=1, max_attempts=1,
        )

    def test_sources_at_different_attempt_counts_batch_separately(self) -> None:
        """``next_attempt`` is a minimum over the ledger, so it must move with it."""
        pipeline = self.assert_identical(
            {
                "s_slow": _Plan(
                    relabel=("no_instance", "ready"), refresh=(False, True),
                    outcome=("completed",),
                ),
                "s_fast": _Plan(),
                "s_pre": _Plan(preready=2, outcome=("retry", "completed"),
                               relabel=("ready",), refresh=(True,)),
            },
            target=3, max_attempts=3,
        )
        batches = [line for line in pipeline.trace if line.startswith("relabel@")]
        self.assertTrue(any("s_slow" in line for line in batches))

    def test_a_source_terminal_on_render_stops_the_loop(self) -> None:
        """Terminal by ``_try_ready_relabel``, not by the budget."""
        pipeline = self.assert_identical(
            {"s": _Plan(relabel=("ready",), refresh=(True,), outcome=("terminal",))},
            target=2, max_attempts=2,
        )
        self.assertEqual(pipeline.groups, [])

    def test_a_queue_left_pending_raises_in_both_bodies(self) -> None:
        """The target is met while a source is still queued — the drain's guard."""
        old = _run_one(
            _oracle_drain,
            {"s_ok": _Plan(), "s_wait": _Plan(relabel=("no_instance",), refresh=(False,))},
            target=1, max_attempts=3,
        )
        self.assertIsInstance(old.error, agent.PipelineError)
        self.assert_identical(
            {"s_ok": _Plan(), "s_wait": _Plan(relabel=("no_instance",), refresh=(False,))},
            target=1, max_attempts=3,
        )

    def test_an_empty_queue_is_a_no_op_in_both(self) -> None:
        self.assert_identical({"s": _Plan(queued=False)}, target=1, max_attempts=2)

    def test_random_scripts_agree(self) -> None:
        """Fuzz the interleavings the hand-written cases cannot enumerate."""
        rng = random.Random(20260813)
        outcomes = ("completed", "retry", "terminal")
        for trial in range(120):
            plans = {}
            for index in range(rng.randint(1, 5)):
                plans[f"s{index}"] = _Plan(
                    relabel=tuple(
                        rng.choice(["ready", "no_instance", "low_score"])
                        for _ in range(rng.randint(1, 3))
                    ),
                    refresh=tuple(
                        rng.random() < 0.7 for _ in range(rng.randint(1, 3))
                    ),
                    outcome=tuple(
                        rng.choice(outcomes) for _ in range(rng.randint(1, 3))
                    ),
                    queued=rng.random() < 0.9,
                    preready=rng.choice([None, None, 0, 1, 2]),
                    preattempts=rng.choice([0, 0, 1, 2]),
                )
            with self.subTest(trial=trial):
                self.assert_identical(
                    plans,
                    target=rng.randint(1, len(plans) + 1),
                    max_attempts=rng.randint(1, 3),
                )


class DrainScanCountTests(_ScriptedDrainCase):
    """The point of the exercise: the drain stops rescanning the journal.

    Counted rather than timed, so the assertion means the same thing on a busy
    box.  The old body's scans are ``O(pending x journal)`` per round; the
    ledger's are ``O(rows appended since the last read)``, which over a whole
    drain is one pass plus the rows the drain itself wrote.
    """

    def test_the_ledger_reads_each_row_once_per_drain(self) -> None:
        plans = {f"s{index:02d}": _Plan() for index in range(30)}
        reads = {"n": 0}
        real_refresh = _Sam3Ledger.refresh

        def counting_refresh(ledger: _Sam3Ledger, failures: list[dict[str, Any]]) -> Any:
            fresh = len(failures) if ledger._rows is not failures \
                else len(failures) - ledger._pos
            reads["n"] += max(fresh, 0)
            return real_refresh(ledger, failures)

        with mock.patch.object(_Sam3Ledger, "refresh", counting_refresh):
            new = _run_one(
                CanonicalPipeline._drain_sam3_and_replacements, plans, 30, 2
            )
        self.assertEqual(len(new.pipeline.groups), 30)
        # One pass over whatever the journal grew to, not one per source.
        self.assertLessEqual(reads["n"], len(new.pipeline.journal()))

        scans = {"n": 0}
        real_attempts = CanonicalPipeline._sam3_attempts
        real_ready = CanonicalPipeline._unconsumed_sam3_ready

        def counting_attempts(pipe: Any, source_id: str) -> int:
            scans["n"] += len(pipe.store.failures)
            return real_attempts(pipe, source_id)

        def counting_ready(pipe: Any, source_id: str) -> bool:
            scans["n"] += 2 * len(pipe.store.failures)
            return real_ready(pipe, source_id)

        with mock.patch.object(CanonicalPipeline, "_sam3_attempts", counting_attempts), \
                mock.patch.object(CanonicalPipeline, "_unconsumed_sam3_ready", counting_ready):
            old = _run_one(_oracle_drain, plans, 30, 2)
        self.assertEqual(old.pipeline.groups, new.pipeline.groups)
        self.assertGreater(scans["n"], 20 * reads["n"])


if __name__ == "__main__":  # pragma: no cover - parity with the sibling suites
    unittest.main()
