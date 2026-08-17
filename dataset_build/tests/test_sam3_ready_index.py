"""``_sam3_ready_maxima``: the manifest's SAM3 accounting, in one journal pass.

``_manifest`` used to ask ``_unconsumed_sam3_ready`` about every queued source,
and that helper answers each question with two full scans of ``failures``.  At
L8 shape — 5,003 ``sam3_relabel_queued`` sources against 17,656 journal rows —
each land checkpoint therefore spent ~1.7e8 dict reads on the main loop while
the render threads waited on it, and throughput fell from 2,800 groups/h to
120 groups/h.

The replacement is a prebuilt per-source maximum, so the only thing worth
testing is that it is the *same* verdict.  ``_unconsumed_sam3_ready`` is kept
in the module for single-source callers, which makes it the oracle here: every
case below asserts the two agree id by id, over journals built to hit the parts
where a "max" is easy to get wrong — attempts out of order, ready and
ready_invalid interleaved, a missing ``attempt`` (``None`` -> 0), and ties.
"""
from __future__ import annotations

import random
import unittest
from types import SimpleNamespace
from typing import Any

from construct.agent import CanonicalPipeline


def _stub(rows: list[dict[str, Any]]) -> Any:
    """The only attribute either implementation touches is ``store.failures``."""
    return SimpleNamespace(store=SimpleNamespace(failures=list(rows)))


def _old(rows: list[dict[str, Any]], source_id: str) -> bool:
    return CanonicalPipeline._unconsumed_sam3_ready(_stub(rows), source_id)


def _new(rows: list[dict[str, Any]], source_id: str) -> bool:
    maxima = CanonicalPipeline._sam3_ready_maxima(_stub(rows))
    return CanonicalPipeline._sam3_ready_beats_invalid(maxima.get(source_id))


def _row(source_id: str | None, event_type: str, attempt: int | None) -> dict[str, Any]:
    row: dict[str, Any] = {"event_type": event_type, "stage": "sam3_relabel"}
    if source_id is not None:
        row["source_id"] = source_id
    row["attempt"] = attempt
    return row


# One journal that carries every shape at once, so the equivalence assertions
# below are made against a single fixture rather than a case per property.
FIXTURE: list[dict[str, Any]] = [
    # Queue rows: what ``_manifest`` derives ``sam3_expected_ids`` from.  They
    # carry no ``attempt`` and neither implementation may count them as ready.
    *[
        {
            "event_type": "queued",
            "stage": "sam3_relabel",
            "error_code": "sam3_relabel_queued",
            "source_id": sid,
            "attempt": None,
        }
        for sid in (
            "descending", "interleaved", "no-attempt", "tie", "invalid-only",
            "never-attempted", "ready-then-invalid", "invalid-then-ready",
            "ready-zero", "duplicate-attempt",
        )
    ],
    # Attempts arriving out of order: 3 lands before 2, so a "last row wins"
    # reading would call this consumed and a max reading would not.
    _row("descending", "sam3_ready", 3),
    _row("descending", "sam3_ready_invalid", 3),
    _row("descending", "sam3_ready", 2),
    # Ready and invalid alternating, highest ready last.
    _row("interleaved", "sam3_ready", 1),
    _row("interleaved", "sam3_ready_invalid", 1),
    _row("interleaved", "sam3_ready", 2),
    _row("interleaved", "sam3_ready_invalid", 2),
    _row("interleaved", "sam3_ready", 3),
    # ``attempt`` absent entirely and ``attempt: None`` both read as 0, so this
    # source is ready at 0 with no invalid — which is *not* unconsumed, because
    # the original compares ``0 > 0``.
    {"event_type": "sam3_ready", "stage": "sam3_relabel", "source_id": "no-attempt"},
    _row("no-attempt", "sam3_ready", None),
    # Equal maxima: the original uses a strict ``>``.
    _row("tie", "sam3_ready", 2),
    _row("tie", "sam3_ready_invalid", 2),
    # Invalid with no ready at all.
    _row("invalid-only", "sam3_ready_invalid", 4),
    # "never-attempted" is queued and has no SAM3 event rows whatsoever.
    _row("ready-then-invalid", "sam3_ready", 1),
    _row("ready-then-invalid", "sam3_ready_invalid", 2),
    _row("invalid-then-ready", "sam3_ready_invalid", 1),
    _row("invalid-then-ready", "sam3_ready", 2),
    _row("ready-zero", "sam3_ready", 0),
    _row("ready-zero", "sam3_ready_invalid", None),
    _row("duplicate-attempt", "sam3_ready", 2),
    _row("duplicate-attempt", "sam3_ready", 2),
    _row("duplicate-attempt", "sam3_ready_invalid", 1),
    # Noise the index has to walk past: other event types on the same sources,
    # a row with no ``source_id``, and a source that was never queued.
    _row("interleaved", "sam3_attempt", 9),
    {"event_type": "attempt", "stage": "rendering", "source_id": "tie", "attempt": 40},
    {"event_type": "landed", "stage": "landing", "message": "{}", "attempt": None},
    _row(None, "sam3_ready", 7),
    _row("not-queued", "sam3_ready", 5),
]

EXPECTED_IDS = {
    str(row["source_id"])
    for row in FIXTURE
    if row.get("error_code") == "sam3_relabel_queued" and row.get("source_id")
}


class Sam3ReadyIndexTests(unittest.TestCase):
    def test_the_index_and_the_per_source_scan_agree_id_by_id(self) -> None:
        maxima = CanonicalPipeline._sam3_ready_maxima(_stub(FIXTURE))
        ids = sorted(EXPECTED_IDS | {"not-queued", "absent-from-journal"})
        for source_id in ids:
            with self.subTest(source_id=source_id):
                self.assertEqual(
                    CanonicalPipeline._sam3_ready_beats_invalid(maxima.get(source_id)),
                    _old(FIXTURE, source_id),
                )

    def test_the_manifest_set_is_unchanged(self) -> None:
        """The exact expression ``_manifest`` computes, old path against new."""
        maxima = CanonicalPipeline._sam3_ready_maxima(_stub(FIXTURE))
        self.assertEqual(
            {
                source_id for source_id in EXPECTED_IDS
                if CanonicalPipeline._sam3_ready_beats_invalid(maxima.get(source_id))
            },
            {source_id for source_id in EXPECTED_IDS if _old(FIXTURE, source_id)},
        )

    def test_the_fixture_actually_separates_ready_from_consumed(self) -> None:
        """Guard against both implementations agreeing on an all-False answer."""
        completed = {source_id for source_id in EXPECTED_IDS if _old(FIXTURE, source_id)}
        self.assertEqual(completed, {"interleaved", "invalid-then-ready", "duplicate-attempt"})

    def test_the_maxima_are_the_maxima_not_the_last_row(self) -> None:
        maxima = CanonicalPipeline._sam3_ready_maxima(_stub(FIXTURE))
        self.assertEqual(maxima["descending"], (3, 3))
        self.assertEqual(maxima["interleaved"], (3, 2))
        self.assertEqual(maxima["no-attempt"], (0, None))
        self.assertEqual(maxima["invalid-only"], (None, 4))
        self.assertNotIn("never-attempted", maxima)
        self.assertNotIn("None", maxima, "a row with no source_id must not be keyed")

    def test_an_empty_journal_completes_nothing(self) -> None:
        self.assertEqual(CanonicalPipeline._sam3_ready_maxima(_stub([])), {})
        self.assertFalse(CanonicalPipeline._sam3_ready_beats_invalid(None))

    def test_random_journals_agree(self) -> None:
        """Fuzz the orderings the hand-written fixture cannot enumerate."""
        rng = random.Random(20260813)
        for trial in range(200):
            ids = [f"s{index}" for index in range(6)]
            rows = [
                _row(
                    rng.choice(ids),
                    rng.choice(["sam3_ready", "sam3_ready_invalid", "sam3_attempt"]),
                    rng.choice([None, 0, 1, 2, 3]),
                )
                for _ in range(rng.randint(0, 24))
            ]
            maxima = CanonicalPipeline._sam3_ready_maxima(_stub(rows))
            with self.subTest(trial=trial):
                self.assertEqual(
                    {
                        source_id for source_id in ids
                        if CanonicalPipeline._sam3_ready_beats_invalid(maxima.get(source_id))
                    },
                    {source_id for source_id in ids if _old(rows, source_id)},
                )


if __name__ == "__main__":  # pragma: no cover - parity with the sibling suites
    unittest.main()
