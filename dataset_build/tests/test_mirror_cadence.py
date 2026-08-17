"""HOTFIX-MirrorCadence-3: incremental ledger mirroring and the land cadence.

Two independent mechanisms, one cause.  ``_land_checkpoint`` used to fire on
every committed group whenever the staged level sat above the water mark (which
on prod-l8 it permanently does, because the counter includes the 8.5 GiB
prefetch buffer), and each firing re-copied the *whole* 391 MB ``groups.jsonl``
to NFS.  The cadence gate stops the firing; the incremental mirror makes each
firing cost the delta.

Nothing here touches the real NFS mount: every mirror in this file is a
directory under ``tempfile``.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from construct import agent
from construct.agent import (
    CanonicalPipeline,
    mirror_artifacts,
    restore_mirror,
    run,
)
from construct.state import ArtifactStore, scan_jsonl

from dataset_build.tests.test_land_integration import LandFixture


BASE = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


def _group(index: int) -> dict:
    """A durable group record: real shape, so a restored ledger really opens."""
    return {
        "build_id": "build",
        "group_id": f"group-{index:05d}",
        "candidates": [
            {"candidate_id": f"group-{index:05d}-candidate-{slot}", "payload": "x" * 32}
            for slot in range(8)
        ],
        "winner_ids": [],
    }


def _row(index: int) -> str:
    """One ledger record, deliberately fat enough that a delta is visible."""
    return json.dumps(_group(index), sort_keys=True, separators=(",", ":"))


class MirrorFixture(unittest.TestCase):
    """An output root holding the three artifacts, and a mirror beside it."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "out"
        self.output.mkdir()
        self.mirror = self.root / "mirror"
        (self.output / "manifest.json").write_text(
            json.dumps({"build_id": "build"}), encoding="utf-8"
        )
        (self.output / "sft.jsonl").write_bytes(b"")
        self.ledger = self.output / "groups.jsonl"
        self.ledger.write_bytes(b"")
        self.append(range(0, 5))

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def append(self, indexes) -> None:
        with self.ledger.open("ab") as handle:
            for index in indexes:
                handle.write((_row(index) + "\n").encode("utf-8"))

    def mirror_once(self) -> agent.MirrorReport:
        return mirror_artifacts(self.output, self.mirror)

    def assert_mirror_matches(self) -> None:
        for name in ("groups.jsonl", "sft.jsonl", "manifest.json"):
            self.assertEqual(
                (self.mirror / name).read_bytes(),
                (self.output / name).read_bytes(),
                f"{name} diverged from its source",
            )
        state = json.loads((self.mirror / agent.MIRROR_STATE_NAME).read_text())
        for name, entry in state["files"].items():
            self.assertEqual(entry["bytes"], (self.mirror / name).stat().st_size)

    def copies(self):
        """Spy that records which artifacts took the whole-file path."""
        seen: list[str] = []
        real = agent._atomic_copy

        def spy(source, target, *, limit=None):
            seen.append(Path(target).name)
            return real(source, target, limit=limit)

        return seen, mock.patch.object(agent, "_atomic_copy", spy)


class IncrementalMirrorTests(MirrorFixture):
    """The mirror is a byte-exact prefix of the source after every pass."""

    def test_first_pass_copies_and_second_pass_only_appends(self) -> None:
        first = self.mirror_once()
        self.assert_mirror_matches()
        # Nothing was journalled: a mirror that does not exist yet is not an
        # anomaly, it is the first checkpoint.
        self.assertEqual(first.events, ())

        before = (self.mirror / "groups.jsonl").stat().st_size
        self.append(range(5, 9))
        seen, patch = self.copies()
        with patch:
            report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(report.events, ())
        # Only the manifest took a whole-file copy; the ledgers were extended.
        self.assertEqual(seen, ["manifest.json"])
        self.assertEqual(
            report.appended_bytes,
            (self.mirror / "groups.jsonl").stat().st_size - before,
        )
        self.assertEqual(report.copied_bytes, (self.output / "manifest.json").stat().st_size)

    def test_an_absent_state_file_scrubs_once_and_then_appends(self) -> None:
        """The live-build migration path: a mirror written by the old whole-file
        code has no state file, so its position in the scrub cadence is unknown
        and the tail proves only the tail — it is copied whole once, and every
        checkpoint after that is an ordinary append (HOTFIX-MirrorScrub-5)."""
        self.mirror_once()
        (self.mirror / agent.MIRROR_STATE_NAME).unlink()
        self.append(range(5, 7))

        seen, patch = self.copies()
        with patch:
            report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(seen, ["groups.jsonl", "sft.jsonl", "manifest.json"])
        self.assertEqual([event["reason"] for event in report.events], ["scrub", "scrub"])

        self.append(range(7, 9))
        seen, patch = self.copies()
        with patch:
            report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(seen, ["manifest.json"])
        self.assertEqual(report.events, ())

    def test_a_changed_prefix_falls_back_to_a_whole_copy(self) -> None:
        self.mirror_once()
        target = self.mirror / "groups.jsonl"
        corrupt = bytearray(target.read_bytes())
        corrupt[10] = corrupt[10] ^ 0x20  # same length, different bytes
        target.write_bytes(bytes(corrupt))
        self.append(range(5, 7))

        report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual([event["reason"] for event in report.events], ["prefix_diverged"])
        self.assertEqual(report.events[0]["action"], "full_copy")
        self.assertEqual(report.events[0]["artifact"], "groups.jsonl")

    def test_a_truncated_mirror_falls_back_to_a_whole_copy(self) -> None:
        self.mirror_once()
        target = self.mirror / "groups.jsonl"
        with target.open("r+b") as handle:
            handle.truncate(target.stat().st_size // 2)
        self.append(range(5, 7))

        report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual([event["reason"] for event in report.events], ["mirror_truncated"])

    def test_a_deleted_mirror_with_recorded_state_falls_back_and_is_journalled(self) -> None:
        self.mirror_once()
        (self.mirror / "groups.jsonl").unlink()

        report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual([event["reason"] for event in report.events], ["mirror_missing"])

    def test_an_offset_past_the_source_falls_back_to_a_whole_copy(self) -> None:
        """A resumed build writing a shorter file under the same name: the
        recorded offset now points past the end of the source."""
        self.mirror_once()
        self.ledger.write_bytes((_row(99) + "\n").encode("utf-8"))

        report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(
            [event["reason"] for event in report.events], ["source_shorter_than_mirror"]
        )
        self.assertEqual(scan_jsonl(self.mirror / "groups.jsonl").records[0]["group_id"],
                         "group-00099")

    def test_an_interrupted_append_is_rolled_back_and_reapplied(self) -> None:
        """A crash between the append and the state file leaves the mirror longer
        than its own recorded length; those bytes are unclaimed, not corrupt."""
        self.mirror_once()
        committed = (self.mirror / "groups.jsonl").stat().st_size
        with (self.mirror / "groups.jsonl").open("ab") as handle:
            handle.write(b'{"build_id":"build","group_id":"group-0000')  # half a record
        self.append(range(5, 8))

        seen, patch = self.copies()
        with patch:
            report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(seen, ["manifest.json"])  # repaired without a whole copy
        self.assertEqual([event["reason"] for event in report.events], ["interrupted_append"])
        self.assertEqual(report.events[0]["action"], "rollback")
        self.assertEqual(report.events[0]["committed"], committed)

    def test_state_is_reread_from_disk_so_a_restart_still_appends(self) -> None:
        self.mirror_once()
        self.append(range(5, 7))
        seen, patch = self.copies()
        with patch:  # a fresh call, exactly as a restarted process would make it
            mirror_artifacts(Path(str(self.output)), Path(str(self.mirror)))
        self.assertEqual(seen, ["manifest.json"])
        self.assert_mirror_matches()

    def test_unparsable_state_costs_a_copy_and_nothing_else(self) -> None:
        self.mirror_once()
        (self.mirror / agent.MIRROR_STATE_NAME).write_text("{not json", encoding="utf-8")
        self.append(range(5, 7))
        report = self.mirror_once()
        self.assert_mirror_matches()
        # One whole copy of each ledger — a state file nobody can read carries no
        # cadence position either, and an unverified prefix is exactly what the
        # scrub exists to re-establish.
        self.assertEqual([event["reason"] for event in report.events], ["scrub", "scrub"])

    def test_a_nonsense_offset_is_refused_rather_than_appended_behind(self) -> None:
        """``true`` is an ``int`` in Python, and offset 1 would make the tail
        check compare a single byte — which passes on almost any two JSON
        ledgers.  Every non-integer offset costs a whole copy instead."""
        for value in (True, 1.5, "12", None, -1):
            with self.subTest(offset=value):
                self.mirror_once()
                state = json.loads((self.mirror / agent.MIRROR_STATE_NAME).read_text())
                state["files"]["groups.jsonl"]["bytes"] = value
                (self.mirror / agent.MIRROR_STATE_NAME).write_text(
                    json.dumps(state), encoding="utf-8"
                )
                self.append(range(20, 22))
                report = self.mirror_once()
                self.assert_mirror_matches()
                self.assertEqual(
                    [event["reason"] for event in report.events], ["mirror_state_unusable"]
                )


class PeriodicScrubTests(MirrorFixture):
    """HOTFIX-MirrorScrub-5 / review N-M1: the 64 KiB tail is a *window*, so
    damage before it is neither detected nor repaired — the old whole-file copy
    healed it at every checkpoint and the incremental mirror never does.  Every
    ``MIRROR_SCRUB_EVERY``-th checkpoint copies whole to bound that.

    The ledger here is deliberately larger than ``MIRROR_VERIFY_WINDOW``: at the
    fixture's default 3.9 KB the window covers the whole file, every corruption
    is caught by the ordinary prefix check, and a scrub proves nothing.
    """

    # The review's own probe: byte 100 of a mirror far larger than the window.
    DAMAGE_AT = 100

    def setUp(self) -> None:
        super().setUp()
        self.append(range(5, 200))
        self.assertGreater(
            self.ledger.stat().st_size, agent.MIRROR_VERIFY_WINDOW + self.DAMAGE_AT
        )

    def state(self) -> dict:
        return json.loads((self.mirror / agent.MIRROR_STATE_NAME).read_text())

    def set_checkpoints(self, value: int) -> None:
        """Rewrite only the cadence counter, as a restart would find it."""
        state = self.state()
        state["checkpoints"] = value
        (self.mirror / agent.MIRROR_STATE_NAME).write_text(
            json.dumps(state), encoding="utf-8"
        )

    def flip(self) -> None:
        target = self.mirror / "groups.jsonl"
        data = bytearray(target.read_bytes())
        data[self.DAMAGE_AT] ^= 0x20
        target.write_bytes(bytes(data))

    def byte_at(self, path: Path) -> int:
        with path.open("rb") as handle:
            handle.seek(self.DAMAGE_AT)
            return handle.read(1)[0]

    def test_the_scrub_heals_damage_the_tail_window_cannot_see(self) -> None:
        self.mirror_once()  # checkpoint 1; the mirror did not exist before it
        self.flip()
        good = self.byte_at(self.ledger)

        # Checkpoints 2..K-1: the incremental path reads only the last 64 KiB,
        # so it never learns about byte 100 and reports nothing.  This is the
        # review's measurement, reproduced.
        for index in range(2, agent.MIRROR_SCRUB_EVERY):
            self.append([1000 + index])
            self.assertEqual(self.mirror_once().events, ())
        self.assertNotEqual(self.byte_at(self.mirror / "groups.jsonl"), good)
        # And a restore taken here hands the bad byte straight back to the build.
        damaged = self.root / "damaged"
        restore_mirror(self.mirror, damaged)
        self.assertNotEqual(self.byte_at(damaged / "groups.jsonl"), good)

        # Checkpoint K: copied whole, mirror healed, journalled as a scrub.  The
        # healing is asserted first because it is the whole point — removing the
        # scrub must fail *here*, not merely on a missing journal row.
        self.append([2000])
        report = self.mirror_once()
        self.assertEqual(self.byte_at(self.mirror / "groups.jsonl"), good)
        self.assertEqual([event["reason"] for event in report.events], ["scrub", "scrub"])
        self.assertEqual({event["action"] for event in report.events}, {"full_copy"})
        self.assertTrue(all(event["journal"] for event in report.events))
        self.assertEqual(report.appended_bytes, 0)
        self.assert_mirror_matches()

        restored = self.root / "restored"
        restore_mirror(self.mirror, restored)
        self.assertEqual(
            (restored / "groups.jsonl").read_bytes(), self.ledger.read_bytes()
        )
        self.assertEqual(self.byte_at(restored / "groups.jsonl"), good)

    def test_the_counter_lives_in_the_state_file_and_survives_a_restart(self) -> None:
        for expected in (1, 2, 3):
            # A fresh call with fresh path objects is exactly what a restarted
            # process makes: nothing is carried in memory between them.
            mirror_artifacts(Path(str(self.output)), Path(str(self.mirror)))
            self.assertEqual(self.state()["checkpoints"], expected)

        # Hand it the state a long-running build would have left one checkpoint
        # short of a scrub: the cadence resumes there rather than from zero.
        self.set_checkpoints(agent.MIRROR_SCRUB_EVERY - 1)
        self.append([3000])
        report = mirror_artifacts(Path(str(self.output)), Path(str(self.mirror)))
        self.assertEqual([event["reason"] for event in report.events], ["scrub", "scrub"])
        self.assertEqual(self.state()["checkpoints"], agent.MIRROR_SCRUB_EVERY)
        self.assert_mirror_matches()

        # ...and the next checkpoint goes back to appending rather than latching.
        self.append([3001])
        self.assertEqual(self.mirror_once().events, ())

    def test_a_scrub_leaves_the_incremental_path_byte_exact(self) -> None:
        scrubbed: list[str] = []
        for index in range(agent.MIRROR_SCRUB_EVERY):
            self.append([4000 + index])
            report = self.mirror_once()
            self.assert_mirror_matches()  # after every single checkpoint
            scrubbed.extend(
                event["artifact"] for event in report.events if event["reason"] == "scrub"
            )
        # Exactly one cadence hit in K checkpoints, and it is the K-th.
        self.assertEqual(scrubbed, ["groups.jsonl", "sft.jsonl"])

        self.append([5000])
        seen, patch = self.copies()
        with patch:
            report = self.mirror_once()
        self.assertEqual(seen, ["manifest.json"])  # the delta again, not the file
        self.assertEqual(report.events, ())
        self.assert_mirror_matches()

    def test_an_anomaly_on_a_scrub_checkpoint_keeps_its_own_reason(self) -> None:
        """The scrub is checked last: a checkpoint that is both must journal the
        anomaly, or the one row that says the mirror was found broken is lost."""
        self.mirror_once()
        self.set_checkpoints(agent.MIRROR_SCRUB_EVERY - 1)
        target = self.mirror / "groups.jsonl"
        with target.open("r+b") as handle:
            handle.truncate(target.stat().st_size // 2)

        report = self.mirror_once()
        self.assert_mirror_matches()
        self.assertEqual(
            [event["reason"] for event in report.events], ["mirror_truncated", "scrub"]
        )


class MirrorRestoreTests(MirrorFixture):
    """What a reboot may find on NFS, and what resume must make of it."""

    def restore(self) -> Path:
        wiped = self.root / "restored"
        restore_mirror(self.mirror, wiped)
        return wiped

    def test_a_torn_half_record_is_dropped_on_restore(self) -> None:
        self.mirror_once()
        with (self.mirror / "groups.jsonl").open("ab") as handle:
            handle.write(b'{"build_id":"build","group_id":"group-000')

        wiped = self.restore()
        restored = (wiped / "groups.jsonl").read_bytes()
        self.assertEqual(restored, (self.output / "groups.jsonl").read_bytes())
        self.assertEqual(len(scan_jsonl(wiped / "groups.jsonl").records), 5)

    def test_a_complete_record_without_its_newline_is_dropped_on_restore(self) -> None:
        """The one cut ``scan_jsonl`` cannot see: the record parses, so resume
        would accept it and the journal's next append would glue the following
        record onto the same line — a mid-file corruption one resume later."""
        self.mirror_once()
        with (self.mirror / "groups.jsonl").open("ab") as handle:
            handle.write(_row(5).encode("utf-8"))  # no trailing newline

        wiped = self.restore()
        self.assertTrue((wiped / "groups.jsonl").read_bytes().endswith(b"\n"))
        self.assertEqual(len(scan_jsonl(wiped / "groups.jsonl").records), 5)

        # And the resume that follows appends cleanly rather than gluing.
        with ArtifactStore(wiped, "build", fsync_every=1) as store:
            store.append_group(_group(5))
        rows = scan_jsonl(wiped / "groups.jsonl").records
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[-1]["group_id"], "group-00005")

    def test_an_intact_mirror_restores_byte_for_byte(self) -> None:
        self.mirror_once()
        wiped = self.restore()
        for name in ("groups.jsonl", "sft.jsonl", "manifest.json"):
            self.assertEqual(
                (wiped / name).read_bytes(), (self.output / name).read_bytes()
            )

    def test_a_mirror_without_a_manifest_restores_nothing(self) -> None:
        self.mirror_once()
        (self.mirror / "manifest.json").unlink()
        wiped = self.root / "restored"
        self.assertEqual(restore_mirror(self.mirror, wiped), [])
        self.assertFalse(wiped.exists())


class LandCadenceGateTests(unittest.TestCase):
    """The gate itself: time AND groups, with a free-space override."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def gate(self, *, groups: int, elapsed: float, pressure: bool = False) -> bool:
        stub = SimpleNamespace(
            store=SimpleNamespace(
                groups={f"group-{index}": {} for index in range(groups)},
                root=self.root,
            ),
            dependencies=SimpleNamespace(
                now=lambda: BASE + timedelta(seconds=elapsed)
            ),
            _last_land_at=BASE,
            _last_land_groups=0,
            _tmpfs_under_pressure=lambda: pressure,
        )
        return CanonicalPipeline._land_cadence_ready(stub)

    def test_both_conditions_are_required(self) -> None:
        many, few = agent.LAND_MIN_GROUPS, agent.LAND_MIN_GROUPS - 1
        long, short = agent.LAND_MIN_INTERVAL_SECONDS, agent.LAND_MIN_INTERVAL_SECONDS - 1
        self.assertFalse(self.gate(groups=few, elapsed=long))          # groups short
        self.assertFalse(self.gate(groups=few, elapsed=long * 100))    # and still short
        self.assertFalse(self.gate(groups=many, elapsed=short))        # clock short
        self.assertFalse(self.gate(groups=many * 100, elapsed=short))  # and still short
        self.assertTrue(self.gate(groups=many, elapsed=long))          # both met

    def test_pressure_overrides_a_gate_that_would_refuse(self) -> None:
        self.assertTrue(self.gate(groups=0, elapsed=0, pressure=True))

    def test_the_counters_are_relative_to_the_last_checkpoint(self) -> None:
        stub = SimpleNamespace(
            store=SimpleNamespace(
                groups={f"group-{index}": {} for index in range(agent.LAND_MIN_GROUPS)},
                root=self.root,
            ),
            dependencies=SimpleNamespace(
                now=lambda: BASE + timedelta(seconds=agent.LAND_MIN_INTERVAL_SECONDS)
            ),
            _last_land_at=BASE,
            # A resume inherits the durable groups; they are not new work.
            _last_land_groups=1,
            _tmpfs_under_pressure=lambda: False,
        )
        self.assertFalse(CanonicalPipeline._land_cadence_ready(stub))

    def test_pressure_reads_the_mount_holding_the_output_root(self) -> None:
        stub = SimpleNamespace(store=SimpleNamespace(root=self.root))
        usage = SimpleNamespace(free=agent.LAND_FREE_BYTES_FLOOR - 1)
        with mock.patch.object(agent.shutil, "disk_usage", return_value=usage) as spy:
            self.assertTrue(CanonicalPipeline._tmpfs_under_pressure(stub))
        spy.assert_called_once_with(self.root)
        usage = SimpleNamespace(free=agent.LAND_FREE_BYTES_FLOOR)
        with mock.patch.object(agent.shutil, "disk_usage", return_value=usage):
            self.assertFalse(CanonicalPipeline._tmpfs_under_pressure(stub))
        with mock.patch.object(agent.shutil, "disk_usage", side_effect=OSError("gone")):
            self.assertFalse(CanonicalPipeline._tmpfs_under_pressure(stub))


class LandCadenceWiringTests(LandFixture):
    """The gate is consulted for unforced checkpoints and only for those."""

    def mirrors(self):
        calls: list[Path] = []
        real = agent.mirror_artifacts

        def spy(output_root, mirror_dir):
            calls.append(Path(output_root))
            return real(output_root, mirror_dir)

        return calls, mock.patch.object(agent, "mirror_artifacts", spy)

    def run_with_gate(self, ready: bool) -> list[Path]:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        calls, patch = self.mirrors()
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(agent, "LAND_WATERMARK_BYTES", 0), \
                mock.patch.object(
                    CanonicalPipeline, "_land_cadence_ready", lambda _self: ready
                ), patch:
            manifest = run(config, dependencies=dependencies)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"]["groups"], 2)
        return calls

    def test_an_open_gate_over_a_passed_water_mark_checkpoints_per_group(self) -> None:
        # This is the pre-hotfix behaviour, reproduced deliberately: it is what
        # the cadence exists to stop, and the next test is the same run with the
        # gate closed.
        self.assertGreater(len(self.run_with_gate(True)), 1)

    def test_a_closed_gate_leaves_only_the_forced_checkpoints(self) -> None:
        # Rendering's ``force=True`` and the final publish; no per-group mirror.
        self.assertEqual(len(self.run_with_gate(False)), 2)

    def test_the_water_mark_still_gates_before_the_clock_is_consulted(self) -> None:
        """Under the mark nothing lands, and the gate is not even asked — the
        cadence loosens the water mark's grip, it does not replace it."""
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        asked: list[bool] = []

        def spy(_self):
            asked.append(True)
            return True

        calls, patch = self.mirrors()
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(CanonicalPipeline, "_land_cadence_ready", spy), patch:
            run(config, dependencies=dependencies)
        self.assertEqual(asked, [])
        self.assertEqual(len(calls), 2)  # both forced

    def test_a_fallback_is_journalled_as_a_non_terminal_mirror_event(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        mirror = self.mirror / config.build_id
        real = agent.mirror_artifacts

        def corrupt_then_mirror(output_root, mirror_dir):
            target = Path(mirror_dir) / "groups.jsonl"
            if target.is_file() and target.stat().st_size:
                data = bytearray(target.read_bytes())
                data[0] = data[0] ^ 0x20
                target.write_bytes(bytes(data))
            return real(output_root, mirror_dir)

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(agent, "LAND_WATERMARK_BYTES", 0), \
                mock.patch.object(
                    CanonicalPipeline, "_land_cadence_ready", lambda _self: True
                ), \
                mock.patch.object(agent, "mirror_artifacts", corrupt_then_mirror):
            manifest = run(config, dependencies=dependencies)

        self.assertEqual(manifest["status"], "complete")
        events = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["stage"] == "mirror"
        ]
        self.assertTrue(events)
        self.assertFalse(any(row["terminal"] for row in events))
        self.assertFalse(any(row["retryable"] for row in events))
        self.assertEqual({row["error_code"] for row in events}, {"mirror_full_copy"})
        # Substring, not ``json.loads``: journalled messages run through
        # ``redact_text`` and this fixture's DSN secrets are the single letters
        # "u" and "p", which mangle any JSON they appear in.  ``_summary_count``
        # tolerates the same thing for land summaries.
        self.assertIn("diverged", events[0]["message"])
        self.assertEqual(len({row["event_id"] for row in events}), len(events))
        # And the mirror it repaired is the authoritative file again.
        self.assertEqual(
            (mirror / "groups.jsonl").read_bytes(),
            (config.output_root / "groups.jsonl").read_bytes(),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
