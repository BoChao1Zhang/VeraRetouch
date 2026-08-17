"""HOTFIX-Watermark-4: the land water mark counts only what landing reclaims.

``_asset_directories`` tracks the prefetch buffer because the buffer shares the
tmpfs, and until this hotfix its bytes went into the same counter the water mark
reads.  Landing cannot release one byte of the buffer, so on prod-l8 — 8.5 GiB of
resident buffer against 0 bytes of staged assets — the comparison
``staged >= LAND_WATERMARK_BYTES`` was permanently true, and *both* of its
readers misfired:

* ``_land_checkpoint`` entered on every call (throttled since MirrorCadence-3,
  but still gated on a level it had no lever over);
* ``_fill_initial_mode`` drained its whole in-flight queue on every source, which
  pinned a configured window of 5 at 1 — the real cause of the throughput
  collapse that MirrorCadence-3 explicitly did not fix (§M-七 告警一).

So the counter is split: ``_staged_bytes`` is the landable assets, ``_buffer_bytes``
is the rebuildable buffer, and the buffer gets its own ceiling
(``PREFETCH_BUFFER_BYTES``) because nothing else bounds it — a source parked on
the SAM3 queue keeps its buffered copy until the rendering phase ends.

Nothing here touches the real NFS mount or ``/mnt/ramstage``: every path below is
under ``tempfile``.
"""
from __future__ import annotations

import os
import tempfile
import threading
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from construct import agent
from construct.agent import CanonicalPipeline, MirrorReport

from tests.test_source_window_and_cgt import WindowFixture


BASE = datetime(2026, 8, 13, 12, 0, 0, tzinfo=timezone.utc)


class StagedFixture(unittest.TestCase):
    """An asset tree and a buffer beside it, plus a pipeline stub over both."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.assets = self.root / "assets"
        (self.assets / "candidates").mkdir(parents=True)
        (self.assets / "masks").mkdir(parents=True)
        self.buffer = self.root / "prefetch"
        self.buffer.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def pipeline(self, *, prefetch: bool = True) -> SimpleNamespace:
        """Just enough pipeline for the accounting methods, which touch nothing else."""
        stub = SimpleNamespace(
            store=SimpleNamespace(assets_root=self.assets, root=self.root, groups={}),
            _staged_lock=threading.Lock(),
            _staged={},
            _staged_bytes=0,
            _buffer_bytes=0,
            _buffer_evicted=0,
            _prefetch=(
                SimpleNamespace(
                    directory=self.buffer,
                    path_for=lambda source: self.buffer / str(source),
                )
                if prefetch else None
            ),
        )
        stub._buffer_prefix = str(self.buffer) + os.sep if prefetch else None
        stub._group_assets = CanonicalPipeline._group_assets
        for name in (
            "_asset_directories", "_is_buffer", "_recalibrate_staged", "_account_asset",
            "_staged_size", "_buffer_size", "_clean_orphan_assets",
            "_trim_prefetch_buffer", "_discard_prefetched",
        ):
            setattr(stub, name, getattr(CanonicalPipeline, name).__get__(stub))
        return stub

    def write(self, path: Path, size: int) -> Path:
        path.write_bytes(b"x" * size)
        return path


class StagedAccountingTests(StagedFixture):
    """Which counter each tracked file lands in, and that neither one drifts."""

    def test_a_full_rescan_splits_the_two_budgets(self) -> None:
        self.write(self.assets / "candidates" / "a.jpg", 100)
        self.write(self.assets / "masks" / "b.png", 20)
        self.write(self.buffer / "cold", 4_000)
        self.write(self.buffer / "warm", 5_000)
        stub = self.pipeline()
        stub._recalibrate_staged()

        self.assertEqual(stub._staged_size(), 120)
        self.assertEqual(stub._buffer_size(), 9_000)
        # Both are still *tracked*, or the orphan sweep would stop seeing them.
        self.assertEqual(len(stub._staged), 4)

    def test_the_buffer_is_tracked_but_never_reaches_the_water_mark(self) -> None:
        """The prod-l8 shape exactly: a resident buffer over an empty asset tree."""
        self.write(self.buffer / "resident", 9_000)
        stub = self.pipeline()
        stub._recalibrate_staged()
        self.assertEqual(stub._staged_size(), 0)
        self.assertEqual(stub._buffer_size(), 9_000)

    def test_accounting_routes_by_directory_and_replaces_rather_than_adds(self) -> None:
        asset = self.write(self.assets / "candidates" / "a.jpg", 100)
        buffered = self.write(self.buffer / "cold", 400)
        stub = self.pipeline()
        stub._recalibrate_staged()
        stub._account_asset(asset)
        stub._account_asset(buffered)
        self.assertEqual((stub._staged_size(), stub._buffer_size()), (100, 400))

        # A refilled slot rewrites the same path; neither counter may drift.
        stub._account_asset(self.write(asset, 250))
        stub._account_asset(self.write(buffered, 900))
        self.assertEqual((stub._staged_size(), stub._buffer_size()), (250, 900))

    def test_discarding_a_buffered_copy_debits_the_buffer_not_the_mark(self) -> None:
        asset = self.write(self.assets / "candidates" / "a.jpg", 100)
        self.write(self.buffer / "s-1", 400)
        stub = self.pipeline()
        stub._recalibrate_staged()
        stub._discard_prefetched(SimpleNamespace(source_path="s-1"))

        self.assertEqual(stub._staged_size(), 100)
        self.assertEqual(stub._buffer_size(), 0)
        self.assertFalse((self.buffer / "s-1").exists())
        self.assertEqual(list(stub._staged), [str(asset)])

    def test_the_orphan_sweep_still_spares_the_buffer_and_only_debits_assets(self) -> None:
        orphan = self.write(self.assets / "candidates" / "orphan.jpg", 100)
        kept = self.write(self.assets / "masks" / "kept.png", 30)
        self.write(self.buffer / "cold", 7_000)
        stub = self.pipeline()
        stub.store.groups = {"g": {"candidates": [{"cgt_path": str(kept)}]}}
        stub._recalibrate_staged()

        self.assertEqual(stub._clean_orphan_assets(), 1)
        self.assertFalse(orphan.exists())
        self.assertTrue((self.buffer / "cold").exists())
        self.assertEqual(stub._staged_size(), 30)
        self.assertEqual(stub._buffer_size(), 7_000)

    def test_without_a_buffer_nothing_is_ever_classified_as_one(self) -> None:
        self.write(self.assets / "candidates" / "a.jpg", 100)
        stub = self.pipeline(prefetch=False)
        stub._recalibrate_staged()
        self.assertEqual((stub._staged_size(), stub._buffer_size()), (100, 0))
        self.assertFalse(stub._is_buffer(str(self.buffer / "cold")))
        self.assertEqual(stub._trim_prefetch_buffer(), 0)


class PrefetchCeilingTests(StagedFixture):
    """``PREFETCH_BUFFER_BYTES``: the bound the buffer never had."""

    def loaded(self, sizes: dict[str, int]) -> SimpleNamespace:
        """A buffer holding ``{name: size}``, mtimes ascending in insertion order."""
        stub = self.pipeline()
        stamp = 1_000_000.0
        for name, size in sizes.items():
            path = self.write(self.buffer / name, size)
            os.utime(path, (stamp, stamp))
            stamp += 60.0
        stub._recalibrate_staged()
        return stub

    def test_a_buffer_inside_its_budget_is_left_alone(self) -> None:
        stub = self.loaded({"a": 400, "b": 400})
        with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", 1_000):
            self.assertEqual(stub._trim_prefetch_buffer(), 0)
        self.assertEqual(stub._buffer_size(), 800)
        self.assertEqual(sorted(p.name for p in self.buffer.iterdir()), ["a", "b"])

    def test_eviction_takes_the_coldest_first_and_stops_at_the_budget(self) -> None:
        # "a" is the oldest stamp, "d" the newest; 1,600 bytes against a 700 byte
        # budget owes 900, which "a" and "b" pay.
        stub = self.loaded({"a": 400, "b": 500, "c": 300, "d": 400})
        with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", 700):
            self.assertEqual(stub._trim_prefetch_buffer(), 2)

        self.assertEqual(sorted(p.name for p in self.buffer.iterdir()), ["c", "d"])
        self.assertEqual(stub._buffer_size(), 700)
        self.assertEqual(stub._buffer_evicted, 2)
        self.assertEqual(
            sorted(Path(p).name for p in stub._staged), ["c", "d"]
        )
        # And the mark never moved: none of this was landable.
        self.assertEqual(stub._staged_size(), 0)

    def test_the_chunk_about_to_render_is_never_evicted(self) -> None:
        """Protection beats age: "a" is the coldest entry *and* the next read."""
        stub = self.loaded({"a": 400, "b": 500, "c": 300})
        with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", 400):
            self.assertEqual(stub._trim_prefetch_buffer(["a"]), 2)
        self.assertEqual([p.name for p in self.buffer.iterdir()], ["a"])
        self.assertEqual(stub._buffer_size(), 400)

    def test_the_budget_is_honoured_even_when_only_protected_copies_remain(self) -> None:
        """Over budget with nothing evictable is a fact to survive, not to fix."""
        stub = self.loaded({"a": 400, "b": 500})
        with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", 100):
            self.assertEqual(stub._trim_prefetch_buffer(["a", "b"]), 0)
        self.assertEqual(stub._buffer_size(), 900)

    def test_a_copy_that_vanished_underneath_still_gives_its_bytes_back(self) -> None:
        stub = self.loaded({"a": 400, "b": 500})
        (self.buffer / "a").unlink()
        with mock.patch.object(agent, "PREFETCH_BUFFER_BYTES", 500):
            self.assertEqual(stub._trim_prefetch_buffer(), 1)
        self.assertEqual(stub._buffer_size(), 500)
        self.assertNotIn(str(self.buffer / "a"), stub._staged)


class RotationWiringTests(unittest.TestCase):
    """Where the ceiling is enforced, and what is in flight when it is."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def rotate(self) -> list[str]:
        """Run one rotation over a stub prefetch, returning the call order."""
        order: list[str] = []
        trimmed: list[tuple[str, ...]] = []
        prefetch = SimpleNamespace(
            directory=self.root / "prefetch",
            submit=lambda paths: order.append("submit"),
            take=lambda: order.append("take") or [],
        )
        stub = SimpleNamespace(
            _prefetch=prefetch,
            _chunk_paths=lambda *args, **kwargs: ["src-a", "src-b"],
            _account_prefetched=lambda: order.append("take"),
            _trim_prefetch_buffer=lambda keep=(): (
                order.append("trim"), trimmed.append(tuple(keep))
            ),
        )
        CanonicalPipeline._rotate_prefetch(stub, (), 1, set(), remaining=8)
        self.trimmed = trimmed
        return order

    def test_the_trim_runs_between_delivery_and_the_next_submit(self) -> None:
        """No writer is in the directory at that instant, which is the whole reason.

        ``_SourcePrefetch.take`` clears the outstanding future, so between it and
        the next ``submit`` the prefetch thread is idle — the one moment a
        rotation may unlink from the buffer without racing the reader filling it.
        """
        self.assertEqual(self.rotate(), ["take", "trim", "submit"])

    def test_the_trim_protects_the_chunk_this_rotation_just_took(self) -> None:
        self.rotate()
        self.assertEqual(self.trimmed, [("src-a", "src-b")])


class LandGateTruthTableTests(unittest.TestCase):
    """water mark × free-space valve × cadence × force, end to end through the gate."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def pipeline(
        self, *, staged: int, pressure: bool, groups: int, elapsed: float,
        calls: list[str],
    ) -> SimpleNamespace:
        """A gate with the *real* cadence method behind it.

        Stubbing ``_land_cadence_ready`` would let the table assert rows that
        cannot happen — the cadence method carries the same free-space valve, so
        "under pressure but cadence refuses" is not a state of this system.  Only
        the mount reading is faked, because the tmpfs it asks about is not ours.
        """
        stub = SimpleNamespace(
            config=SimpleNamespace(build_id="build"),
            dependencies=SimpleNamespace(
                archive_root=None, mirror_root=self.root / "mirror",
                now=lambda: BASE + timedelta(seconds=elapsed),
            ),
            store=SimpleNamespace(
                root=self.root,
                groups={f"group-{index}": {} for index in range(groups)},
                checkpoint=lambda: calls.append("checkpoint"),
                write_manifest=lambda manifest: None,
            ),
            _phase="rendering",
            _last_land_at=BASE, _last_land_groups=0,
            _staged_lock=threading.Lock(), _staged_bytes=staged, _staged={},
            _manifest=lambda phase: {},
            _record_mirror_events=lambda report: None,
            _tmpfs_under_pressure=lambda: pressure,
        )
        for name in ("_staged_size", "_staging_full", "_land_cadence_ready"):
            setattr(stub, name, getattr(CanonicalPipeline, name).__get__(stub))
        return stub

    def lands(
        self, *, staged: int, pressure: bool = False, cadence: bool = True,
        force: bool = False,
    ) -> bool:
        """Did the checkpoint get past the gate?  ``store.checkpoint`` is the tell."""
        calls: list[str] = []
        stub = self.pipeline(
            staged=staged, pressure=pressure, calls=calls,
            groups=agent.LAND_MIN_GROUPS if cadence else agent.LAND_MIN_GROUPS - 1,
            elapsed=agent.LAND_MIN_INTERVAL_SECONDS if cadence else 0.0,
        )
        with mock.patch.object(
            agent, "mirror_artifacts",
            lambda *args, **kwargs: MirrorReport((), 0, 0, ()),
        ):
            CanonicalPipeline._land_checkpoint(stub, force=force)
        return "checkpoint" in calls

    def test_the_truth_table(self) -> None:
        mark = agent.LAND_WATERMARK_BYTES
        # staged, pressure, cadence met, force -> lands
        table = [
            # Under the mark nothing lands, however open the cadence gate is.
            # This row is the whole hotfix: with the prefetch buffer folded into
            # the level, prod-l8 could not reach it at all.
            (0, False, True, False, False),
            (mark - 1, False, True, False, False),
            (0, False, False, False, False),
            # At the mark the cadence gate decides — MirrorCadence-3 unchanged.
            (mark, False, True, False, True),
            (mark, False, False, False, False),
            # The valve reaches over both, and now over the mark as well: the
            # level is one the buffer and the ledgers cannot raise, so it can sit
            # at zero while the mount fills with things landing does not own.
            (0, True, False, False, True),
            (mark, True, False, False, True),
            # force ignores every one of them.
            (0, False, False, True, True),
            (mark - 1, False, False, True, True),
        ]
        for staged, pressure, cadence, force, expected in table:
            with self.subTest(
                staged=staged, pressure=pressure, cadence=cadence, force=force
            ):
                self.assertEqual(
                    self.lands(
                        staged=staged, pressure=pressure, cadence=cadence, force=force
                    ),
                    expected,
                )

    def test_a_healthy_build_never_stats_the_mount(self) -> None:
        """The valve is second in ``_staging_full`` on purpose: it is a syscall,
        and the fill loop asks this question once per source per in-flight check."""
        stub = SimpleNamespace(
            store=SimpleNamespace(root=self.root),
            _staged_size=lambda: agent.LAND_WATERMARK_BYTES,
        )
        stub._tmpfs_under_pressure = CanonicalPipeline._tmpfs_under_pressure.__get__(stub)
        with mock.patch.object(agent.shutil, "disk_usage") as spy:
            self.assertTrue(CanonicalPipeline._staging_full(stub))
        spy.assert_not_called()

    def test_the_cadence_gate_is_still_not_asked_below_the_mark(self) -> None:
        """MirrorCadence-3's ordering survives: the level gates before the clock."""
        calls: list[str] = []
        asked: list[bool] = []
        stub = self.pipeline(
            staged=0, pressure=False, groups=agent.LAND_MIN_GROUPS,
            elapsed=agent.LAND_MIN_INTERVAL_SECONDS, calls=calls,
        )
        stub._land_cadence_ready = lambda: asked.append(True) or True
        CanonicalPipeline._land_checkpoint(stub)
        self.assertEqual(asked, [])
        self.assertEqual(calls, [])


class SourceWindowUnderAResidentBufferTests(WindowFixture):
    """The regression this hotfix exists for: a resident buffer must not close
    the source window.

    Same run twice, differing in one thing only — which counter the bytes are
    booked against.  Booked as buffer the window has to stay open; booked as
    assets the mark has to close it, because that is the mark doing its job.
    """

    # Booked against the real, unpatched mark: the fixture's own assets are
    # kilobytes, so nothing but the seed can ever reach it and the two runs
    # differ in exactly one variable.
    RESIDENT = agent.LAND_WATERMARK_BYTES + 1

    def peak_in_flight(self, *, tag: str, buffered: int, staged: int, hold: bool) -> int:
        config, dependencies = self.build(window=5, tag=tag)
        lock = threading.Lock()
        filled = threading.Event()
        active: set[str] = set()
        leaders: set[str] = set()
        peak = 0
        seeded: list[str] = []
        render = agent.CanonicalPipeline._render_source_buffered
        fill = agent.CanonicalPipeline._fill_initial_mode

        def seed(pipeline, mode, sources, target, **kwargs):
            """Park ``buffered`` bytes in the buffer and ``staged`` in the assets.

            At the top of the pass, before a single future is submitted — seeding
            from inside the render wrapper would measure a window that had already
            filled.  The fixture has no archive root, so there is no real prefetch
            buffer to fill: what is on trial is the classification, not the fetch,
            so the prefix is installed and one oversized entry is booked through
            the very counters the water mark reads.
            """
            if not seeded:
                seeded.append(mode)
                directory = pipeline.store.root / "prefetch"
                directory.mkdir(parents=True, exist_ok=True)
                pipeline._buffer_prefix = str(directory) + os.sep
                resident = directory / "resident"
                resident.touch()
                with pipeline._staged_lock:
                    pipeline._staged[str(resident)] = buffered
                    pipeline._buffer_bytes += buffered
                    pipeline._staged_bytes += staged
            return fill(pipeline, mode, sources, target, **kwargs)

        def wrapper(pipeline, source, mode, *, selector_turn=None, **kwargs):
            nonlocal peak
            with lock:
                active.add(source.source_id)
                peak = max(peak, len(active))
                if len(active) >= 5:
                    filled.set()
                leader = hold and mode not in leaders
                leaders.add(mode)
            try:
                if leader:
                    filled.wait(timeout=30)
                return render(
                    pipeline, source, mode, selector_turn=selector_turn, **kwargs
                )
            finally:
                with lock:
                    active.discard(source.source_id)

        with mock.patch.object(
            agent.CanonicalPipeline, "_render_source_buffered", wrapper
        ), mock.patch.object(
            agent.CanonicalPipeline, "_fill_initial_mode", seed
        ), mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = agent.run(config, dependencies=dependencies)
        self.assertEqual(manifest["completed"]["groups"], self.TARGET)
        self.assertTrue(seeded, "the seed never ran; the measurement proves nothing")
        return peak

    def test_a_resident_buffer_over_the_mark_leaves_the_window_open(self) -> None:
        # The prod-l8 shape exactly: buffer over the mark, 0 bytes of assets.
        # Before this hotfix the fill loop drained to empty on every source and
        # this peaked at 1.
        peak = self.peak_in_flight(
            tag="buffer-resident", buffered=self.RESIDENT, staged=0, hold=True
        )
        self.assertGreaterEqual(peak, 5)

    def test_real_staged_assets_over_the_mark_still_close_it(self) -> None:
        # The mirror image, and the reason this is not simply "ignore the mark":
        # landable bytes over the mark must still drain the queue to empty, which
        # is the only path that reaches the in-loop checkpoint.
        peak = self.peak_in_flight(
            tag="assets-resident", buffered=0, staged=self.RESIDENT, hold=False
        )
        self.assertEqual(peak, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
