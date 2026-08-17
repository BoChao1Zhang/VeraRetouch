"""Annotation interleaved with rendering: pass boundaries, batches, safety.

The build used to run its two expensive stages back to back — every group
rendered, then every winner annotated — which left the relay idle for the whole
render phase and both GPUs idle for the whole annotation phase.  Rendering now
hands each finished pass over the source pool to a background annotator and
carries straight on into the next pass.

These are the properties that makes that safe rather than merely faster:

* a batch is a whole pass and is handed over exactly once, so no task is ever
  paid for twice (``sft_id`` is derived from the task, and a second answer for
  one task is a durable conflict, i.e. a crash after the money is spent);
* a task is only handed over once its ``I_tar`` is somewhere a land checkpoint
  cannot delete it from;
* the two threads share one store, and the collections they walk are snapshotted
  under its lock rather than iterated live;
* whatever the driver could not do is still owed, and the closing drain owns it.
"""
from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from construct import agent
from construct.agent import PipelineError, _AnnotationDriver, run
from construct.responses import ResponsesAnnotator
from construct.sources import SourceInventoryResult
from construct.state import ArtifactStore, scan_jsonl, stable_id
from dataset_build.tools.archive_reader import path_exists, read_bytes

from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink

from dataset_build.tests.test_canonical_orchestration import (
    FakeAnnotator,
    OrchestrationFixture,
)
from dataset_build.tests.test_land_integration import LandFixture


def _group(index: int, *, winners: int = 1) -> dict:
    """A durable group record with real shape, used for store-level races."""
    candidates = [
        {
            "candidate_id": f"group-{index:05d}-candidate-{slot}",
            "after_path": f"/nowhere/group-{index:05d}-{slot}.jpg",
        }
        for slot in range(8)
    ]
    return {
        "build_id": "build",
        "group_id": f"group-{index:05d}",
        "source_id": f"source-{index:05d}",
        "source_path": f"/nowhere/source-{index:05d}.jpg",
        "render_mode": "global",
        "candidates": candidates,
        "winner_ids": [row["candidate_id"] for row in candidates[:winners]],
    }


class BatchRecorder:
    """A drainer double that records the batch boundary it was handed."""

    def __init__(self, store):
        self.store = store
        self.batches: list[frozenset[str]] = []
        self.groups_at_drain: list[int] = []


class RecordingAnnotator(FakeAnnotator):
    """``FakeAnnotator`` that reports what each drain was given and when."""

    log: BatchRecorder

    def drain(self, *, max_workers=None, only=None):
        tasks = self.tasks(only)
        self.log.batches.append(
            frozenset(str(task["task_id"]) for task in tasks)
        )
        self.log.groups_at_drain.append(len(self.store.groups))
        return super().drain(max_workers=max_workers, only=only)


class InterleaveFixture(OrchestrationFixture):
    """A build that can be given more than one pass over its source pool.

    A boundary only exists where a pass ends, so every claim here needs
    ``max_source_uses`` above one — and a bank bigger than the eight presets one
    group consumes, or the second pass would have nothing left to draw.  Two
    majors of sixteen, the same shape ``test_source_reuse`` uses.
    """

    def catalog(self):
        links = []
        for major in ("major-a", "major-b"):
            for minor_index in range(4):
                for preset_index in range(4):
                    preset_id = f"{major}-{minor_index}-{preset_index}"
                    links.append(TaxonomyLink(
                        PresetRecord(
                            preset_id=preset_id,
                            path=self.root / f"{preset_id}.cube",
                            format="lut",
                            kind="lut",
                            style_name="Style Name",
                            fidelity_de=None,
                            render_engine="gpu_lut",
                        ),
                        major,
                        f"minor-{minor_index}",
                    ))
        return PresetCatalog.from_links(links)

    def reuse_config(self, *, target, uses):
        config = self.config(target=target, local=0.0, global_=1.0)
        return dataclasses.replace(
            config,
            sources=dataclasses.replace(config.sources, max_source_uses=uses),
        )

    def run_build(self, *, target, sources, uses, annotator=None, dependencies=None):
        config = self.reuse_config(target=target, uses=uses)
        records = [self.source(index) for index in range(sources)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": sources, "eligible": sources}, "ok"
        )
        dependencies = dependencies or self.dependencies(inventory)
        if annotator is not None:
            dependencies.annotator_factory = lambda _config, store: annotator(store)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=dependencies)
        return config, manifest


class PassBoundaryTests(InterleaveFixture):
    def recorder(self):
        log = BatchRecorder(None)

        class Annotator(RecordingAnnotator):
            pass

        Annotator.log = log
        return Annotator, log

    def test_every_pass_is_handed_over_at_its_own_boundary(self) -> None:
        annotator, log = self.recorder()
        config, manifest = self.run_build(
            target=4, sources=2, uses=2, annotator=annotator
        )
        self.assertEqual(manifest["status"], "complete")
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(groups), 4)
        # Two passes, two batches, and a closing drain that found nothing left.
        non_empty = [batch for batch in log.batches if batch]
        self.assertEqual(len(non_empty), 2)
        self.assertEqual(log.batches[-1], frozenset())
        # The first batch was drained while only the first pass existed, which is
        # the whole claim: annotation started before rendering finished.
        self.assertEqual(log.groups_at_drain[0], 2)
        self.assertEqual(manifest["annotation"]["batches"], 2)
        self.assertEqual(manifest["annotation"]["inflight"], 0)
        self.assertEqual(manifest["annotation"]["pending"], 0)

    def test_the_final_pass_is_pipelined_like_every_other(self) -> None:
        """A budget-limited last pass returns early; the flush is before that."""
        annotator, log = self.recorder()
        _, manifest = self.run_build(target=4, sources=2, uses=2, annotator=annotator)
        # The second pass is the budget's last, so ``_fill_mode`` returns without
        # measuring anything after it.  Its winners still went out as a batch
        # rather than falling through to the closing drain.
        self.assertEqual(len(log.batches[1]), 2 * 2)
        self.assertEqual(manifest["annotation"]["batches"], 2)

    def test_a_single_pass_build_still_pipelines(self) -> None:
        annotator, log = self.recorder()
        _, manifest = self.run_build(target=2, sources=2, uses=1, annotator=annotator)
        self.assertEqual(manifest["annotation"]["batches"], 1)
        self.assertEqual(log.groups_at_drain[0], 2)

    def test_no_task_is_ever_handed_to_two_drains(self) -> None:
        annotator, log = self.recorder()
        config, manifest = self.run_build(
            target=6, sources=3, uses=2, annotator=annotator
        )
        self.assertEqual(manifest["status"], "complete")
        handed = [task_id for batch in log.batches for task_id in batch]
        self.assertEqual(len(handed), len(set(handed)))
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        task_ids = [row["annotation_task_id"] for row in rows]
        self.assertEqual(len(task_ids), len(set(task_ids)))
        self.assertEqual(set(task_ids), set(handed))

    def test_a_pass_that_adds_no_winner_costs_no_batch(self) -> None:
        """The boundary measures first: an empty delta is not handed over."""
        annotator, log = self.recorder()
        # Target 2 from 2 sources at budget 2: the first pass already meets it,
        # so the second pass renders nothing and owes nothing.
        _, manifest = self.run_build(target=2, sources=2, uses=2, annotator=annotator)
        self.assertEqual(manifest["annotation"]["batches"], 1)
        self.assertEqual([len(batch) for batch in log.batches], [4, 0])

    def test_the_phase_stays_rendering_while_a_batch_is_in_flight(self) -> None:
        """``phase`` names what owns the GPUs, not what owns the relay."""
        released = threading.Event()
        seen: list[dict] = []

        class BlockingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                if only:
                    released.wait(10.0)
                return super().drain(max_workers=max_workers, only=only)

        original = agent.ArtifactStore.write_manifest

        def spy(store, manifest):
            seen.append(json.loads(json.dumps(manifest)))
            if manifest["phase"] == "rendering" and manifest["annotation"]["inflight"]:
                released.set()
            return original(store, manifest)

        dependencies = self.dependencies(
            SourceInventoryResult(
                tuple(self.source(index) for index in range(2)),
                {"cache_entries": 2, "eligible": 2}, "ok",
            )
        )
        # A mirror root with no archive root makes every land checkpoint a pure
        # manifest write, which is how a manifest gets written *during* a pass.
        dependencies.mirror_root = self.root / "builds"
        with mock.patch.object(agent, "LAND_WATERMARK_BYTES", 0), \
                mock.patch.object(agent, "LAND_MIN_GROUPS", 0), \
                mock.patch.object(agent, "LAND_MIN_INTERVAL_SECONDS", 0.0), \
                mock.patch.object(agent.ArtifactStore, "write_manifest", spy):
            _, manifest = self.run_build(
                target=4, sources=2, uses=2, annotator=BlockingAnnotator,
                dependencies=dependencies,
            )
        released.set()
        self.assertEqual(manifest["status"], "complete")
        inflight = [
            row for row in seen
            if row["phase"] == "rendering" and row["annotation"]["inflight"]
        ]
        self.assertTrue(inflight, "no manifest observed the pipeline running")
        self.assertEqual(inflight[0]["annotation"]["inflight"], 4)
        # And the closing manifest agrees that nothing is owed any more.
        self.assertEqual(manifest["annotation"]["inflight"], 0)
        self.assertEqual(manifest["annotation"]["pending"], 0)


class ClosingDrainTests(InterleaveFixture):
    def test_the_closing_drain_owns_whatever_the_driver_did_not_finish(self) -> None:
        taken: list[frozenset[str]] = []

        class LazyAnnotator(FakeAnnotator):
            """Refuses batches, so everything falls through to the closing drain."""

            def drain(self, *, max_workers=None, only=None):
                taken.append(frozenset(only or ()))
                if only:
                    return {
                        "completed": 0, "terminal": 0, "transport_failed": 0,
                        "pending": len(self.store.pending_annotation_tasks()),
                    }
                return super().drain(max_workers=max_workers, only=only)

        config, manifest = self.run_build(
            target=4, sources=2, uses=2, annotator=LazyAnnotator
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["annotation"]["pending"], 0)
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["annotation_task_id"] for row in rows}), 8)
        # Two refused batches and one unrestricted closing drain.
        self.assertEqual([bool(batch) for batch in taken], [True, True, False])

    def test_an_unresolved_queue_still_fails_the_build(self) -> None:
        class SilentAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                return {
                    "completed": 0, "terminal": 0, "transport_failed": 0,
                    "pending": len(self.store.pending_annotation_tasks()),
                }

        with self.assertRaisesRegex(PipelineError, "annotation queue remains"):
            self.run_build(target=4, sources=2, uses=2, annotator=SilentAnnotator)

    def test_a_terminal_annotation_failure_still_reads_as_complete_with_failures(
        self,
    ) -> None:
        class FailingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                for index, task in enumerate(self.tasks(only)):
                    if index == 0:
                        self.store.append_failure({
                            "build_id": self.store.build_id,
                            "event_id": stable_id("failure", task["task_id"]),
                            "event_type": "terminal",
                            "stage": "annotation",
                            "task_id": task["task_id"],
                            "group_id": task["group_id"],
                            "candidate_id": task["candidate_id"],
                            "retryable": False,
                            "error_code": "annotation_round_exhausted",
                            "message": "every round was exhausted",
                            "terminal": True,
                        })
                        continue
                    self.append_task(task)
                self.store.checkpoint()
                return {
                    "completed": 0, "terminal": 1, "transport_failed": 0,
                    "pending": len(self.store.pending_annotation_tasks()),
                }

        _, manifest = self.run_build(
            target=2, sources=2, uses=1, annotator=FailingAnnotator
        )
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(manifest["annotation"]["pending"], 0)
        self.assertEqual(manifest["annotation"]["terminal_failures"], 1)


class ResumeTests(InterleaveFixture):
    def test_a_resumed_build_hands_the_whole_backlog_to_the_first_boundary(
        self,
    ) -> None:
        class SilentAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                return {
                    "completed": 0, "terminal": 0, "transport_failed": 0,
                    "pending": len(self.store.pending_annotation_tasks()),
                }

        config = self.reuse_config(target=4, uses=2)
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        dependencies = self.dependencies(inventory)
        dependencies.annotator_factory = lambda _config, store: SilentAnnotator(store)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(PipelineError, "annotation queue remains"):
                run(config, dependencies=dependencies)
        self.assertEqual(list(scan_jsonl(config.output_root / "sft.jsonl").records), [])
        backlog = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(backlog), 4)

        log = BatchRecorder(None)

        class Annotator(RecordingAnnotator):
            pass

        Annotator.log = log
        dependencies.annotator_factory = lambda _config, store: Annotator(store)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        self.assertEqual(resumed["status"], "complete")
        # Nothing was rendered this run, so the very first boundary — the one the
        # spent global pass reaches without doing any work — is what re-derives
        # and hands over the eight tasks the crashed run left owing.
        self.assertEqual(len(log.batches[0]), 8)
        self.assertEqual(log.groups_at_drain[0], 4)
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len({row["annotation_task_id"] for row in rows}), 8)

    def test_a_second_resume_re_annotates_nothing(self) -> None:
        log = BatchRecorder(None)

        class Annotator(RecordingAnnotator):
            pass

        Annotator.log = log
        config, manifest = self.run_build(
            target=4, sources=2, uses=2, annotator=Annotator
        )
        before = (config.output_root / "sft.jsonl").read_bytes()
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        dependencies = self.dependencies(inventory)
        dependencies.annotator_factory = lambda _config, store: Annotator(store)
        log.batches.clear()
        log.groups_at_drain.clear()
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual((config.output_root / "sft.jsonl").read_bytes(), before)
        # Every boundary measured an empty delta, so the driver was never even
        # created and only the closing drain ran.
        self.assertEqual([len(batch) for batch in log.batches], [0])
        self.assertEqual(resumed["annotation"]["batches"], 0)


class SettledBytesTests(LandFixture):
    """A landed asset is deleted from the tmpfs; annotation must not race it."""

    def annotator_asserting_archive_reads(self, db_path, failures):
        class ArchiveReadingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                for task in self.tasks(only):
                    after = str(task["candidate"]["after_path"])
                    try:
                        if only is not None:
                            # The batch contract: handed over only once the local
                            # copy is gone and the archive answers for it.
                            assert not Path(after).is_file(), after
                            assert path_exists(after, db_path=db_path), after
                            assert read_bytes(after, db_path=db_path)
                    except BaseException as exc:  # reported, never swallowed
                        failures.append(exc)
                        raise
                    self.append_task(task)
                self.store.checkpoint()
                return {
                    "completed": 0, "terminal": 0, "transport_failed": 0,
                    "pending": len(self.store.pending_annotation_tasks()),
                }

        return ArchiveReadingAnnotator

    def test_a_still_staged_winner_waits_instead_of_being_handed_over(self) -> None:
        """Without a mid-pass checkpoint nothing has landed, so nothing goes out."""
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        manifest = self.build(config, dependencies)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["annotation"]["batches"], 0)
        self.assertEqual(manifest["annotation"]["pending"], 0)
        self.assertEqual(len(scan_jsonl(config.output_root / "sft.jsonl").records), 4)

    def test_a_landed_winner_is_handed_over_and_read_out_of_the_archive(self) -> None:
        failures: list[BaseException] = []
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        dependencies.annotator_factory = (
            lambda _config, store: self.annotator_asserting_archive_reads(
                self.catalog_db, failures
            )(store)
        )
        # Land after every committed source, so both passes' groups are in the
        # archive — and unlinked from the tmpfs — before their boundary.
        with mock.patch.object(agent, "LAND_WATERMARK_BYTES", 0), \
                mock.patch.object(agent, "LAND_MIN_GROUPS", 0), \
                mock.patch.object(agent, "LAND_MIN_INTERVAL_SECONDS", 0.0):
            manifest = self.build(config, dependencies)
        self.assertEqual(failures, [])
        self.assertEqual(manifest["status"], "complete")
        self.assertGreaterEqual(manifest["annotation"]["batches"], 1)
        self.assertEqual(manifest["annotation"]["pending"], 0)
        self.assertEqual(len(scan_jsonl(config.output_root / "sft.jsonl").records), 4)

    def test_the_settled_test_follows_the_local_file_not_the_journal(self) -> None:
        """Unit form of the predicate the boundary filters on."""
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        self.build(config, dependencies)
        with ArtifactStore(config.output_root, config.build_id) as store:
            pipeline = agent.CanonicalPipeline.__new__(agent.CanonicalPipeline)
            pipeline.dependencies = dependencies
            tasks = store.pending_annotation_tasks()
            self.assertEqual(tasks, [])
            landed = {"candidate": {"after_path": str(self.root / "gone.jpg")}}
            staged = {"candidate": {"after_path": str(self.root / "here.jpg")}}
            Path(staged["candidate"]["after_path"]).write_bytes(b"x")
            self.assertTrue(pipeline._annotation_bytes_settled(landed))
            self.assertFalse(pipeline._annotation_bytes_settled(staged))
            # No archive at all means nothing can ever be unlinked.
            pipeline.dependencies = dataclasses.replace(dependencies, archive_root=None)
            self.assertTrue(pipeline._annotation_bytes_settled(staged))


class DriverTests(unittest.TestCase):
    """The background thread itself: serial, accounted, and never silent."""

    class Drainer:
        def __init__(self, hook=None):
            self.calls: list[frozenset[str]] = []
            self.concurrent = 0
            self.overlap = 0
            self._hook = hook
            self._lock = threading.Lock()

        def drain(self, *, max_workers=None, only=None):
            with self._lock:
                self.concurrent += 1
                self.overlap = max(self.overlap, self.concurrent)
            try:
                self.calls.append(frozenset(only or ()))
                if self._hook is not None:
                    self._hook(only)
            finally:
                with self._lock:
                    self.concurrent -= 1
            return {"completed": 0}

    def test_batches_run_one_at_a_time_in_submission_order(self) -> None:
        drainer = self.Drainer(hook=lambda _only: time.sleep(0.01))
        driver = _AnnotationDriver(drainer)
        try:
            for index in range(4):
                driver.submit(frozenset({f"task-{index}"}))
        finally:
            driver.close()
        self.assertEqual(
            drainer.calls, [frozenset({f"task-{index}"}) for index in range(4)]
        )
        self.assertEqual(drainer.overlap, 1)
        self.assertEqual(driver.batches, 4)
        self.assertEqual(driver.inflight, 0)

    def test_inflight_counts_what_has_not_come_back_yet(self) -> None:
        gate = threading.Event()
        seen: list[int] = []
        driver = None

        def hook(_only):
            seen.append(driver.inflight)
            gate.wait(10.0)

        drainer = self.Drainer(hook=hook)
        driver = _AnnotationDriver(drainer)
        try:
            driver.submit(frozenset({"a", "b", "c"}))
            while not seen:
                time.sleep(0.005)
            self.assertEqual(seen[0], 3)
        finally:
            gate.set()
            driver.close()
        self.assertEqual(driver.inflight, 0)

    def test_the_first_error_is_kept_and_re_raised_on_the_caller(self) -> None:
        def hook(_only):
            raise RuntimeError("relay exploded")

        driver = _AnnotationDriver(self.Drainer(hook=hook))
        try:
            driver.submit(frozenset({"a"}))
            deadline = time.monotonic() + 10.0
            while driver.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertIsInstance(driver.error, RuntimeError)
            with self.assertRaisesRegex(RuntimeError, "relay exploded"):
                driver.submit(frozenset({"b"}))
        finally:
            driver.close()
        self.assertEqual(driver.inflight, 0)

    def test_a_batch_queued_behind_an_error_is_dropped_not_run(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def hook(only):
            started.set()
            release.wait(10.0)
            raise RuntimeError("relay exploded")

        drainer = self.Drainer(hook=hook)
        driver = _AnnotationDriver(drainer)
        try:
            driver.submit(frozenset({"a"}))
            started.wait(10.0)
            driver.submit(frozenset({"b"}))
            release.set()
        finally:
            driver.close()
        self.assertEqual(drainer.calls, [frozenset({"a"})])
        self.assertIsInstance(driver.error, RuntimeError)
        self.assertEqual(driver.inflight, 0)

    def test_close_is_idempotent(self) -> None:
        driver = _AnnotationDriver(self.Drainer())
        driver.close()
        driver.close()
        with self.assertRaisesRegex(PipelineError, "driver is closed"):
            driver.submit(frozenset({"a"}))


class DriverFailureTests(InterleaveFixture):
    def test_a_driver_error_fails_the_build_instead_of_disappearing(self) -> None:
        class ExplodingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                if only:
                    raise RuntimeError("relay exploded")
                return super().drain(max_workers=max_workers, only=only)

        with self.assertRaisesRegex(RuntimeError, "relay exploded"):
            self.run_build(target=4, sources=2, uses=2, annotator=ExplodingAnnotator)

    def test_a_last_pass_error_surfaces_at_the_join(self) -> None:
        """Nothing submits after the last boundary, so ``close`` is what reports."""
        class ExplodingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                if only:
                    raise RuntimeError("relay exploded")
                return super().drain(max_workers=max_workers, only=only)

        config = self.reuse_config(target=2, uses=1)
        inventory = SourceInventoryResult(
            tuple(self.source(index) for index in range(2)),
            {"cache_entries": 2, "eligible": 2}, "ok",
        )
        dependencies = self.dependencies(inventory)
        dependencies.annotator_factory = (
            lambda _config, store: ExplodingAnnotator(store)
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(RuntimeError, "relay exploded"):
                run(config, dependencies=dependencies)
        manifest = json.loads((config.output_root / "manifest.json").read_text())
        self.assertEqual(manifest["phase"], "annotation")


class StoreRaceTests(unittest.TestCase):
    """The collections the two threads share are snapshotted, not walked live.

    ``dict`` refuses to be iterated across an insert, and both authoritative
    indexes now have a writer on one thread and a reader on the other: the
    renderer appends groups while the annotator looks for work, and the annotator
    appends SFT rows while the renderer counts them into a manifest.

    The interpreter switch interval is pinned down for the duration, because at
    the 5 ms default a whole walk usually finishes inside one time slice and the
    race is simply not sampled — which is exactly how a bug like this reaches
    production and then only shows up on the build that runs for a week.
    """

    def setUp(self) -> None:
        self.tmp = __import__("tempfile").TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._switch = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)

    def tearDown(self) -> None:
        sys.setswitchinterval(self._switch)
        self.tmp.cleanup()

    def hammer(self, writer, read) -> int:
        """Run ``read`` against ``writer`` and report how many reads landed."""
        errors: list[BaseException] = []
        stop = threading.Event()

        def write() -> None:
            try:
                writer()
            except BaseException as exc:
                errors.append(exc)
            finally:
                stop.set()

        thread = threading.Thread(target=write)
        thread.start()
        reads = 0
        while not stop.is_set():
            try:
                read()
                reads += 1
            except BaseException as exc:
                errors.append(exc)
                break
        thread.join()
        self.assertEqual([repr(exc) for exc in errors], [])
        self.assertGreater(reads, 0)
        return reads

    def test_the_pending_queue_survives_a_concurrent_render_thread(self) -> None:
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            for index in range(200):
                store.append_group(_group(index))

            def writer() -> None:
                for index in range(200, 3000):
                    store.append_group(_group(index))

            def read() -> None:
                # Every store-level walk of ``groups`` an annotation thread makes.
                store.pending_annotation_tasks()
                store.completed_source_uses()
                store.completed_sources()

            self.hammer(writer, read)
            self.assertEqual(len(store.groups), 3000)

    def test_the_manifest_readers_survive_a_concurrent_annotation_thread(self) -> None:
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            store.append_group(_group(0))
            pipeline = agent.CanonicalPipeline.__new__(agent.CanonicalPipeline)
            pipeline.store = store

            def writer() -> None:
                for index in range(3000):
                    store.append_sft({
                        "build_id": "build",
                        "sft_id": f"sft-{index:05d}",
                        "annotation_task_id": f"task-{index:05d}",
                        "group_id": "group-00000",
                        "candidate_id": f"candidate-{index}",
                        "annot_src": "responses:test",
                        "qa": {"annotation": {"usage": {"input_tokens": 1}}},
                    })
                    store.append_failure({
                        "build_id": "build",
                        "event_id": f"event-{index:05d}",
                        "stage": "annotation",
                        "task_id": f"task-{index:05d}",
                        "error_code": "attempt",
                        "terminal": False,
                    })

            def read() -> None:
                # The render thread's readers of ``sft``, and the annotation
                # thread's own; plus a ``failures`` walk, which is a list and is
                # safe to read under a concurrent append but is included so that
                # the claim is tested rather than asserted.
                store.completed_annotation_tasks()
                pipeline._annotation_counts()
                pipeline._winner_annotation_status()
                sum(1 for row in store.sft_records() if row.get("group_id"))
                len([row for row in store.failures if row.get("terminal")])

            self.hammer(writer, read)
            self.assertEqual(len(store.sft), 3000)
            self.assertEqual(len(store.failures), 3000)

    def test_a_live_walk_of_the_same_index_really_would_break(self) -> None:
        """The control: without the snapshot this is what the readers hit.

        Not a test of the shipped code — a test that the two tests above are
        sampling a race that exists, rather than passing because the workload is
        too gentle to expose one.
        """
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            store.append_group(_group(0))
            broken: list[BaseException] = []
            stop = threading.Event()

            def writer() -> None:
                for index in range(1, 3000):
                    store.append_group(_group(index))
                stop.set()

            thread = threading.Thread(target=writer)
            thread.start()
            while not stop.is_set() and not broken:
                try:
                    for _row in store.groups.values():
                        pass
                except BaseException as exc:
                    broken.append(exc)
            thread.join()
            self.assertTrue(broken, "the race under test was never sampled")
            self.assertIsInstance(broken[0], RuntimeError)

    def test_a_snapshot_is_a_copy_of_the_list_not_of_the_rows(self) -> None:
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            store.append_group(_group(0))
            first = store.group_records()
            second = store.group_records()
            self.assertIsNot(first, second)
            self.assertIs(first[0], second[0])


class RoundFoldTests(OrchestrationFixture):
    """``_next_rounds`` is ``_next_round`` for a queue, and must agree with it."""

    def annotator(self, store) -> ResponsesAnnotator:
        return ResponsesAnnotator(self.config().annotation, store)

    def test_the_fold_agrees_with_the_per_task_definition(self) -> None:
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            for index in range(6):
                store.append_group(_group(index, winners=2))
            annotator = self.annotator(store)
            tasks = [str(row["task_id"]) for row in store.pending_annotation_tasks()]
            self.assertEqual(len(tasks), 12)
            rows = [
                # task 0: nothing yet -> round 1
                # task 1: round 1 exhausted -> round 2
                (tasks[1], "round_exhausted", 1, False),
                # task 2: rounds 1 and 2 exhausted -> round 3
                (tasks[2], "round_exhausted", 1, False),
                (tasks[2], "round_exhausted", 2, False),
                # task 3: every round exhausted -> None
                (tasks[3], "round_exhausted", 1, False),
                (tasks[3], "round_exhausted", 2, False),
                (tasks[3], "round_exhausted", 3, False),
                # task 4: terminal beats any round bookkeeping
                (tasks[4], "terminal", 2, True),
                (tasks[4], "round_exhausted", 1, False),
                # task 5: an attempt event is not a round boundary
                (tasks[5], "attempt", 1, False),
            ]
            for index, (task_id, event, round_number, terminal) in enumerate(rows):
                store.append_failure({
                    "build_id": "build",
                    "event_id": f"event-{index:03d}",
                    "event_type": event,
                    "stage": "annotation",
                    "task_id": task_id,
                    "round": round_number,
                    "error_code": event,
                    "terminal": terminal,
                })
            folded = annotator._next_rounds(tasks)
            self.assertEqual(
                folded,
                {task_id: annotator._next_round(task_id) for task_id in tasks},
            )
            self.assertEqual(
                [folded[task_id] for task_id in tasks[:6]],
                [1, 2, 3, None, None, 1],
            )

    def test_the_fold_only_answers_about_the_batch_it_was_given(self) -> None:
        with ArtifactStore(self.root / "out", "build", fsync_every=64) as store:
            for index in range(3):
                store.append_group(_group(index))
            annotator = self.annotator(store)
            tasks = [str(row["task_id"]) for row in store.pending_annotation_tasks()]
            self.assertEqual(set(annotator._next_rounds(tasks[:1])), {tasks[0]})


if __name__ == "__main__":
    unittest.main()
