"""Stop-fill: retire the rendering half of a build and give the cards back.

L8 is being stopped at the ~61.5k groups it has rather than the 400k it was
configured for, with roughly 50k winners still unannotated.  The two halves of
that are different jobs: the annotation backlog is relay and CPU work that must
still run to completion, and the rendering is what has to stop *now* because the
GPUs are owed to training.

``<output_root>/STOP_FILL`` is the switch.  These are the properties that make it
usable on a build that is already running under production discipline:

* it is a marker file, not a config key, so setting it cannot make the build
  unresumable (``run`` compares ``effective_config`` path by path) and clearing
  it needs no durable edit;
* seen mid-pass it stops the walk at a source boundary, and the sources already
  in flight still commit — nothing is cancelled and no group is half-journalled;
* seen at startup it means no renderer, no OneAlign and no SAM3 are ever
  constructed, so the process holds no CUDA context at all;
* the drain, the shortfall accounting and the projection are untouched: the
  build still finishes, it just finishes short and says so.
"""
from __future__ import annotations

import dataclasses
import os
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from construct import agent
from construct.agent import PipelineError, run
from construct.config import redact_text
from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink
from construct.sources import SourceInventoryResult
from construct.state import scan_jsonl

from dataset_build.tests.test_canonical_orchestration import (
    FakeAnnotator,
    FakeRenderer,
    FakeScorer,
    OrchestrationFixture,
)


class RefusingAnnotator(FakeAnnotator):
    """Answers nothing, so every task it is handed stays pending.

    Used to manufacture the backlog the real build is being stopped with: a run
    that renders its groups and then cannot annotate them leaves a durable queue
    behind, which is the state a stop-fill restart has to drain.
    """

    def drain(self, *, max_workers=None, only=None):
        return {
            "completed": 0, "terminal": 0, "transport_failed": len(self.tasks(only)),
            "pending": len(self.store.pending_annotation_tasks()),
        }


class StopFillFixture(OrchestrationFixture):
    """A global-only build with a bank big enough for more than one pass."""

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

    def fill_config(self, *, target, uses=1, local=0.0, global_=1.0, relabel_attempts=1):
        config = self.config(
            target=target, local=local, global_=global_,
            relabel_attempts=relabel_attempts,
        )
        return dataclasses.replace(
            config,
            sources=dataclasses.replace(config.sources, max_source_uses=uses),
        )

    def marker(self, config) -> Path:
        return Path(config.output_root) / agent.STOP_FILL_MARKER

    def set_marker(self, config) -> Path:
        path = self.marker(config)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("stop", encoding="utf-8")
        return path

    def inventory(self, count, *, border_from=None):
        records = [
            self.source(index, border_mask=border_from is not None and index >= border_from)
            for index in range(count)
        ]
        return SourceInventoryResult(
            tuple(records), {"cache_entries": count, "eligible": count}, "ok"
        )

    def gpu_free_dependencies(self, inventory, *, relabeler=None):
        """Dependencies whose every GPU load point fails the test if it is used.

        This is the whole no-GPU claim, expressed as the thing that would have to
        happen for it to be false.  ``renderer_factory`` is
        ``LocalGpuOnlyRenderer.create`` in production (its preflight allocates a
        probe tensor, and that allocation *is* the CUDA context),
        ``scorer_factory``/``scorer_pool_factory`` are the OneAlign copies on
        cuda:0, and ``relabeler`` is SAM3.  There is no fourth.
        """
        dependencies = self.dependencies(inventory)
        dependencies.renderer_factory = mock.Mock(
            side_effect=AssertionError("renderer constructed under stop-fill")
        )
        dependencies.scorer_factory = mock.Mock(
            side_effect=AssertionError("OneAlign constructed under stop-fill")
        )
        dependencies.scorer_pool_factory = mock.Mock(
            side_effect=AssertionError("OneAlign pool constructed under stop-fill")
        )
        dependencies.relabeler = relabeler or mock.Mock(
            side_effect=AssertionError("SAM3 relabeler called under stop-fill")
        )
        return dependencies

    def assert_no_gpu_load_point_was_used(self, dependencies) -> None:
        self.assertFalse(dependencies.renderer_factory.called)
        self.assertFalse(dependencies.scorer_factory.called)
        self.assertFalse(dependencies.scorer_pool_factory.called)
        self.assertFalse(dependencies.relabeler.called)

    def run_build(self, config, dependencies, *, annotator=None):
        if annotator is not None:
            dependencies.annotator_factory = lambda _config, store: annotator(store)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            return run(config, dependencies=dependencies)

    def stop_events(self, config):
        return [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row.get("error_code") == "stop_fill_requested"
        ]


class MarkerTests(StopFillFixture):
    """The predicate itself: what counts as "stop" and what does not."""

    def test_an_output_root_without_the_marker_is_not_a_stop(self) -> None:
        root = self.root / "out"
        root.mkdir()
        self.assertFalse(agent._stop_fill_requested(root))

    def test_an_output_root_that_does_not_exist_yet_is_not_a_stop(self) -> None:
        """A first run creates the root; absence must not read as "stopped"."""
        self.assertFalse(agent._stop_fill_requested(self.root / "never-created"))

    def test_every_way_an_operator_might_set_it_counts(self) -> None:
        for name, make in (
            ("touch", lambda path: path.touch()),
            ("echo", lambda path: path.write_text("stopping L8\n", encoding="utf-8")),
            ("mkdir", lambda path: path.mkdir()),
        ):
            with self.subTest(name):
                root = self.root / f"out-{name}"
                root.mkdir()
                make(root / agent.STOP_FILL_MARKER)
                self.assertTrue(agent._stop_fill_requested(root))

    def test_the_marker_lives_at_the_documented_path(self) -> None:
        self.assertEqual(agent.STOP_FILL_MARKER, "STOP_FILL")
        self.assertEqual(
            agent._stop_fill_marker(self.root / "out"), self.root / "out" / "STOP_FILL"
        )


class StartupStopFillTests(StopFillFixture):
    """The marker is already there when the process starts: nothing loads."""

    def test_no_renderer_no_scorer_and_no_relabeler_is_ever_constructed(self) -> None:
        config = self.fill_config(target=3)
        self.set_marker(config)
        dependencies = self.gpu_free_dependencies(self.inventory(3))
        manifest = self.run_build(config, dependencies)
        self.assert_no_gpu_load_point_was_used(dependencies)
        self.assertEqual(manifest["completed"]["groups"], 0)
        self.assertEqual(manifest["stop_fill"]["requested"], True)
        self.assertEqual(manifest["stop_fill"]["at_start"], True)
        self.assertEqual(manifest["stop_fill"]["gpu_resources_loaded"], False)

    def test_the_build_still_finishes_the_whole_lifecycle(self) -> None:
        """Stopping the fill is not failing the build: it reaches projection."""
        config = self.fill_config(target=3)
        self.set_marker(config)
        manifest = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(manifest["phase"], "complete_with_failures")
        self.assertTrue(manifest["projection"]["ok"])
        self.assertIn("ended_at", manifest)

    def test_the_shortfall_is_journalled_against_the_configured_target(self) -> None:
        """``local_target_shortfall`` is the honest 400k-minus-rendered row."""
        config = self.fill_config(target=7, local=1.0, global_=0.0)
        self.set_marker(config)
        manifest = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertEqual(manifest["targets"]["local"], 7)
        self.assertEqual(manifest["completed"]["local"], 0)
        self.assertEqual(manifest["sources"]["exhaustion"]["local_shortfall"], 7)
        rows = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "local_target_shortfall"
        ]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["terminal"])
        # Journalled through the same redactor every other failure row uses; the
        # fixture's DSN makes ``u`` and ``p`` secrets, so the raw string differs.
        self.assertEqual(
            rows[0]["message"],
            redact_text("requested 7 local groups, completed 0", config.secrets),
        )

    def test_a_partly_filled_build_reports_the_gap_it_actually_has(self) -> None:
        """Stop a build that already rendered: the gap is target minus rendered."""
        # Target six from a pool of three at one use each: the first run meets
        # what it can, records a shortfall of three, and stops on its own.
        config = self.fill_config(target=6)
        first = self.run_build(config, self.dependencies(self.inventory(3)))
        self.assertEqual(first["completed"]["groups"], 3)
        # The marker changes nothing about the accounting: same target, same gap.
        self.set_marker(config)
        second = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertEqual(second["completed"]["groups"], 3)
        self.assertEqual(second["sources"]["exhaustion"]["global_shortfall"], 3)
        rows = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "global_target_shortfall"
        ]
        self.assertEqual(
            rows[-1]["message"],
            redact_text("requested 6 global groups, completed 3", config.secrets),
        )

    def test_the_stop_is_journalled_once_and_is_not_terminal(self) -> None:
        """The event says why; the shortfall row is the terminal one."""
        config = self.fill_config(target=3)
        self.set_marker(config)
        manifest = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        events = self.stop_events(config)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "stop_fill")
        self.assertFalse(events[0]["terminal"])
        self.assertFalse(events[0]["retryable"])
        self.assertIn("STOP_FILL", events[0]["message"])
        # ``complete_with_failures`` is reached by the shortfall, exactly as it
        # would be for a build that ran out of sources.
        terminal = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["terminal"]
        ]
        self.assertEqual(
            {row["error_code"] for row in terminal}, {"global_target_shortfall"}
        )

    def test_no_pass_is_walked_at_all(self) -> None:
        """The two ledger scans a pass starts with are not even paid for."""
        config = self.fill_config(target=3, uses=2)
        self.set_marker(config)
        with mock.patch.object(
            agent.CanonicalPipeline, "_fill_initial_mode", autospec=True
        ) as walk:
            self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertFalse(walk.called)

    def test_the_annotation_backlog_is_drained_with_no_card(self) -> None:
        """The reason the process is kept alive at all: ~50k owed annotations."""
        config = self.fill_config(target=3)
        inventory = self.inventory(3)
        # A first run that renders everything and can annotate nothing leaves the
        # durable queue a stop-fill restart is there to finish.
        with self.assertRaises(PipelineError):
            self.run_build(
                config, self.dependencies(inventory), annotator=RefusingAnnotator
            )
        self.assertEqual(len(scan_jsonl(config.output_root / "groups.jsonl").records), 3)
        self.assertEqual(len(scan_jsonl(config.output_root / "sft.jsonl").records), 0)

        self.set_marker(config)
        dependencies = self.gpu_free_dependencies(self.inventory(3))
        manifest = self.run_build(config, dependencies)
        self.assert_no_gpu_load_point_was_used(dependencies)
        # Every winner of the stopped build now carries training text, and the
        # build is complete rather than short: its groups already met the target.
        # Two winners a group, because the fixture's scorer ties every candidate.
        self.assertEqual(manifest["annotation"]["pending"], 0)
        self.assertEqual(manifest["completed"]["groups"], 3)
        self.assertEqual(manifest["completed"]["sft"], 6)
        self.assertEqual(manifest["status"], "complete")
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len(rows), 6)

    def test_an_undrainable_backlog_is_still_an_error(self) -> None:
        """Stop-fill retires rendering, never the completion gate on annotation."""
        config = self.fill_config(target=3)
        with self.assertRaises(PipelineError):
            self.run_build(
                config, self.dependencies(self.inventory(3)), annotator=RefusingAnnotator
            )
        self.set_marker(config)
        with self.assertRaises(PipelineError) as caught:
            self.run_build(
                config,
                self.gpu_free_dependencies(self.inventory(3)),
                annotator=RefusingAnnotator,
            )
        self.assertIn("annotation queue remains unresolved", str(caught.exception))


class Sam3UnderStopFillTests(StopFillFixture):
    """The relabel phase is the *other* renderer, and it is skipped whole."""

    def relabeler(self):
        def repair(batch, _config, _attempt):
            for source in batch:
                mask = np.zeros((32, 48), dtype=np.uint8)
                mask[9:24, 16:33] = 255
                Image.fromarray(mask, "L").save(source.subject_path)
            return {source.source_id: "ready" for source in batch}

        return mock.Mock(side_effect=repair)

    def stop_on_queue(self):
        """Set the marker the moment a source is parked on the SAM3 queue.

        The border mask makes ``build_mask_plan`` raise, which is what writes the
        ``sam3_relabel_queued`` row.  Hooking the commit that returns that status
        reproduces the production shape exactly: the marker arrives while the
        queue is non-empty and before the relabel phase opens.
        """
        original = agent.CanonicalPipeline._commit_source_result

        def spy(pipeline, result):
            status = original(pipeline, result)
            if status == "sam3_queued":
                self.set_marker(pipeline.config)
            return status

        return mock.patch.object(
            agent.CanonicalPipeline, "_commit_source_result", spy
        )

    def stopped_build_with_a_queued_source(self, config, *, relabeler=None):
        dependencies = self.dependencies(
            self.inventory(2, border_from=1),
            relabeler=relabeler or mock.Mock(
                side_effect=AssertionError("SAM3 relabeler called under stop-fill")
            ),
        )
        with self.stop_on_queue():
            manifest = self.run_build(config, dependencies)
        return manifest, dependencies

    def build_and_keep_the_pipeline(self, config, dependencies):
        """Run a build and hand back the ``CanonicalPipeline`` it used.

        The closing check in ``_drain_sam3_and_replacements`` is only reachable
        when the loop does not run, and inside a whole build that cannot coincide
        with a set marker — ``_run_phases`` would have skipped the phase.  It is
        reachable in production only by the marker landing between the loop's
        last look and the check, so it is pinned here directly instead.
        """
        captured = {}
        original = agent.CanonicalPipeline.execute

        def capture(pipeline):
            captured["pipeline"] = pipeline
            return original(pipeline)

        with mock.patch.object(agent.CanonicalPipeline, "execute", capture):
            with self.stop_on_queue():
                self.run_build(config, dependencies)
        return captured["pipeline"]

    def test_the_relabeler_is_never_called_and_the_queue_is_left_intact(self) -> None:
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        manifest, dependencies = self.stopped_build_with_a_queued_source(config)
        self.assertFalse(dependencies.relabeler.called)
        queued = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "sam3_relabel_queued"
        ]
        self.assertEqual(len(queued), 1)
        # Still pending, still owed, and not charged an attempt it never got.
        self.assertEqual(manifest["sam3_relabel"]["pending"], 1)
        self.assertEqual(manifest["sam3_relabel"]["attempt_events"], 0)
        self.assertEqual(manifest["sam3_relabel"]["terminal"], 0)
        self.assertEqual(manifest["status"], "complete_with_failures")

    def test_an_unresolved_queue_is_not_reported_as_a_drain_failure(self) -> None:
        """The ``PipelineError`` means "the drain ran and could not finish"."""
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        # No assertRaises: that is the claim.  Without the guard this build ends
        # in ``SAM3 relabel queue remains unresolved`` and never projects.
        manifest, _ = self.stopped_build_with_a_queued_source(config)
        self.assertTrue(manifest["projection"]["ok"])
        self.assertIn("ended_at", manifest)

    def test_the_phase_is_not_entered_at_all(self) -> None:
        """Skipped whole, not entered and turned round inside.

        On L8 the first two lines of the drain are a dict over 26k sources and a
        scan of 61.5k groups, and the phase is skipped for the same reason the
        fill is: there is no card to do any of its work on.
        """
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        with mock.patch.object(
            agent.CanonicalPipeline, "_drain_sam3_and_replacements", autospec=True
        ) as drain:
            self.stopped_build_with_a_queued_source(config)
        self.assertFalse(drain.called)

    def test_a_marker_arriving_during_the_phase_retires_it_between_rounds(self) -> None:
        """The other boundary: the drain itself is hours long on a real build."""
        config = self.fill_config(
            target=2, local=1.0, global_=0.0, relabel_attempts=3
        )

        def set_marker_and_repair_nothing(batch, _config, _attempt):
            self.set_marker(config)
            return {source.source_id: "unchanged" for source in batch}

        relabeler = mock.Mock(side_effect=set_marker_and_repair_nothing)
        manifest = self.run_build(config, self.dependencies(
            self.inventory(2, border_from=1), relabeler=relabeler,
        ))
        # One round, then the loop retires: without the between-rounds check the
        # source would burn all three attempts and go terminal.
        self.assertEqual(relabeler.call_count, 1)
        self.assertEqual(manifest["sam3_relabel"]["pending"], 1)
        self.assertEqual(manifest["sam3_relabel"]["terminal"], 0)
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertTrue(manifest["projection"]["ok"])

    def test_a_marker_that_lands_as_the_drain_exits_is_not_a_crash(self) -> None:
        """The closing check, both ways, on a real pipeline with a real queue.

        A drain whose loop never runs falls straight to "the queue is still
        pending, raise".  That error means *the drain ran and could not finish*,
        so under a marker it must not fire — the drain was told not to try.
        """
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        pipeline = self.build_and_keep_the_pipeline(
            config,
            self.dependencies(
                self.inventory(2, border_from=1),
                relabeler=mock.Mock(side_effect=AssertionError("relabeler called")),
            ),
        )
        self.assertEqual(len(pipeline._pending_sam3_ids()), 1)
        # A target already met is what makes the loop body unreachable.
        pipeline.allocation = dataclasses.replace(
            pipeline.allocation, local_target=0
        )

        # Marker present: the pending queue is a backlog handed on, not an error.
        pipeline._stop_fill_latched = False
        self.assertTrue(self.marker(config).exists())
        pipeline._drain_sam3_and_replacements()

        # Marker gone: the same state is the error it always was.
        self.marker(config).unlink()
        pipeline._stop_fill_latched = False
        with self.assertRaises(PipelineError) as caught:
            pipeline._drain_sam3_and_replacements()
        self.assertIn("SAM3 relabel queue remains unresolved", str(caught.exception))

    def test_removing_the_marker_lets_the_queue_be_rescued_later(self) -> None:
        """A backlog handed on, not thrown away."""
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        self.stopped_build_with_a_queued_source(config)
        self.marker(config).unlink()
        relabeler = self.relabeler()
        manifest = self.run_build(
            config,
            self.dependencies(self.inventory(2, border_from=1), relabeler=relabeler),
        )
        self.assertTrue(relabeler.called)
        self.assertEqual(manifest["sam3_relabel"]["pending"], 0)
        self.assertEqual(manifest["completed"]["local"], 2)


class MidPassStopFillTests(StopFillFixture):
    """The marker appears while the walk is running."""

    def stop_after(self, commits: int):
        """Create the marker once ``commits`` sources have been committed.

        ``_commit_source_result`` is the allocation-order boundary and runs on the
        main thread, so hooking it makes "the marker appeared mid-pass"
        reproducible rather than a race against the render workers.
        """
        original = agent.CanonicalPipeline._commit_source_result
        state = {"seen": 0}

        def spy(pipeline, result):
            status = original(pipeline, result)
            state["seen"] += 1
            if state["seen"] == commits:
                self.set_marker(pipeline.config)
            return status

        return mock.patch.object(
            agent.CanonicalPipeline, "_commit_source_result", spy
        ), state

    def test_the_sources_already_in_flight_still_commit(self) -> None:
        """Graceful: the window drains, nothing is cancelled, nothing is lost.

        The window is two sources, so the marker landing on the first commit lets
        the two already submitted finish and stops the walk before a fourth is
        opened: three groups from a five-source pool.
        """
        config = self.fill_config(target=5)
        dependencies = self.dependencies(self.inventory(5))
        patch, _state = self.stop_after(1)
        with patch:
            manifest = self.run_build(config, dependencies)
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(groups), 3)
        self.assertEqual(manifest["completed"]["groups"], 3)
        # Every one of them is a whole group with its eight candidates and a
        # winner: a stopped walk does not leave a half-built record behind.
        for group in groups:
            self.assertEqual(len(group["candidates"]), 8)
            self.assertTrue(group["winner_ids"])
        self.assertEqual(manifest["sources"]["exhaustion"]["global_shortfall"], 2)
        self.assertEqual(manifest["status"], "complete_with_failures")

    def test_the_manifest_separates_a_mid_run_stop_from_a_startup_one(self) -> None:
        config = self.fill_config(target=5)
        patch, _state = self.stop_after(1)
        with patch:
            manifest = self.run_build(config, self.dependencies(self.inventory(5)))
        self.assertEqual(manifest["stop_fill"]["requested"], True)
        # It did load the cards — it was already rendering — and says so.
        self.assertEqual(manifest["stop_fill"]["at_start"], False)
        self.assertEqual(manifest["stop_fill"]["gpu_resources_loaded"], True)

    def test_the_cut_short_pass_still_hands_its_winners_over(self) -> None:
        """The flush is before the stop-return, so the tail is pipelined too."""
        config = self.fill_config(target=5)
        patch, _state = self.stop_after(1)
        with patch:
            manifest = self.run_build(config, self.dependencies(self.inventory(5)))
        self.assertEqual(manifest["annotation"]["batches"], 1)
        self.assertEqual(manifest["annotation"]["inflight"], 0)
        self.assertEqual(manifest["annotation"]["pending"], 0)
        # Three groups, two tied winners each: the whole tail was annotated.
        self.assertEqual(manifest["completed"]["sft"], 6)

    def test_the_second_mode_is_not_started_after_the_marker(self) -> None:
        """Global stops, and local never opens a pass behind it."""
        config = self.fill_config(target=6, local=0.5, global_=0.5)
        patch, _state = self.stop_after(1)
        with patch:
            manifest = self.run_build(config, self.dependencies(self.inventory(6)))
        self.assertEqual(manifest["completed"]["local"], 0)
        self.assertEqual(manifest["completed"]["global"], 3)
        self.assertEqual(manifest["sources"]["exhaustion"]["local_shortfall"], 3)

    def test_a_later_pass_is_not_opened(self) -> None:
        """Reuse is where a pass boundary actually exists; it must retire there."""
        config = self.fill_config(target=6, uses=3)
        patch, _state = self.stop_after(1)
        with patch:
            manifest = self.run_build(config, self.dependencies(self.inventory(2)))
        # Two sources, window two: both were in flight when the marker landed, so
        # pass 0 completes and pass 1 is never opened.
        self.assertEqual(manifest["completed"]["groups"], 2)
        self.assertEqual(manifest["sources"]["source_reuse"]["uses_histogram"], {"1": 2})

    def test_the_stop_is_recorded_with_the_groups_it_stopped_at(self) -> None:
        config = self.fill_config(target=5)
        patch, _state = self.stop_after(1)
        with patch:
            self.run_build(config, self.dependencies(self.inventory(5)))
        events = self.stop_events(config)
        self.assertEqual(len(events), 1)
        self.assertIn('"at_start": false', events[0]["message"])


class ReversibilityTests(StopFillFixture):
    """Delete the marker, restart, and the build fills again."""

    def test_removing_the_marker_and_restarting_resumes_filling(self) -> None:
        config = self.fill_config(target=3)
        self.set_marker(config)
        stopped = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertEqual(stopped["completed"]["groups"], 0)

        self.marker(config).unlink()
        dependencies = self.dependencies(self.inventory(3))
        resumed = self.run_build(config, dependencies)
        self.assertEqual(resumed["completed"]["groups"], 3)
        self.assertEqual(resumed["sources"]["exhaustion"]["global_shortfall"], 0)
        self.assertEqual(resumed["stop_fill"]["requested"], False)
        self.assertEqual(resumed["stop_fill"]["gpu_resources_loaded"], True)
        # The durable shortfall row the stopped run wrote is history, not a
        # retraction: the build really was short once, so the status stays
        # ``complete_with_failures`` even though the target is now met.
        self.assertEqual(resumed["status"], "complete_with_failures")

    def test_the_marker_is_not_part_of_the_effective_config(self) -> None:
        """Why it is a file: a config key would refuse the resume outright."""
        config = self.fill_config(target=3)
        self.set_marker(config)
        stopped = self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.marker(config).unlink()
        resumed = self.run_build(config, self.dependencies(self.inventory(3)))
        self.assertEqual(stopped["effective_config"], resumed["effective_config"])
        self.assertNotIn("stop", str(stopped["effective_config"]).lower())

    def test_the_switch_can_be_thrown_more_than_once(self) -> None:
        """Stop mid-pass, stay stopped across a restart, then fill again."""
        config = self.fill_config(target=6, uses=3)
        original = agent.CanonicalPipeline._commit_source_result

        def spy(pipeline, result):
            status = original(pipeline, result)
            self.set_marker(pipeline.config)
            return status

        with mock.patch.object(agent.CanonicalPipeline, "_commit_source_result", spy):
            first = self.run_build(config, self.dependencies(self.inventory(2)))
        # Pass 0 finished its window; pass 1 was never opened.
        self.assertEqual(first["completed"]["groups"], 2)

        # Restart with the marker still there: still no card, still two groups.
        dependencies = self.gpu_free_dependencies(self.inventory(2))
        second = self.run_build(config, dependencies)
        self.assert_no_gpu_load_point_was_used(dependencies)
        self.assertEqual(second["completed"]["groups"], 2)

        self.marker(config).unlink()
        third = self.run_build(config, self.dependencies(self.inventory(2)))
        self.assertEqual(third["completed"]["groups"], 6)
        self.assertEqual(third["sources"]["source_reuse"]["uses_histogram"], {"3": 2})


class RestartWithAShortfallTests(StopFillFixture):
    """A stopped build restarts to drain, so it re-enters ``_run_phases``.

    ``append_failure`` deduplicates byte-identical rows only, and every row
    carries a timestamp, so the *second* run's shortfall row is a conflict rather
    than a no-op.  On pristine HEAD (no stop-fill in it at all) that already
    killed any restart of a build that had recorded a shortfall — it was hidden
    because the fixtures freeze the clock and because no production build had
    ever reached ``_record_shortfalls`` twice.  Stop-fill makes it the normal
    path, so these run with a clock that actually moves.
    """

    def moving_clock_dependencies(self, inventory, *, stopped):
        dependencies = (
            self.gpu_free_dependencies(inventory) if stopped
            else self.dependencies(inventory)
        )
        ticks = iter(range(1, 10_000))
        import datetime

        dependencies.now = lambda: datetime.datetime(
            2026, 8, 14, 12, 0, 0, tzinfo=datetime.timezone.utc
        ) + datetime.timedelta(seconds=next(ticks))
        return dependencies

    def test_a_short_build_can_be_restarted(self) -> None:
        """The regression: identical shortfall, later timestamp, no conflict."""
        config = self.fill_config(target=6)
        first = self.run_build(
            config, self.moving_clock_dependencies(self.inventory(3), stopped=False)
        )
        self.assertEqual(first["completed"]["groups"], 3)
        self.assertEqual(first["status"], "complete_with_failures")
        second = self.run_build(
            config, self.moving_clock_dependencies(self.inventory(3), stopped=False)
        )
        self.assertEqual(second["completed"]["groups"], 3)
        self.assertEqual(second["sources"]["exhaustion"]["global_shortfall"], 3)
        rows = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "global_target_shortfall"
        ]
        # Said once, because nothing about the shortfall changed.
        self.assertEqual(len(rows), 1)

    def test_a_stopped_build_can_be_restarted_to_keep_draining(self) -> None:
        config = self.fill_config(target=6)
        self.set_marker(config)
        for _ in range(3):
            manifest = self.run_build(
                config, self.moving_clock_dependencies(self.inventory(3), stopped=True)
            )
            self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(len(self.stop_events(config)), 1)
        rows = [
            row for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "global_target_shortfall"
        ]
        self.assertEqual(len(rows), 1)

    def test_a_build_that_filled_further_journals_the_new_number(self) -> None:
        """Silence is for a restart that changed nothing, not for a real change."""
        config = self.fill_config(target=9, uses=3)
        original = agent.CanonicalPipeline._commit_source_result

        def stop_on_first(pipeline, result):
            status = original(pipeline, result)
            self.set_marker(pipeline.config)
            return status

        with mock.patch.object(
            agent.CanonicalPipeline, "_commit_source_result", stop_on_first
        ):
            first = self.run_build(
                config, self.moving_clock_dependencies(self.inventory(2), stopped=False)
            )
        self.assertEqual(first["completed"]["groups"], 2)
        self.marker(config).unlink()
        second = self.run_build(
            config, self.moving_clock_dependencies(self.inventory(2), stopped=False)
        )
        self.assertEqual(second["completed"]["groups"], 6)
        messages = sorted(
            row["message"]
            for row in scan_jsonl(config.output_root / "failures.jsonl").records
            if row["error_code"] == "global_target_shortfall"
        )
        self.assertEqual(messages, [
            redact_text("requested 9 global groups, completed 2", config.secrets),
            redact_text("requested 9 global groups, completed 6", config.secrets),
        ])


class NoCudaContextTests(StopFillFixture):
    """Nothing on the drain path may initialise the driver."""

    def fake_torch(self, *, initialized: bool):
        cuda = SimpleNamespace(calls=[])
        cuda.is_initialized = lambda: initialized
        cuda.empty_cache = lambda: cuda.calls.append("empty_cache")
        return types.SimpleNamespace(cuda=cuda)

    def test_the_cache_is_not_emptied_when_no_model_was_ever_loaded(self) -> None:
        """``empty_cache`` on an uninitialised driver is what would create one."""
        torch = self.fake_torch(initialized=False)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            agent._empty_cuda_cache()
        self.assertEqual(torch.cuda.calls, [])

    def test_the_cache_is_still_emptied_for_a_build_that_did_load(self) -> None:
        torch = self.fake_torch(initialized=True)
        with mock.patch.dict(sys.modules, {"torch": torch}):
            agent._empty_cuda_cache()
        self.assertEqual(torch.cuda.calls, ["empty_cache"])

    def test_a_stopped_build_never_reaches_the_allocator_at_all(self) -> None:
        config = self.fill_config(target=3)
        self.set_marker(config)
        with mock.patch.object(agent, "_empty_cuda_cache") as empty:
            self.run_build(config, self.gpu_free_dependencies(self.inventory(3)))
        self.assertFalse(empty.called)

    def test_a_normal_build_still_releases_the_cards_before_annotation(self) -> None:
        """The guard must not have turned the ordinary release into a no-op."""
        config = self.fill_config(target=3)
        with mock.patch.object(agent, "_empty_cuda_cache") as empty:
            self.run_build(config, self.dependencies(self.inventory(3)))
        self.assertTrue(empty.called)

    def in_a_clean_interpreter(self, body: str) -> str:
        """Run ``body`` in a fresh process, so no peer test can poison the answer.

        Several tests in this suite legitimately drive real torch, and one of
        them leaves ``torch.cuda.is_initialized()`` True for the rest of the
        process.  A claim about what a *build process* touches has to be made in
        a process that only ran the build.
        """
        import subprocess

        result = subprocess.run(
            [sys.executable, "-c", body],
            capture_output=True, text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        )
        self.assertEqual(result.returncode, 0, result.stderr[-4000:])
        return result.stdout.strip()

    def test_the_drain_path_modules_do_not_import_torch(self) -> None:
        """The audit, as an assertion: no module the drain touches pulls torch in.

        ``construct.agent`` re-exports ``rendering`` and ``visibility``, both of
        which use torch — inside functions only, and none of those functions is
        on the drain path.
        """
        answer = self.in_a_clean_interpreter(
            "import sys;"
            "import construct.agent, construct.responses, construct.projection;"
            "from dataset_build.tools import archive_reader, land, global_catalog;"
            "print('torch' in sys.modules)"
        )
        self.assertEqual(answer, "False")

    def test_a_stopped_build_leaves_the_driver_uninitialised(self) -> None:
        """End to end in a fresh process, against the real torch and real CUDA.

        The build renders nothing, annotates its (empty) queue and projects, and
        the interpreter it did that in never opened a context — which is the
        whole promise made to the training queue.
        """
        answer = self.in_a_clean_interpreter(
            "import tempfile, sys;"
            "from unittest import mock;"
            "from construct.agent import run, STOP_FILL_MARKER;"
            "from construct.sources import SourceInventoryResult;"
            "from dataset_build.tests.test_canonical_orchestration import"
            " OrchestrationFixture;"
            "F = type('F', (OrchestrationFixture,), {'runTest': lambda self: None});"
            "f = F(); f.setUp();"
            "cfg = f.config(target=3, local=0.0, global_=1.0);"
            "cfg.output_root.mkdir(parents=True, exist_ok=True);"
            "(cfg.output_root / STOP_FILL_MARKER).write_text('stop');"
            "inv = SourceInventoryResult(tuple(f.source(i) for i in range(3)),"
            " {'cache_entries': 3, 'eligible': 3}, 'ok');"
            "d = f.dependencies(inv);"
            "d.renderer_factory = d.scorer_factory = d.relabeler ="
            " (lambda *a, **k: (_ for _ in ()).throw(AssertionError('gpu')));"
            "m = mock.patch('construct.agent.preprocess_source', f.small_preprocess);"
            "m.start();"
            "manifest = run(cfg, dependencies=d);"
            "m.stop();"
            "import torch;"
            "print(manifest['status'], manifest['completed']['groups'],"
            " torch.cuda.is_initialized(), 'torch' in sys.modules)"
        )
        # The build finished short, rendered nothing, and the driver is cold.
        # ``torch`` is only in ``sys.modules`` because the assertion imported it.
        self.assertEqual(answer, "complete_with_failures 0 False True")


class UnchangedWithoutTheMarkerTests(StopFillFixture):
    """Every claim above has to cost nothing when the marker is absent."""

    def test_a_build_without_the_marker_fills_and_completes(self) -> None:
        config = self.fill_config(target=3)
        manifest = self.run_build(config, self.dependencies(self.inventory(3)))
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"]["groups"], 3)
        self.assertEqual(self.stop_events(config), [])
        self.assertEqual(manifest["stop_fill"]["requested"], False)
        self.assertEqual(manifest["stop_fill"]["at_start"], False)

    def test_reuse_still_walks_every_pass(self) -> None:
        config = self.fill_config(target=6, uses=3)
        manifest = self.run_build(config, self.dependencies(self.inventory(2)))
        self.assertEqual(manifest["completed"]["groups"], 6)
        self.assertEqual(manifest["sources"]["source_reuse"]["uses_histogram"], {"3": 2})

    def test_the_sam3_drain_still_runs_and_still_resolves_its_queue(self) -> None:
        config = self.fill_config(target=2, local=1.0, global_=0.0)
        relabeler = mock.Mock(side_effect=lambda batch, _config, _attempt: {
            source.source_id: "unchanged" for source in batch
        })
        manifest = self.run_build(config, self.dependencies(
            self.inventory(2, border_from=1), relabeler=relabeler,
        ))
        self.assertTrue(relabeler.called)
        self.assertEqual(manifest["sam3_relabel"]["pending"], 0)
        # A relabeler that repairs nothing spends the budget and retires the
        # source, which is the drain doing its job rather than being skipped.
        self.assertEqual(manifest["sam3_relabel"]["terminal"], 1)

    def test_the_renderer_is_still_constructed_and_bound(self) -> None:
        config = self.fill_config(target=3)
        renderer = FakeRenderer()
        renderer.bind_catalog = mock.Mock(return_value=None)
        renderer.assert_ready = mock.Mock(return_value=None)
        dependencies = self.dependencies(self.inventory(3))
        dependencies.renderer_factory = mock.Mock(return_value=renderer)
        dependencies.scorer_factory = mock.Mock(return_value=FakeScorer())
        self.run_build(config, dependencies)
        self.assertTrue(dependencies.renderer_factory.called)
        self.assertTrue(dependencies.scorer_factory.called)
        self.assertTrue(renderer.bind_catalog.called)
        self.assertTrue(renderer.assert_ready.called)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
