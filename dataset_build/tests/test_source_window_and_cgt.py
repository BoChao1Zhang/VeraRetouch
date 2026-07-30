"""Source-window sizing, its selector parity, and the off-turn C_GT contract.

The window is what decides how many sources may be preprocessing behind the one
holding the ordered selector turn, so widening it past the old hard-coded three
has to leave every durable ordering alone: allocation-ordered group commits,
preset reservations, failure lineage.  The oracle here is the same pipeline run
with ``source_window = 1``, which is literally serial.

The C_GT half checks the other side of the same critical section: the PNG is now
encoded on the postprocess pool, and the only thing the contract still demands is
that it is fsynced before any journal line can name it.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

from construct import agent
from construct.agent import run
from construct.config import ConfigError, load_config
from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink
from construct.sources import SourceInventoryResult
from construct.state import scan_jsonl

from tests.test_canonical_orchestration import FakeScorer, OrchestrationFixture


EXAMPLE = Path(__file__).resolve().parents[2] / "databuild.example.toml"


class SourceWindowConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "databuild.toml"
        self.reset()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reset(self) -> None:
        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)

    def rewrite(self, replacement: str) -> None:
        self.reset()
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("source_window = 5\n", text)
        self.path.write_text(
            text.replace("source_window = 5\n", replacement), encoding="utf-8"
        )
        self.path.chmod(0o600)

    def load(self):
        return load_config(self.path, validate_paths=False)

    def test_example_config_declares_the_five_source_window(self) -> None:
        self.assertEqual(self.load().render.source_window, 5)

    def test_supported_windows_load(self) -> None:
        for value in range(1, 9):
            with self.subTest(source_window=value):
                self.rewrite(f"source_window = {value}\n")
                self.assertEqual(self.load().render.source_window, value)

    def test_out_of_range_windows_are_rejected(self) -> None:
        for value in (0, 9, -1):
            with self.subTest(source_window=value):
                self.rewrite(f"source_window = {value}\n")
                with self.assertRaisesRegex(ConfigError, "source_window"):
                    self.load()

    def test_missing_key_keeps_configs_written_before_it_loading(self) -> None:
        # Deployed configs (eval100 among them) predate the key and must not be
        # invalidated by it; the built-in window is the documented default.
        self.rewrite("")
        config = self.load()
        self.assertNotIn("source_window", self.path.read_text(encoding="utf-8"))
        self.assertEqual(config.render.source_window, 5)

    def test_window_is_independent_of_the_render_semaphore(self) -> None:
        self.rewrite("source_window = 5\n")
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(
            text.replace("gpu_concurrency = 2", "gpu_concurrency = 1"), encoding="utf-8"
        )
        self.path.chmod(0o600)
        config = self.load()
        self.assertEqual(config.render.gpu_concurrency, 1)
        self.assertEqual(config.render.source_window, 5)


class WindowFixture(OrchestrationFixture):
    """A ten-group build wide enough that a window of five is really used."""

    SOURCE_COUNT = 14
    TARGET = 10

    def build(self, *, window: int, tag: str):
        config = self.config(target=self.TARGET, local=0.5, global_=0.5)
        config = dataclasses.replace(
            config,
            output_root=self.root / f"out-{tag}",
            render=dataclasses.replace(config.render, source_window=window),
        )
        sources = tuple(self.source(index) for index in range(self.SOURCE_COUNT))
        inventory = SourceInventoryResult(sources, {"eligible": len(sources)}, "ok")
        dependencies = self.dependencies(inventory)
        # The window only widens for the dual-scorer production shape; injected
        # single scorers keep the pre-pool window, so the pool has to be declared.
        dependencies.scorer_pool_factory = lambda _instances: FakeScorer()
        return config, dependencies

    def journals(self, config):
        def normalize(rows):
            root = str(config.output_root)
            return [
                json.loads(json.dumps(row).replace(root, "<output-root>"))
                for row in rows
            ]

        return {
            name: normalize(scan_jsonl(config.output_root / f"{name}.jsonl").records)
            for name in ("groups", "failures", "sft")
        }

    def run_local_build(self, tag: str, *extra):
        config, dependencies = self.build(window=5, tag=tag)
        config = dataclasses.replace(
            config, mix=dataclasses.replace(config.mix, local=1.0, global_=0.0)
        )
        with ExitStack() as stack:
            stack.enter_context(
                mock.patch("construct.agent.preprocess_source", self.small_preprocess)
            )
            for patch in extra:
                stack.enter_context(patch)
            manifest = run(config, dependencies=dependencies)
        return config, manifest


class SourceWindowPipelineTests(WindowFixture):
    def test_window_comes_from_config_only_when_a_scorer_pool_is_present(self) -> None:
        observed: dict[str, int] = {}
        base = agent.CanonicalPipeline._fill_initial_mode

        def spy(pipeline, mode, sources, target):
            observed["window"] = pipeline._source_window
            return base(pipeline, mode, sources, target)

        for window, pooled, expected in ((5, True, 5), (7, True, 7), (5, False, 2)):
            with self.subTest(window=window, pooled=pooled):
                config, dependencies = self.build(
                    window=window, tag=f"probe-{window}-{pooled}"
                )
                if not pooled:
                    dependencies.scorer_pool_factory = None
                with mock.patch.object(
                    agent.CanonicalPipeline, "_fill_initial_mode", spy
                ), mock.patch(
                    "construct.agent.preprocess_source", self.small_preprocess
                ):
                    run(config, dependencies=dependencies)
                self.assertEqual(observed["window"], expected)
                self.assertEqual(config.render.gpu_concurrency, 2)

    def test_window_five_journals_exactly_what_the_serial_oracle_journals(self) -> None:
        peak = self._run_and_measure_peak(window=5, tag="window5")
        wide = self.journals(self.built["window5"])
        # Without real concurrency past three the comparison would prove nothing.
        self.assertGreaterEqual(peak, 5)

        serial_config, serial_dependencies = self.build(window=1, tag="serial")
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            run(serial_config, dependencies=serial_dependencies)
        serial = self.journals(serial_config)

        self.assertEqual(len(wide["groups"]), self.TARGET)
        self.assertEqual(wide["groups"], serial["groups"])
        self.assertEqual(wide["failures"], serial["failures"])
        self.assertEqual(wide["sft"], serial["sft"])

    def test_window_five_preserves_preset_reservations_and_coverage_order(self) -> None:
        peak = self._run_and_measure_peak(window=5, tag="reservations")
        self.assertGreaterEqual(peak, 5)
        wide = self.journals(self.built["reservations"])
        serial_config, serial_dependencies = self.build(window=1, tag="reservations-serial")
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            run(serial_config, dependencies=serial_dependencies)
        serial = self.journals(serial_config)

        def signature(rows):
            return [
                (
                    row["source_id"],
                    row["render_mode"],
                    row["reservation_id"],
                    row["coverage_cycle"],
                    row["coverage_position"],
                    row["major"],
                    [
                        (
                            candidate["slot_id"],
                            candidate["preset_id"],
                            candidate["attempt_lineage"]["preset_reservation_id"],
                            candidate["attempt_lineage"]["preset_attempt"],
                        )
                        for candidate in row["candidates"]
                    ],
                )
                for row in rows
            ]

        self.assertEqual(signature(wide["groups"]), signature(serial["groups"]))
        positions = [row["coverage_position"] for row in wide["groups"]]
        self.assertEqual(positions, sorted(positions))

    def _run_and_measure_peak(self, *, window: int, tag: str) -> int:
        """Run one build and report how many sources were ever in flight at once.

        The first source is held at the head of the turn chain until the window
        has actually filled, so the count is a fact about the run rather than a
        race the scheduler happened to win.
        """
        config, dependencies = self.build(window=window, tag=tag)
        self.built = getattr(self, "built", {})
        self.built[tag] = config

        lock = threading.Lock()
        filled = threading.Event()
        active: set[str] = set()
        peak = 0
        leaders: set[str] = set()
        base = agent.CanonicalPipeline._render_source_buffered

        def wrapper(pipeline, source, mode, *, selector_turn=None):
            nonlocal peak
            with lock:
                active.add(source.source_id)
                peak = max(peak, len(active))
                if len(active) >= window:
                    filled.set()
                leader = mode not in leaders
                leaders.add(mode)
            try:
                if leader:
                    filled.wait(timeout=30)
                return base(pipeline, source, mode, selector_turn=selector_turn)
            finally:
                with lock:
                    active.discard(source.source_id)

        with mock.patch.object(
            agent.CanonicalPipeline, "_render_source_buffered", wrapper
        ), mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=dependencies)
        self.assertEqual(manifest["completed"]["groups"], self.TARGET)
        return peak


class CgtOffTurnTests(WindowFixture):
    SOURCE_COUNT = 6
    TARGET = 4

    def test_cgt_is_encoded_on_the_postprocess_pool_and_once_per_mask(self) -> None:
        calls: list[tuple[str, str]] = []
        lock = threading.Lock()
        base = agent.save_cgt_png

        def spy(mask, path):
            with lock:
                calls.append((mask.mask_id, threading.current_thread().name))
            return base(mask, path)

        config, _ = self.run_local_build(
            "cgt-threads", mock.patch("construct.agent.save_cgt_png", spy)
        )

        self.assertTrue(calls)
        for _, thread_name in calls:
            self.assertTrue(
                thread_name.startswith("databuild-postprocess"),
                f"C_GT encoded on {thread_name}; it must never run on the turn",
            )
        mask_ids = [mask_id for mask_id, _ in calls]
        self.assertEqual(len(mask_ids), len(set(mask_ids)))
        # Seven physical masks stand behind eight slots: the two semantic slots
        # share one asset, so a source may not pay for eight encodes.
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        for group in groups:
            self.assertEqual(
                len({candidate["mask_id"] for candidate in group["candidates"]}), 7
            )
        self.assertEqual(len(mask_ids), 7 * len({row["source_id"] for row in groups}))

    def test_every_journalled_cgt_is_already_fsynced_and_complete(self) -> None:
        observed: list[str] = []
        base = agent.ArtifactStore.append_group

        def guarded(store, group, **kwargs):
            for candidate in group["candidates"]:
                path = Path(candidate["cgt_path"])
                # The durable contract: the journal may not name a mask that is
                # not already whole on disk, and never the atomic-write scratch.
                if not path.is_file():
                    raise AssertionError(f"journalled C_GT is missing: {path}")
                if path.with_name(path.name + ".tmp").exists():
                    raise AssertionError(f"journalled C_GT is still partial: {path}")
                observed.append(str(path))
            return base(store, group, **kwargs)

        config, manifest = self.run_local_build(
            "cgt-durable",
            mock.patch.object(agent.ArtifactStore, "append_group", guarded),
        )
        self.assertEqual(manifest["completed"]["groups"], self.TARGET)
        self.assertTrue(observed)
        for group in scan_jsonl(config.output_root / "groups.jsonl").records:
            for candidate in group["candidates"]:
                self.assertTrue(Path(candidate["cgt_path"]).is_file())

    def test_a_failed_cgt_write_stops_the_group_before_it_is_journalled(self) -> None:
        def broken(_mask, path):
            raise OSError(f"no space for {path}")

        config, manifest = self.run_local_build(
            "cgt-broken",
            mock.patch("construct.agent.save_cgt_png", broken),
        )
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(list(groups), [])
        self.assertEqual(manifest["completed"]["groups"], 0)
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        self.assertTrue(any(row.get("terminal") for row in failures))


class CgtFailureCostTests(WindowFixture):
    """A dead C_GT write must not cost one full render wave per taxonomy major."""

    SOURCE_COUNT = 4
    TARGET = 2
    MAJORS = 3

    def catalog(self):
        links = []
        for major in range(self.MAJORS):
            for index in range(8):
                preset = PresetRecord(
                    preset_id=f"preset-{major}-{index}",
                    path=self.root / f"preset-{major}-{index}.cube",
                    format="lut",
                    kind="lut",
                    style_name="Style Name",
                    fidelity_de=None,
                    render_engine="gpu_lut",
                )
                links.append(
                    TaxonomyLink(preset, f"major-{major}", f"minor-{index % 2}")
                )
        return PresetCatalog.from_links(links)

    def test_a_failed_cgt_write_is_not_re_rendered_once_per_major(self) -> None:
        encoded: list[str] = []
        lock = threading.Lock()
        base = agent.save_candidate_jpeg

        def counting(after, after_path, **kwargs):
            with lock:
                encoded.append(str(after_path))
            return base(after, after_path, **kwargs)

        def broken(_mask, path):
            raise OSError(f"no space for {path}")

        config, manifest = self.run_local_build(
            "cgt-cost",
            mock.patch("construct.agent.save_cgt_png", broken),
            mock.patch("construct.agent.save_candidate_jpeg", counting),
        )
        self.assertEqual(manifest["completed"]["groups"], 0)
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        exhausted = [
            row for row in failures if row.get("error_code") == "major_exhausted"
        ]
        attempted = {row["source_id"] for row in exhausted}
        self.assertTrue(attempted)
        # Every major is still failed and journalled the same way ...
        self.assertEqual(len(exhausted), self.MAJORS * len(attempted))
        for row in exhausted:
            self.assertIn("OSError", row["message"])
        # ... but only the first of them pays for a render wave.
        self.assertEqual(len(encoded), 8 * len(attempted))
        self.assertTrue(
            any(row.get("terminal") for row in failures)
        )


class ResumeConfigCompatibilityTests(OrchestrationFixture):
    """A manifest written before ``render.source_window`` must stay resumable.

    ``sanitized_dict`` grew the key, so an exact comparison rejects the resume of
    every build created before it — the failure mode this guards is the eval100
    relabel, whose manifest differs from its own config in that one key only.
    """

    def first_run(self):
        config = self.config()
        sources = tuple(self.source(index) for index in range(4))
        inventory = SourceInventoryResult(sources, {"eligible": 4}, "ok")
        dependencies = self.dependencies(inventory)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=dependencies)
        self.assertEqual(manifest["status"], "complete")
        return config, dependencies

    def rewrite_manifest_config(self, config, mutate) -> None:
        path = config.output_root / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mutate(manifest["effective_config"])
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_a_manifest_without_a_neutral_key_still_resumes(self) -> None:
        # Every key on the closed neutral list, not just the first one added.
        for key in ("source_window", "qa_scorer_instances"):
            with self.subTest(key=key):
                config, dependencies = self.first_run()
                before = {
                    name: len(scan_jsonl(config.output_root / name).records)
                    for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
                }
                self.rewrite_manifest_config(
                    config, lambda effective: effective["render"].pop(key)
                )
                with mock.patch(
                    "construct.agent.preprocess_source", self.small_preprocess
                ):
                    resumed = run(config, dependencies=dependencies)
                self.assertEqual(resumed["status"], "complete")
                self.assertEqual(
                    before,
                    {
                        name: len(scan_jsonl(config.output_root / name).records)
                        for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
                    },
                )
                self.assertEqual(
                    resumed["effective_config"]["render"][key],
                    getattr(config.render, key),
                )

    def test_a_real_difference_is_still_refused_and_names_the_key(self) -> None:
        config, dependencies = self.first_run()
        changed = dataclasses.replace(
            config, render=dataclasses.replace(config.render, jpeg_quality=90)
        )
        with self.assertRaises(agent.StateError) as caught:
            run(changed, dependencies=dependencies)
        self.assertIn("render.jpeg_quality", str(caught.exception))
        self.assertNotIn("source_window", str(caught.exception))

    def test_only_the_listed_key_may_be_missing_from_a_manifest(self) -> None:
        config, dependencies = self.first_run()
        self.rewrite_manifest_config(
            config, lambda effective: effective["render"].pop("iaa_batch")
        )
        with self.assertRaises(agent.StateError) as caught:
            run(config, dependencies=dependencies)
        self.assertIn("render.iaa_batch", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
