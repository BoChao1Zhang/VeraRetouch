from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from construct import agent
from construct.agent import PipelineDependencies, run
from construct.config import (
    AnnotationConfig,
    DatabuildConfig,
    ExternalEndpointConfig,
    LegacyImportConfig,
    LocalAnnotationConfig,
    MasksConfig,
    MixConfig,
    PresetsConfig,
    RenderConfig,
    SourcesConfig,
    ViewerConfig,
)
from construct.responses import build_prompt
from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink
from construct.projection import ProjectionResult, project_artifacts
from construct.rendering import PreparedSource, preprocess_source
from construct.sources import (
    SourceInventoryResult,
    SourceRecord,
    allocate_sources,
)
from construct.state import ArtifactStore, scan_jsonl, stable_id


class FakeRenderer:
    device = "cuda:test"

    def bind_catalog(self, catalog):
        return None

    def assert_ready(self):
        return None

    def render(self, source, preset, mask=None):
        edited = np.clip(source.pixels + 0.55, 0.0, 1.0)
        if mask is None:
            output = edited
        else:
            alpha = mask.effective_alpha[..., None]
            mixed = source.pixels * (1.0 - alpha) + edited * alpha
            output = np.where(alpha == 0, source.pixels, mixed).astype(np.float32)
        return SimpleNamespace(
            pixels=output,
            engine=preset.render_engine,
            diagnostics={"fake": True},
        )


class FakeScorer:
    def score(self, path):
        return 82.0 if "candidate_" in Path(path).name else 45.0


class FakeAnnotator:
    def __init__(self, store):
        self.store = store

    def append_task(self, task):
        candidate = task["candidate"]
        group = task["group"]
        return self.store.append_sft({
            "build_id": self.store.build_id,
            "sft_id": stable_id("sft", task["task_id"]),
            "annotation_task_id": task["task_id"],
            "group_id": task["group_id"],
            "candidate_id": task["candidate_id"],
            "winner_rank": task["winner_rank"],
            "I_in": group["source_path"],
            "I_tar": candidate["after_path"],
            "recipe": candidate["recipe"],
            "local": None,
            "task_type": "local" if group["render_mode"] == "local" else "style",
            "instruction": "Make the requested visible photographic adjustment.",
            "instruction_short": "Apply the requested adjustment.",
            "reasoning": "reasoning",
            "annot_src": "responses:test",
            "qa": candidate["qa"],
        })

    def tasks(self, only=None):
        """The pending queue this drain owns, honouring the batch contract."""
        return [
            task for task in self.store.pending_annotation_tasks()
            if only is None or str(task["task_id"]) in only
        ]

    def drain(self, *, max_workers=None, only=None):
        completed = 0
        for task in self.tasks(only):
            self.append_task(task)
            completed += 1
        self.store.checkpoint()
        return {
            "completed": completed, "terminal": 0, "transport_failed": 0,
            "pending": len(self.store.pending_annotation_tasks()),
        }


class OrchestrationFixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def config(self, *, target=2, local=0.5, global_=0.5, relabel_attempts=1):
        return DatabuildConfig(
            schema_version=1,
            build_id="build",
            seed=7,
            target_groups=target,
            output_root=self.root / "out",
            preset_filter="all",
            mix=MixConfig(local=local, global_=global_),
            sources=SourcesConfig(self.root / "cache", "postgresql://u:p@db/source"),
            presets=PresetsConfig(self.root / "bank", self.root / "taxonomy.jsonl", 6.0, ()),
            # ``FakeScorer`` answers one constant for every candidate, so every
            # group here is an exact rank1-rank2 tie.  A margin policy cannot be
            # evaluated against a stub with no ranking signal, so it is switched
            # off (0.0/0.0) for the orchestration fixtures and exercised on its
            # own terms in test_winner_margin.py, which re-enables it on top of
            # this very config precisely because the ties are total.
            render=RenderConfig(1024, 95, 2, 512, 2.5, 2.3, 0.5,
                                qa_winner_margin_abstain=0.0,
                                qa_winner_margin_low=0.0),
            masks=MasksConfig(0.5, relabel_attempts),
            annotation=AnnotationConfig(
                "external", 768, 90, "low", 6000, 4, 3,
                (
                    ExternalEndpointConfig("a", "https://a.example/v1", "a-key", 2),
                    ExternalEndpointConfig("b", "https://b.example/v1", "b-key", 2),
                ),
                LocalAnnotationConfig(
                    "http://127.0.0.1:8003/v1", "EMPTY", "qwen3_5-35b-a3b",
                    0.2, False, 2048,
                ),
            ),
            viewer=ViewerConfig("postgresql://u:p@db/viewer"),
        )

    def catalog(self):
        links = []
        for index in range(8):
            preset = PresetRecord(
                preset_id=f"preset-{index}",
                path=self.root / f"preset-{index}.cube",
                format="lut",
                kind="lut",
                style_name="Style Name",
                fidelity_de=None,
                render_engine="gpu_lut",
            )
            links.append(TaxonomyLink(preset, "major", f"minor-{index % 2}"))
        return PresetCatalog.from_links(links)

    def source(self, index: int, *, border_mask=False):
        cache = self.root / "cache" / f"source-{index}"
        cache.mkdir(parents=True, exist_ok=True)
        source_path = self.root / f"source-{index}.jpg"
        Image.fromarray(np.full((32, 48, 3), 55 + index, np.uint8), "RGB").save(source_path)
        mask = np.zeros((32, 48), dtype=np.uint8)
        if border_mask:
            mask[:2] = 255
            mask[-2:] = 255
            mask[:, :2] = 255
            mask[:, -2:] = 255
        else:
            mask[9:24, 16:33] = 255
        subject_path = cache / "subject.png"
        Image.fromarray(mask, "L").save(subject_path)
        meta = {
            "status": "ready",
            "asset_id": f"source-{index}",
            "source_path": str(source_path),
            "sam_prompt": "person",
            "description": "person near center",
        }
        meta_path = cache / "subject.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return SourceRecord(
            source_id=f"source-{index}",
            source_path=source_path,
            cache_dir=cache,
            subject_path=subject_path,
            subject_meta_path=meta_path,
            scene="portrait",
            subject={"name": "person", "description": "person near center"},
            mask_area=float((mask > 0).mean()),
        )

    def dependencies(self, inventory, *, relabeler=None):
        return PipelineDependencies(
            inventory_loader=lambda _config: inventory,
            catalog_loader=lambda _config: self.catalog(),
            renderer_factory=lambda _config: FakeRenderer(),
            scorer_factory=FakeScorer,
            sdk_preflight=lambda: None,
            annotator_factory=lambda _config, store: FakeAnnotator(store),
            relabeler=relabeler or (lambda sources, config, attempt: {}),
            projector=lambda store, manifest, config: ProjectionResult(
                True,
                {"groups": len(store.groups), "sft": len(store.sft)},
            ),
            now=lambda: __import__("datetime").datetime(
                2026, 7, 20, 12, 0, 0, tzinfo=__import__("datetime").timezone.utc
            ),
        )

    @staticmethod
    def small_preprocess(path, _short_edge):
        return preprocess_source(path, short_edge=32)

    def legacy_config(self):
        legacy_root = self.root / "legacy"
        legacy_root.mkdir()
        source_path = legacy_root / "source.jpg"
        Image.fromarray(np.full((32, 48, 3), 60, np.uint8), "RGB").save(source_path)
        candidates = []
        sft_rows = []
        for index in range(8):
            after_path = legacy_root / f"after-{index}.jpg"
            Image.fromarray(
                np.full((32, 48, 3), 120 + index * 4, np.uint8), "RGB"
            ).save(after_path)
            cgt_path = legacy_root / f"cgt-{index}.png"
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[8:25, 12:37] = 255
            Image.fromarray(mask, "L").save(cgt_path)
            mode = "linear_bisect" if index == 0 else "radial"
            local = {
                "route": "geom",
                "mask_unit_id": f"mask-{index}",
                "mask_type": mode,
                "mode": mode,
                "subject": {"concept": "person", "scope": "single"},
                "geom": {"index": index},
                "cgt_path": str(cgt_path),
                "region": "center",
                "blend_mode": "exact",
                "engine": "gpu_local_preset",
                "base_preset_id": f"base-{index}",
                "base_preset_path": str(legacy_root / f"base-{index}.xmp"),
                "base_preset_content_hash": f"hash-{index}",
            }
            qa = {
                "veto": False,
                "reliable": True,
                "q": 0.8 - index * 0.01,
                "rank": index,
                "iaa_mixed": 80.0 - index,
                "source_iaa": 55.0,
                "iaa_impr": 0.6,
                "det": [],
            }
            candidates.append({
                "after_path": str(after_path),
                "content_hash": f"content-{index}",
                "fmt": "xmp",
                "kind": "local_preset",
                "preset_id": f"preset-{index}",
                "preset_path": str(legacy_root / f"preset-{index}.xmp"),
                "qa": qa,
                "local": local,
            })
            if index < 2:
                sft_rows.append({
                    "I_in": str(source_path),
                    "I_tar": str(after_path),
                    "annot_src": "deferred",
                    "instruction": None,
                    "instruction_short": None,
                    "reasoning": None,
                    "task_type": "local",
                    "local": {"C_GT": str(cgt_path)},
                    "qa": {"rank": index, "q": qa["q"]},
                    "recipe": {"base_preset_id": f"base-{index}"},
                })
        group = {
            "source": str(source_path),
            "source_asset_id": "legacy-source",
            "is_portrait": False,
            "source_iaa": 55.0,
            "local": True,
            "style_major": "legacy-major",
            "candidates": candidates,
        }
        groups_path = legacy_root / "groups.jsonl"
        sft_path = legacy_root / "sft.jsonl"
        groups_path.write_text(json.dumps(group) + "\n", encoding="utf-8")
        sft_path.write_text(
            "".join(json.dumps(row) + "\n" for row in sft_rows), encoding="utf-8"
        )
        config = dataclasses.replace(
            self.config(target=1, local=1.0, global_=0.0),
            output_root=self.root / "legacy-output",
            preset_filter="xmp",
            legacy_import=LegacyImportConfig(
                input_root=legacy_root,
                groups_jsonl=groups_path,
                sft_jsonl=sft_path,
                expected_source_groups=1,
                expected_sft_rows=2,
                protocol="r5_local_50k_v4",
                axis_fix_status="param_only_no_native_lut",
                project_to_viewer=False,
            ),
        )
        return config, groups_path, sft_path


class PipelineTests(OrchestrationFixture):
    def test_legacy_import_reannotates_durably_without_mutating_source(self):
        config, groups_path, sft_path = self.legacy_config()
        before = (groups_path.read_bytes(), sft_path.read_bytes())
        dependencies = self.dependencies(SourceInventoryResult((), {}, "unused"))
        dependencies.inventory_loader = mock.Mock(side_effect=AssertionError("inventory called"))
        dependencies.catalog_loader = mock.Mock(side_effect=AssertionError("catalog called"))
        dependencies.renderer_factory = mock.Mock(side_effect=AssertionError("renderer called"))
        dependencies.scorer_factory = mock.Mock(side_effect=AssertionError("scorer called"))
        dependencies.projector = mock.Mock(side_effect=AssertionError("projector called"))

        manifest = run(config, dependencies=dependencies)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"]["groups"], 1)
        self.assertEqual(manifest["completed"]["sft"], 2)
        self.assertFalse(manifest["legacy_import"]["old_text_imported"])
        self.assertFalse(manifest["legacy_import"]["canonical_mask_contract"])
        self.assertEqual(manifest["projection"]["counts"], {})
        self.assertEqual(before, (groups_path.read_bytes(), sft_path.read_bytes()))

        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]["candidates"]), 8)
        self.assertEqual(len(groups[0]["winner_ids"]), 2)
        winners = {
            row["candidate_id"]: row for row in groups[0]["candidates"]
            if row["candidate_id"] in groups[0]["winner_ids"]
        }
        for candidate in winners.values():
            self.assertTrue(Path(candidate["after_path"]).is_file())
            self.assertTrue(Path(candidate["cgt_path"]).is_file())
            self.assertEqual(
                candidate["objective_hints"]["brightness"]["direction"], "brighter"
            )
            task = {"group": groups[0], "candidate": candidate}
            self.assertIn("local task", build_prompt(task))
        self.assertEqual(groups[0]["candidates"][0]["slot_mode"], "linear_bisect")

        resumed = run(config, dependencies=dependencies)
        self.assertEqual(resumed["completed"]["groups"], 1)
        self.assertEqual(resumed["completed"]["sft"], 2)
        self.assertEqual(len(scan_jsonl(config.output_root / "groups.jsonl").records), 1)
        self.assertEqual(len(scan_jsonl(config.output_root / "sft.jsonl").records), 2)

    def test_onealign_live_preflight_fails_before_output_mutation(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        inventory = SourceInventoryResult((self.source(0),), {"eligible": 1}, "ok")
        dependencies = self.dependencies(inventory)

        class BrokenScorer:
            def score(self, _path):
                raise RuntimeError("forward failed")

        dependencies.scorer_factory = BrokenScorer
        with self.assertRaisesRegex(Exception, "live preflight failed"):
            run(config, dependencies=dependencies)
        self.assertFalse(config.output_root.exists())

    def test_visibility_rejection_refills_without_advancing_successful_coverage(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")
        links = []
        for index in range(9):
            preset = PresetRecord(
                preset_id=f"refill-preset-{index}",
                path=self.root / f"refill-preset-{index}.cube",
                format="lut",
                kind="lut",
                style_name="Style Name",
                fidelity_de=None,
                render_engine="gpu_lut",
            )
            links.append(TaxonomyLink(preset, "major", "minor"))
        catalog = PresetCatalog.from_links(links)

        class NoOpOnceRenderer(FakeRenderer):
            def __init__(self):
                self.calls = 0

            def render(self, prepared, preset, mask=None):
                self.calls += 1
                if self.calls == 1:
                    return SimpleNamespace(
                        pixels=prepared.pixels.copy(),
                        engine=preset.render_engine,
                        diagnostics={"fake": "no-op"},
                    )
                return super().render(prepared, preset, mask)

        dependencies = self.dependencies(inventory)
        dependencies.catalog_loader = lambda _config: catalog
        dependencies.renderer_factory = lambda _config: NoOpOnceRenderer()
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=dependencies)

        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        visibility_failures = [
            row for row in failures if row.get("error_code") == "visibility_rejected"
        ]
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(visibility_failures), 1)
        attempts = [
            row["attempt_lineage"]["preset_attempt"]
            for row in groups[0]["candidates"]
        ]
        self.assertEqual(sorted(attempts), [1, 1, 1, 1, 1, 1, 1, 2])
        selector_usage = manifest["presets"]["usage"]["selector"]["global"]
        self.assertEqual(sum(selector_usage["major"].values()), 1)
        self.assertEqual(sum(selector_usage["preset"].values()), 8)
        self.assertEqual(
            manifest["presets"]["usage"]["candidate_failure_reasons"],
            {"visibility_rejected": 1},
        )

    def test_complete_local_global_run_and_resume_are_idempotent(self):
        config = self.config()
        sources = (self.source(0), self.source(1), self.source(2), self.source(3))
        inventory = SourceInventoryResult(
            sources, {"cache_entries": 4, "eligible": 4}, "ok"
        )
        dependencies = self.dependencies(inventory)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            first = run(config, dependencies=dependencies)
        self.assertEqual(first["status"], "complete")
        self.assertEqual(first["completed"]["groups"], 2)
        self.assertEqual(first["completed"]["local"], 1)
        self.assertEqual(first["completed"]["global"], 1)
        self.assertEqual(first["completed"]["candidates"], 16)
        self.assertEqual(first["completed"]["sft"], 4)
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        for group in groups:
            self.assertEqual(
                group["winner_ranks"], list(range(1, len(group["winner_ids"]) + 1))
            )
        self.assertFalse((config.output_root / "dpo.jsonl").exists())
        manifest_artifact = first["artifacts"]["manifest.json"]
        self.assertEqual(manifest_artifact["records"], 1)
        self.assertIn("excluding artifacts.manifest.json", manifest_artifact["hash_scope"])
        self.assertEqual(
            manifest_artifact["bytes"], (config.output_root / "manifest.json").stat().st_size
        )
        before_counts = {
            name: len(scan_jsonl(config.output_root / name).records)
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        after_counts = {
            name: len(scan_jsonl(config.output_root / name).records)
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(before_counts, after_counts)

    def test_sam3_batches_keep_the_renderer_and_scorer_resident(self):
        """SAM3 与 OneAlign 同卡共存：每批次卸载重建曾花 11.5-11.8s/轮。"""
        config = self.config(target=1, local=1.0, global_=0.0, relabel_attempts=1)
        sources = [self.source(index) for index in range(3)]
        allocation = allocate_sources(
            sources,
            build_id=config.build_id,
            seed=config.seed,
            target_groups=1,
            mix=config.mix,
        )
        first = allocation.local[0]
        border = np.zeros((32, 48), dtype=np.uint8)
        border[:2] = border[-2:] = 255
        border[:, :2] = border[:, -2:] = 255
        Image.fromarray(border, "L").save(first.subject_path)
        inventory = SourceInventoryResult(
            tuple(sources), {"cache_entries": 3, "eligible": 3}, "ok"
        )

        events: list[str] = []
        renderers: list[object] = []
        scorers: list[object] = []

        def relabeler(batch, _config, _attempt):
            events.append("relabel")
            return {source.source_id: "ready" for source in batch}

        dependencies = self.dependencies(inventory, relabeler=relabeler)
        base_renderer, base_scorer = dependencies.renderer_factory, dependencies.scorer_factory
        dependencies.renderer_factory = lambda config_: (
            renderers.append(base_renderer(config_)) or renderers[-1]
        )
        dependencies.scorer_factory = lambda: scorers.append(base_scorer()) or scorers[-1]
        base_release = agent.CanonicalPipeline._release_heavy_resources

        def release_spy(pipeline):
            events.append("release")
            base_release(pipeline)

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(
                    agent.CanonicalPipeline, "_release_heavy_resources", release_spy
                ):
            run(config, dependencies=dependencies)

        # SAM3 批次两侧不再有 release/restore；唯一一次释放留给 annotation。
        self.assertEqual(events, ["relabel", "release"])
        self.assertEqual(len(renderers), 1)
        self.assertEqual(len(scorers), 1)

    def test_annotation_phase_still_hands_the_cards_back(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")
        dependencies = self.dependencies(inventory)
        released: list[tuple[object, object]] = []
        base = agent.CanonicalPipeline._release_heavy_resources

        def spy(pipeline):
            base(pipeline)
            released.append((pipeline.renderer, pipeline.scorer))

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(agent.CanonicalPipeline, "_release_heavy_resources", spy):
            run(config, dependencies=dependencies)
        # 唯一一次释放发生在 annotation 之前，且确实把两个引用清空了。
        self.assertEqual(released, [(None, None)])

    def test_terminal_sam3_source_uses_same_mode_replacement(self):
        config = self.config(target=1, local=1.0, global_=0.0, relabel_attempts=1)
        sources = [self.source(index) for index in range(3)]
        allocation = allocate_sources(
            sources,
            build_id=config.build_id,
            seed=config.seed,
            target_groups=1,
            mix=config.mix,
        )
        first = allocation.local[0]
        border = np.zeros((32, 48), dtype=np.uint8)
        border[:2] = border[-2:] = 255
        border[:, :2] = border[:, -2:] = 255
        Image.fromarray(border, "L").save(first.subject_path)
        inventory = SourceInventoryResult(
            tuple(sources), {"cache_entries": 3, "eligible": 3}, "ok"
        )

        calls = []

        def relabeler(batch, _config, attempt):
            calls.append(([source.source_id for source in batch], attempt))
            return {source.source_id: "ready" for source in batch}

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=self.dependencies(
                inventory, relabeler=relabeler
            ))
        self.assertEqual(manifest["status"], "complete_with_failures")
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(groups), 1)
        self.assertNotEqual(groups[0]["source_id"], first.source_id)
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        self.assertTrue(any(
            row.get("source_id") == first.source_id
            and row.get("error_code") == "sam3_relabel_failed"
            and row.get("terminal")
            for row in failures
        ))
        self.assertEqual(calls, [([first.source_id], 1)])

    def test_render_keyboard_interrupt_restarts_incomplete_group_without_duplicates(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")

        class InterruptingRenderer(FakeRenderer):
            def __init__(self):
                self.calls = 0

            def render(self, source, preset, mask=None):
                self.calls += 1
                if self.calls == 3:
                    raise KeyboardInterrupt("render interrupted")
                return super().render(source, preset, mask)

        interrupted = InterruptingRenderer()
        dependencies = self.dependencies(inventory)
        dependencies.renderer_factory = lambda _config: interrupted
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(KeyboardInterrupt, "render interrupted"):
                run(config, dependencies=dependencies)

        manifest = json.loads((config.output_root / "manifest.json").read_text())
        self.assertEqual(manifest["phase"], "rendering")
        self.assertEqual(manifest["status"], "running")
        self.assertEqual(len(scan_jsonl(config.output_root / "groups.jsonl").records), 0)

        dependencies.renderer_factory = lambda _config: FakeRenderer()
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(groups), 1)
        candidate_ids = [row["candidate_id"] for row in groups[0]["candidates"]]
        self.assertEqual(len(candidate_ids), 8)
        self.assertEqual(len(set(candidate_ids)), 8)

    def test_crash_after_durable_group_before_coverage_commit_resumes_cleanly(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")
        dependencies = self.dependencies(inventory)

        with mock.patch(
            "construct.presets.GroupReservation.commit",
            side_effect=KeyboardInterrupt("coverage commit interrupted"),
        ), mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(KeyboardInterrupt, "coverage commit interrupted"):
                run(config, dependencies=dependencies)

        durable_groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(durable_groups), 1)
        self.assertEqual(
            json.loads((config.output_root / "manifest.json").read_text())["status"],
            "running",
        )

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(groups, durable_groups)
        selector_usage = resumed["presets"]["usage"]["selector"]["global"]
        self.assertEqual(sum(selector_usage["major"].values()), 1)
        self.assertEqual(sum(selector_usage["preset"].values()), 8)

    def test_sam3_batch_interrupt_retries_only_unfinished_source(self):
        config = self.config(target=2, local=1.0, global_=0.0, relabel_attempts=2)
        sources = tuple(self.source(index, border_mask=True) for index in range(4))
        inventory = SourceInventoryResult(
            sources, {"cache_entries": 4, "eligible": 4}, "ok"
        )
        calls = []

        def make_valid(source):
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[9:24, 16:33] = 255
            Image.fromarray(mask, "L").save(source.subject_path)

        class PartialStatuses:
            def __init__(self, completed_id):
                self.completed_id = completed_id

            def get(self, source_id, default=None):
                if source_id == self.completed_id:
                    return "ready"
                raise KeyboardInterrupt("sam3 batch interrupted")

        def relabeler(batch, _config, attempt):
            calls.append(([source.source_id for source in batch], attempt))
            if len(calls) == 1:
                make_valid(batch[0])
                return PartialStatuses(batch[0].source_id)
            for source in batch:
                make_valid(source)
            return {source.source_id: "ready" for source in batch}

        dependencies = self.dependencies(inventory, relabeler=relabeler)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(KeyboardInterrupt, "sam3 batch interrupted"):
                run(config, dependencies=dependencies)
        partial_groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(partial_groups), 1)
        self.assertEqual(len(calls[0][0]), 2)

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(scan_jsonl(config.output_root / "groups.jsonl").records), 2)
        self.assertEqual(calls[1], ([calls[0][0][1]], 1))
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        queued = [row for row in failures if row.get("error_code") == "sam3_relabel_queued"]
        self.assertEqual(len(queued), 2)
        self.assertEqual(len({row["event_id"] for row in queued}), 2)

    def test_annotation_interrupt_resumes_pending_stable_task_ids_only(self):
        config = self.config(target=2)
        sources = tuple(self.source(index) for index in range(4))
        inventory = SourceInventoryResult(
            sources, {"cache_entries": 4, "eligible": 4}, "ok"
        )
        interrupted_task_ids = []
        resumed_task_ids = []

        class InterruptingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                task = self.tasks(only)[0]
                interrupted_task_ids.append(task["task_id"])
                self.append_task(task)
                self.store.checkpoint()
                raise KeyboardInterrupt("annotation interrupted")

        class RecordingAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                resumed_task_ids.extend(
                    task["task_id"] for task in self.tasks(only)
                )
                return super().drain(max_workers=max_workers, only=only)

        dependencies = self.dependencies(inventory)
        dependencies.annotator_factory = (
            lambda _config, store: InterruptingAnnotator(store)
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(KeyboardInterrupt, "annotation interrupted"):
                run(config, dependencies=dependencies)
        first_rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len(first_rows), 1)
        self.assertEqual(first_rows[0]["annotation_task_id"], interrupted_task_ids[0])
        manifest = json.loads((config.output_root / "manifest.json").read_text())
        # "rendering", not "annotation": the annotator is now driven from the
        # render pass boundaries, so the interrupt is raised on the main thread
        # while the renderer still owns the build.  The closing drain's own
        # interrupt still leaves "annotation" — see
        # test_annotation_interleave.AnnotationInterleaveTests.
        self.assertEqual(manifest["phase"], "rendering")

        dependencies.annotator_factory = lambda _config, store: RecordingAnnotator(store)
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        task_ids = [row["annotation_task_id"] for row in rows]
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(rows), 4)
        self.assertEqual(len(set(task_ids)), 4)
        self.assertNotIn(interrupted_task_ids[0], resumed_task_ids)
        self.assertEqual(len(resumed_task_ids), 3)

    def test_projection_interrupt_retries_without_mutating_authoritative_jsonl(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")
        calls = []

        def projector(store, _manifest, _config):
            calls.append({
                "groups": tuple(sorted(store.groups)),
                "sft": tuple(sorted(store.sft)),
                "failures": tuple(row["event_id"] for row in store.failures),
            })
            if len(calls) == 1:
                raise KeyboardInterrupt("projection interrupted")
            return ProjectionResult(
                True, {"groups": len(store.groups), "sft": len(store.sft)}
            )

        dependencies = self.dependencies(inventory)
        dependencies.projector = projector
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(KeyboardInterrupt, "projection interrupted"):
                run(config, dependencies=dependencies)
        manifest = json.loads((config.output_root / "manifest.json").read_text())
        self.assertEqual(manifest["phase"], "projection")
        authoritative = {
            name: (config.output_root / name).read_bytes()
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=dependencies)
        self.assertEqual(resumed["status"], "complete")
        self.assertTrue(resumed["projection"]["ok"])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(authoritative, {
            name: (config.output_root / name).read_bytes()
            for name in authoritative
        })

    def test_preflight_failure_does_not_create_output_artifacts(self):
        config = self.config(target=1, local=0.0, global_=1.0)
        source = self.source(0)
        inventory = SourceInventoryResult((source,), {"eligible": 1}, "ok")
        dependencies = self.dependencies(inventory)
        dependencies.sdk_preflight = lambda: (_ for _ in ()).throw(RuntimeError("bad SDK"))
        with self.assertRaisesRegex(RuntimeError, "bad SDK"):
            run(config, dependencies=dependencies)
        self.assertFalse(config.output_root.exists())


class ProjectionTests(OrchestrationFixture):
    @staticmethod
    def group():
        candidates = [
            {
                "candidate_id": f"candidate-{index}",
                "slot_id": f"slot-{index}",
                "preset_id": f"preset-{index}",
                "format": "lut",
                "major": "major",
                "minor": "minor",
                "after_path": f"/after/{index}.jpg",
                "render_engine": "gpu_lut",
                "qa": {"q": 0.8},
                "rank": index,
            }
            for index in range(8)
        ]
        return {
            "build_id": "build",
            "group_id": "group-1",
            "source_id": "source-1",
            "source_path": "/before.jpg",
            "render_mode": "global",
            "preset_filter": "all",
            "major": "major",
            "scene": "portrait",
            "winner_ids": ["candidate-0"],
            "candidates": candidates,
        }

    def test_projection_uses_new_idempotent_tables(self):
        statements = []

        class Cursor:
            def execute(self, sql, params=None):
                statements.append((sql, params))

            def executemany(self, sql, rows):
                statements.append((sql, list(rows)))

            def close(self):
                return None

        class Connection:
            def cursor(self):
                return Cursor()

            def commit(self):
                self.committed = True

            def rollback(self):
                return None

            def close(self):
                return None

        with ArtifactStore(self.root / "project", "build", fsync_every=1) as store:
            group = self.group()
            store.append_group(group)
            store.append_sft({
                "build_id": "build", "sft_id": "sft-1", "annotation_task_id": "task-1",
                "group_id": "group-1", "candidate_id": "candidate-0", "winner_rank": 1,
                "I_in": "/before.jpg", "I_tar": "/after/0.jpg", "task_type": "style",
                "annot_src": "responses:test", "instruction": "instruction long",
                "instruction_short": "short", "reasoning": "reasoning", "qa": {},
            })
            store.append_failure({
                "build_id": "build", "event_id": "event-1", "event_type": "attempt",
                "stage": "rendering", "retryable": True, "error_code": "retry",
                "message": "retry", "terminal": False,
                "source_id": "", "group_id": "", "candidate_id": "",
            })
            result = project_artifacts(
                store,
                {"schema_version": 1, "build_id": "build", "phase": "complete",
                 "status": "complete"},
                "postgresql://u:p@db/viewer",
                connect_fn=lambda _dsn: Connection(),
            )
        self.assertTrue(result.ok)
        sql = "\n".join(statement for statement, _params in statements)
        for table in (
            "canonical_builds", "canonical_groups", "canonical_candidates",
            "canonical_sft", "canonical_failures",
        ):
            self.assertIn(table, sql)
        self.assertIn("ON CONFLICT", sql)
        for table in (
            "canonical_groups", "canonical_candidates", "canonical_sft",
            "canonical_failures",
        ):
            self.assertIn(f"DELETE FROM {table}", sql)
        self.assertNotIn("construct_dpo", sql)
        failure_rows = next(
            rows for statement, rows in statements
            if "INSERT INTO canonical_failures" in statement
        )
        self.assertEqual((None, None, None), failure_rows[0][5:8])
        self.assertEqual("", json.loads(failure_rows[0][15])["candidate_id"])

    def test_projection_outage_is_non_mutating_and_redacted(self):
        root = self.root / "outage"
        dsn = "postgresql://alice:s3cr3t@db/viewer?token=abc"
        with ArtifactStore(root, "build", fsync_every=1) as store:
            store.append_group(self.group())
            store.checkpoint()
            before = store.groups_path.read_bytes()

            def unavailable(_dsn):
                raise ConnectionError("database password s3cr3t token abc unavailable")

            result = project_artifacts(
                store,
                {"schema_version": 1, "build_id": "build", "phase": "projection",
                 "status": "running"},
                dsn,
                connect_fn=unavailable,
            )
            after = store.groups_path.read_bytes()
        self.assertFalse(result.ok)
        self.assertEqual(before, after)
        self.assertNotIn("s3cr3t", result.error)
        self.assertNotIn("abc", result.error)


if __name__ == "__main__":
    unittest.main()
