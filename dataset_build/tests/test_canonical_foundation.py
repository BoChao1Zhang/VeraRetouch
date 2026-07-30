from __future__ import annotations

import copy
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from PIL import Image

from dataset_build.lut_io import load_lut
from dataset_build.source_qa.iaa import (
    OneAlignRunner,
    _install_qalign_rope_compat,
    _prepare_qalign_config_compat,
    _prepare_qalign_model_compat,
    _prepare_qalign_transformers_compat,
    _redirect_qalign_assets,
)

from construct.canonical_masks import (
    MODE_COUNTS,
    build_mask_plan,
    linear_strength,
    pair_mask_slots,
)
from construct.canonical_qa import OneAlignScorer, QaError, deterministic_veto, rank_candidates
from construct.config import ConfigError, load_config, redact_text, redact_uri
from construct.presets import (
    CoverageSelector,
    PresetCatalog,
    PresetError,
    PresetRecord,
    TaxonomyLink,
    default_capability,
    packed_lut_paths,
)
from construct.sources import SourceRecord, allocate_sources, largest_remainder
from construct.state import ArtifactStore, StateError, scan_jsonl, stable_id
from construct.rendering import (
    GpuRenderError,
    LocalGpuOnlyRenderer,
    PreparedSource,
    apply_lut_cpu_oracle,
    composite_srgb,
    preprocess_source,
)
from dataset_build.source_qa import db as source_db
from construct.visibility import (
    ciede2000,
    objective_edit_hints,
    srgb_to_lab,
    visibility_metrics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "databuild.example.toml"


class ConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "databuild.toml"
        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def load(self):
        return load_config(self.path, validate_paths=False)

    def replace(self, old: str, new: str) -> None:
        text = self.path.read_text(encoding="utf-8")
        if old not in text:
            self.fail(f"databuild.example.toml no longer contains {old!r}")
        self.path.write_text(text.replace(old, new), encoding="utf-8")
        self.path.chmod(0o600)

    def config_line(self, key: str) -> str:
        prefix = f"{key} = "
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.startswith(prefix):
                return line
        self.fail(f"databuild.example.toml has no top-level {key!r} assignment")

    def test_example_parses_and_records_effective_formats(self) -> None:
        config = self.load()
        self.assertEqual(config.schema_version, 1)
        self.assertTrue(config.annotation.local_fallback)
        self.assertEqual(config.effective_formats, {"xmp", "lrtemplate", "lut"})
        self.assertEqual(len(config.annotation.external_endpoints), 5)
        safe = config.sanitized_dict()
        self.assertTrue(all(
            endpoint["api_key"] == "<redacted>"
            for endpoint in safe["annotation"]["external_endpoints"]
        ))
        self.assertNotIn("PASSWORD", json.dumps(safe))

    def test_private_permission_is_required(self) -> None:
        self.path.chmod(0o644)
        with self.assertRaisesRegex(ConfigError, "0600"):
            self.load()
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o644)

    def test_unknown_missing_ratio_and_endpoint_count_fail(self) -> None:
        self.replace("seed = 0", "seed = 0\nlegacy_mode = true")
        with self.assertRaisesRegex(ConfigError, "unknown field"):
            self.load()

        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)
        self.replace("global = 0.30", "global = 0.31")
        with self.assertRaisesRegex(ConfigError, "sum to one"):
            self.load()

        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)
        text = self.path.read_text(encoding="utf-8")
        start = text.index("\n[[annotation.external_endpoints]]")
        end = text.index("\n[annotation.local]")
        self.path.write_text(text[:start] + text[end:], encoding="utf-8")
        self.path.chmod(0o600)
        with self.assertRaises(ConfigError):
            self.load()

    def test_external_endpoint_lane_ids_are_distinct(self) -> None:
        self.replace('id = "provider-b-lane-2"', 'id = "provider-b-lane-1"')
        with self.assertRaisesRegex(ConfigError, "IDs must be distinct"):
            self.load()

    def test_disabled_format_control_is_strict(self) -> None:
        self.replace('preset_filter = "all"', 'preset_filter = "xmp"')
        self.replace("disabled_formats = []", 'disabled_formats = ["xmp"]')
        with self.assertRaisesRegex(ConfigError, "is disabled"):
            self.load()

        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)
        self.replace("disabled_formats = []", 'disabled_formats = ["lut", "lut"]')
        with self.assertRaisesRegex(ConfigError, "duplicates"):
            self.load()

    def test_canonical_render_geometry_and_encoding_are_fixed(self) -> None:
        self.replace("short_edge = 1024", "short_edge = 512")
        with self.assertRaisesRegex(ConfigError, "short edge 1024"):
            self.load()

        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)
        self.replace("jpeg_quality = 95", "jpeg_quality = 94")
        with self.assertRaisesRegex(ConfigError, "JPEG quality 95"):
            self.load()

    def test_invalid_cli_config_fails_before_output_mutation(self) -> None:
        from construct.agent import main

        output_root = Path(self.tmp.name) / "must-not-exist"
        self.replace(
            self.config_line("output_root"),
            f'output_root = "{output_root}"',
        )
        self.replace("seed = 0", "seed = 0\nlegacy_mode = true")
        with mock.patch("sys.stderr"):
            result = main(["run", "--config", str(self.path)])
        self.assertEqual(result, 1)
        self.assertFalse(output_root.exists())

    def test_redaction_covers_uri_userinfo_query_and_assignments(self) -> None:
        dsn = "postgresql://alice:s3cr3t@db.example:5432/vera?sslmode=require&token=abc"
        redacted = redact_uri(dsn)
        self.assertNotIn("alice", redacted)
        self.assertNotIn("s3cr3t", redacted)
        self.assertNotIn("abc", redacted)
        self.assertIn("sslmode=require", redacted)
        message = f"connect failed: {dsn} api_key=sk-live password=hunter2"
        safe = redact_text(message, ("sk-live",))
        for secret in ("alice", "s3cr3t", "abc", "sk-live", "hunter2"):
            self.assertNotIn(secret, safe)

    def test_endpoint_url_credentials_are_redacted_from_effective_config(self) -> None:
        self.replace(
            "https://provider-a.example/v1",
            "https://relay-user:relay-pass@provider-a.example/v1?token=relay-query",
        )
        self.replace(
            "http://127.0.0.1:8003/v1",
            "http://local-user:local-pass@127.0.0.1:8003/v1?api_key=local-query",
        )
        config = self.load()
        serialized = json.dumps(config.sanitized_dict(), sort_keys=True)
        for secret in (
            "relay-user", "relay-pass", "relay-query",
            "local-user", "local-pass", "local-query",
        ):
            self.assertNotIn(secret, serialized)
            self.assertIn(secret, config.secrets)

    def test_explicit_legacy_import_config_is_strict_and_sanitized(self) -> None:
        self.replace("target_groups = 10000", "target_groups = 1")
        self.replace('preset_filter = "all"', 'preset_filter = "xmp"')
        self.replace("local = 0.70", "local = 1.0")
        self.replace("global = 0.30", "global = 0.0")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(
                "\n[legacy_import]\n"
                'input_root = "/data/r5_local_50k"\n'
                'groups_jsonl = "/data/r5_local_50k/groups.jsonl"\n'
                'sft_jsonl = "/data/r5_local_50k/sft.jsonl"\n'
                "expected_source_groups = 22307\n"
                "expected_sft_rows = 33024\n"
                'protocol = "r5_local_50k_v4"\n'
                'axis_fix_status = "param_only_no_native_lut"\n'
                "project_to_viewer = false\n"
            )
        config = self.load()
        self.assertEqual(config.legacy_import.expected_sft_rows, 33024)
        self.assertFalse(config.legacy_import.project_to_viewer)
        safe = config.sanitized_dict()["legacy_import"]
        self.assertEqual(safe["groups_jsonl"], "/data/r5_local_50k/groups.jsonl")

        self.replace(
            'protocol = "r5_local_50k_v4"',
            'protocol = "unreviewed_legacy"',
        )
        with self.assertRaisesRegex(ConfigError, "protocol"):
            self.load()


class StateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def group(build_id: str = "b1") -> dict:
        group_id = stable_id("group", build_id, "source-1", "global", 0)
        candidates = [
            {
                "candidate_id": stable_id("candidate", group_id, index),
                "preset_id": f"p{index}",
            }
            for index in range(8)
        ]
        return {
            "build_id": build_id,
            "group_id": group_id,
            "source_id": "source-1",
            "candidates": candidates,
            "winner_ids": [candidates[0]["candidate_id"]],
        }

    def test_torn_tail_is_repaired_and_resume_is_idempotent(self) -> None:
        path = self.root / "groups.jsonl"
        valid = json.dumps(self.group(), separators=(",", ":"))
        path.write_bytes((valid + "\n{\"group_id\":").encode())
        self.assertTrue(scan_jsonl(path).torn_tail)
        with ArtifactStore(self.root, "b1", fsync_every=1) as store:
            self.assertEqual(len(store.groups), 1)
            self.assertFalse(store.append_group(self.group()))
        self.assertFalse(scan_jsonl(path).torn_tail)
        self.assertEqual(len(scan_jsonl(path).records), 1)

    def test_partial_or_duplicate_candidate_groups_are_rejected(self) -> None:
        with ArtifactStore(self.root, "b1") as store:
            partial = self.group()
            partial["candidates"] = partial["candidates"][:7]
            with self.assertRaisesRegex(StateError, "eight-candidate"):
                store.append_group(partial)
            duplicate = self.group()
            duplicate["candidates"][7]["candidate_id"] = duplicate["candidates"][0][
                "candidate_id"
            ]
            with self.assertRaisesRegex(StateError, "distinct"):
                store.append_group(duplicate)

    def test_group_winners_and_record_ids_are_strict_before_append(self) -> None:
        with ArtifactStore(self.root, "b1") as store:
            for winner_ids in (
                "candidate", ["missing"], ["c", "c"], ["a", "b", "c"], [1],
            ):
                with self.subTest(winner_ids=winner_ids):
                    group = self.group()
                    group["winner_ids"] = winner_ids
                    with self.assertRaisesRegex(StateError, "winner_ids"):
                        store.append_group(group)

            group = self.group()
            group["group_id"] = 123
            with self.assertRaisesRegex(StateError, "group_id"):
                store.append_group(group)

            group = self.group()
            group["candidates"][0]["candidate_id"] = 123
            with self.assertRaisesRegex(StateError, "string IDs"):
                store.append_group(group)

            with self.assertRaisesRegex(StateError, "sft_id"):
                store.append_sft({"build_id": "b1", "sft_id": 123})
            with self.assertRaisesRegex(StateError, "event_id"):
                store.append_failure({"build_id": "b1", "event_id": 123})

    def test_failure_event_duplicate_is_idempotent_but_conflict_is_rejected(self) -> None:
        failure = {
            "build_id": "b1",
            "event_id": stable_id("failure", "task-1", 1),
            "task_id": "task-1",
            "stage": "annotation",
            "terminal": True,
            "error_code": "schema_failed",
        }
        with ArtifactStore(self.root, "b1", fsync_every=1) as store:
            self.assertTrue(store.append_failure(failure))
            self.assertFalse(store.append_failure(dict(failure)))
            conflicting = {**failure, "error_code": "transport_failed"}
            with self.assertRaisesRegex(StateError, "conflicting durable failure"):
                store.append_failure(conflicting)
            self.assertEqual(store.failures, [failure])
        self.assertEqual(scan_jsonl(self.root / "failures.jsonl").records, (failure,))

    def test_appends_snapshot_caller_owned_records(self) -> None:
        group = self.group()
        sft = {
            "build_id": "b1",
            "sft_id": stable_id("sft", "task-1"),
            "annotation_task_id": "task-1",
            "qa": {"status": "completed"},
        }
        failure = {
            "build_id": "b1",
            "event_id": stable_id("failure", "task-1", 1),
            "task_id": "task-1",
            "details": {"attempt": 1},
        }
        expected_group = copy.deepcopy(group)
        expected_sft = copy.deepcopy(sft)
        expected_failure = copy.deepcopy(failure)

        with ArtifactStore(self.root, "b1", fsync_every=1) as store:
            store.append_group(group)
            store.append_sft(sft)
            store.append_failure(failure)
            group["candidates"][0]["preset_id"] = "caller-mutated"
            sft["qa"]["status"] = "caller-mutated"
            failure["details"]["attempt"] = 99
            self.assertEqual(expected_group, store.groups[expected_group["group_id"]])
            self.assertEqual(expected_sft, store.sft[expected_sft["sft_id"]])
            self.assertEqual([expected_failure], store.failures)

        self.assertEqual((expected_group,), scan_jsonl(self.root / "groups.jsonl").records)
        self.assertEqual((expected_sft,), scan_jsonl(self.root / "sft.jsonl").records)
        self.assertEqual((expected_failure,), scan_jsonl(self.root / "failures.jsonl").records)

    def test_manifest_writes_are_serialized(self) -> None:
        first_inside = threading.Event()
        release_first = threading.Event()
        second_inside = threading.Event()
        calls: list[str] = []
        calls_lock = threading.Lock()

        def blocked_write(_path, payload) -> None:
            with calls_lock:
                call_index = len(calls)
                calls.append(payload["phase"])
            if call_index == 0:
                first_inside.set()
                if not release_first.wait(2):
                    raise TimeoutError("first manifest write was not released")
            else:
                second_inside.set()

        with ArtifactStore(self.root, "b1") as store, mock.patch(
            "construct.state.write_json_atomic", side_effect=blocked_write
        ):
            with ThreadPoolExecutor(max_workers=2) as executor:
                first = executor.submit(store.write_manifest, {
                    "build_id": "b1", "phase": "preflight", "status": "running",
                })
                self.assertTrue(first_inside.wait(1))
                second = executor.submit(store.write_manifest, {
                    "build_id": "b1", "phase": "rendering", "status": "running",
                })
                try:
                    self.assertFalse(second_inside.wait(0.1))
                finally:
                    release_first.set()
                first.result(timeout=1)
                second.result(timeout=1)
        self.assertTrue(second_inside.is_set())
        self.assertEqual(["preflight", "rendering"], calls)

    def test_closed_store_rejects_all_durable_writes_and_checkpoints(self) -> None:
        store = ArtifactStore(self.root, "b1")
        manifest = {"build_id": "b1", "phase": "preflight", "status": "running"}
        store.write_manifest(manifest)
        manifest_before_close = (self.root / "manifest.json").read_bytes()
        store.close()

        operations = (
            lambda: store.append_group(self.group()),
            lambda: store.append_sft({"build_id": "b1", "sft_id": "sft-1"}),
            lambda: store.append_failure({"build_id": "b1", "event_id": "failure-1"}),
            lambda: store.write_manifest({
                "build_id": "b1", "phase": "rendering", "status": "running",
            }),
            store.checkpoint,
        )
        for operation in operations:
            with self.subTest(operation=operation), self.assertRaisesRegex(
                StateError, "artifact store is closed"
            ):
                operation()
        self.assertEqual(manifest_before_close, (self.root / "manifest.json").read_bytes())
        store.close()

        with ArtifactStore(self.root, "b1") as reopened:
            reopened.checkpoint()

    def test_output_root_lock_is_nonblocking_and_released_on_close(self) -> None:
        contender = """
import sys
from construct.state import ArtifactStore, StateError

try:
    store = ArtifactStore(sys.argv[1], "b1")
except StateError as exc:
    if "locked by another databuild process" not in str(exc):
        raise
    raise SystemExit(0)
else:
    store.close()
    raise SystemExit(2)
"""
        store = ArtifactStore(self.root, "b1")
        try:
            result = subprocess.run(
                [sys.executable, "-c", contender, str(self.root)],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(any(path.name.endswith(".lock") for path in self.root.iterdir()))
        finally:
            store.close()

        with ArtifactStore(self.root, "b1") as reopened:
            self.assertEqual(reopened.groups, {})

    def test_resume_rejects_invalid_durable_winner_reference(self) -> None:
        group = self.group()
        group["winner_ids"] = ["missing"]
        path = self.root / "groups.jsonl"
        path.write_text(json.dumps(group) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(StateError, "reference group candidates"):
            ArtifactStore(self.root, "b1")

    def test_resume_rejects_cross_build_and_duplicate_durable_records(self) -> None:
        path = self.root / "groups.jsonl"
        path.write_text(json.dumps(self.group("foreign")) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(StateError, "another build"):
            ArtifactStore(self.root, "b1")

        first = self.group("b1")
        path.write_text(
            json.dumps(first) + "\n" + json.dumps(first) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(StateError, "duplicate durable group"):
            ArtifactStore(self.root, "b1")

    def test_annotation_queue_reconstructs_from_groups_and_sft(self) -> None:
        group = self.group()
        with ArtifactStore(self.root, "b1", fsync_every=1) as store:
            store.append_group(group)
            pending = store.pending_annotation_tasks()
            self.assertEqual(len(pending), 1)
            task = pending[0]
            store.append_sft(
                {
                    "build_id": "b1",
                    "sft_id": stable_id("sft", task["task_id"]),
                    "annotation_task_id": task["task_id"],
                }
            )
        with ArtifactStore(self.root, "b1") as resumed:
            self.assertEqual(resumed.pending_annotation_tasks(), [])

    def test_annotation_queue_uses_one_based_winner_ranks(self) -> None:
        group = self.group()
        group["winner_ids"] = [
            group["candidates"][1]["candidate_id"],
            group["candidates"][0]["candidate_id"],
        ]
        with ArtifactStore(self.root, "b1") as store:
            store.append_group(group)
            tasks = store.pending_annotation_tasks()
        self.assertEqual([task["winner_rank"] for task in tasks], [1, 2])
        self.assertEqual(
            [task["task_id"] for task in tasks],
            [
                stable_id("annotation", group["group_id"], candidate_id, rank)
                for rank, candidate_id in enumerate(group["winner_ids"], start=1)
            ],
        )


class SourceAllocationTests(unittest.TestCase):
    def records(self, n: int = 40) -> list[SourceRecord]:
        scenes = ["portrait", "landscape", "food", "unknown"]
        return [
            SourceRecord(
                source_id=f"s{index:03d}",
                source_path=Path(f"/source/{index}.jpg"),
                cache_dir=Path(f"/cache/{index}"),
                subject_path=Path(f"/cache/{index}/subject.png"),
                subject_meta_path=Path(f"/cache/{index}/subject.json"),
                scene=scenes[index % len(scenes)],
                subject={"name": "subject"},
                mask_area=0.2,
            )
            for index in range(n)
        ]

    def test_largest_remainder_is_exact(self) -> None:
        self.assertEqual(largest_remainder({"local": 0.7, "global": 0.3}, 10),
                         {"local": 7, "global": 3})
        self.assertEqual(sum(largest_remainder({"a": 1, "b": 1, "c": 1}, 8).values()), 8)

    def test_allocation_is_deterministic_and_reserves_replacements(self) -> None:
        from construct.config import MixConfig

        kwargs = dict(build_id="build", seed=9, target_groups=10,
                      mix=MixConfig(local=0.7, global_=0.3))
        first = allocate_sources(self.records(), **kwargs)
        second = allocate_sources(reversed(self.records()), **kwargs)
        self.assertEqual(first.local_target, 7)
        self.assertEqual(first.global_target, 3)
        self.assertEqual([row.source_id for row in first.local],
                         [row.source_id for row in second.local])
        self.assertEqual([row.source_id for row in first.global_],
                         [row.source_id for row in second.global_])
        all_ids = [row.source_id for row in first.local + first.global_]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertGreater(len(first.local), first.local_target)
        self.assertGreater(len(first.global_), first.global_target)

    def test_duplicate_source_ids_are_rejected(self) -> None:
        from construct.config import MixConfig

        source = self.records(1)[0]
        with self.assertRaisesRegex(ValueError, "duplicate source_id"):
            allocate_sources(
                [source, source], build_id="build", seed=9, target_groups=1,
                mix=MixConfig(local=1.0, global_=0.0),
            )


class CoverageTests(unittest.TestCase):
    @staticmethod
    def catalog() -> PresetCatalog:
        links = []
        for major in ("major-a", "major-b"):
            for minor_index in range(4):
                minor = f"minor-{minor_index}"
                for preset_index in range(4):
                    preset_id = f"{major}-p-{minor_index}-{preset_index}"
                    preset = PresetRecord(
                        preset_id=preset_id,
                        path=Path(f"/{preset_id}.cube"),
                        format="lut",
                        kind="lut",
                        style_name="Style",
                        fidelity_de=None,
                        render_engine="gpu_lut",
                    )
                    links.append(TaxonomyLink(preset, major, minor))
            # Deliberately expose one ID through two minors to exercise group-wide dedupe.
            links.append(TaxonomyLink(links[-16].preset, major, "minor-1"))
        return PresetCatalog.from_links(links)

    @staticmethod
    def fill(reservation, reject_first: bool = False):
        accepted = []
        first_minor = None
        replacement_minor = None
        for index in range(8):
            slot = f"slot-{index}"
            candidate = reservation.reserve_candidate(slot)
            if candidate is None:
                raise AssertionError("selector exhausted unexpectedly")
            if index == 0 and reject_first:
                first_minor = candidate.link.minor
                reservation.reject(candidate)
                candidate = reservation.reserve_candidate(slot)
                if candidate is None:
                    raise AssertionError("selector failed to refill rejected candidate")
                replacement_minor = candidate.link.minor
            reservation.accept(candidate)
            accepted.append(candidate)
        return accepted, first_minor, replacement_minor

    def selector(self):
        return CoverageSelector(
            self.catalog(), build_id="build", seed=3,
            render_mode="global", preset_filter="all",
        )

    @staticmethod
    def mixed_catalog() -> PresetCatalog:
        links = []
        engines = {
            "xmp": "gpu_local_preset",
            "lrtemplate": "gpu_local_preset",
            "lut": "gpu_lut",
        }
        suffixes = {"xmp": ".xmp", "lrtemplate": ".lrtemplate", "lut": ".cube"}
        for preset_format in ("xmp", "lrtemplate", "lut"):
            for index in range(8):
                preset_id = f"{preset_format}-{index}"
                preset = PresetRecord(
                    preset_id=preset_id,
                    path=Path(f"/{preset_id}{suffixes[preset_format]}"),
                    format=preset_format,
                    kind=preset_format,
                    style_name="Style",
                    fidelity_de=None,
                    render_engine=engines[preset_format],
                )
                links.append(TaxonomyLink(
                    preset, "mixed-major", f"minor-{index % 2}"
                ))
        return PresetCatalog.from_links(links)

    def test_fresh_selector_candidate_sequence_is_deterministic(self) -> None:
        def sequence():
            selector = self.selector()
            reservation = selector.begin_group("source", 0)
            accepted, _first_minor, _replacement_minor = self.fill(reservation)
            return (
                reservation.reservation_id,
                reservation.coverage_cycle,
                reservation.coverage_position,
                tuple(
                    (row.link.major, row.link.minor, row.link.preset.preset_id, row.attempt)
                    for row in accepted
                ),
            )

        self.assertEqual(sequence(), sequence())

    def test_explicit_format_filters_never_leak_and_all_keeps_full_inventory(self) -> None:
        catalog = self.mixed_catalog()
        for preset_filter in ("xmp", "lrtemplate", "lut"):
            with self.subTest(preset_filter=preset_filter):
                selector = CoverageSelector(
                    catalog,
                    build_id="build",
                    seed=5,
                    render_mode="global",
                    preset_filter=preset_filter,
                )
                self.assertEqual(
                    {link.preset.format for link in selector.catalog.links},
                    {preset_filter},
                )
                reservation = selector.begin_group("source", 0)
                accepted, _first_minor, _replacement_minor = self.fill(reservation)
                self.assertEqual(
                    {row.link.preset.format for row in accepted}, {preset_filter}
                )

        all_selector = CoverageSelector(
            catalog,
            build_id="build",
            seed=5,
            render_mode="global",
            preset_filter="all",
        )
        self.assertEqual(
            {link.preset.preset_id for link in all_selector.catalog.links},
            {link.preset.preset_id for link in catalog.links},
        )
        with self.assertRaisesRegex(PresetError, "invalid preset filter"):
            CoverageSelector(
                catalog,
                build_id="build",
                seed=5,
                render_mode="global",
                preset_filter="auto",
            )

    def test_group_is_one_major_and_eight_distinct_even_with_overlap(self) -> None:
        selector = self.selector()
        reservation = selector.begin_group("source", 0)
        accepted, first_minor, replacement_minor = self.fill(reservation, reject_first=True)
        self.assertEqual(first_minor, replacement_minor)
        self.assertEqual({row.link.major for row in accepted}, {reservation.major})
        self.assertEqual(len({row.link.preset.preset_id for row in accepted}), 8)
        minor_counts = {}
        for row in accepted:
            minor_counts[row.link.minor] = minor_counts.get(row.link.minor, 0) + 1
        self.assertLessEqual(max(minor_counts.values()) - min(minor_counts.values()), 1)
        reservation.commit()
        snapshot = selector.snapshot()
        self.assertEqual(sum(snapshot["major"].values()), 1)
        self.assertEqual(sum(snapshot["preset"].values()), 8)

    def test_major_bag_balances_completed_cycles(self) -> None:
        selector = self.selector()
        positions = {}
        for index in range(8):
            reservation = selector.begin_group(f"source-{index}", 0)
            positions.setdefault(reservation.coverage_cycle, []).append(
                reservation.coverage_position
            )
            self.fill(reservation)
            reservation.commit()
        counts = selector.snapshot()["major"]
        self.assertEqual(set(counts), {"major-a", "major-b"})
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertTrue(all(sorted(cycle) == [0, 1] for cycle in positions.values()))

    def test_lut_capability_rejects_malformed_files(self) -> None:
        path = Path(self.catalog().links[0].preset.path)
        with tempfile.TemporaryDirectory() as tmp:
            malformed = Path(tmp) / path.name
            malformed.write_text("LUT_3D_SIZE 2\n0 0 0\n", encoding="utf-8")
            result = default_capability({"path": str(malformed), "kind": "lut"}, None, 6.0)
        self.assertFalse(result.supported)
        self.assertIn("LUT parse failed", result.reason)

    def test_abandoned_group_does_not_advance_successful_coverage(self) -> None:
        selector = self.selector()
        reservation = selector.begin_group("source", 0)
        candidate = reservation.reserve_candidate("slot-0")
        reservation.accept(candidate)
        reservation.abandon()
        snapshot = selector.snapshot()
        self.assertEqual(sum(snapshot["major"].values()), 0)
        self.assertEqual(sum(snapshot["preset"].values()), 0)


class PackedLutPreflightTests(unittest.TestCase):
    """capability preflight 用 luts.npz 的预解析结果代替逐个 load_lut。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.bank = self.root / "bank"
        self.bank.mkdir()
        self.recipes = self.root / "recipes"
        self.recipes.mkdir()
        rows = "\n".join(
            f"{r} {g} {b}"
            for b in (0.0, 1.0) for g in (0.0, 1.0) for r in (0.0, 1.0)
        )
        self.good = self.recipes / "good.cube"
        self.good.write_text(f"TITLE \"good\"\nLUT_3D_SIZE 2\n{rows}\n", encoding="utf-8")
        self.spare = self.recipes / "spare.cube"
        self.spare.write_text(f"TITLE \"spare\"\nLUT_3D_SIZE 2\n{rows}\n", encoding="utf-8")
        # 生产 bank 的 24 个真实样本：结构合法但 TITLE 里是 GBK 字节，
        # 现行 strict 解析拒收，而 npz 里却有它们的网格。
        self.gbk = self.recipes / "gbk.cube"
        self.gbk.write_bytes(
            "TITLE \"胶片\"\n".encode("gbk") + f"LUT_3D_SIZE 2\n{rows}\n".encode("utf-8")
        )
        self.broken = self.recipes / "broken.cube"
        self.broken.write_text("LUT_3D_SIZE 2\n0 0 0\n", encoding="utf-8")
        self.packed = (self.good, self.gbk)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_pack(self, *, npz: bool = True, meta: bool = True) -> None:
        if meta:
            (self.bank / "luts_meta.json").write_text(
                json.dumps({
                    f"rcp_{path.stem}": {
                        "path": os.path.realpath(path), "dmin": [0, 0, 0], "dmax": [1, 1, 1],
                    }
                    for path in self.packed
                }),
                encoding="utf-8",
            )
        if npz:
            np.savez(self.bank / "luts.npz", **{
                f"rcp_{path.stem}": np.zeros((2, 2, 2, 3), np.float32) for path in self.packed
            })

    @staticmethod
    def _counting_load_lut():
        real = load_lut
        calls: list[str] = []

        def spy(path):
            calls.append(str(path))
            return real(path)

        return spy, calls

    def test_packed_hit_answers_without_parsing_the_file(self) -> None:
        self._write_pack()
        spy, calls = self._counting_load_lut()
        with mock.patch("dataset_build.lut_io.load_lut", spy):
            result = default_capability(
                {"path": str(self.good), "kind": "lut"}, None, 6.0,
                packed_lut_paths(self.bank),
            )
        self.assertTrue(result.supported)
        self.assertEqual(result.engine, "gpu_lut")
        self.assertEqual(calls, [])

    def test_a_lut_the_pack_does_not_know_still_goes_through_the_parser(self) -> None:
        self._write_pack()
        spy, calls = self._counting_load_lut()
        with mock.patch("dataset_build.lut_io.load_lut", spy):
            result = default_capability(
                {"path": str(self.spare), "kind": "lut"}, None, 6.0,
                packed_lut_paths(self.bank),
            )
        self.assertTrue(result.supported)
        self.assertEqual(calls, [str(self.spare)])

    def test_bad_luts_are_still_rejected_with_their_original_reason(self) -> None:
        self._write_pack()
        packed = packed_lut_paths(self.bank)
        for path, packed_row, expected in (
            # 未进 npz 的坏 LUT：兜底解析照旧拒收。
            (self.broken, False, "LUT parse failed:ValueError"),
            # 进了 npz 但不是 UTF-8：npz 是 2026-07-18 用更宽松的解析器打的，
            # strict 解码这一关必须留着，否则生产 bank 的 24 个 .cube 会悄悄
            # 挤进 inventory（实测 3522→3546）。
            (self.gbk, True, "LUT parse failed:UnicodeDecodeError"),
        ):
            with self.subTest(path=path.name):
                self.assertEqual(os.path.realpath(path) in packed, packed_row)
                result = default_capability(
                    {"path": str(path), "kind": "lut"}, None, 6.0, packed
                )
                self.assertFalse(result.supported)
                self.assertEqual(result.reason, expected)

    def test_discovery_needs_both_meta_and_npz(self) -> None:
        self.assertEqual(packed_lut_paths(self.bank), frozenset())
        self._write_pack(npz=False)
        # 只有 meta：渲染期 _LutLoader 仍会逐个解析，preflight 不能比它更乐观。
        self.assertEqual(packed_lut_paths(self.bank), frozenset())
        self._write_pack()
        self.assertEqual(
            packed_lut_paths(self.bank),
            frozenset(os.path.realpath(path) for path in self.packed),
        )

    def _catalog_config(self) -> Path:
        (self.bank / "features.jsonl").write_text(
            "".join(
                json.dumps({
                    "preset_id": f"rcp_{path.stem}", "kind": "lut", "fmt": "cube",
                    "path": str(path), "style_name": path.stem,
                }) + "\n"
                for path in (self.good, self.spare)
            ),
            encoding="utf-8",
        )
        taxonomy = self.bank / "taxonomy.jsonl"
        taxonomy.write_text(
            "".join(
                json.dumps({"preset_id": f"rcp_{path.stem}", "major": "m", "minor": "n"}) + "\n"
                for path in (self.good, self.spare)
            ),
            encoding="utf-8",
        )
        path = self.root / "databuild.toml"
        text = EXAMPLE.read_text(encoding="utf-8")
        text = text.replace(
            'bank_dir = "/var/cache/veradata/preset_bank_full"', f'bank_dir = "{self.bank}"'
        ).replace(
            'taxonomy = "/var/cache/veradata/preset_bank_full/taxonomy.jsonl"',
            f'taxonomy = "{taxonomy}"',
        )
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
        return path

    def test_catalog_load_wires_the_pack_and_keeps_the_same_inventory(self) -> None:
        config = load_config(self._catalog_config(), validate_paths=False)
        spy, calls = self._counting_load_lut()
        with mock.patch("dataset_build.lut_io.load_lut", spy):
            plain = PresetCatalog.load(config)
        self.assertEqual(sorted(calls), [str(self.good), str(self.spare)])

        self._write_pack()
        spy, calls = self._counting_load_lut()
        with mock.patch("dataset_build.lut_io.load_lut", spy):
            packed = PresetCatalog.load(config)
        # good 在 npz 里（免解析），spare 不在（照旧解析）；两条路径同一份 inventory。
        self.assertEqual(calls, [str(self.spare)])
        self.assertEqual(
            [link.preset.preset_id for link in packed.links],
            [link.preset.preset_id for link in plain.links],
        )
        self.assertEqual(packed.rejected, plain.rejected)


class OneAlignCompatibilityTests(unittest.TestCase):
    def test_runner_does_not_silence_forward_failures(self) -> None:
        runner = object.__new__(OneAlignRunner)
        runner._score_pils = mock.Mock(side_effect=RuntimeError("forward failed"))
        with mock.patch(
            "dataset_build.source_qa.iaa._decode_rgb", return_value=object()
        ), self.assertRaisesRegex(RuntimeError, "forward failed"):
            runner.score_path("sample.jpg")

    def test_canonical_scorer_rejects_missing_and_invalid_scores(self) -> None:
        scorer = object.__new__(OneAlignScorer)
        for value in (None, float("nan"), -1.0, 101.0):
            scorer.runner = SimpleNamespace(
                score_path=lambda _path, value=value: {"onealign": value}
            )
            with self.assertRaises(QaError):
                scorer.score("sample.jpg")

    def test_recent_transformers_exports_qalign_legacy_symbols(self) -> None:
        llama = importlib.import_module("transformers.models.llama.modeling_llama")
        original = getattr(llama, "__all__", None)
        try:
            setattr(llama, "__all__", ["LlamaModel"])
            _prepare_qalign_transformers_compat(llama)
            self.assertIn("Cache", llama.__all__)
            self.assertIn("BaseModelOutputWithPast", llama.__all__)
        finally:
            if original is None:
                delattr(llama, "__all__")
            else:
                setattr(llama, "__all__", original)

    def test_qalign_rope_compat_preserves_legacy_shapes_and_positions(self) -> None:
        module = SimpleNamespace(
            rotate_half=lambda value: torch.cat(
                (-value[..., value.shape[-1] // 2:], value[..., :value.shape[-1] // 2]),
                dim=-1,
            )
        )
        _install_qalign_rope_compat(module, torch)
        rope = module.LlamaRotaryEmbedding(8, max_position_embeddings=16)
        value = torch.zeros((1, 2, 4, 8), dtype=torch.float32)
        cos, sin = rope(value, seq_len=4)
        self.assertEqual(tuple(cos.shape), (4, 8))
        positions = torch.arange(4).unsqueeze(0)
        query, key = module.apply_rotary_pos_emb(
            value, value, cos, sin, positions
        )
        self.assertEqual(tuple(query.shape), tuple(value.shape))
        self.assertEqual(tuple(key.shape), tuple(value.shape))

    def test_qalign_config_compat_adds_recent_llama_defaults(self) -> None:
        class LegacyConfig:
            pass

        _prepare_qalign_config_compat(LegacyConfig)
        self.assertFalse(LegacyConfig.mlp_bias)

    def test_qalign_model_compat_restores_attention_runtime_flags(self) -> None:
        for implementation, flash, sdpa in (
            ("eager", False, False),
            ("flash_attention_2", True, False),
            ("sdpa", False, True),
        ):
            model = SimpleNamespace(
                config=SimpleNamespace(_attn_implementation=implementation)
            )
            _prepare_qalign_model_compat(model)
            self.assertEqual(model._use_flash_attention_2, flash)
            self.assertEqual(model._use_sdpa, sdpa)

    def test_qalign_hub_assets_are_redirected_to_local_model(self) -> None:
        calls = []

        class Loader:
            @staticmethod
            def from_pretrained(path, *args, **kwargs):
                calls.append(path)
                return path

        module = SimpleNamespace(AutoTokenizer=Loader, CLIPImageProcessor=Loader)
        _redirect_qalign_assets(module, "/models/one-align")
        self.assertEqual(
            module.AutoTokenizer.from_pretrained("q-future/one-align"),
            "/models/one-align",
        )
        self.assertEqual(
            module.CLIPImageProcessor.from_pretrained("other-model"),
            "other-model",
        )
        self.assertEqual(calls, ["/models/one-align", "other-model"])


class MaskRenderingVisibilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        mask = np.zeros((64, 96), dtype=np.uint8)
        mask[20:44, 34:62] = 255
        self.mask_path = root / "subject.png"
        Image.fromarray(mask, "L").save(self.mask_path)
        self.source_path = root / "source.jpg"
        source = np.zeros((64, 96, 3), dtype=np.uint8)
        source[..., 0] = np.arange(96, dtype=np.uint8)[None]
        Image.fromarray(source, "RGB").save(self.source_path)
        self.source = SourceRecord(
            source_id="source",
            source_path=self.source_path,
            cache_dir=root,
            subject_path=self.mask_path,
            subject_meta_path=root / "subject.json",
            scene="portrait",
            subject={"name": "person"},
            mask_area=float((mask > 0).mean()),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_fixed_mask_plan_has_eight_slots_and_seven_physical_masks(self) -> None:
        plan = build_mask_plan(
            self.source, build_id="build", seed=4, width=96, height=64
        )
        counts = {}
        for slot in plan.slots:
            counts[slot.mode] = counts.get(slot.mode, 0) + 1
        self.assertEqual(counts, MODE_COUNTS)
        self.assertEqual(len(plan.slots), 8)
        self.assertEqual(len(plan.physical_masks), 7)
        semantic = [slot for slot in plan.slots if slot.mode == "semantic"]
        self.assertEqual(semantic[0].mask.mask_id, semantic[1].mask.mask_id)
        self.assertIs(semantic[0].mask, semantic[1].mask)
        for slot in plan.slots:
            self.assertEqual(slot.mask.effective_alpha.shape, (64, 96))
            self.assertGreater(slot.mask.effective_alpha_mean, 0)
            if slot.mode == "linear":
                self.assertGreaterEqual(slot.mask.amount, 0.5)
                self.assertLessEqual(slot.mask.effective_alpha_mean, 0.50001)
        first = pair_mask_slots(plan, build_id="build", seed=4, source_id="source")
        second = pair_mask_slots(plan, build_id="build", seed=4, source_id="source")
        self.assertEqual([slot.slot_id for slot in first], [slot.slot_id for slot in second])
        self.assertEqual([slot.pairing_index for slot in first], list(range(8)))

    def test_linear_mass_formula(self) -> None:
        raw = np.full((8, 8), 0.75, dtype=np.float32)
        amount, effective = linear_strength(raw)
        self.assertAlmostEqual(amount, 2.0 / 3.0, places=6)
        self.assertGreaterEqual(amount, 0.5)
        self.assertAlmostEqual(float(effective.mean()), 0.5, places=6)
        amount, effective = linear_strength(np.full((4, 4), 0.25, dtype=np.float32))
        self.assertEqual(amount, 1.0)
        self.assertAlmostEqual(float(effective.mean()), 0.25)

    def test_srgb_composite_endpoints_are_exact(self) -> None:
        before = np.zeros((2, 3, 3), dtype=np.float32)
        edited = np.ones_like(before)
        alpha = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, 1.0]], dtype=np.float32)
        result = composite_srgb(before, edited, alpha)
        self.assertTrue(np.array_equal(result[:, 0], before[:, 0]))
        self.assertTrue(np.array_equal(result[:, 2], edited[:, 2]))
        self.assertTrue(np.allclose(result[:, 1], 0.5))

    def test_common_preprocess_grid_applies_exif_and_exact_short_edge(self) -> None:
        prepared = preprocess_source(self.source_path, short_edge=32)
        self.assertEqual((prepared.width, prepared.height), (48, 32))
        self.assertEqual(prepared.pixels.shape, (32, 48, 3))

    def test_lut_identity_and_axis_sentinel(self) -> None:
        grid = np.empty((2, 2, 2, 3), dtype=np.float32)
        for b in range(2):
            for g in range(2):
                for r in range(2):
                    grid[b, g, r] = (r, g, b)
        image = np.array([[[0.0, 0.25, 1.0], [1.0, 0.75, 0.0]]], dtype=np.float32)
        self.assertTrue(np.allclose(apply_lut_cpu_oracle(image, grid), image, atol=1e-6))
        axis = grid[..., ::-1]
        swapped = apply_lut_cpu_oracle(np.array([[[1.0, 0.0, 0.0]]], np.float32), axis)
        self.assertTrue(np.allclose(swapped, [[[0.0, 0.0, 1.0]]]))

    def test_production_lut_interpolation_matches_cpu_oracle(self) -> None:
        import torch

        axis = np.linspace(0.0, 1.0, 3, dtype=np.float32)
        grid = np.empty((3, 3, 3, 3), dtype=np.float32)
        for blue_index, blue in enumerate(axis):
            for green_index, green in enumerate(axis):
                for red_index, red in enumerate(axis):
                    grid[blue_index, green_index, red_index] = (
                        red * red,
                        np.sqrt(green),
                        0.15 + 0.7 * blue,
                    )
        domain_min = np.array([0.1, 0.2, 0.0], dtype=np.float32)
        domain_max = np.array([0.9, 0.8, 1.0], dtype=np.float32)
        image = np.random.default_rng(7).uniform(-0.1, 1.1, (5, 7, 3)).astype(np.float32)
        renderer = object.__new__(LocalGpuOnlyRenderer)
        renderer._torch = torch
        renderer.device = "cpu"
        renderer._lut_loader = SimpleNamespace(
            load=lambda _path: (grid, domain_min, domain_max)
        )
        before = torch.from_numpy(image.transpose(2, 0, 1)[None])
        actual, diagnostics = renderer._apply_lut(
            before, SimpleNamespace(path=Path("/unused.cube"))
        )
        actual_array = actual[0].permute(1, 2, 0).numpy()
        expected = apply_lut_cpu_oracle(image, grid, domain_min, domain_max)
        np.testing.assert_allclose(actual_array, expected, rtol=0.0, atol=2e-6)
        self.assertEqual(diagnostics["axis_order"], "bgr")

    def test_cube_parser_preserves_red_fastest_bgr_grid(self) -> None:
        path = Path(self.tmp.name) / "axis.cube"
        rows = []
        for b in range(2):
            for g in range(2):
                for r in range(2):
                    rows.append(f"{r} {g} {b}")
        path.write_text(
            "LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n"
            + "\n".join(rows)
            + "\n",
            encoding="ascii",
        )
        grid, domain_min, domain_max = load_lut(path)
        self.assertEqual(grid.shape, (2, 2, 2, 3))
        self.assertTrue(np.array_equal(grid[1, 0, 0], [0, 0, 1]))
        self.assertTrue(np.array_equal(grid[0, 0, 1], [1, 0, 0]))
        image = np.array([[[0.2, 0.6, 0.9]]], dtype=np.float32)
        self.assertTrue(np.allclose(apply_lut_cpu_oracle(image, grid), image, atol=1e-6))
        self.assertTrue(np.array_equal(domain_min, np.zeros(3, dtype=np.float32)))
        self.assertTrue(np.array_equal(domain_max, np.ones(3, dtype=np.float32)))

    def test_ciede2000_reference_and_visibility_cases(self) -> None:
        first = np.array([[[50.0, 2.6772, -79.7751]]])
        second = np.array([[[50.0, 0.0, -82.7485]]])
        self.assertAlmostEqual(float(ciede2000(first, second)[0, 0]), 2.0425, places=4)

        before = np.full((20, 20, 3), 0.25, dtype=np.float32)
        no_op = visibility_metrics(
            before, before.copy(), weight=None, short_edge=20,
            visible_de_min=2.5, visible_fraction_de=2.3, visible_fraction_min=0.5,
        )
        self.assertFalse(no_op.accepted)
        self.assertEqual(no_op.visible_de, 0.0)

        alpha = np.zeros((20, 20), dtype=np.float32)
        alpha[5:15, 5:15] = 1.0
        localized = before.copy()
        localized[alpha == 1] = 0.8
        strong = visibility_metrics(
            before, localized, weight=alpha, short_edge=20,
            visible_de_min=2.5, visible_fraction_de=2.3, visible_fraction_min=0.5,
        )
        self.assertTrue(strong.accepted)

        weak = visibility_metrics(
            before, np.clip(before + 0.005, 0, 1), weight=None, short_edge=20,
            visible_de_min=2.5, visible_fraction_de=2.3, visible_fraction_min=0.5,
        )
        self.assertFalse(weak.accepted)

        broad_before = np.full((64, 96, 3), 0.5, dtype=np.float32)
        broad_gradient = np.clip(
            broad_before
            + np.linspace(-0.05, 0.05, 96, dtype=np.float32)[None, :, None],
            0,
            1,
        )
        broad = visibility_metrics(
            broad_before, broad_gradient, weight=None, short_edge=20,
            visible_de_min=2.5, visible_fraction_de=2.3, visible_fraction_min=0.5,
        )
        self.assertGreater(broad.visible_fraction, 0.5)
        self.assertLess(broad.visible_de, 2.5)
        self.assertFalse(broad.accepted)

        outlier = before.copy()
        outlier[:4] = 1.0
        sparse = visibility_metrics(
            before, outlier, weight=None, short_edge=20,
            visible_de_min=2.5, visible_fraction_de=2.3, visible_fraction_min=0.5,
        )
        self.assertGreater(sparse.visible_de, 2.5)
        self.assertLess(sparse.visible_fraction, 0.5)
        self.assertFalse(sparse.accepted)

    def test_objective_hints_use_cgt_weight_instead_of_rectangle_or_background(self) -> None:
        before = np.full((12, 12, 3), 0.5, dtype=np.float32)
        after = np.full_like(before, 0.1)
        weight = np.zeros((12, 12), dtype=np.float32)
        weight[4:8, 4:8] = 1.0
        after[weight == 1] = 0.8
        local = objective_edit_hints(before, after, weight=weight)
        global_ = objective_edit_hints(before, after, weight=None)
        self.assertEqual(local["brightness"]["direction"], "brighter")
        self.assertGreater(local["brightness"]["delta"], 0)
        self.assertEqual(global_["brightness"]["direction"], "darker")
        self.assertLess(global_["brightness"]["delta"], 0)

    def test_onealign_ranking_selects_at_most_two_and_vetoes_extremes(self) -> None:
        root = Path(self.tmp.name)
        source = root / "qa-source.jpg"
        Image.fromarray(np.full((32, 32, 3), 110, np.uint8), "RGB").save(source)
        candidates = []
        score_by_path = {str(source): 50.0}
        for index in range(8):
            path = root / f"after-{index}.jpg"
            value = 255 if index == 7 else 105 + index * 5
            Image.fromarray(np.full((32, 32, 3), value, np.uint8), "RGB").save(path)
            score_by_path[str(path)] = 90.0 - index * 5
            candidates.append({"candidate_id": f"c{index}", "after_path": str(path)})

        class FakeScorer:
            def score(self, path):
                return score_by_path[path]

        result = rank_candidates(str(source), candidates, FakeScorer())
        self.assertEqual(len(result.candidates), 8)
        self.assertLessEqual(len(result.winner_ids), 2)
        self.assertEqual(result.winner_ids, ("c0", "c1"))
        self.assertEqual(sorted(row["rank"] for row in result.candidates), list(range(1, 9)))
        by_id = {row["candidate_id"]: row for row in result.candidates}
        self.assertEqual([by_id[candidate_id]["rank"] for candidate_id in result.winner_ids], [1, 2])
        last = next(row for row in result.candidates if row["candidate_id"] == "c7")
        self.assertTrue(last["qa"]["veto"])
        veto, flags = deterministic_veto(
            {"highlight": 0, "shadow": 0, "luma": 100, "colorfulness": 210},
            {"highlight": 0, "shadow": 0, "luma": 100, "colorfulness": 20},
        )
        self.assertTrue(veto)
        self.assertIn("extreme_color", flags)


class RendererBindingTests(unittest.TestCase):
    @staticmethod
    def config(root: Path):
        return SimpleNamespace(
            render=SimpleNamespace(gpu_concurrency=1),
            presets=SimpleNamespace(bank_dir=root),
        )

    def test_startup_rejects_missing_cuda_and_environment_redirection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self.config(Path(tmp))
            with mock.patch("torch.cuda.is_available", return_value=False):
                with self.assertRaises(GpuRenderError) as missing:
                    LocalGpuOnlyRenderer(config)
            self.assertEqual(missing.exception.code, "cuda_unavailable")

            with mock.patch("torch.cuda.is_available", return_value=True), \
                    mock.patch("torch.cuda.device_count", return_value=1), \
                    mock.patch.dict(os.environ, {"MONETGPT_TORCH_DEVICE": "cuda:9"}):
                with self.assertRaises(GpuRenderError) as redirected:
                    LocalGpuOnlyRenderer(config)
            self.assertEqual(redirected.exception.code, "gpu_redirection_rejected")

    def test_every_render_rechecks_backend_and_bound_capability(self) -> None:
        renderer = object.__new__(LocalGpuOnlyRenderer)
        renderer._semaphore = threading.BoundedSemaphore(1)
        renderer.assert_ready = mock.Mock()
        renderer._assert_bound_preset = mock.Mock()
        renderer._upload = mock.Mock(
            side_effect=GpuRenderError("sentinel", "stop after immutable checks")
        )
        source = PreparedSource(np.zeros((2, 3, 3), np.float32), 3, 2)
        preset = SimpleNamespace(preset_id="preset")
        for _ in range(2):
            with self.assertRaisesRegex(GpuRenderError, "immutable checks"):
                renderer.render(source, preset)
        self.assertEqual(renderer.assert_ready.call_count, 2)
        self.assertEqual(renderer._assert_bound_preset.call_count, 2)

    def test_catalog_binding_rejects_missing_or_changed_capability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first_path = root / "first.cube"
            second_path = root / "second.cube"
            first_path.touch()
            second_path.touch()
            preset = PresetRecord(
                "preset", first_path, "lut", "lut", "Style", None, "gpu_lut"
            )
            catalog = PresetCatalog.from_links((TaxonomyLink(preset, "major", "minor"),))
            renderer = object.__new__(LocalGpuOnlyRenderer)
            renderer.config = SimpleNamespace(effective_formats=frozenset({"lut"}))
            renderer._allowed_presets = None

            with self.assertRaisesRegex(GpuRenderError, "not bound"):
                renderer._assert_bound_preset(preset)
            renderer.bind_catalog(catalog)
            renderer._assert_bound_preset(preset)

            with self.assertRaisesRegex(GpuRenderError, "differs"):
                renderer._assert_bound_preset(replace(preset, path=second_path))
            with self.assertRaisesRegex(GpuRenderError, "engine does not match"):
                renderer._assert_bound_preset(replace(preset, render_engine="gpu_local_preset"))


class RetainedDatabaseContractTests(unittest.TestCase):
    def test_minimal_schema_creates_only_retained_tables(self) -> None:
        statements = source_db._split_statements(source_db.SCHEMA)
        tables = {
            match.group(1)
            for statement in statements
            if (match := re.match(
                r"CREATE TABLE IF NOT EXISTS\s+([a-z_]+)", statement, re.IGNORECASE
            ))
        }
        self.assertEqual(tables, {"assets", "processing_events", "runs", "source_captions"})
        schema_upper = source_db.SCHEMA.upper()
        self.assertNotIn("DROP TABLE", schema_upper)
        for legacy in (
            "iqa_scores", "llm_qa", "preset_qa_runs", "preset_previews",
            "render_jobs", "decisions", "sam3_masks", "gate_thresholds",
        ):
            self.assertNotIn(legacy, source_db.SCHEMA)


if __name__ == "__main__":
    unittest.main()
