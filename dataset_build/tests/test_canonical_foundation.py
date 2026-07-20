from __future__ import annotations

import copy
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
from PIL import Image

from dataset_build.lut_io import load_lut

from construct.canonical_masks import (
    MODE_COUNTS,
    build_mask_plan,
    linear_strength,
    pair_mask_slots,
)
from construct.canonical_qa import deterministic_veto, rank_candidates
from construct.config import ConfigError, load_config, redact_text, redact_uri
from construct.presets import (
    CoverageSelector,
    PresetCatalog,
    PresetRecord,
    TaxonomyLink,
    default_capability,
)
from construct.sources import SourceRecord, allocate_sources, largest_remainder
from construct.state import ArtifactStore, StateError, scan_jsonl, stable_id
from construct.rendering import (
    GpuRenderError,
    LocalGpuOnlyRenderer,
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
        self.path.write_text(text.replace(old, new), encoding="utf-8")
        self.path.chmod(0o600)

    def test_example_parses_and_records_effective_formats(self) -> None:
        config = self.load()
        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.effective_formats, {"xmp", "lrtemplate", "lut"})
        safe = config.sanitized_dict()
        self.assertEqual(safe["annotation"]["external_endpoints"][0]["api_key"], "<redacted>")
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
        marker = "\n[[annotation.external_endpoints]]\nid = \"relay-b\""
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(text[: text.index(marker)], encoding="utf-8")
        self.path.chmod(0o600)
        with self.assertRaises(ConfigError):
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
            "https://relay-a.example/v1",
            "https://relay-user:relay-pass@relay-a.example/v1?token=relay-query",
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
