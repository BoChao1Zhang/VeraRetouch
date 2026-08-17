"""Land checkpoint: publish, mirror, reclaim, and account for lost assets."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from unittest import mock

from construct import agent
from construct.agent import run
from construct.state import ArtifactStore, scan_jsonl, stable_id
from construct.sources import SourceInventoryResult, allocate_sources
from dataset_build.tools.archive_reader import ArchiveReader, prefetch_dir, prefetch_name
from dataset_build.tools.global_catalog import rebuild
from dataset_build.tools.indexed_tar import IndexedTarDataset, IndexedTarError, verify_dataset
from dataset_build.tools.land import land

from dataset_build.tests.test_canonical_orchestration import (
    FakeAnnotator,
    FakeScorer,
    OrchestrationFixture,
)


def _metadata_rows(dataset: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in (dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _member_bytes(dataset: Path, source_path: str, suffix: str) -> bytes:
    """Read the archived payload that was staged from ``source_path``."""
    row = next(
        row for row in _metadata_rows(dataset)
        if row.get("source_path") == source_path and row["member"].endswith(suffix)
    )
    with IndexedTarDataset(dataset) as reader:
        return reader.read_sample(row["sample_id"])[suffix]


class LandFixture(OrchestrationFixture):
    """A build whose checkpoints publish into a throwaway archive on the tmp tree."""

    def setUp(self) -> None:
        super().setUp()
        self.archive = self.root / "archive"
        self.archive.mkdir()
        self.mirror = self.root / "builds"
        self.catalog_db = self.root / "catalog.sqlite3"

    def tearDown(self) -> None:
        # The buffer is process-wide state; a leaked pointer would make the next
        # test read through a directory this one is about to delete.
        self.assertIsNone(prefetch_dir())
        super().tearDown()

    def land_dependencies(self, inventory, *, archive: bool = True):
        dependencies = self.dependencies(inventory)
        dependencies.archive_root = self.archive if archive else None
        dependencies.mirror_root = self.mirror
        dependencies.catalog_db = self.catalog_db
        return dependencies

    def inventory(self, count: int = 4):
        sources = tuple(self.source(index) for index in range(count))
        return SourceInventoryResult(
            sources, {"cache_entries": count, "eligible": count}, "ok"
        )

    def build(self, config, dependencies):
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            return run(config, dependencies=dependencies)


class LandCheckpointTests(LandFixture):
    def test_checkpoint_lands_committed_groups_and_retires_their_staging(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        manifest = self.build(config, dependencies)

        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 2)
        self.assertEqual(manifest["landing"]["checkpoints"], 1)

        groups_dataset = self.archive / "groups" / "build" / "batch-0000"
        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        self.assertEqual(
            verify_dataset(groups_dataset)["members"],
            len(_metadata_rows(groups_dataset)),
        )
        self.assertEqual(
            verify_dataset(sft_dataset)["members"], len(_metadata_rows(sft_dataset))
        )

        # Every candidate of every committed group reached the groups dataset,
        # addressed by the staging path groups.jsonl still records.
        staged = {
            str(candidate["after_path"])
            for group in groups for candidate in group["candidates"]
        }
        archived = {row["source_path"] for row in _metadata_rows(groups_dataset)}
        self.assertTrue(staged.issubset(archived))
        # C_GT travels with its candidate as a second member of the same sample,
        # and the per-candidate QA lands as that sample's metadata.
        local_group = next(row for row in groups if row["render_mode"] == "local")
        local_candidate = local_group["candidates"][0]
        self.assertIn(str(local_candidate["cgt_path"]), archived)
        cgt_row = next(
            row for row in _metadata_rows(groups_dataset)
            if row["source_path"] == str(local_candidate["after_path"])
        )
        with IndexedTarDataset(groups_dataset) as reader:
            sample = reader.read_sample(cgt_row["sample_id"])
        self.assertEqual(sorted(sample), [".cgt.png", ".jpg", ".vrmeta.json"])
        payload = json.loads(sample[".vrmeta.json"])
        self.assertEqual(payload["candidate_id"], local_candidate["candidate_id"])
        self.assertEqual(payload["group_id"], local_group["group_id"])
        self.assertEqual(payload["qa"]["q"], local_candidate["qa"]["q"])
        self.assertEqual(payload["i_in_path"], local_group["source_path"])
        for path in staged:
            self.assertTrue(str(path).startswith(str(config.output_root)))
            self.assertFalse(Path(path).exists())
        # Staging is empty and the transient land tree is gone.
        self.assertEqual(list((config.output_root / "assets" / "candidates").iterdir()), [])
        self.assertFalse((config.output_root / ".land").exists())

        # The refreshed catalog resolves those same paths out of the archive.
        with ArchiveReader(self.catalog_db, verify_checksum=True) as reader:
            for path in sorted(staged):
                self.assertTrue(reader.exists(path))
                self.assertTrue(reader.read(path).startswith(b"\xff\xd8"))

    def test_same_winner_bytes_are_identical_in_both_datasets(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        self.build(config, dependencies)

        groups_dataset = self.archive / "groups" / "build" / "batch-0000"
        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        winners = [
            str(row["I_tar"]) for row in scan_jsonl(config.output_root / "sft.jsonl").records
        ]
        self.assertTrue(winners)
        for winner in winners:
            from_groups = _member_bytes(groups_dataset, winner, ".jpg")
            from_sft = _member_bytes(sft_dataset, winner, ".jpg")
            self.assertEqual(
                hashlib.sha256(from_groups).hexdigest(),
                hashlib.sha256(from_sft).hexdigest(),
            )
        # The SFT view holds winners only, never the whole group.
        sft_samples = {
            row["sample_id"] for row in _metadata_rows(sft_dataset)
        }
        self.assertEqual(len(sft_samples), len(set(winners)))

    def test_one_group_occupies_a_contiguous_slot_ordered_run_of_members(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        self.build(config, dependencies)

        groups_dataset = self.archive / "groups" / "build" / "batch-0000"
        rows = _metadata_rows(groups_dataset)
        by_source = {row["source_path"]: row for row in rows}
        # Members are packed in member-name order, and the staged keys carry the
        # production ordinal, so each group is one uninterrupted run.
        order = sorted({row["member"] for row in rows})
        for group in scan_jsonl(config.output_root / "groups.jsonl").records:
            candidate_ids = {row["candidate_id"] for row in group["candidates"]}
            positions = [
                order.index(by_source[str(candidate["after_path"])]["member"])
                for candidate in sorted(
                    group["candidates"], key=lambda row: row["slot_index"]
                )
            ]
            self.assertEqual(positions, sorted(positions))
            self.assertEqual(len(set(positions)), 8)
            # Nothing from another group is interleaved into that span.
            span = order[min(positions): max(positions) + 1]
            self.assertTrue(all(
                any(candidate_id in member for candidate_id in candidate_ids)
                for member in span
            ))

    def test_mirror_is_atomic_and_equals_the_authoritative_files(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        self.build(config, dependencies)

        mirror = self.mirror / config.build_id
        names = ("groups.jsonl", "sft.jsonl", "failures.jsonl", "manifest.json")
        for name in names:
            self.assertEqual(
                (mirror / name).read_bytes(), (config.output_root / name).read_bytes()
            )
        self.assertEqual(list(mirror.glob("*.tmp")), [])
        # The ledgers are mirrored incrementally, so the directory also carries
        # the offsets that were committed — and nothing else.
        self.assertEqual(
            sorted(path.name for path in mirror.iterdir()),
            sorted((*names, agent.MIRROR_STATE_NAME)),
        )
        state = json.loads((mirror / agent.MIRROR_STATE_NAME).read_text())
        self.assertEqual(state["version"], 1)
        self.assertEqual(
            {name: entry["bytes"] for name, entry in state["files"].items()},
            {
                name: (config.output_root / name).stat().st_size
                for name in names if name.endswith(".jsonl")
            },
        )

    def test_wiped_output_root_is_restored_from_the_mirror_and_resumes(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        first = self.build(config, dependencies)
        before = {
            name: (config.output_root / name).read_bytes()
            for name in ("groups.jsonl", "sft.jsonl")
        }

        # A reboot empties the tmpfs: only the mirror and the archive survive.
        shutil.rmtree(config.output_root)
        resumed = self.build(config, dependencies)

        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["completed"]["groups"], first["completed"]["groups"])
        self.assertEqual(resumed["completed"]["sft"], first["completed"]["sft"])
        self.assertEqual(resumed["completed"]["groups_assets_lost"], 0)
        self.assertEqual(
            before,
            {
                name: (config.output_root / name).read_bytes()
                for name in ("groups.jsonl", "sft.jsonl")
            },
        )
        # Nothing was re-landed: the second batch would have been empty.
        self.assertFalse((self.archive / "groups" / "build" / "batch-0001").exists())

    def test_checkpoint_reclaims_assets_no_group_references(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        self.build(config, dependencies)

        orphan = config.output_root / "assets" / "candidates" / "candidate_orphan.jpg"
        orphan.write_bytes(b"abandoned group attempt")
        self.build(config, dependencies)
        self.assertFalse(orphan.exists())

    def test_lost_assets_are_terminal_and_leave_annotation_alone(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory(), archive=False)
        first = self.build(config, dependencies)
        self.assertEqual(first["completed"]["sft"], 4)

        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        victim = next(row for row in groups if row["render_mode"] == "global")
        for candidate in victim["candidates"]:
            Path(candidate["after_path"]).unlink()

        second = self.build(config, dependencies)
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        lost = [row for row in failures if row["error_code"] == "group_assets_lost"]
        self.assertEqual([row["group_id"] for row in lost], [victim["group_id"]])
        self.assertTrue(lost[0]["terminal"])
        self.assertEqual(
            lost[0]["event_id"],
            stable_id(
                "failure", config.build_id, "terminal", "rendering",
                stable_id("group-assets", victim["group_id"]),
                "group_assets_lost", None, None, victim["group_id"], None, None,
            ),
        )

        # A replacement source restores the mode's count, and no annotation task
        # was derived from the dead group.
        self.assertEqual(second["completed"]["groups"], 2)
        self.assertEqual(second["completed"]["groups_assets_lost"], 1)
        self.assertEqual(second["completed"]["sft"], 4)
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            sum(row["group_id"] == victim["group_id"] for row in rows), 2
        )

        # Re-running is idempotent: the terminal event is written exactly once.
        third = self.build(config, dependencies)
        self.assertEqual(
            sum(
                row["error_code"] == "group_assets_lost"
                for row in scan_jsonl(config.output_root / "failures.jsonl").records
            ),
            1,
        )
        self.assertEqual(third["completed"]["groups"], 2)


class SftDatasetTests(LandFixture):
    """The SFT view: the winner's input image, and its annotation outcome."""

    def test_winner_input_image_lands_as_a_member_of_its_sample(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        manifest = self.build(config, dependencies)

        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        rows = scan_jsonl(config.output_root / "sft.jsonl").records
        self.assertEqual(len(rows), 4)
        for row in rows:
            payload = _member_bytes(sft_dataset, str(row["I_in"]), ".in.jpg")
            self.assertEqual(payload, Path(row["I_in"]).read_bytes())
        # One I_in per winner, and the manifest reports the real count rather
        # than the note it used to carry.
        self.assertEqual(manifest["landing"]["i_in_members"], len(rows))
        self.assertEqual(manifest["landing"]["sft_winners"], len(rows))

        # The input joins the winner's own sample instead of becoming a second
        # one, so a sequential reader gets I_in/I_tar/C_GT together.
        metadata = _metadata_rows(sft_dataset)
        by_sample: dict[str, set] = {}
        for meta in metadata:
            by_sample.setdefault(meta["sample_id"], set()).add(
                "." + meta["member"].partition(".")[2]
            )
        self.assertEqual(len(by_sample), len(rows))
        local_row = next(row for row in rows if row["task_type"] == "local")
        local_sample = next(
            meta["sample_id"] for meta in metadata
            if meta.get("source_path") == str(local_row["I_tar"])
        )
        self.assertEqual(
            sorted(by_sample[local_sample]),
            [".cgt.png", ".in.jpg", ".jpg", ".vrmeta.json"],
        )

    def test_unreadable_source_lands_the_winner_without_failing_the_checkpoint(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        with mock.patch(
            "construct.agent.read_bytes", side_effect=IndexedTarError("source is gone")
        ):
            manifest = self.build(config, dependencies)

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["landing"]["i_in_members"], 0)
        self.assertEqual(manifest["landing"]["sft_winners"], 4)
        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        self.assertEqual(
            verify_dataset(sft_dataset)["members"], len(_metadata_rows(sft_dataset))
        )
        self.assertFalse(any(
            row["member"].endswith(".in.jpg") for row in _metadata_rows(sft_dataset)
        ))

    def annotator(self, failed_code: str = "annotation_round_exhausted"):
        """An annotator that abandons its first task and annotates the rest."""
        outcome: dict[str, str] = {}

        class PartialAnnotator(FakeAnnotator):
            def drain(self, *, max_workers=None, only=None):
                tasks = self.tasks(only)
                for index, task in enumerate(tasks):
                    if index == 0:
                        outcome[str(task["candidate_id"])] = "failed"
                        self.store.append_failure({
                            "build_id": self.store.build_id,
                            "event_id": stable_id("failure", task["task_id"]),
                            "event_type": "terminal",
                            "stage": "annotation",
                            "task_id": task["task_id"],
                            "group_id": task["group_id"],
                            "candidate_id": task["candidate_id"],
                            "retryable": False,
                            "error_code": failed_code,
                            "message": "every round was exhausted",
                            "terminal": True,
                        })
                        continue
                    outcome[str(task["candidate_id"])] = "annotated"
                    self.append_task(task)
                self.store.checkpoint()
                return {
                    "completed": len(tasks) - 1, "terminal": 1,
                    "transport_failed": 0, "pending": 0,
                }

        return PartialAnnotator, outcome

    def test_annotation_outcome_is_synced_into_the_published_metadata(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        annotator, outcome = self.annotator()
        dependencies.annotator_factory = lambda _config, store: annotator(store)
        manifest = self.build(config, dependencies)

        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(sorted(outcome.values()).count("failed"), 1)
        self.assertEqual(
            manifest["landing"]["annotation_status"],
            {"datasets": 1, "samples": 4, "winners": 4},
        )

        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        by_candidate = {
            str(row["candidate_id"]): row for row in _metadata_rows(sft_dataset)
            if row.get("candidate_id")
        }
        sft_by_candidate = {
            str(row["candidate_id"]): row
            for row in scan_jsonl(config.output_root / "sft.jsonl").records
        }
        self.assertEqual(len(by_candidate), 4)
        for candidate_id, status in outcome.items():
            row = by_candidate[candidate_id]
            if status == "annotated":
                self.assertTrue(row["annotated"])
                self.assertEqual(row["sft_id"], sft_by_candidate[candidate_id]["sft_id"])
                self.assertIsNone(row["annotation_failure_code"])
            else:
                self.assertFalse(row["annotated"])
                self.assertIsNone(row["sft_id"])
                self.assertEqual(
                    row["annotation_failure_code"], "annotation_round_exhausted"
                )
                # The bytes stay: an unannotated winner is still a real render.
                self.assertNotIn(candidate_id, sft_by_candidate)

        # Every member row of a winner's sample carries the flag, not just the
        # sample-level one, so a row-wise reader can filter without a join.
        failed_sample = by_candidate[
            next(key for key, value in outcome.items() if value == "failed")
        ]["sample_id"]
        siblings = [
            row for row in _metadata_rows(sft_dataset)
            if row["sample_id"] == failed_sample
        ]
        self.assertGreaterEqual(len(siblings), 3)
        self.assertTrue(all(row["annotated"] is False for row in siblings))

    def test_status_sync_is_idempotent_across_a_resume(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        annotator, _outcome = self.annotator()
        dependencies.annotator_factory = lambda _config, store: annotator(store)
        self.build(config, dependencies)
        sft_dataset = self.archive / "sft" / "build" / "batch-0000"
        before = (sft_dataset / "metadata.jsonl").read_bytes()

        resumed = self.build(config, dependencies)
        self.assertEqual(
            resumed["landing"]["annotation_status"],
            {"datasets": 0, "samples": 4, "winners": 4},
        )
        self.assertEqual(before, (sft_dataset / "metadata.jsonl").read_bytes())
        self.assertEqual(list(sft_dataset.glob("*.tmp")), [])


class CatalogRefreshTests(LandFixture):
    """Landing 只登记本次发布的批次，不重扫整个归档。"""

    def _catalog_groups(self) -> set[str]:
        connection = sqlite3.connect(self.catalog_db)
        try:
            return {row[0] for row in connection.execute('SELECT "group" FROM groups')}
        finally:
            connection.close()

    def test_refresh_registers_this_build_and_leaves_the_rest_of_the_catalog(self) -> None:
        # 归档里先有一个别人发布的组，登记完就把它的目录删掉：全量 rebuild 会
        # 把它从 catalog 里抹掉，增量登记则必须原样留下。
        staging = self.root / "foreign"
        staging.mkdir()
        (staging / "old_0001.jpg").write_bytes(b"foreign" * 100)
        foreign = str(land(
            staging,
            "img/unknown/foreign",
            self.archive,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
        )["group"])
        rebuild(self.archive, self.catalog_db)
        self.assertIn(foreign, self._catalog_groups())
        shutil.rmtree(self.archive / foreign)

        config = self.config()
        self.build(config, self.land_dependencies(self.inventory()))

        registered = self._catalog_groups()
        self.assertIn(foreign, registered)
        self.assertIn(f"groups/{config.build_id}/batch-0000", registered)
        self.assertIn(f"sft/{config.build_id}/batch-0000", registered)

    def test_landed_datasets_lists_only_this_build_batches(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())
        seen: list[list[str]] = []
        real = agent.upsert_catalog

        def spy(root, db_path, groups):
            seen.append(list(groups))
            return real(root, db_path, groups)

        with mock.patch.object(agent, "upsert_catalog", spy):
            self.build(config, dependencies)
        self.assertTrue(seen)
        for groups in seen:
            self.assertEqual(
                groups,
                [f"groups/{config.build_id}/batch-0000", f"sft/{config.build_id}/batch-0000"],
            )


class PrefetchWiringTests(LandFixture):
    """Double-buffered source prefetch around the render loop."""

    def modes(self, config, inventory):
        allocation = allocate_sources(
            list(inventory.eligible),
            build_id=config.build_id,
            seed=config.seed,
            target_groups=config.target_groups,
            mix=config.mix,
        )
        return allocation

    def test_each_chunk_is_buffered_before_it_renders_and_dropped_after(self) -> None:
        config = self.config()
        inventory = self.inventory()
        dependencies = self.land_dependencies(inventory)
        allocation = self.modes(config, inventory)
        buffer = config.output_root / "prefetch"
        events: list[tuple] = []

        def fake_prefetch(paths, dest, *, db_path=None):
            fetched = {}
            for path in paths:
                target = Path(dest) / prefetch_name(path)
                target.write_bytes(Path(path).read_bytes())
                fetched[path] = target
            return fetched

        submit = agent._SourcePrefetch.submit

        def logged_submit(inner, paths):
            events.append(("submit", tuple(paths)))
            return submit(inner, paths)

        def logged_preprocess(path, short_edge):
            events.append((
                "render", str(path), sorted(item.name for item in buffer.iterdir())
            ))
            self.assertEqual(prefetch_dir(), buffer)
            return self.small_preprocess(path, short_edge)

        with mock.patch.object(agent, "PREFETCH_CHUNK", 1), \
                mock.patch.object(agent, "prefetch", fake_prefetch), \
                mock.patch.object(agent._SourcePrefetch, "submit", logged_submit), \
                mock.patch("construct.agent.preprocess_source", logged_preprocess):
            manifest = run(config, dependencies=dependencies)

        first_global = str(allocation.global_[0].source_path)
        next_global = str(allocation.global_[1].source_path)
        first_local = str(allocation.local[0].source_path)
        next_local = str(allocation.local[1].source_path)
        # Chunk k is taken delivery of before it renders and chunk k+1 is queued
        # behind it — that ordering is the double buffer.
        self.assertEqual(
            [event[:2] for event in events],
            [
                ("submit", (first_global,)),
                ("submit", (next_global,)),
                ("render", first_global),
                ("submit", (first_local,)),
                ("submit", (next_local,)),
                ("render", first_local),
            ],
        )
        renders = {event[1]: event[2] for event in events if event[0] == "render"}
        self.assertIn(prefetch_name(first_global), renders[first_global])
        # The first source's copy is reclaimed once its group is committed.
        self.assertNotIn(prefetch_name(first_global), renders[first_local])
        self.assertIn(prefetch_name(first_local), renders[first_local])

        self.assertEqual(manifest["status"], "complete")
        self.assertTrue(manifest["prefetch"]["enabled"])
        self.assertEqual(manifest["prefetch"]["errors"], 0)
        self.assertGreaterEqual(manifest["prefetch"]["buffered"], 2)
        # The buffer is not an asset: the orphan sweep leaves it alone while the
        # build runs and nothing survives the run.
        self.assertFalse(buffer.exists())
        self.assertIsNone(prefetch_dir())

    def test_a_broken_buffer_degrades_to_direct_reads(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory())

        def broken(paths, dest, *, db_path=None):
            raise IndexedTarError("global catalog is missing")

        with mock.patch.object(agent, "PREFETCH_CHUNK", 1), \
                mock.patch.object(agent, "prefetch", broken):
            manifest = self.build(config, dependencies)

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"]["groups"], 2)
        self.assertGreater(manifest["prefetch"]["errors"], 0)
        self.assertEqual(manifest["prefetch"]["buffered"], 0)

    def test_no_archive_root_means_no_buffer_at_all(self) -> None:
        config = self.config()
        dependencies = self.land_dependencies(self.inventory(), archive=False)
        with mock.patch.object(
            agent, "prefetch", mock.Mock(side_effect=AssertionError("prefetched"))
        ):
            manifest = self.build(config, dependencies)

        self.assertEqual(manifest["status"], "complete")
        # Still exact rather than a subset check: "disabled" has to report every
        # counter at rest, including the buffer budget's two, or a build with no
        # archive would be indistinguishable from one whose buffer was reaped.
        self.assertEqual(
            manifest["prefetch"],
            {"enabled": False, "buffered": 0, "errors": 0, "bytes": 0, "evicted": 0},
        )
        self.assertFalse((config.output_root / "prefetch").exists())
        self.assertIsNone(prefetch_dir())


class PreflightForwardTests(LandFixture):
    """[render] qa_preflight_forward, and what turning it off actually defers."""

    def scorer_factory(self, config, calls):
        class CountingScorer(FakeScorer):
            def score(self, path):
                calls.append(str(path))
                return super().score(path)

        def factory():
            calls.append(f"load:{config.output_root.exists()}")
            return CountingScorer()

        return factory

    def run_with_forward(self, *, enabled: bool):
        config = self.config(target=1, local=0.0, global_=1.0)
        config = dataclasses.replace(
            config,
            render=dataclasses.replace(config.render, qa_preflight_forward=enabled),
        )
        inventory = self.inventory(1)
        dependencies = self.land_dependencies(inventory, archive=False)
        calls: list[str] = []
        dependencies.scorer_factory = self.scorer_factory(config, calls)
        manifest = self.build(config, dependencies)
        self.assertEqual(manifest["status"], "complete")
        return calls

    def test_enabled_forward_runs_before_the_output_root_exists(self) -> None:
        calls = self.run_with_forward(enabled=True)
        # One load and one preflight forward, both before any artifact exists,
        # then the nine ranking scores of the single group.
        self.assertEqual(calls[0], "load:False")
        self.assertEqual(len(calls), 11)

    def test_disabled_forward_defers_the_model_to_the_qa_phase(self) -> None:
        calls = self.run_with_forward(enabled=False)
        # No startup forward at all, and the load itself has moved to the first
        # ranking call — by which time the build is already writing artifacts.
        self.assertEqual(calls[0], "load:True")
        self.assertEqual(len(calls), 10)


class LostGroupQueueTests(OrchestrationFixture):
    @staticmethod
    def group(group_id: str = "group-1") -> dict:
        candidates = [
            {
                "candidate_id": f"{group_id}-candidate-{index}",
                "after_path": f"/staging/{group_id}-{index}.jpg",
            }
            for index in range(8)
        ]
        return {
            "build_id": "build",
            "group_id": group_id,
            "source_id": "source-1",
            "source_path": "/before.jpg",
            "render_mode": "global",
            "candidates": candidates,
            "winner_ids": [candidates[0]["candidate_id"]],
        }

    def test_lost_group_is_dropped_from_the_annotation_queue(self) -> None:
        with ArtifactStore(self.root / "queue", "build", fsync_every=1) as store:
            store.append_group(self.group("group-1"))
            store.append_group(self.group("group-2"))
            self.assertEqual(len(store.pending_annotation_tasks()), 2)
            store.append_failure({
                "build_id": "build",
                "event_id": "event-lost",
                "event_type": "terminal",
                "stage": "rendering",
                "task_id": "task-lost",
                "group_id": "group-1",
                "error_code": "group_assets_lost",
                "message": "gone",
                "retryable": False,
                "terminal": True,
            })
            self.assertEqual(store.lost_group_ids(), {"group-1"})
            tasks = store.pending_annotation_tasks()
            self.assertEqual([task["group_id"] for task in tasks], ["group-2"])
