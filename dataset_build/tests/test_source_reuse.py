"""``[sources] max_source_uses``: walking the source pool more than once.

L7 asked for 400,000 groups from 28,189 eligible sources and stopped at 28,189,
because every build until now rendered each source into at most one group.  This
switch lets the pool be walked again — same scene-stratified order, a fresh pass
— so the target is met by ``source x preset`` combinations rather than by source
count alone.

Everything here is written around one claim: **at the default of 1 nothing
changes**.  Not the allocation branch, not a single derived ID, not the shape of
a journal line, and not a resume against a manifest written before the key
existed.  The tests are therefore paired — a default-off assertion next to the
switched-on one it is the baseline for.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from construct.config import (
    DEFAULT_MAX_SOURCE_USES,
    ConfigError,
    MixConfig,
    load_config,
)
from construct import agent
from construct.agent import _resume_config_differences, run
from construct.presets import PresetCatalog, PresetRecord, TaxonomyLink
from construct.sources import SourceInventoryResult, SourceRecord, allocate_sources
from construct.state import scan_jsonl, stable_id
from dataset_build.tools.archive_reader import prefetch_name
from dataset_build.tools.indexed_tar import verify_dataset

from tests.test_canonical_orchestration import OrchestrationFixture
from tests.test_land_integration import LandFixture, _metadata_rows


# A complete, valid config with every pinned constant the loader demands.  It is
# deliberately self-contained rather than derived from ``databuild.example.toml``
# — that file was deleted in 83da846 and every ConfigTests case in the suite that
# still reads it fails on a missing path (see NOTES).  Restore the example and
# this constant should become a read of it.
MINIMAL_TOML = """\
schema_version = 1
build_id = "reuse-config"
seed = 0
target_groups = 100
output_root = "/tmp/reuse-config-out"
preset_filter = "all"

[mix]
local = 0.70
global = 0.30

[sources]
subject_cache = "/tmp/reuse-config-cache"
postgres_dsn = "postgresql://user:pass@127.0.0.1:5432/source"

[presets]
bank_dir = "/tmp/reuse-config-bank"
taxonomy = "/tmp/reuse-config-bank/taxonomy.jsonl"
fidelity_de_max = 6.0
disabled_formats = []

[render]
short_edge = 1024
jpeg_quality = 95
gpu_concurrency = 2
iaa_batch = 8
postprocess_workers = 16
visibility_backend = "torch"
qa_preflight_forward = true
diff_short_edge = 512
visible_de_min = 2.5
visible_fraction_de = 2.3
visible_fraction_min = 0.50

[masks]
linear_target_alpha_mass = 0.50
sam3_relabel_attempts = 2

[annotation]
external_model = "configured-by-operator"
image_long_edge = 768
image_jpeg_quality = 90
external_reasoning_effort = "low"
external_max_output_tokens = 6000
transport_attempts_per_round = 4
queue_rounds = 3
local_fallback = true

[[annotation.external_endpoints]]
id = "lane-1"
base_url = "https://provider.example/v1"
api_key = "REPLACE"
concurrency = 8

[annotation.local]
base_url = "http://127.0.0.1:8003/v1"
api_key = "EMPTY"
model = "qwen3_5-35b-a3b"
temperature = 0.2
enable_thinking = false
max_output_tokens = 2048

[viewer]
postgres_dsn = "postgresql://user:pass@127.0.0.1:5432/viewer"
"""


class MaxSourceUsesConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "databuild.toml"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write(self, body: str = "") -> None:
        text = MINIMAL_TOML
        if body:
            text = text.replace(
                'postgres_dsn = "postgresql://user:pass@127.0.0.1:5432/source"\n',
                'postgres_dsn = "postgresql://user:pass@127.0.0.1:5432/source"\n' + body,
            )
        self.path.write_text(text, encoding="utf-8")
        self.path.chmod(0o600)

    def load(self):
        return load_config(self.path, validate_paths=False)

    def test_the_minimal_template_itself_loads(self) -> None:
        # Guards the other cases in this class: a template that stopped being
        # valid would otherwise make every assertion below vacuous.
        self.write()
        self.assertEqual(self.load().target_groups, 100)

    def test_omitting_the_key_keeps_the_single_use_default(self) -> None:
        self.write()
        self.assertEqual(self.load().sources.max_source_uses, DEFAULT_MAX_SOURCE_USES)
        self.assertEqual(DEFAULT_MAX_SOURCE_USES, 1)

    def test_supported_budgets_load_and_zero_means_unbounded(self) -> None:
        for value in (0, 1, 2, 11, 1000):
            with self.subTest(max_source_uses=value):
                self.write(f"max_source_uses = {value}\n")
                self.assertEqual(self.load().sources.max_source_uses, value)

    def test_out_of_range_and_non_integer_budgets_are_refused(self) -> None:
        for value in ("-1", "1001", '"3"', "1.5", "true"):
            with self.subTest(max_source_uses=value):
                self.write(f"max_source_uses = {value}\n")
                with self.assertRaises(ConfigError):
                    self.load()

    def test_the_budget_reaches_the_effective_config(self) -> None:
        # ``sanitized_dict`` is what the manifest stores and what a resume is
        # compared against, so the key has to survive the round trip.
        self.write("max_source_uses = 11\n")
        self.assertEqual(self.load().sanitized_dict()["sources"]["max_source_uses"], 11)


class ReuseAllocationTests(unittest.TestCase):
    def records(self, n: int) -> list[SourceRecord]:
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

    def allocate(self, pool: int, target: int, uses: int, *, mix=(0.7, 0.3)):
        return allocate_sources(
            self.records(pool),
            build_id="build",
            seed=9,
            target_groups=target,
            mix=MixConfig(local=mix[0], global_=mix[1]),
            max_source_uses=uses,
        )

    def test_the_default_still_starves_the_second_mode_on_a_scarce_pool(self) -> None:
        """The L7 shape, unchanged: 40 sources against a 100-group target.

        ``allocate_sources`` fills the local quota first, so every source becomes
        local and the global mode gets none.  This is the behaviour the switch
        exists to fix, and it must stay exactly this way while the switch is off.
        """
        default = self.allocate(40, 100, DEFAULT_MAX_SOURCE_USES)
        implicit = allocate_sources(
            self.records(40), build_id="build", seed=9, target_groups=100,
            mix=MixConfig(local=0.7, global_=0.3),
        )
        self.assertEqual(default.local_target, 70)
        self.assertEqual(default.global_target, 30)
        self.assertEqual(len(default.local), 40)
        self.assertEqual(len(default.global_), 0)
        # Passing the default explicitly and omitting it are the same call.
        self.assertEqual([row.source_id for row in default.local],
                         [row.source_id for row in implicit.local])

    def test_reuse_splits_a_scarce_pool_so_both_modes_can_reach_their_target(self) -> None:
        # 40 sources, 100 groups, budget 5: local needs ceil(70/5)=14 sources and
        # global ceil(30/5)=6, which the pool covers, so the rest stay
        # replacements and are split by the same weights.
        reuse = self.allocate(40, 100, 5)
        self.assertEqual((reuse.local_target, reuse.global_target), (70, 30))
        self.assertEqual(len(reuse.local), 28)
        self.assertEqual(len(reuse.global_), 12)
        self.assertGreaterEqual(len(reuse.local) * 5, reuse.local_target)
        self.assertGreaterEqual(len(reuse.global_) * 5, reuse.global_target)

    def test_a_pool_too_small_even_for_reuse_is_split_by_the_mix(self) -> None:
        # 10 sources, 100 groups, budget 5 caps the build at 50 groups; the pool
        # is then divided by the configured weights instead of first-come.
        reuse = self.allocate(10, 100, 5)
        self.assertEqual(len(reuse.local), 7)
        self.assertEqual(len(reuse.global_), 3)

    def test_a_zero_weight_mode_is_never_handed_sources_it_cannot_render(self) -> None:
        reuse = self.allocate(10, 100, 5, mix=(1.0, 0.0))
        self.assertEqual(len(reuse.local), 10)
        self.assertEqual(len(reuse.global_), 0)

    def test_every_source_lands_in_exactly_one_mode_under_every_budget(self) -> None:
        """The orchestrator's per-source use cursor is mode-agnostic.

        ``completed_source_uses`` counts a source's groups across both modes, so
        a source appearing in both pools would have its two budgets silently
        merged.  The allocation must keep the pools disjoint.
        """
        for pool, target, uses in (
            (40, 100, 1), (40, 100, 3), (40, 10, 7), (10, 100, 0), (3, 400, 11),
        ):
            with self.subTest(pool=pool, target=target, uses=uses):
                allocation = self.allocate(pool, target, uses)
                ids = [row.source_id
                       for row in allocation.local + allocation.global_]
                self.assertEqual(len(ids), pool)
                self.assertEqual(len(set(ids)), pool)

    def test_reuse_allocation_is_deterministic_and_order_independent(self) -> None:
        first = self.allocate(40, 100, 5)
        second = allocate_sources(
            list(reversed(self.records(40))),
            build_id="build", seed=9, target_groups=100,
            mix=MixConfig(local=0.7, global_=0.3), max_source_uses=5,
        )
        self.assertEqual([row.source_id for row in first.local],
                         [row.source_id for row in second.local])
        self.assertEqual([row.source_id for row in first.global_],
                         [row.source_id for row in second.global_])

    def test_a_duplicated_inventory_row_is_still_rejected_with_reuse_on(self) -> None:
        """Reuse comes from extra passes, never from a duplicated source record.

        The uniqueness gate is what lets ``source_id`` stay the key of the use
        cursor, the SAM3 queue and the terminal set, so switching reuse on must
        not weaken it.
        """
        source = self.records(1)[0]
        for uses in (1, 3, 0):
            with self.subTest(max_source_uses=uses):
                with self.assertRaisesRegex(ValueError, "duplicate source_id"):
                    allocate_sources(
                        [source, source], build_id="build", seed=9, target_groups=4,
                        mix=MixConfig(local=1.0, global_=0.0), max_source_uses=uses,
                    )


class ReuseResumeTests(unittest.TestCase):
    """A manifest written before the key must still resume."""

    def differences(self, old_extra: dict, new_extra: dict) -> list[str]:
        old = {"sources": {"subject_cache": "/cache", **old_extra}}
        new = {"sources": {"subject_cache": "/cache", **new_extra}}
        return _resume_config_differences(old, new)

    def test_a_manifest_without_the_key_resumes_against_the_default(self) -> None:
        self.assertEqual(
            self.differences({}, {"max_source_uses": DEFAULT_MAX_SOURCE_USES}), []
        )

    def test_a_manifest_without_the_key_does_not_excuse_a_non_default(self) -> None:
        self.assertEqual(
            self.differences({}, {"max_source_uses": 3}),
            ["sources.max_source_uses"],
        )

    def test_a_manifest_that_already_carries_the_key_is_compared_exactly(self) -> None:
        self.assertEqual(self.differences({"max_source_uses": 3},
                                          {"max_source_uses": 3}), [])
        self.assertEqual(self.differences({"max_source_uses": 3},
                                          {"max_source_uses": 1}),
                         ["sources.max_source_uses"])


class ReusePipelineTests(OrchestrationFixture):
    """End-to-end runs of the switch, on the orchestration fixture's fakes."""

    def catalog(self):
        """Two majors of sixteen presets each.

        The fixture's own catalog holds exactly eight presets, which is the
        minimum one group consumes — a second group for the same source would be
        forced to reuse all eight and the diversity claim could not be tested at
        all.
        """
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

    def reuse_config(self, *, target, uses, local=0.0, global_=1.0):
        config = self.config(target=target, local=local, global_=global_)
        return dataclasses.replace(
            config,
            sources=dataclasses.replace(config.sources, max_source_uses=uses),
        )

    def build(self, *, target, sources, uses, local=0.0, global_=1.0):
        config = self.reuse_config(
            target=target, uses=uses, local=local, global_=global_
        )
        records = [self.source(index) for index in range(sources)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": sources, "eligible": sources}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=self.dependencies(inventory))
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        return config, manifest, groups

    def test_the_default_renders_each_source_once_and_writes_no_use_index(self) -> None:
        config, manifest, groups = self.build(target=3, sources=3, uses=1)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 3)
        self.assertEqual(len({row["source_id"] for row in groups}), 3)
        # The journal line keeps the shape every pre-key build wrote.
        self.assertFalse(any("source_use_index" in row for row in groups))
        for row in groups:
            self.assertEqual(
                row["group_id"],
                stable_id("group", config.build_id, row["source_id"], "global", 0),
            )
        reuse = manifest["sources"]["source_reuse"]
        self.assertEqual(reuse["max_source_uses"], 1)
        self.assertEqual(reuse["uses_histogram"], {"1": 3})
        self.assertEqual(reuse["max_observed"], 1)

    def test_the_default_stops_at_the_pool_and_records_the_shortfall(self) -> None:
        _, manifest, groups = self.build(target=4, sources=2, uses=1)
        self.assertEqual(len(groups), 2)
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(manifest["sources"]["exhaustion"]["global_shortfall"], 2)

    def test_reuse_reaches_a_target_larger_than_the_pool(self) -> None:
        config, manifest, groups = self.build(target=4, sources=2, uses=2)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 4)
        self.assertEqual(len({row["source_id"] for row in groups}), 2)
        self.assertEqual(len({row["group_id"] for row in groups}), 4)
        candidates = [
            candidate["candidate_id"]
            for row in groups for candidate in row["candidates"]
        ]
        self.assertEqual(len(set(candidates)), 32)
        reuse = manifest["sources"]["source_reuse"]
        self.assertEqual(reuse["uses_histogram"], {"2": 2})
        self.assertEqual(reuse["max_observed"], 2)
        self.assertLessEqual(reuse["max_observed"], reuse["max_source_uses"])

    def test_the_first_pass_keeps_the_ids_a_build_without_reuse_would_write(self) -> None:
        config, _, groups = self.build(target=4, sources=2, uses=2)
        first_pass = [row for row in groups if "source_use_index" not in row]
        self.assertEqual(len(first_pass), 2)
        for row in first_pass:
            self.assertEqual(
                row["group_id"],
                stable_id("group", config.build_id, row["source_id"], "global", 0),
            )
        second_pass = [row for row in groups if row.get("source_use_index") == 1]
        self.assertEqual(len(second_pass), 2)
        for row in second_pass:
            self.assertEqual(
                row["group_id"],
                stable_id("group", config.build_id, row["source_id"], "global", 0, 1),
            )

    def test_the_pool_is_walked_round_robin_rather_than_source_by_source(self) -> None:
        """A,B,A,B — not A,A,B,B.

        Consecutive groups coming from different images is what keeps a landed
        batch (and the annotation queue behind it) from being a run of near
        duplicates.
        """
        _, _, groups = self.build(target=4, sources=2, uses=2)
        order = [row["source_id"] for row in groups]
        self.assertEqual(len(order), 4)
        self.assertNotEqual(order[0], order[1])
        self.assertEqual(order[:2], order[2:])

    def test_the_repeat_pass_does_not_redraw_the_same_preset_set(self) -> None:
        """The portable guarantee: a source's two groups are not the same group.

        This is what the task card actually asks for, and it is the strongest
        claim that survives outside this file.  Nothing seeds the draw on
        ``source_id`` — the coverage selector picks the least-used major, minor
        and preset, and the first pass moved those counters — so the repeat pass
        is pushed off whatever the first pass consumed.

        **It is not disjointness.**  How far it is pushed depends on how large
        the bank is relative to ``8 x max_source_uses``, and the real bank is not
        large enough: measured on the production bank (3522 presets / 10 majors /
        85 minors) at the L8 shape (26,460 sources x 12 uses = 317,520 groups),
        **5.25%** of ``(source, preset)`` draws repeat one the source already
        had, the largest overlap between two groups of one source is **7 of 8**,
        and exact repeats of a whole 8-preset set are **0**.  The manifest
        carries both numbers per build (``sources.source_reuse``) — see
        ``test_the_manifest_reports_preset_redundancy`` — so this stops being an
        assumption and becomes an observable.  The disjointness asserted at the
        bottom of this test is a property of *this file's* 2 x 16 catalog (a
        group eats 8 of a major's 16, so a second group must switch major), not
        of the mechanism.
        """
        _, _, groups = self.build(target=4, sources=2, uses=2)
        by_source: dict[str, list[set[str]]] = {}
        for row in groups:
            presets = {candidate["preset_id"] for candidate in row["candidates"]}
            # The one thing the selector really does guarantee, at any bank size:
            # eight distinct presets inside a group (GroupReservation.commit).
            self.assertEqual(len(presets), 8)
            by_source.setdefault(row["source_id"], []).append(presets)
        self.assertEqual(len(by_source), 2)
        for source_id, passes in by_source.items():
            with self.subTest(source_id=source_id):
                self.assertEqual(len(passes), 2)
                self.assertNotEqual(passes[0], passes[1])
        # Regime-locked below this line: true here, false on the real bank.
        for source_id, passes in by_source.items():
            with self.subTest(source_id=source_id, regime="2x16 toy catalog"):
                self.assertEqual(passes[0] & passes[1], set())
        majors: dict[str, list[str]] = {}
        for row in groups:
            majors.setdefault(row["source_id"], []).append(row["major"])
        for source_id, seen in majors.items():
            with self.subTest(source_id=source_id, regime="2x16 toy catalog"):
                # 12 uses against the real bank's 10 majors makes this false by
                # the pigeonhole principle; it holds here only because 2 majors
                # of 16 presets cannot serve two groups from one major.
                self.assertEqual(len(set(seen)), 2)

    def test_the_manifest_reports_preset_redundancy(self) -> None:
        """The two columns that keep the diversity claim honest per build."""
        _, manifest, groups = self.build(target=4, sources=2, uses=2)
        reuse = manifest["sources"]["source_reuse"]
        # On this catalog the repeat pass is forced onto the other major, so the
        # redundancy columns read clean; the point is that they exist and are
        # computed off the journal rather than asserted.
        self.assertEqual(reuse["duplicate_source_preset_pairs"], 0)
        self.assertEqual(reuse["duplicate_source_preset_rate"], 0.0)
        self.assertEqual(reuse["max_pair_overlap"], 0)
        self.assertFalse(reuse["budget_exceeded"])
        drawn = sum(len(row["candidates"]) for row in groups)
        self.assertEqual(drawn, 32)

    def test_the_redundancy_columns_count_a_real_overlap(self) -> None:
        """A bank too small to avoid repeats must make the columns move.

        Eight presets in one major is exactly one group's worth, so the second
        pass of a source can only redraw the same eight — the degenerate case the
        real bank approaches from above.  If the columns stayed at 0 here they
        would be measuring nothing.
        """
        links = []
        for index in range(8):
            preset_id = f"only-{index}"
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
                "major",
                f"minor-{index % 2}",
            ))
        narrow = PresetCatalog.from_links(links)
        with mock.patch.object(type(self), "catalog", lambda _self: narrow):
            _, manifest, groups = self.build(target=4, sources=2, uses=2)
        self.assertEqual(len(groups), 4)
        reuse = manifest["sources"]["source_reuse"]
        # Each of the 2 sources draws the same 8 presets twice.
        self.assertEqual(reuse["duplicate_source_preset_pairs"], 16)
        self.assertEqual(reuse["duplicate_source_preset_rate"], 0.5)
        self.assertEqual(reuse["max_pair_overlap"], 8)

    def test_the_budget_caps_the_walk(self) -> None:
        _, manifest, groups = self.build(target=10, sources=2, uses=2)
        self.assertEqual(len(groups), 4)
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(manifest["sources"]["exhaustion"]["global_shortfall"], 6)
        self.assertEqual(manifest["sources"]["source_reuse"]["max_observed"], 2)

    def test_zero_walks_the_pool_until_the_target_is_met(self) -> None:
        _, manifest, groups = self.build(target=5, sources=2, uses=0)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 5)
        reuse = manifest["sources"]["source_reuse"]
        self.assertEqual(reuse["max_source_uses"], 0)
        self.assertEqual(reuse["uses_histogram"], {"2": 1, "3": 1})

    def test_local_reuse_shares_one_mask_plan_across_a_source_s_groups(self) -> None:
        """Reuse does not re-plan masks, so C_GT stays keyed by source.

        ``_write_cgt_once`` adopts an existing file by ``mask_id``; had the
        repeat pass re-planned the geometry, that adoption would hand the second
        group the first group's pixels under a new plan.  It cannot, because the
        mask seeds still depend only on ``source_id``.
        """
        _, manifest, groups = self.build(
            target=4, sources=2, uses=2, local=1.0, global_=0.0
        )
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 4)
        by_source: dict[str, list[set[str]]] = {}
        for row in groups:
            by_source.setdefault(row["source_id"], []).append(
                {candidate["mask_id"] for candidate in row["candidates"]}
            )
        for source_id, passes in by_source.items():
            with self.subTest(source_id=source_id):
                self.assertEqual(passes[0], passes[1])
                self.assertEqual(len(passes[0]), 7)
        # Different sources still get different physical masks.
        pools = [passes[0] for passes in by_source.values()]
        self.assertEqual(pools[0] & pools[1], set())

    def test_a_completed_reuse_build_resumes_without_re_rendering(self) -> None:
        config = self.config(target=4, local=0.0, global_=1.0)
        config = dataclasses.replace(
            config, sources=dataclasses.replace(config.sources, max_source_uses=2)
        )
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            run(config, dependencies=self.dependencies(inventory))
        before = {
            name: scan_jsonl(config.output_root / name).records
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=self.dependencies(inventory))
        after = {
            name: scan_jsonl(config.output_root / name).records
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(before, after)

    def test_a_build_that_crashed_mid_reuse_finishes_its_remaining_passes(self) -> None:
        """The regression that made the pass counter resume-derived.

        A crash after pass 0 leaves every source holding a group, so a walk that
        restarts at pass 0 renders nothing, trips the "no progress, stop" guard
        and retires the build — silently, with only a shortfall line to show for
        it.  At the L8 shape that is one crash discarding every remaining pass.
        """
        config = self.reuse_config(target=6, uses=3)
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        crashed = self._crash_after(config, inventory, 1)
        # Precondition: the crash really did land a partial build.
        self.assertEqual(len(crashed), 4)
        self.assertEqual({row.get("source_use_index", 0) for row in crashed}, {0, 1})

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=self.dependencies(inventory))
        after = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(len(after), 6)
        self.assertEqual(resumed["sources"]["exhaustion"]["global_shortfall"], 0)
        self.assertEqual(
            resumed["sources"]["source_reuse"]["uses_histogram"], {"3": 2}
        )
        # The resumed passes carry the ids the uninterrupted run would have.
        self.assertEqual(len({row["group_id"] for row in after}), 6)
        self.assertEqual(
            sorted(row.get("source_use_index", 0) for row in after),
            [0, 0, 1, 1, 2, 2],
        )

    def test_a_source_rescued_by_the_sam3_drain_still_gets_its_budget(self) -> None:
        """The second half of the same regression, with no crash involved.

        ``_drain_sam3_and_replacements`` calls the walk a second time.  Restarting
        that walk at pass 0 gave the rescued source exactly one use while the
        healthy source had already taken its full budget — a clean L8 run would
        have lost ``(budget - 1) x rescued sources`` groups.
        """
        config = self.reuse_config(target=6, uses=3, local=1.0, global_=0.0)
        good = self.source(0)
        bad = self.source(1, border_mask=True)  # mask plan fails -> sam3_queued
        inventory = SourceInventoryResult(
            (good, bad), {"cache_entries": 2, "eligible": 2}, "ok"
        )

        def relabeler(batch, _config, _attempt):
            for source in batch:
                mask = np.zeros((32, 48), dtype=np.uint8)
                mask[9:24, 16:33] = 255
                Image.fromarray(mask, "L").save(source.subject_path)
            return {source.source_id: "ready" for source in batch}

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            manifest = run(config, dependencies=self.dependencies(
                inventory, relabeler=relabeler
            ))
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(len(groups), 6)
        per_source: dict[str, int] = {}
        for row in groups:
            per_source[row["source_id"]] = per_source.get(row["source_id"], 0) + 1
        self.assertEqual(per_source, {good.source_id: 3, bad.source_id: 3})
        self.assertEqual(
            manifest["sources"]["source_reuse"]["uses_histogram"], {"3": 2}
        )
        self.assertEqual(manifest["sources"]["exhaustion"]["local_shortfall"], 0)

    def _crash_after(self, config, inventory, checkpoints: int, *, preprocess=None):
        """Run until the Nth land checkpoint, then die with the journal intact."""
        from construct.agent import CanonicalPipeline

        real_land = CanonicalPipeline._land_checkpoint
        state = {"calls": 0}

        def boom_land(pipeline, *args, **kwargs):
            state["calls"] += 1
            if state["calls"] > checkpoints:
                raise RuntimeError("simulated crash mid-build")
            return real_land(pipeline, *args, **kwargs)

        with mock.patch("construct.agent.preprocess_source",
                        preprocess or self.small_preprocess), \
                mock.patch.object(CanonicalPipeline, "_land_checkpoint", boom_land):
            with self.assertRaises(Exception):
                run(config, dependencies=self.dependencies(inventory))
        return scan_jsonl(config.output_root / "groups.jsonl").records

    def test_the_walk_resumes_past_a_source_that_can_never_render(self) -> None:
        """Pins the cursor-skip guard specifically — the other half cannot save this.

        The pool is deliberately *uneven*: one source is permanently terminal and
        stays at zero uses, so the "start at the least-used source" shortcut is
        dragged back to pass 0 and provides no help at all.  Pass 0 then renders
        nothing (one source cursor-skipped, one terminal) and only the guard's
        distinction between "deferred" and "refused" keeps the walk alive long
        enough for the healthy source to finish its budget.
        """
        config = self.reuse_config(target=5, uses=5)
        good, broken = self.source(0), self.source(1)
        inventory = SourceInventoryResult(
            (good, broken), {"cache_entries": 2, "eligible": 2}, "ok"
        )

        def refuse_one(path, short_edge):
            if str(path) == str(broken.source_path):
                raise RuntimeError("source is unreadable")
            return self.small_preprocess(path, short_edge)

        crashed = self._crash_after(config, inventory, 2, preprocess=refuse_one)
        self.assertEqual({row["source_id"] for row in crashed}, {good.source_id})
        # Partial by construction: the healthy source still owes passes.
        self.assertGreaterEqual(len(crashed), 1)
        self.assertLess(len(crashed), 5)

        with mock.patch("construct.agent.preprocess_source", refuse_one):
            resumed = run(config, dependencies=self.dependencies(inventory))
        after = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(after), 5)
        self.assertEqual(resumed["status"], "complete_with_failures")  # broken source
        self.assertEqual(resumed["sources"]["exhaustion"]["global_shortfall"], 0)
        self.assertEqual(
            resumed["sources"]["source_reuse"]["uses_histogram"], {"5": 1}
        )

    def test_a_resumed_walk_does_not_re_walk_spent_passes(self) -> None:
        """Pins the least-used start — the guard alone would be correct but slow.

        Without it a resumed L8 build re-walks all 26k sources once per spent
        pass before it renders anything, and each of those idle steps still costs
        a full ``_mode_groups`` scan.
        """
        config = self.reuse_config(target=6, uses=3)
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        self.assertEqual(len(self._crash_after(config, inventory, 1)), 4)

        seen: list[tuple[str, int]] = []
        base = agent.CanonicalPipeline._fill_initial_mode

        def spy(pipeline, mode, sources, target, **kwargs):
            seen.append((mode, kwargs.get("use_index", 0)))
            return base(pipeline, mode, sources, target, **kwargs)

        with mock.patch("construct.agent.preprocess_source", self.small_preprocess), \
                mock.patch.object(agent.CanonicalPipeline, "_fill_initial_mode", spy):
            run(config, dependencies=self.dependencies(inventory))
        self.assertEqual(
            len(scan_jsonl(config.output_root / "groups.jsonl").records), 6
        )
        # Both sources already owned passes 0 and 1, so the resumed walk opens
        # directly on pass 2 of the mode that still has work.
        self.assertEqual([use for mode, use in seen if mode == "global"], [2])

    def test_a_terminal_source_does_not_drag_the_start_back_to_pass_zero(self) -> None:
        """A source that never renders must not pin the least-used cursor at 0.

        Terminal sources sit at zero uses forever, so including them in the
        minimum makes the shortcut a no-op on any real pool — L7 ended with 898
        of them.  The walk would then re-open every spent pass, and each idle
        step still pays a full ``_mode_groups`` scan.
        """
        uses = 4
        config = self.reuse_config(target=8, uses=uses)
        good_a, good_b, broken = (self.source(i) for i in range(3))
        inventory = SourceInventoryResult(
            (good_a, good_b, broken), {"cache_entries": 3, "eligible": 3}, "ok"
        )

        def refuse_one(path, short_edge):
            if str(path) == str(broken.source_path):
                raise RuntimeError("source is unreadable")
            return self.small_preprocess(path, short_edge)

        crashed = self._crash_after(config, inventory, 1, preprocess=refuse_one)
        counts: dict[str, int] = {}
        for row in crashed:
            counts[row["source_id"]] = counts.get(row["source_id"], 0) + 1
        expected = min(counts.get(row.source_id, 0) for row in (good_a, good_b))
        # Preconditions, or the assertion below would prove nothing: the healthy
        # sources really did spend passes, the broken one really is durably
        # terminal with no groups at all, and the budget clamp is nowhere near —
        # so only the terminal exclusion can move the opening pass off zero.
        self.assertGreater(expected, 0)
        self.assertLess(expected, uses - 1)
        self.assertNotIn(broken.source_id, counts)
        failures = scan_jsonl(config.output_root / "failures.jsonl").records
        self.assertTrue(any(
            row.get("source_id") == broken.source_id and row.get("terminal")
            for row in failures
        ))

        seen: list[tuple[str, int]] = []
        base = agent.CanonicalPipeline._fill_initial_mode

        def spy(pipeline, mode, sources, target, **kwargs):
            seen.append((mode, kwargs.get("use_index", 0)))
            return base(pipeline, mode, sources, target, **kwargs)

        with mock.patch("construct.agent.preprocess_source", refuse_one), \
                mock.patch.object(agent.CanonicalPipeline, "_fill_initial_mode", spy):
            resumed = run(config, dependencies=self.dependencies(inventory))
        opened = [use for mode, use in seen if mode == "global"]
        self.assertEqual(opened[0], expected)
        self.assertEqual(resumed["sources"]["terminal"], 1)

    def _drop_group_assets(self, config, group) -> None:
        for candidate in group["candidates"]:
            for key in ("after_path", "cgt_path"):
                value = candidate.get(key)
                if value:
                    Path(value).unlink(missing_ok=True)

    def test_a_lost_group_never_buys_a_source_an_extra_use(self) -> None:
        """The cursor counts lost groups, so the start must stay inside the budget.

        A group whose assets died still owns its ``group_id``, so its use is
        spent — but it no longer counts toward the target.  That combination puts
        the least-used cursor *at* the budget while the mode is still short, and
        an unclamped start would open a pass the budget does not own.  At the
        default of 1 that is the switch silently turning itself on.
        """
        config, _, first = self.build(target=2, sources=2, uses=1)
        self.assertEqual(len(first), 2)
        self._drop_group_assets(config, first[0])

        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=self.dependencies(inventory))
        after = scan_jsonl(config.output_root / "groups.jsonl").records
        # The lost group is accounted, not replaced: no source gets a second use.
        self.assertEqual(len(after), 2)
        self.assertFalse(any("source_use_index" in row for row in after))
        self.assertEqual(resumed["completed"]["groups_assets_lost"], 1)
        self.assertEqual(resumed["sources"]["source_reuse"]["max_observed"], 1)
        self.assertFalse(resumed["sources"]["source_reuse"]["budget_exceeded"])

    def test_a_lost_group_does_not_push_a_reuse_build_over_budget(self) -> None:
        """Same clamp, with the switch on: the budget stays the ceiling."""
        config, _, first = self.build(target=4, sources=2, uses=2)
        self.assertEqual(len(first), 4)
        self._drop_group_assets(config, first[0])

        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=self.dependencies(inventory))
        after = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(after), 4)
        reuse = resumed["sources"]["source_reuse"]
        self.assertEqual(reuse["max_observed"], 2)
        self.assertFalse(reuse["budget_exceeded"])
        self.assertEqual(reuse["uses_histogram"], {"2": 2})

    def test_an_exhausted_pool_still_stops_instead_of_spinning(self) -> None:
        """The guard must not become "never stop".

        Cursor skips are the only thing allowed to keep the walk going; a pass
        that rendered nothing and deferred nothing has to retire the mode, or an
        under-supplied build would loop over a pool that has nothing left.
        """
        _, manifest, groups = self.build(target=10, sources=2, uses=0)
        # Budget 0 is unbounded, so only the empty-pass guard can end this.
        self.assertEqual(len(groups), 10)
        self.assertEqual(manifest["status"], "complete")

    def test_an_all_terminal_pool_retires_on_the_first_pass(self) -> None:
        """The other end of the guard: unbounded budget, nothing renderable.

        Every source goes terminal on pass 0, so there is no output and no cursor
        skip — the walk has to stop rather than lap a dead pool forever.
        """
        config = self.reuse_config(target=10, uses=0)
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )

        def refuse(path, short_edge):
            raise RuntimeError("source is unreadable")

        with mock.patch("construct.agent.preprocess_source", refuse):
            manifest = run(config, dependencies=self.dependencies(inventory))
        self.assertEqual(manifest["status"], "complete_with_failures")
        self.assertEqual(manifest["completed"]["groups"], 0)
        self.assertEqual(manifest["sources"]["exhaustion"]["global_shortfall"], 10)
        self.assertEqual(manifest["sources"]["terminal"], 2)

    def test_a_manifest_written_before_the_key_resumes_a_default_build(self) -> None:
        """The regression the neutral-defaults list exists to prevent.

        Every durable build on disk predates ``sources.max_source_uses``.  If the
        key were compared exactly, none of them could ever be resumed again.
        """
        config, _, _ = self.build(target=2, sources=2, uses=1)
        manifest_path = config.output_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aged = copy.deepcopy(manifest)
        removed = aged["effective_config"]["sources"].pop("max_source_uses")
        self.assertEqual(removed, 1)
        manifest_path.write_text(json.dumps(aged), encoding="utf-8")

        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            resumed = run(config, dependencies=self.dependencies(inventory))
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(
            len(scan_jsonl(config.output_root / "groups.jsonl").records), 2
        )

    def test_an_aged_manifest_does_not_excuse_switching_reuse_on_mid_build(self) -> None:
        from construct.state import StateError

        config, _, _ = self.build(target=2, sources=2, uses=1)
        manifest_path = config.output_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["effective_config"]["sources"].pop("max_source_uses")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        switched = dataclasses.replace(
            config, sources=dataclasses.replace(config.sources, max_source_uses=4)
        )
        records = [self.source(index) for index in range(2)]
        inventory = SourceInventoryResult(
            tuple(records), {"cache_entries": 2, "eligible": 2}, "ok"
        )
        with mock.patch("construct.agent.preprocess_source", self.small_preprocess):
            with self.assertRaisesRegex(StateError, "sources.max_source_uses"):
                run(switched, dependencies=self.dependencies(inventory))


class ReusePrefetchAndLandingTests(LandFixture):
    """The two subsystems a second pass could quietly break.

    The source prefetch decides what to buffer from a "already has a group" test
    that reuse makes wrong by default, and the land checkpoint publishes assets
    that two groups of one source now share.
    """

    def reuse_config(self, *, target, uses, local=0.0, global_=1.0):
        config = self.config(target=target, local=local, global_=global_)
        return dataclasses.replace(
            config,
            sources=dataclasses.replace(config.sources, max_source_uses=uses),
        )

    def test_the_second_pass_still_renders_out_of_the_prefetch_buffer(self) -> None:
        """Every pass must be buffered, not just the first.

        The done test behind ``_chunk_paths`` is the one the render loop skips
        on.  Left as a membership test it would call the whole pool finished the
        moment each source owned one group, and every later pass would read its
        images one random archive pread at a time — a throughput collapse with
        no failed assertion anywhere to show for it.
        """
        config = self.reuse_config(target=4, uses=2)
        inventory = self.inventory(2)
        dependencies = self.land_dependencies(inventory)
        buffer = config.output_root / "prefetch"
        renders: list[tuple[str, list[str]]] = []

        def fake_prefetch(paths, dest, *, db_path=None):
            fetched = {}
            for path in paths:
                target = Path(dest) / prefetch_name(path)
                target.write_bytes(Path(path).read_bytes())
                fetched[path] = target
            return fetched

        def logged_preprocess(path, short_edge):
            renders.append(
                (str(path), sorted(item.name for item in buffer.iterdir()))
            )
            return self.small_preprocess(path, short_edge)

        with mock.patch.object(agent, "PREFETCH_CHUNK", 1), \
                mock.patch.object(agent, "prefetch", fake_prefetch), \
                mock.patch("construct.agent.preprocess_source", logged_preprocess):
            manifest = run(config, dependencies=dependencies)

        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["completed"]["groups"], 4)
        self.assertEqual(len(renders), 4)
        self.assertEqual(manifest["prefetch"]["errors"], 0)
        for path, buffered in renders:
            with self.subTest(source=Path(path).name):
                self.assertIn(prefetch_name(path), buffered)

    def test_two_local_groups_of_one_source_land_their_shared_cgt(self) -> None:
        """Reuse keeps one mask plan per source, so two groups share seven C_GTs.

        The land checkpoint hardlinks each candidate's C_GT into the batch and
        then unlinks the staged original, so a shared mask is staged twice and
        reclaimed twice.  Both groups still have to come out of the archive
        whole.
        """
        config = self.reuse_config(target=4, uses=2, local=1.0, global_=0.0)
        inventory = self.inventory(2)
        dependencies = self.land_dependencies(inventory)
        manifest = self.build(config, dependencies)

        self.assertEqual(manifest["status"], "complete")
        groups = scan_jsonl(config.output_root / "groups.jsonl").records
        self.assertEqual(len(groups), 4)
        self.assertEqual(manifest["sources"]["source_reuse"]["uses_histogram"],
                         {"2": 2})

        shared: dict[str, set[str]] = {}
        for row in groups:
            shared.setdefault(row["source_id"], set()).update(
                str(candidate["cgt_path"]) for candidate in row["candidates"]
            )
        # Seven physical masks behind eight slots, shared by both passes.
        for source_id, paths in shared.items():
            with self.subTest(source_id=source_id):
                self.assertEqual(len(paths), 7)
        # Staged originals are reclaimed once landed, and the archive verifies.
        for paths in shared.values():
            for path in paths:
                self.assertFalse(Path(path).is_file())
        dataset = self.archive / "groups" / config.build_id / "batch-0000"
        self.assertEqual(
            verify_dataset(dataset)["members"],
            len(_metadata_rows(dataset)),
        )
        archived = {row.get("source_path") for row in _metadata_rows(dataset)}
        for paths in shared.values():
            self.assertTrue(paths.issubset(archived))
        # Every candidate of every group is addressable, including the second
        # pass's, which is what the widened ID namespace buys.
        staged = {
            str(candidate["after_path"])
            for row in groups for candidate in row["candidates"]
        }
        self.assertEqual(len(staged), 32)
        self.assertTrue(staged.issubset(archived))


if __name__ == "__main__":  # pragma: no cover - parity with the sibling suites
    unittest.main()
