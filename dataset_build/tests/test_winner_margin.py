"""The IAA winner-margin policy: when the OneAlign order is allowed to pick.

WP5 stage C measured what the ranking can actually resolve: the winner beats
rank2 in 56% of blind pairs (p=0.69, i.e. chance) when their OneAlign scores are
close, and in 92% once the gap exceeds ten points.  A winner drawn from a
near-tie is therefore not a winner, it is a coin flip with a rank attached.

The policy that follows reads the rank1-rank2 gap and does one of three things:
publish nothing (abstain), publish the winner marked ``low``, or publish it
normally.  What is tested here is that abstention reuses the shape a group
without winners has always had -- no new state, no failure event -- that the
verdict travels from the ranking into the group journal, the annotation task,
the SFT row and the packed sample metadata, and that every artifact written
before the policy existed still loads.
"""
from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from construct import agent
from construct.agent import run
from construct.canonical_qa import rank_candidates
from construct.config import (
    DEFAULT_QA_WINNER_MARGIN_ABSTAIN,
    DEFAULT_QA_WINNER_MARGIN_LOW,
    ConfigError,
    load_config,
)
from construct.state import ArtifactStore, StateError, scan_jsonl, stable_id
from dataset_build.tools.sft_pack import _sample_meta

from tests.test_land_integration import LandFixture
from tests.test_canonical_responses import (
    FakeClient,
    FakeResponses,
    ResponseFixture,
    completed_events,
)
from construct.responses import ResponsesAnnotator


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "databuild.example.toml"


class ScriptedScorer:
    """Returns the score booked for each path; anything unbooked is an error."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def score(self, path: str) -> float | None:
        return self.scores[path]


class MarginPolicyRankingTests(unittest.TestCase):
    """``rank_candidates`` is where a winner is chosen, so the gate lives there."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.jpg"
        Image.fromarray(np.full((32, 32, 3), 110, np.uint8), "RGB").save(self.source)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def candidates(self, values: list[float | None]) -> tuple[list[dict], dict[str, float]]:
        """Eight candidates; ``None`` paints a blown-out frame the veto rejects."""
        rows: list[dict] = []
        scores = {str(self.source): 50.0}
        for index, value in enumerate(values):
            path = self.root / f"after-{index}.jpg"
            level = 255 if value is None else 105 + index
            Image.fromarray(np.full((32, 32, 3), level, np.uint8), "RGB").save(path)
            if value is not None:
                scores[str(path)] = value
            rows.append({"candidate_id": f"c{index}", "after_path": str(path)})
        return rows, scores

    def rank(self, values: list[float | None], **kwargs):
        rows, scores = self.candidates(values)
        return rank_candidates(str(self.source), rows, ScriptedScorer(scores), **kwargs)

    @staticmethod
    def policy(**overrides):
        return {"abstain_margin": 1.0, "low_margin": 2.0, **overrides}

    def test_a_gap_wider_than_the_low_threshold_is_a_normal_winner(self) -> None:
        result = self.rank([90.0, 85.0, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                           **self.policy())
        self.assertEqual(result.winner_ids, ("c0", "c1"))
        self.assertEqual(result.winner_confidence, "normal")
        self.assertAlmostEqual(result.winner_margin, 5.0)

    def test_a_gap_inside_the_low_band_keeps_the_winner_and_flags_it(self) -> None:
        # 1.0 <= 1.5 < 2.0: the ranking is weak evidence, not no evidence.
        result = self.rank([90.0, 88.5, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                           **self.policy())
        self.assertEqual(result.winner_ids, ("c0", "c1"))
        self.assertEqual(result.winner_confidence, "low")
        self.assertAlmostEqual(result.winner_margin, 1.5)

    def test_a_gap_under_the_abstain_threshold_publishes_no_winner(self) -> None:
        result = self.rank([90.0, 89.5, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                           **self.policy())
        self.assertEqual(result.winner_ids, ())
        self.assertEqual(result.winner_confidence, "abstain")
        self.assertAlmostEqual(result.winner_margin, 0.5)
        # The group itself is untouched: eight ranked candidates, still archived.
        self.assertEqual(len(result.candidates), 8)
        self.assertEqual(sorted(row["rank"] for row in result.candidates),
                         list(range(1, 9)))

    def test_the_threshold_boundaries_belong_to_the_kinder_band(self) -> None:
        for margin, expected in ((1.0, "low"), (2.0, "normal")):
            with self.subTest(margin=margin):
                result = self.rank(
                    [90.0, 90.0 - margin, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                    **self.policy(),
                )
                self.assertEqual(result.winner_confidence, expected)
                self.assertTrue(result.winner_ids)

    def test_the_gap_is_measured_against_the_next_scored_candidate(self) -> None:
        # c1 is vetoed and never scored, so rank2 is c2 and the gap is 10, not 1.
        result = self.rank([90.0, None, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                           **self.policy())
        self.assertAlmostEqual(result.winner_margin, 10.0)
        self.assertEqual(result.winner_confidence, "normal")
        self.assertEqual(result.winner_ids, ("c0", "c2"))

    def test_a_single_scored_candidate_is_low_rather_than_an_abstention(self) -> None:
        # Seven vetoed frames leave nothing for the winner to be confused with,
        # so it stands -- but nothing corroborates it either.
        result = self.rank([90.0] + [None] * 7, **self.policy())
        self.assertEqual(result.winner_ids, ("c0",))
        self.assertIsNone(result.winner_margin)
        self.assertEqual(result.winner_confidence, "low")

    def test_a_single_scored_candidate_stays_low_with_the_policy_disabled(self) -> None:
        # Structural, not a threshold outcome: zeroing the gates cannot make an
        # uncorroborated winner normal.
        result = self.rank([90.0] + [None] * 7, abstain_margin=0.0, low_margin=0.0)
        self.assertEqual(result.winner_confidence, "low")

    def test_a_group_no_candidate_could_win_carries_no_verdict(self) -> None:
        # Everything below SFT_THRESHOLD: this is the pre-existing "no winner"
        # group, and the policy never got to judge anything.
        result = self.rank([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0], **self.policy())
        self.assertEqual(result.winner_ids, ())
        self.assertIsNone(result.winner_confidence)
        self.assertAlmostEqual(result.winner_margin, 1.0)

    def test_the_bare_call_carries_no_policy_of_its_own(self) -> None:
        result = self.rank([90.0, 89.99, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0])
        self.assertEqual(result.winner_ids, ("c0", "c1"))
        self.assertEqual(result.winner_confidence, "normal")

    def test_the_gap_is_the_one_the_journal_can_reproduce(self) -> None:
        result = self.rank([90.126_39, 88.001_11, 80.0, 75.0, 70.0, 65.0, 60.0, 55.0],
                           **self.policy())
        by_id = {row["candidate_id"]: row for row in result.candidates}
        self.assertAlmostEqual(
            result.winner_margin,
            by_id["c0"]["qa"]["onealign"] - by_id["c1"]["qa"]["onealign"],
            places=4,
        )


class WinnerMarginConfigTests(unittest.TestCase):
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
        original = "qa_winner_margin_abstain = 1.0\nqa_winner_margin_low = 2.0\n"
        self.assertIn(original, text)
        self.path.write_text(text.replace(original, replacement), encoding="utf-8")

    def test_the_example_declares_the_production_policy(self) -> None:
        render = load_config(str(self.path)).render
        self.assertEqual(render.qa_winner_margin_abstain, 1.0)
        self.assertEqual(render.qa_winner_margin_low, 2.0)

    def test_omitting_the_keys_keeps_the_built_in_policy(self) -> None:
        self.rewrite("")
        render = load_config(str(self.path)).render
        self.assertEqual(render.qa_winner_margin_abstain,
                         DEFAULT_QA_WINNER_MARGIN_ABSTAIN)
        self.assertEqual(render.qa_winner_margin_low, DEFAULT_QA_WINNER_MARGIN_LOW)

    def test_zeroing_both_gates_disables_the_policy(self) -> None:
        self.rewrite("qa_winner_margin_abstain = 0.0\nqa_winner_margin_low = 0.0\n")
        render = load_config(str(self.path)).render
        self.assertEqual(render.qa_winner_margin_abstain, 0.0)
        self.assertEqual(render.qa_winner_margin_low, 0.0)

    def test_an_impossible_pair_of_gates_is_refused(self) -> None:
        cases = {
            "abstain above low": "qa_winner_margin_abstain = 3.0\n"
                                 "qa_winner_margin_low = 2.0\n",
            "negative abstain": "qa_winner_margin_abstain = -1.0\n"
                                "qa_winner_margin_low = 2.0\n",
            "low beyond the scale": "qa_winner_margin_abstain = 1.0\n"
                                    "qa_winner_margin_low = 101.0\n",
        }
        for name, replacement in cases.items():
            with self.subTest(case=name):
                self.rewrite(replacement)
                with self.assertRaises(ConfigError) as caught:
                    load_config(str(self.path))
                self.assertIn("qa_winner_margin_abstain", str(caught.exception))

    def test_a_non_numeric_gate_is_refused(self) -> None:
        self.rewrite('qa_winner_margin_abstain = "wide"\nqa_winner_margin_low = 2.0\n')
        with self.assertRaises(ConfigError):
            load_config(str(self.path))


class WinnerMarginPipelineFixture(LandFixture):
    """The shared ``FakeScorer`` answers one constant, so every group is a tie.

    That makes the fixture an exact margin generator: the band a build lands in
    is decided by the thresholds alone, with no scorer arithmetic in between.
    It lands into a throwaway archive so the published datasets, and not only the
    journals, show what an abstaining build produces.
    """

    def configure(self, *, abstain: float, low: float, tag: str, target: int = 2):
        config = dataclasses.replace(
            self.config(target=target),
            output_root=self.root / f"out-{tag}",
            render=dataclasses.replace(
                self.config(target=target).render,
                qa_winner_margin_abstain=abstain,
                qa_winner_margin_low=low,
            ),
        )
        return config, self.land_dependencies(self.inventory())

    def run_build(self, **kwargs):
        config, dependencies = self.configure(**kwargs)
        return config, dependencies, self.build(config, dependencies)

    @staticmethod
    def journal(config, name):
        return list(scan_jsonl(config.output_root / name).records)


class WinnerMarginPipelineTests(WinnerMarginPipelineFixture):
    def test_abstaining_groups_are_archived_without_winners_and_counted(self) -> None:
        config, _dependencies, manifest = self.run_build(
            abstain=1.0, low=2.0, tag="abstain"
        )
        self.assertEqual(manifest["status"], "complete")
        groups = self.journal(config, "groups.jsonl")
        self.assertEqual(len(groups), 2)
        for group in groups:
            # The abstention shape is the one a winnerless group always had.
            self.assertEqual(group["winner_ids"], [])
            self.assertEqual(group["winner_ranks"], [])
            self.assertEqual(group["winner_confidence"], "abstain")
            self.assertEqual(group["winner_margin"], 0.0)
            self.assertEqual(len(group["candidates"]), 8)
        self.assertEqual(self.journal(config, "sft.jsonl"), [])
        self.assertEqual(manifest["completed"]["winner_abstained"], 2)
        self.assertEqual(manifest["completed"]["winner_top1"], 0)
        self.assertEqual(manifest["completed"]["sft"], 0)
        self.assertEqual(manifest["annotation"]["pending"], 0)

    def test_abstention_is_not_a_failure_and_writes_no_event(self) -> None:
        config, _dependencies, manifest = self.run_build(
            abstain=1.0, low=2.0, tag="events"
        )
        # An abstention is a decision, not an incident: the failure journal gains
        # nothing beyond the landing checkpoint every build writes.
        stages = {row.get("stage") for row in self.journal(config, "failures.jsonl")}
        self.assertEqual(stages, {"landing"})
        self.assertEqual(manifest["failures"]["terminal"], 0)
        # The groups dataset still lands; only the SFT view has nothing to show.
        self.assertEqual(manifest["landing"]["checkpoints"], 1)
        self.assertEqual(manifest["landing"]["sft_winners"], 0)
        self.assertEqual(manifest["landing"]["i_in_members"], 0)
        self.assertTrue((self.archive / "groups" / "build" / "batch-0000").is_dir())
        self.assertFalse((self.archive / "sft" / "build").exists())

    def test_a_low_confidence_build_keeps_its_winners_and_marks_them(self) -> None:
        config, _dependencies, manifest = self.run_build(
            abstain=0.0, low=1.0, tag="low"
        )
        groups = self.journal(config, "groups.jsonl")
        self.assertEqual(len(groups), 2)
        for group in groups:
            self.assertEqual(len(group["winner_ids"]), 2)
            self.assertEqual(group["winner_confidence"], "low")
        self.assertEqual(manifest["completed"]["winner_abstained"], 0)
        self.assertEqual(manifest["completed"]["winner_top2"], 2)

    def test_a_disabled_policy_calls_every_winner_normal(self) -> None:
        config, _dependencies, manifest = self.run_build(
            abstain=0.0, low=0.0, tag="normal"
        )
        for group in self.journal(config, "groups.jsonl"):
            self.assertEqual(group["winner_confidence"], "normal")
        self.assertEqual(manifest["completed"]["winner_abstained"], 0)

    def test_a_manifest_without_the_margin_keys_still_resumes(self) -> None:
        # The neutral-default entries: a build journalled before the keys existed
        # must resume against a config that now carries their defaults.
        config, dependencies, manifest = self.run_build(
            abstain=DEFAULT_QA_WINNER_MARGIN_ABSTAIN,
            low=DEFAULT_QA_WINNER_MARGIN_LOW,
            tag="resume",
        )
        before = {
            name: len(self.journal(config, name))
            for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
        }
        for key in ("qa_winner_margin_abstain", "qa_winner_margin_low"):
            with self.subTest(key=key):
                path = config.output_root / "manifest.json"
                stored = json.loads(path.read_text(encoding="utf-8"))
                stored["effective_config"]["render"].pop(key)
                path.write_text(json.dumps(stored), encoding="utf-8")
                with mock.patch(
                    "construct.agent.preprocess_source", self.small_preprocess
                ):
                    resumed = run(config, dependencies=dependencies)
                self.assertEqual(resumed["status"], "complete")
                self.assertEqual(
                    before,
                    {
                        name: len(self.journal(config, name))
                        for name in ("groups.jsonl", "sft.jsonl", "failures.jsonl")
                    },
                )

    def test_a_changed_threshold_is_still_a_refused_resume(self) -> None:
        config, dependencies, _manifest = self.run_build(
            abstain=0.0, low=0.0, tag="changed"
        )
        changed = dataclasses.replace(
            config,
            render=dataclasses.replace(config.render, qa_winner_margin_abstain=1.0),
        )
        with self.assertRaises(agent.StateError) as caught:
            run(changed, dependencies=dependencies)
        self.assertIn("render.qa_winner_margin_abstain", str(caught.exception))


class WinnerConfidenceRowTests(ResponseFixture):
    """The verdict has to survive the journal -> task -> SFT row -> pack chain."""

    def group(self, *, confidence: str | None):
        task = self.task()
        group = dict(task["group"])
        if confidence is not None:
            group["winner_confidence"] = confidence
            group["winner_margin"] = 1.25
        return group

    def annotate(self, store, task) -> dict:
        clients = {
            "relay-a": FakeClient(FakeResponses([completed_events()])),
            "relay-b": FakeClient(FakeResponses([])),
            "local": FakeClient(FakeResponses([])),
        }
        annotator = ResponsesAnnotator(
            self.config, store, client_factory=self.factory(clients),
            sleep=lambda _: None,
        )
        self.assertEqual(annotator.drain(max_workers=1)["completed"], 1)
        return next(iter(store.sft.values()))

    def test_the_verdict_reaches_the_annotation_task_and_the_sft_row(self) -> None:
        with ArtifactStore(self.root / "low", "build", fsync_every=1) as store:
            store.append_group(self.group(confidence="low"))
            tasks = store.pending_annotation_tasks()
            self.assertEqual([task["winner_confidence"] for task in tasks], ["low"])
            row = self.annotate(store, tasks[0])
            self.assertEqual(row["winner_confidence"], "low")

    def test_a_group_from_before_the_policy_annotates_into_a_null_verdict(self) -> None:
        # eval100's shape: winners chosen before the field existed.  ``None`` is
        # "predates the policy", which is exactly not the same claim as "normal".
        with ArtifactStore(self.root / "legacy", "build", fsync_every=1) as store:
            group = self.group(confidence=None)
            self.assertNotIn("winner_confidence", group)
            store.append_group(group)
            tasks = store.pending_annotation_tasks()
            self.assertEqual([task["winner_confidence"] for task in tasks], [None])
            row = self.annotate(store, tasks[0])
            self.assertIsNone(row["winner_confidence"])

    def test_the_group_journal_only_accepts_the_three_verdicts(self) -> None:
        with ArtifactStore(self.root / "vocab", "build", fsync_every=1) as store:
            for verdict in ("abstain", "low", "normal"):
                with self.subTest(verdict=verdict):
                    group = self.group(confidence=verdict)
                    group["group_id"] = stable_id("group", "build", verdict)
                    if verdict == "abstain":
                        group["winner_ids"] = []
                    self.assertTrue(store.append_group(group))
            bad = self.group(confidence="maybe")
            bad["group_id"] = stable_id("group", "build", "bad")
            with self.assertRaises(StateError) as caught:
                store.append_group(bad)
            self.assertIn("winner_confidence", str(caught.exception))

    def test_the_packed_sample_metadata_carries_the_verdict(self) -> None:
        members = {"in": "0001.in.jpg", "tar": "0001.tar.jpg"}
        marked = _sample_meta({"sft_id": "s1", "winner_confidence": "low"}, members)
        self.assertEqual(marked["winner_confidence"], "low")
        # A row packed from before the policy is still packable and stays null.
        legacy = _sample_meta({"sft_id": "s2"}, members)
        self.assertIn("winner_confidence", legacy)
        self.assertIsNone(legacy["winner_confidence"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
