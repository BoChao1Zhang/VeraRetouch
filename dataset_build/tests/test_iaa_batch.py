"""Batched OneAlign ranking: identical output, fewer forward passes."""
from __future__ import annotations

import dataclasses
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from construct.agent import _load_scorer
from construct.canonical_qa import rank_candidates
from construct.config import (
    DEFAULT_QA_SCORER_INSTANCES,
    MAX_QA_SCORER_INSTANCES,
    ConfigError,
    load_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "databuild.example.toml"


class SingleScorer:
    """Legacy scorer surface: only score(), no batching."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[str] = []

    def score(self, path: str) -> float | None:
        self.calls.append(path)
        return self.scores[path]


class BatchScorer(SingleScorer):
    """Batch-capable scorer recording the chunk sizes it was asked for."""

    def __init__(self, scores: dict[str, float]) -> None:
        super().__init__(scores)
        self.batches: list[list[str]] = []

    def score_batch(self, paths: list[str]) -> list[float | None]:
        self.batches.append(list(paths))
        return [self.scores[path] for path in paths]


class IaaBatchRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.source = root / "source.jpg"
        Image.fromarray(np.full((32, 32, 3), 110, np.uint8), "RGB").save(self.source)
        self.candidates: list[dict[str, str]] = []
        self.scores = {str(self.source): 50.0}
        for index in range(8):
            path = root / f"after-{index}.jpg"
            # Candidate 7 is a blown-out frame the deterministic veto rejects.
            value = 255 if index == 7 else 105 + index * 5
            Image.fromarray(np.full((32, 32, 3), value, np.uint8), "RGB").save(path)
            self.scores[str(path)] = 90.0 - index * 5
            self.candidates.append(
                {"candidate_id": f"c{index}", "after_path": str(path)}
            )
        self.veto_path = self.candidates[7]["after_path"]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def rank(self, scorer, **kwargs):
        return rank_candidates(str(self.source), self.candidates, scorer, **kwargs)

    def test_batched_ranking_matches_per_image_ranking(self) -> None:
        per_image = self.rank(SingleScorer(self.scores), batch_size=1)
        batched = self.rank(BatchScorer(self.scores), batch_size=8)
        self.assertEqual(batched.candidates, per_image.candidates)
        self.assertEqual(batched.winner_ids, per_image.winner_ids)
        self.assertEqual(batched.source_score, per_image.source_score)
        self.assertEqual(batched.winner_ids, ("c0", "c1"))

    def test_vetoed_candidate_is_never_scored(self) -> None:
        scorer = BatchScorer(self.scores)
        result = self.rank(scorer, batch_size=8)
        scored = [path for batch in scorer.batches for path in batch]
        self.assertNotIn(self.veto_path, scored)
        self.assertEqual(scorer.calls, [])
        # Source plus the seven non-vetoed candidates fit one batch_size=8 pass.
        self.assertEqual([len(batch) for batch in scorer.batches], [8])
        self.assertEqual(scored[0], str(self.source))
        vetoed = next(
            row for row in result.candidates if row["candidate_id"] == "c7"
        )
        self.assertTrue(vetoed["qa"]["veto"])
        self.assertIsNone(vetoed["qa"]["onealign"])
        self.assertFalse(vetoed["qa"]["reliable"])

    def test_small_batch_size_splits_into_several_passes(self) -> None:
        scorer = BatchScorer(self.scores)
        batched = self.rank(scorer, batch_size=3)
        self.assertEqual([len(batch) for batch in scorer.batches], [3, 3, 2])
        self.assertEqual(batched.winner_ids, ("c0", "c1"))

    def test_batch_size_one_falls_back_to_per_image_scoring(self) -> None:
        scorer = BatchScorer(self.scores)
        result = self.rank(scorer, batch_size=1)
        self.assertEqual(scorer.batches, [])
        self.assertEqual(len(scorer.calls), 8)
        self.assertNotIn(self.veto_path, scorer.calls)
        self.assertEqual(result.winner_ids, ("c0", "c1"))

    def test_scorer_without_score_batch_still_works(self) -> None:
        scorer = SingleScorer(self.scores)
        result = self.rank(scorer, batch_size=8)
        self.assertEqual(len(scorer.calls), 8)
        self.assertNotIn(self.veto_path, scorer.calls)
        self.assertEqual(result.winner_ids, ("c0", "c1"))

    def test_default_batch_size_is_used_when_unspecified(self) -> None:
        scorer = BatchScorer(self.scores)
        result = self.rank(scorer)
        self.assertEqual([len(batch) for batch in scorer.batches], [8])
        self.assertEqual(result.winner_ids, ("c0", "c1"))


class QaScorerInstanceConfigTests(unittest.TestCase):
    """``render.qa_scorer_instances`` sizes the OneAlign pool on the QA device."""

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
        self.assertIn("qa_scorer_instances = 2\n", text)
        self.path.write_text(
            text.replace("qa_scorer_instances = 2\n", replacement), encoding="utf-8"
        )
        self.path.chmod(0o600)

    def load(self):
        return load_config(self.path, validate_paths=False)

    def test_example_config_keeps_the_two_copy_default(self) -> None:
        self.assertEqual(self.load().render.qa_scorer_instances, 2)

    def test_supported_instance_counts_load(self) -> None:
        for value in range(1, MAX_QA_SCORER_INSTANCES + 1):
            with self.subTest(qa_scorer_instances=value):
                self.rewrite(f"qa_scorer_instances = {value}\n")
                self.assertEqual(self.load().render.qa_scorer_instances, value)

    def test_out_of_range_instance_counts_are_rejected(self) -> None:
        for value in (0, MAX_QA_SCORER_INSTANCES + 1, -1):
            with self.subTest(qa_scorer_instances=value):
                self.rewrite(f"qa_scorer_instances = {value}\n")
                with self.assertRaisesRegex(ConfigError, "qa_scorer_instances"):
                    self.load()

    def test_missing_key_keeps_configs_written_before_it_loading(self) -> None:
        self.rewrite("")
        config = self.load()
        self.assertNotIn("qa_scorer_instances", self.path.read_text(encoding="utf-8"))
        self.assertEqual(config.render.qa_scorer_instances, DEFAULT_QA_SCORER_INSTANCES)

    def test_pool_size_follows_the_key_not_the_render_semaphore(self) -> None:
        requested: list[int] = []
        config = self.load()
        pool = SingleScorer({"sample.jpg": 50.0})

        def scorer_pool_factory(instances: int):
            requested.append(instances)
            return pool

        dependencies = SimpleNamespace(
            scorer_factory=lambda: SingleScorer({}),
            scorer_pool_factory=scorer_pool_factory,
        )
        for instances, gpu_concurrency in ((1, 2), (4, 2), (3, 1)):
            with self.subTest(instances=instances, gpu_concurrency=gpu_concurrency):
                render = dataclasses.replace(
                    config.render,
                    qa_scorer_instances=instances,
                    gpu_concurrency=gpu_concurrency,
                    qa_preflight_forward=False,
                )
                scorer = _load_scorer(
                    dataclasses.replace(config, render=render), dependencies, None
                )
                # The deferred wrapper only builds the pool on first use.
                scorer.score("sample.jpg")
                self.assertEqual(requested[-1], instances)


class IaaBatchConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "databuild.toml"
        self.reset()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def reset(self) -> None:
        shutil.copyfile(EXAMPLE, self.path)
        self.path.chmod(0o600)

    def set_batch(self, value: int) -> None:
        self.reset()
        text = self.path.read_text(encoding="utf-8")
        self.assertIn("iaa_batch = 8", text)
        self.path.write_text(
            text.replace("iaa_batch = 8", f"iaa_batch = {value}"), encoding="utf-8"
        )
        self.path.chmod(0o600)

    def load(self):
        return load_config(self.path, validate_paths=False)

    def test_example_config_enables_batch_of_eight(self) -> None:
        self.assertEqual(self.load().render.iaa_batch, 8)

    def test_supported_batch_sizes_load(self) -> None:
        for value in range(1, 9):
            with self.subTest(iaa_batch=value):
                self.set_batch(value)
                self.assertEqual(self.load().render.iaa_batch, value)

    def test_out_of_range_batch_sizes_are_rejected(self) -> None:
        for value in (0, 9, -1):
            with self.subTest(iaa_batch=value):
                self.set_batch(value)
                with self.assertRaisesRegex(ConfigError, "iaa_batch"):
                    self.load()

    def test_missing_batch_key_is_rejected(self) -> None:
        self.reset()
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(
            text.replace("iaa_batch = 8\n", ""), encoding="utf-8"
        )
        self.path.chmod(0o600)
        with self.assertRaisesRegex(ConfigError, "iaa_batch"):
            self.load()


if __name__ == "__main__":
    unittest.main()
