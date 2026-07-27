from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from dataset_build.tools.dataset_plan import PlanError
from dataset_build.tools.indexed_tar import IndexedTarDataset, verify_dataset
from dataset_build.tools.land import land, next_batch


class LandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "archive"
        self.plans = self.root / "plans"
        self.meta = self.root / "meta"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _staging(self, name: str, payloads: dict[str, bytes]) -> Path:
        staging = self.root / name
        staging.mkdir()
        for filename, payload in payloads.items():
            (staging / filename).write_bytes(payload)
        return staging

    def test_landing_publishes_verifies_then_retires_staging(self) -> None:
        staging = self._staging(
            "run-a", {"shot_0001.jpg": b"a" * 600, "shot_0001.mask.png": b"m" * 200}
        )
        result = land(
            staging,
            "renders/r7_global",
            self.out,
            plan_root=self.plans,
            meta_staging=self.meta,
            shard_size_bytes=1024**2,
        )
        self.assertEqual(result["group"], "renders/r7_global/batch-0000")
        self.assertEqual(result["members"], 2)
        self.assertEqual(result["samples"], 1)
        # Staging is only dropped after the shard verified.
        self.assertFalse(staging.exists())
        dataset = self.out / "renders/r7_global/batch-0000"
        self.assertEqual(verify_dataset(dataset)["members"], 2)
        # The archive carries its own metadata, so it survives staging deletion.
        rows = [
            json.loads(line)
            for line in (dataset / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual({row["member"] for row in rows}, {"batch-0000_shot_0001.jpg", "batch-0000_shot_0001.mask.png"})
        with IndexedTarDataset(dataset) as reader:
            sample = reader.read_sample("batch-0000_shot_0001")
            self.assertEqual(sorted(sample), [".jpg", ".mask.png"])

    def test_second_landing_appends_a_new_batch(self) -> None:
        for expected in ("batch-0000", "batch-0001"):
            staging = self._staging(f"run-{expected}", {f"{expected}_x.jpg": b"payload"})
            result = land(
                staging,
                "renders/r7_global",
                self.out,
                plan_root=self.plans,
                meta_staging=self.meta,
                shard_size_bytes=1024**2,
            )
            self.assertEqual(result["group"], f"renders/r7_global/{expected}")
        self.assertEqual(next_batch(self.out / "renders/r7_global"), "batch-0002")

    def test_non_ascii_filenames_are_sanitised_not_rejected(self) -> None:
        # Source names may be anything; turning them into a legal ASCII key is the
        # planner's job, so this must land rather than fail.
        staging = self._staging("run-cjk", {"中文.jpg": b"x"})
        result = land(
            staging,
            "renders/r7_global",
            self.out,
            plan_root=self.plans,
            meta_staging=self.meta,
            shard_size_bytes=1024**2,
        )
        self.assertEqual(result["members"], 1)
        with IndexedTarDataset(self.out / result["group"]) as reader:
            [(key, suffix)] = [
                (row["sample_id"], row["suffix"])
                for row in [reader.lookup(logical_path=f"{result['group']}/{name}")
                            for name in [
                                json.loads(line)["member"]
                                for line in (self.out / result["group"] / "metadata.jsonl")
                                .read_text(encoding="utf-8").splitlines()
                            ]]
            ]
            self.assertTrue(key.isascii())
            self.assertEqual(suffix, ".jpg")

    def test_failed_landing_keeps_staging(self) -> None:
        # A symlink cannot be archived; the batch must abort before publishing and
        # must never delete the operator's only copy of the data.
        staging = self._staging("run-bad", {"real.jpg": b"x"})
        (staging / "link.jpg").symlink_to(staging / "real.jpg")
        with self.assertRaises(Exception):
            land(
                staging,
                "renders/r7_global",
                self.out,
                plan_root=self.plans,
                meta_staging=self.meta,
                shard_size_bytes=1024**2,
            )
        self.assertTrue(staging.exists())
        self.assertFalse((self.out / "renders/r7_global/batch-0000/manifest.json").exists())

    def test_group_shape_is_enforced(self) -> None:
        staging = self._staging("run-c", {"a.jpg": b"x"})
        with self.assertRaisesRegex(PlanError, "group must look like"):
            land(
                staging,
                "not-a-group",
                self.out,
                plan_root=self.plans,
                meta_staging=self.meta,
            )
        self.assertTrue(staging.exists())


if __name__ == "__main__":
    unittest.main()
