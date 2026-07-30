from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
import unittest
from pathlib import Path

from dataset_build.tools.dataset_plan import PlanError
from dataset_build.tools.global_catalog import main as catalog_main, rebuild, upsert
from dataset_build.tools.indexed_tar import IndexedTarDataset, IndexedTarError, verify_dataset
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


class GlobalCatalogUpsertTests(unittest.TestCase):
    """增量登记必须与全量重建落到同一张表：改动组与未改动组都要对得上。"""

    TABLES = ("groups", "members", "samples", "source_paths")

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "archive"
        self.db = self.root / "global.sqlite3"
        self.sources = self._land_sources()
        self._land_batch("b1", 0)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _land(self, staging: Path, group: str, **kwargs) -> str:
        result = land(
            staging,
            group,
            self.out,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=1024**2,
            **kwargs,
        )
        return str(result["group"])

    def _land_sources(self) -> dict[str, Path]:
        staging = self.root / "sources"
        staging.mkdir()
        paths = {}
        for name in ("src_a.jpg", "src_b.jpg"):
            path = staging / name
            path.write_bytes(name.encode() * 200)
            paths[name] = path
        self._land(staging, "img/unknown/fixture", keep_staging=True)
        return paths

    def _land_batch(self, build_id: str, index: int) -> list[str]:
        """一次 land checkpoint 的两个数据集，SFT 侧带 I_in 的 source_path 别名。"""
        landed = []
        for kind in ("groups", "sft"):
            staging = self.root / f"{kind}-{build_id}-{index}"
            staging.mkdir()
            aliases = {}
            for slot in range(2):
                path = staging / f"{kind}_{index}_{slot}.jpg"
                path.write_bytes(f"{kind}-{index}-{slot}".encode() * 100)
                if kind == "sft":
                    # 与 _land_groups 一样：winner 的 I_in 记的是原始源图路径，
                    # 于是同一个 source_path 同时出现在 img 组和 sft 组里。
                    source = self.sources["src_a.jpg" if slot == 0 else "src_b.jpg"]
                    aliases[str(path)] = str(source)
            landed.append(self._land(staging, f"{kind}/{build_id}", source_paths=aliases))
        return landed

    def _dump(self, db: Path | None = None) -> dict[str, list[tuple]]:
        connection = sqlite3.connect(db or self.db)
        try:
            return {
                table: sorted(connection.execute(f"SELECT * FROM {table}"))
                for table in self.TABLES
            }
        finally:
            connection.close()

    def _rebuilt_dump(self) -> dict[str, list[tuple]]:
        """同一份归档全量重建到另一个文件，作为对照。"""
        reference = self.root / "reference.sqlite3"
        reference.unlink(missing_ok=True)
        rebuild(self.out, reference)
        return self._dump(reference)

    def test_upsert_matches_a_full_rebuild_for_touched_and_untouched_groups(self) -> None:
        rebuild(self.out, self.db)
        before = self._dump()
        landed = self._land_batch("b1", 1)
        self.assertEqual(landed, ["groups/b1/batch-0001", "sft/b1/batch-0001"])

        summary = upsert(self.out, self.db, landed)
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(self._dump(), self._rebuilt_dump())
        # 未受影响的组一行不动（连 root/created_at 都不能被重写）。
        for table in self.TABLES:
            self.assertEqual(
                [row for row in self._dump()[table] if "img/unknown/fixture" in row],
                [row for row in before[table] if "img/unknown/fixture" in row],
            )
        # 新组确实进来了。
        self.assertEqual(
            {row[0] for row in self._dump()["groups"]} - {row[0] for row in before["groups"]},
            set(landed),
        )

    def test_the_order_groups_are_named_in_cannot_change_the_catalog(self) -> None:
        """反查表是 last-writer-wins：登记顺序不能由 CLI 参数顺序决定。

        同一个 source_path 会被多个组认领（img 组与每个 batch 的 sft 组），所以
        谁最后写谁赢；upsert 按 rebuild 的排序走，一次调用内的归属才不漂。
        """
        rebuild(self.out, self.db)
        baseline = self.db.read_bytes()
        landed = self._land_batch("b1", 1) + self._land_batch("b1", 2)
        self.assertEqual(landed, [
            "groups/b1/batch-0001", "sft/b1/batch-0001",
            "groups/b1/batch-0002", "sft/b1/batch-0002",
        ])
        upsert(self.out, self.db, landed)
        forward = self._dump()

        reversed_db = self.root / "reversed.sqlite3"
        reversed_db.write_bytes(baseline)
        upsert(self.out, reversed_db, list(reversed(landed)))
        self.assertEqual(self._dump(reversed_db), forward)
        self.assertEqual(forward, self._rebuilt_dump())

    def test_repeated_upsert_of_the_same_group_is_idempotent(self) -> None:
        rebuild(self.out, self.db)
        landed = self._land_batch("b1", 1)
        upsert(self.out, self.db, landed)
        once = self._dump()
        upsert(self.out, self.db, landed)
        upsert(self.out, self.db, landed + landed)
        self.assertEqual(self._dump(), once)
        self.assertEqual(self._dump(), self._rebuilt_dump())

    def test_upsert_rewrites_a_group_whose_batch_was_republished(self) -> None:
        rebuild(self.out, self.db)
        group = "groups/b1/batch-0000"
        # 同一组重新打包（成员多一个），DELETE 必须先把旧行清干净。
        staging = self.root / "republished"
        staging.mkdir()
        for slot in range(3):
            (staging / f"redo_{slot}.jpg").write_bytes(f"redo-{slot}".encode() * 120)
        shutil.rmtree(self.out / group)
        self.assertEqual(self._land(staging, "groups/b1"), group)

        upsert(self.out, self.db, [group])
        self.assertEqual(self._dump(), self._rebuilt_dump())
        self.assertEqual(
            len([row for row in self._dump()["members"] if row[1] == group]), 3
        )

    def test_a_missing_catalog_falls_back_to_a_full_rebuild(self) -> None:
        self.assertFalse(self.db.exists())
        summary = upsert(self.out, self.db, ["groups/b1/batch-0000"])
        # 只登记这一组会让反查表看不见归档里的其他组，所以缺库时必须全量。
        self.assertEqual(summary["groups"], 3)
        self.assertEqual(self._dump(), self._rebuilt_dump())

    def test_nothing_to_register_leaves_the_catalog_alone(self) -> None:
        rebuild(self.out, self.db)
        before = self._dump()
        digest = self.db.read_bytes()
        summary = upsert(self.out, self.db, [])
        self.assertEqual(summary["groups"], 0)
        self.assertEqual(self.db.read_bytes(), digest)
        self.assertEqual(self._dump(), before)

    def test_an_unpublished_group_fails_and_keeps_the_catalog_intact(self) -> None:
        rebuild(self.out, self.db)
        before = self._dump()
        with self.assertRaisesRegex(IndexedTarError, "not published"):
            upsert(self.out, self.db, ["groups/b1/batch-0000", "groups/b1/batch-0009"])
        self.assertEqual(self._dump(), before)
        self.assertFalse((self.root / "global.sqlite3.upserting").exists())

    def test_cli_registers_only_the_named_groups(self) -> None:
        rebuild(self.out, self.db)
        landed = self._land_batch("b1", 1)
        code = catalog_main(
            ["--dataset-root", str(self.out), "--db", str(self.db)]
            + [argument for group in landed for argument in ("--group", group)]
        )
        self.assertEqual(code, 0)
        self.assertEqual(self._dump(), self._rebuilt_dump())


if __name__ == "__main__":
    unittest.main()
