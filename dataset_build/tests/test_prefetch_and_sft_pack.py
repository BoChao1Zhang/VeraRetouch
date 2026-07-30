"""预取缓冲、保序打包与 SFT 重建工具。

用 tmp fixture 造一套真实归档（land → global_catalog），不碰任何生产路径。
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import sqlite3
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from dataset_build.tools.archive_reader import open_image, prefetch_name, read_bytes, set_prefetch_dir
from dataset_build.tools.dataset_plan import META_SUFFIX, PlanError, PlanGroup, PlanRow, write_group
from dataset_build.tools.global_catalog import rebuild
from dataset_build.tools.indexed_tar import build_indexed_tar, verify_dataset
from dataset_build.tools.land import land
from dataset_build.tools.prefetch import _LOCATE_SQL, main as prefetch_main, prefetch
from dataset_build.tools.sft_pack import load_sft_rows, main as sft_pack_main, sft_pack


# 修复前的 SQL（plain JOIN）：留在测试里当 parity oracle。
_LOCATE_SQL_PLAIN_JOIN = _LOCATE_SQL.replace("CROSS JOIN source_paths", "JOIN source_paths")


def _png(width: int, height: int, colour: tuple[int, int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, format="PNG")
    return buffer.getvalue()


class PrefetchTests(unittest.TestCase):
    """按 (shard, offset) 顺序取回归档字节，并让 read_bytes 命中缓冲。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "archive"
        self.buffer = self.root / "prefetch"
        staging = self.root / "run-a"
        staging.mkdir()
        self.payloads = {
            "src_0001.jpg": b"one-" + b"a" * 900,
            "src_0002.jpg": b"two-" + b"b" * 900,
            "src_0003.png": _png(4, 4, (10, 20, 30)),
            "src_0004.jpg": b"four-" + b"d" * 900,
        }
        self.original: dict[str, Path] = {}
        for name, payload in self.payloads.items():
            path = staging / name
            path.write_bytes(payload)
            self.original[name] = path
        # 小 shard_size 逼出多 shard，才谈得上"按 shard 顺序读"。
        land(
            staging,
            "img/unknown/fixture",
            self.out,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=2048,
            keep_staging=True,
        )
        self.db = self.root / "global.sqlite3"
        rebuild(self.out, self.db)
        self.dataset = self.out / "img/unknown/fixture/batch-0000"
        self.assertGreater(len(json.loads((self.dataset / "manifest.json").read_text())["shards"]), 1)

    def tearDown(self) -> None:
        set_prefetch_dir(None)
        self.tmp.cleanup()

    def _delete_locals(self) -> None:
        for path in self.original.values():
            path.unlink()

    def test_returns_the_mapping_and_the_archived_bytes(self) -> None:
        self._delete_locals()
        fetched = prefetch([str(path) for path in self.original.values()], self.buffer, db_path=self.db)
        self.assertEqual(set(fetched), {str(path) for path in self.original.values()})
        for name, payload in self.payloads.items():
            copy = fetched[str(self.original[name])]
            self.assertEqual(copy.name, prefetch_name(self.original[name]))
            self.assertEqual(copy.read_bytes(), payload)

    def test_still_local_paths_are_skipped_and_repeats_are_idempotent(self) -> None:
        kept = self.original["src_0001.jpg"]
        for name, path in self.original.items():
            if name != "src_0001.jpg":
                path.unlink()
        requested = [str(path) for path in self.original.values()]
        fetched = prefetch(requested, self.buffer, db_path=self.db)
        # read_bytes 本来就是 local-first：还在本机的那张不该占缓冲。
        self.assertNotIn(str(kept), fetched)
        self.assertFalse((self.buffer / prefetch_name(kept)).exists())
        before = {path: path.stat().st_mtime_ns for path in fetched.values()}
        again = prefetch(requested, self.buffer, db_path=self.db)
        self.assertEqual(set(again), set(fetched))
        self.assertEqual({path: path.stat().st_mtime_ns for path in again.values()}, before)

    def test_reads_walk_each_shard_forward(self) -> None:
        self._delete_locals()
        # 故意反序请求：物理序必须由 catalog 决定，不是调用方的顺序。
        requested = [str(path) for path in reversed(list(self.original.values()))]
        calls: list[tuple[int, int]] = []
        real_pread = os.pread

        def spy(descriptor: int, size: int, offset: int) -> bytes:
            calls.append((descriptor, offset))
            return real_pread(descriptor, size, offset)

        with mock.patch.object(os, "pread", spy):
            fetched = prefetch(requested, self.buffer, db_path=self.db)
        self.assertEqual(len(fetched), len(self.original))
        self.assertTrue(calls)
        # 每个 fd 只出现一段连续区间（一个 shard 开一次），区间内 offset 单调递增。
        runs: list[int] = []
        for descriptor, _offset in calls:
            if not runs or runs[-1] != descriptor:
                runs.append(descriptor)
        self.assertEqual(len(runs), len(set(runs)), f"a shard was reopened: {calls}")
        for descriptor in runs:
            offsets = [offset for handle, offset in calls if handle == descriptor]
            self.assertEqual(offsets, sorted(offsets), f"backwards seek on fd {descriptor}")

    def test_read_bytes_and_open_image_hit_the_buffer(self) -> None:
        path = self.original["src_0003.png"]
        fetched = prefetch([str(path)], self.buffer, db_path=self.db)  # local still there → skipped
        self.assertEqual(fetched, {})
        path.unlink()
        prefetch([str(path)], self.buffer, db_path=self.db)
        # 缓冲副本换成可区分的字节，才能证明命中的是缓冲而不是归档。
        marker = _png(7, 3, (200, 100, 50))
        (self.buffer / prefetch_name(path)).write_bytes(marker)
        set_prefetch_dir(self.buffer)
        self.assertEqual(read_bytes(path, db_path=self.db), marker)
        self.assertEqual(open_image(path, db_path=self.db).size, (7, 3))
        # 关掉钩子就回落到归档权威副本。
        set_prefetch_dir(None)
        self.assertEqual(read_bytes(path, db_path=self.db), self.payloads["src_0003.png"])
        self.assertEqual(open_image(path, db_path=self.db).size, (4, 4))

    def test_missing_buffer_entry_falls_through_to_the_archive(self) -> None:
        path = self.original["src_0002.jpg"]
        path.unlink()
        set_prefetch_dir(self.root / "never-filled")
        self.assertEqual(read_bytes(path, db_path=self.db), self.payloads["src_0002.jpg"])

    def _locate_with(self, sql: str) -> tuple[list[tuple], list[str]]:
        """跑一遍 _locate 的 SQL，返回 (结果行, 查询计划 detail)。"""
        connection = sqlite3.connect(self.db.as_uri() + "?immutable=1", uri=True)
        try:
            connection.execute("CREATE TEMP TABLE want (source_path TEXT PRIMARY KEY)")
            connection.executemany(
                "INSERT OR IGNORE INTO want VALUES (?)",
                ((str(path),) for path in self.original.values()),
            )
            plan = [str(row[-1]) for row in connection.execute("EXPLAIN QUERY PLAN " + sql)]
            return [tuple(row) for row in connection.execute(sql)], plan
        finally:
            connection.close()

    def test_locate_searches_source_paths_by_key_and_keeps_the_old_rows(self) -> None:
        # 旧 SQL 的 plain JOIN 让优化器把无统计信息的 TEMP want 当成大表，于是
        # SCAN source_paths（生产 4.45M 行，实测每次 5.44s，与请求条数无关）。
        rows, plan = self._locate_with(_LOCATE_SQL)
        self.assertTrue(
            any("SEARCH s USING PRIMARY KEY" in line for line in plan),
            f"source_paths 必须走主键点查: {plan}",
        )
        self.assertFalse(
            any(line.startswith("SCAN s") for line in plan), f"source_paths 又被全表扫: {plan}"
        )
        # 行集与顺序必须与修复前完全一致——CROSS JOIN 只钉连接次序。
        legacy, _legacy_plan = self._locate_with(_LOCATE_SQL_PLAIN_JOIN)
        self.assertEqual(rows, legacy)
        self.assertEqual(len(rows), len(self.original))

    def test_cli_reports_what_it_buffered(self) -> None:
        self._delete_locals()
        paths_file = self.root / "paths.txt"
        paths_file.write_text(
            "\n".join(str(path) for path in self.original.values()) + "\n", encoding="utf-8"
        )
        code = prefetch_main([
            "--paths-file", str(paths_file), "--dest", str(self.buffer), "--db", str(self.db)
        ])
        self.assertEqual(code, 0)
        self.assertEqual(len(list(self.buffer.iterdir())), len(self.original))


class PreserveOrderTests(unittest.TestCase):
    """plan 行序即成员序，且 .vrmeta.json 不能打断 sample 连续性。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "src"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _group(self) -> PlanGroup:
        group = PlanGroup(group="sft/b7")
        # 反字典序，且同一 sample 的两个成员相邻。
        for key, roles in (("s2", ("in", "tar")), ("s1", ("in", "tar"))):
            for role in roles:
                path = self.source / f"{key}.{role}.jpg"
                path.write_bytes(f"{key}-{role}".encode() * 40)
                group.add(PlanRow(path=path, logical_path=f"sft/b7/{key}.{role}.jpg"))
        return group

    def _members(self, plan: Path) -> list[str]:
        return [
            Path(json.loads(line)["logical_path"]).name
            for line in plan.read_text(encoding="utf-8").splitlines()
        ]

    def test_preserved_order_keeps_records_and_tucks_meta_behind_its_sample(self) -> None:
        summary = write_group(
            self._group(),
            self.root / "plan",
            meta_staging=self.root / "meta",
            sample_meta={"s1": {"task_type": "style"}, "s2": {"task_type": "local"}},
            preserve_order=True,
        )
        self.assertEqual(
            self._members(Path(str(summary["plan"]))),
            [
                "s2.in.jpg", "s2.tar.jpg", f"s2{META_SUFFIX}",
                "s1.in.jpg", "s1.tar.jpg", f"s1{META_SUFFIX}",
            ],
        )
        output = self.root / "packed"
        manifest = build_indexed_tar(
            None, output, plan=Path(str(summary["plan"])), shard_size_bytes=1024**2
        )
        self.assertEqual(verify_dataset(output)["members"], 6)
        with tarfile.open(output / str(manifest["shards"][0]["tar"]), mode="r|") as stream:
            self.assertEqual(
                [info.name for info in stream],
                ["s2.in.jpg", "s2.tar.jpg", f"s2{META_SUFFIX}",
                 "s1.in.jpg", "s1.tar.jpg", f"s1{META_SUFFIX}"],
            )

    def test_default_still_sorts_by_member_name(self) -> None:
        summary = write_group(
            self._group(),
            self.root / "plan",
            meta_staging=self.root / "meta",
            sample_meta={"s1": {"task_type": "style"}},
        )
        members = self._members(Path(str(summary["plan"])))
        self.assertEqual(members, sorted(members))
        self.assertEqual(members[0], "s1.in.jpg")

    def test_non_contiguous_caller_order_is_rejected_early(self) -> None:
        group = PlanGroup(group="sft/b7")
        for name in ("s1.in.jpg", "s2.in.jpg", "s1.tar.jpg"):
            path = self.source / name
            path.write_bytes(b"payload")
            group.add(PlanRow(path=path, logical_path=f"sft/b7/{name}"))
        with self.assertRaisesRegex(PlanError, "not contiguous"):
            write_group(
                group, self.root / "plan", meta_staging=self.root / "meta", preserve_order=True
            )


class SftPackTests(unittest.TestCase):
    """从 groups 归档 + sft.jsonl 重建 sft/<id>。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "archive"
        self.build_id = "b7_fixture"

        # 源图归档（I_in 的来源），随后删除本机副本。
        sources = self.root / "sources"
        sources.mkdir()
        self.source_bytes = {
            "src_a.jpg": b"source-a-" + b"a" * 300,
            "src_b.jpg": b"source-b-" + b"b" * 300,
        }
        self.source_paths = {}
        for name, payload in self.source_bytes.items():
            path = sources / name
            path.write_bytes(payload)
            self.source_paths[name] = path
        land(
            sources,
            "img/unknown/fixture",
            self.out,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=1024**2,
            keep_staging=True,
        )
        for path in self.source_paths.values():
            path.unlink()

        # 中间产物归档：每组 2 候选 + winner 的 C_GT。
        staging = self.root / "groups-staging"
        staging.mkdir()
        self.group_bytes = {
            "g1_c0.jpg": b"winner-one-" + b"w" * 400,
            "g1_c1.jpg": b"loser-one-" + b"l" * 400,
            "g1_cgt.png": _png(6, 6, (1, 2, 3)),
            "g2_c0.jpg": b"winner-two-" + b"x" * 400,
            "g2_c1.jpg": b"loser-two-" + b"y" * 400,
        }
        self.group_paths = {}
        for name, payload in self.group_bytes.items():
            path = staging / name
            path.write_bytes(payload)
            self.group_paths[name] = path
        land(
            staging,
            f"groups/{self.build_id}",
            self.out,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=1024**2,
        )
        self.groups_dataset = self.out / "groups" / self.build_id
        self.db = self.root / "global.sqlite3"
        rebuild(self.out, self.db)

        # sft.jsonl：winner 记录，顺序即生产顺序（先 g2 再 g1，验证保序）。
        self.build_dir = self.root / "builds" / self.build_id
        self.build_dir.mkdir(parents=True)
        self.records = [
            {
                "build_id": self.build_id,
                "sft_id": "sft_two",
                "annotation_task_id": "task_two",
                "group_id": "g2",
                "candidate_id": "g2_c0",
                "winner_rank": 1,
                "I_in": str(self.source_paths["src_b.jpg"]),
                "I_tar": str(self.group_paths["g2_c0.jpg"]),
                "recipe": {"kind": "param"},
                "local": None,
                "task_type": "style",
                "instruction": "make it warmer",
                "reasoning": "<problem_light_start>…",
                "annot_src": "responses:external:gpt",
                "qa": {"q": 0.8},
            },
            {
                "build_id": self.build_id,
                "sft_id": "sft_one",
                "annotation_task_id": "task_one",
                "group_id": "g1",
                "candidate_id": "g1_c0",
                "winner_rank": 1,
                "I_in": str(self.source_paths["src_a.jpg"]),
                "I_tar": str(self.group_paths["g1_c0.jpg"]),
                "recipe": {"kind": "lut"},
                "local": {"C_GT": str(self.group_paths["g1_cgt.png"]), "mask_id": "m1"},
                "task_type": "local",
                "instruction": "brighten her face",
                "reasoning": "<problem_light_start>…",
                "annot_src": "responses:local",
                # sft_two deliberately omits annot_model: rows landed before the
                # field existed must still pack.
                "annot_model": "gpt-5.6-sol",
                "qa": {"q": 0.7},
            },
        ]
        (self.build_dir / "sft.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.records),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        set_prefetch_dir(None)
        self.tmp.cleanup()

    def _pack(self, output_name: str = "sft-out", **kwargs) -> dict[str, object]:
        return sft_pack(
            self.build_dir,
            self.groups_dataset,
            self.root / output_name,
            db_path=self.db,
            staging_root=self.root,
            shard_size_bytes=1024**2,
            **kwargs,
        )

    def test_rebuilds_a_verifiable_dataset_in_record_order(self) -> None:
        result = self._pack()
        output = Path(str(result["dataset"]))
        self.assertEqual(result["samples"], len(self.records))
        self.assertEqual(verify_dataset(output)["members"], 7)  # 2+3 assets + 2 meta

        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        with tarfile.open(output / str(manifest["shards"][0]["tar"]), mode="r|") as stream:
            members = [info.name for info in stream]
        self.assertEqual(
            members,
            [
                "sft_two.in.jpg", "sft_two.tar.jpg", f"sft_two{META_SUFFIX}",
                "sft_one.in.jpg", "sft_one.tar.jpg", "sft_one.cgt.png", f"sft_one{META_SUFFIX}",
            ],
        )

    def test_winner_bytes_match_the_groups_dataset(self) -> None:
        result = self._pack()
        output = Path(str(result["dataset"]))
        rows = {
            row["member"]: row
            for row in (
                json.loads(line)
                for line in (output / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }
        index = {
            row["member"]: row
            for root in sorted(self.groups_dataset.glob("*/metadata.jsonl"))
            for row in (
                json.loads(line) for line in root.read_text(encoding="utf-8").splitlines()
            )
        }
        by_source = {row["source_path"]: row for row in index.values()}
        for member, expected_source in (
            ("sft_one.tar.jpg", str(self.group_paths["g1_c0.jpg"])),
            ("sft_one.cgt.png", str(self.group_paths["g1_cgt.png"])),
            ("sft_two.tar.jpg", str(self.group_paths["g2_c0.jpg"])),
        ):
            # 重建成员记的是它的原始产出路径（而不是临时 staging 名），
            # 这样 sft 数据集的反查表与 groups 侧指向同一份来源。
            self.assertEqual(rows[member]["source_path"], expected_source)
            digest = hashlib.sha256(self.group_bytes[Path(expected_source).name]).hexdigest()
            # groups 侧的权威字节
            self.assertIn(expected_source, by_source)
            # sft 侧的重建字节：从 tar 里读回来逐位比较
            payload = self._member_bytes(output, member)
            self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)
        # I_in 走 archive_reader（本机副本已删）
        self.assertEqual(self._member_bytes(output, "sft_one.in.jpg"), self.source_bytes["src_a.jpg"])

    def _member_bytes(self, output: Path, member: str) -> bytes:
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        for shard in manifest["shards"]:
            with tarfile.open(output / str(shard["tar"]), mode="r|") as stream:
                for info in stream:
                    if info.name == member:
                        return stream.extractfile(info).read()
        raise AssertionError(f"member missing: {member}")

    def test_meta_member_carries_ids_not_prose(self) -> None:
        output = Path(str(self._pack()["dataset"]))
        payload = json.loads(self._member_bytes(output, f"sft_one{META_SUFFIX}"))
        self.assertEqual(payload["sft_id"], "sft_one")
        self.assertEqual(payload["task_type"], "local")
        self.assertEqual(payload["annot_src"], "responses:local")
        # 谁写的标注要能被追溯；旧行没有这个字段，落成 None 而不是打包失败。
        self.assertEqual(payload["annot_model"], "gpt-5.6-sol")
        legacy = json.loads(self._member_bytes(output, f"sft_two{META_SUFFIX}"))
        self.assertIn("annot_model", legacy)
        self.assertIsNone(legacy["annot_model"])
        self.assertEqual(payload["qa"], {"q": 0.7})
        self.assertEqual(payload["group"], f"sft/{self.build_id}")
        self.assertNotIn("instruction", payload)
        self.assertNotIn("reasoning", payload)
        self.assertEqual(
            payload["members"][".tar.jpg"], str(self.group_paths["g1_c0.jpg"])
        )

    def test_prefetched_source_is_used_for_i_in(self) -> None:
        buffer = self.root / "prefetch"
        wanted = [str(path) for path in self.source_paths.values()]
        self.assertEqual(len(prefetch(wanted, buffer, db_path=self.db)), 2)
        marker = b"prefetched-a" * 20
        (buffer / prefetch_name(self.source_paths["src_a.jpg"])).write_bytes(marker)
        set_prefetch_dir(buffer)
        try:
            output = Path(str(self._pack("sft-prefetched")["dataset"]))
        finally:
            set_prefetch_dir(None)
        self.assertEqual(self._member_bytes(output, "sft_one.in.jpg"), marker)

    def test_missing_winner_asset_fails_loudly_unless_skipped(self) -> None:
        self.records[1]["I_tar"] = str(self.root / "groups-staging" / "never_landed.jpg")
        (self.build_dir / "sft.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.records),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(PlanError, "does not hold"):
            self._pack("sft-strict")
        result = self._pack("sft-skipped", skip_missing=True)
        self.assertEqual(result["samples"], 1)
        self.assertEqual([row["sft_id"] for row in result["skipped"]], ["sft_one"])

    def test_duplicate_records_do_not_duplicate_samples(self) -> None:
        with (self.build_dir / "sft.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(self.records[0], sort_keys=True) + "\n")
        self.assertEqual(len(load_sft_rows(self.build_dir)), len(self.records))
        self.assertEqual(self._pack()["samples"], len(self.records))

    def test_cli_packs_and_reports(self) -> None:
        code = sft_pack_main([
            "--build-dir", str(self.build_dir),
            "--groups-dataset", str(self.groups_dataset),
            "--output", str(self.root / "sft-cli"),
            "--db", str(self.db),
            "--staging", str(self.root),
        ])
        self.assertEqual(code, 0)
        self.assertEqual(verify_dataset(self.root / "sft-cli")["members"], 7)


if __name__ == "__main__":
    unittest.main()
