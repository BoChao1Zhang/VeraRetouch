from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

from dataset_build.tools.indexed_tar import (
    BLOCK_SIZE,
    IndexedTarDataset,
    IndexedTarError,
    build_indexed_tar,
    validate_shard,
    verify_dataset,
)
from dataset_build.tools.shard_dataset import main as shard_main


class IndexedTarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write_padding_fixture(self) -> dict[str, bytes]:
        values = {
            "a-511.bin": b"a" * 511,
            "b-512.bin": b"b" * 512,
            "nested/c-513.bin": b"c" * 513,
            "nested/metadata.json": b'{"ready":true}\n',
        }
        for relative, payload in values.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        return values

    def test_padding_offsets_and_random_reads(self) -> None:
        values = self._write_padding_fixture()
        output = self.root / "packed"
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=2048)
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["member_count"], len(values))
        self.assertGreater(manifest["shard_count"], 1)

        result = verify_dataset(output)
        self.assertEqual(result["members"], len(values))
        with IndexedTarDataset(output) as dataset:
            for logical_path in reversed(list(values)):
                row = dataset.lookup(logical_path=logical_path)
                self.assertEqual(row["offset_data"] % BLOCK_SIZE, 0)
                self.assertEqual(row["size"], len(values[logical_path]))
                self.assertEqual(dataset.read(logical_path=logical_path), values[logical_path])

        for item in manifest["shards"]:
            tar_path = output / item["tar"]
            self.assertEqual(tar_path.stat().st_size % BLOCK_SIZE, 0)
            with tar_path.open("rb") as handle:
                handle.seek(-2 * BLOCK_SIZE, os.SEEK_END)
                self.assertEqual(handle.read(), b"\0" * 2 * BLOCK_SIZE)

    def test_tar_and_index_are_byte_deterministic(self) -> None:
        self._write_padding_fixture()
        first = self.root / "first"
        second = self.root / "second"
        first_manifest = build_indexed_tar(self.source, first, shard_size_bytes=1024**2)
        second_manifest = build_indexed_tar(self.source, second, shard_size_bytes=1024**2)
        self.assertEqual(first_manifest["dataset_id"], second_manifest["dataset_id"])
        self.assertEqual(first_manifest["shards"], second_manifest["shards"])
        for item in first_manifest["shards"]:
            self.assertEqual((first / item["tar"]).read_bytes(), (second / item["tar"]).read_bytes())
            self.assertEqual((first / item["index"]).read_bytes(), (second / item["index"]).read_bytes())

    def test_long_directories_are_fine_but_illegal_basenames_are_rejected(self) -> None:
        # Directory depth carries no key meaning, so only the basename is bounded.
        deep = self.source / ("d" * 90) / ("e" * 90)
        deep.mkdir(parents=True)
        relative = f"{'d' * 90}/{'e' * 90}/payload.bin"
        (self.source / relative).write_bytes(b"long-path")
        output = self.root / "packed"
        build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        with IndexedTarDataset(output) as dataset:
            row = dataset.lookup(logical_path=relative)
            self.assertEqual(row["sample_id"], "payload")
            self.assertEqual(row["suffix"], ".bin")
            self.assertEqual(row["member"], "payload.bin")
            self.assertEqual(dataset.read(logical_path=relative), b"long-path")

        for index, (illegal, reason) in enumerate((
            ("payload.ODD.png", "violates"),          # extensions must be lowercase
            ("payload." + "z" * 33 + ".png", "violates"),  # 32 chars per segment
            ("no-extension", "violates"),             # a bare key cannot be decoded
            ("中文.jpg", "violates"),           # USTAR names must stay ASCII
            ("x" * 90 + ".aaaaaaaaaaaaaaaa.png", "USTAR limit"),  # 111 > 100 bytes
        )):
            with self.subTest(member=illegal):
                # A fresh directory per case: a failed assertion must not leave
                # state that makes the next case fail for the wrong reason.
                bad = self.root / f"bad-{index}"
                bad.mkdir()
                (bad / illegal).write_bytes(b"x")
                with self.assertRaisesRegex(IndexedTarError, reason):
                    build_indexed_tar(bad, self.root / f"out-{index}", shard_size_bytes=1024**2)

    def test_sample_members_share_one_key_and_never_split_across_shards(self) -> None:
        payloads = {
            "ppr10k_000001.source.png": b"s" * 900,
            "ppr10k_000001.target_a.png": b"a" * 900,
            "ppr10k_000001.json": b'{"scene":"portrait"}\n',
            "ppr10k_000002.source.png": b"t" * 900,
            "ppr10k_000002.json": b'{"scene":"wedding"}\n',
        }
        for name, payload in payloads.items():
            (self.source / name).write_bytes(payload)
        output = self.root / "packed"
        # A shard size far below one sample forces rotation at every opportunity.
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=1024)
        self.assertEqual(manifest["member_count"], 5)
        self.assertEqual(manifest["sample_count"], 2)
        self.assertEqual(manifest["key_policy"], "webdataset_basename_v1")
        self.assertEqual(verify_dataset(output)["members"], 5)

        shard_of: dict[str, set[str]] = {}
        for item in manifest["shards"]:
            for line in (output / item["index"]).read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                shard_of.setdefault(row["sample_id"], set()).add(row["shard"])
        self.assertEqual(sorted(shard_of), ["ppr10k_000001", "ppr10k_000002"])
        for sample_id, shards in shard_of.items():
            self.assertEqual(len(shards), 1, f"{sample_id} was split across {shards}")

        with IndexedTarDataset(output) as dataset:
            self.assertEqual(
                dataset.read_sample("ppr10k_000001"),
                {
                    ".json": payloads["ppr10k_000001.json"],
                    ".source.png": payloads["ppr10k_000001.source.png"],
                    ".target_a.png": payloads["ppr10k_000001.target_a.png"],
                },
            )
            with self.assertRaisesRegex(IndexedTarError, "needs a suffix"):
                dataset.lookup(sample_id="ppr10k_000001")

        # Replay webdataset's group_by_keys over the raw tar stream: it splits
        # each member name at the first dot and flushes whenever the key changes,
        # so this is the property a wds.WebDataset() pipeline actually relies on.
        streamed: list[tuple[str, set[str]]] = []
        for item in manifest["shards"]:
            with tarfile.open(output / item["tar"], mode="r|") as stream:
                current: tuple[str, set[str]] | None = None
                for info in stream:
                    key, _, extension = info.name.partition(".")
                    if current is None or current[0] != key:
                        if current is not None:
                            streamed.append(current)
                        current = (key, set())
                    current[1].add(extension)
                if current is not None:
                    streamed.append(current)
        self.assertEqual(
            streamed,
            [
                ("ppr10k_000001", {"json", "source.png", "target_a.png"}),
                ("ppr10k_000002", {"json", "source.png"}),
            ],
        )

    def test_plan_mode_archives_a_logical_layout(self) -> None:
        physical = {
            "ppr10k/source/ppr10k_000001.png": b"source-bytes",
            "ppr10k/target_a/ppr10k_000001.png": b"target-bytes",
        }
        for relative, payload in physical.items():
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        # Same basename in two directories collapses into one keyed sample.
        plan_rows = [
            {
                "path": str(self.source / "ppr10k/source/ppr10k_000001.png"),
                "logical_path": "img/portrait/ppr10k/ppr10k_000001.source.png",
            },
            {
                "path": str(self.source / "ppr10k/target_a/ppr10k_000001.png"),
                "logical_path": "img/portrait/ppr10k/ppr10k_000001.target_a.png",
            },
        ]
        plan = self.root / "plan.jsonl"
        plan.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in plan_rows), encoding="utf-8"
        )
        output = self.root / "packed"
        manifest = build_indexed_tar(None, output, plan=plan, shard_size_bytes=1024**2)
        self.assertEqual(manifest["input_mode"], "plan")
        self.assertEqual(manifest["sample_count"], 1)
        self.assertEqual(verify_dataset(output)["members"], 2)
        with IndexedTarDataset(output) as dataset:
            self.assertEqual(
                dataset.read(logical_path="img/portrait/ppr10k/ppr10k_000001.source.png"),
                b"source-bytes",
            )
            self.assertEqual(dataset.suffixes("ppr10k_000001"), [".source.png", ".target_a.png"])

        with self.assertRaisesRegex(IndexedTarError, "exactly one of"):
            build_indexed_tar(self.source, self.root / "both", plan=plan)

    def _write_plan(self, name: str, rows: list[tuple[str, str]]) -> Path:
        plan = self.root / name
        plan.write_text(
            "".join(
                json.dumps({"path": str(self.source / src), "logical_path": logical}) + "\n"
                for src, logical in rows
            ),
            encoding="utf-8",
        )
        return plan

    def test_plan_order_is_authoritative_not_lexicographic(self) -> None:
        """存储顺序 = 消费顺序：plan 行序原样成为 tar 成员序。

        这是让"生产顺序 = 训练读取顺序 = Viewer 取组顺序"成立的前提，
        把散落的随机读变成一次连续读。
        """
        for name in ("g2_win.jpg", "g2_cand.jpg", "g1_win.jpg", "g1_cand.jpg"):
            (self.source / name).write_bytes(name.encode() * 40)
        # 故意用反字典序：先第 2 组、组内先 winner
        rows = [(n, n) for n in ("g2_win.jpg", "g2_cand.jpg", "g1_win.jpg", "g1_cand.jpg")]
        plan = self._write_plan("plan.jsonl", rows)
        output = self.root / "packed"
        manifest = build_indexed_tar(None, output, plan=plan, shard_size_bytes=1024**2)
        self.assertEqual(manifest["member_count"], 4)
        self.assertEqual(verify_dataset(output)["members"], 4)

        with tarfile.open(output / manifest["shards"][0]["tar"], mode="r|") as stream:
            self.assertEqual([info.name for info in stream], [logical for _src, logical in rows])

    def test_non_contiguous_sample_is_rejected(self) -> None:
        for name in ("x.jpg", "x.json", "y.jpg"):
            (self.source / name).write_bytes(b"payload")
        # x 的成员被 y 隔断 → 顺序流读者永远拼不出完整的 x
        plan = self._write_plan(
            "plan.jsonl", [("x.jpg", "x.jpg"), ("y.jpg", "y.jpg"), ("x.json", "x.json")]
        )
        with self.assertRaisesRegex(IndexedTarError, "not contiguous"):
            build_indexed_tar(None, self.root / "packed", plan=plan, shard_size_bytes=1024**2)

    def test_duplicate_member_in_plan_is_rejected(self) -> None:
        (self.source / "x.jpg").write_bytes(b"payload")
        plan = self._write_plan("plan.jsonl", [("x.jpg", "x.jpg"), ("x.jpg", "x.jpg")])
        with self.assertRaisesRegex(IndexedTarError, "duplicate member"):
            build_indexed_tar(None, self.root / "packed", plan=plan, shard_size_bytes=1024**2)

    def test_payload_corruption_is_detected(self) -> None:
        values = self._write_padding_fixture()
        output = self.root / "packed"
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        with IndexedTarDataset(output) as dataset:
            row = dataset.lookup(logical_path="a-511.bin")
        tar_path = output / manifest["shards"][0]["tar"]
        with tar_path.open("r+b") as handle:
            handle.seek(row["offset_data"])
            handle.write(b"x")
            handle.flush()
            os.fsync(handle.fileno())
        with self.assertRaisesRegex(IndexedTarError, "checksum"):
            verify_dataset(output)
        self.assertEqual(values["a-511.bin"][0:1], b"a")

    def test_tampered_index_offset_is_detected(self) -> None:
        self._write_padding_fixture()
        output = self.root / "packed"
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        item = manifest["shards"][0]
        index_path = output / item["index"]
        rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
        rows[0]["offset"] += BLOCK_SIZE
        index_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        with self.assertRaisesRegex(IndexedTarError, "offset aliases"):
            validate_shard(output / item["tar"], index_path, item["shard_id"])

    def test_padding_and_trailer_corruption_are_detected(self) -> None:
        (self.source / "data.bin").write_bytes(b"x" * 511)
        output = self.root / "packed"
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        item = manifest["shards"][0]
        tar_path = output / item["tar"]
        index_path = output / item["index"]
        with IndexedTarDataset(output) as dataset:
            row = dataset.lookup(logical_path="data.bin")
        with tar_path.open("r+b") as handle:
            handle.seek(row["offset_data"] + row["size"])
            handle.write(b"p")
        with self.assertRaisesRegex(IndexedTarError, "padding"):
            validate_shard(tar_path, index_path, item["shard_id"])

        with tar_path.open("r+b") as handle:
            handle.seek(row["offset_data"] + row["size"])
            handle.write(b"\0")
            handle.seek(-1, os.SEEK_END)
            handle.write(b"t")
        with self.assertRaisesRegex(IndexedTarError, "trailer"):
            validate_shard(tar_path, index_path, item["shard_id"])

    def test_nested_paths_are_globally_sorted(self) -> None:
        (self.source / "z.bin").write_bytes(b"z")
        (self.source / "a").mkdir()
        (self.source / "a" / "data.bin").write_bytes(b"a")
        output = self.root / "packed"
        manifest = build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        index_path = output / manifest["shards"][0]["index"]
        logical_paths = [
            json.loads(line)["logical_path"]
            for line in index_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(logical_paths, ["a/data.bin", "z.bin"])

    def test_files_over_parallel_threshold_use_streaming_path(self) -> None:
        payload = b"streamed"
        (self.source / "data.bin").write_bytes(payload)
        output = self.root / "packed"
        build_indexed_tar(
            self.source,
            output,
            shard_size_bytes=1024**2,
            parallel_file_max=1,
        )
        with IndexedTarDataset(output) as dataset:
            self.assertEqual(dataset.read(logical_path="data.bin"), payload)

    def test_symlink_failure_never_publishes_output(self) -> None:
        target = self.source / "target.bin"
        target.write_bytes(b"target")
        (self.source / "link.bin").symlink_to(target)
        output = self.root / "packed"
        with self.assertRaisesRegex(IndexedTarError, "symlinks"):
            build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        self.assertFalse(output.exists())
        partials = list(self.root.glob(".packed.partial.*"))
        self.assertEqual(len(partials), 1)
        self.assertFalse((partials[0] / "manifest.json").exists())

    def test_reader_ignores_directory_without_terminal_manifest(self) -> None:
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "shards").mkdir()
        with self.assertRaisesRegex(IndexedTarError, "manifest"):
            IndexedTarDataset(partial)

    def test_manifest_dataset_id_tampering_is_detected(self) -> None:
        (self.source / "data.bin").write_bytes(b"data")
        output = self.root / "packed"
        build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        manifest_path = output / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["dataset_id"] = "indexed_tar_00000000000000000000000000000000"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(IndexedTarError, "dataset ID"):
            verify_dataset(output)

    def test_existing_output_is_never_replaced(self) -> None:
        (self.source / "data.bin").write_bytes(b"data")
        output = self.root / "packed"
        output.mkdir()
        sentinel = output / "sentinel"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(IndexedTarError, "already exists"):
            build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_manifest_parent_symlink_cannot_escape_dataset(self) -> None:
        (self.source / "data.bin").write_bytes(b"data")
        output = self.root / "packed"
        build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        external = self.root / "external-shards"
        (output / "shards").rename(external)
        (output / "shards").symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(IndexedTarError, "symlink"):
            verify_dataset(output)

    def test_sqlite_uri_characters_in_output_path(self) -> None:
        (self.source / "data.bin").write_bytes(b"data")
        output = self.root / "packed?#dataset"
        build_indexed_tar(self.source, output, shard_size_bytes=1024**2)
        self.assertEqual(verify_dataset(output)["members"], 1)
        with IndexedTarDataset(output) as dataset:
            self.assertEqual(dataset.read(logical_path="data.bin"), b"data")

    def test_cli_rejects_non_finite_shard_size(self) -> None:
        output = self.root / "packed"
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = shard_main([
                "pack",
                "--source", str(self.source),
                "--output", str(output),
                "--shard-size-gib", "nan",
            ])
        self.assertEqual(result, 1)
        self.assertIn("finite and positive", stderr.getvalue())
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
