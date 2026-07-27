from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dataset_build.tools.archive_reader import ArchiveReader, path_exists, read_bytes
from dataset_build.tools.global_catalog import rebuild
from dataset_build.tools.indexed_tar import IndexedTarError
from dataset_build.tools.land import land


class ArchiveReaderTests(unittest.TestCase):
    """End-to-end: land a batch, delete the source, still read it by its old path."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.out = self.root / "archive"
        self.staging = self.root / "run-a"
        self.staging.mkdir()
        self.payloads = {
            "shot_0001.jpg": b"pixels-" + b"a" * 500,
            "shot_0001.mask.png": b"mask-" + b"b" * 300,
            "shot_0002.jpg": b"other-" + b"c" * 400,
        }
        self.original: dict[str, Path] = {}
        for name, payload in self.payloads.items():
            path = self.staging / name
            path.write_bytes(payload)
            self.original[name] = path
        land(
            self.staging,
            "renders/r7_global",
            self.out,
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=1024**2,
            keep_staging=True,
        )
        self.db = self.root / "global.sqlite3"
        summary = rebuild(self.out, self.db)
        self.assertEqual(summary["source_paths"], 3)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_reads_every_member_by_its_original_path(self) -> None:
        with ArchiveReader(self.db, verify_checksum=True) as reader:
            for name, payload in self.payloads.items():
                self.assertEqual(reader.read(self.original[name]), payload)

    def test_local_first_then_archive_after_deletion(self) -> None:
        name = "shot_0001.jpg"
        path = self.original[name]
        with ArchiveReader(self.db, verify_checksum=True) as reader:
            # Local copy wins while it exists, even if its bytes differ.
            path.write_bytes(b"local-copy-still-here")
            self.assertEqual(reader.read_bytes(path), b"local-copy-still-here")
            # Once deleted, the same call returns the archived bytes.
            path.unlink()
            self.assertEqual(reader.read_bytes(path), self.payloads[name])
            self.assertEqual(read_bytes(path, db_path=self.db), self.payloads[name])

    def test_existence_gates_see_through_to_the_archive(self) -> None:
        path = self.original["shot_0002.jpg"]
        with ArchiveReader(self.db) as reader:
            self.assertTrue(reader.exists(path))
            path.unlink()
            # The gate must still pass: the asset is usable, just not local.
            self.assertFalse(path.is_file())
            self.assertTrue(reader.exists(path))
            self.assertTrue(path_exists(path, db_path=self.db))
            self.assertFalse(reader.exists(self.root / "never-archived.jpg"))

    def test_unknown_path_and_corrupt_shard_are_reported(self) -> None:
        with ArchiveReader(self.db, verify_checksum=True) as reader:
            with self.assertRaises(KeyError):
                reader.read(self.root / "never-archived.jpg")

        row_path = self.original["shot_0002.jpg"]
        with ArchiveReader(self.db, verify_checksum=True) as reader:
            located = reader.locate(row_path)
        shard = Path(str(located["root"])) / "shards" / f"{located['shard']}.tar"
        with shard.open("r+b") as handle:
            handle.seek(int(located["offset_data"]))
            handle.write(b"X")
        with ArchiveReader(self.db, verify_checksum=True) as reader:
            with self.assertRaisesRegex(IndexedTarError, "checksum"):
                reader.read(row_path)

    def test_missing_catalog_names_the_rebuild_command(self) -> None:
        with self.assertRaisesRegex(IndexedTarError, "global_catalog"):
            ArchiveReader(self.root / "absent.sqlite3")


if __name__ == "__main__":
    unittest.main()
