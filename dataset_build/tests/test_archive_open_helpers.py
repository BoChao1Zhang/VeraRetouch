"""像素消费点在本机文件被删、只剩归档副本时仍然解码得出同一张图（A2 穿透补全）。

`archive_reader.open_image` / `open_rgb` 是这些调用点唯一的解码入口，所以每条用例
都先在本机副本上取一次基准，再删源重取，要求逐像素一致。
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct.canonical_masks import load_subject_alpha  # noqa: E402
from construct.canonical_qa import _stats  # noqa: E402
from construct.rendering import preprocess_source  # noqa: E402
from construct.sources import SourceRecord  # noqa: E402

from dataset_build.tools import archive_reader  # noqa: E402
from dataset_build.tools.archive_reader import open_image, open_rgb  # noqa: E402
from dataset_build.tools.global_catalog import rebuild  # noqa: E402
from dataset_build.tools.land import land  # noqa: E402


def _jpeg(width: int, height: int, *, orientation: int | None = None) -> bytes:
    array = np.random.default_rng(7).integers(0, 255, (height, width, 3), dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    if orientation is None:
        image.save(buffer, format="JPEG", quality=95)
    else:
        exif = image.getexif()
        exif[274] = orientation
        image.save(buffer, format="JPEG", quality=95, exif=exif.tobytes())
    return buffer.getvalue()


def _subject_png(size: int = 64) -> bytes:
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[10:39, 10:39] = 255
    buffer = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


class ArchiveOpenHelperTests(unittest.TestCase):
    """本机文件与仅存归档的文件必须走出同一条解码结果。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.staging = self.root / "batch"
        self.staging.mkdir()
        self.payloads = {
            "src_0001.jpg": _jpeg(96, 64),
            "src_0002.jpg": _jpeg(96, 64, orientation=6),
            "src_0003.png": _subject_png(),
        }
        self.original: dict[str, Path] = {}
        for name, payload in self.payloads.items():
            path = self.staging / name
            path.write_bytes(payload)
            self.original[name] = path
        land(
            self.staging,
            "img/unknown/openhelpers",
            self.root / "archive",
            plan_root=self.root / "plans",
            meta_staging=self.root / "meta",
            shard_size_bytes=1024**2,
            keep_staging=True,
        )
        self.db = self.root / "global.sqlite3"
        rebuild(self.root / "archive", self.db)
        # 集成调用点不传 db_path，只能靠环境变量指向本测试自己的 catalog。
        os.environ[archive_reader.CATALOG_ENV] = str(self.db)
        archive_reader._SHARED.clear()

    def tearDown(self) -> None:
        os.environ.pop(archive_reader.CATALOG_ENV, None)
        archive_reader._SHARED.clear()
        self.tmp.cleanup()

    def test_open_image_decodes_local_and_archived_copies_identically(self) -> None:
        for name in ("src_0001.jpg", "src_0003.png"):
            with self.subTest(name=name):
                path = self.original[name]
                local = open_image(path)
                before = (local.mode, local.size, np.asarray(local).copy())
                path.unlink()
                archived = open_image(path)
                self.assertEqual(archived.mode, before[0])
                self.assertEqual(archived.size, before[1])
                np.testing.assert_array_equal(np.asarray(archived), before[2])

    def test_open_image_keeps_the_stored_mode_instead_of_forcing_rgb(self) -> None:
        # load_subject_alpha 靠这一点：subject.png 必须还是单通道，convert 由调用点做。
        path = self.original["src_0003.png"]
        path.unlink()
        self.assertEqual(open_image(path).mode, "L")

    def test_open_rgb_applies_orientation_and_converts_from_the_archive(self) -> None:
        path = self.original["src_0002.jpg"]
        expected = np.asarray(open_rgb(path)).copy()
        path.unlink()
        archived = open_rgb(path)
        self.assertEqual(archived.mode, "RGB")
        # Orientation=6 是 90° 旋转：宽高必须互换，未做 transpose 会留下 (96, 64)。
        self.assertEqual(archived.size, (64, 96))
        self.assertEqual(open_image(path).size, (96, 64))
        np.testing.assert_array_equal(np.asarray(archived), expected)

    def test_a_path_in_neither_place_still_raises_file_not_found(self) -> None:
        missing = self.root / "never-archived.jpg"
        with self.assertRaises(FileNotFoundError):
            open_image(missing)
        with self.assertRaises(FileNotFoundError):
            open_rgb(missing)

    def test_preprocess_source_survives_the_local_file_being_deleted(self) -> None:
        path = self.original["src_0001.jpg"]
        before = preprocess_source(path, short_edge=32)
        path.unlink()
        after = preprocess_source(path, short_edge=32)
        self.assertEqual((after.width, after.height), (before.width, before.height))
        np.testing.assert_array_equal(after.pixels, before.pixels)

    def test_qa_stats_survives_the_local_file_being_deleted(self) -> None:
        path = self.original["src_0001.jpg"]
        before = _stats(str(path))
        path.unlink()
        after = _stats(str(path))
        self.assertEqual(sorted(after), sorted(before))
        for key, value in before.items():
            self.assertAlmostEqual(after[key], value, places=6)

    def test_load_subject_alpha_survives_the_local_file_being_deleted(self) -> None:
        path = self.original["src_0003.png"]
        record = SourceRecord(
            source_id="src_test_0001",
            source_path=self.original["src_0001.jpg"],
            cache_dir=self.staging,
            subject_path=path,
            subject_meta_path=self.staging / "src_0003.json",
            scene="unknown",
            subject={},
            mask_area=0.2,
        )
        before = load_subject_alpha(record, 48, 48)
        path.unlink()
        np.testing.assert_array_equal(load_subject_alpha(record, 48, 48), before)


if __name__ == "__main__":
    unittest.main()


class InvalidateSharedTests(ArchiveOpenHelperTests):
    """catalog 重建后必须重开 reader：immutable=1 钉死打开时的快照。"""

    def test_rebuild_needs_invalidate_before_new_paths_resolve(self) -> None:
        # 删掉本机副本逼出归档读，令共享 reader 以当前快照被缓存
        # （local-first 短路不会创建 reader）。
        self.original["src_0001.jpg"].unlink()
        archive_reader.read_bytes(str(self.original["src_0001.jpg"]))
        staging2 = self.root / "batch2"
        staging2.mkdir()
        late = staging2 / "src_0009.jpg"
        payload = _jpeg(64, 48)
        late.write_bytes(payload)
        land(
            staging2,
            "img/unknown/openhelpers-late",
            self.root / "archive",
            plan_root=self.root / "plans2",
            meta_staging=self.root / "meta2",
            shard_size_bytes=1024**2,
            keep_staging=True,
        )
        rebuild(self.root / "archive", self.db)
        late.unlink()
        # 旧快照看不到重建后才入库的路径。
        with self.assertRaises((KeyError, FileNotFoundError)):
            archive_reader.read_bytes(str(late))
        archive_reader.invalidate_shared()
        self.assertEqual(archive_reader.read_bytes(str(late)), payload)
