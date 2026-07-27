"""源池发现/门控在本机文件被归档后仍然成立（删源前的第 3 道门禁）。

这些测试不依赖 postgres：scene 元信息用 connection_factory 注入空结果。
"""

from __future__ import annotations

import io
import json
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

from construct.sources import build_inventory  # noqa: E402

from dataset_build.tools import archive_reader  # noqa: E402
from dataset_build.tools.global_catalog import rebuild  # noqa: E402
from dataset_build.tools.land import land  # noqa: E402


def _jpeg(width: int, height: int) -> bytes:
    array = np.random.default_rng(0).integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


class SourceArchiveFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache = self.root / "subject_cache"
        self.entry = self.cache / "0000abcd"
        self.entry.mkdir(parents=True)
        self.source_image = self.root / "images" / "src_0001.jpg"
        self.source_image.parent.mkdir()
        self.source_image.write_bytes(_jpeg(64, 64))
        # 一个 20% 面积的主体 mask：落在 0.005~0.85 的合格区间内
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:39, 10:39] = 255
        buffer = io.BytesIO()
        Image.fromarray(mask, mode="L").save(buffer, format="PNG")
        (self.entry / "subject.png").write_bytes(buffer.getvalue())
        (self.entry / "subject.json").write_text(
            json.dumps({
                "status": "ready",
                "asset_id": "src_test_0001",
                "source_path": str(self.source_image),
                "sam_prompt": "subject",
                "scope": "instance",
                "n_members": 1,
            }),
            encoding="utf-8",
        )
        self.db = self.root / "global.sqlite3"
        archive_reader._SHARED.clear()

    def tearDown(self) -> None:
        os.environ.pop(archive_reader.CATALOG_ENV, None)
        archive_reader._SHARED.clear()
        self.tmp.cleanup()

    def _inventory(self) -> object:
        return build_inventory(
            self.cache,
            "postgresql:///unused",
            workers=1,
            connection_factory=lambda dsn: (_ for _ in ()).throw(RuntimeError("no postgres")),
        )

    def test_local_pool_is_found_before_migration(self) -> None:
        result = self._inventory()
        self.assertEqual(len(result.eligible), 1)
        self.assertEqual(result.eligible[0].source_id, "src_test_0001")

    def test_pool_survives_after_cache_and_source_are_archived(self) -> None:
        # 落 cache 与源图到归档（两个组，键各自独立）
        out = self.root / "archive"
        for staging, group in ((self.cache, "cache/subject"), (self.source_image.parent, "img/unknown/test")):
            land(
                staging,
                group,
                out,
                plan_root=self.root / "plans",
                meta_staging=self.root / "meta",
                shard_size_bytes=1024**2,
                keep_staging=True,
            )
        rebuild(out, self.db)
        # 用环境变量把目录指向本测试自己的 catalog（默认值是运行时解析的）
        os.environ[archive_reader.CATALOG_ENV] = str(self.db)
        archive_reader._SHARED.clear()

        # 归档后仍能用本机副本
        before = self._inventory()
        self.assertEqual(len(before.eligible), 1)

        # 删掉本机 cache 与源图：发现、门控、解码全部必须改走归档
        (self.entry / "subject.json").unlink()
        (self.entry / "subject.png").unlink()
        self.entry.rmdir()
        self.cache.rmdir()
        self.source_image.unlink()
        self.assertFalse(self.cache.exists())

        after = self._inventory()
        self.assertEqual(len(after.eligible), 1, after.counts)
        record = after.eligible[0]
        self.assertEqual(record.source_id, "src_test_0001")
        self.assertAlmostEqual(record.mask_area, before.eligible[0].mask_area, places=6)


if __name__ == "__main__":
    unittest.main()
