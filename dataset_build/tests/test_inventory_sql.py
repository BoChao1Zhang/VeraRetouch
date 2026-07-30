"""预计算的 SQL 源池门控必须与 `_inspect_cache_dir` 逐条等价。

`build_inventory` 的旧路径对每条 cache 条目做完整解码门控（实测 56,777 条 ≈ 8 min
冷启动）。新路径把同一判据在 land 打包时算好写进 metadata，启动时一条 SQL 取回。
"逐条等价"是这套改动唯一的正确性依据，所以这里对同一批条目跑三条路径
（本机树旧实现 / 归档旧实现 / SQL 预计算）并逐字段比对。
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from construct import sources  # noqa: E402
from construct.sources import build_inventory  # noqa: E402

from dataset_build.tools import archive_reader  # noqa: E402
from dataset_build.tools.backfill_subject_meta import backfill  # noqa: E402
from dataset_build.tools.global_catalog import rebuild  # noqa: E402
from dataset_build.tools.land import enrich_for_group, land  # noqa: E402


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    array = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(array, mode="RGB").save(buffer, format="JPEG", quality=90)
    return buffer.getvalue()


def _mask_png(side: int, box: tuple[int, int, int, int]) -> bytes:
    mask = np.zeros((side, side), dtype=np.uint8)
    top, left, bottom, right = box
    mask[top:bottom, left:right] = 255
    buffer = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buffer, format="PNG")
    return buffer.getvalue()


class InventorySqlParityTests(unittest.TestCase):
    """三条路径（本机树 / 归档旧实现 / SQL 预计算）必须给出同一份源池。"""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        # land() resolves its staging root, so the fixture must compare against
        # resolved paths or a symlinked TMPDIR would fail the parity assertions.
        self.root = Path(self.tmp.name).resolve()
        self.cache = self.root / "subject_cache"
        self.images = self.root / "images"
        self.images.mkdir(parents=True)
        self.db = self.root / "global.sqlite3"
        self.archive = self.root / "archive"
        archive_reader._SHARED.clear()
        self._build_pool()

    def tearDown(self) -> None:
        os.environ.pop(archive_reader.CATALOG_ENV, None)
        archive_reader._SHARED.clear()
        self.tmp.cleanup()

    # ---------------------------------------------------------------- fixture
    def _entry(
        self,
        name: str,
        *,
        meta: object,
        mask: bytes | None,
        source: bytes | None = b"",
    ) -> None:
        """一个 cache 条目：meta 可以是 dict/裸 bytes，mask/source 可以缺失。"""
        entry = self.cache / name
        entry.mkdir(parents=True)
        if isinstance(meta, dict) and source is not None:
            image = self.images / f"{name}.jpg"
            image.write_bytes(source or _jpeg(48, 48, seed=len(name)))
            meta = {**meta, "source_path": str(image)}
        if isinstance(meta, (bytes, bytearray)):
            (entry / "subject.json").write_bytes(meta)
        else:
            (entry / "subject.json").write_text(json.dumps(meta), encoding="utf-8")
        if mask is not None:
            (entry / "subject.png").write_bytes(mask)

    def _build_pool(self) -> None:
        ready = {"status": "ready", "sam_prompt": "subject", "scope": "instance", "n_members": 1}
        good = _mask_png(64, (10, 10, 39, 39))
        # 合格：20% 与 6% 两种面积，落在 0.005~0.85 内
        self._entry("aaa0", meta={**ready, "asset_id": "asset_aaa0"}, mask=good)
        small = _mask_png(64, (0, 0, 16, 16))
        self._entry("aaa1", meta={**ready, "asset_id": "asset_aaa1"}, mask=small)
        # asset_id 缺失 → source_id 走 stable_id(realpath)
        self._entry("aaa2", meta=dict(ready), mask=_mask_png(64, (4, 4, 44, 44)))
        # 各类不合格
        self._entry("bad0", meta={"status": "selection_failed"}, mask=good)
        self._entry("bad1", meta=dict(ready), mask=None)
        self._entry("bad2", meta={**ready, "asset_id": "x"}, mask=_mask_png(64, (0, 0, 2, 2)))
        self._entry("bad3", meta={**ready, "asset_id": "y"}, mask=_mask_png(64, (0, 0, 62, 62)))
        self._entry("bad4", meta=b"{not json", mask=good)
        self._entry("bad5", meta={"status": "ready"}, mask=good, source=None)
        self._entry("bad6", meta=dict(ready), mask=b"\x89PNG\r\n\x1a\n garbage")
        # 源图不存在（写完就删）
        self._entry("bad7", meta=dict(ready), mask=good)
        (self.images / "bad7.jpg").unlink()
        # "_" 开头的目录不是 cache 条目，任何一条路径都不该统计它
        self._entry("_scratch", meta=dict(ready), mask=good)

    @staticmethod
    def _migrated_meta(sample_key: str, members: dict[str, Path]) -> dict[str, object]:
        """迁移时 `plan_derived` 写下的样本元数据：有成员映射，没有门控字段。"""
        return {
            "kind": "cache",
            "corpus": "subject",
            "members": {ext: str(path) for ext, path in members.items()},
        }

    def _land(self, *, enrich: bool) -> None:
        for staging, group in ((self.cache, "cache/subject"), (self.images, "img/unknown/test")):
            hook = enrich_for_group(group) if enrich else None
            if hook is None and group == "cache/subject":
                hook = self._migrated_meta
            land(
                staging,
                group,
                self.archive,
                plan_root=self.root / "plans",
                meta_staging=self.root / "meta",
                enrich=hook,
                shard_size_bytes=1024**2,
                keep_staging=True,
            )
        self._publish()

    def _publish(self) -> None:
        rebuild(self.archive, self.db)
        os.environ[archive_reader.CATALOG_ENV] = str(self.db)
        archive_reader._SHARED.clear()

    def _drop_local_tree(self) -> None:
        for path in sorted(self.cache.rglob("*"), reverse=True):
            path.unlink() if path.is_file() else path.rmdir()
        self.cache.rmdir()
        for image in sorted(self.images.iterdir()):
            image.unlink()
        self.assertFalse(self.cache.exists())

    def _inventory(self, **kwargs: object) -> object:
        return build_inventory(
            self.cache,
            "postgresql:///unused",
            workers=1,
            connection_factory=lambda dsn: (_ for _ in ()).throw(RuntimeError("no postgres")),
            **kwargs,
        )

    def _assert_same_pool(self, expected: object, actual: object) -> None:
        self.assertEqual(actual.counts, expected.counts)
        self.assertEqual(
            [row.source_id for row in actual.eligible],
            [row.source_id for row in expected.eligible],
        )
        for left, right in zip(actual.eligible, expected.eligible):
            self.assertAlmostEqual(left.mask_area, right.mask_area, delta=1e-6)
            self.assertEqual(left.source_path, right.source_path)
            self.assertEqual(left.cache_dir, right.cache_dir)
            self.assertEqual(left.subject_path, right.subject_path)
            self.assertEqual(left.subject_meta_path, right.subject_meta_path)
            self.assertEqual(left.scene, right.scene)
            self.assertEqual(left.subject, right.subject)

    @contextmanager
    def _counted_inspect(self):
        """记录旧实现被调用了多少次——这是"到底走了哪条路"的唯一硬证据。"""
        calls: list[Path] = []
        original = sources._inspect_cache_dir

        def spy(path, *args):
            calls.append(path)
            return original(path, *args)

        sources._inspect_cache_dir = spy
        try:
            yield calls
        finally:
            sources._inspect_cache_dir = original

    # ------------------------------------------------------------------ tests
    def test_sql_path_matches_the_decoding_gate_entry_for_entry(self) -> None:
        local = self._inventory()
        self.assertEqual(
            local.counts,
            {
                "cache_entries": 11,
                "eligible": 3,
                "subject_not_ready": 1,
                "missing_subject_png": 1,
                "subject_mask_area_guard": 2,
                "invalid_subject_json": 1,
                "missing_source_path": 1,
                "decode_or_integrity_failure": 1,
                "missing_source_image": 1,
            },
        )
        self._land(enrich=True)
        self._drop_local_tree()

        legacy = self._inventory(legacy_inspect=True)
        self._assert_same_pool(local, legacy)

        with self._counted_inspect() as calls:
            precomputed = self._inventory()
        self.assertEqual(calls, [], "SQL 路径不得回落到逐条解码")
        self._assert_same_pool(local, precomputed)

    def test_legacy_inspect_forces_the_reference_implementation(self) -> None:
        self._land(enrich=True)
        self._drop_local_tree()
        with self._counted_inspect() as calls:
            result = self._inventory(legacy_inspect=True)
        self.assertEqual(len(calls), 11)
        self.assertEqual(len(result.eligible), 3)

    def test_group_without_precomputed_fields_falls_back(self) -> None:
        self._land(enrich=False)
        self._drop_local_tree()
        legacy = self._inventory(legacy_inspect=True)
        with self._counted_inspect() as calls:
            fallback = self._inventory()
        self.assertEqual(len(calls), 11, "缺预计算字段必须回退旧路径")
        self._assert_same_pool(legacy, fallback)

    def test_backfill_makes_a_migrated_group_queryable(self) -> None:
        self._land(enrich=False)
        self._drop_local_tree()
        legacy = self._inventory(legacy_inspect=True)

        summary = backfill(self.archive, "cache/subject/batch-0000")
        self.assertEqual(summary["samples"], 12)
        self.assertEqual(summary["counts"]["eligible"], 3)
        self.assertEqual(summary["counts"]["not_a_cache_entry"], 1)
        self._publish()

        with self._counted_inspect() as calls:
            precomputed = self._inventory()
        self.assertEqual(calls, [])
        self._assert_same_pool(legacy, precomputed)


if __name__ == "__main__":
    unittest.main()


class SparseOverlayTests(InventorySqlParityTests):
    """relabel 重建的本机稀疏树是覆盖层：归档为底座、本机同源覆盖。"""

    def test_sparse_local_tree_overlays_the_precomputed_pool(self) -> None:
        self._land(enrich=True)
        self._drop_local_tree()
        baseline = self._inventory()
        base_ids = {r.source_id for r in baseline.eligible}
        overridden = baseline.eligible[0]
        # relabel 把一个归档合格源写回本机、状态未 ready → 应从池中剔除。
        stale = self.cache / overridden.cache_dir.name
        stale.mkdir(parents=True)
        (stale / "subject.json").write_text(
            json.dumps({"status": "selection_failed"}), encoding="utf-8"
        )
        # relabel 也可能产出一个全新 ready 条目 → 应并入池。
        self.images.mkdir(exist_ok=True)
        self._entry("zzz9", meta={
            "status": "ready", "sam_prompt": "subject", "scope": "instance",
            "n_members": 1, "asset_id": "asset_zzz9",
        }, mask=_mask_png(64, (8, 8, 40, 40)))
        merged = self._inventory()
        merged_ids = {r.source_id for r in merged.eligible}
        self.assertNotIn(overridden.source_id, merged_ids)
        self.assertEqual(len(merged_ids - base_ids), 1)
        self.assertEqual(merged.counts["local_overlay"], 2)
        self.assertEqual(merged.counts["overlay_subject_not_ready"], 1)
        self.assertEqual(merged.counts["overlay_eligible"], 1)
        self.assertEqual(merged.counts["eligible"], len(merged.eligible))
