from __future__ import annotations

import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from dataset_build.tools.archive_reader import prefetch_name
from dataset_build.tools.prefetch import PrefetchResult
from databuild_viewer.backend.materialize import MaterializationManager, collect_asset_paths


class MaterializationManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        self.catalog = self.root / "catalog.sqlite3"
        self.catalog.touch()
        self.paths = [str(self.root / f"retired-{index}.jpg") for index in range(3)]
        self.payloads = {path: bytes([index + 1]) * (100 + index) for index, path in enumerate(self.paths)}
        self.managers: list[MaterializationManager] = []

    def tearDown(self) -> None:
        for manager in self.managers:
            manager.close()
        self.temp.cleanup()

    def manager(self, fake, *, limit: int = 4096, cache: Path | None = None) -> MaterializationManager:
        manager = MaterializationManager(
            cache or self.cache,
            limit,
            self.catalog,
            prefetch_fn=fake,
        )
        self.managers.append(manager)
        return manager

    def wait(self, manager: MaterializationManager, group_id: str, build_id: str = "build-a") -> dict:
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            snapshot = manager.status(group_id, build_id=build_id)
            if snapshot and snapshot["state"] in ("ready", "failed"):
                return snapshot
            time.sleep(0.01)
        self.fail(f"materialization did not finish: {manager.status(group_id, build_id=build_id)}")

    def successful_fake(self, calls: list | None = None):
        def fake(paths, dest, *, db_path, progress, strict, cancelled):
            if calls is not None:
                calls.append((tuple(paths), Path(db_path), strict))
            total = sum(len(self.payloads[path]) for path in paths)
            progress({
                "phase": "locating", "files_done": 0, "files_total": len(paths),
                "bytes_done": 0, "bytes_total": 0, "current_item": None,
            })
            progress({
                "phase": "materializing", "files_done": 0, "files_total": len(paths),
                "bytes_done": 0, "bytes_total": total, "bytes_needed": total,
                "current_item": None,
            })
            result = PrefetchResult()
            done = 0
            for path in paths:
                if cancelled():
                    raise InterruptedError("cancelled")
                payload = self.payloads[path]
                target = Path(dest) / prefetch_name(path)
                target.write_bytes(payload)
                done += len(payload)
                result[path] = target
                result.records[path] = (len(payload), hashlib.sha256(payload).hexdigest())
                progress({
                    "phase": "materializing", "files_done": len(result),
                    "files_total": len(paths), "bytes_done": done, "bytes_total": total,
                    "bytes_needed": total, "current_item": path,
                })
            return result
        return fake

    def test_success_uses_strict_catalog_and_exposes_versioned_progress(self) -> None:
        calls: list = []
        manager = self.manager(self.successful_fake(calls))
        queued = manager.prepare("group-a", self.paths, build_id="build-a")
        self.assertEqual(1, queued["schema_version"])
        self.assertEqual("queued", queued["state"])
        ready = self.wait(manager, "group-a")
        self.assertEqual("ready", ready["state"])
        self.assertEqual({"done": 3, "total": 3}, ready["files"])
        self.assertEqual(303, ready["bytes"]["total"])
        self.assertEqual([(tuple(self.paths), self.catalog.resolve(), True)], calls)
        self.assertEqual(self.payloads[self.paths[1]], manager.read_cached(self.paths[1]))

    def test_concurrent_same_group_requests_share_one_job(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        calls = 0

        def fake(paths, dest, *, db_path, progress, strict, cancelled):
            nonlocal calls
            calls += 1
            progress({
                "phase": "locating", "files_done": 0, "files_total": len(paths),
                "bytes_done": 0, "bytes_total": 0, "current_item": None,
            })
            entered.set()
            self.assertTrue(release.wait(2))
            return self.successful_fake()(
                paths, dest, db_path=db_path, progress=progress, strict=strict, cancelled=cancelled
            )

        manager = self.manager(fake)
        first = manager.prepare("same", self.paths, build_id="build-a")
        self.assertTrue(entered.wait(2))
        second = manager.prepare("same", self.paths, build_id="build-a")
        self.assertIn(first["state"], ("queued", "locating"))
        self.assertEqual("locating", second["state"])
        release.set()
        self.assertEqual("ready", self.wait(manager, "same")["state"])
        self.assertEqual(1, calls)

    def test_failure_removes_partial_and_requires_explicit_retry(self) -> None:
        attempts = 0

        def fake(paths, dest, *, db_path, progress, strict, cancelled):
            nonlocal attempts
            attempts += 1
            digest = prefetch_name(paths[0])
            Path(dest).mkdir(parents=True, exist_ok=True)
            (Path(dest) / f".{digest}.123.{('a' * 32)}.tmp").write_bytes(b"partial")
            progress({
                "phase": "materializing", "files_done": 0, "files_total": len(paths),
                "bytes_done": 0, "bytes_total": 100, "bytes_needed": 100,
                "current_item": paths[0],
            })
            raise OSError("fixture read failed")

        manager = self.manager(fake)
        manager.prepare("broken", self.paths, build_id="build-a")
        failed = self.wait(manager, "broken")
        self.assertEqual("failed", failed["state"])
        self.assertIn("fixture read failed", failed["message"])
        self.assertFalse(list((self.cache / "members").glob(".*.tmp")))
        manager.prepare("broken", self.paths, build_id="build-a")
        time.sleep(0.02)
        self.assertEqual(1, attempts)
        manager.prepare("broken", self.paths, build_id="build-a", retry=True)
        self.wait(manager, "broken")
        self.assertEqual(2, attempts)

    def test_cache_limit_cleans_all_dot_residue_evicts_and_rejects_oversize(self) -> None:
        member_dir = self.cache / "members"
        member_dir.mkdir(parents=True)
        residues = (
            member_dir / f".{prefetch_name(self.paths[0])}.99.{('b' * 32)}.tmp",
            member_dir / f".{prefetch_name(self.paths[1])}.123.tmp",
            member_dir / ".unknown-v0-partial",
        )
        for index, residue in enumerate(residues, start=1):
            residue.write_bytes(bytes([index]) * 50)
        old = member_dir / "old-complete"
        old.write_bytes(b"x" * 250)
        manager = self.manager(self.successful_fake(), limit=300)
        self.assertTrue(all(not residue.exists() for residue in residues))
        manager.prepare("fits-after-eviction", [self.paths[0]], build_id="build-a")
        self.assertEqual("ready", self.wait(manager, "fits-after-eviction")["state"])
        self.assertFalse(old.exists())
        self.assertLessEqual(manager.cache_status()["bytes"], 300)

        tiny_cache = self.root / "tiny-cache"
        oversized = self.manager(self.successful_fake(), limit=50, cache=tiny_cache)
        oversized.prepare("too-large", [self.paths[0]], build_id="build-a")
        failed = self.wait(oversized, "too-large")
        self.assertEqual("failed", failed["state"])
        self.assertIn("limit is 50", failed["message"])
        self.assertLessEqual(oversized.cache_status()["bytes"], 50)

    def test_close_waits_for_worker_and_closed_job_never_becomes_ready(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def fake(paths, dest, *, db_path, progress, strict, cancelled):
            progress({
                "phase": "materializing", "files_done": 0, "files_total": 1,
                "bytes_done": 0, "bytes_total": 100, "bytes_needed": 100,
                "current_item": paths[0],
            })
            entered.set()
            release.wait(2)
            if cancelled():
                raise InterruptedError("cancelled")
            return self.successful_fake()(
                paths, dest, db_path=db_path, progress=progress, strict=strict, cancelled=cancelled
            )

        manager = self.manager(fake)
        manager.prepare("closing", [self.paths[0]], build_id="build-a")
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=manager.close)
        closer.start()
        time.sleep(0.05)
        self.assertTrue(closer.is_alive())
        release.set()
        closer.join(2)
        self.assertFalse(closer.is_alive())
        snapshot = manager.status("closing", build_id="build-a")
        self.assertEqual("failed", snapshot["state"])
        self.assertIsNone(manager.read_cached(self.paths[0]))

    def test_cache_root_rejects_second_owner_until_first_closes(self) -> None:
        first = self.manager(self.successful_fake())
        with self.assertRaisesRegex(RuntimeError, "already has an owner"):
            MaterializationManager(self.cache, 4096, self.catalog, prefetch_fn=self.successful_fake())
        first.close()
        replacement = self.manager(self.successful_fake())
        replacement.close()

    def test_constructor_failure_releases_owner_and_file_descriptor(self) -> None:
        fd_dir = Path("/proc/self/fd")
        before = len(tuple(fd_dir.iterdir())) if fd_dir.is_dir() else None
        with mock.patch.object(
            MaterializationManager,
            "_cleanup_stale_partials",
            side_effect=RuntimeError("forced initialization failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "forced initialization failure"):
                MaterializationManager(
                    self.cache, 4096, self.catalog, prefetch_fn=self.successful_fake()
                )
        if before is not None:
            self.assertEqual(before, len(tuple(fd_dir.iterdir())))

        replacement = self.manager(self.successful_fake())
        replacement.prepare("recovered", [self.paths[0]], build_id="build-a")
        self.assertEqual("ready", self.wait(replacement, "recovered")["state"])

    def test_ready_digest_corruption_is_not_served(self) -> None:
        manager = self.manager(self.successful_fake())
        manager.prepare("digest", [self.paths[0]], build_id="build-a")
        self.assertEqual("ready", self.wait(manager, "digest")["state"])
        target = self.cache / "members" / prefetch_name(self.paths[0])
        target.write_bytes(b"z" * len(self.payloads[self.paths[0]]))
        self.assertIsNone(manager.read_cached(self.paths[0]))
        self.assertEqual("failed", manager.status("digest", build_id="build-a")["state"])

    def test_collect_asset_paths_deduplicates_source_candidates_masks_and_sft_targets(self) -> None:
        detail = {
            "group": {"source_path": "/archive/source.jpg"},
            "candidates": [
                {"after_path": "/archive/c1.jpg", "cgt_path": "/archive/mask.png"},
                {"after_path": "/archive/c2.jpg", "cgt_path": "/archive/mask.png"},
            ],
            "sft": [{
                "I_tar": "/archive/c1.jpg",
                "I_in": "/archive/preview.in.jpg",
                "local": {"C_GT": "/archive/mask.png"},
            }],
        }
        self.assertEqual(
            ("/archive/source.jpg", "/archive/c1.jpg", "/archive/mask.png", "/archive/c2.jpg"),
            collect_asset_paths(detail),
        )


if __name__ == "__main__":
    unittest.main()
