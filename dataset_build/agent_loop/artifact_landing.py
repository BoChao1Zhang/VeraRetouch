"""Accepted-only artifact publication through the repository indexed-tar gate."""
from __future__ import annotations

import fcntl
import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from dataset_build.tools.archive_reader import ArchiveReader
from dataset_build.tools.global_catalog import upsert
from dataset_build.tools.land import land

from .artifacts import ArtifactStore
from .config import ArtifactLandingConfig
from .persistence import AuditStore


_MEDIA_EXTENSIONS = {
    "application/json": ".json",
    "image/jpeg": ".jpg",
    "image/png": ".png",
}


class ArtifactLandingManager:
    def __init__(
        self, store: ArtifactStore, audit: AuditStore, config: ArtifactLandingConfig,
    ) -> None:
        if not config.enabled:
            raise ValueError("artifact landing manager requires enabled configuration")
        self.store = store
        self.audit = audit
        self.config = config
        self._completed_groups = 0
        self._last_land = time.monotonic()
        self._landing_root = self.store.root / ".land"
        self._landing_root.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.store.root / ".landing.lock"

    def maybe_land(
        self, *, completed_groups: int = 0, force: bool = False,
    ) -> dict[str, Any] | None:
        self._completed_groups += max(0, int(completed_groups))
        with self._lock_path.open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            self._recover_publications()
            pending = self._pending_accepted()
            if not pending:
                return None
            pending_bytes = sum(int(row["size"]) for row in pending)
            free_bytes = shutil.disk_usage(self.store.root).free
            cadence_due = (
                pending_bytes >= self.config.watermark_bytes
                and self._completed_groups >= self.config.min_groups
                and time.monotonic() - self._last_land >= self.config.min_interval_seconds
            )
            if not force and free_bytes >= self.config.free_bytes_floor and not cadence_due:
                return None
            result = self._land(pending)
            self._completed_groups = 0
            self._last_land = time.monotonic()
            return result

    def _pending_accepted(self) -> list[dict[str, Any]]:
        rows = [
            dict(row) for row in self.audit.artifacts_with_retention("accepted")
            if self.store.local_path(str(row["sha256"])).is_file()
        ]
        catalog = self.config.catalog_db
        if not rows or catalog is None or not catalog.is_file():
            return rows
        pending = []
        with ArchiveReader(catalog) as reader:
            for row in rows:
                digest = str(row["sha256"])
                try:
                    located = reader.locate(f"sha256://{digest}")
                except KeyError:
                    pending.append(row)
                    continue
                if str(located["sha256"]) != digest:
                    raise RuntimeError(f"artifact archive index mismatch: {digest}")
                self.store.local_path(digest).unlink(missing_ok=True)
        return pending

    def _land(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        staging = self._landing_root / f"landing-{uuid.uuid4().hex}"
        staging.mkdir()
        source_paths: dict[str, str] = {}
        digests = []
        try:
            for row in rows:
                digest = str(row["sha256"])
                extension = _MEDIA_EXTENSIONS.get(str(row.get("media_type") or ""), ".bin")
                linked = staging / f"{digest}{extension}"
                linked.hardlink_to(self.store.local_path(digest))
                source_paths[str(linked)] = f"sha256://{digest}"
                digests.append(digest)
            result = land(
                staging, str(self.config.archive_group), Path(self.config.archive_root),
                plan_root=Path(self.config.plan_root),
                meta_staging=Path(self.config.meta_staging),
                source_paths=source_paths, keep_staging=True,
            )
            receipt = {
                "schema": "agent-artifact-landing-v1",
                "group": result["group"], "digests": digests,
            }
            (staging / ".publication.json").write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            self._finish_publication(staging, receipt)
            return {**result, "accepted_artifacts": len(digests)}
        except BaseException:
            if not (staging / ".publication.json").is_file():
                shutil.rmtree(staging, ignore_errors=True)
            raise

    def _recover_publications(self) -> None:
        for staging in sorted(self._landing_root.glob("landing-*")):
            receipt_path = staging / ".publication.json"
            if not receipt_path.is_file():
                shutil.rmtree(staging, ignore_errors=True)
                continue
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self._finish_publication(staging, receipt)

    def _finish_publication(self, staging: Path, receipt: Mapping[str, Any]) -> None:
        archive_root = Path(self.config.archive_root)
        catalog = Path(self.config.catalog_db)
        group = str(receipt["group"])
        digests = [str(value) for value in receipt["digests"]]
        upsert(archive_root, catalog, [group])
        with ArchiveReader(catalog) as reader:
            for digest in digests:
                located = reader.locate(f"sha256://{digest}")
                if str(located["sha256"]) != digest:
                    raise RuntimeError(f"artifact archive index mismatch: {digest}")
        for digest in digests:
            self.store.local_path(digest).unlink(missing_ok=True)
        shutil.rmtree(staging)


__all__ = ["ArtifactLandingManager"]
