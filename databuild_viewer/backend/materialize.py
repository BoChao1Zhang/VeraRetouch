"""Index-first, bounded materialization for archived viewer assets."""
from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dataset_build.tools.archive_reader import prefetch_name
from dataset_build.tools.prefetch import PrefetchResult, prefetch


PROGRESS_SCHEMA_VERSION = 1
RUNNING_STATES = frozenset({"queued", "locating", "materializing"})
TERMINAL_STATES = frozenset({"ready", "failed"})
MAX_RETAINED_JOBS = 64
MAX_OUTSTANDING_JOBS = 8


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def collect_asset_paths(detail: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the selected group's unique image assets in display order."""
    result: list[str] = []

    def add(value: Any) -> None:
        if isinstance(value, str) and value and value not in result:
            result.append(value)

    group = detail.get("group")
    if isinstance(group, Mapping):
        add(group.get("source_path"))
    candidates = detail.get("candidates")
    if isinstance(candidates, Sequence) and not isinstance(candidates, (str, bytes)):
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                add(candidate.get("after_path"))
                add(candidate.get("cgt_path"))
    sft_rows = detail.get("sft")
    if isinstance(sft_rows, Sequence) and not isinstance(sft_rows, (str, bytes)):
        for row in sft_rows:
            if not isinstance(row, Mapping):
                continue
            add(row.get("I_tar") or row.get("target_path"))
            local = row.get("local")
            if isinstance(local, Mapping):
                add(local.get("C_GT"))
    return tuple(result)


@dataclass
class _Job:
    group_id: str
    build_id: str
    paths: tuple[str, ...]
    state: str = "queued"
    files_done: int = 0
    files_total: int = 0
    bytes_done: int = 0
    bytes_total: int = 0
    current_item: str | None = None
    message: str | None = None
    updated_at: str = field(default_factory=_timestamp)
    capacity_checked: bool = False
    expected: dict[str, tuple[int, str]] = field(default_factory=dict)
    future: Future[None] | None = None

    @property
    def key(self) -> str:
        return f"{self.build_id}\0{self.group_id}"

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "build_id": self.build_id,
            "group_id": self.group_id,
            "state": self.state,
            "files": {"done": self.files_done, "total": self.files_total},
            "bytes": {"done": self.bytes_done, "total": self.bytes_total},
            "current_item": self.current_item,
            "message": self.message,
            "updated_at": self.updated_at,
        }


class MaterializationManager:
    """Own deduplicated prepare jobs and one hard-bounded flat member cache."""

    def __init__(
        self,
        cache_root: str | os.PathLike[str],
        max_bytes: int,
        db_path: str | os.PathLike[str],
        *,
        prefetch_fn: Callable[..., dict[str, Path]] = prefetch,
    ) -> None:
        if max_bytes <= 0:
            raise ValueError("materialization cache limit must be positive")
        self.cache_root = Path(cache_root).expanduser().resolve()
        self.member_dir = self.cache_root / "members"
        self.max_bytes = int(max_bytes)
        self.db_path = Path(db_path).expanduser().resolve()
        self._prefetch = prefetch_fn
        self._lock = threading.RLock()
        self._close_lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._closed = threading.Event()
        self._owner_fd: int | None = None
        self._executor: ThreadPoolExecutor | None = None
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.member_dir.mkdir(parents=True, exist_ok=True)
        self._owner_path = self.cache_root / ".owner.lock"

        owner_fd = os.open(self._owner_path, os.O_RDWR | os.O_CREAT, 0o600)
        locked = False
        try:
            try:
                fcntl.flock(owner_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    f"materialization cache already has an owner: {self.cache_root}"
                ) from exc
            locked = True
            self._owner_fd = owner_fd
            # members/ is exclusively viewer-owned and disposable. Any dot file
            # is an unpublished residue from a current, legacy, or unknown writer.
            self._cleanup_stale_partials()
            if self._usage() > self.max_bytes:
                self._ensure_capacity(0, set(), include_active=False)
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="dbv-materialize"
            )
        except BaseException:
            executor = self._executor
            self._executor = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            if locked:
                try:
                    fcntl.flock(owner_fd, fcntl.LOCK_UN)
                except OSError:
                    pass
            os.close(owner_fd)
            self._owner_fd = None
            self._closed.set()
            raise

    def close(self) -> None:
        with self._close_lock:
            if self._closed.is_set():
                return
            self._closed.set()
            with self._lock:
                for job in self._jobs.values():
                    if job.state in RUNNING_STATES:
                        job.state = "failed"
                        job.message = "materialization manager closed"
                        job.updated_at = _timestamp()
            executor = self._executor
            self._executor = None
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            owner_fd = self._owner_fd
            self._owner_fd = None
            if owner_fd is not None:
                try:
                    fcntl.flock(owner_fd, fcntl.LOCK_UN)
                finally:
                    os.close(owner_fd)

    def prepare(
        self,
        group_id: str,
        paths: Sequence[str],
        *,
        build_id: str,
        retry: bool = False,
    ) -> dict[str, Any]:
        unique = tuple(dict.fromkeys(str(path) for path in paths))
        if not build_id:
            raise ValueError("build_id is required")
        if not unique:
            raise ValueError("group has no materializable image assets")
        key = f"{build_id}\0{group_id}"
        with self._lock:
            if self._closed.is_set():
                raise RuntimeError("materialization manager is closed")
            existing = self._jobs.get(key)
            if existing is not None and existing.paths == unique:
                if existing.state in RUNNING_STATES:
                    return existing.snapshot()
                if existing.state == "ready" and self._all_available(existing):
                    return existing.snapshot()
                if existing.state == "failed" and not retry:
                    return existing.snapshot()
            self._prune_jobs()
            outstanding = sum(job.state in RUNNING_STATES for job in self._jobs.values())
            if outstanding >= MAX_OUTSTANDING_JOBS:
                raise RuntimeError("materialization queue is full")
            job = _Job(group_id=group_id, build_id=build_id, paths=unique)
            self._jobs[key] = job
            executor = self._executor
            if executor is None:
                raise RuntimeError("materialization manager is closed")
            job.future = executor.submit(self._run, job)
            return job.snapshot()

    def status(self, group_id: str, *, build_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(f"{build_id}\0{group_id}")
            return None if job is None else job.snapshot()

    def read_cached(self, logical_path: str) -> bytes | None:
        target = self.member_dir / prefetch_name(logical_path)
        with self._lock:
            expected = None
            owners: list[_Job] = []
            for job in self._jobs.values():
                if job.state == "ready" and logical_path in job.expected:
                    expected = job.expected[logical_path]
                    owners.append(job)
            if expected is None:
                return None
            try:
                stat = target.stat()
                if stat.st_size != expected[0] or _digest(target) != expected[1]:
                    raise OSError("cached digest mismatch")
                payload = target.read_bytes()
                os.utime(target, None)
                return payload
            except OSError:
                target.unlink(missing_ok=True)
                for job in owners:
                    job.state = "failed"
                    job.message = "validated cache entry was lost or corrupted"
                    job.updated_at = _timestamp()
                return None

    def cache_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "root": str(self.cache_root),
                "bytes": self._usage(),
                "max_bytes": self.max_bytes,
                "jobs": len(self._jobs),
            }

    def _is_current(self, job: _Job) -> bool:
        return self._jobs.get(job.key) is job

    def _cancelled(self, job: _Job) -> bool:
        return self._closed.is_set() or not self._is_current(job)

    def _set_state(self, job: _Job, state: str, **values: Any) -> None:
        with self._lock:
            if not self._is_current(job):
                return
            if self._closed.is_set() and state == "ready":
                return
            job.state = state
            for key, value in values.items():
                setattr(job, key, value)
            job.updated_at = _timestamp()

    def _progress(self, job: _Job, event: Mapping[str, object]) -> None:
        state = str(event.get("phase") or "materializing")
        with self._lock:
            if self._cancelled(job):
                raise InterruptedError("materialization cancelled")
            if state == "materializing" and not job.capacity_checked:
                needed = int(event.get("bytes_needed") or 0)
                protected = {prefetch_name(path) for path in job.paths}
                self._ensure_capacity(needed, protected)
                job.capacity_checked = True
            job.state = state
            job.files_done = int(event.get("files_done") or 0)
            job.files_total = int(event.get("files_total") or 0)
            job.bytes_done = int(event.get("bytes_done") or 0)
            job.bytes_total = int(event.get("bytes_total") or 0)
            current = event.get("current_item")
            job.current_item = str(current) if current else None
            job.updated_at = _timestamp()

    def _run(self, job: _Job) -> None:
        try:
            with self._lock:
                if self._cancelled(job):
                    return
                self._ensure_capacity(0, set(), include_active=False)
            result = self._prefetch(
                list(job.paths),
                self.member_dir,
                db_path=self.db_path,
                progress=lambda event: self._progress(job, event),
                strict=True,
                cancelled=lambda: self._cancelled(job),
            )
            with self._lock:
                if self._cancelled(job):
                    return
                records = getattr(result, "records", None)
                if not isinstance(records, dict):
                    raise RuntimeError("strict prefetch did not return verified index metadata")
                job.expected = dict(records)
                if not self._all_available(job):
                    raise RuntimeError("materialization completed without verified cache entries")
                usage = self._usage()
                if usage > self.max_bytes:
                    raise RuntimeError(f"cache limit exceeded after publish: {usage} > {self.max_bytes}")
                job.state = "ready"
                job.files_done = job.files_total
                job.bytes_done = job.bytes_total
                job.current_item = None
                job.message = None
                job.updated_at = _timestamp()
        except Exception as exc:
            self._discard_partials(job.paths)
            self._set_state(job, "failed", current_item=None, message=f"{type(exc).__name__}: {exc}")

    def _all_available(self, job: _Job) -> bool:
        for logical_path in job.paths:
            try:
                if os.stat(logical_path).st_size > 0:
                    continue
            except OSError:
                pass
            expected = job.expected.get(logical_path)
            if expected is None:
                return False
            target = self.member_dir / prefetch_name(logical_path)
            try:
                if target.stat().st_size != expected[0] or _digest(target) != expected[1]:
                    return False
            except OSError:
                return False
        return True

    def _cleanup_stale_partials(self) -> None:
        for path in self.member_dir.iterdir():
            if path.name.startswith(".") and (path.is_file() or path.is_symlink()):
                path.unlink(missing_ok=True)

    def _discard_partials(self, paths: Sequence[str]) -> None:
        if not self.member_dir.is_dir():
            return
        for logical_path in paths:
            digest = prefetch_name(logical_path)
            for partial in self.member_dir.glob(f".{digest}.*.tmp"):
                if partial.is_file() or partial.is_symlink():
                    partial.unlink(missing_ok=True)

    def _usage(self) -> int:
        total = 0
        for path in self.member_dir.iterdir():
            if path.is_file():
                try:
                    total += path.stat().st_size
                except OSError:
                    pass
        return total

    def _active_names(self) -> set[str]:
        result: set[str] = set()
        for job in self._jobs.values():
            if job.state in RUNNING_STATES:
                result.update(prefetch_name(path) for path in job.paths)
        return result

    def _ensure_capacity(
        self, bytes_needed: int, protected: set[str], *, include_active: bool = True
    ) -> None:
        if bytes_needed > self.max_bytes:
            raise RuntimeError(f"selected group needs {bytes_needed} cache bytes, limit is {self.max_bytes}")
        usage = self._usage()
        need_to_free = max(0, usage + bytes_needed - self.max_bytes)
        if not need_to_free:
            return
        pinned = protected | (self._active_names() if include_active else set())
        candidates: list[tuple[int, Path, int]] = []
        for path in self.member_dir.iterdir():
            if not path.is_file() or path.name.startswith(".") or path.name in pinned:
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            candidates.append((stat.st_mtime_ns, path, stat.st_size))
        freed = 0
        for _mtime, path, size in sorted(candidates):
            try:
                path.unlink()
            except FileNotFoundError:
                continue
            freed += size
            if freed >= need_to_free:
                break
        if freed < need_to_free:
            raise RuntimeError(f"cache limit cannot be met: need to free {need_to_free} bytes, freed {freed}")

    def _prune_jobs(self) -> None:
        terminal = sorted(
            (job for job in self._jobs.values() if job.state in TERMINAL_STATES),
            key=lambda job: job.updated_at,
        )
        remove_count = max(0, len(self._jobs) - MAX_RETAINED_JOBS + 1)
        for job in terminal[:remove_count]:
            self._jobs.pop(job.key, None)


__all__ = [
    "MaterializationManager",
    "PROGRESS_SCHEMA_VERSION",
    "RUNNING_STATES",
    "TERMINAL_STATES",
    "collect_asset_paths",
]
