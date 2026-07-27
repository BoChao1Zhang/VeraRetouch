"""Durable JSONL and manifest state for canonical databuild resume."""
from __future__ import annotations

import copy
import errno
import fcntl
import hashlib
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PHASES = (
    "preflight",
    "import",
    "rendering",
    "sam3_relabel",
    "annotation",
    "projection",
    "complete",
    "complete_with_failures",
)
TERMINAL_STATUSES = frozenset({"complete", "complete_with_failures"})


class StateError(RuntimeError):
    """Raised when durable state is corrupt or violates an output invariant."""


def stable_id(kind: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in (kind, *parts)).encode("utf-8")
    return f"{kind}_{hashlib.sha256(payload).hexdigest()[:32]}"


@dataclass(frozen=True, slots=True)
class JsonlScan:
    records: tuple[dict[str, Any], ...]
    valid_bytes: int
    torn_tail: bool


def scan_jsonl(path: str | os.PathLike[str]) -> JsonlScan:
    """Read JSONL, tolerating only one malformed final record."""
    target = Path(path)
    if not target.exists():
        return JsonlScan((), 0, False)
    records: list[dict[str, Any]] = []
    valid_end = 0
    size = target.stat().st_size
    with target.open("rb") as handle:
        line_number = 0
        while True:
            start = handle.tell()
            raw = handle.readline()
            if not raw:
                break
            line_number += 1
            end = handle.tell()
            if not raw.strip():
                valid_end = end
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if end == size:
                    return JsonlScan(tuple(records), start, True)
                raise StateError(
                    f"malformed non-tail JSONL record at {target}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                if end == size:
                    return JsonlScan(tuple(records), start, True)
                raise StateError(f"JSONL record at {target}:{line_number} is not an object")
            records.append(value)
            valid_end = end
    return JsonlScan(tuple(records), valid_end, False)


class JsonlJournal:
    """Serialized append journal with bounded fsync checkpoints."""

    def __init__(self, path: str | os.PathLike[str], fsync_every: int = 32):
        if fsync_every <= 0:
            raise ValueError("fsync_every must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        scan = scan_jsonl(self.path)
        if scan.torn_tail:
            with self.path.open("r+b") as repair:
                repair.truncate(scan.valid_bytes)
                repair.flush()
                os.fsync(repair.fileno())
        self._handle = self.path.open("ab", buffering=0)
        self._lock = threading.Lock()
        self._pending = 0
        self._fsync_every = fsync_every

    def append(self, record: dict[str, Any], *, durable: bool = False) -> None:
        if not isinstance(record, dict):
            raise TypeError("JSONL record must be a dict")
        payload = (json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ) + "\n").encode("utf-8")
        with self._lock:
            self._handle.write(payload)
            self._pending += 1
            if durable or self._pending >= self._fsync_every:
                os.fsync(self._handle.fileno())
                self._pending = 0

    def checkpoint(self) -> None:
        with self._lock:
            if self._pending:
                os.fsync(self._handle.fileno())
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            if self._handle.closed:
                return
            try:
                if self._pending:
                    os.fsync(self._handle.fileno())
            finally:
                self._pending = 0
                self._handle.close()

    def __enter__(self) -> "JsonlJournal":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def write_json_atomic(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )
    with tmp.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def file_digest(path: str | os.PathLike[str]) -> dict[str, Any]:
    target = Path(path)
    digest = hashlib.sha256()
    count = 0
    with target.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            count += chunk.count(b"\n")
    return {"sha256": digest.hexdigest(), "bytes": target.stat().st_size, "records": count}


class ArtifactStore:
    """Authoritative four-artifact store and idempotent resume indexes."""

    def __init__(self, output_root: str | os.PathLike[str], build_id: str, fsync_every: int = 32):
        self.root = Path(output_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.groups_path = self.root / "groups.jsonl"
        self.sft_path = self.root / "sft.jsonl"
        self.failures_path = self.root / "failures.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.assets_root = self.root / "assets"
        self.build_id = build_id
        self._lock = threading.RLock()
        self._directory_lock_fd: int | None = None
        self._groups_journal: JsonlJournal | None = None
        self._sft_journal: JsonlJournal | None = None
        self._failures_journal: JsonlJournal | None = None
        self._closed = False

        lock_fd = os.open(
            self.root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(lock_fd)
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise StateError(
                    f"output root is locked by another databuild process: {self.root}"
                ) from exc
            raise

        self._directory_lock_fd = lock_fd

        try:
            self.assets_root.mkdir(exist_ok=True)
            groups_scan = scan_jsonl(self.groups_path)
            sft_scan = scan_jsonl(self.sft_path)
            failures_scan = scan_jsonl(self.failures_path)
            self.groups = self._load_unique(
                groups_scan.records, "group_id", "group"
            )
            self.sft = self._load_unique(sft_scan.records, "sft_id", "SFT")
            self._failures_by_id = self._load_unique(
                failures_scan.records, "event_id", "failure event"
            )
            self.failures = list(self._failures_by_id.values())
            for group in self.groups.values():
                self._validate_group(group, durable=True)
            self._groups_journal = JsonlJournal(self.groups_path, fsync_every)
            self._sft_journal = JsonlJournal(self.sft_path, fsync_every)
            self._failures_journal = JsonlJournal(self.failures_path, fsync_every)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise StateError("artifact store is closed")

    def _load_unique(
        self,
        records: Iterable[dict[str, Any]],
        id_key: str,
        label: str,
    ) -> dict[str, dict[str, Any]]:
        indexed: dict[str, dict[str, Any]] = {}
        for row in records:
            if row.get("build_id") != self.build_id:
                raise StateError(f"durable {label} belongs to another build")
            record_id = row.get(id_key)
            if not isinstance(record_id, str) or not record_id:
                raise StateError(f"durable {label} requires {id_key}")
            if record_id in indexed:
                raise StateError(f"duplicate durable {label}: {record_id}")
            indexed[record_id] = row
        return indexed

    @staticmethod
    def _validate_group(record: dict[str, Any], *, durable: bool = False) -> None:
        prefix = "durable " if durable else ""
        candidates = record.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 8:
            raise StateError(f"{prefix}group is not an eight-candidate record")
        if any(not isinstance(row, dict) for row in candidates):
            raise StateError(f"{prefix}group candidates must be objects")
        candidate_ids = [row.get("candidate_id") for row in candidates]
        if any(not isinstance(candidate_id, str) or not candidate_id
               for candidate_id in candidate_ids) \
                or len(set(candidate_ids)) != 8:
            raise StateError(
                f"{prefix}group candidates require eight distinct non-empty string IDs"
            )
        winner_ids = record.get("winner_ids")
        if not isinstance(winner_ids, list):
            raise StateError(f"{prefix}group winner_ids must be a list")
        if len(winner_ids) > 2 or any(
            not isinstance(winner_id, str) or not winner_id for winner_id in winner_ids
        ) or len(set(winner_ids)) != len(winner_ids):
            raise StateError(
                f"{prefix}group winner_ids require at most two distinct string IDs"
            )
        if not set(winner_ids).issubset(candidate_ids):
            raise StateError(f"{prefix}group winner_ids must reference group candidates")

    def append_group(self, record: dict[str, Any]) -> bool:
        stored = copy.deepcopy(record)
        group_id = stored.get("group_id")
        if not isinstance(group_id, str) or not group_id \
                or stored.get("build_id") != self.build_id:
            raise StateError("group record has invalid group_id/build_id")
        self._validate_group(stored)
        with self._lock:
            self._ensure_open()
            existing = self.groups.get(group_id)
            if existing is not None:
                if existing != stored:
                    raise StateError(f"conflicting durable group record: {group_id}")
                return False
            assert self._groups_journal is not None
            self._groups_journal.append(stored)
            self.groups[group_id] = stored
            return True

    def append_sft(self, record: dict[str, Any]) -> bool:
        stored = copy.deepcopy(record)
        sft_id = stored.get("sft_id")
        if not isinstance(sft_id, str) or not sft_id \
                or stored.get("build_id") != self.build_id:
            raise StateError("SFT record has invalid sft_id/build_id")
        with self._lock:
            self._ensure_open()
            existing = self.sft.get(sft_id)
            if existing is not None:
                if existing != stored:
                    raise StateError(f"conflicting durable SFT record: {sft_id}")
                return False
            assert self._sft_journal is not None
            self._sft_journal.append(stored)
            self.sft[sft_id] = stored
            return True

    def append_failure(self, record: dict[str, Any], *, durable: bool = False) -> bool:
        stored = copy.deepcopy(record)
        event_id = stored.get("event_id")
        if not isinstance(event_id, str) or not event_id \
                or stored.get("build_id") != self.build_id:
            raise StateError("failure event has invalid event_id/build_id")
        with self._lock:
            self._ensure_open()
            existing = self._failures_by_id.get(event_id)
            if existing is not None:
                if existing != stored:
                    raise StateError(f"conflicting durable failure event: {event_id}")
                return False
            assert self._failures_journal is not None
            self._failures_journal.append(stored, durable=durable)
            self.failures.append(stored)
            self._failures_by_id[event_id] = stored
            return True

    def has_terminal_failure(self, task_id: str) -> bool:
        return any(
            row.get("task_id") == task_id and bool(row.get("terminal"))
            for row in self.failures
        )

    def external_pool_exhausted(self) -> bool:
        return any(row.get("error_code") == "external_pool_exhausted" for row in self.failures)

    def completed_sources(self) -> set[str]:
        return {str(row["source_id"]) for row in self.groups.values()}

    def completed_annotation_tasks(self) -> set[str]:
        completed = {
            str(row["annotation_task_id"])
            for row in self.sft.values()
            if row.get("annotation_task_id")
        }
        completed.update(
            str(row["task_id"])
            for row in self.failures
            if row.get("stage") == "annotation" and row.get("terminal") and row.get("task_id")
        )
        return completed

    def pending_annotation_tasks(self) -> list[dict[str, Any]]:
        done = self.completed_annotation_tasks()
        tasks: list[dict[str, Any]] = []
        for group in sorted(self.groups.values(), key=lambda row: str(row["group_id"])):
            candidates = {row["candidate_id"]: row for row in group["candidates"]}
            for rank, candidate_id in enumerate(group.get("winner_ids") or [], start=1):
                task_id = stable_id("annotation", group["group_id"], candidate_id, rank)
                if task_id in done:
                    continue
                tasks.append({
                    "task_id": task_id,
                    "group_id": group["group_id"],
                    "candidate_id": candidate_id,
                    "winner_rank": rank,
                    "group": group,
                    "candidate": candidates[candidate_id],
                })
        return tasks

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            stored = copy.deepcopy(manifest)
            if stored.get("build_id") != self.build_id:
                raise StateError("manifest build_id mismatch")
            phase = stored.get("phase")
            status = stored.get("status")
            if phase not in PHASES:
                raise StateError(f"invalid manifest phase: {phase!r}")
            if status not in {"running", *TERMINAL_STATUSES}:
                raise StateError(f"invalid manifest status: {status!r}")
            write_json_atomic(self.manifest_path, stored)

    def artifact_digests(self) -> dict[str, dict[str, Any]]:
        self.checkpoint()
        return {
            path.name: file_digest(path)
            for path in (self.groups_path, self.sft_path, self.failures_path, self.manifest_path)
            if path.exists()
        }

    def checkpoint(self) -> None:
        with self._lock:
            self._ensure_open()
            assert self._groups_journal is not None
            assert self._sft_journal is not None
            assert self._failures_journal is not None
            self._groups_journal.checkpoint()
            self._sft_journal.checkpoint()
            self._failures_journal.checkpoint()

    def close(self) -> None:
        errors: list[BaseException] = []
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for attribute in (
                "_groups_journal",
                "_sft_journal",
                "_failures_journal",
            ):
                journal = getattr(self, attribute)
                if journal is None:
                    continue
                try:
                    journal.close()
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    setattr(self, attribute, None)

            lock_fd = self._directory_lock_fd
            if lock_fd is not None:
                try:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    os.close(lock_fd)
                    self._directory_lock_fd = None
        if errors:
            raise errors[0]

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def count_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in records:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))
