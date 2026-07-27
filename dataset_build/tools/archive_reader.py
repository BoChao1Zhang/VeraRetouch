"""Read archived bytes using the path a call site already hardcodes.

Existing pipeline code addresses data by absolute source path.  After migration
those paths are gone, so this reader resolves a path through the global catalog's
reverse map and preads the member out of its uncompressed shard.

``read_bytes`` prefers a surviving local file, which keeps the pipeline working
unchanged while both copies exist and switches to the archive the moment the
local one is deleted.  Reads use ``os.pread`` so several dataloader threads can
share one shard descriptor without seek races.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import tarfile
from pathlib import Path
from threading import Lock
from typing import Mapping

from dataset_build.tools.indexed_tar import BLOCK_SIZE, IndexedTarError, _load_manifest


# Persistent by default: a catalog under /tmp silently vanishes on reboot, and
# although it is rebuildable, an operator should not discover that mid-run.  Set
# VERADATA_CATALOG to move it (e.g. onto the SSD).
DEFAULT_DB = Path("/var/cache/veradata/global.sqlite3")
CATALOG_ENV = "VERADATA_CATALOG"
MAX_OPEN_SHARDS = 64


def default_db() -> Path:
    """Resolve the catalog path at call time, not at import time.

    Binding it into default arguments would freeze whatever the module saw first,
    which breaks both operator overrides and tests that point at their own
    catalog.  ``VERADATA_CATALOG`` wins when set.
    """
    return Path(os.environ.get(CATALOG_ENV) or DEFAULT_DB)


class ArchiveReader:
    """Resolve archived members by their original absolute path."""

    def __init__(self, db_path: Path | None = None, *, verify_checksum: bool = False) -> None:
        db_path = Path(db_path) if db_path is not None else default_db()
        if not db_path.is_file():
            raise IndexedTarError(
                f"global catalog is missing: {db_path} "
                "(rebuild it with python -m dataset_build.tools.global_catalog)"
            )
        # immutable=1 是这里的关键：它让 SQLite 跳过全部加锁并直接 mmap，随机点查
        # 从 25 ms（每次落盘寻道）降到 15 µs。契约由重建流程保证——重建写
        # <name>.rebuilding 再 os.replace 原子替换，因此持有旧 fd 的读者只会看到
        # 旧快照，永远读不到撕裂状态；要看新数据重开 reader 即可。
        self._connection = sqlite3.connect(
            db_path.as_uri() + "?immutable=1", uri=True, check_same_thread=False
        )
        self._connection.execute("PRAGMA mmap_size=1073741824")
        self._connection.row_factory = sqlite3.Row
        self._verify_checksum = verify_checksum
        self._lock = Lock()
        self._descriptors: dict[str, int] = {}
        self._shard_paths: dict[tuple[str, str], str] = {}

    def close(self) -> None:
        with self._lock:
            for descriptor in self._descriptors.values():
                os.close(descriptor)
            self._descriptors.clear()
        self._connection.close()

    def __enter__(self) -> "ArchiveReader":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def locate(self, source_path: str | os.PathLike[str]) -> Mapping[str, object]:
        """Return the archived member row for an original source path."""
        with self._lock:
            row = self._connection.execute(
                'SELECT s."group" AS "group", s.sample_id, s.member, m.shard, m.offset_data, '
                "m.size, m.sha256, g.root "
                'FROM source_paths s JOIN members m ON m."group" = s."group" AND m.member = s.member '
                'JOIN groups g ON g."group" = s."group" WHERE s.source_path = ?',
                (str(source_path),),
            ).fetchone()
        if row is None:
            raise KeyError(str(source_path))
        return dict(row)

    def iter_source_paths(self, prefix: str, *, endswith: str = "") -> list[str]:
        """List archived original paths under a directory prefix.

        This is the archive's stand-in for scanning a directory: discovery code
        that used to walk a cache tree can enumerate the same paths after the
        local tree is gone.
        """
        # 范围比较而不是 LIKE：LIKE 默认大小写不敏感，SQLite 无法用索引，实测在
        # 445 万行上是全表扫 963 ms；范围比较走主键索引 8.3 ms（快 116 倍）。
        # 上界取前缀最后一个字符 +1，覆盖所有以该前缀开头的键。
        upper = prefix[:-1] + chr(ord(prefix[-1]) + 1) if prefix else ""
        with self._lock:
            rows = self._connection.execute(
                "SELECT source_path FROM source_paths "
                "WHERE source_path >= ? AND source_path < ? ORDER BY source_path",
                (prefix, upper),
            ).fetchall()
        return [
            str(row["source_path"])
            for row in rows
            if not endswith or str(row["source_path"]).endswith(endswith)
        ]

    def _shard_descriptor(self, root: str, shard: str) -> int:
        key = (root, shard)
        with self._lock:
            path = self._shard_paths.get(key)
            if path is None:
                manifest = _load_manifest(Path(root))
                for item in manifest["shards"]:
                    self._shard_paths[(root, str(item["shard_id"]))] = str(
                        Path(root) / str(item["tar"])
                    )
                path = self._shard_paths.get(key)
                if path is None:
                    raise IndexedTarError(f"unknown shard {shard} under {root}")
            descriptor = self._descriptors.get(path)
            if descriptor is None:
                if len(self._descriptors) >= MAX_OPEN_SHARDS:
                    stale_path, stale = next(iter(self._descriptors.items()))
                    os.close(stale)
                    del self._descriptors[stale_path]
                descriptor = os.open(path, os.O_RDONLY)
                self._descriptors[path] = descriptor
            return descriptor

    def read(self, source_path: str | os.PathLike[str]) -> bytes:
        """Read one member out of its shard, checking the tar header agrees."""
        row = self.locate(source_path)
        descriptor = self._shard_descriptor(str(row["root"]), str(row["shard"]))
        offset = int(row["offset_data"])
        size = int(row["size"])
        header = os.pread(descriptor, BLOCK_SIZE, offset - BLOCK_SIZE)
        try:
            info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
        except (tarfile.HeaderError, UnicodeError) as exc:
            raise IndexedTarError(f"invalid member header for {source_path}: {exc}") from exc
        if info.name != row["member"] or info.size != size:
            raise IndexedTarError(f"catalog does not match shard header for {source_path}")
        payload = b""
        while len(payload) < size:
            chunk = os.pread(descriptor, size - len(payload), offset + len(payload))
            if not chunk:
                raise IndexedTarError(f"truncated member for {source_path}")
            payload += chunk
        if self._verify_checksum and hashlib.sha256(payload).hexdigest() != row["sha256"]:
            raise IndexedTarError(f"payload checksum mismatch for {source_path}")
        return payload

    def read_bytes(self, source_path: str | os.PathLike[str]) -> bytes:
        """Read the local file while it exists, otherwise read the archive."""
        path = Path(source_path)
        try:
            return path.read_bytes()
        except FileNotFoundError:
            return self.read(source_path)

    def exists(self, source_path: str | os.PathLike[str]) -> bool:
        """Report whether a path is readable locally or from the archive.

        Pipeline code gates on ``Path.is_file()`` to decide whether an asset is
        usable, so those gates need this instead once the local copy is gone.
        """
        path = Path(source_path)
        if path.is_file() and path.stat().st_size > 0:
            return True
        try:
            return int(self.locate(source_path)["size"]) > 0
        except KeyError:
            return False


_SHARED: dict[str, ArchiveReader] = {}
_SHARED_LOCK = Lock()


def read_bytes(source_path: str | os.PathLike[str], *, db_path: Path | None = None) -> bytes:
    """Module-level local-first read for call sites that hold no reader.

    ponytail: one process-wide reader, created on first archive miss.  If the
    local file is still there this never touches SQLite at all, so wiring this
    into a hot path costs nothing until the migration actually removes the file.
    """
    path = Path(source_path)
    try:
        return path.read_bytes()
    except FileNotFoundError:
        pass
    return _shared(db_path).read(source_path)


def iter_source_paths(
    prefix: str, *, endswith: str = "", db_path: Path | None = None
) -> list[str]:
    """Module-level virtual scandir over archived source paths."""
    return _shared(db_path).iter_source_paths(prefix, endswith=endswith)


def path_exists(source_path: str | os.PathLike[str], *, db_path: Path | None = None) -> bool:
    """Local-first existence check for gates that decide whether an asset is usable."""
    path = Path(source_path)
    if path.is_file() and path.stat().st_size > 0:
        return True
    try:
        return _shared(db_path).exists(source_path)
    except IndexedTarError:
        return False


def _shared(db_path: Path | None) -> ArchiveReader:
    # Keyed by database: a single global would silently keep serving the first
    # catalog it ever opened even when asked for another one.
    resolved = Path(db_path) if db_path is not None else default_db()
    key = str(resolved)
    with _SHARED_LOCK:
        reader = _SHARED.get(key)
        if reader is None:
            reader = ArchiveReader(resolved)
            _SHARED[key] = reader
        return reader
