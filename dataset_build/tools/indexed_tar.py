"""Deterministic, crash-safe indexed USTAR datasets.

JSONL indexes are the durable lookup authority.  The SQLite catalog is a
rebuildable acceleration layer for large datasets and never replaces them.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import tarfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, Mapping


SCHEMA_VERSION = 2
BLOCK_SIZE = 512
DEFAULT_SHARD_SIZE = 2 * 1024**3
TAR_WRITE_BUFFER = 8 * 1024**2
INDEX_WRITE_BUFFER = 1024**2
# Measured on the HDD source pool (~100KB images, one competing job): 4 workers
# gave ~21 MiB/s, 16 gave 37.8, 32 gave 33.1.  Deep queues win here because the
# drive reorders many outstanding small reads; the seek-storm intuition only
# applies to large sequential reads.  Re-measure before changing.
DEFAULT_READ_WORKERS = 16
DEFAULT_PREFETCH_FILES = 64
DEFAULT_PREFETCH_BYTES = 128 * 1024**2
DEFAULT_PARALLEL_FILE_MAX = 8 * 1024**2
KEY_POLICY = "webdataset_basename_v1"
MEMBER_ORDER = "member_lexicographic"
# 调用方指定顺序：plan 的行序即 tar 的成员序。用于让存储顺序等于消费顺序
# （生产顺序 = 训练读取顺序 = Viewer 按组取用的顺序），把随机读变顺序读。
MEMBER_ORDER_PLAN = "plan_order"
# USTAR stores the member name in a 100-byte field and a flat basename has no
# "/" for prefix splitting, so keys and extensions must stay short and ASCII.
MAX_MEMBER_BYTES = 100
_KEY_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}")
# Extensions stay lowercase so WebDataset decoders match, and allow "_"/"-" so
# role-qualified members such as ".target_a.png" or ".raw_mask.png" are legal.
_EXT_RE = re.compile(r"(?:\.[a-z0-9][a-z0-9_-]{0,31})+")
_MEMBER_RE = re.compile(_KEY_RE.pattern + _EXT_RE.pattern)
_INDEX_FIELDS = {
    "schema_version",
    "sample_id",
    "logical_path",
    "shard",
    "member",
    "suffix",
    "offset",
    "offset_data",
    "length",
    "size",
    "sha256",
}


class IndexedTarError(RuntimeError):
    """Raised when an indexed tar dataset is invalid or cannot be built."""


@dataclass(frozen=True)
class IndexRecord:
    sample_id: str
    logical_path: str
    shard: str
    member: str
    suffix: str
    offset_data: int
    size: int
    sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "sample_id": self.sample_id,
            "logical_path": self.logical_path,
            "shard": self.shard,
            "member": self.member,
            "suffix": self.suffix,
            "offset": self.offset_data,
            "offset_data": self.offset_data,
            "length": self.size,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class _SourceEntry:
    path: Path
    logical_path: str
    stat_result: os.stat_result


@dataclass(frozen=True)
class _PreparedSource:
    entry: _SourceEntry
    data: bytes
    sha256: str


class _DigestingReader:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        self.digest.update(chunk)
        self.bytes_read += len(chunk)
        return chunk


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise IndexedTarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads_json(data: str, *, source: Path) -> dict[str, object]:
    try:
        value = json.loads(data, object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, IndexedTarError) as exc:
        raise IndexedTarError(f"invalid JSON in {source}: {exc}") from exc
    if not isinstance(value, dict):
        raise IndexedTarError(f"JSON object required in {source}")
    return value


def _json_bytes(value: Mapping[str, object], *, pretty: bool = False) -> bytes:
    if pretty:
        text = json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True)
    else:
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return (text + "\n").encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _stable_member(logical_path: str) -> tuple[str, str, str]:
    """Derive the WebDataset member name and sample key from a logical path.

    The member is the bare basename and the sample key is everything before its
    first dot, so `x.jpg` and `x.mask.png` share key `x` and `wds.WebDataset`
    groups them into one sample with no custom collation.  Directory components
    of the logical path (scene, corpus, style major/minor) carry no key meaning
    and may hold non-ASCII names; the basename may not.
    """
    member = PurePosixPath(logical_path).name
    if len(member.encode("utf-8")) > MAX_MEMBER_BYTES:
        raise IndexedTarError(
            f"member name exceeds the {MAX_MEMBER_BYTES}-byte USTAR limit: {logical_path}"
        )
    if not _MEMBER_RE.fullmatch(member):
        raise IndexedTarError(f"member name violates {KEY_POLICY}: {logical_path}")
    key, _, extension = member.partition(".")
    return key, "." + extension, member


def _iter_source_files(root: Path) -> Iterator[_SourceEntry]:
    def walk(directory: Path, relative_directory: PurePosixPath) -> Iterator[_SourceEntry]:
        with os.scandir(directory) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
        for entry in entries:
            path = Path(entry.path)
            source_stat = entry.stat(follow_symlinks=False)
            mode = source_stat.st_mode
            if stat.S_ISLNK(mode):
                raise IndexedTarError(f"symlinks are not allowed: {path}")
            relative = relative_directory / entry.name
            if stat.S_ISDIR(mode):
                yield from walk(path, relative)
            elif stat.S_ISREG(mode):
                yield _SourceEntry(path, relative.as_posix(), source_stat)
            else:
                raise IndexedTarError(f"only regular files and directories can be sharded: {path}")

    yield from walk(root, PurePosixPath())


_PLAN_FIELDS = frozenset({"path", "logical_path"})


def _iter_plan_entries(plan_path: Path) -> Iterator[_SourceEntry]:
    """Yield sources named by a plan JSONL of {"path", "logical_path"} rows.

    A plan decouples the archived layout from the physical one: the packer never
    walks the source tree, so a logical path may place a file under any
    scene/corpus/style directory regardless of where it currently lives.  Rows
    must already be sorted by member name, which is what groups a sample.
    """
    previous_key: str | None = None
    closed_keys: set[str] = set()
    seen_members: set[str] = set()
    with plan_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise IndexedTarError(f"unterminated plan row at {plan_path}:{line_number}")
            if not line.strip():
                continue
            row = _loads_json(line, source=plan_path)
            if set(row) != _PLAN_FIELDS:
                raise IndexedTarError(
                    f"plan row must hold exactly {sorted(_PLAN_FIELDS)} at {plan_path}:{line_number}"
                )
            raw_path, logical_path = row["path"], row["logical_path"]
            if not isinstance(raw_path, str) or not isinstance(logical_path, str):
                raise IndexedTarError(f"plan fields must be strings at {plan_path}:{line_number}")
            path = Path(raw_path)
            if not path.is_absolute():
                raise IndexedTarError(f"plan paths must be absolute at {plan_path}:{line_number}")
            key, _suffix, member = _stable_member(logical_path)
            if member in seen_members:
                raise IndexedTarError(f"duplicate member at {plan_path}:{line_number}: {member}")
            seen_members.add(member)
            # 行序即成员序，唯一约束是同一 sample 的成员必须连续——这正是
            # WebDataset 分组所需，且比"成员名升序"弱，因此升序的 plan 依然合法。
            if key != previous_key:
                if key in closed_keys:
                    raise IndexedTarError(
                        f"sample {key} is not contiguous at {plan_path}:{line_number}"
                    )
                if previous_key is not None:
                    closed_keys.add(previous_key)
                previous_key = key
            source_stat = os.stat(path, follow_symlinks=False)
            if stat.S_ISLNK(source_stat.st_mode):
                raise IndexedTarError(f"symlinks are not allowed: {path}")
            if not stat.S_ISREG(source_stat.st_mode):
                raise IndexedTarError(f"only regular files can be sharded: {path}")
            yield _SourceEntry(path, logical_path, source_stat)


def _open_source(entry: _SourceEntry):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(entry.path, flags)
    raw = os.fdopen(descriptor, "rb")
    opened = os.fstat(raw.fileno())
    expected = entry.stat_result
    if not stat.S_ISREG(opened.st_mode):
        raw.close()
        raise IndexedTarError(f"source changed to a non-file: {entry.path}")
    if (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino):
        raw.close()
        raise IndexedTarError(f"source changed while opening: {entry.path}")
    if (expected.st_size, expected.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
        raw.close()
        raise IndexedTarError(f"source changed before reading: {entry.path}")
    return raw, opened


def _assert_source_unchanged(raw, opened: os.stat_result, path: Path) -> None:
    after = os.fstat(raw.fileno())
    if (after.st_size, after.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns):
        raise IndexedTarError(f"source changed while archiving: {path}")


def _prepare_source(entry: _SourceEntry) -> _PreparedSource:
    raw, opened = _open_source(entry)
    with raw:
        digest = hashlib.sha256()
        data = _read_exact(raw, opened.st_size, digest)
        _assert_source_unchanged(raw, opened, entry.path)
    return _PreparedSource(entry=entry, data=data, sha256=digest.hexdigest())


def _validate_index_row(value: Mapping[str, object], source: Path) -> IndexRecord:
    if set(value) != _INDEX_FIELDS:
        missing = sorted(_INDEX_FIELDS - set(value))
        extra = sorted(set(value) - _INDEX_FIELDS)
        raise IndexedTarError(f"invalid index fields in {source}: missing={missing}, extra={extra}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise IndexedTarError(f"unsupported index schema in {source}: {value['schema_version']!r}")
    string_fields = ("sample_id", "logical_path", "shard", "member", "suffix", "sha256")
    if any(not isinstance(value[field], str) or not value[field] for field in string_fields):
        raise IndexedTarError(f"invalid string field in {source}")
    integer_fields = ("offset", "offset_data", "length", "size")
    if any(type(value[field]) is not int or value[field] < 0 for field in integer_fields):
        raise IndexedTarError(f"invalid integer field in {source}")
    if value["offset"] != value["offset_data"]:
        raise IndexedTarError(f"offset aliases differ in {source}")
    if value["length"] != value["size"]:
        raise IndexedTarError(f"length and size differ in {source}")
    logical_path = str(value["logical_path"])
    expected_id, expected_suffix, expected_member = _stable_member(logical_path)
    if value["sample_id"] != expected_id:
        raise IndexedTarError(f"sample ID does not match logical path in {source}")
    if value["suffix"] != expected_suffix:
        raise IndexedTarError(f"suffix does not match logical path in {source}")
    if value["member"] != expected_member:
        raise IndexedTarError(f"member name does not match key policy in {source}")
    if not _MEMBER_RE.fullmatch(str(value["member"])):
        raise IndexedTarError(f"unsafe member name in {source}")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value["sha256"])):
        raise IndexedTarError(f"invalid SHA-256 in {source}")
    return IndexRecord(
        sample_id=str(value["sample_id"]),
        logical_path=logical_path,
        shard=str(value["shard"]),
        member=str(value["member"]),
        suffix=str(value["suffix"]),
        offset_data=int(value["offset_data"]),
        size=int(value["size"]),
        sha256=str(value["sha256"]),
    )


def _iter_index_records(path: Path) -> Iterator[IndexRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise IndexedTarError(f"unterminated index row at {path}:{line_number}")
            value = _loads_json(line, source=path)
            yield _validate_index_row(value, path)


def _read_exact(handle, size: int, digest: hashlib._Hash | None = None) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = handle.read(min(8 * 1024 * 1024, remaining))
        if not chunk:
            raise IndexedTarError(f"unexpected EOF with {remaining} bytes remaining")
        if digest is not None:
            digest.update(chunk)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def validate_shard(tar_path: Path, index_path: Path, shard_id: str) -> dict[str, object]:
    """Fully validate one shard and return digest/count metadata."""
    tar_path = Path(tar_path)
    index_path = Path(index_path)
    tar_digest = hashlib.sha256()
    member_count = 0
    payload_bytes = 0
    expected_header_offset = 0
    previous_key: str | None = None
    closed_keys: set[str] = set()
    seen_members: set[str] = set()

    with tar_path.open("rb") as archive:
        for record in _iter_index_records(index_path):
            if record.shard != shard_id:
                raise IndexedTarError(f"wrong shard ID in {index_path}: {record.shard}")
            # 成员顺序由生产方决定（可以是字典序，也可以是生产/读取顺序），这里只
            # 校验 WebDataset 真正依赖的两件事：成员不重复、同一 sample 的成员连续。
            if record.member in seen_members:
                raise IndexedTarError(f"duplicate member in {index_path}: {record.member}")
            seen_members.add(record.member)
            if record.sample_id != previous_key:
                if record.sample_id in closed_keys:
                    raise IndexedTarError(
                        f"sample is not contiguous in {index_path}: {record.sample_id}"
                    )
                if previous_key is not None:
                    closed_keys.add(previous_key)
                previous_key = record.sample_id

            if archive.tell() != expected_header_offset:
                raise IndexedTarError(f"non-contiguous tar layout in {tar_path}")
            header = _read_exact(archive, BLOCK_SIZE, tar_digest)
            if header[257:263] != b"ustar\x00":
                raise IndexedTarError(f"non-USTAR member in {tar_path} at {expected_header_offset}")
            try:
                info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
            except (tarfile.HeaderError, UnicodeError) as exc:
                raise IndexedTarError(f"invalid tar header in {tar_path}: {exc}") from exc
            if not info.isreg() or info.name != record.member or info.size != record.size:
                raise IndexedTarError(f"tar/index member mismatch in {tar_path}: {record.member}")
            if record.offset_data != expected_header_offset + BLOCK_SIZE:
                raise IndexedTarError(f"invalid data offset in {index_path}: {record.logical_path}")

            payload_digest = hashlib.sha256()
            remaining = record.size
            while remaining:
                chunk = archive.read(min(8 * 1024 * 1024, remaining))
                if not chunk:
                    raise IndexedTarError(f"truncated payload in {tar_path}: {record.member}")
                tar_digest.update(chunk)
                payload_digest.update(chunk)
                remaining -= len(chunk)
            if payload_digest.hexdigest() != record.sha256:
                raise IndexedTarError(f"payload checksum mismatch in {tar_path}: {record.member}")

            padding_size = (-record.size) % BLOCK_SIZE
            padding = _read_exact(archive, padding_size, tar_digest)
            if any(padding):
                raise IndexedTarError(f"non-zero member padding in {tar_path}: {record.member}")
            expected_header_offset += BLOCK_SIZE + record.size + padding_size
            member_count += 1
            payload_bytes += record.size

        trailer = archive.read()
        tar_digest.update(trailer)
        if len(trailer) < 2 * BLOCK_SIZE or any(trailer):
            raise IndexedTarError(f"tar trailer must contain at least two zero blocks: {tar_path}")

    tar_size = tar_path.stat().st_size
    if tar_size % BLOCK_SIZE:
        raise IndexedTarError(f"tar size is not block aligned: {tar_path}")
    index_sha256, index_size = _sha256_file(index_path)
    return {
        "member_count": member_count,
        "payload_bytes": payload_bytes,
        "tar_bytes": tar_size,
        "tar_sha256": tar_digest.hexdigest(),
        "index_bytes": index_size,
        "index_sha256": index_sha256,
    }


class _ShardWriter:
    def __init__(self, staging: Path, shard_id: str) -> None:
        self.shard_id = shard_id
        self.tar_tmp = staging / "shards" / f"{shard_id}.tar.tmp"
        self.index_tmp = staging / "indexes" / f"{shard_id}.idx.jsonl.tmp"
        self.tar_final = self.tar_tmp.with_suffix("")
        self.index_final = self.index_tmp.with_suffix("")
        self._tar_raw = self.tar_tmp.open("xb", buffering=TAR_WRITE_BUFFER)
        self._index = self.index_tmp.open("xb", buffering=INDEX_WRITE_BUFFER)
        self._tar = tarfile.open(fileobj=self._tar_raw, mode="w", format=tarfile.USTAR_FORMAT)
        self.member_count = 0
        self.payload_bytes = 0
        self.content_bytes = 0
        self.closed = False

    def projected_size(self, source_size: int) -> int:
        entry_size = BLOCK_SIZE + source_size + (-source_size) % BLOCK_SIZE
        return self.content_bytes + entry_size + 2 * BLOCK_SIZE

    def _info(self, logical_path: str, size: int) -> tuple[str, str, tarfile.TarInfo]:
        sample_id, suffix, member = _stable_member(logical_path)
        info = tarfile.TarInfo(member)
        info.size = size
        info.mode = 0o644
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        info.mtime = 0
        info.offset = self._tar.offset
        info.offset_data = info.offset + BLOCK_SIZE
        return sample_id, suffix, info

    def _record(self, logical_path: str, sample_id: str, suffix: str, info: tarfile.TarInfo, checksum: str) -> IndexRecord:
        record = IndexRecord(
            sample_id=sample_id,
            logical_path=logical_path,
            shard=self.shard_id,
            member=info.name,
            suffix=suffix,
            offset_data=info.offset_data,
            size=info.size,
            sha256=checksum,
        )
        self._index.write(_json_bytes(record.as_dict()))
        self.member_count += 1
        self.payload_bytes += info.size
        self.content_bytes = self._tar.offset
        return record

    def add(self, entry: _SourceEntry) -> IndexRecord:
        raw, opened = _open_source(entry)
        with raw:
            sample_id, suffix, info = self._info(entry.logical_path, opened.st_size)
            reader = _DigestingReader(raw)
            try:
                self._tar.addfile(info, reader)
            except (OSError, tarfile.TarError) as exc:
                raise IndexedTarError(f"failed to archive {entry.path}: {exc}") from exc
            if reader.bytes_read != opened.st_size:
                raise IndexedTarError(f"short source read: {entry.path}")
            _assert_source_unchanged(raw, opened, entry.path)
        return self._record(entry.logical_path, sample_id, suffix, info, reader.digest.hexdigest())

    def add_prepared(self, prepared: _PreparedSource) -> IndexRecord:
        sample_id, suffix, info = self._info(prepared.entry.logical_path, len(prepared.data))
        payload = io.BytesIO(prepared.data)
        try:
            self._tar.addfile(info, payload)
        except (OSError, tarfile.TarError) as exc:
            raise IndexedTarError(f"failed to archive {prepared.entry.path}: {exc}") from exc
        if payload.tell() != len(prepared.data):
            raise IndexedTarError(f"short prepared read: {prepared.entry.path}")
        return self._record(prepared.entry.logical_path, sample_id, suffix, info, prepared.sha256)

    def close_and_publish(self) -> dict[str, object]:
        if self.closed:
            raise IndexedTarError(f"shard already closed: {self.shard_id}")
        self._tar.close()
        self._tar_raw.flush()
        os.fsync(self._tar_raw.fileno())
        self._tar_raw.close()
        self._index.flush()
        os.fsync(self._index.fileno())
        self._index.close()
        self.closed = True

        stats = validate_shard(self.tar_tmp, self.index_tmp, self.shard_id)
        if stats["member_count"] != self.member_count or stats["payload_bytes"] != self.payload_bytes:
            raise IndexedTarError(f"post-write count mismatch for {self.shard_id}")
        os.replace(self.tar_tmp, self.tar_final)
        os.replace(self.index_tmp, self.index_final)
        _fsync_dir(self.tar_final.parent)
        _fsync_dir(self.index_final.parent)
        return {
            "shard_id": self.shard_id,
            "tar": self.tar_final.relative_to(self.tar_final.parents[1]).as_posix(),
            "index": self.index_final.relative_to(self.index_final.parents[1]).as_posix(),
            **stats,
        }

    def abort(self) -> None:
        if self.closed:
            return
        try:
            self._tar.close()
        finally:
            if not self._tar_raw.closed:
                self._tar_raw.close()
            if not self._index.closed:
                self._index.close()
            self.closed = True


_CATALOG_COLUMNS = (
    "sample_id",
    "logical_path",
    "shard",
    "member",
    "suffix",
    "offset_data",
    "size",
    "sha256",
)


def _create_catalog(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        """
        CREATE TABLE members (
            sample_id TEXT NOT NULL,
            logical_path TEXT NOT NULL UNIQUE,
            shard TEXT NOT NULL,
            member TEXT NOT NULL PRIMARY KEY,
            suffix TEXT NOT NULL,
            offset_data INTEGER NOT NULL,
            size INTEGER NOT NULL,
            sha256 TEXT NOT NULL
        ) WITHOUT ROWID
        """
    )
    connection.execute("CREATE INDEX members_by_shard_offset ON members(shard, offset_data)")
    # One sample owns several members, so sample_id is deliberately not unique.
    connection.execute("CREATE INDEX members_by_sample ON members(sample_id, suffix)")
    connection.execute("BEGIN")
    return connection


def _insert_catalog(connection: sqlite3.Connection, record: IndexRecord) -> None:
    try:
        connection.execute(
            "INSERT INTO members VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(getattr(record, column) for column in _CATALOG_COLUMNS),
        )
    except sqlite3.IntegrityError as exc:
        raise IndexedTarError(f"duplicate dataset key for {record.logical_path}: {exc}") from exc


def _catalog_tuple(record: IndexRecord) -> tuple[object, ...]:
    return tuple(getattr(record, column) for column in _CATALOG_COLUMNS)


def _dataset_id(shards: list[Mapping[str, object]]) -> str:
    try:
        identity_material = "\n".join(
            f"{item['shard_id']}:{item['tar_sha256']}:{item['index_sha256']}"
            for item in shards
        ).encode("ascii")
    except (KeyError, UnicodeEncodeError) as exc:
        raise IndexedTarError(f"invalid shard identity metadata: {exc}") from exc
    return "indexed_tar_" + hashlib.sha256(identity_material).hexdigest()[:32]


def _verify_catalog(catalog_path: Path, shard_indexes: list[Path], expected_count: int) -> None:
    uri = catalog_path.as_uri() + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    try:
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise IndexedTarError(f"SQLite integrity check failed for {catalog_path}: {result}")
        count = connection.execute("SELECT COUNT(*) FROM members").fetchone()[0]
        if count != expected_count:
            raise IndexedTarError(f"catalog count mismatch: {count} != {expected_count}")
        cursor = iter(connection.execute(
            "SELECT sample_id, logical_path, shard, member, suffix, offset_data, size, sha256 "
            "FROM members ORDER BY shard, offset_data"
        ))
        compared = 0
        for index_path in shard_indexes:
            for record in _iter_index_records(index_path):
                row = next(cursor, None)
                if row != _catalog_tuple(record):
                    raise IndexedTarError(f"catalog/JSONL mismatch at {record.logical_path}")
                compared += 1
        if next(cursor, None) is not None or compared != expected_count:
            raise IndexedTarError("catalog contains unexpected rows")
    finally:
        connection.close()


def _prepare_output(output_root: Path) -> Path:
    output_root = Path(output_root)
    if not output_root.is_absolute():
        raise IndexedTarError("output path must be absolute")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_parent = output_root.parent.resolve(strict=True)
    output_root = output_parent / output_root.name
    if _lexists(output_root):
        raise IndexedTarError(f"output already exists: {output_root}")
    return output_root


def _prepare_paths(source_root: Path, output_root: Path) -> tuple[Path, Path]:
    source_root = Path(source_root)
    if not source_root.is_absolute():
        raise IndexedTarError("source and output paths must be absolute")
    source_root = source_root.resolve(strict=True)
    if not source_root.is_dir():
        raise IndexedTarError(f"source is not a directory: {source_root}")
    output_root = _prepare_output(output_root)
    if output_root.is_relative_to(source_root):
        raise IndexedTarError("output cannot be inside the source tree")
    return source_root, output_root


def build_indexed_tar(
    source_root: Path | None,
    output_root: Path,
    *,
    plan: Path | None = None,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE,
    read_workers: int = DEFAULT_READ_WORKERS,
    prefetch_files: int = DEFAULT_PREFETCH_FILES,
    prefetch_bytes: int = DEFAULT_PREFETCH_BYTES,
    parallel_file_max: int = DEFAULT_PARALLEL_FILE_MAX,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    member_order: str = MEMBER_ORDER,
) -> dict[str, object]:
    """Build and atomically publish an indexed tar dataset.

    Pass either a `source_root` to archive a directory as-is, or a `plan` JSONL
    naming each source file and the logical path it takes inside the archive.

    Members always land in input order; `member_order` only records which order
    that was, so a plan the caller deliberately left unsorted must declare
    `MEMBER_ORDER_PLAN` (see `dataset_plan.write_group(preserve_order=True)`).
    """
    if type(shard_size_bytes) is not int or shard_size_bytes < 2 * BLOCK_SIZE:
        raise IndexedTarError("shard_size_bytes must be an integer of at least 1024")
    integer_options = {
        "read_workers": read_workers,
        "prefetch_files": prefetch_files,
        "prefetch_bytes": prefetch_bytes,
        "parallel_file_max": parallel_file_max,
    }
    if any(type(value) is not int or value < 1 for value in integer_options.values()):
        raise IndexedTarError(f"prefetch options must be positive integers: {integer_options}")
    if (source_root is None) == (plan is None):
        raise IndexedTarError("provide exactly one of source_root or plan")
    if plan is not None:
        plan = Path(plan)
        if not plan.is_absolute():
            raise IndexedTarError("plan path must be absolute")
        plan = plan.resolve(strict=True)
        output_root = _prepare_output(output_root)
        input_label, input_mode = str(plan), "plan"
    else:
        source_root, output_root = _prepare_paths(source_root, output_root)
        input_label, input_mode = str(source_root), "directory"
    staging = output_root.parent / f".{output_root.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    staging.mkdir(mode=0o700)
    (staging / "shards").mkdir()
    (staging / "indexes").mkdir()
    _fsync_dir(staging)

    catalog_tmp = staging / "indexes" / "catalog.sqlite3.tmp"
    catalog_final = staging / "indexes" / "catalog.sqlite3"
    catalog = _create_catalog(catalog_tmp)
    current: _ShardWriter | None = None
    current_key: str | None = None
    shard_metadata: list[dict[str, object]] = []
    total_members = 0
    total_samples = 0
    total_payload_bytes = 0

    def append_entry(entry: _SourceEntry, prepared: _PreparedSource | None = None) -> None:
        nonlocal current, current_key, total_members, total_payload_bytes, total_samples
        source_size = entry.stat_result.st_size
        key = _stable_member(entry.logical_path)[0]
        if current is None:
            current = _ShardWriter(staging, f"shard-{len(shard_metadata):05d}")
        elif (
            current.member_count
            and key != current_key
            and current.projected_size(source_size) > shard_size_bytes
        ):
            # Rotate on sample boundaries only.  A sample split across two tars
            # can never be reassembled by a sequential WebDataset reader, so the
            # shard size is a target and one oversized sample may exceed it.
            shard_metadata.append(current.close_and_publish())
            current = _ShardWriter(staging, f"shard-{len(shard_metadata):05d}")
        if key != current_key:
            current_key = key
            total_samples += 1
        record = current.add_prepared(prepared) if prepared is not None else current.add(entry)
        _insert_catalog(catalog, record)
        total_members += 1
        total_payload_bytes += record.size
        if progress is not None:
            progress({
                "members": total_members,
                "payload_bytes": total_payload_bytes,
                "shard": current.shard_id,
                "logical_path": entry.logical_path,
                "staging": str(staging),
            })

    def flush_batch(pool: ThreadPoolExecutor, batch: list[_SourceEntry]) -> None:
        for entry, prepared in zip(batch, pool.map(_prepare_source, batch)):
            if prepared.entry != entry:
                raise IndexedTarError("parallel source order changed unexpectedly")
            append_entry(entry, prepared)

    try:
        batch: list[_SourceEntry] = []
        batch_bytes = 0
        with ThreadPoolExecutor(max_workers=read_workers, thread_name_prefix="shard-read") as pool:
            entries = (
                _iter_plan_entries(plan) if plan is not None else _iter_source_files(source_root)
            )
            for entry in entries:
                source_size = entry.stat_result.st_size
                if source_size > parallel_file_max:
                    if batch:
                        flush_batch(pool, batch)
                        batch = []
                        batch_bytes = 0
                    append_entry(entry)
                    continue
                if batch and (len(batch) >= prefetch_files or batch_bytes + source_size > prefetch_bytes):
                    flush_batch(pool, batch)
                    batch = []
                    batch_bytes = 0
                batch.append(entry)
                batch_bytes += source_size
            if batch:
                flush_batch(pool, batch)

        if current is None:
            raise IndexedTarError(f"input contains no regular files: {input_label}")
        shard_metadata.append(current.close_and_publish())
        current = None

        catalog.commit()
        catalog.close()
        with catalog_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(catalog_tmp, catalog_final)
        _fsync_dir(catalog_final.parent)
        shard_indexes = [staging / str(item["index"]) for item in shard_metadata]
        _verify_catalog(catalog_final, shard_indexes, total_members)
        catalog_sha256, catalog_bytes = _sha256_file(catalog_final)

        dataset_id = _dataset_id(shard_metadata)
        manifest: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset_id": dataset_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "producer": "dataset_build.tools.indexed_tar",
            "source_root": input_label,
            "input_mode": input_mode,
            "archive_format": "ustar",
            "compression": "none",
            "member_order": member_order,
            "key_policy": KEY_POLICY,
            "target_shard_size_bytes": shard_size_bytes,
            "read_workers": read_workers,
            "prefetch_files": prefetch_files,
            "prefetch_bytes": prefetch_bytes,
            "parallel_file_max": parallel_file_max,
            "member_count": total_members,
            "sample_count": total_samples,
            "payload_bytes": total_payload_bytes,
            "shard_count": len(shard_metadata),
            "catalog": {
                "path": "indexes/catalog.sqlite3",
                "bytes": catalog_bytes,
                "sha256": catalog_sha256,
                "authority": "derived_from_jsonl_indexes",
            },
            "shards": shard_metadata,
        }
        manifest_tmp = staging / "manifest.json.tmp"
        with manifest_tmp.open("xb") as handle:
            handle.write(_json_bytes(manifest, pretty=True))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(manifest_tmp, staging / "manifest.json")
        _fsync_dir(staging)

        if _lexists(output_root):
            raise IndexedTarError(f"output appeared during build: {output_root}")
        staging.rename(output_root)
        _fsync_dir(output_root.parent)
        return manifest
    except BaseException:
        if current is not None:
            current.abort()
        try:
            catalog.close()
        except sqlite3.Error:
            pass
        raise


def _load_manifest(root: Path) -> dict[str, object]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise IndexedTarError(f"terminal manifest is missing: {manifest_path}")
    manifest = _loads_json(manifest_path.read_text(encoding="utf-8"), source=manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("status") != "complete":
        raise IndexedTarError(f"unsupported or non-terminal manifest: {manifest_path}")
    if manifest.get("archive_format") != "ustar" or manifest.get("compression") != "none":
        raise IndexedTarError(f"only uncompressed USTAR is supported: {manifest_path}")
    if not isinstance(manifest.get("shards"), list) or not isinstance(manifest.get("catalog"), dict):
        raise IndexedTarError(f"invalid manifest structure: {manifest_path}")
    return manifest


def _resolve_manifest_path(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise IndexedTarError("manifest path must be a non-empty string")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts:
        raise IndexedTarError(f"unsafe manifest path: {relative}")
    path = root
    for part in pure.parts:
        path /= part
        if path.is_symlink():
            raise IndexedTarError(f"manifest path cannot traverse symlinks: {relative}")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise IndexedTarError(f"manifest path escapes dataset root: {relative}")
    if not resolved.is_file():
        raise IndexedTarError(f"manifest target is not a regular file: {path}")
    return resolved


def verify_dataset(root: Path) -> dict[str, object]:
    """Fully verify every tar payload, JSONL index, catalog row, and digest."""
    root = Path(root).resolve(strict=True)
    manifest = _load_manifest(root)
    shard_indexes: list[Path] = []
    total_members = 0
    total_payload_bytes = 0
    shard_ids: set[str] = set()
    for item in manifest["shards"]:
        if not isinstance(item, dict):
            raise IndexedTarError("manifest shard entries must be objects")
        shard_id = item.get("shard_id")
        if not isinstance(shard_id, str) or shard_id in shard_ids:
            raise IndexedTarError(f"invalid or duplicate shard ID: {shard_id!r}")
        shard_ids.add(shard_id)
        tar_path = _resolve_manifest_path(root, item.get("tar"))
        index_path = _resolve_manifest_path(root, item.get("index"))
        stats = validate_shard(tar_path, index_path, shard_id)
        for field, actual in stats.items():
            if item.get(field) != actual:
                raise IndexedTarError(f"manifest {field} mismatch for {shard_id}")
        shard_indexes.append(index_path)
        total_members += int(stats["member_count"])
        total_payload_bytes += int(stats["payload_bytes"])

    if manifest.get("shard_count") != len(shard_ids):
        raise IndexedTarError("manifest shard count mismatch")
    if manifest.get("member_count") != total_members:
        raise IndexedTarError("manifest member count mismatch")
    if manifest.get("payload_bytes") != total_payload_bytes:
        raise IndexedTarError("manifest payload byte count mismatch")
    if manifest.get("dataset_id") != _dataset_id(manifest["shards"]):
        raise IndexedTarError("manifest dataset ID mismatch")
    catalog = manifest["catalog"]
    catalog_path = _resolve_manifest_path(root, catalog.get("path"))
    catalog_sha256, catalog_bytes = _sha256_file(catalog_path)
    if catalog.get("sha256") != catalog_sha256 or catalog.get("bytes") != catalog_bytes:
        raise IndexedTarError("catalog digest mismatch")
    _verify_catalog(catalog_path, shard_indexes, total_members)
    return {
        "dataset_id": manifest.get("dataset_id"),
        "shards": len(shard_ids),
        "members": total_members,
        "payload_bytes": total_payload_bytes,
    }


class IndexedTarDataset:
    """Random-access reader backed by the derived SQLite catalog."""

    def __init__(self, root: Path, *, verify_catalog_digest: bool = True) -> None:
        self.root = Path(root).resolve(strict=True)
        self.manifest = _load_manifest(self.root)
        self._shards: dict[str, Path] = {}
        for item in self.manifest["shards"]:
            if not isinstance(item, dict) or not isinstance(item.get("shard_id"), str):
                raise IndexedTarError("invalid shard entry in manifest")
            shard_id = str(item["shard_id"])
            if shard_id in self._shards:
                raise IndexedTarError(f"duplicate shard ID: {shard_id}")
            self._shards[shard_id] = _resolve_manifest_path(self.root, item.get("tar"))
        catalog = self.manifest["catalog"]
        catalog_path = _resolve_manifest_path(self.root, catalog.get("path"))
        if verify_catalog_digest:
            digest, size = _sha256_file(catalog_path)
            if digest != catalog.get("sha256") or size != catalog.get("bytes"):
                raise IndexedTarError("catalog digest mismatch")
        # 同 archive_reader：已发布的组 catalog 永不变更，immutable 省掉加锁并启用 mmap。
        self._connection = sqlite3.connect(catalog_path.as_uri() + "?immutable=1", uri=True)
        self._connection.execute("PRAGMA mmap_size=268435456")
        self._connection.row_factory = sqlite3.Row

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "IndexedTarDataset":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def lookup(
        self,
        *,
        sample_id: str | None = None,
        logical_path: str | None = None,
        suffix: str | None = None,
    ) -> dict[str, object]:
        if (sample_id is None) == (logical_path is None):
            raise IndexedTarError("provide exactly one of sample_id or logical_path")
        columns = ", ".join(_CATALOG_COLUMNS)
        if sample_id is not None:
            # A sample owns one member per extension, so the key alone is ambiguous.
            if suffix is None:
                raise IndexedTarError(f"sample {sample_id} needs a suffix to name one member")
            row = self._connection.execute(
                f"SELECT {columns} FROM members WHERE sample_id = ? AND suffix = ?",
                (sample_id, suffix),
            ).fetchone()
            missing: object = f"{sample_id}{suffix}"
        else:
            if suffix is not None:
                raise IndexedTarError("suffix applies to sample_id lookups only")
            row = self._connection.execute(
                f"SELECT {columns} FROM members WHERE logical_path = ?",
                (logical_path,),
            ).fetchone()
            missing = logical_path
        if row is None:
            raise KeyError(missing)
        return dict(row)

    def suffixes(self, sample_id: str) -> list[str]:
        """Return every extension this sample carries, in index order."""
        rows = self._connection.execute(
            "SELECT suffix FROM members WHERE sample_id = ? ORDER BY suffix", (sample_id,)
        ).fetchall()
        if not rows:
            raise KeyError(sample_id)
        return [str(row["suffix"]) for row in rows]

    def read_sample(self, sample_id: str, *, verify_checksum: bool = True) -> dict[str, bytes]:
        """Return the whole WebDataset sample as {suffix: payload}."""
        return {
            suffix: self.read(sample_id=sample_id, suffix=suffix, verify_checksum=verify_checksum)
            for suffix in self.suffixes(sample_id)
        }

    def read(
        self,
        *,
        sample_id: str | None = None,
        logical_path: str | None = None,
        suffix: str | None = None,
        verify_checksum: bool = True,
    ) -> bytes:
        record = self.lookup(sample_id=sample_id, logical_path=logical_path, suffix=suffix)
        shard_path = self._shards.get(str(record["shard"]))
        if shard_path is None:
            raise IndexedTarError(f"catalog references unknown shard: {record['shard']}")
        offset_data = int(record["offset_data"])
        size = int(record["size"])
        if offset_data < BLOCK_SIZE or offset_data % BLOCK_SIZE:
            raise IndexedTarError(f"invalid catalog offset: {offset_data}")
        with shard_path.open("rb") as handle:
            handle.seek(offset_data - BLOCK_SIZE)
            header = _read_exact(handle, BLOCK_SIZE)
            try:
                info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
            except (tarfile.HeaderError, UnicodeError) as exc:
                raise IndexedTarError(f"invalid member header: {exc}") from exc
            if not info.isreg() or info.name != record["member"] or info.size != size:
                raise IndexedTarError("catalog does not match tar member header")
            payload = _read_exact(handle, size)
        if verify_checksum and hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise IndexedTarError(f"payload checksum mismatch: {record['logical_path']}")
        return payload
