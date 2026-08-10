"""Pull a round's source images off NFS in archive-physical order.

The sampling order of a build is known before the round starts, but it is
scattered across the archive: reading it as it comes is one 51 ms random pread
per source over a 98 MB/s link.  The same bytes read in ``(shard, offset_data)``
order are one sequential pass per shard, which is what the server's readahead is
for.  So the producer prefetches a round ahead into tmpfs and
``archive_reader.set_prefetch_dir`` makes every later read a local one.

The buffer is a cache and nothing else: it holds one flat file per source named
by ``archive_reader.prefetch_name``, every write is atomic, and a missing or
half-recycled entry only costs the archive read it was avoiding.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tarfile
import uuid
from pathlib import Path
from typing import Callable, Mapping, Sequence

from dataset_build.tools.archive_reader import default_db, prefetch_name
from dataset_build.tools.indexed_tar import BLOCK_SIZE, IndexedTarError, _load_manifest


# 一条 SQL：临时表承载"要哪些路径"，避免 IN(?) 撞上 SQLite 的宿主参数上限，也就
# 不必把结果拆成多批再在 Python 里归并——排序由 SQLite 一次给全。root 排在最前面
# 是因为 shard id（shard-00000）只在组内唯一，跨组同名，物理顺序必须先按组走。
#
# CROSS JOIN 是 SQLite 的 join-order barrier，不是笛卡尔积：它只是禁止优化器把
# source_paths 挑成外层表。TEMP 表 want 没有统计信息，优化器据此猜它比 4.45M 行的
# source_paths 更大，于是全表 SCAN s 再对 want 点查——一次 5.4s，且与请求条数无关。
# 钉死外层为 want 后计划变成 SCAN w + SEARCH s USING PRIMARY KEY，5442ms→0.3ms。
# 连接是等值且 source_path 唯一，行集与 ORDER BY 决定的顺序都不受影响。
_LOCATE_SQL = """
SELECT w.source_path AS source_path, g.root AS root, m.shard AS shard,
       m.member AS member, m.offset_data AS offset_data, m.size AS size,
       m.sha256 AS sha256
FROM want w
CROSS JOIN source_paths s ON s.source_path = w.source_path
JOIN members m ON m."group" = s."group" AND m.member = s.member
JOIN groups g ON g."group" = s."group"
ORDER BY g.root, m.shard, m.offset_data
"""


def _locate(source_paths: Sequence[str], db_path: Path) -> list[dict[str, object]]:
    """Resolve every wanted path to its shard placement, already in read order."""
    if not db_path.is_file():
        raise IndexedTarError(
            f"global catalog is missing: {db_path} "
            "(rebuild it with python -m dataset_build.tools.global_catalog)"
        )
    # immutable=1 for the same reason as ArchiveReader: the catalog is replaced
    # atomically, so a reader either sees the old snapshot or reopens.  A TEMP
    # table stays writable regardless — it lives in the temp database, not this one.
    connection = sqlite3.connect(db_path.as_uri() + "?immutable=1", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA mmap_size=1073741824")
        connection.execute("CREATE TEMP TABLE want (source_path TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT OR IGNORE INTO want VALUES (?)", ((path,) for path in source_paths)
        )
        return [dict(row) for row in connection.execute(_LOCATE_SQL)]
    finally:
        connection.close()


def _shard_path(root: str, shard: str, cache: dict[tuple[str, str], str]) -> str:
    """Map a catalog (root, shard) onto the tar file holding it."""
    key = (root, shard)
    path = cache.get(key)
    if path is None:
        for item in _load_manifest(Path(root))["shards"]:
            cache[(root, str(item["shard_id"]))] = str(Path(root) / str(item["tar"]))
        path = cache.get(key)
        if path is None:
            raise IndexedTarError(f"unknown shard {shard} under {root}")
    return path


def _pread_exact(descriptor: int, size: int, offset: int, what: str) -> bytes:
    payload = b""
    while len(payload) < size:
        chunk = os.pread(descriptor, size - len(payload), offset + len(payload))
        if not chunk:
            raise IndexedTarError(f"truncated member for {what}")
        payload += chunk
    return payload


def _read_member(descriptor: int, row: dict[str, object]) -> bytes:
    """Read one member and verify both its tar header and indexed digest."""
    offset, size = int(row["offset_data"]), int(row["size"])
    block = _pread_exact(descriptor, BLOCK_SIZE + size, offset - BLOCK_SIZE, str(row["source_path"]))
    try:
        info = tarfile.TarInfo.frombuf(block[:BLOCK_SIZE], encoding="utf-8", errors="strict")
    except (tarfile.HeaderError, UnicodeError) as exc:
        raise IndexedTarError(f"invalid member header for {row['source_path']}: {exc}") from exc
    if info.name != row["member"] or info.size != size:
        raise IndexedTarError(f"catalog does not match shard header for {row['source_path']}")
    payload = block[BLOCK_SIZE:]
    expected_digest = row.get("sha256")
    if expected_digest and hashlib.sha256(payload).hexdigest() != expected_digest:
        raise IndexedTarError(f"payload checksum mismatch for {row['source_path']}")
    return payload


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PrefetchResult(dict[str, Path]):
    """Backward-compatible mapping plus the index metadata verified for each copy."""

    def __init__(self) -> None:
        super().__init__()
        self.records: dict[str, tuple[int, str]] = {}


def _publish(
    target: Path,
    payload: bytes,
    *,
    cancelled: Callable[[], bool] | None = None,
) -> None:
    """Land one buffered copy atomically, so a reader never sees a partial file."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        if cancelled is not None and cancelled():
            raise InterruptedError("prefetch cancelled")
        tmp.write_bytes(payload)
        if cancelled is not None and cancelled():
            raise InterruptedError("prefetch cancelled")
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # No fsync: the buffer is rebuildable by definition; losing it costs a re-read.


def prefetch(
    source_paths: list[str],
    dest: Path,
    *,
    db_path: Path | None = None,
    progress: Callable[[Mapping[str, object]], None] | None = None,
    strict: bool = False,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Path]:
    """Read the given sources in archive-physical order into ``dest``.

    Returns ``{original path: buffered copy}`` for the paths the buffer now
    serves. Paths that still exist on this machine are skipped because
    ``read_bytes`` is local-first. Unknown catalog paths remain skipped by
    default for compatibility; ``strict=True`` turns them into an error.

    ``progress`` receives synchronous snapshots for ``locating`` and
    ``materializing``. Existing callers pay no callback or strictness cost.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_db = Path(db_path) if db_path is not None else default_db()
    requested = list(dict.fromkeys(map(str, source_paths)))
    local_sizes: dict[str, int] = {}
    wanted: list[str] = []
    for path in requested:
        try:
            size = os.stat(path).st_size
        except OSError:
            wanted.append(path)
        else:
            if size > 0:
                local_sizes[path] = size
            else:
                wanted.append(path)

    def emit(payload: Mapping[str, object]) -> None:
        if progress is not None:
            progress(payload)

    emit({
        "phase": "locating",
        "files_done": len(local_sizes),
        "files_total": len(requested),
        "bytes_done": sum(local_sizes.values()),
        "bytes_total": sum(local_sizes.values()),
        "current_item": None,
    })
    if not wanted:
        return PrefetchResult()

    rows = _locate(wanted, resolved_db)
    located = {str(row["source_path"]) for row in rows}
    missing = [path for path in wanted if path not in located]
    if strict and missing:
        preview = ", ".join(missing[:3])
        suffix = "" if len(missing) <= 3 else f" (+{len(missing) - 3} more)"
        raise IndexedTarError(f"catalog does not contain requested paths: {preview}{suffix}")

    local_bytes = sum(local_sizes.values())
    archive_bytes = sum(int(row["size"]) for row in rows)
    files_done = len(local_sizes)
    bytes_done = local_bytes
    fetched = PrefetchResult()
    pending: list[dict[str, object]] = []
    for row in rows:
        if cancelled is not None and cancelled():
            raise InterruptedError("prefetch cancelled")
        source_path = str(row["source_path"])
        target = dest / prefetch_name(source_path)
        size = int(row["size"])
        digest = str(row["sha256"])
        try:
            cached = target.stat().st_size == size and _file_digest(target) == digest
        except OSError:
            cached = False
        if cached:
            os.utime(target, None)
            fetched[source_path] = target
            fetched.records[source_path] = (size, digest)
            files_done += 1
            bytes_done += size
        else:
            target.unlink(missing_ok=True)
            pending.append(row)

    # Invalid finals have been removed, so the exact peak increase for the
    # single-writer materializer is the sum of all payloads still to publish.
    bytes_needed = sum(int(row["size"]) for row in pending)
    emit({
        "phase": "materializing",
        "files_done": files_done,
        "files_total": len(local_sizes) + len(rows),
        "bytes_done": bytes_done,
        "bytes_total": local_bytes + archive_bytes,
        "bytes_needed": bytes_needed,
        "current_item": None,
    })

    shard_paths: dict[tuple[str, str], str] = {}
    descriptor: int | None = None
    open_shard: str | None = None
    try:
        for row in pending:
            if cancelled is not None and cancelled():
                raise InterruptedError("prefetch cancelled")
            source_path = str(row["source_path"])
            target = dest / prefetch_name(source_path)
            size = int(row["size"])
            digest = str(row["sha256"])
            path = _shard_path(str(row["root"]), str(row["shard"]), shard_paths)
            if path != open_shard:
                if descriptor is not None:
                    os.close(descriptor)
                descriptor = os.open(path, os.O_RDONLY)
                open_shard = path
            assert descriptor is not None
            payload = _read_member(descriptor, row)
            _publish(target, payload, cancelled=cancelled)
            fetched[source_path] = target
            fetched.records[source_path] = (size, digest)
            files_done += 1
            bytes_done += size
            emit({
                "phase": "materializing",
                "files_done": files_done,
                "files_total": len(local_sizes) + len(rows),
                "bytes_done": bytes_done,
                "bytes_total": local_bytes + archive_bytes,
                "bytes_needed": bytes_needed,
                "current_item": source_path,
            })
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return fetched


def _read_paths(paths_file: Path) -> list[str]:
    return [line.strip() for line in paths_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths-file", type=Path, required=True, help="one source path per line")
    parser.add_argument("--dest", type=Path, required=True, help="prefetch buffer directory")
    parser.add_argument("--db", type=Path, default=None, help="global catalog (default: VERADATA_CATALOG)")
    args = parser.parse_args(argv)
    try:
        requested = _read_paths(args.paths_file)
        fetched = prefetch(requested, args.dest, db_path=args.db)
    except (IndexedTarError, OSError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "requested": len(requested),
        "buffered": len(fetched),
        "dest": str(args.dest),
        "bytes": sum(path.stat().st_size for path in fetched.values()),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
