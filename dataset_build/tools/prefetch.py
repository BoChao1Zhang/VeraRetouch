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
import json
import os
import sqlite3
import sys
import tarfile
from pathlib import Path
from typing import Sequence

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
       m.member AS member, m.offset_data AS offset_data, m.size AS size
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
    """Read one member and check the tar header agrees with the catalog.

    Header and payload come out in one pread so the descriptor walks forward
    exactly once per member, which is the whole point of the ordering above.
    """
    offset, size = int(row["offset_data"]), int(row["size"])
    block = _pread_exact(descriptor, BLOCK_SIZE + size, offset - BLOCK_SIZE, str(row["source_path"]))
    try:
        info = tarfile.TarInfo.frombuf(block[:BLOCK_SIZE], encoding="utf-8", errors="strict")
    except (tarfile.HeaderError, UnicodeError) as exc:
        raise IndexedTarError(f"invalid member header for {row['source_path']}: {exc}") from exc
    if info.name != row["member"] or info.size != size:
        raise IndexedTarError(f"catalog does not match shard header for {row['source_path']}")
    return block[BLOCK_SIZE:]


def _publish(target: Path, payload: bytes) -> None:
    """Land one buffered copy atomically, so a reader never sees a partial file."""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    # No fsync: the buffer is tmpfs and rebuildable by definition — losing it to
    # a crash costs a re-read, and fsyncing every source would cost the round.


def _still_local(path: str) -> bool:
    try:
        return os.stat(path).st_size > 0
    except OSError:
        return False


def prefetch(
    source_paths: list[str], dest: Path, *, db_path: Path | None = None
) -> dict[str, Path]:
    """Read the given sources in archive-physical order into ``dest``.

    Returns ``{original path: buffered copy}`` for the paths the buffer now
    serves.  Paths that still exist on this machine are skipped — ``read_bytes``
    is local-first, so buffering them would only duplicate them — and so are
    paths the catalog does not know; both are simply absent from the mapping.
    Already-buffered copies of the right size are kept as they are, which makes
    the call idempotent and cheap to repeat on a resume.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    resolved_db = Path(db_path) if db_path is not None else default_db()

    wanted = [path for path in dict.fromkeys(map(str, source_paths)) if not _still_local(path)]
    if not wanted:
        return {}

    fetched: dict[str, Path] = {}
    shard_paths: dict[tuple[str, str], str] = {}
    descriptor: int | None = None
    open_shard: str | None = None
    try:
        for row in _locate(wanted, resolved_db):
            source_path = str(row["source_path"])
            target = dest / prefetch_name(source_path)
            size = int(row["size"])
            try:
                if target.stat().st_size == size:
                    fetched[source_path] = target
                    continue
            except OSError:
                pass
            path = _shard_path(str(row["root"]), str(row["shard"]), shard_paths)
            if path != open_shard:
                # One descriptor at a time is enough: the rows arrive grouped by
                # shard, so a shard is finished before the next one opens.
                if descriptor is not None:
                    os.close(descriptor)
                descriptor = os.open(path, os.O_RDONLY)
                open_shard = path
            assert descriptor is not None
            _publish(target, _read_member(descriptor, row))
            fetched[source_path] = target
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
