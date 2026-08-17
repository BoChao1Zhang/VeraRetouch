"""Enumerate the ``subject_not_ready`` source pool for the SAM3 mask backfill.

The catalog knows *which* subject-cache entries are not ready (20,009 of 56,778
rows in group ``cache/subject``), but its ``meta.source_path`` for exactly those
rows is ``null`` — the landing pass could not resolve a source for an entry whose
``subject.json`` never reached ``status="ready"``.  The only surviving link is the
directory name: ``sample_id = "subject_" + sha1(<absolute source path>)[:16]``
(``dataset_build/mask_cache.py``).  So the pool is rebuilt by hashing every source
path PostgreSQL still holds and matching the digests back.

Output is one JSON object per line, a superset of what
``sam3_subject_instances dump-pool`` writes, so ``run --pool`` consumes it
unchanged:

    {"asset_id": ..., "source_path": <original absolute path, the cache key>,
     "path_key": ..., "read_path": <where the pixels will actually be>,
     "archive": {"group","member","shard","offset","size","root"} | null}

``source_path`` stays the *logical* path in every row.  It is what ``path_key``
hashes and what ``subject.json`` must record, because the consumer gate
(``construct.sources._inspect_cache_dir``) resolves it through the archive-aware
``path_exists``.  ``read_path`` is only where this machine can open the bytes
today; for archived sources that is the prefetch scratch file that
``tools/mask_backfill/prefetch.py`` fills.

Usage::

    python tools/mask_backfill/enumerate.py --out backfill_pool.jsonl
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sqlite3
import sys
from pathlib import Path

DEFAULT_CATALOG = "/var/cache/veradata/global.sqlite3"
DEFAULT_PG_DSN = os.environ.get(
    "SOURCE_QA_PG_DSN", "postgresql://research:research@127.0.0.1:5432/vera_source_qa"
)
SUBJECT_GROUP = "cache/subject"
NOT_READY = "subject_not_ready"
DEFAULT_SCRATCH = "/home/bc/data/scratch/mask_backfill/src"
# The catalog records the archive under its read-write mount.  Nothing in this
# campaign may touch it (hard mount, a stall is unrecoverable), so every root is
# rewritten onto the soft read-only mount before it reaches an open().
NFS_RW = "/mnt/nfs/"
NFS_RO = "/mnt/nfs-ro/"


def ro_root(root: str) -> str:
    """Rewrite an archive root onto the soft read-only NFS mount."""
    root = str(root)
    return NFS_RO + root[len(NFS_RW):] if root.startswith(NFS_RW) else root


def path_key(path: str) -> str:
    """The stable 16-hex cache directory key (mirrors dataset_build.mask_cache)."""
    return hashlib.sha1(os.fspath(path).encode("utf-8")).hexdigest()[:16]


def not_ready_keys(catalog: str) -> set[str]:
    """The cache keys the landing gate rejected for ``subject_not_ready``."""
    uri = Path(catalog).absolute().as_uri() + "?immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    try:
        rows = connection.execute(
            'SELECT sample_id FROM samples WHERE "group" = ? '
            "AND json_extract(meta, '$.ineligible_reason') = ?",
            (SUBJECT_GROUP, NOT_READY),
        ).fetchall()
    finally:
        connection.close()
    return {str(row[0]).removeprefix("subject_") for row in rows}


def source_rows(dsn: str) -> list[tuple[str, str]]:
    """Every captioned source PostgreSQL knows, the same join ``dump-pool`` uses."""
    import psycopg

    with psycopg.connect(dsn) as connection:
        return [
            (str(asset_id), str(path))
            for asset_id, path in connection.execute(
                "SELECT c.asset_id, s.path FROM source_captions c "
                "JOIN assets s USING(asset_id) ORDER BY c.asset_id"
            ).fetchall()
        ]


def archive_locations(catalog: str, paths: list[str]) -> dict[str, dict]:
    """Locate each source path in the archive, in one indexed join.

    Per-path ``ArchiveReader.locate`` would be 19,699 round trips; a temp table
    joined against ``source_paths`` is a single scan and also returns the
    ``(shard, offset)`` the prefetcher sorts on.
    """
    uri = Path(catalog).absolute().as_uri() + "?immutable=1"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("CREATE TEMP TABLE wanted(p TEXT PRIMARY KEY)")
        connection.executemany(
            "INSERT OR IGNORE INTO wanted VALUES (?)", [(p,) for p in paths]
        )
        rows = connection.execute(
            'SELECT s.source_path, s."group" AS grp, s.member, m.shard, '
            "m.offset_data, m.size, g.root FROM wanted "
            "JOIN source_paths s ON s.source_path = wanted.p "
            'JOIN members m ON m."group" = s."group" AND m.member = s.member '
            'JOIN groups g ON g."group" = s."group"'
        ).fetchall()
    finally:
        connection.close()
    return {
        str(row["source_path"]): {
            "group": str(row["grp"]),
            "member": str(row["member"]),
            "shard": str(row["shard"]),
            "offset": int(row["offset_data"]),
            "size": int(row["size"]),
            "root": ro_root(str(row["root"])),
        }
        for row in rows
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", default=DEFAULT_CATALOG)
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN)
    parser.add_argument("--scratch-dir", default=DEFAULT_SCRATCH,
                        help="where prefetch.py will land archived sources")
    parser.add_argument("--out", required=True)
    parser.add_argument("--report", default="",
                        help="optional path for the machine-readable reconciliation")
    args = parser.parse_args()

    keys = not_ready_keys(args.catalog)
    rows = source_rows(args.pg_dsn)
    by_key: dict[str, tuple[str, str]] = {}
    collisions = 0
    for asset_id, path in rows:
        key = path_key(path)
        if key in by_key:
            collisions += 1
        by_key[key] = (asset_id, path)

    matched = sorted(key for key in keys if key in by_key)
    unmatched = sorted(keys - set(by_key))
    located = archive_locations(args.catalog, [by_key[key][1] for key in matched])

    scratch = Path(args.scratch_dir)
    counts: collections.Counter[str] = collections.Counter()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    archived_bytes = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for key in matched:
            asset_id, source_path = by_key[key]
            archive = located.get(source_path)
            local = os.path.exists(source_path) and os.path.getsize(source_path) > 0
            if local:
                counts["local"] += 1
                read_path = source_path
            elif archive is not None:
                counts["archived"] += 1
                archived_bytes += archive["size"]
                suffix = os.path.splitext(source_path)[1].lower() or ".bin"
                read_path = os.fspath(scratch / f"{key}{suffix}")
            else:
                counts["unreachable"] += 1
                read_path = source_path
            handle.write(json.dumps({
                "asset_id": asset_id,
                "source_path": source_path,
                "path_key": key,
                "read_path": read_path,
                "archive": archive,
            }, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)

    report = {
        "catalog_not_ready": len(keys),
        "pg_source_rows": len(rows),
        "pg_unique_path_keys": len(by_key),
        "pg_path_key_collisions": collisions,
        "matched": len(matched),
        "unmatched_catalog_keys": len(unmatched),
        "local_readable": counts["local"],
        "archived": counts["archived"],
        "unreachable": counts["unreachable"],
        "archived_bytes": archived_bytes,
        "archived_mean_bytes": round(archived_bytes / max(counts["archived"], 1), 1),
        "archive_groups": dict(collections.Counter(
            located[by_key[key][1]]["group"] for key in matched
            if by_key[key][1] in located)),
        "archive_shards": len({
            (located[by_key[key][1]]["root"], located[by_key[key][1]]["shard"])
            for key in matched if by_key[key][1] in located}),
        "out": os.fspath(out_path),
    }
    print(json.dumps(report, ensure_ascii=False, indent=1))
    if args.report:
        Path(args.report).write_text(
            json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0 if not unmatched and not counts["unreachable"] else 1


if __name__ == "__main__":
    sys.exit(main())
