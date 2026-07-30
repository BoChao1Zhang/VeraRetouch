"""Rebuildable global index over every published shard group.

Each group already carries its own durable JSONL indexes and a derived SQLite
catalog, which is what a training job or the offset reader needs.  This tool adds
the cross-group view — "every portrait sample regardless of corpus", "which shard
holds this preset" — and is a pure projection: delete the file and rebuild it from
the archive at any time.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Sequence

from dataset_build.tools.archive_reader import default_db
from dataset_build.tools.indexed_tar import (
    IndexedTarError,
    _iter_index_records,
    _load_manifest,
)


SCHEMA = """
CREATE TABLE groups (
    "group"      TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    class_a      TEXT,
    class_b      TEXT,
    class_c      TEXT,
    root         TEXT NOT NULL,
    dataset_id   TEXT NOT NULL,
    key_policy   TEXT NOT NULL,
    shard_count  INTEGER NOT NULL,
    member_count INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    payload_bytes INTEGER NOT NULL,
    created_at   TEXT NOT NULL
);
CREATE TABLE members (
    member       TEXT NOT NULL,
    "group"      TEXT NOT NULL REFERENCES groups("group"),
    sample_id    TEXT NOT NULL,
    logical_path TEXT NOT NULL,
    suffix       TEXT NOT NULL,
    shard        TEXT NOT NULL,
    offset_data  INTEGER NOT NULL,
    size         INTEGER NOT NULL,
    sha256       TEXT NOT NULL,
    PRIMARY KEY ("group", member)
) WITHOUT ROWID;
CREATE INDEX members_by_sample ON members(sample_id, suffix);
CREATE INDEX members_by_sha ON members(sha256);
CREATE TABLE samples (
    sample_id TEXT NOT NULL,
    "group"   TEXT NOT NULL REFERENCES groups("group"),
    meta      TEXT NOT NULL,
    PRIMARY KEY ("group", sample_id)
) WITHOUT ROWID;
-- Reverse map from the file's original absolute path to its archived member, so
-- existing call sites keep passing the paths they already hardcode and read from
-- the archive once the local copy is gone.
CREATE TABLE source_paths (
    source_path TEXT PRIMARY KEY,
    "group"     TEXT NOT NULL REFERENCES groups("group"),
    sample_id   TEXT NOT NULL,
    suffix      TEXT NOT NULL,
    member      TEXT NOT NULL
) WITHOUT ROWID;
"""

# Group paths are the archive's own taxonomy, so the columns mirror them rather
# than re-deriving classes from per-sample metadata.
_KIND_COLUMNS = {
    "img": ("scene", "corpus", None),
    "preset": ("fmt", "major", "minor"),
    "cgt": ("stage", None, None),
    "cache": ("kind", None, None),
    "renders": ("build_id", None, None),
    # Published by the databuild land checkpoint, one batch per checkpoint: the
    # full eight-candidate groups and the winner-only SFT view of the same bytes.
    "groups": ("build_id", "batch", None),
    "sft": ("build_id", "batch", None),
}


def _classify(group: str) -> tuple[str, str | None, str | None, str | None]:
    parts = group.split("/")
    kind = parts[0]
    if kind not in _KIND_COLUMNS:
        raise IndexedTarError(f"unknown group kind: {group}")
    tail = parts[1:] + [None, None, None]
    return kind, tail[0], tail[1], tail[2]


def _iter_groups(dataset_root: Path) -> Iterator[tuple[str, Path]]:
    for manifest_path in sorted(dataset_root.rglob("manifest.json")):
        group_dir = manifest_path.parent
        yield group_dir.relative_to(dataset_root).as_posix(), group_dir


def _index_group(
    connection: sqlite3.Connection, group: str, group_dir: Path
) -> tuple[int, int, int]:
    """Insert one published group's rows; returns ``(members, samples, source_paths)``."""
    manifest = _load_manifest(group_dir)
    kind, class_a, class_b, class_c = _classify(group)
    connection.execute(
        'INSERT INTO groups VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
        (
            group, kind, class_a, class_b, class_c, str(group_dir),
            str(manifest["dataset_id"]), str(manifest["key_policy"]),
            int(manifest["shard_count"]), int(manifest["member_count"]),
            int(manifest.get("sample_count") or 0),
            int(manifest["payload_bytes"]), str(manifest["created_at"]),
        ),
    )
    members = samples = reverse = 0
    for shard in manifest["shards"]:
        index_path = group_dir / str(shard["index"])
        for record in _iter_index_records(index_path):
            connection.execute(
                "INSERT INTO members VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    record.member, group, record.sample_id, record.logical_path,
                    record.suffix, record.shard, record.offset_data, record.size,
                    record.sha256,
                ),
            )
            members += 1
    metadata_path = group_dir / "metadata.jsonl"
    if metadata_path.is_file():
        # metadata.jsonl holds one row per member; the sample-level payload
        # is the ".vrmeta.json" row, so prefer it over a member row that
        # only describes one file of the sample.
        best: dict[str, tuple[bool, str]] = {}
        with metadata_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                sample_id = str(row.get("sample_id") or "")
                if not sample_id:
                    continue
                member = str(row.get("member") or "")
                is_sample_level = member.endswith(".vrmeta.json")
                source_path = str(row.get("source_path") or "")
                # Every real member earns a reverse mapping; the metadata
                # member is materialised in transient staging, so it never
                # points anywhere useful.
                if source_path and member and not is_sample_level:
                    connection.execute(
                        "INSERT OR REPLACE INTO source_paths VALUES (?,?,?,?,?)",
                        (
                            source_path, group, sample_id,
                            "." + member.partition(".")[2], member,
                        ),
                    )
                    reverse += 1
                if sample_id in best and not is_sample_level:
                    continue
                best[sample_id] = (
                    is_sample_level,
                    json.dumps(row, ensure_ascii=False, sort_keys=True),
                )
        for sample_id, (_level, payload) in best.items():
            connection.execute(
                "INSERT INTO samples VALUES (?,?,?)", (sample_id, group, payload)
            )
            samples += 1
    return members, samples, reverse


def rebuild(dataset_root: Path, db_path: Path) -> dict[str, object]:
    """Rebuild the global index from every published group under dataset_root."""
    dataset_root = Path(dataset_root).resolve(strict=True)
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    staging = db_path.with_name(db_path.name + ".rebuilding")
    staging.unlink(missing_ok=True)

    connection = sqlite3.connect(staging)
    groups = members = samples = reverse = 0
    try:
        connection.executescript(SCHEMA)
        connection.execute("BEGIN")
        for group, group_dir in _iter_groups(dataset_root):
            group_members, group_samples, group_reverse = _index_group(
                connection, group, group_dir
            )
            groups += 1
            members += group_members
            samples += group_samples
            reverse += group_reverse
        connection.commit()
        result = connection.execute("PRAGMA integrity_check").fetchone()
        if result != ("ok",):
            raise IndexedTarError(f"integrity check failed for {staging}: {result}")
    finally:
        connection.close()
    staging.replace(db_path)
    return {
        "db": str(db_path),
        "groups": groups,
        "members": members,
        "samples": samples,
        "source_paths": reverse,
        "bytes": db_path.stat().st_size,
    }


def upsert(dataset_root: Path, db_path: Path, groups: Sequence[str]) -> dict[str, object]:
    """Re-index only the named groups, leaving the rest of the catalog untouched.

    Callers name every batch the build has published so far, so one landing
    checkpoint re-registers a handful of groups in an archive that already holds
    hundreds; rebuilding all of them re-reads every index (5.5 M members,
    ~390 s) to learn nothing new.

    What this does guarantee: for the named groups the ``groups``, ``members``
    and ``samples`` rows end up identical to a full ``rebuild``, because those
    are a pure function of the group's own directory.  ``source_paths`` is
    keyed by path rather than by group and is therefore last-writer-wins: a
    source archived into more than one group belongs to whichever group was
    indexed last, so upserting one of them takes those rows back from the other
    and the attribution can differ from what a full rebuild would leave.  Both
    rows point at the same bytes — only prefetch locality moves — so the
    divergence is tolerated, and ``rebuild`` remains the way to restore the
    canonical attribution.  The named groups are indexed in the same sorted
    order ``rebuild`` walks them in, so at least one call's own ordering is not
    a second source of drift.

    The write still lands through the same copy-and-rename as ``rebuild``: every
    reader opens the catalog with ``immutable=1``, which is only sound while the
    file it has open never changes underneath it.  ``PRAGMA integrity_check`` is
    dropped (28 s of reading pages this call did not write); the full ``rebuild``
    remains the repair tool that checks.
    """
    dataset_root = Path(dataset_root).resolve(strict=True)
    db_path = Path(db_path)
    if not db_path.is_file():
        # Nothing to be incremental about, and a catalog with only this build's
        # groups in it would hide every other archive from the reverse map.
        return rebuild(dataset_root, db_path)
    # sorted(), because ``rebuild`` walks sorted manifest paths: the reverse map
    # is last-writer-wins, so indexing in the caller's argument order would make
    # the attribution depend on how the CLI happened to list the groups.
    wanted = sorted({str(group) for group in groups if str(group)})
    if not wanted:
        return {
            "db": str(db_path), "groups": 0, "members": 0, "samples": 0,
            "source_paths": 0, "bytes": db_path.stat().st_size,
        }

    staging = db_path.with_name(db_path.name + ".upserting")
    staging.unlink(missing_ok=True)
    # The copy itself can fail (a full state partition is the obvious way), and a
    # half-copied ``.upserting`` left behind would be adopted by the next call's
    # ``unlink`` at best and confuse a human at worst.
    try:
        shutil.copyfile(db_path, staging)
        connection = sqlite3.connect(staging)
    except BaseException:
        staging.unlink(missing_ok=True)
        raise
    members = samples = reverse = 0
    try:
        connection.execute("BEGIN")
        for group in wanted:
            if not (dataset_root / group / "manifest.json").is_file():
                raise IndexedTarError(f"group is not published under {dataset_root}: {group}")
        known = [
            group for group in wanted
            if connection.execute(
                'SELECT 1 FROM groups WHERE "group" = ?', (group,)
            ).fetchone()
        ]
        if known:
            # ``source_paths`` is keyed by path, so clearing a group costs one scan
            # of 4.45 M rows; a freshly landed batch is not in the catalog at all,
            # which is why the common case deletes nothing and the rest deletes once.
            placeholders = ",".join("?" * len(known))
            for table in ("source_paths", "samples", "members", "groups"):
                connection.execute(
                    f'DELETE FROM {table} WHERE "group" IN ({placeholders})', known
                )
        for group in wanted:
            group_members, group_samples, group_reverse = _index_group(
                connection, group, dataset_root / group
            )
            members += group_members
            samples += group_samples
            reverse += group_reverse
        connection.commit()
    except BaseException:
        connection.close()
        staging.unlink(missing_ok=True)
        raise
    else:
        connection.close()
    staging.replace(db_path)
    return {
        "db": str(db_path),
        "groups": len(wanted),
        "members": members,
        "samples": samples,
        "source_paths": reverse,
        "bytes": db_path.stat().st_size,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nfs/bc/data/datasets"))
    parser.add_argument("--db", type=Path, default=None,
                        help="默认取 VERADATA_CATALOG 或 ~/.local/state/veradata/global.sqlite3")
    parser.add_argument("--group", action="append", default=[],
                        help="只重建这些 group（可重复）；不给就全量重建")
    args = parser.parse_args(argv)
    db_path = args.db or default_db()
    try:
        summary = (
            upsert(args.dataset_root, db_path, args.group)
            if args.group
            else rebuild(args.dataset_root, db_path)
        )
        print(json.dumps(summary, sort_keys=True))
    except (IndexedTarError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
