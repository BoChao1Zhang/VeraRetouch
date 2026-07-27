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
import sqlite3
import sys
from pathlib import Path
from typing import Iterator

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
            groups += 1
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nfs/bc/data/datasets"))
    parser.add_argument("--db", type=Path, default=None,
                        help="默认取 VERADATA_CATALOG 或 ~/.local/state/veradata/global.sqlite3")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(rebuild(args.dataset_root, args.db or default_db()), sort_keys=True))
    except (IndexedTarError, OSError, sqlite3.Error, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
