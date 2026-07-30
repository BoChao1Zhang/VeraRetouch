"""Backfill the precomputed eligibility gate onto an already published cache group.

``land`` records the gate (``eligible`` / ``ineligible_reason`` / ``mask_area`` …)
while a batch is packed, but the ``cache/subject`` group predates that hook, so
its samples carry no such fields and ``build_inventory`` falls back to decoding
every entry.  This tool computes the same fields for a published group and
rewrites its ``metadata.jsonl``; rebuild the global catalog afterwards to make
them queryable.

The shard is read **sequentially** — members in ascending offset order, one
forward-seeking pass per tar — because the archive lives on NFS where 150k
random preads cost minutes and one streaming pass costs seconds.

The ``.vrmeta.json`` member inside the tar keeps its original payload: rewriting
it would mean repacking the shard, and ``metadata.jsonl`` is what the global
catalog projects into ``samples.meta``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dataset_build.tools.archive_reader import CATALOG_ENV
from dataset_build.tools.dataset_plan import META_SUFFIX
from dataset_build.tools.indexed_tar import (
    IndexedTarError,
    _iter_index_records,
    _load_manifest,
)
from dataset_build.tools.land import SUBJECT_CACHE_GROUP, subject_sample_meta


WANTED = ("subject.json", "subject.png")


def _member_origins(rows: list[dict[str, object]]) -> dict[str, str]:
    """Map each real member to the absolute path it was archived from.

    The ``.vrmeta.json`` row is skipped: it points at the transient staging file
    the planner materialised, not at anything that still exists.
    """
    origins: dict[str, str] = {}
    for row in rows:
        member, source_path = str(row.get("member") or ""), str(row.get("source_path") or "")
        if member and source_path and not member.endswith(META_SUFFIX):
            origins[member] = source_path
    return origins


def _stream_fields(group_dir: Path, origins: dict[str, str]) -> dict[str, dict[str, object]]:
    """One forward pass per shard, yielding the gate fields per sample."""
    manifest = _load_manifest(group_dir)
    fields: dict[str, dict[str, object]] = {}
    pending: dict[str, bytes] = {}
    pending_id: str | None = None
    pending_dir: Path | None = None

    def flush() -> None:
        nonlocal pending, pending_id, pending_dir
        if pending_id is not None:
            fields[pending_id] = subject_sample_meta(
                subject_json=pending.get("subject.json"),
                subject_png=pending.get("subject.png"),
                cache_dir=pending_dir,
            )
        pending, pending_id, pending_dir = {}, None, None

    for shard in manifest["shards"]:
        index_path = group_dir / str(shard["index"])
        tar_path = group_dir / str(shard["tar"])
        records = sorted(_iter_index_records(index_path), key=lambda item: item.offset_data)
        with tar_path.open("rb") as handle:
            for record in records:
                if record.sample_id != pending_id:
                    flush()
                    pending_id = record.sample_id
                origin = origins.get(record.member)
                name = Path(origin).name if origin else ""
                if name not in WANTED:
                    continue
                handle.seek(record.offset_data)
                payload = handle.read(record.size)
                if len(payload) != record.size:
                    raise IndexedTarError(f"truncated member in {tar_path}: {record.member}")
                pending[name] = payload
                pending_dir = Path(origin).parent
        flush()
    return fields


def backfill(dataset_root: Path, group: str) -> dict[str, object]:
    """Recompute and rewrite one published group's sample metadata in place."""
    group_dir = Path(dataset_root) / group
    metadata_path = group_dir / "metadata.jsonl"
    if not metadata_path.is_file():
        raise IndexedTarError(f"group has no metadata.jsonl: {metadata_path}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]

    fields = _stream_fields(group_dir, _member_origins(rows))

    counts: dict[str, int] = {}
    written = 0
    # Atomic: a half-written metadata.jsonl would leave the group unreadable, and
    # the catalog projects this file verbatim.
    temporary = metadata_path.with_name(metadata_path.name + ".backfill")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            extra = (
                fields.get(str(row.get("sample_id") or ""))
                if str(row.get("member") or "").endswith(META_SUFFIX)
                else None
            )
            if extra is not None:
                row = {**row, **extra}
                reason = str(extra["ineligible_reason"] or "eligible")
                counts[reason] = counts.get(reason, 0) + 1
                written += 1
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not written:
        # 没有样本级行就无处安放门控字段：这不是"已回填"，而是组的形状不对。
        temporary.unlink(missing_ok=True)
        raise IndexedTarError(f"group has no sample-level metadata rows: {metadata_path}")
    os.replace(temporary, metadata_path)
    return {
        "group": group,
        "metadata": str(metadata_path),
        "samples": written,
        "counts": dict(sorted(counts.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/mnt/nfs/bc/data/datasets"))
    parser.add_argument("--group", default=SUBJECT_CACHE_GROUP)
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="global catalog used to check that each source image still exists "
        f"(default: ${CATALOG_ENV} or the built-in path)",
    )
    args = parser.parse_args(argv)
    if args.db is not None:
        os.environ[CATALOG_ENV] = str(args.db)
    try:
        result = backfill(args.dataset_root, args.group)
    except (IndexedTarError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    print(
        "next: python -m dataset_build.tools.global_catalog "
        f"--dataset-root {args.dataset_root}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
