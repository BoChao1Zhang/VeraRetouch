"""Rebuild a build's ``sft/<id>`` dataset from its groups archive and sft.jsonl.

Production publishes both datasets at every land checkpoint, but the authority
between them is asymmetric: the intermediate ``groups/<id>`` tar holds every
candidate and is the authority, while ``sft/<id>`` is a projection of it — the
winner's ``I_in`` + ``I_tar`` + ``C_GT`` plus a light metadata member.  This tool
is that projection run offline: to repair drift, or to pack a historical build
whose groups archive predates the double-write.

Reads follow the archive's physical order — the winner members are collected
first, sorted by ``(shard, offset_data)`` and streamed shard by shard — because
the alternative, one random pread per record over a 98 MB/s link, is what makes
"reindex the winners afterwards" slow in the first place.  ``I_in`` goes through
``archive_reader.read_bytes``, so it costs nothing when the source is still local
or already in the prefetch buffer.

The rebuilt member order is the sft record order (``preserve_order=True``), which
is what a sequential WebDataset reader consumes at training time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from dataset_build.tools import archive_reader
from dataset_build.tools.dataset_plan import (
    PlanError,
    PlanGroup,
    PlanRow,
    _split_member_name,
    sanitize_key,
    write_group,
)
from dataset_build.tools.indexed_tar import (
    DEFAULT_READ_WORKERS,
    DEFAULT_SHARD_SIZE,
    MEMBER_ORDER_PLAN,
    IndexedTarError,
    _iter_index_records,
    _load_manifest,
    build_indexed_tar,
    verify_dataset,
)
# 与 prefetch 同一件事：从已发布的 shard 里按物理序读成员，头部与 catalog 对账。
from dataset_build.tools.prefetch import _read_member, _shard_path


@dataclass(frozen=True)
class _Member:
    """Where one archived member physically lives."""

    root: str
    shard: str
    member: str
    offset_data: int
    size: int
    sha256: str


def load_sft_rows(build_dir: Path) -> list[dict[str, object]]:
    """Read ``sft.jsonl`` in record order, dropping repeats of an ``sft_id``.

    The journal is append-only and deduplicated on load by the build's own store,
    so a repeated ID here is a resumed write of the same record, not a second
    sample; keeping the first occurrence preserves production order.
    """
    path = Path(build_dir) / "sft.jsonl"
    if not path.is_file():
        raise PlanError(f"sft.jsonl is missing: {path}")
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"invalid JSON at {path}:{number}: {exc}") from exc
            if not isinstance(row, dict):
                raise PlanError(f"JSON object required at {path}:{number}")
            sft_id = str(row.get("sft_id") or "")
            if not sft_id:
                raise PlanError(f"record without sft_id at {path}:{number}")
            if sft_id in seen:
                continue
            seen.add(sft_id)
            rows.append(row)
    if not rows:
        raise PlanError(f"no SFT records in {path}")
    return rows


def dataset_roots(groups_dataset: Path) -> list[Path]:
    """Accept either one published dataset or the group directory of batches.

    A build lands once per checkpoint, so its intermediate products are usually
    ``groups/<build_id>/batch-0000`` … rather than a single dataset.
    """
    root = Path(groups_dataset)
    if (root / "manifest.json").is_file():
        return [root]
    roots = sorted(child.parent for child in root.glob("*/manifest.json"))
    if not roots:
        raise PlanError(f"no published dataset under {root}")
    return roots


def index_by_source_path(roots: Iterable[Path]) -> dict[str, _Member]:
    """把每个成员的原始 staging 路径映射到它在归档里的物理位置。

    ``metadata.jsonl`` is the reverse map (it carries ``source_path`` per member)
    and the JSONL indexes are the placement authority, so the two together answer
    "where did this winner's file end up" without the global catalog — which a
    historical build may predate.
    """
    index: dict[str, _Member] = {}
    for root in roots:
        manifest = _load_manifest(Path(root))
        placement: dict[str, _Member] = {}
        for shard in manifest["shards"]:
            for record in _iter_index_records(Path(root) / str(shard["index"])):
                placement[record.member] = _Member(
                    root=str(root),
                    shard=record.shard,
                    member=record.member,
                    offset_data=record.offset_data,
                    size=record.size,
                    sha256=record.sha256,
                )
        metadata = Path(root) / "metadata.jsonl"
        if not metadata.is_file():
            raise PlanError(f"published dataset has no metadata.jsonl: {root}")
        with metadata.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                member = str(row.get("member") or "")
                source_path = str(row.get("source_path") or "")
                # The metadata member itself is materialised in transient staging,
                # so its source path points nowhere useful.
                if not member or not source_path or member.endswith(".vrmeta.json"):
                    continue
                located = placement.get(member)
                if located is None:
                    raise PlanError(f"metadata names a member no index holds: {member} in {root}")
                # First batch wins: a path re-landed later is the same bytes.
                index.setdefault(source_path, located)
    return index


def _extension(role: str, source_path: str) -> str:
    """Role-qualified archive extension, e.g. ``.tar.jpg`` for the winner render."""
    return f".{role}{_split_member_name(Path(source_path).name)[1]}"


def _sample_meta(row: Mapping[str, object], members: Mapping[str, str]) -> dict[str, object]:
    """The light per-sample metadata member.

    Deliberately the identifiers, the routing fields and the QA block only.  The
    instruction/reasoning text is not copied: it is long, it is the part most
    likely to be re-annotated, and it already lives in ``sft.jsonl`` (mirrored to
    the NFS builds layer), which the training loader joins on ``sft_id`` anyway.
    """
    return {
        "sft_id": row.get("sft_id"),
        "build_id": row.get("build_id"),
        "annotation_task_id": row.get("annotation_task_id"),
        "group_id": row.get("group_id"),
        "candidate_id": row.get("candidate_id"),
        "winner_rank": row.get("winner_rank"),
        # "low" marks a winner the ranking could not separate from rank2 by more
        # than the scorer's resolution, so training can down-weight or drop it.
        # ``None`` is a row from before the policy, not a normal winner.
        "winner_confidence": row.get("winner_confidence"),
        "task_type": row.get("task_type"),
        "annot_src": row.get("annot_src"),
        # Which model actually wrote the text.  ``.get`` keeps the rows written
        # before the field existed packable; they land as ``None``.
        "annot_model": row.get("annot_model"),
        "recipe": row.get("recipe"),
        "qa": row.get("qa"),
        "members": dict(members),
    }


def _row_sources(row: Mapping[str, object]) -> list[tuple[str, str]]:
    """The (role, original path) members one SFT record contributes."""
    sources = [("in", str(row.get("I_in") or "")), ("tar", str(row.get("I_tar") or ""))]
    local = row.get("local")
    cgt = str((local or {}).get("C_GT") or "") if isinstance(local, dict) else ""
    if cgt:
        sources.append(("cgt", cgt))
    if not sources[0][1] or not sources[1][1]:
        raise PlanError(f"record {row.get('sft_id')} lacks I_in or I_tar")
    return sources


def _drain_from_archive(work: list[tuple[_Member, list[Path]]]) -> int:
    """Copy located members into staging, one forward pass per shard.

    ``work`` is consumed in ``(root, shard, offset_data)`` order so each shard is
    opened once and walked forward; a member several records share is read once
    and written to each of its staging names.
    """
    shard_paths: dict[tuple[str, str], str] = {}
    descriptor: int | None = None
    open_shard: str | None = None
    written = 0
    try:
        for located, targets in work:
            path = _shard_path(located.root, located.shard, shard_paths)
            if path != open_shard:
                if descriptor is not None:
                    os.close(descriptor)
                descriptor = os.open(path, os.O_RDONLY)
                open_shard = path
            assert descriptor is not None
            payload = _read_member(
                descriptor,
                {
                    "source_path": located.member,
                    "member": located.member,
                    "offset_data": located.offset_data,
                    "size": located.size,
                },
            )
            if hashlib.sha256(payload).hexdigest() != located.sha256:
                raise IndexedTarError(f"payload checksum mismatch: {located.member}")
            for target in targets:
                target.write_bytes(payload)
                written += 1
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return written


def sft_pack(
    build_dir: Path,
    groups_dataset: Path,
    output: Path,
    *,
    db_path: Path | None = None,
    staging_root: Path | None = None,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE,
    read_workers: int = DEFAULT_READ_WORKERS,
    skip_missing: bool = False,
) -> dict[str, object]:
    """Rebuild ``sft/<build_id>`` and return the landing summary."""
    build_dir = Path(build_dir)
    output = Path(output).absolute()
    rows = load_sft_rows(build_dir)
    build_id = str(rows[0].get("build_id") or build_dir.name)
    group_name = f"sft/{build_id}"
    index = index_by_source_path(dataset_roots(groups_dataset))

    staging = Path(
        tempfile.mkdtemp(prefix=f"sft_pack.{build_id}.", dir=str(staging_root) if staging_root else None)
    )
    try:
        bucket = PlanGroup(group=group_name)
        sample_meta: dict[str, dict[str, object]] = {}
        archive_work: dict[str, list[Path]] = {}
        local_reads: list[tuple[str, Path]] = []
        skipped: list[dict[str, str]] = []
        for row in rows:
            key = sanitize_key(str(row["sft_id"]))
            sources = _row_sources(row)
            missing = [path for role, path in sources if role != "in" and path not in index]
            if missing:
                if not skip_missing:
                    raise PlanError(
                        f"record {row['sft_id']} references a member the groups dataset "
                        f"does not hold: {missing[0]}"
                    )
                skipped.append({"sft_id": str(row["sft_id"]), "missing": missing[0]})
                continue
            members: dict[str, str] = {}
            for role, source_path in sources:
                extension = _extension(role, source_path)
                target = staging / f"{key}{extension}"
                if role == "in":
                    local_reads.append((source_path, target))
                else:
                    archive_work.setdefault(source_path, []).append(target)
                members[extension] = source_path
                bucket.add(
                    PlanRow(
                        path=target,
                        logical_path=f"{group_name}/{key}{extension}",
                        # source_path 覆盖成原始产出路径而不是本次的临时 staging 名：
                        # staging 随本次运行消失，而原始路径正是 groups 侧记的同一个
                        # 键，两个数据集的反查表因此指向同一来源。
                        meta={"group": group_name, "role": role, "source_path": source_path},
                    )
                )
            sample_meta[key] = _sample_meta(row, members)
        if not bucket.rows:
            raise PlanError(f"every SFT record was skipped for {build_dir}")

        ordered = sorted(
            ((index[path], targets) for path, targets in archive_work.items()),
            key=lambda item: (item[0].root, item[0].shard, item[0].offset_data),
        )
        _drain_from_archive(ordered)
        for source_path, target in local_reads:
            # local-first → prefetch buffer → archive, so a still-local or already
            # buffered source never touches NFS here.
            target.write_bytes(archive_reader.read_bytes(source_path, db_path=db_path))

        plan_dir = staging / "_plan"
        summary = write_group(
            bucket,
            plan_dir,
            meta_staging=staging / "_meta",
            sample_meta=sample_meta,
            preserve_order=True,
        )
        manifest = build_indexed_tar(
            None,
            output,
            plan=Path(str(summary["plan"])),
            shard_size_bytes=shard_size_bytes,
            read_workers=read_workers,
            member_order=MEMBER_ORDER_PLAN,
        )
        verified = verify_dataset(output)
        if verified["members"] != manifest["member_count"]:
            raise IndexedTarError(f"verify disagrees with manifest for {output}")
        shutil.copyfile(str(summary["metadata"]), output / "metadata.jsonl")
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "group": group_name,
        "dataset": str(output),
        "records": len(rows),
        "samples": manifest["sample_count"],
        "members": manifest["member_count"],
        "payload_bytes": manifest["payload_bytes"],
        "skipped": skipped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True, help="holds sft.jsonl")
    parser.add_argument(
        "--groups-dataset", type=Path, required=True,
        help="published groups/<build_id> directory, or one batch inside it",
    )
    parser.add_argument("--output", type=Path, required=True, help="new sft/<build_id> dataset")
    parser.add_argument("--db", type=Path, default=None, help="global catalog for I_in reads")
    parser.add_argument(
        "--staging", type=Path, default=None,
        help="where the rebuilt bytes are materialised (default: TMPDIR)",
    )
    parser.add_argument("--read-workers", type=int, default=DEFAULT_READ_WORKERS)
    parser.add_argument(
        "--skip-missing", action="store_true",
        help="drop records whose winner assets the groups dataset lost, instead of failing",
    )
    args = parser.parse_args(argv)
    try:
        result = sft_pack(
            args.build_dir,
            args.groups_dataset,
            args.output,
            db_path=args.db,
            staging_root=args.staging,
            read_workers=args.read_workers,
            skip_missing=args.skip_missing,
        )
    except (PlanError, IndexedTarError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
