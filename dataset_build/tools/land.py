"""The single door through which new data reaches the archive.

Production writes small files to SSD staging, then calls ``land()``: it derives
the metadata, plans the archived layout, packs an uncompressed indexed tar onto
NFS, verifies it, and only then drops the staging copy.  Nothing reaches the
archive without metadata, and nothing leaves staging until its shard verifies.

Each call publishes a fresh ``<group>/batch-NNNN`` dataset, so landings are
append-only and a failed batch never corrupts a published one.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Mapping

from dataset_build.tools.dataset_plan import (
    ImageSource,
    PlanError,
    PlanGroup,
    PlanRow,
    _iter_samples,
    write_group,
)
from dataset_build.tools.indexed_tar import (
    DEFAULT_READ_WORKERS,
    DEFAULT_SHARD_SIZE,
    IndexedTarError,
    build_indexed_tar,
    verify_dataset,
)


_BATCH_RE = re.compile(r"batch-(\d{4})\Z")
_GROUP_RE = re.compile(r"[a-z0-9_]+(?:/[^/\x00]+){1,3}\Z")


def next_batch(group_dir: Path) -> str:
    """Name the next batch directory without reusing a published one."""
    highest = -1
    if group_dir.is_dir():
        for child in group_dir.iterdir():
            match = _BATCH_RE.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"batch-{highest + 1:04d}"


def land(
    staging: Path,
    group: str,
    out_root: Path,
    *,
    plan_root: Path,
    meta_staging: Path,
    sample_meta: Mapping[str, Mapping[str, object]] | None = None,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE,
    read_workers: int = DEFAULT_READ_WORKERS,
    keep_staging: bool = False,
) -> dict[str, object]:
    """Clean, plan, pack, verify, then retire one staging batch."""
    staging = Path(staging).resolve(strict=True)
    if not staging.is_dir():
        raise PlanError(f"staging is not a directory: {staging}")
    if not _GROUP_RE.fullmatch(group):
        raise PlanError(f"group must look like kind/class[/class]: {group!r}")

    batch = next_batch(Path(out_root) / group)
    archive_group = f"{group}/{batch}"
    source = ImageSource(corpus=batch, root=staging)
    bucket = PlanGroup(group=archive_group)
    for sample in _iter_samples(source):
        for path, extension, role in sample.members:
            bucket.add(
                PlanRow(
                    path=path,
                    logical_path=f"{archive_group}/{sample.key}{extension}",
                    meta={"group": archive_group, "role": role},
                )
            )
    if not bucket.rows:
        raise PlanError(f"staging holds no files to land: {staging}")

    plan_dir = Path(plan_root) / archive_group
    summary = write_group(
        bucket,
        plan_dir,
        meta_staging=Path(meta_staging) / archive_group,
        sample_meta=sample_meta,
    )
    output = Path(out_root) / archive_group
    manifest = build_indexed_tar(
        None,
        output,
        plan=Path(summary["plan"]),
        shard_size_bytes=shard_size_bytes,
        read_workers=read_workers,
    )
    verified = verify_dataset(output)
    if verified["members"] != manifest["member_count"]:
        raise IndexedTarError(f"verify disagrees with manifest for {output}")
    # The archive keeps its own metadata so it stays readable after staging dies.
    shutil.copyfile(summary["metadata"], output / "metadata.jsonl")
    if not keep_staging:
        shutil.rmtree(staging)
    return {
        "group": archive_group,
        "dataset": str(output),
        "members": manifest["member_count"],
        "samples": manifest["sample_count"],
        "payload_bytes": manifest["payload_bytes"],
        "staging_removed": not keep_staging,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--group", required=True, help="e.g. renders/r7_global or cgt/s5")
    parser.add_argument("--out-root", type=Path, default=Path("/mnt/nfs/bc/data/datasets"))
    parser.add_argument("--plan-root", type=Path, default=Path("/tmp/veradata/plans"))
    parser.add_argument("--meta-staging", type=Path, default=Path("/tmp/veradata/meta"))
    parser.add_argument("--read-workers", type=int, default=DEFAULT_READ_WORKERS)
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = land(
            args.staging,
            args.group,
            args.out_root,
            plan_root=args.plan_root,
            meta_staging=args.meta_staging,
            read_workers=args.read_workers,
            keep_staging=args.keep_staging,
        )
    except (PlanError, IndexedTarError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
