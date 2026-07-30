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
import io
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping

from dataset_build.tools.archive_reader import path_exists
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

# One sample's members, keyed by archive extension, as an enrich hook sees them.
Enrich = Callable[[str, Mapping[str, Path]], Mapping[str, object] | None]

SUBJECT_CACHE_GROUP = "cache/subject"
# A directory that is not a cache entry at all (``_``-prefixed scratch, or no
# ``subject.json``).  ``build_inventory`` never enumerated those, so they must
# stay out of the precomputed counts rather than land in a rejection bucket.
NOT_A_CACHE_ENTRY = "not_a_cache_entry"
SUBJECT_MIN_AREA = 0.005
SUBJECT_MAX_AREA = 0.85


def next_batch(group_dir: Path) -> str:
    """Name the next batch directory without reusing a published one."""
    highest = -1
    if group_dir.is_dir():
        for child in group_dir.iterdir():
            match = _BATCH_RE.match(child.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return f"batch-{highest + 1:04d}"


def subject_sample_meta(
    *,
    subject_json: bytes | None,
    subject_png: bytes | None,
    cache_dir: Path | None,
) -> dict[str, object]:
    """Decide one subject-cache entry's eligibility from its member bytes.

    This is the precomputation of ``construct.sources._inspect_cache_dir``: the
    same checks in the same order, so an archived entry can be gated by one SQL
    query instead of re-reading and re-decoding three files per source at every
    cold start.  ``mask_area`` is the one value no index can carry, and it is
    cheapest here — the mask comes off SSD staging the packer is about to read
    anyway (or off the backfill's sequential shard stream).

    One deliberate divergence: the source image is checked for existence but not
    decoded.  Its bytes were verified against their SHA-256 when the shard was
    published, and re-decoding every source would cost the whole 67 GiB read this
    change exists to avoid.
    """
    fields: dict[str, object] = {
        "cache_dir": str(cache_dir) if cache_dir is not None else None,
        "eligible": False,
        "ineligible_reason": None,
        "mask_area": None,
        # PostgreSQL is the scene authority while it lives; this is the value the
        # inventory falls back to once it is gone.
        "scene": "unknown",
        "asset_id": None,
        "source_path": None,
        "subject": None,
    }

    def reject(reason: str) -> dict[str, object]:
        fields["ineligible_reason"] = reason
        return fields

    if subject_json is None or (cache_dir is not None and cache_dir.name.startswith("_")):
        return reject(NOT_A_CACHE_ENTRY)
    try:
        meta = json.loads(subject_json.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return reject("invalid_subject_json")
    if not isinstance(meta, dict) or meta.get("status") != "ready":
        return reject("subject_not_ready")
    if not subject_png:
        # ``path_exists`` counts a zero-byte file as absent, so this must too.
        return reject("missing_subject_png")
    source_value = meta.get("source_path")
    if not isinstance(source_value, str) or not source_value:
        return reject("missing_source_path")
    if not path_exists(Path(source_value)):
        return reject("missing_source_image")
    try:
        import numpy as np
        from PIL import Image

        with Image.open(io.BytesIO(subject_png)) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
        if mask.ndim != 2 or not mask.size or not np.isfinite(mask).all():
            return reject("invalid_subject_mask")
        area = float((mask > 0.5).mean())
    except Exception:  # noqa: BLE001 - a corrupt mask is an eligibility failure
        return reject("decode_or_integrity_failure")
    if area < SUBJECT_MIN_AREA or area > SUBJECT_MAX_AREA:
        return reject("subject_mask_area_guard")
    fields.update(
        eligible=True,
        mask_area=area,
        asset_id=str(meta.get("asset_id") or "") or None,
        source_path=source_value,
        subject={
            "name": meta.get("sam_prompt") or meta.get("description") or "subject",
            "description": meta.get("description"),
            "scope": meta.get("scope"),
            "n_members": meta.get("n_members"),
            "area": round(area, 6),
        },
    )
    return fields


def subject_enrich(sample_key: str, members: Mapping[str, Path]) -> dict[str, object]:
    """``enrich`` hook for ``cache/subject``: precompute the eligibility gate."""
    by_name = {path.name: path for path in members.values()}
    meta_path, mask_path = by_name.get("subject.json"), by_name.get("subject.png")
    anchor = meta_path or mask_path
    return subject_sample_meta(
        subject_json=meta_path.read_bytes() if meta_path is not None else None,
        subject_png=mask_path.read_bytes() if mask_path is not None else None,
        cache_dir=anchor.parent if anchor is not None else None,
    )


ENRICHERS: dict[str, Enrich] = {SUBJECT_CACHE_GROUP: subject_enrich}


def enrich_for_group(group: str) -> Enrich | None:
    """The enrich hook a group earns, so the CLI needs no extra flag."""
    return ENRICHERS.get(group)


def land(
    staging: Path,
    group: str,
    out_root: Path,
    *,
    plan_root: Path,
    meta_staging: Path,
    sample_meta: Mapping[str, Mapping[str, object]] | None = None,
    enrich: Enrich | None = None,
    source_paths: Mapping[str, str] | None = None,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE,
    read_workers: int = DEFAULT_READ_WORKERS,
    keep_staging: bool = False,
) -> dict[str, object]:
    """Clean, plan, pack, verify, then retire one staging batch.

    ``enrich`` is an optional hook over one sample's member bytes: it returns
    extra metadata that is merged into that sample's ``.vrmeta.json`` and
    ``metadata.jsonl`` row, so a derived fact that only the bytes can answer is
    recorded once at landing instead of recomputed on every consumer's cold start.

    ``source_paths`` maps a staged file to the path the producer recorded for it.
    A producer that stages hardlinks (the databuild land checkpoint does, so the
    same bytes can be published into two datasets) would otherwise teach the
    catalog's reverse map a transient staging path, and ``read_bytes`` would stop
    resolving the path already written into groups.jsonl.
    """
    staging = Path(staging).resolve(strict=True)
    if not staging.is_dir():
        raise PlanError(f"staging is not a directory: {staging}")
    if not _GROUP_RE.fullmatch(group):
        raise PlanError(f"group must look like kind/class[/class]: {group!r}")

    batch = next_batch(Path(out_root) / group)
    archive_group = f"{group}/{batch}"
    source = ImageSource(corpus=batch, root=staging)
    bucket = PlanGroup(group=archive_group)
    enriched: dict[str, dict[str, object]] = {
        key: dict(value) for key, value in (sample_meta or {}).items()
    }
    for sample in _iter_samples(source):
        for path, extension, role in sample.members:
            meta: dict[str, object] = {"group": archive_group, "role": role}
            alias = (source_paths or {}).get(str(path))
            if alias:
                meta["source_path"] = alias
            bucket.add(
                PlanRow(
                    path=path,
                    logical_path=f"{archive_group}/{sample.key}{extension}",
                    meta=meta,
                )
            )
        if enrich is not None:
            extra = enrich(sample.key, {ext: path for path, ext, _role in sample.members})
            if extra:
                enriched.setdefault(sample.key, {}).update(extra)
    if not bucket.rows:
        raise PlanError(f"staging holds no files to land: {staging}")

    plan_dir = Path(plan_root) / archive_group
    summary = write_group(
        bucket,
        plan_dir,
        meta_staging=Path(meta_staging) / archive_group,
        sample_meta=enriched or None,
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
            enrich=enrich_for_group(args.group),
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
