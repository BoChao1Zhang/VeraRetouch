"""Write indexed tar shards from in-memory payloads.

``dataset_build.tools.indexed_tar.build_indexed_tar`` can only archive files that
already exist on disk, and materialising 172k derived records (or 172k resized
JPEGs) as loose files first is exactly the "million small files" the storage
contract forbids.  This module keeps the canonical *format* by reusing that
module's shard writer, catalog, validator and manifest layout, and only replaces
the source of the bytes.

Everything the contract asks for therefore still holds and is still enforced by
the canonical code:

* uncompressed USTAR, shard rotation on sample boundaries at a configurable
  target size (METACANVAS 2.3: 1-4 GiB);
* an index row per member with ``shard/member/offset/length/size/sha256`` plus
  ``schema_version``;
* per-shard tar+index SHA-256 in the manifest, plus member and sample counts;
* every shard validated (``validate_shard``) before ``os.replace``, the whole
  dataset staged under a hidden directory and published with a single rename,
  so a partial shard can never appear in the terminal manifest;
* a SQLite catalog verified row-by-row against the JSONL indexes.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from dataset_build.tools.indexed_tar import (
    KEY_POLICY,
    MEMBER_ORDER_PLAN,
    SCHEMA_VERSION,
    IndexedTarError,
    _create_catalog,
    _dataset_id,
    _fsync_dir,
    _insert_catalog,
    _json_bytes,
    _prepare_output,
    _PreparedSource,
    _sha256_file,
    _ShardWriter,
    _SourceEntry,
    _stable_member,
    _verify_catalog,
)

Payload = tuple[str, bytes]  # (logical_path, data)


class _MemStat:
    """Minimal ``os.stat_result`` stand-in: only ``st_size`` is ever read."""

    __slots__ = ("st_size",)

    def __init__(self, size: int) -> None:
        self.st_size = size


def _entry(logical_path: str, data: bytes) -> _PreparedSource:
    _stable_member(logical_path)  # validates the key policy before anything is written
    source = _SourceEntry(Path("<memory>"), logical_path, _MemStat(len(data)))
    return _PreparedSource(entry=source, data=data, sha256=hashlib.sha256(data).hexdigest())


def build_from_memory(
    payloads: Iterable[Payload],
    output_root: Path,
    *,
    shard_size_bytes: int,
    producer: str,
    source_label: str,
    progress: Callable[[int, int], None] | None = None,
    progress_every: int = 20000,
) -> dict[str, Any]:
    """Pack ``(logical_path, bytes)`` pairs into a published indexed tar dataset.

    Members land in iteration order and rotate to a new shard only on a sample
    boundary, so a sequential reader always sees a sample's members together.
    """
    output_root = _prepare_output(Path(output_root))
    staging = output_root.parent / f".{output_root.name}.partial.{os.getpid()}.{uuid.uuid4().hex[:12]}"
    staging.mkdir(mode=0o700)
    (staging / "shards").mkdir()
    (staging / "indexes").mkdir()
    _fsync_dir(staging)

    catalog_tmp = staging / "indexes" / "catalog.sqlite3.tmp"
    catalog_final = staging / "indexes" / "catalog.sqlite3"
    catalog = _create_catalog(catalog_tmp)
    shard_metadata: list[dict[str, Any]] = []
    current: _ShardWriter | None = None
    current_key: str | None = None
    total_members = total_samples = total_payload = 0

    try:
        for logical_path, data in payloads:
            prepared = _entry(logical_path, data)
            key = _stable_member(logical_path)[0]
            if current is None:
                current = _ShardWriter(staging, f"shard-{len(shard_metadata):05d}")
            elif (current.member_count and key != current_key
                  and current.projected_size(len(data)) > shard_size_bytes):
                shard_metadata.append(current.close_and_publish())
                current = _ShardWriter(staging, f"shard-{len(shard_metadata):05d}")
            if key != current_key:
                current_key = key
                total_samples += 1
            record = current.add_prepared(prepared)
            _insert_catalog(catalog, record)
            total_members += 1
            total_payload += record.size
            if progress and total_members % progress_every == 0:
                progress(total_members, total_payload)

        if current is None:
            raise IndexedTarError(f"nothing to pack for {source_label}")
        shard_metadata.append(current.close_and_publish())
        current = None

        catalog.commit()
        catalog.close()
        with catalog_tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(catalog_tmp, catalog_final)
        _fsync_dir(catalog_final.parent)
        _verify_catalog(catalog_final,
                        [staging / str(s["index"]) for s in shard_metadata],
                        total_members)
        catalog_sha256, catalog_bytes = _sha256_file(catalog_final)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "complete",
            "dataset_id": _dataset_id(shard_metadata),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "producer": producer,
            "source_root": source_label,
            "input_mode": "memory",
            "archive_format": "ustar",
            "compression": "none",
            "member_order": MEMBER_ORDER_PLAN,
            "key_policy": KEY_POLICY,
            "target_shard_size_bytes": shard_size_bytes,
            "read_workers": 1,
            "prefetch_files": 1,
            "prefetch_bytes": 1,
            "parallel_file_max": 1,
            "member_count": total_members,
            "sample_count": total_samples,
            "payload_bytes": total_payload,
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


def read_member(root: Path, shard: str, offset: int, length: int,
                sha256: str | None = None) -> bytes:
    """Positional read of one member, optionally checked against its digest."""
    path = Path(root) / "shards" / f"{shard}.tar"
    fd = os.open(path, os.O_RDONLY)
    try:
        data = os.pread(fd, length, offset)
    finally:
        os.close(fd)
    if len(data) != length:
        raise IndexedTarError(f"short read of {shard}:{offset} ({len(data)} != {length})")
    if sha256 is not None:
        got = hashlib.sha256(data).hexdigest()
        if got != sha256:
            raise IndexedTarError(f"checksum mismatch for {shard}:{offset}")
    return data


def iter_index(root: Path) -> Iterator[dict[str, Any]]:
    """Yield every index row of a published dataset, shard by shard."""
    import json

    for index_path in sorted((Path(root) / "indexes").glob("shard-*.idx.jsonl")):
        with index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
