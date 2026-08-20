#!/usr/bin/env python3
"""Build the deterministic agent-loop source manifest (task cards A3a / A3a-v2).

Emits one JSONL row per eligible source in the schema ``annotate-sources``
consumes (see ``configs/agent_loop.smoke5.jsonl``)::

    {"source_id", "source_path", "subject_path", "scene",
     "subject": {"description", "mask_area"}}

``source_annotation_path`` is deliberately absent: it is written by
``dataset_build.agent_loop.cli annotate-sources``.

Two candidate pools
-------------------
``--pool-source nfs`` (default, A3a-v2)
    The pool the canonical local400k config actually draws from: the published
    subject cache ``cache/subject`` on NFS.  Its ``metadata.jsonl`` carries one
    ``*.vrmeta.json`` row per cache entry with ``asset_id`` / ``source_path`` /
    ``mask_area`` / ``subject.description`` / ``members``.  Source bytes are
    resolved exactly the way ``databuild.prod-l8-local400k-*.toml`` resolves
    them: local file first, then the global catalog's reverse path map
    (``dataset_build.tools.archive_reader``) into the indexed tars.  NFS reads
    are re-rooted onto the read-only mount (``--nfs-read-root``); nothing is
    ever written to ``/mnt/nfs``.

``--pool-source backfill`` (A3a v1)
    The local ``mask_backfill`` pool plus the local ``subject_cache`` tree.

Materialisation
---------------
With ``--pool-source nfs`` the selected rows' source image and subject mask are
copied into ``--materialize-dir`` and the manifest points at those local files
(``source_annotations.py`` hashes and decodes both paths directly).  Members are
pread out of their tar at the catalogued offset; the tars are never unpacked.
The run is resumable: a previously written file whose size and SHA-256 match the
recorded index entry is left alone.

Determinism
-----------
No RNG, no timestamps, no wall-clock in the output.  Candidates are ordered by
``sha1(source_id)`` ascending (``source_id`` as tiebreak) and the first
``--target`` rows are kept, so two runs over unchanged inputs produce
byte-identical files.  A selected row that cannot be materialised is a hard
failure rather than a silent substitution, which keeps the manifest a pure
function of its inputs.

Eligibility (evaluated in this fixed order, one reason recorded per row)
-----------------------------------------------------------------------
``no_subject_mask`` -> ``no_asset_id`` -> ``source_unreadable`` -> ``no_scene``
-> ``eval_split_val`` / ``eval_split_test`` -> ``no_description`` ->
``no_mask_area`` -> ``mask_area_not_positive`` -> eligible.

Split discipline
----------------
The frozen S-split rule (``tools/data_splits/vr_common.py``) is
``int(sha1("verasplit-v1:" + source_id).hexdigest()[:8], 16) % 100`` with
0-89 train / 90-94 val / 95-99 test.  Only ``train`` sources are emitted.  The
rule is a pure function of ``source_id``, and where the frozen side table
``tools/data_splits/splits.sqlite3`` also carries the id, the two are asserted
to agree at runtime (any disagreement is a hard failure).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sqlite3
import sys
import tarfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_build.src.construct.sources import SCENE_WEIGHTS  # noqa: E402
from dataset_build.tools.archive_reader import ArchiveReader  # noqa: E402
from dataset_build.tools.indexed_tar import (  # noqa: E402
    BLOCK_SIZE,
    IndexedTarError,
    _load_manifest,
)

DEFAULT_POOL = Path("/home/bc/data/scratch/mask_backfill/backfill_pool.jsonl")
DEFAULT_SUBJECT_CACHE = Path("/home/bc/data/datasets/vera_directionA_1M/subject_cache")
DEFAULT_NFS_METADATA = Path("/mnt/nfs-ro/bc/data/datasets/cache/subject/metadata.jsonl")
DEFAULT_CATALOG = Path("/var/cache/veradata/global.sqlite3")
DEFAULT_MATERIALIZE = Path("/home/bc/data/agent_loop/local-v1/materialized")
DEFAULT_SPLIT_TABLE = REPO_ROOT / "tools" / "data_splits" / "splits.sqlite3"
DEFAULT_OUT = Path("/home/bc/data/agent_loop/local-v1/sources5k.jsonl")
DEFAULT_PG_DSN = os.environ.get(
    "SOURCE_QA_PG_DSN", "postgresql://research:research@127.0.0.1:5432/vera_source_qa"
)
# The catalog records shard roots under the read-write NFS mount; every read here
# is re-rooted onto the read-only mount, which is the only one this tool touches.
NFS_RW_ROOT = "/mnt/nfs/"
DEFAULT_NFS_RO_ROOT = "/mnt/nfs-ro/"

SPLIT_SEED = "verasplit-v1"
SPLIT_RULE = (
    'int(sha1("verasplit-v1:" + source_id).hexdigest()[:8], 16) % 100; '
    "0-89 train / 90-94 val / 95-99 test"
)
ORDER_RULE = "sha1(source_id).hexdigest() ascending, source_id as tiebreak"
MASK_AREA_DECIMALS = 4

EXCLUSION_ORDER = (
    "no_subject_mask",
    "no_asset_id",
    "source_unreadable",
    "no_scene",
    "eval_split_val",
    "eval_split_test",
    "no_description",
    "no_mask_area",
    "mask_area_not_positive",
)

# PIL format -> file suffix for the materialised copies.  The archived bytes are
# authoritative, not the original path's suffix: a source recorded as ``.jpg``
# can be archived as the pipeline's re-encoded PNG.
SUFFIX_BY_FORMAT = {
    "JPEG": ".jpg",
    "MPO": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "BMP": ".bmp",
    "GIF": ".gif",
}


# --------------------------------------------------------------------------- #
# split
# --------------------------------------------------------------------------- #
def s_split(source_id: str) -> str:
    digest = hashlib.sha1(f"{SPLIT_SEED}:{source_id}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "val"
    return "test"


def load_split_table(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return {sid: split for sid, split in connection.execute(
            "SELECT source_id, split FROM sources"
        )}
    finally:
        connection.close()


def assert_split_rule(source_ids: Iterable[str], table: dict[str, str]) -> int:
    """Cross-check the pure rule against the frozen side table. Hard failure."""
    checked = 0
    for source_id in source_ids:
        stored = table.get(source_id)
        if stored is None:
            continue
        checked += 1
        if stored != s_split(source_id):
            raise SystemExit(
                f"S-split disagreement for {source_id!r}: "
                f"side table {stored!r} vs rule {s_split(source_id)!r}"
            )
    return checked


# --------------------------------------------------------------------------- #
# archive access
# --------------------------------------------------------------------------- #
class ReadOnlyArchiveReader(ArchiveReader):
    """``ArchiveReader`` pinned to the read-only mount, safe under many threads.

    Two deliberate departures from the base class:

    * shard roots are re-rooted from ``/mnt/nfs`` onto ``/mnt/nfs-ro``;
    * ``read`` opens and closes its own descriptor instead of using the
      inherited 64-entry descriptor cache.  That cache evicts by closing a
      descriptor other threads may be mid-``pread`` on, which surfaces as
      ``EBADF`` and — worse — as silent ``truncated member`` / cross-file reads
      once a fresh open reuses the number.  A run that touches thousands of
      per-batch shards evicts constantly, so the cache is bypassed entirely.
    """

    def __init__(self, db_path: Path, ro_root: str = DEFAULT_NFS_RO_ROOT) -> None:
        super().__init__(db_path)
        self._ro_root = ro_root

    def _reroot(self, root: str) -> str:
        if root.startswith(NFS_RW_ROOT):
            return self._ro_root + root[len(NFS_RW_ROOT):]
        return root

    def _shard_descriptor(self, root: str, shard: str) -> int:
        return super()._shard_descriptor(self._reroot(root), shard)

    def shard_path(self, root: str, shard: str) -> str:
        root = self._reroot(root)
        key = (root, shard)
        with self._lock:
            path = self._shard_paths.get(key)
            if path is None:
                manifest = _load_manifest(Path(root))
                for item in manifest["shards"]:
                    self._shard_paths[(root, str(item["shard_id"]))] = str(
                        Path(root) / str(item["tar"])
                    )
                path = self._shard_paths.get(key)
                if path is None:
                    raise IndexedTarError(f"unknown shard {shard} under {root}")
        return path

    def read(self, source_path: str | os.PathLike[str]) -> bytes:
        row = self.locate(source_path)
        path = self.shard_path(str(row["root"]), str(row["shard"]))
        offset = int(row["offset_data"])
        size = int(row["size"])
        descriptor = os.open(path, os.O_RDONLY)
        try:
            header = os.pread(descriptor, BLOCK_SIZE, offset - BLOCK_SIZE)
            try:
                info = tarfile.TarInfo.frombuf(header, encoding="utf-8", errors="strict")
            except (tarfile.HeaderError, UnicodeError) as exc:
                raise IndexedTarError(f"invalid member header for {source_path}: {exc}") from exc
            if info.name != row["member"] or info.size != size:
                raise IndexedTarError(f"catalog does not match shard header for {source_path}")
            payload = b""
            while len(payload) < size:
                chunk = os.pread(descriptor, size - len(payload), offset + len(payload))
                if not chunk:
                    raise IndexedTarError(f"truncated member for {source_path}")
                payload += chunk
        finally:
            os.close(descriptor)
        return payload

    def resolvable(self, source_path: str) -> bool:
        if os.path.isfile(source_path):
            return True
        try:
            return int(self.locate(source_path)["size"]) > 0
        except KeyError:
            return False

    def physical_key(self, source_path: str) -> tuple[str, str, int]:
        """Sort key that turns a round of random archive reads into a forward pass."""
        if os.path.isfile(source_path):
            return ("", "", 0)
        try:
            row = self.locate(source_path)
        except KeyError:
            return ("", "", 0)
        return (str(row["root"]), str(row["shard"]), int(row["offset_data"]))


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pool(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_vrmeta(path: Path) -> list[dict[str, Any]]:
    """The ``*.vrmeta.json`` rows of a published cache's ``metadata.jsonl``.

    The file also carries one row per archived member (``.subject.json`` /
    ``.subject.png``); only the per-sample vrmeta rows describe the sample.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if ".vrmeta.json" not in line:
                continue
            entry = json.loads(line)
            if str(entry.get("logical_path", "")).endswith(".vrmeta.json"):
                rows.append(entry)
    return rows


def read_subject_cache(root: Path) -> dict[str, dict[str, Any]]:
    """path_key -> subject.json for entries that also carry a subject.png.

    ``path_key`` is the cache directory name and equals
    ``sha1(source_path).hexdigest()[:16]``.
    """
    ready: dict[str, dict[str, Any]] = {}
    for entry in sorted(os.listdir(root)):
        directory = root / entry
        meta = directory / "subject.json"
        if not meta.is_file() or not (directory / "subject.png").is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") == "ready":
            ready[entry] = payload
    return ready


def read_scenes(dsn: str) -> tuple[dict[str, str], dict[str, str]]:
    """(asset_id -> scene, path -> scene) from the vera_source_qa QA database."""
    import psycopg

    by_id: dict[str, str] = {}
    by_path: dict[str, str] = {}
    with psycopg.connect(dsn, connect_timeout=15) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT asset_id, scene, path FROM assets")
        for asset_id, scene, path in cursor.fetchall():
            if not scene:
                continue
            if asset_id:
                by_id[asset_id] = scene
            if path:
                by_path[path] = scene
    return by_id, by_path


def canonical_scene(scene: str) -> str:
    return scene if scene in SCENE_WEIGHTS else "unknown"


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def build_candidates(
    pool: list[dict[str, Any]],
    subjects: dict[str, dict[str, Any]],
    scene_by_id: dict[str, str],
    scene_by_path: dict[str, str],
    subject_cache: Path,
) -> tuple[list[dict[str, Any]], Counter, Counter, Counter]:
    """A3a v1 pool: local mask_backfill rows + local subject_cache."""
    excluded: Counter = Counter()
    raw_scenes: Counter = Counter()
    canonical_scenes: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for entry in pool:
        source_id = entry.get("asset_id")
        logical_path = entry.get("source_path")
        path_key = entry.get("path_key")
        subject = subjects.get(path_key)
        if subject is None:
            excluded["no_subject_mask"] += 1
            continue
        readable = None
        for candidate in (logical_path, entry.get("read_path")):
            if candidate and os.path.isfile(candidate):
                readable = candidate
                break
        if readable is None:
            excluded["source_unreadable"] += 1
            continue
        scene = scene_by_id.get(source_id) or scene_by_path.get(logical_path)
        if not scene:
            excluded["no_scene"] += 1
            continue
        split = s_split(source_id)
        if split != "train":
            excluded[f"eval_split_{split}"] += 1
            continue
        description = str(subject.get("description") or "").strip()
        if not description:
            excluded["no_description"] += 1
            continue
        mask_area = subject.get("mask_area_cleaned")
        if not isinstance(mask_area, (int, float)) or isinstance(mask_area, bool):
            excluded["no_mask_area"] += 1
            continue
        if not 0.0 < float(mask_area) < 1.0:
            # a degenerate (empty or full-frame) mask is counted, never emitted
            excluded["mask_area_not_positive"] += 1
            continue
        raw_scenes[scene] += 1
        canonical = canonical_scene(scene)
        canonical_scenes[canonical] += 1
        rows.append({
            "source_id": source_id,
            "source_path": readable,
            "subject_path": str(subject_cache / path_key / "subject.png"),
            "scene": canonical,
            "subject": {"description": description, "mask_area": mask_area},
        })
    return rows, excluded, raw_scenes, canonical_scenes


def build_candidates_nfs(
    entries: list[dict[str, Any]],
    scene_by_id: dict[str, str],
    scene_by_path: dict[str, str],
    reader: ReadOnlyArchiveReader,
) -> tuple[list[dict[str, Any]], Counter, Counter, Counter]:
    """A3a-v2 pool: published ``cache/subject`` vrmeta rows.

    Rows keep the archive-side logical paths; ``materialize`` rewrites them to
    the local copies it lands.
    """
    excluded: Counter = Counter()
    raw_scenes: Counter = Counter()
    canonical_scenes: Counter = Counter()
    rows: list[dict[str, Any]] = []
    for entry in entries:
        members = entry.get("members") or {}
        mask_path = members.get(".subject.png")
        if not mask_path:
            excluded["no_subject_mask"] += 1
            continue
        source_id = entry.get("asset_id")
        logical_path = entry.get("source_path")
        if not source_id or not logical_path:
            # cache entries the subject stage rejected carry no asset identity
            excluded["no_asset_id"] += 1
            continue
        if not reader.resolvable(logical_path):
            excluded["source_unreadable"] += 1
            continue
        scene = scene_by_id.get(source_id) or scene_by_path.get(logical_path)
        if not scene:
            excluded["no_scene"] += 1
            continue
        split = s_split(source_id)
        if split != "train":
            excluded[f"eval_split_{split}"] += 1
            continue
        description = str((entry.get("subject") or {}).get("description") or "").strip()
        if not description:
            excluded["no_description"] += 1
            continue
        mask_area = entry.get("mask_area")
        if not isinstance(mask_area, (int, float)) or isinstance(mask_area, bool):
            excluded["no_mask_area"] += 1
            continue
        # subject.json's ``mask_area_cleaned`` is this value rounded to 4
        # decimals; rounding here keeps the smoke5 field convention without a
        # second archive read per row.
        mask_area = round(float(mask_area), MASK_AREA_DECIMALS)
        if not 0.0 < mask_area < 1.0:
            excluded["mask_area_not_positive"] += 1
            continue
        raw_scenes[scene] += 1
        canonical = canonical_scene(scene)
        canonical_scenes[canonical] += 1
        rows.append({
            "source_id": source_id,
            "source_path": logical_path,
            "subject_path": mask_path,
            "scene": canonical,
            "subject": {"description": description, "mask_area": mask_area},
        })
    return rows, excluded, raw_scenes, canonical_scenes


def order_key(row: dict[str, Any]) -> tuple[str, str]:
    source_id = row["source_id"]
    return hashlib.sha1(source_id.encode("utf-8")).hexdigest(), source_id


def render_manifest(rows: list[dict[str, Any]]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in rows
    )


# --------------------------------------------------------------------------- #
# materialisation
# --------------------------------------------------------------------------- #
def image_suffix(payload: bytes, fallback: str) -> str:
    """Suffix implied by the bytes themselves (header parse only, no decode)."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            fmt = image.format or ""
    except Exception:
        return fallback or ".bin"
    return SUFFIX_BY_FORMAT.get(fmt, fallback or ".bin")


def load_index(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    index: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entry = json.loads(line)
                index[entry["source_id"]] = entry
    return index


def _reuse(target: Path, recorded: dict[str, Any] | None, kind: str) -> bool:
    """True when an already-landed file matches its recorded size and digest."""
    if not recorded:
        return False
    path = recorded.get(f"{kind}_path")
    size = recorded.get(f"{kind}_bytes")
    digest = recorded.get(f"{kind}_sha256")
    if path != str(target) or not target.is_file() or target.stat().st_size != size:
        return False
    return sha256_file(target) == digest


def materialize(
    rows: list[dict[str, Any]],
    reader: ReadOnlyArchiveReader,
    root: Path,
    workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Land every selected row's source and mask locally; rewrite its paths."""
    source_dir = root / "source"
    mask_dir = root / "mask"
    source_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    index_path = root / "index.jsonl"
    previous = load_index(index_path)

    written = Counter()
    entries: dict[str, dict[str, Any]] = {}
    failures: list[str] = []

    def land(row: dict[str, Any]) -> None:
        source_id = row["source_id"]
        recorded = previous.get(source_id)
        entry: dict[str, Any] = {
            "source_id": source_id,
            "source_origin": row["source_path"],
            "subject_origin": row["subject_path"],
        }
        try:
            suffix = (recorded or {}).get("source_suffix")
            target = source_dir / f"{source_id}{suffix}" if suffix else None
            if target is not None and _reuse(target, recorded, "source"):
                entry.update({k: recorded[k] for k in (
                    "source_suffix", "source_path", "source_bytes", "source_sha256")})
                written["source_reused"] += 1
            else:
                payload = reader.read_bytes(row["source_path"])
                suffix = image_suffix(payload, Path(row["source_path"]).suffix.lower())
                target = source_dir / f"{source_id}{suffix}"
                target.write_bytes(payload)
                entry.update({
                    "source_suffix": suffix,
                    "source_path": str(target),
                    "source_bytes": len(payload),
                    "source_sha256": hashlib.sha256(payload).hexdigest(),
                })
                written["source_written"] += 1
                written["source_written_bytes"] += len(payload)

            mask_target = mask_dir / f"{source_id}.png"
            if _reuse(mask_target, recorded, "subject"):
                entry.update({k: recorded[k] for k in (
                    "subject_path", "subject_bytes", "subject_sha256")})
                written["mask_reused"] += 1
            else:
                payload = reader.read(row["subject_path"])
                mask_target.write_bytes(payload)
                entry.update({
                    "subject_path": str(mask_target),
                    "subject_bytes": len(payload),
                    "subject_sha256": hashlib.sha256(payload).hexdigest(),
                })
                written["mask_written"] += 1
                written["mask_written_bytes"] += len(payload)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            failures.append(f"{source_id}: {type(exc).__name__}: {exc}")
            return
        entries[source_id] = entry

    # Read in archive-physical order so a round of otherwise random NFS reads
    # walks each shard forward; output order is restored from ``rows`` below.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        ordered = sorted(
            rows, key=lambda row: reader.physical_key(row["source_path"])
        )
        list(pool.map(land, ordered))

    if failures:
        raise SystemExit(
            "materialisation failed for "
            f"{len(failures)} selected rows (no silent substitution):\n  "
            + "\n  ".join(failures[:20])
        )

    index_path.write_text(
        "".join(
            json.dumps(entries[sid], ensure_ascii=False, separators=(",", ":"), sort_keys=True)
            + "\n"
            for sid in sorted(entries)
        ),
        encoding="utf-8",
    )

    landed: list[dict[str, Any]] = []
    for row in rows:
        entry = entries[row["source_id"]]
        landed.append({
            "source_id": row["source_id"],
            "source_path": entry["source_path"],
            "subject_path": entry["subject_path"],
            "scene": row["scene"],
            "subject": row["subject"],
        })

    stats = {
        "materialize_dir": str(root),
        "materialize_index": str(index_path),
        "source_written": written["source_written"],
        "source_reused": written["source_reused"],
        "source_written_bytes": written["source_written_bytes"],
        "mask_written": written["mask_written"],
        "mask_reused": written["mask_reused"],
        "mask_written_bytes": written["mask_written_bytes"],
        "source_total_bytes": sum(e["source_bytes"] for e in entries.values()),
        "mask_total_bytes": sum(e["subject_bytes"] for e in entries.values()),
    }
    return landed, stats


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-source", choices=("nfs", "backfill"), default="nfs")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL)
    parser.add_argument("--subject-cache", type=Path, default=DEFAULT_SUBJECT_CACHE)
    parser.add_argument("--nfs-metadata", type=Path, default=DEFAULT_NFS_METADATA)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--nfs-read-root", default=DEFAULT_NFS_RO_ROOT)
    parser.add_argument("--materialize-dir", type=Path, default=DEFAULT_MATERIALIZE)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--split-table", type=Path, default=DEFAULT_SPLIT_TABLE)
    parser.add_argument("--pg-dsn", default=DEFAULT_PG_DSN)
    parser.add_argument("--target", type=int, default=5000)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--stats", type=Path, default=None)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="compute and print the stats block without writing or landing anything",
    )
    args = parser.parse_args(argv)

    scene_by_id, scene_by_path = read_scenes(args.pg_dsn)
    stats: dict[str, Any] = {
        "pool_source": args.pool_source,
        "split_rule": SPLIT_RULE,
        "split_table": str(args.split_table),
        "order_rule": ORDER_RULE,
        "exclusion_order": list(EXCLUSION_ORDER),
    }
    reader: ReadOnlyArchiveReader | None = None

    if args.pool_source == "nfs":
        entries = read_vrmeta(args.nfs_metadata)
        reader = ReadOnlyArchiveReader(args.catalog, args.nfs_read_root)
        rows, excluded, raw_scenes, canonical_scenes = build_candidates_nfs(
            entries, scene_by_id, scene_by_path, reader
        )
        pool_ids = [e["asset_id"] for e in entries if e.get("asset_id")]
        stats.update({
            "pool_path": str(args.nfs_metadata),
            "pool_sha256": sha256_file(args.nfs_metadata),
            "pool_vrmeta_rows": len(entries),
            "pool_with_subject_png": sum(
                1 for e in entries if (e.get("members") or {}).get(".subject.png")
            ),
            "catalog_path": str(args.catalog),
            "nfs_read_root": args.nfs_read_root,
        })
    else:
        pool = read_pool(args.pool)
        subjects = read_subject_cache(args.subject_cache)
        rows, excluded, raw_scenes, canonical_scenes = build_candidates(
            pool, subjects, scene_by_id, scene_by_path, args.subject_cache
        )
        pool_ids = [entry["asset_id"] for entry in pool]
        stats.update({
            "pool_path": str(args.pool),
            "pool_rows": len(pool),
            "pool_sha256": sha256_file(args.pool),
            "subject_cache": str(args.subject_cache),
            "subject_cache_ready_with_png": len(subjects),
        })

    split_table = load_split_table(args.split_table)
    checked = assert_split_rule(pool_ids, split_table)

    rows.sort(key=order_key)
    selected = rows[: args.target] if args.target > 0 else rows
    selected_scenes = Counter(row["scene"] for row in selected)

    stats.update({
        "split_table_ids_cross_checked": checked,
        "excluded": {reason: excluded.get(reason, 0) for reason in EXCLUSION_ORDER},
        "excluded_total": sum(excluded.values()),
        "eligible": len(rows),
        "requested_n": args.target,
        "produced_n": len(selected),
        "shortfall": max(args.target - len(selected), 0),
        "scene_distribution_qa_raw_eligible": dict(sorted(raw_scenes.items())),
        "scene_distribution_canonical_eligible": dict(sorted(canonical_scenes.items())),
        "scene_distribution_canonical_selected": dict(sorted(selected_scenes.items())),
        "manifest_path": str(args.out),
    })

    if args.dry_run:
        stats["manifest_sha256"] = hashlib.sha256(
            render_manifest(selected).encode("utf-8")
        ).hexdigest()
        stats["manifest_sha256_note"] = "pre-materialisation paths (dry run)"
        print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if reader is not None:
        selected, materialize_stats = materialize(
            selected, reader, args.materialize_dir, args.workers
        )
        stats.update(materialize_stats)

    payload = render_manifest(selected)
    stats["manifest_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(payload, encoding="utf-8")
    stats_path = args.stats or args.out.with_suffix(".stats.json")
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
