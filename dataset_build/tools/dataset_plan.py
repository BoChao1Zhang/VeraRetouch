"""Plan the archived layout of a dataset: clean metadata in, plan JSONL out.

A plan is the only door through which data reaches the archive.  Planning
resolves three things the packer deliberately does not know about:

* the logical layout — class/source directories for images, format and style
  taxonomy for presets — which is independent of where files physically live;
* the WebDataset key and extension of every member, so one sample's files share
  a key and land next to each other;
* the cleaned metadata, which is materialised as an in-sample ``.json`` member
  and mirrored to ``metadata.jsonl`` for the global catalog.

Metadata that needs only ``stat`` and existing snapshots is resolved here.
Anything that needs the bytes (checksums, decoded dimensions) is recorded by the
pack pass, which already reads every byte exactly once.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from dataset_build.tools.indexed_tar import (
    IndexedTarError,
    MAX_MEMBER_BYTES,
    _stable_member,
)


PLAN_SCHEMA_VERSION = 1
MAX_KEY_CHARS = 96
MAX_EXT_SEGMENT = 32
_UNSAFE_KEY_RE = re.compile(r"[^A-Za-z0-9_-]+")
_LEADING_RE = re.compile(r"\A[^A-Za-z0-9]+")
UNKNOWN = "unknown"
UNCLASSIFIED = "_unclassified"
# Namespaced so a source tree's own ".json" file can never collide with the
# metadata member the planner materialises (ArtEdit-Bench ships one).
META_SUFFIX = ".vrmeta.json"
_reported_fallbacks: set[str] = set()


class PlanError(RuntimeError):
    """Raised when a plan cannot be produced from the inputs at hand."""


def sanitize_key(text: str, *, salt: str = "") -> str:
    """Turn arbitrary text into a legal, stable WebDataset key.

    Keys must be ASCII ``[A-Za-z0-9_-]`` with no dot, because the first dot in a
    member name starts the extension.  Anything lost to sanitising or truncation
    is recovered by appending a digest of the original text, so distinct inputs
    stay distinct and the same input always yields the same key.
    """
    cleaned = _LEADING_RE.sub("", _UNSAFE_KEY_RE.sub("_", text)).strip("_-")
    digest = hashlib.sha256((salt + text).encode("utf-8")).hexdigest()[:8]
    if not cleaned:
        return f"x{digest}"
    if cleaned != text or len(cleaned) > MAX_KEY_CHARS - 9:
        return f"{cleaned[: MAX_KEY_CHARS - 9]}_{digest}"
    return cleaned


def normalise_extension(name: str) -> str:
    """Lowercase a file suffix chain and make every segment archive-legal."""
    segments = [segment for segment in name.split(".") if segment]
    if len(segments) < 2:
        raise PlanError(f"a member needs an extension: {name}")
    parts = []
    for segment in segments[1:]:
        cleaned = _UNSAFE_KEY_RE.sub("_", segment).lower().strip("_-")
        if not cleaned:
            raise PlanError(f"unusable extension segment in {name}")
        if len(cleaned) > MAX_EXT_SEGMENT:
            # Two long names sharing a prefix must not truncate onto the same
            # member (sam3 concept masks like "building_silhouette.png" do).
            digest = hashlib.sha256(segment.encode("utf-8")).hexdigest()[:4]
            cleaned = f"{cleaned[: MAX_EXT_SEGMENT - 5]}_{digest}"
        parts.append(cleaned)
    return "." + ".".join(parts)


@dataclass
class PlanRow:
    """One archived member: where it comes from and where it lands."""

    path: Path
    logical_path: str
    meta: dict[str, object] = field(default_factory=dict)

    @property
    def member(self) -> str:
        return _stable_member(self.logical_path)[2]

    @property
    def sample_id(self) -> str:
        return _stable_member(self.logical_path)[0]


@dataclass
class PlanGroup:
    """A set of samples that share one archive directory (one shard series)."""

    group: str
    rows: list[PlanRow] = field(default_factory=list)

    def add(self, row: PlanRow) -> None:
        self.rows.append(row)


def _materialise_meta(staging: Path, sample_id: str, payload: Mapping[str, object]) -> Path:
    """Write one sample's metadata to SSD staging so it can be archived as a member."""
    path = staging / f"{sample_id}.json"
    text = json.dumps(payload, ensure_ascii=False, indent=None, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _interleave_meta(rows: list[PlanRow], meta_rows: list[PlanRow]) -> list[PlanRow]:
    """把每个 sample 的 ``.vrmeta.json`` 排在该 sample 最后一个成员之后。

    保序模式下不能像排序模式那样把 meta 行统一追加到末尾：那会让 A 的 meta 行
    落在 B 的成员之后，打断"同一 sample 的成员必须连续"这个打包不变量。
    没有任何数据成员的 meta 行（纯元数据 sample）按 sample_id 追加在最后。
    """
    pending = {row.sample_id: row for row in meta_rows}
    ordered: list[PlanRow] = []
    for index, row in enumerate(rows):
        ordered.append(row)
        following = rows[index + 1].sample_id if index + 1 < len(rows) else None
        if row.sample_id != following and row.sample_id in pending:
            ordered.append(pending.pop(row.sample_id))
    ordered.extend(pending[sample_id] for sample_id in sorted(pending))
    return ordered


def _assert_contiguous(rows: list[PlanRow]) -> None:
    """调用方给定的行序必须让同一 sample 的成员连续（打包器的硬约束）。

    打包器本来就会拒绝，但那要等到读 plan 文件时才报错；在这里先报，错误信息里
    还带得上产生这一行的物理路径。
    """
    previous: str | None = None
    closed: set[str] = set()
    for row in rows:
        if row.sample_id == previous:
            continue
        if row.sample_id in closed:
            raise PlanError(f"sample {row.sample_id} is not contiguous at {row.path}")
        if previous is not None:
            closed.add(previous)
        previous = row.sample_id


def write_group(
    group: PlanGroup,
    out_dir: Path,
    *,
    meta_staging: Path,
    sample_meta: Mapping[str, Mapping[str, object]] | None = None,
    preserve_order: bool = False,
) -> dict[str, object]:
    """Write ``plan.jsonl`` and ``metadata.jsonl`` for one group.

    Rows are sorted by member name by default, which is one order the packer
    accepts and which groups a sample.  ``preserve_order=True`` keeps the caller's
    row order instead, so the archived member order equals the production order
    (``indexed_tar.MEMBER_ORDER_PLAN``); the caller then owns the weaker invariant
    the packer actually enforces — members unique and one sample's members
    contiguous — and gets a ``PlanError`` here when it breaks it.  Duplicate
    members are a planning bug rather than something to silently resolve, so they
    raise either way.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_staging.mkdir(parents=True, exist_ok=True)

    rows = list(group.rows)
    meta_rows: list[PlanRow] = []
    if sample_meta:
        for sample_id, payload in sorted(sample_meta.items()):
            enriched = {"sample_id": sample_id, "group": group.group, **payload}
            path = _materialise_meta(meta_staging, sample_id, enriched)
            meta_rows.append(
                PlanRow(
                    path=path,
                    logical_path=f"{group.group}/{sample_id}{META_SUFFIX}",
                    meta=enriched,
                )
            )
    rows = _interleave_meta(rows, meta_rows) if preserve_order else rows + meta_rows

    seen: dict[str, str] = {}
    for row in rows:
        member = row.member
        if len(member.encode("utf-8")) > MAX_MEMBER_BYTES:
            raise PlanError(f"member too long for USTAR: {row.logical_path}")
        previous = seen.get(member)
        if previous is not None:
            raise PlanError(f"member collision {member}: {previous} and {row.path}")
        seen[member] = str(row.path)
    if preserve_order:
        _assert_contiguous(rows)
    else:
        rows.sort(key=lambda item: item.member)

    plan_path = out_dir / "plan.jsonl"
    metadata_path = out_dir / "metadata.jsonl"
    samples: set[str] = set()
    with plan_path.open("w", encoding="utf-8") as plan, metadata_path.open(
        "w", encoding="utf-8"
    ) as metadata:
        for row in rows:
            samples.add(row.sample_id)
            plan.write(
                json.dumps(
                    {"path": str(row.path), "logical_path": row.logical_path}, sort_keys=True
                )
                + "\n"
            )
            metadata.write(
                json.dumps(
                    {
                        "schema_version": PLAN_SCHEMA_VERSION,
                        "group": group.group,
                        "sample_id": row.sample_id,
                        "logical_path": row.logical_path,
                        "member": row.member,
                        "source_path": str(row.path),
                        "bytes_size": row.path.stat().st_size,
                        **row.meta,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "group": group.group,
        "plan": str(plan_path),
        "metadata": str(metadata_path),
        "members": len(rows),
        "samples": len(samples),
    }


def load_asset_snapshot(csv_gz: Path) -> dict[str, dict[str, object]]:
    """Index the retired PostgreSQL asset dump by source path.

    The dump is the only surviving inventory of scene/style/dimensions, so it is
    the cleaning input for images.  Paths absent from it are not rejected; they
    are archived with ``scene=unknown`` and can be relabelled later by repacking
    just the affected group.

    Values are kept verbatim as CSV text, so numeric fields such as width and
    aesthetic reach the archive as strings; consumers coerce them.  This is
    deliberate: it keeps the archived metadata identical to the snapshot it came
    from, and it stays consistent with the groups already published.
    """
    wanted = (
        "asset_id", "corpus", "scene", "style", "width", "height", "bytes_size",
        "is_portrait_pool", "aesthetic", "pixel_sha256", "phash", "dup_of", "kind", "fmt",
    )
    index: dict[str, dict[str, object]] = {}
    csv.field_size_limit(1 << 24)
    with gzip.open(csv_gz, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            path = (row.get("path") or "").strip()
            if not path:
                continue
            index[path] = {key: row[key] for key in wanted if row.get(key) not in (None, "")}
    if not index:
        raise PlanError(f"asset snapshot is empty: {csv_gz}")
    return index


def _read_jsonl(path: Path) -> Iterator[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PlanError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise PlanError(f"JSON object required at {path}:{line_number}")
            yield value


def plan_presets(
    bank_dir: Path,
    out_root: Path,
    *,
    meta_staging: Path,
    baked_dir: Path | None = None,
) -> list[dict[str, object]]:
    """Plan ``preset/<fmt>/<major>/<minor>`` groups from the preset bank.

    ``features.jsonl`` is the inventory (one row per preset, with its format,
    source pack and physical path) and ``taxonomy.jsonl`` supplies the two-level
    style class.  Presets the taxonomy filtered out as weak keep their format but
    land under ``_unclassified``; a baked LUT, when one exists, joins its preset
    as a second member of the same sample rather than a separate sample.
    """
    features_path = bank_dir / "features.jsonl"
    taxonomy_path = bank_dir / "taxonomy.jsonl"
    taxonomy: dict[str, tuple[str, str]] = {}
    for row in _read_jsonl(taxonomy_path):
        preset_id = str(row.get("preset_id") or "")
        major, minor = str(row.get("major") or ""), str(row.get("minor") or "")
        if preset_id and major and minor:
            taxonomy[preset_id] = (major, minor)

    groups: dict[str, PlanGroup] = {}
    sample_meta: dict[str, dict[str, dict[str, object]]] = {}
    missing_files = 0
    for row in _read_jsonl(features_path):
        preset_id = str(row.get("preset_id") or "")
        raw_path = str(row.get("path") or "")
        fmt = str(row.get("fmt") or "")
        if not preset_id or not raw_path or not fmt:
            raise PlanError(f"features row lacks preset_id/path/fmt: {row}")
        path = Path(raw_path)
        if not path.is_file():
            missing_files += 1
            continue
        major, minor = taxonomy.get(preset_id, (UNCLASSIFIED, ""))
        group = f"preset/{fmt}/{major}/{minor}" if minor else f"preset/{fmt}/{major}"
        key = sanitize_key(preset_id)
        bucket = groups.setdefault(group, PlanGroup(group=group))
        bucket.add(
            PlanRow(
                path=path,
                logical_path=f"{group}/{key}{normalise_extension(path.name)}",
                meta={"preset_id": preset_id, "fmt": fmt, "role": "preset"},
            )
        )
        if baked_dir is not None:
            baked = baked_dir / f"{preset_id}.cube"
            if baked.is_file():
                bucket.add(
                    PlanRow(
                        path=baked,
                        logical_path=f"{group}/{key}.baked.cube",
                        meta={"preset_id": preset_id, "fmt": "cube", "role": "baked_lut"},
                    )
                )
        sample_meta.setdefault(group, {})[key] = {
            "preset_id": preset_id,
            "fmt": fmt,
            "kind": row.get("kind"),
            "pack_id": row.get("pack_id"),
            "major": major,
            "minor": minor or None,
            "axes": row.get("axes"),
            "coherence": row.get("coherence"),
            "metrics": row.get("metrics"),
            "scene_affinity": row.get("scene_affinity"),
            "has_local_mask": row.get("has_local_mask"),
            "has_ai_mask": row.get("has_ai_mask"),
            "preset_content_hash": row.get("preset_content_hash"),
            "source_path": raw_path,
        }

    if not groups:
        raise PlanError(f"no preset files found under {features_path}")
    summary = [
        write_group(
            group,
            out_root / group.group,
            meta_staging=meta_staging / group.group,
            sample_meta=sample_meta.get(group.group),
        )
        for group in sorted(groups.values(), key=lambda item: item.group)
    ]
    if missing_files:
        print(
            f"[plan] warning: {missing_files} preset rows point at missing files",
            file=sys.stderr,
        )
    return summary


DATASETS_ROOT = Path("/home/bc/data/datasets")
# Archives are redundant with their extracted trees and are handled by the
# separate deduplication step, so planning never pulls them into a shard.
SKIP_SUFFIXES = frozenset({".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"})


@dataclass(frozen=True)
class ImageSource:
    """One physical source of images and how its files map onto samples.

    shape ``files``       - walk the root; every file is its own sample unless it
                            shares a stem with a sibling (an image and its sidecar
                            then become two members of one sample).
    shape ``roles``       - parallel directories holding the same stems; each role
                            contributes one member, e.g. ppr10k source/target_a.
    shape ``sample_dir``  - one directory per sample; ``ext_map`` renames nested
                            paths onto short archive extensions.
    """

    corpus: str
    root: Path
    shape: str = "files"
    roles: Mapping[str, str] = field(default_factory=dict)
    ext_map: Mapping[str, str] = field(default_factory=dict)


IMAGE_SOURCES: tuple[ImageSource, ...] = (
    ImageSource("tad66k", DATASETS_ROOT / "_scratch/TAD66K"),
    ImageSource("unsplash", DATASETS_ROOT / "unsplash-lite/images/original"),
    # A separate corpus, not a duplicate: same filenames, downscaled renditions.
    ImageSource("unsplash_work", DATASETS_ROOT / "_scratch/unsplash"),
    ImageSource("para", DATASETS_ROOT / "PARA/imgs"),
    ImageSource("awards", DATASETS_ROOT / "presets_sources/awards"),
    ImageSource("korean", DATASETS_ROOT / "presets_sources/korean"),
    ImageSource("quandian", DATASETS_ROOT / "presets_sources/quandian"),
    ImageSource("greysky", DATASETS_ROOT / "presets_sources/greysky"),
    ImageSource("artimuse", DATASETS_ROOT / "ArtiMuse/image"),
    ImageSource("artedit_bench", DATASETS_ROOT / "ArtEdit-Bench"),
    ImageSource("fivek_raw", DATASETS_ROOT / "fivek5k/raw"),
    ImageSource("fivek_tiff16", DATASETS_ROOT / "fivek_tiff16_c_cache"),
    ImageSource(
        "ppr10k",
        DATASETS_ROOT / "ppr10k",
        shape="roles",
        roles={
            "source": "source",
            "target_a": "target_a",
            "target_b": "target_b",
            "target_c": "target_c",
            "source_xmp": "xmp/source",
            "target_a_xmp": "xmp/target_a",
            "target_b_xmp": "xmp/target_b",
            "target_c_xmp": "xmp/target_c",
            "mask_360p": "masks/360p",
            "mask_full": "masks/full",
        },
    ),
    ImageSource(
        "raise6k",
        DATASETS_ROOT / "RAISE-6k",
        shape="roles",
        roles={"raw": "raw", "preview": "jpg_preview"},
    ),
    ImageSource(
        "fivek_gold",
        DATASETS_ROOT / "fivek_gold/train_global",
        shape="sample_dir",
        ext_map={
            "before.jpg": ".before.jpg",
            "processed.jpg": ".processed.jpg",
            "meta.json": ".meta.json",
            "param_verify.json": ".param_verify.json",
            "config.lua": ".config.lua",
            "en/user_want_short/user_prompt.txt": ".prompt_short.txt",
            "en/user_want_middle/user_prompt.txt": ".prompt_middle.txt",
            "en/user_want_long/user_prompt.txt": ".prompt_long.txt",
            # 3654 of 20000 samples failed param verification and carry this
            # instead of processed.jpg/param_verify.json.
            "param_verify_error.txt": ".param_error.txt",
        },
    ),
)


def _fallback_extension(relative: str) -> str:
    """Name an unmapped nested member without dropping or renaming its data.

    ponytail: a surprise filename must not kill a multi-hour unattended pack, so
    the flattened relative path becomes the extension and the caller logs it.
    Add a real `ext_map` entry once such a member turns out to matter.
    """
    flattened = relative.replace("/", "_")
    stem, extension = _split_member_name(Path(flattened).name)
    prefix = flattened[: len(flattened) - len(Path(flattened).name)] + stem
    return normalise_extension(f"x.{prefix}{extension}") if prefix else extension


def _walk_files(root: Path) -> Iterator[tuple[Path, str]]:
    """Yield every regular file under root with its POSIX relative path."""
    stack = [(root, "")]
    while stack:
        directory, prefix = stack.pop()
        with os.scandir(directory) as scan:
            for entry in sorted(scan, key=lambda item: item.name):
                relative = f"{prefix}{entry.name}"
                if entry.is_symlink():
                    raise PlanError(f"symlinks are not archivable: {entry.path}")
                if entry.is_dir():
                    stack.append((Path(entry.path), relative + "/"))
                elif entry.is_file():
                    yield Path(entry.path), relative


def _split_member_name(name: str) -> tuple[str, str]:
    """Split a filename into its key stem and archive extension.

    Real source trees hold marker files (``.extract_done``) and extensionless
    notes, and dropping them would quietly lose provenance, so both get a legal
    member name instead: a dotfile becomes suffix-only and a dotless file gets
    ``.bin``.
    """
    if name.startswith("."):
        return "", normalise_extension("marker" + name)
    if "." not in name:
        return name, ".bin"
    # Split at the FIRST dot, exactly like WebDataset: that makes "x.png" and
    # "x.raw_mask.png" two members of sample "x" instead of two samples.
    return name.partition(".")[0], normalise_extension(name)


@dataclass
class _Sample:
    key: str
    members: list[tuple[Path, str, str]] = field(default_factory=list)  # path, ext, role


def _iter_samples(source: ImageSource) -> Iterator[_Sample]:
    if source.shape == "files":
        pending: dict[str, _Sample] = {}
        for path, relative in _walk_files(source.root):
            if path.suffix.lower() in SKIP_SUFFIXES:
                continue
            name = Path(relative).name
            stem, extension = _split_member_name(name)
            base = relative[: len(relative) - len(name)] + (stem or "marker")
            key = sanitize_key(f"{source.corpus}_{base}")
            sample = pending.setdefault(key, _Sample(key=key))
            sample.members.append((path, extension, "primary"))
        yield from (pending[key] for key in sorted(pending))
        return

    if source.shape == "roles":
        by_key: dict[str, _Sample] = {}
        for role, relative_dir in sorted(source.roles.items()):
            directory = source.root / relative_dir
            if not directory.is_dir():
                raise PlanError(f"role directory is missing: {directory}")
            for path, relative in _walk_files(directory):
                stem, suffix = _split_member_name(Path(relative).name)
                key = sanitize_key(f"{source.corpus}_{stem or 'marker'}")
                extension = f".{role}{suffix}"
                by_key.setdefault(key, _Sample(key=key)).members.append((path, extension, role))
        yield from (by_key[key] for key in sorted(by_key))
        return

    if source.shape == "sample_dir":
        with os.scandir(source.root) as scan:
            directories = sorted(
                (entry for entry in scan if entry.is_dir()), key=lambda item: item.name
            )
        for entry in directories:
            key = sanitize_key(f"{source.corpus}_{entry.name}")
            sample = _Sample(key=key)
            for path, relative in _walk_files(Path(entry.path)):
                extension = source.ext_map.get(relative)
                if extension is None:
                    extension = _fallback_extension(relative)
                    if relative not in _reported_fallbacks:
                        _reported_fallbacks.add(relative)
                        print(
                            f"[plan] {source.corpus}: unmapped member {relative!r} archived as "
                            f"{extension!r}",
                            file=sys.stderr,
                        )
                sample.members.append((path, extension, relative))
            yield sample
        return

    raise PlanError(f"unknown source shape: {source.shape}")


def plan_images(
    sources: Iterable[ImageSource],
    out_root: Path,
    *,
    meta_staging: Path,
    snapshot: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Plan ``img/<scene>/<corpus>`` groups for every configured image source.

    The class directory comes from the asset snapshot when it knows the file and
    falls back to ``unknown``, which keeps every image archivable today and keeps
    a later relabelling confined to the two groups involved.
    """
    groups: dict[str, PlanGroup] = {}
    sample_meta: dict[str, dict[str, dict[str, object]]] = {}
    unknown_scene = 0
    total = 0
    for source in sources:
        if not source.root.is_dir():
            raise PlanError(f"source root is missing: {source.root}")
        for sample in _iter_samples(source):
            if not sample.members:
                continue
            snapshot_row: Mapping[str, object] = {}
            for path, _extension, _role in sample.members:
                snapshot_row = snapshot.get(str(path)) or {}
                if snapshot_row:
                    break
            scene = str(snapshot_row.get("scene") or UNKNOWN) or UNKNOWN
            if not snapshot_row:
                unknown_scene += 1
            group_name = f"img/{scene}/{source.corpus}"
            bucket = groups.setdefault(group_name, PlanGroup(group=group_name))
            for path, extension, role in sample.members:
                bucket.add(
                    PlanRow(
                        path=path,
                        logical_path=f"{group_name}/{sample.key}{extension}",
                        meta={"corpus": source.corpus, "role": role},
                    )
                )
            sample_meta.setdefault(group_name, {})[sample.key] = {
                "corpus": source.corpus,
                "scene": scene,
                "source_root": str(source.root),
                "members": {extension: str(path) for path, extension, _ in sample.members},
                **{
                    key: snapshot_row[key]
                    for key in ("asset_id", "style", "width", "height", "aesthetic",
                                "is_portrait_pool", "pixel_sha256", "phash", "dup_of")
                    if key in snapshot_row
                },
            }
            total += 1

    if not groups:
        raise PlanError("no image samples found")
    summary = [
        write_group(
            group,
            out_root / group.group,
            meta_staging=meta_staging / group.group,
            sample_meta=sample_meta.get(group.group),
        )
        for group in sorted(groups.values(), key=lambda item: item.group)
    ]
    print(
        f"[plan] images: {total} samples, {unknown_scene} without snapshot metadata",
        file=sys.stderr,
    )
    return summary


VERA_ROOT = DATASETS_ROOT / "vera_directionA_1M"
DERIVED_SOURCES: tuple[tuple[str, ImageSource], ...] = (
    ("cgt", ImageSource("s1", VERA_ROOT / "cgt/S1")),
    ("cgt", ImageSource("s2", VERA_ROOT / "cgt/S2")),
    ("cgt", ImageSource("s3", VERA_ROOT / "cgt/S3")),
    ("cgt", ImageSource("s4", VERA_ROOT / "cgt/S4")),
    ("cache", ImageSource(
        "subject",
        VERA_ROOT / "subject_cache",
        shape="sample_dir",
        ext_map={"subject.json": ".subject.json", "subject.png": ".subject.png"},
    )),
    ("cache", ImageSource(
        "tag",
        VERA_ROOT / "tag_cache",
        shape="sample_dir",
        ext_map={"tags.json": ".tags.json"},
    )),
    # sam3 masks are named after the concept they segment, so the extension set
    # is open-ended and resolved by the logged fallback instead of a map.
    ("cache", ImageSource("sam3", VERA_ROOT / "sam3_cache", shape="sample_dir")),
)


def plan_derived(
    sources: Iterable[tuple[str, ImageSource]],
    out_root: Path,
    *,
    meta_staging: Path,
) -> list[dict[str, object]]:
    """Plan ``<kind>/<corpus>`` groups for derived assets (C_GT, caches).

    These carry no scene class and no external metadata: the archive records what
    each member is and where it came from, which is all the pipeline needs to
    read them back through the offset helper.
    """
    groups: dict[str, PlanGroup] = {}
    sample_meta: dict[str, dict[str, dict[str, object]]] = {}
    for kind, source in sources:
        if not source.root.is_dir():
            raise PlanError(f"derived source root is missing: {source.root}")
        group_name = f"{kind}/{source.corpus}"
        bucket = groups.setdefault(group_name, PlanGroup(group=group_name))
        for sample in _iter_samples(source):
            for path, extension, role in sample.members:
                bucket.add(
                    PlanRow(
                        path=path,
                        logical_path=f"{group_name}/{sample.key}{extension}",
                        meta={"kind": kind, "corpus": source.corpus, "role": role},
                    )
                )
            sample_meta.setdefault(group_name, {})[sample.key] = {
                "kind": kind,
                "corpus": source.corpus,
                "source_root": str(source.root),
                "members": {extension: str(path) for path, extension, _ in sample.members},
            }
    if not groups:
        raise PlanError("no derived samples found")
    return [
        write_group(
            group,
            out_root / group.group,
            meta_staging=meta_staging / group.group,
            sample_meta=sample_meta.get(group.group),
        )
        for group in sorted(groups.values(), key=lambda item: item.group)
    ]


def _iter_json_strings(value: object) -> Iterator[str]:
    """Yield every string anywhere inside a decoded JSON value."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_json_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _iter_json_strings(item)


def collect_render_references(build_dirs: Iterable[Path], renders_root: Path) -> dict[str, str]:
    """Map each referenced render path to the build that first references it.

    Rather than tracking the record schema of every generation (I_in/I_tar,
    candidate after_path, C_GT, DPO pairs), this walks all strings in each JSONL
    record and keeps the ones that resolve under the renders tree.  Records that
    no build references are not archived; they are reported instead.
    """
    prefix = str(renders_root) + "/"
    referenced: dict[str, str] = {}
    for build_dir in build_dirs:
        build_id = build_dir.name
        for jsonl in sorted(build_dir.rglob("*.jsonl")):
            for row in _read_jsonl(jsonl):
                for text in _iter_json_strings(row):
                    if text.startswith(prefix):
                        referenced.setdefault(text, build_id)
    return referenced


def plan_renders(
    build_dirs: Iterable[Path],
    renders_root: Path,
    out_root: Path,
    *,
    meta_staging: Path,
) -> list[dict[str, object]]:
    """Plan ``renders/<build_id>`` groups holding only referenced renders."""
    referenced = collect_render_references(build_dirs, renders_root)
    missing = 0
    groups: dict[str, PlanGroup] = {}
    for raw_path, build_id in sorted(referenced.items()):
        path = Path(raw_path)
        if not path.is_file():
            missing += 1
            continue
        group_name = f"renders/{build_id}"
        stem, extension = _split_member_name(path.name)
        key = sanitize_key(stem or "render")
        groups.setdefault(group_name, PlanGroup(group=group_name)).add(
            PlanRow(
                path=path,
                logical_path=f"{group_name}/{key}{extension}",
                meta={"build_id": build_id, "role": "render"},
            )
        )
    if not groups:
        raise PlanError(f"no referenced renders found under {renders_root}")
    summary = [
        write_group(group, out_root / group.group, meta_staging=meta_staging / group.group)
        for group in sorted(groups.values(), key=lambda item: item.group)
    ]
    print(
        f"[plan] renders: {len(referenced)} referenced, {missing} referenced but absent",
        file=sys.stderr,
    )
    return summary


def report_unreferenced_renders(
    build_dirs: Iterable[Path], renders_root: Path, report_path: Path
) -> dict[str, object]:
    """List every render no build references, so the operator can decide its fate."""
    referenced = set(collect_render_references(build_dirs, renders_root))
    total = 0
    unreferenced_bytes = 0
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        for path, _relative in _walk_files(renders_root):
            total += 1
            if str(path) not in referenced:
                unreferenced_bytes += path.stat().st_size
                handle.write(f"{path}\n")
    return {
        "files": total,
        "referenced": len(referenced),
        "unreferenced": total - len(referenced),
        "unreferenced_bytes": unreferenced_bytes,
        "report": str(report_path),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    presets = subcommands.add_parser("presets", help="plan preset groups from the preset bank")
    presets.add_argument("--bank", type=Path, required=True)
    presets.add_argument("--baked", type=Path, default=None)
    presets.add_argument("--out", type=Path, required=True)
    presets.add_argument("--meta-staging", type=Path, required=True)
    images = subcommands.add_parser("images", help="plan img/<scene>/<corpus> groups")
    images.add_argument("--snapshot", type=Path, required=True, help="assets.csv.gz")
    images.add_argument("--out", type=Path, required=True)
    images.add_argument("--meta-staging", type=Path, required=True)
    images.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="restrict planning to these corpora (repeatable)",
    )
    derived = subcommands.add_parser("derived", help="plan cgt/<stage> and cache/<kind> groups")
    derived.add_argument("--out", type=Path, required=True)
    derived.add_argument("--meta-staging", type=Path, required=True)
    renders = subcommands.add_parser("renders", help="plan renders/<build_id> groups")
    renders.add_argument("--build", type=Path, action="append", required=True)
    renders.add_argument("--renders-root", type=Path, default=VERA_ROOT / "renders")
    renders.add_argument("--out", type=Path, required=True)
    renders.add_argument("--meta-staging", type=Path, required=True)
    report = subcommands.add_parser(
        "renders-report", help="list renders that no build references"
    )
    report.add_argument("--build", type=Path, action="append", required=True)
    report.add_argument("--renders-root", type=Path, default=VERA_ROOT / "renders")
    report.add_argument("--report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "presets":
            summary = plan_presets(
                args.bank,
                args.out,
                meta_staging=args.meta_staging,
                baked_dir=args.baked,
            )
        elif args.command == "images":
            selected = [
                source
                for source in IMAGE_SOURCES
                if args.corpus is None or source.corpus in set(args.corpus)
            ]
            if not selected:
                raise PlanError(f"no configured corpus matches {args.corpus}")
            summary = plan_images(
                selected,
                args.out,
                meta_staging=args.meta_staging,
                snapshot=load_asset_snapshot(args.snapshot),
            )
        elif args.command == "derived":
            summary = plan_derived(
                DERIVED_SOURCES, args.out, meta_staging=args.meta_staging
            )
        elif args.command == "renders":
            summary = plan_renders(
                args.build, args.renders_root, args.out, meta_staging=args.meta_staging
            )
        elif args.command == "renders-report":
            print(json.dumps(
                report_unreferenced_renders(args.build, args.renders_root, args.report),
                sort_keys=True,
            ))
            return 0
        else:  # pragma: no cover - argparse enforces the choices
            raise PlanError(f"unknown command: {args.command}")
    except (PlanError, IndexedTarError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "groups": len(summary),
        "members": sum(int(item["members"]) for item in summary),
        "samples": sum(int(item["samples"]) for item in summary),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
