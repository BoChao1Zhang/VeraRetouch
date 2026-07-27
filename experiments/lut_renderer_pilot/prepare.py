from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import json
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .common import (
    atomic_write_json,
    atomic_write_jsonl,
    git_state,
    load_config,
    manifests_dir,
    read_jsonl,
    sha256_file,
    stable_digest,
)


def _shape_from_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[int, ...]:
    with archive.open(info, "r") as handle:
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported NPY version {version} for {info.filename}")
    if fortran_order or dtype != np.dtype("float32"):
        raise ValueError(
            f"packed LUT must be C-order float32: {info.filename}, {dtype}, "
            f"fortran={fortran_order}"
        )
    return tuple(int(value) for value in shape)


def inspect_npz(path: Path) -> tuple[dict[str, int], str]:
    shape_by_size: dict[int, tuple[int, ...]] = {}
    result: dict[str, int] = {}
    index_rows: list[tuple[str, int, int, int]] = []
    with zipfile.ZipFile(path, "r") as archive:
        infos = [info for info in archive.infolist() if info.filename.endswith(".npy")]
        for info in infos:
            shape = shape_by_size.get(info.file_size)
            if shape is None:
                shape = _shape_from_member(archive, info)
                shape_by_size[info.file_size] = shape
            if len(shape) != 4 or shape[-1] != 3 or len(set(shape[:3])) != 1:
                raise ValueError(f"non-cubic packed LUT {info.filename}: {shape}")
            key = Path(info.filename).stem
            if key in result:
                raise ValueError(f"duplicate packed LUT key: {key}")
            result[key] = shape[0]
            index_rows.append((info.filename, info.file_size, info.CRC, shape[0]))
    index_hash = stable_digest(*sorted(index_rows))
    return result, index_hash


def _valid_source(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, ValueError):
        return False


def _source_dimensions(row: dict[str, Any]) -> tuple[int, int] | None:
    width, height = row.get("width"), row.get("height")
    if width and height and int(width) > 0 and int(height) > 0:
        return int(width), int(height)
    try:
        with Image.open(row["path"]) as image:
            return image.size
    except (OSError, ValueError):
        return None


def _freeze_luts(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = config["paths"]
    data = config["data"]
    registry_path = Path(paths["lut_registry"])
    npz_path = Path(paths["lut_npz"])
    meta_path = Path(paths["lut_meta"])
    for required in (registry_path, npz_path, meta_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    packed_sizes, packed_index_hash = inspect_npz(npz_path)
    with meta_path.open("r", encoding="utf-8") as handle:
        packed_meta = json.load(handle)

    cube_rows = [
        row for row in read_jsonl(registry_path) if str(row.get("fmt", "")).lower() == "cube"
    ]
    expected_count = int(data["expected_cube_count"])
    if len(cube_rows) != expected_count:
        raise RuntimeError(f"expected {expected_count} cube LUTs, found {len(cube_rows)}")
    hashes = [str(row["content_hash"]) for row in cube_rows]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError("native cube content_hash values are not unique")

    missing_paths: list[str] = []
    missing_packed: list[str] = []
    frozen: list[dict[str, Any]] = []
    for row in sorted(cube_rows, key=lambda item: (item["content_hash"], item["preset_id"])):
        preset_id = str(row["preset_id"])
        lut_path = Path(row["path"])
        if not lut_path.is_file():
            missing_paths.append(str(lut_path))
        if preset_id not in packed_sizes or preset_id not in packed_meta:
            missing_packed.append(preset_id)
            continue
        meta = packed_meta[preset_id]
        frozen.append(
            {
                "style_index": len(frozen),
                "preset_id": preset_id,
                "lut_content_hash": str(row["content_hash"]),
                "path": str(lut_path),
                "fmt": "cube",
                "grid_size": int(packed_sizes[preset_id]),
                "domain_min": [float(value) for value in meta.get("dmin", [0, 0, 0])],
                "domain_max": [float(value) for value in meta.get("dmax", [1, 1, 1])],
                "taxonomy_major": str(row.get("major") or "unknown"),
                "taxonomy_minor": str(row.get("minor") or "unknown"),
                "lut_layout_version": str(row.get("lut_layout_version") or "unknown"),
            }
        )
    if missing_paths or missing_packed:
        raise RuntimeError(
            f"LUT audit failed: missing_paths={len(missing_paths)}, "
            f"missing_packed={len(missing_packed)}"
        )

    actual_sizes = collections.Counter(row["grid_size"] for row in frozen)
    expected_sizes = {int(key): int(value) for key, value in data["expected_grid_sizes"].items()}
    if dict(sorted(actual_sizes.items())) != dict(sorted(expected_sizes.items())):
        raise RuntimeError(
            f"grid distribution mismatch: expected={expected_sizes}, actual={dict(actual_sizes)}"
        )
    audit = {
        "count": len(frozen),
        "unique_content_hashes": len(set(hashes)),
        "grid_size_distribution": dict(sorted(actual_sizes.items())),
        "packed_member_count": len(packed_sizes),
        "packed_index_sha256": packed_index_hash,
        "registry_sha256": sha256_file(registry_path),
        "meta_sha256": sha256_file(meta_path),
        "npz_size_bytes": npz_path.stat().st_size,
    }
    return frozen, audit


def _freeze_sources(config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_index = Path(config["paths"]["source_index"])
    data = config["data"]
    target_count = int(data["test_source_count"])
    salt = str(data["source_split_salt"])
    qa_status = str(data["source_qa_status"])

    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in read_jsonl(source_index):
        source_id = str(row.get("source_id") or "")
        path = Path(str(row.get("path") or ""))
        if (
            not source_id
            or source_id in seen_ids
            or row.get("qa_status") != qa_status
            or row.get("dup_of")
            or not path.is_file()
            or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        ):
            continue
        seen_ids.add(source_id)
        cluster = str(row.get("dup_cluster") or source_id)
        candidates.append(
            {
                "source_id": source_id,
                "source_cluster": cluster,
                "path": str(path),
                "corpus": str(row.get("corpus") or "unknown"),
                "width": row.get("width"),
                "height": row.get("height"),
            }
        )
    dimension_workers = int(data.get("dimension_workers", 16))
    with ThreadPoolExecutor(max_workers=dimension_workers) as executor:
        dimensions = list(executor.map(_source_dimensions, candidates))
    unreadable_sources: list[str] = []
    dimensioned: list[dict[str, Any]] = []
    for row, size in zip(candidates, dimensions, strict=True):
        if size is None:
            unreadable_sources.append(row["source_id"])
            continue
        row["width"], row["height"] = size
        dimensioned.append(row)
    candidates = dimensioned
    if len(candidates) <= target_count:
        raise RuntimeError(f"not enough source candidates: {len(candidates)}")

    grouped: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in candidates:
        grouped[row["source_cluster"]].append(row)
    ordered_groups = sorted(grouped, key=lambda group: stable_digest(salt, group))

    selected: list[dict[str, Any]] = []
    rejected_unreadable: list[str] = []
    selected_groups: set[str] = set()
    for group in ordered_groups:
        rows = grouped[group]
        if len(selected) + len(rows) > target_count:
            continue
        if not all(_valid_source(Path(row["path"])) for row in rows):
            rejected_unreadable.extend(row["source_id"] for row in rows)
            continue
        selected.extend(rows)
        selected_groups.add(group)
        if len(selected) == target_count:
            break
    if len(selected) != target_count:
        raise RuntimeError(
            f"could not form exact grouped test split: wanted={target_count}, got={len(selected)}"
        )

    test = sorted(selected, key=lambda row: stable_digest(salt, row["source_id"]))
    train = [row for row in candidates if row["source_cluster"] not in selected_groups]
    train_clusters = {row["source_cluster"] for row in train}
    test_clusters = {row["source_cluster"] for row in test}
    overlap = train_clusters & test_clusters
    if overlap:
        raise RuntimeError(f"source cluster leakage: {len(overlap)} groups")
    audit = {
        "candidate_count": len(candidates),
        "train_count": len(train),
        "test_count": len(test),
        "train_cluster_count": len(train_clusters),
        "test_cluster_count": len(test_clusters),
        "unreadable_test_candidates": rejected_unreadable,
        "unreadable_source_headers": unreadable_sources,
        "source_index_sha256": sha256_file(source_index),
    }
    return train, test, audit


def _test_pairs(
    luts: list[dict[str, Any]], test_sources: list[dict[str, Any]], salt: str
) -> list[dict[str, Any]]:
    by_major: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in luts:
        by_major[row["taxonomy_major"]].append(row)
    for major, rows in by_major.items():
        rows.sort(key=lambda row: stable_digest(salt, major, row["lut_content_hash"]))

    majors = sorted(by_major)
    selected_luts: list[dict[str, Any]] = []
    cursor = {major: 0 for major in majors}
    while len(selected_luts) < len(test_sources):
        made_progress = False
        for major in majors:
            index = cursor[major]
            if index < len(by_major[major]):
                selected_luts.append(by_major[major][index])
                cursor[major] += 1
                made_progress = True
                if len(selected_luts) == len(test_sources):
                    break
        if not made_progress:
            raise RuntimeError("not enough LUTs for unique test-pair assignment")

    pairs: list[dict[str, Any]] = []
    for index, (source, lut) in enumerate(zip(test_sources, selected_luts, strict=True)):
        pairs.append(
            {
                "sample_id": f"pilot_test_{index:06d}",
                "source_id": source["source_id"],
                "source_cluster": source["source_cluster"],
                "source_path": source["path"],
                "style_index": lut["style_index"],
                "preset_id": lut["preset_id"],
                "lut_content_hash": lut["lut_content_hash"],
                "taxonomy_major": lut["taxonomy_major"],
                "taxonomy_minor": lut["taxonomy_minor"],
            }
        )
    if len({row["style_index"] for row in pairs}) != len(pairs):
        raise RuntimeError("test-pair LUT assignment unexpectedly contains duplicates")
    return pairs


def prepare(config_path: str | Path, *, force: bool = False) -> Path:
    config = load_config(config_path)
    destination = manifests_dir(config)
    lock_path = destination / "protocol_lock.json"
    if lock_path.exists() and not force:
        print(f"[prepare] using existing lock: {lock_path}")
        return lock_path
    destination.mkdir(parents=True, exist_ok=True)

    print("[prepare] auditing native LUT registry and packed grids", flush=True)
    luts, lut_audit = _freeze_luts(config)
    print("[prepare] selecting grouped held-out source panel", flush=True)
    train_sources, test_sources, source_audit = _freeze_sources(config)
    pairs = _test_pairs(luts, test_sources, str(config["data"]["test_pair_salt"]))

    manifest_paths = {
        "luts": destination / "luts.jsonl",
        "train_sources": destination / "train_sources.jsonl",
        "test_sources": destination / "test_sources.jsonl",
        "test_pairs": destination / "test_pairs.jsonl",
    }
    atomic_write_jsonl(manifest_paths["luts"], luts)
    atomic_write_jsonl(manifest_paths["train_sources"], train_sources)
    atomic_write_jsonl(manifest_paths["test_sources"], test_sources)
    atomic_write_jsonl(manifest_paths["test_pairs"], pairs)

    protocol_path = Path(config["pilot"]["protocol_reference"])
    if not protocol_path.is_absolute():
        protocol_path = Path(__file__).resolve().parents[2] / protocol_path
    config_source = Path(config["_config_path"])
    lock = {
        "pilot": config["pilot"],
        "lut_audit": lut_audit,
        "source_audit": source_audit,
        "test_pair_count": len(pairs),
        "test_pair_major_distribution": dict(
            sorted(collections.Counter(row["taxonomy_major"] for row in pairs).items())
        ),
        "manifest_sha256": {
            key: sha256_file(path) for key, path in manifest_paths.items()
        },
        "config_sha256": sha256_file(config_source),
        "protocol_sha256": sha256_file(protocol_path),
        "git": git_state(),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pillow": Image.__version__,
        },
        "declared_deviations": [
            "native 4000 cube LUTs only; 581 baked LUTs excluded",
            "all LUTs are seen calibration styles",
            "500 held-out source images",
            "renderer calibration only; no gate/VLM/SFT",
            "online natural-image synthesis plus native full-grid supervision",
            "single base seed for this pilot",
            "Vera preset latents zero-initialized without teacher prototypes",
        ],
    }
    atomic_write_json(lock_path, lock)
    print(
        f"[prepare] locked LUTs={len(luts)} train_sources={len(train_sources)} "
        f"test_sources={len(test_sources)} -> {lock_path}",
        flush=True,
    )
    return lock_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare(args.config, force=args.force)


if __name__ == "__main__":
    main()
