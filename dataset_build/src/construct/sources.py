"""SAM3-ready source inventory and deterministic scene/mode allocation."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dataset_build.tools.archive_reader import iter_source_paths, path_exists, read_bytes

from .config import MixConfig
from .state import stable_id


SCENE_WEIGHTS: dict[str, float] = {
    "portrait": 16.0,
    "landscape": 14.0,
    "food": 12.0,
    "unknown": 12.0,
    "still_life": 10.0,
    "architecture": 9.0,
    "night": 9.0,
    "street": 8.0,
    "wedding": 6.0,
    "product": 4.0,
}


@dataclass(frozen=True, slots=True)
class SourceRecord:
    source_id: str
    source_path: Path
    cache_dir: Path
    subject_path: Path
    subject_meta_path: Path
    scene: str
    subject: dict[str, Any]
    mask_area: float


@dataclass(frozen=True, slots=True)
class SourceInventoryResult:
    eligible: tuple[SourceRecord, ...]
    counts: dict[str, int]
    scene_metadata_status: str


@dataclass(frozen=True, slots=True)
class SourceAllocation:
    local_target: int
    global_target: int
    local: tuple[SourceRecord, ...]
    global_: tuple[SourceRecord, ...]

    @property
    def target_groups(self) -> int:
        return self.local_target + self.global_target


def largest_remainder(weights: Mapping[str, float], total: int) -> dict[str, int]:
    """Convert non-negative weights to exact integer quotas deterministically."""
    if total < 0:
        raise ValueError("total must be non-negative")
    if not weights:
        if total:
            raise ValueError("weights must not be empty")
        return {}
    if any(not math.isfinite(float(value)) or float(value) < 0 for value in weights.values()):
        raise ValueError("weights must be finite and non-negative")
    weight_sum = float(sum(weights.values()))
    if weight_sum <= 0:
        raise ValueError("at least one weight must be positive")
    exact = {key: total * float(value) / weight_sum for key, value in weights.items()}
    quotas = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(quotas.values())
    order = sorted(weights, key=lambda key: (-(exact[key] - quotas[key]), key))
    for key in order[:remaining]:
        quotas[key] += 1
    return quotas


def mode_targets(mix: MixConfig, target_groups: int) -> dict[str, int]:
    return largest_remainder({"local": mix.local, "global": mix.global_}, target_groups)


def _seed_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _normalize_scene(value: object) -> str:
    scene = str(value or "").strip().lower()
    if scene in {"", "any", "other", "none", "null", "unknown"}:
        return "unknown"
    return scene if scene in SCENE_WEIGHTS else "unknown"


def _require_unique_source_ids(sources: Iterable[SourceRecord]) -> list[SourceRecord]:
    rows = list(sources)
    seen: set[str] = set()
    for source in rows:
        if not isinstance(source.source_id, str) or not source.source_id:
            raise ValueError("source inventory contains an invalid source_id")
        if source.source_id in seen:
            raise ValueError(f"source inventory contains duplicate source_id: {source.source_id}")
        seen.add(source.source_id)
    return rows


def load_scene_metadata(
    postgres_dsn: str,
    *,
    connection_factory: Callable[[str], Any] | None = None,
) -> tuple[dict[str, str], dict[str, str], str]:
    """Best-effort scene metadata; an outage yields the explicit unknown bucket."""
    if connection_factory is None:
        try:
            import psycopg
        except ImportError:
            return {}, {}, "postgres_driver_unavailable"
        connection_factory = psycopg.connect
    by_id: dict[str, str] = {}
    by_path: dict[str, str] = {}
    try:
        with connection_factory(postgres_dsn) as conn:
            rows = conn.execute(
                "SELECT asset_id, path, COALESCE(scene, 'unknown') AS scene "
                "FROM assets WHERE asset_type='image'"
            ).fetchall()
            for row in rows:
                if isinstance(row, Mapping):
                    asset_id, path, scene = row.get("asset_id"), row.get("path"), row.get("scene")
                else:
                    asset_id, path, scene = row[0], row[1], row[2]
                normalized = _normalize_scene(scene)
                if asset_id:
                    by_id[str(asset_id)] = normalized
                if path:
                    by_path[os.path.realpath(os.fspath(path))] = normalized
    except Exception as exc:  # noqa: BLE001 - metadata outage must not stop production
        return {}, {}, f"postgres_unavailable:{type(exc).__name__}"
    return by_id, by_path, "ok"


def _inspect_cache_dir(
    cache_dir: Path,
    by_id: Mapping[str, str],
    by_path: Mapping[str, str],
) -> tuple[SourceRecord | None, str]:
    meta_path = cache_dir / "subject.json"
    try:
        meta_bytes = read_bytes(meta_path)
    except (KeyError, OSError):
        return None, "missing_subject_json"
    try:
        meta = json.loads(meta_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, "invalid_subject_json"
    if not isinstance(meta, dict) or meta.get("status") != "ready":
        return None, "subject_not_ready"
    mask_path = cache_dir / "subject.png"
    if not path_exists(mask_path):
        return None, "missing_subject_png"
    source_value = meta.get("source_path")
    if not isinstance(source_value, str) or not source_value:
        return None, "missing_source_path"
    source_path = Path(source_value)
    if not path_exists(source_path):
        return None, "missing_source_image"
    try:
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(read_bytes(mask_path))) as mask_image:
            mask_image.load()
            mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
        if mask.ndim != 2 or not mask.size or not np.isfinite(mask).all():
            return None, "invalid_subject_mask"
        area = float((mask > 0.5).mean())
        if area < 0.005 or area > 0.85:
            return None, "subject_mask_area_guard"
        with Image.open(io.BytesIO(read_bytes(source_path))) as source_image:
            source_image.load()
            oriented = ImageOps.exif_transpose(source_image)
            if oriented.width <= 0 or oriented.height <= 0:
                return None, "invalid_source_image"
            oriented.convert("RGB").getpixel((0, 0))
    except Exception:  # noqa: BLE001 - corrupt cache/source is an eligibility failure
        return None, "decode_or_integrity_failure"
    asset_id = str(meta.get("asset_id") or "")
    source_id = asset_id or stable_id("source", os.path.realpath(source_path))
    scene = by_id.get(asset_id) or by_path.get(os.path.realpath(source_path)) or "unknown"
    subject = {
        "name": meta.get("sam_prompt") or meta.get("description") or "subject",
        "description": meta.get("description"),
        "scope": meta.get("scope"),
        "n_members": meta.get("n_members"),
        "area": round(area, 6),
    }
    return SourceRecord(
        source_id=source_id,
        source_path=source_path,
        cache_dir=cache_dir,
        subject_path=mask_path,
        subject_meta_path=meta_path,
        scene=_normalize_scene(scene),
        subject=subject,
        mask_area=area,
    ), "eligible"


def refresh_source_record(source: SourceRecord) -> tuple[SourceRecord | None, str]:
    """Re-run canonical integrity checks after an instance-level SAM3 relabel."""
    record, reason = _inspect_cache_dir(
        source.cache_dir,
        {source.source_id: source.scene},
        {os.path.realpath(source.source_path): source.scene},
    )
    if record is None:
        return None, reason
    if record.source_id != source.source_id:
        return None, "source_id_changed_after_relabel"
    return SourceRecord(
        source_id=record.source_id,
        source_path=record.source_path,
        cache_dir=record.cache_dir,
        subject_path=record.subject_path,
        subject_meta_path=record.subject_meta_path,
        scene=source.scene,
        subject=record.subject,
        mask_area=record.mask_area,
    ), reason


def build_inventory(
    subject_cache: str | os.PathLike[str],
    postgres_dsn: str,
    *,
    workers: int = 16,
    connection_factory: Callable[[str], Any] | None = None,
) -> SourceInventoryResult:
    root = Path(subject_cache)
    by_id, by_path, metadata_status = load_scene_metadata(
        postgres_dsn, connection_factory=connection_factory
    )
    if root.is_dir():
        cache_dirs = sorted(
            (
                entry
                for entry in root.iterdir()
                if entry.is_dir() and not entry.name.startswith("_")
            ),
            key=lambda path: path.name,
        )
    else:
        # The local cache tree is archived: enumerate the same entries from the
        # archive instead of failing, so a migrated pool still builds.
        cache_dirs = sorted(
            {
                Path(path).parent
                for path in iter_source_paths(f"{root}/", endswith="/subject.json")
                if not Path(path).parent.name.startswith("_")
            },
            key=lambda path: path.name,
        )
        if not cache_dirs:
            raise FileNotFoundError(root)
    counts: dict[str, int] = {"cache_entries": len(cache_dirs)}
    inspect = lambda path: _inspect_cache_dir(path, by_id, by_path)
    if workers <= 1:
        inspected = map(inspect, cache_dirs)
    else:
        pool = ThreadPoolExecutor(max_workers=workers)
        inspected = pool.map(inspect, cache_dirs)
    eligible: list[SourceRecord] = []
    try:
        for record, reason in inspected:
            counts[reason] = counts.get(reason, 0) + 1
            if record is not None:
                eligible.append(record)
    finally:
        if workers > 1:
            pool.shutdown(wait=True)
    eligible = _require_unique_source_ids(eligible)
    eligible.sort(key=lambda row: row.source_id)
    return SourceInventoryResult(tuple(eligible), dict(sorted(counts.items())), metadata_status)


def scene_stratified_order(
    sources: Iterable[SourceRecord], build_id: str, seed: int
) -> list[SourceRecord]:
    """Seeded shuffle within strata followed by weighted deterministic interleaving."""
    strata: dict[str, list[SourceRecord]] = {scene: [] for scene in SCENE_WEIGHTS}
    for source in _require_unique_source_ids(sources):
        strata.setdefault(_normalize_scene(source.scene), []).append(source)
    for scene, rows in strata.items():
        rows.sort(
            key=lambda row: (_seed_int(build_id, seed, scene, row.source_id), row.source_id)
        )
    positions = {scene: 0 for scene in strata}
    emitted = {scene: 0 for scene in strata}
    ordered: list[SourceRecord] = []
    remaining = sum(len(rows) for rows in strata.values())
    while remaining:
        available = [scene for scene, rows in strata.items() if positions[scene] < len(rows)]
        scene = min(
            available,
            key=lambda item: (
                (emitted[item] + 1) / SCENE_WEIGHTS.get(item, SCENE_WEIGHTS["unknown"]),
                _seed_int(build_id, seed, "scene", item),
                item,
            ),
        )
        ordered.append(strata[scene][positions[scene]])
        positions[scene] += 1
        emitted[scene] += 1
        remaining -= 1
    return ordered


def allocate_sources(
    sources: Iterable[SourceRecord],
    *,
    build_id: str,
    seed: int,
    target_groups: int,
    mix: MixConfig,
) -> SourceAllocation:
    ordered = scene_stratified_order(sources, build_id, seed)
    targets = mode_targets(mix, target_groups)
    if len(ordered) < target_groups:
        # The orchestrator records an explicit target shortfall after consuming this pool.
        target_prefix = len(ordered)
    else:
        target_prefix = target_groups
    prefix_labels = ["local"] * min(targets["local"], target_prefix)
    prefix_labels.extend(["global"] * min(targets["global"], target_prefix - len(prefix_labels)))
    # If a scarce pool truncates one quota, fill remaining prefix positions by configured weight.
    while len(prefix_labels) < target_prefix:
        prefix_labels.append("local" if mix.local >= mix.global_ else "global")
    random.Random(_seed_int(build_id, seed, "initial-modes")).shuffle(prefix_labels)

    remaining_n = len(ordered) - target_prefix
    replacement_counts = largest_remainder(
        {"local": mix.local, "global": mix.global_}, remaining_n
    ) if remaining_n else {"local": 0, "global": 0}
    replacement_labels = ["local"] * replacement_counts["local"]
    replacement_labels.extend(["global"] * replacement_counts["global"])
    random.Random(_seed_int(build_id, seed, "replacement-modes")).shuffle(replacement_labels)

    local_rows: list[SourceRecord] = []
    global_rows: list[SourceRecord] = []
    for source, mode in zip(ordered, [*prefix_labels, *replacement_labels]):
        (local_rows if mode == "local" else global_rows).append(source)
    return SourceAllocation(
        local_target=targets["local"],
        global_target=targets["global"],
        local=tuple(local_rows),
        global_=tuple(global_rows),
    )
