"""SAM3-ready source inventory and deterministic scene/mode allocation."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from dataset_build.tools.archive_reader import (
    default_db,
    iter_source_paths,
    path_exists,
    prefix_range,
    read_bytes,
)
from dataset_build.tools.land import (
    NOT_A_CACHE_ENTRY,
    SUBJECT_CACHE_GROUP,
    SUBJECT_MAX_AREA,
    SUBJECT_MIN_AREA,
)

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
        if area < SUBJECT_MIN_AREA or area > SUBJECT_MAX_AREA:
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


# 前缀过滤一律用 archive_reader.prefix_range 的范围比较（原因见该函数）。批次组名
# 是 "<group>/batch-NNNN"，而 chr(ord('/') + 1) == '0'，所以下界 "cache/subject/"
# 与上界 "cache/subject0" 精确覆盖该组及其全部批次子组。
_GROUP_WHERE = '("group" = ? OR ("group" >= ? AND "group" < ?))'
# ``cache_dir`` 前缀把结果限定在本次请求的 subject_cache 根下，顺带滤掉没有预
# 计算路径的行（json_extract 返回 NULL，范围比较必假）。
_ROW_WHERE = (
    f"{_GROUP_WHERE} "
    "AND json_extract(meta, '$.ineligible_reason') IS NOT ? "
    "AND json_extract(meta, '$.cache_dir') >= ? AND json_extract(meta, '$.cache_dir') < ?"
)


def _precomputed_inventory(
    root: Path,
    by_id: Mapping[str, str],
    by_path: Mapping[str, str],
) -> tuple[list[SourceRecord], dict[str, int]] | None:
    """Read the gate result that ``land`` precomputed, or None to fall back.

    Every check in ``_inspect_cache_dir`` except ``mask_area`` is answerable from
    metadata, and ``mask_area`` itself was computed when the bytes were packed —
    so the whole 56,777-entry, 8-minute decode pass collapses into three queries.
    Anything unexpected (no catalog, a group that predates the precomputation, a
    row that will not parse) returns None so the caller re-inspects the archive.
    """
    db_path = default_db()
    if not db_path.is_file():
        return None
    lower, upper = prefix_range(f"{os.fspath(root).rstrip('/')}/")
    group = (SUBJECT_CACHE_GROUP, *prefix_range(f"{SUBJECT_CACHE_GROUP}/"))
    row_params = (*group, NOT_A_CACHE_ENTRY, lower, upper)
    connection = sqlite3.connect(f"{db_path.as_uri()}?immutable=1", uri=True)
    try:
        connection.row_factory = sqlite3.Row
        total, missing = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(json_extract(meta, '$.eligible') IS NULL), 0) "
            f"FROM samples WHERE {_GROUP_WHERE}",
            group,
        ).fetchone()
        if not total or missing:
            # 组不存在，或还没回填过预计算字段：交回旧路径。
            return None
        counts = {
            str(row["reason"]): int(row["n"])
            for row in connection.execute(
                "SELECT COALESCE(json_extract(meta, '$.ineligible_reason'), 'eligible') AS reason,"
                f" COUNT(*) AS n FROM samples WHERE {_ROW_WHERE} GROUP BY reason",
                row_params,
            )
        }
        rows = connection.execute(
            f"SELECT sample_id, meta FROM samples WHERE {_ROW_WHERE}"
            " AND json_extract(meta, '$.eligible') = 1",
            row_params,
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        connection.close()
    if not counts:
        return None
    eligible: list[SourceRecord] = []
    for row in rows:
        try:
            meta = json.loads(str(row["meta"]))
            cache_dir = Path(str(meta["cache_dir"]))
            source_path = Path(str(meta["source_path"]))
            asset_id = str(meta.get("asset_id") or "")
            real = os.path.realpath(source_path)
            eligible.append(
                SourceRecord(
                    source_id=asset_id or stable_id("source", real),
                    source_path=source_path,
                    cache_dir=cache_dir,
                    subject_path=cache_dir / "subject.png",
                    subject_meta_path=cache_dir / "subject.json",
                    # 本机 postgres 还在时它仍是权威，预计算值只是它消失后的兜底。
                    scene=_normalize_scene(
                        by_id.get(asset_id) or by_path.get(real) or meta.get("scene")
                    ),
                    subject=dict(meta.get("subject") or {}),
                    mask_area=float(meta["mask_area"]),
                )
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
    return eligible, {"cache_entries": sum(counts.values()), **counts}


def build_inventory(
    subject_cache: str | os.PathLike[str],
    postgres_dsn: str,
    *,
    workers: int = 16,
    connection_factory: Callable[[str], Any] | None = None,
    legacy_inspect: bool = False,
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
        # 迁移后本机树是 SAM3 relabel 写出的稀疏覆盖层，不是全量池：有预计算
        # 归档行时以归档为底座、本机目录逐条实检并覆盖同源，否则（legacy 全量
        # 树 / 无预计算）继续走下方的全量实检。
        if not legacy_inspect:
            precomputed = _precomputed_inventory(root, by_id, by_path)
            if precomputed is not None:
                rows, counts = precomputed
                overlay = {str(entry): entry for entry in cache_dirs}
                merged = [r for r in rows if str(r.cache_dir) not in overlay]
                counts = dict(counts)
                counts["local_overlay"] = len(overlay)
                for entry in overlay.values():
                    record, reason = _inspect_cache_dir(entry, by_id, by_path)
                    counts[f"overlay_{reason}"] = counts.get(f"overlay_{reason}", 0) + 1
                    if record is not None:
                        merged.append(record)
                counts["eligible"] = len(merged)
                merged = _require_unique_source_ids(merged)
                merged.sort(key=lambda row: row.source_id)
                return SourceInventoryResult(
                    tuple(merged), dict(sorted(counts.items())), metadata_status
                )
    else:
        # The local cache tree is archived: the gate was precomputed at landing,
        # so read it back instead of re-decoding every entry.  ``legacy_inspect``
        # is the rollback switch onto the reference implementation below.
        if not legacy_inspect:
            precomputed = _precomputed_inventory(root, by_id, by_path)
            if precomputed is not None:
                rows, counts = precomputed
                rows = _require_unique_source_ids(rows)
                rows.sort(key=lambda row: row.source_id)
                return SourceInventoryResult(
                    tuple(rows), dict(sorted(counts.items())), metadata_status
                )
        # Enumerate the same entries from the archive instead of failing, so a
        # migrated pool still builds.
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
    eligible: list[SourceRecord] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for record, reason in pool.map(
            lambda path: _inspect_cache_dir(path, by_id, by_path), cache_dirs
        ):
            counts[reason] = counts.get(reason, 0) + 1
            if record is not None:
                eligible.append(record)
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
