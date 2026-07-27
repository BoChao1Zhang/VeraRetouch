"""Validated, resumable import of the explicitly authorized r5 local snapshot."""
from __future__ import annotations

import errno
import filecmp
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
from PIL import Image, ImageOps

from .config import DatabuildConfig
from .state import ArtifactStore, file_digest, stable_id
from .visibility import bounded_working_pair, objective_edit_hints


_RESAMPLING = getattr(Image, "Resampling", Image)


class LegacyImportError(RuntimeError):
    """The legacy snapshot cannot be represented without losing provenance."""


@dataclass(frozen=True, slots=True)
class LegacyGroupInput:
    record: dict[str, Any]
    winners: tuple[dict[str, Any], ...]

    @property
    def source_asset_id(self) -> str:
        return str(self.record["source_asset_id"])


@dataclass(frozen=True, slots=True)
class LegacySnapshot:
    groups: tuple[LegacyGroupInput, ...]
    source_group_count: int
    sft_row_count: int
    skipped_groups_without_winners: int
    input_artifacts: dict[str, dict[str, Any]]


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise LegacyImportError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise LegacyImportError(f"record is not an object at {path}:{line_number}")
            yield line_number, value


def _required_string(row: Mapping[str, Any], key: str, where: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise LegacyImportError(f"{where} requires non-empty {key}")
    return value


def load_legacy_snapshot(config: DatabuildConfig) -> LegacySnapshot:
    """Validate and bind every legacy SFT winner before opening an output store."""
    legacy = config.legacy_import
    if legacy is None:
        raise LegacyImportError("legacy_import configuration is required")

    winner_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, row in _iter_jsonl(legacy.sft_jsonl):
        where = f"legacy SFT line {line_number}"
        before = _required_string(row, "I_in", where)
        after = _required_string(row, "I_tar", where)
        if row.get("task_type") != "local":
            raise LegacyImportError(f"{where} is not a local task")
        local = row.get("local")
        qa = row.get("qa")
        if not isinstance(local, Mapping) or not isinstance(qa, Mapping):
            raise LegacyImportError(f"{where} requires local and qa objects")
        cgt_path = _required_string(local, "C_GT", where)
        rank = qa.get("rank")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank not in {0, 1}:
            raise LegacyImportError(f"{where} requires zero-based qa.rank 0 or 1")
        for label, path in (("I_in", before), ("I_tar", after), ("C_GT", cgt_path)):
            target = Path(path)
            if not target.is_file() or target.stat().st_size <= 0:
                raise LegacyImportError(f"{where} references missing or empty {label}: {target}")
        key = (before, after)
        if key in winner_rows:
            raise LegacyImportError(f"duplicate legacy SFT before/after pair at line {line_number}")
        winner_rows[key] = row

    if len(winner_rows) != legacy.expected_sft_rows:
        raise LegacyImportError(
            f"legacy SFT count mismatch: expected {legacy.expected_sft_rows}, "
            f"found {len(winner_rows)}"
        )

    matched: set[tuple[str, str]] = set()
    sources: set[str] = set()
    imported: list[LegacyGroupInput] = []
    source_group_count = 0
    for line_number, row in _iter_jsonl(legacy.groups_jsonl):
        source_group_count += 1
        where = f"legacy group line {line_number}"
        source = _required_string(row, "source", where)
        source_asset_id = _required_string(row, "source_asset_id", where)
        if source_asset_id in sources:
            raise LegacyImportError(f"duplicate legacy source_asset_id: {source_asset_id}")
        sources.add(source_asset_id)
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 8 \
                or any(not isinstance(candidate, dict) for candidate in candidates):
            raise LegacyImportError(f"{where} is not an eight-candidate group")

        group_winners: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(candidates):
            after = _required_string(
                candidate, "after_path", f"{where} candidate {candidate_index}"
            )
            key = (source, after)
            winner = winner_rows.get(key)
            if winner is None:
                continue
            if key in matched:
                raise LegacyImportError(
                    f"legacy SFT pair maps to multiple candidates: {source} -> {after}"
                )
            candidate_local = candidate.get("local")
            if not isinstance(candidate_local, Mapping):
                raise LegacyImportError(f"{where} winner candidate is missing local metadata")
            candidate_cgt = _required_string(
                candidate_local, "cgt_path", f"{where} candidate {candidate_index}"
            )
            if candidate_cgt != str(winner["local"]["C_GT"]):
                raise LegacyImportError(f"{where} winner C_GT path does not match its SFT row")
            matched.add(key)
            group_winners.append(winner)

        if not group_winners:
            continue
        group_winners.sort(key=lambda winner: int(winner["qa"]["rank"]))
        expected_ranks = list(range(len(group_winners)))
        actual_ranks = [int(winner["qa"]["rank"]) for winner in group_winners]
        if len(group_winners) > 2 or actual_ranks != expected_ranks:
            raise LegacyImportError(
                f"{where} has invalid winner ranks: {actual_ranks}"
            )
        imported.append(LegacyGroupInput(row, tuple(group_winners)))

    if source_group_count != legacy.expected_source_groups:
        raise LegacyImportError(
            f"legacy group count mismatch: expected {legacy.expected_source_groups}, "
            f"found {source_group_count}"
        )
    if len(imported) != config.target_groups:
        raise LegacyImportError(
            f"legacy imported-group count mismatch: config target_groups={config.target_groups}, "
            f"snapshot groups with winners={len(imported)}"
        )
    unmatched = set(winner_rows).difference(matched)
    if unmatched:
        raise LegacyImportError(f"{len(unmatched)} legacy SFT rows do not map to a group candidate")

    return LegacySnapshot(
        groups=tuple(imported),
        source_group_count=source_group_count,
        sft_row_count=len(winner_rows),
        skipped_groups_without_winners=source_group_count - len(imported),
        input_artifacts={
            "groups.jsonl": file_digest(legacy.groups_jsonl),
            "sft.jsonl": file_digest(legacy.sft_jsonl),
        },
    )


def legacy_group_id(config: DatabuildConfig, group: LegacyGroupInput) -> str:
    source_id = stable_id("source", config.legacy_import.protocol, group.source_asset_id)
    return stable_id("group", config.build_id, source_id, "local", 0)


def _materialize_asset(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise LegacyImportError(f"legacy asset is missing or empty: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination) or filecmp.cmp(source, destination, shallow=False):
            return
        raise LegacyImportError(f"conflicting migrated asset already exists: {destination}")
    try:
        os.link(source, destination)
        return
    except OSError as exc:
        if exc.errno not in {errno.EXDEV, errno.EPERM, errno.EACCES}:
            raise
    temporary = destination.with_name(destination.name + f".tmp-{os.getpid()}")
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _subject(local: Mapping[str, Any]) -> dict[str, Any]:
    value = local.get("subject")
    if isinstance(value, Mapping):
        result = dict(value)
        if not str(result.get("name") or "").strip():
            result["name"] = str(result.get("concept") or "subject")
        return result
    return {"name": str(value or "subject")}


def _winner_hints(
    source_image: Image.Image,
    after_path: Path,
    cgt_path: Path,
) -> tuple[dict[str, dict[str, float | str]], float]:
    with Image.open(after_path) as opened:
        after_image = ImageOps.exif_transpose(opened).convert("RGB")
        after_image.load()
    before_image = source_image.resize(after_image.size, _RESAMPLING.LANCZOS)
    with Image.open(cgt_path) as opened:
        alpha_image = opened.convert("L")
        if alpha_image.size != after_image.size:
            alpha_image = alpha_image.resize(after_image.size, _RESAMPLING.BILINEAR)
        alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    before = np.asarray(before_image, dtype=np.float32) / 255.0
    after = np.asarray(after_image, dtype=np.float32) / 255.0
    before, after, working_alpha = bounded_working_pair(before, after, alpha, 256)
    return objective_edit_hints(before, after, weight=working_alpha), float(alpha.mean())


def convert_legacy_group(
    config: DatabuildConfig,
    group: LegacyGroupInput,
    store: ArtifactStore,
    *,
    timestamp: str,
) -> dict[str, Any]:
    """Convert one validated group and materialize its immutable candidate assets."""
    legacy = config.legacy_import
    if legacy is None:
        raise LegacyImportError("legacy_import configuration is required")
    raw = group.record
    source_path = Path(str(raw["source"]))
    with Image.open(source_path) as opened:
        source_image = ImageOps.exif_transpose(opened).convert("RGB")
        source_image.load()

    source_id = stable_id("source", legacy.protocol, group.source_asset_id)
    group_id = legacy_group_id(config, group)
    winner_by_after = {str(row["I_tar"]): row for row in group.winners}
    mode_counts: dict[str, int] = {}
    converted: list[dict[str, Any]] = []
    winner_bindings: list[tuple[int, str, int]] = []

    for slot_index, candidate in enumerate(raw["candidates"]):
        local = candidate.get("local")
        qa_raw = candidate.get("qa")
        if not isinstance(local, Mapping) or not isinstance(qa_raw, Mapping):
            raise LegacyImportError("legacy candidate requires local and qa objects")
        mode = str(local.get("mode") or local.get("mask_type") or "unknown")
        mode_index = mode_counts.get(mode, 0)
        mode_counts[mode] = mode_index + 1
        slot_id = f"legacy-{slot_index:02d}-{mode}"
        candidate_id = stable_id("candidate", group_id, slot_id)
        old_after = Path(str(candidate["after_path"]))
        old_cgt = Path(str(local.get("cgt_path") or ""))
        mask_key = str(local.get("mask_unit_id") or old_cgt)
        mask_id = stable_id("mask", config.build_id, source_id, mask_key)
        after_path = store.assets_root / "candidates" / f"{candidate_id}.jpg"
        cgt_path = store.assets_root / "masks" / f"{mask_id}.png"
        _materialize_asset(old_after, after_path)
        _materialize_asset(old_cgt, cgt_path)

        winner = winner_by_after.get(str(old_after))
        hints = None
        effective_alpha_mean = None
        recipe: Any = {
            "base_preset_id": local.get("base_preset_id"),
            "base_preset_path": local.get("base_preset_path"),
            "base_preset_content_hash": local.get("base_preset_content_hash"),
            "mask_type": local.get("mask_type"),
            "geom": local.get("geom"),
            "blend_mode": local.get("blend_mode"),
        }
        if winner is not None:
            hints, effective_alpha_mean = _winner_hints(source_image, old_after, old_cgt)
            recipe = winner.get("recipe") or recipe

        onealign = qa_raw.get("iaa_mixed")
        source_onealign = qa_raw.get("source_iaa", raw.get("source_iaa"))
        qa = {
            "onealign": onealign,
            "source_onealign": source_onealign,
            "q": float(qa_raw.get("q") or 0.0),
            "improvement": float(qa_raw.get("iaa_impr") or 0.5),
            "reliable": bool(qa_raw.get("reliable")),
            "veto": bool(qa_raw.get("veto")),
            "veto_flags": list(qa_raw.get("det") or []),
            "qa_mode": "legacy_onealign",
            "legacy": dict(qa_raw),
        }
        rank = int(qa_raw.get("rank", slot_index)) + 1
        subject = _subject(local)
        record = {
            "candidate_id": candidate_id,
            "slot_id": slot_id,
            "slot_index": slot_index,
            "preset_id": str(candidate.get("preset_id") or local.get("base_preset_id") or ""),
            "preset_path": str(candidate.get("preset_path") or local.get("base_preset_path") or ""),
            "format": str(candidate.get("fmt") or "xmp"),
            "kind": str(candidate.get("kind") or "local_preset"),
            "style_name": None,
            "major": str(raw.get("style_major") or "legacy_local"),
            "minor": mode,
            "after_path": str(after_path),
            "render_engine": str(local.get("engine") or "legacy_gpu_local_preset"),
            "render_diagnostics": {
                "legacy_import": True,
                "protocol": legacy.protocol,
                "axis_fix_status": legacy.axis_fix_status,
            },
            "visibility": {
                "accepted": True,
                "legacy_not_recomputed": True,
            },
            "objective_hints": hints,
            "attempt_lineage": {
                "legacy_source_asset_id": group.source_asset_id,
                "legacy_candidate_content_hash": candidate.get("content_hash"),
            },
            "recipe": recipe,
            "qa": qa,
            "rank": rank,
            "slot_mode": mode,
            "mode_index": mode_index,
            "pairing_index": slot_index,
            "mask_id": mask_id,
            "cgt_path": str(cgt_path),
            "subject": subject,
            "region": str(local.get("region") or ""),
            "geometry": local.get("geom"),
            "raw_alpha_mean": None,
            "amount": None,
            "effective_alpha_mean": effective_alpha_mean,
            "legacy": {
                "after_path": str(old_after),
                "cgt_path": str(old_cgt),
                "mask_unit_id": local.get("mask_unit_id"),
                "mode": mode,
            },
        }
        converted.append(record)
        if winner is not None:
            winner_bindings.append((int(winner["qa"]["rank"]), candidate_id, rank))

    winner_bindings.sort()
    if len(winner_bindings) != len(group.winners):
        raise LegacyImportError("legacy winner binding changed during conversion")
    return {
        "schema_version": 1,
        "build_id": config.build_id,
        "group_id": group_id,
        "source_id": source_id,
        "source_path": str(source_path),
        "scene": "legacy_unknown",
        "subject": converted[0]["subject"],
        "render_mode": "local",
        "preset_filter": "xmp",
        "group_attempt": 0,
        "major": str(raw.get("style_major") or "legacy_local"),
        "coverage_cycle": None,
        "coverage_position": None,
        "reservation_id": stable_id("legacy-reservation", group_id),
        "candidates": converted,
        "winner_ids": [candidate_id for _, candidate_id, _ in winner_bindings],
        "winner_ranks": [rank for _, _, rank in winner_bindings],
        "source_onealign": raw.get("source_iaa"),
        "stage_timestamps": {"import_completed_at": timestamp},
        "legacy_import": {
            "protocol": legacy.protocol,
            "source_asset_id": group.source_asset_id,
            "axis_fix_status": legacy.axis_fix_status,
            "canonical_mask_contract": False,
        },
    }


__all__ = [
    "LegacyGroupInput",
    "LegacyImportError",
    "LegacySnapshot",
    "convert_legacy_group",
    "legacy_group_id",
    "load_legacy_snapshot",
]
