"""Single TOML-driven canonical databuild orchestrator."""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

from .canonical_masks import MaskPlanError, build_mask_plan, pair_mask_slots
from .canonical_qa import OneAlignScorer, QaError, rank_candidates
from .config import DatabuildConfig, load_config, redact_text
from .legacy_import import (
    LegacyGroupInput,
    LegacySnapshot,
    convert_legacy_group,
    legacy_group_id,
    load_legacy_snapshot,
)
from .presets import CoverageSelector, PresetCatalog, PresetError
from .projection import ProjectionResult, project_artifacts
from .rendering import (
    LocalGpuOnlyRenderer,
    preprocess_source,
    save_candidate_jpeg,
    save_cgt_png,
)
from .responses import ResponsesAnnotator, preflight_openai_sdk
from .sources import (
    SourceInventoryResult,
    SourceRecord,
    allocate_sources,
    build_inventory,
    refresh_source_record,
)
from .state import ArtifactStore, StateError, file_digest, stable_id
from .visibility import objective_edit_hints, visibility_metrics


class PipelineError(RuntimeError):
    """The canonical lifecycle cannot make safe progress."""


class Renderer(Protocol):
    device: str

    def bind_catalog(self, catalog: PresetCatalog) -> None: ...
    def assert_ready(self) -> None: ...
    def render(self, source: Any, preset: Any, mask: Any = None) -> Any: ...


class Scorer(Protocol):
    def score(self, path: str) -> float | None: ...


class QueueDrainer(Protocol):
    def drain(self, *, max_workers: int | None = None) -> dict[str, int]: ...


@dataclass(slots=True)
class PipelineDependencies:
    inventory_loader: Callable[[DatabuildConfig], SourceInventoryResult]
    catalog_loader: Callable[[DatabuildConfig], PresetCatalog]
    renderer_factory: Callable[[DatabuildConfig], Renderer]
    scorer_factory: Callable[[], Scorer]
    sdk_preflight: Callable[[], None]
    annotator_factory: Callable[[DatabuildConfig, ArtifactStore], QueueDrainer]
    relabeler: Callable[[list[SourceRecord], DatabuildConfig, int], Mapping[str, str]]
    projector: Callable[[ArtifactStore, Mapping[str, Any], DatabuildConfig], ProjectionResult]
    now: Callable[[], datetime]


def _default_relabeler(
    sources: list[SourceRecord], config: DatabuildConfig, attempt: int
) -> Mapping[str, str]:
    from dataset_build.source_qa.sam3_subject_instances import relabel_sources

    rows = [
        {"asset_id": source.source_id, "source_path": str(source.source_path)}
        for source in sources
    ]
    return relabel_sources(
        rows,
        cache_root=str(config.sources.subject_cache),
        device="cuda:0",
        seed=config.seed + attempt,
        vlm_base_url=config.annotation.local.base_url,
        vlm_api_key=config.annotation.local.api_key,
        vlm_model=config.annotation.local.model,
    )


def default_dependencies() -> PipelineDependencies:
    return PipelineDependencies(
        inventory_loader=lambda config: build_inventory(
            config.sources.subject_cache, config.sources.postgres_dsn
        ),
        catalog_loader=PresetCatalog.load,
        renderer_factory=LocalGpuOnlyRenderer.create,
        scorer_factory=lambda: OneAlignScorer.create("cuda:0"),
        sdk_preflight=preflight_openai_sdk,
        annotator_factory=lambda config, store: ResponsesAnnotator(config.annotation, store),
        relabeler=_default_relabeler,
        projector=lambda store, manifest, config: project_artifacts(
            store, manifest, config.viewer.postgres_dsn
        ),
        now=lambda: datetime.now(timezone.utc),
    )


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _add_manifest_self_artifact(manifest: dict[str, Any]) -> None:
    """Record a verifiable scoped digest without claiming a self-referential hash."""
    artifacts = manifest.setdefault("artifacts", {})
    scoped_manifest = dict(manifest)
    scoped_artifacts = dict(artifacts)
    scoped_artifacts.pop("manifest.json", None)
    scoped_manifest["artifacts"] = scoped_artifacts
    canonical = json.dumps(
        scoped_manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    entry = {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "hash_scope": "manifest excluding artifacts.manifest.json",
        "records": 1,
        "bytes": 0,
    }
    artifacts["manifest.json"] = entry
    for _ in range(8):
        size = len((json.dumps(
            manifest, ensure_ascii=False, sort_keys=True, indent=2
        ) + "\n").encode("utf-8"))
        if entry["bytes"] == size:
            break
        entry["bytes"] = size


def _load_existing_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StateError(f"cannot resume invalid manifest: {path}") from exc
    if not isinstance(value, dict):
        raise StateError("manifest must be a JSON object")
    return value


def _preflight_scorer(scorer: Scorer, inventory: SourceInventoryResult) -> None:
    """Exercise one real OneAlign forward before any authoritative artifact exists."""
    if not inventory.eligible:
        raise PipelineError("source inventory contains no eligible image for OneAlign preflight")
    sample_path = str(inventory.eligible[0].source_path)
    try:
        score = scorer.score(sample_path)
    except Exception as exc:  # noqa: BLE001 - startup boundary needs one classified error
        raise QaError(
            f"OneAlign live preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    if score is None or not isinstance(score, (int, float)) \
            or not 0.0 <= float(score) <= 100.0:
        raise QaError("OneAlign live preflight returned an invalid score")


class CanonicalPipeline:
    def __init__(
        self,
        config: DatabuildConfig,
        dependencies: PipelineDependencies,
        inventory: SourceInventoryResult,
        catalog: PresetCatalog,
        renderer: Renderer,
        scorer: Scorer,
        store: ArtifactStore,
        existing_manifest: Mapping[str, Any] | None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.inventory = inventory
        self.catalog = catalog
        self.renderer: Renderer | None = renderer
        self.scorer: Scorer | None = scorer
        self.store = store
        self.started_at = str(
            (existing_manifest or {}).get("started_at")
            or _timestamp(self.dependencies.now())
        )
        self.allocation = allocate_sources(
            inventory.eligible,
            build_id=config.build_id,
            seed=config.seed,
            target_groups=config.target_groups,
            mix=config.mix,
        )
        historical = tuple(store.groups.values())
        self.selectors = {
            "local": CoverageSelector(
                catalog,
                build_id=config.build_id,
                seed=config.seed,
                render_mode="local",
                preset_filter=config.preset_filter,
                historical_groups=historical,
            ) if self.allocation.local_target else None,
            "global": CoverageSelector(
                catalog,
                build_id=config.build_id,
                seed=config.seed,
                render_mode="global",
                preset_filter=config.preset_filter,
                historical_groups=historical,
            ) if self.allocation.global_target else None,
        }

    def _failure(
        self,
        *,
        event_type: str,
        stage: str,
        task_id: str,
        error_code: str,
        message: object,
        retryable: bool,
        terminal: bool,
        source: SourceRecord | None = None,
        group_id: str | None = None,
        candidate_id: str | None = None,
        round_number: int | None = None,
        attempt: int | None = None,
        endpoint_id: str | None = None,
        durable: bool = False,
    ) -> bool:
        source_id = source.source_id if source is not None else None
        event_id = stable_id(
            "failure", self.config.build_id, event_type, stage, task_id, error_code,
            round_number, attempt, group_id, candidate_id, endpoint_id,
        )
        return self.store.append_failure({
            "build_id": self.config.build_id,
            "event_id": event_id,
            "event_type": event_type,
            "stage": stage,
            "task_id": task_id,
            "source_id": source_id,
            "source_path": str(source.source_path) if source is not None else None,
            "group_id": group_id,
            "candidate_id": candidate_id,
            "round": round_number,
            "attempt": attempt,
            "retryable": retryable,
            "error_code": error_code,
            "message": redact_text(message, self.config.secrets),
            "endpoint_id": endpoint_id,
            "terminal": terminal,
            "timestamp": _timestamp(self.dependencies.now()),
        }, durable=durable)

    def _mode_groups(self, mode: str) -> list[dict[str, Any]]:
        return [
            row for row in self.store.groups.values()
            if row.get("render_mode") == mode
        ]

    def _terminal_source_ids(self) -> set[str]:
        return {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("terminal") and row.get("source_id")
            and row.get("stage") in {"rendering", "sam3_relabel"}
        }

    def _pending_sam3_ids(self) -> set[str]:
        queued = {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("error_code") == "sam3_relabel_queued" and row.get("source_id")
        }
        return queued.difference(self.store.completed_sources()).difference(
            self._terminal_source_ids()
        )

    def _sam3_attempts(self, source_id: str) -> int:
        return sum(
            1 for row in self.store.failures
            if row.get("source_id") == source_id
            and row.get("stage") == "sam3_relabel"
            and row.get("event_type") == "sam3_attempt"
        )

    def _unconsumed_sam3_ready(self, source_id: str) -> bool:
        ready = [
            int(row.get("attempt") or 0) for row in self.store.failures
            if row.get("source_id") == source_id and row.get("event_type") == "sam3_ready"
        ]
        invalid = [
            int(row.get("attempt") or 0) for row in self.store.failures
            if row.get("source_id") == source_id
            and row.get("event_type") == "sam3_ready_invalid"
        ]
        return bool(ready) and max(ready) > max(invalid or [0])

    def _manifest(self, phase: str, status: str = "running") -> dict[str, Any]:
        local_groups = self._mode_groups("local")
        global_groups = self._mode_groups("global")
        candidates = [
            candidate
            for group in self.store.groups.values()
            for candidate in group.get("candidates") or []
        ]
        terminal_failures = [row for row in self.store.failures if row.get("terminal")]
        annotation_failures = [
            row for row in terminal_failures if row.get("stage") == "annotation"
        ]
        preset_formats: dict[str, int] = {}
        preset_majors: dict[str, int] = {}
        preset_minors: dict[str, int] = {}
        for candidate in candidates:
            for target, key in (
                (preset_formats, "format"), (preset_majors, "major"),
                (preset_minors, "minor"),
            ):
                value = str(candidate.get(key) or "unknown")
                target[value] = target.get(value, 0) + 1
        completed = len(local_groups) + len(global_groups)
        local_initial = {row.source_id for row in self.allocation.local[:self.allocation.local_target]}
        global_initial = {
            row.source_id for row in self.allocation.global_[:self.allocation.global_target]
        }
        replacement_used = {
            "local": sum(group.get("source_id") not in local_initial for group in local_groups),
            "global": sum(group.get("source_id") not in global_initial for group in global_groups),
        }
        sam3_expected_ids = {
            str(row["source_id"])
            for row in self.store.failures
            if row.get("error_code") == "sam3_relabel_queued" and row.get("source_id")
        }
        sam3_completed_ids = {
            source_id for source_id in sam3_expected_ids
            if self._unconsumed_sam3_ready(source_id)
        }
        candidate_failures: dict[str, int] = {}
        for row in self.store.failures:
            if row.get("stage") == "rendering" and row.get("event_type") == "attempt":
                code = str(row.get("error_code") or "unknown")
                candidate_failures[code] = candidate_failures.get(code, 0) + 1
        ratio = {
            "local": len(local_groups) / completed if completed else 0.0,
            "global": len(global_groups) / completed if completed else 0.0,
        }
        return {
            "schema_version": self.config.schema_version,
            "build_id": self.config.build_id,
            "phase": phase,
            "status": status,
            "started_at": self.started_at,
            "updated_at": _timestamp(self.dependencies.now()),
            "effective_config": self.config.sanitized_dict(),
            "targets": {
                "groups": self.config.target_groups,
                "local": self.allocation.local_target,
                "global": self.allocation.global_target,
            },
            "completed": {
                "groups": completed,
                "local": len(local_groups),
                "global": len(global_groups),
                "actual_ratio": ratio,
                "candidates": len(candidates),
                "winner_top1": sum(bool(row.get("winner_ids")) for row in self.store.groups.values()),
                "winner_top2": sum(len(row.get("winner_ids") or []) >= 2
                                   for row in self.store.groups.values()),
                "sft": len(self.store.sft),
            },
            "sources": {
                "inventory": self.inventory.counts,
                "scene_metadata_status": self.inventory.scene_metadata_status,
                "eligible": len(self.inventory.eligible),
                "replacement_capacity": {
                    "local": max(0, len(self.allocation.local) - self.allocation.local_target),
                    "global": max(0, len(self.allocation.global_) - self.allocation.global_target),
                },
                "replacement_used": replacement_used,
                "exhaustion": {
                    "local_shortfall": max(0, self.allocation.local_target - len(local_groups)),
                    "global_shortfall": max(0, self.allocation.global_target - len(global_groups)),
                },
                "terminal": len(self._terminal_source_ids()),
            },
            "presets": {
                "inventory": {
                    "local": self.catalog.inventory_counts("local"),
                    "global": self.catalog.inventory_counts("global"),
                },
                "usage": {
                    "format": dict(sorted(preset_formats.items())),
                    "major": dict(sorted(preset_majors.items())),
                    "minor": dict(sorted(preset_minors.items())),
                    "selector": {
                        mode: selector.snapshot() if selector is not None else None
                        for mode, selector in self.selectors.items()
                    },
                    "candidate_failure_reasons": dict(sorted(candidate_failures.items())),
                },
            },
            "sam3_relabel": {
                "expected": len(sam3_expected_ids),
                "completed": len(sam3_completed_ids),
                "pending": len(self._pending_sam3_ids()),
                "attempt_events": sum(
                    row.get("event_type") == "sam3_attempt" for row in self.store.failures
                ),
                "terminal": sum(
                    row.get("stage") == "sam3_relabel" and row.get("terminal")
                    for row in self.store.failures
                ),
            },
            "annotation": {
                "pending": len(self.store.pending_annotation_tasks()),
                "terminal_failures": len(annotation_failures),
                "backends": self._annotation_counts(),
            },
            "failures": {
                "events": len(self.store.failures),
                "terminal": len(terminal_failures),
                "by_code": self._failure_counts(),
            },
        }

    def _failure_counts(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in self.store.failures:
            code = str(row.get("error_code") or "unknown")
            result[code] = result.get(code, 0) + 1
        return dict(sorted(result.items()))

    def _annotation_counts(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        models: dict[str, int] = {}
        usage: dict[str, int] = {}
        for row in self.store.sft.values():
            source = str(row.get("annot_src") or "unknown")
            sources[source] = sources.get(source, 0) + 1
            meta = (row.get("qa") or {}).get("annotation") or {}
            model = str(meta.get("returned_model") or "unknown")
            models[model] = models.get(model, 0) + 1
            for key, value in (meta.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        return {
            "sources": dict(sorted(sources.items())),
            "models": dict(sorted(models.items())),
            "usage": dict(sorted(usage.items())),
        }

    def _write_phase(self, phase: str) -> None:
        self.store.write_manifest(self._manifest(phase))
        self.store.checkpoint()

    def _queue_sam3(self, source: SourceRecord, error: MaskPlanError) -> None:
        task_id = stable_id("sam3", self.config.build_id, source.source_id)
        self._failure(
            event_type="queued",
            stage="sam3_relabel",
            task_id=task_id,
            error_code="sam3_relabel_queued",
            message=f"{error.code}: {error}",
            retryable=True,
            terminal=False,
            source=source,
            durable=True,
        )

    def _terminal_source(
        self, source: SourceRecord, stage: str, code: str, message: object
    ) -> None:
        self._failure(
            event_type="terminal",
            stage=stage,
            task_id=stable_id(stage, self.config.build_id, source.source_id),
            error_code=code,
            message=message,
            retryable=False,
            terminal=True,
            source=source,
            durable=True,
        )

    def _render_source(
        self,
        source: SourceRecord,
        mode: str,
        *,
        queue_mask_failure: bool = True,
    ) -> str:
        if self.renderer is None or self.scorer is None:
            raise PipelineError("render/QA resources are not loaded")
        selector = self.selectors[mode]
        if selector is None:
            raise PipelineError(f"missing {mode} selector")
        try:
            prepared = preprocess_source(source.source_path, self.config.render.short_edge)
        except Exception as exc:  # noqa: BLE001 - a corrupt source is replaced in its mode
            self._terminal_source(source, "rendering", "source_preprocess_failed", exc)
            return "terminal"

        slots: list[Any]
        if mode == "local":
            try:
                plan = build_mask_plan(
                    source,
                    build_id=self.config.build_id,
                    seed=self.config.seed,
                    width=prepared.width,
                    height=prepared.height,
                    linear_target=self.config.masks.linear_target_alpha_mass,
                )
                slots = list(pair_mask_slots(
                    plan,
                    build_id=self.config.build_id,
                    seed=self.config.seed,
                    source_id=source.source_id,
                ))
            except MaskPlanError as exc:
                if queue_mask_failure:
                    self._queue_sam3(source, exc)
                    return "sam3_queued"
                return f"mask_failed:{exc.code}:{exc}"
        else:
            slots = [f"global-{index}" for index in range(8)]

        excluded_majors: list[str] = []
        for group_attempt in range(len(selector.majors)):
            group_id = stable_id(
                "group", self.config.build_id, source.source_id, mode, group_attempt
            )
            reservation = None
            group_persisted = False
            try:
                reservation = selector.begin_group(
                    source.source_id, group_attempt, exclude_majors=excluded_majors
                )
                candidates: list[dict[str, Any]] = []
                exhausted = False
                for slot_index, slot in enumerate(slots):
                    slot_id = slot.slot_id if mode == "local" else str(slot)
                    while True:
                        preset_reservation = reservation.reserve_candidate(slot_id)
                        if preset_reservation is None:
                            exhausted = True
                            break
                        preset = preset_reservation.link.preset
                        attempt_task_id = stable_id(
                            "render-attempt", group_id, slot_id,
                            preset_reservation.attempt, preset.preset_id,
                        )
                        mask = slot.mask if mode == "local" else None
                        try:
                            rendered = self.renderer.render(prepared, preset, mask)
                            metrics = visibility_metrics(
                                prepared.pixels,
                                rendered.pixels,
                                weight=mask.effective_alpha if mask is not None else None,
                                short_edge=self.config.render.diff_short_edge,
                                visible_de_min=self.config.render.visible_de_min,
                                visible_fraction_de=self.config.render.visible_fraction_de,
                                visible_fraction_min=self.config.render.visible_fraction_min,
                            )
                            if not metrics.accepted:
                                raise PipelineError(
                                    "visibility gate rejected candidate "
                                    f"(de={metrics.visible_de:.6f}, "
                                    f"fraction={metrics.visible_fraction:.6f})"
                                )
                            candidate_id = stable_id("candidate", group_id, slot_id)
                            after_path = self.store.assets_root / "candidates" / f"{candidate_id}.jpg"
                            save_candidate_jpeg(
                                rendered.pixels, after_path,
                                quality=self.config.render.jpeg_quality,
                            )
                            cgt_path: Path | None = None
                            if mask is not None:
                                cgt_path = self.store.assets_root / "masks" / f"{mask.mask_id}.png"
                                save_cgt_png(mask, cgt_path)
                            hints = objective_edit_hints(
                                prepared.pixels,
                                rendered.pixels,
                                weight=mask.effective_alpha if mask is not None else None,
                            )
                            recipe = {
                                "preset_id": preset.preset_id,
                                "preset_path": str(preset.path),
                                "format": preset.format,
                                "render_engine": rendered.engine,
                                "render_mode": mode,
                            }
                            candidate: dict[str, Any] = {
                                "candidate_id": candidate_id,
                                "slot_id": slot_id,
                                "slot_index": slot_index,
                                "preset_id": preset.preset_id,
                                "preset_path": str(preset.path),
                                "format": preset.format,
                                "kind": preset.kind,
                                "style_name": preset.style_name,
                                "major": preset_reservation.link.major,
                                "minor": preset_reservation.link.minor,
                                "after_path": str(after_path),
                                "render_engine": rendered.engine,
                                "render_diagnostics": rendered.diagnostics,
                                "visibility": {
                                    "visible_de": round(metrics.visible_de, 6),
                                    "visible_fraction": round(metrics.visible_fraction, 6),
                                    "accepted": True,
                                },
                                "objective_hints": hints,
                                "attempt_lineage": {
                                    "group_attempt": group_attempt,
                                    "preset_attempt": preset_reservation.attempt,
                                    "preset_reservation_id": preset_reservation.reservation_id,
                                },
                                "recipe": recipe,
                            }
                            if mask is not None:
                                recipe.update({"mask_id": mask.mask_id, "amount": mask.amount})
                                candidate.update({
                                    "slot_mode": slot.mode,
                                    "mode_index": slot.mode_index,
                                    "pairing_index": slot.pairing_index,
                                    "mask_id": mask.mask_id,
                                    "cgt_path": str(cgt_path),
                                    "subject": source.subject,
                                    "region": mask.region,
                                    "geometry": mask.geometry,
                                    "raw_alpha_mean": round(mask.raw_alpha_mean, 8),
                                    "amount": round(mask.amount, 8),
                                    "effective_alpha_mean": round(mask.effective_alpha_mean, 8),
                                })
                            reservation.accept(preset_reservation)
                            candidates.append(candidate)
                            break
                        except Exception as exc:  # noqa: BLE001 - refill this exact slot
                            code = getattr(exc, "code", None) or (
                                "visibility_rejected"
                                if isinstance(exc, PipelineError) else "candidate_render_failed"
                            )
                            self._failure(
                                event_type="attempt",
                                stage="rendering",
                                task_id=attempt_task_id,
                                error_code=str(code),
                                message=exc,
                                retryable=True,
                                terminal=False,
                                source=source,
                                group_id=group_id,
                                round_number=group_attempt,
                                attempt=preset_reservation.attempt,
                            )
                            reservation.reject(preset_reservation)
                    if exhausted:
                        break
                if exhausted or len(candidates) != 8:
                    raise PresetError("major could not yield eight visible candidates")
                ranked = rank_candidates(str(source.source_path), candidates, self.scorer)
                coverage = {
                    "major": reservation.major,
                    "coverage_cycle": reservation.coverage_cycle,
                    "coverage_position": reservation.coverage_position,
                    "reservation_id": reservation.reservation_id,
                }
                completed_at = _timestamp(self.dependencies.now())
                by_id = {row["candidate_id"]: row for row in ranked.candidates}
                group = {
                    "schema_version": 1,
                    "build_id": self.config.build_id,
                    "group_id": group_id,
                    "source_id": source.source_id,
                    "source_path": str(source.source_path),
                    "scene": source.scene,
                    "subject": source.subject,
                    "render_mode": mode,
                    "preset_filter": self.config.preset_filter,
                    "group_attempt": group_attempt,
                    **coverage,
                    "candidates": list(ranked.candidates),
                    "winner_ids": list(ranked.winner_ids),
                    "winner_ranks": [by_id[candidate_id]["rank"]
                                     for candidate_id in ranked.winner_ids],
                    "source_onealign": ranked.source_score,
                    "stage_timestamps": {
                        "render_completed_at": completed_at,
                        "qa_completed_at": completed_at,
                    },
                }
                if not self.store.append_group(group):
                    raise StateError(f"unexpected existing group during render: {group_id}")
                self.store.checkpoint()
                group_persisted = True
                committed = reservation.commit()
                if committed != coverage:
                    raise StateError("coverage commit metadata changed after durable group append")
                return "completed"
            except QaError:
                raise
            except Exception as exc:  # noqa: BLE001 - restart group in another major
                if group_persisted:
                    raise PipelineError(
                        f"durable group {group_id} could not commit in-memory coverage"
                    ) from exc
                major = reservation.major if reservation is not None else "unknown"
                if reservation is not None:
                    reservation.abandon()
                excluded_majors.append(major)
                self._failure(
                    event_type="group_attempt",
                    stage="rendering",
                    task_id=stable_id("render-group", group_id),
                    error_code="major_exhausted",
                    message=f"{major}: {type(exc).__name__}: {exc}",
                    retryable=True,
                    terminal=False,
                    source=source,
                    group_id=group_id,
                    round_number=group_attempt,
                    attempt=group_attempt + 1,
                )
        self._terminal_source(
            source, "rendering", "preset_inventory_exhausted",
            f"no taxonomy major yielded eight accepted {mode} candidates",
        )
        return "terminal"

    def _fill_initial_mode(self, mode: str, sources: tuple[SourceRecord, ...], target: int) -> None:
        terminal = self._terminal_source_ids()
        pending = self._pending_sam3_ids() if mode == "local" else set()
        for source in sources:
            completed = len(self._mode_groups(mode))
            reserved = len(pending) if mode == "local" else 0
            if completed + reserved >= target:
                break
            if source.source_id in self.store.completed_sources() \
                    or source.source_id in terminal or source.source_id in pending:
                continue
            result = self._render_source(source, mode)
            if result == "sam3_queued":
                pending.add(source.source_id)
            elif result == "terminal":
                terminal.add(source.source_id)

    def _release_heavy_resources(self) -> None:
        self.renderer = None
        self.scorer = None
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _restore_heavy_resources(self) -> None:
        renderer = self.dependencies.renderer_factory(self.config)
        renderer.bind_catalog(self.catalog)
        renderer.assert_ready()
        self.renderer = renderer
        self.scorer = self.dependencies.scorer_factory()
        _preflight_scorer(self.scorer, self.inventory)

    def _record_sam3_attempt(
        self,
        source: SourceRecord,
        attempt: int,
        code: str,
        message: object,
        *,
        event_type: str = "sam3_attempt",
    ) -> None:
        self._failure(
            event_type=event_type,
            stage="sam3_relabel",
            task_id=stable_id("sam3", self.config.build_id, source.source_id),
            error_code=code,
            message=message,
            retryable=True,
            terminal=False,
            source=source,
            round_number=attempt,
            attempt=attempt,
            durable=True,
        )

    def _terminal_sam3(self, source: SourceRecord, message: object) -> None:
        self._terminal_source(source, "sam3_relabel", "sam3_relabel_failed", message)

    def _try_ready_relabel(self, source: SourceRecord, attempt: int) -> str:
        refreshed, reason = refresh_source_record(source)
        if refreshed is None:
            self._record_sam3_attempt(
                source, attempt, "sam3_integrity_failed", reason,
                event_type="sam3_ready_invalid",
            )
            return "retry"
        result = self._render_source(refreshed, "local", queue_mask_failure=False)
        if result == "completed":
            return "completed"
        if result.startswith("mask_failed:"):
            self._record_sam3_attempt(
                source, attempt, "sam3_geometry_failed", result,
                event_type="sam3_ready_invalid",
            )
            return "retry"
        return "terminal"

    def _drain_sam3_and_replacements(self) -> None:
        by_id = {source.source_id: source for source in self.allocation.local}
        target = self.allocation.local_target
        max_attempts = self.config.masks.sam3_relabel_attempts
        while len(self._mode_groups("local")) < target:
            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]

            made_progress = False
            for source in list(pending):
                if not self._unconsumed_sam3_ready(source.source_id):
                    continue
                attempt = self._sam3_attempts(source.source_id)
                status = self._try_ready_relabel(source, attempt)
                made_progress = True
                if status == "retry" and attempt >= max_attempts:
                    self._terminal_sam3(source, "ready relabel still violates canonical geometry")

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            exhausted = [
                source for source in pending
                if self._sam3_attempts(source.source_id) >= max_attempts
                and not self._unconsumed_sam3_ready(source.source_id)
            ]
            for source in exhausted:
                self._terminal_sam3(source, "SAM3 relabel attempt budget exhausted")
                made_progress = True

            pending_ids = self._pending_sam3_ids()
            pending = [by_id[source_id] for source_id in sorted(pending_ids) if source_id in by_id]
            candidates = [
                source for source in pending
                if self._sam3_attempts(source.source_id) < max_attempts
                and not self._unconsumed_sam3_ready(source.source_id)
            ]
            if candidates:
                next_attempt = min(self._sam3_attempts(row.source_id) + 1 for row in candidates)
                batch = [
                    row for row in candidates
                    if self._sam3_attempts(row.source_id) + 1 == next_attempt
                ]
                self._release_heavy_resources()
                try:
                    try:
                        statuses = self.dependencies.relabeler(batch, self.config, next_attempt)
                    except Exception as exc:  # noqa: BLE001 - each source consumes one bounded attempt
                        statuses = {source.source_id: f"relabel_exception:{type(exc).__name__}:{exc}"
                                    for source in batch}
                finally:
                    self._restore_heavy_resources()
                for source in batch:
                    status = str(statuses.get(source.source_id) or "missing_relabel_status")
                    self._record_sam3_attempt(
                        source, next_attempt, f"sam3_{status}", status
                    )
                    refreshed, reason = refresh_source_record(source)
                    if refreshed is None:
                        if next_attempt >= max_attempts:
                            self._terminal_sam3(source, f"{status}; integrity={reason}")
                        continue
                    self._record_sam3_attempt(
                        source, next_attempt, "sam3_relabel_ready", status,
                        event_type="sam3_ready",
                    )
                    result = self._try_ready_relabel(source, next_attempt)
                    if result == "retry" and next_attempt >= max_attempts:
                        self._terminal_sam3(source, "relabel cannot satisfy canonical geometry")
                made_progress = True

            pending_count = len(self._pending_sam3_ids())
            before = len(self._mode_groups("local"))
            if before + pending_count < target:
                self._fill_initial_mode("local", self.allocation.local, target)
                made_progress = made_progress or len(self._mode_groups("local")) > before \
                    or len(self._pending_sam3_ids()) > pending_count
            if not made_progress:
                break

        if self._pending_sam3_ids():
            raise PipelineError("SAM3 relabel queue remains unresolved")

    def _record_shortfalls(self) -> None:
        for mode, target in (
            ("local", self.allocation.local_target),
            ("global", self.allocation.global_target),
        ):
            completed = len(self._mode_groups(mode))
            if completed >= target:
                continue
            task_id = stable_id("target", self.config.build_id, mode)
            self._failure(
                event_type="terminal",
                stage="rendering",
                task_id=task_id,
                error_code=f"{mode}_target_shortfall",
                message=f"requested {target} {mode} groups, completed {completed}",
                retryable=False,
                terminal=True,
                durable=True,
            )

    def execute(self) -> dict[str, Any]:
        self._write_phase("preflight")
        self._write_phase("rendering")
        self._fill_initial_mode(
            "global", self.allocation.global_, self.allocation.global_target
        )
        self._fill_initial_mode(
            "local", self.allocation.local, self.allocation.local_target
        )

        self._write_phase("sam3_relabel")
        self._drain_sam3_and_replacements()
        self._record_shortfalls()

        self._release_heavy_resources()
        self._write_phase("annotation")
        annotation = self.dependencies.annotator_factory(self.config, self.store).drain()
        if annotation.get("pending") or self.store.pending_annotation_tasks():
            self._write_phase("annotation")
            raise PipelineError("annotation queue remains unresolved")

        self._write_phase("projection")
        terminal = any(row.get("terminal") for row in self.store.failures)
        final_status = "complete_with_failures" if terminal else "complete"
        final_manifest = self._manifest(final_status, final_status)
        final_manifest["ended_at"] = _timestamp(self.dependencies.now())
        projection = self.dependencies.projector(self.store, final_manifest, self.config)
        final_manifest["projection"] = dataclasses.asdict(projection)
        self.store.checkpoint()
        final_manifest["artifacts"] = {
            path.name: file_digest(path)
            for path in (self.store.groups_path, self.store.sft_path, self.store.failures_path)
            if path.exists()
        }
        _add_manifest_self_artifact(final_manifest)
        self.store.write_manifest(final_manifest)
        self.store.checkpoint()
        return final_manifest


class LegacyImportPipeline:
    """Import an authorized legacy snapshot, then reuse the canonical Responses queue."""

    def __init__(
        self,
        config: DatabuildConfig,
        dependencies: PipelineDependencies,
        snapshot: LegacySnapshot,
        store: ArtifactStore,
        existing_manifest: Mapping[str, Any] | None,
    ) -> None:
        self.config = config
        self.dependencies = dependencies
        self.snapshot = snapshot
        self.store = store
        self.started_at = str(
            (existing_manifest or {}).get("started_at")
            or _timestamp(self.dependencies.now())
        )

    def _failure(
        self,
        group: LegacyGroupInput | None,
        *,
        code: str,
        message: object,
    ) -> None:
        source_asset_id = group.source_asset_id if group is not None else ""
        group_id = legacy_group_id(self.config, group) if group is not None else None
        source_id = (
            stable_id(
                "source", self.config.legacy_import.protocol, source_asset_id
            )
            if group is not None else None
        )
        task_id = stable_id("legacy-import", self.config.build_id, source_asset_id or code)
        if self.store.has_terminal_failure(task_id):
            return
        self.store.append_failure({
            "build_id": self.config.build_id,
            "event_id": stable_id("failure", self.config.build_id, task_id, code),
            "event_type": "terminal",
            "stage": "import",
            "task_id": task_id,
            "source_id": source_id,
            "source_path": (
                str(group.record.get("source")) if group is not None else None
            ),
            "group_id": group_id,
            "candidate_id": None,
            "round": None,
            "attempt": 1,
            "retryable": False,
            "error_code": code,
            "message": redact_text(message, self.config.secrets),
            "endpoint_id": None,
            "terminal": True,
            "timestamp": _timestamp(self.dependencies.now()),
        }, durable=True)

    def _annotation_counts(self) -> dict[str, Any]:
        sources: dict[str, int] = {}
        models: dict[str, int] = {}
        usage: dict[str, int] = {}
        for row in self.store.sft.values():
            source = str(row.get("annot_src") or "unknown")
            sources[source] = sources.get(source, 0) + 1
            meta = (row.get("qa") or {}).get("annotation") or {}
            model = str(meta.get("returned_model") or "unknown")
            models[model] = models.get(model, 0) + 1
            for key, value in (meta.get("usage") or {}).items():
                if isinstance(value, int):
                    usage[key] = usage.get(key, 0) + value
        return {
            "sources": dict(sorted(sources.items())),
            "models": dict(sorted(models.items())),
            "usage": dict(sorted(usage.items())),
        }

    def _manifest(self, phase: str, status: str = "running") -> dict[str, Any]:
        groups = list(self.store.groups.values())
        candidates = [candidate for group in groups for candidate in group["candidates"]]
        terminal = [row for row in self.store.failures if row.get("terminal")]
        by_code: dict[str, int] = {}
        for row in self.store.failures:
            code = str(row.get("error_code") or "unknown")
            by_code[code] = by_code.get(code, 0) + 1
        formats: dict[str, int] = {}
        modes: dict[str, int] = {}
        for candidate in candidates:
            fmt = str(candidate.get("format") or "unknown")
            mode = str(candidate.get("slot_mode") or "unknown")
            formats[fmt] = formats.get(fmt, 0) + 1
            modes[mode] = modes.get(mode, 0) + 1
        return {
            "schema_version": self.config.schema_version,
            "build_id": self.config.build_id,
            "phase": phase,
            "status": status,
            "started_at": self.started_at,
            "updated_at": _timestamp(self.dependencies.now()),
            "effective_config": self.config.sanitized_dict(),
            "targets": {
                "groups": self.config.target_groups,
                "local": self.config.target_groups,
                "global": 0,
                "sft": self.snapshot.sft_row_count,
            },
            "completed": {
                "groups": len(groups),
                "local": len(groups),
                "global": 0,
                "actual_ratio": {"local": 1.0 if groups else 0.0, "global": 0.0},
                "candidates": len(candidates),
                "winner_top1": sum(bool(group.get("winner_ids")) for group in groups),
                "winner_top2": sum(len(group.get("winner_ids") or []) >= 2 for group in groups),
                "sft": len(self.store.sft),
            },
            "sources": {
                "inventory": {
                    "legacy_source_groups": self.snapshot.source_group_count,
                    "groups_with_winners": len(self.snapshot.groups),
                    "groups_without_winners": self.snapshot.skipped_groups_without_winners,
                },
                "scene_metadata_status": "legacy_not_available",
                "eligible": len(self.snapshot.groups),
                "replacement_capacity": {"local": 0, "global": 0},
                "replacement_used": {"local": 0, "global": 0},
                "exhaustion": {
                    "local_shortfall": max(0, self.config.target_groups - len(groups)),
                    "global_shortfall": 0,
                },
                "terminal": sum(row.get("stage") == "import" for row in terminal),
            },
            "presets": {
                "inventory": {"local": {"xmp": len(candidates)}, "global": {}},
                "usage": {
                    "format": dict(sorted(formats.items())),
                    "major": {},
                    "minor": dict(sorted(modes.items())),
                    "selector": {"local": None, "global": None},
                    "candidate_failure_reasons": {},
                },
            },
            "sam3_relabel": {
                "expected": 0, "completed": 0, "pending": 0,
                "attempt_events": 0, "terminal": 0,
            },
            "annotation": {
                "pending": len(self.store.pending_annotation_tasks()),
                "terminal_failures": sum(
                    row.get("stage") == "annotation" for row in terminal
                ),
                "backends": self._annotation_counts(),
            },
            "failures": {
                "events": len(self.store.failures),
                "terminal": len(terminal),
                "by_code": dict(sorted(by_code.items())),
            },
            "legacy_import": {
                "protocol": self.config.legacy_import.protocol,
                "axis_fix_status": self.config.legacy_import.axis_fix_status,
                "input_root": str(self.config.legacy_import.input_root),
                "input_artifacts": self.snapshot.input_artifacts,
                "selection": "legacy winner set from unannotated sft.jsonl",
                "old_text_imported": False,
                "assets": "hardlink-or-copy into output assets; source paths remain external",
                "canonical_mask_contract": False,
                "viewer_projection_enabled": self.config.legacy_import.project_to_viewer,
            },
        }

    def _write_phase(self, phase: str) -> None:
        self.store.write_manifest(self._manifest(phase))
        self.store.checkpoint()

    def _import_groups(self) -> None:
        for index, group in enumerate(self.snapshot.groups, start=1):
            group_id = legacy_group_id(self.config, group)
            if group_id in self.store.groups:
                continue
            try:
                record = convert_legacy_group(
                    self.config,
                    group,
                    self.store,
                    timestamp=_timestamp(self.dependencies.now()),
                )
                self.store.append_group(record)
            except Exception as exc:  # noqa: BLE001 - preserve one structured import failure
                self._failure(
                    group,
                    code="legacy_group_import_failed",
                    message=f"{type(exc).__name__}: {exc}",
                )
            if index % 32 == 0:
                self.store.checkpoint()
        self.store.checkpoint()

    def execute(self) -> dict[str, Any]:
        self._write_phase("preflight")
        self._write_phase("import")
        self._import_groups()

        self._write_phase("annotation")
        annotation = self.dependencies.annotator_factory(self.config, self.store).drain()
        if annotation.get("pending") or self.store.pending_annotation_tasks():
            self._write_phase("annotation")
            raise PipelineError("legacy annotation queue remains unresolved")

        if len(self.store.groups) != self.config.target_groups \
                or len(self.store.sft) != self.snapshot.sft_row_count:
            self._failure(
                None,
                code="legacy_migration_target_shortfall",
                message=(
                    f"expected groups={self.config.target_groups}, sft={self.snapshot.sft_row_count}; "
                    f"completed groups={len(self.store.groups)}, sft={len(self.store.sft)}"
                ),
            )

        self._write_phase("projection")
        terminal = any(row.get("terminal") for row in self.store.failures)
        final_status = "complete_with_failures" if terminal else "complete"
        final_manifest = self._manifest(final_status, final_status)
        final_manifest["ended_at"] = _timestamp(self.dependencies.now())
        if self.config.legacy_import.project_to_viewer:
            projection = self.dependencies.projector(self.store, final_manifest, self.config)
        else:
            projection = ProjectionResult(True, {})
        final_manifest["projection"] = dataclasses.asdict(projection)
        self.store.checkpoint()
        final_manifest["artifacts"] = {
            path.name: file_digest(path)
            for path in (self.store.groups_path, self.store.sft_path, self.store.failures_path)
            if path.exists()
        }
        _add_manifest_self_artifact(final_manifest)
        self.store.write_manifest(final_manifest)
        self.store.checkpoint()
        return final_manifest


def run(
    config: DatabuildConfig,
    *,
    dependencies: PipelineDependencies | None = None,
) -> dict[str, Any]:
    """Preflight fully, then run or resume one canonical build."""
    dependencies = dependencies or default_dependencies()
    existing = _load_existing_manifest(config.output_root / "manifest.json")
    if existing is not None:
        if existing.get("build_id") != config.build_id:
            raise StateError("output_root belongs to a different build_id")
        old_config = existing.get("effective_config")
        if old_config is not None and old_config != config.sanitized_dict():
            raise StateError("resume config differs from the durable manifest")

    # No authoritative artifact is opened until every startup dependency passes.
    dependencies.sdk_preflight()
    if config.legacy_import is not None:
        snapshot = load_legacy_snapshot(config)
        with ArtifactStore(config.output_root, config.build_id) as store:
            pipeline = LegacyImportPipeline(
                config, dependencies, snapshot, store, existing
            )
            return pipeline.execute()

    catalog = dependencies.catalog_loader(config)
    inventory = dependencies.inventory_loader(config)
    renderer = dependencies.renderer_factory(config)
    renderer.bind_catalog(catalog)
    renderer.assert_ready()
    scorer = dependencies.scorer_factory()
    _preflight_scorer(scorer, inventory)

    with ArtifactStore(config.output_root, config.build_id) as store:
        pipeline = CanonicalPipeline(
            config, dependencies, inventory, catalog, renderer, scorer, store, existing
        )
        return pipeline.execute()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m construct.agent",
        description="Canonical GPU-only SFT databuild",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    config: DatabuildConfig | None = None
    try:
        config = load_config(args.config)
        manifest = run(config)
    except Exception as exc:  # noqa: BLE001 - CLI boundary must redact every failure
        secrets = config.secrets if config is not None else ()
        print(
            "canonical databuild failed: "
            + redact_text(f"{type(exc).__name__}: {exc}", secrets),
            file=sys.stderr,
        )
        return 1
    print(json.dumps({
        "build_id": manifest["build_id"],
        "status": manifest["status"],
        "groups": manifest["completed"]["groups"],
        "sft": manifest["completed"]["sft"],
        "manifest": str(config.output_root / "manifest.json"),
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
