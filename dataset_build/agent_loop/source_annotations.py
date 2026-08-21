"""Offline source diagnosis records consumed by the online edit loop."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

from .artifacts import ArtifactStore
from .candidates import LutCatalog, palette_summary
from .config import AgentLoopConfig
from .histogram_board import BOARD_REVISION, board_png
from .persistence import AuditStore
from .prompts import DIAGNOSE_PROMPT_REVISION, diagnosis_request, semantic_error
from .responses import CachedResponsesClient
from .scheduler import TerraLimiter, TerraRouter
from .source_reach import (
    configured_lut_loader, probe_preset_reach, validate_preset_reach,
)


SOURCE_ANNOTATION_SCHEMA = "local-retouch-source-annotation-v2"


def source_content_hash(path: str | Path) -> str:
    target = Path(path)
    digest = hashlib.sha256()
    if target.is_file():
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    else:
        from dataset_build.tools.archive_reader import read_bytes

        digest.update(read_bytes(path))
    return digest.hexdigest()


def validate_source_annotation(
    annotation: Any, source: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(annotation, Mapping) \
            or annotation.get("schema") != SOURCE_ANNOTATION_SCHEMA:
        raise ValueError("invalid offline source annotation schema")
    row = dict(annotation)
    if str(row.get("source_id")) != str(source["source_id"]):
        raise ValueError("offline source annotation ID mismatch")
    if str(row.get("source_path")) != str(source["source_path"]):
        raise ValueError("offline source annotation path mismatch")
    source_sha256 = str(
        source.get("source_sha256") or source_content_hash(str(source["source_path"]))
    )
    if str(row.get("source_sha256")) != source_sha256:
        raise ValueError("offline source annotation hash mismatch")
    subject_path = str(source["subject_path"])
    if str(row.get("subject_path")) != subject_path:
        raise ValueError("offline source annotation subject path mismatch")
    subject_sha256 = str(
        source.get("subject_sha256") or source_content_hash(subject_path)
    )
    if str(row.get("subject_sha256")) != subject_sha256:
        raise ValueError("offline source annotation subject hash mismatch")
    diagnosis = row.get("diagnosis")
    error = semantic_error("diagnose", diagnosis)
    if error is not None:
        raise ValueError(f"invalid offline source diagnosis ({error})")
    validate_preset_reach(row.get("preset_reach"))
    provenance = row.get("provenance")
    if not isinstance(provenance, Mapping) \
            or provenance.get("reasoning_effort") != "high":
        raise ValueError("offline source diagnosis was not generated at high effort")
    return row


def load_source_annotation(
    path: str | Path, source: Mapping[str, Any]
) -> dict[str, Any]:
    target = Path(path).expanduser().resolve()
    try:
        row = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load source annotation: {target}") from exc
    try:
        return validate_source_annotation(row, source)
    except ValueError as exc:
        raise ValueError(f"{exc}: {target}") from exc


def annotate_source(
    config: AgentLoopConfig,
    artifacts: ArtifactStore,
    audit: AuditStore,
    terra: CachedResponsesClient,
    limiter: TerraLimiter,
    catalog: LutCatalog,
    source: Mapping[str, Any],
    router: TerraRouter | None = None,
) -> dict[str, Any]:
    source_path = str(source["source_path"])
    subject_path = str(source["subject_path"])
    source_sha256 = source_content_hash(source_path)
    subject_sha256 = source_content_hash(subject_path)
    source_artifact = artifacts.normalize_image(source_path, retention="audit")
    # E4: the second diagnosis image. The board is computed on the original file, not
    # on the 512px preview, and is stored as-is because the transport sends its bytes
    # untouched.
    board_artifact = artifacts.put_bytes(
        board_png(source_path), media_type="image/png", retention="audit",
    )
    endpoint = config.terra
    if router is not None:
        lane = router.lane_for(source_sha256)
        endpoint, terra, limiter = lane.endpoint, lane.client, lane.limiter
    diagnosis_endpoint = replace(
        endpoint,
        reasoning_effort=config.source_annotation.reasoning_effort,
    )
    spec = diagnosis_request(
        diagnosis_endpoint, source_artifact.to_dict(), board_artifact.to_dict()
    )
    with limiter.slot(source_sha256, "offline_diagnose", spec.prompt_cache_key):
        result = terra.request(spec)
    diagnosis = dict(result.response["parsed"])
    error = semantic_error("diagnose", diagnosis)
    if error is not None:
        raise ValueError(f"invalid offline source diagnosis: {error}")
    audit.record_request_context(
        result.request_hash, config.campaign_id, source_sha256,
        "offline_diagnose", bool(result.cache_hit),
    )
    palette = palette_summary(artifacts.path_for(source_artifact))
    reach_records = catalog.reach_candidates(
        diagnosis, palette, str(source.get("scene") or "unknown"),
        config.catalog.reach_limit,
    )
    preset_reach = probe_preset_reach(
        artifacts.path_for(source_artifact), source_sha256, reach_records,
        loader=configured_lut_loader(config.databuild_config).load,
    )
    return {
        "schema": SOURCE_ANNOTATION_SCHEMA,
        "source_id": str(source["source_id"]),
        "source_sha256": source_sha256,
        "source_path": source_path,
        "subject_path": subject_path,
        "subject_sha256": subject_sha256,
        "scene": str(source.get("scene") or "unknown"),
        "subject": dict(source.get("subject") or {}),
        "diagnosis": diagnosis,
        "preset_reach": preset_reach,
        "provenance": {
            "stage": "offline_diagnose",
            "model": diagnosis_endpoint.model,
            "prompt_revision": config.prompt_revision,
            "reasoning_effort": diagnosis_endpoint.reasoning_effort,
            "endpoint_identity": diagnosis_endpoint.identity,
            "diagnose_prompt_revision": DIAGNOSE_PROMPT_REVISION,
            "board_revision": BOARD_REVISION,
            "board_sha256": board_artifact.sha256,
            "request_hash": result.request_hash,
            "cache_hit": bool(result.cache_hit),
            "usage": dict(result.usage),
        },
    }


__all__ = [
    "SOURCE_ANNOTATION_SCHEMA", "annotate_source", "load_source_annotation",
    "source_content_hash", "validate_source_annotation",
]
