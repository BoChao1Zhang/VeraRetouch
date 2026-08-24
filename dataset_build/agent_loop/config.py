"""Strict configuration for the isolated local-retouch agent loop."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


class ConfigError(ValueError):
    """The agent-loop configuration is unsafe or inconsistent."""


@dataclass(frozen=True, slots=True)
class EndpointConfig:
    identity: str
    base_url: str
    api_key: str = field(repr=False)
    model: str = ""
    temperature: float = 0.1
    max_output_tokens: int = 1024
    timeout_seconds: float = 120.0
    attempts: int = 3
    reasoning_effort: str | None = None
    allow_uncached_provider: bool = False
    provider_kind: str = "hosted"
    # 2026-08-24: "stream" = SSE (default), "nonstream" = one blocking
    # responses.create call consumed by `consume_response`.
    transport: str = "stream"


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    backend: str
    dsn_env: str | None
    sqlite_path: Path | None
    required_tablespace: str | None
    minimum_free_bytes: int


@dataclass(frozen=True, slots=True)
class ArtifactLandingConfig:
    enabled: bool
    archive_root: Path | None
    archive_group: str | None
    plan_root: Path | None
    meta_staging: Path | None
    catalog_db: Path | None
    watermark_bytes: int = 8 * 1024**3
    min_interval_seconds: int = 300
    min_groups: int = 25
    free_bytes_floor: int = 4 * 1024**3


@dataclass(frozen=True, slots=True)
class CatalogConfig:
    annotations: Path
    global_major_limit: int
    global_per_major_limit: int
    local_limit: int
    reach_limit: int = 300
    cluster_artifact: Path | None = None
    # B10 (R6.1): optional derived segment-fingerprint artifact. Unset = not mounted,
    # and the catalog behaves exactly as before.
    segment_fingerprints: Path | None = None


@dataclass(frozen=True, slots=True)
class SourceAnnotationConfig:
    reasoning_effort: str


@dataclass(frozen=True, slots=True)
class RenderConfig:
    backend: str
    renderer_revision: str
    search_steps: int
    clip_fraction_max: float


@dataclass(frozen=True, slots=True)
class AgentLoopConfig:
    path: Path
    campaign_id: str
    prompt_revision: str
    databuild_config: Path
    artifact_root: Path
    artifact_landing: ArtifactLandingConfig
    checkpoint: CheckpointConfig
    catalog: CatalogConfig
    source_annotation: SourceAnnotationConfig
    terra_lanes: tuple[EndpointConfig, ...]
    validator: EndpointConfig
    validator_enabled: bool
    render: RenderConfig
    terra_concurrency_target: int
    renderer_concurrency: int
    validator_concurrency: int
    max_global_proposals: int
    min_global_proposals: int
    max_local_proposals: int
    max_local_repairs: int
    min_committed_leaves: int
    max_committed_leaves: int
    rejected_asset_ttl_days: int
    request_lease_seconds: int
    keep_audit_forever: bool

    @property
    def terra(self) -> EndpointConfig:
        """Lane 0. Kept so single-lane call sites and hashes stay unchanged."""
        return self.terra_lanes[0]

    @property
    def terra_total_concurrency(self) -> int:
        """Per-lane target times the number of lanes."""
        return self.terra_concurrency_target * len(self.terra_lanes)

    def terra_lane_index(self, source_key: str) -> int:
        from .scheduler import route_lane_index

        return route_lane_index(str(source_key), len(self.terra_lanes))

    def terra_lane(self, source_key: str) -> EndpointConfig:
        return self.terra_lanes[self.terra_lane_index(source_key)]

    @property
    def thread_revision(self) -> str:
        from .prompts import prompt_revision_fingerprint

        payload = {
            "prompt_revision": self.prompt_revision,
            "prompt_registry_sha256": prompt_revision_fingerprint(),
            "terra_model": self.terra.model,
            "validator_enabled": self.validator_enabled,
            "validator_model": self.validator.model if self.validator_enabled else None,
            "render_revision": self.render.renderer_revision,
            "graph_contract": {
                "max_global_proposals": self.max_global_proposals,
                "min_global_proposals": self.min_global_proposals,
                "max_local_proposals": self.max_local_proposals,
                "max_local_repairs": self.max_local_repairs,
                "min_committed_leaves": self.min_committed_leaves,
                "max_committed_leaves": self.max_committed_leaves,
            },
            "catalog_contract": {
                "global_major_limit": self.catalog.global_major_limit,
                "global_per_major_limit": self.catalog.global_per_major_limit,
                "local_limit": self.catalog.local_limit,
                "reach_limit": self.catalog.reach_limit,
                "cluster_artifact": str(self.catalog.cluster_artifact)
                if self.catalog.cluster_artifact else None,
                # C1b item 2: the mounted segment-fingerprint artifact is the online
                # local retrieval's whole candidate universe, so two configs that mount
                # different tables are two different threads. `prompt_registry()`
                # carries the *frozen* table SHA; this carries the *configured* path,
                # and `runtime.require_frozen_segment_fingerprints` asserts at startup
                # that what was actually mounted hashes to the frozen SHA.
                "segment_fingerprints": str(self.catalog.segment_fingerprints)
                if self.catalog.segment_fingerprints else None,
            },
        }
        digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()[:12]
        return f"{self.prompt_revision}-{digest}"

    def thread_id(
        self, source_sha256: str, subject_sha256: str | None = None, pass_index: int = 0
    ) -> str:
        subject_identity = subject_sha256 or "subject-unfrozen"
        return (
            f"{self.campaign_id}:pass-{int(pass_index)}:{source_sha256}:{subject_identity}:"
            f"{self.thread_revision}"
        )

    def checkpoint_dsn(self) -> str:
        if self.checkpoint.backend == "sqlite":
            if self.checkpoint.sqlite_path is None:
                raise ConfigError("checkpoint.sqlite_path is missing")
            return str(self.checkpoint.sqlite_path)
        name = self.checkpoint.dsn_env or ""
        value = os.environ.get(name)
        if not value:
            raise ConfigError(f"checkpoint DSN environment variable {name!r} is unset")
        return value

    def sanitized_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "prompt_revision": self.prompt_revision,
            "thread_revision": self.thread_revision,
            "databuild_config": str(self.databuild_config),
            "artifact_root": str(self.artifact_root),
            "artifact_landing": {
                **dataclasses.asdict(self.artifact_landing),
                "archive_root": str(self.artifact_landing.archive_root)
                if self.artifact_landing.archive_root else None,
                "plan_root": str(self.artifact_landing.plan_root)
                if self.artifact_landing.plan_root else None,
                "meta_staging": str(self.artifact_landing.meta_staging)
                if self.artifact_landing.meta_staging else None,
                "catalog_db": str(self.artifact_landing.catalog_db)
                if self.artifact_landing.catalog_db else None,
            },
            "checkpoint": {
                "backend": self.checkpoint.backend,
                "dsn_env": self.checkpoint.dsn_env,
                "sqlite_path": str(self.checkpoint.sqlite_path)
                if self.checkpoint.sqlite_path else None,
                "required_tablespace": self.checkpoint.required_tablespace,
                "minimum_free_bytes": self.checkpoint.minimum_free_bytes,
            },
            "terra": _safe_endpoint(self.terra),
            "terra_lanes": [_safe_endpoint(lane) for lane in self.terra_lanes],
            "source_annotation": dataclasses.asdict(self.source_annotation),
            "validator": {
                "enabled": self.validator_enabled,
                **_safe_endpoint(self.validator),
            },
            "limits": {
                "terra_concurrency_target": self.terra_concurrency_target,
                "terra_lane_count": len(self.terra_lanes),
                "terra_total_concurrency": self.terra_total_concurrency,
                "renderer_concurrency": self.renderer_concurrency,
                "validator_concurrency": self.validator_concurrency,
                "max_global_proposals": self.max_global_proposals,
                "min_global_proposals": self.min_global_proposals,
                "max_local_proposals": self.max_local_proposals,
                "max_local_repairs": self.max_local_repairs,
                "min_committed_leaves": self.min_committed_leaves,
                "max_committed_leaves": self.max_committed_leaves,
            },
            "catalog": {
                **dataclasses.asdict(self.catalog),
                "annotations": str(self.catalog.annotations),
                "cluster_artifact": str(self.catalog.cluster_artifact)
                if self.catalog.cluster_artifact else None,
                "segment_fingerprints": str(self.catalog.segment_fingerprints)
                if self.catalog.segment_fingerprints else None,
            },
            "render": dataclasses.asdict(self.render),
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _safe_endpoint(endpoint: EndpointConfig) -> dict[str, Any]:
    return {
        "identity": endpoint.identity,
        "model": endpoint.model,
        "temperature": endpoint.temperature,
        "max_output_tokens": endpoint.max_output_tokens,
        "timeout_seconds": endpoint.timeout_seconds,
        "attempts": endpoint.attempts,
        "reasoning_effort": endpoint.reasoning_effort,
        "allow_uncached_provider": endpoint.allow_uncached_provider,
        "provider_kind": endpoint.provider_kind,
        "transport": endpoint.transport,
        "base_url": "<redacted>",
        "api_key": "<redacted>",
    }


def _table(data: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = data.get(name)
    if not isinstance(value, Mapping):
        raise ConfigError(f"missing [{name}] table")
    return value


def _need(table: Mapping[str, Any], key: str, expected: type, where: str) -> Any:
    value = table.get(key)
    if not isinstance(value, expected) or isinstance(value, bool) != (expected is bool):
        raise ConfigError(f"{where}.{key} must be {expected.__name__}")
    return value


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_databuild_endpoint(path: Path, identity: str) -> tuple[str, str, str, str]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except OSError as exc:
        raise ConfigError(f"cannot read local databuild config: {path}") from exc
    annotation = _table(data, "annotation")
    if identity == "local":
        local = _table(annotation, "local")
        return (
            str(local.get("base_url") or ""),
            str(local.get("api_key") or ""),
            str(local.get("model") or ""),
            "local",
        )
    rows = annotation.get("external_endpoints")
    if not isinstance(rows, list):
        raise ConfigError("local databuild config has no external endpoint list")
    for row in rows:
        if isinstance(row, Mapping) and str(row.get("id")) == identity:
            return (
                str(row.get("base_url") or ""),
                str(row.get("api_key") or ""),
                str(annotation.get("external_model") or ""),
                "hosted",
            )
    raise ConfigError(f"endpoint {identity!r} is absent from {path}")


def _endpoint_identities(table: Mapping[str, Any], where: str) -> list[str]:
    """Accept either the single `endpoint` field or an `endpoints` lane list."""
    rows = table.get("endpoints")
    if rows is None:
        return [_need(table, "endpoint", str, where)]
    if table.get("endpoint") is not None:
        raise ConfigError(f"{where}: set either endpoint or endpoints, not both")
    if not isinstance(rows, list) or not rows:
        raise ConfigError(f"{where}.endpoints must be a non-empty list of endpoint ids")
    identities = []
    for row in rows:
        if not isinstance(row, str) or not row:
            raise ConfigError(f"{where}.endpoints entries must be non-empty strings")
        identities.append(row)
    if len(set(identities)) != len(identities):
        raise ConfigError(f"{where}.endpoints must be distinct")
    return identities


def _endpoint(
    table: Mapping[str, Any], *, databuild_path: Path, validator: bool = False,
    identity: str | None = None,
) -> EndpointConfig:
    if identity is None:
        identity = _need(table, "endpoint", str, "validator" if validator else "terra")
    base_url, api_key, inherited_model, provider_kind = _load_databuild_endpoint(
        databuild_path, identity
    )
    env_name = table.get("api_key_env")
    if env_name is not None:
        if not isinstance(env_name, str) or not env_name:
            raise ConfigError("api_key_env must be a non-empty string")
        api_key = os.environ.get(env_name, "")
    model = str(table.get("model") or inherited_model)
    if not base_url or not api_key or not model:
        raise ConfigError(f"endpoint {identity!r} is incomplete")
    effort = table.get("reasoning_effort")
    if effort is not None and not isinstance(effort, str):
        raise ConfigError("reasoning_effort must be a string")
    transport = table.get("transport", "stream")
    if transport not in {"stream", "nonstream"}:
        raise ConfigError("transport must be stream or nonstream")
    return EndpointConfig(
        identity=identity,
        base_url=base_url.rstrip("/"),
        api_key=api_key,
        model=model,
        temperature=float(table.get("temperature", 0.1)),
        max_output_tokens=int(table.get("max_output_tokens", 1024)),
        timeout_seconds=float(table.get("timeout_seconds", 120.0)),
        attempts=int(table.get("attempts", 3)),
        reasoning_effort=effort,
        allow_uncached_provider=bool(table.get("allow_uncached_provider", False)),
        provider_kind=provider_kind,
        transport=str(transport),
    )


def load_config(path: str | os.PathLike[str]) -> AgentLoopConfig:
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"cannot load agent-loop TOML: {config_path}") from exc
    root = config_path.parent
    agent = _table(data, "agent_loop")
    checkpoint_t = _table(data, "checkpoint")
    artifact_t = _table(data, "artifacts")
    catalog_t = _table(data, "catalog")
    source_annotation_t = _table(data, "source_annotation")
    render_t = _table(data, "render")
    retention_t = _table(data, "retention")
    databuild = _resolve(root, _need(agent, "databuild_config", str, "agent_loop"))
    backend = _need(checkpoint_t, "backend", str, "checkpoint")
    if backend not in {"postgres", "sqlite"}:
        raise ConfigError("checkpoint.backend must be postgres or sqlite")
    sqlite_value = checkpoint_t.get("sqlite_path")
    checkpoint = CheckpointConfig(
        backend=backend,
        dsn_env=str(checkpoint_t.get("dsn_env") or "") or None,
        sqlite_path=_resolve(root, str(sqlite_value)) if sqlite_value else None,
        required_tablespace=str(checkpoint_t.get("required_tablespace") or "") or None,
        minimum_free_bytes=int(checkpoint_t.get("minimum_free_bytes", 0)),
    )
    if backend == "postgres" and not checkpoint.dsn_env:
        raise ConfigError("postgres checkpoint requires dsn_env")
    if backend == "postgres" and not checkpoint.required_tablespace:
        raise ConfigError("postgres checkpoint requires required_tablespace")
    if checkpoint.minimum_free_bytes < 0:
        raise ConfigError("checkpoint.minimum_free_bytes must be non-negative")
    if backend == "sqlite" and checkpoint.sqlite_path is None:
        raise ConfigError("sqlite checkpoint requires sqlite_path")
    validator_t = _table(data, "validator")
    terra_t = _table(data, "terra")
    landing_enabled = bool(artifact_t.get("landing_enabled", False))

    def artifact_path(key: str) -> Path | None:
        value = artifact_t.get(key)
        return _resolve(root, str(value)) if value else None

    config = AgentLoopConfig(
        path=config_path,
        campaign_id=_need(agent, "campaign_id", str, "agent_loop"),
        prompt_revision=_need(agent, "prompt_revision", str, "agent_loop"),
        databuild_config=databuild,
        artifact_root=_resolve(root, _need(artifact_t, "root", str, "artifacts")),
        artifact_landing=ArtifactLandingConfig(
            enabled=landing_enabled,
            archive_root=artifact_path("archive_root"),
            archive_group=str(artifact_t.get("archive_group") or "") or None,
            plan_root=artifact_path("plan_root"),
            meta_staging=artifact_path("meta_staging"),
            catalog_db=artifact_path("catalog_db"),
            watermark_bytes=int(artifact_t.get("land_watermark_bytes", 8 * 1024**3)),
            min_interval_seconds=int(artifact_t.get("land_min_interval_seconds", 300)),
            min_groups=int(artifact_t.get("land_min_groups", 25)),
            free_bytes_floor=int(artifact_t.get("land_free_bytes_floor", 4 * 1024**3)),
        ),
        checkpoint=checkpoint,
        catalog=CatalogConfig(
            annotations=_resolve(root, _need(catalog_t, "annotations", str, "catalog")),
            global_major_limit=int(catalog_t.get("global_major_limit", 3)),
            global_per_major_limit=int(catalog_t.get("global_per_major_limit", 7)),
            local_limit=int(catalog_t.get("local_limit", 9)),
            reach_limit=int(catalog_t.get("reach_limit", 300)),
            cluster_artifact=_resolve(root, str(catalog_t["cluster_artifact"]))
            if catalog_t.get("cluster_artifact") else None,
            segment_fingerprints=_resolve(root, str(catalog_t["segment_fingerprints"]))
            if catalog_t.get("segment_fingerprints") else None,
        ),
        source_annotation=SourceAnnotationConfig(
            reasoning_effort=_need(
                source_annotation_t, "reasoning_effort", str, "source_annotation"
            ),
        ),
        terra_lanes=tuple(
            _endpoint(terra_t, databuild_path=databuild, identity=lane_identity)
            for lane_identity in _endpoint_identities(terra_t, "terra")
        ),
        validator=_endpoint(validator_t, databuild_path=databuild, validator=True),
        validator_enabled=bool(validator_t.get("enabled", False)),
        render=RenderConfig(
            backend=str(render_t.get("backend") or "cpu_lut"),
            renderer_revision=_need(render_t, "renderer_revision", str, "render"),
            search_steps=int(render_t.get("search_steps", 5)),
            clip_fraction_max=float(render_t.get("clip_fraction_max", 0.01)),
        ),
        terra_concurrency_target=int(agent.get("terra_concurrency_target", 16)),
        renderer_concurrency=int(agent.get("renderer_concurrency", 2)),
        validator_concurrency=int(agent.get("validator_concurrency", 1)),
        max_global_proposals=int(agent.get("max_global_proposals", 6)),
        min_global_proposals=int(agent.get("min_global_proposals", 2)),
        max_local_proposals=int(agent.get("max_local_proposals", 3)),
        max_local_repairs=int(agent.get("max_local_repairs", 1)),
        min_committed_leaves=int(agent.get("min_committed_leaves", 2)),
        max_committed_leaves=int(agent.get("max_committed_leaves", 6)),
        rejected_asset_ttl_days=int(agent.get("rejected_asset_ttl_days", 30)),
        request_lease_seconds=int(agent.get("request_lease_seconds", 180)),
        keep_audit_forever=bool(retention_t.get("keep_audit_forever", True)),
    )
    _validate(config)
    return config


def _validate(config: AgentLoopConfig) -> None:
    if config.prompt_revision != "local-agent-v1":
        raise ConfigError("unsupported prompt_revision; old threads require their exact registry")
    limits = {
        "terra_concurrency_target": (config.terra_concurrency_target, 1, 16),
        "max_global_proposals": (config.max_global_proposals, 1, 6),
        "min_global_proposals": (config.min_global_proposals, 1, 6),
        "max_local_proposals": (config.max_local_proposals, 1, 3),
        "max_local_repairs": (config.max_local_repairs, 0, 1),
        "min_committed_leaves": (config.min_committed_leaves, 2, 6),
        "max_committed_leaves": (config.max_committed_leaves, 2, 6),
    }
    for name, (value, low, high) in limits.items():
        if not low <= value <= high:
            raise ConfigError(f"{name} must be in [{low}, {high}]")
    if not config.terra_lanes:
        raise ConfigError("terra requires at least one endpoint lane")
    identities = [lane.identity for lane in config.terra_lanes]
    if len(set(identities)) != len(identities):
        raise ConfigError("terra lane identities must be distinct")
    if len({lane.model for lane in config.terra_lanes}) != 1:
        raise ConfigError("terra lanes must serve the same model")
    if config.min_global_proposals > config.max_global_proposals:
        raise ConfigError("minimum global proposals exceeds maximum")
    if config.min_committed_leaves > config.max_committed_leaves:
        raise ConfigError("minimum committed leaves exceeds maximum")
    if not 3 <= config.render.search_steps <= 12:
        raise ConfigError("render.search_steps must be in [3, 12]")
    if config.render.backend not in {"cpu_lut", "gpu_lut"}:
        raise ConfigError("render.backend must be cpu_lut or gpu_lut")
    # 2026-08-24 用户裁决:诊断 effort 开放三档(zzone codex 后端无视 max_output_tokens,
    # effort 是唯一有效的输出量杠杆;low 实测 out~5.8k vs high ~14.6k)。
    if config.source_annotation.reasoning_effort not in {"low", "medium", "high"}:
        raise ConfigError("source_annotation.reasoning_effort must be low, medium or high")
    if config.catalog.reach_limit < 1:
        raise ConfigError("catalog.reach_limit must be positive")
    # Quota floors (decisions doc section 2.3): 3 global bins x 2 per bin, 3 local
    # bins x 3 per bin. Anything smaller silently truncates the per-bin quota.
    if config.catalog.global_major_limit < 1:
        raise ConfigError("catalog.global_major_limit must be positive")
    if config.catalog.global_per_major_limit < 2 * 3:
        raise ConfigError("catalog.global_per_major_limit must be at least 6 (3 bins x 2)")
    if config.catalog.local_limit < 3 * 3:
        raise ConfigError("catalog.local_limit must be at least 9 (3 bins x 3)")
    landing = config.artifact_landing
    if landing.enabled:
        required = {
            "archive_root": landing.archive_root,
            "archive_group": landing.archive_group,
            "plan_root": landing.plan_root,
            "meta_staging": landing.meta_staging,
            "catalog_db": landing.catalog_db,
        }
        missing = sorted(key for key, value in required.items() if not value)
        if missing:
            raise ConfigError(f"artifact landing requires: {', '.join(missing)}")
        if landing.watermark_bytes <= 0 or landing.min_groups <= 0 \
                or landing.min_interval_seconds < 0 or landing.free_bytes_floor < 0:
            raise ConfigError("artifact landing thresholds are invalid")
    if not 0.0 <= config.render.clip_fraction_max <= 1.0:
        raise ConfigError("render.clip_fraction_max must be in [0, 1]")
    if not config.databuild_config.is_file():
        raise ConfigError(f"missing local databuild config: {config.databuild_config}")
    if not config.catalog.annotations.is_file():
        raise ConfigError(f"missing LUT annotations: {config.catalog.annotations}")
    if config.catalog.cluster_artifact is not None \
            and not config.catalog.cluster_artifact.is_file():
        raise ConfigError(
            f"missing LUT cluster artifact: {config.catalog.cluster_artifact}"
        )


__all__ = [
    "AgentLoopConfig", "ArtifactLandingConfig", "CatalogConfig", "CheckpointConfig",
    "ConfigError", "EndpointConfig", "RenderConfig", "SourceAnnotationConfig",
    "load_config",
]
