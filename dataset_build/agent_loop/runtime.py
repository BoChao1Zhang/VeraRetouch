"""Production service assembly without graph logic in transport modules."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from .api_cache import ExactResponseCache
from .artifact_landing import ArtifactLandingManager
from .artifacts import ArtifactStore
from .candidates import LutCatalog
from .checkpoint import validate_postgres_storage
from .config import AgentLoopConfig, ConfigError
from .graph import AgentServices
from .persistence import AuditStore, make_audit_store
from .preflight import require_preflight
from .render import (
    CanonicalCpuLutRenderer, CanonicalGpuRenderer, FullPresetRenderer, StrengthCalibrator,
)
from .responses import CachedResponsesClient, ResponsesAdapter
from .scheduler import TerraLane, TerraLimiter, TerraRouter
from .segment_fingerprints import REGISTERED_SEGMENT_FINGERPRINT_TABLES
from .validator import ChainValidator


def require_frozen_segment_fingerprints(
    catalog: LutCatalog, *,
    expected_sha256: str | Sequence[str] = REGISTERED_SEGMENT_FINGERPRINT_TABLES,
) -> None:
    """C1b item 2: fail loud when the mounted table is not a registered one.

    `prompt_registry()` claims specific tables (`segment_fingerprint_table_sha256` for
    v1, `segment_fingerprint_histogram.table_sha256_v2` for v2), so every thread
    revision computed from them is a lie unless the artifact that was really mounted
    hashes to one of those values. Checked once at service assembly, never silently.

    B12 item 2 widens the check from one digest to the registered set: v1 and v2 are
    both mountable, and which one a run used is recorded by the catalog itself.
    """
    expected = (
        (expected_sha256,) if isinstance(expected_sha256, str)
        else tuple(str(value) for value in expected_sha256)
    )
    mounted = catalog.segment_fingerprints_sha256
    if not mounted:
        raise ConfigError(
            "catalog.segment_fingerprints is not mounted, but the prompt registry "
            f"declares table(s) {list(expected)}"
        )
    if mounted not in expected:
        raise ConfigError(
            "mounted segment-fingerprint table "
            f"{mounted} is not one of the registered tables {list(expected)}"
        )


def audit_location(config: AgentLoopConfig) -> str:
    if config.checkpoint.backend == "postgres":
        return config.checkpoint_dsn()
    checkpoint = Path(config.checkpoint_dsn())
    return str(checkpoint.with_name(checkpoint.stem + ".audit.sqlite"))


def create_audit(config: AgentLoopConfig) -> AuditStore:
    if config.checkpoint.backend == "postgres":
        validate_postgres_storage(config)
    audit = make_audit_store(config.checkpoint.backend, audit_location(config))
    if config.checkpoint.backend == "postgres":
        validate_postgres_storage(config)
    return audit


def build_terra_router(
    config: AgentLoopConfig, artifacts: ArtifactStore, exact_cache: ExactResponseCache,
    audit: AuditStore | None = None, *, client_factory: Any | None = None,
) -> TerraRouter:
    """One client and one limiter per configured lane; the lanes never share a slot."""
    event_sink = audit.record_scheduler_event if audit is not None else None
    lanes = []
    for index, endpoint in enumerate(config.terra_lanes):
        lanes.append(TerraLane(
            index=index,
            identity=endpoint.identity,
            endpoint=endpoint,
            client=CachedResponsesClient(
                ResponsesAdapter(endpoint, artifacts, client_factory=client_factory),
                exact_cache,
            ),
            limiter=TerraLimiter(
                config.terra_concurrency_target, campaign_id=config.campaign_id,
                lane_id=endpoint.identity, event_sink=event_sink,
            ),
        ))
    return TerraRouter(lanes)


def create_services(
    config: AgentLoopConfig, *, audit: AuditStore | None = None,
    renderer_factory: Callable[[AgentLoopConfig, LutCatalog], FullPresetRenderer] | None = None,
    terra_client_factory: Any | None = None, validator_client_factory: Any | None = None,
    check_preflight: bool = True,
    expected_segment_fingerprint_sha256: "str | Sequence[str]" = (
        REGISTERED_SEGMENT_FINGERPRINT_TABLES
    ),
) -> AgentServices:
    audit = audit or create_audit(config)
    artifacts = ArtifactStore(
        config.artifact_root, recorder=audit.record_artifact,
        catalog_db=config.artifact_landing.catalog_db,
    )
    landing = ArtifactLandingManager(
        artifacts, audit, config.artifact_landing
    ) if config.artifact_landing.enabled else None
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    require_frozen_segment_fingerprints(
        catalog, expected_sha256=expected_segment_fingerprint_sha256
    )
    exact_cache = ExactResponseCache(audit, lease_seconds=config.request_lease_seconds)
    router = build_terra_router(
        config, artifacts, exact_cache, audit, client_factory=terra_client_factory
    )
    terra = router.lanes[0].client
    limiter = router.lanes[0].limiter
    if check_preflight:
        require_preflight(config, audit)
    if renderer_factory is not None:
        renderer = renderer_factory(config, catalog)
    elif config.render.backend == "cpu_lut":
        renderer = CanonicalCpuLutRenderer(config.databuild_config, catalog)
    else:
        renderer = CanonicalGpuRenderer(config.databuild_config, catalog)
    renderable = getattr(renderer, "renderable_preset_ids", None)
    if renderable is not None:
        catalog = catalog.restrict(renderable)
    calibrator = StrengthCalibrator(
        renderer, catalog, artifacts, audit,
        renderer_revision=config.render.renderer_revision,
        search_steps=config.render.search_steps,
        clip_fraction_max=config.render.clip_fraction_max,
        concurrency=config.renderer_concurrency,
        backend=config.render.backend,
    )
    validator = None
    if config.validator_enabled:
        validator_client = CachedResponsesClient(
            ResponsesAdapter(
                config.validator, artifacts, client_factory=validator_client_factory
            ), exact_cache,
        )
        validator = ChainValidator(
            validator_client, audit, concurrency=config.validator_concurrency,
            campaign_id=config.campaign_id,
        )
    return AgentServices(
        config=config, artifacts=artifacts, audit=audit, terra=terra, catalog=catalog,
        calibrator=calibrator, validator=validator, limiter=limiter, landing=landing,
        router=router,
    )


__all__ = [
    "audit_location", "build_terra_router", "create_audit", "create_services",
    "require_frozen_segment_fingerprints",
]
