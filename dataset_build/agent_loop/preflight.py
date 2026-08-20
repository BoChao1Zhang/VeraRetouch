"""Persisted endpoint/model/schema/image/provider-cache capability probe."""
from __future__ import annotations

import hashlib
import io
import time
from typing import Any

from PIL import Image, ImageDraw

from .api_cache import canonical_json
from .artifacts import ArtifactStore
from .candidates import LutCatalog
from .config import AgentLoopConfig, EndpointConfig
from .persistence import AuditStore
from .prompts import (
    ADAPTER_REVISION, PROMPT_REVISION, preflight_request, prompt_revision_fingerprint,
)
from .responses import CachedResponsesClient
from .scheduler import TerraLimiter


PREFLIGHT_FIXTURE_REVISION = "provider-preflight-v5-live-prefix-cache"
PREFLIGHT_CACHE_SETTLE_SECONDS = 2.0


def preflight_key(
    config: AgentLoopConfig, endpoint: EndpointConfig | None = None
) -> str:
    endpoint = endpoint or config.terra
    payload = {
        "endpoint_identity": endpoint.identity, "model": endpoint.model,
        "prompt_revision": PROMPT_REVISION, "adapter_revision": ADAPTER_REVISION,
        "prompt_registry_sha256": prompt_revision_fingerprint(),
        "fixture_revision": PREFLIGHT_FIXTURE_REVISION,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _fixture_image(artifacts: ArtifactStore) -> dict[str, Any]:
    image = Image.new("RGB", (96, 64), "white")
    draw = ImageDraw.Draw(image)
    colors = ("#d94b3d", "#e7bd42", "#50a862", "#3978c5", "#777777", "#222222")
    for index, color in enumerate(colors):
        left = (index % 3) * 32
        top = (index // 3) * 32
        draw.rectangle((left, top, left + 31, top + 31), fill=color)
    output = io.BytesIO()
    image.save(output, "JPEG", quality=90, subsampling=0)
    return artifacts.put_bytes(
        output.getvalue(), media_type="image/jpeg", retention="audit"
    ).to_dict()


def _reference(catalog: LutCatalog) -> list[dict[str, Any]]:
    return [record.prompt_view() for record in catalog.records[:4]]


def run_preflight(
    config: AgentLoopConfig, artifacts: ArtifactStore, audit: AuditStore,
    client: CachedResponsesClient, catalog: LutCatalog, limiter: TerraLimiter,
    *, endpoint: EndpointConfig | None = None,
) -> dict[str, Any]:
    endpoint = endpoint or config.terra
    key = preflight_key(config, endpoint)
    fixture = _fixture_image(artifacts)
    reference = _reference(catalog)
    calls = []

    def probe(tail: str) -> None:
        spec = preflight_request(endpoint, fixture, reference, tail)
        with limiter.slot(f"preflight:{key}", "preflight", spec.prompt_cache_key):
            result = client.request(
                spec,
                lambda parsed, expected=tail: None
                if parsed.get("ok") is True and parsed.get("tail") == expected
                else "preflight_echo",
                bypass_exact_cache=True,
            )
        calls.append({
            "request_hash": result.request_hash, "cache_hit": result.cache_hit,
            "usage": result.usage, "model": result.response.get("model"),
            "prompt_cache_key": spec.prompt_cache_key,
        })

    probe("first")
    probe("second")
    if not endpoint.allow_uncached_provider \
            and int(calls[-1]["usage"].get("cached_tokens", 0)) <= 0:
        time.sleep(PREFLIGHT_CACHE_SETTLE_SECONDS)
        probe("third-after-cache-settle")
    shared_key = len({row["prompt_cache_key"] for row in calls}) == 1
    cached_tokens = max(
        (int(row["usage"].get("cached_tokens", 0)) for row in calls[1:]), default=0
    )
    provider_cache_passed = shared_key and cached_tokens > 0
    degraded = not provider_cache_passed and endpoint.allow_uncached_provider
    passed = provider_cache_passed or degraded
    record = {
        "preflight_key": key, "endpoint_identity": endpoint.identity,
        "model": endpoint.model, "prompt_revision": config.thread_revision,
        "image_input": True, "strict_schema": True, "exact_model": all(
            row["model"] == endpoint.model for row in calls
        ),
        "prompt_cache_key_accepted": shared_key,
        "live_provider_probes": True,
        "cached_tokens_readable": all("cached_tokens" in row["usage"] for row in calls),
        "provider_cache_passed": provider_cache_passed,
        "allow_uncached_provider": endpoint.allow_uncached_provider,
        "degraded_uncached": degraded, "calls": calls,
    }
    passed = passed and record["exact_model"] and record["cached_tokens_readable"]
    audit.record_preflight(key, record, bool(passed))
    if not passed:
        raise RuntimeError("provider_preflight_failed_closed")
    return record


def require_preflight(config: AgentLoopConfig, audit: AuditStore) -> dict[str, Any]:
    """Every configured lane is a separate provider identity: every lane needs its own pass."""
    records = []
    for lane in config.terra_lanes:
        result = audit.get_preflight(preflight_key(config, lane))
        if not result or not result["passed"]:
            raise RuntimeError(f"provider_preflight_required:{lane.identity}")
        records.append(result)
    return records[0]


__all__ = ["preflight_key", "require_preflight", "run_preflight"]
