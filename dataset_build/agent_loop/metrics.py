"""Campaign-level API, scheduler, and data-quality metrics from raw audit rows."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

from .persistence import AuditStore


def _object(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def campaign_metrics(audit: AuditStore, campaign_id: str) -> dict[str, Any]:
    tables = audit.export_tables()
    request_rows = {
        row["request_hash"]: _object(row.get("canonical_request_json")) or {}
        for row in tables["api_request"]
    }
    contexts = [row for row in tables["api_request_context"]
                if row["campaign_id"] == campaign_id]
    campaign_hashes = {row["request_hash"] for row in contexts}
    stages = {request_hash: "unknown" for request_hash in campaign_hashes}
    for context in sorted(contexts, key=lambda row: float(row.get("created_at", 0))):
        stages[context["request_hash"]] = str(context.get("stage") or "unknown")
    attempt_by_stage: Counter[str] = Counter()
    usage_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    attempt_usage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in tables["api_attempt"]:
        request_hash = row["request_hash"]
        if request_hash not in campaign_hashes:
            continue
        stage = stages[request_hash]
        usage = _object(row.get("usage_json")) or {}
        attempt_by_stage[stage] += 1
        for key in ("input_tokens", "output_tokens", "cached_tokens"):
            value = int(usage.get(key, 0) or 0)
            usage_by_stage[stage][key] += value
            attempt_usage[request_hash][key] += value
    cache_by_stage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in tables["api_cache_event"]:
        request_hash = row["request_hash"]
        if request_hash in campaign_hashes:
            cache_by_stage[stages[request_hash]][row["event"]] += 1
    prefix_usage: dict[str, Counter[str]] = defaultdict(Counter)
    for request_hash in campaign_hashes:
        request = request_rows.get(request_hash, {})
        prefix = str(request.get("prompt_cache_key") or "unknown")
        for key, value in attempt_usage[request_hash].items():
            prefix_usage[prefix][key] += value

    source_api: dict[str, Counter[str]] = defaultdict(Counter)
    source_usage: dict[str, Counter[str]] = defaultdict(Counter)
    for context in contexts:
        source = context["source_sha256"]
        stage = context["stage"]
        if not bool(context["cache_hit"]):
            source_api[source][stage] += len([
                row for row in tables["api_attempt"]
                if row["request_hash"] == context["request_hash"]
            ])
            for key, value in attempt_usage[context["request_hash"]].items():
                source_usage[source][key] += value

    source_rows = [row for row in tables["agent_source_run"]
                   if row["campaign_id"] == campaign_id]
    source_hashes = {row["source_sha256"] for row in source_rows}
    branch_rows = [row for row in tables["agent_branch"]
                   if row["campaign_id"] == campaign_id]
    branch_status = Counter(row["status"] for row in branch_rows)
    global_bins: Counter[str] = Counter()
    preset_ids: Counter[str] = Counter()
    mask_families: Counter[str] = Counter()
    for row in branch_rows:
        proposal = _object(row.get("proposal_json")) or {}
        if proposal.get("preset_id"):
            preset_ids[str(proposal["preset_id"])] += 1
        if row["level"] == "global" and proposal.get("strength_bin"):
            global_bins[str(proposal["strength_bin"])] += 1
        result = _object(row.get("result_json")) or {}
        if row["level"] == "local" and result.get("mask_family"):
            mask_families[str(result["mask_family"])] += 1
    defects: Counter[str] = Counter()
    for row in tables["validation_record"]:
        if row["source_sha256"] not in source_hashes:
            continue
        for defect in _object(row.get("defects_json")) or []:
            defects[str(defect.get("defect_code") or "unknown")] += 1
    reject_reasons: Counter[str] = Counter()
    committed_distribution: Counter[str] = Counter()
    for row in source_rows:
        counts = _object(row.get("counts_json")) or {}
        committed_distribution[str(int(counts.get("committed_leaves", 0)))] += 1
        for reason in counts.get("reject_reasons", []):
            reject_reasons[str(reason)] += 1

    scheduler = [row for row in tables["scheduler_event"]
                 if row["campaign_id"] == campaign_id]
    start_rows = [row for row in scheduler if row["event"] == "start"]
    total_input = sum(row["input_tokens"] for row in usage_by_stage.values())
    total_cached = sum(row["cached_tokens"] for row in usage_by_stage.values())
    artifact_storage: dict[str, Counter[str]] = defaultdict(Counter)
    for row in tables["artifact_record"]:
        retention = str(row.get("retention") or "unknown")
        artifact_storage[retention]["count"] += 1
        artifact_storage[retention]["bytes"] += int(row.get("size") or 0)
    source_wall = [
        max(0.0, float(row["updated_at"]) - float(row["started_at"]))
        for row in source_rows
    ]
    render_metrics = [
        _object(row.get("metrics_json")) or {} for row in tables["render_record"]
        if row["source_sha256"] in source_hashes
    ]
    return {
        "campaign_id": campaign_id,
        "api": {
            "real_calls_by_stage": dict(sorted(attempt_by_stage.items())),
            "usage_by_stage": {stage: dict(values) for stage, values in sorted(usage_by_stage.items())},
            "exact_cache_by_stage": {stage: dict(values) for stage, values in sorted(cache_by_stage.items())},
            "provider_cached_token_ratio": total_cached / total_input if total_input else 0.0,
            "prefix_usage": {key: dict(value) for key, value in sorted(prefix_usage.items())},
            "per_source_calls": {key: dict(value) for key, value in sorted(source_api.items())},
            "per_source_usage": {key: dict(value) for key, value in sorted(source_usage.items())},
        },
        "scheduler": {
            "events": len(scheduler),
            "max_terra_in_flight": max((int(row["in_flight"]) for row in start_rows), default=0),
            "mean_terra_in_flight_at_start": (
                sum(int(row["in_flight"]) for row in start_rows) / len(start_rows)
                if start_rows else 0.0
            ),
            "effective_limit_min": min(
                (int(row["effective_limit"]) for row in scheduler), default=0
            ),
        },
        "quality": {
            "source_status": dict(Counter(row["status"] for row in source_rows)),
            "branch_status": dict(branch_status), "global_strength_bins": dict(global_bins),
            "preset_ids": dict(preset_ids), "mask_families": dict(mask_families),
            "defect_codes": dict(defects), "reject_reasons": dict(reject_reasons),
            "committed_leaves_per_source": dict(committed_distribution),
        },
        "storage": {
            "artifacts_by_retention": {
                key: dict(value) for key, value in sorted(artifact_storage.items())
            },
            "database_rows_by_table": {
                key: len(value) for key, value in sorted(tables.items())
            },
            "database": audit.database_storage(),
        },
        "runtime": {
            "source_wall_seconds_total": sum(source_wall),
            "source_wall_seconds_mean": sum(source_wall) / len(source_wall)
            if source_wall else 0.0,
            "render_wall_seconds_total": sum(
                float(row.get("render_wall_seconds", 0.0)) for row in render_metrics
            ),
            "render_thread_cpu_seconds_total": sum(
                float(row.get("render_thread_cpu_seconds", 0.0)) for row in render_metrics
            ),
        },
    }


__all__ = ["campaign_metrics"]
