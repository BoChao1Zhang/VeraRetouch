from __future__ import annotations

import base64
import dataclasses
import hashlib
import io
import itertools
import json
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pytest
from PIL import Image

from dataset_build.agent_loop.api_cache import (
    CacheExhausted, ExactResponseCache, RequestSpec, prefix_cache_key,
    retryable_exception,
)
from dataset_build.agent_loop.artifact_landing import ArtifactLandingManager
from dataset_build.agent_loop.artifacts import ArtifactStore
from dataset_build.agent_loop.candidates import (
    BACKGROUND_FAMILY_COUNTS, BACKGROUND_ROLE_GATE, CandidateError, LutCatalog,
    SUBJECT_BAND_GATE, band_geometry_reading,
    LutRecord, _evaluate_geometry, _fit_full_resolution_geometry, _stats,
    allocate_mask_packets, allocate_role_packets, build_local_packets,
    build_mask_bank, intent_admits, intent_offered, mask_summary, region_descriptor,
    validate_mask_bank,
)
from dataset_build.agent_loop.config import (
    ArtifactLandingConfig, ConfigError, EndpointConfig, load_config,
)
from dataset_build.agent_loop.checkpoint import ShardedCheckpointSaver, open_checkpointer
from dataset_build.agent_loop.graph import (
    AgentServices, _direction_cosine, _discard_uncommitted_renders,
    _global_propose_node, _validate_local_response, build_graph, run_source,
    select_committed_leaves, source_content_hash,
)
from dataset_build.agent_loop.direction_match import (
    DirectionVector, MaskDirection, direction_match_score, uniform_tonal_weights,
)
from dataset_build.agent_loop.models import (
    ACTIVE_LOCAL_INTENTS, BAND_GEOMETRY_GATE, DISABLED_LOCAL_INTENTS,
    FINGERPRINT_FIELDS,
    GLOBAL_DELTA_E_TARGETS, INTENT_BIN_QUOTA, INTENT_DIRECTION_GATE,
    INTENT_FINGERPRINT_GATE, INTENT_PACKET_PRIORITY, INTENT_ROLE_DOMAINS, INTENT_ROLES,
    INTENT_SEGMENT_GATE,
    LOCAL_DELTA_E_LADDERS, LOCAL_DELTA_E_TARGETS, LOCAL_INTENTS,
    LOCAL_ONLINE_RETRIEVAL, LOCAL_PACKET_ROW_LIMIT, LOCAL_VISIBILITY_FLOOR,
    LUMA_SOFT_CAP,
    LUMA_SOFT_CAP_INTENTS, MASK_REACH_GATE, REASON_CODES,
    SUBJECT_HEADROOM_GATE, assert_local_visibility_floor, intent_packet_order,
    intent_serves_role, local_ladder,
    local_target_center, luma_capped, resolve_local_target,
)
from dataset_build.agent_loop.segment_fingerprints import (
    DERIVATION_REVISION, SEGMENT_FIELDS, SEGMENT_FINGERPRINT_SCHEMA,
    SEGMENT_FINGERPRINT_TABLE_SHA256, SEGMENT_NAMES,
)
from dataset_build.agent_loop.metrics import campaign_metrics
from dataset_build.agent_loop.lut_annotations import file_sha256, migrate_annotation
from dataset_build.agent_loop.persistence import SQLiteAuditStore, _POSTGRES_SCHEMA
from dataset_build.agent_loop import candidates as candidates_module
from dataset_build.agent_loop import prompts as prompts_module
from dataset_build.agent_loop.histogram_board import (
    BOARD_REVISION, BOARD_SIZE, board_png,
)
from dataset_build.agent_loop.prompts import (
    BOARD_IMAGE_ENCODING, CANDIDATE_SERIALIZATION_REVISION, GLOBAL_BATCH_SCHEMA,
    LOCAL_BATCH_SCHEMA,
    LOCAL_SHORTLIST_NOTE, PROMPT_REGISTRY_KEYS, SHORTLIST_COLUMNS, diagnosis_request,
    global_request, local_request, prompt_registry, prompt_revision_fingerprint,
    semantic_error, shortlist_row_text, shortlist_rows_text, validation_request,
)
from dataset_build.agent_loop.preflight import (
    preflight_key, require_preflight, run_preflight,
)
from dataset_build.agent_loop.render import (
    CanonicalCpuLutRenderer, FullPresetRenderer, LocalDirectionProbe, MaskReachProbe,
    RenderError, StrengthCalibrator,
    load_alpha, subject_clip_regression, subject_highlight_headroom,
)
from dataset_build.agent_loop.responses import (
    ModelSubstituted, ResponsesAdapter, TransportError, consume_response, consume_stream,
)
from dataset_build.agent_loop.runtime import (
    build_terra_router, create_services, require_frozen_segment_fingerprints,
)
from dataset_build.agent_loop.scheduler import (
    TerraLane, TerraLimiter, TerraRouter, route_lane_index,
)
from dataset_build.agent_loop.source_annotations import (
    SOURCE_ANNOTATION_SCHEMA, annotate_source, load_source_annotation,
)
from dataset_build.agent_loop.source_histogram import (
    assert_histogram_columns, source_histogram,
)
from dataset_build.agent_loop.source_reach import (
    REACH_SCHEMA, SAMPLER_REVISION, SAMPLE_PIXELS, configured_lut_loader,
    probe_preset_reach,
)
from dataset_build.agent_loop import cli as agent_loop_cli
from dataset_build.agent_loop.retention import (
    apply_cleanup_manifest, plan_cleanup, write_cleanup_manifest,
)


def _lane_ids(lanes: int) -> list[str]:
    return ["lane"] + [f"lane{index + 1}" for index in range(1, lanes)]


def _segments(**overrides: dict[str, float]) -> dict[str, dict[str, float]]:
    """B10/R6.1 shaped segmented fingerprint, all zeros unless overridden."""
    base = {
        name: {field: 0.0 for field in SEGMENT_FIELDS} for name in SEGMENT_NAMES
    }
    for name, values in overrides.items():
        base[name].update({key: float(value) for key, value in values.items()})
    return base


def _fingerprint_row(preset_id: str, segments: dict[str, dict[str, float]]) -> dict:
    return {
        "schema": SEGMENT_FINGERPRINT_SCHEMA, "preset_id": preset_id,
        "derivation_revision": DERIVATION_REVISION,
        "source_sha256": "c" * 64, "hsl_features_sha256": "d" * 64,
        "bands_per_segment": {name: 0 for name in SEGMENT_NAMES},
        "segments": segments,
    }


def _write_fingerprints(path: Path, rows: dict[str, dict[str, dict[str, float]]]) -> Path:
    path.write_text("".join(
        json.dumps(_fingerprint_row(preset_id, segments)) + "\n"
        for preset_id, segments in sorted(rows.items())
    ), encoding="utf-8")
    return path


def _fixture_segments(index: int) -> dict[str, dict[str, float]]:
    """Deterministic per-preset fingerprint of the `_write_config` bank."""
    return _segments(
        shadows={"dL": -float(index), "cast_b": -float(index % 2)},
        mids={"dL": float(index) - 3.5, "cast_a": float(index % 2),
              "cast_b": -float(index % 2)},
        highlights={"dL": float(index), "cast_b": -float(index % 2)},
    )


def _write_config(root: Path, *, presets: int = 8, validator_enabled: bool = True,
                  clusters: dict[str, str] | None = None, lanes: int = 1):
    bank = root / "bank"
    bank.mkdir()
    feature_rows = []
    annotation_rows = []
    for index in range(presets):
        preset = root / f"p{index}.cube"
        amount = min(0.45, index * 0.07)
        rows = []
        for blue in (0.0, 1.0):
            for green in (0.0, 1.0):
                for red in (0.0, 1.0):
                    rows.append(f"{min(1.0, red + amount)} {green} {blue}\n")
        preset.write_text("LUT_3D_SIZE 2\n" + "".join(rows), encoding="ascii")
        feature_rows.append({
            "preset_id": f"p{index}", "path": str(preset), "kind": "lut", "fmt": "cube",
        })
        source_annotation = {
            "preset_id": f"p{index}", "ok": True, "name": f"preset {index}",
            "style_major": "major", "style_minor": f"minor {index}",
            "scene_affinity": "general", "strength": ("weak", "medium", "strong")[index % 3],
            "caption": f"objective response {index}", "per_probe": {},
            "hsl_features": {"summary": {
                "mid_gray_a": float(index % 2), "mid_gray_b": float(-(index % 2)),
                "mid_gray_dL": float(index) - 3.5, "sat_pct_mean": 0.0,
                "contrast_ratio": 1.0 + index / 100.0,
                "shadow_dL": float(index), "highlight_dL": -float(index),
                "hue_rot_abs_max": float(index) / 2.0,
            }, "bands": {}},
        }
        annotation_rows.append(migrate_annotation(
            source_annotation, float(index + 3),
            annotations_source_sha256="a" * 64,
            perceptual_de_source_sha256="b" * 64,
        ))
    (bank / "features.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in feature_rows), encoding="utf-8"
    )
    annotations = root / "annotations.jsonl"
    annotations.write_text(
        "".join(json.dumps(row) + "\n" for row in annotation_rows), encoding="utf-8"
    )
    # B11 item 1 (R7.1): the online local retrieval needs a mounted fingerprint table.
    fingerprints = _write_fingerprints(root / "segment_fingerprints.jsonl", {
        f"p{index}": _fixture_segments(index) for index in range(presets)
    })
    cluster_line = ""
    if clusters is not None:
        cluster_path = root / "clusters.jsonl"
        cluster_path.write_text("".join(json.dumps({
            "preset_id": preset_id, "style_major": "shared", "cluster_id": cluster_id,
        }) + "\n" for preset_id, cluster_id in sorted(clusters.items())), encoding="utf-8")
        cluster_line = f'cluster_artifact = "{cluster_path}"\n'
    lane_ids = _lane_ids(lanes)
    endpoint_blocks = "".join(
        f'[[annotation.external_endpoints]]\nid = "{lane_id}"\n'
        f'base_url = "https://provider.example/v1"\n'
        f'api_key = "TOP-SECRET-{lane_id}"\nconcurrency = 16\n\n'
        for lane_id in lane_ids
    )
    terra_endpoint_line = 'endpoint = "lane"' if lanes == 1 else \
        "endpoints = [" + ", ".join(f'"{lane_id}"' for lane_id in lane_ids) + "]"
    databuild = root / "local.toml"
    databuild.write_text(f"""
[presets]
bank_dir = "{bank}"

[annotation]
external_model = "gpt-5.6-terra"

{endpoint_blocks}
[annotation.local]
base_url = "http://127.0.0.1:8003/v1"
api_key = "EMPTY"
model = "local-3b"
""", encoding="utf-8")
    agent = root / "agent.toml"
    agent.write_text(f"""
[agent_loop]
campaign_id = "test"
prompt_revision = "local-agent-v1"
databuild_config = "{databuild}"
terra_concurrency_target = 4
renderer_concurrency = 2
validator_concurrency = 2
max_global_proposals = 6
max_local_proposals = 3
max_local_repairs = 1
min_committed_leaves = 2
max_committed_leaves = 6
rejected_asset_ttl_days = 30
request_lease_seconds = 2

[checkpoint]
backend = "sqlite"
sqlite_path = "{root / 'checkpoint.sqlite'}"

[artifacts]
root = "{root / 'artifacts'}"

[catalog]
annotations = "{annotations}"
segment_fingerprints = "{fingerprints}"
reach_limit = 8
{cluster_line}

[source_annotation]
reasoning_effort = "high"

[terra]
{terra_endpoint_line}
model = "gpt-5.6-terra"
reasoning_effort = "medium"
temperature = 0.1
max_output_tokens = 6000
timeout_seconds = 10.0
attempts = 3
allow_uncached_provider = false

[validator]
enabled = {str(validator_enabled).lower()}
endpoint = "local"
temperature = 0.1
max_output_tokens = 1024
timeout_seconds = 10.0
attempts = 3

[render]
renderer_revision = "test-render-v1"
search_steps = 9
clip_fraction_max = 0.2

[retention]
keep_audit_forever = true
rejected_image_days = 30
""", encoding="utf-8")
    return load_config(agent)


def _catalog(count: int = 8, clusters: dict[str, str] | None = None) -> LutCatalog:
    return LutCatalog((LutRecord(
        preset_id=f"p{index}", path=f"/unused/p{index}.cube", format="lut",
        name=f"preset {index}", style_major="major", style_minor=f"minor {index}",
        scene_affinity=("portrait",), de_med=float(index + 3),
        caption="objective response", per_probe={}, hsl_features={
            "summary": {"mid_gray_a": 0.0, "mid_gray_b": 0.0,
                        "mid_gray_dL": 0.0, "sat_pct_mean": 0.0}, "bands": {},
        },
        segment_fingerprint=_fixture_segments(index),
    ) for index in range(count)), clusters, segment_fingerprints_sha256="e" * 64)


def _source_files(root: Path, *, area: str = "large") -> tuple[Path, Path]:
    source = root / "source.jpg"
    Image.new("RGB", (96, 64), (120, 130, 140)).save(source)
    mask = np.zeros((64, 96), dtype=np.uint8)
    if area == "large":
        mask[12:52, 18:78] = 255
    else:
        mask[25:39, 38:58] = 255
    subject = root / "subject.png"
    Image.fromarray(mask, "L").save(subject)
    return source, subject


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (255, 255, 255)).save(buffer, "PNG", optimize=True)
    return buffer.getvalue()


def _board_ref(digest: str = "b" * 64) -> dict[str, Any]:
    """E4: the second diagnosis image (the histogram board PNG artifact)."""
    return {"sha256": digest, "media_type": "image/png", "size": 20,
            "uri": "sha256://" + digest}


def _diagnosis() -> dict[str, Any]:
    return {
        "correction_needs": [], "preserve_intent": ["skin tone"],
        "enhancement_opportunities": ["tone separation", "depth"],
        "forbidden_directions": [], "evidence": ["balanced exposure"],
        "confidence": 0.8, "intent_mode": "enhancement_led",
    }


def _source_row(source: Path, subject: Path) -> dict[str, Any]:
    reach = {
        "schema": REACH_SCHEMA, "sampler_revision": SAMPLER_REVISION,
        "requested_pixels": SAMPLE_PIXELS, "sampled_pixels": 4096,
        "catalog_sha256": "a" * 64, "preset_count": 8,
        "presets": {f"p{index}": {
            "d_full": 10.0,
            "achievable_bins": ["natural", "medium", "bold"],
        } for index in range(8)},
    }
    annotation = {
        "schema": SOURCE_ANNOTATION_SCHEMA,
        "source_id": "source",
        "source_sha256": source_content_hash(source),
        "source_path": str(source),
        "subject_path": str(subject),
        "subject_sha256": source_content_hash(subject),
        "scene": "portrait",
        "subject": {},
        "diagnosis": _diagnosis(),
        "preset_reach": reach,
        "provenance": {
            "stage": "offline_diagnose", "model": "test",
            "reasoning_effort": "high",
        },
    }
    return {
        "source_id": "source", "source_path": str(source),
        "subject_path": str(subject), "scene": "portrait",
        "source_annotation": annotation,
    }


def test_config_uses_local_endpoint_without_leaking_it(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    assert config.terra.base_url == "https://provider.example/v1"
    assert config.terra.api_key == "TOP-SECRET-lane"
    safe = json.dumps(config.sanitized_dict())
    assert "TOP-SECRET" not in safe
    assert "provider.example" not in safe


def test_graph_or_catalog_limit_changes_thread_revision(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    graph_changed = dataclasses.replace(
        config, max_global_proposals=config.max_global_proposals - 1
    )
    catalog_changed = dataclasses.replace(
        config,
        catalog=dataclasses.replace(
            config.catalog, global_per_major_limit=config.catalog.global_per_major_limit - 1,
        ),
    )
    assert graph_changed.thread_revision != config.thread_revision
    assert catalog_changed.thread_revision != config.thread_revision


def test_mounted_segment_fingerprint_path_enters_the_thread_revision(
    tmp_path: Path,
) -> None:
    """C1b item 2 (was B10, inverted).

    B10 froze the invariant `mounting the fingerprint artifact changes nothing`. B11
    then made the mounted table the whole candidate universe of the online local
    retrieval, which makes two configs pointing at two different tables two different
    threads. So the artifact is now part of `catalog_contract`.
    """
    config = _write_config(tmp_path)
    unmounted = dataclasses.replace(
        config, catalog=dataclasses.replace(config.catalog, segment_fingerprints=None),
    )
    other = dataclasses.replace(
        config,
        catalog=dataclasses.replace(
            config.catalog, segment_fingerprints=tmp_path / "other-fingerprints.jsonl"
        ),
    )
    assert unmounted.catalog.segment_fingerprints is None
    assert unmounted.thread_revision != config.thread_revision
    assert other.thread_revision != config.thread_revision
    assert other.thread_revision != unmounted.thread_revision
    catalog_view = other.sanitized_dict()["catalog"]
    assert catalog_view["segment_fingerprints"] == \
        str(tmp_path / "other-fingerprints.jsonl")
    assert json.dumps(other.sanitized_dict())


def test_startup_rejects_a_segment_fingerprint_table_that_is_not_the_frozen_one(
    tmp_path: Path,
) -> None:
    """C1b item 2: the registry claims table SHAs; a different mount fails loud.

    B12 item 2: the claim is now the *set* `REGISTERED_SEGMENT_FINGERPRINT_TABLES`
    (v1 and v2), and a mount that is in the set passes whichever member it matches.
    """
    config = _write_config(tmp_path, presets=2)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    mounted = catalog.segment_fingerprints_sha256
    assert len(mounted) == 64
    require_frozen_segment_fingerprints(catalog, expected_sha256=mounted)
    require_frozen_segment_fingerprints(catalog, expected_sha256=("0" * 64, mounted))
    with pytest.raises(ConfigError, match="not one of the registered tables"):
        require_frozen_segment_fingerprints(catalog, expected_sha256="0" * 64)
    with pytest.raises(ConfigError, match="not one of the registered tables"):
        require_frozen_segment_fingerprints(
            catalog, expected_sha256=("0" * 64, "1" * 64)
        )
    with pytest.raises(ConfigError, match="not one of the registered tables"):
        # The production default is the constant set in `prompt_registry()`, which no
        # fixture table can ever match.
        require_frozen_segment_fingerprints(catalog)
    unmounted = LutCatalog.load(
        dataclasses.replace(config.catalog, segment_fingerprints=None),
        config.databuild_config,
    )
    with pytest.raises(ConfigError, match="not mounted"):
        require_frozen_segment_fingerprints(unmounted, expected_sha256=mounted)


def test_lut_catalog_excludes_non_lut_presets(tmp_path: Path) -> None:
    config = _write_config(tmp_path, presets=1)
    xmp = tmp_path / "not-a-lut.xmp"
    xmp.write_text("<x:xmpmeta/>", encoding="ascii")
    features = tmp_path / "bank" / "features.jsonl"
    features.write_text(
        features.read_text(encoding="utf-8") + json.dumps({
            "preset_id": "xmp0", "path": str(xmp), "kind": "preset", "fmt": "xmp",
        }) + "\n",
        encoding="utf-8",
    )
    config.catalog.annotations.write_text(
        config.catalog.annotations.read_text(encoding="utf-8") + json.dumps({
            "preset_id": "xmp0", "ok": True, "style_major": "major",
            "style_minor": "minor", "hsl_features": {},
        }) + "\n",
        encoding="utf-8",
    )
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    assert set(catalog.by_id) == {"p0"}


def _fingerprint_record() -> LutRecord:
    return dataclasses.replace(
        _catalog(1).records[0], caption="中文 说明  文本", per_probe={"red": "long prose"},
        hsl_features={
            "summary": {
                "mid_gray_dL": 1.23456, "contrast_ratio": 1.0987654,
                "shadow_dL": -2.34567, "highlight_dL": 3.4567,
                "mid_gray_a": 3.0, "mid_gray_b": 3.0,
                "sat_pct_mean": -4.5678, "hue_rot_abs_max": 6.789,
            },
            "bands": {
                name: {"d_hue_deg": float(index), "d_sat_pct": index / 2}
                for index, name in enumerate(("red", "green", "blue", "yellow"), 1)
            },
        },
    )


def test_lut_prompt_view_is_the_eight_number_fingerprint_only() -> None:
    view = _fingerprint_record().prompt_view()
    assert set(view) == {"preset_id", "style_major", "fingerprint", "caption"}
    assert set(view["fingerprint"]) == set(FINGERPRINT_FIELDS)
    assert view["fingerprint"] == {
        "dL": 1.2, "contrast": 1.099, "shadow_dL": -2.3, "highlight_dL": 3.5,
        "cast_hue": 45.0, "cast_mag": 4.2, "dSat": -4.6, "hue_rot": 6.8,
    }
    assert "per_probe" not in view and "style_minor" not in view


def test_shortlist_row_serialization_is_byte_stable() -> None:
    row = {
        **_fingerprint_record().prompt_view(),
        "achievable_bins": ["natural", "medium"],
        "d_full": 9.5, "scorer_rank_raw": 0, "scorer_rank_offered": 0, "score": 1.0,
    }
    line = shortlist_row_text(3, row)
    assert line == (
        "3 | natural,medium | 1.2 1.099 -2.3 3.5 45 4.2 -4.6 6.8 | 中文 说明 文本"
    )
    assert line == shortlist_row_text(3, dict(reversed(list(row.items()))))
    assert SHORTLIST_COLUMNS == (
        "id | achievable_bins | dL contrast shadow_dL highlight_dL cast_hue cast_mag "
        "dSat hue_rot | caption"
    )
    assert "strongest_bands" not in shortlist_rows_text([row])
    assert len(line) / 3 < 80  # single-row prompt cost stays in the ~60 token band
    assert CANDIDATE_SERIALIZATION_REVISION == "lut-intent-v7.1-optB"
    # B8 item 4: a local row carries the mask-conditioned reach as a trailing column;
    # a global row (no `mask_reach_de`) is byte-identical to the B7 line.
    assert shortlist_row_text(3, {**row, "mask_reach_de": 4.125}) == line + " | 4.12"
    # B12 item 3: a v2-mounted row closes with `d_shadow d_mid d_high`, after the
    # mask-reach column when there is one. A row without a `histogram` key is
    # byte-identical to the B11 line.
    histogram = {"d_shadow": -0.0231, "d_mid": 0.4, "d_high": -0.0004}
    assert shortlist_row_text(3, {**row, "histogram": histogram}) == \
        line + " | -0.023 0.400 0.000"
    assert shortlist_row_text(
        3, {**row, "mask_reach_de": 4.125, "histogram": histogram}
    ) == line + " | 4.12 | -0.023 0.400 0.000"


def test_fingerprint_formatting_never_emits_negative_zero(monkeypatch) -> None:
    row = {
        "caption": "objective", "achievable_bins": ["natural"],
        "fingerprint": {name: 0.0 for name in FINGERPRINT_FIELDS},
    }
    signed = {**row, "fingerprint": {
        **row["fingerprint"], "dL": -0.04, "cast_hue": -0.4, "contrast": -0.0004,
        "shadow_dL": -0.6,
    }}
    assert shortlist_row_text(0, signed) == (
        "0 | natural | 0.0 0.000 -0.6 0.0 0 0.0 0.0 0.0 | objective"
    )
    assert "-0" not in shortlist_row_text(0, row)

    before = prompt_revision_fingerprint()
    monkeypatch.setattr(
        prompts_module, "FINGERPRINT_FORMATS",
        {**prompts_module.FINGERPRINT_FORMATS, "dL": "{:.3f}"},
    )
    assert prompt_revision_fingerprint() != before


def test_canonical_hash_and_prefix_isolation(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    source = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "a" * 64}
    diagnosis = {"x": 1}
    shortlist = {"major": [{"preset_id": "p0"}]}
    first = global_request(config.terra, source, diagnosis, shortlist, max_proposals=6)
    changed_secret = dataclasses.replace(
        config.terra, api_key="DIFFERENT", timeout_seconds=999.0
    )
    second = global_request(changed_secret, source, diagnosis, shortlist, max_proposals=6)
    assert first.request_hash == second.request_hash
    reordered = RequestSpec(
        canonical=dict(reversed(list(first.canonical.items()))),
        prompt_cache_key=first.prompt_cache_key,
    )
    assert reordered.request_hash == first.request_hash
    changed_order = dict(first.canonical)
    changed_order["input"] = list(reversed(changed_order["input"]))
    assert RequestSpec(changed_order, first.prompt_cache_key).request_hash != first.request_hash
    changed_schema = dict(first.canonical)
    changed_schema["schema"] = {**GLOBAL_BATCH_SCHEMA, "title": "changed"}
    assert RequestSpec(changed_schema, first.prompt_cache_key).request_hash != first.request_hash


def test_diagnosis_request_hash_includes_high_reasoning_effort(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    source = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "a" * 64}
    medium = diagnosis_request(config.terra, source, _board_ref())
    high = diagnosis_request(dataclasses.replace(
        config.terra, reasoning_effort="high"
    ), source, _board_ref())
    assert medium.request_hash != high.request_hash
    assert high.canonical["behavior"]["reasoning_effort"] == "high"


def test_global_prompt_offers_majors_and_forbids_reusing_a_row(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    source = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "a" * 64}
    request = global_request(
        config.terra, source, _diagnosis(),
        {"small-major": [{"preset_id": "p0", "achievable_bins": ["natural"],
                          "fingerprint": {name: 0.0 for name in FINGERPRINT_FIELDS},
                          "caption": "objective"}]},
        min_proposals=2, max_proposals=4,
    )
    rules = request.canonical["input"][0]["content"][0]["text"]
    assert "Every proposal must use a different\nrow_index" in rules
    assert "return fewer\nproposals and never repeat a row" in rules
    shortlist_text = request.canonical["input"][1]["content"][-2]["text"]
    assert SHORTLIST_COLUMNS in shortlist_text
    assert "[major] small-major" in shortlist_text
    assert "0 | natural | 0.0 0.000 0.0 0.0 0 0.0 0.0 0.0 | objective" in shortlist_text
    assert "preset_id" not in shortlist_text
    task = json.loads(request.canonical["input"][1]["content"][-1]["text"])["task"]
    assert task == {
        "choose_one_major": True, "max_proposals": 4, "min_proposals": 2,
        "offered_majors": ["small-major"],
    }


def test_global_prompt_rejects_invalid_proposal_bounds(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    source = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "a" * 64}
    with pytest.raises(ValueError, match="proposal bounds"):
        global_request(
            config.terra, source, _diagnosis(), {"major": [{"preset_id": "p0"}]},
            min_proposals=4, max_proposals=2,
        )


def test_prefix_cache_key_is_compact_and_fully_isolated() -> None:
    base = {"input": [{"role": "developer", "content": [{"text": "stable"}]}]}
    key = prefix_cache_key(
        model_family="gpt-5.6-terra", prompt_revision="revision", stage="local_propose",
        prefix=base,
    )
    assert len(key) <= 64
    assert len({
        key,
        prefix_cache_key(model_family="other", prompt_revision="revision",
                         stage="local_propose", prefix=base),
        prefix_cache_key(model_family="gpt-5.6-terra", prompt_revision="other",
                         stage="local_propose", prefix=base),
        prefix_cache_key(model_family="gpt-5.6-terra", prompt_revision="revision",
                         stage="global_propose", prefix=base),
        prefix_cache_key(model_family="gpt-5.6-terra", prompt_revision="revision",
                         stage="local_propose", prefix={"input": []}),
    }) == 5


def test_repair_and_sibling_prompt_prefixes_are_stable(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    image_a = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
               "uri": "sha256://" + "a" * 64}
    image_b = {"sha256": "b" * 64, "media_type": "image/jpeg", "size": 10,
               "uri": "sha256://" + "b" * 64}
    masks = [{"mask_id": "m1", "family": "band", "direction": "vertical",
              "effective_alpha_mean": .5, "subject_high_coverage": 1.0,
              "subject_support_coverage": 1.0, "center_hint": "center"}]
    lut = [{"preset_id": "p1", "achievable_bins": ["subtle"], "caption": "objective",
            "fingerprint": {name: 0.0 for name in FINGERPRINT_FIELDS},
            "mask_reach_de": 5.5}]
    source = {"sha256": "c" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "c" * 64}
    initial = local_request(
        config.terra, source, image_a, {"fact": 1}, {"bin": "natural"}, {"delta_e": 4},
        lut, masks, max_proposals=3,
    )
    repair = local_request(
        config.terra, source, image_a, {"fact": 1}, {"bin": "natural"}, {"delta_e": 4},
        lut, masks, max_proposals=3, excluded_row_indices=[0],
        repair={"defects": ["halo"]},
    )
    sibling = local_request(
        config.terra, source, image_b, {"fact": 1}, {"bin": "bold"}, {"delta_e": 5},
        lut, masks, max_proposals=3,
    )
    assert initial.prompt_cache_key == repair.prompt_cache_key
    assert initial.prompt_cache_key == sibling.prompt_cache_key
    # B8 item 4: the local table documents and carries the mask-conditioned reach.
    shortlist_block = json.dumps(initial.canonical["input"], ensure_ascii=False)
    assert LOCAL_SHORTLIST_NOTE in shortlist_block
    assert "objective | 5.50" in shortlist_block
    # B11 item 4 (R7.2): two images in source -> global_after order, the source one
    # inside the stable prefix and global_after first in the tail.
    content = initial.canonical["input"][1]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert [item["artifact_sha256"] for item in images] == ["c" * 64, "a" * 64]
    # The developer text is split off, so the user turn is:
    # 0 intent guide, 1 diagnosis, 2 source image, 3 shortlist, 4 global_after.
    assert content[2] == {
        "type": "input_image", "artifact_sha256": "c" * 64,
        "media_type": "image/jpeg", "detail": "low",
        "encoding": dict(prompts_module.IMAGE_ENCODING),
    }
    assert "Two images are shown" in prompts_module._LOCAL_RULES
    # Stable prefix: rules, guide, diagnosis, source image, shortlist; global_after
    # starts the tail and is the only thing that differs between two siblings.
    assert initial.canonical["input"][1]["content"][:4] == \
        sibling.canonical["input"][1]["content"][:4]
    assert initial.canonical["input"][1]["content"][4] != \
        sibling.canonical["input"][1]["content"][4]
    verify_a = validation_request(config.validator, image_a, image_b, image_a, {"x": 1})
    verify_b = validation_request(config.validator, image_a, image_b, image_b, {"x": 2})
    assert verify_a.prompt_cache_key == verify_b.prompt_cache_key
    assert verify_a.canonical["input"][0] == verify_b.canonical["input"][0]


def test_exact_cache_invalid_attempt_and_concurrent_singleflight(tmp_path: Path) -> None:
    store = SQLiteAuditStore(tmp_path / "audit.sqlite")
    store.setup()
    cache = ExactResponseCache(store, lease_seconds=2)
    spec = RequestSpec({"endpoint_identity": "lane", "model": "m", "stage": "x",
                        "input": [{"role": "user", "content": [1]}]}, "key")
    calls = 0
    lock = threading.Lock()

    def sender(_spec: RequestSpec, _attempt: int):
        nonlocal calls
        with lock:
            calls += 1
            current = calls
        time.sleep(0.08)
        return {"ok": current > 1, "usage": {"input_tokens": 1}}

    def validate(row):
        return None if row["ok"] else "bad_schema"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(
            lambda _: cache.execute(spec, sender, validate, attempts=3), range(2)
        ))
    assert calls == 2  # one invalid redraw plus one valid call, never duplicated by waiter
    assert sum(result.cache_hit for result in results) == 1
    attempts = store.export_tables()["api_attempt"]
    assert [row["valid"] for row in attempts] == [0, 1]
    events = Counter(row["event"] for row in store.export_tables()["api_cache_event"])
    assert events["miss"] == 1 and events["wait"] == 1 and events["hit"] == 1


def test_pending_request_lease_can_be_taken_over(tmp_path: Path) -> None:
    store = SQLiteAuditStore(tmp_path / "audit.sqlite")
    store.setup()
    manifest = {"endpoint_identity": "lane", "model": "m"}
    assert store.acquire_request("a" * 64, manifest, "one", 30)["action"] == "owner"
    store._conn().execute(  # white-box clock control avoids a real 30-second wait
        "UPDATE api_request SET lease_expires_at=0 WHERE request_hash=?", ("a" * 64,)
    )
    assert store.acquire_request("a" * 64, manifest, "two", 30)["action"] == "owner"


def test_retry_classification_stops_on_400_and_retries_429(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = SQLiteAuditStore(tmp_path / "audit.sqlite")
    store.setup()
    cache = ExactResponseCache(store)
    monkeypatch.setattr("dataset_build.agent_loop.api_cache.time.sleep", lambda _delay: None)
    calls: Counter[str] = Counter()

    def bad_request(_spec, _attempt):
        calls["400"] += 1
        raise TransportError("400:invalid_request")

    spec_400 = RequestSpec({"case": "400"}, "retry-400")
    with pytest.raises(CacheExhausted) as stopped:
        cache.execute(spec_400, bad_request, lambda _response: None, attempts=3)
    assert stopped.value.attempts == 1
    assert calls["400"] == 1

    def rate_limited(_spec, _attempt):
        calls["429"] += 1
        if calls["429"] == 1:
            raise TransportError("429:rate_limit", retry_after=0)
        return {"ok": True}

    spec_429 = RequestSpec({"case": "429"}, "retry-429")
    result = cache.execute(spec_429, rate_limited, lambda _response: None, attempts=3)
    assert not result.cache_hit
    assert calls["429"] == 2


def test_postgres_lease_type_and_checkpoint_saver_sharding() -> None:
    assert "lease_expires_at DOUBLE PRECISION" in _POSTGRES_SCHEMA
    assert "lease_expires_at REAL" not in _POSTGRES_SCHEMA
    savers = [SimpleNamespace(serde=None, config_specs=[]) for _ in range(8)]
    router = ShardedCheckpointSaver(savers)
    routed = [router._for_config({"configurable": {"thread_id": f"thread-{i}"}})
              for i in range(64)]
    assert len({id(saver) for saver in routed}) > 1
    config = {"configurable": {"thread_id": "stable-thread"}}
    assert router._for_config(config) is router._for_config(config)


def test_accepted_artifacts_land_and_rematerialize_from_archive(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    catalog = tmp_path / "catalog.sqlite3"
    artifacts = ArtifactStore(
        tmp_path / "ramstage", recorder=audit.record_artifact, catalog_db=catalog
    )
    accepted = artifacts.put_json({"accepted": True}, retention="accepted")
    quarantine = artifacts.put_json({"rejected": True}, retention="quarantine")
    archive = tmp_path / "archive"
    archive.mkdir()
    manager = ArtifactLandingManager(
        artifacts, audit, ArtifactLandingConfig(
            enabled=True, archive_root=archive,
            archive_group="artifacts/test", plan_root=tmp_path / "plans",
            meta_staging=tmp_path / "meta", catalog_db=catalog,
            watermark_bytes=1, min_interval_seconds=0, min_groups=1,
            free_bytes_floor=0,
        ),
    )
    result = manager.maybe_land(completed_groups=1)
    assert result is not None and result["accepted_artifacts"] == 1
    assert not artifacts.local_path(accepted.sha256).exists()
    assert artifacts.local_path(quarantine.sha256).is_file()

    materialized = artifacts.path_for(accepted)
    assert json.loads(materialized.read_text(encoding="utf-8")) == {"accepted": True}
    assert manager.maybe_land(force=True) is None
    assert not materialized.exists()
    published = list((archive / "artifacts/test").glob("batch-*"))
    assert len(published) == 1


def test_terminal_cleanup_keeps_shared_committed_sha_and_purges_rejected(
    tmp_path: Path,
) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    kept = artifacts.put_json({"render": "kept"}, retention="quarantine")
    rejected = artifacts.put_json({"render": "rejected"}, retention="quarantine")
    services = SimpleNamespace(artifacts=artifacts, audit=audit)
    _discard_uncommitted_renders(services, [{
        "global_render": {"artifact": kept.to_dict()},
        "leaves": [{"render": {"artifact": rejected.to_dict()}}],
    }], {kept.sha256})
    assert artifacts.local_path(kept.sha256).is_file()
    assert not artifacts.local_path(rejected.sha256).exists()
    rows = {row["sha256"]: row for row in audit.export_tables()["artifact_record"]}
    assert rows[rejected.sha256]["retention"] == "purged"


def test_terminal_cleanup_keeps_a_blob_another_source_committed(tmp_path: Path) -> None:
    """C1b item 12: `protected_sha256` only knows the current source's committed rows.

    Artifacts are content-addressed, so an identical render committed by another source
    (or an earlier pass) is `accepted` in `artifact_record` and must survive this
    source's cleanup, even though this source never protected it.
    """
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    shared = artifacts.put_json({"render": "shared"}, retention="quarantine")
    # another source committed the same content-addressed blob
    artifacts.promote(shared)
    mine = artifacts.put_json({"render": "mine"}, retention="quarantine")
    services = SimpleNamespace(artifacts=artifacts, audit=audit)
    _discard_uncommitted_renders(services, [{
        "global_render": {"artifact": shared.to_dict()},
        "leaves": [{"render": {"artifact": mine.to_dict()}}],
    }], set())
    assert artifacts.local_path(shared.sha256).is_file()
    assert not artifacts.local_path(mine.sha256).exists()
    rows = {row["sha256"]: row for row in audit.export_tables()["artifact_record"]}
    assert rows[shared.sha256]["retention"] == "accepted"
    assert rows[mine.sha256]["retention"] == "purged"


def test_purge_claim_is_won_exactly_once(tmp_path: Path) -> None:
    """C1b item 12: `mark_artifact_purged` is the claim, and it is not re-entrant."""
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    ref = artifacts.put_json({"render": "once"}, retention="quarantine")
    assert audit.mark_artifact_purged(ref.sha256) is True
    assert audit.mark_artifact_purged(ref.sha256) is False
    accepted = artifacts.put_json({"render": "accepted"}, retention="accepted")
    assert audit.mark_artifact_purged(accepted.sha256) is False
    assert audit.mark_artifact_purged("f" * 64) is False


@pytest.mark.parametrize("area,expected", [
    ("small", {"radial": 3, "band": 3, "linear": 2}),
    ("large", {"semantic": 1, "radial": 2, "band": 2, "linear": 2}),
])
def test_mask_v2_bank_and_sibling_contract(tmp_path: Path, area: str,
                                           expected: dict[str, int]) -> None:
    _source, subject = _source_files(tmp_path, area=area)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts,
    )
    assert Counter(row["family"] for row in masks) == expected
    validate_mask_bank(masks, float(masks[0]["subject_area"]))
    first, remaining = allocate_mask_packets(masks, "global-1")
    assert len(first) == 3 and len({row["mask_id"] for row in first}) == 3
    assert len({row["family"] for row in first}) >= 2
    assert sum(row["family"] == "semantic" for row in first) <= 1
    projections = [np.asarray(row["alpha_projection"], dtype=np.float32) for row in first]
    assert all(float(np.abs(left - right).mean()) >= 0.02
               for left, right in itertools.combinations(projections, 2))
    assert {row["mask_id"] for row in first}.isdisjoint(
        row["mask_id"] for row in remaining
    )
    single, single_remaining = allocate_mask_packets(
        masks, "global-smoke", packet_size=1
    )
    assert len(single) == 1 and len(single_remaining) == len(masks) - 1
    duplicate = [dict(row) for row in masks]
    duplicate[1]["mask_id"] = duplicate[0]["mask_id"]
    with pytest.raises(CandidateError, match="unique"):
        validate_mask_bank(duplicate, float(masks[0]["subject_area"]))


@pytest.mark.parametrize("area", ["small", "large"])
def test_subject_band_minimum_area_gate(tmp_path: Path, area: str, monkeypatch) -> None:
    """B7 item 4: subject band slots below the half_area floor are dropped, counted,
    and rejected by the runtime assertion; radial/linear/semantic are untouched."""
    _source, subject = _source_files(tmp_path, area=area)
    assert SUBJECT_BAND_GATE == {"half_area_min": 0.28}
    artifacts = ArtifactStore(tmp_path / "artifacts")
    kept: list[dict[str, Any]] = []
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts, diagnostics=kept,
    )
    bands = [row for row in masks if row["family"] == "band"]
    assert bands and all(
        float(row["half_area"]) >= SUBJECT_BAND_GATE["half_area_min"] for row in bands
    )
    assert not [row for row in kept if row["reason"] == "subject_band_min_area"]

    monkeypatch.setitem(SUBJECT_BAND_GATE, "half_area_min", 0.99)
    dropped: list[dict[str, Any]] = []
    thinned = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=ArtifactStore(tmp_path / "a2"),
        diagnostics=dropped,
    )
    assert not [row for row in thinned if row["family"] == "band"]
    slots = [row for row in dropped if row["reason"] == "subject_band_min_area"]
    assert len(slots) == len(bands)
    assert all(row["role"] == "subject" and row["family"] == "band" for row in slots)
    assert all(row["half_area"] < row["half_area_min"] for row in slots)
    # Every other family keeps its exact count and the thinned bank still validates.
    assert Counter(row["family"] for row in thinned) == \
        Counter(row["family"] for row in masks if row["family"] != "band")
    validate_mask_bank(thinned, float(masks[0]["subject_area"]))
    with pytest.raises(CandidateError, match="violates half_area_min"):
        validate_mask_bank(masks, float(masks[0]["subject_area"]))


@pytest.mark.parametrize("area", ["small", "large"])
def test_band_geometry_gate_applies_to_both_roles(
    tmp_path: Path, area: str, monkeypatch
) -> None:
    """B8 item 2: every band slot, in either role, must clear the minimum narrow-side
    width and the aspect ceiling; the slabs of a complement band are measured one by
    one, so a thin background stripe is dropped like a thin subject band."""
    _source, subject = _source_files(tmp_path, area=area)
    assert BAND_GEOMETRY_GATE == {"min_width_short": 0.18, "aspect_max": 4.0}
    dropped: list[dict[str, Any]] = []
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=ArtifactStore(tmp_path / "a1"),
        include_background=True, diagnostics=dropped,
    )
    bands = [row for row in masks if row["family"] == "band"]
    assert bands and {row["role"] for row in bands} == {"subject", "background"}
    for row in bands:
        assert float(row["band_min_width"]) >= BAND_GEOMETRY_GATE["min_width_short"]
        assert float(row["band_aspect"]) <= BAND_GEOMETRY_GATE["aspect_max"]
    # The large-subject fixture reproduces the thin background stripe R4 is about: its
    # horizontal complement band leaves two 0.219-wide slabs at aspect 4.55.
    aspect_drops = [row for row in dropped if row["reason"] == "band_aspect"]
    assert len(aspect_drops) == (1 if area == "large" else 0)
    assert all(row["role"] == "background" and row["family"] == "band"
               for row in aspect_drops)
    assert all(row["band_aspect"] > row["band_aspect_max"] for row in aspect_drops)

    monkeypatch.setitem(BAND_GEOMETRY_GATE, "min_width_short", 0.99)
    thin: list[dict[str, Any]] = []
    thinned = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=ArtifactStore(tmp_path / "a2"),
        include_background=True, diagnostics=thin,
    )
    assert not [row for row in thinned if row["family"] == "band"]
    width_drops = [row for row in thin if row["reason"] == "band_min_width"]
    assert len(width_drops) == len(bands) + len(aspect_drops)
    assert all(row["band_min_width"] < row["band_min_width_min"] for row in width_drops)
    validate_mask_bank(thinned, float(masks[0]["subject_area"]))
    with pytest.raises(CandidateError, match="violates min_width_short"):
        validate_mask_bank(masks, float(masks[0]["subject_area"]))


def test_band_geometry_reading_measures_each_slab_of_a_complement() -> None:
    """B8 item 2: a plain band is one slab, its complement is two, and the reading is
    the worst slab of the two."""
    geometry = {"kind": "band", "center_x": 0.5, "center_y": 0.5, "angle": 0.0,
                "half_short": 0.25}
    band = _evaluate_geometry((64, 96), geometry)
    complement = _evaluate_geometry((64, 96), {**geometry, "complement": True})
    inner = band_geometry_reading(band, geometry)
    outer = band_geometry_reading(complement, geometry)
    assert inner["band_min_width"] == pytest.approx(32 / 64, abs=0.02)
    assert inner["band_aspect"] == pytest.approx(96 / 32, abs=0.1)
    # Two 16px slabs at 96px long: half the width, twice the aspect of the band.
    assert outer["band_min_width"] == pytest.approx(16 / 64, abs=0.02)
    assert outer["band_aspect"] == pytest.approx(96 / 16, abs=0.2)
    assert band_geometry_reading(np.zeros((64, 96), dtype=np.float32), geometry) == \
        {"band_min_width": 0.0, "band_aspect": float("inf")}


@pytest.mark.parametrize("area", ["small", "large"])
def test_mask_v2_background_role_gate_and_role_packet(tmp_path: Path, area: str) -> None:
    _source, subject = _source_files(tmp_path, area=area)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    diagnostics: list[dict[str, Any]] = []
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts,
        include_background=True, diagnostics=diagnostics,
    )
    subject_masks = [row for row in masks if row["role"] == "subject"]
    background = [row for row in masks if row["role"] == "background"]
    assert background, diagnostics
    # B8 item 3 (R4.2): the background effective-area floor is 0.12 -> 0.20.
    assert BACKGROUND_ROLE_GATE["half_area_min"] == 0.20
    assert Counter(row["family"] for row in subject_masks) == (
        {"radial": 3, "band": 3, "linear": 2} if area == "small"
        else {"semantic": 1, "radial": 2, "band": 2, "linear": 2}
    )
    for row in background:
        assert row["subject_alpha_mean"] <= BACKGROUND_ROLE_GATE["subject_alpha_mean_max"]
        assert row["subject_high_coverage"] <= \
            BACKGROUND_ROLE_GATE["subject_high_coverage_max"]
        assert row["background_alpha_mean"] >= \
            BACKGROUND_ROLE_GATE["background_alpha_mean_min"]
        assert row["half_area"] >= BACKGROUND_ROLE_GATE["half_area_min"]
        assert row["center_hint"] == "the background around the main subject"
        assert region_descriptor(row).startswith("background:")
    validate_mask_bank(masks, float(masks[0]["subject_area"]))
    assert mask_summary(background[0])["role"] == "background"
    assert mask_summary(subject_masks[0])["role"] == "subject"

    first, remaining, note = allocate_role_packets(masks, "global-1")
    assert len(first) == 3 and len({row["mask_id"] for row in first}) == 3
    assert note["fallback"] is None
    # B6 item 2: the default sibling mix is exactly 2 subject + 1 background.
    assert note["role_target"] == {"subject": 2, "background": 1}
    assert note["role_counts"] == {"subject": 2, "background": 1}
    assert note["role_target_met"] is True
    assert {row["mask_id"] for row in first}.isdisjoint(
        row["mask_id"] for row in remaining
    )
    assert len(first) + len(remaining) == len(masks)

    subject_only, subject_remaining, subject_note = allocate_role_packets(
        subject_masks, "global-1"
    )
    assert subject_note["fallback"] == "background_infeasible"
    assert subject_note["role_counts"] == {"subject": 3, "background": 0}
    assert subject_note["role_target_met"] is False
    assert len({row["family"] for row in subject_only}) >= 2
    assert len(subject_only) + len(subject_remaining) == len(subject_masks)

    # Only one subject mask left: the 2:1 target is unreachable, so the packet
    # relaxes to the >=1 subject / >=1 background rule instead of failing.
    thin = [subject_masks[0], background[0], background[-1]]
    thin_first, _thin_remaining, thin_note = allocate_role_packets(thin, "global-1")
    assert thin_note["fallback"] is None
    assert thin_note["role_target_met"] is False
    assert thin_note["role_counts"] == {"subject": 1, "background": 2}
    assert len(thin_first) == 3


def test_background_gate_assertions_reject_each_column(tmp_path: Path) -> None:
    _source, subject = _source_files(tmp_path, area="small")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts, include_background=True,
    )
    area = float(masks[0]["subject_area"])
    index = next(i for i, row in enumerate(masks) if row["role"] == "background")
    row = masks[index]

    def doctored(**changes: Any) -> list[dict[str, Any]]:
        return [dict(item) if i != index else {**item, **changes}
                for i, item in enumerate(masks)]

    validate_mask_bank(masks, area)
    for column, value in (
        ("subject_alpha_mean", BACKGROUND_ROLE_GATE["subject_alpha_mean_max"] + 0.01),
        ("subject_high_coverage",
         BACKGROUND_ROLE_GATE["subject_high_coverage_max"] + 0.01),
        ("background_alpha_mean",
         BACKGROUND_ROLE_GATE["background_alpha_mean_min"] - 0.01),
        ("half_area", BACKGROUND_ROLE_GATE["half_area_min"] - 0.01),
    ):
        with pytest.raises(CandidateError, match=column):
            validate_mask_bank(doctored(**{column: value}), area)
    stripped = [dict(item) for item in masks]
    stripped[index].pop("half_area")
    with pytest.raises(CandidateError, match="missing"):
        validate_mask_bank(stripped, area)
    with pytest.raises(CandidateError, match="role"):
        validate_mask_bank(doctored(role="foreground"), area)
    assert row["role"] == "background"


def test_prompt_registry_key_set_is_the_explicit_frozen_list() -> None:
    """C1b item 1: adding or deleting a registry key must be a deliberate edit.

    The whole point of `prompt_revision_fingerprint()` is that a constant which changes
    what the model is asked or offered cannot silently stay out of the revision. A key
    set that only exists inside the function body is unauditable, so the explicit list
    below is the contract and this assertion is its runtime guard.
    """
    assert sorted(prompt_registry()) == sorted(PROMPT_REGISTRY_KEYS)
    # B12 adds three keys: 41 -> 44. E4 adds two more: 44 -> 46.
    assert len(set(PROMPT_REGISTRY_KEYS)) == len(PROMPT_REGISTRY_KEYS) == 46
    for key in ("source_histogram", "segment_fingerprint_histogram",
                "histogram_match_gate", "histogram_board", "board_image_encoding"):
        assert key in PROMPT_REGISTRY_KEYS


@pytest.mark.parametrize("key", [
    "intent_fingerprint_gate", "global_delta_e_targets", "subject_headroom_gate",
    "role_packet_target", "background_family_counts", "global_bin_quota",
    "axis_scales", "axis_weights", "keyword_axes", "measure_sample_pixels",
    "band_segment_lum_threshold",
])
def test_c1b_registry_additions_carry_their_live_constant_values(key: str) -> None:
    """The eleven C1b item-1 additions are present and non-empty, not placeholders."""
    value = prompt_registry()[key]
    assert value or value == 0
    assert json.dumps(value, sort_keys=True)


def test_registry_is_the_only_input_of_the_prompt_fingerprint(monkeypatch) -> None:
    payload = json.dumps(
        prompt_registry(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert hashlib.sha256(payload).hexdigest() == prompt_revision_fingerprint()


@pytest.mark.parametrize("module,name,probe", [
    ("models", "INTENT_FINGERPRINT_GATE", {"dL_positive_min": 9.0}),
    ("models", "GLOBAL_DELTA_E_TARGETS", {"natural": (1.0, 2.0, False)}),
    ("models", "SUBJECT_HEADROOM_GATE", {"near_clip_level": 0.5}),
    ("candidates", "ROLE_PACKET_TARGET", {"subject": 9}),
    ("candidates", "BACKGROUND_FAMILY_COUNTS", {"radial": 9}),
    ("candidates", "GLOBAL_BIN_QUOTA", 9),
    ("direction_match", "AXIS_SCALES", {"cast_a": 9.0}),
    ("direction_match", "AXIS_WEIGHTS", {"cast_a": 9.0}),
    ("direction_match", "KEYWORD_AXES", ((("probe",), "cast_a", 1.0),)),
    ("direction_match", "MEASURE_SAMPLE_PIXELS", 9),
    ("segment_fingerprints", "BAND_SEGMENT_LUM_THRESHOLD", 9.0),
])
def test_c1b_constants_enter_the_prompt_revision_chain(
    monkeypatch, module: str, name: str, probe: object
) -> None:
    """C1b item 1: each newly registered constant moves the fingerprint when it moves."""
    before = prompt_revision_fingerprint()
    if module in {"models", "segment_fingerprints"}:
        # `prompts` binds these at import time, so the probe patches the bound name.
        monkeypatch.setattr(prompts_module, name, probe)
    elif module == "direction_match":
        monkeypatch.setattr(prompts_module, name, probe)
    else:
        monkeypatch.setattr(candidates_module, name, probe)
    assert prompt_revision_fingerprint() != before


def test_mask_summary_revision_enters_prompt_revision_chain(monkeypatch) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, "MASK_SUMMARY_REVISION", "mask-summary-probe")
    assert prompt_revision_fingerprint() != before


@pytest.mark.parametrize("name,probe", [
    ("CANDIDATE_SERIALIZATION_REVISION", "candidate-probe"),
    ("DIAGNOSE_PROMPT_REVISION", "diagnose-probe"),
])
def test_candidate_and_diagnose_revisions_enter_prompt_revision_chain(
    monkeypatch, name: str, probe: str
) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, name, probe)
    assert prompt_revision_fingerprint() != before


@pytest.mark.parametrize("name,probe", [
    ("LOCAL_PACKET_ROW_LIMIT", 3),
    ("INTENT_BIN_QUOTA", 2),
])
def test_packet_row_budget_enters_prompt_revision_chain(
    monkeypatch, name: str, probe: int
) -> None:
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, name, probe)
    assert prompt_revision_fingerprint() != before


def test_local_strength_targets_are_the_b8_floored_per_intent_ladders() -> None:
    high = {
        "subtle": (4.0, 4.5, False),
        "natural": (4.5, 5.5, False),
        "strong": (5.5, 6.5, True),
    }
    # B8 item 1 / B11 item 5 (R7.4): every band that sat at 3.5 moves to the 4.0 floor.
    low = {
        "subtle": (4.0, 4.5, False),
        "natural": (4.0, 4.5, False),
        "strong": (4.5, 5.5, True),
    }
    assert LOCAL_DELTA_E_LADDERS == {"high": high, "low": low}
    # B7 item 1: sat_boost / background_control / zonal_contrast keep the B6 band;
    # luminance_pop / hue_shift fall back to the pre-B6 band.
    assert {intent: LOCAL_DELTA_E_TARGETS[intent] for intent in ACTIVE_LOCAL_INTENTS} \
        == {
            "sat_boost": high, "background_control": high, "zonal_contrast": high,
            "luminance_pop": low, "hue_shift": low,
            # B11 item 3: both new intents start on the conservative `low` ladder.
            "contrast_boost": low, "cast_correction_local": low,
        }
    assert set(LOCAL_DELTA_E_TARGETS) == set(LOCAL_INTENTS)
    assert local_ladder("hue_shift") == low
    with pytest.raises(KeyError, match="unknown local intent"):
        local_ladder("vibe")
    assert local_target_center("sat_boost", "strong") == 6.0
    assert local_target_center("luminance_pop", "strong") == 5.0
    assert local_target_center("luminance_pop", "strong", capped=True) == 4.25
    assert local_target_center("luminance_pop", "subtle", capped=True) == 4.25
    assert INTENT_FINGERPRINT_GATE["highlight_dL_negative_max"] == -2.0
    assert BACKGROUND_ROLE_GATE["applied_background_alpha_mean_min"] == 0.18
    assert INTENT_BIN_QUOTA == 1 and LOCAL_PACKET_ROW_LIMIT == 4


def test_luma_soft_cap_shifts_the_luminance_pop_ladder_down_one_bin() -> None:
    # B7 item 2: pre-registered numbers and the one-bin downshift they trigger.
    assert LUMA_SOFT_CAP == {"p99_luma_max": 0.96}
    assert LUMA_SOFT_CAP_INTENTS == frozenset({"luminance_pop"})
    assert luma_capped("luminance_pop", 0.9601) is True
    assert luma_capped("luminance_pop", 0.96) is False
    assert luma_capped("luminance_pop", None) is False
    assert luma_capped("hue_shift", 0.99) is False
    assert resolve_local_target("luminance_pop", "strong", 0.97) == \
        (4.0, 4.5, False, "natural", True)
    # B8 item 1: the downshift can no longer open a band below the visibility floor;
    # on the `low` ladder natural and subtle now name the same band.
    assert resolve_local_target("luminance_pop", "natural", 0.97) == \
        (4.0, 4.5, False, "subtle", True)
    # `subtle` has no lower bin: the band is unchanged but the row is still audited.
    assert resolve_local_target("luminance_pop", "subtle", 0.97) == \
        (4.0, 4.5, False, "subtle", True)
    assert resolve_local_target("luminance_pop", "strong", 0.95) == \
        (4.5, 5.5, True, "strong", False)
    assert resolve_local_target("sat_boost", "strong", 0.99) == \
        (5.5, 6.5, True, "strong", False)


def test_local_visibility_floor_covers_every_active_intent(monkeypatch) -> None:
    """B8 item 1 + B11 item 5: the transition floor and its runtime assertion."""
    assert LOCAL_VISIBILITY_FLOOR == 4.0  # R7.4: 3.5 -> 4.0
    assert LOCAL_DELTA_E_LADDERS["high"]["subtle"] == (4.0, 4.5, False)
    assert LOCAL_DELTA_E_LADDERS["low"]["subtle"] == (4.0, 4.5, False)
    assert MASK_REACH_GATE["reach_de_min"] == LOCAL_VISIBILITY_FLOOR
    assert min(
        low for intent in ACTIVE_LOCAL_INTENTS
        for low, _high, _inclusive in LOCAL_DELTA_E_TARGETS[intent].values()
    ) == LOCAL_VISIBILITY_FLOOR
    assert_local_visibility_floor()
    monkeypatch.setitem(
        LOCAL_DELTA_E_TARGETS["luminance_pop"], "subtle", (3.5, 4.5, False)
    )
    with pytest.raises(ValueError, match="local visibility floor"):
        assert_local_visibility_floor()


def test_local_calibration_rejects_a_leaf_below_the_visibility_floor(
    tmp_path: Path, monkeypatch
) -> None:
    """B8 item 1 / B11 item 5: an edit that cannot reach 4.0 on the mask is rejected
    instead of kept weak, and the soft cap cannot open a band below the floor."""
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    alpha = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    # C1b item 7: a local mask without `subject_artifact` is now a hard render error,
    # so the fixture carries one.
    mask = {"mask_id": "m", "alpha_artifact": alpha.to_dict(),
            "subject_artifact": alpha.to_dict()}

    def calibrate(delta: list[float], *, strength_bin: str = "subtle", **kwargs):
        return StrengthCalibrator(
            _ArrayRenderer(np.asarray(delta, dtype=np.float32)), _catalog(1), artifacts,
            audit, renderer_revision="r1", search_steps=17, clip_fraction_max=0.2,
        ).calibrate_local(
            source_sha256="s", input_artifact=source.to_dict(), preset_id="p0",
            mask=mask, strength_bin=strength_bin, **kwargs
        )

    # dE 3.845 at full strength is below the R7.4 floor, so the leaf is rejected.
    with pytest.raises(RenderError, match="local_visibility_floor") as weak:
        calibrate([0.028, 0.0, 0.0], branch_id="l-weak", intent="luminance_pop")
    assert weak.value.code == "local_visibility_floor"
    rejected = [row for row in audit.export_tables()["render_record"]
                if row["status"] == "local_visibility_floor"]
    assert len(rejected) == 1
    assert rejected[0]["metrics_json"]["delta_e_reached_max"] < LOCAL_VISIBILITY_FLOOR
    assert rejected[0]["metrics_json"]["local_visibility_floor"] == \
        LOCAL_VISIBILITY_FLOOR
    # dE 4.348 at full strength: inside the floored [4.0, 4.5) subtle band, so it renders.
    accepted = calibrate([0.032, 0.0, 0.0], branch_id="l-ok", intent="luminance_pop")
    assert accepted["metrics"]["delta_e"] >= LOCAL_VISIBILITY_FLOOR
    # A LUT that clears the floor but misses a higher band is still an ordinary target
    # miss (dE 5.35 at full strength against the 5.5-6.5 `strong` band of sat_boost).
    with pytest.raises(RenderError, match="strength_target_unreachable"):
        calibrate([0.02, -0.01, 0.01], branch_id="l-miss", intent="sat_boost",
                  strength_bin="strong")
    assert not [row for row in audit.export_tables()["render_record"]
                if row["branch_id"] == "l-miss"
                and row["status"] == "local_visibility_floor"]

    # The soft cap keeps its downshift, but a downshifted band that opened below the
    # floor is rejected before any render happens.
    monkeypatch.setitem(
        LOCAL_DELTA_E_TARGETS["luminance_pop"], "subtle", (3.5, 4.5, False)
    )
    with pytest.raises(RenderError, match="target 3.5 < 4.0") as capped:
        StrengthCalibrator(
            _ArrayRenderer(), _catalog(1), artifacts, audit, renderer_revision="r1",
            search_steps=17, clip_fraction_max=0.2,
        ).calibrate_local(
            source_sha256="s", branch_id="l-capped", input_artifact=source.to_dict(),
            preset_id="p0", mask=mask, strength_bin="natural",
            intent="luminance_pop", subject_p99_luma=0.97,
        )
    assert capped.value.code == "local_visibility_floor"


def _intent_rows(catalog: LutCatalog, intent: str, reach: float, **kwargs):
    """`intent_rows` against a flat mask reach, the R7.1 local reach reading."""
    direction = _StubDirections().direction_value
    combined, correction = catalog.direction_scores(direction)
    return catalog.intent_rows(
        intent, kwargs.pop("exclude", ()), _diagnosis(), {}, "portrait",
        kwargs.pop("limit", LOCAL_PACKET_ROW_LIMIT),
        mask={"mask_id": "m"}, mask_reach=_StubMaskReach({}, default=reach),
        combined_scores=combined, correction_scores=correction, source_sha256="s",
        **kwargs,
    )


def test_local_reach_bins_follow_the_intent_ladder() -> None:
    """B7 item 1 + B11 item 2: the bins a local row advertises come from its
    mask-conditioned reach, read against the ladder of its own intent."""
    rows = _intent_rows(_catalog(1), "background_control", 4.6)["rows"]
    assert rows[0]["achievable_bins"] == ["subtle", "natural"]
    assert rows[0]["mask_reach_de"] == 4.6
    # The same mask reach reaches one bin further on the `low` ladder.
    intent_catalog = _intent_catalog()
    high_rows = _intent_rows(intent_catalog, "sat_boost", 4.6)["rows"]
    low_rows = _intent_rows(intent_catalog, "luminance_pop", 4.6)["rows"]
    assert [row["preset_id"] for row in high_rows] == ["sat"]
    assert [row["preset_id"] for row in low_rows] == ["bright"]
    assert high_rows[0]["achievable_bins"] == ["subtle", "natural"]
    assert low_rows[0]["achievable_bins"] == ["subtle", "natural", "strong"]
    # Below the floor nothing survives at all, on either ladder.
    assert _intent_rows(intent_catalog, "sat_boost", 3.99)["rows"] == []
    assert _intent_rows(intent_catalog, "sat_boost", 4.0)["rows"][0][
        "achievable_bins"
    ] == ["subtle"]


def test_diagnose_rules_are_the_v34_histogram_board_prompt() -> None:
    """E4: the production rules are the harness-validated v3.4 text, verbatim."""
    from dataset_build.tools.test_diagnose_v2 import V2_RULES

    rules = prompts_module._DIAGNOSE_RULES
    assert rules == V2_RULES
    assert "viewer's left and right" in rules
    assert "resolution" in rules
    assert "You receive two images" in rules
    assert "<axis> | <scope> | <observed state> | <move>" in rules
    assert prompts_module.DIAGNOSE_PROMPT_REVISION == "diagnose-v2-histogram-board-v3.4"


def test_diagnosis_request_carries_source_then_board(tmp_path: Path) -> None:
    """E4: image 1 = 512px source JPEG (low), image 2 = board PNG (high, passthrough)."""
    config = _write_config(tmp_path)
    source = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
              "uri": "sha256://" + "a" * 64}
    request = diagnosis_request(config.terra, source, _board_ref())
    assert request.canonical["input"][0]["content"][0]["text"] == \
        prompts_module._DIAGNOSE_RULES
    content = request.canonical["input"][1]["content"]
    images = [item for item in content if item["type"] == "input_image"]
    assert [item["artifact_sha256"] for item in images] == ["a" * 64, "b" * 64]
    assert images[0] == {
        "type": "input_image", "artifact_sha256": "a" * 64,
        "media_type": "image/jpeg", "detail": "low",
        "encoding": dict(prompts_module.IMAGE_ENCODING),
    }
    assert images[1] == {
        "type": "input_image", "artifact_sha256": "b" * 64,
        "media_type": "image/png", "detail": "high",
        "encoding": dict(BOARD_IMAGE_ENCODING),
    }
    assert BOARD_IMAGE_ENCODING == {"format": "png", "passthrough": True}
    # A different board is a different request.
    other = diagnosis_request(config.terra, source, _board_ref("c" * 64))
    assert other.request_hash != request.request_hash


@pytest.mark.parametrize("name,probe", [
    ("BOARD_REVISION", "board-probe"),
    ("BOARD_SIZE", (10, 10)),
    ("BOARD_BIN_GEOMETRY", {"l_bins": 3}),
    ("BOARD_IMAGE_ENCODING", {"format": "png", "passthrough": False}),
])
def test_board_constants_enter_the_prompt_revision_chain(
    monkeypatch, name: str, probe: object
) -> None:
    """E4: the board contract and its transport encoding are pre-registered."""
    before = prompt_revision_fingerprint()
    monkeypatch.setattr(prompts_module, name, probe)
    assert prompt_revision_fingerprint() != before


def test_background_geometry_pool_has_no_linear_family(tmp_path: Path) -> None:
    assert set(BACKGROUND_FAMILY_COUNTS) == {"radial", "band"}
    _source, subject = _source_files(tmp_path, area="small")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    diagnostics: list[dict[str, Any]] = []
    masks = build_mask_bank(
        subject, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts,
        include_background=True, diagnostics=diagnostics,
    )
    background = [row for row in masks if row["role"] == "background"]
    assert background
    assert {row["family"] for row in background} <= {"radial", "band"}
    assert all(row["family"] != "linear" for row in diagnostics)


def _intent_record(
    preset_id: str, segments: dict[str, dict[str, float]] | None = None, **summary: float
) -> LutRecord:
    return LutRecord(
        preset_id=preset_id, path=f"/unused/{preset_id}.cube", format="lut",
        name=preset_id, style_major=preset_id, style_minor="minor",
        scene_affinity=("portrait",), de_med=6.0, caption="objective response",
        per_probe={}, hsl_features={"summary": {
            "mid_gray_a": 0.0, "mid_gray_b": 0.0, "mid_gray_dL": 0.0,
            "sat_pct_mean": 0.0, "contrast_ratio": 1.0, "shadow_dL": 0.0,
            "highlight_dL": 0.0, "hue_rot_abs_max": 0.0, **summary,
        }, "bands": {}},
        segment_fingerprint=_segments() if segments is None else segments,
    )


# B11 item 3: `split` clears the `contrast_boost` segment domain and `chill` points
# against the warm residual the direction stub measures, so each new intent has exactly
# one member. Both carry `mid_gray_dL = 0.2`, which is outside every v1 intent domain.
def _intent_catalog() -> LutCatalog:
    return LutCatalog([
        _intent_record("bright", mid_gray_dL=6.0),
        _intent_record("sat", sat_pct_mean=20.0),
        _intent_record("hue", mid_gray_a=4.0, mid_gray_b=4.0),
        _intent_record("dark", mid_gray_dL=-4.0, highlight_dL=-3.0),
        _intent_record("cool", mid_gray_b=-4.0),
        _intent_record(
            "split", mid_gray_dL=0.2,
            segments=_segments(shadows={"dL": -3.0}, highlights={"dL": 3.0}),
        ),
        _intent_record(
            "chill", mid_gray_dL=0.2,
            segments=_segments(
                shadows={"cast_b": -4.0}, mids={"cast_b": -4.0},
                highlights={"cast_b": -4.0},
            ),
        ),
    ], segment_fingerprints_sha256="e" * 64)


class _StubDirections:
    """Duck-typed stand-in for `LocalDirectionProbe`: one warm residual per mask."""

    def __init__(self, correction: DirectionVector | None = None,
                 enhancement: DirectionVector | None = None,
                 tonal_weights: dict[str, float] | None = None) -> None:
        self.direction_value = MaskDirection(
            correction=correction or DirectionVector(
                cast_b=3.0, mode="correction", origin="measured"
            ),
            enhancement=enhancement or DirectionVector(mode="enhancement"),
            tonal_weights=tonal_weights or uniform_tonal_weights(),
        )
        self.masks: list[str] = []

    def direction(self, mask: Mapping[str, Any]) -> MaskDirection:
        self.masks.append(str(mask["mask_id"]))
        return self.direction_value


def test_intent_fingerprint_domains_gate_each_direction() -> None:
    catalog = _intent_catalog()
    fingerprints = {row.preset_id: row.fingerprint() for row in catalog.records}
    warm_global = fingerprints["hue"]
    assert warm_global["dL"] == 0.0 and warm_global["cast_hue"] == 45.0
    v1_intents = [
        intent for intent in LOCAL_INTENTS
        if intent not in {"contrast_boost", "cast_correction_local"}
    ]
    admitted = {
        intent: {
            preset_id for preset_id, fingerprint in fingerprints.items()
            if intent_admits(intent, fingerprint, warm_global)
        } for intent in v1_intents
    }
    assert admitted["luminance_pop"] == {"bright"}
    assert admitted["sat_boost"] == {"sat"}
    assert admitted["hue_shift"] == {"hue", "cool"}
    assert admitted["highlight_rescue"] == {"dark"}
    assert admitted["background_control"] == {"dark"}
    assert admitted["zonal_contrast"] == {"dark"}
    assert admitted["warm_cool_split"] == {"cool"}
    # B11 item 3: the two v7 intents are outside every v1 domain, and the two v1
    # domains that only read a small |dL| never see them either.
    assert {"split", "chill"}.isdisjoint(set().union(*admitted.values()))
    # The two paired intents read the global fingerprint they are contrasted with.
    dark_global = fingerprints["dark"]
    assert not intent_admits("zonal_contrast", fingerprints["dark"], dark_global)
    cool_global = fingerprints["cool"]
    assert not intent_admits("warm_cool_split", fingerprints["cool"], cool_global)
    with pytest.raises(CandidateError, match="unknown local intent"):
        intent_admits("vibe", fingerprints["dark"], warm_global)


def test_new_intent_domains_read_the_segment_fingerprint_and_the_residual() -> None:
    """B11 item 3 (R7.3): `contrast_boost` is a segmented-fingerprint domain and
    `cast_correction_local` is a measured-residual domain; neither may fall back."""
    catalog = _intent_catalog()
    records = {row.preset_id: row for row in catalog.records}
    direction = _StubDirections().direction_value
    _combined, correction = catalog.direction_scores(direction)
    admitted = {
        intent: {
            preset_id for preset_id, record in records.items()
            if intent_admits(
                intent, record.fingerprint(), records["hue"].fingerprint(),
                segment_fingerprint=record.segment_fingerprint,
                correction_match=correction[preset_id],
            )
        } for intent in ("contrast_boost", "cast_correction_local")
    }
    assert admitted["contrast_boost"] == {"split"}
    assert admitted["cast_correction_local"] == {"chill"}
    # Exactly at the pre-registered edges: the domain is closed on both sides.
    edge = _segments(
        shadows={"dL": INTENT_SEGMENT_GATE["contrast_shadow_dL_max"]},
        highlights={"dL": INTENT_SEGMENT_GATE["contrast_highlight_dL_min"]},
    )
    assert intent_admits(
        "contrast_boost", records["split"].fingerprint(), segment_fingerprint=edge
    )
    inside = _segments(
        shadows={"dL": INTENT_SEGMENT_GATE["contrast_shadow_dL_max"] + 0.01},
        highlights={"dL": INTENT_SEGMENT_GATE["contrast_highlight_dL_min"]},
    )
    assert not intent_admits(
        "contrast_boost", records["split"].fingerprint(), segment_fingerprint=inside
    )
    floor = INTENT_DIRECTION_GATE["cast_correction_match_min"]
    assert intent_admits(
        "cast_correction_local", records["chill"].fingerprint(), correction_match=floor
    )
    assert not intent_admits(
        "cast_correction_local", records["chill"].fingerprint(),
        correction_match=floor - 0.001,
    )
    # A LUT that reinforces the measured warm residual scores negative, never admitted.
    warm = _segments(
        shadows={"cast_b": 4.0}, mids={"cast_b": 4.0}, highlights={"cast_b": 4.0}
    )
    assert correction["chill"] > 0.0 > direction_match_score(
        direction.correction, warm, direction.tonal_weights
    )
    # Unwired inputs fail loudly instead of admitting everything.
    with pytest.raises(CandidateError, match="segment fingerprint"):
        intent_admits("contrast_boost", records["split"].fingerprint())
    with pytest.raises(CandidateError, match="measured mask direction"):
        intent_admits("cast_correction_local", records["chill"].fingerprint())


def test_online_retrieval_reads_the_whole_catalog_and_is_deterministic() -> None:
    """B11 item 2 (R7.1): the prefilter ranks every mounted preset by the combined
    direction score, keeps `prefilter_top_k`, and only then pays for the reach probe."""
    # 60 presets in the `background_control` domain, each a different cast_b response.
    records = [
        _intent_record(
            f"c{index:02d}",
            segments=_segments(
                shadows={"cast_b": float(index) - 30.0},
                mids={"cast_b": float(index) - 30.0},
                highlights={"cast_b": float(index) - 30.0},
            ),
        ) for index in range(60)
    ]
    catalog = LutCatalog(records, segment_fingerprints_sha256="e" * 64)
    top_k = int(LOCAL_ONLINE_RETRIEVAL["prefilter_top_k"])
    assert top_k == 50 and len(catalog.records) > top_k
    direction = _StubDirections().direction_value  # warm residual, correction mode
    combined, correction = catalog.direction_scores(direction)
    assert len(combined) == len(catalog.records) == len(correction)
    probe = _StubMaskReach({}, default=5.0)
    payload = catalog.intent_rows(
        "background_control", (), _diagnosis(), {}, "portrait",
        LOCAL_PACKET_ROW_LIMIT, mask={"mask_id": "m"}, mask_reach=probe,
        combined_scores=combined, correction_scores=correction, source_sha256="s",
    )
    assert payload["domain_size"] == 60
    assert payload["prefiltered"] == top_k
    # Only the prefilter survivors are measured, and the 10 worst-matching presets
    # (the warmest ones, which reinforce the measured warm residual) never are.
    assert len({preset for _mask, preset in probe.calls}) == top_k
    measured = {preset for _mask, preset in probe.calls}
    assert measured == {row.preset_id for row in records[:top_k]}
    repeat = catalog.intent_rows(
        "background_control", (), _diagnosis(), {}, "portrait",
        LOCAL_PACKET_ROW_LIMIT, mask={"mask_id": "m"},
        mask_reach=_StubMaskReach({}, default=5.0),
        combined_scores=combined, correction_scores=correction, source_sha256="s",
    )
    assert [row["preset_id"] for row in payload["rows"]] == \
        [row["preset_id"] for row in repeat["rows"]]
    assert payload["rows"] == repeat["rows"]
    # The excluded parent preset never enters the domain, let alone the probe.
    excluded = catalog.intent_rows(
        "background_control", ("c00",), _diagnosis(), {}, "portrait",
        LOCAL_PACKET_ROW_LIMIT, mask={"mask_id": "m"},
        mask_reach=_StubMaskReach({}, default=5.0),
        combined_scores=combined, correction_scores=correction, source_sha256="s",
    )
    assert excluded["domain_size"] == 59
    assert "c00" not in {row["preset_id"] for row in excluded["rows"]}


def test_local_direction_probe_measures_the_source_to_global_after_residual(
    tmp_path: Path
) -> None:
    """B11 items 2 + 4: the probe reads the same pair the prompt now shows, restricted
    to the mask, and caches one `MaskDirection` per mask."""
    artifacts = ArtifactStore(tmp_path / "artifacts")
    before = np.full((32, 32, 3), 0.5, dtype=np.float32)
    after = before.copy()
    after[:, :16, 2] = 0.35  # left half turns warm (Lab b* up)
    source = artifacts.put_image_array(before)
    global_after = artifacts.put_image_array(after)
    left = np.zeros((32, 32), dtype=np.float32)
    left[:, :16] = 1.0
    right = np.zeros((32, 32), dtype=np.float32)
    right[:, 16:] = 1.0
    masks = [
        {"mask_id": "left", "alpha_artifact": artifacts.put_alpha(left).to_dict()},
        {"mask_id": "right", "alpha_artifact": artifacts.put_alpha(right).to_dict()},
    ]
    probe = LocalDirectionProbe(
        artifacts, source_artifact=source.to_dict(),
        global_after_artifact=global_after.to_dict(),
        diagnosis={"enhancement_opportunities": ["low contrast, add depth"]},
    )
    warm = probe.direction(masks[0])
    cold = probe.direction(masks[1])
    assert warm.correction.mode == "correction" and warm.correction.origin == "measured"
    assert warm.correction.cast_b > 1.0  # the residual the global edit left behind
    assert cold.correction.is_zero()  # the untouched half has no residual at all
    assert sum(warm.tonal_weights.values()) == pytest.approx(1.0)
    # The enhancement half is the keyword direction of the frozen diagnosis, so it is
    # the same vector on every mask of one source.
    assert warm.enhancement.mode == "enhancement"
    assert warm.enhancement.contrast == -1.0
    assert warm.enhancement == cold.enhancement
    assert probe.direction(masks[0]) is warm  # cached per mask


def test_intent_packet_order_leans_on_the_two_new_intents() -> None:
    """B11 item 3 (R7.3): a sort key, not a quota - every active intent survives."""
    order = intent_packet_order(ACTIVE_LOCAL_INTENTS)
    assert set(order) == set(ACTIVE_LOCAL_INTENTS)
    assert order[:3] == ("cast_correction_local", "contrast_boost", "zonal_contrast")
    assert INTENT_PACKET_PRIORITY == {
        "cast_correction_local": 0, "contrast_boost": 1, "zonal_contrast": 2,
    }
    # Ties keep the declaration order of `ACTIVE_LOCAL_INTENTS`.
    assert order[3:] == tuple(
        intent for intent in ACTIVE_LOCAL_INTENTS
        if intent not in INTENT_PACKET_PRIORITY
    )
    # The two new intents serve both roles; every v1 intent keeps its single role.
    assert intent_serves_role("contrast_boost", "subject")
    assert intent_serves_role("contrast_boost", "background")
    assert intent_serves_role("cast_correction_local", "subject")
    assert intent_serves_role("cast_correction_local", "background")
    assert not intent_serves_role("luminance_pop", "background")
    assert INTENT_ROLE_DOMAINS["luminance_pop"] == ("subject",)


def test_intent_conditions_read_the_subject_headroom() -> None:
    calm = {"highlight_pressure": False, "subject_saturation_mean": 0.2}
    blown = {"highlight_pressure": True, "subject_saturation_mean": 0.2}
    assert intent_offered("luminance_pop", "subject", calm) == (True, "offered")
    assert intent_offered("luminance_pop", "subject", blown) == \
        (False, "highlight_headroom_exhausted")
    # B7 item 3: `highlight_rescue` is disabled, so the pressure trigger is never
    # reached on either headroom reading.
    assert intent_offered("highlight_rescue", "subject", calm) == \
        (False, "intent_disabled")
    assert intent_offered("highlight_rescue", "subject", blown) == \
        (False, "intent_disabled")
    assert intent_offered("sat_boost", "subject", {
        "highlight_pressure": False,
        "subject_saturation_mean": SUBJECT_HEADROOM_GATE["sat_mean_max"] + 0.01,
    }) == (False, "subject_saturation_high")
    assert intent_offered("background_control", "subject", calm) == \
        (False, "role_mismatch")
    assert intent_offered("luminance_pop", "background", calm) == \
        (False, "role_mismatch")
    # B6 item 1 / B7 item 3: both stay in the enum but never enter a packet, and the
    # active intent count is 5.
    assert DISABLED_LOCAL_INTENTS == frozenset(
        {"warm_cool_split", "highlight_rescue"}
    )
    assert set(LOCAL_INTENTS) >= DISABLED_LOCAL_INTENTS
    assert set(ACTIVE_LOCAL_INTENTS).isdisjoint(DISABLED_LOCAL_INTENTS)
    assert ACTIVE_LOCAL_INTENTS == (
        "luminance_pop", "sat_boost", "hue_shift", "zonal_contrast",
        "background_control", "contrast_boost", "cast_correction_local",
    )
    # B11 item 3 (R7.3): 5 -> 7.
    assert len(ACTIVE_LOCAL_INTENTS) == 7
    assert intent_offered("contrast_boost", "subject", calm) == (True, "offered")
    assert intent_offered("contrast_boost", "background", calm) == (True, "offered")
    assert intent_offered("cast_correction_local", "background", calm) == \
        (True, "offered")
    assert intent_offered("warm_cool_split", "background", calm) == \
        (False, "intent_disabled")
    for intent in LOCAL_INTENTS:
        assert intent_offered(intent, INTENT_ROLES[intent], calm)[0] is (
            intent not in DISABLED_LOCAL_INTENTS
        )


def test_subject_highlight_headroom_reads_the_measured_columns() -> None:
    subject = np.zeros((32, 32), dtype=np.float32)
    subject[:16] = 1.0
    calm = np.full((32, 32, 3), 0.5, dtype=np.float32)
    calm_reading = subject_highlight_headroom(calm, subject)
    assert calm_reading["near_clip_fraction"] == 0.0
    assert calm_reading["headroom_ok"] is True
    assert calm_reading["highlight_pressure"] is False
    assert calm_reading["subject_pixels"] == 512.0
    blown = calm.copy()
    blown[:16, :8] = 0.99  # 1/4 of the subject sits at or above 250/255
    blown_reading = subject_highlight_headroom(blown, subject)
    assert blown_reading["near_clip_fraction"] == pytest.approx(0.25)
    assert blown_reading["p99_luma"] >= SUBJECT_HEADROOM_GATE["near_clip_level"]
    assert blown_reading["highlight_pressure"] is True
    assert blown_reading["headroom_ok"] is False
    with pytest.raises(RenderError, match="empty_subject_mask"):
        subject_highlight_headroom(calm, np.zeros((32, 32), dtype=np.float32))


def test_build_local_packets_pairs_each_mask_with_intent_row_subsets(
    tmp_path: Path
) -> None:
    _source, subject_path = _source_files(tmp_path, area="small")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    masks = build_mask_bank(
        subject_path, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts, include_background=True,
    )
    packet = [
        next(row for row in masks if row["role"] == "subject"),
        next(row for row in masks if row["role"] == "background"),
    ]
    catalog = _intent_catalog()
    directions = _StubDirections()
    built = build_local_packets(
        catalog, packet, exclude=("hue",), diagnosis=_diagnosis(), palette={},
        scene="portrait", source_sha256="s",
        global_fingerprint=catalog.get("hue").fingerprint(),
        headroom={"highlight_pressure": False, "subject_saturation_mean": 0.2},
        mask_reach=_StubMaskReach({}), direction_probe=directions,
    )
    by_intent = {(row["mask_id"], row["intent"]): row for row in built["packets"]}
    by_intent.update({row["intent"]: row for row in built["packets"]})
    # `warm_cool_split` (B6 item 1) and `highlight_rescue` (B7 item 3) are disabled:
    # they never reach a packet, and they produce no `local_packet_notes` row either.
    assert {row["intent"] for row in built["packets"]} == {
        "luminance_pop", "sat_boost", "hue_shift", "background_control",
        "zonal_contrast", "contrast_boost", "cast_correction_local",
    }
    assert {"warm_cool_split", "highlight_rescue"}.isdisjoint(
        {row["intent"] for row in built["notes"]}
    )
    # B11 item 3 (R7.3): both new intents are offered on both masks, and the packets of
    # one mask are emitted in `intent_packet_order`.
    for mask_id in (str(packet[0]["mask_id"]), str(packet[1]["mask_id"])):
        emitted = [row["intent"] for row in built["packets"]
                   if row["mask_id"] == mask_id]
        assert emitted == [intent for intent in intent_packet_order(
            ACTIVE_LOCAL_INTENTS
        ) if intent in set(emitted)]
        assert emitted[:2] == ["cast_correction_local", "contrast_boost"]
    # Every mask is measured once; the direction stub is asked per mask, not per intent.
    assert directions.masks == [str(row["mask_id"]) for row in packet]
    subject_id = str(packet[0]["mask_id"])
    background_id = str(packet[1]["mask_id"])
    assert {row["mask_id"] for row in built["packets"] if row["role"] == "subject"} \
        == {subject_id}
    assert {row["mask_id"] for row in built["packets"] if row["role"] == "background"} \
        == {background_id}
    presets = [row["preset_id"] for row in built["rows"]]
    assert "hue" not in presets  # the parent global preset stays excluded
    for entry in built["packets"]:
        assert 1 <= len(entry["row_indices"]) <= LOCAL_PACKET_ROW_LIMIT
        assert len(set(entry["row_indices"])) == len(entry["row_indices"])
    assert [presets[index] for index in by_intent["luminance_pop"]["row_indices"]] \
        == ["bright"]
    assert [presets[index] for index in by_intent["zonal_contrast"]["row_indices"]] \
        == ["dark"]
    assert by_intent["zonal_contrast"]["intent_variant"] == "background_darken"
    assert by_intent["luminance_pop"]["intent_variant"] == "luminance_pop"
    blown = build_local_packets(
        catalog, packet, exclude=("hue",), diagnosis=_diagnosis(), palette={},
        scene="portrait", source_sha256="s",
        global_fingerprint=catalog.get("hue").fingerprint(),
        headroom={"highlight_pressure": True, "subject_saturation_mean": 0.2},
        mask_reach=_StubMaskReach({}), direction_probe=_StubDirections(),
    )
    offered = {row["intent"] for row in blown["packets"]}
    assert "luminance_pop" not in offered  # B2 candidate-side guard
    assert "highlight_rescue" not in offered  # B7 item 3
    assert {row["intent"]: row["reason"] for row in blown["notes"]}["luminance_pop"] \
        == "highlight_headroom_exhausted"


class _StubMaskReach:
    """Duck-typed stand-in for `MaskReachProbe` with scripted per-pair readings."""

    def __init__(self, values: dict[tuple[str, str], float], default: float = 6.0):
        self.values = values
        self.default = default
        self.calls: list[tuple[str, str]] = []

    def measure(self, mask: Mapping[str, Any], preset_id: str) -> float:
        key = (str(mask["mask_id"]), str(preset_id))
        self.calls.append(key)
        return float(self.values.get(key, self.default))


def test_mask_reach_probe_is_deterministic_and_reads_the_mask_region(
    tmp_path: Path
) -> None:
    """B8 item 4: the probe samples the mask support deterministically, weights the
    CIEDE2000 by alpha, and reports zero for a LUT that is a dead zone on that mask."""
    config = _write_config(tmp_path, presets=4)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    artifacts = ArtifactStore(tmp_path / "probe")
    ramp = np.linspace(0.05, 0.95, 96, dtype=np.float32)
    image = np.repeat(np.stack([ramp, ramp * 0.8, ramp * 0.6], axis=-1)[None], 64, 0)
    source = artifacts.put_image_array(np.ascontiguousarray(image))
    alpha = np.zeros((64, 96), dtype=np.float32)
    alpha[:, :48] = 1.0  # 3072 support pixels, above the 1024 sampling budget
    mask = {"mask_id": "m", "alpha_artifact": artifacts.put_alpha(alpha).to_dict()}
    loader = configured_lut_loader(config.databuild_config)

    def probe(source_sha256: str = "s") -> MaskReachProbe:
        return MaskReachProbe(
            artifacts, catalog, input_artifact=source.to_dict(),
            source_sha256=source_sha256, loader=loader,
        )

    first, second = probe(), probe()
    # p0 is the identity LUT of the fixture bank: no reach anywhere, mask or not.
    assert first.measure(mask, "p0") == 0.0
    values = [first.measure(mask, f"p{index}") for index in range(4)]
    assert values == [second.measure(mask, f"p{index}") for index in range(4)]
    assert values == sorted(values) and values[1] > MASK_REACH_GATE["reach_de_min"]
    assert first._samples[str(mask["alpha_artifact"]["sha256"])][0].shape == \
        (int(MASK_REACH_GATE["sample_pixels"]), 1, 3)
    # A different source draws a different sample of the same mask support.
    assert probe("other").measure(mask, "p1") != values[1]
    # Repeat pairs are answered from the cache instead of rendering again.
    assert len(first._values) == 4
    assert first.measure(mask, "p1") == values[1] and len(first._values) == 4
    # Support is alpha > 0.05, so a mask that is empty at that level reads as zero.
    faint = {"mask_id": "faint", "alpha_artifact": artifacts.put_alpha(
        np.full((64, 96), 0.04, dtype=np.float32)
    ).to_dict()}
    assert first.measure(faint, "p3") == 0.0


def test_dead_zone_lut_never_enters_a_local_packet(tmp_path: Path) -> None:
    """B8 item 4: a LUT below the mask-conditioned floor is dropped from that mask's
    packet, the survivors carry `mask_reach_de`, and an emptied packet is noted."""
    _source, subject_path = _source_files(tmp_path, area="small")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    masks = build_mask_bank(
        subject_path, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts, include_background=True,
    )
    packet = [
        next(row for row in masks if row["role"] == "subject"),
        next(row for row in masks if row["role"] == "background"),
    ]
    subject_id, background_id = str(packet[0]["mask_id"]), str(packet[1]["mask_id"])
    catalog = _intent_catalog()

    def build(mask_reach: Any) -> dict[str, Any]:
        return build_local_packets(
            catalog, packet, exclude=("hue",), diagnosis=_diagnosis(), palette={},
            scene="portrait", source_sha256="s",
            global_fingerprint=catalog.get("hue").fingerprint(),
            headroom={"highlight_pressure": False, "subject_saturation_mean": 0.2},
            mask_reach=mask_reach, direction_probe=_StubDirections(),
        )

    # B11 item 2: both probes are mandatory now; an unwired caller fails loudly.
    with pytest.raises(CandidateError, match="mask reach probe"):
        build(None)
    with pytest.raises(CandidateError, match="direction probe"):
        build_local_packets(
            catalog, packet, exclude=("hue",), diagnosis=_diagnosis(), palette={},
            scene="portrait", mask_reach=_StubMaskReach({}), direction_probe=None,
        )

    # `bright` is the only row of the luminance_pop packet and it is a dead zone on
    # the subject mask; `dark` reaches on the background mask.
    stub = _StubMaskReach({(subject_id, "bright"): 2.0}, default=5.25)
    built = build(stub)
    assert built["mask_reach_applied"] is True
    assert built["direction_prefilter_applied"] is True
    by_intent = {row["intent"] for row in built["packets"]}
    assert "luminance_pop" not in by_intent
    note = next(row for row in built["notes"] if row["reason"] == "no_row_reaches_mask")
    assert note["intent"] == "luminance_pop" and note["mask_id"] == subject_id
    assert note["mask_reach_de_max"] == 2.0 and note["dropped_rows"] == 1
    assert note["mask_reach_de_min"] == MASK_REACH_GATE["reach_de_min"]
    assert all(row["mask_reach_de"] == 5.25 for row in built["rows"])
    assert "bright" not in {row["preset_id"] for row in built["rows"]}
    assert by_intent == {
        "sat_boost", "hue_shift", "background_control", "zonal_contrast",
        "contrast_boost", "cast_correction_local",
    }
    # Every (mask, prefiltered row) pair is measured, and a pair shared by two intents
    # on the same mask is answered from the probe's own cache.
    assert set(stub.calls) == {
        (subject_id, "bright"), (subject_id, "sat"), (subject_id, "cool"),
        (subject_id, "split"), (subject_id, "chill"),
        (background_id, "dark"), (background_id, "split"), (background_id, "chill"),
    }
    # R7.1 audit column: domain size, prefilter survivors and reach drops per packet.
    retrieval = {
        (row["mask_id"], row["intent"]): row for row in built["retrieval"]
    }
    assert retrieval[(subject_id, "luminance_pop")] == {
        "mask_id": subject_id, "intent": "luminance_pop", "role": "subject",
        "domain_size": 1, "prefiltered": 1, "reach_dropped": 1, "rows": 0,
    }
    assert retrieval[(subject_id, "contrast_boost")]["rows"] == 1

    # The same preset measured differently on two masks is two shortlist rows.
    pair = [packet[0], next(row for row in masks
                            if row["role"] == "subject" and row is not packet[0])]
    other_id = str(pair[1]["mask_id"])
    split = build_local_packets(
        catalog, pair, exclude=("hue",), diagnosis=_diagnosis(), palette={},
        scene="portrait", source_sha256="s",
        global_fingerprint=catalog.get("hue").fingerprint(),
        headroom={"highlight_pressure": False, "subject_saturation_mean": 0.2},
        mask_reach=_StubMaskReach({(subject_id, "sat"): 4.0}, default=5.25),
        direction_probe=_StubDirections(),
    )
    rows = split["rows"]
    by_mask = {(row["mask_id"], row["intent"]): row for row in split["packets"]}
    assert [rows[index]["mask_reach_de"]
            for index in by_mask[(subject_id, "sat_boost")]["row_indices"]] == [4.0]
    assert [rows[index]["mask_reach_de"]
            for index in by_mask[(other_id, "sat_boost")]["row_indices"]] == [5.25]
    assert sum(row["preset_id"] == "sat" for row in rows) == 2


def test_shortlist_splits_one_preset_across_two_intent_ladders(tmp_path: Path) -> None:
    """B7 item 1: a preset admitted by a `low` and a `high` intent claims different
    achievable bins, so the flattened shortlist carries it as two rows."""
    _source, subject_path = _source_files(tmp_path, area="small")
    artifacts = ArtifactStore(tmp_path / "artifacts")
    masks = build_mask_bank(
        subject_path, render_size=(96, 64), source_id="source",
        prompt_revision="local-agent-v1", artifacts=artifacts,
    )
    packet = [next(row for row in masks if row["role"] == "subject")]
    # dL = 2.0 clears luminance_pop (`low` ladder); dSat = 20.0 with |dL| <= 3 clears
    # sat_boost (`high` ladder).
    catalog = LutCatalog(
        (_intent_record("both", mid_gray_dL=2.0, sat_pct_mean=20.0),),
        segment_fingerprints_sha256="e" * 64,
    )
    built = build_local_packets(
        catalog, packet, exclude=(), diagnosis=_diagnosis(), palette={},
        scene="portrait", source_sha256="s",
        global_fingerprint=_intent_record("g").fingerprint(),
        headroom={"highlight_pressure": False, "subject_saturation_mean": 0.2},
        # 5.0 clears `subtle` and `natural` on both ladders, `strong` only on `low`.
        mask_reach=_StubMaskReach({}, default=5.0), direction_probe=_StubDirections(),
    )
    by_intent = {row["intent"]: row for row in built["packets"]}
    assert set(by_intent) == {"luminance_pop", "sat_boost"}
    rows = built["rows"]
    assert [row["preset_id"] for row in rows] == ["both", "both"]
    assert by_intent["luminance_pop"]["row_indices"] != \
        by_intent["sat_boost"]["row_indices"]
    assert rows[by_intent["luminance_pop"]["row_indices"][0]]["achievable_bins"] == \
        ["subtle", "natural", "strong"]
    assert rows[by_intent["sat_boost"]["row_indices"][0]]["achievable_bins"] == \
        ["subtle", "natural"]


def test_full_resolution_mask_geometry_uses_bounded_expansion() -> None:
    core = np.zeros((1024, 1536), dtype=np.float32)
    core[260:765, 400:1137] = 1.0
    geometry = {
        "kind": "band", "center_x": 0.5, "center_y": 0.5,
        "angle": 0.0, "half_short": 0.2412,
    }
    initial = _stats(_evaluate_geometry(core.shape, geometry), core)
    assert initial["subject_high_coverage"] < 0.98

    alpha = _fit_full_resolution_geometry(core, geometry)
    fitted = _stats(alpha, core)
    assert fitted["effective_alpha_mean"] > 0.45
    assert fitted["subject_high_coverage"] >= 0.98
    assert fitted["subject_support_coverage"] == 1.0


class _ArrayRenderer(FullPresetRenderer):
    def __init__(self, delta: np.ndarray | None = None) -> None:
        self.calls = 0
        self.delta = np.asarray(
            delta if delta is not None else [0.28, -0.12, 0.12], dtype=np.float32
        )

    def render_full(self, input_path: Path, _preset: LutRecord) -> np.ndarray:
        self.calls += 1
        with Image.open(input_path) as image:
            before = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        return np.clip(before + self.delta, 0, 1)


class _CapabilityRenderer(_ArrayRenderer):
    def __init__(self, preset_ids: set[str]) -> None:
        super().__init__()
        self._preset_ids = frozenset(preset_ids)

    @property
    def renderable_preset_ids(self) -> frozenset[str]:
        return self._preset_ids


def test_strength_contract_and_render_cache(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    alpha = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    # C1b item 7: every local mask must carry `subject_artifact`; the gate that reads
    # it is no longer skipped when it is absent.
    local_mask = {"mask_id": "m", "alpha_artifact": alpha.to_dict(),
                  "subject_artifact": alpha.to_dict()}
    renderer = _ArrayRenderer()
    calibrator = StrengthCalibrator(
        renderer, _catalog(1), artifacts, audit, renderer_revision="r1",
        search_steps=17, clip_fraction_max=0.2,
    )
    global_result = calibrator.calibrate_global(
        source_sha256="s", branch_id="g", input_artifact=source.to_dict(),
        preset_id="p0", strength_bin="natural",
    )
    assert 3.0 <= global_result["metrics"]["delta_e"] < 4.5
    assert global_result["parameters"]["global_strength"] is not None
    assert global_result["parameters"]["local_strength"] is None
    calls = renderer.calls
    cached = calibrator.calibrate_global(
        source_sha256="s", branch_id="g", input_artifact=source.to_dict(),
        preset_id="p0", strength_bin="natural",
    )
    assert cached["cache_hit"] and renderer.calls == calls
    artifacts.path_for(cached["artifact"]).unlink()
    rematerialized = calibrator.calibrate_global(
        source_sha256="s", branch_id="g", input_artifact=source.to_dict(),
        preset_id="p0", strength_bin="natural",
    )
    assert not rematerialized["cache_hit"] and renderer.calls == calls + 1
    assert artifacts.path_for(rematerialized["artifact"]).is_file()

    # B7 item 1: the local band is looked up per intent, so the contract is checked
    # once on each of the two ladders.
    for intent in ("sat_boost", "luminance_pop"):
        strengths = []
        for strength_bin, (low, high, inclusive) in \
                LOCAL_DELTA_E_TARGETS[intent].items():
            local_result = calibrator.calibrate_local(
                source_sha256="s", branch_id=f"l-{intent}-{strength_bin}",
                input_artifact=source.to_dict(), preset_id="p0",
                mask=local_mask,
                strength_bin=strength_bin, intent=intent,
            )
            actual = local_result["metrics"]["delta_e"]
            assert low <= actual <= high if inclusive else low <= actual < high
            assert local_result["luma_capped"] is False
            assert local_result["effective_strength_bin"] == strength_bin
            strengths.append(local_result["parameters"]["local_strength"])
            assert local_result["render_hash"] != global_result["render_hash"]
        assert strengths == sorted(strengths)

    # B7 item 2: a luminance_pop chain on a subject whose p99 luma is above the soft
    # cap is served the band one bin down; a p99 at or below the cap is not.
    capped = calibrator.calibrate_local(
        source_sha256="s", branch_id="l-capped", input_artifact=source.to_dict(),
        preset_id="p0", mask=local_mask,
        strength_bin="strong", intent="luminance_pop", subject_p99_luma=0.97,
    )
    assert capped["luma_capped"] is True
    assert capped["effective_strength_bin"] == "natural"
    low, high, _inclusive = LOCAL_DELTA_E_TARGETS["luminance_pop"]["natural"]
    assert low <= capped["metrics"]["delta_e"] < high
    uncapped = calibrator.calibrate_local(
        source_sha256="s", branch_id="l-uncapped", input_artifact=source.to_dict(),
        preset_id="p0", mask=local_mask,
        strength_bin="strong", intent="luminance_pop",
        subject_p99_luma=LUMA_SOFT_CAP["p99_luma_max"],
    )
    assert uncapped["luma_capped"] is False
    assert uncapped["effective_strength_bin"] == "strong"
    # sat_boost is not a capped intent: the same near-white subject changes nothing.
    assert calibrator.calibrate_local(
        source_sha256="s", branch_id="l-satboost", input_artifact=source.to_dict(),
        preset_id="p0", mask=local_mask,
        strength_bin="strong", intent="sat_boost", subject_p99_luma=0.99,
    )["luma_capped"] is False
    with pytest.raises(RenderError, match="invalid_local_intent"):
        calibrator.calibrate_local(
            source_sha256="s", branch_id="l-bad", input_artifact=source.to_dict(),
            preset_id="p0", mask=local_mask,
            strength_bin="strong", intent="vibe",
        )


def test_applied_alpha_artifact_and_context_gate(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    alpha = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    subject = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    mask = {
        "mask_id": "context", "family": "radial",
        "alpha_artifact": alpha.to_dict(), "subject_artifact": subject.to_dict(),
    }
    accepted = StrengthCalibrator(
        _ArrayRenderer(np.array([0.025, -0.01075, 0.01075], dtype=np.float32)),
        _catalog(1), artifacts, audit, renderer_revision="applied-pass",
        search_steps=17, clip_fraction_max=0.2,
    ).calibrate_local(
        source_sha256="s", branch_id="accepted", input_artifact=source.to_dict(),
        preset_id="p0", mask=mask, strength_bin="natural", intent="sat_boost",
    )
    applied_ref = accepted["applied_alpha_artifact"]
    assert applied_ref == accepted["parameters"]["applied_alpha_artifact"]
    applied = load_alpha(artifacts.path_for(applied_ref), (32, 32))
    assert float(applied.mean()) > 0.45
    assert accepted["metrics"]["applied_alpha_subject_high_coverage"] >= 0.98
    assert accepted["metrics"]["applied_alpha_subject_support_coverage"] == 1.0

    rejected = StrengthCalibrator(
        _ArrayRenderer(), _catalog(1), artifacts, audit,
        renderer_revision="applied-reject", search_steps=17, clip_fraction_max=0.2,
    )
    with pytest.raises(RenderError) as caught:
        rejected.calibrate_local(
            source_sha256="s", branch_id="rejected", input_artifact=source.to_dict(),
            preset_id="p0", mask=mask, strength_bin="natural", intent="sat_boost",
        )
    assert caught.value.code == "applied_alpha_mask_gate"
    assert any(row["status"] == "applied_alpha_rejected"
               for row in audit.export_tables()["render_record"])


def test_local_render_without_subject_artifact_fails_loud(tmp_path: Path) -> None:
    """C1b item 7: the applied-alpha gate is never skipped in silence."""
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    alpha = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    calibrator = StrengthCalibrator(
        _ArrayRenderer(np.array([0.025, -0.01075, 0.01075], dtype=np.float32)),
        _catalog(1), artifacts, audit, renderer_revision="no-subject",
        search_steps=17, clip_fraction_max=0.2,
    )
    with pytest.raises(RenderError) as caught:
        calibrator.calibrate_local(
            source_sha256="s", branch_id="no-subject", input_artifact=source.to_dict(),
            preset_id="p0", mask={"mask_id": "m", "alpha_artifact": alpha.to_dict()},
            strength_bin="natural", intent="sat_boost",
        )
    assert caught.value.code == "subject_artifact_missing"
    assert [row["status"] for row in audit.export_tables()["render_record"]] == \
        ["subject_artifact_missing"]


def test_calibration_request_key_covers_clip_ceiling_and_backend(
    tmp_path: Path,
) -> None:
    """C1b item 6: two calibrators that differ only in the clip ceiling or the
    renderer backend must not share a cached calibration."""
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))

    renderer = _ArrayRenderer()

    def cache_hit(**kwargs) -> bool:
        return StrengthCalibrator(
            renderer, _catalog(1), artifacts, audit,
            renderer_revision="r1", search_steps=17, **kwargs
        ).calibrate_global(
            source_sha256="s", branch_id="g", input_artifact=source.to_dict(),
            preset_id="p0", strength_bin="natural",
        )["cache_hit"]

    assert cache_hit(clip_fraction_max=0.2, backend="cpu_lut") is False
    assert cache_hit(clip_fraction_max=0.2, backend="cpu_lut") is True
    assert cache_hit(clip_fraction_max=0.3, backend="cpu_lut") is False
    assert cache_hit(clip_fraction_max=0.2, backend="gpu_lut") is False
    assert cache_hit(clip_fraction_max=0.2, backend="cpu_lut") is True


def test_applied_alpha_gate_mirrors_for_the_background_role(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    source = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    subject_alpha = np.zeros((32, 32), dtype=np.float32)
    subject_alpha[:8] = 1.0
    subject = artifacts.put_alpha(subject_alpha)
    background_alpha = 1.0 - subject_alpha
    background = artifacts.put_alpha(background_alpha)
    # Covers 4 of the 24 background rows, so the applied-alpha mean over the
    # background region cannot exceed 4/24 = 0.167 < 0.18 at any strength.
    faint_alpha = np.zeros((32, 32), dtype=np.float32)
    faint_alpha[8:12] = 1.0
    faint = artifacts.put_alpha(faint_alpha)

    def render(mask_id: str, alpha_ref, branch: str) -> dict[str, Any]:
        return StrengthCalibrator(
            _ArrayRenderer(np.array([0.025, -0.01075, 0.01075], dtype=np.float32)),
            _catalog(1), artifacts, audit, renderer_revision=f"bg-{branch}",
            search_steps=17, clip_fraction_max=0.2,
        ).calibrate_local(
            source_sha256="s", branch_id=branch, input_artifact=source.to_dict(),
            preset_id="p0", strength_bin="natural", intent="background_control",
            mask={"mask_id": mask_id, "family": "radial", "role": "background",
                  "alpha_artifact": alpha_ref.to_dict(),
                  "subject_artifact": subject.to_dict()},
        )

    # The background mask covers 75% of the frame and none of the subject: it fails
    # the subject coverage gate by construction and must still render.
    accepted = render("bg", background, "accepted")
    assert accepted["metrics"]["applied_alpha_subject_high_coverage"] == 0.0
    assert accepted["metrics"]["applied_alpha_subject_alpha_mean"] <= \
        BACKGROUND_ROLE_GATE["subject_alpha_mean_max"]
    # B6 item 5: the avoidance half is no longer re-checked (monotone in strength);
    # the pre-registered background visibility floor is what the gate now reads.
    assert accepted["metrics"]["applied_alpha_background_alpha_mean"] >= \
        BACKGROUND_ROLE_GATE["applied_background_alpha_mean_min"]
    with pytest.raises(RenderError) as caught:
        render("bg_faint", faint, "rejected")
    assert caught.value.code == "applied_alpha_mask_gate"
    faint_metrics = next(
        row["metrics_json"] for row in audit.export_tables()["render_record"]
        if row["status"] == "applied_alpha_rejected"
    )
    assert faint_metrics["applied_alpha_background_alpha_mean"] < \
        BACKGROUND_ROLE_GATE["applied_background_alpha_mean_min"]


def test_subject_clip_regression_gate_rejects_the_local_render(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    pixels = np.full((32, 32, 3), 0.5, dtype=np.float32)
    pixels[:8] = 0.99  # a subject that already sits one step below the clip level
    source = artifacts.put_image_array(pixels)
    subject_alpha = np.zeros((32, 32), dtype=np.float32)
    subject_alpha[:8] = 1.0
    subject = artifacts.put_alpha(subject_alpha)
    alpha = artifacts.put_alpha(np.ones((32, 32), dtype=np.float32))
    mask = {
        "mask_id": "semantic", "family": "semantic",
        "alpha_artifact": alpha.to_dict(), "subject_artifact": subject.to_dict(),
    }
    calibrator = StrengthCalibrator(
        _ArrayRenderer(np.array([0.02, -0.009, 0.009], dtype=np.float32)),
        _catalog(1), artifacts, audit, renderer_revision="clip-regression",
        search_steps=17, clip_fraction_max=0.2,
    )
    with pytest.raises(RenderError) as caught:
        calibrator.calibrate_local(
            source_sha256="s", branch_id="blown", input_artifact=source.to_dict(),
            preset_id="p0", mask=mask, strength_bin="natural", intent="sat_boost",
        )
    assert caught.value.code == "highlight_clip_regression"
    row = next(row for row in audit.export_tables()["render_record"]
               if row["status"] == "highlight_clip_regression")
    assert row["metrics_json"]["subject_clip_regression"] > \
        SUBJECT_HEADROOM_GATE["subject_clip_regression_max"]

    calm = artifacts.put_image_array(np.full((32, 32, 3), 0.5, dtype=np.float32))
    accepted = StrengthCalibrator(
        _ArrayRenderer(np.array([0.02, -0.009, 0.009], dtype=np.float32)),
        _catalog(1), artifacts, audit, renderer_revision="clip-regression-calm",
        search_steps=17, clip_fraction_max=0.2,
    ).calibrate_local(
        source_sha256="s", branch_id="calm", input_artifact=calm.to_dict(),
        preset_id="p0", mask=mask, strength_bin="natural", intent="sat_boost",
    )
    assert accepted["metrics"]["subject_clip_regression"] == 0.0
    assert accepted["metrics"]["subject_clip_regression_max"] == \
        SUBJECT_HEADROOM_GATE["subject_clip_regression_max"]
    before = np.full((4, 4, 3), 0.5, dtype=np.float32)
    with pytest.raises(RenderError, match="empty_subject_mask"):
        subject_clip_regression(before, before, np.zeros((4, 4), dtype=np.float32))


def test_runtime_restricts_catalog_to_renderer_capabilities(tmp_path: Path) -> None:
    config = _write_config(tmp_path, presets=3)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    services = create_services(
        config,
        audit=audit,
        renderer_factory=lambda _config, _catalog: _CapabilityRenderer({"p0", "p2"}),
        check_preflight=False,
        # C1b item 2: the fixture bank is not the frozen production table, so the
        # startup assertion is pointed at this fixture's own SHA on purpose.
        expected_segment_fingerprint_sha256=file_sha256(
            config.catalog.segment_fingerprints
        ),
    )
    assert set(services.catalog.by_id) == {"p0", "p2"}
    assert set(services.calibrator.catalog.by_id) == {"p0", "p2"}


def test_cpu_lut_renderer_uses_catalog_without_gpu(tmp_path: Path) -> None:
    config = _write_config(tmp_path, presets=2)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    source, _subject = _source_files(tmp_path)
    renderer = CanonicalCpuLutRenderer(config.databuild_config, catalog)
    identity = renderer.render_full(source, catalog.get("p0"))
    shifted = renderer.render_full(source, catalog.get("p1"))
    assert renderer.renderable_preset_ids == {"p0", "p1"}
    assert identity.shape == shifted.shape == (64, 96, 3)
    assert float(shifted[..., 0].mean()) > float(identity[..., 0].mean())


class _TestArtifacts(ArtifactStore):
    def normalize_source_images(self, path, **kwargs):
        ref = self.normalize_image(
            path, longest_edge=128, quality=95,
            retention=kwargs.get("retention", "accepted"),
        )
        return ref, ref

    def normalize_render_image(self, path, **kwargs):
        return self.normalize_image(
            path, longest_edge=128, quality=95,
            retention=kwargs.get("retention", "accepted"),
        )


def _shortlist_text(content: list[dict[str, Any]]) -> str:
    return next(item["text"] for item in content
                if item["type"] == "input_text"
                and item["text"].startswith("LUT shortlist table."))


def _parse_global_shortlist(content: list[dict[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    current: str | None = None
    for line in _shortlist_text(content).splitlines():
        if line.startswith("[major] "):
            current = line[len("[major] "):]
            result[current] = []
        elif current is not None and "|" in line:
            result[current].append(line)
    return result


def _parse_local_shortlist(content: list[dict[str, Any]]) -> list[str]:
    rows: list[str] = []
    started = False
    for line in _shortlist_text(content).splitlines():
        if line == "[local candidate rows]":
            started = True
        elif started and "|" in line:
            rows.append(line)
    return rows


class _FakeTerra:
    def __init__(self, endpoint: EndpointConfig, *, globals_: int = 6,
                 locals_: int = 2, repair: bool = False,
                 fail_once_stage: str | None = None) -> None:
        self.adapter = SimpleNamespace(endpoint=endpoint)
        self.globals = globals_
        self.locals = locals_
        self.repair = repair
        self.fail_once_stage = fail_once_stage
        self.failed = False
        self.counts: Counter[str] = Counter()
        self.stages: list[str] = []
        self.cache_keys: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def request(self, spec, validate_extra=None):
        stage = spec.canonical["stage"]
        with self._lock:
            self.counts[stage] += 1
            self.stages.append(stage)
            self.cache_keys.setdefault(stage, []).append(spec.prompt_cache_key)
            should_fail = stage == self.fail_once_stage and not self.failed
            if should_fail:
                self.failed = True
        if should_fail:
            raise RuntimeError(f"forced_{stage}_failure")
        if stage == "diagnose":
            parsed = _diagnosis()
        elif stage == "global_propose":
            user_content = spec.canonical["input"][-1]["content"]
            shortlist = _parse_global_shortlist(user_content)
            major, rows = next(iter(shortlist.items()))
            parsed = {"major": major, "proposals": [{
                "row_index": index,
                "bin": ("natural", "medium", "bold")[index % 3],
                "reason_codes": ["scene_mood_fit"],
            } for index in range(min(self.globals, len(rows)))]}
        elif stage == "local_propose":
            user_content = spec.canonical["input"][-1]["content"]
            text_items = [json.loads(item["text"]) for item in user_content
                          if item["type"] == "input_text" and item["text"].startswith("{")]
            tail = next(item for item in text_items if "assigned_masks" in item)
            masks = tail["assigned_masks"]
            # B11 item 2: a row's bins now come from its mask-conditioned reach, so the
            # answer has to read them off the table instead of assuming `natural`.
            bins_of_row = [
                [name for name in row.split(" | ")[1].split(",") if name]
                for row in _parse_local_shortlist(user_content)
            ]
            # Pick (mask, intent) packets: distinct masks first, new intents first,
            # so the sibling set covers >= 2 intents whenever the packets allow it.
            pairs = [(mask, entry) for mask in masks
                     for entry in mask["offered_intents"]]
            picks: list[dict[str, Any]] = []
            used_masks: set[str] = set()
            used_intents: set[str] = set()
            for new_intent_only in (True, False):
                for mask, entry in pairs:
                    if len(picks) >= self.locals:
                        break
                    if mask["mask_id"] in used_masks:
                        continue
                    if new_intent_only and entry["intent"] in used_intents:
                        continue
                    used_masks.add(mask["mask_id"])
                    used_intents.add(entry["intent"])
                    row_index = entry["row_indices"][0]
                    picks.append({
                        "row_index": row_index,
                        "mask_id": mask["mask_id"], "intent": entry["intent"],
                        "bin": bins_of_row[row_index][0],
                        "reason_codes": ["diversity_pick"],
                    })
            parsed = {"proposals": picks}
        else:
            raise AssertionError(stage)
        if validate_extra is not None:
            assert validate_extra(parsed) is None
        return SimpleNamespace(
            response={"parsed": parsed}, request_hash=f"hash-{stage}",
            cache_hit=False, usage={},
        )


class _FakeCalibrator:
    def __init__(self, fail_presets: set[str] = frozenset(), *, fail_local: bool = False) -> None:
        self.fail_presets = set(fail_presets)
        self.fail_local = fail_local
        self.local_calls: list[dict[str, Any]] = []

    def calibrate_global(self, **kwargs):
        if kwargs["preset_id"] in self.fail_presets:
            from dataset_build.agent_loop.render import RenderError
            raise RenderError("forced")
        low, high, _ = GLOBAL_DELTA_E_TARGETS[kwargs["strength_bin"]]
        return {"artifact": kwargs["input_artifact"], "metrics": {"delta_e": (low + high) / 2},
                "parameters": {}, "render_hash": "rg-" + kwargs["branch_id"]}

    def calibrate_local(self, **kwargs):
        self.local_calls.append(dict(kwargs))
        if self.fail_local:
            from dataset_build.agent_loop.render import RenderError
            raise RenderError("forced_local")
        return {"artifact": kwargs["input_artifact"], "metrics": {"delta_e": 4.0},
                "parameters": {}, "render_hash": "rl-" + kwargs["branch_id"]}


class _FakeValidator:
    def __init__(self, audit: SQLiteAuditStore, *, repair_only: bool = False,
                 fail_once: bool = False) -> None:
        self.audit = audit
        self.repair_only = repair_only
        self.fail_once = fail_once
        self.failed = False
        self._lock = threading.Lock()

    def validate(self, **kwargs):
        with self._lock:
            should_fail = self.fail_once and not self.failed
            if should_fail:
                self.failed = True
        if should_fail:
            raise RuntimeError("forced_validator_failure")
        passed = not self.repair_only or int(kwargs["assignment"]["repair_count"]) >= 1
        defects = [] if passed else [{
            "defect_code": "halo", "location": "edge", "evidence": "visible halo",
            "confidence": 0.9,
        }]
        row = {
            "validation_id": "v-" + kwargs["branch_id"],
            "source_sha256": kwargs["source_sha256"], "branch_id": kwargs["branch_id"],
            "request_hash": "validator", "passed": passed, "defects": defects,
            "raw": {"passed": passed, "defects": defects},
        }
        self.audit.record_validation(row)
        return {**row, "cache_hit": False, "usage": {}}


def _graph_services(tmp_path: Path, *, globals_: int = 6, locals_: int = 2,
                    repair_only: bool = False, fail_presets: set[str] = frozenset(),
                    fail_local: bool = False, fail_once_stage: str | None = None,
                    fail_validator_once: bool = False,
                    validator_enabled: bool = True, presets: int = 8):
    config = _write_config(
        tmp_path, validator_enabled=validator_enabled, presets=presets
    )
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = _TestArtifacts(config.artifact_root, recorder=audit.record_artifact)
    terra = _FakeTerra(
        config.terra, globals_=globals_, locals_=locals_, repair=repair_only,
        fail_once_stage=fail_once_stage,
    )
    validator = _FakeValidator(
        audit, repair_only=repair_only, fail_once=fail_validator_once,
    ) if validator_enabled else None
    services = AgentServices(
        config, artifacts, audit, terra,
        LutCatalog.load(config.catalog, config.databuild_config),
        _FakeCalibrator(fail_presets, fail_local=fail_local),
        validator,
        TerraLimiter(4),
    )
    return config, audit, terra, services


def test_offline_source_annotation_round_trip_and_hash_validation(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, terra, services = _graph_services(tmp_path)
    source_row = {
        "source_id": "source", "source_path": str(source),
        "subject_path": str(subject), "scene": "portrait", "subject": {},
    }
    annotation = annotate_source(
        config, services.artifacts, audit, terra, services.limiter,
        services.catalog, source_row
    )
    target = tmp_path / "source_annotation.json"
    target.write_text(json.dumps(annotation), encoding="utf-8")
    assert load_source_annotation(target, source_row) == annotation
    assert terra.stages == ["diagnose"]
    assert annotation["provenance"]["reasoning_effort"] == "high"
    assert annotation["preset_reach"]["preset_count"] == 8
    # E4: the histogram board is rendered from the original file, stored as a PNG
    # artifact, and named in the provenance together with its revision.
    provenance = annotation["provenance"]
    assert provenance["board_revision"] == BOARD_REVISION
    assert provenance["diagnose_prompt_revision"] == \
        prompts_module.DIAGNOSE_PROMPT_REVISION
    stored = services.artifacts.read_bytes(provenance["board_sha256"])
    assert stored == board_png(source)
    with Image.open(io.BytesIO(stored)) as board:
        assert board.size == BOARD_SIZE and board.format == "PNG"

    annotation["source_sha256"] = "0" * 64
    target.write_text(json.dumps(annotation), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_source_annotation(target, source_row)


def test_source_annotation_freezes_and_validates_subject_hash(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    row = _source_row(source, subject)
    target = tmp_path / "source_annotation.json"
    target.write_text(json.dumps(row["source_annotation"]), encoding="utf-8")
    Image.new("L", (96, 64), 255).save(subject)
    with pytest.raises(ValueError, match="subject hash mismatch"):
        load_source_annotation(target, row)


def test_reach_probe_is_deterministic_and_freezes_achievable_bins(tmp_path: Path) -> None:
    config = _write_config(tmp_path, presets=3)
    source, _subject = _source_files(tmp_path)
    catalog = LutCatalog.load(config.catalog, config.databuild_config)
    first = probe_preset_reach(source, source_content_hash(source), catalog.records)
    second = probe_preset_reach(source, source_content_hash(source), reversed(catalog.records))
    assert first == second
    assert first["sampled_pixels"] == SAMPLE_PIXELS
    assert first["presets"]["p0"]["achievable_bins"] == []
    assert first["presets"]["p2"]["d_full"] > first["presets"]["p1"]["d_full"]


def _catalog_config(**overrides) -> SimpleNamespace:
    return SimpleNamespace(**{
        "global_major_limit": 3, "global_per_major_limit": 7, "local_limit": 9,
        **overrides,
    })


def test_shortlist_excludes_unreachable_presets_and_exposes_reachable_bins() -> None:
    catalog = _catalog(3)
    reach = {
        "presets": {
            "p0": {"d_full": 5.0, "achievable_bins": ["natural"]},
            "p1": {"d_full": 2.0, "achievable_bins": []},
            "p2": {"d_full": 7.0,
                   "achievable_bins": ["natural", "medium", "bold"]},
        },
    }
    shortlist = catalog.global_shortlist(
        _diagnosis(), {}, "portrait", _catalog_config(), reach,
    )
    rows = {row["preset_id"]: row for row in shortlist["by_major"]["major"]}
    assert set(rows) == {"p0", "p2"}
    assert rows["p0"]["achievable_bins"] == ["natural"]
    assert rows["p2"]["d_full"] == 7.0
    # B11 item 2: the local round no longer reads the offline reach set at all; the
    # per-mask reach decides which bins each row advertises.
    direction = _StubDirections().direction_value
    combined, correction = catalog.direction_scores(direction)
    local = catalog.intent_rows(
        "background_control", (), _diagnosis(), {}, "portrait", 3,
        mask={"mask_id": "m"}, mask_reach=_StubMaskReach(
            {("m", "p0"): 5.0, ("m", "p1"): 2.0, ("m", "p2"): 7.0}
        ),
        combined_scores=combined, correction_scores=correction,
    )
    local_rows = {row["preset_id"]: row for row in local["rows"]}
    assert set(local_rows) == {"p0", "p2"}
    assert local_rows["p0"]["achievable_bins"] == ["subtle", "natural"]
    assert local_rows["p2"]["achievable_bins"] == ["subtle", "natural", "strong"]
    assert [row["preset_id"] for row in local["below_floor"]] == ["p1"]

    validate = _validate_local_response(
        {"global_proposal": {"preset_id": "global"}},
        [{"mask_id": "mask"}], [local_rows["p0"]],
        [{"mask_id": "mask", "intent": "background_control", "row_indices": [0]}],
    )
    assert validate({"proposals": [{
        "row_index": 0, "mask_id": "mask", "bin": "strong",
        "intent": "background_control", "reason_codes": ["skin_safe"],
    }]}) == "local_strength_bin_unreachable"
    assert validate({"proposals": [{
        "row_index": 1, "mask_id": "mask", "bin": "natural",
        "intent": "background_control", "reason_codes": ["skin_safe"],
    }]}) == "local_row_index_out_of_range"
    assert validate({"proposals": [{
        "row_index": 0, "mask_id": "mask", "bin": "natural",
        "intent": "luminance_pop", "reason_codes": ["skin_safe"],
    }]}) == "local_intent_not_offered"
    assert validate({"proposals": [{
        "row_index": 0, "mask_id": "mask", "bin": "natural",
        "intent": "background_control", "reason_codes": ["skin_safe"],
    }]}) is None


def test_global_shortlist_fills_bin_quota_and_records_shortfalls() -> None:
    catalog = _catalog(8)
    reach = {"presets": {
        "p0": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]},
        "p1": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]},
        "p2": {"d_full": 5.0, "achievable_bins": ["natural", "medium"]},
        "p3": {"d_full": 5.0, "achievable_bins": ["natural", "medium"]},
        "p4": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p5": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p6": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p7": {"d_full": 4.0, "achievable_bins": ["natural"]},
    }}
    shortlist = catalog.global_shortlist(
        _diagnosis(), {}, "portrait", _catalog_config(), reach, source_sha256="s",
    )
    rows = shortlist["by_major"]["major"]
    assert len(rows) == 7
    for name, quota in (("natural", 2), ("medium", 2), ("bold", 2)):
        available = sum(name in row["achievable_bins"] for row in rows)
        assert available >= min(quota, sum(
            name in values["achievable_bins"] for values in reach["presets"].values()
        ))
    assert shortlist["quota_deficits"] == []  # every bin has two reachable LUTs

    thin = {"presets": {
        "p0": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]},
        "p1": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p2": {"d_full": 4.0, "achievable_bins": ["natural"]},
    }}
    deficits = catalog.global_shortlist(
        _diagnosis(), {}, "portrait", _catalog_config(), thin, source_sha256="s",
    )["quota_deficits"]
    assert deficits == [
        {"scope": "global", "style_major": "major", "bin": "medium", "required": 2,
         "covered": 1, "reachable_candidates": 1},
        {"scope": "global", "style_major": "major", "bin": "bold", "required": 2,
         "covered": 1, "reachable_candidates": 1},
    ]


def test_intent_rows_fill_one_per_bin_and_report_shortfalls() -> None:
    catalog = _catalog(12)
    direction = _StubDirections().direction_value
    combined, correction = catalog.direction_scores(direction)
    payload = catalog.intent_rows(
        "background_control", (), _diagnosis(), {}, "portrait",
        LOCAL_PACKET_ROW_LIMIT, mask={"mask_id": "m"},
        mask_reach=_StubMaskReach(
            {("m", f"p{index}"): 6.0 if index < 6 else 4.0 for index in range(12)}
        ),
        combined_scores=combined, correction_scores=correction, source_sha256="s",
    )
    rows = payload["rows"]
    assert len(rows) == LOCAL_PACKET_ROW_LIMIT
    for name in ("subtle", "natural", "strong"):
        assert sum(name in row["achievable_bins"] for row in rows) >= 1
    assert payload["quota_deficits"] == []

    shortfall = _intent_rows(_catalog(2), "background_control", 4.0)["quota_deficits"]
    assert shortfall == [
        {"scope": "local", "intent": "background_control", "bin": "natural",
         "required": 1, "covered": 0, "reachable_candidates": 0},
        {"scope": "local", "intent": "background_control", "bin": "strong",
         "required": 1, "covered": 0, "reachable_candidates": 0},
    ]
    # The zero fingerprints of this fixture carry no brightening direction at all.
    assert _intent_rows(catalog, "luminance_pop", 6.0)["rows"] == []


def test_cluster_dedupe_keeps_one_member_and_rotates_by_source_hash() -> None:
    clusters = {f"p{index}": f"c{index // 3}" for index in range(9)}
    catalog = _catalog(9, {
        preset_id: f"shared\x1f{cluster}" for preset_id, cluster in clusters.items()
    })
    reach = {"presets": {
        f"p{index}": {"d_full": 9.0,
                      "achievable_bins": ["natural", "medium", "bold"]}
        for index in range(9)
    }}

    def picked(source_sha: str) -> list[str]:
        shortlist = catalog.global_shortlist(
            _diagnosis(), {}, "portrait", _catalog_config(), reach,
            source_sha256=source_sha,
        )
        return [row["preset_id"] for row in shortlist["by_major"]["major"]]

    first = picked("source-a")
    assert len(first) == 3  # one representative per cluster
    assert {clusters[preset_id] for preset_id in first} == {"c0", "c1", "c2"}
    assert first == picked("source-a")
    assert len({tuple(picked(f"source-{name}")) for name in "abcdefgh"}) > 1

    fallback = _catalog(9).global_shortlist(
        _diagnosis(), {}, "portrait", _catalog_config(), reach, source_sha256="source-a",
    )["by_major"]["major"]
    assert len(fallback) == 7  # no artifact: every preset is its own cluster

    clustered_rows = catalog.global_shortlist(
        _diagnosis(), {}, "portrait", _catalog_config(), reach, source_sha256="source-a",
    )["by_major"]["major"]
    assert [row["scorer_rank_offered"] for row in clustered_rows] == [0, 1, 2]
    assert [row["scorer_rank_raw"] for row in clustered_rows] != [0, 1, 2]


def test_correction_needs_hard_filters_majors_that_reinforce_the_cast() -> None:
    warm = LutRecord(
        preset_id="warm", path="u", format="lut", name="warm", style_major="warm",
        style_minor="m", scene_affinity=("portrait",), de_med=5.0, caption="warm",
        per_probe={}, hsl_features={"summary": {"mid_gray_a": 4.0, "mid_gray_b": 6.0}},
    )
    cool = dataclasses.replace(
        warm, preset_id="cool", style_major="cool",
        hsl_features={"summary": {"mid_gray_a": -4.0, "mid_gray_b": -6.0}},
    )
    catalog = LutCatalog([warm, cool])
    palette = {"lab_a_mean": 5.0, "lab_b_mean": 12.0}
    diagnosis = {**_diagnosis(), "correction_needs": ["warm cast"]}
    shortlist = catalog.global_shortlist(
        diagnosis, palette, "portrait", _catalog_config(), None, source_sha256="s",
    )
    assert list(shortlist["by_major"]) == ["cool"]
    assert shortlist["quota_deficits"][0]["reason"] == "correction_filter_applied"
    assert catalog.global_shortlist(
        _diagnosis(), palette, "portrait", _catalog_config(), None, source_sha256="s",
    )["offered_majors"] == ["cool", "warm"]

    only_warm = LutCatalog([warm]).global_shortlist(
        diagnosis, palette, "portrait", _catalog_config(), None, source_sha256="s",
    )
    assert list(only_warm["by_major"]) == ["warm"]  # conservative fallback: keep all
    assert only_warm["quota_deficits"][0] == {
        "scope": "global", "reason": "correction_filter_empty",
        "correction_needs": 1, "majors_before": 1, "majors_after": 0,
    }


def test_direction_cosine_is_exact_on_known_vectors() -> None:
    def record(preset_id: str, summary: dict[str, float]) -> LutRecord:
        return LutRecord(
            preset_id=preset_id, path="u", format="lut", name=preset_id,
            style_major="major", style_minor="m", scene_affinity=("portrait",),
            de_med=5.0, caption="c", per_probe={}, hsl_features={"summary": summary},
        )

    catalog = LutCatalog([
        record("a", {"mid_gray_dL": 3.0, "mid_gray_a": 4.0}),
        record("b", {"mid_gray_dL": 4.0, "mid_gray_a": -3.0}),
        record("c", {"mid_gray_dL": 1.0, "mid_gray_a": 1.0}),
        record("d", {"mid_gray_dL": -3.0, "mid_gray_a": -4.0}),
        record("zero", {}),
    ])
    assert _direction_cosine(catalog, "a", "a") == 1.0
    assert _direction_cosine(catalog, "a", "b") == 0.0  # (3,4,0,0).(4,-3,0,0) = 0
    assert _direction_cosine(catalog, "a", "d") == -1.0
    assert _direction_cosine(catalog, "c", "a") == 0.989949  # 7/(sqrt(2)*5)
    assert _direction_cosine(catalog, "a", "zero") is None
    assert _direction_cosine(catalog, "a", "absent") is None


def test_region_descriptor_separates_geometries_center_hint_collapses() -> None:
    def mask(family: str, direction: str, projection: list[float]) -> dict[str, Any]:
        return {"family": family, "direction": direction,
                "center_hint": "through the main subject and surrounding background",
                "alpha_projection": projection}

    flat = [1.0] * 64
    top_left = [1.0 if (index // 8) < 2 and (index % 8) < 2 else 0.0
                for index in range(64)]
    keys = {
        region_descriptor(mask("band", "diagonal", flat)),
        region_descriptor(mask("band", "vertical", flat)),
        region_descriptor(mask("radial", "elliptical", flat)),
        region_descriptor(mask("radial", "elliptical", top_left)),
        region_descriptor(mask("linear", "top", flat)),
        region_descriptor({"family": "semantic", "direction": "subject"}),
    }
    assert keys == {
        "band:diagonal", "band:vertical", "radial:center", "radial:top-left",
        "linear:top", "semantic:subject",
    }


def _source_histogram_fixture() -> dict[str, Any]:
    """B12 item 1: a valid frozen source-histogram reading for hand-built states."""
    return source_histogram(
        np.linspace(0.0, 1.0, 32 * 32 * 3, dtype=np.float32).reshape(32, 32, 3)
    )


def _global_shortlist_fixture() -> dict[str, list[dict[str, Any]]]:
    def row(preset_id: str, bins: list[str], raw: int, offered: int) -> dict[str, Any]:
        return {
            "preset_id": preset_id, "style_major": "major", "caption": "objective",
            "fingerprint": {name: 0.0 for name in FINGERPRINT_FIELDS},
            "achievable_bins": bins, "scorer_rank_raw": raw,
            "scorer_rank_offered": offered, "score": 1.0,
            "cluster_id": preset_id, "d_full": 9.0,
        }

    return {"major": [
        row("p0", ["natural", "medium"], 0, 0),
        row("p1", ["natural"], 1, 1),
        row("p2", ["natural", "medium", "bold"], 5, 2),
    ]}


def test_graph_sends_a_source_to_its_own_lane_only(tmp_path: Path) -> None:
    config = _write_config(tmp_path, lanes=2)
    audit = SQLiteAuditStore(tmp_path / "lane-audit.sqlite")
    audit.setup()
    seen: list[tuple[str, str]] = []

    class _Spy:
        def __init__(self, identity: str) -> None:
            self.identity = identity

        def request(self, spec, validate_extra=None):
            seen.append((self.identity, spec.canonical["endpoint_identity"]))
            parsed = {"major": "major", "proposals": [
                {"row_index": 0, "bin": "medium", "reason_codes": ["tonal_contrast"]},
                {"row_index": 2, "bin": "bold", "reason_codes": ["diversity_pick"]},
            ]}
            assert validate_extra(parsed) is None
            return SimpleNamespace(
                response={"parsed": parsed}, request_hash="h", cache_hit=False, usage={},
            )

    router = TerraRouter([
        TerraLane(index=index, identity=endpoint.identity, endpoint=endpoint,
                  client=_Spy(endpoint.identity),
                  limiter=TerraLimiter(2, lane_id=endpoint.identity))
        for index, endpoint in enumerate(config.terra_lanes)
    ])
    services = AgentServices(
        config, None, audit, router.lanes[0].client,
        LutCatalog.load(config.catalog, config.databuild_config),
        None, None, router.lanes[0].limiter, None, router,
    )
    for source_sha in (hashlib.sha256(f"src{index}".encode()).hexdigest()
                       for index in range(12)):
        state = {
            "source_sha256": source_sha, "thread_id": f"thread-{source_sha}",
            "diagnosis": _diagnosis(),
            "source_artifact": {"sha256": "a" * 64, "media_type": "image/jpeg",
                                "size": 10, "uri": "sha256://" + "a" * 64},
            "global_shortlist": _global_shortlist_fixture(),
            "source_histogram": _source_histogram_fixture(),
        }
        _global_propose_node(services)(state)
        expected = config.terra_lane(source_sha).identity
        assert seen[-1] == (expected, expected)
    assert {row[0] for row in seen} == {"lane", "lane2"}


def _run_global_propose(tmp_path: Path) -> tuple[Any, SQLiteAuditStore, dict[str, Any]]:
    config = _write_config(tmp_path)
    audit = SQLiteAuditStore(tmp_path / "propose-audit.sqlite")
    audit.setup()
    captured: dict[str, Any] = {}
    shortlist = _global_shortlist_fixture()

    class _Spy:
        def request(self, spec, validate_extra=None):
            captured["validate"] = validate_extra
            parsed = {"major": "major", "proposals": [
                {"row_index": 0, "bin": "medium", "reason_codes": ["tonal_contrast"]},
                {"row_index": 2, "bin": "bold", "reason_codes": ["diversity_pick"]},
            ]}
            assert validate_extra(parsed) is None
            return SimpleNamespace(
                response={"parsed": parsed}, request_hash="h", cache_hit=False, usage={},
            )

    services = AgentServices(
        config, None, audit, _Spy(), LutCatalog.load(config.catalog, config.databuild_config),
        None, None, TerraLimiter(2),
    )
    state = {
        "source_sha256": "c" * 64, "thread_id": "thread", "diagnosis": _diagnosis(),
        "source_artifact": {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
                            "uri": "sha256://" + "a" * 64},
        "global_shortlist": shortlist,
        "source_histogram": _source_histogram_fixture(),
    }
    produced = _global_propose_node(services)(state)
    return captured["validate"], audit, produced


def test_global_response_validation_rejects_bad_rows_bins_and_majors(
    tmp_path: Path,
) -> None:
    validate, _audit, produced = _run_global_propose(tmp_path)
    assert [row["preset_id"] for row in produced["global_proposals"]] == ["p0", "p2"]
    assert produced["global_proposals"][0] == {
        "proposal_id": "g0-medium", "preset_id": "p0", "strength_bin": "medium",
        "row_index": 0, "reason_codes": ["tonal_contrast"], "scorer_rank_raw": 0,
        "scorer_rank_offered": 0,
    }

    def payload(*proposals: dict[str, Any], major: str = "major") -> dict[str, Any]:
        return {"major": major, "proposals": list(proposals)}

    good = {"row_index": 0, "bin": "medium", "reason_codes": ["tonal_contrast"]}
    other = {"row_index": 2, "bin": "bold", "reason_codes": ["diversity_pick"]}
    assert validate(payload(good, other)) is None
    assert validate(payload(good, other, major="absent")) == \
        "global_major_outside_shortlist"
    assert validate(payload(good)) == "global_count_below_config_minimum"
    assert validate(payload(good, {**other, "row_index": 3})) == \
        "global_row_index_out_of_range"
    assert validate(payload(good, {**other, "row_index": -1})) == \
        "global_row_index_out_of_range"
    assert validate(payload(good, {**other, "row_index": 0})) == \
        "global_row_index_not_distinct"
    assert validate(payload(good, {"row_index": 1, "bin": "bold",
                                   "reason_codes": ["skin_safe"]})) == \
        "global_strength_bin_unreachable"
    assert validate(payload({**good, "bin": "natural"},
                            {"row_index": 1, "bin": "natural",
                             "reason_codes": ["skin_safe"]})) == \
        "global_strength_bin_diversity"


def test_scorer_agreement_columns_land_in_sql(tmp_path: Path) -> None:
    _validate, audit, _produced = _run_global_propose(tmp_path)
    rows = audit._conn().execute(
        "SELECT preset_id,level,scorer_top1,scorer_top3,scorer_top1_raw,"
        "direction_cosine FROM proposal_audit ORDER BY preset_id"
    ).fetchall()
    # p2 sits at offered rank 2 (top3) while its raw rank before cluster dedupe is 5.
    assert [tuple(row) for row in rows] == [
        ("p0", "global", 1, 1, 1, None), ("p2", "global", 0, 1, 0, None),
    ]
    assert "proposal_audit" in audit.export_tables()
    assert "scorer_top1_raw INTEGER" in _POSTGRES_SCHEMA


def test_reason_codes_are_a_closed_vocabulary() -> None:
    assert len(REASON_CODES) == 8
    good = {"major": "major", "proposals": [
        {"row_index": 0, "bin": "natural", "reason_codes": ["skin_safe"]},
    ]}
    assert semantic_error("global_propose", good) is None
    assert semantic_error("global_propose", {"major": "major", "proposals": [
        {"row_index": 0, "bin": "natural", "reason_codes": ["looks_nice"]},
    ]}) == "$.proposals[0].reason_codes[0]:enum"
    assert semantic_error("global_propose", {"major": "major", "proposals": [
        {"row_index": 0, "bin": "natural", "reason_codes": []},
    ]}) == "$.proposals[0].reason_codes:minItems"
    assert semantic_error("global_propose", {"major": "major", "proposals": [
        {"row_index": 0, "bin": "natural", "reason_codes": ["skin_safe", "skin_safe"]},
    ]}) == "reason_codes_not_distinct"
    assert semantic_error("global_propose", {"major": "major", "proposals": [
        {"row_index": 0.5, "bin": "natural", "reason_codes": ["skin_safe"]},
    ]}) == "$.proposals[0].row_index:integer"
    assert semantic_error("global_propose", {"major": "major", "proposals": [
        {"row_index": 0, "bin": "natural", "reason_codes": ["skin_safe"],
         "direction": "warmer"},
    ]}) == "$.proposals[0]:additional"
    assert semantic_error("local_propose", {"proposals": [
        {"row_index": 0, "bin": "subtle", "mask_id": "m", "intent": "sat_boost",
         "reason_codes": ["halo"]},
    ]}) == "$.proposals[0].reason_codes[0]:enum"
    assert semantic_error("local_propose", {"proposals": [
        {"row_index": 0, "bin": "subtle", "mask_id": "m", "reason_codes": ["skin_safe"]},
    ]}) == "$.proposals[0]:required"
    assert semantic_error("local_propose", {"proposals": [
        {"row_index": 0, "bin": "subtle", "mask_id": "m", "intent": "vibe",
         "reason_codes": ["skin_safe"]},
    ]}) == "$.proposals[0].intent:enum"
    # Contract C3: two or more siblings must cover two or more intents.
    assert semantic_error("local_propose", {"proposals": [
        {"row_index": 0, "bin": "subtle", "mask_id": "m", "intent": "sat_boost",
         "reason_codes": ["skin_safe"]},
        {"row_index": 1, "bin": "subtle", "mask_id": "n", "intent": "sat_boost",
         "reason_codes": ["skin_safe"]},
    ]}) == "local_intent_diversity"
    assert semantic_error("local_propose", {"proposals": [
        {"row_index": 0, "bin": "subtle", "mask_id": "m", "intent": "sat_boost",
         "reason_codes": ["skin_safe"]},
        {"row_index": 1, "bin": "subtle", "mask_id": "n", "intent": "hue_shift",
         "reason_codes": ["skin_safe"]},
    ]}) is None
    assert set(LOCAL_BATCH_SCHEMA["properties"]["proposals"]["items"]["properties"]) == {
        "row_index", "bin", "mask_id", "intent", "reason_codes",
    }


def test_catalog_defaults_follow_the_opt_a_quota_sizes(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    assert config.catalog.global_major_limit == 3
    assert config.catalog.global_per_major_limit == 7
    assert config.catalog.local_limit == 9
    assert config.catalog.cluster_artifact is None
    clustered_root = tmp_path / "clustered"
    clustered_root.mkdir()
    clustered = _write_config(
        clustered_root, clusters={f"p{index}": index // 2 for index in range(8)},
    )
    assert clustered.catalog.cluster_artifact is not None
    catalog = LutCatalog.load(clustered.catalog, clustered.databuild_config)
    assert len({catalog.cluster_id(row) for row in catalog.records}) == 4
    assert clustered.thread_revision != config.thread_revision


@pytest.mark.parametrize(
    "override,message",
    [
        ("global_per_major_limit = 5", "global_per_major_limit"),
        ("local_limit = 8", "local_limit"),
        ("global_major_limit = 0", "global_major_limit"),
    ],
)
def test_catalog_quota_limits_below_the_bin_floor_are_rejected(
    tmp_path: Path, override: str, message: str,
) -> None:
    config = _write_config(tmp_path)
    agent = tmp_path / "agent.toml"
    agent.write_text(
        agent.read_text(encoding="utf-8").replace(
            "reach_limit = 8", f"reach_limit = 8\n{override}"
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match=message):
        load_config(agent)
    assert config.catalog.local_limit == 9


def test_subject_hash_changes_thread_identity(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config_root = tmp_path / "config"
    config_root.mkdir()
    config = _write_config(config_root)
    source_sha = source_content_hash(source)
    subject_sha = source_content_hash(subject)
    first = config.thread_id(source_sha, subject_sha)
    assert first != config.thread_id(source_sha, subject_sha, pass_index=1)
    Image.new("L", (96, 64), 255).save(subject)
    second = config.thread_id(source_sha, source_content_hash(subject))
    assert first != second


def test_annotate_batch_persists_partial_success_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "config"
    config_root.mkdir()
    config = _write_config(config_root)
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    source_a, subject_a = _source_files(tmp_path / "a")
    source_b, subject_b = _source_files(tmp_path / "b")
    manifest = tmp_path / "sources.jsonl"
    rows = [
        {"source_id": "ok", "source_path": str(source_a),
         "subject_path": str(subject_a)},
        {"source_id": "bad", "source_path": str(source_b),
         "subject_path": str(subject_b)},
    ]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    monkeypatch.setattr(agent_loop_cli, "require_preflight", lambda *_args: None)

    calls: Counter[str] = Counter()

    def fake_annotate(_config, _artifacts, _audit, _terra, _limiter, _catalog, source,
                      _router=None):
        calls[source["source_id"]] += 1
        if source["source_id"] == "bad":
            raise RuntimeError("individual failure")
        annotation = _source_row(source_a, subject_a)["source_annotation"]
        annotation["source_id"] = "ok"
        return annotation

    monkeypatch.setattr(agent_loop_cli, "annotate_source", fake_annotate)
    output = tmp_path / "annotated.jsonl"
    code = agent_loop_cli._annotate_sources(
        config, manifest, tmp_path / "annotations", output, 0, 2
    )
    assert code == 1
    success_rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source_id"] for row in success_rows] == ["ok"]
    failures = [json.loads(line) for line in output.with_suffix(
        ".jsonl.failures.jsonl"
    ).read_text().splitlines()]
    assert failures == [{
        "error": "individual failure", "error_type": "RuntimeError",
        "source_id": "bad", "source_path": str(source_b),
    }]
    assert calls == {"ok": 1, "bad": 1}
    assert agent_loop_cli._annotate_sources(
        config, manifest, tmp_path / "annotations", output, 0, 2
    ) == 1
    assert calls == {"ok": 1, "bad": 2}


def test_run_cli_persists_failed_source_as_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_root = tmp_path / "config"
    config_root.mkdir()
    config = _write_config(config_root)
    source, subject = _source_files(tmp_path)
    row = _source_row(source, subject)
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(json.dumps(row.pop("source_annotation")), encoding="utf-8")
    row["source_annotation_path"] = str(annotation_path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    audit = SQLiteAuditStore(tmp_path / "run-audit.sqlite")
    audit.setup()
    monkeypatch.setattr(
        agent_loop_cli, "create_services", lambda _config: SimpleNamespace(audit=audit)
    )
    monkeypatch.setattr(agent_loop_cli, "open_checkpointer", lambda _config: nullcontext(None))
    monkeypatch.setattr(agent_loop_cli, "build_graph", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        agent_loop_cli, "run_source",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("forced failure")),
    )
    assert agent_loop_cli._run(config, manifest, 0, 1, resume=False) == 1
    status = audit.source_status(config.campaign_id)
    assert len(status) == 1
    assert status[0]["status"] == "error"
    assert status[0]["counts_json"] == {
        "error": "forced failure", "error_type": "RuntimeError",
    }


def test_graph_requires_offline_annotation_before_any_terra_call(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, terra, services = _graph_services(tmp_path)
    with pytest.raises(ValueError, match="offline source annotation"):
        run_source(build_graph(services), config, {
            "source_id": "source", "source_path": str(source),
            "subject_path": str(subject), "scene": "portrait",
        })
    assert not terra.counts


def test_graph_fanout_join_and_cap_excluded_audit(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, terra, services = _graph_services(tmp_path, globals_=6, locals_=2)
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    assert len(result["branches"]) == 6
    assert all(row["status"] == "formal_global" for row in result["branches"])
    assert len(result["committed_leaves"]) == 6
    assert terra.counts == {"global_propose": 1, "local_propose": 6}
    assert terra.stages[0] == "global_propose"
    assert len(set(terra.cache_keys["local_propose"])) == 6
    local_rows = [row for row in audit.export_tables()["agent_branch"]
                  if row["level"] == "local"]
    assert Counter(row["status"] for row in local_rows) == {"validator_pass": 12}
    assert Counter(row["result_json"].get("commit_status") for row in local_rows) == {
        "committed": 6, "cap_excluded": 6,
    }
    assert all(row["winner_confidence"] == "normal"
               for row in result["committed_leaves"])
    tree = json.loads(services.artifacts.read_bytes(result["tree_artifact"]))
    committed = [leaf for branch in tree["branches"] for leaf in branch["leaves"]
                 if leaf.get("commit_status") == "committed"]
    assert len(committed) == 6
    assert all(leaf["winner_confidence"] == "normal" for leaf in committed)
    referenced = [
        tree["source_artifact"], tree["source_annotation_artifact"],
        tree["global_shortlist_artifact"],
        *(mask["alpha_artifact"] for mask in tree["mask_bank"]),
        *(mask["subject_artifact"] for mask in tree["mask_bank"]),
    ]
    artifact_rows = {
        row["sha256"]: row for row in audit.export_tables()["artifact_record"]
    }
    assert all(artifact_rows[ref["sha256"]]["retention"] == "accepted"
               for ref in referenced)
    assert all(services.artifacts.path_for(ref).is_file() for ref in referenced)
    assert len(audit.export_tables()["validation_record"]) == 12
    shortlist_payload = json.loads(
        services.artifacts.read_bytes(tree["global_shortlist_artifact"])
    )
    # B12 item 1: the shortlist artifact also carries the source histogram evidence.
    assert set(shortlist_payload) == {
        "by_major", "quota_deficits", "offered_majors", "source_histogram",
    }
    assert_histogram_columns(shortlist_payload["source_histogram"])
    assert shortlist_payload["quota_deficits"] == []
    assert all(branch["local_shortlist_deficits"] == [] for branch in tree["branches"])
    proposal_audit = audit._conn().execute(
        "SELECT level,COUNT(*),COUNT(scorer_top1),COUNT(direction_cosine) "
        "FROM proposal_audit GROUP BY level ORDER BY level"
    ).fetchall()
    assert [tuple(row) for row in proposal_audit] == [
        ("global", 6, 6, 0), ("local", 12, 0, 12),
    ]
    assert all(-1.0 <= float(row["direction_cosine"]) <= 1.0
               for row in audit.export_tables()["proposal_audit"]
               if row["direction_cosine"] is not None)


def test_intent_packets_and_packet_notes_land_in_branch_rows_and_tree(
    tmp_path: Path
) -> None:
    source, subject = _source_files(tmp_path)
    # Three catalog presets, all with a non-positive mid-gray dL: no LUT sits in the
    # brightening domain, so every subject intent packet is dropped as empty and only
    # the background role can carry a local edit.
    config, audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, presets=3,
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    branch_rows = [entry for entry in audit.export_tables()["agent_branch"]
                   if entry["level"] == "global"]
    packets = [entry["result_json"]["local_intent_packets"] for entry in branch_rows]
    assert packets and all(rows for rows in packets)
    # B11 item 3: `contrast_boost` reads the segmented fingerprint, so it survives on
    # both roles where every v1 subject intent is empty.
    assert {row["intent"] for rows in packets for row in rows} == {
        "background_control", "contrast_boost",
    }
    assert {row["role"] for rows in packets for row in rows} == {
        "subject", "background",
    }
    notes = [entry["result_json"]["local_packet_notes"] for entry in branch_rows]
    assert all("no_row_in_intent_domain" in {row["reason"] for row in rows}
               for rows in notes)
    assert all(row["reason"] != "role_mismatch" for rows in notes for row in rows)
    # R7.1 audit column lands on the branch row and covers every emitted packet.
    retrieval = [entry["result_json"]["local_retrieval"] for entry in branch_rows]
    assert all(rows for rows in retrieval)
    assert all(
        {(row["mask_id"], row["intent"]) for row in packet_rows}
        <= {(row["mask_id"], row["intent"]) for row in retrieval_rows}
        for packet_rows, retrieval_rows in zip(packets, retrieval)
    )
    tree = json.loads(services.artifacts.read_bytes(result["tree_artifact"]))
    assert all(branch["local_intent_packets"] for branch in tree["branches"])
    assert all(leaf["intent"] in {"background_control", "contrast_boost"}
               for branch in tree["branches"] for leaf in branch["leaves"])


def test_local_calibration_is_wired_to_the_intent_and_the_luma_cap(
    tmp_path: Path
) -> None:
    """B7 items 1-2 runtime wiring: every local calibration call carries the intent
    and the branch `p99_luma`, and every leaf carries the `luma_capped` column."""
    source, subject = _source_files(tmp_path)
    config, audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, presets=3,
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    calls = services.calibrator.local_calls
    assert calls
    assert all(call["intent"] in LOCAL_DELTA_E_TARGETS for call in calls)
    assert all(isinstance(call["subject_p99_luma"], float) for call in calls)
    branch_rows = [entry for entry in audit.export_tables()["agent_branch"]
                   if entry["level"] == "global"]
    p99 = {float(entry["result_json"]["subject_headroom"]["p99_luma"])
           for entry in branch_rows}
    assert {call["subject_p99_luma"] for call in calls} <= p99
    tree = json.loads(services.artifacts.read_bytes(result["tree_artifact"]))
    leaves = [leaf for branch in tree["branches"] for leaf in branch["leaves"]]
    assert leaves
    # This fixture is a flat mid-gray source, so nothing sits above the soft cap.
    assert all(leaf["luma_capped"] is False for leaf in leaves)
    assert all(value <= LUMA_SOFT_CAP["p99_luma_max"] for value in p99)


def test_mask_reach_gate_is_wired_into_the_branch_and_the_leaf_audit(
    tmp_path: Path, monkeypatch
) -> None:
    """B8 item 4 runtime wiring: the packet builder measures every (mask, LUT) pair on
    the real branch image, every leaf carries `mask_reach_de`, and a build that skipped
    the gate is a hard failure instead of a silent pass."""
    source, subject = _source_files(tmp_path)
    config, audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, presets=4,
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    tree = json.loads(services.artifacts.read_bytes(result["tree_artifact"]))
    leaves = [leaf for branch in tree["branches"] for leaf in branch["leaves"]]
    assert leaves
    reach = [float(leaf["mask_reach_de"]) for leaf in leaves]
    assert all(value >= MASK_REACH_GATE["reach_de_min"] for value in reach)
    rows = [row for branch in tree["branches"]
            for row in branch.get("local_shortlist_deficits", [])]
    assert rows == []
    # p0 is the identity LUT of the fixture bank: it reaches nothing on any mask, so
    # it never appears in a local proposal even though its fingerprint is admissible.
    assert all(leaf["proposal"].get("preset_id") != "p0" for leaf in leaves)

    import dataset_build.agent_loop.graph as graph_module

    def unwired_build(applied: str):
        return lambda *args, **kwargs: {
            "rows": [], "packets": [], "notes": [], "quota_deficits": [],
            "retrieval": [], "mask_reach_applied": applied != "mask_reach",
            "direction_prefilter_applied": applied != "direction",
        }

    for index, (skipped, message) in enumerate((
        ("mask_reach", "mask_reach_gate_not_wired"),
        ("direction", "direction_prefilter_not_wired"),
    )):
        monkeypatch.setattr(
            graph_module, "build_local_packets", unwired_build(skipped)
        )
        unwired_root = tmp_path / f"unwired{index}"
        unwired_root.mkdir()
        _config2, _audit2, _terra2, unwired = _graph_services(
            unwired_root, globals_=2, locals_=2, presets=4,
        )
        with pytest.raises(RuntimeError, match=message):
            run_source(build_graph(unwired), _config2, _source_row(source, subject))


def test_graph_build_requires_a_mounted_segment_fingerprint_table(
    tmp_path: Path
) -> None:
    """B11 item 1 (R7.1): the wiring assertion fires at graph build, and every
    production config mounts the table it asserts on."""
    _source, _subject = _source_files(tmp_path)
    config, _audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, presets=4,
    )
    assert services.catalog.segment_fingerprints_mounted
    assert len(services.catalog.segment_fingerprints_sha256) == 64
    build_graph(services)
    stripped = dataclasses.replace(
        services.catalog.records[0], segment_fingerprint=None
    )
    unmounted = dataclasses.replace(services, catalog=LutCatalog(
        (stripped, *services.catalog.records[1:])
    ))
    assert not unmounted.catalog.segment_fingerprints_mounted
    with pytest.raises(RuntimeError, match="segment_fingerprints_not_mounted"):
        build_graph(unmounted)
    with pytest.raises(CandidateError, match="not mounted"):
        unmounted.catalog.segment_fingerprint_table()
    # Every production agent-loop config mounts the artifact.
    configs = sorted(Path("configs").glob("agent_loop.*.toml"))
    assert configs
    for path in configs:
        assert "segment_fingerprints = " in path.read_text(encoding="utf-8"), path


def test_prompt_revision_fingerprint_covers_the_b11_retrieval_contract(
    monkeypatch
) -> None:
    """B11 item 6: the fingerprint table SHA, the retrieval budget, the role domains,
    the two new intent gates and the packet order all move the revision."""
    baseline = prompt_revision_fingerprint()
    assert prompts_module.CANDIDATE_SERIALIZATION_REVISION == "lut-intent-v7.1-optB"
    assert SEGMENT_FINGERPRINT_TABLE_SHA256 == (
        "bac4db04e402b0eaefb702502027f7287b99ee50e8fa76b6b48871a9f0774939"
    )
    for module, name, value in (
        (prompts_module, "SEGMENT_FINGERPRINT_TABLE_SHA256", "f" * 64),
        (prompts_module, "LOCAL_ONLINE_RETRIEVAL", {"prefilter_top_k": 25}),
        (prompts_module, "INTENT_SEGMENT_GATE", {"contrast_shadow_dL_max": -2.0}),
        (prompts_module, "INTENT_DIRECTION_GATE", {"cast_correction_match_min": 0.9}),
        (prompts_module, "INTENT_PACKET_PRIORITY", {"contrast_boost": 0}),
        (prompts_module, "INTENT_ROLE_DOMAINS", {"contrast_boost": ("subject",)}),
    ):
        monkeypatch.setattr(module, name, value)
        assert prompt_revision_fingerprint() != baseline
        monkeypatch.undo()
    assert prompt_revision_fingerprint() == baseline


def test_local_validator_defends_against_the_parent_preset_reappearing() -> None:
    rows = [{"preset_id": "shared", "achievable_bins": ["natural"]}]
    validate = _validate_local_response(
        {"global_proposal": {"preset_id": "shared"}}, [{"mask_id": "mask"}], rows,
        [{"mask_id": "mask", "intent": "luminance_pop", "row_indices": [0]}],
    )
    assert validate({"proposals": [{
        "row_index": 0, "mask_id": "mask", "bin": "natural",
        "intent": "luminance_pop", "reason_codes": ["skin_safe"],
    }]}) == "global_local_preset_equal"


def _commit_leaf(
    branch: str, leaf_id: str, bin_: str, role: str, delta_e: float
) -> dict[str, Any]:
    return {
        "branch_id": leaf_id, "global_branch_id": branch,
        "global_strength_bin": bin_, "status": "validator_skipped",
        "mask_role": role, "mask_family": "radial", "intent": "background_control",
        "visible_region": f"{role}:center",
        "proposal": {"preset_id": f"p-{leaf_id}", "strength_bin": "natural"},
        "render": {"metrics": {"delta_e": delta_e}},
        "validation": {"passed": None, "skipped": True},
    }


def _commit_branches(subject_leaf: bool) -> list[dict[str, Any]]:
    natural_leaves = [_commit_leaf("g1", "l1", "natural", "background", 5.0)]
    bold_leaves = [_commit_leaf("g2", "l2", "bold", "background", 5.0)]
    if subject_leaf:
        # Worse delta-E deviation than l2, so only the role rule can pull it in.
        bold_leaves.append(_commit_leaf("g2", "l3", "bold", "subject", 5.4))
    return [
        {"branch_id": "g1", "status": "formal_global", "leaves": natural_leaves,
         "global_render": {"metrics": {"delta_e": 3.75}}},
        {"branch_id": "g2", "status": "formal_global", "leaves": bold_leaves,
         "global_render": {"metrics": {"delta_e": 7.5}}},
    ]


def test_committed_set_must_carry_a_subject_role_leaf() -> None:
    selected, reasons = select_committed_leaves(
        _commit_branches(subject_leaf=True), minimum=2, maximum=2
    )
    assert reasons == []
    assert [row["branch_id"] for row in selected] == ["l1", "l3"]
    assert sum(row["mask_role"] == "subject" for row in selected) >= 1
    # Background-only sources are exempt: the rule cannot be satisfied there.
    exempt, exempt_reasons = select_committed_leaves(
        _commit_branches(subject_leaf=False), minimum=2, maximum=2
    )
    assert exempt_reasons == []
    assert [row["branch_id"] for row in exempt] == ["l1", "l2"]
    assert all(row["mask_role"] == "background" for row in exempt)


def test_single_intent_supply_relaxes_the_sibling_diversity_rule() -> None:
    rows = [
        {"preset_id": "a", "achievable_bins": ["natural"]},
        {"preset_id": "b", "achievable_bins": ["natural"]},
    ]
    state = {"global_proposal": {"preset_id": "global"}}
    packet = [{"mask_id": "m1"}, {"mask_id": "m2"}]
    answer = {"proposals": [
        {"row_index": 0, "mask_id": "m1", "bin": "natural",
         "intent": "background_control", "reason_codes": ["skin_safe"]},
        {"row_index": 1, "mask_id": "m2", "bin": "natural",
         "intent": "background_control", "reason_codes": ["skin_safe"]},
    ]}
    single_supply = _validate_local_response(state, packet, rows, [
        {"mask_id": "m1", "intent": "background_control", "row_indices": [0, 1]},
        {"mask_id": "m2", "intent": "background_control", "row_indices": [0, 1]},
    ])
    assert single_supply(answer) is None
    two_supply = _validate_local_response(state, packet, rows, [
        {"mask_id": "m1", "intent": "background_control", "row_indices": [0, 1]},
        {"mask_id": "m2", "intent": "background_control", "row_indices": [0, 1]},
        {"mask_id": "m2", "intent": "zonal_contrast", "row_indices": [0, 1]},
    ])
    assert two_supply(answer) == "local_intent_diversity"


def test_repair_payload_never_carries_preset_ids(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, repair_only=True
    )
    requests: list[str] = []
    forward = terra.request

    def capture(spec, validate_extra=None):
        requests.append(json.dumps(spec.canonical, ensure_ascii=False))
        return forward(spec, validate_extra)

    terra.request = capture
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    repairs = [text for text in requests if "repair" in text]
    assert repairs  # the assertion below must not be vacuous
    assert all("excluded_preset_ids" not in text for text in repairs)
    preset_ids = {row.preset_id for row in services.catalog.records}
    assert all(preset_id not in text for text in repairs for preset_id in preset_ids)
    assert all(entry["result_json"]["intent_supply"] >= 1
               for entry in audit.export_tables()["agent_branch"]
               if entry["level"] == "global"
               and entry["result_json"].get("local_intent_packets"))


def test_agent_branch_and_proposal_audit_are_campaign_scoped(tmp_path: Path) -> None:
    path = tmp_path / "legacy.sqlite"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        "CREATE TABLE agent_branch (branch_id TEXT PRIMARY KEY, campaign_id TEXT "
        "NOT NULL, source_sha256 TEXT NOT NULL, parent_id TEXT, level TEXT NOT NULL, "
        "status TEXT NOT NULL, proposal_json TEXT NOT NULL, result_json TEXT NOT NULL, "
        "created_at REAL NOT NULL);"
        "CREATE INDEX idx_branch_source ON agent_branch(campaign_id,source_sha256);"
        "CREATE TABLE proposal_audit (branch_id TEXT PRIMARY KEY, campaign_id TEXT "
        "NOT NULL, source_sha256 TEXT NOT NULL, level TEXT NOT NULL, preset_id TEXT "
        "NOT NULL, scorer_top1 INTEGER, scorer_top3 INTEGER, scorer_top1_raw INTEGER, "
        "direction_cosine REAL, created_at REAL NOT NULL);"
    )
    legacy.execute(
        "INSERT INTO agent_branch VALUES('b1','camp-a','s',NULL,'global','ok',"
        "'{}','{}',1.0)"
    )
    legacy.execute(
        "INSERT INTO proposal_audit VALUES('b1','camp-a','s','global','p0',1,1,1,0.5,1.0)"
    )
    legacy.commit()
    legacy.close()

    store = SQLiteAuditStore(path)
    store.setup()
    store.setup()  # the migration is idempotent
    for campaign in ("camp-a", "camp-b"):
        store.record_branch({
            "branch_id": "b1", "campaign_id": campaign, "source_sha256": "s",
            "level": "global", "status": f"status-{campaign}",
            "proposal": {}, "result": {},
        })
        store.record_proposal_audit({
            "branch_id": "b1", "campaign_id": campaign, "source_sha256": "s",
            "level": "global", "preset_id": "p0", "scorer_top1": 1,
            "scorer_top3": 1, "scorer_top1_raw": 1, "direction_cosine": 0.5,
        })
    tables = store.export_tables()
    assert {(row["campaign_id"], row["status"]) for row in tables["agent_branch"]} == {
        ("camp-a", "status-camp-a"), ("camp-b", "status-camp-b"),
    }
    assert {row["campaign_id"] for row in tables["proposal_audit"]} == \
        {"camp-a", "camp-b"}
    conn = store._conn()
    for table in ("agent_branch", "proposal_audit"):
        info = conn.execute(f"PRAGMA table_info({table})").fetchall()
        assert {row["name"] for row in info if int(row["pk"]) > 0} == \
            {"campaign_id", "branch_id"}
    assert any(row[1] == "idx_branch_source"
               for row in conn.execute("PRAGMA index_list(agent_branch)"))


def test_graph_repairs_only_after_entire_first_batch_fails(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, repair_only=True
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    assert terra.counts["local_propose"] == 4
    assert all(any(leaf["repair_count"] == 1 and leaf["validation"]["passed"]
                   for leaf in branch["leaves"]) for branch in result["branches"])


def test_local_render_failures_trigger_one_repair_then_reject(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=2, fail_local=True
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "source_rejected"
    assert terra.counts["local_propose"] == 4
    assert not audit.export_tables()["validation_record"]
    assert all(
        leaf["status"] == "render_rejected"
        for branch in result["branches"] for leaf in branch["leaves"]
    )


def test_disabled_validator_commits_without_validator_approval(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1, validator_enabled=False
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert result["terminal_status"] == "accepted"
    assert not audit.export_tables()["validation_record"]
    leaves = [leaf for branch in result["branches"] for leaf in branch["leaves"]]
    assert all(leaf["status"] == "validator_skipped" for leaf in leaves)
    assert all(leaf["validation"] == {
        "passed": None, "skipped": True, "mode": "disabled", "defects": [],
    } for leaf in leaves)
    assert all(leaf["winner_confidence"] == "low"
               for leaf in result["committed_leaves"])
    tree = json.loads(services.artifacts.read_bytes(result["tree_artifact"]))
    committed = [leaf for branch in tree["branches"] for leaf in branch["leaves"]
                 if leaf.get("commit_status") == "committed"]
    assert all(leaf["winner_confidence"] == "low" for leaf in committed)


def test_resume_without_checkpoint_and_completed_run_are_idempotent(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1
    )
    row = _source_row(source, subject)
    with open_checkpointer(config) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        first = run_source(graph, config, row, resume=True)
        calls_after_first = terra.counts.copy()
        second = run_source(graph, config, row, resume=False)
    assert first["terminal_status"] == "accepted"
    assert second["tree_artifact"] == first["tree_artifact"]
    assert terra.counts == calls_after_first


def test_pass_index_creates_independent_checkpoint_and_audit_rows(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1
    )
    row = _source_row(source, subject)
    with open_checkpointer(config) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        first = run_source(graph, config, row)
        second = run_source(graph, config, {**row, "pass_index": 1})
    assert first["thread_id"] != second["thread_id"]
    status = audit.source_status(config.campaign_id)
    assert len(status) == 2
    assert {item["prompt_revision"].rsplit(":", 1)[-1] for item in status} == {
        "pass-0", "pass-1",
    }


@pytest.mark.parametrize("failure", ["api", "validator"])
def test_external_failures_propagate_then_resume_from_checkpoint(
    tmp_path: Path, failure: str,
) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1,
        fail_once_stage="local_propose" if failure == "api" else None,
        fail_validator_once=failure == "validator",
    )
    row = _source_row(source, subject)
    expected = "forced_local_propose_failure" if failure == "api" \
        else "forced_validator_failure"
    with open_checkpointer(config) as checkpointer:
        graph = build_graph(services, checkpointer=checkpointer)
        with pytest.raises(RuntimeError, match=expected):
            run_source(graph, config, row)
        completed_calls = {
            "global_propose": terra.counts["global_propose"],
        }
        resumed = run_source(graph, config, row, resume=True)
    assert resumed["terminal_status"] == "accepted"
    assert completed_calls == {"global_propose": 1}
    assert terra.counts["global_propose"] == 1


def test_failed_global_does_not_cancel_sibling_and_source_contract_rejects(tmp_path: Path) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, _terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1, fail_presets={"p0"}
    )
    result = run_source(build_graph(services), config, _source_row(source, subject))
    assert len(result["branches"]) == 2
    assert Counter(row["status"] for row in result["branches"]) == {
        "global_rejected": 1, "formal_global": 1,
    }
    assert result["terminal_status"] == "source_rejected"
    assert "formal_globals_lt_2" in result["reject_reasons"]


@pytest.mark.parametrize("boundary", [
    "prepare_source", "build_global_shortlist", "global_propose_batch",
    "global_branch", "commit_select",
])
def test_checkpoint_resume_does_not_repeat_completed_external_calls(
    tmp_path: Path, boundary: str
) -> None:
    source, subject = _source_files(tmp_path)
    config, _audit, terra, services = _graph_services(
        tmp_path, globals_=2, locals_=1
    )
    row = _source_row(source, subject)
    source_sha = source_content_hash(source)
    with open_checkpointer(config) as checkpointer:
        interrupted = build_graph(
            services, checkpointer=checkpointer, interrupt_after=[boundary]
        )
        partial = run_source(interrupted, config, row)
        assert "tree_artifact" not in partial
        resumed = run_source(
            build_graph(services, checkpointer=checkpointer), config, row, resume=True
        )
    assert resumed["terminal_status"] == "accepted"
    assert resumed["source_sha256"] == source_sha
    assert terra.counts == {"global_propose": 1, "local_propose": 2}
    assert terra.stages[0] == "global_propose"


class _FakeStream:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def __iter__(self):
        return iter(self.events)


class _FakeResponses:
    def __init__(self, actions):
        self.actions = list(actions)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return _FakeStream(action)


class _FakeHttpError(Exception):
    def __init__(self, status: int, code: str, retry_after: str = "2") -> None:
        super().__init__(code)
        self.status_code = status
        self.body = {"error": {"code": code}}
        self.response = SimpleNamespace(headers={"Retry-After": retry_after})


def _completed_events(
    parsed: dict[str, Any], model: str, *, cached_tokens: int = 0,
    cache_write_tokens: int = 0,
):
    from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

    text = json.dumps(parsed)
    response = SimpleNamespace(
        status="completed", model=model, id="response-1",
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20,
            input_tokens_details=SimpleNamespace(
                cached_tokens=cached_tokens, cache_write_tokens=cache_write_tokens,
            ),
        ),
    )
    return [
        ResponseTextDeltaEvent.model_construct(
            content_index=0, delta=text, item_id="msg", logprobs=[], output_index=0,
            sequence_number=1, type="response.output_text.delta",
        ),
        ResponseCompletedEvent.model_construct(
            response=response, sequence_number=2, type="response.completed"
        ),
    ]


def test_stream_without_text_reports_empty_output_and_stays_retryable() -> None:
    from openai.types.responses import ResponseCompletedEvent

    class _RaisingOutputText:
        status = "completed"
        model = "gpt-5.6-terra"
        id = "response-1"
        usage = SimpleNamespace(
            input_tokens=1, output_tokens=0,
            input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
        )

        @property
        def output_text(self) -> str:
            return "".join([None])  # type: ignore[list-item]

    events = [ResponseCompletedEvent.model_construct(
        response=_RaisingOutputText(), sequence_number=1, type="response.completed",
    )]
    with pytest.raises(TransportError) as raised:
        consume_stream(events)
    assert raised.value.code == "responses_output_empty"
    assert retryable_exception(raised.value)


def test_stream_strips_relay_zero_width_space_prefix() -> None:
    from openai.types.responses import ResponseCompletedEvent, ResponseTextDeltaEvent

    payload = {"confidence": .8, "intent_mode": "enhancement_led"}
    clean = json.dumps(payload)
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1,
        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
    )

    def _deltas(text: str) -> list[Any]:
        return [
            ResponseTextDeltaEvent.model_construct(
                content_index=0, delta=part, item_id="msg", logprobs=[],
                output_index=0, sequence_number=index,
                type="response.output_text.delta",
            )
            for index, part in enumerate([text[:2], text[2:]])
        ]

    completed = SimpleNamespace(
        status="completed", model="gpt-5.6-terra", id="response-1",
        usage=usage, output_text=clean,
    )
    events = [*_deltas("\u200b" + clean), ResponseCompletedEvent.model_construct(
        response=completed, sequence_number=9, type="response.completed",
    )]
    text = consume_stream(events)["text"]
    assert text == clean
    assert json.loads(text) == payload

    class _RaisingOutputText:
        status = "completed"
        model = "gpt-5.6-terra"
        id = "response-1"

        @property
        def output_text(self) -> str:
            return "".join([None])  # type: ignore[list-item]

    raising = _RaisingOutputText()
    raising.usage = usage  # type: ignore[attr-defined]
    events = [*_deltas("\u200b" + clean), ResponseCompletedEvent.model_construct(
        response=raising, sequence_number=9, type="response.completed",
    )]
    text = consume_stream(events)["text"]
    assert text == clean
    assert json.loads(text) == payload


def test_stream_skips_relay_codex_telemetry_but_rejects_other_untyped() -> None:
    """The relay's codex-backed channels inject `codex.*` telemetry events into the
    stream (observed live: type='codex.rate_limits' parsed into
    ResponseAudioDeltaEvent). Those are skipped and counted; any other unofficial
    event still fails closed."""
    from openai.types.responses import (
        ResponseAudioDeltaEvent, ResponseCompletedEvent, ResponseTextDeltaEvent,
    )

    payload = {"confidence": .8, "intent_mode": "enhancement_led"}
    clean = json.dumps(payload)
    usage = SimpleNamespace(
        input_tokens=1, output_tokens=1,
        input_tokens_details=SimpleNamespace(cached_tokens=0, cache_write_tokens=0),
    )
    completed = SimpleNamespace(
        status="completed", model="gpt-5.6-terra", id="response-1",
        usage=usage, output_text=clean,
    )
    telemetry = ResponseAudioDeltaEvent.model_construct(
        delta=None, sequence_number=None, type="codex.rate_limits",
    )
    delta = ResponseTextDeltaEvent.model_construct(
        content_index=0, delta=clean, item_id="msg", logprobs=[],
        output_index=0, sequence_number=1, type="response.output_text.delta",
    )
    done = ResponseCompletedEvent.model_construct(
        response=completed, sequence_number=9, type="response.completed",
    )
    result = consume_stream([telemetry, delta, telemetry, done])
    assert json.loads(result["text"]) == payload
    assert result["relay_telemetry_events"] == 2

    other = ResponseAudioDeltaEvent.model_construct(
        delta=None, sequence_number=None, type="relay.unknown_junk",
    )
    with pytest.raises(TransportError, match="untyped_responses_event"):
        consume_stream([other, delta, done])


class _RaisingOutputTextResponse:
    """Relay shape whose ``output_text`` property raises instead of returning ""."""

    status = "completed"
    model = "gpt-5.6-terra"
    id = "response-1"
    usage = SimpleNamespace(
        input_tokens=3, output_tokens=4,
        input_tokens_details=SimpleNamespace(cached_tokens=2, cache_write_tokens=1),
    )

    def __init__(self, output: list[Any]) -> None:
        self.output = output

    @property
    def output_text(self) -> str:
        return "".join([None])  # type: ignore[list-item]


def test_consume_response_reads_text_usage_and_identity() -> None:
    payload = {"confidence": .8, "intent_mode": "enhancement_led"}
    response = SimpleNamespace(
        status="completed", model="gpt-5.6-terra", id="response-7",
        output_text=json.dumps(payload),
        usage=SimpleNamespace(
            input_tokens=100, output_tokens=20,
            input_tokens_details=SimpleNamespace(cached_tokens=77, cache_write_tokens=128),
        ),
    )
    result = consume_response(response)
    assert json.loads(result["text"]) == payload
    assert result["model"] == "gpt-5.6-terra"
    assert result["response_id"] == "response-7"
    assert result["usage"] == {
        "input_tokens": 100, "output_tokens": 20,
        "cached_tokens": 77, "cache_write_tokens": 128,
    }
    assert result["raw_response"] == {}


def test_consume_response_falls_back_to_message_content_parts() -> None:
    payload = {"confidence": .5, "intent_mode": "enhancement_led"}
    clean = json.dumps(payload)
    response = _RaisingOutputTextResponse([
        SimpleNamespace(type="reasoning", content=None),
        SimpleNamespace(type="message", content=[
            SimpleNamespace(type="output_text", text=clean[:4]),
            SimpleNamespace(type="output_text", text=clean[4:]),
        ]),
    ])
    result = consume_response(response)
    assert result["text"] == clean
    assert json.loads(result["text"]) == payload
    assert result["usage"]["cached_tokens"] == 2
    mapping_shaped = _RaisingOutputTextResponse([
        {"type": "message", "content": [{"type": "output_text", "text": clean}]},
    ])
    assert consume_response(mapping_shaped)["text"] == clean


def test_consume_response_without_any_text_reports_empty_and_stays_retryable() -> None:
    with pytest.raises(TransportError) as raised:
        consume_response(_RaisingOutputTextResponse([
            SimpleNamespace(type="message", content=[]),
        ]))
    assert raised.value.code == "responses_output_empty"
    assert retryable_exception(raised.value)


def test_consume_response_rejects_non_completed_status() -> None:
    response = SimpleNamespace(
        status="incomplete", model="gpt-5.6-terra", id="response-1",
        output_text='{"confidence": 0.8}', usage=None,
    )
    with pytest.raises(TransportError) as raised:
        consume_response(response)
    assert raised.value.code == "responses_not_completed"


def test_consume_response_strips_relay_zero_width_space_prefix() -> None:
    payload = {"confidence": .8, "intent_mode": "enhancement_led"}
    clean = json.dumps(payload)
    response = SimpleNamespace(
        status="completed", model="gpt-5.6-terra", id="response-1",
        output_text="\u200b" + clean, usage=None,
    )
    text = consume_response(response)["text"]
    assert text == clean
    assert json.loads(text) == payload


def test_responses_adapter_payload_model_and_usage(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    artifacts = ArtifactStore(config.artifact_root)
    image = artifacts.put_image_array(np.full((600, 900, 3), .5, dtype=np.float32))
    parsed = {
        "correction_needs": [], "preserve_intent": ["skin"],
        "enhancement_opportunities": ["tone", "depth"],
        "forbidden_directions": [], "evidence": ["balanced"],
        "confidence": .8, "intent_mode": "enhancement_led",
    }
    responses = _FakeResponses([
        _completed_events(
            parsed, config.terra.model, cached_tokens=77, cache_write_tokens=128,
        )
    ])
    adapter = ResponsesAdapter(
        config.terra, artifacts,
        client_factory=lambda _endpoint: SimpleNamespace(responses=responses),
    )
    board = artifacts.put_bytes(
        _png_bytes(BOARD_SIZE), media_type="image/png", retention="audit"
    )
    spec = diagnosis_request(config.terra, image.to_dict(), board.to_dict())
    result = adapter.send_once(spec, 1)
    assert result["parsed"] == parsed
    assert result["usage"] == {
        "input_tokens": 100, "output_tokens": 20,
        "cached_tokens": 77, "cache_write_tokens": 128,
    }
    payload = responses.calls[0]
    assert payload["model"] == config.terra.model
    assert payload["prompt_cache_key"] == spec.prompt_cache_key
    assert payload["store"] is False and payload["stream"] is True
    assert payload["text"]["format"]["strict"] is True
    image_item = payload["input"][1]["content"][0]
    assert image_item["type"] == "input_image"
    assert image_item["image_url"].startswith("data:image/jpeg;base64,")
    encoded = base64.b64decode(image_item["image_url"].split(",", 1)[1])
    with Image.open(io.BytesIO(encoded)) as rendered:
        assert max(rendered.size) <= 512
    assert image_item["detail"] == "low"
    assert "artifact_sha256" not in image_item
    assert "TOP-SECRET" not in json.dumps(payload)
    # E4: the board is the second image and leaves the transport untouched - same
    # bytes, still a PNG, still 640x604, `detail=high`.
    board_item = payload["input"][1]["content"][1]
    assert board_item["type"] == "input_image" and board_item["detail"] == "high"
    assert board_item["image_url"].startswith("data:image/png;base64,")
    board_bytes = base64.b64decode(board_item["image_url"].split(",", 1)[1])
    assert board_bytes == artifacts.read_bytes(board.sha256)
    with Image.open(io.BytesIO(board_bytes)) as rendered:
        assert rendered.size == BOARD_SIZE and rendered.format == "PNG"


def test_responses_adapter_rejects_model_substitution_and_preserves_retry_after(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path)
    artifacts = ArtifactStore(config.artifact_root)
    image = artifacts.put_image_array(np.full((8, 8, 3), .5, dtype=np.float32))
    board = artifacts.put_bytes(
        _png_bytes((16, 16)), media_type="image/png", retention="audit"
    )
    spec = diagnosis_request(config.terra, image.to_dict(), board.to_dict())
    responses = _FakeResponses([
        _completed_events({}, "substituted"), _FakeHttpError(429, "rate_limit"),
    ])
    adapter = ResponsesAdapter(
        config.terra, artifacts,
        client_factory=lambda _endpoint: SimpleNamespace(responses=responses),
    )
    with pytest.raises(ModelSubstituted):
        adapter.send_once(spec, 1)
    with pytest.raises(TransportError) as caught:
        adapter.send_once(spec, 2)
    assert caught.value.code == "429:rate_limit"
    assert caught.value.retry_after == 2.0


def test_terra_limiter_enforces_global_and_per_source_caps() -> None:
    limiter = TerraLimiter(4)
    lock = threading.Lock()
    active_by_source: Counter[str] = Counter()
    maximum_total = 0
    maximum_source = 0

    def task(index: int) -> None:
        nonlocal maximum_total, maximum_source
        source = "shared" if index < 4 else f"source-{index}"
        with limiter.slot(source, "local_propose", f"prefix-{index % 2}"):
            with lock:
                active_by_source[source] += 1
                maximum_total = max(maximum_total, sum(active_by_source.values()))
                maximum_source = max(maximum_source, active_by_source[source])
            time.sleep(.03)
            with lock:
                active_by_source[source] -= 1

    with ThreadPoolExecutor(max_workers=12) as pool:
        list(pool.map(task, range(12)))
    assert maximum_total == 4
    assert maximum_source == 1
    assert max(row["in_flight"] for row in limiter.events) <= 4
    limiter.observe("429:rate_limit")
    assert limiter.effective == 3
    for _ in range(20):
        limiter.observe(None)
    assert limiter.effective == 4


def test_single_endpoint_config_stays_backward_compatible(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    assert [lane.identity for lane in config.terra_lanes] == ["lane"]
    assert config.terra is config.terra_lanes[0]
    assert config.terra_total_concurrency == config.terra_concurrency_target
    assert config.terra_lane("any-source") is config.terra
    assert route_lane_index("any-source", 1) == 0


def test_lane_list_config_loads_every_key_and_keeps_them_out_of_the_export(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, lanes=2)
    assert [lane.identity for lane in config.terra_lanes] == ["lane", "lane2"]
    assert [lane.api_key for lane in config.terra_lanes] == [
        "TOP-SECRET-lane", "TOP-SECRET-lane2"
    ]
    assert config.terra_total_concurrency == 2 * config.terra_concurrency_target
    safe = json.dumps(config.sanitized_dict())
    assert "TOP-SECRET" not in safe
    assert safe.count('"identity": "lane2"') == 1


def test_lane_routing_is_deterministic_per_source_and_covers_both_lanes(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, lanes=2)
    sources = [hashlib.sha256(f"source-{index}".encode()).hexdigest()
               for index in range(200)]
    first = [config.terra_lane_index(source) for source in sources]
    second = [config.terra_lane_index(source) for source in sources]
    assert first == second
    assert set(first) == {0, 1}
    assert 60 <= sum(first) <= 140
    # A fresh process must land on the same lane: the mapping is pure sha256.
    again = tmp_path / "again"
    again.mkdir()
    reloaded = _write_config(again, lanes=2)
    assert [reloaded.terra_lane_index(source) for source in sources] == first
    # The single-lane routing of the same sources is unchanged.
    assert all(route_lane_index(source, 1) == 0 for source in sources)


def test_lane_identity_is_the_only_request_hash_difference_between_lanes(
    tmp_path: Path,
) -> None:
    config = _write_config(tmp_path, lanes=2)
    single_root = tmp_path / "single"
    single_root.mkdir()
    single = _write_config(single_root)
    image = {"sha256": "a" * 64, "media_type": "image/jpeg", "size": 10,
             "uri": "sha256://" + "a" * 64}
    lane_one = diagnosis_request(config.terra_lanes[0], image, _board_ref())
    lane_two = diagnosis_request(config.terra_lanes[1], image, _board_ref())
    legacy = diagnosis_request(single.terra, image, _board_ref())
    # Adding lane 2 does not move lane 1's hash.
    assert lane_one.request_hash == legacy.request_hash
    assert lane_one.canonical["endpoint_identity"] == "lane"
    assert lane_two.canonical["endpoint_identity"] == "lane2"
    assert lane_one.request_hash != lane_two.request_hash
    differences = {
        key for key in set(lane_one.canonical) | set(lane_two.canonical)
        if lane_one.canonical.get(key) != lane_two.canonical.get(key)
    }
    assert differences == {"endpoint_identity"}
    assert lane_one.prompt_cache_key == lane_two.prompt_cache_key


def test_two_lanes_hold_independent_concurrency_and_backoff(tmp_path: Path) -> None:
    config = _write_config(tmp_path, lanes=2)
    artifacts = _TestArtifacts(config.artifact_root)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    cache = ExactResponseCache(audit, lease_seconds=2)
    router = build_terra_router(
        config, artifacts, cache, audit, client_factory=lambda _endpoint: None
    )
    assert len(router) == 2
    assert router.total_concurrency == 2 * config.terra_concurrency_target
    assert router.lanes[0].limiter is not router.lanes[1].limiter

    lock = threading.Lock()
    per_lane_active: Counter[str] = Counter()
    peak: Counter[str] = Counter()
    released = threading.Event()

    def hold(lane_index: int, source: str) -> None:
        lane = router.lanes[lane_index]
        with lane.limiter.slot(source, "diagnose", "prefix"):
            with lock:
                per_lane_active[lane.identity] += 1
                peak[lane.identity] = max(peak[lane.identity],
                                          per_lane_active[lane.identity])
            released.wait(5.0)
            with lock:
                per_lane_active[lane.identity] -= 1

    # Saturate lane 0 (target 4) and check lane 1 still admits work.
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(hold, 0, f"s{index}") for index in range(6)]
        futures += [pool.submit(hold, 1, f"t{index}") for index in range(4)]
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with lock:
                if peak["lane2"] >= 4:
                    break
            time.sleep(0.01)
        with lock:
            assert peak["lane"] <= config.terra_concurrency_target
            assert peak["lane2"] == config.terra_concurrency_target
        released.set()
        for future in futures:
            future.result()

    router.lanes[0].limiter.observe("429:rate_limit")
    assert router.lanes[0].limiter.effective == config.terra_concurrency_target - 1
    assert router.lanes[1].limiter.effective == config.terra_concurrency_target
    metrics = router.metrics()
    assert metrics["lane_count"] == 2
    assert [row["identity"] for row in metrics["lanes"]] == ["lane", "lane2"]
    assert [row["rate_limited"] for row in metrics["lanes"]] == [1, 0]
    assert all(row["lane_id"] in {"lane", "lane2"}
               for row in router.lanes[1].limiter.events)


def test_router_rejects_duplicate_lane_identities(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    artifacts = _TestArtifacts(config.artifact_root)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    cache = ExactResponseCache(audit, lease_seconds=2)
    lane = build_terra_router(config, artifacts, cache, audit).lanes[0]
    with pytest.raises(ValueError, match="distinct"):
        TerraRouter([lane, dataclasses.replace(lane, index=1)])
    with pytest.raises(ValueError, match="at least one lane"):
        TerraRouter([])


def test_config_rejects_endpoint_and_endpoints_together(tmp_path: Path) -> None:
    config = _write_config(tmp_path, lanes=2)
    text = config.path.read_text(encoding="utf-8").replace(
        'endpoints = ["lane", "lane2"]',
        'endpoint = "lane"\nendpoints = ["lane", "lane2"]',
    )
    config.path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="not both"):
        load_config(config.path)


def test_preflight_is_required_per_lane(tmp_path: Path) -> None:
    config = _write_config(tmp_path, lanes=2)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    assert preflight_key(config, config.terra_lanes[0]) \
        != preflight_key(config, config.terra_lanes[1])
    run_preflight(
        config, artifacts, audit, _FakePreflightClient(config.terra.model, 11),
        _catalog(), TerraLimiter(2), endpoint=config.terra_lanes[0],
    )
    with pytest.raises(RuntimeError, match="provider_preflight_required:lane2"):
        require_preflight(config, audit)
    run_preflight(
        config, artifacts, audit, _FakePreflightClient(config.terra.model, 11),
        _catalog(), TerraLimiter(2), endpoint=config.terra_lanes[1],
    )
    assert require_preflight(config, audit)["passed"]
    assert audit.get_preflight(
        preflight_key(config, config.terra_lanes[1])
    )["result"]["endpoint_identity"] == "lane2"


class _FakePreflightClient:
    def __init__(self, model: str, cached_tokens: int) -> None:
        self.model = model
        self.cached_tokens = cached_tokens
        self.calls = 0

    def request(self, spec, validate_extra=None, *, bypass_exact_cache=False):
        assert bypass_exact_cache
        self.calls += 1
        tail = json.loads(spec.canonical["input"][-1]["content"][-1]["text"])["tail"]
        parsed = {"ok": True, "tail": tail}
        assert validate_extra is None or validate_extra(parsed) is None
        return SimpleNamespace(
            request_hash=spec.request_hash, cache_hit=False,
            usage={"input_tokens": 2000, "output_tokens": 5,
                   "cached_tokens": self.cached_tokens if self.calls == 2 else 0},
            response={"model": self.model, "parsed": parsed},
        )


def test_preflight_persists_cache_capability_and_fails_closed(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    client = _FakePreflightClient(config.terra.model, 128)
    result = run_preflight(
        config, artifacts, audit, client, _catalog(), TerraLimiter(2)
    )
    assert result["provider_cache_passed"]
    assert audit.get_preflight(preflight_key(config))["passed"]

    uncached_config = dataclasses.replace(
        config, terra_lanes=(dataclasses.replace(config.terra, model="other-model"),)
    )
    with pytest.raises(RuntimeError, match="provider_preflight_failed_closed"):
        run_preflight(
            uncached_config, artifacts, audit,
            _FakePreflightClient("other-model", 0), _catalog(), TerraLimiter(2),
        )


def test_preflight_still_measures_cache_when_provider_cache_is_best_effort(tmp_path: Path) -> None:
    config = _write_config(tmp_path)
    config = dataclasses.replace(
        config,
        terra_lanes=(dataclasses.replace(config.terra, allow_uncached_provider=True),),
    )
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(config.artifact_root, recorder=audit.record_artifact)
    client = _FakePreflightClient(config.terra.model, 0)
    result = run_preflight(
        config, artifacts, audit, client, _catalog(), TerraLimiter(2)
    )
    assert client.calls == 2
    assert result["degraded_uncached"] and not result["provider_cache_passed"]
    assert audit.get_preflight(preflight_key(config))["passed"]


def test_metrics_join_requests_back_to_source_and_stage(tmp_path: Path) -> None:
    store = SQLiteAuditStore(tmp_path / "audit.sqlite")
    store.setup()
    cache = ExactResponseCache(store, lease_seconds=2)
    spec = RequestSpec({
        "endpoint_identity": "lane", "model": "m", "stage": "diagnose",
        "prompt_cache_key": "prefix", "input": [],
    }, "prefix")
    result = cache.execute(
        spec,
        lambda _spec, _attempt: {"ok": True, "usage": {
            "input_tokens": 100, "output_tokens": 10, "cached_tokens": 40,
        }},
        lambda _row: None,
    )
    store.record_request_context(
        result.request_hash, "campaign", "source", "diagnose", result.cache_hit
    )
    metrics = campaign_metrics(store, "campaign")
    assert metrics["api"]["real_calls_by_stage"] == {"diagnose": 1}
    assert metrics["api"]["per_source_calls"] == {"source": {"diagnose": 1}}
    assert metrics["api"]["provider_cached_token_ratio"] == .4


def test_rejected_cleanup_requires_a_dry_run_manifest(tmp_path: Path) -> None:
    audit = SQLiteAuditStore(tmp_path / "audit.sqlite")
    audit.setup()
    artifacts = ArtifactStore(tmp_path / "artifacts", recorder=audit.record_artifact)
    rejected = artifacts.put_bytes(b"reject", media_type="image/jpeg", retention="quarantine")
    accepted = artifacts.put_bytes(b"accept", media_type="image/jpeg", retention="accepted")
    source = artifacts.put_bytes(b"source", media_type="image/jpeg", retention="audit")
    mask = artifacts.put_bytes(b"mask", media_type="image/png", retention="audit")
    audit._conn().execute("UPDATE artifact_record SET created_at=0")
    manifest = plan_cleanup(audit, artifacts, rejected_days=30, now=31 * 86400)
    assert [row["sha256"] for row in manifest["targets"]] == [rejected.sha256]
    assert {source.sha256, mask.sha256}.isdisjoint(
        row["sha256"] for row in manifest["targets"]
    )
    path = tmp_path / "cleanup.json"
    write_cleanup_manifest(path, manifest)
    result = apply_cleanup_manifest(path, audit, artifacts)
    assert result["removed"] == [rejected.sha256]
    assert not artifacts._path(rejected.sha256).exists()
    assert artifacts.path_for(accepted).is_file()
    row = next(row for row in audit.export_tables()["artifact_record"]
               if row["sha256"] == rejected.sha256)
    assert row["retention"] == "purged"
