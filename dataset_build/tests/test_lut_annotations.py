from __future__ import annotations

import json
from pathlib import Path

import pytest

from types import SimpleNamespace

from dataset_build.agent_loop.candidates import LutCatalog, LutRecord, load_cluster_map
from dataset_build.agent_loop.lut_annotations import (
    ANNOTATION_SCHEMA,
    LIGHTNESS_THRESHOLD,
    LutAnnotationError,
    SATURATION_THRESHOLD,
    SCENE_TAXONOMY,
    STYLE_MAJORS,
    TEMPERATURE_THRESHOLD,
    migrate_annotation,
    migrate_catalog,
    scenes_from_cached_affinity,
    style_major_from_hsl,
    validate_closed_annotation,
)


def _source(
    preset_id: str, *, mid_b: float = 0.0, saturation: float = 0.0,
    lightness: float = 0.0, scene: str = "general",
) -> dict:
    return {
        "preset_id": preset_id,
        "key": preset_id,
        "ok": True,
        "name": f"name {preset_id}",
        "caption": f"caption {preset_id}",
        "per_probe": {"neutral": "measured response"},
        "style_major": "legacy free text",
        "style_minor": f"minor {preset_id}",
        "scene_affinity": scene,
        "strength": "strong",
        "hsl_features": {
            "summary": {
                "mid_gray_a": 10.0,
                "mid_gray_b": mid_b,
                "sat_pct_mean": saturation,
                "mid_gray_dL": lightness,
            },
            "bands": {},
        },
        "provenance": {"model": "gpt-5.6-terra"},
    }


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_style_major_is_a_fixed_objective_27_class_taxonomy() -> None:
    assert len(STYLE_MAJORS) == 27
    assert style_major_from_hsl(_source(
        "p", mid_b=TEMPERATURE_THRESHOLD + 0.01,
        saturation=-SATURATION_THRESHOLD - 0.01,
        lightness=LIGHTNESS_THRESHOLD + 0.01,
    )["hsl_features"]) == "warm__desaturated__bright"
    assert style_major_from_hsl(_source(
        "p", mid_b=TEMPERATURE_THRESHOLD,
        saturation=-SATURATION_THRESHOLD,
        lightness=LIGHTNESS_THRESHOLD,
    )["hsl_features"]) == "neutral__neutral__neutral"


def test_migration_removes_strength_and_aligns_cached_scenes() -> None:
    row = migrate_annotation(
        _source("p", scene="portrait"), 7.25,
        annotations_source_sha256="a" * 64,
        perceptual_de_source_sha256="b" * 64,
    )
    validate_closed_annotation(row)
    assert row["annotation_schema"] == ANNOTATION_SCHEMA
    assert "strength" not in row
    assert row["de_med"] == 7.25
    assert row["scene_affinity"] == ["portrait"]
    assert scenes_from_cached_affinity("general") == SCENE_TAXONOMY


def test_validation_rejects_free_text_or_non_objective_fields() -> None:
    row = migrate_annotation(
        _source("p"), 5.0,
        annotations_source_sha256="a" * 64,
        perceptual_de_source_sha256="b" * 64,
    )
    row["style_major"] = "warm vintage"
    with pytest.raises(LutAnnotationError, match="style_major"):
        validate_closed_annotation(row)
    row = migrate_annotation(
        _source("p"), 5.0,
        annotations_source_sha256="a" * 64,
        perceptual_de_source_sha256="b" * 64,
    )
    row["strength"] = "strong"
    with pytest.raises(LutAnnotationError, match="strength"):
        validate_closed_annotation(row)


def test_catalog_migration_is_resumable_and_audited(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    de_path = tmp_path / "perceptual_de.jsonl"
    output = tmp_path / "closed.jsonl"
    report = tmp_path / "report.json"
    _jsonl(annotations, [
        {"preset_id": "old-failure", "ok": False, "error": "retry"},
        _source("p1", mid_b=-2.0, saturation=-6.0, lightness=-6.0),
        _source("p2", mid_b=0.0, saturation=0.0, lightness=0.0, scene="portrait"),
        _source("p3", mid_b=2.0, saturation=6.0, lightness=6.0, scene="landscape"),
    ])
    _jsonl(de_path, [
        {"preset_id": "p1", "de_med": 4.0},
        {"preset_id": "p2", "de_med": 7.0},
        {"preset_id": "p3", "de_med": 12.0},
        {"preset_id": "unused-param", "de_med": 2.0},
    ])

    first = migrate_catalog(annotations, de_path, output, report=report)
    checksum = first["output"]["sha256"]
    second = migrate_catalog(annotations, de_path, output, report=report)

    assert first["output"]["records"] == 3
    assert first["source_counts"]["failed_lines"] == 1
    assert first["source_counts"]["perceptual_de_unused"] == 1
    assert first["source_counts"]["semantic_model_calls"] == 0
    assert second["output"]["written_this_run"] == 0
    assert second["output"]["resumed_records"] == 3
    assert second["output"]["sha256"] == checksum
    assert json.loads(report.read_text(encoding="utf-8"))["output"]["records"] == 3


def test_resume_refuses_changed_objective_inputs(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.jsonl"
    de_path = tmp_path / "perceptual_de.jsonl"
    output = tmp_path / "closed.jsonl"
    _jsonl(annotations, [_source("p1")])
    _jsonl(de_path, [{"preset_id": "p1", "de_med": 5.0}])
    migrate_catalog(annotations, de_path, output)
    _jsonl(de_path, [{"preset_id": "p1", "de_med": 6.0}])
    with pytest.raises(LutAnnotationError, match="stale or inconsistent"):
        migrate_catalog(annotations, de_path, output)


def _record(preset_id: str, de_med: float = 5.0) -> LutRecord:
    return LutRecord(
        preset_id=preset_id, path="unused", format="lut", name="name",
        style_major="neutral__neutral__neutral", style_minor="minor",
        scene_affinity=("portrait",), de_med=de_med, caption="caption",
        per_probe={}, hsl_features={"summary": {}, "bands": {}},
    )


def test_retrieval_covers_every_strength_bin_before_score_fill() -> None:
    catalog = LutCatalog(_record(f"p{index}", float(index + 3)) for index in range(6))
    reach = {"presets": {
        "p0": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p1": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p2": {"d_full": 4.0, "achievable_bins": ["natural"]},
        "p3": {"d_full": 6.0, "achievable_bins": ["natural", "medium"]},
        "p4": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]},
        "p5": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]},
    }}
    config = SimpleNamespace(global_major_limit=3, global_per_major_limit=4, local_limit=9)
    diagnosis = {"correction_needs": [], "enhancement_opportunities": [],
                 "forbidden_directions": []}
    shortlist = catalog.global_shortlist(
        diagnosis, {}, "portrait", config, reach, source_sha256="source",
    )
    rows = shortlist["by_major"]["neutral__neutral__neutral"]
    assert len(rows) == 4
    assert sum("bold" in row["achievable_bins"] for row in rows) == 2
    assert sum("medium" in row["achievable_bins"] for row in rows) >= 2
    assert shortlist["quota_deficits"] == []


def test_cluster_artifact_collapses_members_to_one_row(tmp_path: Path) -> None:
    path = tmp_path / "clusters.jsonl"
    _jsonl(path, [
        {"preset_id": f"p{index}", "style_major": "neutral__neutral__neutral",
         "cluster_id": index // 2}
        for index in range(6)
    ])
    clusters = load_cluster_map(path)
    catalog = LutCatalog(
        (_record(f"p{index}") for index in range(6)), clusters
    )
    reach = {"presets": {
        f"p{index}": {"d_full": 9.0, "achievable_bins": ["natural", "medium", "bold"]}
        for index in range(6)
    }}
    config = SimpleNamespace(global_major_limit=3, global_per_major_limit=6, local_limit=9)
    diagnosis = {"correction_needs": [], "enhancement_opportunities": [],
                 "forbidden_directions": []}
    rows = catalog.global_shortlist(
        diagnosis, {}, "portrait", config, reach, source_sha256="source",
    )["by_major"]["neutral__neutral__neutral"]
    assert len(rows) == 3
    assert len({row["cluster_id"] for row in rows}) == 3
