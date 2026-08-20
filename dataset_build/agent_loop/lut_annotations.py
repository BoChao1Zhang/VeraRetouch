"""Closed-set, auditable migration for offline LUT annotations."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean
from typing import Any, Mapping, Sequence

from dataset_build.src.construct.sources import SCENE_WEIGHTS


ANNOTATION_SCHEMA = "lut-closed-v1"
MIGRATION_REVISION = "lut-objective-taxonomy-v1"
TAXONOMY_REVISION = "lab-hsl-3x3x3-v1"

TEMPERATURE_THRESHOLD = 1.5
SATURATION_THRESHOLD = 5.0
LIGHTNESS_THRESHOLD = 5.0

TEMPERATURE_LABELS = ("cool", "neutral", "warm")
SATURATION_LABELS = ("desaturated", "neutral", "saturated")
LIGHTNESS_LABELS = ("dark", "neutral", "bright")
STYLE_MAJORS = tuple(
    "__".join(parts)
    for parts in itertools.product(
        TEMPERATURE_LABELS, SATURATION_LABELS, LIGHTNESS_LABELS
    )
)
SCENE_TAXONOMY = tuple(SCENE_WEIGHTS)

DEFAULT_ANNOTATIONS = Path(
    "/home/bc/data/scratch/lut_reannotate/out/annotations.jsonl"
)
DEFAULT_PERCEPTUAL_DE = Path(
    "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/perceptual_de.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/home/bc/data/scratch/lut_reannotate/out/annotations.closed-v1.jsonl"
)


class LutAnnotationError(ValueError):
    """The LUT annotation input or migrated catalog violates its contract."""


CLOSED_ANNOTATION_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Closed-set LUT annotation",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "annotation_schema", "key", "preset_id", "ok", "name", "caption", "per_probe",
        "style_major", "style_components", "style_minor", "scene_affinity",
        "de_med", "hsl_features", "provenance", "migration",
    ],
    "properties": {
        "annotation_schema": {"const": ANNOTATION_SCHEMA},
        "key": {"type": "string", "minLength": 1},
        "preset_id": {"type": "string", "minLength": 1},
        "ok": {"const": True},
        "name": {"type": "string"},
        "caption": {"type": "string"},
        "per_probe": {
            "type": "object", "additionalProperties": {"type": "string"},
        },
        "style_major": {"type": "string", "enum": list(STYLE_MAJORS)},
        "style_components": {
            "type": "object",
            "additionalProperties": False,
            "required": ["temperature", "saturation", "lightness"],
            "properties": {
                "temperature": {"enum": list(TEMPERATURE_LABELS)},
                "saturation": {"enum": list(SATURATION_LABELS)},
                "lightness": {"enum": list(LIGHTNESS_LABELS)},
            },
        },
        "style_minor": {"type": "string"},
        "scene_affinity": {
            "type": "array", "minItems": 1, "uniqueItems": True,
            "items": {"enum": list(SCENE_TAXONOMY)},
        },
        "de_med": {"type": "number", "minimum": 0.0},
        "hsl_features": {"type": "object"},
        "provenance": {"type": "object"},
        "migration": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "revision", "taxonomy_revision", "source_annotation_sha256",
                "annotations_source_sha256", "perceptual_de_source_sha256",
                "scene_derivation", "scene_override_provenance",
            ],
            "properties": {
                "revision": {"const": MIGRATION_REVISION},
                "taxonomy_revision": {"const": TAXONOMY_REVISION},
                "source_annotation_sha256": {"type": "string", "minLength": 64},
                "annotations_source_sha256": {"type": "string", "minLength": 64},
                "perceptual_de_source_sha256": {"type": "string", "minLength": 64},
                "scene_derivation": {"type": "string", "minLength": 1},
                "scene_override_provenance": {
                    "type": ["object", "null"],
                },
            },
        },
    },
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LutAnnotationError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise LutAnnotationError(f"{field} must be finite")
    return result


def _three_way(value: float, threshold: float, labels: Sequence[str]) -> str:
    if value < -threshold:
        return labels[0]
    if value > threshold:
        return labels[2]
    return labels[1]


def style_components_from_hsl(hsl_features: Mapping[str, Any]) -> dict[str, str]:
    """Derive the finite style axes from objective LUT response features."""
    summary = hsl_features.get("summary")
    if not isinstance(summary, Mapping):
        raise LutAnnotationError("hsl_features.summary must be an object")
    # Lab b* is the yellow/blue (warm/cool) axis. Lab a* remains available to
    # retrieval as tint evidence but must not turn a magenta/green tint into a
    # temperature class.
    mid_b = _finite_number(summary.get("mid_gray_b"), "summary.mid_gray_b")
    saturation = _finite_number(
        summary.get("sat_pct_mean"), "summary.sat_pct_mean"
    )
    lightness = _finite_number(summary.get("mid_gray_dL"), "summary.mid_gray_dL")
    return {
        "temperature": _three_way(
            mid_b, TEMPERATURE_THRESHOLD, TEMPERATURE_LABELS
        ),
        "saturation": _three_way(
            saturation, SATURATION_THRESHOLD, SATURATION_LABELS
        ),
        "lightness": _three_way(
            lightness, LIGHTNESS_THRESHOLD, LIGHTNESS_LABELS
        ),
    }


def style_major_from_hsl(hsl_features: Mapping[str, Any]) -> str:
    components = style_components_from_hsl(hsl_features)
    return "__".join(
        components[key] for key in ("temperature", "saturation", "lightness")
    )


def scenes_from_cached_affinity(value: Any) -> tuple[str, ...]:
    """Losslessly expand the legacy broad affinity onto the canonical taxonomy."""
    legacy = str(value or "").strip().lower()
    if legacy in {"general", "any"}:
        return SCENE_TAXONOMY
    if legacy in SCENE_TAXONOMY:
        return (legacy,)
    raise LutAnnotationError(f"unsupported cached scene_affinity: {legacy!r}")


def _validate_scene_affinity(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise LutAnnotationError("scene_affinity must be a non-empty array")
    if any(not isinstance(item, str) or item not in SCENE_TAXONOMY for item in value):
        raise LutAnnotationError("scene_affinity contains a non-canonical scene")
    if len(value) != len(set(value)):
        raise LutAnnotationError("scene_affinity must not contain duplicates")
    canonical = tuple(scene for scene in SCENE_TAXONOMY if scene in value)
    if tuple(value) != canonical:
        raise LutAnnotationError("scene_affinity must follow canonical scene order")
    return canonical


def validate_closed_annotation(row: Mapping[str, Any]) -> None:
    """Validate the invariants used by the runtime catalog without optional deps."""
    required = set(CLOSED_ANNOTATION_SCHEMA["required"])
    missing = sorted(required - set(row))
    unknown = sorted(set(row) - set(CLOSED_ANNOTATION_SCHEMA["properties"]))
    if missing:
        raise LutAnnotationError(f"missing required fields: {missing}")
    if unknown:
        raise LutAnnotationError(f"unknown fields in closed annotation: {unknown}")
    if row.get("annotation_schema") != ANNOTATION_SCHEMA:
        raise LutAnnotationError(f"annotation_schema must be {ANNOTATION_SCHEMA!r}")
    if row.get("ok") is not True:
        raise LutAnnotationError("closed annotation must have ok=true")
    preset_id = row.get("preset_id")
    if not isinstance(preset_id, str) or not preset_id:
        raise LutAnnotationError("preset_id must be a non-empty string")
    if row.get("key") != preset_id:
        raise LutAnnotationError("key must equal preset_id")
    if "strength" in row:
        raise LutAnnotationError("legacy strength is forbidden; use numeric de_med")
    for field in ("name", "caption", "style_minor"):
        if not isinstance(row.get(field), str):
            raise LutAnnotationError(f"{field} must be a string")
    per_probe = row.get("per_probe")
    if not isinstance(per_probe, Mapping):
        raise LutAnnotationError("per_probe must be an object")
    if any(not isinstance(key, str) or not isinstance(value, str)
           for key, value in per_probe.items()):
        raise LutAnnotationError("per_probe keys and values must be strings")
    hsl_features = row.get("hsl_features")
    if not isinstance(hsl_features, Mapping):
        raise LutAnnotationError("hsl_features must be an object")
    if not isinstance(row.get("provenance"), Mapping):
        raise LutAnnotationError("provenance must be an object")
    expected_components = style_components_from_hsl(hsl_features)
    if row.get("style_components") != expected_components:
        raise LutAnnotationError("style_components disagree with objective HSL features")
    expected_major = style_major_from_hsl(hsl_features)
    if row.get("style_major") != expected_major or expected_major not in STYLE_MAJORS:
        raise LutAnnotationError("style_major disagrees with objective HSL features")
    _validate_scene_affinity(row.get("scene_affinity"))
    de_med = _finite_number(row.get("de_med"), "de_med")
    if de_med < 0:
        raise LutAnnotationError("de_med must be non-negative")
    migration = row.get("migration")
    if not isinstance(migration, Mapping):
        raise LutAnnotationError("migration must be an object")
    for field in (
        "revision", "taxonomy_revision", "source_annotation_sha256",
        "annotations_source_sha256", "perceptual_de_source_sha256",
        "scene_derivation",
    ):
        if not isinstance(migration.get(field), str) or not migration[field]:
            raise LutAnnotationError(f"migration.{field} must be a non-empty string")
    if migration.get("revision") != MIGRATION_REVISION:
        raise LutAnnotationError("migration revision mismatch")
    if migration.get("taxonomy_revision") != TAXONOMY_REVISION:
        raise LutAnnotationError("taxonomy revision mismatch")
    override_provenance = migration.get("scene_override_provenance")
    if override_provenance is not None and not isinstance(override_provenance, Mapping):
        raise LutAnnotationError("scene_override_provenance must be an object or null")


def _read_latest_successes(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    rows: dict[str, dict[str, Any]] = {}
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            counts["lines"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LutAnnotationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise LutAnnotationError(f"{path}:{line_number}: row must be an object")
            preset_id = str(row.get("preset_id") or row.get("key") or "")
            if not preset_id:
                raise LutAnnotationError(f"{path}:{line_number}: missing preset_id")
            if row.get("ok") is True:
                if preset_id in rows:
                    counts["replaced_successes"] += 1
                rows[preset_id] = row
                counts["successful_lines"] += 1
            else:
                counts["failed_lines"] += 1
    counts["unique_successes"] = len(rows)
    return rows, dict(counts)


def _read_de_med(path: Path) -> tuple[dict[str, float], int]:
    result: dict[str, float] = {}
    lines = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LutAnnotationError(f"{path}:{line_number}: invalid JSON") from exc
            preset_id = str(row.get("preset_id") or "")
            if not preset_id:
                raise LutAnnotationError(f"{path}:{line_number}: missing preset_id")
            if preset_id in result:
                raise LutAnnotationError(f"duplicate de_med for {preset_id}")
            value = _finite_number(row.get("de_med"), f"{preset_id}.de_med")
            if value < 0:
                raise LutAnnotationError(f"{preset_id}.de_med must be non-negative")
            result[preset_id] = value
    return result, lines


def _read_scene_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LutAnnotationError(f"{path}:{line_number}: invalid JSON") from exc
            preset_id = str(row.get("preset_id") or "")
            if not preset_id or preset_id in result:
                raise LutAnnotationError(
                    f"{path}:{line_number}: missing or duplicate preset_id"
                )
            _validate_scene_affinity(row.get("scene_affinity"))
            provenance = row.get("provenance")
            if not isinstance(provenance, Mapping):
                raise LutAnnotationError(
                    f"{path}:{line_number}: override provenance must be an object"
                )
            result[preset_id] = row
    return result


def migrate_annotation(
    source: Mapping[str, Any], de_med: float, *,
    annotations_source_sha256: str, perceptual_de_source_sha256: str,
    scene_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preset_id = str(source.get("preset_id") or source.get("key") or "")
    if not preset_id or source.get("ok") is not True:
        raise LutAnnotationError("source must be a successful annotation with preset_id")
    hsl_features = source.get("hsl_features")
    if not isinstance(hsl_features, Mapping):
        raise LutAnnotationError(f"{preset_id}: missing hsl_features")
    components = style_components_from_hsl(hsl_features)
    if scene_override is None:
        scenes = scenes_from_cached_affinity(source.get("scene_affinity"))
        scene_derivation = "cached-lutreannot-v1-affinity-map"
        scene_override_provenance = None
    else:
        scenes = _validate_scene_affinity(scene_override.get("scene_affinity"))
        scene_derivation = "semantic-override"
        scene_override_provenance = dict(scene_override.get("provenance") or {})
    result = {
        "annotation_schema": ANNOTATION_SCHEMA,
        "key": preset_id,
        "preset_id": preset_id,
        "ok": True,
        "name": str(source.get("name") or preset_id),
        "per_probe": dict(source.get("per_probe") or {}),
        "caption": str(source.get("caption") or ""),
        "style_major": "__".join(
            components[key] for key in ("temperature", "saturation", "lightness")
        ),
        "style_components": components,
        "style_minor": str(source.get("style_minor") or source.get("name") or preset_id),
        "scene_affinity": list(scenes),
        "de_med": float(de_med),
        "hsl_features": dict(hsl_features),
        "provenance": dict(source.get("provenance") or {}),
        "migration": {
            "revision": MIGRATION_REVISION,
            "taxonomy_revision": TAXONOMY_REVISION,
            "source_annotation_sha256": _value_sha256(source),
            "annotations_source_sha256": annotations_source_sha256,
            "perceptual_de_source_sha256": perceptual_de_source_sha256,
            "scene_derivation": scene_derivation,
            "scene_override_provenance": scene_override_provenance,
        },
    }
    validate_closed_annotation(result)
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise LutAnnotationError("cannot compute quantile of an empty catalog")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _load_resume(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LutAnnotationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise LutAnnotationError(f"{path}:{line_number}: row must be an object")
            validate_closed_annotation(row)
            preset_id = row["preset_id"]
            if preset_id in result:
                raise LutAnnotationError(f"{path}: duplicate preset_id {preset_id}")
            result[preset_id] = row
    return result


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def migrate_catalog(
    annotations: str | Path, perceptual_de: str | Path, output: str | Path,
    *, report: str | Path | None = None,
    scene_overrides: str | Path | None = None,
) -> dict[str, Any]:
    annotations_path = Path(annotations)
    de_path = Path(perceptual_de)
    output_path = Path(output)
    report_path = Path(report) if report else output_path.with_suffix(".report.json")
    override_path = Path(scene_overrides) if scene_overrides else None

    annotations_sha = file_sha256(annotations_path)
    de_sha = file_sha256(de_path)
    sources, source_counts = _read_latest_successes(annotations_path)
    de_values, de_lines = _read_de_med(de_path)
    overrides = _read_scene_overrides(override_path)
    missing_de = sorted(set(sources) - set(de_values))
    unknown_overrides = sorted(set(overrides) - set(sources))
    if missing_de:
        raise LutAnnotationError(
            f"{len(missing_de)} annotations lack de_med; first={missing_de[0]}"
        )
    if unknown_overrides:
        raise LutAnnotationError(
            f"{len(unknown_overrides)} scene overrides are unknown; first={unknown_overrides[0]}"
        )

    expected = {
        preset_id: migrate_annotation(
            source, de_values[preset_id],
            annotations_source_sha256=annotations_sha,
            perceptual_de_source_sha256=de_sha,
            scene_override=overrides.get(preset_id),
        )
        for preset_id, source in sorted(sources.items())
    }
    existing = _load_resume(output_path)
    unknown_existing = sorted(set(existing) - set(expected))
    if unknown_existing:
        raise LutAnnotationError(
            f"resume output contains unknown preset; first={unknown_existing[0]}"
        )
    for preset_id, row in existing.items():
        if row != expected[preset_id]:
            raise LutAnnotationError(
                f"resume output is stale or inconsistent for {preset_id}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for preset_id, row in expected.items():
            if preset_id in existing:
                continue
            handle.write(_canonical_bytes(row).decode("utf-8") + "\n")
            handle.flush()
            written += 1

    final = _load_resume(output_path)
    if set(final) != set(expected):
        raise LutAnnotationError("final output ID set does not match source annotations")
    for row in final.values():
        validate_closed_annotation(row)

    major_counts = Counter(row["style_major"] for row in final.values())
    component_counts = {
        axis: dict(sorted(Counter(
            row["style_components"][axis] for row in final.values()
        ).items()))
        for axis in ("temperature", "saturation", "lightness")
    }
    scene_counts = Counter(
        scene for row in final.values() for scene in row["scene_affinity"]
    )
    legacy_scene_counts = Counter(
        str(row.get("scene_affinity") or "") for row in sources.values()
    )
    values = [float(row["de_med"]) for row in final.values()]
    report_value = {
        "annotation_schema": ANNOTATION_SCHEMA,
        "migration_revision": MIGRATION_REVISION,
        "taxonomy_revision": TAXONOMY_REVISION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "annotations": str(annotations_path),
            "annotations_sha256": annotations_sha,
            "perceptual_de": str(de_path),
            "perceptual_de_sha256": de_sha,
            "scene_overrides": str(override_path) if override_path else None,
            "scene_overrides_sha256": file_sha256(override_path) if override_path else None,
        },
        "output": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "records": len(final),
            "validated_records": len(final),
            "written_this_run": written,
            "resumed_records": len(existing),
        },
        "source_counts": {
            **source_counts,
            "perceptual_de_lines": de_lines,
            "perceptual_de_matched": len(final),
            "perceptual_de_unused": len(de_values) - len(final),
            "scene_overrides": len(overrides),
            "semantic_model_calls": 0,
        },
        "taxonomy": {
            "thresholds": {
                "temperature_mid_gray_b": TEMPERATURE_THRESHOLD,
                "saturation_sat_pct_mean": SATURATION_THRESHOLD,
                "lightness_mid_gray_dL": LIGHTNESS_THRESHOLD,
            },
            "possible_style_majors": len(STYLE_MAJORS),
            "observed_style_majors": len(major_counts),
            "minimum_major_size": min(major_counts.values()),
            "majors_at_least_3": sum(value >= 3 for value in major_counts.values()),
            "majors_at_least_50": sum(value >= 50 for value in major_counts.values()),
            "major_counts": dict(sorted(major_counts.items())),
            "component_counts": component_counts,
        },
        "scenes": {
            "taxonomy": list(SCENE_TAXONOMY),
            "membership_counts": dict(sorted(scene_counts.items())),
            "legacy_counts": dict(sorted(legacy_scene_counts.items())),
        },
        "de_med": {
            "min": min(values), "p05": _quantile(values, 0.05),
            "median": _quantile(values, 0.5), "mean": fmean(values),
            "p95": _quantile(values, 0.95), "max": max(values),
        },
    }
    _write_json_atomic(report_path, report_value)
    return report_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--perceptual-de", type=Path, default=DEFAULT_PERCEPTUAL_DE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--scene-overrides", type=Path,
        help="Optional audited JSONL overrides for genuinely semantic scene refinements",
    )
    parser.add_argument("--schema-out", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.schema_out:
        _write_json_atomic(args.schema_out, CLOSED_ANNOTATION_SCHEMA)
    report = migrate_catalog(
        args.annotations, args.perceptual_de, args.output,
        report=args.report, scene_overrides=args.scene_overrides,
    )
    print(json.dumps({
        "output": report["output"],
        "taxonomy": {
            key: report["taxonomy"][key]
            for key in (
                "observed_style_majors", "minimum_major_size", "majors_at_least_50"
            )
        },
        "semantic_model_calls": report["source_counts"]["semantic_model_calls"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ANNOTATION_SCHEMA", "CLOSED_ANNOTATION_SCHEMA", "LIGHTNESS_THRESHOLD",
    "LutAnnotationError", "MIGRATION_REVISION", "SATURATION_THRESHOLD",
    "SCENE_TAXONOMY", "STYLE_MAJORS", "TAXONOMY_REVISION",
    "TEMPERATURE_THRESHOLD", "migrate_annotation", "migrate_catalog",
    "scenes_from_cached_affinity", "style_components_from_hsl",
    "style_major_from_hsl", "validate_closed_annotation",
]
