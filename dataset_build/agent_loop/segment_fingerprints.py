"""Offline segmented response fingerprints derived from the closed-v1 LUT catalog.

R6.1 (`docs/REQUIREMENTS_local_visibility_lut_selection_20260820.md`): every LUT gets a
shadows/mids/highlights x (dL, dC, d_hue) nine-tuple. Nothing new is measured; the whole
file is a deterministic re-read of `hsl_features` that already lives in the closed-v1
annotations, so it can be rebuilt byte-for-byte from the recorded input SHA.

Derivation, in full:

* `dL` and the neutral cast (`cast_a` / `cast_b`) come from `hsl_features.neutral_ramp`,
  the five neutral probe points at input levels 0.15 / 0.30 / 0.50 / 0.70 / 0.85.
  0.15 + 0.30 are the shadows segment, 0.50 is mids, 0.70 + 0.85 is highlights.
  `dL` = mean(`L_out` - `L_in`) over the points of the segment; `cast_a` / `cast_b` =
  mean(`a_out`) / mean(`b_out`) over the same points.
* `dC` and `d_hue` come from the eight hue `bands` (`d_sat_pct`, `d_hue_deg`). The bands
  carry no tonal segmentation, so they are assigned to a segment by the *sign of their
  `d_lum_pct`*: a band the LUT lifts (`d_lum_pct` > `BAND_SEGMENT_LUM_THRESHOLD`) is read
  as acting in the highlights, a band it darkens (< -threshold) in the shadows, and the
  rest in the mids. `dC` / `d_hue` = plain arithmetic mean of `d_sat_pct` / `d_hue_deg`
  over the bands assigned to the segment; a segment with no assigned band falls back to
  the mean over all eight bands. This is an approximation and is recorded as such in
  `bands_per_segment` on every row.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .lut_annotations import DEFAULT_OUTPUT as DEFAULT_CLOSED_ANNOTATIONS
from .lut_annotations import file_sha256


SEGMENT_FINGERPRINT_SCHEMA = "lut-segment-fingerprint-v1"
# Bump this whenever the arithmetic below changes; loaders reject a foreign revision.
DERIVATION_REVISION = "ramp-band-lumsign-v1"

SEGMENT_NAMES: tuple[str, ...] = ("shadows", "mids", "highlights")
SEGMENT_FIELDS: tuple[str, ...] = ("dL", "dC", "d_hue", "cast_a", "cast_b")
# The nine-tuple of R6.1 proper; `cast_a`/`cast_b` are the neutral colour cast carried
# alongside it because the R6.2 cast axis needs a Lab direction per segment.
HEADLINE_FIELDS: tuple[str, ...] = ("dL", "dC", "d_hue")

RAMP_SEGMENTS: dict[str, tuple[float, ...]] = {
    "shadows": (0.15, 0.30), "mids": (0.50,), "highlights": (0.70, 0.85),
}
BAND_SEGMENT_LUM_THRESHOLD = 2.0
# L* boundaries that split a measured pixel population the same way `RAMP_SEGMENTS`
# splits the neutral ramp: midpoints of the neighbouring ramp probes' L_in values
# (~32.5 / 53.4 / 72.8 on the reference catalog).
SEGMENT_L_BOUNDS: tuple[float, float] = (43.0, 63.0)
ROUND_DIGITS = 4

DEFAULT_ANNOTATIONS = DEFAULT_CLOSED_ANNOTATIONS
DEFAULT_FINGERPRINTS = Path(
    "/home/bc/data/scratch/lut_reannotate/out/segment_fingerprints.v1.jsonl"
)
# SHA-256 of the production artifact derived from `annotations.closed-v1.jsonl`
# (4,051 rows). B11 item 6 puts it into `prompt_revision_fingerprint()`: the table
# decides which LUTs the online local retrieval can even see, so a different table is a
# different prompt revision. `LutCatalog.segment_fingerprints_sha256` reports what was
# actually mounted, which is what the runtime assertion compares against.
SEGMENT_FINGERPRINT_TABLE_SHA256 = (
    "bac4db04e402b0eaefb702502027f7287b99ee50e8fa76b6b48871a9f0774939"
)


class SegmentFingerprintError(ValueError):
    """The segmented fingerprint input or artifact violates its contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _value_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SegmentFingerprintError(f"{field} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise SegmentFingerprintError(f"{field} must be finite")
    return result


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _ramp_points(hsl_features: Mapping[str, Any]) -> dict[float, dict[str, float]]:
    ramp = hsl_features.get("neutral_ramp")
    if not isinstance(ramp, list) or not ramp:
        raise SegmentFingerprintError("hsl_features.neutral_ramp must be a non-empty array")
    points: dict[float, dict[str, float]] = {}
    for index, entry in enumerate(ramp):
        if not isinstance(entry, Mapping):
            raise SegmentFingerprintError(f"neutral_ramp[{index}] must be an object")
        level = round(_finite(entry.get("in"), f"neutral_ramp[{index}].in"), 3)
        points[level] = {
            field: _finite(entry.get(field), f"neutral_ramp[{index}].{field}")
            for field in ("L_in", "L_out", "a_out", "b_out")
        }
    missing = sorted(
        level for levels in RAMP_SEGMENTS.values() for level in levels
        if level not in points
    )
    if missing:
        raise SegmentFingerprintError(f"neutral_ramp is missing input levels: {missing}")
    return points


def _bands(hsl_features: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    bands = hsl_features.get("bands")
    if not isinstance(bands, Mapping) or not bands:
        raise SegmentFingerprintError("hsl_features.bands must be a non-empty object")
    result: dict[str, dict[str, float]] = {}
    for name, entry in bands.items():
        if not isinstance(entry, Mapping):
            raise SegmentFingerprintError(f"bands.{name} must be an object")
        result[str(name)] = {
            field: _finite(entry.get(field), f"bands.{name}.{field}")
            for field in ("d_sat_pct", "d_hue_deg", "d_lum_pct")
        }
    return result


def assign_bands(bands: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    """Approximate tonal placement of each hue band by the sign of its `d_lum_pct`."""
    assignment: dict[str, list[str]] = {name: [] for name in SEGMENT_NAMES}
    for name in sorted(bands):
        d_lum = float(bands[name]["d_lum_pct"])
        if d_lum > BAND_SEGMENT_LUM_THRESHOLD:
            assignment["highlights"].append(name)
        elif d_lum < -BAND_SEGMENT_LUM_THRESHOLD:
            assignment["shadows"].append(name)
        else:
            assignment["mids"].append(name)
    return assignment


def derive_segment_fingerprint(
    hsl_features: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Shadows/mids/highlights x (dL, dC, d_hue, cast_a, cast_b), rounded and pure."""
    points = _ramp_points(hsl_features)
    bands = _bands(hsl_features)
    assignment = assign_bands(bands)
    all_bands = sorted(bands)
    result: dict[str, dict[str, float]] = {}
    for segment in SEGMENT_NAMES:
        levels = RAMP_SEGMENTS[segment]
        members = assignment[segment] or all_bands
        result[segment] = {
            "dL": round(
                _mean([points[level]["L_out"] - points[level]["L_in"] for level in levels]),
                ROUND_DIGITS,
            ),
            "dC": round(
                _mean([bands[name]["d_sat_pct"] for name in members]), ROUND_DIGITS
            ),
            "d_hue": round(
                _mean([bands[name]["d_hue_deg"] for name in members]), ROUND_DIGITS
            ),
            "cast_a": round(
                _mean([points[level]["a_out"] for level in levels]), ROUND_DIGITS
            ),
            "cast_b": round(
                _mean([points[level]["b_out"] for level in levels]), ROUND_DIGITS
            ),
        }
    return result


def validate_segment_fingerprint_row(row: Mapping[str, Any]) -> None:
    if row.get("schema") != SEGMENT_FINGERPRINT_SCHEMA:
        raise SegmentFingerprintError(
            f"schema must be {SEGMENT_FINGERPRINT_SCHEMA!r}"
        )
    if row.get("derivation_revision") != DERIVATION_REVISION:
        raise SegmentFingerprintError(
            f"derivation_revision must be {DERIVATION_REVISION!r}"
        )
    preset_id = row.get("preset_id")
    if not isinstance(preset_id, str) or not preset_id:
        raise SegmentFingerprintError("preset_id must be a non-empty string")
    for field in ("source_sha256", "hsl_features_sha256"):
        value = row.get(field)
        if not isinstance(value, str) or len(value) != 64:
            raise SegmentFingerprintError(f"{field} must be a sha256 hex digest")
    segments = row.get("segments")
    if not isinstance(segments, Mapping) or set(segments) != set(SEGMENT_NAMES):
        raise SegmentFingerprintError(f"segments must have exactly {list(SEGMENT_NAMES)}")
    for name, values in segments.items():
        if not isinstance(values, Mapping) or set(values) != set(SEGMENT_FIELDS):
            raise SegmentFingerprintError(
                f"segments.{name} must have exactly {list(SEGMENT_FIELDS)}"
            )
        for field, value in values.items():
            _finite(value, f"segments.{name}.{field}")


def build_row(
    preset_id: str, hsl_features: Mapping[str, Any], *, source_sha256: str
) -> dict[str, Any]:
    bands = _bands(hsl_features)
    assignment = assign_bands(bands)
    row = {
        "schema": SEGMENT_FINGERPRINT_SCHEMA,
        "preset_id": preset_id,
        "derivation_revision": DERIVATION_REVISION,
        "source_sha256": source_sha256,
        "hsl_features_sha256": _value_sha256(hsl_features),
        "bands_per_segment": {
            name: len(assignment[name]) for name in SEGMENT_NAMES
        },
        "segments": derive_segment_fingerprint(hsl_features),
    }
    validate_segment_fingerprint_row(row)
    return row


def load_segment_fingerprints(
    path: str | Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Read the derived artifact into `preset_id -> segment -> field -> value`."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SegmentFingerprintError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise SegmentFingerprintError(f"{path}:{line_number}: row must be an object")
            validate_segment_fingerprint_row(row)
            preset_id = row["preset_id"]
            if preset_id in result:
                raise SegmentFingerprintError(f"{path}: duplicate preset_id {preset_id}")
            result[preset_id] = {
                name: {field: float(value) for field, value in values.items()}
                for name, values in row["segments"].items()
            }
    if not result:
        raise SegmentFingerprintError(f"{path}: segment fingerprint artifact is empty")
    return result


def _quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build_segment_fingerprints(
    annotations: str | Path, output: str | Path, *, report: str | Path | None = None,
) -> dict[str, Any]:
    """Derive the whole artifact from closed-v1 annotations; deterministic per input SHA."""
    annotations_path = Path(annotations)
    output_path = Path(output)
    report_path = Path(report) if report else output_path.with_suffix(".report.json")
    source_sha = file_sha256(annotations_path)

    rows: dict[str, dict[str, Any]] = {}
    with annotations_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SegmentFingerprintError(
                    f"{annotations_path}:{line_number}: invalid JSON"
                ) from exc
            if record.get("ok") is not True:
                continue
            preset_id = str(record.get("preset_id") or record.get("key") or "")
            if not preset_id:
                raise SegmentFingerprintError(
                    f"{annotations_path}:{line_number}: missing preset_id"
                )
            if preset_id in rows:
                raise SegmentFingerprintError(
                    f"{annotations_path}: duplicate preset_id {preset_id}"
                )
            hsl_features = record.get("hsl_features")
            if not isinstance(hsl_features, Mapping):
                raise SegmentFingerprintError(f"{preset_id}: missing hsl_features")
            rows[preset_id] = build_row(
                preset_id, hsl_features, source_sha256=source_sha
            )
    if not rows:
        raise SegmentFingerprintError(f"{annotations_path}: no successful annotations")

    payload = b"".join(
        _canonical_bytes(rows[preset_id]) + b"\n" for preset_id in sorted(rows)
    )
    _write_atomic(output_path, payload)

    band_counts: Counter[str] = Counter()
    empty_segments: Counter[str] = Counter()
    for row in rows.values():
        for name, count in row["bands_per_segment"].items():
            band_counts[name] += count
            if count == 0:
                empty_segments[name] += 1
    distribution: dict[str, dict[str, float]] = {}
    for segment in SEGMENT_NAMES:
        for field in SEGMENT_FIELDS:
            series = [row["segments"][segment][field] for row in rows.values()]
            distribution[f"{segment}.{field}"] = {
                "min": round(min(series), 4),
                "p05": round(_quantile(series, 0.05), 4),
                "p25": round(_quantile(series, 0.25), 4),
                "p50": round(_quantile(series, 0.50), 4),
                "p75": round(_quantile(series, 0.75), 4),
                "p95": round(_quantile(series, 0.95), 4),
                "max": round(max(series), 4),
            }
    report_value = {
        "schema": SEGMENT_FINGERPRINT_SCHEMA,
        "derivation_revision": DERIVATION_REVISION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "annotations": str(annotations_path), "annotations_sha256": source_sha,
        },
        "output": {
            "path": str(output_path), "sha256": file_sha256(output_path),
            "records": len(rows), "bytes": len(payload),
        },
        "derivation": {
            "ramp_segments": {k: list(v) for k, v in RAMP_SEGMENTS.items()},
            "band_segment_lum_threshold": BAND_SEGMENT_LUM_THRESHOLD,
            "segment_l_bounds": list(SEGMENT_L_BOUNDS),
            "round_digits": ROUND_DIGITS,
            "bands_assigned_total": dict(sorted(band_counts.items())),
            "presets_with_no_band_in_segment": dict(sorted(empty_segments.items())),
        },
        "distribution": dict(sorted(distribution.items())),
    }
    _write_atomic(
        report_path,
        (json.dumps(report_value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        .encode("utf-8"),
    )
    return report_value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Derive R6.1 segment fingerprints")
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_FINGERPRINTS)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = build_segment_fingerprints(
        args.annotations, args.output, report=args.report
    )
    print(json.dumps({
        "output": report["output"], "derivation_revision": report["derivation_revision"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BAND_SEGMENT_LUM_THRESHOLD", "DERIVATION_REVISION", "DEFAULT_FINGERPRINTS",
    "HEADLINE_FIELDS", "RAMP_SEGMENTS", "SEGMENT_FIELDS", "SEGMENT_FINGERPRINT_SCHEMA",
    "SEGMENT_L_BOUNDS", "SEGMENT_NAMES", "SegmentFingerprintError", "assign_bands",
    "SEGMENT_FINGERPRINT_TABLE_SHA256",
    "build_row", "build_segment_fingerprints", "derive_segment_fingerprint",
    "load_segment_fingerprints", "validate_segment_fingerprint_row",
]
