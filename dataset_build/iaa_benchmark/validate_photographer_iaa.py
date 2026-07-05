#!/usr/bin/env python3
"""Validate a Photographer-IAA benchmark directory."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXPECTED_CATEGORIES = {
    "animal_pet",
    "building",
    "food",
    "indoor",
    "landscape_scene",
    "night_scene",
    "plant",
    "portrait",
    "still_life",
}

EXPECTED_MODES = {
    "success_high_all",
    "low_aesthetic",
    "composition_failure",
    "color_failure",
    "light_failure",
    "dof_failure",
    "quality_failure",
    "content_failure",
}

MODE_FLAG = {
    "success_high_all": "flag_success_high_all",
    "low_aesthetic": "flag_low_aesthetic",
    "composition_failure": "flag_composition_failure",
    "color_failure": "flag_color_failure",
    "light_failure": "flag_light_failure",
    "dof_failure": "flag_dof_failure",
    "quality_failure": "flag_quality_failure",
    "content_failure": "flag_content_failure",
}

REQUIRED_COLUMNS = {
    "benchmark_id",
    "source",
    "source_split",
    "image_name",
    "session_id",
    "semantic",
    "category",
    "image_path",
    "source_zip_member",
    "primary_mode",
    "mode_attribute",
    "mode_score_1_5",
    "mode_percentile_in_category",
    "rater_count",
    "gt_aesthetic_mean_1_5",
    "gt_aesthetic_mean_0_100",
    "gt_aesthetic_std",
    "gt_quality_mean_1_5",
    "gt_composition_mean_1_5",
    "gt_color_mean_1_5",
    "gt_dof_mean_1_5",
    "gt_light_mean_1_5",
    "gt_content_mean_1_5",
    "flag_success_high_all",
    "flag_low_aesthetic",
    "flag_quality_failure",
    "flag_composition_failure",
    "flag_color_failure",
    "flag_dof_failure",
    "flag_light_failure",
    "flag_content_failure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Photographer-IAA benchmark artifacts.")
    parser.add_argument("benchmark_dir", type=Path, help="Directory containing metadata.csv and summary.json.")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Do not require image_path files to exist.",
    )
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def validate_score_range(row: dict[str, str], column: str, low: float, high: float, errors: list[str]) -> None:
    try:
        value = float(row[column])
    except (KeyError, ValueError):
        fail(errors, f"{row.get('benchmark_id', '<unknown>')}: {column} is not numeric")
        return
    if not low <= value <= high:
        fail(errors, f"{row['benchmark_id']}: {column}={value} outside [{low}, {high}]")


def validate(benchmark_dir: Path, skip_images: bool) -> dict[str, Any]:
    benchmark_dir = benchmark_dir.expanduser()
    errors: list[str] = []
    metadata_path = benchmark_dir / "metadata.csv"
    jsonl_path = benchmark_dir / "metadata.jsonl"
    summary_path = benchmark_dir / "summary.json"

    for path in (metadata_path, jsonl_path, summary_path):
        if not path.exists():
            fail(errors, f"missing required file: {path}")
    if errors:
        return {"ok": False, "errors": errors}

    rows = read_metadata(metadata_path)
    with summary_path.open("r", encoding="utf-8") as fh:
        summary = json.load(fh)

    if not rows:
        fail(errors, "metadata.csv has no rows")
        return {"ok": False, "errors": errors}

    missing_columns = REQUIRED_COLUMNS - set(rows[0])
    if missing_columns:
        fail(errors, f"metadata.csv missing columns: {sorted(missing_columns)}")

    if len(rows) != int(summary.get("selected_rows", -1)):
        fail(errors, f"row count {len(rows)} != summary selected_rows {summary.get('selected_rows')}")

    with jsonl_path.open("r", encoding="utf-8") as fh:
        jsonl_count = sum(1 for _ in fh)
    if jsonl_count != len(rows):
        fail(errors, f"metadata.jsonl line count {jsonl_count} != metadata.csv row count {len(rows)}")

    ids = [row["benchmark_id"] for row in rows]
    names = [row["image_name"] for row in rows]
    if len(set(ids)) != len(ids):
        fail(errors, "benchmark_id values are not unique")
    if len(set(names)) != len(names):
        fail(errors, "image_name values are not unique")

    category_counts = Counter(row["category"] for row in rows)
    mode_counts = Counter(row["primary_mode"] for row in rows)
    if set(category_counts) != EXPECTED_CATEGORIES:
        fail(errors, f"category set mismatch: {sorted(category_counts)}")
    if set(mode_counts) != EXPECTED_MODES:
        fail(errors, f"mode set mismatch: {sorted(mode_counts)}")

    expected_cell_size = summary.get("per_category_mode")
    cell_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        cell_counts[row["category"]][row["primary_mode"]] += 1
    if expected_cell_size is not None:
        for category in EXPECTED_CATEGORIES:
            for mode in EXPECTED_MODES:
                count = cell_counts[category][mode]
                if count != expected_cell_size:
                    fail(errors, f"{category}/{mode} count {count} != {expected_cell_size}")

    for row in rows:
        if row["source"] != "PARA":
            fail(errors, f"{row['benchmark_id']}: source is not PARA")
        if row["primary_mode"] not in MODE_FLAG:
            fail(errors, f"{row['benchmark_id']}: unknown mode {row['primary_mode']}")
        else:
            flag = MODE_FLAG[row["primary_mode"]]
            if row.get(flag) != "1":
                fail(errors, f"{row['benchmark_id']}: {row['primary_mode']} does not set {flag}")
        for column in (
            "gt_aesthetic_mean_1_5",
            "gt_quality_mean_1_5",
            "gt_composition_mean_1_5",
            "gt_color_mean_1_5",
            "gt_dof_mean_1_5",
            "gt_light_mean_1_5",
            "gt_content_mean_1_5",
            "mode_score_1_5",
        ):
            validate_score_range(row, column, 1.0, 5.0, errors)
        for column in ("gt_aesthetic_mean_0_100", "mode_percentile_in_category"):
            high = 100.0 if column.endswith("_0_100") else 1.0
            validate_score_range(row, column, 0.0, high, errors)
        try:
            if int(row["rater_count"]) <= 0:
                fail(errors, f"{row['benchmark_id']}: rater_count must be positive")
        except ValueError:
            fail(errors, f"{row['benchmark_id']}: rater_count is not an integer")
        if not skip_images:
            image_path = row.get("image_path", "")
            if not image_path:
                fail(errors, f"{row['benchmark_id']}: image_path is empty")
            elif not (benchmark_dir / image_path).exists():
                fail(errors, f"{row['benchmark_id']}: missing image file {image_path}")

    if summary.get("sampling_shortfalls"):
        fail(errors, f"summary reports sampling shortfalls: {summary['sampling_shortfalls']}")

    return {
        "ok": not errors,
        "errors": errors,
        "rows": len(rows),
        "categories": dict(sorted(category_counts.items())),
        "modes": dict(sorted(mode_counts.items())),
        "per_category_mode": expected_cell_size,
    }


def main() -> None:
    args = parse_args()
    result = validate(args.benchmark_dir, args.skip_images)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
