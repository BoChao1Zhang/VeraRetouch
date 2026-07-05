#!/usr/bin/env python3
"""Export the public ArtiMuse-10K test split into benchmark-style inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export ArtiMuse-10K test inputs.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/bc/data/datasets/ArtiMuse"),
        help="Directory containing ArtiMuse-10K_test.json and image/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/bc/data/datasets/artimuse10k_eval"),
        help="Output directory for metadata and model inputs.",
    )
    return parser.parse_args()


def normalize_1_10_to_0_100(score: float) -> float:
    return max(0.0, min(100.0, (score - 1.0) / 9.0 * 100.0))


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = dataset_root / "ArtiMuse-10K_test.json"
    image_root = dataset_root / "image"
    rows: list[dict[str, Any]] = json.loads(input_path.read_text(encoding="utf-8"))

    metadata_rows: list[dict[str, Any]] = []
    generic_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_name = str(row["image"])
        image_path = (image_root / image_name).resolve()
        if not image_path.exists():
            raise FileNotFoundError(f"missing image: {image_path}")
        gt_1_10 = float(row["gt_score"])
        gt_0_100 = normalize_1_10_to_0_100(gt_1_10)
        benchmark_id = f"artimuse10k:{Path(image_name).stem}"
        metadata_row = {
            "benchmark_id": benchmark_id,
            "source": "ArtiMuse-10K",
            "benchmark_split": "test",
            "image_name": image_name,
            "image_path": str(image_path),
            "category": "unknown",
            "primary_mode": "test",
            "gt_score_1_10": gt_1_10,
            "gt_aesthetic_mean_0_100": gt_0_100,
            "row_index": index,
        }
        metadata_rows.append(metadata_row)
        generic_rows.append(
            {
                "benchmark_id": benchmark_id,
                "image_path": str(image_path),
                "gt_score_1_10": gt_1_10,
                "gt_score_0_100": gt_0_100,
                "category": "unknown",
                "primary_mode": "test",
            }
        )

    metadata_path = output_dir / "metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metadata_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metadata_rows)

    generic_path = output_dir / "generic_image_score.jsonl"
    with generic_path.open("w", encoding="utf-8") as fh:
        for row in generic_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    summary = {
        "dataset": "ArtiMuse-10K public test",
        "dataset_root": str(dataset_root),
        "rows": len(rows),
        "metadata": str(metadata_path),
        "generic_image_score": str(generic_path),
        "gt_scale": "1-10 normalized to 0-100 by (score - 1) / 9 * 100",
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
