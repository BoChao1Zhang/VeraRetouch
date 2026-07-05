#!/usr/bin/env python3
"""Export Photographer-IAA metadata into common model-eval input formats."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


PROMPT_SCORE_0_100 = (
    "Rate the overall photographic aesthetic quality of this image on a 0 to 100 scale. "
    "Answer with only one number."
)

PROMPT_FIVE_GEAR = (
    "Rate this image from an aesthetic perspective. "
    "The aesthetic quality is"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Photographer-IAA model inputs.")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy images instead of symlinking them.",
    )
    return parser.parse_args()


def read_metadata(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    if copy_images:
        shutil.copy2(src, dst)
    else:
        os.symlink(src, dst)


def main() -> None:
    args = parse_args()
    benchmark_dir = args.benchmark_dir.expanduser()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_metadata(benchmark_dir / "metadata.csv")
    artimuse_dir = output_dir / "artimuse" / "PhotographerIAA"
    vllm_path = output_dir / "vllm_requests.jsonl"
    generic_path = output_dir / "generic_image_score.jsonl"

    artimuse_images = artimuse_dir / "images"
    artimuse_test = []
    vllm_rows = []
    generic_rows = []

    for row in rows:
        image_src = (benchmark_dir / row["image_path"]).resolve()
        flat_name = f"{row['category']}__{row['image_name']}"
        image_dst = artimuse_images / flat_name
        link_or_copy(image_src, image_dst, args.copy_images)
        artimuse_test.append(
            {
                "benchmark_id": row["benchmark_id"],
                "image": flat_name,
                "gt_score": float(row["gt_aesthetic_mean_0_100"]),
                "gt_score_1_5": float(row["gt_aesthetic_mean_1_5"]),
                "category": row["category"],
                "primary_mode": row["primary_mode"],
            }
        )
        vllm_rows.append(
            {
                "benchmark_id": row["benchmark_id"],
                "image_path": str(image_src),
                "prompt": PROMPT_SCORE_0_100,
                "five_gear_prompt_prefix": PROMPT_FIVE_GEAR,
                "gt_score_0_100": float(row["gt_aesthetic_mean_0_100"]),
                "category": row["category"],
                "primary_mode": row["primary_mode"],
            }
        )
        generic_rows.append(
            {
                "benchmark_id": row["benchmark_id"],
                "image_path": str(image_src),
                "gt_score_0_100": float(row["gt_aesthetic_mean_0_100"]),
                "gt_score_1_5": float(row["gt_aesthetic_mean_1_5"]),
                "category": row["category"],
                "primary_mode": row["primary_mode"],
            }
        )

    with (artimuse_dir / "test.json").open("w", encoding="utf-8") as fh:
        json.dump(artimuse_test, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    with vllm_path.open("w", encoding="utf-8") as fh:
        for row in vllm_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with generic_path.open("w", encoding="utf-8") as fh:
        for row in generic_rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "artimuse_dataset": str(artimuse_dir),
                "vllm_requests": str(vllm_path),
                "generic_image_score": str(generic_path),
                "copy_images": args.copy_images,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
