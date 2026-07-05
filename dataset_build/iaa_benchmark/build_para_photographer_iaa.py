#!/usr/bin/env python3
"""Build a balanced Photographer-IAA benchmark subset from PARA.

The script reads PARA annotations directly from the password-protected zip,
selects a deterministic category x success/failure subset, and optionally
extracts the selected images into a benchmark directory.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import io
import json
import os
import random
import shutil
import statistics
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_PASSWORD = "E99A8FE69"
DEFAULT_DATASETS_ROOT = Path("~/data/datasets").expanduser()
DEFAULT_OUTPUT_DIR = DEFAULT_DATASETS_ROOT / "photographer_iaa_benchmark" / "para_v1"

ANNOTATION_FILES = {
    "train": "PARA/annotation/PARA-GiaaTrain.csv",
    "test": "PARA/annotation/PARA-GiaaTest.csv",
}

CATEGORY_MAP = {
    "portrait": ("portrait", "Portrait"),
    "scene": ("landscape_scene", "Landscape / Scene"),
    "animal": ("animal_pet", "Animal / Pet"),
    "food": ("food", "Food"),
    "indoor": ("indoor", "Indoor"),
    "building": ("building", "Building"),
    "stilllife": ("still_life", "Still Life"),
    "nightScene": ("night_scene", "Night Scene"),
    "plant": ("plant", "Plant"),
}

SCORE_COLUMNS = {
    "aesthetic": "aestheticScore_mean",
    "quality": "qualityScore_mean",
    "composition": "compositionScore_mean",
    "color": "colorScore_mean",
    "dof": "dofScore_mean",
    "light": "lightScore_mean",
    "content": "contentScore_mean",
    "content_preference": "contentPreference_mean",
    "willingness_to_share": "willingnessToShare_mean",
}

STD_COLUMNS = {
    "aesthetic": "aestheticScore_std",
    "quality": "qualityScore_std",
    "composition": "compositionScore_std",
    "color": "colorScore_std",
    "dof": "dofScore_std",
    "light": "lightScore_std",
    "content": "contentScore_std",
    "content_preference": "contentPreference_std",
    "willingness_to_share": "willingnessToShare_std",
}

CORE_ATTRIBUTES = ("quality", "composition", "color", "dof", "light", "content")

MODE_ORDER = (
    "success_high_all",
    "low_aesthetic",
    "composition_failure",
    "color_failure",
    "light_failure",
    "dof_failure",
    "quality_failure",
    "content_failure",
)

MODE_ATTRIBUTE = {
    "success_high_all": "aesthetic",
    "low_aesthetic": "aesthetic",
    "composition_failure": "composition",
    "color_failure": "color",
    "light_failure": "light",
    "dof_failure": "dof",
    "quality_failure": "quality",
    "content_failure": "content",
}

DIST_PREFIXES = (
    "aestheticScore_",
    "compositionScore_",
    "colorScore_",
    "dofScore_",
    "lightScore_",
    "contentScore_",
)


@dataclass(frozen=True)
class Quantiles:
    q25: float
    q75: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a balanced Photographer-IAA benchmark subset from PARA."
    )
    parser.add_argument(
        "--para-zip",
        type=Path,
        default=DEFAULT_DATASETS_ROOT / "PARA.zip",
        help="Path to PARA.zip. Defaults to ~/data/datasets/PARA.zip.",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("PARA_ZIP_PASSWORD", DEFAULT_PASSWORD),
        help="Password for PARA.zip. Defaults to PARA_ZIP_PASSWORD or the public PARA password.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Benchmark output directory.",
    )
    parser.add_argument(
        "--per-category-mode",
        type=int,
        default=25,
        help="Number of unique images to select for each category x mode cell.",
    )
    parser.add_argument("--seed", type=int, default=20260704, help="Deterministic sampling seed.")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Write metadata without extracting selected images.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark metadata and image files.",
    )
    return parser.parse_args()


def read_giaa_rows(para_zip: Path, password: str) -> list[dict[str, Any]]:
    if not para_zip.exists():
        raise FileNotFoundError(f"PARA zip not found: {para_zip}")

    rows: list[dict[str, Any]] = []
    pwd = password.encode("utf-8") if password else None
    with zipfile.ZipFile(para_zip) as zf:
        for split, member in ANNOTATION_FILES.items():
            with zf.open(member, pwd=pwd) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8-sig", newline="")
                for row in csv.DictReader(text):
                    if row.get("semantic") not in CATEGORY_MAP:
                        continue
                    row = dict(row)
                    row["source_split"] = split
                    row["source_zip_member"] = f"PARA/imgs/{row['sessionId']}/{row['imageName']}"
                    for col in tuple(SCORE_COLUMNS.values()) + tuple(STD_COLUMNS.values()):
                        if col in row and row[col] != "":
                            row[col] = float(row[col])
                    rows.append(row)
    return rows


def quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute quantile for an empty list")
    values = sorted(values)
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def build_quantiles(rows: list[dict[str, Any]]) -> dict[str, dict[str, Quantiles]]:
    by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_semantic[row["semantic"]].append(row)

    quantiles: dict[str, dict[str, Quantiles]] = {}
    for semantic, semantic_rows in by_semantic.items():
        quantiles[semantic] = {}
        for attr, col in SCORE_COLUMNS.items():
            values = [float(row[col]) for row in semantic_rows]
            quantiles[semantic][attr] = Quantiles(q25=quantile(values, 0.25), q75=quantile(values, 0.75))
    return quantiles


def build_percentile_tables(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
    values: dict[str, dict[str, list[float]]] = defaultdict(dict)
    by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_semantic[row["semantic"]].append(row)
    for semantic, semantic_rows in by_semantic.items():
        for attr, col in SCORE_COLUMNS.items():
            values[semantic][attr] = sorted(float(row[col]) for row in semantic_rows)
    return values


def percentile_rank(sorted_values: list[float], value: float) -> float:
    if not sorted_values:
        return 0.0
    return bisect.bisect_right(sorted_values, value) / len(sorted_values)


def annotate_rows(
    rows: list[dict[str, Any]],
    quantiles: dict[str, dict[str, Quantiles]],
    percentile_tables: dict[str, dict[str, list[float]]],
) -> None:
    for row in rows:
        semantic = row["semantic"]
        row["flag_success_high_all"] = (
            row[SCORE_COLUMNS["aesthetic"]] >= quantiles[semantic]["aesthetic"].q75
            and all(row[SCORE_COLUMNS[attr]] >= quantiles[semantic][attr].q75 for attr in CORE_ATTRIBUTES)
        )
        row["flag_low_aesthetic"] = (
            row[SCORE_COLUMNS["aesthetic"]] <= quantiles[semantic]["aesthetic"].q25
        )
        for attr in CORE_ATTRIBUTES:
            row[f"flag_{attr}_failure"] = row[SCORE_COLUMNS[attr]] <= quantiles[semantic][attr].q25
        for attr, col in SCORE_COLUMNS.items():
            row[f"{attr}_percentile_in_category"] = percentile_rank(
                percentile_tables[semantic][attr], row[col]
            )


def candidate_filter(row: dict[str, Any], mode: str) -> bool:
    if mode == "success_high_all":
        return bool(row["flag_success_high_all"])
    if mode == "low_aesthetic":
        return bool(row["flag_low_aesthetic"])
    attr = MODE_ATTRIBUTE[mode]
    return bool(row[f"flag_{attr}_failure"])


def mode_sort_key(row: dict[str, Any], mode: str, tie_breaker: float) -> tuple[float, float, str]:
    attr = MODE_ATTRIBUTE[mode]
    value = float(row[SCORE_COLUMNS[attr]])
    if mode == "success_high_all":
        return (-float(row[SCORE_COLUMNS["aesthetic"]]), tie_breaker, row["imageName"])
    return (value, tie_breaker, row["imageName"])


def select_rows(
    rows: list[dict[str, Any]], per_category_mode: int, seed: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    by_semantic: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_semantic[row["semantic"]].append(row)

    selected: list[dict[str, Any]] = []
    used_names: set[str] = set()
    cell_counts: dict[str, dict[str, int]] = defaultdict(dict)
    shortfalls: list[dict[str, Any]] = []

    for semantic in CATEGORY_MAP:
        semantic_rows = by_semantic[semantic]
        for mode in MODE_ORDER:
            candidates = [row for row in semantic_rows if row["imageName"] not in used_names and candidate_filter(row, mode)]
            tie_breakers = {id(row): rng.random() for row in candidates}
            candidates.sort(key=lambda row, m=mode: mode_sort_key(row, m, tie_breakers[id(row)]))
            picked = candidates[:per_category_mode]
            if len(picked) < per_category_mode:
                shortfalls.append(
                    {
                        "semantic": semantic,
                        "mode": mode,
                        "requested": per_category_mode,
                        "selected": len(picked),
                    }
                )
            for row in picked:
                used_names.add(row["imageName"])
                row = dict(row)
                row["primary_mode"] = mode
                row["mode_attribute"] = MODE_ATTRIBUTE[mode]
                selected.append(row)
            cell_counts[semantic][mode] = len(picked)

    selected.sort(key=lambda row: (row["semantic"], row["primary_mode"], row["imageName"]))
    stats = {"cell_counts": cell_counts, "shortfalls": shortfalls}
    return selected, stats


def normalized_0_100(value_1_5: float) -> float:
    return max(0.0, min(100.0, (float(value_1_5) - 1.0) / 4.0 * 100.0))


def is_distribution_column(key: str, prefix: str) -> bool:
    if not key.startswith(prefix):
        return False
    suffix = key.removeprefix(prefix)
    return suffix not in {"mean", "std"}


def count_distribution(row: dict[str, Any], prefix: str) -> int:
    total = 0
    for key, value in row.items():
        if is_distribution_column(key, prefix):
            try:
                total += int(float(value))
            except ValueError:
                pass
    return total


def output_row(row: dict[str, Any], output_dir: Path, extract_images: bool) -> dict[str, Any]:
    semantic = row["semantic"]
    category_slug, category_display = CATEGORY_MAP[semantic]
    image_relpath = Path("images") / category_slug / row["imageName"]
    mode_attr = row["mode_attribute"]
    out: dict[str, Any] = {
        "benchmark_id": f"para:{Path(row['imageName']).stem}",
        "source": "PARA",
        "source_split": row["source_split"],
        "image_name": row["imageName"],
        "session_id": row["sessionId"],
        "semantic": semantic,
        "category": category_slug,
        "category_display": category_display,
        "image_path": str(image_relpath) if extract_images else "",
        "source_zip_member": row["source_zip_member"],
        "benchmark_split": "test",
        "primary_mode": row["primary_mode"],
        "mode_attribute": mode_attr,
        "mode_score_1_5": row[SCORE_COLUMNS[mode_attr]],
        "mode_percentile_in_category": row[f"{mode_attr}_percentile_in_category"],
        "rater_count": count_distribution(row, "aestheticScore_"),
    }

    for attr, col in SCORE_COLUMNS.items():
        out[f"gt_{attr}_mean_1_5"] = row[col]
        out[f"gt_{attr}_mean_0_100"] = normalized_0_100(row[col])
        std_col = STD_COLUMNS.get(attr)
        if std_col:
            out[f"gt_{attr}_std"] = row[std_col]
        out[f"{attr}_percentile_in_category"] = row[f"{attr}_percentile_in_category"]

    out["flag_success_high_all"] = int(row["flag_success_high_all"])
    out["flag_low_aesthetic"] = int(row["flag_low_aesthetic"])
    for attr in CORE_ATTRIBUTES:
        out[f"flag_{attr}_failure"] = int(row[f"flag_{attr}_failure"])

    for prefix in DIST_PREFIXES:
        for key, value in row.items():
            if is_distribution_column(key, prefix):
                out[f"dist_{key}"] = value

    out["absolute_image_path"] = str(output_dir / image_relpath) if extract_images else ""
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows selected; refusing to write empty CSV")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def extract_images(
    para_zip: Path,
    password: str,
    rows: list[dict[str, Any]],
    output_dir: Path,
    overwrite: bool,
) -> None:
    pwd = password.encode("utf-8") if password else None
    with zipfile.ZipFile(para_zip) as zf:
        for row in rows:
            category_slug = row["category"]
            target = output_dir / "images" / category_slug / row["image_name"]
            if target.exists() and not overwrite:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(row["source_zip_member"], pwd=pwd) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def summarize(
    rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    quantiles: dict[str, dict[str, Quantiles]],
    sampling_stats: dict[str, Any],
    args: argparse.Namespace,
    extract_images: bool,
) -> dict[str, Any]:
    by_category: dict[str, int] = defaultdict(int)
    by_mode: dict[str, int] = defaultdict(int)
    by_cell: dict[str, dict[str, int]] = defaultdict(dict)
    raters = []
    aesthetic_scores = []
    for row in rows:
        by_category[row["category"]] += 1
        by_mode[row["primary_mode"]] += 1
        by_cell[row["category"]][row["primary_mode"]] = by_cell[row["category"]].get(row["primary_mode"], 0) + 1
        raters.append(int(row["rater_count"]))
        aesthetic_scores.append(float(row["gt_aesthetic_mean_1_5"]))

    quantile_summary: dict[str, dict[str, dict[str, float]]] = {}
    for semantic, attrs in quantiles.items():
        if semantic not in CATEGORY_MAP:
            continue
        category_slug = CATEGORY_MAP[semantic][0]
        quantile_summary[category_slug] = {
            attr: {"p25": q.q25, "p75": q.q75} for attr, q in attrs.items()
        }

    return {
        "benchmark": "Photographer-IAA PARA v1",
        "source": "PARA",
        "para_zip": str(args.para_zip),
        "output_dir": str(args.output_dir),
        "selection_seed": args.seed,
        "per_category_mode": args.per_category_mode,
        "images_extracted": extract_images,
        "total_source_rows_used": len(all_rows),
        "selected_rows": len(rows),
        "target_categories": len(CATEGORY_MAP),
        "modes": list(MODE_ORDER),
        "core_attributes": list(CORE_ATTRIBUTES),
        "category_counts": dict(sorted(by_category.items())),
        "mode_counts": dict(sorted(by_mode.items())),
        "cell_counts": {k: dict(sorted(v.items())) for k, v in sorted(by_cell.items())},
        "sampling_shortfalls": sampling_stats["shortfalls"],
        "rater_count": {
            "min": min(raters) if raters else None,
            "median": statistics.median(raters) if raters else None,
            "max": max(raters) if raters else None,
        },
        "gt_aesthetic_mean_1_5": {
            "min": min(aesthetic_scores) if aesthetic_scores else None,
            "mean": statistics.mean(aesthetic_scores) if aesthetic_scores else None,
            "max": max(aesthetic_scores) if aesthetic_scores else None,
        },
        "category_quantiles": quantile_summary,
    }


def write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Photographer-IAA PARA v1",
        "",
        "Balanced PARA subset for image aesthetics assessment of photographer-facing models.",
        "",
        "## Files",
        "",
        "- `metadata.csv`: main benchmark manifest with GT scores, mode labels, and relative image paths.",
        "- `metadata.jsonl`: same rows as JSONL.",
        "- `summary.json`: selection counts, quantile thresholds, and validation summary.",
        "- `images/`: selected PARA images when the builder is run without `--metadata-only`.",
        "",
        "## Selection",
        "",
        f"- Source rows considered: {summary['total_source_rows_used']}",
        f"- Selected rows: {summary['selected_rows']}",
        f"- Categories: {', '.join(summary['category_counts'].keys())}",
        f"- Modes: {', '.join(summary['modes'])}",
        "- Success mode: category-local P75 aesthetic and P75 for all core attributes.",
        "- Failure modes: category-local P25 for the named aesthetic or attribute dimension.",
        "",
        "## Scores",
        "",
        "Human GT is preserved in the PARA 1-5 scale and normalized to 0-100 for model comparability.",
        "Core attributes are quality, composition, color, depth-of-field, light, and content.",
        "",
    ]
    if summary["sampling_shortfalls"]:
        lines.extend(
            [
                "## Sampling Shortfalls",
                "",
                "Some category x mode cells did not reach the requested size. See `summary.json`.",
                "",
            ]
        )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.para_zip = args.para_zip.expanduser()
    args.output_dir = args.output_dir.expanduser()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_giaa_rows(args.para_zip, args.password)
    quantiles = build_quantiles(rows)
    percentile_tables = build_percentile_tables(rows)
    annotate_rows(rows, quantiles, percentile_tables)
    selected_raw, sampling_stats = select_rows(rows, args.per_category_mode, args.seed)
    extract = not args.metadata_only
    output_rows = [output_row(row, args.output_dir, extract) for row in selected_raw]

    metadata_csv = args.output_dir / "metadata.csv"
    metadata_jsonl = args.output_dir / "metadata.jsonl"
    if not args.overwrite:
        for path in (metadata_csv, metadata_jsonl, args.output_dir / "summary.json"):
            if path.exists():
                raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    write_csv(metadata_csv, output_rows)
    write_jsonl(metadata_jsonl, output_rows)

    if extract:
        extract_images(args.para_zip, args.password, output_rows, args.output_dir, args.overwrite)

    summary = summarize(output_rows, rows, quantiles, sampling_stats, args, extract)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2, sort_keys=True)
        fh.write("\n")
    write_readme(args.output_dir, summary)

    if summary["sampling_shortfalls"]:
        print(json.dumps({"status": "partial", **summary}, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps({"status": "ok", **summary}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
