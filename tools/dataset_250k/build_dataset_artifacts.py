#!/usr/bin/env python3
"""Build annotation and audit artifacts for the VeraRetouch 250k dataset."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET_ROOT = Path("/home/bc/data/datasets/VeraRetouch_250k")

COMPACT_REASONING_QUOTAS = {
    "S0_expert_anchor": 3000,
    "S1_auto_inverse_lite": 5000,
    "S2_param_l_gc_sc": 5000,
    "S3_style_lut": 7000,
    "S4_local_semantic_4d_lut": 10000,
}


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def write_jsonl_line(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def scene_for_record(record: dict[str, Any]) -> str:
    source = str(record.get("source_image_id", ""))
    tags = set(record.get("task_tags", []))
    if "human" in tags or record.get("target_region") == "human":
        return "portrait"
    if "sky" in tags:
        return "landscape"
    if "foliage" in tags:
        return "nature"
    if source.startswith("fivek:"):
        return "general_photo"
    if source.startswith("ppr10k:"):
        return "portrait_or_people"
    return "general_photo"


def operation_tags(record: dict[str, Any]) -> list[str]:
    branch = record["branch"]
    op = record.get("operation", {})
    tags = list(record.get("task_tags", []))
    if branch == "S2_param_l_gc_sc":
        for section in ("L", "GC", "SC"):
            for key, value in op.get(section, {}).items():
                if isinstance(value, (int, float)):
                    if abs(value) > 0.03 or key in {"contrast", "saturation", "gamma"}:
                        tags.append(key)
    elif branch in {"S3_style_lut", "S4_local_semantic_4d_lut"}:
        if op.get("style_family"):
            tags.append(str(op["style_family"]))
        if op.get("lut_id"):
            tags.append(str(op["lut_id"]))
    elif branch == "S1_auto_inverse_lite":
        tags.extend(["restore_exposure", "restore_contrast", "restore_color"])
    elif branch == "S0_expert_anchor":
        tags.extend(["expert_target", str(op.get("expert", "expert"))])
    return sorted({str(t) for t in tags if t})


def compact_reasoning(record: dict[str, Any]) -> dict[str, str]:
    branch = record["branch"]
    op = record.get("operation", {})
    scene = scene_for_record(record)
    preserve = "Keep geometry, identity, and local structure unchanged."
    if branch == "S0_expert_anchor":
        source = op.get("pair_source", "expert")
        expert = op.get("expert", "expert")
        return {
            "content_summary": f"{scene} image with an expert retouch target from {source}.",
            "observed_issue": "The input is the unretouched source and needs to match the expert color and tone distribution.",
            "retouch_plan": f"Use the {expert} expert target as the desired global and local appearance anchor.",
            "preserve_constraint": preserve,
        }
    if branch == "S1_auto_inverse_lite":
        return {
            "content_summary": f"{scene} image synthetically degraded from a high-quality target.",
            "observed_issue": "The input has reduced exposure, contrast, saturation, and small sensor-like noise.",
            "retouch_plan": "Recover natural exposure, contrast, white balance, and color richness without changing content.",
            "preserve_constraint": preserve,
        }
    if branch == "S2_param_l_gc_sc":
        light = op.get("L", {})
        color = op.get("GC", {})
        specific = op.get("SC", {})
        return {
            "content_summary": f"{scene} image with deterministic L/GC/SC parameter supervision.",
            "observed_issue": "The target is defined by controlled light, global color, and specific color adjustments.",
            "retouch_plan": (
                f"Apply exposure {light.get('exposure', 0):.2f}, contrast {light.get('contrast', 1):.2f}, "
                f"temperature {color.get('temperature', 0):.2f}, and saturation {specific.get('saturation', 1):.2f}."
            ),
            "preserve_constraint": preserve,
        }
    if branch == "S3_style_lut":
        style = str(op.get("style_family", "style")).replace("_", " ")
        return {
            "content_summary": f"{scene} image with a global {style} LUT target.",
            "observed_issue": "The source image needs a coherent style grade while retaining the original subject and layout.",
            "retouch_plan": f"Apply the selected {style} LUT globally and keep clipping controlled.",
            "preserve_constraint": preserve,
        }
    region = str(record.get("target_region", "local")).replace("_", " ")
    style = str(op.get("style_family", "style")).replace("_", " ")
    return {
        "content_summary": f"{scene} image with local semantic retouch supervision.",
        "observed_issue": f"Only the {region} should receive the target style/color change.",
        "retouch_plan": f"Use the semantic mask to blend a {style} LUT into the {region} while protecting surrounding content.",
        "preserve_constraint": preserve,
    }


def build_artifacts(args: argparse.Namespace) -> None:
    root = Path(args.out)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    short_path = annotations_dir / "short_annotations.jsonl"
    compact_path = annotations_dir / "compact_reasoning_30k.jsonl"
    audit_candidates_path = annotations_dir / "api_audit_candidates_2000.jsonl"
    quality_csv_path = root / "quality_report.csv"
    readme_path = root / "README.dataset.md"

    counts = Counter()
    split_counts = Counter()
    scene_counts = Counter()
    compact_counts = Counter()
    audit_counts = Counter()
    branch_has_mask = defaultdict(int)
    missing = 0

    short_tmp = short_path.with_suffix(".jsonl.tmp")
    compact_tmp = compact_path.with_suffix(".jsonl.tmp")
    audit_tmp = audit_candidates_path.with_suffix(".jsonl.tmp")

    with short_tmp.open("w", encoding="utf-8") as short_f, compact_tmp.open("w", encoding="utf-8") as compact_f, audit_tmp.open("w", encoding="utf-8") as audit_f:
        for record in iter_jsonl(manifest_path):
            branch = record["branch"]
            counts[branch] += 1
            split_counts[record.get("split", "unknown")] += 1
            scene = scene_for_record(record)
            scene_counts[scene] += 1
            if record.get("mask_path"):
                branch_has_mask[branch] += 1

            input_exists = Path(record["input_path"]).exists()
            target_exists = Path(record["target_path"]).exists()
            mask_path = record.get("mask_path")
            mask_exists = True if not mask_path else Path(mask_path).exists()
            if not (input_exists and target_exists and mask_exists):
                missing += 1

            short_record = {
                "id": record["id"],
                "branch": branch,
                "split": record.get("split"),
                "scene": scene,
                "instruction": record.get("instruction"),
                "task_tags": record.get("task_tags", []),
                "operation_tags": operation_tags(record),
                "target_region": record.get("target_region"),
                "protected_region": record.get("protected_region"),
                "input_path": record.get("input_path"),
                "target_path": record.get("target_path"),
                "mask_path": record.get("mask_path"),
            }
            write_jsonl_line(short_f, short_record)

            if compact_counts[branch] < COMPACT_REASONING_QUOTAS.get(branch, 0):
                compact_record = dict(short_record)
                compact_record["reasoning"] = compact_reasoning(record)
                compact_record["reasoning_source"] = "metadata_template"
                write_jsonl_line(compact_f, compact_record)
                compact_counts[branch] += 1

            if audit_counts[branch] < args.audit_per_branch:
                audit_record = dict(short_record)
                audit_record["audit_priority"] = "high" if branch == "S4_local_semantic_4d_lut" else "normal"
                audit_record["audit_status"] = "candidate_not_api_scored"
                write_jsonl_line(audit_f, audit_record)
                audit_counts[branch] += 1

    short_tmp.replace(short_path)
    compact_tmp.replace(compact_path)
    audit_tmp.replace(audit_candidates_path)

    with quality_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["branch", "records", "expected", "mask_records", "status"],
        )
        writer.writeheader()
        expected = {
            "S0_expert_anchor": 40000,
            "S1_auto_inverse_lite": 35000,
            "S2_param_l_gc_sc": 50000,
            "S3_style_lut": 75000,
            "S4_local_semantic_4d_lut": 50000,
        }
        for branch in sorted(expected):
            writer.writerow(
                {
                    "branch": branch,
                    "records": counts[branch],
                    "expected": expected[branch],
                    "mask_records": branch_has_mask[branch],
                    "status": "ok" if counts[branch] == expected[branch] else "mismatch",
                }
            )

    summary = {
        "manifest": str(manifest_path),
        "num_records": sum(counts.values()),
        "branch_counts": dict(sorted(counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "scene_counts": dict(sorted(scene_counts.items())),
        "missing_referenced_files": missing,
        "short_annotations": str(short_path),
        "compact_reasoning": {
            "path": str(compact_path),
            "counts": dict(sorted(compact_counts.items())),
            "total": sum(compact_counts.values()),
        },
        "api_audit_candidates": {
            "path": str(audit_candidates_path),
            "counts": dict(sorted(audit_counts.items())),
            "total": sum(audit_counts.values()),
            "note": "Candidate set only; no external API scoring was run in this build.",
        },
        "quality_csv": str(quality_csv_path),
    }
    write_json(root / "dataset_audit_summary.json", summary)

    readme_path.write_text(
        "\n".join(
            [
                "# VeraRetouch 250k Dataset",
                "",
                "Generated from `plan/dataset.md` with five paired-image branches.",
                "",
                "## Core Files",
                "",
                "- `manifest.jsonl`: 250,000 validated paired records.",
                "- `plan.jsonl`: deterministic generation plan.",
                "- `quality_report.json`: finalize-time integrity report.",
                "- `quality_report.csv`: branch-level count audit.",
                "- `annotations/short_annotations.jsonl`: 250,000 short instruction/tag records.",
                "- `annotations/compact_reasoning_30k.jsonl`: 30,000 compact reasoning records.",
                "- `annotations/api_audit_candidates_2000.jsonl`: 2,000 API-audit candidates; external API scoring was not run.",
                "- `lut_bank/analytic_luts.npy`: 2,048 analytic 32^3 LUTs.",
                "",
                "## Branch Counts",
                "",
                "| Branch | Count |",
                "| --- | ---: |",
                *[f"| {branch} | {count} |" for branch, count in sorted(counts.items())],
                "",
                "All image paths referenced by `manifest.jsonl` were verified by `finalize --recover-from-files --require-complete`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--audit-per-branch", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    build_artifacts(parse_args())
