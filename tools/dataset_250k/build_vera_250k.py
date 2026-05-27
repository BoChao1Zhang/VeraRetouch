#!/usr/bin/env python3
"""Build the VeraRetouch 250k paired-image dataset.

The pipeline is intentionally restartable:

1. plan      - scan local sources and write a deterministic 250k plan
2. luts      - build an analytic LUT bank used by S3/S4
3. generate  - generate one shard of image pairs on one CUDA device
4. finalize  - merge shard manifests and validate output counts
5. status    - print current progress
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFilter, ImageOps
from tqdm import tqdm

try:
    cv2.setLogLevel(0)
except AttributeError:
    pass


PPR_ROOT = Path("/home/bc/retouching/monetGPT/data/ppr10k")
FIVEK_ROOT = Path("/home/bc/retouching/monetGPT/data/fivek_mmart_like")
MMART_PPR_ROOT = Path("/home/bc/datasets/MMArt-PPR10k/global")
DEFAULT_OUT = Path("/home/bc/data/datasets/VeraRetouch_250k")

IMAGE_SIZE = 512
JPEG_QUALITY = 94
SEED = 20260527

BRANCH_COUNTS = {
    "S0_expert_anchor": 40_000,
    "S1_auto_inverse_lite": 35_000,
    "S2_param_l_gc_sc": 50_000,
    "S3_style_lut": 75_000,
    "S4_local_semantic_4d_lut": 50_000,
}

STYLE_FAMILIES = [
    "warm",
    "cool",
    "vivid",
    "faded",
    "cinematic",
    "film",
    "bright_airy",
    "moody",
    "high_contrast",
    "low_contrast",
    "portrait_clean",
    "natural",
]

INSTRUCTION_TEMPLATES = {
    "S0_expert_anchor": "Match the expert retouch while preserving the original scene structure.",
    "S1_auto_inverse_lite": "Restore exposure, contrast, and natural color from the degraded input.",
    "S2_param_l_gc_sc": "Apply the requested global light, color, and saturation adjustment.",
    "S3_style_lut": "Apply a {style_family} color grade while keeping the content unchanged.",
    "S4_local_semantic_4d_lut": "Enhance the {target_region} region while preserving the rest of the image.",
}


def stable_hash(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def split_for_key(key: str) -> str:
    value = stable_hash(key) % 10_000
    if value < 9600:
        return "train"
    if value < 9800:
        return "val"
    return "test"


def json_dump_line(fp: Any, record: dict[str, Any]) -> None:
    fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def cycle_take(items: list[dict[str, Any]], count: int, rng: random.Random) -> list[dict[str, Any]]:
    if not items:
        raise RuntimeError("Cannot sample from an empty source pool")
    order = list(items)
    rng.shuffle(order)
    out: list[dict[str, Any]] = []
    while len(out) < count:
        out.extend(order)
        rng.shuffle(order)
    return out[:count]


def scan_ppr() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    id_map = PPR_ROOT / "manifests" / "id_map.csv"
    if not id_map.exists():
        raise FileNotFoundError(id_map)

    pairs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    with id_map.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            new_id = row["new_id"]
            source = PPR_ROOT / "source" / f"{new_id}.png"
            if not source.exists():
                continue
            mask = PPR_ROOT / "masks" / "360p" / "masks_360p" / f"{row['orig_base']}.png"
            mask_path = str(mask) if mask.exists() else None
            source_record = {
                "source_dataset": "ppr10k",
                "source_image_id": f"ppr10k:{new_id}",
                "split_key": f"ppr10k:{row['orig_base']}",
                "image_path": str(source),
                "mask_path": mask_path,
            }
            sources.append(source_record)
            for expert in ("a", "b", "c"):
                target = PPR_ROOT / f"target_{expert}" / f"{new_id}.png"
                if target.exists():
                    pairs.append({
                        **source_record,
                        "target_path": str(target),
                        "expert": expert,
                        "pair_source": "ppr10k",
                    })
    return pairs, sources


def scan_fivek() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for split_dir in (FIVEK_ROOT / "train_global", FIVEK_ROOT / "test_global"):
        if not split_dir.exists():
            continue
        for sample_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            before = sample_dir / "before.jpg"
            processed = sample_dir / "processed.jpg"
            if not before.exists():
                continue
            source_record = {
                "source_dataset": "fivek_mmart_like",
                "source_image_id": f"fivek:{sample_dir.name}",
                "split_key": f"fivek:{sample_dir.name.rsplit('_', 1)[0]}",
                "image_path": str(before),
                "mask_path": None,
            }
            sources.append(source_record)
            if processed.exists():
                expert = sample_dir.name.rsplit("_", 1)[-1]
                pairs.append({
                    **source_record,
                    "target_path": str(processed),
                    "expert": expert,
                    "pair_source": "fivek_mmart_like",
                })
    return pairs, sources


def scan_mmart_ppr() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    if not MMART_PPR_ROOT.exists():
        return pairs, sources
    for sample_dir in sorted(p for p in MMART_PPR_ROOT.iterdir() if p.is_dir()):
        before = sample_dir / "before.jpg"
        processed = sample_dir / "processed.jpg"
        if not before.exists():
            continue
        source_record = {
            "source_dataset": "mmart_ppr10k",
            "source_image_id": f"mmart_ppr:{sample_dir.name}",
            "split_key": f"mmart_ppr:{sample_dir.name}",
            "image_path": str(before),
            "mask_path": None,
        }
        sources.append(source_record)
        if processed.exists():
            pairs.append({
                **source_record,
                "target_path": str(processed),
                "expert": "mmart",
                "pair_source": "mmart_ppr10k",
            })
    return pairs, sources


def plan_record(
    index: int,
    branch: str,
    input_source_path: str,
    source_image_id: str,
    split_key: str,
    operation: dict[str, Any],
    target_source_path: str | None = None,
    mask_source_path: str | None = None,
    target_region: str = "global",
    protected_region: str = "none",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    sample_id = f"{branch[:2].lower()}_{index:06d}"
    if branch == "S3_style_lut":
        instruction = INSTRUCTION_TEMPLATES[branch].format(style_family=operation["style_family"].replace("_", " "))
    elif branch == "S4_local_semantic_4d_lut":
        instruction = INSTRUCTION_TEMPLATES[branch].format(target_region=target_region.replace("_", " "))
    else:
        instruction = INSTRUCTION_TEMPLATES[branch]
    split = split_for_key(split_key)
    return {
        "id": sample_id,
        "index": index,
        "branch": branch,
        "split": split,
        "source_image_id": source_image_id,
        "split_key": split_key,
        "input_source_path": input_source_path,
        "target_source_path": target_source_path,
        "mask_source_path": mask_source_path,
        "output_input_path": f"images/input/{branch}/{sample_id}.jpg",
        "output_target_path": f"images/target/{branch}/{sample_id}.jpg",
        "output_mask_path": f"masks/{branch}/{sample_id}.png" if branch == "S4_local_semantic_4d_lut" else None,
        "instruction": instruction,
        "task_tags": tags or [],
        "target_region": target_region,
        "protected_region": protected_region,
        "operation": operation,
        "annotation": {
            "scene": "unknown",
            "reasoning_tier": "short_template",
            "compact_reasoning": None,
            "gold_audit": None,
        },
    }


def sample_degrade_params(rng: random.Random) -> dict[str, Any]:
    return {
        "kind": "inverse_degrade",
        "exposure": rng.uniform(-0.55, -0.10),
        "contrast": rng.uniform(0.72, 0.95),
        "saturation": rng.uniform(0.65, 0.92),
        "temperature": rng.uniform(-0.08, 0.06),
        "tint": rng.uniform(-0.05, 0.05),
        "gamma": rng.uniform(0.92, 1.14),
        "noise_std": rng.uniform(0.0, 0.018),
    }


def sample_param_retouch(rng: random.Random) -> dict[str, Any]:
    return {
        "kind": "param_l_gc_sc",
        "L": {
            "exposure": rng.uniform(-0.25, 0.45),
            "contrast": rng.uniform(0.82, 1.28),
            "gamma": rng.uniform(0.86, 1.12),
            "shadow_lift": rng.uniform(-0.05, 0.14),
            "highlight_rolloff": rng.uniform(0.0, 0.12),
        },
        "GC": {
            "temperature": rng.uniform(-0.11, 0.13),
            "tint": rng.uniform(-0.07, 0.07),
            "vibrance": rng.uniform(-0.10, 0.22),
        },
        "SC": {
            "saturation": rng.uniform(0.78, 1.36),
            "red": rng.uniform(-0.05, 0.06),
            "green": rng.uniform(-0.05, 0.06),
            "blue": rng.uniform(-0.05, 0.06),
        },
    }


def build_plan(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    plan_path = out_dir / "plan.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    ppr_pairs, ppr_sources = scan_ppr()
    fivek_pairs, fivek_sources = scan_fivek()
    mmart_pairs, mmart_sources = scan_mmart_ppr()

    expert_pairs = ppr_pairs + fivek_pairs + mmart_pairs
    target_pool = [
        {**p, "image_path": p["target_path"]}
        for p in expert_pairs
        if Path(p["target_path"]).exists()
    ]
    source_pool = ppr_sources + fivek_sources + mmart_sources
    ppr_mask_pool = [p for p in ppr_sources if p.get("mask_path")]

    if len(expert_pairs) < BRANCH_COUNTS["S0_expert_anchor"]:
        raise RuntimeError(f"Only {len(expert_pairs)} expert pairs, need 40000")
    if not source_pool or not target_pool or not ppr_mask_pool:
        raise RuntimeError("Source scan produced an empty required pool")

    records: list[dict[str, Any]] = []

    rng.shuffle(expert_pairs)
    for pair in expert_pairs[:BRANCH_COUNTS["S0_expert_anchor"]]:
        idx = len(records)
        records.append(plan_record(
            idx,
            "S0_expert_anchor",
            input_source_path=pair["image_path"],
            target_source_path=pair["target_path"],
            source_image_id=pair["source_image_id"],
            split_key=pair["split_key"],
            mask_source_path=pair.get("mask_path"),
            operation={
                "kind": "expert_pair",
                "expert": pair.get("expert"),
                "pair_source": pair.get("pair_source"),
            },
            tags=["expert", "paired", pair.get("pair_source", "unknown")],
        ))

    for src in cycle_take(target_pool, BRANCH_COUNTS["S1_auto_inverse_lite"], rng):
        idx = len(records)
        records.append(plan_record(
            idx,
            "S1_auto_inverse_lite",
            input_source_path=src["image_path"],
            target_source_path=src["image_path"],
            source_image_id=src["source_image_id"],
            split_key=src["split_key"],
            mask_source_path=src.get("mask_path"),
            operation=sample_degrade_params(rng),
            tags=["auto", "inverse_degradation", "teacher_lite"],
        ))

    for src in cycle_take(source_pool, BRANCH_COUNTS["S2_param_l_gc_sc"], rng):
        idx = len(records)
        records.append(plan_record(
            idx,
            "S2_param_l_gc_sc",
            input_source_path=src["image_path"],
            source_image_id=src["source_image_id"],
            split_key=src["split_key"],
            mask_source_path=src.get("mask_path"),
            operation=sample_param_retouch(rng),
            tags=["param", "L", "GC", "SC"],
        ))

    for src in cycle_take(source_pool, BRANCH_COUNTS["S3_style_lut"], rng):
        idx = len(records)
        lut_id = rng.randrange(args.num_luts)
        style_family = STYLE_FAMILIES[lut_id % len(STYLE_FAMILIES)]
        records.append(plan_record(
            idx,
            "S3_style_lut",
            input_source_path=src["image_path"],
            source_image_id=src["source_image_id"],
            split_key=src["split_key"],
            mask_source_path=src.get("mask_path"),
            operation={
                "kind": "style_lut",
                "lut_id": f"analytic_{lut_id:04d}",
                "lut_index": lut_id,
                "lut_source": "analytic",
                "style_family": style_family,
            },
            tags=["style", "lut", style_family],
        ))

    s4_ppr_count = min(30_000, BRANCH_COUNTS["S4_local_semantic_4d_lut"], len(ppr_mask_pool) * 4)
    for src in cycle_take(ppr_mask_pool, s4_ppr_count, rng):
        idx = len(records)
        lut_id = rng.randrange(args.num_luts)
        style_family = STYLE_FAMILIES[lut_id % len(STYLE_FAMILIES)]
        records.append(plan_record(
            idx,
            "S4_local_semantic_4d_lut",
            input_source_path=src["image_path"],
            source_image_id=src["source_image_id"],
            split_key=src["split_key"],
            mask_source_path=src.get("mask_path"),
            target_region="human",
            protected_region="background",
            operation={
                "kind": "local_mask_lut",
                "mask_kind": "ppr_human",
                "lut_id": f"analytic_{lut_id:04d}",
                "lut_index": lut_id,
                "lut_source": "analytic",
                "style_family": style_family,
                "blend": rng.uniform(0.55, 0.90),
            },
            tags=["local", "semantic", "mask", "human", style_family],
        ))

    remaining_s4 = BRANCH_COUNTS["S4_local_semantic_4d_lut"] - s4_ppr_count
    heuristic_regions = ["sky", "foliage", "background"]
    for src in cycle_take(source_pool, remaining_s4, rng):
        idx = len(records)
        lut_id = rng.randrange(args.num_luts)
        style_family = STYLE_FAMILIES[lut_id % len(STYLE_FAMILIES)]
        region = heuristic_regions[idx % len(heuristic_regions)]
        records.append(plan_record(
            idx,
            "S4_local_semantic_4d_lut",
            input_source_path=src["image_path"],
            source_image_id=src["source_image_id"],
            split_key=src["split_key"],
            mask_source_path=None,
            target_region=region,
            protected_region="subject" if region != "background" else "foreground",
            operation={
                "kind": "local_mask_lut",
                "mask_kind": f"heuristic_{region}",
                "lut_id": f"analytic_{lut_id:04d}",
                "lut_index": lut_id,
                "lut_source": "analytic",
                "style_family": style_family,
                "blend": rng.uniform(0.45, 0.82),
            },
            tags=["local", "semantic", "heuristic_mask", region, style_family],
        ))

    expected = sum(BRANCH_COUNTS.values())
    if len(records) != expected:
        raise RuntimeError(f"Planned {len(records)} records, expected {expected}")

    tmp = plan_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            json_dump_line(f, record)
    tmp.replace(plan_path)

    counts = count_by(records, "branch")
    write_json(out_dir / "plan_stats.json", {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": args.seed,
        "num_records": len(records),
        "branch_counts": counts,
        "source_counts": {
            "ppr_pairs": len(ppr_pairs),
            "fivek_pairs": len(fivek_pairs),
            "mmart_ppr_pairs": len(mmart_pairs),
            "source_pool": len(source_pool),
            "target_pool": len(target_pool),
            "ppr_mask_pool": len(ppr_mask_pool),
        },
    })
    print(f"Wrote {plan_path} with {len(records)} records")


def count_by(records: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def apply_basic_np(x: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    y = x.astype(np.float32)
    exposure = params.get("exposure", 0.0)
    contrast = params.get("contrast", 1.0)
    saturation = params.get("saturation", 1.0)
    temperature = params.get("temperature", 0.0)
    tint = params.get("tint", 0.0)
    gamma = params.get("gamma", 1.0)

    y = y * (2.0 ** exposure)
    y = (y - 0.5) * contrast + 0.5
    y[..., 0] *= 1.0 + temperature
    y[..., 2] *= 1.0 - temperature
    y[..., 1] *= 1.0 + tint
    lum = y[..., 0:1] * 0.299 + y[..., 1:2] * 0.587 + y[..., 2:3] * 0.114
    y = lum + (y - lum) * saturation
    y = np.clip(y, 0.0, 1.0)
    y = np.power(y, gamma)
    return np.clip(y, 0.0, 1.0)


def lut_params_for(style_family: str, rng: random.Random) -> dict[str, float]:
    base = {
        "exposure": rng.uniform(-0.08, 0.12),
        "contrast": rng.uniform(0.90, 1.15),
        "saturation": rng.uniform(0.88, 1.18),
        "temperature": rng.uniform(-0.04, 0.04),
        "tint": rng.uniform(-0.025, 0.025),
        "gamma": rng.uniform(0.92, 1.08),
    }
    if style_family == "warm":
        base.update({"temperature": rng.uniform(0.05, 0.15), "saturation": rng.uniform(0.95, 1.15)})
    elif style_family == "cool":
        base.update({"temperature": rng.uniform(-0.15, -0.05), "contrast": rng.uniform(0.95, 1.16)})
    elif style_family == "vivid":
        base.update({"contrast": rng.uniform(1.08, 1.30), "saturation": rng.uniform(1.16, 1.45)})
    elif style_family == "faded":
        base.update({"contrast": rng.uniform(0.72, 0.92), "saturation": rng.uniform(0.62, 0.88), "gamma": rng.uniform(0.88, 1.03)})
    elif style_family == "cinematic":
        base.update({"contrast": rng.uniform(1.10, 1.34), "temperature": rng.uniform(-0.06, 0.08), "saturation": rng.uniform(0.82, 1.05)})
    elif style_family == "film":
        base.update({"contrast": rng.uniform(0.86, 1.10), "saturation": rng.uniform(0.76, 1.02), "temperature": rng.uniform(0.02, 0.12)})
    elif style_family == "bright_airy":
        base.update({"exposure": rng.uniform(0.12, 0.32), "contrast": rng.uniform(0.84, 1.02), "saturation": rng.uniform(0.86, 1.04)})
    elif style_family == "moody":
        base.update({"exposure": rng.uniform(-0.24, -0.05), "contrast": rng.uniform(1.08, 1.32), "saturation": rng.uniform(0.72, 0.96)})
    elif style_family == "high_contrast":
        base.update({"contrast": rng.uniform(1.22, 1.48), "saturation": rng.uniform(0.92, 1.18)})
    elif style_family == "low_contrast":
        base.update({"contrast": rng.uniform(0.68, 0.88), "saturation": rng.uniform(0.82, 1.02)})
    elif style_family == "portrait_clean":
        base.update({"exposure": rng.uniform(0.04, 0.20), "contrast": rng.uniform(0.92, 1.10), "saturation": rng.uniform(0.88, 1.06), "temperature": rng.uniform(0.01, 0.08)})
    return base


def build_luts(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    lut_dir = out_dir / "lut_bank"
    lut_dir.mkdir(parents=True, exist_ok=True)
    luts_path = lut_dir / "analytic_luts.npy"
    manifest_path = lut_dir / "analytic_luts.jsonl"
    if luts_path.exists() and manifest_path.exists() and not args.force:
        print(f"LUT bank exists: {luts_path}")
        return

    rng = random.Random(args.seed)
    axis = np.linspace(0.0, 1.0, args.lut_size, dtype=np.float32)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack([rr, gg, bb], axis=-1)
    luts = np.empty((args.num_luts, args.lut_size, args.lut_size, args.lut_size, 3), dtype=np.float16)
    tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
    with tmp_manifest.open("w", encoding="utf-8") as f:
        for i in tqdm(range(args.num_luts), desc="analytic LUTs"):
            family = STYLE_FAMILIES[i % len(STYLE_FAMILIES)]
            params = lut_params_for(family, rng)
            lut = apply_basic_np(grid, params)
            luts[i] = lut.astype(np.float16)
            json_dump_line(f, {
                "lut_id": f"analytic_{i:04d}",
                "lut_index": i,
                "lut_source": "analytic",
                "style_family": family,
                "params": params,
            })
    np.save(luts_path, luts)
    tmp_manifest.replace(manifest_path)
    write_json(lut_dir / "stats.json", {
        "num_luts": args.num_luts,
        "lut_size": args.lut_size,
        "dtype": "float16",
        "path": str(luts_path),
    })
    print(f"Wrote {luts_path}")


@dataclass
class LoadedBatch:
    records: list[dict[str, Any]]
    images: torch.Tensor
    masks: torch.Tensor | None


def resize_crop_pil(img: Image.Image, size: int = IMAGE_SIZE, resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    scale = size / min(w, h)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    img = img.resize((new_w, new_h), resample)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def resize_crop_cv2(img: np.ndarray, size: int = IMAGE_SIZE, interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    h, w = img.shape[:2]
    scale = size / min(w, h)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    if scale > 1.0:
        interpolation = cv2.INTER_CUBIC
    img = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return img[top:top + size, left:left + size]


def resize_crop_mask(mask: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    w, h = mask.size
    scale = size / min(w, h)
    new_w = max(size, int(round(w * scale)))
    new_h = max(size, int(round(h * scale)))
    mask = mask.resize((new_w, new_h), Image.Resampling.BILINEAR)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    mask = mask.crop((left, top, left + size, top + size))
    return mask.filter(ImageFilter.GaussianBlur(radius=3.0))


def load_image_tensor(path: str) -> torch.Tensor:
    if path.lower().endswith(".png"):
        with Image.open(path) as img_pil:
            arr = np.asarray(resize_crop_pil(img_pil), dtype=np.float32) / 255.0
    else:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(resize_crop_cv2(img), cv2.COLOR_BGR2RGB)
        arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def load_mask_tensor(path: str | None) -> torch.Tensor | None:
    if not path:
        return None
    mask_path = Path(path)
    if not mask_path.exists():
        return None
    with Image.open(mask_path) as img:
        arr = np.asarray(resize_crop_mask(img), dtype=np.float32) / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0)


def save_tensor_image(tensor: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().clamp(0.0, 1.0).mul(255.0).byte().permute(1, 2, 0).cpu().numpy()
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]):
        raise RuntimeError(f"Failed to write image: {path}")


def save_tensor_mask(mask: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask.detach().clamp(0.0, 1.0).mul(255.0).byte().squeeze(0).cpu().numpy()
    if not cv2.imwrite(str(path), arr):
        raise RuntimeError(f"Failed to write mask: {path}")


def apply_ops_torch(images: torch.Tensor, params: list[dict[str, Any]]) -> torch.Tensor:
    y = images
    b = y.shape[0]
    exposure = torch.tensor([p.get("exposure", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    contrast = torch.tensor([p.get("contrast", 1.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    saturation = torch.tensor([p.get("saturation", 1.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    temperature = torch.tensor([p.get("temperature", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    tint = torch.tensor([p.get("tint", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    gamma = torch.tensor([p.get("gamma", 1.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    shadow_lift = torch.tensor([p.get("shadow_lift", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    highlight_rolloff = torch.tensor([p.get("highlight_rolloff", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)
    vibrance = torch.tensor([p.get("vibrance", 0.0) for p in params], device=y.device, dtype=y.dtype).view(b, 1, 1, 1)

    y = y * torch.pow(torch.tensor(2.0, device=y.device, dtype=y.dtype), exposure)
    y = (y - 0.5) * contrast + 0.5

    red_gain = 1.0 + temperature
    blue_gain = 1.0 - temperature
    green_gain = 1.0 + tint
    gains = torch.cat([red_gain, green_gain, blue_gain], dim=1)
    y = y * gains

    lum = y[:, 0:1] * 0.299 + y[:, 1:2] * 0.587 + y[:, 2:3] * 0.114
    sat_boost = saturation + vibrance * (1.0 - (y - lum).abs().mean(dim=1, keepdim=True).clamp(0.0, 1.0))
    y = lum + (y - lum) * sat_boost
    y = y + (1.0 - y) * shadow_lift * (1.0 - lum).pow(2.0)
    y = y - y * highlight_rolloff * lum.pow(2.0)
    y = y.clamp(0.0, 1.0).pow(gamma)

    channel_bias = torch.tensor(
        [[p.get("red", 0.0), p.get("green", 0.0), p.get("blue", 0.0)] for p in params],
        device=y.device,
        dtype=y.dtype,
    ).view(b, 3, 1, 1)
    y = y + channel_bias
    return y.clamp(0.0, 1.0)


def flat_param(record: dict[str, Any]) -> dict[str, Any]:
    op = record["operation"]
    if op["kind"] == "inverse_degrade":
        return op
    merged: dict[str, Any] = {}
    for key in ("L", "GC", "SC"):
        merged.update(op.get(key, {}))
    return merged


def add_degrade_noise(images: torch.Tensor, params: list[dict[str, Any]], base_seed: int) -> torch.Tensor:
    std = torch.tensor([p.get("noise_std", 0.0) for p in params], device=images.device, dtype=images.dtype).view(-1, 1, 1, 1)
    if float(std.max()) <= 0:
        return images
    generator = torch.Generator(device=images.device)
    generator.manual_seed(base_seed)
    noise = torch.randn(images.shape, device=images.device, dtype=images.dtype, generator=generator)
    return (images + noise * std).clamp(0.0, 1.0)


def apply_lut(images: torch.Tensor, luts: torch.Tensor) -> torch.Tensor:
    b, _, h, w = images.shape
    d = luts.shape[1]
    coords = images.permute(0, 2, 3, 1).clamp(0.0, 1.0) * (d - 1)
    low = coords.floor().long()
    high = (low + 1).clamp(max=d - 1)
    frac = coords - low.to(coords.dtype)

    r0, g0, b0 = low[..., 0], low[..., 1], low[..., 2]
    r1, g1, b1 = high[..., 0], high[..., 1], high[..., 2]
    wr, wg, wb = frac[..., 0:1], frac[..., 1:2], frac[..., 2:3]

    lut_flat = luts.reshape(b, d * d * d, 3)

    def gather(ri: torch.Tensor, gi: torch.Tensor, bi: torch.Tensor) -> torch.Tensor:
        idx = (ri * d * d + gi * d + bi).reshape(b, -1, 1).expand(-1, -1, 3)
        return torch.gather(lut_flat, 1, idx).reshape(b, h, w, 3)

    c000 = gather(r0, g0, b0)
    c001 = gather(r0, g0, b1)
    c010 = gather(r0, g1, b0)
    c011 = gather(r0, g1, b1)
    c100 = gather(r1, g0, b0)
    c101 = gather(r1, g0, b1)
    c110 = gather(r1, g1, b0)
    c111 = gather(r1, g1, b1)

    c00 = c000 * (1 - wb) + c001 * wb
    c01 = c010 * (1 - wb) + c011 * wb
    c10 = c100 * (1 - wb) + c101 * wb
    c11 = c110 * (1 - wb) + c111 * wb
    c0 = c00 * (1 - wg) + c01 * wg
    c1 = c10 * (1 - wg) + c11 * wg
    out = c0 * (1 - wr) + c1 * wr
    return out.permute(0, 3, 1, 2).clamp(0.0, 1.0)


def heuristic_mask(images: torch.Tensor, kinds: list[str]) -> torch.Tensor:
    b, _, h, w = images.shape
    yy = torch.linspace(0.0, 1.0, h, device=images.device, dtype=images.dtype).view(1, 1, h, 1)
    xx = torch.linspace(0.0, 1.0, w, device=images.device, dtype=images.dtype).view(1, 1, 1, w)
    masks: list[torch.Tensor] = []
    r, g, bl = images[:, 0:1], images[:, 1:2], images[:, 2:3]
    lum = r * 0.299 + g * 0.587 + bl * 0.114
    for i, kind in enumerate(kinds):
        if kind == "heuristic_sky":
            m = ((bl[i:i + 1] - r[i:i + 1]) * 4.0 + (0.58 - yy) * 2.6 + (lum[i:i + 1] - 0.35)).sigmoid()
        elif kind == "heuristic_foliage":
            m = ((g[i:i + 1] - r[i:i + 1]) * 5.0 + (g[i:i + 1] - bl[i:i + 1]) * 3.0).sigmoid()
        else:
            dist = ((xx - 0.5) ** 2 + (yy - 0.52) ** 2).sqrt()
            m = ((dist - 0.30) * 10.0).sigmoid()
        masks.append(m.clamp(0.0, 1.0))
    mask = torch.cat(masks, dim=0)
    mask = F.avg_pool2d(mask, kernel_size=17, stride=1, padding=8)
    return mask.clamp(0.0, 1.0)


def load_batch(records: list[dict[str, Any]], device: torch.device, needs_mask: bool) -> LoadedBatch:
    images = torch.stack([load_image_tensor(r["input_source_path"]) for r in records], dim=0).to(device, non_blocking=True)
    masks = None
    if needs_mask:
        loaded_masks: list[torch.Tensor] = []
        for r in records:
            mask = load_mask_tensor(r.get("mask_source_path"))
            if mask is None:
                loaded_masks.append(torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE))
            else:
                loaded_masks.append(mask)
        masks = torch.stack(loaded_masks, dim=0).to(device, non_blocking=True)
    return LoadedBatch(records=records, images=images, masks=masks)


def abs_out(out_dir: Path, rel_path: str | None) -> Path | None:
    if rel_path is None:
        return None
    return out_dir / rel_path


def output_exists(out_dir: Path, record: dict[str, Any]) -> bool:
    input_path = abs_out(out_dir, record["output_input_path"])
    target_path = abs_out(out_dir, record["output_target_path"])
    mask_path = abs_out(out_dir, record.get("output_mask_path"))
    if input_path is None or target_path is None:
        return False
    ok = input_path.exists() and target_path.exists()
    if mask_path is not None:
        ok = ok and mask_path.exists()
    return ok


def generate_s0(out_dir: Path, records: list[dict[str, Any]]) -> None:
    for record in records:
        input_path = abs_out(out_dir, record["output_input_path"])
        target_path = abs_out(out_dir, record["output_target_path"])
        assert input_path is not None and target_path is not None
        if input_path.exists() and target_path.exists():
            continue
        img_in = load_image_tensor(record["input_source_path"])
        img_tgt = load_image_tensor(record["target_source_path"])
        save_tensor_image(img_in, input_path)
        save_tensor_image(img_tgt, target_path)


def generate_non_s0(
    out_dir: Path,
    records: list[dict[str, Any]],
    device: torch.device,
    lut_bank: np.ndarray,
    seed: int,
) -> None:
    branch = records[0]["branch"]
    needs_mask = branch == "S4_local_semantic_4d_lut"
    batch = load_batch(records, device, needs_mask=needs_mask)
    images = batch.images

    if branch == "S1_auto_inverse_lite":
        params = [r["operation"] for r in records]
        target = images
        degraded = apply_ops_torch(images, params)
        degraded = add_degrade_noise(degraded, params, seed)
        input_images = degraded
        target_images = target
        masks = None
    elif branch == "S2_param_l_gc_sc":
        params = [flat_param(r) for r in records]
        input_images = images
        target_images = apply_ops_torch(images, params)
        masks = None
    elif branch == "S3_style_lut":
        indices = [int(r["operation"]["lut_index"]) for r in records]
        luts = torch.from_numpy(lut_bank[indices].astype(np.float32)).to(device=device, dtype=images.dtype, non_blocking=True)
        input_images = images
        target_images = apply_lut(images, luts)
        masks = None
    elif branch == "S4_local_semantic_4d_lut":
        indices = [int(r["operation"]["lut_index"]) for r in records]
        luts = torch.from_numpy(lut_bank[indices].astype(np.float32)).to(device=device, dtype=images.dtype, non_blocking=True)
        styled = apply_lut(images, luts)
        mask_kinds = [r["operation"]["mask_kind"] for r in records]
        heuristic_needed = [m.startswith("heuristic_") for m in mask_kinds]
        assert batch.masks is not None
        masks = batch.masks
        if any(heuristic_needed):
            h_masks = heuristic_mask(images, mask_kinds)
            selector = torch.tensor(heuristic_needed, device=device, dtype=torch.bool).view(-1, 1, 1, 1)
            masks = torch.where(selector, h_masks, masks)
        blends = torch.tensor([r["operation"].get("blend", 0.7) for r in records], device=device, dtype=images.dtype).view(-1, 1, 1, 1)
        masks = (masks * blends).clamp(0.0, 1.0)
        input_images = images
        target_images = images * (1.0 - masks) + styled * masks
    else:
        raise ValueError(branch)

    for i, record in enumerate(records):
        input_path = abs_out(out_dir, record["output_input_path"])
        target_path = abs_out(out_dir, record["output_target_path"])
        assert input_path is not None and target_path is not None
        save_tensor_image(input_images[i], input_path)
        save_tensor_image(target_images[i], target_path)
        if masks is not None:
            mask_path = abs_out(out_dir, record.get("output_mask_path"))
            assert mask_path is not None
            save_tensor_mask(masks[i], mask_path)


def generate_shard(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    plan_path = out_dir / "plan.jsonl"
    if not plan_path.exists():
        raise FileNotFoundError(f"Missing plan: {plan_path}")
    luts_path = out_dir / "lut_bank" / "analytic_luts.npy"
    if not luts_path.exists():
        raise FileNotFoundError(f"Missing LUT bank: {luts_path}")

    records = load_jsonl(plan_path)
    branch_filter = set(args.branches.split(",")) if args.branches else None
    if branch_filter:
        unknown = sorted(branch_filter - set(BRANCH_COUNTS))
        if unknown:
            raise ValueError(f"Unknown branches in --branches: {unknown}")
        records = [r for r in records if r["branch"] in branch_filter]
    if args.start_index is not None:
        records = [r for r in records if r["index"] >= args.start_index]
    if args.end_index is not None:
        records = [r for r in records if r["index"] < args.end_index]
    shard_records = [r for r in records if r["index"] % args.num_shards == args.shard_id]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    lut_bank = np.load(luts_path, mmap_mode="r")

    shard_dir = out_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_name = args.run_name or "all"
    safe_run_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in run_name)
    manifest_path = shard_dir / f"manifest_{safe_run_name}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"
    tmp_manifest = manifest_path.with_suffix(".jsonl.tmp")
    error_path = shard_dir / f"errors_{safe_run_name}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.jsonl"

    generated = 0
    skipped = 0
    errors = 0
    with tmp_manifest.open("w", encoding="utf-8") as mf, error_path.open("a", encoding="utf-8") as ef:
        for start in tqdm(range(0, len(shard_records), args.batch_size), desc=f"shard {args.shard_id}/{args.num_shards}"):
            chunk = shard_records[start:start + args.batch_size]
            pending = [r for r in chunk if not output_exists(out_dir, r)]
            try:
                s0_pending = [r for r in pending if r["branch"] == "S0_expert_anchor"]
                if s0_pending:
                    generate_s0(out_dir, s0_pending)
                for branch in ("S1_auto_inverse_lite", "S2_param_l_gc_sc", "S3_style_lut", "S4_local_semantic_4d_lut"):
                    branch_pending = [r for r in pending if r["branch"] == branch]
                    if branch_pending:
                        generate_non_s0(out_dir, branch_pending, device, lut_bank, args.seed + start)
                generated += len(pending)
                skipped += len(chunk) - len(pending)
                for record in chunk:
                    out_record = dict(record)
                    out_record["input_path"] = str(abs_out(out_dir, record["output_input_path"]))
                    out_record["target_path"] = str(abs_out(out_dir, record["output_target_path"]))
                    out_record["mask_path"] = str(abs_out(out_dir, record["output_mask_path"])) if record.get("output_mask_path") else None
                    json_dump_line(mf, out_record)
            except Exception as exc:  # keep long shard runs alive where possible
                errors += len(chunk)
                for record in chunk:
                    json_dump_line(ef, {
                        "id": record.get("id"),
                        "index": record.get("index"),
                        "branch": record.get("branch"),
                        "error": repr(exc),
                    })
                if args.stop_on_error:
                    raise
    tmp_manifest.replace(manifest_path)
    write_json(shard_dir / f"stats_{safe_run_name}_shard_{args.shard_id:02d}_of_{args.num_shards:02d}.json", {
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "run_name": safe_run_name,
        "branches": sorted(branch_filter) if branch_filter else "all",
        "start_index": args.start_index,
        "end_index": args.end_index,
        "records": len(shard_records),
        "generated": generated,
        "skipped_existing": skipped,
        "errors": errors,
        "device": str(device),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    print(f"Shard {args.shard_id}: generated={generated} skipped={skipped} errors={errors}")


def finalize(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    plan_path = out_dir / "plan.jsonl"
    records = load_jsonl(plan_path)
    shard_paths = sorted((out_dir / "shards").glob("manifest_*_shard_*_of_*.jsonl"))
    by_id: dict[str, dict[str, Any]] = {}
    for path in shard_paths:
        for row in load_jsonl(path):
            by_id[row["id"]] = row

    missing_manifest = []
    missing_files = []
    branch_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    for record in records:
        row = by_id.get(record["id"])
        if row is None and args.recover_from_files:
            input_path = abs_out(out_dir, record["output_input_path"])
            target_path = abs_out(out_dir, record["output_target_path"])
            mask_path = abs_out(out_dir, record.get("output_mask_path"))
            if input_path and target_path and input_path.exists() and target_path.exists() and (mask_path is None or mask_path.exists()):
                row = dict(record)
                row["input_path"] = str(input_path)
                row["target_path"] = str(target_path)
                row["mask_path"] = str(mask_path) if mask_path else None
                by_id[row["id"]] = row
        if row is None:
            missing_manifest.append(record["id"])
            continue
        input_path = Path(row["input_path"])
        target_path = Path(row["target_path"])
        mask_path = Path(row["mask_path"]) if row.get("mask_path") else None
        if not input_path.exists() or not target_path.exists() or (mask_path is not None and not mask_path.exists()):
            missing_files.append(record["id"])
            continue
        branch_counts[row["branch"]] = branch_counts.get(row["branch"], 0) + 1
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1

    manifest_path = out_dir / "manifest.jsonl"
    tmp = manifest_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            row = by_id.get(record["id"])
            if row is not None:
                json_dump_line(f, row)
    tmp.replace(manifest_path)

    stats = {
        "num_planned": len(records),
        "num_manifest_rows": len(by_id),
        "num_valid": sum(branch_counts.values()),
        "missing_manifest": len(missing_manifest),
        "missing_files": len(missing_files),
        "branch_counts": dict(sorted(branch_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "manifest_path": str(manifest_path),
        "finalized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json(out_dir / "quality_report.json", stats)
    print(json.dumps(stats, indent=2, sort_keys=True))
    if args.require_complete and stats["num_valid"] != sum(BRANCH_COUNTS.values()):
        raise SystemExit("Dataset is not complete")


def status(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    plan_path = out_dir / "plan.jsonl"
    records = load_jsonl(plan_path) if plan_path.exists() else []
    existing_inputs = list((out_dir / "images" / "input").glob("*/*.jpg")) if (out_dir / "images" / "input").exists() else []
    existing_targets = list((out_dir / "images" / "target").glob("*/*.jpg")) if (out_dir / "images" / "target").exists() else []
    masks = list((out_dir / "masks").glob("*/*.png")) if (out_dir / "masks").exists() else []
    print(json.dumps({
        "out": str(out_dir),
        "planned": len(records),
        "planned_by_branch": count_by(records, "branch"),
        "input_images": len(existing_inputs),
        "target_images": len(existing_targets),
        "masks": len(masks),
        "shard_manifests": len(list((out_dir / "shards").glob("manifest_*_shard_*_of_*.jsonl"))) if (out_dir / "shards").exists() else 0,
    }, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="dataset output directory")
    parser.add_argument("--seed", type=int, default=SEED)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--num-luts", type=int, default=2048)
    p_plan.set_defaults(func=build_plan)

    p_luts = sub.add_parser("luts")
    p_luts.add_argument("--num-luts", type=int, default=2048)
    p_luts.add_argument("--lut-size", type=int, default=32)
    p_luts.add_argument("--force", action="store_true")
    p_luts.set_defaults(func=build_luts)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--num-shards", type=int, required=True)
    p_gen.add_argument("--shard-id", type=int, required=True)
    p_gen.add_argument("--device", default="cuda:0")
    p_gen.add_argument("--batch-size", type=int, default=32)
    p_gen.add_argument("--branches", default=None, help="comma-separated branch filter")
    p_gen.add_argument("--start-index", type=int, default=None, help="inclusive global plan index lower bound")
    p_gen.add_argument("--end-index", type=int, default=None, help="exclusive global plan index upper bound")
    p_gen.add_argument("--run-name", default=None, help="label used in shard manifest filenames")
    p_gen.add_argument("--stop-on-error", action="store_true")
    p_gen.set_defaults(func=generate_shard)

    p_final = sub.add_parser("finalize")
    p_final.add_argument("--require-complete", action="store_true")
    p_final.add_argument("--recover-from-files", action="store_true", help="include records whose files exist even if a killed shard did not finalize its manifest")
    p_final.set_defaults(func=finalize)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=status)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
