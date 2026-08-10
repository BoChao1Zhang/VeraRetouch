"""Frozen paths, build codes and schema identifiers for the sft2seg dataset.

Every path here was verified to exist on 2026-08-04 before being written down;
nothing is inferred from a naming convention.
"""

from __future__ import annotations

from pathlib import Path

# The frozen numbers live in one place only (spec 5 / spec 6); re-declaring them
# here would let the data and the trainer drift apart.
from q3vl.train.constants import (  # noqa: F401  (re-exported on purpose)
    IMAGE_ALIGN_FACTOR,
    IMAGE_LONG_SIDE_MAX,
    IMAGE_MAX_ASPECT_RATIO,
    IMAGE_SHORT_SIDE,
    MODEL_MAX_LENGTH,
)

# --- schema ---------------------------------------------------------------
RECORD_SCHEMA = "q3vl.sft2seg/1"
SPLIT_SCHEMA = "q3vl.sft2seg.splits/1"

# --- inputs ---------------------------------------------------------------
# build code -> build_id.  The ten production builds named by SFT spec 3.1.
BUILDS: dict[str, str] = {
    "g1": "prod-g1-global25k-20260731",
    "g2": "prod-g2-global25k-20260731",
    "g3": "prod-g3-global25k-20260801",
    "g4": "prod-g4-global25k-20260801",
    "l1": "prod-l1-local17k-20260731",
    "l2": "prod-l2-local17k-20260731",
    "l3": "prod-l3-local17k-20260731",
    "l4": "prod-l4-local17k-20260801",
    "l5": "prod-l5-local17k-20260801",
    "l6": "prod-l6-local17k-20260801",
}

# ``sft.jsonl`` (instruction + seven-segment reasoning + recipe) lives here.
BUILD_ROOT = Path("/mnt/nfs-ro/bc/data/builds")          # READ (see NFS_RO_ROOT)
# the published indexed-tar projection (images + per-sample vrmeta) lives here.
DATASET_ROOT = Path("/mnt/nfs-ro/bc/data/datasets/sft")  # READ
# frozen split authority (SFT spec 3.1).
SPLIT_ROOT = DATASET_ROOT / "splits-20260803"
TRAIN_IDS = SPLIT_ROOT / "train_sft_ids.txt"
EVAL_IDS = SPLIT_ROOT / "eval_sft_ids.txt"
DEDUP_DROP_IDS = SPLIT_ROOT / "dedup_drop_sft_ids.txt"

MODEL_DIR = Path("/home/bc/data/models/Qwen3-VL-4B-Instruct")

# --- outputs --------------------------------------------------------------
# WRITE: publication root stays on the rw (hard) mount; writes go through nfsx.
OUT_ROOT = Path("/mnt/nfs/bc/data/datasets/sft2seg-20260804")
WORK_DIR = OUT_ROOT / "work"          # a handful of large JSONL intermediates
RECORDS_DIR = OUT_ROOT / "records"    # indexed tar dataset: <sft_id>.rec.json
IMAGES_DIR = OUT_ROOT / "images"      # indexed tar dataset: <sft_id>.jpg (spec-5 sized)
SPLIT_DIR = OUT_ROOT / "splits"       # per-split id lists + nested training indexes
MANIFEST_DIR = OUT_ROOT / "manifest"  # terminal manifest + digests
LOG_DIR = OUT_ROOT / "logs"

REPORT_DIR = Path(
    "/home/bc/VeraRetouch/experiments/Q3VL_metacanvas_where_what_20260804/s0_preflight/data"
)

# --- intermediates --------------------------------------------------------
SAMPLES_JSONL = WORK_DIR / "samples.jsonl"        # pass 1: metadata + locators
GEOMETRY_JSONL = WORK_DIR / "geometry.jsonl"      # pass 2: header geometry
LENGTHS_JSONL = WORK_DIR / "lengths.jsonl"        # pass 3: token lengths
REJECTS_JSONL = WORK_DIR / "rejections.jsonl"     # every dropped sample, with reason
PLAN_JSONL = WORK_DIR / "plan.jsonl"              # survivors + split assignment

# --- packing ---------------------------------------------------------------
# METACANVAS 2.3: uncompressed tar, 1-4 GiB per shard.
RECORD_SHARD_BYTES = 1 * 1024**3
IMAGE_SHARD_BYTES = 2 * 1024**3
JPEG_QUALITY = 95
# 4:4:4.  A colour-grading dataset must not be stored with chroma subsampling.
JPEG_SUBSAMPLING = 0

# --- split construction ----------------------------------------------------
SPLIT_SEED = 20260804
# METACANVAS 2.2 asks for a T_lut_unseen with enough samples to report on.
LUT_RESERVE_TARGET_EVAL = 700
# Task card: <=5% of the training set removed is adopted without escalation.
LUT_RESERVE_TRAIN_BUDGET = 0.05


def all_batch_dirs(build_id: str) -> list[Path]:
    root = DATASET_ROOT / build_id
    return sorted(p for p in root.glob("batch-*") if p.is_dir())
