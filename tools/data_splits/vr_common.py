"""Shared helpers for tools/data_splits (task T1).

Only Python stdlib. Paths and constants are the single source of truth
for every script in this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Iterator

# ---------------------------------------------------------------------------
# Constants (frozen; see README.md)
# ---------------------------------------------------------------------------

# S-split hash seed. NEVER change without regenerating the side tables and
# bumping the table name / version everywhere.
SPLIT_SEED = "verasplit-v1"

# bucket ranges (inclusive) over sha1 mod 100
S_TRAIN = range(0, 90)   # 0-89
S_VAL = range(90, 95)    # 90-94
S_TEST = range(95, 100)  # 95-99

JOURNAL_ROOT = "/var/cache/veradata/annot_review/journal-archive"
DATASETS_ROOT = "/mnt/nfs/bc/data/datasets"
IMG_BANK_ROOT = os.path.join(DATASETS_ROOT, "img", "unknown")

TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_DIR = "/home/bc/VeraRetouch/experiments/tooling-wave1/data_splits"

# PPR10K official split (verified 2026-08-02 against github.com/csjliang/PPR10K
# README: "train with the first 8,875 files and validate with the last 2286
# files"; arXiv 2105.09180: 1,681 groups / 11,161 photos total).
PPR10K_TRAIN_FILES = 8875
PPR10K_VAL_FIRST_INDEX = 8875  # any flat index >= this is official-val territory
# Paper-derived group boundary (1,356 train groups). NOT re-verified verbatim
# from a primary source; treat as advisory (see NOTES.md 待主 agent 决策 #1).
PPR10K_TRAIN_GROUPS_PAPER = 1356


# ---------------------------------------------------------------------------
# S-split
# ---------------------------------------------------------------------------

def s_bucket(source_id: str) -> int:
    """Stable hash bucket in [0, 100) for a source_id.

    sha1 of "<seed>:<source_id>", first 8 hex chars as int, mod 100.
    """
    h = hashlib.sha1(f"{SPLIT_SEED}:{source_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % 100


def s_split(source_id: str) -> str:
    b = s_bucket(source_id)
    if b in S_TRAIN:
        return "train"
    if b in S_VAL:
        return "val"
    return "test"


# ---------------------------------------------------------------------------
# P-split (stratified by minor, deterministic within layer)
# ---------------------------------------------------------------------------

def p_split_layer(preset_ids: list[str]) -> dict[str, str]:
    """Assign train/val/test within one minor layer.

    Sort preset_ids ascending (hash-like ids => lexicographic ~ random),
    take the last n_test as test, the n_val before them as val, rest train.
    n_val = n_test = floor(n * 0.05 + 0.5) (round-half-up), clamped so that
    train stays non-empty.
    """
    ids = sorted(preset_ids)
    n = len(ids)
    n_side = int(n * 0.05 + 0.5)
    n_val = n_test = n_side
    while n_val + n_test >= n and (n_val > 0 or n_test > 0):
        if n_test >= n_val and n_test > 0:
            n_test -= 1
        elif n_val > 0:
            n_val -= 1
    out: dict[str, str] = {}
    for i, pid in enumerate(ids):
        if i >= n - n_test:
            out[pid] = "test"
        elif i >= n - n_test - n_val:
            out[pid] = "val"
        else:
            out[pid] = "train"
    return out


def p_split_increment(layer_existing: dict[str, str],
                      new_ids: list[str]) -> dict[str, str]:
    """Assign splits to NEW preset_ids joining one minor layer (frozen-append).

    Existing assignments in `layer_existing` (preset_id -> split) are FROZEN —
    this function never reassigns them and only returns entries for `new_ids`.
    Target val/test quotas for the grown layer are computed with the same
    round-half-up + train-non-empty rule as `p_split_layer`; the new ids
    (sorted ascending) fill only the *incremental* quota
    (target − already held), tail = test, previous = val, rest train.
    With `layer_existing == {}` this is identical to `p_split_layer`.
    """
    ids = sorted(new_ids)
    n_total = len(layer_existing) + len(ids)
    cur_val = sum(1 for v in layer_existing.values() if v == "val")
    cur_test = sum(1 for v in layer_existing.values() if v == "test")
    n_side = int(n_total * 0.05 + 0.5)
    n_val_t = n_test_t = n_side
    while n_val_t + n_test_t >= n_total and (n_val_t > 0 or n_test_t > 0):
        if n_test_t >= n_val_t and n_test_t > 0:
            n_test_t -= 1
        elif n_val_t > 0:
            n_val_t -= 1
    add_test = min(max(0, n_test_t - cur_test), len(ids))
    add_val = min(max(0, n_val_t - cur_val), len(ids) - add_test)
    out: dict[str, str] = {}
    m = len(ids)
    for i, pid in enumerate(ids):
        if i >= m - add_test:
            out[pid] = "test"
        elif i >= m - add_test - add_val:
            out[pid] = "val"
        else:
            out[pid] = "train"
    return out


# ---------------------------------------------------------------------------
# Pools
# ---------------------------------------------------------------------------

def pool_of(source_path: str) -> str:
    p = source_path
    if "/presets_sources/" in p:
        return p.split("/presets_sources/")[1].split("/")[0]
    if "/_scratch/unsplash/" in p:
        return "unsplash"
    if "/ppr10k/source/" in p:
        return "ppr10k"
    if "/RAISE-6k/" in p:
        return "raise6k"
    if "/fivek_gold/" in p:
        return "fivek_gold"
    if "MMArt-PPR10k" in p:
        return "mmart_ppr10k"
    return "other"


# ---------------------------------------------------------------------------
# Journal iteration
# ---------------------------------------------------------------------------

def completed_builds(root: str = JOURNAL_ROOT) -> list[str]:
    """A build counts as completed iff its journal archive has groups.jsonl."""
    out = []
    for name in sorted(os.listdir(root)):
        if os.path.isfile(os.path.join(root, name, "groups.jsonl")):
            out.append(name)
    return out


def iter_groups(build: str, root: str = JOURNAL_ROOT,
                limit: int | None = None) -> Iterator[dict]:
    path = os.path.join(root, build, "groups.jsonl")
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield json.loads(line)


def group_conf_class(g: dict) -> str:
    """normal / low / abstain / unannotated."""
    c = g.get("winner_confidence")
    if c is None:
        return "unannotated"
    return c
