"""Pass 1: join the ten builds' ``sft.jsonl`` text rows with their published
indexed-tar image members, producing one row per ``sft_id``.

Nothing is read from the tar payloads here -- only ``metadata.jsonl`` and the
per-shard ``*.idx.jsonl`` -- so this pass is cheap and can be re-run freely.

Two independent sources are joined:

* ``/mnt/nfs/bc/data/builds/<build_id>/sft.jsonl``  -- instruction, seven-segment
  reasoning, recipe (LUT identity), task type, winner confidence;
* ``/mnt/nfs/bc/data/datasets/sft/<build_id>/batch-*/`` -- ``metadata.jsonl``
  (source_id / mask / taxonomy per sample) and the shard indexes that locate the
  ``I_in`` member (``.in.jpg`` / ``.in.png`` / ...).

A sample that exists in one source but not the other is *not* silently dropped:
it is emitted into the rejection stream with an explicit reason.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterator

from .config import BUILD_ROOT, BUILDS, DATASET_ROOT, all_batch_dirs

IN_SUFFIX_PREFIX = ".in."


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def scan_batch(batch_dir: Path) -> dict[str, dict[str, Any]]:
    """Return ``sft_id -> {meta..., image locator}`` for one published batch."""
    meta_by_sample: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(batch_dir / "metadata.jsonl"):
        # the ``.vrmeta.json`` row is the only one carrying the sample-level
        # fields; the two image rows carry ``role`` and nothing else useful.
        if "preset_id" not in row:
            continue
        meta_by_sample[row["sample_id"]] = row

    locators: dict[str, dict[str, Any]] = {}
    for index_path in sorted((batch_dir / "indexes").glob("shard-*.idx.jsonl")):
        for row in _iter_jsonl(index_path):
            if not row["suffix"].startswith(IN_SUFFIX_PREFIX):
                continue
            sample_id = row["sample_id"]
            if sample_id in locators:
                raise RuntimeError(f"duplicate I_in member for {sample_id} in {batch_dir}")
            locators[sample_id] = {
                "root": str(batch_dir),
                "shard": row["shard"],
                "member": row["member"],
                "suffix": row["suffix"],
                "offset": row["offset_data"],
                "length": row["length"],
                "sha256": row["sha256"],
            }

    out: dict[str, dict[str, Any]] = {}
    for sample_id, meta in meta_by_sample.items():
        # A published candidate whose annotation failed carries ``sft_id: null``
        # and has no ``sft.jsonl`` row: it was never an SFT sample, so it is
        # counted (below) rather than reported as a dropped one.
        if not meta.get("sft_id"):
            out.setdefault("__unannotated__", {"count": 0})["count"] += 1
            continue
        loc = locators.get(sample_id)
        out[meta["sft_id"]] = {
            "sample_id": sample_id,
            "batch": batch_dir.name,
            "source_id": meta.get("source_id"),
            "group_id": meta.get("group_id"),
            "candidate_id": meta.get("candidate_id"),
            "preset_id": meta.get("preset_id"),
            "mask_id": meta.get("mask_id"),
            "major": meta.get("major"),
            "minor": meta.get("minor"),
            "region": meta.get("region"),
            "render_mode": meta.get("render_mode"),
            "slot_id": meta.get("slot_id"),
            "scene": meta.get("scene"),
            "winner_confidence": meta.get("winner_confidence"),
            "winner_rank": meta.get("winner_rank"),
            "i_in_path": meta.get("i_in_path"),
            "image": loc,
        }
    return out


def scan_build(code: str, build_id: str, workers: int = 16) -> tuple[dict[str, dict[str, Any]], int]:
    batches = all_batch_dirs(build_id)
    if not batches:
        raise RuntimeError(f"no published batches for {build_id}")
    merged: dict[str, dict[str, Any]] = {}
    unannotated = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for part in pool.map(scan_batch, batches):
            unannotated += part.pop("__unannotated__", {}).get("count", 0)
            for sft_id, row in part.items():
                if sft_id in merged:
                    raise RuntimeError(f"{build_id}: sft_id {sft_id} published twice")
                row["build"] = code
                row["build_id"] = build_id
                merged[sft_id] = row
    return merged, unannotated


def scan_text(build_id: str) -> dict[str, dict[str, Any]]:
    """Read ``sft.jsonl``: instruction, reasoning, recipe, task type."""
    rows: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(BUILD_ROOT / build_id / "sft.jsonl"):
        recipe = row.get("recipe") or {}
        if not isinstance(recipe, dict):
            recipe = {"preset_id": recipe}
        rows[row["sft_id"]] = {
            "instruction": row.get("instruction"),
            "instruction_short": row.get("instruction_short"),
            "reasoning": row.get("reasoning"),
            "task_type": row.get("task_type"),
            "winner_confidence": row.get("winner_confidence"),
            "winner_rank": row.get("winner_rank"),
            "lut_id": recipe.get("preset_id"),
            "preset_path": recipe.get("preset_path"),
            "recipe_format": recipe.get("format"),
            "render_mode": recipe.get("render_mode"),
            "local_amount": (row.get("local") or {}).get("amount") if isinstance(row.get("local"), dict) else None,
            "local_region": (row.get("local") or {}).get("region") if isinstance(row.get("local"), dict) else None,
            "local_subject": (row.get("local") or {}).get("subject") if isinstance(row.get("local"), dict) else None,
            "i_in_path": row.get("I_in"),
        }
    return rows


def scan_all(workers: int = 16) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Return ``(rows, rejections, unannotated_per_build)`` for every build."""
    rows: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    unannotated: dict[str, int] = {}
    for code, build_id in BUILDS.items():
        published, unannotated[code] = scan_build(code, build_id, workers=workers)
        text = scan_text(build_id)
        for sft_id, t in text.items():
            pub = published.get(sft_id)
            if pub is None:
                rejections.append({
                    "sft_id": sft_id, "build": code, "reason": "not_published",
                    "detail": "sft.jsonl row has no member in the published dataset",
                })
                continue
            if pub["image"] is None:
                rejections.append({
                    "sft_id": sft_id, "build": code, "reason": "image_missing",
                    "detail": f"no {IN_SUFFIX_PREFIX}* member for sample {pub['sample_id']}",
                })
                continue
            merged = {"sft_id": sft_id, **pub, **{k: v for k, v in t.items() if v is not None or k not in pub}}
            # ``winner_confidence``/``render_mode`` exist in both; the build's
            # sft.jsonl is the authority for the text-side fields.
            merged["winner_confidence"] = t["winner_confidence"] or pub["winner_confidence"]
            merged["lut_id"] = t["lut_id"]
            rows.append(merged)
        for sft_id in published.keys() - text.keys():
            rejections.append({
                "sft_id": sft_id, "build": code, "reason": "no_text_row",
                "detail": "published sample has no sft.jsonl row",
            })
    return rows, rejections, unannotated
