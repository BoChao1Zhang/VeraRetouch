"""Stage functions for the sft2seg build.  Each stage is independently re-runnable.

    scan      -> work/samples.jsonl     (text x published-member join)
    geometry  -> work/geometry.jsonl    (header probe + spec-5 plan)
    convert   -> work/lengths.jsonl     (two-segment target + token lengths)
    plan      -> work/plan.jsonl        (survivors + split assignment)
    records   -> records/               (indexed tar dataset of .rec.json)
    images    -> images/                (indexed tar dataset of spec-5 .jpg)
    manifest  -> manifest/terminal_manifest.json + splits/*.index.jsonl

Rejections accumulate in ``work/rejections.jsonl`` with one row per dropped
sample: ``{sft_id, build, stage, reason, detail}``.  Nothing is dropped silently
anywhere in this file.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from . import config as C
from .twoseg import ReasoningRejected, contains_legacy_tag, convert


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            n += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return n


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


# --------------------------------------------------------------------------
# stage 1: scan
# --------------------------------------------------------------------------
def stage_scan() -> dict[str, Any]:
    from .scan import scan_all

    rows, rejects, unannotated = scan_all()
    n = _write_jsonl(C.SAMPLES_JSONL, rows)
    _write_jsonl(C.WORK_DIR / "rejections_scan.jsonl",
                 [{**r, "stage": "scan"} for r in rejects])
    _log(f"scan: {n} rows, {len(rejects)} rejected, "
         f"{sum(unannotated.values())} published-but-unannotated candidates skipped")
    return {"rows": n, "rejected": len(rejects),
            "reasons": dict(Counter(r["reason"] for r in rejects)),
            "unannotated_published_candidates": unannotated,
            "per_build": dict(Counter(r["build"] for r in rows))}


# --------------------------------------------------------------------------
# stage 2: geometry
# --------------------------------------------------------------------------
def stage_geometry(workers: int = 32) -> dict[str, Any]:
    from .headers import run

    rows = _read_jsonl(C.SAMPLES_JSONL)
    _log(f"geometry: probing {len(rows)} image headers")
    results = run(rows, workers=workers, log=_log)
    n = _write_jsonl(C.GEOMETRY_JSONL, results)
    reasons = Counter(r["reason"] for r in results if r.get("reason"))
    _log(f"geometry: {n} probed, {sum(reasons.values())} rejected {dict(reasons)}")
    return {"probed": n, "rejected": sum(reasons.values()), "reasons": dict(reasons)}


# --------------------------------------------------------------------------
# stage 3: convert + measure
# --------------------------------------------------------------------------
def stage_convert() -> dict[str, Any]:
    from .lengths import LengthCalculator, load_processor

    rows = {r["sft_id"]: r for r in _read_jsonl(C.SAMPLES_JSONL)}
    geometry = {g["sft_id"]: g for g in _read_jsonl(C.GEOMETRY_JSONL)}
    rejects: list[dict[str, Any]] = []
    staged: list[dict[str, Any]] = []
    closing_count = 0

    for sft_id, row in rows.items():
        geom = geometry.get(sft_id)
        if geom is None:
            rejects.append({"sft_id": sft_id, "build": row["build"], "stage": "convert",
                            "reason": "geometry_missing", "detail": ""})
            continue
        if geom.get("reason"):
            rejects.append({"sft_id": sft_id, "build": row["build"], "stage": "image",
                            "reason": geom["reason"], "detail": str(geom.get("detail", ""))})
            continue
        try:
            seg = convert(row["reasoning"])
        except ReasoningRejected as exc:
            rejects.append({"sft_id": sft_id, "build": row["build"], "stage": "reasoning",
                            "reason": exc.reason, "detail": exc.detail})
            continue
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            rejects.append({"sft_id": sft_id, "build": row["build"], "stage": "reasoning",
                            "reason": "instruction_missing", "detail": ""})
            continue
        leaked = contains_legacy_tag(seg.where + seg.color + instruction)
        if leaked:
            rejects.append({"sft_id": sft_id, "build": row["build"], "stage": "reasoning",
                            "reason": "legacy_tag_leaked", "detail": ",".join(leaked)})
            continue
        closing_count += int(seg.has_closing)
        staged.append({
            "sft_id": sft_id,
            "instruction": instruction.strip(),
            "where": seg.where,
            "color": seg.color,
            "has_closing": seg.has_closing,
            "vision_tokens": geom["vision_tokens"],
        })

    _log(f"convert: {len(staged)} converted, {len(rejects)} rejected; "
         f"{closing_count} carried an original closing text")
    processor, special_ids = load_processor(str(C.MODEL_DIR))
    calc = LengthCalculator(processor)
    _log(f"convert: tokenising ({calc.template_overhead} template tokens, "
         f"special ids {special_ids})")
    lengths = calc.measure(staged)

    by_id = {row["sft_id"]: row for row in staged}
    out: list[dict[str, Any]] = []
    for length in lengths:
        row = by_id[length["sft_id"]]
        if length["total_tokens"] > C.MODEL_MAX_LENGTH:
            rejects.append({"sft_id": row["sft_id"], "build": rows[row["sft_id"]]["build"],
                            "stage": "length", "reason": "sequence_too_long",
                            "detail": str(length["total_tokens"])})
            continue
        out.append({**row, **length})
    n = _write_jsonl(C.LENGTHS_JSONL, out)
    _write_jsonl(C.WORK_DIR / "rejections_convert.jsonl", rejects)
    reasons = Counter(r["reason"] for r in rejects)
    _log(f"convert: {n} survivors, rejections {dict(reasons)}")
    return {
        "survivors": n,
        "rejected": len(rejects),
        "reasons": dict(reasons),
        "with_closing_text": closing_count,
        "special_token_ids": special_ids,
        "template_overhead_tokens": calc.template_overhead,
    }


# --------------------------------------------------------------------------
# stage 4: split plan
# --------------------------------------------------------------------------
def stage_plan(drop_low_confidence: bool = False) -> dict[str, Any]:
    from .splits import audit, choose_lut_reserve, partition_eval, source_groups

    samples = {r["sft_id"]: r for r in _read_jsonl(C.SAMPLES_JSONL)}
    geometry = {g["sft_id"]: g for g in _read_jsonl(C.GEOMETRY_JSONL)}
    survivors = _read_jsonl(C.LENGTHS_JSONL)

    train_ids = _load_ids(C.TRAIN_IDS)
    eval_ids = _load_ids(C.EVAL_IDS)
    dedup_ids = _load_ids(C.DEDUP_DROP_IDS)

    rejects: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for row in survivors:
        sft_id = row["sft_id"]
        meta = samples[sft_id]
        geom = geometry[sft_id]
        if sft_id in dedup_ids:
            rejects.append({"sft_id": sft_id, "build": meta["build"], "stage": "split",
                            "reason": "dedup_drop", "detail": ""})
            continue
        if sft_id in train_ids:
            pool = "train"
        elif sft_id in eval_ids:
            pool = "eval"
        else:
            rejects.append({"sft_id": sft_id, "build": meta["build"], "stage": "split",
                            "reason": "split_inconsistent", "detail": "id in neither list"})
            continue
        if drop_low_confidence and meta.get("winner_confidence") == "low":
            rejects.append({"sft_id": sft_id, "build": meta["build"], "stage": "split",
                            "reason": "winner_confidence_low", "detail": ""})
            continue
        rows.append({**meta, **geom, **row, "pool": pool})

    groups = source_groups(rows)
    train_rows = [r for r in rows if r["pool"] == "train"]
    eval_rows = [r for r in rows if r["pool"] == "eval"]
    _log(f"plan: {len(train_rows)} train / {len(eval_rows)} eval after dedup + filters")

    reserve = choose_lut_reserve(train_rows, eval_rows)
    _log(f"plan: reserved {len(reserve.lut_ids)} LUTs -> {reserve.eval_gain} eval samples, "
         f"{reserve.train_cost} train samples removed ({100 * reserve.train_fraction:.2f}%)")

    plan = partition_eval(eval_rows, reserve.lut_ids, groups)
    for sft_id, reason in plan.unused.items():
        rejects.append({"sft_id": sft_id, "build": samples[sft_id]["build"], "stage": "split",
                        "reason": reason, "detail": ""})

    assignment = dict(plan.assignment)
    train_final: list[str] = []
    for row in train_rows:
        if row["lut_id"] in reserve.lut_ids:
            rejects.append({"sft_id": row["sft_id"], "build": row["build"], "stage": "split",
                            "reason": "lut_reserved_for_T_lut_unseen", "detail": row["lut_id"]})
            continue
        assignment[row["sft_id"]] = "train"
        train_final.append(row["sft_id"])

    rows_by_id = {r["sft_id"]: r for r in rows}
    audit_result = audit(rows_by_id, plan.assignment, set(train_final), groups)

    out_rows = []
    for sft_id, split in assignment.items():
        row = rows_by_id[sft_id]
        out_rows.append({
            "sft_id": sft_id,
            "split": split,
            "group": groups[sft_id],
            "build": row["build"],
            "build_id": row["build_id"],
            "batch": row["batch"],
            "sample_id": row["sample_id"],
            "task_type": row["task_type"],
            "source_id": row["source_id"],
            "group_id": row["group_id"],
            "candidate_id": row["candidate_id"],
            "lut_id": row["lut_id"],
            "preset_path": row.get("preset_path"),
            "major": row.get("major"),
            "minor": row.get("minor"),
            "mask_id": row.get("mask_id"),
            "region": row.get("region"),
            "render_mode": row.get("render_mode"),
            "winner_confidence": row.get("winner_confidence"),
            "winner_rank": row.get("winner_rank"),
            "i_in_path": row.get("i_in_path"),
            "instruction": row["instruction"],
            "where": row["where"],
            "color": row["color"],
            "has_closing": row["has_closing"],
            "image_src": row["image"],
            "raw_w": row["raw_w"], "raw_h": row["raw_h"],
            "format": row["format"], "exif_orientation": row["exif_orientation"],
            "oriented_w": row["oriented_w"], "oriented_h": row["oriented_h"],
            "out_w": row["out_w"], "out_h": row["out_h"],
            "grid_w": row["grid_w"], "grid_h": row["grid_h"],
            "vision_tokens": row["vision_tokens"],
            "aspect_in": row["aspect_in"], "aspect_out": row["aspect_out"],
            "upscaled": row["upscaled"],
            "prompt_tokens": row["prompt_tokens"],
            "instruction_tokens": row["instruction_tokens"],
            "where_tokens": row["where_tokens"],
            "color_tokens": row["color_tokens"],
            "total_tokens": row["total_tokens"],
        })
    out_rows.sort(key=lambda r: (r["build"], r["batch"], r["image_src"]["shard"],
                                 r["image_src"]["offset"]))
    n = _write_jsonl(C.PLAN_JSONL, out_rows)
    _write_jsonl(C.WORK_DIR / "rejections_plan.jsonl", rejects)

    sizes = Counter(r["split"] for r in out_rows)
    summary = {
        "n_effective_train": sizes["train"],
        "split_sizes": dict(sizes),
        "reserve": {
            "n_luts": len(reserve.lut_ids),
            "eval_gain": reserve.eval_gain,
            "train_removed": reserve.train_cost,
            "train_fraction": reserve.train_fraction,
            "budget_fraction": C.LUT_RESERVE_TRAIN_BUDGET,
            "per_major": reserve.per_major,
            "lut_ids": sorted(reserve.lut_ids),
        },
        "eval_group_counts": plan.group_counts,
        "unused_eval": len(plan.unused),
        "audit": audit_result,
        "rejected": len(rejects),
        "reasons": dict(Counter(r["reason"] for r in rejects)),
        "rows": n,
    }
    (C.WORK_DIR / "plan_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    _log(f"plan: {dict(sizes)}")
    return summary


# --------------------------------------------------------------------------
# stage 5: pack records
# --------------------------------------------------------------------------
def _record_payloads(rows: list[dict[str, Any]], image_index: dict[str, dict[str, Any]] | None):
    for row in rows:
        record = {
            "schema_version": C.RECORD_SCHEMA,
            "sft_id": row["sft_id"],
            "sample_id": row["sft_id"],
            "split": row["split"],
            "group": row["group"],
            "build": row["build"],
            "build_id": row["build_id"],
            "batch": row["batch"],
            "source_sample_id": row["sample_id"],
            "task_type": row["task_type"],
            "source_image_id": row["source_id"],
            "source_group_id": row["group_id"],
            "candidate_id": row["candidate_id"],
            "lut_id": row["lut_id"],
            "preset_path": row["preset_path"],
            "major": row["major"],
            "minor": row["minor"],
            "mask_id": row["mask_id"],
            "region": row["region"],
            "render_mode": row["render_mode"],
            "winner_confidence": row["winner_confidence"],
            "winner_rank": row["winner_rank"],
            "instruction": row["instruction"],
            "where": row["where"],
            "color": row["color"],
            "has_closing_text": row["has_closing"],
            "image": {
                "origin": {"i_in_path": row["i_in_path"], **row["image_src"]},
                "raw_w": row["raw_w"], "raw_h": row["raw_h"],
                "format": row["format"],
                "exif_orientation": row["exif_orientation"],
                "oriented_w": row["oriented_w"], "oriented_h": row["oriented_h"],
                "out_w": row["out_w"], "out_h": row["out_h"],
                "grid_w": row["grid_w"], "grid_h": row["grid_h"],
                "vision_tokens": row["vision_tokens"],
                "aspect_in": row["aspect_in"], "aspect_out": row["aspect_out"],
                "upscaled": row["upscaled"],
                "baked": (image_index or {}).get(row["sft_id"]),
            },
            "tokens": {
                "prompt": row["prompt_tokens"],
                "instruction": row["instruction_tokens"],
                "where": row["where_tokens"],
                "color": row["color_tokens"],
                "vision": row["vision_tokens"],
                "total": row["total_tokens"],
            },
        }
        blob = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")
        yield f"sft2seg/records/{row['sft_id']}.rec.json", blob


def stage_records() -> dict[str, Any]:
    from .shardio import build_from_memory, iter_index

    rows = _read_jsonl(C.PLAN_JSONL)
    image_index: dict[str, dict[str, Any]] | None = None
    if (C.IMAGES_DIR / "manifest.json").is_file():
        image_index = {}
        for entry in iter_index(C.IMAGES_DIR):
            image_index[entry["sample_id"]] = {
                "shard": str(C.IMAGES_DIR / "shards" / f"{entry['shard']}.tar"),
                "member": entry["member"],
                "offset": entry["offset_data"],
                "length": entry["length"],
                "size": entry["size"],
                "sha256": entry["sha256"],
            }
        _log(f"records: linking {len(image_index)} baked images")
    manifest = build_from_memory(
        _record_payloads(rows, image_index),
        C.RECORDS_DIR,
        shard_size_bytes=C.RECORD_SHARD_BYTES,
        producer="q3vl.data.pipeline.stage_records",
        source_label=str(C.PLAN_JSONL),
        progress=lambda n, b: _log(f"records: {n} members, {b / 1e6:.0f} MB"),
    )
    _log(f"records: {manifest['sample_count']} samples in {manifest['shard_count']} shards")
    return manifest


# --------------------------------------------------------------------------
# stage 6: pack spec-5 images
# --------------------------------------------------------------------------
def stage_images(workers: int = 24) -> dict[str, Any]:
    from .bake import bake_payloads
    from .shardio import build_from_memory

    rows = _read_jsonl(C.PLAN_JSONL)
    _log(f"images: baking {len(rows)} images with {workers} workers")
    manifest = build_from_memory(
        bake_payloads(rows, workers=workers, log=_log),
        C.IMAGES_DIR,
        shard_size_bytes=C.IMAGE_SHARD_BYTES,
        producer="q3vl.data.pipeline.stage_images",
        source_label=str(C.PLAN_JSONL),
        progress=lambda n, b: _log(f"images: {n} members, {b / 1e9:.2f} GB"),
        progress_every=5000,
    )
    _log(f"images: {manifest['sample_count']} samples in {manifest['shard_count']} shards, "
         f"{manifest['payload_bytes'] / 1e9:.1f} GB")
    return manifest


# --------------------------------------------------------------------------
# stage 7: terminal manifest + per-split indexes
# --------------------------------------------------------------------------
def stage_manifest() -> dict[str, Any]:
    from .shardio import iter_index

    rows = _read_jsonl(C.PLAN_JSONL)
    plan_summary = json.loads((C.WORK_DIR / "plan_summary.json").read_text())

    record_members = {e["sample_id"]: e for e in iter_index(C.RECORDS_DIR)}
    image_members: dict[str, dict[str, Any]] = {}
    if (C.IMAGES_DIR / "manifest.json").is_file():
        image_members = {e["sample_id"]: e for e in iter_index(C.IMAGES_DIR)}

    C.SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[row["split"]].append(row)

    split_files: dict[str, Any] = {}
    for split, split_rows in sorted(by_split.items()):
        ids_path = C.SPLIT_DIR / f"{split}_sft_ids.txt"
        ids_path.write_text("".join(f"{r['sft_id']}\n" for r in sorted(
            split_rows, key=lambda r: r["sft_id"])))
        index_rows = []
        for row in split_rows:
            rec = record_members[row["sft_id"]]
            members = {"record": {
                "shard": str(C.RECORDS_DIR / "shards" / f"{rec['shard']}.tar"),
                "member": rec["member"], "offset": rec["offset_data"],
                "length": rec["length"], "size": rec["size"], "sha256": rec["sha256"],
            }}
            img = image_members.get(row["sft_id"])
            if img is not None:
                members["image"] = {
                    "shard": str(C.IMAGES_DIR / "shards" / f"{img['shard']}.tar"),
                    "member": img["member"], "offset": img["offset_data"],
                    "length": img["length"], "size": img["size"], "sha256": img["sha256"],
                }
            index_rows.append({
                "sample_id": row["sft_id"], "split": split, "members": members,
                "build": row["build"], "task_type": row["task_type"],
                "source_image_id": row["source_id"], "lut_id": row["lut_id"],
                "winner_confidence": row["winner_confidence"],
                "n_visual_tokens": row["vision_tokens"],
                "total_tokens": row["total_tokens"],
            })
        index_rows.sort(key=lambda r: r["sample_id"])
        index_path = C.SPLIT_DIR / f"{split}.index.jsonl"
        _write_jsonl(index_path, index_rows)
        split_files[split] = {
            "ids": str(ids_path), "ids_sha256": sha256_file(ids_path),
            "index": str(index_path), "index_sha256": sha256_file(index_path),
            "n": len(split_rows),
        }

    def stats(key: str, subset: list[dict[str, Any]]) -> dict[str, Any]:
        values = sorted(r[key] for r in subset)
        if not values:
            return {}
        return {
            "min": values[0], "p50": values[len(values) // 2],
            "p95": values[int(0.95 * (len(values) - 1))], "max": values[-1],
            "mean": round(sum(values) / len(values), 2),
        }

    counts = {split: {"n_effective": len(rs), "sample_count": len(rs)}
              for split, rs in by_split.items()}
    manifest: dict[str, Any] = {
        "schema_version": C.SPLIT_SCHEMA,
        "status": "complete",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "producer": "q3vl.data.pipeline.stage_manifest",
        "model_dir": str(C.MODEL_DIR),
        "model_max_length": C.MODEL_MAX_LENGTH,
        "source_builds": C.BUILDS,
        "split_authority": {
            "train": str(C.TRAIN_IDS), "eval": str(C.EVAL_IDS),
            "dedup_drop": str(C.DEDUP_DROP_IDS),
            "train_sha256": sha256_file(C.TRAIN_IDS),
            "eval_sha256": sha256_file(C.EVAL_IDS),
            "dedup_drop_sha256": sha256_file(C.DEDUP_DROP_IDS),
        },
        "counts": counts,
        "n_effective": len(by_split.get("train", [])),
        "datasets": {
            "records": _dataset_ref(C.RECORDS_DIR),
            "images": _dataset_ref(C.IMAGES_DIR) if image_members else None,
        },
        "splits": split_files,
        "reserve": plan_summary["reserve"],
        "audit": plan_summary["audit"],
        "eval_group_counts": plan_summary["eval_group_counts"],
        "distributions": {
            split: {
                "total_tokens": stats("total_tokens", rs),
                "vision_tokens": stats("vision_tokens", rs),
                "out_w": stats("out_w", rs), "out_h": stats("out_h", rs),
            } for split, rs in sorted(by_split.items())
        },
    }
    digest_material = json.dumps(
        {k: v for k, v in manifest.items() if k not in ("created_at",)},
        ensure_ascii=False, sort_keys=True).encode("utf-8")
    manifest["digest"] = hashlib.sha256(digest_material).hexdigest()

    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = C.MANIFEST_DIR / "terminal_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    _log(f"manifest: N_effective={manifest['n_effective']} digest={manifest['digest'][:16]}")
    return manifest


def _dataset_ref(root: Path) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        return None
    raw = json.loads(manifest_path.read_text())
    return {
        "root": str(root),
        "dataset_id": raw["dataset_id"],
        "schema_version": raw["schema_version"],
        "shard_count": raw["shard_count"],
        "member_count": raw["member_count"],
        "sample_count": raw["sample_count"],
        "payload_bytes": raw["payload_bytes"],
        "manifest_sha256": sha256_file(manifest_path),
        "shards": [{"shard_id": s["shard_id"], "tar_sha256": s["tar_sha256"],
                    "index_sha256": s["index_sha256"], "tar_bytes": s["tar_bytes"],
                    "member_count": s["member_count"]} for s in raw["shards"]],
    }
