"""Re-runnable audits for the published sft2seg dataset (SFT spec 9, items 5-9).

Nothing here trusts the build-time bookkeeping: every number is re-derived from
the published artefacts.  The canonical ``verify_dataset`` from
``dataset_build.tools.indexed_tar`` re-walks every tar and index; on top of it
this module does random positional reads, resume simulation, the four-way split
isolation audit and the sampled seven-vs-two segment comparison.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from dataset_build.tools.indexed_tar import verify_dataset

from . import config as C
from .shardio import iter_index as _iter_index, read_member
from .twoseg import COLOR_FIELDS, WHERE_FIELD, contains_legacy_tag, convert


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check_indexed_datasets() -> dict[str, Any]:
    """Full walk of both published datasets by the canonical verifier."""
    out: dict[str, Any] = {}
    for name, root in (("records", C.RECORDS_DIR), ("images", C.IMAGES_DIR)):
        if not (root / "manifest.json").is_file():
            out[name] = {"status": "absent"}
            continue
        out[name] = {"status": "ok", **verify_dataset(root)}
    return out


def check_random_access(n: int = 400, seed: int = 0) -> dict[str, Any]:
    """Random positional reads with checksum verification, both datasets."""
    rng = random.Random(seed)
    result: dict[str, Any] = {}
    for name, root in (("records", C.RECORDS_DIR), ("images", C.IMAGES_DIR)):
        index_path = C.SPLIT_DIR
        if not (root / "manifest.json").is_file():
            result[name] = {"status": "absent"}
            continue
        rows: list[dict[str, Any]] = []
        for path in sorted((root / "indexes").glob("shard-*.idx.jsonl")):
            rows.extend(json.loads(line) for line in path.open() if line.strip())
        picked = rng.sample(rows, min(n, len(rows)))
        checked = 0
        for row in picked:
            data = read_member(root, row["shard"], row["offset_data"],
                               row["length"], row["sha256"])
            if name == "records":
                record = json.loads(data)
                assert record["sft_id"] == row["sample_id"], row["sample_id"]
            checked += 1
        result[name] = {"status": "ok", "checked": checked, "pool": len(rows),
                        "index_dir": str(index_path)}
    return result


def check_resume() -> dict[str, Any]:
    """Resume authority: manifest + shard index only, never directory mtime.

    Simulated by re-deriving the sample inventory from the indexes alone and
    comparing it with the terminal manifest's per-split id lists.
    """
    manifest = json.loads((C.MANIFEST_DIR / "terminal_manifest.json").read_text())
    from_index: set[str] = set()
    for path in sorted((C.RECORDS_DIR / "indexes").glob("shard-*.idx.jsonl")):
        for line in path.open():
            if line.strip():
                from_index.add(json.loads(line)["sample_id"])
    from_splits: set[str] = set()
    per_split: dict[str, int] = {}
    for split, info in manifest["splits"].items():
        ids = {line.strip() for line in Path(info["ids"]).read_text().splitlines() if line.strip()}
        per_split[split] = len(ids)
        if from_splits & ids:
            raise AssertionError(f"{split} overlaps a previous split id list")
        from_splits |= ids
    return {
        "index_samples": len(from_index),
        "split_samples": len(from_splits),
        "equal": from_index == from_splits,
        "per_split": per_split,
        "manifest_digest": manifest["digest"],
    }


def check_split_disjointness() -> dict[str, Any]:
    """train / eval / dedup authority vs the published splits."""
    train = {l.strip() for l in C.TRAIN_IDS.read_text().splitlines() if l.strip()}
    evals = {l.strip() for l in C.EVAL_IDS.read_text().splitlines() if l.strip()}
    dedup = {l.strip() for l in C.DEDUP_DROP_IDS.read_text().splitlines() if l.strip()}
    rows = _read_jsonl(C.PLAN_JSONL)
    published: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        published[row["split"]].add(row["sft_id"])
    all_published = set().union(*published.values())
    return {
        "authority_raw": {
            "train": len(train), "eval": len(evals), "dedup_drop": len(dedup),
            "train_x_eval": len(train & evals),
            "train_x_dedup": len(train & dedup),
            "eval_x_dedup": len(evals & dedup),
        },
        "authority_after_dedup_applied": {
            "train": len(train - dedup), "eval": len(evals - dedup),
            "train_x_eval": len((train - dedup) & (evals - dedup)),
            "train_x_dedup": len((train - dedup) & dedup),
            "eval_x_dedup": len((evals - dedup) & dedup),
        },
        "published": {k: len(v) for k, v in sorted(published.items())},
        "published_x_dedup": len(all_published & dedup),
        "published_train_x_authority_eval": len(published["train"] & evals),
        "published_eval_x_authority_train": len(
            (all_published - published["train"]) & train),
    }


def check_lut_isolation() -> dict[str, Any]:
    rows = _read_jsonl(C.PLAN_JSONL)
    by_split: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_split[row["split"]].add(row["lut_id"])
    unseen = by_split["T_lut_unseen"]
    return {
        "n_train_luts": len(by_split["train"]),
        "n_unseen_luts": len(unseen),
        "unseen_x_train": len(unseen & by_split["train"]),
        "unseen_x_V_where": len(unseen & by_split["V_where"]),
        "unseen_x_V_what": len(unseen & by_split["V_what"]),
        "unseen_x_T_final": len(unseen & by_split["T_final"]),
    }


def check_image_contract() -> dict[str, Any]:
    """Short side / long side / aspect / 32-alignment / token counts over the plan."""
    rows = _read_jsonl(C.PLAN_JSONL)
    violations = Counter()
    short_sides, long_sides, tokens, aspects, deltas = [], [], [], [], []
    for row in rows:
        w, h = row["out_w"], row["out_h"]
        if min(w, h) != C.IMAGE_SHORT_SIDE:
            violations["short_side"] += 1
        if max(w, h) > C.IMAGE_LONG_SIDE_MAX:
            violations["long_side"] += 1
        if w % C.IMAGE_ALIGN_FACTOR or h % C.IMAGE_ALIGN_FACTOR:
            violations["alignment"] += 1
        if row["aspect_in"] > C.IMAGE_MAX_ASPECT_RATIO:
            violations["aspect_in"] += 1
        if (w // C.IMAGE_ALIGN_FACTOR) * (h // C.IMAGE_ALIGN_FACTOR) != row["vision_tokens"]:
            violations["vision_tokens"] += 1
        short_sides.append(min(row["oriented_w"], row["oriented_h"]))
        long_sides.append(max(w, h))
        tokens.append(row["vision_tokens"])
        aspects.append(row["aspect_in"])
        deltas.append(abs(row["aspect_out"] - row["aspect_in"]) / row["aspect_in"])

    def dist(values: list[float]) -> dict[str, Any]:
        values = sorted(values)
        n = len(values)
        return {
            "n": n, "min": values[0], "p05": values[int(0.05 * (n - 1))],
            "p50": values[n // 2], "p95": values[int(0.95 * (n - 1))], "max": values[-1],
            "mean": round(sum(values) / n, 4),
        }

    return {
        "violations": dict(violations),
        "orig_short_side": dist(short_sides),
        "out_long_side": dist(long_sides),
        "vision_tokens": dist([float(t) for t in tokens]),
        "vision_token_hist": dict(sorted(Counter(tokens).items())),
        "aspect_in": dist(aspects),
        "aspect_rounding_error": dist(deltas),
        "upscaled": sum(1 for r in rows if r["upscaled"]),
        "exif_orientation_hist": dict(sorted(Counter(
            r["exif_orientation"] for r in rows).items())),
        "format_hist": dict(sorted(Counter(r["format"] for r in rows).items())),
    }


def check_length_contract() -> dict[str, Any]:
    rows = _read_jsonl(C.PLAN_JSONL)
    totals = sorted(r["total_tokens"] for r in rows)
    over = sum(1 for t in totals if t > C.MODEL_MAX_LENGTH)
    rejections = []
    for name in ("rejections_convert.jsonl",):
        path = C.WORK_DIR / name
        if path.is_file():
            rejections.extend(_read_jsonl(path))
    n = len(totals)
    return {
        "n": n,
        "max": totals[-1],
        "p50": totals[n // 2],
        "p95": totals[int(0.95 * (n - 1))],
        "p99": totals[int(0.99 * (n - 1))],
        "over_limit_remaining": over,
        "filtered_too_long": sum(1 for r in rejections if r["reason"] == "sequence_too_long"),
        "histogram_256": dict(sorted(Counter(t // 256 * 256 for t in totals).items())),
    }


def sample_conversion_audit(n: int = 24, seed: int = 7) -> list[dict[str, Any]]:
    """Word-for-word comparison of the seven original bodies against the two new ones."""
    from .scan import scan_text

    rows = _read_jsonl(C.PLAN_JSONL)
    rng = random.Random(seed)
    picked = rng.sample(rows, n)
    by_build: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in picked:
        by_build[row["build_id"]].append(row)

    out: list[dict[str, Any]] = []
    for build_id, subset in by_build.items():
        text = scan_text(build_id)
        for row in subset:
            source = text[row["sft_id"]]
            seg = convert(source["reasoning"])
            record = json.loads(read_member(
                C.RECORDS_DIR, *_locate(row["sft_id"])))
            expected_parts = [seg.fields[f] for f in COLOR_FIELDS]
            if seg.closing:
                expected_parts.append(seg.closing)
            checks = {
                "where_equals_region_scope": record["where"] == seg.fields[WHERE_FIELD],
                "color_is_six_bodies_in_order": record["color"] == "\n".join(expected_parts),
                "region_scope_not_in_color": all(
                    part != seg.fields[WHERE_FIELD] for part in expected_parts),
                "no_legacy_tag": not contains_legacy_tag(record["where"] + record["color"]),
                "no_invented_closing": bool(seg.closing) == record["has_closing_text"],
                "instruction_verbatim": record["instruction"] == source["instruction"].strip(),
            }
            out.append({
                "sft_id": row["sft_id"], "build": row["build"],
                "checks": checks, "all_pass": all(checks.values()),
                "original_fields": {f: seg.fields[f] for f in seg.fields},
                "new_where": record["where"], "new_color": record["color"],
            })
    return out


_LOCATOR_CACHE: dict[str, tuple[str, int, int, str]] = {}


def _locate(sft_id: str) -> tuple[str, int, int, str]:
    if not _LOCATOR_CACHE:
        for path in sorted((C.RECORDS_DIR / "indexes").glob("shard-*.idx.jsonl")):
            for line in path.open():
                if line.strip():
                    row = json.loads(line)
                    _LOCATOR_CACHE[row["sample_id"]] = (
                        row["shard"], row["offset_data"], row["length"], row["sha256"])
    return _LOCATOR_CACHE[sft_id]


def check_bake_fidelity(n: int = 24, seed: int = 5) -> dict[str, Any]:
    """The baked image must equal what the trainer would compute from the original.

    This is the evidence for NOTES.md decision D-1 (store contract-sized copies
    instead of the raw bytes).  For each sample the original member is re-read
    from its production build, pushed through ``q3vl.train.imageproc.prepare_image``
    and compared pixel-by-pixel with the decoded baked JPEG.  Sizes must match
    exactly; the only permitted difference is JPEG quantisation.
    """
    import io

    import numpy as np
    from PIL import Image

    from q3vl.train.imageproc import prepare_image

    if not (C.IMAGES_DIR / "manifest.json").is_file():
        return {"status": "absent"}
    baked = {e["sample_id"]: e for e in _iter_index(C.IMAGES_DIR)}
    rows = [r for r in _read_jsonl(C.PLAN_JSONL) if r["sft_id"] in baked]
    picks = random.Random(seed).sample(rows, min(n, len(rows)))

    size_mismatch = 0
    rmse: list[float] = []
    max_abs: list[int] = []
    for row in picks:
        src = row["image_src"]
        raw = read_member(Path(src["root"]), src["shard"], src["offset"],
                          src["length"], src["sha256"])
        reference, _geom = prepare_image(raw)
        entry = baked[row["sft_id"]]
        blob = read_member(C.IMAGES_DIR, entry["shard"], entry["offset_data"],
                           entry["length"], entry["sha256"])
        stored = Image.open(io.BytesIO(blob))
        stored.load()
        if stored.size != reference.size or stored.size != (row["out_w"], row["out_h"]):
            size_mismatch += 1
            continue
        a = np.asarray(reference, dtype=np.int16)
        b = np.asarray(stored.convert("RGB"), dtype=np.int16)
        diff = a - b
        rmse.append(float(np.sqrt((diff.astype(np.float64) ** 2).mean())))
        max_abs.append(int(np.abs(diff).max()))
    return {
        "status": "ok",
        "n": len(picks),
        "size_mismatch": size_mismatch,
        "rmse_mean": round(sum(rmse) / len(rmse), 4) if rmse else None,
        "rmse_max": round(max(rmse), 4) if rmse else None,
        "max_abs_diff": max(max_abs) if max_abs else None,
        "note": "difference is JPEG q95 4:4:4 quantisation only; geometry is exact",
        "all_pass": size_mismatch == 0 and bool(rmse) and max(rmse) < 3.0,
    }


def check_length_identity(n: int = 16, seed: int = 11) -> dict[str, Any]:
    """The stored ``total_tokens`` must equal what the trainer's collator builds.

    Two independent things are asserted: the arithmetic shortcut this pipeline
    uses (tokenise the prompt with one ``<|image_pad|>``, add the visual count)
    equals the fully expanded tokenisation, and the resulting number equals
    ``Sft2SegCollator.encode_one``'s sequence length on the same sample.
    """
    from types import SimpleNamespace

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.imageproc import plan_geometry

    from .lengths import LengthCalculator, load_processor

    rows = _read_jsonl(C.PLAN_JSONL)
    picks = random.Random(seed).sample(rows, min(n, len(rows)))
    processor, special_ids = load_processor(str(C.MODEL_DIR))
    calc = LengthCalculator(processor)
    collator = Sft2SegCollator(processor)

    shortcut_ok = concat_ok = collator_ok = 0
    failures: list[dict[str, Any]] = []
    for row in picks:
        identity = calc.verify_identity(row)
        shortcut_ok += identity["prompt_equal"]
        concat_ok += identity["concat_equal"]
        geom = plan_geometry(row["oriented_h"], row["oriented_w"])
        sample = SimpleNamespace(
            sample_id=row["sft_id"], geometry=geom, instruction=row["instruction"],
            where_text=row["where"], color_text=row["color"])
        length = len(collator.encode_one(sample)["input_ids"])
        collator_ok += length == row["total_tokens"]
        if not (identity["prompt_equal"] and identity["concat_equal"]
                and length == row["total_tokens"]):
            failures.append({**identity, "collator_len": length,
                             "stored": row["total_tokens"]})
    return {
        "n": len(picks),
        "placeholder_shortcut_exact": shortcut_ok,
        "piecewise_equals_whole_string": concat_ok,
        "equals_training_collator": collator_ok,
        "all_pass": len(failures) == 0,
        "failures": failures,
        "special_token_ids": special_ids,
        "template_overhead_tokens": calc.template_overhead,
    }


def check_training_loader(n: int = 12, seed: int = 3) -> dict[str, Any]:
    """Pull real samples through the *training side's* reader, not ours.

    This is the only check that proves the published index/shard layout is the
    one ``q3vl.train`` can actually consume; everything else in this module
    verifies the producer against itself.
    """
    from q3vl.train.dataset import Sft2SegDataset
    from q3vl.train.imageproc import plan_geometry
    from q3vl.train.shards import ShardIndex, ShardStore

    out: dict[str, Any] = {}
    rng = random.Random(seed)
    for split in ("train", "V_where", "V_what", "T_final", "T_lut_unseen"):
        index_path = C.SPLIT_DIR / f"{split}.index.jsonl"
        if not index_path.is_file():
            out[split] = {"status": "absent"}
            continue
        index = ShardIndex.load(index_path)
        store = ShardStore(shard_root="/", verify="checksum")
        dataset = Sft2SegDataset(index, store)
        picks = rng.sample(range(len(index)), min(n, len(index)))
        checked = []
        for i in picks:
            sample = dataset[i]
            record = json.loads(store.read(index[i].members["record"]))
            geom = plan_geometry(record["image"]["oriented_h"], record["image"]["oriented_w"])
            checked.append({
                "sample_id": sample.sample_id,
                "size_matches_plan": sample.image.size == (record["image"]["out_w"],
                                                           record["image"]["out_h"]),
                "geometry_reproduces": (geom.out_w, geom.out_h, geom.n_visual_tokens) == (
                    record["image"]["out_w"], record["image"]["out_h"],
                    record["image"]["vision_tokens"]),
                "where_matches_record": sample.where_text == record["where"],
                "color_matches_record": sample.color_text == record["color"],
                "no_legacy_tag": not contains_legacy_tag(
                    sample.where_text + sample.color_text),
            })
        out[split] = {
            "status": "ok",
            "layout": index.layout,
            "n_index": len(index),
            "n_checked": len(checked),
            "all_pass": all(all(v for k, v in c.items() if k != "sample_id")
                            for c in checked),
            "checksum_verified_reads": store.n_checksum_verified,
            "examples": checked[:3],
        }
    return out


def rejection_report() -> dict[str, Any]:
    """Every dropped sample, grouped by stage and reason."""
    grouped: dict[str, Counter] = defaultdict(Counter)
    total = 0
    for path in sorted(C.WORK_DIR.glob("rejections_*.jsonl")):
        for row in _read_jsonl(path):
            grouped[row.get("stage", path.stem)][row["reason"]] += 1
            total += 1
    return {"total": total,
            "by_stage": {k: dict(sorted(v.items())) for k, v in sorted(grouped.items())}}


def run_all(quick: bool = False) -> dict[str, Any]:
    report: dict[str, Any] = {
        "split_disjointness": check_split_disjointness(),
        "lut_isolation": check_lut_isolation(),
        "image_contract": check_image_contract(),
        "length_contract": check_length_contract(),
        "rejections": rejection_report(),
        "random_access": check_random_access(),
        "resume": check_resume(),
        "bake_fidelity": check_bake_fidelity(),
        "length_identity": check_length_identity(),
        "training_loader": check_training_loader(),
    }
    if not quick:
        report["indexed_datasets"] = check_indexed_datasets()
    report["conversion_samples"] = sample_conversion_audit()
    report["conversion_samples_all_pass"] = all(
        s["all_pass"] for s in report["conversion_samples"])
    return report
