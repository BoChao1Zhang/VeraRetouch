"""Evaluate VLM selection of one SAM3 subject instance without production writes.

The current SAM3 cache contains concept unions, so it cannot answer whether a
VLM can reliably choose one visual-center instance. This tool reruns SAM3 for a
small, stratified audit sample, preserves every proposal, renders numbered
overlays, and asks the VLM to select twice after deterministically renumbering
the same proposals.

Outputs are self-contained under ``--out``. The source-QA database and the
production SAM3 cache are read-only.

Typical container invocation::

    python -m dataset_build.tools.eval_subject_instance_selector \
        --audit-csv _mask_review/subject_audit_v1/audit.csv \
        --out _mask_review/instance_selector_v1 --n 32
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont, ImageOps

from dataset_build.core.responses_vlm import ResponsesVlmError, request_text


PALETTE = (
    (255, 74, 74),
    (66, 180, 255),
    (255, 202, 58),
    (92, 214, 118),
    (218, 112, 255),
    (255, 137, 61),
    (66, 224, 205),
    (245, 105, 173),
    (164, 205, 57),
    (129, 140, 248),
    (255, 216, 128),
    (93, 211, 243),
    (248, 113, 113),
    (74, 222, 128),
    (192, 132, 252),
    (251, 146, 60),
)

SUBJECT_LABEL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "has_localizable_subject", "subject_scope", "sam_prompt", "expected_count",
        "description", "visual_center", "confidence", "reason_code",
    ],
    "properties": {
        "has_localizable_subject": {"type": "boolean"},
        "subject_scope": {"type": "string", "enum": ["single", "group"]},
        "sam_prompt": {"type": "string"},
        "expected_count": {"type": "integer", "minimum": 1, "maximum": 16},
        "description": {"type": "string"},
        "visual_center": {
            "anyOf": [
                {
                    "type": "array", "minItems": 2, "maxItems": 2,
                    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                {"type": "null"},
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "reason_code": {
            "type": "string",
            "enum": ["clear_primary", "clear_group", "no_discrete_subject"],
        },
    },
}

SUBJECT_SELECTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["decision", "instance_ids", "confidence", "subject", "reason_code"],
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["select", "no_subject", "no_valid_mask", "ambiguous"],
        },
        "instance_ids": {
            "type": "array", "maxItems": 16, "uniqueItems": True,
            "items": {"type": "integer", "minimum": 1},
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "subject": {"type": "string"},
        "reason_code": {
            "type": "string",
            "enum": ["clear_primary", "clear_group", "pure_landscape", "proposal_miss", "tie"],
        },
    },
}


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sample_rows(path: Path, n: int, seed: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    rows = [
        row for row in rows
        if os.path.exists(row["source_path"]) and row.get("main_subject")
    ]
    buckets: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[row.get("stratum") or "unknown"].append(row)
    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    names = sorted(buckets)
    selected: list[dict[str, str]] = []
    while len(selected) < min(n, len(rows)):
        progressed = False
        for name in names:
            if buckets[name] and len(selected) < n:
                selected.append(buckets[name].pop())
                progressed = True
        if not progressed:
            break
    rng.shuffle(selected)
    return selected


def _mask_hash(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask, axis=None).tobytes()).hexdigest()


def _geometry(mask: np.ndarray) -> dict[str, Any]:
    ys, xs = np.nonzero(mask)
    height, width = mask.shape
    return {
        "area": float(mask.mean()),
        "bbox": [
            float(xs.min() / width),
            float(ys.min() / height),
            float((xs.max() + 1) / width),
            float((ys.max() + 1) / height),
        ],
        "centroid": [float(xs.mean() / width), float(ys.mean() / height)],
    }


def _iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if not union:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def _containment(left: np.ndarray, right: np.ndarray) -> float:
    smaller = min(int(left.sum()), int(right.sum()))
    if not smaller:
        return 0.0
    return float(np.logical_and(left, right).sum() / smaller)


def _save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(
        tmp, format="PNG", compress_level=1
    )
    os.replace(tmp, path)


def _sam_proposals(
    masker: Any,
    row: dict[str, str],
    out_dir: Path,
    min_score: float,
    dedupe_iou: float,
) -> dict[str, Any]:
    torch = masker._torch
    image, native = masker._as_pil(row["source_path"])
    height, width = native
    inputs = masker._proc(
        images=image, text=row["main_subject"], return_tensors="pt"
    ).to(masker._det.device)
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = masker._det(**inputs)
    results = masker._proc.post_process_instance_segmentation(
        outputs,
        threshold=float(min_score),
        mask_threshold=masker.mask_threshold,
        target_sizes=[(height, width)],
    )
    elapsed = time.perf_counter() - started
    return _proposals_from_result(
        torch, results[0], row, (width, height), elapsed, out_dir,
        min_score, dedupe_iou)


def _proposals_from_result(
    torch: Any,
    result: dict[str, Any],
    row: dict[str, str],
    size_wh: tuple[int, int],
    elapsed: float,
    out_dir: Path,
    min_score: float,
    dedupe_iou: float,
) -> dict[str, Any]:
    """Per-image post-processing shared by single and batched SAM3 forwards."""
    width, height = size_wh
    masks = result.get("masks")
    scores = result.get("scores")
    if masks is None or len(masks) == 0:
        return {
            "asset_id": row["asset_id"],
            "source_path": row["source_path"],
            "legacy_union_path": row.get("mask_path") or "",
            "stratum": row.get("stratum") or "",
            "scene": row.get("scene") or "",
            "main_subject": row["main_subject"],
            "native_size": [width, height],
            "sam_seconds": elapsed,
            "proposals": [],
            "status": "sam_miss",
        }

    mask_array = masks.detach().to(torch.bool).cpu().numpy()
    if mask_array.ndim == 2:
        mask_array = mask_array[None]
    if scores is None:
        score_array = np.ones(len(mask_array), dtype=np.float32)
    else:
        score_array = scores.detach().to(torch.float32).cpu().numpy()

    raw: list[tuple[float, np.ndarray, str, dict[str, Any]]] = []
    for score, mask in zip(score_array.tolist(), mask_array):
        hard = np.asarray(mask, dtype=bool)
        if score < min_score or not hard.any():
            continue
        digest = _mask_hash(hard)
        raw.append((float(score), hard, digest, _geometry(hard)))
    raw.sort(key=lambda item: (-item[0], item[3]["area"], item[2]))

    kept: list[tuple[float, np.ndarray, str, dict[str, Any]]] = []
    for candidate in raw:
        if any(
            _iou(candidate[1], previous[1]) >= dedupe_iou
            or _containment(candidate[1], previous[1]) >= 0.98
            for previous in kept
        ):
            continue
        kept.append(candidate)
    kept.sort(
        key=lambda item: (
            -item[0],
            item[3]["bbox"][1],
            item[3]["bbox"][0],
            item[3]["area"],
            item[2],
        )
    )

    proposals: list[dict[str, Any]] = []
    instance_dir = out_dir / "instances" / row["asset_id"]
    for stable_id, (score, mask, digest, geometry) in enumerate(kept, 1):
        path = instance_dir / f"{stable_id:03d}.png"
        _save_mask(mask, path)
        proposals.append({
            "stable_id": stable_id,
            "score": score,
            "mask_sha256": digest,
            "mask_path": str(path),
            **geometry,
        })
    return {
        "asset_id": row["asset_id"],
        "source_path": row["source_path"],
        "legacy_union_path": row.get("mask_path") or "",
        "stratum": row.get("stratum") or "",
        "scene": row.get("scene") or "",
        "main_subject": row["main_subject"],
        "native_size": [width, height],
        "sam_seconds": elapsed,
        "raw_proposal_count": len(raw),
        "proposals": proposals,
        "status": "ready" if proposals else "sam_miss",
    }


def _dense_group_union(record: dict[str, Any], out_dir: Path) -> None:
    """Dense group (e.g. a flower field): instance-level selection is the wrong
    granularity, so collapse every proposal into one union candidate and let
    the selector verify or reject the cluster as a whole."""
    union = None
    for proposal in record["proposals"]:
        with Image.open(proposal["mask_path"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8) >= 128
        union = mask if union is None else (union | mask)
    if union is None or not union.any():
        record["status"] = "too_many_instances"
        return
    path = Path(out_dir) / "instances" / record["asset_id"] / "union.png"
    _save_mask(union, path)
    record["dense_union"] = True
    record["dense_union_members"] = len(record["proposals"])
    record["proposals"] = [{
        "stable_id": 1,
        "score": max(float(p["score"]) for p in record["proposals"]),
        "mask_sha256": _mask_hash(union),
        "mask_path": str(path),
        **_geometry(union),
    }]


def _fit(image: Image.Image, size: tuple[int, int], color=(18, 18, 18)) -> Image.Image:
    canvas = Image.new("RGB", size, color)
    fitted = ImageOps.contain(image.convert("RGB"), size, Image.Resampling.LANCZOS)
    canvas.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
    return canvas


def _preview_source(path: str, long_edge: int = 960) -> Image.Image:
    with Image.open(path) as image:
        source = image.convert("RGB")
        source.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
    return source


def _load_preview_mask(path: str, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("L").resize(size, Image.Resampling.NEAREST)


def _edge(mask: Image.Image) -> Image.Image:
    expanded = mask.filter(ImageFilter.MaxFilter(5))
    contracted = mask.filter(ImageFilter.MinFilter(5))
    return ImageChops.difference(expanded, contracted).point(lambda value: 255 if value else 0)


def _draw_number(draw: ImageDraw.ImageDraw, xy: tuple[int, int], number: int) -> None:
    label = str(number)
    font = _font(26)
    box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    width = box[2] - box[0] + 14
    height = box[3] - box[1] + 10
    x = max(0, min(xy[0] - width // 2, draw._image.width - width))
    y = max(0, min(xy[1] - height // 2, draw._image.height - height))
    draw.rounded_rectangle((x, y, x + width, y + height), radius=4, fill=(0, 0, 0))
    draw.text(
        (x + 7, y + 3), label, font=font, fill=(255, 255, 255),
        stroke_width=1, stroke_fill=(0, 0, 0),
    )


def _numbered_overlay(
    record: dict[str, Any],
    stable_order: list[int],
    out_path: Path,
) -> dict[int, int]:
    source = _preview_source(record["source_path"])
    output = source.copy()
    display_to_stable: dict[int, int] = {}
    proposal_by_id = {item["stable_id"]: item for item in record["proposals"]}
    for display_id, stable_id in enumerate(stable_order, 1):
        proposal = proposal_by_id[stable_id]
        display_to_stable[display_id] = stable_id
        mask = _load_preview_mask(proposal["mask_path"], source.size)
        color = PALETTE[(display_id - 1) % len(PALETTE)]
        layer = Image.new("RGB", source.size, color)
        alpha = mask.point(lambda value: int(value * 0.28))
        output = Image.composite(layer, output, alpha)
        outline = Image.new("RGB", source.size, color)
        output = Image.composite(outline, output, _edge(mask))

    draw = ImageDraw.Draw(output)
    for display_id, stable_id in display_to_stable.items():
        cx, cy = proposal_by_id[stable_id]["centroid"]
        _draw_number(draw, (int(cx * source.width), int(cy * source.height)), display_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path, "JPEG", quality=92, subsampling=0)
    return display_to_stable


def _prepare_overlays(record: dict[str, Any], out_dir: Path, seed: int) -> None:
    stable = [item["stable_id"] for item in record["proposals"]]
    rng = random.Random(f"{seed}:{record['asset_id']}")
    permuted = list(stable)
    rng.shuffle(permuted)
    if len(permuted) > 1 and permuted == stable:
        permuted = permuted[1:] + permuted[:1]

    overlay_dir = out_dir / "overlays"
    path_a = overlay_dir / f"{record['asset_id']}.a.jpg"
    path_b = overlay_dir / f"{record['asset_id']}.b.jpg"
    map_a = _numbered_overlay(record, stable, path_a)
    map_b = _numbered_overlay(record, permuted, path_b)
    record["variants"] = {
        "a": {
            "overlay_path": str(path_a),
            "display_to_stable": {str(k): v for k, v in map_a.items()},
        },
        "b": {
            "overlay_path": str(path_b),
            "display_to_stable": {str(k): v for k, v in map_b.items()},
        },
    }


def _focus_proposals(record: dict[str, Any], radius_fraction: float) -> None:
    """Ground proposals with the source-only VLM visual-center point.

    Every SAM3 proposal remains in ``all_proposals``. Only proposals whose mask
    touches a small neighborhood around the chosen subject point are shown to
    the instance selector; proposals for other peers cannot be the annotated
    individual.
    """
    annotation = record.get("subject_annotation", {}).get("a", {})
    center = annotation.get("visual_center")
    proposals = list(record.get("proposals", ()))
    record["all_proposals"] = proposals
    if annotation.get("subject_scope") == "group":
        # 群体主体：成员分布在画面各处，中心点过滤会误杀同伴实例；全量给选择器。
        record["focus_center"] = None
        record["focus_radius_fraction"] = None
        return
    if center is None or not proposals:
        return
    width, height = record["native_size"]
    x = int(round(float(center[0]) * max(width - 1, 0)))
    y = int(round(float(center[1]) * max(height - 1, 0)))
    radius = max(2, int(round(radius_fraction * min(width, height))))
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    win = (x0 / max(width, 1), y0 / max(height, 1),
           x1 / max(width, 1), y1 / max(height, 1))
    focused: list[dict[str, Any]] = []
    for proposal in proposals:
        bx0, by0, bx1, by1 = proposal["bbox"]
        if bx1 < win[0] or bx0 > win[2] or by1 < win[1] or by0 > win[3]:
            continue   # bbox 与中心窗不相交：免掉整张 mask PNG 的解码
        with Image.open(proposal["mask_path"]) as image:
            mask = np.asarray(image.convert("L"), dtype=np.uint8)
        if bool((mask[y0:y1, x0:x1] >= 128).any()):
            focused.append(proposal)
    record["focus_center"] = [float(center[0]), float(center[1])]
    record["focus_radius_fraction"] = radius_fraction
    record["proposals"] = focused
    if not focused:
        record["status"] = "no_center_candidate"


def _data_uri(path: str, long_edge: int = 1024) -> str:
    with Image.open(path) as image:
        value = image.convert("RGB")
        value.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        value.save(buffer, "JPEG", quality=90, subsampling=0)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return "data:image/jpeg;base64," + encoded


def _subject_prompt() -> str:
    return (
        "Identify whether this image has a discrete, localizable primary subject "
        "for a local photo edit. First judge the source itself; do not invent a "
        "subject merely to satisfy the task. A valid subject has a coherent visible "
        "extent: a person, animal, vehicle, product, building, sculpture, plant or "
        "flower (a distinct flowering plant, flower cluster, potted plant, or a "
        "clear foreground tuft such as cotton grass), or another clearly isolated "
        "natural object. Sky, clouds, open water, sunlight, shadow, roads, uniform "
        "texture fields, a forest with no dominant element, and general scenery are "
        "not subjects.\n\n"
        "Decide the subject scope. Use \"single\" when one individual is the "
        "subject. Use \"group\" when several same-class individuals jointly form "
        "the subject a retoucher would adjust together: a couple, a small group of "
        "hikers, two cats, a cluster of flowers. For a group do NOT pick one "
        "member; the group as a whole is the subject. Incidental background "
        "passers-by are not group members.\n\n"
        "For a valid subject, sam_prompt must be a short English SINGULAR noun "
        "phrase naming the member class that a text-prompted segmentation model "
        "can detect (for a couple use \"person\", not \"couple\"; for a flower "
        "cluster use \"flower\"). A person or object depicted inside a framed "
        "artwork, poster, screen, mirror, or reflection is NOT a subject; a "
        "gallery wall of artworks with no real subject returns false. "
        "description must identify the subject (the "
        "individual, or the group and its members). visual_center is the "
        "approximate normalized [x,y] center from 0 to 1 of the individual, or of "
        "the whole group. expected_count is the number of member instances you "
        "expect (1 for single; your best estimate for a group).\n\n"
        "Return JSON only: "
        '{"has_localizable_subject":true,"subject_scope":"single|group",'
        '"sam_prompt":"woman","expected_count":1,'
        '"description":"woman in blue near center","visual_center":[0.48,0.55],'
        '"confidence":0.0,"reason_code":"clear_primary|clear_group|no_discrete_subject"}'
    )


def _parse_subject_label(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    required = {
        "has_localizable_subject", "subject_scope", "sam_prompt", "expected_count",
        "description", "visual_center", "confidence", "reason_code",
    }
    if not isinstance(parsed, dict) or set(parsed) != required:
        return {"status": "parse_error", "raw_response": raw}
    has_subject = parsed.get("has_localizable_subject")
    if not isinstance(has_subject, bool):
        return {"status": "parse_error", "raw_response": raw}
    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not 0.0 <= float(confidence) <= 1.0:
        return {"status": "parse_error", "raw_response": raw}
    scope = parsed["subject_scope"]
    expected = parsed["expected_count"]
    reason_code = parsed["reason_code"]
    if scope not in {"single", "group"} or type(expected) is not int \
            or not 1 <= expected <= 16:
        return {"status": "parse_error", "raw_response": raw}
    if reason_code not in {"clear_primary", "clear_group", "no_discrete_subject"}:
        return {"status": "parse_error", "raw_response": raw}
    result: dict[str, Any] = {
        "status": "ok",
        "has_localizable_subject": has_subject,
        "confidence": float(confidence),
        "reason_code": reason_code,
        "raw_response": raw,
    }
    if not has_subject:
        if parsed["visual_center"] is not None:
            return {"status": "parse_error", "raw_response": raw}
        if not isinstance(parsed["sam_prompt"], str) or not isinstance(parsed["description"], str):
            return {"status": "parse_error", "raw_response": raw}
        result.update({
            "sam_prompt": parsed["sam_prompt"],
            "description": parsed["description"],
            "visual_center": None,
            "subject_scope": scope,
            "expected_count": expected,
        })
        return result
    prompt = parsed["sam_prompt"]
    description = parsed["description"]
    center = parsed["visual_center"]
    if not isinstance(prompt, str) or not prompt or prompt != prompt.strip().lower() \
            or len(prompt.split()) > 10:
        return {"status": "parse_error", "raw_response": raw}
    if not isinstance(description, str) or not description or description != description.strip():
        return {"status": "parse_error", "raw_response": raw}
    if not isinstance(center, list) or len(center) != 2 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not 0.0 <= float(value) <= 1.0 for value in center
    ):
        return {"status": "parse_error", "raw_response": raw}
    result.update({"sam_prompt": prompt, "description": description,
                   "visual_center": [float(center[0]), float(center[1])],
                   "subject_scope": scope, "expected_count": expected})
    return result


def _call_subject_label(
    row: dict[str, str],
    variant_name: str,
    base_url: str,
    model: str,
    timeout: float,
    *,
    api_key: str = "EMPTY",
) -> tuple[str, str, dict[str, Any]]:
    started = time.perf_counter()
    try:
        response = request_text(
            base_url=base_url,
            api_key=api_key,
            model=model,
            content=[
                {"type": "input_image", "image_url": _data_uri(row["source_path"])},
                {"type": "input_text", "text": _subject_prompt()},
            ],
            schema_name="sam3_subject_label",
            schema=SUBJECT_LABEL_SCHEMA,
            timeout=timeout,
        )
    except ResponsesVlmError as error:
        return row["asset_id"], variant_name, {
            "status": "transport_error",
            "error": error.error_type,
            "attempt": error.attempts,
            "seconds": time.perf_counter() - started,
        }
    result = _parse_subject_label(response.text)
    result["attempt"] = response.attempt
    result["seconds"] = time.perf_counter() - started
    return row["asset_id"], variant_name, result


def _subject_label_agrees(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("status") != "ok" or right.get("status") != "ok":
        return False
    if left.get("has_localizable_subject") != right.get("has_localizable_subject"):
        return False
    if not left.get("has_localizable_subject"):
        return True
    if left.get("visual_center") is None or right.get("visual_center") is None:
        return left.get("sam_prompt") == right.get("sam_prompt")
    left_center = np.asarray(left.get("visual_center"), dtype=np.float32)
    right_center = np.asarray(right.get("visual_center"), dtype=np.float32)
    return bool(np.linalg.norm(left_center - right_center) <= 0.20)


def _selector_prompt(record: dict[str, Any], display_ids: list[int]) -> str:
    annotation = record.get("subject_annotation", {}).get("a", {})
    center = annotation.get("visual_center")
    scope = annotation.get("subject_scope") or "single"
    if scope == "group":
        task = (
            "The subject is a GROUP. Select EVERY proposal that is a member of "
            "the described group (each person of the couple, each hiker, each "
            "flower of the cluster); their union becomes the subject mask. "
            "Exclude incidental background passers-by, reflections, and objects "
            "of a different class. If one merged proposal already covers the "
            "whole group, selecting just that one is correct. A member covered "
            "by no proposal is acceptable; select the members that are covered."
        )
    else:
        task = (
            "Choose exactly one proposal only when it covers the single strongest "
            "visual-center subject a retoucher would naturally adjust. Do not "
            "choose by largest area. When the type is shared by several peers, "
            "selecting the one described individual is required and valid; never "
            "reject it merely because peers are omitted."
        )
    return (
        "You are selecting the canonical local-edit subject mask. Image 1 is the "
        "unmodified source. Image 2 shows SAM3 proposals as colored, numbered "
        "regions. The caption concept is only a hint and may be wrong.\n\n"
        f"Pre-annotated subject type: {record['main_subject']}\n"
        f"Subject scope: {scope}\n"
        f"Subject description: {annotation.get('description', '')}\n"
        f"Approximate subject center [x,y]: {center}\n"
        f"Available proposal IDs: {display_ids}\n\n"
        f"{task} If a chosen proposal covers only a fragment of its subject, "
        "such as one leaflet of a fern frond, it is invalid; if no proposal "
        "matches the subject, return no_valid_mask. If the source has no "
        "localizable primary subject, return no_subject. Use ambiguous only for "
        "a genuine unresolved tie.\n\n"
        "Return JSON only: "
        '{"decision":"select|no_subject|no_valid_mask|ambiguous",'
        '"instance_ids":[1],"confidence":0.0,"subject":"...",'
        '"reason_code":"clear_primary|clear_group|pure_landscape|proposal_miss|tie"}'
    )


def _parse_selection(raw: str, valid_ids: set[int]) -> dict[str, Any]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        parsed = None
    required = {"decision", "instance_ids", "confidence", "subject", "reason_code"}
    if not isinstance(parsed, dict) or set(parsed) != required:
        return {"status": "parse_error", "raw_response": raw}
    decision = parsed["decision"]
    if decision not in {"select", "no_subject", "no_valid_mask", "ambiguous"}:
        return {"status": "parse_error", "raw_response": raw}
    raw_ids = parsed["instance_ids"]
    if not isinstance(raw_ids, list) or len(raw_ids) > 16 \
            or any(type(value) is not int or value < 1 for value in raw_ids) \
            or len(raw_ids) != len(set(raw_ids)):
        return {"status": "parse_error", "raw_response": raw}
    if (decision == "select") != bool(raw_ids):
        return {"status": "parse_error", "raw_response": raw}
    ids = sorted(raw_ids)
    if not set(ids) <= valid_ids:
        return {"status": "invalid_id", "raw_response": raw}
    confidence = parsed["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) \
            or not 0.0 <= float(confidence) <= 1.0:
        return {"status": "parse_error", "raw_response": raw}
    if not isinstance(parsed["subject"], str) or parsed["subject"] != parsed["subject"].strip():
        return {"status": "parse_error", "raw_response": raw}
    reason_code = parsed["reason_code"]
    if reason_code not in {
        "clear_primary", "clear_group", "pure_landscape", "proposal_miss", "tie",
    }:
        return {"status": "parse_error", "raw_response": raw}
    return {
        "status": "ok",
        "decision": decision,
        "display_instance_ids": ids or None,
        "display_instance_id": ids[0] if ids else None,
        "confidence": float(confidence),
        "subject": parsed["subject"],
        "reason_code": reason_code,
        "raw_response": raw,
    }


def _call_selector(
    record: dict[str, Any],
    variant_name: str,
    base_url: str,
    model: str,
    timeout: float,
    *,
    api_key: str = "EMPTY",
) -> tuple[str, str, dict[str, Any]]:
    variant = record["variants"][variant_name]
    mapping = {int(k): int(v) for k, v in variant["display_to_stable"].items()}
    started = time.perf_counter()
    try:
        response = request_text(
            base_url=base_url,
            api_key=api_key,
            model=model,
            content=[
                {"type": "input_image", "image_url": _data_uri(record["source_path"])},
                {"type": "input_image", "image_url": _data_uri(variant["overlay_path"])},
                {"type": "input_text", "text": _selector_prompt(record, sorted(mapping))},
            ],
            schema_name="sam3_subject_selection",
            schema=SUBJECT_SELECTION_SCHEMA,
            timeout=timeout,
        )
    except ResponsesVlmError as error:
        return record["asset_id"], variant_name, {
            "status": "transport_error",
            "error": error.error_type,
            "attempt": error.attempts,
            "seconds": time.perf_counter() - started,
        }
    result = _parse_selection(response.text, set(mapping))
    result["attempt"] = response.attempt
    result["seconds"] = time.perf_counter() - started
    if result.get("decision") == "select":
        stable_ids = sorted(mapping[did] for did in result["display_instance_ids"])
        result["stable_instance_ids"] = stable_ids
        result["stable_instance_id"] = stable_ids[0]
        shas = []
        for sid in stable_ids:
            proposal = next(item for item in record["proposals"] if item["stable_id"] == sid)
            shas.append(proposal["mask_sha256"])
        result["mask_sha256_list"] = sorted(shas)
        result["mask_sha256"] = result["mask_sha256_list"][0]
    return record["asset_id"], variant_name, result


def _selection_agrees(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.get("status") != "ok" or right.get("status") != "ok":
        return False
    if left.get("decision") != right.get("decision"):
        return False
    if left.get("decision") == "select":
        return (left.get("mask_sha256_list") or [left.get("mask_sha256")]) == \
               (right.get("mask_sha256_list") or [right.get("mask_sha256")])
    return True


def _single_mask_overlay(
    record: dict[str, Any],
    result: dict[str, Any] | None,
    size: tuple[int, int],
) -> Image.Image:
    source = _preview_source(record["source_path"])
    if not result or result.get("status") != "ok" or result.get("decision") != "select":
        panel = _fit(source, size)
        draw = ImageDraw.Draw(panel)
        decision = (result or {}).get("decision") or (result or {}).get("status") or "missing"
        draw.rectangle((5, 5, size[0] - 5, 42), fill=(0, 0, 0))
        draw.text((12, 10), str(decision), font=_font(20), fill=(255, 220, 120))
        return panel
    stable_ids = [int(v) for v in
                  (result.get("stable_instance_ids") or [result["stable_instance_id"]])]
    union = None
    for sid in stable_ids:
        proposal = next(item for item in record["proposals"] if item["stable_id"] == sid)
        mask = _load_preview_mask(proposal["mask_path"], source.size)
        union = mask if union is None else Image.fromarray(
            np.maximum(np.asarray(union), np.asarray(mask)))
    mask = union
    layer = Image.new("RGB", source.size, (38, 220, 108))
    output = Image.composite(layer, source, mask.point(lambda value: int(value * 0.40)))
    output = Image.composite(Image.new("RGB", source.size, (255, 235, 80)), output, _edge(mask))
    panel = _fit(output, size)
    draw = ImageDraw.Draw(panel)
    ids_txt = ",".join(str(s) for s in stable_ids)
    label = f"stable #{ids_txt}  conf={float(result.get('confidence', 0)):.2f}"
    draw.rectangle((5, 5, min(size[0] - 5, 285), 38), fill=(0, 0, 0))
    draw.text((10, 9), label, font=_font(16), fill=(255, 255, 255))
    return panel


def _legacy_overlay(record: dict[str, Any], size: tuple[int, int]) -> Image.Image:
    source = _preview_source(record["source_path"])
    path = record.get("legacy_union_path") or ""
    if not path or not os.path.exists(path):
        return _fit(source, size)
    mask = _load_preview_mask(path, source.size)
    layer = Image.new("RGB", source.size, (246, 80, 80))
    output = Image.composite(layer, source, mask.point(lambda value: int(value * 0.36)))
    output = Image.composite(Image.new("RGB", source.size, (255, 230, 80)), output, _edge(mask))
    return _fit(output, size)


def _label_panel(panel: Image.Image, label: str) -> Image.Image:
    output = Image.new("RGB", (panel.width, panel.height + 32), (10, 10, 10))
    output.paste(panel, (0, 32))
    ImageDraw.Draw(output).text((8, 6), label, font=_font(17), fill=(235, 235, 235))
    return output


def _contact_sheets(records: list[dict[str, Any]], out_dir: Path) -> list[str]:
    panel_size = (300, 220)
    rows_per_sheet = 4
    paths: list[str] = []
    for page, start in enumerate(range(0, len(records), rows_per_sheet), 1):
        batch = records[start:start + rows_per_sheet]
        row_height = panel_size[1] + 72
        sheet = Image.new("RGB", (panel_size[0] * 5, row_height * len(batch)), (8, 8, 8))
        for row_index, record in enumerate(batch):
            source = _fit(_preview_source(record["source_path"]), panel_size)
            variant_a = record.get("variants", {}).get("a", {})
            overlay_path = variant_a.get("overlay_path")
            if overlay_path and os.path.exists(overlay_path):
                overlay = _fit(Image.open(overlay_path).convert("RGB"), panel_size)
            else:
                overlay = source.copy()
            selection_a = record.get("selection", {}).get("a")
            selection_b = record.get("selection", {}).get("b")
            if record.get("status") == "no_subject":
                selection_a = {"status": "ok", "decision": "no_subject"}
                selection_b = {"status": "ok", "decision": "no_subject"}
            panels = [
                _label_panel(source, "source"),
                _label_panel(_legacy_overlay(record, panel_size), "legacy union"),
                _label_panel(
                    overlay,
                    f"new={record.get('main_subject') or record.get('status')} | "
                    f"focus/all={len(record['proposals'])}/{len(record.get('all_proposals', record['proposals']))}",
                ),
                _label_panel(_single_mask_overlay(record, selection_a, panel_size), "selector A"),
                _label_panel(_single_mask_overlay(record, selection_b, panel_size), "selector B, IDs permuted"),
            ]
            y = row_index * row_height + 40
            for column, panel in enumerate(panels):
                sheet.paste(panel, (column * panel_size[0], y))
            select_state = (
                "n/a" if record.get("status") != "ready"
                else ("yes" if record.get("selection_agreement") else "NO")
            )
            title = (
                f"#{record['audit_index']:03d} {record['asset_id']} | "
                f"{record['stratum']} | legacy={record.get('legacy_main_subject', '')} | "
                f"label={'yes' if record.get('subject_label_agreement') else 'NO'} | "
                f"select={select_state}"
            )
            ImageDraw.Draw(sheet).text(
                (8, row_index * row_height + 8), title[:150],
                font=_font(19), fill=(255, 255, 255),
            )
        path = out_dir / f"sheet_{page:03d}.jpg"
        sheet.save(path, "JPEG", quality=91, subsampling=0)
        paths.append(str(path))
    return paths


def _write_review_csv(records: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "index", "asset_id", "stratum", "legacy_main_subject", "main_subject",
        "subject_a_has_subject", "subject_a_description", "subject_b_has_subject",
        "subject_b_description", "subject_label_agreement", "pipeline_status",
        "proposal_count", "all_proposal_count",
        "vlm_a_decision", "vlm_a_stable_id", "vlm_b_decision",
        "vlm_b_stable_id", "selection_agreement", "human_primary_correct",
        "human_single_instance_correct", "human_mask_usable", "human_should_skip",
        "human_preferred_stable_id", "notes",
    ]
    with (out_dir / "review.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            left = record.get("selection", {}).get("a", {})
            right = record.get("selection", {}).get("b", {})
            subject_left = record.get("subject_annotation", {}).get("a", {})
            subject_right = record.get("subject_annotation", {}).get("b", {})
            writer.writerow({
                "index": record["audit_index"],
                "asset_id": record["asset_id"],
                "stratum": record["stratum"],
                "legacy_main_subject": record.get("legacy_main_subject") or "",
                "main_subject": record["main_subject"],
                "subject_a_has_subject": subject_left.get("has_localizable_subject"),
                "subject_a_description": subject_left.get("description") or "",
                "subject_b_has_subject": subject_right.get("has_localizable_subject"),
                "subject_b_description": subject_right.get("description") or "",
                "subject_label_agreement": int(bool(record.get("subject_label_agreement"))),
                "pipeline_status": record.get("status") or "",
                "proposal_count": len(record["proposals"]),
                "all_proposal_count": len(record.get("all_proposals", record["proposals"])),
                "vlm_a_decision": left.get("decision") or left.get("status") or "",
                "vlm_a_stable_id": ",".join(map(str, left.get("stable_instance_ids") or ([left["stable_instance_id"]] if left.get("stable_instance_id") else []))),
                "vlm_b_decision": right.get("decision") or right.get("status") or "",
                "vlm_b_stable_id": ",".join(map(str, right.get("stable_instance_ids") or ([right["stable_instance_id"]] if right.get("stable_instance_id") else []))),
                "selection_agreement": int(bool(record.get("selection_agreement"))),
                "human_primary_correct": "",
                "human_single_instance_correct": "",
                "human_mask_usable": "",
                "human_should_skip": "",
                "human_preferred_stable_id": "",
                "notes": "",
            })


def _summary(records: list[dict[str, Any]], sheets: list[str]) -> dict[str, Any]:
    relabeled = [
        record for record in records
        if record.get("subject_annotation", {}).get("a", {}).get("reason_code") != "legacy_subject"
    ]
    subject_parse_ok = sum(
        result.get("status") == "ok"
        for record in relabeled
        for result in record.get("subject_annotation", {}).values()
    )
    subject_agreed = [
        record for record in relabeled if record.get("subject_label_agreement")
    ]
    ready = [record for record in records if record.get("status") == "ready"]
    evaluated = [record for record in ready if "selection" in record]
    agreed = [record for record in evaluated if record.get("selection_agreement")]
    both_select = [
        record for record in evaluated
        if record["selection"]["a"].get("decision") == "select"
        and record["selection"]["b"].get("decision") == "select"
    ]
    select_agreed = [record for record in both_select if record.get("selection_agreement")]
    parse_ok = sum(
        result.get("status") == "ok"
        for record in evaluated
        for result in record["selection"].values()
    )
    return {
        "sample_size": len(records),
        "subject_label_calls": 2 * len(relabeled),
        "subject_label_parse_ok": subject_parse_ok,
        "subject_label_agreement": (
            len(subject_agreed) / len(relabeled) if relabeled else None
        ),
        "no_subject": sum(record.get("status") == "no_subject" for record in records),
        "ready": len(ready),
        "sam_miss": sum(record.get("status") == "sam_miss" for record in records),
        "no_center_candidate": sum(
            record.get("status") == "no_center_candidate" for record in records
        ),
        "too_many_instances": sum(record.get("status") == "too_many_instances" for record in records),
        "degenerate_full_frame": sum(
            record.get("status") == "degenerate_full_frame" for record in records),
        "group_envelope_too_large": sum(
            record.get("status") == "group_envelope_too_large" for record in records),
        "vlm_calls": 2 * len(evaluated),
        "vlm_parse_ok": parse_ok,
        "overall_renumber_agreement": len(agreed) / len(evaluated) if evaluated else None,
        "both_select_count": len(both_select),
        "selected_mask_renumber_agreement": len(select_agreed) / len(both_select) if both_select else None,
        "mean_sam_seconds": float(np.mean([record["sam_seconds"] for record in records])) if records else None,
        "mean_vlm_seconds": float(np.mean([
            result.get("seconds", 0.0)
            for record in evaluated for result in record["selection"].values()
        ])) if evaluated else None,
        "disagreements": [record["asset_id"] for record in evaluated if not record.get("selection_agreement")],
        "sheets": sheets,
        "note": "Renumber agreement measures stability, not human correctness. Complete review.csv before activation.",
    }


def _write_report(summary: dict[str, Any], out_dir: Path) -> None:
    subject_agreement = summary.get("subject_label_agreement")
    overall = summary.get("overall_renumber_agreement")
    selected = summary.get("selected_mask_renumber_agreement")
    lines = [
        "# Subject Instance Selector Pilot",
        "",
        f"- Sample: `{summary['sample_size']}`",
        f"- Source-label JSON success: `{summary['subject_label_parse_ok']}/{summary['subject_label_calls']}`",
        f"- Source-label repeat agreement: `{subject_agreement:.1%}`" if subject_agreement is not None else "- Source-label repeat agreement: n/a",
        f"- Source-label no-subject: `{summary['no_subject']}`",
        f"- SAM3 ready / miss / no-center / too-many: `{summary['ready']}` / `{summary['sam_miss']}` / `{summary['no_center_candidate']}` / `{summary['too_many_instances']}`",
        f"- VLM parse success: `{summary['vlm_parse_ok']}/{summary['vlm_calls']}`",
        f"- Same decision after ID permutation: `{overall:.1%}`" if overall is not None else "- Same decision after ID permutation: n/a",
        f"- Same selected mask when both select: `{selected:.1%}`" if selected is not None else "- Same selected mask when both select: n/a",
        "",
        "The permutation test detects ID/order sensitivity. It does not establish semantic correctness.",
        "Review the sheets and fill `review.csv` before deciding whether a single VLM call is trusted.",
        "",
        "## Disagreements",
        "",
        *(f"- `{asset_id}`" for asset_id in summary["disagreements"]),
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-score", type=float, default=0.30)
    parser.add_argument("--dedupe-iou", type=float, default=0.92)
    parser.add_argument("--max-proposals", type=int, default=16)
    parser.add_argument("--focus-radius", type=float, default=0.025)
    parser.add_argument("--vlm-base-url", default=os.environ.get("SOURCE_QA_VLLM", "http://localhost:8003/v1"))
    parser.add_argument("--vlm-model", default="qwen3_5-35b-a3b")
    parser.add_argument("--vlm-workers", type=int, default=8)
    parser.add_argument("--vlm-timeout", type=float, default=120.0)
    parser.add_argument(
        "--use-legacy-subject", action="store_true",
        help="skip source-only VLM relabeling and use audit.csv main_subject",
    )
    parser.add_argument("--max-mask-area", type=float, default=0.85)
    parser.add_argument("--max-group-envelope", type=float, default=0.67)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sampled = _sample_rows(args.audit_csv, args.n, args.seed)
    sample_ids = [row["asset_id"] for row in sampled]
    records_path = args.out / "records.json"
    existing: dict[str, dict[str, Any]] = {}
    if args.resume and records_path.exists():
        for record in json.loads(records_path.read_text(encoding="utf-8")):
            if record["asset_id"] in sample_ids:
                existing[record["asset_id"]] = record

    missing = [row for row in sampled if row["asset_id"] not in existing]
    subject_labels: dict[str, dict[str, dict[str, Any]]] = {
        row["asset_id"]: {} for row in missing
    }
    if not args.use_legacy_subject:
        label_tasks = [
            (row, variant_name)
            for row in missing for variant_name in ("a", "b")
        ]
        with ThreadPoolExecutor(max_workers=args.vlm_workers) as pool:
            futures = [
                pool.submit(
                    _call_subject_label, row, variant_name, args.vlm_base_url,
                    args.vlm_model, args.vlm_timeout,
                )
                for row, variant_name in label_tasks
            ]
            for index, future in enumerate(as_completed(futures), 1):
                asset_id, variant_name, result = future.result()
                subject_labels[asset_id][variant_name] = result
                print(
                    f"[subject] {index}/{len(futures)} {asset_id}/{variant_name} "
                    f"status={result.get('status')} "
                    f"has_subject={result.get('has_localizable_subject')}",
                    flush=True,
                )

    masker = None

    for audit_index, row in enumerate(sampled, 1):
        if row["asset_id"] in existing:
            existing[row["asset_id"]]["audit_index"] = audit_index
            continue
        legacy_subject = row["main_subject"]
        if args.use_legacy_subject:
            labels = {
                "a": {
                    "status": "ok", "has_localizable_subject": True,
                    "sam_prompt": legacy_subject, "description": legacy_subject,
                    "visual_center": None, "confidence": 1.0,
                    "reason_code": "legacy_subject",
                },
                "b": {
                    "status": "ok", "has_localizable_subject": True,
                    "sam_prompt": legacy_subject, "description": legacy_subject,
                    "visual_center": None, "confidence": 1.0,
                    "reason_code": "legacy_subject",
                },
            }
        else:
            labels = subject_labels[row["asset_id"]]
        left_label = labels.get("a", {})
        right_label = labels.get("b", {})
        label_agreement = _subject_label_agrees(left_label, right_label)
        selected_subject = str(left_label.get("sam_prompt") or "")

        if left_label.get("status") != "ok":
            record = {
                "asset_id": row["asset_id"],
                "source_path": row["source_path"],
                "legacy_union_path": row.get("mask_path") or "",
                "stratum": row.get("stratum") or "",
                "scene": row.get("scene") or "",
                "main_subject": selected_subject,
                "proposals": [],
                "status": "subject_label_error",
                "error": left_label.get("error") or left_label.get("raw_response") or "",
                "sam_seconds": 0.0,
            }
        elif not left_label.get("has_localizable_subject"):
            record = {
                "asset_id": row["asset_id"],
                "source_path": row["source_path"],
                "legacy_union_path": row.get("mask_path") or "",
                "stratum": row.get("stratum") or "",
                "scene": row.get("scene") or "",
                "main_subject": "",
                "proposals": [],
                "status": "no_subject",
                "sam_seconds": 0.0,
            }
        else:
            sam_row = dict(row)
            sam_row["main_subject"] = selected_subject
            try:
                if masker is None:
                    from dataset_build.masking import Sam3Masker

                    masker = Sam3Masker(
                        device=args.device, score_threshold=args.min_score
                    )
                    masker._ensure_loaded()
                record = _sam_proposals(
                    masker, sam_row, args.out, args.min_score, args.dedupe_iou
                )
            except Exception as error:  # noqa: BLE001
                record = {
                    "asset_id": row["asset_id"],
                    "source_path": row["source_path"],
                    "legacy_union_path": row.get("mask_path") or "",
                    "stratum": row.get("stratum") or "",
                    "scene": row.get("scene") or "",
                    "main_subject": selected_subject,
                    "proposals": [],
                    "status": "sam_error",
                    "error": str(error),
                    "sam_seconds": 0.0,
                }
        record["audit_index"] = audit_index
        record["legacy_main_subject"] = legacy_subject
        record["subject_annotation"] = labels
        record["subject_label_agreement"] = label_agreement
        if record.get("status") == "ready":
            _focus_proposals(record, args.focus_radius)
        if len(record.get("proposals", ())) > args.max_proposals:
            scope = labels.get("a", {}).get("subject_scope")
            if scope == "group" and record.get("status") == "ready":
                _dense_group_union(record, args.out)
            else:
                record["status"] = "too_many_instances"
        if record.get("status") == "ready":
            _prepare_overlays(record, args.out, args.seed)
        existing[row["asset_id"]] = record
        ordered = [existing[asset_id] for asset_id in sample_ids if asset_id in existing]
        _atomic_json(records_path, ordered)
        print(
            f"[sam] {audit_index}/{len(sampled)} {row['asset_id']} "
            f"status={record['status']} proposals={len(record.get('proposals', ()))}",
            flush=True,
        )

    records = [existing[asset_id] for asset_id in sample_ids]
    tasks = []
    for record in records:
        if record.get("status") != "ready":
            continue
        record["selection"] = {}
        for variant_name in ("a", "b"):
            tasks.append((record, variant_name))
    with ThreadPoolExecutor(max_workers=args.vlm_workers) as pool:
        futures = [
            pool.submit(
                _call_selector, record, variant_name, args.vlm_base_url,
                args.vlm_model, args.vlm_timeout,
            )
            for record, variant_name in tasks
        ]
        for index, future in enumerate(as_completed(futures), 1):
            asset_id, variant_name, result = future.result()
            existing[asset_id]["selection"][variant_name] = result
            print(
                f"[vlm] {index}/{len(futures)} {asset_id}/{variant_name} "
                f"status={result.get('status')} decision={result.get('decision')}",
                flush=True,
            )

    for record in records:
        selection = record.get("selection", {})
        record["selection_agreement"] = _selection_agrees(
            selection.get("a", {}), selection.get("b", {})
        )
    # 退化守卫：全画幅 mask / 多成员 group 包络占幅过大 → 无明确主体，舍弃
    for record in records:
        if record.get("status") != "ready":
            continue
        sel = (record.get("selection") or {}).get("a") or {}
        ids = sel.get("stable_instance_ids") or []
        props = [p for p in record.get("proposals", ()) if p["stable_id"] in ids]
        if not props:
            continue
        area = sum(float(p["area"]) for p in props)
        env = ((max(p["bbox"][2] for p in props) - min(p["bbox"][0] for p in props))
               * (max(p["bbox"][3] for p in props) - min(p["bbox"][1] for p in props)))
        record["selected_mask_area"] = round(area, 4)
        record["selected_envelope"] = round(env, 4)
        scope = record.get("subject_annotation", {}).get("a", {}).get("subject_scope")
        multi = len(ids) > 1 or bool(record.get("dense_union"))
        if area > args.max_mask_area:
            record["status"] = "degenerate_full_frame"
        elif scope == "group" and multi and env > args.max_group_envelope:
            record["status"] = "group_envelope_too_large"
    _atomic_json(records_path, records)
    _atomic_json(records_path, records)
    sheets = _contact_sheets(records, args.out)
    _write_review_csv(records, args.out)
    summary = _summary(records, sheets)
    _atomic_json(args.out / "summary.json", summary)
    _write_report(summary, args.out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
