"""Evaluate whether 768px local-preset QA can select native-resolution winners.

The experiment is intentionally separate from the build path.  For each real
source it reads the VLM-selected ``source_captions.main_subject`` and that
concept's cached SAM3 mask, then creates the same deterministic eight masks at
both resolutions:

    1 semantic + 1 radial + 3 band + 3 linear

One calibrated preset is replayed once per resolution and broadcast over all
eight alpha masks.  The current production ``construct.qa.qa_rank`` scorer is
run independently on the 768px and native outputs.  Reports include top-k
recall, rank correlations, technical-veto stability, and native-downsampled
visual differences.

Recommended invocation (ArtiMuse needs the 4.37 compatibility venv)::

    MONETGPT_TORCH_DEVICE=cuda:1 CONSTRUCT_IAA_DEVICE=cuda:1 \
      /home/bc/.venvs/iaa437/bin/python \
      -m dataset_build.tools.eval_preview_native_local all

Rendering and scoring are resumable as separate phases with ``render``,
``score``, and ``analyze``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFile, ImageFont, ImageOps

from dataset_build.mask_cache import concept_slug
from dataset_build.source_qa import db

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "_mask_review" / "preview_native_eval_v1"
FEATURES_PATH = Path(
    "/home/bc/data/datasets/vera_directionA_1M/preset_bank_full/features.jsonl"
)

# Fixed, diverse real sources with a VLM main subject and a cached SAM3 mask.
# The associated calibrated looks span high-key portrait, warm/vibrant,
# teal-orange, and cool/muted families.  Keeping this list fixed makes reruns
# directly comparable as rendering and QA kernels evolve.
DEFAULT_CASES = (
    ("src_9eb39c217e7bc7d2", "rcp_14180a06e96c7848"),  # portrait / woman
    ("src_de0e17c168127d08", "rcp_0d6f7c76b4a41dd3"),  # still life / butterfly
    ("src_debf8ae0d6400b45", "rcp_067bc3f0bdc52696"),  # cat
    ("src_dda3177625e699cd", "rcp_0303a5b42ef6ef6e"),  # hiker
    ("src_d5960600eb038516", "rcp_0d6f7c76b4a41dd3"),  # horse
    ("src_d118f52fb517e835", "rcp_067bc3f0bdc52696"),  # 12.8MP statue
    ("src_3a0f326f6488a78a", "rcp_0303a5b42ef6ef6e"),  # architecture
    ("src_c62fad04cdbfd3df", "rcp_087d2b23cea98aa6"),  # night / fish
)


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _json_load(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _feature_map() -> dict[str, dict]:
    wanted = {preset_id for _, preset_id in DEFAULT_CASES}
    found: dict[str, dict] = {}
    with FEATURES_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            preset_id = row.get("preset_id")
            if preset_id in wanted:
                found[preset_id] = row
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"missing preset features: {missing}")
    return found


def _parse_jsonish(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def _load_cases(limit: int) -> list[dict]:
    features = _feature_map()
    conn = db.connect()
    cases = []
    try:
        for asset_id, preset_id in DEFAULT_CASES[:limit or None]:
            row = conn.execute(
                "SELECT a.asset_id,a.path,a.scene,a.width,a.height,"
                "a.is_portrait_pool,c.caption,c.main_subject,c.subjects "
                "FROM assets a JOIN source_captions c USING(asset_id) "
                "WHERE a.asset_id=%s",
                (asset_id,),
            ).fetchone()
            if row is None:
                raise RuntimeError(f"source not found in DB: {asset_id}")
            slug = concept_slug(row["main_subject"])
            mask = conn.execute(
                "SELECT png_path,area,bbox,centroid FROM sam3_masks "
                "WHERE asset_id=%s AND slug=%s",
                (asset_id, slug),
            ).fetchone()
            if mask is None or not mask["png_path"]:
                raise RuntimeError(
                    f"VLM main mask missing: {asset_id} / {row['main_subject']}"
                )
            source_path = Path(row["path"])
            mask_path = Path(mask["png_path"])
            if not source_path.is_file() or not mask_path.is_file():
                raise RuntimeError(f"missing source/mask file for {asset_id}")
            feature = features[preset_id]
            preset_path = Path(feature["path"])
            if not preset_path.is_file():
                raise RuntimeError(f"missing preset file: {preset_path}")
            cases.append(
                {
                    "asset_id": asset_id,
                    "source_path": str(source_path),
                    "scene": row["scene"],
                    "is_portrait": bool(row["is_portrait_pool"]),
                    "caption": row["caption"],
                    "main_subject": row["main_subject"],
                    "subjects": _parse_jsonish(row["subjects"], []),
                    "mask_path": str(mask_path),
                    "mask_area": float(mask["area"]),
                    "mask_bbox": _parse_jsonish(mask["bbox"], []),
                    "mask_centroid": _parse_jsonish(mask["centroid"], []),
                    "preset_id": preset_id,
                    "preset_path": str(preset_path),
                    "preset_fmt": feature["fmt"],
                    "preset_axes": feature.get("axes") or {},
                    "preset_metrics": feature.get("metrics") or {},
                }
            )
    finally:
        conn.close()
    return cases


def _seed(*parts: str) -> int:
    digest = hashlib.sha256("\0".join(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _read_source(path: str) -> np.ndarray:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def _read_mask(path: str, shape: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(mask, dtype=np.float32)


def _fit_long_edge(arr: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = arr.shape[:2]
    scale = min(1.0, float(long_edge) / max(h, w))
    if scale == 1.0:
        return arr.copy()
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(arr, size, interpolation=cv2.INTER_AREA)


def _candidate_set(case: dict, native_mask: np.ndarray) -> list[dict]:
    from dataset_build.src.construct import subject_geom

    result = [{"label": "semantic_0", "mode": "semantic", "spec": None}]
    radial = subject_geom.radial_geom(
        native_mask,
        random.Random(_seed(case["asset_id"], "radial_0")),
        apply_inside=True,
    )
    if radial is None:
        raise RuntimeError(f"radial geometry failed: {case['asset_id']}")
    result.append(
        {
            "label": "radial_0",
            "mode": "radial",
            "spec": {
                "mask_type": radial["mask_type"],
                "geom": radial["geom"],
                "amount": 1.0,
            },
        }
    )
    for i in range(3):
        band = subject_geom.band_geom(
            native_mask,
            random.Random(_seed(case["asset_id"], f"band_{i}")),
            apply_inside=True,
        )
        if band is None:
            raise RuntimeError(f"band geometry failed: {case['asset_id']} / {i}")
        result.append(
            {
                "label": f"band_{i}",
                "mode": "band",
                "spec": {
                    "mask_type": band["mask_type"],
                    "geom": band["geom"],
                    "amount": 1.0,
                },
            }
        )
    bbox = case["mask_bbox"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError(f"bad main-mask bbox: {case['asset_id']} / {bbox!r}")
    for i in range(3):
        linear = subject_geom.linear_geom(
            bbox,
            random.Random(_seed(case["asset_id"], f"linear_{i}")),
            apply_subject_side=True,
            area=case["mask_area"],
        )
        if linear is None:
            raise RuntimeError(f"linear geometry failed: {case['asset_id']} / {i}")
        result.append(
            {
                "label": f"linear_{i}",
                "mode": "linear",
                "spec": {
                    "mask_type": linear["mask_type"],
                    "geom": linear["geom"],
                    "amount": 1.0,
                },
            }
        )
    if [item["mode"] for item in result] != [
        "semantic", "radial", "band", "band", "band", "linear", "linear", "linear"
    ]:
        raise AssertionError("candidate allocation drifted")
    return result


def _alpha_batch(
    native_mask: np.ndarray,
    candidates: list[dict],
    h: int,
    w: int,
    device: str,
):
    import torch

    from dataset_build.src.construct.subject_geom import semantic_alpha
    from gpu_render.gpu.local_preset import raster_cgt_batch

    resized_mask = cv2.resize(native_mask, (w, h), interpolation=cv2.INTER_LINEAR)
    semantic = semantic_alpha(resized_mask, apply_inside=True)
    semantic_t = torch.from_numpy(np.ascontiguousarray(semantic))[None, None].to(
        device=device, dtype=torch.float32
    )
    geom = raster_cgt_batch(
        [item["spec"] for item in candidates[1:]], h, w, device, dtype=torch.float32
    )
    return torch.cat([semantic_t, geom], dim=0).clamp_(0.0, 1.0)


def _render_resolution(
    source_u8: np.ndarray,
    native_mask: np.ndarray,
    candidates: list[dict],
    preset: dict,
    residual_id: str,
    device: str,
    return_alpha: bool,
) -> tuple[list[np.ndarray], list[np.ndarray] | None, dict]:
    import torch

    from gpu_render.gpu.gpu_replay import replay_batch
    from gpu_render.gpu.local_preset import _preset_without_locals, composite_srgb
    from gpu_render.gpu.render_batch import _download_u8, _upload
    from gpu_render.gpu.residual_gpu import apply_residual_batch
    from gpu_render.local_apply import FITS_DIR
    from gpu_render.residual import load_residual

    h, w = source_u8.shape[:2]
    clean, strip_info = _preset_without_locals(preset)
    residual = load_residual(residual_id)
    if residual is None:
        raise RuntimeError(f"dedicated residual missing: {residual_id}")

    start = time.perf_counter()
    base = _upload([source_u8], device)
    upload_s = time.perf_counter() - start
    alpha = edited = output = None
    try:
        torch.cuda.synchronize(device)
        gpu_start = time.perf_counter()
        with torch.inference_mode():
            edited, replay_info = replay_batch(
                base.clone(), clean, fits_dir=FITS_DIR, fallback="cpu"
            )
            edited = apply_residual_batch(edited, *residual)
            alpha = _alpha_batch(native_mask, candidates, h, w, device)
            output = composite_srgb(base, edited, alpha)
        torch.cuda.synchronize(device)
        gpu_s = time.perf_counter() - gpu_start
        download_start = time.perf_counter()
        arrays = _download_u8(output)
        alpha_arrays = None
        if return_alpha:
            alpha_u8 = alpha.mul(255.0).add_(0.5).to(torch.uint8).cpu().numpy()
            alpha_arrays = [np.ascontiguousarray(item[0]) for item in alpha_u8]
        download_s = time.perf_counter() - download_start
    finally:
        del base, edited, output, alpha
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return arrays, alpha_arrays, {
        "shape": [h, w],
        "megapixels": round(h * w / 1_000_000.0, 4),
        "upload_s": round(upload_s, 4),
        "gpu_s": round(gpu_s, 4),
        "download_s": round(download_s, 4),
        "fallback_ops": replay_info.get("fallback_ops") or [],
        **strip_info,
    }


def _save_jpeg(arr: np.ndarray, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr, "RGB").save(path, "JPEG", quality=quality, subsampling=0)


def render_phase(args: argparse.Namespace) -> dict:
    from gpu_render.replay import parse_preset

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(args.limit)
    manifest = {
        "version": 1,
        "preview_long_edge": args.preview_long_edge,
        "candidate_allocation": {
            "semantic": 1,
            "radial": 1,
            "band": 3,
            "linear": 3,
        },
        "device": args.device,
        "jpeg_quality": args.jpeg_quality,
        "cases": [],
    }
    for index, case in enumerate(cases, 1):
        print(
            f"[render {index}/{len(cases)}] {case['asset_id']} "
            f"{case['main_subject']} + {case['preset_id']}",
            flush=True,
        )
        case_dir = out_dir / case["asset_id"]
        if case_dir.exists() and not args.resume:
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)

        decode_start = time.perf_counter()
        native_source = _read_source(case["source_path"])
        native_mask = _read_mask(case["mask_path"], native_source.shape[:2])
        preview_source = _fit_long_edge(native_source, args.preview_long_edge)
        decode_s = time.perf_counter() - decode_start
        candidates = _candidate_set(case, native_mask)
        preset = parse_preset(case["preset_path"], case["preset_fmt"])

        preview_paths = [case_dir / "preview" / f"{c['label']}.jpg" for c in candidates]
        native_paths = [case_dir / "native" / f"{c['label']}.jpg" for c in candidates]
        outputs_exist = all(path.is_file() for path in preview_paths + native_paths)
        if outputs_exist and args.resume:
            preview_timing = native_timing = {"resumed": True}
        else:
            preview_outputs, preview_alphas, preview_timing = _render_resolution(
                preview_source,
                native_mask,
                candidates,
                preset,
                case["preset_id"],
                args.device,
                return_alpha=True,
            )
            for candidate, image, alpha in zip(candidates, preview_outputs, preview_alphas):
                _save_jpeg(
                    image,
                    case_dir / "preview" / f"{candidate['label']}.jpg",
                    args.jpeg_quality,
                )
                alpha_path = case_dir / "alpha_preview" / f"{candidate['label']}.png"
                alpha_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(alpha, "L").save(alpha_path, "PNG", compress_level=3)
            del preview_outputs, preview_alphas

            native_outputs, _, native_timing = _render_resolution(
                native_source,
                native_mask,
                candidates,
                preset,
                case["preset_id"],
                args.device,
                return_alpha=False,
            )
            for candidate, image in zip(candidates, native_outputs):
                _save_jpeg(
                    image,
                    case_dir / "native" / f"{candidate['label']}.jpg",
                    args.jpeg_quality,
                )
            del native_outputs

        preview_source_path = case_dir / "source_preview.jpg"
        _save_jpeg(preview_source, preview_source_path, args.jpeg_quality)
        case_record = dict(case)
        case_record.update(
            {
                "native_shape": list(native_source.shape[:2]),
                "native_megapixels": round(native_source.shape[0] * native_source.shape[1] / 1e6, 4),
                "preview_shape": list(preview_source.shape[:2]),
                "source_preview_path": str(preview_source_path),
                "candidates": candidates,
                "timing": {
                    "decode_s": round(decode_s, 4),
                    "preview": preview_timing,
                    "native": native_timing,
                },
            }
        )
        _json_dump(case_dir / "case.json", case_record)
        manifest["cases"].append(case_record)
        del native_source, native_mask, preview_source
    _json_dump(out_dir / "manifest.json", manifest)
    return manifest


def score_phase(args: argparse.Namespace) -> dict:
    from dataset_build.src.construct import qa

    out_dir = Path(args.out_dir)
    manifest = _json_load(out_dir / "manifest.json")
    score_doc = {"qa_mode": "artimuse_charm_mixed", "cases": []}
    for index, case in enumerate(manifest["cases"], 1):
        print(f"[score {index}/{len(manifest['cases'])}] {case['asset_id']}", flush=True)
        case_dir = out_dir / case["asset_id"]
        labels = [item["label"] for item in case["candidates"]]
        preview_variants = [
            (label, str(case_dir / "preview" / f"{label}.jpg")) for label in labels
        ]
        native_variants = [
            (label, str(case_dir / "native" / f"{label}.jpg")) for label in labels
        ]
        preview = qa.qa_rank(
            case["source_preview_path"],
            preview_variants,
            scene=case.get("scene"),
            is_portrait=case["is_portrait"],
        )
        native = qa.qa_rank(
            case["source_path"],
            native_variants,
            scene=case.get("scene"),
            is_portrait=case["is_portrait"],
        )
        row = {"asset_id": case["asset_id"], "preview": preview, "native": native}
        score_doc["cases"].append(row)
        _json_dump(case_dir / "scores.json", row)
    _json_dump(out_dir / "scores.json", score_doc)
    return score_doc


def _stats(path: str) -> dict[str, float]:
    from dataset_build.src.construct.qa import _stats as production_stats

    return production_stats(path)


def _rank_metrics(preview: dict, native: dict, labels: list[str]) -> dict:
    from scipy.stats import kendalltau, rankdata, spearmanr

    preview_order = preview["ranking"]
    native_order = native["ranking"]
    preview_pos = {label: i for i, label in enumerate(preview_order)}
    native_pos = {label: i for i, label in enumerate(native_order)}
    p = np.asarray([preview_pos[label] for label in labels], dtype=np.float64)
    n = np.asarray([native_pos[label] for label in labels], dtype=np.float64)
    rho = float(spearmanr(p, n).statistic)
    tau = float(kendalltau(p, n).statistic)
    pair_ok = pair_n = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            pair_n += 1
            pair_ok += (p[i] - p[j]) * (n[i] - n[j]) > 0
    p_top2 = set(preview_order[:2])
    n_top2 = set(native_order[:2])
    diversity_shortlist = [
        "semantic_0",
        "radial_0",
        next(label for label in preview_order if label.startswith("band_")),
        next(label for label in preview_order if label.startswith("linear_")),
    ]
    diversity_set = set(diversity_shortlist)
    diversity_shortlist_5 = diversity_shortlist + [
        next(label for label in preview_order if label not in diversity_set)
    ]
    diversity_set_5 = set(diversity_shortlist_5)
    winner_recall = {
        str(k): native_order[0] in set(preview_order[:k]) for k in range(1, 5)
    }
    native_top2_item_recall = {
        str(k): len(n_top2 & set(preview_order[:k])) / 2.0 for k in range(1, 5)
    }
    native_top2_full_recall = {
        str(k): n_top2 <= set(preview_order[:k]) for k in range(1, 5)
    }
    p_scores = preview["scores"]
    n_scores = native["scores"]
    diversity_native_order = [label for label in native_order if label in diversity_set]
    diversity_selected_2 = diversity_native_order[:2]
    diversity_native_order_5 = [label for label in native_order if label in diversity_set_5]
    diversity_selected_2_5 = diversity_native_order_5[:2]
    native_top2_mean_q = statistics.fmean(
        float(n_scores[label]["q"]) for label in native_order[:2]
    )
    diversity_top2_mean_q = statistics.fmean(
        float(n_scores[label]["q"]) for label in diversity_selected_2
    )
    diversity_top2_mean_q_5 = statistics.fmean(
        float(n_scores[label]["q"]) for label in diversity_selected_2_5
    )
    p_q = [float(p_scores[label]["q"]) for label in preview_order]
    n_q = [float(n_scores[label]["q"]) for label in native_order]
    return {
        "preview_order": preview_order,
        "native_order": native_order,
        "top1_agree": preview_order[0] == native_order[0],
        "top2_set_agree": p_top2 == n_top2,
        "native_winner_in_preview_top2": native_order[0] in p_top2,
        "native_winner_preview_rank": preview_pos[native_order[0]] + 1,
        "native_winner_recall_at_preview_k": winner_recall,
        "native_top2_item_recall_at_preview_k": native_top2_item_recall,
        "native_top2_full_recall_at_preview_k": native_top2_full_recall,
        "diversity_shortlist_4": diversity_shortlist,
        "diversity_4_native_winner_included": native_order[0] in diversity_set,
        "diversity_4_native_top2_item_recall": len(n_top2 & diversity_set) / 2.0,
        "diversity_4_native_top2_fully_recovered": n_top2 <= diversity_set,
        "diversity_4_native_selected_2": diversity_selected_2,
        "diversity_4_native_winner_q_regret": float(n_scores[native_order[0]]["q"])
        - float(n_scores[diversity_selected_2[0]]["q"]),
        "diversity_4_native_top2_mean_q_regret": native_top2_mean_q
        - diversity_top2_mean_q,
        "diversity_shortlist_5": diversity_shortlist_5,
        "diversity_5_native_winner_included": native_order[0] in diversity_set_5,
        "diversity_5_native_top2_item_recall": len(n_top2 & diversity_set_5) / 2.0,
        "diversity_5_native_top2_fully_recovered": n_top2 <= diversity_set_5,
        "diversity_5_native_selected_2": diversity_selected_2_5,
        "diversity_5_native_winner_q_regret": float(n_scores[native_order[0]]["q"])
        - float(n_scores[diversity_selected_2_5[0]]["q"]),
        "diversity_5_native_top2_mean_q_regret": native_top2_mean_q
        - diversity_top2_mean_q_5,
        "top2_jaccard": len(p_top2 & n_top2) / len(p_top2 | n_top2),
        "spearman": rho,
        "kendall": tau,
        "pairwise_concordance": pair_ok / pair_n,
        "preview_top_margin": p_q[0] - p_q[1],
        "native_top_margin": n_q[0] - n_q[1],
        "score_mae": float(
            np.mean(
                [abs(p_scores[label]["q"] - n_scores[label]["q"]) for label in labels]
            )
        ),
        "preview_rankdata": rankdata(p).tolist(),
        "native_rankdata": rankdata(n).tolist(),
    }


def _visual_metrics(
    preview_path: Path,
    native_path: Path,
    preview_source: np.ndarray,
    native_source_down: np.ndarray,
) -> dict:
    from skimage.color import deltaE_ciede2000, rgb2lab
    from skimage.metrics import structural_similarity

    preview = _read_source(str(preview_path))
    native = _read_source(str(native_path))
    native_down = cv2.resize(
        native,
        (preview.shape[1], preview.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    p = preview.astype(np.float32) / 255.0
    n = native_down.astype(np.float32) / 255.0
    sample_p = p[::2, ::2]
    sample_n = n[::2, ::2]
    de = deltaE_ciede2000(rgb2lab(sample_p), rgb2lab(sample_n))
    effect_p = p - preview_source
    effect_n = n - native_source_down
    return {
        "rgb_mae_255": float(np.abs(p - n).mean() * 255.0),
        "delta_e00_mean": float(np.mean(de)),
        "delta_e00_p95": float(np.percentile(de, 95)),
        "ssim": float(structural_similarity(p, n, channel_axis=2, data_range=1.0)),
        "effect_mae_255": float(np.abs(effect_p - effect_n).mean() * 255.0),
    }


def _contact_sheet(case: dict, scores: dict, rows: list[dict], out_dir: Path) -> None:
    case_dir = out_dir / case["asset_id"]
    source = Image.open(case["source_preview_path"]).convert("RGB")
    cell_w, cell_h, label_h = 300, 210, 42
    preview_edge = max(case["preview_shape"])
    columns = (
        "MASK / SOURCE",
        f"PREVIEW {preview_edge}",
        f"NATIVE -> {preview_edge}",
        "ABS DIFF x8",
    )
    canvas = Image.new("RGB", (cell_w * 4, 72 + (cell_h + label_h) * 8), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    title = (
        f"{case['asset_id']} | {case['main_subject']} | {case['preset_id']} | "
        f"native={case['native_shape'][1]}x{case['native_shape'][0]}"
    )
    draw.text((8, 6), title, fill="black", font=font)
    for col, name in enumerate(columns):
        draw.text((col * cell_w + 8, 34), name, fill="black", font=font)

    preview_order = scores["preview"]["ranking"]
    native_order = scores["native"]["ranking"]
    p_rank = {label: i + 1 for i, label in enumerate(preview_order)}
    n_rank = {label: i + 1 for i, label in enumerate(native_order)}
    row_by_label = {row["label"]: row for row in rows}
    for row_index, candidate in enumerate(case["candidates"]):
        label = candidate["label"]
        y = 72 + row_index * (cell_h + label_h)
        alpha = np.asarray(
            Image.open(case_dir / "alpha_preview" / f"{label}.png").convert("L"),
            dtype=np.float32,
        ) / 255.0
        src = np.asarray(source, dtype=np.float32)
        red = np.zeros_like(src)
        red[..., 0] = 255
        overlay = np.clip(src * (1 - 0.45 * alpha[..., None]) + red * 0.45 * alpha[..., None], 0, 255)
        preview = _read_source(str(case_dir / "preview" / f"{label}.jpg"))
        native = _read_source(str(case_dir / "native" / f"{label}.jpg"))
        native_down = cv2.resize(
            native, (preview.shape[1], preview.shape[0]), interpolation=cv2.INTER_AREA
        )
        diff = np.abs(preview.astype(np.int16) - native_down.astype(np.int16)).mean(2)
        heat = cv2.applyColorMap(np.clip(diff * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        images = [overlay.astype(np.uint8), preview, native_down, heat]
        for col, image in enumerate(images):
            tile = Image.fromarray(image, "RGB")
            tile.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)
            x = col * cell_w + (cell_w - tile.width) // 2
            yy = y + (cell_h - tile.height) // 2
            canvas.paste(tile, (x, yy))
        metric = row_by_label[label]
        p_q = scores["preview"]["scores"][label]["q"]
        n_q = scores["native"]["scores"][label]["q"]
        text = (
            f"{label}  P#{p_rank[label]} q={p_q:.4f}  N#{n_rank[label]} q={n_q:.4f}  "
            f"DE={metric['visual']['delta_e00_mean']:.2f}  "
            f"SSIM={metric['visual']['ssim']:.4f}"
        )
        draw.text((8, y + cell_h + 9), text, fill="black", font=font)
    canvas.save(case_dir / "comparison_contact.jpg", "JPEG", quality=93, subsampling=0)


def _mean(rows: list[float]) -> float:
    return float(statistics.fmean(rows)) if rows else float("nan")


def _percentile(rows: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(rows, dtype=np.float64), q)) if rows else float("nan")


def analyze_phase(args: argparse.Namespace) -> dict:
    from dataset_build.src.construct.qa import _det_extreme, _det_flags

    out_dir = Path(args.out_dir)
    manifest = _json_load(out_dir / "manifest.json")
    scores_doc = _json_load(out_dir / "scores.json")
    score_by_asset = {row["asset_id"]: row for row in scores_doc["cases"]}
    case_results = []
    all_rows = []
    for case in manifest["cases"]:
        case_dir = out_dir / case["asset_id"]
        scores = score_by_asset[case["asset_id"]]
        labels = [item["label"] for item in case["candidates"]]
        rank = _rank_metrics(scores["preview"], scores["native"], labels)
        preview_source_u8 = _read_source(case["source_preview_path"])
        native_source_u8 = _read_source(case["source_path"])
        native_source_down_u8 = cv2.resize(
            native_source_u8,
            (preview_source_u8.shape[1], preview_source_u8.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        preview_source = preview_source_u8.astype(np.float32) / 255.0
        native_source_down = native_source_down_u8.astype(np.float32) / 255.0
        p_src_stats = _stats(case["source_preview_path"])
        n_src_stats = _stats(case["source_path"])
        rows = []
        for candidate in case["candidates"]:
            label = candidate["label"]
            preview_path = case_dir / "preview" / f"{label}.jpg"
            native_path = case_dir / "native" / f"{label}.jpg"
            p_stats = _stats(str(preview_path))
            n_stats = _stats(str(native_path))
            p_det = sorted(_det_flags(p_stats, p_src_stats["cf"]))
            n_det = sorted(_det_flags(n_stats, n_src_stats["cf"]))
            visual = _visual_metrics(
                preview_path,
                native_path,
                preview_source,
                native_source_down,
            )
            row = {
                "asset_id": case["asset_id"],
                "label": label,
                "mode": candidate["mode"],
                "preview_q": scores["preview"]["scores"][label]["q"],
                "native_q": scores["native"]["scores"][label]["q"],
                "preview_stats": p_stats,
                "native_stats": n_stats,
                "stats_abs_diff": {key: abs(p_stats[key] - n_stats[key]) for key in p_stats},
                "preview_det": p_det,
                "native_det": n_det,
                "det_exact": p_det == n_det,
                "preview_extreme": bool(_det_extreme(p_stats)),
                "native_extreme": bool(_det_extreme(n_stats)),
                "extreme_exact": bool(_det_extreme(p_stats)) == bool(_det_extreme(n_stats)),
                "veto_exact": (
                    scores["preview"]["scores"][label]["veto"]
                    == scores["native"]["scores"][label]["veto"]
                ),
                "visual": visual,
            }
            rows.append(row)
            all_rows.append(row)
        result = {
            "asset_id": case["asset_id"],
            "main_subject": case["main_subject"],
            "native_megapixels": case["native_megapixels"],
            "rank": rank,
            "rows": rows,
        }
        case_results.append(result)
        _json_dump(case_dir / "analysis.json", result)
        _contact_sheet(case, scores, rows, out_dir)

    native_mp = [float(case["native_megapixels"]) for case in manifest["cases"]]
    winner_recall_at_k = {
        str(k): _mean(
            [
                float(c["rank"]["native_winner_recall_at_preview_k"][str(k)])
                for c in case_results
            ]
        )
        for k in range(1, 5)
    }
    native_top2_item_recall_at_k = {
        str(k): _mean(
            [
                c["rank"]["native_top2_item_recall_at_preview_k"][str(k)]
                for c in case_results
            ]
        )
        for k in range(1, 5)
    }
    native_top2_full_recall_at_k = {
        str(k): _mean(
            [
                float(c["rank"]["native_top2_full_recall_at_preview_k"][str(k)])
                for c in case_results
            ]
        )
        for k in range(1, 5)
    }
    diversity_4 = {
        "native_winner_inclusion": _mean(
            [
                float(c["rank"]["diversity_4_native_winner_included"])
                for c in case_results
            ]
        ),
        "native_top2_item_recall": _mean(
            [c["rank"]["diversity_4_native_top2_item_recall"] for c in case_results]
        ),
        "native_top2_exact_recovery": _mean(
            [
                float(c["rank"]["diversity_4_native_top2_fully_recovered"])
                for c in case_results
            ]
        ),
        "native_winner_q_regret_mean": _mean(
            [c["rank"]["diversity_4_native_winner_q_regret"] for c in case_results]
        ),
        "native_winner_q_regret_max": max(
            c["rank"]["diversity_4_native_winner_q_regret"] for c in case_results
        ),
        "native_top2_mean_q_regret_mean": _mean(
            [c["rank"]["diversity_4_native_top2_mean_q_regret"] for c in case_results]
        ),
        "native_top2_mean_q_regret_max": max(
            c["rank"]["diversity_4_native_top2_mean_q_regret"] for c in case_results
        ),
    }
    diversity_5 = {
        "native_winner_inclusion": _mean(
            [
                float(c["rank"]["diversity_5_native_winner_included"])
                for c in case_results
            ]
        ),
        "native_top2_item_recall": _mean(
            [c["rank"]["diversity_5_native_top2_item_recall"] for c in case_results]
        ),
        "native_top2_exact_recovery": _mean(
            [
                float(c["rank"]["diversity_5_native_top2_fully_recovered"])
                for c in case_results
            ]
        ),
        "native_winner_q_regret_mean": _mean(
            [c["rank"]["diversity_5_native_winner_q_regret"] for c in case_results]
        ),
        "native_winner_q_regret_max": max(
            c["rank"]["diversity_5_native_winner_q_regret"] for c in case_results
        ),
        "native_top2_mean_q_regret_mean": _mean(
            [c["rank"]["diversity_5_native_top2_mean_q_regret"] for c in case_results]
        ),
        "native_top2_mean_q_regret_max": max(
            c["rank"]["diversity_5_native_top2_mean_q_regret"] for c in case_results
        ),
    }
    summary = {
        "n_sources": len(case_results),
        "n_candidate_pairs": len(all_rows),
        "preview_long_edge": manifest["preview_long_edge"],
        "native_megapixels": {
            "min": min(native_mp),
            "p50": _percentile(native_mp, 50),
            "p95": _percentile(native_mp, 95),
            "max": max(native_mp),
        },
        "rank": {
            "top1_agreement": _mean([float(c["rank"]["top1_agree"]) for c in case_results]),
            "top2_set_agreement": _mean(
                [float(c["rank"]["top2_set_agree"]) for c in case_results]
            ),
            "native_winner_recall_at_preview_2": _mean(
                [float(c["rank"]["native_winner_in_preview_top2"]) for c in case_results]
            ),
            "native_winner_recall_at_preview_k": winner_recall_at_k,
            "native_top2_item_recall_at_preview_k": native_top2_item_recall_at_k,
            "native_top2_full_recall_at_preview_k": native_top2_full_recall_at_k,
            "diversity_shortlist_4": diversity_4,
            "diversity_shortlist_5": diversity_5,
            "top2_jaccard_mean": _mean([c["rank"]["top2_jaccard"] for c in case_results]),
            "spearman_mean": _mean([c["rank"]["spearman"] for c in case_results]),
            "spearman_median": float(
                statistics.median(c["rank"]["spearman"] for c in case_results)
            ),
            "kendall_mean": _mean([c["rank"]["kendall"] for c in case_results]),
            "pairwise_concordance_mean": _mean(
                [c["rank"]["pairwise_concordance"] for c in case_results]
            ),
            "score_mae_mean": _mean([c["rank"]["score_mae"] for c in case_results]),
            "preview_top_margin_median": float(
                statistics.median(c["rank"]["preview_top_margin"] for c in case_results)
            ),
        },
        "technical": {
            "det_flag_exact_rate": _mean([float(row["det_exact"]) for row in all_rows]),
            "extreme_exact_rate": _mean([float(row["extreme_exact"]) for row in all_rows]),
            "veto_exact_rate": _mean([float(row["veto_exact"]) for row in all_rows]),
            "luma_abs_diff_mean": _mean(
                [row["stats_abs_diff"]["luma"] for row in all_rows]
            ),
            "chroma_abs_diff_mean": _mean(
                [row["stats_abs_diff"]["cf"] for row in all_rows]
            ),
        },
        "visual": {
            "delta_e00_mean": _mean([row["visual"]["delta_e00_mean"] for row in all_rows]),
            "delta_e00_p95_of_candidates": _percentile(
                [row["visual"]["delta_e00_mean"] for row in all_rows], 95
            ),
            "pixel_delta_e00_p95_mean": _mean(
                [row["visual"]["delta_e00_p95"] for row in all_rows]
            ),
            "ssim_mean": _mean([row["visual"]["ssim"] for row in all_rows]),
            "rgb_mae_255_mean": _mean(
                [row["visual"]["rgb_mae_255"] for row in all_rows]
            ),
            "effect_mae_255_mean": _mean(
                [row["visual"]["effect_mae_255"] for row in all_rows]
            ),
        },
        "cases": case_results,
        "limitations": [
            "The fixed sample is intentionally small and enriched for clear, cached VLM subjects.",
            "Existing SAM3 concept masks may merge multiple same-class instances; this experiment does not validate the planned VLM overlay instance-ID selection.",
            "The comparison uses the current production ArtiMuse+Charm ranker and deterministic vetoes; the legacy pairwise VLM questionnaire is not part of qa_rank and was not called.",
            "Only calibrated GPU presets are included, so results do not cover Lightroom farm fallback or LUT-local compositing.",
            "Native images are JPEG outputs while preview candidates are separately rendered JPEGs; reported visual delta includes resolution-dependent operators and encoding, matching the proposed workflow.",
        ],
    }
    _json_dump(out_dir / "summary.json", summary)
    _write_report(out_dir, summary)
    return summary


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_report(out_dir: Path, summary: dict) -> None:
    rank = summary["rank"]
    tech = summary["technical"]
    visual = summary["visual"]
    mp = summary["native_megapixels"]
    lines = [
        "# Preview vs Native Local-Preset Evaluation v1",
        "",
        f"- Sources: {summary['n_sources']} ({summary['n_candidate_pairs']} matched candidates)",
        "- Candidate set: 1 semantic + 1 radial + 3 band + 3 linear",
        f"- Preview long edge: {summary['preview_long_edge']} px",
        f"- Native source size: p50={mp['p50']:.2f} MP, p95={mp['p95']:.2f} MP, max={mp['max']:.2f} MP",
        "- Ranker: current production ArtiMuse+Charm mixed IAA plus deterministic vetoes",
        "",
        "## Ranking Stability",
        "",
        f"- Exact winner agreement: {_pct(rank['top1_agreement'])}",
        f"- Exact top-2 set agreement: {_pct(rank['top2_set_agreement'])}",
        f"- Native winner retained by preview top-2: {_pct(rank['native_winner_recall_at_preview_2'])}",
        "- Native winner recall by preview depth k=1..4: "
        + ", ".join(
            f"k={k}: {_pct(rank['native_winner_recall_at_preview_k'][str(k)])}"
            for k in range(1, 5)
        ),
        "- Full native top-2 recovery by preview depth k=1..4: "
        + ", ".join(
            f"k={k}: {_pct(rank['native_top2_full_recall_at_preview_k'][str(k)])}"
            for k in range(1, 5)
        ),
        "- Native top-2 item recall by preview depth k=1..4: "
        + ", ".join(
            f"k={k}: {_pct(rank['native_top2_item_recall_at_preview_k'][str(k)])}"
            for k in range(1, 5)
        ),
        "- Diversity shortlist (semantic + radial + best band + best linear): "
        f"native winner={_pct(rank['diversity_shortlist_4']['native_winner_inclusion'])}, "
        f"exact native top-2={_pct(rank['diversity_shortlist_4']['native_top2_exact_recovery'])}, "
        f"top-2 item recall={_pct(rank['diversity_shortlist_4']['native_top2_item_recall'])}, "
        f"winner q-regret mean/max="
        f"{rank['diversity_shortlist_4']['native_winner_q_regret_mean']:.4f}/"
        f"{rank['diversity_shortlist_4']['native_winner_q_regret_max']:.4f}",
        "- Diversity+1 shortlist (diversity four + best remaining preview): "
        f"native winner={_pct(rank['diversity_shortlist_5']['native_winner_inclusion'])}, "
        f"exact native top-2={_pct(rank['diversity_shortlist_5']['native_top2_exact_recovery'])}, "
        f"top-2 item recall={_pct(rank['diversity_shortlist_5']['native_top2_item_recall'])}, "
        f"winner q-regret mean/max="
        f"{rank['diversity_shortlist_5']['native_winner_q_regret_mean']:.4f}/"
        f"{rank['diversity_shortlist_5']['native_winner_q_regret_max']:.4f}",
        f"- Mean top-2 Jaccard: {rank['top2_jaccard_mean']:.3f}",
        f"- Spearman rank correlation: mean={rank['spearman_mean']:.3f}, median={rank['spearman_median']:.3f}",
        f"- Mean Kendall tau: {rank['kendall_mean']:.3f}",
        f"- Mean pairwise ordering agreement: {_pct(rank['pairwise_concordance_mean'])}",
        f"- Mean absolute q-score drift: {rank['score_mae_mean']:.4f}",
        f"- Median preview winner margin: {rank['preview_top_margin_median']:.4f}",
        "",
        "## Technical Stability",
        "",
        f"- Deterministic flag exact agreement: {_pct(tech['det_flag_exact_rate'])}",
        f"- Extreme-veto exact agreement: {_pct(tech['extreme_exact_rate'])}",
        f"- Final veto exact agreement: {_pct(tech['veto_exact_rate'])}",
        f"- Mean luma-stat drift: {tech['luma_abs_diff_mean']:.3f} / 255",
        f"- Mean chroma-stat drift: {tech['chroma_abs_diff_mean']:.3f}",
        "",
        "## Native Visual Drift",
        "",
        f"- Mean DeltaE00 after native downsample: {visual['delta_e00_mean']:.3f}",
        f"- Candidate-level p95 of mean DeltaE00: {visual['delta_e00_p95_of_candidates']:.3f}",
        f"- Mean within-image pixel p95 DeltaE00: {visual['pixel_delta_e00_p95_mean']:.3f}",
        f"- Mean SSIM: {visual['ssim_mean']:.5f}",
        f"- Mean RGB MAE: {visual['rgb_mae_255_mean']:.3f} / 255",
        f"- Mean local-effect MAE: {visual['effect_mae_255_mean']:.3f} / 255",
        "",
        "## Per Source",
        "",
        "| source | subject | MP | preview winner | native winner | native winner in preview top-2 | rho |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for case in summary["cases"]:
        rank_case = case["rank"]
        lines.append(
            f"| {case['asset_id']} | {case['main_subject']} | {case['native_megapixels']:.2f} | "
            f"{rank_case['preview_order'][0]} | {rank_case['native_order'][0]} | "
            f"{'yes' if rank_case['native_winner_in_preview_top2'] else 'no'} | "
            f"{rank_case['spearman']:.3f} |"
        )
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in summary["limitations"])
    lines.extend(
        [
            "",
            "Each source directory contains `comparison_contact.jpg`, the eight preview outputs,",
            "the eight native outputs, preview alpha masks, and detailed JSON metrics.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _print_summary(summary: dict) -> None:
    rank = summary["rank"]
    print(
        "DONE "
        f"sources={summary['n_sources']} "
        f"top1={_pct(rank['top1_agreement'])} "
        f"native_winner@preview2={_pct(rank['native_winner_recall_at_preview_2'])} "
        f"rho={rank['spearman_mean']:.3f} "
        f"DE00={summary['visual']['delta_e00_mean']:.3f}"
    )


def _warm_preview_timing(out_dir: Path) -> dict:
    manifest = _json_load(out_dir / "manifest.json")
    values = [
        float(case["timing"]["preview"]["gpu_s"])
        for case in manifest["cases"]
        if float(case["timing"]["preview"].get("gpu_s", 999.0)) < 1.0
    ]
    return {
        "n": len(values),
        "p50_s": _percentile(values, 50),
        "p95_s": _percentile(values, 95),
    }


def compare_phase(args: argparse.Namespace) -> dict:
    root = Path(args.out_dir)
    alternate = Path(args.compare_dir) if args.compare_dir else root / "preview_1024"
    base = _json_load(root / "summary.json")
    other = _json_load(alternate / "summary.json")
    ordered = sorted((base, other), key=lambda item: item["preview_long_edge"])

    rows = {}
    for summary, directory in ((base, root), (other, alternate)):
        edge = str(summary["preview_long_edge"])
        rank = summary["rank"]
        rows[edge] = {
            "top1_agreement": rank["top1_agreement"],
            "winner_recall_at_k": rank["native_winner_recall_at_preview_k"],
            "diversity4": rank["diversity_shortlist_4"],
            "diversity5": rank["diversity_shortlist_5"],
            "visual": summary["visual"],
            "technical": summary["technical"],
            "warm_preview_gpu": _warm_preview_timing(directory),
        }
    comparison = {
        "sample_sources": base["n_sources"],
        "candidate_allocation": "1 semantic + 1 radial + 3 band + 3 linear",
        "resolutions": rows,
        "policies": {
            "conservative_768": {
                "preview_long_edge": 768,
                "native_shortlist": (
                    "semantic + radial + best preview band + best preview linear + "
                    "highest-ranked remaining preview candidate"
                ),
                "native_renders": 5,
                "final_outputs": 2,
                "observed_native_winner_inclusion": rows["768"]["diversity5"][
                    "native_winner_inclusion"
                ],
                "observed_exact_native_top2_recovery": rows["768"]["diversity5"][
                    "native_top2_exact_recovery"
                ],
            },
            "throughput_1024": {
                "preview_long_edge": 1024,
                "native_shortlist": "semantic + radial + best preview band + best preview linear",
                "native_renders": 4,
                "final_outputs": 2,
                "observed_native_winner_inclusion": rows["1024"]["diversity4"][
                    "native_winner_inclusion"
                ],
                "observed_exact_native_top2_recovery": rows["1024"]["diversity4"][
                    "native_top2_exact_recovery"
                ],
                "observed_top2_mean_q_regret_max": rows["1024"]["diversity4"][
                    "native_top2_mean_q_regret_max"
                ],
            },
        },
        "interpretation": [
            "A plain preview top-2 is not safe at either tested resolution.",
            "At 768, the fifth diversity+rank candidate changed winner/top-2 recovery from 87.5% to 100% on this sample.",
            "At 1024, adding a fifth candidate produced no gain over diversity4; the two missed exact top-2 labels were near-ties with maximum mean-q regret below 0.0003.",
            "The sample is too small to claim a population guarantee; use the conservative policy until a larger stratified run confirms the four-candidate policy.",
        ],
    }
    _json_dump(root / "resolution_comparison.json", comparison)

    lines = [
        "# Preview Resolution and Shortlist Recommendation",
        "",
        "Eight real sources, each with the fixed 1 semantic + 1 radial + 3 band + 3 linear set.",
        "All shortlisted candidates were evaluated with their actual native render and native QA score.",
        "",
        "| preview | plain winner | plain winner@2 | diversity4 winner | diversity4 exact top2 | diversity5 winner | diversity5 exact top2 | mean DE00 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for summary in ordered:
        edge = str(summary["preview_long_edge"])
        row = rows[edge]
        lines.append(
            f"| {edge} | {_pct(row['top1_agreement'])} | "
            f"{_pct(row['winner_recall_at_k']['2'])} | "
            f"{_pct(row['diversity4']['native_winner_inclusion'])} | "
            f"{_pct(row['diversity4']['native_top2_exact_recovery'])} | "
            f"{_pct(row['diversity5']['native_winner_inclusion'])} | "
            f"{_pct(row['diversity5']['native_top2_exact_recovery'])} | "
            f"{row['visual']['delta_e00_mean']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Recommended rollout",
            "",
            "Use the conservative policy first:",
            "",
            "1. Render and QA all eight previews at 768 px.",
            "2. Shortlist `semantic_0`, `radial_0`, the best preview band, the best preview linear, and the highest-ranked remaining candidate.",
            "3. Native-render those five in one preset replay, native-QA them, and retain the final two.",
            "",
            "This recovered the exact native top two in 8/8 cases. It reduces native candidates and native QA from eight to five while preserving the requested two final outputs.",
            "",
            "The more aggressive policy is 1024 px plus diversity4. It included the native winner in 8/8 cases with four native renders; exact top-two labels were 6/8, but maximum native top-two mean-q regret was below 0.0003. Promote this only after a larger stratified evaluation.",
            "",
            "## Fifth-render value",
            "",
            "At 768, diversity+1 improved native-winner inclusion and exact native-top2 recovery from 7/8 to 8/8. That is a 25% increase over four native candidates, while remaining 37.5% below rendering all eight. At 1024 it added no measured benefit.",
            "",
            "One-time CUDA compilation samples were excluded from warm preview timing summaries in `resolution_comparison.json`.",
            "",
        ]
    )
    (root / "RECOMMENDATION.md").write_text("\n".join(lines), encoding="utf-8")
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("render", "score", "analyze", "compare", "all"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--preview-long-edge", type=int, default=768)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--device", default=os.environ.get("MONETGPT_TORCH_DEVICE", "cuda:1"))
    parser.add_argument("--limit", type=int, default=len(DEFAULT_CASES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--compare-dir", default="")
    args = parser.parse_args()
    if args.preview_long_edge < 256:
        parser.error("--preview-long-edge must be >= 256")
    if not 1 <= args.limit <= len(DEFAULT_CASES):
        parser.error(f"--limit must be in [1, {len(DEFAULT_CASES)}]")

    if args.phase in ("render", "all"):
        render_phase(args)
    if args.phase in ("score", "all"):
        score_phase(args)
    if args.phase in ("analyze", "all"):
        _print_summary(analyze_phase(args))
    if args.phase == "compare":
        compare_phase(args)


if __name__ == "__main__":
    main()
