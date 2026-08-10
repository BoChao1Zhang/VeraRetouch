from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from .common import (
    append_jsonl,
    atomic_write_json,
    load_config,
    manifests_dir,
    output_root,
    read_jsonl,
)
from .data import (
    FixedPairDataset,
    PackedLutStore,
    collate_images,
    load_lut_manifest,
    native_lattice,
    render_lut_batch,
)
from .prepare import prepare
from .train import _autocast, _build_model, _resolve_device, _seed_everything


def _load_model(
    model_name: str,
    config: dict[str, Any],
    num_styles: int,
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    model = _build_model(model_name, num_styles, config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("model_name") != model_name:
        raise RuntimeError(f"checkpoint model mismatch: {checkpoint_path}")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model


def _render_model(
    model: torch.nn.Module,
    images: torch.Tensor,
    style_indices: torch.Tensor,
    chunk_size: int,
    config: dict[str, Any],
) -> torch.Tensor:
    points = images.permute(0, 2, 3, 1).reshape(images.shape[0], -1, 3)
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, points.shape[1], chunk_size):
            with _autocast(config):
                value = model.forward_points(points[:, start : start + chunk_size], style_indices)
            outputs.append(value.float())
    rendered = torch.cat(outputs, dim=1)
    return rendered.reshape(images.shape[0], images.shape[2], images.shape[3], 3).permute(0, 3, 1, 2)


def _image_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    difference = prediction.astype(np.float64) - target.astype(np.float64)
    mse = float(np.mean(difference**2))
    prediction_lab = rgb2lab(np.clip(prediction, 0.0, 1.0))
    target_lab = rgb2lab(np.clip(target, 0.0, 1.0))
    delta_e = deltaE_ciede2000(prediction_lab, target_lab)
    return {
        "l1": float(np.mean(np.abs(difference))),
        "psnr": float("inf") if mse == 0.0 else float(10.0 * math.log10(1.0 / mse)),
        "ssim": float(
            structural_similarity(
                target,
                prediction,
                channel_axis=-1,
                data_range=1.0,
            )
        ),
        "delta_e_00_mean": float(np.mean(delta_e)),
        "delta_e_00_p95": float(np.percentile(delta_e, 95)),
    }


def _save_comparison(
    path: Path,
    source: np.ndarray,
    target: np.ndarray,
    cglut: np.ndarray,
    vera: np.ndarray,
    sample_id: str,
) -> None:
    arrays = [source, target, cglut, vera]
    labels = ["Source", "LUT GT", "CGLUT", "Vera"]
    images = [Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255.0)) for array in arrays]
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + 28
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    offset = 0
    for image, label in zip(images, labels, strict=True):
        sheet.paste(image, (offset, 28))
        draw.text((offset + 6, 7), label, fill="black")
        offset += image.width
    path.mkdir(parents=True, exist_ok=True)
    sheet.save(path / f"{sample_id}.jpg", quality=94)


def _bootstrap_difference(
    rows: list[dict[str, Any]], metric: str, replicates: int, seed: int
) -> dict[str, float]:
    cglut = np.asarray([row["cglut"][metric] for row in rows], dtype=np.float64)
    vera = np.asarray([row["vera"][metric] for row in rows], dtype=np.float64)
    if metric in {"l1", "delta_e_00_mean", "delta_e_00_p95"}:
        paired = vera - cglut
        orientation = "positive means CGLUT lower/better"
    else:
        paired = cglut - vera
        orientation = "positive means CGLUT higher/better"
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(rows), size=(replicates, len(rows)))
    bootstrap = paired[indices].mean(axis=1)
    return {
        "difference": float(paired.mean()),
        "ci95_low": float(np.percentile(bootstrap, 2.5)),
        "ci95_high": float(np.percentile(bootstrap, 97.5)),
        "orientation": orientation,
    }


def _aggregate_natural(
    rows: list[dict[str, Any]], replicates: int, seed: int
) -> dict[str, Any]:
    metrics = ["l1", "psnr", "ssim", "delta_e_00_mean", "delta_e_00_p95"]
    summary: dict[str, Any] = {"samples": len(rows), "methods": {}, "paired": {}}
    for method in ("cglut", "vera"):
        summary["methods"][method] = {
            metric: float(np.mean([row[method][metric] for row in rows]))
            for metric in metrics
        }
    for metric in metrics:
        summary["paired"][metric] = _bootstrap_difference(rows, metric, replicates, seed)

    by_major: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_major.setdefault(row["taxonomy_major"], []).append(row)
    summary["taxonomy_macro_delta_e_00"] = {
        method: float(
            np.mean(
                [
                    np.mean([row[method]["delta_e_00_mean"] for row in major_rows])
                    for major_rows in by_major.values()
                ]
            )
        )
        for method in ("cglut", "vera")
    }
    summary["taxonomy_major_count"] = len(by_major)
    return summary


def _natural_evaluation(
    config: dict[str, Any],
    models: dict[str, torch.nn.Module],
    luts: list[Any],
    store: PackedLutStore,
    device: torch.device,
    metrics_root: Path,
    reports_root: Path,
    *,
    smoke: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = list(read_jsonl(manifests_dir(config) / "test_pairs.jsonl"))
    if smoke:
        pairs = pairs[:2]
    dataset = FixedPairDataset(
        pairs,
        short_edge=int(config["data"]["image_short_edge"]),
        long_edge=int(config["data"]["image_long_edge"]),
    )
    loader = DataLoader(
        dataset,
        batch_size=2 if smoke else 4,
        shuffle=False,
        num_workers=0 if smoke else min(4, int(config["data"]["num_workers"])),
        pin_memory=True,
        collate_fn=collate_images,
    )
    per_sample_path = metrics_root / "natural_per_sample.jsonl"
    per_sample_path.unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    qualitative_limit = int(config["evaluation"]["qualitative_count"])
    chunk_size = int(config["evaluation"]["image_chunk_points"])
    for batch_index, batch in enumerate(loader, 1):
        images = batch["images"].to(device, non_blocking=True)
        style_indices = batch["style_indices"].to(device, non_blocking=True)
        with torch.inference_mode():
            targets = render_lut_batch(images, style_indices, luts, store)
        predictions = {
            name: _render_model(model, images, style_indices, chunk_size, config)
            for name, model in models.items()
        }
        for local_index, metadata in enumerate(batch["metadata"]):
            height, width = batch["sizes"][local_index]
            source = (
                images[local_index, :, :height, :width]
                .permute(1, 2, 0)
                .float()
                .cpu()
                .numpy()
            )
            target = (
                targets[local_index, :, :height, :width]
                .permute(1, 2, 0)
                .float()
                .cpu()
                .numpy()
            )
            arrays = {
                name: value[local_index, :, :height, :width]
                .permute(1, 2, 0)
                .float()
                .cpu()
                .numpy()
                for name, value in predictions.items()
            }
            row = {
                "sample_id": metadata["sample_id"],
                "source_id": metadata["source_id"],
                "source_cluster": metadata["source_cluster"],
                "style_index": int(metadata["style_index"]),
                "preset_id": metadata["preset_id"],
                "lut_content_hash": metadata["lut_content_hash"],
                "taxonomy_major": metadata["taxonomy_major"],
                "taxonomy_minor": metadata["taxonomy_minor"],
                "height": height,
                "width": width,
                "cglut": _image_metrics(arrays["cglut"], target),
                "vera": _image_metrics(arrays["vera"], target),
            }
            append_jsonl(per_sample_path, row)
            rows.append(row)
            if len(rows) <= qualitative_limit:
                _save_comparison(
                    reports_root / "qualitative",
                    source,
                    target,
                    arrays["cglut"],
                    arrays["vera"],
                    metadata["sample_id"],
                )
        print(
            f"[eval] natural batch={batch_index}/{math.ceil(len(dataset) / loader.batch_size)}",
            flush=True,
        )
    summary = _aggregate_natural(
        rows,
        int(config["evaluation"]["bootstrap_replicates"]),
        int(config["pilot"]["seed"]),
    )
    return rows, summary


def _full_grid_evaluation(
    config: dict[str, Any],
    models: dict[str, torch.nn.Module],
    luts: list[Any],
    store: PackedLutStore,
    device: torch.device,
    metrics_root: Path,
    *,
    smoke: bool,
) -> dict[str, Any]:
    selected_luts = luts[:2] if smoke else luts
    output_path = metrics_root / "full_grid_per_style.jsonl"
    output_path.unlink(missing_ok=True)
    chunk_size = int(config["evaluation"]["grid_chunk_points"])
    aggregates: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("rgb_mean", "rgb_p95", "rgb_max", "de00_mean", "de00_p95")}
        for name in models
    }
    for index, record in enumerate(selected_luts, 1):
        inputs, targets = native_lattice(record, store.get(record.style_index), device=device)
        batched_inputs = inputs.unsqueeze(0)
        style_index = torch.tensor([record.style_index], device=device)
        target_numpy = targets.cpu().numpy().reshape(-1, 1, 3)
        target_lab = rgb2lab(np.clip(target_numpy, 0.0, 1.0))
        row: dict[str, Any] = {
            "style_index": record.style_index,
            "preset_id": record.preset_id,
            "lut_content_hash": record.content_hash,
            "taxonomy_major": record.taxonomy_major,
            "taxonomy_minor": record.taxonomy_minor,
            "grid_size": record.grid_size,
        }
        for name, model in models.items():
            prediction_chunks: list[torch.Tensor] = []
            with torch.inference_mode():
                for start in range(0, inputs.shape[0], chunk_size):
                    with _autocast(config):
                        prediction_chunks.append(
                            model.forward_points(
                                batched_inputs[:, start : start + chunk_size], style_index
                            ).float()
                        )
            prediction = torch.cat(prediction_chunks, dim=1).squeeze(0)
            rgb_error = (prediction - targets).abs().mean(dim=-1).cpu().numpy()
            prediction_numpy = prediction.cpu().numpy().reshape(-1, 1, 3)
            prediction_lab = rgb2lab(np.clip(prediction_numpy, 0.0, 1.0))
            delta_e = deltaE_ciede2000(prediction_lab, target_lab).reshape(-1)
            values = {
                "rgb_mean": float(np.mean(rgb_error)),
                "rgb_p95": float(np.percentile(rgb_error, 95)),
                "rgb_max": float(np.max(rgb_error)),
                "de00_mean": float(np.mean(delta_e)),
                "de00_p95": float(np.percentile(delta_e, 95)),
            }
            row[name] = values
            for metric, value in values.items():
                aggregates[name][metric].append(value)
        append_jsonl(output_path, row)
        if index == 1 or index % 50 == 0 or index == len(selected_luts):
            print(f"[eval] full-grid style={index}/{len(selected_luts)}", flush=True)
    return {
        name: {
            metric: float(np.mean(values))
            for metric, values in method_values.items()
        }
        for name, method_values in aggregates.items()
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    natural = summary["natural"]
    lines = [
        "# LUT Renderer Pilot Summary",
        "",
        f"Natural-image samples: {natural['samples']}",
        "",
        "| Method | L1 | PSNR | SSIM | Delta E00 mean | Delta E00 P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in ("cglut", "vera"):
        values = natural["methods"][method]
        lines.append(
            f"| {method} | {values['l1']:.6f} | {values['psnr']:.3f} | "
            f"{values['ssim']:.6f} | {values['delta_e_00_mean']:.4f} | "
            f"{values['delta_e_00_p95']:.4f} |"
        )
    lines.extend(["", "Paired bootstrap differences are recorded in `summary.json`.", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def evaluate(
    config_path: str | Path,
    *,
    smoke: bool = False,
    skip_full_grid: bool = False,
) -> Path:
    config = load_config(config_path)
    prepare(config_path)
    _seed_everything(int(config["pilot"]["seed"]))
    device = _resolve_device(config)
    if smoke:
        config["data"]["image_short_edge"] = 128
        config["data"]["image_long_edge"] = 192
        config["evaluation"]["image_chunk_points"] = 4096
        config["evaluation"]["grid_chunk_points"] = 4096
        config["evaluation"]["bootstrap_replicates"] = 100

    luts = load_lut_manifest(manifests_dir(config) / "luts.jsonl")
    checkpoint_root = output_root(config) / ("smoke" if smoke else "training")
    models = {
        name: _load_model(
            name,
            config,
            len(luts),
            checkpoint_root / name / "latest.pt",
            device,
        )
        for name in ("cglut", "vera")
    }
    result_root = output_root(config) / ("smoke_eval" if smoke else "evaluation")
    metrics_root = result_root / "metrics"
    reports_root = result_root / "reports"
    metrics_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)
    store = PackedLutStore(config["paths"]["lut_npz"], luts)
    try:
        _, natural_summary = _natural_evaluation(
            config, models, luts, store, device, metrics_root, reports_root, smoke=smoke
        )
        full_grid_summary = None
        if bool(config["evaluation"]["full_grid"]) and not skip_full_grid:
            full_grid_summary = _full_grid_evaluation(
                config, models, luts, store, device, metrics_root, smoke=smoke
            )
    finally:
        store.close()
    summary = {
        "pilot": config["pilot"],
        "checkpoints": {
            name: str(checkpoint_root / name / "latest.pt") for name in models
        },
        "natural": natural_summary,
        "full_grid": full_grid_summary,
    }
    summary_path = metrics_root / "summary.json"
    atomic_write_json(summary_path, summary)
    _write_report(reports_root / "summary.md", summary)
    print(f"[eval] complete -> {summary_path}", flush=True)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-full-grid", action="store_true")
    args = parser.parse_args()
    evaluate(args.config, smoke=args.smoke, skip_full_grid=args.skip_full_grid)


if __name__ == "__main__":
    main()
