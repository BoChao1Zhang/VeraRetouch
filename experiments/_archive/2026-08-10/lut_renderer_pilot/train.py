from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import (
    append_jsonl,
    atomic_write_json,
    count_parameters,
    load_config,
    manifests_dir,
    output_root,
    read_jsonl,
)
from .data import (
    LutRecord,
    OnlinePairDataset,
    OnlinePairBatchSampler,
    PackedLutStore,
    collate_images,
    load_lut_manifest,
    native_lattice,
    render_lut_batch,
)
from .losses import hard_count, hue_chroma_per_point
from .models import SharedGeometryCGLUT, VeraStyleRenderer
from .prepare import prepare


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(config: dict[str, Any]) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment")
    physical_gpu = str(config["pilot"]["physical_gpu"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first_visible = visible.split(",")[0].strip()
        if first_visible != physical_gpu:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES={visible!r}, expected physical GPU {physical_gpu}"
            )
        device = torch.device("cuda:0")
    else:
        device = torch.device(f"cuda:{physical_gpu}")
    torch.cuda.set_device(device)
    return device


def _autocast(config: dict[str, Any]):
    if str(config["training"]["precision"]).lower() == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def _forward(
    model: SharedGeometryCGLUT | VeraStyleRenderer,
    points: torch.Tensor,
    style_indices: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    with _autocast(config):
        return model.forward_points(points, style_indices)


def _image_objective(
    model: SharedGeometryCGLUT | VeraStyleRenderer,
    images: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor,
    style_indices: torch.Tensor,
    config: dict[str, Any],
) -> float:
    chunk_size = int(config["training"]["image_chunk_points"])
    weight = float(config["training"]["image_l1_weight"])
    points = images.permute(0, 2, 3, 1).reshape(images.shape[0], -1, 3)
    target_points = targets.permute(0, 2, 3, 1).reshape(targets.shape[0], -1, 3)
    valid_points = valid.reshape(valid.shape[0], -1)
    denominators = valid_points.sum(dim=1).clamp_min(1).float() * 3.0
    reported = 0.0
    for start in range(0, points.shape[1], chunk_size):
        end = min(start + chunk_size, points.shape[1])
        prediction = _forward(model, points[:, start:end], style_indices, config)
        absolute = (prediction - target_points[:, start:end]).abs().sum(dim=-1)
        per_style = (
            absolute * valid_points[:, start:end].to(dtype=absolute.dtype)
        ).sum(dim=1) / denominators
        loss = weight * per_style.mean()
        loss.backward()
        reported += float(loss.detach())
    return reported


def _grid_groups(
    style_indices: torch.Tensor, records: list[LutRecord]
) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = defaultdict(list)
    for batch_index, style_index in enumerate(style_indices.tolist()):
        groups[records[style_index].grid_size].append(batch_index)
    return dict(groups)


def _grid_objective(
    model_name: str,
    model: SharedGeometryCGLUT | VeraStyleRenderer,
    style_indices: torch.Tensor,
    records: list[LutRecord],
    store: PackedLutStore,
    epoch: int,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    batch_size = style_indices.numel()
    chunk_size = int(config["training"]["grid_chunk_points"])
    reconstruction_weight = float(config["training"]["grid_l1_weight"])
    hue_weight = float(config["training"]["hue_chroma_weight"])
    use_hard_mining = model_name == "cglut" and hard_count(1, epoch) > 0
    base_coefficient = 0.5 if use_hard_mining else 1.0
    reported = {"grid_rec": 0.0, "grid_hc": 0.0, "grid_hard": 0.0}

    for _, batch_indices in sorted(_grid_groups(style_indices, records).items()):
        group_style_indices = style_indices[batch_indices]
        inputs_list: list[torch.Tensor] = []
        targets_list: list[torch.Tensor] = []
        for style_index in group_style_indices.tolist():
            inputs, targets = native_lattice(
                records[style_index], store.get(style_index), device=device
            )
            inputs_list.append(inputs)
            targets_list.append(targets)
        inputs = torch.stack(inputs_list)
        targets = torch.stack(targets_list)
        point_count = inputs.shape[1]

        for start in range(0, point_count, chunk_size):
            end = min(start + chunk_size, point_count)
            prediction = _forward(
                model, inputs[:, start:end], group_style_indices, config
            )
            point_l1 = (prediction - targets[:, start:end]).abs().sum(dim=-1)
            rec_loss = (
                reconstruction_weight
                * base_coefficient
                * point_l1.sum(dim=1).div(point_count).sum()
                / batch_size
            )
            hc_loss = (
                hue_weight
                * hue_chroma_per_point(prediction, targets[:, start:end])
                .sum(dim=1)
                .div(point_count)
                .sum()
                / batch_size
            )
            (rec_loss + hc_loss).backward()
            reported["grid_rec"] += float(rec_loss.detach())
            reported["grid_hc"] += float(hc_loss.detach())

        if use_hard_mining:
            error_chunks: list[torch.Tensor] = []
            with torch.no_grad():
                for start in range(0, point_count, chunk_size):
                    end = min(start + chunk_size, point_count)
                    prediction = _forward(
                        model, inputs[:, start:end], group_style_indices, config
                    )
                    error_chunks.append(
                        (prediction - targets[:, start:end]).abs().sum(dim=-1)
                    )
            errors = torch.cat(error_chunks, dim=1)
            count = hard_count(point_count, epoch)
            indices = torch.argsort(
                errors, dim=1, descending=True, stable=True
            )[:, :count]
            selected_inputs = torch.gather(
                inputs, 1, indices[..., None].expand(-1, -1, 3)
            )
            selected_targets = torch.gather(
                targets, 1, indices[..., None].expand(-1, -1, 3)
            )
            for start in range(0, count, chunk_size):
                end = min(start + chunk_size, count)
                prediction = _forward(
                    model, selected_inputs[:, start:end], group_style_indices, config
                )
                point_l1 = (
                    prediction - selected_targets[:, start:end]
                ).abs().sum(dim=-1)
                hard_loss = (
                    reconstruction_weight
                    * 0.5
                    * point_l1.sum(dim=1).div(count).sum()
                    / batch_size
                )
                hard_loss.backward()
                reported["grid_hard"] += float(hard_loss.detach())
    return reported


def _regularization(
    model_name: str,
    model: SharedGeometryCGLUT | VeraStyleRenderer,
    style_indices: torch.Tensor,
    config: dict[str, Any],
) -> dict[str, float]:
    if model_name != "cglut":
        return {"opacity": 0.0, "embedding": 0.0}
    assert isinstance(model, SharedGeometryCGLUT)
    entropy, embedding_l2 = model.regularization(style_indices)
    opacity_loss = float(config["training"]["opacity_weight"]) * entropy
    embedding_loss = float(config["training"]["embedding_weight"]) * embedding_l2
    (opacity_loss + embedding_loss).backward()
    return {
        "opacity": float(opacity_loss.detach()),
        "embedding": float(embedding_loss.detach()),
    }


def _save_checkpoint(
    path: Path,
    *,
    model_name: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    update: int,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_name": model_name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch_completed": epoch,
            "update": update,
            "pilot": config["pilot"],
        },
        temporary,
    )
    os.replace(temporary, path)


def _build_model(
    model_name: str, num_styles: int, config: dict[str, Any]
) -> SharedGeometryCGLUT | VeraStyleRenderer:
    if model_name == "cglut":
        model = SharedGeometryCGLUT.from_config(num_styles, config["cglut"])
        if model.shared_parameter_count() != 162_892:
            raise RuntimeError(
                f"CGLUT shared parameter count {model.shared_parameter_count()} != 162892"
            )
        return model
    model = VeraStyleRenderer.from_config(
        num_styles, config["paths"]["vera_checkpoint"], config["vera"]
    )
    decoder_parameters = count_parameters(model.decoder)
    if decoder_parameters != 2_577_795:
        raise RuntimeError(f"Vera decoder parameter count {decoder_parameters} != 2577795")
    return model


def _worker_init(base_seed: int, epoch: int) -> Callable[[int], None]:
    def initialize(worker_id: int) -> None:
        seed = (base_seed + 1009 * epoch + worker_id) % (2**32)
        random.seed(seed)
        np.random.seed(seed)

    return initialize


def train(
    config_path: str | Path,
    model_name: str,
    *,
    smoke: bool = False,
    profile: bool = False,
    max_batches: int | None = None,
    no_resume: bool = False,
) -> Path:
    config = load_config(config_path)
    prepare(config_path)
    seed = int(config["pilot"]["seed"])
    _seed_everything(seed)
    device = _resolve_device(config)

    manifest_root = manifests_dir(config)
    luts = load_lut_manifest(manifest_root / "luts.jsonl")
    sources = list(read_jsonl(manifest_root / "train_sources.jsonl"))
    if smoke:
        config["data"]["image_short_edge"] = 128
        config["data"]["image_long_edge"] = 192
        config["data"]["logical_batch_size"] = 2
        config["data"]["num_workers"] = 0
        config["training"]["epochs"] = 1
        config["training"]["image_chunk_points"] = 4096
        config["training"]["grid_chunk_points"] = 4096
        max_batches = 1 if max_batches is None else max_batches
    elif profile:
        config["training"]["epochs"] = 1
        max_batches = 1 if max_batches is None else max_batches

    run_stage = "smoke" if smoke else ("profile" if profile else "training")
    run_root = output_root(config) / run_stage / model_name
    run_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_root / "resolved_config.json", config)
    log_path = run_root / "train.jsonl"
    failure_path = run_root / "failures.jsonl"

    model = _build_model(model_name, len(luts), config).to(device)
    model_config = config["cglut"] if model_name == "cglut" else config["vera"]
    optimizer_groups = model.optimizer_groups(model_config)
    optimizer_class = torch.optim.Adam if model_name == "cglut" else torch.optim.AdamW
    optimizer = optimizer_class(
        optimizer_groups,
        weight_decay=float(config["training"]["weight_decay"]),
    )
    batch_size = int(config["data"]["logical_batch_size"])
    updates_per_epoch = math.ceil(len(luts) / batch_size)
    epochs = int(config["training"]["epochs"])
    total_updates = max(1, epochs * (max_batches or updates_per_epoch))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_updates, eta_min=0.0
    )

    start_epoch = 1
    global_update = 0
    latest_path = run_root / "latest.pt"
    if latest_path.is_file() and not no_resume and not smoke and not profile:
        checkpoint = torch.load(latest_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_name:
            raise RuntimeError(f"checkpoint model mismatch: {latest_path}")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch_completed"]) + 1
        global_update = int(checkpoint["update"])
        print(f"[train:{model_name}] resumed after epoch {start_epoch - 1}", flush=True)

    dataset = OnlinePairDataset(
        sources,
        luts,
        short_edge=int(config["data"]["image_short_edge"]),
        long_edge=int(config["data"]["image_long_edge"]),
    )
    store = PackedLutStore(config["paths"]["lut_npz"], luts)
    torch.cuda.reset_peak_memory_stats(device)
    print(
        f"[train:{model_name}] device={torch.cuda.get_device_name(device)} "
        f"styles={len(luts)} sources={len(sources)} params={count_parameters(model):,} "
        f"epochs={epochs} batch={batch_size}",
        flush=True,
    )

    try:
        for epoch in range(start_epoch, epochs + 1):
            batch_sampler = OnlinePairBatchSampler(
                sources,
                len(luts),
                batch_size,
                base_seed=seed,
                epoch=epoch,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=int(config["data"]["num_workers"]),
                pin_memory=bool(config["data"]["pin_memory"]),
                collate_fn=collate_images,
                worker_init_fn=_worker_init(seed, epoch),
                persistent_workers=False,
            )
            epoch_start = time.monotonic()
            for batch_index, batch in enumerate(loader, 1):
                if max_batches is not None and batch_index > max_batches:
                    break
                update_start = time.monotonic()
                images = batch["images"].to(device, non_blocking=True)
                valid = batch["valid"].to(device, non_blocking=True)
                style_indices = batch["style_indices"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.no_grad():
                        targets = render_lut_batch(
                            images, style_indices, luts, store
                        )
                    image_loss = _image_objective(
                        model, images, targets, valid, style_indices, config
                    )
                    grid_losses = _grid_objective(
                        model_name,
                        model,
                        style_indices,
                        luts,
                        store,
                        epoch,
                        device,
                        config,
                    )
                    regularization = _regularization(
                        model_name, model, style_indices, config
                    )
                    gradients_finite = all(
                        parameter.grad is None
                        or bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    )
                    if not gradients_finite:
                        raise FloatingPointError("non-finite gradient")
                    optimizer.step()
                    scheduler.step()
                except BaseException as error:
                    append_jsonl(
                        failure_path,
                        {
                            "epoch": epoch,
                            "batch": batch_index,
                            "update": global_update,
                            "type": type(error).__name__,
                            "message": str(error),
                            "style_indices": style_indices.detach().cpu().tolist(),
                        },
                    )
                    raise

                global_update += 1
                elapsed = time.monotonic() - update_start
                metrics = {
                    "epoch": epoch,
                    "batch": batch_index,
                    "update": global_update,
                    "styles": int(style_indices.numel()),
                    "image_l1": image_loss,
                    **grid_losses,
                    **regularization,
                    "total_reported": image_loss
                    + sum(grid_losses.values())
                    + sum(regularization.values()),
                    "seconds": elapsed,
                    "lr": [group["lr"] for group in optimizer.param_groups],
                    "peak_vram_bytes": torch.cuda.max_memory_allocated(device),
                }
                append_jsonl(log_path, metrics)
                log_every = int(config["training"]["log_every_updates"])
                if global_update == 1 or global_update % log_every == 0:
                    print(
                        f"[train:{model_name}] epoch={epoch}/{epochs} "
                        f"batch={batch_index}/{updates_per_epoch} update={global_update} "
                        f"loss={metrics['total_reported']:.6f} sec={elapsed:.2f}",
                        flush=True,
                    )

            if max_batches is None or max_batches >= updates_per_epoch:
                checkpoint_epoch = epoch
            else:
                checkpoint_epoch = epoch
            checkpoint_path = run_root / f"epoch_{checkpoint_epoch:03d}.pt"
            _save_checkpoint(
                checkpoint_path,
                model_name=model_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=checkpoint_epoch,
                update=global_update,
                config=config,
            )
            _save_checkpoint(
                latest_path,
                model_name=model_name,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=checkpoint_epoch,
                update=global_update,
                config=config,
            )
            print(
                f"[train:{model_name}] epoch={epoch} complete in "
                f"{time.monotonic() - epoch_start:.1f}s -> {checkpoint_path}",
                flush=True,
            )
    finally:
        store.close()
    return latest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=("cglut", "vera"), required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--profile", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    train(
        args.config,
        args.model,
        smoke=args.smoke,
        profile=args.profile,
        max_batches=args.max_batches,
        no_resume=args.no_resume,
    )


if __name__ == "__main__":
    main()
