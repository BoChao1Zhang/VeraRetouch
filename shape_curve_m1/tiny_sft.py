from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .action import action_main_l1, action_vector_mae, decode_raw_action, identity_raw, raw_layout
from .metrics import delta_e_per_image, lpips_per_image
from .render import build_dictionary, free_tail_lut, render_hybrid
from .synthetic import sample_raw_actions


class TinyActionPredictor(nn.Module):
    def __init__(self, raw_dim: int, image_size: int, device: torch.device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(6, 24, 5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(96, 256),
            nn.GELU(),
            nn.Linear(256, raw_dim),
        )
        identity = identity_raw(1, device).squeeze(0)
        with torch.no_grad():
            self.net[-1].bias.copy_(identity)
            self.net[-1].weight.mul_(0.02)

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([source, target], dim=1))


@dataclass(frozen=True)
class TinySFTConfig:
    train_count: int
    val_count: int
    epochs: int
    batch_size: int
    image_size: int
    seed: int


def make_synthetic_pairs(
    base_images: torch.Tensor,
    count: int,
    seed: int,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if base_images.shape[0] < count:
        repeats = (count + base_images.shape[0] - 1) // base_images.shape[0]
        source = base_images.repeat(repeats, 1, 1, 1)[:count]
    else:
        source = base_images[:count]
    raw = sample_raw_actions(count, seed, source.device)
    action = decode_raw_action(raw)
    with torch.no_grad():
        target, _ = render_hybrid(source, action, dictionary, rho, True, True)
    return source, target.detach(), raw.detach()


def _iter_batches(count: int, batch_size: int):
    for start in range(0, count, batch_size):
        yield slice(start, min(start + batch_size, count))


def _grad_norm(loss: torch.Tensor, model: nn.Module) -> float:
    grads = torch.autograd.grad(loss, [p for p in model.parameters() if p.requires_grad], retain_graph=True, allow_unused=True)
    total = torch.zeros((), device=loss.device)
    for grad in grads:
        if grad is not None:
            total = total + grad.detach().pow(2).sum()
    return float(total.sqrt().cpu())


def run_tiny_sft(
    train_base: torch.Tensor,
    val_base: torch.Tensor,
    config: TinySFTConfig,
    lpips_model,
) -> dict[str, object]:
    device = train_base.device
    dictionary, rho, _ = build_dictionary(device=device)
    layout = raw_layout()
    train_source, train_target, train_raw = make_synthetic_pairs(train_base, config.train_count, config.seed + 1000, dictionary, rho)
    val_source, val_target, val_raw = make_synthetic_pairs(val_base, config.val_count, config.seed + 2000, dictionary, rho)

    model = TinyActionPredictor(layout.dim, config.image_size, device).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    grad_audit: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        perm = torch.randperm(config.train_count, device=device)
        totals = {"loss": 0.0, "tail_grid": 0.0}
        batches = 0
        for sl in _iter_batches(config.train_count, config.batch_size):
            idx = perm[sl]
            source = train_source[idx]
            target = train_target[idx]
            gt = decode_raw_action(train_raw[idx])
            pred_raw = model(source, target)
            pred = decode_raw_action(pred_raw)
            pred_render, _ = render_hybrid(source, pred, dictionary, rho, True, True)
            render_loss = F.mse_loss(pred_render, target) + 0.1 * F.l1_loss(pred_render, target)
            main_loss = action_main_l1(pred, gt)
            dict_loss = F.l1_loss(pred.dict_coef, gt.dict_coef)
            tail_loss = F.smooth_l1_loss(free_tail_lut(pred), free_tail_lut(gt), beta=0.01)
            total = render_loss + main_loss + 0.5 * dict_loss + 0.3 * tail_loss

            if epoch == 0 and batches == 0:
                norms = {
                    "render": _grad_norm(render_loss, model),
                    "main": _grad_norm(main_loss, model),
                    "dict": _grad_norm(dict_loss, model),
                    "tail_grid": _grad_norm(tail_loss, model),
                }
                denom = sum(norms.values()) + 1e-12
                grad_audit.append({f"{k}_grad_norm": v for k, v in norms.items()})
                grad_audit[-1]["max_component_ratio"] = max(norms.values()) / denom

            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            totals["loss"] += float(total.detach().cpu())
            totals["tail_grid"] += float(tail_loss.detach().cpu())
            batches += 1
        history.append({key: value / max(1, batches) for key, value in totals.items()})

    val_rows: list[dict[str, float]] = []
    with torch.no_grad():
        for sl in _iter_batches(config.val_count, config.batch_size):
            source = val_source[sl]
            target = val_target[sl]
            gt = decode_raw_action(val_raw[sl])
            pred = decode_raw_action(model(source, target))
            pred_render, _ = render_hybrid(source, pred, dictionary, rho, True, True)
            lp = lpips_per_image(pred_render, target, lpips_model)
            de, _ = delta_e_per_image(pred_render, target)
            action_metrics = action_vector_mae(pred, gt)
            tail_loss = F.smooth_l1_loss(free_tail_lut(pred), free_tail_lut(gt), beta=0.01)
            for i in range(source.shape[0]):
                row = {
                    "LPIPS": float(lp[i].detach().cpu()),
                    "mean_deltaE2000": float(de[i].detach().cpu()),
                    **action_metrics,
                    "free_tail_reconstructed_grid_loss": float(tail_loss.detach().cpu()),
                }
                val_rows.append(row)
    summary = {}
    for key in val_rows[0]:
        summary[key] = float(sum(row[key] for row in val_rows) / len(val_rows))
    summary["free_tail_grid_loss_start"] = history[0]["tail_grid"] if history else float("nan")
    summary["free_tail_grid_loss_end"] = history[-1]["tail_grid"] if history else float("nan")
    summary["free_tail_grid_loss_decreased"] = bool(history and history[-1]["tail_grid"] < history[0]["tail_grid"])
    summary["max_component_grad_ratio"] = grad_audit[0]["max_component_ratio"] if grad_audit else float("nan")
    summary["no_single_loss_dominates"] = bool(summary["max_component_grad_ratio"] < 0.8)
    return {
        "summary": summary,
        "history": history,
        "gradient_audit": grad_audit,
        "train_count": config.train_count,
        "val_count": config.val_count,
    }
