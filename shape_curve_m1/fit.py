from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .action import decode_raw_action, make_raw_parameter, mask_raw_for_config
from .metrics import metric_summary
from .render import (
    apply_lut,
    gamut_penalty,
    identity_lut,
    lut_stats,
    render_hybrid,
    smoothness2_3d,
    tv3d,
)


@dataclass(frozen=True)
class FitConfig:
    name: str
    include_dictionary: bool
    include_free_tail: bool
    dense: bool = False
    smoothness_weight: float = 0.01
    gamut_weight: float = 0.5
    dict_l1_weight: float = 0.0005
    dict_entropy_weight: float = 0.0
    active_budget_weight: float = 0.0
    active_budget: float = 6.0
    tail_gate_weight: float = 0.01


FIT_MAIN = FitConfig("Fit-Main", include_dictionary=False, include_free_tail=False)
FIT_DICT = FitConfig("Fit-Dict", include_dictionary=True, include_free_tail=False)
FIT_FULL = FitConfig("Fit-Full", include_dictionary=True, include_free_tail=True)
D4_DENSE = FitConfig("D4-Dense", include_dictionary=False, include_free_tail=False, dense=True)
ALL_FIT_CONFIGS = (FIT_MAIN, FIT_DICT, FIT_FULL, D4_DENSE)
A1_NO_REG_FULL = FitConfig(
    "A1-NoReg-Full",
    include_dictionary=True,
    include_free_tail=True,
    smoothness_weight=0.0,
    gamut_weight=0.0,
    dict_l1_weight=0.0,
    tail_gate_weight=0.0,
)


def fit_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    config: FitConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if config.dense:
        return fit_dense_batch(source, target, lpips_model, steps=steps, lr=lr)
    return fit_hybrid_batch(source, target, config, dictionary, rho, lpips_model, steps=steps, lr=lr)


def fit_hybrid_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    config: FitConfig,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    lpips_model,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    raw = make_raw_parameter(source.shape[0], source.device, config.include_dictionary, config.include_free_tail)
    opt = torch.optim.AdamW([raw], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        masked = mask_raw_for_config(raw, config.include_dictionary, config.include_free_tail)
        action = decode_raw_action(masked)
        pred, luts = render_hybrid(source, action, dictionary, rho, config.include_dictionary, config.include_free_tail)
        loss = F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target)
        loss = loss + config.smoothness_weight * smoothness2_3d(luts["final"]).mean()
        loss = loss + config.gamut_weight * gamut_penalty(luts["pre"])
        if config.include_dictionary:
            abs_coef = action.dict_coef.abs()
            loss = loss + config.dict_l1_weight * abs_coef.mean()
            if config.dict_entropy_weight:
                probs = abs_coef / abs_coef.sum(dim=1, keepdim=True).clamp_min(1e-6)
                entropy = -(probs * (probs + 1e-6).log()).sum(dim=1).mean()
                loss = loss + config.dict_entropy_weight * entropy
            if config.active_budget_weight:
                soft_active = torch.sigmoid(80.0 * (abs_coef - 0.03)).sum(dim=1)
                loss = loss + config.active_budget_weight * F.relu(soft_active - config.active_budget).pow(2).mean()
        if config.include_free_tail:
            loss = loss + config.tail_gate_weight * action.tail_gate.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        masked = mask_raw_for_config(raw, config.include_dictionary, config.include_free_tail)
        action = decode_raw_action(masked)
        pred, luts = render_hybrid(source, action, dictionary, rho, config.include_dictionary, config.include_free_tail)
        aux = lut_stats(action, luts)
        metrics = metric_summary(pred, target, aux, lpips_model)
    return pred.detach(), metrics


def fit_dense_batch(
    source: torch.Tensor,
    target: torch.Tensor,
    lpips_model,
    steps: int,
    lr: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    batch = source.shape[0]
    identity = identity_lut(device=source.device, dtype=source.dtype).unsqueeze(0).repeat(batch, 1, 1, 1, 1)
    raw_delta = nn.Parameter(torch.zeros_like(identity))
    opt = torch.optim.AdamW([raw_delta], lr=lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        lut_pre = identity + 0.35 * torch.tanh(raw_delta)
        lut_final = lut_pre.clamp(0, 1)
        pred = apply_lut(lut_final, source)
        loss = F.mse_loss(pred, target) + 0.15 * F.l1_loss(pred, target)
        loss = loss + 0.002 * tv3d(lut_final) + 0.5 * gamut_penalty(lut_pre)
        loss.backward()
        opt.step()
    with torch.no_grad():
        lut_pre = identity + 0.35 * torch.tanh(raw_delta)
        lut_final = lut_pre.clamp(0, 1)
        pred = apply_lut(lut_final, source)
        delta = lut_final - identity
        energy = delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
        aux = {
            "clipping_ratio": ((lut_pre < 0.0) | (lut_pre > 1.0)).float().mean(dim=(1, 2, 3, 4)),
            "gamut_violation_ratio": ((lut_pre < -1e-4) | (lut_pre > 1.0001)).float().mean(dim=(1, 2, 3, 4)),
            "banding_smoothness_penalty": smoothness2_3d(lut_final),
            "main_energy": torch.zeros_like(energy),
            "dictionary_energy": torch.zeros_like(energy),
            "free_tail_energy": torch.zeros_like(energy),
            "residual_ratio": torch.ones_like(energy),
            "dictionary_explained_ratio": torch.zeros_like(energy),
            "active_atom_count": torch.zeros_like(energy),
            "inactive_atom_ratio": torch.ones_like(energy),
        }
        metrics = metric_summary(pred, target, aux, lpips_model)
    return pred.detach(), metrics
