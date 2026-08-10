from __future__ import annotations

import math

import torch


def srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Differentiable D65 sRGB to CIELab conversion for tensors ending in RGB."""
    rgb = rgb.clamp(0.0, 1.0)
    linear = torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).pow(2.4),
    )
    matrix = rgb.new_tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = torch.einsum("...c,dc->...d", linear, matrix)
    white = rgb.new_tensor([0.95047, 1.0, 1.08883])
    normalized = xyz / white
    delta = 6.0 / 29.0
    transformed = torch.where(
        normalized > delta**3,
        # `where` evaluates both branches. Keeping the inactive cube-root branch
        # away from zero prevents an infinite derivative and 0*inf NaNs.
        torch.pow(normalized.clamp_min(delta**3), 1.0 / 3.0),
        normalized / (3.0 * delta**2) + 4.0 / 29.0,
    )
    x_value, y_value, z_value = transformed.unbind(dim=-1)
    return torch.stack(
        (
            116.0 * y_value - 16.0,
            500.0 * (x_value - y_value),
            200.0 * (y_value - z_value),
        ),
        dim=-1,
    )


def hue_chroma_per_point(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction_lab = srgb_to_lab(prediction)
    target_lab = srgb_to_lab(target)
    prediction_ab = prediction_lab[..., 1:]
    target_ab = target_lab[..., 1:]
    prediction_chroma = torch.linalg.vector_norm(prediction_ab, dim=-1, keepdim=True)
    target_chroma = torch.linalg.vector_norm(target_ab, dim=-1)
    prediction_hue = prediction_ab / prediction_chroma.clamp_min(1.0e-6)
    target_hue = target_ab / target_chroma[..., None].clamp_min(1.0e-6)
    cosine = (prediction_hue * target_hue).sum(dim=-1).clamp(-1.0, 1.0)
    return target_chroma * (1.0 - cosine)


def hard_mining_ratio(epoch: int) -> float | None:
    """Protocol epoch is one-based; epochs before five have no extra hard term."""
    if epoch < 5:
        return None
    if epoch <= 20:
        return 0.10 + 0.30 * (epoch - 5) / 15.0
    return 0.40


def hard_count(point_count: int, epoch: int) -> int:
    ratio = hard_mining_ratio(epoch)
    if ratio is None:
        return 0
    return max(1, math.ceil(ratio * point_count))
