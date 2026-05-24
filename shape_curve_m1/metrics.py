from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from skimage.color import deltaE_ciede2000, rgb2lab


@dataclass
class MetricAccumulator:
    values: dict[str, list[float]]

    @classmethod
    def create(cls) -> "MetricAccumulator":
        return cls(values={})

    def add(self, key: str, value: float) -> None:
        self.values.setdefault(key, []).append(float(value))

    def extend_tensor(self, key: str, tensor: torch.Tensor) -> None:
        for item in tensor.detach().float().cpu().flatten().tolist():
            self.add(key, item)

    def summary(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for key, vals in self.values.items():
            arr = np.asarray(vals, dtype=np.float64)
            out[key] = float(arr.mean()) if arr.size else float("nan")
        return out


def psnr_per_image(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-12)
    return 10.0 * torch.log10(1.0 / mse)


def lpips_per_image(pred: torch.Tensor, target: torch.Tensor, model: Any | None) -> torch.Tensor:
    if model is None:
        return torch.full((pred.shape[0],), float("nan"), device=pred.device)
    with torch.no_grad():
        vals = model(pred.clamp(0, 1) * 2.0 - 1.0, target.clamp(0, 1) * 2.0 - 1.0)
    return vals.flatten()


def delta_e_per_image(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_np = pred.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    target_np = target.detach().clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    means: list[float] = []
    p95s: list[float] = []
    for p_img, t_img in zip(pred_np, target_np):
        de = deltaE_ciede2000(rgb2lab(p_img), rgb2lab(t_img))
        means.append(float(np.mean(de)))
        p95s.append(float(np.percentile(de, 95)))
    device = pred.device
    return torch.tensor(means, device=device), torch.tensor(p95s, device=device)


def metric_summary(
    pred: torch.Tensor,
    target: torch.Tensor,
    aux: dict[str, torch.Tensor],
    lpips_model: Any | None,
) -> dict[str, torch.Tensor]:
    de_mean, de_p95 = delta_e_per_image(pred, target)
    out = {
        "PSNR": psnr_per_image(pred, target),
        "LPIPS": lpips_per_image(pred, target, lpips_model),
        "mean_deltaE2000": de_mean,
        "p95_deltaE2000": de_p95,
    }
    out.update(aux)
    return out


def average_metric_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    out = {}
    for key in keys:
        vals = [row[key] for row in rows if key in row and not np.isnan(row[key])]
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out
