from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import torch

from .action import R_FREE, raw_layout, decode_raw_action, identity_raw
from .render import (
    hybrid_lut,
    apply_lut,
    hue_mask,
    identity_lut,
    lut_stats,
    luma,
    rgb_to_hsv,
    soft_range,
    smooth3d,
)


TIER_B_FAMILIES = (
    "smooth_random",
    "matrix_plus_smooth",
    "hue_twist",
    "color_isolation",
    "sat_compress",
)


@dataclass(frozen=True)
class SyntheticActionPolicy:
    name: str
    max_clipping_ratio: float = 0.01
    max_gamut_violation_ratio: float = 0.01
    min_active_atom_count: float = 2.0
    max_active_atom_count: float = 6.0
    min_free_tail_energy: float = 1e-8
    max_free_tail_energy: float = 0.003
    max_attempt_multiplier: int = 80
    oversample_multiplier: int = 8
    max_candidate_batch: int = 512

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLEAN_RENDERER_POLICY = SyntheticActionPolicy(name="clean_renderer_aligned")


def sample_raw_actions(batch: int, seed: int, device: torch.device | str) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    layout = raw_layout()
    raw = identity_raw(batch, device)
    raw[:, layout.curve_delta] += 0.35 * torch.randn(batch, 3 * 7, generator=gen, device=device)
    raw[:, layout.curve_black_white] = -4.5 + 0.6 * torch.randn(batch, 3 * 2, generator=gen, device=device)
    raw[:, layout.hsl] = 0.45 * torch.randn(batch, 8 * 3, generator=gen, device=device)
    raw[:, layout.wb] = 0.55 * torch.randn(batch, 2, generator=gen, device=device)
    raw[:, layout.dictionary] = 0.0
    for b in range(batch):
        active = torch.randperm(16, generator=gen, device=device)[: int(torch.randint(2, 7, (1,), generator=gen, device=device))]
        signs = torch.where(torch.rand(active.numel(), generator=gen, device=device) > 0.5, 1.0, -1.0)
        coef = signs * (0.35 + 0.55 * torch.rand(active.numel(), generator=gen, device=device))
        raw[b, layout.dictionary][active] = torch.atanh(coef.clamp(-0.98, 0.98))
    raw[:, layout.tail_gate] = -4.0 + 0.8 * torch.randn(batch, R_FREE, generator=gen, device=device)
    raw[:, layout.tail_color] = 0.4 * torch.randn(batch, R_FREE * 3, generator=gen, device=device)
    raw[:, layout.tail_alpha] = 0.3 * torch.randn(batch, R_FREE * 10, generator=gen, device=device)
    raw[:, layout.tail_beta] = 0.3 * torch.randn(batch, R_FREE * 10, generator=gen, device=device)
    raw[:, layout.tail_gamma] = 0.3 * torch.randn(batch, R_FREE * 10, generator=gen, device=device)
    return raw


def sample_filtered_raw_actions(
    batch: int,
    seed: int,
    device: torch.device | str,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    policy: SyntheticActionPolicy = CLEAN_RENDERER_POLICY,
) -> tuple[torch.Tensor, dict[str, Any]]:
    accepted: list[torch.Tensor] = []
    accepted_stats: dict[str, list[torch.Tensor]] = {}
    attempted = 0
    rounds = 0
    max_attempts = max(batch, batch * policy.max_attempt_multiplier)
    while sum(item.shape[0] for item in accepted) < batch and attempted < max_attempts:
        remaining = batch - sum(item.shape[0] for item in accepted)
        candidate_count = min(
            policy.max_candidate_batch,
            max(remaining * policy.oversample_multiplier, remaining),
            max_attempts - attempted,
        )
        raw = sample_raw_actions(candidate_count, seed + rounds * 9_973, device)
        action = decode_raw_action(raw)
        with torch.no_grad():
            luts = hybrid_lut(action, dictionary, rho, True, True)
            stats = lut_stats(action, luts)
            mask = _policy_mask(stats, policy)
        if bool(mask.any()):
            selected = raw[mask][:remaining].detach()
            accepted.append(selected)
            for key, value in stats.items():
                accepted_stats.setdefault(key, []).append(value[mask][: selected.shape[0]].detach())
        attempted += candidate_count
        rounds += 1

    accepted_count = sum(item.shape[0] for item in accepted)
    if accepted_count < batch:
        raise RuntimeError(
            f"{policy.name} accepted {accepted_count}/{batch} actions after {attempted} attempts; "
            "relax policy or reduce raw action amplitudes"
        )
    raw_out = torch.cat(accepted, dim=0)[:batch]
    summary: dict[str, Any] = {
        "policy": policy.to_dict(),
        "requested": batch,
        "attempted": attempted,
        "accepted": batch,
        "acceptance_rate": batch / max(1, attempted),
    }
    for key, chunks in accepted_stats.items():
        values = torch.cat(chunks, dim=0)[:batch].detach().float().cpu()
        summary[f"accepted_{key}"] = float(values.mean())
    return raw_out, summary


def _policy_mask(stats: dict[str, torch.Tensor], policy: SyntheticActionPolicy) -> torch.Tensor:
    return (
        (stats["clipping_ratio"] <= policy.max_clipping_ratio)
        & (stats["gamut_violation_ratio"] <= policy.max_gamut_violation_ratio)
        & (stats["active_atom_count"] >= policy.min_active_atom_count)
        & (stats["active_atom_count"] <= policy.max_active_atom_count)
        & (stats["free_tail_energy"] >= policy.min_free_tail_energy)
        & (stats["free_tail_energy"] <= policy.max_free_tail_energy)
    )


def generate_dense_teacher_lut(
    family: str,
    seed: int,
    g: int,
    device: torch.device | str,
    stress: bool = False,
) -> torch.Tensor:
    torch.manual_seed(seed)
    random.seed(seed)
    lut = identity_lut(g, device)
    amp = 0.10 if stress else 0.065
    if family == "smooth_random":
        lut = lut + smooth_random_residual(g, control=5, amp=amp, device=device, seed=seed)
    elif family == "matrix_plus_smooth":
        mat = torch.eye(3, device=device) + 0.08 * torch.randn(3, 3, device=device)
        lut = torch.einsum("...c,dc->...d", lut, mat)
        lut = lut + smooth_random_residual(g, control=4, amp=0.045 if not stress else 0.08, device=device, seed=seed + 17)
    elif family == "hue_twist":
        h, s, _ = rgb_to_hsv(lut)
        mask = hue_mask(h, 190, 70) * soft_range(s, 0.2, 1.0)
        lut = lut + mask[..., None] * torch.tensor([-0.06, 0.05, 0.08], device=device)
    elif family == "color_isolation":
        h, s, _ = rgb_to_hsv(lut)
        center = [25, 120, 220, 300][seed % 4]
        keep = hue_mask(h, center, 35) * soft_range(s, 0.08, 1.0)
        gray = luma(lut)[..., None].expand_as(lut)
        lut = lut * keep[..., None] + (0.65 * gray + 0.35 * lut) * (1.0 - keep[..., None])
    elif family == "sat_compress":
        y = luma(lut)[..., None]
        _, s, _ = rgb_to_hsv(lut)
        gray = y.expand_as(lut)
        mask = soft_range(s, 0.55, 1.0)
        lut = lut + mask[..., None] * (gray - lut) * (0.25 if not stress else 0.38)
    else:
        raise ValueError(family)
    return lut.clamp(0, 1)


def smooth_random_residual(
    g: int,
    control: int,
    amp: float,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(control, control, control, 3, generator=gen, device=device)
    up = torch.nn.functional.interpolate(
        noise.permute(3, 0, 1, 2).unsqueeze(0),
        size=(g, g, g),
        mode="trilinear",
        align_corners=True,
    ).squeeze(0).permute(1, 2, 3, 0)
    up = smooth3d(up, passes=2)
    up = up - up.mean(dim=(0, 1, 2), keepdim=True)
    rgb = identity_lut(g, device)
    boundary = torch.minimum(rgb, 1.0 - rgb).amin(dim=-1, keepdim=True)
    return amp * (0.25 + boundary).clamp(0.15, 1.0) * up / (up.abs().amax() + 1e-6)


def render_dense_teacher(image: torch.Tensor, family: str, seeds: list[int], g: int) -> tuple[torch.Tensor, torch.Tensor]:
    luts = [generate_dense_teacher_lut(family, seed, g, image.device, stress=(idx % 7 == 0)) for idx, seed in enumerate(seeds)]
    lut_batch = torch.stack(luts, dim=0).to(image.dtype)
    return apply_lut(lut_batch, image), lut_batch


def decode_raw_batch(raw: torch.Tensor):
    return decode_raw_action(raw)
