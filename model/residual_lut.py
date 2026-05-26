from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


AlphaActivation = Literal["tanh", "softmax", "none"]
LUTMode = Literal["3d", "4d"]


@dataclass
class ResidualLUTOutput:
    image: torch.Tensor
    lut_image: torch.Tensor
    lut: torch.Tensor
    alpha: torch.Tensor
    gate: torch.Tensor
    context: torch.Tensor | None


def identity_lut_3d(
    grid_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    axis = torch.linspace(0.0, 1.0, grid_size, device=device, dtype=dtype)
    r, g, b = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((r, g, b), dim=-1)


def identity_lut_4d(
    grid_size: int,
    context_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    lut3d = identity_lut_3d(grid_size, device=device, dtype=dtype)
    return lut3d.unsqueeze(3).expand(grid_size, grid_size, grid_size, context_size, 3)


def apply_lut_3d(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Apply a batched 3D LUT.

    Args:
        lut: (B, G, G, G, 3), RGB grid order.
        image: (B, 3, H, W), values in [0, 1].

    Returns:
        (B, 3, H, W)
    """
    if lut.ndim != 5:
        raise ValueError(f"3D LUT must have shape (B, G, G, G, 3), got {tuple(lut.shape)}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B, 3, H, W), got {tuple(image.shape)}")
    if lut.shape[0] != image.shape[0]:
        raise ValueError("lut and image batch sizes must match")

    lut_chw = lut.permute(0, 4, 1, 2, 3)
    coords = image.clamp(0.0, 1.0).permute(0, 2, 3, 1) * 2.0 - 1.0

    # grid_sample expects x/y/z to index W/H/D. Our LUT axes are R/G/B,
    # so the coordinate order must be B/G/R.
    grid = coords.unsqueeze(1)[..., [2, 1, 0]]
    out = F.grid_sample(
        lut_chw,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return out.squeeze(2)


def _lower_upper_weight(coord: torch.Tensor, size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scaled = coord.clamp(0.0, 1.0) * float(size - 1)
    lower = scaled.floor().long()
    upper = (lower + 1).clamp(max=size - 1)
    weight = scaled - lower.to(dtype=scaled.dtype)
    return lower, upper, weight


def _gather_lut4d(
    lut_flat: torch.Tensor,
    r: torch.Tensor,
    g: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    grid_size: int,
    context_size: int,
) -> torch.Tensor:
    index = (((r * grid_size + g) * grid_size + b) * context_size + c).long()
    return lut_flat.gather(1, index.unsqueeze(-1).expand(-1, -1, 3))


def apply_lut_4d(lut: torch.Tensor, image: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    """Apply a batched 4D LUT using explicit quadrilinear interpolation.

    Args:
        lut: (B, G, G, G, C, 3), RGB-context grid order.
        image: (B, 3, H, W), values in [0, 1].
        context: (B, 1, H, W), values in [0, 1].

    Returns:
        (B, 3, H, W)
    """
    if lut.ndim != 6:
        raise ValueError(f"4D LUT must have shape (B, G, G, G, C, 3), got {tuple(lut.shape)}")
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"image must have shape (B, 3, H, W), got {tuple(image.shape)}")
    if context.ndim != 4 or context.shape[1] != 1:
        raise ValueError(f"context must have shape (B, 1, H, W), got {tuple(context.shape)}")
    if lut.shape[0] != image.shape[0] or image.shape[0] != context.shape[0]:
        raise ValueError("lut, image, and context batch sizes must match")
    if image.shape[-2:] != context.shape[-2:]:
        raise ValueError("image and context spatial sizes must match")

    batch, grid_size, _, _, context_size, _ = lut.shape
    height, width = image.shape[-2:]
    pixels = height * width

    coords = torch.cat((image.clamp(0.0, 1.0), context.clamp(0.0, 1.0)), dim=1)
    coords = coords.permute(0, 2, 3, 1).reshape(batch, pixels, 4)
    r, g, b, c = coords.unbind(dim=-1)

    r0, r1, wr = _lower_upper_weight(r, grid_size)
    g0, g1, wg = _lower_upper_weight(g, grid_size)
    b0, b1, wb = _lower_upper_weight(b, grid_size)
    c0, c1, wc = _lower_upper_weight(c, context_size)

    lut_flat = lut.reshape(batch, grid_size * grid_size * grid_size * context_size, 3)
    out = image.new_zeros(batch, pixels, 3)

    for ri, rw in ((r0, 1.0 - wr), (r1, wr)):
        for gi, gw in ((g0, 1.0 - wg), (g1, wg)):
            for bi, bw in ((b0, 1.0 - wb), (b1, wb)):
                for ci, cw in ((c0, 1.0 - wc), (c1, wc)):
                    weight = (rw * gw * bw * cw).unsqueeze(-1)
                    out = out + weight * _gather_lut4d(
                        lut_flat,
                        ri,
                        gi,
                        bi,
                        ci,
                        grid_size=grid_size,
                        context_size=context_size,
                    )

    return out.reshape(batch, height, width, 3).permute(0, 3, 1, 2)


class ImageCoeffHead(nn.Module):
    def __init__(self, in_channels: int, basis_count: int, hidden_channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=2, padding=1),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, basis_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiddenCoeffHead(nn.Module):
    def __init__(self, in_dim: int, basis_count: int, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, basis_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 2:
            x = x.flatten(start_dim=1)
        if x.shape[-1] != self.in_dim:
            raise ValueError(f"expected coeff feature dim {self.in_dim}, got {x.shape[-1]}")
        return self.net(x)


class ContextHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 32,
        vlm_channels: int | None = None,
        vlm_proj_channels: int = 64,
    ):
        super().__init__()
        self.vlm_channels = vlm_channels
        self.vlm_proj = (
            nn.Sequential(
                nn.Conv2d(vlm_channels, vlm_proj_channels, kernel_size=1),
                nn.SiLU(inplace=True),
            )
            if vlm_channels is not None
            else None
        )
        total_channels = in_channels + (vlm_proj_channels if vlm_channels is not None else 0)
        self.net = nn.Sequential(
            nn.Conv2d(total_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, shallow: torch.Tensor, vlm_spatial: torch.Tensor | None = None) -> torch.Tensor:
        features = [shallow]
        if self.vlm_proj is not None:
            if vlm_spatial is None:
                raise ValueError("vlm_spatial is required when ContextHead was built with vlm_channels")
            vlm_map = self._as_feature_map(vlm_spatial)
            vlm_map = F.interpolate(vlm_map, size=shallow.shape[-2:], mode="bilinear", align_corners=False)
            features.append(self.vlm_proj(vlm_map))
        return torch.sigmoid(self.net(torch.cat(features, dim=1)))

    def _as_feature_map(self, vlm_spatial: torch.Tensor) -> torch.Tensor:
        if vlm_spatial.ndim == 4:
            return vlm_spatial
        if vlm_spatial.ndim != 3:
            raise ValueError(
                "vlm_spatial must be (B, C, H, W) or square-token (B, N, C), "
                f"got {tuple(vlm_spatial.shape)}"
            )

        batch, tokens, channels = vlm_spatial.shape
        side = int(math.sqrt(tokens))
        if side * side != tokens:
            raise ValueError("token-form vlm_spatial must have a square token count")
        return vlm_spatial.transpose(1, 2).reshape(batch, channels, side, side)


class ResidualLUTHead(nn.Module):
    """Small residual LUT layer for VeraRetouch I_base refinement.

    Forward contract:
        i_in: (B, 3, H, W), normalized to [0, 1]
        i_base: (B, 3, H, W), VeraRetouch output normalized to [0, 1]

    The LUT lookup is always applied to i_base, not i_in.
    """

    def __init__(
        self,
        mode: LUTMode = "4d",
        grid_size: int = 17,
        context_size: int = 4,
        basis_count: int | None = None,
        rho: float = 0.10,
        gamma_init: float = -4.0,
        alpha_activation: AlphaActivation = "tanh",
        coeff_feature_dim: int | None = None,
        vlm_channels: int | None = None,
    ):
        super().__init__()
        if mode not in ("3d", "4d"):
            raise ValueError("mode must be '3d' or '4d'")
        if grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if context_size < 2:
            raise ValueError("context_size must be >= 2")
        if alpha_activation not in ("tanh", "softmax", "none"):
            raise ValueError("alpha_activation must be 'tanh', 'softmax', or 'none'")

        self.mode = mode
        self.grid_size = grid_size
        self.context_size = context_size
        self.basis_count = basis_count if basis_count is not None else (48 if mode == "3d" else 8)
        self.rho = float(rho)
        self.alpha_activation = alpha_activation

        if mode == "3d":
            delta_shape = (self.basis_count, grid_size, grid_size, grid_size, 3)
        else:
            delta_shape = (self.basis_count, grid_size, grid_size, grid_size, context_size, 3)
            self.context_head = ContextHead(in_channels=9, vlm_channels=vlm_channels)

        self.delta_basis = nn.Parameter(torch.empty(delta_shape))
        nn.init.normal_(self.delta_basis, mean=0.0, std=1e-4)

        self.image_coeff_head = ImageCoeffHead(in_channels=9, basis_count=self.basis_count)
        self.hidden_coeff_head = (
            HiddenCoeffHead(coeff_feature_dim, self.basis_count) if coeff_feature_dim is not None else None
        )
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(
        self,
        i_in: torch.Tensor,
        i_base: torch.Tensor,
        *,
        vlm_spatial: torch.Tensor | None = None,
        coeff_features: torch.Tensor | None = None,
        context: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | ResidualLUTOutput:
        if i_in.shape != i_base.shape:
            raise ValueError(f"i_in and i_base shapes must match, got {tuple(i_in.shape)} and {tuple(i_base.shape)}")
        if i_in.ndim != 4 or i_in.shape[1] != 3:
            raise ValueError(f"i_in/i_base must have shape (B, 3, H, W), got {tuple(i_in.shape)}")

        shallow = torch.cat((i_in, i_base, i_base - i_in), dim=1)

        alpha_logits = self._predict_alpha(shallow, coeff_features)
        alpha = self._activate_alpha(alpha_logits)
        lut = self._mix_lut(alpha, device=i_base.device, dtype=i_base.dtype)

        if self.mode == "3d":
            context_map = None
            lut_image = apply_lut_3d(lut, i_base)
        else:
            context_map = context
            if context_map is None:
                context_map = self.context_head(shallow, vlm_spatial=vlm_spatial)
            lut_image = apply_lut_4d(lut, i_base, context_map)

        gate = torch.sigmoid(self.gamma).to(dtype=i_base.dtype)
        image = (i_base + gate * (lut_image - i_base)).clamp(0.0, 1.0)

        if not return_aux:
            return image
        return ResidualLUTOutput(
            image=image,
            lut_image=lut_image,
            lut=lut,
            alpha=alpha,
            gate=gate,
            context=context_map,
        )

    def _predict_alpha(self, shallow: torch.Tensor, coeff_features: torch.Tensor | None) -> torch.Tensor:
        if coeff_features is not None:
            if self.hidden_coeff_head is None:
                raise ValueError("coeff_features were passed but coeff_feature_dim was not set")
            return self.hidden_coeff_head(coeff_features)
        return self.image_coeff_head(shallow)

    def _activate_alpha(self, logits: torch.Tensor) -> torch.Tensor:
        if self.alpha_activation == "tanh":
            return torch.tanh(logits)
        if self.alpha_activation == "softmax":
            return F.softmax(logits, dim=1)
        return logits

    def _mix_lut(self, alpha: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        basis = torch.tanh(self.delta_basis).to(device=device, dtype=dtype)
        alpha = alpha.to(device=device, dtype=dtype)

        if self.mode == "3d":
            delta = torch.einsum("bk,kijlc->bijlc", alpha, basis)
            identity = identity_lut_3d(self.grid_size, device=device, dtype=dtype).unsqueeze(0)
        else:
            delta = torch.einsum("bk,kijlmc->bijlmc", alpha, basis)
            identity = identity_lut_4d(
                self.grid_size,
                self.context_size,
                device=device,
                dtype=dtype,
            ).unsqueeze(0)

        return identity + self.rho * delta


def lut_tv(lut: torch.Tensor) -> torch.Tensor:
    """First-order TV regularizer over LUT grid axes."""
    if lut.ndim not in (5, 6):
        raise ValueError(f"lut must be 3D or 4D batched LUT, got {tuple(lut.shape)}")
    grid_dims = range(1, lut.ndim - 1)
    total = lut.new_tensor(0.0)
    for dim in grid_dims:
        total = total + (lut.diff(dim=dim).abs().mean())
    return total


def lut_smoothness2(lut: torch.Tensor) -> torch.Tensor:
    """Second-order smoothness regularizer over LUT grid axes."""
    if lut.ndim not in (5, 6):
        raise ValueError(f"lut must be 3D or 4D batched LUT, got {tuple(lut.shape)}")
    grid_dims = range(1, lut.ndim - 1)
    total = lut.new_tensor(0.0)
    for dim in grid_dims:
        if lut.shape[dim] < 3:
            continue
        total = total + lut.diff(n=2, dim=dim).abs().mean()
    return total
