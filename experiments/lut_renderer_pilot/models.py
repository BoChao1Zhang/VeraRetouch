from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


@dataclass
class GaussianParameters:
    opacity: torch.Tensor
    local_matrix: torch.Tensor
    local_bias: torch.Tensor
    global_matrix: torch.Tensor
    global_bias: torch.Tensor


class GaussianParameterGenerator(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, num_gaussians: int) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.opacity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_gaussians),
        )
        self.local_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 12 * num_gaussians),
        )
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 12),
        )

    def initialize_identity(self, opacity_initial: float) -> None:
        opacity_output = self.opacity_head[-1]
        local_output = self.local_head[-1]
        global_output = self.global_head[-1]
        assert isinstance(opacity_output, nn.Linear)
        assert isinstance(local_output, nn.Linear)
        assert isinstance(global_output, nn.Linear)
        identity = torch.eye(3).reshape(-1)
        local_bias = torch.cat((identity, torch.zeros(3))).repeat(self.num_gaussians)
        global_bias = torch.cat((identity, torch.zeros(3)))
        with torch.no_grad():
            opacity_output.weight.zero_()
            opacity_output.bias.fill_(math.log(opacity_initial / (1.0 - opacity_initial)))
            local_output.weight.zero_()
            local_output.bias.copy_(local_bias)
            global_output.weight.zero_()
            global_output.bias.copy_(global_bias)

    def forward(self, embedding: torch.Tensor) -> GaussianParameters:
        features = self.encoder(embedding)
        opacity = torch.sigmoid(self.opacity_head(features))
        local = self.local_head(features).reshape(-1, self.num_gaussians, 12)
        global_parameters = self.global_head(features)
        return GaussianParameters(
            opacity=opacity,
            local_matrix=local[..., :9].reshape(-1, self.num_gaussians, 3, 3),
            local_bias=local[..., 9:],
            global_matrix=global_parameters[..., :9].reshape(-1, 3, 3),
            global_bias=global_parameters[..., 9:],
        )


class SharedGeometryCGLUT(nn.Module):
    min_cholesky_diagonal = 1.0e-4

    def __init__(
        self,
        num_styles: int,
        *,
        num_gaussians: int = 32,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        sobol_seed: int = 1701,
        covariance_sigma: float = 0.15,
        opacity_initial: float = 0.9999,
        epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if num_gaussians < 8:
            raise ValueError("CGLUT needs at least the eight RGB cube corners")
        self.num_styles = num_styles
        self.num_gaussians = num_gaussians
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon
        self.style_embeddings = nn.Embedding(num_styles, embedding_dim)
        self.generator = GaussianParameterGenerator(
            embedding_dim, hidden_dim, num_gaussians
        )

        corners = torch.tensor(
            list(itertools.product((0.0, 1.0), repeat=3)), dtype=torch.float32
        )
        remaining = num_gaussians - len(corners)
        if remaining:
            sobol = torch.quasirandom.SobolEngine(3, scramble=True, seed=sobol_seed)
            centers = torch.cat((corners, sobol.draw(remaining)), dim=0)
        else:
            centers = corners
        self.means = nn.Parameter(centers)

        raw_cholesky = torch.zeros((num_gaussians, 6), dtype=torch.float32)
        raw_diagonal = math.log(covariance_sigma - self.min_cholesky_diagonal)
        raw_cholesky[:, (0, 2, 5)] = raw_diagonal
        self.raw_cholesky = nn.Parameter(raw_cholesky)

        nn.init.normal_(self.style_embeddings.weight, mean=0.0, std=0.02)
        self.generator.initialize_identity(opacity_initial)

    @classmethod
    def from_config(cls, num_styles: int, config: dict[str, Any]) -> "SharedGeometryCGLUT":
        return cls(
            num_styles,
            num_gaussians=int(config["num_gaussians"]),
            embedding_dim=int(config["embedding_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            sobol_seed=int(config["sobol_seed"]),
            covariance_sigma=float(config["covariance_sigma"]),
            opacity_initial=float(config["opacity_initial"]),
            epsilon=float(config["epsilon"]),
        )

    def cholesky(self) -> torch.Tensor:
        raw = self.raw_cholesky
        lower = raw.new_zeros((self.num_gaussians, 3, 3))
        lower[:, 0, 0] = raw[:, 0].exp() + self.min_cholesky_diagonal
        lower[:, 1, 0] = raw[:, 1]
        lower[:, 1, 1] = raw[:, 2].exp() + self.min_cholesky_diagonal
        lower[:, 2, 0] = raw[:, 3]
        lower[:, 2, 1] = raw[:, 4]
        lower[:, 2, 2] = raw[:, 5].exp() + self.min_cholesky_diagonal
        return lower

    def generated_parameters(self, style_indices: torch.Tensor) -> GaussianParameters:
        return self.generator(self.style_embeddings(style_indices))

    def forward_points(self, points: torch.Tensor, style_indices: torch.Tensor) -> torch.Tensor:
        """Render `[B,P,3]` float sRGB points for one style per batch row."""
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("points must have shape [B,P,3]")
        if style_indices.shape != (points.shape[0],):
            raise ValueError("style_indices must have shape [B]")

        generated = self.generated_parameters(style_indices)
        # Density evaluation stays FP32 even when the surrounding Vera path uses BF16.
        x = points.float()
        means = self.means.float()
        lower = self.cholesky().float()
        inverse_lower = torch.linalg.inv(lower)
        difference = x[:, :, None, :] - means[None, None, :, :]
        whitened = torch.einsum("bpni,nji->bpnj", difference, inverse_lower)
        mahalanobis = whitened.square().sum(dim=-1)
        log_determinant = torch.log(torch.diagonal(lower, dim1=-2, dim2=-1)).sum(-1)
        log_density = (
            -0.5 * mahalanobis
            - 1.5 * math.log(2.0 * math.pi)
            - log_determinant[None, None, :]
        )
        opacity = generated.opacity.float().clamp_min(torch.finfo(torch.float32).tiny)
        log_unnormalized = log_density + opacity.log()[:, None, :]
        log_sum = torch.logsumexp(log_unnormalized, dim=-1, keepdim=True)
        log_epsilon = torch.full_like(log_sum, math.log(self.epsilon))
        log_denominator = torch.logaddexp(log_sum, log_epsilon)
        weights = torch.exp(log_unnormalized - log_denominator)

        identity = torch.eye(3, device=x.device, dtype=torch.float32)
        local_delta_matrix = generated.local_matrix.float() - identity[None, None]
        local_delta = torch.einsum("bnij,bpj->bpni", local_delta_matrix, x)
        local_delta = local_delta + generated.local_bias.float()[:, None, :, :]
        local = (weights[..., None] * local_delta).sum(dim=2)
        global_output = torch.einsum(
            "bij,bpj->bpi", generated.global_matrix.float(), x
        ) + generated.global_bias.float()[:, None, :]
        return (global_output + local).clamp(0.0, 1.0)

    def regularization(
        self, style_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generated = self.generated_parameters(style_indices)
        opacity = generated.opacity
        entropy = -(
            opacity * torch.log(opacity + self.epsilon)
            + (1.0 - opacity) * torch.log(1.0 - opacity + self.epsilon)
        ).mean()
        embedding_l2 = self.style_embeddings(style_indices).square().sum(dim=-1).mean()
        return entropy, embedding_l2

    def optimizer_groups(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"params": self.generator.parameters(), "lr": float(config["generator_lr"])},
            {"params": self.style_embeddings.parameters(), "lr": float(config["embedding_lr"])},
            {
                "params": [self.means, self.raw_cholesky],
                "lr": float(config["geometry_lr"]),
            },
        ]

    def shared_parameter_count(self) -> int:
        embedding_parameter = self.style_embeddings.weight
        return sum(parameter.numel() for parameter in self.parameters()) - embedding_parameter.numel()


class VeraConditionalMLPDecoder(nn.Module):
    def __init__(self, style_dim: int = 2688, hidden_dims: tuple[int, ...] = (128, 256, 512)) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        input_dim = 3
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        self.out_layer = nn.Linear(input_dim, 3)
        self.z_projs = nn.ModuleList(
            [nn.Linear(style_dim, hidden_dim) for hidden_dim in hidden_dims]
        )
        self.layer_norms = nn.ModuleList([nn.LayerNorm(value) for value in hidden_dims])

    def forward_points(self, points: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        value = points.mul(2.0).sub(1.0)
        activated_style = F.relu(style)
        for layer, layer_norm, projection in zip(
            self.layers, self.layer_norms, self.z_projs, strict=True
        ):
            value = layer_norm(F.relu(layer(value)))
            value = value + projection(activated_style).unsqueeze(1)
        return torch.sigmoid(self.out_layer(value))


class VeraStyleRenderer(nn.Module):
    def __init__(
        self,
        num_styles: int,
        checkpoint_path: str | Path,
        *,
        style_dim: int = 2688,
        hidden_dims: tuple[int, ...] = (128, 256, 512),
    ) -> None:
        super().__init__()
        self.num_styles = num_styles
        self.style_dim = style_dim
        self.style_embeddings = nn.Embedding(num_styles, style_dim)
        self.decoder = VeraConditionalMLPDecoder(style_dim, hidden_dims)
        nn.init.zeros_(self.style_embeddings.weight)
        self.load_official_decoder(checkpoint_path)

    @classmethod
    def from_config(
        cls, num_styles: int, checkpoint_path: str | Path, config: dict[str, Any]
    ) -> "VeraStyleRenderer":
        return cls(
            num_styles,
            checkpoint_path,
            style_dim=int(config["style_dim"]),
            hidden_dims=tuple(int(value) for value in config["hidden_dims"]),
        )

    def load_official_decoder(self, checkpoint_path: str | Path) -> None:
        checkpoint = Path(checkpoint_path)
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        prefix = "retouch_decoder."
        state: dict[str, torch.Tensor] = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key.removeprefix(prefix)] = handle.get_tensor(key)
        expected = set(self.decoder.state_dict())
        actual = set(state)
        if expected != actual:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                f"Vera decoder checkpoint mismatch: missing={missing}, unexpected={unexpected}"
            )
        self.decoder.load_state_dict(state, strict=True)

    def forward_points(self, points: torch.Tensor, style_indices: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward_points(points, self.style_embeddings(style_indices))

    def optimizer_groups(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"params": self.decoder.parameters(), "lr": float(config["decoder_lr"])},
            {"params": self.style_embeddings.parameters(), "lr": float(config["embedding_lr"])},
        ]

