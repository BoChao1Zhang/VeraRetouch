"""The shared basis projector ``B: 1024 -> 64`` (protocol 3 / 4.2 / 4.4).

``BA-0-Fixed`` uses a *seeded orthogonal* ``B`` and never trains it; the other
three arms train exactly this one tensor and nothing else.  No bias: the
64 semantic channels are per-image zero-meaned right after the projection
(protocol 4.2), so a bias term would be structurally unidentifiable.
"""

from __future__ import annotations

import hashlib
from typing import Any

import torch
import torch.nn as nn

from .config import FPRE_DIM, PROJECTOR_SEED, SEM_DIM

__all__ = ["BasisProjector"]


class BasisProjector(nn.Module):
    def __init__(
        self,
        in_dim: int = FPRE_DIM,
        out_dim: int = SEM_DIM,
        seed: int = PROJECTOR_SEED,
        init: str = "orthogonal",
    ):
        super().__init__()
        self.in_dim, self.out_dim, self.seed, self.init = in_dim, out_dim, seed, init
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        if self.init == "orthogonal":
            # nn.init.orthogonal_ ignores an explicit generator in some
            # versions, so the random matrix is drawn here and orthogonalised
            # by QR: rows end up orthonormal and the draw is reproducible.
            a = torch.randn(self.in_dim, self.out_dim, generator=g)
            q, r = torch.linalg.qr(a)
            q = q * torch.sign(torch.diagonal(r)).unsqueeze(0)   # sign-fixed QR
            with torch.no_grad():
                self.weight.copy_(q.transpose(0, 1).contiguous())
        elif self.init == "normal":
            with torch.no_grad():
                self.weight.copy_(
                    torch.randn(self.out_dim, self.in_dim, generator=g) / self.in_dim**0.5
                )
        else:
            raise ValueError(f"unknown init {self.init!r}")

    def forward(self, fpre: torch.Tensor) -> torch.Tensor:
        """``(..., 1024) -> (..., 64)``."""
        if fpre.shape[-1] != self.in_dim:
            raise ValueError(f"expected last dim {self.in_dim}, got {tuple(fpre.shape)}")
        return torch.nn.functional.linear(fpre.to(self.weight.dtype), self.weight)

    # -- provenance ---------------------------------------------------------
    def digest(self) -> str:
        w = self.weight.detach().to(torch.float32).cpu().contiguous()
        return hashlib.sha256(w.numpy().tobytes()).hexdigest()

    def orthogonality_error(self) -> float:
        """``max |W W^T - I|`` -- 0 for the seeded orthogonal init."""
        with torch.no_grad():
            w = self.weight.detach().double()
            g = w @ w.transpose(0, 1)
            return float((g - torch.eye(g.shape[0], dtype=g.dtype, device=g.device)).abs().max())

    def facts(self) -> dict[str, Any]:
        with torch.no_grad():
            sv = torch.linalg.svdvals(self.weight.detach().double())
        return {
            "in_dim": self.in_dim, "out_dim": self.out_dim, "seed": self.seed,
            "init": self.init, "digest": self.digest(),
            "orthogonality_error": self.orthogonality_error(),
            "singular_value_min": float(sv.min()), "singular_value_max": float(sv.max()),
            "condition_number": float(sv.max() / sv.min().clamp_min(1e-30)),
            "n_params": int(self.weight.numel()),
        }
