#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/align_predictor_time.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · stage0 · sprf · T-ALIGN-TIMEFILM predictor.

Matched stage-conditioning block replacement: the baseline trainable stage
embedding is replaced as a complete block by fixed q, FiLM, and identity
zero-initialization, with exactly the same parameter count.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Keep the production-visible histogram path byte-for-byte equivalent in
# semantics to align_predictor.py.
HIST_STATS = 8


def hist_bins(grid: int) -> int:
    return int(grid) ** 3


def feature_dim(grid: int) -> int:
    return 2 * hist_bins(grid) + HIST_STATS


def _bin_index(c: torch.Tensor, grid: int) -> torch.Tensor:
    q = (c.clamp(0.0, 1.0) * int(grid)).floor().long().clamp_(0, int(grid) - 1)
    r, g, b = q[:, 0], q[:, 1], q[:, 2]
    return (b * int(grid) + g) * int(grid) + r


def hist_features(y_px: torch.Tensor, alphas_px: torch.Tensor,
                  grid: int) -> torch.Tensor:
    """Production-visible beta-weighted/global histograms, identical to baseline."""
    n = hist_bins(grid)
    p = y_px.shape[0]
    idx = _bin_index(y_px, grid)
    ones = torch.ones(p, dtype=torch.float32)
    hg = torch.zeros(n, dtype=torch.float32).index_add_(0, idx, ones) * (n / max(p, 1))
    gmean = y_px.mean(0)
    out = []
    for k in range(alphas_px.shape[0]):
        w = alphas_px[k].to(torch.float32)
        s = float(w.sum())
        hb = torch.zeros(n, dtype=torch.float32).index_add_(0, idx, w)
        hb = hb * (n / s) if s > 0 else hb
        wm = ((y_px * w.unsqueeze(-1)).sum(0) / s) if s > 0 else torch.zeros(3)
        stats = torch.cat([wm, gmean,
                           torch.tensor([w.mean(), (w > 0).float().mean()])])
        out.append(torch.cat([hb, hg, stats]))
    return torch.stack(out)


class LatentPredictor(nn.Module):
    """`(c, feature, fixed stage/time code) -> latent` with TimeFiLM.

    q_m = [1, sin(pi*m/(K-1)), cos(pi*m/(K-1))] is a non-trainable buffer.
    Linear(3, 2H, bias=False) produces gamma/beta and is initialized to zero,
    so the initial modulation is exactly the identity.
    """

    def __init__(self, c_dim: int, feat_dim: int, n_steps: int, latent: int,
                 hidden: int, layers: int):
        super().__init__()
        if int(layers) < 2:
            raise SystemExit(f"align_predictor_time: layers = {layers} 必须 >= 2")
        if int(n_steps) != 6:
            raise SystemExit(f"align_predictor_time: n_steps = {n_steps} 必须 == 6")
        self.proj_c = nn.Linear(int(c_dim), int(hidden))
        self.proj_f = nn.Linear(int(feat_dim), int(hidden))
        # Consume exactly the RNG that baseline emb_stage consumes at this
        # construction point, without registering or retaining the dummy.
        _rng_dummy = nn.Embedding(int(n_steps), int(hidden))
        del _rng_dummy
        t = torch.arange(int(n_steps), dtype=torch.float32) / float(int(n_steps) - 1)
        q = torch.stack([torch.ones_like(t), torch.sin(math.pi * t),
                         torch.cos(math.pi * t)], dim=-1)
        # Runtime-only fixed code: moves with the module but never enters a
        # state_dict/checkpoint. The formal checkpoint contains parameters only.
        self.register_buffer("time_code", q, persistent=False)
        mid: list[nn.Module] = []
        for _ in range(int(layers) - 1):
            mid += [nn.Linear(int(hidden), int(hidden)), nn.SiLU()]
        self.trunk = nn.Sequential(*mid)
        self.head = nn.Linear(int(hidden), int(latent))
        # Construct after trunk/head so all non-stage parameters are paired
        # bit-for-bit with baseline. Its own RNG is irrelevant after zero_.
        self.time_affine = nn.Linear(3, 2 * int(hidden), bias=False)
        nn.init.zeros_(self.time_affine.weight)
        self.n_steps, self.latent, self.hidden = (
            int(n_steps), int(latent), int(hidden))

    def modulation(self) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.time_affine(self.time_code).chunk(2, dim=-1)
        return gamma, beta

    def forward(self, c: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        b, k, _ = feat.shape
        if k != self.n_steps:
            raise SystemExit(f"align_predictor_time: 阶段数 {k} != {self.n_steps}")
        u = self.proj_c(c).unsqueeze(1) + self.proj_f(feat)
        gamma, beta = self.modulation()
        h = (1.0 + gamma.unsqueeze(0)) * u + beta.unsqueeze(0)
        return self.head(self.trunk(F.silu(h)))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


@torch.no_grad()
def build_target_latents(edit_enc: nn.Module, inv_table: torch.Tensor,
                         chunk: int = 256) -> torch.Tensor:
    outs = []
    for i in range(0, inv_table.shape[0], int(chunk)):
        outs.append(edit_enc(inv_table[i:i + int(chunk)]))
    return torch.cat(outs)
