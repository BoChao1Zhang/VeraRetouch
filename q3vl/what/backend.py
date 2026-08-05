"""Protocol 7.3 -- the shared 48-slot Transformer backend.

    48 learned slot embeddings + WC tokens
     -> 2 seed Transformer blocks
     -> provisional/fixed Gaussian geometry
     -> Gaussian-aligned pooling v_i
     -> 4 refinement Transformer blocks
     -> 48 independent decoder heads + one global head

    "Each block is width 512, 8 heads, FFN 2048, pre-norm.  Every block:
     1. self-attends across the 48 slots, modelling complementarity and
        competition between the Gaussians;
     2. cross-attends the complete ``M_color``, so the semantics are not squeezed
        into a single vector;
     3. receives the projected ``v_i`` and the WC tokens;
     4. is modulated by ModLN scale/shift generated from ``z_style``;
     5. uses a zero-init gated residual, so neither the visual nor the Where
        branch overwhelms the colour semantics early in training."

Both generators use *this* backend at *this* size (protocol 7.3's opening
sentence), so the FG/SB comparison is "full generation vs shared geometry" and
not "transformer vs small MLP".

The 48 decoder heads are batched parameters ``(48, in, out)`` contracted with an
einsum rather than 48 ``nn.Linear`` modules.  That is the same set of 48
independent affine maps -- no weight is shared across slots -- expressed as one
kernel launch; ``tests/test_backend.py`` asserts head ``i``'s output depends on
slot ``i`` only.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

from .attention import MultiheadAttention
from .config import BackendConfig

__all__ = ["ModLN", "SlotBlock", "SlotBackend", "SlotHeads", "n_head_params"]


class ModLN(nn.Module):
    """LayerNorm whose scale/shift come from ``z_style`` (zero-init projection).

    At initialisation the projection is exactly zero, so the module *is* a plain
    LayerNorm and the style code cannot destabilise the first steps.  Every
    sub-layer owns its own projection (RD-G / PLAN 1.4 precedent: per-sub-layer
    ModLN parameters are what makes the style code interpolatable).
    """

    def __init__(self, dim: int, style_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.proj = nn.Linear(style_dim, 2 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        scale, shift = self.proj(style).unsqueeze(1).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale) + shift


class SlotBlock(nn.Module):
    """One pre-norm block: self-attn, cross(M_color), cross(WC), v_i, FFN.

    Gates: every *injection* branch has its own zero-initialised scalar gate
    (protocol 7.3 item 5).  Self-attention and the FFN are plain residuals, the
    same convention protocol 5.1 uses for the Where connector -- only newly
    injected information is gated.
    """

    def __init__(self, cfg: BackendConfig, use_v: bool):
        super().__init__()
        d = cfg.dim
        self.use_v = bool(use_v)
        self.mod_self = ModLN(d, cfg.z_style_dim)
        self.self_attn = MultiheadAttention(d, cfg.n_heads, cfg.dropout)

        self.mod_q_color = ModLN(d, cfg.z_style_dim)
        self.norm_kv_color = nn.LayerNorm(d)
        self.cross_color = MultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.gate_color = nn.Parameter(torch.zeros(1))

        self.mod_q_wc = ModLN(d, cfg.z_style_dim)
        self.norm_kv_wc = nn.LayerNorm(d)
        self.cross_wc = MultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.gate_wc = nn.Parameter(torch.zeros(1))

        if use_v:
            self.norm_v = nn.LayerNorm(d)
            self.gate_v = nn.Parameter(torch.zeros(1))

        self.mod_ffn = ModLN(d, cfg.z_style_dim)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn), nn.GELU(),
                                 nn.Linear(cfg.ffn, d))

    def forward(self, q: torch.Tensor, style: torch.Tensor,
                m_color: torch.Tensor, m_color_mask: torch.Tensor | None,
                wc: torch.Tensor, wc_mask: torch.Tensor | None,
                v: torch.Tensor | None = None) -> torch.Tensor:
        h = self.mod_self(q, style)
        q = q + self.self_attn(h, h, None)
        q = q + self.gate_color * self.cross_color(
            self.mod_q_color(q, style), self.norm_kv_color(m_color), m_color_mask)
        q = q + self.gate_wc * self.cross_wc(
            self.mod_q_wc(q, style), self.norm_kv_wc(wc), wc_mask)
        if self.use_v:
            if v is None:
                raise ValueError("a refinement block was called without v_i")
            q = q + self.gate_v * self.norm_v(v)
        return q + self.ffn(self.mod_ffn(q, style))


class SlotBackend(nn.Module):
    """48 slots -> ``h_i`` after the seed stage and after the refinement stage."""

    def __init__(self, cfg: BackendConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.slots = nn.Parameter(
            torch.randn(cfg.n_slots, cfg.dim, generator=g) * (cfg.dim ** -0.5))
        self.v_proj = nn.Linear(cfg.v_dim, cfg.dim)
        self.seed_blocks = nn.ModuleList(
            SlotBlock(cfg, use_v=False) for _ in range(cfg.seed_blocks))
        self.refine_blocks = nn.ModuleList(
            SlotBlock(cfg, use_v=True) for _ in range(cfg.refine_blocks))
        self.norm_out = nn.LayerNorm(cfg.dim)

    def initial(self, batch: int) -> torch.Tensor:
        return self.slots.unsqueeze(0).expand(batch, -1, -1)

    def run_seed(self, style, m_color, m_color_mask, wc, wc_mask,
                 q: torch.Tensor | None = None) -> torch.Tensor:
        h = self.initial(style.shape[0]) if q is None else q
        for blk in self.seed_blocks:
            h = blk(h, style, m_color, m_color_mask, wc, wc_mask)
        return h

    def run_refine(self, h: torch.Tensor, v: torch.Tensor, style, m_color,
                   m_color_mask, wc, wc_mask) -> torch.Tensor:
        vp = self.v_proj(v)
        for blk in self.refine_blocks:
            h = blk(h, style, m_color, m_color_mask, wc, wc_mask, vp)
        return self.norm_out(h)

    def gate_values(self) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {"gate_color": [], "gate_wc": [], "gate_v": []}
        for blk in list(self.seed_blocks) + list(self.refine_blocks):
            out["gate_color"].append(float(blk.gate_color.detach()))
            out["gate_wc"].append(float(blk.gate_wc.detach()))
            if blk.use_v:
                out["gate_v"].append(float(blk.gate_v.detach()))
        return out


def n_head_params(in_dim: int, bottleneck: int, out_dim: int, n_slots: int) -> int:
    """Trainable parameters of the 48 independent heads (weights + biases)."""
    return n_slots * (in_dim * bottleneck + bottleneck + bottleneck * out_dim + out_dim)


class SlotHeads(nn.Module):
    """48 independent decoder heads + one global head.

    Head ``i`` reads ``[h_i, P(z_style)]`` (protocol 7.3) and is
    ``Linear -> GELU -> Linear`` with the **second Linear zero-initialised**, so
    every arm's first forward is exactly the identity LUT (protocol 7.6's
    identity-centred parameterisation is only an identity when the raw outputs
    are zero).  The bottleneck is the knob protocol 7.5 names for equalising the
    two generators' trainable parameter counts to within 2%.
    """

    def __init__(self, cfg: BackendConfig, n_out: int, bottleneck: int,
                 n_global: int = 12, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.n_slots = cfg.n_slots
        self.n_out = int(n_out)
        self.bottleneck = int(bottleneck)
        self.style_proj = nn.Linear(cfg.z_style_dim, cfg.z_style_head_proj)
        in_dim = cfg.dim + cfg.z_style_head_proj
        self.in_dim = in_dim

        g = torch.Generator(device="cpu").manual_seed(seed)
        std = 1.0 / math.sqrt(in_dim)
        self.w1 = nn.Parameter(
            torch.randn(self.n_slots, in_dim, self.bottleneck, generator=g) * std)
        self.b1 = nn.Parameter(torch.zeros(self.n_slots, self.bottleneck))
        self.w2 = nn.Parameter(torch.zeros(self.n_slots, self.bottleneck, self.n_out))
        self.b2 = nn.Parameter(torch.zeros(self.n_slots, self.n_out))

        self.global_head = nn.Sequential(
            nn.Linear(in_dim, self.bottleneck), nn.GELU(),
            nn.Linear(self.bottleneck, n_global))
        nn.init.zeros_(self.global_head[-1].weight)
        nn.init.zeros_(self.global_head[-1].bias)
        self.act = nn.GELU()

    def forward(self, h: torch.Tensor, style: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        p = self.style_proj(style)                                   # (B, Ps)
        x = torch.cat([h, p.unsqueeze(1).expand(-1, self.n_slots, -1)], dim=-1)
        y = self.act(torch.einsum("bni,nio->bno", x, self.w1) + self.b1)
        z_prim = torch.einsum("bnk,nko->bno", y, self.w2) + self.b2
        z_glob = self.global_head(torch.cat([h.mean(1), p], dim=-1))
        return z_prim, z_glob

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def facts(self) -> dict[str, Any]:
        return {
            "n_slots": self.n_slots, "n_out_per_slot": self.n_out,
            "bottleneck": self.bottleneck, "head_in_dim": self.in_dim,
            "n_params": self.n_params(),
            "per_slot_params": n_head_params(self.in_dim, self.bottleneck,
                                             self.n_out, 1),
        }
