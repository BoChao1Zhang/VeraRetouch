"""Batch-first multi-head attention with an explicit key mask and no NaN rows.

This is the same implementation as ``q3vl.whereb.connector.MultiheadAttention``
and ``tests/test_attention.py`` pins the two to bit equality on random inputs.
It is *restated* rather than imported for one reason: protocol 14.8 asks for a
proof that ``Q_color`` does not read ``H_where``, and the cheapest proof is that
no module on the colour path contains the identifier at all.  Importing from the
Where-B connector would put ``h_where``, ``gate_text`` and friends one import
edge away from the colour path and turn a structural proof into an argument.

The all-masked row behaviour matters here too: the strict no-where control
(protocol 8.2 ``C01``/``C02``) removes the ``<where>`` prefix, and a sample whose
WC token set is empty must contribute exactly zero from that branch, not a NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MultiheadAttention", "AttentionPool"]


class MultiheadAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % n_heads:
            raise ValueError(f"dim {dim} not divisible by n_heads {n_heads}")
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.reshape(b, n, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor,
                key_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, nq, _ = query.shape
        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key_value))
        v = self._split(self.v_proj(key_value))

        any_valid = None
        attn_mask = None
        if key_mask is not None:
            if key_mask.shape != key_value.shape[:2]:
                raise ValueError(
                    f"key_mask {tuple(key_mask.shape)} does not match keys "
                    f"{tuple(key_value.shape[:2])}"
                )
            km = key_mask.bool()
            any_valid = km.any(dim=1)
            km = torch.where(any_valid.unsqueeze(1), km, torch.ones_like(km))
            attn_mask = km[:, None, None, :]

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        out = self.out_proj(out)
        if any_valid is not None:
            out = out * any_valid.reshape(b, 1, 1).to(out.dtype)
        return out


class AttentionPool(nn.Module):
    """One learnable probe cross-attending a token set -> ``(B, dim)``."""

    def __init__(self, dim: int, n_heads: int, seed: int = 0):
        super().__init__()
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.probe = nn.Parameter(torch.randn(1, dim, generator=g) * (dim ** -0.5))
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, n_heads)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        b = tokens.shape[0]
        q = self.probe.unsqueeze(0).expand(b, -1, -1)
        return self.norm_out(self.attn(q, self.norm_kv(tokens), mask)).squeeze(1)
