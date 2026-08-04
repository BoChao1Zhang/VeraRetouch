"""Protocol 5.1 -- the Where connector.

    "The connector width is fixed at 512, with 6 pre-norm Transformer blocks,
     8 heads and FFN 2048:

         self-attn(Q) -> cross-attn(Q, H_where) -> cross-attn(Q, F_pre) -> FFN

     Each cross-attention residual gate uses zero initialisation; ``H_where``
     and ``F_pre`` are linearly projected to 512."

Two things this module is deliberately strict about:

1. **No ``H_color`` anywhere.**  There is no parameter, buffer or argument in
   this file that could carry it; :meth:`ConnectorStream.forward` takes exactly
   ``(queries, h_where, h_where_mask, f_pre, f_pre_pos, f_pre_mask)``.  Protocol
   14.8 asks for a *proof*, and the cheapest proof is a signature that cannot
   express the thing.
2. **A fully-masked key set returns exactly zero, not NaN.**  The ``null``
   context (protocol 5.4) has no ``<where>`` tokens at all.  A softmax over an
   all-masked row is ``0/0``; silently letting that become NaN would turn one of
   the four mandatory evaluation contexts into a crash or, worse, into a
   poisoned gradient.  Rows with no valid key are computed with an all-valid
   mask and then zeroed, which makes the ``null`` context exactly "this
   cross-attention branch contributes nothing".
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ConnectorConfig
from .qwhere import PositionEncoder

__all__ = ["MultiheadAttention", "ConnectorBlock", "ConnectorStream"]


class MultiheadAttention(nn.Module):
    """Batch-first MHA with an explicit key-padding mask and no NaN rows."""

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

    def forward(
        self,
        query: torch.Tensor,                    # (B, Nq, C)
        key_value: torch.Tensor,                # (B, Nk, C)
        key_mask: torch.Tensor | None = None,   # (B, Nk) bool, True = valid
    ) -> torch.Tensor:
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
            any_valid = km.any(dim=1)                       # (B,)
            # a sample with no valid key gets an all-valid mask here and a zeroed
            # output below; that keeps the softmax well defined.
            km = torch.where(any_valid.unsqueeze(1), km, torch.ones_like(km))
            attn_mask = km[:, None, None, :]                 # (B, 1, 1, Nk)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(b, nq, self.dim)
        out = self.out_proj(out)
        if any_valid is not None:
            out = out * any_valid.reshape(b, 1, 1).to(out.dtype)
        return out


class ConnectorBlock(nn.Module):
    """One pre-norm block in the protocol 5.1 order, with zero-init cross gates."""

    def __init__(self, cfg: ConnectorConfig):
        super().__init__()
        d = cfg.dim
        self.norm_self = nn.LayerNorm(d)
        self.self_attn = MultiheadAttention(d, cfg.n_heads, cfg.dropout)

        self.norm_q_text = nn.LayerNorm(d)
        self.norm_kv_text = nn.LayerNorm(d)
        self.cross_text = MultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.gate_text = nn.Parameter(torch.zeros(1))     # zero init (protocol 5.1)

        self.norm_q_vis = nn.LayerNorm(d)
        self.norm_kv_vis = nn.LayerNorm(d)
        self.cross_vis = MultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.gate_vis = nn.Parameter(torch.zeros(1))      # zero init (protocol 5.1)

        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, cfg.ffn), nn.GELU(), nn.Linear(cfg.ffn, d)
        )

    def forward(
        self,
        q: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor | None,
        vis: torch.Tensor,
        vis_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        h = self.norm_self(q)
        q = q + self.self_attn(h, h, None)

        q = q + self.gate_text * self.cross_text(
            self.norm_q_text(q), self.norm_kv_text(text), text_mask
        )
        q = q + self.gate_vis * self.cross_vis(
            self.norm_q_vis(q), self.norm_kv_vis(vis), vis_mask
        )
        return q + self.ffn(self.norm_ffn(q))


class ConnectorStream(nn.Module):
    """Input projections + ``n_blocks`` blocks.  One stream = one canvas.

    ``MC16-DualCanvas`` instantiates two of these; every other structure has one.
    The projections live *inside* the stream on purpose: protocol 5.2 says the
    dual-canvas arm shares only the frozen VLM.
    """

    def __init__(self, cfg: ConnectorConfig):
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Linear(cfg.text_dim, cfg.dim)
        self.vision_proj = nn.Linear(cfg.vision_dim, cfg.dim)
        self.pos = PositionEncoder(cfg.dim, cfg.pos_bands, cfg.pos_max_freq)
        self.blocks = nn.ModuleList(ConnectorBlock(cfg) for _ in range(cfg.n_blocks))
        self.norm_out = nn.LayerNorm(cfg.dim)

    def forward(
        self,
        queries: torch.Tensor,                      # (B, Nq, C)
        h_where: torch.Tensor,                      # (B, T, text_dim)
        h_where_mask: torch.Tensor | None,          # (B, T) bool
        f_pre: torch.Tensor,                        # (B, P, vision_dim)
        f_pre_pos: torch.Tensor,                    # (B, P, 2)
        f_pre_mask: torch.Tensor | None,            # (B, P) bool
    ) -> torch.Tensor:
        text = self.text_proj(h_where)
        vis = self.vision_proj(f_pre) + self.pos(f_pre_pos.to(f_pre.dtype))
        q = queries
        for blk in self.blocks:
            q = blk(q, text, h_where_mask, vis, f_pre_mask)
        return self.norm_out(q)

    # -- protocol 14.8 evidence --------------------------------------------
    def gate_values(self) -> dict[str, list[float]]:
        return {
            "gate_text": [float(b.gate_text.detach()) for b in self.blocks],
            "gate_vis": [float(b.gate_vis.detach()) for b in self.blocks],
        }
