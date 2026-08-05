"""Protocol 7.1 -- ``Q_color`` and the continuous style code.

    "``Q_color`` is completely independent of ``Q_where``.  16 learnable colour
     queries cross-attend only the ``<color>...</color>`` hidden states:

         M_color = ColorConnector(Q_color, H_color) in R^[16 x 512]
         z_style = LN(MLP(AttentionPool(M_color)))  in R^1024

     ``Q_color`` does not read ``H_where`` directly.  Where information enters
     only through the four explicit WC interfaces, so each interface's gain can
     be attributed."

Protocol 14.8 asks for a *proof* of the second sentence.  This module is written
so the proof is structural and mechanical:

* :meth:`ColorConnector.forward` takes exactly ``(queries, h_color,
  h_color_mask)``.  There is no argument, buffer, parameter or attribute here
  that could carry ``H_where``, ``F_pre``, ``m_pred``, ``w``, ``rho`` or
  ``I_tar``;
* no identifier in this module or in :mod:`q3vl.what.attention` contains
  ``where``, which is what :func:`q3vl.what.preflight.check_no_h_where` scans
  for.

The queries carry **no** 2-D position: ``Q_where`` is a canvas over an image and
needs one (protocol 5.1), while ``Q_color`` reads a token sequence and a
positional prior over 16 abstract colour slots would be an invented structure.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from .attention import AttentionPool, MultiheadAttention
from .config import ColorConnectorConfig

__all__ = ["ColorQueryBank", "ColorConnectorBlock", "ColorConnector", "StyleHead",
           "ColorStack"]


class ColorQueryBank(nn.Module):
    """``n_queries`` learnable tokens, independent of ``Q_where``'s bank."""

    def __init__(self, n_queries: int, dim: int, seed: int = 0):
        super().__init__()
        self.n_queries = int(n_queries)
        self.dim = int(dim)
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.tokens = nn.Parameter(
            torch.randn(self.n_queries, dim, generator=g) * (dim ** -0.5)
        )

    def forward(self, batch: int) -> torch.Tensor:
        return self.tokens.unsqueeze(0).expand(batch, -1, -1)


class ColorConnectorBlock(nn.Module):
    """pre-norm ``self-attn -> cross-attn(H_color) -> FFN``, zero-init cross gate.

    Same block shape as protocol 5.1's Where connector minus the ``F_pre``
    cross-attention, which protocol 7.1 forbids here: the colour queries read
    language only, and every visual signal arrives later through the aligned
    pooling and the WC tokens.
    """

    def __init__(self, cfg: ColorConnectorConfig):
        super().__init__()
        d = cfg.dim
        self.norm_self = nn.LayerNorm(d)
        self.self_attn = MultiheadAttention(d, cfg.n_heads, cfg.dropout)

        self.norm_q_text = nn.LayerNorm(d)
        self.norm_kv_text = nn.LayerNorm(d)
        self.cross_text = MultiheadAttention(d, cfg.n_heads, cfg.dropout)
        self.gate_text = nn.Parameter(torch.zeros(1))       # zero init

        self.norm_ffn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, cfg.ffn), nn.GELU(),
                                 nn.Linear(cfg.ffn, d))

    def forward(self, q: torch.Tensor, text: torch.Tensor,
                text_mask: torch.Tensor | None) -> torch.Tensor:
        h = self.norm_self(q)
        q = q + self.self_attn(h, h, None)
        q = q + self.gate_text * self.cross_text(
            self.norm_q_text(q), self.norm_kv_text(text), text_mask
        )
        return q + self.ffn(self.norm_ffn(q))


class ColorConnector(nn.Module):
    """``(Q_color, H_color) -> M_color in R^[B x 16 x 512]``."""

    def __init__(self, cfg: ColorConnectorConfig):
        super().__init__()
        self.cfg = cfg
        self.text_proj = nn.Linear(cfg.text_dim, cfg.dim)
        self.blocks = nn.ModuleList(ColorConnectorBlock(cfg) for _ in range(cfg.n_blocks))
        self.norm_out = nn.LayerNorm(cfg.dim)

    def forward(self, queries: torch.Tensor, h_color: torch.Tensor,
                h_color_mask: torch.Tensor | None) -> torch.Tensor:
        text = self.text_proj(h_color)
        q = queries
        for blk in self.blocks:
            q = blk(q, text, h_color_mask)
        return self.norm_out(q)

    def gate_values(self) -> dict[str, list[float]]:
        return {"gate_text": [float(b.gate_text.detach()) for b in self.blocks]}


class StyleHead(nn.Module):
    """``z_style = LN(MLP(AttentionPool(M_color))) in R^1024``."""

    def __init__(self, cfg: ColorConnectorConfig, seed: int = 0):
        super().__init__()
        self.pool = AttentionPool(cfg.dim, cfg.n_heads, seed=seed)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.dim, cfg.z_style_hidden), nn.GELU(),
            nn.Linear(cfg.z_style_hidden, cfg.z_style_dim),
        )
        self.norm = nn.LayerNorm(cfg.z_style_dim)

    def forward(self, m_color: torch.Tensor) -> torch.Tensor:
        return self.norm(self.mlp(self.pool(m_color)))


class ColorStack(nn.Module):
    """Bank + connector + style head: everything protocol 7.1 defines, together."""

    def __init__(self, cfg: ColorConnectorConfig, seed: int = 0):
        super().__init__()
        self.cfg = cfg
        self.bank = ColorQueryBank(cfg.n_queries, cfg.dim, seed=seed)
        self.connector = ColorConnector(cfg)
        self.style = StyleHead(cfg, seed=seed + 3)

    def forward(self, h_color: torch.Tensor,
                h_color_mask: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.bank(h_color.shape[0])
        m_color = self.connector(q, h_color, h_color_mask)
        return m_color, self.style(m_color)

    def facts(self) -> dict[str, Any]:
        return {
            "n_queries": self.bank.n_queries,
            "dim": self.cfg.dim,
            "n_blocks": self.cfg.n_blocks,
            "z_style_dim": self.cfg.z_style_dim,
            "n_params": sum(p.numel() for p in self.parameters()),
            "reads": ["H_color"],
        }
