"""PCH: Prototype-token Cross-attention + Hypernetwork geometry injection.

The shared injection module for every arm of the geometry-injection proposal.
It exists because broadcasting a low-dimensional code as constant channels --
what the B2 arms do -- is the *lower bound* form of injection: every spatial
position receives the identical code, so the network must discover on its own
that "oval" should shape the field differently at the centre than at the edge.

PCH gives the code a structured route instead:

    code (21-d)  --hypernetwork-->  modulates K prototype tokens
    dense tower features  --cross-attention-->  attends over those tokens
    result added residually back into the tower

Two properties this buys over broadcast:

* **spatially selective** -- attention lets each cell draw on the prototypes it
  needs, so "a broad band, horizontal" can act differently along and across the
  band without the tower having to re-derive that from a constant channel;
* **compositional** -- the hypernetwork modulates a shared prototype bank rather
  than selecting from a fixed table, so unseen slot combinations
  (shape_band + dir_diagonal + ext_moderate) interpolate instead of falling off
  a lookup.

Graceful fallback (proposal §D-1) is not decoration.  The deployable code is
parsed from *generated* text at ~82% word accuracy, and a sample whose `<where>`
named no geometry yields an all-zero code.  Injecting hard on an empty or
near-empty code would be injecting noise, so the module scales its own output by
a confidence derived from code density and **returns exactly zero contribution
at zero density** -- i.e. it degrades to the un-injected model rather than to a
random one.

Sizing: ``PCHConfig.full()`` ~3.9M params, ``PCHConfig.lite()`` ~0.5M.  Both are
reported by :meth:`PCH.facts` so a run records which it used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PCHConfig", "PCH"]


@dataclass(frozen=True)
class PCHConfig:
    code_dim: int = 21          #: GEOM_DIM after the q=3 middle-bucket fix
    feat_dim: int = 128         #: tower channel count
    dim: int = 256              #: internal attention width
    n_proto: int = 16           #: prototype tokens
    n_heads: int = 4
    hyper_rank: int = 64        #: bottleneck of the hypernetwork
    min_conf: float = 0.0       #: output is scaled to 0 at zero code density
    residual_scale: float = 1.0

    @staticmethod
    def full(**kw) -> "PCHConfig":
        """~3.9M params, matching the proposal's Full budget.

        The hypernetwork dominates (dim x n_proto x 2 outputs), which is the
        honest place for the capacity to sit: it is what turns a 21-bit code
        into a conditioned prototype bank, and it is the part being tested.
        """
        return PCHConfig(dim=512, n_proto=32, n_heads=8, hyper_rank=96, **kw)

    @staticmethod
    def lite(**kw) -> "PCHConfig":
        """Same structure, narrower -- so Full-vs-Lite is a capacity ablation
        and not a change of mechanism."""
        return PCHConfig(dim=192, n_proto=12, n_heads=4, hyper_rank=80, **kw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PCH(nn.Module):
    """``(features, code) -> residual`` of the same shape as ``features``."""

    def __init__(self, cfg: PCHConfig | None = None):
        super().__init__()
        self.cfg = cfg or PCHConfig.full()
        c = self.cfg
        # prototype bank: what geometry "kinds" look like, before conditioning
        self.protos = nn.Parameter(torch.randn(c.n_proto, c.dim) * c.dim ** -0.5)
        # hypernetwork: code -> per-prototype (scale, shift).  Low-rank on
        # purpose -- a dense 21 -> n_proto*dim*2 map is ~2M params by itself and
        # would dominate the budget without adding expressiveness over the
        # bottleneck.
        self.hyper = nn.Sequential(
            nn.Linear(c.code_dim, c.hyper_rank),
            nn.GELU(),
            nn.Linear(c.hyper_rank, c.n_proto * c.dim * 2),
        )
        nn.init.zeros_(self.hyper[-1].weight)
        nn.init.zeros_(self.hyper[-1].bias)          # step 0: scale=1, shift=0

        self.q = nn.Linear(c.feat_dim, c.dim)
        self.k = nn.Linear(c.dim, c.dim)
        self.v = nn.Linear(c.dim, c.dim)
        self.proj = nn.Linear(c.dim, c.feat_dim)
        # zero-init the output projection: the module is an exact no-op at step
        # 0, so adding it to a *resumed* checkpoint cannot damage it before it
        # has learned anything.  Every arm here resumes from P3'.
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        self.norm_f = nn.LayerNorm(c.feat_dim)
        self.norm_p = nn.LayerNorm(c.dim)

    # -- confidence ---------------------------------------------------------
    def confidence(self, code: torch.Tensor) -> torch.Tensor:
        """Code density -> [0, 1].  Zero code => exactly zero contribution.

        The parsed code is multi-hot over 21 slots and a typical real phrase
        lights 3-6 of them, so density is a direct proxy for "did the sentence
        actually say anything about geometry".  An all-zero code means the
        `<where>` span named none -- roughly 21% of samples for shape -- and the
        honest response there is to inject nothing.
        """
        d = code.reshape(-1).abs().sum()
        return torch.clamp(d / 6.0, max=1.0).clamp_min(self.cfg.min_conf)

    # -- forward ------------------------------------------------------------
    def forward(self, feat: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        """``feat`` ``(1, C, H, W)``; ``code`` ``(code_dim,)``.

        Returns a residual to ADD to ``feat`` -- the caller owns the addition so
        the injection point stays visible at the call site rather than hidden
        inside this module.
        """
        c = self.cfg
        if code is None:
            return torch.zeros_like(feat)
        conf = self.confidence(code)
        b, ch, h, w = feat.shape

        # hypernetwork modulation of the prototype bank
        ss = self.hyper(code.reshape(1, -1).float())
        scale, shift = ss.reshape(2, c.n_proto, c.dim)
        protos = self.protos * (1.0 + scale) + shift            # (P, dim)
        protos = self.norm_p(protos).unsqueeze(0)               # (1, P, dim)

        # dense features attend over the modulated prototypes
        x = self.norm_f(feat.reshape(b, ch, h * w).transpose(1, 2))   # (1, HW, C)
        q = self.q(x)
        k = self.k(protos)
        v = self.v(protos)
        nh, hd = c.n_heads, c.dim // c.n_heads
        q = q.reshape(b, h * w, nh, hd).transpose(1, 2)
        k = k.reshape(b, -1, nh, hd).transpose(1, 2)
        v = v.reshape(b, -1, nh, hd).transpose(1, 2)
        att = F.scaled_dot_product_attention(q, k, v)                 # (1,nh,HW,hd)
        att = att.transpose(1, 2).reshape(b, h * w, c.dim)
        out = self.proj(att).transpose(1, 2).reshape(b, ch, h, w)
        return out * (conf * c.residual_scale)

    # -- reporting ----------------------------------------------------------
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def facts(self) -> dict[str, Any]:
        return {"pch": self.cfg.to_dict(), "n_params": self.n_params(),
                "zero_init_output": True,
                "fallback": "output scaled by code density; exactly 0 at 0 density"}
