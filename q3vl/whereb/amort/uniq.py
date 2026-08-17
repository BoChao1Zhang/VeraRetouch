"""UNIQ -- the unified query head (EPR-011).

One head for both mask populations, in the mask-classification shape the
segmentation literature converged on (MaskFormer arXiv:2107.06278 "mask
classification is sufficiently general ... using the exact same model, loss,
and training procedure"; Mask2Former 2112.01527; task-token conditioning
OneFormer 2211.06220; LMM-side PSALM 2403.14598):

    K learnable queries --cross-attend--> <where> token hiddens (1, T, 2560)
    each query emits:  (a) a mask logit field   <w_q , pixel-embedding(x)>
                       (b) a family class       {linear, radial, band, semantic}
                       (c) a selection score    (deployable argmax; no GT)
    training = winner-takes-all over the K per-query field losses
    inference = the argmax-selection query's field

Why the queries read the TOKEN SEQUENCE and not the pooled vector: H25 measured
pooling the dense channel away at -0.086 IoU, and S13 measured teacher-forced
perfect reasoning buying +0.0007..0.0087 through the pooled/FiLM path.  The
information provably survives in the sequence (P0 probe, GT leg paired
Delta +0.1111, p=4.8e-7) and dies in the pooling.  NOTE the existing pooled
FiLM conditioning of the tower is kept unchanged -- the queries are additive,
so the comparison against the P3'@1200 anchor isolates the head redesign.

Fourier pixel basis (arm A only; EPR-011 pre-registered violation of DELTA
S5.6)
------------------------------------------------------------------------------
DELTA S5.6 forbids coordinate channels "to stop the head learning the centre
prior itself" -- written for the W01-era heads that E3 caught with
corr(output, prior) 0.64 > corr(output, GT) 0.47.  The analytic families are
low-dimensional functions of image coordinates, and a dot-product mask head
can express them exactly when the pixel embedding carries a Fourier coordinate
basis (Tancik et al., NeurIPS 2020, arXiv:2006.10739; same mechanism as SAM's
``PositionEmbeddingRandom``).  Arm A therefore carries the basis, arm B does
not, and the E3/M3 falsification column (``corr_center_minus_corr_gt``,
computed on every board) is the pre-registered execution line for arm A: if
the coordinate basis resurrects the centre-prior disease, S5.6 stands and the
basis dies.  ``fourier_scale`` is the calibration knob Tancik's bandwidth
argument says must exist.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from q3vl.where.config import S_SCALE

from .heads import ConvTower

__all__ = ["UniQHead", "UNIQ_FAMILIES", "fourier_grid", "SelMLP"]

#: class-head vocabulary; order is the label encoding and is frozen with the arm
UNIQ_FAMILIES = ("linear", "radial", "band", "semantic")


def fourier_grid(gh: int, gw: int, bands: int, scale: float, seed: int,
                 device=None, dtype=torch.float32) -> torch.Tensor:
    """``(2 + 2*bands, gh, gw)`` coordinate basis: raw (x, y) + random Fourier.

    Coordinates live in [-1, 1] (aspect ignored: the grid is already the
    image's own aspect).  Frequencies are drawn ONCE from a seeded Gaussian --
    the SAM ``PositionEmbeddingRandom`` form -- so the basis is a fixed,
    deterministic function of (bands, scale, seed) and never trains; what
    trains is only each query's weight over it.  Raw (x, y) is included
    because the `linear` family is literally a linear function of coordinates.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    freqs = torch.randn(2, bands, generator=g) * scale          # (2, B)
    ys = torch.linspace(-1.0, 1.0, gh)
    xs = torch.linspace(-1.0, 1.0, gw)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")              # (gh, gw)
    p = torch.stack([xx, yy], dim=-1)                           # (gh, gw, 2)
    proj = 2.0 * math.pi * (p @ freqs)                          # (gh, gw, B)
    feats = [xx.unsqueeze(0), yy.unsqueeze(0),
             proj.sin().permute(2, 0, 1), proj.cos().permute(2, 0, 1)]
    out = torch.cat(feats, dim=0)
    return out.to(device=device, dtype=dtype)


class SelMLP(nn.Module):
    """SAM's ``MLP`` (``segment_anything/modeling/mask_decoder.py`` L154-176),
    copied form for form: ``num_layers`` Linears, ReLU between them, optional
    sigmoid on the output.  EPR-012 uses it as the IoU-regression selection
    head, with ``sigmoid_output=True`` (SAM2 training yaml
    ``iou_prediction_use_sigmoid: True``).
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_layers: int = 3, sigmoid_output: bool = False):
        super().__init__()
        self.num_layers = int(num_layers)
        h = [hidden_dim] * (self.num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([in_dim] + h, h + [out_dim]))
        self.sigmoid_output = bool(sigmoid_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return torch.sigmoid(x) if self.sigmoid_output else x


class UniQHead(nn.Module):
    """Tower -> K query fields + family class + selection score.

    Zero-init discipline (same as every head in this package): ``to_mask`` is
    zero-initialised, so every query's field is exactly 0 -> mask 0.5 at step 0,
    and the cross-attention output projection is zero-initialised, so queries
    start at their learnable base rather than at a random read of the text.
    """

    def __init__(self, ch: int = 128, in_dim: int = 1024, extra_ch: int = 1,
                 n_blocks: int = 6, cond_dim: int | None = 288,
                 text_dim: int = 2560, n_queries: int = 4,
                 fourier_bands: int = 0, fourier_scale: float = 1.0,
                 gain: float = 2.0, seed: int = 0,
                 iou_head: bool = False, iou_head_depth: int = 3,
                 iou_head_hidden: int | None = None,
                 sel_stability: float = 0.0,
                 sel_stability_delta: float = 0.05):
        super().__init__()
        self.tower = ConvTower(in_dim, extra_ch, ch, n_blocks, cond_dim)
        self.n_queries = int(n_queries)
        self.fourier_bands = int(fourier_bands)
        self.fourier_scale = float(fourier_scale)
        self.fourier_seed = int(seed) + 77
        self.pix_extra = (2 + 2 * self.fourier_bands) if self.fourier_bands else 0

        g = torch.Generator(device="cpu").manual_seed(seed)
        self.queries = nn.Parameter(
            torch.randn(self.n_queries, ch, generator=g) * 0.02)
        self.text_proj = nn.Sequential(nn.LayerNorm(text_dim),
                                       nn.Linear(text_dim, ch))
        self.xattn = nn.MultiheadAttention(ch, num_heads=4, batch_first=True)
        nn.init.zeros_(self.xattn.out_proj.weight)
        nn.init.zeros_(self.xattn.out_proj.bias)
        self.q_norm = nn.LayerNorm(ch)
        self.ffn = nn.Sequential(nn.Linear(ch, 2 * ch), nn.GELU(),
                                 nn.Linear(2 * ch, ch))
        nn.init.zeros_(self.ffn[-1].weight)
        nn.init.zeros_(self.ffn[-1].bias)

        pix_dim = ch + self.pix_extra
        self.to_mask = nn.Linear(ch, pix_dim + 1)   # per-query weight + bias
        nn.init.zeros_(self.to_mask.weight)
        nn.init.zeros_(self.to_mask.bias)
        self.cls = nn.Linear(ch, len(UNIQ_FAMILIES))
        #: EPR-012.  Default = the pre-registered ``Linear(ch, 1)`` logit head
        #: distilled from the winner index.  ``iou_head=True`` swaps in SAM's
        #: 3-layer MLP + sigmoid quality head; the output KEY stays
        #: ``sel_logits`` and sigmoid is monotone, so the argmax selection
        #: protocol at inference is untouched.
        self.iou_head = bool(iou_head)
        if self.iou_head:
            self.sel = SelMLP(ch, int(iou_head_hidden or ch), 1,
                              int(iou_head_depth), sigmoid_output=True)
        else:
            self.sel = nn.Linear(ch, 1)
        #: EPR-012 ablation: SAM2's inference-only stability fallback
        #: (`mask_decoder.py` L28-29/L247-295).  0.0 = off = plain argmax.
        self.sel_stability = float(sel_stability)
        self.sel_stability_delta = float(sel_stability_delta)
        self.gain = nn.Parameter(torch.tensor(float(gain)))
        self._fourier_cache: dict[tuple[int, int], torch.Tensor] = {}

    # -- pieces --------------------------------------------------------------
    def _fourier(self, gh: int, gw: int, device, dtype) -> torch.Tensor:
        key = (gh, gw)
        f = self._fourier_cache.get(key)
        if f is None or f.device != device:
            f = fourier_grid(gh, gw, self.fourier_bands, self.fourier_scale,
                             self.fourier_seed, device=device, dtype=dtype)
            self._fourier_cache[key] = f
        return f.to(dtype)

    def _query_states(self, h_where: torch.Tensor,
                      h_mask: torch.Tensor | None) -> torch.Tensor:
        kv = self.text_proj(h_where.float())                    # (1, T, ch)
        pad = None
        if h_mask is not None:
            pad = h_mask.reshape(1, -1) < 0.5                   # True = ignore
            if bool(pad.all()):
                pad = None                                      # degenerate mask
        q = self.queries.unsqueeze(0)                           # (1, K, ch)
        att, _ = self.xattn(q, kv, kv, key_padding_mask=pad, need_weights=False)
        q = q + att
        q = q + self.ffn(self.q_norm(q))
        return q[0]                                             # (K, ch)

    # -- forward -------------------------------------------------------------
    def forward(self, feat: torch.Tensor, extra: torch.Tensor | None,
                cond: torch.Tensor | None, h_where: torch.Tensor,
                h_mask: torch.Tensor | None) -> dict[str, Any]:
        codes = self.tower(feat, extra, cond)                   # (1, ch, gh, gw)
        gh, gw = codes.shape[-2:]
        pix = codes[0].reshape(codes.shape[1], gh * gw)         # (ch, N)
        if self.fourier_bands:
            four = self._fourier(gh, gw, codes.device, pix.dtype)
            pix = torch.cat([pix, four.reshape(four.shape[0], gh * gw)], dim=0)

        q = self._query_states(h_where, h_mask)                 # (K, ch)
        wb = self.to_mask(q)                                    # (K, pix+1)
        raw = wb[:, :-1] @ pix + wb[:, -1:]                     # (K, N)
        s_all = S_SCALE * torch.tanh(raw / S_SCALE)
        s_all = s_all.reshape(self.n_queries, gh, gw)
        return {"s_all": s_all,
                "cls_logits": self.cls(q),                      # (K, F)
                "sel_logits": self.sel(q).reshape(-1)}          # (K,)

    def mask_of(self, s_field: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gain * s_field)

    # -- inference-time selection -------------------------------------------
    def select_index(self, u: dict[str, Any]) -> int:
        """The deployable selection rule: argmax over the selection head.

        With ``sel_stability > 0`` (EPR-012 ablation, inference only, SAM2
        `mask_decoder.py` L149-150 gate) a top pick whose field is not stable
        under a +-delta logit perturbation is handed to the next-highest
        scoring query instead.  ``sel_stability == 0`` is a plain argmax --
        the pre-registered protocol, bit for bit.
        """
        sel = u["sel_logits"].detach().reshape(-1)
        idx = int(sel.argmax())
        if self.sel_stability <= 0.0 or self.training:
            return idx
        s = u["s_all"].detach()
        logits = self.gain.detach() * s
        d = self.sel_stability_delta
        hi = (logits > d).reshape(s.shape[0], -1).sum(-1).float()
        lo = (logits > -d).reshape(s.shape[0], -1).sum(-1).float()
        stab = hi / lo.clamp_min(1.0)
        if float(stab[idx]) >= self.sel_stability:
            return idx
        for cand in sel.argsort(descending=True).tolist():
            if int(cand) != idx:
                return int(cand)
        return idx

    def facts(self) -> dict[str, Any]:
        return {"n_queries": self.n_queries, "families": list(UNIQ_FAMILIES),
                "fourier_bands": self.fourier_bands,
                "fourier_scale": self.fourier_scale,
                "fourier_seed": self.fourier_seed,
                "pix_extra_channels": self.pix_extra,
                "sel_head": "mlp_sigmoid_iou" if self.iou_head else "linear",
                "sel_stability": self.sel_stability}
