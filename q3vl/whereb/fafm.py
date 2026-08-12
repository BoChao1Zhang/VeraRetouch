"""FAFM -- Field-Anchored Flow Matching (proposal B), model and sampler.

``RESEARCH_unified-field-prediction_2026-08-10`` section 3.  One learnable network, one
scalar loss, one generated variable: the coarse field ``c`` that lives at the
neck of the frozen decoder, immediately before the image-guided upsample ``A_I``.

    train:  c0 ~ N(0,I);  c_t = (1-t) c0 + t c*
            [c_t || S || V] -> conv stem -> DiT blocks -> c_hat
                                   ^ AdaLN(t)   ^ cross-attn(K,V = E_T)
            loss = (c_hat - c*)^T Lambda (c_hat - c*)

    infer:  N=8 Euler steps of dc/dt = (c_hat - c_t)/(1-t), K samples, CFG,
            then mode-seeking selection; render clip(A_I c, 0, 1).

Three things here are load-bearing and easy to get wrong, so they are stated:

* **x0-parameterisation.** The network predicts ``c*`` directly, not the noise.
  Lotus's ablation found noise-prediction actively harmful for dense fields.
* **Never average the K samples.** The whole point is mode-seeking; the mean of
  a multi-modal conditional is the union-shaped over-coverage this campaign has
  collapsed into twice.  :func:`select_mode` only ever returns one drawn sample.
* **Clip the field, never the coordinate.** ``c*`` is not confined to [0,1] --
  measured arm-wide domain [-48.96, +69.59] with 27.4% of cells below 0 --
  so clamping ``c`` silently destroys the ridge solution (s-cache contract,
  failure mode 2: the orphan check stays silent while the axis is gone).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["FAFMConfig", "FAFMNet", "lambda_metric_loss", "sample_fafm",
           "select_mode", "sigma_max_sq", "SIGMA_MAX_SQ_FULLRES"]

#: ``||A_I||_2^2`` at the **full** delivery resolution: the median largest
#: eigenvalue of ``A_I^T A_I`` over 1000 S-train samples in uni_gate0 case B.
#: Kept for provenance only -- **never consume it directly**, see below.
SIGMA_MAX_SQ_FULLRES = 256.0


def sigma_max_sq(n_hi: int, n_low: int) -> float:
    """``||A_I||_2^2`` for an operator that maps ``n_low`` cells to ``n_hi`` pixels.

    **This is resolution-bound, and that is the whole point of the function.**
    The guided upsample preserves constants, so the constant mode's gain is
    exactly the number of output pixels per coarse cell::

        ||A 1||^2 / ||1||^2 = n_hi / n_low          (an identity, any guide)

    For a **smooth** guide -- which every real image is -- the constant mode is
    the top eigenvector, so this equals ``sigma_max^2``.  Measured against an
    explicit Gram eigendecomposition: on 8 real V_where images the ratio of
    measured ``lambda_max`` to this closed form is 1.00014 / 1.00014 / 1.00012 at
    div 1 / 2 / 4.

    Honest limit: for a pathological high-frequency guide (e.g. i.i.d. noise) the
    constant mode is no longer dominant and this becomes a **lower** bound --
    measured 27.6 vs 16.0 on a uniform-noise guide.  That regime does not occur
    here (the guide is Rec.709 luma of a photograph), and the consequence of the
    bound being loose would be a metric term weighted slightly *high*, not the
    16x *underweight* this function exists to prevent.

    Why it is a function and not the constant it used to be
    -------------------------------------------------------
    ``SIGMA_MAX_SQ_FULLRES`` (256.0) was measured at full resolution but consumed
    against the **quarter-resolution** operator the trainer actually applies
    (``GUIDE_DIV = 4``), where the true value is 16.  That divided the metric term
    by 16x too much: the ``(1-lambda) A^T A / ||A||^2`` component of ``Lambda`` --
    the "task-aligned metric" that section 3.1 makes the case's headline, and the
    literal execution of constraint 3 -- carried about **6%** of its intended
    weight, i.e. the probe's core mechanism was very nearly switched off.
    Deriving the value from the operator's own shape makes that class of mistake
    unrepresentable.  (REVIEW-impl-amort-uni U5.)
    """
    if n_low <= 0:
        raise ValueError("n_low must be positive")
    return float(n_hi) / float(n_low)


@dataclass
class FAFMConfig:
    d_model: int = 384
    depth: int = 8
    heads: int = 6
    patch: int = 2
    in_sim: int = 4          # similarity-field channels S
    in_vis: int = 64         # visual channels V (B @ F_pre)
    d_text: int = 2560       # frozen text embedding width
    lambda_mix: float = 0.5  # Lambda = lam I + (1-lam) A^T A / ||A||^2
    p_drop_text: float = 0.1
    p_drop_sim: float = 0.1


def sincos_2d(d: int, h: int, w: int, device, dtype) -> torch.Tensor:
    """``(h*w, d)`` 2-D sin-cos position embedding, built for whatever grid arrives.

    The stage keeps true aspect ratio, so grids differ per image (29 distinct
    shapes in the 20k training subset).  A learned position table would have to
    be interpolated per shape; sin-cos is exact at every shape and costs nothing.
    """
    def axis(n: int, dd: int) -> torch.Tensor:
        pos = torch.arange(n, device=device, dtype=torch.float32)
        omega = torch.arange(dd // 2, device=device, dtype=torch.float32)
        omega = 1.0 / (10000 ** (omega / (dd / 2)))
        out = pos[:, None] * omega[None, :]
        return torch.cat([out.sin(), out.cos()], dim=1)

    dh = d // 2
    eh = axis(h, dh)[:, None, :].expand(h, w, dh)
    ew = axis(w, d - dh)[None, :, :].expand(h, w, d - dh)
    return torch.cat([eh, ew], dim=-1).reshape(h * w, d).to(dtype)


def timestep_embedding(t: torch.Tensor, d: int) -> torch.Tensor:
    half = d // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device,
                                                      dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None] * 1000.0
    return torch.cat([a.cos(), a.sin()], dim=-1).to(t.dtype)


class Block(nn.Module):
    """DiT block: AdaLN-modulated self-attn + cross-attn to the instruction + MLP.

    Cross-attention (not concatenation, not a hypernetwork) is how the sequence
    condition enters: the systematic comparison in "Attention Beats
    Concatenation" puts attention ahead of both, and the hypernetwork form
    ("condition -> parameters") is precisely the ``x -> w`` shape this campaign
    has already judged dead.
    """

    def __init__(self, d: int, heads: int, d_text: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 9 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, temb, txt, txt_mask):
        s1, b1, g1, s2, b2, g2, s3, b3, g3 = self.ada(temb).chunk(9, dim=-1)
        h = self.n1(x) * (1 + s1[:, None]) + b1[:, None]
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + g1[:, None] * h
        h = self.n2(x) * (1 + s2[:, None]) + b2[:, None]
        h, _ = self.xattn(h, txt, txt, key_padding_mask=txt_mask, need_weights=False)
        x = x + g2[:, None] * h
        h = self.n3(x) * (1 + s3[:, None]) + b3[:, None]
        x = x + g3[:, None] * self.mlp(h)
        return x


class FAFMNet(nn.Module):
    """``(c_t, S, V, E_T, t) -> c_hat``.  ~30M parameters at the default config."""

    def __init__(self, cfg: FAFMConfig | None = None):
        super().__init__()
        self.cfg = cfg = cfg or FAFMConfig()
        d, p = cfg.d_model, cfg.patch
        cin = 1 + cfg.in_sim + cfg.in_vis
        self.stem = nn.Sequential(nn.Conv2d(cin, d, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(d, d, p, stride=p))
        self.tproj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        # One shared 2560 -> d text projection instead of a 2560-wide K/V in every
        # block: per-block projections put the model at 65.8M, four times the
        # section 3.7 budget ("~30M, DiT-S"), and almost all of the excess was
        # twelve copies of the same linear map.
        self.text_in = nn.Linear(cfg.d_text, d)
        self.blocks = nn.ModuleList([Block(d, cfg.heads, d)
                                     for _ in range(cfg.depth)])
        self.nout = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.adaout = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.head = nn.Linear(d, p * p)
        # zero-init the output head: the network starts by predicting nothing and
        # has to earn every deviation.  (Red-line adjacent: the global affine G
        # in the renderer is initialised to 0 rather than I for the same reason.)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.adaout[1].weight)
        nn.init.zeros_(self.adaout[1].bias)
        #: null text token for classifier-free guidance / the unconditional branch
        self.null_text = nn.Parameter(torch.zeros(1, 1, cfg.d_text))

    def forward(self, c_t, sim, vis, t, txt, txt_mask):
        """``c_t (B,1,h,w)``, ``sim (B,S,h,w)``, ``vis (B,V,h,w)``, ``t (B,)``,
        ``txt (B,L,d_text)``, ``txt_mask (B,L)`` True where padded."""
        B, _, h, w = c_t.shape
        p = self.cfg.patch
        x = self.stem(torch.cat([c_t, sim, vis], dim=1))         # (B,d,h/p,w/p)
        hh, ww = x.shape[-2], x.shape[-1]
        x = x.flatten(2).transpose(1, 2)                          # (B,N,d)
        x = x + sincos_2d(self.cfg.d_model, hh, ww, x.device, x.dtype)[None]
        temb = self.tproj(timestep_embedding(t, self.cfg.d_model))
        txt = self.text_in(txt)
        for blk in self.blocks:
            x = blk(x, temb, txt, txt_mask)
        s, b = self.adaout(temb).chunk(2, dim=-1)
        x = self.nout(x) * (1 + s[:, None]) + b[:, None]
        x = self.head(x)                                          # (B,N,p*p)
        x = x.transpose(1, 2).reshape(B, p * p, hh, ww)
        return F.pixel_shuffle(x, p)                              # (B,1,h,w)


def lambda_metric_loss(c_hat: torch.Tensor, c_star: torch.Tensor,
                       apply_A, lambda_mix: float,
                       sigma_max_sq_value: float) -> torch.Tensor:
    """``(c_hat - c*)^T Lambda (c_hat - c*)`` with
    ``Lambda = lam I + (1-lam) A^T A / ||A||^2``.

    The second term is evaluated as ``||A_I r||^2`` -- **applying** the frozen
    operator to the residual, never materialising ``A^T A`` (9.4 MB/sample, i.e.
    188 GB for the 20k subset).  ``A_I`` is linear (uni_gate0 G0a: superposition
    residual 4.7e-16), which is exactly what makes this substitution exact rather
    than approximate, and it is why the clamp must stay off inside ``apply_A``.

    This is not a second loss term: ``A_I`` is frozen and linear, so ``Lambda``
    is a fixed quadratic metric on one residual -- the "task-aligned metric" of
    section 3.1, and the literal execution of constraint 3 (supervision measured
    through the frozen decoder, in field space).
    """
    r = c_hat - c_star
    flat = r.flatten(1)
    term_c = (flat * flat).sum(dim=1)
    if sigma_max_sq_value <= 0:
        raise ValueError("sigma_max_sq_value must be positive")
    term_f = (apply_A(r) ** 2).flatten(1).sum(dim=1) / sigma_max_sq_value
    return (lambda_mix * term_c + (1.0 - lambda_mix) * term_f).mean()


@torch.no_grad()
def sample_fafm(net: FAFMNet, sim, vis, txt, txt_mask, *, k: int, steps: int,
                cfg_scale: float, generator: torch.Generator | None = None,
                use_text: bool = True) -> torch.Tensor:
    """``(K, B, 1, h, w)`` -- K independent ODE trajectories, no averaging.

    Euler integration of ``dc/dt = (c_hat - c_t)/(1-t)`` from ``t=0`` to ``t=1``,
    which is the rectified-flow velocity implied by x0-parameterisation.
    """
    B, h, w = sim.shape[0], sim.shape[-2], sim.shape[-1]
    dev, dt = sim.device, sim.dtype
    # All K trajectories run as one batch of K*B.  Looping K times in Python was
    # the eval bottleneck: these are small tensors, so wall time is dominated by
    # kernel launches, not arithmetic, and the loop multiplied the launch count
    # by K for no reason.  Layout is (K, B) flattened -- repeat_interleave on the
    # conditions keeps every trajectory paired with its own sample.
    simK = sim.repeat(k, 1, 1, 1)
    visK = vis.repeat(k, 1, 1, 1)
    txtK = txt.repeat(k, 1, 1)
    maskK = txt_mask.repeat(k, 1)
    null = net.null_text.expand(k * B, 1, -1).to(dt)
    null_mask = torch.zeros(k * B, 1, dtype=torch.bool, device=dev)
    c = torch.randn(k * B, 1, h, w, device=dev, dtype=dt, generator=generator)
    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps
        tt = torch.full((k * B,), t0, device=dev, dtype=dt)
        if use_text and cfg_scale != 1.0:
            ch_c = net(c, simK, visK, tt, txtK, maskK)
            ch_u = net(c, simK, visK, tt, null, null_mask)
            c_hat = ch_u + cfg_scale * (ch_c - ch_u)
        elif use_text:
            c_hat = net(c, simK, visK, tt, txtK, maskK)
        else:
            c_hat = net(c, simK, visK, tt, null, null_mask)
        # dc = (c_hat - c)/(1-t) * dt ; at t0=0 this is just c_hat - c
        c = c + (c_hat - c) * ((t1 - t0) / max(1.0 - t0, 1e-6))
    return c.reshape(k, B, 1, h, w)


def select_mode(fields: torch.Tensor, tau: float = 0.6) -> tuple[torch.Tensor, float]:
    """R1: agglomerate the K rendered fields by soft-IoU, return the largest
    cluster's medoid and the ambiguity signal ``1 - |largest|/K``.

    **Mode-seeking, never averaging.**  Taking the mean of the K samples would
    re-introduce exactly the mean collapse the generative formulation exists to
    avoid, and the campaign has two prior collapses with that signature.

    ``fields`` is ``(K, n)`` of rendered, clipped fields for ONE sample.
    """
    k = fields.shape[0]
    x = fields.reshape(k, -1)
    inter = torch.minimum(x[:, None], x[None, :]).sum(-1)
    union = torch.maximum(x[:, None], x[None, :]).sum(-1).clamp_min(1e-9)
    sim = inter / union
    best_members, best_idx = None, 0
    for i in range(k):
        members = (sim[i] >= tau).nonzero(as_tuple=True)[0]
        if best_members is None or members.numel() > best_members.numel():
            best_members, best_idx = members, i
    if best_members is None or best_members.numel() == 0:
        return fields[0], 1.0 - 1.0 / k
    sub = sim[best_members][:, best_members]
    medoid = best_members[sub.sum(dim=1).argmax()]
    return fields[medoid], 1.0 - best_members.numel() / k
