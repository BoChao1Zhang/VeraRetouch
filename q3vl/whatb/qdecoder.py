"""EPR-030: the ``N+1``-query transformer decoder that replaces the MLP generator.

What this module is
-------------------
``CGLUTGenerator`` (``q3vl/whatb/generator.py``) is "one vector in, ``22N+12``
numbers out": ``z`` -> ``LayerNorm+Linear`` -> a 3-layer shared MLP encoder ->
five parameter-specific heads.  :class:`GlutQueryDecoder` replaces the whole of
that with **one query per Gaussian plus one global-affine query**, decoded by
``L`` layers of (cross-attention -> FFN) against a memory built from the same
``z``.  Downstream is unchanged: the forward returns the same
:class:`~q3vl.whatb.glut.GlutParams` the carrier already consumes, so the GLUT
Eq.1-5 carrier, the criteria and the board do not know which backbone produced
the parameters.

Where each choice comes from (each one opened at the source, 2026-08-15/16)
--------------------------------------------------------------------------
================================  ==========================================
choice                            source
================================  ==========================================
``N`` Gaussian queries + 1        EPR-029 §3.2 (one Gaussian = one query, plus
global-affine query                one extra query for ``G, g``)
``q_i = emb[i] + PE_R + PE_G      StatLUT (arXiv:2607.08227) §3.2 Eq.2:
+ PE_B``                          learnable 3D positional encodings added to
                                  the query, ``(+)`` = broadcast addition; the
                                  grid index is that of
                                  ``glut.uniform_grid_positions(N)``
pre-norm (cross-attn -> FFN),     StatLUT appendix: 6 layers, d_model 512,
``d=512, L=6, H=8``, FFN 4d       8 heads.  Self-attention is **off** by
                                  default and is an ablation row.
GELU                              Hendrycks & Gimpel arXiv:1606.08415;
                                  the ReLU that died in EPR-025 is the
                                  ``Dying ReLU`` Thm 3.4 object
                                  (arXiv:1903.06733v3)
memory = ``LayerNorm(2560) +      this project's condition is one vector;
Linear(2560 -> d)``, M rows       ``M`` is the ablation knob (EPR-030 §4)
head ``Linear(d -> 22)`` shared   EPR-029 §3.2
+ ``Linear(d -> 12)`` global
weight zeros, **bias = the        Bias-HyperInit (Beck et al.,
target init**                     https://proceedings.mlr.press/v205/beck23a/beck23a.pdf);
                                  Text-to-LoRA ``src/hyper_llm_modulator/
                                  hyper_modulator.py``: ``nn.init.zeros_(
                                  layer.weight)`` + ``head.bias.copy_(init_bias)``;
                                  Splatter Image ``scene/gaussian_predictor.py``
                                  ``get_splits_and_inits`` (per-group gain/bias)
================================  ==========================================

The 22 output dimensions, in order, and the bias each one is initialised to::

    [0:3]    d_mu             0.0                 mu = grid + d_mu
    [3:6]    chol_diag        softplus^-1(0.15)   = -1.8212...
    [6:9]    chol_off         0.0
    [9]      opacity_logit    +4.0
    [10:19]  d_M              0.0                 M = I + d_M
    [19:22]  b                0.0

and the global query's 12::

    [0:9]    d_G              0.0                 G = d_G (see below)
    [9:12]   g                0.0

so at step 0 the decoder emits **exactly** the parameter set
``SharedGeometry`` + ``GlutParams.identity`` hold, and ``f`` is the identity
map.  :meth:`GlutQueryDecoder.assert_step0_identity` checks both halves of that
sentence with a real forward and raises (never warns) on failure.

``G`` is anchored at **0**, not at ``I``
----------------------------------------
The task card's table spells the global slice ``G = I + dG``.  That cannot hold
at the same time as "step 0 is the identity": with ``G = I`` the carrier
computes ``f(x) = sum_i w_i (I x + 0) + clamp(I x + 0) = 2x`` (EPR-029 §3.2's
NOVEL note, and ``GlutParams.identity`` -- which spells ``g_matrix = 0`` --
agrees).  The conservative default is therefore ``G = dG`` with ``dG`` bias 0,
and ``g_residual=True`` is kept as the other reading of the card.  Recorded in
``NOTES.md`` item 1 for the user to rule on; nothing was silently decided.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
from torch import Tensor, nn

from q3vl.whatb.glut import (
    EPS,
    GlutParams,
    grid_axis_sizes,
    softplus_inverse,
    uniform_grid_positions,
)

__all__ = [
    "QDEC_ACTIVATIONS",
    "QDEC_HEAD_INITS",
    "GAUSS_SLICES",
    "GLOBAL_SLICES",
    "QueryDecoderLayer",
    "GlutQueryDecoder",
    "qdecoder_param_count",
]

#: ``--qdec-act``.  ``gelu`` is the default (arXiv:1606.08415); ``relu`` is the
#: ablation row that reproduces the activation family EPR-025 watched die.
QDEC_ACTIVATIONS: tuple[str, ...] = ("gelu", "relu")

#: ``--head-init``.  ``bias`` = Bias-HyperInit (weight 0, bias = target init);
#: ``zero`` = weight 0 **and** bias 0 (the EPR-029 spelling), kept as the row.
QDEC_HEAD_INITS: tuple[str, ...] = ("bias", "zero")

#: name -> (start, stop) inside the 22-dim Gaussian head output.
GAUSS_SLICES: dict[str, tuple[int, int]] = {
    "d_mu": (0, 3),
    "chol_diag": (3, 6),
    "chol_off": (6, 9),
    "opacity_logit": (9, 10),
    "d_m": (10, 19),
    "b": (19, 22),
}

#: name -> (start, stop) inside the 12-dim global head output.
GLOBAL_SLICES: dict[str, tuple[int, int]] = {"d_g_matrix": (0, 9), "g_bias": (9, 12)}

#: one float32 ULP at 1.0 (2^-24).  The floor of the "the head bias really is the
#: SharedGeometry init" comparison: the reference recomputes
#: ``uniform_grid_positions`` on the *current* device and CPU / CUDA disagree by up
#: to this much on ``(i + 0.5) / k`` (measured on the first GPU smoke of EPR-030:
#: 5.960e-08 on the 3-cell B axis).  Every other field is compared against an exact
#: constant and lands at 0.
_F32_ULP: float = 2.0 ** -24


def _activation(name: str) -> nn.Module:
    if name not in QDEC_ACTIVATIONS:
        raise ValueError(f"activation must be one of {QDEC_ACTIVATIONS}, got {name!r}")
    return nn.GELU() if name == "gelu" else nn.ReLU()


class QueryDecoderLayer(nn.Module):
    """pre-norm ``cross-attention -> [self-attention] -> FFN``, all residual.

    Pre-norm rather than the post-norm of official DETR / Mask2Former
    (``--pre_norm`` off, ``PRE_NORM: False``): with post-norm
    ``LayerNorm(q + 0) != q``, so a zero-weight output head would not leave the
    query untouched and the step-0 identity witness would be a lie.  Same
    reasoning the where-side ``RefineLayer``
    (``q3vl/whereb/amort/uniq4.py:164-214``) is written under.

    Self-attention is optional and **off** by default: with one condition vector
    the queries have nothing to coordinate through the memory, so whether they
    should talk to each other is exactly the ablation row.

    The FFN is kept as three named children (``fc1`` / ``act`` / ``fc2``) rather
    than an ``nn.Sequential`` so :meth:`GlutQueryDecoder.capture_activations`
    can read the **pre-activation** and report its near-zero rate -- the number
    nobody had when ``head_color``'s ReLU went from 0.469 to 1.0000 dead.
    """

    def __init__(self, dim: int, heads: int, *, ffn_mult: int = 4,
                 activation: str = "gelu", self_attn: bool = False) -> None:
        super().__init__()
        self.dim, self.heads, self.ffn_mult = int(dim), int(heads), int(ffn_mult)
        self.has_self_attn = bool(self_attn)
        self.norm_cross = nn.LayerNorm(self.dim)
        self.cross_attn = nn.MultiheadAttention(self.dim, self.heads, batch_first=True)
        if self.has_self_attn:
            self.norm_self = nn.LayerNorm(self.dim)
            self.self_attn = nn.MultiheadAttention(self.dim, self.heads, batch_first=True)
        else:
            self.norm_self = None
            self.self_attn = None
        self.norm_ffn = nn.LayerNorm(self.dim)
        self.fc1 = nn.Linear(self.dim, self.ffn_mult * self.dim)
        self.act = _activation(activation)
        self.fc2 = nn.Linear(self.ffn_mult * self.dim, self.dim)

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.heads}, ffn={self.ffn_mult}x, "
                f"self_attn={self.has_self_attn}")

    def forward(self, q: Tensor, memory: Tensor,
                sink: list[dict[str, float]] | None = None) -> Tensor:
        """``q (B, N+1, d)`` x ``memory (B, M, d)`` -> ``(B, N+1, d)``."""
        h = self.norm_cross(q)
        q = q + self.cross_attn(h, memory, memory, need_weights=False)[0]
        if self.self_attn is not None:
            h = self.norm_self(q)
            q = q + self.self_attn(h, h, h, need_weights=False)[0]
        h = self.norm_ffn(q)
        pre = self.fc1(h)
        if sink is not None:
            with torch.no_grad():
                sink.append({
                    "near_zero_rate": float((pre.abs() < 1e-6).to(pre.dtype).mean()),
                    "pre_act_mean_abs": float(pre.abs().mean()),
                    "post_act_zero_rate": float(
                        (self.act(pre) == 0).to(pre.dtype).mean()),
                })
        return q + self.fc2(self.act(pre))


class GlutQueryDecoder(nn.Module):
    """``z (B, 2560) -> GlutParams``: ``N`` Gaussian queries + 1 global query.

    Parameters
    ----------
    n_gauss
        ``N``.  48 for this campaign; the colour PE is indexed by
        :func:`~q3vl.whatb.glut.grid_axis_sizes` (48 -> 4x4x3).
    dim, layers, heads, ffn_mult
        StatLUT's published decoder shape: 512 / 6 / 8 / 4x.
    self_attn
        Insert a query self-attention between the cross-attention and the FFN.
    mem_rows
        ``M``: how many rows the single condition vector is expanded into, one
        independent ``Linear(2560 -> d)`` each, sharing one front ``LayerNorm``.
    activation, head_init
        ``gelu`` / ``relu``; ``bias`` / ``zero`` (see :data:`QDEC_HEAD_INITS`).
    m_residual, g_residual
        ``M_i = I + dM_i`` (default on) and ``G = I + dG`` (default **off** --
        see the module docstring).
    sigma, opacity_logit
        The two ``SharedGeometry`` init values the Gaussian head's bias carries:
        ``softplus^-1(sigma)`` on ``chol_diag`` and ``opacity_logit`` on the
        opacity slot.

    The forward is batched and returns parameters on the decoder's own
    ``(device, dtype)``; ``z`` is cast, never assumed.
    """

    def __init__(
        self,
        *,
        n_gauss: int = 48,
        dim: int = 512,
        layers: int = 6,
        heads: int = 8,
        ffn_mult: int = 4,
        self_attn: bool = False,
        mem_rows: int = 1,
        activation: str = "gelu",
        head_init: str = "bias",
        in_dim: int = 2560,
        sigma: float = 0.15,
        opacity_logit: float = 4.0,
        m_residual: bool = True,
        g_residual: bool = False,
        query_init_std: float = 0.02,
        eps: float = EPS,
    ) -> None:
        super().__init__()
        if head_init not in QDEC_HEAD_INITS:
            raise ValueError(f"head_init must be one of {QDEC_HEAD_INITS}, got {head_init!r}")
        if int(mem_rows) < 1:
            raise ValueError(f"mem_rows must be >= 1, got {mem_rows!r}")
        if int(dim) % int(heads):
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        n = int(n_gauss)
        self.n_gauss = n
        self.dim, self.n_layers, self.heads = int(dim), int(layers), int(heads)
        self.ffn_mult, self.mem_rows = int(ffn_mult), int(mem_rows)
        self.has_self_attn = bool(self_attn)
        self.activation_name = activation
        self.head_init = head_init
        self.in_dim = int(in_dim)
        self.init_sigma, self.init_opacity_logit = float(sigma), float(opacity_logit)
        self.m_residual, self.g_residual = bool(m_residual), bool(g_residual)
        self.eps = float(eps)

        # -- queries: one per Gaussian + one global affine ---------------------
        self.q_emb = nn.Parameter(torch.randn(n + 1, self.dim) * float(query_init_std))
        axes = grid_axis_sizes(n)
        self.grid_axes = tuple(int(a) for a in axes)
        self.pe_r = nn.Parameter(torch.zeros(axes[0], self.dim))
        self.pe_g = nn.Parameter(torch.zeros(axes[1], self.dim))
        self.pe_b = nn.Parameter(torch.zeros(axes[2], self.dim))
        idx = torch.stack(torch.meshgrid(
            *[torch.arange(a) for a in axes], indexing="ij"), dim=-1).reshape(-1, 3)
        self.register_buffer("grid_index", idx.contiguous(), persistent=False)
        self.register_buffer("mu_base", uniform_grid_positions(n), persistent=False)
        self.register_buffer("eye3", torch.eye(3), persistent=False)

        # -- memory: the one condition vector, expanded to M rows --------------
        self.norm_z = nn.LayerNorm(self.in_dim)
        self.mem_proj = nn.ModuleList(
            [nn.Linear(self.in_dim, self.dim) for _ in range(self.mem_rows)])

        # -- decoder -----------------------------------------------------------
        self.layers = nn.ModuleList([
            QueryDecoderLayer(self.dim, self.heads, ffn_mult=self.ffn_mult,
                              activation=activation, self_attn=self.has_self_attn)
            for _ in range(self.n_layers)])

        # -- heads (Bias-HyperInit) --------------------------------------------
        self.head_gauss = nn.Linear(self.dim, 22)
        self.head_global = nn.Linear(self.dim, 12)
        self._init_heads()
        self._act_sink: list[dict[str, float]] | None = None

    # ---- initialisation ---------------------------------------------------- #
    def gauss_bias(self) -> Tensor:
        """The 22-vector the Gaussian head's bias is set to (``head_init=bias``)."""
        bias = torch.zeros(22)
        lo, hi = GAUSS_SLICES["chol_diag"]
        bias[lo:hi] = softplus_inverse(self.init_sigma)
        lo, hi = GAUSS_SLICES["opacity_logit"]
        bias[lo:hi] = self.init_opacity_logit
        return bias

    def global_bias(self) -> Tensor:
        """The 12-vector the global head's bias is set to.

        All zeros in both readings: ``d_G`` is a residual on ``I`` when
        ``g_residual`` and on ``0`` otherwise, and ``g`` is 0 either way.
        """
        return torch.zeros(12)

    def _init_heads(self) -> None:
        """Weight zeros; bias = the target init (or zeros for the ablation row).

        Text-to-LoRA's ``hyper_modulator.py`` is the shape being copied:
        ``nn.init.zeros_(layer.weight)`` followed by
        ``head.bias.copy_(torch.cat(init_bias))``.  Step 0 therefore emits one
        legal parameter set that does not depend on the condition -- not a
        parameter set of zeros.
        """
        for head in (self.head_gauss, self.head_global):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.head_init == "bias":
            with torch.no_grad():
                self.head_gauss.bias.copy_(self.gauss_bias())
                self.head_global.bias.copy_(self.global_bias())

    # ---- bookkeeping ------------------------------------------------------- #
    def extra_repr(self) -> str:
        return (f"n_gauss={self.n_gauss}, dim={self.dim}, layers={self.n_layers}, "
                f"heads={self.heads}, mem_rows={self.mem_rows}, "
                f"act={self.activation_name}, head_init={self.head_init}, "
                f"self_attn={self.has_self_attn}, grid={self.grid_axes}")

    @property
    def theta_dim(self) -> int:
        """``22N + 12`` -- every GLUT parameter is generated by this backbone."""
        return 22 * self.n_gauss + 12

    @property
    def config(self) -> dict[str, object]:
        """What ``run_setup.json`` records for the backbone."""
        return {
            "backbone": "qdec",
            "n_gauss": self.n_gauss,
            "n_queries": self.n_gauss + 1,
            "dim": self.dim,
            "layers": self.n_layers,
            "heads": self.heads,
            "ffn_mult": self.ffn_mult,
            "self_attn": self.has_self_attn,
            "mem_rows": self.mem_rows,
            "activation": self.activation_name,
            "head_init": self.head_init,
            "grid_axes": list(self.grid_axes),
            "m_residual": self.m_residual,
            "g_residual": self.g_residual,
            "sigma": self.init_sigma,
            "opacity_logit": self.init_opacity_logit,
            "theta_dim": self.theta_dim,
            "n_params": int(sum(p.numel() for p in self.parameters())),
            "n_params_closed_form": qdecoder_param_count(
                n_gauss=self.n_gauss, dim=self.dim, layers=self.n_layers,
                mem_rows=self.mem_rows, ffn_mult=self.ffn_mult,
                self_attn=self.has_self_attn, in_dim=self.in_dim),
        }

    def param_groups(self, base_lr: float, *, prior_lr_scale: float = 0.1,
                     head_lr_scale: float = 0.1) -> list[dict[str, object]]:
        """Three Adam groups.  The query prior **and the output head** run at ``0.1x``.

        GLUT App A.1 gives 0.1x to "the style embeddings and shared geometry
        parameters".  This backbone has neither by that name, so the mapping is
        NOVEL (as it is in EPR-029 §3.5) and is made in two places:

        * ``q_emb`` + the three colour PEs -- the only parameters indexed by the
          ``mu`` grid, i.e. the ones holding a geometry prior;
        * ``head_gauss`` / ``head_global`` -- under Bias-HyperInit **the head's
          bias *is* the shared geometry** (``mu`` grid residual base,
          ``sigma = 0.15``, ``o`` logit ``+4``, ``M = I``), which is exactly the
          object EPR-029 calls ``theta_base`` and puts in the 0.1x group.

        Measured (CPU, real z + real bank, fp32, B=32 x Q=256, seed 20260810,
        d=512 L=6, cosine over 300 steps -- ``scratchpad/probe_epr030.py``):

        ==========================  ==========================================
        head at ``1.0x`` (1e-3)     L_rec 0.19 -> 0.49 (s10) -> 0.58 (s30);
                                    ``gnorm`` 138 / 252 / 196; ``point_std`` and
                                    ``cross_std`` exactly 0 at s60; **NaN from s80**
        head at ``0.1x`` (1e-4)     L_rec 0.157-0.168 through s160 (identity is
                                    0.1756); ``gnorm`` 8-28 with two spikes;
                                    ``cross_std`` 4.8e-4 -> 3.1e-2; no NaN
        ==========================  ==========================================

        ``--qdec-head-lr-scale 1.0`` restores the other reading and is an
        ablation row.  Both numbers above go on the board; neither is a claim.
        """
        prior = [self.q_emb, self.pe_r, self.pe_g, self.pe_b]
        head = [self.head_gauss.weight, self.head_gauss.bias,
                self.head_global.weight, self.head_global.bias]
        taken = {id(p) for p in prior} | {id(p) for p in head}
        rest = [p for p in self.parameters() if id(p) not in taken]
        return [
            {"params": rest, "lr": float(base_lr), "name": "qdec"},
            {"params": prior, "lr": float(base_lr) * float(prior_lr_scale),
             "name": "qdec_query_prior"},
            {"params": head, "lr": float(base_lr) * float(head_lr_scale),
             "name": "qdec_head"},
        ]

    # ---- forward ----------------------------------------------------------- #
    @contextlib.contextmanager
    def capture_activations(self) -> Iterator[list[dict[str, float]]]:
        """Collect per-layer pre-activation statistics for the next forward(s).

        Yields the list the layers append to: one dict per layer per forward,
        with ``near_zero_rate`` (``|x| < 1e-6``), ``pre_act_mean_abs`` and
        ``post_act_zero_rate``.  Diagnostics only -- no graph is kept.
        """
        sink: list[dict[str, float]] = []
        prev, self._act_sink = self._act_sink, sink
        try:
            yield sink
        finally:
            self._act_sink = prev

    def queries(self, batch: int, *, device=None, dtype=None) -> Tensor:
        """``(B, N+1, d)``: ``emb[i] + PE_R[r] + PE_G[g] + PE_B[b]``, global query bare."""
        q = self.q_emb
        idx = self.grid_index.to(device=q.device)
        pe = self.pe_r[idx[:, 0]] + self.pe_g[idx[:, 1]] + self.pe_b[idx[:, 2]]
        q = torch.cat([q[: self.n_gauss] + pe, q[self.n_gauss:]], dim=0)
        if device is not None or dtype is not None:
            q = q.to(device=device or q.device, dtype=dtype or q.dtype)
        return q.unsqueeze(0).expand(int(batch), -1, -1)

    def memory(self, z: Tensor) -> Tensor:
        """``(B, 2560) -> (B, M, d)``: ``M`` independent projections of ``LN(z)``."""
        ref = self.mem_proj[0].weight
        if z.dim() != 2 or z.shape[-1] != self.in_dim:
            raise ValueError(f"z must be (B, {self.in_dim}), got {tuple(z.shape)}")
        h = self.norm_z(z.to(device=ref.device, dtype=ref.dtype))
        return torch.stack([p(h) for p in self.mem_proj], dim=1)

    def decode(self, memory: Tensor) -> GlutParams:
        """``(B, M, d)`` memory -> parameters.

        Split out from :meth:`forward` so an arm can interpolate **after** the
        condition projection (the ``--interp-mix-point post_pi`` column) the same
        way the MLP backbone interpolates after ``pi``.
        """
        if memory.dim() != 3 or memory.shape[-1] != self.dim:
            raise ValueError(f"memory must be (B, M, {self.dim}), got {tuple(memory.shape)}")
        q = self.queries(int(memory.shape[0]), device=memory.device, dtype=memory.dtype)
        for layer in self.layers:
            q = layer(q, memory, self._act_sink)
        return self.to_params(self.head_gauss(q[:, : self.n_gauss]),
                              self.head_global(q[:, self.n_gauss]))

    def forward(self, z: Tensor) -> GlutParams:  # noqa: D102
        return self.decode(self.memory(z))

    def to_params(self, out22: Tensor, out12: Tensor) -> GlutParams:
        """``(B, N, 22)`` + ``(B, 12)`` -> :class:`~q3vl.whatb.glut.GlutParams`."""
        b, n = int(out22.shape[0]), self.n_gauss
        eye = self.eye3.to(device=out22.device, dtype=out22.dtype)
        cut = lambda t, k: t[..., GAUSS_SLICES[k][0]: GAUSS_SLICES[k][1]]
        mu = self.mu_base.to(device=out22.device, dtype=out22.dtype) + cut(out22, "d_mu")
        d_m = cut(out22, "d_m").reshape(b, n, 3, 3)
        m_local = eye + d_m if self.m_residual else d_m
        d_g = out12[:, GLOBAL_SLICES["d_g_matrix"][0]: GLOBAL_SLICES["d_g_matrix"][1]]
        d_g = d_g.reshape(b, 3, 3)
        g_matrix = eye + d_g if self.g_residual else d_g
        return GlutParams(
            mu=mu,
            chol_diag=cut(out22, "chol_diag"),
            chol_off=cut(out22, "chol_off"),
            opacity_logit=cut(out22, "opacity_logit").squeeze(-1),
            m_local=m_local,
            b_local=cut(out22, "b"),
            g_matrix=g_matrix,
            g_bias=out12[:, GLOBAL_SLICES["g_bias"][0]: GLOBAL_SLICES["g_bias"][1]],
        )

    # ---- the step-0 assertion ---------------------------------------------- #
    @torch.no_grad()
    def reference_params(self, batch: int = 1) -> GlutParams:
        """The parameter set the head's **bias alone** encodes.

        With the weight at zero the head output is its bias for every query and
        every condition, so this is what step 0 must emit -- exactly.  Under
        ``head_init="bias"`` it is ``SharedGeometry``'s init
        (``mu`` = the grid, ``sigma`` = 0.15, ``o`` logit = +4) together with
        ``M = I``, ``b = 0``, ``G = 0``, ``g = 0``, i.e.
        ``GlutParams.identity``; under ``head_init="zero"`` the two geometry
        scalars are 0 instead.
        """
        ref = self.head_gauss.weight
        dev, dt = ref.device, ref.dtype
        bias = self.head_gauss.bias.detach()
        lo, hi = GAUSS_SLICES["chol_diag"]
        chol = bias[lo:hi]
        lo, hi = GAUSS_SLICES["opacity_logit"]
        opa = bias[lo]
        n, b = self.n_gauss, int(batch)
        eye = torch.eye(3, device=dev, dtype=dt)
        g_matrix = eye.expand(b, 3, 3).clone() if self.g_residual else \
            torch.zeros((b, 3, 3), device=dev, dtype=dt)
        return GlutParams(
            mu=self.mu_base.to(device=dev, dtype=dt).unsqueeze(0).expand(b, n, 3),
            chol_diag=chol.reshape(1, 1, 3).expand(b, n, 3).contiguous(),
            chol_off=torch.zeros((b, n, 3), device=dev, dtype=dt),
            opacity_logit=opa.reshape(1, 1).expand(b, n).contiguous(),
            m_local=eye.expand(b, n, 3, 3).clone(),
            b_local=torch.zeros((b, n, 3), device=dev, dtype=dt),
            g_matrix=g_matrix,
            g_bias=torch.zeros((b, 3), device=dev, dtype=dt))

    @torch.no_grad()
    def assert_step0_identity(self, *, n_grid: int = 17, atol: float = 1e-6,
                              batch: int = 4, generator: torch.Generator | None = None
                              ) -> dict[str, float]:
        """One real forward on random ``z``; raise unless step 0 is the init and the identity.

        Three things are asserted, all with a real forward, all raising (never
        warning -- "the zero-init head was not actually wired" is precisely the
        failure this catches):

        1. every :class:`GlutParams` field equals :meth:`reference_params`
           element-wise, i.e. the head emitted its bias and nothing else;
        2. under ``head_init="bias"`` that reference **is**
           ``GlutParams.identity`` (``SharedGeometry``'s ``mu`` grid,
           ``sigma = 0.15``, ``o`` logit ``+4``, ``M = I``, ``b = G = g = 0``);
        3. ``max |f(x) - x| <= atol`` on the ``n_grid^3`` sRGB grid.

        On (3) the floor is ``max(atol, the carrier's own deviation at the
        identity point)``: Eq.2's ``+ eps`` in the denominator means
        ``sum_i w_i = 1 - eps / (sum_j p_j o_j + eps) < 1`` exactly, so the
        identity point of the *carrier* is only identity up to that ratio.  At
        the campaign's ``N = 48`` it is ``4.2e-7`` (inside ``1e-6``); a small
        ``N`` spreads the same ``eps`` over fewer Gaussians and the deviation
        grows.  Attributing the carrier's ``eps`` to the decoder would make this
        assertion fire on a decoder that is bit-for-bit correct, so the two are
        reported separately (``carrier_eps_dev``).
        """
        from q3vl.whatb.glut import glut_forward   # local import: keeps the module graph acyclic

        ref = self.head_gauss.weight
        z = torch.randn(int(batch), self.in_dim, device=ref.device, dtype=ref.dtype,
                        generator=generator)
        params = self(z)
        want = self.reference_params(int(batch))
        report: dict[str, float] = {}
        for name in ("mu", "chol_diag", "chol_off", "opacity_logit",
                     "m_local", "b_local", "g_matrix", "g_bias"):
            dev = float((getattr(params, name) - getattr(want, name)).abs().max())
            report[f"init_dev_{name}"] = dev
            if dev != 0.0:
                raise AssertionError(
                    f"GlutQueryDecoder step-0 parameter {name} deviates from the "
                    f"head's own bias by {dev:.3e} (must be exactly 0); the "
                    "Bias-HyperInit head is not wired")
        if self.head_init == "bias":
            init = GlutParams.identity(
                self.n_gauss, batch=int(batch), device=want.device, dtype=want.dtype,
                sigma=self.init_sigma, opacity_logit=self.init_opacity_logit)
            # ``atol`` and not ``== 0``: ``init`` **recomputes** the grid on the
            # current device, while ``mu_base`` was computed at construction (CPU)
            # and moved.  ``(i + 0.5) / k`` differs between the CPU and the CUDA
            # division by up to one float32 ULP -- 2^-24 = 5.960e-08 at the 0.5 and
            # 0.8333 centres of a 3-cell axis, which is exactly what the first GPU
            # smoke of this arm reported.  A real mis-encoding of these fields is
            # O(0.1) (sigma), O(1) (M = I) or O(4) (the opacity logit), so a float32
            # resolution floor keeps the assertion sharp; the measured deviation is
            # reported either way.
            for name in ("mu", "chol_diag", "chol_off", "opacity_logit",
                         "m_local", "b_local", "g_bias"):
                dev = float((getattr(want, name) - getattr(init, name)).abs().max())
                report[f"shared_geometry_dev_{name}"] = dev
                if not (dev <= _F32_ULP):
                    raise AssertionError(
                        f"GlutQueryDecoder head bias encodes {name} as something other "
                        f"than the SharedGeometry init (max |diff| = {dev:.3e} > "
                        f"{_F32_ULP:.3e}, one float32 ULP)")

        x = torch.linspace(0.0, 1.0, int(n_grid), device=params.device, dtype=params.dtype)
        grid = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1).reshape(-1, 3)
        y = glut_forward(grid, params, clamp="two")
        y_ref = glut_forward(grid, want, clamp="two")
        if not torch.equal(y, y_ref):
            raise AssertionError(
                "GlutQueryDecoder step-0 forward differs from the forward of its own "
                f"bias-encoded parameters by {float((y - y_ref).abs().max()):.3e}")
        dev = float((y - grid.unsqueeze(0)).abs().max())
        carrier_dev = float((y_ref - grid.unsqueeze(0)).abs().max())
        report["step0_maxabs_f_minus_id"] = dev
        report["carrier_eps_dev"] = carrier_dev
        report["n_grid"] = float(n_grid)
        floor = max(float(atol), carrier_dev)
        if not (dev <= floor):
            raise AssertionError(
                f"GlutQueryDecoder step-0 forward is not the identity: "
                f"max |f(x) - x| = {dev:.3e} > {floor:.1e} on the {n_grid}^3 grid")
        if self.g_residual and dev > float(atol):
            raise AssertionError(
                "--g-residual anchors G at I, so step 0 is f = 2x, not the identity; "
                "this assertion must not be run on that row")
        return report


def qdecoder_param_count(*, n_gauss: int = 48, dim: int = 512, layers: int = 6,
                         mem_rows: int = 1, ffn_mult: int = 4,
                         self_attn: bool = False, in_dim: int = 2560) -> int:
    """Closed form of ``sum(p.numel())``, checked against the module in the tests.

    ``(N=48, d=512, L=6, M=1, gelu, no self-attn)`` -> **20,278,818**.
    """
    n, d, m = int(n_gauss), int(dim), int(mem_rows)
    lin = lambda i, o: i * o + o
    axes = grid_axis_sizes(n)
    total = (n + 1) * d                       # q_emb
    total += sum(axes) * d                    # PE_R + PE_G + PE_B
    total += 2 * int(in_dim)                  # LayerNorm(2560)
    total += m * lin(int(in_dim), d)          # M independent memory projections
    attn = 3 * d * d + 3 * d + lin(d, d)      # in_proj (w+b) + out_proj
    per_layer = 2 * d + attn + 2 * d + lin(d, ffn_mult * d) + lin(ffn_mult * d, d)
    if self_attn:
        per_layer += 2 * d + attn
    total += int(layers) * per_layer
    total += lin(d, 22) + lin(d, 12)          # the two heads
    return int(total)
