"""QDUAL (EPR-029): N Gaussian queries + L layers of cross-attention decoding.

Authority, in order: the arm's proposal
``experiments/prs/EPR-029_gaussian-query-dual-cond-decoder/PROPOSAL.md`` (the
cross-arm frozen block at ``:15-70``, the model pseudo-code at ``:374-424``, the
parameter-count table at ``:436-453``, the maths at ``:493-529``, the fusion
ladder at ``:531-548``, the loss at ``:554-578``, the optimiser at ``:580-608``,
the diagnostics at ``:610-631``, the wiring table at ``:633-641``, the collapse
guard at ``:642-656`` and the pre-registered criteria at ``:657-774``), then
``docs/HANDOFF_whatb_2026-08-15.md``.

The one structural change
------------------------
The generator ``G_v: (z_color, S) -> Theta`` stops being "shared 3-layer MLP +
five parameter heads" (CGLUT App A.2, which is EPR-024 and lives in
``q3vl/whatb/generator.py``) and becomes a **Transformer decoder**: one learnable
query per Gaussian, the language vector and the spatial field in one condition
memory as K/V, ``L`` layers of (cross-attention -> FFN), one 22-dim output per
query plus one extra query for the 12 global-affine numbers.  The carrier (GLUT
Eq.1-5), the supervision space, the optimiser, the step budget and the criteria
are untouched -- they are imported, not re-implemented.

    q_i^(0) = emb[i] + PE_R[r_i] (+) PE_G[g_i] (+) PE_B[b_i]      StatLUT Eq.2
    m^z_k   = W^(k) LN(z_color) + E_type[0]                       StatLUT Eq.1
    m^s_t   = W_p vec(P_t) + PE_row[r_t] + PE_col[c_t] + E_type[1]
    q^(l+.5)= q^(l) + W_o softmax(W_q LN(q) (W_k M)^T / sqrt(d/H)) W_v M
    q^(l+1) = q^(l+.5) + FFN(LN(q^(l+.5)))                        StatLUT Eq.3
    dtheta_i= W_g q_i^(L) + b_g ,  (dG, dg) = W_a q_{N+1}^(L) + b_a
    theta   = theta_base + (dtheta, dG, dg)                       StatLUT Eq.4

``W_g, b_g, W_a, b_a`` are **zero-initialised** (StatLUT section 3.2: "we
zero-initialize the final FFN projection layer ... guarantees an initial identity
mapping"), so step 0 has ``dtheta == 0`` exactly, ``theta == theta_base``, and
``theta_base`` is GLUT App A.1's init plus ``G = 0, g = 0`` (NOVEL, NOTES 3 --
the only choice that makes step 0 exactly the identity, proposition 2).

Fusion ladder (``--rung``), each row changing exactly one thing::

    a  memory = language rows only                      theta = G(z)
    b  + one pooled field token                         theta = G(z, pool(S))
    c  + T field tokens                (arm default)    theta = G(z, S)
    d  cross-attention -> FiLM/GFM     baseline is (b)  CSRNet_arch.py:66
    e  cross-attention -> broadcast add baseline is (b) HDRNet section 3.1.4 Eq.2

(d)/(e) consume the rung-(b) memory: their operators eat a single vector, so
pairing them against (c) would change operator *and* field granularity in one
row.  That is a bookkeeping constraint, not an interpretation (proposal :546).

Discipline this module runs under (each item paid for on the where side)
------------------------------------------------------------------------
* Every constant that a ``forward`` needs is a ``register_buffer(...,
  persistent=False)``; there is no ``torch.tensor(...)`` inside a forward, and
  every incoming tensor is moved with ``.to(device=ref.device, dtype=ref.dtype)``
  before it meets a parameter (EPR-022/MATTE died on exactly that).
* Nothing here calls ``.cpu()`` on a quantity that a criterion then reduces
  (CPU/CUDA top-k tie-breaks moved a where-side IoU by 0.296).
* The degenerate-solution guard is wired at the **first quick eval**
  (:func:`assert_not_degenerate`, three conditions, ``SystemExit(2)``), because
  PRND/CONDINST burned 2.6 GPU-hours emitting a constant field.
* Nothing imports ``q3vl.what`` / ``model.glut_repro`` / ``gpu_render``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field as _field
from typing import Any, Literal, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from q3vl.whatb.caliber import (
    effective_lambda_hc,
    effective_lambda_sparse,
    pure_l1_record,
)
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.glut import (
    CLAMP_FLAG_CHOICES,
    EPS,
    GlutAux,
    GlutCarrier,
    GlutParams,
    n_params_glut,
    softplus_inverse,
    uniform_grid_positions,
)
from q3vl.whatb.guards import (
    DegeneracyReport,
    DegeneracyThresholds,
    assert_transform_not_degenerate,
)
from q3vl.whatb.queries import mining_ratio

__all__ = [
    "ARM",
    "ARM_EPR",
    "ARM_AXES",
    "REQUIRED_QDUAL",
    "ARM_REQUIRED_TABLE",
    "LADDER_ROWS",
    "SEG_COLOR_HIDDEN_DIM",
    "LAMBDA_HC",
    "LAMBDA_SPARSE",
    "EPS_CHROMA",
    "COLLAPSE_M_THRESHOLD",
    "COLLAPSE_DELTA_THRESHOLD",
    "QDualConfig",
    "ThetaBase",
    "XAttnFFN",
    "GaussianQueryDecoder",
    "QDualArm",
    "LossOutput",
    "qdual_losses",
    "rec_loss",
    "hue_chroma_loss",
    "LAB_FLOOR",
    "sparse_regulariser",
    "mine_hard_colors",
    "zero_init_witness",
    "assert_zero_init",
    "assert_not_degenerate",
    "hungarian_match",
    "query_match_repeat",
    "query_match_path",
    "collapse_guard",
    "loss_preregistration",
    "param_count_breakdown",
]

# --------------------------------------------------------------------------- #
# 0. names, frozen constants, the pre-registered table
# --------------------------------------------------------------------------- #
#: run/arm name.  The proposal spells the ``assert_criteria_ran`` table's key
#: ``"GQDEC"`` (:758); the campaign task card calls the arm ``QDUAL``.  Both名
#: resolve to the same required list -- see :data:`ARM_REQUIRED_TABLE`.
ARM: str = "QDUAL"
ARM_EPR: str = "EPR-029"
#: P3 (does theta depend on the spatial field S) -- HANDOFF section 五 index table.
ARM_AXES: tuple[str, ...] = ("P3",)

#: ``assert_criteria_ran`` required table, transcribed from ``EPR-029:756-773``.
#: The first twelve are the frozen cross-arm key table; the P2/P3 block adds the
#: locality and field-consumption columns (``field_pred`` is in this arm's list
#: and NOT in ``criteria.REQUIRED_P2P3``, so the list is passed explicitly); the
#: last six are this arm's own diagnostics.
REQUIRED_QDUAL: tuple[str, ...] = (
    "headline_normal_only",
    "B0_identity", "B1_libmean", "B2_librandom", "B3_bucket_retrieval", "B4_oracle",
    "N1_shuffle_delta", "N1_shuffle_M",
    "N2_irrelevant_delta", "N2_irrelevant_M",
    "N3_const_delta", "N3_const_M",
    "loc_in", "loc_band", "loc_out",
    "field_gt", "field_pred", "field_const", "field_shuffle",
    "query_match_drift_repeat", "query_match_drift_path",
    "query_match_mu_shift_indexed", "query_match_mu_shift_hungarian",
    "ladder_row", "zeroinit_step0_maxabs",
)

#: ``--no-zero-init-head`` is the one ablation row whose required table drops
#: ``zeroinit_step0_maxabs`` (``EPR-029:772-773``); the board notes it.
REQUIRED_QDUAL_NO_ZEROINIT: tuple[str, ...] = tuple(
    k for k in REQUIRED_QDUAL if k != "zeroinit_step0_maxabs")

ARM_REQUIRED_TABLE: dict[str, tuple[str, ...]] = {
    "QDUAL": REQUIRED_QDUAL,
    "GQDEC": REQUIRED_QDUAL,
    "EPR-029": REQUIRED_QDUAL,
}

LADDER_ROWS: tuple[str, ...] = ("a", "b", "c", "d", "e")
Rung = Literal["a", "b", "c", "d", "e"]

#: Qwen3-VL-4B v2seg last-layer hidden width at ``<seg_color>`` (id 151674).
SEG_COLOR_HIDDEN_DIM: int = 2560

#: GLUT section 4.1 -- transcribed, not tuned.
LAMBDA_HC: float = 10.0
LAMBDA_SPARSE: float = 0.001
#: frozen block: ``h = (a,b)/max(C, eps_C)`` AND a hard mask ``1[C >= eps_C]``.
EPS_CHROMA: float = 1e-3

#: collapse guard (proposal section 3.8, NOVEL thresholds, NOTES 7 -- both in dE00)
COLLAPSE_M_THRESHOLD: float = 0.5
COLLAPSE_DELTA_THRESHOLD: float = 0.1

#: NOTES 13: ``PE_row`` / ``PE_col`` are 32 long; a patch grid larger than that
#: raises rather than being silently truncated.
FIELD_PE_MAX: int = 32

#: the 22 per-Gaussian output dims are laid out exactly as ``GlutParams.flat()``
_THETA_SLICES: dict[str, slice] = {
    "mu": slice(0, 3), "chol_diag": slice(3, 6), "chol_off": slice(6, 9),
    "opacity_logit": slice(9, 10), "m_local": slice(10, 19), "b_local": slice(19, 22),
}


# --------------------------------------------------------------------------- #
# 1. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QDualConfig:
    """Every structural flag of ``EPR-029:640``, in one record for ``run_setup``.

    Defaults are the main arm: rung (c), N=48, L=4, d=256, H=8 heads, K=4 with
    the ``proj`` expansion, ``m_low`` field source with 4x4 patches, GT field,
    no attention temperature, zero-initialised heads, uniform colour sampling,
    ``--clamp two``.
    """

    rung: str = "c"
    n_gauss: int = 48
    decoder_layers: int = 4
    decoder_width: int = 256
    decoder_heads: int = 8
    z_expand: str = "proj"            # proj (default) | qtok
    z_expand_k: int = 4
    field_source: str = "m_low"       # m_low (patch 4) | m_pix (patch 16)
    field_kind: str = "gt"            # gt | pred | const | shuffle
    attn_temperature: bool = False
    zero_init_head: bool = True
    color_sampling: str = "uniform"   # uniform | alpha_hist
    readout: str = "seg_color"
    readout_qtok: int = 0
    clamp: str = "two"
    residual: bool = True
    # theta_base geometry (GLUT App A.1; the opacity logit is the NOVEL value
    # EPR-025 fixed, because sigmoid cannot reach the paper's o = 1.0)
    sigma: float = 0.15
    opacity_logit: float = 4.0
    # colour PE (StatLUT Eq.2) and E_type (StatLUT Eq.1) can each be ablated off
    color_pe: bool = True
    e_type: bool = True
    field_grid_h: int = 32            # m_low grid (gh, gw); short side 512 -> 32x48
    field_grid_w: int = 48
    eps: float = EPS

    def __post_init__(self) -> None:
        if self.rung not in LADDER_ROWS:
            raise ValueError(f"--rung must be one of {LADDER_ROWS}, got {self.rung!r}")
        if self.z_expand not in ("proj", "qtok"):
            raise ValueError(f"--z-expand must be proj|qtok, got {self.z_expand!r}")
        if int(self.z_expand_k) < 1:
            raise ValueError("--z-expand-k must be >= 1")
        if self.field_source not in ("m_low", "m_pix"):
            raise ValueError(
                f"--field-source must be m_low|m_pix, got {self.field_source!r}")
        if self.field_kind not in ("gt", "pred", "const", "shuffle"):
            raise ValueError(f"unknown --field-kind {self.field_kind!r}")
        if self.color_sampling not in ("uniform", "alpha_hist"):
            raise ValueError(f"unknown --color-sampling {self.color_sampling!r}")
        if self.clamp not in CLAMP_FLAG_CHOICES:
            raise ValueError(
                f"--clamp must be one of {CLAMP_FLAG_CHOICES} (frozen block; "
                f"'none' has no CLI spelling), got {self.clamp!r}")
        if self.decoder_width % self.decoder_heads:
            raise ValueError(
                f"d={self.decoder_width} is not divisible by H={self.decoder_heads}")
        if self.n_gauss < 1:
            raise ValueError("--n-gauss must be >= 1")

    # -- derived ------------------------------------------------------------
    @property
    def patch(self) -> int:
        """4x4 blocks on ``m_low``, 16x16 on ``m_pix`` (proposal :640)."""
        return 4 if self.field_source == "m_low" else 16

    @property
    def field_hw(self) -> tuple[int, int]:
        """The field resolution this carrier consumes.

        ``m_low``: the where grid itself (32x48 for a short-side-512 image, from
        ``q3vl/where/fpre.py:45-49``).  ``m_pix``: 4x that, so both sources give
        the same T ~ 96 tokens with their own patch size (proposal :640).
        """
        k = 1 if self.field_source == "m_low" else 4
        return (int(self.field_grid_h) * k, int(self.field_grid_w) * k)

    @property
    def token_grid(self) -> tuple[int, int]:
        h, w = self.field_hw
        p = self.patch
        if h % p or w % p:
            raise ValueError(
                f"field {h}x{w} is not divisible by the {p}x{p} patch")
        return h // p, w // p

    @property
    def n_field_tokens(self) -> int:
        r, c = self.token_grid
        return r * c

    @property
    def uses_field(self) -> bool:
        """Rung (a) is the only row whose memory has no field row at all."""
        return self.rung != "a"

    @property
    def memory_rung(self) -> str:
        """(d)/(e) build the rung-(b) memory and then average it."""
        return "b" if self.rung in ("d", "e") else self.rung

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update({"arm": ARM, "epr": ARM_EPR, "patch": self.patch,
                    "field_hw": list(self.field_hw),
                    "token_grid": list(self.token_grid),
                    "n_field_tokens": self.n_field_tokens,
                    "memory_rung": self.memory_rung,
                    "theta_dim": n_params_glut(self.n_gauss)})
        return out


# --------------------------------------------------------------------------- #
# 2. theta_base -- GLUT App A.1 init, plus G = 0, g = 0 (NOVEL, NOTES 3)
# --------------------------------------------------------------------------- #
class ThetaBase(nn.Module):
    """The ``22N + 12`` learnable base parameters the decoder writes residuals on.

    Init (GLUT App A.1, verbatim except where marked): ``mu`` on the uniform grid
    covering ``[0,1]^3``; ``Sigma`` isotropic with ``sigma = 0.15`` through the
    Cholesky diagonal (``softplus^-1(0.15) = -1.8212...``); ``o`` at logit 4.0
    (**NOVEL**: App A.1 says 1.0, which ``sigmoid`` cannot reach -- the same
    value ``q3vl/whatb/generator.py`` fixed for EPR-025); ``M_i = I``, ``b_i = 0``;
    and ``G = 0, g = 0`` (**NOVEL**, NOTES 3: App A.1 does not separate ``M_i``
    from ``G``, and ``G = I`` would make step 0 ``f = 2x`` instead of identity).

    With ``dtheta = 0`` this is proposition 2's exact point: ``sum_i w_i = 1`` at
    ``eps = 0`` gives ``f(x) = x``.
    """

    def __init__(self, n_gauss: int, *, sigma: float = 0.15,
                 opacity_logit: float = 4.0) -> None:
        super().__init__()
        n = int(n_gauss)
        self.n_gauss = n
        self.init_sigma = float(sigma)
        self.init_opacity_logit = float(opacity_logit)
        grid = uniform_grid_positions(n)
        self.mu = nn.Parameter(grid.clone())
        self.chol_diag = nn.Parameter(torch.full((n, 3), softplus_inverse(sigma)))
        self.chol_off = nn.Parameter(torch.zeros(n, 3))
        self.opacity_logit = nn.Parameter(torch.full((n,), float(opacity_logit)))
        self.m_local = nn.Parameter(torch.eye(3).expand(n, 3, 3).clone())
        self.b_local = nn.Parameter(torch.zeros(n, 3))
        self.g_matrix = nn.Parameter(torch.zeros(3, 3))
        self.g_bias = nn.Parameter(torch.zeros(3))
        #: the (r, g, b) grid index of every Gaussian, derived from ``mu`` itself
        #: so the colour PE can never disagree with the mean it is meant to name.
        self.register_buffer("grid_index", _grid_index(grid), persistent=False)
        self.register_buffer("grid_extent", torch.tensor(
            [int(grid[:, k].unique().numel()) for k in range(3)]), persistent=False)

    @property
    def n_params(self) -> int:
        return n_params_glut(self.n_gauss)

    def extra_repr(self) -> str:
        return (f"n_gauss={self.n_gauss}, sigma={self.init_sigma}, "
                f"opacity_logit={self.init_opacity_logit}, G=0, g=0")

    def parameters_list(self) -> list[nn.Parameter]:
        """The 0.1x-lr group's share of ``theta_base`` (proposal section 2.3)."""
        return [self.mu, self.chol_diag, self.chol_off, self.opacity_logit,
                self.m_local, self.b_local, self.g_matrix, self.g_bias]

    def compose(self, dtheta: Tensor | None, dglob: Tensor | None,
                *, batch: int = 1) -> GlutParams:
        """``theta = theta_base + (dtheta, dglob)`` -- StatLUT Eq.4's residual form.

        ``dtheta`` is ``(B, N, 22)`` in :meth:`GlutParams.flat`'s per-Gaussian
        order and ``dglob`` is ``(B, 12)``; ``None`` means "no residual", which
        is what a step-0 witness and the identity baseline want.
        """
        ref = self.mu
        if dtheta is None:
            b = int(batch)
            zeros = ref.new_zeros((b, self.n_gauss, 22))
        else:
            zeros = dtheta
            b = int(zeros.shape[0])
        if dglob is None:
            dg = ref.new_zeros((b, 12))
        else:
            dg = dglob
        dev, dt = zeros.device, zeros.dtype
        base = lambda t: t.to(device=dev, dtype=dt).unsqueeze(0)
        s = _THETA_SLICES
        return GlutParams(
            mu=base(self.mu) + zeros[..., s["mu"]],
            chol_diag=base(self.chol_diag) + zeros[..., s["chol_diag"]],
            chol_off=base(self.chol_off) + zeros[..., s["chol_off"]],
            opacity_logit=(self.opacity_logit.to(device=dev, dtype=dt).unsqueeze(0)
                           + zeros[..., s["opacity_logit"]].squeeze(-1)),
            m_local=base(self.m_local) + zeros[..., s["m_local"]].reshape(
                b, self.n_gauss, 3, 3),
            b_local=base(self.b_local) + zeros[..., s["b_local"]],
            g_matrix=self.g_matrix.to(device=dev, dtype=dt).unsqueeze(0)
            + dg[:, :9].reshape(b, 3, 3),
            g_bias=self.g_bias.to(device=dev, dtype=dt).unsqueeze(0) + dg[:, 9:],
        )


def _grid_index(grid: Tensor) -> Tensor:
    """``(N, 3)`` long: which R / G / B level of the regular grid each mean sits on."""
    idx = []
    for k in range(3):
        levels = torch.unique(grid[:, k])
        idx.append(torch.searchsorted(levels, grid[:, k].contiguous()))
    return torch.stack(idx, dim=-1).long()


# --------------------------------------------------------------------------- #
# 3. one decoder layer
# --------------------------------------------------------------------------- #
class XAttnFFN(nn.Module):
    """Pre-norm ``(cross-attention -> FFN)``; the ladder swaps the first sublayer.

    ``op = "xattn"`` (rungs a/b/c) is proposal :510-511 -- multi-head
    ``softmax(W_q LN(q) (W_k M)^T / sqrt(d/H)) W_v M`` with an output projection
    and a residual.  ``--attn-temperature`` divides the logits by a learnable
    scalar initialised to 1.0, which is SA-LUT ``model.py:128`` (an ablation row;
    the main arm does not carry it).

    ``op = "film"`` (rung d) is CSRNet ``CSRNet_arch.py:66`` --
    ``out = out*scale + shift + out`` = ``(1 + gamma) q + beta`` with ``gamma``
    and ``beta`` from **two independent** ``Linear`` (``:38-44``) on the mean of
    the memory.

    ``op = "badd"`` (rung e) is HDRNet section 3.1.4 Eq.2 --
    ``q <- sigma(b + W' mbar + W q)`` with ``sigma = ReLU``.

    The FFN sublayer is identical in all three: only the first operator is the
    experiment's variable.
    """

    def __init__(self, d: int, heads: int, *, ffn_mult: int = 4,
                 op: str = "xattn", attn_temperature: bool = False) -> None:
        super().__init__()
        if op not in ("xattn", "film", "badd"):
            raise ValueError(f"unknown decoder operator {op!r}")
        self.d = int(d)
        self.heads = int(heads)
        self.op = op
        self.head_dim = self.d // self.heads
        self.norm1 = nn.LayerNorm(self.d)
        self.norm2 = nn.LayerNorm(self.d)
        self.ffn = nn.Sequential(nn.Linear(self.d, ffn_mult * self.d), nn.ReLU(),
                                 nn.Linear(ffn_mult * self.d, self.d))
        if op == "xattn":
            self.w_q = nn.Linear(self.d, self.d)
            self.w_k = nn.Linear(self.d, self.d)
            self.w_v = nn.Linear(self.d, self.d)
            self.w_o = nn.Linear(self.d, self.d)
            self.attn_temperature = (
                nn.Parameter(torch.tensor(1.0)) if attn_temperature else None)
        elif op == "film":
            self.cond_scale = nn.Linear(self.d, self.d)
            self.cond_shift = nn.Linear(self.d, self.d)
            self.attn_temperature = None
        else:                                   # badd
            self.w_local = nn.Linear(self.d, self.d)          # W q + b
            self.w_global = nn.Linear(self.d, self.d, bias=False)   # W' mbar
            self.attn_temperature = None

    def extra_repr(self) -> str:
        return (f"d={self.d}, heads={self.heads}, op={self.op}, "
                f"temperature={self.attn_temperature is not None}")

    def _attend(self, q: Tensor, memory: Tensor) -> Tensor:
        b, nq, d = q.shape
        h, hd = self.heads, self.head_dim
        qh = self.w_q(q).reshape(b, nq, h, hd).transpose(1, 2)
        kh = self.w_k(memory).reshape(b, memory.shape[1], h, hd).transpose(1, 2)
        vh = self.w_v(memory).reshape(b, memory.shape[1], h, hd).transpose(1, 2)
        scale = math.sqrt(hd)
        logits = torch.matmul(qh, kh.transpose(-1, -2)) / scale
        if self.attn_temperature is not None:
            logits = logits / self.attn_temperature.to(
                device=logits.device, dtype=logits.dtype)
        attn = torch.softmax(logits, dim=-1)
        ctx = torch.matmul(attn, vh).transpose(1, 2).reshape(b, nq, d)
        return self.w_o(ctx)

    def forward(self, q: Tensor, memory: Tensor) -> Tensor:  # noqa: D102
        qn = self.norm1(q)
        if self.op == "xattn":
            q = q + self._attend(qn, memory)
        elif self.op == "film":
            mbar = memory.mean(dim=1, keepdim=True)
            gamma, beta = self.cond_scale(mbar), self.cond_shift(mbar)
            q = qn * gamma + beta + q               # CSRNet_arch.py:66 verbatim
        else:
            mbar = memory.mean(dim=1, keepdim=True)
            q = F.relu(self.w_local(qn) + self.w_global(mbar))   # HDRNet Eq.2
        return q + self.ffn(self.norm2(q))


# --------------------------------------------------------------------------- #
# 4. the decoder
# --------------------------------------------------------------------------- #
class GaussianQueryDecoder(nn.Module):
    """``(z_color, S) -> GlutParams``.  The arm's only structural change.

    ``forward(z, field)``:

    ``z``     ``(B, 2560)`` for ``--z-expand proj`` (the default: K independent
              ``Linear(2560 -> d)`` behind one shared ``LayerNorm(2560)``), or
              ``(B, K, 2560)`` for ``--z-expand qtok`` (K vectors read out of the
              VLM, each through **the same** ``Linear``; proposal :601-608).
    ``field`` ``(B, fh, fw)`` (or ``(B, 1, fh, fw)``) -- the spatial field at this
              carrier's own resolution; ``None`` is only legal at rung (a).

    Returns :class:`GlutParams` with batch ``B``.
    """

    def __init__(self, cfg: QDualConfig | None = None, **kw: Any) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else QDualConfig(**kw)
        c = self.cfg
        n, d, k = c.n_gauss, c.decoder_width, int(c.z_expand_k)
        self.n_gauss, self.d, self.k_rows = n, d, k

        # -- query side (proposal :380-382) --
        self.q_emb = nn.Parameter(torch.randn(n + 1, d) * 0.02)
        self.theta_base = ThetaBase(n, sigma=c.sigma, opacity_logit=c.opacity_logit)
        extent = [int(v) for v in self.theta_base.grid_extent.tolist()]
        self.grid_extent = tuple(extent)
        if c.color_pe:
            self.pe_r = nn.Parameter(torch.zeros(extent[0], d))
            self.pe_g = nn.Parameter(torch.zeros(extent[1], d))
            self.pe_b = nn.Parameter(torch.zeros(extent[2], d))
        else:
            self.pe_r = self.pe_g = self.pe_b = None

        # -- memory side (proposal :385-389) --
        self.ln_z = nn.LayerNorm(SEG_COLOR_HIDDEN_DIM)
        if c.z_expand == "proj":
            self.proj_z = nn.ModuleList(
                [nn.Linear(SEG_COLOR_HIDDEN_DIM, d) for _ in range(k)])
        else:                                   # qtok: one shared projection
            self.proj_z = nn.ModuleList([nn.Linear(SEG_COLOR_HIDDEN_DIM, d)])
        self.patch = nn.Linear(c.patch * c.patch, d)
        rows, cols = c.token_grid
        if rows > FIELD_PE_MAX or cols > FIELD_PE_MAX:
            raise ValueError(
                f"patch grid {rows}x{cols} exceeds the {FIELD_PE_MAX}x{FIELD_PE_MAX} "
                "field PE (NOTES 13); this raises rather than truncating silently")
        self.pe_row = nn.Parameter(torch.zeros(FIELD_PE_MAX, d))
        self.pe_col = nn.Parameter(torch.zeros(FIELD_PE_MAX, d))
        self.e_type = nn.Parameter(torch.zeros(2, d)) if c.e_type else None

        # -- decoder + heads (proposal :392-394) --
        op = {"a": "xattn", "b": "xattn", "c": "xattn",
              "d": "film", "e": "badd"}[c.rung]
        self.layers = nn.ModuleList([
            XAttnFFN(d, c.decoder_heads, op=op,
                     attn_temperature=c.attn_temperature)
            for _ in range(c.decoder_layers)])
        self.head_g = nn.Linear(d, 22)
        self.head_a = nn.Linear(d, 12)
        if c.zero_init_head:
            for head in (self.head_g, self.head_a):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            # the assertion the proposal asks for (:637-①), at construction time
            with torch.no_grad():
                worst = max(float(t.abs().max()) for t in
                            (self.head_g.weight, self.head_g.bias,
                             self.head_a.weight, self.head_a.bias))
            if worst != 0.0:
                raise AssertionError(
                    f"zero-init of the output heads did not take (max |w| = {worst})")

        # -- constants as buffers, never a forward-time torch.tensor(...) --
        self.register_buffer("row_index", torch.arange(rows), persistent=False)
        self.register_buffer("col_index", torch.arange(cols), persistent=False)

    # -- bookkeeping --------------------------------------------------------
    def extra_repr(self) -> str:
        c = self.cfg
        return (f"rung={c.rung}, N={c.n_gauss}, L={c.decoder_layers}, "
                f"d={c.decoder_width}, H={c.decoder_heads}, K={c.z_expand_k}, "
                f"z_expand={c.z_expand}, field={c.field_source}{list(c.field_hw)}, "
                f"zero_init={c.zero_init_head}")

    @property
    def config(self) -> dict[str, Any]:
        out = self.cfg.to_dict()
        out.update(param_count_breakdown(self))
        out["grid_extent"] = list(self.grid_extent)
        return out

    def param_groups(self, base_lr: float, *, geometry_lr_scale: float = 0.1
                     ) -> list[dict[str, Any]]:
        """Two groups (proposal section 2.3).

        0.1x base lr: ``q_emb``, the colour PE and ``theta_base`` -- the only
        parameters holding a geometric prior, mapped onto App A.1's "style
        embeddings and shared geometry parameters" rule (**NOVEL mapping**,
        NOTES 4; the alternative is a flat 1x and is not the default because the
        paper explicitly asks for 0.1x).  Base lr: everything generative.
        """
        slow: list[nn.Parameter] = [self.q_emb, *self.theta_base.parameters_list()]
        for pe in (self.pe_r, self.pe_g, self.pe_b):
            if pe is not None:
                slow.append(pe)
        slow_ids = {id(p) for p in slow}
        fast = [p for p in self.parameters() if id(p) not in slow_ids]
        return [
            {"params": fast, "lr": float(base_lr), "name": "generator"},
            {"params": slow, "lr": float(base_lr) * float(geometry_lr_scale),
             "name": "query_prior"},
        ]

    # -- pieces -------------------------------------------------------------
    def queries(self, batch: int, *, ref: Tensor | None = None) -> Tensor:
        """``(B, N+1, d)`` -- ``emb`` plus the broadcast-added colour PE."""
        ref = self.q_emb if ref is None else ref
        q = self.q_emb.to(device=ref.device, dtype=ref.dtype)
        if self.pe_r is not None:
            gi = self.theta_base.grid_index.to(device=q.device)
            pe = (self.pe_r.to(dtype=q.dtype)[gi[:, 0]]
                  + self.pe_g.to(dtype=q.dtype)[gi[:, 1]]
                  + self.pe_b.to(dtype=q.dtype)[gi[:, 2]])
            q = torch.cat([q[: self.n_gauss] + pe, q[self.n_gauss:]], dim=0)
        return q.unsqueeze(0).expand(int(batch), -1, -1)

    def _language_rows(self, z: Tensor) -> Tensor:
        ref = self.ln_z.weight
        z = z.to(device=ref.device, dtype=ref.dtype)
        if self.cfg.z_expand == "proj":
            if z.dim() != 2 or z.shape[-1] != SEG_COLOR_HIDDEN_DIM:
                raise ValueError(
                    f"--z-expand proj wants z of (B, {SEG_COLOR_HIDDEN_DIM}), got "
                    f"{tuple(z.shape)}")
            h = self.ln_z(z)
            rows = torch.stack([p(h) for p in self.proj_z], dim=1)   # (B, K, d)
        else:
            if z.dim() != 3 or z.shape[-1] != SEG_COLOR_HIDDEN_DIM:
                raise ValueError(
                    f"--z-expand qtok wants z of (B, K, {SEG_COLOR_HIDDEN_DIM}), got "
                    f"{tuple(z.shape)}")
            if z.shape[1] != self.k_rows:
                raise ValueError(
                    f"--z-expand-k {self.k_rows} but the cache handed "
                    f"{z.shape[1]} query-token vectors")
            rows = self.proj_z[0](self.ln_z(z))                      # (B, K, d)
        if self.e_type is not None:
            rows = rows + self.e_type[0].to(dtype=rows.dtype)
        return rows

    def _field_rows(self, field: Tensor, ref: Tensor) -> Tensor:
        """``(B, T, d)`` (rung c) or ``(B, 1, d)`` (rungs b/d/e)."""
        c = self.cfg
        p = c.patch
        s = field.to(device=ref.device, dtype=ref.dtype)
        if s.dim() == 3:
            s = s.unsqueeze(1)
        if s.dim() != 4 or s.shape[1] != 1:
            raise ValueError(f"field must be (B, fh, fw) or (B, 1, fh, fw), got "
                             f"{tuple(field.shape)}")
        if c.memory_rung == "b":
            pooled = F.adaptive_avg_pool2d(s, (p, p))                # area pooling
            rows = self.patch(pooled.reshape(s.shape[0], 1, p * p))
        else:
            fh, fw = c.field_hw
            if (s.shape[-2], s.shape[-1]) != (fh, fw):
                raise ValueError(
                    f"field is {tuple(s.shape[-2:])} but --field-source "
                    f"{c.field_source} declares {fh}x{fw}; resample it with "
                    "q3vl.where.upsample.area_resize before the forward, and "
                    "record the resolution in run_config (frozen block)")
            blocks = F.unfold(s, kernel_size=p, stride=p)             # (B, p*p, T)
            rows = self.patch(blocks.transpose(1, 2))                 # (B, T, d)
            r, cc = c.token_grid
            ri = self.row_index.to(device=rows.device)
            ci = self.col_index.to(device=rows.device)
            pe = (self.pe_row.to(dtype=rows.dtype)[ri].reshape(r, 1, self.d)
                  + self.pe_col.to(dtype=rows.dtype)[ci].reshape(1, cc, self.d))
            rows = rows + pe.reshape(1, r * cc, self.d)
        if self.e_type is not None:
            rows = rows + self.e_type[1].to(dtype=rows.dtype)
        return rows

    def build_memory(self, z: Tensor, field: Tensor | None) -> Tensor:
        """``(B, K [+ 1 | + T], d)`` -- StatLUT Eq.1's additive type embedding."""
        rows_z = self._language_rows(z)
        if self.cfg.memory_rung == "a":
            return rows_z
        if field is None:
            raise ValueError(
                f"--rung {self.cfg.rung} consumes the spatial field but none was "
                "handed in; only rung (a) may run without one")
        return torch.cat([rows_z, self._field_rows(field, rows_z)], dim=1)

    def residuals(self, z: Tensor, field: Tensor | None
                  ) -> tuple[Tensor, Tensor]:
        """``(dtheta (B, N, 22), dglob (B, 12))`` -- exactly 0 at step 0."""
        memory = self.build_memory(z, field)
        q = self.queries(memory.shape[0], ref=memory)
        for layer in self.layers:
            q = layer(q, memory)
        return self.head_g(q[:, : self.n_gauss]), self.head_a(q[:, self.n_gauss])

    def forward(self, z: Tensor, field: Tensor | None = None) -> GlutParams:  # noqa: D102
        dtheta, dglob = self.residuals(z, field)
        return self.theta_base.compose(dtheta, dglob)

    def base_params(self, batch: int = 1) -> GlutParams:
        """``theta_base`` with no residual -- the step-0 / identity reference."""
        return self.theta_base.compose(None, None, batch=batch)


def param_count_breakdown(model: nn.Module) -> dict[str, Any]:
    """Per-block parameter counts; the total is pinned by the proposal's table.

    ``EPR-029:442-451``: d=256/L=4/K=4 -> **5,833,038**; d=128/L=2/K=4 ->
    **1,736,654**; d=256/L=4/K=1 -> **3,866,190**.  A structural slip (a missing
    LayerNorm, a bias, an extra projection) shows up here before it shows up in
    a headline.
    """
    groups: dict[str, int] = {}
    for name, p in model.named_parameters():
        head = name.split(".")[0]
        groups[head] = groups.get(head, 0) + int(p.numel())
    return {"n_params": int(sum(groups.values())),
            "n_params_by_block": dict(sorted(groups.items()))}


# --------------------------------------------------------------------------- #
# 5. the arm: decoder + carrier
# --------------------------------------------------------------------------- #
class QDualArm(nn.Module):
    """The trainable object: :class:`GaussianQueryDecoder` + :class:`GlutCarrier`.

    The carrier is imported, never re-implemented (frozen block: one GLUT forward
    for the six arms), and it opts out of autocast internally, so a bf16 training
    step still evaluates Eq.1-2 in fp32.
    """

    def __init__(self, cfg: QDualConfig | None = None, **kw: Any) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else QDualConfig(**kw)
        self.decoder = GaussianQueryDecoder(self.cfg)
        self.carrier = GlutCarrier(clamp=self.cfg.clamp, residual=self.cfg.residual,
                                   eps=self.cfg.eps)

    @property
    def config(self) -> dict[str, Any]:
        return {**self.decoder.config, "carrier": self.carrier.config}

    def param_groups(self, base_lr: float, *, geometry_lr_scale: float = 0.1):
        return self.decoder.param_groups(base_lr,
                                         geometry_lr_scale=geometry_lr_scale)

    def forward(self, colors: Tensor, z: Tensor, field: Tensor | None = None, *,
                clamp: str | None = None, return_aux: bool = False
                ) -> Tensor | tuple[Tensor, GlutAux]:
        """``f_theta(x)`` on ``colors`` ``(B, Q, 3)`` (or ``(Q, 3)``, shared)."""
        params = self.decoder(z, field)
        return self.carrier(colors, params, clamp=clamp, return_aux=return_aux)

    @torch.no_grad()
    def transform_grid(self, grid: Tensor, z: Tensor, field: Tensor | None = None
                       ) -> Tensor:
        """``(B, P, 3)`` function values on a shared query grid ``(P, 3)``."""
        return self.forward(grid, z, field)

    @torch.no_grad()
    def apply_image(self, img: Tensor, z: Tensor, field: Tensor | None = None,
                    *, point_chunk: int | None = None) -> Tensor:
        """``f_hat(I)`` for one ``(3, H, W)`` image, evaluated on its own device."""
        if img.dim() != 3 or img.shape[0] != 3:
            raise ValueError(f"img must be (3, H, W), got {tuple(img.shape)}")
        params = self.decoder(z, field)
        out = self.carrier(img.permute(1, 2, 0).unsqueeze(0), params,
                           point_chunk=point_chunk)
        return out[0].permute(2, 0, 1)


# --------------------------------------------------------------------------- #
# 6. losses -- GLUT Eq.6-8, transcribed, nothing added (proposal section 3.4)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LossOutput:
    """The loss and the columns the first ``steps.jsonl`` row must carry."""

    total: Tensor
    l_rec: Tensor
    l_hc: Tensor
    r_sparse: Tensor
    n_hc_masked: int
    n_colors: int
    columns: dict[str, Any] = _field(default_factory=dict)


def rec_loss(y_hat: Tensor, y: Tensor) -> Tensor:
    """GLUT Eq.6 ``L_rec = || y_hat - y ||_1`` (mean over colours and channels)."""
    if y_hat.shape != y.shape:
        raise ValueError(f"shape mismatch {tuple(y_hat.shape)} vs {tuple(y.shape)}")
    return (y_hat - y.to(device=y_hat.device, dtype=y_hat.dtype)).abs().mean()


#: Kept as a recorded number, no longer applied.  EPR-029 floored sRGB at 1e-6
#: on the way into CIELab because ``colorimetry.xyz_to_lab``'s cube-root branch
#: had a ``0 * inf = NaN`` gradient at pure black.  That defect is now fixed **in
#: the shared function** (``xyz_to_lab`` clamps the cube-root base to the branch
#: boundary; the forward is bit-identical), so the private floor -- and the
#: forward deviation it caused -- is gone and all six arms convert colour the
#: same way.  Review blocker B1: the same defect was killing the other five arms,
#: which had no floor.
LAB_FLOOR: float = 1e-6


def hue_chroma_loss(y_hat: Tensor, y: Tensor, *, eps_c: float = EPS_CHROMA,
                    lab_floor: float = LAB_FLOOR) -> tuple[Tensor, int]:
    """GLUT Eq.7 ``L_hc = C (1 - <h_hat, h>)`` with the frozen ``C -> 0`` handling.

    Frozen block: ``h = (a, b) / max(C, eps_C)`` **and** the whole term is
    multiplied by the hard mask ``1[C >= eps_C]`` with ``eps_C = 1e-3``; the
    number of masked points is returned so the trainer can log ``n_hc_masked``.
    ``C`` is the **target**'s chroma (Eq.7 weights by the target).

    CIELab comes from :mod:`q3vl.whatb.colorimetry` -- one implementation for the
    loss and the criteria, so the two cannot drift.  ``C``, ``h`` and the mask
    are that module's :func:`chroma_hue` verbatim.

    **The two ``0 * inf = NaN`` points EPR-029 found are fixed in the shared
    module now**, not here (review blocker B1 -- the other five arms consume the
    same two functions and were dying on them):

    * ``xyz_to_lab``'s cube-root branch at ``t == 0`` (pure black is a legal
      training colour and a legal clamped prediction);
    * ``C = sqrt(a^2 + b^2)`` at ``a = b = 0`` (an exactly neutral prediction --
      a clamp to black or white produces one).

    Both are now handled inside :mod:`q3vl.whatb.colorimetry` with a
    **bit-identical forward**, so this arm no longer carries a floor or a second
    hue spelling and the frozen ``h = (a, b) / max(C, eps_C)``口径 holds for all
    six arms.  ``lab_floor`` is accepted and recorded for the artefact, and is
    not applied.
    """
    y = y.to(device=y_hat.device, dtype=y_hat.dtype)
    lab_t = srgb_to_lab(y).detach()
    lab_p = srgb_to_lab(y_hat)
    c, h, valid = chroma_hue(lab_t, eps_c)
    _, h_hat, _ = chroma_hue(lab_p, eps_c)
    term = c * (1.0 - (h_hat * h).sum(dim=-1))
    mask = valid.to(term.dtype)
    n_valid = mask.sum()
    loss = (term * mask).sum() / n_valid.clamp_min(1.0)
    n_masked = int(valid.numel() - int(n_valid))
    return loss, n_masked


def sparse_regulariser(opacity: Tensor, *, eps: float = EPS) -> Tensor:
    """GLUT Eq.8 ``R_sparse``: the binary entropy of the opacities, mean over batch."""
    o = opacity
    ent = o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)
    return -(ent.mean())


def qdual_losses(y_hat: Tensor, y: Tensor, aux: GlutAux, *,
                 lambda_hc: float = LAMBDA_HC,
                 lambda_sparse: float = LAMBDA_SPARSE,
                 eps_c: float = EPS_CHROMA, eps: float = EPS,
                 extra: Mapping[str, Any] | None = None) -> LossOutput:
    """``L_total = L_rec + 10 L_hc + 0.001 R_sparse`` (GLUT section 4.1).

    No term is added: 3D/4D TV, monotonicity, interpolation consistency and
    adversarial terms belong to other EPRs' variables (proposal :577).
    """
    l_rec = rec_loss(y_hat, y)
    l_hc, n_masked = hue_chroma_loss(y_hat, y, eps_c=eps_c)
    r_sparse = sparse_regulariser(aux.opacity, eps=eps)
    total = l_rec + float(lambda_hc) * l_hc + float(lambda_sparse) * r_sparse
    n_colors = int(y_hat.shape[0] * y_hat.shape[1]) if y_hat.dim() == 3 \
        else int(y_hat.numel() // 3)
    cols: dict[str, Any] = {
        "L_rec": float(l_rec.detach()),
        "L_hc": float(l_hc.detach()),
        # the frozen first-row column is spelled ``L_sparse``
        # (guards.FIRST_STEP_COLUMNS); the proposal's prose calls the same
        # quantity ``R_sparse``.  Both keys are written, same value.
        "L_sparse": float(r_sparse.detach()),
        "R_sparse": float(r_sparse.detach()),
        "L_total": float(total.detach()),
        "n_hc_masked": int(n_masked),
        "n_colors": int(n_colors),
        "lambda_hc": float(lambda_hc), "lambda_sparse": float(lambda_sparse),
        "eps_c": float(eps_c),
        "lab_floor": None,          # see LAB_FLOOR: recorded, not applied
    }
    if extra:
        cols.update(dict(extra))
    return LossOutput(total=total, l_rec=l_rec, l_hc=l_hc, r_sparse=r_sparse,
                      n_hc_masked=n_masked, n_colors=n_colors, columns=cols)


# --------------------------------------------------------------------------- #
# 7. hard-example mining (ruling 11.1-4)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def mine_hard_colors(model: QDualArm, z: Tensor, field: Tensor | None,
                     probe: Tensor, fresh: Tensor, targets: Tensor,
                     ratio: float) -> tuple[Tensor, dict[str, Any]]:
    """Within-batch top-r resampling; no cross-step state (ruling 11.1-4).

    ``probe`` and ``fresh`` are ``(B, Q, 3)`` uniform draws and ``targets`` is
    ``L_l(probe)``; the returned batch keeps ``Q`` colours per sample (so the
    frozen ``B x Q = 8192`` organisation survives): the ``r*Q`` worst-L1 probe
    colours plus the first ``(1-r)*Q`` fresh ones.

    The ``topk`` runs on the colours' own device: CPU and CUDA break ties
    differently, and mining is exactly a top-k.
    """
    b, q, _ = probe.shape
    k = int(round(float(ratio) * q))
    k = max(0, min(k, q))
    if k == 0:
        return fresh, {"mining_ratio": float(ratio), "n_mined": 0,
                       "n_fresh": int(q)}
    y_hat = model(probe, z, field)
    err = (y_hat - targets.to(device=y_hat.device, dtype=y_hat.dtype)) \
        .abs().mean(dim=-1)                                       # (B, Q)
    idx = torch.topk(err, k, dim=1, largest=True, sorted=False).indices
    hard = torch.gather(probe, 1, idx.unsqueeze(-1).expand(b, k, 3))
    out = torch.cat([hard, fresh[:, : q - k]], dim=1)
    return out, {"mining_ratio": float(ratio), "n_mined": int(k),
                 "n_fresh": int(q - k)}


def mining_ratio_for_step(step: int, steps_per_epoch: int) -> float:
    """``r`` at a global step -- GLUT App A.1's epoch 5->20, 10%->40% ramp.

    Step-shaped rather than epoch-shaped (the campaign's step budget is the U4
    unit), which is the "步数保形" NOVEL adaptation of proposal :590.
    """
    return mining_ratio(float(step) / max(1, int(steps_per_epoch)))


# --------------------------------------------------------------------------- #
# 8. step-0 witness and the two hard assertions
# --------------------------------------------------------------------------- #
@torch.no_grad()
def zero_init_witness(model: QDualArm, z: Tensor, field: Tensor | None,
                      grid: Tensor) -> dict[str, Any]:
    """``zeroinit_step0_maxabs`` and ``step0_maxabs_f_minus_id``.

    The first is ``max |dtheta|`` over both heads and must be **exactly** 0 while
    the heads are zero-initialised -- that is the arm's strongest wiring evidence
    (proposal :639).  The second is ``max |f_theta(x) - x|`` on the query grid;
    it is *not* zero, and cannot be: with ``eps = 1e-6`` proposition 2 gives
    ``f(x) = (1 - delta(x)) x`` with ``delta`` up to ~1.1e-7, so this column is
    recorded as a witness with the measured value rather than asserted to 0.
    """
    dtheta, dglob = model.decoder.residuals(z, field)
    y = model(grid, z, field)
    x = grid.to(device=y.device, dtype=y.dtype)
    if x.dim() == 2:
        x = x.unsqueeze(0).expand_as(y)
    return {
        "zeroinit_step0_maxabs": float(torch.maximum(
            dtheta.abs().max(), dglob.abs().max())),
        "step0_maxabs_f_minus_id": float((y - x).abs().max()),
        "n_queries": int(grid.shape[-2]),
    }


def assert_zero_init(witness: Mapping[str, Any], *, enabled: bool = True) -> None:
    """``zeroinit_step0_maxabs != 0`` is an ``AssertionError`` (proposal :637-⑧).

    ``enabled=False`` is the ``--no-zero-init-head`` ablation row, which drops
    both this assertion and the ``zeroinit_step0_maxabs`` entry of the required
    table (``EPR-029:772-773``); the board says so in its own note.
    """
    if not enabled:
        return
    v = float(witness.get("zeroinit_step0_maxabs", float("nan")))
    if v != 0.0:
        raise AssertionError(
            f"zeroinit_step0_maxabs = {v!r}, expected exactly 0: the output "
            "heads are declared zero-initialised, so step 0 must have "
            "dtheta = 0 and theta = theta_base.  A non-zero value means the "
            "zero-init did not reach the head that is actually being used.")


def assert_not_degenerate(y: Tensor, x: Tensor, *,
                          thresholds: DegeneracyThresholds | None = None,
                          where: str = "quick_eval",
                          exit_process: bool = True,
                          extra: Mapping[str, Any] | None = None
                          ) -> DegeneracyReport:
    """The first-quick-eval guard (three conditions), wired to ``SystemExit(2)``.

    ``y`` is ``(B, P, 3)`` = the arm's transform on ``B`` **different** samples'
    conditions over ``P`` query colours, ``x`` the query colours.  Flat across
    colours / equal to the identity / equal across samples -- any one of them
    stops the run with the three measured numbers printed next to their floors.
    The thresholds travel into ``run_setup.json``.
    """
    return assert_transform_not_degenerate(
        y, x, thresholds=thresholds or DegeneracyThresholds(), where=where,
        exit_process=exit_process, extra=extra)


# --------------------------------------------------------------------------- #
# 9. this arm's own diagnostics: query -> Gaussian correspondence (section 3.6)
# --------------------------------------------------------------------------- #
def hungarian_match(mu_a: Tensor, mu_b: Tensor) -> tuple[Tensor, Tensor]:
    """``(assignment, cost)`` minimising ``sum_i ||mu_a[i] - mu_b[match(i)]||_2``.

    ``mu_*`` are ``(N, 3)``.  The cost matrix is built on the tensors' device;
    only the (tiny) matrix crosses to SciPy for the assignment itself, which has
    no GPU implementation -- the *metric* is not computed after a ``.cpu()``, the
    combinatorial solve is.
    """
    if mu_a.shape != mu_b.shape or mu_a.dim() != 2 or mu_a.shape[-1] != 3:
        raise ValueError(
            f"means must be two matching (N, 3) tensors, got {tuple(mu_a.shape)} "
            f"and {tuple(mu_b.shape)}")
    cost = torch.cdist(mu_a.unsqueeze(0).float(), mu_b.unsqueeze(0).float())[0]
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    assign = torch.as_tensor(cols, device=mu_a.device, dtype=torch.long)
    return assign, cost


def _drift_and_shift(mu_ref: Tensor, mu_new: Tensor) -> tuple[float, float, float]:
    """``(drift, max shift by index, max shift after matching)`` for one pair."""
    assign, cost = hungarian_match(mu_ref, mu_new)
    n = mu_ref.shape[0]
    ident = torch.arange(n, device=assign.device)
    drift = float((assign != ident).to(torch.float32).mean())
    shift_idx = float((mu_new - mu_ref).norm(dim=-1).max())
    shift_match = float(cost[ident, assign].max())
    return drift, shift_idx, shift_match


@torch.no_grad()
def query_match_repeat(model: QDualArm, z: Tensor, field: Tensor | None, *,
                       repeats: int = 8) -> dict[str, Any]:
    """Protocol 1 of section 3.6: the same condition forwarded ``R`` times.

    Deterministic in fp32 (dropout is 0 everywhere in this arm), so on CPU the
    drift is structurally 0; under bf16 reductions on GPU it need not be, which
    is the reason the column exists at all.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    ref = model.decoder(z, field).mu[0]
    drifts: list[float] = []
    shifts_idx: list[float] = []
    shifts_match: list[float] = []
    for _ in range(int(repeats) - 1):
        mu = model.decoder(z, field).mu[0]
        d, si, sm = _drift_and_shift(ref, mu)
        drifts.append(d)
        shifts_idx.append(si)
        shifts_match.append(sm)
    return {"query_match_drift_repeat": max(drifts) if drifts else 0.0,
            "query_match_mu_shift_indexed_repeat": max(shifts_idx) if shifts_idx else 0.0,
            "query_match_mu_shift_hungarian_repeat": max(shifts_match) if shifts_match else 0.0,
            "repeats": int(repeats)}


@torch.no_grad()
def query_match_path(model: QDualArm, z_a: Tensor, z_b: Tensor,
                     field: Tensor | None, *, k_steps: int = 20) -> dict[str, Any]:
    """Protocol 2 of section 3.6: drift along ``z_alpha = (1-a) z_a + a z_b``.

    ``K = 20`` is the same grid the interpolation criterion IP-B uses.  Both
    displacement columns are returned: by index and after matching -- one alone
    cannot separate "the Gaussian moved" from "the index moved".
    """
    if z_a.dim() == 1:
        z_a = z_a.unsqueeze(0)
    if z_b.dim() == 1:
        z_b = z_b.unsqueeze(0)
    mus: list[Tensor] = []
    for k in range(int(k_steps) + 1):
        a = k / float(k_steps)
        z = (1.0 - a) * z_a + a * z_b
        mus.append(model.decoder(z, field).mu[0])
    drifts, shifts_idx, shifts_match = [], [], []
    for i in range(len(mus) - 1):
        d, si, sm = _drift_and_shift(mus[i], mus[i + 1])
        drifts.append(d)
        shifts_idx.append(si)
        shifts_match.append(sm)
    return {"query_match_drift_path": float(sum(drifts)),
            "query_match_drift_path_step_max": max(drifts) if drifts else 0.0,
            "query_match_mu_shift_indexed": max(shifts_idx) if shifts_idx else 0.0,
            "query_match_mu_shift_hungarian": max(shifts_match) if shifts_match else 0.0,
            "k_steps": int(k_steps)}


# --------------------------------------------------------------------------- #
# 10. collapse guard (section 3.8, Neural Preset Fig.8 form)
# --------------------------------------------------------------------------- #
def collapse_guard(delta_const: float | None, m_const: float | None, *,
                   m_threshold: float = COLLAPSE_M_THRESHOLD,
                   delta_threshold: float = COLLAPSE_DELTA_THRESHOLD,
                   label: str = "N3_const") -> dict[str, Any]:
    """``M_const < 0.5`` **and** ``|Delta_const| < 0.1`` -> ``COLLAPSED = true``.

    Both thresholds are **NOVEL** (NOTES 7): they are ~1.2% and ~0.25% of the
    measured scale "GT LUT vs identity on the 17^3 grid, mean dE76 = 40.3", and
    have no external source.  The verdict is recorded on the board either way --
    a collapsed run is counted and printed, never silently dropped -- and a
    collapsed run may not carry a paired-delta claim.
    """
    if delta_const is None or m_const is None:
        return {"COLLAPSED": None, "reason": "delta_const / M_const not computed",
                "m_threshold": m_threshold, "delta_threshold": delta_threshold,
                "label": label}
    collapsed = (float(m_const) < m_threshold
                 and abs(float(delta_const)) < delta_threshold)
    return {"COLLAPSED": bool(collapsed),
            "M_const": float(m_const), "delta_const": float(delta_const),
            "m_threshold": float(m_threshold),
            "delta_threshold": float(delta_threshold),
            "units": "dE00", "label": label,
            "note": ("thresholds are NOVEL (EPR-029 NOTES 7); a COLLAPSED run is "
                     "reported and excluded from paired-delta claims")}


# --------------------------------------------------------------------------- #
# 11. the pre-registration record
# --------------------------------------------------------------------------- #
def loss_preregistration(cfg: QDualConfig, *, loss_level: int = 3,
                         lambda_hc: float | None = None,
                         lambda_sparse: float | None = None) -> dict[str, Any]:
    """``config/loss_preregistration.json`` for this arm.

    States the loss form and its constants with their sources, so a run that
    quietly changed a weight cannot publish under the pre-registered name.

    ``lambda_hc`` / ``lambda_sparse`` default to the ladder's own values for
    ``loss_level`` (``arms/carrier.py:347/:351`` through
    :mod:`q3vl.whatb.caliber`): at level 3 they are 10 / 0.001, at level 1 both
    are 0 and the run is on EPR-030's pure-L1 caliber.
    """
    lam_hc = (effective_lambda_hc(LAMBDA_HC, loss_level) if lambda_hc is None
              else float(lambda_hc))
    lam_sparse = (effective_lambda_sparse(LAMBDA_SPARSE, loss_level)
                  if lambda_sparse is None else float(lambda_sparse))
    return {
        "arm": ARM, "epr": ARM_EPR, "axes": list(ARM_AXES),
        "loss_level": int(loss_level),
        "loss_ladder": pure_l1_record(loss_level=loss_level, lambda_hc=lam_hc,
                                      lambda_sparse=lam_sparse),
        "form": "L_total = L_rec + lambda_hc * L_hc + lambda_sparse * R_sparse",
        "terms": {
            "L_rec": {"definition": "|| f_theta(x) - L_l(x) ||_1",
                      "source": "GLUT Eq.6", "space": "function value"},
            "L_hc": {"definition": "C * (1 - <h_hat, h>), C = sqrt(a^2 + b^2) of "
                                   "the target",
                     "source": "GLUT Eq.7",
                     "hue_denominator": ("(a,b)/max(C, eps_C) on both sides -- "
                                         "the frozen block spelling, shared with "
                                         "the other five arms.  The infinite "
                                         "d sqrt/dx at an exactly neutral colour "
                                         "is removed inside colorimetry with a "
                                         "bit-identical forward, so this is no "
                                         "longer a deviation"),
                     "lab_floor": ("not applied: colorimetry.xyz_to_lab is "
                                   "gradient-safe at pure black by construction "
                                   "(the cube-root base is clamped to the branch "
                                   "boundary, forward unchanged)"),
                     "c_to_zero": (f"h = (a,b)/max(C, {EPS_CHROMA}) AND a hard "
                                   f"mask 1[C >= {EPS_CHROMA}] (frozen block; "
                                   "NOVEL numeric), masked count logged as "
                                   "n_hc_masked")},
            "R_sparse": {"definition": "-(1/N) sum_i [o log(o+eps) + "
                                       "(1-o) log(1-o+eps)]",
                         "source": "GLUT Eq.8", "eps": EPS},
        },
        "lambda_hc": lam_hc, "lambda_sparse": lam_sparse,
        "lambda_hc_declared": LAMBDA_HC, "lambda_sparse_declared": LAMBDA_SPARSE,
        "lambda_source": "GLUT section 4.1 (10 / 0.001), gated by --loss-level",
        "added_terms": [],
        "added_terms_note": ("this arm adds no loss term: TV / monotonicity / "
                             "interpolation consistency / Jacobian / adversarial "
                             "belong to D7 / D12 and to other EPRs (:577)"),
        "supervision_space": "function value (image space is evaluation only)",
        "target": "L_l(x) -- independent of alpha in family F1 under uniform x "
                  "(proposal :549-552)",
        "clamp": cfg.clamp, "residual": cfg.residual,
        "colour_sampling": cfg.color_sampling,
        "hard_mining": {"source": "GLUT App A.1", "epochs": [5, 20],
                        "ratio": [0.10, 0.40],
                        "granularity": "within-batch top-r resampling, no "
                                       "cross-step state (ruling 11.1-4), applied "
                                       "per sample so B x Q stays 32 x 256"},
        "criteria_required": list(REQUIRED_QDUAL if cfg.zero_init_head
                                  else REQUIRED_QDUAL_NO_ZEROINIT),
        "collapse_guard": {"m_threshold": COLLAPSE_M_THRESHOLD,
                           "delta_threshold": COLLAPSE_DELTA_THRESHOLD,
                           "novel": True, "notes": "EPR-029 NOTES 7"},
    }


def required_criteria_table(cfg: QDualConfig) -> list[str]:
    """The list handed to ``assert_criteria_ran`` / ``assert_publishable``."""
    return list(REQUIRED_QDUAL if cfg.zero_init_head
                else REQUIRED_QDUAL_NO_ZEROINIT)
