"""EPR-028 R1 G4D: conditional RGB-s parameterisation on the GLUT carrier.

Spec: ``experiments/prs/EPR-028_4d-gaussian-conditional-slice/PROPOSAL_R1.md``
(R1, 2026-08-16) -- **the only implementation authority**.  ``PROPOSAL.md`` (R0)
is superseded except for its cross-arm frozen block, its §1.2 data counts and
its §4 external-fact verification record.

What R1 changed against R0 (and therefore against the previous revision of this
module)
-----------------------------------------------------------------------------
* **carrier** (§4.1): the 4D primitive is no longer ``scale4 + two quaternions``
  sliced with a run-time Schur complement.  Each primitive now *directly* emits
  the conditional quantities -- ``L_C`` (3x3 lower Cholesky factor of the
  conditional colour covariance, ``diag`` through ``softplus``), the regression
  slope ``beta in R^3``, the ``s``-axis scale ``tau > 0``, the means
  ``(mu_x in R^3, mu_s in R)`` -- so the conditional law is *read off*::

      mu_{x|s} = mu_x + beta (s - mu_s)
      Sigma_{x|s} = C = L_C L_C^T

  The implied joint 4D covariance is ``Sigma = [[C + tau^2 beta beta^T,
  tau^2 beta], [tau^2 beta^T, tau^2]]`` (:func:`sigma4_from_cond`), and every
  SPD 4D ``Sigma`` maps back onto ``(C, beta, tau^2)``
  (:func:`cond_from_sigma4`) -- so the expressivity is unchanged and the run
  time carries no Schur subtraction, no division by a possibly-tiny
  ``Sigma_44``, no quaternion normalisation and no ``q``/``-q`` gauge.
* **``s`` gate** (§4.2): peak-normalised ``log g_i(s) = -0.5 ((s-mu_s)/tau)^2``
  is the default and the only main-arm setting; ``tau = 0.1 + 0.9 sigmoid(r)``
  is bounded.  The fully normalised marginal (``- log tau - 0.5 log 2pi``) is a
  flag-only appendix row (``--marg-norm full``).  4DGS's own implementation
  comments the ``/sqrt(2 pi sigma)`` out (``gaussian_model.py:242``; verified in
  the R0 session, record retained there).
* **forward** (§4.3): genuinely in the log domain.  ``exp(logp)`` is gone; the
  mixture is normalised through ``logsumexp`` + ``logaddexp(., log eps)``, which
  keeps GLUT Eq.2's ``+ eps`` exactly and produces ``null_mass = eps / (sum_i
  a_i + eps)`` as a first-class diagnostic.  Mahalanobis goes through a
  triangular solve; **no explicit inverse, no determinant**.
* **arms** (§3): ``A0`` (3D GLUT, 22/prim), ``A1`` (3D GLUT + explicit ``s``
  gate, 22/prim), ``A2`` (conditional forward with ``beta`` forced to zero *in
  the forward*, 27/prim), ``A3`` (conditional forward with ``beta`` live,
  27/prim).  ``A2`` and ``A3`` share generator structure, initialisation and
  parameter count exactly, so the paper question is the single difference
  ``A3 - A2``.  ``A4`` (R0's double-quaternion + Schur appendix arm) keeps its
  name in :data:`G4D_MODES` and raises :class:`NotImplementedError` on
  construction -- R1 §11-2 leaves it undecided until ``A3`` clears G2.
* **losses** (§4.4): ``L_m4d`` is **deleted** (the generating law has
  ``d y*_c / d s = L_c(x) - x_c``, which is legitimately negative, so a penalty
  on negative differences penalises the correct target);
  :func:`r_line` replaces it.
* **image formation** (§5): per-arm, no shared outer compositor.  ``A0`` is
  ``I + S (T(I) - I)``; ``A1``/``A2``/``A3`` are ``f(I, S)``.  R0's outer
  ``mix_alpha`` on top of a carrier that already ate ``s`` was the double-alpha
  bug (R1 §0.1).
* **numerics** (§7): the conditional geometry, the Mahalanobis form, the log
  determinant, the mixture weights and RGB->Lab all run at >= FP32.
  ``cholesky_ex.info != 0`` / a non-positive or non-finite ``L_C`` diagonal /
  any non-finite forward value **raises**.  There is no identity fallback and
  no determinant floor, so R0's ``slice_det_fallback`` /
  ``sigma44_floor_hits`` / ``schur_pd_violations`` / ``logdet_floor_hits``
  counters are gone with the code paths they counted.

Discipline carried over unchanged
---------------------------------
1. the board-time step-row assertion fetches its own row (:func:`assert_step_row`);
2. every tensor entering the maths is moved onto one resolved
   ``(device, dtype)``; constants live in ``register_buffer(persistent=False)``
   and no ``torch.tensor(...)`` appears in a forward;
3. the first quick eval calls :func:`assert_not_degenerate`;
4. nothing here calls ``.cpu()``; every metric reduction stays on the device.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from q3vl.whatb.caliber import (
    effective_lambda_hc,
    effective_lambda_sparse,
    pure_l1_record,
)
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.criteria import ARM_AXES as _CRITERIA_ARM_AXES
from q3vl.whatb.criteria import required_criteria as _base_required_criteria
from q3vl.whatb.generator import SEG_COLOR_HIDDEN_DIM, SegColorProjection
from q3vl.whatb.generator import _mlp as _app_a2_mlp  # App A.2 head shape, one copy
from q3vl.whatb.glut import (
    EPS,
    LOG_2PI,
    GlutParams,
    glut_forward,
    softplus_inverse,
    uniform_grid_positions,
)
from q3vl.whatb.glut import _no_autocast  # autocast opt-out, one copy
from q3vl.whatb.guards import (
    DegeneracyReport,
    DegeneracyThresholds,
    assert_first_step_columns,
    assert_transform_not_degenerate,
)
from q3vl.whatb.lutdata import mix_alpha

__all__ = [
    "ARM",
    "ARM_AXES",
    "G4D_MODES",
    "CONDITIONAL_MODES",
    "MARG_NORM_CHOICES",
    "FIELD_CHOICES",
    "TAU_MIN",
    "TAU_MAX",
    "TAU_INIT",
    "TAU_RAW_INIT",
    "WEIGHT_UNDERFLOW_TAU",
    "EPS_CHROMA",
    "LAMBDA_HC",
    "LAMBDA_SPARSE",
    "LAMBDA_LINE",
    "OPACITY_LOGIT_INIT",
    "SIGMA_RGB_INIT",
    "MU_S_INIT_RANGE",
    "GUARD_COLUMNS",
    "STEP_EXTRA_COLUMNS",
    "OFF_BY_DEFAULT_LOSS_COLUMNS",
    "ARM_EXTRA_CRITERIA",
    "S_AXIS_GRID",
    "G4DNumericalError",
    "G4DConfig",
    "G4DParams",
    "G4DAux",
    "G4DGenerator",
    "Glut4DCarrier",
    "G4DArm",
    "sigma4_from_cond",
    "cond_from_sigma4",
    "cholesky_lower",
    "lower_from_raw",
    "tau_from_raw",
    "raw_from_tau",
    "conditional_params",
    "log_gauss3",
    "mixture_log_weights",
    "n_params_g4d",
    "compose_headline",
    "l_rec",
    "l_hc",
    "r_sparse",
    "r_line",
    "l_s4d",
    "l_img",
    "total_loss",
    "target_4d",
    "assert_not_degenerate",
    "assert_step_row",
    "step_columns",
    "step_extra_columns",
    "required_criteria",
    "loss_preregistration",
    "grid_4d",
]

# --------------------------------------------------------------------------- #
# 0. frozen constants
# --------------------------------------------------------------------------- #
ARM: str = "EPR-028"
#: P2 (six-family table, F3b).  Kept next to the shared table so a drift is loud.
ARM_AXES: tuple[str, ...] = _CRITERIA_ARM_AXES[ARM]

#: R1 §3.  ``A4`` keeps its name (so a board that names it is not silently
#: renamed) but :class:`G4DConfig` refuses to build it -- R1 §11-2.
G4D_MODES: tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4")
#: the two arms that run the R1 §2.1 conditional forward.
CONDITIONAL_MODES: tuple[str, ...] = ("A2", "A3")
#: ``--marg-norm``: R1 §4.2.  ``peak`` is the default and the only main-arm value.
MARG_NORM_CHOICES: tuple[str, ...] = ("peak", "full")
#: ``--glut4d-field``: which spatial field the carrier eats at evaluation time.
FIELD_CHOICES: tuple[str, ...] = ("gt", "pred", "const", "shuffle")

#: R1 §4.2 ``tau = tau_min + (tau_max - tau_min) sigmoid(r)`` -- bounded, so a
#: ``tau`` collapse cannot run away into a delta gate.
TAU_MIN: float = 0.1
TAU_MAX: float = 1.0
#: R1 §8.4: 4DGS ``dist_t = (t_max - t_min)/5 = 0.2`` on ``[0,1]`` -> ``tau = sqrt(0.2)``.
TAU_INIT: float = math.sqrt(0.2)
#: R0's weight-underflow counter.  Retained (R1 §4.3 "保留但不再是主诊断"); the
#: main diagnostic is ``null_mass``.
WEIGHT_UNDERFLOW_TAU: float = 1e-12
#: frozen block: ``L_hc`` hard mask ``1[C >= eps_C]``, ``eps_C = 1e-3``.
EPS_CHROMA: float = 1e-3
#: GLUT §4.1
LAMBDA_HC: float = 10.0
LAMBDA_SPARSE: float = 0.001
#: R1 §4.4, pre-registered (ablation row: 0 / 0.1 / 1.0).
LAMBDA_LINE: float = 0.1

#: R1 §8.4: ``logit(0.5) = 0``.  R0's ``logit(0.99)`` was already near
#: saturation and pushed the same way as the entropy regulariser.
OPACITY_LOGIT_INIT: float = 0.0
#: R1 §8.4 / GLUT App A.1: isotropic conditional covariance, ``L_C`` diag = 0.15.
SIGMA_RGB_INIT: float = 0.15
#: 4DGS ``fused_times = (rand*1.2 - 0.1)*(t_max - t_min) + t_min`` -> U(-0.1, 1.1).
MU_S_INIT_RANGE: tuple[float, float] = (-0.1, 1.1)


def raw_from_tau(tau: float, *, tau_min: float = TAU_MIN,
                 tau_max: float = TAU_MAX) -> float:
    """``r`` such that ``tau_min + (tau_max - tau_min) sigmoid(r) == tau``."""
    if not (tau_min < float(tau) < tau_max):
        raise ValueError(f"tau must lie in ({tau_min}, {tau_max}); got {tau!r}")
    p = (float(tau) - tau_min) / (tau_max - tau_min)
    return float(math.log(p / (1.0 - p)))


#: R1 §8.4: ``r`` initialised so ``tau ~= sqrt(0.2) = 0.4472136``.
TAU_RAW_INIT: float = raw_from_tau(TAU_INIT)

#: R1 §7 killed R0's four fallback counters with the code paths they counted.
#: What is left is the hard error counter (normally identically 0, because a
#: non-zero value raises) plus R0's retained weight-underflow fraction.
GUARD_COLUMNS: tuple[str, ...] = ("cholesky_info_nonzero", "weight_underflow_frac")

#: R1 §10, the maximal set (``A3``).  :func:`step_extra_columns` narrows it per
#: mode: ``beta_absmean`` only exists on ``A3``, ``tau_*`` only on ``A2``/``A3``.
STEP_EXTRA_COLUMNS: tuple[str, ...] = (
    "null_mass_mean", "null_mass_p99", "cholesky_info_nonzero",
    "tau_p05", "tau_p50", "tau_p95", "beta_absmean",
    "opacity_p05", "opacity_p50", "opacity_p95", "n_pairs_s", "gnorm",
)

#: R1 §4.4: ``L_m4d`` is deleted, so only two optional terms remain.  These keys
#: must be ABSENT from the main arm's first row -- "off but still computed" is
#: asserted on both sides.
OFF_BY_DEFAULT_LOSS_COLUMNS: tuple[str, ...] = ("L_s4d", "L_img")

#: this arm's additions on top of the twelve frozen keys and the P2 block.
ARM_EXTRA_CRITERIA: tuple[str, ...] = (
    "field_pred",
    "grid_s0", "grid_s25", "grid_s50", "grid_s75", "grid_s100",
    "headline_alpha_inside_only",
)
#: the ``s`` values of the 4D extension column and their key names.
S_AXIS_GRID: tuple[tuple[float, str], ...] = (
    (0.0, "grid_s0"), (0.25, "grid_s25"), (0.5, "grid_s50"),
    (0.75, "grid_s75"), (1.0, "grid_s100"),
)

#: R1 §3's parameter table.  ``A0``/``A1`` = GLUT's 22 (mu 3 + chol 6 + o 1 +
#: M 9 + b 3); ``A2``/``A3`` = 27 (mu_x 3 + L_C 6 + beta 3 + mu_s 1 + tau_r 1 +
#: o 1 + M 9 + b 3); ``A4`` = R0's 29.  Plus the global affine ``G, g`` = 12.
_MODE_PER_PRIMITIVE: dict[str, int] = {
    "A0": 22, "A1": 22, "A2": 27, "A3": 27, "A4": 29,
}


def n_params_g4d(n_gauss: int, mode: str = "A3") -> int:
    """Generated dimension of ``Theta`` (R1 §3).

    ``A0``/``A1`` N=48 -> **1068**; ``A2``/``A3`` N=48 -> **1308**;
    ``A3`` N=39 -> **1065** (the generation-dimension-matched control row).
    """
    if mode not in G4D_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {G4D_MODES}")
    return _MODE_PER_PRIMITIVE[mode] * int(n_gauss) + 12


def step_extra_columns(mode: str) -> tuple[str, ...]:
    """R1 §10's new ``steps.jsonl`` columns for ``mode``.

    ``beta_absmean`` exists only on ``A3`` (it is the direct read-out of the
    paper question and would be a constant zero elsewhere); ``tau_*`` exist only
    on the two conditional arms.  ``gnorm`` is produced by the training loop,
    not by the carrier, and is listed here because the first-row assertion is
    what makes a missing column loud.
    """
    if mode not in G4D_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {G4D_MODES}")
    drop: set[str] = set()
    if mode not in CONDITIONAL_MODES:
        drop |= {"tau_p05", "tau_p50", "tau_p95"}
    if mode != "A3":
        drop.add("beta_absmean")
    return tuple(c for c in STEP_EXTRA_COLUMNS if c not in drop)


class G4DNumericalError(RuntimeError):
    """A numerical precondition of R1 §7 was violated -- the run must stop.

    Raised on ``cholesky_ex.info != 0``, on a non-positive / non-finite ``L_C``
    diagonal, and on a non-finite forward value.  There is deliberately no
    fallback branch: R1 §7 forbids identity fallbacks and determinant floors so
    that an unexplainable checkpoint cannot be produced quietly.
    """


# --------------------------------------------------------------------------- #
# 1. configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class G4DConfig:
    """Everything the carrier's behaviour depends on -- straight into ``run_setup``."""

    mode: str = "A3"
    n_gauss: int = 48
    cond_dim: int = 64
    hidden: int = 128
    clamp: str = "two"
    residual: bool = True
    marg_norm: str = "peak"
    eps: float = EPS
    tau_min: float = TAU_MIN
    tau_max: float = TAU_MAX
    weight_underflow_tau: float = WEIGHT_UNDERFLOW_TAU
    point_chunk: int | None = None
    init_seed: int = 20260810

    def __post_init__(self) -> None:
        if self.mode not in G4D_MODES:
            raise ValueError(f"unknown mode {self.mode!r}; expected {G4D_MODES}")
        if self.mode == "A4":
            raise NotImplementedError(
                "A4 附录臂待 A3 过 G2 后再实现（R1 §11-2 待裁定）")
        if self.marg_norm not in MARG_NORM_CHOICES:
            raise ValueError(f"marg_norm must be one of {MARG_NORM_CHOICES}")
        if self.clamp not in ("two", "one", "none"):
            raise ValueError("clamp must be two / one / none ('none' is internal)")
        if not (0.0 < self.tau_min < self.tau_max):
            raise ValueError(f"need 0 < tau_min < tau_max; got {self.tau_min}, {self.tau_max}")

    @property
    def conditional(self) -> bool:
        """Does this mode run the R1 §2.1 conditional forward?"""
        return self.mode in CONDITIONAL_MODES

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "n_gauss": self.n_gauss, "cond_dim": self.cond_dim,
            "hidden": self.hidden, "clamp": self.clamp, "residual": self.residual,
            "marg_norm": self.marg_norm, "eps": self.eps,
            "tau_min": self.tau_min, "tau_max": self.tau_max,
            "weight_underflow_tau": self.weight_underflow_tau,
            "point_chunk": self.point_chunk, "init_seed": self.init_seed,
            "theta_dim": n_params_g4d(self.n_gauss, self.mode),
        }


# --------------------------------------------------------------------------- #
# 2. the conditional parameterisation (R1 §4.1)
# --------------------------------------------------------------------------- #
def _math_dtype(*dtypes: torch.dtype) -> torch.dtype:
    """Promote across inputs and floor at float32 (R1 §7: never bf16/fp16 here)."""
    out = torch.float32
    for d in dtypes:
        out = torch.promote_types(out, d)
    if out in (torch.float16, torch.bfloat16):
        out = torch.float32
    if out not in (torch.float32, torch.float64):
        raise TypeError(f"the G4D maths runs at float32/float64 only; got {out}")
    return out


def cholesky_lower(cov: Tensor) -> Tensor:
    """``L`` with ``L L^T == cov`` via ``cholesky_ex``; ``info != 0`` raises.

    R1 §7: no identity fallback, no determinant floor.  A covariance that is not
    positive definite is a bug, and a bug that keeps training produces a
    checkpoint nobody can explain.
    """
    lower, info = torch.linalg.cholesky_ex(cov)
    n_bad = int(info.ne(0).sum())
    if n_bad:
        raise G4DNumericalError(
            f"torch.linalg.cholesky_ex returned info != 0 on {n_bad} of "
            f"{int(info.numel())} matrices (R1 §7: no fallback, the run stops)")
    return lower


def sigma4_from_cond(chol_cond: Tensor, beta: Tensor, tau: Tensor) -> Tensor:
    """The implied joint ``Sigma in R^{4x4}`` of R1 §4.1.

    ``chol_cond`` ``(..., 3, 3)`` lower triangular, ``beta`` ``(..., 3)``,
    ``tau`` ``(...)``::

        Sigma = [[ C + tau^2 beta beta^T , tau^2 beta ],
                 [   tau^2 beta^T        ,   tau^2    ]]      C = L_C L_C^T

    Assembly only -- **never called by the forward**, which reads ``C`` and
    ``beta`` directly.  It exists so G0 can check the parameterisation against
    an arbitrary SPD 4D covariance (:func:`cond_from_sigma4` is its inverse).
    """
    if chol_cond.shape[-2:] != (3, 3):
        raise ValueError(f"chol_cond must be (...,3,3); got {tuple(chol_cond.shape)}")
    if beta.shape[-1] != 3:
        raise ValueError(f"beta must be (...,3); got {tuple(beta.shape)}")
    cov = chol_cond @ chol_cond.transpose(-1, -2)
    t2 = tau * tau                                              # (...)
    t2c = t2.unsqueeze(-1)                                      # (...,1)
    cross = t2c * beta                                          # (...,3)
    top_left = cov + t2c.unsqueeze(-1) * (beta.unsqueeze(-1) @ beta.unsqueeze(-2))
    top = torch.cat([top_left, cross.unsqueeze(-1)], dim=-1)            # (...,3,4)
    bottom = torch.cat([cross, t2c], dim=-1).unsqueeze(-2)             # (...,1,4)
    return torch.cat([top, bottom], dim=-2)


def cond_from_sigma4(sigma4: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Inverse of :func:`sigma4_from_cond`: ``(C, beta, tau^2)`` from ``Sigma``.

    R1 §4.1's expressivity claim, verbatim::

        tau^2 = Sigma_44
        beta  = Sigma_{1:3,4} / Sigma_44
        C     = Sigma_{1:3,1:3} - Sigma_{1:3,4} Sigma_44^{-1} Sigma_{4,1:3}

    ``C`` is validated with ``cholesky_ex`` (R1 §7); a non-SPD input raises.
    """
    if sigma4.shape[-2:] != (4, 4):
        raise ValueError(f"Sigma must be (...,4,4); got {tuple(sigma4.shape)}")
    tau2 = sigma4[..., 3, 3]
    if not bool(torch.isfinite(tau2).all()) or bool((tau2 <= 0).any()):
        raise G4DNumericalError("Sigma_44 must be finite and strictly positive")
    cross = sigma4[..., :3, 3]                                   # (...,3)
    beta = cross / tau2.unsqueeze(-1)
    cov = sigma4[..., :3, :3] - (cross.unsqueeze(-1) @ cross.unsqueeze(-2)) \
        / tau2.unsqueeze(-1).unsqueeze(-1)
    cholesky_lower(cov)                                          # R1 §7 validation
    return cov, beta, tau2


def lower_from_raw(chol_diag: Tensor, chol_off: Tensor, *, strict: bool = True
                   ) -> tuple[Tensor, Tensor, Tensor]:
    """Raw heads -> ``(L_C, diag, n_bad)``; ``diag`` through ``softplus``.

    ``chol_diag`` / ``chol_off`` are ``(..., N, 3)``.  ``L_C`` is
    ``(..., N, 3, 3)`` lower triangular with ``L[0,0], L[1,1], L[2,2]`` positive.

    ``n_bad`` counts diagonals that are non-finite or ``<= 0`` -- the
    ``cholesky_ex.info != 0`` equivalent for a factor that is generated rather
    than factorised (``softplus`` underflows to exactly 0 for raw values below
    about ``-104`` in float64).  ``strict`` raises on the first such entry,
    which is what R1 §7 demands of the training path.
    """
    diag = F.softplus(chol_diag, beta=1.0, threshold=20.0)   # demo :446-449
    bad = (~torch.isfinite(diag)) | (diag <= 0)
    n_bad = bad.sum()
    if strict and bool(n_bad):
        raise G4DNumericalError(
            f"L_C has {int(n_bad)} non-positive / non-finite diagonal entries "
            "(the cholesky_ex info != 0 equivalent); R1 §7 forbids a fallback")
    zero = torch.zeros_like(diag[..., 0])
    rows = [
        torch.stack([diag[..., 0], zero, zero], dim=-1),
        torch.stack([chol_off[..., 0], diag[..., 1], zero], dim=-1),
        torch.stack([chol_off[..., 1], chol_off[..., 2], diag[..., 2]], dim=-1),
    ]
    return torch.stack(rows, dim=-2), diag, n_bad


def tau_from_raw(raw: Tensor, *, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX
                 ) -> Tensor:
    """R1 §4.2 ``tau = tau_min + (tau_max - tau_min) sigmoid(r)`` -- bounded."""
    return tau_min + (tau_max - tau_min) * torch.sigmoid(raw)


def conditional_params(mu_x: Tensor, mu_s: Tensor, beta: Tensor, tau: Tensor,
                       s: Tensor, *, marg_norm: str = "peak"
                       ) -> tuple[Tensor, Tensor, Tensor]:
    """R1 §4.1's conditional mean and §4.2's ``s`` gate, in one place.

    ``mu_x`` ``(B,N,3)``, ``mu_s`` / ``tau`` ``(B,N)``, ``beta`` ``(B,N,3)``,
    ``s`` ``(B,P)``.  Returns ``(mu_{x|s} (B,P,N,3), log g (B,P,N), ds (B,P,N))``.

    The fourth axis is ``s`` **only**: it moves the conditional mean along
    ``beta`` and scales the gate; it never touches ``C``.
    """
    if marg_norm not in MARG_NORM_CHOICES:
        raise ValueError(f"marg_norm must be one of {MARG_NORM_CHOICES}")
    ds = s.unsqueeze(-1) - mu_s.unsqueeze(1)                     # (B,P,N)
    mu_cs = mu_x.unsqueeze(1) + beta.unsqueeze(1) * ds.unsqueeze(-1)
    z = ds / tau.unsqueeze(1)
    log_g = -0.5 * z * z                                         # peak-normalised
    if marg_norm == "full":
        log_g = log_g - torch.log(tau).unsqueeze(1) - 0.5 * LOG_2PI
    return mu_cs, log_g, ds


def log_gauss3(x: Tensor, mu_cs: Tensor, chol_cond: Tensor) -> Tensor:
    """R1 §4.3 ``log p_i(x|s)``, through a triangular solve.

    ``x`` ``(B,P,3)``, ``mu_cs`` ``(B,P,N,3)``, ``chol_cond`` ``(B,N,3,3)``::

        log p = -0.5 [ ||L_C^{-1}(x - mu_{x|s})||^2 + 2 sum_k log L_C[k,k]
                       + 3 log 2pi ]

    The solve is laid out as ``A (B,N,3,3)`` against ``B (B,N,3,P)`` so the
    triangular factor is **not** broadcast into a ``(B,P,N,3,3)`` intermediate.
    No inverse and no determinant are formed (R1 §7).
    """
    diff = x.unsqueeze(2) - mu_cs                                # (B,P,N,3)
    rhs = diff.permute(0, 2, 3, 1)                               # (B,N,3,P)
    sol = torch.linalg.solve_triangular(chol_cond, rhs.contiguous(), upper=False)
    mahal = (sol * sol).sum(dim=2).permute(0, 2, 1)              # (B,P,N)
    log_diag = torch.log(torch.diagonal(chol_cond, dim1=-2, dim2=-1)).sum(-1)
    return -0.5 * (mahal + 2.0 * log_diag.unsqueeze(1) + 3.0 * LOG_2PI)


def mixture_log_weights(log_a: Tensor, *, eps: float = EPS
                        ) -> tuple[Tensor, Tensor, Tensor]:
    """R1 §4.3's normalisation.  ``log_a`` ``(B,P,N)`` -> ``(w, log_Z, null_mass)``.

    ``log Z = logaddexp(logsumexp_i log a_i, log eps)`` keeps GLUT Eq.2's
    ``sum_i a_i + eps`` **exactly** while never materialising ``a_i`` itself, so
    nothing is pressed to zero on the way.  ``null_mass = exp(log eps - log Z)``
    is the direct measure of "every Gaussian lost coverage and the point fell
    through to the global branch".
    """
    lse = torch.logsumexp(log_a, dim=-1)                         # (B,P)
    log_eps = lse.new_full((), math.log(eps))
    log_z = torch.logaddexp(lse, log_eps.expand_as(lse))
    w = torch.exp(log_a - log_z.unsqueeze(-1))
    null_mass = torch.exp(log_eps - log_z)
    return w, log_z, null_mass


# --------------------------------------------------------------------------- #
# 3. parameters
# --------------------------------------------------------------------------- #
_COMMON_FIELDS: tuple[str, ...] = (
    "mu_x", "chol_diag", "chol_off", "opacity_logit",
    "m_local", "b_local", "g_matrix", "g_bias",
)
_COND_FIELDS: tuple[str, ...] = ("beta", "mu_s", "tau_raw")
_ALL_FIELDS: tuple[str, ...] = _COMMON_FIELDS + _COND_FIELDS


@dataclass(frozen=True)
class G4DParams:
    """One batch of this arm's parameter sets.

    ==================  =====================  ============================
    field               shape                  modes
    ==================  =====================  ============================
    ``mu_x``            ``(B, N, 3)``          all
    ``chol_diag``       ``(B, N, 3)``          all -- RAW, ``softplus`` here
    ``chol_off``        ``(B, N, 3)``          all -- ``L[1,0], L[2,0], L[2,1]``
    ``opacity_logit``   ``(B, N)``             all -- RAW, ``sigmoid``/``logsigmoid`` here
    ``m_local``         ``(B, N, 3, 3)``       all
    ``b_local``         ``(B, N, 3)``          all
    ``g_matrix``        ``(B, 3, 3)``          all
    ``g_bias``          ``(B, 3)``             all
    ``beta``            ``(B, N, 3)``          A2, A3
    ``mu_s``            ``(B, N)``             A2, A3
    ``tau_raw``         ``(B, N)``             A2, A3 -- RAW, :func:`tau_from_raw`
    ==================  =====================  ============================

    ``A2`` and ``A3`` carry **the same fields with the same shapes**; ``beta`` is
    generated in both and only ``A3`` lets it into the forward.  That is the
    whole difference between the two arms (R1 §3).
    """

    mode: str
    mu_x: Tensor
    chol_diag: Tensor
    chol_off: Tensor
    opacity_logit: Tensor
    m_local: Tensor
    b_local: Tensor
    g_matrix: Tensor
    g_bias: Tensor
    beta: Tensor | None = None
    mu_s: Tensor | None = None
    tau_raw: Tensor | None = None

    def __post_init__(self) -> None:
        if self.mode not in G4D_MODES:
            raise ValueError(f"unknown mode {self.mode!r}")
        if self.mode == "A4":
            raise NotImplementedError(
                "A4 附录臂待 A3 过 G2 后再实现（R1 §11-2 待裁定）")
        cond = self.mode in CONDITIONAL_MODES
        for name in _COND_FIELDS:
            got = getattr(self, name)
            if cond and got is None:
                raise ValueError(f"mode {self.mode!r} requires {name!r}")
            if not cond and got is not None:
                raise ValueError(
                    f"mode {self.mode!r} has no {name!r}; a 3D arm carrying a "
                    "conditional field is an unnoticed mode mix-up")
        b, n = self.opacity_logit.shape[0], self.opacity_logit.shape[1]
        want = {"mu_x": (b, n, 3), "chol_diag": (b, n, 3), "chol_off": (b, n, 3),
                "opacity_logit": (b, n), "m_local": (b, n, 3, 3),
                "b_local": (b, n, 3), "g_matrix": (b, 3, 3), "g_bias": (b, 3)}
        if cond:
            want.update({"beta": (b, n, 3), "mu_s": (b, n), "tau_raw": (b, n)})
        devices: dict[str, torch.device] = {}
        for name, shape in want.items():
            t = getattr(self, name)
            if tuple(t.shape) != shape:
                raise ValueError(f"G4DParams.{name}: expected {shape}, got {tuple(t.shape)}")
            devices[name] = t.device
        if len(set(devices.values())) != 1:
            raise ValueError(
                "G4DParams tensors straddle devices -- the EPR-022/MATTE failure "
                f"mode; place them explicitly: {devices}")

    # ---- bookkeeping ----
    @property
    def batch_size(self) -> int:
        return int(self.opacity_logit.shape[0])

    @property
    def n_gauss(self) -> int:
        return int(self.opacity_logit.shape[1])

    @property
    def device(self) -> torch.device:
        return self.opacity_logit.device

    @property
    def dtype(self) -> torch.dtype:
        return self.opacity_logit.dtype

    def _map(self, fn) -> "G4DParams":
        kw = {}
        for name in _ALL_FIELDS:
            t = getattr(self, name)
            kw[name] = None if t is None else fn(t)
        return G4DParams(mode=self.mode, **kw)

    def to(self, *, device: Any = None, dtype: torch.dtype | None = None) -> "G4DParams":
        return self._map(lambda t: t.to(device=device if device is not None else t.device,
                                        dtype=dtype if dtype is not None else t.dtype))

    def to_ref(self, ref: Tensor) -> "G4DParams":
        return self.to(device=ref.device, dtype=ref.dtype)

    def detach(self) -> "G4DParams":
        return self._map(lambda t: t.detach())

    def select(self, index: int) -> "G4DParams":
        """The single-sample slice ``[i:i+1]`` (batch axis kept)."""
        i = int(index)
        return self._map(lambda t: t[i:i + 1])

    def expand_batch(self, batch: int) -> "G4DParams":
        if self.batch_size == batch:
            return self
        if self.batch_size != 1:
            raise ValueError(f"cannot expand batch {self.batch_size} -> {batch}")
        return self._map(lambda t: t.expand(batch, *t.shape[1:]))

    def as_glut_params(self) -> GlutParams:
        """The shared 3D container -- ``A0`` / ``A1`` run the shared carrier.

        ``A2``/``A3`` may also be viewed this way (the geometry fields are the
        same eight tensors), but the conditional forward does **not** go through
        it: the mean is per point there.
        """
        return GlutParams(mu=self.mu_x, chol_diag=self.chol_diag,
                          chol_off=self.chol_off, opacity_logit=self.opacity_logit,
                          m_local=self.m_local, b_local=self.b_local,
                          g_matrix=self.g_matrix, g_bias=self.g_bias)


#: ``torch.quantile`` hard-errors above ``2**24`` inputs.  A telemetry column
#: must not be able to kill a run, so above that the tensor is decimated with a
#: fixed stride first -- deterministic, and recorded here rather than silent.
_QUANTILE_MAX: int = 1 << 24


def _quantiles(t: Tensor, qs: Sequence[float]) -> list[Tensor]:
    """Device-resident quantiles of a flattened tensor (never ``.cpu()``)."""
    flat = t.reshape(-1)
    if flat.numel() > _QUANTILE_MAX:
        flat = flat[:: (flat.numel() + _QUANTILE_MAX - 1) // _QUANTILE_MAX]
    if flat.numel() == 0:
        return [flat.new_zeros(()) for _ in qs]
    probs = flat.new_tensor(list(qs))
    out = torch.quantile(flat.to(torch.float32), probs.to(torch.float32))
    return [out[i] for i in range(len(qs))]


@dataclass(frozen=True)
class G4DAux:
    """Per-forward diagnostics (R1 §10).  Every reduction stays on the device."""

    mode: str
    log_influence_sum: Tensor        # (B, P)   logsumexp_i log a_i, BEFORE + eps
    null_mass: Tensor                # (B, P)   eps / (sum_i a_i + eps)
    pre_clamp: Tensor                # (B, P, 3)
    opacity: Tensor                  # (B, N)   sigmoid(o_logit)
    cholesky_info_nonzero: Tensor    # scalar, identically 0 (a non-zero raises)
    n_pairs_s: int                   # B * P -- pins the R1 §8.1 sampler structure
    tau: Tensor | None = None        # (B, N),  A2 / A3
    beta: Tensor | None = None       # (B, N, 3), A2 / A3

    def oob_mask(self, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        return ((self.pre_clamp < lo) | (self.pre_clamp > hi)).any(dim=-1)

    def degenerate_weight_mask(self, tau: float = WEIGHT_UNDERFLOW_TAU) -> Tensor:
        """``sum_i a_i < tau``, compared in the log domain (no ``exp`` round trip)."""
        return self.log_influence_sum < math.log(float(tau))

    def columns(self, *, weight_underflow_tau: float = WEIGHT_UNDERFLOW_TAU
                ) -> dict[str, Tensor]:
        """R1 §10's step columns for this mode, as device tensors.

        ``gnorm`` is NOT here: it belongs to the training loop, which is the only
        place that has the gradients.
        """
        nm = self.null_mass
        nm_p99, = _quantiles(nm, (0.99,))
        o5, o50, o95 = _quantiles(self.opacity, (0.05, 0.5, 0.95))
        cols: dict[str, Tensor] = {
            "null_mass_mean": nm.mean() if nm.numel() else nm.new_zeros(()),
            "null_mass_p99": nm_p99,
            "cholesky_info_nonzero": self.cholesky_info_nonzero,
            "opacity_p05": o5, "opacity_p50": o50, "opacity_p95": o95,
            "n_pairs_s": torch.as_tensor(self.n_pairs_s, device=nm.device),
            "weight_underflow_frac": self.degenerate_weight_mask(
                weight_underflow_tau).to(nm.dtype).mean()
            if nm.numel() else nm.new_zeros(()),
        }
        if self.tau is not None:
            t5, t50, t95 = _quantiles(self.tau, (0.05, 0.5, 0.95))
            cols.update({"tau_p05": t5, "tau_p50": t50, "tau_p95": t95})
        if self.mode == "A3" and self.beta is not None:
            cols["beta_absmean"] = self.beta.abs().mean()
        return cols


# --------------------------------------------------------------------------- #
# 4. the carrier
# --------------------------------------------------------------------------- #
def _auto_point_chunk(n_points: int, n_gauss: int, per_point_terms: int) -> int:
    """Block size keeping the biggest ``(B,P,N,·)`` intermediate bounded.

    Chunking is numerically inert: points are independent.
    """
    budget = 1 << 22
    return max(1, min(int(n_points), budget // max(1, n_gauss * per_point_terms)))


class Glut4DCarrier(nn.Module):
    """``f(x, s)`` for every live arm.  Zero parameters.

    ``A0``  -- ``glut_forward`` on the shared 3D carrier; ``s`` is not consumed.
    ``A1``  -- ``f(x,s) = x + s [T(x) - x]`` with ``T`` the same shared carrier.
               Written through :func:`~q3vl.whatb.lutdata.mix_alpha`, the data
               law's own mixture, so ``s = 0`` and ``s = 1`` are bit-exact
               endpoints (``rendering.py:311-313``).
    ``A2``  -- R1 §2.1 with ``beta`` forced to zero **inside the forward**.
    ``A3``  -- R1 §2.1 with ``beta`` live.

    Autocast is disabled for the maths (``matmul`` / ``solve_triangular`` are on
    autocast's low-precision list) and the compute dtype is floored at float32.
    """

    def __init__(self, cfg: G4DConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.register_buffer("eye3", torch.eye(3), persistent=False)

    @property
    def config(self) -> dict[str, Any]:
        return self.cfg.as_dict()

    # -- helpers ----------------------------------------------------------- #
    def _resolve(self, x: Tensor, params: G4DParams) -> tuple[torch.device, torch.dtype]:
        if x.device != params.device:
            raise ValueError(
                f"x is on {x.device} and the parameters on {params.device}; the "
                "caller must place them (.to(device=ref.device, dtype=ref.dtype))")
        return x.device, _math_dtype(x.dtype, params.dtype)

    @staticmethod
    def _as_bp(x: Tensor, s: Tensor | float, batch: int) -> tuple[Tensor, Tensor]:
        """``x -> (B,P,3)``, ``s -> (B,P)``, broadcasting a shared query set."""
        if x.dim() == 2:
            x = x.unsqueeze(0).expand(batch, -1, -1)
        elif x.dim() != 3:
            raise ValueError(f"x must be (B,P,3) or (P,3); got {tuple(x.shape)}")
        if x.shape[0] == 1 and batch > 1:
            x = x.expand(batch, -1, -1)
        elif x.shape[0] != batch:
            raise ValueError(
                f"x batch {x.shape[0]} is neither 1 nor the parameter batch {batch}")
        if not torch.is_tensor(s):
            s = x.new_full(x.shape[:2], float(s))
        else:
            s = s.to(device=x.device, dtype=x.dtype)
            if s.dim() == 0:
                s = s.expand(x.shape[0], x.shape[1])
            elif s.dim() == 1:
                s = s.unsqueeze(0).expand(x.shape[0], -1) if s.shape[0] == x.shape[1] \
                    else s.unsqueeze(-1).expand(-1, x.shape[1])
            elif s.dim() == 2 and s.shape[1] == 1:
                s = s.expand(-1, x.shape[1])
        if s.shape != x.shape[:2]:
            raise ValueError(f"s {tuple(s.shape)} does not match x {tuple(x.shape)}")
        return x, s

    # -- forward ----------------------------------------------------------- #
    def forward(self, x: Tensor, s: Tensor | float, params: G4DParams, *,
                clamp: str | None = None, return_aux: bool = False,
                point_chunk: int | None = None
                ) -> Tensor | tuple[Tensor, G4DAux]:
        cfg = self.cfg
        if params.mode != cfg.mode:
            raise ValueError(f"carrier is {cfg.mode!r}, parameters are {params.mode!r}")
        device, dtype = self._resolve(x, params)
        clamp_mode = cfg.clamp if clamp is None else clamp
        p = params.to(device=device, dtype=dtype)
        xb, sb = self._as_bp(x.to(device=device, dtype=dtype), s, p.batch_size)

        with _no_autocast(device):
            if cfg.conditional:
                out = self._conditional_forward(xb, sb, p, clamp_mode, point_chunk)
            else:
                out = self._explicit_gate_forward(xb, sb, p, clamp_mode, point_chunk)
        y, aux = out
        return (y, aux) if return_aux else y

    # -- A0 / A1 ----------------------------------------------------------- #
    def _explicit_gate_forward(self, x: Tensor, s: Tensor, p: G4DParams,
                               clamp_mode: str, point_chunk: int | None
                               ) -> tuple[Tensor, G4DAux]:
        cfg = self.cfg
        # R1 §7 applies to every arm: a degenerate Cholesky diagonal raises here
        # rather than silently taking the shared carrier's identity fallback.
        _, _, n_bad = lower_from_raw(p.chol_diag, p.chol_off)
        t, aux = glut_forward(x, p.as_glut_params(), clamp=clamp_mode,
                              residual=cfg.residual, eps=cfg.eps,
                              point_chunk=point_chunk or cfg.point_chunk,
                              return_aux=True)
        if p.mode == "A1":
            y = mix_alpha(x, t, s.unsqueeze(-1))
            pre = mix_alpha(x, aux.pre_clamp, s.unsqueeze(-1))
        else:                                   # A0: the carrier does not eat s
            y, pre = t, aux.pre_clamp
        self._assert_finite(y)
        log_infl = torch.log(aux.influence_sum.clamp_min(_tiny(aux.influence_sum)))
        null_mass = cfg.eps / (aux.influence_sum + cfg.eps)
        return y, G4DAux(
            mode=p.mode, log_influence_sum=log_infl, null_mass=null_mass,
            pre_clamp=pre, opacity=aux.opacity, cholesky_info_nonzero=n_bad,
            n_pairs_s=int(x.shape[0]) * int(x.shape[1]))

    # -- A2 / A3 ----------------------------------------------------------- #
    def _conditional_forward(self, x: Tensor, s: Tensor, p: G4DParams,
                             clamp_mode: str, point_chunk: int | None
                             ) -> tuple[Tensor, G4DAux]:
        cfg = self.cfg
        b, n_pts, n_g = x.shape[0], x.shape[1], p.n_gauss

        chol, _, n_bad = lower_from_raw(p.chol_diag, p.chol_off)      # (B,N,3,3)
        tau = tau_from_raw(p.tau_raw, tau_min=cfg.tau_min, tau_max=cfg.tau_max)
        # R1 §3: A2 generates beta and does not let it into the forward.  Zeroing
        # (rather than dropping the head) is what makes A2 and A3 bit-identical
        # at step 0 and identical in parameter count for ever after.
        beta = p.beta if p.mode == "A3" else torch.zeros_like(p.beta)
        log_o = F.logsigmoid(p.opacity_logit)                        # (B,N)

        chunk = (point_chunk or cfg.point_chunk
                 or _auto_point_chunk(n_pts, n_g, 12))
        ys: list[Tensor] = []
        pres: list[Tensor] = []
        lses: list[Tensor] = []
        nulls: list[Tensor] = []
        for start in range(0, max(n_pts, 1), chunk):
            xs = x[:, start:start + chunk, :]
            ss = s[:, start:start + chunk]
            if xs.shape[1] == 0:
                continue
            mu_cs, log_g, _ = conditional_params(p.mu_x, p.mu_s, beta, tau, ss,
                                                 marg_norm=cfg.marg_norm)
            log_p = log_gauss3(xs, mu_cs, chol)                      # (B,P,N)
            log_a = log_p + log_o.unsqueeze(1) + log_g
            w, log_z, null_mass = mixture_log_weights(log_a, eps=cfg.eps)

            mx = (p.m_local.unsqueeze(1) @ xs.unsqueeze(2).unsqueeze(-1)).squeeze(-1)
            local = (w.unsqueeze(-1) * (mx + p.b_local.unsqueeze(1))).sum(dim=-2)
            glob = xs @ p.g_matrix.transpose(-1, -2) + p.g_bias.unsqueeze(1)
            if clamp_mode == "two":
                glob = glob.clamp(0.0, 1.0)                          # demo :574-579
            pre = glob + local if cfg.residual else local            # demo :596-604
            y = pre if clamp_mode == "none" else pre.clamp(0.0, 1.0)  # demo :606-610
            ys.append(y)
            pres.append(pre)
            lses.append(log_z)
            nulls.append(null_mass)

        y = torch.cat(ys, dim=1) if ys else x.new_zeros((b, 0, 3))
        pre = torch.cat(pres, dim=1) if pres else x.new_zeros((b, 0, 3))
        log_z = torch.cat(lses, dim=1) if lses else x.new_zeros((b, 0))
        null_mass = torch.cat(nulls, dim=1) if nulls else x.new_zeros((b, 0))
        self._assert_finite(y, log_z)
        # sum_i a_i = Z - eps = Z (1 - null_mass), so log(sum_i a_i) never leaves
        # the log domain.  A point whose Gaussians all lost coverage has
        # null_mass == 1 and log_influence_sum == -inf, which is the truth.
        log_infl = log_z + torch.log1p(-null_mass)
        return y, G4DAux(
            mode=p.mode, log_influence_sum=log_infl, null_mass=null_mass,
            pre_clamp=pre, opacity=torch.sigmoid(p.opacity_logit),
            cholesky_info_nonzero=n_bad, n_pairs_s=b * n_pts,
            tau=tau, beta=p.beta)

    @staticmethod
    def _assert_finite(*tensors: Tensor) -> None:
        """R1 §7: a NaN / Inf in the forward stops the run; it is never absorbed."""
        for t in tensors:
            if t.numel() and not bool(torch.isfinite(t).all()):
                raise G4DNumericalError(
                    "the G4D forward produced a non-finite value (R1 §7: the run "
                    "stops rather than emit an unexplainable checkpoint)")


def _tiny(ref: Tensor) -> float:
    return float(torch.finfo(ref.dtype).tiny)


# --------------------------------------------------------------------------- #
# 5. the generator (App A.2 shape, this arm's heads)
# --------------------------------------------------------------------------- #
class G4DGenerator(nn.Module):
    """``G: R^d -> Theta``.  App A.2's shape, R1 §4.1's heads.

    App A.2 verbatim for the shape: shared encoder = 3 linear layers of ``H``
    hidden units with ReLU after each; every head is 2 linear layers **except**
    the local colour head, which is 3; the global affine head outputs 12.

    ``A2`` and ``A3`` build **the same modules in the same order with the same
    shapes**, so two generators constructed under the same RNG state are
    parameter-for-parameter identical (G0 pins it).

    Initialisation (R1 §8.4): the last layer of every head is zero-weighted and
    its bias carries the init, so step 0 emits ``mu_x`` on the RGB grid,
    isotropic ``L_C`` diag ``= 0.15``, ``beta = 0``, ``mu_s ~ U(-0.1, 1.1)``,
    ``tau ~= sqrt(0.2)``, ``o_logit = 0`` (``sigmoid = 0.5``), ``M = I``,
    ``b = 0``, ``G = 0``, ``g = 0``; with ``sum_i w_i ~= 1`` that is
    ``f(x, s) ~= x`` in every mode.
    """

    def __init__(self, cfg: G4DConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, h, n = cfg.cond_dim, cfg.hidden, cfg.n_gauss
        self.encoder = _app_a2_mlp([d, h, h, h])
        self.encoder.append(nn.ReLU())

        self.head_color = _app_a2_mlp([h, h, h, 12 * n])      # 3 layers (App A.2)
        self.head_global = _app_a2_mlp([h, h, 12])
        self.head_opacity = _app_a2_mlp([h, h, n])
        self.head_mu = _app_a2_mlp([h, h, 3 * n])
        self.head_cov = _app_a2_mlp([h, h, 6 * n])
        if cfg.conditional:
            self.head_beta = _app_a2_mlp([h, h, 3 * n])
            self.head_stime = _app_a2_mlp([h, h, 2 * n])     # (mu_s, tau_raw)
        else:
            self.head_beta = None
            self.head_stime = None

        self.register_buffer("eye3", torch.eye(3), persistent=False)
        self._init_heads()

    # -- init --------------------------------------------------------------- #
    def _init_heads(self) -> None:
        cfg = self.cfg
        n = cfg.n_gauss
        gen = torch.Generator().manual_seed(int(cfg.init_seed))
        grid = uniform_grid_positions(n)                             # (N,3)
        lo, hi = MU_S_INIT_RANGE
        mu_s0 = torch.rand(n, generator=gen) * (hi - lo) + lo         # 4DGS :268

        def _zero_last(head: nn.Module, bias: Tensor) -> None:
            last = head[-1]
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(bias.reshape(-1))

        colour_bias = torch.zeros(n, 12)
        colour_bias[:, :9] = torch.eye(3).reshape(-1)                 # M = I, b = 0
        _zero_last(self.head_color, colour_bias)
        _zero_last(self.head_global, torch.zeros(12))                 # G = 0, g = 0
        _zero_last(self.head_opacity, torch.full((n,), OPACITY_LOGIT_INIT))
        _zero_last(self.head_mu, grid)
        chol = torch.zeros(n, 6)
        chol[:, :3] = softplus_inverse(SIGMA_RGB_INIT)                # App A.1
        _zero_last(self.head_cov, chol)
        if self.head_beta is not None:
            _zero_last(self.head_beta, torch.zeros(n, 3))             # beta = 0
        if self.head_stime is not None:
            st = torch.stack([mu_s0, torch.full((n,), TAU_RAW_INIT)], dim=-1)
            _zero_last(self.head_stime, st)

    # -- bookkeeping -------------------------------------------------------- #
    @property
    def theta_dim(self) -> int:
        return n_params_g4d(self.cfg.n_gauss, self.cfg.mode)

    @property
    def config(self) -> dict[str, Any]:
        return {**self.cfg.as_dict(),
                "n_params": sum(p.numel() for p in self.parameters()),
                "heads": sorted(name for name, m in self.named_children()
                                if name.startswith("head_") and m is not None)}

    def encode(self, u: Tensor) -> Tensor:
        if u.dim() != 2 or u.shape[-1] != self.cfg.cond_dim:
            raise ValueError(f"condition must be (B, {self.cfg.cond_dim}), got {tuple(u.shape)}")
        ref = self.head_global[0].weight
        return self.encoder(u.to(device=ref.device, dtype=ref.dtype))

    # -- forward ------------------------------------------------------------ #
    def forward(self, u: Tensor) -> G4DParams:
        cfg = self.cfg
        h = self.encode(u)
        b, n = h.shape[0], cfg.n_gauss

        colour = self.head_color(h).reshape(b, n, 12)
        glob = self.head_global(h)
        cov = self.head_cov(h).reshape(b, n, 6)
        out: dict[str, Any] = dict(
            mu_x=self.head_mu(h).reshape(b, n, 3),
            chol_diag=cov[..., :3], chol_off=cov[..., 3:],
            opacity_logit=self.head_opacity(h).reshape(b, n),
            m_local=colour[..., :9].reshape(b, n, 3, 3),
            b_local=colour[..., 9:12],
            g_matrix=glob[:, :9].reshape(b, 3, 3),
            g_bias=glob[:, 9:12])
        if cfg.conditional:
            st = self.head_stime(h).reshape(b, n, 2)
            out["beta"] = self.head_beta(h).reshape(b, n, 3)
            out["mu_s"] = st[..., 0]
            out["tau_raw"] = st[..., 1]
        return G4DParams(mode=cfg.mode, **out)


# --------------------------------------------------------------------------- #
# 6. the arm
# --------------------------------------------------------------------------- #
class G4DArm(nn.Module):
    """``pi`` + generator + carrier: everything trainable in EPR-028.

    ``pi`` is the shared :class:`~q3vl.whatb.generator.SegColorProjection`
    (LayerNorm(2560) + Linear(2560 -> d)).  R1 §6 splits the campaign into
    Experiment C (``cond`` = a learnable ``E[lut_id]``, fed straight in as the
    ``d``-vector, so ``pi`` is untouched) and Experiment Z (``pi(z_color)``);
    ``theta`` accepts either width.
    """

    def __init__(self, cfg: G4DConfig | None = None, **kw: Any) -> None:
        super().__init__()
        self.cfg = cfg if cfg is not None else G4DConfig(**kw)
        self.pi = SegColorProjection(in_dim=SEG_COLOR_HIDDEN_DIM,
                                     cond_dim=self.cfg.cond_dim)
        self.generator = G4DGenerator(self.cfg)
        self.carrier = Glut4DCarrier(self.cfg)

    # -- bookkeeping -------------------------------------------------------- #
    @property
    def config(self) -> dict[str, Any]:
        return {"arm": ARM, "axes": list(ARM_AXES), **self.cfg.as_dict(),
                "generator": self.generator.config,
                "carrier": self.carrier.config,
                "n_params_pi": sum(p.numel() for p in self.pi.parameters()),
                "n_params_generator": sum(p.numel() for p in self.generator.parameters()),
                "n_params_total": sum(p.numel() for p in self.parameters()),
                "opacity_logit_init": OPACITY_LOGIT_INIT,
                "sigma_rgb_init": SIGMA_RGB_INIT,
                "tau_init": TAU_INIT, "tau_raw_init": TAU_RAW_INIT,
                "mu_s_init_range": list(MU_S_INIT_RANGE)}

    def param_groups(self, base_lr: float, *, pi_lr_scale: float = 0.1
                     ) -> list[dict[str, Any]]:
        """``pi`` at ``0.1x`` (it stands where CGLUT's style embedding stands)."""
        return [
            {"params": list(self.generator.parameters()), "lr": float(base_lr),
             "name": "generator"},
            {"params": list(self.pi.parameters()),
             "lr": float(base_lr) * float(pi_lr_scale), "name": "pi"},
        ]

    # -- forward ------------------------------------------------------------ #
    def theta(self, z: Tensor) -> G4DParams:
        """``z (B, 2560) -> Theta``.  ``z`` may already be the ``d``-vector."""
        u = z if z.shape[-1] == self.cfg.cond_dim else self.pi(z)
        return self.generator(u)

    def transform(self, params: G4DParams, x: Tensor, s: Tensor | float, *,
                  clamp: str | None = None, return_aux: bool = False,
                  point_chunk: int | None = None):
        """``f_theta(x, s)`` on ``(B,P,3)`` colours (or a shared ``(P,3)`` set)."""
        return self.carrier(x, s, params, clamp=clamp, return_aux=return_aux,
                            point_chunk=point_chunk)

    def forward(self, z: Tensor, x: Tensor, s: Tensor | float, **kw):
        return self.transform(self.theta(z), x, s, **kw)

    def apply_to_image(self, params: G4DParams, img: Tensor, field: Tensor | float,
                       *, chunk: int | None = None) -> Tensor:
        """``f_theta(I(p), s = S(p))`` for ``img (B,3,H,W)``; returns ``(B,3,H,W)``.

        For ``A0`` the carrier ignores ``field`` (``A0`` does not eat ``s``); the
        field enters through :func:`compose_headline` instead.  For ``A1``/``A2``/
        ``A3`` the returned tensor is already ``f(I, S)`` and
        :func:`compose_headline` is the identity on it.
        """
        if img.dim() != 4 or img.shape[1] != 3:
            raise ValueError(f"img must be (B,3,H,W), got {tuple(img.shape)}")
        b, _, h, w = img.shape
        x = img.permute(0, 2, 3, 1).reshape(b, h * w, 3)
        if torch.is_tensor(field):
            f = field.to(device=img.device, dtype=img.dtype)
            f = f.reshape(b, -1) if f.numel() == b * h * w else f.reshape(b, 1).expand(b, h * w)
        else:
            f = float(field)
        y = self.transform(params, x, f, point_chunk=chunk)
        return y.reshape(b, h, w, 3).permute(0, 3, 1, 2)


def compose_headline(img: Tensor, field: Tensor | float, f_img: Tensor, *,
                     mode: str) -> Tensor:
    """R1 §5's image formation, **per arm**.

    ==========  =======================================================
    mode        ``Î``
    ==========  =======================================================
    ``A0``      ``I + S ⊙ [T(I) - I]``  (``f_img`` is ``T(I)``)
    ``A1``..``A3``  ``f(I, S)``         (``f_img`` already ate ``S``)
    ==========  =======================================================

    ``mode`` is keyword-only and has no default on purpose: R0's single outer
    ``mix_alpha`` applied on top of a carrier that had already consumed ``s`` is
    the double-alpha bug of R1 §0.1 (``Î - I* = -a(1-a)[L(I) - I]``, 25 % of the
    full LUT residual at ``a = 0.5``), and it was invisible precisely because the
    caller could not tell which convention it was in.

    Fairness across arms is carried by the same GT, samples, metrics and step
    count -- not by a shared outer compositor.
    """
    if mode not in G4D_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected {G4D_MODES}")
    if mode == "A0":
        alpha = field if torch.is_tensor(field) else float(field)
        return mix_alpha(img, f_img, alpha)
    return f_img


# --------------------------------------------------------------------------- #
# 7. losses (R1 §4.4)
# --------------------------------------------------------------------------- #
def target_4d(x: Tensor, s: Tensor, lut_values: Tensor) -> Tensor:
    """``y* = (1-s) x + s L_l(x)`` -- the data law's own mixture.

    ``x`` / ``lut_values`` are ``(..., 3)`` and ``s`` is ``(...)``
    (``rendering.py:311-313`` via :func:`~q3vl.whatb.lutdata.mix_alpha`), so the
    training target and the dataset's GT image are the same formula.
    """
    return mix_alpha(x, lut_values, s.unsqueeze(-1))


def l_rec(y_hat: Tensor, y: Tensor) -> Tensor:
    """GLUT Eq.6 ``|| ŷ - y ||_1`` (mean over points and channels)."""
    return (y_hat - y).abs().mean()


def l_hc(y_hat: Tensor, y: Tensor, *, eps_c: float = EPS_CHROMA
         ) -> tuple[Tensor, int]:
    """GLUT Eq.7 with the frozen ``C -> 0`` treatment.  Returns ``(value, n_masked)``."""
    lab_hat = srgb_to_lab(y_hat)
    lab = srgb_to_lab(y)
    c_t, h_t, valid = chroma_hue(lab, eps_c)
    _, h_p, _ = chroma_hue(lab_hat, eps_c)
    cos = (h_p * h_t).sum(-1)
    term = c_t * (1.0 - cos) * valid.to(c_t.dtype)
    return term.mean(), int((~valid).sum())


def r_sparse(opacity: Tensor, *, eps: float = EPS) -> Tensor:
    """GLUT Eq.8, the binary entropy of the ``s``-independent ``o_i``."""
    o = opacity
    ent = o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)
    return -ent.mean()


def r_line(f_s: Tensor, f_0: Tensor, f_1: Tensor, s: Tensor) -> Tensor:
    """R1 §4.4 ``|| f(x,s) - [(1-s) f(x,0) + s f(x,1)] ||_1``.

    Replaces R0's deleted ``L_m4d``.  Zero extra forward cost: R1 §8.1's paired
    anchors already evaluate every colour at ``s = 0`` and ``s = 1``.

    ``f_*`` are ``(..., 3)`` and ``s`` is ``(...)``.
    """
    a = s.unsqueeze(-1)
    return (f_s - ((1.0 - a) * f_0 + a * f_1)).abs().mean()


def grid_4d(n_color: int = 17, n_s: int = 17, *, device: Any = "cpu",
            dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """The ``17^4`` **evaluation** lattice (R1 §8.3: never a per-step training cost).

    Returns ``(x (P,3), s (P,))`` ordered so a reshape to ``(n,n,n,n_s)`` gives
    the four axes (R, G, B, s) for differencing.
    """
    ax = torch.linspace(0.0, 1.0, n_color, device=device, dtype=dtype)
    sx = torch.linspace(0.0, 1.0, n_s, device=device, dtype=dtype)
    r, g, b, s = torch.meshgrid(ax, ax, ax, sx, indexing="ij")
    x = torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)
    return x, s.reshape(-1)


def _as_lattice(values: Tensor, n_color: int, n_s: int) -> Tensor:
    """``(B, P, 3) -> (B, n, n, n, n_s, 3)``."""
    b = values.shape[0]
    return values.reshape(b, n_color, n_color, n_color, n_s, 3)


def l_s4d(values: Tensor, *, n_color: int = 17, n_s: int = 17) -> Tensor:
    """4D LUT Eq.12: squared first differences along all four axes (R,G,B,s)."""
    v = _as_lattice(values, n_color, n_s)
    total = v.new_zeros(())
    for axis in (1, 2, 3, 4):
        d = v.narrow(axis, 1, v.shape[axis] - 1) - v.narrow(axis, 0, v.shape[axis] - 1)
        total = total + (d * d).mean()
    return total


def l_img(i_hat: Tensor, i_star: Tensor) -> Tensor:
    """``|| Î - I* ||_1`` with ``Î`` built by :func:`compose_headline`."""
    return (i_hat - i_star).abs().mean()


def total_loss(y_hat: Tensor, y: Tensor, opacity: Tensor, *,
               lam_hc: float = LAMBDA_HC, lam_sparse: float = LAMBDA_SPARSE,
               lam_line: float = LAMBDA_LINE, line: Tensor | None = None,
               eps_c: float = EPS_CHROMA, eps: float = EPS,
               extra: Mapping[str, tuple[Tensor, float]] | None = None
               ) -> tuple[Tensor, dict[str, Any]]:
    """R1 §4.4's main arm: ``L_rec + lam_line R_line`` (``--loss-level 1``).

    ``lam_hc`` / ``lam_sparse`` come from
    :func:`q3vl.whatb.caliber.effective_lambda_hc` /
    :func:`~q3vl.whatb.caliber.effective_lambda_sparse`, which zero both at
    ``--loss-level 1``.  ``line`` is :func:`r_line`'s value; passing it with
    ``lam_line == 0`` is an error -- a term that is switched off must not be
    computed (the same "off but still computed" discipline the first-row
    assertion enforces for ``L_s4d`` / ``L_img``).
    """
    rec = l_rec(y_hat, y)
    hc, n_masked = l_hc(y_hat, y, eps_c=eps_c)
    sp = r_sparse(opacity, eps=eps)
    loss = rec + lam_hc * hc + lam_sparse * sp
    cols: dict[str, Any] = {"L_rec": float(rec.detach()), "L_hc": float(hc.detach()),
                            "L_sparse": float(sp.detach()), "n_hc_masked": n_masked}
    if line is not None:
        if float(lam_line) == 0.0:
            raise ValueError(
                "R_line was computed with lam_line = 0 -- either drop the term "
                "or give it a weight (the lambda_line = 0 ablation row must not "
                "pay for a term it does not use)")
        loss = loss + float(lam_line) * line
        cols["R_line"] = float(line.detach())
    for name, (value, weight) in dict(extra or {}).items():
        if weight == 0.0:
            raise ValueError(
                f"{name} was computed with weight 0 -- either drop the term or "
                "give it a weight; a zero-weighted term still costs the step and "
                "silently appears in the first row")
        loss = loss + float(weight) * value
        cols[name] = float(value.detach())
    cols["L_total"] = float(loss.detach())
    return loss, cols


# --------------------------------------------------------------------------- #
# 8. guards, pre-registration, criteria
# --------------------------------------------------------------------------- #
def assert_not_degenerate(arm: G4DArm, z: Tensor, x: Tensor, *, s: float = 1.0,
                          thresholds: DegeneracyThresholds | None = None,
                          where: str = "quick_eval", exit_process: bool = True,
                          params: G4DParams | None = None) -> DegeneracyReport:
    """The first-quick-eval degeneracy gate.

    Evaluates ``f̂`` at ``s = 1`` (full strength, where the target is ``L_l``
    itself) for every sample in ``z`` and hands the ``(B, P, 3)`` block to
    :func:`q3vl.whatb.guards.assert_transform_not_degenerate`, which exits the
    process (``SystemExit(2)``) on a flat / identity / sample-invariant transform.
    """
    with torch.no_grad():
        th = arm.theta(z) if params is None else params
        y = arm.transform(th, x, s)
    return assert_transform_not_degenerate(
        y, x, thresholds=thresholds or DegeneracyThresholds(), where=where,
        exit_process=exit_process,
        extra={"arm": ARM, "mode": arm.cfg.mode, "s": s})


def step_columns(loss_level: int = 1, *, mode: str = "A3", r_line: bool = True,
                 extra: Sequence[str] = ()) -> tuple[str, ...]:
    """The columns this arm promises on the FIRST line of ``steps.jsonl``.

    The frozen seven (``L_rec`` / ``L_hc`` / ``L_sparse`` / ``n_colors`` /
    ``n_luts_in_batch`` / ``mining_ratio`` / ``n_hc_masked``), plus ``L_img`` at
    ``--loss-level 4``, plus ``R_line`` when it is on, plus R1 §10's new columns
    for ``mode``.
    """
    from q3vl.whatb.publish import step_columns_for

    own = list(step_extra_columns(mode))
    if r_line:
        own.append("R_line")
    return step_columns_for(loss_level, extra=tuple(own) + tuple(extra))


def assert_step_row(*, steps_row: Mapping[str, Any] | None = None,
                    steps_path: Any = None, loss_level: int = 1,
                    mode: str = "A3", r_line: bool = True,
                    extra: Sequence[str] = (),
                    forbidden: Sequence[str] = OFF_BY_DEFAULT_LOSS_COLUMNS,
                    ) -> tuple[dict[str, Any], str]:
    """Board-time step-row assertion that **fetches its own data**.

    Three tiers -- caller, ``steps.jsonl`` first line, in-process witness -- via
    :func:`q3vl.whatb.guards.assert_first_step_columns`, so a call site that was
    handed nothing still reaches the row.  "Nobody handed me a row" and "the row
    has no loss columns" are different exceptions.

    Additionally: any key in ``forbidden`` present on the row is an error.
    """
    want = step_columns(loss_level, mode=mode, r_line=r_line, extra=extra)
    row, source = assert_first_step_columns(want, steps_row=steps_row,
                                            steps_path=steps_path)
    present = [k for k in forbidden if k in row and row[k] is not None]
    if present:
        raise ValueError(
            f"first step row (source={source}) carries {present}, which this "
            "configuration switched off; a term that is off must not be computed")
    return row, source


def required_criteria(*, extra: Sequence[str] = ()) -> list[str]:
    """The keys this arm may not publish a board without.

    The twelve frozen keys + the P2 block (``loc_*``, ``field_gt/const/shuffle``)
    from the shared table, plus this arm's own additions: ``field_pred``, the five
    ``grid_s*`` columns of the s-axis and the ``headline_alpha_inside_only``
    diagnostic.
    """
    return sorted(set(_base_required_criteria(ARM)) | set(ARM_EXTRA_CRITERIA) | set(extra))


def loss_preregistration(args: Any = None) -> dict[str, Any]:
    """What ``config/loss_preregistration.json`` says for EPR-028 R1."""
    get = (lambda name, default: getattr(args, name, default)) if args is not None \
        else (lambda name, default: default)
    mode = str(get("glut4d_mode", "A3"))
    w_img = float(get("w_img", 0.0))
    a_s = float(get("alpha_s", 0.0))
    lam_line = float(get("lam_line", LAMBDA_LINE))
    level = get("loss_level", None)
    level = int(level) if level is not None else (4 if w_img else 1)
    lam_hc = effective_lambda_hc(float(get("lam_hc", LAMBDA_HC)), level)
    lam_sparse = effective_lambda_sparse(float(get("lam_sparse", LAMBDA_SPARSE)),
                                         level)
    return {
        "arm": ARM,
        "axes": list(ARM_AXES),
        "mode": mode,
        "loss_level": level,
        "loss_ladder": pure_l1_record(loss_level=level, lambda_hc=lam_hc,
                                      lambda_sparse=lam_sparse),
        "total": "L_rec"
                 + (" + lam_line * R_line" if lam_line else "")
                 + (" + lam_hc * L_hc" if lam_hc else "")
                 + (" + lam_sparse * R_sparse" if lam_sparse else "")
                 + (" + alpha_s * L_s4d" if a_s else "")
                 + (" + w_img * L_img" if w_img else ""),
        "terms": {
            "L_rec": {"formula": "|| f_theta(x, s) - ((1-s) x + s L_l(x)) ||_1",
                      "weight": 1.0, "source": "GLUT Eq.6 with the R1 §4.4 target",
                      "novel": "target (1-s)x + s L_l(x) -- the data law rendering.py:311"},
            "R_line": {"formula": "|| f(x,s) - [(1-s) f(x,0) + s f(x,1)] ||_1",
                       "weight": lam_line, "enabled": bool(lam_line),
                       "source": "R1 §4.4, pre-registered 0.1 (ablation 0 / 0.1 / 1.0)",
                       "replaces": "R0's L_m4d, deleted: d y*_c/d s = L_c(x) - x_c "
                                   "is legitimately negative"},
            "L_hc": {"formula": "C * (1 - <h_hat, h>) * 1[C >= 1e-3]",
                     "weight": lam_hc,
                     "source": "GLUT Eq.7 + §4.1 lambda_hc = 10; C->0 mask = frozen block"},
            "R_sparse": {"formula": "-mean_i [o_i log(o_i+eps) + (1-o_i) log(1-o_i+eps)]",
                         "weight": lam_sparse,
                         "source": "GLUT Eq.8 + §4.1 lambda_sparse = 0.001",
                         "acts_on": "o_i (s-independent), NOT o_i(s)"},
            "L_s4d": {"formula": "sum over 4 axes of squared first differences",
                      "weight": a_s, "enabled": bool(a_s),
                      "source": "4D LUT Eq.12 + §IV-A alpha_s = 1e-4 (ablation row); "
                                "R1 §8.3: 256 random 4D points per sample, never 17^4"},
            "L_img": {"formula": "|| compose_headline(I, S, f) - I* ||_1",
                      "weight": w_img, "enabled": bool(w_img),
                      "source": "NOVEL weight 1.0; GLUT trains in function space only"},
        },
        "eps": EPS, "eps_chroma": EPS_CHROMA,
        "tau_min": TAU_MIN, "tau_max": TAU_MAX,
        "guard_columns": list(GUARD_COLUMNS),
        "first_row_columns": list(step_columns(level, mode=mode,
                                               r_line=bool(lam_line))),
        "off_by_default": [k for k, on in
                           (("L_s4d", a_s), ("L_img", w_img)) if not on],
    }
