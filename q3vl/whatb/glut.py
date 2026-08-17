"""GLUT Eq.1-5: the ONE colour-transform carrier the EPR-024..029 arms share.

Authority for every number in this file, in order:

1. the cross-arm frozen block (six ``PROPOSAL.md`` copies, byte-identical),
2. ``docs/HANDOFF_whatb_2026-08-15.md`` section 6 (parameterisation + the
   13-row line table of the official demo),
3. the official demo itself --
   ``https://color.cvc.uab.cat/assets/html/glut_editor.html``, sha256
   ``863bb1cbb3a22d8a162929db52f44c9c310353a908238115fe1e4b05088147c2``,
   1217 lines, fetched 2026-08-15 over a *fully validated* chain (the site
   serves no intermediate; the AIA cert ``http://crt.harica.gr/HARICA-GEANT-TLS-R1.cer``
   has to be appended to the CA bundle -- ``-k`` is NOT used).

The demo is the only executable official implementation and it ships seven
trained GLUT-32 weight sets, so this file is verified by point-wise parity
against it (``tests/test_glut_demo_parity.py``), not by reading alone.

Parameterisation (22N + 12)
---------------------------
per Gaussian ``i``: ``mu_i`` (3) + Cholesky ``(diag 3, off 3)`` (6) +
``opacity logit`` (1) + ``M_i`` (9) + ``b_i`` (3) = 22; plus one global affine
``G`` (9) + ``g`` (3) = 12.  ``N = 48`` -> 1068.  N=48 is a project value, NOT
a paper value (paper Table 9 grid is 8/16/32/64/128, default 32).

Forward (paper Eq.1-5, with the demo's numerics)::

    L_i      = [[softplus(d0),0,0],[o0,softplus(d1),0],[o1,o2,softplus(d2)]]  demo :457-465
    Sigma_i  = L_i L_i^T ;  Sigma_i[k,k] += eps                                demo :520-522
    det_i    = det3(Sigma_i)                                                   demo :481-486
    Prec_i   = |det_i| < eps ? I : inv3(Sigma_i)                               demo :496
    logdet_i = log(max(det_i, eps))                                            demo :526
    o_i      = sigmoid(opacity_logit_i)                                        demo :527
    d_i(x)   = (x-mu_i)^T Prec_i (x-mu_i)                                      Eq.1
    logp_i   = -0.5 * (d_i + logdet_i + 3*log(2 pi)) ; p_i = exp(logp_i)       demo :544-545
    w_i(x)   = p_i o_i / (sum_j p_j o_j + eps)                                 Eq.2 / demo :565
    local    = sum_i w_i (M_i x + b_i)                                         Eq.3
    glob     = G x + g ; if clamp == "two": glob = clamp(glob, 0, 1)           Eq.4 / demo :574-579
    out      = glob + local            (residual, demo :596-604 -- all seven
                                        embedded models carry residual=true)
    y        = clamp(out, 0, 1)                                                Eq.5 / demo :606-610

``eps = 1e-6`` (demo :441).  ``--clamp {two,one}`` defaults to ``two`` (frozen
block); ``"none"`` is an *internal* mode with no CLI spelling -- it returns the
pre-clamp value the propositions and EPR-027's post-gate clamp need.

``clamp_grad {hard,st}`` (EPR-030) selects the clamp's **backward** only: the
forward is the same digit either way.  ``"hard"`` is ``x.clamp(0,1)`` and hands
back exactly zero gradient at a saturated point; ``"st"`` is the same
``x.clamp(0,1)`` forward wrapped in an ``autograd.Function`` whose backward is
the identity.  The module default is ``"hard"`` so EPR-024..029 keep the
gradients they ran with.

Discipline this module enforces (each one paid for on the where side)
---------------------------------------------------------------------
* **device/dtype**: everything entering the maths is explicitly moved with
  ``.to(device=..., dtype=...)`` onto one resolved (device, compute dtype).
  The dtype is *promoted*, never silently truncated, and it is floated to at
  least float32 -- a bf16 ``exp(logpdf)`` is not a thing this carrier does.
  (Where-side EPR-022/MATTE died because a CPU tensor reached an autocast
  region; here a stray device raises a named error instead.)
* **no bare ``torch.tensor(...)`` in a module forward**: :class:`GlutCarrier`
  keeps its only constant, the 3x3 identity, in a non-persistent buffer.
* **no ``.cpu()``**: every quantity is produced on the input device.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "EPS",
    "LOG_2PI",
    "CLAMP_FLAG_CHOICES",
    "CLAMP_GRAD_CHOICES",
    "ClampMode",
    "ClampGrad",
    "GlutParams",
    "GlutAux",
    "GlutCarrier",
    "glut_forward",
    "glut_geometry",
    "n_params_glut",
    "uniform_grid_positions",
    "grid_axis_sizes",
    "softplus_inverse",
]

#: demo :441 -- the single epsilon of Eq.2, Eq.8, the covariance jitter and the
#: determinant fallback.  One constant, four uses, exactly as the demo has it.
EPS: float = 1e-6

#: demo's ``log_2pi`` constant (also shipped inside every embedded model).
LOG_2PI: float = math.log(2.0 * math.pi)

ClampMode = Literal["two", "one", "none"]

_CLAMP_MODES: tuple[str, ...] = ("two", "one", "none")
#: what ``--clamp`` may spell.  ``"none"`` is internal-only on purpose: a run
#: that publishes headline numbers with no clamp at all is not a frozen-block run.
CLAMP_FLAG_CHOICES: tuple[str, ...] = ("two", "one")

ClampGrad = Literal["hard", "st"]

_CLAMP_GRADS: tuple[str, ...] = ("hard", "st")
#: what ``--clamp-grad`` may spell (EPR-030).  ``"hard"`` is ``x.clamp(0,1)``:
#: a saturated point returns **exactly zero** gradient.  ``"st"`` is the
#: straight-through estimator :class:`_STClamp01` -- the forward *is* the hard
#: clamp (bit-for-bit on every input), only the backward becomes the identity.
#:
#: The module default stays ``"hard"``: EPR-024..029 all ran under it and their
#: gradients must not move retroactively.  EPR-030's entry point defaults its own
#: arm to ``"st"`` and records the value in ``run_setup.json``.
CLAMP_GRAD_CHOICES: tuple[str, ...] = ("st", "hard")


class _STClamp01(torch.autograd.Function):
    """``clamp(x, 0, 1)`` forward, identity backward.

    The arithmetic spelling ``x + (clamp(x) - x).detach()`` is only equal to
    ``clamp(x, 0, 1)`` while ``x`` is small enough for ``x - x`` to be exact in
    the working dtype.  Measured in fp32: ``x = 2**25`` and ``x = 1e12`` both
    return ``0.0`` instead of ``1.0``, and ``x = inf`` returns ``nan``
    (``inf + (1 - inf) == nan``).  The global branch of this carrier does
    overflow in practice (NOTES §3: ``--qdec-head-lr-scale 1.0`` reaches NaN at
    step 80), so the forward is written as the clamp itself and only the
    backward is replaced.
    """

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:          # type: ignore[override]
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g: Tensor):                   # type: ignore[override]
        return g


def _clamp01(x: Tensor, grad: str) -> Tensor:
    """``clamp(x, 0, 1)`` with either the hard or the straight-through backward.

    ``forward`` is the same digit in both modes -- bit-for-bit, for every input
    including ``2**25``, ``1e12``, ``+-inf`` and ``nan`` -- and ``"st"`` only
    replaces ``d clamp / dx = 1[0 <= x <= 1]`` with the constant 1.  Measured
    motivation (HANDOFF 2026-08-15 §1.2): from step 50 on, 100% of the query
    points of the global branch were saturated, so the branch returned an
    exactly-zero gradient and died.
    """
    if grad == "st":
        return _STClamp01.apply(x)
    return x.clamp(0.0, 1.0)


def softplus_inverse(y: float) -> float:
    """``x`` such that ``softplus(x) == y``.  ``sigma = 0.15 -> -1.8212...``."""
    if y <= 0.0:
        raise ValueError(f"softplus is positive; got target {y!r}")
    return float(math.log(math.expm1(y)))


def n_params_glut(n_gauss: int) -> int:
    """``22N + 12`` -- the carrier's parameter count (N=48 -> 1068, N=32 -> 716)."""
    return 22 * int(n_gauss) + 12


def _factor_triple(n: int) -> tuple[int, int, int]:
    """The most cube-like ``(a, b, c)`` with ``a*b*c == n``, descending.

    48 -> (4, 4, 3) (the ``uniform_grid_4x4x3`` of EPR-025:372); 32 -> (4, 4, 2);
    64 -> (4, 4, 4).  Tie-break: smallest spread, then smallest maximum.
    """
    best: tuple[int, int, int] | None = None
    for a in range(1, n + 1):
        if n % a:
            continue
        m = n // a
        for b in range(a, m + 1):
            if m % b:
                continue
            c = m // b
            if c < b:
                continue
            cand = (a, b, c)
            if best is None or (cand[2] - cand[0], cand[2]) < (best[2] - best[0], best[2]):
                best = cand
    assert best is not None
    return (best[2], best[1], best[0])


def grid_axis_sizes(n_gauss: int) -> tuple[int, int, int]:
    """``(n_R, n_G, n_B)`` of :func:`uniform_grid_positions` -- 48 -> ``(4, 4, 3)``.

    Public because EPR-030's query decoder needs the *same* factorisation to
    index its per-axis colour positional encodings: ``q_i`` gets
    ``PE_R[r_i] + PE_G[g_i] + PE_B[b_i]`` with ``(r_i, g_i, b_i)`` the grid
    index of the Gaussian whose ``mu`` init is ``uniform_grid_positions(N)[i]``.
    Reading the factorisation off a private helper is how two spellings of the
    same grid start to drift.
    """
    return _factor_triple(int(n_gauss))


def uniform_grid_positions(
    n_gauss: int, *, device: torch.device | str | None = None, dtype: torch.dtype = torch.float32
) -> Tensor:
    """``(N, 3)`` cell-centre grid in the unit RGB cube -- GLUT App A.1's init.

    "Gaussian means are distributed on a uniform regular grid in the [0,1]^3 RGB
    cube."  Axis extents are the most cube-like factorisation of ``N`` in
    descending order, so N=48 is 4x4x3 over (R, G, B) exactly as EPR-025:372
    spells it, and centres sit at ``(i + 0.5) / a``.
    """
    a, b, c = _factor_triple(int(n_gauss))
    axes = [
        (torch.arange(k, device=device, dtype=dtype) + 0.5) / k for k in (a, b, c)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    return grid.reshape(-1, 3).contiguous()


# --------------------------------------------------------------------------- #
# parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GlutParams:
    """One batch of GLUT parameter sets.  All tensors share ``(device, dtype)``.

    ==================  =================  ==================================
    field               shape              meaning
    ==================  =================  ==================================
    ``mu``              ``(B, N, 3)``      Gaussian means, unconstrained
    ``chol_diag``       ``(B, N, 3)``      RAW diagonal, softplus applied here
    ``chol_off``        ``(B, N, 3)``      ``L[1,0], L[2,0], L[2,1]``, raw
    ``opacity_logit``   ``(B, N)``         RAW logit, sigmoid applied here
    ``m_local``         ``(B, N, 3, 3)``   per-Gaussian ``M_i``
    ``b_local``         ``(B, N, 3)``      per-Gaussian ``b_i``
    ``g_matrix``        ``(B, 3, 3)``      global ``G``
    ``g_bias``          ``(B, 3)``         global ``g``
    ==================  =================  ==================================

    "RAW" means pre-activation: the carrier owns softplus/sigmoid so the demo's
    storage convention (``cholesky_diag``, ``opacities_logit``) is the only one
    in the codebase and no arm can apply an activation twice.
    """

    mu: Tensor
    chol_diag: Tensor
    chol_off: Tensor
    opacity_logit: Tensor
    m_local: Tensor
    b_local: Tensor
    g_matrix: Tensor
    g_bias: Tensor

    # ---- shape / device contract ----
    def __post_init__(self) -> None:
        b, n = self.mu.shape[0], self.mu.shape[1]
        want = {
            "mu": (b, n, 3),
            "chol_diag": (b, n, 3),
            "chol_off": (b, n, 3),
            "opacity_logit": (b, n),
            "m_local": (b, n, 3, 3),
            "b_local": (b, n, 3),
            "g_matrix": (b, 3, 3),
            "g_bias": (b, 3),
        }
        for name, shape in want.items():
            got = tuple(getattr(self, name).shape)
            if got != shape:
                raise ValueError(f"GlutParams.{name}: expected {shape}, got {got}")
        devs = {name: getattr(self, name).device for name in want}
        if len(set(devs.values())) != 1:
            raise ValueError(
                "GlutParams tensors straddle devices -- this is the EPR-022/MATTE "
                f"failure mode, refusing to guess: {devs}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.mu.shape[0])

    @property
    def n_gauss(self) -> int:
        return int(self.mu.shape[1])

    @property
    def device(self) -> torch.device:
        return self.mu.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mu.dtype

    def to(self, *, device: torch.device | str | None = None, dtype: torch.dtype | None = None) -> GlutParams:
        """Explicit move.  Keyword-only so a positional dtype can never be a device."""
        f = lambda t: t.to(device=device if device is not None else t.device,
                           dtype=dtype if dtype is not None else t.dtype)
        return GlutParams(*(f(getattr(self, k)) for k in _FIELDS))

    def to_ref(self, ref: Tensor) -> GlutParams:
        """Move onto ``ref``'s ``(device, dtype)`` -- the pitfall-2 one-liner."""
        return self.to(device=ref.device, dtype=ref.dtype)

    def detach(self) -> GlutParams:
        return GlutParams(*(getattr(self, k).detach() for k in _FIELDS))

    def expand_batch(self, batch: int) -> GlutParams:
        """Broadcast a ``B == 1`` set to ``B == batch`` (no copy)."""
        if self.batch_size == batch:
            return self
        if self.batch_size != 1:
            raise ValueError(f"cannot expand batch {self.batch_size} -> {batch}")
        return GlutParams(*(getattr(self, k).expand(batch, *getattr(self, k).shape[1:])
                            for k in _FIELDS))

    # ---- flat views: the interpolation算术 of propositions 1 and 3 ----
    def flat(self) -> Tensor:
        """``(B, 22N + 12)``: ``[mu, chol_diag, chol_off, logit, M, b]`` per
        Gaussian (22 each, in that order), then ``[G, g]`` (12)."""
        b, n = self.batch_size, self.n_gauss
        per = torch.cat(
            [
                self.mu,
                self.chol_diag,
                self.chol_off,
                self.opacity_logit.unsqueeze(-1),
                self.m_local.reshape(b, n, 9),
                self.b_local,
            ],
            dim=-1,
        )
        return torch.cat([per.reshape(b, n * 22), self.global_flat()], dim=-1)

    def global_flat(self) -> Tensor:
        """``(B, 12)`` -- ``[G (row-major 9), g (3)]``."""
        return torch.cat([self.g_matrix.reshape(-1, 9), self.g_bias], dim=-1)

    def affine_flat(self) -> Tensor:
        """``theta_gen`` of proposition 1: ``(B, 12N + 12)`` = ``[M_i, b_i]_i`` then ``[G, g]``."""
        b, n = self.batch_size, self.n_gauss
        per = torch.cat([self.m_local.reshape(b, n, 9), self.b_local], dim=-1)
        return torch.cat([per.reshape(b, n * 12), self.global_flat()], dim=-1)

    def with_affine_flat(self, vec: Tensor) -> GlutParams:
        """Replace ``{M, b, G, g}`` from a ``(B, 12N + 12)`` vector; geometry kept."""
        n = self.n_gauss
        if vec.shape[-1] != 12 * n + 12:
            raise ValueError(f"affine_flat expects (B, {12 * n + 12}), got {tuple(vec.shape)}")
        b = vec.shape[0]
        per = vec[:, : n * 12].reshape(b, n, 12)
        glob = vec[:, n * 12 :]
        return replace(
            self,
            m_local=per[..., :9].reshape(b, n, 3, 3),
            b_local=per[..., 9:],
            g_matrix=glob[:, :9].reshape(b, 3, 3),
            g_bias=glob[:, 9:],
        )

    @staticmethod
    def from_flat(vec: Tensor, n_gauss: int) -> GlutParams:
        """Inverse of :meth:`flat`."""
        n = int(n_gauss)
        if vec.shape[-1] != 22 * n + 12:
            raise ValueError(f"from_flat expects (B, {22 * n + 12}), got {tuple(vec.shape)}")
        b = vec.shape[0]
        per = vec[:, : n * 22].reshape(b, n, 22)
        glob = vec[:, n * 22 :]
        return GlutParams(
            mu=per[..., 0:3],
            chol_diag=per[..., 3:6],
            chol_off=per[..., 6:9],
            opacity_logit=per[..., 9],
            m_local=per[..., 10:19].reshape(b, n, 3, 3),
            b_local=per[..., 19:22],
            g_matrix=glob[:, :9].reshape(b, 3, 3),
            g_bias=glob[:, 9:],
        )

    @staticmethod
    def identity(
        n_gauss: int,
        *,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        sigma: float = 0.15,
        opacity_logit: float = 4.0,
    ) -> GlutParams:
        """Proposition 2's exact point: ``M_i = I``, ``b_i = 0``, ``G = 0``, ``g = 0``.

        Geometry follows GLUT App A.1 (grid means, isotropic ``sigma = 0.15``);
        it is irrelevant to the value of ``f`` -- with ``eps = 0`` the weights
        sum to 1 whatever it is -- but it keeps the object well-formed and the
        degenerate-weight rate meaningful.
        """
        n = int(n_gauss)
        mu = uniform_grid_positions(n, device=device, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1)
        raw = softplus_inverse(sigma)
        eye = torch.eye(3, device=device, dtype=dtype)
        return GlutParams(
            mu=mu,
            chol_diag=torch.full((batch, n, 3), raw, device=device, dtype=dtype),
            chol_off=torch.zeros((batch, n, 3), device=device, dtype=dtype),
            opacity_logit=torch.full((batch, n), opacity_logit, device=device, dtype=dtype),
            m_local=eye.expand(batch, n, 3, 3).clone(),
            b_local=torch.zeros((batch, n, 3), device=device, dtype=dtype),
            g_matrix=torch.zeros((batch, 3, 3), device=device, dtype=dtype),
            g_bias=torch.zeros((batch, 3), device=device, dtype=dtype),
        )


_FIELDS: tuple[str, ...] = (
    "mu",
    "chol_diag",
    "chol_off",
    "opacity_logit",
    "m_local",
    "b_local",
    "g_matrix",
    "g_bias",
)


@dataclass(frozen=True)
class GlutAux:
    """Diagnostics the criteria need.  Every tensor stays on the input device.

    ``weights``      ``(B, P, N)``  Eq.2 ``w_i(x)``
    ``influence_sum````(B, P)``     ``sum_j p_j o_j`` BEFORE ``+ eps`` -- the
                                    criterion "degenerate-weight rate"
                                    ``Pr[sum_j p_j o_j < tau]`` reads this one.
    ``pre_clamp``    ``(B, P, 3)``  ``glob + local`` before the final clamp;
                                    the out-of-gamut rate ``A(alpha)`` is
                                    ``(pre_clamp outside [0,1]).any(-1)``.
    ``opacity``      ``(B, N)``     ``sigmoid(logit)``; ``R_sparse`` reads it.
    ``logdet``       ``(B, N)``     ``log(max(det Sigma, eps))``
    ``degenerate_precision`` ``(B, N)`` bool: the ``|det| < eps -> I`` fallback fired.
    """

    weights: Tensor
    influence_sum: Tensor
    pre_clamp: Tensor
    opacity: Tensor
    logdet: Tensor
    degenerate_precision: Tensor

    def oob_mask(self, lo: float = 0.0, hi: float = 1.0) -> Tensor:
        """``(B, P)`` bool -- pre-clamp value outside the gamut on any channel."""
        return ((self.pre_clamp < lo) | (self.pre_clamp > hi)).any(dim=-1)

    def degenerate_weight_mask(self, tau: float) -> Tensor:
        """``(B, P)`` bool -- ``sum_j p_j o_j < tau`` (proposition 2's ``delta``)."""
        return self.influence_sum < tau


# --------------------------------------------------------------------------- #
# device / dtype resolution
# --------------------------------------------------------------------------- #
def _resolve_device_dtype(
    tensors: dict[str, Tensor], *, compute_dtype: torch.dtype | None
) -> tuple[torch.device, torch.dtype]:
    devices = {name: t.device for name, t in tensors.items()}
    uniq = set(devices.values())
    if len(uniq) != 1:
        raise ValueError(
            "glut_forward got tensors on more than one device; the caller must "
            "place them (`.to(device=ref.device, dtype=ref.dtype)`) rather than "
            f"let autocast decide: {devices}"
        )
    device = next(iter(uniq))
    if compute_dtype is not None:
        return device, compute_dtype
    dtype = torch.float32
    for t in tensors.values():
        if not t.is_floating_point():
            raise TypeError("glut_forward takes floating point tensors only")
        dtype = torch.promote_types(dtype, t.dtype)
    # promote_types(float32, bfloat16) is float32 already; the max() guards the
    # hypothetical future where it is not.  No silent downcast, ever.
    if dtype in (torch.float16, torch.bfloat16):
        dtype = torch.float32
    return device, dtype


# --------------------------------------------------------------------------- #
# geometry (Eq.1 preliminaries) -- verbatim demo numerics
# --------------------------------------------------------------------------- #
def _det3(a: Tensor) -> Tensor:
    """``det3x3`` of demo :481-486, cofactor-by-cofactor (not ``linalg.det``)."""
    return (
        a[..., 0, 0] * (a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1])
        - a[..., 0, 1] * (a[..., 1, 0] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 0])
        + a[..., 0, 2] * (a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0])
    )


def _inv3(a: Tensor, det: Tensor) -> Tensor:
    """``inverse3x3`` of demo :489-509 -- explicit adjugate, same term order."""
    inv_det = 1.0 / det
    r = torch.stack(
        [
            a[..., 1, 1] * a[..., 2, 2] - a[..., 1, 2] * a[..., 2, 1],
            a[..., 0, 2] * a[..., 2, 1] - a[..., 0, 1] * a[..., 2, 2],
            a[..., 0, 1] * a[..., 1, 2] - a[..., 0, 2] * a[..., 1, 1],
            a[..., 1, 2] * a[..., 2, 0] - a[..., 1, 0] * a[..., 2, 2],
            a[..., 0, 0] * a[..., 2, 2] - a[..., 0, 2] * a[..., 2, 0],
            a[..., 0, 2] * a[..., 1, 0] - a[..., 0, 0] * a[..., 1, 2],
            a[..., 1, 0] * a[..., 2, 1] - a[..., 1, 1] * a[..., 2, 0],
            a[..., 0, 1] * a[..., 2, 0] - a[..., 0, 0] * a[..., 2, 1],
            a[..., 0, 0] * a[..., 1, 1] - a[..., 0, 1] * a[..., 1, 0],
        ],
        dim=-1,
    )
    return (inv_det.unsqueeze(-1) * r).reshape(*a.shape[:-2], 3, 3)


def glut_geometry(
    chol_diag: Tensor,
    chol_off: Tensor,
    opacity_logit: Tensor,
    *,
    eps: float = EPS,
    eye: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """``(precision, logdet, opacity, degenerate_mask)`` from raw Cholesky + logit.

    Shapes: ``chol_* (..., N, 3)``, ``opacity_logit (..., N)`` ->
    ``precision (..., N, 3, 3)``, ``logdet (..., N)``, ``opacity (..., N)``,
    ``degenerate_mask (..., N)`` bool.

    Split out because EPR-025's affine-only head calls it **once per step** on
    condition-independent ``nn.Parameter`` geometry and reuses the result for
    the whole batch (that reuse is the computational content of proposition 1).
    """
    diag = F.softplus(chol_diag, beta=1.0, threshold=20.0)  # demo :446-449, x>20 -> x
    zero = torch.zeros_like(diag[..., 0])
    rows = [
        torch.stack([diag[..., 0], zero, zero], dim=-1),
        torch.stack([chol_off[..., 0], diag[..., 1], zero], dim=-1),
        torch.stack([chol_off[..., 1], chol_off[..., 2], diag[..., 2]], dim=-1),
    ]
    lower = torch.stack(rows, dim=-2)  # (..., N, 3, 3), demo :457-465
    sigma = lower @ lower.transpose(-1, -2)
    if eye is None:
        eye = torch.eye(3, device=sigma.device, dtype=sigma.dtype)
    sigma = sigma + eps * eye  # demo :520-522, jitter AFTER L L^T
    det = _det3(sigma)
    degenerate = det.abs() < eps  # demo :496
    precision = torch.where(
        degenerate[..., None, None], eye.expand_as(sigma), _inv3(sigma, torch.where(degenerate, torch.ones_like(det), det))
    )
    logdet = torch.log(det.clamp_min(eps))  # demo :526
    opacity = torch.sigmoid(opacity_logit)  # demo :527
    return precision, logdet, opacity, degenerate


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
def _auto_chunk(n_points: int, n_gauss: int) -> int:
    """Point-block size keeping the ``(B, P, N, 3)`` difference tensor bounded.

    Chunking changes nothing numerically -- points are independent -- it only
    keeps a full 512x640 image apply from allocating a 566 MB intermediate.
    """
    budget = 1 << 22
    return max(1, min(n_points, budget // max(1, n_gauss)))


def glut_forward(
    x: Tensor,
    params: GlutParams,
    *,
    clamp: ClampMode = "two",
    clamp_grad: ClampGrad = "hard",
    residual: bool = True,
    eps: float = EPS,
    compute_dtype: torch.dtype | None = None,
    point_chunk: int | None = None,
    return_aux: bool = False,
    _geometry: tuple[Tensor, Tensor, Tensor, Tensor] | None = None,
) -> Tensor | tuple[Tensor, GlutAux]:
    """Evaluate ``f_theta`` on colour queries.  Differentiable, batched.

    Parameters
    ----------
    x
        ``(B, *S, 3)`` or ``(*S, 3)``; the latter is broadcast to every batch
        element.  Values are sRGB in ``[0, 1]`` (nothing enforces it -- GLUT
        means themselves leave the cube in the trained demo weights).
    params
        :class:`GlutParams` with batch ``B`` or ``1``.
    clamp
        ``"two"`` (frozen default, demo :574-579 + :606-610), ``"one"`` (paper
        Eq.4/5, final clamp only), ``"none"`` (no clamp -- pre-clamp value, for
        the propositions and for EPR-027 which clamps after its gate).
    clamp_grad
        ``"hard"`` (default, EPR-024..029's behaviour) or ``"st"``
        (straight-through: same forward digit for digit, identity backward).
        Applies to **both** clamp sites when ``clamp == "two"``.
    residual
        ``True`` = Eq.5 / demo ``residual: true`` (all seven embedded models).
        ``False`` reproduces the paper's "w/o Global" ablation.
    compute_dtype
        Force the maths dtype.  ``None`` promotes across inputs and floors at
        float32; it never truncates.
    point_chunk
        Points per block; ``None`` picks one from ``N``.  Numerically inert.
    return_aux
        Also return :class:`GlutAux` (weights, influence sum, pre-clamp value,
        opacities, logdets, degenerate-precision mask).

    Returns
    -------
    ``(B, *S, 3)`` -- and :class:`GlutAux` when ``return_aux``.
    """
    if clamp not in _CLAMP_MODES:
        raise ValueError(f"clamp must be one of {_CLAMP_MODES}, got {clamp!r}")
    if clamp_grad not in _CLAMP_GRADS:
        raise ValueError(f"clamp_grad must be one of {_CLAMP_GRADS}, got {clamp_grad!r}")
    if x.shape[-1] != 3:
        raise ValueError(f"x must end in a channel axis of 3, got {tuple(x.shape)}")

    named = {"x": x, **{k: getattr(params, k) for k in _FIELDS}}
    device, dtype = _resolve_device_dtype(named, compute_dtype=compute_dtype)

    # Casting the inputs is not enough: `einsum` is on autocast's low-precision
    # list, so an enclosing `autocast` region would pull the Mahalanobis form
    # and `exp(logpdf)` back down to bf16 behind the caller's back.  The carrier
    # opts out for its own maths; it is 8192x48 Gaussian evaluations per step,
    # the cost is noise and the failure mode is not.
    with _no_autocast(device):
        return _glut_forward_impl(
            x, params, device, dtype, clamp=clamp, clamp_grad=clamp_grad,
            residual=residual, eps=eps,
            point_chunk=point_chunk, return_aux=return_aux, _geometry=_geometry,
        )


def _no_autocast(device: torch.device):
    kind = device.type
    if kind not in ("cpu", "cuda", "xpu"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=kind, enabled=False)


def _glut_forward_impl(
    x: Tensor,
    params: GlutParams,
    device: torch.device,
    dtype: torch.dtype,
    *,
    clamp: ClampMode,
    clamp_grad: ClampGrad,
    residual: bool,
    eps: float,
    point_chunk: int | None,
    return_aux: bool,
    _geometry: tuple[Tensor, Tensor, Tensor, Tensor] | None,
) -> Tensor | tuple[Tensor, GlutAux]:
    x_c = x.to(device=device, dtype=dtype)
    p = params.to(device=device, dtype=dtype)

    # x is (B, *S, 3) when it has 3+ axes, else (*S, 3) shared by the whole batch.
    if x_c.dim() >= 3:
        xb = int(x_c.shape[0])
        if xb not in (1, p.batch_size) and p.batch_size != 1:
            raise ValueError(f"x batch {xb} incompatible with params batch {p.batch_size}")
        lead = max(xb, p.batch_size)
        spatial = tuple(x_c.shape[1:-1])
        xf = x_c.reshape(xb, -1, 3)
        if xb != lead:
            xf = xf.expand(lead, -1, 3)
    else:
        lead = p.batch_size
        spatial = tuple(x_c.shape[:-1])
        xf = x_c.reshape(1, -1, 3).expand(lead, -1, 3)
    p = p.expand_batch(lead)

    n_pts, n_g = xf.shape[1], p.n_gauss
    eye = torch.eye(3, device=device, dtype=dtype)
    if _geometry is None:
        precision, logdet, opacity, degen = glut_geometry(
            p.chol_diag, p.chol_off, p.opacity_logit, eps=eps, eye=eye
        )
    else:
        precision, logdet, opacity, degen = (
            t.to(device=device, dtype=dtype) if t.is_floating_point() else t.to(device=device)
            for t in _geometry
        )
        # Condition-independent geometry (EPR-025) arrives unbatched as
        # (N,3,3)/(N,)/(N,)/(N,); give it the batch axis the maths expects.
        if logdet.dim() == 1:
            precision = precision.unsqueeze(0).expand(lead, -1, 3, 3)
            logdet = logdet.unsqueeze(0).expand(lead, -1)
            opacity = opacity.unsqueeze(0).expand(lead, -1)
            degen = degen.unsqueeze(0).expand(lead, -1)

    chunk = int(point_chunk) if point_chunk else _auto_chunk(n_pts, n_g)
    outs: list[Tensor] = []
    aux_w: list[Tensor] = []
    aux_s: list[Tensor] = []
    aux_pre: list[Tensor] = []

    for start in range(0, max(n_pts, 1), chunk):
        xs = xf[:, start : start + chunk, :]                        # (B, p, 3)
        if xs.shape[1] == 0:
            continue
        diff = xs.unsqueeze(2) - p.mu.unsqueeze(1)                  # (B, p, N, 3)
        mahal = torch.einsum("bpni,bnij,bpnj->bpn", diff, precision, diff)   # Eq.1
        logp = -0.5 * (mahal + logdet.unsqueeze(1) + 3.0 * LOG_2PI)  # demo :544-545
        influence = torch.exp(logp) * opacity.unsqueeze(1)          # p_i * o_i
        infl_sum = influence.sum(dim=-1)                            # sum_j p_j o_j
        w = influence / (infl_sum.unsqueeze(-1) + eps)              # Eq.2 / demo :565
        local = torch.einsum("bpn,bnij,bpj->bpi", w, p.m_local, xs) + torch.einsum(
            "bpn,bni->bpi", w, p.b_local
        )                                                            # Eq.3 mixed
        glob = torch.einsum("bij,bpj->bpi", p.g_matrix, xs) + p.g_bias.unsqueeze(1)  # Eq.4
        if clamp == "two":
            glob = _clamp01(glob, clamp_grad)                       # demo :574-579
        pre = glob + local if residual else local                   # demo :596-604 / Eq.5
        y = pre if clamp == "none" else _clamp01(pre, clamp_grad)   # demo :606-610
        outs.append(y)
        if return_aux:
            aux_w.append(w)
            aux_s.append(infl_sum)
            aux_pre.append(pre)

    yf = torch.cat(outs, dim=1) if outs else xf.new_zeros((lead, 0, 3))
    out = yf.reshape(lead, *spatial, 3)
    if not return_aux:
        return out
    aux = GlutAux(
        weights=torch.cat(aux_w, dim=1) if aux_w else xf.new_zeros((lead, 0, n_g)),
        influence_sum=torch.cat(aux_s, dim=1) if aux_s else xf.new_zeros((lead, 0)),
        pre_clamp=torch.cat(aux_pre, dim=1) if aux_pre else xf.new_zeros((lead, 0, 3)),
        opacity=opacity,
        logdet=logdet,
        degenerate_precision=degen,
    )
    return out, aux


class GlutCarrier(nn.Module):
    """``nn.Module`` wrapper: zero parameters, one non-persistent constant.

    Exists so an arm can hold the clamp/residual configuration as module state
    (and get it into ``run_setup.json``) without any forward-time
    ``torch.tensor(...)``: the 3x3 identity lives in a buffer registered with
    ``persistent=False``, which is the pitfall-2 rule this batch runs under.
    """

    def __init__(self, *, clamp: ClampMode = "two", residual: bool = True,
                 eps: float = EPS, clamp_grad: ClampGrad = "hard") -> None:
        super().__init__()
        if clamp not in _CLAMP_MODES:
            raise ValueError(f"clamp must be one of {_CLAMP_MODES}, got {clamp!r}")
        if clamp_grad not in _CLAMP_GRADS:
            raise ValueError(f"clamp_grad must be one of {_CLAMP_GRADS}, got {clamp_grad!r}")
        self.clamp_mode = clamp
        self.clamp_grad = clamp_grad
        self.residual = bool(residual)
        self.eps = float(eps)
        self.register_buffer("eye3", torch.eye(3), persistent=False)

    def extra_repr(self) -> str:
        return (f"clamp={self.clamp_mode}, clamp_grad={self.clamp_grad}, "
                f"residual={self.residual}, eps={self.eps}")

    @property
    def config(self) -> dict[str, object]:
        """What ``run_setup.json`` records for the carrier."""
        return {"clamp": self.clamp_mode, "clamp_grad": self.clamp_grad,
                "residual": self.residual, "eps": self.eps}

    def forward(  # noqa: D102 -- see glut_forward
        self,
        x: Tensor,
        params: GlutParams,
        *,
        clamp: ClampMode | None = None,
        clamp_grad: ClampGrad | None = None,
        return_aux: bool = False,
        point_chunk: int | None = None,
        compute_dtype: torch.dtype | None = None,
    ) -> Tensor | tuple[Tensor, GlutAux]:
        return glut_forward(
            x,
            params,
            clamp=self.clamp_mode if clamp is None else clamp,
            clamp_grad=self.clamp_grad if clamp_grad is None else clamp_grad,
            residual=self.residual,
            eps=self.eps,
            compute_dtype=compute_dtype,
            point_chunk=point_chunk,
            return_aux=return_aux,
        )
