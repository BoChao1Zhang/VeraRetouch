"""JetLUT carrier: analytic atlas, exact partition-of-unity gate, local jets,
and a batched ADMM solver for the min-norm L1 projection (PROPOSAL §2-§4).

Layout
------
The carrier is

    F_theta(x) = x + Theta^T f(x),      Theta in R^{P' x 3},   P' = 4 + N K

with the per-point feature vector

    f(x) = [ psi_1(x) ;  { pi_i(x) * psi_p(delta_i(x)) }_{i=1..N} ]

    psi_1(x)     = [1, x_R, x_G, x_B]                       (global affine)
    delta_i(x)   = (x - mu_i) / sigma
    psi_1(delta) = [1, d1, d2, d3]                          K = 4   (p = 1)
    psi_2(delta) = psi_1 + [d1^2, d2^2, d3^2,
                            d1 d2, d1 d3, d2 d3]            K = 10  (p = 2)

The three output channels share ``f`` and get their own coefficient column, so
the dynamic parameter count is ``3 P' = 12 + 12 N`` (p=1) or ``12 + 30 N``
(p=2), exactly the ladder of PROPOSAL §2.2.  Because the L1 loss is
element-wise it is *separable across output channels*, so the whole library is
one batched right-hand side: ``R in R^{Q x 3L}``.

Only ``theta`` is per-LUT.  ``f`` depends on ``(m, c, p)`` alone, which is what
makes ``F^T F + (lam/rho) I`` a single Cholesky reused by every LUT and every
ADMM iteration (PROPOSAL §4.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "Atlas",
    "admm_lad",
    "design_matrix",
    "fill_distance",
    "jet_dim",
    "n_dynamic_params",
    "pou_weights",
]


def fill_distance(m: int) -> float:
    """Fill distance of the ``m^3`` cell-centre grid in ``[0,1]^3``.

    Worst point is a cube corner; nearest centre is ``(0.5/m,)*3``, hence
    ``h_m = sqrt(3) / (2m)`` (PROPOSAL §2, §3.5).
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    return math.sqrt(3.0) / (2.0 * m)


def jet_dim(p: int) -> int:
    """``K``: number of monomials of a 3-D jet of order ``p``."""
    if p == 1:
        return 4
    if p == 2:
        return 10
    raise ValueError(f"only p in {{1, 2}} is pre-registered, got {p}")


def n_dynamic_params(m: int, p: int) -> int:
    """LUT-specific coefficient count ``3 (4 + N K)``."""
    return 3 * (4 + m ** 3 * jet_dim(p))


@dataclass(frozen=True)
class Atlas:
    """The fully analytic atlas ``A_m(c)`` -- no training, two scalars.

    ``mu`` are the ``m^3`` cell centres ``(i+0.5)/m``; the bandwidth is
    ``sigma = c * h_m`` with ``h_m = sqrt(3)/(2m)``; gate log-amplitudes are
    identically zero and the shape matrix is isotropic ``B = I / sigma``
    (PROPOSAL §2).  ``gamma == 0`` is a design decision, not a default: a free
    ``gamma`` would let a primitive claim territory and turn the regular grid
    back into a learned non-uniform partition.
    """

    m: int
    c: float

    @property
    def n(self) -> int:
        return self.m ** 3

    @property
    def h(self) -> float:
        return fill_distance(self.m)

    @property
    def sigma(self) -> float:
        return self.c * self.h

    def centres(self, *, device: torch.device | str = "cpu",
                dtype: torch.dtype = torch.float64) -> Tensor:
        """``(N, 3)`` cell centres, R-major (matches ``grid_axis_sizes`` order)."""
        ax = (torch.arange(self.m, device=device, dtype=dtype) + 0.5) / self.m
        r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
        return torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)


def pou_weights(x: Tensor, mu: Tensor, sigma: float) -> Tensor:
    """``(Q, N)`` exact partition of unity: ``softmax_i(-|(x-mu_i)/sigma|^2 / 2)``.

    No ``eps`` in a denominator, no ``log|Sigma|`` term, no opacity -- the three
    couplings PROPOSAL §0.4 records in the GLUT gate.  Rows sum to 1 by
    construction, which ``test_core.py::test_pou_sums_to_one`` asserts.
    """
    d2 = ((x.unsqueeze(1) - mu.unsqueeze(0)) / sigma).pow(2).sum(-1)   # (Q, N)
    return torch.softmax(-0.5 * d2, dim=-1)


def _psi(delta: Tensor, p: int) -> Tensor:
    """``(..., K)`` jet monomials of ``delta`` (last axis is the 3 coordinates)."""
    d1, d2, d3 = delta.unbind(-1)
    one = torch.ones_like(d1)
    cols = [one, d1, d2, d3]
    if p == 2:
        cols += [d1 * d1, d2 * d2, d3 * d3, d1 * d2, d1 * d3, d2 * d3]
    return torch.stack(cols, dim=-1)


def design_matrix(x: Tensor, atlas: Atlas, p: int, *, chunk: int = 4096) -> Tensor:
    """``(Q, 4 + N K)`` feature matrix ``f(x)``.

    Chunked over ``x`` because the intermediate ``(Q, N, 3)`` local coordinates
    are the memory peak (N up to 512, Q up to 65^3).
    """
    mu = atlas.centres(device=x.device, dtype=x.dtype)
    sigma = atlas.sigma
    out = x.new_empty((x.shape[0], 4 + atlas.n * jet_dim(p)))
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]                                   # (q, 3)
        pi = pou_weights(xb, mu, sigma)                       # (q, N)
        delta = (xb.unsqueeze(1) - mu.unsqueeze(0)) / sigma    # (q, N, 3)
        local = pi.unsqueeze(-1) * _psi(delta, p)             # (q, N, K)
        glob = _psi(xb, 1)                                    # (q, 4) == [1, x]
        out[s:s + chunk] = torch.cat(
            (glob, local.reshape(xb.shape[0], -1)), dim=-1)
    return out


def _soft_threshold(v: Tensor, t: float) -> Tensor:
    return torch.sign(v) * torch.clamp(v.abs() - t, min=0.0)


@torch.no_grad()
def admm_lad(f: Tensor, r: Tensor, *, lam: float = 1e-8, rho: float = 100.0,
             max_iter: int = 2000, eps_abs: float = 1e-8, eps_rel: float = 1e-8,
             rho_update_every: int = 10, alpha: float = 1.7,
             verbose: bool = False) -> dict[str, Tensor | float | int]:
    """Batched minimum-norm LAD:  ``min_T ||f T - r||_1 + (lam/2) ||T||_F^2``.

    ``f`` is ``(Q, P)`` and shared; ``r`` is ``(Q, B)`` with one column per
    (LUT, channel) pair.  ADMM splits ``z = f T - r`` so the T-update is

        (f^T f + (lam/rho) I) T = f^T (r + z - u)

    whose factor does **not** depend on ``r`` or on the iteration -- one
    Cholesky for the whole library (PROPOSAL §4.1).

    Returns the solution plus a *feasible dual lower bound*: project
    ``clip(u, -1, 1)`` onto ``null(f^T)`` and rescale to the unit inf-ball, so
    ``r^T y`` is a valid lower bound on the LAD optimum and the reported gap is
    a certificate rather than a residual (PROPOSAL §4.1's convexity gate).
    """
    q, pdim = f.shape
    gram0 = f.T @ f
    # numerical floor: the POU features are strongly collinear at large N, so a
    # pure lam=0 Gram is not reliably positive definite even in fp64.
    ridge_floor = 1e-12 * float(gram0.diagonal().mean())
    ft = f.T.contiguous()

    def _factor(rho_: float) -> Tensor:
        g = gram0.clone()
        g.diagonal().add_(max(lam / rho_, ridge_floor))
        return torch.linalg.cholesky(g)

    chol = _factor(rho)
    theta = torch.zeros((pdim, r.shape[1]), dtype=f.dtype, device=f.device)
    z = torch.zeros_like(r)
    u = torch.zeros_like(r)
    n_iter = 0
    r_pri = r_dual = float("inf")

    for n_iter in range(1, max_iter + 1):
        theta = torch.cholesky_solve(ft @ (r + z - u), chol)
        ax = f @ theta
        z_old = z
        # over-relaxation (Boyd §3.4.3): alpha in [1.5, 1.8] is the standard
        # 2-3x iteration saving on LAD-shaped problems.
        ax_hat = alpha * ax + (1.0 - alpha) * (r + z_old)
        z = _soft_threshold(ax_hat - r + u, 1.0 / rho)
        u = u + (ax_hat - r - z)

        r_pri = float((ax - r - z).norm())
        r_dual = float((rho * (ft @ (z_old - z))).norm())
        tol_pri = math.sqrt(q * r.shape[1]) * eps_abs + eps_rel * max(
            float(ax.norm()), float(r.norm()), float(z.norm()))
        tol_dual = math.sqrt(pdim * r.shape[1]) * eps_abs + eps_rel * float(
            (rho * (ft @ u)).norm())
        if r_pri <= tol_pri and r_dual <= tol_dual:
            break
        if rho_update_every and n_iter % rho_update_every == 0:
            if r_pri > 10.0 * r_dual:
                rho, u, chol = rho * 2.0, u / 2.0, _factor(rho * 2.0)
            elif r_dual > 10.0 * r_pri:
                rho, u, chol = rho / 2.0, u * 2.0, _factor(rho / 2.0)
        if verbose and n_iter % 100 == 0:
            print(f"  admm {n_iter:5d}  r_pri {r_pri:.3e}  r_dual {r_dual:.3e}"
                  f"  rho {rho:.3g}")

    resid = f @ theta - r
    primal = resid.abs().sum(0)                                  # (B,)

    # Dual of  min_T ||f T - r||_1  is  max_y  -r^T y  s.t.  f^T y = 0,
    # ||y||_inf <= 1.  Both +-y are feasible, so |r^T y| is a valid lower bound.
    # Build one: clip the scaled dual variable, project onto null(f^T) (twice --
    # the factor carries a ridge, so one pass leaves a small residual), rescale.
    # Alternating projection between the box and null(f^T), started from the
    # scaled dual y = rho * u.  A single project-then-rescale would divide the
    # whole column by its worst entry, which throws most of the bound away.
    y = rho * u
    for _ in range(30):
        y = torch.clamp(y, -1.0, 1.0)
        y = y - f @ torch.cholesky_solve(ft @ y, chol)
    scale = torch.clamp(y.abs().amax(dim=0), min=1.0)   # final feasibility
    y = y / scale
    dual = (r * y).sum(0).abs()                                  # (B,)
    gap = (primal - dual) / torch.clamp(primal.abs(), min=1.0)
    dual_feas = float((ft @ y).abs().max())     # how close to f^T y = 0

    return {"theta": theta, "primal": primal, "dual": dual, "gap": gap,
            "dual_feas": dual_feas, "n_iter": n_iter, "rho": rho,
            "r_pri": r_pri, "r_dual": r_dual}
