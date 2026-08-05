"""Protocol 7.6 -- the 48-Gaussian LUT function and its constrained parameters.

    "Gaussian centres are confined to the RGB cube by a sigmoid; the covariance
     is built from a Cholesky factor with a softplus diagonal, guaranteeing SPD;
     opacity/existence use a sigmoid.  Global and local affines both use an
     identity-centred residual parameterisation.  With ``q_i(x)`` the normalised
     weights,

        T_pred(x) = clamp((I + dG) x + b_g + sum_i q_i(x) (M_i x + b_i), 0, 1)."

The one deliberate departure from that formula is documented at
``q3vl.what.config.GLOBAL_AFFINE_MODE``: taking it literally, with ``M_i`` also
identity-centred and ``sum_i q_i = 1``, gives ``T(x) = 2x`` at zero
initialisation -- the exact failure the campaign red line names.  The default
mode keeps the local affines identity-centred and makes the global affine a pure
residual initialised at 0, which is both what the red line demands and, term for
term, ``model/glut_repro/model_rdg.py::render`` (CI-pinned against
``BatchedGLUT`` to 1e-5, GLUT Eq.1-3).  ``identity_centered`` reproduces the
literal reading so the 2x claim can be *shown*.

Numerical form of the mixture, taken from the same reference implementation:

* the Mahalanobis distance is computed by forward substitution through the
  lower-triangular Cholesky factor rather than by inverting anything;
* the log-density max is factored out of the normaliser, so a collapsing sigma
  drives ``w -> 0`` instead of producing NaN;
* ``eps = 1e-6`` sits in the denominator exactly where GLUT puts it.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ANCHOR_GRID, BAKE_CHUNK_POINTS, BAKE_SIZE, LutConfig, N_SLOTS
from .lut import lattice_points

__all__ = [
    "anchor_points", "inv_sigmoid", "softplus_inv", "decode_primitives",
    "decode_global", "gaussian_log_density", "mixture_weights", "render",
    "bake", "GeometryBank", "PRIM_LAYOUT_FG", "PRIM_LAYOUT_SB", "GEOM_LAYOUT",
    "identity_raw", "parameter_report",
]

LOG_2PI = math.log(2.0 * math.pi)

#: slice layout of the 23 per-primitive outputs (protocol 7.4's table order)
PRIM_LAYOUT_FG = {
    "mu": (0, 3), "chol_diag": (3, 6), "chol_off": (6, 9),
    "opacity": (9, 10), "existence": (10, 11), "M": (11, 20), "b": (20, 23),
}
#: SB48 generates only the payload (protocol 7.5): 14 per primitive
PRIM_LAYOUT_SB = {
    "opacity": (0, 1), "existence": (1, 2), "M": (2, 11), "b": (11, 14),
}
#: the 9 geometry outputs, shared (SB48) or provisional-then-refined (FG48)
GEOM_LAYOUT = {"mu": (0, 3), "chol_diag": (3, 6), "chol_off": (6, 9)}


# --- anchors and inverse maps ----------------------------------------------

def anchor_points(grid: tuple[int, int, int] = ANCHOR_GRID,
                  device=None, dtype=torch.float32) -> torch.Tensor:
    """The fixed ``4 x 4 x 3`` RGB anchors of protocol 7.5, as ``(48, 3)``.

    Cell *centres*, not cell corners: ``mu`` is a sigmoid, so an anchor at 0.0 or
    1.0 has no finite pre-image and could not be reproduced by the
    parameterisation the protocol mandates.
    """
    axes = [(torch.arange(n, dtype=dtype, device=device) + 0.5) / n for n in grid]
    r, g, b = torch.meshgrid(*axes, indexing="ij")
    pts = torch.stack([r, g, b], dim=-1).reshape(-1, 3)
    if pts.shape[0] != N_SLOTS:
        raise AssertionError(f"anchor grid {grid} gives {pts.shape[0]} points, need {N_SLOTS}")
    return pts


def inv_sigmoid(y: torch.Tensor | float, eps: float = 1e-6) -> torch.Tensor:
    t = torch.as_tensor(y, dtype=torch.float32).clamp(eps, 1.0 - eps)
    return torch.log(t / (1.0 - t))


def softplus_inv(y: float) -> float:
    """``x`` with ``softplus(x) == y``; the large-``y`` branch avoids expm1(-inf)."""
    return float(y + math.log(-math.expm1(-y))) if y < 20.0 else float(y)


def _sigma_diag(raw: torch.Tensor, cfg: LutConfig) -> torch.Tensor:
    """Strictly positive Cholesky diagonal -> SPD covariance.

    ``softplus_floor`` (protocol 7.6, the default): ``lo + softplus(raw + c)``
    with ``c`` chosen so ``raw = 0`` gives ``sigma_init``.  ``bounded_sigmoid``
    is the campaign red line's alternative (RD-G's ``0.02 + 0.48 sigmoid``).
    Neither is a bare ``exp``.
    """
    if cfg.sigma_param == "softplus_floor":
        shift = softplus_inv(max(cfg.sigma_init - cfg.sigma_lo, 1e-6))
        return cfg.sigma_lo + F.softplus(raw + shift)
    if cfg.sigma_param == "bounded_sigmoid":
        z0 = math.log((cfg.sigma_init - cfg.sigma_lo)
                      / max(cfg.sigma_span - (cfg.sigma_init - cfg.sigma_lo), 1e-6))
        return cfg.sigma_lo + cfg.sigma_span * torch.sigmoid(raw + z0)
    raise ValueError(f"unknown sigma parameterisation {cfg.sigma_param!r}")


# --- decoding ---------------------------------------------------------------

def _slice(z: torch.Tensor, layout: dict[str, tuple[int, int]], key: str) -> torch.Tensor:
    a, b = layout[key]
    return z[..., a:b]


def decode_geometry(z_geom: torch.Tensor, anchors: torch.Tensor,
                    cfg: LutConfig) -> dict[str, torch.Tensor]:
    """``(..., N, 9) -> mu (…,N,3), sigma (…,N,3), off (…,N,3)``.

    ``mu = sigmoid(z + logit(anchor))`` -- a sigmoid, so the centre can never
    leave the RGB cube (protocol 7.6), and equal to the anchor at ``z = 0`` so a
    zero-initialised head starts from the ``4 x 4 x 3`` grid rather than from 48
    coincident centres at 0.5.
    """
    mu_raw = _slice(z_geom, GEOM_LAYOUT, "mu")
    bias = inv_sigmoid(anchors).to(mu_raw.device, mu_raw.dtype)
    return {
        "mu": torch.sigmoid(mu_raw + bias),
        "sigma": _sigma_diag(_slice(z_geom, GEOM_LAYOUT, "chol_diag"), cfg),
        "off": cfg.chol_off_scale * _slice(z_geom, GEOM_LAYOUT, "chol_off"),
    }


def decode_primitives(z_prim: torch.Tensor, cfg: LutConfig, layout: dict,
                      anchors: torch.Tensor | None = None,
                      geometry: dict[str, torch.Tensor] | None = None
                      ) -> dict[str, torch.Tensor]:
    """Raw head outputs -> constrained renderer parameters.

    ``layout`` is :data:`PRIM_LAYOUT_FG` (geometry is generated per sample) or
    :data:`PRIM_LAYOUT_SB` (geometry comes from ``geometry``, shared across
    samples).  Every constraint of protocol 7.6 is applied here and nowhere else.
    """
    B, N = z_prim.shape[0], z_prim.shape[1]
    eye = torch.eye(3, device=z_prim.device, dtype=z_prim.dtype)
    if "mu" in layout:
        if anchors is None:
            raise ValueError("FG48 decoding needs the anchor grid")
        geom = decode_geometry(
            torch.cat([_slice(z_prim, layout, "mu"),
                       _slice(z_prim, layout, "chol_diag"),
                       _slice(z_prim, layout, "chol_off")], dim=-1),
            anchors, cfg)
    else:
        if geometry is None:
            raise ValueError("SB48 decoding needs the shared geometry")
        geom = {k: (v if v.dim() == 3 else v.unsqueeze(0).expand(B, -1, -1))
                for k, v in geometry.items()}
    out = {
        "mu": geom["mu"], "sigma": geom["sigma"], "off": geom["off"],
        "opacity": torch.sigmoid(_slice(z_prim, layout, "opacity").squeeze(-1)
                                 + cfg.opacity_bias),
        "existence": torch.sigmoid(_slice(z_prim, layout, "existence").squeeze(-1)
                                   + cfg.existence_bias),
        "M": eye + cfg.affine_scale * _slice(z_prim, layout, "M").reshape(B, N, 3, 3),
        "b": cfg.affine_scale * _slice(z_prim, layout, "b"),
    }
    return out


def decode_global(z_glob: torch.Tensor, cfg: LutConfig) -> dict[str, torch.Tensor]:
    """``(B, 12) -> G (B,3,3), b_g (B,3)``.

    ``residual_zero`` (default): ``G = 0.1 z``, i.e. **G is 0 at init, not I**.
    ``identity_centered``: ``G = I + 0.1 z``, the protocol's literal formula,
    which together with the identity-centred ``M_i`` gives ``T(x) = 2x``.
    """
    G = cfg.global_scale * z_glob[..., :9].reshape(-1, 3, 3)
    if cfg.global_affine_mode == "identity_centered":
        G = G + torch.eye(3, device=z_glob.device, dtype=z_glob.dtype)
    elif cfg.global_affine_mode != "residual_zero":
        raise ValueError(f"unknown global affine mode {cfg.global_affine_mode!r}")
    return {"G": G, "b_g": cfg.global_scale * z_glob[..., 9:12]}


def identity_raw(n_slots: int = N_SLOTS, layout: dict | None = None,
                 batch: int = 1, device=None, dtype=torch.float32
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """The raw outputs a zero-initialised head produces: all zeros.

    Exists so ``tests/test_gaussians.py`` can state the identity assertion as
    "the head's own initial output", not as a hand-written zero tensor.
    """
    n = (max(b for _, b in (layout or PRIM_LAYOUT_FG).values()))
    return (torch.zeros(batch, n_slots, n, device=device, dtype=dtype),
            torch.zeros(batch, 12, device=device, dtype=dtype))


# --- the mixture ------------------------------------------------------------

def gaussian_log_density(x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                         off: torch.Tensor) -> torch.Tensor:
    """``log N(x; mu, L L^T)`` for ``x (B,P,3)`` and ``(B,N,3)`` parameters.

    ``L = [[s0,0,0],[o0,s1,0],[o1,o2,s2]]``; solved by forward substitution, so
    nothing is ever inverted and SPD holds for any ``s_i > 0``.
    Returns ``(B, N, P)``.  Also used by :mod:`q3vl.what.pooling` -- the aligned
    pooling of protocol 7.2 must use *the same* density as the renderer, or the
    "each Gaussian's real colour distribution" claim is false.
    """
    diff = x.unsqueeze(1) - mu.unsqueeze(2)                    # (B,N,P,3)
    d0, d1, d2 = diff.unbind(-1)
    z0 = d0 / sigma[..., 0:1]
    z1 = (d1 - off[..., 0:1] * z0) / sigma[..., 1:2]
    z2 = (d2 - off[..., 1:2] * z0 - off[..., 2:3] * z1) / sigma[..., 2:3]
    maha = z0 * z0 + z1 * z1 + z2 * z2
    log_det = 2.0 * torch.log(sigma).sum(-1)                   # (B,N)
    return -1.5 * LOG_2PI - 0.5 * log_det.unsqueeze(-1) - 0.5 * maha


def mixture_weights(p: dict[str, torch.Tensor], x: torch.Tensor,
                    eps: float) -> torch.Tensor:
    """``q_i(x) = o_i g_i N_i(x) / (sum_j o_j g_j N_j(x) + eps)`` -> ``(B,N,P)``."""
    ld = gaussian_log_density(x, p["mu"], p["sigma"], p["off"])
    og = (p["opacity"] * p["existence"]).unsqueeze(-1)
    m = ld.max(dim=1, keepdim=True).values
    num = torch.exp(ld - m) * og
    eps_term = torch.exp(torch.clamp(-m, max=80.0)) * eps
    return num / (num.sum(dim=1, keepdim=True) + eps_term)


def render(p: dict[str, torch.Tensor], x: torch.Tensor, cfg: LutConfig,
           return_weights: bool = False):
    """``T_pred(x)`` for ``x (B,P,3)`` -> ``(B,P,3)`` (clamped to the cube)."""
    x32 = x.float() if x.dtype not in (torch.float32, torch.float64) else x
    q = mixture_weights(p, x32, cfg.mixture_eps)
    A = torch.einsum("bnp,bnij->bpij", q, p["M"])
    mix = (torch.einsum("bpij,bpj->bpi", A, x32)
           + torch.einsum("bnp,bnj->bpj", q, p["b"]))
    out = mix + torch.einsum("bij,bpj->bpi", p["G"], x32) + p["b_g"].unsqueeze(1)
    if cfg.clamp_output:
        out = out.clamp(0.0, 1.0)
    return (out, q) if return_weights else out


def bake(p: dict[str, torch.Tensor], cfg: LutConfig, size: int = BAKE_SIZE,
         chunk: int = BAKE_CHUNK_POINTS) -> torch.Tensor:
    """Evaluate the analytic renderer on the uniform ``size^3`` lattice.

    ``(B, size, size, size, 3)``, indexed ``[r, g, b]`` -- the canonical layout of
    :mod:`q3vl.what.lut`, so :func:`q3vl.what.lut.tetra_lookup` reads it back
    without any transpose.  Chunked over lattice points because 33^3 = 35937
    points times 48 Gaussians is a ``(B, 48, 35937)`` intermediate.
    """
    B = p["mu"].shape[0]
    pts = lattice_points(size, p["mu"].device, p["mu"].dtype)
    outs = []
    for i in range(0, pts.shape[0], chunk):
        sub = pts[i:i + chunk].unsqueeze(0).expand(B, -1, -1)
        outs.append(render(p, sub, cfg))
    return torch.cat(outs, dim=1).reshape(B, size, size, size, 3)


# --- SB48's shared geometry -------------------------------------------------

class GeometryBank(nn.Module):
    """Protocol 7.5 -- the 48 ``mu_i, Sigma_i`` shared across samples.

    "initialised from fixed 4x4x3 RGB anchors, then jointly optimised over all
     training LUTs".  A ``nn.Parameter`` rather than a buffer: it *is* trained,
     just at its own learning rate (``shared_geometry_lr = 5e-5``), and
     :func:`q3vl.what.trainer.build_optimizer` puts it in its own group by
     matching this module's parameter names.
    """

    def __init__(self, cfg: LutConfig):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("anchors", anchor_points(cfg.anchor_grid), persistent=True)
        self.raw = nn.Parameter(torch.zeros(cfg.n_slots, 9))

    def forward(self, batch: int = 1) -> dict[str, torch.Tensor]:
        g = decode_geometry(self.raw.unsqueeze(0), self.anchors, self.cfg)
        return {k: v.expand(batch, -1, -1) for k, v in g.items()}

    def facts(self) -> dict[str, Any]:
        with torch.no_grad():
            g = self.forward(1)
        return {
            "n_slots": int(self.cfg.n_slots),
            "anchor_grid": list(self.cfg.anchor_grid),
            "mu_range": [float(g["mu"].min()), float(g["mu"].max())],
            "sigma_range": [float(g["sigma"].min()), float(g["sigma"].max())],
            "n_params": int(self.raw.numel()),
        }


# --- reporting --------------------------------------------------------------

def parameter_report(p: dict[str, torch.Tensor], tol: float = 1e-6) -> dict[str, Any]:
    """Protocol 14.12 evidence: SPD, weight normalisation, bounded parameters.

    ``mu`` strictly inside the cube, ``sigma`` strictly positive (hence SPD),
    ``opacity``/``existence`` strictly in (0,1), everything finite.
    """
    with torch.no_grad():
        mu, sig = p["mu"], p["sigma"]
        finite = all(bool(torch.isfinite(v).all()) for v in p.values()
                     if torch.is_tensor(v))
        return {
            "mu_in_cube": bool(((mu > 0.0) & (mu < 1.0)).all()),
            "mu_range": [float(mu.min()), float(mu.max())],
            "sigma_positive": bool((sig > tol).all()),
            "sigma_range": [float(sig.min()), float(sig.max())],
            "spd": bool((sig > tol).all()),          # L is lower-triangular with sig>0
            "opacity_range": [float(p["opacity"].min()), float(p["opacity"].max())],
            "existence_range": [float(p["existence"].min()), float(p["existence"].max())],
            "n_existence_gt_half": int((p["existence"] > 0.5).sum()),
            "n_opacity_gt_half": int((p["opacity"] > 0.5).sum()),
            "all_finite": finite,
        }
