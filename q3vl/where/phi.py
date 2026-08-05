"""Protocol 4.2 -- the 71-dimensional direction feature ``phi_dir``.

    geo5(p)          = [x, y, P2(x), P2(y), x*y]
    range(p)         = [L, S]                     (per-image standardised)
    semantic_low(p)  = B(F_pre(p)),  B: 1024 -> 64
    phi_dir(p)       = [geo5, L, S, semantic_1..64]  in R^71

The 64 semantic channels are least-squares residualised against
``[1, geo5, L, S]`` and then zero-mean / unit-variance standardised, so the
projector cannot smuggle coordinates, lightness or saturation back in.

Every function here works on one image: a flattened ``(P, D)`` layout with
``P = grid_h * grid_w`` in row-major order.  Images in a batch have different
grids, so batching is done by the caller looping over samples; the whole module
is differentiable with respect to the semantic block (that is the only path by
which the shared projector ``B`` receives gradient).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import (
    GEO_NAMES,
    PHI_DIR_DIM,
    PHI_DIR_NAMES,
    RANGE_NAMES,
    RESID_BLOCK_DIM,
    SEM_DIM,
    PhiConfig,
)

__all__ = [
    "PhiParts",
    "norm_coords",
    "legendre_p2",
    "geo5_grid",
    "range_channels",
    "standardize",
    "residualize",
    "build_phi_dir",
    "design_block",
]


def legendre_p2(t: torch.Tensor) -> torch.Tensor:
    """Second Legendre polynomial P2(t) = (3 t^2 - 1) / 2."""
    return 0.5 * (3.0 * t * t - 1.0)


def norm_coords(
    grid_h: int,
    grid_w: int,
    *,
    device=None,
    dtype=torch.float64,
    coord_mode: str = "short_side_unit",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates on the low-res grid.

    ``short_side_unit`` (protocol 4.1 "true aspect ratio, never squashed to
    512x512"): both axes are divided by *half the short side*, so the short
    side spans [-1, 1], the long side spans [-AR, +AR] and a grid cell is
    square in feature space.
    """
    if coord_mode != "short_side_unit":
        raise ValueError(f"unknown coord_mode {coord_mode!r}")
    half = min(grid_h, grid_w) / 2.0
    ys = (torch.arange(grid_h, device=device, dtype=dtype) + 0.5 - grid_h / 2.0) / half
    xs = (torch.arange(grid_w, device=device, dtype=dtype) + 0.5 - grid_w / 2.0) / half
    Y, X = torch.meshgrid(ys, xs, indexing="ij")
    return X, Y


def geo5_grid(
    grid_h: int,
    grid_w: int,
    *,
    device=None,
    dtype=torch.float64,
    coord_mode: str = "short_side_unit",
) -> torch.Tensor:
    """``(P, 5)`` = [x, y, P2(x), P2(y), x*y], row-major over the grid."""
    X, Y = norm_coords(grid_h, grid_w, device=device, dtype=dtype, coord_mode=coord_mode)
    cols = [X, Y, legendre_p2(X), legendre_p2(Y), X * Y]
    return torch.stack([c.reshape(-1) for c in cols], dim=1)


def range_channels(
    img: torch.Tensor, *, luma: str = "rec709", saturation: str = "hsv"
) -> tuple[torch.Tensor, torch.Tensor]:
    """``img`` is ``(3, h, w)`` sRGB in [0, 1] -> ``(L, S)`` each ``(h*w,)``."""
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(f"expected (3,h,w) sRGB image, got {tuple(img.shape)}")
    if luma != "rec709":
        raise ValueError(f"unknown luma {luma!r}")
    if saturation != "hsv":
        raise ValueError(f"unknown saturation {saturation!r}")
    r, g, b = img[0], img[1], img[2]
    L = 0.2126 * r + 0.7152 * g + 0.0722 * b
    mx = img.max(dim=0).values
    mn = img.min(dim=0).values
    S = torch.where(mx > 1e-6, (mx - mn) / mx.clamp_min(1e-6), torch.zeros_like(mx))
    return L.reshape(-1), S.reshape(-1)


def standardize(v: torch.Tensor, eps: float, dim: int = 0) -> torch.Tensor:
    """Zero-mean / unit-variance along ``dim`` (per-image, fixed protocol)."""
    mean = v.mean(dim=dim, keepdim=True)
    std = v.std(dim=dim, unbiased=False, keepdim=True)
    return (v - mean) / (std + eps)


DEAD_CHANNEL_REL = 1e-8


def standardize_live(
    v: torch.Tensor, std_before: torch.Tensor, eps: float, rel: float = DEAD_CHANNEL_REL
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standardise only the channels that survived residualisation.

    A projector row that learns a pure copy of ``x`` (or of ``L``) is fully
    annihilated by the residualisation; its residual is numerical noise of order
    1e-15.  Dividing that by ``std + 1e-6`` would amplify pure round-off into a
    unit-variance "feature".  Such channels are set to exactly zero instead and
    counted, which is both honest and what the residualisation is *for*.
    """
    mean = v.mean(dim=0, keepdim=True)
    std = v.std(dim=0, unbiased=False, keepdim=True)
    live = (std > rel * (std_before.reshape(1, -1) + eps))
    z = (v - mean) / (std + eps)
    return torch.where(live, z, torch.zeros_like(z)), live.reshape(-1)


def design_block(geo5: torch.Tensor, L: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """``[1, geo5, L, S]`` -- the ``(P, 8)`` block the semantics are made
    orthogonal to (protocol 4.2)."""
    ones = torch.ones_like(L).unsqueeze(1)
    return torch.cat([ones, geo5, L.unsqueeze(1), S.unsqueeze(1)], dim=1)


def effective_gram_cond(M: torch.Tensor, tol: float = 1e-12) -> dict[str, Any]:
    """Condition number of ``M^T M`` over the columns that actually carry signal.

    A perfectly flat channel -- a **greyscale photograph** makes HSV saturation
    identically 0, and 3.3-3.8% of the local pool is greyscale -- becomes an
    all-zero column after per-image standardisation, so the Gram is exactly
    singular and ``cond`` is ``inf``.  That is a true statement about a
    structurally absent dimension, not a defect in the sample: the oracle fit is
    unaffected (the ridge solve ignores the column) and the image is a perfectly
    ordinary black-and-white photo that Where has to handle anyway.

    So the reported number is the conditioning of the *effective* subspace, with
    the dropped dimensions counted next to it and the full (possibly infinite)
    value kept for transparency.
    """
    with torch.no_grad():
        Md = M.double()
        norms = Md.norm(dim=0)
        keep = norms > tol * float(norms.max().clamp_min(1e-30))
        n_dropped = int((~keep).sum())
        full = float(torch.linalg.cond(Md.transpose(0, 1) @ Md))
        sub = Md[:, keep]
        eff = (float(torch.linalg.cond(sub.transpose(0, 1) @ sub))
               if sub.shape[1] else float("inf"))
        return {
            "cond": eff,
            "cond_full": full,
            "n_dropped_columns": n_dropped,
            "n_columns": int(M.shape[1]),
            "rank": int(torch.linalg.matrix_rank(sub)) if sub.shape[1] else 0,
        }


def _ridge_lstsq(A: torch.Tensor, E: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Least squares ``argmin ||A c - E||`` via the normal equations.

    ``torch.linalg.lstsq`` has no autograd support for the rank-deficient
    drivers, and the semantic block *must* stay differentiable (that is the only
    path to ``B``).  A trace-relative ridge of 1e-10 keeps the solve stable for
    a degenerate image (constant L or S) without measurably biasing a
    well-conditioned one.  The Gram condition number is returned so preflight
    item 5 can report it.
    """
    G = A.transpose(-2, -1) @ A
    j = G.shape[-1]
    lam = 1e-10 * (torch.diagonal(G, dim1=-2, dim2=-1).sum() / j).clamp_min(1e-30)
    Gr = G + lam * torch.eye(j, device=G.device, dtype=G.dtype)
    coef = torch.linalg.solve(Gr, A.transpose(-2, -1) @ E)
    with torch.no_grad():
        cond = float(torch.linalg.cond(G.double()))
    return coef, cond


def _max_abs_corr(A: torch.Tensor, E: torch.Tensor, eps: float) -> float:
    """max |normalised cross-correlation| between the columns of A and of E."""
    with torch.no_grad():
        a = A.double()
        e = E.double()
        a = a - a.mean(0, keepdim=True)
        e = e - e.mean(0, keepdim=True)
        an = a / (a.norm(dim=0, keepdim=True) + eps)
        en = e / (e.norm(dim=0, keepdim=True) + eps)
        c = an.transpose(0, 1) @ en
        return float(c.abs().max()) if c.numel() else 0.0


def residualize(
    E: torch.Tensor, A: torch.Tensor, eps: float
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Project ``E`` (P,K) onto the orthogonal complement of ``A`` (P,J)."""
    coef, _ = _ridge_lstsq(A, E)
    before = _max_abs_corr(A, E, eps)
    Er = E - A @ coef
    after = _max_abs_corr(A, Er, eps)
    g = effective_gram_cond(A)
    return Er, {
        "resid_corr_before": before,
        "resid_corr_after": after,
        "design_gram_cond": g["cond"],
        "design_gram_cond_full": g["cond_full"],
        "design_dropped_columns": g["n_dropped_columns"],
    }


@dataclass
class PhiParts:
    phi_dir: torch.Tensor            # (P, 71)
    geo5: torch.Tensor               # (P, 5)
    L: torch.Tensor                  # (P,)
    S: torch.Tensor                  # (P,)
    semantic: torch.Tensor           # (P, 64) residualised + standardised
    grid_h: int
    grid_w: int
    diag: dict[str, Any]

    @property
    def names(self) -> tuple[str, ...]:
        return PHI_DIR_NAMES


def build_phi_dir(
    semantic_low: torch.Tensor,
    img_low: torch.Tensor,
    grid_h: int,
    grid_w: int,
    cfg: PhiConfig | None = None,
) -> PhiParts:
    """Assemble ``phi_dir`` for one image.

    ``semantic_low``: ``(P, 64)`` = ``B(F_pre)`` on the H/16 x W/16 grid,
    row-major.  ``img_low``: ``(3, grid_h, grid_w)`` sRGB in [0,1], the spec-5
    image area-averaged onto the same grid.
    """
    cfg = cfg or PhiConfig()
    if semantic_low.dim() != 2 or semantic_low.shape[1] != cfg.sem_dim:
        raise ValueError(
            f"semantic_low must be (P,{cfg.sem_dim}), got {tuple(semantic_low.shape)}"
        )
    P = grid_h * grid_w
    if semantic_low.shape[0] != P:
        raise ValueError(f"semantic_low has {semantic_low.shape[0]} rows, grid says {P}")
    if tuple(img_low.shape) != (3, grid_h, grid_w):
        raise ValueError(
            f"img_low must be (3,{grid_h},{grid_w}), got {tuple(img_low.shape)}"
        )

    dtype = semantic_low.dtype
    device = semantic_low.device
    geo5 = geo5_grid(
        grid_h, grid_w, device=device, dtype=dtype, coord_mode=cfg.coord_mode
    )
    L, S = range_channels(
        img_low.to(dtype), luma=cfg.luma, saturation=cfg.saturation
    )
    dead_range: list[str] = []
    if cfg.standardize_range:
        # Same rule as the semantic block: a channel that carries no variation
        # (greyscale photo -> S is identically 0) is set to exactly zero and
        # counted, rather than having its round-off amplified by 1/(0 + eps)
        # into a unit-variance "feature".
        rng = torch.stack([L, S], dim=1)
        rng_z, live = standardize_live(rng, rng.detach().std(dim=0, unbiased=False),
                                       cfg.std_eps)
        L, S = rng_z[:, 0], rng_z[:, 1]
        dead_range = [n for n, ok in zip(RANGE_NAMES, live.tolist()) if not ok]

    A = design_block(geo5, L, S)
    if A.shape[1] != RESID_BLOCK_DIM:
        raise AssertionError(f"design block is {A.shape[1]} wide, expected {RESID_BLOCK_DIM}")

    std_before = semantic_low.detach().std(dim=0, unbiased=False)
    if cfg.residualize:
        sem, diag = residualize(semantic_low, A, cfg.std_eps)
    else:
        sem, diag = semantic_low, {"resid_corr_before": None, "resid_corr_after": None,
                                   "design_gram_cond": None}
    sem, live = standardize_live(sem, std_before, cfg.std_eps)
    diag = dict(diag)
    diag["n_dead_semantic"] = int((~live).sum())
    diag["n_dead_range"] = len(dead_range)
    diag["dead_range_names"] = dead_range

    phi = torch.cat([geo5, L.unsqueeze(1), S.unsqueeze(1), sem], dim=1)
    if phi.shape[1] != PHI_DIR_DIM:
        raise AssertionError(f"phi_dir is {phi.shape[1]}-dim, protocol 4.2 says {PHI_DIR_DIM}")

    with torch.no_grad():
        g = effective_gram_cond(phi)
        diag = dict(diag)
        diag["phi_gram_cond"] = g["cond"]
        diag["phi_gram_cond_full"] = g["cond_full"]
        diag["phi_dropped_columns"] = g["n_dropped_columns"]
        diag["phi_rank"] = g["rank"]
        diag["n_points"] = int(phi.shape[0])
    return PhiParts(
        phi_dir=phi, geo5=geo5, L=L, S=S, semantic=sem,
        grid_h=grid_h, grid_w=grid_w, diag=diag,
    )


def geo_names() -> tuple[str, ...]:
    return GEO_NAMES


def semantic_slice() -> slice:
    """Columns of ``phi_dir`` occupied by the semantic block."""
    return slice(PHI_DIR_DIM - SEM_DIM, PHI_DIR_DIM)
