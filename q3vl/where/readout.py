"""Protocol 4.3 -- the two readouts ``R-Band`` and ``R-CBand12``.

``R-Band`` (learnable flat-top band-pass with polarity)::

    b(z) = sigmoid(k * (z - mu + h)) - sigmoid(k * (z - mu - h))
    m(z) = pi * b(z) + (1 - pi) * (1 - b(z))
    h > 0,  k in [1, 40],  pi = sigmoid(pi_raw)

``R-CBand12`` (E2's verified fixed-centre normalised Gaussian competition)::

    mu_i = linspace(-3, 3, 12)                  # fixed, a buffer not a parameter
    sigma_i in [0.025, 0.30]
    g_i(z) = o_i * exp(-0.5 * ((z - mu_i) / sigma_i)^2)
    m(z)   = sum_i c_i g_i(z) / (sum_i g_i(z) + eps)
    o_i, c_i in (0, 1)

Red line: every bounded parameter (sigma above all) uses a bounded sigmoid, never
a bare ``exp``.  ``mu_i`` is a fixed buffer, never optimised.

Both readouts also expose ``mirror``: the parameter map that leaves ``m(s(p))``
numerically unchanged when ``s -> -s``.  It is what makes the ``w -> -w`` sign
canonicalisation in :mod:`q3vl.where.basis` exact rather than approximate.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .config import (
    BAND_H_HI,
    BAND_H_LO,
    BAND_K_HI,
    BAND_K_LO,
    CBAND_EPS,
    CBAND_M,
    CBAND_MU_HI,
    CBAND_MU_LO,
    CBAND_NORMALIZATION,
    CBAND_SIG_HI,
    CBAND_SIG_LO,
)

__all__ = [
    "READOUTS",
    "bounded_sigmoid",
    "inv_bounded_sigmoid",
    "cband_centres",
    "band_params",
    "cband_params",
    "apply_readout",
    "mirror_params",
    "describe_params",
    "param_shapes",
]

READOUTS = ("band", "cband12")


def bounded_sigmoid(raw: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """``lo + (hi - lo) * sigmoid(raw)`` -- the only sanctioned bounded map."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def inv_bounded_sigmoid(value, lo: float, hi: float) -> float:
    """Inverse of :func:`bounded_sigmoid`, clipped away from the asymptotes."""
    z = (float(value) - lo) / (hi - lo)
    z = min(max(z, 1e-6), 1 - 1e-6)
    return math.log(z / (1 - z))


def cband_centres(device=None, dtype=torch.float64) -> torch.Tensor:
    """The 12 fixed centres.  ``linspace(-3, 3, 12)`` is symmetric, which is why
    ``mirror`` for CBand12 is exactly an index reversal."""
    return torch.linspace(CBAND_MU_LO, CBAND_MU_HI, CBAND_M, device=device, dtype=dtype)


def param_shapes(readout: str) -> dict[str, tuple[int, ...]]:
    if readout == "band":
        return {"mu": (), "h_raw": (), "k_raw": (), "pi_raw": ()}
    if readout == "cband12":
        return {"sig_raw": (CBAND_M,), "o_raw": (CBAND_M,), "c_raw": (CBAND_M,)}
    raise ValueError(f"unknown readout {readout!r}")


# --- R-Band -----------------------------------------------------------------

def band_params(raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "mu": raw["mu"],
        "h": bounded_sigmoid(raw["h_raw"], BAND_H_LO, BAND_H_HI),
        "k": bounded_sigmoid(raw["k_raw"], BAND_K_LO, BAND_K_HI),
        "pi": torch.sigmoid(raw["pi_raw"]),
    }


def _band_apply(z: torch.Tensor, raw: Mapping[str, torch.Tensor]) -> torch.Tensor:
    p = band_params(raw)
    mu = p["mu"].unsqueeze(-1) if p["mu"].dim() else p["mu"]
    h = p["h"].unsqueeze(-1) if p["h"].dim() else p["h"]
    k = p["k"].unsqueeze(-1) if p["k"].dim() else p["k"]
    pi = p["pi"].unsqueeze(-1) if p["pi"].dim() else p["pi"]
    b = torch.sigmoid(k * (z - mu + h)) - torch.sigmoid(k * (z - mu - h))
    return pi * b + (1.0 - pi) * (1.0 - b)


# --- R-CBand12 --------------------------------------------------------------

def cband_params(raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "sigma": bounded_sigmoid(raw["sig_raw"], CBAND_SIG_LO, CBAND_SIG_HI),
        "o": torch.sigmoid(raw["o_raw"]),
        "c": torch.sigmoid(raw["c_raw"]),
        "mu": cband_centres(raw["sig_raw"].device, raw["sig_raw"].dtype),
    }


def _cband_apply(
    z: torch.Tensor, raw: Mapping[str, torch.Tensor], normalization: str | None = None
) -> torch.Tensor:
    """``m(z) = sum_i c_i g_i(z) / (sum_i g_i(z) + eps)``.

    Two evaluations of the same formula:

    ``"eps"``       the literal expression.  Wherever every ``g_i`` underflows
                    below ``eps`` -- which happens between two centres already at
                    ``sigma = 0.025`` (``g ~ 6e-27``) and everywhere past the end
                    centres -- it returns exactly ``0``, silently.
    ``"logsumexp"`` the ``eps -> 0`` limit, evaluated as a softmax over
                    ``log g_i``.  Identical in the well-conditioned regime,
                    and out past the last centre it returns the nearest
                    primitive's ``c_i`` instead of collapsing.

    Default is ``logsumexp`` (REVIEW-impl-WhereA B-4).
    """
    mode = normalization or CBAND_NORMALIZATION
    p = cband_params(raw)
    sigma, o, c, mu = p["sigma"], p["o"], p["c"], p["mu"]
    # z: (..., P) -> (..., P, 1); the 12-vectors get a point axis inserted.
    zz = z.unsqueeze(-1)
    if sigma.dim() > 1:                      # batched parameters
        sigma = sigma.unsqueeze(-2)
        o = o.unsqueeze(-2)
        c = c.unsqueeze(-2)
    if mode == "eps":
        g = o * torch.exp(-0.5 * ((zz - mu) / sigma) ** 2)
        return (c * g).sum(-1) / (g.sum(-1) + CBAND_EPS)
    if mode != "logsumexp":
        raise ValueError(f"unknown cband normalization {mode!r}")
    # logsigmoid, not log(sigmoid(.)): an o_raw of -800 makes sigmoid underflow to
    # 0.0 and log(0) = -inf, which would poison the softmax if every primitive did it.
    log_o = torch.nn.functional.logsigmoid(
        raw["o_raw"].unsqueeze(-2) if raw["o_raw"].dim() > 1 else raw["o_raw"]
    )
    log_g = log_o - 0.5 * ((zz - mu) / sigma) ** 2
    resp = torch.softmax(log_g, dim=-1)      # = g_i / sum_j g_j, computed stably
    return (c * resp).sum(-1)


_APPLY = {"band": _band_apply, "cband12": _cband_apply}


def apply_readout(
    readout: str, z: torch.Tensor, raw: Mapping[str, torch.Tensor],
    normalization: str | None = None,
) -> torch.Tensor:
    """``m(z)`` for one of the two protocol readouts."""
    if readout == "band":
        return _band_apply(z, raw)
    if readout == "cband12":
        return _cband_apply(z, raw, normalization)
    raise ValueError(f"unknown readout {readout!r}")


# --- mirror (s -> -s) -------------------------------------------------------

def mirror_params(readout: str, raw: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Parameters ``rho'`` with ``m(-z; rho') == m(z; rho)`` for all ``z``.

    Band: ``b`` depends on ``z - mu`` symmetrically, so ``mu -> -mu`` suffices.
    CBand12: the centre grid is symmetric (``mu_i = -mu_{M-1-i}``), so reversing
    the per-primitive vectors maps the mixture onto its reflection.
    """
    if readout == "band":
        out = {k: v.clone() for k, v in raw.items()}
        out["mu"] = -raw["mu"]
        return out
    if readout == "cband12":
        return {k: torch.flip(v, dims=(-1,)).clone() for k, v in raw.items()}
    raise ValueError(f"unknown readout {readout!r}")


def describe_params(readout: str, raw: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Interpretable (bounded, not raw) values, for reports and shard payloads."""
    def _l(t: torch.Tensor):
        return t.detach().double().cpu().tolist()

    if readout == "band":
        p = band_params(raw)
        return {"mu": _l(p["mu"]), "h": _l(p["h"]), "k": _l(p["k"]), "pi": _l(p["pi"])}
    if readout == "cband12":
        p = cband_params(raw)
        return {
            "mu_grid": _l(p["mu"]), "sigma": _l(p["sigma"]),
            "o": _l(p["o"]), "c": _l(p["c"]),
        }
    raise ValueError(f"unknown readout {readout!r}")


def bounds_report(readout: str, raw: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Protocol 14.6 -- readout boundary test evidence.

    The intervals are checked **closed**.  ``sigmoid(x)`` returns exactly ``1.0``
    in float64 once ``x >= 37``, which an L-BFGS fit reaches whenever it commits
    to a hard polarity or a fully-open primitive; a strict ``pi < 1`` then reads
    that as "out of bounds" even though hitting the endpoint is precisely what
    the bounded-sigmoid parameterisation guarantees, and ``m(z)`` stays perfectly
    well defined there (``pi = 1`` is a pure band-pass).  What the strict test was
    really detecting -- a parameter pinned at its limit -- is reported separately
    as ``*_saturated``, which is diagnostic, not a failure.
    """
    if readout == "band":
        p = band_params(raw)
        return {
            "h_in_bounds": bool(((p["h"] >= BAND_H_LO) & (p["h"] <= BAND_H_HI)).all()),
            "k_in_bounds": bool(((p["k"] >= BAND_K_LO) & (p["k"] <= BAND_K_HI)).all()),
            "pi_in_bounds": bool(((p["pi"] >= 0.0) & (p["pi"] <= 1.0)).all()),
            "h_positive": bool((p["h"] > 0).all()),          # protocol 4.3: h > 0
            "pi_saturated": bool(((p["pi"] <= 0.0) | (p["pi"] >= 1.0)).any()),
            "h_range": [float(p["h"].min()), float(p["h"].max())],
            "k_range": [float(p["k"].min()), float(p["k"].max())],
            "pi_range": [float(p["pi"].min()), float(p["pi"].max())],
        }
    if readout == "cband12":
        p = cband_params(raw)
        return {
            "sigma_in_bounds": bool(
                ((p["sigma"] >= CBAND_SIG_LO) & (p["sigma"] <= CBAND_SIG_HI)).all()
            ),
            "o_in_bounds": bool(((p["o"] >= 0.0) & (p["o"] <= 1.0)).all()),
            "c_in_bounds": bool(((p["c"] >= 0.0) & (p["c"] <= 1.0)).all()),
            "o_saturated": bool(((p["o"] <= 0.0) | (p["o"] >= 1.0)).any()),
            "c_saturated": bool(((p["c"] <= 0.0) | (p["c"] >= 1.0)).any()),
            "sigma_range": [float(p["sigma"].min()), float(p["sigma"].max())],
            "o_range": [float(p["o"].min()), float(p["o"].max())],
            "c_range": [float(p["c"].min()), float(p["c"].max())],
            "mu_grid_fixed": bool(
                torch.allclose(p["mu"], cband_centres(p["mu"].device, p["mu"].dtype))
            ),
        }
    raise ValueError(f"unknown readout {readout!r}")
