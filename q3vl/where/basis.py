"""Protocol 4.2 -- the scalar basis ``s_low`` and its sign canonicalisation.

    w_dir     = normalize(w_raw)
    alpha     = softplus(alpha_raw)
    s_low(p)  = 3 * tanh((w0 + <phi_dir(p), w_dir>) * alpha / 3)

The protocol also fixes the sign of ``w_dir`` -- "the coefficient with the
largest absolute value is positive" -- to kill the ``w -> -w`` non-identifiability.
Flipping ``w_dir`` alone would change the mask, so the canonicalisation flips
``(w_raw, w0)`` *and* mirrors the readout; ``s -> -s`` and ``m(-s; mirror(rho))
== m(s; rho)`` make the whole map exactly mask-preserving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .config import PHI_DIR_DIM, S_SCALE
from .readout import apply_readout, describe_params, mirror_params, param_shapes

__all__ = ["Latent", "w_dir_of", "alpha_of", "s_low", "mask_from_latent",
           "sign_index", "canonicalize", "is_canonical"]

_NORM_EPS = 1e-12


def w_dir_of(w_raw: torch.Tensor) -> torch.Tensor:
    return w_raw / (w_raw.norm(dim=-1, keepdim=True) + _NORM_EPS)


def alpha_of(alpha_raw: torch.Tensor) -> torch.Tensor:
    return F.softplus(alpha_raw)


@dataclass
class Latent:
    """One image's oracle latent for one readout: ``(w*, rho*)``."""

    readout: str
    w0: torch.Tensor           # ()
    alpha_raw: torch.Tensor    # ()
    w_raw: torch.Tensor        # (71,)
    rho: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.w_raw.shape[-1] != PHI_DIR_DIM:
            raise ValueError(
                f"w_raw must be {PHI_DIR_DIM}-dim (protocol 4.2), got {tuple(self.w_raw.shape)}"
            )
        expected = set(param_shapes(self.readout))
        got = set(self.rho)
        if got != expected:
            raise ValueError(f"{self.readout} rho keys {sorted(got)} != {sorted(expected)}")

    # -- derived ------------------------------------------------------------
    @property
    def w_dir(self) -> torch.Tensor:
        return w_dir_of(self.w_raw)

    @property
    def alpha(self) -> torch.Tensor:
        return alpha_of(self.alpha_raw)

    def tensors(self) -> dict[str, torch.Tensor]:
        out = {"w0": self.w0, "alpha_raw": self.alpha_raw, "w_raw": self.w_raw}
        out.update(self.rho)
        return out

    def detach(self) -> "Latent":
        return Latent(
            self.readout,
            self.w0.detach().clone(),
            self.alpha_raw.detach().clone(),
            self.w_raw.detach().clone(),
            {k: v.detach().clone() for k, v in self.rho.items()},
        )

    def to(self, *args, **kwargs) -> "Latent":
        return Latent(
            self.readout,
            self.w0.to(*args, **kwargs),
            self.alpha_raw.to(*args, **kwargs),
            self.w_raw.to(*args, **kwargs),
            {k: v.to(*args, **kwargs) for k, v in self.rho.items()},
        )

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "readout": self.readout,
            "w0": float(self.w0),
            "alpha_raw": float(self.alpha_raw),
            "alpha": float(self.alpha),
            "w_raw": self.w_raw.detach().double().cpu().tolist(),
            "w_dir": self.w_dir.detach().double().cpu().tolist(),
            "rho_raw": {k: v.detach().double().cpu().tolist() for k, v in self.rho.items()},
            "rho": describe_params(self.readout, self.rho),
            "sign_index": int(sign_index(self.w_dir)),
            "canonical": bool(is_canonical(self)),
        }
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], dtype=torch.float64, device=None) -> "Latent":
        t = lambda v: torch.tensor(v, dtype=dtype, device=device)  # noqa: E731
        return cls(
            d["readout"], t(d["w0"]), t(d["alpha_raw"]), t(d["w_raw"]),
            {k: t(v) for k, v in d["rho_raw"].items()},
        )


def s_low(phi_dir: torch.Tensor, latent: Latent) -> torch.Tensor:
    """``(P,)`` scalar field on the F_pre grid."""
    q = latent.w0 + latent.alpha * (phi_dir @ latent.w_dir)
    return S_SCALE * torch.tanh(q / S_SCALE)


def mask_from_latent(phi_dir: torch.Tensor, latent: Latent) -> tuple[torch.Tensor, torch.Tensor]:
    """``(m, s)`` on the low-res grid."""
    s = s_low(phi_dir, latent)
    return apply_readout(latent.readout, s, latent.rho), s


# --- sign canonicalisation --------------------------------------------------

def sign_index(w_dir: torch.Tensor) -> int:
    """Index of the largest-|.| coefficient (first one wins on a tie)."""
    return int(torch.argmax(w_dir.abs()))


def is_canonical(latent: Latent, tol: float = 0.0) -> bool:
    wd = latent.w_dir
    return bool(wd[sign_index(wd)] > tol)


def canonicalize(latent: Latent) -> Latent:
    """Return an equivalent latent whose ``w_dir`` obeys the protocol sign rule.

    ``mask_from_latent`` is invariant under this map (asserted in tests to
    1e-12 in float64).
    """
    wd = latent.w_dir
    if float(wd.abs().max()) <= _NORM_EPS:
        # w_raw == 0: the direction is undefined, s is constant.  Nothing to
        # canonicalise; the caller sees this through ``alpha`` / the fit report.
        return latent.detach()
    if float(wd[sign_index(wd)]) > 0:
        return latent.detach()
    flipped = Latent(
        latent.readout,
        (-latent.w0).detach().clone(),
        latent.alpha_raw.detach().clone(),
        (-latent.w_raw).detach().clone(),
        mirror_params(latent.readout, latent.rho),
    )
    return flipped
