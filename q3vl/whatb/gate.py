"""Identity-anchored strength gate -- ``f_u(x) = x + u * (f_theta(x) - x)``.

EPR-027's one structural change, sitting **outside** the GLUT forward: theta
does not depend on ``u``.  ``u`` is a scalar (per run, per image) or a field
(per pixel); with ``u(p) = alpha(p)`` and ``f_theta = L_l`` the gate reproduces
the data-generating law ``F*(x,p) = (1-alpha)x + alpha L_l(x)``
(``dataset_build/src/construct/rendering.py:311``) exactly -- that is
proposition 3 / family F1.

Five identities the arm must print next to its numbers (EPR-027:345-356) --
they are arithmetic, not results::

    G1   f_u - y_u = u (f_theta - L_l)     => L_rec(u) = u * L_rec(1)
    G2   ||f_u - x|| = u ||f_theta - x||   => the u-monotonicity rate is 1 by construction
    G3   u = 0  => f_u = x, L_hc = 0       (a zero-work step; count it)
    G4   u(p) = alpha(p), f_theta = L_l => f_u = F*   => E_out is 0 by construction
    G5   u in [0,1] with clamp BEFORE the gate => out-of-gamut rate is 0 by construction

``u`` has **no default**.  A gate that silently defaults to 1 is a gate that
can be "wired" while doing nothing, and there is no way to tell from the board.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["identity_gate", "IdentityGate"]


def _as_gate_tensor(u: float | int | Tensor, ref: Tensor) -> Tensor:
    """``u`` onto ``ref``'s device/dtype, broadcast-checked against ``ref``."""
    if isinstance(u, Tensor):
        ut = u.to(device=ref.device, dtype=ref.dtype)
    else:
        ut = torch.as_tensor(float(u), device=ref.device, dtype=ref.dtype)
    try:
        torch.broadcast_shapes(ut.shape, ref.shape)
    except RuntimeError as exc:  # pragma: no cover - message is the point
        raise ValueError(
            f"gate strength of shape {tuple(ut.shape)} does not broadcast against "
            f"the transform output {tuple(ref.shape)}"
        ) from exc
    return ut


def identity_gate(
    x: Tensor,
    y: Tensor,
    u: float | int | Tensor,
    *,
    clamp: bool = True,
    lo: float = 0.0,
    hi: float = 1.0,
) -> Tensor:
    """``x + u * (y - x)``, optionally clamped afterwards (EPR-027's default).

    Parameters
    ----------
    x
        The identity anchor: the query colours ``(B, *S, 3)``.
    y
        ``f_theta(x)``, same shape.  Pass the carrier's ``clamp="none"`` output
        when the run puts the clamp after the gate (``--gate-clamp after``).
    u
        Scalar, or any tensor broadcasting to ``y``: ``(B,1,1)`` per image,
        ``(B,P,1)`` per query, ``(B,H,W,1)`` per pixel.  **Required.**
    clamp
        Clamp the gated value to ``[lo, hi]``.  ``--gate-clamp after`` (default)
        -> ``True``; ``before`` -> the caller clamps ``y`` and passes ``False``,
        which makes the out-of-gamut column degenerate (identity G5).
    """
    if x.shape != y.shape:
        raise ValueError(f"gate anchor {tuple(x.shape)} != transform output {tuple(y.shape)}")
    xr = x.to(device=y.device, dtype=y.dtype)
    ut = _as_gate_tensor(u, y)
    out = xr + ut * (y - xr)
    return out.clamp(lo, hi) if clamp else out


class IdentityGate(nn.Module):
    """Zero-parameter ``nn.Module`` form of :func:`identity_gate`.

    Exists only so ``--gate-clamp`` can live in module state and reach
    ``run_setup.json``; ``forward`` still refuses to run without ``u``.
    """

    def __init__(self, *, clamp_after: bool = True, lo: float = 0.0, hi: float = 1.0) -> None:
        super().__init__()
        self.clamp_after = bool(clamp_after)
        self.lo, self.hi = float(lo), float(hi)

    def extra_repr(self) -> str:
        return f"clamp_after={self.clamp_after}, range=({self.lo}, {self.hi})"

    @property
    def config(self) -> dict[str, object]:
        return {"gate_clamp": "after" if self.clamp_after else "before", "gate_range": [self.lo, self.hi]}

    def forward(self, x: Tensor, y: Tensor, u: float | int | Tensor) -> Tensor:  # noqa: D102
        return identity_gate(x, y, u, clamp=self.clamp_after, lo=self.lo, hi=self.hi)
