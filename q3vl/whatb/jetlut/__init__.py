"""EPR-032 JetLUT: fixed analytic atlas + exact-POU gate + local jets, fitted by
a certified convex L1 (LAD) projection.

Nothing here is condition-dependent: the basis is fully determined by
``(m, c)``, so ``F_theta(x) = x + Phi(x) theta`` is linear in ``theta`` and the
per-LUT teacher is a convex program with one shared Cholesky for the whole
library (PROPOSAL §3, §4).
"""

from q3vl.whatb.jetlut.core import (
    Atlas,
    admm_lad,
    design_matrix,
    fill_distance,
    n_dynamic_params,
    pou_weights,
)

__all__ = [
    "Atlas",
    "admm_lad",
    "design_matrix",
    "fill_distance",
    "n_dynamic_params",
    "pou_weights",
]
