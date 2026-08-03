"""GLUT losses (IMPL_DOSSIER 2.2, section 4.1 of the paper).

L = L_rec + 10 * L_hc + 0.001 * R_sparse
  L_rec    = |y_hat - y|_1
  L_hc     = C_gt * (1 - <h_hat, h_gt>)  with C = sqrt(a^2 + b^2),
             h = (a/C, b/C) in CIELab (chroma-weighted hue cosine distance,
             weighted by TARGET chroma)
  R_sparse = -(1/N) sum_i [o_i log(o_i+eps) + (1-o_i) log(1-o_i+eps)]

A0 runs L_rec-first (dossier 2.3 step 2); the full loss adds < +0.4 dB.
The torch Lab conversion below matches the cubelib sRGB(D65)->Lab pipeline
(colour-science derived D65 white point); the CI check compares against
cubelib.srgb_to_lab on random colors.
"""

from __future__ import annotations

import torch

# D65 white point as derived by colour-science from xy (0.3127, 0.3290)
# (see tools/cube NOTES 5.3 -- colour uses the chromaticity-derived value).
_XN = 0.3127 / 0.3290
_YN = 1.0
_ZN = (1.0 - 0.3127 - 0.3290) / 0.3290

# sRGB -> XYZ (IEC 61966-2-1, D65), the same primaries colour-science uses.
_M_RGB2XYZ = [
    [0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
    [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
    [0.0193308187155918, 0.1191947797946259, 0.9505321522496607],
]


def srgb_to_lab_torch(rgb: torch.Tensor) -> torch.Tensor:
    """Gamma-encoded sRGB in [0,1] -> CIELab (D65).  (...,3) -> (...,3)."""
    rgb = rgb.clamp(0.0, 1.0)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92,
                      ((rgb + 0.055) / 1.055) ** 2.4)
    m = lin.new_tensor(_M_RGB2XYZ)
    xyz = lin @ m.T
    xr = xyz[..., 0] / _XN
    yr = xyz[..., 1] / _YN
    zr = xyz[..., 2] / _ZN
    eps = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    def f(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp(min=1e-12) ** (1.0 / 3.0),
                           (kappa * t + 16.0) / 116.0)
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack([L, a, b], dim=-1)


def l_rec(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """L1 reconstruction, mean over everything."""
    return (pred - gt).abs().mean()


def l_hc(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Chroma-weighted hue cosine distance, weighted by GT chroma."""
    lab_p = srgb_to_lab_torch(pred)
    lab_g = srgb_to_lab_torch(gt)
    ap, bp = lab_p[..., 1], lab_p[..., 2]
    ag, bg = lab_g[..., 1], lab_g[..., 2]
    cp = torch.sqrt(ap * ap + bp * bp + eps)
    cg = torch.sqrt(ag * ag + bg * bg + eps)
    cos_h = (ap * ag + bp * bg) / (cp * cg)
    return (cg * (1.0 - cos_h)).mean()


def r_sparse(opacity_raw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy on clamped opacity, pushes o towards {0,1}."""
    o = opacity_raw.clamp(0.0, 1.0)
    ent = o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps)
    return -ent.mean()


def glut_full_loss(pred: torch.Tensor, gt: torch.Tensor,
                   opacity_raw: torch.Tensor,
                   w_hc: float = 10.0, w_sparse: float = 0.001) -> torch.Tensor:
    return l_rec(pred, gt) + w_hc * l_hc(pred, gt) \
        + w_sparse * r_sparse(opacity_raw)
