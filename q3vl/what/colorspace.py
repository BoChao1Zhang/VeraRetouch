"""CIELAB, hue/chroma and CIEDE2000 -- all of it dimensionless where it matters.

Protocol 9.2:

    "Lab/chroma must be normalised to a dimensionless range before entering the
     loss; the unit error of old A0, where absolute Lab values amplified the
     gradient by about 229x, must not be reproduced."

So this module exposes Lab in **two** forms and the loss may only see one:

``srgb_to_lab``       CIELAB in its natural units (L in [0,100], a/b in ~[-128,127]).
                      For *reporting* -- CIEDE2000, hue angular error in degrees.
``srgb_to_lab_norm``  ``(L/100, a/128, b/128)``, every channel O(1).  The only
                      form allowed inside ``L_hc`` and inside the pooled ``v_i``.

The 229x is arithmetic, not folklore: a gradient through ``a`` in raw units is
128x one through ``a/128``, and through ``L`` 100x, so a loss mixing raw Lab with
RGB residuals is dominated by Lab by two orders of magnitude.
``tests/test_colorspace.py`` measures the ratio directly instead of quoting it.

The sRGB -> XYZ matrix and the CIEDE2000 expression are the ones already pinned
in ``model/glut_repro/model_rdg.py`` (which matches skimage/colour-science);
they are restated here so Stage-What does not import the previous campaign's
package, and ``tests/test_colorspace.py`` re-pins them against it.
"""

from __future__ import annotations

import torch

from .config import CHROMA_NORM, LAB_AB_SCALE, LAB_L_SCALE

__all__ = ["srgb_to_lab", "srgb_to_lab_norm", "lab_norm_of_lab", "chroma_hue",
           "hue_cos_diff", "delta_e00", "hue_angular_error_deg", "chroma_error"]

_XN = 0.3127 / 0.3290
_ZN = (1.0 - 0.3127 - 0.3290) / 0.3290
_M_RGB2XYZ = [[0.4123907992659595, 0.3575843393838780, 0.1804807884018343],
              [0.2126390058715104, 0.7151686787677559, 0.0721923153607337],
              [0.0193308187155918, 0.1191947797946259, 0.9505321522496607]]
_EPS = 1e-12


def srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """Gamma-encoded sRGB in [0,1] ``(...,3)`` -> CIELAB (D65, 2 deg) ``(...,3)``."""
    rgb = rgb.clamp(0.0, 1.0)
    lin = torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    xyz = lin @ lin.new_tensor(_M_RGB2XYZ).T
    xr, yr, zr = xyz[..., 0] / _XN, xyz[..., 1], xyz[..., 2] / _ZN
    eps, kappa = 216.0 / 24389.0, 24389.0 / 27.0

    def fn(t: torch.Tensor) -> torch.Tensor:
        return torch.where(t > eps, t.clamp(min=1e-12) ** (1.0 / 3.0),
                           (kappa * t + 16.0) / 116.0)

    fx, fy, fz = fn(xr), fn(yr), fn(zr)
    return torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], -1)


def lab_norm_of_lab(lab: torch.Tensor) -> torch.Tensor:
    """``(L, a, b) -> (L/100, a/128, b/128)``."""
    scale = lab.new_tensor([LAB_L_SCALE, LAB_AB_SCALE, LAB_AB_SCALE])
    return lab / scale


def srgb_to_lab_norm(rgb: torch.Tensor) -> torch.Tensor:
    """The only Lab that may enter a loss or a pooled feature."""
    return lab_norm_of_lab(srgb_to_lab(rgb))


def chroma_hue(lab_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(C, h)`` from *normalised* Lab.  ``C`` in [0, sqrt(2)], ``h`` in radians."""
    a, b = lab_norm[..., 1], lab_norm[..., 2]
    return torch.sqrt(a * a + b * b + _EPS), torch.atan2(b, a)


def hue_cos_diff(lab_pred_n: torch.Tensor, lab_gt_n: torch.Tensor) -> torch.Tensor:
    """``1 - cos(h_pred - h_gt)`` computed without ``atan2``.

    ``cos(h1 - h2) = (a1 a2 + b1 b2) / (C1 C2)`` -- the same number, but the
    gradient does not go through the branch cut at +-pi.  The result is clamped
    into [0, 2] so floating point cannot make the loss negative.

    A fully achromatic point has ``a = b = 0``, so ``cos`` evaluates to ``0`` and
    this function returns **1.0**, not 0 (review nit N-2: the docstring used to
    claim 0).  That is harmless and deliberate: ``L_hc`` multiplies it by
    ``normalize(C_gt)``, which is ~1e-7 there, so a grey ground truth contributes
    nothing whatever this term says.  What matters is that the value is finite and
    bounded rather than an ``atan2`` of ``(0, 0)``.
    """
    a1, b1 = lab_pred_n[..., 1], lab_pred_n[..., 2]
    a2, b2 = lab_gt_n[..., 1], lab_gt_n[..., 2]
    c1 = torch.sqrt(a1 * a1 + b1 * b1 + _EPS)
    c2 = torch.sqrt(a2 * a2 + b2 * b2 + _EPS)
    cos = (a1 * a2 + b1 * b2) / (c1 * c2)
    return (1.0 - cos.clamp(-1.0, 1.0)).clamp(0.0, 2.0)


def normalised_chroma(lab_gt_n: torch.Tensor) -> torch.Tensor:
    """``normalize(C_gt)`` of protocol 9.2 -- in [0, 1] by a **global constant**.

    ``CHROMA_NORM = sqrt(2)`` is the largest chroma the normalised Lab space can
    hold.  Dividing by a per-image or per-batch maximum would be exactly the
    per-image normalisation the campaign red lines forbid, and would make the
    weight of a sample depend on the other samples in its batch.
    """
    c, _ = chroma_hue(lab_gt_n)
    return (c / CHROMA_NORM).clamp(0.0, 1.0)


# --- reporting-only metrics -------------------------------------------------

def hue_angular_error_deg(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
    """Absolute hue difference in degrees, wrapped to [0, 180]."""
    hp = chroma_hue(srgb_to_lab_norm(rgb_pred))[1]
    hg = chroma_hue(srgb_to_lab_norm(rgb_gt))[1]
    d = torch.rad2deg(hp - hg).abs() % 360.0
    return torch.minimum(d, 360.0 - d)


def chroma_error(rgb_pred: torch.Tensor, rgb_gt: torch.Tensor) -> torch.Tensor:
    """Absolute chroma difference in *raw* Lab units (reporting)."""
    lp, lg = srgb_to_lab(rgb_pred), srgb_to_lab(rgb_gt)
    cp = torch.sqrt(lp[..., 1] ** 2 + lp[..., 2] ** 2 + _EPS)
    cg = torch.sqrt(lg[..., 1] ** 2 + lg[..., 2] ** 2 + _EPS)
    return (cp - cg).abs()


def delta_e00(rgb1: torch.Tensor, rgb2: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between two gamma-encoded sRGB tensors ``(...,3)`` -> ``(...)``.

    Reporting and checkpoint selection only.  Protocol 1.4: CIEDE2000 is *not* a
    high-weight backward loss -- its piecewise terms would dominate training.
    """
    l1, l2 = srgb_to_lab(rgb1), srgb_to_lab(rgb2)
    L1, a1, b1 = l1.unbind(-1)
    L2, a2, b2 = l2.unbind(-1)
    C1 = torch.sqrt(a1 * a1 + b1 * b1)
    C2 = torch.sqrt(a2 * a2 + b2 * b2)
    Cb = 0.5 * (C1 + C2)
    Cb7 = Cb ** 7
    G = 0.5 * (1 - torch.sqrt(Cb7 / (Cb7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p = torch.sqrt(a1p * a1p + b1 * b1)
    C2p = torch.sqrt(a2p * a2p + b2 * b2)
    h1p = torch.rad2deg(torch.atan2(b1, a1p)) % 360.0
    h2p = torch.rad2deg(torch.atan2(b2, a2p)) % 360.0
    dLp, dCp = L2 - L1, C2p - C1p
    prod = C1p * C2p
    dh = h2p - h1p
    dhp = torch.where(prod == 0, torch.zeros_like(dh),
                      torch.where(dh > 180.0, dh - 360.0,
                                  torch.where(dh < -180.0, dh + 360.0, dh)))
    dHp = 2.0 * torch.sqrt(prod.clamp(min=0)) * torch.sin(torch.deg2rad(dhp) / 2.0)
    Lbp, Cbp = 0.5 * (L1 + L2), 0.5 * (C1p + C2p)
    hsum, hdiff = h1p + h2p, torch.abs(h1p - h2p)
    hbp = torch.where(
        prod == 0, hsum,
        torch.where(hdiff <= 180.0, 0.5 * hsum,
                    torch.where(hsum < 360.0, 0.5 * (hsum + 360.0), 0.5 * (hsum - 360.0))))
    T = (1 - 0.17 * torch.cos(torch.deg2rad(hbp - 30.0))
         + 0.24 * torch.cos(torch.deg2rad(2 * hbp))
         + 0.32 * torch.cos(torch.deg2rad(3 * hbp + 6.0))
         - 0.20 * torch.cos(torch.deg2rad(4 * hbp - 63.0)))
    dtheta = 30.0 * torch.exp(-(((hbp - 275.0) / 25.0) ** 2))
    Cbp7 = Cbp ** 7
    Rc = 2.0 * torch.sqrt(Cbp7 / (Cbp7 + 25.0 ** 7))
    Lm = (Lbp - 50.0) ** 2
    Sl = 1.0 + 0.015 * Lm / torch.sqrt(20.0 + Lm)
    Sc = 1.0 + 0.045 * Cbp
    Sh = 1.0 + 0.015 * Cbp * T
    Rt = -torch.sin(torch.deg2rad(2 * dtheta)) * Rc
    return torch.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2
                      + Rt * (dCp / Sc) * (dHp / Sh))
