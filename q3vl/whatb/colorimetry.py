"""sRGB -> CIELab and the two colour differences.  **One** copy for the package.

Frozen block "新建包落点": the CIELab / dE00 implementation is written once and
shared by the criteria side and the loss side, so the two cannot drift.  The
training-side ``losses.py`` (``L_hc``'s hue term) must import ``srgb_to_lab``
from here rather than convert on its own.

Conventions, all pinned by tests:

* input sRGB is **float in [0,1]**, last dimension 3, any leading shape;
* the EOTF is the piecewise sRGB one (threshold 0.04045, a = 0.055) and the
  matrix is the sRGB/D65 primaries, white point D65 = (0.95047, 1, 1.08883)
  (HANDOFF §4.C: "每色转 CIELab（D65，sRGB EOTF）");
* ``delta_e76`` is the plain Lab L2 -- the metric the pre-registered baseline
  floors (B0 32.79 / B1 25.33 / B2 35.37 / B4 9.93 / B6 10.17) were measured in;
* ``delta_e00`` is CIEDE2000 with kL = kC = kH = 1 -- the headline metric.

Everything runs on the device and in the dtype of its input.  Nothing in this
module moves a tensor to the CPU: the topk tie-break lesson (CPU and CUDA
disagreed enough to move an IoU by 0.296) generalises to "compute the criterion
where the tensor already is".
"""

from __future__ import annotations

import math

import torch

__all__ = [
    "D65_WHITE",
    "srgb_to_linear",
    "linear_to_srgb",
    "srgb_to_xyz",
    "xyz_to_lab",
    "srgb_to_lab",
    "delta_e76",
    "delta_e00",
    "delta_e00_srgb",
    "delta_e76_srgb",
    "chroma_hue",
]

#: CIE D65, the sRGB reference white (2 degree observer)
D65_WHITE: tuple[float, float, float] = (0.95047, 1.00000, 1.08883)

#: sRGB (IEC 61966-2-1) primaries -> CIE XYZ, D65
_M_SRGB_TO_XYZ: tuple[tuple[float, float, float], ...] = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

_LAB_DELTA = 6.0 / 29.0
_POW25_7 = 25.0 ** 7


def _safe_sqrt(s: torch.Tensor) -> torch.Tensor:
    """``sqrt(s)`` with a **finite** backward at ``s == 0``.

    ``d sqrt/ds = 1/(2 sqrt(s))`` is ``inf`` at zero, and every downstream mask
    then turns it into ``0 * inf = NaN`` -- masking in the forward cannot save a
    backward.  Measured (EPR-029, and the review that found the same thing on
    five arms): one exactly neutral colour anywhere in a 8192-colour batch
    poisons every head's gradient, and the degeneracy guard only notices
    thousands of steps later.

    The double-``where`` below is the standard remedy: the value fed to ``sqrt``
    is replaced by 1 wherever ``s == 0``, so the derivative there is finite, and
    the result is then replaced by the exact 0.  **The forward is bit-identical
    to ``sqrt(s.clamp_min(0))``**; only the subgradient at 0 changes, from
    ``NaN`` to 0 (0 is the conventional subgradient of ``|.|`` at the origin).
    """
    s = s.clamp_min(0.0)
    ok = s > 0
    return torch.where(ok, torch.sqrt(torch.where(ok, s, torch.ones_like(s))),
                       torch.zeros_like(s))


def _check_rgb(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(x).__name__}")
    if x.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3, got {tuple(x.shape)}")
    return x


def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB EOTF.  ``rgb`` in [0,1], same shape out."""
    rgb = _check_rgb(rgb, "rgb")
    lo = rgb / 12.92
    hi = ((rgb.clamp_min(0.0) + 0.055) / 1.055) ** 2.4
    return torch.where(rgb <= 0.04045, lo, hi)


def linear_to_srgb(lin: torch.Tensor) -> torch.Tensor:
    """Inverse EOTF (diagnostics / visualisation only)."""
    lin = _check_rgb(lin, "lin")
    lo = lin * 12.92
    hi = 1.055 * lin.clamp_min(0.0) ** (1.0 / 2.4) - 0.055
    return torch.where(lin <= 0.0031308, lo, hi)


def srgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    lin = srgb_to_linear(rgb)
    m = torch.tensor(_M_SRGB_TO_XYZ, dtype=lin.dtype, device=lin.device)
    return lin @ m.transpose(0, 1)


def xyz_to_lab(xyz: torch.Tensor, white: tuple[float, float, float] = D65_WHITE
               ) -> torch.Tensor:
    xyz = _check_rgb(xyz, "xyz")
    w = torch.tensor(white, dtype=xyz.dtype, device=xyz.device)
    t = xyz / w
    d3 = _LAB_DELTA ** 3
    # ``clamp_min(d3)`` inside the cube root, not ``clamp_min(0)``: at ``t == 0``
    # (pure black -- a legal training colour, level 0 of the 128^3 grid, and a
    # legal clamped prediction) the *unselected* branch's derivative is
    # ``inf``, and ``torch.where``'s backward multiplies it by the zero mask,
    # giving ``0 * inf = NaN``.  Clamping the base to the branch boundary makes
    # that derivative finite while leaving every selected value untouched:
    # where the branch is taken, ``t > d3`` and the clamp is the identity, so
    # **the forward is bit-identical**.
    f = torch.where(t > d3,
                    t.clamp_min(d3) ** (1.0 / 3.0),
                    t / (3.0 * _LAB_DELTA ** 2) + 4.0 / 29.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    return torch.stack((116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)),
                       dim=-1)


def srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """``(..., 3)`` sRGB in [0,1] -> ``(..., 3)`` CIELab (D65)."""
    return xyz_to_lab(srgb_to_xyz(rgb))


def chroma_hue(lab: torch.Tensor, eps_c: float = 1e-3
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(C, h_unit, valid)`` for GLUT Eq.7.

    ``C = sqrt(a^2 + b^2)``; ``h_unit = (a, b) / max(C, eps_c)``; ``valid`` is the
    frozen-block hard mask ``1[C >= eps_c]`` (eps_c = 1e-3), whose False count is
    what the trainer logs as ``n_hc_masked``.  Kept here so the loss and the
    criteria use the same chroma convention.

    **Gradient.**  ``C`` goes through :func:`_safe_sqrt`, so an exactly neutral
    colour (``a == b == 0``: pure black, pure white, any grey -- and a clamp to
    black or white produces one) has a finite backward instead of ``0/0``.  The
    forward is unchanged, so the frozen ``h = (a, b) / max(C, eps_c)`` spelling
    and every published number stay exactly as they were.
    """
    a, b = lab[..., 1], lab[..., 2]
    c = _safe_sqrt(a * a + b * b)
    denom = c.clamp_min(eps_c)
    h = torch.stack((a / denom, b / denom), dim=-1)
    return c, h, c >= eps_c


def delta_e76(lab1: torch.Tensor, lab2: torch.Tensor) -> torch.Tensor:
    """CIE76: Euclidean distance in Lab.  ``(..., 3), (..., 3) -> (...)``."""
    _check_rgb(lab1, "lab1")
    _check_rgb(lab2, "lab2")
    d = lab1 - lab2
    # _safe_sqrt: two identical Lab values are the common case (a prediction
    # that has converged, an unchanged pixel) and the plain sqrt backward is
    # NaN there (W8 of the review).
    return _safe_sqrt((d * d).sum(-1))


def delta_e00(lab1: torch.Tensor, lab2: torch.Tensor,
              *, k_l: float = 1.0, k_c: float = 1.0, k_h: float = 1.0
              ) -> torch.Tensor:
    """CIEDE2000.  ``(..., 3), (..., 3) -> (...)``, on the inputs' device.

    Implementation follows Sharma, Wu & Dalal (2005), "The CIEDE2000
    Color-Difference Formula", Eq. 2-24, including the two discontinuous
    branches (mean hue when one chroma is zero; the +/-360 wrap).  The unit test
    checks it against the authors' own 34-pair test table, fetched from
    ``https://hajim.rochester.edu/ece/sites/gsharma/ciede2000/dataNprograms/
    ciede2000testdata.txt`` (HTTP 200, 1830 B, 2026-08-15).
    """
    _check_rgb(lab1, "lab1")
    _check_rgb(lab2, "lab2")
    if lab1.device != lab2.device:
        raise ValueError(
            f"delta_e00 operands are on different devices ({lab1.device} vs "
            f"{lab2.device}); the criterion is computed where the tensors are")
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    c1 = _safe_sqrt(a1 * a1 + b1 * b1)
    c2 = _safe_sqrt(a2 * a2 + b2 * b2)
    c_bar = 0.5 * (c1 + c2)
    c_bar7 = c_bar ** 7
    g = 0.5 * (1.0 - _safe_sqrt(c_bar7 / (c_bar7 + _POW25_7)))

    a1p, a2p = (1.0 + g) * a1, (1.0 + g) * a2
    c1p = _safe_sqrt(a1p * a1p + b1 * b1)
    c2p = _safe_sqrt(a2p * a2p + b2 * b2)

    deg = 180.0 / math.pi
    zero = torch.zeros_like(c1p)
    h1p = torch.where((a1p == 0) & (b1 == 0), zero,
                      torch.atan2(b1, a1p) * deg) % 360.0
    h2p = torch.where((a2p == 0) & (b2 == 0), zero,
                      torch.atan2(b2, a2p) * deg) % 360.0

    cprod_zero = (c1p * c2p) == 0

    dl = l2 - l1
    dc = c2p - c1p
    dh_raw = h2p - h1p
    dh = torch.where(dh_raw > 180.0, dh_raw - 360.0,
                     torch.where(dh_raw < -180.0, dh_raw + 360.0, dh_raw))
    dh = torch.where(cprod_zero, zero, dh)
    big_dh = 2.0 * _safe_sqrt(c1p * c2p) * torch.sin(0.5 * dh / deg)

    l_bar = 0.5 * (l1 + l2)
    cp_bar = 0.5 * (c1p + c2p)
    hsum = h1p + h2p
    habs = torch.abs(h1p - h2p)
    h_bar = torch.where(
        cprod_zero, hsum,
        torch.where(habs <= 180.0, 0.5 * hsum,
                    torch.where(hsum < 360.0, 0.5 * (hsum + 360.0),
                                0.5 * (hsum - 360.0))))

    t = (1.0
         - 0.17 * torch.cos((h_bar - 30.0) / deg)
         + 0.24 * torch.cos((2.0 * h_bar) / deg)
         + 0.32 * torch.cos((3.0 * h_bar + 6.0) / deg)
         - 0.20 * torch.cos((4.0 * h_bar - 63.0) / deg))

    d_theta = 30.0 * torch.exp(-(((h_bar - 275.0) / 25.0) ** 2))
    cp_bar7 = cp_bar ** 7
    r_c = 2.0 * _safe_sqrt(cp_bar7 / (cp_bar7 + _POW25_7))
    lm50 = (l_bar - 50.0) ** 2
    s_l = 1.0 + 0.015 * lm50 / _safe_sqrt(20.0 + lm50)
    s_c = 1.0 + 0.045 * cp_bar
    s_h = 1.0 + 0.015 * cp_bar * t
    r_t = -torch.sin((2.0 * d_theta) / deg) * r_c

    tl = dl / (k_l * s_l)
    tc = dc / (k_c * s_c)
    th = big_dh / (k_h * s_h)
    return _safe_sqrt(tl * tl + tc * tc + th * th + r_t * tc * th)


def delta_e00_srgb(rgb1: torch.Tensor, rgb2: torch.Tensor) -> torch.Tensor:
    """dE00 between two sRGB tensors in [0,1].  ``(..., 3) -> (...)``."""
    return delta_e00(srgb_to_lab(rgb1), srgb_to_lab(rgb2))


def delta_e76_srgb(rgb1: torch.Tensor, rgb2: torch.Tensor) -> torch.Tensor:
    """dE76 between two sRGB tensors in [0,1] (the baseline-floor protocol)."""
    return delta_e76(srgb_to_lab(rgb1), srgb_to_lab(rgb2))
