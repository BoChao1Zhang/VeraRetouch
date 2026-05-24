from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy.interpolate import BSpline

from .action import G_LUT, K_SPLINE, M_ATOMS, R_FREE, ShapeCurveAction


RHO_DEFAULT = torch.tensor(
    [
        0.035,
        0.035,
        0.030,
        0.045,
        0.035,
        0.035,
        0.040,
        0.040,
        0.035,
        0.030,
        0.055,
        0.055,
        0.040,
        0.040,
        0.045,
        0.045,
    ],
    dtype=torch.float32,
)

ATOM_NAMES = (
    "warm_highlight_color",
    "cool_shadow_color",
    "skin_hue_stabilizer",
    "cyan_shadow_twist",
    "blue_sky_luma_chroma",
    "foliage_saturation_luma",
    "magenta_green_axis",
    "yellow_blue_axis",
    "shadow_matte_desat",
    "highlight_rolloff_chroma",
    "cross_process_green",
    "cross_process_orange",
    "vibrance_low_sat",
    "saturation_compress",
    "sepia_tint",
    "cyanotype_tint",
)

HUE_CENTERS = torch.tensor([0.0, 30.0, 60.0, 120.0, 180.0, 220.0, 280.0, 320.0])


def identity_lut(g: int = G_LUT, device: torch.device | str = "cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    x = torch.linspace(0, 1, g, device=device, dtype=dtype)
    r, gg, b = torch.meshgrid(x, x, x, indexing="ij")
    return torch.stack([r, gg, b], dim=-1)


def luma(rgb: torch.Tensor) -> torch.Tensor:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def rgb_to_hsv(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r, g, b = rgb.unbind(dim=-1)
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    delta = maxc - minc
    eps = 1e-6
    hue_r = ((g - b) / (delta + eps)) % 6.0
    hue_g = ((b - r) / (delta + eps)) + 2.0
    hue_b = ((r - g) / (delta + eps)) + 4.0
    hue = torch.where(maxc == r, hue_r, torch.where(maxc == g, hue_g, hue_b))
    hue = torch.where(delta < eps, torch.zeros_like(hue), hue / 6.0)
    sat = torch.where(maxc < eps, torch.zeros_like(maxc), delta / (maxc + eps))
    return hue, sat, maxc


def hsv_to_rgb(h: torch.Tensor, s: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    h6 = (h % 1.0) * 6.0
    i = torch.floor(h6)
    f = h6 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i_mod = (i.to(torch.int64) % 6)
    out = torch.stack(
        [
            torch.where((i_mod == 0) | (i_mod == 5), v, torch.where((i_mod == 1) | (i_mod == 4), q, p)),
            torch.where((i_mod == 1) | (i_mod == 2), v, torch.where((i_mod == 0) | (i_mod == 3), t, p)),
            torch.where((i_mod == 3) | (i_mod == 4), v, torch.where((i_mod == 2) | (i_mod == 5), t, p)),
        ],
        dim=-1,
    )
    return out.clamp(0, 1)


def hue_mask(h: torch.Tensor, center_deg: float, width_deg: float) -> torch.Tensor:
    center = center_deg / 360.0
    width = width_deg / 360.0
    dist = torch.minimum((h - center).abs(), 1.0 - (h - center).abs())
    return torch.exp(-0.5 * (dist / (width / 2.355 + 1e-6)).pow(2))


def soft_range(x: torch.Tensor, lo: float, hi: float, sharp: float = 30.0) -> torch.Tensor:
    return torch.sigmoid(sharp * (x - lo)) * torch.sigmoid(sharp * (hi - x))


def smooth3d(field: torch.Tensor, passes: int = 2) -> torch.Tensor:
    x = field.permute(3, 0, 1, 2).unsqueeze(0)
    kernel_1d = torch.tensor([1.0, 2.0, 1.0], device=field.device, dtype=field.dtype)
    kernel_1d = kernel_1d / kernel_1d.sum()
    for _ in range(passes):
        for dim in (2, 3, 4):
            shape = [1, 1, 1, 1, 1]
            shape[dim] = 3
            kernel = kernel_1d.view(shape).repeat(3, 1, 1, 1, 1)
            pad = [0, 0, 0, 0, 0, 0]
            pad[2 * (4 - dim)] = 1
            pad[2 * (4 - dim) + 1] = 1
            x = F.conv3d(F.pad(x, pad, mode="replicate"), kernel, groups=3)
    return x.squeeze(0).permute(1, 2, 3, 0)


def normalize_atom(delta: torch.Tensor) -> torch.Tensor:
    delta = delta - 0.25 * delta.mean(dim=(0, 1, 2), keepdim=True)
    delta = smooth3d(delta, passes=2)
    amp = torch.quantile(delta.abs().amax(dim=-1).flatten(), 0.95)
    return delta / (amp + 1e-6)


def make_atom(name: str, g: int = G_LUT, device: torch.device | str = "cpu") -> torch.Tensor:
    rgb = identity_lut(g, device)
    y = luma(rgb)
    h, s, _ = rgb_to_hsv(rgb)
    gray = y[..., None].expand_as(rgb)
    vec = lambda values: torch.tensor(values, device=rgb.device, dtype=rgb.dtype)

    if name == "warm_highlight_color":
        m = soft_range(y, 0.55, 1.0) * (0.5 + 0.5 * s)
        delta = m[..., None] * vec([1.0, 0.35, -0.45])
    elif name == "cool_shadow_color":
        delta = soft_range(y, 0.0, 0.45)[..., None] * vec([-0.55, -0.10, 1.0])
    elif name == "skin_hue_stabilizer":
        m = hue_mask(h, 28, 32) * soft_range(s, 0.12, 0.85) * soft_range(y, 0.25, 0.90)
        target = vec([1.0, 0.56, 0.38])
        target = target / target.norm()
        current = rgb / (rgb.norm(dim=-1, keepdim=True) + 1e-6)
        delta = m[..., None] * (target - current) * 0.8
    elif name == "cyan_shadow_twist":
        delta = (soft_range(y, 0.0, 0.50) * hue_mask(h, 200, 70))[..., None] * vec([-0.35, 0.45, 0.65])
    elif name == "blue_sky_luma_chroma":
        m = hue_mask(h, 210, 45) * soft_range(s, 0.15, 0.95) * soft_range(y, 0.35, 1.0)
        delta = m[..., None] * vec([0.05, 0.10, 0.55])
    elif name == "foliage_saturation_luma":
        m = hue_mask(h, 105, 50) * soft_range(s, 0.12, 0.90) * soft_range(y, 0.20, 0.85)
        delta = m[..., None] * vec([-0.10, 0.45, -0.12])
    elif name == "magenta_green_axis":
        delta = soft_range(s, 0.05, 1.0)[..., None] * vec([0.65, -0.75, 0.65])
    elif name == "yellow_blue_axis":
        delta = soft_range(s, 0.05, 1.0)[..., None] * vec([0.55, 0.45, -0.75])
    elif name == "shadow_matte_desat":
        m = soft_range(y, 0.0, 0.35)
        delta = m[..., None] * (gray - rgb) * 1.3 + m[..., None] * vec([0.10, 0.08, 0.06])
    elif name == "highlight_rolloff_chroma":
        m = soft_range(y, 0.70, 1.0)
        delta = m[..., None] * (gray - rgb) * 0.8 + m[..., None] * vec([-0.05, -0.04, -0.02])
    elif name == "cross_process_green":
        delta = (soft_range(y, 0.15, 0.90) * (0.4 + 0.6 * s))[..., None] * vec([-0.25, 0.65, -0.10])
    elif name == "cross_process_orange":
        delta = (soft_range(y, 0.20, 1.0) * (0.4 + 0.6 * s))[..., None] * vec([0.65, 0.28, -0.35])
    elif name == "vibrance_low_sat":
        delta = (soft_range(s, 0.0, 0.35) * soft_range(y, 0.15, 0.95))[..., None] * (rgb - gray) * 1.8
    elif name == "saturation_compress":
        delta = soft_range(s, 0.60, 1.0)[..., None] * (gray - rgb) * 1.2
    elif name == "sepia_tint":
        sepia = torch.stack([y * 1.10, y * 0.88, y * 0.55], dim=-1).clamp(0, 1)
        delta = soft_range(y, 0.05, 0.95)[..., None] * (sepia - rgb)
    elif name == "cyanotype_tint":
        cyan = torch.stack([y * 0.45, y * 0.80, y * 1.10], dim=-1).clamp(0, 1)
        delta = soft_range(y, 0.05, 0.95)[..., None] * (cyan - rgb)
    else:
        raise ValueError(name)
    return normalize_atom(delta)


def build_dictionary(g: int = G_LUT, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    atoms = torch.stack([make_atom(name, g, device) for name in ATOM_NAMES], dim=0)
    return atoms, RHO_DEFAULT.to(device), ATOM_NAMES


@lru_cache(maxsize=16)
def _bspline_basis_np(g: int, k: int) -> np.ndarray:
    n_interior = max(k - 4, 0)
    interior = np.linspace(0, 1, n_interior + 2)[1:-1] if n_interior > 0 else []
    knots = np.concatenate([np.zeros(4), interior, np.ones(4)])
    x = np.linspace(0, 1, g)
    basis = np.zeros((g, k), dtype=np.float32)
    for idx in range(k):
        coeff = np.zeros(k)
        coeff[idx] = 1.0
        basis[:, idx] = BSpline(knots, coeff, 3, extrapolate=False)(x)
    return np.nan_to_num(basis, nan=0.0)


def bspline_basis(g: int = G_LUT, k: int = K_SPLINE, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.from_numpy(_bspline_basis_np(g, k)).to(device=device, dtype=torch.float32)


def softclip_identity(x: torch.Tensor, beta: float = 20.0) -> torch.Tensor:
    return x + F.softplus(-beta * x) / beta - F.softplus(beta * (x - 1.0)) / beta


def apply_curve_lut(lut: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
    batch = lut.shape[0]
    values = []
    for channel in range(3):
        x = lut[..., channel].clamp(0, 1) * 7.0
        idx0 = torch.floor(x).long().clamp(0, 6)
        idx1 = idx0 + 1
        t = (x - idx0.to(x.dtype)).unsqueeze(-1)
        curve = curves[:, channel, :].view(batch, 1, 1, 1, 8).expand(-1, *idx0.shape[1:], -1)
        y0 = torch.gather(curve, -1, idx0.unsqueeze(-1)).squeeze(-1)
        y1 = torch.gather(curve, -1, idx1.unsqueeze(-1)).squeeze(-1)
        values.append(y0 * (1.0 - t.squeeze(-1)) + y1 * t.squeeze(-1))
    return torch.stack(values, dim=-1)


def apply_hsl_lut(lut: torch.Tensor, hsl: torch.Tensor) -> torch.Tensor:
    batch = lut.shape[0]
    h, s, v = rgb_to_hsv(lut)
    h_deg = h * 360.0
    centers = HUE_CENTERS.to(lut.device, lut.dtype)
    dist = torch.minimum((h_deg.unsqueeze(-1) - centers).abs(), 360.0 - (h_deg.unsqueeze(-1) - centers).abs())
    weights = torch.exp(-dist.pow(2) / (2 * 25.0**2))
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    delta = (weights.unsqueeze(-1) * hsl.view(batch, 1, 1, 1, 8, 3)).sum(dim=-2)
    h_new = (h + delta[..., 0] / 360.0) % 1.0
    s_new = (s + delta[..., 1]).clamp(0, 1)
    v_new = (v + delta[..., 2]).clamp(0, 1)
    return hsv_to_rgb(h_new, s_new, v_new)


def apply_wb_lut(lut: torch.Tensor, wb: torch.Tensor) -> torch.Tensor:
    temp = wb[:, 0].view(-1, 1, 1, 1)
    tint = wb[:, 1].view(-1, 1, 1, 1)
    scales = torch.stack([1.0 + temp, 1.0 + tint, 1.0 - temp], dim=-1)
    return (lut * scales).clamp(0, 1)


def main_lut(action: ShapeCurveAction, g: int = G_LUT) -> torch.Tensor:
    batch = action.curves.shape[0]
    lut = identity_lut(g, action.curves.device, action.curves.dtype).unsqueeze(0).repeat(batch, 1, 1, 1, 1)
    lut = apply_curve_lut(lut, action.curves)
    lut = apply_hsl_lut(lut, action.hsl)
    lut = apply_wb_lut(lut, action.wb)
    return lut


def free_tail_lut(action: ShapeCurveAction, g: int = G_LUT) -> torch.Tensor:
    basis = bspline_basis(g, K_SPLINE, action.curves.device).to(action.curves.dtype)
    u = torch.einsum("gk,brk->brg", basis, action.tail_alpha)
    v = torch.einsum("gk,brk->brg", basis, action.tail_beta)
    w = torch.einsum("gk,brk->brg", basis, action.tail_gamma)
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-6)
    w = w / (w.norm(dim=-1, keepdim=True) + 1e-6)
    return torch.einsum("br,brc,bri,brj,brk->bijkc", action.tail_gate, action.tail_color, u, v, w)


def hybrid_lut(
    action: ShapeCurveAction,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    include_dictionary: bool = True,
    include_free_tail: bool = True,
    g: int = G_LUT,
) -> dict[str, torch.Tensor]:
    lut_main = main_lut(action, g)
    if include_dictionary:
        lut_dict = torch.einsum("bm,mijkc,m->bijkc", action.dict_coef, dictionary, rho)
    else:
        lut_dict = torch.zeros_like(lut_main)
    if include_free_tail:
        lut_tail = free_tail_lut(action, g)
    else:
        lut_tail = torch.zeros_like(lut_main)
    lut_pre = lut_main + lut_dict + lut_tail
    lut_final = softclip_identity(lut_pre)
    identity = identity_lut(g, lut_pre.device, lut_pre.dtype).unsqueeze(0)
    return {
        "main": lut_main,
        "dict": lut_dict,
        "tail": lut_tail,
        "pre": lut_pre,
        "final": lut_final,
        "identity": identity,
    }


def apply_lut(lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    lut_chw = lut.permute(0, 4, 1, 2, 3)
    coords = image.permute(0, 2, 3, 1).clamp(0, 1) * 2.0 - 1.0
    grid = coords.unsqueeze(1)[..., [2, 1, 0]]
    out = F.grid_sample(lut_chw, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)


def render_hybrid(
    image: torch.Tensor,
    action: ShapeCurveAction,
    dictionary: torch.Tensor,
    rho: torch.Tensor,
    include_dictionary: bool = True,
    include_free_tail: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    luts = hybrid_lut(action, dictionary, rho, include_dictionary, include_free_tail)
    return apply_lut(luts["final"], image), luts


def lut_stats(action: ShapeCurveAction, luts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    identity = luts["identity"]
    main_delta = luts["main"] - identity
    dict_delta = luts["dict"]
    tail_delta = luts["tail"]
    main_energy = main_delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    dict_energy = dict_delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    tail_energy = tail_delta.pow(2).mean(dim=(1, 2, 3, 4)).sqrt()
    residual = dict_energy + tail_energy
    total = main_energy + residual + 1e-6
    pre = luts["pre"]
    final = luts["final"]
    band = smoothness2_3d(final)
    return {
        "clipping_ratio": ((pre < 0.0) | (pre > 1.0)).float().mean(dim=(1, 2, 3, 4)),
        "gamut_violation_ratio": ((pre < -1e-4) | (pre > 1.0001)).float().mean(dim=(1, 2, 3, 4)),
        "banding_smoothness_penalty": band,
        "main_energy": main_energy,
        "dictionary_energy": dict_energy,
        "free_tail_energy": tail_energy,
        "residual_ratio": residual / total,
        "dictionary_explained_ratio": dict_energy / (residual + 1e-6),
        "active_atom_count": (action.dict_coef.abs() >= 0.03).float().sum(dim=1),
        "inactive_atom_ratio": (action.dict_coef.abs() < 0.03).float().mean(dim=1),
    }


def smoothness2_3d(lut: torch.Tensor) -> torch.Tensor:
    vals = []
    for dim in (1, 2, 3):
        n = lut.shape[dim] - 2
        vals.append((lut.narrow(dim, 2, n) - 2 * lut.narrow(dim, 1, n) + lut.narrow(dim, 0, n)).abs().mean(dim=(1, 2, 3, 4)))
    return sum(vals)


def tv3d(lut: torch.Tensor) -> torch.Tensor:
    return (
        (lut[:, 1:] - lut[:, :-1]).abs().mean()
        + (lut[:, :, 1:] - lut[:, :, :-1]).abs().mean()
        + (lut[:, :, :, 1:] - lut[:, :, :, :-1]).abs().mean()
    )


def gamut_penalty(lut_pre: torch.Tensor) -> torch.Tensor:
    return F.relu(lut_pre - 1.0).pow(2).mean() + F.relu(-lut_pre).pow(2).mean()
