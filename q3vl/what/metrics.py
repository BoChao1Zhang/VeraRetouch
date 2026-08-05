"""Protocol 12 -- LUT metrics, image metrics, negative controls, gates.

12.1  On the fixed uniform set, the natural set and the full 33^3 grid: RGB
      MAE/RMSE/PSNR, CIEDE2000 mean/median/p90/p95, hue angular error and chroma
      error, analytic-vs-baked-tetrahedral MAE/p99/max, out-of-range fraction and
      non-finite counts.  Production bake gate::

          mean RGB MAE <= 1e-4,  p99 RGB error <= 5e-4,  non-finite = 0

12.2  After rendering ``I_out = I_in + m (T(I_in) - I_in)``: PSNR, SSIM, LPIPS,
      CIEDE2000, partitioned into mask interior / 3px boundary band / exterior and
      stratified; global and local reported separately.

12.3  Causal negative controls and anti-collapse: instruction shuffle, image
      shuffle, ``z_style`` effective rank and per-dimension variance, the Spearman
      correlation between ``z_style`` pairwise distance and GT function distance,
      the distribution of active existence/opacity, cross-sample variance of the
      predicted geometry and payload, ``WC-1/2/3`` versus ``WC-0`` paired, and the
      distance to strict no-where and oracle-where.

12.4  Selection is lexicographic and happens only on ``V_what``.

LPIPS is the one metric with an external dependency (a pretrained AlexNet/VGG).
:func:`image_metrics` computes it when a backend is supplied and reports ``None``
otherwise, and the missing value is listed in ``PREFLIGHT_WHAT_PENDING.md``
rather than silently replaced by something else.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

from .colorspace import chroma_error, delta_e00, hue_angular_error_deg
from .config import BAKE_GATE, FINITE_GATE, SELECTION_ORDER

__all__ = [
    "percentile", "lut_metrics", "bake_metrics", "ssim", "psnr", "image_metrics",
    "compose_image", "style_diagnostics", "activation_stats", "spearman",
    "effective_rank", "evaluate_gates", "lexicographic_best",
]


def percentile(x: torch.Tensor, p: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.detach().flatten().float(), p / 100.0))


def _finite(x: torch.Tensor) -> int:
    return int((~torch.isfinite(x)).sum())


# --- protocol 12.1 ----------------------------------------------------------

def lut_metrics(t_pred: torch.Tensor, t_gt: torch.Tensor,
                prefix: str = "") -> dict[str, float]:
    """RGB / colour-difference metrics between two LUT function evaluations."""
    d = (t_pred - t_gt)
    mae = d.abs().mean()
    mse = (d * d).mean()
    de = delta_e00(t_pred.clamp(0, 1), t_gt.clamp(0, 1))
    out = {
        "mae": float(mae), "rmse": float(torch.sqrt(mse)),
        "psnr": float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12))),
        "de00_mean": float(de.mean()), "de00_median": percentile(de, 50),
        "de00_p90": percentile(de, 90), "de00_p95": percentile(de, 95),
        "hue_err_deg_mean": float(hue_angular_error_deg(t_pred.clamp(0, 1),
                                                        t_gt.clamp(0, 1)).mean()),
        "chroma_err_mean": float(chroma_error(t_pred.clamp(0, 1),
                                              t_gt.clamp(0, 1)).mean()),
        "out_of_range_frac": float(((t_pred < 0.0) | (t_pred > 1.0)).float().mean()),
        "non_finite": _finite(t_pred),
    }
    return {f"{prefix}{k}": v for k, v in out.items()} if prefix else out


def bake_metrics(t_analytic: torch.Tensor, t_read: torch.Tensor) -> dict[str, float]:
    """The protocol 12.1 bake gate's three numbers, plus the max.

    ``bake_mae_mean`` is the mean absolute RGB error over all points and
    channels; ``bake_err_p99`` is the 99th percentile of the *per-channel*
    absolute error, which is what "p99 RGB error" bounds.
    """
    err = (t_read - t_analytic).abs()
    return {
        "bake_mae_mean": float(err.mean()),
        "bake_err_p99": percentile(err, 99),
        "bake_err_max": float(err.max()) if err.numel() else float("nan"),
        "bake_non_finite": float(_finite(t_read)),
    }


# --- protocol 12.2 ----------------------------------------------------------

def compose_image(i_in: torch.Tensor, t_of_i_in: torch.Tensor,
                  m: torch.Tensor) -> torch.Tensor:
    """``I_out = I_in + m (T(I_in) - I_in)`` -- protocol 0 / 12.2, verbatim."""
    if m.dim() == i_in.dim() - 1:
        m = m.unsqueeze(-3)
    return (i_in + m * (t_of_i_in - i_in)).clamp(0.0, 1.0)


def psnr(a: torch.Tensor, b: torch.Tensor,
         mask: torch.Tensor | None = None) -> float:
    d = (a - b) ** 2
    if mask is not None:
        w = mask.unsqueeze(-3) if mask.dim() == a.dim() - 1 else mask
        denom = w.sum() * a.shape[-3]
        if float(denom) <= 0:
            return float("nan")
        mse = (d * w).sum() / denom
    else:
        mse = d.mean()
    return float(10.0 * torch.log10(1.0 / mse.clamp_min(1e-12)))


def _gaussian_window(size: int = 11, sigma: float = 1.5, device=None,
                     dtype=torch.float32) -> torch.Tensor:
    x = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0,
         size: int = 11, sigma: float = 1.5) -> float:
    """Standard Wang et al. SSIM on ``(C, H, W)`` or ``(B, C, H, W)`` in [0,1]."""
    x = a.unsqueeze(0) if a.dim() == 3 else a
    y = b.unsqueeze(0) if b.dim() == 3 else b
    c = x.shape[1]
    w = _gaussian_window(size, sigma, x.device, x.dtype).expand(c, 1, size, size)
    pad = size // 2
    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mxx, myy, mxy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sxx = F.conv2d(x * x, w, padding=pad, groups=c) - mxx
    syy = F.conv2d(y * y, w, padding=pad, groups=c) - myy
    sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mxy
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    s = ((2 * mxy + c1) * (2 * sxy + c2)) / ((mxx + myy + c1) * (sxx + syy + c2))
    return float(s.mean())


def boundary_band(mask: torch.Tensor, width: int = 3) -> torch.Tensor:
    """The ``width``-pixel band around the mask's 0.5 level set."""
    m = (mask > 0.5).float().unsqueeze(0).unsqueeze(0)
    k = 2 * width + 1
    dil = F.max_pool2d(m, k, stride=1, padding=width)
    ero = -F.max_pool2d(-m, k, stride=1, padding=width)
    return (dil - ero).squeeze(0).squeeze(0)


def image_metrics(i_out: torch.Tensor, i_tar: torch.Tensor,
                  mask: torch.Tensor | None = None, *, band_px: int = 3,
                  lpips_fn=None) -> dict[str, float]:
    """Protocol 12.2 for one image: whole, interior, boundary band, exterior."""
    out: dict[str, float] = {
        "psnr": psnr(i_out, i_tar),
        "ssim": ssim(i_out, i_tar),
        "de00_mean": float(delta_e00(i_out.permute(1, 2, 0),
                                     i_tar.permute(1, 2, 0)).mean()),
    }
    de = delta_e00(i_out.permute(1, 2, 0), i_tar.permute(1, 2, 0))
    out["de00_median"] = percentile(de, 50)
    if mask is not None:
        band = boundary_band(mask, band_px)
        inside = (mask > 0.5).float() * (1 - band)
        outside = (mask <= 0.5).float() * (1 - band)
        for name, w in (("inside", inside), ("boundary", band), ("outside", outside)):
            s = float(w.sum())
            out[f"{name}_frac"] = s / float(w.numel())
            if s > 0:
                out[f"{name}_de00_mean"] = float((de * w).sum() / s)
                out[f"{name}_de00_median"] = percentile(de[w > 0], 50)
                out[f"{name}_psnr"] = psnr(i_out, i_tar, w)
            else:
                out[f"{name}_de00_mean"] = float("nan")
                out[f"{name}_de00_median"] = float("nan")
                out[f"{name}_psnr"] = float("nan")
    out["lpips"] = (float(lpips_fn(i_out.unsqueeze(0), i_tar.unsqueeze(0)))
                    if lpips_fn is not None else float("nan"))
    return out


# --- protocol 12.3 ----------------------------------------------------------

def effective_rank(z: torch.Tensor) -> float:
    """``exp(H(sigma / sum sigma))`` -- the entropy-based effective rank."""
    if z.shape[0] < 2:
        return float("nan")
    s = torch.linalg.svdvals(z.float() - z.float().mean(0, keepdim=True))
    p = s / s.sum().clamp_min(1e-12)
    h = -(p * torch.log(p.clamp_min(1e-12))).sum()
    return float(torch.exp(h))


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Spearman rank correlation of two 1-D tensors."""
    if a.numel() < 3:
        return float("nan")

    def rank(x: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(x)
        r = torch.empty_like(order, dtype=torch.float64)
        r[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
        return r

    ra, rb = rank(a.flatten().double()), rank(b.flatten().double())
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra @ rb) / denom)


def style_diagnostics(z_style: torch.Tensor, z_gt: torch.Tensor | None = None,
                      u_gt: torch.Tensor | None = None) -> dict[str, float]:
    """Anti-collapse evidence for the continuous code (protocol 12.3).

    Protocol 12.3 asks for "the Spearman correlation between ``z_style`` pairwise
    distance and **GT LUT function distance**".  The headline number
    (``z_dist_spearman_vs_func``) is therefore computed against the raw
    ``||u_i - u_j||`` -- the unnormalised, un-projected function distance
    (review blocker B-4).

    ``z_dist_spearman_vs_zgt`` -- the correlation against the *direction-only*
    ``z_gt`` code -- is kept beside it, clearly named, but it is not the
    diagnostic: ``z_gt`` discards the magnitude of ``u``, so a model that got
    every hue right and every strength wrong scores well on it.  Reporting only
    that number would be asking the loss whether it converged, not whether the
    code encodes the function.
    """
    z = z_style.detach().float()
    out = {
        "z_effective_rank": effective_rank(z),
        "z_dim_var_min": float(z.var(0, unbiased=False).min()),
        "z_dim_var_mean": float(z.var(0, unbiased=False).mean()),
        "z_dim_var_max": float(z.var(0, unbiased=False).max()),
        "n": int(z.shape[0]),
    }
    if z.shape[0] < 3:
        return out
    zs = F.normalize(z, dim=-1)
    iu = torch.triu_indices(z.shape[0], z.shape[0], offset=1, device=z.device)
    d_style = (zs[iu[0]] - zs[iu[1]]).norm(dim=-1)
    if u_gt is not None:
        u = u_gt.detach().float().to(z.device)
        out["z_dist_spearman_vs_func"] = spearman(
            d_style, (u[iu[0]] - u[iu[1]]).norm(dim=-1))
    if z_gt is not None:
        zg = F.normalize(z_gt.detach().float(), dim=-1)
        out["z_dist_spearman_vs_zgt"] = spearman(
            d_style, (zg[iu[0]] - zg[iu[1]]).norm(dim=-1))
    return out


def activation_stats(params: Mapping[str, torch.Tensor]) -> dict[str, float]:
    """The 48 gates' activation distribution and the payload's cross-sample spread."""
    o, e = params["opacity"].detach(), params["existence"].detach()
    active = ((o > 0.5) & (e > 0.5)).float()
    out = {
        "opacity_mean": float(o.mean()), "existence_mean": float(e.mean()),
        "n_active_mean": float(active.sum(-1).mean()),
        "n_active_min": float(active.sum(-1).min()),
        "n_active_max": float(active.sum(-1).max()),
        "frac_all_on": float((active.mean(-1) > 0.99).float().mean()),
        "frac_all_off": float((active.mean(-1) < 0.01).float().mean()),
    }
    if params["mu"].shape[0] > 1:
        out["mu_cross_sample_std"] = float(params["mu"].detach().std(0).mean())
        out["M_cross_sample_std"] = float(params["M"].detach().std(0).mean())
        out["b_cross_sample_std"] = float(params["b"].detach().std(0).mean())
    return out


# --- gates and selection ----------------------------------------------------

def evaluate_gates(metrics: Mapping[str, Any],
                   gates: Sequence[tuple[str, str, float]] = BAKE_GATE + FINITE_GATE
                   ) -> dict[str, Any]:
    rows = []
    for key, op, thr in gates:
        v = metrics.get(key)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            rows.append({"metric": key, "value": v, "op": op, "threshold": thr,
                         "pass": False, "reason": "missing"})
            continue
        ok = (float(v) >= thr) if op == ">=" else (float(v) <= thr)
        rows.append({"metric": key, "value": float(v), "op": op,
                     "threshold": thr, "pass": bool(ok)})
    return {"all_pass": all(r["pass"] for r in rows), "rows": rows}


def lexicographic_best(candidates: Sequence[Mapping[str, Any]],
                       order: Sequence[tuple[str, bool]] = SELECTION_ORDER
                       ) -> Mapping[str, Any] | None:
    """Protocol 12.4's ordering.  ``True`` = larger is better."""
    usable = [c for c in candidates if c is not None]
    if not usable:
        return None

    def key(c: Mapping[str, Any]):
        out = []
        for k, bigger in order:
            v = c.get(k)
            v = math.inf if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
            out.append(-v if bigger else v)
        return tuple(out)

    return min(usable, key=key)
