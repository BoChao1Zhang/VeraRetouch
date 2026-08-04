"""Protocol 4.4 / 10.2 -- per-image multi-start L-BFGS oracle fit.

    "Where-A uses only the GT masks of local l1-l6, fits an oracle ``w*, rho*``
     per image with multi-start L-BFGS ..."
    "The L-BFGS oracle fit uses float64, multiple starts and a fixed tolerance;
     failed samples go into an explicit rejection / fit report and are never
     silently replaced by a zero vector."

The fit runs on the F_pre grid (``P = grid_h * grid_w`` points).  Full-resolution
numbers are produced by :func:`evaluate_latent`, which routes the fitted scalar
through the one sanctioned guided upsample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .basis import Latent, alpha_of, canonicalize, mask_from_latent, w_dir_of
from .config import CBAND_M, FitConfig, PHI_DIR_DIM, S_SCALE, UpsampleConfig
from .readout import (
    CBAND_SIG_HI,
    CBAND_SIG_LO,
    apply_readout,
    bounds_report,
    cband_centres,
    inv_bounded_sigmoid,
    param_shapes,
)
from .upsample import combine_then_upsample

__all__ = ["FitResult", "fit_latent", "evaluate_latent", "mask_metrics",
           "objective_value", "OBJECTIVES"]

OBJECTIVES = ("soft_iou_minmax", "mse", "bce")
_EPS = 1e-6


# --- objectives and metrics -------------------------------------------------

def soft_iou_minmax(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.minimum(a, b).sum() / (torch.maximum(a, b).sum() + _EPS)


def soft_iou_prod(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a * b).sum()
    return inter / (a.sum() + b.sum() - inter + _EPS)


def objective_value(kind: str, m: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Scalar loss.  ``soft_iou_minmax`` is the PLAN v2 L205 / E2 oracle-fit
    objective; ``mse`` is the default for anything that trains a *parameter*
    (red line: IoU must not be an optimisation target for trained parameters)."""
    if kind == "soft_iou_minmax":
        return 1.0 - soft_iou_minmax(m, t)
    if kind == "mse":
        return F.mse_loss(m, t)
    if kind == "bce":
        return F.binary_cross_entropy(m.clamp(1e-6, 1 - 1e-6), t.clamp(0.0, 1.0))
    raise ValueError(f"unknown objective {kind!r}; expected one of {OBJECTIVES}")


def mask_metrics(m: torch.Tensor, t: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        return {
            "soft_iou_minmax": float(soft_iou_minmax(m, t)),
            "soft_iou_minmax_comp": float(soft_iou_minmax(1 - m, 1 - t)),
            "soft_iou_prod": float(soft_iou_prod(m, t)),
            "mae": float((m - t).abs().mean()),
            "mse": float(((m - t) ** 2).mean()),
            "pred_mean": float(m.mean()),
            "pred_std": float(m.std(unbiased=False)),
            "target_mean": float(t.mean()),
            "target_std": float(t.std(unbiased=False)),
        }


# --- starts -----------------------------------------------------------------

def _softplus_inv(y: float) -> float:
    y = max(float(y), 1e-6)
    return float(y + np.log(-np.expm1(-y))) if y < 20 else float(y)


def _lsq_start(phi: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least squares on ``logit(target)`` -> ``(w0, alpha, w_dir)`` (E2 precedent)."""
    tc = np.clip(t, 1e-3, 1 - 1e-3)
    z = np.log(tc / (1 - tc))
    A = np.concatenate([np.ones((phi.shape[0], 1)), phi], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    w0 = float(coef[0])
    v = coef[1:]
    a = float(np.linalg.norm(v))
    u = v / a if a > 1e-8 else np.ones_like(v) / np.sqrt(len(v))
    return w0, max(a, 1e-3), u


def _radial_start(phi: np.ndarray, t: np.ndarray) -> tuple[float, float, np.ndarray] | None:
    """Centroid-radial informed start (E2 precedent).  Columns 0..4 of
    ``phi_dir`` are ``[x, y, P2(x), P2(y), xy]`` so ``-( (x-a)^2 + (y-b)^2 )``
    is expressible: ``x^2 = (2 P2(x) + 1) / 3``."""
    if phi.shape[1] < 5:
        return None
    tw = t.sum() + _EPS
    a = float((t * phi[:, 0]).sum() / tw)
    b = float((t * phi[:, 1]).sum() / tw)
    v = np.zeros(phi.shape[1])
    v[0], v[1], v[2], v[3] = 2 * a, 2 * b, -2.0 / 3.0, -2.0 / 3.0
    n = np.linalg.norm(v)
    if n < 1e-12:
        return None
    u = v / n
    z = phi @ u
    alpha = 2.0 / (z.std() + _EPS)
    w0 = -alpha * float(z.mean())
    return w0, alpha, u


def _s_stats(phi: np.ndarray, t: np.ndarray, w0: float, alpha: float,
             u: np.ndarray) -> tuple[float, float]:
    s = S_SCALE * np.tanh((w0 + alpha * (phi @ u)) / S_SCALE)
    mu = float((s * t).sum() / (t.sum() + _EPS))
    sd = float(np.sqrt((t * (s - mu) ** 2).sum() / (t.sum() + _EPS)) + 0.1)
    return mu, sd


def _band_raw(mu: float, hw: float, polarity: float, dtype, device) -> dict[str, torch.Tensor]:
    T = lambda v: torch.tensor(float(v), dtype=dtype, device=device)  # noqa: E731
    return {
        "mu": T(mu),
        "h_raw": T(inv_bounded_sigmoid(min(max(hw, 0.05), 2.4), 0.02, 2.50)),
        "k_raw": T(inv_bounded_sigmoid(8.0, 1.0, 40.0)),
        "pi_raw": T(polarity),
    }


def _cband_raw(mu: float, hw: float, dtype, device) -> dict[str, torch.Tensor]:
    grid = cband_centres(device=device, dtype=dtype).cpu().numpy()
    step = float(grid[1] - grid[0])
    sig0 = float(np.clip(0.45 * step, CBAND_SIG_LO * 1.05, CBAND_SIG_HI * 0.95))
    on = np.abs(grid - grid[int(np.argmin(np.abs(grid - mu)))]) <= max(hw, 1e-6)
    if not on.any():
        on[int(np.argmin(np.abs(grid - mu)))] = True
    c0 = np.where(on, 0.98, 0.02)
    T = lambda v: torch.tensor(v, dtype=dtype, device=device)  # noqa: E731
    return {
        "sig_raw": T([inv_bounded_sigmoid(sig0, CBAND_SIG_LO, CBAND_SIG_HI)] * CBAND_M),
        "o_raw": T([inv_bounded_sigmoid(0.5, 0.0, 1.0)] * CBAND_M),
        "c_raw": T([inv_bounded_sigmoid(float(v), 0.0, 1.0) for v in c0]),
    }


def build_starts(
    phi: torch.Tensor, target: torch.Tensor, readout: str, cfg: FitConfig
) -> tuple[list[dict[str, torch.Tensor]], int]:
    """Informed + random starts.  Deterministic in ``cfg.seed``.

    Returns ``(starts, n_informed_dropped)``."""
    dtype, device = phi.dtype, phi.device
    phi_np = phi.detach().double().cpu().numpy()
    t_np = target.detach().double().cpu().numpy()
    rng = np.random.default_rng(cfg.seed)
    D = phi.shape[1]

    seeds: list[tuple[float, float, np.ndarray]] = []
    # The informed starts are a convenience, not a dependency: a degenerate phi
    # (rank-deficient, or carrying a NaN from a broken upstream stage) makes
    # LAPACK's lstsq raise, and losing the whole multi-start fit to that would
    # turn a recoverable sample into a hard crash.  Random starts always remain.
    # N-21: `_radial_start` does *not* raise on a NaN phi -- it returns NaNs --
    # so the finiteness check below is what actually catches that case, and the
    # dropped-start count is surfaced as a flag rather than only as `n_starts`
    # quietly falling from 18 to 16.
    n_informed_dropped = 0
    for maker in (lambda: _lsq_start(phi_np, t_np), lambda: _radial_start(phi_np, t_np)):
        try:
            seed = maker()
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            seed = None
        if seed is not None and not (
            np.isfinite(seed[0]) and np.isfinite(seed[1]) and np.all(np.isfinite(seed[2]))
        ):
            seed = None
        if seed is None:
            n_informed_dropped += 1
        else:
            seeds.append(seed)
    for _ in range(cfg.n_random):
        seeds.append((0.0, 1.0, rng.normal(size=D)))
    seeds.append((0.0, 0.05, rng.normal(size=D)))          # near-degenerate start

    starts: list[dict[str, torch.Tensor]] = []
    T = lambda v: torch.tensor(float(v), dtype=dtype, device=device)  # noqa: E731
    for w0, alpha, u in seeds:
        u = np.asarray(u, dtype=np.float64)
        u = u / (np.linalg.norm(u) + 1e-12)
        try:
            mu, sd = _s_stats(phi_np, t_np, w0, alpha, u)
        except (ValueError, FloatingPointError):
            mu, sd = 0.0, 0.5
        if not (np.isfinite(mu) and np.isfinite(sd)):
            mu, sd = 0.0, 0.5
        base = {
            "w0": T(w0),
            "alpha_raw": T(_softplus_inv(alpha)),
            "w_raw": torch.tensor(u, dtype=dtype, device=device),
        }
        if readout == "band":
            # both polarities: pi ~ 0.88 (band) and pi ~ 0.12 (notch)
            for pol in (2.0, -2.0):
                st = {k: v.clone() for k, v in base.items()}
                st.update(_band_raw(mu, min(max(sd, 0.1), 2.0), pol, dtype, device))
                starts.append(st)
        elif readout == "cband12":
            st = {k: v.clone() for k, v in base.items()}
            st.update(_cband_raw(mu, min(max(sd, 0.1), 2.0), dtype, device))
            starts.append(st)
        else:
            raise ValueError(f"unknown readout {readout!r}")
    return starts, n_informed_dropped


# --- fit --------------------------------------------------------------------

@dataclass
class FitResult:
    """One image's fit.

    ``latent`` is ``None`` when *every* start failed.  It is deliberately not a
    zero vector: protocol 10.2 forbids silently substituting one, and a ``None``
    makes any consumer that forgets to check ``status`` fail loudly at the first
    use instead of quietly training ``B`` on a constant mask
    (REVIEW-impl-WhereA B-1).  Use :meth:`usable`.
    """

    latent: Latent | None
    readout: str
    loss: float
    status: str                    # "ok" | "rejected"
    reject_reason: str | None
    flags: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    start_losses: list[float] = field(default_factory=list)
    best_start: int = -1
    n_starts: int = 0
    n_failed_starts: int = 0
    objective: str = ""
    seed: int = 0

    @property
    def usable(self) -> bool:
        """True only for a fit that may drive a gradient or a ceiling number."""
        return self.status == "ok" and self.latent is not None

    def rejection_row(self, **extra: Any) -> dict[str, Any]:
        """The compact row that goes into ``fit_rejections.jsonl`` (protocol 10.2
        "failed samples go into an explicit rejection / fit report")."""
        row = {
            "readout": self.readout,
            "status": self.status,
            "reject_reason": self.reject_reason,
            "flags": list(self.flags),
            "loss": self.loss,
            "objective": self.objective,
            "best_start": self.best_start,
            "n_starts": self.n_starts,
            "n_failed_starts": self.n_failed_starts,
            "seed": self.seed,
            "soft_iou_minmax": self.metrics.get("soft_iou_minmax"),
        }
        row.update(extra)
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "readout": self.readout,
            "status": self.status,
            "reject_reason": self.reject_reason,
            "flags": list(self.flags),
            "loss": self.loss,
            "objective": self.objective,
            "metrics": dict(self.metrics),
            "start_losses": list(self.start_losses),
            "best_start": self.best_start,
            "n_starts": self.n_starts,
            "n_failed_starts": self.n_failed_starts,
            "seed": self.seed,
            "usable": self.usable,
            "latent": self.latent.to_dict() if self.latent is not None else None,
        }


def _forward(raw: dict[str, torch.Tensor], phi: torch.Tensor, readout: str):
    w_dir = w_dir_of(raw["w_raw"])
    alpha = alpha_of(raw["alpha_raw"])
    q = raw["w0"] + alpha * (phi @ w_dir)
    s = S_SCALE * torch.tanh(q / S_SCALE)
    rho = {k: raw[k] for k in param_shapes(readout)}
    return apply_readout(readout, s, rho), s, alpha


def fit_latent(
    phi: torch.Tensor,
    target: torch.Tensor,
    readout: str,
    cfg: FitConfig | None = None,
) -> FitResult:
    """Fit one image's ``(w*, rho*)`` on the low-res grid.

    ``phi``: ``(P, 71)``.  ``target``: ``(P,)`` soft GT mask in [0, 1].
    """
    cfg = cfg or FitConfig()
    if phi.dim() != 2 or phi.shape[1] != PHI_DIR_DIM:
        raise ValueError(f"phi must be (P,{PHI_DIR_DIM}), got {tuple(phi.shape)}")
    if target.shape != (phi.shape[0],):
        raise ValueError(f"target must be ({phi.shape[0]},), got {tuple(target.shape)}")
    dtype = getattr(torch, cfg.dtype)
    phi = phi.detach().to(dtype)
    target = target.detach().to(dtype).clamp(0.0, 1.0)

    starts, n_informed_dropped = build_starts(phi, target, readout, cfg)
    best: dict[str, Any] | None = None
    start_losses: list[float] = []
    n_failed = 0

    for i, st in enumerate(starts):
        raw = {k: v.detach().clone().requires_grad_(True) for k, v in st.items()}
        opt = torch.optim.LBFGS(
            list(raw.values()),
            max_iter=cfg.max_iter,
            history_size=cfg.history_size,
            line_search_fn=cfg.line_search_fn,
            tolerance_grad=cfg.tol_grad,
            tolerance_change=cfg.tol_change,
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            m, _, _ = _forward(raw, phi, readout)
            loss = objective_value(cfg.objective, m, target)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:                      # a diverged line search, not a sample failure
            n_failed += 1
            start_losses.append(float("inf"))
            continue
        with torch.no_grad():
            m, _, alpha = _forward(raw, phi, readout)
            loss = float(objective_value(cfg.objective, m, target))
            alpha_v = float(alpha)
        start_losses.append(loss)
        if not np.isfinite(loss):
            n_failed += 1
            continue
        cand = {"loss": loss, "alpha": alpha_v, "i": i,
                "raw": {k: v.detach().clone() for k, v in raw.items()}}
        if (best is None or cand["loss"] < best["loss"] - 1e-4
                or (abs(cand["loss"] - best["loss"]) <= 1e-4 and cand["alpha"] < best["alpha"])):
            best = cand

    base_flags = ["informed_start_unavailable"] if n_informed_dropped else []
    if best is None:
        # No zero-vector substitute: protocol 10.2 forbids it, and a fabricated
        # latent would go on to produce a constant mask and a real gradient.
        return FitResult(
            latent=None, readout=readout, loss=float("inf"), status="rejected",
            reject_reason="all_starts_failed", flags=base_flags,
            start_losses=start_losses,
            n_starts=len(starts), n_failed_starts=n_failed, objective=cfg.objective,
            seed=cfg.seed,
        )

    raw = best["raw"]
    latent = canonicalize(Latent(
        readout, raw["w0"], raw["alpha_raw"], raw["w_raw"],
        {k: raw[k] for k in param_shapes(readout)},
    ))
    with torch.no_grad():
        m, _ = mask_from_latent(phi, latent)
    metrics = mask_metrics(m, target)
    metrics["loss_recheck"] = float(objective_value(cfg.objective, m, target))

    flags: list[str] = list(base_flags)
    if best["alpha"] < cfg.reject_alpha:
        flags.append("alpha_collapsed")
    if metrics["pred_std"] < 1e-6:
        flags.append("constant_mask")
    status, reason = "ok", None
    if not np.isfinite(best["loss"]):
        status, reason = "rejected", "non_finite_loss"
    elif best["loss"] > cfg.reject_loss:
        status, reason = "rejected", "loss_above_threshold"
    elif abs(metrics["loss_recheck"] - best["loss"]) > 1e-6:
        status, reason = "rejected", "canonicalisation_changed_loss"

    return FitResult(
        latent=latent, readout=readout, loss=best["loss"], status=status,
        reject_reason=reason, flags=flags, metrics=metrics,
        start_losses=start_losses, best_start=best["i"], n_starts=len(starts),
        n_failed_starts=n_failed, objective=cfg.objective, seed=cfg.seed,
    )


def evaluate_latent(
    phi: torch.Tensor,
    latent: Latent,
    target_low: torch.Tensor,
    grid_h: int,
    grid_w: int,
    guide_hi: torch.Tensor | None = None,
    target_hi: torch.Tensor | None = None,
    up_cfg: UpsampleConfig | None = None,
) -> dict[str, Any]:
    """Low-res metrics, plus full-res metrics through the one guided upsample.

    The high-res branch is the *delivered* mask: the ceiling that goes into a
    report has to be the one measured at the resolution the mask is consumed at
    (REVIEW-impl-WhereA B-4).  It also carries the s-domain report, so
    "the field left (-3,3) after upsampling" can never be a silent event.
    """
    with torch.no_grad():
        m_low, s = mask_from_latent(phi, latent)
        out: dict[str, Any] = {"low": mask_metrics(m_low, target_low),
                               "s_low_range": [float(s.min()), float(s.max())],
                               "readout_bounds": bounds_report(latent.readout, latent.rho)}
        if guide_hi is not None and target_hi is not None:
            s_hi, _, domain = combine_then_upsample(
                phi, latent, grid_h, grid_w, guide_hi, up_cfg, return_domain_report=True
            )
            m_hi = apply_readout(latent.readout, s_hi.squeeze(0).squeeze(0).reshape(-1), latent.rho)
            out["hi"] = mask_metrics(m_hi, target_hi.reshape(-1))
            out["s_domain"] = domain
            out["hi_minus_low_soft_iou"] = (
                out["hi"]["soft_iou_minmax"] - out["low"]["soft_iou_minmax"]
            )
            out["s_hi_range"] = [float(s_hi.min()), float(s_hi.max())]
    return out
