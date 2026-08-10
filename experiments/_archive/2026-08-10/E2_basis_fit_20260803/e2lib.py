"""E2 shared library: 14-dim mask basis (PLAN v2 section 1.3), readouts,
soft-IoU, per-mask L-BFGS fitting, CLIP dense semantic channels.

Basis (single-axis v1, PLAN section 1.3):
    q(p) = w0 + alpha * (w_dir . phi_dir(p)),  ||w_dir||=1, alpha>=0
    s    = 3 * tanh(q / 3)
    phi  = [1] + [x, y, P2(x), P2(y), x*y]  (Legendre, centered / short side)
         + [L, S] (range)  + [e1..e6] (semantic, per-image standardized +
           residual-orthogonalized against the geometric+range block)

Readouts (unconstrained = free readout shape; constrained = the renderer's
own s-axis parameterization, PLAN L56 + L124-R2 + L48):
    monotone     : m = sigmoid(g * s + b)
    bandpass     : m = sigmoid(k(s-mu+h)) - sigmoid(k(s-mu-h))   (flat-top band)
    gauss        : m = exp(-0.5 * ((s - mu) / sigma)^2), amplitude fixed at 1,
                   sigma bounded via sigmoid (NO bare exp parameterization)
    cband_norm   : m = sum_i c_i o_i N(s;mu_i,sig_i) / sum_j o_j N(s;mu_j,sig_j)
                   with M=12 primitives, mu_i on a FIXED uniform grid over
                   [-3,3], sig_i in [0.025,0.30] via bounded sigmoid, o_i and
                   c_i in (0,1) via sigmoid.  This is exactly "the mask a
                   subset of the renderer's primitives can paint" (PLAN L48
                   weight formula + L75 "the M=12 Gaussians on the s axis ARE
                   12 band-pass masks").
    cband_unnorm : same mixture without the normalizing denominator, clamped
                   to <=1 (the R-6 / RD-D form).
    cgauss       : single primitive of that bank: mu locked to a grid point,
                   sigma in [0.025,0.30], amplitude fixed at 1.

Loss = 1 - softIoU_minmax, softIoU_minmax = sum(min(m,t)) / sum(max(m,t))
(exact recovery of a *soft* target scores 1.0, which is what the >=0.97
pre-registered thresholds semantically require; the product-form soft IoU is
also reported as a secondary metric).
"""

from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------
# Geometry features (Legendre, centered, normalized by half short side)
# --------------------------------------------------------------------------

GEO_NAMES = ["x", "y", "P2x", "P2y", "xy"]
CUBIC_NAMES = ["P3x", "P3y", "P2x_y", "x_P2y"]
RANGE_NAMES = ["L", "S"]
SEM_NAMES = [f"e{i}" for i in range(1, 7)]
DIR_NAMES_14 = GEO_NAMES + RANGE_NAMES + SEM_NAMES          # 13 dirs + [1]

ANCHORS = {
    "sky": ["the sky", "clouds in the sky", "a photo of the sky"],
    "skin": ["human skin", "a person", "a person's face",
             "a portrait photo of a person"],
    "foliage": ["green foliage", "trees and plants", "leaves of a plant"],
    "water": ["water", "a lake or a river", "waves of the sea"],
    "architecture": ["a building", "architecture", "an urban street scene"],
    "subject": ["the main subject of the photo",
                "the foreground subject of the photo",
                "the most salient object in the photo"],
}
ANCHOR_ORDER = ["sky", "skin", "foliage", "water", "architecture", "subject"]
CONTENT_CLASSES = ["sky", "skin", "foliage", "water", "architecture"]

# --------------------------------------------------------------------------
# Renderer s-axis spec (constrained readouts).  Sources, verbatim:
#   PLAN L56  : "s axis M=12 Gaussians, mu uniform [-3,3] FIXED, sigma_s
#                bounded sigmoid; NO smoothness regularizer on the s axis"
#   PLAN L124 : R-2 row, "sigma_s in [0.025, 0.30] sigmoid"
# (the M=12/K=6 and the sigma-range reading are in tension across the two
#  lines; see NOTES "pending decision 5" -- the default below is the reading
#  that is HARDEST for the constrained arm, and A-5 sweeps the range.)
CONSTRAINED_AXIS = {"M": 12, "mu_lo": -3.0, "mu_hi": 3.0,
                    "sig_lo": 0.025, "sig_hi": 0.30}
CONSTRAINED_READOUTS = ("cband_norm", "cband_unnorm", "cgauss")


def axis_grid(axis: dict | None = None) -> np.ndarray:
    ax = axis or CONSTRAINED_AXIS
    return np.linspace(ax["mu_lo"], ax["mu_hi"], int(ax["M"]))


def norm_coords(h: int, w: int, stride: int = 1):
    """Centered pixel coords normalized by half the SHORT side -> short side
    spans [-1, 1]."""
    half = min(h, w) / 2.0
    ys = (np.arange(0, h, stride) + 0.5 - h / 2.0) / half
    xs = (np.arange(0, w, stride) + 0.5 - w / 2.0) / half
    X, Y = np.meshgrid(xs, ys)
    return X.astype(np.float64), Y.astype(np.float64)


def P2(t):
    return 0.5 * (3.0 * t * t - 1.0)


def P3(t):
    return 0.5 * (5.0 * t ** 3 - 3.0 * t)


def geo_features(h: int, w: int, stride: int = 1, cubic: bool = False):
    """(P, 5) or (P, 9) float64 matrix of direction features (no constant)."""
    X, Y = norm_coords(h, w, stride)
    cols = [X, Y, P2(X), P2(Y), X * Y]
    if cubic:
        cols += [P3(X), P3(Y), P2(X) * Y, X * P2(Y)]
    return np.stack([c.reshape(-1) for c in cols], axis=1)


def monomial_features(h: int, w: int, stride: int = 1):
    X, Y = norm_coords(h, w, stride)
    cols = [np.ones_like(X), X, Y, X * X, Y * Y, X * Y]
    return np.stack([c.reshape(-1) for c in cols], axis=1)


def gram_cond(F: np.ndarray) -> float:
    G = (F.T @ F) / F.shape[0]
    return float(np.linalg.cond(G))


# --------------------------------------------------------------------------
# Range channels + per-image standardization / residualization
# --------------------------------------------------------------------------

def range_channels(img: np.ndarray):
    """img: (H,W,3) sRGB [0,1] -> L (Rec.709 luma), S (HSV saturation)."""
    L = (0.2126 * img[..., 0] + 0.7152 * img[..., 1]
         + 0.0722 * img[..., 2])
    mx = img.max(axis=-1)
    mn = img.min(axis=-1)
    S = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return L.astype(np.float64), S.astype(np.float64)


def standardize(v: np.ndarray, eps: float = 1e-6):
    return (v - v.mean()) / (v.std() + eps)


def residualize(E: np.ndarray, A: np.ndarray):
    """Orthogonalize semantic block E (P,6) against block A (P,k) via lstsq.
    Returns (E_resid, corr_before, corr_after) where corr_* is the (k,6)
    normalized cross-correlation of standardized columns."""

    def _corr(A_, E_):
        As = (A_ - A_.mean(0)) / (A_.std(0) + 1e-9)
        Es = (E_ - E_.mean(0)) / (E_.std(0) + 1e-9)
        return (As.T @ Es) / A_.shape[0]

    before = _corr(A, E)
    coef, *_ = np.linalg.lstsq(A, E, rcond=None)
    Er = E - A @ coef
    after = _corr(A, Er)
    return Er, before, after


# --------------------------------------------------------------------------
# Guided filter (edge-aware smoothing of basis channels; PLAN 1.3 line 73)
# --------------------------------------------------------------------------

def _box(a: np.ndarray, r: int):
    """Box mean with window 2r+1 (reflect padding), separable cumsum."""
    from scipy.ndimage import uniform_filter
    return uniform_filter(a, size=2 * r + 1, mode="reflect")


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int = 32,
                  eps: float = 1e-3):
    mI = _box(guide, r)
    mP = _box(src, r)
    corr = _box(guide * src, r)
    var = _box(guide * guide, r) - mI * mI
    a = (corr - mI * mP) / (var + eps)
    b = mP - a * mI
    return _box(a, r) * guide + _box(b, r)


# --------------------------------------------------------------------------
# soft-IoU
# --------------------------------------------------------------------------

def soft_iou_minmax_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    return float(np.minimum(a, b).sum() / (np.maximum(a, b).sum() + eps))


def soft_iou_prod_np(a: np.ndarray, b: np.ndarray, eps: float = 1e-6):
    inter = (a * b).sum()
    return float(inter / (a.sum() + b.sum() - inter + eps))


# --------------------------------------------------------------------------
# Per-mask L-BFGS fit
# --------------------------------------------------------------------------

def _sig_bounded(raw, lo, hi):
    """Bounded sigmoid sigma parameterization (redline: no bare exp)."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def _constrained_mask(params, s, readout, axis):
    """Renderer-faithful s-axis response.  mu is a FIXED grid (a buffer, not a
    parameter); sigma is bounded-sigmoid; o/c are sigmoids in (0,1).
    No smoothness regularizer anywhere (redline)."""
    ax = axis or CONSTRAINED_AXIS
    sig = _sig_bounded(params["sig_raw"], ax["sig_lo"], ax["sig_hi"])
    mu = params["mu_grid"]
    if readout == "cgauss":                       # single primitive, amp == 1
        return torch.exp(-0.5 * ((s - mu) / sig) ** 2)
    z = (s.unsqueeze(-1) - mu) / sig              # (P, M)
    N = torch.exp(-0.5 * z * z)
    o = torch.sigmoid(params["o_raw"])
    c = torch.sigmoid(params["c_raw"])
    if readout == "cband_norm":                   # PLAN L48 weight formula
        return (N * (o * c)).sum(-1) / ((N * o).sum(-1) + 1e-9)
    return torch.clamp((N * (o * c)).sum(-1), max=1.0)   # cband_unnorm (R-6)


def _forward(params: dict, Phi: torch.Tensor, readout: str,
             axis: dict | None = None):
    u = params["u"]
    w_dir = u / (u.norm() + 1e-12)
    alpha = torch.nn.functional.softplus(params["a_raw"])
    q = params["w0"] + alpha * (Phi @ w_dir)
    s = 3.0 * torch.tanh(q / 3.0)
    if readout in CONSTRAINED_READOUTS:
        return _constrained_mask(params, s, readout, axis), s, alpha
    if readout == "monotone":
        m = torch.sigmoid(params["g"] * s + params["b"])
    elif readout == "bandpass":
        # logistic band sigma(k(s-mu+h)) - sigma(k(s-mu-h)): a passband with
        # a flat top, the single-band analogue of what the renderer's
        # M=12-Gaussian s-axis mixture can realize (a lone Gaussian cannot
        # form the plateau of a feathered ring; measured med IoU 0.78).
        h = 0.02 + 2.48 * torch.sigmoid(params["h_raw"])
        k = 1.0 + 39.0 * torch.sigmoid(params["k_raw"])
        m = (torch.sigmoid(k * (s - params["mu"] + h))
             - torch.sigmoid(k * (s - params["mu"] - h)))
    elif readout == "gauss":
        sigma = _sig_bounded(params["sig_raw"], 0.05, 3.0)
        m = torch.exp(-0.5 * ((s - params["mu"]) / sigma) ** 2)
    else:
        raise ValueError(readout)
    return m, s, alpha


def _loss(params, Phi, t, readout, axis=None):
    m, _, _ = _forward(params, Phi, readout, axis)
    iou = torch.minimum(m, t).sum() / (torch.maximum(m, t).sum() + 1e-6)
    return 1.0 - iou


def _lsq_init(Phi_np: np.ndarray, t_np: np.ndarray):
    """Least-squares on logit(target) -> (w0, alpha, u)."""
    z = np.log(np.clip(t_np, 1e-3, 1 - 1e-3) /
               (1 - np.clip(t_np, 1e-3, 1 - 1e-3)))
    A = np.concatenate([np.ones((Phi_np.shape[0], 1)), Phi_np], axis=1)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    w0 = float(coef[0])
    v = coef[1:]
    a = float(np.linalg.norm(v))
    u = v / a if a > 1e-8 else np.ones_like(v) / np.sqrt(len(v))
    return w0, max(a, 1e-3), u


def softplus_inv(y: float) -> float:
    y = max(y, 1e-6)
    return float(y + np.log(-np.expm1(-y))) if y < 20 else y


def _logit_bounded(y: float, lo: float, hi: float) -> float:
    """Inverse of lo + (hi-lo)*sigmoid(raw), safely clipped."""
    z = float(np.clip((y - lo) / (hi - lo), 1e-4, 1 - 1e-4))
    return float(np.log(z / (1 - z)))


def _axis_start(st: dict, readout: str, axis: dict, rng) -> dict:
    """Complete a (mu, sig)-style start into constrained-axis raw fields.

    `st` may already carry explicit axis fields (warm starts); otherwise the
    band centre is snapped to the nearest grid point (w0 is free, so the
    residual shift is absorbable) and sigma is initialised at ~0.45*grid step.
    """
    ax = axis or CONSTRAINED_AXIS
    grid = axis_grid(ax)
    M = len(grid)
    dmu = float(grid[1] - grid[0])
    out = dict(st)
    if "k_idx" not in out:
        out["k_idx"] = int(np.argmin(np.abs(grid - float(st.get("mu", 0.0)))))
    if "sig0" not in out:
        out["sig0"] = float(np.clip(0.45 * dmu, ax["sig_lo"] * 1.05,
                                    ax["sig_hi"] * 0.95))
    if readout == "cgauss":
        return out
    if "c0" not in out:
        hw = float(st.get("hw", max(float(st.get("sig", 0.3)), 0.5 * dmu)))
        on = np.abs(grid - grid[out["k_idx"]]) <= max(hw, 1e-6)
        if not on.any():
            on[out["k_idx"]] = True
        out["c0"] = np.where(on, 0.98, 0.02)
    if "o0" not in out:
        out["o0"] = np.full(M, 0.5)
    return out


def fit_mask(Phi_fit: np.ndarray, t_fit: np.ndarray,
             Phi_eval: np.ndarray, t_eval: np.ndarray,
             readout: str, seed: int = 0, n_random: int = 6,
             max_iter: int = 120,
             extra_starts: list[dict] | None = None,
             axis: dict | None = None) -> dict:
    """Fit one mask; returns best params + eval IoUs.

    Restarts: LSQ-informed + n_random random + caller-provided informed
    starts (e.g. centroid-radial). Tie-break (within 1e-4 loss): smallest
    alpha (degeneracy check for constant masks; PLAN: alpha init ~0,
    alpha=0 must stay reachable).
    """
    rng = np.random.default_rng(seed)
    D = Phi_fit.shape[1]
    Phi = torch.from_numpy(Phi_fit.astype(np.float64))
    t = torch.from_numpy(t_fit.astype(np.float64))

    w0_i, a_i, u_i = _lsq_init(Phi_fit, t_fit)
    s_lsq = 3.0 * np.tanh((w0_i + a_i * (Phi_fit @ u_i)) / 3.0)
    mu_w = float((s_lsq * t_fit).sum() / (t_fit.sum() + 1e-6))
    sd_w = float(np.sqrt((t_fit * (s_lsq - mu_w) ** 2).sum()
                         / (t_fit.sum() + 1e-6)) + 0.1)

    starts = []
    if readout == "monotone":
        starts.append({"w0": w0_i, "a": a_i, "u": u_i, "g": 1.0, "b": 0.0})
        for k in range(n_random):
            starts.append({"w0": 0.0, "a": 1.0,
                           "u": rng.normal(size=D), "g": float((-1) ** k * 3.0),
                           "b": 0.0})
        starts.append({"w0": 0.0, "a": 0.05,
                       "u": rng.normal(size=D), "g": 2.0, "b": 0.0})
    else:
        starts.append({"w0": w0_i, "a": a_i, "u": u_i,
                       "mu": mu_w, "sig": min(max(sd_w, 0.1), 2.0)})
        for k in range(n_random):
            starts.append({"w0": 0.0, "a": 1.0, "u": rng.normal(size=D),
                           "mu": float(rng.uniform(-2, 2)),
                           "sig": float(rng.uniform(0.2, 1.5))})
        starts.append({"w0": 0.0, "a": 0.05, "u": rng.normal(size=D),
                       "mu": 0.0, "sig": 0.5})
    for st in (extra_starts or []):
        starts.append(st)
    if readout in CONSTRAINED_READOUTS:
        ax = axis or CONSTRAINED_AXIS
        grid = axis_grid(ax)
        starts = [_axis_start(st, readout, ax, rng) for st in starts]
        if readout == "cgauss":       # second grid anchor (tanh-saturation)
            alt = dict(starts[0])
            alt["k_idx"] = int(len(grid) // 2)
            starts.append(_axis_start(alt, readout, ax, rng))

    best = None
    for st in starts:
        params = {
            "w0": torch.tensor(st["w0"], dtype=torch.float64,
                               requires_grad=True),
            "a_raw": torch.tensor(softplus_inv(st["a"]), dtype=torch.float64,
                                  requires_grad=True),
            "u": torch.tensor(np.asarray(st["u"], dtype=np.float64),
                              requires_grad=True),
        }
        if readout in CONSTRAINED_READOUTS:
            ax = axis or CONSTRAINED_AXIS
            grid = axis_grid(ax)
            if readout == "cgauss":
                params["mu_grid"] = torch.tensor(float(grid[st["k_idx"]]),
                                                 dtype=torch.float64)
                params["sig_raw"] = torch.tensor(
                    _logit_bounded(st["sig0"], ax["sig_lo"], ax["sig_hi"]),
                    dtype=torch.float64, requires_grad=True)
            else:
                params["mu_grid"] = torch.tensor(grid, dtype=torch.float64)
                sig0 = np.full(len(grid), float(st["sig0"])) \
                    if np.ndim(st["sig0"]) == 0 else np.asarray(st["sig0"])
                params["sig_raw"] = torch.tensor(
                    [_logit_bounded(float(v), ax["sig_lo"], ax["sig_hi"])
                     for v in sig0],
                    dtype=torch.float64, requires_grad=True)
                params["o_raw"] = torch.tensor(
                    [_logit_bounded(float(v), 0.0, 1.0) for v in st["o0"]],
                    dtype=torch.float64, requires_grad=True)
                params["c_raw"] = torch.tensor(
                    [_logit_bounded(float(v), 0.0, 1.0) for v in st["c0"]],
                    dtype=torch.float64, requires_grad=True)
        elif readout == "monotone":
            params["g"] = torch.tensor(st["g"], dtype=torch.float64,
                                       requires_grad=True)
            params["b"] = torch.tensor(st["b"], dtype=torch.float64,
                                       requires_grad=True)
        else:
            params["mu"] = torch.tensor(st["mu"], dtype=torch.float64,
                                        requires_grad=True)
            if readout == "bandpass":
                hr = float(np.log(max((st["sig"] - 0.02) / 2.48, 1e-4)
                                  / max(1 - (st["sig"] - 0.02) / 2.48,
                                        1e-4)))
                params["h_raw"] = torch.tensor(hr, dtype=torch.float64,
                                               requires_grad=True)
                kr = float(np.log((8.0 - 1.0) / 39.0
                                  / (1 - (8.0 - 1.0) / 39.0)))
                params["k_raw"] = torch.tensor(kr, dtype=torch.float64,
                                               requires_grad=True)
            else:
                sr = float(np.log(max((st["sig"] - 0.05) / 2.95, 1e-4)
                                  / max(1 - (st["sig"] - 0.05) / 2.95,
                                        1e-4)))
                params["sig_raw"] = torch.tensor(sr, dtype=torch.float64,
                                                 requires_grad=True)
        # mu_grid is a fixed buffer (mu is NOT learnable, PLAN L56)
        opt = torch.optim.LBFGS([v for v in params.values()
                                 if v.requires_grad], max_iter=max_iter,
                                history_size=20, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-11)

        def closure():
            opt.zero_grad()
            loss = _loss(params, Phi, t, readout, axis)
            loss.backward()
            return loss

        try:
            opt.step(closure)
        except Exception:
            continue
        with torch.no_grad():
            loss = float(_loss(params, Phi, t, readout, axis))
            alpha = float(torch.nn.functional.softplus(params["a_raw"]))
        cand = {"loss": loss, "alpha": alpha,
                "state": {k: v.detach().clone() for k, v in params.items()}}
        if (best is None or cand["loss"] < best["loss"] - 1e-4
                or (abs(cand["loss"] - best["loss"]) <= 1e-4
                    and cand["alpha"] < best["alpha"])):
            best = cand

    assert best is not None, "all restarts failed"
    # evaluate at full resolution
    with torch.no_grad():
        Phi_e = torch.from_numpy(Phi_eval.astype(np.float64))
        m_e, s_e, alpha = _forward(best["state"], Phi_e, readout, axis)
    m_np = m_e.numpy()
    st = best["state"]
    # complement IoU keeps the metric meaningful for near-empty targets
    # (an all-zero target scores 0 under min/max IoU even at exact recovery)
    out = {
        "loss_fit": best["loss"],
        "iou_minmax": soft_iou_minmax_np(m_np, t_eval),
        "iou_minmax_comp": soft_iou_minmax_np(1 - m_np, 1 - t_eval),
        "mae": float(np.abs(m_np - t_eval).mean()),
        "iou_prod": soft_iou_prod_np(m_np, t_eval),
        "pred_std": float(m_np.std()),
        "pred_mean": float(m_np.mean()),
        "alpha": float(alpha),
        "w0": float(st["w0"]),
        "w_dir": (st["u"] / st["u"].norm()).numpy().tolist(),
    }
    if readout == "monotone":
        out["g"] = float(st["g"])
        out["b"] = float(st["b"])
    elif readout == "bandpass":
        out["mu"] = float(st["mu"])
        out["h"] = float(0.02 + 2.48 * torch.sigmoid(st["h_raw"]))
        out["k"] = float(1.0 + 39.0 * torch.sigmoid(st["k_raw"]))
    elif readout in CONSTRAINED_READOUTS:
        ax = axis or CONSTRAINED_AXIS
        out["axis"] = dict(ax)
        out["mu_grid"] = np.atleast_1d(st["mu_grid"].numpy()).tolist()
        out["sigma"] = np.atleast_1d(
            _sig_bounded(st["sig_raw"], ax["sig_lo"],
                         ax["sig_hi"]).numpy()).tolist()
        if readout != "cgauss":
            out["o"] = torch.sigmoid(st["o_raw"]).numpy().tolist()
            c = torch.sigmoid(st["c_raw"])
            out["c"] = c.numpy().tolist()
            # A-3: hard payload grouping (c rounded to {0,1}); in the renderer
            # a payload group IS a 0/1 subset of primitives.
            hard = dict(st)
            hard["c_raw"] = torch.where(c >= 0.5,
                                        torch.full_like(c, 20.0),
                                        torch.full_like(c, -20.0))
            with torch.no_grad():
                m_h, _, _ = _forward(hard, Phi_e, readout, ax)
            out["iou_hardc"] = soft_iou_minmax_np(m_h.numpy(), t_eval)
            out["n_on"] = int((c >= 0.5).sum())
    else:
        out["mu"] = float(st["mu"])
        out["sigma"] = float(_sig_bounded(st["sig_raw"], 0.05, 3.0))
    return out


def radial_starts(Phi_fit: np.ndarray, t_fit: np.ndarray,
                  readout: str) -> list[dict]:
    """Centroid-radial informed starts. Assumes dir-columns start with
    [x, y, P2(x), P2(y), xy, ...]. Builds q ~ -((x-a)^2 + (y-b)^2) around the
    target centroid (x^2 = (2*P2(x)+1)/3), scaled so s spans ~[-2, 2]; for
    every band-type readout, mu/sigma come from the target-weighted s stats.

    NOTE (bug fix, REVIEW-result section 2.3): this used to branch on
    `readout == "bandpass"`, so every *other* non-monotone readout ("gauss",
    and now the constrained ones) got monotone-style g/b starts while
    `fit_mask` asked for st["mu"] -> KeyError('mu') on 600/600 gauss fits.
    The branch now mirrors `fit_mask`: monotone -> g/b, everything else ->
    mu/sig."""
    D = Phi_fit.shape[1]
    if D < 5:
        return []
    tw = t_fit.sum() + 1e-6
    a = float((t_fit * Phi_fit[:, 0]).sum() / tw)
    b = float((t_fit * Phi_fit[:, 1]).sum() / tw)
    v = np.zeros(D)
    v[0], v[1], v[2], v[3] = 2 * a, 2 * b, -2.0 / 3.0, -2.0 / 3.0
    u = v / np.linalg.norm(v)
    z = Phi_fit @ u
    alpha0 = 2.0 / (z.std() + 1e-6)
    w0 = -alpha0 * float(z.mean())
    s = 3.0 * np.tanh((w0 + alpha0 * z) / 3.0)
    mu_w = float((t_fit * s).sum() / tw)
    sd_w = float(np.sqrt((t_fit * (s - mu_w) ** 2).sum() / tw)) + 0.05
    out = []
    if readout == "monotone":
        for g in (3.0, -3.0):
            out.append({"w0": w0, "a": alpha0, "u": u.copy(),
                        "g": g, "b": -g * mu_w})
    else:
        for sig in (min(max(sd_w, 0.1), 1.5), 0.3):
            out.append({"w0": w0, "a": alpha0, "u": u.copy(),
                        "mu": mu_w, "sig": sig})
    return out


def band_warm_start(row: dict, axis: dict | None = None,
                    n_on: int = 1) -> dict | None:
    """Warm start for the constrained readouts, built from a *previous*
    unconstrained band-pass fit of the same mask (fairness measure: we ask
    about expressiveness, not optimizer luck).

    Re-parameterizes the s field so the fitted band [mu-h, mu+h] lands on
    `n_on` grid cells:  q~ = lam*(q - q_mu) + q_c  with
    lam = h_target/h, q_x = 3*artanh(x/3), and sets sigma from the slope
    match  k_new = dmu/sigma^2  (k_new = k/lam).
    """
    if row is None or "h" not in row or "k" not in row:
        return None
    ax = axis or CONSTRAINED_AXIS
    grid = axis_grid(ax)
    dmu = float(grid[1] - grid[0])
    k_idx = int(len(grid) // 2)
    h_t = 0.5 * n_on * dmu
    lam = h_t / max(float(row["h"]), 1e-6)
    at = lambda x: 3.0 * np.arctanh(np.clip(x / 3.0, -0.999, 0.999))  # noqa
    q_mu, q_c = at(float(row["mu"])), at(float(grid[k_idx]))
    w0 = lam * (float(row["w0"]) - q_mu) + q_c
    sig = float(np.sqrt(dmu * lam / max(float(row["k"]), 1e-6)))
    sig = float(np.clip(sig, ax["sig_lo"] * 1.05, ax["sig_hi"] * 0.95))
    on = np.abs(np.arange(len(grid)) - k_idx) <= (n_on - 1) / 2.0
    return {"w0": w0, "a": max(lam * float(row["alpha"]), 1e-3),
            "u": np.asarray(row["w_dir"], dtype=np.float64),
            "mu": float(grid[k_idx]), "sig": sig, "hw": h_t,
            "k_idx": k_idx, "sig0": sig,
            "c0": np.where(on, 0.98, 0.02),
            "o0": np.full(len(grid), 0.5)}


def axis_response(row: dict, s: np.ndarray) -> np.ndarray:
    """Constrained-axis response r(s) rebuilt from a stored result row."""
    mu = np.asarray(row["mu_grid"])
    sig = np.asarray(row["sigma"])
    if row["readout"] == "cgauss":
        return np.exp(-0.5 * ((s - mu[0]) / sig[0]) ** 2)
    N = np.exp(-0.5 * ((s[..., None] - mu) / sig) ** 2)
    o, c = np.asarray(row["o"]), np.asarray(row["c"])
    if row["readout"] == "cband_norm":
        return (N * (o * c)).sum(-1) / ((N * o).sum(-1) + 1e-9)
    return np.clip((N * (o * c)).sum(-1), None, 1.0)


def predict_mask(row: dict, Phi: np.ndarray, readout: str) -> np.ndarray:
    """Rebuild m-hat from a stored result row (for viz)."""
    w_dir = np.asarray(row["w_dir"])
    q = row["w0"] + row["alpha"] * (Phi @ w_dir)
    s = 3.0 * np.tanh(q / 3.0)
    if readout == "monotone":
        return 1.0 / (1.0 + np.exp(-(row["g"] * s + row["b"])))
    if readout == "bandpass":
        sg = lambda z: 1.0 / (1.0 + np.exp(-z))  # noqa: E731
        return (sg(row["k"] * (s - row["mu"] + row["h"]))
                - sg(row["k"] * (s - row["mu"] - row["h"])))
    if readout in CONSTRAINED_READOUTS:
        return axis_response(row, s)
    return np.exp(-0.5 * ((s - row["mu"]) / row["sigma"]) ** 2)


# --------------------------------------------------------------------------
# CLIP dense semantic channels (MaskCLIP-style value-projection readout)
# --------------------------------------------------------------------------

class ClipDense:
    """openai/clip-vit-large-patch14-336 dense patch features projected to
    the joint space (MaskCLIP trick: final block attention replaced by
    identity over the value path), cosine sims against 6 fixed text anchors.
    """

    def __init__(self, device: str = "cuda:0",
                 model_id: str = "openai/clip-vit-large-patch14-336",
                 input_px: int = 672):
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.model = CLIPModel.from_pretrained(
            model_id, local_files_only=True).eval().to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_id,
                                                  local_files_only=True)
        self.input_px = input_px
        with torch.no_grad():
            embs = []
            for name in ANCHOR_ORDER:
                tok = self.proc(text=ANCHORS[name], return_tensors="pt",
                                padding=True).to(self.device)
                e = self.model.get_text_features(**tok)
                e = torch.nn.functional.normalize(e, dim=-1).mean(0)
                embs.append(torch.nn.functional.normalize(e, dim=-1))
            self.text = torch.stack(embs)          # (6, 768)

    @torch.no_grad()
    def sim_maps(self, img: np.ndarray) -> np.ndarray:
        """img: (H,W,3) sRGB [0,1] -> (g,g,6) float32 cosine sims,
        g = input_px/14."""
        from PIL import Image
        pil = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
        pil = pil.resize((self.input_px, self.input_px),
                         Image.Resampling.LANCZOS)
        px = self.proc(images=pil, return_tensors="pt",
                       do_resize=False, do_center_crop=False)
        pv = px["pixel_values"].to(self.device)

        vm = self.model.vision_model
        # manual embedding with bicubic pos-emb interpolation (transformers
        # 4.36 CLIPVisionEmbeddings has no interpolate_pos_encoding kwarg)
        pe = vm.embeddings
        patch = pe.patch_embedding(pv)                 # (1, dim, g, g)
        g = patch.shape[-1]
        patch = patch.flatten(2).transpose(1, 2)       # (1, g*g, dim)
        cls_tok = pe.class_embedding.reshape(1, 1, -1)
        pos = pe.position_embedding.weight             # (577, dim)
        g0 = int(np.sqrt(pos.shape[0] - 1))
        grid_pos = pos[1:].reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
        if g != g0:
            grid_pos = torch.nn.functional.interpolate(
                grid_pos, size=(g, g), mode="bicubic", align_corners=False)
        grid_pos = grid_pos.permute(0, 2, 3, 1).reshape(g * g, -1)
        hidden = torch.cat([cls_tok, patch], dim=1) + torch.cat(
            [pos[:1], grid_pos], dim=0).unsqueeze(0)
        hidden = vm.pre_layrnorm(hidden)
        for layer in vm.encoder.layers[:-1]:
            hidden = layer(hidden, None, None)[0]
        last = vm.encoder.layers[-1]
        hn = last.layer_norm1(hidden)
        v = last.self_attn.v_proj(hn)
        attn_out = last.self_attn.out_proj(v)      # identity attention
        hidden = hidden + attn_out
        hidden = hidden + last.mlp(last.layer_norm2(hidden))
        hidden = vm.post_layernorm(hidden)
        feats = self.model.visual_projection(hidden[:, 1:])   # (1, g*g, 768)
        feats = torch.nn.functional.normalize(feats, dim=-1)
        sims = feats[0] @ self.text.T                          # (g*g, 6)
        g = int(np.sqrt(sims.shape[0]))
        return sims.reshape(g, g, 6).float().cpu().numpy()
