"""Analytic edge-quality metrics (E_HF / kappa~ / MVR / AFR).

Registered in RESEARCH_analytic-edge-quality_2026-08-11.md §2.0 as the missing
"analytically clean" column (gap M6), and they are the acceptance yardstick for
the P1/P2/P3 fixes -- so they are implemented once, here, rather than inside a
one-off script.

Project red lines honoured:
  * **no AUC in any form** -- nothing here computes a ranking statistic;
  * **arm-constant calibration, never per image** -- the high-frequency band
    `B_hi` and the gradient threshold `tau` are fitted once over the whole arm
    (:class:`EdgeQualCalibration`) and then frozen.  A per-image band would make
    every field its own yardstick and the numbers incomparable, which is the
    same failure the s-cache contract forbids for normalisation.

Scale convention: all spatial frequencies are expressed in **normalised**
cycles/sample (``fftfreq``, so the Nyquist edge is 0.5) precisely because the
H/16 grid is not a fixed shape here -- 13 distinct grid shapes appear in 400
samples, and an unnormalised radius would mean a different band per aspect
ratio.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["EdgeQualCalibration", "calibrate", "calibrate_per_family",
           "radial_freq", "hf_energy", "e_hf", "kappa_tilde", "mvr", "afr",
           "direction_field", "edge_quality_row", "shape_residual",
           "best_fit_analytic", "band_occupancy"]


def radial_freq(h: int, w: int) -> np.ndarray:
    """Normalised radial frequency magnitude, shape ``(h, w)``, max ~0.707."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    return np.sqrt(fy ** 2 + fx ** 2)


def _spectrum(y: np.ndarray) -> np.ndarray:
    """Power spectrum of a zero-meaned field.

    The DC term is removed first: it carries the field's mean (i.e. its area),
    which is already reported by its own column, and leaving it in would let a
    pure area shift masquerade as spectral content.
    """
    a = np.asarray(y, dtype=np.float64)
    a = a - a.mean()
    return np.abs(np.fft.fft2(a)) ** 2


@dataclass
class EdgeQualCalibration:
    """Arm-wide constants.  Fit once, freeze, record next to the numbers."""

    r_hi: float = 0.0          #: B_hi = { f : |f| > r_hi }, normalised units
    energy_frac: float = 0.995  #: the disk |f| <= r_hi holds this much GT energy
    tau: float = 0.0           #: narrow-band gradient threshold (arm quantile)
    tau_quantile: float = 0.70
    n_samples: int = 0
    note: str = ("B_hi and tau are ARM CONSTANTS fitted on GT analytic fields; "
                 "per-image calibration is forbidden (numbers must be comparable "
                 "across samples and across arms)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EdgeQualCalibration":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def calibrate(gt_fields: Sequence[np.ndarray], *, energy_frac: float = 0.995,
              tau_quantile: float = 0.70) -> EdgeQualCalibration:
    """Find the smallest low-frequency disk holding ``energy_frac`` of GT energy.

    Pooled over the arm's GT fields: the cumulative radial energy profile is
    accumulated first and the radius read off the pooled curve, so one unusually
    sharp mask cannot set the band for everybody.
    """
    edges = np.linspace(0.0, 0.75, 151)
    cum = np.zeros(len(edges))
    tot = 0.0
    grads: list[float] = []
    for y in gt_fields:
        y = np.asarray(y, dtype=np.float64)
        p = _spectrum(y)
        r = radial_freq(*y.shape)
        tot += p.sum()
        idx = np.searchsorted(edges, r.ravel(), side="right") - 1
        idx = np.clip(idx, 0, len(edges) - 1)
        cum += np.bincount(idx, weights=p.ravel(), minlength=len(edges))
        gy, gx = np.gradient(y)
        grads.append(np.sqrt(gy ** 2 + gx ** 2).ravel())
    c = np.cumsum(cum) / max(tot, 1e-12)
    k = int(np.searchsorted(c, energy_frac))
    r_hi = float(edges[min(k, len(edges) - 1)])
    allg = np.concatenate(grads) if grads else np.zeros(1)
    tau = float(np.quantile(allg[allg > 0], tau_quantile)) if (allg > 0).any() else 0.0
    return EdgeQualCalibration(r_hi=r_hi, energy_frac=energy_frac, tau=tau,
                               tau_quantile=tau_quantile, n_samples=len(gt_fields))


def calibrate_per_family(gt_by_family: dict[str, Sequence[np.ndarray]],
                        **kw) -> dict[str, EdgeQualCalibration]:
    """One calibration per family -- still an ARM CONSTANT, just not a global one.

    Measured 2026-08-11: GT gradient magnitudes differ by **two orders of
    magnitude** between families (linear ramps max ~0.006, semantic silhouettes
    ~0.60).  A single pooled ``tau`` is therefore set by the sharp families and
    lands *above the entire gradient range* of every linear ramp, so the narrow
    band is empty and ``kappa~`` is undefined for that whole family -- silently,
    as NaN, which then drops out of any median.

    Per-family constants keep the property that actually matters (the threshold
    is not a function of the individual image, so numbers stay comparable within
    a family) while making the metric defined everywhere.  Family is a
    structural label available at eval time, and the type-word router recovers
    it at 400/400, so this does not introduce a new dependency.
    """
    return {f: calibrate(v, **kw) for f, v in gt_by_family.items() if v}


def hf_energy(y: np.ndarray, cal: EdgeQualCalibration) -> tuple[float, float]:
    """``(energy above B_hi, total energy)`` for one field."""
    p = _spectrum(y)
    r = radial_freq(*np.asarray(y).shape)
    return float(p[r > cal.r_hi].sum()), float(p.sum())


def e_hf(y: np.ndarray, gt: np.ndarray, cal: EdgeQualCalibration) -> float:
    """Excess high-frequency energy of ``y`` over ``gt``, relative to GT total.

    Positive = the field carries high-frequency content the GT does not, which
    is the quantity "dirty edge" names.  Negative is possible and meaningful
    (over-smoothed relative to GT) and is **not** clipped -- clipping would hide
    the direction of the error.
    """
    hy, _ = hf_energy(y, cal)
    hg, tg = hf_energy(gt, cal)
    return float((hy - hg) / max(tg, 1e-12))


def kappa_tilde(y: np.ndarray, cal: EdgeQualCalibration,
                eps: float = 1e-6, band_from: np.ndarray | None = None) -> float:
    """Median ``|div(grad y / |grad y|)|`` inside a narrow band.

    ``band_from`` (strongly recommended: pass the GT) fixes **where** curvature
    is measured.  Deriving the band from ``y`` itself, as the first version did,
    breaks any A/B comparison in two ways measured on 2026-08-11:

    * a field that is *smoother* than the threshold has an empty band and scores
      NaN -- so the smoother of two candidates silently drops out of the median
      instead of scoring well, which is the opposite of the intended reading;
    * the ratio ``kappa(y)/kappa(gt)`` explodes when the GT's own band is nearly
      empty (observed 2.4e7 on a band-family sample).

    Measuring both candidates inside the **GT's** band puts them on identical
    pixels and asks the right question: how curved is this field where the
    target actually has a contour.
    """
    a = np.asarray(y, dtype=np.float64)
    gy, gx = np.gradient(a)
    mag = np.sqrt(gy ** 2 + gx ** 2)
    ny, nx = gy / (mag + eps), gx / (mag + eps)
    div = np.gradient(ny, axis=0) + np.gradient(nx, axis=1)
    if band_from is not None:
        bgy, bgx = np.gradient(np.asarray(band_from, dtype=np.float64))
        band = np.sqrt(bgy ** 2 + bgx ** 2) > cal.tau
    else:
        band = mag > cal.tau
    if band.sum() < 4:
        # Report rather than silently NaN: an empty band means the threshold is
        # wrong for this field's family, not that the field has no curvature.
        return float("nan")
    return float(np.median(np.abs(div[band])))


def band_occupancy(y: np.ndarray, cal: EdgeQualCalibration) -> float:
    """Fraction of cells inside the narrow band -- the coverage audit for kappa~.

    Any kappa~ table must be read together with this: a family whose occupancy
    is ~0 contributed nothing to the aggregate, however healthy the aggregate
    looks.
    """
    gy, gx = np.gradient(np.asarray(y, dtype=np.float64))
    return float((np.sqrt(gy ** 2 + gx ** 2) > cal.tau).mean())


def direction_field(gt: np.ndarray, family: str) -> np.ndarray:
    """Per-cell unit direction ``d`` along which the GT is monotone.

    ``linear``/``band`` -> one constant vector for the whole field, taken as the
    dominant direction of the GT gradient (magnitude-weighted principal axis).
    ``radial`` -> the radial direction from the GT's intensity centroid.
    GT parameters are privileged training-time information and this is a
    diagnostic, which §2.0 explicitly permits.
    """
    a = np.asarray(gt, dtype=np.float64)
    gy, gx = np.gradient(a)
    h, w = a.shape
    if family == "radial":
        m = np.clip(a, 0, None)
        s = m.sum()
        if s <= 0:
            cy, cx = (h - 1) / 2, (w - 1) / 2
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = float((yy * m).sum() / s), float((xx * m).sum() / s)
        yy, xx = np.mgrid[0:h, 0:w]
        dy, dx = yy - cy, xx - cx
        n = np.sqrt(dy ** 2 + dx ** 2) + 1e-9
        return np.stack([dy / n, dx / n], axis=-1)
    wgt = np.sqrt(gy ** 2 + gx ** 2)
    vy = float((gy * wgt).sum())
    vx = float((gx * wgt).sum())
    n = np.hypot(vy, vx) + 1e-9
    d = np.array([vy / n, vx / n])
    return np.broadcast_to(d, (h, w, 2)).copy()


def mvr(y: np.ndarray, gt: np.ndarray, family: str,
        cal: EdgeQualCalibration) -> float:
    """Monotonicity violation rate along the GT direction field.

    The expected sign is taken from the **GT's own** directional derivative, so
    a field that is monotone the other way round is not scored as 100% violating
    for a sign convention reason.
    """
    d = direction_field(gt, family)
    gy, gx = np.gradient(np.asarray(y, dtype=np.float64))
    ggy, ggx = np.gradient(np.asarray(gt, dtype=np.float64))
    dot_y = gy * d[..., 0] + gx * d[..., 1]
    dot_g = ggy * d[..., 0] + ggx * d[..., 1]
    mag = np.sqrt(gy ** 2 + gx ** 2)
    band = mag > cal.tau
    if band.sum() < 4:
        return float("nan")
    sign = np.sign(dot_g[band].sum()) or 1.0
    return float((dot_y[band] * sign < 0).mean())


def _geo5(h: int, w: int) -> np.ndarray:
    """``[1, x, y, x^2, y^2, xy]`` on the campaign's normalised grid."""
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - (w - 1) / 2) / max((w - 1) / 2, 1)
    y = (yy - (h - 1) / 2) / max((h - 1) / 2, 1)
    return np.stack([np.ones_like(x), x, y, x * x, y * y, x * y], axis=-1)


def afr(y: np.ndarray, *, refine: bool = True) -> float:
    """Best-fit residual (RMSE) to the analytic family, in field space.

    "The analytic family" is this project's own: the quadratic form spanned by
    ``geo5`` pushed through a squashing non-linearity -- exactly what
    ``s = 3 tanh(q/3)`` followed by the readout produces, and what `radial`,
    `band` and `linear` GT masks are generated from.

    Closed-form initialisation in logit space, then a bounded Levenberg-
    Marquardt refinement in **field** space (§2.0 asks for grid init + LM).  The
    refinement matters: a logit-space least squares is weighted wrongly for this
    purpose, which is the same mismatch E5 measured when a closed-form ridge on
    logits lost 9-29 IoU points against a direct fit.

    Diagnostic only -- never a training target (avoids the ruling-1 dispute).
    """
    a = np.clip(np.asarray(y, dtype=np.float64), 1e-4, 1 - 1e-4)
    h, w = a.shape
    A = _geo5(h, w).reshape(-1, 6)
    t = np.log(a / (1 - a)).reshape(-1)
    c0, *_ = np.linalg.lstsq(A, t, rcond=None)

    def model(c: np.ndarray) -> np.ndarray:
        q = A @ c
        return 1.0 / (1.0 + np.exp(-np.clip(q, -30, 30)))

    if refine:
        try:
            from scipy.optimize import least_squares

            r = least_squares(lambda c: model(c) - a.reshape(-1), c0,
                              method="lm", max_nfev=200)
            c0 = r.x
        except Exception:
            pass
    return float(np.sqrt(np.mean((model(c0) - a.reshape(-1)) ** 2)))


def best_fit_analytic(y: np.ndarray, family: str) -> np.ndarray:
    """Least-squares best-fitting member of the family's analytic shape."""
    a = np.clip(np.asarray(y, dtype=np.float64), 1e-4, 1 - 1e-4)
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x = (xx - (w - 1) / 2) / max((w - 1) / 2, 1)
    yv = (yy - (h - 1) / 2) / max((h - 1) / 2, 1)
    if family == "linear":
        A = np.stack([np.ones_like(x), x, yv], -1).reshape(-1, 3)   # ramp
    else:
        # ellipse / band: full quadratic form (a radial falloff and a two-sided
        # band are both level sets of a quadratic)
        A = np.stack([np.ones_like(x), x, yv, x * x, yv * yv, x * yv],
                     -1).reshape(-1, 6)
    t = np.log(a / (1 - a)).reshape(-1)
    c, *_ = np.linalg.lstsq(A, t, rcond=None)
    return 1.0 / (1.0 + np.exp(-np.clip(A @ c, -30, 30))).reshape(h, w)


def shape_residual(y: np.ndarray, family: str, k: int | None = None) -> float:
    """``1 - IoU(y, best-fit analytic member of y's own family)``.

    Quantifies "is this the right SHAPE", which is a different question from
    "is this smooth".  A smooth blob and a proper ellipse can have identical
    curvature statistics; only this column separates them, because it asks how
    well the field is explained by *any* member of the analytic family it is
    supposed to belong to.  The GT scores ~0 by construction, which is the
    control that makes the number readable.
    """
    fit = best_fit_analytic(y, family)
    a = np.asarray(y, dtype=np.float64)
    if k is None:
        k = int(max(1, (a > 0.5).sum()))
    def topk(z):
        f = z.reshape(-1)
        idx = np.argpartition(-f, min(k, f.size - 1))[:k]
        m = np.zeros(f.size, bool)
        m[idx] = True
        return m.reshape(z.shape)
    p, q = topk(a), topk(fit)
    inter = float((p & q).sum())
    union = float((p | q).sum())
    return float(1.0 - inter / max(union, 1.0))


def edge_quality_row(y: np.ndarray, gt: np.ndarray, family: str,
                     cal: EdgeQualCalibration, *, with_afr: bool = True
                     ) -> dict[str, float]:
    """All four registered metrics for one field, plus the GT references."""
    k_y, k_g = kappa_tilde(y, cal, band_from=gt), kappa_tilde(gt, cal, band_from=gt)
    row = {
        "E_HF": e_hf(y, gt, cal),
        "kappa_y": k_y, "kappa_gt": k_g,
        "kappa_ratio": (float("nan") if not np.isfinite(k_y) or not np.isfinite(k_g)
                        or k_g <= 0 else k_y / k_g),
        "MVR": mvr(y, gt, family, cal),
        "MVR_gt": mvr(gt, gt, family, cal),
    }
    if with_afr:
        row["AFR"] = afr(y)
        row["AFR_gt"] = afr(gt)
    return row
