"""Paired before/after metrics for preset render-QA (replaces after-only NR-IQA).

All from established libraries (scikit-image + scipy), no hand-rolled math:
  delta_e2000_mean/_p95  CIEDE2000 color difference (edit magnitude / worst color shift)
  ssim                   structural similarity (low => destructive clipping/posterize)
  hist_emd_L / hist_emd_ab  Wasserstein distance of L and a/b distributions (look footprint)
  clip_pct               newly blown highlights / crushed blacks the look introduced
  noop_score             deterministic near-no-op flag (ΔE<1.5 and ssim>0.99)
"""
from __future__ import annotations

from typing import Dict


def paired(before_pil, after_pil, work_longedge: int = 768) -> Dict[str, float]:
    import numpy as np
    from skimage.color import rgb2lab, deltaE_ciede2000
    from skimage.metrics import structural_similarity as ssim
    from scipy.stats import wasserstein_distance

    b = before_pil.convert("RGB")
    # bound work size for speed; keep aspect; after is resized to before's grid
    w, h = b.size
    if max(w, h) > work_longedge:
        s = work_longedge / max(w, h)
        b = b.resize((max(1, int(w * s)), max(1, int(h * s))))
    a = after_pil.convert("RGB").resize(b.size)

    bb = np.asarray(b, dtype="float32") / 255.0
    aa = np.asarray(a, dtype="float32") / 255.0
    lab_b, lab_a = rgb2lab(bb), rgb2lab(aa)
    de = deltaE_ciede2000(lab_b, lab_a)

    def _clipfrac(x):
        return float(((x <= 0.003) | (x >= 0.997)).mean())

    out = {
        "delta_e2000_mean": round(float(de.mean()), 3),
        "delta_e2000_p95": round(float(np.percentile(de, 95)), 3),
        "ssim": round(float(ssim(bb, aa, channel_axis=-1, data_range=1.0)), 4),
        "hist_emd_L": round(float(wasserstein_distance(
            lab_b[..., 0].ravel(), lab_a[..., 0].ravel())), 3),
        "hist_emd_ab": round(float(0.5 * (
            wasserstein_distance(lab_b[..., 1].ravel(), lab_a[..., 1].ravel())
            + wasserstein_distance(lab_b[..., 2].ravel(), lab_a[..., 2].ravel()))), 3),
        "clip_pct": round(max(0.0, _clipfrac(aa) - _clipfrac(bb)), 4),
    }
    out["noop_score"] = 1 if (out["delta_e2000_mean"] < 1.5 and out["ssim"] > 0.99) else 0
    return out
