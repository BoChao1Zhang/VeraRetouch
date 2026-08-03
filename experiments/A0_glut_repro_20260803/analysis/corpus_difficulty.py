"""Hypothesis (c): is our 64^3 corpus harder than GLUT's own 75-LUT subset?

GLUT's files are not public (official repo is an empty shell, IMPL_DOSSIER
2.1), so the claim cannot be settled by re-running their data.  What we CAN do
is make the difficulty axis explicit and measurable, then show (i) where our
75 sit on it, (ii) how strongly difficulty predicts the achieved PSNR, and
(iii) what the achieved PSNR would be on the smooth end of the axis -- the end
where a film-emulation corpus like NILUT's (the source of GLUT's own 7-LUT
subset, dossier 4.1) lives.

Model-free per-LUT difficulty features, all computed on the frozen A0 GT cache
(native-resolution cube -> colour tetrahedral, 128^3 train colours):

  affine_psnr    PSNR of the best global 3x3+bias least-squares fit.  This is
                 exactly what GLUT's global branch (G,g) gets for free, so it
                 is the "floor" the 32 Gaussians have to improve on.
  resid_rms      RMS of LUT(x) - x  (departure from identity)
  nonaffine_rms  RMS of the residual left after the best global affine
  clip_frac      fraction of output channels pinned at 0 or 1 (hard gamut
                 clipping = a non-smooth, piecewise-flat map)
  lip_p999       99.9th pct of the local Jacobian Frobenius norm on the 64^3
                 grid (sharp transitions / selective-colour edits)
  curv_mean      mean |second difference| on the 64^3 grid (curvature)
  hue_rot_std    std of the per-colour hue rotation in CIELab (selective hue
                 manipulation -- a global film LUT rotates hue coherently, a
                 retouch preset does not)

Output: analysis/corpus_difficulty.json + .csv
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools", "cube"))

from model.glut_repro import data  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def native_table(path: str) -> np.ndarray:
    """(S,S,S,3) table of the original cube, index [r,g,b].

    `_read_cube_with_fallback` may return a LUTSequence for multi-part cubes;
    the 3D table is the last element in that case (colour's own convention).
    """
    lut, _ = data.load_lut_native(path)
    tbl = getattr(lut, "table", None)
    if tbl is None:                      # LUTSequence -> take the LUT3D member
        members: list = list(lut)        # type: ignore[call-overload]
        cands = [getattr(x, "table", None) for x in reversed(members)]
        tbl = next(c for c in cands if c is not None and np.ndim(c) == 4)
    t = np.asarray(tbl, dtype=np.float32)
    if t.ndim != 4:
        raise ValueError(f"{path}: expected a 3D LUT table, got shape {t.shape}")
    return t


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    from cubelib import srgb_to_lab as _f
    return _f(rgb.astype(np.float32))


def features(lut_id: str, path: str, colors: np.ndarray) -> dict:
    gt = np.load(data._cache_path("a0", lut_id, "train")).astype(np.float32)
    x = colors

    # --- global affine fit (with bias) -----------------------------------
    A = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], 1)
    # normal equations on a subsample (exact enough: 4x4 system)
    sub = np.random.default_rng(0).choice(x.shape[0], 1 << 19, replace=False)
    W, *_ = np.linalg.lstsq(A[sub], gt[sub], rcond=None)
    fit = A @ W
    mse_aff = float(np.mean((fit - gt) ** 2))
    affine_psnr = 99.0 if mse_aff <= 1e-12 else float(10 * np.log10(1.0 / mse_aff))

    resid_rms = float(np.sqrt(np.mean((gt - x) ** 2)))
    nonaffine_rms = float(np.sqrt(mse_aff))
    clip_frac = float(np.mean((gt <= 1e-6) | (gt >= 1 - 1e-6)))

    # --- grid smoothness on the native table ------------------------------
    t = native_table(path)                       # (S,S,S,3)
    S = t.shape[0]
    d = [np.diff(t, axis=ax) * (S - 1) for ax in range(3)]   # d out / d in
    jac = np.sqrt(sum((di[:S-1, :S-1, :S-1] ** 2).sum(-1) for di in d))
    lip_p999 = float(np.percentile(jac, 99.9))
    lip_med = float(np.median(jac))
    curv = np.mean([np.abs(np.diff(t, n=2, axis=ax)).mean() * (S - 1) ** 2
                    for ax in range(3)])
    curv_mean = float(curv)

    # --- hue behaviour ----------------------------------------------------
    sub2 = np.random.default_rng(1).choice(x.shape[0], 1 << 18, replace=False)
    lab_i, lab_o = srgb_to_lab(x[sub2]), srgb_to_lab(gt[sub2])
    hi = np.arctan2(lab_i[:, 2], lab_i[:, 1])
    ho = np.arctan2(lab_o[:, 2], lab_o[:, 1])
    c_i = np.hypot(lab_i[:, 1], lab_i[:, 2])
    m = c_i > 5.0                                  # ignore near-neutral inputs
    dh = np.angle(np.exp(1j * (ho[m] - hi[m])))
    hue_rot_std = float(np.std(dh))
    hue_rot_mean = float(np.abs(np.mean(dh)))

    return {"lut_id": lut_id, "affine_psnr": affine_psnr,
            "resid_rms": resid_rms, "nonaffine_rms": nonaffine_rms,
            "clip_frac": clip_frac, "lip_p999": lip_p999, "lip_med": lip_med,
            "curv_mean": curv_mean, "hue_rot_std": hue_rot_std,
            "hue_rot_mean": hue_rot_mean, "native_size": int(S)}


def main() -> None:
    lut_list = os.path.join(HERE, "..", "config", "a0_luts_75.txt")
    luts = []
    with open(lut_list) as f:
        for line in f:
            if line.strip():
                lid, p = line.rstrip("\n").split("\t")
                luts.append((lid, p))
    # reference: the one public NILUT cube (GLUT's own 7-LUT subset source)
    if os.path.exists(data.NILUT_LUT01):
        luts.append(("nilut__LUT01", data.NILUT_LUT01))

    colors = data.train_colors()
    rows = []
    for lid, p in luts:
        try:
            rows.append(features(lid, p, colors))
            print(json.dumps(rows[-1]), flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {lid}: {e!r}", flush=True)

    # join achieved PSNR from the A0 rec arm
    per = {}
    pl = os.path.join(HERE, "..", "runs", "rec", "per_lut.jsonl")
    with open(pl) as f:
        for line in f:
            r = json.loads(line)
            per[r["lut_id"]] = r
    for r in rows:
        pr = per.get(r["lut_id"])
        r["rec_psnr"] = pr["psnr_float"] if pr else None
        r["rec_de00"] = pr["de00"]["mean"] if pr else None

    # correlations with achieved PSNR
    have = [r for r in rows if r["rec_psnr"] is not None]
    y = np.array([r["rec_psnr"] for r in have])
    corr = {}
    for k in ("affine_psnr", "resid_rms", "nonaffine_rms", "clip_frac",
              "lip_p999", "lip_med", "curv_mean", "hue_rot_std",
              "hue_rot_mean"):
        v = np.array([r[k] for r in have])
        corr[k] = {"pearson_r": float(np.corrcoef(v, y)[0, 1]),
                   "spearman_r": float(np.corrcoef(
                       np.argsort(np.argsort(v)),
                       np.argsort(np.argsort(y)))[0, 1])}

    # single-feature OLS on the strongest predictor -> what PSNR would a corpus
    # as smooth as the top decile of ours (or as NILUT LUT01) achieve?
    best = max(corr, key=lambda k: abs(corr[k]["pearson_r"]))
    v = np.array([r[best] for r in have])
    A = np.stack([v, np.ones_like(v)], 1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    pred_at = {}
    for q, lab in ((90, "our_top_decile"), (75, "our_top_quartile"),
                   (50, "our_median")):
        vq = float(np.percentile(v, q if coef[0] > 0 else 100 - q))
        pred_at[lab] = {f"{best}": vq, "predicted_rec_psnr": float(coef[0] * vq + coef[1])}
    nil = next((r for r in rows if r["lut_id"] == "nilut__LUT01"), None)
    if nil:
        pred_at["nilut_LUT01"] = {
            f"{best}": nil[best],
            "predicted_rec_psnr": float(coef[0] * nil[best] + coef[1]),
            "measured_rec_psnr_smoke": 49.02}

    out = {"n_luts": len(rows), "per_lut": rows,
           "corr_with_rec_psnr": corr,
           "best_predictor": best,
           "ols_slope_intercept": [float(coef[0]), float(coef[1])],
           "psnr_projection": pred_at,
           "corpus_summary": {
               k: {"p10": float(np.percentile([r[k] for r in have], 10)),
                   "p50": float(np.percentile([r[k] for r in have], 50)),
                   "p90": float(np.percentile([r[k] for r in have], 90))}
               for k in ("affine_psnr", "nonaffine_rms", "clip_frac",
                         "lip_p999", "curv_mean", "hue_rot_std")},
           "nilut_LUT01": nil}
    with open(os.path.join(HERE, "corpus_difficulty.json"), "w") as f:
        json.dump(out, f, indent=1)
    keys = ["lut_id", "rec_psnr", "affine_psnr", "nonaffine_rms", "clip_frac",
            "lip_p999", "curv_mean", "hue_rot_std"]
    with open(os.path.join(HERE, "corpus_difficulty.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k)) for k in keys) + "\n")
    print(json.dumps({k: out[k] for k in
                      ("corr_with_rec_psnr", "best_predictor",
                       "psnr_projection", "corpus_summary")}, indent=1))


if __name__ == "__main__":
    main()
