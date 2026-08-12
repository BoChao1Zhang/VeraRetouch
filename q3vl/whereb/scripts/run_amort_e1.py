"""E1: is the per-image Phi-71 fit ill-conditioned / non-unique?  (CPU only)

M1 (mode averaging) needs the solution manifold to be genuinely multi-modal, and
M2 (pathological regression target) needs ``dw*/dx = -H^-1 B`` to blow up.  Both
hinge on the same measurable object: the inner Hessian ``A_x^T Λ A_x``.  This
measures it on every V_where local sample and reports the spectrum.

**The exact object, taken from the code rather than from the doc's idealisation.**
``q3vl.where.oracle._forward`` is::

    q = w0 + alpha * (phi @ w_dir)
    s = S_SCALE * tanh(q / S_SCALE)          # S_SCALE = 3
    m = apply_readout(readout, s, rho)

so the fit is **not** a linear weighted least squares -- it is a multi-start
non-linear fit of ``soft_iou_minmax`` over ``(w_dir, w0, alpha, rho)``.  The
research doc's "per-image WLS" is an idealisation.  What survives exactly is that
``m`` depends on ``w_dir`` only through ``phi @ w_dir``, so the Gauss-Newton
Hessian in ``w_dir`` really is ``A^T Λ A`` with

    Λ = diag[ ( dm/ds * sech^2(q/S_SCALE) * alpha )^2 ]

evaluated at the oracle solution.  Two conditionings are therefore reported and
kept apart:

* **structural** -- ``cond(Φ^T Φ)``: is the 71-column basis itself degenerate?
* **effective**  -- ``cond(Φ^T Λ Φ)``: is the *fit the oracle actually solved*
  ill-posed?  This is the one M1/M2 are about; the readout saturates outside the
  band, so Λ can be near-zero over most of the image and the effective problem
  can be far worse conditioned than the structural one.

Two further pieces of evidence, both already on disk and both free:

* **oracle w\\* per-dimension cross-sample variance** (asked for by the card);
* **the multi-start landscape**: every oracle record carries ``start_losses``
  over 18 (band) / 9 (cband12) restarts plus ``best_start``.  That is a direct,
  already-paid-for reading on how bumpy the per-image objective is -- the closest
  thing to positive M1 evidence obtainable without training anything.
  Note also ``latent.sign_index``: ``w_dir`` is a *direction* with a canonicalised
  sign, i.e. an exact 2-fold symmetry of the solution set that any w-space
  regressor sees as bimodal unless the canonicalisation is applied consistently.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _agg(v) -> dict[str, Any] | None:
    a = np.asarray([x for x in v if x is not None and np.isfinite(x)], dtype=np.float64)
    if not a.size:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--oracle", default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", default="band")
    ap.add_argument("--near-zero-rel", type=float, default=1e-8,
                    help="eigenvalue < rel * lambda_max counts as near-zero")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.where.config import S_SCALE
    from q3vl.where.oracle import _forward
    from q3vl.where.phi import build_phi_dir, effective_gram_cond
    from q3vl.whereb.stores import OracleStore

    cache = Path(args.cache)
    man = json.loads((cache / "manifest.json").read_text())
    samples = man["samples"][: args.limit] if args.limit else man["samples"]
    ost = OracleStore(Path(args.oracle))

    rows: list[dict[str, Any]] = []
    w_dirs, w0s, alphas = [], [], []
    start_loss_rows = []

    for n, rec in enumerate(samples):
        sid = rec["sample_id"]
        gh, gw = rec["grid16"]
        d = np.load(cache / "cache" / f"{sid}.npz")
        sem = torch.from_numpy(d["semantic_low"]).double()
        img = torch.from_numpy(d["img_low"]).double()
        parts = build_phi_dir(sem, img, gh, gw)
        A = parts.phi_dir                                     # (P, 71) double

        try:
            orec = ost.read_json(sid, ".oracle.json")
        except Exception:
            continue
        fit = orec["fits"].get(args.readout)
        if not fit or not fit.get("usable"):
            continue
        lat = fit["latent"]
        w_dir = torch.tensor(lat["w_dir"], dtype=torch.float64)
        w0 = float(lat["w0"])
        alpha = float(lat.get("alpha", fit["latent"].get("alpha", 1.0)))
        rho = {k: torch.tensor(float(v), dtype=torch.float64)
               for k, v in lat["rho_raw"].items()}

        # --- structural conditioning: the basis itself ---------------------
        st = effective_gram_cond(A)

        # --- effective conditioning: the fit the oracle actually solved ----
        q = w0 + alpha * (A @ w_dir)
        s = S_SCALE * torch.tanh(q / S_SCALE)
        s_ = s.clone().requires_grad_(True)
        from q3vl.where.readout import apply_readout
        m = apply_readout(args.readout, s_, rho)
        (dm_ds,) = torch.autograd.grad(m.sum(), s_)
        sech2 = 1.0 - torch.tanh(q / S_SCALE) ** 2
        lam = (dm_ds * sech2 * alpha) ** 2                     # (P,)
        Aw = A * lam.sqrt().unsqueeze(1)                       # so Aw^T Aw = A^T Λ A
        ef = effective_gram_cond(Aw)

        H = (A.T * lam.unsqueeze(0)) @ A
        ev = torch.linalg.eigvalsh(H).clamp_min(0).numpy()
        lmax = float(ev.max()) if ev.size else 0.0
        n_near_zero = int((ev < args.near_zero_rel * max(lmax, 1e-300)).sum())

        w_dirs.append(w_dir.numpy())
        w0s.append(w0)
        alphas.append(alpha)
        sl = fit.get("start_losses") or []
        if sl:
            sl = np.asarray(sl, dtype=np.float64)
            finite = sl[np.isfinite(sl)]
            if finite.size:
                best = float(finite.min())
                start_loss_rows.append({
                    "sample_id": sid, "n_starts": int(finite.size),
                    "best": best, "worst": float(finite.max()),
                    "spread": float(finite.max() - best),
                    # how many restarts land essentially on the best basin
                    "n_within_1pct": int((finite <= best + 0.01 * max(abs(best), 1e-9)).sum()),
                    "n_within_abs_0.005": int((finite <= best + 0.005).sum()),
                    "best_start": fit.get("best_start"),
                })

        rows.append({
            "sample_id": sid, "grid": [gh, gw], "P": int(A.shape[0]),
            "structural_cond": st["cond"], "structural_dropped": st["n_dropped_columns"],
            "structural_rank": st["rank"],
            "effective_cond": ef["cond"], "effective_rank": ef["rank"],
            "effective_dropped": ef["n_dropped_columns"],
            "n_near_zero_eig": n_near_zero,
            "lambda_max": lmax,
            "lambda_min_pos": float(ev[ev > 0].min()) if (ev > 0).any() else None,
            "weight_mass_frac": float((lam > 0.01 * lam.max()).double().mean()),
            "soft_iou": fit["metrics"]["soft_iou_minmax"],
            "sign_index": lat.get("sign_index"),
        })
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(samples)}] {time.time()-t0:.0f}s", flush=True)

    W = np.asarray(w_dirs)                                    # (N, 71)
    out: dict[str, Any] = {
        "card": "PR-AMORT / E1 -- conditioning of the per-image Phi-71 fit",
        "n_samples": len(rows),
        "readout": args.readout,
        "near_zero_rel": args.near_zero_rel,
        "what_was_measured": {
            "structural": "cond(Phi^T Phi) -- the 71-column basis alone",
            "effective": ("cond(Phi^T Lambda Phi), Lambda = (dm/ds * sech^2(q/3) * alpha)^2 "
                          "at the oracle solution -- the Gauss-Newton Hessian in w_dir of "
                          "the fit the oracle actually solved"),
            "caveat": ("the Where-A oracle is NOT a linear WLS: it is a multi-start "
                       "non-linear fit of soft_iou_minmax over (w_dir, w0, alpha, rho). "
                       "m depends on w_dir only through phi @ w_dir, which is what makes "
                       "A^T Lambda A the right object anyway."),
        },
        "structural_cond": _agg([r["structural_cond"] for r in rows]),
        "effective_cond": _agg([r["effective_cond"] for r in rows]),
        "n_near_zero_eig": _agg([r["n_near_zero_eig"] for r in rows]),
        "effective_rank": _agg([r["effective_rank"] for r in rows]),
        "weight_mass_frac": _agg([r["weight_mass_frac"] for r in rows]),
        "oracle_soft_iou": _agg([r["soft_iou"] for r in rows]),
        "oracle_w_star": {
            "n": int(W.shape[0]), "dim": int(W.shape[1]),
            "per_dim_std": W.std(axis=0).tolist(),
            "per_dim_mean_abs": np.abs(W).mean(axis=0).tolist(),
            "per_dim_std_summary": _agg(W.std(axis=0).tolist()),
            "mean_norm": float(np.linalg.norm(W, axis=1).mean()),
            "note": ("w_dir is a normalised DIRECTION with a canonicalised sign "
                     "(latent.sign_index): the solution set has an exact 2-fold "
                     "symmetry that a w-space regressor sees as bimodal unless the "
                     "canonicalisation is applied identically at train time."),
            "alpha": _agg(alphas), "w0": _agg(w0s),
        },
        "multistart_landscape": {
            "note": ("already-on-disk evidence about how bumpy the per-image objective "
                     "is: each record stores the loss of every restart"),
            "n_samples": len(start_loss_rows),
            "n_starts": _agg([r["n_starts"] for r in start_loss_rows]),
            "loss_spread_best_to_worst": _agg([r["spread"] for r in start_loss_rows]),
            "n_restarts_within_1pct_of_best": _agg(
                [r["n_within_1pct"] for r in start_loss_rows]),
            "n_restarts_within_abs_0.005": _agg(
                [r["n_within_abs_0.005"] for r in start_loss_rows]),
            "frac_samples_where_only_one_start_is_best": float(np.mean(
                [r["n_within_1pct"] <= 1 for r in start_loss_rows])) if start_loss_rows else None,
        },
        "per_sample": rows,
    }

    # ---- the registered verdict ----------------------------------------
    med = out["effective_cond"]["median"] if out["effective_cond"] else float("inf")
    nz = out["n_near_zero_eig"]["median"] if out["n_near_zero_eig"] else None
    out["VERDICT"] = {
        "gate": "condition number median < 1e2 AND no near-zero spectrum -> M1 downgraded",
        "effective_cond_median": med,
        "near_zero_eig_median": nz,
        "cond_below_1e2": bool(med < 1e2),
        "no_near_zero": bool(nz is not None and nz == 0),
        "M1_downgraded": bool(med < 1e2 and nz == 0),
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("\n=== E1 ===")
    print(f"n = {out['n_samples']}   oracle soft-IoU median "
          f"{out['oracle_soft_iou']['median']:.4f}")
    for k in ("structural_cond", "effective_cond"):
        a = out[k]
        print(f"{k:18s} median {a['median']:.4g}  p10 {a['p10']:.4g}  p90 {a['p90']:.4g}  "
              f"max {a['max']:.4g}")
    a = out["n_near_zero_eig"]
    print(f"near-zero eigenvalues (rel<{args.near_zero_rel}): median {a['median']:.1f} "
          f"p90 {a['p90']:.1f} max {a['max']:.0f}")
    a = out["effective_rank"]
    print(f"effective rank (of 71): median {a['median']:.1f}  min {a['min']:.0f}")
    a = out["weight_mass_frac"]
    print(f"frac cells with non-negligible Lambda: median {a['median']:.3f}")
    ms = out["multistart_landscape"]
    print(f"multi-start: {ms['n_starts']['median']:.0f} restarts, "
          f"loss spread median {ms['loss_spread_best_to_worst']['median']:.4f}, "
          f"restarts within 1% of best: median "
          f"{ms['n_restarts_within_1pct_of_best']['median']:.1f}")
    w = out["oracle_w_star"]["per_dim_std_summary"]
    print(f"oracle w* per-dim std: median {w['median']:.4f} min {w['min']:.4f} "
          f"max {w['max']:.4f}")
    print(f"\nVERDICT: {json.dumps(out['VERDICT'], indent=2)}")
    print(f"done in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
