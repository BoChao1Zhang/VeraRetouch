"""E2 -- is the regression target itself the disease?  Tikhonov-recompute the oracle.

E1 left M2 as the prime suspect: the effective Hessian ``Phi^T Lambda Phi`` has
condition number ~8.4e5 (median), Spearman(cond, oracle IoU) = -0.52, and the
18-restart landscape finds the best basin exactly once.  Under DDN's
``dw*/dx = -H^-1 B`` that makes ``x -> w*`` violently non-smooth, so a regressor
trained on it is fitting solver noise.  The cheapest possible repair is not a new
architecture but one term in the oracle objective.

Where the penalty goes (this is the part that is easy to get wrong)
------------------------------------------------------------------
``_forward`` splits the coefficient vector into a unit direction and a scale::

    w_dir = w_raw / ||w_raw||      (magnitude of w_raw is pure gauge)
    q     = w0 + alpha * (Phi @ w_dir)

so the *effective* coefficient vector is ``w_eff = alpha * w_dir`` and
``||w_eff|| = alpha``.  A penalty on ``w_raw`` would therefore be meaningless --
it only shrinks a gauge freedom and never touches the mask.  ``lambda * alpha^2``
IS the honest Tikhonov term: spending part of the unit direction on a near-null
direction of ``Phi`` shrinks ``||Phi w_dir||``, which forces ``alpha`` up to keep
``q``'s scale, which the penalty then charges for.  That is exactly the
``sigma^2/(sigma^2 + lambda)`` shrinkage of ridge, expressed in this
parametrisation.

Calibration
-----------
Adding ``lambda ||w_eff||^2`` adds ``2 lambda I`` to the Gauss-Newton Hessian, so
``cond -> (l_max + 2 lambda)/(l_min + 2 lambda) ~ l_max / (2 lambda)``.  Targeting
``cond ~ 1e3`` gives ``lambda = l_max / 2e3`` with ``l_max`` the **arm-median**
top eigenvalue (an arm constant, never per image).

Acceptance gate, in order
-------------------------
1. recomputed oracle decoded IoU >= 0.95  (below that the ceiling has been spent
   and nothing downstream is worth running);
2. only then is the retrain worth a GPU slot.

Reported next to the IoU, because they are the actual point of the card:
effective condition number, and the per-dimension coefficient of variation of
``w_eff`` across samples (E1 measured CV median 10.66 with 100% of the 71 dims
above 1 -- "no dimension's cross-sample mean beats its own spread").  If
Tikhonov works, that CV must come down; the IoU only has to survive.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def agg(x: Any) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/home/bc/data/runs/where_b/amort_cache_20260810")
    ap.add_argument("--oracle",
                    default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--readout", default="band")
    ap.add_argument("--target-cond", type=float, default=1e3)
    ap.add_argument("--lambda-scan", type=float, nargs="*",
                    default=[0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
                    help="multipliers of the calibrated lambda_0")
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.where.basis import w_dir_of
    from q3vl.where.config import FitConfig, S_SCALE
    from q3vl.where.fitpool import FitPool, FitTask
    from q3vl.where.oracle import mask_metrics
    from q3vl.where.phi import build_phi_dir
    from q3vl.where.readout import apply_readout
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.config import WHERE_A_MASKVIEW_DIR
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import OracleStore

    out_dir = Path(args.out)
    for sub in ("config", "logs", "viz"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    man = json.loads((cache / "manifest.json").read_text())
    samples = man["samples"][: args.limit] if args.limit else man["samples"]
    ost = OracleStore(Path(args.oracle))
    ds, _ = open_dataset("V_where", need_mask=True, maskview_root=WHERE_A_MASKVIEW_DIR)
    idx_of = {r.sample_id: i for i, r in enumerate(ds.refs)}

    # ---- pass 1: phi, GT, and the published latent's spectrum --------------
    print("pass 1: rebuild phi, read published latents, measure spectrum", flush=True)
    items: list[dict[str, Any]] = []
    lmax_list, cond_pub = [], []
    for rec in samples:
        sid = rec["sample_id"]
        if sid not in idx_of:
            continue
        gh, gw = rec["grid16"]
        d = np.load(cache / "cache" / f"{sid}.npz")
        phi = build_phi_dir(torch.from_numpy(d["semantic_low"]).double(),
                            torch.from_numpy(d["img_low"]).double(), gh, gw).phi_dir
        try:
            fit = ost.read_json(sid, ".oracle.json")["fits"].get(args.readout)
        except Exception:
            continue
        if not fit or fit.get("status") != "ok":
            continue
        lat = fit["latent"]
        w_dir = torch.tensor(lat["w_dir"], dtype=torch.float64)
        w0 = float(lat["w0"]); alpha = float(lat["alpha"])
        rho = {k: torch.tensor(float(v) if np.isscalar(v) or not isinstance(v, list) else v,
                               dtype=torch.float64) for k, v in lat["rho_raw"].items()}

        # GN Hessian of w_eff (no alpha^2 factor: d m / d w_eff = (dm/ds)(ds/dq) Phi)
        q = w0 + alpha * (phi @ w_dir)
        s = S_SCALE * torch.tanh(q / S_SCALE)
        s_ = s.clone().requires_grad_(True)
        m = apply_readout(args.readout, s_, rho)
        (dm_ds,) = torch.autograd.grad(m.sum(), s_)
        sech2 = 1.0 - torch.tanh(q / S_SCALE) ** 2
        lam = (dm_ds * sech2) ** 2
        H = (phi.T * lam.unsqueeze(0)) @ phi
        ev = torch.linalg.eigvalsh(H).clamp_min(0).numpy()
        lmax = float(ev.max())
        lmax_list.append(lmax)
        cond_pub.append(lmax / max(float(ev[ev > 0].min()) if (ev > 0).any() else 0.0, 1e-300))

        gt_hi = ds[idx_of[sid]].mask_target_hi().double()
        gt16 = area_resize(gt_hi[None, None], (gh, gw))[0, 0].reshape(-1).clamp(0, 1)
        items.append({"sample_id": sid, "phi": phi, "gt": gt16, "grid": (gh, gw),
                      "w_eff_pub": (alpha * w_dir).numpy()})

    lmax_med = float(np.median(lmax_list))
    lambda_0 = lmax_med / (2.0 * args.target_cond)
    print(f"  n={len(items)}  lambda_max median {lmax_med:.4e}  "
          f"-> lambda_0 = {lambda_0:.4e} (targets cond {args.target_cond:.0e})", flush=True)
    print(f"  published effective cond median {np.median(cond_pub):.3e}", flush=True)

    # ---- pass 2: refit at each lambda --------------------------------------
    fitcfg = FitConfig(seed=args.seed)
    results: dict[str, Any] = {}
    for mult in args.lambda_scan:
        lam_v = lambda_0 * mult
        tasks = [FitTask(key=it["sample_id"], phi=it["phi"], target=it["gt"],
                         readouts=(args.readout,),
                         cfg=type(fitcfg)(**{**fitcfg.__dict__,
                                             "ridge_lambda": lam_v}))
                 for it in items]
        with FitPool(args.workers) as fp:
            fp.warmup()
            fits = fp.run(tasks)
        ious, conds, w_effs, alphas = [], [], [], []
        for it in items:
            fr = fits[it["sample_id"]][args.readout]
            if fr.latent is None:
                continue
            lat = fr.latent
            wd = w_dir_of(lat.w_raw)
            a = float(torch.nn.functional.softplus(lat.alpha_raw))
            phi = it["phi"]
            qq = lat.w0 + a * (phi @ wd)
            ss = S_SCALE * torch.tanh(qq / S_SCALE)
            mm = apply_readout(args.readout, ss, lat.rho)
            ious.append(mask_metrics(mm, it["gt"])["soft_iou_minmax"])
            w_effs.append((a * wd).numpy())
            alphas.append(a)
            s_ = ss.clone().requires_grad_(True)
            m2 = apply_readout(args.readout, s_, lat.rho)
            (dm_ds,) = torch.autograd.grad(m2.sum(), s_)
            sech2 = 1.0 - torch.tanh(qq / S_SCALE) ** 2
            lm = (dm_ds.detach() * sech2) ** 2
            H = (phi.T * lm.unsqueeze(0)) @ phi + 2.0 * lam_v * torch.eye(
                phi.shape[1], dtype=phi.dtype)
            ev = torch.linalg.eigvalsh(H).clamp_min(1e-300).numpy()
            conds.append(float(ev.max() / ev.min()))
        W = np.asarray(w_effs)
        cv = np.abs(W.std(axis=0) / (np.abs(W.mean(axis=0)) + 1e-12))
        results[f"mult_{mult:g}"] = {
            "lambda": lam_v, "multiplier": mult,
            "oracle_soft_iou": agg(ious),
            "effective_cond_after": agg(conds),
            "alpha": agg(alphas),
            "w_eff_per_dim_CV_median": float(np.median(cv)),
            "frac_dims_CV_above_1": float((cv > 1).mean()),
            "gate_iou_ge_0.95": bool(np.median(ious) >= 0.95),
        }
        print(f"  mult={mult:<5g} lambda={lam_v:.3e}  IoU med {np.median(ious):.4f}  "
              f"cond med {np.median(conds):.3e}  CV med {np.median(cv):.2f}  "
              f"dims CV>1 {100*(cv>1).mean():.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    # published baseline CV, for the same 71 dims
    Wp = np.asarray([it["w_eff_pub"] for it in items])
    cvp = np.abs(Wp.std(axis=0) / (np.abs(Wp.mean(axis=0)) + 1e-12))
    summary = {
        "card": "E2 stage A (Tikhonov oracle recompute + acceptance gate)",
        "n": len(items), "readout": args.readout,
        "lambda_max_median": lmax_med, "lambda_0": lambda_0,
        "target_cond": args.target_cond,
        "published": {
            "effective_cond_median": float(np.median(cond_pub)),
            "w_eff_per_dim_CV_median": float(np.median(cvp)),
            "frac_dims_CV_above_1": float((cvp > 1).mean()),
        },
        "scan": results,
        "penalty_form": "lambda * ||w_eff||^2 = lambda * alpha^2  (w_eff = alpha*w_dir)",
    }
    (out_dir / "metrics_stageA.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "config" / "run_setup_stageA.json").write_text(json.dumps({
        "argv": vars(args),
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd="/home/bc/VeraRetouch").stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__,
    }, indent=2), encoding="utf-8")
    print(f"\ndone in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
