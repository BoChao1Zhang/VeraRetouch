"""EPR-032 driver: c* calibration and the Panel I / II carrier ladders.

    .venv-lens/bin/python -m q3vl.whatb.jetlut.run probe
    .venv-lens/bin/python -m q3vl.whatb.jetlut.run calib  --out DIR
    .venv-lens/bin/python -m q3vl.whatb.jetlut.run ladder --out DIR --c-star C

Caliber (PROPOSAL §1):
  * fit colours   X_fit  = uniform_grid(33)      (endpoint-inclusive, 35,937)
  * eval colours  X_eval = a fixed random draw from uniform_grid(65) \\ X_fit
  * held-out pool = lut ids with ``lut_id ∩ train = 0``  -> the headline
  * train pool    = a size-matched draw from the train ids -> secondary column
  * the LUT operator is the bank's own (``bank.apply``), byte-for-byte the one
    ``lutcode.lut_residual_matrix`` uses, so the numbers sit next to EPR-031 C0.

Everything is deterministic: the atlas is analytic, the projection is convex,
and the only stochastic element is the pre-registered subsample seed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from q3vl.whatb.codec.lutcode import (
    assert_fit_set_clean, eval_only_lut_ids, open_bank, train_lut_ids,
)
from q3vl.whatb.colorimetry import delta_e00_srgb, srgb_to_lab
from q3vl.whatb.jetlut.core import (
    Atlas, admm_lad, design_matrix, n_dynamic_params,
)
from q3vl.whatb.queries import uniform_grid

#: pre-registered, never tuned
SEED = 20260818
GRID_FIT = 33          # settable by --grid-fit; GRID_EVAL follows as 2*GRID_FIT-1
GRID_EVAL = 65
N_EVAL_POINTS = 32768
#: PROPOSAL §2.2; p1 reaches m=7 (P=4128) which already covers three p2 points
LADDER_P1 = (3, 4, 5, 6, 7)
LADDER_P2 = (3, 4, 5, 6)
#: PROPOSAL §4.3 -- coarse grid first, the PU-GKAN interval logic sets the ends
C_GRID = (0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0)


# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #
def fit_colours(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    return uniform_grid(GRID_FIT, dtype=torch.float32).to(dtype)


def eval_colours(dtype: torch.dtype = torch.float64) -> torch.Tensor:
    """A fixed draw from ``uniform_grid(65)`` minus the 33^3 fit lattice.

    ``65 = 2*33 - 1`` so the fit lattice is exactly the even-index sublattice;
    dropping it is an index test, not a float comparison.
    """
    n = 2 * GRID_FIT - 1
    idx = torch.arange(n ** 3)
    r, rem = idx // (n * n), idx % (n * n)
    g, b = rem // n, rem % n
    on_fit = (r % 2 == 0) & (g % 2 == 0) & (b % 2 == 0)
    keep = idx[~on_fit]
    gen = torch.Generator().manual_seed(SEED)
    pick = keep[torch.randperm(keep.numel(), generator=gen)[:N_EVAL_POINTS]]
    full = uniform_grid(n, dtype=torch.float32)
    return full[pick].to(dtype)


# --------------------------------------------------------------------------- #
# LUT pools and residuals
# --------------------------------------------------------------------------- #
def all_pools(n_train: int | None) -> dict[str, list[str]]:
    """Every pool this EPR can board, by name.

    ``all`` is the whole 4,051-LUT bank; ``train_full`` is the complete 3,149
    train pool (the exact set EPR-031 C0 fitted its PCA basis on, so a 17^3 run
    on this pool sits directly next to the published PCA ladder).
    """
    held, tr, tlu = pools(n_train)
    bank_ids = sorted(open_bank().lut_ids())
    return {"held_out": held, "t_lut_unseen": tlu, "train": tr,
            "train_full": sorted(train_lut_ids()), "all": bank_ids}


def pools(n_train: int | None) -> tuple[list[str], list[str], list[str]]:
    """``(held_out, train_subsample, t_lut_unseen)``.

    Measured 2026-08-18: bank 4,051, train 3,149, **bank - train = 902** with
    ``T_lut_unseen``'s 259 ids fully inside it (intersection with train = 0).
    ``V_what`` (531) and ``T_final`` (577) are sample-level splits whose LUT ids
    lie entirely inside train, so they are not a held-out *style* pool.

    Headline pool is the full 902: it is disjoint from everything c* was
    calibrated on and is the largest such set.  The 259 is carried alongside as
    a sub-column because it is the campaign's registered LUT-disjoint split.
    """
    tr = sorted(train_lut_ids())
    assert_fit_set_clean(tr)
    bank_ids = set(open_bank().lut_ids())
    held = sorted(bank_ids - set(tr))
    tlu = sorted(set(eval_only_lut_ids()["T_lut_unseen"]) & set(held))
    if n_train is not None and n_train < len(tr):
        gen = torch.Generator().manual_seed(SEED)
        sel = torch.randperm(len(tr), generator=gen)[:n_train].tolist()
        tr = sorted(tr[i] for i in sel)
    return held, tr, tlu


def residuals(bank, lut_ids: Sequence[str], x: torch.Tensor,
              device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """``(Q, 3L)`` with column ``3*l + ch``: ``L_l(x) - x`` per channel.

    Streamed one LUT at a time -- the full library at 33^3 in fp64 would be
    terabytes, and the whole point of a shared design matrix is that it never
    has to be resident (PROPOSAL §4.1).
    """
    x32 = x.to(torch.float32).cpu()
    out = torch.empty((x.shape[0], 3 * len(lut_ids)), dtype=dtype, device=device)
    for i, lid in enumerate(lut_ids):
        v = bank.apply(x32, lid) - x32                     # (Q, 3)
        out[:, 3 * i:3 * i + 3] = v.to(device=device, dtype=dtype)
    return out


# --------------------------------------------------------------------------- #
# strata (PROPOSAL §5) -- pre-registered operational definitions
# --------------------------------------------------------------------------- #
def colour_strata(x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Colour-only strata: boundary and saturated-red/yellow and highlight.

    The curvature stratum is LUT-specific and is built separately.
    """
    lab = srgb_to_lab(x.to(torch.float32))
    lstar, a, b = lab.unbind(-1)
    chroma = torch.sqrt(a * a + b * b)
    hue = torch.rad2deg(torch.atan2(b, a))
    return {
        "boundary": (torch.minimum(x, 1.0 - x).amin(-1) < 1.0 / (GRID_FIT - 1) - 1e-9),  # noqa: E501
        "red_yellow": ((hue >= -30.0) & (hue < 90.0) & (chroma > 40.0)),
        "highlight": (lstar > 80.0),
    }


def curvature_mask(r: torch.Tensor, n_grid: int, decile: float = 0.9
                   ) -> torch.Tensor:
    """``(Q,)`` top-decile mask of the Frobenius norm of the residual Hessian.

    Second differences on the ``n_grid^3`` lattice, summed over the three output
    channels and over the 6 unique second-derivative entries.  Frobenius rather
    than spectral: the two orderings agree up to a factor sqrt(3) and the
    spectral norm would need ~10^8 3x3 eigendecompositions per config.  Averaged
    over the LUT batch so the stratum is a property of the *corpus*, fixed
    across arms (an arm-dependent stratum could not be compared).
    """
    q, cols = r.shape
    acc = torch.zeros(n_grid ** 3, device=r.device, dtype=r.dtype)
    n_lut = cols // 3
    chunk = 64
    for s in range(0, n_lut, chunk):                       # mean of |H|^2 over
        e = min(s + chunk, n_lut)                          # LUTs, NOT |H(mean)|^2
        g = r[:, 3 * s:3 * e].reshape(n_grid, n_grid, n_grid, e - s, 3)
        for ax in range(3):
            d2 = torch.zeros_like(g)
            sl = [slice(None)] * 3
            idx_c, idx_p, idx_m = sl.copy(), sl.copy(), sl.copy()
            idx_c[ax] = slice(1, -1)
            idx_p[ax] = slice(2, None)
            idx_m[ax] = slice(0, -2)
            ic, ip, im = tuple(idx_c), tuple(idx_p), tuple(idx_m)
            d2[ic] = g[ip] - 2 * g[ic] + g[im]
            acc += (d2 ** 2).sum(-1).sum(-1).reshape(-1)
        del g
    score = acc / max(n_lut, 1)
    thr = torch.quantile(score.float(), decile).to(score.dtype)
    return score >= thr


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _q(v: torch.Tensor, q: float) -> float:
    """Quantile without ``torch.quantile``'s ~16M element ceiling.

    The pooled tensor here is up to 35,937 x 902 = 32.4M, which trips that
    limit; ``kthvalue`` on the sorted order has no such cap.
    """
    n = v.numel()
    k = min(max(int(round(q * (n - 1))) + 1, 1), n)
    return float(v.reshape(-1).kthvalue(k).values)


def _stats(de: torch.Tensor) -> dict[str, float]:
    """``de`` is (Q, L) -- pooled over both axes, plus the per-LUT-mean view."""
    flat = de.reshape(-1).float()
    per_lut = de.mean(0).float()
    return {
        "mean": float(flat.mean()), "p95": _q(flat, 0.95), "p99": _q(flat, 0.99),
        "max": float(flat.max()),
        "per_lut_mean_median": float(per_lut.median()),
        "per_lut_mean_p95": _q(per_lut, 0.95),
        "per_lut_mean_max": float(per_lut.max()),
        "n_lut": int(de.shape[1]), "n_points": int(de.shape[0]),
    }


def de00_of(x: torch.Tensor, r: torch.Tensor, rhat: torch.Tensor, *,
            clamp: bool, chunk: int = 64) -> torch.Tensor:
    """``(Q, L)`` dE00 between ``x+rhat`` and ``x+r`` (columns are 3L)."""
    q = x.shape[0]
    lut_n = r.shape[1] // 3
    out = torch.empty((q, lut_n), dtype=torch.float32, device=x.device)
    xf = x.to(torch.float32)
    for s in range(0, lut_n, chunk):
        e = min(s + chunk, lut_n)
        y = xf.unsqueeze(0) + r[:, 3 * s:3 * e].T.reshape(e - s, q, 3).to(torch.float32)
        yh = xf.unsqueeze(0) + rhat[:, 3 * s:3 * e].T.reshape(e - s, q, 3).to(torch.float32)
        if clamp:
            yh = yh.clamp(0.0, 1.0)
        out[:, s:e] = delta_e00_srgb(yh, y).T
    return out


# --------------------------------------------------------------------------- #
# one (m, p) configuration
# --------------------------------------------------------------------------- #
def fit_and_score(x_fit, x_eval, r_fit, r_eval, build_fit, build_eval,
                  meta: dict, *, max_iter: int,
                  strata: dict[str, torch.Tensor] | None = None,
                  lam: float = 1e-8, tie_probe: bool = False,
                  return_theta: bool = False) -> dict:
    """Solve one LAD projection on a caller-supplied design and score it.

    Split out of :func:`fit_config` verbatim so the EPR-032 ablation arms
    (``ablation_arms.py``: the AFFONLY gate, the 4-D atlas, the diagonal-p2
    basis) reuse *this* solver, these metrics and this record schema instead of
    growing a second copy.  ``meta`` carries the arm's own descriptors; the
    order of operations below is unchanged, so the ladder numbers are
    bit-identical to the ones already on the board.
    """
    t0 = time.time()
    f_fit = build_fit()
    sol = admm_lad(f_fit, r_fit, lam=lam, max_iter=max_iter)
    t_fit = time.time() - t0
    theta = sol["theta"]
    rec = {**meta,
           "P_feat": int(f_fit.shape[1]),
           "l1_fit": float(sol["primal"].sum()),
           "l1_fit_mean_per_point": float(sol["primal"].sum()
                                          / (r_fit.numel())),
           "admm_iters": int(sol["n_iter"]), "admm_rho": float(sol["rho"]),
           "gap_max": float(sol["gap"].max()), "gap_mean": float(sol["gap"].mean()),
           "dual_feas": float(sol["dual_feas"]),
           "r_pri": sol["r_pri"], "r_dual": sol["r_dual"],
           "fit_seconds": t_fit}

    if tie_probe:                       # PROPOSAL §3.3: tie-break must not bite
        s2 = admm_lad(f_fit, r_fit, lam=lam / 10.0, max_iter=max_iter)
        base = float(s2["primal"].sum())
        rec["tie_gap_rel"] = (rec["l1_fit"] - base) / max(base, 1e-30)

    for clamp in (False, True):
        tag = "post_clamp" if clamp else "pre_clamp"
        de = de00_of(x_fit, r_fit, f_fit @ theta, clamp=clamp)
        rec[f"fitgrid_{tag}"] = _stats(de)
        if strata is not None and not clamp:
            rec["fitgrid_strata"] = {
                k: _stats(de[mask]) for k, mask in strata.items()
                if int(mask.sum()) > 0}
        del de
    yhat = f_fit @ theta                                   # (Q, 3L)
    viol = 0
    n_tot = 0
    for s in range(0, yhat.shape[1], 3 * 64):
        blk = x_fit.unsqueeze(1) + yhat[:, s:s + 3 * 64].reshape(
            x_fit.shape[0], -1, 3)                         # (Q, l, 3)
        viol += int(((blk < 0.0) | (blk > 1.0)).any(-1).sum())
        n_tot += blk.shape[0] * blk.shape[1]
    rec["gamut_violation_rate"] = viol / max(n_tot, 1)
    del f_fit, yhat

    f_eval = build_eval()
    for clamp in (False, True):
        tag = "post_clamp" if clamp else "pre_clamp"
        de = de00_of(x_eval, r_eval, f_eval @ theta, clamp=clamp)
        rec[f"heldgrid_{tag}"] = _stats(de)
        del de
    del f_eval
    torch.cuda.empty_cache()
    if return_theta:
        rec["_theta"] = theta
    return rec


def fit_config(x_fit, x_eval, r_fit, r_eval, m: int, p: int, c: float, *,
               max_iter: int, strata: dict[str, torch.Tensor] | None = None,
               lam: float = 1e-8, tie_probe: bool = False) -> dict:
    atlas = Atlas(m=m, c=c)
    meta = {"m": m, "p": p, "c": c, "N": atlas.n, "h": atlas.h,
            "sigma": atlas.sigma, "P_dyn": n_dynamic_params(m, p)}
    return fit_and_score(
        x_fit, x_eval, r_fit, r_eval,
        lambda: design_matrix(x_fit, atlas, p),
        lambda: design_matrix(x_eval, atlas, p),
        meta, max_iter=max_iter, strata=strata, lam=lam, tie_probe=tie_probe)


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _setup(args):
    dev = torch.device(args.device)
    dt = torch.float64
    bank = open_bank()
    held, tr, tlu = pools(args.n_train)
    return dev, dt, bank, held, tr, tlu


def cmd_probe(args) -> None:
    dev, dt, bank, held, tr, tlu = _setup(args)
    x = fit_colours(dt).to(dev)
    xe = eval_colours(dt).to(dev)
    ids = held[:args.n_probe]
    t0 = time.time()
    r = residuals(bank, ids, x, dev, dt)
    re = residuals(bank, ids, xe, dev, dt)
    print(f"residuals for {len(ids)} luts: {time.time()-t0:.1f}s  "
          f"R {tuple(r.shape)} {r.element_size()*r.nelement()/2**30:.2f} GiB")
    for m, p in ((4, 1), (4, 2), (6, 2), (7, 1)):
        rec = fit_config(x, xe, r, re, m, p, float(args.c_star or 1.0),
                         max_iter=args.max_iter)
        print(f"m={m} p={p} P={rec['P_dyn']:5d} iters={rec['admm_iters']:5d} "
              f"{rec['fit_seconds']:7.1f}s  gap={rec['gap_max']:.2e}  "
              f"fit dE00 mean/p95={rec['fitgrid_pre_clamp']['mean']:.4f}/"
              f"{rec['fitgrid_pre_clamp']['p95']:.4f}  "
              f"held {rec['heldgrid_pre_clamp']['mean']:.4f}/"
              f"{rec['heldgrid_pre_clamp']['p95']:.4f}")


def cmd_calib(args) -> None:
    """c* on the TRAIN pool only, using p=1 only (PROPOSAL §4.3, §7 S1)."""
    dev, dt, bank, held, tr, tlu = _setup(args)
    ids = tr[:args.n_calib]
    x = uniform_grid(args.calib_grid, dtype=torch.float32).to(device=dev, dtype=dt)
    xe = eval_colours(dt).to(dev)
    r = residuals(bank, ids, x, dev, dt)
    re = residuals(bank, ids, xe, dev, dt)
    rows = []
    for m in args.calib_m:
        for c in C_GRID:
            rec = fit_config(x, xe, r, re, m, 1, c, max_iter=args.max_iter)
            rows.append(rec)
            print(f"m={m} c={c:.2f}  l1/pt={rec['l1_fit_mean_per_point']:.6f}  "
                  f"dE00 mean={rec['fitgrid_pre_clamp']['mean']:.4f}  "
                  f"gap={rec['gap_max']:.1e}", flush=True)
    best = {m: min((r_ for r_ in rows if r_["m"] == m),
                   key=lambda z: z["l1_fit_mean_per_point"])["c"]
            for m in args.calib_m}
    out = {"seed": SEED, "n_calib": len(ids), "calib_grid": args.calib_grid,
           "calib_lut_ids_sha": _sha(ids), "c_grid": list(C_GRID),
           "best_c_per_m": best, "rows": rows}
    _dump(args.out, "calib.json", out)
    print("best c per m:", best)


SYNTH = ("constant", "affine", "quadratic", "smooth_hi_curv",
         "sharp_continuous", "posterize")


def synth_residual(name: str, x):
    """Deterministic synthetic targets for the convergence-order check (§6.1).

    Returned as a residual ``F(x) - x`` so the carrier's identity anchor is not
    doing the work.  ``posterize`` is discontinuous on purpose: the O(h^{p+1})
    argument needs C^{p+1} smoothness, so this row is where it must fail.
    """
    r, g, b = x.unbind(-1)
    if name == "constant":
        return torch.stack([0.05 * torch.ones_like(r)] * 3, -1)
    if name == "affine":
        return torch.stack([0.10 * r - 0.05 * g, 0.08 * g + 0.02 * b,
                            -0.04 * r + 0.06 * b], -1)
    if name == "quadratic":
        return torch.stack([0.30 * r * r - 0.10 * g * b, 0.25 * g * g,
                            0.20 * b * b + 0.05 * r * g], -1)
    if name == "smooth_hi_curv":
        return torch.stack([0.20 * torch.sin(6.0 * r) * torch.cos(4.0 * g),
                            0.15 * torch.exp(-6.0 * ((b - 0.5) ** 2)),
                            0.18 * torch.sin(5.0 * (r + b))], -1)
    if name == "sharp_continuous":
        return torch.stack([0.25 * torch.tanh(20.0 * (r - 0.5)),
                            0.20 * torch.tanh(20.0 * (g - 0.35)),
                            0.15 * torch.tanh(20.0 * (b - 0.65))], -1)
    if name == "posterize":
        return torch.stack([torch.floor(r * 4.0) / 4.0 - r,
                            torch.floor(g * 4.0) / 4.0 - g,
                            torch.floor(b * 4.0) / 4.0 - b], -1)
    raise ValueError(name)


def cmd_synth(args) -> None:
    """§6.1: log-log convergence slope of the fit error against fill distance.

    Six targets are fitted simultaneously as 18 right-hand-side columns on one
    shared design matrix; the per-target L1 is read off ``primal`` in blocks of
    three, so the slope is per target rather than pooled.
    """
    dev, dt = torch.device(args.device), torch.float64
    x = fit_colours(dt).to(dev)
    r = torch.cat([synth_residual(n, x) for n in SYNTH], dim=-1)
    n_pt = x.shape[0]
    rows = []
    for p in args.p:
        for m in args.m:
            atlas = Atlas(m=m, c=args.synth_c)
            t0 = time.time()
            f = design_matrix(x, atlas, p)
            sol = admm_lad(f, r, max_iter=args.max_iter)
            prim = sol["primal"]
            per = {n: float(prim[3 * i:3 * i + 3].sum()) / (3 * n_pt)
                   for i, n in enumerate(SYNTH)}
            de = de00_of(x, r, f @ sol["theta"], clamp=False)
            rows.append({
                "p": p, "m": m, "N": atlas.n, "h": atlas.h,
                "sigma": atlas.sigma, "P_dyn": n_dynamic_params(m, p),
                "l1_per_point": per,
                "de00_mean": {n: float(de[:, i].mean()) for i, n in enumerate(SYNTH)},
                "gap_max": float(sol["gap"].max()),
                "admm_iters": int(sol["n_iter"]),
                "seconds": time.time() - t0,
            })
            print(f"[synth] p={p} m={m} h={atlas.h:.5f} P={rows[-1]['P_dyn']:5d} "
                  + "  ".join(f"{n}={per[n]:.3e}" for n in SYNTH), flush=True)
            _dump(args.out, f"synth{args.tag}.json",
                  {"seed": SEED, "grid_fit": GRID_FIT, "c": args.synth_c,
                   "targets": list(SYNTH), "rows": rows})
            del f
            torch.cuda.empty_cache()
    _dump(args.out, f"synth{args.tag}.DONE.json", {"n_rows": len(rows)})


def quantize(theta, bits: int):
    """Per-coefficient-dimension symmetric uniform quantization.

    The scale is one number per basis dimension shared by the whole library, so
    it amortises to ~0 bits/LUT; the per-LUT payload is exactly ``P_dyn * bits``.
    """
    if bits >= 32:
        return theta
    scale = theta.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    qmax = 2 ** (bits - 1) - 1
    return torch.round(theta / scale * qmax).clamp(-qmax - 1, qmax) * scale / qmax


def cmd_rate(args) -> None:
    """Panel II rate axis: dE00 vs bits/LUT at fp32 / fp16 / int8 / int4."""
    dev, dt, bank, held, tr, tlu = _setup(args)
    ids = all_pools(args.n_train)[args.pool[0]]
    x = fit_colours(dt).to(dev)
    r = residuals(bank, ids, x, dev, dt)
    rows = []
    for p, ladder in ((1, LADDER_P1), (2, LADDER_P2)):
        if p not in args.p:
            continue
        for m in ladder:
            if m not in args.m:
                continue
            atlas = Atlas(m=m, c=args.c_star_num)
            f = design_matrix(x, atlas, p)
            sol = admm_lad(f, r, max_iter=args.max_iter)
            pdyn = n_dynamic_params(m, p)
            rec = {"p": p, "m": m, "P_dyn": pdyn, "c": args.c_star_num, "bits": {}}
            for b in (32, 16, 8, 4):
                de = de00_of(x, r, f @ quantize(sol["theta"], b), clamp=False)
                st = _stats(de)
                rec["bits"][str(b)] = {"bits_per_lut": pdyn * b,
                                       "mean": st["mean"], "p95": st["p95"],
                                       "p99": st["p99"], "max": st["max"]}
                del de
            rows.append(rec)
            print("[rate] p=%d m=%d P=%5d  " % (p, m, pdyn) + "  ".join(
                "b%d:%d bits mean=%.4f" % (b, rec["bits"][str(b)]["bits_per_lut"],
                                           rec["bits"][str(b)]["mean"])
                for b in (32, 16, 8, 4)), flush=True)
            _dump(args.out, f"rate_{args.pool[0]}{args.tag}.json",
                  {"seed": SEED, "pool": args.pool[0], "n_lut": len(ids),
                   "lut_ids_sha": _sha(ids), "grid_fit": GRID_FIT, "rows": rows})
            del f
            torch.cuda.empty_cache()
    _dump(args.out, f"rate_{args.pool[0]}{args.tag}.DONE.json", {"n_rows": len(rows)})


def cmd_sweep(args) -> None:
    """Bandwidth ablation on the real pool at the real ladder configs.

    Two things at once (PROPOSAL §4.3 appendix + Panel III):
      * family-optimal ``c_p*``: c* was selected on p=1 only and shared, which
        is the conservative choice -- this measures what that choice costs p=2;
      * ``--sigma-fixed 0.15`` sets ``c = sigma / h_m`` per m, i.e. GLUT's own
        fixed bandwidth carried onto the exact-POU gate (arm A2b).
    """
    dev, dt, bank, held, tr, tlu = _setup(args)
    ids = all_pools(args.n_train)[args.pool[0]]
    x = fit_colours(dt).to(dev)
    xe = eval_colours(dt).to(dev)
    r = residuals(bank, ids, x, dev, dt)
    re = residuals(bank, ids, xe, dev, dt)
    rows = []
    for p in args.p:
        for m in args.m:
            cs = ([args.sigma_fixed / Atlas(m=m, c=1.0).h] if args.sigma_fixed
                  else list(args.c_grid))
            for c in cs:
                rec = fit_config(x, xe, r, re, m, p, c, max_iter=args.max_iter)
                rec["pool"] = args.pool[0]
                rec["sigma_fixed"] = args.sigma_fixed
                rows.append(rec)
                print(f"[sweep] p={p} m={m} c={c:.4f} sigma={rec['sigma']:.4f} "
                      f"P={rec['P_dyn']:5d} l1/pt={rec['l1_fit_mean_per_point']:.6f} "
                      f"pre mean/p95={rec['fitgrid_pre_clamp']['mean']:.4f}/"
                      f"{rec['fitgrid_pre_clamp']['p95']:.4f}", flush=True)
                _dump(args.out, f"sweep_{args.pool[0]}{args.tag}.json",
                      {"seed": SEED, "pool": args.pool[0], "n_lut": len(ids),
                       "lut_ids_sha": _sha(ids), "grid_fit": GRID_FIT,
                       "sigma_fixed": args.sigma_fixed, "rows": rows})
    _dump(args.out, f"sweep_{args.pool[0]}{args.tag}.DONE.json",
          {"pool": args.pool[0], "n_rows": len(rows)})


def cmd_ladder(args) -> None:
    dev, dt, bank, held, tr, tlu = _setup(args)
    c_star = resolve_c_star(args.c_star, args.out)
    print(f'c_star = {c_star}', flush=True)
    x = fit_colours(dt).to(dev)
    xe = eval_colours(dt).to(dev)
    strata = {k: v.to(dev) for k, v in colour_strata(x).items()}
    registry = all_pools(args.n_train)
    for pool_name in args.pool:
        ids = registry[pool_name]
        r = residuals(bank, ids, x, dev, dt)
        re = residuals(bank, ids, xe, dev, dt)
        st = dict(strata)
        st["high_curvature"] = curvature_mask(r, GRID_FIT)
        st["rest"] = ~(st["high_curvature"] | st["boundary"]
                       | st["red_yellow"] | st["highlight"])
        rows = []
        for p, ladder in ((1, LADDER_P1), (2, LADDER_P2)):
            for m in ladder:
                rec = fit_config(x, xe, r, re, m, p, c_star,
                                 max_iter=args.max_iter, strata=st,
                                 tie_probe=(m == 4))
                rec["pool"] = pool_name
                rows.append(rec)
                print(f"[{pool_name}] m={m} p={p} P={rec['P_dyn']:5d} "
                      f"{rec['fit_seconds']:7.1f}s gap={rec['gap_max']:.1e} "
                      f"pre mean/p95/p99={rec['fitgrid_pre_clamp']['mean']:.4f}/"
                      f"{rec['fitgrid_pre_clamp']['p95']:.4f}/"
                      f"{rec['fitgrid_pre_clamp']['p99']:.4f}", flush=True)
                _dump(args.out, f"ladder_{pool_name}{args.tag}.json",
                      {"seed": SEED, "c_star": c_star, "pool": pool_name,
                       "n_lut": len(ids), "lut_ids_sha": _sha(ids),
                       "grid_fit": GRID_FIT, "grid_eval": GRID_EVAL,
                       "n_eval_points": N_EVAL_POINTS,
                       "strata_sizes": {k: int(v.sum()) for k, v in st.items()},
                       "rows": rows})
        _dump(args.out, f"ladder_{pool_name}{args.tag}.DONE.json",
              {"pool": pool_name, "n_rows": len(rows), "n_lut": len(ids),
               "grid_fit": GRID_FIT, "c_star": c_star})
        del r, re
        torch.cuda.empty_cache()


def resolve_c_star(spec: str, out: str) -> float:
    """``--c-star auto`` reads ``calib.json`` and picks the shared c.

    One c for every m (PROPOSAL §2, §4.3), chosen by normalising each m's L1
    curve by its own minimum and averaging -- so a resolution whose absolute
    error is larger does not dominate the pick.  Selected on p=1 only; A4 is
    never allowed to retune it.
    """
    if spec != "auto":
        return float(spec)
    rows = json.loads((Path(out) / "calib.json").read_text())["rows"]
    by_m: dict[int, dict[float, float]] = {}
    for r in rows:
        by_m.setdefault(r["m"], {})[r["c"]] = r["l1_fit_mean_per_point"]
    score: dict[float, float] = {}
    for curve in by_m.values():
        lo = min(curve.values())
        for c, v in curve.items():
            score[c] = score.get(c, 0.0) + v / lo
    return min(score, key=lambda c: score[c])


def _sha(ids: Sequence[str]) -> str:
    import hashlib
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def _dump(out: str, name: str, obj) -> None:
    d = Path(out)
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(obj, indent=1, sort_keys=True))


def main() -> None:
    # imported here, not at module scope: ``ablation_arms`` imports this module
    # back (it reuses ``fit_and_score`` / ``all_pools`` / the metrics).
    from q3vl.whatb.jetlut import ablation_arms as A

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("probe", "calib", "ladder", "sweep",
                                   "synth", "rate", "jgate", "j4d", "jp15"))
    ap.add_argument("--out", default="experiments/prs/EPR-032_jetlut-canonical-color-field/out")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--c-star", default="auto",
                    help="a float, or 'auto' to read calib.json in --out")
    ap.add_argument("--max-iter", type=int, default=1500)
    ap.add_argument("--n-train", type=int, default=902)
    ap.add_argument("--n-calib", type=int, default=256)
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--calib-grid", type=int, default=17)
    ap.add_argument("--calib-m", type=int, nargs="+", default=[3, 4, 5])
    ap.add_argument("--pool", nargs="+", default=["held_out"],
                    choices=("held_out", "t_lut_unseen", "train",
                             "train_full", "all"))
    ap.add_argument("--grid-fit", type=int, default=GRID_FIT)
    ap.add_argument("--p", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--m", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--c-grid", type=float, nargs="+", default=list(C_GRID))
    ap.add_argument("--synth-c", type=float, default=0.7)
    ap.add_argument("--c-star-num", type=float, default=0.7)
    ap.add_argument("--sigma-fixed", type=float, default=0.0,
                    help="if set, c = sigma/h_m per m (GLUT's fixed bandwidth)")
    ap.add_argument("--tag", default="",
                    help="suffix on the artefact name, e.g. '_g17'")
    A.add_arguments(ap)          # EPR-032 ablation arms (jgate / j4d / jp15)
    args = ap.parse_args()
    globals()["GRID_FIT"] = int(args.grid_fit)
    globals()["GRID_EVAL"] = 2 * int(args.grid_fit) - 1
    torch.manual_seed(SEED)
    {"probe": cmd_probe, "calib": cmd_calib, "ladder": cmd_ladder,
     "sweep": cmd_sweep, "synth": cmd_synth,
     "rate": cmd_rate, **A.COMMANDS}[args.cmd](args)


if __name__ == "__main__":
    main()
