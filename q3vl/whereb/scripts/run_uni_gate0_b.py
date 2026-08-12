"""Proposal B (FAFM) -- Gate 0.  Two zero-training checks, before any training.

``RESEARCH_unified-field-prediction_2026-08-10`` section 3.6:

* **G0a linearity check** -- proposal B's *only* unhedged assumption (H1) is that
  the frozen image-guided upsample ``U_I`` is linear **in the signal**.  Every
  closed form downstream rests on it: the canonical coarse coordinate
  ``c* = (A^T A + eps I)^-1 A^T y``, the fixed quadratic metric ``Lambda``, the
  spectral bound on the condition number.  Test: draw ``c1, c2``, report the
  relative superposition residual ``||U(c1+c2) - U(c1) - U(c2)|| / scale``.
  Pre-registered pass line: median <= 1%.  On failure, a fixed guided
  linearisation ``A_I`` must be constructed and everything re-measured.

* **G0b neck reconstruction upper bound** -- how much of the GT field survives a
  round trip through the neck?  ``y -> c* -> clip(A_I c*, 0, 1)``.
  Pre-registered pass line: overall median soft-IoU >= 0.93 **and** per-family
  (including the ``semantic`` contour family) >= 0.88.  On failure: re-measure at
  a finer coarse grid (x1.5, x2); still failing falsifies the neck hypothesis and
  sends proposal B back to full-resolution field diffusion.

Both are properties of the frozen operator and the data, so there is no model
forward, no training and no checkpoint anywhere in this card.

Criteria discipline (CLAUDE.md 2026-08-05): no AUC in any form; soft/hard-IoU at
matched-GT-area top-k, grid boundary F1, the zero-parameter centre-prior column
on the same support and the same top-k rule, the ``a/(2-a)`` random floor, and
area strata.  ``c*``'s realised domain is declared arm-wide per the s-cache
consumption contract.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

EPS_SWEEP = (1e-4, 1e-3, 1e-2)
EPS_PRIMARY = 1e-3          # pre-registered; project precedent (run_amort_e5)

GATE_G0A_MAX_RESIDUAL = 0.01
GATE_G0B_OVERALL = 0.93
GATE_G0B_FAMILY = 0.88


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                        # noqa: BLE001
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--n-g0a", type=int, default=256)
    ap.add_argument("--n-g0b", type=int, default=1000)
    ap.add_argument("--grid-mul", type=float, default=1.0,
                    help="coarse-grid escalation: 1.0 = the native F_pre H/16 "
                         "grid, 1.5 / 2.0 are the pre-registered retries")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--loaders", type=int, default=12)
    ap.add_argument("--n-viz", type=int, default=6)
    args = ap.parse_args(argv)

    t0 = time.time()
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.unifield import (
        FAMILIES, GuidedOp, agg, by_group, field_row, gram_exact, guide_of,
        load_families, median,
    )

    out = Path(args.out)
    for sub in ("viz", "config", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    dev = torch.device(args.device)
    dt = getattr(torch, args.dtype)
    rng = np.random.default_rng(args.seed)

    # ---------------- population ------------------------------------------
    ds, ds_facts = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    print(f"{args.split}: {len(local)} local of {len(rows)}  ({time.time()-t0:.0f}s)",
          flush=True)

    n_need = max(args.n_g0a, args.n_g0b)
    pick = sorted(rng.choice(len(local), size=min(n_need, len(local)),
                             replace=False).tolist())
    idxs = [local[i] for i in pick]
    fams = load_families(ds, idxs)
    print(f"families resolved for {len(fams)}  ({time.time()-t0:.0f}s)", flush=True)

    def load(i: int) -> dict[str, Any] | None:
        try:
            s = ds[i]
            return {
                "idx": i, "sample_id": s.sample_id,
                "guide": guide_of(s), "gt": s.mask_target_hi().double(),
                "grid": (int(s.grid_h), int(s.grid_w)),
            }
        except Exception as exc:                             # noqa: BLE001
            print(f"  load fail {i}: {type(exc).__name__}: {exc}", flush=True)
            return None

    # ---------------- G0a: linearity of U_I -------------------------------
    print(f"G0a: superposition on {args.n_g0a} images", flush=True)
    g0a_rows: list[dict[str, Any]] = []
    adjoint_errs: list[float] = []
    gen = torch.Generator().manual_seed(args.seed)
    with ThreadPoolExecutor(max_workers=args.loaders) as ex:
        for pk in ex.map(load, idxs[: args.n_g0a]):
            if pk is None:
                continue
            gh, gw = pk["grid"]
            gh = max(2, int(round(gh * args.grid_mul)))
            gw = max(2, int(round(gw * args.grid_mul)))
            op = GuidedOp(pk["guide"].to(dev), gh, gw, dtype=dt)
            n = op.n_low
            # two draws: the [0,1] range an actual coarse mask lives in, and a
            # zero-mean gaussian (which probes cancellation, the harder case)
            c1u = torch.rand(n, generator=gen, dtype=dt).to(dev)
            c2u = torch.rand(n, generator=gen, dtype=dt).to(dev)
            c1g = torch.randn(n, generator=gen, dtype=dt).to(dev)
            c2g = torch.randn(n, generator=gen, dtype=dt).to(dev)
            ru = op.superposition_residual(c1u, c2u)
            rg = op.superposition_residual(c1g, c2g)
            adjoint_errs.append(op.adjoint_check())
            g0a_rows.append({
                "sample_id": pk["sample_id"], "grid": [gh, gw],
                "uniform_rel_sum": ru["rel_sum"], "uniform_rel_parts": ru["rel_parts"],
                "gauss_rel_sum": rg["rel_sum"], "gauss_rel_parts": rg["rel_parts"],
            })
            del op
    torch.cuda.empty_cache()
    g0a = {
        "n": len(g0a_rows),
        "uniform_rel_sum": agg(r["uniform_rel_sum"] for r in g0a_rows),
        "uniform_rel_parts": agg(r["uniform_rel_parts"] for r in g0a_rows),
        "gauss_rel_sum": agg(r["gauss_rel_sum"] for r in g0a_rows),
        "gauss_rel_parts": agg(r["gauss_rel_parts"] for r in g0a_rows),
        "adjoint_identity_err": agg(adjoint_errs),
    }
    worst = max(g0a[k]["median"] for k in
                ("uniform_rel_sum", "uniform_rel_parts", "gauss_rel_sum",
                 "gauss_rel_parts"))
    g0a["worst_median_residual"] = worst
    g0a["gate"] = "pass" if worst <= GATE_G0A_MAX_RESIDUAL else "fail"
    g0a["gate_line"] = f"median relative superposition residual <= {GATE_G0A_MAX_RESIDUAL}"
    print(f"  G0a {g0a['gate'].upper()}  worst median residual {worst:.3e}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---------------- G0b: neck reconstruction upper bound ----------------
    print(f"G0b: neck reconstruction on {args.n_g0b} images", flush=True)
    rows_out: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    cstar_cells: list[np.ndarray] = []
    viz_pool: list[dict[str, Any]] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.loaders) as ex:
        for pk in ex.map(load, idxs[: args.n_g0b]):
            if pk is None:
                continue
            gh0, gw0 = pk["grid"]
            gh = max(2, int(round(gh0 * args.grid_mul)))
            gw = max(2, int(round(gw0 * args.grid_mul)))
            gt = pk["gt"]
            op = GuidedOp(pk["guide"].to(dev), gh, gw, dtype=dt)
            gt_d = gt.to(dev).to(dt)

            G = gram_exact(op).double()
            ev = torch.linalg.eigvalsh(0.5 * (G + G.T))
            emin, emax = float(ev.min()), float(ev.max())
            b = op.adjoint(gt_d.reshape(-1)).reshape(-1).double()
            eye = torch.eye(op.n_low, dtype=torch.float64, device=G.device)

            row: dict[str, Any] = {
                "sample_id": pk["sample_id"], "family": fams.get(pk["sample_id"], "unknown"),
                "grid": [gh, gw], "out_hw": [int(gt.shape[-2]), int(gt.shape[-1])],
            }
            best_c = None
            for eps in EPS_SWEEP:
                c = torch.linalg.solve(G + eps * eye, b)
                pred = op.render(c.to(dt)).cpu()
                tag = f"eps{eps:g}__"
                fr = field_row(pred, gt, gh0, gw0, prefix=tag)
                if eps == EPS_PRIMARY:
                    row.update(fr)
                    best_c = c
                    cstar_cells.append(c.cpu().numpy())
                    row["cstar_min"] = float(c.min())
                    row["cstar_max"] = float(c.max())
                    viz_pool.append({
                        "sample_id": pk["sample_id"], "family": row["family"],
                        "softiou_hi": fr[f"{tag}softiou_hi"],
                        "pred": pred.float().numpy(), "gt": gt.float().numpy(),
                        "cstar": c.reshape(gh, gw).cpu().float().numpy(),
                    })
                else:
                    row[f"{tag}softiou_hi"] = fr[f"{tag}softiou_hi"]
                    row[f"{tag}softiou"] = fr[f"{tag}softiou"]
                row[f"cond_eps{eps:g}"] = (emax + eps) / (emin + eps)
            row["eig_min"] = emin
            row["eig_max"] = emax
            row["cond_raw"] = emax / max(emin, 1e-300)
            spectra.append({"sample_id": pk["sample_id"], "eig_min": emin,
                            "eig_max": emax})
            rows_out.append(row)
            del op, G, eye
            done += 1
            if done % 50 == 0:
                torch.cuda.empty_cache()
                print(f"  [{done}/{args.n_g0b}] {time.time()-t0:.0f}s", flush=True)
    torch.cuda.empty_cache()

    p = f"eps{EPS_PRIMARY:g}__"
    overall = agg(r[f"{p}softiou_hi"] for r in rows_out)
    per_family = by_group(rows_out, "family", f"{p}softiou_hi")
    g0b: dict[str, Any] = {
        "n": len(rows_out),
        "eps_primary": EPS_PRIMARY,
        "eps_sweep": {f"{e:g}": agg(r[f"eps{e:g}__softiou_hi"] for r in rows_out)
                      for e in EPS_SWEEP},
        "softiou_hi_overall": overall,
        "softiou_hi_by_family": per_family,
        "grid_softiou_overall": agg(r[f"{p}softiou"] for r in rows_out),
        "grid_softiou_by_family": by_group(rows_out, "family", f"{p}softiou"),
        "grid_hard_iou_overall": agg(r[f"{p}hard_iou"] for r in rows_out),
        "grid_boundary_f1_overall": agg(r[f"{p}gbf1"] for r in rows_out),
        "centre_prior_softiou": agg(r["centre_prior_softiou"] for r in rows_out),
        "centre_prior_gbf1": agg(r["centre_prior_gbf1"] for r in rows_out),
        "random_floor": agg(r["random_floor"] for r in rows_out),
        "by_area_stratum": by_group(rows_out, "area_stratum", f"{p}softiou_hi"),
        "spectrum": {
            "eig_min": agg(s["eig_min"] for s in spectra),
            "eig_max": agg(s["eig_max"] for s in spectra),
            "cond_raw": agg(r["cond_raw"] for r in rows_out),
            **{f"cond_eps{e:g}": agg(r[f"cond_eps{e:g}"] for r in rows_out)
               for e in EPS_SWEEP},
        },
    }
    # s-cache consumption contract: declare c*'s arm-wide realised domain
    allc = np.concatenate(cstar_cells) if cstar_cells else np.zeros(1)
    g0b["cstar_domain"] = {
        "arm": "uni_gate0_B_cstar",
        "eps": EPS_PRIMARY,
        "domain": [float(allc.min()), float(allc.max())],
        "p01": float(np.percentile(allc, 1)), "p50": float(np.percentile(allc, 50)),
        "p99": float(np.percentile(allc, 99)),
        "frac_below_0": float((allc < 0).mean()),
        "frac_above_1": float((allc > 1).mean()),
        "n_cells": int(allc.size),
        "advice": "c* is NOT confined to [0,1]; a consumer that clamps it to "
                  "[0,1] silently destroys the ridge solution. Render as "
                  "clip(A_I c, 0, 1) -- clip the FIELD, never the coordinate.",
    }

    fam_ok = {f: (per_family.get(f, {}).get("median") or 0.0) >= GATE_G0B_FAMILY
              for f in FAMILIES if f in per_family}
    overall_ok = (overall.get("median") or 0.0) >= GATE_G0B_OVERALL
    g0b["gate"] = "pass" if (overall_ok and all(fam_ok.values())) else "fail"
    g0b["gate_line"] = (f"overall median soft-IoU >= {GATE_G0B_OVERALL} and every "
                        f"family >= {GATE_G0B_FAMILY}")
    g0b["gate_detail"] = {"overall_pass": overall_ok, "per_family_pass": fam_ok}
    print(f"  G0b {g0b['gate'].upper()}  overall median "
          f"{overall.get('median')}  per-family {fam_ok}  ({time.time()-t0:.0f}s)",
          flush=True)

    # ---------------- viz --------------------------------------------------
    _write_viz(out / "viz", viz_pool, args.n_viz)

    metrics = {
        "card": "uni_gate0_B_fafm",
        "proposal": "B (FAFM, field-anchored flow matching)",
        "doc": "docs/RESEARCH_unified-field-prediction_2026-08-10.md section 3.6",
        "split": args.split, "n_local_population": len(local),
        "seed": args.seed, "grid_mul": args.grid_mul,
        "dtype": args.dtype, "device": args.device,
        "G0a": g0a, "G0b": g0b,
        "verdict": "pass" if g0a["gate"] == "pass" and g0b["gate"] == "pass" else "fail",
        "dataset_facts": ds_facts,
        "git_commit": git_commit(), "python": platform.python_version(),
        "torch": torch.__version__,
        "elapsed_s": time.time() - t0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    (out / "config" / "rows_g0b.json").write_text(json.dumps(rows_out, indent=1,
                                                             default=str))
    (out / "config" / "rows_g0a.json").write_text(json.dumps(g0a_rows, indent=1,
                                                             default=str))
    (out / "config" / "sample_ids.json").write_text(json.dumps(
        [r["sample_id"] for r in rows_out], indent=1))
    print(f"wrote {out/'metrics.json'}  ({time.time()-t0:.0f}s)", flush=True)
    return 0


def _write_viz(viz: Path, pool: list[dict[str, Any]], n: int) -> None:
    """success_* / failure_* panels.

    Colour discipline (CLAUDE.md 2026-08-05): masks are drawn on a **fixed**
    [0,1] scale -- never per-image min-max, whose denominator is set by whatever
    the extreme cell happens to be.  ``c*`` is not a mask and has its own domain,
    so it gets a fixed symmetric scale shared by every panel, printed in the
    title, with the [0,1] band marked in the colourbar.
    """
    if not pool:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lim = float(np.percentile(np.abs(np.concatenate(
        [p["cstar"].reshape(-1) for p in pool])), 99.5))
    order = sorted(pool, key=lambda r: r["softiou_hi"])
    picks = [("failure", r) for r in order[:n]] + [("success", r) for r in order[-n:]]
    for tag, r in picks:
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
        ax[0].imshow(r["gt"], cmap="magma", vmin=0.0, vmax=1.0)
        ax[0].set_title("GT field y  (fixed 0..1)", fontsize=9)
        ax[1].imshow(r["pred"], cmap="magma", vmin=0.0, vmax=1.0)
        ax[1].set_title("clip(A_I c*,0,1)  (fixed 0..1)", fontsize=9)
        d = ax[2].imshow(r["pred"] - r["gt"], cmap="coolwarm", vmin=-1.0, vmax=1.0)
        ax[2].set_title("pred - GT  (fixed -1..1)", fontsize=9)
        plt.colorbar(d, ax=ax[2], fraction=0.046)
        cm = ax[3].imshow(r["cstar"], cmap="coolwarm", vmin=-lim, vmax=lim,
                          interpolation="nearest")
        ax[3].set_title(f"c* coarse  (fixed +-{lim:.2f}, arm-wide)", fontsize=9)
        plt.colorbar(cm, ax=ax[3], fraction=0.046)
        for a in ax:
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(f"{tag}  {r['sample_id']}  family={r['family']}  "
                     f"soft-IoU(hi)={r['softiou_hi']:.4f}", fontsize=10)
        fig.tight_layout()
        fig.savefig(viz / f"{tag}_{r['family']}_{r['sample_id'][:24]}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
