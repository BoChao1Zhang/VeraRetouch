"""E5 -- the zero-training baseline: how good is ``w_0`` from the prior field alone?

P0-E5 asks one question with a two-sided gate: project the coarse prior field
onto the Phi-71 basis with **no GT anywhere in the fit**, decode it through the
frozen ``D``, and measure the decoded IoU against GT.

    decoded IoU >= 0.70  ->  this is a REFINEMENT problem   (P2 weight up)
    decoded IoU <= 0.40  ->  this is a FROM-SCRATCH problem (P1/P3 main line)

The prior field is P-W5's similarity channel: ``merger_out . E[subject noun]`` on
the H/32 merged grid, in both the ``dot`` and ``cosine`` readings P-W5 reported
side by side.  P-W5 measured that field directly (soft-IoU 0.341, below the
zero-parameter centre prior's 0.459); this card measures what survives a trip
through ``D``, which is the quantity P1/P2/P3 actually inherit.

Leakage discipline (the whole point of the card)
-----------------------------------------------
The fit target is the **prior field**, never the GT.  GT enters only as a metric
afterwards.  That keeps IoU off the optimisation path entirely, which the project
red line requires for trained parameters and which is doubly important here: an
oracle fit against GT is exactly the 0.97 ceiling this card is *not* allowed to
touch.

Normalisation discipline (s-cache consumption contract, 2026-08-03)
-------------------------------------------------------------------
The similarity field has to be squashed into [0,1] to be a fit target.  The
contract forbids per-image min-max / softmax normalisation, so the squash uses
**arm-wide constants** computed in a first pass over all samples (robust median
and MAD over valid cells of the whole arm), and the realised domain is asserted
and reported.  Per-image normalisation would have made every field look equally
confident and destroyed exactly the cross-sample variation the card measures.

Two projections are reported, in this order
-------------------------------------------
1. ``full_fit`` -- ``fit_latent(phi, target=prior)``: multi-start L-BFGS over
   ``(w_dir, w0, alpha, rho)`` against the prior.  This is the **upper bound** on
   what any projection of this prior can achieve, so a failure here is the
   strongest available form of the result.
2. ``ridge_lsq`` -- the closed form ``w = (A^T A + eps I)^-1 A^T z`` on
   ``z = logit(prior)`` with an arm-constant ``rho``.  This is literally the
   layer P3 embeds, so its gap to ``full_fit`` is the price P3 pays.

Arms: ``dot``, ``cosine``, and ``centre_prior`` -- the last one because P-W5
found the zero-parameter centre prior *beats* the similarity field, which makes
its projection the honest best zero-training ``w_0`` candidate for P2.

Criteria follow the revised discipline (E1 review R5 / DELTA section 7-4): grid
soft-IoU, grid boundary F1, the centre-prior column, the ``a/(2-a)`` random
floor, and area strata.  No AUC anywhere.
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

EPS = 1e-6


def agg(x: Any) -> dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "min": float(a.min()), "max": float(a.max())}


def gaussian_soften(field: torch.Tensor, sigma: float) -> torch.Tensor:
    """SAMRefiner's lesson: a hard/raw prior reads badly, a softened one reads well.

    Separable Gaussian with reflect padding so the frame edge is not darkened
    (a zero-padded blur would manufacture a vignette and hand the fit a centre
    prior for free -- precisely the confound this stage is policed for).
    """
    if sigma <= 0:
        return field
    rad = max(1, int(round(3 * sigma)))
    t = torch.arange(-rad, rad + 1, dtype=field.dtype, device=field.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    x = field.reshape(1, 1, *field.shape[-2:])
    x = torch.nn.functional.pad(x, (rad, rad, 0, 0), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.reshape(1, 1, 1, -1))
    x = torch.nn.functional.pad(x, (0, 0, rad, rad), mode="reflect")
    x = torch.nn.functional.conv2d(x, k.reshape(1, 1, -1, 1))
    return x.reshape(field.shape)


def load_embeddings(checkpoint: str) -> torch.Tensor:
    """``embed_tokens.weight`` straight off the safetensors shard, on CPU.

    E5 is a CPU card; pulling the whole VLM onto a GPU to read one matrix would
    put it back on the queue for no reason.
    """
    from safetensors import safe_open

    idx = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())
    key = "model.language_model.embed_tokens.weight"
    shard = idx["weight_map"][key]
    with safe_open(Path(checkpoint) / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(key).float()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/home/bc/data/runs/where_b/amort_cache_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--oracle",
                    default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/oracle/BA-3-Joint/s5/V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--readout", default="band")
    ap.add_argument("--sigma", type=float, default=1.0,
                    help="Gaussian softening sigma, in H/32 cells")
    ap.add_argument("--ridge-eps", type=float, default=1e-3,
                    help="epsilon of the (A^T A + eps I) closed form")
    ap.add_argument("--gain", type=float, default=1.0,
                    help="slope of the arm-constant squash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-viz", type=int, default=6)
    ap.add_argument("--workers", type=int, default=32,
                    help="fit-pool workers; 0 = serial")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    t0 = time.time()
    from transformers import AutoTokenizer

    from q3vl.where.basis import Latent, mask_from_latent
    from q3vl.where.config import FitConfig, S_SCALE, UpsampleConfig
    from q3vl.where.fitpool import FitPool, FitTask
    from q3vl.where.oracle import _softplus_inv
    from q3vl.where.phi import build_phi_dir
    from q3vl.where.readout import apply_readout, param_shapes
    from q3vl.where.upsample import area_resize, combine_then_upsample, luma_guide
    from q3vl.whereb.attnread import merged_grid
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import (
        center_prior_field, grid_boundary_f1, gt_area_k, hard_iou, soft_iou_value,
        topk_mask,
    )
    from q3vl.whereb.scripts.run_pw5_fpresim import subject_nouns

    out_dir = Path(args.out)
    for sub in ("viz", "config", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

    cache = Path(args.cache)
    man = json.loads((cache / "manifest.json").read_text())
    samples = man["samples"][: args.limit] if args.limit else man["samples"]
    print(f"cache: {len(samples)} samples", flush=True)

    tok = AutoTokenizer.from_pretrained(args.checkpoint)
    E = load_embeddings(args.checkpoint)
    print(f"embeddings {tuple(E.shape)}  ({time.time()-t0:.0f}s)", flush=True)

    ds, ds_facts = open_dataset(args.split, need_mask=True)
    meta = ds.meta_rows()
    by_id = {}
    for i, r in enumerate(meta):
        if r.get("render_mode") == "local":
            by_id.setdefault(r.get("sample_id") or ds.refs[i].sample_id, i)

    wcache: dict[str, torch.Tensor] = {}

    def wemb(word: str) -> torch.Tensor:
        if word not in wcache:
            ids = tok(" " + word, add_special_tokens=False)["input_ids"]
            wcache[word] = E[torch.tensor(ids)].mean(dim=0)
        return wcache[word]

    # ---------------- pass 1: raw fields, arm-wide normalisation stats -------
    print("pass 1: raw similarity fields", flush=True)
    recs: list[dict[str, Any]] = []
    pool: dict[str, list[np.ndarray]] = {"dot": [], "cosine": []}

    for rec in samples:
        sid = rec["sample_id"]
        if sid not in by_id:
            continue
        idx = by_id[sid]
        nouns = subject_nouns(ds.record(idx).get("where", "") or "")
        if not nouns:
            continue
        gh32, gw32 = rec["grid32"]
        d = np.load(cache / "cache" / f"{sid}.npz")
        m = torch.from_numpy(d["merger_out"]).float()          # (n32, 2560)
        if m.shape[0] != gh32 * gw32:
            continue
        e = wemb(nouns[0])
        dot = (m @ e).double()
        cos = (torch.nn.functional.normalize(m, dim=-1)
               @ torch.nn.functional.normalize(e, dim=0)).double()
        recs.append({"sample_id": sid, "idx": idx, "noun": nouns[0],
                     "grid16": rec["grid16"], "grid32": [gh32, gw32],
                     "dot": dot.numpy(), "cosine": cos.numpy()})
        pool["dot"].append(dot.numpy())
        pool["cosine"].append(cos.numpy())

    print(f"  {len(recs)} samples with nouns + cache ({time.time()-t0:.0f}s)", flush=True)

    # arm-wide constants: robust centre and scale over EVERY valid cell of the
    # whole arm.  Explicitly not per image (s-cache contract red line).
    norm: dict[str, dict[str, float]] = {}
    for basis in ("dot", "cosine"):
        allv = np.concatenate(pool[basis])
        med = float(np.median(allv))
        mad = float(np.median(np.abs(allv - med))) * 1.4826
        norm[basis] = {"center": med, "scale": max(mad, 1e-9),
                       "raw_min": float(allv.min()), "raw_max": float(allv.max()),
                       "n_cells": int(allv.size)}
        print(f"  arm-wide {basis}: center={med:.4f} scale={mad:.4f} "
              f"domain=[{allv.min():.3f}, {allv.max():.3f}]", flush=True)

    def squash(v: np.ndarray, basis: str) -> np.ndarray:
        n = norm[basis]
        return 1.0 / (1.0 + np.exp(-args.gain * (v - n["center"]) / n["scale"]))

    # ---------------- pass 2: soften -> H/16 -> project -> decode -> score ---
    print("pass 2: project and decode", flush=True)
    fitcfg = FitConfig(seed=args.seed)
    ucfg = UpsampleConfig()
    rows_out: list[dict[str, Any]] = []
    rho_pool: list[dict[str, np.ndarray]] = []
    viz_pool: list[dict[str, Any]] = []

    ARMS = ("dot", "cosine", "centre_prior")

    # 2a -- everything cheap and deterministic, plus the fit tasks.  The fits go
    # to the resident single-threaded pool: serial they are 25 s/sample (3 arms x
    # 18 restarts of float64 L-BFGS), which would be ~2.8 h for the split.
    prep: dict[str, dict[str, Any]] = {}
    tasks = []
    for n_i, r in enumerate(recs):
        sid = r["sample_id"]
        gh16, gw16 = r["grid16"]
        gh32, gw32 = r["grid32"]
        d = np.load(cache / "cache" / f"{sid}.npz")
        sem = torch.from_numpy(d["semantic_low"]).double()
        img = torch.from_numpy(d["img_low"]).double()
        phi = build_phi_dir(sem, img, gh16, gw16).phi_dir       # (P16, 71)

        sample = ds[r["idx"]]
        gt_hi = sample.mask_target_hi().double()
        gt16 = area_resize(gt_hi[None, None], (gh16, gw16))[0, 0]
        guide_hi = luma_guide(sample.image_tensor().double().unsqueeze(0))

        a16 = float((gt16.reshape(-1) > 0.5).double().mean())
        row: dict[str, Any] = {
            "sample_id": sid, "noun": r["noun"],
            "grid16": [gh16, gw16], "grid32": [gh32, gw32],
            "area_frac": a16,
            "random_floor": a16 / (2 - a16) if a16 < 1 else 1.0,
        }

        # centre-prior column, on the SAME grid and the SAME top-k rule
        k16 = gt_area_k(gt16)
        cp16 = center_prior_field(gh16, gw16).double()
        row["centre_prior_softiou"] = soft_iou_value(
            topk_mask(cp16, k16), (gt16 > 0.5).double())
        row["centre_prior_bf1"] = grid_boundary_f1(
            topk_mask(cp16, k16), (gt16 > 0.5).double())

        priors: dict[str, torch.Tensor] = {}
        for arm in ARMS:
            if arm == "centre_prior":
                p16 = torch.sigmoid(args.gain * (cp16 - cp16.mean()) / (cp16.std() + EPS))
            else:
                f32 = torch.from_numpy(r[arm]).reshape(gh32, gw32)
                f32s = gaussian_soften(f32, args.sigma)
                p32 = torch.from_numpy(
                    squash(f32s.reshape(-1).numpy(), arm)).reshape(1, 1, gh32, gw32)
                p16 = torch.nn.functional.interpolate(
                    p32, size=(gh16, gw16), mode="bilinear", align_corners=False
                )[0, 0].double()
            priors[arm] = p16
            row[f"{arm}__prior_softiou16"] = soft_iou_value(
                topk_mask(p16, k16), (gt16 > 0.5).double())
            tasks.append(FitTask(
                key=f"{sid}::{arm}", phi=phi,
                target=p16.reshape(-1).clamp(0.0, 1.0),
                readouts=(args.readout,), cfg=fitcfg))

        prep[sid] = {"phi": phi, "gt16": gt16, "gt_hi": gt_hi, "guide_hi": guide_hi,
                     "k16": k16, "cp16": cp16, "priors": priors, "row": row,
                     "grid16": (gh16, gw16), "noun": r["noun"], "area": a16}
        if (n_i + 1) % 100 == 0:
            print(f"  prep [{n_i+1}/{len(recs)}] {time.time()-t0:.0f}s", flush=True)

    print(f"2b: {len(tasks)} fits on {args.workers} workers "
          f"({time.time()-t0:.0f}s)", flush=True)
    with FitPool(args.workers) as fp:
        fp.warmup()
        if args.workers:
            fp.self_check(tasks[0])
        fits = fp.run(tasks)
    print(f"  fits done ({time.time()-t0:.0f}s)", flush=True)

    print("2c: decode and score", flush=True)
    for sid, pk in prep.items():
        row = pk["row"]
        phi = pk["phi"]
        gh16, gw16 = pk["grid16"]
        gt16, gt_hi, guide_hi = pk["gt16"], pk["gt_hi"], pk["guide_hi"]
        k16, cp16, a16 = pk["k16"], pk["cp16"], pk["area"]
        for arm in ARMS:
            p16 = pk["priors"][arm]
            prior = p16.reshape(-1).clamp(0.0, 1.0)
            fr = fits[f"{sid}::{arm}"][args.readout]
            if fr.latent is None:
                row[f"{arm}__fit_status"] = fr.status
                continue
            lat = fr.latent
            m_low, s_low_v = mask_from_latent(phi, lat)
            m_low2 = m_low.reshape(gh16, gw16)
            row[f"{arm}__fit_status"] = fr.status
            row[f"{arm}__fit_iou_vs_prior"] = 1.0 - float(fr.loss)
            row[f"{arm}__full_softiou16"] = soft_iou_value(
                topk_mask(m_low2, k16), (gt16 > 0.5).double())
            row[f"{arm}__full_softiou16_raw"] = soft_iou_value(m_low2, gt16)
            row[f"{arm}__full_hardiou16"] = hard_iou(
                topk_mask(m_low2, k16), (gt16 > 0.5).double())
            row[f"{arm}__full_bf1_16"] = grid_boundary_f1(
                topk_mask(m_low2, k16), (gt16 > 0.5).double())

            # hi-res, through the one sanctioned guided upsample
            s_hi, _ = combine_then_upsample(phi, lat, gh16, gw16, guide_hi, ucfg)
            m_hi = apply_readout(args.readout, s_hi, lat.rho)
            row[f"{arm}__full_softiou_hi"] = soft_iou_value(m_hi.reshape(-1), gt_hi.reshape(-1))

            if arm != "centre_prior":
                rho_pool.append({k: v.detach().numpy().copy() for k, v in lat.rho.items()})

            # (2) ridge_lsq: exactly the closed form P3 embeds
            z = torch.logit(prior.clamp(1e-3, 1 - 1e-3))
            A = torch.cat([torch.ones(phi.shape[0], 1, dtype=phi.dtype), phi], dim=1)
            G = A.T @ A
            reg = torch.eye(G.shape[0], dtype=G.dtype) * args.ridge_eps * float(
                torch.diagonal(G).mean())
            reg[0, 0] = 0.0                      # never damp the intercept
            c = torch.linalg.solve(G + reg, A.T @ z)
            w0r, v = c[0], c[1:]
            alpha_r = float(v.norm())
            if alpha_r > 1e-9:
                lat_r = Latent(
                    args.readout, w0r.clone(),
                    torch.tensor(_softplus_inv(alpha_r), dtype=phi.dtype),
                    (v / alpha_r).clone(),
                    {k: torch.as_tensor(vv, dtype=phi.dtype).clone()
                     for k, vv in lat.rho.items()},
                )
                m_low_r, _ = mask_from_latent(phi, lat_r)
                m_low_r2 = m_low_r.reshape(gh16, gw16)
                row[f"{arm}__ridge_softiou16"] = soft_iou_value(
                    topk_mask(m_low_r2, k16), (gt16 > 0.5).double())
                row[f"{arm}__ridge_bf1_16"] = grid_boundary_f1(
                    topk_mask(m_low_r2, k16), (gt16 > 0.5).double())

            if arm == "dot" and len(viz_pool) < 4 * args.n_viz:
                viz_pool.append({
                    "sample_id": sid, "noun": pk["noun"], "grid16": [gh16, gw16],
                    "prior": p16.numpy(), "decoded": m_low2.numpy(),
                    "gt": gt16.numpy(), "centre": cp16.numpy(), "k": k16,
                    "score": row.get(f"{arm}__full_softiou16", 0.0),
                    "prior_score": row.get(f"{arm}__prior_softiou16", 0.0),
                    "centre_score": row["centre_prior_softiou"],
                    "area": a16, "floor": row["random_floor"],
                })
        rows_out.append(row)

    # ---------------- oracle ceiling (published latents, sanity) ------------
    from q3vl.whereb.stores import OracleStore
    ost = OracleStore(Path(args.oracle))
    orc: list[float] = []
    for r in recs:
        try:
            fit = ost.read_json(r["sample_id"], ".oracle.json")["fits"].get(args.readout)
        except Exception:
            continue
        if not fit or fit.get("status") != "ok":
            continue
        orc.append(float(fit.get("metrics", {}).get("soft_iou_minmax", np.nan)))

    # ---------------- summary ------------------------------------------------
    def col(k: str) -> np.ndarray:
        return np.asarray([r[k] for r in rows_out if k in r], dtype=np.float64)

    summary: dict[str, Any] = {
        "card": "E5 (prior-field projection onto Phi-71, zero training, no GT in the fit)",
        "n_samples": len(rows_out),
        "readout": args.readout,
        "sigma_cells": args.sigma,
        "ridge_eps": args.ridge_eps,
        "gain": args.gain,
        "leakage": "GT is used ONLY as a metric; every fit target is the prior field",
        "normalisation": {
            "rule": "arm-wide robust median/MAD then sigmoid; per-image normalisation "
                    "is forbidden by the s-cache consumption contract",
            **norm,
        },
        "registered_gate": ">=0.70 -> refinement problem (P2); <=0.40 -> from-scratch (P1/P3)",
        "oracle_ceiling_softiou": agg(orc),
        "random_floor": agg(col("random_floor")),
        "area_frac": agg(col("area_frac")),
        "centre_prior_softiou": agg(col("centre_prior_softiou")),
        "centre_prior_bf1": agg(col("centre_prior_bf1")),
        "arms": {},
    }
    for arm in ARMS:
        summary["arms"][arm] = {
            "prior_field_softiou16": agg(col(f"{arm}__prior_softiou16")),
            "fit_iou_vs_prior": agg(col(f"{arm}__fit_iou_vs_prior")),
            "full_fit_softiou16": agg(col(f"{arm}__full_softiou16")),
            "full_fit_softiou16_unthresholded": agg(col(f"{arm}__full_softiou16_raw")),
            "full_fit_hardiou16": agg(col(f"{arm}__full_hardiou16")),
            "full_fit_grid_bf1": agg(col(f"{arm}__full_bf1_16")),
            "full_fit_softiou_hi": agg(col(f"{arm}__full_softiou_hi")),
            "ridge_lsq_softiou16": agg(col(f"{arm}__ridge_softiou16")),
            "ridge_lsq_grid_bf1": agg(col(f"{arm}__ridge_bf1_16")),
        }

    # area strata (E1 review R5: the >=0.45 band has a 0.527 random floor)
    a = col("area_frac")
    strata = {}
    for lo, hi in ((0.0, 0.15), (0.15, 0.30), (0.30, 0.45), (0.45, 1.01)):
        sel = [i for i, r in enumerate(rows_out) if lo <= r["area_frac"] < hi]
        if not sel:
            continue
        pick = lambda k: np.asarray(  # noqa: E731
            [rows_out[i][k] for i in sel if k in rows_out[i]], dtype=np.float64)
        strata[f"area_{lo:.2f}_{hi:.2f}"] = {
            "n": len(sel),
            "random_floor": agg(pick("random_floor")),
            "centre_prior": agg(pick("centre_prior_softiou")),
            "dot_full_fit": agg(pick("dot__full_softiou16")),
            "cosine_full_fit": agg(pick("cosine__full_softiou16")),
            "centre_prior_projected": agg(pick("centre_prior__full_softiou16")),
        }
    summary["area_strata"] = strata

    # paired deltas against the mandatory columns
    from q3vl.whereb.attnprobe import paired_wilcoxon
    summary["paired"] = {}
    for arm in ARMS:
        v = col(f"{arm}__full_softiou16")
        if v.size == len(rows_out):
            summary["paired"][f"{arm}_vs_centre_prior"] = paired_wilcoxon(
                v, col("centre_prior_softiou"))
            summary["paired"][f"{arm}_vs_random_floor"] = paired_wilcoxon(
                v, col("random_floor"))
            summary["paired"][f"{arm}_decode_minus_prior"] = paired_wilcoxon(
                v, col(f"{arm}__prior_softiou16"))

    med = summary["arms"]["dot"]["full_fit_softiou16"].get("median", float("nan"))
    summary["VERDICT"] = {
        "best_arm_median_softiou16": max(
            summary["arms"][a_]["full_fit_softiou16"].get("median", 0.0) for a_ in ARMS),
        "dot_median_softiou16": med,
        "refinement_problem_ge_0.70": bool(med >= 0.70),
        "from_scratch_le_0.40": bool(med <= 0.40),
    }

    (out_dir / "metrics.json").write_text(json.dumps(
        {"summary": summary, "per_sample": rows_out}, indent=2), encoding="utf-8")

    setup = {
        "cache": str(cache), "checkpoint": args.checkpoint, "split": args.split,
        "readout": args.readout, "seed": args.seed, "sigma": args.sigma,
        "ridge_eps": args.ridge_eps, "gain": args.gain,
        "dataset_facts": ds_facts,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True, cwd="/home/bc/VeraRetouch").stdout.strip(),
        "python": platform.python_version(), "torch": torch.__version__,
        "numpy": np.__version__, "platform": platform.platform(),
    }
    (out_dir / "config" / "run_setup.json").write_text(
        json.dumps(setup, indent=2), encoding="utf-8")

    # ---------------- viz ----------------------------------------------------
    try:
        _write_viz(out_dir / "viz", viz_pool, args.n_viz)
    except Exception as exc:                                    # pragma: no cover
        print(f"viz failed: {exc}", flush=True)

    print("\n=== E5 ===", flush=True)
    print(f"n={len(rows_out)}  oracle ceiling median "
          f"{summary['oracle_ceiling_softiou'].get('median', float('nan')):.4f}", flush=True)
    print(f"random floor median  {summary['random_floor']['median']:.4f}", flush=True)
    print(f"centre prior median  {summary['centre_prior_softiou']['median']:.4f}", flush=True)
    for arm in ARMS:
        s = summary["arms"][arm]
        print(f"  {arm:>14}: prior {s['prior_field_softiou16'].get('median', float('nan')):.4f}"
              f" -> full_fit {s['full_fit_softiou16'].get('median', float('nan')):.4f}"
              f" | ridge {s['ridge_lsq_softiou16'].get('median', float('nan')):.4f}"
              f" | bf1 {s['full_fit_grid_bf1'].get('median', float('nan')):.4f}"
              f" | hi {s['full_fit_softiou_hi'].get('median', float('nan')):.4f}", flush=True)
    print(f"VERDICT: {json.dumps(summary['VERDICT'])}", flush=True)
    print(f"done in {(time.time()-t0)/60:.1f} min -> {out_dir}", flush=True)
    return 0


def _write_viz(viz_dir: Path, pool: list[dict[str, Any]], n: int) -> None:
    """Five-panel: image-grid GT / prior / decoded / decoded top-k vs GT / centre.

    Colour discipline (2026-08-05): no per-image min-max.  The prior and the
    decoded mask are both already in [0,1] by construction, so they are drawn on
    a fixed 0..1 scale and are directly comparable across panels and samples.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not pool:
        return
    pool = sorted(pool, key=lambda p: p["score"])
    picks = [("failure", p) for p in pool[:n]] + [("success", p) for p in pool[-n:]]
    for tag, p in picks:
        gh, gw = p["grid16"]
        gt = p["gt"]
        k = p["k"]
        dec = p["decoded"]
        flat = dec.reshape(-1)
        topk = np.zeros_like(flat)
        if k:
            topk[np.argsort(-flat)[:k]] = 1.0
        topk = topk.reshape(gh, gw)
        fig, ax = plt.subplots(1, 5, figsize=(18, 3.4))
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
        ax[0].imshow(gt, cmap="gray", vmin=0, vmax=1)
        ax[0].set_title(f"GT (area={p['area']:.2f})", fontsize=9)
        ax[1].imshow(p["prior"], cmap="magma", vmin=0, vmax=1)
        ax[1].set_title(f"prior '{p['noun']}' (softIoU {p['prior_score']:.3f})", fontsize=9)
        ax[2].imshow(dec, cmap="magma", vmin=0, vmax=1)
        ax[2].set_title(f"D(w) decoded (softIoU {p['score']:.3f})", fontsize=9)
        ax[3].imshow(topk - (gt > 0.5), cmap="bwr", vmin=-1, vmax=1)
        ax[3].set_title("top-k(D(w)) - GT   red=FP blue=FN", fontsize=9)
        ax[4].imshow(p["centre"], cmap="viridis")
        ax[4].set_title(f"centre prior ({p['centre_score']:.3f})", fontsize=9)
        fig.suptitle(f"{tag}  {p['sample_id']}   random floor {p['floor']:.3f}",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(viz_dir / f"{tag}_{p['sample_id']}.png", dpi=110)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
