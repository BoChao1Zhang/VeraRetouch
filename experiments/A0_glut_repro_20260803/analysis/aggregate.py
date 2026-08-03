"""Aggregate every A0 gap-attribution artefact into the top-level metrics.json.

Reads (whatever exists):
  runs/{rec,full}/metrics.json + per_lut.jsonl   original A0 full run
  ablate/ablate_20ep.json                        ingredient isolation
  ablate/ablate_fix.json                         repaired L_hc arms
  ablate/rec_{40,60}ep.json                      epoch sweep
  hypA/e1_on_a0_*.json                           E1 engine, same LUTs
  analysis/lhc_grad_probe.json                   loss-scale evidence
  analysis/corpus_difficulty.json                corpus-difficulty regression
Writes: metrics.json (top-level summary the harness / result reviewer reads)
"""

from __future__ import annotations

import json
import os
import subprocess

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
_REPO = os.path.abspath(os.path.join(EXP, "..", ".."))

ANCHOR = {"psnr_db": 45.47, "tolerance": 0.3, "de00": 0.41,
          "source": "GLUT arXiv:2605.19889v1, GLUT-32 @ 75-LUT Hald "
                    "(IMPL_DOSSIER 2.2/2.4); their 75 files are not public"}


def jload(p):
    p = os.path.join(EXP, p)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def jlines(p):
    p = os.path.join(EXP, p)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", _REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> None:
    out: dict = {"exp": "A0_glut_repro / anchor-gap attribution",
                 "date": "2026-08-03", "git_commit": git_commit(),
                 "anchor": ANCHOR}

    # ---- headline: the original 75-LUT full run -------------------------
    rec, full = jload("runs/rec/metrics.json"), jload("runs/full/metrics.json")
    out["headline_75lut"] = {}
    for nm, m in (("rec", rec), ("full", full)):
        if not m:
            continue
        out["headline_75lut"][nm] = {
            "psnr_float_mean": m["psnr_float_mean"],
            "psnr_float_std": m["psnr_float_std"],
            "psnr_8bit_mean": m["psnr_8bit_mean"],
            "de00_mean_over_luts": m["de00_mean_over_luts"],
            "de00_p50": m["de00_p50"],
            "psnr_min": m["psnr_min"],
            "anchor_pass": m["anchor_pass"],
            "gap_vs_anchor_db": m["psnr_float_mean"] - ANCHOR["psnr_db"],
        }
    if rec and full:
        out["headline_75lut"]["full_minus_rec_db"] = (
            full["psnr_float_mean"] - rec["psnr_float_mean"])

    # ---- task 1: ingredient isolation -----------------------------------
    abl, fix = jload("ablate/ablate_20ep.json"), jload("ablate/ablate_fix.json")
    arms: dict = {}
    for blob in (abl, fix):
        if not blob:
            continue
        for k, v in blob["results"].items():
            arms[k] = {"psnr_float_mean": v["psnr_float_mean"],
                       "de00_mean": v["de00_mean"],
                       "cfg": v["cfg"],
                       "alive_frac_mean": v["health"]["alive_frac_mean"],
                       "opacity_mean": v["health"]["opacity_mean"],
                       "opacity_frac_below_0.05":
                           v["health"]["opacity_frac_below_0.05"],
                       "gnorm_p50": v["health"]["gnorm_p50"],
                       "gnorm_max_over_p50":
                           v["health"]["gnorm_max"] /
                           max(v["health"]["gnorm_p50"], 1e-12),
                       "gnorm_spikes_100x": v["health"]["gnorm_spikes_100x"],
                       # what the arm actually ended up minimising
                       "final_l_rec": v["loss_log"][-1]["l_rec"],
                       "final_l_hc_monitor": v["loss_log"][-1]["l_hc_monitor"],
                       "opacity_frac_at_1": v["health"]["opacity_frac_at_1"]}
    if arms and "rec" in arms:
        base = arms["rec"]["psnr_float_mean"]
        for k in arms:
            arms[k]["delta_vs_rec_db"] = arms[k]["psnr_float_mean"] - base
    out["task1_ingredient_isolation"] = {
        "protocol": "9 stratified LUTs of the A0 75 (spanning the rec-arm PSNR "
                    "range), 20 ep, N=32, identical seed -> paired comparison; "
                    "eval on the fixed 2^21 held-out colour subsample",
        "lut_ids": (abl or fix or {}).get("lut_ids"),
        "arms": arms,
    }

    probe = jload("analysis/lhc_grad_probe.json")
    if probe:
        gi = probe["at_glut_init"]["batch_grad_norm_wrt_f"]
        gc = probe["at_rec_converged"]["batch_grad_norm_wrt_f"]
        out["task1_root_cause"] = {
            "defect_D2_loss_scale_mismatch": {
                "grad_ratio_10Lhc_over_Lrec_at_init":
                    gi["10*L_hc_paper"]["ratio_to_L_rec_p50"],
                "grad_ratio_10Lhc_over_Lrec_at_convergence":
                    gc["10*L_hc_paper"]["ratio_to_L_rec_p50"],
                "lambda_that_would_make_it_a_10pct_perturbation":
                    probe["at_rec_converged"]["lambda_for_10pct_of_L_rec"],
            },
            "defect_D1_chroma_singularity": {
                "grad_max_over_p50_paper_at_init":
                    gi["10*L_hc_paper"]["max_over_p50"],
                "grad_max_over_p50_paper_at_convergence":
                    gc["10*L_hc_paper"]["max_over_p50"],
                "grad_max_over_p50_chroma_floored_at_convergence":
                    gc["10*L_hc_stable"]["max_over_p50"],
                "singular_samples_per_1024_batch_at_convergence":
                    probe["at_rec_converged"]["occupancy"][
                        "singular_samples_per_1024_batch"],
            },
        }

    # ---- task 2a: E1 engine, same LUTs ----------------------------------
    hyp_a = {}
    recs = {r["lut_id"]: r["psnr_float"] for r in
            (jlines("runs/rec/per_lut.jsonl") or [])}
    for tag in ("e1_on_a0_3k", "e1_on_a0_12k", "e1_on_a0_N48", "e1_on_a0_N64"):
        s = jload(f"hypA/{tag}.json")
        if not s:
            continue
        rows = jlines(f"hypA/{tag}.jsonl") or []
        d = [r["psnr_float"] - recs[r["lut_id"]] for r in rows
             if r["lut_id"] in recs]
        hyp_a[tag] = {
            "n": s["n_gaussians"], "steps": s["steps"], "n_luts": s["n_luts"],
            "psnr_float_mean": s["psnr_float_mean"],
            "de00_mean_over_luts": s["de00_mean_over_luts"],
            "alive_frac_mean": s["alive_frac_mean"],
            "paired_delta_vs_a0_rec_db": float(np.mean(d)) if d else None,
            "n_luts_e1_better": int(np.sum(np.asarray(d) > 0)) if d else None,
            "gap_vs_anchor_db": s["psnr_float_mean"] - ANCHOR["psnr_db"],
        }
    out["task2a_representation_vs_recipe"] = {
        "protocol": "E1 direct-overfit engine (k-means init, f(x)=x at step 0, "
                    "sigma annealing, density control, bs 8192 / lr 5e-3) on "
                    "the SAME 75 LUTs / same GT / same held-out colours",
        "runs": hyp_a}

    # ---- task 2b: epoch budget ------------------------------------------
    ep = {}
    if abl and "rec" in abl["results"]:
        ep["20"] = {"psnr_float_mean": abl["results"]["rec"]["psnr_float_mean"],
                    "de00_mean": abl["results"]["rec"]["de00_mean"]}
    for e in (40, 60):
        b = jload(f"ablate/rec_{e}ep.json")
        if b:
            r = b["results"]["rec"]
            ep[str(e)] = {"psnr_float_mean": r["psnr_float_mean"],
                          "de00_mean": r["de00_mean"]}
    if "20" in ep:
        for k in ep:
            ep[k]["delta_vs_20ep_db"] = (ep[k]["psnr_float_mean"]
                                         - ep["20"]["psnr_float_mean"])
    out["task2b_epoch_budget"] = {
        "protocol": "same 9 stratified LUTs, rec arm, 20 / 40 / 60 epochs",
        "by_epochs": ep}

    # ---- task 2c: corpus difficulty -------------------------------------
    cd = jload("analysis/corpus_difficulty.json")
    if cd:
        out["task2c_corpus_difficulty"] = {
            "best_predictor": cd["best_predictor"],
            "corr_with_rec_psnr": cd["corr_with_rec_psnr"],
            "corpus_summary": cd["corpus_summary"],
            "psnr_projection": cd["psnr_projection"],
            "nilut_LUT01_reference": cd["nilut_LUT01"],
        }

    with open(os.path.join(EXP, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1)[:4000])


if __name__ == "__main__":
    main()
