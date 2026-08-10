"""E2 analysis: pre-registered criteria table, ablations, condition checks,
metrics.json, and viz (ring evidence figure + per-family success/failure)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP = Path("/home/bc/VeraRetouch/experiments/E2_basis_fit_20260803")
CACHE = Path("/var/cache/veradata/e2_basis_20260803")
sys.path.insert(0, str(EXP))
import e2lib  # noqa: E402
from run_fit import build_phi  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e7e6e2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True,
    "grid.color": GRID, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 11})

GEO_FAMS = ["linear", "radial_ell", "ring", "wedge"]


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def q(xs, p):
    return float(np.quantile(xs, p)) if xs else float("nan")


def load_rows(suffix):
    rows = [json.loads(line) for line in open(EXP / f"results{suffix}.jsonl")]
    errs = [r for r in rows if "error" in r]
    return [r for r in rows if "error" not in r], errs


def group(rows):
    g = defaultdict(list)
    for r in rows:
        g[(r["config"], r["readout"], r["family"])].append(r)
    return g


def rebuild_pred(row, suffix):
    z = np.load(CACHE / f"masks{suffix}" / f"{row['mask_id']}.npz")
    mask = z["mask"].astype(np.float64)
    h, w = mask.shape
    fz = np.load(CACHE / f"feats{suffix}" / f"{row['feat_key']}.npz")
    ch = {"L": fz["L"].astype(np.float64), "S": fz["S"].astype(np.float64),
          "e": fz["e"].astype(np.float64)}
    cols = {"main14": "full", "lin3": "lin", "cubic18": "cubic",
            "dims6": "geo", "dims8": "geo_range", "geo6_ring": "geo"}[
        row["config"]]
    Phi = build_phi(h, w, 1, cols, ch)
    pred = e2lib.predict_mask(row, Phi, row["readout"]).reshape(h, w)
    img = fz["img_u8"]
    return mask, pred, img


def panel(rows_sel, suffix, path, title):
    n = len(rows_sel)
    fig, axes = plt.subplots(n, 3, figsize=(10.5, 3.2 * n), dpi=130,
                             squeeze=False)
    for i, row in enumerate(rows_sel):
        mask, pred, img = rebuild_pred(row, suffix)
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{row['mask_id']} image", fontsize=9)
        axes[i, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("target mask", fontsize=9)
        axes[i, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(
            f"fit ({row['readout']}) IoU={row['iou_minmax']:.3f}", fontsize=9)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ring_evidence(g, suffix):
    mono = [r["iou_minmax"] for r in g[("main14", "monotone", "ring")]]
    band = [r["iou_minmax"] for r in g[("main14", "bandpass", "ring")]]
    mono_g = [r["iou_minmax"] for r in g[("geo6_ring", "monotone", "ring")]]
    band_g = [r["iou_minmax"] for r in g[("geo6_ring", "bandpass", "ring")]]
    fig = plt.figure(figsize=(12, 6.4), dpi=140)
    gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    data = [mono, band, mono_g, band_g]
    labels = ["monotone\n(14-dim)", "bandpass\n(14-dim)",
              "monotone\n(geo-only)", "bandpass\n(geo-only)"]
    colors = [C2, C1, C2, C1]
    for i, (d, c) in enumerate(zip(data, colors)):
        if not d:
            continue
        xs = np.random.default_rng(i).uniform(i - 0.16, i + 0.16, len(d))
        ax.scatter(xs, d, s=14, color=c, alpha=0.55, linewidths=0)
        ax.hlines(np.median(d), i - 0.28, i + 0.28, color=c, lw=2.5)
        ax.annotate(f"med {np.median(d):.2f}", (i + 0.30, np.median(d)),
                    fontsize=9, color=c, va="center")
    ax.axhline(0.40, color=INK2, ls="--", lw=1)
    ax.axhline(0.90, color=INK2, ls=":", lw=1)
    ax.text(3.55, 0.41, "monotone gate <=0.40", fontsize=8, color=INK2)
    ax.text(3.55, 0.91, "bandpass gate >=0.90", fontsize=8, color=INK2)
    ax.set_xticks(range(4), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("soft-IoU (min/max)")
    ax.set_title("Ring masks: what the band-pass s-axis readout buys "
                 "(core pre-registered evidence)")
    # example ring: median bandpass case
    rows_band = sorted(g[("main14", "bandpass", "ring")],
                       key=lambda r: r["iou_minmax"])
    if rows_band:
        ex = rows_band[len(rows_band) // 2]
        ex_mono = next((r for r in g[("main14", "monotone", "ring")]
                        if r["mask_id"] == ex["mask_id"]), None)
        mask, pred_b, _ = rebuild_pred(ex, suffix)
        titles = ["target ring", f"bandpass fit IoU={ex['iou_minmax']:.2f}"]
        images = [mask, pred_b]
        if ex_mono:
            _, pred_m, _ = rebuild_pred(ex_mono, suffix)
            titles.append(f"monotone fit IoU={ex_mono['iou_minmax']:.2f}")
            images.append(pred_m)
        w_dir = np.asarray(ex["w_dir"])
        z = np.load(CACHE / f"masks{suffix}" / f"{ex['mask_id']}.npz")
        h, w = z["mask"].shape
        fz = np.load(CACHE / f"feats{suffix}" / f"{ex['feat_key']}.npz")
        ch = {"L": fz["L"].astype(np.float64),
              "S": fz["S"].astype(np.float64),
              "e": fz["e"].astype(np.float64)}
        Phi = build_phi(h, w, 1, "full", ch)
        s = 3 * np.tanh((ex["w0"] + ex["alpha"] * (Phi @ w_dir)) / 3)
        titles.append("fitted s field")
        images.append(s.reshape(h, w))
        for j, (im, ti) in enumerate(zip(images, titles)):
            axj = fig.add_subplot(gs[1, j])
            axj.imshow(im, cmap="gray" if j < 3 else "viridis")
            axj.set_title(ti, fontsize=9)
            axj.set_xticks([])
            axj.set_yticks([])
            axj.grid(False)
    fig.tight_layout()
    fig.savefig(EXP / "viz" / "ring_evidence.png")
    plt.close(fig)


def ablation_dims_fig(g):
    sems = {"dims6 (geo)": [r["iou_minmax"]
                            for r in g[("dims6", "monotone", "semantic")]],
            "dims8 (+range)": [r["iou_minmax"]
                               for r in g[("dims8", "monotone", "semantic")]],
            "main14 (+sem)": [r["iou_minmax"]
                              for r in g[("main14", "monotone", "semantic")]]}
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=140)
    for i, (name, d) in enumerate(sems.items()):
        if not d:
            continue
        xs = np.random.default_rng(i).uniform(i - 0.15, i + 0.15, len(d))
        ax.scatter(xs, d, s=12, color=C1, alpha=0.45, linewidths=0)
        ax.hlines(np.median(d), i - 0.25, i + 0.25, color=C2, lw=2.5)
        ax.annotate(f"{np.median(d):.3f}", (i + 0.27, np.median(d)),
                    fontsize=9, color=C2, va="center")
    ax.axhline(0.85, color=INK2, ls="--", lw=1)
    ax.text(2.35, 0.855, "gate 0.85", fontsize=8, color=INK2)
    ax.set_xticks(range(len(sems)), list(sems))
    ax.set_ylabel("soft-IoU (min/max)")
    ax.set_title("Semantic masks: marginal contribution of basis blocks "
                 "(monotone readout)")
    fig.tight_layout()
    fig.savefig(EXP / "viz" / "ablation_dims.png")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    suffix = "_smoke" if args.smoke else ""
    rows, errs = load_rows(suffix)
    g = group(rows)
    (EXP / "viz").mkdir(exist_ok=True)

    def ious(cfg, ro, fam):
        return [r["iou_minmax"] for r in g[(cfg, ro, fam)]]

    def stat(cfg, ro, fam):
        xs = ious(cfg, ro, fam)
        return {"n": len(xs), "median": med(xs), "p10": q(xs, 0.10),
                "p90": q(xs, 0.90), "min": q(xs, 0.0) if xs else None}

    const_rows = g[("main14", "monotone", "constant")] + \
        g[("main14", "bandpass", "constant")]
    const_alpha = max((r["alpha"] for r in const_rows), default=float("nan"))
    const_std = max((r["pred_std"] for r in const_rows),
                    default=float("nan"))
    const_mae = max((r["mae"] for r in const_rows), default=float("nan"))

    crit = {
        "linear_mono_min097": {
            "measured_median": med(ious("main14", "monotone", "linear")),
            "gate": ">=0.97",
            "pass": med(ious("main14", "monotone", "linear")) >= 0.97},
        "radial_ell_mono_min097": {
            "measured_median": med(ious("main14", "monotone", "radial_ell")),
            "gate": ">=0.97",
            "pass": med(ious("main14", "monotone", "radial_ell")) >= 0.97},
        "ring_mono_max040": {
            "measured_median": med(ious("main14", "monotone", "ring")),
            "gate": "<=0.40",
            "pass": med(ious("main14", "monotone", "ring")) <= 0.40},
        "ring_band_min090_CORE": {
            "measured_median": med(ious("main14", "bandpass", "ring")),
            "gate": ">=0.90",
            "pass": med(ious("main14", "bandpass", "ring")) >= 0.90},
        "wedge_mono_expected_055_070": {
            "measured_median": med(ious("main14", "monotone", "wedge")),
            "gate": "0.55-0.70 expected; <0.50 = impl bug",
            "pass": med(ious("main14", "monotone", "wedge")) >= 0.50},
        "semantic_mono_min085": {
            "measured_median": med(ious("main14", "monotone", "semantic")),
            "gate": ">=0.85",
            "pass": med(ious("main14", "monotone", "semantic")) >= 0.85},
        "constant_alpha_lt_1e2_std_lt_001": {
            "measured_alpha_max": const_alpha,
            "measured_pred_std_max": const_std,
            "measured_mae_max": const_mae,
            "gate": "alpha<1e-2 and std<0.01",
            "pass": bool(const_alpha < 1e-2 and const_std < 0.01)},
    }

    # ablations
    abl = {
        "lin3_vs_main14_geo": {
            fam: {"lin3": med(ious("lin3", "monotone", fam)),
                  "main14": med(ious("main14", "monotone", fam))}
            for fam in GEO_FAMS},
        "cubic_vs_main14_geo": {
            fam: {"main14": med(ious("main14", "monotone", fam)),
                  "cubic18": med(ious("cubic18", "monotone", fam))}
            for fam in GEO_FAMS},
        "dims_semantic": {
            "dims6": med(ious("dims6", "monotone", "semantic")),
            "dims8": med(ious("dims8", "monotone", "semantic")),
            "main14": med(ious("main14", "monotone", "semantic"))},
        "geo6_ring_control": {
            "monotone": med(ious("geo6_ring", "monotone", "ring")),
            "bandpass": med(ious("geo6_ring", "bandpass", "ring"))},
    }

    # per-class semantic breakdown
    per_class = defaultdict(list)
    for r in g[("main14", "monotone", "semantic")]:
        per_class[r.get("class5") or "?"].append(r["iou_minmax"])
    sem_classes = {k: {"n": len(v), "median": med(v)}
                   for k, v in sorted(per_class.items())}

    # condition checks
    cond_legendre = e2lib.gram_cond(np.concatenate(
        [np.ones((512 * 512, 1)), e2lib.geo_features(512, 512)], axis=1))
    cond_monomial = e2lib.gram_cond(e2lib.monomial_features(512, 512))
    resid = json.load(open(EXP / "cache_meta" /
                           f"residualization{suffix}.json"))
    fro_b = [r["fro_offblock_before"] for r in resid]
    fro_a = [r["fro_offblock_after"] for r in resid]
    cond = {
        "gram_cond_legendre6": cond_legendre,
        "gram_cond_monomial6": cond_monomial,
        "gram_cond_gate_lt10": cond_legendre < 10,
        "offblock_corr_fro_before_mean": float(np.mean(fro_b)),
        "offblock_corr_fro_after_max": float(np.max(fro_a)),
        "offblock_gate_lt005": float(np.max(fro_a)) < 0.05,
        "note": "residualization is an exact per-image lstsq projection, "
                "so the off-block correlation after is ~0 by construction",
    }

    summary = {(f"{c}|{ro}|{fam}"): stat(c, ro, fam)
               for (c, ro, fam) in sorted(g)}
    metrics = {"criteria": crit, "ablations": abl,
               "semantic_class5": sem_classes,
               "condition_checks": cond, "per_cell": summary,
               "n_fit_errors": len(errs),
               "n_results": len(rows)}
    out = EXP / f"metrics{suffix}.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps({"criteria": crit, "cond": cond}, indent=1))

    # ---- viz ----
    ring_evidence(g, suffix)
    ablation_dims_fig(g)
    for fam in GEO_FAMS + ["semantic"]:
        sel = sorted(g[("main14", "monotone", fam)],
                     key=lambda r: r["iou_minmax"])
        selb = sorted(g[("main14", "bandpass", fam)],
                      key=lambda r: r["iou_minmax"])
        if not sel:
            continue
        best = sel[-2:][::-1] + selb[-1:]
        worst = sel[:2] + selb[:1]
        panel(best, suffix, EXP / "viz" / f"success_{fam}.png",
              f"E2 success cases: {fam}")
        panel(worst, suffix, EXP / "viz" / f"failure_{fam}.png",
              f"E2 failure cases: {fam}")
    print("analyze done ->", out)


if __name__ == "__main__":
    main()
