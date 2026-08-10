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
# result shards: "" = round 1, the rest = supplement round (gaps A and B).
# An explicit list (not a glob) so that suffix="" never picks up "_smoke".
SHARDS = ["", "_gauss", "_constrained", "_c2", "_c3", "_c4", "_c5", "_c6",
          "_sweep"]
READOUT_LABEL = {
    "monotone": "monotone sigma(g*s+b)",
    "bandpass": "unconstrained flat-top band",
    "gauss": "unconstrained single Gaussian",
    "cband_norm": "CONSTRAINED axis (M=12, mu fixed, sigma bounded)",
    "cband_unnorm": "constrained axis, no normalization (R-6/RD-D)",
    "cgauss": "constrained single primitive",
}


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def q(xs, p):
    return float(np.quantile(xs, p)) if xs else float("nan")


def load_rows(suffix):
    rows = []
    for sh in SHARDS:
        p = EXP / f"results{suffix}{sh}.jsonl"
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            r.setdefault("arm", "main")
            r["_shard"] = sh or "_main"
            rows.append(r)
    errs = [r for r in rows if "error" in r]
    return [r for r in rows if "error" not in r], errs


def group(rows):
    """Main grouping = the `main` arm only; sweep / budget arms are kept
    apart so they can never contaminate a criteria cell."""
    g = defaultdict(list)
    for r in rows:
        if r.get("arm", "main") != "main":
            continue
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
            "dims6": "geo", "dims8": "geo_range", "geo6_ring": "geo",
            "geo6": "geo"}[row["config"]]
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


def constrained_evidence_fig(g, suffix):
    """Gap A core figure: can the renderer's CONSTRAINED s axis synthesize the
    flat top that the 0.975 depends on?  Six readouts on the same 200 rings."""
    order = ["monotone", "gauss", "cgauss", "cband_unnorm", "cband_norm",
             "bandpass"]
    short = {"monotone": "monotone", "gauss": "single\nGaussian",
             "cgauss": "constrained\nsingle prim.",
             "cband_unnorm": "constrained\n(no norm.)",
             "cband_norm": "CONSTRAINED\nM=12 axis",
             "bandpass": "unconstrained\nflat-top band"}
    cols = {"monotone": C2, "gauss": C2, "cgauss": C2,
            "cband_unnorm": C3, "cband_norm": C1, "bandpass": INK2}
    fig = plt.figure(figsize=(13, 7.6), dpi=140)
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1])
    ax = fig.add_subplot(gs[0, :])
    xt, xl = [], []
    for j, cfg in enumerate(("main14", "geo6_ring")):
        for i, ro in enumerate(order):
            d = [r["iou_minmax"] for r in g[(cfg, ro, "ring")]]
            if not d:
                continue
            x = j * 6.6 + i
            xs = np.random.default_rng(i + 7 * j).uniform(x - .16, x + .16,
                                                          len(d))
            ax.scatter(xs, d, s=13, color=cols[ro], alpha=0.5, linewidths=0)
            ax.hlines(np.median(d), x - .3, x + .3, color=cols[ro], lw=2.6)
            ax.annotate(f"{np.median(d):.3f}", (x, np.median(d) + 0.035),
                        fontsize=8.5, color=cols[ro], ha="center")
            xt.append(x)
            xl.append(short[ro])
    ax.axhline(0.90, color=INK2, ls=":", lw=1)
    ax.axhline(0.40, color=INK2, ls="--", lw=1)
    ax.text(-0.45, 0.915, "band-pass gate >=0.90", fontsize=8, color=INK2)
    ax.text(-0.45, 0.415, "monotone gate <=0.40", fontsize=8, color=INK2)
    ax.set_xticks(xt, xl, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("soft-IoU (min/max)")
    ax.set_title("Ring masks, 6 s-axis readouts.  LEFT block = 14-dim basis, "
                 "RIGHT block = geometry-only control.\n"
                 "The question of gap A: does the CONSTRAINED axis (blue) "
                 "keep the unconstrained band's score?", fontsize=11)
    ax.text(2.5, 0.06, "14-dim basis", fontsize=9, color=INK2, ha="center")
    ax.text(9.1, 0.06, "geometry-only", fontsize=9, color=INK2, ha="center")

    # bottom: one ring rendered through each readout + the fitted s field
    ref = sorted(g[("main14", "cband_norm", "ring")],
                 key=lambda r: r["iou_minmax"])
    if ref:
        ex = ref[len(ref) // 2]
        mask, pred_c, _ = rebuild_pred(ex, suffix)
        panels = [(mask, "target ring")]
        panels.append((pred_c, f"CONSTRAINED axis  IoU={ex['iou_minmax']:.3f}"))
        for ro in ("bandpass", "gauss", "monotone"):
            r2 = next((r for r in g[("main14", ro, "ring")]
                       if r["mask_id"] == ex["mask_id"]), None)
            if r2 is None:
                continue
            _, p2, _ = rebuild_pred(r2, suffix)
            panels.append((p2, f"{ro}  IoU={r2['iou_minmax']:.3f}"))
        for j, (im, ti) in enumerate(panels[:5]):
            axj = fig.add_subplot(gs[1, j])
            axj.imshow(im, cmap="gray", vmin=0, vmax=1)
            axj.set_title(ti, fontsize=8.5)
            axj.set_xticks([])
            axj.set_yticks([])
            axj.grid(False)
    fig.tight_layout()
    fig.savefig(EXP / "viz" / "constrained_axis_evidence.png")
    plt.close(fig)


def constrained_sweep_fig(rows):
    """A-5 (mask level) + A-4 (budget control)."""
    sw = [r for r in rows if r.get("arm", "main").startswith(("sighi_", "M_"))]
    bd = [r for r in rows if r.get("arm") == "budget300"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=140)
    ax = axes[0]
    sig = defaultdict(list)
    for r in sw:
        if r["arm"].startswith("sighi_"):
            sig[float(r["arm"].split("_")[1])].append(r["iou_minmax"])
    xs = sorted(sig)
    if xs:
        ax.plot(xs, [np.median(sig[x]) for x in xs], "o-", color=C1, lw=2)
        for x in xs:
            ax.scatter([x] * len(sig[x]), sig[x], s=8, color=C1, alpha=0.25,
                       linewidths=0)
        ax.axhline(0.90, color=INK2, ls=":", lw=1)
        ax.axvline(0.30, color=C2, ls="--", lw=1.2)
        ax.text(0.31, 0.4, "PLAN R-2 sigma_max", fontsize=8, color=C2,
                rotation=90)
        ax.set_xscale("log")
    else:
        ax.text(0.5, 0.5, "mask-level sigma sweep not run\n(cost; the "
                          "response-level A-5 sweep\ncovers it at 27 cells "
                          "x n=30)", ha="center", va="center", fontsize=9,
                color=INK2, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("sigma_s upper bound")
    ax.set_ylabel("ring soft-IoU (geo-only)")
    ax.set_title("A-5: mask-level sigma_max sweep", fontsize=10)

    ax = axes[1]
    mm = defaultdict(list)
    for r in sw:
        if r["arm"].startswith("M_"):
            mm[int(r["arm"].split("_")[1])].append(r["iou_minmax"])
    base = [r["iou_minmax"] for r in rows
            if r.get("arm") == "main" and r["config"] == "geo6_ring"
            and r["readout"] == "cband_norm" and r["family"] == "ring"]
    if base:
        mm[12] = base
    xs = sorted(mm)
    if len(xs) > 1:
        ax.bar([str(x) for x in xs], [np.median(mm[x]) for x in xs],
               color=[C1 if x == 12 else C3 for x in xs], alpha=0.85)
        for i, x in enumerate(xs):
            ax.annotate(f"{np.median(mm[x]):.3f}",
                        (i, np.median(mm[x]) + 0.01), ha="center",
                        fontsize=8.5)
        ax.axhline(0.90, color=INK2, ls=":", lw=1)
        ax.set_ylim(0, 1.08)
    else:
        ax.text(0.5, 0.5, "mask-level M sweep not run\n(cost; response-level "
                          "A-5 covers M in {6,12,24})", ha="center",
                va="center", fontsize=9, color=INK2, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_xlabel("M (number of s-axis Gaussians)")
    ax.set_ylabel("ring soft-IoU (geo-only)")
    ax.set_title("A-5: mode-count sweep (M=12 = PLAN L56)", fontsize=10)

    ax = axes[2]
    if bd:
        ids = {r["mask_id"] for r in bd}
        base2 = {r["mask_id"]: r["iou_minmax"] for r in rows
                 if r.get("arm") == "main" and r["config"] == "main14"
                 and r["readout"] == "cband_norm" and r["family"] == "ring"
                 and r["mask_id"] in ids}
        pairs = [(base2[r["mask_id"]], r["iou_minmax"]) for r in bd
                 if r["mask_id"] in base2]
        if pairs:
            a, b = zip(*pairs)
            ax.scatter(a, b, s=16, color=C1, alpha=0.7, linewidths=0)
            lo = min(min(a), min(b)) - 0.01
            ax.plot([lo, 1.0], [lo, 1.0], color=INK2, lw=1, ls="--")
            ax.set_xlabel("max_iter=120 (protocol)")
            ax.set_ylabel("max_iter=300 (generous budget)")
            ax.set_title(f"A-4 budget control: median delta = "
                         f"{np.median([y - x for x, y in pairs]):+.4f}",
                         fontsize=10)
    fig.tight_layout()
    fig.savefig(EXP / "viz" / "constrained_axis_sweep.png")
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
    # the wedge prediction window was REFUTED, not passed (review nit 1)
    crit["wedge_mono_expected_055_070"] = {
        "prediction": "0.55-0.70 (bow-tie argument, PLAN L206)",
        "measured_median": med(ious("main14", "monotone", "wedge")),
        "impl_bug_floor": 0.50,
        "floor_ok": med(ious("main14", "monotone", "wedge")) >= 0.50,
        "prediction_refuted": not (0.55 <= med(ious("main14", "monotone",
                                                    "wedge")) <= 0.70),
        "note": "floor_ok = 'no implementation bug'; the pre-registered "
                "prediction window itself is refuted upward -- see REPORT"}

    # ---------------- supplement round: gaps A and B ----------------------
    band_ring = med(ious("main14", "bandpass", "ring"))
    cband_ring = med(ious("main14", "cband_norm", "ring"))
    band_ring_g = med(ious("geo6_ring", "bandpass", "ring"))
    cband_ring_g = med(ious("geo6_ring", "cband_norm", "ring"))
    hard = [r["iou_hardc"] for r in g[("main14", "cband_norm", "ring")]
            if "iou_hardc" in r]

    def paired(cfg, ro_a, ro_b, fam):
        """Median of the per-mask difference a-b over the common masks (the
        constrained cells can have a smaller n than the round-1 cells)."""
        A = {r["mask_id"]: r["iou_minmax"] for r in g[(cfg, ro_a, fam)]}
        B = {r["mask_id"]: r["iou_minmax"] for r in g[(cfg, ro_b, fam)]}
        common = sorted(set(A) & set(B))
        if not common:
            return {"n_common": 0}
        d = [A[m] - B[m] for m in common]
        return {"n_common": len(common),
                "median_a": float(np.median([A[m] for m in common])),
                "median_b": float(np.median([B[m] for m in common])),
                "median_diff": float(np.median(d)),
                "p10_diff": float(np.quantile(d, 0.1)),
                "p90_diff": float(np.quantile(d, 0.9)),
                "frac_a_worse_by_gt_003": float(np.mean([x < -0.03
                                                         for x in d]))}
    crit_supp = {
        "A1_ring_constrained_axis_min090_and_drop_le003": {
            "gate": "median >=0.90 AND (unconstrained - constrained) <=0.03",
            "measured_median_main14": cband_ring,
            "measured_median_geo_only": cband_ring_g,
            "unconstrained_band_main14": band_ring,
            "unconstrained_band_geo_only": band_ring_g,
            "drop_main14": band_ring - cband_ring,
            "drop_geo_only": band_ring_g - cband_ring_g,
            "paired_main14_constrained_minus_unconstrained":
                paired("main14", "cband_norm", "bandpass", "ring"),
            "paired_geo_only_constrained_minus_unconstrained":
                paired("geo6_ring", "cband_norm", "bandpass", "ring"),
            "n_constrained_main14": len(g[("main14", "cband_norm", "ring")]),
            "n_note": "pre-registered n=200; the realized n is stated in the "
                      "REPORT together with the machine-contention reason",
            "pass": bool(cband_ring >= 0.90
                         and (band_ring - cband_ring) <= 0.03)},
        "A3_hard_payload_grouping_drop_le002": {
            "gate": "median drop when c_i rounded to {0,1} <= 0.02",
            "measured_median_soft_c": cband_ring,
            "measured_median_hard_c": med(hard),
            "drop": cband_ring - med(hard),
            "n_on_median": med([r["n_on"] for r in
                                g[("main14", "cband_norm", "ring")]
                                if "n_on" in r]),
            "pass": bool((cband_ring - med(hard)) <= 0.02)},
        "B_gauss_path_runs": {
            "gate": "the delivered `gauss` readout must run (was KeyError "
                    "600/600) and be reported for every family",
            "n_gauss_results": sum(len(v) for (c, ro, f), v in g.items()
                                   if ro == "gauss"),
            "n_gauss_errors": sum(1 for r in errs
                                  if r.get("readout") == "gauss"),
            "ring_strict_gauss_main14": med(ious("main14", "gauss", "ring")),
            "ring_strict_gauss_geo_only": med(ious("geo6_ring", "gauss",
                                                   "ring")),
            "pass": bool(sum(len(v) for (c, ro, f), v in g.items()
                             if ro == "gauss") > 0
                         and sum(1 for r in errs
                                 if r.get("readout") == "gauss") == 0)},
    }
    # A-4 budget control (paired, same masks)
    bd = [r for r in rows if r.get("arm") == "budget300"]
    base2 = {r["mask_id"]: r["iou_minmax"]
             for r in g[("main14", "cband_norm", "ring")]}
    pairs = [(base2[r["mask_id"]], r["iou_minmax"]) for r in bd
             if r["mask_id"] in base2]
    crit_supp["A4_budget_control_gain_lt001"] = {
        "gate": "max_iter 120 -> 300 must gain <0.01 median (otherwise A-1 "
                "is an optimizer-budget statement, not an expressiveness one)",
        "n_pairs": len(pairs),
        "median_gain": med([y - x for x, y in pairs]) if pairs else None,
        "pass": bool(pairs and med([y - x for x, y in pairs]) < 0.01)}

    # full 4-families x {bandpass, gauss} x {unconstrained, constrained} table
    matrix = {}
    for fam in GEO_FAMS + ["semantic", "constant"]:
        matrix[fam] = {ro: stat("main14", ro, fam)
                       for ro in ("monotone", "bandpass", "gauss",
                                  "cband_norm", "cgauss", "cband_unnorm")}
        matrix[fam] = {k: v for k, v in matrix[fam].items() if v["n"]}
    matrix_geo_only = {"ring": {ro: stat("geo6_ring", ro, "ring")
                                for ro in ("monotone", "bandpass", "gauss",
                                           "cband_norm", "cgauss",
                                           "cband_unnorm")}}
    matrix_geo_only["ring"] = {k: v for k, v in matrix_geo_only["ring"].items()
                               if v["n"]}

    # A-5 mask-level sweeps
    sweeps: dict = {"sigma_hi": {}, "M": {}}
    for r in rows:
        arm = r.get("arm", "main")
        if arm.startswith("sighi_"):
            sweeps["sigma_hi"].setdefault(arm.split("_")[1],
                                          []).append(r["iou_minmax"])
        elif arm.startswith("M_"):
            sweeps["M"].setdefault(arm.split("_")[1],
                                   []).append(r["iou_minmax"])
    sweeps = {k: {kk: {"n": len(vv), "median": med(vv), "p10": q(vv, 0.1)}
                  for kk, vv in sorted(v.items())} for k, v in sweeps.items()}
    sweeps["M"]["12_default"] = stat("geo6_ring", "cband_norm", "ring")

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
        # wedge attribution: lin3 -> geo6 -> main14 isolates the quadratic
        # geometric terms from the image/semantic channels (REVIEW section 3.1)
        "wedge_attribution_monotone": {
            "lin3": med(ious("lin3", "monotone", "wedge")),
            "geo6_geometry_only": med(ious("geo6", "monotone", "wedge")),
            "main14": med(ious("main14", "monotone", "wedge")),
            "cubic18": med(ious("cubic18", "monotone", "wedge"))},
        "readout_ladder_ring": {
            cfg: {ro: med(ious(cfg, ro, "ring"))
                  for ro in ("monotone", "gauss", "cgauss", "cband_unnorm",
                             "cband_norm", "bandpass")}
            for cfg in ("main14", "geo6_ring")},
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
    for k, v in summary.items():
        v["readout_meaning"] = READOUT_LABEL.get(k.split("|")[1])
        if k.endswith("|constant"):        # review nit 2
            v["metric_na"] = True
            v["note"] = ("min/max soft-IoU is identically 0 for an all-zero "
                         "target; the constant family is judged by alpha and "
                         "pred_std only")
    metrics = {"criteria": crit,
               "criteria_supplement_gapAB": crit_supp,
               "criteria_matrix_main14": matrix,
               "criteria_matrix_geo_only": matrix_geo_only,
               "A5_sweeps_mask_level": sweeps,
               "ablations": abl,
               "semantic_class5": sem_classes,
               "condition_checks": cond, "per_cell": summary,
               "n_fit_errors": len(errs),
               "n_errors_by_readout": {
                   ro: sum(1 for r in errs if r.get("readout") == ro)
                   for ro in sorted({r.get("readout") for r in errs})},
               "n_results": len(rows),
               "shards_loaded": [sh or "_main" for sh in SHARDS
                                 if (EXP / f"results{suffix}{sh}.jsonl"
                                     ).exists()]}
    resp_path = EXP / "metrics_axis_response.json"
    if resp_path.exists():
        metrics["A2_axis_response"] = json.load(open(resp_path))
    out = EXP / f"metrics{suffix}.json"
    with open(out, "w") as fh:
        json.dump(metrics, fh, indent=2)
    print(json.dumps({"criteria": crit, "supplement": crit_supp}, indent=1))

    # ---- viz ----
    ring_evidence(g, suffix)
    ablation_dims_fig(g)
    if g[("main14", "cband_norm", "ring")]:
        constrained_evidence_fig(g, suffix)
    if any(r.get("arm", "main") != "main" for r in rows):
        constrained_sweep_fig(rows)
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
    # supplement: success/failure of the CONSTRAINED axis itself
    for fam in GEO_FAMS:
        sel = sorted(g[("main14", "cband_norm", fam)],
                     key=lambda r: r["iou_minmax"])
        if not sel:
            continue
        panel(sel[-2:][::-1], suffix,
              EXP / "viz" / f"success_constrained_{fam}.png",
              f"E2 gap A success cases (constrained M=12 axis): {fam}")
        panel(sel[:2], suffix,
              EXP / "viz" / f"failure_constrained_{fam}.png",
              f"E2 gap A failure cases (constrained M=12 axis): {fam}")
    sel = sorted(g[("main14", "gauss", "ring")],
                 key=lambda r: r["iou_minmax"])
    if sel:
        panel(sel[-1:] + sel[:1], suffix,
              EXP / "viz" / "failure_gauss_ring.png",
              "E2 gap B: strict single Gaussian on rings "
              "(best case top, worst case bottom) -- no plateau")
    print("analyze done ->", out)


if __name__ == "__main__":
    main()
