"""G3 aggregation + figures (Gate D3).

  python experiments/G3_collapse_20260803/analyze_g3.py \
      --runs-dir experiments/G3_collapse_20260803/runs \
      --out experiments/G3_collapse_20260803

Produces:
  metrics.json                    aggregated criteria-vs-measured table
  viz/sigma_s_trajectory.png      THE two-arm sigma_s figure (paper appendix)
  viz/s_sensitivity.png           s-sensitivity vs step
  viz/delta_shuffle.png           Delta_shuffle / Delta_const vs step
  viz/mu_s_spread.png             std(mu_s) vs step (R-3 diversity statistic)
  viz/success_*.png viz/failure_*.png   rendered examples, collapsed & not
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools", "harness"))

import matplotlib                                            # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
import torch                                                 # noqa: E402

from model.glut_repro import data_construct as dc            # noqa: E402
from model.glut_repro.model4d_naive import (                 # noqa: E402
    GLUT4D, GAMMA_MU, SIGMA_S_MAX, SIGMA_S_MIN)
from model.glut_repro.train_g3 import Renderer               # noqa: E402
from metrics import masked_psnr, psnr_full                   # noqa: E402

# validated 2-slot categorical palette (dataviz skill, light surface,
# CVD dE 24.7 / normal dE 33.6 -> PASS on all six checks)
C_NAIVE, C_ANCHORED = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
ARM_COLOR = {"naive": C_NAIVE, "anchored": C_ANCHORED}
ARM_LABEL = {"naive": "naive 4D  (mu_s learnable, sigma_s free)",
             "anchored": "R-2 min  (mu_s K=6 anchored, sigma_s bounded)"}
DATASETS = ["fixed", "tiered", "mixed"]
DS_TITLE = {
    "fixed": "fixed  ($s$ fully sufficient)",
    "tiered": "tiered  ($s$ partially sufficient)",
    "mixed": "mixed  ($s$ nearly worthless)",
}
RUN_RE = re.compile(r"^(fixed|tiered|mixed)_(naive|anchored)_s(\d+)$")


def style(ax, xlabel: str, ylabel: str, title: str = "") -> None:
    ax.set_facecolor("#fcfcfb")
    ax.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.set_xlabel(xlabel, color="#52514e", fontsize=9)
    ax.set_ylabel(ylabel, color="#52514e", fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, pad=8)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_runs(runs_dir: str) -> dict:
    runs = {}
    for name in sorted(os.listdir(runs_dir)):
        m = RUN_RE.match(name)
        d = os.path.join(runs_dir, name)
        if not m or not os.path.isdir(d):
            continue
        tp, mp = os.path.join(d, "trace.jsonl"), os.path.join(d, "metrics.json")
        if not os.path.exists(tp):
            continue
        trace = [json.loads(x) for x in open(tp) if x.strip()]
        met = json.load(open(mp)) if os.path.exists(mp) else None
        runs[name] = {"dataset": m.group(1), "arm": m.group(2),
                      "seed": int(m.group(3)), "trace": trace,
                      "metrics": met, "dir": d}
    return runs


def series(runs: dict, dataset: str, arm: str, path: list):
    """(steps, vals[seed, step]) for trace[path], or None if the arm is absent.

    Returns a single Optional (not a tuple of Optionals) so one `is None`
    check narrows both halves for the type checker and for the reader."""
    sel = [r for r in runs.values()
           if r["dataset"] == dataset and r["arm"] == arm]
    if not sel:
        return None
    n = min(len(r["trace"]) for r in sel)
    steps = [sel[0]["trace"][i]["step"] for i in range(n)]
    vals = []
    for r in sel:
        v = []
        for i in range(n):
            o = r["trace"][i]
            for k in path:
                o = o[k]
            v.append(o)
        vals.append(v)
    return np.array(steps), np.array(vals, dtype=np.float64)


def _need(runs: dict, dataset: str, arm: str, path: list):
    got = series(runs, dataset, arm, path)
    if got is None:
        raise KeyError(f"no {dataset}/{arm} runs for {path}")
    return got


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_sigma_s(runs: dict, out: str) -> None:
    """The deliverable figure: both arms' sigma_s trajectories, side by side."""
    present = [d for d in DATASETS
               if any(r["dataset"] == d for r in runs.values())]
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.0),
                             squeeze=False)
    for ax, ds in zip(axes[0], present):
        for arm in ("naive", "anchored"):
            got = series(runs, ds, arm, ["sigma_s", "q50"])
            if got is None:
                continue
            st, q50 = got
            _, q05 = _need(runs, ds, arm, ["sigma_s", "q05"])
            _, q95 = _need(runs, ds, arm, ["sigma_s", "q95"])
            _, mx = _need(runs, ds, arm, ["sigma_s", "max"])
            c = ARM_COLOR[arm]
            ax.fill_between(st, q05.mean(0), q95.mean(0), color=c, alpha=0.16,
                            lw=0, zorder=2)
            ax.plot(st, q50.mean(0), color=c, lw=2.0, zorder=4,
                    label=f"{ARM_LABEL[arm]} — median")
            ax.plot(st, mx.mean(0), color=c, lw=1.2, ls=":", zorder=3,
                    label=f"{ARM_LABEL[arm]} — max")
        for y, lbl in ((SIGMA_S_MIN, "R-2 lower 0.025"),
                       (SIGMA_S_MAX, "R-2 upper 0.30")):
            ax.axhline(y, color=MUTED, lw=1.0, ls="--", zorder=1)
            ax.text(ax.get_xlim()[1], y, f" {lbl}", color=MUTED, fontsize=7,
                    va="bottom", ha="right")
        ax.set_yscale("log")
        style(ax, "training step", r"$\sigma_s$  (marginal std of the s axis)",
              DS_TITLE[ds])
    # lower-left: the R-2 bound annotations live at the right edge, and with a
    # log axis the space below the trajectories is always empty
    axes[0][0].legend(loc="lower left", fontsize=7, frameon=False,
                      labelcolor=INK)
    fig.suptitle("G3 / Gate D3 — $\\sigma_s$ trajectory under a PURE "
                 "reconstruction loss (shaded: q05–q95 across the 32 "
                 "Gaussians; mean over seeds)", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out, "sigma_s_trajectory.png"), dpi=150,
                facecolor="#fcfcfb")
    plt.close(fig)


def fig_trace(runs: dict, out: str, path: list, ylabel: str, title: str,
              fname: str, hlines=(), logy: bool = False) -> None:
    present = [d for d in DATASETS
               if any(r["dataset"] == d for r in runs.values())]
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 3.8),
                             squeeze=False)
    for ax, ds in zip(axes[0], present):
        for arm in ("naive", "anchored"):
            got = series(runs, ds, arm, path)
            if got is None:
                continue
            st, v = got
            c = ARM_COLOR[arm]
            if v.shape[0] > 1:
                ax.fill_between(st, v.min(0), v.max(0), color=c, alpha=0.16,
                                lw=0, zorder=2)
            ax.plot(st, v.mean(0), color=c, lw=2.0, zorder=4,
                    label=ARM_LABEL[arm])
        for y, lbl, col in hlines:
            ax.axhline(y, color=col, lw=1.0, ls="--", zorder=1)
            ax.text(ax.get_xlim()[1], y, f" {lbl}", color=col, fontsize=7,
                    va="bottom", ha="right")
        if logy:
            ax.set_yscale("log")
        style(ax, "training step", ylabel, DS_TITLE[ds])
    axes[0][0].legend(loc="best", fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(out, fname), dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------------------
# rendered examples
# ---------------------------------------------------------------------------

def render_examples(runs: dict, out: str, n_each: int = 2,
                    device: str = "cuda") -> list:
    """For each run flagged as the collapsed / non-collapsed exemplar, render
    best (success_) and worst (failure_) val pairs."""
    made = []
    picks = []
    for name, r in runs.items():
        if r["seed"] != 0 or r["metrics"] is None:
            continue
        picks.append((name, r))
    va = dc.load_pairs("val", ("L1", "L4"))
    for name, r in picks:
        ck = os.path.join(r["dir"], "ckpt.pt")
        if not os.path.exists(ck):
            continue
        blob = torch.load(ck, map_location=device, weights_only=False)
        model = GLUT4D(blob["n_gaussians"], arm=blob["arm"]).to(device)
        model.load_state_dict(blob["state_dict"])
        spec = {"fixed": dc.FIXED_SPEC, "tiered": dc.tiered_spec,
                "mixed": None}[r["dataset"]]
        rend = Renderer(model, device)
        rows = []
        for p in va:
            x, y, s32, _ = dc.read_pair(p, fixed_spec=spec)
            pred = rend(x, s32)
            mp = masked_psnr(pred, y, dc.read_mask(p))
            rows.append({"uid": p.uid, "x": x, "y": y, "s32": s32,
                         "pred": pred, "psnr_in": mp["psnr_in"],
                         "psnr_full": psnr_full(pred, y),
                         "ident_in": masked_psnr(x, y, dc.read_mask(p))["psnr_in"]})
        rows = [q for q in rows if np.isfinite(q["psnr_in"])]
        rows.sort(key=lambda q: q["psnr_in"])
        verdict = r["metrics"]["verdict"]
        for kind, subset in (("failure", rows[:n_each]),
                             ("success", rows[-n_each:][::-1])):
            for q in subset:
                fn = f"{kind}_{name}_{q['uid']}.png"
                _panel(q, os.path.join(out, fn),
                       f"{name}  [{verdict}]  {q['uid']}")
                made.append(fn)
    return made


def _panel(q: dict, path: str, title: str) -> None:
    s_full = dc.s32_to_full(q["s32"], q["x"].shape[:2])
    err = np.abs(q["pred"] - q["y"]).mean(-1)
    fig, ax = plt.subplots(1, 5, figsize=(19, 3.8))
    for a, im, t in (
            (ax[0], q["x"], "input $I_{in}$"),
            (ax[1], q["y"], "target $I_{tar}$"),
            (ax[2], q["pred"], "prediction $f(x,s)$"),
            (ax[3], s_full, "oracle $s$ (32x32 -> bilinear)")):
        a.imshow(im, cmap=None if im.ndim == 3 else "magma",
                 vmin=None if im.ndim == 3 else 0,
                 vmax=None if im.ndim == 3 else 1)
        a.set_title(t, fontsize=9, color=INK)
        a.axis("off")
    im = ax[4].imshow(err, cmap="magma", vmin=0, vmax=max(err.max(), 1e-3))
    ax[4].set_title(f"|error|  in-mask PSNR {q['psnr_in']:.2f} dB "
                    f"(identity {q['ident_in']:.2f})", fontsize=9, color=INK)
    ax[4].axis("off")
    fig.colorbar(im, ax=ax[4], fraction=0.03)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=105, facecolor="#fcfcfb")
    plt.close(fig)


# ---------------------------------------------------------------------------

def aggregate(runs: dict) -> dict:
    by = defaultdict(list)
    for r in runs.values():
        if r["metrics"]:
            by[(r["dataset"], r["arm"])].append(r["metrics"])
    table = []
    for (ds, arm), ms in sorted(by.items()):
        def col(f):
            return [f(m) for m in ms]
        dsh = col(lambda m: m["metrics"]["delta_shuffle"])
        dc_ = col(lambda m: m["metrics"]["delta_const"])
        table.append({
            "dataset": ds, "arm": arm, "n_seeds": len(ms),
            "delta_shuffle_mean": float(np.mean(dsh)),
            "delta_shuffle_min": float(np.min(dsh)),
            "delta_shuffle_max": float(np.max(dsh)),
            "delta_const_mean": float(np.mean(dc_)),
            "s_sensitivity": float(np.mean(
                col(lambda m: m["metrics"]["s_sensitivity"]))),
            "sigma_s_q50": float(np.mean(
                col(lambda m: m["metrics"]["sigma_s"]["q50"]))),
            "sigma_s_max": float(np.mean(
                col(lambda m: m["metrics"]["sigma_s"]["max"]))),
            "sigma_s_frac_ge_r2_max": float(np.mean(
                col(lambda m: m["metrics"]["sigma_s"]["frac_ge_r2_max"]))),
            "mu_s_std": float(np.mean(
                col(lambda m: m["metrics"]["mu_s"]["std"]))),
            "psnr_in": float(np.mean(col(lambda m: m["metrics"]["psnr_in"]))),
            "psnr_full": float(np.mean(
                col(lambda m: m["metrics"]["psnr_full"]))),
            "identity_psnr_in": float(np.mean(
                col(lambda m: m["identity_baseline"]["psnr_in"]))),
            "verdicts": col(lambda m: m["verdict"]),
        })
    return {
        "exp": "G3_collapse", "gate": "D3",
        "criteria": {
            "delta_shuffle_collapsed_below": 0.3,
            "delta_shuffle_no_collapse_at_or_above": 3.0,
            "m3_sigma_s_frac_at_upper_red_line": 0.80,
            "r3_gamma_mu": GAMMA_MU,
        },
        "rows": table,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    viz = os.path.join(args.out, "viz")
    os.makedirs(viz, exist_ok=True)

    runs = load_runs(args.runs_dir)
    print(f"[g3] {len(runs)} runs: {sorted(runs)}")
    if not runs:
        raise SystemExit("no runs found")

    fig_sigma_s(runs, viz)
    fig_trace(runs, viz, ["s_sensitivity"],
              r"$E\,\|f(x,s+\delta)-f(x,s)\|_2$   ($\delta=0.1$)",
              "G3 — s-sensitivity on held-out pixels "
              "(0 = the s axis has been switched off)",
              "s_sensitivity.png", logy=True)
    fig_trace(runs, viz, ["delta_shuffle"], r"$\Delta_{shuffle}$  (dB)",
              "G3 — $\\Delta_{shuffle}$ vs step "
              "(band: min–max across seeds)", "delta_shuffle.png",
              hlines=((0.3, "collapse red line 0.3 dB", "#d03b3b"),
                      (3.0, "no-collapse 3 dB", "#0ca30c")))
    fig_trace(runs, viz, ["delta_const"], r"$\Delta_{const}$  (dB)",
              "G3 — $\\Delta_{const}$ vs step", "delta_const.png",
              hlines=((0.05, "collapse red line 0.05 dB", "#d03b3b"),))
    fig_trace(runs, viz, ["mu_s", "std"], r"$\mathrm{std}(\mu_s)$",
              "G3 — cross-Gaussian spread of $\\mu_s$ "
              "(R-3 diversity statistic, reported not optimised)",
              "mu_s_spread.png",
              hlines=((GAMMA_MU, f"R-3 gamma_mu {GAMMA_MU}", MUTED),))
    print("[g3] figures written")

    agg = aggregate(runs)
    if not args.no_render:
        agg["viz"] = render_examples(runs, viz, device=args.device)
        print(f"[g3] {len(agg['viz'])} rendered examples")
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(agg, f, indent=1)
    for r in agg["rows"]:
        print(f"  {r['dataset']:7s} {r['arm']:9s} n={r['n_seeds']} "
              f"D_shuf {r['delta_shuffle_mean']:+7.3f} "
              f"sig_s q50 {r['sigma_s_q50']:.4f} max {r['sigma_s_max']:.4f} "
              f"sens {r['s_sensitivity']:.5f} "
              f"psnr_in {r['psnr_in']:.2f} ({r['identity_psnr_in']:.2f}) "
              f"{set(r['verdicts'])}")
    print("G3_ANALYZE_DONE")


if __name__ == "__main__":
    main()
