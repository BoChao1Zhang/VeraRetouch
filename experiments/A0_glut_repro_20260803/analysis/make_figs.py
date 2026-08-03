"""Evidence figures for the A0 anchor-gap report.

fig_task1_arms.png    paired per-arm PSNR + gradient-norm traces (bug hunt)
fig_task2_hypA.png    A0-rec vs E1 engine, paired per LUT + capacity sweep
fig_task2_hypC.png    corpus difficulty vs achieved PSNR, NILUT reference
"""

from __future__ import annotations

import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
VIZ = os.path.join(EXP, "viz")
ANCHOR = 45.47


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


# ---------------------------------------------------------------- task 1
def fig_task1() -> None:
    blobs = [b for b in (jload("ablate/ablate_20ep.json"),
                         jload("ablate/ablate_fix.json"),
                         jload("ablate/ablate_opacity.json")) if b]
    if not blobs:
        return
    res = {}
    for b in blobs:
        res.update(b["results"])
    order = [a for a in ("rec", "hc", "hc_w1", "hc_fix", "hc_cal", "sparse",
                         "mining", "full", "full_fix", "g0", "g0_full",
                         "opac_sig", "opac_sig_g0") if a in res]
    base = res["rec"]["psnr_float_mean"]

    fig, ax = plt.subplots(1, 3, figsize=(21, 5.6))
    vals = [res[a]["psnr_float_mean"] for a in order]
    cols = ["#4C72B0" if a == "rec" else
            ("#C44E52" if res[a]["psnr_float_mean"] < base - 0.15 else
             ("#55A868" if res[a]["psnr_float_mean"] > base + 0.15 else "#8C8C8C"))
            for a in order]
    ax[0].bar(range(len(order)), vals, color=cols)
    ax[0].axhline(base, ls="--", c="#4C72B0", lw=1,
                  label=f"rec baseline {base:.2f} dB")
    ax[0].axhline(ANCHOR, ls=":", c="k", lw=1, label=f"GLUT anchor {ANCHOR}")
    for i, a in enumerate(order):
        d = res[a]["psnr_float_mean"] - base
        ax[0].text(i, vals[i] + 0.4, f"{d:+.2f}", ha="center", fontsize=8)
    ax[0].set_xticks(range(len(order)))
    ax[0].set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax[0].set_ylabel("held-out PSNR (dB), 9 paired LUTs")
    ax[0].set_title("task 1 — ingredient isolation (delta vs rec)")
    ax[0].set_ylim(min(vals) - 2, max(max(vals), ANCHOR) + 2.5)
    ax[0].legend(fontsize=8)

    for a in order:
        tr = res[a]["health"].get("gnorm_trace_every100")
        if not tr:
            continue
        ax[1].semilogy(np.arange(len(tr)) * 100, np.maximum(tr, 1e-12),
                       lw=0.9, label=a)
    ax[1].set_xlabel("step")
    ax[1].set_ylabel("total gradient L2 norm")
    ax[1].set_title("gradient norm (log) — L_hc arms sit 3-4 decades higher")
    ax[1].legend(fontsize=7, ncol=2)

    alive = [res[a]["health"]["alive_frac_mean"] for a in order]
    opac0 = [res[a]["health"]["opacity_frac_below_0.05"] for a in order]
    xs = np.arange(len(order))
    ax[2].bar(xs - 0.2, alive, 0.4, label="alive_frac (weight-mass)")
    ax[2].bar(xs + 0.2, opac0, 0.4, label="frac opacity < 0.05")
    ax[2].axhline(0.8, ls=":", c="r", lw=1, label="PLAN red flag 0.8N")
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(order, rotation=35, ha="right", fontsize=9)
    ax[2].set_title("effective primitives / opacity collapse")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "fig_task1_arms.png"), dpi=110)
    plt.close(fig)
    print("wrote fig_task1_arms.png")


# ---------------------------------------------------------------- task 2a/b
def fig_task2a() -> None:
    rec = {r["lut_id"]: r["psnr_float"] for r in
           (jlines("runs/rec/per_lut.jsonl") or [])}
    if not rec:
        return
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.6))

    rows = jlines("hypA/e1_on_a0_3k.jsonl")
    if rows:
        a = np.array([[rec[r["lut_id"]], r["psnr_float"]] for r in rows
                      if r["lut_id"] in rec])
        ax[0].scatter(a[:, 0], a[:, 1], s=22, alpha=.8)
        lo, hi = a.min() - 1, a.max() + 1
        ax[0].plot([lo, hi], [lo, hi], "k--", lw=1)
        ax[0].axvline(ANCHOR, ls=":", c="r", lw=1)
        ax[0].axhline(ANCHOR, ls=":", c="r", lw=1)
        ax[0].set_xlabel("A0 rec recipe (Adam 1e-3, 20 ep, GLUT init)")
        ax[0].set_ylabel("E1 direct-overfit engine (kmeans init, 3k x 8192)")
        ax[0].set_title(f"hyp (a): same 75 LUTs, two recipes\n"
                        f"means {a[:,0].mean():.2f} vs {a[:,1].mean():.2f} dB "
                        f"(paired delta {np.mean(a[:,1]-a[:,0]):+.2f})")

    caps = []
    for tag, n in (("e1_on_a0_3k", 32), ("e1_on_a0_N48", 48),
                   ("e1_on_a0_N64", 64)):
        s = jload(f"hypA/{tag}.json")
        if s:
            caps.append((s["n_gaussians"], s["psnr_float_mean"]))
    e1_33 = [(8, 36.40), (16, 39.75), (24, 41.50), (32, 42.71), (48, 44.33),
             (64, 45.38), (96, 46.75), (128, 47.70)]
    paper = [(8, 37.01), (16, 41.50), (32, 45.47), (64, 48.42), (128, 50.31)]
    if caps:
        caps.sort()
        ax[1].plot(*zip(*caps), "o-", label="ours: A0 75x 64^3 prod presets")
    ax[1].plot(*zip(*e1_33), "s--", label="ours: E1 400x 33^3 prod presets")
    ax[1].plot(*zip(*paper), "^:", c="k", label="GLUT paper (their corpus)")
    ax[1].axhline(ANCHOR, ls=":", c="r", lw=1, label="anchor 45.47")
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("N Gaussians")
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title("capacity curve — same slope, corpus-shifted level")
    ax[1].legend(fontsize=8)

    ep, y = [], []
    for e, tag in ((20, "ablate/ablate_20ep.json"), (40, "ablate/rec_40ep.json"),
                   (60, "ablate/rec_60ep.json")):
        b = jload(tag)
        if b and "rec" in b["results"]:
            ep.append(e)
            y.append(b["results"]["rec"]["psnr_float_mean"])
    if ep:
        ax[2].plot(ep, y, "o-")
        for e, v in zip(ep, y):
            ax[2].text(e, v + 0.05, f"{v:.2f}", ha="center", fontsize=9)
        ax[2].axhline(ANCHOR, ls=":", c="r", lw=1, label="anchor 45.47")
        ax[2].set_xlabel("epochs")
        ax[2].set_ylabel("held-out PSNR (dB), 9 paired LUTs")
        ax[2].set_title("hyp (b): epoch budget")
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "fig_task2_hypAB.png"), dpi=110)
    plt.close(fig)
    print("wrote fig_task2_hypAB.png")


# ---------------------------------------------------------------- task 2c
def fig_task2c() -> None:
    cd = jload("analysis/corpus_difficulty.json")
    if not cd:
        return
    rows = [r for r in cd["per_lut"] if r["rec_psnr"] is not None]
    nil = cd["nilut_LUT01"]
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.6))
    for j, key in enumerate(("affine_psnr", "curv_mean", "lip_p999")):
        v = np.array([r[key] for r in rows])
        y = np.array([r["rec_psnr"] for r in rows])
        ax[j].scatter(v, y, s=22, alpha=.75, label="our 75 (64^3 prod presets)")
        A = np.stack([v, np.ones_like(v)], 1)
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        xs = np.linspace(min(v.min(), nil[key]), max(v.max(), nil[key]), 50)
        ax[j].plot(xs, c[0] * xs + c[1], "k-", lw=1,
                   label=f"OLS  r={np.corrcoef(v, y)[0,1]:+.2f}")
        ax[j].axhline(ANCHOR, ls=":", c="r", lw=1, label="anchor 45.47 dB")
        ax[j].scatter([nil[key]], [c[0] * nil[key] + c[1]], marker="*", s=260,
                      c="#DD8452", zorder=5,
                      label="NILUT LUT01 (GLUT's declared source family)")
        ax[j].scatter([nil[key]], [49.02], marker="P", s=140, c="#C44E52",
                      zorder=5, label="NILUT LUT01 measured (49.02 dB)")
        ax[j].set_xlabel(key)
        ax[j].set_ylabel("A0 rec held-out PSNR (dB)")
        ax[j].legend(fontsize=7.5)
    ax[0].set_title("hyp (c): corpus difficulty explains the level")
    fig.tight_layout()
    fig.savefig(os.path.join(VIZ, "fig_task2_hypC.png"), dpi=110)
    plt.close(fig)
    print("wrote fig_task2_hypC.png")


if __name__ == "__main__":
    os.makedirs(VIZ, exist_ok=True)
    fig_task1()
    fig_task2a()
    fig_task2c()
