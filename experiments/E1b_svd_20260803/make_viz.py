"""E1b viz: rank-dE00 percentile curves, singular spectrum, success/failure
LUT reconstructions at a probe rank."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path("/home/bc/VeraRetouch")
sys.path.insert(0, str(REPO / "tools" / "cube"))
import cubelib  # noqa: E402

OUT = REPO / "experiments/E1b_svd_20260803"
NPY_DIR = Path("/var/cache/veradata/dcube/npy33")
PROBE_R = 48  # middle of pre-registered knee window [32, 64]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e7e6e2"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"  # blue / orange / aqua

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12,
})


def fig_rank_curve(m):
    cur = m["variants"]["plain"]["curves"]
    r = np.array(cur["rank"])
    fig, ax = plt.subplots(figsize=(8.4, 5.2), dpi=150)
    for key, col, lab in [("p50", C1, "p50"), ("p90", C2, "p90"),
                          ("p99", C3, "p99")]:
        ax.plot(r, cur[key], color=col, lw=2, marker="o", ms=4, label=lab)
        ax.annotate(lab, (r[-1], cur[key][-1]), textcoords="offset points",
                    xytext=(8, 0), color=col, fontsize=10, fontweight="bold")
    ax.axhline(1.0, color=INK2, lw=1, ls="--")
    ax.axhline(2.0, color=INK2, lw=1, ls=":")
    ax.text(r[0], 1.03, "gate p90 < 1.0", color=INK2, fontsize=9)
    ax.text(r[0], 2.06, "gate p99 < 2.0", color=INK2, fontsize=9)
    ax.axvspan(32, 64, color="#f0efec", zorder=0)
    ax.text(45, ax.get_ylim()[1] * 0.55, "pre-registered\nknee window\n32-64",
            color=INK2, fontsize=9, ha="center")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(r[[0, 2, 3, 4, 5, 7, 9, 11, 13, 14, 16, 17, 18, 19]])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("SVD rank r (log2)")
    ax.set_ylabel("cross-LUT percentile of per-LUT mean dE00 (log)")
    ax.set_title("E1b: truncated-SVD reconstruction error vs rank "
                 f"(N={m['n_presets']} production presets, 33^3 grid)")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "viz" / "rank_vs_de00.png")
    plt.close(fig)


def fig_spectrum(m):
    sv = np.array(m["variants"]["plain"]["singular_values"])
    energy = np.cumsum(sv ** 2) / np.sum(sv ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=150)
    ax = axes[0]
    ax.plot(np.arange(1, len(sv) + 1), sv, color=C1, lw=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("index")
    ax.set_ylabel("singular value")
    ax.set_title("Singular spectrum (log-log)")
    ax = axes[1]
    ax.plot(np.arange(1, len(sv) + 1), energy, color=C2, lw=2)
    for tau, r in m["variants"]["plain"]["rank_at_energy"].items():
        ax.axvline(int(r), color=GRID, lw=1)
        ax.text(int(r), 0.35 + 0.13 * list(
            m["variants"]["plain"]["rank_at_energy"]).index(tau),
            f"{float(tau):.1%} @ r={r}", color=INK2, fontsize=8, rotation=0)
    ax.set_xscale("log")
    ax.set_xlabel("rank")
    ax.set_ylabel("cumulative energy")
    ax.set_title("Cumulative energy")
    fig.tight_layout()
    fig.savefig(OUT / "viz" / "singular_spectrum.png")
    plt.close(fig)


def fig_cases(m):
    """Success (best-3) / failure (worst-3) LUTs at PROBE_R."""
    z = np.load(OUT / "per_lut_mean_de00.npz", allow_pickle=True)
    ids = z["ids"]
    de = z[f"plain_r{PROBE_R}"]
    order = np.argsort(de)
    picks = [("success", order[:3]), ("failure", order[-3:][::-1])]

    hald = np.load("/var/cache/veradata/dcube/hald/hald_train_1024x2048.npy")
    hald_small = hald[::4, ::4]  # 256x512

    used = [line.strip() for line in open(
        REPO / "experiments/tooling-wave1/cube/inventory/used_presets.txt")
        if line.strip()]
    slug2i = {cubelib.preset_slug(p): i for i, p in enumerate(used)}
    X = np.stack([np.load(NPY_DIR / f"{i}.npy").reshape(-1)
                  for i in ids]).astype(np.float64)
    Xt = torch.from_numpy(X)
    G = Xt @ Xt.T
    _, U = torch.linalg.eigh(G)
    U = torch.flip(U, dims=[1])[:, :PROBE_R]
    Xr = (U @ (U.T @ Xt)).clamp(0, 1).numpy()

    for tag, idxs in picks:
        fig, axes = plt.subplots(len(idxs), 3, figsize=(12, 2.6 * len(idxs)),
                                 dpi=140)
        for row, i in enumerate(idxs):
            table_o = X[i].reshape(33, 33, 33, 3).astype(np.float32)
            table_r = Xr[i].reshape(33, 33, 33, 3).astype(np.float32)
            img_o = cubelib.apply_lut_grid_sample(table_o, hald_small)
            img_r = cubelib.apply_lut_grid_sample(table_r, hald_small)
            demap = cubelib.delta_e00(img_o, img_r)
            axes[row, 0].imshow(np.clip(img_o, 0, 1))
            axes[row, 0].set_title(f"{ids[i]}  original", fontsize=9)
            axes[row, 1].imshow(np.clip(img_r, 0, 1))
            axes[row, 1].set_title(f"recon r={PROBE_R}  "
                                   f"mean dE00={de[i]:.2f}", fontsize=9)
            im = axes[row, 2].imshow(demap, cmap="Blues", vmin=0,
                                     vmax=max(4, demap.max()))
            axes[row, 2].set_title("dE00 map (hald)", fontsize=9)
            fig.colorbar(im, ax=axes[row, 2], shrink=0.8)
            for ax in axes[row]:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.grid(False)
        fig.suptitle(f"E1b {tag} cases @ r={PROBE_R} "
                     "(hald preview of original vs reconstructed LUT)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT / "viz" / f"{tag}_r{PROBE_R}.png")
        plt.close(fig)
    # persist case list
    with open(OUT / "viz" / "cases.json", "w") as fh:
        json.dump({tag: [{"id": str(ids[i]),
                          "mean_de00": float(de[i]),
                          "row_in_used_presets": slug2i.get(str(ids[i]))}
                         for i in idxs] for tag, idxs in picks}, fh, indent=2)


def main():
    m = json.load(open(OUT / "metrics.json"))
    (OUT / "viz").mkdir(exist_ok=True)
    fig_rank_curve(m)
    fig_spectrum(m)
    fig_cases(m)
    print("viz done")


if __name__ == "__main__":
    main()
