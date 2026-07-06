# E4: layer-wise hidden-norm comparison — visual vs text vs retouch tokens.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from common import RESULTS, DUMPS, load_manifest, TOKENS
import plotstyle as ps
import matplotlib.pyplot as plt

DUMP_DIR = os.path.join(DUMPS, "baseline")


def main():
    rows = load_manifest()
    vis, txt, tok = [], [], {t: [] for t in TOKENS}
    for r in rows:
        p = os.path.join(DUMP_DIR, r["key"] + ".npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        vis.append(d["vis_norm"]); txt.append(d["txt_norm"])
        for t in TOKENS:
            tok[t].append(np.linalg.norm(d[f"lat_{t}"].astype(np.float32), axis=1))
    vis = np.stack(vis); txt = np.stack(txt)
    tokm = {t: np.stack(v) for t, v in tok.items()}
    L = vis.shape[1]
    layers = np.arange(L)

    df = pd.DataFrame({"layer": layers,
                       "vis_norm_mean": vis.mean(0), "vis_norm_std": vis.std(0),
                       "txt_norm_mean": txt.mean(0), "txt_norm_std": txt.std(0),
                       "vis_txt_ratio": vis.mean(0) / txt.mean(0)})
    for t in TOKENS:
        df[f"{t}_norm_mean"] = tokm[t].mean(0)
        df[f"{t}_txt_ratio"] = tokm[t].mean(0) / txt.mean(0)
    df.to_csv(os.path.join(RESULTS, "e4_norms.csv"), index=False)

    ratio = df["vis_txt_ratio"].values
    # exclude embedding layer 0 (pre-transformer scale artifact) from the verdict
    peak_ratio, peak_l = float(ratio[1:].max()), int(ratio[1:].argmax()) + 1
    med_ratio = float(np.median(ratio[1:]))
    imbal = "imbalanced" if peak_ratio > 3 else "no strong imbalance"

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax.plot(layers, vis.mean(0), color=ps.C_VIOLET, label="visual tokens (prompt)")
    ax.fill_between(layers, vis.mean(0)-vis.std(0), vis.mean(0)+vis.std(0), color=ps.C_VIOLET, alpha=0.15, lw=0)
    ax.plot(layers, txt.mean(0), color=ps.INK2, label="text tokens (prompt)")
    ax.fill_between(layers, txt.mean(0)-txt.std(0), txt.mean(0)+txt.std(0), color=ps.INK2, alpha=0.15, lw=0)
    for t in TOKENS:
        ax.plot(layers, tokm[t].mean(0), color=ps.TOKEN_COLORS[t], label=ps.TOKEN_LABELS[t], lw=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("hidden_states index (0=emb, 24=final)")
    ax.set_ylabel("mean L2 norm (log)")
    ax.legend(fontsize=8)
    ax.set_title("per-layer hidden norms")

    ax2.plot(layers, ratio, color=ps.C_VIOLET, marker="o", markersize=3.5, label="visual / text")
    for t in TOKENS:
        ax2.plot(layers, df[f"{t}_txt_ratio"], color=ps.TOKEN_COLORS[t], lw=1.4,
                 label=f"{ps.TOKEN_LABELS[t]} / text")
    ax2.axhline(3.0, color=ps.C_HILITE, lw=1.2, ls="--")
    ax2.text(0.3, 3.05, "3x imbalance threshold", color=ps.C_HILITE, fontsize=8)
    ax2.set_xlabel("hidden_states index")
    ax2.set_ylabel("norm ratio vs text tokens")
    ax2.legend(fontsize=8)
    ax2.set_title("norm ratios")

    ps.conclusion_title(fig,
        f"E4: visual/text norm ratio peaks at {peak_ratio:.1f}x (L{peak_l}), median {med_ratio:.1f}x — {imbal}",
        sub=f"N={vis.shape[0]} samples; shaded = +-1 std across samples; verdict excludes embedding layer L0")
    ps.save(fig, os.path.join(RESULTS, "e4_norm_ratio.png"))

    json.dump(dict(peak_ratio=peak_ratio, peak_layer=peak_l, median_ratio=med_ratio,
                   imbalance=peak_ratio > 3),
              open(os.path.join(RESULTS, "e4_summary.json"), "w"), indent=1)
    print("E4 done:", peak_ratio, "at layer", peak_l, "| median", med_ratio)


if __name__ == "__main__":
    main()
