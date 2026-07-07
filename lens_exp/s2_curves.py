# S2 (round 3): grounding-per-layer curve (E7 head IoU aggregated over heads) vs
# perception-per-layer curve (S1 ridge probe R^2). Zero-cost aggregation of R2 artifacts.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

from common import TOKENS
from common_r2 import RESULTS_R2
from common_r3 import RESULTS_R3
import plotstyle as ps
import matplotlib.pyplot as plt


def main():
    head_df = pd.read_csv(os.path.join(RESULTS_R2, "e7_head_iou.csv"))
    tr = head_df[head_df.split == "train"]
    # per (token, layer, head): median IoU over samples; then aggregate over heads
    med = tr.groupby(["token", "layer", "head"]).iou.median().reset_index()
    g_best = med.groupby("layer").iou.max()          # best head per layer (any token)
    g_mean = med.groupby("layer").iou.mean()         # mean over heads+tokens
    chance = float(tr.chance.median())

    probe = pd.read_csv(os.path.join(RESULTS_R3, "s1_probe.csv"))
    ridge = probe[probe.probe == "ridge"].pivot(index="layer", columns="token", values="r2_mean")
    r2_light = ridge["light"]
    r2_mean = ridge[TOKENS].mean(axis=1)

    # attention at transformer layer l writes hidden_states index l+1
    gx = g_best.index.values + 1
    peak_g = int(gx[np.argmax(g_best.values)])
    peak_p = int(r2_light.iloc[1:].idxmax())
    sep = abs(peak_g - peak_p)

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    ax.plot(r2_light.index, r2_light, color=ps.C_LIGHT, marker="o", markersize=3.5,
            label=f"perception: probe R² (light, peak L{peak_p})")
    ax.plot(r2_mean.index, r2_mean, color=ps.C_LIGHT, ls=":", lw=1.4, alpha=0.7,
            label="perception: probe R² (mean of 3 tokens)")
    ax.set_ylabel("probe R² (S1 ridge, 5-fold OOF)", color=ps.C_LIGHT)
    ax.set_xlabel("hidden_states index (attention layer l plotted at l+1)")
    ax2 = ax.twinx()
    ax2.plot(gx, g_best.values, color=ps.C_ORANGE, marker="s", markersize=3.5,
             label=f"grounding: best-head median IoU (peak L{peak_g})")
    ax2.plot(gx, g_mean.values, color=ps.C_ORANGE, ls=":", lw=1.4, alpha=0.7,
             label="grounding: head-mean IoU")
    ax2.axhline(chance, color=ps.INK3, lw=1.0, ls="--", label=f"IoU chance {chance:.2f}")
    ax2.set_ylabel("median IoU vs <box> (E7, top-p 0.5)", color=ps.C_ORANGE)
    ax2.grid(False)
    ax.axvline(peak_p, color=ps.C_LIGHT, lw=1.0, ls=":")
    ax2.axvline(peak_g, color=ps.C_ORANGE, lw=1.0, ls=":")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")

    verdict = ("same depth" if sep <= 2 else f"{sep} layers apart (shallow vs deep)")
    summ = dict(grounding_peak_hs_index=peak_g, perception_peak_hs_index=peak_p,
                separation=sep, chance_iou=chance,
                grounding_best_iou=float(g_best.max()),
                grounding_curve_best=dict(zip(map(int, gx), map(float, g_best.values))),
                perception_curve_light=dict(zip(map(int, r2_light.index), map(float, r2_light))))
    json.dump(summ, open(os.path.join(RESULTS_R3, "s2_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if not k.endswith("curve_best")
                      and not k.endswith("curve_light")}, indent=1))
    ps.conclusion_title(fig,
        f"S2: grounding peaks L{peak_g}, perception peaks L{peak_p} — {verdict}",
        sub="grounding = E7 per-head median IoU (129 box train samples) aggregated over heads; "
            "perception = S1 ridge probe rerun (n=792); dual y-axis, x aligned to hidden_states index")
    ps.save(fig, os.path.join(RESULTS_R3, "s2_grounding_vs_perception.png"))
    print("[s2] done")


if __name__ == "__main__":
    main()
