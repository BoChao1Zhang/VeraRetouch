# Assemble summary.png: 2x3 dashboard from the per-experiment main figures.
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from common import RESULTS
import plotstyle as ps


def main():
    panels = [
        ("e1_layer_curve.png", "E1 layer probe"),
        ("e2_metrics_bar.png", "E2 readout swap / retrain"),
        ("e3_box_iou_hist.png", "E3 box IoU"),
        ("e3_entropy_scatter.png", "E3 entropy vs error"),
        ("e4_norm_ratio.png", "E4 norm balance"),
    ]
    overlays = sorted(glob.glob(os.path.join(RESULTS, "e3_attn_overlay", "good_*.png")))
    if overlays:
        panels.append((overlays[-1], "E3 attention example"))

    fig, axes = plt.subplots(3, 2, figsize=(16, 13.5))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (p, ttl) in zip(axes.flat, panels):
        path = p if os.path.isabs(p) else os.path.join(RESULTS, p)
        if not os.path.exists(path):
            ax.text(0.5, 0.5, f"missing: {ttl}", ha="center", color=ps.C_HILITE)
            continue
        ax.imshow(mpimg.imread(path))
        ax.set_title(ttl, fontsize=10, color=ps.INK2)

    head = ("Lens x VeraRetouch — light info peaks mid-stack (L11, +0.07 R2); retrained L11/14/23 readout: "
            "deltaE00 11.26 -> 10.15; box IoU 0.43 vs chance 0.23; norms balanced")
    ps.conclusion_title(fig, head, sub="details: REPORT.md; per-experiment CSV/PNG pairs in this directory")
    ps.save(fig, os.path.join(RESULTS, "summary.png"))
    print("summary.png saved")


if __name__ == "__main__":
    main()
