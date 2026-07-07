# Round-2 summary: 2x3 overview panel from the per-experiment main figures.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from common_r2 import RESULTS_R2
import plotstyle as ps

PANELS = [
    ("e6_metrics_bar.png", "E6 readout"),
    ("e7_head_matrix.png", "E7 head-level IoU"),
    ("e7_intervention_scatter.png", "E7 head rescale"),
    ("e8_metrics_bar.png", "E8 localized render"),
    ("e9_concept_matrix.png", "E9 SAE concepts"),
    ("e9_steering_curves.png", "E9 steering"),
]


def main():
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))
    for ax, (fn, ttl) in zip(axes.ravel(), PANELS):
        p = os.path.join(RESULTS_R2, fn)
        ax.axis("off")
        if os.path.exists(p):
            ax.imshow(mpimg.imread(p))
        else:
            ax.text(0.5, 0.5, f"missing: {fn}", ha="center")
        ax.set_title(ttl, fontsize=13, fontweight="bold")
    ps.conclusion_title(fig, "Lens round 2 overview: C0 full-800 capture -> E6 readout / E7 heads / E8 localization / E9 SAE")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(RESULTS_R2, "summary.png"), dpi=110)
    print("summary saved")


if __name__ == "__main__":
    main()
