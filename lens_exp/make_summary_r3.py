# Round-3 overview collage: stack the per-experiment main figures into summary.png.
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_r3 import RESULTS_R3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PANELS = [
    "s1_probe_curves.png",
    "s2_grounding_vs_perception.png",
    "s3_iou_conditions.png",
    "s4_masks_and_render.png",
    "s5_metrics_bar.png",
    "s6_concept_matrix_l11x32.png",
    "s6_concept_matrix_l11l14l23x8.png",
    "s7a_probe_l11x32.png",
    "s7b_dose_response_l11x32.png",
    "s7_controls_l11x32.png",
    "s7c_decomposition_l11x32.png",
]


def main():
    imgs = [(p, mpimg.imread(os.path.join(RESULTS_R3, p)))
            for p in PANELS if os.path.exists(os.path.join(RESULTS_R3, p))]
    n = len(imgs)
    fig, axes = plt.subplots(n, 1, figsize=(14, sum(im.shape[0] / im.shape[1] * 14 for _, im in imgs)))
    for ax, (name, im) in zip(axes, imgs):
        ax.imshow(im)
        ax.axis("off")
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(RESULTS_R3, "summary.png"), dpi=110)
    print(f"summary.png with {n} panels")


if __name__ == "__main__":
    main()
