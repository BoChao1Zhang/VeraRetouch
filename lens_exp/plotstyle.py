# lens-exp unified matplotlib style (dataviz reference palette, light mode)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a897f"

# fixed categorical assignment across ALL lens-exp figures
C_LIGHT = "#2a78d6"      # slot1 blue  -> <retouch_light>
C_COLORTEMP = "#1baf7a"  # slot2 aqua  -> <retouch_color&temp>
C_COLORMIXER = "#eda100" # slot3 yellow-> <retouch_colormixer>
C_HILITE = "#e34948"     # red accent  -> final-layer / bad
C_VIOLET = "#4a3aa7"     # slot5       -> extra series (e.g. text tokens)
C_ORANGE = "#eb6834"     # slot8
TOKEN_COLORS = {"light": C_LIGHT, "colortemp": C_COLORTEMP, "colormixer": C_COLORMIXER}
TOKEN_LABELS = {"light": "<retouch_light>", "colortemp": "<retouch_color&temp>",
                "colormixer": "<retouch_colormixer>"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 150,
    "text.color": INK, "axes.edgecolor": INK3, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.labelsize": 10, "legend.frameon": False,
    "font.size": 10, "lines.linewidth": 2.0,
    "font.family": "DejaVu Sans",
})


def conclusion_title(fig, text, sub=None):
    h = fig.get_size_inches()[1]
    fig.suptitle(text, fontsize=13, fontweight="bold", color=INK,
                 x=0.02, y=1 - 0.06 / h, ha="left", va="top")
    if sub:
        fig.text(0.02, 1 - 0.38 / h, sub, fontsize=9.5, color=INK2, ha="left", va="top")
    fig._lens_top = 1 - (0.72 if sub else 0.5) / h


def save(fig, path):
    fig.tight_layout(rect=(0, 0, 1, getattr(fig, "_lens_top", 0.92)))
    fig.savefig(path)
    plt.close(fig)
