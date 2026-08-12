"""P9 · MCQ-L 端到端条件消融柱状图（零 GPU）。

出图：docs/presentation_2026-08-05/figs/P9_condition_ablation.png

**图上每一个数字都从落盘 json 读出，脚本内不硬编码任何实测值。**
数据源：experiments/MCQ_full_local_l1l6_20260804/runs/<arm>/metrics.json → .final.select

三个 arm 的 condition_mode 语义（出处 train.py:113-135 `condition_inputs`）：
  config_a                 condition_mode="full"          图 + 本样本真实指令
  condition_fixed_shuffle  condition_mode="fixed_shuffle" 图 + **别的样本的指令**（确定性错位，data.py:116）
  condition_image_only     condition_mode="image_only"    图 + **对所有图逐字相同的中性句**
                                                          NEUTRAL_INSTRUCTION（train.py:115）

⚠ 纵轴一律从 0 起，禁止截断——本页的说服力恰恰来自"三根柱子一样高"。

运行：/home/bc/VeraRetouch/.venv-lens/bin/python <本文件>
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = REPO / "experiments/MCQ_full_local_l1l6_20260804/runs"
TRAIN_PY = REPO / "experiments/MCQ_full_local_l1l6_20260804/train.py"

# (arm 目录, 图上的中文标签, 颜色)
ARMS = [
    ("config_a", "图 + 真实指令\n（完整输入）", "真实指令", "#2a6ebb"),
    ("condition_fixed_shuffle", "图 + 别的样本的指令\n（固定错位指令）", "错位指令", "#b8791f"),
    ("condition_image_only", "图 + 对所有图相同的中性句\n（= 不给指令信息）", "中性句", "#9aa0a6"),
]


def mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字形：沿用 ro3_viz._mpl() 已在本机核过的字体链
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def neutral_instruction() -> str:
    """从 train.py 直接读出中性句，避免手敲。"""
    for line in TRAIN_PY.read_text(encoding="utf-8").splitlines():
        if line.startswith("NEUTRAL_INSTRUCTION"):
            return line.split("=", 1)[1].strip().strip('"')
    return "(未在 train.py 中找到 NEUTRAL_INSTRUCTION)"


def load():
    out = []
    for arm, label, short, color in ARMS:
        m = json.loads((RUNS / arm / "metrics.json").read_text())
        sel = m["final"]["select"]
        run = json.loads((RUNS / arm / "run.json").read_text())
        out.append({
            "arm": arm, "label": label, "short": short, "color": color,
            "mode": run["config"]["condition_mode"] if "condition_mode" in run["config"] else "full",
            "n": sel["n"],
            "auc": sel["auc_cgt_p50"],
            "var_ratio": sel["var_ratio"],
            "dshuf": sel["delta_shuffle_db_p50"],
            "de00": sel["de00_p50"],
            "psnr_in": sel["psnr_in_p50"],
            "soft_iou": sel["soft_iou_p50"],
            "step": m["selected_step"],
            "digest": m["manifest_digest"],
        })
    return out


def bars(ax, rows, key, title, ylabel, fmt, ymax=None, long_labels=False,
         zero_eps=None):
    xs = range(len(rows))
    vals = [r[key] for r in rows]
    ax.bar(xs, vals, color=[r["color"] for r in rows], width=0.62,
           edgecolor="white", linewidth=1.4, zorder=3)
    top = ymax if ymax is not None else max(vals) * 1.32
    ax.set_ylim(0, top)                                   # ⚠ 纵轴从 0 起，不截断
    ax.set_xticks(list(xs))
    ax.set_xticklabels([r["label"] if long_labels else r["short"] for r in rows],
                       fontsize=11.5 if long_labels else 12.5)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, pad=11)
    ax.grid(axis="y", color="#d9d9d9", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for x, v in zip(xs, vals):
        near_zero = zero_eps is not None and abs(v) < zero_eps
        txt = f"{fmt.format(v)}\n≈ 0" if near_zero else fmt.format(v)
        ax.text(x, max(v, 0) + top * 0.028, txt, ha="center", va="bottom",
                fontsize=15, fontweight="bold",
                color="#b00020" if near_zero else "black", linespacing=1.15)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    plt = mpl()
    rows = load()
    n = rows[0]["n"]
    assert all(r["n"] == n for r in rows), "三个 arm 的评测样本量不一致"
    assert len({r["digest"] for r in rows}) == 1, "manifest digest 不一致，唯一变量不成立"

    fig = plt.figure(figsize=(17.5, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.0, 1.0], wspace=0.30,
                          left=0.055, right=0.985, top=0.80, bottom=0.245)

    ax0 = fig.add_subplot(gs[0, 0])
    bars(ax0, rows, "auc", "掩膜 AUC（对 .cgt 真值）\n⚑ 三档几乎相同 —— 不给指令甚至更高",
         "auc_cgt  p50", "{:.4f}", ymax=1.0, long_labels=True)
    ax0.axhline(0.5, color="#b00020", lw=1.6, ls="--", zorder=4)
    ax0.text(len(rows) - 0.42, 0.515, "随机基线 0.5", color="#b00020",
             fontsize=11, ha="right")

    ax1 = fig.add_subplot(gs[0, 1])
    bars(ax1, rows, "var_ratio", "颜色改动的方差比\n（预测 Δ 方差 / 真值 Δ 方差）",
         "var_ratio", "{:.3f}")

    ax2 = fig.add_subplot(gs[0, 2])
    bars(ax2, rows, "dshuf", "Δ_shuffle（换指令重渲的 PSNR 差）\n",
         "delta_shuffle_db  p50 (dB)", "{:.3f}", zero_eps=0.01)

    fig.suptitle("P9 · MCQ-L 端到端条件消融：掩膜 AUC 分不开三档，颜色与像素分得开",
                 fontsize=19, fontweight="bold", y=0.955)
    fig.text(0.5, 0.885,
             "同一 manifest / 同一 seed / 同一 6000 step × batch 8，唯一变量 = 喂进模型的条件输入",
             ha="center", fontsize=12.5, color="#444444")

    supp = "   |   ".join(
        f"{r['label'].splitlines()[0]}：ΔE00 p50 {r['de00']:.3f}・"
        f"mask 内 PSNR {r['psnr_in']:.2f} dB・soft-IoU {r['soft_iou']:.3f}"
        for r in rows)
    fig.text(0.5, 0.145, supp, ha="center", fontsize=11.2, color="#222222")

    cap = (
        f"口径：S-train 内 source-disjoint 的完整 select 池，n = {n}（三档同一批样本）；"
        f"数字全部取自 runs/<arm>/metrics.json → .final.select，"
        f"选中 step = {'/'.join(str(r['step']) for r in rows)}。\n"
        "arm 语义（train.py:113-135 condition_inputs）："
        "「固定错位指令」= 喂别的样本的真实指令（data.py:116，逐样本不同但都是错的）；"
        f"「中性句」= 对所有图逐字相同的 “{neutral_instruction()}”（train.py:115）。\n"
        "⚠ 纵轴一律从 0 起、未截断。Δ_shuffle 是 batch 内循环移位指令后重渲的 PSNR 差"
        "（train.py:296-343）——按 EXPERIMENT_SCHEDULE 自述，它并「不是」严格的同图反事实，"
        "只能读作「换指令后输出会变」，不能读作「按指令变对了」。")
    fig.text(0.055, 0.012, cap, ha="left", va="bottom", fontsize=10.2, color="#333333")

    out = HERE / "P9_condition_ablation.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"[written] {out}")
    for r in rows:
        print(f"  {r['arm']:26s} mode={r['mode']:14s} n={r['n']} "
              f"auc={r['auc']:.4f} var_ratio={r['var_ratio']:.4f} "
              f"dshuf={r['dshuf']:+.4f} de00={r['de00']:.3f} psnr_in={r['psnr_in']:.2f}")


if __name__ == "__main__":
    main()
