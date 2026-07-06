# E2 evaluation: render variants offline on the test split, compute metrics, make figures.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2

from common import RESULTS, DUMPS, load_manifest
from e2_lib import load_head_decoder, latents_at_layer, load_input_tensor, render
from e2_train import get_split
from metrics import image_metrics, ClipScorer
import plotstyle as ps
import matplotlib.pyplot as plt

DUMP_DIR = os.path.join(DUMPS, "baseline")


def build_variant(name, l_star):
    """returns (head, decoder, layer) for a variant name"""
    if name == "baseline":
        h, d = load_head_decoder(dtype=torch.bfloat16)
        return h, d, 24
    if name == "e2a":
        h, d = load_head_decoder(dtype=torch.bfloat16)
        return h, d, l_star
    if name in ("e2b", "e2b_ctrl"):
        h, d = load_head_decoder(dtype=torch.float32)
        ck = torch.load(os.path.join(RESULTS, f"e2b_{name}.pt"), weights_only=False)
        h.load_state_dict(ck["state"]["head"]); d.load_state_dict(ck["state"]["decoder"])
        return h.to(torch.bfloat16), d.to(torch.bfloat16), ck["layer"]
    raise ValueError(name)


def render_variant(name, l_star, keys, rows):
    out_dir = os.path.join(RESULTS, f"preds_{name}" if name != "baseline" else "preds_baseline")
    os.makedirs(out_dir, exist_ok=True)
    head, decoder, layer = build_variant(name, l_star)
    for k in keys:
        op = os.path.join(out_dir, k + ".png")
        if os.path.exists(op):
            continue
        d = np.load(os.path.join(DUMP_DIR, k + ".npz"))
        lat = latents_at_layer(d, layer)
        x = load_input_tensor(rows[k]["input_path"])
        img = render(head, decoder, lat, x)
        cv2.imwrite(op, img)
    del head, decoder
    torch.cuda.empty_cache()
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["e2a", "e2b", "e2b_ctrl"])
    ap.add_argument("--skip-clip", action="store_true")
    args = ap.parse_args()

    e1 = json.load(open(os.path.join(RESULTS, "e1_summary.json")))
    pk = e1["peak_layer"]
    l_star = (pk["light"], pk["colortemp"], pk["colormixer"])  # per-token optimal layers
    l_lbl = "/".join(str(v) for v in l_star)
    rows = {r["key"]: r for r in load_manifest()}
    test = get_split()["test"]
    print(f"l*={l_star}, test n={len(test)}", flush=True)

    variant_dirs = {"baseline": os.path.join(RESULTS, "preds_baseline")}
    for v in args.variants:
        print("render", v, flush=True)
        variant_dirs[v] = render_variant(v, l_star, test, rows)

    clip = None if args.skip_clip else ClipScorer()
    recs = []
    for v, d in variant_dirs.items():
        for k in test:
            pp = os.path.join(d, k + ".png")
            if not os.path.exists(pp):
                continue
            m = image_metrics(pp, rows[k]["gt_path"])
            m.update(variant=v, key=k, lang=rows[k]["lang"])
            if clip is not None:
                m["clip"] = clip.score(pp, rows[k]["prompt"])
            recs.append(m)
        print("metrics done", v, flush=True)
    df = pd.DataFrame(recs)
    # input-vs-gt reference (do-nothing lower bound)
    ref = []
    for k in test:
        m = image_metrics(rows[k]["input_path"], rows[k]["gt_path"])
        m.update(variant="input(no-op)", key=k, lang=rows[k]["lang"])
        if clip is not None:
            m["clip"] = clip.score(rows[k]["input_path"], rows[k]["prompt"])
        ref.append(m)
    df = pd.concat([df, pd.DataFrame(ref)], ignore_index=True)
    df.to_csv(os.path.join(RESULTS, "e2_results.csv"), index=False)

    # ---------- bar + strip figure ----------
    order = ["input(no-op)", "baseline", "e2a", "e2b_ctrl", "e2b"]
    order = [v for v in order if v in df.variant.unique()]
    labels = {"input(no-op)": "input\n(no-op)", "baseline": "baseline\nL24",
              "e2a": f"E2a frozen\nL{l_lbl}", "e2b_ctrl": "E2b-ctrl\nL24 retr.",
              "e2b": f"E2b retr.\nL{l_lbl}"}
    colors = {"input(no-op)": ps.INK3, "baseline": ps.C_HILITE, "e2a": ps.C_ORANGE,
              "e2b_ctrl": ps.C_VIOLET, "e2b": ps.C_LIGHT}
    metrics_list = [("de00", "deltaE00 (lower better)"), ("psnr", "PSNR (higher better)"),
                    ("ssim", "SSIM (higher better)")] + ([("clip", "CLIP img-text sim")] if clip else [])
    fig, axes = plt.subplots(1, len(metrics_list), figsize=(3.9 * len(metrics_list), 4.6))
    rng = np.random.default_rng(0)
    for ax, (mname, mtitle) in zip(np.atleast_1d(axes), metrics_list):
        for i, v in enumerate(order):
            vals = df[df.variant == v][mname].values
            ax.bar(i, vals.mean(), width=0.62, color=colors[v], alpha=0.85)
            ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=5, color=ps.INK,
                       alpha=0.25, linewidths=0)
            ax.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center",
                    va="bottom", fontsize=8.5, color=ps.INK, fontweight="bold")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([labels[v] for v in order], fontsize=7)
        ax.set_title(mtitle)
    base_de = df[df.variant == "baseline"].de00.mean()
    e2b_de = df[df.variant == "e2b"].de00.mean() if "e2b" in df.variant.unique() else np.nan
    ps.conclusion_title(fig,
        f"E2: retrained L{l_lbl} readout reaches deltaE00 {e2b_de:.2f} vs baseline {base_de:.2f} "
        f"({e2b_de-base_de:+.2f})",
        sub=f"test split n={len(test)} (held out from decoder retraining); dots = per-sample values")
    ps.save(fig, os.path.join(RESULTS, "e2_metrics_bar.png"))

    # ---------- compare grid ----------
    show_variants = [v for v in ["baseline", "e2a", "e2b"] if v in variant_dirs]
    de_b = df[df.variant == "baseline"].set_index("key").de00
    de_n = df[df.variant == (("e2b") if "e2b" in df.variant.unique() else "e2a")].set_index("key").de00
    diff = (de_b - de_n).sort_values()
    picks = list(diff.index[-6:]) + list(diff.index[:3]) + list(diff.index[len(diff)//2-1:len(diff)//2+2])
    picks = picks[:12]
    ncol = 2 + len(show_variants)
    fig, axes = plt.subplots(12, ncol, figsize=(2.3 * ncol, 12 * 1.62))
    col_titles = ["input"] + [labels[v].replace("\n", " ") for v in show_variants] + ["gt (expert)"]
    for ri, k in enumerate(picks):
        imgs = [rows[k]["input_path"]] + [os.path.join(variant_dirs[v], k + ".png") for v in show_variants] + [rows[k]["gt_path"]]
        for ci, p in enumerate(imgs):
            ax = axes[ri, ci]
            im = cv2.imread(p, cv2.IMREAD_COLOR)
            s = 360 / max(im.shape[:2])
            im = cv2.resize(im, (int(im.shape[1]*s), int(im.shape[0]*s)))
            ax.imshow(im[..., ::-1]); ax.axis("off")
            if ri == 0:
                ax.set_title(col_titles[ci], fontsize=9)
        d0 = de_b.get(k, np.nan); d1 = de_n.get(k, np.nan)
        axes[ri, 0].set_ylabel(k, fontsize=7)
        axes[ri, 0].text(-0.06, 0.5, f"{k}\ndE base {d0:.1f} -> new {d1:.1f}",
                         transform=axes[ri, 0].transAxes, fontsize=7, rotation=90,
                         va="center", ha="right", color=ps.INK2)
    ps.conclusion_title(fig, f"E2 qualitative: baseline vs layer-{l_lbl} readout (top: most improved; bottom rows: regressions / median)")
    ps.save(fig, os.path.join(RESULTS, "e2_compare_grid.png"))
    print("E2 eval done")


if __name__ == "__main__":
    main()
