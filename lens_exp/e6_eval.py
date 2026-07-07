# E6 (round 2) eval: offline render all variants on the R2 test split, metrics + gate + figure.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import wilcoxon

from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from e2_lib import load_head_decoder, load_input_tensor, render
from e6_lib import build_variant_model, load_latent, PEAK
import plotstyle as ps
import matplotlib.pyplot as plt
from metrics import image_metrics

RETRAINED = ["e6a", "e6b", "e6c", "e6_ctrl"]


def get_variant(name):
    if name == "baseline":
        h, d = load_head_decoder(dtype=torch.bfloat16)
        return h, d, "l24", 24
    head, decoder, lat_kind, layers = build_variant_model(name, dtype=torch.float32)
    ck = torch.load(os.path.join(RESULTS_R2, f"{name}.pt"), weights_only=False)
    head.load_state_dict(ck["state"]["head"])
    decoder.load_state_dict(ck["state"]["decoder"])
    return head.to(torch.bfloat16), decoder.to(torch.bfloat16), ck["lat_kind"], ck["layers"]


def render_variant(name, keys, rows):
    out_dir = os.path.join(RESULTS_R2, f"preds_{name}")
    os.makedirs(out_dir, exist_ok=True)
    missing = [k for k in keys if not os.path.exists(os.path.join(out_dir, k + ".png"))]
    if missing:
        head, decoder, lat_kind, layers = get_variant(name)
        for k in missing:
            d = np.load(os.path.join(C0_DIR, k + ".npz"))
            lat = load_latent(d, lat_kind, layers)
            img = render(head, decoder, lat, load_input_tensor(rows[k]["input_path"]))
            cv2.imwrite(os.path.join(out_dir, k + ".png"), img)
        del head, decoder
        torch.cuda.empty_cache()
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=RETRAINED)
    args = ap.parse_args()
    rows = {r["key"]: r for r in load_manifest_r2()}
    test = get_split_r2()["test"]
    print(f"test n={len(test)}", flush=True)

    dirs = {"baseline": os.path.join(RESULTS_R2, "preds_c0")}  # C0 online preds == frozen baseline
    for v in args.variants:
        print("render", v, flush=True)
        dirs[v] = render_variant(v, test, rows)

    recs = []
    for v, dd in dirs.items():
        for k in test:
            pp = os.path.join(dd, k + ".png")
            if not os.path.exists(pp):
                continue
            m = image_metrics(pp, rows[k]["gt_path"])
            m.update(variant=v, key=k, lang=rows[k]["lang"])
            recs.append(m)
        print("metrics done", v, flush=True)
    for k in test:  # no-op reference
        m = image_metrics(rows[k]["input_path"], rows[k]["gt_path"])
        m.update(variant="input(no-op)", key=k, lang=rows[k]["lang"])
        recs.append(m)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R2, "e6_results.csv"), index=False)

    piv = df.pivot_table(index="key", columns="variant", values="de00")
    mean = piv.mean()

    def wtest(a, b):
        s = piv[[a, b]].dropna()
        return float(wilcoxon(s[a], s[b]).pvalue) if len(s) > 5 else np.nan

    pairs = [("e6a", "baseline"), ("e6a", "e6_ctrl"), ("e6b", "e6a"), ("e6c", "e6a"),
             ("e6b", "e6_ctrl"), ("e6c", "e6_ctrl"), ("baseline", "input(no-op)")]
    pvals = {f"{a}_vs_{b}": wtest(a, b) for a, b in pairs}

    gain_b = mean.get("e6a", np.nan) - mean.get("e6b", np.nan)
    gain_c = mean.get("e6a", np.nan) - mean.get("e6c", np.nan)
    gate = bool(max(gain_b, gain_c) >= 0.15)
    # round-1 question: does the layer-swap gain survive more data?
    swap_gain = mean.get("e6_ctrl", np.nan) - mean.get("e6a", np.nan)
    best_eps = {}
    for v in RETRAINED:
        p = os.path.join(RESULTS_R2, f"{v}.pt")
        if os.path.exists(p):
            best_eps[v] = int(torch.load(p, weights_only=False)["best_epoch"])
    summary = dict(mean_de00={k: float(v) for k, v in mean.items()},
                   wilcoxon_p=pvals, swap_gain_vs_ctrl=float(swap_gain),
                   fusion_gain=float(gain_b), affine_gain=float(gain_c),
                   gate_fusion_or_affine_ge_0p15=gate, best_epochs=best_eps,
                   n_test=int(len(test)))
    json.dump(summary, open(os.path.join(RESULTS_R2, "e6_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

    # ---------- main figure ----------
    order = ["input(no-op)", "baseline", "e6_ctrl", "e6a", "e6c", "e6b"]
    order = [v for v in order if v in piv.columns]
    lbl = {"input(no-op)": "input\n(no-op)", "baseline": "baseline\nfrozen L24",
           "e6_ctrl": "retr. L24\n(ctrl)", "e6a": f"E6a retr.\nL{'/'.join(map(str,PEAK))}",
           "e6c": "E6c affine\n+peak", "e6b": "E6b fuse\nL11/14/23/24"}
    col = {"input(no-op)": ps.INK3, "baseline": ps.C_HILITE, "e6_ctrl": ps.C_VIOLET,
           "e6a": ps.C_LIGHT, "e6c": ps.C_ORANGE, "e6b": ps.C_COLORTEMP}
    fig, ax = plt.subplots(figsize=(9.4, 5.0))
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        vals = piv[v].dropna().values
        ax.bar(i, vals.mean(), width=0.62, color=col[v], alpha=0.88,
               yerr=vals.std() / np.sqrt(len(vals)), capsize=3, ecolor=ps.INK2)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=4, color=ps.INK, alpha=0.18, linewidths=0)
        ax.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=ps.INK)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lbl[v] for v in order], fontsize=8)
    ax.set_ylabel("deltaE00 vs expert gt (lower = better)")
    ax.set_ylim(0, np.percentile(np.concatenate([piv[v].dropna().values for v in order]), 97))
    verdict = ("fusion/affine helps" if gate else "fusion/affine gain < 0.15 — layer choice, not capacity, is the story")
    ps.conclusion_title(fig,
        f"E6: peak-layer retrain {mean.get('e6a', float('nan')):.2f} vs ctrl-L24 {mean.get('e6_ctrl', float('nan')):.2f} "
        f"(swap gain {swap_gain:+.2f}); fusion {gain_b:+.2f}, affine {gain_c:+.2f} vs E6a — {verdict}",
        sub=f"test n={len(test)} (8:2 split, ~{640} train imgs); bars = mean +/- SEM, dots = samples; "
            f"Wilcoxon p(e6a vs ctrl)={pvals['e6a_vs_e6_ctrl']:.2g}, p(e6b vs e6a)={pvals['e6b_vs_e6a']:.2g}, "
            f"p(e6c vs e6a)={pvals['e6c_vs_e6a']:.2g}")
    ps.save(fig, os.path.join(RESULTS_R2, "e6_metrics_bar.png"))
    print("E6 eval done")


if __name__ == "__main__":
    main()
