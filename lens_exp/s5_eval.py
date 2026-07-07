# S5 (round 3) eval: render s5_lam* variants on the R2 test split, compare deltaE00 against
# e6_ctrl (L24 retrain, no aux) and e6a (peak-layer retrain) using the cached e6_results.csv.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import wilcoxon

from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from common_r3 import RESULTS_R3
from e2_lib import load_head_decoder, load_input_tensor, render, latents_at_layer
from s5_aux_train import AuxReadout
from metrics import image_metrics
import plotstyle as ps
import matplotlib.pyplot as plt


class RenderHead(torch.nn.Module):
    """AuxReadout minus the aux output, for the render() helper."""

    def __init__(self, aux_readout):
        super().__init__()
        self.m = aux_readout

    def forward(self, x):
        return self.m(x)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="+", default=[0.1, 0.5])
    args = ap.parse_args()
    rows = {r["key"]: r for r in load_manifest_r2()}
    test = get_split_r2()["test"]

    cache = os.path.join(RESULTS_R3, "s5_results.csv")
    if os.path.exists(cache):
        df = pd.read_csv(cache)
        make_outputs(df, args, test)
        return

    recs = []
    for lam in args.lams:
        tag = f"s5_lam{lam:g}"
        ck = torch.load(os.path.join(RESULTS_R3, f"{tag}.pt"), weights_only=False)
        base_head, decoder = load_head_decoder(dtype=torch.float32)
        head = AuxReadout(base_head).cuda().float()
        head.load_state_dict(ck["state"]["head"])
        decoder.load_state_dict(ck["state"]["decoder"])
        rhead = RenderHead(head).to(torch.bfloat16).eval()
        decoder = decoder.to(torch.bfloat16).eval()
        out_dir = os.path.join(RESULTS_R3, f"preds_{tag}")
        os.makedirs(out_dir, exist_ok=True)
        for k in test:
            p = os.path.join(out_dir, k + ".png")
            if not os.path.exists(p):
                d = np.load(os.path.join(C0_DIR, k + ".npz"))
                lat = latents_at_layer(d, 24)
                d.close()
                img = render(rhead, decoder, lat, load_input_tensor(rows[k]["input_path"]))
                cv2.imwrite(p, img)
            m = image_metrics(p, rows[k]["gt_path"])
            recs.append(dict(key=k, variant=tag, **m))
        del head, decoder, rhead
        torch.cuda.empty_cache()
        print(f"[s5eval] {tag} rendered+scored", flush=True)

    df = pd.DataFrame(recs)
    e6 = pd.read_csv(os.path.join(RESULTS_R2, "e6_results.csv"))
    e6 = e6[e6.variant.isin(["e6a", "e6_ctrl", "input(no-op)"])][["key", "variant", "de00", "psnr", "ssim"]]
    df = pd.concat([df[["key", "variant", "de00", "psnr", "ssim"]], e6], ignore_index=True)
    df.to_csv(os.path.join(RESULTS_R3, "s5_results.csv"), index=False)
    make_outputs(df, args, test)


def make_outputs(df, args, test):
    piv = df.pivot_table(index="key", columns="variant", values="de00")
    mean = piv.mean()

    def wp(a, b):
        s = piv[[a, b]].dropna()
        return float(wilcoxon(s[a], s[b]).pvalue) if len(s) > 5 else np.nan

    tags = [f"s5_lam{l:g}" for l in args.lams]
    best_tag = min(tags, key=lambda t: mean[t])
    gap_to_e6a = float(mean[best_tag] - mean["e6a"])
    gain_vs_ctrl = float(mean["e6_ctrl"] - mean[best_tag])
    gate = bool(gap_to_e6a <= 0.2)
    summ = dict(mean_de00={k: float(v) for k, v in mean.items()},
                best_tag=best_tag, gap_to_e6a=gap_to_e6a, gain_vs_ctrl=gain_vs_ctrl,
                gate_aux_substitutes_swap=gate,
                p_best_vs_ctrl=wp(best_tag, "e6_ctrl"), p_best_vs_e6a=wp(best_tag, "e6a"),
                best_epochs={t: int(torch.load(os.path.join(RESULTS_R3, f"{t}.pt"),
                                               weights_only=False)["best_epoch"]) for t in tags},
                n_test=len(test))
    json.dump(summ, open(os.path.join(RESULTS_R3, "s5_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    order = ["input(no-op)", "e6_ctrl"] + tags + ["e6a"]
    lbl = {"input(no-op)": "input\n(no-op)", "e6_ctrl": "retr. L24\n(no aux)",
           "e6a": "E6a retr.\nL11/14/23"}
    lbl.update({t: f"L24 + aux\nλ={t.split('lam')[1]}" for t in tags})
    col = {"input(no-op)": ps.INK3, "e6_ctrl": ps.C_VIOLET, "e6a": ps.C_LIGHT}
    col.update({t: ps.C_ORANGE for t in tags})
    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        vals = piv[v].dropna().values
        ax.bar(i, vals.mean(), width=0.62, color=col[v], alpha=0.88,
               yerr=vals.std() / np.sqrt(len(vals)), capsize=3, ecolor=ps.INK2)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=4, color=ps.INK, alpha=0.18)
        ax.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lbl[v] for v in order], fontsize=8)
    ax.set_ylabel("deltaE00 vs gt (lower = better)")
    ax.set_ylim(0, float(np.percentile(np.concatenate([piv[v].dropna().values for v in order]), 97)))
    verdict = ("aux supervision substitutes for the layer swap" if gate else
               "layer swap remains necessary")
    ps.conclusion_title(fig,
        f"S5: L24+aux {mean[best_tag]:.2f} = ctrl {mean['e6_ctrl']:.2f} < e6a "
        f"{mean['e6a']:.2f} (gap {gap_to_e6a:+.2f}) — {verdict}",
        sub=f"LLL-style aux head (26-dim photometric delta, standardized) on the readout trunk's "
            f"penultimate features; backbone frozen so regularization acts on the readout path only; "
            f"test n={len(test)}, p(best vs ctrl)={summ['p_best_vs_ctrl']:.2g}")
    ps.save(fig, os.path.join(RESULTS_R3, "s5_metrics_bar.png"))
    print("[s5] done")


if __name__ == "__main__":
    main()
