# S7c (round 3): are style features a COMBINATION of low-level photometric features, or an
# independent channel? For each style: (i) overlap between style-selective features and
# photometric-correlated features (|rho|>0.2 with any of the 26 deltas, train split);
# (ii) share of the style group's mean-z mass carried by photometric-correlated features.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata

from common import TOKENS
from common_r2 import C0_DIR, get_split_r2, load_targets_r2, DIM_NAMES
from common_r3 import RESULTS_R3
from s7_style import load_sae
import plotstyle as ps
import matplotlib.pyplot as plt

LAYER = 11


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l11x32")
    ap.add_argument("--rho", type=float, default=0.2)
    ap.add_argument("--topn", type=int, default=10)
    args = ap.parse_args()

    sae = load_sae(args.tag)
    lab = json.load(open(os.path.join(RESULTS_R3, "s7_labels.json")))
    labels, styles = lab["labels"], lab["styles"]
    split = get_split_r2()
    targets = load_targets_r2()
    tr = [k for k in split["train"] if os.path.exists(os.path.join(C0_DIR, k + ".npz"))
          and k in targets]

    # encode all train samples once per token
    lat = {t: [] for t in TOKENS}
    for k in tr:
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        for t in TOKENS:
            lat[t].append(d[f"lat_{t}"][LAYER].astype(np.float32))
        d.close()
    Z = {}
    with torch.no_grad():
        for t in TOKENS:
            Z[t] = sae.encode(torch.from_numpy(np.stack(lat[t])).cuda()).cpu().numpy()
    T = np.stack([targets[k] for k in tr])
    Tr = rankdata(T, axis=0)
    Tr = (Tr - Tr.mean(0)) / (Tr.std(0) + 1e-9)

    # photometric-correlated feature set per token: max |rho| over the 26 dims
    photo = {}
    for t in TOKENS:
        A = Z[t]
        live = (A > 0).sum(axis=0) >= max(10, len(tr) // 20)
        Ar = rankdata(A, axis=0)
        Ar = (Ar - Ar.mean(0)) / (Ar.std(0) + 1e-9)
        R = (Ar.T @ Tr) / len(tr)
        R[~live] = 0
        photo[t] = np.abs(R).max(axis=1)          # [D]

    sel = pd.read_csv(os.path.join(RESULTS_R3, f"s7_selectivity_{args.tag}.csv"))
    y = np.array([labels.get(k) for k in tr])

    recs, summ = [], dict(tag=args.tag, rho_gate=args.rho)
    for sty in styles:
        top = sel[sel["style"] == sty].sort_values("selectivity", ascending=False).head(args.topn)
        mx = [float(photo[r.token][r.feature]) for r in top.itertuples()]
        overlap = float(np.mean([v > args.rho for v in mx]))
        # mass decomposition: style-group mean z, mass on photometric features
        mass_ph, mass_all = 0.0, 0.0
        for t in TOKENS:
            gz = Z[t][y == sty].mean(0) if (y == sty).any() else np.zeros(Z[t].shape[1])
            mass_all += gz.sum()
            mass_ph += gz[photo[t] > args.rho].sum()
        summ[sty] = dict(top_feature_max_photo_rho=mx, frac_top_features_photometric=overlap,
                         mass_share_on_photometric=float(mass_ph / (mass_all + 1e-9)),
                         n_group=int((y == sty).sum()))
        for r, v in zip(top.itertuples(), mx):
            recs.append(dict(style=sty, feature=r.feature, token=r.token,
                             selectivity=r.selectivity, max_photo_rho=v))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R3, f"s7c_loadings_{args.tag}.csv"), index=False)
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s7c_summary_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.7))
    rng = np.random.default_rng(0)
    for i, sty in enumerate(styles):
        v = df[df["style"] == sty].max_photo_rho.values
        ax1.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, s=14,
                    color=ps.C_LIGHT, alpha=0.8)
        ax1.scatter([i], [np.median(v)], marker="_", s=500, color=ps.C_HILITE, zorder=5)
    ax1.axhline(args.rho, color=ps.INK3, ls="--", lw=1,
                label=f"photometric gate |rho|={args.rho}")
    ax1.set_xticks(range(len(styles))); ax1.set_xticklabels(styles, fontsize=8, rotation=20)
    ax1.set_ylabel("max |rho| with 26 photometric deltas")
    ax1.set_title("are top style features photometric-correlated?")
    ax1.legend(fontsize=8)
    shares = [summ[s]["mass_share_on_photometric"] for s in styles]
    ax2.bar(range(len(styles)), shares, color=ps.C_COLORTEMP, width=0.6)
    for i, v in enumerate(shares):
        ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(styles))); ax2.set_xticklabels(styles, fontsize=8, rotation=20)
    ax2.set_ylabel("group mean-z mass share on photometric features")
    ax2.set_title("style vector mass carried by photometric features")
    med_share = float(np.median(shares))
    verdict = ("styles largely REUSE photometric feature combinations (hierarchical encapsulation)"
               if med_share > 0.5 else
               "styles occupy mostly NON-photometric features — a separate language-concept channel")
    ps.conclusion_title(fig,
        f"S7c: median mass share on photometric features {med_share:.2f} — {verdict}",
        sub=f"photometric feature = SAE feature with max train |rho|>{args.rho} against the 26 deltas; "
            f"top-{args.topn} selective features per style; train n={len(tr)}")
    ps.save(fig, os.path.join(RESULTS_R3, f"s7c_decomposition_{args.tag}.png"))
    print("[s7c] done")


if __name__ == "__main__":
    main()
