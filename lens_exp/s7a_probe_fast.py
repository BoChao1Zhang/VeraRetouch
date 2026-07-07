# S7a probe (fast variant): style linear probe on SAE z vs raw hidden, 5-fold OOF.
# The full 20x-permutation run was computationally infeasible under host contention;
# this variant restricts z to features active in >=5% of labeled samples (same claim,
# documented deviation) and tests significance vs majority-chance with a binomial test.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common_r3 import RESULTS_R3
from s7_style import load_sae, encode_keys
import plotstyle as ps
import matplotlib.pyplot as plt


def oof_acc(X, y, scale=False, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    return float((pred == y).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l11x32")
    ap.add_argument("--n-perm", type=int, default=3)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    lab = json.load(open(os.path.join(RESULTS_R3, "s7_labels.json")))
    keys = sorted(lab["labels"])
    y = np.array([lab["labels"][k] for k in keys])
    styles = lab["styles"]
    sae = load_sae(args.tag)
    H, Z = encode_keys(keys, sae)
    live = (Z > 0).mean(0) >= 0.05
    Zl = Z[:, live]
    print(f"[s7a-fast] n={len(keys)} live z dims={live.sum()} raw dims={H.shape[1]}", flush=True)

    acc_z = oof_acc(Zl, y)
    acc_h = oof_acc(H, y, scale=True)
    chance = float(pd.Series(y).value_counts(normalize=True).iloc[0])
    n_correct = int(round(acc_z * len(y)))
    p_binom = float(binomtest(n_correct, len(y), chance, alternative="greater").pvalue)
    perm = [oof_acc(Zl, rng.permutation(y), seed=i) for i in range(args.n_perm)]
    gate = bool(p_binom < 0.01 and acc_z > chance)
    summ = dict(tag=args.tag, styles=styles, n=len(keys),
                counts={s: int((y == s).sum()) for s in styles},
                probe_acc_sae_z=acc_z, probe_acc_raw_hidden=acc_h,
                chance_majority=chance, p_binomial_vs_chance=p_binom,
                perm_accs=perm, live_dims=int(live.sum()),
                gate_style_encoded=gate,
                note="fast variant: z restricted to >=5%-active features; binomial test vs "
                     "majority chance; full 20-perm run infeasible under host CPU contention")
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s7a_summary_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = [("chance\n(majority)", chance, ps.INK3),
            (f"permuted\nlabels (x{args.n_perm})", float(np.mean(perm)), ps.INK3),
            ("raw hidden\n(3x896)", acc_h, ps.C_VIOLET),
            (f"SAE z\n({args.tag}, {live.sum()}d)", acc_z, ps.C_LIGHT)]
    for i, (l, v, c) in enumerate(bars):
        ax.bar(i, v, color=c, width=0.6)
        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("style probe accuracy (5-fold OOF)")
    verdict = ("style semantics linearly decodable — gate to (b)(c) PASSES" if gate
               else "style probe does not beat chance — (b)(c) negative expected")
    extra = ("; SAE z ≈ raw hidden" if abs(acc_z - acc_h) <= 0.03 else
             ("; SAE z more separable" if acc_z > acc_h else "; raw hidden better"))
    ps.conclusion_title(fig,
        f"S7a: probe acc z={acc_z:.2f} / hidden={acc_h:.2f} vs chance {chance:.2f} "
        f"(binomial p={p_binom:.1e}) — {verdict}{extra}",
        sub=f"{len(keys)} single-label samples, styles {summ['counts']}; multinomial logistic; "
            f"fast variant (see JSON note)")
    ps.save(fig, os.path.join(RESULTS_R3, f"s7a_probe_{args.tag}.png"))
    print("[s7a-fast] done")


if __name__ == "__main__":
    main()
