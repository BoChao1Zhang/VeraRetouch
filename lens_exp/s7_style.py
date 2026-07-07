# S7a (round 3): style -> activation. Mine style labels from user_want.txt (bilingual vocab),
# compute SAE feature selectivity per style, and run the decisive holdout probe:
# style linear probe on SAE z vs on raw hidden state (both at the retouch tokens, L11).
import os, sys, json, argparse, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from common import TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2
from common_r3 import RESULTS_R3
from e9_sae import TopKSAE
import plotstyle as ps
import matplotlib.pyplot as plt

# bilingual style vocabulary (matched on lowercased prompt)
STYLE_VOCAB = {
    "cinematic":  ["电影感", "电影级", "影院", "大片感", "cinematic", "movie", "film look",
                    "fantasy film", "sci-fi movie"],
    "film_retro": ["胶片", "胶卷", "菲林", "复古", "怀旧", "年代感", "film grain", "retro",
                    "vintage", "nostalgic", "kodak"],
    "warm_sunset": ["日落", "黄昏", "夕阳", "金黄", "暖冬", "sunset", "golden hour", "dusk",
                     "golden glow"],
    "cold_blue":  ["冷峻", "清冷", "冷酷", "冷淡", "冷色调", "moody blue", "blueish", "bluish",
                    "cool blue", "cold tone"],
    "cyberpunk":  ["赛博朋克", "赛博", "霓虹", "cyberpunk", "neon"],
    "fresh":      ["清新", "日系", "airy", "fresh and clean", "fresh, "],
    "dreamy":     ["朦胧", "梦幻", "柔焦", "dreamy", "hazy", "soft focus", "soft and dreamy"],
}
STYLES = list(STYLE_VOCAB)


def mine_labels(rows):
    """key -> style, single-label samples only (multi-hit samples dropped, counts reported)."""
    lab, multi = {}, 0
    for r in rows:
        p = r["prompt"].lower()
        hits = [s for s, ws in STYLE_VOCAB.items() if any(w.lower() in p for w in ws)]
        if len(hits) == 1:
            lab[r["key"]] = hits[0]
        elif len(hits) > 1:
            multi += 1
    return lab, multi


def load_sae(tag):
    if tag == "e9":
        ck = torch.load(os.path.join(RESULTS_R2, "e9_sae_l11.pt"), weights_only=False)
        sae = TopKSAE(896, ck["expansion"], ck["k"]).cuda().eval()
    else:
        ck = torch.load(os.path.join(RESULTS_R3, f"s6_sae_{tag}.pt"), weights_only=False)
        sae = TopKSAE(ck["d_in"], ck["expansion"], ck["k"]).cuda().eval()
    sae.load_state_dict(ck["state"])
    return sae


def encode_keys(keys, sae, layer=11):
    """returns raw hidden concat [N, 3*896] and SAE z concat [N, 3*D]."""
    H, Z = [], []
    with torch.no_grad():
        for k in keys:
            d = np.load(os.path.join(C0_DIR, k + ".npz"))
            h = np.stack([d[f"lat_{t}"][layer].astype(np.float32) for t in TOKENS])
            d.close()
            z = sae.encode(torch.from_numpy(h).cuda()).cpu().numpy()
            H.append(h.reshape(-1)); Z.append(z.reshape(-1))
    return np.stack(H), np.stack(Z)


def oof_probe_acc(X, y, seed=0, scale=False):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    pred = np.empty(len(y), dtype=object)
    for tr, te in cv.split(X, y):
        Xtr, Xte = X[tr], X[te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=3000, C=1.0)
        clf.fit(Xtr, y[tr])
        pred[te] = clf.predict(Xte)
    return float((pred == y).mean()), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l11x32", help="SAE: e9 | l11x32 | l11l14l23x8")
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    rows = load_manifest_r2()
    captured = {os.path.basename(p)[:-4] for p in
                __import__("glob").glob(os.path.join(C0_DIR, "*.npz"))}
    lab, multi = mine_labels(rows)
    lab = {k: v for k, v in lab.items() if k in captured}
    counts = pd.Series(lab).value_counts()
    styles = [s for s in STYLES if counts.get(s, 0) >= args.min_n]
    keys = sorted(k for k, v in lab.items() if v in styles)
    y = np.array([lab[k] for k in keys])
    print(f"[s7a] single-label={len(lab)} multi-dropped={multi} styles kept={styles}")
    print(counts.to_string())

    sae = load_sae(args.tag)
    H, Z = encode_keys(keys, sae)
    D = Z.shape[1] // 3

    # ---- probes: SAE z vs raw hidden vs chance ----
    acc_z, _ = oof_probe_acc(Z, y)
    acc_h, _ = oof_probe_acc(H, y, scale=True)
    chance_maj = float(pd.Series(y).value_counts(normalize=True).iloc[0])
    perm_acc = []
    for i in range(20):
        yp = rng.permutation(y)
        a, _ = oof_probe_acc(Z, yp, seed=i)
        perm_acc.append(a)
    perm_mu, perm_sd = float(np.mean(perm_acc)), float(np.std(perm_acc))
    zscore = (acc_z - perm_mu) / (perm_sd + 1e-9)
    gate = bool(acc_z > perm_mu + 3 * perm_sd and acc_z > chance_maj)

    # ---- per-feature selectivity (max over tokens): in-group vs out-group activation rate ----
    Z3 = Z.reshape(len(keys), 3, D)
    sel_rows = []
    for s in styles:
        m = y == s
        for ti, t in enumerate(TOKENS):
            A = Z3[:, ti, :]
            act_in = (A[m] > 0).mean(0)
            act_out = (A[~m] > 0).mean(0)
            sel = act_in * (act_in - act_out)          # frequent AND differential
            top = np.argsort(-sel)[:10]
            for f in top:
                sel_rows.append(dict(style=s, token=t, feature=int(f),
                                     act_in=float(act_in[f]), act_out=float(act_out[f]),
                                     selectivity=float(sel[f]),
                                     mean_z_in=float(A[m][:, f].mean()),
                                     mean_z_out=float(A[~m][:, f].mean())))
    sdf = pd.DataFrame(sel_rows).sort_values(["style", "selectivity"], ascending=[True, False])
    sdf.to_csv(os.path.join(RESULTS_R3, f"s7_selectivity_{args.tag}.csv"), index=False)

    summ = dict(tag=args.tag, styles=styles, counts={s: int(counts[s]) for s in styles},
                n=len(keys), n_multi_dropped=multi,
                probe_acc_sae_z=acc_z, probe_acc_raw_hidden=acc_h,
                chance_majority=chance_maj, perm_acc_mean=perm_mu, perm_acc_sd=perm_sd,
                z_vs_perm_zscore=float(zscore), gate_style_encoded=gate)
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s7a_summary_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---- figure: probe bars + selectivity heatmap of top features ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2), width_ratios=[0.7, 1.6])
    bars = [("chance\n(majority)", chance_maj, ps.INK3), ("permuted\nlabels", perm_mu, ps.INK3),
            ("raw hidden\n(3x896)", acc_h, ps.C_VIOLET), (f"SAE z\n({args.tag})", acc_z, ps.C_LIGHT)]
    for i, (l, v, c) in enumerate(bars):
        ax1.bar(i, v, color=c, width=0.6)
        ax1.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.errorbar(1, perm_mu, yerr=perm_sd, color=ps.INK, capsize=4)
    ax1.set_xticks(range(4)); ax1.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax1.set_ylabel("style probe accuracy (5-fold OOF)")
    top_feats = sdf.groupby("style").head(3)
    fl = [(r.style, r.token, r.feature) for r in top_feats.itertuples()]
    M = np.zeros((len(styles), len(fl)))
    tok_i = {t: i for i, t in enumerate(TOKENS)}
    for j, (s0, t0, f0) in enumerate(fl):
        A = Z3[:, tok_i[t0], f0]
        for i, s in enumerate(styles):
            M[i, j] = A[y == s].mean()
    M = M / (M.max(0, keepdims=True) + 1e-9)
    im = ax2.imshow(M, aspect="auto", cmap="magma")
    ax2.set_yticks(range(len(styles))); ax2.set_yticklabels(styles, fontsize=8)
    ax2.set_xticks(range(len(fl)))
    ax2.set_xticklabels([f"f{f}\n{t[:5]}\n({s[:6]})" for s, t, f in fl], fontsize=6)
    ax2.grid(False)
    fig.colorbar(im, ax=ax2, shrink=0.8, label="mean z (column-normalized)")
    ax2.set_title("style x top-3 selective features per style")
    verdict = ("style semantics ARE linearly decodable from SAE features — proceed to steering (b)"
               if gate else "style probe does not beat chance — (b)(c) will be stopped")
    extra = ("; SAE features MORE separable than raw hidden" if acc_z > acc_h + 0.03 else
             ("; raw hidden as good — dictionary adds no separation" if acc_z < acc_h + 0.03 else ""))
    ps.conclusion_title(fig,
        f"S7a: style probe acc {acc_z:.2f} (chance {chance_maj:.2f}, permuted {perm_mu:.2f}±{perm_sd:.2f}) — "
        f"{verdict}{extra}",
        sub=f"{len(keys)} single-label samples over {len(styles)} styles {dict((s, int(counts[s])) for s in styles)}; "
            f"multinomial logistic, retouch-token L11; permutation control x20")
    ps.save(fig, os.path.join(RESULTS_R3, f"s7a_probe_{args.tag}.png"))

    # persist labels for downstream steps
    json.dump(dict(labels={k: lab[k] for k in keys}, styles=styles),
              open(os.path.join(RESULTS_R3, "s7_labels.json"), "w"), indent=1)
    print("[s7a] done")


if __name__ == "__main__":
    main()
