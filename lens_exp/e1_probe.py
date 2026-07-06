# E1: layer-wise ridge probes from retouch-token latents to (input,gt)-derived targets.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score

from common import RESULTS, DUMPS, load_manifest, TOKENS, TARGET_NAMES
import plotstyle as ps
import matplotlib.pyplot as plt

DUMP_DIR = os.path.join(DUMPS, "baseline")
ALPHAS = np.logspace(-1, 5, 13)


def load_data():
    rows = load_manifest()
    targets = json.load(open(os.path.join(RESULTS, "probe_targets.json")))
    keys, lats, ys = [], {t: [] for t in TOKENS}, {t: [] for t in TOKENS}
    for r in rows:
        p = os.path.join(DUMP_DIR, r["key"] + ".npz")
        if not os.path.exists(p) or r["key"] not in targets:
            continue
        d = np.load(p)
        for t in TOKENS:
            lats[t].append(d[f"lat_{t}"].astype(np.float32))
            ys[t].append(np.asarray(targets[r["key"]][t], dtype=np.float32))
        keys.append(r["key"])
    X = {t: np.stack(lats[t]) for t in TOKENS}   # [N, 25, 896]
    Y = {t: np.stack(ys[t]) for t in TOKENS}     # [N, dim]
    return keys, X, Y


def probe_all(keys, X, Y, seed=0):
    recs = []
    cv = KFold(5, shuffle=True, random_state=seed)
    for t in TOKENS:
        n_layers = X[t].shape[1]
        for l in range(n_layers):
            Xl = X[t][:, l, :]
            model = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            pred = cross_val_predict(model, Xl, Y[t], cv=cv)
            r2_dims = [r2_score(Y[t][:, j], pred[:, j]) for j in range(Y[t].shape[1])]
            rec = dict(token=t, layer=l, r2_mean=float(np.mean(r2_dims)))
            for j, name in enumerate(TARGET_NAMES[t]):
                rec[f"r2_{name}"] = float(r2_dims[j])
            recs.append(rec)
            print(f"{t} L{l:02d} R2={rec['r2_mean']:.3f}", flush=True)
    return pd.DataFrame(recs)


def main():
    keys, X, Y = load_data()
    print(f"N={len(keys)} samples", flush=True)
    df = probe_all(keys, X, Y)
    df.to_csv(os.path.join(RESULTS, "e1_layer_probe.csv"), index=False)

    # gate + best layer (mean over tokens, excluding embedding layer 0 for readout choice)
    piv = df.pivot(index="layer", columns="token", values="r2_mean")
    mean_curve = piv[TOKENS].mean(axis=1)
    final_idx = piv.index.max()
    l_star = int(mean_curve.iloc[1:].idxmax())
    gaps = {t: float(piv[t].iloc[1:].max() - piv[t].loc[final_idx]) for t in TOKENS}
    gap_mean = float(mean_curve.iloc[1:].max() - mean_curve.loc[final_idx])
    gate = gap_mean >= 0.05 or max(gaps.values()) >= 0.05
    summary = dict(l_star=l_star, final_idx=int(final_idx), gap_mean=gap_mean,
                   gaps=gaps, gate_pass=bool(gate),
                   r2_final={t: float(piv[t].loc[final_idx]) for t in TOKENS},
                   r2_peak={t: float(piv[t].iloc[1:].max()) for t in TOKENS},
                   peak_layer={t: int(piv[t].iloc[1:].idxmax()) for t in TOKENS},
                   n=len(keys))
    with open(os.path.join(RESULTS, "e1_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    # ---- figure: 3 subplot layer-R2 curves ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharex=True)
    for ax, t in zip(axes, TOKENS):
        c = ps.TOKEN_COLORS[t]
        ax.plot(piv.index, piv[t], color=c, marker="o", markersize=3.5)
        pk = summary["peak_layer"][t]
        ax.axvline(pk, color=ps.INK3, lw=1.2, ls="--")
        ax.scatter([final_idx], [piv[t].loc[final_idx]], color=ps.C_HILITE, zorder=5, s=45,
                   label=f"final layer R²={piv[t].loc[final_idx]:.2f}")
        ax.scatter([pk], [piv[t].loc[pk]], color=c, zorder=5, s=45,
                   edgecolor=ps.INK, linewidth=0.8,
                   label=f"peak L{pk} R²={piv[t].loc[pk]:.2f}")
        ax.set_title(ps.TOKEN_LABELS[t], color=c)
        ax.set_xlabel("hidden_states index (0=emb, 24=final)")
        ax.legend(fontsize=8, loc="lower center")
    axes[0].set_ylabel("probe R² (5-fold OOF, mean over targets)")
    verdict = ("suboptimal" if gate else "near-optimal")
    ps.conclusion_title(fig,
        f"E1: retouch info peaks at layer {l_star} vs final 24 (mean ΔR²={gap_mean:+.03f}) — "
        f"final-layer readout is {verdict}",
        sub=f"ridge probes, N={len(keys)} ArtEdit-Lr samples (CN+EN), targets derived from (input, gt) pairs; "
            f"red dot = current readout [-1]; dashed line = per-token peak")
    ps.save(fig, os.path.join(RESULTS, "e1_layer_curve.png"))
    print("saved e1_layer_curve.png")


if __name__ == "__main__":
    main()
