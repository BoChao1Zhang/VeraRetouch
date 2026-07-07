# S1 (round 3): nonlinear (2-layer MLP) probes vs linear ridge probes, same protocol,
# 25 layers x 3 tokens, 5-fold out-of-fold R^2 on all C0-captured samples (n=792).
# Answers Q1: does the peak layer move when the probe family is stronger?
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score

from common import TOKENS, TARGET_NAMES
from common_r3 import RESULTS_R3, TOKEN_SLICES, captured_keys, load_latents_all_layers, targets_matrix
import plotstyle as ps
import matplotlib.pyplot as plt

ALPHAS = np.logspace(-1, 5, 13)
N_LAYERS = 25


class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=256, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p_drop),
                                 nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def fit_predict_mlp(Xtr, Ytr, Xte, seed=0, device="cuda", epochs=300, patience=20):
    """standardize on train, fit with inner val split + early stop, return test preds."""
    torch.manual_seed(seed)
    mx, sx = Xtr.mean(0), Xtr.std(0) + 1e-8
    my, sy = Ytr.mean(0), Ytr.std(0) + 1e-8
    Xtr_ = torch.from_numpy((Xtr - mx) / sx).float()
    Ytr_ = torch.from_numpy((Ytr - my) / sy).float()
    Xte_ = torch.from_numpy((Xte - mx) / sx).float().to(device)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr_))
    n_val = max(20, len(idx) // 10)
    vi, ti = idx[:n_val], idx[n_val:]
    xt, yt = Xtr_[ti].to(device), Ytr_[ti].to(device)
    xv, yv = Xtr_[vi].to(device), Ytr_[vi].to(device)
    model = MLP(Xtr.shape[1], Ytr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best, best_state, bad = float("inf"), None, 0
    bs = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xt), device=device)
        for i in range(0, len(xt), bs):
            j = perm[i:i + bs]
            loss = nn.functional.mse_loss(model(xt[j]), yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = nn.functional.mse_loss(model(xv), yv).item()
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte_).cpu().numpy()
    return pred * sy + my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    keys = captured_keys()
    X = load_latents_all_layers(keys)          # token -> [N,25,896]
    T = targets_matrix(keys)                   # [N,26]
    Y = {t: T[:, TOKEN_SLICES[t]] for t in TOKENS}
    print(f"[s1] N={len(keys)}", flush=True)

    cv = KFold(5, shuffle=True, random_state=args.seed)
    recs = []
    for t in TOKENS:
        for l in range(N_LAYERS):
            Xl = X[t][:, l, :]
            # ---- linear (identical to E1 protocol, rerun on the full 792 for comparability)
            lin = make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
            pred_lin = cross_val_predict(lin, Xl, Y[t], cv=cv)
            # ---- MLP, same folds
            pred_mlp = np.zeros_like(Y[t])
            for tr, te in cv.split(Xl):
                pred_mlp[te] = fit_predict_mlp(Xl[tr], Y[t][tr], Xl[te], seed=args.seed)
            for probe, pred in (("ridge", pred_lin), ("mlp", pred_mlp)):
                r2_dims = [r2_score(Y[t][:, j], pred[:, j]) for j in range(Y[t].shape[1])]
                rec = dict(token=t, layer=l, probe=probe, r2_mean=float(np.mean(r2_dims)))
                for j, name in enumerate(TARGET_NAMES[t]):
                    rec[f"r2_{name}"] = float(r2_dims[j])
                recs.append(rec)
            print(f"[s1] {t} L{l:02d} ridge={recs[-2]['r2_mean']:.3f} mlp={recs[-1]['r2_mean']:.3f}",
                  flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R3, "s1_probe.csv"), index=False)

    # ---- summary: peak layers (excluding embedding index 0), shift MLP vs ridge
    summ = dict(n=len(keys), peak={}, shift={})
    piv = {p: df[df.probe == p].pivot(index="layer", columns="token", values="r2_mean")
           for p in ("ridge", "mlp")}
    for t in TOKENS:
        pk = {p: int(piv[p][t].iloc[1:].idxmax()) for p in ("ridge", "mlp")}
        summ["peak"][t] = dict(ridge=pk["ridge"], mlp=pk["mlp"],
                               r2_ridge=float(piv["ridge"][t].loc[pk["ridge"]]),
                               r2_mlp=float(piv["mlp"][t].loc[pk["mlp"]]))
        summ["shift"][t] = pk["mlp"] - pk["ridge"]
    max_shift = max(abs(v) for v in summ["shift"].values())
    summ["max_abs_shift"] = int(max_shift)
    summ["gate_peaks_stable"] = bool(max_shift < 3)
    json.dump(summ, open(os.path.join(RESULTS_R3, "s1_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---- figure: 3 panels, linear vs MLP curves ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharex=True)
    CLIP = -0.3  # embedding layer (idx 0) can degenerate for the MLP; clip for display only
    for ax, t in zip(axes, TOKENS):
        c = ps.TOKEN_COLORS[t]
        ax.plot(piv["ridge"].index, piv["ridge"][t].clip(lower=CLIP), color=c, marker="o",
                markersize=3, label=f"ridge (peak L{summ['peak'][t]['ridge']})")
        ax.plot(piv["mlp"].index, piv["mlp"][t].clip(lower=CLIP), color=ps.INK, marker="s",
                markersize=3, ls="--", label=f"MLP (peak L{summ['peak'][t]['mlp']})")
        for p, cc in (("ridge", c), ("mlp", ps.INK)):
            pk = summ["peak"][t][p]
            ax.axvline(pk, color=cc, lw=1.0, ls=":", alpha=0.7)
        ax.set_title(ps.TOKEN_LABELS[t], color=c)
        ax.set_xlabel("hidden_states index (0=emb, 24=final)")
        ax.legend(fontsize=8, loc="lower center")
    axes[0].set_ylabel("probe R² (5-fold OOF, mean over targets)")
    if summ["gate_peaks_stable"]:
        verdict = (f"S1: MLP probe reproduces the linear peak layers "
                   f"({'/'.join(str(summ['peak'][t]['mlp']) for t in TOKENS)} vs "
                   f"{'/'.join(str(summ['peak'][t]['ridge']) for t in TOKENS)}, max shift "
                   f"{max_shift}) — 'linear probe too weak' concern closed")
    else:
        verdict = (f"S1: MLP probe MOVES a peak layer (max shift {max_shift} ≥ 3) — "
                   f"report to main session before extending E6 configs")
    ps.conclusion_title(fig, verdict,
        sub=f"same 5-fold OOF protocol, n={len(keys)} C0 samples; MLP = 896→256→dim, dropout 0.1, "
            f"early-stopped; ridge rerun on identical data for comparability")
    ps.save(fig, os.path.join(RESULTS_R3, "s1_probe_curves.png"))
    print("[s1] done")


if __name__ == "__main__":
    main()
