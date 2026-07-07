# S6 (round 3) concept dictionary on scaled SAEs: identical protocol to E9
# (train 633 / test 159, per-feature best Spearman vs 26 photometric deltas,
# |rho|>0.3 gate on train, test reconfirmation), generalized to any s6 SAE checkpoint.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, rankdata

from common import TOKENS
from common_r2 import RESULTS_R2, C0_DIR, get_split_r2, load_targets_r2, DIM_NAMES
from common_r3 import RESULTS_R3
from e9_sae import TopKSAE
import plotstyle as ps
import matplotlib.pyplot as plt

SEQ3_DIR = os.path.join(RESULTS_R2, "dumps", "s6_seq3")


def load_lat(key, layers, src):
    d = np.load(os.path.join(src, key + ".npz"))
    out = {t: np.concatenate([d[f"lat_{t}"][l].astype(np.float32) for l in layers])
           for t in TOKENS}
    d.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="s6_sae_<tag>.pt")
    ap.add_argument("--rho", type=float, default=0.3)
    args = ap.parse_args()

    ck = torch.load(os.path.join(RESULTS_R3, f"s6_sae_{args.tag}.pt"), weights_only=False)
    layers = ck["layers"]
    src = C0_DIR if len(layers) == 1 else SEQ3_DIR
    sae = TopKSAE(ck["d_in"], ck["expansion"], ck["k"]).cuda().eval()
    sae.load_state_dict(ck["state"])
    D = sae.dec.shape[0]

    split = get_split_r2()
    targets = load_targets_r2()
    keys, acts, tg = {}, {t: {} for t in TOKENS}, {}
    for part in ("train", "test"):
        keys[part] = [k for k in split[part]
                      if os.path.exists(os.path.join(src, k + ".npz")) and k in targets]
        lat = {t: [] for t in TOKENS}
        for k in keys[part]:
            l = load_lat(k, layers, src)
            for t in TOKENS:
                lat[t].append(l[t])
        with torch.no_grad():
            for t in TOKENS:
                x = torch.from_numpy(np.stack(lat[t])).cuda()
                acts[t][part] = sae.encode(x).cpu().numpy()
        tg[part] = np.stack([targets[k] for k in keys[part]])
    n_tr = len(keys["train"])
    print(f"[s6dict:{args.tag}] layers={layers} D={D} train={n_tr} test={len(keys['test'])}",
          flush=True)

    rho_mat = np.zeros((D, len(DIM_NAMES)), dtype=np.float32)
    best_tok = np.zeros(D, dtype=np.int8)
    Tr = rankdata(tg["train"], axis=0)
    Tr = (Tr - Tr.mean(0)) / (Tr.std(0) + 1e-9)
    for ti, t in enumerate(TOKENS):
        A = acts[t]["train"]
        live = (A > 0).sum(axis=0) >= max(10, n_tr // 20)
        Ar = rankdata(A, axis=0)
        Ar = (Ar - Ar.mean(0)) / (Ar.std(0) + 1e-9)
        R = (Ar.T @ Tr) / n_tr
        R[~live] = 0
        upd = np.abs(R).max(axis=1) > np.abs(rho_mat).max(axis=1)
        rho_mat[upd] = R[upd]
        best_tok[upd] = ti

    best_dim = np.abs(rho_mat).argmax(axis=1)
    best_rho = rho_mat[np.arange(D), best_dim]
    cand = np.where(np.abs(best_rho) > args.rho)[0]
    cand = cand[np.argsort(-np.abs(best_rho[cand]))]
    ranked = np.argsort(-np.abs(best_rho))[: max(30, len(cand))]
    rec = []
    for f in ranked:
        t = TOKENS[best_tok[f]]
        a_te = acts[t]["test"][:, f]
        y_te = tg["test"][:, best_dim[f]]
        if (a_te > 0).sum() >= 5:
            r_te, p_te = spearmanr(a_te, y_te)
        else:
            r_te, p_te = np.nan, np.nan
        rec.append(dict(feature=int(f), token=t, dim=DIM_NAMES[best_dim[f]],
                        rho_train=float(best_rho[f]), rho_test=float(r_te), p_test=float(p_te),
                        passes_gate=bool(np.abs(best_rho[f]) > args.rho),
                        act_freq=float((acts[t]["train"][:, f] > 0).mean())))
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(RESULTS_R3, f"s6_concepts_{args.tag}.csv"), index=False)
    gated = df[df.passes_gate]
    conf = gated[(np.sign(gated.rho_test) == np.sign(gated.rho_train))
                 & (np.abs(gated.rho_test) > 0.2)]
    summ = dict(tag=args.tag, layers=layers, D=D, expansion=ck["expansion"], k=ck["k"],
                val_r2=ck["hist"][-1]["val_r2"], dead_frac=ck["hist"][-1]["dead_frac"],
                n_candidates=int(len(cand)), n_test_confirmed=int(len(conf)),
                rho_gate=args.rho, n_train=n_tr, n_test=len(keys["test"]),
                gate_capacity_explanation=bool(len(conf) >= 5))
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s6_dict_{args.tag}_summary.json"), "w"),
              indent=1)
    print(json.dumps(summ, indent=1))

    top = ranked[:20]
    fig, ax = plt.subplots(figsize=(12.5, max(5.2, 0.42 * len(top) + 2.2)))
    im = ax.imshow(rho_mat[top], aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(len(DIM_NAMES))); ax.set_xticklabels(DIM_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"f{f} ({TOKENS[best_tok[f]][:5]}, "
                        f"{df[df.feature == f].rho_test.iloc[0]:+.2f} test)" for f in top],
                       fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman rho (train)")
    ps.conclusion_title(fig,
        f"S6 [{args.tag}]: {len(cand)} pass |rho|>{args.rho}, {len(conf)} test-confirmed — "
        + ("capacity WAS the bottleneck" if summ["gate_capacity_explanation"] else
           "scaling does not surface monosemantic concepts"),
        sub=f"TopK-SAE layers={layers} x{ck['expansion']} k={ck['k']} "
            f"(val R2={summ['val_r2']:.3f}, dead={summ['dead_frac']:.1%}); E9 protocol "
            f"(train {n_tr} discover / test {len(keys['test'])} confirm)")
    ps.save(fig, os.path.join(RESULTS_R3, f"s6_concept_matrix_{args.tag}.png"))
    print(f"[s6dict:{args.tag}] done")


if __name__ == "__main__":
    main()
