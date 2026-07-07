# E9 (round 2) concept dictionary: SAE feature activations at retouch-token positions (L11)
# vs the 26-dim photometric-delta vector -> Spearman matrix, candidate concepts, top-8 grids.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import spearmanr

from common import TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2, load_targets_r2, DIM_NAMES
from e9_sae import TopKSAE, HID
import plotstyle as ps
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--rho", type=float, default=0.3)
    args = ap.parse_args()

    ck = torch.load(os.path.join(RESULTS_R2, f"e9_sae_l{args.layer}.pt"), weights_only=False)
    sae = TopKSAE(HID, ck["expansion"], ck["k"]).cuda().eval()
    sae.load_state_dict(ck["state"])
    D = sae.dec.shape[0]

    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    targets = load_targets_r2()
    keys = {}
    for part in ("train", "test"):
        keys[part] = [k for k in split[part]
                      if os.path.exists(os.path.join(C0_DIR, k + ".npz")) and k in targets]

    # encode retouch-token latents for all samples
    acts = {t: {} for t in TOKENS}   # token -> part -> [N, D]
    tg = {}
    for part in ("train", "test"):
        lat = {t: [] for t in TOKENS}
        for k in keys[part]:
            d = np.load(os.path.join(C0_DIR, k + ".npz"))
            for t in TOKENS:
                lat[t].append(d[f"lat_{t}"][args.layer].astype(np.float32))
            d.close()
        with torch.no_grad():
            for t in TOKENS:
                x = torch.from_numpy(np.stack(lat[t])).cuda()
                acts[t][part] = sae.encode(x).cpu().numpy()
        tg[part] = np.stack([targets[k] for k in keys[part]])

    # Spearman on TRAIN, per token; keep best token per feature
    n_tr = len(keys["train"])
    rho_mat = np.zeros((D, len(DIM_NAMES)), dtype=np.float32)
    best_tok = np.zeros(D, dtype=np.int8)
    # rank-transform once for speed (Spearman == Pearson on ranks)
    from scipy.stats import rankdata
    Tr = rankdata(tg["train"], axis=0)
    Tr = (Tr - Tr.mean(0)) / (Tr.std(0) + 1e-9)
    for ti, t in enumerate(TOKENS):
        A = acts[t]["train"]
        live = (A > 0).sum(axis=0) >= max(10, n_tr // 20)  # active in >=5% samples
        Ar = rankdata(A, axis=0)
        Ar = (Ar - Ar.mean(0)) / (Ar.std(0) + 1e-9)
        R = (Ar.T @ Tr) / n_tr  # [D, 26]
        R[~live] = 0
        upd = np.abs(R).max(axis=1) > np.abs(rho_mat).max(axis=1)
        rho_mat[upd] = R[upd]
        best_tok[upd] = ti

    best_dim = np.abs(rho_mat).argmax(axis=1)
    best_rho = rho_mat[np.arange(D), best_dim]
    cand = np.where(np.abs(best_rho) > args.rho)[0]
    cand = cand[np.argsort(-np.abs(best_rho[cand]))]
    print(f"[e9dict] candidates |rho|>{args.rho}: {len(cand)}")

    # verify on TEST split
    rec = []
    for f in cand:
        t = TOKENS[best_tok[f]]
        a_te = acts[t]["test"][:, f]
        y_te = tg["test"][:, best_dim[f]]
        if (a_te > 0).sum() >= 5:
            r_te, p_te = spearmanr(a_te, y_te)
        else:
            r_te, p_te = np.nan, np.nan
        rec.append(dict(feature=int(f), token=t, dim=DIM_NAMES[best_dim[f]],
                        rho_train=float(best_rho[f]), rho_test=float(r_te), p_test=float(p_te),
                        act_freq=float((acts[t]["train"][:, f] > 0).mean())))
    df = pd.DataFrame(rec)
    df.to_csv(os.path.join(RESULTS_R2, "e9_concepts.csv"), index=False)
    n_confirm = int((np.sign(df.rho_test) == np.sign(df.rho_train)).fillna(False)
                    .where(np.abs(df.rho_test) > 0.2).sum()) if len(df) else 0
    json.dump(dict(n_candidates=len(df), n_test_confirmed=n_confirm,
                   rho_gate=args.rho, n_train=n_tr, n_test=len(keys["test"])),
              open(os.path.join(RESULTS_R2, "e9_dict_summary.json"), "w"), indent=1)

    # ---------- figure: top-20 features x 26 dims ----------
    top = cand[:20]
    fig, ax = plt.subplots(figsize=(12.5, max(5.2, 0.42 * len(top) + 2.2)))
    M = rho_mat[top]
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-0.7, vmax=0.7)
    ax.set_xticks(range(len(DIM_NAMES))); ax.set_xticklabels(DIM_NAMES, rotation=90, fontsize=7)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([f"f{f} ({TOKENS[best_tok[f]][:5]}, {df[df.feature==f].rho_test.iloc[0]:+.2f} test)"
                        for f in top], fontsize=7)
    ax.grid(False)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman rho (train)")
    ps.conclusion_title(fig,
        f"E9 concept dictionary: {len(df)} SAE features pass |rho|>{args.rho} on train; "
        f"{n_confirm} keep |rho|>0.2 same-sign on test",
        sub=f"TopK-SAE (L{args.layer}, x{ck['expansion']}, k={ck['k']}); activation at retouch-token position vs "
            f"26 photometric deltas; y-label shows test-split rho")
    ps.save(fig, os.path.join(RESULTS_R2, "e9_concept_matrix.png"))

    # ---------- top-8 activation grids for the strongest 8 candidates ----------
    grid_dir = os.path.join(RESULTS_R2, "e9_feature_grids")
    os.makedirs(grid_dir, exist_ok=True)
    for f in cand[:8]:
        t = TOKENS[best_tok[f]]
        a = acts[t]["train"][:, f]
        order = np.argsort(-a)[:8]
        fig, axes = plt.subplots(2, 4, figsize=(13, 6.4))
        for ax, i in zip(axes.ravel(), order):
            k = keys["train"][i]
            im0 = cv2.imread(rows[k]["input_path"], cv2.IMREAD_COLOR)
            s = 320 / max(im0.shape[:2])
            ax.imshow(cv2.resize(im0, None, fx=s, fy=s)[..., ::-1]); ax.axis("off")
            ax.set_title(f"{k} act={a[i]:.1f}", fontsize=8)
        ps.conclusion_title(fig, f"E9 feature f{f} ({t}) top-8 activating inputs — "
                                 f"corr {DIM_NAMES[best_dim[f]]} rho={best_rho[f]:+.2f}")
        ps.save(fig, os.path.join(grid_dir, f"f{f}_{DIM_NAMES[best_dim[f]]}.png"))
    print("[e9dict] done")


if __name__ == "__main__":
    main()
