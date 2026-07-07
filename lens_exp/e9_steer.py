# E9 (round 2) steering causal check: edit retouch-token L11 latents along SAE decoder
# directions (+/- 1,2 sigma), replay through the E6-best L11-consuming readout path, and
# measure whether the rendered photometric shift moves monotonically in the expected direction.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2

from common import TOKENS, _lab, _cct_mired
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from e6_lib import build_variant_model, load_latent, FUSE_LAYERS
from e2_lib import load_input_tensor, render
from e9_sae import TopKSAE, HID
import plotstyle as ps
import matplotlib.pyplot as plt

ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
TARGET_DIMS = {"dL_mean": "light", "dCCT_mired": "colortemp"}  # steer the token that owns the dim


def photometric(img_bgr, ref_bgr):
    """dL* mean and dCCT (mired) of img relative to ref."""
    L1, _, _ = _lab(img_bgr); L0, _, _ = _lab(ref_bgr)
    return dict(dL=float(L1.mean() - L0.mean()),
                dCCT=float(_cct_mired(img_bgr) - _cct_mired(ref_bgr)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=11)
    ap.add_argument("--readout", default=None, help="e6 variant for the replay path (default: from e6 summary)")
    ap.add_argument("--n-samples", type=int, default=30)
    ap.add_argument("--max-side", type=int, default=512)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    # ---- choose readout path: must consume L11 for the steered token -> e6b always does ----
    variant = args.readout or "e6b"
    head, decoder, lat_kind, layers = build_variant_model(variant, dtype=torch.float32)
    ck = torch.load(os.path.join(RESULTS_R2, f"{variant}.pt"), weights_only=False)
    head.load_state_dict(ck["state"]["head"]); decoder.load_state_dict(ck["state"]["decoder"])
    head = head.to(torch.bfloat16).eval(); decoder = decoder.to(torch.bfloat16).eval()

    sck = torch.load(os.path.join(RESULTS_R2, f"e9_sae_l{args.layer}.pt"), weights_only=False)
    sae = TopKSAE(HID, sck["expansion"], sck["k"]).cuda().eval()
    sae.load_state_dict(sck["state"])

    con = pd.read_csv(os.path.join(RESULTS_R2, "e9_concepts.csv"))
    feats = []
    for dim, tok in TARGET_DIMS.items():
        sub = con[(con.dim == dim) & (con.token == tok)].reindex(
            con[(con.dim == dim) & (con.token == tok)].rho_train.abs().sort_values(ascending=False).index)
        if len(sub) < 2:  # fall back: any token owning that dim correlation
            sub = con[con.dim == dim].reindex(con[con.dim == dim].rho_train.abs().sort_values(ascending=False).index)
        feats += [dict(feature=int(r.feature), token=r.token, dim=dim, rho=float(r.rho_train))
                  for r in sub.head(2).itertuples()]
    print("[e9steer] features:", feats)

    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    keys = [k for k in split["test"] if os.path.exists(os.path.join(C0_DIR, k + ".npz"))][: args.n_samples]

    # sigma of each feature's activation over train samples (at its token)
    tr_keys = [k for k in split["train"] if os.path.exists(os.path.join(C0_DIR, k + ".npz"))]
    lat_tr = {t: [] for t in TOKENS}
    for k in tr_keys:
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        for t in TOKENS:
            lat_tr[t].append(d[f"lat_{t}"][args.layer].astype(np.float32))
        d.close()
    sigma = {}
    with torch.no_grad():
        for t in TOKENS:
            z = sae.encode(torch.from_numpy(np.stack(lat_tr[t])).cuda()).cpu().numpy()
            sigma[t] = z.std(axis=0)

    tok_idx = {t: i for i, t in enumerate(TOKENS)}
    nl = len(layers) if lat_kind == "fused" else 1
    recs = []
    for k in keys:
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        base_lat = load_latent(d, lat_kind, layers)[0].numpy()  # [2688 or 10752]
        d.close()
        x = load_input_tensor(rows[k]["input_path"], max_side=args.max_side)
        inp_bgr = cv2.imread(rows[k]["input_path"], cv2.IMREAD_COLOR)
        s = args.max_side / max(inp_bgr.shape[:2])
        if s < 1:
            inp_bgr = cv2.resize(inp_bgr, (int(inp_bgr.shape[1]*s), int(inp_bgr.shape[0]*s)))
        for fs in feats:
            t = fs["token"]; f = fs["feature"]
            dvec = sae.dec.data[f].detach().cpu().numpy()  # unit norm [896]
            sg = sigma[t][f] if sigma[t][f] > 1e-4 else 1.0
            if lat_kind == "fused":
                off = (tok_idx[t] * nl + list(layers).index(args.layer)) * HID
            else:
                off = tok_idx[t] * HID
            for a in ALPHAS:
                lat = base_lat.copy()
                lat[off:off + HID] += a * sg * dvec
                img = render(head, decoder, torch.from_numpy(lat).unsqueeze(0), x)
                if img.shape[:2] != inp_bgr.shape[:2]:
                    img = cv2.resize(img, (inp_bgr.shape[1], inp_bgr.shape[0]))
                ph = photometric(img, inp_bgr)
                recs.append(dict(key=k, feature=f, token=t, dim=fs["dim"], rho=fs["rho"],
                                 alpha=a, dL=ph["dL"], dCCT=ph["dCCT"]))
        torch.cuda.empty_cache()
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R2, "e9_steering.csv"), index=False)

    # ---- dose-response + direction consistency ----
    from scipy.stats import spearmanr
    summ = {"readout": variant, "n_samples": len(keys)}
    fig, axes = plt.subplots(1, len(feats), figsize=(3.6 * len(feats), 4.4), sharex=True)
    cons_all = []
    for ax, fs in zip(np.atleast_1d(axes), feats):
        f = fs["feature"]
        met = "dL" if fs["dim"] == "dL_mean" else "dCCT"
        sub = df[df.feature == f]
        exp_sign = np.sign(fs["rho"])
        cons = []
        for k, g in sub.groupby("key"):
            g = g.sort_values("alpha")
            r, _ = spearmanr(g.alpha, g[met])
            cons.append(np.sign(r) == exp_sign)
        cons = float(np.mean(cons))
        cons_all.append(cons)
        m = sub.groupby("alpha")[met].agg(["mean", "sem"])
        ax.errorbar(m.index, m["mean"], yerr=m["sem"], marker="o", color=ps.TOKEN_COLORS[fs["token"]], capsize=3)
        ax.axhline(0, color=ps.INK3, lw=0.8)
        ax.set_title(f"f{f} ({fs['token']})\n{fs['dim']} rho={fs['rho']:+.2f} | consist {cons:.0%}", fontsize=9)
        ax.set_xlabel("steering alpha (x sigma)")
        ax.set_ylabel(("render dL* vs input" if met == "dL" else "render dCCT (mired) vs input"))
        summ[f"f{f}"] = dict(dim=fs["dim"], rho_train=fs["rho"], consistency=cons,
                             slope=float(np.polyfit(m.index, m["mean"], 1)[0]))
    mean_cons = float(np.mean(cons_all))
    summ["mean_consistency"] = mean_cons
    verdict = ("steering moves outputs in the expected direction" if mean_cons >= 0.7
               else "steering effect weak/inconsistent")
    ps.conclusion_title(fig,
        f"E9 steering: mean direction-consistency {mean_cons:.0%} over {len(keys)} samples — {verdict}",
        sub=f"latent edit +/- alpha*sigma along SAE decoder direction at L{args.layer} retouch-token, "
            f"replayed offline through the {variant} readout (correlation != causation check)")
    ps.save(fig, os.path.join(RESULTS_R2, "e9_steering_curves.png"))
    json.dump(summ, open(os.path.join(RESULTS_R2, "e9_steer_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
