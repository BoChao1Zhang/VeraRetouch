# S7 controls (round 3), anti-"language parroting":
#  A. synth: same style phrase x 16 different images -> does the style feature's activation
#     vary with the image? (parroting = constant) + cross-style text specificity.
#  B. delstyle: natural style sample vs same sample with style words removed -> does the
#     feature survive on image content alone?
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch

from common import TOKENS
from common_r2 import C0_DIR
from common_r3 import RESULTS_R3
from s7_style import load_sae
from s7_steer import pick_features
import plotstyle as ps
import matplotlib.pyplot as plt

S7_DIR = os.path.join(RESULTS_R3, "dumps_s7")
LAYER = 11


def z_at(sae, path, token, feature):
    d = np.load(path)
    h = d[f"lat_{token}"][LAYER].astype(np.float32)
    d.close()
    with torch.no_grad():
        z = sae.encode(torch.from_numpy(h[None]).cuda())[0].cpu().numpy()
    return float(z[feature])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l11x32")
    args = ap.parse_args()
    sae = load_sae(args.tag)
    feats = pick_features(args.tag)
    jobs = json.load(open(os.path.join(RESULTS_R3, "s7_capture_jobs.json")))
    lab = json.load(open(os.path.join(RESULTS_R3, "s7_labels.json")))["labels"]

    # ---- A. synth grid: feature (rows) x condition ----
    recs = []
    for j in jobs:
        p = os.path.join(S7_DIR, j["uid"] + ".npz")
        if not os.path.exists(p):
            continue
        for fs in feats:
            recs.append(dict(kind=j["kind"], prompt_style=j["style"], base_key=j["base_key"],
                             feat_style=fs["style"], feature=fs["feature"], token=fs["token"],
                             z=z_at(sae, p, fs["token"], fs["feature"])))
    df = pd.DataFrame(recs)

    # natural activations of each feature on its own style group + neutral baseline (C0)
    nat = []
    for fs in feats:
        own = [k for k, v in lab.items() if v == fs["style"]]
        for k in own:
            p = os.path.join(C0_DIR, k + ".npz")
            if os.path.exists(p):
                nat.append(dict(cond="natural_own", feat_style=fs["style"],
                                z=z_at(sae, p, fs["token"], fs["feature"]), base_key=k))
        for k in {j["base_key"] for j in jobs if j["kind"] == "synth"}:
            p = os.path.join(C0_DIR, k + ".npz")
            if os.path.exists(p):
                nat.append(dict(cond="neutral_c0", feat_style=fs["style"],
                                z=z_at(sae, p, fs["token"], fs["feature"]), base_key=k))
    ndf = pd.DataFrame(nat)
    df.to_csv(os.path.join(RESULTS_R3, f"s7_controls_{args.tag}.csv"), index=False)

    summ = dict(tag=args.tag, features=feats)
    for fs in feats:
        sty = fs["style"]
        own_syn = df[(df.kind == "synth") & (df.prompt_style == sty) & (df.feat_style == sty)].z
        oth_syn = df[(df.kind == "synth") & (df.prompt_style != sty) & (df.feat_style == sty)].z
        neu = ndf[(ndf.cond == "neutral_c0") & (ndf.feat_style == sty)].z
        nat_own = ndf[(ndf.cond == "natural_own") & (ndf.feat_style == sty)].z
        dels = df[(df.kind == "delstyle") & (df.prompt_style == sty) & (df.feat_style == sty)]
        # paired: natural (C0) vs del for the same keys
        nat_pair = []
        for k in dels.base_key:
            p = os.path.join(C0_DIR, k + ".npz")
            if os.path.exists(p):
                nat_pair.append(z_at(sae, p, fs["token"], fs["feature"]))
        cv = float(own_syn.std() / (own_syn.mean() + 1e-9)) if len(own_syn) else np.nan
        summ[sty] = dict(
            synth_own_mean=float(own_syn.mean()), synth_own_cv=cv,
            synth_own_fire_rate=float((own_syn > 0).mean()),
            synth_other_mean=float(oth_syn.mean()),
            neutral_c0_mean=float(neu.mean()),
            natural_own_mean=float(nat_own.mean()),
            del_mean=float(dels.z.mean()) if len(dels) else np.nan,
            natural_paired_mean=float(np.mean(nat_pair)) if nat_pair else np.nan,
            n_del=int(len(dels)),
            image_varies=bool(cv > 0.25),
            survives_del=bool(len(dels) and dels.z.mean() > 0.5 * np.mean(nat_pair))
            if nat_pair else None,
        )
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s7_controls_{args.tag}_summary.json"), "w"),
              indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- figure ----------
    fig, axes = plt.subplots(1, len(feats), figsize=(4.6 * len(feats), 4.6), sharey=False)
    conds = ["synth_own", "synth_other", "neutral_c0", "natural_own", "natural_pair", "del"]
    lbl = {"synth_own": "synth\nown style", "synth_other": "synth\nother styles",
           "neutral_c0": "neutral\n(no style)", "natural_own": "natural\nown style",
           "natural_pair": "natural\n(del subset)", "del": "style word\nDELETED"}
    col = {"synth_own": ps.C_LIGHT, "synth_other": ps.INK3, "neutral_c0": ps.INK3,
           "natural_own": ps.C_COLORTEMP, "natural_pair": ps.C_COLORTEMP, "del": ps.C_HILITE}
    rng = np.random.default_rng(0)
    for ax, fs in zip(np.atleast_1d(axes), feats):
        sty = fs["style"]
        data = {
            "synth_own": df[(df.kind == "synth") & (df.prompt_style == sty) & (df.feat_style == sty)].z.values,
            "synth_other": df[(df.kind == "synth") & (df.prompt_style != sty) & (df.feat_style == sty)].z.values,
            "neutral_c0": ndf[(ndf.cond == "neutral_c0") & (ndf.feat_style == sty)].z.values,
            "natural_own": ndf[(ndf.cond == "natural_own") & (ndf.feat_style == sty)].z.values,
        }
        dels = df[(df.kind == "delstyle") & (df.prompt_style == sty) & (df.feat_style == sty)]
        pair = [z_at(sae, os.path.join(C0_DIR, k + ".npz"), fs["token"], fs["feature"])
                for k in dels.base_key if os.path.exists(os.path.join(C0_DIR, k + ".npz"))]
        data["natural_pair"] = np.array(pair)
        data["del"] = dels.z.values
        for i, c in enumerate(conds):
            v = data[c]
            if len(v) == 0:
                continue
            ax.scatter(np.full(len(v), i) + rng.uniform(-0.12, 0.12, len(v)), v, s=12,
                       color=col[c], alpha=0.7)
            ax.scatter([i], [np.mean(v)], marker="_", s=500, color=ps.INK, zorder=5)
        # paired lines natural -> del
        if len(dels) and len(pair) == len(dels):
            for zv, nv in zip(dels.z.values, pair):
                ax.plot([4, 5], [nv, zv], color=ps.INK3, lw=0.7, alpha=0.5)
        ax.set_xticks(range(len(conds))); ax.set_xticklabels([lbl[c] for c in conds], fontsize=7)
        ax.set_title(f"f{fs['feature']} ({sty}, {fs['token']})\n"
                     f"CV over images {summ[sty]['synth_own_cv']:.2f}", fontsize=9)
        ax.set_ylabel("SAE activation z")
    varies = [s for s in summ if isinstance(summ.get(s), dict) and summ[s].get("image_varies")]
    survives = [s for s in summ if isinstance(summ.get(s), dict) and summ[s].get("survives_del")]
    ps.conclusion_title(fig,
        f"S7 controls: activation varies with image for {len(varies)}/{len(feats)} features "
        f"(not pure parroting); survives style-word deletion for {survives or 'none'}",
        sub="synth = fixed style phrase x 16 images; del = natural prompt with style words removed; "
            "paired grey lines = same sample natural vs deleted")
    ps.save(fig, os.path.join(RESULTS_R3, f"s7_controls_{args.tag}.png"))
    print("[s7ctrl] done")


if __name__ == "__main__":
    main()
