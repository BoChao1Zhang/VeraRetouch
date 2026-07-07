# S7b (round 3): activation -> pixels. Steer the top style-selective SAE features
# (+/- alpha*sigma along the decoder direction at the owning retouch token, L11), replay
# through the e6b readout (E9 protocol), and measure computable PIXEL STYLE SIGNATURES:
# dose-response of the steered style's signature + cross-style specificity.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import torch
import cv2
from scipy.stats import spearmanr

from common import TOKENS, _lab
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from common_r3 import RESULTS_R3
from e6_lib import build_variant_model, load_latent
from e2_lib import load_input_tensor, render
from s7_style import load_sae
import plotstyle as ps
import matplotlib.pyplot as plt

ALPHAS = (-2.0, -1.0, 0.0, 1.0, 2.0)
HID = 896
SIG_STYLES = ("cinematic", "cyberpunk", "fresh")


def sig_components(img_bgr):
    """raw pixel-signature components; combined into z-scored style scores downstream."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32) / 255.0
    v = hsv[..., 2].astype(np.float32) / 255.0
    L, a, b = _lab(img_bgr)
    chrom = (s > 0.10) & (v > 0.05)
    hh = h[chrom] if chrom.sum() > 100 else h.reshape(-1)

    def mass(lo, hi):
        return float(((hh >= lo) & (hh <= hi)).mean())

    shadow = L <= np.percentile(L, 25)
    return dict(
        ot_mass=mass(5, 25) + mass(85, 105),          # orange + teal
        mc_mass=mass(95, 135) + mass(145, 175),       # blue-cyan + magenta
        gc_mass=mass(40, 100),                        # green-cyan
        sat_mean=float(s[chrom].mean() if chrom.sum() > 100 else s.mean()),
        L_mean=float(L.mean()), L_std=float(L.std()),
        shadow_sat=float(s[shadow].mean()), shadow_b=float(b[shadow].mean()),
    )


# style score = mean of signed z-scored components (z over the alpha=0 renders)
SIG_DEF = {
    "cinematic": [("ot_mass", +1), ("shadow_sat", -1), ("L_std", -1)],
    "cyberpunk": [("mc_mass", +1), ("sat_mean", +1), ("shadow_b", -1)],
    "fresh":     [("L_mean", +1), ("sat_mean", -1), ("gc_mass", +1)],
}


def style_scores(comp_df, mu, sd):
    out = {}
    for sty, terms in SIG_DEF.items():
        z = np.zeros(len(comp_df))
        for c, sgn in terms:
            z += sgn * (comp_df[c].values - mu[c]) / (sd[c] + 1e-9)
        out[sty] = z / len(terms)
    return out


def synth_selectivity(tag, style):
    """fallback for styles with <20 natural single-label samples (e.g. cyberpunk/fresh):
    selectivity from the synth captures (fixed style phrase x 16 images) vs the SAME
    16 neutral images' C0 activations."""
    import glob
    from s7_style import load_sae
    sae = load_sae(tag)
    s7_dir = os.path.join(RESULTS_R3, "dumps_s7")
    files = sorted(glob.glob(os.path.join(s7_dir, f"synth__{style}__*.npz")))
    base_keys = [os.path.basename(p)[:-4].split("__")[-1] for p in files]
    rows = []
    with torch.no_grad():
        for src, paths in (("in", files),
                           ("out", [os.path.join(C0_DIR, k + ".npz") for k in base_keys])):
            for p in paths:
                if not os.path.exists(p):
                    continue
                d = np.load(p)
                h = np.stack([d[f"lat_{t}"][11].astype(np.float32) for t in TOKENS])
                d.close()
                z = sae.encode(torch.from_numpy(h).cuda()).cpu().numpy()
                rows.append((src, z))
    Zin = np.stack([z for s, z in rows if s == "in"])    # [n,3,D]
    Zout = np.stack([z for s, z in rows if s == "out"])
    best = None
    for ti, t in enumerate(TOKENS):
        ai = (Zin[:, ti] > 0).mean(0)
        ao = (Zout[:, ti] > 0).mean(0)
        sel = ai * (ai - ao)
        f = int(np.argsort(-sel)[0])
        cand = dict(style=style, feature=f, token=t, selectivity=float(sel[f]),
                    act_in=float(ai[f]), source="synth", n_in=len(Zin))
        if best is None or cand["selectivity"] > best["selectivity"]:
            best = cand
    return best


def pick_features(tag):
    sel = pd.read_csv(os.path.join(RESULTS_R3, f"s7_selectivity_{tag}.csv"))
    feats = []
    for sty in SIG_STYLES:
        sub = sel[(sel["style"] == sty) & (sel.act_in >= 0.25)].sort_values(
            "selectivity", ascending=False)
        if len(sub) == 0:
            sub = sel[sel["style"] == sty].sort_values("selectivity", ascending=False)
        if len(sub) == 0:
            feats.append(synth_selectivity(tag, sty))
            continue
        r = sub.iloc[0]
        feats.append(dict(style=sty, feature=int(r.feature), token=r.token,
                          selectivity=float(r.selectivity), act_in=float(r.act_in),
                          source="natural"))
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="l11x32")
    ap.add_argument("--n-samples", type=int, default=24)
    ap.add_argument("--max-side", type=int, default=512)
    ap.add_argument("--layer", type=int, default=11)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True

    sae = load_sae(args.tag)
    feats = pick_features(args.tag)
    print("[s7b] steering features:", feats, flush=True)

    variant = "e6b"   # consumes L11 for every token
    head, decoder, lat_kind, layers = build_variant_model(variant, dtype=torch.float32)
    ck = torch.load(os.path.join(RESULTS_R2, f"{variant}.pt"), weights_only=False)
    head.load_state_dict(ck["state"]["head"]); decoder.load_state_dict(ck["state"]["decoder"])
    head = head.to(torch.bfloat16).eval(); decoder = decoder.to(torch.bfloat16).eval()

    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    keys = [k for k in split["test"] if os.path.exists(os.path.join(C0_DIR, k + ".npz"))][: args.n_samples]
    tr_keys = [k for k in split["train"] if os.path.exists(os.path.join(C0_DIR, k + ".npz"))]

    # sigma of each steered feature's activation over train samples at its token
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
    nl = len(layers)
    csv_path = os.path.join(RESULTS_R3, f"s7b_steering_{args.tag}.csv")
    if not os.path.exists(csv_path):
        recs = []
        for k in keys:
            d = np.load(os.path.join(C0_DIR, k + ".npz"))
            base_lat = load_latent(d, lat_kind, layers)[0].numpy()
            d.close()
            x = load_input_tensor(rows[k]["input_path"], max_side=args.max_side)
            for fs in feats:
                t, f = fs["token"], fs["feature"]
                dvec = sae.dec.data[f].detach().cpu().numpy()
                sg = sigma[t][f] if sigma[t][f] > 1e-4 else 1.0
                off = (tok_idx[t] * nl + list(layers).index(args.layer)) * HID
                for a in ALPHAS:
                    lat = base_lat.copy()
                    lat[off:off + HID] += a * sg * dvec
                    img = render(head, decoder, torch.from_numpy(lat).unsqueeze(0), x)
                    comp = sig_components(img)
                    recs.append(dict(key=k, steer_style=fs["style"], feature=f, token=t,
                                     alpha=a, **comp))
            torch.cuda.empty_cache()
            print(f"[s7b] {k} done", flush=True)
        pd.DataFrame(recs).to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)

    # z-normalize components over alpha=0 renders
    base = df[df.alpha == 0]
    comp_cols = ["ot_mass", "mc_mass", "gc_mass", "sat_mean", "L_mean", "L_std",
                 "shadow_sat", "shadow_b"]
    mu = {c: base[c].mean() for c in comp_cols}
    sd = {c: base[c].std() for c in comp_cols}
    for sty, z in style_scores(df, mu, sd).items():
        df[f"score_{sty}"] = z

    # dose-response + specificity: steer style A, slope of score B vs alpha (per sample, relative to alpha=0)
    spec = np.zeros((len(SIG_STYLES), len(SIG_STYLES)))
    cons = {}
    for i, sa in enumerate(SIG_STYLES):
        sub = df[df.steer_style == sa]
        for j, sb in enumerate(SIG_STYLES):
            slopes, consist = [], []
            for k, g in sub.groupby("key"):
                g = g.sort_values("alpha")
                rel = g[f"score_{sb}"].values - g[f"score_{sb}"].values[list(g.alpha).index(0.0)]
                slopes.append(np.polyfit(g.alpha.values, rel, 1)[0])
                r, _ = spearmanr(g.alpha.values, g[f"score_{sb}"].values)
                consist.append(np.sign(r) if not np.isnan(r) else 0)
            spec[i, j] = float(np.median(slopes))
            if sa == sb:
                cons[sa] = float(np.mean(np.array(consist) > 0))
    diag = np.diag(spec)
    off = spec[~np.eye(len(SIG_STYLES), dtype=bool)]
    gate_dose = bool((diag > 0).all() and np.median(np.abs(off)) < np.median(np.abs(diag)))
    summ = dict(tag=args.tag, features=feats, n_samples=len(keys),
                self_slopes={s: float(d) for s, d in zip(SIG_STYLES, diag)},
                self_consistency=cons,
                median_abs_off_diag=float(np.median(np.abs(off))),
                median_abs_diag=float(np.median(np.abs(diag))),
                gate_dose_response_specific=gate_dose,
                specificity_matrix=[[float(v) for v in row] for row in spec])
    json.dump(summ, open(os.path.join(RESULTS_R3, f"s7b_summary_{args.tag}.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- figure: dose-response curves (3 panels) + specificity matrix ----------
    fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.4), width_ratios=[1, 1, 1, 0.9])
    for ax, sty in zip(axes[:3], SIG_STYLES):
        sub = df[df.steer_style == sty]
        curves = []
        for k, g in sub.groupby("key"):
            g = g.sort_values("alpha")
            rel = g[f"score_{sty}"].values - g[f"score_{sty}"].values[list(g.alpha).index(0.0)]
            curves.append(rel)
            ax.plot(sorted(sub.alpha.unique()), rel, color=ps.C_LIGHT, alpha=0.15, lw=0.8)
        C = np.stack(curves)
        al = sorted(sub.alpha.unique())
        ax.plot(al, np.median(C, 0), marker="o", color=ps.INK, lw=2.2, zorder=5)
        ax.fill_between(al, np.percentile(C, 25, 0), np.percentile(C, 75, 0),
                        color=ps.C_LIGHT, alpha=0.25)
        ax.axhline(0, color=ps.INK3, lw=0.8)
        f = [x for x in feats if x["style"] == sty][0]
        ax.set_title(f"steer f{f['feature']} ({sty})\nself-slope {summ['self_slopes'][sty]:+.3f}, "
                     f"consist {cons[sty]:.0%}", fontsize=9)
        ax.set_xlabel("alpha (x sigma)")
        ax.set_ylabel(f"Δ {sty} signature (z)" if sty == SIG_STYLES[0] else "")
    axm = axes[3]
    vmax = np.abs(spec).max() + 1e-9
    im = axm.imshow(spec, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axm.set_xticks(range(3)); axm.set_xticklabels([s[:6] for s in SIG_STYLES], fontsize=8)
    axm.set_yticks(range(3)); axm.set_yticklabels([s[:6] for s in SIG_STYLES], fontsize=8)
    axm.set_xlabel("measured signature"); axm.set_ylabel("steered style")
    for i in range(3):
        for j in range(3):
            axm.text(j, i, f"{spec[i, j]:+.2f}", ha="center", va="center", fontsize=8)
    axm.grid(False); axm.set_title("median slope matrix", fontsize=9)
    fig.colorbar(im, ax=axm, shrink=0.8)
    verdict = ("style features are causal AND specific pixel-style knobs" if gate_dose else
               "no clean specific dose-response — style features are weak/entangled knobs")
    ps.conclusion_title(fig,
        f"S7b: self-slopes {', '.join(f'{s}:{v:+.3f}' for s, v in summ['self_slopes'].items())} "
        f"(off-diag median |slope| {summ['median_abs_off_diag']:.3f}) — {verdict}",
        sub=f"steering = +/- alpha*sigma along SAE ({args.tag}) decoder direction at L11 retouch token, "
            f"replayed through e6b readout on {len(keys)} test images; signature = mean z of "
            f"style-specific pixel components")
    ps.save(fig, os.path.join(RESULTS_R3, f"s7b_dose_response_{args.tag}.png"))
    print("[s7b] done")


if __name__ == "__main__":
    main()
