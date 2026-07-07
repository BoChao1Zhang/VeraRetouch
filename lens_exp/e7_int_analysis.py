# E7 (round 2) intervention analysis: paired deltaE00 (intervened vs C0 baseline) + latent drift.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from common import TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2
from metrics import image_metrics
import plotstyle as ps
import matplotlib.pyplot as plt


def main():
    rows = {r["key"]: r for r in load_manifest_r2()}
    lams = []
    for d in sorted(os.listdir(RESULTS_R2)):
        if d.startswith("preds_e7int_lam"):
            lams.append(d[len("preds_e7int_lam"):])
    recs = []
    for lam in lams:
        pdir = os.path.join(RESULTS_R2, f"preds_e7int_lam{lam}")
        ldir = os.path.join(RESULTS_R2, "dumps", f"e7int_lam{lam}")
        for fn in sorted(os.listdir(pdir)):
            k = fn[:-4]
            base_p = os.path.join(RESULTS_R2, "preds_c0", k + ".png")
            if not os.path.exists(base_p):
                continue
            de_int = image_metrics(os.path.join(pdir, fn), rows[k]["gt_path"])["de00"]
            de_base = image_metrics(base_p, rows[k]["gt_path"])["de00"]
            drift = np.nan
            lp = os.path.join(ldir, k + ".npz")
            if os.path.exists(lp):
                di = np.load(lp); d0 = np.load(os.path.join(C0_DIR, k + ".npz"))
                ds = []
                for t in TOKENS:
                    a = di[f"lat_{t}"][24].astype(np.float32)
                    b = d0[f"lat_{t}"][24].astype(np.float32)
                    ds.append(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-9))
                drift = float(np.mean(ds))
                di.close(); d0.close()
            recs.append(dict(key=k, lam=float(lam), de00_int=de_int, de00_base=de_base, latent_drift=drift))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R2, "e7_intervention.csv"), index=False)

    summ = {}
    fig, axes = plt.subplots(1, len(lams), figsize=(5.4 * len(lams), 5.0), squeeze=False)
    for ax, lam in zip(axes[0], lams):
        sub = df[df.lam == float(lam)]
        d = sub.de00_int - sub.de00_base
        p = float(wilcoxon(sub.de00_int, sub.de00_base).pvalue) if len(sub) > 5 else np.nan
        summ[f"lam{lam}"] = dict(n=len(sub), mean_delta=float(d.mean()), median_delta=float(d.median()),
                                 wilcoxon_p=p, mean_latent_drift=float(sub.latent_drift.mean()),
                                 frac_improved=float((d < 0).mean()))
        lim = max(sub.de00_int.max(), sub.de00_base.max()) * 1.05
        ax.plot([0, lim], [0, lim], color=ps.INK3, lw=1, ls="--")
        ax.scatter(sub.de00_base, sub.de00_int, s=26, color=ps.C_LIGHT, alpha=0.75, linewidths=0)
        ax.set_xlabel("baseline deltaE00"); ax.set_ylabel("head-rescaled deltaE00")
        ax.set_title(f"lambda={lam}: mean {d.mean():+.2f} (p={p:.2g}), "
                     f"{(d<0).mean():.0%} improved, drift {sub.latent_drift.mean():.3f}", fontsize=9.5)
    worst = min(summ.values(), key=lambda s: abs(s["mean_delta"]))
    strongest = max(summ.values(), key=lambda s: abs(s["mean_delta"]))
    direction = "improves" if strongest["mean_delta"] < 0 and strongest["wilcoxon_p"] < 0.05 else \
                "worsens" if strongest["mean_delta"] > 0 and strongest["wilcoxon_p"] < 0.05 else "does not significantly change"
    ps.conclusion_title(fig,
        f"E7 intervention: amplifying positive / suppressing negative heads {direction} retouch quality "
        f"(best |effect| {strongest['mean_delta']:+.2f} deltaE00)",
        sub="V-SEAM-style rescale on image-token attention columns, renormalized; box test subset; below diagonal = improved")
    ps.save(fig, os.path.join(RESULTS_R2, "e7_intervention_scatter.png"))
    json.dump(summ, open(os.path.join(RESULTS_R2, "e7_int_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
