# S3 (round 3) analysis: does retouch-token attention follow the LANGUAGE?
# Conditions: orig (C0 dumps) / del (referent removed) / flip (referent box mirrored).
# Fixed a-priori mask recipe (= E8): mean over E7 positive heads of the 3 retouch tokens,
# top-p 0.5 binarized on valid patches; IoU vs the ORIGINAL box (and vs the flipped box).
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2
from scipy.stats import wilcoxon

from common import parse_boxes, box_to_patch_mask, image_valid_patch_mask, TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2
from common_r3 import RESULTS_R3
from e7_analysis import topp_binarize, iou
import plotstyle as ps
import matplotlib.pyplot as plt

S3_DIR = os.path.join(RESULTS_R3, "dumps_s3")
GRID = 16


def posmean_map(d, pos_heads):
    maps = []
    for t in TOKENS:
        fk = f"att_{t}_self_full"
        if fk not in d.files:
            return None
        att = d[fk].astype(np.float32)
        maps.append(np.mean([att[l, h] for l, h in pos_heads[t]], axis=0))
    return np.mean(maps, axis=0)  # [256]


def main():
    rows = {r["key"]: r for r in load_manifest_r2()}
    jobs = json.load(open(os.path.join(RESULTS_R3, "s3_jobs.json")))
    flip_info = {j["key"]: j for j in jobs if j["cond"] == "flip"}
    keys = sorted({j["key"] for j in jobs})
    heads = json.load(open(os.path.join(RESULTS_R2, "e7_heads.json")))
    pos_heads = {t: [tuple(x) for x in heads[t]["pos"]] for t in TOKENS}

    recs, maps = [], {}
    missing = []
    for k in keys:
        r = rows[k]
        im = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        valid = image_valid_patch_mask(w, h, GRID).reshape(-1)
        box = np.zeros(GRID * GRID, dtype=bool)
        for b in parse_boxes(r["prompt"]):
            box |= box_to_patch_mask(b, w, h, GRID).reshape(-1)
        box &= valid
        fbox = None
        if k in flip_info:
            fbox = box_to_patch_mask(tuple(flip_info[k]["flip_box"]), w, h, GRID).reshape(-1) & valid
        rec = dict(key=k, has_flip=k in flip_info)
        maps[k] = {}
        for cond in ("orig", "del", "flip"):
            if cond == "orig":
                p = os.path.join(C0_DIR, k + ".npz")
            else:
                p = os.path.join(S3_DIR, f"{k}__{cond}.npz")
                if cond == "flip" and k not in flip_info:
                    continue
            if not os.path.exists(p):
                missing.append((k, cond))
                continue
            d = np.load(p)
            m = posmean_map(d, pos_heads)
            d.close()
            if m is None:
                missing.append((k, cond))
                continue
            mb = topp_binarize(m, valid)
            rec[f"iou_{cond}"] = iou(mb, box)
            if cond == "flip" and fbox is not None:
                rec["iou_flip_vs_flipbox"] = iou(mb, fbox)
            maps[k][cond] = m.reshape(GRID, GRID)
        recs.append(rec)
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R3, "s3_iou.csv"), index=False)

    def wtest(a, b):
        s = df[[a, b]].dropna()
        return (float(wilcoxon(s[a], s[b]).pvalue), len(s)) if len(s) > 5 else (np.nan, len(s))

    p_del, n_del = wtest("iou_orig", "iou_del")
    p_flip, n_flip = wtest("iou_orig", "iou_flip")
    p_fbox, n_fbox = wtest("iou_flip_vs_flipbox", "iou_flip")
    med = {c: float(df[f"iou_{c}"].median()) for c in ("orig", "del", "flip")}
    med["flip_vs_flipbox"] = float(df["iou_flip_vs_flipbox"].median())
    collapse = med["orig"] - med["del"]
    gate_language = bool(p_del < 0.05 and collapse > 0)
    follows_flip = bool(p_fbox < 0.05 and med["flip_vs_flipbox"] > med["flip"])
    summ = dict(n=len(df), n_flip=int(df.has_flip.sum()), median_iou=med,
                p_orig_vs_del=p_del, p_orig_vs_flip=p_flip,
                p_flipbox_vs_origbox_under_flip=p_fbox,
                collapse_del=collapse, gate_language_driven=gate_language,
                strongest_evidence_attention_follows_flipped_box=follows_flip,
                missing=missing)
    json.dump(summ, open(os.path.join(RESULTS_R3, "s3_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- main figure: paired IoU ----------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.9), width_ratios=[1.15, 1])
    conds = ["iou_orig", "iou_del", "iou_flip"]
    lbls = ["orig", "del referent", "flip referent\n(vs ORIG box)"]
    for _, rr in df.iterrows():
        xs = [i for i, c in enumerate(conds) if not np.isnan(rr.get(c, np.nan))]
        ys = [rr[c] for c in conds if not np.isnan(rr.get(c, np.nan))]
        ax.plot(xs, ys, color=ps.INK3, lw=0.7, alpha=0.5)
    rng = np.random.default_rng(0)
    for i, c in enumerate(conds):
        vals = df[c].dropna().values
        ax.scatter(np.full(len(vals), i) + rng.uniform(-0.06, 0.06, len(vals)), vals,
                   s=16, color=ps.C_LIGHT, alpha=0.75, zorder=5)
        ax.scatter([i], [np.median(vals)], marker="_", s=600, color=ps.C_HILITE, zorder=6)
    ax.set_xticks(range(3)); ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel("IoU vs original <box> (pos-head mask, top-p 0.5)")
    ax.set_title(f"Wilcoxon orig vs del p={p_del:.2g} (n={n_del}); orig vs flip p={p_flip:.2g}")
    sub = df[df.has_flip].dropna(subset=["iou_flip", "iou_flip_vs_flipbox"])
    ax2.scatter(sub.iou_flip, sub.iou_flip_vs_flipbox, s=22, color=ps.C_ORANGE, alpha=0.8)
    lim = max(0.05, sub.iou_flip.max(), sub.iou_flip_vs_flipbox.max()) * 1.1
    ax2.plot([0, lim], [0, lim], color=ps.INK3, lw=1, ls="--")
    ax2.set_xlabel("flip condition: IoU vs ORIGINAL box")
    ax2.set_ylabel("flip condition: IoU vs FLIPPED box")
    ax2.set_title(f"above diagonal = attention moved with the language (p={p_fbox:.2g}, n={n_fbox})")
    if gate_language:
        verdict = f"del collapses IoU {med['orig']:.2f}→{med['del']:.2f} — language-driven grounding CONFIRMED"
    else:
        verdict = (f"IoU survives referent deletion ({med['orig']:.2f}→{med['del']:.2f}, p={p_del:.2f}), "
                   f"ignores flipped box — saliency, not language")
    if follows_flip:
        verdict += f"; attention FOLLOWS the flipped box ({med['flip_vs_flipbox']:.2f} vs {med['flip']:.2f})"
    ps.conclusion_title(fig, "S3: " + verdict,
        sub=f"34 box test samples (23 with a non-degenerate flip); eager online re-inference; "
            f"mask recipe fixed from R2 E7 positive heads (no per-condition tuning)")
    ps.save(fig, os.path.join(RESULTS_R3, "s3_iou_conditions.png"))

    # ---------- qualitative: 3 samples x 3 conditions heatmaps ----------
    sub = df[df.has_flip].dropna(subset=["iou_orig", "iou_del", "iou_flip"]).copy()
    sub["drop"] = sub.iou_orig - sub.iou_del
    picks = sub.sort_values("drop", ascending=False).key.tolist()[:3]
    fig, axes = plt.subplots(3, 3, figsize=(11.4, 10.6))
    for ri, k in enumerate(picks):
        r = rows[k]
        im = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        S = max(w, h)
        pad = np.full((S, S, 3), 114, np.uint8)
        oy, ox = (S - h) // 2, (S - w) // 2
        pad[oy:oy + h, ox:ox + w] = im
        disp = cv2.resize(pad, (512, 512))[..., ::-1]
        for ci, cond in enumerate(("orig", "del", "flip")):
            ax = axes[ri, ci]
            ax.imshow(disp)
            if cond in maps[k]:
                mm = cv2.resize(maps[k][cond], (512, 512), interpolation=cv2.INTER_CUBIC)
                lo, hi = np.percentile(mm, 2), np.percentile(mm, 98)
                mm = np.clip((mm - lo) / (hi - lo + 1e-9), 0, 1)
                ax.imshow(mm, cmap="magma", alpha=0.45)
            for b in parse_boxes(r["prompt"]):
                x1, y1, x2, y2 = b
                ax.add_patch(plt.Rectangle(((x1 * w + ox) / S * 512, (y1 * h + oy) / S * 512),
                                           (x2 - x1) * w / S * 512, (y2 - y1) * h / S * 512,
                                           fill=False, edgecolor="#2a78d6", lw=2))
            if cond == "flip" and k in flip_info:
                x1, y1, x2, y2 = flip_info[k]["flip_box"]
                ax.add_patch(plt.Rectangle(((x1 * w + ox) / S * 512, (y1 * h + oy) / S * 512),
                                           (x2 - x1) * w / S * 512, (y2 - y1) * h / S * 512,
                                           fill=False, edgecolor="#e34948", lw=2, ls="--"))
            ax.axis("off")
            v = df[df.key == k][f"iou_{cond}"].iloc[0]
            ax.set_title(f"{k} {cond} IoU={v:.2f}", fontsize=9)
    ps.conclusion_title(fig, "S3 qualitative: attention vs language perturbation\n"
                             "(blue = original box, red dashed = flipped box)")
    ps.save(fig, os.path.join(RESULTS_R3, "s3_heatmaps.png"))
    print("[s3] analysis done")


if __name__ == "__main__":
    main()
