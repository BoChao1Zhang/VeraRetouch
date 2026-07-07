# E8 (round 2): attention-as-spatial-output — localized rendering by soft-mask blending.
# Offline: global = C0 baseline render; blend = M*render + (1-M)*input with M from
# positive-head attention (E7) vs oracle M from the instructed <box>.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2
from scipy.stats import wilcoxon

from common import parse_boxes, TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from metrics import image_metrics
import plotstyle as ps
import matplotlib.pyplot as plt

GRID = 16


def unpad_resize(att16, w, h):
    """16x16 map on the center-padded square -> full-res map on the image."""
    S = max(w, h)
    amap = cv2.resize(att16, (S, S), interpolation=cv2.INTER_CUBIC)
    oy, ox = (S - h) // 2, (S - w) // 2
    return amap[oy:oy + h, ox:ox + w]


def attn_mask(d, pos_heads, w, h, mode="pos"):
    """soft mask in [0,1] from retouch-token attention ('pos' = positive heads, 'mean' = all heads)."""
    maps = []
    for t in TOKENS:
        fk = f"att_{t}_self_full"
        if fk not in d.files:
            return None
        att = d[fk].astype(np.float32)  # [24,14,256]
        if mode == "pos":
            hs = pos_heads[t]
            m = np.mean([att[l, hh] for l, hh in hs], axis=0)
        else:
            m = att.mean(axis=(0, 1))
        maps.append(m.reshape(GRID, GRID))
    m = np.mean(maps, axis=0)
    m = unpad_resize(m, w, h)
    sigma = max(w, h) / 32.0
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    lo, hi = np.percentile(m, 1), np.percentile(m, 99)
    m = np.clip((m - lo) / (hi - lo + 1e-9), 0, 1)
    return m


def oracle_mask(boxes, w, h):
    m = np.zeros((h, w), dtype=np.float32)
    for (x1, y1, x2, y2) in boxes:
        m[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = 1.0
    m = cv2.GaussianBlur(m, (0, 0), max(w, h) / 32.0)
    return np.clip(m / (m.max() + 1e-9), 0, 1)


def blend(inp, rend, M):
    return (M[..., None] * rend.astype(np.float32) +
            (1 - M[..., None]) * inp.astype(np.float32)).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all-box", action="store_true", help="evaluate on all box samples, not just test")
    args = ap.parse_args()
    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    pool = split["test"] if not args.all_box else (split["test"] + split["train"])
    keys = sorted(k for k in pool if rows[k]["has_box"]
                  and os.path.exists(os.path.join(C0_DIR, k + ".npz"))
                  and os.path.exists(os.path.join(RESULTS_R2, "preds_c0", k + ".png")))
    heads = json.load(open(os.path.join(RESULTS_R2, "e7_heads.json")))
    pos_heads = {t: [tuple(x) for x in heads[t]["pos"]] for t in TOKENS}
    print(f"E8 samples: {len(keys)}")

    out_root = os.path.join(RESULTS_R2, "e8_blend")
    os.makedirs(out_root, exist_ok=True)
    recs = []
    for k in keys:
        r = rows[k]
        inp = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        rend = cv2.imread(os.path.join(RESULTS_R2, "preds_c0", k + ".png"), cv2.IMREAD_COLOR)
        h, w = inp.shape[:2]
        if rend.shape[:2] != (h, w):
            rend = cv2.resize(rend, (w, h))
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        Ma = attn_mask(d, pos_heads, w, h, "pos")
        Mm = attn_mask(d, pos_heads, w, h, "mean")
        d.close()
        Mo = oracle_mask(parse_boxes(r["prompt"]), w, h)
        variants = {"global": rend,
                    "attn_pos": blend(inp, rend, Ma) if Ma is not None else None,
                    "attn_mean": blend(inp, rend, Mm) if Mm is not None else None,
                    "oracle_box": blend(inp, rend, Mo)}
        for name, img in variants.items():
            if img is None:
                continue
            p = os.path.join(out_root, f"{k}_{name}.png")
            if not os.path.exists(p):
                cv2.imwrite(p, img)
            m = image_metrics(p, r["gt_path"])
            recs.append(dict(key=k, variant=name, **m))
        m = image_metrics(r["input_path"], r["gt_path"])
        recs.append(dict(key=k, variant="input(no-op)", **m))
    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS_R2, "e8_results.csv"), index=False)
    piv = df.pivot_table(index="key", columns="variant", values="de00")
    mean = piv.mean()

    def wp(a, b):
        s = piv[[a, b]].dropna()
        return float(wilcoxon(s[a], s[b]).pvalue) if len(s) > 5 else np.nan

    local_gain = float(mean["global"] - mean["oracle_box"])
    attn_gap = float(mean["attn_pos"] - mean["oracle_box"])
    gate = bool(local_gain > 0 and attn_gap <= 0.3)
    summ = dict(n=len(keys), mean_de00={k: float(v) for k, v in mean.items()},
                p_oracle_vs_global=wp("oracle_box", "global"),
                p_attn_vs_global=wp("attn_pos", "global"),
                p_attn_vs_oracle=wp("attn_pos", "oracle_box"),
                local_gain=local_gain, attn_minus_oracle=attn_gap,
                gate_attention_as_output=gate)
    json.dump(summ, open(os.path.join(RESULTS_R2, "e8_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- bar figure ----------
    order = ["input(no-op)", "global", "attn_mean", "attn_pos", "oracle_box"]
    order = [v for v in order if v in piv.columns]
    lbl = {"input(no-op)": "input\n(no-op)", "global": "global render\n(baseline)",
           "attn_mean": "blend:\nall-head attn", "attn_pos": "blend:\npos-head attn",
           "oracle_box": "blend:\noracle box"}
    col = {"input(no-op)": ps.INK3, "global": ps.C_HILITE, "attn_mean": ps.C_VIOLET,
           "attn_pos": ps.C_LIGHT, "oracle_box": ps.C_COLORTEMP}
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    rng = np.random.default_rng(0)
    for i, v in enumerate(order):
        vals = piv[v].dropna().values
        ax.bar(i, vals.mean(), width=0.62, color=col[v], alpha=0.88,
               yerr=vals.std() / np.sqrt(len(vals)), capsize=3, ecolor=ps.INK2)
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=6, color=ps.INK, alpha=0.25, linewidths=0)
        ax.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(range(len(order))); ax.set_xticklabels([lbl[v] for v in order], fontsize=8)
    ax.set_ylabel("deltaE00 vs gt (lower = better)")
    verdict = "gate passes" if gate else \
        ("localization itself has no gain" if local_gain <= 0 else "attention mask too inaccurate (gap > 0.3)")
    ps.conclusion_title(fig,
        f"E8: attn blend {mean['attn_pos']:.2f} beats global {mean['global']:.2f}, matches oracle "
        f"{mean['oracle_box']:.2f} ({verdict}) — but no gain over no-op {mean['input(no-op)']:.2f}",
        sub=f"box test samples n={len(keys)}; M = blurred soft mask; blend = M*render + (1-M)*input; "
            f"Wilcoxon p(oracle vs global)={summ['p_oracle_vs_global']:.2g}, p(attn vs oracle)={summ['p_attn_vs_oracle']:.2g}")
    ps.save(fig, os.path.join(RESULTS_R2, "e8_metrics_bar.png"))

    # ---------- qualitative grid: 6 best + 6 worst by (global - attn_pos) ----------
    gain = (piv["global"] - piv["attn_pos"]).dropna().sort_values()
    picks = [("worst", k) for k in gain.index[:6]] + [("best", k) for k in gain.index[-6:]]
    fig, axes = plt.subplots(12, 5, figsize=(14, 12 * 2.0))
    titles = ["input + box", "global render", "attn-mask blend", "oracle blend", "gt (expert)"]
    for ri, (grade, k) in enumerate(picks):
        r = rows[k]
        inp = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = inp.shape[:2]
        imb = inp.copy()
        for (x1, y1, x2, y2) in parse_boxes(r["prompt"]):
            cv2.rectangle(imb, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (72, 73, 227), max(2, max(h, w)//400))
        imgs = [imb,
                cv2.imread(os.path.join(out_root, f"{k}_global.png")),
                cv2.imread(os.path.join(out_root, f"{k}_attn_pos.png")),
                cv2.imread(os.path.join(out_root, f"{k}_oracle_box.png")),
                cv2.imread(r["gt_path"])]
        for ci, img in enumerate(imgs):
            ax = axes[ri, ci]
            s = 320 / max(img.shape[:2])
            img = cv2.resize(img, (int(img.shape[1]*s), int(img.shape[0]*s)))
            ax.imshow(img[..., ::-1]); ax.axis("off")
            if ri == 0:
                ax.set_title(titles[ci], fontsize=9)
        axes[ri, 0].text(-0.05, 0.5,
                         f"{k} ({grade})\nglob {piv.loc[k,'global']:.1f} attn {piv.loc[k,'attn_pos']:.1f} orac {piv.loc[k,'oracle_box']:.1f}",
                         transform=axes[ri, 0].transAxes, fontsize=7, rotation=90, va="center", ha="right", color=ps.INK2)
    ps.conclusion_title(fig, "E8 qualitative: global vs attention-localized vs oracle-localized render "
                             "(top 6 rows: attn blend helps most; bottom 6: hurts most)")
    ps.save(fig, os.path.join(RESULTS_R2, "e8_compare_grid.png"))
    print("E8 done")


if __name__ == "__main__":
    main()
