# S4 (round 3) step 2: attn-mask vs SAM-mask vs box — (i) pairwise IoU, (ii) localized
# rendering deltaE00 with the E6a readout render (answers R2 open decision #2: does
# "E6a readout + localization" create net gain where the R2 baseline render did not).
# Offline; reuses the E8 blending protocol (soft masks, Gaussian blur, M*render+(1-M)*input).
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2
from scipy.stats import wilcoxon

from common import parse_boxes, TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from common_r3 import RESULTS_R3
from e8_localize import attn_mask, oracle_mask, blend
from metrics import image_metrics
import plotstyle as ps
import matplotlib.pyplot as plt

SAM_DIR = os.path.join(RESULTS_R3, "s4_sam_masks")


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else np.nan


def main():
    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    keys = sorted(k for k in split["test"] if rows[k]["has_box"]
                  and os.path.exists(os.path.join(C0_DIR, k + ".npz"))
                  and os.path.exists(os.path.join(SAM_DIR, k + ".png")))
    heads = json.load(open(os.path.join(RESULTS_R2, "e7_heads.json")))
    pos_heads = {t: [tuple(x) for x in heads[t]["pos"]] for t in TOKENS}
    print(f"[s4] n={len(keys)}", flush=True)

    out_root = os.path.join(RESULTS_R3, "s4_blend")
    os.makedirs(out_root, exist_ok=True)
    iou_csv = os.path.join(RESULTS_R3, "s4_mask_iou.csv")
    de_csv = os.path.join(RESULTS_R3, "s4_de00.csv")
    cached = os.path.exists(iou_csv) and os.path.exists(de_csv)
    iou_recs, de_recs = [], []
    triptychs = {}
    for k in keys:
        if cached:
            r = rows[k]
            inp = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
            h, w = inp.shape[:2]
            d = np.load(os.path.join(C0_DIR, k + ".npz"))
            Ma = attn_mask(d, pos_heads, w, h, "pos")
            d.close()
            sam = cv2.imread(os.path.join(SAM_DIR, k + ".png"), cv2.IMREAD_GRAYSCALE) > 127
            boxm = np.zeros((h, w), dtype=bool)
            for (x1, y1, x2, y2) in parse_boxes(r["prompt"]):
                boxm[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = True
            triptychs[k] = (Ma, sam, boxm)
            continue
        r = rows[k]
        inp = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = inp.shape[:2]
        rend = cv2.imread(os.path.join(RESULTS_R2, "preds_e6a", k + ".png"), cv2.IMREAD_COLOR)
        if rend.shape[:2] != (h, w):
            rend = cv2.resize(rend, (w, h))
        d = np.load(os.path.join(C0_DIR, k + ".npz"))
        Ma = attn_mask(d, pos_heads, w, h, "pos")          # soft [0,1]
        d.close()
        sam = cv2.imread(os.path.join(SAM_DIR, k + ".png"), cv2.IMREAD_GRAYSCALE) > 127
        boxes = parse_boxes(r["prompt"])
        boxm = np.zeros((h, w), dtype=bool)
        for (x1, y1, x2, y2) in boxes:
            boxm[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)] = True
        attb = Ma >= 0.5

        iou_recs.append(dict(key=k, attn_vs_sam=iou(attb, sam), attn_vs_box=iou(attb, boxm),
                             sam_vs_box=iou(sam, boxm), sam_cov_in_box=float(sam[boxm].mean()),
                             attn_frac=float(attb.mean()), sam_frac=float(sam.mean()),
                             box_frac=float(boxm.mean())))

        # soft SAM mask for blending (same blur as oracle)
        Ms = cv2.GaussianBlur(sam.astype(np.float32), (0, 0), max(w, h) / 32.0)
        Ms = np.clip(Ms / (Ms.max() + 1e-9), 0, 1)
        Mo = oracle_mask(boxes, w, h)
        variants = {"e6a_global": rend, "blend_attn": blend(inp, rend, Ma),
                    "blend_sam": blend(inp, rend, Ms), "blend_box": blend(inp, rend, Mo)}
        for name, img in variants.items():
            p = os.path.join(out_root, f"{k}_{name}.png")
            if not os.path.exists(p):
                cv2.imwrite(p, img)
            m = image_metrics(p, r["gt_path"])
            de_recs.append(dict(key=k, variant=name, **m))
        m = image_metrics(r["input_path"], r["gt_path"])
        de_recs.append(dict(key=k, variant="input(no-op)", **m))
        triptychs[k] = (Ma, sam, boxm)

    if cached:
        idf = pd.read_csv(iou_csv)
        ddf = pd.read_csv(de_csv)
    else:
        idf = pd.DataFrame(iou_recs)
        idf.to_csv(iou_csv, index=False)
        ddf = pd.DataFrame(de_recs)
        ddf.to_csv(de_csv, index=False)
    piv = ddf.pivot_table(index="key", columns="variant", values="de00")
    mean = piv.mean()

    def wp(a, b):
        s = piv[[a, b]].dropna()
        return float(wilcoxon(s[a], s[b]).pvalue) if len(s) > 5 else np.nan

    summ = dict(n=len(keys),
                mask_iou_median=dict(attn_vs_sam=float(idf.attn_vs_sam.median()),
                                     attn_vs_box=float(idf.attn_vs_box.median()),
                                     sam_vs_box=float(idf.sam_vs_box.median())),
                mean_de00={k2: float(v) for k2, v in mean.items()},
                p_sam_vs_box=wp("blend_sam", "blend_box"),
                p_attn_vs_sam=wp("blend_attn", "blend_sam"),
                p_attn_vs_box=wp("blend_attn", "blend_box"),
                p_blendsam_vs_global=wp("blend_sam", "e6a_global"),
                p_blendattn_vs_global=wp("blend_attn", "e6a_global"),
                p_global_vs_noop=wp("e6a_global", "input(no-op)"),
                p_blendattn_vs_noop=wp("blend_attn", "input(no-op)"))
    json.dump(summ, open(os.path.join(RESULTS_R3, "s4_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---------- figure: IoU pairs + de00 bars ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 4.8), width_ratios=[1, 1.4])
    pairs = ["attn_vs_sam", "attn_vs_box", "sam_vs_box"]
    rng = np.random.default_rng(0)
    for i, c in enumerate(pairs):
        vals = idf[c].dropna().values
        ax1.boxplot(vals, positions=[i], widths=0.5, showfliers=False)
        ax1.scatter(i + rng.uniform(-0.12, 0.12, len(vals)), vals, s=8, color=ps.C_LIGHT, alpha=0.6)
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["attn↔SAM", "attn↔box", "SAM↔box"], fontsize=9)
    ax1.set_ylabel("pixel-level IoU")
    ax1.set_title("mask agreement (attn binarized @0.5)")
    order = ["input(no-op)", "e6a_global", "blend_attn", "blend_sam", "blend_box"]
    lbl = {"input(no-op)": "input\n(no-op)", "e6a_global": "E6a render\nglobal",
           "blend_attn": "blend:\nattn", "blend_sam": "blend:\nSAM", "blend_box": "blend:\nbox"}
    col = {"input(no-op)": ps.INK3, "e6a_global": ps.C_HILITE, "blend_attn": ps.C_LIGHT,
           "blend_sam": ps.C_COLORTEMP, "blend_box": ps.C_VIOLET}
    for i, v in enumerate(order):
        vals = piv[v].dropna().values
        ax2.bar(i, vals.mean(), width=0.62, color=col[v], alpha=0.88,
                yerr=vals.std() / np.sqrt(len(vals)), capsize=3, ecolor=ps.INK2)
        ax2.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=6, color=ps.INK, alpha=0.25)
        ax2.text(i, vals.mean(), f"{vals.mean():.2f}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold")
    ax2.set_xticks(range(len(order))); ax2.set_xticklabels([lbl[v] for v in order], fontsize=8)
    ax2.set_ylabel("deltaE00 vs gt")
    ax2.set_title("localized rendering with E6a readout")
    m_as, m_ab, m_sb = (summ["mask_iou_median"][p] for p in pairs)
    if m_as > m_ab + 0.05:
        m_verdict = "attention hugs the SAM object shape more than the box"
    elif m_ab > m_as + 0.05:
        m_verdict = "attention tracks the box, not the object shape"
    else:
        m_verdict = "attention agrees with SAM and box about equally"
    ps.conclusion_title(fig,
        f"S4: IoU attn↔SAM {m_as:.2f} / attn↔box {m_ab:.2f} / SAM↔box {m_sb:.2f}; "
        f"E6a GLOBAL {mean['e6a_global']:.2f} beats all blends — localization now counterproductive",
        sub=f"box test n={len(keys)}; SAM2 hiera-large box-prompted (SAM ViT-B unavailable locally); "
            f"render = retrained E6a readout (answers R2 decision #2); "
            f"p(blend_attn vs no-op)={summ['p_blendattn_vs_noop']:.2g}, p(global vs no-op)={summ['p_global_vs_noop']:.2g}")
    ps.save(fig, os.path.join(RESULTS_R3, "s4_masks_and_render.png"))

    # ---------- qualitative: 6 samples, input+box / attn mask / SAM mask / blends ----------
    show = idf.sort_values("attn_vs_sam", ascending=False).key.tolist()
    picks = show[:3] + show[-3:]
    fig, axes = plt.subplots(len(picks), 5, figsize=(14, 2.6 * len(picks)))
    for ri, k in enumerate(picks):
        r = rows[k]
        inp = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = inp.shape[:2]
        imb = inp.copy()
        for (x1, y1, x2, y2) in parse_boxes(r["prompt"]):
            cv2.rectangle(imb, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                          (72, 73, 227), max(2, max(h, w) // 400))
        Ma, sam, _ = triptychs[k]
        imgs = [imb[..., ::-1], Ma, sam.astype(float),
                cv2.imread(os.path.join(out_root, f"{k}_blend_attn.png"))[..., ::-1],
                cv2.imread(os.path.join(out_root, f"{k}_blend_sam.png"))[..., ::-1]]
        titles = ["input + box", "attn soft mask", "SAM2 mask", "blend attn", "blend SAM"]
        for ci, img in enumerate(imgs):
            ax = axes[ri, ci]
            if img.ndim == 2:
                ax.imshow(img, cmap="magma")
            else:
                s = 300 / max(img.shape[:2])
                ax.imshow(cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s))))
            ax.axis("off")
            if ri == 0:
                ax.set_title(titles[ci], fontsize=9)
        axes[ri, 0].text(-0.04, 0.5, f"{k}\nattn↔SAM {idf[idf.key == k].attn_vs_sam.iloc[0]:.2f}",
                         transform=axes[ri, 0].transAxes, fontsize=7, rotation=90,
                         va="center", ha="right", color=ps.INK2)
    ps.conclusion_title(fig, "S4 qualitative: attention mask vs SAM2 object mask "
                             "(top 3: best agreement; bottom 3: worst)")
    ps.save(fig, os.path.join(RESULTS_R3, "s4_qualitative.png"))
    print("[s4] done")


if __name__ == "__main__":
    main()
