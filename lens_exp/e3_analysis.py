# E3: attention diagnosis — box IoU, luminance-extreme overlap, entropy vs error.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2
from scipy.stats import spearmanr

from common import (RESULTS, DUMPS, load_manifest, TOKENS, parse_boxes,
                    box_to_patch_mask, image_valid_patch_mask)
from metrics import image_metrics
import plotstyle as ps
import matplotlib.pyplot as plt

ATTN_DIR = os.path.join(DUMPS, "attn")
GRID = 16
OVERLAY_DIR = os.path.join(RESULTS, "e3_attn_overlay")


def get_att(d, token, step_tag="self"):
    k = f"att_{token}_{step_tag}_mean"
    if k not in d.files:
        k = f"att_{token}_gen_mean"
    return d[k].astype(np.float32)  # [24, 256]


def topk_iou(att_flat, target_mask_flat, valid_flat):
    """binarize attention by taking k=|target| patches among valid; IoU vs target."""
    k = int(target_mask_flat[valid_flat].sum())
    if k == 0:
        return np.nan, 0
    a = np.where(valid_flat, att_flat, -np.inf)
    top = np.zeros_like(target_mask_flat)
    top[np.argsort(a)[-k:]] = True
    inter = (top & target_mask_flat).sum()
    union = (top | target_mask_flat).sum()
    return inter / union, k


def chance_iou(k, V):
    ov = k * k / V
    return ov / (2 * k - ov)


def entropy_norm(att_flat, valid_flat):
    a = att_flat[valid_flat]
    a = a / (a.sum() + 1e-12)
    h = -(a * np.log(a + 1e-12)).sum()
    return float(h / np.log(len(a)))


def patch_luminance(img_path):
    im = cv2.imread(img_path, cv2.IMREAD_COLOR)
    h, w = im.shape[:2]
    S = max(h, w)
    sq = np.full((S, S), np.nan, dtype=np.float32)
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)
    oy, ox = (S - h) // 2, (S - w) // 2
    sq[oy:oy+h, ox:ox+w] = gray
    cell = S / GRID
    out = np.full((GRID, GRID), np.nan, dtype=np.float32)
    for i in range(GRID):
        for j in range(GRID):
            c = sq[int(i*cell):int((i+1)*cell), int(j*cell):int((j+1)*cell)]
            v = c[~np.isnan(c)]
            out[i, j] = v.mean() if len(v) else np.nan
    return out, (w, h)


def main():
    rows = {r["key"]: r for r in load_manifest()}
    keys = [k for k in sorted(rows) if os.path.exists(os.path.join(ATTN_DIR, k + ".npz"))]
    print(f"E3 samples: {len(keys)}")

    recs = []
    per_layer_iou = {t: [] for t in TOKENS}
    cache = {}
    for key in keys:
        r = rows[key]
        d = np.load(os.path.join(ATTN_DIR, key + ".npz"))
        lum, (w, h) = patch_luminance(r["input_path"])
        valid = image_valid_patch_mask(w, h, GRID).reshape(-1)
        V = int(valid.sum())
        boxes = parse_boxes(r["prompt"])
        box_mask = np.zeros(GRID * GRID, dtype=bool)
        for b in boxes:
            box_mask |= box_to_patch_mask(b, w, h, GRID).reshape(-1)
        box_mask &= valid

        # luminance extreme mask (top/bottom 10% among valid patches)
        lv = lum.reshape(-1)
        lv_valid = lv[valid]
        lo, hi = np.percentile(lv_valid, 10), np.percentile(lv_valid, 90)
        lum_mask = (valid & ((lv <= lo) | (lv >= hi)))

        # baseline pred error
        pred = os.path.join(RESULTS, "preds_baseline", key + ".png")
        if key not in cache:
            cache[key] = image_metrics(pred, r["gt_path"])["de00"] if os.path.exists(pred) else np.nan
        de00 = cache[key]

        mean3 = None
        for t in TOKENS:
            att = get_att(d, t)                    # [24, 256]
            mean3 = att if mean3 is None else mean3 + att
            iou_layers = np.array([topk_iou(att[l], box_mask, valid)[0] for l in range(att.shape[0])]) \
                if box_mask.any() else np.full(att.shape[0], np.nan)
            per_layer_iou[t].append(iou_layers)
            att_g = get_att(d, t, "gen")
            iou_layers_gen = np.array([topk_iou(att_g[l], box_mask, valid)[0] for l in range(att_g.shape[0])]) \
                if box_mask.any() else np.full(att_g.shape[0], np.nan)
            ent = np.mean([entropy_norm(att[l], valid) for l in range(att.shape[0])])
            # luminance lift
            att_m = att.mean(0)
            att_m = att_m / att_m[valid].sum()
            lift = float(att_m[lum_mask].sum() / (lum_mask.sum() / V))
            k = int(box_mask.sum())
            recs.append(dict(
                key=key, lang=r["lang"], token=t, has_box=int(box_mask.any()),
                box_patches=k, valid_patches=V,
                iou_best=np.nanmax(iou_layers) if box_mask.any() else np.nan,
                iou_meanL=np.nanmean(iou_layers) if box_mask.any() else np.nan,
                iou_best_gen=np.nanmax(iou_layers_gen) if box_mask.any() else np.nan,
                chance_iou=chance_iou(k, V) if k else np.nan,
                entropy=ent, lum_lift=lift, de00=de00,
                img_mass_mean=float(d[f"att_{t}_self_mass"].mean()) if f"att_{t}_self_mass" in d.files else np.nan,
            ))
        d.close()

    df = pd.DataFrame(recs)
    df.to_csv(os.path.join(RESULTS, "e3_attention.csv"), index=False)

    # ---- pick reporting layer: best median IoU across samples (mean of 3 tokens) ----
    stack = {t: np.stack(v) for t, v in per_layer_iou.items()}   # [N, 24]
    med_by_layer = np.nanmedian(np.nanmean(np.stack([stack[t] for t in TOKENS]), axis=0), axis=0)
    best_layer = int(np.nanargmax(med_by_layer))
    boxed = df[df.has_box == 1]
    piv_best = boxed.pivot_table(index="key", columns="token", values="iou_best")
    med_iou = float(boxed.iou_best.median())
    med_chance = float(boxed.chance_iou.median())

    rho_all = {}
    for t in TOKENS:
        sub = df[df.token == t].dropna(subset=["entropy", "de00"])
        rho, p = spearmanr(sub.entropy, sub.de00)
        rho_all[t] = (float(rho), float(p))

    lift_med = {t: float(df[df.token == t].lum_lift.median()) for t in TOKENS}

    json.dump(dict(best_layer=best_layer, median_iou_best=med_iou, median_chance=med_chance,
                   spearman=rho_all, lum_lift_median=lift_med,
                   iou_by_layer_median=med_by_layer.tolist(), n_box=int(len(piv_best))),
              open(os.path.join(RESULTS, "e3_summary.json"), "w"), indent=1)
    print("best layer", best_layer, "median IoU", med_iou, "chance", med_chance)
    print("spearman", rho_all)

    # ---- fig: IoU histogram (CN/EN) ----
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(11, 4.3))
    bins = np.linspace(0, 1, 21)
    for lang, c in (("CN", ps.C_LIGHT), ("EN", ps.C_COLORTEMP)):
        vals = boxed[boxed.lang == lang].groupby("key").iou_best.mean()
        axl.hist(vals, bins=bins, alpha=0.6, color=c, label=f"{lang} (n={len(vals)})")
    axl.axvline(med_chance, color=ps.C_HILITE, ls="--", lw=1.4)
    axl.text(med_chance + 0.01, axl.get_ylim()[1]*0.9, "chance", color=ps.C_HILITE, fontsize=8)
    axl.axvline(med_iou, color=ps.INK, ls="-", lw=1.4)
    axl.text(med_iou + 0.01, axl.get_ylim()[1]*0.78, f"median {med_iou:.2f}", fontsize=8)
    axl.set_xlabel("attention-vs-<box> IoU (best layer per sample)")
    axl.set_ylabel("samples")
    axl.legend()
    axl.set_title("IoU distribution")
    axr.plot(range(24), med_by_layer, color=ps.C_VIOLET, marker="o", markersize=3.5)
    axr.axhline(med_chance, color=ps.C_HILITE, ls="--", lw=1.2)
    axr.set_xlabel("layer"); axr.set_ylabel("median IoU")
    axr.set_title(f"median IoU by layer (peak L{best_layer})")
    looks = "does look at" if med_iou > 2 * med_chance else "barely looks at"
    ps.conclusion_title(fig,
        f"E3: retouch tokens {looks} the instructed <box> region — median IoU {med_iou:.2f} vs chance {med_chance:.2f}",
        sub=f"{len(piv_best)} box-annotated samples; top-k binarization (k = box size in patches), pad patches excluded")
    ps.save(fig, os.path.join(RESULTS, "e3_box_iou_hist.png"))

    # ---- fig: entropy vs de00 scatter ----
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), sharey=True)
    for ax, t in zip(axes, TOKENS):
        sub = df[df.token == t].dropna(subset=["entropy", "de00"])
        ax.scatter(sub.entropy, sub.de00, s=14, color=ps.TOKEN_COLORS[t], alpha=0.6, linewidths=0)
        z = np.polyfit(sub.entropy, sub.de00, 1)
        xs = np.linspace(sub.entropy.min(), sub.entropy.max(), 20)
        ax.plot(xs, np.polyval(z, xs), color=ps.INK, lw=1.4)
        rho, p = rho_all[t]
        ax.set_title(f"{ps.TOKEN_LABELS[t]}  rho={rho:.2f} (p={p:.1g})", color=ps.TOKEN_COLORS[t])
        ax.set_xlabel("attention entropy over image patches (norm.)")
    axes[0].set_ylabel("baseline deltaE00 vs gt")
    r_max = max(abs(v[0]) for v in rho_all.values())
    rel = "predicts" if r_max >= 0.3 else "weakly relates to"
    ps.conclusion_title(fig,
        f"E3: attention entropy {rel} retouch error (max |rho|={r_max:.2f})",
        sub=f"n={len(df[df.token=='light'].dropna(subset=['de00']))} samples, entropy = layer-mean, 'self' step")
    ps.save(fig, os.path.join(RESULTS, "e3_entropy_scatter.png"))

    # ---- overlays: 8 good + 8 bad ----
    os.makedirs(OVERLAY_DIR, exist_ok=True)
    key_iou = boxed.groupby("key").iou_best.mean().sort_values()
    picks = [("bad", k) for k in key_iou.index[:8]] + [("good", k) for k in key_iou.index[-8:]]
    for grade, key in picks:
        r = rows[key]
        d = np.load(os.path.join(ATTN_DIR, key + ".npz"))
        att = np.mean([get_att(d, t)[best_layer] for t in TOKENS], axis=0).reshape(GRID, GRID)
        im = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        S = max(h, w)
        amap = cv2.resize(att, (S, S), interpolation=cv2.INTER_CUBIC)
        oy, ox = (S - h) // 2, (S - w) // 2
        amap = amap[oy:oy+h, ox:ox+w]
        amap = (amap - amap.min()) / (amap.ptp() + 1e-9)
        heat = cv2.applyColorMap((amap * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        overlay = cv2.addWeighted(im, 0.45, heat, 0.55, 0)
        gt = cv2.imread(r["gt_path"], cv2.IMREAD_COLOR)
        boxes = parse_boxes(r["prompt"])
        im_box = im.copy()
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(im_box, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (72, 73, 227), max(2, S//400))
            cv2.rectangle(overlay, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), (255, 255, 255), max(2, S//400))
        iou = key_iou[key]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
        for ax, img, ttl in zip(axs, [im_box, overlay, gt],
                                ["input + instructed box", f"retouch-token attention (L{best_layer})", "gt (expert)"]):
            ax.imshow(img[..., ::-1]); ax.axis("off"); ax.set_title(ttl, fontsize=9)
        ps.conclusion_title(fig, f"{key}: IoU={iou:.2f} ({grade})")
        ps.save(fig, os.path.join(OVERLAY_DIR, f"{grade}_{key}_iou{iou:.2f}.png"))
    print("overlays saved")


if __name__ == "__main__":
    main()
