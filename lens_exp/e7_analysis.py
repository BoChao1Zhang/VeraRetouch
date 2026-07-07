# E7 (round 2): head-level attention decomposition — per-(layer,head) IoU vs instructed <box>,
# positive/negative head identification (V-SEAM style), dilution test vs token-level round-1 IoU.
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import cv2

from common import parse_boxes, box_to_patch_mask, image_valid_patch_mask, TOKENS
from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
import plotstyle as ps
import matplotlib.pyplot as plt

GRID = 16
NL, NH = 24, 14


def topp_binarize(att_flat, valid_flat, p=0.5):
    """smallest set of valid patches holding >= p of the valid attention mass."""
    a = np.where(valid_flat, att_flat, 0.0).astype(np.float64)
    tot = a.sum()
    if tot <= 0:
        return np.zeros_like(valid_flat)
    order = np.argsort(a)[::-1]
    cs = np.cumsum(a[order])
    k = int(np.searchsorted(cs, p * tot)) + 1
    m = np.zeros_like(valid_flat)
    m[order[:k]] = True
    return m & valid_flat


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else np.nan


def topk_iou(att_flat, box_flat, valid_flat):
    k = int(box_flat.sum())
    if k == 0:
        return np.nan
    a = np.where(valid_flat, att_flat, -np.inf)
    top = np.zeros_like(box_flat)
    top[np.argsort(a)[-k:]] = True
    return iou(top, box_flat)


def chance_iou_topp(k, kb, V):
    ov = k * kb / V
    return ov / (k + kb - ov)


def sample_maps(d, token):
    att = d[f"att_{token}_self_full"].astype(np.float32)  # [24,14,256]
    return att


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="use round-1 attn dumps (98 samples), no split")
    args = ap.parse_args()

    if args.dev:
        from common import RESULTS as R1, DUMPS as D1, load_manifest
        dump_dir = os.path.join(D1, "attn")
        rows = {r["key"]: r for r in load_manifest()}
        out_dir = os.path.join(RESULTS_R2, "dev")
        train_keys = test_keys = sorted(k for k in rows if os.path.exists(os.path.join(dump_dir, k + ".npz")))
    else:
        dump_dir = C0_DIR
        rows = {r["key"]: r for r in load_manifest_r2()}
        split = get_split_r2()
        boxed = {k for k, r in rows.items() if r["has_box"]
                 and os.path.exists(os.path.join(dump_dir, k + ".npz"))}
        train_keys = sorted(boxed & set(split["train"]))
        test_keys = sorted(boxed & set(split["test"]))
        out_dir = RESULTS_R2
    os.makedirs(out_dir, exist_ok=True)
    all_keys = sorted(set(train_keys) | set(test_keys))
    print(f"E7 box samples: train={len(train_keys)} test={len(test_keys)}")

    # ---- pass 1: per-sample per-(token,layer,head) IoU ----
    recs = []
    tok_recs = []
    for key in all_keys:
        d = np.load(os.path.join(dump_dir, key + ".npz"))
        r = rows[key]
        im = cv2.imread(r["input_path"], cv2.IMREAD_COLOR)
        h, w = im.shape[:2]
        valid = image_valid_patch_mask(w, h, GRID).reshape(-1)
        V = int(valid.sum())
        box = np.zeros(GRID * GRID, dtype=bool)
        for b in parse_boxes(r["prompt"]):
            box |= box_to_patch_mask(b, w, h, GRID).reshape(-1)
        box &= valid
        kb = int(box.sum())
        if kb == 0:
            continue
        for t in TOKENS:
            fk = f"att_{t}_self_full"
            if fk not in d.files:
                continue
            att = d[fk].astype(np.float32)  # [24,14,256]
            mass = att.sum(axis=2)          # attention mass on image per head
            for l in range(NL):
                for hh in range(NH):
                    m = topp_binarize(att[l, hh], valid)
                    recs.append(dict(key=key, token=t, layer=l, head=hh,
                                     iou=iou(m, box), k=int(m.sum()),
                                     chance=chance_iou_topp(int(m.sum()), kb, V),
                                     iou_topk=topk_iou(att[l, hh], box, valid),
                                     img_mass=float(mass[l, hh]),
                                     split="train" if key in set(train_keys) else "test"))
            # token-level (head-mean) references
            hm = att.mean(axis=1)  # [24,256]
            best_topp = np.nanmax([iou(topp_binarize(hm[l], valid), box) for l in range(NL)])
            best_topk = np.nanmax([topk_iou(hm[l], box, valid) for l in range(NL)])
            am = att.mean(axis=(0, 1))
            tok_recs.append(dict(key=key, token=t,
                                 headmean_bestlayer_topp=float(best_topp),
                                 headmean_bestlayer_topk=float(best_topk),
                                 allmean_topp=float(iou(topp_binarize(am, valid), box)),
                                 chance_kb=chance_iou_topp(kb, kb, V),
                                 split="train" if key in set(train_keys) else "test"))
        d.close()
    df = pd.DataFrame(recs)
    tdf = pd.DataFrame(tok_recs)
    df.to_csv(os.path.join(out_dir, "e7_head_iou.csv"), index=False)
    tdf.to_csv(os.path.join(out_dir, "e7_token_iou.csv"), index=False)

    # ---- head matrices (median IoU over train samples) + pos/neg head selection ----
    tr = df[df.split == "train"]
    med = tr.groupby(["token", "layer", "head"]).agg(iou=("iou", "median"),
                                                     iou_topk=("iou_topk", "median"),
                                                     img_mass=("img_mass", "median")).reset_index()
    mats = {t: np.full((NL, NH), np.nan) for t in TOKENS}
    for _, rr in med.iterrows():
        mats[rr["token"]][int(rr["layer"]), int(rr["head"])] = rr["iou"]

    heads = {}
    for t in TOKENS:
        sub = med[med.token == t].copy()
        n_top = max(3, int(np.ceil(len(sub) * 0.05)))
        pos = sub.nlargest(n_top, "iou")
        neg_pool = sub[sub.img_mass >= sub.img_mass.median()]
        neg = neg_pool.nsmallest(n_top, "iou")
        heads[t] = dict(pos=[[int(a), int(b)] for a, b in zip(pos["layer"], pos["head"])],
                        neg=[[int(a), int(b)] for a, b in zip(neg["layer"], neg["head"])],
                        pos_median_iou=float(pos.iou.max()),
                        pos_median_iou_topk=float(sub.nlargest(n_top, "iou_topk").iou_topk.max()))
    json.dump(heads, open(os.path.join(out_dir, "e7_heads.json"), "w"), indent=1)

    # ---- dilution test on TEST samples: best positive head vs head-mean token level ----
    same = set(train_keys) == set(test_keys)
    te = df if same else df[df.split == "test"]
    tte = tdf if same else tdf[tdf.split == "test"]
    summ = dict(n_train=len(train_keys), n_test=len(test_keys))
    for t in TOKENS:
        ph = set(map(tuple, heads[t]["pos"]))
        sub = te[(te.token == t) & te.apply(lambda x: (x["layer"], x["head"]) in ph, axis=1)]
        best_per_sample = sub.groupby("key").iou.max()
        best_per_sample_topk = sub.groupby("key").iou_topk.max()
        hm = tte[tte.token == t]
        summ[t] = dict(
            poshead_median_iou_topp=float(best_per_sample.median()),
            poshead_median_iou_topk=float(best_per_sample_topk.median()),
            headmean_median_iou_topp=float(hm.headmean_bestlayer_topp.median()),
            headmean_median_iou_topk=float(hm.headmean_bestlayer_topk.median()),
            chance_median=float(te[te.token == t].chance.median()),
        )
    json.dump(summ, open(os.path.join(out_dir, "e7_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))

    # ---- figure: 3 head-IoU matrices ----
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 6.6))
    vmax = max(np.nanmax(mats[t]) for t in TOKENS)
    for ax, t in zip(axes, TOKENS):
        imh = ax.imshow(mats[t], aspect="auto", cmap="magma", vmin=0, vmax=vmax)
        for (l, hh) in heads[t]["pos"]:
            ax.scatter(hh, l, s=36, facecolors="none", edgecolors="#2a78d6", linewidths=1.6)
        for (l, hh) in heads[t]["neg"]:
            ax.scatter(hh, l, s=36, marker="x", color="#e34948", linewidths=1.4)
        ax.set_title(f"{ps.TOKEN_LABELS[t]}\nmax median IoU {np.nanmax(mats[t]):.2f}", color=ps.TOKEN_COLORS[t], fontsize=9)
        ax.set_xlabel("head"); ax.set_ylabel("layer" if t == TOKENS[0] else "")
        ax.grid(False)
    cax = fig.add_axes([0.925, 0.12, 0.014, 0.58])
    fig.colorbar(imh, cax=cax, label="median IoU vs <box> (top-p 0.5 binarized)")
    best_all = max(np.nanmax(mats[t]) for t in TOKENS)
    hm_med = np.median([summ[t]["headmean_median_iou_topp"] for t in TOKENS])
    ch = np.median([summ[t]["chance_median"] for t in TOKENS])
    verdict = ("head-mean DILUTES strong heads" if best_all > 1.5 * hm_med else
               "no strong single-head advantage over head-mean")
    ps.conclusion_title(fig,
        f"E7: best single-head median IoU {best_all:.2f} vs head-mean {hm_med:.2f} (chance {ch:.2f}) — {verdict}",
        sub=f"median over {len(train_keys)} box train samples; circles = positive heads (top 5%), "
            f"x = negative heads (bottom 5% among high image-mass); dilution stats on {len(test_keys)} test samples")
    fig.subplots_adjust(left=0.06, right=0.905, top=0.76, bottom=0.09, wspace=0.25)
    fig.savefig(os.path.join(out_dir, "e7_head_matrix.png"), bbox_inches="tight")
    plt.close(fig)
    print("E7 analysis done ->", out_dir)


if __name__ == "__main__":
    main()
