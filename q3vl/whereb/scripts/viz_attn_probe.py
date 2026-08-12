"""Success / failure panels for PR-ATT1-E1.

Every rule from CLAUDE.md's "spatial field visualisation discipline" is enforced
by :mod:`q3vl.whereb.viz` rather than re-implemented here:

* colour scale over **valid cells only**, sink cells drawn white (never filled);
* the grid -> image overlay uses ``grid_to_img``'s exact integer inverse map,
  never a resize;
* the numbers printed in the titles come from the **raw** field, and are the same
  numbers ``metrics.json`` carries -- colouring and arithmetic stay separate.

Failure cases are mandatory: "no failure cases" means they were not looked for.
Two kinds are drawn, because this experiment has two distinct ones -- the field
that misses the target, and the field that *hits* it no better than a centre
prior does, which is the failure the criteria actually turned on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--sink-k", type=float, default=3.0)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from q3vl.whereb.attnread import sink_mask_conjunctive
    from q3vl.whereb.attnprobe import combine_heads, head_norm_constants
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import center_prior_field
    from q3vl.whereb.viz import overlay_grid_on_image, render_field

    exp = Path(args.export)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(Path(args.metrics).read_text())
    pool = metrics["best_pool"]
    pinfo = metrics["pools"][pool]
    rows = {r["sample_id"]: r for r in pinfo["per_sample"]}
    idx = [(h["layer"], h["head"]) for h in pinfo["selected_heads"]]
    weights = [h["fit_median_soft_iou"] for h in pinfo["selected_heads"]]

    srows = [json.loads(l) for l in (exp / "samples.jsonl").read_text().splitlines()
             if l.strip()]
    meta = {r["sample_id"]: r for r in srows if r["arm"] == "gt"}
    fit_ids = [r["sample_id"] for r in srows
               if r["arm"] == "gt" and r["fold"] == "fit"
               and (exp / "fields" / f"{r['sample_id']}__shuffled.npz").exists()]

    # arm constants must be the SAME ones the metrics used: fit fold, this pool
    fit_arrays, fit_ncells = [], []
    for sid in fit_ids:
        d = np.load(exp / "fields" / f"{sid}__gt.npz")
        if f"field_{pool}" not in d:
            continue
        fit_arrays.append(d[f"field_{pool}"].astype(np.float64))
        fit_ncells.append(meta[sid]["grid_h"] * meta[sid]["grid_w"])
    mu, sigma = head_norm_constants(fit_arrays, fit_ncells)

    ds, _ = open_dataset(args.split, need_mask=False)
    by_id = {}
    for i in range(len(ds)):
        by_id[ds.refs[i].sample_id] = i

    ranked = sorted(rows.values(), key=lambda r: r["grid_soft_iou_gt"])
    picks = (
        [("failure_lowiou", r) for r in ranked[: args.n]]
        + [("success", r) for r in ranked[-args.n:][::-1]]
        # the failure this experiment actually turned on: the field scores fine
        # in absolute terms but loses to a zero-parameter centre prior
        + [("failure_losestoprior", r) for r in sorted(
            rows.values(),
            key=lambda r: r["grid_soft_iou_gt"] - r["grid_soft_iou_center_prior"]
        )[: args.n]]
    )

    for tag, r in picks:
        sid = r["sample_id"]
        m = meta[sid]
        gh, gw = m["grid_h"], m["grid_w"]
        dg = np.load(exp / "fields" / f"{sid}__gt.npz")
        dsh = np.load(exp / "fields" / f"{sid}__shuffled.npz")
        pg = dg["col_profile"].astype(np.float64).mean(axis=(0, 1))
        psh = dsh["col_profile"].astype(np.float64).mean(axis=(0, 1))
        sink = sink_mask_conjunctive(pg, psh, k=args.sink_k)
        valid = ~sink
        gt = np.load(exp / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)

        f_gt = combine_heads(dg[f"field_{pool}"].astype(np.float64), idx, weights,
                             mu, sigma, gh * gw)
        f_sh = combine_heads(dsh[f"field_{pool}"].astype(np.float64), idx, weights,
                             mu, sigma, gh * gw)
        prior = center_prior_field(gh, gw).numpy().astype(np.float64).reshape(-1)

        kk = max(1, min(int((gt[valid] > 0.5).sum()), int(valid.sum())))

        def topk(f):
            z = np.where(valid, f, -np.inf)
            mm = np.zeros(gh * gw)
            mm[np.argsort(-z)[:kk]] = 1.0
            return mm

        sample = ds[by_id[sid]]
        img = sample.image_tensor()
        oh, ow = img.shape[-2:]
        # the overlay demands an exact integer inverse map; crop to a multiple
        img = img[:, : (oh // gh) * gh, : (ow // gw) * gw]

        V = torch.from_numpy(valid.reshape(gh, gw))
        # N4: these two panels exist only to be compared with each other, so they
        # must share one scale.  Computed over VALID cells of BOTH fields.
        both = np.concatenate([f_gt[valid], f_sh[valid]])
        shared = (float(both.min()), float(both.max()))
        panels = [
            ("input", None, None, None),
            ("GT (soft, merged grid)", gt, None, None),
            (f"top-8 convex field [{pool}]", f_gt, V, shared),
            ("same field, shuffled instruction", f_sh, V, shared),
            (f"pred top-k (k={kk}) vs GT", topk(f_gt) + 2.0 * (gt > 0.5), None, None),
            ("centre prior top-k", topk(prior) + 2.0 * (gt > 0.5), None, None),
            ("sink mask (excluded)", sink.astype(float), None, None),
        ]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.4))
        for ax, (title, fld, vmask, scale) in zip(axes, panels):
            if fld is None:
                ax.imshow(img.numpy().transpose(1, 2, 0))
            else:
                t = torch.from_numpy(np.asarray(fld, dtype=np.float64).reshape(gh, gw))
                if vmask is None:
                    rr = render_field(t, allow_all_valid=True)
                    ax.imshow(rr.rgba)
                else:
                    blend, rr = overlay_grid_on_image(
                        t, img, alpha=0.6, valid=vmask,
                        mode="fixed" if scale else "valid_cells", fixed=scale)
                    ax.imshow(blend.clip(0, 1))
                tag_scale = "SHARED" if scale else "own"
                title = (f"{title}\n[{rr.vmin:.3g}, {rr.vmax:.3g}] ({tag_scale}) "
                         f"valid={rr.n_valid}")
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.suptitle(
            f"{tag}  {sid}  grid {gh}x{gw}\n"
            f"grid soft-IoU: field {r['grid_soft_iou_gt']:.3f} | "
            f"shuffled {r['grid_soft_iou_shuffled']:.3f} | "
            f"centre prior {r['grid_soft_iou_center_prior']:.3f} | "
            f"oracle ceiling {r['oracle_ceiling']:.3f}   "
            f"(area_frac={r.get('area_frac')}, centroid_dist={r.get('centroid_dist')})",
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(out / f"{tag}_{sid}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {tag}_{sid}.png")

    print(f"done -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
