"""P-W5 panels: target-noun similarity field vs its within-image control.

The two similarity panels share one colour scale (E1 review nit N4): they exist
only to be compared with each other, so two independent scales would show
nothing.  Overlay uses ``viz.grid_to_img``'s exact inverse map, never a resize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--gt-export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--n", type=int, default=4)
    args = ap.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoProcessor

    from q3vl.train.modeling import load_model
    from q3vl.whereb.attnread import merged_grid
    from q3vl.whereb.attnprobe import field_scores
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.metrics import center_prior_field
    from q3vl.whereb.viz import overlay_grid_on_image, render_field
    from q3vl.whereb.scripts.run_pw5_fpresim import subject_nouns

    d = json.loads(Path(args.metrics).read_text())
    rows = {r["sample_id"]: r for r in d["per_sample"]}
    gtx = Path(args.gt_export)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    model = load_model(args.checkpoint, attn_implementation="eager",
                       dtype="bfloat16").to("cuda:0").eval()
    E = model.get_input_embeddings().weight.detach()
    tok = proc.tokenizer

    def wemb(w):
        ids = tok(" " + w, add_special_tokens=False)["input_ids"]
        return E[torch.tensor(ids, device=E.device)].float().mean(dim=0)

    ds, _ = open_dataset("V_where", need_mask=False)
    idx = {ds.refs[i].sample_id: i for i in range(len(ds))}

    ranked = sorted(rows.values(), key=lambda r: r["dot_target_all"] - r["dot_cross_all"])
    picks = ([("failure", r) for r in ranked[: args.n]]
             + [("success", r) for r in ranked[-args.n:][::-1]])

    for tag, r in picks:
        sid = r["sample_id"]
        s = ds[idx[sid]]
        gh, gw = merged_grid(s.geometry.out_h, s.geometry.out_w)
        gt = np.load(gtx / "gt" / f"{sid}.npy").astype(np.float64).reshape(-1)
        enc = proc.image_processor(images=[s.image], do_resize=False, return_tensors="pt")
        with torch.no_grad():
            f, _ = model.model.get_image_features(
                enc["pixel_values"].to("cuda:0", torch.bfloat16),
                enc["image_grid_thw"].to("cuda:0"))
        f = (f[0] if isinstance(f, (list, tuple)) else f).reshape(-1, 2560).float()
        tgt = (f @ wemb(r["target"])).cpu().numpy().astype(np.float64)
        ctl = (f @ wemb(r["control_cross"])).cpu().numpy().astype(np.float64)
        prior = center_prior_field(gh, gw).numpy().astype(np.float64).reshape(-1)
        v = np.ones(gh * gw, dtype=bool)
        kk = max(1, int((gt > 0.5).sum()))

        def tk(x):
            m = np.zeros(gh * gw); m[np.argsort(-x)[:kk]] = 1.0; return m

        img = s.image_tensor()
        oh, ow = img.shape[-2:]
        img = img[:, : (oh // gh) * gh, : (ow // gw) * gw]
        both = np.concatenate([tgt, ctl])
        shared = (float(both.min()), float(both.max()))
        V = torch.from_numpy(v.reshape(gh, gw))

        panels = [("input", None, None, None),
                  ("GT (soft, merged grid)", gt, None, None),
                  (f"sim: TARGET '{r['target']}'", tgt, V, shared),
                  (f"sim: control '{r['control_cross']}'", ctl, V, shared),
                  (f"target top-k (k={kk}) vs GT", tk(tgt) + 2.0 * (gt > 0.5), None, None),
                  ("centre prior top-k", tk(prior) + 2.0 * (gt > 0.5), None, None)]
        fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.4))
        for ax, (title, fld, vm, sc) in zip(axes, panels):
            if fld is None:
                ax.imshow(img.numpy().transpose(1, 2, 0))
            else:
                t = torch.from_numpy(np.asarray(fld, float).reshape(gh, gw))
                if vm is None:
                    rr = render_field(t, allow_all_valid=True); ax.imshow(rr.rgba)
                else:
                    b, rr = overlay_grid_on_image(t, img, alpha=0.6, valid=vm,
                                                  mode="fixed", fixed=sc)
                    ax.imshow(b.clip(0, 1))
                title = f"{title}\n[{rr.vmin:.3g}, {rr.vmax:.3g}]{' SHARED' if sc else ''}"
            ax.set_title(title, fontsize=8); ax.axis("off")
        fig.suptitle(
            f"{tag}  {sid}  grid {gh}x{gw}  nouns={r['nouns']}\n"
            f"soft-IoU: target {r['dot_target_all']:.3f} | control '{r['control_cross']}' "
            f"{r['dot_cross_all']:.3f} | centre prior {r['center_prior_all']:.3f} | "
            f"random floor {r['random_floor']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / f"{tag}_{sid}.png", dpi=95, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {tag}_{sid}.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
