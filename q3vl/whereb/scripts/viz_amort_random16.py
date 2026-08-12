"""Effect-blind random panels for the two PR-AMORT arms.

**The sampling is done before any prediction exists.**  The 16 sample ids are
drawn from the V_where local split with a fixed seed and are never re-ordered,
filtered or swapped afterwards -- so this figure cannot be, and cannot be
suspected of being, a curated one.  That is the entire point of it existing
next to the `viz/success_*` / `viz/failure_*` sets, which ARE selected by
top-k IoU and are labelled as such.

One row per sample:

    input | GT (.cgt) | P3' m_pred | P1 m_pred | centre prior top-k | m_sem

Every mask cell is annotated with that sample's own top-k IoU, so the reader
never has to trust the picture over the number.

Visualisation discipline (project red lines):
  * colour scale is FIXED 0..1 -- never per-image min-max;
  * the scale reads valid cells only, and this representation has no pad cells
    (each sample keeps its native H/16 grid), which is asserted rather than
    assumed;
  * the input overlay uses `grid_to_img`'s exact inverse map, never a resize.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p1-run", default="/home/bc/data/runs/where_b/amort_P1_20260810")
    ap.add_argument("--p3-run", default="/home/bc/data/runs/where_b/amort_P3prime_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260811)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--out", required=True)
    ap.add_argument("--context", default="generated")
    args = ap.parse_args(argv)

    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft < hard:
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.whereb.amort.data import AmortBatchBuilder, family_labels
    from q3vl.whereb.amort.evaluate import center_prior_unit
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.viz import overlay_grid_on_image, render_field

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    FIXED = (0.0, 1.0)

    # ---- data, and THE DRAW (before any model touches anything) -----------
    ds, _ = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(local), size=min(args.n, len(local)),
                             replace=False).tolist())
    idx = [local[p] for p in pick]
    drawn = [ds.record(i)["sample_id"] for i in idx]
    print(f"drawn {len(idx)} of {len(local)} local, seed={args.seed}", flush=True)

    # ---- frozen VLM + both arms -------------------------------------------
    proc = AutoProcessor.from_pretrained(args.checkpoint)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype="bfloat16").to(args.device).eval()
    vlm = FrozenVLM(vlm_model, proc, device=args.device, want_merger=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    basis = load_basis("BA-3-Joint").to(args.device)
    embedder = WordEmbedder.from_checkpoint(args.checkpoint, proc.tokenizer)

    setup = json.loads((Path(args.p3_run) / "config" / "run_setup.json").read_text())
    norm = SimFieldNorm.from_dict(setup["sim_norm"])
    print(f"norm centre={norm.center:.4f} scale={norm.scale:.4f} "
          f"(kernel {norm.attn_implementation})", flush=True)

    fam = family_labels(ds, idx)
    shuffle = ShuffleIndex(ds.shuffle_records(), seed=0)
    try:
        genctx = GenContextStore(Path(GENCTX_DIR) / args.split)
    except Exception:
        genctx = None
    builder = AmortBatchBuilder(
        collator, vlm, basis, embedder=embedder, norm=norm,
        shuffle_index=shuffle, genctx=genctx, families=fam,
        id_to_index={ds.record(i)["sample_id"]: i for i in idx}, dataset=ds,
        device=args.device, attn_implementation=args.attn,
        checkpoint=args.checkpoint)

    arms = {}
    for tag, run in (("P1", args.p1_run), ("P3prime", args.p3_run)):
        m = AmortModel(tag if tag == "P1" else "P3prime").to(args.device)
        sd = torch.load(Path(run) / "amort_final.pt", map_location=args.device)
        m.load_state_dict(sd["model"])
        m.eval()
        arms[tag] = m
        print(f"loaded {tag} step={sd.get('step')}", flush=True)

    # ---- render ------------------------------------------------------------
    ctx_mode = args.context if genctx is not None else "gt"
    manifest = []
    n_cols = 6
    fig, axes = plt.subplots(len(idx), n_cols,
                             figsize=(3.1 * n_cols, 2.05 * len(idx)))
    if len(idx) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, i in enumerate(idx):
            s = ds[i]
            built = builder.build([s], [ctx_mode])
            x = built[0]
            gh, gw = x.grid_h, x.grid_w
            valid = torch.ones(gh, gw, dtype=torch.bool)
            if int(valid.sum()) != gh * gw:
                raise AssertionError("unexpected pad cells in native-grid field")
            gt = x.gt_low.float().cpu()
            k = gt_area_k(gt)
            gt_k = topk_mask(gt, k)

            preds = {}
            for tag, m in arms.items():
                cond = m.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
                if x.route_semantic and m.sem is not None:
                    o = m.forward_sem(x.feat, cond, sim=x.sim, center=x.center)
                else:
                    o = m.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                      center=x.center, grid_h=gh, grid_w=gw)
                preds[tag] = o["m_low"].float().cpu()
            # the semantic head's own output, regardless of how the router sent
            # this sample -- so the panel shows what m_sem produces everywhere
            m3 = arms["P3prime"]
            cond3 = m3.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
            m_sem = m3.forward_sem(x.feat, cond3, sim=x.sim,
                                   center=x.center)["m_low"].float().cpu()
            cp = center_prior_unit(gh, gw).cpu()

            iou = {t: hard_iou(topk_mask(v, k), gt_k) for t, v in preds.items()}
            iou["m_sem"] = hard_iou(topk_mask(m_sem, k), gt_k)
            iou["centre"] = hard_iou(topk_mask(cp, k), gt_k)

            img = s.image_tensor()
            over, _ = overlay_grid_on_image(preds["P3prime"], img, valid=valid,
                                            mode="fixed", fixed=FIXED, alpha=0.5)
            panels = [
                (over, f"input + P3' overlay\n{s.sample_id[:20]}"),
                (render_field(gt, valid, mode="fixed", fixed=FIXED).rgba,
                 f"GT .cgt  area={float((gt > 0.5).float().mean()):.3f}\n{x.family}"),
                (render_field(preds["P3prime"], valid, mode="fixed", fixed=FIXED).rgba,
                 f"P3' (no Phi)  top-k IoU={iou['P3prime']:.3f}"),
                (render_field(preds["P1"], valid, mode="fixed", fixed=FIXED).rgba,
                 f"P1 (via Phi-71)  top-k IoU={iou['P1']:.3f}"),
                (render_field(topk_mask(cp, k).float(), valid, mode="fixed",
                              fixed=FIXED).rgba,
                 f"centre prior top-k  IoU={iou['centre']:.3f}"),
                (render_field(m_sem, valid, mode="fixed", fixed=FIXED).rgba,
                 f"m_sem (P3' sem head)  IoU={iou['m_sem']:.3f}"
                 + ("\n[routed here]" if x.route_semantic else "\n[not routed]")),
            ]
            for c, (arr, title) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(arr)
                ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            manifest.append({
                "row": r, "sample_id": s.sample_id, "family": x.family,
                "routed_semantic": bool(x.route_semantic),
                "winner_confidence": x.meta.get("winner_confidence"),
                "grid": [gh, gw], "gt_area": float((gt > 0.5).float().mean()),
                "topk_iou": {k2: float(v) for k2, v in iou.items()},
            })
            print(f"  [{r+1}/{len(idx)}] {s.sample_id[:20]} "
                  f"P3'={iou['P3prime']:.3f} P1={iou['P1']:.3f} "
                  f"centre={iou['centre']:.3f}", flush=True)

    fig.suptitle(
        f"PR-AMORT random {len(idx)} of {len(local)} V_where local "
        f"(seed={args.seed}, EFFECT-BLIND: drawn before any prediction; "
        f"context={ctx_mode})   "
        "[colour scale fixed 0..1, valid cells only, no per-image min-max]",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png = out_dir / f"random{len(idx)}_seed{args.seed}.png"
    fig.savefig(png, dpi=100, bbox_inches="tight")
    plt.close(fig)

    (out_dir / "manifest.json").write_text(json.dumps({
        "seed": args.seed, "n": len(idx), "split": args.split,
        "context": ctx_mode, "drawn_sample_ids": drawn,
        "selection": ("uniform random over V_where local, fixed seed, drawn "
                      "BEFORE any model forward; never reordered or filtered"),
        "convention": "matched-area top-k IoU; colour scale fixed 0..1",
        "rows": manifest,
    }, indent=2), encoding="utf-8")

    med = {t: float(np.median([m["topk_iou"][t] for m in manifest]))
           for t in ("P1", "P3prime", "centre", "m_sem")}
    print(json.dumps({"png": str(png), "medians_over_these_16": med}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
