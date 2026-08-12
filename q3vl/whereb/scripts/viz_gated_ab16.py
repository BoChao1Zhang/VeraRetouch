"""Before/after panels for the adopted gated-upsampling fix.

Deliberately reuses the **exact 16 sample ids** drawn for `viz_random16`
(read from its `manifest.json`, not re-drawn), so this figure is row-for-row
comparable with the one already reviewed.  Re-drawing with the same seed would
have been fragile -- any change to the dataset ordering would silently produce a
different 16 while still looking reproducible.

One row per sample:

    input | GT | A: guided (before) | B: family-gated (after) | A - B

The difference panel uses a **symmetric, zero-centred** diverging scale so that
"no change" is unambiguously the midpoint colour; the mask panels stay on the
fixed 0..1 scale the project requires.  Every cell is annotated with that
sample's own kappa~ and top-k IoU.

Semantic rows are expected to be **identical** in A and B -- the router sends
them down the guided path in both -- and they are shown rather than dropped,
because "the fix does nothing here" is part of what the figure has to establish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

ANALYTIC = ("radial", "linear", "band")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="/home/bc/VeraRetouch/experiments/"
                    "Q3VL_metacanvas_where_what_20260804/where_b/"
                    "amort_p3prime_20260810/viz_random16/manifest.json")
    ap.add_argument("--probe-metrics", default="/home/bc/VeraRetouch/experiments/"
                    "Q3VL_metacanvas_where_what_20260804/where_b/"
                    "probe_gated_upsample_20260811/metrics.json")
    ap.add_argument("--run", default="/home/bc/data/runs/where_b/amort_P3prime_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import resource

        s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if s_ < h_:
            resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    except Exception:
        pass

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.where.config import UpsampleConfig
    from q3vl.where.upsample import area_resize, guided_upsample
    from q3vl.whereb.amort.data import AmortBatchBuilder, family_labels
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.edgequal import (EdgeQualCalibration, band_occupancy,
                                      calibrate_per_family, kappa_tilde)
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.viz import render_field

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ucfg = UpsampleConfig()
    FIXED = (0.0, 1.0)

    man = json.loads(Path(args.manifest).read_text())
    want = list(man["drawn_sample_ids"])
    cal_global = EdgeQualCalibration.from_dict(
        json.loads(Path(args.probe_metrics).read_text())["calibration"])
    print(f"reusing {len(want)} ids from viz_random16 (seed {man['seed']})", flush=True)

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype="bfloat16").to(args.device).eval()
    vlm = FrozenVLM(vlm_model, proc, device=args.device, want_merger=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    basis = load_basis("BA-3-Joint").to(args.device)
    embedder = WordEmbedder.from_checkpoint(args.checkpoint, proc.tokenizer)
    norm = SimFieldNorm.from_dict(json.loads(
        (Path(args.run) / "config" / "run_setup.json").read_text())["sim_norm"])

    ds, _ = open_dataset(args.split, need_mask=True)
    by_id = {ds.record(i)["sample_id"]: i for i in range(len(ds))}
    idx = [by_id[s] for s in want if s in by_id]
    fam = family_labels(ds, idx)

    # per-family tau, calibrated over the whole split's GT (arm constants, one
    # per family) -- a single pooled tau leaves every linear ramp with an empty
    # band and an undefined kappa~
    all_idx = [i for i, r in enumerate(ds.meta_rows())
               if r.get("render_mode") == "local"]
    fam_all = family_labels(ds, all_idx)
    pool: dict[str, list] = {}
    for i in all_idx:
        f_ = fam_all.get(ds.record(i)["sample_id"])
        if f_:
            pool.setdefault(f_, []).append(
                ds[i].mask_target_hi().float().numpy())
    cals = calibrate_per_family(pool)
    for f_, c in sorted(cals.items()):
        print(f"  tau[{f_}] = {c.tau:.6f}  (global was {cal_global.tau:.6f})",
              flush=True)
    try:
        genctx = GenContextStore(Path(GENCTX_DIR) / args.split)
    except Exception:
        genctx = None
    builder = AmortBatchBuilder(
        collator, vlm, basis, embedder=embedder, norm=norm,
        shuffle_index=ShuffleIndex(ds.shuffle_records(), seed=0), genctx=genctx,
        families=fam, id_to_index={ds.record(i)["sample_id"]: i for i in idx},
        dataset=ds, device=args.device, want_hi=True,
        attn_implementation=args.attn, checkpoint=args.checkpoint)

    model = AmortModel("P3prime").to(args.device)
    model.load_state_dict(torch.load(Path(args.run) / "amort_final.pt",
                                     map_location=args.device)["model"])
    model.eval()
    ctx = "generated" if genctx is not None else "gt"

    rows = []
    fig, axes = plt.subplots(len(idx), 5, figsize=(3.2 * 5, 2.05 * len(idx)))
    if len(idx) == 1:
        axes = axes[None, :]

    with torch.no_grad():
        for r, i in enumerate(idx):
            s = ds[i]
            x = builder.build([s], [ctx])[0]
            gh, gw = x.grid_h, x.grid_w
            gt = x.gt_hi.float().cpu()
            hi = tuple(gt.shape)
            cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
            if x.route_semantic and model.sem is not None:
                o = model.forward_sem(x.feat, cond, sim=x.sim, center=x.center)
                to_m = model.sem.mask_of
            else:
                o = model.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                      center=x.center, grid_h=gh, grid_w=gw)
                to_m = model.geo.mask_of
            s_low = o["s_low"].reshape(1, 1, gh, gw).float()
            s_guided = guided_upsample(s_low, x.guide_hi, ucfg).reshape(hi)
            s_lowpass = F.interpolate(s_low, size=hi, mode="bilinear",
                                      align_corners=False).reshape(hi)
            mA = to_m(s_guided).float().cpu()
            mB = to_m(s_guided if x.route_semantic else s_lowpass).float().cpu()

            k = gt_area_k(gt)
            gtk = topk_mask(gt, k)
            cal = cals.get(x.family, cal_global)
            gtn = gt.numpy()
            kg = kappa_tilde(gtn, cal, band_from=gtn)
            occ = band_occupancy(gtn, cal)
            # A GT whose own band curvature is ~0 makes the ratio meaningless
            # (observed 2.1e5 on one band sample); report NaN rather than a
            # number that would dominate any mean.
            KG_MIN = 1e-3

            def kr(m):
                # band taken from the GT so A and B are scored on identical pixels
                kv = kappa_tilde(m.numpy(), cal, band_from=gtn)
                return (float("nan") if not np.isfinite(kv) or kg <= KG_MIN
                        else kv / kg)
            iouA = hard_iou(topk_mask(mA, k), gtk)
            iouB = hard_iou(topk_mask(mB, k), gtk)
            krA, krB = kr(mA), kr(mB)
            diff = (mA - mB).numpy()
            vmax = float(np.abs(diff).max()) or 1e-6

            valid = torch.ones(*hi, dtype=torch.bool)
            panels = [
                (s.image_tensor().permute(1, 2, 0).numpy(),
                 f"{s.sample_id[:20]}\n{x.family}"
                 + ("  [semantic route]" if x.route_semantic else "  [analytic route]")),
                (render_field(gt, valid, mode="fixed", fixed=FIXED).rgba,
                 f"GT   kappa~={kg:.3f}  band={occ:.0%}"),
                (render_field(mA, valid, mode="fixed", fixed=FIXED).rgba,
                 f"A guided (before)\nkappa~ratio={krA:.2f}  IoU={iouA:.3f}"),
                (render_field(mB, valid, mode="fixed", fixed=FIXED).rgba,
                 f"B gated (after)\nkappa~ratio={krB:.2f}  IoU={iouB:.3f}"),
                (None, f"A - B   max|d|={vmax:.3f}"
                 + ("\n(identical by routing)" if x.route_semantic else "")),
            ]
            for c, (arr, title) in enumerate(panels):
                ax = axes[r, c]
                if c == 4:
                    # symmetric, zero-centred: "no change" is unambiguously mid-tone
                    ax.imshow(diff, cmap="coolwarm", vmin=-vmax, vmax=vmax)
                else:
                    ax.imshow(arr)
                ax.set_title(title, fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            rows.append({"sample_id": s.sample_id, "family": x.family,
                         "routed_semantic": bool(x.route_semantic),
                         "kappa_gt": kg, "band_occupancy": occ,
                         "tau_family": cal.tau,
                         "kappa_ratio_A": krA, "kappa_ratio_B": krB,
                         "iou_A": iouA, "iou_B": iouB,
                         "max_abs_diff": vmax})
            print(f"  [{r+1}/{len(idx)}] {s.sample_id[:18]} {x.family:9s} "
                  f"kappa {krA:.2f}->{krB:.2f}  IoU {iouA:.3f}->{iouB:.3f}", flush=True)

    fig.suptitle(
        "Gated upsampling, before (A) vs after (B) -- same 16 samples as viz_random16 "
        f"(seed {man['seed']}).  Analytic families route to low-pass; semantic keeps "
        "guidance, so those rows are identical by construction.   "
        "[mask panels fixed 0..1; difference panel symmetric about 0]", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    png = out / "gated_ab16.png"
    fig.savefig(png, dpi=100, bbox_inches="tight")
    plt.close(fig)
    (out / "manifest.json").write_text(json.dumps(
        {"source_manifest": str(args.manifest), "seed": man["seed"],
         "n": len(rows), "calibration": cal.to_dict(), "rows": rows}, indent=2),
        encoding="utf-8")
    ana = [r for r in rows if not r["routed_semantic"]]
    ka = np.array([r["kappa_ratio_A"] for r in ana], dtype=float)
    kb = np.array([r["kappa_ratio_B"] for r in ana], dtype=float)
    ok = np.isfinite(ka) & np.isfinite(kb)
    print(json.dumps({
        "png": str(png), "n_analytic": len(ana),
        "n_kappa_defined": int(ok.sum()),
        "median_kappa_A": float(np.median(ka[ok])) if ok.any() else None,
        "median_kappa_B": float(np.median(kb[ok])) if ok.any() else None,
        "median_iou_A": float(np.median([r["iou_A"] for r in ana])),
        "median_iou_B": float(np.median([r["iou_B"] for r in ana])),
        "semantic_rows_identical": all(
            abs(r["iou_A"] - r["iou_B"]) < 1e-9
            for r in rows if r["routed_semantic"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
