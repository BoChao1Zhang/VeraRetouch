"""Zero-training intervention test of mechanism (b): is the guided upsampler
the analytic-family contamination source?

DX-1/DX-2 attribute (b) by *measurement*; this probe attributes it by
*intervention*, on the already-trained P3' checkpoint, with no training at all.
Three inference-time conditions over the same forward pass:

    A  current guided upsampling everywhere                (status quo)
    B  family-conditional gate: semantic keeps the guided
       operator, analytic families go to pure low-pass      (the proposed fix)
    C  guidance off for everybody                           (negative control)

C is what makes B interpretable.  If B improves the analytic families it could
still be that guidance is simply useless everywhere; C tests that directly, and
the registered expectation is that the **semantic** family degrades in C -- i.e.
guidance earns its place exactly where the target correlates with image edges.

Routing in B uses the **zero-training type-word rule**, not the GT family label,
because that is the deployable path (NOTES §3 measured it at 396/396 on
generated text).  Agreement with the GT label is reported rather than assumed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

ANALYTIC = ("radial", "linear", "band")


def _med(xs):
    v = [float(x) for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="/home/bc/data/runs/where_b/amort_P3prime_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    try:
        import resource

        s_, h_ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if s_ < h_:
            resource.setrlimit(resource.RLIMIT_NOFILE, (h_, h_))
    except Exception:
        pass

    import torch.nn.functional as F
    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.where.config import UpsampleConfig
    from q3vl.where.upsample import guided_upsample
    from q3vl.whereb.amort.data import AmortBatchBuilder, family_labels
    from q3vl.whereb.amort.model import AmortModel
    from q3vl.whereb.amort.simfield import SimFieldNorm, WordEmbedder
    from q3vl.whereb.context import ShuffleIndex
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.edgequal import calibrate, e_hf, kappa_tilde
    from q3vl.whereb.fields import load_basis
    from q3vl.whereb.hiddens import FrozenVLM
    from q3vl.whereb.metrics import gt_area_k, grid_boundary_f1, hard_iou, topk_mask
    from q3vl.whereb.stores import GenContextStore
    from q3vl.whereb.config import GENCTX_DIR

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ucfg = UpsampleConfig()

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    vlm_model = load_model(args.checkpoint, attn_implementation=args.attn,
                           dtype="bfloat16").to(args.device).eval()
    vlm = FrozenVLM(vlm_model, proc, device=args.device, want_merger=True)
    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)
    basis = load_basis("BA-3-Joint").to(args.device)
    embedder = WordEmbedder.from_checkpoint(args.checkpoint, proc.tokenizer)
    setup = json.loads((Path(args.run) / "config" / "run_setup.json").read_text())
    norm = SimFieldNorm.from_dict(setup["sim_norm"])

    ds, _ = open_dataset(args.split, need_mask=True)
    rows = ds.meta_rows()
    idx = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    if args.limit:
        idx = idx[: args.limit]
    fam = family_labels(ds, idx)
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
    sd = torch.load(Path(args.run) / "amort_final.pt", map_location=args.device)
    model.load_state_dict(sd["model"])
    model.eval()
    ctx = "generated" if genctx is not None else "gt"
    print(f"P3' step={sd.get('step')}  context={ctx}  n={len(idx)}", flush=True)

    per: list[dict[str, Any]] = []
    gt_pool = []
    with torch.no_grad():
        for n, i in enumerate(idx):
            s = ds[i]
            x = builder.build([s], [ctx])[0]
            gh, gw = x.grid_h, x.grid_w
            gt_hi = x.gt_hi.float().cpu()
            hi = tuple(gt_hi.shape)
            guide = x.guide_hi
            cond = model.cond_of(x.cond_h, x.cond_mask, x.word_ids, x.word_offsets)
            if x.route_semantic and model.sem is not None:
                o = model.forward_sem(x.feat, cond, sim=x.sim, center=x.center)
                to_m = model.sem.mask_of
            else:
                o = model.forward_geo(x.feat, cond, x.phi_dir, sim=x.sim,
                                      center=x.center, grid_h=gh, grid_w=gw)
                to_m = model.geo.mask_of
            s_low = o["s_low"].reshape(1, 1, gh, gw).float()

            s_guided = guided_upsample(s_low, guide, ucfg).reshape(hi)
            s_lowpass = F.interpolate(s_low, size=hi, mode="bilinear",
                                      align_corners=False).reshape(hi)
            routed_semantic = bool(x.route_semantic)
            fields = {
                "A_guided": s_guided,
                # the gate: semantic keeps guidance, analytic goes low-pass
                "B_gated": s_guided if routed_semantic else s_lowpass,
                "C_lowpass": s_lowpass,
            }
            k = gt_area_k(gt_hi)
            gtk = topk_mask(gt_hi, k)
            row = {"sample_id": x.sample_id, "family": x.family,
                   "routed_semantic": routed_semantic,
                   "gt_is_semantic": x.family == "semantic",
                   "winner_confidence": x.meta.get("winner_confidence"),
                   "analytic": x.family in ANALYTIC}
            for tag, sf in fields.items():
                m = to_m(sf).float().cpu()
                row[f"{tag}_iou"] = hard_iou(topk_mask(m, k), gtk)
                row[f"{tag}_bf1"] = grid_boundary_f1(topk_mask(m, k), gtk)
                row[f"_{tag}"] = m.numpy()
            row["_gt"] = gt_hi.numpy()
            per.append(row)
            gt_pool.append(gt_hi.numpy())
            if (n + 1) % 50 == 0:
                print(f"  [{n+1}/{len(idx)}]", flush=True)

    # arm-constant calibration on the analytic GT fields (never per image)
    cal = calibrate([r["_gt"] for r in per if r["analytic"]])
    print(f"calibration r_hi={cal.r_hi:.4f} tau={cal.tau:.5f}", flush=True)
    for r in per:
        gt = r["_gt"]
        kg = kappa_tilde(gt, cal)
        for tag in ("A_guided", "B_gated", "C_lowpass"):
            y = r.pop(f"_{tag}")
            r[f"{tag}_E_HF"] = e_hf(y, gt, cal)
            ky = kappa_tilde(y, cal)
            r[f"{tag}_kappa_ratio"] = (float("nan") if not np.isfinite(ky)
                                       or not np.isfinite(kg) or kg <= 0 else ky / kg)
        r.pop("_gt")

    def block(sel, name):
        rs = [r for r in per if sel(r)]
        b = {"n": len(rs)}
        for tag in ("A_guided", "B_gated", "C_lowpass"):
            b[tag] = {
                "iou": _med(r[f"{tag}_iou"] for r in rs),
                "boundary_f1": _med(r[f"{tag}_bf1"] for r in rs),
                "E_HF": _med(r[f"{tag}_E_HF"] for r in rs),
                "kappa_ratio": _med(r[f"{tag}_kappa_ratio"] for r in rs),
            }
        # paired deltas vs the status quo
        from q3vl.whereb.metrics import paired_delta

        for tag in ("B_gated", "C_lowpass"):
            for col in ("iou", "bf1", "kappa_ratio"):
                a = [r[f"{tag}_{col}"] for r in rs]
                c = [r[f"A_guided_{col}"] for r in rs]
                pair = [(u, v) for u, v in zip(a, c)
                        if u is not None and v is not None
                        and np.isfinite(u) and np.isfinite(v)]
                if pair:
                    b.setdefault("delta_vs_A", {})[f"{tag}_{col}"] = paired_delta(
                        [u for u, _ in pair], [v for _, v in pair])
        return name, b

    res: dict[str, Any] = {
        "n": len(per), "context": ctx, "checkpoint_step": sd.get("step"),
        "calibration": cal.to_dict(),
        "conditions": {
            "A_guided": "current guided upsampling everywhere (status quo)",
            "B_gated": "type-word routed: semantic guided, analytic low-pass",
            "C_lowpass": "guidance off everywhere (negative control)",
        },
        "routing_agreement_with_gt_family": float(
            np.mean([r["routed_semantic"] == r["gt_is_semantic"] for r in per])),
    }
    for sel, name in ((lambda r: r["analytic"], "analytic"),
                      (lambda r: not r["analytic"], "semantic"),
                      (lambda r: True, "all")):
        k, v = block(sel, name)
        res[k] = v
    for f_ in ("radial", "linear", "band", "semantic"):
        k, v = block(lambda r, f_=f_: r["family"] == f_, f_)
        res[f"family_{k}"] = v

    (out / "metrics.json").write_text(json.dumps(res, indent=2, default=str),
                                      encoding="utf-8")
    with (out / "per_sample.jsonl").open("w") as fh:
        for r in per:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: res[k] for k in ("analytic", "semantic",
                                          "routing_agreement_with_gt_family")},
                     indent=2, default=str)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
