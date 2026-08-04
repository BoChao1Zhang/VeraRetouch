#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  Oracle latents for the Where-B *train* split.

Why this exists
---------------
Where-A's ``run_calibration.py`` publishes oracle latents for ``V_where`` only
(``ORACLE_DIR/<arm>/V_where``), because that is the split its ceiling is measured
on.  Protocol 5.5 gives Where-B three oracle auxiliaries -- ``L_s``, ``L_curve``,
``L_dir`` -- which need ``w*, rho*`` on **every training sample**.  That gap is
closed here rather than by editing ``q3vl/where/`` (the Where-A task owns those
files): this job re-uses Where-A's own fitter and packer with the frozen
``BA-3-Joint`` basis, so the latents are produced by exactly the code that
produced the V_where ones.

    B        <- frozen BA-3-Joint basis (digest-checked)
    phi      <- B(F_pre) residualised + standardised, with geo5 / L / S
    w*, rho* <- multi-start L-BFGS in float64 (n_random=6, max_iter=120)
    publish  -> ORACLE_DIR/<arm>/<split>/  (indexed tar shards, atomic)

Local samples only: a global sample's mask is all ones, has no basis direction
to recover, and its auxiliaries are masked off in the loss.

Requires one GPU for the vision forward.  **Do not start while Base SFT holds
both cards, and run it after Where-A has published its basis.**
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from q3vl.where.config import FitConfig, ORACLE_DIR
from q3vl.where.oracle import fit_latent
from q3vl.where.packing import pack_oracle, verify_published
from q3vl.where.upsample import area_resize
from q3vl.whereb.config import BASIS_ARM, MODEL_DIR, READOUTS, REPORT_DIR, SFT_CHECKPOINT
from q3vl.whereb.data import open_dataset
from q3vl.whereb.fields import load_basis, phi_dir_fast
from q3vl.whereb.preflight import _env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--arm", default=BASIS_ARM)
    ap.add_argument("--checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--readouts", nargs="+", default=list(READOUTS))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-random", type=int, default=6)
    ap.add_argument("--max-iter", type=int, default=120)
    ap.add_argument("--basis-root", default=None,
                    help="override the Where-A basis root (smoke runs only)")
    ap.add_argument("--out-root", default=None,
                    help="override the oracle publication root (smoke runs only)")
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; sample i uses seed+i, matching "
                         "q3vl.where.calibrate.fit_sample's seed_offset=j")
    args = ap.parse_args()

    from transformers import AutoProcessor

    from q3vl.where.fpre import FPreHook, load_vision_tower

    out_root = (Path(args.out_root) if args.out_root
                else ORACLE_DIR / args.arm / args.split)
    if out_root.exists():
        raise SystemExit(f"{out_root} exists; move it aside before re-running")

    basis = load_basis(args.arm, Path(args.basis_root) if args.basis_root else None)
    visual = load_vision_tower(
        Path(args.checkpoint) if Path(args.checkpoint).exists() else Path(args.model_dir),
        dtype=getattr(torch, args.dtype), device=args.device,
    )
    processor = AutoProcessor.from_pretrained(args.model_dir)
    basis = basis.to(args.device)

    # review blocker B2: the old call passed neither a maskview store nor a
    # resolver and raised on its first sample.  ``open_dataset`` prefers
    # Where-A's published mask views and falls back to the live resolver.
    dataset, ds_info = open_dataset(args.split, include_global=False,
                                    limit=args.limit)
    fit_cfg = FitConfig(n_random=args.n_random, max_iter=args.max_iter)
    setup = {
        "split": args.split, "arm": args.arm, "env": _env(),
        "basis": basis.facts(), "n_samples": len(dataset), "dataset": ds_info,
        "fit": fit_cfg.__dict__, "readouts": args.readouts, "seed": args.seed,
        "out_root": str(out_root),
    }
    print(json.dumps(setup, indent=2), flush=True)

    rows, report = [], {"n_ok": {r: 0 for r in args.readouts}, "n_rejected": {}}
    t0 = time.time()
    for i in range(len(dataset)):
        s = dataset[i]
        img = s.image_tensor()
        inputs = processor.image_processor(images=[s.image], do_resize=False,
                                           return_tensors="pt")
        hook = FPreHook(visual)
        with torch.no_grad(), hook.attached():
            visual(inputs["pixel_values"].to(args.device, next(visual.parameters()).dtype),
                   grid_thw=inputs["image_grid_thw"].to(args.device))
        fpre = hook.split(inputs["image_grid_thw"])[0].float().reshape(
            s.grid_h * s.grid_w, -1)
        img_low = area_resize(img.unsqueeze(0), (s.grid_h, s.grid_w))[0]
        with torch.no_grad():
            phi = phi_dir_fast(basis(fpre), img_low.to(args.device), s.grid_h, s.grid_w)
        # nit N4: prefer Where-A's published float `.masklow.npy` over
        # re-deriving it from the uint8 `.maskhi.png` (which would quantise to
        # 1/255 first and then downsample, giving a slightly different target
        # than the V_where latents were fitted against).
        mv = dataset.maskviews
        if mv is not None and mv.has(s.sample_id, mv.LOW):
            mask_low = mv.mask_low(s.sample_id).reshape(-1).to(args.device)
            mask_source = "published_masklow"
        else:
            mask_low = area_resize(
                s.mask_target_hi()[None, None], (s.grid_h, s.grid_w)
            ).reshape(-1).to(args.device)
            mask_source = "derived_from_mask_hi"

        fits = {}
        for r in args.readouts:
            # nit N3: Where-A's calibrate.fit_sample varies the multi-start seed
            # per sample (seed_offset=j); match it so this job's latents are
            # produced the same way the V_where ones were.
            cfg_i = FitConfig(**{**fit_cfg.__dict__, "seed": args.seed + i})
            fit = fit_latent(phi.double(), mask_low.double(), r, cfg_i)
            fits[r] = fit.to_dict()
            fits[r]["mask_source"] = mask_source
            if fit.status == "ok":
                report["n_ok"][r] += 1
            else:
                key = f"{r}:{fit.reject_reason}"
                report["n_rejected"][key] = report["n_rejected"].get(key, 0) + 1
        rows.append((s.sample_id, fits, dict(s.meta)))
        if (i + 1) % 200 == 0:
            print(json.dumps({"done": i + 1, "of": len(dataset),
                              "elapsed_s": round(time.time() - t0, 1)}), flush=True)

    manifest = pack_oracle(iter(rows), out_root,
                           source_label=f"where_a.oracle/{args.arm}/{args.split}")
    report.update({
        "setup": setup, "elapsed_s": round(time.time() - t0, 1),
        "manifest": {k: manifest.get(k) for k in
                     ("status", "member_count", "sample_count", "shard_count")},
        "verify": verify_published(out_root, n_random=64),
        "ok_rate": {r: report["n_ok"][r] / max(1, len(rows)) for r in args.readouts},
    })
    rd = Path(args.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"oracle_{args.arm}_{args.split}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "setup"}, indent=2),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
