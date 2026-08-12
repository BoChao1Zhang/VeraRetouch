"""One cheap vision-tower pass -> the cache E1/E5 need, after which both are pure CPU.

The task card calls E1 "pure CPU" and REQUIREMENTS section 2.2 says the Where-A
``F_pre`` cache is ready to use.  Neither holds: ``where_a-20260805/`` publishes
only ``basis/``, ``maskviews/`` and ``oracle/``, and there is no ``F_pre`` cache
anywhere on NFS or locally.  So E1's design matrix cannot be rebuilt without
re-running the vision tower.

It is cheap -- the tower is frozen and SFT-invariant (WA-P4b measured diff=0), and
400 images took 24 s in P-W5 -- so this dumps everything both cards need in ONE
pass and writes it locally.  After this, E1 and E5 really are CPU-only.

Per sample it stores:

* ``semantic_low`` ``(P, 64)`` = ``B(F_pre)`` on the H/16 grid -- the input
  ``phi.build_phi_dir`` wants (P = grid_h*grid_w, row-major);
* ``img_low`` ``(3, grid_h, grid_w)`` -- the spec-5 image area-averaged onto the
  same grid, ``build_phi_dir``'s second input;
* ``merger_out`` ``(n_img, 2560)`` on the H/32 grid -- what the LLM receives, the
  basis P-W5 scored in and the one plan C is specified to score in.

``F_pre`` itself (P, 1024) is deliberately NOT stored: only ``B @ F_pre`` enters
phi, and the full tensor is 16x larger for no consumer.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--basis", default="/mnt/nfs-ro/bc/data/datasets/where_a-20260805/basis/BA-3-Joint/B.npy")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args(argv)

    t0 = time.time()
    from transformers import AutoProcessor

    from q3vl.train.modeling import load_model
    from q3vl.where.fpre import FPreHook, grid_from_geometry
    from q3vl.where.upsample import area_resize
    from q3vl.whereb.attnread import merged_grid
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.hiddens import resolve_visual

    out = Path(args.out)
    (out / "cache").mkdir(parents=True, exist_ok=True)

    B = torch.from_numpy(np.load(args.basis)).float()          # (64, 1024)
    print(f"basis B {tuple(B.shape)}", flush=True)

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    model = load_model(args.checkpoint, attn_implementation="eager",
                       dtype="bfloat16").to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    visual = resolve_visual(model)

    ds, _ = open_dataset(args.split, need_mask=False)
    rows = ds.meta_rows()
    local = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    if args.limit:
        local = local[: args.limit]
    print(f"{args.split} local: {len(local)}", flush=True)

    manifest = []
    for n, i in enumerate(local):
        s = ds[i]
        sid = s.sample_id
        gh16, gw16 = grid_from_geometry(s.geometry.out_h, s.geometry.out_w)
        gh32, gw32 = merged_grid(s.geometry.out_h, s.geometry.out_w)

        enc = proc.image_processor(images=[s.image], do_resize=False, return_tensors="pt")
        grid_thw = enc["image_grid_thw"]
        hook = FPreHook(visual)
        with torch.no_grad(), hook.attached():
            feats, _ = model.model.get_image_features(
                enc["pixel_values"].to(args.device, torch.bfloat16),
                grid_thw.to(args.device))
        fpre = hook.split(grid_thw)[0].float()                  # (gh16, gw16, 1024)
        if tuple(fpre.shape[:2]) != (gh16, gw16):
            raise RuntimeError(f"{sid}: F_pre grid {tuple(fpre.shape[:2])} != {(gh16, gw16)}")
        sem = (fpre.reshape(-1, fpre.shape[-1]).cpu() @ B.T)     # (P, 64)

        m = feats[0] if isinstance(feats, (list, tuple)) else feats
        m = m.reshape(-1, m.shape[-1]).float().cpu()             # (n_img, 2560)
        if m.shape[0] != gh32 * gw32:
            raise RuntimeError(f"{sid}: merger {m.shape[0]} != {gh32*gw32}")

        img = s.image_tensor()
        img_low = area_resize(img.unsqueeze(0), (gh16, gw16))[0]  # (3, gh16, gw16)

        np.savez(out / "cache" / f"{sid}.npz",
                 semantic_low=sem.numpy().astype(np.float32),
                 img_low=img_low.numpy().astype(np.float32),
                 merger_out=m.numpy().astype(np.float16))
        manifest.append({"sample_id": sid, "grid16": [gh16, gw16], "grid32": [gh32, gw32],
                         "out_h": s.geometry.out_h, "out_w": s.geometry.out_w,
                         "render_mode": s.meta.get("render_mode"),
                         "build": s.meta.get("build"),
                         "winner_confidence": s.meta.get("winner_confidence")})
        if (n + 1) % 100 == 0:
            print(f"  [{n+1}/{len(local)}] {time.time()-t0:.0f}s", flush=True)

    (out / "manifest.json").write_text(json.dumps({
        "split": args.split, "n": len(manifest), "basis": args.basis,
        "checkpoint": args.checkpoint,
        "note": ("semantic_low = B @ F_pre on the H/16 grid; merger_out on H/32. "
                 "F_pre is SFT-invariant (WA-P4b diff=0), so this cache is reusable."),
        "samples": manifest,
    }, indent=2), encoding="utf-8")
    print(f"done {len(manifest)} samples in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
