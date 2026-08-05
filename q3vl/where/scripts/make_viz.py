#!/usr/bin/env python
"""Deliverable visualisations: input / predicted mask / GT / s field, side by side.

Visualisation discipline (result-review requirement):

* the colour scale for ``s`` is taken from the **valid cells only** -- the cells
  the readout actually responds to -- not from the whole field, so a handful of
  saturated cells cannot flatten everything else into one colour;
* every number printed on a panel is read from the **raw field**, never from the
  rescaled image the panel displays;
* failure panels are mandatory, and the two kinds the result review isolated are
  labelled by kind, not lumped into "hard sample".

Reads the published S5 oracle latents (so the mask shown is exactly the one that
was delivered) and recomputes only ``phi`` from the frozen VLM.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from q3vl.where.basis import Latent, mask_from_latent
from q3vl.where.config import (
    MODEL_DIR, REPORT_DIR, S_DOMAIN, CalibConfig, PhiConfig, UpsampleConfig,
)
from q3vl.where.fpre import load_vision_tower
from q3vl.where.readout import apply_readout
from q3vl.where.upsample import combine_then_upsample


def _load_latents(root: Path) -> dict:
    """sample_id -> {readout: fit dict} from a published oracle shard set."""
    out = {}
    for idx in sorted((root / "indexes").glob("shard-*.idx.jsonl")):
        with idx.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                fd = os.open(str(root / "shards" / f"{row['shard']}.tar"), os.O_RDONLY)
                data = os.pread(fd, row["length"], row["offset_data"])
                os.close(fd)
                rec = json.loads(data)
                out[rec["sample_id"]] = rec["fits"]
    return out


def _valid_scale(s: np.ndarray, m: np.ndarray, thresh: float = 0.02):
    """Colour range from the cells the mask actually responds to.

    Using the full field lets a few saturated cells set the range and wash out
    everything the reader is supposed to see.
    """
    live = s[m > thresh]
    if live.size < 8:
        live = s
    lo, hi = float(np.percentile(live, 2)), float(np.percentile(live, 98))
    if hi - lo < 1e-6:
        lo, hi = float(s.min()), float(s.max())
    return lo, hi


def _panel(ax, img, title, cmap=None, vmin=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="BA-3-Joint")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--oracle-root", default=None)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--maskview-root", default="/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews")
    ap.add_argument("--basis", default="/mnt/nfs/bc/data/datasets/where_a-20260805/basis/BA-3-Joint/B.npy")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--readout", default="cband12")
    ap.add_argument("--n-success", type=int, default=3)
    ap.add_argument("--failures", nargs="*", default=[])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    root = Path(args.oracle_root) if args.oracle_root else Path(
        f"/mnt/nfs/bc/data/datasets/where_a-20260805/oracle/{args.arm}/s5/{args.split}")
    out_dir = Path(args.out_dir) if args.out_dir else REPORT_DIR / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    latents = _load_latents(root)
    print(f"loaded {len(latents)} published latents from {root}", flush=True)

    # rank by the delivered (hi-tier) soft-IoU of the chosen readout
    scored = []
    for sid, fits in latents.items():
        f = fits.get(args.readout)
        if not f or f.get("status") != "ok":
            continue
        ev = f["eval"]
        hi = (ev.get("hi") or {}).get("soft_iou_minmax")
        lo = ev["low"]["soft_iou_minmax"]
        scored.append((sid, lo, hi if hi is not None else lo,
                       (lo - hi) if hi is not None else 0.0))
    scored.sort(key=lambda t: -t[2])

    picks = [(s[0], "success") for s in scored[: args.n_success]]
    # the result review names samples by prefix
    named = [sid for sid in latents
             if any(sid.startswith(f) for f in args.failures)]
    picks += [(f, "failure") for f in named]
    if not named:                       # fall back to the worst delivered masks
        picks += [(s[0], "failure") for s in scored[-3:]]

    from transformers import AutoProcessor

    from q3vl.where.calibrate import Calibrator
    from q3vl.where.pipeline import WhereADataSource
    from q3vl.where.projector import BasisProjector

    visual = load_vision_tower(Path(args.checkpoint), dtype=getattr(torch, args.dtype),
                               device=args.device)
    processor = AutoProcessor.from_pretrained(args.model_dir)
    source = WhereADataSource(visual, processor, device=args.device,
                              maskview_root=args.maskview_root, attach_hi=True)
    proj = BasisProjector()
    with torch.no_grad():
        proj.weight.copy_(torch.from_numpy(np.load(args.basis)))
    cal = Calibrator(CalibConfig(arm=args.arm, phi=PhiConfig()), projector=proj,
                     device=args.device)
    cal.freeze_projector()

    want = {sid for sid, _ in picks}
    made = []
    for prepared in source.iter_split(args.split):
        sid = prepared.sample.sample_id
        if sid not in want:
            continue
        kind = dict(picks)[sid]
        smp = prepared.sample
        with torch.no_grad():
            parts = cal.phi_for(smp)
            f = latents[sid][args.readout]
            lat = Latent.from_dict(f["latent"]).to(args.device)
            m_low, s_lo = mask_from_latent(parts.phi_dir.double(), lat)
            s_hi, _, dom = combine_then_upsample(
                parts.phi_dir.double(), lat, smp.grid_h, smp.grid_w,
                prepared.guide().to(args.device).double(),
                UpsampleConfig(), return_domain_report=True)
            m_hi = apply_readout(lat.readout, s_hi.reshape(-1), lat.rho)

        H, W = prepared.mask_hi.shape
        img = prepared.image_hi.permute(1, 2, 0).cpu().numpy()
        gt = prepared.mask_hi.cpu().numpy()
        pred = m_hi.reshape(H, W).float().cpu().numpy()
        s_field = s_hi.reshape(H, W).float().cpu().numpy()
        s_grid = s_lo.reshape(smp.grid_h, smp.grid_w).float().cpu().numpy()

        # numbers come from the raw fields, never from the displayed images
        ev = f["eval"]
        iou_hi = (ev.get("hi") or {}).get("soft_iou_minmax")
        iou_lo = ev["low"]["soft_iou_minmax"]
        vmin, vmax = _valid_scale(s_field, pred)

        fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
        _panel(axes[0], np.clip(img, 0, 1), f"I_in\n{sid[:20]}")
        _panel(axes[1], gt, "GT mask (C_GT)", cmap="magma", vmin=0, vmax=1)
        _panel(axes[2], pred, f"pred mask ({args.readout})\nhi soft-IoU {iou_hi:.4f}",
               cmap="magma", vmin=0, vmax=1)
        _panel(axes[3], s_field, f"s(p) delivered\nscale [{vmin:.2f},{vmax:.2f}] "
                                f"(valid cells)", cmap="coolwarm", vmin=vmin, vmax=vmax)
        _panel(axes[4], s_grid, f"s_low (F_pre {smp.grid_h}x{smp.grid_w})\n"
                                f"std {s_grid.std():.3f}", cmap="coolwarm",
               vmin=vmin, vmax=vmax)
        drop = (iou_lo - iou_hi) if iou_hi is not None else 0.0
        fig.suptitle(
            f"{kind.upper()}  {sid}  |  low {iou_lo:.4f} -> hi {iou_hi:.4f} "
            f"(drop {drop:+.4f})  |  raw s in [{dom['raw_min']:.2f},{dom['raw_max']:.2f}], "
            f"out-of-domain {dom['frac_out_of_domain']:.3%}  |  domain {list(S_DOMAIN)}",
            fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        name = out_dir / f"{kind}_{args.readout}_{sid}.png"
        fig.savefig(name, dpi=110)
        plt.close(fig)
        made.append({"sample_id": sid, "kind": kind, "file": str(name),
                     "low": iou_lo, "hi": iou_hi, "drop": drop,
                     "s_domain": dom, "s_low_std": float(s_grid.std())})
        print(json.dumps(made[-1], default=str), flush=True)
        want.discard(sid)
        if not want:
            break

    (out_dir / "index.json").write_text(json.dumps(made, indent=2, default=str))
    print(f"\nwrote {len(made)} figures to {out_dir}")
    source.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
