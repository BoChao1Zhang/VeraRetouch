"""A0 delivery viz: success / failure panels for the rec arm + the full-arm
collapse.

Each panel is two rows:
  row 1 (natural photo)  input | GT = native cube via tetrahedral | GLUT pred |
                         dE00 heat map
  row 2 (Hald canvas)    GT hald | pred hald | dE00 hald map | stats box

`--mode rec`   : 3 best + 3 worst LUTs of runs/rec (viz/success_*, viz/failure_*)
`--mode full`  : the same 3 worst LUTs, rec vs full arm side by side, to show
                 what the mis-scaled L_hc does (viz/failure_fullarm_*)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tools", "cube"))

from cubelib import delta_e00  # noqa: E402
from model.glut_repro import data  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = os.path.abspath(os.path.join(HERE, ".."))
VIZ = os.path.join(EXP, "viz")
ASSETS = os.path.join(VIZ, "assets")
HALD_SHAPE = (1024, 2048, 3)


def load_arm(arm: str):
    """{lut_id: (model, index)} over all chunk checkpoints of an arm."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    d = os.path.join(EXP, "runs", arm)
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("ckpt_chunk"):
            continue
        ck = torch.load(os.path.join(d, fn), map_location=dev,
                        weights_only=False)
        m = BatchedGLUT(len(ck["lut_ids"]), ck["n_gaussians"]).to(dev)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        for i, lid in enumerate(ck["lut_ids"]):
            out[lid] = (m, i)
    return out, dev


@torch.no_grad()
def predict(model: BatchedGLUT, k: int, flat: np.ndarray, dev: str,
            chunk: int = 1 << 16) -> np.ndarray:
    out = np.empty_like(flat, dtype=np.float32)
    for i in range(0, flat.shape[0], chunk):
        j = min(i + chunk, flat.shape[0])
        xb = torch.from_numpy(flat[i:j].astype(np.float32)).to(dev)
        out[i:j] = model.predict(
            xb.unsqueeze(0).expand(model.B, -1, -1))[k].cpu().numpy()
    return out


def nat_images() -> list[tuple[str, np.ndarray]]:
    from PIL import Image
    ims = []
    for fn in sorted(os.listdir(ASSETS)):
        if fn.endswith(".png"):
            a = np.asarray(Image.open(os.path.join(ASSETS, fn)).convert("RGB"),
                           dtype=np.float32) / 255.0
            ims.append((fn, a))
    return ims


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = img.shape[0] // factor * factor, img.shape[1] // factor * factor
    return img[:h, :w].reshape(h // factor, factor, w // factor, factor,
                               3).mean(axis=(1, 3))


def panel(out_png: str, title: str, nat: dict, hald: dict, stats: str,
          de_vmax: float = 3.0) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.4))
    for j, (k, t) in enumerate((("x", "input photo"),
                                ("gt", "GT = native .cube (tetrahedral)"),
                                ("pred", nat["pred_title"]))):
        ax[0, j].imshow(np.clip(nat[k], 0, 1))
        ax[0, j].set_title(t, fontsize=10)
        ax[0, j].axis("off")
    im = ax[0, 3].imshow(nat["de"], cmap="magma", vmin=0, vmax=de_vmax)
    ax[0, 3].set_title(f"dE00 on photo   mean {nat['de'].mean():.3f}   "
                       f"p99 {np.percentile(nat['de'], 99):.2f}", fontsize=10)
    ax[0, 3].axis("off")
    fig.colorbar(im, ax=ax[0, 3], fraction=0.03)

    for j, (k, t) in enumerate((("gt", "GT hald (128^3 train colours)"),
                                ("pred", hald["pred_title"]))):
        ax[1, j].imshow(np.clip(hald[k], 0, 1))
        ax[1, j].set_title(t, fontsize=10)
        ax[1, j].axis("off")
    im = ax[1, 2].imshow(hald["de"], cmap="magma", vmin=0, vmax=de_vmax)
    ax[1, 2].set_title(f"dE00 on hald   mean {hald['de'].mean():.3f}   "
                       f"p99 {np.percentile(hald['de'], 99):.2f}", fontsize=10)
    ax[1, 2].axis("off")
    fig.colorbar(im, ax=ax[1, 2], fraction=0.03)
    ax[1, 3].axis("off")
    ax[1, 3].text(0.0, 0.98, stats, va="top", ha="left", fontsize=10,
                  family="monospace", transform=ax[1, 3].transAxes)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)
    plt.close(fig)
    print("wrote", out_png, flush=True)


def build(lut_id: str, arm_models, dev, nat_img, per_row, out_png, tag,
          arm2_models=None):
    lid_path = None
    with open(os.path.join(EXP, "config", "a0_luts_75.txt")) as f:
        for line in f:
            a, b = line.rstrip("\n").split("\t")
            if a == lut_id:
                lid_path = b
    assert lid_path
    name, img = nat_img
    flat = img.reshape(-1, 3)
    gt_nat = data.apply_native_tetrahedral(lid_path, flat).reshape(img.shape)
    m, k = arm_models[lut_id]
    pr_nat = predict(m, k, flat, dev).reshape(img.shape)
    de_nat = delta_e00(pr_nat.astype(np.float32), gt_nat.astype(np.float32))

    x_h = data.train_colors()
    gt_h = np.load(data._cache_path("a0", lut_id, "train")).astype(np.float32)
    pr_h = predict(m, k, x_h, dev)
    f = 4
    gt_hi = downsample(gt_h.reshape(HALD_SHAPE), f)
    pr_hi = downsample(pr_h.reshape(HALD_SHAPE), f)
    de_h = delta_e00(pr_hi.astype(np.float32), gt_hi.astype(np.float32))

    extra = ""
    if arm2_models is not None and lut_id in arm2_models:
        m2, k2 = arm2_models[lut_id]
        pr2 = predict(m2, k2, flat, dev).reshape(img.shape)
        de2 = delta_e00(pr2.astype(np.float32), gt_nat.astype(np.float32))
        extra = (f"\nfull arm (L_rec+10*L_hc+R_sp+mining)\n"
                 f"  photo dE00 mean {de2.mean():.3f}  (rec {de_nat.mean():.3f})")
        # replace the prediction tile with the full-arm one
        pr_nat, de_nat = pr2, de2

    stats = (f"LUT      {lut_id}\n"
             f"arm      {tag}\n"
             f"N        32   (22N+12 = 716 params)\n"
             f"held-out PSNR  {per_row['psnr_float']:.2f} dB\n"
             f"held-out dE00  {per_row['de00']['mean']:.3f} "
             f"(p99 {per_row['de00']['p99']:.2f})\n"
             f"GLUT anchor    45.47 dB / dE00 0.41\n"
             f"gap            {per_row['psnr_float'] - 45.47:+.2f} dB"
             + extra)
    panel(out_png, f"{lut_id}   [{tag}]",
          {"x": img, "gt": gt_nat, "pred": pr_nat, "de": de_nat,
           "pred_title": f"GLUT-32 prediction ({tag})"},
          {"gt": gt_hi, "pred": pr_hi, "de": de_h,
           "pred_title": f"GLUT-32 hald prediction ({tag})"},
          stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rec", "full"], default="rec")
    args = ap.parse_args()
    os.makedirs(VIZ, exist_ok=True)
    rows = {}
    with open(os.path.join(EXP, "runs", "rec", "per_lut.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            rows[r["lut_id"]] = r
    order = sorted(rows, key=lambda k: rows[k]["psnr_float"])
    worst, best = order[:3], order[-3:][::-1]
    nats = nat_images()
    rec_m, dev = load_arm("rec")

    if args.mode == "rec":
        for i, lid in enumerate(best):
            build(lid, rec_m, dev, nats[i % len(nats)], rows[lid],
                  os.path.join(VIZ, f"success_rec_{lid}.png"), "rec")
        for i, lid in enumerate(worst):
            build(lid, rec_m, dev, nats[i % len(nats)], rows[lid],
                  os.path.join(VIZ, f"failure_rec_{lid}.png"), "rec")
    else:
        full_m, _ = load_arm("full")
        frows = {}
        with open(os.path.join(EXP, "runs", "full", "per_lut.jsonl")) as f:
            for line in f:
                r = json.loads(line)
                frows[r["lut_id"]] = r
        # the 3 LUTs the full arm damaged most
        dmg = sorted(rows, key=lambda k: frows[k]["psnr_float"]
                     - rows[k]["psnr_float"])[:3]
        for i, lid in enumerate(dmg):
            build(lid, rec_m, dev, nats[i % len(nats)], frows[lid],
                  os.path.join(VIZ, f"failure_fullarm_{lid}.png"), "full",
                  arm2_models=full_m)


if __name__ == "__main__":
    main()
