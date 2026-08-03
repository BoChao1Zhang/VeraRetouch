"""Minimal viz for A0/E1: GT | prediction | DeltaE00 heatmap panels on the
train-hald canvas (qualitative; quantitative eval is on held-out colors).

Usage:
  python -m model.glut_repro.viz a0-ckpt --ckpt CKPT.pt --lut-id ID \
      --out OUT.png
  python -m model.glut_repro.viz e1-refit --lut-id ID --npy NPY33 --n 8 \
      --out OUT.png [--steps 3000]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO)

from model.glut_repro import data  # noqa: E402
from model.glut_repro.model import BatchedGLUT  # noqa: E402

sys.path.insert(0, os.path.join(_REPO, "tools", "cube"))
from cubelib import delta_e00  # noqa: E402


def render(x_img: np.ndarray, gt_img: np.ndarray, pred_img: np.ndarray,
           out_png: str, title: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    de = delta_e00(pred_img, gt_img)
    fig, axes = plt.subplots(1, 4, figsize=(22, 3.2))
    for ax, img, name in zip(
            axes[:3], (x_img, gt_img, pred_img),
            ("input (identity hald)", "GT (tetrahedral)", "prediction")):
        ax.imshow(img)
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    im = axes[3].imshow(de, cmap="magma", vmin=0.0, vmax=3.0)
    axes[3].set_title(
        f"dE00  mean {de.mean():.3f}  p99 {np.percentile(de, 99):.2f}  "
        f"max {de.max():.2f}", fontsize=9)
    axes[3].axis("off")
    fig.colorbar(im, ax=axes[3], fraction=0.02)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return {"de_mean": float(de.mean()), "de_max": float(de.max())}


def _hald_pair(kind: str, lut_id: str) -> tuple[np.ndarray, np.ndarray]:
    x = data.train_colors()
    gt = np.load(data._cache_path(kind, lut_id, "train")).astype(np.float32)
    shape = (1024, 2048, 3)
    return x.reshape(shape), gt.reshape(shape)


@torch.no_grad()
def _predict_img(model: BatchedGLUT, k: int, x_img: np.ndarray,
                 device: str) -> np.ndarray:
    flat = x_img.reshape(-1, 3)
    out = np.empty_like(flat)
    for i in range(0, flat.shape[0], 1 << 16):
        j = min(i + (1 << 16), flat.shape[0])
        xb = torch.from_numpy(flat[i:j]).to(device).unsqueeze(0)
        out[i:j] = model.predict(xb)[0].cpu().numpy() if model.B == 1 else \
            model.predict(xb.expand(model.B, -1, -1))[k].cpu().numpy()
    return out.reshape(x_img.shape)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("a0-ckpt")
    a.add_argument("--ckpt", required=True)
    a.add_argument("--lut-id", required=True)
    a.add_argument("--out", required=True)
    e = sub.add_parser("e1-refit")
    e.add_argument("--lut-id", required=True)
    e.add_argument("--npy", required=True)
    e.add_argument("--n", type=int, required=True)
    e.add_argument("--steps", type=int, default=3000)
    e.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.cmd == "a0-ckpt":
        ck = torch.load(args.ckpt, map_location=device, weights_only=False)
        ids = ck["lut_ids"]
        k = ids.index(args.lut_id)
        model = BatchedGLUT(len(ids), ck["n_gaussians"]).to(device)
        model.load_state_dict(ck["state_dict"])
        x_img, gt_img = _hald_pair("a0", args.lut_id)
        pred = _predict_img(model, k, x_img, device)
        stats = render(x_img, gt_img, pred, args.out,
                       f"A0 {args.lut_id}  N={ck['n_gaussians']} "
                       f"arm={ck['arm']}")
    else:
        from model.glut_repro.fit_e1 import E1Fitter
        data.build_gt("e1", args.lut_id, args.npy, ("train",))
        x_img, gt_img = _hald_pair("e1", args.lut_id)
        gts = gt_img.reshape(1, -1, 3).astype(np.float16)
        fitter = E1Fitter(x_img.reshape(-1, 3), gts, args.n, device=device,
                          steps=args.steps)
        fitter.fit()
        pred = _predict_img(fitter.model, 0, x_img, device)
        stats = render(x_img, gt_img, pred, args.out,
                       f"E1 {args.lut_id}  N={args.n}  "
                       f"alive={fitter.stats['alive_frac'][0]:.2f}")
    print(stats)


if __name__ == "__main__":
    main()
