"""Local edit 的 CPU↔GPU parity 验收（对标 gpu/parity.py 标准：ΔE00 de_mean<0.5）。

三层对拍：
  A. α 光栅 golden ↔ dataset_build cgt_raster（语义同源断言，可选）
  B. α 光栅 CPU(numpy) ↔ GPU(torch)（max|Δ|）
  C. 端到端：replay(全局+local) ↔ replay_batch(全局+local)，ΔE00 per case
     （含 radial/gradient/semantic α、亮/暗组合、全局preset+local 叠加、batch 广播）

monetgpt_sam3 env，cwd=/home/bc/VeraRetouch：
  MONETGPT_TORCH_DEVICE=cuda:0 python -m gpu_render.gpu.local_parity \
      --images img1.png img2.png [--semantic-mask m.png]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

PASS_DE = 0.5

CASES = [
    ("radial_in_bright",
     [{"params": {"LocalExposure2012": 0.85, "LocalClarity2012": 30,
                  "LocalContrast2012": 20},
       "amount": 1.0,
       "masks": [{"mask_type": "circulargradient",
                  "geom": {"Top": 0.18, "Bottom": 0.86, "Left": 0.22, "Right": 0.7,
                           "Angle": 23.0, "Feather": 70, "Flipped": "true"}}]}],
     {}),
    ("radial_out_dark",
     [{"params": {"LocalExposure2012": -0.9, "LocalHighlights2012": -45,
                  "LocalContrast2012": 15},
       "amount": 1.0,
       "masks": [{"mask_type": "circulargradient",
                  "geom": {"Top": 0.3, "Bottom": 0.9, "Left": 0.1, "Right": 0.55,
                           "Angle": -12.0, "Feather": 55, "Flipped": "false"}}]}],
     {}),
    ("gradient_angled",
     [{"params": {"LocalExposure2012": 0.7, "LocalShadows2012": 45,
                  "LocalHighlights2012": -55},
       "amount": 1.0,
       "masks": [{"mask_type": "gradient",
                  "geom": {"ZeroX": 0.62, "ZeroY": 0.44, "FullX": 0.3,
                           "FullY": 0.58, "Flipped": "false"}}]}],
     {}),
    ("semantic_sat",
     [{"params": {"LocalSaturation": 55, "LocalExposure2012": 0.25},
       "amount": 1.0, "masks": [], "alpha": "SEMANTIC"}],
     {}),
    ("global_plus_local",
     [{"params": {"LocalExposure2012": 0.6, "LocalClarity2012": 22},
       "amount": 0.85,
       "masks": [{"mask_type": "circulargradient",
                  "geom": {"Top": 0.2, "Bottom": 0.8, "Left": 0.3, "Right": 0.75,
                           "Angle": 40.0, "Feather": 85, "Flipped": "true"}}]}],
     {"Exposure2012": "0.3", "Contrast2012": "12", "Vibrance": "15"}),
]


def _de(a, b):
    from skimage.color import deltaE_ciede2000, rgb2lab
    de = deltaE_ciede2000(rgb2lab(np.clip(a, 0, 1)), rgb2lab(np.clip(b, 0, 1)))
    return float(de.mean()), float(np.percentile(de, 95))


def _semantic_alpha(h, w):
    """合成软斑语义 α（无外部 mask 时）：两个高斯软块的并集。"""
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    g1 = np.exp(-(((xx / w - 0.4) / 0.13) ** 2 + ((yy / h - 0.55) / 0.22) ** 2))
    g2 = np.exp(-(((xx / w - 0.52) / 0.09) ** 2 + ((yy / h - 0.35) / 0.1) ** 2))
    return np.clip(np.maximum(g1, g2), 0, 1).astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--semantic-mask", default=None)
    args = ap.parse_args()

    from PIL import Image
    import torch  # noqa: F401

    from gpu_render import ops_v2
    from gpu_render.local_replay import apply_locals, raster_alpha
    from gpu_render.replay import replay
    from gpu_render.gpu.gpu_replay import (DEVICE, _apply_cfg_np, replay_batch,
                                           to_batch, to_hwc_list)
    from gpu_render.gpu.local_gpu import raster_alpha_t
    from gpu_render.local_apply import FITS_DIR

    imgs = [np.asarray(Image.open(p).convert("RGB"), np.float32) / 255
            for p in args.images]
    h, w = imgs[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in imgs):
        import cv2
        imgs = [im if im.shape[:2] == (h, w) else cv2.resize(im, (w, h))
                for im in imgs]  # batch 锁步需同形

    # ---- A. golden ↔ dataset_build cgt_raster ----
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)),
                                        "dataset_build", "src"))
        from construct.mask_synth import cgt_raster
        worst = 0.0
        for _, corrs, _ in CASES:
            for c in corrs:
                for m in c["masks"]:
                    d = np.abs(raster_alpha(m["mask_type"], m["geom"], h, w)
                               - cgt_raster(m["mask_type"], m["geom"], h, w)).max()
                    worst = max(worst, float(d))
        print(f"A. raster golden↔cgt_raster max|Δ| = {worst:.2e} "
              f"{'PASS' if worst < 1e-5 else 'FAIL'}")
    except Exception as e:  # noqa: BLE001
        print(f"A. cgt_raster 对拍跳过（{type(e).__name__}: {e}）")

    # ---- B. raster CPU ↔ GPU ----
    worst = 0.0
    for _, corrs, _ in CASES:
        for c in corrs:
            for m in c["masks"]:
                a_np = raster_alpha(m["mask_type"], m["geom"], h, w)
                a_t = raster_alpha_t(m["mask_type"], m["geom"], h, w,
                                     DEVICE).cpu().numpy()
                worst = max(worst, float(np.abs(a_np - a_t).max()))
    print(f"B. raster CPU↔GPU max|Δ| = {worst:.2e} "
          f"{'PASS' if worst < 1e-4 else 'FAIL'}")

    # ---- C. 端到端 ----
    if args.semantic_mask:
        import cv2
        sm = cv2.imread(args.semantic_mask, 0).astype(np.float32) / 255
        sem = cv2.resize(sm, (w, h))
        k = max(3, int(round(min(h, w) * 0.012 * 6)) | 1)
        sem = np.clip(cv2.GaussianBlur(sem, (k, k), min(h, w) * 0.012), 0, 1)
    else:
        sem = _semantic_alpha(h, w)

    n_fail = 0
    for name, corrs, extra_attrs in CASES:
        corrs = [dict(c) for c in corrs]
        for c in corrs:
            if c.get("alpha") == "SEMANTIC":
                c["alpha"] = sem
        preset = {"attrs": dict(extra_attrs), "curves": {}, "locals": corrs}
        # CPU golden（逐图）
        refs = [np.clip(replay(im.copy(), preset, ops_v2.REGISTRY,
                               _apply_cfg_np)[0], 0, 1) for im in imgs]
        # GPU（batch 广播）
        outs, info = replay_batch(to_batch(imgs), preset)
        outs = to_hwc_list(outs)
        des = [_de(r, o) for r, o in zip(refs, outs)]
        de_mean = max(d[0] for d in des)
        de_p95 = max(d[1] for d in des)
        ok = de_mean < PASS_DE
        n_fail += (not ok)
        fb = f" fallback={info['fallback_ops']}" if info["fallback_ops"] else ""
        print(f"C. {name:<18} de_mean={de_mean:.4f} de_p95={de_p95:.4f} "
              f"{'PASS' if ok else 'FAIL'}{fb}")
    print("=== LOCAL PARITY", "PASS" if n_fail == 0 else f"FAIL({n_fail})", "===")


if __name__ == "__main__":
    main()
