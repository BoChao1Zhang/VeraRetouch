#!/usr/bin/env python3
"""T2 optional cross-validation (IMPL_DOSSIER §4.3 item 8): ImageMagick
`hald:8` + `-hald-clut` versus our own tetrahedral applier.

Flow:
  1. `convert hald:8` -> identity hald image (512^2, 64^3 CLUT, 16-bit).
  2. Our tetrahedral applier bakes a real preset into that hald (16-bit PNG).
  3. ImageMagick applies the baked hald to a synthetic test image (-hald-clut).
  4. Compare against our applier run directly on the test image.
A row-order/domain mistake anywhere would blow the difference up to tens of
ΔE; agreement at ~quantization level validates the conventions end to end.

Usage:  python3 im_crosscheck.py --npy LUT.npy --work-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import apply_lut_tetrahedral, delta_e00


def sh(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def imread16(path: str) -> np.ndarray:
    import cv2

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cv2.imread failed: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]  # drop alpha (BGRA -> BGR)
    if img.dtype == np.uint16:
        img = img.astype(np.float64) / 65535.0
    else:
        img = img.astype(np.float64) / 255.0
    return img[:, :, ::-1].copy()  # BGR -> RGB


def imwrite16(path: str, rgb: np.ndarray) -> None:
    import cv2

    arr = np.round(np.clip(rgb, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(path, arr[:, :, ::-1])  # RGB -> BGR


def test_image(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w = 256, 384
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([
        xx / (w - 1),
        yy / (h - 1),
        0.5 + 0.5 * np.sin(6.28 * (xx / w + yy / h)),
    ], axis=-1)
    img = 0.8 * img + 0.2 * rng.random((h, w, 3))
    return np.clip(img, 0, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--work-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    wd = args.work_dir

    table = np.load(args.npy)
    p_hald = os.path.join(wd, "im_hald8.png")
    p_baked = os.path.join(wd, "hald8_baked.png")
    p_test = os.path.join(wd, "test_in.png")
    p_im_out = os.path.join(wd, "im_out.png")

    sh("convert", "hald:8", p_hald)
    hald = imread16(p_hald)
    # sanity on IM's encoding: R varies fastest, top-left black
    assert hald[0, 0].max() == 0.0 and hald[0, 1][0] > hald[0, 1][1:].max()
    imwrite16(p_baked, apply_lut_tetrahedral(table, hald.astype(np.float32)))

    img = test_image()
    imwrite16(p_test, img)
    sh("convert", p_test, p_baked, "-hald-clut", p_im_out)

    im_out = imread16(p_im_out)
    ours = apply_lut_tetrahedral(table, img.astype(np.float32))
    diff = np.abs(im_out - ours)
    de = delta_e00(np.clip(im_out, 0, 1), np.clip(ours, 0, 1))
    res = {
        "npy": args.npy,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "de00_max": float(de.max()),
        "de00_mean": float(de.mean()),
        "verdict_convention_ok": bool(np.percentile(de, 99) < 1.0),
        "note": "residual = IM trilinear-on-64^3 vs our tetrahedral-on-33^3 "
                "+ 16-bit quantization; a row-order error would give tens of ΔE",
    }
    with open(os.path.join(wd, "im_crosscheck.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
