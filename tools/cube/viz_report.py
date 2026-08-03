#!/usr/bin/env python3
"""T2 report visualization: side-by-side panels for a few presets.

For each chosen preset id: hald-train input | grid_sample output | tetrahedral
output | 20x-amplified |gs-tt| difference heatmap, stacked into one PNG.

Usage:
  python3 viz_report.py --npy-dir DIR --hald-dir DIR --out-dir DIR id1 [id2 ...]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import apply_lut_grid_sample, apply_lut_tetrahedral


def to8(img: np.ndarray) -> np.ndarray:
    return np.round(np.clip(img, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    from PIL import Image

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy-dir", required=True)
    ap.add_argument("--hald-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("ids", nargs="+")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    hald = np.load(os.path.join(args.hald_dir, "hald_train_1024x2048.npy"))
    small = hald[::4, ::4]  # 256x512 per panel keeps files small
    for pid in args.ids:
        table = np.load(os.path.join(args.npy_dir, pid + ".npy"))
        gs = apply_lut_grid_sample(table, small)
        tt = apply_lut_tetrahedral(table, small)
        diff = np.abs(gs - tt) * 20.0
        panel = np.concatenate([to8(small), to8(gs), to8(tt), to8(diff)], axis=0)
        out = os.path.join(args.out_dir, f"panel_{pid}.png")
        Image.fromarray(panel).save(out)
        print(f"{out}  (rows: input | grid_sample | tetrahedral | 20x|gs-tt|)"
              f"  max|gs-tt|={np.abs(gs - tt).max():.5f}")


if __name__ == "__main__":
    main()
