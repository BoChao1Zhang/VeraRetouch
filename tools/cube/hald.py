#!/usr/bin/env python3
"""T2 hald: Hald image generator + two-path LUT applier (GLUT protocol).

Generator:
  train : 128^3 uniformly sampled 8-bit colors (values {0,2,...,254})
          -> 1024x2048x3, R varies fastest (Hald convention, top-left black,
          bottom-right near-white).
  eval  : held-out 256^3 - 128^3 colors (>=1 odd channel) -> 3584x4096x3,
          deterministic R-fastest order over the complement set.
  This is a COLOR-SPACE split (GLUT protocol), not an image split.

Applier:
  grid_sample  : torch F.grid_sample trilinear, align_corners=True
                 (training path — differentiable, matches renderer).
  tetrahedral  : colour-science table_interpolation_tetrahedral (GT path).

Both consume the canonical (33,33,33,3) [r,g,b]-indexed RGB npy from parse.py.

Usage:
  python3 hald.py gen --out-dir DIR [--png]          # write train/eval npy (+png)
  python3 hald.py apply --npy LUT.npy --hald H.npy --method {grid_sample,tetrahedral} \
                  --out OUT.npy
  python3 hald.py selftest                           # generator + both paths mutual check
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cubelib import (
    EVAL_HALD_SHAPE,
    TRAIN_HALD_SHAPE,
    apply_lut_grid_sample,
    apply_lut_tetrahedral,
    delta_e00,
    hald_eval_image,
    hald_train_image,
    identity_table,
)


def save_png(img: np.ndarray, path: str) -> None:
    from PIL import Image

    Image.fromarray(np.round(img * 255).astype(np.uint8)).save(path)


def cmd_gen(args) -> None:
    os.makedirs(args.out_dir, exist_ok=True)
    train = hald_train_image()
    ev = hald_eval_image()
    assert train.shape == TRAIN_HALD_SHAPE and ev.shape == EVAL_HALD_SHAPE
    np.save(os.path.join(args.out_dir, "hald_train_1024x2048.npy"), train)
    np.save(os.path.join(args.out_dir, "hald_eval_3584x4096.npy"), ev)
    if args.png:
        save_png(train, os.path.join(args.out_dir, "hald_train_1024x2048.png"))
        save_png(ev, os.path.join(args.out_dir, "hald_eval_3584x4096.png"))
    # split sanity: disjoint and exhaustive over 8-bit color space
    t = np.round(train.reshape(-1, 3) * 255).astype(np.int64)
    e = np.round(ev.reshape(-1, 3) * 255).astype(np.int64)
    t_keys = (t[:, 0] << 16) | (t[:, 1] << 8) | t[:, 2]
    e_keys = (e[:, 0] << 16) | (e[:, 1] << 8) | e[:, 2]
    assert len(np.unique(t_keys)) == 128 ** 3
    assert len(np.unique(e_keys)) == 256 ** 3 - 128 ** 3
    assert len(np.intersect1d(t_keys, e_keys)) == 0
    print(json.dumps({
        "train": list(train.shape), "eval": list(ev.shape),
        "disjoint_and_exhaustive": True, "out_dir": args.out_dir,
    }))


def cmd_apply(args) -> None:
    table = np.load(args.npy)
    hald = np.load(args.hald)
    if args.method == "grid_sample":
        out = apply_lut_grid_sample(table, hald, device=args.device)
    else:
        out = apply_lut_tetrahedral(table, hald)
    np.save(args.out, out.astype(np.float32))
    print(json.dumps({"out": args.out, "shape": list(out.shape),
                      "range": [float(out.min()), float(out.max())]}))


def cmd_selftest(args) -> None:
    """Small-sample self-check on real conventions, no corpus files needed."""
    rng = np.random.default_rng(0)

    # 1) generator invariants (subsampled shapes for speed of assertion only)
    train = hald_train_image()
    assert train.shape == TRAIN_HALD_SHAPE
    # R varies fastest along a row: first two pixels differ only in R by 2/255
    d0 = (train[0, 1] - train[0, 0]) * 255
    assert np.allclose(d0, [2, 0, 0]), d0
    assert train[0, 0].tolist() == [0, 0, 0]
    assert np.allclose(train[-1, -1] * 255, [254, 254, 254])

    # 2) identity LUT through both paths must reproduce the input exactly
    ident = identity_table(33)
    sub = train[::8, ::8]  # 128x256 probe, still spans the full color range
    gs = apply_lut_grid_sample(ident, sub)
    tt = apply_lut_tetrahedral(ident, sub)
    e_gs = float(np.abs(gs - sub).max())
    e_tt = float(np.abs(tt - sub).max())
    assert e_gs < 1e-6 and e_tt < 1e-6, (e_gs, e_tt)

    # 3) mutual check on a random smooth non-identity LUT: both interpolators
    #    agree at LUT nodes exactly and closely off-node
    t = identity_table(33) ** 1.5
    t += rng.normal(0, 0.01, t.shape).astype(np.float32)
    t = np.clip(t, 0, 1)
    nodes = identity_table(33).reshape(-1, 3)[::7][:4096].reshape(64, 64, 3)
    gs = apply_lut_grid_sample(t, nodes)
    tt = apply_lut_tetrahedral(t, nodes)
    node_dev = float(np.abs(gs - tt).max())
    assert node_dev < 1e-5, node_dev  # exact at nodes for both schemes
    gs = apply_lut_grid_sample(t, sub)
    tt = apply_lut_tetrahedral(t, sub)
    off = np.abs(gs - tt)
    de = delta_e00(np.clip(gs, 0, 1), np.clip(tt, 0, 1))
    print(json.dumps({
        "identity_max_abs": {"grid_sample": e_gs, "tetrahedral": e_tt},
        "node_agreement_max_abs": node_dev,
        "offnode_max_abs": float(off.max()),
        "offnode_mean_abs": float(off.mean()),
        "offnode_de00_max": float(de.max()),
        "offnode_de00_mean": float(de.mean()),
        "note": "off-node differences are trilinear-vs-tetrahedral scheme "
                "difference, expected small but nonzero",
    }, indent=1))
    print("hald.py selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen", help="generate train/eval hald images")
    g.add_argument("--out-dir", required=True)
    g.add_argument("--png", action="store_true", help="also write PNGs")
    g.set_defaults(fn=cmd_gen)

    a = sub.add_parser("apply", help="apply a canonical LUT npy to a hald npy")
    a.add_argument("--npy", required=True)
    a.add_argument("--hald", required=True)
    a.add_argument("--method", choices=["grid_sample", "tetrahedral"],
                   required=True)
    a.add_argument("--device", default="cpu")
    a.add_argument("--out", required=True)
    a.set_defaults(fn=cmd_apply)

    s = sub.add_parser("selftest", help="run built-in checks")
    s.set_defaults(fn=cmd_selftest)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
