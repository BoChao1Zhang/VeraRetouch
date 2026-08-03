#!/usr/bin/env python3
"""Selfcheck for tools/bgr_check — validates the checker itself before the
100-pair verdict is trusted.

Checks:
 1. parser cross-check on 3 real production .cube files:
    colour.read_LUT_IridasCube table[r,g,b] == load_lut grid[b,g,r].T (<1e-6).
 2. identity LUT through the tetrahedral path is a no-op (<1e-5).
 3. detector sensitivity: on a synthetic smooth non-symmetric LUT and a random
    image, the simulated BGR-swap bug must be loudly visible
    (ΔE00 p50 > 5) while the correct-path tri-vs-tet disagreement stays small
    (ΔE00 p50 < 1), and the permutation test must pick RGB for the correct
    path and NOT RGB for the swapped one.
 4. composite endpoint semantics match production composite_srgb on the
    endpoints alpha==0 / alpha==1 exactly.
 5. quantile/corr helpers sanity (identity corr == I, quantiles monotone).

Usage: /home/bc/miniconda3/bin/python3 selfcheck.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

from dataset_build.lut_io import load_lut  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


def main() -> int:
    rng = np.random.default_rng(20260803)

    # 1. parser cross-check on real cubes
    cubes = sorted(glob.glob("/home/bc/data/datasets/recipes/quandian/*.cube"))[:2]
    cubes += sorted(glob.glob("/home/bc/data/datasets/recipes/e18/*.cube"))[:1]
    for path in cubes:
        table, domain = C.read_cube_colour(path)
        grid, dmin, dmax = load_lut(path)
        d = float(np.max(np.abs(table - grid.transpose(2, 1, 0, 3))))
        check(f"parser-crosscheck {os.path.basename(path)}", d < 1e-6,
              f"maxabs={d:.2e}")

    # 2. identity LUT no-op
    n = 17
    ax = np.linspace(0, 1, n)
    r, g, b = np.meshgrid(ax, ax, ax, indexing="ij")
    ident = np.stack([r, g, b], axis=-1)  # table[r,g,b] identity
    img = rng.random((64, 96, 3)).astype(np.float32)
    out = C.apply_lut_tetrahedral_rgb(ident, img)
    check("identity tetrahedral no-op", float(np.max(np.abs(out - img))) < 1e-5,
          f"maxabs={float(np.max(np.abs(out - img))):.2e}")

    # 3. sensitivity: smooth asymmetric LUT
    n = 33
    ax = np.linspace(0, 1, n)
    r, g, b = np.meshgrid(ax, ax, ax, indexing="ij")
    table = np.stack([
        np.clip(r ** 0.8 * 0.9 + 0.05 * g, 0, 1),
        np.clip(g * 0.7 + 0.2 * b, 0, 1),
        np.clip(b ** 1.3 * 0.95 + 0.05, 0, 1),
    ], axis=-1)  # table[r,g,b] -> RGB, channel-asymmetric
    grid_bgr = table.transpose(2, 1, 0, 3)  # prod storage convention
    img = rng.random((128, 128, 3)).astype(np.float32)

    tri = C.apply_lut_trilinear_bgr(img, grid_bgr)          # prod math
    tet = C.apply_lut_tetrahedral_rgb(table, img)           # colour path
    swap = C.apply_lut_tetrahedral_rgb(
        table, np.ascontiguousarray(img[..., ::-1]))[..., ::-1]

    de_ok = C.quantiles(C.delta_e00_tiled(tri, tet))
    de_bug = C.quantiles(C.delta_e00_tiled(tri, swap))
    check("tri-vs-tet small on correct path", de_ok["p50"] < 1.0,
          f"p50={de_ok['p50']:.3f}")
    check("BGR-swap bug loudly visible", de_bug["p50"] > 5.0,
          f"p50={de_bug['p50']:.2f}")
    check("perm test picks RGB on correct path",
          C.permutation_test(tet, tri)["best_perm"] == "RGB")
    # hypothesis (b) detector: an output-side channel permutation in the
    # archived after must be flagged as the best-matching non-RGB perm
    bgr_after = np.ascontiguousarray(tri[..., ::-1])
    check("perm test detects output BGR permutation",
          C.permutation_test(tet, bgr_after)["best_perm"] == "BGR",
          f"best={C.permutation_test(tet, bgr_after)['best_perm']}")
    # note: a full BGR-*apply* bug (input+output swapped) is not a pure output
    # permutation; its detector is the ΔE00 magnitude check above.

    # 4. composite endpoints
    before = rng.random((32, 32, 3)).astype(np.float32)
    edited = rng.random((32, 32, 3)).astype(np.float32)
    alpha = rng.random((32, 32)).astype(np.float32)
    alpha[:8] = 0.0
    alpha[-8:] = 1.0
    comp = C.composite_prod(before, edited, alpha)
    check("composite alpha==0 returns before exactly",
          bool((comp[:8] == before[:8]).all()))
    check("composite alpha==1 returns edited exactly",
          bool((comp[-8:] == edited[-8:]).all()))

    # 5. helpers
    cm = np.asarray(C.corr_matrix(img, img))
    # diag must be exactly 1 (self-correlation); off-diag is the natural
    # cross-channel correlation of a random image (~1/sqrt(N)), just small
    check("corr(self) diag == 1", bool(np.allclose(np.diag(cm), 1.0, atol=1e-9)),
          f"diag={np.diag(cm)}")
    check("corr(self) offdiag small (random img)",
          float(np.max(np.abs(cm - np.diag(np.diag(cm))))) < 0.05)
    q = C.quantiles(np.abs(rng.normal(size=(64, 64))).astype(np.float32))
    check("quantiles monotone",
          q["p50"] <= q["p90"] <= q["p95"] <= q["p99"] <= q["max"])

    print(f"\n{'ALL PASS' if not FAILURES else 'FAILURES: ' + ', '.join(FAILURES)}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
