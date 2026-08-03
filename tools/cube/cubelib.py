"""Shared helpers for the T2 cube corpus toolchain.

Conventions (verified against colour-science 0.4.7 source, see NOTES.md):
- .cube file row order: R varies fastest, B slowest; colour reshapes with
  order='F' into table[r_idx, g_idx, b_idx] -> (R, G, B).
- Canonical npy: shape (33, 33, 33, 3) float32, index [r, g, b], RGB channel
  order, implicit domain [0, 1] (DOMAIN_MIN/MAX already baked in at resample).
- Hald train split (GLUT protocol): 128 uniform 8-bit values per channel
  {0, 2, ..., 254} -> 128^3 colors -> 1024x2048x3 image, R fastest.
- Hald eval split: the remaining 256^3 - 128^3 8-bit colors -> 3584x4096x3.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

CANONICAL_SIZE = 33
TRAIN_HALD_SHAPE = (1024, 2048, 3)   # 128^3 = 2,097,152 px
EVAL_HALD_SHAPE = (3584, 4096, 3)    # 256^3 - 128^3 = 14,680,064 px

JOURNAL_ARCHIVE = "/var/cache/veradata/annot_review/journal-archive"
RECIPE_ROOTS = ["/home/bc/data/datasets/recipes"]
DEFAULT_NPY_DIR = "/var/cache/veradata/dcube/npy33"


def md5_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def preset_slug(path: str) -> str:
    """Stable id for a preset file: <bucket>__<stem> (e.g. quandian__quandian_011451)."""
    bucket = os.path.basename(os.path.dirname(path))
    stem = os.path.splitext(os.path.basename(path))[0]
    return f"{bucket}__{stem}"


# ---------------------------------------------------------------------------
# Color difference (sRGB D65 -> Lab -> CIEDE2000), matching skimage rgb2lab
# ---------------------------------------------------------------------------

def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Gamma-encoded sRGB in [0,1] -> CIELAB (D65 2-deg), vectorized."""
    import colour

    xyz = colour.sRGB_to_XYZ(np.clip(rgb, 0.0, 1.0))
    return colour.XYZ_to_Lab(xyz)


def delta_e00(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """CIEDE2000 between two gamma-encoded sRGB arrays in [0,1]."""
    import colour

    return colour.difference.delta_E(
        srgb_to_lab(rgb_a), srgb_to_lab(rgb_b), method="CIE 2000"
    )


# ---------------------------------------------------------------------------
# Hald generation
# ---------------------------------------------------------------------------

def hald_train_image() -> np.ndarray:
    """128^3 uniformly sampled 8-bit colors (values {0,2,...,254}) as a
    1024x2048x3 float32 image in [0,1]. R varies fastest, then G, then B."""
    b, g, r = np.mgrid[0:256:2, 0:256:2, 0:256:2]  # index order: R fastest last
    colors = np.stack([r, g, b], axis=-1).reshape(-1, 3)  # (128^3, 3), R fastest
    return (colors.astype(np.float32) / 255.0).reshape(TRAIN_HALD_SHAPE)


def hald_eval_image() -> np.ndarray:
    """The held-out 256^3 - 128^3 8-bit colors (at least one odd channel) as a
    3584x4096x3 float32 image in [0,1]. Deterministic order: R fastest."""
    n = 256
    vals = np.arange(n, dtype=np.uint8)
    b, g, r = np.meshgrid(vals, vals, vals, indexing="ij")
    colors = np.stack([r, g, b], axis=-1).reshape(-1, 3)  # R fastest
    train_mask = (colors % 2 == 0).all(axis=1)
    held = colors[~train_mask]
    assert held.shape[0] == EVAL_HALD_SHAPE[0] * EVAL_HALD_SHAPE[1]
    return (held.astype(np.float32) / 255.0).reshape(EVAL_HALD_SHAPE)


# ---------------------------------------------------------------------------
# LUT application (two paths)
# ---------------------------------------------------------------------------

def apply_lut_grid_sample(table: np.ndarray, img: np.ndarray,
                          device: str = "cpu") -> np.ndarray:
    """Training path: torch F.grid_sample (trilinear, align_corners=True).

    table: (S,S,S,3) float, index [r,g,b], RGB output channels, domain [0,1].
    img:   (H,W,3) float in [0,1], gamma-encoded sRGB.
    """
    import torch
    import torch.nn.functional as F

    t = torch.from_numpy(np.ascontiguousarray(table, dtype=np.float32))
    # input (N,C,D,H,W) with D<-b, H<-g, W<-r  =>  permute [r,g,b,c] -> [c,b,g,r]
    lut = t.permute(3, 2, 1, 0).unsqueeze(0).to(device)
    grid = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32))
    grid = (grid * 2.0 - 1.0).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,3) xyz=RGB
    out = F.grid_sample(lut, grid, mode="bilinear",
                        padding_mode="border", align_corners=True)
    return out[0, :, 0].permute(1, 2, 0).cpu().numpy()  # (H,W,3)


def apply_lut_tetrahedral(table: np.ndarray, img: np.ndarray,
                          tile_rows: int = 256) -> np.ndarray:
    """GT path: colour tetrahedral interpolation, tiled over rows to bound RAM.

    Same table/img conventions as apply_lut_grid_sample.
    """
    from colour import LUT3D
    from colour.algebra import table_interpolation_tetrahedral

    lut = LUT3D(table.astype(np.float64))
    out = np.empty(img.shape, dtype=np.float32)
    for y0 in range(0, img.shape[0], tile_rows):
        y1 = min(y0 + tile_rows, img.shape[0])
        out[y0:y1] = lut.apply(
            img[y0:y1].astype(np.float64),
            interpolator=table_interpolation_tetrahedral,
        ).astype(np.float32)
    return out


def identity_table(size: int = CANONICAL_SIZE) -> np.ndarray:
    from colour import LUT3D

    return LUT3D.linear_table(size).astype(np.float32)
