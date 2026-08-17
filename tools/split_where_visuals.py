#!/usr/bin/env python
"""Split the dense where overview into PPT-friendly family and method panels."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


SOURCE = Path("docs/assets/where_arms_20260815/where_arms_overlay.png")
OUT_DIR = Path("docs/assets/where_arms_20260815/ppt_panels")

# Coordinates are tied to the fixed 3485x4998 overview rendered on 2026-08-15.
# Each family has three samples.  Splitting columns avoids a 10-method strip.
FAMILIES = {
    "radial": (300, 1150),
    "band": (1480, 1150),
    "linear": (2630, 1150),
    "semantic": (3780, 1150),
}
PANELS = {
    "a_input_gt_st_segsam_matte": (0, 0, 1720, 1150),
    "b_samdec_liif_prnd_condinst_prior": (1720, 0, 1765, 1150),
}


def main() -> None:
    image = Image.open(SOURCE)
    if image.size != (3485, 4998):
        raise RuntimeError(f"unexpected source geometry: {image.size}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for family, (top, height) in FAMILIES.items():
        family_crop = image.crop((0, top, image.width, top + height))
        for suffix, (left, upper, width, panel_height) in PANELS.items():
            panel = family_crop.crop((left, upper, left + width, upper + panel_height))
            panel.save(OUT_DIR / f"where_{family}_{suffix}.png")


if __name__ == "__main__":
    main()
