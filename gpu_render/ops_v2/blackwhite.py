"""BlackWhite —— thin 适配器：实现已合入 image_ops/non_gimp_ops.py。

带参数（原 fits/BlackWhite.json bands）已烘焙进 non_gimp_ops.BW_BAND_PARAMS。
"""
from __future__ import annotations

import numpy as np

from gpu_render.image_ops.non_gimp_ops import BW_BANDS, convert_black_white


def _bw_v2(img: np.ndarray, ctx: dict) -> np.ndarray:
    attrs = ctx.get("attrs") or {}
    mixer = {b: float(attrs[f"GrayMixer{b}"]) for b in BW_BANDS
             if f"GrayMixer{b}" in attrs}
    return convert_black_white(np.asarray(img, dtype=np.float32), mixer)


OPS_V2 = {"BlackWhite": _bw_v2}
