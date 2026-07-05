"""本地算子应用：对每个 (op, value, probe) 用 monetGPT 算子复现 LR 效果。

before 图 = LR identity 渲染（gt/Identity/<probe>__+000.jpg），消除 LR 导出管线偏置，
只比较『算子本身』的变换。校正来自 fits/<op>.json（精调 workflow 产出）：
    {"value_map": [[lr_v, local_v], ...],   # 单调分段线性 LR值->本地值 remap
     "post_luma_lut": {"<lr_v>": [[x,y],...]}}  # 可选：每扫描值的残差 luma 曲线

monetGPT venv 运行：/home/bc/miniconda3/envs/monetgpt_sam3/bin/python -m gpu_render.local_apply [--ops ...] [--fits fits]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # VeraRetouch 根（gpu_render 包父目录）

from gpu_render.sweeps import (ALL_OPS, CALIB_ROOT, IDENTITY, OPS, fmt_value,
                                   baseline_local_config, sweep_points)

FITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fits")


def load_fit(op: str, fits_dir: str) -> dict:
    p = os.path.join(fits_dir, f"{op}.json")
    if os.path.exists(p):
        return json.load(open(p))
    return {}


def local_value(op: str, lr_v: float, fit: dict) -> float:
    vm = fit.get("value_map")
    if vm:
        xs, ys = zip(*sorted(vm))
        return float(np.interp(lr_v, xs, ys))
    cfg = baseline_local_config(op, lr_v)
    return cfg[op]


def _lut_keys(lr_v) -> list:
    keys = [str(lr_v)]
    if isinstance(lr_v, (int, float)):
        keys.append(fmt_value(lr_v))
    return keys


def apply_post_lut(img: np.ndarray, fit: dict, lr_v) -> np.ndarray:
    """声明式后校正：luma 残差曲线，和/或每通道 1D LUT（Temperature/Tint/HSL 类色偏用）。"""
    luts = fit.get("post_luma_lut") or {}
    pts = next((luts[k] for k in _lut_keys(lr_v) if k in luts), None)
    if pts:
        xs, ys = zip(*sorted(pts))
        luma = img @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        gain = np.interp(luma, xs, ys).astype(np.float32) - luma
        img = np.clip(img + gain[..., None], 0.0, 1.0)
    rgb = (fit.get("post_rgb_lut") or {})
    ch_luts = next((rgb[k] for k in _lut_keys(lr_v) if k in rgb), None)
    if ch_luts:
        out = img.copy()
        for i, ch in enumerate(("r", "g", "b")):
            pts = ch_luts.get(ch)
            if pts:
                xs, ys = zip(*sorted(pts))
                out[..., i] = np.interp(img[..., i], xs, ys).astype(np.float32)
        img = np.clip(out, 0.0, 1.0)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ops", default="")
    ap.add_argument("--fits", default=FITS_DIR)
    a = ap.parse_args()
    from gpu_render.image_ops.non_gimp_ops import apply_non_gimp_config, read_image, write_image
    from gpu_render import ops_v2

    ops = [o.strip() for o in a.ops.split(",") if o.strip()] or list(ALL_OPS)
    id_dir = os.path.join(CALIB_ROOT, "gt", IDENTITY)
    befores = {}   # stem -> (img, norm)
    for f in sorted(os.listdir(id_dir)):
        stem = f.split("__")[0]
        befores[stem] = read_image(os.path.join(id_dir, f))

    n = 0
    for op in ops:
        fit = load_fit(op, a.fits)
        impl = ops_v2.REGISTRY.get(op)   # ops_v2 重写/新实现优先
        if impl is None and op not in OPS:
            print(f"[local] {op}: no ops_v2 impl yet, skipped", flush=True)
            continue
        for label, attrs, elements in sweep_points(op):
            for stem, (img, norm) in befores.items():
                dst = os.path.join(CALIB_ROOT, "local", op, f"{stem}__{label}.jpg")
                gt = os.path.join(CALIB_ROOT, "gt", op, f"{stem}__{label}.jpg")
                if not os.path.exists(gt):
                    continue                      # GT 未渲出，跳过
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                if impl is not None:
                    out = impl(img.copy().astype(np.float32),
                               {"label": label, "attrs": attrs, "elements": elements,
                                "fit": fit})
                else:
                    lr_v = float(label)
                    out = apply_non_gimp_config({op: local_value(op, lr_v, fit)}, img.copy(), norm)
                    out = apply_post_lut(np.asarray(out, dtype=np.float32), fit, label)
                write_image(out, norm, dst + ".png")   # write_image 非 tif 走 PNG
                os.replace(dst + ".png", dst)          # 统一 .jpg 扩展名存 PNG 内容（无损）
                n += 1
        print(f"[local] {op} done", flush=True)
    print(f"[local] TOTAL {n} images", flush=True)


if __name__ == "__main__":
    main()
