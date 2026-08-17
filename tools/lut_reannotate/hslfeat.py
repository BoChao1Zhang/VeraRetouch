"""Deterministic HSL 8-band response features read straight off a LUT grid.

The VLM sees pictures; pictures do not say "the blue band rotated 14 degrees
toward cyan and lost 30% of its saturation".  This module answers that question
from the grid itself, so the annotation is grounded in arithmetic the annotator
cannot argue with, exactly as the retired ``_probe_table`` grounded the old
naming pass in measured LAB deltas.

Sampling spec (the whole thing is reproducible from these five constants):

* ``BANDS`` -- the eight Lightroom HSL bands at their standard sRGB hue-wheel
  centres.  These centres are this tool's convention, not a value read out of
  Adobe: the panel exposes band *names*, and 0/30/60/120/180/240/270/300 are the
  canonical wheel positions those names denote.
* ``HUE_OFFSETS`` / ``SAT_LEVELS`` / ``LUM_LEVELS`` -- a 3x3x3 grid per band, so
  27 probe colours per band and 216 in total.  Mid lightness and mid-to-high
  saturation on purpose: a band's response is only meaningful where the band has
  colour to work with, and near-black / near-white samples mostly measure the
  tone curve twice.
* ``GRAY_LEVELS`` -- the neutral ramp, which is where colour cast and contrast
  are read.  Keeping cast off the saturated samples is the same discipline the
  old prompt spelled out as "colour temperature is read on the neutral row only".

Reported per band: median hue rotation in degrees, median *relative* saturation
change in percent, and median lightness change in HSL points.  Median rather
than mean because a band that clips at one corner of the grid should not drag
the whole row.
"""
from __future__ import annotations

import colorsys
import math
from typing import Any, Mapping, Sequence

import numpy as np

BANDS: tuple[tuple[str, float], ...] = (
    ("红", 0.0),
    ("橙", 30.0),
    ("黄", 60.0),
    ("绿", 120.0),
    ("浅绿", 180.0),
    ("蓝", 240.0),
    ("紫", 270.0),
    ("洋红", 300.0),
)
HUE_OFFSETS: tuple[float, ...] = (-12.0, 0.0, 12.0)
SAT_LEVELS: tuple[float, ...] = (0.45, 0.70, 0.95)
LUM_LEVELS: tuple[float, ...] = (0.35, 0.50, 0.65)
GRAY_LEVELS: tuple[float, ...] = (0.15, 0.30, 0.50, 0.70, 0.85)

SPEC_REV = "hsl8-v1"


def _wrap180(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def band_samples() -> tuple[np.ndarray, list[tuple[str, float, float, float]]]:
    """The 216 probe colours as linear-index RGB rows plus their HSL labels."""
    rows: list[list[float]] = []
    labels: list[tuple[str, float, float, float]] = []
    for name, centre in BANDS:
        for offset in HUE_OFFSETS:
            hue = (centre + offset) % 360.0
            for sat in SAT_LEVELS:
                for lum in LUM_LEVELS:
                    red, green, blue = colorsys.hls_to_rgb(hue / 360.0, lum, sat)
                    rows.append([red, green, blue])
                    labels.append((name, hue, sat, lum))
    return np.asarray(rows, dtype=np.float32), labels


def gray_samples() -> np.ndarray:
    return np.asarray([[level] * 3 for level in GRAY_LEVELS], dtype=np.float32)


def _srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """sRGB in 0..1 -> CIE L*a*b* (D65), same convention as the preset bank."""
    value = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    linear = np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [[0.4124564, 0.3575761, 0.1804375],
         [0.2126729, 0.7151522, 0.0721750],
         [0.0193339, 0.1191920, 0.9503041]]
    )
    xyz = linear @ matrix.T
    white = np.array([0.95047, 1.0, 1.08883])
    ratio = xyz / white
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = np.where(ratio > epsilon, np.cbrt(ratio), (kappa * ratio + 16.0) / 116.0)
    lightness = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([lightness, a, b], axis=-1)


def compute(apply_lut, grid: np.ndarray, dmin: Sequence[float] | None = None,
            dmax: Sequence[float] | None = None) -> dict[str, Any]:
    """Run the sample sets through ``apply_lut`` and aggregate the response.

    ``apply_lut`` is injected rather than imported so the features are produced
    by exactly the renderer that produced the after-images the annotator sees.
    """
    colours, labels = band_samples()
    grays = gray_samples()
    stacked = np.concatenate([colours, grays], axis=0).reshape(-1, 1, 3)
    out = apply_lut(stacked, grid, dmin, dmax).reshape(-1, 3)
    colour_out = out[: colours.shape[0]]
    gray_out = out[colours.shape[0]:]

    per_sample: dict[str, list[tuple[float, float, float]]] = {name: [] for name, _ in BANDS}
    for index, (name, _hue, sat_in, lum_in) in enumerate(labels):
        red, green, blue = (float(v) for v in colour_out[index])
        hue_out, lum_out, sat_out = colorsys.rgb_to_hls(red, green, blue)
        hue_in = colorsys.rgb_to_hls(*(float(v) for v in colours[index]))[0]
        per_sample[name].append((
            _wrap180(hue_out * 360.0 - hue_in * 360.0),
            (sat_out - sat_in) / max(sat_in, 1e-6) * 100.0,
            (lum_out - lum_in) * 100.0,
        ))

    bands: dict[str, dict[str, float]] = {}
    for name, _centre in BANDS:
        rows = np.asarray(per_sample[name], dtype=np.float64)
        bands[name] = {
            "d_hue_deg": round(float(np.median(rows[:, 0])), 2),
            "d_sat_pct": round(float(np.median(rows[:, 1])), 1),
            "d_lum_pct": round(float(np.median(rows[:, 2])), 1),
            "d_hue_deg_iqr": round(float(np.subtract(*np.percentile(rows[:, 0], [75, 25]))), 2),
            "d_sat_pct_mean": round(float(rows[:, 1].mean()), 1),
            "d_lum_pct_mean": round(float(rows[:, 2].mean()), 1),
        }

    lab_in = _srgb_to_lab(grays)
    lab_out = _srgb_to_lab(gray_out)
    neutral = []
    for index, level in enumerate(GRAY_LEVELS):
        neutral.append({
            "in": round(level, 2),
            "L_in": round(float(lab_in[index, 0]), 1),
            "L_out": round(float(lab_out[index, 0]), 1),
            "a_out": round(float(lab_out[index, 1]), 2),
            "b_out": round(float(lab_out[index, 2]), 2),
        })
    span_in = float(lab_in[-1, 0] - lab_in[0, 0])
    span_out = float(lab_out[-1, 0] - lab_out[0, 0])
    mid = lab_out[len(GRAY_LEVELS) // 2]
    summary = {
        "contrast_ratio": round(span_out / span_in, 3) if abs(span_in) > 1e-6 else None,
        "mid_gray_a": round(float(mid[1]), 2),
        "mid_gray_b": round(float(mid[2]), 2),
        "mid_gray_dL": round(float(mid[0] - lab_in[len(GRAY_LEVELS) // 2, 0]), 1),
        "shadow_dL": round(float(lab_out[0, 0] - lab_in[0, 0]), 1),
        "highlight_dL": round(float(lab_out[-1, 0] - lab_in[-1, 0]), 1),
        "sat_pct_mean": round(float(np.mean([b["d_sat_pct"] for b in bands.values()])), 1),
        "hue_rot_abs_max": round(float(max(abs(b["d_hue_deg"]) for b in bands.values())), 2),
    }
    return {
        "spec_rev": SPEC_REV,
        "grid_size": int(grid.shape[0]),
        "bands": bands,
        "neutral_ramp": neutral,
        "summary": summary,
    }


def render_table(features: Mapping[str, Any]) -> str:
    """The same numbers as a compact Chinese table for the prompt."""
    bands = features["bands"]
    lines = [
        "【8 色相带响应】(程序从 LUT 网格直接算出, 不可推翻; ΔSat 为相对变化, ΔLum 为 HSL 明度点数)",
        "色带   ΔHue      ΔSat      ΔLum",
    ]
    for name, _centre in BANDS:
        row = bands[name]
        # CJK glyphs are two columns wide, so pad by display width, not len().
        label = name + " " * (6 - 2 * len(name))
        lines.append(
            f"{label}{row['d_hue_deg']:+7.1f}°  {row['d_sat_pct']:+7.1f}%  {row['d_lum_pct']:+7.1f}"
        )
    lines.append("")
    lines.append("【中性灰响应】(色温/色罩只看这一段: a*>0 品红 / <0 绿, b*>0 暖 / <0 冷)")
    lines.append("灰阶  L*in → L*out   a*      b*")
    for row in features["neutral_ramp"]:
        lines.append(
            f"{row['in']:.2f}  {row['L_in']:5.1f} → {row['L_out']:5.1f}   "
            f"{row['a_out']:+6.2f}  {row['b_out']:+6.2f}"
        )
    summary = features["summary"]
    contrast = "n/a" if summary["contrast_ratio"] is None else f"{summary['contrast_ratio']:.3f}"
    lines.append("")
    lines.append(
        f"【聚合】对比(L* 跨度比)={contrast}  暗部ΔL*={summary['shadow_dL']:+.1f}  "
        f"高光ΔL*={summary['highlight_dL']:+.1f}  中灰色罩 a*={summary['mid_gray_a']:+.2f} "
        f"b*={summary['mid_gray_b']:+.2f}  八带平均ΔSat={summary['sat_pct_mean']:+.1f}%"
    )
    return "\n".join(lines)


__all__ = ["BANDS", "SPEC_REV", "compute", "render_table", "band_samples", "gray_samples"]
