"""
Tone-curve utilities for Photoshop/Lightroom-style point curves.

The runtime representation is a dict with optional per-channel curves:
{
  "master": [[x, y], ...],   # required, 2-8 points
  "r": [[x, y], ...] | None,
  "g": [[x, y], ...] | None,
  "b": [[x, y], ...] | None
}

All coordinates use the 8-bit domain [0, 255] so configs remain portable.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np


_CURVE_KEYS = ("master", "r", "g", "b")
_DEFAULT_LINEAR_POINTS = [[0, 0], [255, 255]]


def _is_point_like(item: Any) -> bool:
    return isinstance(item, (list, tuple)) and len(item) == 2


def _to_point_list(points: Any) -> list[list[int]]:
    """Normalize arbitrary point input into sorted [x, y] control points."""
    if not isinstance(points, Iterable):
        return [list(p) for p in _DEFAULT_LINEAR_POINTS]

    pairs: list[tuple[float, float]] = []
    for item in points:
        if not _is_point_like(item):
            continue
        try:
            x = float(item[0])
            y = float(item[1])
        except (TypeError, ValueError):
            continue
        x = float(np.clip(x, 0.0, 255.0))
        y = float(np.clip(y, 0.0, 255.0))
        pairs.append((x, y))

    if not pairs:
        return [list(p) for p in _DEFAULT_LINEAR_POINTS]

    # Keep only the last y-value for duplicated x, then sort.
    dedup: dict[int, int] = {}
    for x, y in pairs:
        dedup[int(round(x))] = int(round(y))
    points_i = sorted((x, int(np.clip(y, 0, 255))) for x, y in dedup.items())

    if len(points_i) < 2:
        return [list(p) for p in _DEFAULT_LINEAR_POINTS]

    # Ensure endpoints exist; preserve user trend by extending endpoint y-values.
    if points_i[0][0] > 0:
        points_i.insert(0, (0, points_i[0][1]))
    if points_i[-1][0] < 255:
        points_i.append((255, points_i[-1][1]))

    # Keep runtime specs compact and bounded.
    max_points = 8
    if len(points_i) > max_points:
        keep_idx = np.linspace(0, len(points_i) - 1, num=max_points, dtype=int)
        points_i = [points_i[i] for i in keep_idx]

    return [[int(x), int(y)] for x, y in points_i]


def sanitize_tone_curve_spec(curve_spec: Any) -> dict[str, list[list[int]] | None]:
    """
    Convert raw JSON/model output into a canonical tone-curve spec.

    Accepted inputs:
    - {"ToneCurve": {...}}
    - {"master": [...], "r": [...], ...}
    - {"control_points": [...]}
    - [[x,y], ...]   (treated as master curve)
    """
    raw = curve_spec
    if isinstance(raw, dict) and "ToneCurve" in raw:
        raw = raw["ToneCurve"]

    # Treat a bare point-list as master.
    if isinstance(raw, (list, tuple)):
        raw = {"master": raw}
    elif not isinstance(raw, dict):
        raw = {"master": _DEFAULT_LINEAR_POINTS}

    master_raw = raw.get("master")
    if master_raw is None:
        master_raw = raw.get("control_points", _DEFAULT_LINEAR_POINTS)

    normalized: dict[str, list[list[int]] | None] = {
        "master": _to_point_list(master_raw),
        "r": None,
        "g": None,
        "b": None,
    }

    channel_alias = {"r": "red", "g": "green", "b": "blue"}
    for key in ("r", "g", "b"):
        raw_points = raw.get(key)
        if raw_points is None:
            raw_points = raw.get(channel_alias[key])
        if raw_points in (None, False):
            continue
        normalized[key] = _to_point_list(raw_points)

    return normalized


def is_tone_curve_spec(value: Any) -> bool:
    """Return True if a value looks like a tone-curve configuration."""
    if isinstance(value, (list, tuple)):
        return True
    if not isinstance(value, dict):
        return False
    if "ToneCurve" in value:
        return True
    return any(k in value for k in _CURVE_KEYS) or any(
        k in value for k in ("control_points", "red", "green", "blue")
    )


def build_curve_lut(
    points: Any,
    size: int = 256,
    method: str = "pchip",
) -> np.ndarray:
    """Build an 8-bit LUT from control points."""
    norm_points = _to_point_list(points)
    xs = np.asarray([p[0] for p in norm_points], dtype=np.float32)
    ys = np.asarray([p[1] for p in norm_points], dtype=np.float32)
    xq = np.linspace(0.0, 255.0, num=int(size), dtype=np.float32)

    lut_vals = None
    if str(method).strip().lower() == "pchip":
        try:
            from scipy.interpolate import PchipInterpolator

            interp = PchipInterpolator(xs, ys, extrapolate=True)
            lut_vals = interp(xq)
        except Exception:
            lut_vals = None

    if lut_vals is None:
        lut_vals = np.interp(xq, xs, ys)

    lut_vals = np.clip(np.rint(lut_vals), 0.0, 255.0).astype(np.uint8)
    return lut_vals


def _apply_lut_float01(channel: np.ndarray, lut_u8: np.ndarray) -> np.ndarray:
    """Apply an 8-bit LUT to a float [0,1] channel via interpolation."""
    x = np.linspace(0.0, 1.0, num=lut_u8.shape[0], dtype=np.float32)
    y = lut_u8.astype(np.float32) / 255.0
    out = np.interp(np.clip(channel, 0.0, 1.0), x, y)
    return out.astype(np.float32, copy=False)


def build_blacks_lut(intensity: float) -> np.ndarray:
    """Build the 1D LUT used by `adjust_blacks` for HISTOGRAM-VALUE semantics."""
    intensity = float(np.clip(intensity, -1.0, 1.0))
    y_pt = 30
    x_pt = y_pt + ((abs(intensity) + 1) ** 4) * 6
    if intensity > 0:
        ctrl_x, ctrl_y = y_pt, x_pt
    else:
        ctrl_x, ctrl_y = x_pt, y_pt
    return build_curve_lut(
        [[0, 0], [int(ctrl_x), int(ctrl_y)], [255, 255]],
        size=256,
        method="pchip",
    )



# ---------------------------------------------------------------------------
# LR ToneCurvePV2012 v2（Melissa RGB 域 PCHIP 点曲线；合入自 tools/lr_calib/ops_v2）
# ---------------------------------------------------------------------------
_LI_RE = re.compile(r"<rdf:li>\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*</rdf:li>")


def parse_pv2012_curves(elements: str) -> dict:
    """XMP 子元素串 -> {"master": [(x,y)...]|None, "r":..., "g":..., "b":...}。

    形如 <crs:ToneCurvePV2012><rdf:Seq><rdf:li>x, y</rdf:li>...</rdf:Seq></crs:...>。
    主曲线标签是通道标签的前缀，正则以 '>' 收尾避免误匹配。
    """
    out = {"master": None, "r": None, "g": None, "b": None}
    if not elements:
        return out
    for key, name in (("master", "ToneCurvePV2012"),
                      ("r", "ToneCurvePV2012Red"),
                      ("g", "ToneCurvePV2012Green"),
                      ("b", "ToneCurvePV2012Blue")):
        m = re.search(rf"<crs:{name}>(.*?)</crs:{name}>", elements, re.S)
        if not m:
            continue
        pts = [(float(x), float(y)) for x, y in _LI_RE.findall(m.group(1))]
        if len(pts) >= 2:
            out[key] = sorted(pts)
    return out


# ---------------------------------------------------------------------------
# PCHIP 曲线 -> float LUT（无 uint8 取整）
# ---------------------------------------------------------------------------
def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson 单调三次样条节点导数（与 scipy PchipInterpolator 同式）。"""
    h = np.diff(x)
    d = np.diff(y) / h
    n = x.size
    m = np.zeros(n, dtype=np.float64)
    if n == 2:
        m[:] = d[0]
        return m
    d0, d1 = d[:-1], d[1:]
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        mk = (w1 + w2) / (w1 / d0 + w2 / d1)
    mk[(d0 == 0) | (d1 == 0) | (np.sign(d0) != np.sign(d1))] = 0.0
    m[1:-1] = mk

    def edge(h0, h1, dd0, dd1):
        t = ((2.0 * h0 + h1) * dd0 - h0 * dd1) / (h0 + h1)
        if np.sign(t) != np.sign(dd0):
            return 0.0
        if np.sign(dd0) != np.sign(dd1) and abs(t) > 3.0 * abs(dd0):
            return 3.0 * dd0
        return t

    m[0] = edge(h[0], h[1], d[0], d[1])
    m[-1] = edge(h[-1], h[-2], d[-1], d[-2])
    return m


def pchip_lut(points, size: int = 4096) -> np.ndarray:
    """显示域 0-255 控制点 -> float32 LUT（域/值均 [0,1]，size 个等距采样）。

    单调三次（PCHIP）插值；首/末控制点外水平延伸（LR 行为）。
    """
    dedup: dict[float, float] = {}
    for px, py in points:
        dedup[float(px)] = float(py)
    pts = sorted(dedup.items())
    if len(pts) < 2:
        return np.linspace(0.0, 1.0, size, dtype=np.float32)
    x = np.asarray([p[0] for p in pts], dtype=np.float64)
    y = np.asarray([p[1] for p in pts], dtype=np.float64)
    m = _pchip_slopes(x, y)
    xq = np.linspace(0.0, 255.0, size)
    idx = np.clip(np.searchsorted(x, xq, side="right") - 1, 0, x.size - 2)
    h = x[idx + 1] - x[idx]
    t = np.clip((xq - x[idx]) / h, 0.0, 1.0)   # clip: 区间外水平延伸
    t2 = t * t
    t3 = t2 * t
    vals = ((2 * t3 - 3 * t2 + 1) * y[idx] + (t3 - 2 * t2 + t) * h * m[idx]
            + (-2 * t3 + 3 * t2) * y[idx + 1] + (t3 - t2) * h * m[idx + 1])
    vals[xq <= x[0]] = y[0]
    vals[xq >= x[-1]] = y[-1]
    return np.clip(vals / 255.0, 0.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Melissa RGB（ProPhoto 基色 D50 + sRGB 传递函数）往返
# ---------------------------------------------------------------------------
# linear sRGB(D65) -> XYZ(D65) -> Bradford -> XYZ(D50) -> linear ProPhoto(D50)
_M_SRGB2PP = np.array([
    [0.5292841932, 0.3300827798, 0.1406331764],
    [0.0983734118, 0.8734577155, 0.0281688727],
    [0.0168718797, 0.1176535723, 0.8654745480]], dtype=np.float32)
_M_PP2SRGB = np.linalg.inv(_M_SRGB2PP.astype(np.float64)).astype(np.float32)


def _srgb_decode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92,
                    ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _srgb_encode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92,
                    1.055 * v ** (1.0 / 2.4) - 0.055).astype(np.float32)


def _apply_lut(channel: np.ndarray, lut: np.ndarray) -> np.ndarray:
    xs = np.linspace(0.0, 1.0, lut.size, dtype=np.float32)
    return np.interp(channel, xs, lut).astype(np.float32)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def apply_pv2012_tone_curve(img: np.ndarray, curves: dict) -> np.ndarray:
    """对 float32 [0,1] sRGB 编码 RGB 图应用 PV2012 点曲线（LR-faithful）。

    curves: {"master": pts|None, "r": pts|None, "g": pts|None, "b": pts|None}，
    pts 为显示域 0-255 控制点。主/通道曲线均在 Melissa RGB 内逐通道应用；
    主曲线在通道曲线之后（同空间 LUT 复合；两者同时出现的次序 GT 未覆盖）。
    """
    if not any(curves.get(k) for k in ("master", "r", "g", "b")):
        return np.asarray(img, dtype=np.float32)
    work = _srgb_encode(np.clip(_srgb_decode(np.asarray(img, dtype=np.float32))
                                @ _M_SRGB2PP.T, 0.0, 1.0))
    for i, key in enumerate(("r", "g", "b")):
        if curves.get(key):
            work[..., i] = _apply_lut(work[..., i], pchip_lut(curves[key]))
    if curves.get("master"):
        lut = pchip_lut(curves["master"])
        for i in range(3):
            work[..., i] = _apply_lut(work[..., i], lut)
    out = _srgb_encode(np.clip(_srgb_decode(work) @ _M_PP2SRGB.T, 0.0, 1.0))
    return out.astype(np.float32)


_LINEAR_MASTER_POINTS = [[0, 0], [255, 255]]


def apply_tone_curve(
    image: np.ndarray,
    curve_spec: Any,
    norm_factor: float | None = None,  # kept for call-site compatibility
) -> np.ndarray:
    """Apply master + optional per-channel tone curves to an RGB image.

    LR-faithful v2：曲线在 Melissa RGB（ProPhoto 基色 D50 + sRGB 传递函数）内
    逐通道作用（apply_pv2012_tone_curve），PCHIP float LUT、无 uint8 量化。
    旧实现（输出 sRGB 直接查曲线 + uint8 取整）已废弃。
    """
    _ = norm_factor  # tone-curve works in normalized float space
    if image is None:
        raise ValueError("image cannot be None")

    img = np.clip(image.astype(np.float32, copy=False), 0.0, 1.0)
    if img.ndim != 3 or img.shape[2] < 3:
        raise ValueError(f"expected HxWx3 image, got shape={img.shape}")

    spec = sanitize_tone_curve_spec(curve_spec)

    def _pts(points):
        if not points:
            return None
        pts = [(float(x), float(y)) for x, y in points]
        return pts if pts != [(0.0, 0.0), (255.0, 255.0)] else None

    curves = {
        "master": _pts(spec.get("master")),
        "r": _pts(spec.get("r")),
        "g": _pts(spec.get("g")),
        "b": _pts(spec.get("b")),
    }
    if not any(curves.values()):
        return img.astype(np.float32, copy=False)
    return apply_pv2012_tone_curve(img, curves)
