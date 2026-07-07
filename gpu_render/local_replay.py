"""Local edit（MaskGroupBasedCorrections / 语义 α）CPU golden 回放。

语义约定（与 dataset_build mask_synth Route 1 / subject_geom 一致）：
  - 一个 correction = 一组 Local* 参数 + 一个或多个 mask（几何并集，取 max）。
  - α = raster(masks) × CorrectionAmount；out = out·(1-α) + edited·α，
    edited = 对当前图按 LR 管线相对次序施加 Local*（tone → localcon → sat）。
  - Local* 数值语义 = 同名全局标定算子的 LR 值（LocalExposure2012 即 EV），
    CPU 走 ops_v2.REGISTRY / fits apply_scalar_op，与全局路径同源。
  - 几何光栅化与 dataset_build construct.mask_synth.cgt_raster 逐像素一致
    （local_parity.py 里有对拍断言）。
  - 语义 mask（SAM3 主体等）无法进 XMP：correction dict 直接带 "alpha"
    (H,W) float01 ndarray（subject_geom.semantic_alpha 的输出）。

不改动全局管线本体；replay.replay() 在 vignette 后调 apply_locals（stage "local"）。
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

# LR local 键 → (标定算子名, 施加次序)。次序对齐全局管线：tone → localcon → sat
LOCAL_OP_MAP = {
    "LocalExposure2012": ("Exposure", 0),
    "LocalContrast2012": ("Contrast", 1),
    "LocalHighlights2012": ("Highlights", 2),
    "LocalShadows2012": ("Shadows", 3),
    "LocalClarity2012": ("Clarity", 4),
    "LocalTexture": ("Texture", 5),
    "LocalSaturation": ("Saturation", 6),
}

_CRS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_RDF = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_GEOM_KEYS = ("Top", "Left", "Bottom", "Right", "Angle", "Midpoint", "Roundness",
              "Feather", "Flipped", "ZeroX", "ZeroY", "FullX", "FullY")


def _f(d, k, default=0.0):
    try:
        return float(d.get(k, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 几何 α 光栅（与 construct.mask_synth.cgt_raster 语义逐行一致）
# --------------------------------------------------------------------------- #
def raster_alpha(mask_type: str, geom: dict, h: int, w: int) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype("float32")
    x, y = xx / w, yy / h
    if mask_type == "circulargradient":
        cx = (_f(geom, "Left") + _f(geom, "Right")) / 2
        cy = (_f(geom, "Top") + _f(geom, "Bottom")) / 2
        rx = max(abs(_f(geom, "Right") - _f(geom, "Left")) / 2, 1e-3)
        ry = max(abs(_f(geom, "Bottom") - _f(geom, "Top")) / 2, 1e-3)
        ang = math.radians(_f(geom, "Angle"))
        xr = (x - cx) * math.cos(ang) + (y - cy) * math.sin(ang)
        yr = -(x - cx) * math.sin(ang) + (y - cy) * math.cos(ang)
        d = np.sqrt((xr / rx) ** 2 + (yr / ry) ** 2)
        feather = max(_f(geom, "Feather", 50) / 100.0, 0.05)
        # LR CircularGradient 默认校正椭圆外（0 内 1 外）；Flipped 再反转
        m = np.clip((d - 1.0) / feather + 0.5, 0, 1).astype("float32")
    else:
        zx, zy = _f(geom, "ZeroX"), _f(geom, "ZeroY")
        fx, fy = _f(geom, "FullX", 1), _f(geom, "FullY")
        dxv, dyv = fx - zx, fy - zy
        L2 = dxv * dxv + dyv * dyv + 1e-6
        m = np.clip(((x - zx) * dxv + (y - zy) * dyv) / L2, 0, 1).astype("float32")
    if str(geom.get("Flipped", "false")).lower().lstrip("+") == "true":
        m = 1.0 - m
    return m


def corr_alpha(corr: dict, h: int, w: int) -> np.ndarray:
    """一个 correction 的 α：语义 alpha 直取，几何 masks 取并集(max)，×Amount。"""
    if corr.get("alpha") is not None:
        a = np.asarray(corr["alpha"], np.float32)
        if a.shape != (h, w):
            import cv2
            a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        ms = [raster_alpha(m["mask_type"], m["geom"], h, w) for m in corr["masks"]]
        a = np.maximum.reduce(ms) if ms else np.zeros((h, w), np.float32)
    amount = float(corr.get("amount", 1.0))
    return np.clip(a * amount, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# XMP 解析
# --------------------------------------------------------------------------- #
def parse_locals(xmp_text: str) -> list:
    """MaskGroupBasedCorrections → [{params:{LocalKey:float}, amount, masks:[{mask_type,geom}]}]"""
    try:
        root = ET.fromstring(xmp_text)
    except ET.ParseError:
        return []
    ns = {"crs": _CRS, "rdf": _RDF}
    out = []
    for grp in root.iter(f"{{{_CRS}}}MaskGroupBasedCorrections"):
        for li in grp.findall(f"./rdf:Seq/rdf:li", ns):
            cd = li.find("./rdf:Description", ns)
            src = cd if cd is not None else li
            attrs = {k.split("}")[-1]: v for k, v in src.attrib.items()}
            params = {k: _f(attrs, k) for k in LOCAL_OP_MAP if _f(attrs, k)}
            if not params:
                continue
            masks = []
            for mli in src.iter(f"{{{_RDF}}}li"):
                md = mli.find("./rdf:Description", ns)
                msrc = md if md is not None else mli
                mattrs = {k.split("}")[-1]: v for k, v in msrc.attrib.items()}
                what = mattrs.get("What", "")
                if what == "Mask/CircularGradient":
                    mt = "circulargradient"
                elif what == "Mask/Gradient":
                    mt = "gradient"
                else:
                    continue
                masks.append({"mask_type": mt,
                              "geom": {k: mattrs[k] for k in _GEOM_KEYS if k in mattrs}})
            if masks:
                out.append({"params": params, "masks": masks,
                            "amount": _f(attrs, "CorrectionAmount", 1.0)})
    return out


# --------------------------------------------------------------------------- #
# 施加（CPU golden）
# --------------------------------------------------------------------------- #
def _scalar(img: np.ndarray, op: str, lr_v: float, registry: dict, apply_cfg,
            fits_dir: str) -> np.ndarray:
    """与 replay.scalar 同源：ops_v2 REGISTRY 优先，否则 fits apply_scalar_op。"""
    from gpu_render.local_apply import load_fit
    from gpu_render.replay import apply_scalar_op
    from gpu_render.sweeps import OPS

    fn = registry.get(op)
    if fn is not None:
        return np.clip(np.asarray(
            fn(img.copy(), {"label": f"{lr_v:+g}", "attrs": {OPS[op]["crs"]: lr_v},
                            "elements": "", "fit": load_fit(op, fits_dir)}),
            dtype=np.float32), 0, 1)
    return np.clip(np.asarray(
        apply_scalar_op(img, op, lr_v, load_fit(op, fits_dir), apply_cfg),
        dtype=np.float32), 0, 1)


def apply_locals(img: np.ndarray, corrections: list, registry: dict, apply_cfg,
                 fits_dir: str) -> np.ndarray:
    """逐 correction：edited = Local* 依次施加于当前图；out = lerp(out, edited, α)。"""
    out = np.asarray(img, np.float32)
    h, w = out.shape[:2]
    for corr in corrections:
        params = corr.get("params") or {}
        ops = sorted(((LOCAL_OP_MAP[k][1], LOCAL_OP_MAP[k][0], v)
                      for k, v in params.items() if k in LOCAL_OP_MAP and v),
                     key=lambda t: t[0])
        if not ops:
            continue
        alpha = corr_alpha(corr, h, w)
        if float(alpha.max()) <= 0.0:
            continue
        edited = out.copy()
        for _, op, v in ops:
            edited = _scalar(edited, op, v, registry, apply_cfg, fits_dir)
        a3 = alpha[..., None]
        out = np.clip(out * (1.0 - a3) + edited * a3, 0.0, 1.0).astype(np.float32)
    return out
