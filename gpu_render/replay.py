"""preset 本地回放器：XMP/lrtemplate → 按 LR 内部管线顺序调用标定算子 → 成片。

用途：端到端保真测试（对比 LR 农场 GT）与效率 benchmark。
- 标量算子（旧 OPS 达标者）：fits value_map 插值 + post-LUT 在相邻扫描值间线性混合。
- 家族算子（ops_v2 REGISTRY）：直接以 preset 的 crs 参数作 ctx["attrs"] 调用。
- 未覆盖键：逐 preset 记录（coverage 审计）。

monetgpt_sam3 env 运行（cwd=/home/bc/VeraRetouch）：
  python -m gpu_render.replay --preset x.xmp --image p.jpg --out out.jpg
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # VeraRetouch 根（gpu_render 包父目录）

from gpu_render.sweeps import CALIB_ROOT, OPS
from gpu_render.local_apply import load_fit, FITS_DIR

# ---------------------------------------------------------------------------
# preset 解析
# ---------------------------------------------------------------------------
_ATTR_RE = re.compile(r'crs:([A-Za-z0-9]+)="([^"]*)"')
_CURVE_KEYS = ("ToneCurvePV2012", "ToneCurvePV2012Red", "ToneCurvePV2012Green", "ToneCurvePV2012Blue")


def parse_preset(path: str, fmt: str) -> dict:
    """→ {"attrs": {crs_key: str_val}, "curves": {curve_key: [(x,y),...]}}"""
    txt = open(path, encoding="utf-8", errors="ignore").read()
    if fmt == "xmp" or path.endswith(".xmp"):
        attrs = dict(_ATTR_RE.findall(txt))
        curves = {}
        try:
            ns = {"rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                  "crs": "http://ns.adobe.com/camera-raw-settings/1.0/"}
            root = ET.fromstring(txt)
            for ck in _CURVE_KEYS:
                el = root.find(f".//crs:{ck}", ns)
                if el is not None:
                    pts = []
                    for li in el.findall(".//rdf:li", ns):
                        if li.text and "," in li.text:
                            x, y = li.text.split(",")
                            pts.append((float(x), float(y)))
                    if pts:
                        curves[ck] = pts
        except ET.ParseError:
            pass
        from gpu_render.local_replay import parse_locals
        return {"attrs": attrs, "curves": curves, "locals": parse_locals(txt)}
    # lrtemplate：用 LR 农场同款转换器，保证语义一致
    sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "lrc_scripts", "utils"))
    from lua_converter import LuaConverter
    i = txt.find("{")
    obj = LuaConverter.from_lua(txt[i:])
    settings = (obj.get("value") or {}).get("settings") if isinstance(obj, dict) else {}
    settings = settings or {}
    attrs, curves = {}, {}
    for k, v in settings.items():
        if k in _CURVE_KEYS and isinstance(v, (list, dict)):
            vals = list(v.values()) if isinstance(v, dict) else v
            nums = [float(x) for x in vals]
            curves[k] = list(zip(nums[0::2], nums[1::2]))
        elif isinstance(v, (int, float, bool, str)):
            attrs[k] = str(v)
    return {"attrs": attrs, "curves": curves}


# ---------------------------------------------------------------------------
# 标量算子：fits 连续值应用（value_map 插值 + 相邻扫描值 LUT 混合）
# ---------------------------------------------------------------------------
def _interp_lut(pts_lo, pts_hi, w):
    if pts_lo is None:
        return pts_hi
    if pts_hi is None:
        return pts_lo
    xs = sorted({p[0] for p in pts_lo} | {p[0] for p in pts_hi})
    lx, ly = zip(*sorted(pts_lo)); hx, hy = zip(*sorted(pts_hi))
    return [[x, (1 - w) * np.interp(x, lx, ly) + w * np.interp(x, hx, hy)] for x in xs]


def _lut_at(fit_luts: dict, sweep_vals: list, v: float):
    """在扫描值网格上取/混合 LUT；|v| 小于最小扫描值时按比例衰减到恒等。"""
    if not fit_luts:
        return None
    from gpu_render.sweeps import fmt_value
    grid = sorted(sweep_vals)
    def get(g):
        return fit_luts.get(fmt_value(g)) or fit_luts.get(str(g))
    if v <= grid[0]:
        return get(grid[0])
    if v >= grid[-1]:
        return get(grid[-1])
    import bisect
    i = bisect.bisect_left(grid, v)
    lo, hi = grid[i - 1], grid[i]
    w = (v - lo) / (hi - lo) if hi > lo else 0.0
    return _interp_lut(get(lo), get(hi), w)


def apply_scalar_op(img: np.ndarray, op: str, lr_v: float, fit: dict, apply_cfg) -> np.ndarray:
    from gpu_render.local_apply import local_value
    out = apply_cfg({op: local_value(op, lr_v, fit)}, img)
    out = np.asarray(out, dtype=np.float32)
    vals = [float(v) for v in OPS[op]["values"]]
    pts = _lut_at(fit.get("post_luma_lut") or {}, vals, lr_v)
    if pts:
        xs, ys = zip(*sorted(pts))
        luma = out @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        out = np.clip(out + (np.interp(luma, xs, ys).astype(np.float32) - luma)[..., None], 0, 1)
    # 每通道 RGB LUT：结构是 {扫描值: {"r":[[x,y]...], "g":..., "b":...}}。通道曲线不做
    # 跨值插值（太复杂、收益小），取扫描范围内最近的扫描值。
    rgb_luts = fit.get("post_rgb_lut") or {}
    if rgb_luts and vals:
        from gpu_render.sweeps import fmt_value
        nearest = min(vals, key=lambda g: abs(g - lr_v))
        chl = (rgb_luts.get(fmt_value(nearest))
               or rgb_luts.get(str(int(nearest)) if float(nearest).is_integer() else str(nearest)))
        if chl and abs(nearest - lr_v) <= (max(vals) - min(vals)):
            o = out.copy()
            for i, ch in enumerate(("r", "g", "b")):
                p = chl.get(ch)
                if p:
                    xs, ys = zip(*sorted(p))
                    o[..., i] = np.interp(out[..., i], xs, ys).astype(np.float32)
            out = np.clip(o, 0, 1)
    return out


# ---------------------------------------------------------------------------
# 管线：LR 内部顺序 → (阶段名, 消耗的键集合, 执行器)
# ---------------------------------------------------------------------------
_HSL_COLORS8 = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_SCALAR_MAP = {   # crs key -> 旧 OPS 标量算子
    "Exposure2012": ("Exposure", 1.0), "Contrast2012": ("Contrast", 1.0),
    "Highlights2012": ("Highlights", 1.0), "Whites2012": ("Whites", 1.0),
    "Blacks2012": ("Blacks", 1.0), "Texture": ("Texture", 1.0),
    "Clarity2012": ("Clarity", 1.0), "Vibrance": ("Vibrance", 1.0),
    "Saturation": ("Saturation", 1.0),
    "IncrementalTemperature": ("Temperature", 1.0), "IncrementalTint": ("Tint", 1.0),
    "Sharpness": ("Sharpness", 1.0), "LuminanceSmoothing": ("LuminanceNoiseReduction", 1.0),
    "Shadows2012": ("Shadows", 1.0),
}
_SKIP_PREFIXES = (   # 明确不在覆盖范围（几何/元数据/相机 profile/局部）——审计时归类 out_of_scope
    "Crop", "Perspective", "LensProfile", "LensManual", "CameraProfile", "Look",
    "Defringe", "ChromaticAberration", "AutoLateralCA", "Orientation",
    "ProcessVersion", "Version", "PresetType", "HasSettings", "WhiteBalance",
    "Enable", "UUID", "SupportsAmount", "SupportsColor", "SupportsMono",
    "SupportsHigh", "SupportsNormal", "SupportsOutput", "SupportsScene",
    "Cluster", "ShortName", "SortName", "Group", "Description", "Copyright",
    "ContactInfo", "CameraModelRestriction", "ToneCurveName", "ConvertToGrayscale",
    "Upright",   # 实测无像素效果（GT 与 identity 几何对齐，corner NCC≈0.99，见 diag 2026-07-05）
    "GrainSeed", "OverrideLook", "RequiresRGBTables", "Name", "Amount",
    "ParametricShadowSplit", "ParametricMidtoneSplit", "ParametricHighlightSplit",
    "SharpenRadius", "SharpenDetail", "SharpenEdgeMasking",   # 与 Sharpness 同阶段消耗
    "LuminanceNoiseReductionDetail", "LuminanceNoiseReductionContrast",
    "ColorNoiseReduction", "GrainSize", "GrainFrequency",     # 同族键在家族阶段消耗
    "SplitToning", "ColorGrade", "PostCropVignette", "VignetteAmount", "VignetteMidpoint",
    "GrayMixer", "HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment",
    "RedHue", "RedSaturation", "GreenHue", "GreenSaturation", "BlueHue", "BlueSaturation",
    "ShadowTint", "Temperature", "Tint", "Dehaze", "GrainAmount", "Exposure", "Contrast",
    "Highlights", "Shadows", "Whites", "Blacks", "Clarity", "Brightness",
)


def _f(attrs, k, default=0.0):
    try:
        return float(str(attrs.get(k, default)).lstrip("+"))
    except (TypeError, ValueError):
        return default


STAGES = ("wb", "tone", "pcurve", "curve", "localcon", "calib",
          "hslbw", "splittone", "sat", "detail", "grain", "vignette")


def replay(img: np.ndarray, preset: dict, registry: dict, apply_cfg,
           fits_dir: str = FITS_DIR, enable=None, capture=None) -> tuple:
    """按 LR 管线顺序回放。返回 (out_img, consumed_keys)。img float32 [0,1] RGB。
    enable=None 全开；否则只跑 enable 集合里的阶段（消融/诊断用）。
    capture 传 dict 时，每个阶段结束后存 out.copy()（诊断/残差训练用）。"""
    attrs, curves = preset["attrs"], preset["curves"]
    consumed: set = set()
    out = img

    def _on(stage):
        return enable is None or stage in enable

    def _snap(stage):
        if capture is not None:
            capture[stage] = out.copy()

    def reg(op, ctx_attrs, label="replay"):
        nonlocal out
        fn = registry.get(op)
        if fn is None:
            return False
        out = np.clip(np.asarray(
            fn(out.copy(), {"label": label, "attrs": ctx_attrs, "elements": "", "fit": load_fit(op, fits_dir)}),
            dtype=np.float32), 0, 1)
        return True

    def scalar(op, lr_v):
        nonlocal out
        # ops_v2 重写版优先（Temperature/Shadows/Sharpness/Clarity/Vibrance/Vignette/HSL蓝）
        if reg(op, {OPS[op]["crs"]: lr_v}, label=f"{lr_v:+g}"):
            return
        out = apply_scalar_op(out, op, lr_v, load_fit(op, fits_dir), apply_cfg)

    # 1. 白平衡
    if _on("wb"):
        if _f(attrs, "IncrementalTemperature"):
            scalar("Temperature", _f(attrs, "IncrementalTemperature")); consumed.add("IncrementalTemperature")
        if _f(attrs, "IncrementalTint"):
            scalar("Tint", _f(attrs, "IncrementalTint")); consumed.add("IncrementalTint")
        if attrs.get("Temperature") not in (None, "", "0") or attrs.get("Tint") not in (None, "", "0"):
            if reg("AbsTemperature", {"Temperature": attrs.get("Temperature", ""),
                                      "Tint": attrs.get("Tint", ""),
                                      "WhiteBalance": attrs.get("WhiteBalance", "Custom")}):
                consumed.update(k for k in ("Temperature", "Tint") if k in attrs)
    _snap("wb")
    # 2. 基础 tone
    if _on("tone"):
        for key in ("Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
                    "Whites2012", "Blacks2012"):
            v = _f(attrs, key)
            if v:
                scalar(_SCALAR_MAP[key][0], v); consumed.add(key)
    _snap("tone")
    # 3. 参数曲线
    if _on("pcurve"):
        for pk in ("ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"):
            v = _f(attrs, pk)
            if v and reg(pk, {pk: v}):
                consumed.add(pk)
    _snap("pcurve")
    # 4. 点曲线
    if _on("curve") and curves:
        els = "".join(
            f"<crs:{ck}><rdf:Seq>" + "".join(f"<rdf:li>{int(x)}, {int(y)}</rdf:li>" for x, y in pts)
            + f"</rdf:Seq></crs:{ck}>" for ck, pts in curves.items())
        fn = registry.get("ToneCurve")
        if fn is not None:
            out = np.clip(np.asarray(fn(out.copy(), {"label": "replay", "attrs": {},
                                                     "elements": els, "fit": load_fit("ToneCurve", fits_dir)}),
                                     dtype=np.float32), 0, 1)
            consumed.update(curves)
    _snap("curve")
    # 5. 局部对比/雾
    if _on("localcon"):
        for key, op in (("Clarity2012", "Clarity"), ("Texture", "Texture")):
            v = _f(attrs, key)
            if v:
                scalar(op, v); consumed.add(key)
        if _f(attrs, "Dehaze") and reg("Dehaze", {"Dehaze": _f(attrs, "Dehaze")}):
            consumed.add("Dehaze")
    _snap("localcon")
    # 6. 校准面板
    if _on("calib"):
        calib = {k: _f(attrs, k) for k in ("RedHue", "RedSaturation", "GreenHue", "GreenSaturation",
                                           "BlueHue", "BlueSaturation", "ShadowTint") if _f(attrs, k)}
        for k, v in calib.items():
            if reg(f"Calib{k}", {k: v}):
                consumed.add(k)
    _snap("calib")
    # 7. 黑白 or HSL
    if _on("hslbw"):
        if str(attrs.get("ConvertToGrayscale", "")).lower() in ("true", "1"):
            gm = {k: _f(attrs, k) for k in attrs if k.startswith("GrayMixer") and _f(attrs, k)}
            if reg("BlackWhite", {"ConvertToGrayscale": "True", **{k: str(int(v)) for k, v in gm.items()}}):
                consumed.add("ConvertToGrayscale"); consumed.update(gm)
        else:
            for pfx in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
                for c in _HSL_COLORS8:
                    k = f"{pfx}{c}"
                    v = _f(attrs, k)
                    if not v:
                        continue
                    if reg(k, {k: v}):
                        consumed.add(k)
                    elif k in OPS:
                        out = apply_scalar_op(out, k, v, load_fit(k, fits_dir), apply_cfg); consumed.add(k)
    _snap("hslbw")
    # 8. 分离色调 / ColorGrade（饱和度或 Lum 有非零值才触发；Hue 单独非零无效果）
    if _on("splittone") and (
            any(_f(attrs, k) for k in attrs if "Saturation" in k and k.startswith(("SplitToning", "ColorGrade")))
            or any(_f(attrs, k) for k in attrs if k.startswith("ColorGrade") and k.endswith("Lum"))):
        ctx = {k: attrs[k] for k in attrs if k.startswith(("SplitToning", "ColorGrade"))}
        for op in ("SplitToneShadow", "SplitToneHighlight", "SplitToneBalance",
                   "ColorGradeMid", "ColorGradeGlobal", "ColorGradeLum", "ColorGradeBlending"):
            if reg(op, ctx):
                consumed.update(ctx)
                break     # 统一核心：一次调用消费全部分离色调键（约定见 colorgrade 模块）
    _snap("splittone")
    # 9. 饱和族
    if _on("sat"):
        for key in ("Vibrance", "Saturation"):
            v = _f(attrs, key)
            if v:
                scalar(key, v); consumed.add(key)
    _snap("sat")
    # 10. 细节
    if _on("detail"):
        if _f(attrs, "Sharpness"):
            sp = {k: attrs[k] for k in ("Sharpness", "SharpenRadius", "SharpenDetail",
                                        "SharpenEdgeMasking") if k in attrs}
            if reg("Sharpness", sp):
                consumed.update(sp)
            else:
                scalar("Sharpness", _f(attrs, "Sharpness")); consumed.add("Sharpness")
        if _f(attrs, "LuminanceSmoothing"):
            scalar("LuminanceNoiseReduction", _f(attrs, "LuminanceSmoothing"))
            consumed.add("LuminanceSmoothing")
        if _f(attrs, "ColorNoiseReduction") and reg("ColorNR", {"ColorNoiseReduction": _f(attrs, "ColorNoiseReduction")}):
            consumed.add("ColorNoiseReduction")
    _snap("detail")
    # 11. 颗粒
    if _on("grain") and _f(attrs, "GrainAmount") and reg("Grain", {
            "GrainAmount": _f(attrs, "GrainAmount"),
            "GrainSize": _f(attrs, "GrainSize", 25), "GrainFrequency": _f(attrs, "GrainFrequency", 50)}):
        consumed.update(k for k in ("GrainAmount", "GrainSize", "GrainFrequency") if k in attrs)
    _snap("grain")
    # 12. 暗角
    if _on("vignette"):
        vig = {k: attrs[k] for k in attrs if k.startswith("PostCropVignette") or k == "VignetteAmount"}
        if any(_f(attrs, k) for k in ("PostCropVignetteAmount", "VignetteAmount")):
            if reg("Vignette", vig):
                consumed.update(vig)
    _snap("vignette")
    # 13. 局部修正（MaskGroupBasedCorrections / 语义 α）
    if _on("local") and preset.get("locals"):
        from gpu_render.local_replay import apply_locals
        out = apply_locals(out, preset["locals"], registry, apply_cfg, fits_dir)
    _snap("local")

    leftover = [k for k, v in attrs.items()
                if k not in consumed and _f(attrs, k)
                and not any(k.startswith(p) for p in _SKIP_PREFIXES)]
    return out, sorted(leftover)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset"); ap.add_argument("--image"); ap.add_argument("--out")
    ap.add_argument("--fmt", default="xmp")
    a = ap.parse_args()
    from gpu_render.image_ops.non_gimp_ops import apply_non_gimp_config, read_image, write_image
    from gpu_render import ops_v2

    def apply_cfg(cfg, img):
        return apply_non_gimp_config(cfg, img, 255.0)

    img, norm = read_image(a.image)
    pre = parse_preset(a.preset, a.fmt)
    t0 = time.time()
    out, leftover = replay(np.asarray(img, dtype=np.float32), pre, ops_v2.REGISTRY, apply_cfg)
    print(f"replayed in {time.time()-t0:.2f}s; uncovered keys: {leftover}")
    write_image(out, norm, a.out)


if __name__ == "__main__":
    main()
