"""LR↔monetGPT 算子标定的扫描定义：每算子的 LR crs 参数、扫描值、基线映射。

设计：对每个算子做单参数扫描，经 LR 农场渲出 ground-truth；本地算子按
fits/<op>.json 的校正（value remap + 可选残差 LUT）复现，评测 ΔE00/SSIM。
达标判据（"95 分位效果"）：全扫描 × 探针上 ΔE00 均值 ≤ 2.0 且 p95 ≤ 4.0。

Dehaze 被排除：monetGPT 侧已弃用（non_gimp_ops.adjust_dehaze raises）——记录为能力缺口。
"""
from __future__ import annotations

CALIB_ROOT = "/home/bc/data/datasets/lr_calib"

_PCT6 = [-100, -60, -30, 30, 60, 100]

# op -> {crs: LR XMP 属性名, values: LR 侧扫描值, to_local(v): 基线 LR值->monetGPT config 值}
# monetGPT 全局 config 值域 [-100,100]（内部 /100 归一），见 _normalize_global_compat_intensity。
OPS: dict[str, dict] = {
    "Exposure":   {"crs": "Exposure2012", "values": [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
                   "to_local": lambda v: v * 20.0},        # ev -> [-100,100]（内部 ×5 stops）
    "Contrast":   {"crs": "Contrast2012", "values": _PCT6},
    "Highlights": {"crs": "Highlights2012", "values": _PCT6},
    "Shadows":    {"crs": "Shadows2012", "values": _PCT6},
    "Whites":     {"crs": "Whites2012", "values": _PCT6},
    "Blacks":     {"crs": "Blacks2012", "values": _PCT6},
    "Texture":    {"crs": "Texture", "values": _PCT6},
    "Clarity":    {"crs": "Clarity2012", "values": _PCT6},
    "Vibrance":   {"crs": "Vibrance", "values": _PCT6},
    "Saturation": {"crs": "Saturation", "values": _PCT6},
    # JPEG 源：LR 用增量白平衡（非 kelvin）。语义差异大，靠 remap 拟合。
    "Temperature": {"crs": "IncrementalTemperature", "values": _PCT6},
    "Tint":        {"crs": "IncrementalTint", "values": _PCT6},
    "Sharpness":   {"crs": "Sharpness", "values": [30, 60, 90, 120, 150]},
    "LuminanceNoiseReduction": {"crs": "LuminanceSmoothing", "values": [20, 40, 60, 80, 100]},
    "Vignette":    {"crs": "PostCropVignetteAmount", "values": [-80, -40, 40, 80]},
}
for _color in ("Red", "Yellow", "Green", "Aqua", "Blue", "Magenta"):
    for _pfx in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment"):
        OPS[f"{_pfx}{_color}"] = {"crs": f"{_pfx}{_color}", "values": [-100, -50, 50, 100]}

IDENTITY = "Identity"   # 空预设：量化 LR 导出管线本身的偏置，作为对比基线

# ---------------------------------------------------------------------------
# 第二批：preset 库普查后的全覆盖扫描（复合参数用 dict 扫描点，曲线用 elements）。
# 扫描点格式：标量（单 crs 属性）或 {"label": str, "attrs": {crs属性: 值},
# "elements": "<crs:...>...</crs:...>"（可选，rdf:Description 子元素）}。
# 这些 op 没有 monetGPT baseline —— 由 tools/lr_calib/ops_v2/ 注册实现。
# ---------------------------------------------------------------------------
def _pt(label: str, attrs: dict, elements: str = "") -> dict:
    return {"label": label, "attrs": attrs, "elements": elements}


def _curve_el(name: str, pts: list) -> str:
    li = "".join(f"<rdf:li>{x}, {y}</rdf:li>" for x, y in pts)
    return f"<crs:{name}><rdf:Seq>{li}</rdf:Seq></crs:{name}>"


_LIN = [(0, 0), (255, 255)]
NEW_OPS: dict[str, dict] = {
    # 参数曲线（region-based tone curve）
    **{f"Parametric{n}": {"crs": f"Parametric{n}", "values": [-100, -50, 50, 100]}
       for n in ("Shadows", "Darks", "Lights", "Highlights")},
    # 校准面板（channel primaries）。命名带 Calib 前缀避免与 HSL 混淆，crs 名在 attrs 里。
    **{f"Calib{n}": {"values": [_pt(f"{v:+04d}", {n: v}) for v in (-100, -50, 50, 100)]}
       for n in ("RedHue", "RedSaturation", "GreenHue", "GreenSaturation",
                 "BlueHue", "BlueSaturation", "ShadowTint")},
    # 分离色调
    "SplitToneShadow": {"values": [
        *[_pt(f"hue{h:03d}", {"SplitToningShadowHue": h, "SplitToningShadowSaturation": 60})
          for h in (30, 120, 215, 300)],
        _pt("sat030", {"SplitToningShadowHue": 215, "SplitToningShadowSaturation": 30}),
        _pt("sat100", {"SplitToningShadowHue": 215, "SplitToningShadowSaturation": 100}),
    ]},
    "SplitToneHighlight": {"values": [
        *[_pt(f"hue{h:03d}", {"SplitToningHighlightHue": h, "SplitToningHighlightSaturation": 60})
          for h in (30, 120, 215, 300)],
        _pt("sat030", {"SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 30}),
        _pt("sat100", {"SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 100}),
    ]},
    "SplitToneBalance": {"values": [
        _pt(f"{b:+04d}", {"SplitToningShadowHue": 215, "SplitToningShadowSaturation": 50,
                          "SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 50,
                          "SplitToningBalance": b}) for b in (-70, 0, 70)]},
    # ColorGrade（3-way + global + lum + blending；SplitToning 的超集）
    "ColorGradeMid": {"values": [
        _pt("hue045", {"ColorGradeMidtoneHue": 45, "ColorGradeMidtoneSat": 60}),
        _pt("hue215", {"ColorGradeMidtoneHue": 215, "ColorGradeMidtoneSat": 60}),
        _pt("sat100", {"ColorGradeMidtoneHue": 45, "ColorGradeMidtoneSat": 100}),
    ]},
    "ColorGradeGlobal": {"values": [
        _pt("hue045", {"ColorGradeGlobalHue": 45, "ColorGradeGlobalSat": 60}),
        _pt("hue215", {"ColorGradeGlobalHue": 215, "ColorGradeGlobalSat": 60}),
    ]},
    "ColorGradeLum": {"values": [
        _pt("sh-60", {"ColorGradeShadowLum": -60}), _pt("sh+60", {"ColorGradeShadowLum": 60}),
        _pt("hi-60", {"ColorGradeHighlightLum": -60}), _pt("hi+60", {"ColorGradeHighlightLum": 60}),
        _pt("mid-60", {"ColorGradeMidtoneLum": -60}), _pt("mid+60", {"ColorGradeMidtoneLum": 60}),
        _pt("glob-60", {"ColorGradeGlobalLum": -60}), _pt("glob+60", {"ColorGradeGlobalLum": 60}),
    ]},
    "ColorGradeBlending": {"values": [
        _pt(f"{b:03d}", {"ColorGradeBlending": b,
                         "SplitToningShadowHue": 215, "SplitToningShadowSaturation": 50,
                         "SplitToningHighlightHue": 45, "SplitToningHighlightSaturation": 50})
        for b in (0, 100)]},
    # 点曲线（PV2012 显示域）。主曲线 6 形状 + RGB 通道 3 组。
    "ToneCurve": {"values": [
        _pt("strong_s", {}, _curve_el("ToneCurvePV2012", [(0, 0), (64, 44), (128, 128), (192, 212), (255, 255)])),
        _pt("soft_s", {}, _curve_el("ToneCurvePV2012", [(0, 0), (64, 54), (192, 202), (255, 255)])),
        _pt("fade_black", {}, _curve_el("ToneCurvePV2012", [(0, 30), (128, 135), (255, 255)])),
        _pt("crush_high", {}, _curve_el("ToneCurvePV2012", [(0, 0), (128, 120), (255, 225)])),
        _pt("lift_shadow", {}, _curve_el("ToneCurvePV2012", [(0, 0), (48, 78), (128, 138), (255, 255)])),
        _pt("neg_contrast", {}, _curve_el("ToneCurvePV2012", [(0, 20), (64, 80), (192, 180), (255, 235)])),
        _pt("rgb_warm", {}, _curve_el("ToneCurvePV2012Red", [(0, 10), (128, 145), (255, 255)])
                          + _curve_el("ToneCurvePV2012Blue", [(0, 0), (128, 112), (255, 245)])),
        _pt("rgb_cool", {}, _curve_el("ToneCurvePV2012Red", [(0, 0), (128, 115), (255, 250)])
                          + _curve_el("ToneCurvePV2012Blue", [(0, 12), (128, 142), (255, 255)])),
        _pt("green_shift", {}, _curve_el("ToneCurvePV2012Green", [(0, 8), (128, 138), (255, 250)])),
    ]},
    # HSL Orange / Purple（8 色补齐）
    **{f"{pfx}{c}": {"values": [_pt(f"{v:+04d}", {f"{pfx}{c}": v}) for v in (-100, -50, 50, 100)]}
       for pfx in ("HueAdjustment", "SaturationAdjustment", "LuminanceAdjustment")
       for c in ("Orange", "Purple")},
    # 颗粒（统计校准：pixel-ΔE 无意义）
    "Grain": {"values": [_pt(f"a{a:03d}", {"GrainAmount": a, "GrainSize": 25, "GrainFrequency": 50})
                         for a in (25, 50, 80)]},
    # Dehaze（monetGPT 弃用后重生）
    "Dehaze": {"values": [-70, -40, 40, 70, 100], "crs": "Dehaze"},
    # 锐化三参数（amount 固定 80）
    "SharpenParams": {"values": [
        _pt("rad1.0", {"Sharpness": 80, "SharpenRadius": "+1.0"}),
        _pt("rad2.5", {"Sharpness": 80, "SharpenRadius": "+2.5"}),
        _pt("det005", {"Sharpness": 80, "SharpenDetail": 5}),
        _pt("det080", {"Sharpness": 80, "SharpenDetail": 80}),
        _pt("mask60", {"Sharpness": 80, "SharpenEdgeMasking": 60}),
        _pt("mask95", {"Sharpness": 80, "SharpenEdgeMasking": 95}),
    ]},
    # 彩噪（JPEG 上预计近 identity，实测定论）
    "ColorNR": {"values": [_pt("025", {"ColorNoiseReduction": 25}),
                           _pt("100", {"ColorNoiseReduction": 100})]},
    # 黑白 + GrayMixer
    "BlackWhite": {"values": [
        _pt("auto", {"ConvertToGrayscale": "True"}),
        _pt("mix_a", {"ConvertToGrayscale": "True", "GrayMixerRed": -40, "GrayMixerOrange": 40,
                      "GrayMixerBlue": -60}),
        _pt("mix_b", {"ConvertToGrayscale": "True", "GrayMixerGreen": 50, "GrayMixerAqua": 60,
                      "GrayMixerBlue": 40}),
    ]},
    # 绝对 kelvin 白平衡（1196 preset 带 raw 语义 Temperature/Tint；实测 LR 对 JPEG 的换算）
    "AbsTemperature": {"values": [
        _pt("k3800", {"WhiteBalance": "Custom", "Temperature": 3800}),
        _pt("k5500", {"WhiteBalance": "Custom", "Temperature": 5500}),
        _pt("k8000", {"WhiteBalance": "Custom", "Temperature": 8000}),
        _pt("tint-50", {"WhiteBalance": "Custom", "Tint": -50}),
        _pt("tint+50", {"WhiteBalance": "Custom", "Tint": 50}),
    ]},
    # 暗角扩展参数 + 镜头暗角
    "VignetteParams": {"values": [
        _pt("mid10", {"PostCropVignetteAmount": -70, "PostCropVignetteMidpoint": 10}),
        _pt("mid90", {"PostCropVignetteAmount": -70, "PostCropVignetteMidpoint": 90}),
        _pt("fea10", {"PostCropVignetteAmount": -70, "PostCropVignetteFeather": 10}),
        _pt("fea90", {"PostCropVignetteAmount": -70, "PostCropVignetteFeather": 90}),
        _pt("rnd-70", {"PostCropVignetteAmount": -70, "PostCropVignetteRoundness": -70}),
        _pt("rnd+70", {"PostCropVignetteAmount": -70, "PostCropVignetteRoundness": 70}),
        _pt("style2", {"PostCropVignetteAmount": -70, "PostCropVignetteStyle": 2}),
        _pt("lens-80", {"VignetteAmount": -80}),
        _pt("lens+80", {"VignetteAmount": 80}),
    ]},
    # 影响验证组（预计 ≈ identity → 文档化排除）
    "FringeProbe": {"values": [
        _pt("purple", {"DefringePurpleAmount": 20, "DefringePurpleHueLo": 30, "DefringePurpleHueHi": 70}),
        _pt("green", {"DefringeGreenAmount": 20, "DefringeGreenHueLo": 40, "DefringeGreenHueHi": 60}),
        _pt("autoca", {"AutoLateralCA": 1}),
        _pt("dist-50", {"LensManualDistortionAmount": -50}),
        _pt("dist+50", {"LensManualDistortionAmount": 50}),
    ]},
}


def sweep_points(op: str) -> list:
    """统一扫描点接口：[(label, attrs_dict, elements_str)]，兼容新旧两张表。"""
    spec = OPS.get(op) or NEW_OPS[op]
    out = []
    for v in spec["values"]:
        if isinstance(v, dict):
            out.append((v["label"], v["attrs"], v.get("elements", "")))
        else:
            crs = spec["crs"]
            sval = f"{v:+.2f}" if isinstance(v, float) else f"{v:+d}"
            if crs in ("Sharpness", "LuminanceSmoothing"):
                sval = str(int(v))
            out.append((fmt_value(v), {crs: sval}, ""))
    return out


ALL_OPS = {**OPS, **NEW_OPS}


def fmt_value(v) -> str:
    """文件名安全的值标签：+060 / -100 / +1.50。"""
    if isinstance(v, float) and not v.is_integer():
        return f"{v:+.2f}"
    return f"{int(v):+04d}"


def xmp_from(attrs: dict, elements: str = "") -> str:
    """crs 属性 dict（+ 可选 rdf:Description 子元素串）→ 最小合法 develop-preset XMP。"""
    parts = ['crs:PresetType="Normal"', 'crs:ProcessVersion="15.4"']
    for k, v in attrs.items():
        parts.append(f'crs:{k}="{v}"')
    head = ('<x:xmpmeta xmlns:x="adobe:ns:meta/">'
            '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
            '<rdf:Description rdf:about="" '
            'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
            + " ".join(parts))
    if elements:
        return head + ">" + elements + "</rdf:Description></rdf:RDF></x:xmpmeta>"
    return head + "/></rdf:RDF></x:xmpmeta>"


def xmp_for(op: str | None, value=None) -> str:
    if op is None:
        return xmp_from({})
    crs = OPS[op]["crs"]
    sval = f"{value:+.2f}" if isinstance(value, float) else f"{value:+d}"
    if crs in ("Sharpness", "LuminanceSmoothing"):
        sval = str(int(value))                     # 无符号参数
    return xmp_from({crs: sval})


def baseline_local_config(op: str, lr_value) -> dict:
    """LR 扫描值 -> monetGPT apply_non_gimp_config 的基线 config（未校正）。"""
    spec = OPS[op]
    to_local = spec.get("to_local", lambda v: float(v))
    return {op: to_local(lr_value)}
