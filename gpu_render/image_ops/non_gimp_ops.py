from PIL import Image
import numpy as np
from skimage import color
import cv2
import json
import os
import warnings
import zlib
from scipy.ndimage import gaussian_filter, median_filter
from functools import lru_cache
import yaml
import tifffile
from skimage.transform import resize
from . import image_dehazer
# [gpu_render vendor 注] 原 `from shared.repo_config import read_config` 已裁剪：
# read_config 仅被 execute_non_gimp_pipeline（monetGPT 的文件式执行入口）使用，
# 不在 preset 渲染路径上；该入口在包内已 stub 为 NotImplementedError。


DEFAULT_NON_GIMP_RUNTIME_CONFIG = {
    "non_gimp_backend": "torch",
    "torch_device": "cuda",
    "torch_compile": False,
    "torch_compile_mode": "reduce-overhead",
    "torch_amp": False,
    "torch_amp_dtype": "float16",
    "torch_channels_last": False,
}
_RGB_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
_TINT_DEFAULT_BASE_KELVIN = 6504.0

_GEGL_COLOR_TEMPERATURE_RGB_R55 = np.asarray(
    [
        [
            6.9389923563552169e-01,
            2.7719388100974670e03,
            2.0999316761104289e07,
            -4.8889434162208414e09,
            -1.1899785506796783e07,
            -4.7418427686099203e04,
            1.0,
            3.5434394338546258e03,
            -5.6159353379127791e05,
            2.7369467137870544e08,
            1.6295814912940913e08,
            4.3975072422421846e05,
        ],
        [
            9.5417426141210926e-01,
            2.2041043287098860e03,
            -3.0142332673634286e06,
            -3.5111986367681120e03,
            -5.7030969525354260e00,
            6.1810926909962016e-01,
            1.0,
            1.3728609973644000e03,
            1.3099184987576159e06,
            -2.1757404458816318e03,
            -2.3892456292510311e00,
            8.1079012401293249e-01,
        ],
        [
            -7.1151622540856201e10,
            3.3728185802339764e16,
            -7.9396187338868539e19,
            2.9699115135330123e22,
            -9.7520399221734228e22,
            -2.9250107732225114e20,
            1.0,
            1.3888666482167408e16,
            2.3899765140914549e19,
            1.4583606312383295e23,
            1.9766018324502894e22,
            2.9395068478016189e18,
        ],
    ],
    dtype=np.float64,
)


def resolve_non_gimp_runtime_config(raw_cfg=None):
    cfg = dict(DEFAULT_NON_GIMP_RUNTIME_CONFIG)
    # Env override (below explicit runtime_config, above the built-in default):
    # lets CPU-parallel jobs (calibration / fidelity benchmark) force the numpy
    # backend and pin the device without touching production torch defaults.
    _env_backend = os.environ.get("MONETGPT_NON_GIMP_BACKEND")
    if _env_backend:
        cfg["non_gimp_backend"] = _env_backend.strip().lower()
    _env_device = os.environ.get("MONETGPT_TORCH_DEVICE")
    if _env_device:
        cfg["torch_device"] = _env_device.strip().lower()
    if isinstance(raw_cfg, dict):
        if isinstance(raw_cfg.get("processing"), dict):
            raw_cfg = raw_cfg.get("processing", {})
        backend = raw_cfg.get("non_gimp_backend", cfg["non_gimp_backend"])
        device = raw_cfg.get("torch_device", cfg["torch_device"])
        compile_flag = raw_cfg.get("torch_compile", cfg["torch_compile"])
        compile_mode = raw_cfg.get("torch_compile_mode", cfg["torch_compile_mode"])
        amp_flag = raw_cfg.get("torch_amp", cfg["torch_amp"])
        amp_dtype = raw_cfg.get("torch_amp_dtype", cfg["torch_amp_dtype"])
        channels_last_flag = raw_cfg.get("torch_channels_last", cfg["torch_channels_last"])
        cfg["non_gimp_backend"] = str(backend or cfg["non_gimp_backend"]).strip().lower()
        cfg["torch_device"] = str(device or cfg["torch_device"]).strip().lower()
        cfg["torch_compile"] = bool(compile_flag)
        cfg["torch_compile_mode"] = str(compile_mode or cfg["torch_compile_mode"]).strip() or cfg["torch_compile_mode"]
        cfg["torch_amp"] = bool(amp_flag)
        cfg["torch_amp_dtype"] = str(amp_dtype or cfg["torch_amp_dtype"]).strip().lower()
        cfg["torch_channels_last"] = bool(channels_last_flag)
    return cfg


def _use_torch_backend(runtime_config=None):
    cfg = resolve_non_gimp_runtime_config(runtime_config)
    return cfg["non_gimp_backend"] == "torch"


def _get_torch_backend_module(runtime_config=None):
    cfg = resolve_non_gimp_runtime_config(runtime_config)
    from . import non_gimp_ops_torch

    non_gimp_ops_torch.ensure_torch_cuda_runtime(cfg["torch_device"])
    return non_gimp_ops_torch, cfg


@lru_cache(maxsize=4)
def _load_hsl_config_cached(config_path: str):
    with open(config_path, "r") as file:
        return yaml.safe_load(file) or {}


# [gpu_render vendor 注] 原默认值 "configs/hsl.yaml" 依赖 CWD==monetGPT 根；
# 改为包内数据文件 gpu_render/configs/hsl.yaml，env 覆盖语义不变。
_DEFAULT_HSL_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "hsl.yaml")


def get_hsl_config(config_path: str | None = None):
    resolved_path = (config_path or os.environ.get("MONETGPT_HSL_CONFIG_PATH")
                     or _DEFAULT_HSL_CONFIG_PATH)
    return _load_hsl_config_cached(os.path.abspath(resolved_path))


def _resolve_exposure_params(intensity) -> tuple[float, float]:
    if isinstance(intensity, dict):
        return (
            float(intensity.get("ev_delta", 0.0)),
            float(intensity.get("black_level", 0.0)),
        )
    local_value = float(np.clip(intensity, -1.0, 1.0))
    return local_value * 5.0, 0.0


def _resolve_contrast_params(intensity) -> tuple[float, float]:
    if isinstance(intensity, dict):
        return (
            float(intensity.get("contrast_factor", 1.0)),
            float(intensity.get("brightness", 0.0)),
        )
    local_value = float(np.clip(intensity, -1.0, 1.0))
    return 1.0 + local_value, 0.0


def _resolve_temperature_params(intensity) -> tuple[float, float]:
    if isinstance(intensity, dict):
        return (
            float(intensity.get("base_kelvin", 6500.0)),
            float(intensity.get("delta_kelvin", 0.0)),
        )
    local_value = float(np.clip(intensity, -1.0, 1.0))
    return 6500.0, local_value * 5000.0


def _resolve_tint_payload(intensity) -> tuple[float, float]:
    if isinstance(intensity, dict):
        tint_units = float(intensity.get("tint_units", 0.0))
        base_kelvin = float(intensity.get("base_kelvin", _TINT_DEFAULT_BASE_KELVIN))
        return tint_units, base_kelvin
    if isinstance(intensity, (int, float)):
        local_value = float(np.clip(intensity, -1.0, 1.0))
        return local_value * 150.0, _TINT_DEFAULT_BASE_KELVIN
    raise TypeError(f"Unsupported tint intensity payload: {type(intensity)!r}")


def _resolve_saturation_params(intensity) -> float:
    if isinstance(intensity, dict):
        return float(intensity.get("saturation_pct", 0.0))
    local_value = float(np.clip(intensity, -1.0, 1.0))
    return local_value * 100.0


def _resolve_vibrance_params(intensity) -> float:
    if isinstance(intensity, dict):
        return float(intensity.get("vibrance_pct", 0.0))
    local_value = float(np.clip(intensity, -1.0, 1.0))
    # LR-faithful v2：归一 [-1,1] 直接映射 LR Vibrance 滑杆 [-100,100]（旧 ×60 压缩废弃）。
    return local_value * 100.0


_LR2DT_BLACKS_TABLE = (
    (-100.0, 0.020),
    (-50.0, 0.005),
    (0.0, 0.0),
    (50.0, -0.005),
    (100.0, -0.010),
)


def _invert_piecewise_table(table: tuple[tuple[float, float], ...], value: float) -> float:
    pairs = sorted(((float(y), float(x)) for x, y in table), key=lambda item: item[0])
    if value <= pairs[0][0]:
        return pairs[0][1]
    if value >= pairs[-1][0]:
        return pairs[-1][1]

    for (y0, x0), (y1, x1) in zip(pairs, pairs[1:]):
        if y0 <= value <= y1:
            ratio = (value - y0) / max(y1 - y0, 1e-8)
            return float(x0 + ratio * (x1 - x0))
    return pairs[-1][1]


def _normalize_global_compat_intensity(intensity):
    if isinstance(intensity, dict):
        return intensity
    value = float(intensity)
    if abs(value) > 1.0:
        value /= 100.0
    return value


def _has_nonzero_payload(value) -> bool:
    if isinstance(value, dict):
        for item in value.values():
            if isinstance(item, (int, float)) and abs(float(item)) > 1e-8:
                return True
        return False
    if isinstance(value, (int, float)):
        return abs(float(value)) > 1e-8
    return value not in (None, "", [], {})


def _resolve_tone_execution_payloads(
    highlights,
    shadows,
    whites,
    *,
    model,
    radius,
    compress,
    shadows_ccorrect,
    highlights_ccorrect,
    shadows_gain,
    highlights_gain,
    shadows_response_scale,
    highlights_response_scale,
    shadow_tonal_width_pct,
    whites_gain,
):
    resolved = {
        "model": model,
        "radius": float(radius),
        "compress": float(np.clip(compress, 0.0, 0.99)),
        "shadows_ccorrect": float(np.clip(shadows_ccorrect, 0.0, 1.0)),
        "highlights_ccorrect": float(np.clip(highlights_ccorrect, 0.0, 1.0)),
        "shadows_gain": float(shadows_gain),
        "highlights_gain": float(highlights_gain),
        "shadows_response_scale": float(np.clip(shadows_response_scale, 0.0, 3.0)),
        "highlights_response_scale": float(np.clip(highlights_response_scale, 0.0, 3.0)),
        "whites_gain": float(whites_gain),
        "shadow_tonal_width_pct": float(np.clip(shadow_tonal_width_pct, 1.0, 100.0)),
        "highlight_tonal_width_pct": float(SELECTED_TONE_RT_HIGHLIGHT_TONAL_WIDTH),
        "highlights_from_dict": isinstance(highlights, dict),
        "shadows_from_dict": isinstance(shadows, dict),
    }

    def _resolve_slider(value, key: str) -> float:
        if isinstance(value, dict):
            return float(np.clip(float(value.get(key, 0.0)) / 100.0, -1.0, 1.0))
        return float(np.clip(float(value), -1.0, 1.0))

    h_val = _resolve_slider(highlights, "highlights_pct")
    s_val = _resolve_slider(shadows, "shadows_pct")
    w_val = _resolve_slider(whites, "whites_pct")

    if isinstance(highlights, dict):
        from gpu_render.image_ops.operator_spec import get_operator_spec

        spec_defaults = dict(get_operator_spec("Highlights").default_aux_params or {})
        radius_px = highlights.get("radius_px", spec_defaults.get("radius_px", resolved["radius"]))
        compress_pct = highlights.get("compress_pct", spec_defaults.get("compress_pct", resolved["compress"] * 100.0))
        ccorrect_pct = highlights.get(
            "highlights_ccorrect_pct",
            spec_defaults.get("highlights_ccorrect_pct", resolved["highlights_ccorrect"] * 100.0),
        )
        if isinstance(radius_px, (int, float)):
            resolved["radius"] = max(0.1, float(radius_px))
        if isinstance(compress_pct, (int, float)):
            resolved["compress"] = float(np.clip(float(compress_pct) / 100.0, 0.0, 0.99))
        if isinstance(ccorrect_pct, (int, float)):
            resolved["highlights_ccorrect"] = float(np.clip(float(ccorrect_pct) / 100.0, 0.0, 1.0))
        if isinstance(highlights.get("highlights_response_scale"), (int, float)):
            resolved["highlights_response_scale"] = float(
                np.clip(float(highlights["highlights_response_scale"]), 0.0, 3.0)
            )
        if isinstance(highlights.get("highlight_tonal_width_pct"), (int, float)):
            resolved["highlight_tonal_width_pct"] = float(
                np.clip(float(highlights["highlight_tonal_width_pct"]), 1.0, 100.0)
            )

    if isinstance(shadows, dict):
        from gpu_render.image_ops.operator_spec import get_operator_spec

        spec_defaults = dict(get_operator_spec("Shadows").default_aux_params or {})
        radius_px = shadows.get("radius_px", spec_defaults.get("radius_px", resolved["radius"]))
        compress_pct = shadows.get("compress_pct", spec_defaults.get("compress_pct", resolved["compress"] * 100.0))
        ccorrect_pct = shadows.get(
            "shadows_ccorrect_pct",
            spec_defaults.get("shadows_ccorrect_pct", resolved["shadows_ccorrect"] * 100.0),
        )
        if isinstance(radius_px, (int, float)):
            resolved["radius"] = max(0.1, float(radius_px))
        if isinstance(compress_pct, (int, float)):
            resolved["compress"] = float(np.clip(float(compress_pct) / 100.0, 0.0, 0.99))
        if isinstance(ccorrect_pct, (int, float)):
            resolved["shadows_ccorrect"] = float(np.clip(float(ccorrect_pct) / 100.0, 0.0, 1.0))
        if isinstance(shadows.get("shadows_response_scale"), (int, float)):
            resolved["shadows_response_scale"] = float(
                np.clip(float(shadows["shadows_response_scale"]), 0.0, 3.0)
            )
        if isinstance(shadows.get("shadow_tonal_width_pct"), (int, float)):
            resolved["shadow_tonal_width_pct"] = float(
                np.clip(float(shadows["shadow_tonal_width_pct"]), 1.0, 100.0)
            )

    return h_val, s_val, w_val, resolved


def _apply_rgb_gains_np(image: np.ndarray, gains: np.ndarray) -> np.ndarray:
    out = np.asarray(image, dtype=np.float32) * np.asarray(gains, dtype=np.float32).reshape(1, 1, 3)
    return np.clip(out, 0.0, 1.0).astype(np.float32)




def _tint_coeffs_np(
    tint_units: float,
    *,
    base_kelvin: float,
) -> np.ndarray:
    del base_kelvin
    normalized = float(np.clip(tint_units / 150.0, -1.0, 1.0))
    coeffs = np.asarray(
        [
            np.exp(0.18 * normalized),
            np.exp(-0.28 * normalized),
            np.exp(0.18 * normalized),
        ],
        dtype=np.float32,
    )
    coeffs /= max(float(np.dot(coeffs, _RGB_LUMA)), 1e-6)
    return coeffs.astype(np.float32)


def _apply_saturation_np(image: np.ndarray, saturation_pct: float) -> np.ndarray:
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    scale = max(0.0, 1.0 + float(saturation_pct) / 100.0)
    lab = color.rgb2lab(img)
    lab[..., 1] = np.clip(lab[..., 1] * scale, -128.0, 127.0)
    lab[..., 2] = np.clip(lab[..., 2] * scale, -128.0, 127.0)
    # Higher-chroma Lab values can legitimately fall outside the sRGB gamut.
    # skimage clips those invalid Z values internally before XYZ->RGB, but emits
    # a warning for each image. The warning is noisy in dataset rendering and
    # does not change the final RGB result, so suppress this specific message.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Conversion from CIE-LAB, via XYZ to sRGB color space resulted in .* negative Z values.*",
            category=UserWarning,
        )
        rgb = color.lab2rgb(lab)
    return np.clip(rgb, 0.0, 1.0).astype(np.float32)


def _apply_native_saturation_np(image: np.ndarray, saturation_pct: float) -> np.ndarray:
    scale = max(0.0, 1.0 + float(saturation_pct) / 100.0)
    luma = np.tensordot(image, _RGB_LUMA, axes=([-1], [0])).astype(np.float32)
    gray = np.repeat(luma[..., None], 3, axis=2)
    return np.clip(gray * (1.0 - scale) + image * scale, 0.0, 1.0).astype(np.float32)


def _gegl_kelvin_to_rgb_np(temp_kelvin: float) -> np.ndarray:
    temp = float(np.clip(temp_kelvin, 1000.0, 12000.0))
    rgb = np.empty(3, dtype=np.float64)
    for channel in range(3):
        nomin = _GEGL_COLOR_TEMPERATURE_RGB_R55[channel, 0]
        for degree in range(1, 6):
            nomin = nomin * temp + _GEGL_COLOR_TEMPERATURE_RGB_R55[channel, degree]

        denom = _GEGL_COLOR_TEMPERATURE_RGB_R55[channel, 6]
        for degree in range(1, 6):
            denom = denom * temp + _GEGL_COLOR_TEMPERATURE_RGB_R55[channel, 6 + degree]

        rgb[channel] = nomin / denom

    return rgb.astype(np.float32)



def _apply_exposure_np(
    image: np.ndarray,
    ev_delta: float,
    black_level: float,
) -> np.ndarray:
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    white = float(np.exp2(-ev_delta))
    gain = 1.0 / max(white - black_level, 1e-6)
    adjusted = np.maximum((img - black_level) * gain, 0.0).astype(np.float32)
    pivot = 0.18
    gamma = 1.0 / (1.0 + 0.25 * ev_delta) if ev_delta >= 0.0 else 1.0 - 0.20 * ev_delta
    normalized = np.maximum(adjusted / max(pivot, 1e-6), 0.0)
    out = pivot * np.power(normalized, gamma)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _apply_contrast_np(
    image: np.ndarray,
    contrast_factor: float,
    brightness: float,
) -> np.ndarray:
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    normalized = float(contrast_factor) - 1.0
    if normalized >= 0.0:
        slope = 2.0 + 6.0 * float(np.clip(normalized, 0.0, 1.0))
        raw = 1.0 / (1.0 + np.exp(-slope * (img - 0.5)))
        lo = 1.0 / (1.0 + np.exp(-slope * (-0.5)))
        hi = 1.0 / (1.0 + np.exp(-slope * (0.5)))
        out = (raw - lo) / max(hi - lo, 1e-6)
    else:
        flatten = max(0.05, 1.0 + float(np.clip(normalized, -0.95, 0.0)))
        out = (img - 0.5) * flatten + 0.5
    out = out + float(brightness)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_image(image_path):
    norm_factor = 255.0
    if image_path.endswith(".tif"):
        img = tifffile.imread(image_path)
        if img.dtype == np.uint8:
            norm_factor = 255.0
        elif img.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            raise ValueError(f"Unsupported image bit depth: {img.dtype}")
        img = (img / norm_factor).astype(np.float32)
    else:
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)
        img = img / 255.0
    return img, norm_factor


def save_tif(image, norm_factor, output_path):
    img = image * norm_factor
    if norm_factor == 255.0:
        dtype = np.uint8
        img = img.astype(np.uint8)
    else:
        dtype = np.uint16
        img = img.astype(np.uint16)
    # Save as uint16
    tifffile.imwrite(output_path, img, dtype=dtype)


def read_image_low_res(image_path, max_size=700):
    """
    Reads an image (TIFF or common 8-bit formats), normalizes it to [0, 1] in float32,
    and resizes it to have a maximum dimension of `max_size`.
    """
    norm_factor = 255.0
    if image_path.lower().endswith(".tif") or image_path.lower().endswith(".tiff"):
        # Read using tifffile to preserve 8-bit or 16-bit depth
        img = tifffile.imread(image_path)

        # Determine normalization factor dynamically
        if img.dtype == np.uint8:
            norm_factor = 255.0
        elif img.dtype == np.uint16:
            norm_factor = 65535.0
        else:
            raise ValueError(f"Unsupported TIFF bit depth: {img.dtype}")

        # Convert to float32 in [0, 1]
        img = (img / norm_factor).astype(np.float32)

        # Get current width and height from the NumPy array shape
        # (img can be HxW or HxWxC)
        height, width = img.shape[:2]

        # Compute the scale factor
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize using skimage to keep float32 data without clamping to 8-bit
            img_resized = resize(
                img,
                (new_height, new_width),
                preserve_range=True,  # keep data in [0,1]
                anti_aliasing=True,
            ).astype(np.float32)
        else:
            img_resized = img

        return img_resized, norm_factor

    else:
        # Non-TIFF branch (typical 8-bit images)
        pil_img = Image.open(image_path).convert("RGB")
        width, height = pil_img.size

        # Compute the scale factor
        scale = min(max_size / width, max_size / height)

        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            # Resize with PIL (this is already 8-bit, so no data-depth loss here)
            pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)

        # Convert to NumPy float32 in [0, 1]
        img_arr = np.array(pil_img, dtype=np.float32) / 255.0

        return img_arr, norm_factor


def adjust_tint(image, intensity, runtime_config=None):
    """
    Apply the ART/RawTherapee-style green-magenta white-balance axis.

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to `tint_units = value * 150`. Dict inputs may pass canonical
    `{"tint_units": ..., "base_kelvin": ...}` directly.
    """
    tint_units, base_kelvin = _resolve_tint_payload(intensity)
    if abs(tint_units) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_tint_torch(
            img,
            intensity,
            device=resolved_runtime["torch_device"],
        )

    return _apply_rgb_gains_np(
        img,
        _tint_coeffs_np(
            tint_units,
            base_kelvin=base_kelvin,
        ),
    )


def adjust_vibrance(image, t, sigma_s=0.2, sigma_v=0.2, runtime_config=None):
    """
    Apply the LR-faithful vibrance implementation (calibrated on Lightroom GT).

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to `vibrance_pct = value * 100` (LR slider units). Dict inputs may
    pass canonical `{"vibrance_pct": ...}` directly.
    """
    del sigma_s, sigma_v
    vibrance_pct = _resolve_vibrance_params(t)
    if abs(vibrance_pct) < 1e-8:
        return image
    image = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    # LR-faithful v2：torch 后端镜像的是旧算法（结果分叉），统一走 numpy 实现。
    return apply_vibrance_lr(image, vibrance_pct)


def _rgb_luma_np(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image, dtype=np.float32)
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def _recompose_from_luma_np(image: np.ndarray, src_luma: np.ndarray, dst_luma: np.ndarray) -> np.ndarray:
    ratio = dst_luma / (src_luma + 1e-6)
    return np.clip(image * ratio[..., None], 0.0, 1.0).astype(np.float32)


def _midtone_mask_np(light: np.ndarray) -> np.ndarray:
    broad = smoothstep(light, 0.10, 0.40) * (1.0 - smoothstep(light, 0.60, 0.90))
    center = 1.0 - smoothstep(np.abs(light - 0.5), 0.12, 0.34)
    return np.clip(broad * (0.55 + 0.90 * center), 0.0, 1.0)


def _detail_activity_np(detail: np.ndarray, low: float = 0.0015, high: float = 0.03) -> np.ndarray:
    return smoothstep(np.abs(detail), low, high)


def _local_variance_np(light: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    mean = gaussian_filter(light, sigma=sigma)
    mean_sq = gaussian_filter(light * light, sigma=sigma)
    return np.clip(mean_sq - mean * mean, 0.0, None).astype(np.float32)


def _luminance_nr_spike_mask_np(light: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median3 = median_filter(light, size=3, mode="nearest").astype(np.float32)
    base_noise = np.sqrt(_local_variance_np(median3, sigma=1.0))
    spike_floor = np.maximum(0.08, 4.0 * base_noise + 0.02)
    spike_mask = np.abs(light - median3) > spike_floor
    return median3, spike_mask


def _apply_rgb_spike_cleanup_np(image: np.ndarray, spike_mask: np.ndarray) -> np.ndarray:
    if not np.any(spike_mask):
        return image.astype(np.float32, copy=False)
    cleaned = image.astype(np.float32, copy=True)
    for channel in range(cleaned.shape[2]):
        median_channel = median_filter(cleaned[..., channel], size=3, mode="nearest").astype(np.float32)
        cleaned[..., channel] = np.where(spike_mask, median_channel, cleaned[..., channel])
    return cleaned


def _edge_preserving_luma_smooth_np(
    light: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    diameter = max(3, int(diameter))
    if diameter % 2 == 0:
        diameter += 1
    return cv2.bilateralFilter(
        light.astype(np.float32, copy=False),
        diameter,
        float(max(1e-4, sigma_color)),
        float(max(1e-4, sigma_space)),
    ).astype(np.float32)


def _resolve_clarity_lr_value(intensity) -> float:
    """把归一 [-1,1] 标量或旧 canonical dict 载荷解析为 LR Clarity2012 滑杆值 [-100,100]。"""
    if isinstance(intensity, dict):
        if "clarity_pct" in intensity:
            return float(np.clip(_safe_float(intensity.get("clarity_pct", 0.0), 0.0), -100.0, 100.0))
        # 旧 canonical 载荷 clarity_detail ∈ [-0.65, 0.65]（= 归一值 ×0.65）→ 反解归一后 ×100
        detail = float(np.clip(_safe_float(intensity.get("clarity_detail", 0.0), 0.0), -0.65, 0.65))
        return detail / 0.65 * 100.0
    return float(np.clip(float(intensity), -1.0, 1.0)) * 100.0


def adjust_clarity(image, intensity, runtime_config=None):
    """LR-faithful Clarity（图像自适应大半径局部对比；LR GT 标定）。

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to LR Clarity2012 slider units via `value * 100`.
    torch 后端镜像的是旧算法（结果分叉），统一走 numpy v2 实现。
    """
    del runtime_config
    lr_value = _resolve_clarity_lr_value(intensity)
    if abs(lr_value) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    return adjust_clarity_v2(img, lr_value)


def _resolve_sharpness_lr_params(intensity) -> dict[str, float]:
    """解析为 LR Detail 面板参数：amount 0..150 / radius 0.5..3 / detail 0..100 / masking 0..100。"""
    if isinstance(intensity, dict):
        if "sharpness_amount" in intensity and "amount" not in intensity:
            # 旧 canonical 载荷 sharpness_amount ∈ [0,2] → LR amount = ×75（2 → 150）
            amount = float(np.clip(_safe_float(intensity.get("sharpness_amount", 0.0), 0.0), 0.0, 2.0)) * 75.0
        else:
            amount = float(np.clip(_safe_float(intensity.get("amount", 0.0), 0.0), 0.0, 150.0))
        return {
            "amount": amount,
            "radius": float(np.clip(_safe_float(intensity.get("radius", 1.0), 1.0), 0.5, 3.0)),
            "detail": float(np.clip(_safe_float(intensity.get("detail", 25.0), 25.0), 0.0, 100.0)),
            "masking": float(np.clip(_safe_float(intensity.get("masking", 0.0), 0.0), 0.0, 100.0)),
        }
    amount = max(0.0, float(np.clip(float(intensity), -1.0, 1.0))) * 100.0
    return {"amount": amount, "radius": 1.0, "detail": 25.0, "masking": 0.0}


def adjust_sharpness(image, intensity, runtime_config=None):
    """LR-faithful sharpening（Detail 面板；LR GT 标定的 USM + 软限幅）。

    Scalar inputs are interpreted as local compatibility units in [0, 1],
    mapped to LR Amount via `value * 100`（LR 滑杆 0..150）。
    torch 后端镜像的是旧算法（结果分叉），统一走 numpy v2 实现。
    """
    del runtime_config
    params = _resolve_sharpness_lr_params(intensity)
    if params["amount"] < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    return lr_sharpen(
        img,
        amount=params["amount"],
        radius=params["radius"],
        detail=params["detail"],
        masking=params["masking"],
    )


def _resolve_texture_params(intensity) -> dict[str, float | str]:
    from gpu_render.image_ops.operator_spec import get_operator_spec

    spec_defaults = dict(get_operator_spec("Texture").default_aux_params or {})
    if isinstance(intensity, dict):
        payload = dict(spec_defaults)
        payload.update(intensity)
        texture_gain = float(payload.get("texture_gain", 0.0))
    else:
        payload = dict(spec_defaults)
        texture_gain = float(np.clip(float(intensity), -1.0, 1.0))
        payload["texture_gain"] = texture_gain

    texture_gain = float(np.clip(texture_gain, -1.0, 1.0))
    return {
        "texture_gain": texture_gain,
        "profile": str(payload.get("profile", "darktable_diffuse_fine_selected")),
        "fine_sigma_px": float(max(0.1, _safe_float(payload.get("fine_sigma_px", 0.7), 0.7))),
        "detail_sigma_px": float(max(0.1, _safe_float(payload.get("detail_sigma_px", 2.0), 2.0))),
        "coarse_sigma_px": float(max(0.1, _safe_float(payload.get("coarse_sigma_px", 6.0), 6.0))),
        "mask_low": float(max(1e-6, _safe_float(payload.get("mask_low", 0.001), 0.001))),
        "mask_high": float(max(1e-5, _safe_float(payload.get("mask_high", 0.020), 0.020))),
        "gain_scale": float(max(0.0, _safe_float(payload.get("gain_scale", 1.45), 1.45))),
        "tone_strength": float(np.clip(_safe_float(payload.get("tone_strength", 0.65), 0.65), 0.0, 1.0)),
    }


def adjust_texture(image, intensity, runtime_config=None):
    """Enhance or suppress texture with the selected darktable diffuse-fine style band-pass profile."""
    params = _resolve_texture_params(intensity)
    amount = float(params["texture_gain"])
    if abs(amount) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_texture_torch(
            img,
            params,
            device=resolved_runtime["torch_device"],
        )

    luma = _rgb_luma_np(img)
    g1 = gaussian_filter(luma, sigma=float(params["fine_sigma_px"]))
    g2 = gaussian_filter(luma, sigma=float(params["detail_sigma_px"]))
    g3 = gaussian_filter(luma, sigma=float(params["coarse_sigma_px"]))
    band = 0.75 * (g1 - g2) + 0.35 * (g2 - g3)
    mask = _detail_activity_np(
        band,
        low=float(params["mask_low"]),
        high=float(params["mask_high"]),
    ) * (0.35 + float(params["tone_strength"]) * _midtone_mask_np(luma))
    out_luma = np.clip(luma + amount * float(params["gain_scale"]) * band * mask, 0.0, 1.0)
    return _recompose_from_luma_np(img, luma, out_luma)


def adjust_luminance_noise_reduction(image, intensity, runtime_config=None):
    """Reduce luma noise with edge-preserving smoothing and detail-aware protection."""
    if isinstance(intensity, dict):
        amount = float(intensity.get("luma_denoise_pct", 0.0)) / 100.0
    else:
        amount = float(np.clip(intensity, 0.0, 1.0))
    if amount <= 0:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_luminance_noise_reduction_torch(
            img,
            intensity,
            device=resolved_runtime["torch_device"],
        )

    raw_luma = _rgb_luma_np(img)
    _, spike_mask = _luminance_nr_spike_mask_np(raw_luma)
    work_img = _apply_rgb_spike_cleanup_np(img, spike_mask)
    luma = _rgb_luma_np(work_img)
    preclean = luma

    noise_level = np.sqrt(_local_variance_np(preclean, sigma=1.0))
    flat_mask = 1.0 - smoothstep(noise_level, 0.01, 0.045)

    diameter = max(5, 5 + int(round(amount * 4.0)) * 2)
    sigma_color = 0.04 + amount * 0.03
    sigma_space = 2.0 + amount * 2.5
    smooth = _edge_preserving_luma_smooth_np(
        preclean,
        diameter=diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    )

    detail_ref = np.abs(preclean - smooth)
    detail_mask = 1.0 - smoothstep(detail_ref, 0.015, 0.06)
    variance_mask = 1.0 - smoothstep(np.sqrt(_local_variance_np(preclean, sigma=1.2)), 0.015, 0.05)
    shadow_weight = 1.0 - 0.25 * smoothstep(preclean, 0.72, 0.95)
    blend = np.clip(
        amount
        * (0.15 + 0.85 * flat_mask)
        * (0.25 + 0.75 * detail_mask)
        * (0.3 + 0.7 * variance_mask)
        * shadow_weight,
        0.0,
        1.0,
    )
    out_luma = np.clip(preclean * (1.0 - blend) + smooth * blend, 0.0, 1.0)
    return _recompose_from_luma_np(work_img, luma, out_luma)


def _coerce_vignette_unit(value, default=0.0, allow_negative=False) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    if abs(value) > 1.0:
        value = value / 100.0
    lo, hi = (-1.0, 1.0) if allow_negative else (0.0, 1.0)
    return float(np.clip(value, lo, hi))


def _resolve_vignette_settings(config: dict | None) -> dict:
    """解析 config 中的暗角设置为 LR 原生参数（LR-faithful v2）。

    优先 CRS 原生键：PostCropVignetteAmount ∈ [-100,100]（LR 符号：负=压暗）、
    PostCropVignetteMidpoint/Feather（默认 50）、Roundness（默认 0）、Style（默认 1），
    以及镜头暗角 VignetteAmount（仅当 PostCropVignetteAmount 存在时按 CRS 语义解释）。
    兼容旧键 Vignette / VignetteAmount（旧约定 [-1,1] 或 [-100,100]，正=压暗；
    feather/roundness ∈ [0,1]）→ 映射为 PostCropVignetteAmount = -v×100。
    返回 {"postcrop": kwargs|None, "lens_amount": float}。
    """
    cfg = config if isinstance(config, dict) else {}
    if "PostCropVignetteAmount" in cfg:
        pc_amount = _safe_float(cfg.get("PostCropVignetteAmount", 0), 0.0)
        postcrop = None
        if abs(pc_amount) > 1e-8:
            postcrop = {
                "amount": pc_amount,
                "midpoint": _safe_float(cfg.get("PostCropVignetteMidpoint", 50), 50.0),
                "feather": _safe_float(cfg.get("PostCropVignetteFeather", 50), 50.0),
                "roundness": _safe_float(cfg.get("PostCropVignetteRoundness", 0), 0.0),
                "style": int(_safe_float(cfg.get("PostCropVignetteStyle", 1), 1.0)),
            }
        return {
            "postcrop": postcrop,
            "lens_amount": _safe_float(cfg.get("VignetteAmount", 0), 0.0),
        }

    legacy = cfg.get("Vignette")
    if legacy in (None, 0):
        legacy = cfg.get("VignetteAmount", 0)
    amount = _coerce_vignette_unit(legacy, default=0.0, allow_negative=True)
    if amount == 0:
        return {"postcrop": None, "lens_amount": 0.0}
    feather = _coerce_vignette_unit(cfg.get("VignetteFeather", 0.5), default=0.5)
    roundness = _coerce_vignette_unit(cfg.get("VignetteRoundness", 0.5), default=0.5)
    return {
        "postcrop": {
            "amount": -amount * 100.0,           # 旧约定 正=压暗 → LR 符号翻转
            "midpoint": 50.0,
            "feather": feather * 100.0,
            "roundness": (roundness - 0.5) * 200.0,
            "style": 1,
        },
        "lens_amount": 0.0,
    }


def adjust_vignette(
    image,
    intensity,
    feather=0.5,
    roundness=0.5,
    center_x=0.0,
    center_y=0.0,
    dither_amount=0.0,
    runtime_config=None,
):
    """Legacy 兼容包装：旧约定（正=压暗，[-1,1] 单位）→ LR PostCropVignette v2。

    center_x/center_y/dither_amount 在 LR v2 实现中不支持，忽略；
    torch 后端镜像的是旧算法（结果分叉），统一走 numpy v2 实现。
    """
    del center_x, center_y, dither_amount, runtime_config
    amount = _coerce_vignette_unit(intensity, default=0.0, allow_negative=True)
    if amount == 0:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    feather = _coerce_vignette_unit(feather, default=0.5)
    roundness = _coerce_vignette_unit(roundness, default=0.5)
    return apply_postcrop_vignette(
        img,
        amount=-amount * 100.0,
        midpoint=50.0,
        feather=feather * 100.0,
        roundness=(roundness - 0.5) * 200.0,
        style=1,
    )


def map_linear_to_exponential(user_input, min_exp=-2.5, max_exp=2.5):
    """
    Map user linear input (e.g., -5 to +5) to an exponential scale for exposure adjustment.
    - user_input: Linear input from the user (-5 to +5)
    - min_exp, max_exp: Internal exponential scaling range
    """
    # Map input (-5 to +5) to a normalized range (0 to 1)
    normalized = (user_input + 5) / 10  # Scale from -5..+5 to 0..1
    # Map normalized range to exponential range
    return min_exp + (max_exp - min_exp) * normalized


def adjust_exposure(image, intensity, runtime_config=None):
    """
    Apply the selected exposure implementation.

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to EV via `ev_delta = value * 5`. Dict inputs may pass canonical
    `{"ev_delta": ..., "black_level": ...}` directly.
    """
    ev_delta, black_level = _resolve_exposure_params(intensity)
    if abs(ev_delta) < 1e-8 and abs(black_level) < 1e-8:
        return image

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_exposure_torch(
            np.clip(image, 0.0, 1.0).astype(np.float32, copy=False),
            intensity,
            device=resolved_runtime["torch_device"],
        )

    return _apply_exposure_np(image, ev_delta, black_level)


##############################################################################
# 1. Vectorized RGB <-> HLS
##############################################################################
def rgb_to_hls_np(rgb):
    """
    Vectorized conversion from RGB to HLS, same definitions as colorsys.rgb_to_hls:
      - rgb: float32 or float64 array of shape (..., 3), in [0..1]
      - returns: H, L, S in [0..1], shape (..., 3)
    """
    # Extract R, G, B
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    maxc = np.max(rgb, axis=-1)
    minc = np.min(rgb, axis=-1)
    L = (maxc + minc) / 2.0

    # Avoid division-by-zero:
    # delta = 0 => R=G=B => H=0 (arbitrary), S=0
    delta = maxc - minc
    small_delta_mask = delta < 1e-20

    # Hue
    # We'll fill H with zeros then update only where delta != 0
    H = np.zeros_like(L)

    # For the pixels where max is r:
    mask_r = (r == maxc) & (~small_delta_mask)
    H[mask_r] = (g[mask_r] - b[mask_r]) / delta[mask_r]
    # For the pixels where max is g:
    mask_g = (g == maxc) & (~small_delta_mask)
    H[mask_g] = 2.0 + (b[mask_g] - r[mask_g]) / delta[mask_g]
    # For the pixels where max is b:
    mask_b = (b == maxc) & (~small_delta_mask)
    H[mask_b] = 4.0 + (r[mask_b] - g[mask_b]) / delta[mask_b]

    # Scale H from [0..6] range into [0..1] range
    H = (H / 6.0) % 1.0  # ensure positive mod 1

    # Saturation
    # S = delta / (2L) if L<=0.5, else delta / (2 - 2L)
    # but we must handle delta=0 => S=0
    S = np.zeros_like(L)
    not_small_mask = ~small_delta_mask
    L_le_half = (L <= 0.5) & not_small_mask
    L_gt_half = (L > 0.5) & not_small_mask

    # Avoid divide-by-zero at pure black/white (maxc+minc ~= 0 or ~= 2).
    eps = 1e-8
    denom1 = (maxc[L_le_half] + minc[L_le_half])
    denom2 = (2.0 - (maxc[L_gt_half] + minc[L_gt_half]))
    S[L_le_half] = (delta[L_le_half]) / (denom1 + eps)
    S[L_gt_half] = (delta[L_gt_half]) / (denom2 + eps)

    # Combine
    hls = np.stack([H, L, S], axis=-1)
    return hls


def hls_to_rgb_np(hls):
    """
    Vectorized conversion from HLS to RGB, matching colorsys.hls_to_rgb:
      - hls: float32 or float64 array of shape (..., 3), each channel in [0..1]
      - returns: float array in shape (..., 3), each channel in [0..1]
    """
    H = hls[..., 0]
    L = hls[..., 1]
    S = hls[..., 2]

    # If S=0 => Gray => R=G=B=L
    rgb = np.zeros_like(hls)
    R = rgb[..., 0]
    G = rgb[..., 1]
    B = rgb[..., 2]

    # For convenience, define a helper that adds a fractional hue:
    def hue_to_rgb(m1, m2, h):
        h_mod = h % 1.0
        # 6 segments in [0..1]
        return np.where(
            h_mod < 1 / 6,
            m1 + (m2 - m1) * 6 * h_mod,
            np.where(
                h_mod < 1 / 2,
                m2,
                np.where(h_mod < 2 / 3, m1 + (m2 - m1) * 6 * (2 / 3 - h_mod), m1),
            ),
        )

    # We only do the fancy stuff where S>0; otherwise it's just L.
    # We'll mask the S>0 part.
    s_pos_mask = S > 1e-7
    # Intermediate values
    m2 = np.where(L < 0.5, L + L * S, L + S - L * S)
    m1 = 2 * L - m2

    # Fill R,G,B only where S>0
    R[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask] + 1 / 3)
    G[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask])
    B[s_pos_mask] = hue_to_rgb(m1[s_pos_mask], m2[s_pos_mask], H[s_pos_mask] - 1 / 3)

    # Where S=0, it's simply gray => L
    gray_mask = ~s_pos_mask
    R[gray_mask] = L[gray_mask]
    G[gray_mask] = L[gray_mask]
    B[gray_mask] = L[gray_mask]

    return rgb


_GIMP_HUE_RANGE_ALL = 0
_GIMP_HUE_RANGE_RED = 1
_GIMP_HUE_RANGE_YELLOW = 2
_GIMP_HUE_RANGE_GREEN = 3
_GIMP_HUE_RANGE_CYAN = 4
_GIMP_HUE_RANGE_BLUE = 5
_GIMP_HUE_RANGE_MAGENTA = 6

_BASE_COLOR_TO_GIMP_RANGE = {
    "red": _GIMP_HUE_RANGE_RED,
    "yellow": _GIMP_HUE_RANGE_YELLOW,
    "green": _GIMP_HUE_RANGE_GREEN,
    "aqua": _GIMP_HUE_RANGE_CYAN,
    "blue": _GIMP_HUE_RANGE_BLUE,
    "magenta": _GIMP_HUE_RANGE_MAGENTA,
}


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _deep_update_dict(target, updates):
    if not isinstance(target, dict) or not isinstance(updates, dict):
        return target
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update_dict(target[key], value)
        else:
            target[key] = value
    return target


def _get_hsl_color_weights(color, cfg):
    del cfg
    color_key = str(color or "").strip().lower()
    if color_key in _BASE_COLOR_TO_GIMP_RANGE:
        return {_BASE_COLOR_TO_GIMP_RANGE[color_key]: 1.0}

    return {_GIMP_HUE_RANGE_RED: 1.0}


def _scale_hsl_change(change, gain, fallback_gain):
    base = _safe_float(change, 0.0)
    if abs(base) > 1.0:
        base = base / 100.0
    base = float(np.clip(base, -1.0, 1.0))
    if gain is None:
        gain = fallback_gain
    scaled = base * max(0.0, _safe_float(gain, fallback_gain))
    return float(np.clip(scaled, -1.0, 1.0))


def _sanitize_hsl_stability_cfg(cfg):
    cfg = cfg if isinstance(cfg, dict) else {}
    sat_floor_low = float(np.clip(_safe_float(cfg.get("sat_floor_low", 0.02), 0.02), 0.0, 1.0))
    sat_floor_high = float(np.clip(_safe_float(cfg.get("sat_floor_high", 0.12), 0.12), 0.0, 1.0))
    if sat_floor_high <= sat_floor_low:
        sat_floor_high = min(1.0, sat_floor_low + 1e-3)

    chroma_floor_low = float(np.clip(_safe_float(cfg.get("chroma_floor_low", 0.006), 0.006), 0.0, 1.0))
    chroma_floor_high = float(np.clip(_safe_float(cfg.get("chroma_floor_high", 0.040), 0.040), 0.0, 1.0))
    if chroma_floor_high <= chroma_floor_low:
        chroma_floor_high = min(1.0, chroma_floor_low + 1e-3)

    artifact_raw = cfg.get("artifact_guard", {})
    artifact_raw = artifact_raw if isinstance(artifact_raw, dict) else {}
    kernel = int(_safe_float(artifact_raw.get("kernel", 3), 3))
    kernel = max(1, kernel)
    if kernel % 2 == 0:
        kernel += 1

    luma_raw = cfg.get("luma_guard", {})
    luma_raw = luma_raw if isinstance(luma_raw, dict) else {}
    luma_sat_floor_low = float(
        np.clip(_safe_float(luma_raw.get("sat_floor_low", 0.06), 0.06), 0.0, 1.0)
    )
    luma_sat_floor_high = float(
        np.clip(_safe_float(luma_raw.get("sat_floor_high", 0.22), 0.22), 0.0, 1.0)
    )
    if luma_sat_floor_high <= luma_sat_floor_low:
        luma_sat_floor_high = min(1.0, luma_sat_floor_low + 1e-3)

    luma_chroma_floor_low = float(
        np.clip(_safe_float(luma_raw.get("chroma_floor_low", chroma_floor_low), chroma_floor_low), 0.0, 1.0)
    )
    luma_chroma_floor_high = float(
        np.clip(
            _safe_float(luma_raw.get("chroma_floor_high", chroma_floor_high), chroma_floor_high),
            0.0,
            1.0,
        )
    )
    if luma_chroma_floor_high <= luma_chroma_floor_low:
        luma_chroma_floor_high = min(1.0, luma_chroma_floor_low + 1e-3)

    luma_guided_raw = luma_raw.get("guided", {})
    luma_guided_raw = luma_guided_raw if isinstance(luma_guided_raw, dict) else {}
    luma_radius = max(1, int(_safe_float(luma_guided_raw.get("radius", 5), 5)))

    luma_speckle_raw = luma_raw.get("speckle", {})
    luma_speckle_raw = luma_speckle_raw if isinstance(luma_speckle_raw, dict) else {}
    luma_speckle_kernel = max(1, int(_safe_float(luma_speckle_raw.get("kernel", 3), 3)))
    if luma_speckle_kernel % 2 == 0:
        luma_speckle_kernel += 1

    return {
        "enabled": bool(cfg.get("enabled", True)),
        "sat_floor_low": sat_floor_low,
        "sat_floor_high": sat_floor_high,
        "chroma_floor_low": chroma_floor_low,
        "chroma_floor_high": chroma_floor_high,
        "artifact_guard": {
            "enabled": bool(artifact_raw.get("enabled", True)),
            "diff_thresh": float(
                np.clip(_safe_float(artifact_raw.get("diff_thresh", 0.10), 0.10), 0.0, 1.0)
            ),
            "kernel": kernel,
            "low_sat_threshold": float(
                np.clip(
                    _safe_float(artifact_raw.get("low_sat_threshold", sat_floor_high), sat_floor_high),
                    0.0,
                    1.0,
                )
            ),
            "low_chroma_threshold": float(
                np.clip(
                    _safe_float(artifact_raw.get("low_chroma_threshold", chroma_floor_high), chroma_floor_high),
                    0.0,
                    1.0,
                )
            ),
            "blend": float(np.clip(_safe_float(artifact_raw.get("blend", 0.85), 0.85), 0.0, 1.0)),
        },
        "luma_guard": {
            "enabled": bool(luma_raw.get("enabled", True)),
            "sat_floor_low": luma_sat_floor_low,
            "sat_floor_high": luma_sat_floor_high,
            "chroma_floor_low": luma_chroma_floor_low,
            "chroma_floor_high": luma_chroma_floor_high,
            "strength": float(np.clip(_safe_float(luma_raw.get("strength", 1.0), 1.0), 0.0, 1.0)),
            "guided": {
                "enabled": bool(luma_guided_raw.get("enabled", True)),
                "radius": luma_radius,
                "eps": float(max(1e-8, _safe_float(luma_guided_raw.get("eps", 1e-3), 1e-3))),
                "sigma_space": float(
                    max(1.0, _safe_float(luma_guided_raw.get("sigma_space", float(luma_radius)), float(luma_radius)))
                ),
                "sigma_color": float(max(1e-6, _safe_float(luma_guided_raw.get("sigma_color", 0.06), 0.06))),
            },
            "speckle": {
                "enabled": bool(luma_speckle_raw.get("enabled", True)),
                "low_sat_threshold": float(
                    np.clip(
                        _safe_float(
                            luma_speckle_raw.get("low_sat_threshold", luma_sat_floor_high),
                            luma_sat_floor_high,
                        ),
                        0.0,
                        1.0,
                    )
                ),
                "low_chroma_threshold": float(
                    np.clip(
                        _safe_float(
                            luma_speckle_raw.get("low_chroma_threshold", max(0.06, luma_chroma_floor_high)),
                            max(0.06, luma_chroma_floor_high),
                        ),
                        0.0,
                        1.0,
                    )
                ),
                "delta_thresh": float(
                    np.clip(_safe_float(luma_speckle_raw.get("delta_thresh", 0.06), 0.06), 0.0, 1.0)
                ),
                "min_area": max(1, int(_safe_float(luma_speckle_raw.get("min_area", 6), 6))),
                "kernel": luma_speckle_kernel,
                "blend": float(
                    np.clip(_safe_float(luma_speckle_raw.get("blend", 0.85), 0.85), 0.0, 1.0)
                ),
            },
        },
    }


def _resolve_hsl_stability_cfg(color_cfg, stability_override=None):
    color_cfg = color_cfg if isinstance(color_cfg, dict) else {}
    base_cfg = color_cfg.get("stability", {})
    base_cfg = dict(base_cfg) if isinstance(base_cfg, dict) else {}

    if isinstance(stability_override, dict):
        merged = dict(base_cfg)
        _deep_update_dict(merged, stability_override)
        return _sanitize_hsl_stability_cfg(merged)

    return _sanitize_hsl_stability_cfg(base_cfg)


def _build_gimp_hsl_adjustments(cfg, color, hue_change, saturation_change, luminance_change):
    hue_gain = cfg.get("gimp_gain_h")
    if hue_gain is None:
        hue_gain = _safe_float(cfg.get("max_range_h", 180.0), 180.0) / 180.0
    sat_gain = cfg.get("gimp_gain_s", cfg.get("max_range_s", 1.0))
    light_gain = cfg.get("gimp_gain_l", cfg.get("max_range_l", 1.0))

    hue_delta = _scale_hsl_change(hue_change, hue_gain, fallback_gain=1.0)
    sat_delta = _scale_hsl_change(saturation_change, sat_gain, fallback_gain=1.0)
    light_delta = _scale_hsl_change(luminance_change, light_gain, fallback_gain=1.0)

    hue_adj = np.zeros(7, dtype=np.float32)
    sat_adj = np.zeros(7, dtype=np.float32)
    light_adj = np.zeros(7, dtype=np.float32)

    for hue_range, weight in _get_hsl_color_weights(color, cfg).items():
        w = float(np.clip(weight, 0.0, 1.0))
        hue_adj[hue_range] += hue_delta * w
        sat_adj[hue_range] += sat_delta * w
        light_adj[hue_range] += light_delta * w

    hue_adj = np.clip(hue_adj, -1.0, 1.0)
    sat_adj = np.clip(sat_adj, -1.0, 1.0)
    light_adj = np.clip(light_adj, -1.0, 1.0)
    return hue_adj, sat_adj, light_adj


def _build_all_hsl_adjustments(all_color_settings, adjustments_by_color):
    """将 6 个 canonical 颜色的 H/S/L 调整累积到单组 adj 数组 (size 7)。"""
    hue_adj = np.zeros(7, dtype=np.float32)
    sat_adj = np.zeros(7, dtype=np.float32)
    light_adj = np.zeros(7, dtype=np.float32)

    for color_name, adj in adjustments_by_color.items():
        color_key = color_name.lower()
        cfg = all_color_settings.get(color_key, {})
        h = adj["HueAdjustment"]
        s = adj["SaturationAdjustment"]
        l_ = adj["LuminanceAdjustment"]
        if h == 0 and s == 0 and l_ == 0:
            continue
        h_arr, s_arr, l_arr = _build_gimp_hsl_adjustments(cfg, color_key, h, s, l_)
        hue_adj += h_arr
        sat_adj += s_arr
        light_adj += l_arr

    hue_adj = np.clip(hue_adj, -1.0, 1.0)
    sat_adj = np.clip(sat_adj, -1.0, 1.0)
    light_adj = np.clip(light_adj, -1.0, 1.0)
    return hue_adj, sat_adj, light_adj


def _gimp_map_hue(values, hue_all, hue_range):
    return np.mod(values + (hue_all + hue_range) * 0.5, 1.0)


def _gimp_map_hue_overlap(
    values,
    hue_all,
    primary_hue_adj,
    secondary_hue_adj,
    primary_intensity,
    secondary_intensity,
):
    blended = primary_hue_adj * primary_intensity + secondary_hue_adj * secondary_intensity
    return np.mod(values + (hue_all + blended) * 0.5, 1.0)


def _gimp_map_saturation(values, sat_all, sat_range):
    out = values * (sat_all + sat_range + 1.0)
    return np.clip(out, 0.0, 1.0)


def _gimp_map_lightness(values, light_all, light_range):
    v = light_all + light_range
    out = np.where(v < 0.0, values * (v + 1.0), values + (v * (1.0 - values)))
    return np.clip(out, 0.0, 1.0)


def _gimp_map_lightness_achromatic(values, light_all):
    return np.where(light_all < 0.0, values * (light_all + 1.0), values + (light_all * (1.0 - values)))


def _resolve_gimp_hue_ranges(hue_values, overlap):
    hue_values = np.mod(hue_values, 1.0)
    hue_six = hue_values * 6.0
    overlap = float(np.clip(overlap, 0.0, 1.0))
    overlap_half = overlap * 0.5

    primary_hue = np.zeros_like(hue_values, dtype=np.int8)
    secondary_hue = np.zeros_like(hue_values, dtype=np.int8)
    use_secondary = np.zeros_like(hue_values, dtype=bool)
    primary_intensity = np.ones_like(hue_values, dtype=np.float32)
    secondary_intensity = np.zeros_like(hue_values, dtype=np.float32)
    assigned = np.zeros_like(hue_values, dtype=bool)

    for hue_counter in range(7):
        hue_threshold = float(hue_counter) + 0.5
        mask = (~assigned) & (hue_six < (hue_threshold + overlap_half))
        if not np.any(mask):
            continue

        primary_hue[mask] = hue_counter

        if overlap_half > 0.0:
            sec_mask = mask & (hue_six > (hue_threshold - overlap_half))
            if np.any(sec_mask):
                use_secondary[sec_mask] = True
                secondary_hue[sec_mask] = hue_counter + 1
                sec = (hue_six[sec_mask] - hue_threshold + overlap_half) / (2.0 * overlap_half)
                sec = np.clip(sec, 0.0, 1.0).astype(np.float32)
                secondary_intensity[sec_mask] = sec
                primary_intensity[sec_mask] = 1.0 - sec

        assigned[mask] = True

    wrap_primary = primary_hue >= 6
    if np.any(wrap_primary):
        primary_hue[wrap_primary] = 0
        secondary_hue[wrap_primary] = 0
        use_secondary[wrap_primary] = False
        primary_intensity[wrap_primary] = 1.0
        secondary_intensity[wrap_primary] = 0.0

    wrap_secondary = secondary_hue >= 6
    if np.any(wrap_secondary):
        secondary_hue[wrap_secondary] = 0

    primary_range = primary_hue + 1
    secondary_range = secondary_hue + 1
    return primary_range, secondary_range, use_secondary, primary_intensity, secondary_intensity


def _smooth_lightness_delta(delta_light, rgb01, guided_cfg):
    guided_cfg = guided_cfg if isinstance(guided_cfg, dict) else {}
    if not guided_cfg.get("enabled", True):
        return delta_light

    radius = max(1, int(_safe_float(guided_cfg.get("radius", 5), 5)))
    eps = float(max(1e-8, _safe_float(guided_cfg.get("eps", 1e-3), 1e-3)))
    src = delta_light.astype(np.float32, copy=False)
    guide_gray = (
        0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]
    ).astype(np.float32, copy=False)

    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
        try:
            out = cv2.ximgproc.guidedFilter(
                guide=guide_gray,
                src=src,
                radius=radius,
                eps=eps,
            )
            return out.astype(np.float32, copy=False)
        except Exception:
            pass

    d = max(3, radius * 2 + 1)
    sigma_space = float(max(1.0, _safe_float(guided_cfg.get("sigma_space", float(radius)), float(radius))))
    sigma_color = float(max(1e-6, _safe_float(guided_cfg.get("sigma_color", 0.06), 0.06)))
    try:
        out = cv2.bilateralFilter(src, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        return out.astype(np.float32, copy=False)
    except Exception:
        return src


def _suppress_lightness_speckles(delta_light, sat, chroma, speckle_cfg):
    speckle_cfg = speckle_cfg if isinstance(speckle_cfg, dict) else {}
    if not speckle_cfg.get("enabled", True):
        return delta_light

    low_sat_threshold = float(np.clip(_safe_float(speckle_cfg.get("low_sat_threshold", 0.22), 0.22), 0.0, 1.0))
    low_chroma_default = 0.06
    low_chroma_threshold = float(
        np.clip(
            _safe_float(speckle_cfg.get("low_chroma_threshold", low_chroma_default), low_chroma_default),
            0.0,
            1.0,
        )
    )
    delta_thresh = float(np.clip(_safe_float(speckle_cfg.get("delta_thresh", 0.06), 0.06), 0.0, 1.0))
    min_area = max(1, int(_safe_float(speckle_cfg.get("min_area", 6), 6)))
    kernel = max(1, int(_safe_float(speckle_cfg.get("kernel", 3), 3)))
    if kernel % 2 == 0:
        kernel += 1
    blend = float(np.clip(_safe_float(speckle_cfg.get("blend", 0.85), 0.85), 0.0, 1.0))

    low_mask = sat <= low_sat_threshold
    if chroma is not None:
        low_mask = low_mask | (chroma <= low_chroma_threshold)

    spike_mask = low_mask & (np.abs(delta_light) >= delta_thresh)
    if not np.any(spike_mask):
        return delta_light

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        spike_mask.astype(np.uint8),
        connectivity=8,
    )
    if num_labels <= 1:
        return delta_light

    median_delta = median_filter(delta_light, size=kernel, mode="nearest").astype(np.float32, copy=False)
    out = delta_light.astype(np.float32, copy=True)
    for label_idx in range(1, num_labels):
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area > min_area:
            continue
        comp_mask = labels == label_idx
        if blend >= 1.0:
            out[comp_mask] = median_delta[comp_mask]
        else:
            out[comp_mask] = (
                (1.0 - blend) * out[comp_mask] + blend * median_delta[comp_mask]
            )
    return out


def _apply_gimp_hsl_transform(rgb01, hue_adj, sat_adj, light_adj, overlap, stability_cfg=None):
    if (
        np.max(np.abs(hue_adj)) < 1e-8
        and np.max(np.abs(sat_adj)) < 1e-8
        and np.max(np.abs(light_adj)) < 1e-8
    ):
        return rgb01

    hls = rgb_to_hls_np(rgb01).astype(np.float32, copy=False)
    hue = hls[..., 0]
    light = hls[..., 1]
    sat = hls[..., 2]
    chroma = (np.max(rgb01, axis=-1) - np.min(rgb01, axis=-1)).astype(np.float32, copy=False)

    (
        primary_range,
        secondary_range,
        use_secondary,
        primary_intensity,
        secondary_intensity,
    ) = _resolve_gimp_hue_ranges(hue, overlap)

    hue_out = hue.copy()
    sat_out = sat.copy()
    light_out = light.copy()

    secondary_mask = use_secondary
    if np.any(secondary_mask):
        primary_idx = primary_range[secondary_mask]
        secondary_idx = secondary_range[secondary_mask]
        p_weight = primary_intensity[secondary_mask]
        s_weight = secondary_intensity[secondary_mask]

        hue_out[secondary_mask] = _gimp_map_hue_overlap(
            hue[secondary_mask],
            hue_adj[_GIMP_HUE_RANGE_ALL],
            hue_adj[primary_idx],
            hue_adj[secondary_idx],
            p_weight,
            s_weight,
        )

        sat_primary = _gimp_map_saturation(
            sat[secondary_mask],
            sat_adj[_GIMP_HUE_RANGE_ALL],
            sat_adj[primary_idx],
        )
        sat_secondary = _gimp_map_saturation(
            sat[secondary_mask],
            sat_adj[_GIMP_HUE_RANGE_ALL],
            sat_adj[secondary_idx],
        )
        sat_out[secondary_mask] = sat_primary * p_weight + sat_secondary * s_weight

        light_primary = _gimp_map_lightness(
            light[secondary_mask],
            light_adj[_GIMP_HUE_RANGE_ALL],
            light_adj[primary_idx],
        )
        light_secondary = _gimp_map_lightness(
            light[secondary_mask],
            light_adj[_GIMP_HUE_RANGE_ALL],
            light_adj[secondary_idx],
        )
        light_out[secondary_mask] = light_primary * p_weight + light_secondary * s_weight

    _ACHROMATIC_THRESHOLD = 0.005
    no_secondary_mask = ~secondary_mask
    achromatic_mask = no_secondary_mask & (sat <= _ACHROMATIC_THRESHOLD)
    if np.any(achromatic_mask):
        light_out[achromatic_mask] = _gimp_map_lightness_achromatic(
            light[achromatic_mask],
            light_adj[_GIMP_HUE_RANGE_ALL],
        )

    chroma_mask = no_secondary_mask & (sat > _ACHROMATIC_THRESHOLD)
    if np.any(chroma_mask):
        primary_idx = primary_range[chroma_mask]
        hue_out[chroma_mask] = _gimp_map_hue(
            hue[chroma_mask],
            hue_adj[_GIMP_HUE_RANGE_ALL],
            hue_adj[primary_idx],
        )
        light_out[chroma_mask] = _gimp_map_lightness(
            light[chroma_mask],
            light_adj[_GIMP_HUE_RANGE_ALL],
            light_adj[primary_idx],
        )
        sat_out[chroma_mask] = _gimp_map_saturation(
            sat[chroma_mask],
            sat_adj[_GIMP_HUE_RANGE_ALL],
            sat_adj[primary_idx],
        )

    if isinstance(stability_cfg, dict) and stability_cfg.get("enabled", True):
        sat_conf = smoothstep(
            sat,
            stability_cfg["sat_floor_low"],
            stability_cfg["sat_floor_high"],
        ).astype(np.float32, copy=False)
        chroma_conf = smoothstep(
            chroma,
            stability_cfg.get("chroma_floor_low", 0.006),
            stability_cfg.get("chroma_floor_high", 0.040),
        ).astype(np.float32, copy=False)
        conf = (sat_conf * chroma_conf).astype(np.float32, copy=False)

        hue_delta = np.mod(hue_out - hue + 0.5, 1.0) - 0.5
        hue_out = np.mod(hue + hue_delta * conf, 1.0).astype(np.float32, copy=False)
        sat_out = np.clip(sat + (sat_out - sat) * conf, 0.0, 1.0).astype(np.float32, copy=False)

        luma_cfg = stability_cfg.get("luma_guard", {})
        if isinstance(luma_cfg, dict) and luma_cfg.get("enabled", True):
            luma_sat_conf = smoothstep(
                sat,
                luma_cfg.get("sat_floor_low", 0.06),
                luma_cfg.get("sat_floor_high", 0.22),
            ).astype(np.float32, copy=False)
            luma_chroma_conf = smoothstep(
                chroma,
                luma_cfg.get("chroma_floor_low", stability_cfg.get("chroma_floor_low", 0.006)),
                luma_cfg.get("chroma_floor_high", stability_cfg.get("chroma_floor_high", 0.040)),
            ).astype(np.float32, copy=False)
            luma_conf = (luma_sat_conf * luma_chroma_conf).astype(np.float32, copy=False)
            delta_light = (light_out - light).astype(np.float32, copy=False)
            delta_light = delta_light * luma_conf * float(np.clip(_safe_float(luma_cfg.get("strength", 1.0), 1.0), 0.0, 1.0))
            delta_light = _smooth_lightness_delta(
                delta_light,
                rgb01,
                luma_cfg.get("guided", {}),
            )
            delta_light = _suppress_lightness_speckles(
                delta_light,
                sat,
                chroma,
                luma_cfg.get("speckle", {}),
            )
            light_out = np.clip(light + delta_light, 0.0, 1.0).astype(np.float32, copy=False)

    hls_out = np.stack([hue_out, light_out, sat_out], axis=-1)
    rgb_out = hls_to_rgb_np(hls_out).astype(np.float32, copy=False)

    if isinstance(stability_cfg, dict) and stability_cfg.get("enabled", True):
        artifact_cfg = stability_cfg.get("artifact_guard", {})
        if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled", True):
            diff = np.mean(np.abs(rgb_out - rgb01), axis=-1)
            low_sat_mask = sat <= artifact_cfg.get("low_sat_threshold", stability_cfg["sat_floor_high"])
            low_chroma_mask = chroma <= artifact_cfg.get(
                "low_chroma_threshold",
                stability_cfg.get("chroma_floor_high", 0.040),
            )
            low_mask = low_sat_mask | low_chroma_mask
            spike_mask = low_mask & (diff >= artifact_cfg.get("diff_thresh", 0.10))

            if np.any(spike_mask):
                kernel = int(artifact_cfg.get("kernel", 3))
                blend = float(np.clip(_safe_float(artifact_cfg.get("blend", 0.85), 0.85), 0.0, 1.0))

                filtered = np.stack(
                    [
                        median_filter(rgb_out[..., 0], size=kernel, mode="nearest"),
                        median_filter(rgb_out[..., 1], size=kernel, mode="nearest"),
                        median_filter(rgb_out[..., 2], size=kernel, mode="nearest"),
                    ],
                    axis=-1,
                ).astype(np.float32, copy=False)

                if blend >= 1.0:
                    rgb_out[spike_mask] = filtered[spike_mask]
                else:
                    rgb_out[spike_mask] = (
                        (1.0 - blend) * rgb_out[spike_mask] + blend * filtered[spike_mask]
                    )

    return np.clip(rgb_out, 0.0, 1.0)


def _apply_hsl_adjustment_with_cfg(
    image_array,
    color,
    hue_change,
    saturation_change,
    luminance_change,
    color_cfg,
    engine=None,
    overlap=None,
    stability=None,
    runtime_config=None,
):
    if hue_change == 0 and saturation_change == 0 and luminance_change == 0:
        return image_array

    input_dtype = image_array.dtype
    img = image_array.astype(np.float32, copy=False)
    if np.issubdtype(input_dtype, np.integer):
        denom = float(np.iinfo(input_dtype).max)
        img = img / max(1.0, denom)
    elif img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    cfg = color_cfg if isinstance(color_cfg, dict) else {}
    if engine is None:
        engine_name = str(cfg.get("engine", "gimp_approx_8color"))
    else:
        engine_name = str(engine)
    engine_name = engine_name.strip().lower()
    if engine_name == "gimp_approx_6color":
        engine_name = "gimp_approx_8color"
    elif engine_name == "gimp_stable_6color":
        engine_name = "gimp_stable_8color"
    if engine_name not in {"gimp_approx_8color", "gimp_stable_8color"}:
        print(f"WARNING: unsupported HSL engine '{engine_name}', fallback to gimp_approx_8color.")
        engine_name = "gimp_approx_8color"

    if overlap is None:
        overlap = cfg.get("overlap_default", 0.0)
    overlap = float(np.clip(_safe_float(overlap, 0.0), 0.0, 1.0))
    hue_adj, sat_adj, light_adj = _build_gimp_hsl_adjustments(
        cfg,
        color,
        hue_change,
        saturation_change,
        luminance_change,
    )
    stability_cfg = None
    if engine_name == "gimp_stable_8color" or stability is not None:
        stability_cfg = _resolve_hsl_stability_cfg(cfg, stability_override=stability)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        out = torch_backend.apply_gimp_hsl_transform_torch(
            img,
            hue_adj,
            sat_adj,
            light_adj,
            overlap,
            stability_cfg=stability_cfg,
            device=resolved_runtime["torch_device"],
        )
    else:
        out = _apply_gimp_hsl_transform(
            img,
            hue_adj,
            sat_adj,
            light_adj,
            overlap,
            stability_cfg=stability_cfg,
        )

    if np.issubdtype(input_dtype, np.integer):
        max_value = float(np.iinfo(input_dtype).max)
        out = np.clip(out * max_value, 0.0, max_value)
        return out.astype(input_dtype)
    return out.astype(np.float32, copy=False)


def adjust_hsl(
    image_array,
    color="red",
    hue_change=0.0,  # fraction of the max *range* hue shift ([-1..+1])
    saturation_change=0.0,  # fraction of the max *range* saturation shift ([-1..+1])
    luminance_change=0.0,  # fraction of the max *range* luminance shift ([-1..+1])
    blur_sigma=5.0,
    engine=None,
    overlap=None,
    stability=None,
    runtime_config=None,
):
    _ = blur_sigma  # kept for API compatibility
    color_key = str(color or "").strip().lower()

    if color_key in {"blue", "orange", "purple"}:
        # LR-faithful v2 带（不读 hsl.yaml；滑杆 [-1,1] 归一 → LR [-100,100]）
        def _to_lr(v):
            v = _safe_float(v, 0.0)
            return float(np.clip(v * 100.0 if abs(v) <= 1.0 else v, -100.0, 100.0))

        input_dtype = image_array.dtype
        img = image_array.astype(np.float32, copy=False)
        if np.issubdtype(input_dtype, np.integer):
            img = img / max(1.0, float(np.iinfo(input_dtype).max))
        elif img.max() > 1.0:
            img = img / 255.0
        img = np.clip(img, 0.0, 1.0)
        h, s, l_ = _to_lr(hue_change), _to_lr(saturation_change), _to_lr(luminance_change)
        if color_key == "blue":
            out = apply_hsl_blue_v2(img, hue=h, sat=s, lum=l_)
        else:
            out = apply_hsl_orange_purple_v2(img, color_key, hue=h, sat=s, lum=l_)
        if np.issubdtype(input_dtype, np.integer):
            max_value = float(np.iinfo(input_dtype).max)
            return np.clip(out * max_value, 0.0, max_value).astype(input_dtype)
        return np.asarray(out, dtype=np.float32)

    color_settings = get_hsl_config()
    if color_key not in color_settings:
        print(f"WARNING: '{color_key}' is not in color_settings; using 'red' instead.")
        color_key = "red"

    cfg = color_settings.get(color_key, {})
    return _apply_hsl_adjustment_with_cfg(
        image_array=image_array,
        color=color_key,
        hue_change=hue_change,
        saturation_change=saturation_change,
        luminance_change=luminance_change,
        color_cfg=cfg,
        engine=engine,
        overlap=overlap,
        stability=stability,
        runtime_config=runtime_config,
    )


def _hsl_band_lr_value(config, key) -> float:
    """config HSL 滑杆值 → LR 单位 [-100,100]（|v|≤1 视为归一值 ×100）。"""
    v = _safe_float(config.get(key, 0), 0.0)
    if abs(v) <= 1.0:
        v *= 100.0
    return float(np.clip(v, -100.0, 100.0))


def _apply_hsl_v2_bands(img: np.ndarray, v2_bands: dict) -> np.ndarray:
    """按 LR 面板顺序 Orange→Blue→Purple 应用 v2 带（各自 hue→sat→lum 复合）。"""
    for color in ("Orange", "Blue", "Purple"):
        if color not in v2_bands:
            continue
        h, s, l_ = v2_bands[color]
        if color == "Blue":
            img = apply_hsl_blue_v2(img, hue=h, sat=s, lum=l_)
        else:
            img = apply_hsl_orange_purple_v2(img, color, hue=h, sat=s, lum=l_)
    return img


def execute_hsl(config, image, runtime_config=None):
    """8 色 HSL：Blue/Orange/Purple 走 LR-faithful v2 表驱动带（不读 hsl.yaml），
    其余 5 色仍走 gimp_stable 复合路径。"""
    all_color_settings = get_hsl_config()

    # LR v2 带（Orange/Blue/Purple；Blue 从 gimp 路径拦截，Orange/Purple 为新增能力）
    v2_bands = {}
    for color in ("Orange", "Blue", "Purple"):
        h = _hsl_band_lr_value(config, f"HueAdjustment{color}")
        s = _hsl_band_lr_value(config, f"SaturationAdjustment{color}")
        l_ = _hsl_band_lr_value(config, f"LuminanceAdjustment{color}")
        if h or s or l_:
            v2_bands[color] = (h, s, l_)

    colors = ["Red", "Yellow", "Green", "Aqua", "Magenta"]
    adjustments_by_color = {}
    for color in colors:
        adjustments_by_color[color] = {
            "HueAdjustment": config.get("HueAdjustment" + color, 0),
            "SaturationAdjustment": config.get("SaturationAdjustment" + color, 0),
            "LuminanceAdjustment": config.get("LuminanceAdjustment" + color, 0),
        }

    hue_adj, sat_adj, light_adj = _build_all_hsl_adjustments(
        all_color_settings, adjustments_by_color
    )
    gimp_active = not (
        np.max(np.abs(hue_adj)) < 1e-8
        and np.max(np.abs(sat_adj)) < 1e-8
        and np.max(np.abs(light_adj)) < 1e-8
    )

    if not v2_bands and not gimp_active:
        return image

    input_dtype = image.dtype
    img = image.astype(np.float32, copy=False)
    if np.issubdtype(input_dtype, np.integer):
        denom = float(np.iinfo(input_dtype).max)
        img = img / max(1.0, denom)
    elif img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    img = _apply_hsl_v2_bands(img, v2_bands)

    out = img
    if gimp_active:
        overlap = config.get("HslOverlap")
        first_cfg = next(iter(all_color_settings.values()), {})
        if overlap is None:
            overlap = first_cfg.get("overlap_default", 0.0)
        overlap = float(np.clip(_safe_float(overlap, 0.0), 0.0, 1.0))

        stability_cfg = _resolve_hsl_stability_cfg(
            first_cfg, stability_override=config.get("HslStability"),
        )

        if _use_torch_backend(runtime_config):
            torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
            out = torch_backend.apply_gimp_hsl_transform_torch(
                img,
                hue_adj,
                sat_adj,
                light_adj,
                overlap,
                stability_cfg=stability_cfg,
                device=resolved_runtime["torch_device"],
            )
        else:
            out = _apply_gimp_hsl_transform(img, hue_adj, sat_adj, light_adj, overlap, stability_cfg)

    if np.issubdtype(input_dtype, np.integer):
        max_value = float(np.iinfo(input_dtype).max)
        out = np.clip(out * max_value, 0.0, max_value)
        return out.astype(input_dtype)
    return np.asarray(out, dtype=np.float32)


def smoothstep(x, edge0, edge1):
    """
    Smoothly interpolate from 0 to 1 as x goes from edge0 to edge1.
    For x < edge0, result = 0.
    For x > edge1, result = 1.
    Between edge0 and edge1, it's a cubic smooth step.
    """
    t = np.clip((x - edge0) / (edge1 - edge0 + 1e-8), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def create_mask(L, lower, upper, softness=0.1):
    """
    Returns a mask in [0..1] which is 0 below 'lower' and 0 above 'upper',
    but transitions smoothly in the boundary regions.
    The 'softness' is a fraction of the (upper-lower) range used for transitions.
    """
    mask = np.zeros_like(L, dtype=np.float32)

    # Ensure lower < upper
    if lower > upper:
        lower, upper = upper, lower

    # How big is the total range?
    span = upper - lower

    # We'll define two "transition" zones:
    #   [lower, lower + softness*span] for ramping from 0..1
    #   [upper - softness*span, upper] for ramping from 1..0
    transition = max(softness * span, 1e-8)

    # Region where we want the mask near 1.0
    # i.e., from (lower + transition) to (upper - transition)
    mid_low = lower + transition
    mid_high = upper - transition

    # 1) Ramp up from 0..1 as L goes from [lower..mid_low]
    ramp_up = smoothstep(L, lower, mid_low)

    # 2) Ramp down from 1..0 as L goes from [mid_high..upper]
    # We'll invert the smoothstep
    ramp_down = 1.0 - smoothstep(L, mid_high, upper)

    # Combine the two
    # In [lower..mid_low], ramp_up goes from 0..1; ramp_down is still 1
    # In [mid_low..mid_high], both ramp_up and ramp_down should be 1
    # In [mid_high..upper], ramp_down goes from 1..0; ramp_up is 1
    mask = np.minimum(ramp_up, ramp_down)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


DEFAULT_TONE_CONTROLS = {
    "model": "selected",
    "radius": 100.0,
    "compress": 0.5,
    "shadows_ccorrect": 1.0,
    "highlights_ccorrect": 0.5,
    "shadows_gain": 1.8,
    "highlights_gain": 1.0,
    "shadows_response_scale": 0.45,
    "highlights_response_scale": 0.8,
    "whites_gain": 0.9,
}
SELECTED_TONE_RT_SHADOW_RADIUS = 40.0
SELECTED_TONE_RT_SHADOW_TONAL_WIDTH = 30.0
SELECTED_TONE_RT_HIGHLIGHT_TONAL_WIDTH = 70.0
RT_EPSILON = 0.075


def _safe_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def resolve_tone_controls(raw_cfg=None):
    """
    Normalize optional tone-control config from JSON/YAML into runtime kwargs.

    Supported keys (and legacy aliases):
      model / tone_model
      radius / tone_radius
      compress / tone_compress
      shadows_ccorrect / tone_shadows_ccorrect
      highlights_ccorrect / tone_highlights_ccorrect
      shadows_gain / highlights_gain / whites_gain
      shadows_response_scale / highlights_response_scale
      shadow_tonal_width_pct
    """
    cfg = dict(raw_cfg or {}) if isinstance(raw_cfg, dict) else {}
    out = dict(DEFAULT_TONE_CONTROLS)

    model = str(cfg.get("model", cfg.get("tone_model", out["model"]))).strip().lower()
    if model == "gegl_like":
        model = "selected"
    if model in {"legacy", "selected"}:
        out["model"] = model

    out["radius"] = max(
        0.1,
        _safe_float(cfg.get("radius", cfg.get("tone_radius", out["radius"])), out["radius"]),
    )

    compress_raw = _safe_float(
        cfg.get("compress", cfg.get("tone_compress", out["compress"])),
        out["compress"],
    )
    if abs(compress_raw) > 1.0:
        compress_raw /= 100.0
    out["compress"] = float(np.clip(compress_raw, 0.0, 0.99))

    for key in ("shadows_ccorrect", "highlights_ccorrect"):
        raw = _safe_float(cfg.get(key, cfg.get(f"tone_{key}", out[key])), out[key])
        if abs(raw) > 1.0:
            raw /= 100.0
        out[key] = float(np.clip(raw, 0.0, 1.0))

    for key in ("shadows_gain", "highlights_gain", "whites_gain"):
        out[key] = float(np.clip(_safe_float(cfg.get(key, out[key]), out[key]), 0.0, 3.0))

    for key in ("shadows_response_scale", "highlights_response_scale"):
        out[key] = float(np.clip(_safe_float(cfg.get(key, out[key]), out[key]), 0.0, 3.0))

    if "shadow_tonal_width_pct" in cfg:
        out["shadow_tonal_width_pct"] = float(
            np.clip(_safe_float(cfg.get("shadow_tonal_width_pct", SELECTED_TONE_RT_SHADOW_TONAL_WIDTH), SELECTED_TONE_RT_SHADOW_TONAL_WIDTH), 1.0, 100.0)
        )

    return out


def _adjust_tones_legacy(img, highlights, shadows, whites):
    """Legacy fixed-mask tone model kept for backward compatibility."""
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    l = 0.299 * r + 0.587 * g + 0.114 * b

    highlights = float(np.clip(highlights, -1.0, 1.0)) * 0.3
    shadows = float(np.clip(shadows, -1.0, 1.0)) * 0.1
    whites = float(np.clip(whites, -1.0, 1.0)) * 0.3

    shadows_mask = create_mask(l, 0.0, 0.65, softness=0.13)
    highlights_mask = create_mask(l, 0.35, 1.0, softness=0.5)
    whites_mask = create_mask(l, 0.65, 1.0, softness=0.3)

    l_shadows = l + shadows * shadows_mask * 0.5
    l_highlights = l_shadows + highlights * highlights_mask * 0.5
    l_whites = l_highlights + whites * whites_mask * 0.5

    ratio = l_whites / (l + 1e-8)
    out = np.stack([r * ratio, g * ratio, b * ratio], axis=-1)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _bilateral_base_from_luma_np(light: np.ndarray, radius: float) -> np.ndarray:
    light = np.asarray(light, dtype=np.float32)
    small_w = max(1, light.shape[1] // 4)
    small_h = max(1, light.shape[0] // 4)
    light_small = cv2.resize(light, (small_w, small_h), interpolation=cv2.INTER_AREA)
    sigma_space = max(2.0, float(radius) * 0.12)
    base_small = cv2.bilateralFilter(
        light_small,
        d=0,
        sigmaColor=0.08,
        sigmaSpace=sigma_space,
    ).astype(np.float32)
    return cv2.resize(base_small, (light.shape[1], light.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _guided_filter_gray_np(
    guide: np.ndarray,
    src: np.ndarray,
    radius: float,
    eps: float,
    subsampling: int = 1,
) -> np.ndarray:
    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)
    if subsampling > 1:
        small_w = max(1, guide.shape[1] // subsampling)
        small_h = max(1, guide.shape[0] // subsampling)
        guide_small = cv2.resize(guide, (small_w, small_h), interpolation=cv2.INTER_AREA)
        src_small = cv2.resize(src, (small_w, small_h), interpolation=cv2.INTER_AREA)
        filtered_small = _guided_filter_gray_np(
            guide_small,
            src_small,
            max(1.0, float(radius) / float(subsampling)),
            eps,
            subsampling=1,
        )
        return cv2.resize(filtered_small, (guide.shape[1], guide.shape[0]), interpolation=cv2.INTER_LINEAR).astype(
            np.float32
        )

    r = max(1, int(round(radius)))
    win = (2 * r + 1, 2 * r + 1)
    mean_i = cv2.boxFilter(guide, -1, win, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, -1, win, borderType=cv2.BORDER_REFLECT)
    corr_i = cv2.boxFilter(guide * guide, -1, win, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src, -1, win, borderType=cv2.BORDER_REFLECT)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, win, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, -1, win, borderType=cv2.BORDER_REFLECT)
    return (mean_a * guide + mean_b).astype(np.float32)



def _apply_gegl_tone_core_np(
    img: np.ndarray,
    *,
    highlights: float,
    shadows: float,
    whites: float,
    base: np.ndarray,
    compress: float,
    shadows_ccorrect: float,
    highlights_ccorrect: float,
    shadows_response_scale: float,
    highlights_response_scale: float,
) -> np.ndarray:
    r = img[..., 0]
    g = img[..., 1]
    b = img[..., 2]
    l = 0.299 * r + 0.587 * g + 0.114 * b

    tb0 = np.clip(1.0 - np.asarray(base, dtype=np.float32), 0.0, 1.0)
    c = float(np.clip(compress, 0.0, 0.99))
    eps = 1e-8
    ta = l.copy()

    h_scaled = 2.0 * float(np.clip(highlights, -1.0, 1.0)) * float(np.clip(highlights_response_scale, 0.0, 3.0))
    if h_scaled != 0.0:
        highlights_xform = np.clip(1.0 - tb0 / (1.0 - c + eps), 0.0, 1.0)
        highlights2 = h_scaled * h_scaled
        sign_neg = -1.0 if h_scaled > 0.0 else 1.0
        while highlights2 > 0.0:
            la = ta
            lb = np.clip((tb0 - 0.5) * sign_neg * np.sign(1.0 - la) + 0.5, 0.0, 1.0)
            chunk = 1.0 if highlights2 > 1.0 else highlights2
            optrans = chunk * highlights_xform
            highlights2 -= 1.0
            mapped = np.where(
                la > 0.5,
                1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb),
                2.0 * la * lb,
            )
            ta = np.clip(la * (1.0 - optrans) + mapped * optrans, 0.0, 1.0)

    s_scaled = 2.0 * float(np.clip(shadows, -1.0, 1.0)) * float(np.clip(shadows_response_scale, 0.0, 3.0))
    if s_scaled != 0.0:
        shadows_xform = np.clip(tb0 / (1.0 - c + eps) - c / (1.0 - c + eps), 0.0, 1.0)
        shadows2 = s_scaled * s_scaled
        sign_pos = 1.0 if s_scaled > 0.0 else -1.0
        while shadows2 > 0.0:
            la = ta
            lb = np.clip((tb0 - 0.5) * sign_pos * np.sign(1.0 - la) + 0.5, 0.0, 1.0)
            chunk = 1.0 if shadows2 > 1.0 else shadows2
            optrans = chunk * shadows_xform
            shadows2 -= 1.0
            mapped = np.where(
                la > 0.5,
                1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb),
                2.0 * la * lb,
            )
            ta = np.clip(la * (1.0 - optrans) + mapped * optrans, 0.0, 1.0)

    if whites != 0.0:
        whites_mask = smoothstep(ta, 0.72, 0.98)
        ta = np.clip(ta + (0.35 * float(np.clip(whites, -1.0, 1.0))) * whites_mask, 0.0, 1.0)

    ratio = ta / (l + 1e-8)
    out = np.stack([r * ratio, g * ratio, b * ratio], axis=-1)
    out = np.clip(out, 0.0, 1.0)

    if h_scaled != 0.0 or s_scaled != 0.0:
        hsv = cv2.cvtColor(out.astype(np.float32), cv2.COLOR_RGB2HSV)
        sat = hsv[..., 1]
        sh_mask = np.clip(tb0 / (1.0 - c + eps) - c / (1.0 - c + eps), 0.0, 1.0)
        hi_mask = np.clip(1.0 - tb0 / (1.0 - c + eps), 0.0, 1.0)
        sh_scale = (float(np.clip(shadows_ccorrect, 0.0, 1.0)) - 0.5) * abs(s_scaled) * 0.7
        hi_scale = (float(np.clip(highlights_ccorrect, 0.0, 1.0)) - 0.5) * abs(h_scaled) * 0.7
        sat *= 1.0 + sh_scale * sh_mask + hi_scale * hi_mask
        hsv[..., 1] = np.clip(sat, 0.0, 1.0)
        hsv[..., 2] = np.clip(hsv[..., 2], 0.0, 1.0)
        out = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
        out = np.clip(out, 0.0, 1.0)

    return out.astype(np.float32)


def _adjust_tones_gegl_like(
    img,
    highlights,
    shadows,
    whites,
    radius,
    compress,
    shadows_ccorrect,
    highlights_ccorrect,
    shadows_gain,
    highlights_gain,
    shadows_response_scale,
    highlights_response_scale,
    whites_gain,
):
    """GEGL-inspired tone model with the historical MonetGPT gain profile."""
    h_val = float(np.clip(highlights, -1.0, 1.0)) * float(highlights_gain)
    s_val = float(np.clip(shadows, -1.0, 1.0)) * float(shadows_gain)
    w_val = float(np.clip(whites, -1.0, 1.0)) * float(whites_gain)
    if h_val == 0.0 and s_val == 0.0 and w_val == 0.0:
        return img.astype(np.float32, copy=True)

    light = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    sigma = max(0.5, float(radius) * 0.05)
    base = gaussian_filter(light, sigma=sigma)
    return _apply_gegl_tone_core_np(
        img,
        highlights=h_val,
        shadows=s_val,
        whites=w_val,
        base=base,
        compress=compress,
        shadows_ccorrect=shadows_ccorrect,
        highlights_ccorrect=highlights_ccorrect,
        shadows_response_scale=shadows_response_scale,
        highlights_response_scale=highlights_response_scale,
    )



def _adjust_tones_selected_np(
    img: np.ndarray,
    highlights: float,
    shadows: float,
    whites: float,
    *,
    resolved: dict,
) -> np.ndarray:
    out = img.astype(np.float32, copy=True)
    radius = float(resolved["radius"])
    compress = float(resolved["compress"])

    if abs(highlights) > 1e-8:
        light = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
        out = _apply_gegl_tone_core_np(
            out,
            highlights=highlights,
            shadows=0.0,
            whites=0.0,
            base=_bilateral_base_from_luma_np(light, radius),
            compress=compress,
            shadows_ccorrect=resolved["shadows_ccorrect"],
            highlights_ccorrect=resolved["highlights_ccorrect"],
            shadows_response_scale=1.0,
            highlights_response_scale=1.0,
        )

    if abs(shadows) > 1e-8:
        # LR-faithful Shadows v2：图像自适应乘性 luma 曲线（正负向统一；
        # 旧 _adjust_shadows_rt_guided_np / _apply_gegl_tone_core_np 双路废弃）。
        out = _apply_shadows_v2(out, float(np.clip(shadows, -1.0, 1.0)) * 100.0)

    if abs(whites) > 1e-8:
        out = _adjust_tones_gegl_like(
            out,
            0.0,
            0.0,
            whites,
            radius=radius,
            compress=compress,
            shadows_ccorrect=resolved["shadows_ccorrect"],
            highlights_ccorrect=resolved["highlights_ccorrect"],
            shadows_gain=1.0,
            highlights_gain=1.0,
            shadows_response_scale=1.0,
            highlights_response_scale=1.0,
            whites_gain=resolved["whites_gain"],
        )

    return out.astype(np.float32)


def adjust_tones(
    image,
    highlights=0.0,
    shadows=0.0,
    whites=0.0,
    model="selected",
    radius=100.0,
    compress=0.5,
    shadows_ccorrect=1.0,
    highlights_ccorrect=0.5,
    shadows_gain=1.8,
    highlights_gain=1.0,
    shadows_response_scale=0.45,
    highlights_response_scale=0.8,
    shadow_tonal_width_pct=30.0,
    whites_gain=0.9,
    runtime_config=None,
):
    """
    Adjust highlights/shadows/whites with selectable tone model.

    Inputs are normalized slider values in [-1, 1]. For model selection:
      - "selected": default production family split
      - "legacy": previous fixed-mask additive model
    """
    highlights, shadows, whites, resolved = _resolve_tone_execution_payloads(
        highlights,
        shadows,
        whites,
        model=model,
        radius=radius,
        compress=compress,
        shadows_ccorrect=shadows_ccorrect,
        highlights_ccorrect=highlights_ccorrect,
        shadows_gain=shadows_gain,
        highlights_gain=highlights_gain,
        shadows_response_scale=shadows_response_scale,
        highlights_response_scale=highlights_response_scale,
        shadow_tonal_width_pct=shadow_tonal_width_pct,
        whites_gain=whites_gain,
    )
    if abs(highlights) < 1e-8 and abs(shadows) < 1e-8 and abs(whites) < 1e-8:
        return image

    img = image.astype(np.float32, copy=False)
    input_dtype = image.dtype
    if img.max() > 1.0:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    resolved_model = str(resolved["model"]).strip().lower()

    # Shadows 已替换为 LR-faithful v2（纯 numpy）；torch 后端仍是旧算法，
    # shadows 非零时整体回落 numpy 路径以避免 CPU/GPU 结果分叉。
    use_torch_runtime = (
        _use_torch_backend(runtime_config)
        and resolved_model != "legacy"
        and abs(shadows) < 1e-8
    )
    if use_torch_runtime:
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        out = torch_backend.adjust_tones_torch(
            img,
            highlights=highlights,
            shadows=shadows,
            whites=whites,
            model=resolved_model,
            radius=resolved["radius"],
            compress=resolved["compress"],
            shadows_ccorrect=resolved["shadows_ccorrect"],
            highlights_ccorrect=resolved["highlights_ccorrect"],
            shadows_gain=resolved["shadows_gain"],
            highlights_gain=resolved["highlights_gain"],
            shadows_response_scale=resolved["shadows_response_scale"],
            highlights_response_scale=resolved["highlights_response_scale"],
            shadow_tonal_width_pct=resolved["shadow_tonal_width_pct"],
            whites_gain=resolved["whites_gain"],
            device=resolved_runtime["torch_device"],
        )
        if input_dtype == np.uint8:
            return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    if resolved_model == "legacy":
        out = _adjust_tones_legacy(img, highlights, shadows, whites)
    elif resolved_model == "selected":
        out = _adjust_tones_selected_np(
            img,
            highlights,
            shadows,
            whites,
            resolved=resolved,
        )
    else:
        out = _adjust_tones_selected_np(
            img,
            highlights,
            shadows,
            whites,
            resolved=resolved,
        )

    if input_dtype == np.uint8:
        return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def adjust_saturation(image, intensity, runtime_config=None):
    """
    Apply canonical GEGL-style native saturation.

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to `saturation_pct = value * 100`. Dict inputs may pass canonical
    `{"saturation_pct": ...}` directly.
    """
    saturation_pct = _resolve_saturation_params(intensity)
    if abs(saturation_pct) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_saturation_torch(
            img,
            intensity,
            device=resolved_runtime["torch_device"],
        )

    return _apply_saturation_np(img, saturation_pct)


def adjust_contrast(image, intensity, runtime_config=None):
    """
    Apply the selected contrast implementation.

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to `contrast_factor = 1 + value`. Dict inputs may pass canonical
    `{"contrast_factor": ..., "brightness": ...}` directly.
    """
    contrast_factor, brightness = _resolve_contrast_params(intensity)
    if abs(contrast_factor - 1.0) < 1e-8 and abs(brightness) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_contrast_torch(
            img,
            intensity,
            device=resolved_runtime["torch_device"],
        )

    return _apply_contrast_np(img, contrast_factor, brightness)


def adjust_temperature(image, intensity, runtime_config=None):
    """LR-faithful temperature（LR IncrementalTemperature；wb_cat_v3 标定模型）。

    Scalar inputs are interpreted as local compatibility units in [-1, 1],
    mapped to LR slider units via `value * 100`. Dict inputs may pass legacy
    canonical `{"delta_kelvin": ...}`（±5000K ↔ LR ±100 线性折算）。
    torch 后端镜像的是旧 exp 增益算法（结果分叉），统一走 numpy v2 实现。
    """
    del runtime_config
    if isinstance(intensity, dict):
        _, delta_kelvin = _resolve_temperature_params(intensity)
        lr_value = float(np.clip(delta_kelvin / 50.0, -100.0, 100.0))
    else:
        lr_value = float(np.clip(float(intensity), -1.0, 1.0)) * 100.0
    if abs(lr_value) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    return apply_lr_temperature(img, lr_value)


def adjust_blacks(image, intensity, runtime_config=None):
    """
    Adjust blacks via a HISTOGRAM-VALUE spline curve.
    Replicates GIMP's gimp-drawable-curves-spline with 3 control points.

    intensity: float in [-1.0, 1.0]
        -1.0 = deeper/crushed blacks
         0.0 = no change
         1.0 = lifted blacks
    """
    if isinstance(intensity, dict):
        amount = _invert_piecewise_table(
            _LR2DT_BLACKS_TABLE,
            float(intensity.get("black_level_offset", 0.0)),
        ) / 100.0
    else:
        amount = float(np.clip(intensity, -1.0, 1.0))
    if abs(amount) < 1e-8:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)

    if _use_torch_backend(runtime_config):
        torch_backend, resolved_runtime = _get_torch_backend_module(runtime_config)
        return torch_backend.adjust_blacks_torch(
            img,
            intensity,
            device=resolved_runtime["torch_device"],
        )

    from .curve import build_blacks_lut, _apply_lut_float01

    lut = build_blacks_lut(amount)

    # Apply on value channel (HISTOGRAM-VALUE semantics).
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 2] = _apply_lut_float01(hsv[..., 2], lut)
    hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 1.0)
    hsv[..., 2] = np.clip(hsv[..., 2], 0.0, 1.0)
    result = cv2.cvtColor(hsv.astype(np.float32), cv2.COLOR_HSV2RGB)
    return np.clip(result, 0.0, 1.0).astype(np.float32)



# ============================================================================
# LR-faithful ops v2（由 tools/lr_calib/ops_v2/* 合入；LR GT 标定，2026-07）
# 约定：全部函数输入/输出 float32 [0,1] sRGB RGB；滑杆值为 LR 原生单位。
# 标定常数与 fits/*.json 拟合结果烘焙为模块常量，运行时无文件依赖。
# 再生成请用 tools/lr_calib/scratch 下对应 fit 脚本，勿手改数值。
# ============================================================================

# ---- Vignette（PostCropVignette 全参数 + 镜头暗角；LR 符号：负=压暗）----
_LUMA_W = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0).astype(np.float32, copy=False)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)


# ---------------------------------------------------------------------------
# 拟合常数（scratch/vignette_v2_fit.py 输出；勿手改）
# ---------------------------------------------------------------------------
VIG_CONSTS: dict = {
    'profile_t': [0.0, 0.08, 0.18, 0.3, 0.42, 0.55, 0.68, 0.82, 1.0],
    'profile_c': [0.0, 0.00499, 0.0118, 0.03255, 0.07746, 0.17058, 0.36694, 0.70475, 1.0],
    'u0_mid': [[10.0, 0.00311], [50.0, 0.24165], [90.0, 0.58047]],
    'w_fea': [[10.0, 0.10666], [50.0, 0.66617], [90.0, 0.92683]],
    'u0_fea_delta': [[10.0, 0.39323], [50.0, 0.0], [90.0, -0.13016]],
    'w_mid_delta': [[10.0, 0.10407], [50.0, 0.0], [90.0, -0.20853]],
    'round_pos_k': 0.72426,
    'round_pos_gamma': 0.86131,
    'round_u0_delta_pos': 0.0703,
    'round_w_delta_pos': -0.05125,
    'round_neg_alpha': 1.19229,
    'round_neg_beta': 0.60112,
    'round_u0_delta_neg': 0.18795,
    'round_w_delta_neg': -0.0525,
    'dark_v': [[0.0, 0.0], [40.0, 0.19532], [80.0, 0.48913], [100.0, 0.63604]],
    'dark_pd': 2.98346,
    'dark_prot_h': 0.46115,
    'dark_prot_q': 5.08336,
    'bright_v': [[0.0, 0.0], [40.0, 0.25297], [80.0, 1.0], [100.0, 1.0]],
    'bright_beta': 1.63354,
    'style2_v_scale': 0.98407,
    'lens_profile_t': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    'lens_profile_c': [0.0, 0.00057, 0.00364, 0.05572, 0.33519, 1.0],
    'lens_amp': [[-100.0, -1.82348], [-80.0, -1.45878], [0.0, 0.0], [80.0, 1.56295], [100.0, 1.95369]],
}


def _anchor_interp(x: float, anchors: list) -> float:
    xs, ys = zip(*anchors)
    return float(np.interp(x, xs, ys))


def _vig_field(height: int, width: int, roundness: float = 0.0,
               consts: dict = VIG_CONSTS) -> np.ndarray:
    """归一化暗角距离场 u：中心 0，角点 1。roundness ∈ [-100,100]（LR 值域）。

    roundness>0：框贴合椭圆 与 绝对纵横比正圆 混合后再幂锐化（LR 实测形状）；
    roundness<0：d_box^alpha·d_ell^beta 积式——等值线贴边框、对角内域塌缩。
    """
    yy = np.linspace(-1.0, 1.0, num=height, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, num=width, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    d_ell = np.sqrt(gx * gx + gy * gy) / np.float32(np.sqrt(2.0))   # 框贴合椭圆
    rho = float(np.clip(roundness / 100.0, -1.0, 1.0))
    if rho == 0.0:
        return d_ell.astype(np.float32)
    if rho > 0:
        # 向绝对纵横比正圆混合 + 幂锐化
        mn = float(max(1, min(height, width)))
        sx, sy = width / mn, height / mn
        norm = np.float32(np.sqrt(sx * sx + sy * sy))
        d_cir = np.sqrt((gx * sx) ** 2 + (gy * sy) ** 2) / norm
        s = rho / 0.7
        k = min(1.0, s * float(consts["round_pos_k"]))
        g = 1.0 + s * (float(consts["round_pos_gamma"]) - 1.0)
        d = ((1.0 - k) * d_ell + k * d_cir) ** g
    else:
        d_box = np.maximum(np.abs(gx), np.abs(gy))
        s = -rho / 0.7
        alpha = s * float(consts["round_neg_alpha"])
        beta = 1.0 + s * (float(consts["round_neg_beta"]) - 1.0)
        d = d_box ** alpha * d_ell ** beta
    return d.astype(np.float32)


def _vig_mask(u: np.ndarray, midpoint: float = 50.0, feather: float = 50.0,
              consts: dict = VIG_CONSTS, roundness: float = 0.0) -> np.ndarray:
    """空间 mask ∈ [0,1]：中心平台 0，角点方向陡升到 1。

    midpoint 主导起点 u0、feather 主导过渡宽度 w，双向各有轻微交叉耦合（*_delta）；
    roundness 对 u0/w 也有实测偏移（round_*_delta，按 |roundness|/70 比例）。
    """
    rho = float(np.clip(roundness / 100.0, -1.0, 1.0))
    s = abs(rho) / 0.7
    sfx = "neg" if rho < 0 else "pos"
    u0 = (_anchor_interp(float(midpoint), consts["u0_mid"])
          + _anchor_interp(float(feather), consts["u0_fea_delta"])
          + s * float(consts.get(f"round_u0_delta_{sfx}", 0.0)))
    w = (_anchor_interp(float(feather), consts["w_fea"])
         + _anchor_interp(float(midpoint), consts["w_mid_delta"])
         + s * float(consts.get(f"round_w_delta_{sfx}", 0.0)))
    w = max(1e-3, w)
    t = np.clip((u - u0) / w, 0.0, 1.0)
    return np.interp(t, consts["profile_t"], consts["profile_c"]).astype(np.float32)


def _darken_hp(img: np.ndarray, v: float, mask: np.ndarray,
               consts: dict = VIG_CONSTS) -> np.ndarray:
    """Highlight Priority 压暗：显示域逐通道减法 + 亮度保护。"""
    pd = float(consts["dark_pd"])
    h = float(consts["dark_prot_h"])
    q = float(consts["dark_prot_q"])
    luma = np.clip(img @ _LUMA_W, 0.0, 1.0)
    prot = 1.0 - h * luma ** q
    sub = (v * mask * prot).astype(np.float32)
    if pd == 1.0:
        out = img - sub[..., None]
    else:
        e = np.clip(img, 0.0, 1.0) ** (1.0 / pd)
        out = np.clip(e - sub[..., None], 0.0, 1.0) ** pd
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _brighten_hp(img: np.ndarray, v: float, mask: np.ndarray,
                 consts: dict = VIG_CONSTS) -> np.ndarray:
    """提亮：线性光域 screen 混合（暗部大增益、高光软压缩）。"""
    beta = float(consts["bright_beta"])
    lin = _srgb_to_linear(img)
    veff = np.clip(v * mask ** beta, 0.0, 1.0).astype(np.float32)
    out = 1.0 - (1.0 - lin) * (1.0 - veff[..., None])
    return _linear_to_srgb(out)


def _darken_cp(img: np.ndarray, v: float, mask: np.ndarray,
               consts: dict = VIG_CONSTS) -> np.ndarray:
    """Color Priority 压暗：亮度按 HP 模型压暗，线性域等比缩放 RGB 保色度。"""
    pd = float(consts["dark_pd"])
    h = float(consts["dark_prot_h"])
    q = float(consts["dark_prot_q"])
    luma = np.clip(img @ _LUMA_W, 1e-4, 1.0).astype(np.float32)
    prot = 1.0 - h * luma ** q
    sub = (v * float(consts["style2_v_scale"]) * mask * prot).astype(np.float32)
    l2 = np.clip(luma ** (1.0 / pd) - sub, 0.0, 1.0) ** pd
    gain = _srgb_to_linear(l2) / np.maximum(_srgb_to_linear(luma), 1e-6)
    out = _srgb_to_linear(img) * gain[..., None]
    return _linear_to_srgb(out)


def apply_postcrop_vignette(img: np.ndarray, amount: float, midpoint: float = 50.0,
                            feather: float = 50.0, roundness: float = 0.0,
                            style: int = 1, consts: dict = VIG_CONSTS) -> np.ndarray:
    """LR PostCropVignette。amount ∈ [-100,100]，负=压暗四角（LR 符号约定）。"""
    amount = float(np.clip(amount, -100.0, 100.0))
    if amount == 0.0:
        return img.astype(np.float32, copy=False)
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    u = _vig_field(img.shape[0], img.shape[1], roundness=roundness, consts=consts)
    mask = _vig_mask(u, midpoint=midpoint, feather=feather, consts=consts,
                     roundness=roundness)
    if amount < 0:
        v = _anchor_interp(-amount, consts["dark_v"])
        if int(style) == 2:
            return _darken_cp(img, v, mask, consts)
        return _darken_hp(img, v, mask, consts)
    v = _anchor_interp(amount, consts["bright_v"])
    return _brighten_hp(img, v, mask, consts)


def apply_lens_vignette(img: np.ndarray, amount: float, midpoint: float = 50.0,
                        consts: dict = VIG_CONSTS) -> np.ndarray:
    """LR 镜头暗角（手动 Lens Vignetting）。线性光域径向增益 2^(A·P(u))。"""
    amount = float(np.clip(amount, -100.0, 100.0))
    if amount == 0.0:
        return img.astype(np.float32, copy=False)
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    h, w = img.shape[:2]
    yy = np.linspace(-1.0, 1.0, num=h, dtype=np.float32)
    xx = np.linspace(-1.0, 1.0, num=w, dtype=np.float32)
    gx, gy = np.meshgrid(xx, yy)
    mn = float(max(1, min(h, w)))
    sx, sy = w / mn, h / mn
    u = np.sqrt((gx * sx) ** 2 + (gy * sy) ** 2) / np.float32(np.sqrt(sx * sx + sy * sy))
    u = np.clip(u * (50.0 / max(1.0, float(midpoint))) ** 0.0, 0.0, 1.0)  # midpoint 未标定，恒等
    p = np.interp(u, consts["lens_profile_t"], consts["lens_profile_c"]).astype(np.float32)
    amp = _anchor_interp(amount, consts["lens_amp"])
    lin = _srgb_to_linear(img) * (2.0 ** (amp * p))[..., None]
    return _linear_to_srgb(lin)

# ---- Temperature（LR IncrementalTemperature；wb_cat_v3 模型）----
# --- 色彩空间常量（sRGB D65 → linear ProPhoto D50，Bradford） ---
_M_RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
_M_D65toD50 = np.array([[1.0478112, 0.0228866, -0.0501270],
                        [0.0295424, 0.9904844, -0.0170491],
                        [-0.0092345, 0.0150436, 0.7521316]])
_M_XYZ2PP = np.array([[1.3459433, -0.2556075, -0.0511118],
                      [-0.5445989, 1.5081673, 0.0205351],
                      [0.0000000, 0.0000000, 1.2118128]])
M_SRGB2PP = (_M_XYZ2PP @ _M_D65toD50 @ _M_RGB2XYZ).astype(np.float64)
M_PP2SRGB = np.linalg.inv(M_SRGB2PP)
Y_PP = np.linalg.inv(_M_XYZ2PP)[1]           # ProPhoto -> Y (D50 亮度)


def _srgb_to_lin(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def _interp_model(value: float, model: dict):
    """扫描值间逐元素线性插值 (M_shared, W)；v=0 处 M=I, W=0。"""
    vs = sorted(int(k) for k in model["shared_M"])
    knots = sorted(set(vs) | {0})
    Ms, Ws = [], []
    for k in knots:
        if k == 0 and str(k) not in model["shared_M"]:
            Ms.append(np.eye(3)); Ws.append(np.zeros((4, 3)))
        else:
            Ms.append(np.asarray(model["shared_M"][str(k)], dtype=np.float64))
            Ws.append(np.asarray(model["values"][str(k)]["W"], dtype=np.float64))
    v = float(np.clip(value, knots[0], knots[-1]))
    i = int(np.searchsorted(knots, v, side="right")) - 1
    i = min(max(i, 0), len(knots) - 2)
    t = (v - knots[i]) / (knots[i + 1] - knots[i])
    M = (1 - t) * Ms[i] + t * Ms[i + 1]
    W = (1 - t) * Ws[i] + t * Ws[i + 1]
    return M, W


def _image_feats(pp: np.ndarray, B: np.ndarray) -> np.ndarray:
    """before 图统计特征：灰世界 + 亮区（亮度 60–95 分位）在基 B 下的 log 色度。"""
    flat = pp.reshape(-1, 3)
    if flat.shape[0] > 200000:                      # 统计用降采样，结果不敏感
        flat = flat[:: flat.shape[0] // 200000 + 1]
    lms = np.clip(flat @ B.T, 1e-7, None)
    lum = flat @ Y_PP
    gw = lms.mean(0)
    lo, hi = np.quantile(lum, [0.60, 0.95])
    sel = (lum >= lo) & (lum <= hi)
    br = lms[sel].mean(0) if sel.sum() > 50 else gw
    return np.array([np.log(gw[0] / gw[1]), np.log(gw[2] / gw[1]),
                     np.log(br[0] / br[1]), np.log(br[2] / br[1])])


def apply_lr_temperature(img: np.ndarray, value: float, model: dict | None = None) -> np.ndarray:
    """LR IncrementalTemperature 的本地复现。img: float [0,1] sRGB HxWx3。"""
    if abs(value) < 1e-9:
        return img.astype(np.float32)
    if model is None:
        model = _lr_temperature_model()
    grid = np.asarray(model["grid_log10x"], dtype=np.float64)
    u = np.asarray(model["u"], dtype=np.float64)
    shoulder = float(model.get("shoulder", 0.0))
    B = np.asarray(model["B"], dtype=np.float64)
    Bi = np.linalg.inv(B)

    M_sh, W = _interp_model(value, model)
    pp = np.clip(_srgb_to_lin(np.clip(img.astype(np.float64), 0.0, 1.0)) @ M_SRGB2PP.T,
                 1e-7, None)
    f = _image_feats(pp, B) - np.asarray(model["feat_center"], dtype=np.float64)
    d = f @ W                                        # 逐图自适应残差 diag（log 增益）
    M = Bi @ np.diag(np.exp(d)) @ B @ M_sh

    scene = np.exp(np.interp(np.log10(np.clip(pp, 10.0 ** grid[0], 1.0)), grid, u))
    scene2 = np.clip(scene @ M.T, 1e-9, None)
    u2 = np.log(scene2)
    if shoulder > 1e-4:                              # 高光软肩
        knee = u[-1] - shoulder
        u2 = np.where(u2 > knee, knee + shoulder * np.tanh((u2 - knee) / shoulder), u2)
    pp2 = 10.0 ** np.interp(u2, u, grid)
    dark = pp < 10.0 ** grid[0]                      # 极暗端：线性直通（u 在 toe 恒等）
    pp2 = np.where(dark, np.clip(pp @ M.T, 0.0, None), pp2)
    return _lin_to_srgb(pp2 @ M_PP2SRGB.T).astype(np.float32)


def apply_lr_abs_temperature(img: np.ndarray) -> np.ndarray:
    """LR 绝对 kelvin WB（WhiteBalance=Custom + Temperature/Tint）对 JPEG = no-op（实测）。"""
    return img.astype(np.float32)

_TEMPERATURE_V2_MODEL_JSON = r'''{"model": "wb_cat_v3", "note": "LR IncrementalTemperature 重写模型；由 ops_v2/temperature.py 消费。非声明式 value_map/post_lut 词汇。", "grid_log10x": [-3.6989700043360187, -3.620268514882061, -3.541567025428103, -3.4628655359741454, -3.3841640465201874, -3.3054625570662295, -3.2267610676122715, -3.148059578158314, -3.069358088704356, -2.990656599250398, -2.9119551097964402, -2.8332536203424823, -2.7545521308885244, -2.675850641434567, -2.597149151980609, -2.5184476625266514, -2.4397461730726935, -2.3610446836187355, -2.2823431941647776, -2.2036417047108197, -2.1249402152568617, -2.046238725802904, -1.9675372363489463, -1.8888357468949883, -1.8101342574410304, -1.7314327679870727, -1.652731278533115, -1.574029789079157, -1.495328299625199, -1.4166268101712411, -1.3379253207172837, -1.2592238312633257, -1.1805223418093678, -1.1018208523554098, -1.023119362901452, -0.9444178734474944, -0.8657163839935365, -0.7870148945395785, -0.7083134050856206, -0.6296119156316631, -0.5509104261777051, -0.4722089367237472, -0.39350744726978926, -0.3148059578158313, -0.23610446836187382, -0.15740297890791588, -0.07870148945395794, 0.0], "u": [-8.607578648287134, -8.417028758784848, -8.163725759543825, -7.864248995280116, -7.460124483122478, -6.945283554563353, -6.506845279057353, -6.246262686243358, -6.0388621248752505, -5.891600799501831, -5.811693645894674, -5.761983559762388, -5.719814765461305, -5.667858464483887, -5.609746584967367, -5.5510272925081825, -5.495235313541167, -5.441286131799001, -5.389312917977651, -5.340516812635069, -5.295047244740625, -5.251847723467008, -5.209876709606252, -5.168701527485039, -5.128754623931652, -5.090409397119583, -5.054299617773892, -5.019801378024397, -4.985150575106744, -4.949272305009441, -4.912876520143967, -4.876169998134625, -4.839354225317457, -4.802412718117852, -4.764946816274369, -4.726516889709606, -4.687031092606504, -4.647159818721895, -4.607935841564348, -4.570985350873408, -4.532243289254017, -4.486911487907194, -4.435557459160646, -4.374814710442844, -4.29886115674694, -4.194064834879203, -4.059839749889158, -3.9038766663559685], "shoulder": 0.2, "B": [[0.7907327697502091, 0.3106533515505385, -0.10510164809160848], [-0.10485876257748608, 1.1183754705757611, 0.006910743782432185], [0.01129879774100603, -0.04350441927524606, 0.8508499575929863]], "feat_center": [-0.007918063092019326, -0.24868330560397814, -0.001970937983522431, -0.2522589100177815], "shared_M": {"-100": [[0.10403419269304531, 0.7607497598256265, 0.361856099388314], [-0.049804960405422344, 1.0782494101982785, 0.17988569552744893], [-1.3168507002280583, 1.622654398288141, 1.7779034679577483]], "-60": [[0.8067461493311066, -0.036188091660527597, 0.29233495985440494], [0.022595033585195726, 1.028186857307415, 0.03329176141475061], [-0.1035659771455178, 0.047632323468854626, 1.5052550648373835]], "-30": [[0.9703034801936736, -0.08551721969439147, 0.12790519195960875], [0.041461747542376776, 0.9764694571345034, 0.015432677689367028], [-0.0894852581477778, 0.11420848146221457, 1.1517879297517237]], "30": [[1.1377912177987968, 0.043015482780644376, -0.04168519854952033], [-0.11253907856841108, 1.2109464745165972, 0.012548407233468666], [-0.02095248628250438, 0.023252225334732108, 0.997433740032818]], "60": [[1.1367722900088753, 0.20808852452330207, -0.08737833068886647], [-0.21822261043422664, 1.393689738662184, 0.02759823592376416], [-0.01796785562926915, 0.047477221435475, 0.9789323620032766]], "100": [[1.116723158126987, 0.2143474859137265, 0.03444189923708754], [-0.37670789102059943, 1.531654064822054, 0.13625358973860296], [-0.07368086013378528, 0.18008348420410314, 0.9175002930216809]]}, "values": {"-100": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, "-60": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, "-30": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}, "30": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.0011102300149095996, -0.0010488786088562213, 0.00038008108520753536], [-0.0003681760889759357, 0.0018967790824844582, 0.000512700044676185]]}, "60": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.004254637215214807, -0.0041030639232151654, -0.001856204185029366], [0.00030305461486099006, 0.003139046492350321, -0.006794779224090748]]}, "100": {"W": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}}}'''   # fits/Temperature.json 烘焙
_TEMPERATURE_V2_MODEL: dict | None = None


def _lr_temperature_model() -> dict:
    global _TEMPERATURE_V2_MODEL
    if _TEMPERATURE_V2_MODEL is None:
        _TEMPERATURE_V2_MODEL = json.loads(_TEMPERATURE_V2_MODEL_JSON)
    return _TEMPERATURE_V2_MODEL

# ---- Vibrance（LR-faithful；Lab 色度乘性增益 + 肤色带压低 + 负向灰目标插值）----
_VIB_C_NODES = np.array([0.0, 3.0, 7.0, 12.0, 17.0, 25.0, 35.0, 45.0, 55.0, 70.0, 90.0])
_VIB_H_NODES = np.arange(0.0, 360.0, 30.0)          # Lab 色相（度）
_VIB_L_NODES = np.array([0.0, 25.0, 40.0, 55.0, 70.0, 85.0, 100.0])   # L* 节点
_VIB_AMTS = [30.0, 60.0, 100.0]                     # 标定幅度节点（|pct|）
# 达标参数（fits/Vibrance.json v2_params 烘焙；模块旧粗初值已废弃）
_VIB_DEFAULTS: dict = json.loads(r'''{"pos": {"amts": [30.0, 60.0, 100.0], "k": [[0.04689070080000001, 0.1408341504, 0.268509312, 0.25789977599999997, 0.25776, 0.236928, 0.19345855999999997, 0.18810001407999996, 0.1429856, 0.10368986111999998, 0.009], [0.13977321676799997, 0.615320559104, 0.6807723161600001, 0.6661864038400002, 0.6523652, 0.56680494208, 0.46197138519040004, 0.5101417225359359, 0.3955246158336, 0.15221188884480003, 0.012], [0.8792926638080001, 1.9730139944960006, 2.02402046464, 1.89745116416, 1.7680059224, 1.4527591012736, 1.0248109056, 1.1228706738175998, 0.655829256603648, 0.17218218844159996, 0.016]], "mh": [[0.1499999999999999, 0.0, 0.32500000000000007, 1.0, 0.75, 0.975, 1.0, 0.8500000000000001, 0.85, 0.85, 0.4499999999999999, 0.6000000000000001], [0.29999999999999993, 0.15000000000000002, 0.425, 1.0, 0.825, 1.05, 0.9750000000000001, 0.9500000000000002, 0.85, 0.8749999999999999, 0.6499999999999999, 0.8], [0.32499999999999996, 0.25000000000000006, 0.45, 1.0, 0.7999999999999999, 1.1, 0.95, 0.9, 0.75, 0.7999999999999999, 0.775, 0.85]], "dl": [[-0.2, 0.1, 0.2, 0.65, 1.0999999999999999, 0.2, 0.25, -0.7000000000000001, -1.5, -2.3, -3.0, -0.7], [-0.8999999999999999, 0.2, 0.35, 1.2999999999999998, 1.6, 0.3, 0.25, -1.5499999999999998, -4.1499999999999995, -5.65, -5.8999999999999995, -1.7], [-2.4, -0.24999999999999997, 0.5499999999999999, 1.25, 2.05, -0.7, 0.44999999999999996, -4.200000000000001, -7.899999999999999, -13.75, -10.2, -5.2]], "dl_ramp": [[-2.0, 29.0], [-4.0, 27.0], [-4.0, 19.0]], "ml": [[0.21999999999999995, 0.85, 1.09, 1.0, 1.12, 0.7899999999999999, 0.6699999999999999], [0.21999999999999995, 0.77, 1.11, 1.0, 1.08, 0.7899999999999999, 0.6699999999999999], [0.21999999999999995, 0.6499999999999999, 1.25, 1.0, 1.16, 0.7299999999999999, 0.6699999999999999]]}, "neg": {"amts": [30.0, 60.0, 100.0], "r": [[0.77, 0.63, 0.637, 0.639, 0.644, 0.671, 0.6900000000000001, 0.701, 0.722, 0.785, 0.859], [0.11999999999999998, 0.337, 0.352, 0.35500000000000004, 0.38100000000000006, 0.40099999999999997, 0.465, 0.491, 0.511, 0.555, 0.5860000000000001], [0.0, 0.0, 0.0, 0.063, 0.11299999999999999, 0.158, 0.228, 0.275, 0.294, 0.343, 0.382]], "beta": [-0.3, -0.25999999999999995, -0.07999999999999996], "gs": [[0.4, 0.41000000000000003, 0.42, 0.5, 0.43, -0.26, -0.1, 0.27999999999999997, 0.1, -0.16, 0.26, 0.3], [0.53, 0.54, 0.61, 0.72, 0.58, -0.03999999999999999, -0.03, 0.43, 0.16, 0.0, 0.33, 0.4], [0.78, 0.81, 0.7999999999999999, 1.01, 0.88, 0.27999999999999997, 0.20999999999999996, 0.62, 0.31, 0.22, 0.43, 0.58]], "gsc": [[17.5, 45.0], [11.25, 50.0], [13.75, 55.0]]}}''')

def _vib_row(amt: float, amts, rows, identity_row) -> np.ndarray:
    """幅度节点间的参数行分段线性插值；amt<amts[0] 时向 identity 行过渡。"""
    rows = np.asarray(rows, dtype=np.float64)
    amts = list(amts)
    if amt <= amts[0]:
        w = amt / amts[0]
        return np.asarray(identity_row, dtype=np.float64) * (1.0 - w) + rows[0] * w
    if amt >= amts[-1]:
        return rows[-1]
    j = int(np.searchsorted(amts, amt))
    w = (amt - amts[j - 1]) / (amts[j] - amts[j - 1])
    return rows[j - 1] * (1.0 - w) + rows[j] * w


def _vib_hue_interp(h_deg: np.ndarray, vals: np.ndarray) -> np.ndarray:
    """周期色相剖面插值（节点 _VIB_H_NODES，360° 回绕）。"""
    xs = np.concatenate([_VIB_H_NODES, [360.0]])
    ys = np.concatenate([vals, [vals[0]]])
    return np.interp(h_deg % 360.0, xs, ys)


def apply_vibrance_lr(image: np.ndarray, vibrance_pct: float, params: dict | None = None) -> np.ndarray:
    """LR-faithful Vibrance。image: float [0,1] RGB；vibrance_pct: [-100,100]。"""
    pct = float(np.clip(vibrance_pct, -100.0, 100.0))
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if abs(pct) < 1e-8:
        return img
    p = params or _VIB_DEFAULTS
    lab = color.rgb2lab(img)
    chroma = np.hypot(lab[..., 1], lab[..., 2])

    if pct > 0.0:
        pp = p["pos"]
        k_nodes = _vib_row(pct, pp["amts"], pp["k"], np.zeros(len(_VIB_C_NODES)))
        mh_nodes = _vib_row(pct, pp["amts"], pp["mh"], pp["mh"][0])
        dl_nodes = _vib_row(pct, pp["amts"], pp["dl"], np.zeros(len(_VIB_H_NODES)))
        ramp = _vib_row(pct, pp["amts"], pp["dl_ramp"], pp["dl_ramp"][0])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
        k = np.interp(chroma, _VIB_C_NODES, k_nodes)
        mh = _vib_hue_interp(hue, mh_nodes)
        ml_rows = pp.get("ml")
        if ml_rows is not None:
            ml_nodes = _vib_row(pct, pp["amts"], ml_rows, ml_rows[0])
            mh = mh * np.interp(lab[..., 0], _VIB_L_NODES, ml_nodes)
        gain = 1.0 + np.maximum(k * mh, 0.0)
        lab[..., 1] *= gain
        lab[..., 2] *= gain
        c0, c1 = float(ramp[0]), float(max(ramp[1], ramp[0] + 1e-3))
        t = np.clip((chroma - c0) / (c1 - c0), 0.0, 1.0)
        w = t * t * (3.0 - 2.0 * t)                       # smoothstep chroma 权重
        lab[..., 0] = np.clip(lab[..., 0] + _vib_hue_interp(hue, dl_nodes) * w, 0.0, 100.0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Conversion from CIE-LAB.*negative Z values.*",
                category=UserWarning,
            )
            out = color.lab2rgb(lab)
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    pn = p["neg"]
    amt = -pct
    r_nodes = _vib_row(amt, pn["amts"], pn["r"], np.ones(len(_VIB_C_NODES)))
    beta = float(np.interp(amt, [0.0] + list(pn["amts"]), [0.0] + list(pn["beta"])))
    r = np.interp(chroma, _VIB_C_NODES, r_nodes).astype(np.float32)
    mean_rgb = img.mean(axis=-1)
    luma = img @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gray = (1.0 - beta) * mean_rgb + beta * luma
    gs_rows = pn.get("gs")
    if gs_rows is not None:
        gs_nodes = _vib_row(amt, pn["amts"], gs_rows, np.zeros(len(_VIB_H_NODES)))
        gsc = _vib_row(amt, pn["amts"], pn["gsc"], pn["gsc"][0])
        hue = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
        c0, c1 = float(gsc[0]), float(max(gsc[1], gsc[0] + 1e-3))
        t = np.clip((chroma - c0) / (c1 - c0), 0.0, 1.0)
        wc = 1.0 - t * t * (3.0 - 2.0 * t)                # 高色度处偏移衰减
        s = _vib_hue_interp(hue, gs_nodes) * wc
        gray = gray + s * (img.min(axis=-1) - gray)
    out = gray[..., None] + (img - gray[..., None]) * r[..., None]
    return np.clip(out, 0.0, 1.0).astype(np.float32)

# ---- Sharpen（LR Detail 面板：Amount 0..150 / Radius / Detail / Masking）----
def _smoothstep(x, lo: float, hi: float):
    t = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lr_sharpen(
    image: np.ndarray,
    amount: float,
    radius: float = 1.0,
    detail: float = 25.0,
    masking: float = 0.0,
) -> np.ndarray:
    """Lightroom-faithful sharpening (Detail panel: Amount/Radius/Detail/Masking).

    image: float32 [0,1] RGB; amount: LR 0..150; radius: LR 0.5..3.0;
    detail: LR 0..100 (default 25); masking: LR 0..100 (default 0).
    """
    amount = float(np.clip(amount, 0.0, 150.0))
    if amount < 1e-6:
        return image
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    radius = float(np.clip(radius, 0.5, 3.0))
    detail = float(np.clip(detail, 0.0, 100.0))
    masking = float(np.clip(masking, 0.0, 100.0))

    luma = img @ _LUMA_W

    # Detail 归一化（1.0 = LR 默认 25）；调制均锚定 d=1 / radius=1 不变
    d = max(detail / 25.0, 0.05)

    # USM 半径：随 SharpenRadius 线性；低 Detail 变宽（halo 主导）、高 Detail 收紧（近似去卷积）
    sigma = 0.75 * radius
    sigma *= d ** (-0.24 if d < 1.0 else -0.123)
    hf = luma - gaussian_filter(luma, sigma=sigma)

    # amount 响应（在 radius=1 / detail=25 / masking=0 的 GT 上标定）
    a = amount * (0.019515 + 0.00047057 * amount)   # 近原点增益（细节增益）
    a *= radius ** -1.15        # LR 随半径增大压低增益（边缘总提升近似恒定）
    a *= d ** 0.7               # Detail 低抑制细节增益、高放大
    cap = 0.165 * (0.75 + 0.25 * d)                  # 过冲软上限（halo 抑制）

    delta = a * hf / (1.0 + (a / cap) * np.abs(hf))

    # 色调权重：阴影强抑制、高光缓滚降（标定自 GT 分箱增益）
    tone_w = _smoothstep(luma, 0.02, 0.32) * (1.0 - 0.55 * _smoothstep(luma, 0.70, 1.02))
    delta *= tone_w

    # EdgeMasking：阈值化边缘掩码（0 = 全图，100 = 仅强边）
    if masking > 1e-6:
        m = masking / 100.0
        edge = gaussian_filter(np.abs(hf), sigma=2.0)
        t0 = 0.002 + 0.020 * m * m
        mask = _smoothstep(edge, t0 * 0.5, t0 * 2.0)
        delta *= mask

    out_luma = np.clip(luma + delta, 0.0, 1.0)
    ratio = out_luma / np.maximum(luma, 1e-6)
    return np.clip(img * ratio[..., None], 0.0, 1.0).astype(np.float32)

# ---- Clarity v2（图像自适应大半径局部对比；表数据烘焙自 fits/Clarity.json）----
_CLARITY_V2_TABLES_JSON = """{"v2":true,"short_edge_ref":962,"pos":{"sig_frac":0.1,"sig_color":0.2,"NL":12,"NU":6,"A":[[-0.013172,-0.010293,-0.008605,-0.006232,-0.004344,-0.003344],[-0.018551,-0.014088,-0.009917,-0.006534,-0.004365,-0.003314],[-0.02623,-0.019388,-0.010961,-0.006389,-0.004174,-0.003202],[-0.032957,-0.024243,-0.012584,-0.004676,-0.00366,-0.003084],[-0.037926,-0.028086,-0.014698,-0.005196,-0.003657,-0.003338],[-0.042596,-0.031869,-0.017487,-0.007031,-0.00336,-0.004143],[-0.047421,-0.036052,-0.021196,-0.010701,-0.006692,-0.006422],[-0.050101,-0.039146,-0.024833,-0.01501,-0.011643,-0.011664],[-0.048215,-0.039467,-0.027186,-0.018577,-0.015982,-0.016477],[-0.040752,-0.036671,-0.027936,-0.021057,-0.019037,-0.01968],[-0.036478,-0.033504,-0.028242,-0.023091,-0.021333,-0.021744],[-0.034602,-0.032284,-0.028464,-0.02482,-0.023014,-0.023122]],"B":[[0.046282,0.050737,0.056332,0.059858,0.061846,0.062726],[0.049841,0.053807,0.058433,0.061158,0.062558,0.063141],[0.055432,0.058645,0.062342,0.06348,0.063667,0.063665],[0.061742,0.063589,0.06578,0.066335,0.064512,0.063721],[0.067733,0.067592,0.067349,0.06642,0.063881,0.062574],[0.071717,0.069672,0.066899,0.064296,0.06169,0.059838],[0.071765,0.068708,0.064402,0.060466,0.057263,0.055182],[0.067295,0.064258,0.059953,0.055902,0.052665,0.050636],[0.059777,0.057172,0.053802,0.050693,0.048246,0.046723],[0.053247,0.050639,0.04796,0.045523,0.043815,0.042808],[0.049544,0.047955,0.045512,0.04219,0.040191,0.039198],[0.047821,0.046562,0.044381,0.041528,0.038438,0.037121]],"c_nodes":[-0.5,-0.322361,-0.207833,-0.133994,-0.086389,-0.055697,-0.035909,-0.023151,-0.014926,-0.009623,-0.006204,-0.004,0.0,0.004,0.006204,0.009623,0.014926,0.023151,0.035909,0.055697,0.086389,0.133994,0.207833,0.322361,0.5],"shape":[-2.198316,-2.198316,-1.66306,-1.312962,-0.936502,-0.621473,-0.414154,-0.271647,-0.181872,-0.137913,-0.09399,-0.064535,0.0,0.078406,0.101559,0.142961,0.213742,0.315386,0.458036,0.697619,1.079489,1.494365,2.073202,2.935988,2.935988],"s_v":[[0.0,0.0],[30.0,0.3234],[60.0,0.6431],[100.0,1.02]]},"neg":{"sig_frac":0.02,"NLW":20,"W":[-0.002396,-0.003635,-0.00607,-0.009605,-0.014064,-0.01917,-0.02457,-0.029865,-0.03465,-0.03857,-0.041408,-0.043132,-0.043806,-0.043421,-0.041866,-0.039016,-0.034827,-0.02954,-0.024139,-0.020545],"c_nodes":[-0.5,-0.322361,-0.207833,-0.133994,-0.086389,-0.055697,-0.035909,-0.023151,-0.014926,-0.009623,-0.006204,-0.004,0.0,0.004,0.006204,0.009623,0.014926,0.023151,0.035909,0.055697,0.086389,0.133994,0.207833,0.322361,0.5],"shape":[-5.630295,-5.630295,-3.71515,-2.405296,-1.567241,-1.014689,-0.655634,-0.424233,-0.272691,-0.176341,-0.108305,-0.069938,0.0,0.056169,0.088119,0.139659,0.217702,0.32907,0.49413,0.725659,1.072118,1.613248,2.499707,3.838875,6.160329],"s_v":[[0.0,0.0],[30.0,0.3838],[60.0,0.6868],[100.0,1.02]]}}"""
_BAKED: dict | None = None


def _load_tables(fit: dict | None) -> dict:
    global _BAKED
    if fit and fit.get("v2"):
        return fit
    if _BAKED is None:
        _BAKED = json.loads(_CLARITY_V2_TABLES_JSON)
    return _BAKED


# ---------------------------------------------------------------------------
# 纯函数实现
# ---------------------------------------------------------------------------
def _lab_light(img: np.ndarray) -> np.ndarray:
    """Lab L / 100，float32 [0,1]。"""
    lab = cv2.cvtColor(np.clip(img, 0.0, 1.0).astype(np.float32, copy=False), cv2.COLOR_RGB2LAB)
    return np.clip(lab[..., 0] / 100.0, 0.0, 1.0).astype(np.float32)


def _luma_percentile_map(light: np.ndarray) -> np.ndarray:
    """u = 每像素亮度在图内的 CDF 分位（图像自适应色调坐标）。"""
    sample = np.sort(light[::2, ::2].ravel())
    u = np.searchsorted(sample, light.ravel()) / max(1, len(sample))
    return u.astype(np.float32).reshape(light.shape)


def _bilateral_base_np(light: np.ndarray, sigma_space: float, sigma_color: float) -> np.ndarray:
    """下采样双边滤波大半径 base（≈标定用 962px 短边 / ds4 的配置，按图缩放）。"""
    h, w = light.shape
    ds = max(1, int(round(min(h, w) / 240.0)))
    if ds > 1:
        small = cv2.resize(light, (max(2, w // ds), max(2, h // ds)), interpolation=cv2.INTER_AREA)
    else:
        small = light
    d = max(3, int(round(2.0 * sigma_space / ds)) | 1)
    base = cv2.bilateralFilter(small, d, float(sigma_color), float(sigma_space / ds))
    if ds > 1:
        base = cv2.resize(base, (w, h), interpolation=cv2.INTER_LINEAR)
    return base.astype(np.float32)


def _interp_table_2d(table: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """在 [0,1]² 上按 bin 中心双线性插值 2D 表。table: (NX, NY)。"""
    nx, ny = table.shape
    fx = np.clip(x * nx - 0.5, 0.0, nx - 1.0)
    fy = np.clip(y * ny - 0.5, 0.0, ny - 1.0)
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, nx - 1)
    y1 = np.minimum(y0 + 1, ny - 1)
    tx = (fx - x0).astype(np.float32)
    ty = (fy - y0).astype(np.float32)
    v00, v01 = table[x0, y0], table[x0, y1]
    v10, v11 = table[x1, y0], table[x1, y1]
    return ((v00 * (1 - ty) + v01 * ty) * (1 - tx) + (v10 * (1 - ty) + v11 * ty) * tx).astype(np.float32)


def _interp_table_1d(vals: np.ndarray, x: np.ndarray) -> np.ndarray:
    n = len(vals)
    centers = (np.arange(n) + 0.5) / n
    return np.interp(x, centers, vals).astype(np.float32)


def adjust_clarity_v2(image: np.ndarray, value: float, tables: dict | None = None) -> np.ndarray:
    """LR-faithful Clarity。value: LR 滑杆值 [-100, 100]；image: float32 [0,1] RGB。"""
    if tables is None:
        tables = _load_tables(None)
    v = float(np.clip(value, -100.0, 100.0))
    if abs(v) < 1e-6:
        return image.astype(np.float32, copy=False)
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    light = _lab_light(img)
    short = float(min(light.shape))

    if v > 0:
        t = tables["pos"]
        sig = float(t["sig_frac"]) * short
        base = _bilateral_base_np(light, sig, float(t.get("sig_color", 0.2)))
        c = light - base
        u = _luma_percentile_map(light)
        A = np.asarray(t["A"], dtype=np.float32)
        B = np.asarray(t["B"], dtype=np.float32)
        shape = np.interp(c, np.asarray(t["c_nodes"], dtype=np.float64),
                          np.asarray(t["shape"], dtype=np.float64)).astype(np.float32)
        scale = float(np.interp(v, *zip(*t["s_v"])))
        delta = scale * (_interp_table_2d(A, light, u) + _interp_table_2d(B, light, u) * shape)
    else:
        t = tables["neg"]
        sig = float(t["sig_frac"]) * short
        c = light - gaussian_filter(light, sigma=sig).astype(np.float32)
        W = np.asarray(t["W"], dtype=np.float32)
        shape = np.interp(c, np.asarray(t["c_nodes"], dtype=np.float64),
                          np.asarray(t["shape"], dtype=np.float64)).astype(np.float32)
        scale = float(np.interp(-v, *zip(*t["s_v"])))
        delta = scale * _interp_table_1d(W, light) * shape

    new_light = np.clip(light + delta, 0.0, 1.0)
    ratio = new_light / np.maximum(light, 1e-4)
    ratio = np.clip(ratio, 0.0, 4.0)
    return np.clip(img * ratio[..., None], 0.0, 1.0).astype(np.float32)

# ---- Shadows v2（图像自适应乘性 luma 曲线；fits/Shadows.json[shadows_v2] 烘焙）----
_SHADOWS_V2_FIT: dict = json.loads(r'''{"stats": ["mean"], "bin_centers": [0.015625, 0.046875, 0.078125, 0.109375, 0.140625, 0.171875, 0.203125, 0.234375, 0.265625, 0.296875, 0.328125, 0.359375, 0.390625, 0.421875, 0.453125, 0.484375, 0.515625, 0.546875, 0.578125, 0.609375, 0.640625, 0.671875, 0.703125, 0.734375, 0.765625, 0.796875, 0.828125, 0.859375, 0.890625, 0.921875, 0.953125, 0.984375], "values": {"-100": {"curves": [[-0.017136, -0.033329, -0.031507, -0.019613, -0.004568, 0.014005, 0.029148, 0.03881, 0.047198, 0.056509, 0.066883, 0.075285, 0.079381, 0.079386, 0.07626, 0.072297, 0.071048, 0.074354, 0.079136, 0.081109, 0.077256, 0.068427, 0.059629, 0.05508, 0.053755, 0.050054, 0.043808, 0.036086, 0.025744, 0.016422, 0.009109, 0.000896], [0.0, -0.004241, -0.039717, -0.087831, -0.13487, -0.182569, -0.218855, -0.239362, -0.250772, -0.261984, -0.280803, -0.299961, -0.310844, -0.312713, -0.30508, -0.291414, -0.278137, -0.267996, -0.25735, -0.242414, -0.219604, -0.189662, -0.161407, -0.143177, -0.133893, -0.122509, -0.105697, -0.086242, -0.063254, -0.040148, -0.020341, -0.0021]]}, "-060": {"curves": [[-0.015356, -0.028741, -0.022342, -0.009495, 0.002474, 0.016044, 0.026064, 0.031258, 0.035035, 0.039581, 0.045244, 0.049888, 0.052006, 0.051694, 0.049492, 0.046783, 0.04559, 0.04689, 0.048764, 0.048999, 0.046159, 0.04066, 0.035288, 0.032435, 0.03153, 0.029248, 0.025537, 0.021075, 0.015065, 0.009668, 0.005408, 0.000468], [0.0, 0.004883, -0.028176, -0.07016, -0.104944, -0.137584, -0.159552, -0.168698, -0.171056, -0.174228, -0.183713, -0.194099, -0.199695, -0.199959, -0.194395, -0.185025, -0.175413, -0.166994, -0.15777, -0.146441, -0.131504, -0.113061, -0.095918, -0.084791, -0.079067, -0.072133, -0.062103, -0.050732, -0.037299, -0.02382, -0.012163, -0.001133]]}, "-030": {"curves": [[-0.010843, -0.017857, -0.012004, -0.003199, 0.003932, 0.011542, 0.01657, 0.018654, 0.020061, 0.022033, 0.024709, 0.026917, 0.027869, 0.027586, 0.026325, 0.024757, 0.023798, 0.023942, 0.024402, 0.024254, 0.022734, 0.019925, 0.017256, 0.015861, 0.015371, 0.014217, 0.012402, 0.010198, 0.007224, 0.004664, 0.002711, 0.000305], [0.0, 0.004042, -0.018188, -0.044712, -0.064248, -0.081344, -0.091417, -0.094088, -0.093596, -0.094116, -0.098234, -0.103013, -0.105515, -0.105314, -0.102049, -0.096682, -0.090768, -0.085102, -0.079249, -0.072957, -0.065286, -0.055959, -0.047402, -0.041876, -0.038964, -0.035493, -0.030556, -0.024947, -0.018296, -0.01177, -0.006184, -0.000643]]}, "+030": {"curves": [[0.015532, 0.026603, 0.021306, 0.011622, 0.002221, -0.008696, -0.01594, -0.019074, -0.021141, -0.023678, -0.026818, -0.029361, -0.030505, -0.030214, -0.028538, -0.026156, -0.024484, -0.024221, -0.02447, -0.02418, -0.022596, -0.019876, -0.017259, -0.015859, -0.015386, -0.014283, -0.01258, -0.010484, -0.007576, -0.004843, -0.00263, -0.000373], [0.0, -0.018746, -0.003775, 0.024135, 0.051092, 0.076176, 0.090874, 0.095891, 0.096995, 0.09875, 0.104028, 0.109639, 0.112543, 0.112232, 0.107738, 0.099969, 0.091708, 0.08462, 0.078123, 0.071485, 0.063694, 0.054573, 0.0462, 0.040767, 0.03788, 0.034499, 0.029821, 0.024402, 0.017902, 0.011236, 0.005458, 0.000567]]}, "+060": {"curves": [[0.038076, 0.069706, 0.053651, 0.026224, 0.00353, -0.021051, -0.037015, -0.043463, -0.047272, -0.052123, -0.058182, -0.062804, -0.06446, -0.062851, -0.058243, -0.052638, -0.04876, -0.047852, -0.048153, -0.047511, -0.044366, -0.038862, -0.033577, -0.030841, -0.029979, -0.027847, -0.024464, -0.020314, -0.014642, -0.009383, -0.005134, -0.000594], [0.0, -0.059531, -0.020441, 0.051791, 0.113947, 0.168842, 0.200067, 0.209452, 0.209942, 0.21213, 0.221792, 0.231877, 0.236105, 0.232812, 0.220359, 0.202027, 0.183621, 0.168372, 0.154961, 0.141564, 0.125979, 0.107638, 0.090866, 0.080137, 0.074524, 0.067888, 0.058627, 0.047946, 0.035183, 0.022298, 0.011126, 0.001063]]}, "+100": {"curves": [[0.088731, 0.166274, 0.119765, 0.050334, 0.002574, -0.045602, -0.075358, -0.086165, -0.091599, -0.09828, -0.106812, -0.112689, -0.112951, -0.107577, -0.097761, -0.087076, -0.079989, -0.078165, -0.07844, -0.077218, -0.071999, -0.063075, -0.054509, -0.049964, -0.048404, -0.044899, -0.039498, -0.032841, -0.023653, -0.015226, -0.008461, -0.000953], [0.0, -0.163989, -0.070035, 0.095148, 0.218833, 0.323211, 0.379454, 0.393017, 0.389385, 0.387702, 0.398964, 0.410956, 0.411426, 0.39846, 0.371211, 0.336177, 0.303362, 0.277077, 0.25419, 0.231513, 0.205632, 0.17572, 0.148426, 0.130759, 0.121316, 0.11036, 0.095329, 0.078053, 0.057371, 0.036505, 0.018442, 0.001965]]}}}''')

_W_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _shadows_delta_curves(fit_block: dict, v: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """任意 v∈[-100,100] 的 (bin_centers, a, b)：扫描锚点曲线线性插值，0 锚点=零曲线。"""
    centers = np.asarray(fit_block["bin_centers"], dtype=np.float64)
    anchors = {float(k): np.asarray(blk["curves"], dtype=np.float64)
               for k, blk in fit_block["values"].items()}
    zero = np.zeros((2, len(centers)))
    anchors[0.0] = zero
    keys = sorted(anchors)
    v = float(np.clip(v, keys[0], keys[-1]))
    hi = next(k for k in keys if k >= v)
    lo = next(k for k in reversed(keys) if k <= v)
    if hi == lo:
        cur = anchors[lo]
    else:
        t = (v - lo) / (hi - lo)
        cur = (1.0 - t) * anchors[lo] + t * anchors[hi]
    return centers, cur[0], cur[1]


def _apply_shadows_v2(img: np.ndarray, v: float, fit_block: dict | None = None) -> np.ndarray:
    """img: float32 RGB [0,1]；v: LR Shadows2012 值 [-100,100]。"""
    if abs(v) < 1e-6:
        return img.astype(np.float32, copy=True)
    if fit_block is None:
        fit_block = _SHADOWS_V2_FIT
    centers, a, b = _shadows_delta_curves(fit_block, v)
    luma = (img @ _W_LUMA).astype(np.float32)
    m = float(luma.mean())
    tone = np.maximum.accumulate(centers + (a + b * m))   # 单调守卫：极端统计外推防 tone 反转
    delta = np.interp(luma, centers, tone - centers).astype(np.float32)
    ratio = (luma + delta) / np.maximum(luma, 1e-4)
    return np.clip(img * ratio[..., None], 0.0, 1.0).astype(np.float32)

# ---- HSL Blue v2（LR 蓝带三滑杆；响应表由 scratch/hueadjustmentblue_v3_fit.py 生成）----
_HUE_ANCHORS_DEG = [150.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0]
_V_ANCHORS = [-100.0, -50.0, 50.0, 100.0]
_C_ANCHORS = [0.0, 0.04, 0.09, 0.16, 0.25, 0.42]
_L_ANCHORS = [0.075, 0.225, 0.375, 0.525, 0.675, 0.825, 0.95]
_HSL_BLUE_TABLES: dict = json.loads(r'''{"HueAdjustmentBlue": {"dh": [[0.0, 0.0, 0.0, 0.0, -7.2, -12.63, -19.58, -25.31, -30.76, -34.68, -33.68, -41.35, -43.48, -48.28, -50.8, -52.31, -50.72, -52.21, -47.28, -37.78, -28.28, -9.37, -4.68, 0.0], [0.0, 0.0, 0.0, 0.0, -2.25, -6.32, -10.49, -14.34, -18.54, -21.43, -21.67, -26.23, -28.66, -30.35, -30.6, -33.85, -30.25, -33.75, -28.59, -22.15, -15.71, -4.56, -2.28, 0.0], [0.0, 0.0, 0.0, 0.0, 1.23, 6.92, 9.49, 12.42, 16.53, 20.36, 38.38, 30.0, 33.94, 29.23, 31.0, 26.0, 27.86, 22.5, 16.43, 11.28, 6.14, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 3.75, 15.43, 20.25, 29.62, 36.43, 49.48, 77.12, 66.69, 70.29, 65.88, 61.15, 57.37, 54.43, 44.01, 33.38, 23.36, 13.35, 4.56, 2.28, 0.0]], "w_sr": [[0.0, 0.0, 0.0, 0.0, 0.029, 0.392, 0.339, 0.612, 0.797, 0.853, 0.901, 0.969, 1.142, 1.3, 1.153, 1.3, 1.3, 1.3, 1.141, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.258, 0.457, 0.532, 0.631, 0.782, 0.909, 0.871, 1.212, 1.242, 1.3, 1.3, 1.3, 1.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.224, 0.516, 0.825, 0.896, 1.047, 1.199, 1.071, 0.897, 0.464, 0.0, 0.065, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.061, 0.198, 0.553, 1.0, 1.3, 1.291, 1.048, 1.019, 0.837, 0.721, 0.443, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "SR1": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.2857, 0.3896, 0.4912, 0.4362, 0.2727, 0.3341, 0.275], [0.5861, 0.5996, 0.5522, 0.4462, 0.24, 0.2644, 0.2222], [0.6314, 0.9601, 0.8202, 0.3958, 0.2831, 0.2706, 0.2222], [0.0, 0.2439, 1.6296, 1.1997, 0.3633, 0.2817, 0.2222], [0.0, 0.5435, 0.7816, 1.6591, 0.6573, 0.2817, 0.2222]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.1818, 0.2123, 0.3039, 0.2677, 0.1304, 0.17, 0.1404], [0.421, 0.3951, 0.3523, 0.2661, 0.1084, 0.1047, 0.0505], [0.6164, 0.6109, 0.5254, 0.2219, 0.1212, 0.0935, 0.0505], [0.0, 0.2439, 1.4286, 0.6593, 0.1943, 0.1237, 0.0505], [0.0, 0.5435, 0.7813, 1.6364, 0.2418, 0.1237, 0.0505]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.2987, -0.2636, -0.2, -0.1539, -0.1881, -0.1592, -0.1178], [-0.3616, -0.3152, -0.2547, -0.2322, -0.152, -0.161, -0.1587], [-0.4628, -0.3571, -0.2948, -0.2309, -0.192, -0.1806, -0.1587], [-0.6055, -0.5695, -0.3267, -0.2366, -0.1919, -0.1696, -0.1587], [-0.6055, -0.5428, -0.5159, -0.1785, -0.174, -0.1696, -0.1587]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.3697, -0.2806, -0.1412, -0.0984, -0.2303, -0.1519, -0.1091], [-0.3899, -0.2183, -0.1776, -0.1565, -0.2672, -0.2219, -0.1855], [-0.4028, -0.2949, -0.2348, -0.2293, -0.2222, -0.16, -0.1855], [-0.5283, -0.4955, -0.2422, -0.1478, -0.0946, -0.0921, -0.1855], [-0.5283, -0.4532, -0.4261, -0.0964, -0.0818, -0.0921, -0.1855]]], "w_dl": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.157, 0.35, 0.669, 0.888, 0.968, 1.201, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.3, 1.005, 0.67, 0.335, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.213, 0.515, 0.868, 0.942, 0.992, 1.178, 1.229, 1.3, 1.3, 1.3, 1.3, 0.934, 0.747, 0.56, 0.374, 0.187, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.183, 0.264, 0.485, 0.689, 0.929, 1.058, 1.125, 0.993, 0.879, 0.706, 0.392, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.38, 0.583, 0.793, 1.06, 1.187, 1.067, 1.017, 0.902, 0.737, 0.446, 0.237, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "DL": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.002, -0.0039, -0.0059, -0.0059, -0.002, -0.0059, -0.0059], [-0.0118, -0.0137, -0.0137, -0.0118, -0.0039, -0.0059, -0.0078], [-0.0196, -0.0373, -0.0373, -0.0196, -0.0137, -0.0137, -0.0078], [0.0059, -0.0118, -0.1118, -0.102, -0.0569, -0.0353, -0.0078], [0.0059, -0.0392, -0.0608, -0.1529, -0.1412, -0.0353, -0.0078]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.002, -0.0039, -0.0059, -0.0059, -0.0039, -0.0059, -0.0039], [-0.0118, -0.0137, -0.0137, -0.0118, -0.0059, -0.0078, -0.0098], [-0.0235, -0.0314, -0.0314, -0.0196, -0.0157, -0.0157, -0.0098], [0.0, -0.0196, -0.1118, -0.0725, -0.049, -0.0333, -0.0098], [0.0, -0.049, -0.0706, -0.1549, -0.0941, -0.0333, -0.0098]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0039, 0.0059, 0.0039, 0.0039, 0.0039, 0.0039, 0.0039], [0.0118, 0.0118, 0.0118, 0.0098, 0.0078, 0.0098, 0.0078], [0.0255, 0.0216, 0.0216, 0.0196, 0.0157, 0.0157, 0.0078], [0.0647, 0.0706, 0.0412, 0.0451, 0.0412, 0.0333, 0.0078], [0.0647, 0.0765, 0.0765, 0.051, 0.0549, 0.0333, 0.0078]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0039, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059], [0.0137, 0.0137, 0.0098, 0.0098, 0.0157, 0.0137, 0.0118], [0.0235, 0.0216, 0.0216, 0.0235, 0.0216, 0.0216, 0.0118], [0.0588, 0.0627, 0.0373, 0.0392, 0.0373, 0.0294, 0.0118], [0.0588, 0.0667, 0.0686, 0.0451, 0.049, 0.0294, 0.0118]]]}, "SaturationAdjustmentBlue": {"dh": [[0.0, 0.0, 0.0, 0.0, 0.88, 2.31, 2.73, 2.11, 9.03, 7.03, -0.54, -3.93, 0.38, 8.75, 10.21, 11.67, 13.01, 5.61, -0.59, -1.18, -1.76, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.1, 0.0, 1.14, -0.77, 0.0, 1.32, 5.91, 2.3, 2.21, 0.0, -1.66, -5.0, 7.06, -3.54, -1.36, -0.95, -0.54, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.53, 1.1, 0.43, -0.56, 0.0, -0.42, -1.52, -2.57, -2.65, -1.8, -3.77, -1.61, -3.89, -1.11, -0.56, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 2.22, 0.46, -0.41, -1.42, -1.18, 5.99, -4.07, -4.26, -5.09, -3.95, -6.19, -2.41, -3.97, -0.99, -0.63, -0.27, 0.0, 0.0, 0.0]], "w_sr": [[0.0, 0.0, 0.0, 0.0, 0.245, 0.331, 0.539, 0.743, 0.87, 0.95, 1.0, 1.0, 1.0, 1.006, 1.0, 1.3, 0.741, 1.3, 1.156, 0.252, 0.416, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.2, 0.342, 0.587, 0.769, 0.89, 0.978, 1.014, 1.026, 1.01, 1.048, 1.048, 0.781, 0.826, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.101, 0.399, 0.532, 0.781, 0.924, 0.986, 1.008, 1.011, 1.055, 1.024, 0.912, 0.514, 0.682, 1.078, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.091, 0.279, 0.388, 0.655, 0.855, 0.956, 0.897, 1.084, 1.084, 1.019, 0.848, 0.896, 0.526, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "SR1": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -0.8725, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -0.9408, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -0.9745, -1.0, -1.0, -1.0]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [-0.5541, -0.4691, -0.4976, -0.4928, -0.4, -0.4715, -0.449], [-0.5604, -0.5452, -0.5071, -0.506, -0.428, -0.4524, -0.4548], [-0.6, -0.5497, -0.531, -0.4879, -0.4677, -0.4644, -0.4548], [-0.7358, -0.7029, -0.5594, -0.5074, -0.4789, -0.4583, -0.4548], [-0.7358, -0.6757, -0.6489, -0.4956, -0.4864, -0.4583, -0.4548]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.4112, 0.5497, 0.4929, 0.3433, 0.2568, 0.1864], [0.2963, 0.6172, 0.5992, 0.5036, 0.3086, 0.2575, 0.155], [0.4944, 0.7788, 0.6792, 0.501, 0.3308, 0.2385, 0.155], [0.0, 0.2439, 0.9228, 0.6891, 0.2705, 0.2063, 0.155], [0.0, 0.5435, 0.7807, 0.8528, 0.2708, 0.2063, 0.155]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0535, 1.5921, 2.7343, 2.0318, 1.1522, 0.7487, 0.4521], [0.7255, 3.8261, 3.572, 2.5346, 0.945, 0.7684, 0.3726], [0.716, 2.1234, 3.4003, 3.5841, 1.0394, 0.6429, 0.3726], [0.0, 0.2439, 1.7561, 2.3562, 1.4146, 0.5658, 0.3726], [0.0, 0.5393, 0.7742, 1.6591, 1.2222, 0.5658, 0.3726]]], "w_dl": [[0.0, 0.0, 0.0, 0.0, 0.355, 0.803, 0.982, 0.976, 1.057, 1.123, 1.049, 0.996, 0.934, 0.823, 0.469, 0.442, 0.176, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.222, 0.739, 0.902, 1.008, 1.108, 1.151, 1.056, 0.968, 0.901, 0.74, 0.535, 0.453, 0.421, 0.388, 0.324, 0.259, 0.194, 0.129, 0.065, 0.0], [0.0, 0.0, 0.0, 0.0, 0.14, 0.351, 0.523, 0.685, 0.878, 1.004, 1.02, 1.064, 1.066, 1.016, 1.055, 0.936, 0.792, 0.878, 0.38, 0.227, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.091, 0.237, 0.385, 0.586, 0.817, 0.967, 0.947, 1.107, 1.07, 0.998, 0.841, 0.863, 0.557, 0.51, 0.275, 0.155, 0.0, 0.0, 0.0, 0.0]], "DL": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0039, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059, 0.0059], [0.0157, 0.0137, 0.0118, 0.0098, 0.0157, 0.0157, 0.0118], [0.0284, 0.0255, 0.0255, 0.0255, 0.0235, 0.0235, 0.0118], [0.0745, 0.0784, 0.051, 0.0529, 0.051, 0.0392, 0.0118], [0.0745, 0.0843, 0.0843, 0.0686, 0.0725, 0.0392, 0.0118]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.002, 0.0039, 0.0039, 0.0039, 0.0039, 0.0039, 0.0039], [0.0078, 0.0078, 0.0059, 0.0059, 0.0078, 0.0078, 0.0039], [0.0196, 0.0157, 0.0157, 0.0157, 0.0137, 0.0137, 0.0039], [0.0588, 0.0588, 0.0333, 0.0333, 0.0314, 0.0255, 0.0039], [0.0588, 0.0608, 0.0608, 0.0451, 0.049, 0.0255, 0.0039]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0078, -0.0098, -0.0098, -0.0078, -0.0059, -0.0039], [-0.0078, -0.0196, -0.0196, -0.0176, -0.0157, -0.0137, -0.0098], [-0.0176, -0.0333, -0.0373, -0.0333, -0.0275, -0.0235, -0.0098], [0.002, -0.0176, -0.0745, -0.0706, -0.0588, -0.0412, -0.0098], [0.002, -0.0471, -0.0686, -0.0941, -0.0922, -0.0412, -0.0098]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0255, -0.0431, -0.0412, -0.0255, -0.0176, -0.0118], [-0.0157, -0.0902, -0.0922, -0.0784, -0.0529, -0.0431, -0.0255], [-0.0216, -0.0804, -0.1373, -0.1686, -0.1039, -0.0765, -0.0255], [0.0039, -0.0157, -0.1235, -0.1745, -0.2431, -0.1588, -0.0255], [0.0039, -0.0431, -0.0647, -0.1569, -0.2059, -0.1588, -0.0255]]]}, "LuminanceAdjustmentBlue": {"dh": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.32, -0.6, -0.3, -1.89, -0.97, -0.9, -1.73, -1.0, -3.86, -0.94, -1.46, -0.71, -0.36, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.11, -0.77, -0.0, -0.0, 0.0, 0.0, -1.0, -0.06, -0.12, -0.9, -0.45, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.35, 0.25, 0.84, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.35, 0.79, 1.57, 0.29, 0.36, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], "w_sr": [[0.0, 0.0, 0.0, 0.0, 0.287, 0.394, 0.668, 0.692, 0.964, 1.083, 1.13, 1.008, 0.895, 0.955, 1.166, 1.3, 1.3, 0.553, 1.006, 0.646, 0.312, 0.161, 0.08, 0.0], [0.0, 0.0, 0.0, 0.129, 0.258, 0.387, 0.517, 0.646, 0.836, 1.072, 1.113, 1.003, 0.986, 0.722, 1.228, 1.146, 1.064, 0.982, 0.818, 0.655, 0.491, 0.327, 0.164, 0.0], [0.0, 0.0, 0.0, 0.176, 0.353, 0.531, 0.71, 0.818, 0.909, 0.996, 1.062, 1.107, 1.173, 1.2, 1.19, 0.849, 0.789, 0.728, 0.607, 0.485, 0.364, 0.243, 0.121, 0.0], [0.0, 0.0, 0.0, 0.095, 0.192, 0.51, 0.551, 0.82, 0.844, 0.957, 1.118, 1.131, 1.22, 1.088, 0.834, 1.052, 1.002, 1.3, 0.0, 0.305, 0.725, 0.0, 0.0, 0.0]], "SR1": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0444, 0.0597, -0.0168, -0.0856, -0.1, -0.1005], [0.0025, 0.0602, 0.0643, 0.0233, -0.1202, -0.1409, -0.0827], [0.0102, 0.0778, 0.1227, 0.1122, -0.1036, -0.0482, -0.0827], [-0.0, 0.2, 0.2969, 0.4851, 1.0556, 1.1561, -0.0827], [-0.0, 0.4476, 0.531, 1.0106, 1.2222, 1.1561, -0.0827]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0215, 0.0333, -0.0314, -0.0528, -0.063, -0.0735], [0.0, 0.0411, 0.0403, -0.0081, -0.0632, -0.0789, -0.0792], [-0.0023, 0.0418, 0.047, 0.0345, -0.1009, -0.0967, -0.0792], [0.0, 0.124, 0.1134, 0.1846, -0.0226, -0.1127, -0.0792], [0.0, 0.1714, 0.1698, 0.3245, 0.3283, -0.1127, -0.0792]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0194, -0.0293, 0.0452, 0.0635, 0.0562, 0.0496], [-0.0042, -0.0358, -0.0437, 0.0473, 0.0674, 0.1083, 0.0933], [-0.0196, -0.0346, -0.0496, 0.06, 0.0951, 0.125, 0.0933], [-0.0465, -0.0934, -0.1057, 0.1129, 0.1498, 0.1609, 0.0933], [-0.0465, -0.108, -0.1236, 0.1387, 0.1686, 0.1609, 0.0933]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0388, -0.0498, 0.0858, 0.0941, 0.1214, 0.359], [-0.0013, -0.0418, -0.0543, 0.0925, 0.1244, 0.2273, 0.3095], [-0.0318, -0.0667, -0.0839, 0.1378, 0.1972, 0.3313, 0.3095], [-0.1031, -0.1717, -0.0788, 0.244, 0.3421, 0.4, 0.3095], [-0.1031, -0.1941, -0.2605, 0.2938, 0.3817, 0.4, 0.3095]]], "w_dl": [[0.0, 0.0, 0.0, 0.0, 0.148, 0.317, 0.471, 0.601, 0.798, 0.935, 1.016, 1.028, 1.045, 1.004, 1.021, 0.925, 0.993, 0.425, 0.374, 0.225, 0.185, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.163, 0.325, 0.482, 0.619, 0.823, 0.956, 1.034, 1.041, 1.053, 1.021, 1.022, 0.924, 1.01, 0.425, 0.425, 0.252, 0.243, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.173, 0.378, 0.542, 0.693, 0.876, 0.993, 1.014, 1.047, 1.064, 1.049, 1.035, 0.939, 0.95, 0.731, 0.48, 0.24, 0.296, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.162, 0.381, 0.559, 0.703, 0.89, 0.996, 1.016, 1.035, 1.06, 1.034, 1.004, 0.938, 0.966, 0.597, 0.488, 0.242, 0.272, 0.0, 0.0, 0.0]], "DL": [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0118, -0.0275, -0.0431, -0.0549, -0.1078, -0.1176], [-0.0078, -0.0255, -0.051, -0.0725, -0.1196, -0.1804, -0.2078], [-0.0157, -0.0451, -0.0843, -0.1235, -0.202, -0.3314, -0.2078], [-0.0353, -0.0667, -0.1549, -0.2333, -0.4216, -0.5196, -0.2078], [-0.0353, -0.1196, -0.1529, -0.3, -0.4275, -0.5196, -0.2078]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, -0.0039, -0.0118, -0.0235, -0.0275, -0.0471, -0.0471], [-0.0039, -0.0137, -0.0255, -0.0353, -0.0549, -0.0745, -0.0765], [-0.0078, -0.0235, -0.0412, -0.0608, -0.0902, -0.1235, -0.0765], [-0.0157, -0.0353, -0.0765, -0.1157, -0.1902, -0.1961, -0.0765], [-0.0157, -0.0588, -0.0745, -0.1451, -0.2216, -0.1961, -0.0765]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0039, 0.0118, 0.0196, 0.0235, 0.0353, 0.0333], [0.0039, 0.0118, 0.0255, 0.0353, 0.0471, 0.0549, 0.0431], [0.0078, 0.0235, 0.0412, 0.0569, 0.0706, 0.0765, 0.0431], [0.0196, 0.0392, 0.0765, 0.1039, 0.1196, 0.1039, 0.0431], [0.0196, 0.0588, 0.0745, 0.1255, 0.1431, 0.1039, 0.0431]], [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0118, 0.0275, 0.0412, 0.0451, 0.0647, 0.0569], [0.0078, 0.0275, 0.051, 0.0667, 0.0863, 0.0961, 0.0745], [0.0157, 0.0471, 0.0804, 0.1078, 0.1255, 0.1294, 0.0745], [0.0451, 0.0745, 0.1471, 0.1882, 0.198, 0.1686, 0.0745], [0.0451, 0.1157, 0.149, 0.2216, 0.2353, 0.1686, 0.0745]]]}}''')

def _interp_v(tab: np.ndarray, v: float) -> np.ndarray:
    """扫描值维插值：tab (nv, ...) @ _V_ANCHORS ∪ {0=恒等}，返回 (...,)。"""
    va = list(_V_ANCHORS)
    v = float(np.clip(v, va[0], va[-1]))
    zero = np.zeros_like(tab[0])
    rows = [tab[i] for i in range(len(va))]
    if 0.0 not in va:
        k = int(np.searchsorted(va, 0.0))
        va = va[:k] + [0.0] + va[k:]
        rows = rows[:k] + [zero] + rows[k:]
    i1 = int(np.searchsorted(va, v))
    i1 = min(max(i1, 1), len(va) - 1)
    i0 = i1 - 1
    w = (v - va[i0]) / (va[i1] - va[i0] + 1e-12)
    return rows[i0] * (1.0 - w) + rows[i1] * w


def _bilinear_cl(c: np.ndarray, l_: np.ndarray, tab2d: np.ndarray) -> np.ndarray:
    """(nc, nl) 表在 (chroma, L) 上双线性插值；越界→端点。"""
    ca = np.asarray(_C_ANCHORS, dtype=np.float32)
    la = np.asarray(_L_ANCHORS, dtype=np.float32)
    i = np.clip(np.searchsorted(ca, c) - 1, 0, len(ca) - 2)
    j = np.clip(np.searchsorted(la, l_) - 1, 0, len(la) - 2)
    c0, c1 = ca[i], ca[i + 1]
    l0, l1 = la[j], la[j + 1]
    wc = np.clip((c - c0) / (c1 - c0 + 1e-12), 0.0, 1.0).astype(np.float32)
    wl = np.clip((l_ - l0) / (l1 - l0 + 1e-12), 0.0, 1.0).astype(np.float32)
    t = tab2d.astype(np.float32)
    return (t[i, j] * (1 - wc) * (1 - wl) + t[i, j + 1] * (1 - wc) * wl
            + t[i + 1, j] * wc * (1 - wl) + t[i + 1, j + 1] * wc * wl)


def _apply_blue_tables(img: np.ndarray, op_key: str, v: float) -> np.ndarray:
    """按 op 的响应表施加蓝带 HSL 变换。img: float32 [0,1] RGB。"""
    if abs(v) < 1e-6 or op_key not in _HSL_BLUE_TABLES:
        return img
    tabs = _HSL_BLUE_TABLES[op_key]
    dh_cur = _interp_v(np.asarray(tabs["dh"], dtype=np.float32), v)       # (nh,)
    wsr_cur = _interp_v(np.asarray(tabs["w_sr"], dtype=np.float32), v)    # (nh,)
    sr1_cur = _interp_v(np.asarray(tabs["SR1"], dtype=np.float32), v)     # (nc,nl)
    wdl_cur = _interp_v(np.asarray(tabs["w_dl"], dtype=np.float32), v)    # (nh,)
    dl_cur = _interp_v(np.asarray(tabs["DL"], dtype=np.float32), v)       # (nc,nl)

    img = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    hls = rgb_to_hls_np(img).astype(np.float32, copy=False)
    h = hls[..., 0] * 360.0
    l_ = hls[..., 1]
    s = hls[..., 2]
    c = (img.max(axis=-1) - img.min(axis=-1)).astype(np.float32)

    ha = np.asarray(_HUE_ANCHORS_DEG, dtype=np.float32)
    dh = np.interp(h, ha, dh_cur).astype(np.float32)     # 端点=0 → 带外恒等
    w_sr = np.interp(h, ha, wsr_cur).astype(np.float32)
    w_dl = np.interp(h, ha, wdl_cur).astype(np.float32)
    sr1 = _bilinear_cl(c, l_, sr1_cur)
    dl = _bilinear_cl(c, l_, dl_cur)

    h_out = np.mod(h + dh, 360.0) / 360.0
    s_out = np.clip(s * (1.0 + w_sr * sr1), 0.0, 1.0)
    l_out = np.clip(l_ + w_dl * dl, 0.0, 1.0)
    out = hls_to_rgb_np(np.stack([h_out, l_out, s_out], axis=-1).astype(np.float32))
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def apply_hsl_blue_v2(img: np.ndarray, hue: float = 0.0, sat: float = 0.0,
                      lum: float = 0.0) -> np.ndarray:
    """LR Blue 通道 HSL 调整（滑杆值 [-100,100]）。多滑杆时按 hue→sat→lum 顺序复合。"""
    out = img
    if hue:
        out = _apply_blue_tables(out, "HueAdjustmentBlue", float(hue))
    if sat:
        out = _apply_blue_tables(out, "SaturationAdjustmentBlue", float(sat))
    if lum:
        out = _apply_blue_tables(out, "LuminanceAdjustmentBlue", float(lum))
    return out

# ---- HSL Orange/Purple 带（LR 8 色补齐 monetGPT 6 色缺口）----
_BAND_PRIOR = {
    "orange": {"hc": 19.0, "wl": 26.0, "wr": 27.0, "p": 1.0},
    "purple": {"hc": 285.0, "wl": 68.0, "wr": 52.0, "p": 1.0},
}
# 每算子类型的幅度/耦合参数缺省（拟合前占位）
_KIND_PRIOR = {
    "hue": {"amp": 0.0, "q": 0.0, "kc": 0.0, "kl": 0.0},
    "sat": {"gain": 0.0, "kl": 0.0, "kh": 0.0},
    "lum": {"amp": 0.0, "c0": 0.30, "m": 0.65, "beta": 0.5},
}

def _band_weight(hue01: np.ndarray, hc: float, wl: float, wr: float, p: float) -> np.ndarray:
    """raised-cosine^p 色相带权重：hue01 [0,1)，hc/wl/wr 单位度。峰值 1，支撑 [hc-wl, hc+wr]。"""
    d = np.mod(hue01 * 360.0 - hc + 180.0, 360.0) - 180.0
    t = np.where(d < 0.0, -d / max(wl, 1e-3), d / max(wr, 1e-3))
    w = 0.5 + 0.5 * np.cos(np.pi * np.minimum(t, 1.0))
    if abs(p - 1.0) > 1e-6:
        w = np.power(np.maximum(w, 0.0), p)
    return w.astype(np.float32, copy=False)


def _hls_f(light: np.ndarray) -> np.ndarray:
    """HLS 中 chroma = S · f(L)，f(L) = 2·min(L, 1-L)。"""
    return 2.0 * np.minimum(light, 1.0 - light)


def _apply_lr_band_hsl(img: np.ndarray, kind: str, prm: dict) -> np.ndarray:
    """对 float32 [0,1] RGB 应用一次 LR 风格的单带 HSL 调整。"""
    img = np.clip(img.astype(np.float32, copy=False), 0.0, 1.0)
    hls = rgb_to_hls_np(img).astype(np.float32, copy=False)
    hue, light, sat = hls[..., 0], hls[..., 1], hls[..., 2]
    chroma = (img.max(-1) - img.min(-1)).astype(np.float32, copy=False)

    w = _band_weight(hue, prm["hc"], prm["wl"], prm["wr"], prm["p"])
    # 近中性微弱守卫：只排除真正的灰（LR 对 c≈0.02 像素依然全量旋转）
    w = w * np.clip(chroma / 0.004, 0.0, 1.0)

    hue_out, light_out, sat_out = hue, light, sat
    if kind == "hue":
        rot = prm["amp"] * (1.0 + prm["q"] * (chroma - 0.2))
        hue_out = np.mod(hue + w * rot / 360.0, 1.0)
        sat_out = np.clip(sat * (1.0 + prm["kc"] * w), 0.0, 1.0)
        light_out = np.clip(light + prm["kl"] * chroma * w, 0.0, 1.0)
    elif kind == "sat":
        sat_out = np.clip(sat * (1.0 + prm["gain"] * w), 0.0, 1.0)
        light_out = np.clip(light + prm["kl"] * chroma * w, 0.0, 1.0)
        rot = prm["kh"] * np.clip(chroma / 0.3, 0.0, 1.0)
        hue_out = np.mod(hue + w * rot / 360.0, 1.0)
    elif kind == "lum":
        ramp = np.clip(chroma / max(prm["c0"], 1e-3), 0.0, 1.0)
        eff = prm["amp"] * w * ramp
        if prm["amp"] < 0.0:
            light_out = light * np.maximum(1.0 + eff, 0.0)
        else:
            m = float(np.clip(prm["m"], 0.0, 1.0))
            light_out = light + eff * (m * (1.0 - light) + (1.0 - m) * light)
        light_out = np.clip(light_out, 0.0, 1.0)
        # chroma 行为：const-S(beta=1) 与 const-chroma(beta=0) 之间混合
        f0, f1 = _hls_f(light), _hls_f(light_out)
        beta = float(np.clip(prm["beta"], 0.0, 1.0))
        sat_out = np.clip(sat * (f0 + beta * (f1 - f0)) / np.maximum(f1, 1e-4), 0.0, 1.0)
    else:  # pragma: no cover
        raise ValueError(f"unknown band-HSL kind: {kind}")

    out = hls_to_rgb_np(np.stack([hue_out, light_out, sat_out], axis=-1))
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def _interp_params(fit: dict, color: str, kind: str, lr_v: float) -> dict:
    """从 fits JSON 取该扫描值的参数；非扫描值在 (v=0 恒等) 与相邻拟合值间线性插值。

    fit["values"]: {"+100": {hc,wl,wr,p, amp/gain/... }, ...}（键 = LR 滑杆值）。
    幅度类参数向 v=0 线性归零；形状/常数类参数取最近拟合点。
    """
    base = {**_BAND_PRIOR[color], **_KIND_PRIOR[kind]}
    table = {float(k): v for k, v in (fit.get("values") or {}).items()}
    if not table:
        return base
    xs = sorted(table)
    amp_keys = {"amp", "gain", "kc", "kl", "kh", "q"} & set(base)
    if lr_v in table:
        return {**base, **table[lr_v]}
    # 插值网格：拟合点 + 隐式恒等点 v=0
    grid = sorted(set(xs + [0.0]))
    v = float(np.clip(lr_v, grid[0], grid[-1]))
    hi = next(i for i, x in enumerate(grid) if x >= v)
    lo = max(hi - 1, 0)
    x0, x1 = grid[lo], grid[hi]
    t = 0.0 if x1 == x0 else (v - x0) / (x1 - x0)
    p0 = {**base, **table.get(x0, {})} if x0 != 0.0 else None
    p1 = {**base, **table.get(x1, {})} if x1 != 0.0 else None
    if p0 is None and p1 is None:
        return base
    out = dict(p1 or p0)
    for k in out:
        a = (p0 or {}).get(k, 0.0 if k in amp_keys else out[k])
        b = (p1 or {}).get(k, 0.0 if k in amp_keys else out[k])
        if p0 is None:
            a = 0.0 if k in amp_keys else b
        if p1 is None:
            b = 0.0 if k in amp_keys else a
        out[k] = (1.0 - t) * a + t * b
    return out


_ORANGE_PURPLE_FITS: dict = json.loads(r'''{"HueAdjustmentOrange": {"values": {"-100": {"hc": 19.23655, "wl": 27.38773, "wr": 27.09137, "p": 1.02519, "amp": -26.36753, "q": 1.46335, "kc": 0.10414, "kl": 0.02857}, "+100": {"hc": 19.31764, "wl": 28.33787, "wr": 26.36895, "p": 1.05812, "amp": 28.57288, "q": 0.40625, "kc": -0.25015, "kl": -0.07852}, "-050": {"hc": 19.23655, "wl": 27.38773, "wr": 27.09137, "p": 1.02519, "amp": -13.49749, "q": 1.25343, "kc": 0.04952, "kl": 0.03127}, "+050": {"hc": 19.31764, "wl": 28.33787, "wr": 26.36895, "p": 1.05812, "amp": 13.85253, "q": 0.5965, "kc": -0.10976, "kl": -0.03214}}, "_fit_objective": {"-100": 1.2491, "+100": 0.9265, "-050": 0.822, "+050": 0.6794}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}, "HueAdjustmentPurple": {"values": {"-100": {"hc": 279.14598, "wl": 73.04998, "wr": 47.38091, "p": 1.30115, "amp": -65.086, "q": 1.58127, "kc": 0.24389, "kl": -0.1453}, "+100": {"hc": 271.85547, "wl": 62.78015, "wr": 65.85568, "p": 1.12782, "amp": 54.74644, "q": 0.27386, "kc": 0.10566, "kl": 0.0268}, "-050": {"hc": 279.14598, "wl": 73.04998, "wr": 47.38091, "p": 1.30115, "amp": -30.73561, "q": 0.93954, "kc": -0.02252, "kl": -0.00975}, "+050": {"hc": 271.85547, "wl": 62.78015, "wr": 65.85568, "p": 1.12782, "amp": 20.84928, "q": 0.54066, "kc": 0.03899, "kl": 0.03295}}, "_fit_objective": {"-100": 0.37, "+100": 0.4882, "-050": 0.3032, "+050": 0.3848}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}, "SaturationAdjustmentOrange": {"values": {"-100": {"hc": 20.80784, "wl": 32.83206, "wr": 24.05755, "p": 1.07191, "gain": -0.96442, "kl": -0.05973, "kh": -3.88423}, "+100": {"hc": 16.80336, "wl": 25.11681, "wr": 27.74453, "p": 1.1524, "gain": 0.93204, "kl": -0.21751, "kh": 6.03958}, "-050": {"hc": 20.80784, "wl": 32.83206, "wr": 24.05755, "p": 1.07191, "gain": -0.47674, "kl": -0.00997, "kh": -2.58788}, "+050": {"hc": 16.80336, "wl": 25.11681, "wr": 27.74453, "p": 1.1524, "gain": 0.33976, "kl": -0.07736, "kh": 2.29}}, "_fit_objective": {"-100": 1.1797, "+100": 1.4505, "-050": 0.7649, "+050": 0.7244}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}, "SaturationAdjustmentPurple": {"values": {"-100": {"hc": 283.56709, "wl": 72.12679, "wr": 57.24309, "p": 0.77394, "gain": -0.87603, "kl": 0.02204, "kh": -0.95381}, "+100": {"hc": 291.4852, "wl": 61.32219, "wr": 54.83353, "p": 1.36839, "gain": 0.80952, "kl": -0.27914, "kh": 1.15586}, "-050": {"hc": 283.56709, "wl": 72.12679, "wr": 57.24309, "p": 0.77394, "gain": -0.31621, "kl": -0.04368, "kh": -1.27111}, "+050": {"hc": 291.4852, "wl": 61.32219, "wr": 54.83353, "p": 1.36839, "gain": 0.23964, "kl": -0.08274, "kh": 4.88531}}, "_fit_objective": {"-100": 0.3845, "+100": 0.4195, "-050": 0.3272, "+050": 0.2998}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}, "LuminanceAdjustmentOrange": {"values": {"-100": {"hc": 20.6799, "wl": 31.88279, "wr": 25.33051, "p": 1.22253, "amp": -0.55213, "c0": 0.35029, "m": 0.72367, "beta": 0.50473}, "+100": {"hc": 20.45882, "wl": 31.2934, "wr": 25.08002, "p": 0.8222, "amp": 0.34662, "c0": 0.34888, "m": 0.14017, "beta": 0.51508}, "-050": {"hc": 20.6799, "wl": 31.88279, "wr": 25.33051, "p": 1.22253, "amp": -0.46699, "c0": 0.64413, "m": 1.03969, "beta": 0.47458}, "+050": {"hc": 20.45882, "wl": 31.2934, "wr": 25.08002, "p": 0.8222, "amp": 0.26204, "c0": 0.48959, "m": -0.29305, "beta": 0.50657}}, "_fit_objective": {"-100": 1.6653, "+100": 1.0068, "-050": 0.835, "+050": 0.6974}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}, "LuminanceAdjustmentPurple": {"values": {"-100": {"hc": 281.88809, "wl": 71.16551, "wr": 54.94637, "p": 0.80767, "amp": -0.52404, "c0": 0.29688, "m": 0.67631, "beta": 0.74971}, "+100": {"hc": 280.37735, "wl": 68.93903, "wr": 57.74532, "p": 0.78395, "amp": 0.53384, "c0": 0.38207, "m": 0.13547, "beta": 1.03383}, "-050": {"hc": 281.88809, "wl": 71.16551, "wr": 54.94637, "p": 0.80767, "amp": -0.29327, "c0": 0.35537, "m": 0.96331, "beta": 0.41003}, "+050": {"hc": 280.37735, "wl": 68.93903, "wr": 57.74532, "p": 0.78395, "amp": 0.32384, "c0": 0.45208, "m": -0.04537, "beta": 0.41665}}, "_fit_objective": {"-100": 0.4078, "+100": 0.348, "-050": 0.3023, "+050": 0.2911}, "_note": "obj = de_mean + 0.35*de_p95 (downscaled)"}}''')   # fits/<op>.json 烘焙


def apply_hsl_orange_purple_v2(img: np.ndarray, color: str, hue: float = 0.0,
                               sat: float = 0.0, lum: float = 0.0) -> np.ndarray:
    """LR Orange/Purple 带 HSL 调整（滑杆值 [-100,100]）。多滑杆按 hue→sat→lum 复合。"""
    color = str(color).strip().lower()
    out = img
    for pfx, kind, v in (("HueAdjustment", "hue", hue),
                         ("SaturationAdjustment", "sat", sat),
                         ("LuminanceAdjustment", "lum", lum)):
        if not v:
            continue
        op_fit = _ORANGE_PURPLE_FITS.get(f"{pfx}{color.capitalize()}") or {}
        prm = _interp_params(op_fit, color, kind, float(v))
        amp_key = "gain" if kind == "sat" else "amp"
        if abs(prm.get(amp_key, 0.0)) < 1e-9 and all(
            abs(prm.get(k, 0.0)) < 1e-9 for k in ("kc", "kl", "kh", "q") if k in prm
        ):
            continue
        out = _apply_lr_band_hsl(out, kind, prm)
    return out

# ---- 参数曲线（LR Parametric Tone Curve 四滑杆）----
_EPS = 1e-6
_LUMA709 = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _parametric_delta_curve(fit: dict, v: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    """由标定 fit 与滑杆值 v 得到 (grid, delta, blend) 采样曲线；v=0 或无 fit 返回 None。"""
    cd = (fit or {}).get("channel_delta")
    if not cd or abs(v) < _EPS:
        return None
    blend = float(cd.get("blend", 1.0))
    grid = np.asarray(cd["grid"], dtype=np.float32)
    knots = sorted(((float(k), np.asarray(c, dtype=np.float32))
                    for k, c in cd["curves"].items()), key=lambda t: t[0])
    vs = np.array([k for k, _ in knots], dtype=np.float32)
    if 0.0 not in vs:  # v=0 -> identity（Δ=0）作为插值锚点
        knots.append((0.0, np.zeros_like(grid)))
        knots.sort(key=lambda t: t[0])
        vs = np.array([k for k, _ in knots], dtype=np.float32)
    curves = np.stack([c for _, c in knots])          # (K, N)
    v = float(np.clip(v, vs[0], vs[-1]))
    i = int(np.searchsorted(vs, v))
    if i == 0 or abs(vs[min(i, len(vs) - 1)] - v) < _EPS:
        delta = curves[min(i, len(vs) - 1)]
    else:
        t = (v - vs[i - 1]) / max(vs[i] - vs[i - 1], _EPS)
        delta = (1.0 - t) * curves[i - 1] + t * curves[i]
    return grid, delta.astype(np.float32), blend


def apply_parametric_slider(image: np.ndarray, v: float, fit: dict) -> np.ndarray:
    """LR 参数曲线单滑杆：标定 Δ 曲线逐通道应用（点曲线语义）+ 少量 luma 乘性分量。"""
    gd = _parametric_delta_curve(fit, v)
    if gd is None:
        return np.asarray(image, dtype=np.float32)
    grid, delta, blend = gd
    img = np.asarray(image, dtype=np.float32)
    out = img + np.interp(img, grid, delta).astype(np.float32)
    if blend < 1.0 - _EPS:
        y = img @ _LUMA709
        y2 = np.clip(y + np.interp(y, grid, delta).astype(np.float32), 0.0, 1.0)
        lm = img * (y2 / (y + _EPS))[..., None]
        out = blend * out + (1.0 - blend) * lm
    return np.clip(out, 0.0, 1.0)


_PARAMETRIC_FITS: dict = json.loads(r'''{"ParametricShadows": {"channel_delta": {"grid": [0.005208, 0.015625, 0.026042, 0.036458, 0.046875, 0.057292, 0.067708, 0.078125, 0.088542, 0.098958, 0.109375, 0.119792, 0.130208, 0.140625, 0.151042, 0.161458, 0.171875, 0.182292, 0.192708, 0.203125, 0.213542, 0.223958, 0.234375, 0.244792, 0.255208, 0.265625, 0.276042, 0.286458, 0.296875, 0.307292, 0.317708, 0.328125, 0.338542, 0.348958, 0.359375, 0.369792, 0.380208, 0.390625, 0.401042, 0.411458, 0.421875, 0.432292, 0.442708, 0.453125, 0.463542, 0.473958, 0.484375, 0.494792, 0.505208, 0.515625, 0.526042, 0.536458, 0.546875, 0.557292, 0.567708, 0.578125, 0.588542, 0.598958, 0.609375, 0.619792, 0.630208, 0.640625, 0.651042, 0.661458, 0.671875, 0.682292, 0.692708, 0.703125, 0.713542, 0.723958, 0.734375, 0.744792, 0.755208, 0.765625, 0.776042, 0.786458, 0.796875, 0.807292, 0.817708, 0.828125, 0.838542, 0.848958, 0.859375, 0.869792, 0.880208, 0.890625, 0.901042, 0.911458, 0.921875, 0.932292, 0.942708, 0.953125, 0.963542, 0.973958, 0.984375, 0.994792], "curves": {"-100": [-0.01584, -0.019375, -0.0257, -0.035077, -0.045504, -0.058442, -0.070563, -0.078779, -0.087752, -0.098158, -0.108312, -0.118618, -0.127229, -0.133439, -0.139125, -0.143167, -0.144348, -0.143399, -0.141348, -0.138707, -0.134655, -0.129867, -0.124962, -0.118405, -0.111418, -0.105237, -0.098091, -0.090961, -0.084672, -0.077643, -0.070507, -0.063758, -0.057442, -0.051654, -0.045237, -0.039301, -0.034462, -0.028966, -0.023642, -0.019344, -0.014863, -0.010955, -0.008187, -0.005407, -0.002818, -0.000953, 0.000544, 0.00149, 0.001839, 0.001886, 0.001776, 0.00163, 0.001509, 0.001449, 0.001435, 0.001411, 0.001339, 0.00124, 0.001127, 0.001037, 0.000974, 0.000903, 0.000824, 0.000754, 0.000701, 0.000646, 0.000561, 0.000478, 0.000429, 0.000384, 0.00032, 0.000236, 0.000127, 3.9e-05, 1.9e-05, 6.2e-05, 0.000129, 0.000162, 0.000147, 0.000111, 9.5e-05, 0.000107, 0.000127, 0.000138, 0.000148, 0.000171, 0.000217, 0.000274, 0.000289, 0.000241, 0.000176, 0.000171, 0.000214, 0.000184, 8.1e-05, -0.0], "-50": [-0.010314, -0.0144, -0.021588, -0.03191, -0.042299, -0.05219, -0.059338, -0.063401, -0.067746, -0.072719, -0.076907, -0.079494, -0.079831, -0.078206, -0.07487, -0.070836, -0.067102, -0.063271, -0.060214, -0.058148, -0.055921, -0.053605, -0.051444, -0.048729, -0.045931, -0.04349, -0.040594, -0.037604, -0.034929, -0.031991, -0.029087, -0.026391, -0.023854, -0.021499, -0.018906, -0.016551, -0.014631, -0.012391, -0.010188, -0.008428, -0.006644, -0.005092, -0.003934, -0.002681, -0.001481, -0.000608, 8.7e-05, 0.00051, 0.000659, 0.000685, 0.000646, 0.000586, 0.000548, 0.000543, 0.000548, 0.000542, 0.000525, 0.000505, 0.000472, 0.000432, 0.000396, 0.000363, 0.000336, 0.000317, 0.000307, 0.000292, 0.000258, 0.000228, 0.000218, 0.000212, 0.000182, 0.000121, 2.9e-05, -4.2e-05, -4.9e-05, 3e-06, 7.4e-05, 0.000106, 8.8e-05, 4.5e-05, 2.7e-05, 3.7e-05, 5.7e-05, 6.6e-05, 6.7e-05, 7.1e-05, 9.3e-05, 0.00013, 0.000138, 9.6e-05, 4.5e-05, 5.8e-05, 0.000119, 0.000119, 4.8e-05, -1.7e-05], "50": [0.023939, 0.0236, 0.026583, 0.03414, 0.043484, 0.048952, 0.048214, 0.047128, 0.048422, 0.051643, 0.054855, 0.056455, 0.056132, 0.055007, 0.053989, 0.05341, 0.052942, 0.05222, 0.05144, 0.050696, 0.049639, 0.048391, 0.047083, 0.045258, 0.043198, 0.041213, 0.038774, 0.036269, 0.033944, 0.031191, 0.028312, 0.025592, 0.023099, 0.020854, 0.01833, 0.015907, 0.013872, 0.011593, 0.009442, 0.007715, 0.005858, 0.004202, 0.003046, 0.001909, 0.000825, 3.7e-05, -0.000532, -0.000826, -0.000909, -0.000923, -0.000925, -0.000932, -0.000927, -0.000898, -0.000862, -0.000827, -0.000794, -0.00076, -0.00071, -0.000657, -0.000613, -0.000561, -0.000514, -0.000475, -0.000435, -0.000391, -0.000336, -0.000279, -0.000222, -0.000161, -0.000127, -0.000144, -0.000197, -0.000237, -0.000226, -0.000155, -7.2e-05, -3.4e-05, -5.4e-05, -9.7e-05, -0.000112, -9e-05, -6.2e-05, -5.6e-05, -7.6e-05, -0.000107, -0.000124, -0.000116, -0.000117, -0.00015, -0.000175, -0.000126, -3.9e-05, -1.2e-05, -5.5e-05, -0.000101], "100": [0.047231, 0.04741, 0.053492, 0.067673, 0.085063, 0.096024, 0.096038, 0.095349, 0.099531, 0.107665, 0.11566, 0.120652, 0.121813, 0.121149, 0.120872, 0.121315, 0.121508, 0.121085, 0.120443, 0.119707, 0.118303, 0.116206, 0.113487, 0.109271, 0.104409, 0.099721, 0.093875, 0.087741, 0.082007, 0.075318, 0.068383, 0.061726, 0.055441, 0.049651, 0.043216, 0.037241, 0.032341, 0.026801, 0.021521, 0.017347, 0.013022, 0.009255, 0.006635, 0.004026, 0.00156, -0.000219, -0.001574, -0.002353, -0.002579, -0.002534, -0.002393, -0.002273, -0.002171, -0.00208, -0.002006, -0.001913, -0.001805, -0.001701, -0.001589, -0.00148, -0.001375, -0.00124, -0.001106, -0.001, -0.000908, -0.000815, -0.000699, -0.000578, -0.000472, -0.000364, -0.000294, -0.000288, -0.000319, -0.000341, -0.000316, -0.000237, -0.00015, -0.000114, -0.000138, -0.000185, -0.000199, -0.000173, -0.000141, -0.000134, -0.000166, -0.000225, -0.000275, -0.000294, -0.000303, -0.000327, -0.000329, -0.000253, -0.000146, -0.000107, -0.000141, -0.000182]}, "blend": 0.85}}, "ParametricDarks": {"channel_delta": {"grid": [0.005208, 0.015625, 0.026042, 0.036458, 0.046875, 0.057292, 0.067708, 0.078125, 0.088542, 0.098958, 0.109375, 0.119792, 0.130208, 0.140625, 0.151042, 0.161458, 0.171875, 0.182292, 0.192708, 0.203125, 0.213542, 0.223958, 0.234375, 0.244792, 0.255208, 0.265625, 0.276042, 0.286458, 0.296875, 0.307292, 0.317708, 0.328125, 0.338542, 0.348958, 0.359375, 0.369792, 0.380208, 0.390625, 0.401042, 0.411458, 0.421875, 0.432292, 0.442708, 0.453125, 0.463542, 0.473958, 0.484375, 0.494792, 0.505208, 0.515625, 0.526042, 0.536458, 0.546875, 0.557292, 0.567708, 0.578125, 0.588542, 0.598958, 0.609375, 0.619792, 0.630208, 0.640625, 0.651042, 0.661458, 0.671875, 0.682292, 0.692708, 0.703125, 0.713542, 0.723958, 0.734375, 0.744792, 0.755208, 0.765625, 0.776042, 0.786458, 0.796875, 0.807292, 0.817708, 0.828125, 0.838542, 0.848958, 0.859375, 0.869792, 0.880208, 0.890625, 0.901042, 0.911458, 0.921875, 0.932292, 0.942708, 0.953125, 0.963542, 0.973958, 0.984375, 0.994792], "curves": {"-100": [-0.016246, -0.01968, -0.025864, -0.035089, -0.045506, -0.05868, -0.070926, -0.079002, -0.08775, -0.098001, -0.108329, -0.119589, -0.130053, -0.139201, -0.150077, -0.160887, -0.169857, -0.179968, -0.189951, -0.198541, -0.208268, -0.217577, -0.225141, -0.233446, -0.241093, -0.246535, -0.251398, -0.255271, -0.257524, -0.258694, -0.258797, -0.25807, -0.257053, -0.255811, -0.253926, -0.251733, -0.249454, -0.246356, -0.242979, -0.239713, -0.235541, -0.231254, -0.227482, -0.222832, -0.217864, -0.213342, -0.208081, -0.202949, -0.198602, -0.193563, -0.188459, -0.183948, -0.178584, -0.173165, -0.168435, -0.162852, -0.157204, -0.152342, -0.146639, -0.140765, -0.135652, -0.13004, -0.124092, -0.118063, -0.112201, -0.106794, -0.100571, -0.094514, -0.089343, -0.083414, -0.077685, -0.07291, -0.067504, -0.062136, -0.057598, -0.052403, -0.047182, -0.042924, -0.03832, -0.033743, -0.030002, -0.025901, -0.021881, -0.018584, -0.01498, -0.011479, -0.008838, -0.006372, -0.004403, -0.003245, -0.002267, -0.001445, -0.000893, -0.000405, -0.000112, -1.1e-05], "-50": [-0.012661, -0.01635, -0.022937, -0.032634, -0.042992, -0.054439, -0.064407, -0.071417, -0.079551, -0.088841, -0.097079, -0.104256, -0.109474, -0.112814, -0.115908, -0.118422, -0.119699, -0.120341, -0.120786, -0.12122, -0.121713, -0.122179, -0.122799, -0.123889, -0.124836, -0.124749, -0.123421, -0.121671, -0.120259, -0.118889, -0.117487, -0.116071, -0.114675, -0.113306, -0.111692, -0.110152, -0.108809, -0.10709, -0.105237, -0.103502, -0.101405, -0.099348, -0.097613, -0.095487, -0.093177, -0.091042, -0.088594, -0.086259, -0.084316, -0.082068, -0.079798, -0.077811, -0.075461, -0.073097, -0.071061, -0.068684, -0.066291, -0.064234, -0.061826, -0.05935, -0.057191, -0.054807, -0.052272, -0.049704, -0.047219, -0.044946, -0.042351, -0.039828, -0.037675, -0.035226, -0.032885, -0.030956, -0.028773, -0.026588, -0.024706, -0.022516, -0.020294, -0.018483, -0.016544, -0.014635, -0.013074, -0.011343, -0.009631, -0.008225, -0.006689, -0.005196, -0.004055, -0.002973, -0.002116, -0.001644, -0.001269, -0.000912, -0.000586, -0.00025, -6.4e-05, -1.5e-05], "50": [0.019718, 0.020397, 0.023868, 0.030768, 0.038503, 0.043744, 0.045083, 0.045739, 0.048162, 0.052444, 0.057197, 0.061146, 0.063168, 0.064051, 0.065473, 0.067375, 0.068914, 0.070396, 0.071913, 0.073489, 0.075572, 0.077739, 0.079725, 0.08218, 0.084544, 0.086199, 0.087646, 0.088895, 0.089874, 0.090782, 0.091522, 0.092145, 0.092757, 0.093274, 0.093673, 0.09389, 0.093913, 0.093733, 0.093395, 0.092948, 0.092249, 0.091427, 0.090607, 0.089484, 0.088172, 0.086871, 0.085282, 0.083666, 0.082195, 0.080369, 0.07843, 0.076641, 0.074485, 0.07231, 0.070394, 0.06808, 0.065698, 0.063616, 0.06116, 0.05863, 0.056433, 0.054029, 0.051488, 0.048921, 0.046435, 0.044138, 0.041492, 0.038934, 0.036787, 0.034327, 0.031903, 0.029814, 0.027414, 0.025057, 0.023144, 0.021066, 0.01903, 0.017333, 0.015375, 0.013352, 0.011711, 0.00997, 0.008288, 0.006916, 0.005452, 0.004087, 0.003118, 0.002247, 0.001508, 0.001007, 0.000606, 0.000379, 0.000269, 5.9e-05, -0.000153, -0.000263], "100": [0.03547, 0.038392, 0.045595, 0.057569, 0.070018, 0.07933, 0.083656, 0.08648, 0.092055, 0.100785, 0.11038, 0.118946, 0.12413, 0.127083, 0.131222, 0.136242, 0.140399, 0.14465, 0.148943, 0.153185, 0.158685, 0.164438, 0.16976, 0.176408, 0.182957, 0.187833, 0.192487, 0.196762, 0.200338, 0.203994, 0.207353, 0.210368, 0.213214, 0.215715, 0.218145, 0.220064, 0.22124, 0.222091, 0.222538, 0.222454, 0.221823, 0.220852, 0.219644, 0.217681, 0.215201, 0.212557, 0.209109, 0.205473, 0.202077, 0.197784, 0.193191, 0.18892, 0.183662, 0.178232, 0.173343, 0.167391, 0.161261, 0.155886, 0.149516, 0.142954, 0.137275, 0.131081, 0.124555, 0.118021, 0.111745, 0.10599, 0.09937, 0.092935, 0.087479, 0.081252, 0.075209, 0.070115, 0.064317, 0.058605, 0.053916, 0.048766, 0.043714, 0.039553, 0.03487, 0.030129, 0.026364, 0.022419, 0.018562, 0.015324, 0.011786, 0.008484, 0.006208, 0.004303, 0.002804, 0.001857, 0.001129, 0.000757, 0.000601, 0.000205, -0.000235, -0.000475]}, "blend": 0.85}}, "ParametricLights": {"channel_delta": {"grid": [0.005208, 0.015625, 0.026042, 0.036458, 0.046875, 0.057292, 0.067708, 0.078125, 0.088542, 0.098958, 0.109375, 0.119792, 0.130208, 0.140625, 0.151042, 0.161458, 0.171875, 0.182292, 0.192708, 0.203125, 0.213542, 0.223958, 0.234375, 0.244792, 0.255208, 0.265625, 0.276042, 0.286458, 0.296875, 0.307292, 0.317708, 0.328125, 0.338542, 0.348958, 0.359375, 0.369792, 0.380208, 0.390625, 0.401042, 0.411458, 0.421875, 0.432292, 0.442708, 0.453125, 0.463542, 0.473958, 0.484375, 0.494792, 0.505208, 0.515625, 0.526042, 0.536458, 0.546875, 0.557292, 0.567708, 0.578125, 0.588542, 0.598958, 0.609375, 0.619792, 0.630208, 0.640625, 0.651042, 0.661458, 0.671875, 0.682292, 0.692708, 0.703125, 0.713542, 0.723958, 0.734375, 0.744792, 0.755208, 0.765625, 0.776042, 0.786458, 0.796875, 0.807292, 0.817708, 0.828125, 0.838542, 0.848958, 0.859375, 0.869792, 0.880208, 0.890625, 0.901042, 0.911458, 0.921875, 0.932292, 0.942708, 0.953125, 0.963542, 0.973958, 0.984375, 0.994792], "curves": {"-100": [0.01062, 0.007366, 0.005421, 0.006524, 0.010752, 0.012939, 0.009602, 0.006297, 0.004996, 0.005491, 0.006443, 0.005759, 0.003034, -0.00053, -0.004546, -0.008108, -0.011227, -0.015155, -0.019059, -0.022271, -0.025939, -0.029711, -0.033176, -0.037505, -0.042101, -0.046464, -0.051961, -0.057646, -0.062801, -0.068836, -0.075317, -0.081784, -0.087981, -0.093797, -0.100546, -0.107116, -0.112844, -0.119724, -0.126639, -0.132581, -0.139353, -0.145872, -0.151278, -0.157612, -0.164077, -0.169584, -0.175613, -0.181269, -0.18578, -0.190628, -0.195234, -0.198958, -0.203003, -0.206823, -0.20987, -0.21314, -0.2162, -0.218571, -0.221046, -0.223344, -0.225035, -0.226569, -0.22788, -0.228755, -0.229208, -0.229212, -0.228753, -0.227969, -0.226885, -0.225147, -0.223128, -0.221117, -0.218492, -0.215612, -0.21283, -0.209196, -0.205217, -0.20163, -0.197317, -0.192562, -0.187868, -0.181519, -0.174448, -0.167878, -0.159779, -0.15092, -0.142424, -0.131368, -0.119801, -0.109917, -0.097097, -0.082507, -0.067078, -0.04402, -0.023967, -0.01364], "-50": [0.006376, 0.004388, 0.003163, 0.003723, 0.00615, 0.007506, 0.005704, 0.003843, 0.003116, 0.003481, 0.004189, 0.004004, 0.002627, 0.000782, -0.001188, -0.002843, -0.004283, -0.006137, -0.007985, -0.009478, -0.011118, -0.012725, -0.014095, -0.01572, -0.017448, -0.019164, -0.021424, -0.02381, -0.026017, -0.028638, -0.031425, -0.034122, -0.036646, -0.039007, -0.041796, -0.044574, -0.047057, -0.05007, -0.053102, -0.055717, -0.058729, -0.06166, -0.06412, -0.067022, -0.070014, -0.072598, -0.075463, -0.078175, -0.080375, -0.08282, -0.085225, -0.087235, -0.089448, -0.091547, -0.093256, -0.095165, -0.097026, -0.098543, -0.100187, -0.101741, -0.102917, -0.104042, -0.105087, -0.105937, -0.106561, -0.106913, -0.107069, -0.107043, -0.106813, -0.106314, -0.105676, -0.104991, -0.104033, -0.102913, -0.101736, -0.100105, -0.098274, -0.09661, -0.094606, -0.092358, -0.090046, -0.086837, -0.083261, -0.07998, -0.075987, -0.071627, -0.067396, -0.061825, -0.056028, -0.051121, -0.044706, -0.037483, -0.030388, -0.020173, -0.011154, -0.006381], "50": [5.9e-05, -0.000494, -0.001642, -0.003833, -0.006842, -0.008969, -0.008178, -0.006833, -0.006902, -0.009136, -0.012579, -0.014401, -0.012561, -0.008428, -0.00379, -0.000415, 0.001916, 0.004616, 0.007108, 0.008843, 0.010534, 0.012143, 0.013461, 0.014932, 0.016488, 0.018179, 0.020655, 0.023363, 0.025766, 0.028435, 0.031244, 0.034071, 0.036846, 0.039556, 0.0428, 0.045961, 0.048673, 0.051954, 0.055368, 0.058472, 0.062152, 0.065769, 0.068871, 0.072614, 0.07649, 0.079901, 0.083854, 0.087767, 0.091097, 0.094869, 0.09855, 0.101623, 0.105121, 0.108575, 0.111477, 0.11475, 0.117933, 0.120534, 0.123414, 0.126242, 0.128561, 0.130962, 0.133332, 0.135433, 0.137196, 0.138505, 0.139615, 0.140387, 0.140662, 0.140492, 0.139954, 0.139116, 0.137802, 0.136275, 0.134726, 0.132618, 0.130231, 0.12795, 0.124977, 0.121469, 0.11776, 0.112543, 0.106607, 0.100901, 0.093636, 0.085685, 0.07861, 0.070579, 0.062932, 0.056886, 0.049384, 0.040763, 0.031982, 0.021098, 0.013218, 0.009853], "100": [0.000784, 2.2e-05, -0.001272, -0.003502, -0.00645, -0.008429, -0.007538, -0.006502, -0.007454, -0.010762, -0.014799, -0.016751, -0.015048, -0.011192, -0.006597, -0.002471, 0.001722, 0.007653, 0.013547, 0.017962, 0.022384, 0.026491, 0.029714, 0.03324, 0.037084, 0.041489, 0.047979, 0.054944, 0.061076, 0.068033, 0.075513, 0.083075, 0.090405, 0.097413, 0.105719, 0.113938, 0.121233, 0.130127, 0.139216, 0.147292, 0.156832, 0.166236, 0.174273, 0.183876, 0.19371, 0.20218, 0.211761, 0.221033, 0.228635, 0.236875, 0.24464, 0.250788, 0.257332, 0.263405, 0.268054, 0.272773, 0.276973, 0.279932, 0.282591, 0.284696, 0.285825, 0.286333, 0.286096, 0.284706, 0.28229, 0.27881, 0.27329, 0.266952, 0.260751, 0.252994, 0.24513, 0.238181, 0.229859, 0.22129, 0.213819, 0.205139, 0.196328, 0.188767, 0.179747, 0.169968, 0.161181, 0.150782, 0.140173, 0.131088, 0.120607, 0.109747, 0.100194, 0.088774, 0.077429, 0.068559, 0.058308, 0.047285, 0.036827, 0.024297, 0.01529, 0.011462]}, "blend": 0.75}}, "ParametricHighlights": {"channel_delta": {"grid": [0.005208, 0.015625, 0.026042, 0.036458, 0.046875, 0.057292, 0.067708, 0.078125, 0.088542, 0.098958, 0.109375, 0.119792, 0.130208, 0.140625, 0.151042, 0.161458, 0.171875, 0.182292, 0.192708, 0.203125, 0.213542, 0.223958, 0.234375, 0.244792, 0.255208, 0.265625, 0.276042, 0.286458, 0.296875, 0.307292, 0.317708, 0.328125, 0.338542, 0.348958, 0.359375, 0.369792, 0.380208, 0.390625, 0.401042, 0.411458, 0.421875, 0.432292, 0.442708, 0.453125, 0.463542, 0.473958, 0.484375, 0.494792, 0.505208, 0.515625, 0.526042, 0.536458, 0.546875, 0.557292, 0.567708, 0.578125, 0.588542, 0.598958, 0.609375, 0.619792, 0.630208, 0.640625, 0.651042, 0.661458, 0.671875, 0.682292, 0.692708, 0.703125, 0.713542, 0.723958, 0.734375, 0.744792, 0.755208, 0.765625, 0.776042, 0.786458, 0.796875, 0.807292, 0.817708, 0.828125, 0.838542, 0.848958, 0.859375, 0.869792, 0.880208, 0.890625, 0.901042, 0.911458, 0.921875, 0.932292, 0.942708, 0.953125, 0.963542, 0.973958, 0.984375, 0.994792], "curves": {"-100": [0.004375, 0.002448, 0.000769, -5.5e-05, 0.000378, 0.000989, 0.00074, 0.000372, 0.000238, 0.000286, 0.000356, 0.00034, 0.000289, 0.000287, 0.000351, 0.000431, 0.000462, 0.000425, 0.000347, 0.000278, 0.000228, 0.000194, 0.000184, 0.000236, 0.000369, 0.000526, 0.000662, 0.000729, 0.000763, 0.000839, 0.000946, 0.001041, 0.001122, 0.001207, 0.001322, 0.001443, 0.001559, 0.001712, 0.001888, 0.002042, 0.002179, 0.002289, 0.002437, 0.002708, 0.002997, 0.003195, 0.003329, 0.003318, 0.002976, 0.002004, 0.000494, -0.001458, -0.004439, -0.007895, -0.011408, -0.016107, -0.021239, -0.026036, -0.032014, -0.038346, -0.044032, -0.050505, -0.057585, -0.064911, -0.072067, -0.078721, -0.086476, -0.094027, -0.100332, -0.107322, -0.113881, -0.119136, -0.124816, -0.130182, -0.134294, -0.138429, -0.14218, -0.144908, -0.147521, -0.149683, -0.150525, -0.150096, -0.148808, -0.146986, -0.143931, -0.139635, -0.134014, -0.124947, -0.114687, -0.105589, -0.093552, -0.07959, -0.064608, -0.042307, -0.023033, -0.01315], "-50": [0.00286, 0.001553, 0.000416, -0.00013, 0.000202, 0.000655, 0.000474, 0.000201, 0.000112, 0.000168, 0.000234, 0.000217, 0.000167, 0.000156, 0.0002, 0.000258, 0.000279, 0.00025, 0.00019, 0.000146, 0.000123, 0.000113, 0.000108, 0.000128, 0.000184, 0.000256, 0.000325, 0.000374, 0.000403, 0.000435, 0.000468, 0.000497, 0.000536, 0.000582, 0.000638, 0.000692, 0.00074, 0.0008, 0.000865, 0.000918, 0.000958, 0.000991, 0.001054, 0.00117, 0.001272, 0.001313, 0.001312, 0.001255, 0.001063, 0.00058, -0.000142, -0.001034, -0.002349, -0.003851, -0.005359, -0.007352, -0.009509, -0.011513, -0.014017, -0.016691, -0.019119, -0.021906, -0.024953, -0.028101, -0.031182, -0.034058, -0.037411, -0.040672, -0.043391, -0.046411, -0.049262, -0.051579, -0.054123, -0.056543, -0.058393, -0.060235, -0.061895, -0.063122, -0.064349, -0.065414, -0.065893, -0.065823, -0.065363, -0.064635, -0.06336, -0.061548, -0.059158, -0.055279, -0.050875, -0.046894, -0.041398, -0.034957, -0.028399, -0.018893, -0.010513, -0.006077], "50": [0.000344, 4.6e-05, -0.000235, -0.000387, -0.000301, -0.000141, -0.000161, -0.00023, -0.000243, -0.000209, -0.00019, -0.000229, -0.000288, -0.000319, -0.000318, -0.000298, -0.000273, -0.000248, -0.000237, -0.000229, -0.000195, -0.000147, -0.000123, -0.000155, -0.000242, -0.000345, -0.000413, -0.00041, -0.000389, -0.000424, -0.000518, -0.000616, -0.000684, -0.000727, -0.000781, -0.000841, -0.000906, -0.001002, -0.001095, -0.00115, -0.001187, -0.001228, -0.001283, -0.001381, -0.001499, -0.001589, -0.001622, -0.001561, -0.001351, -0.000866, -0.000154, 0.000751, 0.00214, 0.003737, 0.005311, 0.00735, 0.009541, 0.011579, 0.01414, 0.01689, 0.019418, 0.022367, 0.02566, 0.029153, 0.032618, 0.035861, 0.039633, 0.0433, 0.046355, 0.049711, 0.052822, 0.055282, 0.057953, 0.060552, 0.062683, 0.065012, 0.067224, 0.068835, 0.0703, 0.071489, 0.072074, 0.072202, 0.071952, 0.071319, 0.070007, 0.068049, 0.06533, 0.060544, 0.054702, 0.04946, 0.043221, 0.036441, 0.029224, 0.01945, 0.01204, 0.008783], "100": [0.000341, 3.9e-05, -0.000243, -0.000398, -0.000317, -0.00016, -0.00017, -0.000234, -0.00025, -0.000224, -0.000214, -0.000264, -0.000338, -0.000406, -0.000488, -0.000571, -0.000617, -0.000626, -0.000609, -0.000564, -0.000467, -0.000349, -0.000279, -0.00031, -0.000471, -0.000687, -0.000857, -0.000888, -0.000886, -0.001003, -0.001227, -0.001435, -0.001568, -0.001669, -0.001806, -0.001958, -0.002107, -0.002324, -0.002559, -0.002725, -0.002852, -0.002973, -0.003145, -0.003456, -0.003809, -0.004062, -0.004226, -0.004212, -0.003844, -0.002807, -0.001184, 0.000947, 0.004245, 0.008101, 0.012039, 0.0173, 0.023014, 0.028326, 0.034983, 0.042136, 0.048692, 0.05627, 0.064609, 0.07327, 0.08172, 0.089453, 0.098199, 0.106471, 0.113081, 0.120011, 0.126141, 0.1306, 0.134967, 0.138836, 0.141539, 0.143911, 0.145732, 0.146503, 0.146122, 0.144245, 0.140392, 0.133283, 0.124726, 0.11692, 0.108175, 0.099577, 0.092325, 0.083608, 0.0746, 0.067003, 0.057555, 0.047023, 0.036735, 0.024243, 0.015243, 0.011413]}, "blend": 0.65}}}''')   # fits/Parametric*.json 烘焙


def apply_lr_parametric(image: np.ndarray, op_name: str, v: float) -> np.ndarray:
    """op_name ∈ {"ParametricShadows","ParametricDarks","ParametricLights","ParametricHighlights"}。"""
    return apply_parametric_slider(image, float(v), _PARAMETRIC_FITS.get(op_name) or {})

# ---- Camera Calibration 面板（线性光域 3x3 通道混合；ShadowTint 实测恒等）----
_CALIB_OPS = ("CalibRedHue", "CalibRedSaturation", "CalibGreenHue", "CalibGreenSaturation",
              "CalibBlueHue", "CalibBlueSaturation", "CalibShadowTint")

def _calib_anchors(fit: dict):
    """fits json -> 排序锚点 [(v, M(3x3), b(3))]，隐含 v=0 恒等锚点。"""
    mats = fit.get("matrix") or {}
    offs = fit.get("offset") or {}
    pts = {0.0: (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))}
    for k, m in mats.items():
        v = float(k)
        b = np.asarray(offs.get(k, [0.0, 0.0, 0.0]), dtype=np.float32)
        pts[v] = (np.asarray(m, dtype=np.float32), b)
    return sorted(pts.items())


def _interp_matrix(anchors, v: float):
    """锚点矩阵/偏置的逐元素分段线性插值（端点外夹紧）。"""
    vs = [p[0] for p in anchors]
    if v <= vs[0]:
        return anchors[0][1]
    if v >= vs[-1]:
        return anchors[-1][1]
    for i in range(len(vs) - 1):
        v0, v1 = vs[i], vs[i + 1]
        if v0 <= v <= v1:
            t = (v - v0) / (v1 - v0)
            M0, b0 = anchors[i][1]
            M1, b1 = anchors[i + 1][1]
            return (1 - t) * M0 + t * M1, (1 - t) * b0 + t * b1
    return anchors[-1][1]


def apply_camera_calibration(img: np.ndarray, M: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
    """线性光域 3x3 混合（+offset）：LR 校准面板对 JPEG 输入的等效变换。"""
    lin = _srgb_to_linear(img)
    out = lin @ M.T.astype(np.float32)
    if b is not None:
        out = out + b.astype(np.float32)
    return _linear_to_srgb(out)


_CALIB_FITS: dict = json.loads(r'''{"CalibRedHue": {"matrix": {"-100": [[1.035922, -0.14696, 0.106862], [-0.165885, 1.212226, -0.0453], [0.222391, 0.069971, 0.710253]], "-050": [[1.016999, -0.070566, 0.050957], [-0.082794, 1.102862, -0.020038], [0.111408, 0.035166, 0.855423]], "+050": [[0.977026, 0.076286, -0.052186], [0.087932, 0.885323, 0.025423], [-0.118489, -0.022154, 1.136483]], "+100": [[0.958083, 0.145274, -0.100512], [0.173541, 0.77779, 0.046891], [-0.243011, -0.037525, 1.272047]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibRedSaturation": {"matrix": {"-100": [[0.799086, 0.137055, 0.060845], [0.218774, 0.720716, 0.06069], [0.219082, 0.143902, 0.639941]], "-050": [[0.896978, 0.071462, 0.029621], [0.110526, 0.857799, 0.031438], [0.109583, 0.073348, 0.819309]], "+050": [[1.097523, -0.065034, -0.030958], [-0.107301, 1.130667, -0.025156], [-0.117199, -0.058339, 1.170683]], "+100": [[1.204125, -0.145742, -0.053645], [-0.219631, 1.271631, -0.055435], [-0.241479, -0.101592, 1.332932]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibGreenHue": {"matrix": {"-100": [[0.733665, 0.430627, -0.160678], [0.035375, 0.954678, 0.009153], [-0.043171, -0.285596, 1.322714]], "-050": [[0.867147, 0.214772, -0.080104], [0.017656, 0.976472, 0.00499], [-0.021389, -0.144324, 1.162741]], "+050": [[1.123729, -0.203508, 0.076479], [-0.012956, 1.012259, 0.00022], [0.015767, 0.154658, 0.83103]], "+100": [[1.247631, -0.409421, 0.155938], [-0.026661, 1.028572, -0.001919], [0.033811, 0.303982, 0.664727]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibGreenSaturation": {"matrix": {"-100": [[0.646054, 0.33701, 0.016786], [0.046776, 0.942801, 0.010982], [0.043479, 0.36086, 0.597639]], "-050": [[0.823062, 0.16677, 0.010057], [0.023916, 0.96992, 0.005991], [0.020677, 0.182204, 0.798464]], "+050": [[1.169123, -0.156038, -0.013957], [-0.019811, 1.018307, -0.000137], [-0.026626, -0.167993, 1.191122]], "+100": [[1.340403, -0.316136, -0.025005], [-0.04297, 1.045427, -0.005412], [-0.05506, -0.328789, 1.37732]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibBlueHue": {"matrix": {"-100": [[1.320187, 0.339156, -0.67189], [-0.069866, 0.669703, 0.399278], [0.004598, 0.042548, 0.953556]], "-050": [[1.159499, 0.175168, -0.340609], [-0.035461, 0.83196, 0.202337], [0.002148, 0.021791, 0.976205]], "+050": [[0.823879, -0.161961, 0.339081], [0.043134, 1.150608, -0.193702], [-0.005463, -0.014311, 1.018636]], "+100": [[0.641385, -0.327688, 0.686843], [0.082276, 1.301743, -0.383862], [-0.007936, -0.026738, 1.033646]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibBlueSaturation": {"matrix": {"-100": [[0.620124, 0.058124, 0.326128], [0.01132, 0.653441, 0.335915], [0.00336, 0.049308, 0.946322]], "-050": [[0.81213, 0.031443, 0.159547], [0.006502, 0.827012, 0.166583], [0.000592, 0.02627, 0.972385]], "+050": [[1.17505, -0.01308, -0.168168], [-0.003933, 1.16375, -0.162548], [-0.004774, -0.016425, 1.020923]], "+100": [[1.349406, -0.028573, -0.33304], [-0.010762, 1.333041, -0.327708], [-0.006213, -0.03998, 1.046542]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}, "CalibShadowTint": {"matrix": {"-100": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "-050": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "+050": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "+100": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}, "offset": {"-100": [0.0, 0.0, 0.0], "-050": [0.0, 0.0, 0.0], "+050": [0.0, 0.0, 0.0], "+100": [0.0, 0.0, 0.0]}}}''')   # fits/Calib*.json 烘焙


def apply_lr_calibration_panel(img: np.ndarray, sliders: dict) -> np.ndarray:
    """校准面板复合应用。sliders: {"RedHue": v, ...}（LR crs 参数名，值 [-100,100]）。

    各滑杆的插值矩阵先乘合成单个仿射 (M, b) 再一次线性域应用（近单位阵、乘序不敏感）。
    """
    M = np.eye(3, dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    active = False
    for name, v in sliders.items():
        f = _CALIB_FITS.get(f"Calib{name}")
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not f or not f.get("matrix") or abs(v) < 1e-9:
            continue
        Mi, bi = _interp_matrix(_calib_anchors(f), v)
        Mi = np.asarray(Mi, dtype=np.float64)
        M = Mi @ M
        b = Mi @ b + np.asarray(bi, dtype=np.float64)
        active = True
    if not active:
        return np.asarray(img, dtype=np.float32)
    return apply_camera_calibration(np.asarray(img, dtype=np.float32),
                                    M.astype(np.float32), b.astype(np.float32))

# ---- SplitToning / ColorGrade（统一 3-way color grade 核心；常量由 scratch/colorgrade_genmod5.py 生成）----
_CG_NB = 48
_CG_NBX = 40
_CG_NBL = 40
_CG_W = {
    "shadow": [0.07442, 0.12530, 0.23735, 0.36128, 0.45552, 0.56268, 0.64827, 0.75824, 0.81684, 0.86423, 0.90372, 0.94033, 0.96702, 0.98517, 0.99553, 1.00000, 0.99637, 0.98588, 0.97194, 0.95269, 0.93432, 0.91982, 0.91056, 0.90792, 0.90804, 0.90535, 0.89993, 0.89422, 0.88997, 0.88593, 0.87890, 0.85872, 0.82460, 0.78711, 0.74688, 0.70726, 0.66986, 0.63217, 0.58956, 0.52011, 0.43210, 0.35987, 0.29259, 0.20410, 0.08759, 0.02624, 0.00482, 0.00025],
    "highlight": [0.00038, 0.00069, 0.00231, 0.00429, 0.00954, 0.02239, 0.03607, 0.08352, 0.11693, 0.15020, 0.17265, 0.20310, 0.22765, 0.26235, 0.29365, 0.29403, 0.30074, 0.31324, 0.32663, 0.34316, 0.35842, 0.37637, 0.39558, 0.41188, 0.43097, 0.45550, 0.48645, 0.51946, 0.55692, 0.59782, 0.64292, 0.69786, 0.75778, 0.81347, 0.86427, 0.90702, 0.94036, 0.96612, 0.98294, 0.99316, 1.00000, 0.99574, 0.97841, 0.94577, 0.88102, 0.77334, 0.59448, 0.21002],
    "mid": [0.00029, 0.00046, 0.00230, 0.00654, 0.01793, 0.04139, 0.06280, 0.13298, 0.19186, 0.24970, 0.30152, 0.37274, 0.43381, 0.51112, 0.58389, 0.61650, 0.66152, 0.70821, 0.75321, 0.80019, 0.84424, 0.88607, 0.92439, 0.95387, 0.97434, 0.98830, 0.99592, 0.99916, 1.00000, 0.99879, 0.99181, 0.97289, 0.93878, 0.89638, 0.84761, 0.79701, 0.74547, 0.69462, 0.64095, 0.56371, 0.46074, 0.37667, 0.30067, 0.22211, 0.11238, 0.03881, 0.00820, 0.00040],
    "global": [0.02777, 0.04565, 0.08150, 0.12586, 0.16322, 0.20644, 0.24609, 0.31383, 0.35623, 0.39561, 0.42833, 0.46847, 0.50250, 0.54381, 0.58263, 0.60020, 0.62212, 0.64569, 0.66940, 0.69401, 0.71886, 0.74551, 0.77304, 0.79773, 0.82044, 0.84241, 0.86361, 0.88390, 0.90488, 0.92613, 0.94676, 0.96570, 0.98111, 0.99096, 0.99691, 1.00000, 0.99901, 0.99516, 0.98295, 0.95403, 0.90872, 0.85986, 0.80801, 0.73483, 0.60882, 0.49136, 0.36437, 0.11683],
}
_CG_CAPS = {
    "shadow": ([30.0, 60.0, 100.0], [[0.29914, 0.32599, 0.45531, 0.61950, 0.76315, 0.90914, 0.99638, 1.06890, 1.15694, 1.23259, 1.28417, 1.32569, 1.34903, 1.33870, 1.31174, 1.26365, 1.21803, 1.16965, 1.12580, 1.10939, 1.09661, 1.09847, 1.09607, 1.09622, 1.08335, 1.05737, 1.01495, 0.97991, 0.95545, 0.93039, 0.90543, 0.85486, 0.77477, 0.72054, 0.70666, 0.72363, 0.68567, 0.70097, 0.83176, 0.85736], [0.56682, 0.61271, 0.64950, 0.69922, 0.78431, 0.85519, 0.93722, 1.03827, 1.15474, 1.25761, 1.33604, 1.38428, 1.39754, 1.38288, 1.34466, 1.29596, 1.24282, 1.19453, 1.15419, 1.12769, 1.11684, 1.11248, 1.10857, 1.10369, 1.09571, 1.07774, 1.04293, 0.99962, 0.95701, 0.91988, 0.87968, 0.82750, 0.77291, 0.72367, 0.67046, 0.60906, 0.55834, 0.52595, 0.50658, 0.34409], [0.28957, 0.22320, 0.28583, 0.38607, 0.45930, 0.57957, 0.73202, 0.87681, 1.04057, 1.18440, 1.30663, 1.40757, 1.44667, 1.43377, 1.39225, 1.33306, 1.27352, 1.21814, 1.17657, 1.14669, 1.13882, 1.14136, 1.13952, 1.13710, 1.13272, 1.11686, 1.08235, 1.03345, 0.98752, 0.94611, 0.91204, 0.85539, 0.80888, 0.78861, 0.77651, 0.76075, 0.72924, 0.76529, 0.85511, 0.87801]]),
    "highlight": ([30.0, 60.0, 100.0], [[0.02310, 0.10855, 0.17242, 0.21973, 0.23463, 0.19560, 0.15112, 0.12549, 0.12427, 0.15896, 0.22253, 0.26137, 0.30998, 0.36192, 0.42271, 0.48666, 0.56014, 0.63034, 0.69848, 0.77352, 0.85622, 0.95582, 1.06980, 1.17788, 1.30485, 1.45203, 1.64119, 1.82763, 1.98275, 2.09859, 2.18708, 2.27718, 2.33953, 2.36035, 2.31777, 2.24955, 2.15876, 1.98563, 1.49771, 1.06717], [0.06220, 0.07565, 0.09326, 0.10838, 0.12933, 0.14025, 0.14338, 0.13865, 0.14519, 0.17141, 0.21660, 0.25343, 0.29965, 0.35521, 0.41935, 0.48106, 0.55245, 0.62512, 0.70393, 0.78654, 0.87141, 0.98083, 1.10080, 1.21748, 1.35518, 1.51156, 1.69313, 1.86680, 2.01250, 2.13594, 2.23885, 2.33237, 2.40629, 2.42155, 2.33459, 2.20409, 2.01344, 1.71299, 1.35983, 0.99500], [0.02734, 0.12313, 0.18679, 0.20999, 0.23423, 0.21783, 0.18787, 0.15620, 0.14819, 0.18192, 0.24586, 0.28460, 0.33690, 0.39741, 0.45936, 0.51976, 0.59345, 0.66247, 0.73940, 0.81842, 0.89582, 0.99677, 1.10935, 1.21795, 1.34629, 1.49625, 1.67646, 1.85071, 1.99373, 2.10024, 2.18055, 2.25625, 2.32332, 2.35414, 2.31969, 2.17119, 1.89645, 1.48857, 1.18138, 0.87555]]),
    "mid": ([60.0, 100.0], [[0.18161, 0.26051, 0.30184, 0.30499, 0.33918, 0.35273, 0.35005, 0.32733, 0.33203, 0.40475, 0.53131, 0.59861, 0.69119, 0.79476, 0.90829, 1.01508, 1.13103, 1.23690, 1.33638, 1.42436, 1.48599, 1.53183, 1.56593, 1.58723, 1.59910, 1.59093, 1.55460, 1.50072, 1.44153, 1.38681, 1.32569, 1.24001, 1.14580, 1.07503, 1.00511, 0.92584, 0.83794, 0.76177, 0.67205, 0.45416], [0.14772, 0.35880, 0.42358, 0.42503, 0.45874, 0.46086, 0.43777, 0.37746, 0.35468, 0.42858, 0.57410, 0.63949, 0.73496, 0.84303, 0.96207, 1.07700, 1.20048, 1.31019, 1.41094, 1.49339, 1.54838, 1.58772, 1.61260, 1.62272, 1.61705, 1.58547, 1.52298, 1.44790, 1.37412, 1.30518, 1.22994, 1.14276, 1.03632, 0.93243, 0.83902, 0.75897, 0.68121, 0.60011, 0.47446, 0.17144]]),
    "global": ([60.0], [[0.23788, 0.27881, 0.30197, 0.32419, 0.36324, 0.39575, 0.42603, 0.45726, 0.49925, 0.55789, 0.63414, 0.68856, 0.74907, 0.80983, 0.87353, 0.93101, 0.99339, 1.05110, 1.10905, 1.16494, 1.21097, 1.25329, 1.29156, 1.32339, 1.35453, 1.38107, 1.40132, 1.41594, 1.42617, 1.43546, 1.43810, 1.42734, 1.40901, 1.38901, 1.36178, 1.32729, 1.28406, 1.22870, 1.05587, 0.75300]]),
}
_CG_PLANE = {
    "shadow": ([0.05325, -0.01585, -0.00674], [-0.02349, 0.01052, -0.03730]),
    "highlight": ([0.14096, -0.04221, -0.01707], [-0.06092, 0.03233, -0.09452]),
    "mid": ([0.08183, -0.05254, -0.04712], [0.03296, 0.07469, -0.02179]),
    "global": ([0.13134, -0.07860, -0.05853], [0.03510, 0.11093, -0.04149]),
}
_CG_HUE_CORR = {
    "shadow": ([30, 120, 215, 300], [[0.00060, -0.00011, 0.00110], [0.00383, -0.00019, 0.00137], [0.00060, -0.00011, 0.00110], [0.00378, -0.00018, 0.00127]]),
    "highlight": ([30, 120, 215, 300], [[-0.00205, -0.00283, 0.00477], [0.00175, -0.00305, 0.00488], [-0.00206, -0.00284, 0.00479], [0.00193, -0.00280, 0.00447]]),
    "mid": ([45, 215], [[-0.00000, 0.00000, 0.00000], [0.00000, 0.00000, -0.00000]]),
    "global": ([45, 215], [[-0.00000, 0.00000, 0.00000], [0.00000, 0.00000, 0.00000]]),
}
_CG_SAT = {
    "shadow": ([0.0, 30.0, 60.0, 100.0], [0.00000, 0.51150, 1.00000, 1.64532]),
    "highlight": ([0.0, 30.0, 60.0, 100.0], [0.00000, 0.51917, 1.00000, 1.61074]),
    "mid": ([0.0, 30.0, 60.0, 100.0], [0.00000, 0.51533, 1.00000, 1.69212]),
    "global": ([0.0, 30.0, 60.0, 100.0], [0.00000, 0.51533, 1.00000, 1.64939]),
}
_CG_SAT_CORR = {
    "shadow": ([0.0, 30.0, 60.0, 100.0], [[0.00000, 0.00000, 0.00000], [-0.00047, -0.00008, -0.00048], [0.00000, 0.00000, 0.00000], [0.00043, -0.00122, 0.00075]]),
    "highlight": ([0.0, 30.0, 60.0, 100.0], [[0.00000, 0.00000, 0.00000], [-0.00099, -0.00089, -0.00062], [0.00000, 0.00000, 0.00000], [0.02101, 0.01018, 0.01429]]),
    "mid": ([0.0, 60.0, 100.0], [[0.00000, 0.00000, 0.00000], [0.00000, 0.00000, 0.00000], [0.00282, -0.00037, 0.00458]]),
    "global": ([0.0, 60.0], [[0.00000, 0.00000, 0.00000], [0.00000, 0.00000, 0.00000]]),
}
_CG_LUM = {
    "shadow+": [0.01340, 0.02009, 0.02057, 0.01860, 0.01603, 0.01349, 0.01174, 0.01092, 0.01057, 0.01042, 0.01026, 0.01011, 0.00990, 0.00956, 0.00922, 0.00883, 0.00839, 0.00792, 0.00743, 0.00689, 0.00637, 0.00585, 0.00532, 0.00475, 0.00417, 0.00361, 0.00315, 0.00264, 0.00220, 0.00176, 0.00142, 0.00106, 0.00062, 0.00029, 0.00001, -0.00013, -0.00014, -0.00014, -0.00013, -0.00014],
    "shadow-": [-0.00588, -0.01592, -0.01967, -0.01999, -0.01875, -0.01727, -0.01603, -0.01531, -0.01492, -0.01488, -0.01479, -0.01462, -0.01440, -0.01405, -0.01361, -0.01303, -0.01238, -0.01171, -0.01099, -0.01028, -0.00965, -0.00899, -0.00825, -0.00751, -0.00674, -0.00605, -0.00538, -0.00472, -0.00405, -0.00338, -0.00290, -0.00235, -0.00193, -0.00160, -0.00142, -0.00103, -0.00051, -0.00027, -0.00004, -0.00008],
    "mid+": [-0.00025, 0.00292, 0.00906, 0.01703, 0.02586, 0.03429, 0.04188, 0.04815, 0.05309, 0.05656, 0.05873, 0.05952, 0.05918, 0.05798, 0.05618, 0.05360, 0.05039, 0.04693, 0.04311, 0.03911, 0.03550, 0.03182, 0.02818, 0.02454, 0.02110, 0.01813, 0.01533, 0.01251, 0.00991, 0.00752, 0.00564, 0.00388, 0.00237, 0.00142, 0.00052, 0.00001, -0.00005, -0.00016, -0.00022, -0.00026],
    "mid-": [0.00031, -0.00237, -0.00737, -0.01402, -0.02121, -0.02841, -0.03494, -0.04044, -0.04505, -0.04874, -0.05158, -0.05344, -0.05436, -0.05464, -0.05420, -0.05291, -0.05094, -0.04860, -0.04570, -0.04247, -0.03940, -0.03618, -0.03280, -0.02923, -0.02569, -0.02266, -0.01977, -0.01681, -0.01396, -0.01123, -0.00902, -0.00665, -0.00468, -0.00341, -0.00252, -0.00181, -0.00118, -0.00072, -0.00024, -0.00017],
    "highlight+": [-0.00009, 0.00064, 0.00187, 0.00353, 0.00551, 0.00770, 0.00999, 0.01235, 0.01489, 0.01779, 0.02168, 0.02626, 0.03146, 0.03762, 0.04418, 0.05177, 0.06012, 0.06809, 0.07632, 0.08426, 0.09104, 0.09734, 0.10317, 0.10824, 0.11243, 0.11538, 0.11735, 0.11803, 0.11723, 0.11518, 0.11239, 0.10743, 0.09924, 0.08918, 0.07809, 0.06532, 0.05149, 0.03743, 0.02001, 0.00382],
    "highlight-": [0.00009, -0.00053, -0.00139, -0.00258, -0.00387, -0.00538, -0.00691, -0.00837, -0.00989, -0.01175, -0.01451, -0.01794, -0.02209, -0.02707, -0.03230, -0.03830, -0.04499, -0.05159, -0.05858, -0.06591, -0.07261, -0.07944, -0.08628, -0.09312, -0.09980, -0.10560, -0.11133, -0.11699, -0.12225, -0.12731, -0.13140, -0.13570, -0.13952, -0.14209, -0.14442, -0.14595, -0.14650, -0.14731, -0.14739, -0.14669],
    "global+": [0.00356, 0.01190, 0.01954, 0.02638, 0.03281, 0.03854, 0.04375, 0.04849, 0.05270, 0.05646, 0.06007, 0.06313, 0.06566, 0.06792, 0.06976, 0.07139, 0.07259, 0.07325, 0.07351, 0.07331, 0.07278, 0.07193, 0.07076, 0.06924, 0.06750, 0.06577, 0.06375, 0.06118, 0.05814, 0.05485, 0.05179, 0.04791, 0.04285, 0.03779, 0.03278, 0.02734, 0.02107, 0.01602, 0.01044, 0.00261],
    "global-": [-0.00258, -0.00914, -0.01520, -0.02090, -0.02622, -0.03141, -0.03626, -0.04074, -0.04492, -0.04879, -0.05264, -0.05608, -0.05916, -0.06214, -0.06468, -0.06701, -0.06910, -0.07076, -0.07208, -0.07304, -0.07358, -0.07383, -0.07375, -0.07326, -0.07240, -0.07147, -0.07025, -0.06849, -0.06613, -0.06349, -0.06096, -0.05726, -0.05231, -0.04700, -0.04198, -0.03590, -0.02748, -0.02027, -0.01339, -0.00390],
}
_CG_BAL_W = {
    "shadow": ([-70.0, 0.0, 70.0], [[0.85243, 1.10025, 1.26959, 1.52365, 1.40788, 1.29640, 1.36651, 1.47200, 1.41140, 1.54995, 1.91659, 3.09125, 3.35217, 3.38862, 3.43910, 3.35813, 3.30927, 3.38043, 3.52789, 3.68958, 3.82642, 3.92138, 3.97748, 4.00437, 3.99006, 3.92480, 3.82585, 3.72682, 3.66195, 3.60962, 3.55279, 3.45015, 3.26636, 3.09576, 2.94720, 2.77577, 2.57150, 2.31147, 2.06163, 1.86583, 1.49616, 1.19378, 0.91533, 0.53126, 0.31512, 0.34423, 0.45385, 0.00577], [0.07442, 0.12530, 0.23735, 0.36128, 0.45552, 0.56268, 0.64827, 0.75824, 0.81684, 0.86423, 0.90372, 0.94033, 0.96702, 0.98517, 0.99553, 1.00000, 0.99637, 0.98588, 0.97194, 0.95269, 0.93432, 0.91982, 0.91056, 0.90792, 0.90804, 0.90535, 0.89993, 0.89422, 0.88997, 0.88593, 0.87890, 0.85872, 0.82460, 0.78711, 0.74688, 0.70726, 0.66986, 0.63217, 0.58956, 0.52011, 0.43210, 0.35987, 0.29259, 0.20410, 0.08759, 0.02624, 0.00482, 0.00025], [1.12155, 1.14857, 0.00000, 0.00000, 0.00000, 0.00000, 0.39453, 2.33414, 3.30628, 3.73565, 3.87499, 3.77398, 3.25819, 2.43133, 1.83119, 1.80844, 1.64371, 1.43086, 1.10369, 0.79066, 0.45447, 0.15233, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.03502, 0.17726, 0.31907, 0.00000]]),
    "highlight": ([-70.0, 0.0, 70.0], [[0.01341, 0.15522, 0.17889, 0.19193, 0.07030, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.62147, 0.75391, 0.71413, 0.68118, 0.56426, 0.47299, 0.45599, 0.48177, 0.52598, 0.57540, 0.60800, 0.61833, 0.61781, 0.59776, 0.55534, 0.50630, 0.47732, 0.49853, 0.54069, 0.58752, 0.62355, 0.62951, 0.63798, 0.64853, 0.64069, 0.61534, 0.57008, 0.53147, 0.52323, 0.48512, 0.45431, 0.42115, 0.36548, 0.33639, 0.33811, 0.34195, 0.08526], [0.00038, 0.00069, 0.00231, 0.00429, 0.00954, 0.02239, 0.03607, 0.08352, 0.11693, 0.15020, 0.17265, 0.20310, 0.22765, 0.26235, 0.29365, 0.29403, 0.30074, 0.31324, 0.32663, 0.34316, 0.35842, 0.37637, 0.39558, 0.41188, 0.43097, 0.45550, 0.48645, 0.51946, 0.55692, 0.59782, 0.64292, 0.69786, 0.75778, 0.81347, 0.86427, 0.90702, 0.94036, 0.96612, 0.98294, 0.99316, 1.00000, 0.99574, 0.97841, 0.94577, 0.88102, 0.77334, 0.59448, 0.21002], [0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.10018, 1.77516, 2.13151, 2.33573, 2.41459, 2.24062, 1.94698, 1.69853, 1.68992, 1.62710, 1.55529, 1.42467, 1.31403, 1.19019, 1.09017, 0.98343, 0.88224, 0.83896, 0.86697, 0.93463, 1.02563, 1.13403, 1.25167, 1.36447, 1.45562, 1.53579, 1.60546, 1.64870, 1.68252, 1.71842, 1.74666, 1.74348, 1.75625, 1.74643, 1.72315, 1.68967, 1.63907, 1.54767, 1.62105, 1.76812, 0.37631]]),
}
_CG_BLEND_W = {
    "shadow": ([0.0, 50.0, 100.0], [[26.63581, 7.03744, 5.04297, 5.88806, 6.84281, 7.71325, 9.34950, 11.93146, 12.59960, 13.44890, 14.16989, 14.73023, 14.97170, 14.83507, 14.38996, 13.77042, 12.55762, 10.68435, 8.53781, 6.52365, 4.94802, 3.61498, 2.82656, 2.45109, 2.11211, 1.83080, 1.64575, 1.49655, 1.38621, 1.30617, 1.16870, 1.02498, 0.91684, 0.81352, 0.68389, 0.53522, 0.37642, 0.21394, 0.03808, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.04932, 0.20494, 0.38801, 0.00000], [0.07442, 0.12530, 0.23735, 0.36128, 0.45552, 0.56268, 0.64827, 0.75824, 0.81684, 0.86423, 0.90372, 0.94033, 0.96702, 0.98517, 0.99553, 1.00000, 0.99637, 0.98588, 0.97194, 0.95269, 0.93432, 0.91982, 0.91056, 0.90792, 0.90804, 0.90535, 0.89993, 0.89422, 0.88997, 0.88593, 0.87890, 0.85872, 0.82460, 0.78711, 0.74688, 0.70726, 0.66986, 0.63217, 0.58956, 0.52011, 0.43210, 0.35987, 0.29259, 0.20410, 0.08759, 0.02624, 0.00482, 0.00025], [3.78436, 4.63482, 4.63621, 3.59787, 3.15734, 3.49349, 4.11853, 4.90399, 5.38024, 5.87369, 6.31652, 6.73832, 7.14297, 7.44469, 7.62795, 7.81521, 7.88555, 7.92006, 7.89022, 7.77709, 7.57631, 7.27620, 6.82590, 6.29594, 5.76556, 5.25388, 4.78244, 4.28714, 3.76985, 3.30135, 2.85564, 2.41240, 1.98090, 1.60758, 1.26637, 0.96577, 0.72719, 0.54333, 0.37940, 0.19608, 0.06930, 0.01838, 0.00000, 0.00394, 0.36425, 0.55329, 0.82885, 0.00000]]),
    "highlight": ([0.0, 50.0, 100.0], [[14.32270, 1.77191, 0.00000, 0.00000, 0.00000, 0.00000, 0.98367, 4.00166, 4.73797, 5.68738, 6.30127, 6.63569, 6.80858, 6.78320, 6.60946, 6.46016, 5.71975, 4.57197, 3.48090, 2.64378, 2.07816, 1.56931, 1.27998, 1.15755, 1.03898, 0.94631, 0.89549, 0.85970, 0.84323, 0.84175, 0.81760, 0.80160, 0.80681, 0.81520, 0.80882, 0.78505, 0.74747, 0.70039, 0.63745, 0.60613, 0.58054, 0.56631, 0.54916, 0.52830, 0.54987, 0.55860, 0.61218, 0.13503], [0.00038, 0.00069, 0.00231, 0.00429, 0.00954, 0.02239, 0.03607, 0.08352, 0.11693, 0.15020, 0.17265, 0.20310, 0.22765, 0.26235, 0.29365, 0.29403, 0.30074, 0.31324, 0.32663, 0.34316, 0.35842, 0.37637, 0.39558, 0.41188, 0.43097, 0.45550, 0.48645, 0.51946, 0.55692, 0.59782, 0.64292, 0.69786, 0.75778, 0.81347, 0.86427, 0.90702, 0.94036, 0.96612, 0.98294, 0.99316, 1.00000, 0.99574, 0.97841, 0.94577, 0.88102, 0.77334, 0.59448, 0.21002], [2.04766, 2.41065, 2.28837, 0.50485, 0.00000, 0.00000, 0.19589, 0.65695, 1.12396, 1.70402, 2.28806, 2.67529, 2.99602, 3.18029, 3.22291, 3.29832, 3.30602, 3.33432, 3.34248, 3.32585, 3.26221, 3.16503, 3.00334, 2.81768, 2.64246, 2.48543, 2.34710, 2.19259, 2.02298, 1.87439, 1.74007, 1.61563, 1.50222, 1.40744, 1.31994, 1.24108, 1.18068, 1.13872, 1.09789, 1.04413, 1.00658, 0.98632, 0.96144, 0.92763, 0.97507, 0.94629, 0.97949, 0.27419]]),
}
_CG_BAL_CAP = {
    "shadow": ([-70.0, 0.0, 70.0], [[0.00000, 0.04897, 0.13403, 0.25066, 0.32814, 0.40528, 0.49476, 0.58488, 0.65814, 0.69790, 0.75914, 0.87280, 1.08291, 1.25741, 1.31571, 1.25394, 1.05852, 1.00412, 1.01495, 0.99676, 0.98356, 0.99466, 0.98454, 0.98847, 1.01257, 1.04565, 1.10277, 1.16715, 1.21530, 1.21946, 1.20146, 1.20776, 1.25954, 1.43441, 1.66885, 1.80716, 1.89720, 1.92436, 1.69890, 1.35044], [0.47759, 0.51713, 0.58477, 0.67265, 0.77726, 0.87317, 0.95694, 1.04848, 1.15547, 1.24927, 1.31875, 1.36475, 1.38137, 1.36815, 1.33369, 1.28519, 1.23456, 1.18624, 1.14473, 1.12159, 1.11009, 1.10781, 1.10440, 1.10120, 1.09159, 1.07095, 1.03361, 0.99305, 0.95649, 0.92338, 0.88826, 0.83662, 0.77353, 0.72262, 0.68252, 0.64725, 0.60078, 0.58429, 0.61497, 0.51518], [0.03037, 0.05521, 0.00000, 0.08239, 0.28449, 0.27514, 0.18720, 0.14230, 0.12721, 0.15051, 0.22638, 0.32518, 0.42488, 0.58187, 0.76359, 0.95954, 1.15733, 1.33023, 1.54077, 1.66251, 1.69275, 1.74561, 1.72276, 1.57730, 1.39342, 1.30574, 1.32705, 1.36031, 1.22916, 1.07507, 0.93779, 0.79413, 0.67555, 0.59865, 0.58731, 0.69914, 1.38819, 5.05665, 3.58690, 1.19452]]),
    "highlight": ([-70.0, 0.0, 70.0], [[0.00000, 0.00000, 0.00000, 0.29999, 0.41163, 0.44802, 0.44431, 0.47840, 0.46505, 0.40413, 0.40545, 0.53422, 0.93766, 1.27187, 1.30858, 1.02658, 0.32712, 0.08720, 0.08265, 0.01710, 0.00080, 0.11004, 0.17944, 0.29422, 0.48945, 0.73305, 1.08773, 1.45850, 1.72650, 1.82938, 1.86494, 1.96460, 2.17487, 2.61366, 3.14476, 3.50111, 3.77618, 3.79344, 3.33697, 2.56457], [0.04917, 0.08661, 0.11964, 0.14549, 0.16443, 0.15870, 0.14596, 0.13426, 0.13822, 0.16726, 0.21857, 0.25608, 0.30309, 0.35745, 0.42047, 0.48293, 0.55501, 0.62686, 0.70211, 0.78220, 0.86634, 0.97249, 1.09047, 1.20428, 1.33840, 1.49172, 1.67582, 1.85374, 2.00258, 2.12349, 2.22159, 2.31397, 2.38403, 2.40115, 2.32898, 2.21925, 2.06188, 1.80387, 1.40579, 1.01905], [0.02988, 0.08133, 0.03266, 0.09446, 0.23080, 0.19799, 0.12485, 0.09231, 0.08527, 0.11162, 0.18463, 0.26676, 0.36029, 0.48832, 0.63280, 0.77458, 0.91650, 1.04266, 1.17924, 1.29790, 1.39392, 1.50446, 1.58631, 1.62135, 1.63929, 1.66511, 1.70779, 1.73814, 1.74722, 1.73693, 1.72445, 1.69904, 1.67219, 1.64775, 1.58988, 1.50624, 1.40444, 1.26618, 0.86821, 0.47820]]),
}
_CG_BLEND_CAP = {
    "shadow": ([0.0, 50.0, 100.0], [[0.00924, 0.02109, 0.03346, 0.04505, 0.05480, 0.06332, 0.07220, 0.08038, 0.08387, 0.08222, 0.07809, 0.07991, 0.07978, 0.08030, 0.09098, 0.10134, 0.12248, 0.13894, 0.14884, 0.16047, 0.16219, 0.22061, 0.37613, 0.55574, 0.78426, 1.06876, 1.44136, 1.85265, 2.22172, 2.51022, 2.74411, 3.03128, 3.35324, 3.75824, 4.17209, 4.13872, 3.42703, 3.55825, 2.97413, 1.70838], [0.47759, 0.51713, 0.58477, 0.67265, 0.77726, 0.87317, 0.95694, 1.04848, 1.15547, 1.24927, 1.31875, 1.36475, 1.38137, 1.36815, 1.33369, 1.28519, 1.23456, 1.18624, 1.14473, 1.12159, 1.11009, 1.10781, 1.10440, 1.10120, 1.09159, 1.07095, 1.03361, 0.99305, 0.95649, 0.92338, 0.88826, 0.83662, 0.77353, 0.72262, 0.68252, 0.64725, 0.60078, 0.58429, 0.61497, 0.51518], [0.00000, 0.02766, 0.06381, 0.09933, 0.12529, 0.14583, 0.16459, 0.18850, 0.21642, 0.24555, 0.25700, 0.28506, 0.31210, 0.33284, 0.35144, 0.38064, 0.42726, 0.49053, 0.57334, 0.67718, 0.79682, 0.94445, 1.11211, 1.28092, 1.46233, 1.64572, 1.84183, 2.02532, 2.17124, 2.27262, 2.34192, 2.39576, 2.40165, 2.37892, 2.30448, 2.15177, 1.94628, 1.99042, 1.44781, 0.59453]]),
    "highlight": ([0.0, 50.0, 100.0], [[0.00626, 0.01320, 0.01918, 0.02239, 0.02086, 0.01659, 0.01488, 0.01506, 0.01245, 0.00677, 0.00022, 0.00029, 0.00037, 0.00449, 0.02211, 0.03945, 0.07081, 0.09625, 0.11496, 0.13281, 0.13666, 0.20814, 0.39715, 0.62558, 0.91784, 1.26926, 1.71206, 2.15599, 2.51642, 2.77722, 2.99275, 3.22752, 3.45737, 3.63808, 3.72683, 3.67513, 3.55364, 3.33609, 2.47447, 1.52598], [0.04917, 0.08661, 0.11964, 0.14549, 0.16443, 0.15870, 0.14596, 0.13426, 0.13822, 0.16726, 0.21857, 0.25608, 0.30309, 0.35745, 0.42047, 0.48293, 0.55501, 0.62686, 0.70211, 0.78220, 0.86634, 0.97249, 1.09047, 1.20428, 1.33840, 1.49172, 1.67582, 1.85374, 2.00258, 2.12349, 2.22159, 2.31397, 2.38403, 2.40115, 2.32898, 2.21925, 2.06188, 1.80387, 1.40579, 1.01905], [0.00000, 0.00208, 0.02542, 0.05553, 0.06250, 0.05231, 0.03592, 0.03332, 0.03816, 0.05026, 0.05211, 0.07085, 0.09254, 0.11115, 0.13197, 0.17043, 0.23400, 0.32093, 0.43560, 0.57726, 0.73541, 0.92769, 1.13675, 1.33855, 1.55367, 1.76959, 1.99538, 2.20040, 2.35357, 2.45788, 2.53272, 2.59079, 2.60898, 2.58599, 2.52427, 2.44773, 2.34464, 2.20823, 1.64091, 0.93639]]),
}
_CG_MID_SATD = [
    [([-0.00000, 0.00022, 0.00126, 0.00464, 0.01220, 0.02348, 0.04335, 0.07339, 0.10255, 0.14419, 0.20181, 0.24377, 0.26742, 0.28104, 0.28650, 0.28531, 0.27181, 0.25081, 0.22662, 0.19851, 0.16536, 0.13045, 0.10081, 0.07722, 0.05897, 0.04515, 0.03457, 0.02622, 0.01736, 0.00680, -0.00426, -0.01437, -0.02190, -0.02678, -0.03057, -0.03268, -0.03183, -0.02928, -0.02496, -0.01794, -0.01082, -0.00552, -0.00216, -0.00052, 0.00016, 0.00025, 0.00014, 0.00004], [0.00326, 0.00246, 0.00132, -0.00013, -0.00034, 0.00311, 0.01377, 0.03115, 0.04644, 0.04898, 0.04233, 0.04063, 0.04201, 0.03880, 0.03408, 0.03219, 0.03499, 0.04277, 0.05855, 0.08512, 0.12022, 0.16288, 0.20908, 0.25072, 0.28426, 0.31778, 0.35361, 0.36792, 0.35109, 0.31831, 0.27716, 0.21996, 0.14689, 0.08574, 0.04865, 0.02485, 0.01159, 0.00522, 0.00177, 0.00063]), ([0.00000, 0.00045, 0.00232, 0.00053, -0.01736, -0.05073, -0.09528, -0.14392, -0.17352, -0.18387, -0.17951, -0.15042, -0.10250, -0.04366, 0.01694, 0.06103, 0.08817, 0.10782, 0.11920, 0.12342, 0.11653, 0.09927, 0.08011, 0.05820, 0.03486, 0.01475, -0.00100, -0.01102, -0.01254, -0.00693, 0.00147, 0.00959, 0.01331, 0.01045, 0.00386, -0.00401, -0.01194, -0.01819, -0.02077, -0.01865, -0.01347, -0.00781, -0.00353, -0.00094, 0.00030, 0.00044, 0.00021, 0.00005], [-0.01362, -0.00583, 0.00664, 0.02410, 0.04369, 0.05755, 0.06504, 0.07067, 0.07832, 0.08487, 0.08455, 0.08387, 0.07930, 0.05587, 0.01677, -0.03173, -0.09035, -0.15255, -0.21424, -0.26973, -0.30260, -0.32052, -0.32604, -0.28766, -0.20307, -0.11863, -0.04337, 0.06165, 0.17454, 0.25261, 0.29156, 0.28489, 0.23148, 0.16073, 0.09813, 0.04778, 0.01652, 0.00185, -0.00273, -0.00171]), ([0.00000, -0.00030, -0.00372, -0.02484, -0.06749, -0.10551, -0.11944, -0.10995, -0.08351, -0.04193, 0.00789, 0.04688, 0.07024, 0.08121, 0.07398, 0.05447, 0.03171, 0.00601, -0.02084, -0.05152, -0.07639, -0.09003, -0.10071, -0.10345, -0.09590, -0.08169, -0.06662, -0.05327, -0.03529, -0.01626, -0.00454, -0.00114, -0.00421, -0.01007, -0.01390, -0.01311, -0.00805, -0.00066, 0.00614, 0.01007, 0.01013, 0.00786, 0.00498, 0.00173, -0.00074, -0.00135, -0.00084, -0.00026], [-0.01085, 0.00157, 0.01633, 0.02383, 0.02244, 0.01439, 0.00308, -0.01138, -0.03345, -0.05632, -0.07990, -0.12729, -0.19338, -0.24390, -0.27135, -0.29505, -0.31593, -0.30877, -0.27798, -0.23515, -0.15981, -0.08421, -0.02836, 0.05483, 0.14407, 0.18695, 0.18703, 0.15598, 0.09342, 0.01333, -0.07020, -0.15886, -0.23326, -0.25461, -0.21496, -0.14745, -0.08483, -0.03330, -0.00360, 0.00163])],
    [([0.00000, 0.00001, 0.00001, 0.00004, 0.00015, 0.00045, 0.00121, 0.00289, 0.00576, 0.01012, 0.01600, 0.02214, 0.02797, 0.03181, 0.02975, 0.02299, 0.01587, 0.00977, 0.00498, 0.00163, -0.00032, -0.00079, -0.00021, 0.00058, 0.00105, 0.00111, 0.00088, 0.00067, 0.00053, 0.00029, 0.00013, 0.00009, 0.00006, 0.00002, 0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.00158, -0.01144, -0.04090, -0.09711, -0.22096, -0.39249, -0.48492, -0.46280, -0.39066, -0.31258, -0.23164, -0.14368, -0.06583, -0.00630, 0.03431, 0.05033, 0.04768, 0.03483, 0.01720, 0.00619, 0.00381, 0.00372, 0.00329, 0.00218, 0.00094, 0.00007, -0.00019, -0.00013, -0.00007, -0.00005, -0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, -0.00000, -0.00000]), ([-0.00000, -0.00001, -0.00001, -0.00004, -0.00015, -0.00041, -0.00104, -0.00220, -0.00388, -0.00592, -0.00763, -0.00793, -0.00658, -0.00333, 0.00225, 0.00806, 0.01185, 0.01258, 0.01046, 0.00747, 0.00445, 0.00143, -0.00032, 0.00075, 0.00416, 0.00849, 0.01242, 0.01654, 0.01840, 0.01476, 0.01049, 0.00789, 0.00513, 0.00254, 0.00071, -0.00012, -0.00030, -0.00023, -0.00012, -0.00004, -0.00001, 0.00000, 0.00000, 0.00000, -0.00000, 0.00000, 0.00000, 0.00000], [0.00270, 0.01614, 0.05113, 0.10440, 0.16995, 0.20494, 0.15381, 0.03796, -0.08416, -0.18112, -0.23448, -0.23426, -0.19165, -0.12290, -0.03767, 0.05752, 0.15908, 0.28922, 0.38122, 0.35922, 0.31916, 0.29729, 0.25249, 0.18627, 0.10227, 0.02607, -0.01235, -0.01736, -0.01440, -0.00985, -0.00307, 0.00027, 0.00058, 0.00043, 0.00020, 0.00007, 0.00001, 0.00000, -0.00000, -0.00000]), ([-0.00000, -0.00001, -0.00001, -0.00006, -0.00024, -0.00068, -0.00170, -0.00352, -0.00604, -0.00893, -0.01105, -0.01071, -0.00785, -0.00252, 0.00551, 0.01267, 0.01522, 0.01330, 0.00869, 0.00275, -0.00263, -0.00449, -0.00282, -0.00075, -0.00037, -0.00221, -0.00579, -0.01030, -0.01307, -0.01193, -0.00953, -0.00745, -0.00506, -0.00266, -0.00086, 0.00003, 0.00028, 0.00024, 0.00014, 0.00005, 0.00001, -0.00000, -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000], [0.00502, 0.02887, 0.08928, 0.17697, 0.27056, 0.29830, 0.19220, 0.00420, -0.17180, -0.28994, -0.33499, -0.30901, -0.23014, -0.11964, -0.00719, 0.06775, 0.08006, -0.02968, -0.19979, -0.26912, -0.27429, -0.26842, -0.23414, -0.18016, -0.10673, -0.03471, 0.00675, 0.01640, 0.01580, 0.01146, 0.00379, -0.00022, -0.00068, -0.00054, -0.00027, -0.00009, -0.00002, -0.00000, 0.00000, 0.00000])],
    [([0.00004, 0.00006, 0.00007, 0.00002, -0.00016, -0.00056, -0.00132, -0.00250, -0.00388, -0.00540, -0.00712, -0.00908, -0.01130, -0.01365, -0.01733, -0.02302, -0.02770, -0.02904, -0.02819, -0.02648, -0.02343, -0.01802, -0.01030, -0.00282, 0.00074, 0.00005, -0.00260, -0.00473, -0.00558, -0.00706, -0.01138, -0.01965, -0.03159, -0.04600, -0.06230, -0.08048, -0.09969, -0.11814, -0.13406, -0.14677, -0.16573, -0.16950, -0.11951, -0.05374, -0.01939, -0.00656, -0.00166, -0.00032], [-0.04996, -0.08057, -0.09981, -0.09747, -0.08762, -0.06845, -0.03519, 0.00652, 0.03672, 0.03639, 0.01855, 0.00716, 0.00882, 0.02234, 0.03704, 0.04965, 0.06593, 0.07707, 0.09315, 0.15423, 0.25417, 0.31134, 0.30248, 0.28884, 0.28660, 0.28232, 0.28575, 0.29128, 0.27944, 0.25204, 0.21528, 0.17227, 0.12826, 0.08336, 0.04066, 0.01298, 0.00325, 0.00068, 0.00054, 0.00125]), ([0.00014, 0.00023, 0.00030, 0.00023, -0.00012, -0.00038, -0.00027, -0.00069, -0.00194, -0.00306, -0.00456, -0.00753, -0.01243, -0.01983, -0.02640, -0.02929, -0.03245, -0.03720, -0.04189, -0.04765, -0.05215, -0.05178, -0.04748, -0.04273, -0.04276, -0.05004, -0.05974, -0.06453, -0.06301, -0.05893, -0.05373, -0.04625, -0.03585, -0.02288, -0.00839, 0.00458, 0.01334, 0.01702, 0.01666, 0.01412, 0.01209, 0.01328, 0.01247, 0.00780, 0.00433, 0.00218, 0.00070, 0.00015], [-0.14996, -0.24493, -0.31735, -0.33937, -0.36413, -0.35269, -0.26569, -0.14526, -0.06447, -0.07499, -0.12570, -0.14478, -0.15193, -0.16973, -0.18711, -0.19999, -0.20378, -0.19128, -0.16862, -0.15149, -0.13823, -0.10795, -0.06038, -0.01722, 0.01826, 0.04852, 0.06057, 0.04976, 0.02449, -0.00718, -0.03061, -0.03979, -0.04159, -0.03474, -0.01847, -0.00587, -0.00208, -0.00070, -0.00029, -0.00069]), ([0.00011, 0.00016, 0.00015, -0.00002, -0.00055, -0.00198, -0.00549, -0.01090, -0.01640, -0.02165, -0.02650, -0.03016, -0.03125, -0.02801, -0.02272, -0.01809, -0.01303, -0.00678, -0.00011, 0.00525, 0.01108, 0.01998, 0.02864, 0.03335, 0.03278, 0.02514, 0.01261, 0.00229, -0.00388, -0.00917, -0.01486, -0.02090, -0.02745, -0.03502, -0.04328, -0.04886, -0.04896, -0.04205, -0.01953, 0.01409, 0.03705, 0.04612, 0.04565, 0.03523, 0.02124, 0.01009, 0.00373, 0.00088], [-0.15488, -0.24550, -0.28838, -0.25923, -0.19729, -0.12204, -0.04577, 0.03628, 0.10878, 0.13737, 0.12820, 0.12197, 0.14576, 0.19756, 0.24413, 0.26753, 0.28114, 0.28516, 0.27164, 0.20390, 0.06699, -0.03481, -0.03755, -0.01595, -0.01638, -0.02269, -0.05306, -0.10691, -0.14244, -0.15694, -0.16400, -0.16416, -0.15524, -0.13767, -0.10462, -0.05793, -0.02023, -0.00380, -0.00210, -0.00503])],
]
_CG_BAL_R = {
    -70: [
    [([0.00001, 0.00012, 0.00046, 0.00028, -0.00253, -0.00743, -0.01043, -0.01038, -0.00735, 0.00234, 0.01928, 0.03740, 0.05124, 0.05929, 0.06100, 0.05554, 0.04713, 0.04036, 0.03416, 0.02893, 0.02605, 0.02537, 0.02834, 0.03449, 0.04283, 0.05283, 0.06276, 0.07083, 0.07279, 0.06718, 0.05425, 0.03492, 0.01304, -0.00735, -0.02206, -0.02875, -0.02786, -0.02333, -0.01807, -0.01387, -0.01145, -0.00747, -0.00321, -0.00122, -0.00094, -0.00171, -0.00066, 0.00215], [0.00011, 0.00218, 0.00593, 0.00865, 0.01334, 0.02316, 0.03484, 0.04078, 0.03531, 0.01965, 0.00446, 0.00247, 0.01516, 0.03576, 0.05750, 0.07917, 0.10105, 0.11900, 0.12993, 0.12806, 0.10231, 0.04623, -0.03039, -0.10957, -0.18145, -0.24404, -0.29682, -0.32566, -0.31301, -0.26498, -0.19927, -0.12254, -0.02502, 0.09740, 0.21527, 0.29429, 0.33047, 0.30139, 0.19561, 0.09717]), ([-0.00002, -0.00012, -0.00056, -0.00165, -0.00496, -0.01150, -0.02259, -0.03863, -0.05365, -0.06282, -0.05700, -0.03832, -0.02910, -0.03042, -0.03142, -0.03173, -0.03132, -0.02917, -0.02507, -0.02118, -0.01897, -0.01736, -0.01431, -0.01006, -0.00327, 0.00761, 0.02055, 0.03331, 0.04291, 0.04907, 0.05135, 0.04883, 0.04010, 0.02663, 0.01565, 0.00876, 0.00436, 0.00328, 0.00323, 0.00115, -0.00139, -0.00268, -0.00267, -0.00179, -0.00126, -0.00137, -0.00019, 0.00241], [0.00556, 0.00213, -0.00743, -0.01491, -0.01645, -0.01887, -0.02659, -0.03458, -0.03251, -0.01822, -0.00635, -0.01920, -0.05842, -0.09550, -0.10239, -0.06382, 0.00663, 0.07285, 0.13331, 0.20202, 0.26545, 0.30751, 0.32588, 0.31579, 0.27796, 0.24204, 0.22452, 0.20043, 0.15819, 0.11220, 0.07398, 0.04487, 0.03422, 0.06415, 0.12987, 0.19748, 0.25366, 0.27126, 0.21164, 0.13414]), ([0.00000, 0.00017, 0.00049, -0.00037, -0.00590, -0.01590, -0.02758, -0.04112, -0.05186, -0.05076, -0.02879, 0.00555, 0.02780, 0.03568, 0.03769, 0.03277, 0.02517, 0.02080, 0.01843, 0.01588, 0.01230, 0.00670, -0.00228, -0.01278, -0.02360, -0.03482, -0.04248, -0.04221, -0.03204, -0.01656, -0.00164, 0.00980, 0.01354, 0.01059, 0.00698, 0.00385, 0.00022, -0.00383, -0.00646, -0.00601, -0.00405, -0.00193, -0.00104, -0.00095, 0.00016, 0.00157, 0.00089, -0.00145], [0.00728, 0.00791, 0.00493, 0.00236, 0.01269, 0.03545, 0.05706, 0.06399, 0.05328, 0.02991, 0.00329, -0.01970, -0.03607, -0.03425, -0.00552, 0.05431, 0.13293, 0.19474, 0.23447, 0.26681, 0.28144, 0.27854, 0.25755, 0.19149, 0.08158, -0.02394, -0.10275, -0.16872, -0.20469, -0.19634, -0.16273, -0.12547, -0.11401, -0.15639, -0.23253, -0.29330, -0.31768, -0.26925, -0.13872, -0.02841])],
    [([-0.00003, -0.00009, -0.00037, -0.00094, -0.00217, -0.00433, -0.00690, -0.00931, -0.01217, -0.01470, -0.01464, -0.01273, -0.01174, -0.01148, -0.00954, -0.00648, -0.00394, -0.00205, -0.00092, -0.00042, -0.00017, -0.00010, -0.00017, -0.00017, -0.00010, -0.00003, -0.00000, 0.00001, 0.00001, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.10792, -0.18328, -0.30144, -0.37238, -0.41076, -0.43927, -0.40510, -0.32205, -0.23663, -0.16342, -0.09408, -0.02512, 0.02152, 0.03901, 0.03484, 0.01801, 0.00348, -0.00193, -0.00169, -0.00092, -0.00079, -0.00050, -0.00026, -0.00012, -0.00003, -0.00001, -0.00002, -0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000]), ([-0.00010, -0.00029, -0.00097, -0.00227, -0.00434, -0.00702, -0.00903, -0.00859, -0.00671, -0.00470, -0.00160, 0.00209, 0.00541, 0.00792, 0.00898, 0.00839, 0.00655, 0.00426, 0.00239, 0.00130, 0.00058, 0.00027, 0.00037, 0.00040, 0.00026, 0.00008, -0.00002, -0.00004, -0.00005, -0.00004, -0.00002, -0.00002, -0.00001, -0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000], [-0.30979, -0.37234, -0.40908, -0.33328, -0.14505, 0.09001, 0.25918, 0.32736, 0.33305, 0.29864, 0.23013, 0.12626, 0.01116, -0.06810, -0.08336, -0.04941, -0.00781, 0.01080, 0.00908, 0.00545, 0.00456, 0.00289, 0.00150, 0.00072, 0.00030, 0.00024, 0.00024, 0.00017, 0.00011, 0.00007, 0.00004, 0.00002, 0.00001, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]), ([-0.00012, -0.00033, -0.00100, -0.00217, -0.00368, -0.00514, -0.00529, -0.00305, -0.00039, 0.00143, 0.00381, 0.00534, 0.00439, 0.00168, -0.00239, -0.00616, -0.00769, -0.00686, -0.00494, -0.00314, -0.00148, -0.00046, -0.00036, -0.00045, -0.00030, -0.00004, 0.00011, 0.00016, 0.00016, 0.00012, 0.00007, 0.00005, 0.00005, 0.00004, 0.00004, 0.00003, 0.00003, 0.00002, 0.00002, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.35630, -0.31965, -0.18343, 0.02292, 0.22109, 0.30857, 0.20513, -0.00919, -0.21541, -0.36983, -0.43534, -0.36147, -0.17381, 0.00869, 0.09301, 0.06993, -0.00146, -0.04072, -0.03162, -0.01812, -0.01331, -0.00758, -0.00364, -0.00204, -0.00182, -0.00234, -0.00250, -0.00217, -0.00185, -0.00157, -0.00110, -0.00067, -0.00044, -0.00031, -0.00021, -0.00011, -0.00004, -0.00001, -0.00000, -0.00000])],
    [([-0.00090, -0.00158, -0.00265, -0.00345, -0.00440, -0.00692, -0.01141, -0.01786, -0.02444, -0.02532, -0.02420, -0.02921, -0.03379, -0.03283, -0.02967, -0.02497, -0.01979, -0.01526, -0.01069, -0.00754, -0.00625, -0.00466, -0.00214, -0.00050, -0.00564, -0.02126, -0.04154, -0.06028, -0.07726, -0.09382, -0.11114, -0.12955, -0.14879, -0.16819, -0.18666, -0.20069, -0.20642, -0.20169, -0.14853, -0.05651, -0.01146, -0.00866, -0.00213, 0.00412, 0.00444, 0.00205, 0.00048, -0.00019], [-0.30705, -0.29171, -0.27914, -0.28592, -0.29234, -0.28071, -0.25878, -0.23795, -0.21486, -0.18204, -0.14698, -0.12706, -0.14006, -0.18112, -0.21022, -0.19716, -0.16060, -0.13169, -0.11634, -0.10622, -0.09729, -0.08660, -0.07192, -0.05735, -0.04534, -0.03629, -0.02846, -0.01657, 0.00351, 0.03102, 0.05791, 0.07319, 0.07361, 0.06340, 0.04749, 0.03190, 0.01761, 0.00558, 0.00045, -0.00013]), ([-0.00018, -0.00027, -0.00042, -0.00052, -0.00052, -0.00071, -0.00141, -0.00216, -0.00210, -0.00085, 0.00068, 0.00132, 0.00164, 0.00230, 0.00347, 0.00547, 0.00741, 0.00729, 0.00341, -0.00360, -0.01099, -0.01728, -0.02224, -0.02819, -0.04003, -0.05742, -0.07155, -0.07673, -0.07435, -0.06562, -0.05259, -0.03670, -0.02028, -0.00601, 0.00668, 0.01958, 0.03357, 0.04564, 0.05658, 0.06904, 0.08210, 0.08659, 0.06745, 0.03594, 0.01414, 0.00404, 0.00153, 0.00249], [-0.08631, -0.06080, -0.03429, -0.02700, -0.03912, -0.03767, -0.01096, 0.01672, 0.02150, -0.00246, -0.03123, -0.04402, -0.03666, -0.00900, 0.02193, 0.04921, 0.07771, 0.08973, 0.09715, 0.14317, 0.21911, 0.25606, 0.23487, 0.20753, 0.19938, 0.21194, 0.25207, 0.30324, 0.33510, 0.34289, 0.33238, 0.28659, 0.19809, 0.09806, 0.01600, -0.03569, -0.05676, -0.05265, -0.03056, -0.01018]), ([-0.00040, -0.00070, -0.00119, -0.00156, -0.00177, -0.00012, 0.00700, 0.01841, 0.02757, 0.02804, 0.02414, 0.02485, 0.02490, 0.01926, 0.01114, 0.00257, -0.00448, -0.00859, -0.00941, -0.00821, -0.00655, -0.00504, -0.00362, -0.00265, -0.00581, -0.01487, -0.02392, -0.02693, -0.02352, -0.01596, -0.00656, 0.00337, 0.01320, 0.02247, 0.03067, 0.02807, 0.00502, -0.03226, -0.05012, -0.03160, -0.01516, -0.01184, -0.00257, 0.00549, 0.00451, 0.00089, -0.00111, -0.00263], [-0.13573, -0.13207, -0.12706, -0.12205, -0.13261, -0.13564, -0.11473, -0.08649, -0.01652, 0.11913, 0.24511, 0.32079, 0.38238, 0.40862, 0.36611, 0.25357, 0.10351, 0.01216, -0.02979, -0.09423, -0.15384, -0.15692, -0.12318, -0.10175, -0.08229, -0.05047, -0.01804, 0.01174, 0.05376, 0.11352, 0.15821, 0.13748, 0.04312, -0.03919, -0.01611, 0.07369, 0.10486, 0.05756, 0.01272, -0.00145])],
],
    70: [
    [([-0.00001, 0.00006, 0.00023, 0.00081, 0.00294, 0.01173, 0.03337, 0.05709, 0.06469, 0.05749, 0.04359, 0.02532, 0.00309, -0.01991, -0.03365, -0.03547, -0.03383, -0.03273, -0.02906, -0.02062, -0.00901, 0.01033, 0.05083, 0.10355, 0.14447, 0.16473, 0.16788, 0.16056, 0.14119, 0.11310, 0.08729, 0.06818, 0.05414, 0.04475, 0.04163, 0.04097, 0.03921, 0.03797, 0.03612, 0.03000, 0.02118, 0.01235, 0.00352, -0.00453, -0.01197, -0.01705, -0.01251, -0.00212], [0.00114, -0.00053, -0.00364, -0.00621, -0.00755, -0.00959, -0.01284, -0.01522, -0.01594, -0.01604, -0.01397, -0.00437, 0.01758, 0.05184, 0.09334, 0.14985, 0.21408, 0.25216, 0.26338, 0.26212, 0.24599, 0.21595, 0.18205, 0.15483, 0.13965, 0.13333, 0.12954, 0.12955, 0.13637, 0.14858, 0.16311, 0.17802, 0.19263, 0.21370, 0.24683, 0.28518, 0.29783, 0.23835, 0.11142, 0.00117]), ([-0.00005, 0.00054, 0.00241, 0.00404, 0.00508, 0.00726, 0.01560, 0.02924, 0.03534, 0.03025, 0.02009, 0.00842, -0.00360, -0.01532, -0.02374, -0.02986, -0.03766, -0.04683, -0.05632, -0.06686, -0.07460, -0.07712, -0.07639, -0.06688, -0.04794, -0.02691, -0.00814, 0.00697, 0.01974, 0.02793, 0.02916, 0.02526, 0.01624, 0.00416, -0.00541, -0.01205, -0.01700, -0.01960, -0.01979, -0.01701, -0.01162, -0.00508, 0.00155, 0.00680, 0.01060, 0.01196, 0.00789, 0.00226], [0.00217, -0.00217, -0.01017, -0.01820, -0.02728, -0.04370, -0.06912, -0.09802, -0.13058, -0.15748, -0.17070, -0.20028, -0.24360, -0.25304, -0.22171, -0.16541, -0.08760, 0.00461, 0.09850, 0.18091, 0.23994, 0.27145, 0.27383, 0.24818, 0.21213, 0.17573, 0.13211, 0.08754, 0.04818, 0.00114, -0.05386, -0.10853, -0.16813, -0.22870, -0.25875, -0.23891, -0.16376, -0.04693, 0.02874, 0.03355]), ([0.00010, -0.00032, -0.00166, -0.00217, -0.00000, 0.01028, 0.03360, 0.05583, 0.06111, 0.05720, 0.05377, 0.04872, 0.03905, 0.02648, 0.01996, 0.02228, 0.02657, 0.02857, 0.02695, 0.02305, 0.01791, 0.01247, 0.00683, 0.00114, -0.00278, -0.00485, -0.00123, 0.00628, 0.00866, 0.00420, -0.00496, -0.01636, -0.02730, -0.03521, -0.03768, -0.03539, -0.03078, -0.02729, -0.02443, -0.01888, -0.01263, -0.00758, -0.00288, 0.00160, 0.00614, 0.00932, 0.00716, 0.00176], [0.00789, 0.00193, -0.00669, -0.00033, 0.03066, 0.06255, 0.07891, 0.08564, 0.09009, 0.09518, 0.09951, 0.10290, 0.10320, 0.10180, 0.10985, 0.14134, 0.19479, 0.25962, 0.31682, 0.32363, 0.26800, 0.19160, 0.08720, -0.07711, -0.22800, -0.28454, -0.26522, -0.22780, -0.19954, -0.16907, -0.13541, -0.10743, -0.08370, -0.07571, -0.10077, -0.14753, -0.17356, -0.14518, -0.04176, 0.08451])],
    [([-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00001, -0.00000, 0.00001, 0.00003, 0.00007, 0.00014, 0.00023, 0.00034, 0.00045, 0.00077, 0.00153, 0.00265, 0.00372, 0.00467, 0.00575, 0.00653, 0.00748, 0.01040, 0.01528, 0.01996, 0.02274, 0.02297, 0.02204, 0.02013, 0.01551, 0.01072, 0.00782, 0.00551, 0.00347, 0.00200, 0.00105, 0.00048, 0.00022, 0.00011, 0.00005, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.00000, -0.00003, -0.00011, -0.00034, -0.00107, -0.00249, -0.00466, -0.00841, -0.01449, -0.02793, -0.04920, -0.06874, -0.08715, -0.11246, -0.14953, -0.19916, -0.25239, -0.31448, -0.36189, -0.36017, -0.34322, -0.32844, -0.30614, -0.27866, -0.22827, -0.15734, -0.09473, -0.04947, -0.02495, -0.01443, -0.00616, -0.00210, -0.00099, -0.00042, -0.00017, -0.00006, -0.00002, -0.00000, -0.00000, -0.00000]), ([-0.00000, -0.00000, -0.00001, -0.00003, -0.00004, -0.00005, -0.00003, 0.00005, 0.00022, 0.00048, 0.00084, 0.00132, 0.00185, 0.00238, 0.00352, 0.00573, 0.00842, 0.01073, 0.01295, 0.01505, 0.01555, 0.01437, 0.01240, 0.00933, 0.00479, -0.00055, -0.00505, -0.00766, -0.00890, -0.00886, -0.00775, -0.00658, -0.00537, -0.00393, -0.00265, -0.00163, -0.00090, -0.00049, -0.00028, -0.00014, -0.00005, -0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000], [-0.00005, -0.00036, -0.00138, -0.00391, -0.01140, -0.02434, -0.04014, -0.06239, -0.09472, -0.16072, -0.25909, -0.33293, -0.36669, -0.37376, -0.35329, -0.30686, -0.23334, -0.11897, 0.00456, 0.07950, 0.12108, 0.15211, 0.17901, 0.20447, 0.20512, 0.17619, 0.13093, 0.08264, 0.05168, 0.03373, 0.01680, 0.00734, 0.00399, 0.00195, 0.00088, 0.00037, 0.00011, 0.00002, 0.00000, 0.00000]), ([0.00000, 0.00001, 0.00002, 0.00006, 0.00010, 0.00011, 0.00005, -0.00009, -0.00033, -0.00068, -0.00110, -0.00159, -0.00211, -0.00266, -0.00344, -0.00449, -0.00535, -0.00573, -0.00610, -0.00617, -0.00472, -0.00172, 0.00230, 0.00630, 0.00837, 0.00738, 0.00439, 0.00140, -0.00232, -0.00651, -0.00877, -0.00982, -0.01023, -0.00934, -0.00764, -0.00573, -0.00399, -0.00270, -0.00178, -0.00103, -0.00046, -0.00016, -0.00007, -0.00002, -0.00001, -0.00000, -0.00000, -0.00000], [0.00019, 0.00103, 0.00362, 0.00957, 0.02561, 0.05058, 0.07455, 0.09939, 0.12858, 0.17731, 0.23822, 0.25784, 0.21834, 0.13355, 0.01769, -0.10142, -0.19316, -0.23153, -0.22049, -0.17715, -0.10363, -0.01609, 0.08064, 0.18576, 0.28056, 0.34828, 0.34879, 0.28849, 0.23412, 0.18047, 0.11080, 0.06381, 0.04075, 0.02313, 0.01210, 0.00585, 0.00203, 0.00043, 0.00007, 0.00001])],
    [([-0.00002, -0.00004, -0.00007, -0.00005, 0.00001, -0.00006, -0.00021, -0.00007, 0.00046, 0.00111, 0.00169, 0.00224, 0.00234, 0.00149, 0.00142, 0.00354, 0.00536, 0.00563, 0.00675, 0.01054, 0.01730, 0.03066, 0.05585, 0.08890, 0.11585, 0.12380, 0.11116, 0.08448, 0.04951, 0.01038, -0.02730, -0.05928, -0.08556, -0.10590, -0.11955, -0.12911, -0.13684, -0.13857, -0.13046, -0.11303, -0.09456, -0.07407, -0.04024, -0.00993, 0.00008, -0.00017, -0.00068, 0.00012], [-0.01060, -0.00923, -0.00054, -0.00499, -0.01116, 0.00771, 0.03671, 0.05648, 0.06187, 0.04990, 0.02881, 0.01053, -0.00220, -0.01644, -0.03884, -0.06931, -0.10396, -0.13661, -0.17055, -0.21361, -0.26067, -0.29210, -0.30199, -0.30330, -0.30079, -0.29910, -0.30138, -0.30002, -0.27972, -0.23367, -0.17451, -0.10852, -0.04924, -0.02354, -0.02521, -0.02603, -0.01841, -0.00859, -0.00110, 0.00087]), ([-0.00018, -0.00048, -0.00107, -0.00157, -0.00180, -0.00201, -0.00266, -0.00332, -0.00305, -0.00206, -0.00114, -0.00043, -0.00059, -0.00222, -0.00296, -0.00138, -0.00039, -0.00160, -0.00406, -0.00819, -0.01539, -0.02483, -0.02974, -0.03224, -0.04581, -0.06771, -0.08249, -0.08770, -0.08842, -0.08445, -0.07732, -0.06925, -0.06134, -0.05492, -0.04981, -0.04562, -0.04097, -0.03463, -0.02227, -0.00309, 0.00862, 0.00740, 0.00465, 0.00141, -0.00230, -0.00126, 0.00043, 0.00012], [-0.04261, -0.04421, -0.03177, -0.07584, -0.14910, -0.12709, -0.02867, 0.05804, 0.11003, 0.12415, 0.10684, 0.07928, 0.04757, 0.00186, -0.06022, -0.12568, -0.17862, -0.21553, -0.24093, -0.24965, -0.24219, -0.22606, -0.18763, -0.11260, -0.01474, 0.07867, 0.15681, 0.22616, 0.28476, 0.32490, 0.34018, 0.30190, 0.19949, 0.08786, 0.02107, 0.00118, 0.00154, 0.00156, 0.00003, 0.00007]), ([-0.00049, -0.00114, -0.00241, -0.00356, -0.00416, -0.00386, -0.00296, -0.00312, -0.00450, -0.00594, -0.00751, -0.00927, -0.01121, -0.01343, -0.01433, -0.01356, -0.01321, -0.01412, -0.01656, -0.02175, -0.02962, -0.03692, -0.03996, -0.03671, -0.02418, -0.00305, 0.01414, 0.02094, 0.02245, 0.01909, 0.01270, 0.00583, -0.00071, -0.00564, -0.00905, -0.01115, -0.01258, -0.01244, -0.00776, 0.00015, 0.00504, 0.00300, -0.00205, -0.00043, 0.00378, 0.00198, -0.00042, -0.00055], [-0.06057, -0.09308, -0.12699, -0.19702, -0.27830, -0.25484, -0.13939, -0.03545, 0.02021, 0.02030, -0.01586, -0.04728, -0.06626, -0.08971, -0.12336, -0.15126, -0.15560, -0.13551, -0.11101, -0.09116, -0.07235, -0.05479, -0.01705, 0.06269, 0.16214, 0.23822, 0.26143, 0.21866, 0.06108, -0.19332, -0.38424, -0.40572, -0.27407, -0.10853, -0.02234, -0.00016, 0.00814, 0.01044, 0.00373, -0.00348])],
],
}
_CG_BLEND_R = {
    0: [
    [([-0.00000, 0.00001, 0.00010, 0.00023, 0.00014, -0.00023, -0.00062, -0.00083, -0.00078, -0.00059, -0.00081, -0.00264, -0.00879, -0.02270, -0.04218, -0.06382, -0.11206, -0.17978, -0.21977, -0.23072, -0.21005, -0.16418, -0.12877, -0.10565, -0.08506, -0.06740, -0.05267, -0.04085, -0.03101, -0.02162, -0.01083, 0.00148, 0.01260, 0.02006, 0.02215, 0.01844, 0.01069, 0.00155, -0.00826, -0.02119, -0.03066, -0.02736, -0.01747, -0.00756, 0.00017, 0.00268, 0.00161, 0.00010], [-0.00001, -0.00001, -0.00005, -0.00014, -0.00012, -0.00004, -0.00007, 0.00038, 0.00085, -0.00092, -0.00491, -0.01033, -0.01540, -0.01654, -0.01555, -0.01621, -0.01749, -0.01648, -0.01392, -0.01221, -0.01130, -0.01166, -0.01137, -0.00666, -0.00156, -0.00643, -0.03387, -0.09441, -0.18374, -0.27869, -0.36791, -0.45141, -0.47480, -0.41649, -0.31135, -0.17825, -0.08181, -0.04504, -0.02204, -0.00305]), ([-0.00000, -0.00001, 0.00001, 0.00000, -0.00087, -0.00300, -0.00475, -0.00498, -0.00380, -0.00193, -0.00041, 0.00171, 0.00741, 0.01850, 0.02942, 0.03929, 0.06082, 0.08839, 0.08766, 0.03825, -0.01923, -0.04834, -0.06939, -0.08317, -0.08416, -0.08310, -0.08113, -0.08012, -0.07753, -0.07082, -0.05957, -0.04414, -0.02599, -0.00575, 0.01356, 0.02911, 0.03863, 0.04177, 0.03675, 0.02320, 0.00641, -0.00881, -0.01588, -0.01074, 0.00013, 0.00631, 0.00424, -0.00120], [-0.00017, -0.00017, -0.00013, 0.00012, 0.00026, 0.00010, -0.00033, -0.00151, -0.00208, 0.00154, 0.00728, 0.00550, -0.00603, -0.01839, -0.02741, -0.04277, -0.06660, -0.08539, -0.09616, -0.10288, -0.10220, -0.08666, -0.05933, -0.03079, 0.00018, 0.03791, 0.08760, 0.15019, 0.21209, 0.25464, 0.26971, 0.23943, 0.07181, -0.23359, -0.45436, -0.43926, -0.32007, -0.25337, -0.18777, -0.10194]), ([-0.00003, 0.00001, 0.00002, -0.00026, -0.00210, -0.00658, -0.01259, -0.01775, -0.01763, -0.01137, -0.00509, -0.00239, -0.00245, -0.00519, -0.00770, -0.01068, -0.02224, -0.03773, -0.03486, -0.00330, 0.02860, 0.03796, 0.03587, 0.03011, 0.01894, 0.00472, -0.01130, -0.02987, -0.04568, -0.05608, -0.06124, -0.06124, -0.05267, -0.03645, -0.02375, -0.01538, -0.00767, -0.00203, 0.00046, -0.00068, -0.00337, -0.00447, -0.00459, -0.00440, -0.00325, -0.00092, -0.00061, -0.00405], [-0.00173, -0.00205, -0.00258, -0.00266, -0.00195, -0.00140, -0.00255, -0.00432, -0.00542, -0.00940, -0.01440, -0.00687, 0.01251, 0.02429, 0.02198, 0.00320, -0.03769, -0.09300, -0.15854, -0.23549, -0.31200, -0.34845, -0.32538, -0.27424, -0.21461, -0.15091, -0.09410, -0.06193, -0.06587, -0.08776, -0.11373, -0.12184, 0.00308, 0.22876, 0.25662, 0.08621, -0.08410, -0.29407, -0.36385, -0.18688])],
    [([-0.00000, -0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00001, 0.00002, 0.00005, 0.00011, 0.00015, 0.00019, 0.00029, 0.00038, 0.00044, 0.00063, 0.00108, 0.00175, 0.00275, 0.00395, 0.00533, 0.00724, 0.00894, 0.01014, 0.01127, 0.01162, 0.01075, 0.00881, 0.00625, 0.00405, 0.00248, 0.00136, 0.00064, 0.00023, 0.00008, 0.00004, 0.00002, 0.00001, 0.00000, 0.00000, 0.00000], [0.00000, 0.00001, 0.00001, 0.00001, 0.00001, -0.00001, -0.00009, -0.00028, -0.00057, -0.00111, -0.00198, -0.00303, -0.00440, -0.00671, -0.01155, -0.02089, -0.03531, -0.06711, -0.10634, -0.13419, -0.17489, -0.23647, -0.30925, -0.37969, -0.41963, -0.41469, -0.36224, -0.28373, -0.21543, -0.15203, -0.08351, -0.03663, -0.01651, -0.00802, -0.00446, -0.00235, -0.00105, -0.00034, -0.00007, -0.00001]), ([0.00000, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00001, -0.00001, -0.00001, -0.00001, -0.00001, -0.00003, -0.00007, -0.00014, -0.00027, -0.00045, -0.00059, -0.00067, -0.00088, -0.00109, -0.00125, -0.00165, -0.00242, -0.00333, -0.00417, -0.00495, -0.00585, -0.00620, -0.00528, -0.00364, -0.00126, 0.00198, 0.00494, 0.00651, 0.00671, 0.00597, 0.00457, 0.00307, 0.00181, 0.00090, 0.00046, 0.00029, 0.00016, 0.00007, 0.00003, 0.00001, 0.00000], [-0.00007, -0.00013, -0.00021, -0.00025, -0.00021, 0.00006, 0.00098, 0.00292, 0.00555, 0.00953, 0.01512, 0.02131, 0.02859, 0.03924, 0.05873, 0.09113, 0.13217, 0.19782, 0.26437, 0.28731, 0.29961, 0.30628, 0.27088, 0.18236, 0.03890, -0.12153, -0.24625, -0.32566, -0.35426, -0.31632, -0.22531, -0.13220, -0.07349, -0.04515, -0.03128, -0.02115, -0.01236, -0.00542, -0.00150, -0.00020]), ([0.00000, 0.00000, -0.00000, -0.00001, -0.00001, -0.00002, -0.00003, -0.00004, -0.00005, -0.00006, -0.00005, -0.00006, -0.00012, -0.00025, -0.00048, -0.00080, -0.00114, -0.00134, -0.00140, -0.00160, -0.00185, -0.00209, -0.00256, -0.00322, -0.00370, -0.00365, -0.00317, -0.00271, -0.00167, 0.00049, 0.00254, 0.00351, 0.00303, 0.00141, -0.00057, -0.00267, -0.00406, -0.00422, -0.00379, -0.00318, -0.00256, -0.00214, -0.00178, -0.00122, -0.00062, -0.00029, -0.00013, -0.00003], [-0.00146, -0.00199, -0.00267, -0.00279, -0.00217, -0.00010, 0.00563, 0.01613, 0.02863, 0.04364, 0.06165, 0.07955, 0.09834, 0.12207, 0.16065, 0.21586, 0.26706, 0.30245, 0.30641, 0.25426, 0.15954, 0.04270, -0.08817, -0.19092, -0.22038, -0.17953, -0.07450, 0.07574, 0.21183, 0.29306, 0.30034, 0.24239, 0.18478, 0.16318, 0.15800, 0.14655, 0.11434, 0.06626, 0.02294, 0.00418])],
    [([-0.00034, -0.00064, -0.00125, -0.00199, -0.00291, -0.00425, -0.00553, -0.00607, -0.00592, -0.00524, -0.00435, -0.00411, -0.00505, -0.00734, -0.01078, -0.01460, -0.01949, -0.02637, -0.03434, -0.04302, -0.04974, -0.05122, -0.04874, -0.04642, -0.05144, -0.06740, -0.08511, -0.09584, -0.10058, -0.10245, -0.10307, -0.10013, -0.09383, -0.08691, -0.07947, -0.07164, -0.06352, -0.05578, -0.03830, -0.01338, -0.00374, -0.00697, -0.01158, -0.01435, -0.01173, -0.00699, -0.00325, -0.00092], [-0.25543, -0.25762, -0.26229, -0.26979, -0.28660, -0.29680, -0.29037, -0.27757, -0.26032, -0.23602, -0.20514, -0.17323, -0.14395, -0.11598, -0.09094, -0.07089, -0.05579, -0.04405, -0.03357, -0.02072, -0.00565, -0.00230, -0.01813, -0.04049, -0.05757, -0.06632, -0.05947, -0.02700, 0.02901, 0.09776, 0.16391, 0.19809, 0.18357, 0.13065, 0.06305, 0.01795, 0.00419, 0.00146, 0.00136, 0.00188]), ([-0.00008, -0.00023, -0.00050, -0.00078, -0.00107, -0.00130, -0.00132, -0.00136, -0.00161, -0.00187, -0.00200, -0.00214, -0.00254, -0.00330, -0.00456, -0.00673, -0.01030, -0.01446, -0.01798, -0.02137, -0.02371, -0.02190, -0.01682, -0.01081, -0.00276, 0.00893, 0.01836, 0.02319, 0.02760, 0.02984, 0.02813, 0.01878, 0.00415, -0.00984, -0.02308, -0.03320, -0.03843, -0.03859, -0.02804, -0.00678, 0.01284, 0.02553, 0.03382, 0.03502, 0.02649, 0.01545, 0.00796, 0.00386], [-0.03032, -0.05856, -0.10598, -0.13547, -0.13313, -0.12453, -0.11640, -0.10465, -0.08819, -0.06017, -0.02554, -0.00180, 0.00377, 0.00006, -0.00281, -0.00377, -0.00705, -0.01026, -0.00973, -0.00838, -0.02260, -0.06351, -0.10787, -0.13204, -0.13870, -0.14138, -0.15723, -0.18487, -0.22977, -0.31044, -0.40446, -0.45084, -0.39714, -0.25581, -0.12560, -0.07115, -0.04099, -0.02189, -0.02962, -0.03376]), ([0.00036, 0.00048, 0.00055, 0.00032, -0.00041, -0.00179, -0.00254, -0.00109, 0.00180, 0.00514, 0.00830, 0.01049, 0.01176, 0.01240, 0.01245, 0.01070, 0.00598, -0.00111, -0.00861, -0.01348, -0.01694, -0.02290, -0.03002, -0.03563, -0.03854, -0.03416, -0.02294, -0.01169, -0.00331, 0.00436, 0.01208, 0.01562, 0.01755, 0.02020, 0.01951, 0.01806, 0.01855, 0.01948, 0.01643, 0.00922, 0.00611, 0.00628, 0.00625, 0.00900, 0.01330, 0.01437, 0.00999, 0.00445], [0.28596, 0.31100, 0.29973, 0.22838, 0.07520, -0.07655, -0.13992, -0.15755, -0.17668, -0.20074, -0.22160, -0.22762, -0.21247, -0.18410, -0.15118, -0.11156, -0.06171, -0.01084, 0.02532, 0.03904, 0.03140, 0.00731, -0.03068, -0.07180, -0.07953, -0.02880, 0.05982, 0.16125, 0.23283, 0.21789, 0.11602, -0.04110, -0.20069, -0.26709, -0.20672, -0.10919, -0.04862, -0.02241, -0.01839, -0.02370])],
],
    100: [
    [([0.00001, 0.00005, 0.00012, 0.00004, -0.00023, -0.00070, -0.00248, -0.00506, -0.00530, -0.00103, 0.00615, 0.01239, 0.01815, 0.03005, 0.05143, 0.07238, 0.08847, 0.10162, 0.11056, 0.12349, 0.13513, 0.13440, 0.12581, 0.11028, 0.08844, 0.06478, 0.04227, 0.02249, 0.00496, -0.00947, -0.01999, -0.02735, -0.02992, -0.02665, -0.02014, -0.01286, -0.00618, -0.00125, 0.00058, -0.00098, -0.00375, -0.00555, -0.00600, -0.00477, -0.00266, -0.00172, -0.00104, -0.00003], [0.00004, 0.00010, 0.00030, 0.00096, 0.00276, 0.00579, 0.00821, 0.00707, 0.00237, -0.00125, 0.00168, 0.01760, 0.04115, 0.05316, 0.05088, 0.04509, 0.03613, 0.01859, -0.00572, -0.03103, -0.05249, -0.06270, -0.05505, -0.03009, 0.00780, 0.05661, 0.11635, 0.18581, 0.26297, 0.34038, 0.40258, 0.42720, 0.40175, 0.34925, 0.27180, 0.16811, 0.09065, 0.04533, 0.01546, 0.01293]), ([0.00001, -0.00002, -0.00009, -0.00001, -0.00230, -0.01076, -0.02510, -0.03668, -0.03414, -0.01744, 0.00215, 0.01409, 0.01938, 0.02466, 0.03062, 0.03506, 0.03615, 0.03461, 0.03054, 0.01896, 0.00451, -0.00692, -0.02059, -0.03413, -0.04620, -0.05804, -0.06750, -0.07482, -0.07351, -0.06164, -0.04455, -0.02464, -0.00378, 0.01323, 0.02103, 0.02076, 0.01601, 0.00951, 0.00343, -0.00138, -0.00379, -0.00342, -0.00113, 0.00065, 0.00092, 0.00189, 0.00141, -0.00154], [0.00132, -0.00057, -0.00240, -0.00170, 0.00311, 0.01010, 0.01292, 0.00551, -0.00961, -0.01892, -0.01634, -0.01587, -0.02766, -0.04683, -0.06986, -0.11695, -0.18927, -0.24548, -0.27533, -0.28876, -0.26996, -0.21676, -0.14323, -0.04720, 0.05446, 0.12923, 0.17280, 0.19506, 0.19629, 0.17260, 0.12400, 0.04163, -0.08212, -0.21021, -0.27421, -0.25600, -0.26213, -0.30582, -0.22032, -0.06222]), ([-0.00004, -0.00023, -0.00050, -0.00046, -0.00164, -0.00764, -0.02537, -0.05080, -0.06268, -0.05459, -0.03788, -0.02115, -0.00962, -0.00947, -0.01959, -0.02950, -0.03357, -0.03275, -0.02754, -0.01635, -0.00348, 0.00666, 0.01641, 0.02371, 0.02528, 0.02163, 0.01316, 0.00039, -0.01535, -0.03084, -0.04277, -0.04920, -0.04527, -0.03296, -0.02306, -0.01671, -0.01082, -0.00607, -0.00255, -0.00024, 0.00046, -0.00044, -0.00202, -0.00272, -0.00224, -0.00187, -0.00125, -0.00035], [0.00368, -0.00473, -0.01385, -0.01699, -0.01555, -0.01324, -0.01238, -0.01394, -0.01726, -0.02362, -0.03361, -0.03963, -0.03185, -0.01302, 0.00418, 0.01060, -0.00369, -0.04812, -0.11943, -0.21257, -0.30887, -0.37911, -0.40694, -0.37571, -0.29793, -0.23541, -0.20479, -0.16966, -0.13942, -0.12881, -0.11913, -0.07587, 0.02537, 0.13539, 0.18965, 0.19082, 0.13715, 0.01322, -0.03653, 0.06029])],
    [([-0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00001, -0.00001, -0.00002, -0.00004, -0.00005, -0.00006, -0.00007, 0.00002, 0.00030, 0.00070, 0.00122, 0.00227, 0.00398, 0.00620, 0.00816, 0.00966, 0.01123, 0.01166, 0.01088, 0.00974, 0.00782, 0.00558, 0.00363, 0.00189, 0.00078, 0.00038, 0.00018, 0.00003, -0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.00002, -0.00004, -0.00006, -0.00008, -0.00011, -0.00016, -0.00021, -0.00024, -0.00022, -0.00026, -0.00070, -0.00247, -0.00656, -0.01380, -0.02694, -0.04991, -0.08205, -0.14076, -0.20722, -0.24468, -0.28393, -0.33464, -0.38085, -0.41232, -0.39900, -0.33831, -0.24525, -0.13986, -0.06892, -0.03525, -0.01303, -0.00303, -0.00107, -0.00056, -0.00027, -0.00006, -0.00001, -0.00000, -0.00000, -0.00000]), ([-0.00004, -0.00011, -0.00072, -0.00198, -0.00385, -0.00608, -0.00785, -0.00839, -0.00873, -0.00924, -0.00832, -0.00614, -0.00435, -0.00317, -0.00186, -0.00072, -0.00020, -0.00002, 0.00004, 0.00009, 0.00010, 0.00007, 0.00004, 0.00002, 0.00001, 0.00000, -0.00000, -0.00000, -0.00000, -0.00001, -0.00001, -0.00001, -0.00001, -0.00001, -0.00001, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, 0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000, -0.00000], [-0.26949, -0.34303, -0.43627, -0.45764, -0.41788, -0.35495, -0.26615, -0.16805, -0.08790, -0.04101, -0.02198, -0.01543, -0.00981, -0.00459, -0.00217, -0.00099, -0.00026, 0.00004, -0.00011, -0.00024, -0.00018, 0.00006, 0.00026, 0.00039, 0.00045, 0.00044, 0.00037, 0.00024, 0.00014, 0.00008, 0.00003, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000]), ([-0.00001, -0.00002, -0.00006, -0.00013, -0.00022, -0.00030, -0.00031, -0.00020, -0.00006, 0.00006, 0.00019, 0.00028, 0.00030, 0.00030, 0.00030, 0.00035, 0.00038, 0.00037, 0.00034, 0.00005, -0.00066, -0.00160, -0.00258, -0.00363, -0.00456, -0.00481, -0.00449, -0.00421, -0.00273, 0.00014, 0.00249, 0.00427, 0.00542, 0.00553, 0.00474, 0.00318, 0.00183, 0.00117, 0.00064, 0.00013, -0.00002, 0.00001, 0.00001, 0.00000, 0.00000, 0.00000, 0.00000, 0.00000], [-0.02393, -0.02203, -0.01505, -0.00415, 0.00766, 0.01719, 0.02057, 0.01884, 0.01488, 0.01100, 0.01153, 0.02319, 0.05117, 0.09307, 0.14717, 0.20804, 0.25791, 0.30231, 0.33091, 0.31044, 0.25060, 0.16548, 0.05540, -0.06237, -0.19676, -0.33106, -0.36903, -0.30920, -0.23002, -0.14619, -0.06609, -0.02200, -0.01006, -0.00635, -0.00340, -0.00091, -0.00020, -0.00011, -0.00005, -0.00001])],
    [([-0.00097, -0.00199, -0.00304, -0.00359, -0.00497, -0.00703, -0.00827, -0.00881, -0.00952, -0.01070, -0.01189, -0.01235, -0.01214, -0.01193, -0.01183, -0.01117, -0.01054, -0.01103, -0.01242, -0.01536, -0.01879, -0.02081, -0.02210, -0.02501, -0.03566, -0.05693, -0.08057, -0.09882, -0.11137, -0.12095, -0.12948, -0.13547, -0.13894, -0.14153, -0.14255, -0.13966, -0.13109, -0.11863, -0.08185, -0.02523, 0.00153, 0.00080, -0.00212, -0.00641, -0.00905, -0.00787, -0.00438, -0.00133], [-0.28499, -0.28202, -0.28520, -0.29390, -0.30058, -0.29336, -0.27496, -0.25887, -0.24435, -0.22337, -0.19514, -0.16997, -0.15610, -0.14647, -0.13119, -0.11027, -0.08904, -0.06991, -0.05267, -0.03953, -0.03319, -0.03064, -0.02843, -0.02597, -0.02198, -0.01538, -0.00349, 0.01681, 0.04523, 0.07971, 0.11476, 0.13742, 0.13862, 0.11545, 0.07118, 0.02991, 0.00919, 0.00185, 0.00026, 0.00085]), ([-0.00046, -0.00092, -0.00127, -0.00124, -0.00139, -0.00116, 0.00026, 0.00148, 0.00129, 0.00027, -0.00082, -0.00189, -0.00307, -0.00433, -0.00547, -0.00615, -0.00656, -0.00700, -0.00700, -0.00676, -0.00513, 0.00098, 0.01022, 0.01950, 0.02835, 0.03404, 0.03329, 0.03028, 0.03011, 0.02884, 0.02329, 0.01458, 0.00353, -0.00724, -0.01529, -0.02412, -0.03621, -0.04663, -0.04304, -0.02586, -0.01371, -0.00714, 0.00122, 0.00657, 0.00351, -0.00462, -0.00730, -0.00416], [-0.07923, -0.12388, -0.17242, -0.17612, -0.10873, -0.04915, -0.04515, -0.05447, -0.03515, 0.03451, 0.12513, 0.17975, 0.16632, 0.10004, 0.03775, 0.00793, -0.00638, -0.01350, -0.01301, -0.01204, -0.02970, -0.05830, -0.07420, -0.08561, -0.11136, -0.16444, -0.23949, -0.31673, -0.37290, -0.39294, -0.37807, -0.30826, -0.16381, 0.01414, 0.12560, 0.12013, 0.06718, 0.02870, 0.00840, 0.00138]), ([-0.00022, -0.00058, -0.00097, -0.00119, -0.00157, -0.00173, -0.00106, -0.00032, -0.00033, -0.00083, -0.00139, -0.00186, -0.00232, -0.00293, -0.00367, -0.00429, -0.00479, -0.00533, -0.00591, -0.00677, -0.00777, -0.00855, -0.00870, -0.00843, -0.00928, -0.00991, -0.00954, -0.00742, -0.00294, 0.00346, 0.01052, 0.01074, 0.00863, 0.00881, 0.00393, -0.00257, -0.00713, -0.01146, -0.00799, 0.00162, 0.00425, 0.00401, 0.00889, 0.02074, 0.03722, 0.04503, 0.03478, 0.01768], [-0.05171, -0.04618, -0.05693, -0.08592, -0.13068, -0.14689, -0.11702, -0.08187, -0.05288, -0.01520, 0.02618, 0.05249, 0.05436, 0.03925, 0.02834, 0.02896, 0.03239, 0.03659, 0.03999, 0.03462, 0.01377, -0.00840, -0.01495, -0.00800, 0.01284, 0.03992, 0.05646, 0.06761, 0.06861, 0.03777, -0.04452, -0.17472, -0.33869, -0.49292, -0.51805, -0.38778, -0.22011, -0.10562, -0.08147, -0.12788])],
],
}

_CG_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
_CG_YC = ((np.arange(_CG_NB) + 0.5) / _CG_NB).astype(np.float32)
_CG_XCAP = ((np.arange(_CG_NBX) + 0.5) / _CG_NBX).astype(np.float32)
_CG_XC = ((np.arange(_CG_NBL) + 0.5) / _CG_NBL).astype(np.float32)


def _cg_srgb2lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _cg_lin2srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055).astype(np.float32)


def _cg_tint_vec(zone, hue, sat):
    """zone wheel 的线性域色偏向量（sat=60 为拟合基准）。"""
    b1, b2 = _CG_PLANE[zone]
    h = np.radians(float(hue) % 360.0)
    v = np.cos(h) * np.asarray(b1) + np.sin(h) * np.asarray(b2)
    hs, dvs = _CG_HUE_CORR[zone]
    hh = float(hue) % 360.0
    corr = np.array([np.interp(hh, hs, [dv[c] for dv in dvs], period=360.0)
                     for c in range(3)])
    ss = float(np.clip(float(sat), 0.0, 100.0))
    xs, ys = _CG_SAT[zone]
    r = np.interp(ss, xs, ys)
    sx, sv = _CG_SAT_CORR[zone]
    scorr = np.array([np.interp(ss, sx, [dv[c] for dv in sv]) for c in range(3)])
    return ((v + corr) * r + scorr).astype(np.float32)


def _cg_anchor_curve(table, zone, val):
    """锚点曲线族在 val 处的线性插值（锚点范围外线性外推，供 balance ±100）。"""
    ks, curves = table[zone]
    Cv = np.asarray(curves, dtype=np.float32)
    if val <= ks[0]:
        i0, i1 = 0, 1
    elif val >= ks[-1]:
        i0, i1 = len(ks) - 2, len(ks) - 1
    else:
        i1 = int(np.searchsorted(ks, val))
        i0 = i1 - 1
    t = (val - ks[i0]) / (ks[i1] - ks[i0])
    return (1.0 - t) * Cv[i0] + t * Cv[i1]


def _cg_sat_cap(zone, sat):
    """cap(x) 在 sat 处的锚点插值（锚点外夹取）。"""
    ks, curves = _CG_CAPS[zone]
    Cv = np.asarray(curves, dtype=np.float32)
    if len(ks) == 1:
        return Cv[0]
    return np.stack([np.interp(sat, ks, Cv[:, i]) for i in range(Cv.shape[1])])


def _cg_zone_weight(zone, y, balance, blending):
    """W_zone(Y)；balance/blending 用 GT 逐条件拟合的锚点曲线（仅 shadow/highlight）。"""
    W = np.asarray(_CG_W[zone], dtype=np.float32)
    if zone in ("shadow", "highlight"):
        bal = float(np.clip(balance, -100.0, 100.0))
        bl = float(np.clip(blending, 0.0, 100.0))
        if bal != 0.0 or bl != 50.0:
            Wb = _cg_anchor_curve(_CG_BAL_W, zone, bal)
            Wl = _cg_anchor_curve(_CG_BLEND_W, zone, bl)
            W = np.maximum(Wb + Wl - W, 0.0)
    return np.interp(y, _CG_YC, W).astype(np.float32)


def _cg_zone_cap(zone, sat, balance, blending):
    """cap_zone(x)；sat 锚点插值 + balance/blending 条件锚点修正（仅 shadow/highlight）。"""
    cap = _cg_sat_cap(zone, float(np.clip(sat, 0.0, 100.0)))
    if zone in ("shadow", "highlight"):
        bal = float(np.clip(balance, -100.0, 100.0))
        bl = float(np.clip(blending, 0.0, 100.0))
        if bal != 0.0 or bl != 50.0:
            cap50 = _cg_sat_cap(zone, 50.0)
            cb = _cg_anchor_curve(_CG_BAL_CAP, zone, bal)
            cl = _cg_anchor_curve(_CG_BLEND_CAP, zone, bl)
            cap = np.maximum(cap + (cb - cap50) + (cl - cap50), 0.0)
    return cap.astype(np.float32)


def _cg_lowrank_add(d, y, x, terms, w):
    """d_c += w * Σ_k a_k(Y) * b_k(x_c)（低秩 (Y,x_c) 残差表，每通道）。"""
    if w == 0.0:
        return
    for c in range(3):
        acc = 0.0
        for ay, bx in terms[c]:
            acc = acc + (np.interp(y, _CG_YC, np.asarray(ay, dtype=np.float32))
                         * np.interp(x[..., c], _CG_XCAP, np.asarray(bx, dtype=np.float32)))
        d[..., c] += (w * acc).astype(np.float32)


def lr_color_grade(img, shadow=(0.0, 0.0, 0.0), midtone=(0.0, 0.0, 0.0),
                   highlight=(0.0, 0.0, 0.0), global_=(0.0, 0.0, 0.0),
                   balance=0.0, blending=50.0):
    """统一 3-way color grade（LR 语义，GT 标定）。

    img: float32 [0,1] sRGB RGB；每 zone = (hue 0-360, sat 0-100, lum -100..100)。
    """
    x = np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0)
    y = (x @ _CG_LUMA).astype(np.float32)
    xl = _cg_srgb2lin(x)
    d = np.zeros_like(xl)
    zones = (("shadow", shadow), ("mid", midtone), ("highlight", highlight),
             ("global", global_))
    for zone, (h, s, l) in zones:
        if s and s > 0:
            v = _cg_tint_vec(zone, h, s)
            W = _cg_zone_weight(zone, y, float(balance), float(blending))
            cap = _cg_zone_cap(zone, float(s), float(balance), float(blending))
            for c in range(3):
                d[..., c] += W * v[c] * np.interp(x[..., c], _CG_XCAP, cap).astype(np.float32)
            if zone == "mid" and s > 60.0:
                _cg_lowrank_add(d, y, x, _CG_MID_SATD, (min(float(s), 100.0) - 60.0) / 40.0)
        if l:
            g = np.asarray(_CG_LUM[zone + ("+" if l > 0 else "-")], dtype=np.float32)
            k = abs(float(l)) / 60.0
            for c in range(3):
                d[..., c] += k * np.interp(xl[..., c], _CG_XC, g).astype(np.float32)
    # 双 wheel 复合下 balance/blending 的交互残差（sat50 标定，按幅度缩放）
    s_sh, s_hi = float(shadow[1] or 0.0), float(highlight[1] or 0.0)
    if s_sh > 0 and s_hi > 0:
        amp = 0.5 * (np.interp(min(s_sh, 100.0), *_CG_SAT["shadow"])
                     / max(np.interp(50.0, *_CG_SAT["shadow"]), 1e-6)
                     + np.interp(min(s_hi, 100.0), *_CG_SAT["highlight"])
                     / max(np.interp(50.0, *_CG_SAT["highlight"]), 1e-6))
        bal = float(np.clip(balance, -100.0, 100.0))
        bl = float(np.clip(blending, 0.0, 100.0))
        if bal != 0.0:
            _cg_lowrank_add(d, y, x, _CG_BAL_R[-70 if bal < 0 else 70],
                            amp * abs(bal) / 70.0)
        if bl != 50.0:
            _cg_lowrank_add(d, y, x, _CG_BLEND_R[0 if bl < 50 else 100],
                            amp * abs(bl - 50.0) / 50.0)
    return _cg_lin2srgb(xl + d)

# ---- Grain（胶片颗粒；仅 Size=25/Freq=50 有 GT，size/freq 为参数化外推）----
# ---- 标定常数（tools/lr_calib/scratch/grain_*.py 拟合，2026-07）----
_GRAIN_HP_MIX = 0.46          # 高通混合系数 m（frequency=50）
_GRAIN_HP_SIGMA = 0.65        # 高通高斯尺度 σg px（size=25）
_GRAIN_RES_K = 244.9          # 分辨率因子：σ ∝ 1 + k / min(H,W)
_GRAIN_AMP_X = (0.0, 25.0, 50.0, 80.0, 100.0)              # LR GrainAmount
_GRAIN_AMP_Y = (0.0, 0.03048, 0.06002, 0.09508, 0.11846)   # 平台 σ（min_dim→∞ 基准）
# 亮度 rolloff（在 [0,1] 硬剪之外的额外衰减；对 GT 池化亮度分箱 σ 剖面网格拟合）
_GRAIN_SHADOW_Y0 = 0.0        # 阴影 ramp 起点（w=_GRAIN_SHADOW_W）
_GRAIN_SHADOW_Y1 = 0.10       # 阴影 ramp 终点（w=1）
_GRAIN_SHADOW_W = 0.52
_GRAIN_HIGH_Y0 = 0.915        # 高光 ramp 起点（w=1）
_GRAIN_HIGH_Y1 = 1.0          # 高光 ramp 终点（w=_GRAIN_HIGH_W）
_GRAIN_HIGH_W = 0.51

_GRAIN_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def _grain_seed(image: np.ndarray) -> int:
    """图像内容哈希 → 可复现种子（同图同输出；对下采样 uint8 量化取 CRC32）。"""
    q = np.clip(image[::16, ::16] * 255.0, 0, 255).astype(np.uint8)
    return zlib.crc32(q.tobytes()) & 0xFFFFFFFF


def _grain_luma_weight(luma: np.ndarray) -> np.ndarray:
    """亮度权重：中间调平台 1.0，极暗/极亮端线性 ramp 衰减（剪切之外的部分）。"""
    w = np.ones_like(luma)
    if _GRAIN_SHADOW_Y1 > _GRAIN_SHADOW_Y0:
        t = np.clip((luma - _GRAIN_SHADOW_Y0) / (_GRAIN_SHADOW_Y1 - _GRAIN_SHADOW_Y0), 0.0, 1.0)
        w *= _GRAIN_SHADOW_W + (1.0 - _GRAIN_SHADOW_W) * t
    if _GRAIN_HIGH_Y1 > _GRAIN_HIGH_Y0:
        t = np.clip((luma - _GRAIN_HIGH_Y0) / (_GRAIN_HIGH_Y1 - _GRAIN_HIGH_Y0), 0.0, 1.0)
        w *= 1.0 + (_GRAIN_HIGH_W - 1.0) * t
    return w


def apply_grain(image: np.ndarray, amount: float, size: float = 25.0,
                frequency: float = 50.0, seed: int | None = None) -> np.ndarray:
    """LR 风格胶片颗粒。image: float32 [0,1] RGB；amount/size/frequency: LR 0..100 语义。

    单通道伪随机颗粒场（高斯带通谱形）× 亮度权重 → 等量加到 RGB → [0,1] 剪切。
    """
    if amount <= 0:
        return image
    img = np.asarray(image, dtype=np.float32)
    h, w = img.shape[:2]
    if seed is None:
        seed = _grain_seed(img)
    rng = np.random.default_rng(seed)
    n = rng.standard_normal((h, w), dtype=np.float32)
    # 谱形：size 拉大 → 低通尺度变大（颗粒变粗）；frequency 提高 → 高通占比变大（更糙）
    sg = _GRAIN_HP_SIGMA * max(float(size), 1.0) / 25.0
    m = min(_GRAIN_HP_MIX * float(frequency) / 50.0, 0.95)
    g = n - m * gaussian_filter(n, sg)
    g /= max(float(g.std()), 1e-8)
    # 幅度：amount 标定曲线 × 分辨率因子
    sigma = float(np.interp(float(amount), _GRAIN_AMP_X, _GRAIN_AMP_Y))
    sigma *= 1.0 + _GRAIN_RES_K / float(min(h, w))
    luma = img @ _GRAIN_LUMA
    field = (sigma * g * _grain_luma_weight(luma)).astype(np.float32)
    return np.clip(img + field[..., None], 0.0, 1.0)

# ---- Dehaze（LR-faithful 轻量重实现；per-value 参数烘焙自 fits/Dehaze.json；
# +70/+100 于 2026-07 用 scratch/dehaze_p70_refit.py 重拟合（新增 kd/hk/acx/wvl
# 机制）。全分辨率 8 探针 ΔE00 mean/p95：-70 2.73/5.88、-40 1.77/3.58、
# +40 1.74/3.65、+70 2.82/5.78、+100 4.08/8.13。
# 已知缺口：+70/+100 仍未达 2.0/4.0 门——逐探针 oracle（旧形态逐图自由调参）
# 下界即 +70≈2.56 / +100≈3.69，残差是传输图逐像素结构差异（LR 对白色物体/
# 平滑雾区的区分优于暗通道），需更根本的传输估计模型。----
# 每扫描值参数（fits/Dehaze.json per_value 烘焙）。键 = str(int(v))。
_DEHAZE_DEFAULT_PARAMS: dict = json.loads(r'''{"-70": {"k0": -0.0172, "k1": 0.7368, "k2": 0.1693, "gdc": 0.5052, "sp": 0.8113, "fl0": 0.3123, "fl1": 3.6512, "flg": 0.0, "vk": 0.0, "dthr": 0.09, "desat": 0.2536, "a_mix": 0.964}, "-40": {"k0": -0.0046, "k1": 0.42, "k2": 0.1074, "gdc": 0.5854, "sp": 0.8546, "fl0": 0.0, "fl1": 0.0, "flg": 0.0427, "vk": 0.0, "dthr": 0.0907, "desat": 0.092, "a_mix": 0.9}, "40": {"k0": 0.01, "k1": 0.3899, "k2": 0.05, "gdc": 1.1461, "sp": 1.0, "sat": 0.0671, "a_mix": 1.0, "lc": 0.02, "fl0": 0.0, "fl1": 0.0, "dthr": 0.0}, "70": {"k0": 0.0201, "k1": 0.3728, "k2": 0.1766, "gdc": 0.9224, "sp": 1.0, "sat": 0.1035, "a_mix": 1.0, "lc": 0.0743, "fl0": 0.0271, "fl1": 0.0, "dthr": 0.0943, "conv": 0.98, "apiv": 1.0, "smax": 0.9841, "kq": -1.0164, "kd": 0.6015, "hk": 0.33, "acx": 0.058, "wvl": 0.06}, "100": {"k0": 0.005, "k1": 0.5395, "k2": 0.2448, "gdc": 0.739, "sp": 1.0, "sat": 0.1058, "a_mix": 0.9693, "lc": 0.1434, "fl0": 0.179, "fl1": 0.8205, "dthr": 0.1103, "conv": 0.7185, "apiv": 1.0, "smax": 1.6856, "kq": -0.6323, "kd": 0.6329, "hk": 0.3957, "acx": 0.166, "wvl": 0.0473}}''')
# 强度类参数：跨值插值时以 v=0 → 0 为锚；其余（比例/混合类）取最近锚点值。
# 某锚点缺失的强度类键按 0 处理（防止相邻锚点的模式开关互相泄漏）。
_DEHAZE_STRENGTH_KEYS = {"k0", "k1", "k2", "sat", "desat", "lc", "fl0", "conv", "flg", "vk",
                         "kd", "hk", "acx", "wvl"}

def _srgb2lin(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92,
                    ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def _lin2srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * x ** (1.0 / 2.4) - 0.055).astype(np.float32)

def _dehaze_box(img: np.ndarray, r: int) -> np.ndarray:
    k = 2 * int(r) + 1
    return cv2.boxFilter(img, -1, (k, k), borderType=cv2.BORDER_REFLECT)


def _dehaze_guided_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float,
                 sub: int = 4) -> np.ndarray:
    """灰度引导滤波（可下采样加速）。"""
    if sub > 1:
        h, w = guide.shape
        sw, sh = max(1, w // sub), max(1, h // sub)
        g = cv2.resize(guide, (sw, sh), interpolation=cv2.INTER_AREA)
        s = cv2.resize(src, (sw, sh), interpolation=cv2.INTER_AREA)
        f = _dehaze_guided_gray(g, s, max(1, radius // sub), eps, sub=1)
        return cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    mean_i, mean_p = _dehaze_box(guide, radius), _dehaze_box(src, radius)
    corr_i, corr_ip = _dehaze_box(guide * guide, radius), _dehaze_box(guide * src, radius)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return (_dehaze_box(a, radius) * guide + _dehaze_box(b, radius)).astype(np.float32)


def _haze_features(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (dc_fine 细化暗通道, 大气光 A[3])。img: float32 [0,1] RGB。

    A = max(top-1% 暗通道像素均色, 每通道 q95)（逐通道取大，对 8 探针的 GT
    隐含 A 全部命中；见 scratch/dehaze_aest.py）。
    """
    short = min(img.shape[:2])
    luma = (img @ _LUMA_W).astype(np.float32)
    dc_raw = img.min(axis=2)
    er = max(1, int(round(short * 0.01)))
    dc = cv2.erode(dc_raw, np.ones((2 * er + 1, 2 * er + 1), np.uint8))
    dc = _dehaze_guided_gray(luma, dc, radius=max(4, int(round(short * 0.04))), eps=1e-3)
    dc = np.clip(dc, 0.0, 1.0)
    n = dc.size
    k = max(1, int(n * 0.01))
    idx = np.argpartition(dc.ravel(), n - k)[n - k:]
    flat = img.reshape(-1, 3)
    a_col = np.maximum(flat[idx].mean(axis=0), np.quantile(flat, 0.95, axis=0))
    return dc, np.clip(a_col, 0.0, 1.0).astype(np.float32)


def _coarse_dc(img: np.ndarray) -> np.ndarray:
    """大半径暗通道（图像内容自适应分量，~25% 短边 box）。"""
    short = min(img.shape[:2])
    return _dehaze_box(img.min(axis=2), max(8, int(round(short * 0.25))))


def _sat_gain(img: np.ndarray, amount: float) -> np.ndarray:
    """色度乘性增益，vibrance 型高色度 rolloff（正向去雾的饱和补偿）。"""
    if abs(amount) < 1e-6:
        return img
    chroma = img.max(axis=2) - img.min(axis=2)
    roll = np.clip(1.0 - chroma / 0.85, 0.0, 1.0)
    gain = (1.0 + amount * roll)[..., None]
    mean = img.mean(axis=2, keepdims=True)
    return mean + (img - mean) * gain


def lr_dehaze(img: np.ndarray, amount: float, params: dict | None = None) -> np.ndarray:
    """LR Dehaze 近似。img: float32 [0,1] RGB；amount ∈ [-100, 100]。"""
    if abs(amount) < 1e-3:
        return img
    p = _dehaze_resolve_params(amount, params)
    img = np.asarray(img, dtype=np.float32)
    dcf, a_col = _haze_features(img)
    dcc = _coarse_dc(img)
    a_mix = float(p.get("a_mix", 1.0))
    A = ((1.0 - a_mix) * np.ones(3, dtype=np.float32) + a_mix * a_col)[None, None, :]
    a_luma = max(float(a_col @ _LUMA_W) * a_mix + (1.0 - a_mix), 0.4)
    k0, k1, k2 = (float(p.get(k, 0.0)) for k in ("k0", "k1", "k2"))
    # 雾密度驱动：逐像素 min-channel 与平滑 dc 混合（sp）→ 按大气光归一 → 次线性 ramp
    sp = float(p.get("sp", 0.0))
    driver = ((1.0 - sp) * img.min(axis=2) + sp * dcf) / a_luma
    gdc = float(p.get("gdc", 1.0))
    dark_frac = float((driver < float(p.get("dthr", 0.12))).mean())
    # 图像级自适应：高雾像素越常见（driver q95 越高），场越强（scratch/dehaze_sfield.py）
    kq = float(p.get("kq", 0.0))
    if abs(kq) > 1e-6:
        k1 = k1 * max(1.0 + kq * (float(np.quantile(driver, 0.95)) - 0.7), 0.0)
    if abs(gdc - 1.0) > 1e-6:
        driver = np.power(np.clip(driver, 0.0, 1.5), gdc)
    field = k0 + k1 * driver + k2 * dcc
    # 暗部保护（2026-07 +70/+100 重拟合，scratch/dehaze_p70_refit.py）：暗像素占比
    # 越大整体场越弱——LR 对多暗部图缩减正向 dehaze 以保护黑点（乘性版 darkpix）。
    kd = float(p.get("kd", 0.0))
    if abs(kd) > 1e-6:
        field = field * max(1.0 + kd * (0.2 - dark_frac), 0.0)
    # 效果下限（直方图自适应）：暗像素占比越小，整图（含暗像素）施加的雾/拉伸越接近
    # 全局水平——LR 对高调 / 少暗部图不保护黑点（scratch/dehaze_darkpix.py）。
    fl0 = float(p.get("fl0", 0.0))
    flg = float(p.get("flg", 0.0))
    vk = float(p.get("vk", 0.0))
    if abs(flg) > 1e-6:
        # log 形式（darkpix 回归）：floor = fl0 + flg·(−ln(dark_frac))，暗像素越少雾越均匀
        floor_v = max(fl0 + flg * float(-np.log(dark_frac + 0.005)), 0.0)
    else:
        floor_v = fl0 * np.exp(-float(p.get("fl1", 2.6)) * dark_frac)
    if abs(floor_v) > 1e-9:
        if vk > 1e-6:
            # 加性衰减 floor（GT veil 曲线 = v0 + 饱和上升，见 dehaze_sfield.py -070）：
            # 场弱处补 v0，场强处 floor 自然淡出，总量饱和
            field = field + floor_v * np.exp(-vk * np.clip(field, 0.0, None))
        else:
            field = np.maximum(field, floor_v)

    if amount > 0:
        # 亮度倾斜（2026-07 重拟合）：场强随原图 luma 线性调制（亮区略强/暗区略弱可调）
        hk = float(p.get("hk", 0.0))
        if abs(hk) > 1e-6:
            field = field * (1.0 + hk * ((img @ _LUMA_W) - 0.6))
        # 传输图凸性：s = ω·dc/(1-ω·dc)（conv→1 时为物理 dehaze 形式，conv=0 退化为线性）
        conv = float(p.get("conv", 0.0))
        if conv > 1e-6:
            field = field / (1.0 - conv * np.clip(field, 0.0, 0.92))
        s = np.clip(field, 0.0, float(p.get("smax", 3.0)))
        # 大气光色度放大（2026-07 重拟合）：拉伸 pivot 的色度按 (1+acx) 放大，
        # 匹配 GT 去雾后向 A 补色方向的色偏（去蓝 → 黄化）。
        acx = float(p.get("acx", 0.0))
        if abs(acx) > 1e-6:
            A = a_luma + (1.0 + acx) * (A - a_luma)
        # 超白 pivot：LR 在强正向时把高光也向下压 → 拉伸中心抬到白点之上
        apiv = float(p.get("apiv", 1.0))
        if float(p.get("lin", 0.0)) > 0.5:
            # 线性光域拉伸（GT s(driver) 曲线族在线性域更收拢，scratch/dehaze_lin.py）
            img_l = _srgb2lin(img)
            A_l3 = _srgb2lin(np.clip(A, 0.0, 1.0)) * apiv
            out = _lin2srgb((img_l - A_l3) * (1.0 + s)[..., None] + A_l3)
        else:
            A_piv = A * apiv if apiv > 1.0 else A
            # 波长相关传输（2026-07 重拟合，Rayleigh 散射近似）：蓝通道透过率更低
            # → 拉伸更强；s_c = s·(1 + wvl·d_c)，d = (-0.4, 0, 1)。
            wvl = float(p.get("wvl", 0.0))
            if abs(wvl) > 1e-6:
                d_c = np.asarray([-0.4, 0.0, 1.0], dtype=np.float32)
                out = (img - A_piv) * (1.0 + s[..., None] * (1.0 + wvl * d_c)) + A_piv
            else:
                out = (img - A_piv) * (1.0 + s)[..., None] + A_piv
        out = _sat_gain(out, float(p.get("sat", 0.0)))
        lc = float(p.get("lc", 0.0))
        if abs(lc) > 1e-6:
            luma = (np.clip(out, 0.0, 1.0) @ _LUMA_W).astype(np.float32)
            short = min(img.shape[:2])
            base = _dehaze_guided_gray(luma, luma, radius=max(4, int(round(short * 0.05))),
                                eps=4e-3)
            out = out + (lc * (luma - base))[..., None]
    else:
        veil = np.clip(field, 0.0, float(p.get("vmax", 0.95)))
        out = img * (1.0 - veil[..., None]) + A * veil[..., None]
        desat = float(p.get("desat", 0.0))
        if abs(desat) > 1e-6:
            # 去饱和与局部雾量成比例（雾越浓越灰），向 luma 灰
            luma = (out @ _LUMA_W)[..., None]
            w = np.clip(desat * veil[..., None], 0.0, 1.0)
            out = out + w * (luma - out)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _dehaze_resolve_params(amount: float, fit_params: dict | None) -> dict:
    """按 |v| 在拟合锚点间插值；强度类参数以 v=0 → 0 锚定，比例类取邻近锚点。"""
    table = dict(_DEHAZE_DEFAULT_PARAMS)
    if fit_params:
        for k, v in fit_params.items():
            table[str(int(float(k)))] = v
    anchors = sorted(((float(k), v) for k, v in table.items()
                      if (float(k) > 0) == (amount > 0)), key=lambda kv: abs(kv[0]))
    if not anchors:
        return {}
    a = abs(amount)
    lo_v, lo_p, hi_v, hi_p = 0.0, None, None, None
    for av, ap in anchors:
        if abs(av) >= a:
            hi_v, hi_p = abs(av), ap
            break
        lo_v, lo_p = abs(av), ap
    if hi_p is None:              # 超出最大锚点：用最后两个锚点线性外推
        hi_v, hi_p = abs(anchors[-1][0]), anchors[-1][1]
        if lo_v == hi_v and len(anchors) >= 2:
            lo_v, lo_p = abs(anchors[-2][0]), anchors[-2][1]
    if lo_p is None:              # 下锚点是 v=0 恒等：强度类=0，比例类取上锚点
        lo_p = {k: (0.0 if k in _DEHAZE_STRENGTH_KEYS else v) for k, v in hi_p.items()}
    if hi_v == lo_v:
        return dict(hi_p)
    f = (a - lo_v) / (hi_v - lo_v)
    keys = set(lo_p) | set(hi_p)
    out = {}
    for k in keys:
        if k in _DEHAZE_STRENGTH_KEYS:      # 缺失强度键 = 该锚点无此项效果（0），不得借值
            lo = float(lo_p.get(k, 0.0))
            hi = float(hi_p.get(k, 0.0))
        else:
            lo = float(lo_p.get(k, hi_p.get(k, 0.0)))
            hi = float(hi_p.get(k, lo))
        out[k] = lo + f * (hi - lo)
    return out

# ---- BlackWhite（LR ConvertToGrayscale + GrayMixer 8 色混合）----
BW_BANDS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
# 标定常数（fits/BlackWhite.json bands 烘焙；旧模块初值已废弃）
BW_BAND_PARAMS: dict = json.loads(r'''{"Red": {"center": 355.0, "width": 45.0, "gain": 1.7978, "default": 2.99}, "Orange": {"center": 25.0, "width": 45.0, "gain": 1.807, "default": 3.0}, "Yellow": {"center": 60.0, "width": 45.0, "gain": 1.8002, "default": -4.0}, "Green": {"center": 115.0, "width": 45.0, "gain": 1.8014, "default": 2.0}, "Aqua": {"center": 180.0, "width": 45.0, "gain": 1.8035, "default": -6.0}, "Blue": {"center": 215.0, "width": 45.0, "gain": 1.7843, "default": 2.0}, "Purple": {"center": 275.0, "width": 45.0, "gain": 1.8, "default": 2.0}, "Magenta": {"center": 320.0, "width": 45.0, "gain": 1.8, "default": 2.0}}''')

def _hsv_hue_sat(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """HSV 色相（度）与饱和度，向量化 numpy。"""
    mx = rgb.max(-1)
    c = mx - rgb.min(-1)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    h = np.zeros_like(mx)
    m = c > 1e-8
    idx = m & (mx == r)
    h[idx] = ((g - b)[idx] / c[idx]) % 6.0
    idx = m & (mx == g) & (mx != r)
    h[idx] = (b - r)[idx] / c[idx] + 2.0
    idx = m & (mx == b) & (mx != r) & (mx != g)
    h[idx] = (r - g)[idx] / c[idx] + 4.0
    h *= 60.0
    s = np.where(mx > 1e-8, c / np.maximum(mx, 1e-8), 0.0).astype(np.float32)
    return h.astype(np.float32), s


def convert_black_white(image: np.ndarray, gray_mixer: dict | None = None,
                        band_params: dict | None = None) -> np.ndarray:
    """LR 黑白转换。image: float32 [0,1] RGB；gray_mixer: {band: slider(-100..100)}，
    缺省带用标定的 LR 默认混合。返回三通道相等的灰度 RGB。"""
    params = band_params or BW_BAND_PARAMS
    mixer = gray_mixer or {}
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    lin = _srgb_to_linear(img)
    y = lin @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    hue, sat = _hsv_hue_sat(img)

    f = np.zeros_like(y)
    for band in BW_BANDS:
        p = params[band]
        m = float(mixer.get(band, p["default"]))
        if abs(m) < 1e-9:
            continue
        dh = np.abs((hue - float(p["center"]) + 180.0) % 360.0 - 180.0)
        t = np.minimum(dh / max(float(p["width"]), 1.0), 1.0)
        w = np.cos(0.5 * np.pi * t) ** 2
        f += float(p["gain"]) * w * (m / 100.0)
    gray = y * np.maximum(1.0 + sat * f, 0.0)
    out = _linear_to_srgb(gray)
    return np.repeat(out[..., None], 3, axis=-1)

# ---- ColorNR（彩噪抑制）与 LensManualDistortion（径向畸变，越界白填充）----
_YW = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)

# ---------------------------------------------------------------------------
# 拟合常数（scratch/colornr_fit.py + colornr_refine.py / fringeprobe_dist_fit.py 输出）
# amount=0 锚点为恒等；中间 amount 线性插值。
# ---------------------------------------------------------------------------
COLOR_NR_PARAMS = [  # [amount, sigma_space, sigma_color, alpha]
    [0.0, 0.0, 0.05, 0.0],
    [25.0, 10.0, 0.05, 0.85],
    [100.0, 20.0, 0.05, 1.0],
]
DIST_AB = [  # [LensManualDistortionAmount, a, b]
    [-50.0, -0.2508341, 0.0009294],
    [0.0, 0.0, 0.0],
    [50.0, 0.1975663, 0.0027494],
]


def _interp_rows(rows: list, x: float) -> list:
    """按首列对表格逐列线性插值。"""
    rows = sorted(rows)
    xs = [r[0] for r in rows]
    return [float(np.interp(x, xs, [r[i] for r in rows])) for i in range(1, len(rows[0]))]


def apply_color_nr(img: np.ndarray, amount: float, params: list | None = None) -> np.ndarray:
    """LR ColorNoiseReduction：opponent 色度通道保边（bilateral）平滑，亮度不动。"""
    if amount <= 0:
        return img
    sigma_s, sigma_c, alpha = _interp_rows(params or COLOR_NR_PARAMS, float(amount))
    if sigma_s <= 0 or alpha <= 0:
        return img
    img = np.clip(img, 0.0, 1.0).astype(np.float32, copy=False)
    y = img @ _YW
    cb = img[..., 2] - y
    cr = img[..., 0] - y
    fcb = cv2.bilateralFilter(cb, 0, sigma_c, sigma_s)
    fcr = cv2.bilateralFilter(cr, 0, sigma_c, sigma_s)
    cb = cb + alpha * (fcb - cb)
    cr = cr + alpha * (fcr - cr)
    r = y + cr
    b = y + cb
    g = (y - 0.299 * r - 0.114 * b) / 0.587
    return np.clip(np.stack([r, g, b], axis=-1), 0.0, 1.0).astype(np.float32)


def apply_manual_distortion(img: np.ndarray, amount: float,
                            ab_table: list | None = None) -> np.ndarray:
    """LR LensManualDistortionAmount：角点锚定径向畸变，越界白色填充。"""
    if amount == 0:
        return img
    a, b = _interp_rows(ab_table or DIST_AB, float(amount))
    h, w = img.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    R = float(np.sqrt(cx * cx + cy * cy))
    nx, ny = np.meshgrid((np.arange(w, dtype=np.float32) - cx) / R,
                         (np.arange(h, dtype=np.float32) - cy) / R)
    u = 1.0 - (nx * nx + ny * ny)
    scale = 1.0 + a * u + b * u * u
    map_x = (nx * scale * R + cx).astype(np.float32)
    map_y = (ny * scale * R + cy).astype(np.float32)
    return np.clip(cv2.remap(np.clip(img, 0.0, 1.0).astype(np.float32, copy=False),
                             map_x, map_y, cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(1.0, 1.0, 1.0)), 0.0, 1.0)

# ============================================================================
# LR-faithful ops v2 —— 合入块结束
# ============================================================================


def adjust_dehaze(image, norm_factor, delta, runtime_config=None):
    """LR-faithful Dehaze（轻量重实现，替代已弃用的 FFT 版路径）。

    delta: 归一 [-1,1]（×100 转 LR 单位）或 LR 原生 [-100,100]。
    已知缺口：+70/+100 强正向存在结构性残差（标定聚合 2.69/5.73）。
    """
    del norm_factor, runtime_config
    amount = _safe_float(delta, 0.0)
    if abs(amount) < 1e-8:
        return image
    if abs(amount) <= 1.0:
        amount *= 100.0
    img = np.clip(image, 0.0, 1.0).astype(np.float32, copy=False)
    return lr_dehaze(img, amount)


def _lr_units_value(value) -> float:
    """config 值 → LR 滑杆单位：|v|≤1 视为归一值 ×100，其余原样透传。"""
    v = _safe_float(value, 0.0)
    return v * 100.0 if abs(v) <= 1.0 else v


def _clip01_f32(image) -> np.ndarray:
    return np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)


def _resolve_color_grade_settings(config: dict) -> dict | None:
    """从 config 的 CRS 键解析 lr_color_grade 参数；无有效设置时返回 None。

    SplitToning 与 ColorGrade 的 shadow/highlight 共用同一组 CRS 字段；
    ColorGradeBlending 缺省必须取 50（LR 默认），不是 0。
    """
    def f(key, default=0.0):
        return _safe_float(config.get(key, default), default)

    shadow = (f("SplitToningShadowHue"), f("SplitToningShadowSaturation"),
              f("ColorGradeShadowLum"))
    highlight = (f("SplitToningHighlightHue"), f("SplitToningHighlightSaturation"),
                 f("ColorGradeHighlightLum"))
    midtone = (f("ColorGradeMidtoneHue"), f("ColorGradeMidtoneSat"),
               f("ColorGradeMidtoneLum"))
    global_ = (f("ColorGradeGlobalHue"), f("ColorGradeGlobalSat"),
               f("ColorGradeGlobalLum"))
    active = any(z[1] > 0 or abs(z[2]) > 1e-8 for z in (shadow, highlight, midtone, global_))
    if not active:
        return None
    return {
        "shadow": shadow,
        "midtone": midtone,
        "highlight": highlight,
        "global_": global_,
        "balance": f("SplitToningBalance"),
        "blending": f("ColorGradeBlending", 50.0),
    }


def _apply_non_gimp_config_impl(config, image, norm_factor, runtime_config=None):
    """Apply one normalized non-GIMP config dict to an in-memory image.

    阶段顺序对齐 LR 内部管线：
      几何(LensManualDistortion) → WB(Temperature/Tint) → Exposure/Contrast →
      Highlights/Shadows/Whites/Blacks → 参数曲线 → 点曲线(ToneCurve) →
      Clarity/Texture/Dehaze → 校准面板 → HSL/黑白 → SplitTone/ColorGrade →
      Vibrance/Saturation → Detail(Sharpen/NR) → Grain → Vignette。
    值域约定与旧版兼容：标量键 [-100,100]（|v|≤1 视为归一值）。
    """
    # 0. 几何（LR 先做镜头校正再渲染其余）
    dist = _safe_float(config.get("LensManualDistortionAmount", 0), 0.0)
    if abs(dist) > 1e-8:
        image = apply_manual_distortion(_clip01_f32(image), dist)

    # 1. 白平衡
    for op_name, op_fn in (("Temperature", adjust_temperature), ("Tint", adjust_tint)):
        val = config.get(op_name, 0)
        if isinstance(val, dict) or _has_nonzero_payload(val):
            image = op_fn(image, _normalize_global_compat_intensity(val),
                          runtime_config=runtime_config)

    # 2. Exposure / Contrast
    exposure = _normalize_global_compat_intensity(config.get("Exposure", 0))
    image = adjust_exposure(image, exposure, runtime_config=runtime_config)
    contrast_val = config.get("Contrast", 0)
    if isinstance(contrast_val, dict) or _has_nonzero_payload(contrast_val):
        image = adjust_contrast(image, _normalize_global_compat_intensity(contrast_val),
                                runtime_config=runtime_config)

    # 3. Highlights / Shadows / Whites / Blacks
    shadows = _normalize_global_compat_intensity(config.get("Shadows", 0))
    highlights = _normalize_global_compat_intensity(config.get("Highlights", 0))
    whites = _normalize_global_compat_intensity(config.get("Whites", 0))
    tone_kwargs = resolve_tone_controls(config.get("ToneControls", {}))
    image = adjust_tones(
        image,
        shadows=shadows,
        highlights=highlights,
        whites=whites,
        **tone_kwargs,
        runtime_config=runtime_config,
    )
    blacks_val = config.get("Blacks", 0)
    if isinstance(blacks_val, dict) or _has_nonzero_payload(blacks_val):
        image = adjust_blacks(image, _normalize_global_compat_intensity(blacks_val),
                              runtime_config=runtime_config)

    # 4. 参数曲线（LR Parametric Tone Curve，与点曲线同一阶段、先于点曲线）
    for pk in ("ParametricShadows", "ParametricDarks", "ParametricLights", "ParametricHighlights"):
        v = _lr_units_value(config.get(pk, 0))
        if abs(v) > 1e-8:
            image = apply_lr_parametric(_clip01_f32(image), pk, v)

    # 5. 点曲线（Melissa RGB 域 PCHIP；见 image_ops/curve.py）
    from .curve import apply_tone_curve, is_tone_curve_spec, sanitize_tone_curve_spec

    tone_curve_cfg = config.get("ToneCurve")
    if tone_curve_cfg is None and is_tone_curve_spec(config):
        # Allow writing curve-only JSON files either as {"ToneCurve": ...}
        # or as a direct curve object/list for convenience.
        tone_curve_cfg = config
    if tone_curve_cfg is not None:
        curve_spec = sanitize_tone_curve_spec(tone_curve_cfg)
        # torch 后端的 apply_tone_curve_torch 仍是旧输出域算法，未移植 Melissa 往返，
        # 统一走 numpy 路径以免两后端结果分叉。
        image = apply_tone_curve(image, curve_spec, norm_factor=norm_factor)

    # 6. 局部对比 / 雾
    for op_name, op_fn in (("Clarity", adjust_clarity), ("Texture", adjust_texture)):
        val = config.get(op_name, 0)
        if isinstance(val, dict) or _has_nonzero_payload(val):
            image = op_fn(image, _normalize_global_compat_intensity(val),
                          runtime_config=runtime_config)
    dehaze_delta = config.get("Dehaze", 0)
    if _has_nonzero_payload(dehaze_delta):
        image = adjust_dehaze(image, norm_factor, dehaze_delta, runtime_config=runtime_config)

    # 6.5 校准面板（相机 profile 阶段的等效线性变换）
    calib_sliders = {
        name: _lr_units_value(config.get(name, 0))
        for name in ("RedHue", "RedSaturation", "GreenHue", "GreenSaturation",
                     "BlueHue", "BlueSaturation", "ShadowTint")
        if _has_nonzero_payload(config.get(name, 0))
    }
    if calib_sliders:
        image = apply_lr_calibration_panel(_clip01_f32(image), calib_sliders)

    # 7. 黑白 or HSL（LR 灰度模式下 HSL/Saturation/Vibrance 无效，GrayMixer 取而代之）
    grayscale = str(config.get("ConvertToGrayscale", "")).strip().lower() in {"true", "1"}
    if grayscale:
        mixer = {
            band: _lr_units_value(config.get(f"GrayMixer{band}", 0))
            for band in BW_BANDS
            if f"GrayMixer{band}" in config
        }
        image = convert_black_white(_clip01_f32(image), mixer)
    else:
        image = execute_hsl(config, image, runtime_config=runtime_config)

    # 8. 分离色调 / ColorGrade
    cg_settings = _resolve_color_grade_settings(config)
    if cg_settings is not None:
        image = lr_color_grade(_clip01_f32(image), **cg_settings)

    # 9. 饱和族（灰度模式跳过）
    if not grayscale:
        for op_name, op_fn in (("Vibrance", adjust_vibrance), ("Saturation", adjust_saturation)):
            val = config.get(op_name, 0)
            if isinstance(val, dict) or _has_nonzero_payload(val):
                image = op_fn(image, _normalize_global_compat_intensity(val),
                              runtime_config=runtime_config)

    # 10. 细节（锐化 / 降噪）
    sharp_val = config.get("Sharpness", 0)
    if isinstance(sharp_val, dict):
        image = adjust_sharpness(image, sharp_val, runtime_config=runtime_config)
    elif _has_nonzero_payload(sharp_val):
        amount = float(np.clip(_lr_units_value(sharp_val), 0.0, 150.0))
        if amount > 0:
            image = lr_sharpen(
                _clip01_f32(image),
                amount=amount,
                radius=float(np.clip(_safe_float(config.get("SharpenRadius", 1.0), 1.0), 0.5, 3.0)),
                detail=float(np.clip(_safe_float(config.get("SharpenDetail", 25.0), 25.0), 0.0, 100.0)),
                masking=float(np.clip(_safe_float(config.get("SharpenEdgeMasking", 0.0), 0.0), 0.0, 100.0)),
            )
    nr_val = config.get("LuminanceNoiseReduction", 0)
    if isinstance(nr_val, dict) or _has_nonzero_payload(nr_val):
        image = adjust_luminance_noise_reduction(
            image, _normalize_global_compat_intensity(nr_val), runtime_config=runtime_config)
    cnr = _lr_units_value(config.get("ColorNoiseReduction", 0))
    if cnr > 0:
        image = apply_color_nr(_clip01_f32(image), cnr)

    # 11. 颗粒（渲染末段、暗角之前）
    grain_amount = _lr_units_value(config.get("GrainAmount", 0))
    if grain_amount > 0:
        image = apply_grain(
            _clip01_f32(image),
            grain_amount,
            size=_safe_float(config.get("GrainSize", 25), 25.0),
            frequency=_safe_float(config.get("GrainFrequency", 50), 50.0),
        )

    # 12. 暗角（post-crop 在前、镜头暗角在后，与 LR 一致）
    vig = _resolve_vignette_settings(config)
    if vig["postcrop"] is not None:
        image = apply_postcrop_vignette(_clip01_f32(image), **vig["postcrop"])
    if abs(vig["lens_amount"]) > 1e-8:
        image = apply_lens_vignette(_clip01_f32(image), vig["lens_amount"])

    return image


def apply_non_gimp_config(config, image, norm_factor, runtime_config=None):
    """
    Compatibility-only in-memory helper retained for low-level tests and primitive composition.

    Active file-based execution now goes through `execution_core.ImageOpExecutionCore`.
    """
    return _apply_non_gimp_config_impl(
        config,
        image,
        norm_factor,
        runtime_config=runtime_config,
    )


def write_image(image, norm_factor, output_path):
    """Write a processed image to disk, preserving TIFF bit depth."""
    if output_path.endswith(".tif"):
        save_tif(image, norm_factor, output_path)
    else:
        # Convert to uint8 for PNG output
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 1)
            image = (image * 255).astype(np.uint8)
        output_image = Image.fromarray(image)
        output_image.save(output_path)


def execute_non_gimp_pipeline(config_path, image_path, output_path, pipeline_config=None):
    """[gpu_render vendor 注] monetGPT 的文件式执行入口，依赖其 execution_core /
    shared.repo_config，不在 preset 渲染路径上（渲染走 apply_non_gimp_config /
    gpu_render.replay / gpu_render.gpu.render_batch），故裁剪为 stub。
    如需该功能请用 monetGPT 原仓库。"""
    raise NotImplementedError(
        "execute_non_gimp_pipeline 未随 gpu_render 迁移（依赖 monetGPT 的 "
        "execution_core）；渲染请走 gpu_render.replay 或 gpu_render.gpu.render_batch"
    )
