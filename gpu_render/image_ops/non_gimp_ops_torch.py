from __future__ import annotations

import math

import numpy as np

from .curve import build_curve_lut, build_blacks_lut, sanitize_tone_curve_spec

try:
    import torch
except Exception as exc:  # pragma: no cover - environment guard
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:  # pragma: no cover
    _TORCH_IMPORT_ERROR = None

try:
    import kornia
except Exception as exc:  # pragma: no cover - environment guard
    kornia = None
    _KORNIA_IMPORT_ERROR = exc
else:  # pragma: no cover
    _KORNIA_IMPORT_ERROR = None


_TAU = math.tau
_ACHROMATIC_THRESHOLD = 0.005
_RGB_LUMA = np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
_TINT_DEFAULT_BASE_KELVIN = 6504.0
SELECTED_TONE_RT_SHADOW_RADIUS = 40.0
SELECTED_TONE_RT_SHADOW_TONAL_WIDTH = 30.0
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


def ensure_torch_cuda_runtime(device: str = "cuda"):
    if torch is None:
        raise RuntimeError(f"Torch backend requested, but importing torch failed: {_TORCH_IMPORT_ERROR}")
    if kornia is None:
        raise RuntimeError(
            f"Torch backend requested, but importing kornia failed: {_KORNIA_IMPORT_ERROR}"
        )
    if str(device).strip().lower() != "cuda":
        raise RuntimeError("Torch non-GIMP backend requires torch_device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("Torch non-GIMP backend requires CUDA, but torch.cuda.is_available() is False.")
    return torch.device("cuda")


def _to_tensor_image(image: np.ndarray, device: torch.device) -> torch.Tensor:
    img = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 float image, got shape={img.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(img)).to(device=device, dtype=torch.float32)
    return tensor.permute(2, 0, 1).unsqueeze(0)


def _to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.squeeze(0)
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .detach()
        .float()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )


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
    return local_value * 60.0


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


def _resolve_tone_execution_payloads(
    highlights,
    shadows,
    whites,
    *,
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


def _gegl_kelvin_to_rgb_torch(temp_kelvin: float, device: torch.device) -> torch.Tensor:
    temp = float(np.clip(temp_kelvin, 1000.0, 12000.0))
    coeffs = torch.as_tensor(_GEGL_COLOR_TEMPERATURE_RGB_R55, device=device, dtype=torch.float64)
    rgb = []
    for channel in range(3):
        nomin = coeffs[channel, 0]
        for degree in range(1, 6):
            nomin = nomin * temp + coeffs[channel, degree]

        denom = coeffs[channel, 6]
        for degree in range(1, 6):
            denom = denom * temp + coeffs[channel, 6 + degree]

        rgb.append(nomin / denom)
    return torch.stack(rgb).to(dtype=torch.float32)


def _temperature_coeffs_torch(base_kelvin: float, delta_kelvin: float, device: torch.device) -> torch.Tensor:
    del base_kelvin
    normalized = float(np.clip(delta_kelvin / 5000.0, -1.0, 1.0))
    rb_coeffs = torch.as_tensor(
        [
            math.exp(0.35 * normalized),
            1.0 - 0.08 * normalized,
            math.exp(-0.35 * normalized),
        ],
        device=device,
        dtype=torch.float32,
    )
    luma = torch.as_tensor(_RGB_LUMA, device=device, dtype=torch.float32)
    return rb_coeffs / torch.dot(rb_coeffs, luma).clamp(min=1e-6)


def _wrap_hue_distance_torch(hue: torch.Tensor, center: float) -> torch.Tensor:
    diff = torch.abs(hue - float(center))
    return torch.minimum(diff, 1.0 - diff)


def _apply_vibrance_rt_torch(
    image: torch.Tensor,
    vibrance_pct: float,
    *,
    protect_skin: bool,
) -> torch.Tensor:
    amount = float(vibrance_pct) / 100.0
    hue, sat, val = _rgb_to_hsv_unit(image)
    if amount >= 0.0:
        weight = torch.pow((1.0 - sat).clamp(0.0, 1.0), 1.5)
        if protect_skin:
            skin_hue = 1.0 - _smoothstep(_wrap_hue_distance_torch(hue, 0.08), 0.02, 0.09)
            skin_sat = _smoothstep(sat, 0.10, 0.65)
            skin_val = _smoothstep(val, 0.15, 0.95)
            skin_mask = (skin_hue * skin_sat * skin_val).clamp(0.0, 1.0)
            weight = weight * (1.0 - 0.65 * skin_mask)
        sat = sat + amount * weight * (1.0 - sat)
    else:
        reduce_weight = 0.45 + 0.55 * sat
        sat = sat + amount * reduce_weight * sat
    sat = sat.clamp(0.0, 1.0)
    return _hsv_unit_to_rgb(hue, sat, val.clamp(0.0, 1.0)).clamp(0.0, 1.0)


def _rgb_luma_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return 0.299 * tensor[:, 0] + 0.587 * tensor[:, 1] + 0.114 * tensor[:, 2]


def _luma_ratio_recompose(tensor: torch.Tensor, src_luma: torch.Tensor, dst_luma: torch.Tensor) -> torch.Tensor:
    ratio = dst_luma / (src_luma + 1e-6)
    return (tensor * ratio.unsqueeze(1)).clamp(0.0, 1.0)


def _midtone_mask_torch(light: torch.Tensor) -> torch.Tensor:
    broad = _smoothstep(light, 0.10, 0.40) * (1.0 - _smoothstep(light, 0.60, 0.90))
    center = 1.0 - _smoothstep(torch.abs(light - 0.5), 0.12, 0.34)
    return (broad * (0.55 + 0.90 * center)).clamp(0.0, 1.0)


def _detail_activity_torch(detail: torch.Tensor, low: float = 0.0015, high: float = 0.03) -> torch.Tensor:
    return _smoothstep(detail.abs(), low, high)


def _clarity_edge_guard_torch(light: torch.Tensor) -> torch.Tensor:
    edge_strength = (light - _gaussian_blur_image(light.unsqueeze(1), 0.75, border_type="reflect")[:, 0]).abs()
    return 1.0 - _smoothstep(edge_strength, 0.025, 0.10)


def _sharpness_edge_mask_torch(light: torch.Tensor) -> torch.Tensor:
    edge_strength = (
        _gaussian_blur_image(light.unsqueeze(1), 0.5, border_type="reflect")[:, 0]
        - _gaussian_blur_image(light.unsqueeze(1), 1.4, border_type="reflect")[:, 0]
    ).abs()
    return _smoothstep(edge_strength, 0.004, 0.035)


def _sharpness_tone_mask_torch(light: torch.Tensor) -> torch.Tensor:
    return _smoothstep(light, 0.04, 0.18) * (1.0 - _smoothstep(light, 0.82, 0.96))


def _sharpness_halo_guard_torch(light: torch.Tensor) -> torch.Tensor:
    strong_edge = (light - _gaussian_blur_image(light.unsqueeze(1), 1.6, border_type="reflect")[:, 0]).abs()
    return 1.0 - _smoothstep(strong_edge, 0.03, 0.12)


def _local_variance_torch(light: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    mean = _gaussian_blur_image(light.unsqueeze(1), sigma, border_type="reflect")[:, 0]
    mean_sq = _gaussian_blur_image((light * light).unsqueeze(1), sigma, border_type="reflect")[:, 0]
    return (mean_sq - mean * mean).clamp_min(0.0)


def _luminance_nr_spike_mask_torch(light: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    median3 = _median_blur_image(light.unsqueeze(1), 3)[:, 0]
    base_noise = _local_variance_torch(median3, sigma=1.0).sqrt()
    spike_floor = torch.maximum(torch.full_like(base_noise, 0.08), 4.0 * base_noise + 0.02)
    spike_mask = (light - median3).abs() > spike_floor
    return median3, spike_mask


def _apply_rgb_spike_cleanup_torch(image: torch.Tensor, spike_mask: torch.Tensor) -> torch.Tensor:
    if not bool(spike_mask.any()):
        return image
    median_rgb = _median_blur_image(image, 3)
    return torch.where(spike_mask.unsqueeze(1), median_rgb, image)


def _edge_preserving_luma_smooth_torch(
    light: torch.Tensor,
    kernel: int,
    sigma_color: float,
    sigma_space: float,
    downsample: int = 1,
) -> torch.Tensor:
    if downsample > 1:
        height = max(1, int(light.shape[-2]) // downsample)
        width = max(1, int(light.shape[-1]) // downsample)
        small = torch.nn.functional.interpolate(
            light.unsqueeze(1),
            size=(height, width),
            mode="area",
        )
        kernel_size = max(5, int(round(kernel / downsample)))
        small_out = kornia.filters.bilateral_blur(
            small,
            _odd_kernel_size(kernel_size),
            float(max(1e-6, sigma_color)),
            (float(max(1e-6, sigma_space)), float(max(1e-6, sigma_space))),
            border_type="reflect",
            color_distance_type="l1",
        )
        return torch.nn.functional.interpolate(
            small_out,
            size=(int(light.shape[-2]), int(light.shape[-1])),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
    kernel_size = _odd_kernel_size(kernel)
    return kornia.filters.bilateral_blur(
        light.unsqueeze(1),
        kernel_size,
        float(max(1e-6, sigma_color)),
        (float(max(1e-6, sigma_space)), float(max(1e-6, sigma_space))),
        border_type="reflect",
        color_distance_type="l1",
    )[:, 0]


def _kernel_size_from_sigma(sigma: float) -> tuple[int, int]:
    base = max(3, int(math.ceil(float(max(0.1, sigma)) * 6.0)))
    if base % 2 == 0:
        base += 1
    return (base, base)


def _odd_kernel_size(kernel: int) -> tuple[int, int]:
    kernel = max(1, int(kernel))
    if kernel % 2 == 0:
        kernel += 1
    return (kernel, kernel)


def _gaussian_blur_image(tensor: torch.Tensor, sigma: float, border_type: str = "reflect") -> torch.Tensor:
    sigma = float(max(0.1, sigma))
    return kornia.filters.gaussian_blur2d(
        tensor,
        _kernel_size_from_sigma(sigma),
        (sigma, sigma),
        border_type=border_type,
    )


def _median_blur_image(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
    return kornia.filters.median_blur(tensor, _odd_kernel_size(kernel))


def _unsharp_mask_image(tensor: torch.Tensor, sigma: float, border_type: str = "reflect") -> torch.Tensor:
    sigma = float(max(0.1, sigma))
    return kornia.filters.unsharp_mask(
        tensor,
        _kernel_size_from_sigma(sigma),
        (sigma, sigma),
        border_type=border_type,
    )


def _edge_aware_smooth_delta(
    delta_light: torch.Tensor,
    guide_gray: torch.Tensor,
    guided_cfg: dict | None,
) -> torch.Tensor:
    guided_cfg = guided_cfg if isinstance(guided_cfg, dict) else {}
    if not guided_cfg.get("enabled", True):
        return delta_light

    radius = max(1, int(guided_cfg.get("radius", 5)))
    eps = float(max(1e-8, guided_cfg.get("eps", 1e-3)))
    kernel = _odd_kernel_size(radius * 2 + 1)
    delta_4d = delta_light.unsqueeze(0).unsqueeze(0)
    guide_4d = guide_gray.unsqueeze(0).unsqueeze(0)

    try:
        return kornia.filters.guided_blur(
            guide_4d,
            delta_4d,
            kernel,
            eps,
            border_type="reflect",
        )[0, 0]
    except Exception:
        sigma_color = float(max(1e-6, guided_cfg.get("sigma_color", 0.06)))
        sigma_space = float(max(1.0, guided_cfg.get("sigma_space", float(radius))))
        try:
            return kornia.filters.bilateral_blur(
                delta_4d,
                kernel,
                sigma_color,
                (sigma_space, sigma_space),
                border_type="reflect",
            )[0, 0]
        except Exception:
            sigma = max(1.0, sigma_space * 0.5)
            return _gaussian_blur_image(delta_4d, sigma, border_type="reflect")[0, 0]


def _median_blend_where(
    tensor: torch.Tensor,
    spike_mask: torch.Tensor,
    kernel: int,
    blend: float,
) -> torch.Tensor:
    filtered = _median_blur_image(tensor, kernel)
    mask = spike_mask
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    return torch.where(mask, (1.0 - blend) * tensor + blend * filtered, tensor)


def _smoothstep(x: torch.Tensor, edge0: float, edge1: float) -> torch.Tensor:
    t = ((x - edge0) / (edge1 - edge0 + 1e-8)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _vignette_geometry_torch(
    height: int,
    width: int,
    roundness: float,
    center_x: float,
    center_y: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    yy = torch.linspace(-1.0, 1.0, steps=height, device=device, dtype=torch.float32) - float(center_y)
    xx = torch.linspace(-1.0, 1.0, steps=width, device=device, dtype=torch.float32) - float(center_x)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")

    boxy = torch.maximum(grid_x.abs(), grid_y.abs())
    ellipse = torch.sqrt((grid_x * grid_x + grid_y * grid_y).clamp_min(0.0))

    min_dim = float(max(1, min(height, width)))
    scale_x = float(width) / min_dim
    scale_y = float(height) / min_dim
    circular = torch.sqrt(((grid_x * scale_x) ** 2 + (grid_y * scale_y) ** 2).clamp_min(0.0))

    t = float(np.clip(roundness, 0.0, 1.0))
    if t <= 0.5:
        mix = t / 0.5
        distance = (1.0 - mix) * boxy + mix * ellipse
    else:
        mix = (t - 0.5) / 0.5
        distance = (1.0 - mix) * ellipse + mix * circular
    return distance, grid_x, grid_y


def _apply_vignette_dither_torch(
    mask: torch.Tensor,
    grid_x: torch.Tensor,
    grid_y: torch.Tensor,
    dither_amount: float,
) -> torch.Tensor:
    if dither_amount <= 0:
        return mask
    phase = torch.sin((grid_x + 1.37) * 12.9898 + (grid_y - 0.41) * 78.233)
    noise = phase * 43758.5453
    noise = noise - torch.floor(noise)
    noise = (noise - 0.5) * 2.0
    jitter = noise * (float(dither_amount) / 255.0)
    return (mask + jitter).clamp(0.0, 1.0)


def _rgb_to_hls_unit(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hls = kornia.color.rgb_to_hls(tensor)
    hue = torch.remainder(hls[:, 0] / _TAU, 1.0)
    light = hls[:, 1]
    sat = hls[:, 2]
    return hue, light, sat


def _rgb_to_hsv_unit(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hsv = kornia.color.rgb_to_hsv(tensor)
    hue = torch.remainder(hsv[:, 0] / _TAU, 1.0)
    sat = hsv[:, 1]
    val = hsv[:, 2]
    return hue, sat, val


def _hls_unit_to_rgb(hue: torch.Tensor, light: torch.Tensor, sat: torch.Tensor) -> torch.Tensor:
    hls = torch.stack([torch.remainder(hue, 1.0) * _TAU, light, sat], dim=1)
    return kornia.color.hls_to_rgb(hls).clamp(0.0, 1.0)


def _hsv_unit_to_rgb(hue: torch.Tensor, sat: torch.Tensor, val: torch.Tensor) -> torch.Tensor:
    hsv = torch.stack([torch.remainder(hue, 1.0) * _TAU, sat, val], dim=1)
    return kornia.color.hsv_to_rgb(hsv).clamp(0.0, 1.0)


def _rgb_to_lab_image(tensor: torch.Tensor) -> torch.Tensor:
    return kornia.color.rgb_to_lab(tensor)


def _lab_to_rgb_image(tensor: torch.Tensor) -> torch.Tensor:
    return kornia.color.lab_to_rgb(tensor).clamp(0.0, 1.0)


def _resolve_clarity_params(intensity) -> dict[str, float | str]:
    from gpu_render.image_ops.operator_spec import get_operator_spec

    spec_defaults = dict(get_operator_spec("Clarity").default_aux_params or {})
    if isinstance(intensity, dict):
        payload = dict(spec_defaults)
        payload.update(intensity)
        clarity_detail = float(payload.get("clarity_detail", 0.0))
    else:
        payload = dict(spec_defaults)
        clarity_detail = float(np.clip(float(intensity), -1.0, 1.0)) * 0.65
        payload["clarity_detail"] = clarity_detail

    return {
        "clarity_detail": float(np.clip(clarity_detail, -0.65, 0.65)),
        "profile": str(payload.get("profile", "local_laplacian_selected")),
        "midtone_range": float(np.clip(float(payload.get("midtone_range", 0.5)), 0.05, 1.0)),
        "guide_radius_px": float(max(1.0, float(payload.get("guide_radius_px", 18.0)))),
        "guide_sigma_color": float(np.clip(float(payload.get("guide_sigma_color", 0.04)), 1e-4, 1.0)),
        "clip_limit_scale": float(np.clip(float(payload.get("clip_limit_scale", 0.9)), 0.1, 2.0)),
    }


def _clarity_curve_response_torch(
    light: torch.Tensor,
    guide: torch.Tensor,
    *,
    clarity_detail: float,
    sigma: float,
    clip_limit_scale: float,
) -> torch.Tensor:
    c = light - guide
    sigma = float(max(0.05, sigma))
    denom = float(max(1e-6, 2.0 * sigma * sigma / 3.0))
    tone_weight = (0.05 + 0.95 * _midtone_mask_torch(guide)).clamp(0.0, 1.0)
    delta = float(clarity_detail) * c * torch.exp(-(c * c) / denom)
    delta = delta * tone_weight
    if clarity_detail < 0.0:
        delta = delta * 2.15
    limit_scale = 1.15 if clarity_detail < 0.0 else float(max(0.1, clip_limit_scale))
    limit = torch.maximum(torch.full_like(delta, 0.004), c.abs() * limit_scale)
    return torch.clamp(delta, min=-limit, max=limit)


def _lut_tensor_from_u8(lut_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(lut_u8.astype(np.float32) / 255.0, device=device, dtype=torch.float32)


def _apply_lut_channel_torch(channel: torch.Tensor, lut_u8: np.ndarray) -> torch.Tensor:
    lut = _lut_tensor_from_u8(lut_u8, channel.device)
    pos = channel.clamp(0.0, 1.0) * float(lut.shape[0] - 1)
    idx0 = torch.floor(pos).long().clamp(0, lut.shape[0] - 1)
    idx1 = (idx0 + 1).clamp(0, lut.shape[0] - 1)
    frac = (pos - idx0.float()).clamp(0.0, 1.0)
    return lut[idx0] * (1.0 - frac) + lut[idx1] * frac


def adjust_clarity_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    params = _resolve_clarity_params(intensity)
    clarity_detail = float(params["clarity_detail"])
    if abs(clarity_detail) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    lab = _rgb_to_lab_image(img)
    light = (lab[:, 0] / 100.0).clamp(0.0, 1.0)
    guide_radius = float(params["guide_radius_px"])
    guide = _edge_preserving_luma_smooth_torch(
        light,
        kernel=max(5, int(round(guide_radius * 2.0))),
        sigma_color=float(params["guide_sigma_color"]),
        sigma_space=guide_radius,
    )
    delta = _clarity_curve_response_torch(
        light,
        guide,
        clarity_detail=clarity_detail,
        sigma=float(params["midtone_range"]),
        clip_limit_scale=float(params["clip_limit_scale"]),
    )
    lab[:, 0] = (light + delta).clamp(0.0, 1.0) * 100.0
    return _to_numpy_image(_lab_to_rgb_image(lab))


def _resolve_sharpness_params(intensity) -> dict[str, float | str]:
    from gpu_render.image_ops.operator_spec import get_operator_spec

    spec_defaults = dict(get_operator_spec("Sharpness").default_aux_params or {})
    if isinstance(intensity, dict):
        payload = dict(spec_defaults)
        payload.update(intensity)
        sharpness_amount = float(payload.get("sharpness_amount", 0.0))
    else:
        payload = dict(spec_defaults)
        amount_norm = max(0.0, float(np.clip(float(intensity), -1.0, 1.0)))
        sharpness_amount = amount_norm * 2.0
        payload["sharpness_amount"] = sharpness_amount

    sharpness_amount = float(np.clip(sharpness_amount, 0.0, 2.0))
    return {
        "sharpness_amount": sharpness_amount,
        "amount_norm": sharpness_amount / 2.0,
        "profile": str(payload.get("profile", "rt_usm_selected")),
        "radius_px": float(max(0.1, float(payload.get("radius_px", 2.0)))),
        "threshold_pct": float(np.clip(float(payload.get("threshold_pct", 0.5)), 0.0, 100.0)),
    }


def adjust_vignette_torch(
    image: np.ndarray,
    intensity: float,
    feather: float = 0.5,
    roundness: float = 0.5,
    center_x: float = 0.0,
    center_y: float = 0.0,
    dither_amount: float = 0.0,
    device: str = "cuda",
) -> np.ndarray:
    if intensity == 0:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    amount = float(np.clip(intensity, -1.0, 1.0))
    feather = float(np.clip(feather, 0.0, 1.0))
    roundness = float(np.clip(roundness, 0.0, 1.0))
    center_x = float(np.clip(center_x, -1.0, 1.0))
    center_y = float(np.clip(center_y, -1.0, 1.0))
    dither_amount = float(np.clip(dither_amount, 0.0, 1.0))

    distance, grid_x, grid_y = _vignette_geometry_torch(
        img.shape[-2],
        img.shape[-1],
        roundness=roundness,
        center_x=center_x,
        center_y=center_y,
        device=torch_device,
    )
    inner = max(0.0, 1.0 - 1.6 * feather)
    outer = 1.0 + 0.6 * feather
    mask = _smoothstep(distance, inner, outer)
    mask = _apply_vignette_dither_torch(mask, grid_x, grid_y, dither_amount)

    if amount > 0:
        gain = 1.0 - amount * mask
    else:
        gain = 1.0 + (-amount) * mask
    return _to_numpy_image((img * gain.unsqueeze(0).unsqueeze(0)).clamp(0.0, 1.0))


def adjust_sharpness_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    params = _resolve_sharpness_params(intensity)
    amount = float(params["amount_norm"])
    if abs(amount) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    luma = _rgb_luma_tensor(img)
    sigma = max(0.26, float(params["radius_px"]) * 0.45)
    blur = _gaussian_blur_image(luma.unsqueeze(1), sigma, border_type="reflect")[:, 0]
    detail = luma - blur
    threshold = float(params["threshold_pct"]) / 100.0
    detail = detail.sign() * torch.clamp(detail.abs() - threshold, min=0.0)
    detail_mask = _detail_activity_torch(
        detail,
        low=max(0.003, threshold * 0.5),
        high=max(0.020, threshold * 4.0 + 0.020),
    )
    edge_mask = _sharpness_edge_mask_torch(luma)
    tone_mask = _sharpness_tone_mask_torch(luma)
    halo_guard = _sharpness_halo_guard_torch(luma)
    mask = detail_mask * edge_mask * (0.35 + 0.65 * tone_mask) * (0.20 + 0.80 * halo_guard)
    limit = torch.maximum(
        torch.full_like(detail, 0.006 + threshold),
        detail.abs() * (0.32 + 0.28 * halo_guard),
    )
    delta = torch.clamp(amount * 1.8 * detail * mask, min=-limit, max=limit)

    out_luma = (luma + delta).clamp(0.0, 1.0)
    return _to_numpy_image(_luma_ratio_recompose(img, luma, out_luma))


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
        "fine_sigma_px": float(max(0.1, float(payload.get("fine_sigma_px", 0.7)))),
        "detail_sigma_px": float(max(0.1, float(payload.get("detail_sigma_px", 2.0)))),
        "coarse_sigma_px": float(max(0.1, float(payload.get("coarse_sigma_px", 6.0)))),
        "mask_low": float(max(1e-6, float(payload.get("mask_low", 0.001)))),
        "mask_high": float(max(1e-5, float(payload.get("mask_high", 0.020)))),
        "gain_scale": float(max(0.0, float(payload.get("gain_scale", 1.45)))),
        "tone_strength": float(np.clip(float(payload.get("tone_strength", 0.65)), 0.0, 1.0)),
    }


def adjust_texture_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    params = _resolve_texture_params(intensity)
    amount = float(params["texture_gain"])
    if abs(amount) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    luma = _rgb_luma_tensor(img)
    g1 = _gaussian_blur_image(luma.unsqueeze(1), float(params["fine_sigma_px"]), border_type="reflect")[:, 0]
    g2 = _gaussian_blur_image(luma.unsqueeze(1), float(params["detail_sigma_px"]), border_type="reflect")[:, 0]
    g3 = _gaussian_blur_image(luma.unsqueeze(1), float(params["coarse_sigma_px"]), border_type="reflect")[:, 0]
    band = 0.75 * (g1 - g2) + 0.35 * (g2 - g3)
    mask = _detail_activity_torch(
        band,
        low=float(params["mask_low"]),
        high=float(params["mask_high"]),
    ) * (0.35 + float(params["tone_strength"]) * _midtone_mask_torch(luma))
    out_luma = (luma + amount * float(params["gain_scale"]) * band * mask).clamp(0.0, 1.0)
    return _to_numpy_image(_luma_ratio_recompose(img, luma, out_luma))


def adjust_luminance_noise_reduction_torch(
    image: np.ndarray,
    intensity: float,
    device: str = "cuda",
) -> np.ndarray:
    if isinstance(intensity, dict):
        amount = float(intensity.get("luma_denoise_pct", 0.0)) / 100.0
    else:
        amount = float(np.clip(intensity, 0.0, 1.0))
    if amount <= 0:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    raw_luma = _rgb_luma_tensor(img)
    _, spike_mask = _luminance_nr_spike_mask_torch(raw_luma)
    work_img = _apply_rgb_spike_cleanup_torch(img, spike_mask)
    luma = _rgb_luma_tensor(work_img)
    preclean = luma

    noise_level = _local_variance_torch(preclean, sigma=1.0).sqrt()
    flat_mask = 1.0 - _smoothstep(noise_level, 0.01, 0.045)

    kernel = max(5, 5 + int(round(amount * 4.0)) * 2)
    sigma_color = 0.04 + amount * 0.03
    sigma_space = 2.0 + amount * 2.5
    smooth = _edge_preserving_luma_smooth_torch(
        preclean,
        kernel=kernel,
        sigma_color=sigma_color,
        sigma_space=sigma_space,
    )

    detail_ref = (preclean - smooth).abs()
    detail_mask = 1.0 - _smoothstep(detail_ref, 0.015, 0.06)
    variance_mask = 1.0 - _smoothstep(_local_variance_torch(preclean, sigma=1.2).sqrt(), 0.015, 0.05)
    shadow_weight = 1.0 - 0.25 * _smoothstep(preclean, 0.72, 0.95)
    blend = (
        amount
        * (0.15 + 0.85 * flat_mask)
        * (0.25 + 0.75 * detail_mask)
        * (0.3 + 0.7 * variance_mask)
        * shadow_weight
    ).clamp(0.0, 1.0)
    out_luma = (preclean * (1.0 - blend) + smooth * blend).clamp(0.0, 1.0)
    return _to_numpy_image(_luma_ratio_recompose(work_img, luma, out_luma))


def adjust_exposure_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    ev_delta, black_level = _resolve_exposure_params(intensity)
    if abs(ev_delta) < 1e-8 and abs(black_level) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    white = float(np.exp2(-ev_delta))
    gain = 1.0 / max(white - black_level, 1e-6)
    adjusted = ((img - black_level) * gain).clamp(min=0.0)
    pivot = 0.18
    gamma = 1.0 / (1.0 + 0.25 * ev_delta) if ev_delta >= 0.0 else 1.0 - 0.20 * ev_delta
    out = pivot * torch.pow((adjusted / max(pivot, 1e-6)).clamp(min=0.0), gamma)
    out = out.clamp(0.0, 1.0)
    return _to_numpy_image(out)


def adjust_contrast_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    contrast_factor, brightness = _resolve_contrast_params(intensity)
    if abs(contrast_factor - 1.0) < 1e-8 and abs(brightness) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    normalized = float(contrast_factor) - 1.0
    if normalized >= 0.0:
        slope = 2.0 + 6.0 * float(np.clip(normalized, 0.0, 1.0))
        raw = 1.0 / (1.0 + torch.exp(-slope * (img - 0.5)))
        lo = 1.0 / (1.0 + math.exp(-slope * (-0.5)))
        hi = 1.0 / (1.0 + math.exp(-slope * 0.5))
        out = (raw - lo) / max(hi - lo, 1e-6)
    else:
        flatten = max(0.05, 1.0 + float(np.clip(normalized, -0.95, 0.0)))
        out = (img - 0.5) * flatten + 0.5
    out = out + brightness
    out = out.clamp(0.0, 1.0)
    return _to_numpy_image(out)


def apply_tone_curve_torch(
    image: np.ndarray,
    curve_spec,
    device: str = "cuda",
) -> np.ndarray:
    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    spec = sanitize_tone_curve_spec(curve_spec)
    out = img.clone()

    master_lut = build_curve_lut(spec["master"])
    for c in range(3):
        out[:, c] = _apply_lut_channel_torch(out[:, c], master_lut)

    channel_map = {"r": 0, "g": 1, "b": 2}
    for key, idx in channel_map.items():
        if spec.get(key) is None:
            continue
        lut = build_curve_lut(spec[key])
        out[:, idx] = _apply_lut_channel_torch(out[:, idx], lut)

    return _to_numpy_image(out)


def adjust_blacks_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    if isinstance(intensity, dict):
        amount = _invert_piecewise_table(
            _LR2DT_BLACKS_TABLE,
            float(intensity.get("black_level_offset", 0.0)),
        ) / 100.0
    else:
        amount = float(np.clip(intensity, -1.0, 1.0))
    if abs(amount) < 1e-8:
        return np.asarray(image, dtype=np.float32)
    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    hue, sat, val = _rgb_to_hsv_unit(img)
    val = _apply_lut_channel_torch(val, build_blacks_lut(float(amount))).clamp(0.0, 1.0)
    return _to_numpy_image(_hsv_unit_to_rgb(hue, sat.clamp(0.0, 1.0), val))


def adjust_tint_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    tint_units, base_kelvin = _resolve_tint_payload(intensity)
    if abs(tint_units) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    del base_kelvin
    normalized = float(np.clip(tint_units / 150.0, -1.0, 1.0))
    coeffs = torch.as_tensor(
        [
            math.exp(0.18 * normalized),
            math.exp(-0.28 * normalized),
            math.exp(0.18 * normalized),
        ],
        device=torch_device,
        dtype=torch.float32,
    )
    luma = torch.as_tensor(_RGB_LUMA, device=torch_device, dtype=torch.float32)
    coeffs = coeffs / torch.dot(coeffs, luma).clamp(min=1e-6)
    out = (img * coeffs.view(1, 3, 1, 1)).clamp(0.0, 1.0)
    return _to_numpy_image(out)


def adjust_saturation_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    saturation_pct = _resolve_saturation_params(intensity)
    if abs(saturation_pct) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    scale = max(0.0, 1.0 + float(saturation_pct) / 100.0)
    lab = _rgb_to_lab_image(img)
    lab[:, 1] = (lab[:, 1] * scale).clamp(-128.0, 127.0)
    lab[:, 2] = (lab[:, 2] * scale).clamp(-128.0, 127.0)
    out = _lab_to_rgb_image(lab).clamp(0.0, 1.0)
    return _to_numpy_image(out)


def adjust_vibrance_torch(
    image: np.ndarray,
    intensity: float,
    sigma_s: float = 0.2,
    sigma_v: float = 0.2,
    device: str = "cuda",
) -> np.ndarray:
    del sigma_s, sigma_v
    vibrance_pct = _resolve_vibrance_params(intensity)
    if abs(vibrance_pct) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    out = _apply_vibrance_rt_torch(img, vibrance_pct, protect_skin=True)
    return _to_numpy_image(out)


def adjust_temperature_torch(image: np.ndarray, intensity: float, device: str = "cuda") -> np.ndarray:
    base_kelvin, delta_kelvin = _resolve_temperature_params(intensity)
    if abs(delta_kelvin) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    coeffs = _temperature_coeffs_torch(base_kelvin, delta_kelvin, torch_device)
    out = (img * coeffs.view(1, 3, 1, 1)).clamp(0.0, 1.0)
    return _to_numpy_image(out)


def _reflect_indices(length: int, pad: int, device: torch.device) -> torch.Tensor:
    if length <= 1:
        return torch.zeros(length + 2 * pad, dtype=torch.long, device=device)
    idx = torch.arange(-pad, length + pad, device=device, dtype=torch.long)
    period = 2 * length
    idx = torch.remainder(idx, period)
    idx = torch.where(idx >= length, period - idx - 1, idx)
    return idx


def _box_blur_image(tensor: torch.Tensor, kernel: int) -> torch.Tensor:
    kernel_h, kernel_w = _odd_kernel_size(kernel)
    pad_h = kernel_h // 2
    pad_w = kernel_w // 2
    idx_h = _reflect_indices(int(tensor.shape[-2]), pad_h, tensor.device)
    idx_w = _reflect_indices(int(tensor.shape[-1]), pad_w, tensor.device)
    padded = tensor.index_select(-2, idx_h).index_select(-1, idx_w)
    return torch.nn.functional.avg_pool2d(padded, kernel_size=(kernel_h, kernel_w), stride=1)


def _guided_filter_gray_torch(
    guide_gray: torch.Tensor,
    src: torch.Tensor,
    radius: float,
    eps: float,
    subsampling: int = 1,
) -> torch.Tensor:
    guide = guide_gray.unsqueeze(0).unsqueeze(0)
    target = src.unsqueeze(0).unsqueeze(0)
    if subsampling > 1:
        small_h = max(1, guide.shape[-2] // subsampling)
        small_w = max(1, guide.shape[-1] // subsampling)
        guide_small = torch.nn.functional.interpolate(guide, size=(small_h, small_w), mode="bilinear", align_corners=False)
        target_small = torch.nn.functional.interpolate(target, size=(small_h, small_w), mode="bilinear", align_corners=False)
        filtered_small = _guided_filter_gray_torch(
            guide_small[0, 0],
            target_small[0, 0],
            max(1.0, float(radius) / float(subsampling)),
            eps,
            subsampling=1,
        )
        return torch.nn.functional.interpolate(
            filtered_small.unsqueeze(0).unsqueeze(0),
            size=(guide.shape[-2], guide.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    r = max(1, int(round(radius)))
    kernel = r * 2 + 1
    mean_i = _box_blur_image(guide, kernel)
    mean_p = _box_blur_image(target, kernel)
    corr_i = _box_blur_image(guide * guide, kernel)
    corr_ip = _box_blur_image(guide * target, kernel)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = _box_blur_image(a, kernel)
    mean_b = _box_blur_image(b, kernel)
    return (mean_a * guide + mean_b)[0, 0]


def _shadow_contrast_curve_torch(values: torch.Tensor, contrast: float) -> torch.Tensor:
    knots_x = torch.tensor([0.0, 0.125, 0.25, 0.375, 1.0], device=values.device, dtype=values.dtype)
    knots_y = torch.tensor(
        [
            0.0,
            float(np.power(0.125 / 0.25, contrast) * 0.25),
            0.25,
            float(np.power(0.375 / 0.25, contrast) * 0.25),
            1.0,
        ],
        device=values.device,
        dtype=values.dtype,
    )
    pos = values.clamp(0.0, 1.0)
    idx = torch.bucketize(pos.reshape(-1), knots_x)
    idx = idx.clamp(1, knots_x.numel() - 1)
    x0 = knots_x[idx - 1]
    x1 = knots_x[idx]
    y0 = knots_y[idx - 1]
    y1 = knots_y[idx]
    frac = ((pos.reshape(-1) - x0) / (x1 - x0 + 1e-8)).clamp(0.0, 1.0)
    out = y0 * (1.0 - frac) + y1 * frac
    return out.reshape_as(values)


def _apply_gegl_tone_core_torch(
    img: torch.Tensor,
    *,
    highlights: float,
    shadows: float,
    whites: float,
    base: torch.Tensor,
    compress: float,
    shadows_ccorrect: float,
    highlights_ccorrect: float,
    shadows_response_scale: float,
    highlights_response_scale: float,
) -> torch.Tensor:
    r = img[:, 0]
    g = img[:, 1]
    b = img[:, 2]
    light = 0.299 * r + 0.587 * g + 0.114 * b
    tb0 = (1.0 - base).clamp(0.0, 1.0)

    c = float(np.clip(compress, 0.0, 0.99))
    eps = 1e-8
    ta = light.clone()

    h_scaled = 2.0 * float(np.clip(highlights, -1.0, 1.0)) * float(np.clip(highlights_response_scale, 0.0, 3.0))
    if h_scaled != 0.0:
        highlights_xform = (1.0 - tb0 / (1.0 - c + eps)).clamp(0.0, 1.0)
        highlights2 = h_scaled * h_scaled
        sign_neg = -1.0 if h_scaled > 0.0 else 1.0
        while highlights2 > 0.0:
            la = ta
            lb = ((tb0 - 0.5) * sign_neg * torch.sign(1.0 - la) + 0.5).clamp(0.0, 1.0)
            chunk = 1.0 if highlights2 > 1.0 else highlights2
            optrans = chunk * highlights_xform
            highlights2 -= 1.0
            mapped = torch.where(
                la > 0.5,
                1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb),
                2.0 * la * lb,
            )
            ta = (la * (1.0 - optrans) + mapped * optrans).clamp(0.0, 1.0)

    s_scaled = 2.0 * float(np.clip(shadows, -1.0, 1.0)) * float(np.clip(shadows_response_scale, 0.0, 3.0))
    if s_scaled != 0.0:
        shadows_xform = (tb0 / (1.0 - c + eps) - c / (1.0 - c + eps)).clamp(0.0, 1.0)
        shadows2 = s_scaled * s_scaled
        sign_pos = 1.0 if s_scaled > 0.0 else -1.0
        while shadows2 > 0.0:
            la = ta
            lb = ((tb0 - 0.5) * sign_pos * torch.sign(1.0 - la) + 0.5).clamp(0.0, 1.0)
            chunk = 1.0 if shadows2 > 1.0 else shadows2
            optrans = chunk * shadows_xform
            shadows2 -= 1.0
            mapped = torch.where(
                la > 0.5,
                1.0 - (1.0 - 2.0 * (la - 0.5)) * (1.0 - lb),
                2.0 * la * lb,
            )
            ta = (la * (1.0 - optrans) + mapped * optrans).clamp(0.0, 1.0)

    if whites != 0.0:
        whites_mask = _smoothstep(ta, 0.72, 0.98)
        ta = (ta + (0.35 * float(np.clip(whites, -1.0, 1.0))) * whites_mask).clamp(0.0, 1.0)

    ratio = ta / (light + 1e-8)
    out = torch.stack([r * ratio, g * ratio, b * ratio], dim=1).clamp(0.0, 1.0)

    if h_scaled != 0.0 or s_scaled != 0.0:
        hue, sat, val = _rgb_to_hsv_unit(out)
        sh_mask = (tb0 / (1.0 - c + eps) - c / (1.0 - c + eps)).clamp(0.0, 1.0)
        hi_mask = (1.0 - tb0 / (1.0 - c + eps)).clamp(0.0, 1.0)
        sh_scale = (float(np.clip(shadows_ccorrect, 0.0, 1.0)) - 0.5) * abs(s_scaled) * 0.7
        hi_scale = (float(np.clip(highlights_ccorrect, 0.0, 1.0)) - 0.5) * abs(h_scaled) * 0.7
        sat = (sat * (1.0 + sh_scale * sh_mask + hi_scale * hi_mask)).clamp(0.0, 1.0)
        out = _hsv_unit_to_rgb(hue, sat, val.clamp(0.0, 1.0))

    return out.clamp(0.0, 1.0)


def _adjust_shadows_rt_guided_torch(
    img: torch.Tensor,
    shadows: float,
    *,
    radius_px: float,
    tonal_width_pct: float,
) -> torch.Tensor:
    if shadows <= 0.0:
        return img.clone()

    light = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    thresh = float(np.clip(tonal_width_pct / 100.0, 1e-4, 1.0))
    scale = thresh * 0.9
    mask = torch.where(
        light <= thresh,
        torch.ones_like(light),
        torch.clamp(scale / light.clamp_min(1e-4), 0.0, 1.0).pow(4.0),
    )
    refined_mask = _guided_filter_gray_torch(light[0], mask[0], radius_px * 10.0, 0.075, subsampling=4).clamp(0.0, 1.0)
    amount_pct = float(np.clip(shadows * 100.0, 0.0, 100.0)) * 0.6
    base = float(np.power(4.0, amount_pct / 100.0))
    gamma = 1.0 / base
    mapped = img.clamp(0.0, 1.0).pow(gamma)
    contrast = float(np.power(2.0, amount_pct / 100.0))
    mapped = _shadow_contrast_curve_torch(mapped, contrast)
    blend = refined_mask.unsqueeze(0).unsqueeze(0)
    return (img * (1.0 - blend) + mapped * blend).clamp(0.0, 1.0)


def adjust_tones_torch(
    image: np.ndarray,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    model: str = "selected",
    radius: float = 100.0,
    compress: float = 0.5,
    shadows_ccorrect: float = 1.0,
    highlights_ccorrect: float = 0.5,
    shadows_gain: float = 1.8,
    highlights_gain: float = 1.0,
    shadows_response_scale: float = 0.45,
    highlights_response_scale: float = 0.8,
    shadow_tonal_width_pct: float = 30.0,
    whites_gain: float = 0.9,
    device: str = "cuda",
) -> np.ndarray:
    highlights, shadows, whites, resolved = _resolve_tone_execution_payloads(
        highlights,
        shadows,
        whites,
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
    if abs(float(highlights)) < 1e-8 and abs(float(shadows)) < 1e-8 and abs(float(whites)) < 1e-8:
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    img = _to_tensor_image(image, torch_device)
    resolved_model = str(model).strip().lower()

    if resolved_model == "selected":
        out = img.clone()
        if abs(float(highlights)) > 1e-8:
            light = 0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]
            kernel = max(5, int(round(float(resolved["radius"]) * 0.3)) * 2 + 1)
            base = _edge_preserving_luma_smooth_torch(
                light,
                kernel=kernel,
                sigma_color=0.08,
                sigma_space=max(2.0, float(resolved["radius"]) * 0.12),
                downsample=4,
            )
            out = _apply_gegl_tone_core_torch(
                out,
                highlights=float(highlights),
                shadows=0.0,
                whites=0.0,
                base=base,
                compress=resolved["compress"],
                shadows_ccorrect=resolved["shadows_ccorrect"],
                highlights_ccorrect=resolved["highlights_ccorrect"],
                shadows_response_scale=1.0,
                highlights_response_scale=1.0,
            )
        if abs(float(shadows)) > 1e-8:
            shadow_radius = float(resolved["radius"]) if bool(resolved.get("shadows_from_dict")) else SELECTED_TONE_RT_SHADOW_RADIUS
            if float(shadows) > 0.0:
                out = _adjust_shadows_rt_guided_torch(
                    out,
                    float(shadows),
                    radius_px=shadow_radius,
                    tonal_width_pct=float(resolved.get("shadow_tonal_width_pct", SELECTED_TONE_RT_SHADOW_TONAL_WIDTH)),
                )
            else:
                light = 0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]
                sigma = max(0.5, shadow_radius * 0.05)
                base = _gaussian_blur_image(light.unsqueeze(1), sigma, border_type="reflect")[:, 0]
                out = _apply_gegl_tone_core_torch(
                    out,
                    highlights=0.0,
                    shadows=float(shadows),
                    whites=0.0,
                    base=base,
                    compress=resolved["compress"],
                    shadows_ccorrect=resolved["shadows_ccorrect"],
                    highlights_ccorrect=resolved["highlights_ccorrect"],
                    shadows_response_scale=1.0,
                    highlights_response_scale=1.0,
                )
        if abs(float(whites)) > 1e-8:
            light = 0.299 * out[:, 0] + 0.587 * out[:, 1] + 0.114 * out[:, 2]
            sigma = max(0.5, float(resolved["radius"]) * 0.05)
            base = _gaussian_blur_image(light.unsqueeze(1), sigma, border_type="reflect")[:, 0]
            out = _apply_gegl_tone_core_torch(
                out,
                highlights=0.0,
                shadows=0.0,
                whites=float(whites) * float(resolved["whites_gain"]),
                base=base,
                compress=resolved["compress"],
                shadows_ccorrect=resolved["shadows_ccorrect"],
                highlights_ccorrect=resolved["highlights_ccorrect"],
                shadows_response_scale=1.0,
                highlights_response_scale=1.0,
            )
        return _to_numpy_image(out)

    h_val = float(highlights) * float(resolved["highlights_gain"])
    s_val = float(shadows) * float(resolved["shadows_gain"])
    w_val = float(whites) * float(resolved["whites_gain"])
    light = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    sigma = max(0.5, float(resolved["radius"]) * 0.05)
    base = _gaussian_blur_image(light.unsqueeze(1), sigma, border_type="reflect")[:, 0]
    out = _apply_gegl_tone_core_torch(
        img,
        highlights=h_val,
        shadows=s_val,
        whites=w_val,
        base=base,
        compress=resolved["compress"],
        shadows_ccorrect=resolved["shadows_ccorrect"],
        highlights_ccorrect=resolved["highlights_ccorrect"],
        shadows_response_scale=resolved["shadows_response_scale"],
        highlights_response_scale=resolved["highlights_response_scale"],
    )
    return _to_numpy_image(out)


def _resolve_gimp_hue_ranges_torch(
    hue_values: torch.Tensor,
    overlap: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hue_values = torch.remainder(hue_values, 1.0)
    hue_six = hue_values * 6.0
    overlap = float(np.clip(overlap, 0.0, 1.0))
    overlap_half = overlap * 0.5

    primary_hue = torch.zeros_like(hue_values, dtype=torch.long)
    secondary_hue = torch.zeros_like(hue_values, dtype=torch.long)
    use_secondary = torch.zeros_like(hue_values, dtype=torch.bool)
    primary_intensity = torch.ones_like(hue_values, dtype=torch.float32)
    secondary_intensity = torch.zeros_like(hue_values, dtype=torch.float32)
    assigned = torch.zeros_like(hue_values, dtype=torch.bool)

    for hue_counter in range(7):
        hue_threshold = float(hue_counter) + 0.5
        mask = (~assigned) & (hue_six < (hue_threshold + overlap_half))
        if not torch.any(mask):
            continue
        primary_hue[mask] = hue_counter

        if overlap_half > 0.0:
            sec_mask = mask & (hue_six > (hue_threshold - overlap_half))
            if torch.any(sec_mask):
                use_secondary[sec_mask] = True
                secondary_hue[sec_mask] = hue_counter + 1
                sec = (hue_six[sec_mask] - hue_threshold + overlap_half) / (2.0 * overlap_half)
                sec = sec.clamp(0.0, 1.0).float()
                secondary_intensity[sec_mask] = sec
                primary_intensity[sec_mask] = 1.0 - sec

        assigned[mask] = True

    wrap_primary = primary_hue >= 6
    if torch.any(wrap_primary):
        primary_hue[wrap_primary] = 0
        secondary_hue[wrap_primary] = 0

    wrap_secondary = secondary_hue >= 6
    if torch.any(wrap_secondary):
        secondary_hue[wrap_secondary] = 0

    return primary_hue + 1, secondary_hue + 1, use_secondary, primary_intensity, secondary_intensity


def _gimp_map_hue(values: torch.Tensor, hue_all: torch.Tensor, hue_range: torch.Tensor) -> torch.Tensor:
    return torch.remainder(values + (hue_all + hue_range) * 0.5, 1.0)


def _gimp_map_hue_overlap(
    values: torch.Tensor,
    hue_all: torch.Tensor,
    primary_hue_adj: torch.Tensor,
    secondary_hue_adj: torch.Tensor,
    primary_intensity: torch.Tensor,
    secondary_intensity: torch.Tensor,
) -> torch.Tensor:
    blended = primary_hue_adj * primary_intensity + secondary_hue_adj * secondary_intensity
    return torch.remainder(values + (hue_all + blended) * 0.5, 1.0)


def _gimp_map_saturation(values: torch.Tensor, sat_all: torch.Tensor, sat_range: torch.Tensor) -> torch.Tensor:
    return (values * (sat_all + sat_range + 1.0)).clamp(0.0, 1.0)


def _gimp_map_lightness(values: torch.Tensor, light_all: torch.Tensor, light_range: torch.Tensor) -> torch.Tensor:
    v = light_all + light_range
    return torch.where(v < 0.0, values * (v + 1.0), values + (v * (1.0 - values))).clamp(0.0, 1.0)


def _gimp_map_lightness_achromatic(values: torch.Tensor, light_all: torch.Tensor) -> torch.Tensor:
    return torch.where(light_all < 0.0, values * (light_all + 1.0), values + (light_all * (1.0 - values))).clamp(0.0, 1.0)


def apply_gimp_hsl_transform_torch(
    image: np.ndarray,
    hue_adj: np.ndarray,
    sat_adj: np.ndarray,
    light_adj: np.ndarray,
    overlap: float,
    stability_cfg: dict | None = None,
    device: str = "cuda",
) -> np.ndarray:
    if (
        np.max(np.abs(hue_adj)) < 1e-8
        and np.max(np.abs(sat_adj)) < 1e-8
        and np.max(np.abs(light_adj)) < 1e-8
    ):
        return np.asarray(image, dtype=np.float32)

    torch_device = ensure_torch_cuda_runtime(device)
    rgb01 = _to_tensor_image(image, torch_device)

    hue, light, sat = _rgb_to_hls_unit(rgb01)
    hue = hue[0]
    light = light[0]
    sat = sat[0]
    chroma = (torch.amax(rgb01, dim=1) - torch.amin(rgb01, dim=1))[0]

    (
        primary_range,
        secondary_range,
        use_secondary,
        primary_intensity,
        secondary_intensity,
    ) = _resolve_gimp_hue_ranges_torch(hue, overlap)

    hue_adj_t = torch.as_tensor(hue_adj, device=torch_device, dtype=torch.float32)
    sat_adj_t = torch.as_tensor(sat_adj, device=torch_device, dtype=torch.float32)
    light_adj_t = torch.as_tensor(light_adj, device=torch_device, dtype=torch.float32)

    hue_out = hue.clone()
    sat_out = sat.clone()
    light_out = light.clone()

    secondary_mask = use_secondary
    if torch.any(secondary_mask):
        primary_idx = primary_range[secondary_mask]
        secondary_idx = secondary_range[secondary_mask]
        p_weight = primary_intensity[secondary_mask]
        s_weight = secondary_intensity[secondary_mask]

        hue_out[secondary_mask] = _gimp_map_hue_overlap(
            hue[secondary_mask],
            hue_adj_t[0],
            hue_adj_t[primary_idx],
            hue_adj_t[secondary_idx],
            p_weight,
            s_weight,
        )

        sat_primary = _gimp_map_saturation(sat[secondary_mask], sat_adj_t[0], sat_adj_t[primary_idx])
        sat_secondary = _gimp_map_saturation(sat[secondary_mask], sat_adj_t[0], sat_adj_t[secondary_idx])
        sat_out[secondary_mask] = sat_primary * p_weight + sat_secondary * s_weight

        light_primary = _gimp_map_lightness(
            light[secondary_mask], light_adj_t[0], light_adj_t[primary_idx]
        )
        light_secondary = _gimp_map_lightness(
            light[secondary_mask], light_adj_t[0], light_adj_t[secondary_idx]
        )
        light_out[secondary_mask] = light_primary * p_weight + light_secondary * s_weight

    no_secondary_mask = ~secondary_mask
    achromatic_mask = no_secondary_mask & (sat <= _ACHROMATIC_THRESHOLD)
    if torch.any(achromatic_mask):
        light_out[achromatic_mask] = _gimp_map_lightness_achromatic(light[achromatic_mask], light_adj_t[0])

    chroma_mask = no_secondary_mask & (sat > _ACHROMATIC_THRESHOLD)
    if torch.any(chroma_mask):
        primary_idx = primary_range[chroma_mask]
        hue_out[chroma_mask] = _gimp_map_hue(hue[chroma_mask], hue_adj_t[0], hue_adj_t[primary_idx])
        light_out[chroma_mask] = _gimp_map_lightness(
            light[chroma_mask], light_adj_t[0], light_adj_t[primary_idx]
        )
        sat_out[chroma_mask] = _gimp_map_saturation(sat[chroma_mask], sat_adj_t[0], sat_adj_t[primary_idx])

    if isinstance(stability_cfg, dict) and stability_cfg.get("enabled", True):
        sat_conf = _smoothstep(
            sat,
            stability_cfg.get("sat_floor_low", 0.02),
            stability_cfg.get("sat_floor_high", 0.12),
        )
        chroma_conf = _smoothstep(
            chroma,
            stability_cfg.get("chroma_floor_low", 0.006),
            stability_cfg.get("chroma_floor_high", 0.040),
        )
        conf = sat_conf * chroma_conf

        hue_delta = torch.remainder(hue_out - hue + 0.5, 1.0) - 0.5
        hue_out = torch.remainder(hue + hue_delta * conf, 1.0)
        sat_out = (sat + (sat_out - sat) * conf).clamp(0.0, 1.0)

        luma_cfg = stability_cfg.get("luma_guard", {})
        if isinstance(luma_cfg, dict) and luma_cfg.get("enabled", True):
            luma_sat_conf = _smoothstep(
                sat,
                luma_cfg.get("sat_floor_low", 0.06),
                luma_cfg.get("sat_floor_high", 0.22),
            )
            luma_chroma_conf = _smoothstep(
                chroma,
                luma_cfg.get("chroma_floor_low", stability_cfg.get("chroma_floor_low", 0.006)),
                luma_cfg.get("chroma_floor_high", stability_cfg.get("chroma_floor_high", 0.040)),
            )
            luma_conf = luma_sat_conf * luma_chroma_conf
            delta_light = (light_out - light) * luma_conf * float(np.clip(luma_cfg.get("strength", 1.0), 0.0, 1.0))

            guided_cfg = luma_cfg.get("guided", {})
            guide_gray = 0.299 * rgb01[0, 0] + 0.587 * rgb01[0, 1] + 0.114 * rgb01[0, 2]
            delta_light = _edge_aware_smooth_delta(delta_light, guide_gray, guided_cfg)

            speckle_cfg = luma_cfg.get("speckle", {})
            if isinstance(speckle_cfg, dict) and speckle_cfg.get("enabled", True):
                low_mask = sat <= float(speckle_cfg.get("low_sat_threshold", 0.22))
                low_mask = low_mask | (chroma <= float(speckle_cfg.get("low_chroma_threshold", 0.06)))
                spike_mask = low_mask & (torch.abs(delta_light) >= float(speckle_cfg.get("delta_thresh", 0.06)))
                if torch.any(spike_mask):
                    kernel = max(1, int(speckle_cfg.get("kernel", 3)))
                    if kernel % 2 == 0:
                        kernel += 1
                    blend = float(np.clip(speckle_cfg.get("blend", 0.85), 0.0, 1.0))
                    delta_light = _median_blend_where(
                        delta_light.unsqueeze(0).unsqueeze(0),
                        spike_mask,
                        kernel,
                        blend,
                    )[0, 0]

            light_out = (light + delta_light).clamp(0.0, 1.0)

    rgb_out = _hls_unit_to_rgb(hue_out.unsqueeze(0), light_out.unsqueeze(0), sat_out.unsqueeze(0))

    if isinstance(stability_cfg, dict) and stability_cfg.get("enabled", True):
        artifact_cfg = stability_cfg.get("artifact_guard", {})
        if isinstance(artifact_cfg, dict) and artifact_cfg.get("enabled", True):
            diff = torch.mean(torch.abs(rgb_out - rgb01), dim=1)[0]
            low_sat_mask = sat <= float(artifact_cfg.get("low_sat_threshold", stability_cfg.get("sat_floor_high", 0.12)))
            low_chroma_mask = chroma <= float(
                artifact_cfg.get("low_chroma_threshold", stability_cfg.get("chroma_floor_high", 0.040))
            )
            spike_mask = (low_sat_mask | low_chroma_mask) & (diff >= float(artifact_cfg.get("diff_thresh", 0.10)))
            if torch.any(spike_mask):
                kernel = max(1, int(artifact_cfg.get("kernel", 3)))
                if kernel % 2 == 0:
                    kernel += 1
                blend = float(np.clip(artifact_cfg.get("blend", 0.85), 0.0, 1.0))
                rgb_out = _median_blend_where(rgb_out, spike_mask, kernel, blend)

    return _to_numpy_image(rgb_out)
