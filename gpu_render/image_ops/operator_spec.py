from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class OperatorSpec:
    op_name: str
    reference_backend: str
    reference_source_path: str
    reference_formula_id: str
    canonical_param: str
    canonical_unit: str
    canonical_range: tuple[float, float]
    neutral_value: float
    roi_safe: bool
    color_space: str
    default_aux_params: dict[str, Any] = field(default_factory=dict)
    primary_execution_path: str = "gpu_canonical"
    cpu_role: str = "reference_oracle"
    behavior_profile_id: str = ""
    public_slider_mapping_id: str = ""
    acceptance_suite_id: str = ""


OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "Exposure": OperatorSpec(
        op_name="Exposure",
        reference_backend="gegl+darktable",
        reference_source_path="gegl/operations/common/exposure.c; darktable/src/iop/exposure.c",
        reference_formula_id="scene_linear_ev",
        canonical_param="ev_delta",
        canonical_unit="stops",
        canonical_range=(-5.0, 5.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="scene_linear_rgb",
        default_aux_params={"black_level": 0.0},
        behavior_profile_id="exposure_midtone_gamma_experimental",
        public_slider_mapping_id="slider_exposure_ev_v1",
        acceptance_suite_id="acceptance_point_numeric_parity_v1",
    ),
    "Contrast": OperatorSpec(
        op_name="Contrast",
        reference_backend="gegl",
        reference_source_path="gegl/operations/common/brightness-contrast.c",
        reference_formula_id="brightness_contrast_midgray",
        canonical_param="contrast_factor",
        canonical_unit="scale",
        canonical_range=(0.0, 2.0),
        neutral_value=1.0,
        roi_safe=True,
        color_space="linear_rgb",
        default_aux_params={"brightness": 0.0},
        behavior_profile_id="contrast_sigmoid_experimental",
        public_slider_mapping_id="slider_contrast_factor_v1",
        acceptance_suite_id="acceptance_point_numeric_parity_v1",
    ),
    "Highlights": OperatorSpec(
        op_name="Highlights",
        reference_backend="darktable+gegl",
        reference_source_path="darktable/src/iop/shadhi.c; gegl/operations/common-gpl3+/shadows-highlights.c",
        reference_formula_id="shadhi_highlights_mask_xform",
        canonical_param="highlights_pct",
        canonical_unit="percent",
        canonical_range=(-100.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="luminance_mask",
        default_aux_params={
            "radius_px": 100.0,
            "compress_pct": 50.0,
            "highlights_ccorrect_pct": 50.0,
        },
        behavior_profile_id="tone_gegl_bilateral_selected",
        public_slider_mapping_id="slider_highlights_pct_v1",
        acceptance_suite_id="acceptance_tone_sweep_v1",
    ),
    "Shadows": OperatorSpec(
        op_name="Shadows",
        reference_backend="RawTherapee+GEGL",
        reference_source_path="RawTherapee/rtengine/ipshadowshighlights.cc; darktable/src/iop/shadhi.c; gegl/operations/common-gpl3+/shadows-highlights.c",
        reference_formula_id="hybrid_signed_rt_shadow_family",
        canonical_param="shadows_pct",
        canonical_unit="percent",
        canonical_range=(-100.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="luminance_mask",
        default_aux_params={
            "radius_px": 40.0,
            "compress_pct": 50.0,
            "shadows_ccorrect_pct": 100.0,
            "shadow_tonal_width_pct": 30.0,
        },
        behavior_profile_id="tone_hybrid_signed_rt_selected",
        public_slider_mapping_id="slider_shadows_pct_v1",
        acceptance_suite_id="acceptance_tone_sweep_v1",
    ),
    "Whites": OperatorSpec(
        op_name="Whites",
        reference_backend="darktable",
        reference_source_path="darktable/src/iop/toneequal.c; darktable/src/iop/shadhi.c",
        reference_formula_id="toneequal_whites_band",
        canonical_param="whites_pct",
        canonical_unit="percent",
        canonical_range=(-100.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="ev_band_mask",
        default_aux_params={"band_ev_center": -1.0, "band_width": "fixed_profile"},
        behavior_profile_id="whites_gegl_band_selected",
        public_slider_mapping_id="slider_whites_pct_v1",
        acceptance_suite_id="acceptance_tone_sweep_v1",
    ),
    "Blacks": OperatorSpec(
        op_name="Blacks",
        reference_backend="darktable+gegl",
        reference_source_path="darktable/src/develop/lightroom.c; gegl/operations/common/exposure.c",
        reference_formula_id="lr2dt_blacks",
        canonical_param="black_level_offset",
        canonical_unit="linear_offset",
        canonical_range=(-0.01, 0.02),
        neutral_value=0.0,
        roi_safe=True,
        color_space="linear_rgb",
        default_aux_params={},
        behavior_profile_id="blacks_curve_calibrated_selected",
        public_slider_mapping_id="slider_blacks_table_v1",
        acceptance_suite_id="acceptance_tone_sweep_v1",
    ),
    "Temperature": OperatorSpec(
        op_name="Temperature",
        reference_backend="gegl",
        reference_source_path="gegl/operations/common/color-temperature.c; darktable/src/iop/temperature.c",
        reference_formula_id="kelvin_rgb_ratio",
        canonical_param="delta_kelvin",
        canonical_unit="kelvin_delta",
        canonical_range=(-5000.0, 5000.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="linear_rgb",
        default_aux_params={"base_kelvin": 6500.0},
        behavior_profile_id="temperature_rb_balance_experimental",
        public_slider_mapping_id="slider_temperature_kelvin_v1",
        acceptance_suite_id="acceptance_point_numeric_parity_v1",
    ),
    "Tint": OperatorSpec(
        op_name="Tint",
        reference_backend="rawtherapee+darktable",
        reference_source_path="RawTherapee/rtengine/colortemp.cc; darktable/src/iop/temperature.c",
        reference_formula_id="temp_green_white_balance_axis",
        canonical_param="tint_units",
        canonical_unit="tint_units",
        canonical_range=(-150.0, 150.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="working_rgb",
        default_aux_params={"base_kelvin": 6504.0},
        behavior_profile_id="tint_rgb_split_experimental",
        public_slider_mapping_id="slider_tint_units_v1",
        acceptance_suite_id="acceptance_color_response_v1",
    ),
    "Saturation": OperatorSpec(
        op_name="Saturation",
        reference_backend="gegl+darktable",
        reference_source_path="gegl/operations/common/saturation.c; darktable saturation semantics",
        reference_formula_id="saturation_pct",
        canonical_param="saturation_pct",
        canonical_unit="percent",
        canonical_range=(-100.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="working_rgb",
        default_aux_params={"colorspace": "native"},
        behavior_profile_id="saturation_lab_chroma_experimental",
        public_slider_mapping_id="slider_saturation_pct_v1",
        acceptance_suite_id="acceptance_color_response_v1",
    ),
    "Vibrance": OperatorSpec(
        op_name="Vibrance",
        reference_backend="rawtherapee+darktable",
        reference_source_path="RawTherapee/rtgui/tools/vibrance.cc; darktable/src/iop/vibrance.c",
        reference_formula_id="vibrance_rt_skin_protect_experimental",
        canonical_param="vibrance_pct",
        canonical_unit="percent",
        canonical_range=(-100.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="working_rgb",
        default_aux_params={},
        behavior_profile_id="vibrance_rt_skin_protect_experimental",
        public_slider_mapping_id="slider_vibrance_pct_capped60_v1",
        acceptance_suite_id="acceptance_color_response_v1",
    ),
    "Clarity": OperatorSpec(
        op_name="Clarity",
        reference_backend="darktable",
        reference_source_path="darktable/data/kernels/locallaplacian.cl; darktable/src/develop/lightroom.c",
        reference_formula_id="lr2dt_clarity",
        canonical_param="clarity_detail",
        canonical_unit="detail_gain",
        canonical_range=(-0.65, 0.65),
        neutral_value=0.0,
        roi_safe=True,
        color_space="local_laplacian_luma",
        default_aux_params={
            "profile": "local_laplacian_selected",
            "midtone_range": 0.5,
            "guide_radius_px": 18.0,
            "guide_sigma_color": 0.04,
            "clip_limit_scale": 0.9,
        },
        behavior_profile_id="clarity_local_laplacian_selected",
        public_slider_mapping_id="slider_clarity_lr_v1",
        acceptance_suite_id="acceptance_clarity_artifact_v1",
    ),
    "Texture": OperatorSpec(
        op_name="Texture",
        reference_backend="darktable+rawtherapee",
        reference_source_path="darktable/src/iop/diffuse.c; RawTherapee/rtgui/tools/localcontrast.cc",
        reference_formula_id="midband_texture_gain",
        canonical_param="texture_gain",
        canonical_unit="gain",
        canonical_range=(-1.0, 1.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="mid_frequency_detail",
        default_aux_params={"profile": "darktable_diffuse_fine_selected"},
        behavior_profile_id="texture_darktable_diffuse_fine_selected",
        public_slider_mapping_id="slider_texture_gain_v1",
        acceptance_suite_id="acceptance_detail_artifact_v1",
    ),
    "Sharpness": OperatorSpec(
        op_name="Sharpness",
        reference_backend="rawtherapee+darktable",
        reference_source_path="RawTherapee/rtengine/ipsharpen.cc; darktable/src/iop/sharpen.c",
        reference_formula_id="usm_amount_threshold",
        canonical_param="sharpness_amount",
        canonical_unit="amount",
        canonical_range=(0.0, 2.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="luminance_detail",
        default_aux_params={"profile": "rt_usm_selected", "radius_px": 2.0, "threshold_pct": 0.5},
        behavior_profile_id="sharp_rt_usm_selected",
        public_slider_mapping_id="slider_sharpness_amount_v1",
        acceptance_suite_id="acceptance_detail_artifact_v1",
    ),
}


COMPATIBILITY_ONLY_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    "Vignette": OperatorSpec(
        op_name="Vignette",
        reference_backend="darktable",
        reference_source_path="darktable/src/iop/vignette.c; darktable/src/develop/lightroom.c",
        reference_formula_id="lightroom_compatible_brightness",
        canonical_param="brightness",
        canonical_unit="brightness_gain",
        canonical_range=(-1.0, 1.0),
        neutral_value=0.0,
        roi_safe=False,
        color_space="image_plane",
        default_aux_params={
            "scale": 80.0,
            "falloff_scale": 50.0,
            "shape": 1.0,
            "center_x": 0.0,
            "center_y": 0.0,
            "saturation": 0.0,
        },
        behavior_profile_id="vignette_compatibility_default",
        public_slider_mapping_id="slider_vignette_compatibility_v1",
        acceptance_suite_id="acceptance_compatibility_only_v1",
    ),
    "LuminanceNoiseReduction": OperatorSpec(
        op_name="LuminanceNoiseReduction",
        reference_backend="darktable",
        reference_source_path="darktable/src/iop/denoiseprofile.c; darktable/src/iop/diffuse.c",
        reference_formula_id="non_negative_luma_denoise",
        canonical_param="luma_denoise_pct",
        canonical_unit="percent",
        canonical_range=(0.0, 100.0),
        neutral_value=0.0,
        roi_safe=True,
        color_space="luminance",
        default_aux_params={"profile_preset": "lite"},
        behavior_profile_id="luma_denoise_compatibility_default",
        public_slider_mapping_id="slider_luma_denoise_compatibility_v1",
        acceptance_suite_id="acceptance_compatibility_only_v1",
    ),
}


ALL_OPERATOR_SPECS: dict[str, OperatorSpec] = {
    **OPERATOR_SPECS,
    **COMPATIBILITY_ONLY_OPERATOR_SPECS,
}


def has_operator_spec(op_name: str) -> bool:
    return str(op_name) in ALL_OPERATOR_SPECS


def get_operator_spec(op_name: str) -> OperatorSpec:
    try:
        return ALL_OPERATOR_SPECS[str(op_name)]
    except KeyError as exc:
        raise KeyError(f"Unknown operator spec: {op_name}") from exc
