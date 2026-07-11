from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import torch

from gpu_render.gpu.local_preset import (
    composite_srgb,
    raster_cgt_batch,
    render_local_preset_tensor,
)


def test_composite_srgb_has_bit_exact_endpoints_for_all_alpha_abis():
    base = torch.rand(2, 3, 5, 7, dtype=torch.float32)
    edited = torch.rand_like(base)

    assert torch.equal(composite_srgb(base, edited, torch.zeros(5, 7)), base)
    assert torch.equal(composite_srgb(base, edited, torch.ones(2, 5, 7)), edited)
    assert torch.equal(
        composite_srgb(base, edited, torch.ones(2, 1, 5, 7)), edited
    )


def test_composite_srgb_expands_one_image_for_multiple_alphas():
    base = torch.full((1, 3, 2, 3), 0.2)
    edited = torch.full_like(base, 0.8)
    alpha = torch.stack([
        torch.zeros(2, 3),
        torch.ones(2, 3),
        torch.full((2, 3), 0.25),
    ])

    out = composite_srgb(base, edited, alpha)

    assert out.shape == (3, 3, 2, 3)
    assert torch.equal(out[0], base[0])
    assert torch.equal(out[1], edited[0])
    assert torch.allclose(out[2], torch.full_like(out[2], 0.35))


def test_raster_cgt_batch_delegates_canonical_specs():
    radial = {
        "mask_type": "circulargradient",
        "geom": {
            "Top": 0.2, "Bottom": 0.8, "Left": 0.25, "Right": 0.75,
            "Angle": 13.0, "Feather": 60.0, "Flipped": "true",
        },
    }
    linear_geom = {
        "ZeroX": 0.2, "ZeroY": 0.5, "FullX": 0.8, "FullY": 0.5,
        "Flipped": "false",
    }
    linear = {"mask_type": "gradient", "geom": linear_geom, "amount": 0.5}

    got = raster_cgt_batch([radial, linear], 11, 17, "cpu")

    from gpu_render.gpu.local_gpu import raster_alpha_t

    expected = torch.stack([
        raster_alpha_t(
            "circulargradient", radial["geom"], 11, 17, "cpu", smoothstep=True),
        raster_alpha_t(
            "gradient", linear_geom, 11, 17, "cpu", smoothstep=True) * 0.5,
    ]).unsqueeze(1)
    assert got.shape == (2, 1, 11, 17)
    assert torch.equal(got, expected)
    assert float(got.min()) >= 0.0
    assert float(got.max()) <= 1.0


def test_legacy_rasters_stay_linear_while_local_preset_uses_smoothstep():
    from gpu_render.gpu.local_gpu import raster_alpha_t
    from gpu_render.local_replay import raster_alpha

    geom = {
        "ZeroX": 0.0, "ZeroY": 0.0, "FullX": 1.0, "FullY": 0.0,
        "Flipped": "false",
    }
    legacy_cpu = torch.from_numpy(raster_alpha("gradient", geom, 1, 4))
    legacy_gpu = raster_alpha_t("gradient", geom, 1, 4, "cpu")
    dedicated = raster_cgt_batch(
        {"mask_type": "gradient", "geom": geom}, 1, 4, "cpu")[0, 0]

    assert torch.allclose(legacy_gpu, legacy_cpu, atol=1e-7, rtol=0)
    assert torch.allclose(dedicated, legacy_cpu.square() * (3.0 - 2.0 * legacy_cpu))
    assert dedicated[0, 1] < legacy_cpu[0, 1]


def test_raster_cgt_batch_semantic_alpha_resizes_scales_and_clamps():
    alpha = torch.tensor([[0.0, 1.0], [0.5, 2.0]], dtype=torch.float32)
    spec = {"mask_type": "semantic", "alpha": alpha.numpy(), "amount": 0.5}

    got = raster_cgt_batch(spec, 4, 6, "cpu")

    expected = torch.nn.functional.interpolate(
        alpha[None, None], size=(4, 6), mode="bilinear",
        align_corners=False)[0, 0]
    expected = (expected * 0.5).clamp(0.0, 1.0)
    assert got.shape == (1, 1, 4, 6)
    assert torch.equal(got[0, 0], expected)
    assert float(got.min()) >= 0.0
    assert float(got.max()) <= 1.0


def test_raster_cgt_batch_semantic_full_res_passthrough_keeps_exact_zero():
    alpha = np.zeros((3, 5), np.float32)
    alpha[1, 2] = 0.75
    mixed = [
        {"mask_type": "semantic", "alpha": alpha},
        {"mask_type": "gradient", "geom": {
            "ZeroX": 0.0, "ZeroY": 0.0, "FullX": 1.0, "FullY": 0.0,
            "Flipped": "false",
        }},
    ]

    got = raster_cgt_batch(mixed, 3, 5, "cpu")

    assert got.shape == (2, 1, 3, 5)
    # 全分辨率语义 alpha 不 resize，原值直取——α=0 保持精确 0（合成端点不变量）
    assert torch.equal(got[0, 0], torch.from_numpy(alpha))


@pytest.mark.parametrize("spec, match", [
    ({"mask_type": "semantic"}, "'alpha'"),
    ({"mask_type": "semantic", "alpha": np.zeros((2, 2), np.int32)},
     "2-D floating"),
    ({"mask_type": "semantic", "alpha": np.zeros((2, 2, 2), np.float32)},
     "2-D floating"),
])
def test_raster_cgt_batch_semantic_rejects_bad_alpha(spec, match):
    with pytest.raises(ValueError, match=match):
        raster_cgt_batch(spec, 2, 2, "cpu")


def test_raster_cgt_batch_rejects_non_finite_amount():
    spec = {"mask_type": "gradient", "amount": float("nan"), "geom": {
        "ZeroX": 0, "ZeroY": 0, "FullX": 1, "FullY": 0,
    }}

    with pytest.raises(ValueError, match="finite number"):
        raster_cgt_batch(spec, 3, 5, "cpu")


def test_render_replays_once_strips_locals_and_residual_precedes_composite():
    base = torch.full((1, 3, 2, 2), 0.1)
    preset = {
        "attrs": {"Exposure2012": "1", "LocalExposure2012": "0.5"},
        "curves": {},
        "locals": [{"params": {"LocalExposure2012": 0.5}}],
    }
    specs = [
        {"mask_type": "gradient", "geom": {"ZeroX": 0, "ZeroY": 0,
                                               "FullX": 1, "FullY": 0}},
        {"mask_type": "gradient", "geom": {"ZeroX": 0, "ZeroY": 0,
                                               "FullX": 1, "FullY": 0}},
    ]
    alpha = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)]).unsqueeze(1)
    replay_info = {"consumed": {"Exposure2012"}, "fallback_ops": [],
                   "skipped_ops": []}

    def fake_replay(value, clean_preset, **kwargs):
        assert value.shape[0] == 1
        assert "locals" not in clean_preset
        assert "LocalExposure2012" not in clean_preset["attrs"]
        return value + 0.3, replay_info

    with mock.patch(
        "gpu_render.gpu.local_preset.raster_cgt_batch", return_value=alpha
    ), mock.patch(
        "gpu_render.gpu.gpu_replay.replay_batch", side_effect=fake_replay
    ) as replay_mock, mock.patch(
        "gpu_render.residual.load_residual", return_value=(object(), (2, 2, 2))
    ), mock.patch(
        "gpu_render.gpu.residual_gpu.apply_residual_batch",
        side_effect=lambda value, *_: value + 0.2,
    ):
        out, info = render_local_preset_tensor(
            base, preset, specs, "/unused/fits", residual_id="preset-id"
        )

    assert replay_mock.call_count == 1
    assert out.shape == (2, 3, 2, 2)
    assert torch.equal(out[0], base[0])
    assert torch.equal(out[1], torch.full_like(out[1], 0.6))
    assert torch.equal(info["alpha"], alpha)
    assert not info["alpha"].requires_grad
    assert info["expanded_single_input"] is True
    assert info["preset_locals_stripped"] is True
    assert info["stripped_local_corrections"] == 1
    assert info["stripped_local_attr_keys"] == ("LocalExposure2012",)
    assert info["residual_applied"] is True


def test_render_rejects_batch_without_one_spec_per_image():
    base = torch.zeros(2, 3, 4, 4)
    preset = {"attrs": {}, "curves": {}}
    spec = {"mask_type": "gradient", "geom": {
        "ZeroX": 0, "ZeroY": 0, "FullX": 1, "FullY": 0,
    }}

    with pytest.raises(ValueError, match="one CGT spec per image"):
        render_local_preset_tensor(base, preset, [spec], "/unused/fits")


def test_render_supports_distinct_per_image_specs_for_batched_sources():
    base = torch.stack([
        torch.full((3, 2, 2), 0.1),
        torch.full((3, 2, 2), 0.4),
    ])
    preset = {"attrs": {}, "curves": {}}
    specs = [
        {"mask_type": "gradient", "geom": {
            "ZeroX": 0, "ZeroY": 0, "FullX": 1, "FullY": 0,
        }},
        {"mask_type": "gradient", "geom": {
            "ZeroX": 1, "ZeroY": 0, "FullX": 0, "FullY": 0,
        }},
    ]
    alpha = torch.stack([torch.zeros(2, 2), torch.ones(2, 2)]).unsqueeze(1)

    with mock.patch(
        "gpu_render.gpu.local_preset.raster_cgt_batch", return_value=alpha
    ), mock.patch(
        "gpu_render.gpu.gpu_replay.replay_batch",
        side_effect=lambda value, *_args, **_kwargs: (
            value + 0.2,
            {"consumed": set(), "fallback_ops": [], "skipped_ops": []},
        ),
    ) as replay_mock:
        out, info = render_local_preset_tensor(base, preset, specs, "/unused/fits")

    assert replay_mock.call_count == 1
    assert out.shape == base.shape
    assert torch.equal(out[0], base[0])
    assert torch.equal(out[1], base[1] + 0.2)
    assert info["expanded_single_input"] is False
