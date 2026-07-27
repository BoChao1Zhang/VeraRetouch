from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from dataset_build.src.construct.rendering import apply_lut_cpu_oracle
from experiments.lut_renderer_pilot.common import count_parameters
from experiments.lut_renderer_pilot.data import (
    LutRecord,
    OnlinePairBatchSampler,
    OnlinePairSampler,
    native_lattice,
    render_lut_batch,
)
from experiments.lut_renderer_pilot.losses import hard_mining_ratio, srgb_to_lab
from experiments.lut_renderer_pilot.models import SharedGeometryCGLUT, VeraStyleRenderer


class _SingleGridStore:
    def __init__(self, grid: np.ndarray) -> None:
        self.grid = grid

    def get(self, style_index: int) -> np.ndarray:
        if style_index != 0:
            raise IndexError(style_index)
        return self.grid


def _identity_grid(size: int) -> np.ndarray:
    values = np.linspace(0.0, 1.0, size, dtype=np.float32)
    blue, green, red = np.meshgrid(values, values, values, indexing="ij")
    return np.stack((red, green, blue), axis=-1)


def _record(size: int, domain_min=(0.0, 0.0, 0.0), domain_max=(1.0, 1.0, 1.0)) -> LutRecord:
    return LutRecord(
        style_index=0,
        preset_id="identity",
        content_hash="0" * 64,
        path="identity.cube",
        grid_size=size,
        domain_min=domain_min,
        domain_max=domain_max,
        taxonomy_major="test",
        taxonomy_minor="identity",
    )


class CanonicalLutTests(unittest.TestCase):
    def test_gpu_contract_matches_cpu_oracle_and_preserves_axes(self) -> None:
        grid = _identity_grid(5)
        image = torch.tensor(
            [[[[0.0, 0.2, 1.0], [0.8, 0.4, 0.1]],
              [[0.1, 0.7, 0.3], [0.5, 0.9, 0.0]],
              [[1.0, 0.0, 0.6], [0.2, 0.3, 0.9]]]],
            dtype=torch.float32,
        ).permute(0, 3, 1, 2)
        rendered = render_lut_batch(
            image, torch.tensor([0]), [_record(5)], _SingleGridStore(grid)
        )
        expected = apply_lut_cpu_oracle(image[0].permute(1, 2, 0).numpy(), grid)
        np.testing.assert_allclose(
            rendered[0].permute(1, 2, 0).numpy(), expected, atol=2.0e-6
        )
        torch.testing.assert_close(rendered, image, atol=2.0e-6, rtol=0.0)

    def test_non_unit_domain_is_applied_before_sampling(self) -> None:
        grid = _identity_grid(3)
        image = torch.tensor([[[[0.0]], [[0.5]], [[1.0]]]], dtype=torch.float32)
        record = _record(3, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
        rendered = render_lut_batch(
            image, torch.tensor([0]), [record], _SingleGridStore(grid)
        )
        torch.testing.assert_close(
            rendered.flatten(), torch.tensor([0.5, 0.75, 1.0]), atol=1.0e-6, rtol=0.0
        )

    def test_native_lattice_uses_bgr_storage_and_rgb_coordinates(self) -> None:
        grid = _identity_grid(4)
        inputs, targets = native_lattice(
            _record(4), grid, device=torch.device("cpu")
        )
        torch.testing.assert_close(inputs, targets, atol=1.0e-7, rtol=0.0)


class CglutTests(unittest.TestCase):
    def test_parameter_count_and_identity_initialization(self) -> None:
        model = SharedGeometryCGLUT(3)
        self.assertEqual(model.shared_parameter_count(), 162_892)
        self.assertEqual(count_parameters(model), 162_892 + 3 * 64)
        points = torch.rand((3, 257, 3))
        output = model.forward_points(points, torch.arange(3))
        torch.testing.assert_close(output, points, atol=1.0e-6, rtol=0.0)
        torch.testing.assert_close(
            torch.diagonal(model.cholesky(), dim1=-2, dim2=-1),
            torch.full((32, 3), 0.15),
            atol=1.0e-7,
            rtol=0.0,
        )

    def test_hard_mining_schedule_is_one_based(self) -> None:
        self.assertIsNone(hard_mining_ratio(4))
        self.assertAlmostEqual(hard_mining_ratio(5), 0.10)
        self.assertAlmostEqual(hard_mining_ratio(20), 0.40)
        self.assertAlmostEqual(hard_mining_ratio(40), 0.40)


class VeraTests(unittest.TestCase):
    def test_official_decoder_loads_strictly(self) -> None:
        checkpoint = Path("/home/bc/data/models/VeraRetouch/model.safetensors")
        if not checkpoint.is_file():
            self.skipTest(f"missing local checkpoint: {checkpoint}")
        model = VeraStyleRenderer(2, checkpoint)
        self.assertEqual(count_parameters(model.decoder), 2_577_795)
        self.assertEqual(count_parameters(model.style_embeddings), 2 * 2688)
        self.assertEqual(torch.count_nonzero(model.style_embeddings.weight).item(), 0)
        points = torch.rand((2, 17, 3))
        output = model.forward_points(points, torch.tensor([0, 1]))
        self.assertEqual(output.shape, points.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(((0.0 <= output) & (output <= 1.0)).all())


class LossAndSamplerTests(unittest.TestCase):
    def test_lab_golden_values(self) -> None:
        colors = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            dtype=torch.float64,
        )
        lab = srgb_to_lab(colors)
        torch.testing.assert_close(lab[0], torch.zeros(3, dtype=torch.float64), atol=1e-8, rtol=0)
        torch.testing.assert_close(
            lab[1], torch.tensor([100.0, 0.0, 0.0], dtype=torch.float64), atol=2e-4, rtol=0
        )
        torch.testing.assert_close(
            lab[2],
            torch.tensor([53.2408, 80.0925, 67.2032], dtype=torch.float64),
            atol=5e-4,
            rtol=0,
        )

    def test_lab_has_finite_endpoint_gradients(self) -> None:
        colors = torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.04045, 0.0, 1.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        srgb_to_lab(colors).sum().backward()
        self.assertTrue(torch.isfinite(colors.grad).all())

    def test_online_sampler_is_deterministic_and_covers_styles_once(self) -> None:
        first = list(OnlinePairSampler(31, 7, base_seed=1701, epoch=3))
        second = list(OnlinePairSampler(31, 7, base_seed=1701, epoch=3))
        other_epoch = list(OnlinePairSampler(31, 7, base_seed=1701, epoch=4))
        self.assertEqual(first, second)
        self.assertNotEqual(first, other_epoch)
        self.assertEqual(sorted(style for style, _ in first), list(range(31)))
        self.assertTrue(all(0 <= source < 7 for _, source in first))

    def test_aspect_batch_sampler_keeps_exact_style_coverage(self) -> None:
        sources = [
            {"width": 400, "height": 800},
            {"width": 800, "height": 800},
            {"width": 1200, "height": 600},
        ]
        sampler = OnlinePairBatchSampler(
            sources, 31, 8, base_seed=1701, epoch=2
        )
        batches = list(sampler)
        flattened = [pair for batch in batches for pair in batch]
        self.assertEqual(len(batches), 4)
        self.assertEqual(sorted(style for style, _ in flattened), list(range(31)))
        self.assertTrue(all(len(batch) <= 8 for batch in batches))


if __name__ == "__main__":
    unittest.main()
