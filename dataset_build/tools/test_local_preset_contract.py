from __future__ import annotations

import numpy as np
import pytest

from dataset_build.src.construct import mask_synth
from gpu_render.gpu.local_preset import raster_cgt_batch


@pytest.mark.parametrize("spec", [
    {
        "mask_type": "circulargradient",
        "geom": {
            "Top": 0.17, "Bottom": 0.83, "Left": 0.21, "Right": 0.79,
            "Angle": 17.0, "Feather": 65.0, "Flipped": "true",
        },
    },
    {
        "mask_type": "gradient",
        "geom": {
            "ZeroX": 0.18, "ZeroY": 0.42, "FullX": 0.81, "FullY": 0.61,
            "Flipped": "false",
        },
    },
])
def test_construct_cgt_matches_gpu_composite_alpha(spec):
    expected = mask_synth.cgt_raster(
        spec["mask_type"], spec["geom"], 123, 177)
    actual = raster_cgt_batch(spec, 123, 177, "cpu")[0, 0].numpy()

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=5e-7)
    np.testing.assert_array_equal(
        (actual * 255).astype(np.uint8),
        (expected * 255).astype(np.uint8),
    )
