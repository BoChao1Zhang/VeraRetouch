"""Protocol 4.1 -- F_pre grid, and the Qwen merge-order unshuffle.

The unshuffle is the single most dangerous line in the Where-A pipeline: the
pre-merger token sequence is *not* row-major, and a wrong reshape scrambles the
image into 2x2 blocks without raising anything.  ``test_unshuffle_matches_the_real_processor``
pins it against the actual ``Qwen2VLImageProcessorFast`` output, not against a
re-derivation of the same assumption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.where.config import MODEL_DIR, SPATIAL_MERGE
from q3vl.where.fpre import grid_from_geometry, shuffle_from_grid, unshuffle_to_grid


def test_grid_from_geometry():
    assert grid_from_geometry(512, 896) == (32, 56)
    assert grid_from_geometry(768, 512) == (48, 32)
    with pytest.raises(ValueError):
        grid_from_geometry(500, 512)


def test_unshuffle_roundtrip():
    gh, gw, c = 8, 12, 5
    g = torch.Generator().manual_seed(0)
    x = torch.randn(gh * gw, c, generator=g)
    grid = unshuffle_to_grid(x, gh, gw)
    assert grid.shape == (gh, gw, c)
    assert torch.allclose(shuffle_from_grid(grid), x, atol=0)


def test_unshuffle_is_not_a_plain_view():
    """If it were, the whole thing would be a silent no-op bug."""
    gh, gw, c = 4, 6, 1
    x = torch.arange(gh * gw, dtype=torch.float32).reshape(-1, 1)
    grid = unshuffle_to_grid(x, gh, gw)
    naive = x.reshape(gh, gw, c)
    assert not torch.allclose(grid, naive)
    # token 1 is the (0,1) cell of the first 2x2 block -> grid[0,1]
    assert float(grid[0, 1, 0]) == 1.0
    # token 2 is the (1,0) cell of that block -> grid[1,0]
    assert float(grid[1, 0, 0]) == 2.0


def test_unshuffle_rejects_bad_shapes():
    with pytest.raises(ValueError):
        unshuffle_to_grid(torch.zeros(10, 3), 4, 4)
    with pytest.raises(ValueError):
        unshuffle_to_grid(torch.zeros(15, 3), 3, 5)      # odd grid, merge=2
    with pytest.raises(ValueError):
        unshuffle_to_grid(torch.zeros(3), 1, 3)


@pytest.mark.skipif(not Path(MODEL_DIR).exists(), reason="model dir not present")
def test_unshuffle_matches_the_real_processor():
    """Feed an image whose every 16x16 patch carries its own (row, col) code and
    check that the unshuffled grid reproduces the spatial layout exactly."""
    from PIL import Image
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    ip = proc.image_processor
    patch = int(ip.patch_size)
    assert int(ip.merge_size) == SPATIAL_MERGE
    gh, gw = 8, 12

    codes = np.arange(gh * gw, dtype=np.float64).reshape(gh, gw)
    img = np.repeat(np.repeat(codes, patch, axis=0), patch, axis=1)
    rgb = np.stack([img, img, img], axis=-1).astype(np.uint8)
    out = ip(images=[Image.fromarray(rgb)], do_resize=False, return_tensors="pt")

    t, ph, pw = (int(v) for v in out["image_grid_thw"][0])
    assert (t, ph, pw) == (1, gh, gw)
    # channel 0 of the first temporal patch is the first patch*patch block of
    # the flattened patch vector; its mean recovers the code
    tokens = out["pixel_values"].reshape(gh * gw, -1)[:, : patch * patch].mean(-1, keepdim=True)
    grid = unshuffle_to_grid(tokens.double(), gh, gw)[..., 0].numpy()

    # the processor normalises with mean/std 0.5, so codes are affine-mapped;
    # rank order is enough to pin the layout
    order = np.argsort(grid.reshape(-1))
    assert np.array_equal(order, np.argsort(codes.reshape(-1))), (
        "unshuffled grid does not follow the processor's spatial layout"
    )
    naive = tokens.reshape(gh, gw).double().numpy()
    assert not np.array_equal(np.argsort(naive.reshape(-1)), order), (
        "a naive reshape would have worked -- the test is not proving anything"
    )
