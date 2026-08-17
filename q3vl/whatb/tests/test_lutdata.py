"""LUT evaluation must agree with the generator's own operator, bit for axis.

The one bug this file exists to catch is a transposed axis: ``.cube`` grids are
stored ``grid[b, g, r]`` and ``grid_sample``'s last coordinate axis is ``W``, so
a wrong permute produces a perfectly plausible -- and completely wrong -- LUT for
every sample in the campaign.  The reference is ``apply_lut_cpu_oracle``
(``dataset_build/src/construct/rendering.py:77-109``), the generator's own
test-only trilinear oracle.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dataset_build.src.construct.rendering import apply_lut_cpu_oracle
from q3vl.whatb.lutdata import LutBank, f_star, mix_alpha


def _fake_grid(d: int, seed: int = 0) -> np.ndarray:
    """A LUT that is a different, non-symmetric function of each axis."""
    rng = np.random.default_rng(seed)
    ax = np.linspace(0.0, 1.0, d, dtype=np.float32)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")
    out = np.stack((r ** 0.7, 0.3 * g + 0.2 * r, 1.0 - b ** 1.3), axis=-1)
    out = np.clip(out + 0.02 * rng.standard_normal(out.shape), 0.0, 1.0)
    return out.astype(np.float32)


@pytest.fixture()
def bank(tmp_path):
    grids = {"lut_a": _fake_grid(5, 1), "lut_b": _fake_grid(9, 2),
             "lut_c": _fake_grid(17, 3)}
    np.savez(tmp_path / "luts.npz", **grids)
    meta = {k: {"path": str(tmp_path / f"{k}.cube"),
                "dmin": [0.0, 0.0, 0.0], "dmax": [1.0, 1.0, 1.0]} for k in grids}
    (tmp_path / "luts_meta.json").write_text(json.dumps(meta))
    b = LutBank(tmp_path)
    b._grids = grids                                   # for the oracle side
    return b


def test_apply_matches_generator_cpu_oracle(bank):
    x = torch.rand(257, 3, dtype=torch.float32)
    for lid in ("lut_a", "lut_b", "lut_c"):
        got = bank.apply(x, lid).numpy()
        want = apply_lut_cpu_oracle(x.numpy().reshape(-1, 1, 3),
                                    bank._grids[lid]).reshape(-1, 3)
        assert np.abs(got - want).max() < 2e-6, lid


def test_apply_image_matches_apply_on_the_same_pixels(bank):
    img = torch.rand(3, 7, 11)
    per_pixel = bank.apply(img.permute(1, 2, 0), "lut_b").permute(2, 0, 1)
    assert torch.allclose(bank.apply_image(img, "lut_b"), per_pixel, atol=1e-7)


def test_grid_corners_are_the_stored_entries(bank):
    """align_corners=True + border padding: the eight corners are exact."""
    g = bank._grids["lut_a"]
    d = g.shape[0]
    corners = torch.tensor([[r, gg, b] for b in (0, 1) for gg in (0, 1)
                            for r in (0, 1)], dtype=torch.float32)
    got = bank.apply(corners, "lut_a").numpy()
    want = np.stack([g[b * (d - 1), gg * (d - 1), r * (d - 1)]
                     for b in (0, 1) for gg in (0, 1) for r in (0, 1)])
    assert np.abs(got - want).max() < 1e-6


def test_out_of_range_clamps_like_the_generator(bank):
    x = torch.tensor([[-0.5, 0.5, 1.7]], dtype=torch.float32)
    clamped = torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float32)
    assert torch.allclose(bank.apply(x, "lut_a"), bank.apply(clamped, "lut_a"),
                          atol=1e-7)


def test_f_star_is_the_data_law_with_snapped_endpoints(bank):
    x = torch.rand(64, 3)
    y = bank.apply(x, "lut_b")
    a = torch.rand(64, 1)
    a[0] = 0.0
    a[1] = 1.0
    got = bank.f_star(x, a, "lut_b")
    assert torch.equal(got[0], x[0])             # out[alpha == 0] = before
    assert torch.equal(got[1], y[1])             # out[alpha == 1] = edited
    mid = x[2:] * (1 - a[2:]) + y[2:] * a[2:]
    assert torch.allclose(got[2:], mid, atol=1e-7)
    assert torch.allclose(f_star(x, a, y), got)


def test_style_samples_are_alpha_one(bank):
    x = torch.rand(16, 3)
    y = bank.apply(x, "lut_a")
    assert torch.equal(mix_alpha(x, y, 1.0), y)


def test_lru_evicts_and_counts(bank):
    bank.cache_size = 2
    for lid in ("lut_a", "lut_b", "lut_c", "lut_a"):
        bank.volume(lid)
    assert len(bank._cache) == 2
    f = bank.facts()
    assert f["cache_misses"] >= 3 and f["resample"] == "none"
    assert f["grid_size_histogram"] == {5: 1, 9: 1, 17: 1}


def test_resample_flag_only_implements_none(tmp_path, bank):
    with pytest.raises(ValueError, match="lut-resample"):
        LutBank(bank.bank_dir, resample="33")


def test_missing_lut_id_is_loud(bank):
    with pytest.raises(KeyError, match="luts_meta"):
        bank.apply(torch.rand(2, 3), "nope")


def test_volume_and_data_must_share_a_device(bank):
    from q3vl.whatb.lutdata import apply_lut_volume

    vol = bank.volume("lut_a")
    with pytest.raises(ValueError, match="different|device"):
        apply_lut_volume(vol, torch.rand(4, 3).to(torch.float32).to("meta"))


def test_evaluate_library_stacks_every_lut(bank):
    x = torch.rand(32, 3)
    lib = bank.evaluate_library(["lut_a", "lut_b"], x)
    assert lib.shape == (2, 32, 3)
    assert torch.allclose(lib[1], bank.apply(x, "lut_b"))
