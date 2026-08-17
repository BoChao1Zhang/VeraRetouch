"""Query sampling: the frozen (B, Q), the 128^3 grid, and RNG isolation."""

from __future__ import annotations

import pytest
import torch

from q3vl.whatb.queries import (
    BATCH_SAMPLES,
    COLORS_PER_STEP,
    QUERIES_PER_SAMPLE,
    QuerySampler,
    heldout_color_levels,
    image_histogram_colors,
    mining_ratio,
    select_hard,
    train_color_levels,
    uniform_grid,
)


def test_frozen_batch_organisation():
    assert (BATCH_SAMPLES, QUERIES_PER_SAMPLE) == (32, 256)
    assert COLORS_PER_STEP == 8192


def test_default_draw_is_one_frozen_step():
    x = QuerySampler().sample()
    assert x.shape == (32, 256, 3) and x.numel() // 3 == COLORS_PER_STEP
    assert float(x.min()) >= 0.0 and float(x.max()) <= 1.0


def test_training_colours_are_the_even_8bit_levels():
    lv = train_color_levels()
    assert lv.numel() == 128
    assert torch.allclose(lv * 255.0, torch.arange(0, 256, 2, dtype=lv.dtype))
    x = QuerySampler().sample(4, 64)
    codes = (x * 255.0).round()
    assert torch.equal(codes % 2, torch.zeros_like(codes))


def test_heldout_colours_are_the_complement():
    assert torch.equal(heldout_color_levels() * 255.0,
                       torch.arange(1, 256, 2, dtype=torch.float32))
    codes = (QuerySampler().sample_heldout(2, 32) * 255.0).round()
    assert torch.equal(codes % 2, torch.ones_like(codes))


def test_private_generator_does_not_touch_the_global_stream():
    torch.manual_seed(0)
    before = torch.rand(3)
    torch.manual_seed(0)
    QuerySampler(seed=1).sample(8, 128)
    after = torch.rand(3)
    assert torch.equal(before, after)


def test_same_seed_same_draw_and_state_round_trip():
    a, b = QuerySampler(seed=7), QuerySampler(seed=7)
    assert torch.equal(a.sample(2, 16), b.sample(2, 16))
    st = a.state()
    x = a.sample(2, 16)
    a.load_state(st)
    assert torch.equal(a.sample(2, 16), x)
    assert a.facts()["seed"] == 7 and a.facts()["n_draws"] == 3


def test_uniform_grid_shape_and_corners():
    g = uniform_grid(17)
    assert g.shape == (17 ** 3, 3)
    assert torch.equal(g[0], torch.zeros(3)) and torch.equal(g[-1], torch.ones(3))
    assert uniform_grid(9).shape == (729, 3)


def test_image_histogram_returns_normalised_weights():
    img = torch.rand(3, 32, 32)
    colors, w = image_histogram_colors(img, bits=5, top_k=64)
    assert colors.shape[0] == w.shape[0] <= 64
    assert abs(float(w.sum()) - 1.0) < 1e-5
    assert float(w[0]) >= float(w[-1])              # top-k is sorted descending


def test_image_histogram_is_alpha_weightable():
    img = torch.zeros(3, 4, 4)
    img[:, 0, 0] = 1.0
    alpha = torch.zeros(4, 4)
    alpha[0, 0] = 1.0
    colors, w = image_histogram_colors(img, bits=5, top_k=4, alpha=alpha)
    assert colors.shape[0] == 1 and torch.allclose(colors[0], torch.ones(3))


def test_mining_ratio_schedule():
    assert mining_ratio(0) == pytest.approx(0.10)
    assert mining_ratio(5) == pytest.approx(0.10)
    assert mining_ratio(12.5) == pytest.approx(0.25)
    assert mining_ratio(20) == pytest.approx(0.40)
    assert mining_ratio(40) == pytest.approx(0.40)


def test_select_hard_takes_the_largest_errors_on_device():
    err = torch.tensor([0.1, 0.9, 0.5, 0.3])
    idx = select_hard(err, 0.5)
    assert idx.device == err.device
    assert sorted(idx.tolist()) == [1, 2]
    assert select_hard(err, 0.0).numel() == 0


def test_image_hist_mode_refuses_the_grid_draw():
    s = QuerySampler(mode="image_hist")
    with pytest.raises(ValueError, match="uniform grid"):
        s.sample(2, 8)
    pool = torch.rand(50, 3)
    assert s.sample_from_pool(pool, 16).shape == (16, 3)
