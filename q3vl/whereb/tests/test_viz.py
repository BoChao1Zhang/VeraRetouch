"""Amendment A-5 item 3: the spatial-field visualisation red lines, in code."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.whereb.viz import (
    PerImageMinMaxError,
    color_scale,
    grid_to_img,
    overlay_grid_on_image,
    render_field,
)


def _field_with_pad():
    """A field whose pad cells dominate the range -- the RO-9c situation."""
    f = torch.full((8, 8), 0.1)
    f[2:6, 2:6] = 0.4                      # the real signal, inside the valid area
    valid = torch.zeros(8, 8, dtype=torch.bool)
    valid[1:7, 1:7] = True
    f[~valid] = 9.0                        # pad cells carry the huge mass
    return f, valid


def test_per_image_minmax_is_refused_not_warned():
    f, valid = _field_with_pad()
    with pytest.raises(PerImageMinMaxError, match="red line"):
        color_scale(f, valid, mode="per_image_minmax")


def test_color_scale_uses_valid_cells_only():
    f, valid = _field_with_pad()
    vmin, vmax = color_scale(f, valid)
    assert vmax == pytest.approx(0.4), "pad cells leaked into the colour scale"
    assert vmin == pytest.approx(0.1)
    # without the mask the pad mass would dominate, which is the bug
    assert color_scale(f, None)[1] == pytest.approx(9.0)


def test_render_marks_pad_cells_instead_of_filling_them():
    f, valid = _field_with_pad()
    r = render_field(f, valid, pad_style="white")
    assert r.n_pad == int((~valid).sum()) and r.n_valid == int(valid.sum())
    assert np.allclose(r.rgba[0, 0, :3], 1.0)        # a pad cell is drawn white
    assert not np.allclose(r.rgba[4, 4, :3], 1.0)    # a valid cell is coloured
    hatched = render_field(f, valid, pad_style="hatch")
    pad = ~valid.numpy()
    assert hatched.rgba[pad][:, :3].min() == 0.0     # visible hatching


def test_signal_is_visible_once_pad_is_excluded():
    """The whole point: with a valid-cell scale the subject is not flattened."""
    f, valid = _field_with_pad()
    r = render_field(f, valid)
    inner = r.rgba[3, 3, :3]
    ring = r.rgba[1, 1, :3]
    assert not np.allclose(inner, ring), "signal and background render identically"


def test_raw_stats_come_from_the_unnormalised_field():
    f, valid = _field_with_pad()
    r = render_field(f, valid)
    assert r.raw_stats["max"] == pytest.approx(0.4)     # not 1.0
    assert r.raw_stats["min"] == pytest.approx(0.1)
    assert r.scale_source == "valid_cells"


def test_render_returns_no_criterion_number():
    f, valid = _field_with_pad()
    r = render_field(f, valid)
    keys = set(r.to_dict()) | set(r.raw_stats)
    for banned in ("iou", "f1", "auc", "gate", "score"):
        assert not any(banned in k.lower() for k in keys), keys


def test_grid_to_img_is_the_exact_inverse_not_a_resize():
    m = grid_to_img(32, 48, 512, 768)
    assert m["scale_y"] == 16 and m["scale_x"] == 16
    assert m["y_edges"][0] == 0 and m["y_edges"][-1] == 512
    assert m["x_edges"][-1] == 768
    assert len(m["y_edges"]) == 33 and len(m["x_edges"]) == 49


def test_grid_to_img_refuses_a_non_integer_mapping():
    with pytest.raises(ValueError, match="not an integer multiple"):
        grid_to_img(32, 48, 500, 768)


def test_overlay_lands_on_cell_boundaries():
    field = torch.zeros(4, 4)
    field[0, 0] = 1.0
    img = torch.zeros(3, 16, 16)
    out, r = overlay_grid_on_image(field, img, alpha=1.0)
    assert out.shape == (16, 16, 3)
    # the hot cell occupies exactly its 4x4 pixel block, no interpolation bleed
    block = out[0:4, 0:4]
    assert np.allclose(block, block[0, 0])
    assert not np.allclose(out[0, 4], block[0, 0])


def test_fixed_scale_mode_needs_explicit_endpoints():
    f, valid = _field_with_pad()
    with pytest.raises(ValueError, match="explicit"):
        color_scale(f, valid, mode="fixed")
    assert color_scale(f, valid, mode="fixed", fixed=(0.0, 1.0)) == (0.0, 1.0)


def test_unknown_scale_mode_is_rejected():
    f, valid = _field_with_pad()
    with pytest.raises(ValueError, match="unknown colour-scale mode"):
        color_scale(f, valid, mode="whatever")
