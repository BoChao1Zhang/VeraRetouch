"""The four MetaCanvas structures, their parameter inventory, and the H_color ban."""

from __future__ import annotations

import inspect

import pytest
import torch

from q3vl.where.readout import band_params, cband_params, param_shapes
from q3vl.whereb.config import ARM_IDS, arm_config
from q3vl.whereb.heads import LatentHeads, RhoOutput, rho_numel, rho_split
from q3vl.whereb.model import MODEL_INPUT_KEYS, WhereBModel, parameter_table


def _small(arm: str):
    from q3vl.whereb.config import ConnectorConfig

    cfg = arm_config(arm)
    return WhereBModel(type(cfg)(
        arm=arm, seed=cfg.seed,
        connector=ConnectorConfig(dim=32, n_blocks=2, n_heads=4, ffn=64,
                                  text_dim=48, vision_dim=24, pos_bands=4),
    ))


def _inputs(b=2, p=20, t=5, text=48, vis=24):
    return dict(
        f_pre=torch.randn(b, p, vis), f_pre_pos=torch.randn(b, p, 2),
        f_pre_mask=torch.ones(b, p, dtype=torch.bool),
        h_where=torch.randn(b, t, text),
        h_where_mask=torch.ones(b, t, dtype=torch.bool),
    )


@pytest.mark.parametrize("arm", ARM_IDS)
def test_every_arm_builds_and_produces_only_global_parameters(arm):
    m = _small(arm).eval()
    out = m(**_inputs())
    assert out.w0.shape == (2,)
    assert out.w_raw.shape == (2, 71)
    assert out.alpha_raw.shape == (2,)
    for name, shape in param_shapes(m.readout).items():
        assert out.rho[name].shape == (2,) + shape
    assert torch.allclose(out.w_dir.norm(dim=-1), torch.ones(2), atol=1e-5)
    assert bool((out.alpha > 0).all())
    # no dense spatial logits anywhere in the output
    assert set(out.rho) == set(param_shapes(m.readout))


@pytest.mark.parametrize("arm,canvas,streams", [
    ("W01", 8, 1), ("W03", 16, 1), ("W05", 16, 1), ("W07", 16, 2),
])
def test_canvas_layout_matches_the_structure_table(arm, canvas, streams):
    m = _small(arm)
    assert m.canvas == canvas
    assert len(m.banks) == len(m.streams) == streams
    assert m.banks[0].n_queries == canvas * canvas
    assert m.facts()["n_queries"] == canvas * canvas


def test_dual_canvas_streams_are_fully_independent():
    m = _small("W07")
    n0 = {n for n, _ in m.streams[0].named_parameters()}
    n1 = {n for n, _ in m.streams[1].named_parameters()}
    assert n0 == n1                                    # same shape of stream
    p0 = dict(m.streams[0].named_parameters())
    p1 = dict(m.streams[1].named_parameters())
    assert all(p0[k] is not p1[k] for k in n0)         # but not the same tensors
    assert m.banks[0].tokens is not m.banks[1].tokens
    assert not torch.equal(m.banks[0].tokens, m.banks[1].tokens)


def test_split_head_and_joint_differ_by_a_second_pool_and_trunk():
    joint = {n for n, _ in _small("W03").heads.named_parameters()}
    split = {n for n, _ in _small("W05").heads.named_parameters()}
    assert any(n.startswith("pool.") for n in joint)
    assert not any(n.startswith("pool.") for n in split)
    assert {n for n in split if n.startswith("pool_axis.")}
    assert {n for n in split if n.startswith("pool_rho.")}
    assert {n for n in split if n.startswith("trunk_axis.")}
    assert {n for n in split if n.startswith("trunk_rho.")}
    # the output projections are shared in name but the *inputs* differ
    assert {n for n in joint if n.startswith("axis_out.")} == \
           {n for n in split if n.startswith("axis_out.")}


def _open_output_projections(h: LatentHeads) -> None:
    """The output projections are zero-init by design; wake them up so a test can
    observe which pooled vector each one is reading."""
    torch.manual_seed(3)
    with torch.no_grad():
        h.axis_out.proj.weight.normal_(std=0.1)
        h.rho_out.proj.weight.normal_(std=0.1)


def test_joint_head_uses_one_pool_for_both_outputs():
    torch.manual_seed(0)
    h = LatentHeads(16, 4, "band", n_pools=1, seed=0).eval()
    _open_output_projections(h)
    a, b = torch.randn(2, 9, 16), torch.randn(2, 9, 16)
    # a joint head ignores its second argument entirely
    assert torch.equal(h(a, b)[0], h(a, a)[0])
    assert torch.equal(h(a, b)[3]["mu"], h(a, a)[3]["mu"])


def test_split_head_rho_depends_only_on_the_rho_canvas():
    torch.manual_seed(0)
    hs = LatentHeads(16, 4, "cband12", n_pools=2, seed=0).eval()
    _open_output_projections(hs)
    a, b, c = torch.randn(1, 9, 16), torch.randn(1, 9, 16), torch.randn(1, 9, 16)
    _, _, _, r1 = hs(a, b)
    _, _, _, r2 = hs(c, b)
    assert torch.allclose(r1["sig_raw"], r2["sig_raw"])   # rho canvas unchanged
    _, _, _, r3 = hs(a, c)
    assert not torch.allclose(r1["sig_raw"], r3["sig_raw"])
    w1, _, _, _ = hs(a, b)
    w3, _, _, _ = hs(a, c)
    assert torch.allclose(w1, w3)                          # w reads only its own


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_rho_vector_round_trips_through_split(readout):
    n = rho_numel(readout)
    flat = torch.arange(2 * n, dtype=torch.float32).reshape(2, n)
    parts = rho_split(readout, flat)
    assert set(parts) == set(param_shapes(readout))
    for name, shape in param_shapes(readout).items():
        assert parts[name].shape == (2,) + shape
    assert n == (4 if readout == "band" else 36)


@pytest.mark.parametrize("readout", ["band", "cband12"])
def test_head_bias_init_lands_inside_the_readout_bounds(readout):
    out = RhoOutput(8, readout).eval()
    with torch.no_grad():
        raw = out(torch.zeros(1, 8))
    if readout == "band":
        p = band_params(raw)
        assert 0.02 < float(p["h"]) < 2.50
        assert 1.0 <= float(p["k"]) <= 40.0
        assert 0.5 < float(p["pi"]) < 1.0
    else:
        p = cband_params(raw)
        assert bool(((p["sigma"] >= 0.025) & (p["sigma"] <= 0.30)).all())
        assert bool(((p["o"] > 0) & (p["o"] < 1)).all())


def test_axis_bias_is_a_unit_vector_not_zeros():
    """w_dir = w_raw / (||w_raw|| + 1e-12) has a 1e12 gradient at the origin."""
    m = _small("W01").eval()
    with torch.no_grad():
        out = m(**_inputs())
    assert float(out.w_raw[0].norm()) > 0.5
    assert abs(float(out.w0[0])) < 1e-6
    assert abs(float(out.alpha[0]) - 1.0) < 1e-3


def test_forward_has_no_h_color_parameter_and_rejects_the_keyword():
    params = list(inspect.signature(WhereBModel.forward).parameters)
    assert params == ["self", *MODEL_INPUT_KEYS]
    assert not any("color" in p for p in params)
    m = _small("W01")
    with pytest.raises(TypeError):
        m(**_inputs(), h_color=torch.zeros(2, 3, 4))


def test_output_exposes_q_axis_and_q_readout_for_the_what_interface():
    m = _small("W07").eval()
    out = m(**_inputs())
    assert out.canvas_axis.shape == out.canvas_rho.shape == (2, 256, 32)
    assert not torch.equal(out.canvas_axis, out.canvas_rho)   # dual canvas
    mj = _small("W03").eval()
    oj = mj(**_inputs())
    assert torch.equal(oj.canvas_axis, oj.canvas_rho)         # shared canvas


def test_latent_view_is_a_where_a_latent():
    m = _small("W02").eval()
    out = m(**_inputs())
    lat = out.latent(0)
    assert lat.readout == "cband12"
    assert lat.w_raw.shape == (71,)
    assert set(lat.rho) == set(param_shapes("cband12"))


def test_full_size_parameter_table_is_monotone_in_capacity():
    rows = {r["arm"]: r for r in parameter_table(("W01", "W03", "W05", "W07"))}
    assert rows["W03"]["n_trainable_params"] > rows["W01"]["n_trainable_params"]
    assert rows["W05"]["n_trainable_params"] > rows["W03"]["n_trainable_params"]
    assert rows["W07"]["n_trainable_params"] > rows["W05"]["n_trainable_params"]
    # DualCanvas ~ 2x a single stream plus the split heads
    assert rows["W07"]["params_by_group"]["streams"] > \
        1.9 * rows["W05"]["params_by_group"]["streams"]
