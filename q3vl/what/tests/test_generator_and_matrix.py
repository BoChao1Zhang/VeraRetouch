"""Protocol 7.4 / 7.5 / 8 -- the two generators, the 12 arms, the 2% budget."""

from __future__ import annotations

import pytest
import torch

from q3vl.what.config import (
    ARMS,
    ARM_IDS,
    CEILING_ARM_IDS,
    FG_TOTAL_PER_SAMPLE,
    LOSS_WEIGHTS,
    PARAM_MATCH_TOLERANCE,
    SB_TOTAL_PER_SAMPLE,
    WC_IDS,
    WC_INTERFACES,
    arm_config,
)
from q3vl.what.generator import (
    FG48,
    SB48,
    build_generator,
    match_report,
    solve_sb_bottleneck,
)
from q3vl.what.model import WhatModel

from .conftest import small_arm


def _pool_fn(v_dim):
    def fn(geom):
        b, n = geom["mu"].shape[:2]
        return torch.zeros(b, n, v_dim), {"invalid_fraction": 0.0}
    return fn


# --- the two generators ------------------------------------------------------

@pytest.mark.parametrize("arm,kind,n_out,per_sample",
                         [("T01", FG48, 23, FG_TOTAL_PER_SAMPLE),
                          ("T05", SB48, 14, SB_TOTAL_PER_SAMPLE)])
def test_generator_shapes_and_per_sample_budget(arm, kind, n_out, per_sample):
    cfg = small_arm(arm)
    gen = build_generator(cfg)
    assert isinstance(gen, kind)
    assert gen.n_prim_out == n_out
    assert gen.per_sample_params == per_sample
    style = torch.randn(2, cfg.backend.z_style_dim)
    params, extra = gen(style, torch.randn(2, 16, cfg.backend.dim), None,
                        torch.randn(2, 3, cfg.backend.dim), None,
                        _pool_fn(cfg.backend.v_dim))
    assert params["mu"].shape == (2, 48, 3)
    assert params["M"].shape == (2, 48, 3, 3)
    assert params["G"].shape == (2, 3, 3)
    assert extra["z_prim"].shape == (2, 48, n_out)
    assert extra["z_glob"].shape == (2, 12)


def test_fg_geometry_is_per_sample_and_sb_geometry_is_not():
    """The one structural difference between the arms (protocol 7.4 vs 7.5)."""
    for arm, shared in (("T01", False), ("T05", True)):
        cfg = small_arm(arm)
        gen = build_generator(cfg)
        with torch.no_grad():                # open the heads so outputs differ
            gen.heads.w2.normal_(std=0.2)
            if hasattr(gen, "provisional"):
                gen.provisional.w.normal_(std=0.2)
        style = torch.randn(2, cfg.backend.z_style_dim)
        m_color = torch.randn(2, 16, cfg.backend.dim)
        m_color[1] *= 3.0
        params, _ = gen(style, m_color, None, torch.randn(2, 3, cfg.backend.dim),
                        None, _pool_fn(cfg.backend.v_dim))
        same = torch.allclose(params["mu"][0], params["mu"][1], atol=1e-6)
        assert same is shared, arm
        # the payload is per sample in both arms
        assert not torch.allclose(params["M"][0], params["M"][1], atol=1e-6)


def _pool_from_geometry(cfg, seen):
    def pool(geom):
        seen["mu"] = geom["mu"]
        return geom["mu"].sum(-1, keepdim=True).expand(-1, -1, cfg.backend.v_dim), {}
    return pool


def test_the_v_gate_receives_gradient_at_step_zero():
    """Zero-init gating means ``v_i`` has *no effect* on the first forward -- by
    design (protocol 7.3 item 5).  What must be true at step 0 is that the gate
    itself gets a gradient, otherwise the branch could never open."""
    cfg = small_arm("T01")
    gen = build_generator(cfg)
    with torch.no_grad():
        gen.heads.w2.normal_(std=0.2)         # open the zero-init output layer
    params, _ = gen(torch.randn(1, cfg.backend.z_style_dim),
                    torch.randn(1, 16, cfg.backend.dim), None,
                    torch.randn(1, 3, cfg.backend.dim), None,
                    _pool_from_geometry(cfg, {}))
    params["M"].sum().backward()
    gates = [b.gate_v.grad for b in gen.backend.refine_blocks]
    assert all(g is not None for g in gates)
    assert sum(float(g.abs().sum()) for g in gates) > 0.0


def test_gradient_reaches_the_provisional_geometry_through_the_pooling():
    """Protocol 7.4: "the gradient can pass through the pooling back to the
    geometry".  Checked with the gates open, i.e. at any step past the first."""
    cfg = small_arm("T01")
    gen = build_generator(cfg)
    with torch.no_grad():
        gen.heads.w2.normal_(std=0.2)
        for b in gen.backend.refine_blocks:
            b.gate_v.fill_(1.0)
    seen: dict = {}
    params, _ = gen(torch.randn(1, cfg.backend.z_style_dim),
                    torch.randn(1, 16, cfg.backend.dim), None,
                    torch.randn(1, 3, cfg.backend.dim), None,
                    _pool_from_geometry(cfg, seen))
    params["M"].sum().backward()
    assert seen["mu"].requires_grad
    assert float(gen.provisional.w.grad.abs().sum()) > 0.0


def test_sb_geometry_gets_its_own_gradient():
    cfg = small_arm("T05")
    gen = build_generator(cfg)
    params, _ = gen(torch.randn(1, cfg.backend.z_style_dim),
                    torch.randn(1, 16, cfg.backend.dim), None,
                    torch.randn(1, 3, cfg.backend.dim), None,
                    lambda g: (g["mu"].sum(-1, keepdim=True)
                               .expand(-1, -1, cfg.backend.v_dim), {}))
    params["mu"].sum().backward()
    assert float(gen.geometry.raw.grad.abs().sum()) > 0.0


# --- the 2% budget -----------------------------------------------------------

def test_fg_and_sb_are_within_two_percent_at_the_real_widths():
    cfg = arm_config("T01")
    rep = match_report(cfg.backend, cfg.lut, cfg.fg_bottleneck)
    assert rep["within_tolerance"], rep
    assert rep["rel_diff"] <= PARAM_MATCH_TOLERANCE
    assert rep["sb_bottleneck"] == solve_sb_bottleneck(cfg.backend, cfg.fg_bottleneck)


def test_every_wc_pair_is_within_two_percent():
    """The comparison the protocol actually makes is per wave: T01 vs T05, ..."""
    for wc, (fg_arm, sb_arm) in zip(WC_IDS, [("T01", "T05"), ("T02", "T06"),
                                             ("T03", "T07"), ("T04", "T08")]):
        a = WhatModel(arm_config(fg_arm)).n_trainable()
        b = WhatModel(arm_config(sb_arm)).n_trainable()
        rel = abs(a - b) / max(a, b)
        assert rel <= PARAM_MATCH_TOLERANCE, (wc, fg_arm, sb_arm, a, b, rel)


def test_the_two_generators_share_depth_width_and_query_count():
    fg, sb = build_generator(arm_config("T01")), build_generator(arm_config("T05"))
    for attr in ("dim", "n_heads", "ffn", "seed_blocks", "refine_blocks", "n_slots"):
        assert getattr(fg.cfg, attr) == getattr(sb.cfg, attr), attr
    assert fg.backend.slots.shape == sb.backend.slots.shape


# --- the 12-arm matrix -------------------------------------------------------

def test_the_arm_matrix_is_the_protocol_8_matrix():
    assert len(ARMS) == 12
    main = [a for a in ARM_IDS if a.startswith("T")]
    assert len(main) == 8
    assert {ARMS[a][0] for a in main} == set(WC_IDS)
    assert {ARMS[a][1] for a in main} == {"FG48", "SB48"}
    # 4 WC x 2 generators, each combination exactly once
    assert len({(ARMS[a][0], ARMS[a][1]) for a in main}) == 8
    assert CEILING_ARM_IDS == ("C03", "C04")
    assert [ARMS[a][2] for a in ("C01", "C02")] == ["none", "none"]
    assert [ARMS[a][2] for a in ("C03", "C04")] == ["oracle", "oracle"]


def test_wc_interfaces_follow_protocol_6():
    assert WC_INTERFACES["WC-0"]["mask_pool"] is False
    assert WC_INTERFACES["WC-1"]["mask_pool"] is True
    assert WC_INTERFACES["WC-2"]["mask_pool"] is False
    assert WC_INTERFACES["WC-3"]["mask_pool"] is True
    # WC-3 = WC-1 + WC-2
    union = set(WC_INTERFACES["WC-1"]["tokens"]) | set(WC_INTERFACES["WC-2"]["tokens"])
    assert set(WC_INTERFACES["WC-3"]["tokens"]) == union
    # the strict no-where control drops the <where> prefix; WC-0 keeps it
    assert WC_INTERFACES["WC-0"]["where_prefix"] is True
    assert WC_INTERFACES["NOWHERE"]["where_prefix"] is False


def test_loss_weights_are_the_protocol_9_5_numbers():
    assert LOSS_WEIGHTS == {
        "L_func": 1.00, "L_hc": 10.00, "R_sparse": 0.001, "L_style_cos": 0.05,
        "L_style_dist": 0.05, "L_varcov": 0.02, "L_bake": 0.10,
    }


def test_unknown_arm_is_rejected():
    with pytest.raises(ValueError):
        arm_config("T99")
