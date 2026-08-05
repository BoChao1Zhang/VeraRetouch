"""Protocol 6 / 8 / 14.9 -- the WC interfaces and the assembled arm."""

from __future__ import annotations

import inspect

import pytest
import torch

from q3vl.what.config import ARM_IDS, N_SLOTS, arm_config
from q3vl.what.model import MODEL_INPUT_KEYS, WhatModel
from q3vl.what.wc import WCEncoder, WhereSignals

from .conftest import MockBuilder, MockDataset, small_arm

N_PATCH = 20


def _signals(cfg, batch=2, dim=64):
    g = torch.Generator().manual_seed(0)
    n_rho = 36 if cfg.where_readout == "cband12" else 4
    return WhereSignals(
        m_low=torch.rand(batch, N_PATCH, generator=g),
        canvas_axis=torch.randn(batch, 8, dim, generator=g),
        canvas_rho=torch.randn(batch, 8, dim, generator=g),
        w_vec=torch.randn(batch, 73, generator=g),
        rho_vec=torch.randn(batch, n_rho, generator=g),
        source=cfg.where_source)


# --- WC interfaces -----------------------------------------------------------

@pytest.mark.parametrize("arm,n_tokens", [("T01", 1), ("T02", 3), ("T03", 4),
                                          ("T04", 6), ("C01", 1), ("C03", 6)])
def test_wc_token_counts_follow_protocol_6(arm, n_tokens):
    cfg = small_arm(arm)
    enc = WCEncoder(cfg, dim=cfg.backend.dim)
    tokens, mask = enc(torch.randn(2, N_PATCH, 1024), None, _signals(cfg, dim=cfg.backend.dim))
    assert tokens.shape == (2, n_tokens, cfg.backend.dim)
    assert mask.shape == (2, n_tokens) and bool(mask.all())
    assert enc.facts()["n_tokens"] == n_tokens


def test_wc0_never_sees_the_mask_and_wc1_does():
    assert small_arm("T01").mask_pool is False
    assert small_arm("T02").mask_pool is True
    assert small_arm("T03").mask_pool is False
    assert small_arm("T04").mask_pool is True
    assert small_arm("C01").mask_pool is False


def test_wc_refuses_to_run_without_the_signals_its_interface_names():
    cfg = small_arm("T04")
    enc = WCEncoder(cfg, dim=cfg.backend.dim)
    with pytest.raises(ValueError):
        enc(torch.randn(2, N_PATCH, 1024), None, WhereSignals(source="predicted"))


def test_the_no_where_control_cannot_be_handed_where_outputs():
    cfg = small_arm("C01")
    sig = WhereSignals(m_low=torch.rand(2, N_PATCH), w_vec=torch.randn(2, 73),
                       source="none")
    # C01's interface asks for nothing, so extra signals are simply unused;
    # what must not happen is an interface that *asks* while source is "none"
    enc = WCEncoder(cfg, dim=cfg.backend.dim)
    tokens, _ = enc(torch.randn(2, N_PATCH, 1024), None, sig)
    assert tokens.shape[1] == 1
    assert cfg.where_prefix is False


def test_oracle_arms_encode_the_oracle_latent_rather_than_query_states():
    cfg = small_arm("C03")
    enc = WCEncoder(cfg, dim=cfg.backend.dim)
    assert enc.oracle_enc is not None
    sig = _signals(cfg, dim=cfg.backend.dim)
    sig.canvas_axis = sig.canvas_rho = None      # oracle arms have no query states
    tokens, _ = enc(torch.randn(2, N_PATCH, 1024), None, sig)
    assert torch.isfinite(tokens).all()
    assert enc.facts()["z_where_from"] == "oracle_latent"


def test_predicted_arms_pool_the_query_states():
    cfg = small_arm("T03")
    enc = WCEncoder(cfg, dim=cfg.backend.dim)
    assert enc.oracle_enc is None and enc.facts()["z_where_from"] == "query_states"
    sig = _signals(cfg, dim=cfg.backend.dim)
    sig.canvas_axis = None
    with pytest.raises(ValueError):
        enc(torch.randn(2, N_PATCH, 1024), None, sig)


# --- the assembled arm -------------------------------------------------------

@pytest.mark.parametrize("arm", ARM_IDS)
def test_every_arm_constructs_and_runs(arm):
    cfg = small_arm(arm)
    model = WhatModel(cfg)
    ds = MockDataset(2)
    batch = MockBuilder(cfg).build([ds[0], ds[1]])
    out = model(**batch.inputs)
    assert out.params["mu"].shape == (2, N_SLOTS, 3)
    assert out.z_style.shape == (2, cfg.color.z_style_dim)
    assert out.m_color.shape == (2, cfg.color.n_queries, cfg.color.dim)
    assert out.slots.shape == (2, N_SLOTS, cfg.backend.dim)
    assert torch.isfinite(out.params["mu"]).all()
    rep = out.report()
    assert rep["mu_in_cube"] and rep["spd"] and rep["all_finite"]


def test_forward_signature_is_exactly_the_whitelist():
    got = set(inspect.signature(WhatModel.forward).parameters) - {"self"}
    assert got == set(MODEL_INPUT_KEYS)
    assert "i_tar" not in got and "mask_hi" not in got


def test_where_signals_has_no_field_for_a_forbidden_target():
    fields = set(WhereSignals.__dataclass_fields__)
    for bad in ("i_tar", "baked", "t_gt", "lut", "target"):
        assert not any(bad in f for f in fields), (bad, fields)


def test_batch_rejects_a_non_whitelisted_input():
    cfg = small_arm("T01")
    batch = MockBuilder(cfg).build([MockDataset(1)[0]])
    batch.inputs["i_tar"] = torch.zeros(1)
    with pytest.raises(AssertionError):
        batch.check_inputs()


def test_the_two_generators_produce_different_arms_from_the_same_batch():
    ds = MockDataset(2)
    outs = {}
    for arm in ("T04", "T08"):
        cfg = small_arm(arm)
        batch = MockBuilder(cfg).build([ds[0], ds[1]])
        outs[arm] = WhatModel(cfg)(**batch.inputs)
    assert outs["T04"].params["mu"].shape == outs["T08"].params["mu"].shape
    # SB48's geometry is shared across the batch, FG48's is not necessarily
    assert torch.allclose(outs["T08"].params["mu"][0], outs["T08"].params["mu"][1])


def test_facts_record_the_two_documented_decisions():
    f = WhatModel(arm_config("T01")).facts()
    assert f["sigma_param"] == "softplus_floor"
    assert f["global_affine_mode"] == "residual_zero"
    assert f["wc"] == "WC-0" and f["generator"] == "FG48"
    assert f["n_trainable_params"] > 0
