"""EPR-030: the query backbone, the straight-through clamp and the L0 loss.

Everything runs on the CPU (``CUDA_VISIBLE_DEVICES=""``).  Five things are
pinned, one per failure this arm exists to close:

(a) step 0 is **exactly** the identity and every parameter equals the
    ``SharedGeometry`` init -- the Bias-HyperInit head is wired, not just
    written;
(b) ``sum(p.numel())`` equals the closed form and the hand count, so the
    proposal's parameter table cannot drift from the module;
(c) the 22-d bias slices carry the values the proposal names;
(d) the straight-through clamp's forward is bit-for-bit the hard clamp and its
    backward is non-zero where the hard one is exactly zero;
(e) ``--loss l0`` with a non-zero extra weight raises before the first step.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import losses_l0 as L0
from q3vl.whatb.arms import carrier as A
from q3vl.whatb.glut import (
    EPS,
    GlutCarrier,
    GlutParams,
    _clamp01,
    glut_forward,
    grid_axis_sizes,
    softplus_inverse,
    uniform_grid_positions,
)
from q3vl.whatb.lutdata import LutBank
from q3vl.whatb.qdecoder import (
    GAUSS_SLICES,
    GLOBAL_SLICES,
    GlutQueryDecoder,
    qdecoder_param_count,
)
from q3vl.whatb.queries import QuerySampler
from q3vl.whatb.scripts import run_epr030_arm as E

RUNNER_PY = Path(E.__file__)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _fake_grid(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ax = np.linspace(0.0, 1.0, d, dtype=np.float32)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")
    out = np.stack((r ** 0.8, 0.4 * g + 0.3 * r, 1.0 - b ** 1.2), axis=-1)
    out = np.clip(out + 0.03 * rng.standard_normal(out.shape), 0.0, 1.0)
    return out.astype(np.float32)


@pytest.fixture()
def bank(tmp_path) -> LutBank:
    grids = {"lut_a": _fake_grid(9, 1), "lut_b": _fake_grid(9, 2)}
    np.savez(tmp_path / "luts.npz", **grids)
    meta = {k: {"path": str(tmp_path / f"{k}.cube"), "dmin": [0.0] * 3,
                "dmax": [1.0] * 3} for k in grids}
    (tmp_path / "luts_meta.json").write_text(json.dumps(meta))
    b = LutBank(tmp_path)
    b._grids = grids
    return b


def tiny_cfg(**kw) -> E.Epr030Config:
    base = dict(cond_dim=8, n_gauss=6, gen_width=16, lib_size=2, n_repeats=2,
                bake_grid=9, interp_pairs=2, loss_level=1,
                qdec_dim=16, qdec_layers=2, qdec_heads=2, batch_split="32x256")
    base.update(kw)
    return E.Epr030Config(**base)


def _z(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, A.SEG_COLOR_HIDDEN_DIM, generator=g)


# --------------------------------------------------------------------------- #
# (a) step 0 is the identity, and the parameters are the init
# --------------------------------------------------------------------------- #
def test_step0_is_the_identity_and_the_shared_geometry_init():
    torch.manual_seed(0)
    dec = GlutQueryDecoder()
    rep = dec.assert_step0_identity()          # raises on failure, never warns
    assert rep["step0_maxabs_f_minus_id"] <= 1e-6
    for k, v in rep.items():
        if k.startswith("init_dev_"):
            assert v == 0.0, k                 # same tensors: exact, on any device
        if k.startswith("shared_geometry_dev_"):
            assert v <= 2.0 ** -24, k          # recomputed grid: one float32 ULP

    p = dec(_z(4, seed=7))
    want = GlutParams.identity(dec.n_gauss, batch=4, sigma=0.15, opacity_logit=4.0)
    for name in ("mu", "chol_diag", "chol_off", "opacity_logit", "m_local",
                 "b_local", "g_matrix", "g_bias"):
        assert torch.equal(getattr(p, name), getattr(want, name)), name
    # ... and the mu grid is the carrier's own, element for element
    assert torch.equal(p.mu[0], uniform_grid_positions(dec.n_gauss))


def test_step0_identity_holds_for_the_zero_init_ablation_row():
    torch.manual_seed(0)
    dec = GlutQueryDecoder(head_init="zero")
    # the parameters are NOT the SharedGeometry init here (that is the point of
    # the row) but sum_i w_i == 1 still makes f the identity
    rep = dec.assert_step0_identity()
    assert rep["step0_maxabs_f_minus_id"] <= 1e-6
    # the SharedGeometry comparison is skipped on this row by construction
    assert not any(k.startswith("shared_geometry_dev_") for k in rep)
    p = dec(_z(2))
    assert float(p.chol_diag.detach().abs().max()) == 0.0
    assert float(p.opacity_logit.detach().abs().max()) == 0.0


def test_a_mis_encoded_head_bias_still_raises_above_the_ulp_floor():
    """The ULP floor must not swallow a real mis-encoding.

    The first GPU smoke of this arm failed on ``mu`` with 5.960e-08 = one float32
    ULP, which is the CPU/CUDA disagreement on ``(i + 0.5) / 3``, not a defect.
    A defect is O(0.1) or larger -- this pins that the floor separates the two.
    """
    torch.manual_seed(0)
    dec = GlutQueryDecoder(dim=16, layers=1, heads=2)
    lo, hi = GAUSS_SLICES["chol_diag"]
    with torch.no_grad():
        dec.head_gauss.bias[lo:hi] += 1e-3      # sigma silently changed
    with pytest.raises(AssertionError, match="chol_diag"):
        dec.assert_step0_identity()

    torch.manual_seed(0)
    dec = GlutQueryDecoder(dim=16, layers=1, heads=2)
    with torch.no_grad():
        dec.mu_base[0, 2] += 1e-4               # a grid point moved off the lattice
    with pytest.raises(AssertionError, match="mu"):
        dec.assert_step0_identity()


def test_a_broken_head_makes_the_assertion_raise():
    torch.manual_seed(0)
    dec = GlutQueryDecoder(dim=32, layers=1, heads=2)
    with torch.no_grad():
        dec.head_gauss.weight.normal_(0.0, 0.5)     # "zero init not wired"
    with pytest.raises(AssertionError):
        dec.assert_step0_identity()


# --------------------------------------------------------------------------- #
# (b) the parameter count
# --------------------------------------------------------------------------- #
def test_parameter_count_matches_the_closed_form_and_the_hand_count():
    torch.manual_seed(0)
    dec = GlutQueryDecoder()                       # N=48, d=512, L=6, M=1
    n = sum(p.numel() for p in dec.parameters())
    assert n == qdecoder_param_count() == 20_278_818

    d, n_g, m = 512, 48, 1
    q_emb = (n_g + 1) * d
    pe = sum(grid_axis_sizes(n_g)) * d
    ln = 2 * 2560
    mem = m * (2560 * d + d)
    attn = 3 * d * d + 3 * d + (d * d + d)
    ffn = (d * 4 * d + 4 * d) + (4 * d * d + d)
    layer = 2 * d + attn + 2 * d + ffn
    heads = (d * 22 + 22) + (d * 12 + 12)
    assert q_emb + pe + ln + mem + 6 * layer + heads == n

    # the default configuration is bigger than the published Retouch Renderer
    # (2,577,795) -- the campaign's declared floor for this backbone
    assert n >= 2_577_795
    assert qdecoder_param_count(self_attn=True) > n
    assert qdecoder_param_count(dim=128) < qdecoder_param_count(dim=256) < n


@pytest.mark.parametrize("kw", [
    dict(dim=128), dict(dim=256), dict(layers=2), dict(layers=4),
    dict(mem_rows=4), dict(mem_rows=8), dict(self_attn=True),
])
def test_every_ablation_rung_counts_itself_correctly(kw):
    torch.manual_seed(0)
    dec = GlutQueryDecoder(**kw)
    assert sum(p.numel() for p in dec.parameters()) == qdecoder_param_count(**kw)
    assert dec.assert_step0_identity()["step0_maxabs_f_minus_id"] <= 1e-6


# --------------------------------------------------------------------------- #
# (c) the bias slices
# --------------------------------------------------------------------------- #
def test_the_head_bias_carries_the_target_init_slice_by_slice():
    torch.manual_seed(0)
    dec = GlutQueryDecoder(dim=32, layers=1, heads=2)
    b = dec.head_gauss.bias.detach()
    assert torch.equal(dec.head_gauss.weight, torch.zeros_like(dec.head_gauss.weight))
    lo, hi = GAUSS_SLICES["d_mu"]
    assert float(b[lo:hi].abs().max()) == 0.0
    lo, hi = GAUSS_SLICES["chol_diag"]
    assert torch.allclose(b[lo:hi], torch.full((3,), softplus_inverse(0.15)))
    assert math.isclose(softplus_inverse(0.15), -1.8212, abs_tol=1e-4)
    lo, hi = GAUSS_SLICES["chol_off"]
    assert float(b[lo:hi].abs().max()) == 0.0
    lo, hi = GAUSS_SLICES["opacity_logit"]
    assert float(b[lo]) == 4.0
    lo, hi = GAUSS_SLICES["d_m"]
    assert float(b[lo:hi].abs().max()) == 0.0        # M = I + dM, dM bias 0
    lo, hi = GAUSS_SLICES["b"]
    assert float(b[lo:hi].abs().max()) == 0.0
    g = dec.head_global.bias.detach()
    assert float(g.abs().max()) == 0.0
    assert (GLOBAL_SLICES["d_g_matrix"], GLOBAL_SLICES["g_bias"]) == ((0, 9), (9, 12))


def test_the_colour_pe_is_indexed_by_the_mu_grid():
    torch.manual_seed(0)
    dec = GlutQueryDecoder(n_gauss=48, dim=8, layers=1, heads=2)
    assert dec.grid_axes == grid_axis_sizes(48) == (4, 4, 3)
    assert tuple(dec.pe_r.shape) == (4, 8)
    assert tuple(dec.pe_b.shape) == (3, 8)
    # the query of Gaussian i uses the (r, g, b) cell whose centre is mu_i
    mu = uniform_grid_positions(48)
    idx = dec.grid_index
    for i in (0, 5, 17, 47):
        cell = (idx[i].to(torch.float32) + 0.5) / torch.tensor(
            [float(a) for a in dec.grid_axes])
        assert torch.allclose(cell, mu[i]), i
    # PEs start at zero, so at step 0 the query is q_emb alone
    q = dec.queries(1)[0]
    assert torch.equal(q[: dec.n_gauss], dec.q_emb[: dec.n_gauss])


def test_the_global_query_is_anchored_at_zero_not_at_the_identity():
    """``G = dG`` (NOTES 1): ``G = I`` would make step 0 ``f = 2x``."""
    torch.manual_seed(0)
    zero_anchor = GlutQueryDecoder(dim=16, layers=1, heads=2)
    assert float(zero_anchor(_z(2)).g_matrix.detach().abs().max()) == 0.0
    i_anchor = GlutQueryDecoder(dim=16, layers=1, heads=2, g_residual=True)
    p = i_anchor(_z(2))
    assert torch.equal(p.g_matrix[0], torch.eye(3))
    x = torch.rand(1, 64, 3)
    y = glut_forward(x, p, clamp="two")
    assert float((y - x).detach().abs().max()) > 0.1  # not the identity: f = 2x clamped
    with pytest.raises(AssertionError):
        i_anchor.assert_step0_identity()


# --------------------------------------------------------------------------- #
# (d) the straight-through clamp
# --------------------------------------------------------------------------- #
def _saturating_params(batch: int = 2, n: int = 4) -> GlutParams:
    """A parameter set whose global branch is far outside the gamut."""
    p = GlutParams.identity(n, batch=batch)
    return GlutParams(
        mu=p.mu, chol_diag=p.chol_diag, chol_off=p.chol_off,
        opacity_logit=p.opacity_logit, m_local=p.m_local, b_local=p.b_local,
        g_matrix=torch.eye(3).expand(batch, 3, 3).clone().requires_grad_(True),
        g_bias=torch.full((batch, 3), 3.0, requires_grad=True))


@pytest.mark.parametrize("clamp", ["two", "one"])
def test_straight_through_clamp_is_bit_identical_in_the_forward(clamp):
    torch.manual_seed(0)
    p = _saturating_params()
    x = torch.rand(2, 128, 3)
    hard = glut_forward(x, p, clamp=clamp, clamp_grad="hard")
    st = glut_forward(x, p, clamp=clamp, clamp_grad="st")
    assert torch.equal(hard, st)                    # every bit, not allclose


def test_the_hard_clamp_returns_zero_gradient_and_the_st_one_does_not():
    torch.manual_seed(0)
    x = torch.rand(2, 128, 3)

    p = _saturating_params()
    glut_forward(x, p, clamp="two", clamp_grad="hard").sum().backward()
    assert float(p.g_bias.grad.abs().max()) == 0.0      # the measured failure
    assert float(p.g_matrix.grad.abs().max()) == 0.0

    q = _saturating_params()
    glut_forward(x, q, clamp="two", clamp_grad="st").sum().backward()
    assert float(q.g_bias.grad.abs().max()) > 0.0
    assert float(q.g_matrix.grad.abs().max()) > 0.0


#: fp32 inputs that separate ``x.clamp(0,1)`` from ``x + (clamp(x) - x).detach()``.
#: The arithmetic form is exact below 2**25 and wrong at and above it; ``inf``
#: turns into ``nan`` (``inf + (1 - inf)``).
_CLAMP_EXTREMES = (2.0 ** 24 - 1.0, 2.0 ** 24, 2.0 ** 25, 1e12, -1e12,
                   float("inf"), float("-inf"), float("nan"),
                   -0.5, 0.0, 0.25, 1.0, 1.5)


def test_the_st_clamp_forward_is_the_hard_clamp_bit_for_bit_on_every_input():
    x = torch.tensor(_CLAMP_EXTREMES, dtype=torch.float32)
    ref = x.clamp(0.0, 1.0)
    hard = _clamp01(x, "hard")
    st = _clamp01(x, "st")
    # nan never compares equal, so it is checked as a mask on both sides
    keep = ~torch.isnan(ref)
    for got in (hard, st):
        assert torch.equal(torch.isnan(got), torch.isnan(ref))
        assert torch.equal(got[keep], ref[keep])    # every bit, not allclose
    # the two inputs the arithmetic straight-through form used to get wrong
    assert float(_clamp01(torch.tensor(2.0 ** 25), "st")) == 1.0
    assert float(_clamp01(torch.tensor(1e12), "st")) == 1.0
    assert float(_clamp01(torch.tensor(float("inf")), "st")) == 1.0


@pytest.mark.parametrize("v", [2.0 ** 25, 1e12, 3.0, -0.5])
def test_the_st_clamp_backward_is_the_identity_where_the_hard_one_is_zero(v):
    xh = torch.tensor([v], requires_grad=True)
    _clamp01(xh, "hard").backward(torch.tensor([3.0]))
    assert float(xh.grad[0]) == 0.0                 # saturated -> exactly zero

    xs = torch.tensor([v], requires_grad=True)
    _clamp01(xs, "st").backward(torch.tensor([3.0]))
    assert float(xs.grad[0]) == 3.0                 # identity: the upstream grad


def test_the_carrier_module_carries_the_switch_and_defaults_to_hard():
    assert GlutCarrier().clamp_grad == "hard"           # EPR-024..029 unchanged
    c = GlutCarrier(clamp="two", clamp_grad="st")
    assert c.config == {"clamp": "two", "clamp_grad": "st", "residual": True,
                        "eps": EPS}
    with pytest.raises(ValueError, match="clamp_grad"):
        GlutCarrier(clamp_grad="soft")
    p = GlutParams.identity(4, batch=2)
    x = torch.rand(2, 16, 3)
    assert torch.equal(c(x, p), glut_forward(x, p, clamp="two", clamp_grad="st"))


# --------------------------------------------------------------------------- #
# (e) the L0 loss and its two assertions
# --------------------------------------------------------------------------- #
def test_l0_is_one_term_and_equals_the_l1_of_the_batch():
    L0.reset_l0_call_count()
    y_hat, y = torch.rand(3, 16, 3), torch.rand(3, 16, 3)
    out = L0.l0_losses(y_hat, y)
    assert torch.equal(out.total, (y_hat - y).abs().mean())
    assert float(out.l_hc) == 0.0 and float(out.l_sparse) == 0.0
    assert L0.l0_call_count() == 1
    row = out.row(n_luts_in_batch=2, mining_ratio_value=0.0)
    for col in ("L_rec", "L_hc", "L_sparse", "n_colors", "n_luts_in_batch",
                "mining_ratio", "n_hc_masked"):
        assert col in row, col


def test_the_purity_assertion_raises_when_an_extra_weight_is_on():
    assert L0.assert_l0_pure(lambda_hc=0.0, lambda_sparse=0.0, lambda_mono=0.0)
    for kw in (dict(lambda_hc=1.0), dict(lambda_sparse=1e-4), dict(lambda_mono=10.0)):
        base = dict(lambda_hc=0.0, lambda_sparse=0.0, lambda_mono=0.0)
        base.update(kw)
        with pytest.raises(AssertionError, match="single L1 term"):
            L0.assert_l0_pure(**base)


def test_the_purity_assertion_is_wired_into_the_production_call_site():
    """The function itself was already tested; the *call site* was not.

    ``run_setup_record`` is what the runner calls before step 0
    (``run_carrier_arm.main``, section 5), so this is the path a non-zero
    ``--lambda-*`` has to die on.  The former ``if pure_l1`` guard around the
    assertion was its own complement, i.e. a no-op for every input.
    """
    torch.manual_seed(0)
    cfg = tiny_cfg()
    check = E.run_setup_record(cfg, E.Epr030Model(cfg))["epr030"]["l0_purity_check"]
    assert "skipped" not in check
    assert check == {"lambda_hc": 0.0, "lambda_sparse": 0.0, "lambda_mono": 0.0}

    for kw in (dict(l0_lambda_hc=1.0), dict(l0_lambda_sparse=1e-4),
               dict(l0_lambda_mono=10.0)):
        bad = tiny_cfg(**kw)
        assert not bad.l0_weights.pure_l1          # the branch that used to skip
        with pytest.raises(AssertionError, match="single L1 term"):
            E.run_setup_record(bad, E.Epr030Model(bad))


def test_the_call_counter_assertion_is_the_defined_but_not_wired_guard():
    L0.reset_l0_call_count()
    with pytest.raises(L0.L0NotCalled):
        L0.assert_l0_ran(where="quick_eval@step2936")
    L0.l0_losses(torch.rand(1, 4, 3), torch.rand(1, 4, 3))
    assert L0.assert_l0_ran() == 1


def test_the_optional_terms_are_implemented_and_additive():
    L0.reset_l0_call_count()
    y_hat, y = torch.rand(2, 32, 3), torch.rand(2, 32, 3)
    opa = torch.rand(2, 6).clamp(0.05, 0.95)
    base = L0.l0_losses(y_hat, y).total
    hc = L0.l0_losses(y_hat, y, weights=L0.L0Weights(lambda_hc=1.0))
    assert float(hc.total) != float(base) and float(hc.l_hc) != 0.0
    sp = L0.l0_losses(y_hat, y, opacity=opa,
                      weights=L0.L0Weights(lambda_sparse=1e-4))
    assert float(sp.total) == pytest.approx(float(base) + 1e-4 * float(sp.l_sparse))
    with pytest.raises(ValueError, match="opacity"):
        L0.l0_losses(y_hat, y, weights=L0.L0Weights(lambda_sparse=1e-4))
    with pytest.raises(ValueError, match="grid values"):
        L0.l0_losses(y_hat, y, weights=L0.L0Weights(lambda_mono=10.0))


def test_the_monotonicity_hinge_is_zero_on_a_monotone_map_and_positive_otherwise():
    n = 5
    t = torch.linspace(0.0, 1.0, n)
    grid = torch.stack(torch.meshgrid(t, t, t, indexing="ij"), dim=-1).reshape(1, -1, 3)
    assert float(L0.monotonicity_hinge(grid, n)) == 0.0
    assert float(L0.monotonicity_hinge(1.0 - grid, n)) > 0.0


def test_the_c_normalisation_divides_by_a_detached_mean():
    torch.manual_seed(0)
    y_hat = torch.rand(2, 64, 3, requires_grad=True)
    y = torch.rand(2, 64, 3)
    plain, _ = L0.hue_chroma_term(y_hat, y, cnorm=False)
    normed, _ = L0.hue_chroma_term(y_hat, y, cnorm=True)
    assert float(plain) != float(normed)
    normed.backward()
    assert torch.isfinite(y_hat.grad).all()


def test_grad_norm_shares_sum_to_one_and_change_no_weight():
    torch.manual_seed(0)
    lin = torch.nn.Linear(3, 3)
    y_hat = lin(torch.rand(2, 16, 3))
    y = torch.rand(2, 16, 3)
    out = L0.l0_losses(y_hat, y, opacity=torch.rand(2, 4).clamp(0.05, 0.95),
                       weights=L0.L0Weights(lambda_hc=1.0))
    shares = L0.term_grad_norm_shares(out.terms, list(lin.parameters()))
    assert set(shares) == {"L_rec", "L_hc"}
    assert sum(v["share"] for v in shares.values()) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# the arm: flags, wiring, one real step
# --------------------------------------------------------------------------- #
def test_the_flag_surface_is_the_task_card_s():
    ap = E.build_parser()
    flags = {a.option_strings[0] for a in ap._actions if a.option_strings}
    for want in ("--qdec-dim", "--qdec-layers", "--qdec-heads", "--qdec-self-attn",
                 "--qdec-mem-rows", "--qdec-act", "--head-init", "--clamp-grad",
                 "--loss", "--lambda-hc", "--lambda-sparse", "--lambda-mono",
                 "--backbone", "--diag-every", "--qdec-head-lr-scale",
                 # the inherited EPR-024 surface must survive
                 "--readout", "--n-gauss", "--clamp", "--batch-split", "--context"):
        assert want in flags, want
    cfg = E.config_from_args(ap.parse_args(["--zcache-root", "/tmp/x"]))
    assert (cfg.backbone, cfg.qdec_dim, cfg.qdec_layers, cfg.qdec_heads) == \
        ("qdec", 512, 6, 8)
    assert (cfg.qdec_mem_rows, cfg.qdec_act, cfg.head_init) == (1, "gelu", "bias")
    assert (cfg.clamp_grad, cfg.loss) == ("st", "l0")
    assert (cfg.qdec_prior_lr_scale, cfg.qdec_head_lr_scale) == (0.1, 0.1)
    assert cfg.lambda_hc == 0.0 and cfg.lambda_sparse == 0.0 and cfg.lambda_mono == 0.0
    assert cfg.l0_weights.pure_l1


def test_the_frozen_block_survives_the_subclass():
    import math as _math

    from q3vl.whatb import splits as _S

    n = _S.active_dataset_version().train_normal_n
    cfg = E.Epr030Config(loss_level=1)
    assert cfg.train_n == n
    assert (cfg.batch_samples, cfg.queries_per_sample) == (32, 256)
    assert cfg.steps_per_epoch == _math.ceil(n / 32)
    assert cfg.total_steps == cfg.steps_per_epoch * 40
    assert cfg.clamp == "two" and cfg.hc_eps == 1e-3
    with pytest.raises(ValueError, match="ladder at level 1"):
        E.Epr030Config(loss_level=3)
    with pytest.raises(ValueError, match="backbone"):
        E.Epr030Config(loss_level=1, backbone="transformer")


def test_the_two_backbones_are_the_only_difference_between_the_rows():
    torch.manual_seed(0)
    q = E.Epr030Model(tiny_cfg(backbone="qdec"))
    m = E.Epr030Model(tiny_cfg(backbone="mlp"))
    assert q.pi is None and q.generator is None and q.qdec is not None
    assert m.qdec is None and m.pi is not None
    assert isinstance(m.generator, type(A.CarrierModel(tiny_cfg()).generator))
    # both carry the same carrier configuration, including the clamp gradient
    assert q.carrier.config == m.carrier.config
    assert q.carrier.clamp_grad == "st"
    z = _z(3)
    for model in (q, m):
        p = model.params_for(z)
        assert p.batch_size == 3 and p.n_gauss == model.cfg.n_gauss
        assert model.params_from_condition(model.condition(z)).mu.shape == p.mu.shape


def test_one_real_training_step_runs_the_l0_loss_and_the_diagnostics(bank):
    torch.manual_seed(0)
    L0.reset_l0_call_count()
    cfg = tiny_cfg(max_steps=2, diag_every=1, mining=False)
    model = E.Epr030Model(cfg)
    opt = E.build_optimizer(model, cfg)
    sched = E.build_scheduler(opt, cfg)
    row = E.train_step(model, cfg, opt, sched, step=0, z=_z(2),
                       lut_ids=["lut_a", "lut_b"], bank=bank,
                       sampler=QuerySampler(seed=cfg.seed, q=8))
    assert L0.l0_call_count() == 1
    assert row["L_hc"] == 0.0 and row["L_sparse"] == 0.0
    assert row["loss"] == pytest.approx(row["L_rec"])
    # the diagnostics of the task card, section D
    assert row["grad_shares"]["L_rec"]["share"] == pytest.approx(1.0)
    assert 0.0 <= row["clamp_sat_final"] <= 1.0
    assert 0.0 <= row["clamp_sat_global"] <= 1.0
    assert "act_nearzero_l0" in row and "act_dead_l1" in row
    for key in ("point_std", "identity_dev", "cross_std"):
        assert key in row
    assert row["gnorm"] > 0.0, "the whole backbone got an exactly zero gradient"


def test_the_quick_eval_asserts_the_l0_loss_actually_ran(bank):
    torch.manual_seed(0)
    L0.reset_l0_call_count()
    cfg = tiny_cfg()
    model = E.Epr030Model(cfg)
    # a live (non-degenerate) transform, so the shared degeneracy guard passes and
    # the counter check is the one under test
    with torch.no_grad():
        model.qdec.head_gauss.weight.normal_(0.0, 0.05)
        model.qdec.head_global.weight.normal_(0.0, 0.05)
    z, ids = _z(4), ["lut_a", "lut_b"] * 2
    with pytest.raises(L0.L0NotCalled):
        E.quick_eval(model, cfg, z=z, lut_ids=ids, bank=bank, step=2936, first=True,
                     exit_process=False)
    L0.l0_losses(torch.rand(1, 4, 3), torch.rand(1, 4, 3))
    row = E.quick_eval(model, cfg, z=z, lut_ids=ids, bank=bank, step=2936, first=True,
                       exit_process=False)
    assert row["l0_calls"] >= 1
    # the --eval-only tail call (step < 0) must NOT assert the counter
    L0.reset_l0_call_count()
    assert "l0_calls" not in E.quick_eval(model, cfg, z=z, lut_ids=ids, bank=bank,
                                         step=-1, first=True, exit_process=False)


def test_the_run_record_freezes_the_new_sources_and_the_step0_witness():
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = E.Epr030Model(cfg)
    rec = E.run_setup_record(cfg, model)
    assert rec["arm"] == "EPR-030" and rec["arm_name"] == "QDEC"
    for name in ("qdecoder.py", "losses_l0.py", "glut.py", "run_epr030_arm.py"):
        assert rec["source_sha256"][name], name
    assert rec["epr030"]["step0"]["step0_identity_asserted"] == 1.0
    assert rec["epr030"]["weights"]["pure_l1"] is True
    assert rec["epr030"]["clamp_grad"] == "st"
    assert rec["frozen_block"]["train_normal_only_n"] == cfg.train_n
    groups = {g["name"]: g for g in rec["epr030"]["optimizer_groups"]}
    assert groups["qdec_query_prior"]["lr"] == pytest.approx(cfg.base_lr * 0.1)
    # the output head sits in the 0.1x group too: its bias IS the shared geometry
    # (Bias-HyperInit).  Measured basis for the default: head at 1.0x reaches NaN
    # by step 80 on CPU with real z + real bank; at 0.1x L_rec stays at 0.157-0.168.
    assert groups["qdec_head"]["lr"] == pytest.approx(cfg.base_lr * 0.1)
    assert groups["qdec"]["lr"] == pytest.approx(cfg.base_lr)
    n_total = sum(g["n_params"] for g in rec["epr030"]["optimizer_groups"])
    assert n_total == sum(p.numel() for p in model.qdec.parameters())
    pre = E.loss_preregistration(cfg)
    assert [t["active"] for t in pre["terms"]] == [True, False, False, False]
    assert pre["arm"] == "EPR-030"


def test_the_arm_publishes_under_its_own_name_with_the_p1_table():
    assert E.AXES == ("P1",)
    from q3vl.whatb.criteria import required_criteria
    req = set(required_criteria(E.ARM, E.AXES))
    assert {"headline_normal_only", "B0_identity", "B3_bucket_retrieval",
            "N3_const_M", "interp_grid", "mono_rate"} <= req


def test_no_banned_column_and_no_contaminated_import():
    import ast

    for path in (Path(__file__).parent.parent / "qdecoder.py",
                 Path(__file__).parent.parent / "losses_l0.py", RUNNER_PY):
        src = path.read_text(encoding="utf-8")
        defs = re.findall(r"^\s*def\s+(\w+)", src, re.MULTILINE)
        banned = ("auc", "roc", "minmax", "min_max", "softmax_norm", "iou")
        assert not [d for d in defs if any(b in d.lower() for b in banned)], path
        assert "headline_pooled" not in src
        names: list[str] = []
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        assert not [n for n in names
                    if n.startswith(("q3vl.what.", "model.glut_repro", "gpu_render",
                                     "trash")) or n == "q3vl.what"], path
