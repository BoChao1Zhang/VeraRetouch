"""IDGATE (EPR-027) unit tests -- CPU only, no GPU process is started.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests/test_idgate.py -q

Four groups:

* **the proposal's numbers** -- flags, defaults, parameter counts, the frozen batch
  organisation (8192 / 2936 / 117,440) and the twenty-six pre-registered criterion keys;
* **the five identities** G1 / G1' / G2 / G3 / G4 / G5 and proposition 2, each measured
  rather than asserted in prose;
* **the three where-side failures** -- the degeneracy guard exits the process, the
  publication gate tells "no row" from "no loss columns", and no forward builds a bare
  ``torch.tensor`` or moves a metric to the CPU;
* **the loss and the optimiser** -- L_rec / L_hc (mask included) / R_sparse against
  hand-computed values, the two lr groups and the cosine schedule.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import criteria as _criteria
from q3vl.whatb import guards as _guards
from q3vl.whatb import queries as _queries
from q3vl.whatb.arms import idgate as ig
from q3vl.whatb.colorimetry import chroma_hue, srgb_to_lab
from q3vl.whatb.gate import identity_gate
from q3vl.whatb.glut import EPS, GlutParams, glut_forward
from q3vl.whatb.lutdata import BANK_DIR, LutBank, mix_alpha
from q3vl.whatb.scripts import run_idgate_arm as runner

ARM_FILE = Path(ig.__file__)
RUNNER_FILE = Path(runner.__file__)

torch.set_num_threads(min(4, torch.get_num_threads()))


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def tiny_cfg(**kw):
    """A configuration small enough for a loaded CPU box; frozen values overridden."""
    base = dict(batch_samples=2, queries=16, train_n=8, epochs=1, gate_ku=2,
                grid_n=4, floor_grid_n=4, b1_grid_n=5, hist_top_k=32, n_repeats=2)
    base.update(kw)
    return ig.IdGateConfig(**base)


@pytest.fixture(scope="module")
def bank():
    if not (BANK_DIR / "luts_meta.json").is_file():
        pytest.skip(f"LUT bank {BANK_DIR} is not mounted")
    return LutBank()


@pytest.fixture()
def model():
    torch.manual_seed(20260810)
    return ig.IdGateArm(tiny_cfg())


@pytest.fixture()
def queries():
    torch.manual_seed(3)
    return torch.rand(2, 24, 3)


# --------------------------------------------------------------------------- #
# 1. the proposal's numbers
# --------------------------------------------------------------------------- #
def test_defaults_are_the_proposals_main_arm():
    c = ig.IdGateConfig()
    assert (c.gate, c.gate_u_source, c.gate_clamp) == (True, "sample", "after")
    assert (c.gate_p_end, c.gate_ku) == (0.2, 4)
    assert (c.gate_u_dist, c.gate_lambda_sign) == ("uniform01", "fixed")
    assert c.gate_zhead == "none" and c.gate_zhead_weight == 0.1
    assert c.gate_field_src == "gt"
    assert c.gate_u_eval == (-0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
    assert c.null_prompt == "Please keep the colors of this photo unchanged."
    assert ig.NULL_PROMPTS["cliptone"] == "normal photo"
    # loss: GLUT section 4.1 verbatim
    assert (c.lambda_hc, c.lambda_sparse, c.hc_eps_c) == (10.0, 0.001, 1e-3)
    assert (c.l_rec_scale, c.chroma_weight_src, c.loss_level) == (1.0, "target", 3)
    # optimiser: GLUT section 4.1 + App A.1
    assert (c.lr, c.pi_lr_scale) == (1e-3, 0.1)
    assert c.adam_betas == (0.9, 0.999) and c.weight_decay == 0.0
    assert c.grad_clip is None            # EPR-027:449 -- clipping is NOT added
    assert (c.n_gauss, c.cond_dim, c.gen_width, c.clamp) == (48, 64, 128, "two")
    assert c.seed == 20260810 and c.mining is True


def test_frozen_batch_organisation():
    c = ig.IdGateConfig()
    assert c.colors_per_step == 8192 == 32 * 256
    assert c.steps_per_epoch == 2936 == math.ceil(93934 / 32)
    assert c.total_steps == 117_440 == 2936 * 40
    assert c.n_pairs_per_step == 8192 * 4     # NOTES 12: pairs enter the batch...
    assert ig.assert_frozen_organisation(c)["total_steps"] == 117_440
    # ... and the step count stays EPR-024's
    assert c.total_steps == ig.IdGateConfig(gate=False).total_steps


def test_frozen_organisation_records_a_capacity_row_instead_of_refusing():
    """EPR-030 (2026-08-16) opened the colour budget and the corpus.

    ``64x128`` is still step-matched to the EPR-024 board (same B * Q and the
    same B, so the same steps/epoch); ``256x8192`` on ``--data v2seg+l8`` is
    not, and the record says exactly which numbers differ.
    """
    matched = ig.assert_frozen_organisation(
        ig.IdGateConfig(batch_samples=64, queries=128))
    assert matched["colours_per_step"] == 8192
    assert matched["batch_split_step_matched_to_epr024"] is True
    assert matched["differs_from_epr024"] == {
        "batch_samples": [64, 32], "queries": [128, 256],
        "steps_per_epoch": [1468, 2936], "total_steps": [58720, 117440]}

    big = ig.assert_frozen_organisation(
        ig.IdGateConfig(batch_samples=256, queries=8192, train_n=119828,
                        data="v2seg+l8"))
    assert big["colours_per_step"] == 2_097_152
    assert big["steps_per_epoch"] == 469 and big["total_steps"] == 18_760
    assert big["batch_split_step_matched_to_epr024"] is False
    assert big["step_matched_to_epr024"] is False


def test_frozen_organisation_still_refuses_broken_arithmetic():
    """``steps_per_epoch != ceil(n / B)`` is arithmetic, not a caliber."""
    cfg = ig.IdGateConfig(batch_samples=256, queries=8192, train_n=119828)
    assert cfg.steps_per_epoch == 469
    with pytest.raises(AssertionError, match="steps_per_epoch"):
        ig.assert_steps_per_epoch(470, n_train=119828, batch_samples=256)


def test_loss_level_one_zeroes_both_optional_weights():
    base = ig.IdGateConfig()
    assert (base.lambda_hc_effective, base.lambda_sparse_effective) == (10.0, 0.001)
    pure = ig.IdGateConfig(loss_level=1)
    assert (pure.lambda_hc_effective, pure.lambda_sparse_effective) == (0.0, 0.0)


def test_param_counts():
    m = ig.IdGateArm(ig.IdGateConfig())
    c = m.param_counts
    assert c["pi"] == 169_024            # LayerNorm(2560) 5120 + Linear 163904
    assert c["generator"] == 278_188     # d=64, H=128, N=48, full generation
    assert c["gate"] == 0                # the gate itself has no parameters
    assert c["total"] == 447_212
    assert c["theta_dim"] == 22 * 48 + 12 == 1068
    probe = ig.IdGateArm(ig.IdGateConfig(gate_zhead="linear"))
    assert probe.param_counts["u_probe"] == 2561 == 2560 + 1
    assert probe.param_counts["total"] == 449_773


def test_required_criteria_is_the_union():
    req = set(ig.REQUIRED_CRITERIA)
    assert set(_criteria.PREREGISTERED_KEYS) <= req            # the twelve frozen keys
    assert _criteria.REQUIRED_P1 <= req                        # P1
    assert _criteria.REQUIRED_P2P3 <= req                      # P2
    assert set(ig.ARM_CRITERIA) <= req                         # this arm's four
    assert len(req) == 12 + 4 + 6 + 4
    assert ig.AXES == ("P1", "P2")


def test_zero_initialised_probe_starts_at_one_half():
    m = ig.IdGateArm(tiny_cfg(gate_zhead="linear"))
    assert torch.count_nonzero(m.u_probe.weight) == 0
    assert torch.count_nonzero(m.u_probe.bias) == 0
    u = m.predict_u(torch.randn(3, 2560))
    assert torch.allclose(u, torch.full_like(u, 0.5))


def test_gate_off_rejects_a_probe():
    with pytest.raises(ValueError, match="probe"):
        ig.IdGateConfig(gate=False, gate_zhead="linear")


# --------------------------------------------------------------------------- #
# 2. the gate, the identities, proposition 2
# --------------------------------------------------------------------------- #
def test_u_has_no_default_anywhere(model, queries):
    params = model.theta(torch.randn(2, 2560))
    with pytest.raises(TypeError, match="u was not given"):
        model.apply_transform(queries, params, u=None)
    with pytest.raises(TypeError):
        identity_gate(queries, queries)          # gate.py's own contract


def test_gate_identity_at_u_one(model, queries):
    params = model.theta(torch.randn(2, 2560))
    rep = ig.assert_gate_identity(model, queries, params)
    assert rep["passed"] and rep["max_abs"] <= rep["atol"]
    assert rep["atol_source"].startswith("4*eps")
    assert rep["max_abs"] < 4 * torch.finfo(torch.float32).eps


def test_literal_zero_tolerance_is_unreachable_in_float32():
    """EPR-027:461 writes ``== 0``; ``x + 1*(y - x)`` cannot deliver it.

    With ``x = 1.0`` and ``y = 1e-9`` (a pre-clamp GLUT value can be that small) the
    subtraction rounds to ``-1.0`` and the sum returns ``0.0``, so the residual is the
    whole of ``|y|``.  Recorded here so the tolerance in
    :meth:`IdGateConfig.identity_atol` is a measured fact, not a preference.
    """
    x = torch.tensor([1.0, 0.9], dtype=torch.float32)
    y = torch.tensor([1e-9, 1e-7], dtype=torch.float32)
    resid = (identity_gate(x, y, 1.0, clamp=False) - y).abs()
    assert float(resid[0]) == pytest.approx(1e-9, rel=1e-6)   # the whole value is lost
    assert float(resid[1]) > 0.0                              # and 1.92e-8 here
    assert float(resid.max()) < 4 * torch.finfo(torch.float32).eps
    # and with the flag set to the literal zero the arm refuses rather than rounds
    m = ig.IdGateArm(tiny_cfg(gate_identity_atol=0.0))
    assert m.cfg.identity_atol(torch.float32) == (0.0, "flag")


@torch.no_grad()
def test_G1_l_rec_is_linear_in_u(model, queries, bank):
    """``f_u - y_u = u (f_theta - L_l)``  =>  ``L_rec(u) = u L_rec(1)`` (pre-clamp)."""
    params = model.theta(torch.randn(2, 2560))
    lut = bank.apply(queries, bank.lut_ids()[0])
    ref = (model.apply_transform(queries, params, u=1.0).y_pre_clamp - lut).abs().mean()
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        got = (model.apply_transform(queries, params, u=u).y_pre_clamp
               - ig.strength_target(queries, lut, u)).abs().mean()
        assert float(got) == pytest.approx(u * float(ref), rel=1e-5, abs=1e-7)


def test_G1prime_chroma_weight_of_y_u_is_not_u_times_chroma():
    """``C(y_u) != u C(L_l)``: the row 1'' control cannot be replaced by a scalar."""
    x = torch.rand(64, 3)
    lut = torch.rand(64, 3)
    u = 0.5
    c_yu, _, _ = chroma_hue(srgb_to_lab(mix_alpha(x, lut, u)))
    c_l, _, _ = chroma_hue(srgb_to_lab(lut))
    assert not torch.allclose(c_yu, u * c_l, atol=1e-2)
    # and at u = 0 the weight is C(x), generally non-zero
    c_x, _, _ = chroma_hue(srgb_to_lab(x))
    assert float(c_x.mean()) > 0.0


def test_G2_magnitude_scales_with_u(model, queries):
    params = model.theta(torch.randn(2, 2560))
    base = (model.apply_transform(queries, params, u=1.0).y_pre_clamp - queries).norm(dim=-1)
    for u in (0.2, 0.6, 1.0):
        got = (model.apply_transform(queries, params, u=u).y_pre_clamp
               - queries).norm(dim=-1)
        assert torch.allclose(got, u * base, atol=1e-6)


@torch.no_grad()
def test_G3_u_zero_is_the_identity_and_a_zero_loss(model, queries, bank):
    params = model.theta(torch.randn(2, 2560))
    out = model.apply_transform(queries, params, u=0.0)
    assert torch.equal(out.y, queries)                     # bit-exact, not approx
    lut = bank.apply(queries, bank.lut_ids()[0])
    terms = ig.glut_loss(out.y, ig.strength_target(queries, lut, 0.0), out.opacity)
    assert float(terms.l_rec) == 0.0
    assert float(terms.l_hc) == pytest.approx(0.0, abs=1e-5)


def test_G4_proposition3_the_gate_reproduces_the_data_law(bank, queries):
    """``u(p) = alpha(p)`` and ``f_theta = L_l``  =>  ``f_u = F*`` point-wise."""
    x = queries[0]
    lid = bank.lut_ids()[1]
    lut = bank.apply(x, lid)
    alpha = torch.rand(x.shape[0], 1)
    gated = identity_gate(x, lut, alpha, clamp=False)
    assert torch.allclose(gated, bank.f_star(x, alpha, lid), atol=1e-6)
    # alpha = 0 is bit-exact -- that is why E_out is 0 by construction
    zero = identity_gate(x, lut, torch.zeros_like(alpha), clamp=False)
    assert torch.equal(zero, x)


@torch.no_grad()
def test_G5_clamp_before_the_gate_kills_the_out_of_gamut_column(queries):
    torch.manual_seed(5)
    before = ig.IdGateArm(tiny_cfg(gate_clamp="before"))
    params = before.theta(torch.randn(2, 2560))
    for u in (0.0, 0.4, 1.0):
        out = before.apply_transform(queries, params, u=u)
        assert not bool(out.oob_mask().any())              # convex combination
        assert float(out.y.min()) >= 0.0 and float(out.y.max()) <= 1.0
    # the main arm (clamp after) leaves the column alive: an untrained head does leave
    # the gamut, and the pre-clamp value is what A(u) reads
    after = ig.IdGateArm(tiny_cfg(gate_clamp="after"))
    p2 = after.theta(torch.randn(2, 2560))
    assert bool(after.apply_transform(queries, p2, u=1.0).oob_mask().any())


def test_G5_extrapolated_u_is_out_of_gamut_in_both_clamp_modes(queries):
    torch.manual_seed(6)
    m = ig.IdGateArm(tiny_cfg(gate_clamp="before"))
    params = m.theta(torch.randn(2, 2560))
    assert bool(m.apply_transform(queries, params, u=2.0).oob_mask().any())


def test_proposition2_identity_is_an_exact_point_of_the_parameter_space(queries):
    """``M=I, b=0, G=0, g=0``: ``f_theta(x) = (1 - delta(x)) x`` with
    ``delta = eps / (sum_j p_j o_j + eps)`` (HANDOFF section 2.3)."""
    x = queries[0].double()
    ident = GlutParams.identity(48, dtype=torch.float64)
    exact = glut_forward(x, ident, clamp="none", eps=0.0)
    assert float((exact[0] - x).abs().max()) < 1e-12
    with_eps = glut_forward(x, ident, clamp="none", eps=EPS)
    assert float((with_eps[0] - x).abs().max()) < 1e-5
    # through the gate at u = 1 it is the same object
    m = ig.IdGateArm(tiny_cfg())
    gated = m.apply_transform(x.float(), ident.to(dtype=torch.float32), u=1.0)
    assert float((gated.y[0] - x.float()).abs().max()) < 1e-5


def test_u_field_broadcasts_over_pixels(model, bank):
    torch.manual_seed(7)
    img = torch.rand(3, 8, 10)
    params = model.theta(torch.randn(1, 2560))
    field = torch.rand(8, 10)
    out = model.apply_image(img, params, u=field)
    assert out.y.shape == img.shape
    zero_field = model.apply_image(img, params, u=torch.zeros(8, 10))
    assert torch.equal(zero_field.y, img)          # u = 0 everywhere -> untouched


def test_lambda_path_is_a_bit_exact_bypass_at_one(model):
    z = torch.randn(3, 2560)
    z_null = torch.randn(3, 2560)
    assert torch.equal(model.z_lambda(z, z_null, 1.0), z)
    assert torch.equal(model.z_lambda(z, None, 0.3), z)
    half = model.z_lambda(z, z_null, 0.5)
    assert torch.allclose(half, z_null + 0.5 * (z - z_null))


# --------------------------------------------------------------------------- #
# 3. the u sampler
# --------------------------------------------------------------------------- #
def test_sample_u_shape_endpoints_and_p_end():
    g = torch.Generator().manual_seed(11)
    d = ig.sample_u(400, 8, generator=g, p_end=0.2)
    assert d.u.shape == (400, 8)
    assert float(d.u.min()) >= 0.0 and float(d.u.max()) <= 1.0
    n = d.u.numel()
    assert d.n_forced / n == pytest.approx(0.2, abs=0.02)
    assert d.n_zero / max(1, d.n_forced) == pytest.approx(0.5, abs=0.05)
    assert d.n_zero == int((d.u == 0.0).sum()) and d.n_one == int((d.u == 1.0).sum())
    st = d.stats
    assert st["u0_frac"] == pytest.approx(d.n_zero / n)
    assert st["n_u"] == n


def test_sample_u_excl_band_has_no_zero_draws():
    g = torch.Generator().manual_seed(12)
    d = ig.sample_u(500, 4, generator=g, p_end=0.2, dist="excl_band")
    assert float(d.u.min()) >= 0.1        # U([0,1] \\ (0, 0.1))
    assert d.stats["u0_frac"] == 0.0      # ablation row 6 pre-registers this
    assert d.n_forced == 0


def test_sample_u_does_not_touch_the_global_rng_stream():
    torch.manual_seed(99)
    a = torch.rand(4)
    torch.manual_seed(99)
    ig.sample_u(64, 4, generator=torch.Generator().manual_seed(1), p_end=0.2)
    assert torch.equal(a, torch.rand(4))


def test_sample_u_is_reproducible_from_its_seed():
    u1 = ig.sample_u(8, 4, generator=torch.Generator().manual_seed(5), p_end=0.2).u
    u2 = ig.sample_u(8, 4, generator=torch.Generator().manual_seed(5), p_end=0.2).u
    assert torch.equal(u1, u2)


# --------------------------------------------------------------------------- #
# 4. the loss
# --------------------------------------------------------------------------- #
def test_loss_terms_against_hand_computed_values():
    torch.manual_seed(13)
    y_hat = torch.rand(2, 32, 3)
    y = torch.rand(2, 32, 3)
    o = torch.rand(2, 48).clamp(0.01, 0.99)
    t = ig.glut_loss(y_hat, y, o)

    assert float(t.l_rec) == pytest.approx(float((y_hat - y).abs().mean()))
    ent = o * torch.log(o + EPS) + (1 - o) * torch.log(1 - o + EPS)
    assert float(t.l_sparse) == pytest.approx(float(-ent.mean(dim=-1).mean()), rel=1e-6)
    c, h, valid = chroma_hue(srgb_to_lab(y), 1e-3)
    _, h_hat, _ = chroma_hue(srgb_to_lab(y_hat), 1e-3)
    manual = (c * (1 - (h_hat * h).sum(-1)))[valid].mean()
    assert float(t.l_hc) == pytest.approx(float(manual), rel=1e-6)
    assert float(t.total) == pytest.approx(float(t.l_rec + 10 * t.l_hc
                                                 + 0.001 * t.l_sparse), rel=1e-6)


def test_hc_mask_counts_the_achromatic_points():
    grey = torch.full((1, 5, 3), 0.5)             # a = b = 0 -> C = 0 -> masked
    y_hat = torch.rand(1, 5, 3)
    o = torch.full((1, 4), 0.9)
    t = ig.glut_loss(y_hat, grey, o, hc_mask=True)
    assert t.n_hc_masked == 5 and t.n_hc_points == 5
    assert float(t.l_hc) == 0.0
    off = ig.glut_loss(y_hat, grey, o, hc_mask=False)
    assert off.n_hc_masked == 0


def test_l_rec_scale_is_row_one_prime():
    torch.manual_seed(14)
    y_hat, y = torch.rand(1, 8, 3), torch.rand(1, 8, 3)
    o = torch.full((1, 4), 0.8)
    full = ig.glut_loss(y_hat, y, o)
    half = ig.glut_loss(y_hat, y, o, l_rec_scale=0.5)
    # the totals are dominated by 10*L_hc, so the difference of two float32 sums
    # cancels about four digits away; the tolerance is that cancellation, not slack
    assert float(half.total - full.total) == pytest.approx(-0.5 * float(full.l_rec),
                                                           rel=1e-3)
    assert float(half.l_rec) == float(full.l_rec)      # the term itself is unscaled


def test_chroma_ref_changes_only_the_weight():
    torch.manual_seed(15)
    y_hat, y = torch.rand(1, 16, 3), torch.rand(1, 16, 3)
    ref = torch.rand(1, 16, 3)
    o = torch.full((1, 4), 0.8)
    a = ig.glut_loss(y_hat, y, o)
    b = ig.glut_loss(y_hat, y, o, chroma_ref=ref)
    assert b.chroma_weight_src == "y_u" and a.chroma_weight_src == "target"
    assert float(a.l_hc) != float(b.l_hc)
    assert float(a.l_rec) == float(b.l_rec)
    assert float(a.mean_c_weight) != float(b.mean_c_weight)


def test_gate_probe_loss_is_l1():
    u = torch.tensor([0.1, 0.9])
    p = torch.tensor([[0.3], [0.5]])
    assert float(ig.gate_probe_loss(p, u)) == pytest.approx((0.2 + 0.4) / 2, rel=1e-6)


# --------------------------------------------------------------------------- #
# 5. mining, optimiser, schedule
# --------------------------------------------------------------------------- #
def test_mining_ratio_follows_glut_app_a1():
    assert _queries.mining_ratio(0) == 0.10
    assert _queries.mining_ratio(5) == 0.10
    assert _queries.mining_ratio(12.5) == pytest.approx(0.25)
    assert _queries.mining_ratio(20) == 0.40
    assert _queries.mining_ratio(39) == 0.40


def test_mining_keeps_the_colour_budget(model, bank):
    torch.manual_seed(16)
    x = torch.rand(2, 16, 3)
    params = model.theta(torch.randn(2, 2560))
    lut = bank.apply(x, bank.lut_ids()[0])
    u = ig.sample_u(2, 2, generator=torch.Generator().manual_seed(1), p_end=0.0).u
    sampler = _queries.QuerySampler(seed=1, q=16)
    mined, n_hard = ig.mine_hard_colors(model, params, x, lut, u, ratio=0.25,
                                        sampler=sampler)
    assert mined.shape == x.shape                  # B*Q stays 8192 in the real config
    assert n_hard == 2 * 4                         # round(0.25 * 16) per sample
    zero, n0 = ig.mine_hard_colors(model, params, x, lut, u, ratio=0.0, sampler=sampler)
    assert n0 == 0 and torch.equal(zero, x)


def test_optimizer_groups_and_cosine_schedule():
    m = ig.IdGateArm(ig.IdGateConfig(gate_zhead="linear"))
    opt = ig.build_optimizer(m)
    by_name = {g["name"]: g for g in opt.param_groups}
    assert by_name["generator"]["lr"] == 1e-3
    assert by_name["condition_side"]["lr"] == pytest.approx(1e-4)   # 0.1x, App A.1
    n_slow = sum(p.numel() for p in by_name["condition_side"]["params"])
    assert n_slow == 169_024 + 2561           # pi + the probe, both condition-side
    sch = ig.build_scheduler(opt, 100)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3)
    for _ in range(100):
        opt.step()
        sch.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)
    assert opt.defaults["betas"] == (0.9, 0.999) and opt.defaults["weight_decay"] == 0.0


# --------------------------------------------------------------------------- #
# 6. the training step and its columns
# --------------------------------------------------------------------------- #
def _step(model, bank, **kw):
    torch.manual_seed(17)
    cfg = model.cfg
    ids = bank.lut_ids()[:cfg.batch_samples]
    x = torch.rand(cfg.batch_samples, cfg.queries, 3)
    u = ig.sample_u(cfg.batch_samples, cfg.k_u,
                    generator=torch.Generator().manual_seed(2), p_end=cfg.gate_p_end)
    return ig.train_step(model, z=torch.randn(cfg.batch_samples, 2560),
                         lut_values_fn=lambda xx: torch.stack(
                             [bank.apply(xx[i], ids[i]) for i in range(len(ids))]),
                         lut_ids=ids, x=x, u_draw=u, epoch=0.0,
                         sampler=_queries.QuerySampler(seed=1, q=cfg.queries), **kw)


def test_train_step_publishes_every_pre_registered_column(model, bank):
    loss, row = _step(model, bank, mining_ratio=0.10)
    assert set(ig.step_columns(model.cfg)) <= set(row)
    assert row["n_colors"] == model.cfg.batch_samples * model.cfg.queries
    assert row["n_pairs_per_step"] == row["n_colors"] * model.cfg.k_u
    assert row["n_luts_in_batch"] == model.cfg.batch_samples
    assert row["chroma_weight_src"] == "y_u"   # the gated target IS y_u
    assert 0.0 <= row["u0_frac"] <= 1.0
    loss.backward()
    assert any(p.grad is not None for p in model.generator.parameters())


def test_train_step_with_the_probe_adds_L_gate(bank):
    m = ig.IdGateArm(tiny_cfg(gate_zhead="linear"))
    _, row = _step(m, bank, mining_ratio=0.0)
    assert "L_gate" in row and "gate_u_mae" in row
    assert set(ig.step_columns(m.cfg)) <= set(row)
    # the probe starts at 0.5, so the MAE is |0.5 - u| averaged over the draws
    assert row["gate_u_mae"] == pytest.approx(row["L_gate"], rel=1e-6)


def test_row_one_is_the_epr024_shape(bank):
    """``--no-gate``: no u reaches the graph and the target is L_l(x) itself."""
    m = ig.IdGateArm(tiny_cfg(gate=False))
    _, row = _step(m, bank, mining_ratio=0.0)
    assert row["n_pairs_per_step"] == row["n_colors"]     # K_u collapses to 1
    assert m.cfg.k_u == 1
    assert row["chroma_weight_src"] == "gt_lut"


def test_row_one_double_prime_reweights_only_the_chroma(bank):
    m = ig.IdGateArm(tiny_cfg(gate=False, chroma_weight_src="y_u"))
    _, row = _step(m, bank, mining_ratio=0.0)
    assert row["chroma_weight_src"] == "y_u"
    base = ig.IdGateArm(tiny_cfg(gate=False))
    base.load_state_dict(m.state_dict())
    _, row0 = _step(base, bank, mining_ratio=0.0)
    assert row["L_rec"] == pytest.approx(row0["L_rec"], rel=1e-6)   # unchanged
    assert row["mean_C_weight"] != row0["mean_C_weight"]            # the weight moved


# --------------------------------------------------------------------------- #
# 7. the three where-side failures
# --------------------------------------------------------------------------- #
def test_degeneracy_guard_exits_on_a_constant_transform():
    """A generator whose heads are all zero emits one colour for every input."""
    torch.manual_seed(18)
    m = ig.IdGateArm(tiny_cfg())
    for head in (m.generator.head_color, m.generator.head_global, m.generator.head_mu,
                 m.generator.head_cov, m.generator.head_opacity):
        torch.nn.init.zeros_(head[-1].weight)
        torch.nn.init.zeros_(head[-1].bias)
    with pytest.raises(SystemExit) as exc:
        ig.first_quick_eval_guard(m, torch.randn(4, 2560), torch.rand(32, 3))
    assert exc.value.code == 2


def test_degeneracy_guard_exits_on_the_identity_solution(monkeypatch):
    m = ig.IdGateArm(tiny_cfg())
    ident = GlutParams.identity(m.cfg.n_gauss, batch=4)
    monkeypatch.setattr(m, "theta", lambda *a, **k: ident)
    with pytest.raises(SystemExit):
        ig.first_quick_eval_guard(m, torch.randn(4, 2560), torch.rand(32, 3))


def test_degeneracy_guard_exits_when_every_sample_shares_one_transform(monkeypatch):
    torch.manual_seed(19)
    m = ig.IdGateArm(tiny_cfg())
    one = m.theta(torch.randn(1, 2560)).expand_batch(4)
    monkeypatch.setattr(m, "theta", lambda *a, **k: one)
    with pytest.raises(SystemExit):
        ig.first_quick_eval_guard(m, torch.randn(4, 2560), torch.rand(32, 3))


def test_degeneracy_guard_passes_a_live_head_and_reports_three_numbers():
    torch.manual_seed(20)
    m = ig.IdGateArm(tiny_cfg())
    rep = ig.first_quick_eval_guard(m, torch.randn(4, 2560), torch.rand(32, 3),
                                    exit_process=False)
    assert rep["degeneracy"]["failures"] == []
    for key in ("point_std", "identity_dev", "cross_std"):
        assert rep["degeneracy"][key] > 0
    assert rep["gate_identity_check"]["passed"]


def test_publication_gate_distinguishes_no_row_from_no_columns():
    _guards.clear_step_witness()
    cfg = tiny_cfg()
    board = _fake_board(cfg)
    with pytest.raises(_guards.StepsRowUnavailable):
        ig.publish_arm_board(board, cfg=cfg)                       # nobody handed a row
    with pytest.raises(_guards.LossColumnsMissing):
        ig.publish_arm_board(board, cfg=cfg, steps_row={"step": 0})  # a row, no columns
    assert not issubclass(_guards.LossColumnsMissing, _guards.StepsRowUnavailable)
    assert not issubclass(_guards.StepsRowUnavailable, _guards.LossColumnsMissing)


def test_publication_gate_finds_the_row_through_the_witness_tier():
    cfg = tiny_cfg()
    _guards.clear_step_witness()
    ig.record_first_step({c: 0 for c in ig.step_columns(cfg)})
    rep = ig.publish_arm_board(_fake_board(cfg), cfg=cfg)
    assert rep["steps"]["source"] == "witness"
    _guards.clear_step_witness()


def test_publication_gate_refuses_a_board_with_a_missing_column():
    cfg = tiny_cfg()
    board = _fake_board(cfg)
    del board["criteria_columns"]["dlib_u"]
    with pytest.raises(_criteria.CriterionNotComputed, match="dlib_u"):
        ig.publish_arm_board(board, cfg=cfg,
                             steps_row={c: 0 for c in ig.step_columns(cfg)})


def test_probe_run_adds_L_gate_to_the_required_step_columns():
    cfg = tiny_cfg(gate_zhead="linear")
    assert "L_gate" in ig.step_columns(cfg)
    board = _fake_board(cfg)
    row = {c: 0 for c in ig.step_columns(cfg)}
    del row["L_gate"]
    with pytest.raises(_guards.LossColumnsMissing, match="L_gate"):
        ig.publish_arm_board(board, cfg=cfg, steps_row=row)


def _fake_board(cfg):
    """A board whose columns all exist with n > 0 -- the publication gate's happy path."""
    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "style" if i % 2 else "local", "E_arm": 1.0 + i,
             "E_B0_identity": 2.0, "E_B1_libmean": 2.0, "E_B2_librandom": 2.0,
             "E_B3_bucket_retrieval": 2.0, "E_B4_oracle": 1.0,
             "E_N1_shuffle": 1.5, "M_N1_shuffle": 0.5,
             "E_N2_irrelevant": 1.5, "M_N2_irrelevant": 0.5,
             "E_N3_const": 1.5, "M_N3_const": 0.5,
             "loc_in": 1.0, "loc_band": 1.0, "loc_out": 0.0,
             "field_gt": 1.0, "field_const": 1.0, "field_shuffle": 1.0}
            for i in range(4)]
    one = lambda: {"n": 4, "mean": 1.0}
    extra = ig.extra_criteria_columns(
        gate_identity={"n": 1, "max_abs": 0.0, "passed": True},
        u_histogram={"n": 100, "mean": 0.5},
        strength={"strength_dE_u": one(), "dlib_u": one()},
        interpolation={"interp_grid": one(), "path_len": one(), "mono_rate": one(),
                       "oob_rate": one()})
    return ig.build_arm_board(rows, split="V_what", extra_columns=extra, cfg=cfg,
                              published=True)


# --------------------------------------------------------------------------- #
# 8. discipline: no bare tensors in a forward, no .cpu() on a metric path
# --------------------------------------------------------------------------- #
FORWARD_FUNCS = ("forward", "apply_transform", "apply_image", "theta", "predict_u",
                 "z_lambda", "train_step", "glut_loss", "gate_probe_loss")


def _funcs(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def test_no_bare_torch_tensor_construction_in_a_forward_path():
    offences = []
    for fn in _funcs(ARM_FILE):
        if fn.name not in FORWARD_FUNCS:
            continue
        for node in ast.walk(fn):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in ("tensor", "zeros", "ones", "eye", "full",
                                           "arange", "linspace")
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "torch"):
                offences.append(f"{fn.name}:{node.lineno} torch.{node.func.attr}")
    assert offences == [], ("constants must be module buffers, not built in a forward: "
                            + "; ".join(offences))


def test_no_cpu_round_trip_on_a_metric_path():
    """``.cpu()`` may appear only in ``u_histogram`` -- bookkeeping, not a criterion.

    Every comparison, ``topk`` and reduction must run where the tensor already is; the
    where side moved an IoU by 0.296 by breaking ties on the wrong device.
    """
    offenders = set()
    for fn in _funcs(ARM_FILE):
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr == "cpu":
                    offenders.add(fn.name)
                if node.func.attr == "to" and any(
                        isinstance(a, ast.Constant) and a.value == "cpu"
                        for a in node.args):
                    offenders.add(fn.name)
    assert offenders <= {"u_histogram"}, sorted(offenders)


def test_no_banned_criterion_is_implemented_here():
    names = {fn.name.lower() for fn in _funcs(ARM_FILE)}
    for banned in ("auc", "roc", "minmax", "min_max", "trimmed", "iou"):
        assert not any(banned in n for n in names), f"{banned} must not exist"


def test_the_arm_does_not_import_the_contaminated_tree():
    for path in (ARM_FILE, RUNNER_FILE):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        mods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.append(node.module)
        for m in mods:
            assert not (m == "q3vl.what" or m.startswith("q3vl.what.")), m
            assert not m.startswith(("model.glut_repro", "gpu_render", "trash")), m


def test_device_and_dtype_are_placed_explicitly():
    """float64 model + float32 queries must cast, never crash (the MATTE failure)."""
    m = ig.IdGateArm(tiny_cfg()).double()
    x = torch.rand(2, 8, 3, dtype=torch.float32)
    out = m(x, torch.randn(2, 2560, dtype=torch.float32), u=0.5)
    assert out.y.dtype == torch.float64
    m32 = ig.IdGateArm(tiny_cfg())
    out32 = m32(torch.rand(2, 8, 3, dtype=torch.float64),
                torch.randn(2, 2560, dtype=torch.float64), u=0.5)
    assert out32.y.dtype == torch.float64        # promoted, never truncated


# --------------------------------------------------------------------------- #
# 9. evaluation, the board and the runner
# --------------------------------------------------------------------------- #
@pytest.fixture()
def eval_kit(bank):
    torch.manual_seed(21)
    cfg = tiny_cfg()
    m = ig.IdGateArm(cfg)
    grids = ig.GridSpec.build(cfg, n_heldout=32)
    ids = bank.lut_ids()[:3]
    lib = ig.LibraryContext.build(bank, ids, grids.floor_grid,
                                  bucket_pools={"b0": list(ids)},
                                  mean_grid_n=cfg.b1_grid_n)
    mk = lambda i: ig.EvalSample(
        sample_id=f"s{i}", winner_confidence="normal",
        task_type="style" if i % 2 else "local", lut_id=ids[i % len(ids)],
        image=torch.rand(3, 6, 8), alpha=torch.rand(6, 8), z=torch.randn(2560),
        z_ctrl={k: torch.randn(2560) for k in ("N1_shuffle", "N2_irrelevant",
                                               "N3_const")},
        alpha_pred=torch.rand(6, 8), alpha_shuffle=torch.rand(6, 8), minor="b0",
        source_image_id="src0")
    return cfg, m, grids, lib, [mk(i) for i in range(2)]


def test_evaluate_sample_fills_every_row_key(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    row = ig.evaluate_sample(m, samples[0], bank=bank, grids=grids, library=lib)
    for key in ("E_arm", "grid_error", "img_error", "unseen_color_error",
                "E_B0_identity", "E_B1_libmean", "E_B2_librandom",
                "E_B3_bucket_retrieval", "E_B4_oracle",
                "E_N1_shuffle", "M_N1_shuffle", "E_N2_irrelevant", "M_N2_irrelevant",
                "E_N3_const", "M_N3_const", "loc_in", "loc_band", "loc_out",
                "field_gt", "field_const", "field_shuffle", "headline_predalpha"):
        assert row.get(key) is not None, key
    assert len(row["E_B2_librandom_repeats"]) == cfg.n_repeats


def test_board_carries_every_required_column_and_publishes(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    board = runner.evaluate(m, samples, bank=bank, grids=grids, library=lib, cfg=cfg,
                            split="V_what", u_hist=[0.0, 0.5, 1.0], quick=False,
                            published=True, interp_pairs=2, strength_n=2)
    cols = board["criteria_columns"]
    for key in ig.REQUIRED_CRITERIA:
        assert int(cols[key].get("n", 0)) > 0, key
    assert board["contexts"]["all"]["headline_normal_only"]["n"] == 2
    assert "identity_footnotes" in board and "G2" in board["identity_footnotes"]
    rep = ig.publish_arm_board(board, cfg=cfg,
                               steps_row={c: 0 for c in ig.step_columns(cfg)})
    assert rep["criteria"]["headline_normal_only_n"] == 2


def test_headline_uses_the_frozen_image_formation(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    s = samples[0]
    params = m.theta(s.z.unsqueeze(0))
    f_img = m.apply_image(s.image, params, u=s.alpha).y
    manual = mix_alpha(s.image, f_img, s.alpha.unsqueeze(0))
    assert torch.equal(_criteria.compose_hat(s.image, s.alpha.unsqueeze(0), f_img),
                       manual)


def test_volume_round_trip_is_the_bank_operator(bank):
    """The library-mean transform is applied as a LUT volume; the packing must be the
    bank's own (BGR storage) or every B1 number would be axis-swapped."""
    lid = bank.lut_ids()[0]
    n = 33
    q = ig.lut_grid_queries(n)
    vol = ig.volume_from_values(bank.apply(q, lid), n)
    x = torch.rand(64, 3)
    from q3vl.whatb.lutdata import apply_lut_volume
    assert torch.allclose(apply_lut_volume(vol, x), bank.apply(x, lid), atol=2e-2)
    # exact at the sampled grid points themselves
    assert torch.allclose(apply_lut_volume(vol, q), bank.apply(q, lid), atol=1e-5)


def test_strength_columns_report_the_constructive_rows(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    cols = ig.strength_columns(m, samples[:1], bank=bank, grid=grids.grid, library=lib)
    assert cols["strength_mono_rate_lambda"]["random_floor"] == 0.5
    assert cols["strength_mono_rate_lambda"]["mono_rate_vs_u"] == 1.000
    assert cols["strength_spearman_lambda"]["spearman_vs_u"] == 1.000
    assert set(cols["strength_dE_u"]["per_u"]) == {str(float(u)) for u in ig.STRENGTH_U}
    assert set(cols["oob_rate_u"]["per_u"]) == {str(float(u)) for u in ig.EXTRAP_U}
    assert cols["dlib_u"]["n"] > 0


def test_strength_columns_survive_x_grid_wider_than_the_library_grid(bank):
    """d_lib must be read on the LIBRARY's grid, not on X_grid.

    The frozen shapes are ``grid_n=17`` (X_grid) and ``floor_grid_n=9`` (the grid
    the pre-registered floors were measured with), so the two query sets are
    *different sizes* in every real run.  ``tiny_cfg`` sets them equal, which hid a
    hard crash: ``strength_columns`` evaluated the transform on X_grid and handed it
    to ``LibraryValues.distance_to``, whose ``lab`` is on the library grid ->
    ``RuntimeError: The size of tensor a (729) must match the size of tensor b
    (4913)`` from ``colorimetry.delta_e00``.  It took the whole board down after
    training had already run.
    """
    torch.manual_seed(21)
    cfg = tiny_cfg(grid_n=5, floor_grid_n=4)
    grids = ig.GridSpec.build(cfg, n_heldout=32)
    assert grids.grid.shape[0] != grids.floor_grid.shape[0]
    ids = bank.lut_ids()[:3]
    lib = ig.LibraryContext.build(bank, ids, grids.floor_grid,
                                  bucket_pools={"b0": list(ids)},
                                  mean_grid_n=cfg.b1_grid_n)
    m = ig.IdGateArm(cfg)
    s = ig.EvalSample(sample_id="s0", winner_confidence="normal", task_type="style",
                      lut_id=ids[0], image=torch.rand(3, 6, 8), alpha=torch.rand(6, 8),
                      z=torch.randn(2560), source_image_id="src0")
    cols = ig.strength_columns(m, [s], bank=bank, grid=grids.grid, library=lib)
    assert cols["dlib_u"]["n"] > 0
    assert set(cols["dlib_u"]["per_u"]) == {str(float(u)) for u in ig.DLIB_U}


def test_interpolation_columns_include_the_trivial_output_mix(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    cols = ig.interpolation_columns(m, [(samples[0], samples[1])], bank=bank,
                                    grid=grids.grid, k_path=4)
    assert cols["interp_grid"]["output_mix_trivial"]["n"] > 0
    assert cols["interp_grid"]["alphas"] == list(ig.INTERP_ALPHAS)
    assert cols["mono_rate"]["random_floor"] == 0.5
    assert "no percentile trimming" in cols["path_len"]["note"]


def test_extra_columns_reject_a_column_without_n():
    with pytest.raises(ValueError, match="without an 'n'"):
        ig.extra_criteria_columns(gate_identity={"max_abs": 0.0},
                                  u_histogram={"n": 1}, strength={})


def test_loss_preregistration_records_what_is_not_added():
    rec = ig.loss_preregistration(ig.IdGateConfig())
    names = {t["name"] for t in rec["terms"]}
    assert names == {"L_rec", "L_hc", "R_sparse"}
    assert {n["name"] for n in rec["not_added"]} >= {"L_interval", "L_img"}
    assert rec["optimizer"]["schedule"].startswith("cosine annealing from 1e-3")
    assert rec["checkpoint_selection"].startswith(".contexts.all.headline_normal_only")
    assert "AUC in any form" in rec["banned"]
    with_probe = ig.loss_preregistration(ig.IdGateConfig(gate_zhead="linear"))
    assert {t["name"] for t in with_probe["terms"]} == {"L_rec", "L_hc", "R_sparse",
                                                        "L_gate"}


def test_runner_parser_defaults_match_the_proposal():
    args = runner.build_parser().parse_args(["--run-dir", "/tmp/x"])
    assert args.gate is True and args.gate_clamp == "after"
    assert (args.gate_p_end, args.gate_ku) == (0.2, 4)
    assert args.gate_u_source == "sample" and args.gate_u_dist == "uniform01"
    assert args.gate_null_prompt == "keep" and args.gate_zhead == "none"
    assert args.gate_zhead_weight == 0.1 and args.gate_field_src == "gt"
    assert args.clamp == "two" and args.n_gauss == 48 and args.gen_width == 128
    assert args.lr == 1e-3 and args.pi_lr_scale == 0.1 and args.epochs == 40
    assert args.batch_split == "32x256" and args.seed == 20260810
    assert args.lut_resample == "none" and args.colorspan_samples == 256
    cfg = runner.config_from_args(args)
    assert (cfg.batch_samples, cfg.queries) == (32, 256)
    assert cfg.total_steps == 117_440
    assert cfg.gate_u_eval == ig.DEFAULT_U_EVAL


def test_runner_no_gate_flag_gives_the_epr024_shape():
    args = runner.build_parser().parse_args(["--run-dir", "/tmp/x", "--no-gate"])
    cfg = runner.config_from_args(args)
    assert cfg.gate is False and cfg.k_u == 1
    assert cfg.total_steps == ig.IdGateConfig().total_steps      # step-matched (U4)


def test_runner_ablation_flags_reach_the_config():
    argv = ["--run-dir", "/tmp/x", "--gate-clamp", "before", "--gate-u-dist",
            "excl_band", "--gate-p-end", "0.5", "--gate-ku", "8", "--gate-zhead",
            "linear", "--gate-zhead-weight", "1.0", "--l-rec-scale", "0.5",
            "--chroma-weight-src", "y_u", "--pi-lr-scale", "1.0", "--gate-u-source",
            "lambda", "--gate-field-src", "shuffle", "--no-mining"]
    cfg = runner.config_from_args(runner.build_parser().parse_args(argv))
    assert cfg.gate_clamp == "before" and cfg.gate_u_dist == "excl_band"
    assert cfg.gate_p_end == 0.5 and cfg.gate_ku == 8
    assert cfg.gate_zhead == "linear" and cfg.gate_zhead_weight == 1.0
    assert cfg.l_rec_scale == 0.5 and cfg.chroma_weight_src == "y_u"
    assert cfg.pi_lr_scale == 1.0 and cfg.gate_u_source == "lambda"
    assert cfg.gate_field_src == "shuffle" and cfg.mining is False


def test_config_rejects_unknown_choices():
    for kw in ({"gate_u_source": "nope"}, {"gate_clamp": "middle"},
               {"gate_u_dist": "gauss"}, {"clamp": "none"}, {"gate_p_end": 1.5},
               {"gate_ku": 0}, {"chroma_weight_src": "lut"}):
        with pytest.raises(ValueError):
            ig.IdGateConfig(**kw)
    with pytest.raises(NotImplementedError):
        ig.IdGateConfig(loss_level=4)


def _write_leaf(root, split, tag, *, checkpoint="/other/ckpt", dtype=np.float32):
    """One leaf of the shared cache layout (q3vl.whatb.zcache)."""
    d = Path(root) / f"{split}__{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.jsonl").write_text(
        json.dumps({"sample_id": "a", "row": 0}) + "\n", encoding="utf-8")
    np.save(d / "z.npy", np.zeros((1, 2560), dtype=dtype))
    (d / "meta.json").write_text(json.dumps(
        {"checkpoint": checkpoint, "readout_kind": "seg_color", "split": split,
         "control_tag": tag, "context_source": "generated", "n": 1,
         "dtype": str(np.dtype(dtype))}), encoding="utf-8")
    return d


def test_zcache_refuses_a_foreign_checkpoint(tmp_path):
    _write_leaf(tmp_path, "train", "none")
    with pytest.raises(AssertionError, match="checkpoint"):
        runner.open_z_cache(tmp_path, "train",
                            checkpoint="/home/bc/data/runs/base/checkpoint-4976")
    cache = runner.open_z_cache(tmp_path, "train", checkpoint="/other/ckpt")
    assert len(cache) == 1 and "a" in cache


def test_zcache_refuses_a_non_fp32_array(tmp_path):
    """Ruling 11.1-5: rejected, never upcast (that is what hid a bf16 cache)."""
    from q3vl.whatb.zcache import ZCacheDtypeError

    _write_leaf(tmp_path, "train", "none", checkpoint="c", dtype=np.float16)
    with pytest.raises(ZCacheDtypeError, match="fp32"):
        runner.open_z_cache(tmp_path, "train", checkpoint="c")


def test_synthetic_samples_are_reproducible(bank):
    ids = bank.lut_ids()[:2]
    a = runner.synthetic_eval_samples(3, bank, ids, seed=5)
    b = runner.synthetic_eval_samples(3, bank, ids, seed=5)
    assert torch.equal(a[0].image, b[0].image) and torch.equal(a[1].z, b[1].z)
    assert a[0].task_type == "style" and float(a[0].alpha.min()) == 1.0


# --------------------------------------------------------------------------- #
# 10. the additions the HANDOFF requires on every board, and the field calibration
# --------------------------------------------------------------------------- #
def test_degenerate_weight_rate_is_on_the_board(eval_kit, bank):
    """HANDOFF section 2.3: proposition 2's ``delta`` column is published every time."""
    cfg, m, grids, lib, samples = eval_kit
    cols = ig.degeneracy_columns(m, samples[:1], grid=grids.grid)
    assert cols["degenerate_weight_rate"]["n"] == 1
    assert cols["degenerate_weight_rate"]["tau"] == 1e-3
    assert 0.0 <= cols["degenerate_weight_rate"]["mean"] <= 1.0
    assert cols["degenerate_precision_rate"]["n"] == 1
    board = runner.evaluate(m, samples, bank=bank, grids=grids, library=lib, cfg=cfg,
                            split="V_what", u_hist=[0.5], quick=True, published=False,
                            interp_pairs=2, strength_n=1)
    assert board["criteria_columns"]["degenerate_weight_rate"]["n"] > 0


def test_grid_initialised_geometry_has_no_degenerate_weights(monkeypatch):
    """App A.1's initialisation puts the rate at 0; a non-zero value means drift."""
    m = ig.IdGateArm(tiny_cfg())
    ident = GlutParams.identity(m.cfg.n_gauss, batch=1)
    monkeypatch.setattr(m, "theta", lambda *a, **k: ident)
    s = ig.EvalSample(sample_id="s", winner_confidence="normal", task_type="style",
                      lut_id="x", image=torch.rand(3, 4, 4), alpha=torch.ones(4, 4),
                      z=torch.randn(2560))
    cols = ig.degeneracy_columns(m, [s], grid=_queries.uniform_grid(5))
    assert cols["degenerate_weight_rate"]["mean"] == 0.0
    assert cols["degenerate_precision_rate"]["mean"] == 0.0


def test_lambda_sign_row_nine_is_deterministic_per_sample():
    cfg = tiny_cfg(gate_lambda_sign="random")
    signs = {ig.lambda_sign(f"s{i}", seed=cfg.seed) for i in range(50)}
    assert signs == {1.0, -1.0}                      # both signs occur
    assert ig.lambda_sign("s3", seed=cfg.seed) == ig.lambda_sign("s3", seed=cfg.seed)
    mk = lambda sid: ig.EvalSample(sample_id=sid, winner_confidence="normal",
                                   task_type="style", lut_id="x",
                                   image=torch.zeros(3, 2, 2), alpha=torch.ones(2, 2),
                                   z=torch.zeros(2560), lam=1.0)
    us = {ig.effective_lambda_u(mk(f"s{i}"), cfg) for i in range(50)}
    assert us == {0.0, 1.0}                          # a negative lambda lands on u = 0
    fixed = tiny_cfg()
    assert ig.effective_lambda_u(mk("s0"), fixed) == 1.0


def test_row_carries_the_gate_u_column(eval_kit, bank):
    cfg, m, grids, lib, samples = eval_kit
    row = ig.evaluate_sample(m, samples[0], bank=bank, grids=grids, library=lib)
    assert row["gate_u"] == row["u_scalar"]
    assert row["u_source"] == "sample" and row["field_src"] == "gt"


def test_align_fields_enforces_the_frozen_calibration():
    s = ig.EvalSample(sample_id="s", winner_confidence="normal", task_type="local",
                      lut_id="x", image=torch.rand(3, 8, 10), alpha=torch.rand(8, 10),
                      z=torch.zeros(2560), alpha_pred=torch.rand(4, 5))
    out = runner.align_fields(s, short_side=8)
    assert tuple(out.alpha_pred.shape) == (8, 10)     # area_resize'd up to the grid
    assert torch.equal(out.alpha, s.alpha)            # GT is never resampled
    bad_gt = ig.EvalSample(**{**s.__dict__, "alpha": torch.rand(4, 5)})
    with pytest.raises(ValueError, match="does not match the image grid"):
        runner.align_fields(bad_gt, short_side=8)
    with pytest.raises(ValueError, match="short side"):
        runner.align_fields(s, short_side=512)


def test_runner_records_the_readout_contract():
    from q3vl.whatb.readout import WhatReadoutSpec
    spec = WhatReadoutSpec("seg_color").to_dict()
    assert spec["readout"] == "seg_color" and spec["needs_v2seg"] is True
    assert spec["hidden_layer"] == -1 and spec["hidden_final_norm"] is True


@torch.no_grad()
def test_u_shapes_are_right_padded_to_the_query_rank(model):
    """``(B,)`` / ``(B,1)`` are per-sample; ``(B,P)`` is per-query.  Never left-aligned."""
    torch.manual_seed(22)
    x = torch.rand(3, 3, 3)                     # B == P == 3: the ambiguous case
    params = model.theta(torch.randn(3, 2560))
    per_sample = torch.tensor([0.0, 0.5, 1.0])
    a = model.apply_transform(x, params, u=per_sample).y
    b = model.apply_transform(x, params, u=per_sample.reshape(3, 1)).y
    c = model.apply_transform(x, params, u=per_sample.reshape(3, 1, 1)).y
    assert torch.equal(a, c) and torch.equal(b, c)
    assert torch.equal(a[0], x[0])              # sample 0 has u = 0 on every query
    per_query = per_sample.reshape(1, 3).expand(3, 3).contiguous()
    d = model.apply_transform(x, params, u=per_query).y
    assert torch.equal(d[1, 0], x[1, 0])        # query 0 has u = 0 for every sample
    assert not torch.equal(a, d)


# --------------------------------------------------------------------------- #
# the eval bundle PRODUCER (q3vl/whatb/scripts/build_eval_bundle.py)
#
# These pin the producer against this arm's reader; they touch neither
# ``arms/idgate.py`` nor ``run_idgate_arm.py``.  No NFS: the images and fields
# are made up here and written with the producer's own writer, so a schema drift
# between the two sides fails in CI rather than after the run has started.
# --------------------------------------------------------------------------- #
from q3vl.whatb.scripts import build_eval_bundle as producer     # noqa: E402


def _bundle_rows(spec):
    """``spec = [(sample_id, source, task_type)]`` -> IndexRow list."""
    from q3vl.whatb.splits import IndexRow

    return [IndexRow.from_json({"sample_id": sid, "split": "V_what",
                                "lut_id": "lut_a", "source_image_id": src,
                                "task_type": tt, "winner_confidence": "normal"})
            for sid, src, tt in spec]


def test_shuffle_donor_never_reuses_the_sample_s_own_source():
    rows = _bundle_rows([(f"s{i}", f"src{i // 2}", "local" if i % 2 else "style")
                         for i in range(10)])
    by_id = {r.sample_id: r for r in rows}
    donors = producer.shuffle_donors(rows, pool="local", seed=3)
    assert set(donors) == {r.sample_id for r in rows}
    for sid, did in donors.items():
        assert did != sid
        assert by_id[did].source_image_id != by_id[sid].source_image_id
        assert not by_id[did].is_style          # the 'local' pool: real masks only
    assert donors == producer.shuffle_donors(rows, pool="local", seed=3)   # seeded
    assert donors != producer.shuffle_donors(rows, pool="local", seed=4)
    every = producer.shuffle_donors(rows, pool="all", seed=3)
    assert any(by_id[d].is_style for d in every.values())


def test_shuffle_donor_pool_must_not_be_empty():
    rows = _bundle_rows([(f"s{i}", f"src{i}", "style") for i in range(4)])
    with pytest.raises(ValueError, match="empty"):
        producer.shuffle_donors(rows, pool="local")


def test_producer_npz_keys_are_exactly_what_the_arm_reads(tmp_path):
    """The consumer's own loader on the producer's own writer."""
    torch.manual_seed(0)
    lines = []
    for i in range(3):
        h, w = (512, 640) if i % 2 else (768, 512)
        arrays = {
            "image": torch.rand(3, h, w).numpy(),
            "alpha": torch.rand(h, w).numpy(),
            "alpha_shuffle": torch.rand(h, w).numpy(),
            "z": torch.randn(2560).numpy(),
        }
        for key in producer.CONTROL_KEYS.values():
            arrays[key] = torch.randn(2560).numpy()
        sha, nbytes = producer.write_npz(tmp_path / f"s{i}.npz", arrays)
        assert len(sha) == 64 and nbytes > 0
        lines.append(json.dumps({"sample_id": f"s{i}", "lut_id": "lut_a",
                                 "task_type": "style" if i else "local",
                                 "winner_confidence": "normal",
                                 "minor": "bucket_0", "source_image_id": "src0"}))
    (tmp_path / "index.jsonl").write_text("\n".join(lines) + "\n")

    samples = runner.load_eval_bundle(tmp_path, device="cpu")
    assert [s.sample_id for s in samples] == ["s0", "s1", "s2"]
    for s in samples:
        assert s.image.shape[0] == 3 and min(s.image.shape[-2:]) == 512
        assert tuple(s.alpha.shape) == tuple(s.image.shape[-2:])
        assert set(s.z_ctrl) == {"N1_shuffle", "N2_irrelevant", "N3_const"}
        assert s.alpha_shuffle is not None and s.z_null is None
        assert s.minor == "bucket_0" and s.source_image_id == "src0"
    # the producer's self-check is the same path, and it passes on this bundle
    report = producer.self_check(tmp_path, 3)
    assert report["n_checked"] == 3 and report["z_ctrl_is_not_z"] is True


def test_producer_control_keys_match_the_arm_s_z_ctrl_names():
    """A rename on either side would silently drop a required N column."""
    assert set(producer.CONTROL_KEYS.values()) == {
        "z_N1_shuffle", "z_N2_irrelevant", "z_N3_const"}
    assert set(producer.CONTROL_KEYS) <= set(
        __import__("q3vl.whatb.zcache", fromlist=["CONTROL_TAGS"]).CONTROL_TAGS)
    for name in ("N1_shuffle", "N2_irrelevant", "N3_const"):
        assert f"E_{name}" and any(name in k for k in _criteria.PREREGISTERED_KEYS)


def test_producer_refuses_to_write_onto_the_nfs_mount():
    a = producer.build_parser().parse_args(
        ["--out", "/mnt/nfs/bc/whatever", "--split", "V_what"])
    with pytest.raises(SystemExit, match="NFS"):
        producer.build(a)
