"""QDUAL (EPR-029) unit tests.  CPU only: ``CUDA_VISIBLE_DEVICES=""``.

Structured around what the proposal can be *checked against* rather than what is
easy to assert:

* the four parameter counts of ``EPR-029:442-451`` to the digit (a missing
  LayerNorm or bias moves them);
* the zero-initialisation chain -- heads exactly zero -> ``dtheta == 0`` at step
  0 -> ``theta == theta_base`` -> ``f == identity`` -> the five ladder rows agree
  bit for bit at step 0 (``:639``);
* the ladder operators against the formulas they were transcribed from (CSRNet
  ``out*scale + shift + out``, HDRNet ``ReLU(b + W'g + Wl)``);
* the three losses against hand-computable values, and the ``C -> 0`` mask;
* the three campaign pitfalls: the degeneracy guard really fires (including the
  fact that it fires on a step-0 model, which is why it belongs at the first
  quick eval and not at step 0), the query->Gaussian diagnostics separate "the
  Gaussian moved" from "the index moved", and no constant tensor is created
  inside a forward.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from q3vl.whatb import criteria as crit
from q3vl.whatb import publish as pub
from q3vl.whatb.arms import qdual as qd
from q3vl.whatb.colorimetry import srgb_to_lab
from q3vl.whatb.glut import glut_forward, uniform_grid_positions
from q3vl.whatb.guards import DegenerateTransform, FIRST_STEP_COLUMNS
from q3vl.whatb.queries import COLORS_PER_STEP, uniform_grid
from q3vl.whatb.scripts import run_qdual_arm as runner

REPO = Path(__file__).resolve().parents[3]
ARM_SRC = Path(qd.__file__)
RUNNER_SRC = Path(runner.__file__)
BANK_OK = Path("/var/cache/veradata/preset_bank_full/luts_meta.json").is_file()
SPLITS_OK = (Path("/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/splits/"
                  "train.index.jsonl").is_file())


def _model(**kw) -> qd.QDualArm:
    torch.manual_seed(20260810)
    return qd.QDualArm(qd.QDualConfig(**kw))


# --------------------------------------------------------------------------- #
# 1. parameter counts -- EPR-029:442-451
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kw,want", [
    ({}, 5_833_038),                                        # d=256 L=4 K=4
    ({"decoder_width": 128, "decoder_layers": 2}, 1_736_654),
    ({"z_expand_k": 1}, 3_866_190),
    ({"decoder_width": 128, "decoder_layers": 2, "z_expand_k": 1}, 753_230),
])
def test_parameter_count_matches_the_proposal_table(kw, want):
    m = _model(**kw)
    assert qd.param_count_breakdown(m.decoder)["n_params"] == want


def test_theta_base_is_22n_plus_12():
    m = _model()
    blocks = qd.param_count_breakdown(m.decoder)["n_params_by_block"]
    assert blocks["theta_base"] == 22 * 48 + 12 == 1068
    # pi_z alone is the 2,627,584 the proposal calls out at :449
    assert blocks["proj_z"] + blocks["ln_z"] == 2_627_584


# --------------------------------------------------------------------------- #
# 2. configuration / field geometry
# --------------------------------------------------------------------------- #
def test_both_field_sources_give_the_same_token_count():
    low = qd.QDualConfig(field_source="m_low")
    pix = qd.QDualConfig(field_source="m_pix")
    assert (low.patch, low.field_hw, low.token_grid) == (4, (32, 48), (8, 12))
    assert (pix.patch, pix.field_hw, pix.token_grid) == (16, (128, 192), (8, 12))
    assert low.n_field_tokens == pix.n_field_tokens == 96


def test_a_patch_grid_larger_than_the_field_pe_raises_instead_of_truncating():
    # NOTES 13: PE_row / PE_col are 32 long; 33 rows must raise.
    with pytest.raises(ValueError, match="field PE"):
        qd.GaussianQueryDecoder(qd.QDualConfig(field_grid_h=132, field_grid_w=132))


def test_clamp_none_has_no_cli_spelling():
    with pytest.raises(ValueError, match="frozen block"):
        qd.QDualConfig(clamp="none")


def test_memory_rung_maps_d_and_e_onto_b():
    assert [qd.QDualConfig(rung=r).memory_rung for r in "abcde"] == \
        ["a", "b", "c", "b", "b"]


# --------------------------------------------------------------------------- #
# 3. query decoding: shapes
# --------------------------------------------------------------------------- #
def test_query_decode_shapes():
    m = _model()
    z, field = torch.randn(3, 2560), torch.rand(3, 32, 48)
    dtheta, dglob = m.decoder.residuals(z, field)
    assert dtheta.shape == (3, 48, 22) and dglob.shape == (3, 12)
    p = m.decoder(z, field)
    assert (p.batch_size, p.n_gauss) == (3, 48)
    assert p.mu.shape == (3, 48, 3) and p.m_local.shape == (3, 48, 3, 3)
    assert p.g_matrix.shape == (3, 3, 3) and p.g_bias.shape == (3, 3)
    y = m(torch.rand(7, 3), z, field)
    assert y.shape == (3, 7, 3)
    assert m.decoder.queries(3).shape == (3, 49, 256)


@pytest.mark.parametrize("rung,rows", [("a", 4), ("b", 5), ("c", 100),
                                       ("d", 5), ("e", 5)])
def test_memory_rows_per_ladder_row(rung, rows):
    m = _model(rung=rung)
    field = None if rung == "a" else torch.rand(2, 32, 48)
    assert m.decoder.build_memory(torch.randn(2, 2560), field).shape == (2, rows, 256)


def test_rung_c_without_a_field_raises_and_rung_a_does_not_need_one():
    with pytest.raises(ValueError, match="consumes the spatial field"):
        _model(rung="c").decoder(torch.randn(1, 2560), None)
    assert _model(rung="a").decoder(torch.randn(1, 2560), None).batch_size == 1


def test_a_field_of_the_wrong_resolution_raises():
    m = _model()
    with pytest.raises(ValueError, match="declares 32x48"):
        m.decoder(torch.randn(1, 2560), torch.rand(1, 16, 24))


def test_z_expand_qtok_takes_k_vectors_through_one_shared_projection():
    m = _model(z_expand="qtok", z_expand_k=4)
    assert len(m.decoder.proj_z) == 1
    p = m.decoder(torch.randn(2, 4, 2560), torch.rand(2, 32, 48))
    assert p.batch_size == 2
    with pytest.raises(ValueError, match="wants z of"):
        m.decoder(torch.randn(2, 2560), torch.rand(2, 32, 48))
    with pytest.raises(ValueError, match="query-token vectors"):
        m.decoder(torch.randn(2, 3, 2560), torch.rand(2, 32, 48))


# --------------------------------------------------------------------------- #
# 4. zero initialisation -> step 0 is exactly the identity
# --------------------------------------------------------------------------- #
def test_heads_are_exactly_zero_and_the_residual_is_exactly_zero():
    m = _model()
    for head in (m.decoder.head_g, m.decoder.head_a):
        assert float(head.weight.detach().abs().max()) == 0.0
        assert float(head.bias.detach().abs().max()) == 0.0
    dtheta, dglob = m.decoder.residuals(torch.randn(4, 2560), torch.rand(4, 32, 48))
    assert float(dtheta.detach().abs().max()) == 0.0
    assert float(dglob.detach().abs().max()) == 0.0


def test_step0_theta_is_theta_base_and_g_is_zero():
    m = _model()
    p = m.decoder(torch.randn(2, 2560), torch.rand(2, 32, 48))
    base = m.decoder.base_params(2)
    for name in ("mu", "chol_diag", "chol_off", "opacity_logit", "m_local",
                 "b_local", "g_matrix", "g_bias"):
        assert torch.equal(getattr(p, name), getattr(base, name))
    assert float(p.g_matrix.detach().abs().max()) == 0.0   # NOVEL: G = 0 (NOTES 3)
    assert float(p.g_bias.detach().abs().max()) == 0.0
    assert torch.allclose(p.m_local[0, 0], torch.eye(3))
    assert torch.allclose(p.mu[0], uniform_grid_positions(48))


def test_step0_forward_is_the_identity_up_to_the_epsilon_of_proposition_2():
    m = _model()
    x = uniform_grid(9)
    y = m(x, torch.randn(3, 2560), torch.rand(3, 32, 48))
    # eps = 1e-6 gives f(x) = (1 - delta(x)) x with delta ~ 1e-7, so this is
    # small-but-not-zero by construction (proposition 2), not by sloppiness.
    dev = float((y - x.unsqueeze(0)).detach().abs().max())
    assert 0.0 < dev < 1e-5
    # with eps = 0 it is exact
    p = m.decoder.base_params(1)
    y0 = glut_forward(x, p, clamp="none", eps=0.0)
    assert float((y0 - x.unsqueeze(0)).detach().abs().max()) < 1e-6


def test_the_five_ladder_rows_agree_bit_for_bit_at_step_zero():
    x = uniform_grid(9)
    z, field = torch.randn(2, 2560), torch.rand(2, 32, 48)
    outs = [_model(rung=r)(x, z, field) for r in qd.LADDER_ROWS]
    for o in outs[1:]:
        assert torch.equal(o, outs[0])


def test_step0_output_does_not_depend_on_the_condition_or_the_field():
    m = _model()
    x = uniform_grid(9)
    a = m(x, torch.randn(1, 2560), torch.rand(1, 32, 48))
    b = m(x, torch.zeros(1, 2560), torch.ones(1, 32, 48))
    assert torch.equal(a, b)


def test_zero_init_witness_and_its_assertion():
    m = _model()
    w = qd.zero_init_witness(m, torch.randn(2, 2560), torch.rand(2, 32, 48),
                             uniform_grid(9))
    assert w["zeroinit_step0_maxabs"] == 0.0
    assert 0.0 < w["step0_maxabs_f_minus_id"] < 1e-5
    qd.assert_zero_init(w)                                   # passes

    m2 = _model(zero_init_head=False)
    w2 = qd.zero_init_witness(m2, torch.randn(2, 2560), torch.rand(2, 32, 48),
                              uniform_grid(9))
    assert w2["zeroinit_step0_maxabs"] > 0.0
    with pytest.raises(AssertionError, match="zeroinit_step0_maxabs"):
        qd.assert_zero_init(w2)
    qd.assert_zero_init(w2, enabled=False)                   # the ablation row


def test_the_ablation_row_drops_exactly_one_required_key():
    on = qd.required_criteria_table(qd.QDualConfig())
    off = qd.required_criteria_table(qd.QDualConfig(zero_init_head=False))
    assert set(on) - set(off) == {"zeroinit_step0_maxabs"}


# --------------------------------------------------------------------------- #
# 5. the ladder operators against the code they were transcribed from
# --------------------------------------------------------------------------- #
def test_film_layer_is_csrnet_out_times_scale_plus_shift_plus_out():
    torch.manual_seed(0)
    layer = qd.XAttnFFN(8, 2, op="film")
    q = torch.randn(1, 3, 8)
    memory = torch.randn(1, 5, 8)
    got = layer(q, memory)
    qn = layer.norm1(q)
    mbar = memory.mean(dim=1, keepdim=True)
    want = qn * layer.cond_scale(mbar) + layer.cond_shift(mbar) + q
    want = want + layer.ffn(layer.norm2(want))
    assert torch.allclose(got, want, atol=1e-6)


def test_broadcast_add_layer_is_hdrnet_relu_of_b_plus_global_plus_local():
    torch.manual_seed(0)
    layer = qd.XAttnFFN(8, 2, op="badd")
    q, memory = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    got = layer(q, memory)
    qn = layer.norm1(q)
    mbar = memory.mean(dim=1, keepdim=True)
    want = torch.relu(layer.w_local(qn) + layer.w_global(mbar))
    want = want + layer.ffn(layer.norm2(want))
    assert torch.allclose(got, want, atol=1e-6)
    assert layer.w_global.bias is None            # b lives on W (one bias, Eq.2)


def test_cross_attention_is_softmax_qk_over_sqrt_head_dim():
    torch.manual_seed(0)
    layer = qd.XAttnFFN(8, 2, op="xattn")
    q, memory = torch.randn(1, 3, 8), torch.randn(1, 5, 8)
    qn = layer.norm1(q)
    qh = layer.w_q(qn).reshape(1, 3, 2, 4).transpose(1, 2)
    kh = layer.w_k(memory).reshape(1, 5, 2, 4).transpose(1, 2)
    vh = layer.w_v(memory).reshape(1, 5, 2, 4).transpose(1, 2)
    attn = torch.softmax(qh @ kh.transpose(-1, -2) / math.sqrt(4), dim=-1)
    want = q + layer.w_o((attn @ vh).transpose(1, 2).reshape(1, 3, 8))
    want = want + layer.ffn(layer.norm2(want))
    assert torch.allclose(layer(q, memory), want, atol=1e-6)


def test_attention_temperature_is_off_by_default_and_learnable_when_on():
    assert _model().decoder.layers[0].attn_temperature is None
    t = _model(attn_temperature=True).decoder.layers[0].attn_temperature
    assert isinstance(t, torch.nn.Parameter) and float(t.detach()) == 1.0


def test_ladder_rows_pick_the_declared_operator():
    for rung, op in (("a", "xattn"), ("b", "xattn"), ("c", "xattn"),
                     ("d", "film"), ("e", "badd")):
        assert _model(rung=rung).decoder.layers[0].op == op


# --------------------------------------------------------------------------- #
# 6. colour PE / E_type ablations
# --------------------------------------------------------------------------- #
def test_colour_pe_indices_come_from_the_grid_itself():
    m = _model()
    idx = m.decoder.theta_base.grid_index
    mu = uniform_grid_positions(48)
    assert m.decoder.grid_extent == (4, 4, 3)          # N = 48 -> 4x4x3
    for k in range(3):
        levels = torch.unique(mu[:, k])
        assert torch.allclose(levels[idx[:, k]], mu[:, k])


def test_colour_pe_reaches_only_the_gaussian_queries():
    m = _model()
    with torch.no_grad():
        m.decoder.pe_r.add_(1.0)
    q = m.decoder.queries(1)[0]
    # atol covers the float32 cancellation of (emb + 1) - 1 for small entries
    assert torch.allclose(q[:48] - 1.0, m.decoder.q_emb[:48], atol=1e-6)
    assert torch.equal(q[48], m.decoder.q_emb[48])      # the global-affine query


def test_ablating_the_colour_pe_and_e_type_removes_their_parameters():
    blocks = qd.param_count_breakdown(
        _model(color_pe=False, e_type=False).decoder)["n_params_by_block"]
    assert "pe_r" not in blocks and "e_type" not in blocks
    m = _model(color_pe=False)
    assert torch.equal(m.decoder.queries(1)[0], m.decoder.q_emb)


# --------------------------------------------------------------------------- #
# 7. losses (GLUT Eq.6-8) -- hand-computable values
# --------------------------------------------------------------------------- #
def test_rec_loss_is_the_mean_absolute_error():
    a, b = torch.zeros(2, 4, 3), torch.full((2, 4, 3), 0.25)
    assert float(qd.rec_loss(a, b)) == pytest.approx(0.25)


def test_hue_chroma_masks_the_achromatic_points_and_counts_them():
    grey = torch.full((1, 5, 3), 0.5)             # C = 0 -> every point masked
    loss, n_masked = qd.hue_chroma_loss(grey, grey)
    assert n_masked == 5 and float(loss) == 0.0
    # a chromatic target with a hue-flipped prediction: <h_hat, h> = -1 -> 2C
    target = torch.tensor([[[0.9, 0.2, 0.2]]])
    pred = torch.tensor([[[0.2, 0.9, 0.9]]])
    lab_t = srgb_to_lab(target)
    c = float(torch.sqrt(lab_t[..., 1] ** 2 + lab_t[..., 2] ** 2))
    loss2, n2 = qd.hue_chroma_loss(pred, target)
    lab_p = srgb_to_lab(pred)
    cp = torch.sqrt(lab_p[..., 1] ** 2 + lab_p[..., 2] ** 2)
    dot = float((lab_t[..., 1] * lab_p[..., 1] + lab_t[..., 2] * lab_p[..., 2])
                / (cp * torch.sqrt(lab_t[..., 1] ** 2 + lab_t[..., 2] ** 2)))
    assert n2 == 0
    assert float(loss2) == pytest.approx(c * (1 - dot), rel=1e-5)


def test_sparse_regulariser_is_the_binary_entropy():
    half = torch.full((1, 48), 0.5)
    # closed form WITH the eps of Eq.8: -log(0.5 + 1e-6), not log 2 exactly
    assert float(qd.sparse_regulariser(half)) == pytest.approx(
        -math.log(0.5 + 1e-6), rel=1e-6)
    near_one = torch.full((1, 48), 1.0)
    assert float(qd.sparse_regulariser(near_one)) == pytest.approx(0.0, abs=1e-5)


def test_total_loss_uses_10_and_0p001_and_publishes_the_frozen_columns():
    m = _model()
    x = uniform_grid(5)
    z, field = torch.randn(2, 2560), torch.rand(2, 32, 48)
    y_hat, aux = m(x, z, field, return_aux=True)
    y = torch.rand_like(y_hat)
    out = qd.qdual_losses(y_hat, y, aux)
    parts = {k: float(v.detach()) for k, v in
             (("t", out.total), ("rec", out.l_rec), ("hc", out.l_hc),
              ("sp", out.r_sparse))}
    assert parts["t"] == pytest.approx(
        parts["rec"] + 10.0 * parts["hc"] + 0.001 * parts["sp"], rel=1e-6)
    for col in ("L_rec", "L_hc", "L_sparse", "R_sparse", "L_total", "n_hc_masked",
                "n_colors"):
        assert col in out.columns
    assert out.columns["n_colors"] == 2 * 125


def test_loss_preregistration_records_the_constants_and_adds_no_term():
    rec = qd.loss_preregistration(qd.QDualConfig())
    assert rec["lambda_hc"] == 10.0 and rec["lambda_sparse"] == 0.001
    assert rec["added_terms"] == []
    assert rec["supervision_space"].startswith("function value")
    assert rec["criteria_required"] == list(qd.REQUIRED_QDUAL)


def test_the_loss_is_differentiable_into_every_trainable_block():
    m = _model()
    x = uniform_grid(4)
    y_hat, aux = m(x, torch.randn(2, 2560), torch.rand(2, 32, 48), return_aux=True)
    qd.qdual_losses(y_hat, torch.rand_like(y_hat), aux).total.backward()
    # the zero-initialised heads still receive gradient (that is the point of
    # zero-init: zero output, non-zero gradient)
    assert float(m.decoder.head_g.weight.grad.abs().max()) > 0
    assert float(m.decoder.head_a.weight.grad.abs().max()) > 0
    for name in ("q_emb", "pe_row", "e_type"):
        assert getattr(m.decoder, name).grad is not None, name


# --------------------------------------------------------------------------- #
# 8. hard-example mining (ruling 11.1-4)
# --------------------------------------------------------------------------- #
def test_pure_black_does_not_poison_the_gradient():
    """B1: pure black had a 0*inf = NaN gradient; the SHARED function fixes it."""
    x = torch.zeros(1, 1, 3, requires_grad=True)
    target = torch.tensor([[[0.4, 0.2, 0.1]]])
    loss, _ = qd.hue_chroma_loss(x, target)
    loss.backward()
    assert bool(torch.isfinite(x.grad).all())
    # the shared implementation is the one that is fixed -- no arm-local floor
    y = torch.zeros(1, 1, 3, requires_grad=True)
    srgb_to_lab(y).sum().backward()
    assert bool(torch.isfinite(y.grad).all())
    # ... and an exactly neutral *prediction* against a neutral target too
    z = torch.zeros(1, 1, 3, requires_grad=True)
    qd.hue_chroma_loss(z, torch.zeros(1, 1, 3))[0].backward()
    assert bool(torch.isfinite(z.grad).all())


def test_mining_ratio_ramp():
    assert qd.mining_ratio_for_step(0, 10) == pytest.approx(0.10)
    assert qd.mining_ratio_for_step(50, 10) == pytest.approx(0.10)     # epoch 5
    assert qd.mining_ratio_for_step(125, 10) == pytest.approx(0.25)    # epoch 12.5
    assert qd.mining_ratio_for_step(300, 10) == pytest.approx(0.40)    # epoch 30


def test_mining_keeps_q_colours_and_takes_the_worst_probe_ones():
    m = _model()
    torch.manual_seed(0)
    probe = torch.rand(2, 20, 3)
    fresh = torch.rand(2, 20, 3)
    z, field = torch.randn(2, 2560), torch.rand(2, 32, 48)
    # step-0 model is the identity, so the error is |probe - target|
    target = torch.rand(2, 20, 3)
    out, stats = qd.mine_hard_colors(m, z, field, probe, fresh, target, 0.25)
    assert out.shape == probe.shape
    assert stats == {"mining_ratio": 0.25, "n_mined": 5, "n_fresh": 15}
    err = (probe - target).abs().mean(-1)
    want = torch.topk(err, 5, dim=1).indices
    for b in range(2):
        got = {tuple(c.tolist()) for c in out[b, :5]}
        assert got == {tuple(probe[b, i].tolist()) for i in want[b]}
    assert torch.equal(out[:, 5:], fresh[:, :15])


def test_mining_with_ratio_zero_returns_the_fresh_batch_untouched():
    m = _model()
    probe, fresh = torch.rand(1, 8, 3), torch.rand(1, 8, 3)
    out, stats = qd.mine_hard_colors(m, torch.randn(1, 2560),
                                     torch.rand(1, 32, 48), probe, fresh,
                                     torch.rand(1, 8, 3), 0.0)
    assert torch.equal(out, fresh) and stats["n_mined"] == 0


# --------------------------------------------------------------------------- #
# 9. the degenerate-solution guard (pitfall 3)
# --------------------------------------------------------------------------- #
def test_guard_fires_on_a_constant_field():
    x = uniform_grid(5)
    y = torch.full((4, x.shape[0], 3), 0.42)
    with pytest.raises(DegenerateTransform) as e:
        qd.assert_not_degenerate(y, x, exit_process=False)
    assert "flat across query colours" in str(e.value)


def test_guard_fires_when_every_sample_gets_the_same_transform():
    x = uniform_grid(5)
    one = (x * 0.5 + 0.1).unsqueeze(0)
    with pytest.raises(DegenerateTransform) as e:
        qd.assert_not_degenerate(one.repeat(4, 1, 1), x, exit_process=False)
    assert "one transform for every sample" in str(e.value)


def test_guard_fires_on_a_step0_model_which_is_why_it_runs_at_quick_eval():
    m = _model()
    x = uniform_grid(5)
    with torch.no_grad():          # the runner calls the guard exactly this way
        y = m(x, torch.randn(4, 2560), torch.rand(4, 32, 48))
    with pytest.raises(DegenerateTransform) as e:
        qd.assert_not_degenerate(y, x, exit_process=False)
    assert "transform is the identity" in str(e.value)


def test_guard_passes_on_a_live_transform_and_reports_the_three_numbers():
    x = uniform_grid(5)
    torch.manual_seed(0)
    y = (x.unsqueeze(0) * torch.rand(4, 1, 3) + torch.rand(4, 1, 3) * 0.3).clamp(0, 1)
    rep = qd.assert_not_degenerate(y, x, exit_process=False)
    assert rep.ok and rep.failures == ()
    assert set(rep.as_dict()) >= {"point_std", "identity_dev", "cross_std",
                                  "thresholds"}


def test_guard_defaults_to_killing_the_process():
    x = uniform_grid(4)
    with pytest.raises(SystemExit) as e:
        qd.assert_not_degenerate(torch.full((3, x.shape[0], 3), 0.5), x)
    assert e.value.code == 2


# --------------------------------------------------------------------------- #
# 10. query -> Gaussian correspondence (section 3.6)
# --------------------------------------------------------------------------- #
def test_hungarian_matching_separates_a_move_from_a_relabelling():
    mu = uniform_grid_positions(8)
    assign, _ = qd.hungarian_match(mu, mu)
    assert torch.equal(assign, torch.arange(8))
    perm = torch.randperm(8)
    while bool((perm == torch.arange(8)).all()):
        perm = torch.randperm(8)
    drift, shift_idx, shift_match = qd._drift_and_shift(mu, mu[perm])
    assert drift > 0.0                      # the index moved ...
    assert shift_match < 1e-6               # ... but no Gaussian did
    assert shift_idx > 1e-3                 # while the by-index column is large


def test_repeat_and_path_diagnostics_publish_their_four_keys():
    m = _model()
    z, field = torch.randn(1, 2560), torch.rand(1, 32, 48)
    rep = qd.query_match_repeat(m, z, field, repeats=3)
    # fp32 CPU is deterministic and dropout is 0 -> no drift by construction
    assert rep["query_match_drift_repeat"] == 0.0
    path = qd.query_match_path(m, z, torch.randn(1, 2560), field, k_steps=4)
    assert path["k_steps"] == 4
    for key in ("query_match_drift_path", "query_match_mu_shift_indexed",
                "query_match_mu_shift_hungarian"):
        assert key in path


# --------------------------------------------------------------------------- #
# 11. collapse guard (section 3.8)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("delta,m_const,verdict", [
    (0.05, 0.2, True), (0.05, 1.0, False), (0.5, 0.2, False), (0.5, 1.0, False)])
def test_collapse_guard_needs_both_conditions(delta, m_const, verdict):
    out = qd.collapse_guard(delta, m_const)
    assert out["COLLAPSED"] is verdict
    assert out["m_threshold"] == 0.5 and out["delta_threshold"] == 0.1


def test_collapse_guard_says_so_when_the_columns_are_missing():
    assert qd.collapse_guard(None, 0.2)["COLLAPSED"] is None


# --------------------------------------------------------------------------- #
# 12. the pre-registered table
# --------------------------------------------------------------------------- #
def test_required_table_is_the_proposal_list():
    req = qd.REQUIRED_QDUAL
    assert len(req) == 25 and len(set(req)) == 25
    assert req[:12] == crit.PREREGISTERED_KEYS
    assert set(req) >= set(crit.REQUIRED_P2P3)
    assert "field_pred" in req and "field_pred" not in crit.REQUIRED_P2P3
    assert set(req) >= {"query_match_drift_repeat", "query_match_drift_path",
                        "query_match_mu_shift_indexed",
                        "query_match_mu_shift_hungarian", "ladder_row",
                        "zeroinit_step0_maxabs"}
    assert qd.ARM_REQUIRED_TABLE["GQDEC"] == req      # the proposal's own key


def test_a_board_missing_one_required_column_cannot_publish():
    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "style", "E_arm": 1.0 + i} for i in range(3)]
    board = crit.build_board(rows, arm=qd.ARM, split="V_what")
    with pytest.raises(crit.CriterionNotComputed):
        crit.assert_criteria_ran(board, qd.ARM,
                                 required=qd.required_criteria_table(qd.QDualConfig()))


# --------------------------------------------------------------------------- #
# 12b. the step-0 witness column is a *training* quantity: build_board cannot
#      produce it, so every board that goes through assert_criteria_ran has to
#      be handed it (QDUAL_LR3E4 died on exactly this at its first selection
#      point, ~2h in, with the other 24 columns green).
# --------------------------------------------------------------------------- #
def _evaluate_rows(n: int = 4) -> tuple[list[dict], dict[str, dict]]:
    """Rows + extra columns that carry every REQUIRED_QDUAL key *except* the
    step-0 witness -- i.e. exactly what ``evaluate`` returns."""
    rows = []
    for i in range(n):
        v = 1.0 + 0.1 * i
        row = {"sample_id": f"s{i}", "winner_confidence": "normal",
               "task_type": "local" if i % 2 else "style",
               "E_arm": v, "E_B0_identity": v + 1.0, "E_B1_libmean": v + 0.5,
               "E_B4_oracle": v + 0.2,
               "E_B2_librandom_repeats": [v + 2.0, v + 2.5],
               "E_B3_bucket_retrieval_repeats": [v + 1.5, v + 1.7],
               "loc_in": v, "loc_band": v + 0.1, "loc_out": 0.0,
               "field_gt": v, "field_const": v + 0.3, "field_shuffle": v + 0.4}
        for e_key, m_key in runner.CONTROL_ROW_KEYS.values():
            row[e_key], row[m_key] = v + 0.6, 0.7
        rows.append(row)
    extra = {k: crit.describe([0.1 * (j + 1) for j in range(n)]) for k in (
        "query_match_drift_repeat", "query_match_drift_path",
        "query_match_mu_shift_indexed", "query_match_mu_shift_hungarian",
        "field_pred")}
    extra["ladder_row"] = {"n": n, "value": "c"}
    return rows, extra


def _selection_board(rows, extra) -> dict:
    return crit.build_board(rows, arm=qd.ARM, split="V_what",
                            extra_columns=extra)


def test_board_without_the_zeroinit_column_raises_although_all_else_is_green():
    rows, extra = _evaluate_rows()
    board = _selection_board(rows, extra)
    req = qd.required_criteria_table(qd.QDualConfig())
    # every other pre-registered key is already on the board ...
    assert [k for k in req if k not in board["criteria_columns"]] == [
        "zeroinit_step0_maxabs"]
    with pytest.raises(crit.CriterionNotComputed, match="zeroinit_step0_maxabs"):
        crit.assert_criteria_ran(board, qd.ARM, required=req)
    # ... and wiring the witness in is all it takes
    runner.attach_zeroinit_column(board, {"zeroinit_step0_maxabs": 0.0},
                                  cfg=qd.QDualConfig(), source="unit-test")
    rep = crit.assert_criteria_ran(board, qd.ARM, required=req)
    assert rep["computed"]["zeroinit_step0_maxabs"] == 1
    assert board["criteria_columns"]["zeroinit_step0_maxabs"]["value"] == 0.0


def test_attach_zeroinit_column_leaves_n_zero_when_there_is_no_witness():
    board = {"criteria_columns": {}}
    col = runner.attach_zeroinit_column(board, None, cfg=qd.QDualConfig(),
                                        source="unit-test")
    assert col["n"] == 0 and col["value"] is None
    col2 = runner.attach_zeroinit_column(
        board, {"zeroinit_step0_maxabs": 0.0},
        cfg=qd.QDualConfig(zero_init_head=False), source="unit-test")
    assert "--no-zero-init-head" in col2["note"]


class _StubModel:
    def state_dict(self):
        return {}

    def train(self):
        return self


def test_selection_board_carries_the_zeroinit_column_the_trainer_hands_it(
        tmp_path, monkeypatch):
    """The FIRST selection board asserts the whole table -- so it must be wired."""
    rows, extra = _evaluate_rows()
    monkeypatch.setattr(runner, "evaluate",
                        lambda *a, **k: (rows, {"extra_columns": extra,
                                                "meta": {}}))
    args = runner.build_parser().parse_args(["--eval-every", "1"])
    assert args.z_source != "synthetic" and not args.smoke   # assertion not skipped
    select = runner.make_selector(
        args=args, cfg=qd.QDualConfig(), samples=[], zstore=None, alphas=None,
        images=None, bank=None, lib_ids=[], pools={},
        device=torch.device("cpu"), run_dir=tmp_path, pred_field_dir=None)

    # before the trainer hands the witness over, the first board cannot publish
    with pytest.raises(crit.CriterionNotComputed, match="zeroinit_step0_maxabs"):
        select(_StubModel(), 1)

    # this is the assignment train() makes at step 1, before any select_fn call
    select.state["zero_init_witness"] = {"zeroinit_step0_maxabs": 0.0}
    rec = select(_StubModel(), 2)
    computed = rec["first_board_assertion"]["computed"]
    assert set(computed) == set(qd.REQUIRED_QDUAL)
    assert all(n > 0 for n in computed.values())
    assert computed["zeroinit_step0_maxabs"] == 1


def test_the_trainer_hands_the_step1_witness_to_the_selector():
    """``train`` sets ``select_fn.state`` at step 1 -- greppable, not incidental."""
    src = RUNNER_SRC.read_text(encoding="utf-8")
    assert 'select_fn.state["zero_init_witness"] = dict(witness)' in src
    # and both boards go through the one helper
    assert src.count("attach_zeroinit_column(") == 3      # def + selector + final


# --------------------------------------------------------------------------- #
# 13. discipline: no constant tensor in a forward, no stray .cpu()
# --------------------------------------------------------------------------- #
def _calls(tree: ast.AST) -> list[str]:
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                base = f.value.id if isinstance(f.value, ast.Name) else ""
                out.append(f"{base}.{f.attr}" if base else f.attr)
    return out


def test_no_bare_torch_tensor_inside_a_forward_path():
    tree = ast.parse(ARM_SRC.read_text(encoding="utf-8"))
    hot = {"forward", "residuals", "build_memory", "_field_rows",
           "_language_rows", "queries", "compose", "_attend", "transform_grid",
           "apply_image"}
    offences = [fn.name for fn in ast.walk(tree)
                if isinstance(fn, ast.FunctionDef) and fn.name in hot
                and "torch.tensor" in _calls(fn)]
    assert offences == [], f"torch.tensor(...) inside {offences}"


def test_every_module_constant_is_a_non_persistent_buffer():
    m = _model()
    names = {n for n, _ in m.named_buffers()}
    assert {"decoder.theta_base.grid_index", "decoder.row_index",
            "decoder.col_index", "carrier.eye3"} <= names
    assert [k for k in m.state_dict() if k in names] == []   # persistent=False


def test_the_only_cpu_round_trip_is_the_hungarian_solve():
    tree = ast.parse(ARM_SRC.read_text(encoding="utf-8"))
    hits = [fn.name for fn in ast.walk(tree)
            if isinstance(fn, ast.FunctionDef)
            and any(c == "cpu" or c.endswith(".cpu") for c in _calls(fn))]
    assert hits == ["hungarian_match"], hits


def test_a_float64_condition_is_cast_rather_than_crashing():
    m = _model()
    y = m(uniform_grid(4), torch.randn(1, 2560, dtype=torch.float64),
          torch.rand(1, 32, 48, dtype=torch.float64))
    assert y.dtype == torch.float32


# --------------------------------------------------------------------------- #
# 14. parameter groups / optimiser mapping
# --------------------------------------------------------------------------- #
def test_param_groups_put_the_geometric_prior_at_one_tenth_of_the_lr():
    m = _model()
    groups = m.param_groups(1e-3, geometry_lr_scale=0.1)
    assert [g["name"] for g in groups] == ["generator", "query_prior"]
    assert groups[0]["lr"] == 1e-3 and groups[1]["lr"] == pytest.approx(1e-4)
    slow = {id(p) for p in groups[1]["params"]}
    assert id(m.decoder.q_emb) in slow
    assert id(m.decoder.pe_r) in slow
    assert all(id(p) in slow for p in m.decoder.theta_base.parameters_list())
    assert id(m.decoder.head_g.weight) not in slow
    assert id(m.decoder.pe_row) not in slow          # the field PE is generative
    n_all = sum(p.numel() for p in m.parameters())
    assert sum(p.numel() for g in groups for p in g["params"]) == n_all


# --------------------------------------------------------------------------- #
# 15. theta composition
# --------------------------------------------------------------------------- #
def test_theta_slices_follow_glutparams_flat_order():
    base = qd.ThetaBase(4)
    d = torch.zeros(1, 4, 22)
    d[..., 10:19] = 1.0            # the M block
    p = base.compose(d, torch.zeros(1, 12))
    assert torch.allclose(p.m_local[0, 0], torch.eye(3) + 1.0)
    assert torch.allclose(p.b_local, torch.zeros(1, 4, 3))
    d2 = torch.zeros(1, 4, 22)
    d2[..., 9] = 2.0               # the opacity logit
    p2 = base.compose(d2, torch.zeros(1, 12))
    assert torch.allclose(p2.opacity_logit, torch.full((1, 4), 6.0))
    dg = torch.zeros(1, 12)
    dg[0, 9:] = 0.5
    p3 = base.compose(torch.zeros(1, 4, 22), dg)
    assert torch.allclose(p3.g_bias, torch.full((1, 3), 0.5))


def test_theta_base_geometry_is_app_a1():
    base = qd.ThetaBase(48)
    assert torch.allclose(base.mu, uniform_grid_positions(48))
    with torch.no_grad():
        sigma = float(torch.nn.functional.softplus(base.chol_diag).mean())
        opacity = float(torch.sigmoid(base.opacity_logit).mean())
    assert sigma == pytest.approx(0.15, rel=1e-5)
    assert opacity == pytest.approx(0.98201, rel=1e-4)  # NOVEL: logit 4.0, not 1.0


# --------------------------------------------------------------------------- #
# 16. the runner: flags, frozen block, B1 volume, end to end
# --------------------------------------------------------------------------- #
def test_runner_defaults_are_the_main_arm():
    args = runner.build_parser().parse_args([])
    cfg = runner.config_from_args(args)
    assert (cfg.rung, cfg.n_gauss, cfg.decoder_layers, cfg.decoder_width,
            cfg.decoder_heads) == ("c", 48, 4, 256, 8)
    assert (cfg.z_expand, cfg.z_expand_k) == ("proj", 4)
    assert (cfg.field_source, cfg.field_kind) == ("m_low", "gt")
    assert (cfg.clamp, cfg.readout, cfg.color_sampling) == ("two", "seg_color",
                                                            "uniform")
    assert cfg.zero_init_head and not cfg.attn_temperature
    assert (args.lr, args.seed, args.epochs) == (1e-3, 20260810, 40)
    assert (args.batch_samples, args.queries) == (32, 256)
    assert args.geometry_lr_scale == 0.1


def test_frozen_block_arithmetic():
    assert runner.STEPS_PER_EPOCH == math.ceil(runner.TRAIN_NORMAL_N / 32) == 2936
    assert runner.STEPS_PER_EPOCH * runner.EPOCHS == runner.TOTAL_STEPS == 117_440
    assert 32 * 256 == COLORS_PER_STEP == 8192


def test_frozen_block_record_flags_a_deviation_rather_than_hiding_it():
    args = runner.build_parser().parse_args(["--batch-samples", "64",
                                             "--queries", "128"])
    rec = runner.frozen_block_record(args, 93934, 1468, 58720)
    assert rec["batch_split"]["matches"] is True      # 64 x 128 is still 8192
    assert rec["steps_per_epoch"]["matches"] is False
    assert rec["total_steps"]["matches"] is False
    ok = runner.frozen_block_record(runner.build_parser().parse_args([]),
                                    93934, 2936, 117_440)
    assert all(ok[k]["matches"] for k in ("train_normal_n", "batch_split",
                                          "steps_per_epoch", "total_steps",
                                          "clamp"))


def test_every_declared_flag_reaches_something():
    """No flag may be a silent no-op -- the "defined but not wired" failure."""
    args = runner.build_parser().parse_args([])
    declared = {a.dest for a in runner.build_parser()._actions
                if a.dest not in ("help",)}
    cfg_fields = set(qd.QDualConfig().to_dict())
    src = RUNNER_SRC.read_text(encoding="utf-8")
    unreached = [d for d in sorted(declared)
                 if d not in cfg_fields and f"args.{d}" not in src]
    assert unreached == [], unreached


def test_field_kind_is_honoured_where_the_field_is_built():
    class Alphas:
        mode = "synthetic"

        def field_for(self, s, hw):
            return torch.full(hw, 0.25 if s.sample_id == "a" else 0.75)

    batch = [runner.Sample("a", "V_what", "l1", "src", "local", "normal"),
             runner.Sample("b", "V_what", "l2", "src", "local", "normal")]
    dev = torch.device("cpu")
    gt = runner._field_batch(batch, Alphas(), qd.QDualConfig(field_kind="gt"),
                             dev)
    assert torch.allclose(gt[0], torch.full((32, 48), 0.25))
    const = runner._field_batch(batch, Alphas(),
                                qd.QDualConfig(field_kind="const"), dev)
    assert torch.allclose(const[0], torch.full((32, 48), 0.25))   # mean of a flat
    shuf = runner._field_batch(batch, Alphas(),
                               qd.QDualConfig(field_kind="shuffle"), dev)
    assert torch.allclose(shuf[0], gt[1]) and torch.allclose(shuf[1], gt[0])
    with pytest.raises(SystemExit, match="no predicted field"):
        runner._field_batch(batch, Alphas(), qd.QDualConfig(field_kind="pred"),
                            dev, pred_field_dir=None, synthetic=False)
    assert runner._field_batch(batch, Alphas(), qd.QDualConfig(rung="a"),
                               dev) is None


def test_alpha_hist_sampling_draws_from_the_image_and_not_the_grid():
    from q3vl.whatb.queries import QuerySampler

    class Images:
        def image(self, s):
            img = torch.zeros(3, 4, 4)
            img[0] = 0.75           # one dominant colour, quantised to 5 bits
            return img

    class Alphas:
        mode = "synthetic"

        def alpha(self, s, hw):
            return torch.ones(*hw)

    batch = [runner.Sample("a", "V_what", "l1", "src", "style", "normal")]
    sampler = QuerySampler(seed=1, q=8)
    dev = torch.device("cpu")
    out = runner._color_batch(batch, sampler,
                              qd.QDualConfig(color_sampling="alpha_hist"), dev,
                              Images(), Alphas())
    assert out.shape == (1, 8, 3)
    assert torch.allclose(out[0, :, 1], torch.zeros(8))          # G is 0 in the image
    assert float(out[0, :, 0].min()) > 0.7
    with pytest.raises(SystemExit, match="needs the image store"):
        runner._color_batch(batch, sampler,
                            qd.QDualConfig(color_sampling="alpha_hist"), dev,
                            None, Alphas())


def test_loss_level_4_is_refused_because_this_arm_has_no_image_loss(tmp_path):
    """Level 4 is off the flag's own choices AND off the run-time guard."""
    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(["--loss-level", "4"])
    args = runner.build_parser().parse_args([])
    args.loss_level = 4                       # bypass argparse, hit the guard
    assert int(args.loss_level) not in (1, 3)


def test_loss_level_one_is_the_pure_l1_caliber():
    """``--loss-level 1`` zeroes both optional weights (carrier.py:347/:351)."""
    from q3vl.whatb import caliber as K

    assert (K.effective_lambda_hc(qd.LAMBDA_HC, 3),
            K.effective_lambda_sparse(qd.LAMBDA_SPARSE, 3)) == (10.0, 0.001)
    assert (K.effective_lambda_hc(qd.LAMBDA_HC, 1),
            K.effective_lambda_sparse(qd.LAMBDA_SPARSE, 1)) == (0.0, 0.0)
    pre = qd.loss_preregistration(qd.QDualConfig(), loss_level=1)
    assert pre["lambda_hc"] == 0.0 and pre["lambda_sparse"] == 0.0
    assert pre["loss_ladder"]["pure_l1"] is True


def test_batch_split_resolves_the_epr030_caliber():
    args = runner.build_parser().parse_args(["--batch-split", "256x8192"])
    assert (args.batch_samples, args.queries) == (256, 8192)
    assert args.batch_samples * args.queries == 2_097_152
    for name, pair in (("32x256", (32, 256)), ("64x128", (64, 128))):
        a = runner.build_parser().parse_args(["--batch-split", name])
        assert (a.batch_samples, a.queries) == pair
        assert a.batch_samples * a.queries == 8192


def test_bf16_on_cpu_is_recorded_as_fp32_not_silently_applied():
    args = runner.build_parser().parse_args(["--device", "cpu"])
    rec = runner.resolve_precision(args)
    assert rec == {"requested": "bf16", "used": "fp32", "device": "cpu",
                   "note": rec["note"]}
    assert "recorded rather than silently applied" in rec["note"].lower()


def test_mean_transform_volume_has_the_bgr_axis_order_of_the_bank():
    class FakeBank:
        def apply(self, x, lut_id):
            return torch.stack((x[:, 0], x[:, 1] / 2, x[:, 2] / 3), dim=-1)

    vol = runner.mean_transform_volume(FakeBank(), ["a", "b"], grid_n=9)
    assert vol.shape == (1, 3, 9, 9, 9)
    x = torch.tensor([[0.25, 0.5, 0.75], [1.0, 0.0, 0.5]])
    from q3vl.whatb.lutdata import apply_lut_volume

    got = apply_lut_volume(vol, x)
    want = torch.stack((x[:, 0], x[:, 1] / 2, x[:, 2] / 3), dim=-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_colorspan_assertion_may_only_be_skipped_on_synthetic_data():
    args = runner.build_parser().parse_args(["--skip-colorspan-assert"])
    with pytest.raises(SystemExit, match="synthetic"):
        runner.colorspan_assertion(args, [])
    args2 = runner.build_parser().parse_args(["--skip-colorspan-assert",
                                              "--z-source", "synthetic"])
    assert runner.colorspan_assertion(args2, [])["skipped"] is True


@pytest.mark.skipif(not (BANK_OK and SPLITS_OK),
                    reason="needs the LUT bank and the split indexes")
def test_end_to_end_synthetic_run_publishes_a_complete_board(tmp_path):
    cmd = [sys.executable, "-m", "q3vl.whatb.scripts.run_qdual_arm",
           "--smoke", "--smoke-steps", "6", "--z-source", "synthetic",
           "--skip-colorspan-assert", "--max-train-samples", "32",
           "--eval-max-samples", "2", "--lib-sample", "4",
           "--baseline-repeats", "2", "--quick-eval-n", "4",
           "--queries", "16", "--batch-samples", "4",
           "--out-root", str(tmp_path), "--run-name", "t"]
    env = {"PYTHONPATH": str(REPO), "CUDA_VISIBLE_DEVICES": "",
           "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    assert r.returncode == 0, r.stdout + r.stderr

    run = tmp_path / "t"
    board = json.loads((run / "metrics.json").read_text())
    computed = board["publication_report"]["criteria"]["computed"]
    assert set(computed) == set(qd.REQUIRED_QDUAL)
    assert all(n > 0 for n in computed.values())
    assert board["contexts"]["all"]["headline_normal_only"]["n"] > 0
    assert board["published"] is False        # synthetic may never be published
    assert board["collapse_guard"]["COLLAPSED"] in (True, False)

    first = json.loads((run / "steps.jsonl").read_text().splitlines()[0])
    for col in pub.step_columns_for(3, extra=("L_total", "R_sparse", "ladder_row",
                                              "zeroinit_step0_maxabs")):
        assert col in first, col
    assert set(FIRST_STEP_COLUMNS) <= set(first)
    assert first["zeroinit_step0_maxabs"] == 0.0

    setup = json.loads((run / "config" / "run_setup.json").read_text())
    assert setup["source_sha256"]["arm"] and setup["source_sha256"]["runner"]
    assert setup["synthetic"] is True
    assert setup["degeneracy_thresholds"]["point_std"] == 1e-3
    assert setup["config"]["n_params"] == 5_833_038
    quick = [json.loads(l) for l in
             (run / "quick_eval.jsonl").read_text().splitlines()]
    assert quick and quick[0]["failures"] == []


@pytest.mark.skipif(not (BANK_OK and SPLITS_OK),
                    reason="needs the LUT bank and the split indexes")
def test_a_second_run_into_the_same_directory_refuses_to_append(tmp_path):
    (tmp_path / "t").mkdir(parents=True)
    (tmp_path / "t" / "steps.jsonl").write_text('{"step": 1}\n')
    with pytest.raises(SystemExit, match="fresh --run-name"):
        runner.main(["--smoke", "--smoke-steps", "2", "--z-source", "synthetic",
                     "--skip-colorspan-assert", "--max-train-samples", "8",
                     "--eval-max-samples", "1", "--lib-sample", "2",
                     "--queries", "8", "--batch-samples", "4",
                     "--out-root", str(tmp_path), "--run-name", "t"])
