"""EPR-026 INTERPC: config, pairing, loss, guards, criteria columns, runner.

Everything here runs on the CPU with ``CUDA_VISIBLE_DEVICES=""``; the two GPUs
are occupied by the where-side UNIQ arms and nothing in this arm needs one.

The tests are grouped by what they pin:

* the **frozen block** (B*Q = 8192, 2936 steps/epoch, 117,440 steps, clamp two,
  the twelve keys plus the four P1 keys),
* the **one change** (``lambda_int = 0`` is EPR-024 bit-for-bit; the pair stream
  is short-circuited; ``post_pi`` and ``pre_pi`` really differ),
* the **three where-side failures** the task card requires to be closed
  structurally (assertion contract, device/dtype, degenerate solutions), and
* the arithmetic the proposal quotes (pair counts, ICT's ramp, ACAI's alpha
  clip, the mining schedule).
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb.arms import interpc as ic
from q3vl.whatb.criteria import assert_criteria_ran, build_board
from q3vl.whatb.guards import (
    LossColumnsMissing,
    StepsRowUnavailable,
    clear_step_witness,
)
from q3vl.whatb.queries import uniform_grid
from q3vl.whatb.scripts import run_interpc_arm as runner

ARM_SRC = Path(ic.__file__)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def tiny_cfg(**kw) -> ic.InterpcConfig:
    """Small but *structurally frozen*: B*Q is still 8192 (4 x 2048)."""
    base = dict(batch_samples=4, queries_per_sample=2048, n_gauss=8, gen_width=16,
                cond_dim=8, eval_grid_n=5, ipb_k=4, total_steps=8,
                steps_per_epoch=2)
    base.update(kw)
    return ic.InterpcConfig(**base)


@pytest.fixture(autouse=True)
def _clean_process_state():
    ic.reset_degeneracy_check()
    clear_step_witness()
    yield
    ic.reset_degeneracy_check()
    clear_step_witness()


@pytest.fixture()
def arm() -> ic.InterpcArm:
    torch.manual_seed(0)
    return ic.InterpcArm(tiny_cfg())


def fake_rows(spec):
    """``spec = {source: [(sample_id, lut_id, confidence)]}`` -> index-like rows."""
    out = []
    for src, members in spec.items():
        for sid, lid, conf in members:
            out.append({"sample_id": sid, "lut_id": lid, "source_image_id": src,
                        "task_type": "style", "winner_confidence": conf})
    return out


def synth_batch(cfg: ic.InterpcConfig, *, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(cfg.batch_samples, 2560, generator=g)
    x = torch.rand(cfg.batch_samples, cfg.queries_per_sample, 3, generator=g)
    gains = torch.tensor([[1.1, 0.9, 1.0], [0.8, 1.0, 1.2], [1.0, 1.2, 0.8],
                          [0.9, 1.1, 1.05]])
    target = (x * gains[: cfg.batch_samples, None, :]).clamp(0, 1)
    return ic.FitBatch(z=z, x=x, target=target,
                       lut_ids=tuple(f"lut{i}" for i in range(cfg.batch_samples)))


def synth_pair(cfg: ic.InterpcConfig, fit: ic.FitBatch, alpha: float | None = None):
    p = min(cfg.pairs_per_step, cfg.batch_samples)
    g = torch.Generator().manual_seed(7)
    a = (torch.full((p, 1), float(alpha)) if alpha is not None
         else torch.rand(p, 1, generator=g))
    return ic.PairBatch(z_a=fit.z[:p], z_b=fit.z.flip(0)[:p], x=fit.x[:p],
                        values_a=fit.target[:p], values_b=fit.target.flip(0)[:p],
                        alpha=a)


# --------------------------------------------------------------------------- #
# 1. frozen block
# --------------------------------------------------------------------------- #
def test_default_config_is_the_frozen_block():
    cfg = ic.InterpcConfig()
    assert cfg.batch_samples * cfg.queries_per_sample == 8192
    assert (cfg.batch_samples, cfg.queries_per_sample) == (32, 256)
    assert cfg.steps_per_epoch == 2936 == -(-93934 // 32)
    assert cfg.total_steps == 117_440 == 2936 * 40
    assert cfg.clamp == "two"
    assert cfg.eps == 1e-6 and cfg.hc_eps == 1e-3 and cfg.hc_mask is True
    assert (cfg.lambda_hc, cfg.lambda_sparse) == (10.0, 0.001)
    assert cfg.lr == 1e-3 and cfg.pi_lr_scale == 0.1 and cfg.scheduler == "cosine"
    # the arm's own defaults: lambda_int = 0 is the EPR-024 baseline row
    assert cfg.interp_weight == 0.0 and cfg.interp_enabled is False
    assert cfg.interp_alpha == "beta0.5" and cfg.beta == 0.5
    assert cfg.interp_p_end == pytest.approx(1 / 3)
    assert cfg.interp_where == "post_pi" and cfg.interp_dist == "l1"
    assert cfg.interp_target == "gt_mix" and cfg.interp_ramp == "const"
    assert cfg.pairs_per_step == 32               # 1:1 with the fit stream


def test_batch_split_is_recorded_not_pinned_to_8192():
    """EPR-030 (2026-08-16) opened the colour budget; the product is RECORDED.

    The two step-matched rows still resolve to exactly 8192 and still report
    ``step_matched_to_epr024``; a capacity row reports ``False``.
    """
    for b, q in ((32, 256), (64, 128)):
        ok = ic.InterpcConfig(batch_samples=b, queries_per_sample=q)
        assert ok.colors_per_step == 8192
        assert ok.batch_split == f"{b}x{q}"
        assert ok.step_matched_to_epr024 is True
    big = ic.InterpcConfig(batch_samples=256, queries_per_sample=8192,
                           train_n=119828)
    assert big.colors_per_step == 2_097_152
    assert big.batch_split == "256x8192"
    assert big.step_matched_to_epr024 is False
    assert big.as_dict()["train_normal_n"] == 119828


def test_loss_level_one_zeroes_both_optional_weights():
    """``--loss-level 1`` = the single L1 term (carrier.py:347/:351's rule)."""
    base = ic.InterpcConfig()
    assert (base.lambda_hc_effective, base.lambda_sparse_effective) == (10.0, 0.001)
    pure = ic.InterpcConfig(loss_level=1)
    assert (pure.lambda_hc_effective, pure.lambda_sparse_effective) == (0.0, 0.0)


def test_step_columns_baseline_vs_treatment():
    base = ic.InterpcConfig().step_columns()
    treat = ic.InterpcConfig(interp_weight=1.0).step_columns()
    for c in ("L_rec", "L_hc", "L_sparse", "n_colors", "n_luts_in_batch",
              "mining_ratio", "n_hc_masked"):
        assert c in base and c in treat
    assert "L_interp" not in base and "L_interp" in treat
    assert "L_img" in ic.InterpcConfig(loss_level=4).step_columns()
    assert "L_acai" in ic.InterpcConfig(alternate="acai").step_columns()
    assert "L_jac" in ic.InterpcConfig(alternate="jacobian").step_columns()


def test_mutually_exclusive_and_undecided_flags_refuse_rather_than_guess():
    with pytest.raises(ValueError, match="mutually exclusive"):
        ic.InterpcConfig(interp_weight=1.0, alternate="acai")
    with pytest.raises(ValueError, match="interp-finetune-start"):
        ic.InterpcConfig(interp_weight=1.0, interp_stage="finetune")
    ic.InterpcConfig(interp_weight=1.0, interp_stage="finetune",
                     interp_finetune_start=1000)     # explicit -> allowed
    with pytest.raises(ValueError):
        ic.InterpcConfig(interp_alpha="beta2.0")


def test_arm_registers_itself_on_the_P1_axis():
    from q3vl.whatb.criteria import ARM_AXES, required_criteria

    assert ic.ARM == "EPR-026" and ARM_AXES[ic.ARM] == ("P1",)
    req = required_criteria(ic.ARM)
    for k in ("interp_grid", "path_len", "mono_rate", "oob_rate",
              "headline_normal_only", "B3_bucket_retrieval", "N3_const_M"):
        assert k in req
    assert len(req) == 16


# --------------------------------------------------------------------------- #
# 2. the pairing index
# --------------------------------------------------------------------------- #
def test_pair_index_keeps_only_sources_with_two_luts():
    rows = fake_rows({
        "srcA": [("a1", "L1", "normal"), ("a2", "L2", "normal")],
        "srcB": [("b1", "L1", "normal"), ("b2", "L1", "normal")],   # one LUT
        "srcC": [("c1", "L1", "low"), ("c2", "L2", "normal")],      # low dropped
        "srcD": [("d1", "L1", "normal"), ("d2", "L2", "normal"),
                 ("d3", "L2", "normal")],
    })
    idx = ic.PairIndex.build(rows, split="t")
    assert list(idx.sources) == ["srcA", "srcD"]
    assert idx.n_pairs == 1 + 2                    # A: 1 ; D: (d1d2, d1d3)
    assert sorted(idx.all_pairs(1)) == [(0, 1), (0, 2)]


def test_pair_count_formula_matches_brute_force_enumeration():
    rng = np.random.default_rng(3)
    spec = {}
    for s in range(25):
        n = int(rng.integers(1, 7))
        spec[f"s{s}"] = [(f"s{s}_{i}", f"L{int(rng.integers(0, 3))}", "normal")
                         for i in range(n)]
    idx = ic.PairIndex.build(fake_rows(spec), split="t")
    brute = sum(len(idx.all_pairs(i)) for i in range(len(idx)))
    assert idx.n_pairs == brute > 0


def test_pair_draw_always_returns_two_different_luts_and_is_reproducible():
    rows = fake_rows({"s": [("x1", "L1", "normal"), ("x2", "L1", "normal"),
                            ("x3", "L2", "normal")]})
    idx = ic.PairIndex.build(rows, split="t")
    draws = idx.draw(64, np.random.default_rng(0))
    assert len(draws) == 64
    assert all(a[1] != b[1] for _, a, b in draws)
    again = idx.draw(64, np.random.default_rng(0))
    assert draws == again
    assert idx.sha256 == ic.PairIndex.build(rows, split="t").sha256
    assert idx.facts()["n_pairs"] == 2


@pytest.mark.skipif(not Path("/mnt/nfs-ro/bc/data/datasets/sft2seg-20260804/"
                             "splits/V_what.index.jsonl").is_file(),
                    reason="dataset split index not mounted")
def test_pair_counts_on_the_real_splits_match_the_proposal():
    """EPR-026 section 1.1 / 3.6: V_what normal-only = 120 sources / 1311 pairs."""
    from q3vl.whatb.splits import load_index

    idx = ic.PairIndex.build(load_index("V_what"), split="V_what")
    facts = idx.facts()
    assert (facts["n_sources_with_pairs"], facts["n_pairs"]) == (120, 1311)
    assert facts["group_size_max"] == 12 and facts["group_size_median"] == 4.0
    t = ic.PairIndex.build(load_index("T_lut_unseen"), split="T_lut_unseen").facts()
    assert (t["n_sources_with_pairs"], t["n_pairs"]) == (67, 137)


# --------------------------------------------------------------------------- #
# 3. alpha sampling
# --------------------------------------------------------------------------- #
def test_alpha_sampler_endpoint_mass_and_support():
    s = ic.AlphaSampler(tiny_cfg(interp_weight=1.0, interp_p_end=1 / 3))
    a = s.sample_numpy(20000)
    assert ((a >= 0) & (a <= 1)).all()
    frac_end = float(np.mean((a == 0.0) | (a == 1.0)))
    assert 0.30 < frac_end < 0.37                  # p_end = 1/3 plus Beta's own mass
    zero = ic.AlphaSampler(tiny_cfg(interp_weight=1.0, interp_p_end=0.0))
    assert float(np.mean(zero.sample_numpy(5000) == 0.0)) < 0.01
    ends = ic.AlphaSampler(tiny_cfg(interp_weight=1.0, interp_alpha="endpoints"))
    assert set(np.unique(ends.sample_numpy(200))) <= {0.0, 1.0}


def test_alpha_sampler_does_not_touch_the_global_rng_streams():
    np.random.seed(1234)
    torch.manual_seed(1234)
    before_np = np.random.rand(3).tolist()
    before_t = torch.rand(3).tolist()
    np.random.seed(1234)
    torch.manual_seed(1234)
    ic.AlphaSampler(tiny_cfg(interp_weight=1.0)).sample(1000)
    assert np.random.rand(3).tolist() == before_np
    assert torch.rand(3).tolist() == before_t


def test_alpha_sample_shape_and_device_dtype():
    a = ic.AlphaSampler(tiny_cfg(interp_weight=1.0)).sample(5, dtype=torch.float64)
    assert a.shape == (5, 1) and a.dtype == torch.float64


# --------------------------------------------------------------------------- #
# 4. the model
# --------------------------------------------------------------------------- #
def test_arm_adds_no_parameters_and_puts_pi_on_the_0_1x_group(arm):
    n_pi = sum(p.numel() for p in arm.pi.parameters())
    n_gen = sum(p.numel() for p in arm.generator.parameters())
    assert sum(p.numel() for p in arm.parameters()) == n_pi + n_gen
    assert sum(p.numel() for p in arm.carrier.parameters()) == 0
    assert arm.facts()["n_params_added_vs_epr024"] == 0
    groups = arm.param_groups()
    by_name = {g["name"]: g for g in groups}
    assert by_name["pi"]["lr"] == pytest.approx(1e-4)
    assert by_name["generator"]["lr"] == pytest.approx(1e-3)


def test_constants_are_non_persistent_buffers_not_forward_time_tensors(arm):
    assert "grid_eval" in dict(arm.named_buffers())
    assert not any(k.endswith("grid_eval") for k in arm.state_dict())
    src = ARM_SRC.read_text()
    tree = ast.parse(src)
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
                "forward", "transform_from_u", "mix_condition", "condition"):
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr in ("tensor", "as_tensor")
                        and getattr(sub.func.value, "id", "") == "torch"):
                    offenders.append(node.name)
    assert offenders == []


def test_forward_casts_condition_and_queries_onto_the_module(arm):
    z = torch.randn(3, 2560, dtype=torch.float64)
    x = torch.rand(3, 32, 3, dtype=torch.float64)
    y = arm(z, x)
    assert y.shape == (3, 32, 3)
    assert torch.isfinite(y).all()
    y2 = arm(z.float(), uniform_grid(4))            # shared (P,3) query set
    assert y2.shape == (3, 64, 3)


def test_mix_condition_endpoints_are_exactly_pi(arm):
    z_a, z_b = torch.randn(2, 2560), torch.randn(2, 2560)
    zero, one = torch.zeros(2, 1), torch.ones(2, 1)
    assert torch.equal(arm.mix_condition(z_a, z_b, zero), arm.condition(z_a))
    assert torch.equal(arm.mix_condition(z_a, z_b, one), arm.condition(z_b))


def test_post_pi_and_pre_pi_are_different_maps():
    """``pi`` carries a LayerNorm, so ablation 5 is a real change (EPR-026:413)."""
    torch.manual_seed(0)
    post = ic.InterpcArm(tiny_cfg(interp_weight=1.0))
    pre = ic.InterpcArm(tiny_cfg(interp_weight=1.0, interp_where="pre_pi"))
    pre.load_state_dict(post.state_dict())
    z_a, z_b = torch.randn(2, 2560), torch.randn(2, 2560) * 3.0
    a = torch.full((2, 1), 0.5)
    u_post = post.mix_condition(z_a, z_b, a)
    u_pre = pre.mix_condition(z_a, z_b, a)
    assert not torch.allclose(u_post, u_pre, atol=1e-4)
    # ... and both agree at the endpoints
    for end in (0.0, 1.0):
        e = torch.full((2, 1), end)
        assert torch.allclose(post.mix_condition(z_a, z_b, e),
                              pre.mix_condition(z_a, z_b, e), atol=1e-6)


def test_step0_witness_reports_a_number_on_the_grid(arm):
    w = arm.step0_witness(torch.randn(2, 2560))
    assert w["grid_n"] == 5 and w["step0_maxabs_f_minus_id"] >= 0.0


# --------------------------------------------------------------------------- #
# 5. losses
# --------------------------------------------------------------------------- #
def test_glut_loss_terms_match_hand_computation_and_mask_low_chroma():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    x = torch.rand(2, 64, 3)
    y, aux = arm(torch.randn(2, 2560), x, return_aux=True)
    # a grey target: chroma is ~0 everywhere, so every point is masked
    grey = x.mean(dim=-1, keepdim=True).expand_as(x).contiguous()
    terms = ic.glut_loss_terms(y, grey, aux, cfg)
    assert terms["n_hc_masked"] == 2 * 64
    assert float(terms["L_hc"].detach()) == pytest.approx(0.0, abs=1e-6)
    assert float(terms["L_rec"].detach()) == pytest.approx(
        float((y - grey).abs().mean().detach()))
    o = aux.opacity
    want = -(o * torch.log(o + cfg.eps) + (1 - o) * torch.log(1 - o + cfg.eps)).mean()
    assert float(terms["L_sparse"].detach()) == pytest.approx(
        float(want.detach()), rel=1e-6)
    assert float(terms["L_glut"].detach()) == pytest.approx(
        float((terms["L_rec"] + 10.0 * terms["L_hc"]
               + 0.001 * terms["L_sparse"]).detach()), rel=1e-6)


def test_loss_levels_stack_additively():
    cfg1, cfg2, cfg3 = tiny_cfg(loss_level=1), tiny_cfg(loss_level=2), tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg3)
    x = torch.rand(2, 64, 3)
    y, aux = arm(torch.randn(2, 2560), x, return_aux=True)
    tgt = (x * 1.2).clamp(0, 1)
    t1 = ic.glut_loss_terms(y, tgt, aux, cfg1)
    t2 = ic.glut_loss_terms(y, tgt, aux, cfg2)
    t3 = ic.glut_loss_terms(y, tgt, aux, cfg3)
    assert float(t1["L_glut"].detach()) == pytest.approx(float(t1["L_rec"].detach()))
    assert float(t2["L_glut"].detach()) == pytest.approx(
        float((t1["L_rec"] + 10.0 * t2["L_hc"]).detach()), rel=1e-6)
    assert float(t3["L_glut"].detach()) > float(t2["L_glut"].detach())
    with pytest.raises(ValueError, match="loss-level 4"):
        ic.glut_loss_terms(y, tgt, aux, tiny_cfg(loss_level=4))


def test_hc_mask_off_is_the_shared_ablation_row():
    cfg = tiny_cfg(hc_mask=False)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    x = torch.rand(2, 64, 3)
    y, aux = arm(torch.randn(2, 2560), x, return_aux=True)
    grey = x.mean(dim=-1, keepdim=True).expand_as(x).contiguous()
    terms = ic.glut_loss_terms(y, grey, aux, cfg)
    assert terms["n_hc_masked"] == 0


def test_interp_distance_forms():
    a, b = torch.rand(4, 3), torch.rand(4, 3)
    assert float(ic.interp_distance(a, b, "l1")) == pytest.approx(
        float((a - b).abs().mean()))
    assert float(ic.interp_distance(a, b, "mse")) == pytest.approx(
        float(((a - b) ** 2).mean()))
    with pytest.raises(ValueError):
        ic.interp_distance(a, b, "huber")


def test_sigmoid_rampup_is_the_ICT_formula():
    """``exp(-5 (1 - t)^2)`` -- ICT ``mean_teacher/ramps.py::sigmoid_rampup``."""
    assert ic.sigmoid_rampup(0, 100) == pytest.approx(math.exp(-5.0))
    assert ic.sigmoid_rampup(50, 100) == pytest.approx(math.exp(-5.0 * 0.25))
    assert ic.sigmoid_rampup(100, 100) == pytest.approx(1.0)
    assert ic.sigmoid_rampup(250, 100) == pytest.approx(1.0)     # clipped
    assert ic.sigmoid_rampup(5, 0) == 1.0


def test_interp_weight_schedules():
    const = tiny_cfg(interp_weight=1.0)
    assert ic.interp_weight_at(0, const) == 1.0
    assert ic.interp_weight_at(10**6, const) == 1.0
    assert ic.interp_weight_at(0, tiny_cfg()) == 0.0             # baseline row
    ramp = tiny_cfg(interp_weight=1.0, interp_ramp="sigmoid", total_steps=400)
    assert ic.interp_weight_at(0, ramp) == pytest.approx(math.exp(-5.0))
    assert ic.interp_weight_at(100, ramp) == pytest.approx(1.0)  # 1/4 of 400
    ft = tiny_cfg(interp_weight=1.0, interp_stage="finetune",
                  interp_finetune_start=5)
    assert ic.interp_weight_at(4, ft) == 0.0 and ic.interp_weight_at(5, ft) == 1.0


# --------------------------------------------------------------------------- #
# 6. mining (ruling 11.1-4)
# --------------------------------------------------------------------------- #
def test_mining_schedule_matches_App_A1():
    cfg = tiny_cfg(steps_per_epoch=10)
    assert ic._mining_ratio_for(cfg, 0, None) == pytest.approx(0.10)
    assert ic._mining_ratio_for(cfg, 50, None) == pytest.approx(0.10)   # epoch 5
    assert ic._mining_ratio_for(cfg, 125, None) == pytest.approx(0.25)  # epoch 12.5
    assert ic._mining_ratio_for(cfg, 300, None) == pytest.approx(0.40)  # epoch 30
    assert ic._mining_ratio_for(tiny_cfg(mining=False), 300, None) == 0.0
    assert ic._mining_ratio_for(cfg, 300, 0.123) == pytest.approx(0.123)


def test_mine_step_keeps_the_batch_shape_and_picks_the_worst_colours(arm):
    z = torch.randn(2, 2560)
    x_pool = torch.rand(2, 16, 3)
    t_pool = torch.rand(2, 16, 3)
    x_fresh = torch.rand(2, 16, 3)
    t_fresh = torch.rand(2, 16, 3)
    x, t, n_hard = ic.mine_step(arm, z, x_pool, t_pool, x_fresh, t_fresh, 0.25)
    assert x.shape == (2, 16, 3) and t.shape == (2, 16, 3) and n_hard == 8
    err = (arm(z, x_pool) - t_pool).abs().mean(-1)
    worst = torch.topk(err, 4, dim=1).indices
    for b in range(2):
        kept = {tuple(v.tolist()) for v in x[b, :4]}
        assert kept == {tuple(x_pool[b, i].tolist()) for i in worst[b]}
    x0, t0, n0 = ic.mine_step(arm, z, x_pool, t_pool, x_fresh, t_fresh, 0.0)
    assert n0 == 0 and torch.equal(x0, x_fresh) and torch.equal(t0, t_fresh)


# --------------------------------------------------------------------------- #
# 7. the step: lambda_int = 0 is EPR-024 bit-for-bit
# --------------------------------------------------------------------------- #
def test_baseline_row_is_bit_for_bit_and_carries_no_interp_column():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    loss_a, row_a = ic.train_step(arm, fit, cfg, step=0)
    # handing in a pair batch changes nothing: the stream is short-circuited
    loss_b, row_b = ic.train_step(arm, fit, cfg, step=0, pair=synth_pair(cfg, fit))
    assert torch.equal(loss_a, loss_b)
    assert "L_interp" not in row_a and "L_interp" not in row_b
    y, aux = arm(fit.z, fit.x, return_aux=True)
    assert float(loss_a.detach()) == pytest.approx(
        float(ic.glut_loss_terms(y, fit.target, aux, cfg)["L_glut"].detach()), rel=1e-6)
    for col in cfg.step_columns():
        assert col in row_a


def test_gradients_are_identical_to_the_pure_glut_loss_at_lambda_zero():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    loss, _ = ic.train_step(arm, fit, cfg, step=0)
    loss.backward()
    g_arm = [p.grad.clone() for p in arm.parameters()]
    arm.zero_grad(set_to_none=True)
    y, aux = arm(fit.z, fit.x, return_aux=True)
    ic.glut_loss_terms(y, fit.target, aux, cfg)["L_glut"].backward()
    for a, b in zip(g_arm, (p.grad for p in arm.parameters())):
        assert torch.allclose(a, b, atol=0, rtol=0)


def test_interp_term_enters_the_total_and_the_row():
    cfg = tiny_cfg(interp_weight=2.0)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    pair = synth_pair(cfg, fit, alpha=0.5)
    loss, row = ic.train_step(arm, fit, cfg, step=0, pair=pair)
    y, aux = arm(fit.z, fit.x, return_aux=True)
    glut = float(ic.glut_loss_terms(y, fit.target, aux, cfg)["L_glut"].detach())
    assert float(loss.detach()) == pytest.approx(
        glut + 2.0 * row["L_interp"], rel=1e-6)
    assert row["lambda_interp"] == 2.0 and row["n_pairs"] == pair.n_pairs
    assert row["alpha_mean"] == pytest.approx(0.5)
    assert row["interp_where"] == "post_pi"
    with pytest.raises(ValueError, match="no PairBatch"):
        ic.train_step(arm, fit, cfg, step=0, pair=None)


def test_interp_target_at_alpha_zero_is_the_endpoint_lut():
    cfg = tiny_cfg(interp_weight=1.0)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    pair = synth_pair(cfg, fit, alpha=0.0)
    _, row = ic.train_step(arm, fit, cfg, step=0, pair=pair)
    with torch.no_grad():
        y_a = arm(pair.z_a, pair.x)
    want = float((y_a - pair.values_a).abs().mean().detach())
    assert row["L_interp"] == pytest.approx(want, rel=1e-6)


def test_interp_term_reaches_pi_and_the_generator():
    cfg = tiny_cfg(interp_weight=1.0)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    # zero the fit stream's contribution by making the target the arm's own output
    with torch.no_grad():
        fit.target = arm(fit.z, fit.x)
    loss, row = ic.train_step(arm, fit, cfg, step=0,
                              pair=synth_pair(cfg, fit, alpha=0.5))
    loss.backward()
    assert arm.pi.proj.weight.grad.abs().sum() > 0
    assert arm.generator.head_color[-1].weight.grad.abs().sum() > 0


def test_loss_level_4_consumes_the_image_pair_when_one_is_handed_in():
    """``L_img`` composes with the frozen headline formation and GT alpha."""
    cfg = tiny_cfg(loss_level=4, lambda_img=1.0)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    fit.image = torch.rand(cfg.batch_samples, 3, 8, 8)
    fit.alpha = 1.0
    fit.i_star = (fit.image * 1.1).clamp(0, 1)
    loss, row = ic.train_step(arm, fit, cfg, step=0)
    assert "L_img" in row and "L_img" in cfg.step_columns()
    assert torch.isfinite(loss)


def test_interp_hc_and_mse_and_ema_variants_run():
    torch.manual_seed(0)
    cfg = tiny_cfg(interp_weight=1.0, interp_hc=True, interp_dist="mse")
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    loss, row = ic.train_step(arm, fit, cfg, step=0, pair=synth_pair(cfg, fit))
    assert row["L_interp"] > 0 and torch.isfinite(loss)

    cfg_t = tiny_cfg(interp_weight=1.0, interp_target="ema_teacher",
                     interp_dist="mse")
    arm_t = ic.InterpcArm(cfg_t)
    teacher = ic.EmaTeacher(arm_t, cfg_t.interp_ema_decay)
    loss_t, row_t = ic.train_step(arm_t, fit, cfg_t, step=0,
                                  pair=synth_pair(cfg_t, fit), teacher=teacher)
    assert row_t["interp_target"] == "ema_teacher" and torch.isfinite(loss_t)
    with pytest.raises(ValueError, match="EmaTeacher"):
        ic.train_step(arm_t, fit, cfg_t, step=0, pair=synth_pair(cfg_t, fit))


def test_ema_teacher_moves_by_one_minus_decay():
    torch.manual_seed(0)
    arm = ic.InterpcArm(tiny_cfg())
    teacher = ic.EmaTeacher(arm, 0.9)
    before = teacher.model.pi.proj.weight.detach().clone()
    with torch.no_grad():
        arm.pi.proj.weight.add_(1.0)
    teacher.update(arm)
    after = teacher.model.pi.proj.weight
    assert torch.allclose(after, before + 0.1, atol=1e-6)


# --------------------------------------------------------------------------- #
# 8. alternate rows
# --------------------------------------------------------------------------- #
def test_acai_alpha_is_clipped_to_the_lower_half():
    a = torch.tensor([[0.0], [0.25], [0.5], [0.75], [1.0]])
    got = ic._acai_alpha(a).squeeze(-1)
    assert torch.allclose(got, torch.tensor([0.0, 0.25, 0.5, 0.25, 0.0]))


def test_acai_row_runs_and_logs_its_column():
    cfg = tiny_cfg(alternate="acai")
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    critic = ic.AcaiCritic(grid_n=4, hidden=16)
    fit = synth_batch(cfg)
    p = cfg.pairs_per_step
    anchor = critic.anchor
    pair = ic.PairBatch(z_a=fit.z[:p], z_b=fit.z.flip(0)[:p],
                        x=anchor.unsqueeze(0).expand(p, -1, 3),
                        values_a=(anchor * 1.1).clamp(0, 1).expand(p, -1, 3),
                        values_b=(anchor * 0.9).expand(p, -1, 3),
                        alpha=torch.rand(p, 1))
    loss, row = ic.train_step(arm, fit, cfg, step=0, pair=pair, critic=critic)
    assert "L_acai" in row and torch.isfinite(loss)
    closs, crow = ic.acai_critic_loss(arm, pair, cfg, critic)
    assert torch.isfinite(closs) and "L_acai_critic" in crow


def test_jacobian_row_runs_and_keeps_an_ema_anchor():
    cfg = tiny_cfg(alternate="jacobian")
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    fit = synth_batch(cfg)
    state: dict[str, float] = {}
    loss, row = ic.train_step(arm, fit, cfg, step=0, jac_state=state)
    assert "L_jac" in row and "anchor" in state and torch.isfinite(loss)
    loss.backward()                      # create_graph=True path is differentiable
    assert arm.pi.proj.weight.grad is not None
    with pytest.raises(ValueError, match="jac_state"):
        ic.train_step(arm, fit, cfg, step=0)


# --------------------------------------------------------------------------- #
# 9. guards -- the three where-side failures
# --------------------------------------------------------------------------- #
def test_step_row_assertion_is_three_tiered_with_distinct_failures(tmp_path):
    cfg = tiny_cfg(interp_weight=1.0)
    good = {c: 0.0 for c in cfg.step_columns()}

    row, source = ic.assert_interp_step_columns(cfg, steps_row=good)
    assert source == "caller"

    p = tmp_path / "steps.jsonl"
    p.write_text(json.dumps(good) + "\n")
    _, source = ic.assert_interp_step_columns(cfg, steps_path=p)
    assert source == "disk"

    from q3vl.whatb.guards import record_step_witness

    record_step_witness(good)
    _, source = ic.assert_interp_step_columns(cfg, steps_path=tmp_path / "nope.jsonl")
    assert source == "witness"

    clear_step_witness()
    with pytest.raises(StepsRowUnavailable):          # "nobody handed me a row"
        ic.assert_interp_step_columns(cfg, steps_path=tmp_path / "nope.jsonl")
    missing = {k: v for k, v in good.items() if k != "L_interp"}
    with pytest.raises(LossColumnsMissing):           # "the loss did not run"
        ic.assert_interp_step_columns(cfg, steps_row=missing)
    assert not issubclass(StepsRowUnavailable, LossColumnsMissing)
    assert not issubclass(LossColumnsMissing, StepsRowUnavailable)


def test_baseline_row_carrying_L_interp_is_rejected():
    cfg = tiny_cfg()                                   # lambda_int = 0
    row = {c: 0.0 for c in cfg.step_columns()}
    ic.assert_interp_step_columns(cfg, steps_row=row)
    row["L_interp"] = 0.4
    with pytest.raises(ic.BaselineRowNotClean):
        ic.assert_interp_step_columns(cfg, steps_row=row)


def test_degeneracy_guard_catches_the_three_degenerate_solutions():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    z = torch.randn(4, 2560)

    # (a) healthy: a randomly initialised full generator is not degenerate
    report = ic.quick_eval_guard(arm, z, cfg, exit_process=False)
    assert report["failures"] == []
    assert ic.degeneracy_check_ran()["where"] == "quick_eval"

    # (b) the identity: an affine-only head with zero-initialised outputs
    ident = ic.InterpcArm(tiny_cfg(gen_mode="affine_only"))
    with pytest.raises(SystemExit) as exc:
        ic.quick_eval_guard(ident, z, tiny_cfg(gen_mode="affine_only"))
    assert exc.value.code == 2

    # (c) one transform for every sample: identical conditions
    from q3vl.whatb.degeneracy import DegenerateTransform

    same = z[:1].expand(4, 2560).contiguous()
    with pytest.raises(DegenerateTransform) as err:
        ic.quick_eval_guard(arm, same, cfg, exit_process=False)
    assert any("every sample" in f for f in err.value.report.failures)


def test_degeneracy_guard_needs_more_than_one_condition(arm):
    with pytest.raises(ValueError, match="two distinct conditions"):
        ic.quick_eval_guard(arm, torch.randn(1, 2560), tiny_cfg())


def test_publication_refuses_when_the_degeneracy_guard_never_ran():
    cfg = tiny_cfg(interp_weight=1.0)
    board = {"interp": {"interp_where": "post_pi"}, "contexts": {}}
    with pytest.raises(ic.DegeneracyCheckNotRun):
        ic.assert_publishable_interpc(board, cfg, eval_only=True)


def test_interp_path_source_must_match_between_trainer_and_board():
    cfg = tiny_cfg(interp_weight=1.0)
    assert ic.assert_interp_path_source({"interp_where": "post_pi"}, cfg) == "post_pi"
    with pytest.raises(ic.InterpPathSourceMismatch):
        ic.assert_interp_path_source({"interp_where": "pre_pi"}, cfg)
    with pytest.raises(ic.InterpPathSourceMismatch, match="records no interp_where"):
        ic.assert_interp_path_source({}, cfg)


def test_context_cache_assertion():
    ok = {t: {"checkpoint": "ckpt-4976", "readout_kind": "seg_color",
              "context_source": "generated", "n": 10}
          for t in ("none", "shuffle", "irrelevant", "const")}
    assert set(assertion := ic.assert_context_caches(ok, checkpoint="ckpt-4976")) == {
        "none", "shuffle", "irrelevant", "const"}
    assert assertion["none"]["readout_kind"] == "seg_color"
    with pytest.raises(ic.ContextCacheMismatch, match="missing"):
        ic.assert_context_caches({k: v for k, v in ok.items() if k != "const"},
                                 checkpoint="ckpt-4976")
    bad_ck = {**ok, "shuffle": {**ok["shuffle"], "checkpoint": "other"}}
    with pytest.raises(ic.ContextCacheMismatch, match="checkpoint"):
        ic.assert_context_caches(bad_ck, checkpoint="ckpt-4976")
    teacher_forced = {**ok, "shuffle": {**ok["shuffle"], "context_source": "teacher"}}
    with pytest.raises(ic.ContextCacheMismatch, match="regenerated"):
        ic.assert_context_caches(teacher_forced, checkpoint="ckpt-4976")


# --------------------------------------------------------------------------- #
# 10. the interpolation criteria
# --------------------------------------------------------------------------- #
def _pair_inputs(arm, cfg):
    x = uniform_grid(cfg.eval_grid_n)
    z_a, z_b = torch.randn(1, 2560), torch.randn(1, 2560)
    va = (x * torch.tensor([1.1, 0.9, 1.0])).clamp(0, 1)
    vb = (x * torch.tensor([0.8, 1.0, 1.2])).clamp(0, 1)
    return x, z_a, z_b, va, vb


def test_ipa_has_the_four_columns_and_agrees_with_the_trivial_one_at_endpoints(arm):
    cfg = tiny_cfg()
    x, z_a, z_b, va, vb = _pair_inputs(arm, cfg)
    r = ic.ipa_pair_errors(arm, z_a, z_b, va, vb, x)
    assert r["alphas"] == list(ic.IPA_ALPHA_GRID) and len(r["arm"]) == 6
    assert r["endpoint"] == [r["arm"][0], r["arm"][-1]]
    assert r["interp_grid"] == pytest.approx(float(np.mean(r["arm"])))
    # at alpha in {0,1} the conditional path and the output mix are the same map
    assert r["output_mix"][0] == pytest.approx(r["arm"][0], rel=1e-5)
    assert r["output_mix"][-1] == pytest.approx(r["arm"][-1], rel=1e-5)


def test_ipb_reports_the_path_quantities_with_their_floors(arm):
    cfg = tiny_cfg()
    x, z_a, z_b, _, _ = _pair_inputs(arm, cfg)
    r = ic.ipb_pair_path(arm, z_a, z_b, x, k=cfg.ipb_k)
    for k in ("path_len", "chord", "rho", "sigma_bar", "jump_max", "mono_rate",
              "oob_rate", "oob_rate_postclamp", "degenerate_weight_rate"):
        assert k in r
    assert r["mono_rate_random_floor"] == 0.5
    assert r["k_steps"] == cfg.ipb_k
    assert 0.0 <= r["oob_rate"] <= 1.0
    assert "no percentile trimming" in r["note"]
    assert r["oob_rate_postclamp"] == 0.0      # clamp "two" -> the clamped path


def test_ipb_oob_rate_is_measured_before_the_clamp():
    """Section 2.4 defines ``A(alpha)`` on the pre-clamp value; a huge global
    affine therefore has to show up in ``oob_rate`` and not in the clamped path."""
    cfg = tiny_cfg()
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    with torch.no_grad():                     # push the head far out of gamut
        arm.generator.head_global[-1].bias.fill_(5.0)
        arm.generator.head_color[-1].bias.fill_(5.0)
    x = uniform_grid(cfg.eval_grid_n)
    r = ic.ipb_pair_path(arm, torch.randn(1, 2560), torch.randn(1, 2560), x, k=2)
    assert r["oob_rate"] == pytest.approx(1.0)
    assert r["oob_rate_postclamp"] == 0.0


def test_ipb_d_lib_needs_the_library_on_the_path_grid(arm):
    from q3vl.whatb.criteria import LibraryValues

    cfg = tiny_cfg()
    x = uniform_grid(cfg.eval_grid_n)
    lib = LibraryValues(("a", "b"), x,
                        torch.stack([(x * 1.1).clamp(0, 1), (x * 0.7)]))
    r = ic.ipb_pair_path(arm, torch.randn(1, 2560), torch.randn(1, 2560), x,
                         k=2, library=lib)
    assert len(r["d_lib"]) == 3 and r["d_lib_mid"] == r["d_lib"][1]
    wrong = LibraryValues(("a",), uniform_grid(3),
                          (uniform_grid(3) * 1.1).clamp(0, 1)[None])
    with pytest.raises(ValueError, match="SAME X"):
        ic.ipb_pair_path(arm, torch.randn(1, 2560), torch.randn(1, 2560), x,
                         k=2, library=wrong)


def test_interp_extra_columns_satisfy_the_P1_required_table(arm):
    cfg = tiny_cfg()
    x, z_a, z_b, va, vb = _pair_inputs(arm, cfg)
    ipa = [ic.ipa_pair_errors(arm, z_a, z_b, va, vb, x) for _ in range(2)]
    ipb = [ic.ipb_pair_path(arm, z_a, z_b, x, k=cfg.ipb_k) for _ in range(2)]
    cols = ic.interp_extra_columns(ipa, ipb, cfg)
    for key in ("interp_grid", "path_len", "mono_rate", "oob_rate"):
        assert cols[key]["n"] == 2
    assert cols["mono_rate"]["random_floor"] == 0.5
    assert cols["interp_output_mix"]["n"] == 2      # the trivial column is not optional
    assert cols["glut_external_reference"]["CGLUT-32L_Full_PSNR"][2] == 31.16
    assert "never a paired delta" in cols["glut_external_reference"]["caveat"]

    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "style", "E_arm": 1.0 + i,
             "E_B0_identity": 3.0, "E_B1_libmean": 2.0,
             "E_B2_librandom_repeats": [3.0, 3.2],
             "E_B3_bucket_retrieval_repeats": [3.1, 3.3], "E_B4_oracle": 0.5,
             "E_N1_shuffle": 1.2, "M_N1_shuffle": 2.0,
             "E_N2_irrelevant": 1.3, "M_N2_irrelevant": 2.1,
             "E_N3_const": 1.4, "M_N3_const": 2.2} for i in range(4)]
    board = build_board(rows, arm=ic.ARM, split="V_what", extra_columns=cols)
    report = assert_criteria_ran(board, ic.ARM)
    assert report["headline_normal_only_n"] == 4
    assert set(report["computed"]) >= {"interp_grid", "path_len", "mono_rate",
                                       "oob_rate"}


# --------------------------------------------------------------------------- #
# 11. banned-list hygiene (section 4.I, enforced by absence)
# --------------------------------------------------------------------------- #
def test_the_arm_defines_no_banned_metric_and_never_leaves_the_device():
    tree = ast.parse(ARM_SRC.read_text())
    names = [n.name.lower() for n in ast.walk(tree)
             if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for banned in ("auc", "roc", "minmax", "min_max", "trim", "iou", "pooled"):
        assert not any(banned in n for n in names), f"banned metric name: {banned}"
    # the ban is on *code*, not on the docstring that explains it
    leaves: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr == "cpu":
            leaves.append("x.cpu()")
        if node.func.attr == "to" and any(
                isinstance(a, ast.Constant) and a.value == "cpu" for a in node.args):
            leaves.append('x.to("cpu")')
    assert leaves == [], f"metric path leaves the device: {leaves}"


# --------------------------------------------------------------------------- #
# 12. run_setup + the runner
# --------------------------------------------------------------------------- #
def test_run_setup_block_carries_the_frozen_numbers():
    cfg = ic.InterpcConfig(interp_weight=1.0)
    torch.manual_seed(0)
    block = ic.run_setup_block(cfg, ic.InterpcArm(tiny_cfg()),
                               alpha_sampler=ic.AlphaSampler(cfg))
    frozen = block["frozen"]
    assert frozen["train_normal_n"] == 93934
    assert frozen["batch"] == "32x256" and frozen["colors_per_step"] == 8192
    assert (frozen["steps_per_epoch"], frozen["total_steps"]) == (2936, 117440)
    assert frozen["clamp"] == "two"
    assert frozen["headline_formation"] == "I_hat = (1-a) * I + a * f_hat(I)"
    assert block["degeneracy_thresholds"]["point_std"] == 1e-3
    assert "L_interp" in block["step_columns"]
    assert block["alpha_sampler"]["p_end"] == pytest.approx(1 / 3)


def test_runner_config_reproduces_the_frozen_step_budget():
    a = runner.build_parser().parse_args([])
    cfg = runner.config_from_args(a)
    assert (cfg.steps_per_epoch, cfg.total_steps) == (2936, 117440)
    assert cfg.interp_weight == 0.0 and cfg.clamp == "two"
    a2 = runner.build_parser().parse_args(
        ["--interp-weight", "1", "--interp-alpha", "uniform", "--batch-split",
         "64x128", "--n-gauss", "32"])
    cfg2 = runner.config_from_args(a2)
    assert cfg2.beta == 1.0 and cfg2.n_gauss == 32
    assert (cfg2.steps_per_epoch, cfg2.total_steps) == (1468, 58720)


def test_runner_train_stage_refuses_without_the_z_cache(tmp_path):
    with pytest.raises(SystemExit, match="zcache"):
        runner.main(["--stage", "train", "--out-root", str(tmp_path),
                     "--interp-weight", "1"])


def test_runner_self_test_stage_runs_end_to_end(tmp_path, capsys):
    rc = runner.main(["--stage", "self-test", "--out-root", str(tmp_path),
                      "--interp-weight", "1", "--run-name", "st"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] is True
    assert out["criteria"]["interp_grid"] > 0
    run_dir = tmp_path / "runs" / "st"
    rows = [json.loads(l) for l in (run_dir / "steps.jsonl").read_text().splitlines()]
    assert "L_interp" in rows[0] and rows[0]["n_colors"] == 4 * 2048
    board = json.loads((run_dir / "self_test_board.json").read_text())
    assert board["criteria_columns"]["interp_grid"]["n"] > 0
    assert board["contexts"]["all"]["headline_normal_only"]["n"] == 6


def test_eval_pair_selection_modes():
    """``all`` is EPR-026 section 3.6's n; ``one`` is EPR-024's board protocol."""
    rows = fake_rows({
        "s1": [("a", "L1", "normal"), ("b", "L2", "normal"), ("c", "L3", "normal")],
        "s2": [("d", "L1", "normal"), ("e", "L2", "normal")],
    })
    idx = ic.PairIndex.build(rows, split="V_what")
    every = runner.select_eval_pairs(idx, mode="all")
    assert len(every) == idx.n_pairs == 4
    one = runner.select_eval_pairs(idx, mode="one", seed=1)
    assert len(one) == len(idx) == 2
    assert one == runner.select_eval_pairs(idx, mode="one", seed=1)      # seeded
    assert len(runner.select_eval_pairs(idx, mode="all", limit=3)) == 3
    with pytest.raises(ValueError):
        runner.select_eval_pairs(idx, mode="two")


def test_z_cache_reader_follows_the_on_disk_contract(tmp_path):
    """The reader is the shared q3vl.whatb.zcache one, addressed by (split, tag)."""
    for tag in ("none", "shuffle", "irrelevant", "const"):
        d = tmp_path / f"V_what__{tag}"
        d.mkdir()
        np.save(d / "z.npy", np.arange(3 * 2560, dtype=np.float32).reshape(3, 2560))
        (d / "index.jsonl").write_text("\n".join(
            json.dumps({"sample_id": f"s{i}", "row": i}) for i in range(3)))
        (d / "meta.json").write_text(json.dumps(
            {"checkpoint": "ckpt", "readout_kind": "seg_color", "n": 3,
             "split": "V_what", "control_tag": tag, "dtype": "float32",
             "context_source": "generated"}))
    cfg = ic.InterpcConfig()
    cache = runner.open_z(tmp_path, "V_what", cfg, checkpoint="ckpt",
                          required=("none", "shuffle", "irrelevant", "const"))
    z = cache.z(["s2", "s0"], tag="shuffle")
    assert z.shape == (2, 2560) and z.dtype == torch.float32
    assert float(z[0, 0]) == 2 * 2560 and float(z[1, 0]) == 0.0
    assert ic.assert_context_caches(cache.meta, checkpoint="ckpt")["const"]["n"] == 3
    with pytest.raises(AssertionError, match="checkpoint"):
        runner.open_z(tmp_path, "V_what", cfg, checkpoint="other")
    with pytest.raises(FileNotFoundError):
        runner.open_z(tmp_path, "T_final", cfg, checkpoint="ckpt")


def test_runner_loss_preregistration_lists_the_sixteen_keys():
    pre = runner.loss_preregistration(ic.InterpcConfig(interp_weight=1.0))
    assert len(pre["required_criteria"]) == 16
    assert pre["lambda_int"] == 1.0
    assert "AUC in any form" in pre["banned"]
    assert pre["headline"].startswith(".contexts.all.headline_normal_only")


# --------------------------------------------------------------------------- #
# 12. the image-space board (the twelve pre-registered keys)
# --------------------------------------------------------------------------- #
def _fake_lut_grid(d: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    ax = np.linspace(0.0, 1.0, d, dtype=np.float32)
    b, g, r = np.meshgrid(ax, ax, ax, indexing="ij")
    out = np.stack((r ** 0.8, 0.4 * g + 0.3 * r, 1.0 - b ** 1.2), axis=-1)
    out = np.clip(out + 0.03 * rng.standard_normal(out.shape), 0.0, 1.0)
    return out.astype(np.float32)


@pytest.fixture()
def lut_bank(tmp_path):
    from q3vl.whatb.lutdata import LutBank

    grids = {f"lut_{k}": _fake_lut_grid(9, i + 1)
             for i, k in enumerate("abcd")}
    np.savez(tmp_path / "luts.npz", **grids)
    (tmp_path / "luts_meta.json").write_text(json.dumps(
        {k: {"path": str(tmp_path / f"{k}.cube"), "dmin": [0.0] * 3,
             "dmax": [1.0] * 3} for k in grids}))
    return LutBank(tmp_path)


class _FakeStore:
    """``SampleStore`` seam: an image and a GT field per sample, no NFS."""

    def __init__(self, hw=(6, 8)):
        self.hw = hw

    def load(self, row, device="cpu", dtype=torch.float32):
        g = torch.Generator().manual_seed(abs(hash(row.sample_id)) % (2 ** 31))
        img = torch.rand(3, *self.hw, generator=g, dtype=dtype)
        if row.task_type == "style":
            return img, 1.0
        return img, torch.rand(1, *self.hw, generator=g, dtype=dtype)


class _FakeCacheDir:
    """``ZCacheDir`` seam: a distinct z per (tag, sample_id)."""

    TAGS = ("none", "shuffle", "irrelevant", "const")

    def __contains__(self, tag):
        return tag in self.TAGS

    def z(self, sample_ids, *, tag="none", device="cpu", dtype=torch.float32):
        out = []
        for sid in sample_ids:
            g = torch.Generator().manual_seed(
                abs(hash(f"{tag}/{sid}")) % (2 ** 31))
            out.append(torch.randn(2560, generator=g, dtype=dtype))
        return torch.stack(out).to(device=device)


def _index_rows(n: int = 6):
    from q3vl.whatb.splits import IndexRow

    return [IndexRow.from_json({
        "sample_id": f"s{i}", "split": "V_what", "lut_id": f"lut_{'abcd'[i % 4]}",
        "source_image_id": f"src{i // 2}", "task_type": "style" if i % 2 else "local",
        "winner_confidence": "normal"}) for i in range(n)]


def test_library_mean_volume_matches_the_pointwise_library_mean(lut_bank):
    """A transposed axis here is a plausible and completely wrong B1 column."""
    from q3vl.whatb.lutdata import apply_lut_volume

    ids = ["lut_a", "lut_b", "lut_c"]
    x = uniform_grid(9)
    want = torch.stack([lut_bank.apply(x, i) for i in ids]).mean(dim=0)
    vol = runner.library_mean_volume(lut_bank, ids, grid_n=9, device="cpu")
    assert vol.shape == (1, 3, 9, 9, 9)
    got = apply_lut_volume(vol, x)
    assert torch.allclose(got, want, atol=2e-6)


def test_image_board_rows_produce_every_preregistered_key(lut_bank):
    """The wiring the eval stage was missing: rows -> the sixteen required keys."""
    from q3vl.whatb.criteria import LibraryValues

    cfg = tiny_cfg(interp_weight=1.0)
    torch.manual_seed(0)
    arm = ic.InterpcArm(cfg)
    rows = _index_rows(6)
    lib_ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    grid = uniform_grid(cfg.eval_grid_n)
    lib9 = LibraryValues.build(lut_bank, lib_ids, uniform_grid(5))
    vol = runner.library_mean_volume(lut_bank, lib_ids, grid_n=5, device="cpu")
    cache = _FakeCacheDir()

    out = runner.image_board_rows(
        arm, rows, cache=cache, bank=lut_bank, store=_FakeStore(),
        records={r.sample_id: {"minor": "bucket_0"} for r in rows},
        lib=lib9, lib_mean_vol=vol, pools={"bucket_0": lib_ids}, grid=grid,
        device="cpu", repeats=2, seed=0,
        base={r.sample_id: {"grid_error": 1.0, "unseen_color_error": 2.0}
              for r in rows})

    for key in ("E_arm", "E_B0_identity", "E_B1_libmean",
                "E_B2_librandom_repeats", "E_B3_bucket_retrieval_repeats",
                "E_B4_oracle", "E_N1_shuffle", "M_N1_shuffle", "E_N2_irrelevant",
                "M_N2_irrelevant", "E_N3_const", "M_N3_const"):
        assert key in out[0], key
    # the function-space columns of the same sample travel with the row
    assert out[0]["grid_error"] == 1.0 and out[0]["unseen_color_error"] == 2.0
    # the three controls really are three different conditions
    assert len({round(out[0][k], 9) for k in
                ("E_arm", "E_N1_shuffle", "E_N2_irrelevant", "E_N3_const")}) == 4
    # locality is a GT-alpha column: the style rows (alpha == 1.0) have none
    assert "loc_in" in out[0] and "loc_in" not in out[1]

    x_eval = uniform_grid(cfg.eval_grid_n)
    z = cache.z([r.sample_id for r in rows])
    ipa = [ic.ipa_pair_errors(arm, z[i: i + 1], z[i + 1: i + 2],
                              lut_bank.apply(x_eval, rows[i].lut_id),
                              lut_bank.apply(x_eval, rows[i + 1].lut_id), x_eval)
           for i in range(2)]
    ipb = [ic.ipb_pair_path(arm, z[i: i + 1], z[i + 1: i + 2], x_eval, k=cfg.ipb_k)
           for i in range(2)]
    board = build_board(out, arm=ic.ARM, split="V_what",
                        extra_columns=ic.interp_extra_columns(ipa, ipb, cfg))
    report = assert_criteria_ran(board, ic.ARM, axes=ic.AXES)
    assert all(n > 0 for n in report["computed"].values())
    assert board["contexts"]["all"]["headline_normal_only"]["n"] == len(rows)


def test_function_space_rows_alone_cannot_publish(lut_bank):
    """The regression this wiring closes: the old eval board had two columns."""
    from q3vl.whatb.criteria import CriterionNotComputed

    rows = [{"sample_id": f"s{i}", "winner_confidence": "normal",
             "task_type": "local", "grid_error": 1.0 + i,
             "unseen_color_error": 2.0 + i} for i in range(4)]
    board = build_board(rows, arm=ic.ARM, split="V_what")
    with pytest.raises(CriterionNotComputed, match="headline_normal_only"):
        assert_criteria_ran(board, ic.ARM, axes=ic.AXES)


def test_eval_stage_exposes_the_headline_board_flags():
    a = runner.build_parser().parse_args([])
    assert a.headline_samples == 0          # 0 = every normal-only row
    assert a.lib_size == 1137 and a.libmean_grid == 33
    assert a.select_metric == "de76" and a.repeats == 8
    ids = runner.select_library_ids([f"l{i}" for i in range(50)], 10, seed=1)
    assert len(ids) == 10 == len(set(ids))
    assert ids == runner.select_library_ids([f"l{i}" for i in range(50)], 10, seed=1)
    assert runner.select_library_ids(["a", "b"], 99) == ["a", "b"]


# --------------------------------------------------------------------------- #
# the degeneracy guard's BINDING window (family-wide 口径, EPR-028's rule)
# --------------------------------------------------------------------------- #
def test_degeneracy_binding_truth_table():
    """Waived only when the run is BOTH --smoke AND shorter than one epoch."""
    import argparse

    assert runner.DEGENERACY_BINDING_MIN_STEPS == 2936
    smoke = argparse.Namespace(smoke=True)
    plain = argparse.Namespace(smoke=False)
    noflag = argparse.Namespace()               # this runner has no --smoke flag

    assert runner.degeneracy_binding(smoke, 10)[0] is False
    assert runner.degeneracy_binding(smoke, 2936)[0] is True
    assert runner.degeneracy_binding(plain, 10)[0] is True
    assert runner.degeneracy_binding(plain, 117440)[0] is True
    # no --smoke flag on this parser -> the waiver is unreachable, never a crash
    assert runner.degeneracy_binding(noflag, 400)[0] is True
    assert "--smoke" not in {a.dest for a in runner.build_parser()._actions}
    for args, total in ((smoke, 10), (plain, 400), (noflag, 117440)):
        assert runner.degeneracy_binding(args, total)[1]      # always a reason


def test_degeneracy_binding_matches_the_g4d_constant():
    """One number for the whole family: EPR-026 may not drift from EPR-028."""
    from q3vl.whatb.scripts import run_g4d_arm as g4d_runner

    assert (runner.DEGENERACY_BINDING_MIN_STEPS
            == g4d_runner.DEGENERACY_BINDING_MIN_STEPS == 2936)
    for total, smoke in ((10, True), (10, False), (400, False), (2936, True),
                         (117440, False)):
        import argparse
        ns = argparse.Namespace(smoke=smoke)
        assert (runner.degeneracy_binding(ns, total)[0]
                == g4d_runner.degeneracy_binding(ns, total)[0])


def test_the_guard_fires_at_the_first_quick_eval_not_at_step_49():
    """``min(50, total)`` pinned the check to the bottom of the start-up trough.

    Measured on CPU with the real z cache and the real bank: cross_std 4.97e-5
    at step 49 against a 1e-4 floor, 2.27e-3 at step 399.  Any run with
    ``total >= 50`` -- the 117,440-step one included -- was checked at step 49.
    """
    src = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "stage_train")
    calls = [n for n in ast.walk(fn)
             if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "quick_eval_guard"]
    assert len(calls) == 1
    # the one call site passes exit_process=<binding>, so a waived verdict still
    # measures / prints / records the witness instead of being skipped
    kw = {k.arg for k in calls[0].keywords}
    assert {"where", "exit_process"} <= kw
    guard_ifs = [n for n in ast.walk(fn)
                 if isinstance(n, ast.If)
                 and any(isinstance(c, ast.Call)
                         and getattr(c.func, "id", "") == "quick_eval_guard"
                         for c in ast.walk(n))]
    # the innermost `if` that owns the call is the trigger
    cond = ast.dump(min(guard_ifs, key=lambda n: len(list(ast.walk(n)))).test)
    assert "guard_done" in cond and "due" in cond
    assert "50" not in cond
