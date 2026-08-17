"""EPR-025 AFFONLY: the arm's own tests.  CPU only, no GPU process is started.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests/test_affonly.py -q

Groups, in the order the proposal states them:

1. the frozen numbers and the parameter-count arithmetic of section 2.3
   (336,236 = 169,024 + 41,344 + 107,328 + 18,060 + 480);
2. step 0 is the identity (section 3.4-(4)) for all three ``--global-affine`` rows;
3. **proposition 1** -- the numerical assertion 2 of section 3.6, with two
   negative controls (Full Generation, and GLUT's own Shared Geometry where ``o``
   is still generated, which is exactly the configuration the proposal says does
   NOT satisfy the premise);
4. assertion 1 (bit-identical geometry) and assertion 3 (IP-A's ``f^par`` column
   equals the trivial output-blend column, pre-clamp);
5. the loss: GLUT Eq.6-8 on small examples with hand-computed values;
6. the optimiser groups and the 4-tensor / 10N start-up assertion;
7. the mining rule (ruling 11.1-4) and the first ``steps.jsonl`` row;
8. the guards: the degeneracy check fires on a constant / identity / cross-sample
   -flat transform and the publication gate refuses a board missing a column;
9. the runner's argument surface.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import numpy as np

import pytest
import torch

from q3vl.whatb.arms import affonly as A
from q3vl.whatb.colorimetry import srgb_to_lab
from q3vl.whatb.criteria import CriterionNotComputed, PREREGISTERED_KEYS
from q3vl.whatb.glut import EPS
from q3vl.whatb.guards import (
    DegenerateTransform,
    LossColumnsMissing,
    StepsRowUnavailable,
    clear_step_witness,
)
from q3vl.whatb.publish import step_columns_for
from q3vl.whatb.queries import QuerySampler, uniform_grid

torch.manual_seed(20260810)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_head(**kw) -> A.AffineOnlyHead:
    cfg = A.AffineOnlyConfig(**kw)
    return A.AffineOnlyHead(cfg)


def perturb(head: A.AffineOnlyHead, *, scale: float = 0.05, seed: int = 7) -> A.AffineOnlyHead:
    """Give a zero-initialised head the parameters of a *trained* one.

    Without this every head is the identity map and every negative control is
    vacuous: a zero last layer emits the same output for every condition, so
    "the geometry does not depend on the condition" would pass even in Full
    Generation.
    """
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(torch.randn(p.shape, generator=g) * scale)
    return head


def some_z(n: int = 8, *, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, 2560, generator=g)


class FakeBank:
    """A LUT bank stand-in: two analytic, invertible-ish colour maps."""

    def __init__(self) -> None:
        self.calls = 0

    def apply(self, x: torch.Tensor, lut_id: str) -> torch.Tensor:
        self.calls += 1
        if lut_id == "warm":
            return (x * torch.tensor([1.1, 1.0, 0.85])).clamp(0, 1)
        if lut_id == "cool":
            return (x * torch.tensor([0.8, 1.0, 1.15])).clamp(0, 1)
        return x.clamp(0, 1)


# --------------------------------------------------------------------------- #
# 1. frozen numbers and the parameter arithmetic
# --------------------------------------------------------------------------- #
def test_frozen_block_numbers() -> None:
    f = A.FROZEN
    assert f["train_n"] == 93934
    assert (f["batch_samples"], f["queries_per_sample"]) == (32, 256)
    assert f["colors_per_step"] == 8192
    assert f["steps_per_epoch"] == 2936 == math.ceil(93934 / 32)
    assert f["total_steps"] == 117440 == 2936 * 40
    assert f["clamp_default"] == "two"
    assert f["headline_formation"] == "I_hat = (1 - a) * I + a * f_hat(I)"
    assert tuple(f["preregistered_keys"]) == PREREGISTERED_KEYS
    assert len(PREREGISTERED_KEYS) == 12


def test_config_defaults_are_the_proposal_defaults() -> None:
    c = A.AffineOnlyConfig()
    assert (c.share, c.global_affine) == ("geo_opacity", "affine")
    assert (c.n_gauss, c.cond_dim, c.gen_width) == (48, 64, 128)
    assert (c.clamp, c.zero_init_heads, c.mu_init) == ("two", True, "grid")
    assert (c.lambda_hc, c.lambda_sparse) == (10.0, 0.001)
    assert (c.base_lr, c.shared_lr_scale, c.weight_decay, c.grad_clip) == (1e-3, 0.1, 0.0, 0.0)
    assert (c.sigma_init, c.opacity_logit_init) == (0.15, 4.0)
    assert (c.mining_start_epoch, c.mining_end_epoch) == (5, 20)
    assert (c.mining_r_start, c.mining_r_end) == (0.10, 0.40)
    assert c.seed == 20260810
    # section 1.4 table: 12N + 12 = 588 generated, 10N = 480 shared (N = 48)
    assert c.theta_gen_dim == 588
    assert c.n_shared == 480


def test_generated_dimension_per_share_row() -> None:
    """Section 1.4's D4 table: 1068 / 636 / 588 at N = 48; 716 / 428 / 396 at N = 32."""
    assert A.AffineOnlyConfig(share="none").theta_gen_dim == 22 * 48 + 12 == 1068
    assert A.AffineOnlyConfig(share="geo").theta_gen_dim == 13 * 48 + 12 == 636
    assert A.AffineOnlyConfig(share="geo_opacity").theta_gen_dim == 12 * 48 + 12 == 588
    assert A.AffineOnlyConfig(share="none", n_gauss=32).theta_gen_dim == 716
    assert A.AffineOnlyConfig(share="geo", n_gauss=32).theta_gen_dim == 428
    assert A.AffineOnlyConfig(share="geo_opacity", n_gauss=32).theta_gen_dim == 396


def test_parameter_counts_match_section_2_3() -> None:
    head = make_head()
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(head.proj) == 169_024                    # LayerNorm 5,120 + Linear 163,904
    assert n(head.generator.encoder) == 41_344        # 8,320 + 16,512 + 16,512
    assert n(head.generator.head_color) == 107_328    # 16,512 + 16,512 + 74,304
    assert n(head.generator.head_global) == 18_060    # 16,512 + 1,548
    assert n(head.generator.shared_geometry) == 480   # 10N, N = 48
    assert sum(p.numel() for p in head.parameters()) == 336_236


def test_shared_tables_are_initialised_per_app_a1() -> None:
    sg = make_head().shared_geometry
    assert sg is not None
    assert tuple(sg.mu.shape) == (48, 3)
    # 4x4x3 cell centres: three distinct blue levels at (2k+1)/6
    blue = sorted({round(float(v), 5) for v in sg.mu.detach()[:, 2]})
    assert blue == pytest.approx([1 / 6, 0.5, 5 / 6], abs=1e-5)
    # softplus^-1(0.15) = -1.821182660604379 (fp32 storage -> ~1e-7 of resolution)
    assert float(sg.chol_diag.detach()[0, 0]) == pytest.approx(-1.8211826606, abs=1e-6)
    assert float(torch.nn.functional.softplus(sg.chol_diag.detach()[0, 0])) == pytest.approx(
        0.15, abs=1e-7)
    assert torch.equal(sg.chol_off.detach(), torch.zeros(48, 3))
    assert float(sg.opacity_logit.detach()[0]) == 4.0
    assert float(torch.sigmoid(sg.opacity_logit.detach()[0])) == pytest.approx(0.98201, abs=1e-5)


# --------------------------------------------------------------------------- #
# 2. step 0 is the identity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("global_affine", ["affine", "residual", "none"])
def test_step0_is_the_identity(global_affine: str) -> None:
    """Section 3.4-(4): zero last layer + ``M = I + dM`` -> ``f(x) = (1 - delta(x)) x``."""
    head = make_head(global_affine=global_affine)
    x = uniform_grid(9)
    with torch.no_grad():
        y = head.transform(some_z(4), x)
    dev = float((y - x[None]).abs().max())
    assert dev < 1e-5, f"step0 deviation from identity is {dev:.3e}"


def test_global_affine_residual_pins_g_to_identity() -> None:
    head = perturb(make_head(global_affine="residual"))
    p = head.theta(some_z(2))
    assert torch.equal(p.g_matrix, torch.eye(3).expand(2, 3, 3))
    assert torch.equal(p.g_bias, torch.zeros(2, 3))
    # row 4 also drops the identity anchor on the local branch (M_i = dM_i)
    assert head.generator.m_residual is False


def test_global_affine_none_drops_the_global_branch() -> None:
    head = make_head(global_affine="none")
    assert head.carrier.residual is False
    assert head.generator.m_residual is True


# --------------------------------------------------------------------------- #
# 3. proposition 1 (assertion 2) with its two negative controls
# --------------------------------------------------------------------------- #
def test_proposition_1_holds_for_the_arm() -> None:
    head = perturb(make_head())
    out = A.assert_affine_linearity(head, some_z(8), n_pairs=4, grid_n=9)
    assert out["passed"] and out["value"] < A.LINEARITY_TOL
    assert out["clamp"].startswith("none")
    assert set(out["per_alpha"]) == {f"alpha_{a}" for a in (0.1, 0.3, 0.5, 0.7, 0.9)}


def test_proposition_1_fails_under_full_generation() -> None:
    """Negative control: with mu/Sigma/o generated, ``w_i`` moves with theta."""
    head = perturb(make_head(share="none"))
    out = A.assert_affine_linearity(head, some_z(8), n_pairs=4, grid_n=9,
                                    raise_on_fail=False)
    assert not out["passed"]
    assert out["value"] > 1e-3
    with pytest.raises(AssertionError, match="assertion 2 FAILED"):
        A.assert_affine_linearity(head, some_z(8), n_pairs=4, grid_n=9)


def test_proposition_1_fails_under_glut_shared_geometry() -> None:
    """The proposal's own note at :282 -- sharing only ``{mu, Sigma}`` is not enough,
    because ``o`` enters both the numerator and the denominator of Eq.2."""
    head = perturb(make_head(share="geo"))
    out = A.assert_affine_linearity(head, some_z(8), n_pairs=4, grid_n=9,
                                    raise_on_fail=False)
    assert not out["passed"]


def test_linearity_assertion_needs_fp32() -> None:
    """Assertion 2 runs in fp32 by construction: bf16 eps ~ 7.8e-3 >> 1e-5."""
    out = A.assert_affine_linearity(perturb(make_head()), some_z(4), n_pairs=2, grid_n=5)
    assert out["dtype"] == "float32"
    assert torch.finfo(torch.bfloat16).eps > A.LINEARITY_TOL


# --------------------------------------------------------------------------- #
# 4. assertion 1 and assertion 3
# --------------------------------------------------------------------------- #
def test_assertion_1_geometry_is_bit_identical() -> None:
    head = perturb(make_head())
    out = A.assert_shared_geometry_identical(head, some_z(8), n_probe=8)
    assert out["identical"] and out["value"] == 1.0 and out["n"] == 8


@pytest.mark.parametrize("share", ["none", "geo"])
def test_assertion_1_fails_when_the_geometry_is_conditional(share: str) -> None:
    head = perturb(make_head(share=share))
    with pytest.raises(AssertionError, match="assertion 1 FAILED"):
        A.assert_shared_geometry_identical(head, some_z(8), n_probe=8)
    out = A.assert_shared_geometry_identical(head, some_z(8), n_probe=8, raise_on_fail=False)
    assert not out["identical"] and out["n_conditions_differing"] > 0


def test_assertion_3_ip_a_par_column_equals_the_output_blend() -> None:
    head = perturb(make_head())
    z = some_z(4)
    pairs = [(z[0], "warm", z[1], "cool"), (z[2], "cool", z[3], "warm")]
    out = A.interp_ip_a(head, pairs, FakeBank(), grid_n=9)
    assert out["assertion3_passed"], out["assertion3_maxdev"]
    assert out["n"] == 2
    assert set(out["per_alpha"]) == {"f_cond", "f_par", "out_mix"}
    assert len(out["per_alpha"]["f_cond"]) == 6            # the six App B.3 alphas
    assert out["value"] is not None


def test_assertion_3_fails_under_full_generation() -> None:
    head = perturb(make_head(share="none"))
    z = some_z(2)
    out = A.interp_ip_a(head, [(z[0], "warm", z[1], "cool")], FakeBank(), grid_n=9)
    assert not out["assertion3_passed"]


def test_ip_b_reports_every_path_quantity_with_its_floor() -> None:
    head = perturb(make_head())
    z = some_z(4)
    cols = A.interp_ip_b(head, [(z[0], z[1]), (z[2], z[3])], k_steps=4, grid_n=5)
    for key in ("path_len", "chord", "rho", "sigma_bar", "jump_max", "mono_rate", "oob_rate"):
        assert cols[key]["n"] == 2
    assert cols["mono_rate"]["random_floor"] == 0.5
    assert "no percentile trimming" in cols["jump_max"]["note"]


@pytest.mark.parametrize("global_affine", ["affine", "residual", "none"])
def test_step0_witness_is_recorded_and_tiny(global_affine: str) -> None:
    """Ruling 11.1-1's witness column, on the 17^3 grid, before any step."""
    out = A.step0_witness(make_head(global_affine=global_affine), some_z(4), grid_n=17)
    assert out["grid_n"] == 17 and out["n_conditions"] == 4
    assert out["zero_init_heads"] is True
    assert out["step0_maxabs_f_minus_id"] < 1e-5


def test_degenerate_weight_rate_is_zero_on_the_grid_init() -> None:
    """Proposition 2's column: with App A.1 grid means nothing is uncovered."""
    out = A.degenerate_weight_rate(make_head(), some_z(4), uniform_grid(9))
    assert out["tau"] == 1e-3
    assert out["value"] == 0.0
    assert out["min_influence_sum"] > 1e-3


def test_degenerate_weight_rate_is_non_zero_when_gaussians_are_crowded() -> None:
    head = make_head()
    with torch.no_grad():                       # push every mean into the black corner
        head.shared_geometry.mu.mul_(0.0).add_(0.02)   # sigma stays at App A.1's 0.15
    out = A.degenerate_weight_rate(head, some_z(2), uniform_grid(9))
    assert out["value"] > 0.5


# --------------------------------------------------------------------------- #
# 5. the loss (GLUT Eq.6-8)
# --------------------------------------------------------------------------- #
def test_loss_rec_is_the_mean_absolute_error() -> None:
    y = torch.rand(2, 16, 3)
    y_hat = y + 0.25
    o = torch.full((48,), 0.5)
    t = A.loss_terms(y_hat, y, o)
    assert float(t.l_rec) == pytest.approx(0.25, abs=1e-6)
    assert t.n_colors == 32


def test_loss_hc_is_zero_when_the_hue_matches_and_is_positive_otherwise() -> None:
    y = torch.rand(1, 32, 3) * 0.8 + 0.1
    o = torch.full((48,), 0.5)
    assert float(A.loss_terms(y, y, o).l_hc) == pytest.approx(0.0, abs=1e-5)
    y2 = y.flip(-1)
    assert float(A.loss_terms(y2, y, o).l_hc) > 0.0


def test_loss_hc_masks_the_achromatic_points_and_counts_them() -> None:
    """Frozen block: ``h = (a,b)/max(C, 1e-3)`` AND the whole term x ``1[C >= 1e-3]``."""
    gray = torch.linspace(0.1, 0.9, 8).reshape(1, 8, 1).repeat(1, 1, 3)   # a = b = 0
    lab = srgb_to_lab(gray)
    assert float(lab[..., 1:].abs().max()) < 1e-3
    o = torch.full((48,), 0.5)
    t = A.loss_terms(gray.flip(-1) * 0 + 0.5, gray, o)
    assert t.n_hc_masked == 8
    assert float(t.l_hc) == 0.0
    # and without the mask the same points would contribute (that is EPR-024's row)
    t2 = A.loss_terms(gray * 0 + 0.5, gray, o, hc_mask=False)
    assert t2.n_hc_masked == 8


def test_r_sparse_is_the_binary_entropy_of_the_opacities() -> None:
    o_half = torch.full((48,), 0.5)
    t = A.loss_terms(torch.rand(1, 4, 3), torch.rand(1, 4, 3), o_half)
    assert float(t.l_sparse) == pytest.approx(math.log(2.0), abs=1e-5)
    o_sat = torch.full((48,), 1.0 - 1e-9)
    t2 = A.loss_terms(torch.rand(1, 4, 3), torch.rand(1, 4, 3), o_sat)
    assert float(t2.l_sparse) == pytest.approx(-math.log(1.0 + EPS) * 1.0, abs=1e-3)


def test_total_uses_the_paper_weights() -> None:
    y = torch.rand(1, 16, 3)
    y_hat = torch.rand(1, 16, 3)
    o = torch.sigmoid(torch.randn(48))
    t = A.loss_terms(y_hat, y, o, lambda_hc=10.0, lambda_sparse=0.001)
    assert float(t.total) == pytest.approx(
        float(t.l_rec) + 10.0 * float(t.l_hc) + 0.001 * float(t.l_sparse), rel=1e-6)
    row = t.as_row()
    assert {"L_rec", "L_hc", "L_sparse", "n_hc_masked", "n_colors"} <= set(row)


# --------------------------------------------------------------------------- #
# 6. optimiser groups
# --------------------------------------------------------------------------- #
def test_param_groups_put_the_shared_tables_on_a_tenth_of_the_lr() -> None:
    head = make_head()
    groups = head.param_groups()
    assert [g["name"] for g in groups] == ["generator", "shared_geometry"]
    assert groups[0]["lr"] == pytest.approx(1e-3)
    assert groups[1]["lr"] == pytest.approx(1e-4)
    # section 3.4-(8): exactly 4 tensors and 10N elements
    assert len(groups[1]["params"]) == 4
    assert sum(p.numel() for p in groups[1]["params"]) == 480
    assert sum(p.numel() for g in groups for p in g["params"]) == 336_236


def test_param_group_assertion_fires_when_the_shared_set_is_wrong(monkeypatch) -> None:
    head = make_head()
    sg = head.shared_geometry
    monkeypatch.setattr(sg, "parameters_list", lambda: [sg.mu, sg.chol_diag])
    with pytest.raises(AssertionError, match="10N"):
        head.param_groups()


def test_full_generation_has_a_single_group() -> None:
    groups = make_head(share="none").param_groups()
    assert len(groups) == 1 and groups[0]["lr"] == pytest.approx(1e-3)


def test_optimizer_is_adam_and_the_scheduler_is_cosine() -> None:
    head = make_head()
    opt = A.build_optimizer(head)
    assert isinstance(opt, torch.optim.Adam)
    assert all(g["weight_decay"] == 0.0 for g in opt.param_groups)
    sched = A.build_scheduler(opt, head.cfg, total_steps=10)
    assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)
    lrs = []
    for p in head.parameters():
        p.grad = torch.zeros_like(p)          # so opt.step() precedes sched.step()
    for _ in range(10):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-3) and lrs[-1] < lrs[0]
    with pytest.raises(ValueError, match="Adam"):
        A.build_optimizer(head, A.AffineOnlyConfig(optimizer="adamw"))


# --------------------------------------------------------------------------- #
# 7. mining and the first step row
# --------------------------------------------------------------------------- #
def test_mining_keeps_the_frozen_batch_size_and_takes_the_hardest() -> None:
    head = perturb(make_head())
    sampler = QuerySampler(seed=1, q=16)
    bank = FakeBank()
    target = lambda x: torch.stack([bank.apply(x[i], "warm") for i in range(x.shape[0])])
    x, y, stats = A.build_training_colors(head, some_z(4), sampler, target, 0.25)
    assert tuple(x.shape) == (4, 16, 3) and tuple(y.shape) == (4, 16, 3)
    assert stats == {"mining_ratio": 0.25, "n_mined": 16, "n_fresh": 48}
    # r = 0 keeps the plain uniform draw
    _, _, s0 = A.build_training_colors(head, some_z(4), sampler, target, 0.0)
    assert s0["n_mined"] == 0


def test_mining_ratio_schedule_is_the_paper_ramp() -> None:
    from q3vl.whatb.queries import mining_ratio

    assert mining_ratio(0) == 0.10 and mining_ratio(5) == 0.10
    assert mining_ratio(12.5) == pytest.approx(0.25)
    assert mining_ratio(20) == 0.40 and mining_ratio(39) == 0.40


def test_train_step_publishes_every_preregistered_column() -> None:
    clear_step_witness()
    head = make_head(n_gauss=8)
    opt = A.build_optimizer(head)
    sampler = QuerySampler(seed=2, q=8)
    bank = FakeBank()
    ids = ["warm", "cool", "warm", "cool"]
    target = lambda x: torch.stack([bank.apply(x[i], ids[i]) for i in range(x.shape[0])])
    row = A.train_step(head, opt, some_z(4), sampler, target, epoch=0.0, lut_ids=ids)
    for col in step_columns_for(3):
        assert col in row and row[col] is not None, col
    assert row["n_luts_in_batch"] == 2
    assert row["n_colors"] == 32
    assert row["mining_ratio"] == 0.10
    assert row["lr_shared_geometry"] == pytest.approx(1e-4)
    assert "degenerate_weight_rate" in row and "n_degenerate_precision" in row


def test_training_moves_the_head_off_the_identity() -> None:
    """A short real optimisation: L_rec must fall and f must stop being the identity."""
    torch.manual_seed(0)
    head = make_head(n_gauss=8)
    opt = A.build_optimizer(head)
    sampler = QuerySampler(seed=5, q=64)
    bank = FakeBank()
    ids = ["warm"] * 4
    target = lambda x: torch.stack([bank.apply(x[i], ids[i]) for i in range(x.shape[0])])
    z = some_z(4)
    first = A.train_step(head, opt, z, sampler, target, epoch=0.0, lut_ids=ids)
    for _ in range(40):
        last = A.train_step(head, opt, z, sampler, target, epoch=0.0, lut_ids=ids)
    assert last["L_rec"] < first["L_rec"]
    x = uniform_grid(5)
    with torch.no_grad():
        y = head.transform(z, x)
    assert float((y - x[None]).abs().max()) > 1e-3


# --------------------------------------------------------------------------- #
# 8. the guards
# --------------------------------------------------------------------------- #
def test_degeneracy_guard_fires_on_the_untrained_identity_head() -> None:
    """Step 0 IS the identity here, so the guard must catch it -- which is why it
    runs at the first quick eval and not at step 0."""
    head = make_head()
    with pytest.raises(DegenerateTransform) as exc:
        A.assert_not_degenerate(head, some_z(4), exit_process=False)
    assert any("identity" in f for f in exc.value.report.failures)


def test_degeneracy_guard_passes_a_live_head() -> None:
    head = perturb(make_head(), scale=0.2, seed=11)
    report = A.assert_not_degenerate(head, some_z(6), exit_process=False)
    assert report.ok
    assert report.point_std > 1e-3 and report.identity_dev > 1e-3 and report.cross_std > 1e-4


def test_degeneracy_guard_catches_a_condition_independent_head() -> None:
    """The PRND / CONDINST failure: one transform for every instruction."""
    head = perturb(make_head(), scale=0.2, seed=11)
    with torch.no_grad():                          # kill the condition path
        head.proj.proj.weight.zero_()
    with pytest.raises(DegenerateTransform) as exc:
        A.assert_not_degenerate(head, some_z(6), exit_process=False)
    assert any("every sample" in f for f in exc.value.report.failures)


def test_required_columns_are_the_twelve_plus_p1_plus_the_arm_three() -> None:
    req = A.required_columns()
    assert set(PREREGISTERED_KEYS) <= set(req)
    assert set(A.P1_CRITERIA) <= set(req)
    assert set(A.ARM_CRITERIA) <= set(req)
    assert len(req) == 12 + 4 + 3
    assert "auc" not in " ".join(req).lower()


def _board_rows(n: int = 6) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "sample_id": f"s{i}", "winner_confidence": "normal",
            "task_type": "style" if i % 2 else "local",
            "E_arm": 3.0 + 0.1 * i,
            "E_B0_identity": 30.0, "E_B1_libmean": 25.0,
            "E_B2_librandom_repeats": [35.0, 34.0], "E_B3_bucket_retrieval_repeats": [20.0, 21.0],
            "E_B4_oracle": 9.9, "E_B6_libfill": 10.2,
            "E_N1_shuffle": 6.0, "M_N1_shuffle": 4.0,
            "E_N2_irrelevant": 7.0, "M_N2_irrelevant": 5.0,
            "E_N3_const": 8.0, "M_N3_const": 6.0,
            "grid_error": 4.0, "img_error": 3.5,
        })
    return rows


def _extra_columns() -> dict:
    one = {"n": 4, "value": 1.0, "mean": 1.0}
    return {"interp_grid": dict(one), "path_len": dict(one), "mono_rate": dict(one),
            "oob_rate": dict(one), "shared_geom_identical": dict(one),
            "affine_linearity_maxdev": dict(one), "degenerate_weight_rate": dict(one)}


def _steps_row() -> dict:
    return {c: 1.0 for c in step_columns_for(3)}


def test_board_publishes_when_every_column_is_present() -> None:
    board = A.build_arm_board(_board_rows(), split="V_what", extra_columns=_extra_columns())
    assert board["arm"] == "EPR-025" and board["arm_name"] == "AFFONLY"
    assert board["contexts"]["all"]["headline_normal_only"]["n"] == 6
    report = A.publish_board(board, steps_row=_steps_row())
    assert report["criteria"]["computed"]["affine_linearity_maxdev"] == 4
    assert report["headline"]["n"] == 6


@pytest.mark.parametrize("dropped", ["affine_linearity_maxdev", "interp_grid",
                                     "shared_geom_identical", "degenerate_weight_rate"])
def test_board_refuses_a_missing_arm_column(dropped: str) -> None:
    extra = _extra_columns()
    extra.pop(dropped)
    board = A.build_arm_board(_board_rows(), split="V_what", extra_columns=extra)
    with pytest.raises(CriterionNotComputed, match=dropped):
        A.publish_board(board, steps_row=_steps_row())


def test_board_refuses_a_missing_frozen_column() -> None:
    rows = [{k: v for k, v in r.items() if k != "E_B4_oracle"} for r in _board_rows()]
    board = A.build_arm_board(rows, split="V_what", extra_columns=_extra_columns())
    with pytest.raises(CriterionNotComputed, match="B4_oracle"):
        A.publish_board(board, steps_row=_steps_row())


def test_publish_distinguishes_no_row_from_no_loss_columns(tmp_path: Path) -> None:
    """The SEGSAM / PRND failure, closed structurally: two different exceptions."""
    clear_step_witness()
    board = A.build_arm_board(_board_rows(), split="V_what", extra_columns=_extra_columns())
    with pytest.raises(StepsRowUnavailable):
        A.publish_board(board, steps_path=tmp_path / "does-not-exist.jsonl")
    bad = tmp_path / "steps.jsonl"
    bad.write_text(json.dumps({"L_rec": 1.0}) + "\n", encoding="utf-8")
    with pytest.raises(LossColumnsMissing):
        A.publish_board(board, steps_path=bad)
    good = tmp_path / "ok.jsonl"
    good.write_text(json.dumps(_steps_row()) + "\n", encoding="utf-8")
    report = A.publish_board(board, steps_path=good)
    assert report["steps"]["source"] == "disk"


def test_run_setup_records_the_frozen_block_and_the_thresholds() -> None:
    head = make_head()
    setup = A.run_setup(head)
    assert setup["arm"] == "AFFONLY" and setup["epr"] == "EPR-025"
    assert setup["frozen"]["total_steps"] == 117440
    assert setup["optimizer"]["shared_geometry_lr"] == pytest.approx(1e-4)
    assert setup["assertions"]["affine_linearity_maxdev"]["tol"] == 1e-5
    assert setup["assertions"]["degeneracy_guard"] == {
        "point_std": 1e-3, "identity_dev": 1e-3, "cross_std": 1e-4}
    assert setup["head"]["generator"]["mode"] == "affine_only"
    assert json.loads(json.dumps(setup, default=str))       # it has to serialise


# --------------------------------------------------------------------------- #
# 9. device / dtype discipline
# --------------------------------------------------------------------------- #
def test_no_bare_tensor_construction_in_a_forward() -> None:
    """Pitfall 2: constants live in non-persistent buffers, never in a forward."""
    import ast

    src = Path(A.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    offences = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in ("forward", "transform", "theta", "theta_from_u",
                             "_apply_global_affine", "transform_image"):
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Attribute) and sub.attr in ("tensor", "as_tensor"):
                if isinstance(sub.value, ast.Name) and sub.value.id == "torch":
                    offences.append(node.name)
    assert offences == [], f"bare torch.tensor(...) inside {offences}"


def test_head_buffers_are_non_persistent() -> None:
    head = make_head()
    sd = head.state_dict()
    assert not [k for k in sd if k.endswith("eye3") or k.endswith("zero3")]
    assert torch.equal(head.eye3, torch.eye(3))


def test_transform_accepts_a_foreign_dtype_condition() -> None:
    """MATTE died on a stray dtype/device; the projection casts instead of raising."""
    head = make_head()
    z = some_z(2).to(torch.bfloat16)
    with torch.no_grad():
        y = head.transform(z, uniform_grid(5))
    assert y.dtype == torch.float32 and tuple(y.shape) == (2, 125, 3)


def test_criteria_module_is_never_handed_a_cpu_copy() -> None:
    """No ``.cpu()`` call anywhere on this arm's metric path (AST, not grep:
    the module docstring names the pitfall)."""
    import ast

    tree = ast.parse(Path(A.__file__).read_text(encoding="utf-8"))
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr in ("cpu", "numpy", "item")]
    assert calls == [], f"{len(calls)} device round trips on the metric path"


# --------------------------------------------------------------------------- #
# 10. the runner's surface
# --------------------------------------------------------------------------- #
def test_runner_flags_carry_the_proposal_defaults() -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    args = R.build_argparser().parse_args(["--run-dir", "/tmp/x"])
    assert (args.share, args.global_affine) == ("geo_opacity", "affine")
    assert (args.n_gauss, args.cond_dim, args.gen_width) == (48, 64, 128)
    assert (args.clamp, args.mu_init, args.zero_init_heads) == ("two", "grid", True)
    assert (args.shared_lr_scale, args.base_lr) == (0.1, 1e-3)
    assert (args.lambda_hc, args.lambda_sparse) == (10.0, 0.001)
    # --total-steps 0 = "the horizon the measured population implies"; on the
    # frozen sft2seg split that is still 40 * ceil(93934 / 32) = 117,440.
    assert args.total_steps == 0 and args.eval_split == "V_what"
    assert (args.data, args.batch_split) == ("v2seg", "32x256")
    assert args.readout == "seg_color"
    cfg = R.config_from_args(args)
    assert cfg == A.AffineOnlyConfig()
    assert cfg.total_steps == 117440 and cfg.steps_per_epoch == 2936


def test_runner_resolves_the_epr030_caliber() -> None:
    """--batch-split 256x8192 + a measured n = 119,828 -> 469 / 18,760."""
    from q3vl.whatb.scripts import run_affonly_arm as R

    args = R.build_argparser().parse_args(
        ["--run-dir", "/tmp/x", "--batch-split", "256x8192",
         "--data", "v2seg+l8", "--loss-level", "1", "--base-lr", "1e-3"])
    cfg = R.config_from_args(args, train_n=119828)
    assert (cfg.batch_samples, cfg.queries) == (256, 8192)
    assert cfg.colors_per_step == 2_097_152
    assert cfg.steps_per_epoch == 469 and cfg.total_steps == 18_760
    assert cfg.lambda_hc_effective == 0.0 and cfg.lambda_sparse_effective == 0.0


def test_frozen_batch_splits_are_bit_identical_on_this_arm() -> None:
    """The two step-matched splits still resolve to exactly EPR-024's numbers."""
    from q3vl.whatb.scripts import run_affonly_arm as R

    for name, (b, q) in (("32x256", (32, 256)), ("64x128", (64, 128))):
        args = R.build_argparser().parse_args(
            ["--run-dir", "/tmp/x", "--batch-split", name])
        cfg = R.config_from_args(args)
        assert (cfg.batch_samples, cfg.queries) == (b, q)
        assert cfg.colors_per_step == 8192
        assert cfg.lambda_hc_effective == 10.0
        assert cfg.lambda_sparse_effective == 0.001


def test_runner_accepts_both_flag_spellings() -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    ap = R.build_argparser()
    a = ap.parse_args(["--run-dir", "/tmp/x", "--num-gauss", "32", "--shared-lr-scale", "1.0"])
    b = ap.parse_args(["--run-dir", "/tmp/x", "--n-gauss", "32", "--shared-geom-lr-scale", "1.0"])
    assert (a.n_gauss, a.shared_lr_scale) == (32, 1.0) == (b.n_gauss, b.shared_lr_scale)


def test_runner_refuses_a_hard_mount_run_dir() -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    with pytest.raises(SystemExit, match="local disk"):
        R.main(["--run-dir", "/mnt/nfs/whatever"])


def test_runner_refuses_synthetic_conditions_outside_smoke(tmp_path: Path) -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    with pytest.raises(SystemExit, match="smoke-only"):
        R.main(["--run-dir", str(tmp_path), "--z-source", "synthetic",
                "--skip-colorspan-assert"])


def _leaf(root: Path, split: str, tag: str, *, checkpoint: str = "/base/ckpt",
          readout: str = "seg_color", meta_tag: str | None = None,
          dtype=np.float32) -> Path:
    """One leaf of the shared cache layout (q3vl.whatb.zcache)."""
    import json

    d = Path(root) / f"{split}__{tag}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.jsonl").write_text(
        json.dumps({"sample_id": "a", "row": 0}) + "\n", encoding="utf-8")
    np.save(d / "z.npy", np.zeros((1, 2560), dtype=dtype))
    (d / "meta.json").write_text(json.dumps(
        {"checkpoint": checkpoint, "readout_kind": readout, "split": split,
         "control_tag": meta_tag or tag, "context_source": "generated", "n": 1,
         "dtype": str(np.dtype(dtype))}), encoding="utf-8")
    return d


def test_open_z_asserts_the_checkpoint_the_readout_and_the_tag(tmp_path: Path) -> None:
    """The shared reader's three start-up assertions, through this arm's seam."""
    from q3vl.whatb.scripts import run_affonly_arm as R
    from q3vl.whatb.zcache import ZCacheDtypeError

    d = _leaf(tmp_path / "a", "V_what", "none", checkpoint="/other/ckpt")
    with pytest.raises(AssertionError, match="checkpoint"):
        R.open_z(d, tag="none", checkpoint="/base/ckpt", readout="seg_color",
                 split="V_what")
    d = _leaf(tmp_path / "b", "V_what", "none", readout="im_end")
    with pytest.raises(AssertionError, match="readout_kind"):
        R.open_z(d, tag="none", checkpoint="/base/ckpt", readout="seg_color",
                 split="V_what")
    d = _leaf(tmp_path / "c", "V_what", "none", meta_tag="shuffle")
    with pytest.raises(AssertionError, match="control_tag"):
        R.open_z(d, tag="none", checkpoint="/base/ckpt", readout="seg_color",
                 split="V_what")
    d = _leaf(tmp_path / "d", "V_what", "none", dtype=np.float16)
    with pytest.raises(ZCacheDtypeError, match="fp32"):
        R.open_z(d, tag="none", checkpoint="/base/ckpt", readout="seg_color",
                 split="V_what")
    # the leaf is derived from (split, tag) when a root is given
    _leaf(tmp_path / "e", "V_what", "none")
    cache = R.open_z(None, root=tmp_path / "e", tag="none",
                     checkpoint="/base/ckpt", readout="seg_color", split="V_what")
    assert "a" in cache and cache.vector("a").shape == (2560,)


def test_synthetic_z_is_deterministic_and_condition_dependent() -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    src = R.open_z(None, tag="none", checkpoint="c", readout="seg_color",
                   split="V_what", synthetic=True)
    assert torch.equal(src.get("s1"), src.get("s1"))
    assert not torch.equal(src.get("s1"), src.get("s2"))
    other = R.open_z(None, tag="shuffle", checkpoint="c", readout="seg_color",
                     split="V_what", synthetic=True)
    assert not torch.equal(src.get("s1"), other.get("s1"))


def test_the_degeneracy_guard_is_not_tied_to_the_quick_eval_flag() -> None:
    """``--eval-every 0`` must not be a way to reach a board unguarded."""
    from q3vl.whatb.scripts import run_affonly_arm as R

    src = Path(R.__file__).read_text(encoding="utf-8")
    assert 'guard_once("final_eval")' in src
    assert src.count("def guard_once") == 1
    assert "A.assert_not_degenerate(" in src


def test_same_source_pairs_takes_one_pair_per_source_with_distinct_luts() -> None:
    from q3vl.whatb.scripts import run_affonly_arm as R

    class Row:
        def __init__(self, sid, src):
            self.sample_id, self.source_image_id = sid, src

    class S:
        def __init__(self, sid, src, lut):
            self.row, self.lut_id = Row(sid, src), lut

    samples = [S("a", "src1", "L1"), S("b", "src1", "L1"), S("c", "src1", "L2"),
               S("d", "src2", "L3"), S("e", "src2", "L4"), S("f", "src3", "L5")]
    pairs = R._same_source_pairs(samples, limit=10)
    assert [(a.lut_id, b.lut_id) for a, b in pairs] == [("L1", "L2"), ("L3", "L4")]
    assert R._same_source_pairs(samples, limit=1) == pairs[:1]


def test_mean_lut_volume_axis_order_round_trips() -> None:
    """B1's volume is built on an (r,g,b)-indexed grid and consumed as (b,g,r)."""
    from q3vl.whatb.lutdata import apply_lut_volume
    from q3vl.whatb.scripts import run_affonly_arm as R

    class OneBank:
        def apply(self, x, lut_id):        # a map that is different on all three axes
            return torch.stack([x[..., 0] * 0.5, x[..., 1] * 0.25, x[..., 2]], dim=-1)

    vol = R.mean_lut_volume(OneBank(), ["only"], grid_n=9)
    x = uniform_grid(9)
    got = apply_lut_volume(vol, x)
    assert torch.allclose(got, OneBank().apply(x, "only"), atol=1e-6)
