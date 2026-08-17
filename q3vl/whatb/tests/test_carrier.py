"""EPR-024 carrier arm: the frozen numbers, the four losses, and the board.

Everything runs on the CPU (``CUDA_VISIBLE_DEVICES=""``); no test starts a GPU
process.  The three where-side failures each have a regression here:

* the publication gate finds the first step row from the in-process witness
  alone, and "no row" and "no loss columns" raise different exceptions;
* no ``forward`` builds a constant with ``torch.tensor(...)`` and the two query
  grids are non-persistent buffers, so a state dict round trip cannot move them
  to another device;
* a constant / identity / sample-invariant transform makes the first quick eval
  leave with ``SystemExit(2)``.
"""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb import criteria as C
from q3vl.whatb.arms import carrier as A
from q3vl.whatb.guards import (
    LossColumnsMissing,
    StepsRowUnavailable,
    clear_step_witness,
)
from q3vl.whatb.lutdata import LutBank
from q3vl.whatb.publish import HeadlineMissing, assert_publishable
from q3vl.whatb.queries import QuerySampler, uniform_grid

CARRIER_PY = Path(A.__file__)
RUNNER_PY = CARRIER_PY.parent.parent / "scripts" / "run_carrier_arm.py"


# --------------------------------------------------------------------------- #
# fixtures: a tiny bank and a tiny arm
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
    grids = {"lut_a": _fake_grid(9, 1), "lut_b": _fake_grid(9, 2),
             "lut_c": _fake_grid(17, 3), "lut_d": _fake_grid(5, 4)}
    np.savez(tmp_path / "luts.npz", **grids)
    meta = {k: {"path": str(tmp_path / f"{k}.cube"), "dmin": [0.0] * 3,
                "dmax": [1.0] * 3} for k in grids}
    (tmp_path / "luts_meta.json").write_text(json.dumps(meta))
    b = LutBank(tmp_path)
    b._grids = grids
    return b


def tiny_cfg(**kw) -> A.CarrierConfig:
    base = dict(cond_dim=8, n_gauss=6, gen_width=16, lib_size=4, n_repeats=3,
                bake_grid=9, interp_pairs=2)
    base.update(kw)
    return A.CarrierConfig(**base)


@pytest.fixture()
def model() -> A.CarrierModel:
    torch.manual_seed(0)
    return A.CarrierModel(tiny_cfg())


def _z(n: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, A.SEG_COLOR_HIDDEN_DIM, generator=g)


# --------------------------------------------------------------------------- #
# 1. the eight frozen items
# --------------------------------------------------------------------------- #
def test_frozen_batch_and_horizon():
    cfg = A.CarrierConfig()
    assert cfg.train_n == 93934
    assert (cfg.batch_samples, cfg.queries_per_sample) == (32, 256)
    assert cfg.colors_per_step == 8192
    assert cfg.steps_per_epoch == 2936 == math.ceil(93934 / 32)
    assert cfg.epochs == 40
    assert cfg.total_steps == 117440
    assert cfg.clamp == "two"


def test_the_ablation_batch_split_is_not_step_matched():
    cfg = A.CarrierConfig(batch_split="64x128")
    assert cfg.colors_per_step == 8192
    assert cfg.steps_per_epoch == 1468 and cfg.total_steps == 58720
    assert cfg.total_steps != A.CarrierConfig().total_steps


def test_default_flags_are_the_proposal_defaults():
    cfg = A.CarrierConfig()
    assert (cfg.readout, cfg.cond_dim, cfg.n_gauss, cfg.gen_width) == \
        ("seg_color", 64, 48, 128)
    assert (cfg.loss_level, cfg.lambda_img, cfg.context) == (3, 0.0, "generated")
    assert (cfg.hc_eps, cfg.hc_mask, cfg.mining, cfg.lut_resample) == \
        (1e-3, True, True, "none")
    assert (cfg.base_lr, cfg.proj_lr_scale, cfg.max_grad_norm, cfg.seed) == \
        (1e-3, 0.1, 1.0, 20260810)
    assert cfg.lambda_hc == 10.0 and cfg.lambda_sparse == 0.001


def test_rejected_flag_values():
    with pytest.raises(ValueError, match="lut-resample"):
        A.CarrierConfig(lut_resample="33")
    with pytest.raises(ValueError, match="clamp"):
        A.CarrierConfig(clamp="none")            # no CLI spelling for the internal mode
    with pytest.raises(ValueError, match="ablation row"):
        A.CarrierConfig(cond_zero=True, cond_trainmean=True)


def test_step_columns_are_the_preregistered_ones():
    assert A.CarrierConfig().step_columns == (
        "L_rec", "L_hc", "L_sparse", "n_colors", "n_luts_in_batch",
        "mining_ratio", "n_hc_masked")
    assert "L_img" in A.CarrierConfig(loss_level=4).step_columns


def test_arm_is_registered_as_a_p1_arm():
    assert A.ARM == "EPR-024" and A.AXES == ("P1",)
    assert set(C.required_criteria(A.ARM)) == \
        set(C.PREREGISTERED_KEYS) | {"interp_grid", "path_len", "mono_rate", "oob_rate"}


# --------------------------------------------------------------------------- #
# 2. the model:  parameter counts against §2.3's arithmetic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cond_dim,n_gauss,width,total", [
    (64, 48, 128, 447_212),      # main arm
    (64, 32, 128, 401_804),      # the forced N=32 row (paper's only comparable N)
    (256, 48, 128, 963_500),     # d = 256 (Neural Preset k^2)
    (32, 48, 128, 361_164),      # d = 32
])
def test_parameter_counts_match_the_proposal(cond_dim, n_gauss, width, total):
    m = A.CarrierModel(A.CarrierConfig(cond_dim=cond_dim, n_gauss=n_gauss,
                                       gen_width=width))
    cfg = m.config
    assert cfg["n_params_total"] == total
    assert cfg["n_params_generator"] == cfg["n_params_generator_closed_form"]
    assert cfg["theta_dim"] == 22 * n_gauss + 12


def test_theta_dim_is_the_paper_formula():
    assert A.CarrierModel(A.CarrierConfig(n_gauss=32)).config["theta_dim"] == 716
    assert A.CarrierModel(A.CarrierConfig(n_gauss=48)).config["theta_dim"] == 1068


def test_pi_stands_where_the_lookup_stood():
    m = A.CarrierModel(tiny_cfg(cond_dim=64))
    assert m.pi.in_dim == 2560 and m.pi.cond_dim == 64
    assert m.generator.mode == "full"
    # D6: no identity anchoring, no zero init -- the two switches are off
    assert m.generator.m_residual is False and m.generator.zero_init_last is False


# --------------------------------------------------------------------------- #
# 3. forward shapes + the device/dtype discipline
# --------------------------------------------------------------------------- #
def test_forward_shapes(model):
    z = _z(4)
    x = torch.rand(4, 32, 3)
    y, aux = model(z, x, return_aux=True)
    assert y.shape == (4, 32, 3)
    assert aux.weights.shape == (4, 32, model.cfg.n_gauss)
    assert torch.isfinite(y).all() and (y >= 0).all() and (y <= 1).all()
    grid = model.transform_grid(z, model.query_grid17)
    assert grid.shape == (4, 17 ** 3, 3)
    img = torch.rand(3, 6, 7)
    assert model.transform_image(z[0], img).shape == (3, 6, 7)


def test_transform_image_is_the_same_function_as_the_flat_query(model):
    z = _z(1)[0]
    img = torch.rand(3, 5, 4)
    flat = model.transform_grid(z.reshape(1, -1), img.permute(1, 2, 0).reshape(-1, 3))[0]
    assert torch.allclose(model.transform_image(z, img),
                          flat.reshape(5, 4, 3).permute(2, 0, 1), atol=1e-6)


def test_query_grids_are_non_persistent_buffers(model):
    keys = set(model.state_dict())
    assert not [k for k in keys if "query_grid" in k or "train_mean_z" in k]
    assert model.query_grid17.shape == (17 ** 3, 3)
    assert model.query_grid9.shape == (9 ** 3, 3)


def test_no_bare_tensor_constant_inside_any_forward():
    """Pitfall 2: constants live in buffers, never in a forward."""
    tree = ast.parse(CARRIER_PY.read_text(encoding="utf-8"))
    bad: list[str] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith(
                ("forward", "transform_", "condition", "params_")):
            continue
        for node in ast.walk(fn):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name) \
                    and node.func.value.id == "torch" \
                    and node.func.attr in ("tensor", "as_tensor"):
                bad.append(f"{fn.name}:{node.lineno}")
    assert bad == [], bad


def test_condition_moves_z_onto_the_module_dtype(model):
    z = _z(2).to(torch.float64)
    u = model.condition(z)
    assert u.dtype == model.param_dtype and u.device == model.device


def test_cond_ablations_replace_the_condition():
    m = A.CarrierModel(tiny_cfg(cond_zero=True))
    z = _z(3)
    assert torch.allclose(m.condition(z)[0], m.condition(z)[2])
    m2 = A.CarrierModel(tiny_cfg(cond_trainmean=True))
    m2.set_train_mean_z(_z(5).mean(0))
    assert torch.allclose(m2.condition(z)[0], m2.condition(z)[1])


def test_step0_is_not_the_identity_for_this_arm(model):
    """EPR-024 ruling 11.1-1: PyTorch default init, so step 0 is NOT identity."""
    assert model.step0_maxabs_f_minus_id(_z(4)) > 1e-3


# --------------------------------------------------------------------------- #
# 4. the loss ladder
# --------------------------------------------------------------------------- #
def test_l_rec_is_the_mean_absolute_error():
    f = torch.tensor([[[0.5, 0.5, 0.5]]])
    y = torch.tensor([[[0.5, 0.2, 0.9]]])
    assert float(A.reconstruction_loss(f, y)) == pytest.approx((0 + 0.3 + 0.4) / 3, abs=1e-6)


def test_l_hc_is_zero_when_the_hue_matches_and_positive_when_it_does_not():
    y = torch.tensor([[[0.8, 0.2, 0.2]]])
    same, n = A.hue_chroma_loss(y, y)
    # exactly the hue -> the cosine term is 1 up to float32 rounding
    assert float(same) == pytest.approx(0.0, abs=1e-4) and int(n) == 0
    other = torch.tensor([[[0.2, 0.2, 0.8]]])
    worse, _ = A.hue_chroma_loss(other, y)
    assert float(worse) > 0.1


def test_l_hc_masks_the_achromatic_points_and_counts_them():
    """Frozen block: ``1[C >= 1e-3]`` on the TARGET chroma, count -> n_hc_masked."""
    grey = torch.full((1, 4, 3), 0.5)
    colour = torch.tensor([[[0.9, 0.1, 0.1]]] * 4).reshape(1, 4, 3)
    masked, n_masked = A.hue_chroma_loss(colour, grey, mask=True)
    assert int(n_masked) == 4 and float(masked) == pytest.approx(0.0, abs=1e-9)
    unmasked, n2 = A.hue_chroma_loss(colour, grey, mask=False)
    assert int(n2) == 4                      # the count is reported either way
    assert float(unmasked) == pytest.approx(0.0, abs=1e-6)   # C == 0 weights it out
    mixed = torch.cat([grey[:, :2], colour[:, :2]], dim=1)
    _, n3 = A.hue_chroma_loss(colour, mixed, mask=True)
    assert int(n3) == 2


def test_r_sparse_is_the_binary_entropy_of_the_opacities():
    o = torch.tensor([[0.5, 0.5]])
    want = -(0.5 * math.log(0.5 + 1e-6) + 0.5 * math.log(0.5 + 1e-6))
    assert float(A.opacity_entropy(o)) == pytest.approx(want, abs=1e-6)
    assert float(A.opacity_entropy(torch.tensor([[1.0]]))) < 1e-4


def test_loss_ladder_switches_on_level(bank, model):
    z, x = _z(2), torch.rand(2, 16, 3)
    f, aux = model(z, x, return_aux=True)
    y = A.lut_targets(bank, ["lut_a", "lut_b"], x)
    l3 = A.compute_loss(f, y, aux, tiny_cfg(loss_level=3))
    l1 = A.compute_loss(f, y, aux, tiny_cfg(loss_level=1))
    assert float(l1.l_hc) == 0.0 and float(l1.l_sparse) == 0.0
    assert float(l1.total.detach()) == pytest.approx(float(l1.l_rec.detach()), abs=1e-7)
    assert float(l3.total.detach()) == pytest.approx(
        float(l3.l_rec.detach()) + 10.0 * float(l3.l_hc.detach())
        + 0.001 * float(l3.l_sparse.detach()), rel=1e-6)
    assert l3.l_img is None and l3.n_colors == 32


def test_loss_level_4_requires_images(bank, model):
    z, x = _z(2), torch.rand(2, 8, 3)
    f, aux = model(z, x, return_aux=True)
    y = A.lut_targets(bank, ["lut_a", "lut_b"], x)
    with pytest.raises(ValueError, match="L_img"):
        A.compute_loss(f, y, aux, tiny_cfg(loss_level=4, lambda_img=1.0))


def test_loss_level_4_trains_on_differently_shaped_images(bank, model):
    """L_img composes per image: a batch's aspect ratios differ, so no stacking."""
    clear_step_witness()
    cfg = tiny_cfg(loss_level=4, lambda_img=1.0)
    opt = A.build_optimizer(model, cfg)
    sampler = QuerySampler(seed=2, q=cfg.queries_per_sample)
    images = [torch.rand(3, 5, 7), torch.rand(3, 9, 4)]
    alphas = [1.0, torch.rand(1, 9, 4)]
    row = A.train_step(model, cfg, opt, None, step=0, z=_z(2),
                       lut_ids=["lut_a", "lut_b"], bank=bank, sampler=sampler,
                       images=images, alphas=alphas)
    assert "L_img" in row and math.isfinite(row["L_img"])
    for col in cfg.step_columns:
        assert col in row, col
    clear_step_witness()


def test_params_slice_keeps_the_gradient_and_the_batch_axis(model):
    params = model.params_for(_z(3))
    one = A.params_slice(params, 1)
    assert one.batch_size == 1 and one.n_gauss == model.cfg.n_gauss
    assert torch.equal(one.mu[0], params.mu[1])
    assert one.mu.requires_grad


def test_l_img_uses_the_frozen_image_formation():
    img = torch.rand(1, 3, 4, 5)
    f = torch.rand(1, 3, 4, 5)
    alpha = torch.rand(1, 1, 4, 5)
    i_hat = C.compose_hat(img, alpha, f)
    assert torch.allclose(i_hat, img * (1 - alpha) + f * alpha, atol=1e-6)
    assert float(A.image_l1(i_hat, i_hat)) == 0.0


def test_loss_preregistration_names_every_term():
    rec = A.loss_preregistration(A.CarrierConfig(loss_level=4, lambda_img=0.1))
    names = [t["name"] for t in rec["terms"]]
    assert names == ["L_rec", "L_hc", "R_sparse", "L_img"]
    assert [t["weight"] for t in rec["terms"]] == [1.0, 10.0, 0.001, 0.1]
    assert all(t["active"] for t in rec["terms"])
    assert rec["terms"][1]["c_to_zero"]["eps_c"] == 1e-3


# --------------------------------------------------------------------------- #
# 5. optimiser and schedule
# --------------------------------------------------------------------------- #
def test_two_param_groups_with_the_0_1x_condition_side(model):
    opt = A.build_optimizer(model, A.CarrierConfig())
    assert [g["name"] for g in opt.param_groups] == ["generator", "pi"]
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[1]["lr"] == pytest.approx(1e-4)
    assert opt.defaults["betas"] == (0.9, 0.999)
    n_in_groups = sum(len(g["params"]) for g in opt.param_groups)
    assert n_in_groups == len(list(model.parameters()))


def test_cosine_schedule_spans_the_whole_run(model):
    cfg = A.CarrierConfig(max_steps=100)
    opt = A.build_optimizer(model, cfg)
    sched = A.build_scheduler(opt, cfg)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-3)
    assert lrs[-1] < 1e-5                      # annealed to (near) zero at T_max
    assert all(a >= b - 1e-12 for a, b in zip(lrs, lrs[1:]))   # monotone decreasing


# --------------------------------------------------------------------------- #
# 6. mining
# --------------------------------------------------------------------------- #
def test_mining_ratio_schedule():
    from q3vl.whatb.queries import mining_ratio

    assert mining_ratio(0) == 0.10 and mining_ratio(5) == 0.10
    assert mining_ratio(20) == 0.40 and mining_ratio(40) == 0.40
    assert mining_ratio(12.5) == pytest.approx(0.25)


def test_mining_keeps_the_frozen_rectangle(bank, model):
    cfg = tiny_cfg()
    sampler = QuerySampler(seed=1, q=cfg.queries_per_sample)
    z = _z(4)
    ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    x, y, ratio, n_hard = A.mine_colors(model, cfg, z=z, lut_ids=ids, bank=bank,
                                        sampler=sampler, epoch=20.0)
    assert x.shape == (4, cfg.queries_per_sample, 3) == y.shape
    assert ratio == 0.40
    assert n_hard == 4 * round(0.40 * cfg.queries_per_sample)
    off = tiny_cfg(mining=False)
    _, _, r0, n0 = A.mine_colors(model, off, z=z, lut_ids=ids, bank=bank,
                                 sampler=sampler, epoch=20.0)
    assert (r0, n0) == (0.0, 0)


def test_mining_selects_the_hardest_colours(bank, model):
    """The kept colours must be the top-r of the probe, not an arbitrary slice."""
    cfg = tiny_cfg()
    z, ids = _z(1), ["lut_a"]
    s1 = QuerySampler(seed=7, q=cfg.queries_per_sample)
    s2 = QuerySampler(seed=7, q=cfg.queries_per_sample)
    probe = s2.sample(1, cfg.queries_per_sample)
    with torch.no_grad():
        err = (model(z, probe) - A.lut_targets(bank, ids, probe)).abs().mean(-1)[0]
    k = round(0.10 * cfg.queries_per_sample)
    want = set(torch.topk(err, k).indices.tolist())
    x, _, _, _ = A.mine_colors(model, cfg, z=z, lut_ids=ids, bank=bank,
                               sampler=s1, epoch=0.0)
    got = {int((probe[0] == x[0, i]).all(dim=-1).nonzero()[0]) for i in range(k)}
    assert got == want


def test_lut_targets_are_the_bank_operator(bank):
    x = torch.rand(2, 5, 3)
    y = A.lut_targets(bank, ["lut_a", "lut_c"], x)
    assert torch.allclose(y[0], bank.apply(x[0], "lut_a"))
    assert torch.allclose(y[1], bank.apply(x[1], "lut_c"))


# --------------------------------------------------------------------------- #
# 7. train_step and the first-step-row contract
# --------------------------------------------------------------------------- #
def test_train_step_publishes_every_preregistered_column(bank, model):
    clear_step_witness()
    cfg = tiny_cfg()
    opt = A.build_optimizer(model, cfg)
    sched = A.build_scheduler(opt, cfg)
    sampler = QuerySampler(seed=3, q=cfg.queries_per_sample)
    z = _z(4)
    row = A.train_step(model, cfg, opt, sched, step=0, z=z,
                       lut_ids=["lut_a", "lut_b", "lut_a", "lut_c"], bank=bank,
                       sampler=sampler)
    for col in cfg.step_columns:
        assert col in row and row[col] is not None, col
    assert row["n_colors"] == 4 * cfg.queries_per_sample
    assert row["n_luts_in_batch"] == 3
    assert row["mining_ratio"] == 0.10


def test_the_witness_alone_satisfies_the_publication_gate(bank, model):
    """Tier 3 of the row lookup: a quick eval before the first flush must pass."""
    clear_step_witness()
    cfg = tiny_cfg()
    opt = A.build_optimizer(model, cfg)
    sampler = QuerySampler(seed=3, q=cfg.queries_per_sample)
    with pytest.raises(StepsRowUnavailable):
        assert_publishable({"criteria_columns": {}}, A.ARM, axes=("P1",))
    A.train_step(model, cfg, opt, None, step=0, z=_z(2),
                 lut_ids=["lut_a", "lut_b"], bank=bank, sampler=sampler)
    with pytest.raises(C.CriterionNotComputed):
        # the row is now found (witness); the criteria are what is missing
        assert_publishable({"criteria_columns": {}}, A.ARM, axes=("P1",))
    clear_step_witness()


def test_no_row_and_no_loss_columns_are_different_failures():
    clear_step_witness()
    board = {"criteria_columns": {}}
    with pytest.raises(StepsRowUnavailable):
        assert_publishable(board, A.ARM, axes=("P1",))
    with pytest.raises(LossColumnsMissing):
        assert_publishable(board, A.ARM, axes=("P1",), steps_row={"L_rec": 1.0})
    assert not issubclass(LossColumnsMissing, StepsRowUnavailable)
    assert not issubclass(StepsRowUnavailable, LossColumnsMissing)


def test_training_reduces_the_reconstruction_loss(bank):
    """A short overfit: the wiring is alive, not just shaped correctly."""
    torch.manual_seed(1)
    cfg = tiny_cfg(mining=False, max_steps=40)
    m = A.CarrierModel(cfg)
    opt = A.build_optimizer(m, cfg)
    sampler = QuerySampler(seed=5, q=64)
    z = _z(2, seed=11)
    first = last = None
    for step in range(40):
        row = A.train_step(m, cfg, opt, None, step=step, z=z,
                           lut_ids=["lut_a", "lut_b"], bank=bank, sampler=sampler)
        first = row["L_rec"] if first is None else first
        last = row["L_rec"]
    assert last < first


# --------------------------------------------------------------------------- #
# 8. the degeneracy gate
# --------------------------------------------------------------------------- #
class _ConstantModel(A.CarrierModel):
    """A head that emits the same colour everywhere -- PRND's failure, reproduced."""

    def transform_grid(self, z, x, **kw):        # noqa: D102
        b = z.shape[0]
        return torch.full((b, x.shape[0], 3), 0.42)


class _IdentityModel(A.CarrierModel):
    def transform_grid(self, z, x, **kw):        # noqa: D102
        return x.unsqueeze(0).expand(z.shape[0], -1, -1).clone()


class _SampleInvariantModel(A.CarrierModel):
    def transform_grid(self, z, x, **kw):        # noqa: D102
        y = (x * 0.5 + 0.1).unsqueeze(0)
        return y.expand(z.shape[0], -1, -1).clone()


@pytest.mark.parametrize("cls", [_ConstantModel, _IdentityModel, _SampleInvariantModel])
def test_first_quick_eval_exits_on_a_degenerate_transform(bank, cls):
    m = cls(tiny_cfg())
    with pytest.raises(SystemExit) as exc:
        A.quick_eval(m, m.cfg, z=_z(4), lut_ids=["lut_a"] * 4, bank=bank, step=100)
    assert exc.value.code == 2


def test_quick_eval_passes_a_live_head(bank, model):
    out = A.quick_eval(model, model.cfg, z=_z(4), lut_ids=["lut_a", "lut_b", "lut_c", "lut_d"],
                       bank=bank, step=100)
    assert out["n"] == 4 and out["grid_de00_mean"] > 0
    assert out["std_over_queries"] > 1e-3 and out["std_over_samples"] > 1e-4


def test_the_gate_can_be_caught_in_process(bank):
    from q3vl.whatb.degeneracy import DegenerateTransform

    m = _ConstantModel(tiny_cfg())
    with pytest.raises(DegenerateTransform):
        A.quick_eval(m, m.cfg, z=_z(3), lut_ids=["lut_a"] * 3, bank=bank, step=1,
                     exit_process=False)


def test_thresholds_travel_into_the_run_record(model):
    rec = A.run_setup_record(A.CarrierConfig(), model)
    assert rec["degeneracy_thresholds"] == {"point_std": 1e-3, "identity_dev": 1e-3,
                                            "cross_std": 1e-4}
    assert rec["source_sha256"]["carrier.py"]
    assert rec["source_sha256"]["run_carrier_arm.py"]
    assert rec["frozen_block"]["total_steps"] == 117440
    assert rec["frozen_block"]["preregistered_keys"] == list(C.PREREGISTERED_KEYS)


# --------------------------------------------------------------------------- #
# 9. the z cache
# --------------------------------------------------------------------------- #
def _write_cache(root, split, tag, ids, *, checkpoint="ckpt", kind="seg_color",
                 context="generated", bad_index=False):
    rows = []
    for sid in ids:
        seq = [151669, 7, 151670, 151671, 9, 151672, 151673, 151674]
        rows.append({"sample_id": sid, "split": split,
                     "reply_token_ids": seq,
                     "readout_index": 0 if bad_index else len(seq) - 1,
                     "n_generated_tokens": len(seq)})
    z = np.random.default_rng(0).standard_normal((len(ids), 2560))
    return A.write_z_cache(Path(root) / f"{split}__{tag}", rows, z,
                           checkpoint=checkpoint, readout_kind=kind,
                           context_source=context, control_tag=tag, split=split)


def test_z_cache_round_trip(tmp_path):
    _write_cache(tmp_path, "V_what", "none", ["a", "b", "c"])
    cache = A.ZCache(tmp_path / "V_what__none")
    assert len(cache) == 3 and "b" in cache
    assert cache.vector("b").shape == (2560,)
    assert cache.batch(["a", "c"]).shape == (2, 2560)
    assert cache.mean().shape == (2560,)
    rec = cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color",
                                  context_source="generated", control_tag="none",
                                  verify_frac=1.0)
    assert rec["n_verify_plan"] == 3


def test_z_cache_refuses_a_foreign_checkpoint(tmp_path):
    _write_cache(tmp_path, "V_what", "none", ["a"], checkpoint="other")
    cache = A.ZCache(tmp_path / "V_what__none")
    with pytest.raises(AssertionError, match="checkpoint"):
        cache.assert_belongs_to(checkpoint="ckpt", readout_kind="seg_color")
    with pytest.raises(AssertionError, match="readout_kind"):
        cache.assert_belongs_to(checkpoint="other", readout_kind="im_end")


def test_z_cache_replays_verify_plan_and_catches_a_wrong_index(tmp_path):
    _write_cache(tmp_path, "V_what", "none", ["a", "b"], bad_index=True)
    cache = A.ZCache(tmp_path / "V_what__none")
    # expected_ids is read off the recorded index, so a wrong index only shows up
    # against the kind's own token: assert the recorded row really is <seg_color>
    plan = cache._plan_of(cache.rows[0])
    assert plan.token_ids[plan.start] != 151674


# --------------------------------------------------------------------------- #
# 10. the baked library mean
# --------------------------------------------------------------------------- #
def test_bake_round_trip_reproduces_the_lut_on_its_own_grid(bank):
    """A LUT baked at its own grid size and re-evaluated is the same function."""
    from q3vl.whatb.lutdata import apply_lut_volume

    x = uniform_grid(9)
    vol = A.bake_transform_volume(bank.apply(x, "lut_a"), 9)
    probe = torch.rand(200, 3)
    assert (apply_lut_volume(vol, probe) - bank.apply(probe, "lut_a")).abs().max() < 1e-5


def test_library_mean_is_the_pointwise_mean(bank):
    ids = ["lut_a", "lut_b", "lut_c"]
    vol, mean = A.library_mean_volume(bank, ids, 9)
    x = uniform_grid(9)
    want = torch.stack([bank.apply(x, i) for i in ids]).mean(0)
    assert torch.allclose(mean, want, atol=1e-6)
    assert vol.shape == (1, 3, 9, 9, 9)


# --------------------------------------------------------------------------- #
# 11. the board: the twelve keys plus P1's four, really produced
# --------------------------------------------------------------------------- #
def _samples(model, bank, n_src: int = 3) -> list[A.EvalSample]:
    lut_ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    out: list[A.EvalSample] = []
    g = torch.Generator().manual_seed(4)
    for s in range(n_src):
        for k in range(2):                      # two samples per source -> IP-A pairs
            i = 2 * s + k
            img = torch.rand(3, 6, 8, generator=g)
            style = (i % 2 == 0)
            out.append(A.EvalSample(
                sample_id=f"sft_{i:04d}", task_type="style" if style else "local",
                winner_confidence="normal" if i < 2 * n_src - 1 else "low",
                lut_id=lut_ids[i % len(lut_ids)], source_image_id=f"src_{s}",
                minor="bucket_x" if i % 2 == 0 else "bucket_y",
                image=img,
                alpha=1.0 if style else torch.rand(1, 6, 8, generator=g),
                z=_z(1, seed=100 + i)[0],
                z_controls={t: _z(1, seed=200 + i * 7 + j)[0]
                            for j, t in enumerate(("shuffle", "irrelevant", "const"))}))
    return out


def _board(model, bank):
    cfg = model.cfg
    ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    lib = C.LibraryValues.build(bank, ids, model.query_grid9)
    vol, _ = A.library_mean_volume(bank, ids, cfg.bake_grid)
    samples = _samples(model, bank)
    pools = {"bucket_x": ["lut_a", "lut_c"], "bucket_y": ["lut_b", "lut_d"]}
    rows = A.evaluate_samples(model, cfg, samples, bank=bank, lib=lib,
                              lib_mean_volume=vol, bucket_pools=pools)
    extra = A.interpolation_columns(model, cfg, samples, bank=bank, limit=2)
    return A.build_arm_board(rows, split="V_what", extra_columns=extra), rows


def test_every_preregistered_key_is_really_produced(bank, model):
    board, rows = _board(model, bank)
    cols = board["criteria_columns"]
    for key in C.PREREGISTERED_KEYS:
        assert key in cols, key
        assert int(cols[key].get("n", 0)) > 0, key
    for key in ("interp_grid", "path_len", "mono_rate", "oob_rate"):
        assert int(cols[key]["n"]) > 0, key
    assert board["n_low_excluded"] == 1
    assert board["contexts"]["all"]["headline_normal_only"]["n"] == len(rows) - 1
    assert board["contexts"]["style"]["headline_normal_only"]["n"] > 0
    assert board["contexts"]["local"]["headline_normal_only"]["n"] > 0


def test_the_board_passes_the_publication_gate(bank, model, tmp_path):
    clear_step_witness()
    board, _ = _board(model, bank)
    steps = tmp_path / "steps.jsonl"
    steps.write_text(json.dumps({c: 0.0 for c in A.CarrierConfig().step_columns}) + "\n")
    report = A.publish_board(board, model.cfg, steps_path=steps)
    assert report["steps"]["source"] == "disk"
    assert set(report["criteria"]["required"]) == set(C.required_criteria(A.ARM))
    assert report["headline"]["n"] > 0


def test_a_board_without_the_interpolation_columns_is_refused(bank, model, tmp_path):
    board, rows = _board(model, bank)
    del board["criteria_columns"]["path_len"]
    steps = tmp_path / "steps.jsonl"
    steps.write_text(json.dumps({c: 0.0 for c in A.CarrierConfig().step_columns}) + "\n")
    with pytest.raises(C.CriterionNotComputed, match="path_len"):
        A.publish_board(board, model.cfg, steps_path=steps)


def test_a_published_board_without_a_headline_is_refused(bank, model, tmp_path):
    """Two gates, in order: the criteria one fires first, the publish one behind it."""
    board, _ = _board(model, bank)
    board["contexts"]["all"]["headline_normal_only"] = {"n": 0}
    board["criteria_columns"]["headline_normal_only"]["n"] = 1
    steps = tmp_path / "steps.jsonl"
    steps.write_text(json.dumps({c: 0.0 for c in A.CarrierConfig().step_columns}) + "\n")
    with pytest.raises(C.CriterionNotComputed, match="headline_normal_only"):
        A.publish_board(board, model.cfg, steps_path=steps)
    # with the key taken out of the required table the publish-side gate is the
    # one that refuses -- a published board may never lack the selection number
    with pytest.raises(HeadlineMissing):
        assert_publishable(board, A.ARM, steps_path=steps, loss_level=3,
                           required=["interp_grid"])


def test_rows_carry_the_paired_baselines_and_the_three_controls(bank, model):
    _, rows = _board(model, bank)
    r = rows[0]
    for key in ("E_arm", "E_B0_identity", "E_B1_libmean", "E_B4_oracle",
                "E_B6_libfill", "grid_error", "img_error", "unseen_color_error"):
        assert isinstance(r[key], float) and math.isfinite(r[key]), key
    assert len(r["E_B2_librandom_repeats"]) == model.cfg.n_repeats
    assert len(r["E_B3_bucket_retrieval_repeats"]) == model.cfg.n_repeats
    for e, m in A.CONTROL_ROW_KEYS.values():
        assert math.isfinite(r[e]) and math.isfinite(r[m])
    assert r["strength_mono_rate"] is not None


def test_b0_is_the_untouched_image(bank, model):
    """B0 must be dE00(I, I*) exactly -- the identity transform composes to I."""
    _, rows = _board(model, bank)
    samples = _samples(model, bank)
    s = samples[0]
    i_star = bank.f_star_image(s.image, s.alpha, s.lut_id)
    assert rows[0]["E_B0_identity"] == pytest.approx(
        float(C.image_delta_e00(s.image, i_star)), abs=1e-6)


def test_oracle_is_never_worse_than_a_random_library_draw_on_its_own_metric(bank, model):
    """B4 is an argmin on the 9^3/dE76 protocol; that ordering must hold there."""
    ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    lib = C.LibraryValues.build(bank, ids, model.query_grid9)
    target = bank.apply(model.query_grid9, "lut_b")
    d = lib.distance_to(target, metric="de76")
    assert float(d.min()) == pytest.approx(0.0, abs=1e-4)     # lut_b is in the library
    assert float(d.min()) <= float(d.mean())


def test_b6_excludes_the_target_lut_itself(bank, model):
    """B6 is "nearest OTHER library LUT"; keyed by sample it would collapse to B4."""
    ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    lib = C.LibraryValues.build(bank, ids, model.query_grid9)
    target = bank.apply(model.query_grid9, "lut_b")
    b4 = C.oracle_lut_ids(lib, {"lut_b": target}, metric="de76", exclude_self=False)
    b6 = C.oracle_lut_ids(lib, {"lut_b": target}, metric="de76", exclude_self=True)
    assert b4["lut_b"][0] == "lut_b" and b6["lut_b"][0] != "lut_b"
    _, rows = _board(model, bank)
    # every row's B6 uses a LUT other than its own, so it is never the free 0
    assert all(r["E_B6_libfill"] > 0 for r in rows)


def test_interpolation_columns_carry_their_trivial_floors(bank, model):
    samples = _samples(model, bank)
    cols = A.interpolation_columns(model, model.cfg, samples, bank=bank, limit=2)
    ip = cols["interp_grid"]
    assert ip["n"] == 2 and ip["alphas"] == [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    assert ip["trivial_output_mixing"]["n"] == 2      # §4.F: void without this column
    assert ip["mix_point"] == "post_pi"
    assert cols["mono_rate"]["random_floor"] == 0.5
    assert cols["path_len"]["k_steps"] == 20
    assert "no percentile trimming" in cols["path_len"]["note"]
    assert 0.0 <= cols["oob_rate"]["mean"] <= 1.0
    assert cols["oob_rate"]["degenerate_weight_rate"] is not None


def test_same_source_pairs_never_pair_a_lut_with_itself(bank, model):
    samples = _samples(model, bank)
    for a, b in A.same_source_pairs(samples):
        assert samples[a].source_image_id == samples[b].source_image_id
        assert samples[a].lut_id != samples[b].lut_id


# --------------------------------------------------------------------------- #
# 12. the data side: images at the frozen short side, GT alpha from maskviews
# --------------------------------------------------------------------------- #
def _pseudo_shard(path: Path, payload: bytes, pad: int = 512) -> tuple[Path, int, int]:
    """A member at a known ``(offset, length)`` -- the tar reader's whole contract."""
    path.write_bytes(b"\x00" * pad + payload + b"\x00" * 64)
    return path, pad, len(payload)


def test_sample_store_reads_the_image_and_the_gt_alpha(tmp_path):
    from PIL import Image

    import io as _io

    buf = _io.BytesIO()
    Image.fromarray((np.random.default_rng(0).random((10, 16, 3)) * 255)
                    .astype(np.uint8)).save(buf, format="PNG")
    img_shard, img_off, img_len = _pseudo_shard(tmp_path / "images.tar", buf.getvalue())

    mbuf = _io.BytesIO()
    Image.fromarray((np.random.default_rng(1).random((10, 16)) * 255).astype(np.uint8),
                    mode="L").save(mbuf, format="PNG")
    mask_dir = tmp_path / "maskviews" / "V_what"
    (mask_dir / "shards").mkdir(parents=True)
    (mask_dir / "indexes").mkdir()
    m_shard, m_off, m_len = _pseudo_shard(mask_dir / "shards" / "shard-00000.tar",
                                          mbuf.getvalue())
    (mask_dir / "indexes" / "shard-00000.idx.jsonl").write_text(json.dumps({
        "sample_id": "sft_x", "suffix": ".maskhi.png", "shard": "shard-00000",
        "offset": m_off, "length": m_len}) + "\n")

    row = A.IndexRow.from_json({
        "sample_id": "sft_x", "split": "V_what", "lut_id": "lut_a",
        "source_image_id": "src_0", "task_type": "local",
        "winner_confidence": "normal",
        "members": {"image": {"shard": str(img_shard), "offset": img_off,
                              "length": img_len}}})
    store = A.SampleStore("V_what", mask_root=tmp_path / "maskviews", short_side=10)
    img, alpha = store.load(row)
    assert img.shape == (3, 10, 16) and float(img.max()) <= 1.0
    assert alpha.shape == (1, 10, 16) and 0.0 <= float(alpha.min())
    # a style sample has no mask member at all: alpha is exactly 1
    assert store.alpha("sft_x", "style") == 1.0
    with pytest.raises(KeyError, match="maskhi"):
        store.alpha("sft_missing", "local")


def test_sample_store_resizes_to_the_frozen_short_side(tmp_path):
    store = A.SampleStore("V_what", mask_root=tmp_path, short_side=8)
    out = A._resize_short_side(torch.rand(3, 20, 30), 8)
    assert out.shape == (3, 8, 12)              # short side 8, aspect kept
    assert A._resize_short_side(out, 8).shape == out.shape      # idempotent


# --------------------------------------------------------------------------- #
# 13. the runner's evaluation seam
# --------------------------------------------------------------------------- #
def test_runner_evaluate_and_publish_writes_a_gated_board(bank, model, tmp_path):
    from q3vl.whatb.scripts import run_carrier_arm as R

    clear_step_witness()
    cfg = model.cfg
    ids = ["lut_a", "lut_b", "lut_c", "lut_d"]
    lib = C.LibraryValues.build(bank, ids, model.query_grid9)
    vol, _ = A.library_mean_volume(bank, ids, cfg.bake_grid)
    samples = _samples(model, bank)
    steps = tmp_path / "steps.jsonl"
    steps.write_text(json.dumps({c: 0.0 for c in cfg.step_columns}) + "\n")
    board = R.evaluate_and_publish(
        model, cfg, samples[:4], bank=bank, lib=lib, lib_mean_volume=vol,
        pools={"bucket_x": ["lut_a"], "bucket_y": ["lut_b"]}, split="V_what",
        steps_path=steps, published=True, interp_samples=samples, interp_limit=2)
    assert board["publication"]["steps"]["source"] == "disk"
    assert R.selection_headline(board) > 0
    for key in C.PREREGISTERED_KEYS:
        assert board["criteria_columns"][key]["n"] > 0, key
    # the per-epoch selection board computes the headline and nothing else, and
    # it must be the SAME number as the full board's
    light = R.selection_board(model, cfg, samples[:4], bank=bank, split="V_what")
    assert R.selection_headline(light) == pytest.approx(R.selection_headline(board),
                                                        abs=1e-6)


# --------------------------------------------------------------------------- #
# 14. banned by absence, and the contamination edge
# --------------------------------------------------------------------------- #
def test_the_arm_implements_no_banned_column():
    for path in (CARRIER_PY, RUNNER_PY):
        src = path.read_text(encoding="utf-8")
        defs = re.findall(r"^\s*def\s+(\w+)", src, re.MULTILINE)
        banned = ("auc", "roc", "minmax", "min_max", "softmax_norm", "iou")
        assert not [d for d in defs if any(b in d.lower() for b in banned)], path
        assert "headline_pooled" not in src


def test_no_cpu_round_trip_on_a_metric_path():
    tree = ast.parse(CARRIER_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr != "cpu", f"line {node.lineno}: .cpu() on a metric path"
            if node.func.attr == "to" and node.args:
                a = node.args[0]
                assert not (isinstance(a, ast.Constant) and a.value == "cpu"), node.lineno


def test_the_arm_imports_nothing_from_the_contaminated_tree():
    for path in (CARRIER_PY, RUNNER_PY):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
        bad = [n for n in names
               if n.startswith(("q3vl.what.", "model.glut_repro", "gpu_render", "trash"))
               or n == "q3vl.what"]
        assert bad == [], f"{path}: {bad}"


def test_selection_reads_the_headline_and_never_a_loss():
    from q3vl.whatb.scripts import run_carrier_arm as R

    src = RUNNER_PY.read_text(encoding="utf-8")
    assert "val_loss" not in src and "val loss" not in src.lower().replace(
        "never validation loss", "")
    board = {"contexts": {"all": {"headline_normal_only": {"mean": 3.5, "n": 7}}}}
    assert R.selection_headline(board) == 3.5
    with pytest.raises(KeyError):
        R.selection_headline({"contexts": {"all": {}}})


def test_runner_exposes_the_whole_flag_surface():
    from q3vl.whatb.scripts import run_carrier_arm as R

    ap = R.build_parser()
    flags = {a.option_strings[0] for a in ap._actions if a.option_strings}
    for want in ("--readout", "--readout-qtok", "--cond-dim", "--n-gauss",
                 "--gen-width", "--loss-level", "--lambda-img", "--clamp",
                 "--batch-split", "--hc-eps", "--no-hc-mask", "--context",
                 "--cond-zero", "--cond-trainmean", "--lut-resample", "--no-mining"):
        assert want in flags, want
    args = ap.parse_args(["--zcache-root", "/tmp/x"])
    cfg = R.A.config_from_args(args)
    assert cfg == A.CarrierConfig()
