"""Carrier mechanics: shapes, clamp modes, dtype/device discipline, autograd.

Nothing here is a result; every assertion is either an arithmetic identity or a
contract the six arms are entitled to rely on.
"""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.whatb.glut import (
    EPS,
    LOG_2PI,
    GlutCarrier,
    GlutParams,
    glut_forward,
    n_params_glut,
    softplus_inverse,
    uniform_grid_positions,
)


def _random_params(batch: int = 3, n: int = 8, *, dtype: torch.dtype = torch.float64, seed: int = 0) -> GlutParams:
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=dtype)
    return GlutParams(
        mu=r(batch, n, 3),
        chol_diag=torch.full((batch, n, 3), softplus_inverse(0.2), dtype=dtype) + 0.1 * r(batch, n, 3),
        chol_off=0.05 * (r(batch, n, 3) - 0.5),
        opacity_logit=2.0 * (r(batch, n) - 0.5) + 2.0,
        m_local=torch.eye(3, dtype=dtype).expand(batch, n, 3, 3) + 0.2 * (r(batch, n, 3, 3) - 0.5),
        b_local=0.1 * (r(batch, n, 3) - 0.5),
        g_matrix=0.1 * (r(batch, 3, 3) - 0.5),
        g_bias=0.1 * (r(batch, 3) - 0.5),
    )


# --------------------------------------------------------------------------- #
# arithmetic of the parameterisation
# --------------------------------------------------------------------------- #
def test_param_count_is_22n_plus_12() -> None:
    assert n_params_glut(48) == 1068
    assert n_params_glut(32) == 716          # paper Table 9, the only published value
    assert n_params_glut(8) == 188
    assert n_params_glut(128) == 2828


def test_flat_roundtrip_and_layout() -> None:
    p = _random_params()
    flat = p.flat()
    assert flat.shape == (3, n_params_glut(8))
    back = GlutParams.from_flat(flat, 8)
    for name in ("mu", "chol_diag", "chol_off", "opacity_logit", "m_local", "b_local", "g_matrix", "g_bias"):
        assert torch.equal(getattr(back, name), getattr(p, name)), name
    assert p.affine_flat().shape == (3, 12 * 8 + 12)
    assert torch.equal(p.with_affine_flat(p.affine_flat()).flat(), flat)


def test_softplus_inverse_matches_epr025_constant() -> None:
    assert softplus_inverse(0.15) == pytest.approx(-1.8212, abs=1e-4)
    assert torch.nn.functional.softplus(torch.tensor(softplus_inverse(0.15))).item() == pytest.approx(0.15)


def test_uniform_grid_is_4x4x3_for_48() -> None:
    """EPR-025:372 spells the N=48 init ``uniform_grid_4x4x3``."""
    grid = uniform_grid_positions(48, dtype=torch.float64)
    assert grid.shape == (48, 3)
    assert sorted(len(torch.unique(grid[:, k])) for k in range(3)) == [3, 4, 4]
    assert len(torch.unique(grid[:, 0])) == 4 and len(torch.unique(grid[:, 2])) == 3
    assert grid.min() > 0.0 and grid.max() < 1.0
    assert torch.allclose(torch.unique(grid[:, 2]), torch.tensor([1 / 6, 0.5, 5 / 6], dtype=torch.float64))
    assert uniform_grid_positions(32).shape == (32, 3)
    assert len(torch.unique(uniform_grid_positions(64)[:, 0])) == 4


# --------------------------------------------------------------------------- #
# shapes / broadcasting
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "x_shape,out_shape",
    [((17, 3), (3, 17, 3)), ((3, 17, 3), (3, 17, 3)), ((3, 4, 5, 3), (3, 4, 5, 3)), ((1, 7, 3), (3, 7, 3))],
)
def test_shape_contract(x_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> None:
    p = _random_params(batch=3, n=6)
    x = torch.rand(*x_shape, dtype=torch.float64)
    assert glut_forward(x, p).shape == out_shape


def test_single_param_set_broadcasts_over_a_batch_of_queries() -> None:
    p = _random_params(batch=1, n=6)
    x = torch.rand(5, 11, 3, dtype=torch.float64)
    assert glut_forward(x, p).shape == (5, 11, 3)


def test_point_chunking_is_numerically_inert() -> None:
    """Points are independent, so chunking only re-associates float ops.

    Not bit-exact -- ``einsum`` picks different kernels for different block
    shapes -- but the spread is at the ulp level, and the *default* chunking is
    a pure function of ``(P, N)``, so two runs of the same configuration agree
    bit for bit.
    """
    p = _random_params(batch=2, n=5)
    x = torch.rand(2, 130, 3, dtype=torch.float64)
    full = glut_forward(x, p, point_chunk=None)
    assert torch.equal(glut_forward(x, p, point_chunk=None), full), "default chunking must be deterministic"
    worst = max(
        (glut_forward(x, p, point_chunk=c) - full).abs().max().item() for c in (1, 7, 129, 1000)
    )
    print(f"\nchunking spread over chunk in (1, 7, 129, 1000): {worst:.3e}")
    assert worst < 1e-12


# --------------------------------------------------------------------------- #
# clamp modes
# --------------------------------------------------------------------------- #
def test_clamp_modes() -> None:
    """``two`` (demo) and ``one`` (paper Eq.4/5) part company by construction.

    Global branch ``1.3`` with local branch ``-0.6``: ``two`` clamps the global
    branch to 1 first and lands on 0.4; ``one`` sums to 0.7 and keeps it.
    """
    p = _random_params(batch=2, n=5)
    fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
    fields["g_matrix"] = torch.zeros_like(fields["g_matrix"])
    fields["g_bias"] = torch.full_like(fields["g_bias"], 1.3)
    fields["m_local"] = torch.zeros_like(fields["m_local"])
    fields["b_local"] = torch.full_like(fields["b_local"], -0.6)
    p = GlutParams(**fields)
    torch.manual_seed(0)
    x = torch.rand(2, 40, 3, dtype=torch.float64)
    two, one, none = (glut_forward(x, p, clamp=m) for m in ("two", "one", "none"))
    # sum_i w_i = 1 - delta (eps in Eq.2's denominator), so local = -0.6 (1 - delta).
    _, aux = glut_forward(x, p, clamp="none", return_aux=True)
    w_sum = aux.weights.sum(-1, keepdim=True)
    assert torch.allclose(two, 1.0 - 0.6 * w_sum.expand_as(two), atol=1e-12)
    assert torch.allclose(one, 1.3 - 0.6 * w_sum.expand_as(one), atol=1e-12)
    assert (two - 0.4).abs().max() < 1e-3 and (one - 0.7).abs().max() < 1e-3
    assert torch.equal(one, none.clamp(0.0, 1.0))

    fields["g_bias"] = torch.full_like(fields["g_bias"], 3.0)
    hot = GlutParams(**fields)
    assert glut_forward(x, hot, clamp="none").max() > 1.0
    assert glut_forward(x, hot, clamp="two").max() <= 1.0


def test_unknown_clamp_mode_raises() -> None:
    with pytest.raises(ValueError, match="clamp must be one of"):
        glut_forward(torch.rand(4, 3), _random_params(), clamp="triple")  # type: ignore[arg-type]


def test_residual_flag_is_the_w_o_global_ablation() -> None:
    p = _random_params(batch=2, n=5)
    x = torch.rand(2, 20, 3, dtype=torch.float64)
    with_glob = glut_forward(x, p, clamp="none", residual=True)
    without = glut_forward(x, p, clamp="none", residual=False)
    glob = torch.einsum("bij,bpj->bpi", p.g_matrix, x) + p.g_bias.unsqueeze(1)
    assert torch.allclose(with_glob - without, glob, atol=1e-12)


# --------------------------------------------------------------------------- #
# Eq.1 / Eq.2 spot-checks against a scalar re-derivation
# --------------------------------------------------------------------------- #
def test_weights_are_the_normalised_opacity_weighted_pdf() -> None:
    """Recompute Eq.1-2 for one query with plain Python and compare."""
    p = _random_params(batch=1, n=4)
    x = torch.rand(1, 1, 3, dtype=torch.float64)
    _, aux = glut_forward(x, p, return_aux=True)
    infl = []
    for i in range(4):
        d = torch.nn.functional.softplus(p.chol_diag[0, i])
        o = p.chol_off[0, i]
        low = torch.tensor(
            [[d[0], 0.0, 0.0], [o[0], d[1], 0.0], [o[1], o[2], d[2]]], dtype=torch.float64
        )
        sigma = low @ low.T + EPS * torch.eye(3, dtype=torch.float64)
        diff = (x[0, 0] - p.mu[0, i]).reshape(3, 1)
        mahal = (diff.T @ torch.linalg.inv(sigma) @ diff).item()
        logpdf = -0.5 * (mahal + math.log(torch.linalg.det(sigma).item()) + 3 * LOG_2PI)
        infl.append(math.exp(logpdf) * torch.sigmoid(p.opacity_logit[0, i]).item())
    total = sum(infl)
    want = torch.tensor([v / (total + EPS) for v in infl], dtype=torch.float64)
    assert torch.allclose(aux.weights[0, 0], want, atol=1e-11)
    assert aux.influence_sum[0, 0].item() == pytest.approx(total, rel=1e-11)
    assert aux.weights.sum(-1)[0, 0].item() < 1.0  # eps in the denominator: sum = 1 - delta


def test_aux_columns_the_criteria_read() -> None:
    p = _random_params(batch=2, n=5)
    x = torch.rand(2, 33, 3, dtype=torch.float64)
    y, aux = glut_forward(x, p, clamp="two", return_aux=True)
    assert aux.weights.shape == (2, 33, 5)
    assert aux.influence_sum.shape == (2, 33)
    assert aux.pre_clamp.shape == (2, 33, 3)
    assert aux.opacity.shape == (2, 5) and aux.logdet.shape == (2, 5)
    assert aux.degenerate_precision.dtype == torch.bool
    assert not aux.degenerate_precision.any()
    assert torch.equal(y, aux.pre_clamp.clamp(0, 1))  # pre_clamp is post-global-clamp in "two"
    assert aux.oob_mask().shape == (2, 33)
    assert aux.degenerate_weight_mask(1e-3).shape == (2, 33)
    assert torch.allclose(aux.opacity, torch.sigmoid(p.opacity_logit))


# --------------------------------------------------------------------------- #
# device / dtype discipline (pitfall 2)
# --------------------------------------------------------------------------- #
def test_dtype_is_promoted_never_truncated() -> None:
    p = _random_params(batch=2, n=4, dtype=torch.float32)
    x = torch.rand(2, 9, 3, dtype=torch.float64)
    assert glut_forward(x, p).dtype == torch.float64


def test_low_precision_inputs_are_floored_at_float32() -> None:
    p = _random_params(batch=2, n=4, dtype=torch.float32).to(dtype=torch.bfloat16)
    x = torch.rand(2, 9, 3, dtype=torch.bfloat16)
    assert glut_forward(x, p).dtype == torch.float32, "exp(logpdf) is never evaluated in bf16"


def test_enclosing_autocast_cannot_pull_the_maths_into_bf16() -> None:
    """``einsum`` is on autocast's low-precision list; the carrier opts out.

    Without the opt-out the Mahalanobis form and ``exp(logpdf)`` would run in
    bf16 inside an ``autocast`` region even though the inputs were float32 --
    the EPR-022/MATTE lesson, one layer deeper.
    """
    p = _random_params(batch=2, n=6, dtype=torch.float32)
    x = torch.rand(2, 64, 3, dtype=torch.float32)
    plain = glut_forward(x, p)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        inside = glut_forward(x, p)
    assert inside.dtype == torch.float32
    assert torch.equal(inside, plain), "autocast changed the carrier's arithmetic"


def test_compute_dtype_override() -> None:
    p = _random_params(batch=1, n=4, dtype=torch.float64)
    x = torch.rand(6, 3, dtype=torch.float64)
    assert glut_forward(x, p, compute_dtype=torch.float32).dtype == torch.float32


def test_mixed_devices_raise_a_named_error() -> None:
    """The EPR-022/MATTE failure mode: refuse, do not guess."""
    p = _random_params(batch=1, n=4)
    bad = GlutParams(**{**{k: getattr(p, k) for k in p.__dataclass_fields__}})
    x = torch.rand(6, 3, dtype=torch.float64)

    class _Elsewhere(torch.Tensor):
        @property
        def device(self):  # type: ignore[override]
            return torch.device("meta")

    with pytest.raises(ValueError, match="more than one device"):
        glut_forward(x.as_subclass(_Elsewhere), bad)


def test_params_straddling_devices_are_rejected_at_construction() -> None:
    p = _random_params(batch=1, n=4)
    fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
    fields["b_local"] = fields["b_local"].to("meta")
    with pytest.raises(ValueError, match="straddle devices"):
        GlutParams(**fields)


def test_bad_shapes_are_rejected_by_name() -> None:
    p = _random_params(batch=1, n=4)
    fields = {k: getattr(p, k) for k in p.__dataclass_fields__}
    fields["g_matrix"] = fields["g_matrix"].reshape(1, 9)
    with pytest.raises(ValueError, match="GlutParams.g_matrix"):
        GlutParams(**fields)


# --------------------------------------------------------------------------- #
# autograd
# --------------------------------------------------------------------------- #
def test_gradients_reach_every_parameter_group() -> None:
    p = _random_params(batch=2, n=4)
    leaves = {k: getattr(p, k).clone().requires_grad_(True) for k in p.__dataclass_fields__}
    y = glut_forward(torch.rand(2, 12, 3, dtype=torch.float64), GlutParams(**leaves), clamp="none")
    y.pow(2).sum().backward()
    for name, t in leaves.items():
        assert t.grad is not None and torch.isfinite(t.grad).all(), name
        assert t.grad.abs().sum() > 0, f"{name} received no gradient"


def test_gradcheck_on_a_small_configuration() -> None:
    p = _random_params(batch=1, n=3)
    leaves = tuple(getattr(p, k).clone().requires_grad_(True) for k in p.__dataclass_fields__)
    x = torch.rand(1, 4, 3, dtype=torch.float64)
    fn = lambda *a: glut_forward(x, GlutParams(*a), clamp="none")
    assert torch.autograd.gradcheck(fn, leaves, eps=1e-6, atol=1e-6, rtol=1e-4)


# --------------------------------------------------------------------------- #
# module wrapper
# --------------------------------------------------------------------------- #
def test_carrier_module_has_no_parameters_and_a_non_persistent_constant() -> None:
    carrier = GlutCarrier(clamp="two")
    assert list(carrier.parameters()) == []
    assert carrier.state_dict() == {}, "eye3 must be persistent=False"
    assert "eye3" in dict(carrier.named_buffers())
    # ``clamp_grad`` joined the record with EPR-030; the default is "hard", i.e.
    # exactly the behaviour EPR-024..029 ran under.
    assert carrier.config == {"clamp": "two", "clamp_grad": "hard",
                              "residual": True, "eps": EPS}
    p = _random_params(batch=2, n=4)
    x = torch.rand(2, 7, 3, dtype=torch.float64)
    assert torch.equal(carrier(x, p), glut_forward(x, p, clamp="two"))
