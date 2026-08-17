"""Generator structure.  The load-bearing test is the paper Table 2 arithmetic.

GLUT App A.2 names only two output widths (``mu`` -> ``3N``, global -> 12) and
says the other heads "follow a similar structure with adjusted output
dimensions".  ``6N`` / ``N`` / ``12N`` are therefore an inference from the
section 3.1 parameter list -- and the inference is falsifiable: with the
projection swapped back for CGLUT's own ``E in R^{225x64}`` lookup table
(14,400 parameters) the count has to land on the paper's reported 98K / 338K.
It does, to the digit.  If a head were read wrong, it would not.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from q3vl.whatb.generator import (
    SEG_COLOR_HIDDEN_DIM,
    CGLUTGenerator,
    SegColorProjection,
    SharedGeometry,
    generator_param_count,
)
from q3vl.whatb.glut import GlutParams, glut_forward, n_params_glut, softplus_inverse

LOOKUP_225x64 = 225 * 64  # CGLUT's condition embedding matrix E, 14,400


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# --------------------------------------------------------------------------- #
# the arithmetic cross-check against the published table
# --------------------------------------------------------------------------- #
def test_paper_table2_param_count() -> None:
    small = CGLUTGenerator(cond_dim=64, hidden=64, n_gauss=32)
    large = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=64)
    assert _count(small) == 83_980
    assert _count(large) == 323_596
    assert _count(small) + LOOKUP_225x64 == 98_380         # paper Table 2: ~98K
    assert _count(large) + LOOKUP_225x64 == 337_996        # paper Table 2: ~338K
    print(
        f"\nCGLUT-32(Small) {_count(small)} + {LOOKUP_225x64} = {_count(small) + LOOKUP_225x64} (paper 98K)"
        f"\nCGLUT-64(Large) {_count(large)} + {LOOKUP_225x64} = {_count(large) + LOOKUP_225x64} (paper 338K)"
    )


def test_closed_form_count_matches_the_built_module() -> None:
    for d, h, n, mode in [
        (64, 64, 32, "full"), (64, 128, 64, "full"), (64, 128, 48, "full"),
        (64, 128, 48, "affine_only"), (256, 128, 48, "full"), (32, 64, 32, "affine_only"),
    ]:
        gen = CGLUTGenerator(cond_dim=d, hidden=h, n_gauss=n, mode=mode)
        expect = generator_param_count(cond_dim=d, hidden=h, n_gauss=n, mode=mode)
        shared = 0 if gen.shared_geometry is None else _count(gen.shared_geometry)
        assert _count(gen) - shared == expect, (d, h, n, mode)


def test_epr024_and_epr025_headline_configurations() -> None:
    """The numbers ``EPR-024:466-476`` and ``EPR-025:421-427`` print."""
    pi = SegColorProjection(cond_dim=64)
    gen24 = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=48, mode="full")
    assert _count(pi) == 169_024                     # LayerNorm 5,120 + Linear 163,904
    assert _count(gen24) == 278_188
    assert _count(pi) + _count(gen24) == 447_212

    gen25 = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=48, mode="affine_only")
    assert _count(gen25.shared_geometry) == 48 * 3 * 3 + 48        # 480
    assert _count(gen25) - 480 == 107_328 + 18_060 + 41_344        # colour + global + encoder
    assert _count(pi) + _count(gen25) == 336_236

    gen32 = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=32, mode="full")
    assert _count(pi) + _count(gen32) == 401_804     # the forced N=32 companion row


def test_theta_dim_is_22n_plus_12_or_12n_plus_12() -> None:
    assert CGLUTGenerator(n_gauss=48, mode="full").theta_dim == n_params_glut(48) == 1068
    assert CGLUTGenerator(n_gauss=48, mode="affine_only").theta_dim == 12 * 48 + 12 == 588
    assert CGLUTGenerator(n_gauss=32, mode="full").theta_dim == 716


# --------------------------------------------------------------------------- #
# structure
# --------------------------------------------------------------------------- #
def test_layer_counts_follow_appendix_a2() -> None:
    gen = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=48, mode="full")
    linear = lambda m: [l for l in m if isinstance(l, nn.Linear)]
    assert len(linear(gen.encoder)) == 3                       # 3-layer shared encoder
    assert isinstance(gen.encoder[-1], nn.ReLU)                # ReLU after every encoder layer
    assert len(linear(gen.head_mu)) == 2 and linear(gen.head_mu)[-1].out_features == 3 * 48
    assert len(linear(gen.head_cov)) == 2 and linear(gen.head_cov)[-1].out_features == 6 * 48
    assert len(linear(gen.head_opacity)) == 2 and linear(gen.head_opacity)[-1].out_features == 48
    assert len(linear(gen.head_color)) == 3                    # the one head App A.2 calls 3-layer
    assert linear(gen.head_color)[-1].out_features == 12 * 48
    assert len(linear(gen.head_global)) == 2 and linear(gen.head_global)[-1].out_features == 12
    assert sum(isinstance(l, nn.ReLU) for l in gen.head_color) == 2


def test_forward_shapes_and_activations() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=5, mode="full")
    p = gen(torch.randn(4, 8))
    assert isinstance(p, GlutParams)
    assert p.batch_size == 4 and p.n_gauss == 5
    assert p.mu.shape == (4, 5, 3) and p.chol_diag.shape == (4, 5, 3)
    assert p.opacity_logit.shape == (4, 5) and p.m_local.shape == (4, 5, 3, 3)
    assert p.g_matrix.shape == (4, 3, 3) and p.g_bias.shape == (4, 3)
    # raw storage: activations belong to the carrier, applied exactly once
    assert p.opacity_logit.abs().max() > 0
    assert glut_forward(torch.rand(4, 9, 3), p).shape == (4, 9, 3)


def test_condition_shape_is_checked() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=4)
    with pytest.raises(ValueError, match=r"condition must be \(B, 8\)"):
        gen(torch.randn(4, 9))
    with pytest.raises(ValueError, match="condition must be"):
        gen(torch.randn(2, 3, 8))


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        CGLUTGenerator(mode="shared_geometry")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# affine-only mode: EPR-025's structural guarantees
# --------------------------------------------------------------------------- #
def test_affine_only_geometry_is_condition_independent() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="affine_only")
    pa, pb = gen(torch.randn(3, 8)), gen(torch.randn(3, 8))
    for name in ("mu", "chol_diag", "chol_off", "opacity_logit"):
        assert torch.equal(getattr(pa, name), getattr(pb, name)), name
    assert gen.head_mu is None and gen.head_cov is None and gen.head_opacity is None


def test_shared_geometry_forward_takes_no_condition() -> None:
    """Type-level proof the sharing is wired (EPR-025 section 3.4-1)."""
    sg = SharedGeometry(6)
    import inspect

    assert list(inspect.signature(sg.forward).parameters) == []
    prec, logdet, opacity, degen = sg()
    assert prec.shape == (6, 3, 3) and logdet.shape == (6,) and opacity.shape == (6,)
    assert degen.dtype == torch.bool and not degen.any()
    assert torch.allclose(opacity, torch.full((6,), torch.sigmoid(torch.tensor(4.0)).item()))


def test_shared_geometry_init_values_are_the_epr025_constants() -> None:
    sg = SharedGeometry(48)
    assert torch.allclose(sg.chol_diag, torch.full((48, 3), softplus_inverse(0.15)))
    assert sg.chol_diag[0, 0].item() == pytest.approx(-1.8212, abs=1e-4)
    assert torch.count_nonzero(sg.chol_off) == 0
    assert torch.allclose(sg.opacity_logit, torch.full((48,), 4.0))
    assert torch.sigmoid(sg.opacity_logit[0]).item() == pytest.approx(0.98201, abs=1e-5)
    assert sg.mu.shape == (48, 3) and sg.mu.min() > 0 and sg.mu.max() < 1


def test_affine_only_step0_is_the_identity_transform() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="affine_only").double()
    p = gen(torch.randn(3, 8, dtype=torch.float64))
    assert torch.equal(p.m_local, torch.eye(3, dtype=torch.float64).expand(3, 6, 3, 3))
    assert torch.count_nonzero(p.b_local) + torch.count_nonzero(p.g_matrix) + torch.count_nonzero(p.g_bias) == 0


def test_full_mode_defaults_are_pytorch_default_init_no_anchoring() -> None:
    """Ruling 11.1-1: EPR-024 takes PyTorch default init, no zero-init, no ``I + dM``."""
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="full")
    assert gen.m_residual is False and gen.zero_init_last is False
    assert gen.head_color[-1].weight.abs().sum() > 0
    p = gen(torch.randn(3, 8))
    assert not torch.allclose(p.m_local, torch.eye(3).expand(3, 6, 3, 3))


def test_shared_geometry_terms_are_reusable_and_agree_with_the_full_path() -> None:
    """The once-per-step reuse that makes proposition 1 cheap must be exact."""
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="affine_only").double()
    with torch.no_grad():
        gen.head_color[-1].weight.normal_(0, 0.05)
        gen.head_global[-1].bias.normal_(0, 0.05)
    z = torch.randn(3, 8, dtype=torch.float64)
    p = gen(z)
    x = torch.rand(3, 20, 3, dtype=torch.float64)
    shared = gen.shared_geometry_terms()
    assert shared is not None
    assert torch.equal(glut_forward(x, p, clamp="two", _geometry=shared), glut_forward(x, p, clamp="two"))
    assert CGLUTGenerator(mode="full").shared_geometry_terms() is None


def test_param_groups_put_shared_geometry_on_a_tenth_of_the_lr() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="affine_only")
    groups = gen.param_groups(1e-3)
    assert [g["name"] for g in groups] == ["generator", "shared_geometry"]
    assert groups[0]["lr"] == 1e-3 and groups[1]["lr"] == pytest.approx(1e-4)
    assert len(groups[1]["params"]) == 4
    assert sum(p.numel() for g in groups for p in g["params"]) == _count(gen)
    ablation = gen.param_groups(1e-3, geometry_lr_scale=1.0)
    assert ablation[1]["lr"] == 1e-3

    full = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode="full")
    assert [g["name"] for g in full.param_groups(1e-3)] == ["generator"]


# --------------------------------------------------------------------------- #
# discipline
# --------------------------------------------------------------------------- #
def test_constants_are_non_persistent_buffers_not_forward_time_tensors() -> None:
    for mode in ("full", "affine_only"):
        gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=6, mode=mode)
        buffers = dict(gen.named_buffers())
        assert "eye3" in buffers
        assert not any(k.endswith("eye3") for k in gen.state_dict()), mode


def test_condition_is_cast_onto_the_module_not_the_other_way_round() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=4).double()
    p = gen(torch.randn(2, 8, dtype=torch.float32))
    assert p.mu.dtype == torch.float64
    pi = SegColorProjection(cond_dim=8).float()
    assert pi(torch.randn(2, SEG_COLOR_HIDDEN_DIM, dtype=torch.float64)).dtype == torch.float32


def test_config_dict_is_what_run_setup_records() -> None:
    gen = CGLUTGenerator(cond_dim=64, hidden=128, n_gauss=48, mode="affine_only")
    cfg = gen.config
    assert cfg["cond_dim"] == 64 and cfg["hidden"] == 128 and cfg["n_gauss"] == 48
    assert cfg["mode"] == "affine_only" and cfg["theta_dim"] == 588
    assert cfg["m_residual"] is True and cfg["zero_init_last"] is True
    assert cfg["shared_sigma"] == 0.15 and cfg["shared_opacity_logit"] == 4.0
    assert cfg["n_params"] == _count(gen)


def test_gradients_flow_from_the_carrier_back_to_the_condition() -> None:
    gen = CGLUTGenerator(cond_dim=8, hidden=16, n_gauss=5, mode="full").double()
    z = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    glut_forward(torch.rand(2, 12, 3, dtype=torch.float64), gen(z), clamp="none").pow(2).sum().backward()
    assert z.grad is not None and z.grad.abs().sum() > 0
    for name, p in gen.named_parameters():
        assert p.grad is not None, name
