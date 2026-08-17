"""Identity-anchored gate: the five arithmetic identities of ``EPR-027:345-356``.

They are consequences of ``f_u(x) = x + u (f_theta(x) - x)``, not results, and
the arm has to print them next to its numbers so nobody reads a constructed 1.0
as a learned one.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whatb.gate import IdentityGate, identity_gate
from q3vl.whatb.glut import GlutParams, glut_forward


def _setup(batch: int = 2, p: int = 64):
    torch.manual_seed(1)
    params = GlutParams.identity(8, batch=batch, dtype=torch.float64)
    flat = params.affine_flat() + 0.1 * torch.randn(batch, 12 * 8 + 12, dtype=torch.float64)
    params = params.with_affine_flat(flat)
    x = torch.rand(batch, p, 3, dtype=torch.float64)
    y = glut_forward(x, params, clamp="none")
    return x, y


def test_g_endpoints() -> None:
    x, y = _setup()
    assert torch.allclose(identity_gate(x, y, 0.0, clamp=False), x, atol=1e-15)
    assert torch.allclose(identity_gate(x, y, 1.0, clamp=False), y, atol=1e-15)


def test_g1_residual_scales_linearly_in_u() -> None:
    """``f_u - y_u = u (f_theta - L)`` => ``L_rec(u) = u L_rec(1)``."""
    x, y = _setup()
    lut = (0.85 * x + 0.07).clamp(0, 1)
    l_rec_1 = (identity_gate(x, y, 1.0, clamp=False) - lut).abs().mean()
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_u = (1 - u) * x + u * lut
        got = (identity_gate(x, y, u, clamp=False) - y_u).abs().mean()
        assert got.item() == pytest.approx(u * l_rec_1.item(), rel=1e-12, abs=1e-15)


def test_g2_magnitude_is_exactly_u_times_the_ungated_magnitude() -> None:
    """``||f_u - x|| = u ||f_theta - x||`` -- so the u-monotonicity rate is 1 by construction."""
    x, y = _setup()
    base = (y - x).norm(dim=-1)
    prev = None
    for u in (0.0, 0.25, 0.5, 0.75, 1.0):
        mag = (identity_gate(x, y, u, clamp=False) - x).norm(dim=-1)
        assert torch.allclose(mag, u * base, atol=1e-14)
        if prev is not None:
            assert bool((mag >= prev).all())
        prev = mag


def test_g3_u_zero_is_a_no_op() -> None:
    x, y = _setup()
    assert torch.equal(identity_gate(x, y, torch.zeros(2, 64, 1, dtype=torch.float64), clamp=False), x)


def test_g5_clamping_before_the_gate_makes_the_oob_rate_degenerate() -> None:
    """``u in [0,1]`` on two in-gamut endpoints is a convex combination."""
    x, _ = _setup()
    y_clamped = glut_forward(x, GlutParams.identity(8, batch=2, dtype=torch.float64), clamp="two")
    u = torch.rand(2, 64, 1, dtype=torch.float64)
    out = identity_gate(x, y_clamped, u, clamp=False)
    assert bool(((out >= 0.0) & (out <= 1.0)).all())


def test_u_has_no_default() -> None:
    x, y = _setup()
    with pytest.raises(TypeError):
        identity_gate(x, y)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        IdentityGate()(x, y)  # type: ignore[call-arg]


@pytest.mark.parametrize("shape", [(), (2, 1, 1), (2, 64, 1), (2, 64, 3), (1, 1, 1)])
def test_u_broadcast_shapes(shape: tuple[int, ...]) -> None:
    x, y = _setup()
    u = torch.rand(*shape, dtype=torch.float64) if shape else torch.rand((), dtype=torch.float64)
    assert identity_gate(x, y, u, clamp=False).shape == y.shape


def test_u_that_cannot_broadcast_is_named() -> None:
    x, y = _setup()
    with pytest.raises(ValueError, match="does not broadcast"):
        identity_gate(x, y, torch.rand(5, 7, dtype=torch.float64))


def test_pixel_field_gate() -> None:
    """``u(p)`` at image resolution -- the spatial axis of EPR-027."""
    x = torch.rand(2, 8, 9, 3, dtype=torch.float64)
    y = (0.7 * x + 0.2).clamp(0, 1)
    alpha = torch.rand(2, 8, 9, 1, dtype=torch.float64)
    out = identity_gate(x, y, alpha, clamp=False)
    assert torch.allclose(out, (1 - alpha) * x + alpha * y, atol=1e-15)
    assert bool((out[alpha.expand_as(out) == 0] == x[alpha.expand_as(out) == 0]).all())


def test_gate_moves_u_onto_the_output_device_and_dtype() -> None:
    x = torch.rand(2, 5, 3, dtype=torch.float64)
    y = torch.rand(2, 5, 3, dtype=torch.float64)
    out = identity_gate(x, y, torch.tensor(0.5, dtype=torch.float32))
    assert out.dtype == torch.float64
    assert identity_gate(x.float(), y, 0.5).dtype == torch.float64  # anchor cast onto y


def test_mismatched_anchor_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="gate anchor"):
        identity_gate(torch.rand(2, 5, 3), torch.rand(2, 6, 3), 0.5)


def test_module_form_carries_the_flag_into_run_setup() -> None:
    gate = IdentityGate(clamp_after=True)
    assert list(gate.parameters()) == [] and gate.state_dict() == {}
    assert gate.config == {"gate_clamp": "after", "gate_range": [0.0, 1.0]}
    assert IdentityGate(clamp_after=False).config["gate_clamp"] == "before"
    x = torch.rand(2, 5, 3, dtype=torch.float64)
    y = torch.full_like(x, 3.0)
    assert gate(x, y, 1.0).max().item() == 1.0
    assert IdentityGate(clamp_after=False)(x, y, 1.0).max().item() == 3.0
