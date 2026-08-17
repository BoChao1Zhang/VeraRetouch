"""Propositions 1 and 2 of ``docs/HANDOFF_whatb_2026-08-15.md`` section 2.3.

**Proposition 1 (affine-only linearisation)** -- ``EPR-025:272-285``.  If
``{mu_i, Sigma_i, o_i}`` are condition-independent then ``w_i(x)`` is too, so
``f_theta(x) = sum_i w_i(x)(M_i x + b_i) + Gx + g`` is *linear* in
``theta_gen = ({M_i}, {b_i}, G, g)`` and therefore, before the clamp and point
by point,

    f_{(1-a) theta_a + a theta_b}(x) == (1-a) f_{theta_a}(x) + a f_{theta_b}(x)

i.e. the parameter-space path and the function-space path coincide.  Tolerance
1e-5 by task card; measured deviation is reported.

Every proposition test here carries its **negative control**: the same identity
under Full Generation, where ``mu / Sigma / o`` do move with the condition and
the equality must fail by a wide margin.  A proposition test that also passes
where the proposition does not hold is measuring nothing.

**Proposition 2 (exact representability of the identity)** -- ``EPR-025:287-291``,
``EPR-024:583-584``.  With ``eps = 0`` the weights sum to 1, so ``M_i = I``,
``b_i = 0``, ``G = 0``, ``g = 0`` gives ``f = id`` exactly.  With the frozen
``eps = 1e-6`` they sum to ``1 - delta(x)``, ``delta = eps / (sum_j p_j o_j + eps)``,
so ``f(x) = (1 - delta(x)) x`` -- which is why the degenerate-weight rate
``Pr[sum_j p_j o_j < tau]`` is a per-board column and not a footnote.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whatb.gate import identity_gate
from q3vl.whatb.generator import CGLUTGenerator, SegColorProjection
from q3vl.whatb.glut import EPS, GlutParams, glut_forward

TOL_PROP1 = 1e-5           # task card
ALPHAS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)   # GLUT App B.3 / Table 7 grid


def _conditions(gen: CGLUTGenerator, *, batch: int = 4, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    d = gen.cond_dim
    return (torch.randn(batch, d, generator=g, dtype=torch.float64),
            torch.randn(batch, d, generator=g, dtype=torch.float64))


def _fresh(mode: str, *, n: int = 12, seed: int = 3) -> CGLUTGenerator:
    """A generator whose heads actually respond to the condition.

    Step 0 of a zero-initialised affine-only head is the identity, which
    satisfies proposition 1 for the wrong reason; and a default-initialised
    full head barely moves ``mu / Sigma / o``, which would make the negative
    control weak.  Both are perturbed to a trained-like scale.
    """
    torch.manual_seed(seed)
    gen = CGLUTGenerator(cond_dim=16, hidden=32, n_gauss=n, mode=mode).double()
    with torch.no_grad():
        for head in (gen.head_color, gen.head_global):
            head[-1].weight.normal_(0.0, 0.05)
            head[-1].bias.normal_(0.0, 0.05)
        for head in (gen.head_mu, gen.head_cov, gen.head_opacity):
            if head is not None:
                head[-1].weight.normal_(0.0, 0.5)
                head[-1].bias.normal_(0.0, 0.5)
    return gen


# --------------------------------------------------------------------------- #
# proposition 1
# --------------------------------------------------------------------------- #
def test_proposition_1_affine_only_parameter_path_equals_function_path() -> None:
    gen = _fresh("affine_only")
    za, zb = _conditions(gen)
    pa, pb = gen(za), gen(zb)
    x = torch.rand(4, 64, 3, dtype=torch.float64)

    fa = glut_forward(x, pa, clamp="none")
    fb = glut_forward(x, pb, clamp="none")
    theta_a, theta_b = pa.affine_flat(), pb.affine_flat()

    worst = 0.0
    per_alpha: list[tuple[float, float]] = []
    for a in ALPHAS:
        mixed = pa.with_affine_flat((1 - a) * theta_a + a * theta_b)
        f_par = glut_forward(x, mixed, clamp="none")
        f_fun = (1 - a) * fa + a * fb
        dev = (f_par - f_fun).abs().max().item()
        per_alpha.append((a, dev))
        worst = max(worst, dev)
    print("\nproposition 1 (affine-only), max |f^par - f^fun| per alpha:")
    for a, dev in per_alpha:
        print(f"  alpha={a:.1f}  {dev:.3e}")
    print(f"  worst = {worst:.3e}   tolerance {TOL_PROP1:.0e}")
    assert worst < TOL_PROP1


def test_proposition_1_negative_control_full_generation_breaks_it() -> None:
    """Full Generation moves ``mu / Sigma / o`` too, so ``w_i`` moves: not linear."""
    gen = _fresh("full")
    za, zb = _conditions(gen)
    pa, pb = gen(za), gen(zb)
    x = torch.rand(4, 64, 3, dtype=torch.float64)
    fa = glut_forward(x, pa, clamp="none")
    fb = glut_forward(x, pb, clamp="none")

    worst = 0.0
    for a in (0.2, 0.4, 0.6, 0.8):
        full_mix = GlutParams.from_flat((1 - a) * pa.flat() + a * pb.flat(), gen.n_gauss)
        dev = (glut_forward(x, full_mix, clamp="none") - ((1 - a) * fa + a * fb)).abs().max().item()
        worst = max(worst, dev)
    print(f"\nproposition 1 negative control (full generation): worst deviation {worst:.3e}")
    assert worst > 100 * TOL_PROP1, "the affine-only test would be vacuous if this also passed"


def test_proposition_1_negative_control_geometry_alone_breaks_it() -> None:
    """Sharpest form: two parameter sets differing **only** in ``mu``.

    ``{M, b, G, g}`` are identical, so the function-space mixture is a single
    fixed function; the parameter-space mixture moves the partition and cannot
    equal it.  No initialisation scale to argue about.
    """
    n = 12
    base = GlutParams.identity(n, batch=1, dtype=torch.float64)
    theta = torch.randn(1, 12 * n + 12, generator=torch.Generator().manual_seed(4), dtype=torch.float64) * 0.4
    pa = base.with_affine_flat(theta)
    fields = {k: getattr(pa, k) for k in pa.__dataclass_fields__}
    fields["mu"] = 1.0 - pa.mu                      # mirrored grid, same affine payload
    pb = GlutParams(**fields)
    x = torch.rand(1, 128, 3, dtype=torch.float64)
    fa = glut_forward(x, pa, clamp="none")
    fb = glut_forward(x, pb, clamp="none")
    mixed = GlutParams.from_flat(0.5 * (pa.flat() + pb.flat()), n)
    dev = (glut_forward(x, mixed, clamp="none") - 0.5 * (fa + fb)).abs().max().item()
    print(f"proposition 1 negative control (geometry only): deviation at alpha=0.5 = {dev:.3e}")
    assert dev > 100 * TOL_PROP1


def test_proposition_1_holds_for_the_shared_geometry_of_any_two_parameter_sets() -> None:
    """The premise is "geometry is shared", not "the generator produced it"."""
    torch.manual_seed(11)
    base = GlutParams.identity(10, batch=2, dtype=torch.float64)
    ta = torch.randn(2, 12 * 10 + 12, dtype=torch.float64) * 0.3
    tb = torch.randn(2, 12 * 10 + 12, dtype=torch.float64) * 0.3
    x = torch.rand(2, 40, 3, dtype=torch.float64)
    fa = glut_forward(x, base.with_affine_flat(ta), clamp="none")
    fb = glut_forward(x, base.with_affine_flat(tb), clamp="none")
    for a in ALPHAS:
        mixed = glut_forward(x, base.with_affine_flat((1 - a) * ta + a * tb), clamp="none")
        assert (mixed - ((1 - a) * fa + a * fb)).abs().max().item() < TOL_PROP1


def test_proposition_1_is_about_the_pre_clamp_value() -> None:
    """Stated "clamp前逐点": the clamp is not linear and is excluded on purpose."""
    torch.manual_seed(5)
    base = GlutParams.identity(8, batch=1, dtype=torch.float64)
    ta = torch.zeros(1, 12 * 8 + 12, dtype=torch.float64)
    tb = torch.zeros(1, 12 * 8 + 12, dtype=torch.float64)
    tb[:, -3:] = 2.0                      # g = (2,2,2): way out of gamut at alpha=1
    for i in range(8):
        ta[:, i * 12 + 0] = ta[:, i * 12 + 4] = ta[:, i * 12 + 8] = 1.0
        tb[:, i * 12 + 0] = tb[:, i * 12 + 4] = tb[:, i * 12 + 8] = 1.0
    x = torch.rand(1, 16, 3, dtype=torch.float64)
    fa = glut_forward(x, base.with_affine_flat(ta), clamp="two")
    fb = glut_forward(x, base.with_affine_flat(tb), clamp="two")
    mixed = glut_forward(x, base.with_affine_flat(0.5 * (ta + tb)), clamp="two")
    assert (mixed - 0.5 * (fa + fb)).abs().max().item() > 1e-2


# --------------------------------------------------------------------------- #
# proposition 2
# --------------------------------------------------------------------------- #
def test_proposition_2_identity_is_exact_at_eps_zero() -> None:
    params = GlutParams.identity(48, batch=2, dtype=torch.float64)
    x = torch.rand(2, 512, 3, dtype=torch.float64)
    y = glut_forward(x, params, clamp="none", eps=0.0)
    dev = (y - x).abs().max().item()
    print(f"\nproposition 2 at eps=0: max |f(x) - x| = {dev:.3e}")
    assert dev < 1e-12


def test_proposition_2_at_frozen_eps_is_the_predicted_one_minus_delta() -> None:
    """``f(x) = (1 - delta(x)) x`` with ``delta = eps / (sum_j p_j o_j + eps)``."""
    params = GlutParams.identity(48, batch=2, dtype=torch.float64)
    x = torch.rand(2, 512, 3, dtype=torch.float64)
    y, aux = glut_forward(x, params, clamp="none", eps=EPS, return_aux=True)
    delta = EPS / (aux.influence_sum + EPS)
    predicted = (1.0 - delta).unsqueeze(-1) * x
    assert (y - predicted).abs().max().item() < 1e-12
    print(
        f"\nproposition 2 at eps=1e-6 on a 48-Gaussian grid: "
        f"max delta = {delta.max().item():.3e}, max |f(x) - x| = {(y - x).abs().max().item():.3e}"
    )
    assert delta.max().item() < 1e-3, "grid geometry keeps the corner degeneracy small here"


def test_proposition_2_degenerate_weight_rate_is_observable() -> None:
    """The column the frozen criteria demand: ``Pr[sum_j p_j o_j < tau]``.

    On the App A.1 grid init the column is identically 0 -- the grid covers the
    cube.  It becomes non-trivial exactly when a region of colour space has no
    Gaussian over it, so the test builds that: all 48 means crowded into the
    black corner, ``sigma = 0.15`` unchanged.  The far corner then falls off the
    partition and ``f(x) = (1 - delta) x`` shrinks visibly toward black.
    """
    grid = GlutParams.identity(48, batch=1, dtype=torch.float64)
    x = torch.rand(1, 4096, 3, dtype=torch.float64)
    _, aux_grid = glut_forward(x, grid, clamp="none", return_aux=True)
    rate_grid = aux_grid.degenerate_weight_mask(1e-3).double().mean().item()

    fields = {k: getattr(grid, k) for k in grid.__dataclass_fields__}
    fields["mu"] = grid.mu * 0.1
    crowded = GlutParams(**fields)
    y, aux = glut_forward(x, crowded, clamp="none", return_aux=True)
    rate = aux.degenerate_weight_mask(1e-3).double().mean().item()
    shrink = (x - y).abs().max().item()
    print(
        f"\ndegenerate-weight rate(tau=1e-3): grid init = {rate_grid:.3f}, "
        f"means crowded into the black corner = {rate:.3f}; max shrink |x - f(x)| = {shrink:.3e}"
    )
    assert rate_grid == 0.0
    assert rate > 0.0 and shrink > 1e-6


def test_demo_precision_fallback_threshold_for_isotropic_covariance() -> None:
    """demo :496 -- ``|det Sigma| < eps`` replaces the precision by the identity.

    With ``Sigma = (sigma^2 + eps) I`` that fires below
    ``sigma = sqrt(1e-2 - 1e-6) = 0.09999...``.  The frozen App A.1 init
    (``sigma = 0.15``, ``det = 1.14e-5``) sits above it, but a trained
    covariance head can walk under, and when it does the Gaussian's shape
    parameters stop receiving gradient (``det.clamp_min(eps)`` zeroes the other
    path too).  ``GlutAux.degenerate_precision`` is how a run notices.
    """
    x = torch.rand(1, 64, 3, dtype=torch.float64)
    seen: dict[float, bool] = {}
    for sigma in (0.02, 0.09, 0.0999, 0.1001, 0.15, 0.3):
        p = GlutParams.identity(8, batch=1, dtype=torch.float64, sigma=sigma)
        _, aux = glut_forward(x, p, clamp="none", return_aux=True)
        seen[sigma] = bool(aux.degenerate_precision.any())
    print("\ndemo :496 identity-precision fallback by isotropic sigma:")
    for sigma, fired in seen.items():
        print(f"  sigma={sigma:<7} fallback fired = {fired}")
    assert seen[0.02] and seen[0.09] and seen[0.0999]
    assert not seen[0.1001] and not seen[0.15] and not seen[0.3]

    frozen = GlutParams.identity(48, batch=1, dtype=torch.float64, sigma=0.15)
    _, aux = glut_forward(x, frozen, clamp="none", return_aux=True)
    assert not aux.degenerate_precision.any(), "the frozen init must not sit on the cliff"


def test_proposition_2_via_the_generator_zero_init_step_zero() -> None:
    """EPR-025's step 0 *is* proposition 2: zero-init heads + ``M = I + dM``."""
    torch.manual_seed(0)
    gen = CGLUTGenerator(cond_dim=16, hidden=32, n_gauss=24, mode="affine_only").double()
    params = gen(torch.randn(3, 16, dtype=torch.float64))
    assert torch.equal(params.m_local, torch.eye(3, dtype=torch.float64).expand(3, 24, 3, 3))
    assert torch.count_nonzero(params.b_local) == 0
    assert torch.count_nonzero(params.g_matrix) == 0 and torch.count_nonzero(params.g_bias) == 0
    x = torch.rand(3, 256, 3, dtype=torch.float64)
    dev = (glut_forward(x, params, clamp="none", eps=0.0) - x).abs().max().item()
    print(f"\nstep0 of the affine-only generator: max |f(x) - x| = {dev:.3e} (eps=0)")
    assert dev < 1e-12
    # `step0_maxabs_f_minus_id` (EPR-024:612) at the frozen eps, for the board:
    print(f"  at eps=1e-6: {(glut_forward(x, params, clamp='none') - x).abs().max().item():.3e}")


# --------------------------------------------------------------------------- #
# proposition 3's mechanism (the piece that lives in this batch)
# --------------------------------------------------------------------------- #
def test_proposition_3_pixelwise_parameter_interpolation_reproduces_the_generating_law() -> None:
    """``theta(p) = (1-alpha(p)) theta_id + alpha(p) theta_L`` gives ``F*`` exactly.

    ``theta_L`` here is any affine transform the carrier represents exactly (a
    single global affine); the point under test is the *interpolation*
    machinery, not whether a given ``.cube`` is representable -- that one is an
    empirical fitting column, as HANDOFF section 2.3 says.
    """
    torch.manual_seed(2)
    n = 16
    base = GlutParams.identity(n, batch=1, dtype=torch.float64)
    theta_id = base.affine_flat()
    theta_l = theta_id.clone()
    lut_m = torch.tensor([[0.9, 0.05, 0.0], [0.0, 1.1, 0.02], [0.03, 0.0, 0.8]], dtype=torch.float64)
    lut_b = torch.tensor([0.02, -0.01, 0.05], dtype=torch.float64)
    for i in range(n):
        theta_l[:, i * 12 : i * 12 + 9] = lut_m.reshape(9)
        theta_l[:, i * 12 + 9 : i * 12 + 12] = lut_b
    theta_l[:, -12:-3] = 0.0
    theta_l[:, -3:] = 0.0

    x = torch.rand(1, 128, 3, dtype=torch.float64)
    lut_x = torch.einsum("ij,bpj->bpi", lut_m, x) + lut_b
    for alpha in ALPHAS:
        mixed = base.with_affine_flat((1 - alpha) * theta_id + alpha * theta_l)
        f = glut_forward(x, mixed, clamp="none", eps=0.0)
        star = (1 - alpha) * x + alpha * lut_x
        assert (f - star).abs().max().item() < 1e-12, alpha


def test_gate_reaches_the_same_place_from_outside(  # proposition 3 via EPR-027's G4
) -> None:
    """``f_u`` with ``u = alpha`` and ``f_theta = L`` is the same ``F*`` -- identity G4."""
    x = torch.rand(2, 64, 3, dtype=torch.float64)
    lut_x = (0.8 * x + 0.1).clamp(0, 1)
    alpha = torch.rand(2, 64, 1, dtype=torch.float64)
    got = identity_gate(x, lut_x, alpha, clamp=False)
    star = (1 - alpha) * x + alpha * lut_x
    assert torch.allclose(got, star, atol=1e-15)


def test_projection_is_a_pure_reparameterisation_of_the_condition() -> None:
    """``pi`` stands where ``e_l`` stood: (B,2560) -> (B,d), nothing else."""
    pi = SegColorProjection(cond_dim=64).double()
    z = torch.randn(5, 2560, dtype=torch.float64)
    assert pi(z).shape == (5, 64)
    with pytest.raises(ValueError, match=r"pi expects"):
        pi(torch.randn(5, 1280, dtype=torch.float64))
