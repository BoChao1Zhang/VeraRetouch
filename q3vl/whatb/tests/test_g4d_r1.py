"""EPR-028 R1 gate G0 -- CPU only, no GPU process is started.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=/home/bc/VeraRetouch \\
    /home/bc/envs/q3vl_sft/bin/python -m pytest q3vl/whatb/tests/test_g4d_r1.py -q

Spec: ``experiments/prs/EPR-028_4d-gaussian-conditional-slice/PROPOSAL_R1.md``
§9's G0 row -- "float64 条件参数化对拍、log-domain、gradcheck、axis-order 单测;
数学逐点对拍通过; step 0 严格 identity; A3(beta=0) 与 A2 逐位相同".

What each block pins
--------------------
1. :func:`~q3vl.whatb.arms.g4d.cond_from_sigma4` /
   :func:`~q3vl.whatb.arms.g4d.sigma4_from_cond` are exact inverses on random
   SPD 4D covariances (R1 §4.1's expressivity claim);
2. the log-domain forward equals a naive ordinary-domain float64 implementation
   on well-conditioned parameters, and survives parameters on which the naive
   one returns ``nan`` / zeros (R1 §4.3);
3. the fourth axis really is ``s``: it moves the conditional mean and the gate
   and never touches ``C``;
4. ``gradcheck`` through ``L_C``, ``beta``, ``tau``, ``mu_x``, ``mu_s``,
   ``o_logit``;
5. step 0 is the identity on A1 / A2 / A3;
6. A2 and A3 are **bit-identical** at step 0 (R1 §3);
7. the R1 §3 parameter table, including the ``A3, N=39 -> 1065`` control row;
8. a degenerate ``L_C`` raises instead of falling back to the identity (R1 §7);
9. A1's ``s`` endpoints are bit-exact;
10. ``R_line`` is exactly zero on A1, whose path is linear in ``s`` by construction.
"""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.whatb.arms import g4d
from q3vl.whatb.glut import EPS, LOG_2PI, glut_forward, softplus_inverse

DOUBLE = torch.float64


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _random_spd4(n: int, *, seed: int = 0, dtype=DOUBLE) -> torch.Tensor:
    """``(n, 4, 4)`` SPD matrices with a healthy spread of conditioning."""
    gen = torch.Generator().manual_seed(seed)
    a = torch.randn(n, 4, 4, generator=gen, dtype=dtype)
    scale = torch.exp(torch.randn(n, 4, 1, generator=gen, dtype=dtype) * 0.8)
    a = a * scale
    eye = torch.eye(4, dtype=dtype)
    return a @ a.transpose(-1, -2) + 0.05 * eye


def _cond_params(b: int = 2, n: int = 4, *, mode: str = "A3", dtype=DOUBLE,
                 seed: int = 0, sigma: float = 0.15) -> g4d.G4DParams:
    """A well-conditioned conditional parameter set (not from the generator)."""
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, dtype=dtype)
    return g4d.G4DParams(
        mode=mode,
        mu_x=torch.rand(b, n, 3, generator=gen, dtype=dtype),
        chol_diag=torch.full((b, n, 3), softplus_inverse(sigma), dtype=dtype) + 0.1 * r(b, n, 3),
        chol_off=0.02 * r(b, n, 3),
        opacity_logit=0.5 * r(b, n),
        m_local=torch.eye(3, dtype=dtype).expand(b, n, 3, 3) + 0.05 * r(b, n, 3, 3),
        b_local=0.05 * r(b, n, 3),
        g_matrix=0.05 * r(b, 3, 3),
        g_bias=0.05 * r(b, 3),
        beta=0.3 * r(b, n, 3),
        mu_s=torch.rand(b, n, generator=gen, dtype=dtype),
        tau_raw=0.5 * r(b, n),
    )


def _naive_forward(x: torch.Tensor, s: torch.Tensor, p: g4d.G4DParams,
                   cfg: g4d.G4DConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """R1 §2.1 written the naive way: build ``Sigma^{-1}``, ``det``, ``exp``.

    This is deliberately the implementation R1 §0.4 rejects (explicit inverse,
    explicit determinant, ordinary-domain normalisation).  It exists only as the
    cross-check; nothing in the arm may look like it.
    """
    diag = torch.nn.functional.softplus(p.chol_diag)
    zero = torch.zeros_like(diag[..., 0])
    lower = torch.stack([
        torch.stack([diag[..., 0], zero, zero], dim=-1),
        torch.stack([p.chol_off[..., 0], diag[..., 1], zero], dim=-1),
        torch.stack([p.chol_off[..., 1], p.chol_off[..., 2], diag[..., 2]], dim=-1),
    ], dim=-2)
    cov = lower @ lower.transpose(-1, -2)                       # (B,N,3,3)
    prec = torch.linalg.inv(cov)
    det = torch.linalg.det(cov)
    tau = g4d.tau_from_raw(p.tau_raw, tau_min=cfg.tau_min, tau_max=cfg.tau_max)
    beta = p.beta if p.mode == "A3" else torch.zeros_like(p.beta)

    ds = s.unsqueeze(-1) - p.mu_s.unsqueeze(1)                  # (B,P,N)
    mu_cs = p.mu_x.unsqueeze(1) + beta.unsqueeze(1) * ds.unsqueeze(-1)
    diff = x.unsqueeze(2) - mu_cs                               # (B,P,N,3)
    mahal = torch.einsum("bpni,bnij,bpnj->bpn", diff, prec, diff)
    pdf = torch.exp(-0.5 * mahal) / torch.sqrt(((2.0 * math.pi) ** 3) * det).unsqueeze(1)
    gate = torch.exp(-0.5 * (ds / tau.unsqueeze(1)) ** 2)
    if cfg.marg_norm == "full":
        gate = gate / (tau.unsqueeze(1) * math.sqrt(2.0 * math.pi))
    a = pdf * torch.sigmoid(p.opacity_logit).unsqueeze(1) * gate
    w = a / (a.sum(dim=-1, keepdim=True) + cfg.eps)

    mx = torch.einsum("bnij,bpj->bpni", p.m_local, x)
    local = (w.unsqueeze(-1) * (mx + p.b_local.unsqueeze(1))).sum(dim=-2)
    glob = torch.einsum("bij,bpj->bpi", p.g_matrix, x) + p.g_bias.unsqueeze(1)
    if cfg.clamp == "two":
        glob = glob.clamp(0.0, 1.0)
    pre = glob + local if cfg.residual else local
    y = pre if cfg.clamp == "none" else pre.clamp(0.0, 1.0)
    return y, w


def _arm(mode: str, *, n: int = 6, d: int = 8, h: int = 12, seed: int = 7,
         **kw) -> g4d.G4DArm:
    torch.manual_seed(seed)
    return g4d.G4DArm(g4d.G4DConfig(mode=mode, n_gauss=n, cond_dim=d, hidden=h, **kw))


# --------------------------------------------------------------------------- #
# 1. equivalence: the conditional parameterisation loses nothing (R1 §4.1)
# --------------------------------------------------------------------------- #
def test_cond_parameterisation_is_exact_on_random_spd4():
    sigma = _random_spd4(256, seed=11)
    cov, beta, tau2 = g4d.cond_from_sigma4(sigma)

    # (a) the textbook conditional mean shift, written independently here
    naive_shift = sigma[..., :3, 3] / sigma[..., 3, 3].unsqueeze(-1)
    ds = torch.linspace(-1.5, 1.5, 7, dtype=DOUBLE)
    mu_naive = naive_shift.unsqueeze(1) * ds[None, :, None]
    mu_param = beta.unsqueeze(1) * ds[None, :, None]
    assert float((mu_naive - mu_param).abs().max()) == 0.0

    # (b) the textbook Schur complement vs the recovered C
    naive_cov = sigma[..., :3, :3] - (
        sigma[..., :3, 3].unsqueeze(-1) @ sigma[..., 3, :3].unsqueeze(-2)
    ) / sigma[..., 3, 3].unsqueeze(-1).unsqueeze(-1)
    assert float((naive_cov - cov).abs().max()) == 0.0

    # (c) round trip through the assembly
    chol = g4d.cholesky_lower(cov)
    rebuilt = g4d.sigma4_from_cond(chol, beta, tau2.sqrt())
    assert float((rebuilt - sigma).abs().max()) < 1e-12
    cov2, beta2, tau2b = g4d.cond_from_sigma4(rebuilt)
    assert float((cov2 - cov).abs().max()) < 1e-12
    assert float((beta2 - beta).abs().max()) < 1e-12
    assert float((tau2b - tau2).abs().max()) < 1e-12


def test_sigma4_from_cond_places_s_on_the_fourth_axis():
    n = 32
    gen = torch.Generator().manual_seed(3)
    chol = torch.tril(torch.randn(n, 3, 3, generator=gen, dtype=DOUBLE))
    chol = chol + torch.diag_embed(torch.rand(n, 3, generator=gen, dtype=DOUBLE) + 0.5)
    beta = torch.randn(n, 3, generator=gen, dtype=DOUBLE)
    tau = torch.rand(n, generator=gen, dtype=DOUBLE) + 0.2
    sigma = g4d.sigma4_from_cond(chol, beta, tau)
    assert float((sigma[..., 3, 3] - tau * tau).abs().max()) == 0.0
    assert float((sigma[..., :3, 3] - (tau * tau).unsqueeze(-1) * beta).abs().max()) == 0.0
    assert float((sigma - sigma.transpose(-1, -2)).abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 2. the forward really is in the log domain (R1 §4.3)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["A2", "A3"])
@pytest.mark.parametrize("marg_norm", ["peak", "full"])
def test_log_domain_matches_naive_float64(mode: str, marg_norm: str):
    cfg = g4d.G4DConfig(mode=mode, n_gauss=4, marg_norm=marg_norm)
    p = _cond_params(mode=mode, n=4, seed=5)
    carrier = g4d.Glut4DCarrier(cfg)
    gen = torch.Generator().manual_seed(9)
    x = torch.rand(2, 32, 3, generator=gen, dtype=DOUBLE)
    s = torch.rand(2, 32, generator=gen, dtype=DOUBLE)

    y = carrier(x, s, p)
    y_naive, w_naive = _naive_forward(x, s, p, cfg)
    assert torch.allclose(y, y_naive, rtol=1e-10, atol=1e-12)

    chol, _, _ = g4d.lower_from_raw(p.chol_diag, p.chol_off)
    tau = g4d.tau_from_raw(p.tau_raw)
    beta = p.beta if mode == "A3" else torch.zeros_like(p.beta)
    mu_cs, log_g, _ = g4d.conditional_params(p.mu_x, p.mu_s, beta, tau, s,
                                             marg_norm=marg_norm)
    log_a = g4d.log_gauss3(x, mu_cs, chol) \
        + torch.nn.functional.logsigmoid(p.opacity_logit).unsqueeze(1) + log_g
    w, log_z, null_mass = g4d.mixture_log_weights(log_a, eps=cfg.eps)
    assert torch.allclose(w, w_naive, rtol=1e-10, atol=1e-14)

    # log Z is exactly GLUT Eq.2's "sum_i a_i + eps"
    sum_a = torch.exp(log_a).sum(-1)
    assert torch.allclose(torch.exp(log_z), sum_a + cfg.eps, rtol=1e-12)
    assert torch.allclose(null_mass, cfg.eps / (sum_a + cfg.eps), rtol=1e-10)


def test_log_domain_survives_where_the_ordinary_domain_underflows():
    """Far-from-mean primitives: ``exp(log p) == 0`` yet ``w != 0``, ``null_mass < 1``."""
    b, n = 1, 4
    mu_x = torch.zeros(b, n, 3)
    mu_x[:, 1:, :] = 50.0                                # three unreachable primitives
    p = g4d.G4DParams(
        mode="A3", mu_x=mu_x,
        chol_diag=torch.full((b, n, 3), softplus_inverse(0.05)),
        chol_off=torch.zeros(b, n, 3),
        opacity_logit=torch.zeros(b, n),
        m_local=torch.eye(3).expand(b, n, 3, 3).contiguous(),
        b_local=torch.zeros(b, n, 3),
        g_matrix=torch.zeros(b, 3, 3), g_bias=torch.zeros(b, 3),
        beta=torch.zeros(b, n, 3), mu_s=torch.zeros(b, n),
        tau_raw=torch.full((b, n), g4d.TAU_RAW_INIT))
    x = torch.zeros(b, 1, 3)
    s = torch.zeros(b, 1)

    chol, _, _ = g4d.lower_from_raw(p.chol_diag, p.chol_off)
    tau = g4d.tau_from_raw(p.tau_raw)
    mu_cs, log_g, _ = g4d.conditional_params(p.mu_x, p.mu_s, p.beta, tau, s)
    log_p = g4d.log_gauss3(x, mu_cs, chol)
    # the ordinary domain has already lost these three, in float32 and float64
    assert float(torch.exp(log_p)[..., 1:].max()) == 0.0
    assert float(torch.exp(log_p.double())[..., 1:].max()) == 0.0
    assert bool(torch.isfinite(log_p).all())            # the log domain has not

    log_a = log_p + torch.nn.functional.logsigmoid(p.opacity_logit).unsqueeze(1) + log_g
    w, _, null_mass = g4d.mixture_log_weights(log_a, eps=EPS)
    assert float(w[..., 0].min()) > 0.0
    assert float(null_mass.max()) < 1.0


def test_log_domain_survives_where_the_ordinary_domain_overflows():
    """A covariance narrow enough that ``exp(log p)`` overflows float32."""
    b, n = 1, 4
    cfg = g4d.G4DConfig(mode="A3", n_gauss=n)
    mu_x = torch.zeros(b, n, 3)
    mu_x[:, 1:, :] = 1.0
    p = g4d.G4DParams(
        mode="A3", mu_x=mu_x,
        chol_diag=torch.full((b, n, 3), softplus_inverse(1e-15)),
        chol_off=torch.zeros(b, n, 3),
        opacity_logit=torch.zeros(b, n),
        m_local=torch.eye(3).expand(b, n, 3, 3).contiguous(),
        b_local=torch.full((b, n, 3), 0.25),
        g_matrix=torch.zeros(b, 3, 3), g_bias=torch.zeros(b, 3),
        beta=torch.zeros(b, n, 3), mu_s=torch.zeros(b, n),
        tau_raw=torch.full((b, n), g4d.TAU_RAW_INIT))
    x = torch.zeros(b, 1, 3)
    s = torch.zeros(b, 1)

    y_naive, w_naive = _naive_forward(x, s, p, cfg)
    assert not bool(torch.isfinite(w_naive).all())       # the naive route is nan/inf

    y, aux = g4d.Glut4DCarrier(cfg)(x, s, p, return_aux=True)
    assert bool(torch.isfinite(y).all())
    assert float(aux.null_mass.max()) < 1.0


# --------------------------------------------------------------------------- #
# 3. axis order: the fourth axis is s, and only s
# --------------------------------------------------------------------------- #
def test_fourth_axis_is_s_and_does_not_touch_the_conditional_covariance():
    p = _cond_params(b=2, n=5, seed=13)
    tau = g4d.tau_from_raw(p.tau_raw)
    chol, _, _ = g4d.lower_from_raw(p.chol_diag, p.chol_off)
    cov = chol @ chol.transpose(-1, -2)

    s1 = torch.rand(2, 9, dtype=DOUBLE)
    s2 = s1 + 0.37
    mu1, g1, ds1 = g4d.conditional_params(p.mu_x, p.mu_s, p.beta, tau, s1)
    mu2, g2, ds2 = g4d.conditional_params(p.mu_x, p.mu_s, p.beta, tau, s2)

    # the mean moves exactly along beta, by exactly (s2 - s1)
    want = p.beta.unsqueeze(1) * (s2 - s1).unsqueeze(-1).unsqueeze(-1)
    assert float((mu2 - mu1 - want).abs().max()) < 1e-14
    assert float((ds2 - ds1 - (s2 - s1).unsqueeze(-1)).abs().max()) < 1e-14
    # the gate moved too
    assert float((g2 - g1).abs().max()) > 0.0
    # C is not a function of s at all: the implied Sigma's (1:3,1:3) conditional
    # block is the same object for either s
    for s in (s1, s2):
        sigma = g4d.sigma4_from_cond(chol, p.beta, tau)
        cov_back, beta_back, tau2_back = g4d.cond_from_sigma4(sigma)
        assert float((cov_back - cov).abs().max()) < 1e-12
        assert float((beta_back - p.beta).abs().max()) < 1e-12
        assert float((tau2_back - tau * tau).abs().max()) < 1e-14


# --------------------------------------------------------------------------- #
# 4. gradcheck (R1 §9 G0)
# --------------------------------------------------------------------------- #
def test_gradcheck_through_every_conditional_parameter():
    cfg = g4d.G4DConfig(mode="A3", n_gauss=3, clamp="none")
    carrier = g4d.Glut4DCarrier(cfg)
    base = _cond_params(b=2, n=3, seed=21)
    gen = torch.Generator().manual_seed(22)
    x = torch.rand(2, 5, 3, generator=gen, dtype=DOUBLE)
    s = torch.rand(2, 5, generator=gen, dtype=DOUBLE)

    names = ("chol_diag", "chol_off", "beta", "tau_raw", "mu_x", "mu_s",
             "opacity_logit")
    args = tuple(getattr(base, k).clone().requires_grad_(True) for k in names)

    def fn(*vals):
        kw = dict(zip(names, vals))
        p = g4d.G4DParams(
            mode="A3", m_local=base.m_local, b_local=base.b_local,
            g_matrix=base.g_matrix, g_bias=base.g_bias, **kw)
        return carrier(x, s, p)

    assert torch.autograd.gradcheck(fn, args, eps=1e-6, atol=1e-7, rtol=1e-5)


# --------------------------------------------------------------------------- #
# 5 / 6. step 0: identity, and A2 == A3 bit for bit
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["A1", "A2", "A3"])
def test_step0_is_the_identity(mode: str):
    arm = _arm(mode, n=8)
    u = torch.randn(4, 8)
    p = arm.theta(u)
    x = torch.rand(4, 64, 3)
    s = torch.rand(4, 64)
    with torch.no_grad():
        y = arm.transform(p, x, s)
    assert float((y - x).abs().max()) < 1e-5


def test_a2_and_a3_are_bit_identical_at_step_zero():
    a2, a3 = _arm("A2", n=8, seed=31), _arm("A3", n=8, seed=31)
    for (n2, t2), (n3, t3) in zip(a2.state_dict().items(), a3.state_dict().items()):
        assert n2 == n3
        assert torch.equal(t2, t3), n2

    gen = torch.Generator().manual_seed(32)
    u = torch.randn(3, 8, generator=gen)
    x = torch.rand(3, 40, 3, generator=gen)
    s = torch.rand(3, 40, generator=gen)
    with torch.no_grad():
        y2 = a2.transform(a2.theta(u), x, s)
        y3 = a3.transform(a3.theta(u), x, s)
    assert torch.equal(y2, y3)
    assert float(a3.theta(u).beta.detach().abs().max()) == 0.0   # beta is zero-init


# --------------------------------------------------------------------------- #
# 7. the R1 §3 parameter table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,n,want", [
    ("A0", 48, 1068), ("A1", 48, 1068),
    ("A2", 48, 1308), ("A3", 48, 1308),
    ("A3", 39, 1065),               # the generation-dimension-matched control row
    ("A4", 48, 1404),               # kept in the table, not constructible
])
def test_parameter_counts(mode: str, n: int, want: int):
    assert g4d.n_params_g4d(n, mode) == want


@pytest.mark.parametrize("mode,n,want", [
    ("A0", 48, 1068), ("A1", 48, 1068),
    ("A2", 48, 1308), ("A3", 48, 1308), ("A3", 39, 1065),
])
def test_generated_tensors_match_the_parameter_count(mode: str, n: int, want: int):
    arm = _arm(mode, n=n)
    p = arm.theta(torch.randn(2, 8))
    per = 0
    for name, k in (("mu_x", 3), ("chol_diag", 3), ("chol_off", 3),
                    ("opacity_logit", 1), ("m_local", 9), ("b_local", 3),
                    ("beta", 3), ("mu_s", 1), ("tau_raw", 1)):
        t = getattr(p, name)
        if t is not None:
            per += k
    assert per * n + 12 == want
    assert arm.generator.theta_dim == want


def test_a4_is_named_but_not_constructible():
    assert "A4" in g4d.G4D_MODES
    with pytest.raises(NotImplementedError):
        g4d.G4DConfig(mode="A4")


# --------------------------------------------------------------------------- #
# 8. no fallback anywhere (R1 §7)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["A0", "A1", "A2", "A3"])
def test_degenerate_cholesky_raises_instead_of_falling_back(mode: str):
    arm = _arm(mode, n=4)
    p = arm.theta(torch.randn(1, 8))
    broken = {k: (None if getattr(p, k) is None else getattr(p, k).detach().clone())
              for k in ("mu_x", "chol_diag", "chol_off", "opacity_logit",
                        "m_local", "b_local", "g_matrix", "g_bias",
                        "beta", "mu_s", "tau_raw")}
    broken["chol_diag"] = torch.full_like(broken["chol_diag"], -1e6)   # softplus -> 0
    bad = g4d.G4DParams(mode=mode, **broken)
    with pytest.raises(g4d.G4DNumericalError):
        arm.transform(bad, torch.rand(1, 8, 3), torch.rand(1, 8))


def test_cholesky_lower_raises_on_a_non_pd_matrix():
    cov = torch.eye(3, dtype=DOUBLE).expand(2, 3, 3).clone()
    cov[1, 2, 2] = -1.0
    with pytest.raises(g4d.G4DNumericalError):
        g4d.cholesky_lower(cov)


def test_cond_from_sigma4_raises_on_a_non_pd_joint():
    sigma = torch.eye(4, dtype=DOUBLE).unsqueeze(0).clone()
    sigma[0, 0, 3] = sigma[0, 3, 0] = 2.0     # Schur complement goes negative
    with pytest.raises(g4d.G4DNumericalError):
        g4d.cond_from_sigma4(sigma)


def test_forward_raises_on_a_non_finite_value():
    cfg = g4d.G4DConfig(mode="A3", n_gauss=3)
    p = _cond_params(b=1, n=3, seed=41)
    x = torch.rand(1, 4, 3, dtype=DOUBLE)
    x[0, 0, 0] = float("nan")
    with pytest.raises(g4d.G4DNumericalError):
        g4d.Glut4DCarrier(cfg)(x, torch.rand(1, 4, dtype=DOUBLE), p)


def test_the_deleted_r0_symbols_are_gone():
    for name in ("build_rotation_4d", "build_scaling_rotation_4d", "covariance_4d",
                 "conditional_slice", "SliceGeometry", "SIGMA44_FLOOR", "l_m4d",
                 "ROT_SIGN_CHOICES", "S_TAU_INIT", "SIGMA_S_INIT"):
        assert not hasattr(g4d, name), name
    for name in ("slice_det_fallback", "sigma44_floor_hits", "schur_pd_violations",
                 "logdet_floor_hits"):
        assert name not in g4d.GUARD_COLUMNS
        assert name not in g4d.STEP_EXTRA_COLUMNS
    assert "L_m4d" not in g4d.OFF_BY_DEFAULT_LOSS_COLUMNS


# --------------------------------------------------------------------------- #
# 9 / 10. A1's explicit gate: bit-exact endpoints, exactly linear path
# --------------------------------------------------------------------------- #
def _perturbed_a1(seed: int = 51) -> g4d.G4DArm:
    """A1 with a non-trivial ``T`` (step 0's ``T`` is the identity)."""
    arm = _arm("A1", n=6, seed=seed)
    gen = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for prm in arm.generator.parameters():
            prm.add_(0.25 * torch.randn(prm.shape, generator=gen))
    return arm


def test_a1_endpoints_are_bit_exact():
    arm = _perturbed_a1()
    p = arm.theta(torch.randn(2, 8))
    x = torch.rand(2, 64, 3)
    with torch.no_grad():
        t = glut_forward(x, p.as_glut_params(), clamp=arm.cfg.clamp,
                         residual=arm.cfg.residual, eps=arm.cfg.eps)
        f0 = arm.transform(p, x, torch.zeros(2, 64))
        f1 = arm.transform(p, x, torch.ones(2, 64))
    assert float((t - x).abs().max()) > 1e-3          # T is not the identity
    assert torch.equal(f0, x)
    assert torch.equal(f1, t)


def test_r_line_is_exactly_zero_on_a1():
    arm = _perturbed_a1(53)
    p = arm.theta(torch.randn(2, 8))
    x = torch.rand(2, 128, 3)
    s = torch.rand(2, 128)
    with torch.no_grad():
        fs = arm.transform(p, x, s)
        f0 = arm.transform(p, x, torch.zeros_like(s))
        f1 = arm.transform(p, x, torch.ones_like(s))
    assert float(g4d.r_line(fs, f0, f1, s)) == 0.0


def test_r_line_is_positive_on_a3():
    arm = _arm("A3", n=6, seed=55)
    gen = torch.Generator().manual_seed(56)
    with torch.no_grad():
        for prm in arm.generator.parameters():
            prm.add_(0.25 * torch.randn(prm.shape, generator=gen))
    p = arm.theta(torch.randn(2, 8))
    x = torch.rand(2, 128, 3)
    s = torch.rand(2, 128)
    with torch.no_grad():
        fs = arm.transform(p, x, s)
        f0 = arm.transform(p, x, torch.zeros_like(s))
        f1 = arm.transform(p, x, torch.ones_like(s))
    assert float(g4d.r_line(fs, f0, f1, s)) > 0.0


# --------------------------------------------------------------------------- #
# image formation and telemetry (R1 §5 / §10)
# --------------------------------------------------------------------------- #
def test_compose_headline_is_per_arm_and_mode_is_mandatory():
    img = torch.rand(1, 3, 4, 4)
    field = torch.rand(1, 1, 4, 4)
    f_img = torch.rand(1, 3, 4, 4)
    with pytest.raises(TypeError):
        g4d.compose_headline(img, field, f_img)              # type: ignore[call-arg]
    a0 = g4d.compose_headline(img, field, f_img, mode="A0")
    assert torch.allclose(a0, img + field * (f_img - img), atol=1e-6)
    for mode in ("A1", "A2", "A3"):
        assert torch.equal(g4d.compose_headline(img, field, f_img, mode=mode), f_img)


@pytest.mark.parametrize("mode,has_tau,has_beta", [
    ("A0", False, False), ("A1", False, False),
    ("A2", True, False), ("A3", True, True),
])
def test_step_extra_columns_and_aux_agree(mode: str, has_tau: bool, has_beta: bool):
    want = set(g4d.step_extra_columns(mode))
    assert ("tau_p05" in want) is has_tau
    assert ("beta_absmean" in want) is has_beta
    arm = _arm(mode, n=6)
    p = arm.theta(torch.randn(2, 8))
    with torch.no_grad():
        _, aux = arm.transform(p, torch.rand(2, 16, 3), torch.rand(2, 16),
                               return_aux=True)
    cols = aux.columns()
    # gnorm is the training loop's column, not the carrier's
    assert want - {"gnorm"} <= set(cols)
    assert int(aux.n_pairs_s) == 2 * 16
    assert int(aux.cholesky_info_nonzero) == 0
    assert set(g4d.step_columns(1, mode=mode)) >= want | {"L_rec", "R_line"}


def test_total_loss_refuses_a_zero_weighted_r_line():
    y_hat = torch.rand(2, 8, 3)
    y = torch.rand(2, 8, 3)
    o = torch.rand(2, 4)
    line = torch.rand(())
    with pytest.raises(ValueError):
        g4d.total_loss(y_hat, y, o, lam_hc=0.0, lam_sparse=0.0, lam_line=0.0,
                       line=line)
    loss, cols = g4d.total_loss(y_hat, y, o, lam_hc=0.0, lam_sparse=0.0, line=line)
    assert "R_line" in cols and "L_m4d" not in cols
    assert math.isclose(cols["L_total"],
                        cols["L_rec"] + g4d.LAMBDA_LINE * float(line), rel_tol=1e-6)
