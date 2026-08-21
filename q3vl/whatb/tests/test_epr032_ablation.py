"""EPR-032 ablation arms (J-GATE-AFF / J-4D / J-P15) -- §8.1-style assertions.

Eight things are pinned here, each of which has a silent failure mode that would
leave the arms' tables looking perfectly healthy:

1. the generalised monomial design reproduces ``core.design_matrix`` bit for bit
   at p1 / p2, so the new p15 column block sits in the same layout;
2. the 4-D partition of unity really sums to 1, and the arm's runtime assertion
   on it raises when it does not;
3. the arm-1 gate is ``glut.glut_forward``'s gate and the arm-1 design matrix is
   ``glut``'s carrier -- not merely something built out of its pieces -- and its
   point blocking moves no reduction;
4. the ``E_p1 >= E_p15 >= E_p2`` nesting assertion still *raises* on a真 error
   (a passing assertion that cannot fail is the "defined but never wired" bug
   this campaign has hit three times), **including** the vacuous-pass route
   where an order was never boarded and the check list is empty;
5. the pool sha assertion refuses a drifted LUT id set;
6. a row pool cut by ``--n-lut`` writes no reference and no derived column;
7. ``de00_paired`` pairs (point, LUT, channel) the way ``run.residuals`` writes
   them, and differs from ``run.de00_of`` on non-trivial input;
8. the arms emit both dE00 calibers (``*_legacy`` / ``*_paired``).
"""

from __future__ import annotations

import functools
import json
import math
from pathlib import Path

import pytest
import torch

from q3vl.whatb import glut as G
from q3vl.whatb.codec import lutcode as C
from q3vl.whatb.jetlut import ablation_arms as A
from q3vl.whatb.jetlut.core import Atlas, admm_lad, design_matrix

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_HAS_INDEX = C.S.split_index_path("train").is_file()
needs_index = pytest.mark.skipif(
    not _HAS_INDEX, reason="the split index is on the soft NFS mount")
_HAS_CKPT = Path(A.AFFONLY_CKPT).is_file()
needs_ckpt = pytest.mark.skipif(
    not _HAS_CKPT, reason=f"{A.AFFONLY_CKPT} not mounted")


def _grid(n: int) -> torch.Tensor:
    ax = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    r, g, b = torch.meshgrid(ax, ax, ax, indexing="ij")
    return torch.stack((r.reshape(-1), g.reshape(-1), b.reshape(-1)), dim=-1)


def _fake_geometry(n: int = 12, seed: int = 7) -> dict[str, torch.Tensor]:
    """A ``SharedGeometry``-shaped set of tables, no checkpoint needed."""
    g = torch.Generator().manual_seed(seed)
    return {
        "mu": G.uniform_grid_positions(n, dtype=torch.float64)
              + 0.02 * torch.randn((n, 3), generator=g, dtype=torch.float64),
        "chol_diag": torch.full((n, 3), 1.5, dtype=torch.float64)
                     + 0.1 * torch.randn((n, 3), generator=g, dtype=torch.float64),
        "chol_off": 0.05 * torch.randn((n, 3), generator=g, dtype=torch.float64),
        "opacity_logit": torch.randn((n,), generator=g, dtype=torch.float64),
    }


# --------------------------------------------------------------------------- #
# 1. the generalised design matrix
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order,p", [("p1", 1), ("p2", 2)])
def test_mono_design_matches_core(order, p):
    x = _grid(9)
    a = Atlas(m=3, c=0.7)
    assert torch.equal(A.design_matrix_mono(x, a, order), design_matrix(x, a, p))


def test_p15_shape_and_param_count():
    x = _grid(7)
    a = Atlas(m=3, c=0.7)
    f = A.design_matrix_p15(x, a)
    assert f.shape == (x.shape[0], 4 + 7 * 27)
    for m in (3, 4, 5):
        assert A.n_dynamic_params_order(m, "p15") == 21 * m ** 3 + 12
    assert A.n_dynamic_params_order(3, "p1") == 336
    assert A.n_dynamic_params_order(3, "p2") == 822


def test_p15_is_between_p1_and_p2_spans():
    """p15 columns are a subset of p2's and a superset of p1's."""
    x = _grid(7)
    a = Atlas(m=3, c=0.7)
    f1 = A.design_matrix_mono(x, a, "p1")
    f15 = A.design_matrix_mono(x, a, "p15")
    f2 = A.design_matrix_mono(x, a, "p2")
    n = a.n
    assert torch.equal(f1[:, :4], f15[:, :4])
    for i in range(n):
        assert torch.equal(f1[:, 4 + 4 * i:8 + 4 * i],
                           f15[:, 4 + 7 * i:8 + 7 * i])
        assert torch.equal(f15[:, 4 + 7 * i:11 + 7 * i],
                           f2[:, 4 + 10 * i:11 + 10 * i])


# --------------------------------------------------------------------------- #
# 2. the 4-D atlas
# --------------------------------------------------------------------------- #
def test_pou_4d_sums_to_one():
    torch.manual_seed(0)
    xs = torch.rand((2048, 4), dtype=torch.float64)
    for m, m_s, c in ((3, 3, 0.7), (4, 3, 0.7), (3, 3, 0.4), (3, 5, 1.2)):
        atlas = A.Atlas4D(m=m, m_s=m_s, c=c)
        pi = A.pou_weights_4d(xs, atlas)
        assert pi.shape == (xs.shape[0], m ** 3 * m_s)
        assert float((pi.sum(-1) - 1.0).abs().max()) < 1e-6


def test_assert_pou_4d_is_the_arms_runtime_check():
    """B4: ``cmd_j4d`` runs this before any fit; it must return the deviation."""
    rec = A.assert_pou_4d(A.Atlas4D(m=3, m_s=3, c=0.7), n_probe=4096)
    assert rec["n_probe"] == 4096 and rec["N4"] == 81
    assert rec["max_abs_dev_from_one"] < A.POU4D_TOL


def test_assert_pou_4d_raises_when_the_rows_do_not_sum_to_one():
    """The assertion must be able to fail: hand it a non-normalised gate."""
    real = A.pou_weights_4d
    try:
        A.pou_weights_4d = lambda xs, atlas: real(xs, atlas) * 0.5
        with pytest.raises(AssertionError, match="not a partition of unity"):
            A.assert_pou_4d(A.Atlas4D(m=2, m_s=2, c=0.7), n_probe=256)
    finally:
        A.pou_weights_4d = real


def test_atlas4d_geometry_and_counts():
    atlas = A.Atlas4D(m=4, m_s=3, c=0.7)
    assert atlas.n == 64 * 3
    assert A.n_dynamic_params_4d(atlas) == 3 * 5 * 192 + 12
    assert abs(atlas.sigma_x - 0.7 * math.sqrt(3) / 8.0) < 1e-12
    assert abs(atlas.sigma_s - 0.7 / 6.0) < 1e-12
    mu = atlas.centres()
    assert mu.shape == (192, 4)
    assert sorted({float(v) for v in mu[:, 3]}) == pytest.approx(
        [1 / 6, 3 / 6, 5 / 6])
    assert A.fill_distance_1d(3) == pytest.approx(1 / 6)


def test_design_matrix_4d_shape_and_global_block():
    xs = A.stack_s(_grid(5), [0.0, 0.5, 1.0])
    atlas = A.Atlas4D(m=3, m_s=3, c=0.7)
    f = A.design_matrix_4d(xs, atlas, chunk=64)
    assert f.shape == (xs.shape[0], 4 + 5 * 81)
    assert torch.equal(f[:, 0], torch.ones_like(f[:, 0]))
    assert torch.equal(f[:, 1:4], xs[:, :3])       # no s in the global block
    # chunking is numerically inert
    assert torch.equal(f, A.design_matrix_4d(xs, atlas, chunk=7))


def test_j4d_scoring_path_takes_three_channel_colours():
    """``fit_and_score``'s metrics consume colours, the design consumes ``(x,s)``.

    Regression for the J-4D driver: the design matrix is built from the stacked
    4-D rows, but ``de00_of`` and the gamut count add ``x`` to a ``(l, Q, 3)``
    residual block, so handing them the 4-D tensor broadcasts ``(Q,1,4)`` against
    ``(Q,l,3)`` and raises.  The arm must pass ``xs[:, :3]``.
    """
    from q3vl.whatb.jetlut import run as R

    x = _grid(5)
    atlas = A.Atlas4D(m=2, m_s=2, c=0.7)
    s_values = [0.0, 1.0]
    xs = A.stack_s(x, s_values)
    r = torch.stack([0.10 * x[:, 0], 0.05 * x[:, 1] ** 2,
                     -0.08 * x[:, 2]], dim=-1)
    r4 = A.stack_s_residual(r, s_values)
    xc = xs[:, :3].contiguous()
    build = lambda: A.design_matrix_4d(xs, atlas, chunk=64)   # noqa: E731
    rec = R.fit_and_score(xc, xc, r4, r4, build, build, {"arm": "J-4D"},
                          max_iter=60)
    assert rec["fitgrid_pre_clamp"]["n_points"] == xs.shape[0]
    assert rec["fitgrid_pre_clamp"]["n_lut"] == 1
    assert rec["P_feat"] == 4 + 5 * atlas.n
    with pytest.raises(RuntimeError):                 # the shape the arm avoids
        R.fit_and_score(xs, xs, r4, r4, build, build, {}, max_iter=5)


def test_explicit_gate_carrier_is_free_of_the_s_axis():
    """The NOTES proposition, as a runnable check.

    ``sum_{x,s} |s (Phi(x) th - r(x))| = (sum_s s) sum_x |Phi(x) th - r(x)|`` for
    ``s >= 0``, so the LAD optimum of the explicit-gate carrier does not move
    when the s axis is added and its cost is identically 0.  This is the anchor
    the J-4D arm's excess is measured against.
    """
    torch.manual_seed(3)
    x = _grid(7)
    a = Atlas(m=2, c=0.7)             # 36 columns on 343 points: overdetermined
    f = design_matrix(x, a, 1)
    r = torch.stack([0.2 * torch.sin(4.0 * x[:, 1]),
                     0.1 * x[:, 2] ** 2,
                     -0.15 * x[:, 0] * x[:, 2]], dim=-1)
    s_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    f_s = torch.cat([float(s) * f for s in s_values], dim=0)
    r_s = A.stack_s_residual(r, s_values)

    base = admm_lad(f, r, max_iter=2000)
    stacked = admm_lad(f_s, r_s, max_iter=2000)
    scale = sum(s_values)
    lhs = float(stacked["primal"].sum())
    rhs = scale * float(base["primal"].sum())
    assert abs(lhs - rhs) <= 1e-4 * max(rhs, 1e-12), (lhs, rhs)


# --------------------------------------------------------------------------- #
# 3. arm 1 is wired to glut.py
# --------------------------------------------------------------------------- #
def test_gate_matches_glut_forward():
    geo = _fake_geometry()
    rec = A.assert_gate_matches_glut(geo, n_probe=4096)
    assert rec["n_probe"] == 4096
    # the measured deviation and the floor it is judged against are both in the
    # artefact; on CPU the two evaluations are still exactly equal
    assert rec["max_abs_diff"] <= rec["tol"] == A.GATE_XCHECK_TOL
    assert rec["max_abs_diff"] == 0.0 and rec["bitwise_equal"]
    # Shepard + eps + opacity: not an exact partition of unity
    assert rec["weight_row_sum_max"] <= 1.0 + 1e-12


@pytest.mark.parametrize("eps,raises", [(1e-15, False), (1e-13, False),
                                        (1e-9, True), (1e-6, True)])
def test_gate_cross_check_tolerance_floor(eps, raises):
    """The floor absorbs CUDA's reduction-order noise and nothing more.

    On cuda:0 the second gate evaluation (different point chunk) differs from the
    first at the denormal level (6.776e-21 measured, N=48), which the exact
    ``torch.equal`` form rejected.  ``GATE_XCHECK_TOL = 1e-12`` is the floor;
    a perturbation at 1e-9 or above -- still far below any real mis-wiring, which
    moves ``w_i`` by O(1) -- must still raise.
    """
    geo = _fake_geometry()
    real = A.gate_weights
    calls = {"n": 0}

    def perturbed(x, g, **kw):
        calls["n"] += 1
        w = real(x, g, **kw)
        return w if calls["n"] == 1 else w + eps
    try:
        A.gate_weights = perturbed
        if raises:
            with pytest.raises(AssertionError, match="gate cross-check failed"):
                A.assert_gate_matches_glut(geo, n_probe=256)
        else:
            rec = A.assert_gate_matches_glut(geo, n_probe=256)
            assert rec["max_abs_diff"] == pytest.approx(eps, rel=1e-6)
            assert not rec["bitwise_equal"]     # equal-to-tol, not bit-equal
    finally:
        A.gate_weights = real


def test_gate_cross_check_raises_when_the_gate_is_not_gluts():
    """The assertion must be able to fail: feed it a gate that is not glut's."""
    geo = _fake_geometry()
    real = A.gate_weights
    try:
        calls = {"n": 0}

        def flaky(x, g, **kw):
            calls["n"] += 1
            w = real(x, g, **kw)
            return w if calls["n"] == 1 else w * 0.5
        A.gate_weights = flaky
        with pytest.raises(AssertionError, match="gate cross-check failed"):
            A.assert_gate_matches_glut(geo, n_probe=256)
    finally:
        A.gate_weights = real


def test_design_matrix_gate_is_the_glut_carrier():
    """``x + Phi(x) theta`` == ``glut_forward`` on the relabelled coefficients."""
    torch.manual_seed(11)
    geo = _fake_geometry(n=10)
    x = _grid(9)
    n = geo["mu"].shape[0]
    f = A.design_matrix_gate(x, geo)
    assert f.shape == (x.shape[0], 4 + 4 * n)
    theta = 0.1 * torch.randn((4 + 4 * n, 3), dtype=torch.float64)
    rec = A.assert_carrier_matches_glut(x, geo, theta, n_probe=x.shape[0])
    assert rec["max_abs_diff"] <= 1e-9

    # and the relabelling is the one the arm reports
    params = A.theta_to_glut_params(theta, geo)
    assert params.m_local.shape == (1, n, 3, 3)
    assert params.g_matrix.shape == (1, 3, 3)
    ref = G.glut_forward(x.unsqueeze(0), params, clamp="none", residual=True)[0]
    assert torch.allclose(ref, x + f @ theta, atol=1e-9)


def test_design_matrix_gate_chunking_is_bitwise_inert():
    """N2: ``chunk`` must move no reduction, at any (non-dividing) block size.

    The gate's ``sum_j p_j o_j`` denominator is a reduction, so calling the gate
    once per block makes the design matrix a function of ``chunk`` at ~1 ulp;
    the arm evaluates the gate once and blocks only the elementwise assembly.
    """
    geo = _fake_geometry(n=12)
    torch.manual_seed(1)
    x = torch.rand((1000, 3), dtype=torch.float64)
    f = A.design_matrix_gate(x, geo, chunk=4096)          # one block
    for chunk in (1, 97, 333, 999, 1000, 4096):
        assert torch.equal(f, A.design_matrix_gate(x, geo, chunk=chunk)), chunk


def test_carrier_check_raises_on_a_wrong_relabelling():
    """Corrupt the *relabelling*, not theta: forget to fold the identity into G.

    Corrupting theta alone cannot fail -- both sides of the identity read the
    same theta -- so the失效 mode this assertion must catch is a wrong
    ``theta_to_glut_params``.
    """
    geo = _fake_geometry(n=6)
    x = _grid(7)
    n = geo["mu"].shape[0]
    torch.manual_seed(5)
    theta = 0.2 * torch.randn((4 + 4 * n, 3), dtype=torch.float64)
    real = A.theta_to_glut_params

    def no_identity_fold(th, g):
        params = real(th, g)
        eye = torch.eye(3, device=th.device, dtype=th.dtype)
        return A._glut_params(g, params.m_local[0], params.b_local[0],
                              params.g_matrix[0] - eye, params.g_bias[0])
    try:
        A.theta_to_glut_params = no_identity_fold
        with pytest.raises(AssertionError, match="carrier cross-check failed"):
            A.assert_carrier_matches_glut(x, geo, theta, n_probe=x.shape[0])
    finally:
        A.theta_to_glut_params = real


# --------------------------------------------------------------------------- #
# 4. the nesting assertion, both ways
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _fit_orders_cached(m: int) -> tuple[dict, ...]:
    x = _grid(9)
    a = Atlas(m=m, c=0.7)
    r = torch.stack([0.25 * torch.sin(4.0 * x[:, 1]) * x[:, 0],
                     0.20 * (1.0 - torch.exp(-3.0 * x[:, 2])),
                     0.15 * x[:, 0] * x[:, 2] - 0.05 * x[:, 2]], dim=-1)
    rows = []
    for order in ("p1", "p15", "p2"):
        f = A.design_matrix_mono(x, a, order)
        sol = admm_lad(f, r, max_iter=2000)
        rows.append({"m": m, "order": order,
                     "l1_fit_mean_per_point": float(sol["primal"].sum()
                                                    / r.numel())})
    return tuple(rows)


def _fit_orders(m: int = 2) -> list[dict]:
    """Fresh copies of the cached fit -- the degradation tests mutate rows."""
    return [dict(r) for r in _fit_orders_cached(m)]


def test_nested_three_holds_on_a_real_fit():
    rows = _fit_orders()
    checks = A.assert_nested_three(rows)
    assert len(checks) == 1 and checks[0]["ok"]
    assert checks[0]["l1_p1"] >= checks[0]["l1_p15"] >= checks[0]["l1_p2"]


def test_nested_three_raises_when_p15_is_degraded():
    """A真 error still raises -- the assertion is not decorative."""
    rows = _fit_orders()
    by = {r["order"]: r for r in rows}
    by["p15"]["l1_fit_mean_per_point"] = by["p1"]["l1_fit_mean_per_point"] * 1.01
    with pytest.raises(AssertionError, match="nested subspace inequality"):
        A.assert_nested_three(rows)


def test_nested_three_raises_when_p2_is_degraded():
    rows = _fit_orders()
    by = {r["order"]: r for r in rows}
    by["p2"]["l1_fit_mean_per_point"] = by["p15"]["l1_fit_mean_per_point"] * 1.01
    with pytest.raises(AssertionError, match="nested subspace inequality"):
        A.assert_nested_three(rows)


def test_nested_three_raises_when_an_order_was_never_boarded():
    """B2: ``--orders p1 p2`` yields ``checks == []`` -- a vacuous pass.

    ``check_nested_three`` skips any m that is missing one of the three orders,
    so a run boarded without p15 produced an empty check list, the assertion
    could not fail, and ``nested_ok: true`` was written anyway.
    """
    rows = [r for r in _fit_orders() if r["order"] != "p15"]
    assert A.check_nested_three(rows) == []          # the vacuous-pass shape
    with pytest.raises(AssertionError, match="cannot fail"):
        A.assert_nested_three(rows, orders=["p1", "p2"])
    with pytest.raises(AssertionError, match="needs rows for orders"):
        A.assert_nested_three(rows)                  # rows alone also suffice
    # and a complete --orders with an incomplete row set still raises
    with pytest.raises(AssertionError, match="needs rows for orders"):
        A.assert_nested_three(rows, orders=["p1", "p15", "p2"])


def test_nested_three_raises_when_an_m_is_missing_a_triple():
    """Complete corpus-wide, incomplete at one m: a skipped m is unchecked."""
    rows = _fit_orders(2) + [dict(r, m=3) for r in _fit_orders(2)
                             if r["order"] != "p15"]
    assert len(A.check_nested_three(rows)) == 1      # m=3 silently skipped
    with pytest.raises(AssertionError, match="complete p1/p15/p2"):
        A.assert_nested_three(rows, orders=A.NESTED_ORDERS)
    with pytest.raises(AssertionError, match="complete p1/p15/p2"):
        A.assert_nested_three(rows, orders=A.NESTED_ORDERS, n_m=2)


def test_nested_three_tolerance_is_not_a_free_pass():
    rows = _fit_orders()
    by = {r["order"]: r for r in rows}
    eps = A.NESTED_TOL / 10.0
    by["p15"]["l1_fit_mean_per_point"] = by["p1"]["l1_fit_mean_per_point"] * (1 + eps)
    by["p2"]["l1_fit_mean_per_point"] = by["p15"]["l1_fit_mean_per_point"] * (1 + eps)
    assert all(q["ok"] for q in A.assert_nested_three(rows))   # inside tol
    by["p2"]["l1_fit_mean_per_point"] *= 1 + 10 * A.NESTED_TOL
    with pytest.raises(AssertionError):
        A.assert_nested_three(rows)


# --------------------------------------------------------------------------- #
# 5. the pool sha assertion
# --------------------------------------------------------------------------- #
def test_pool_sha_rejects_a_drifted_id_set():
    with pytest.raises(AssertionError, match="PROPOSAL"):
        A.assert_pool_sha("held_out", ["lut_%04d" % i for i in range(902)])
    with pytest.raises(AssertionError):
        A.assert_pool_sha("t_lut_unseen", [])
    with pytest.raises(AssertionError, match="PROPOSAL"):
        A.assert_pool_sha("train_full", ["lut_%04d" % i for i in range(3149)])
    with pytest.raises(AssertionError, match="PROPOSAL"):
        A.assert_pool_sha("all", ["lut_%04d" % i for i in range(4051)])
    # an unregistered pool is recorded, not guessed at (``train`` is the
    # ``--n-train`` subsample, whose id set is a function of that flag)
    rec = A.assert_pool_sha("train", ["a", "b"])
    assert rec["expected"] is None and rec["n_lut"] == 2


def test_pool_sha_registry_matches_the_proposal():
    assert A.POOL_SHA["held_out"] == ("d3f17955816f39b1", 902)
    assert A.POOL_SHA["t_lut_unseen"] == ("72f1dd33ce35a12c", 259)
    assert A.POOL_SHA["train_full"] == ("06da414b16ea0c6c", 3149)   # §11.2
    assert A.POOL_SHA["all"] == ("f964eba67b9cdc43", 4051)          # §11.1
    assert A.DATASET_VERSION == "v20260804"


@needs_index
def test_every_registered_pool_is_boardable_and_reproduces_its_sha():
    """``run.all_pools`` must resolve every registered name, at the registered sha.

    A sha registered under a name the pool loader cannot resolve is an assertion
    that never runs (``_pool_ids`` would ``KeyError`` first), and a name that
    resolves to a drifted id set is the failure this registry exists to catch.
    ``train_full`` / ``all`` come straight from ``all_pools``, the same loader
    the ``ladder`` subcommand used for the §11 rows -- there is no second
    implementation here.
    """
    from q3vl.whatb.jetlut import run as R

    previous = C.S.active_dataset_version().name
    try:
        A.pin_dataset_version(force=True)
        registry = R.all_pools(902)
        for pool, (want_sha, want_n) in A.POOL_SHA.items():
            ids = registry[pool]                       # KeyError = unboardable
            rec = A.assert_pool_sha(pool, ids)         # raises on drift
            assert (rec["lut_ids_sha"], rec["n_lut"]) == (want_sha, want_n)
    finally:
        C.S.use_dataset_version(previous, force=True)


@needs_index
def test_pinned_dataset_version_reproduces_the_proposal_pools():
    """EPR-030 NOTES 6: the default口径 (cut-p45) gives 948/232, not 902/259."""
    previous = C.S.active_dataset_version().name
    try:
        A.pin_dataset_version(force=True)     # the suite may have read cut-p45
        assert C.S.active_dataset_version().name == "v20260804"
        tr = sorted(C.train_lut_ids())
        held = sorted(set(C.open_bank().lut_ids()) - set(tr))
        tlu = sorted(set(C.eval_only_lut_ids()["T_lut_unseen"]) & set(held))
        A.assert_pool_sha("held_out", held)          # raises on drift
        A.assert_pool_sha("t_lut_unseen", tlu)
        assert (len(held), len(tlu)) == (902, 259)
    finally:
        C.S.use_dataset_version(previous, force=True)


# --------------------------------------------------------------------------- #
# 6. the borrowed checkpoint
# --------------------------------------------------------------------------- #
@needs_ckpt
def test_shared_geometry_loads_with_the_expected_keys():
    geo, prov = A.load_shared_geometry()
    assert prov["n_gauss"] == 48
    assert set(geo) == {"mu", "chol_diag", "chol_off", "opacity_logit"}
    assert geo["mu"].shape == (48, 3)
    assert geo["opacity_logit"].shape == (48,)
    for k in A.SHARED_GEOMETRY_KEYS:
        assert len(prov["tensors"][k]["sha256"]) == 64
    assert 3 * (4 + 4 * prov["n_gauss"]) == 588


def test_load_shared_geometry_refuses_a_checkpoint_without_it(tmp_path):
    p = tmp_path / "no_geo.pt"
    torch.save({"state_dict": {"head.weight": torch.zeros(2, 2)}}, p)
    with pytest.raises(KeyError, match="shared geometry"):
        A.load_shared_geometry(p)


# --------------------------------------------------------------------------- #
# 7. the §10.2 reference ladder
# --------------------------------------------------------------------------- #
def test_reference_ladder_transcription_interpolates_on_log_p(tmp_path):
    ref = A.reference_ladder(tmp_path, "held_out")
    assert ref["lut_ids_sha"] == "d3f17955816f39b1"
    assert A.interp_reference(ref, 1, 336) == pytest.approx(2.5725)
    assert A.interp_reference(ref, 1, 4128) == pytest.approx(0.8966)
    mid = A.interp_reference(ref, 1, 588)
    assert 1.7511 < mid < 2.5725
    assert A.interp_reference(ref, 1, 100) is None          # outside the hull
    assert [r["P_dyn"] for r in A.bracket_rows(ref, 1, 588)] == [336, 780]


def test_reference_ladder_rejects_a_drifted_on_disk_ladder(tmp_path):
    (tmp_path / "ladder_held_out.json").write_text(
        '{"lut_ids_sha": "deadbeefdeadbeef", "rows": []}')
    with pytest.raises(AssertionError, match="lut_ids_sha"):
        A.reference_ladder(tmp_path, "held_out")


def test_reference_is_written_only_for_the_full_pool(tmp_path):
    """B3: an 8-LUT smoke row must not be printed next to a 902-LUT ladder."""
    ref, omitted = A.reference_or_omitted(tmp_path, "held_out", n_rows=902)
    assert omitted is None and ref["lut_ids_sha"] == "d3f17955816f39b1"
    assert A._ref_fields(ref, omitted) == {"reference_ladder": ref}


@pytest.mark.parametrize("pool,n_rows", [("held_out", 8),
                                         ("t_lut_unseen", 259 - 1),
                                         ("train_full", 4),
                                         ("train_full", 3149 - 1),
                                         ("all", 4),
                                         ("train", 8)])
def test_reference_is_omitted_when_the_row_pool_is_not_the_reference_pool(
        tmp_path, pool, n_rows):
    ref, omitted = A.reference_or_omitted(tmp_path, pool, n_rows=n_rows)
    assert ref is None
    assert omitted["n_lut_rows"] == n_rows and omitted["reason"]
    fields = A._ref_fields(ref, omitted)
    assert set(fields) == {"reference_omitted"}      # no reference, no derived


@pytest.mark.parametrize("pool,sha,n", [("train_full", "06da414b16ea0c6c", 3149),
                                        ("all", "f964eba67b9cdc43", 4051)])
def test_new_pools_read_their_on_disk_ladder_as_the_reference(tmp_path, pool,
                                                              sha, n):
    """§11's two full pools have no transcribed table: the reference is on disk.

    ``ladder_<pool>.json`` is accepted only after its ``lut_ids_sha`` matches the
    registered one, and only when the row pool is the full ``n``.
    """
    (tmp_path / f"ladder_{pool}.json").write_text(
        '{"lut_ids_sha": "%s", "c_star": 0.7, "rows": [' % sha
        + '{"p": 1, "P_dyn": 336, "fitgrid_pre_clamp": {"p95": 2.0}},'
          '{"p": 1, "P_dyn": 4128, "fitgrid_pre_clamp": {"p95": 1.0}}]}')
    ref, omitted = A.reference_or_omitted(tmp_path, pool, n_rows=n)
    assert omitted is None and ref["lut_ids_sha"] == sha
    assert [r["P_dyn"] for r in ref["rows"]] == [336, 4128]
    assert A.interp_reference(ref, 1, 336) == pytest.approx(2.0)
    # a drifted on-disk ladder for the same pool is refused
    (tmp_path / f"ladder_{pool}.json").write_text(
        '{"lut_ids_sha": "deadbeefdeadbeef", "rows": []}')
    with pytest.raises(AssertionError, match="lut_ids_sha"):
        A.reference_or_omitted(tmp_path, pool, n_rows=n)


def test_new_pools_have_no_transcribed_fallback(tmp_path):
    """Absent the on-disk ladder there is no §10.2 table for the new pools."""
    for pool in ("train_full", "all"):
        with pytest.raises(FileNotFoundError, match="no reference ladder"):
            A.reference_ladder(tmp_path, pool)


# --------------------------------------------------------------------------- #
# 8. the two dE00 calibers
# --------------------------------------------------------------------------- #
def _tiny_residuals(x, n_lut: int, seed: int = 4):
    """``(Q, 3L)`` in ``run.residuals``' layout: column ``3l + ch``."""
    g = torch.Generator().manual_seed(seed)
    r = 0.10 * torch.randn((x.shape[0], 3 * n_lut), generator=g,
                           dtype=torch.float64)
    rhat = r + 0.01 * torch.randn(r.shape, generator=g, dtype=torch.float64)
    return r, rhat


@pytest.mark.parametrize("clamp", [False, True])
def test_de00_paired_is_the_per_lut_pairing(clamp):
    """B1: column ``3l+ch`` is one LUT's RGB at one point, bit for bit."""
    from q3vl.whatb.colorimetry import delta_e00_srgb

    q = n_lut = 8
    torch.manual_seed(0)
    x = torch.rand((q, 3), dtype=torch.float64)
    xf = x.to(torch.float32)
    r, rhat = _tiny_residuals(x, n_lut)
    # ground truth: one LUT at a time, stacked -- no shape gymnastics at all
    y = torch.stack([xf + r[:, 3 * l:3 * l + 3].to(torch.float32)
                     for l in range(n_lut)], dim=0)
    yh = torch.stack([xf + rhat[:, 3 * l:3 * l + 3].to(torch.float32)
                      for l in range(n_lut)], dim=0)
    want = delta_e00_srgb(yh.clamp(0.0, 1.0) if clamp else yh, y).T
    assert torch.equal(A.de00_paired(x, r, rhat, clamp=clamp), want)


def test_de00_paired_differs_from_the_legacy_shape_taking():
    """The two calibers are not the same number on non-trivial input."""
    from q3vl.whatb.jetlut import run as R

    q = n_lut = 8
    torch.manual_seed(0)
    x = torch.rand((q, 3), dtype=torch.float64)
    r, rhat = _tiny_residuals(x, n_lut)
    paired = A.de00_paired(x, r, rhat, clamp=False)
    legacy = R.de00_of(x, r, rhat, clamp=False)
    assert paired.shape == legacy.shape == (q, n_lut)
    assert not torch.allclose(paired, legacy, atol=1e-3)
    # ...and the legacy take really is the transposed one, still bit-reproducible
    assert torch.equal(legacy, R.de00_of(x, r, rhat, clamp=False))
    # a per-LUT-constant residual makes the two agree: every triple the legacy
    # take assembles is then the same colour it should have been
    const = torch.cat([torch.full((q, 3), 0.05 * (l + 1), dtype=torch.float64)
                       for l in range(n_lut)], dim=-1)
    assert torch.allclose(A.de00_paired(x, const, const * 1.05, clamp=False),
                          R.de00_of(x, const, const * 1.05, clamp=False),
                          atol=1e-4)


def test_paired_de00_stats_emits_both_calibers():
    """B1 wiring: the arms rename run.py's blocks and add the paired twins."""
    from q3vl.whatb.jetlut import run as R

    x = _grid(5)
    atlas = Atlas(m=2, c=0.7)
    r, _ = _tiny_residuals(x, 1)
    build = lambda: design_matrix(x, atlas, 1)          # noqa: E731
    strata = {"all": torch.ones(x.shape[0], dtype=torch.bool)}
    rec = R.fit_and_score(x, x, r, r, build, build, {"arm": "T"}, max_iter=80,
                          strata=strata, return_theta=True)
    theta = rec.pop("_theta")
    A.paired_de00_stats(R, rec, x_fit=x, x_eval=x, r_fit=r, r_eval=r,
                        theta=theta, build_fit=build, build_eval=build,
                        strata=strata)
    for key in A.DE00_STAT_KEYS:
        assert key not in rec                          # renamed, not duplicated
        assert f"{key}_legacy" in rec and f"{key}_paired" in rec
    assert rec["fitgrid_pre_clamp_legacy"]["n_lut"] == 1
    assert rec["fitgrid_pre_clamp_paired"]["n_points"] == x.shape[0]
    assert set(rec["fitgrid_strata_paired"]) == {"all"}
    assert set(A.DE00_CALIBERS) == {"legacy", "paired"}


# --------------------------------------------------------------------------- #
# 9. pool-dimension chunking (``--lut-chunk``)
# --------------------------------------------------------------------------- #
def test_lut_bounds_default_is_the_whole_pool():
    """``0`` / negative / ``>= n`` all mean "do not split"."""
    for chunk in (0, -1, None, 8, 9, 1000):
        assert A.lut_bounds(8, chunk) == [(0, 8)]
    with pytest.raises(ValueError):
        A.lut_bounds(0, 3)


@pytest.mark.parametrize("n,chunk,want", [
    (8, 3, [(0, 3), (3, 6), (6, 8)]),          # boundary does not divide
    (8, 5, [(0, 5), (5, 8)]),
    (8, 1, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8)]),
    (9, 3, [(0, 3), (3, 6), (6, 9)]),          # boundary divides exactly
])
def test_lut_bounds_tiles_the_pool_in_registry_order(n, chunk, want):
    got = A.lut_bounds(n, chunk)
    assert got == want
    assert [i for lo, hi in got for i in range(lo, hi)] == list(range(n))
    assert all(hi - lo <= chunk for lo, hi in got)


def _tiny_fit_problem(n_lut: int = 8, seed: int = 11):
    """``(x, atlas, r, build, strata)`` -- a 125-point, ``n_lut``-LUT problem."""
    x = _grid(5)
    atlas = Atlas(m=2, c=0.7)
    g = torch.Generator().manual_seed(seed)
    base = torch.stack([0.20 * torch.sin(4.0 * x[:, 1]),
                        0.12 * x[:, 2] ** 2,
                        -0.15 * x[:, 0] * x[:, 2]], dim=-1)
    r = torch.cat([base * (0.5 + 0.2 * l)
                   + 0.02 * torch.randn(base.shape, generator=g,
                                        dtype=torch.float64)
                   for l in range(n_lut)], dim=-1)
    build = lambda: design_matrix(x, atlas, 1)              # noqa: E731
    strata = {"lo": x[:, 0] < 0.5, "hi": x[:, 0] >= 0.5}
    return x, atlas, r, build, strata


def _separable_solver(f, r, *, lam=1e-8, max_iter=0, **kw):
    """A column-separable stand-in for ``core.admm_lad``.

    Every column of ``theta`` is a reduction of the *same* column of ``r`` and of
    nothing else, so this solver is exactly what pool chunking assumes a solver
    to be: independent per LUT.  It is what makes the chunked-vs-whole-pool
    comparison below a test of the *aggregation* rather than of the ADMM -- the
    real ``core.admm_lad`` is not separable (its ``rho`` adapts on batch-global
    residual norms), which is pinned by
    ``test_admm_rho_schedule_is_batch_global`` below.
    """
    p = f.shape[1]
    theta = torch.zeros((p, r.shape[1]), dtype=f.dtype, device=f.device)
    # only per-column reductions that are exactly column-block invariant: a row
    # pick and ``amax``.  ``sum`` / ``mean`` over dim 0 are *not* (torch
    # vectorises across the column axis, so the row accumulation order is a
    # function of the batch width) -- see the tolerance split below.
    theta[0] = r[0] * (1.0 - 1000.0 * lam)           # lam-dependent: tie_probe
    theta[1] = 0.5 * r.amax(0)
    resid = f @ theta - r
    primal = resid.abs().sum(0)
    gap = 0.01 + r.abs().amax(0)                 # per column, not per position
    # r_pri / r_dual are Frobenius norms over the whole right-hand side, exactly
    # as ``core.admm_lad`` reports them -- that is what the sqrt-of-sum-of-
    # squares aggregation has to reproduce
    r_pri = float(resid.norm())
    return {"theta": theta, "primal": primal, "dual": primal - gap, "gap": gap,
            "dual_feas": 1e-9, "n_iter": 7, "rho": 100.0,
            "r_pri": r_pri, "r_dual": 0.5 * r_pri}


#: fields ``fit_score_pooled`` rebuilds by a single whole-pool reduction (or by
#: an integer count, or by a max), so a chunked run reproduces them exactly
#: (atol 0) when the solver is column-separable
_EXACT_SCALARS = ("l1_fit", "l1_fit_mean_per_point", "gamut_violation_rate",
                  "P_feat", "gap_max", "dual_feas", "admm_iters", "admm_rho")
#: fields aggregated arithmetically across blocks (weighted mean, Frobenius
#: recombination, ratio reconstruction): equal in exact arithmetic, and
#: summation-order-close in floating point
_APPROX_SCALARS = ("gap_mean", "r_pri", "r_dual", "tie_gap_rel")


@pytest.mark.parametrize("chunk", [1, 3, 5, 7])
def test_chunked_pool_reproduces_the_whole_pool_record(monkeypatch, chunk):
    """The hard criterion: ``--lut-chunk N`` == whole pool, field by field.

    Run with a column-separable solver, so any difference is the chunking's own
    aggregation and nothing else.  Every dE00 block (both calibers, both clamps,
    both grids, and the strata split) must be *bit* identical: they are rebuilt
    from the concatenated per-LUT ``(Q, L)`` field and reduced once, never
    combined from block-level statistics.
    """
    from q3vl.whatb.jetlut import run as R

    monkeypatch.setattr(R, "admm_lad", _separable_solver)
    x, atlas, r, build, strata = _tiny_fit_problem()
    kw = dict(x_fit=x, x_eval=x, r_fit=r, r_eval=r, build_fit=build,
              build_eval=build, meta={"arm": "T", "m": 2}, max_iter=5,
              strata=strata, tie_probe=True)
    whole, th_whole = A.fit_score_pooled(R, lut_chunk=0, **kw)
    part, th_part = A.fit_score_pooled(R, lut_chunk=chunk, **kw)

    assert A.lut_bounds(8, chunk) != [(0, 8)]            # the split really split
    assert [(lo, hi) for lo, hi, _ in th_part] == A.lut_bounds(8, chunk)
    assert torch.equal(torch.cat([t for _, _, t in th_part], dim=1),
                       th_whole[0][2])

    blocks = [k for k in whole if k.endswith(("_legacy", "_paired"))]
    assert len(blocks) == 10                             # 4 + 4 + 2 strata
    for k in blocks:
        assert part[k] == whole[k], k                    # atol 0
    for k in _EXACT_SCALARS:
        assert part[k] == whole[k], k                    # atol 0
    for k in _APPROX_SCALARS:
        assert part[k] == pytest.approx(whole[k], rel=1e-9, abs=1e-15), k
    # the per-block solver diagnostics are recorded, not silently pooled
    assert len(part["lut_chunk_solver"]) == len(A.lut_bounds(8, chunk))
    assert "lut_chunk_solver" not in whole
    assert set(whole) - set(part) == set()
    assert set(part) - set(whole) == {"lut_chunk_solver"}


def test_pooled_rhat_rebuilds_the_whole_pool_prediction(chunk=3):
    """The blocks are assembled into the pool field *before* any dE00 call."""
    x, atlas, r, build, _ = _tiny_fit_problem()
    f = build()
    torch.manual_seed(5)
    theta = 0.05 * torch.randn((f.shape[1], r.shape[1]), dtype=torch.float64)
    whole = A.pooled_rhat(f, [(0, 8, theta)], n_lut=8)
    split = A.pooled_rhat(f, [(lo, hi, theta[:, 3 * lo:3 * hi].contiguous())
                              for lo, hi in A.lut_bounds(8, chunk)], n_lut=8)
    assert torch.equal(whole, f @ theta)
    assert torch.equal(split, whole)


def test_dE00_is_not_batch_shape_invariant():
    """Why the assembly happens before the dE00 and not after.

    ``colorimetry.srgb_to_xyz``'s ``lin @ M^T`` picks a different kernel for a
    different leading batch, so evaluating dE00 on a 3-LUT block and on an 8-LUT
    block gives different last bits.  Calling ``de00_of`` per solver block would
    therefore make the reported statistic a function of ``--lut-chunk``; this is
    the same sensitivity ``de00_paired``'s ``.contiguous()`` already guards.
    """
    from q3vl.whatb.jetlut import run as R

    q, n_lut = 125, 8
    torch.manual_seed(0)
    x = torch.rand((q, 3), dtype=torch.float64)
    r, rhat = _tiny_residuals(x, n_lut)
    whole = R.de00_of(x, r, rhat, clamp=False)
    stitched = torch.cat(
        [R.de00_of(x, r[:, 3 * lo:3 * hi].contiguous(),
                   rhat[:, 3 * lo:3 * hi].contiguous(), clamp=False)
         for lo, hi in A.lut_bounds(n_lut, 3)], dim=1)
    assert torch.allclose(stitched, whole, atol=1e-3)
    assert not torch.equal(stitched, whole)


def test_chunked_per_s_stats_match_the_whole_pool():
    """J-4D's per-strength table is assembled over blocks, not synthesised."""
    from q3vl.whatb.jetlut import run as R

    x = _grid(4)
    atlas = A.Atlas4D(m=2, m_s=2, c=0.7)
    n_lut = 5
    torch.manual_seed(2)
    r = 0.1 * torch.randn((x.shape[0], 3 * n_lut), dtype=torch.float64)
    s_values = [0.0, 0.5, 1.0]
    f_cols = A.design_matrix_4d(A.stack_s(x, s_values), atlas).shape[1]
    theta = 0.05 * torch.randn((f_cols, 3 * n_lut), dtype=torch.float64)
    whole = A._per_s_stats(R, x, x, r, r, [(0, n_lut, theta)], atlas, s_values,
                           chunk=64)
    split = A._per_s_stats(
        R, x, x, r, r,
        [(lo, hi, theta[:, 3 * lo:3 * hi].contiguous())
         for lo, hi in A.lut_bounds(n_lut, 2)],
        atlas, s_values, chunk=64)
    assert set(whole) == {"0.0", "0.5", "1.0"}
    assert whole == split


def test_admm_rho_schedule_is_batch_global():
    """Why ``--lut-chunk`` is not free: the ADMM iterate is *not* separable.

    ``core.admm_lad`` adapts ``rho`` on ``r_pri`` / ``r_dual``, which are
    Frobenius norms over **all** right-hand-side columns, and stops on a
    batch-global tolerance.  The exact LAD minimiser is column-separable; the
    finite-iteration iterate this EPR reports is not, so a pool block's ``theta``
    is not the whole pool's ``theta`` restricted to that block.  Recorded here so
    a chunked artefact is never read as bit-comparable to a whole-pool one.
    """
    x = _grid(5)
    f = design_matrix(x, Atlas(m=2, c=0.7), 1)
    g = torch.Generator().manual_seed(11)
    # three near-zero LUTs next to five O(1) ones: the batch-global tolerance is
    # set by the loud columns, so the quiet ones stop somewhere else on their own
    r = torch.cat([(1e-4 if l < 3 else 1.0)
                   * torch.randn((x.shape[0], 3), generator=g,
                                 dtype=torch.float64) for l in range(8)], dim=-1)
    whole = admm_lad(f, r, max_iter=1000)
    block = admm_lad(f, r[:, 0:9].contiguous(), max_iter=1000)
    assert block["rho"] != whole["rho"]                  # different rho schedule
    assert not torch.equal(block["theta"], whole["theta"][:, 0:9])
    assert float((block["theta"] - whole["theta"][:, 0:9]).abs().max()) > 1e-6
    assert A.LUT_CHUNK_CONTRACT["not_preserved"].startswith(
        "the ADMM trajectory")


def test_lut_chunk_fields_records_the_plan():
    ids = [f"lut{i:02d}" for i in range(8)]
    assert A.lut_chunk_fields(ids, 0) == {"lut_chunk": 0}
    rec = A.lut_chunk_fields(ids, 3)
    assert rec["lut_chunk"] == 3
    agg = rec["lut_chunk_aggregation"]
    assert agg["n_lut"] == 8 and agg["n_blocks"] == 3
    assert agg["block_sizes"] == [3, 3, 2]
    assert agg["pool_sha"].startswith("asserted on the full pool")


# --------------------------------------------------------------------------- #
# 10. ``_dump`` does not silently replace a wider artefact with a narrower one
# --------------------------------------------------------------------------- #
def _rows(*cfgs) -> dict:
    return {"rows": [{"m": m, "order": o} for m, o in cfgs]}


def test_dump_moves_a_narrower_overwrite_aside(tmp_path):
    """2026-08-20: a ``--m 6`` run wrote over the m=3/4/5 artefact."""
    A._dump(tmp_path, "j4d_held_out.json", _rows((3, None), (4, None),
                                                 (5, None)))
    moved = A._dump(tmp_path, "j4d_held_out.json", _rows((6, None)),
                    ts="20260820-0301")
    assert moved == tmp_path / "j4d_held_out.overwritten-20260820-0301.json"
    kept = json.loads(moved.read_text())
    assert [r["m"] for r in kept["rows"]] == [3, 4, 5]
    assert [r["m"] for r in json.loads(
        (tmp_path / "j4d_held_out.json").read_text())["rows"]] == [6]


def test_dump_leaves_a_runs_own_incremental_dump_alone(tmp_path):
    """The drivers rewrite the whole file per row: old ⊆ new, so no rename."""
    name = "jp15_held_out.json"
    A._dump(tmp_path, name, _rows((3, "p1")))
    assert A._dump(tmp_path, name, _rows((3, "p1"), (3, "p15"))) is None
    assert A._dump(tmp_path, name,
                   _rows((3, "p1"), (3, "p15"), (4, "p1"))) is None
    assert list(p.name for p in tmp_path.iterdir()) == [name]


def test_dump_distinguishes_orders_at_the_same_m(tmp_path):
    """``(m, order)``, not ``m``: a p1-only rerun must not eat the p2 rows."""
    name = "jp15_held_out.json"
    A._dump(tmp_path, name, _rows((3, "p1"), (3, "p2")))
    moved = A._dump(tmp_path, name, _rows((3, "p1")), ts="t")
    assert moved is not None and moved.name.endswith(".overwritten-t.json")


def test_dump_leaves_files_without_rows_alone(tmp_path):
    """``*.DONE.json`` and the calib/gate artefacts carry no ``rows``."""
    A._dump(tmp_path, "j4d_held_out.DONE.json", {"pool": "held_out",
                                                 "n_rows": 3})
    assert A._dump(tmp_path, "j4d_held_out.DONE.json",
                   {"pool": "held_out", "n_rows": 1}) is None
    assert [p.name for p in tmp_path.iterdir()] == ["j4d_held_out.DONE.json"]


def test_dump_stamp_defaults_to_the_old_files_mtime(tmp_path):
    """The stamp is recoverable from the file, never from a wall clock."""
    import os
    import time as _time

    name = "j4d_held_out.json"
    A._dump(tmp_path, name, _rows((3, None)))
    when = _time.mktime((2026, 8, 20, 3, 1, 2, 0, 0, -1))
    os.utime(tmp_path / name, (when, when))
    moved = A._dump(tmp_path, name, _rows((6, None)))
    assert moved.name == "j4d_held_out.overwritten-20260820-030102.json"


def test_row_config_set_ignores_non_row_artifacts():
    assert A.row_config_set({"rows": [{"m": 3}, {"m": 4, "order": "p2"}]}) == {
        (3, None), (4, "p2")}
    assert A.row_config_set({"n_rows": 2}) is None
    assert A.row_config_set(None) is None
    assert A.row_config_set({"rows": "not a list"}) is None
