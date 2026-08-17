"""EPR-031 C0 + C1 unit tests.  **CPU only** -- nothing here touches a GPU.

Ten properties, one per pre-registered claim of the task card:

C0 (:mod:`q3vl.whatb.codec.lutcode`)
    1. the PCA fit set carries no ``lut_id`` of a LUT-disjoint evaluation split;
    2. the whitened PCA round-trips exactly at ``d_LUT = min(n-1, d)``;
    3. the three ``d_LUT`` are prefixes of ONE SVD, not three fits.

C1 (:mod:`q3vl.whatb.codec.tables`, :mod:`q3vl.whatb.scripts.run_whatcodec_arm`)
    4. paired anchors: four ``s`` per colour, ``0`` and ``1`` exact, the two
       interior ones complementary, ``n_pairs_s == L*Q*4``;
    5. mining selects whole colour groups;
    6. ``|Theta| == n_lut * n_params_g4d(N, mode)`` and two rows' gradients do
       not cross;
    7. ``R_line`` is (numerically) zero on ``A1``;
    8. the quick-eval NaN re-check exits non-zero and writes ``void_reason``;
    9. the S1 / S2 / S3 ladder resolves to the spec's numbers, digit for digit;
    10. ``--dry-run`` completes with no z cache, no images and no pred field.

Tests 1, 9 and 10 read the split index off the soft NFS mount and are skipped
when it is not present; everything else is self-contained.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from q3vl.whatb.arms import g4d
from q3vl.whatb.codec import lutcode as C
from q3vl.whatb.codec import tables as T
from q3vl.whatb.scripts import run_whatcodec_arm as R

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

_HAS_INDEX = (Path(__file__).exists()
              and C.S.split_index_path("train").is_file())
needs_index = pytest.mark.skipif(
    not _HAS_INDEX, reason="the split index is on the soft NFS mount")


# --------------------------------------------------------------------------- #
# 1. the fit set is clean
# --------------------------------------------------------------------------- #
@needs_index
def test_1_fit_set_has_no_lut_disjoint_eval_lut_id():
    fit = C.train_lut_ids("train")
    assert len(fit) == len(set(fit))
    report = C.assert_fit_set_clean(fit)              # raises on a violation
    assert report["T_lut_unseen"]["n_in_fit_set"] == 0
    # V_what / T_final are SAMPLE-level splits: their lut_id are drawn from the
    # same library and are a subset of train's.  Recorded here so the
    # "no eval LUT in the PCA" claim can never be read as covering them.
    for split in ("V_what", "T_final"):
        n_split = report[split]["n_split_ids"]
        assert report[split]["n_in_fit_set"] == n_split


def test_1b_fit_set_guard_raises_on_a_lut_disjoint_id():
    unseen = C.eval_only_lut_ids()["T_lut_unseen"] if _HAS_INDEX else []
    if not unseen:
        pytest.skip("the split index is on the soft NFS mount")
    with pytest.raises(AssertionError):
        C.assert_fit_set_clean(["rcp_not_a_real_id", unseen[0]])


# --------------------------------------------------------------------------- #
# 2. the whitened PCA round-trips at full rank
# --------------------------------------------------------------------------- #
def test_2_whitened_pca_round_trip_is_exact_at_full_rank():
    rng = np.random.default_rng(20260816)
    n, d = 12, 20
    x = rng.normal(size=(n, d)) @ rng.normal(size=(d, d))
    pca = C.fit_whitened_pca(x)                       # k = min(n-1, d) = 11
    assert pca.n_components == min(n - 1, d)
    code = pca.transform(x)
    back = pca.inverse_transform(code)
    assert np.abs(back - x).max() < 1e-9
    # whitening is what the name says: unit variance per canonical coordinate
    assert np.abs(code.var(axis=0, ddof=1) - 1.0).max() < 1e-9
    # the ratios are reported over the full rank, so they sum to 1
    assert abs(float(pca.explained_variance_ratio.sum()) - 1.0) < 1e-12


def test_2b_more_components_than_rank_is_an_error():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(6, 30))
    with pytest.raises(ValueError):
        C.fit_whitened_pca(x, n_components=6)         # min(n-1, d) = 5


def test_2c_cumulative_dims_is_the_smallest_prefix():
    evr = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
    got = C.cumulative_dims(evr, (0.90, 0.95, 0.99))
    assert got == {"90": 3, "95": 3, "99": 4}


# --------------------------------------------------------------------------- #
# 3. one SVD, three d_LUT
# --------------------------------------------------------------------------- #
def test_3_the_three_d_lut_are_prefixes_of_one_svd():
    rng = np.random.default_rng(31)
    x = rng.normal(size=(300, 400))
    pca = C.fit_whitened_pca(x, n_components=256)
    c256 = pca.transform(x, 256)
    for k in (128, 192):
        ck = pca.transform(x, k)
        assert ck.shape == (300, k)
        # bit-for-bit, not "close": a prefix is a slice, never a second fit
        assert np.array_equal(ck, c256[:, :k])
    assert np.array_equal(pca.components[:128], pca.components[:256][:128])


# --------------------------------------------------------------------------- #
# 4. paired anchors
# --------------------------------------------------------------------------- #
def _sampler(q: int):
    from q3vl.whatb import queries

    return queries.QuerySampler(seed=20260810, q=q)


@pytest.mark.parametrize("b,q", [(1, 8), (4, 5), (3, 1)])
def test_4_paired_anchors_are_four_per_colour(b, q):
    x, s = T.paired_anchor_batch(_sampler(q), b, q, device="cpu")
    assert x.shape == (b, q * T.S_ANCHORS, 3)
    assert s.shape == (b, q * T.S_ANCHORS)
    ss = s.reshape(b, q, T.S_ANCHORS)
    assert torch.equal(ss[..., 0], torch.zeros_like(ss[..., 0]))
    assert torch.equal(ss[..., 1], torch.ones_like(ss[..., 1]))
    assert torch.allclose(ss[..., 2] + ss[..., 3], torch.ones_like(ss[..., 2]),
                          atol=1e-6)
    # the four anchors sit on the SAME colour
    xx = x.reshape(b, q, T.S_ANCHORS, 3)
    for a in range(1, T.S_ANCHORS):
        assert torch.equal(xx[:, :, a, :], xx[:, :, 0, :])
    facts = T.assert_anchor_structure(s, b, q)
    assert facts["n_pairs_s"] == b * q * T.S_ANCHORS


def test_4b_n_pairs_s_equals_l_q_4_in_the_carrier_aux():
    b, q = 3, 7
    cfg = g4d.G4DConfig(mode="A1", n_gauss=8)
    table = T.DirectCarrierTable([f"l{i}" for i in range(b)], cfg)
    x, s = T.paired_anchor_batch(_sampler(q), b, q, device="cpu")
    _y, aux = g4d.Glut4DCarrier(cfg)(x, s, table.params_for(table.lut_ids),
                                     return_aux=True)
    assert aux.n_pairs_s == b * q * T.S_ANCHORS


def test_4c_a_broken_anchor_set_is_caught():
    b, q = 2, 3
    s = torch.rand(b, q, T.S_ANCHORS).reshape(b, q * T.S_ANCHORS)
    with pytest.raises(AssertionError):
        T.assert_anchor_structure(s, b, q)


# --------------------------------------------------------------------------- #
# 5. mining granularity
# --------------------------------------------------------------------------- #
def test_5_mining_selects_whole_colour_groups():
    b, q = 2, 10
    err = torch.zeros(b, q, T.S_ANCHORS)
    # make three colours of row 0 and two of row 1 unambiguously the worst
    hot = [(0, 1), (0, 4), (0, 7), (1, 2), (1, 9)]
    for i, j in hot:
        err[i, j] = 10.0 + i + j
    keep = T.select_hard_colour_groups(err.reshape(b, q * T.S_ANCHORS),
                                       len(hot) / (b * q), b, q)
    kk = keep.reshape(b, q, T.S_ANCHORS)
    for i in range(b):
        for j in range(q):
            group = kk[i, j]
            # all four in, or all four out -- never a split group
            assert bool(group.all()) or not bool(group.any())
    for i, j in hot:
        assert bool(kk[i, j].all())
    assert int(kk[..., 0].sum()) == len(hot)


def test_5b_ratio_zero_keeps_nothing():
    b, q = 2, 4
    err = torch.rand(b, q * T.S_ANCHORS)
    assert int(T.select_hard_colour_groups(err, 0.0, b, q).sum()) == 0


# --------------------------------------------------------------------------- #
# 6. the table
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode,n", [("A1", 48), ("A3", 48), ("A1", 8), ("A3", 8)])
def test_6_theta_parameter_count_is_exactly_n_lut_times_n_params(mode, n):
    ids = [f"lut{i}" for i in range(5)]
    cfg = g4d.G4DConfig(mode=mode, n_gauss=n)
    table = T.DirectCarrierTable(ids, cfg)
    want = len(ids) * g4d.n_params_g4d(n, mode)
    assert sum(p.numel() for p in table.parameters()) == want
    assert table.config()["n_params_total"] == want
    assert table.theta_dim == g4d.n_params_g4d(n, mode)


def test_6b_rows_do_not_share_gradients():
    ids = ["a", "b", "c"]
    cfg = g4d.G4DConfig(mode="A3", n_gauss=8)
    table = T.DirectCarrierTable(ids, cfg)
    carrier = g4d.Glut4DCarrier(cfg)
    x = torch.rand(1, 16, 3)
    s = torch.rand(1, 16)
    y = carrier(x, s, table.params_for(["b"]))
    y.abs().mean().backward()
    grad = table.theta.grad
    assert grad is not None
    assert float(grad[1].abs().sum()) > 0.0
    assert float(grad[0].abs().sum()) == 0.0
    assert float(grad[2].abs().sum()) == 0.0


def test_6c_every_row_starts_from_the_same_initial_vector():
    cfg = g4d.G4DConfig(mode="A3", n_gauss=48, init_seed=1234)
    table = T.DirectCarrierTable([f"l{i}" for i in range(4)], cfg)
    for i in range(1, table.n_lut):
        assert torch.equal(table.theta.data[i], table.theta.data[0])
    assert torch.equal(table.theta.data[0],
                       T.initial_theta("A3", 48, init_seed=1234))


def test_6d_the_initialisation_is_the_r1_8_4_one():
    n = 48
    cfg = g4d.G4DConfig(mode="A3", n_gauss=n)
    table = T.DirectCarrierTable(["a"], cfg)
    p = table.params_for(["a"])
    assert torch.allclose(p.opacity_logit,
                          torch.full_like(p.opacity_logit,
                                          g4d.OPACITY_LOGIT_INIT))
    diag = torch.nn.functional.softplus(p.chol_diag)
    assert torch.allclose(diag, torch.full_like(diag, g4d.SIGMA_RGB_INIT),
                          atol=1e-6)
    assert torch.equal(p.chol_off, torch.zeros_like(p.chol_off))
    assert torch.equal(p.beta, torch.zeros_like(p.beta))
    tau = g4d.tau_from_raw(p.tau_raw)
    assert torch.allclose(tau, torch.full_like(tau, g4d.TAU_INIT), atol=1e-6)
    lo, hi = g4d.MU_S_INIT_RANGE
    assert float(p.mu_s.min()) >= lo and float(p.mu_s.max()) <= hi
    assert torch.equal(p.g_matrix, torch.zeros_like(p.g_matrix))
    assert torch.equal(p.g_bias, torch.zeros_like(p.g_bias))
    eye = torch.eye(3).expand_as(p.m_local)
    assert torch.equal(p.m_local, eye)
    assert torch.equal(p.b_local, torch.zeros_like(p.b_local))
    # ... and therefore f(x, s) == x at step 0
    x = torch.rand(1, 64, 3)
    y = g4d.Glut4DCarrier(cfg)(x, torch.rand(1, 64), p)
    assert float((y - x).abs().max()) < 1e-4


def test_6e_an_unknown_lut_id_cannot_be_trained_by_accident():
    table = T.DirectCarrierTable(["a"], g4d.G4DConfig(mode="A1", n_gauss=8))
    with pytest.raises(KeyError):
        table.params_for(["zzz"])


# --------------------------------------------------------------------------- #
# 7. R_line on A1
# --------------------------------------------------------------------------- #
def test_7_r_line_is_zero_on_a1():
    b, q = 2, 32
    cfg = g4d.G4DConfig(mode="A1", n_gauss=16)
    table = T.DirectCarrierTable(["a", "b"], cfg)
    # move away from the identity initialisation: the claim is about A1's form
    # (f(x,s) = x + s[T(x) - x] is affine in s), not about a trivial transform
    with torch.no_grad():
        table.theta.add_(0.1 * torch.randn_like(table.theta))
    x, s = T.paired_anchor_batch(_sampler(q), b, q, device="cpu")
    y = g4d.Glut4DCarrier(cfg)(x, s, table.params_for(table.lut_ids))
    assert float(T.r_line_from_anchors(y, s, b, q)) < 1e-6


# --------------------------------------------------------------------------- #
# 8. the quick-eval NaN re-check
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_8_a_non_finite_l_rec_stops_the_run_and_writes_void_reason(tmp_path, bad):
    with pytest.raises(SystemExit) as exc:
        R.assert_l_rec_finite(bad, step=7, where="quick_eval@step7",
                              run_dir=tmp_path)
    assert exc.value.code == R.RC_NONFINITE != 0
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["void"] is True
    assert metrics["void_reason"] == "L_rec is not finite"
    assert metrics["step"] == 7 and metrics["rc"] == R.RC_NONFINITE
    assert metrics["published"] is False and metrics["oracle_reference"] is True


def test_8b_a_finite_l_rec_passes_and_writes_nothing(tmp_path):
    assert R.assert_l_rec_finite(0.25, step=1, where="w", run_dir=tmp_path) == 0.25
    assert not (tmp_path / "metrics.json").exists()
    # "not measured" is a different failure and is NOT this guard's job
    assert R.assert_l_rec_finite(None, step=1, where="w", run_dir=tmp_path) is None
    assert not (tmp_path / "metrics.json").exists()


def test_8c_the_recheck_is_wired_into_every_quick_eval():
    """The guard is CALLED, not merely defined (five silent misses so far)."""
    import inspect

    src = inspect.getsource(R.main)
    hook = src[src.index("def eval_hook"):src.index("train_facts")]
    assert hook.count("assert_l_rec_finite") >= 2, (
        "a quick eval must re-check both the training L_rec and the grid score")
    assert "assert_l_rec_finite" in inspect.getsource(R.train)


# --------------------------------------------------------------------------- #
# 9. the stage ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("stage,pool,per_step,steps", [
    ("S1", 1, 8192, 2000),
    ("S2", 32, 65536, 4000),
    ("S3", 3149, 2097152, 18760),
])
def test_9_stage_ladder_matches_the_spec_table(stage, pool, per_step, steps):
    spec = T.resolve_stage(stage, n_train_luts=3149)
    assert spec.n_lut_pool == pool
    assert spec.colors_per_step == per_step
    assert spec.total_steps == steps
    assert spec.s_anchors == 4
    assert spec.b_luts * spec.q_colors * 4 == per_step


def test_9b_the_ladder_numbers_are_the_documented_factorisation():
    assert T.resolve_stage("S1", n_train_luts=3149).b_luts == 1
    assert T.resolve_stage("S1", n_train_luts=3149).q_colors == 2048
    assert T.resolve_stage("S2", n_train_luts=3149).b_luts == 32
    assert T.resolve_stage("S2", n_train_luts=3149).q_colors == 512
    assert T.resolve_stage("S3", n_train_luts=3149).b_luts == 256
    assert T.resolve_stage("S3", n_train_luts=3149).q_colors == 2048
    assert T.resolve_stage("S1", n_train_luts=3149).mining is False
    assert T.resolve_stage("S2", n_train_luts=3149).mining is True


@needs_index
def test_9c_s3_pool_is_the_measured_train_pool():
    n = len(C.train_lut_ids("train"))
    assert T.resolve_stage("S3", n_train_luts=n).n_lut_pool == n


# --------------------------------------------------------------------------- #
# 10. dry-run without z cache / images / pred field
# --------------------------------------------------------------------------- #
@needs_index
@pytest.mark.parametrize("stage,carrier", [("S1", "A1"), ("S2", "A1"),
                                           ("S2", "A3")])
def test_10_dry_run_needs_no_zcache_no_image_no_pred_field(tmp_path, stage,
                                                           carrier):
    out = tmp_path / f"{stage}_{carrier}"
    rc = R.main(["--stage", stage, "--carrier", carrier, "--device", "cpu",
                 "--out", str(out), "--dry-run"])
    assert rc == 0
    setup = json.loads((out / "run_setup.json").read_text(encoding="utf-8"))
    assert setup["dry_run"] is True
    assert setup["published"] is False and setup["oracle_reference"] is True
    assert setup["data"]["needs_zcache"] is False
    assert setup["data"]["needs_images"] is False
    assert setup["data"]["needs_pred_field"] is False
    assert setup["conditioner"].startswith("none")
    assert setup["stage"]["stage"] == stage
    assert setup["carrier"]["mode"] == carrier
    assert setup["table"]["theta_dim"] == g4d.n_params_g4d(48, carrier)


# --------------------------------------------------------------------------- #
# extras: the geometric read-out and the residual matrix
# --------------------------------------------------------------------------- #
def test_code_recon_de00_is_zero_at_full_rank():
    rng = np.random.default_rng(11)
    n, p = 9, 25
    grid = rng.random((p, 3))
    residual = rng.normal(scale=0.05, size=(n, p * 3))
    pca = C.fit_whitened_pca(residual)
    got = C.code_recon_de00(pca, residual, grid, d_lut=pca.n_components)
    assert got["mean"] < 1e-8
    assert got["n_lut"] == n and got["n_points"] == p


def test_code_recon_de00_decreases_with_d_lut():
    rng = np.random.default_rng(12)
    n, p = 40, 25
    grid = rng.random((p, 3))
    basis = rng.normal(size=(10, p * 3))
    residual = rng.normal(size=(n, 10)) @ basis * 0.02
    pca = C.fit_whitened_pca(residual)
    means = [C.code_recon_de00(pca, residual, grid, d_lut=k)["mean"]
             for k in (2, 4, 8)]
    assert means[0] > means[1] > means[2]
    assert all(math.isfinite(m) for m in means)
