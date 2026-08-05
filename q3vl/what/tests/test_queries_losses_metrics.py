"""Protocol 9.1-9.5 and 12 -- query sampling, every loss term, the gates."""

from __future__ import annotations

import math

import pytest
import torch

from q3vl.what.colorspace import srgb_to_lab_norm
from q3vl.what.config import (
    CHARBONNIER_EPS,
    LOSS_WEIGHTS,
    LutConfig,
    N_QUERY_NATURAL,
    N_QUERY_TOTAL,
    N_QUERY_UNIFORM,
    N_SLOTS,
    STYLE_QUEUE_MIN,
    UNIFORM_STRATA,
)
from q3vl.what.gaussians import PRIM_LAYOUT_FG, anchor_points, decode_global, decode_primitives
from q3vl.what.srht import u_of_table
from q3vl.what.losses import (
    StyleQueue,
    pairwise_d_func,
    bake_readback,
    binary_entropy,
    charbonnier,
    compute_loss,
    grad_norm_ratio,
    loss_bake,
    loss_func,
    loss_hue_chroma,
    loss_sparse,
    loss_style_cos,
    loss_style_dist,
    loss_var_cov,
)
from q3vl.what.metrics import (
    activation_stats,
    bake_metrics,
    boundary_band,
    compose_image,
    effective_rank,
    evaluate_gates,
    image_metrics,
    lexicographic_best,
    lut_metrics,
    psnr,
    spearman,
    ssim,
    style_diagnostics,
)
from q3vl.what.queries import natural_query_points, query_kind_index, sample_seed, uniform_query_points

CFG = LutConfig()


def _params(batch=2, scale=0.3, seed=0):
    torch.manual_seed(seed)
    p = decode_primitives(torch.randn(batch, N_SLOTS, 23) * scale, CFG,
                          PRIM_LAYOUT_FG, anchors=anchor_points())
    p.update(decode_global(torch.randn(batch, 12) * scale, CFG))
    return p


# --- protocol 9.1: query points ---------------------------------------------

def test_uniform_set_is_fixed_stratified_and_covers_the_cube():
    a, b = uniform_query_points(), uniform_query_points()
    assert a.shape == (N_QUERY_UNIFORM, 3)
    assert torch.equal(a, b)                              # cached, identical
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0
    # exactly 2 points per 8^3 stratum
    idx = (a * UNIFORM_STRATA).floor().clamp(max=UNIFORM_STRATA - 1).long()
    flat = (idx[:, 0] * UNIFORM_STRATA + idx[:, 1]) * UNIFORM_STRATA + idx[:, 2]
    counts = torch.bincount(flat, minlength=UNIFORM_STRATA ** 3)
    assert int(counts.min()) == 2 and int(counts.max()) == 2


def test_natural_points_come_from_the_image_and_follow_the_mask():
    torch.manual_seed(0)
    img = torch.rand(3, 16, 16)
    # a mask concentrated on one pixel must return that pixel's colour only
    m = torch.zeros(16, 16)
    m[5, 7] = 1.0
    pts = natural_query_points(img, 64, weights=m, seed=1)
    assert torch.allclose(pts, img[:, 5, 7].expand(64, 3), atol=1e-6)
    # global sampling (weights=None) returns colours that exist in the image
    flat = img.reshape(3, -1).t()
    pts = natural_query_points(img, 128, weights=None, seed=2)
    assert all(bool((flat == p).all(-1).any()) for p in pts[:16])


def test_natural_sampling_is_deterministic_in_the_sample_seed():
    img = torch.rand(3, 8, 8)
    s = sample_seed(7, "sft_abc")
    assert s == sample_seed(7, "sft_abc")
    assert s != sample_seed(7, "sft_abd")
    assert torch.equal(natural_query_points(img, 32, seed=s),
                       natural_query_points(img, 32, seed=s))


def test_a_zero_mask_falls_back_to_the_whole_image_instead_of_crashing():
    img = torch.rand(3, 8, 8)
    pts = natural_query_points(img, 16, weights=torch.zeros(8, 8), seed=3)
    assert pts.shape == (16, 3) and torch.isfinite(pts).all()


def test_query_kind_index_splits_50_50():
    k = query_kind_index()
    assert k.shape == (N_QUERY_TOTAL,)
    assert int((k == 0).sum()) == N_QUERY_UNIFORM
    assert int((k == 1).sum()) == N_QUERY_NATURAL


# --- protocol 9.1/9.2: the two main terms ------------------------------------

def test_charbonnier_is_the_smooth_l1_and_reduces_to_abs():
    d = torch.tensor([0.0, 0.5, -0.5])
    want = torch.sqrt(d ** 2 + CHARBONNIER_EPS ** 2)
    assert torch.allclose(charbonnier(d), want)
    assert abs(float(charbonnier(torch.tensor(0.5))) - 0.5) < 1e-5


def test_loss_func_reports_the_two_halves_separately():
    torch.manual_seed(1)
    a = torch.rand(2, N_QUERY_TOTAL, 3)
    b = a.clone()
    b[:, :N_QUERY_UNIFORM] += 0.2               # error only in the uniform half
    out = loss_func(b, a, query_kind_index())
    assert float(out["L_func_natural"]) < 1e-2
    assert float(out["L_func_uniform"]) > 0.15
    assert abs(float(out["L_func"])
               - 0.5 * float(out["L_func_uniform"] + out["L_func_natural"])) < 1e-5


def test_hue_chroma_is_zero_for_identical_inputs_and_weighted_by_gt_chroma():
    torch.manual_seed(2)
    a = torch.rand(2, 256, 3)
    assert float(loss_hue_chroma(a, a)) < 1e-6
    # grey GT (chroma ~ 0) contributes ~nothing however wrong the prediction is
    grey = torch.full((1, 64, 3), 0.5)
    wrong = torch.rand(1, 64, 3)
    assert float(loss_hue_chroma(wrong, grey)) < 1e-3
    # a saturated GT does contribute
    red = torch.tensor([[1.0, 0.0, 0.0]]).expand(1, 64, 3).contiguous()
    green = torch.tensor([[0.0, 1.0, 0.0]]).expand(1, 64, 3).contiguous()
    assert float(loss_hue_chroma(green, red)) > 0.3


def test_hue_chroma_matches_a_hand_computed_value():
    """One pair, computed from the definition with atan2 and the chroma weight."""
    pred = torch.tensor([[[0.9, 0.2, 0.3]]])
    gt = torch.tensor([[[0.2, 0.7, 0.4]]])
    lp, lg = srgb_to_lab_norm(pred), srgb_to_lab_norm(gt)
    hp = math.atan2(float(lp[..., 2]), float(lp[..., 1]))
    hg = math.atan2(float(lg[..., 2]), float(lg[..., 1]))
    c = math.hypot(float(lg[..., 1]), float(lg[..., 2])) / math.sqrt(2.0)
    want = c * (1.0 - math.cos(hp - hg))
    assert abs(float(loss_hue_chroma(pred, gt)) - want) < 1e-5


def test_grad_norm_ratio_detects_the_a0_unit_error():
    """With raw Lab the weighted hue term dwarfs L_func; with normalised Lab it
    does not.  This is the number protocol 9.2 orders into the training log."""
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.zeros(1, 1, 3))
    gt = torch.rand(1, 512, 3)
    pred = (torch.rand(1, 512, 3) + p).clamp(0, 1)
    r = grad_norm_ratio([p], loss_func(pred, gt)["L_func"], loss_hue_chroma(pred, gt))
    assert math.isfinite(r["grad_ratio_hc_over_func"])
    assert r["grad_norm_L_func"] > 0.0 and r["grad_norm_w_L_hc"] > 0.0
    assert r["grad_ratio_hc_over_func"] < 100.0


# --- protocol 9.3: sparsity and the style code -------------------------------

def test_binary_entropy_and_sparsity():
    assert abs(float(binary_entropy(torch.tensor(0.5))) - math.log(2)) < 1e-6
    assert float(binary_entropy(torch.tensor(1.0))) < 1e-4
    p = {"opacity": torch.full((2, N_SLOTS), 0.5),
         "existence": torch.full((2, N_SLOTS), 0.5)}
    assert abs(float(loss_sparse(p)) - math.log(2)) < 1e-5
    hard = {"opacity": torch.full((2, N_SLOTS), 1.0 - 1e-6),
            "existence": torch.full((2, N_SLOTS), 1e-6)}
    assert float(loss_sparse(hard)) < 1e-4


def test_style_cos_term():
    z = torch.nn.functional.normalize(torch.randn(4, 32), dim=-1)
    assert float(loss_style_cos(z, z)) < 1e-6
    assert float(loss_style_cos(z, -z)) > 1.9


def test_d_func_is_the_raw_u_distance_not_the_normalised_code(monkeypatch):
    """Amendment A-2 / review blocker B-3.

    The failure the old implementation could not see: two LUTs that are the same
    look at different strengths.  Their ``u`` differ by a large factor, but their
    ``z_gt`` -- being L2-normalised -- are identical, so a ``d_func`` built on
    ``z_gt`` reports distance 0 and nothing in ``L_what`` supervises magnitude.
    """
    from q3vl.what.srht import encode_z_gt, identity_grid

    torch.manual_seed(0)
    grid = identity_grid()
    delta = torch.randn_like(grid) * 0.05
    # the same look at two strengths: T_s(x) = x + s * delta(x), so u_s = s * delta
    weak, strong = grid + 0.2 * delta, grid + 1.0 * delta
    u = torch.stack([u_of_table(weak.unsqueeze(0))[0],
                     u_of_table(strong.unsqueeze(0))[0]])
    assert torch.allclose(u[1], 5.0 * u[0], atol=1e-5)      # a pure scaling
    # the raw function distance sees it
    assert float(pairwise_d_func(u, 1.0)[0]) > 0.5 * float(u[1].norm())
    # the direction-only code does not: L2 normalisation removes the scale
    z_gt = encode_z_gt(torch.stack([weak, strong]))
    assert float((z_gt[0] - z_gt[1]).norm()) < 1e-4


def test_style_dist_matches_a_hand_computed_huber():
    torch.manual_seed(1)
    z = torch.randn(3, 16)
    u = torch.randn(3, 64)
    c = 2.5
    got = loss_style_dist(z, u, c)
    zs = torch.nn.functional.normalize(z, dim=-1)
    ds, df = [], []
    for i in range(3):
        for j in range(i + 1, 3):
            ds.append(float((zs[i] - zs[j]).norm()))
            df.append(float((u[i] - u[j]).norm() / c))
    want = torch.nn.functional.huber_loss(torch.tensor(ds), torch.tensor(df), delta=1.0)
    assert abs(float(got) - float(want)) < 1e-5


def test_style_dist_is_zero_for_a_batch_of_one_and_rejects_a_bad_scale():
    z = torch.randn(1, 16)
    assert float(loss_style_dist(z, torch.randn(1, 64), 1.0)) == 0.0
    with pytest.raises(ValueError):
        loss_style_dist(torch.randn(3, 16), torch.randn(3, 64), 0.0)


def test_style_queue_is_fifo_bounded_and_stop_gradient():
    q = StyleQueue(size=4, min_size=1)
    for i in range(3):
        q.push(torch.full((2, 8), float(i), requires_grad=True))
    assert len(q) == 4                                      # bounded
    st = q.stack()
    assert st.shape == (4, 8) and not st.requires_grad
    assert float(st[0, 0]) == 1.0                           # oldest dropped
    assert q.facts()["size"] == 4


def test_var_cov_penalises_a_collapsed_code_and_not_a_spread_one():
    q = StyleQueue(size=64, min_size=1)
    collapsed = torch.zeros(8, 16, requires_grad=True)
    spread = torch.randn(8, 16, requires_grad=True) * 3.0
    a = loss_var_cov(collapsed, q)
    b = loss_var_cov(spread, q)
    assert float(a["L_var"]) > float(b["L_var"])
    assert float(a["L_var"]) > 0.9                          # max(0, 1 - 0) = 1
    assert torch.autograd.grad(a["L_var"], collapsed, allow_unused=True)[0] is not None


def test_var_cov_is_silent_until_the_queue_warms_up():
    q = StyleQueue(size=64, min_size=STYLE_QUEUE_MIN)
    out = loss_var_cov(torch.randn(2, 16), q)
    assert float(out["L_var"]) == 0.0 and float(out["L_cov"]) == 0.0


# --- protocol 9.4: bake consistency ------------------------------------------

def test_bake_loss_is_zero_for_an_affine_function():
    from dataclasses import replace

    cfg = replace(CFG, clamp_output=False)
    p = decode_primitives(torch.zeros(1, N_SLOTS, 23), cfg, PRIM_LAYOUT_FG,
                          anchors=anchor_points())
    p.update(decode_global(torch.randn(1, 12) * 0.3, cfg))
    x = torch.rand(1, 512, 3)
    from q3vl.what.gaussians import render

    t = render(p, x, cfg)
    assert float(loss_bake(t, bake_readback(p, cfg, x))) < 2e-3     # == charbonnier eps


def test_bake_readback_is_differentiable():
    p = _params(1, 0.3, seed=4)
    for v in p.values():
        v.requires_grad_(True)
    x = torch.rand(1, 128, 3)
    bake_readback(p, CFG, x, size=9).sum().backward()
    assert torch.isfinite(p["mu"].grad).all()


# --- protocol 9.5: the total -------------------------------------------------

def test_compute_loss_uses_the_frozen_weights_and_has_no_i_tar():
    import inspect

    sig = set(inspect.signature(compute_loss).parameters)
    assert not any("tar" in s or "image" in s for s in sig)
    p = _params(2, 0.3, seed=5)
    x = torch.rand(2, 64, 3)
    t_gt = torch.rand(2, 64, 3)
    from q3vl.what.gaussians import render

    t_pred = render(p, x, CFG)
    z_style = torch.randn(2, 32, requires_grad=True)
    z_gt = torch.nn.functional.normalize(torch.randn(2, 32), dim=-1)
    out = compute_loss(t_pred=t_pred, t_gt=t_gt, x=x, params=p, z_style=z_style,
                       z_gt=z_gt, u_gt=torch.randn(2, 128), d_func_scale=2.0,
                       lut_cfg=CFG, t_read=bake_readback(p, CFG, x, size=9))
    manual = sum(LOSS_WEIGHTS[k] * float(out.parts[k])
                 for k in ("L_func", "L_hc", "R_sparse", "L_style_cos", "L_style_dist",
                           "L_bake"))
    manual += LOSS_WEIGHTS["L_varcov"] * float(out.parts["L_var"] + out.parts["L_cov"])
    assert abs(float(out.total) - manual) < 1e-5
    assert torch.isfinite(out.total)


# --- protocol 12: metrics and gates ------------------------------------------

def test_lut_metrics_are_zero_for_a_perfect_prediction():
    a = torch.rand(4, 128, 3)
    m = lut_metrics(a, a)
    assert m["mae"] < 1e-6 and m["de00_mean"] < 1e-3 and m["non_finite"] == 0
    assert m["psnr"] > 100.0


def test_bake_gate_is_the_protocol_12_1_gate():
    good = {"bake_mae_mean": 5e-5, "bake_err_p99": 1e-4, "bake_non_finite": 0.0,
            "lut_non_finite": 0.0}
    assert evaluate_gates(good)["all_pass"]
    bad = dict(good, bake_err_p99=1e-3)
    res = evaluate_gates(bad)
    assert not res["all_pass"]
    assert [r["metric"] for r in res["rows"] if not r["pass"]] == ["bake_err_p99"]
    missing = evaluate_gates({})
    assert not missing["all_pass"]
    assert all(r["reason"] == "missing" for r in missing["rows"])


def test_bake_metrics_shape():
    a = torch.rand(2, 64, 3)
    m = bake_metrics(a, a + 1e-5)
    assert abs(m["bake_mae_mean"] - 1e-5) < 1e-6 and m["bake_non_finite"] == 0.0


def test_ssim_and_psnr_are_perfect_on_identical_images():
    torch.manual_seed(6)
    a = torch.rand(3, 32, 32)
    assert abs(ssim(a, a) - 1.0) < 1e-4
    assert psnr(a, a) > 100.0


def test_compose_image_is_the_protocol_formula_and_respects_the_mask():
    i_in = torch.rand(3, 8, 8)
    t = torch.rand(3, 8, 8)
    m = torch.zeros(8, 8)
    m[:4] = 1.0
    out = compose_image(i_in, t, m)
    assert torch.allclose(out[:, 4:], i_in[:, 4:], atol=1e-6)
    assert torch.allclose(out[:, :4], t[:, :4].clamp(0, 1), atol=1e-6)


def test_image_metrics_partition_covers_the_frame():
    torch.manual_seed(7)
    a, b = torch.rand(3, 32, 32), torch.rand(3, 32, 32)
    m = torch.zeros(32, 32)
    m[8:24, 8:24] = 1.0
    r = image_metrics(a, b, m)
    total = r["inside_frac"] + r["boundary_frac"] + r["outside_frac"]
    assert abs(total - 1.0) < 1e-6
    assert r["boundary_frac"] > 0.0
    assert int(boundary_band(m, 3).sum()) > 0


def test_effective_rank_spearman_and_style_diagnostics():
    torch.manual_seed(8)
    full = torch.randn(64, 8)
    assert effective_rank(full) > 6.0
    rank1 = torch.randn(64, 1) @ torch.randn(1, 8)
    assert effective_rank(rank1) < 1.5
    a = torch.arange(20.0)
    assert abs(spearman(a, 2 * a + 1) - 1.0) < 1e-9
    assert abs(spearman(a, -a) + 1.0) < 1e-9
    d = style_diagnostics(full, torch.nn.functional.normalize(torch.randn(64, 8), dim=-1),
                          u_gt=torch.randn(64, 32))
    assert "z_effective_rank" in d
    # review blocker B-4: the headline diagnostic is against the raw function
    # distance, not against the code the loss optimises toward
    assert "z_dist_spearman_vs_func" in d
    assert "z_dist_spearman_vs_zgt" in d
    only_func = style_diagnostics(full, u_gt=torch.randn(64, 32))
    assert "z_dist_spearman_vs_func" in only_func
    assert "z_dist_spearman_vs_zgt" not in only_func


def test_the_function_spearman_sees_magnitude_that_the_zgt_spearman_misses():
    """B-4 made concrete: a code that captures direction but not magnitude."""
    torch.manual_seed(11)
    n = 40
    direction = torch.nn.functional.normalize(torch.randn(n, 24), dim=-1)
    magnitude = torch.rand(n, 1) * 5.0 + 0.1
    u = direction * magnitude                       # true function offsets
    z_gt = direction                                # what L2-normalising keeps
    z_style = direction.clone()                     # a magnitude-blind code
    d = style_diagnostics(z_style, z_gt, u_gt=u)
    assert d["z_dist_spearman_vs_zgt"] > 0.99       # looks perfect ...
    assert d["z_dist_spearman_vs_func"] < 0.9       # ... but it is not


def test_activation_stats_flags_all_on_and_all_off():
    on = {"opacity": torch.ones(2, N_SLOTS), "existence": torch.ones(2, N_SLOTS),
          "mu": torch.rand(2, N_SLOTS, 3), "M": torch.rand(2, N_SLOTS, 3, 3),
          "b": torch.rand(2, N_SLOTS, 3)}
    s = activation_stats(on)
    assert s["frac_all_on"] == 1.0 and s["n_active_mean"] == N_SLOTS
    off = dict(on, opacity=torch.zeros(2, N_SLOTS))
    assert activation_stats(off)["frac_all_off"] == 1.0


def test_lexicographic_best_uses_the_declared_order():
    a = {"local_image_de00_median": 1.0, "lut_de00_p90": 5.0}
    b = {"local_image_de00_median": 1.0, "lut_de00_p90": 4.0}
    c = {"local_image_de00_median": 0.9, "lut_de00_p90": 9.0}
    assert lexicographic_best([a, b]) is b        # tie on primary -> secondary
    assert lexicographic_best([a, b, c]) is c     # primary wins outright
    assert lexicographic_best([]) is None
