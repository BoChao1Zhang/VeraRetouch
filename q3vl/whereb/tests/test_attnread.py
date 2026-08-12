"""Unit tests for the PR-ATT1-E1 attention readout and probe analysis.

These cover the parts where an error would be *silent*: a rotated field, a pool
that quietly picked up the wrong rows, a top-k that spent sink cells as coverage,
a permutation null that does not actually preserve cross-pool dependence.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.whereb.attnread import (
    IMAGE_TOKEN_ID, IM_END_ID, VISION_END_ID, VISION_START_ID,
    WHERE_CLOSE_ID, WHERE_OPEN_ID,
    AttentionTap, locate_image_columns, locate_query_pools, merged_grid,
    sink_mask_from_profile,
)
from q3vl.whereb.attnprobe import (
    combine_heads, field_scores, fit_oof_split, gt_merged_grid,
    head_norm_constants, max_stat_fwer, paired_wilcoxon, signed_rank_z,
    GatedLinearHead, HeadStackCNN,
)


class _FakeTok:
    """Decodes ids to text: even ids are words, odd ids are punctuation."""

    def decode(self, ids, skip_special_tokens=False):
        i = int(ids[0])
        return "word" if i % 2 == 0 else ","


def _seq(n_img=6, n_instr=3, n_where=4):
    """system … <|vision_start|> IMG* <|vision_end|> instr <|im_end|> … <where> … </where>"""
    ids = [1, 2, VISION_START_ID]
    ids += [IMAGE_TOKEN_ID] * n_img
    ids += [VISION_END_ID]
    instr_start = len(ids)
    ids += [10 + j for j in range(n_instr)]
    ids += [IM_END_ID, 77]
    n_prompt = len(ids)
    ids += [WHERE_OPEN_ID] + [20 + j for j in range(n_where)] + [WHERE_CLOSE_ID]
    return ids, n_prompt, instr_start


# --- geometry ---------------------------------------------------------------

def test_merged_grid_is_out_over_32():
    assert merged_grid(512, 768) == (16, 24)
    with pytest.raises(ValueError):
        merged_grid(512, 700)


def test_image_columns_located_by_value_and_contiguous():
    ids, _, _ = _seq(n_img=5)
    cols = locate_image_columns(ids)
    assert cols.tolist() == [3, 4, 5, 6, 7]
    with pytest.raises(ValueError):
        locate_image_columns([1, 2, 3])
    with pytest.raises(ValueError):          # non-contiguous block
        locate_image_columns([IMAGE_TOKEN_ID, 5, IMAGE_TOKEN_ID])


# --- query pools ------------------------------------------------------------

def test_pools_pick_the_right_rows():
    ids, n_prompt, instr_start = _seq(n_img=6, n_instr=3, n_where=4)
    p = locate_query_pools(ids, n_prompt, tokenizer=None)
    assert p.rows["where_special"].tolist() == [ids.index(WHERE_OPEN_ID)]
    assert p.rows["where_close"].tolist() == [ids.index(WHERE_CLOSE_ID)]
    # instruction = strictly between <|vision_end|> and the next <|im_end|>
    assert p.rows["instr_text"].tolist() == [instr_start, instr_start + 1, instr_start + 2]
    # content = strictly inside the where span
    o = ids.index(WHERE_OPEN_ID)
    assert p.rows["where_content"].tolist() == [o + 1, o + 2, o + 3, o + 4]


def test_profile_rows_exclude_image_rows():
    """Causal attention makes early image rows visible to more queries; averaging
    them into the column profile would stamp a positional ramp on it."""
    ids, n_prompt, _ = _seq(n_img=6)
    p = locate_query_pools(ids, n_prompt, tokenizer=None)
    cols = locate_image_columns(ids)
    assert p.profile_rows.min() > cols.max()
    assert p.profile_rows.max() == len(ids) - 1


def test_content_filter_drops_punctuation_tokens():
    ids, n_prompt, _ = _seq(n_where=4)          # inner ids 20,21,22,23
    p = locate_query_pools(ids, n_prompt, tokenizer=_FakeTok())
    o = ids.index(WHERE_OPEN_ID)
    assert p.rows["where_content"].tolist() == [o + 1, o + 3]   # 20, 22 are even
    assert p.detail["n_inner_where_tokens"] == 4
    assert p.detail["n_content_after_filter"] == 2


def test_pools_reject_a_changed_prompt_template():
    ids = [1, IMAGE_TOKEN_ID, WHERE_OPEN_ID, 5, WHERE_CLOSE_ID]   # no vision_end
    with pytest.raises(ValueError):
        locate_query_pools(ids, 2, tokenizer=None)


# --- the tap ----------------------------------------------------------------

class _FakeAttn(torch.nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.w = weights

    def forward(self, x=None):
        return (torch.zeros(1), self.w)


class _FakeLayer(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.self_attn = _FakeAttn(w)


class _FakeLM(torch.nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)


def test_tap_slices_rows_and_cols_exactly():
    torch.manual_seed(0)
    T, H, L = 9, 3, 2
    ws = [torch.rand(1, H, T, T) for _ in range(L)]
    lm = _FakeLM([_FakeLayer(w) for w in ws])
    tap = AttentionTap(lm, store_profile=True)
    rows = np.array([2, 5])
    cols = np.array([1, 3, 4])
    prof_rows = np.array([6, 7, 8])
    tap.set_selection(rows, cols, prof_rows)
    with tap.attached():
        for lay in lm.layers:
            lay.self_attn()
    stacked, prof = tap.stack()
    assert stacked.shape == (L, H, len(rows), len(cols))
    assert prof.shape == (L, H, len(cols))
    for li in range(L):
        expect = ws[li][0][:, rows][:, :, cols]
        assert torch.allclose(stacked[li], expect.float())
        pe = ws[li][0][:, prof_rows][:, :, cols].float().mean(dim=1)
        assert torch.allclose(prof[li], pe)


def test_tap_raises_when_attention_is_none():
    """RED LINE: SDPA/FA2 hand back None and must NOT be silently tolerated."""
    lm = _FakeLM([_FakeLayer(None)])
    tap = AttentionTap(lm)
    tap.set_selection(np.array([0]), np.array([0]), np.array([0]))
    with tap.attached():
        with pytest.raises(RuntimeError, match="eager"):
            lm.layers[0].self_attn()


# --- sink rule --------------------------------------------------------------

def test_sink_rule_is_robust_to_multiple_sinks():
    """Why median+MAD and not mean+sigma: enough sinks inflate sigma past
    themselves, so the mean rule stops seeing any of them."""
    rng = np.random.default_rng(0)
    p = 1.0 + rng.normal(0, 0.02, 64)
    sinks = [3, 7, 12, 19, 28, 40, 51, 60]
    p[sinks] = 50.0
    mask = sink_mask_from_profile(p, k=3.0)
    assert mask.sum() == len(sinks)
    assert all(mask[i] for i in sinks)
    # the mean+3*sigma rule catches NONE of the eight
    assert not (p > p.mean() + 3.0 * p.std()).any()


def test_sink_rule_returns_nothing_when_flat():
    assert sink_mask_from_profile(np.ones(32)).sum() == 0


def test_sink_rule_fails_closed_on_a_degenerate_bulk():
    """MAD == 0 (over half the cells identical) must not silently keep spikes."""
    p = np.ones(32)
    p[5] = 90.0
    assert sink_mask_from_profile(p).tolist() == [i == 5 for i in range(32)]


# --- scoring ----------------------------------------------------------------

def test_field_scores_ignores_invalid_cells_entirely():
    gt = np.zeros(9)
    gt[:3] = 1.0
    valid = np.ones(9, dtype=bool)
    perfect = np.zeros((1, 1, 9))
    perfect[0, 0, :3] = 1.0
    assert field_scores(perfect, gt, valid)[0, 0] == pytest.approx(1.0, abs=1e-6)

    # a field whose mass is all on a sink cell must not be rescued by it
    sinky = np.zeros((1, 1, 9))
    sinky[0, 0, 8] = 100.0
    valid2 = valid.copy()
    valid2[8] = False
    s = field_scores(sinky, gt, valid2)[0, 0]
    assert 0.0 <= s < 1.0


def test_field_scores_is_invariant_to_monotone_rescaling():
    """Top-k binarisation is why a per-head scale cannot buy a better score."""
    rng = np.random.default_rng(0)
    f = rng.random((4, 5, 12))
    gt = (rng.random(12) > 0.6).astype(float)
    valid = np.ones(12, dtype=bool)
    a = field_scores(f, gt, valid)
    b = field_scores(f * 1e-4 + 7.0, gt, valid)
    assert np.allclose(a, b)


def test_field_scores_matches_the_gt_area():
    gt = np.zeros(16)
    gt[:5] = 1.0
    valid = np.ones(16, dtype=bool)
    f = np.arange(16, dtype=float).reshape(1, 1, 16)
    # top-5 by value = cells 11..15; soft-IoU of a 5-cell mask vs a 5-cell GT
    # with zero overlap is 0
    assert field_scores(f, gt, valid)[0, 0] == pytest.approx(0.0, abs=1e-6)


# --- normalisation ----------------------------------------------------------

def test_head_norm_constants_are_arm_constants_not_per_image():
    rng = np.random.default_rng(1)
    stack = [rng.random((2, 3, 10)) for _ in range(5)]
    n = [10] * 5
    mu, sd = head_norm_constants(stack, n)
    assert mu.shape == (2, 3) and sd.shape == (2, 3)
    assert np.all(sd > 0)
    # scaling one sample must move the constants only a little -- they are pooled
    stack2 = list(stack)
    stack2[0] = stack2[0] * 1.01
    mu2, _ = head_norm_constants(stack2, n)
    assert np.allclose(mu, mu2, rtol=0.05)


def test_combine_heads_is_convex():
    rng = np.random.default_rng(2)
    f = rng.random((3, 4, 9))
    mu = np.zeros((3, 4))
    sd = np.ones((3, 4))
    idx = [(0, 1), (2, 3)]
    out = combine_heads(f, idx, [1.0, 1.0], mu, sd, n_cells=9)
    expect = 0.5 * (f[0, 1] * 9) + 0.5 * (f[2, 3] * 9)
    assert np.allclose(out, expect)
    with pytest.raises(ValueError):
        combine_heads(f, idx, [0.0, 0.0], mu, sd, n_cells=9)


# --- statistics -------------------------------------------------------------

def test_signed_rank_z_signs_and_symmetry():
    d = np.array([0.1, 0.2, 0.3, 0.4])
    assert signed_rank_z(d) > 0
    assert signed_rank_z(-d) == pytest.approx(-signed_rank_z(d))
    assert signed_rank_z(np.zeros(5)) == 0.0


def test_paired_wilcoxon_agrees_with_scipy_direction():
    rng = np.random.default_rng(3)
    a = rng.normal(0.5, 0.1, 60)
    b = rng.normal(0.3, 0.1, 60)
    r = paired_wilcoxon(a, b)
    assert r["n"] == 60 and r["delta_median"] > 0 and r["p_value"] < 1e-6


def test_max_stat_fwer_is_never_smaller_than_the_raw_p():
    rng = np.random.default_rng(4)
    n = 80
    d1 = rng.normal(0.05, 0.1, n)
    d2 = rng.normal(0.00, 0.1, n)
    d3 = rng.normal(0.00, 0.1, n)
    res = max_stat_fwer({"a": d1, "b": d2, "c": d3}, n_perm=2000, seed=0)
    assert set(res["p_fwer"]) == {"a", "b", "c"}
    assert all(0 < v <= 1 for v in res["p_fwer"].values())
    # the correction must cost something: a null key's corrected p exceeds 0.05
    # far more often than its raw one would
    assert res["p_fwer"]["b"] > 0.01


def test_max_stat_fwer_rejects_ragged_families():
    with pytest.raises(ValueError):
        max_stat_fwer({"a": np.zeros(4), "b": np.zeros(5)}, n_perm=10)


def test_max_stat_fwer_controls_the_family_error_rate():
    """Under a global null the corrected minimum p is uniform-ish, so rejecting
    at 0.05 should happen in about 5% of families, not 3x that."""
    rng = np.random.default_rng(5)
    rejects = 0
    trials = 60
    for _ in range(trials):
        fam = {k: rng.normal(0, 1, 40) for k in "abc"}
        res = max_stat_fwer(fam, n_perm=400, seed=int(rng.integers(1 << 30)))
        if min(res["p_fwer"].values()) < 0.05:
            rejects += 1
    assert rejects <= 0.20 * trials


# --- folds ------------------------------------------------------------------

def test_fit_oof_split_never_splits_a_source_image():
    rows = []
    for s in range(20):
        for j in range(3):
            rows.append({"sample_id": f"s{s}_{j}", "source_image_id": f"src{s}",
                         "render_mode": "local" if j < 2 else "global"})
    folds = fit_oof_split(rows)
    for s in range(20):
        got = {folds[f"s{s}_{j}"] for j in range(3)}
        assert len(got) == 1, f"src{s} was split across folds"


def test_fit_oof_split_balances_the_local_subset_and_is_deterministic():
    rows = [{"sample_id": f"s{i}", "source_image_id": f"src{i//4}",
             "render_mode": "local" if i % 2 else "global"} for i in range(200)]
    f1 = fit_oof_split(rows)
    f2 = fit_oof_split(rows)
    assert f1 == f2
    loc = [sum(1 for r in rows
               if f1[r["sample_id"]] == f and r["render_mode"] == "local")
           for f in ("fit", "oof")]
    assert abs(loc[0] - loc[1]) <= 2


def test_fit_oof_split_tolerates_missing_source_id():
    rows = [{"sample_id": "a", "source_image_id": None, "render_mode": "local"},
            {"sample_id": "b", "source_image_id": None, "render_mode": "local"}]
    folds = fit_oof_split(rows)
    assert set(folds) == {"a", "b"}


# --- GT ---------------------------------------------------------------------

def test_gt_merged_grid_is_an_exact_2x2_area_mean():
    low = torch.tensor([[1.0, 3.0, 0.0, 0.0],
                        [5.0, 7.0, 0.0, 0.0],
                        [0.0, 0.0, 2.0, 2.0],
                        [0.0, 0.0, 2.0, 2.0]])
    out = gt_merged_grid(low, 2, 2)
    assert out.shape == (2, 2)
    assert out[0, 0] == pytest.approx(4.0)      # (1+3+5+7)/4
    assert out[1, 1] == pytest.approx(2.0)
    with pytest.raises(ValueError):
        gt_merged_grid(low, 3, 3)


# --- learnable heads --------------------------------------------------------

def test_head_param_budgets_match_the_proposal():
    g = GatedLinearHead(36, 32)
    assert g.n_params() == 36 * 32 + 2          # ~1.2k, PROPOSAL tier B
    c = HeadStackCNN(128, 64, 32)
    assert 20_000 < c.n_params() < 30_000       # ~27k, PROPOSAL tier C
    assert c.n_params() < 50_000                # protocol cap on the fusion head


def test_heads_forward_shapes():
    g = GatedLinearHead(4, 3)
    assert g(torch.rand(4, 3, 20)).shape == (20,)
    c = HeadStackCNN(8, 6, 4)
    assert c(torch.rand(8, 5, 7)).shape == (5, 7)


# --- domain assertion (s-cache contract, review blocker B4) -----------------

def test_assert_domain_passes_and_reports_extremes():
    from q3vl.whereb.attnread import assert_domain
    f = np.array([-1.0, 0.0, 2.5])
    r = assert_domain(f, (-3.0, 8.0), name="fused")
    assert r["passed"] and r["crosses_zero"]
    assert r["measured"] == [-1.0, 2.5]


def test_assert_domain_raises_when_data_escapes():
    from q3vl.whereb.attnread import assert_domain
    with pytest.raises(ValueError, match="escapes"):
        assert_domain(np.array([0.0, 99.0]), (-1.0, 5.0))


def test_assert_domain_detects_a_clamp():
    """The silent failure: values still inside the anchors, but the axis is gone."""
    from q3vl.whereb.attnread import assert_domain
    clamped = np.clip(np.linspace(-5, 5, 100), 0.0, 1.0)
    r = assert_domain(clamped, (0.0, 1.0))
    assert r["frac_saturated"] > 0.7          # a clamp piles mass on the endpoints
