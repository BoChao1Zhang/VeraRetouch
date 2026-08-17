"""PCH-Full / PCH-Lite (proposal §2.2) and the M-series harness (§2.5).

Every test here fences a failure this campaign has already paid for:

* a module that perturbs step 0, so the injection Delta measured against a
  resumed checkpoint is a mixture of two effects;
* a confidence gate that leaks code content into the null path, so ABSTAIN and
  M3 samples are not on the path the board says they are;
* a control arm that changes how *much* signal is present while claiming to
  change only *what* it says (EPR-003 registered this and asserted it nowhere);
* a criterion that is defined, imported and never called (three occurrences);
* the 21-d contract drifting away from ``GEOM_SLOTS`` (AMD-6).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from q3vl.whereb.amort.geocode import (DISC_GROUPS, GROUP_NAMES, GROUP_SIZES,
                                       GROUP_SPANS, CodePool, ContNorm, GeoCode,
                                       assert_bitcount_conserved,
                                       condition_dropout, label_smooth,
                                       sample_conf_beta, scale_conf, group_only)
from q3vl.whereb.amort.geomparse import GEOM_DIM, GEOM_SLOTS
from q3vl.whereb.amort.model import AmortModel
from q3vl.whereb.amort.pch_full import PCHFull, PCHFullConfig
from q3vl.whereb.amort.resume import load_resumable

GH, GW = 6, 5
IN_DIM = 1024
CH = 128


# --- helpers ----------------------------------------------------------------

def _base_model(**kw) -> AmortModel:
    torch.manual_seed(0)
    return AmortModel("P3prime", in_dim=IN_DIM, n_blocks=2, with_semantic=False,
                      **kw)


def _inputs():
    g = torch.Generator().manual_seed(7)
    feat = torch.randn(1, IN_DIM, GH, GW, generator=g)
    cond_h = torch.randn(1, 4, 2560, generator=g)
    sim = torch.randn(1, 1, GH, GW, generator=g)
    return feat, cond_h, sim, torch.zeros(1, dtype=torch.long), torch.zeros(1, dtype=torch.long)


def _field(model: AmortModel, geom=None) -> torch.Tensor:
    feat, cond_h, sim, word_ids, word_off = _inputs()
    model.eval()
    with torch.no_grad():
        cond = model.cond_of(cond_h, None, word_ids, word_off)
        out = model.forward_geo(feat, cond, torch.zeros(GH * GW, 71), sim=sim,
                                geom=geom, grid_h=GH, grid_w=GW)
    return out["s_low"].clone()


def _trained_checkpoint():
    base = _base_model()
    with torch.no_grad():
        for p in base.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return base, {k: v.clone() for k, v in base.state_dict().items()}


def _code(bits=(0, 6, 15), cont=(0.4, 0.6, 0.25), conf=(1.0, 1.0, 1.0, 1.0)):
    v = torch.zeros(GEOM_DIM)
    for b in bits:
        v[b] = 1.0
    return GeoCode(v, torch.tensor(cont), torch.tensor(conf), True)


# --- the AMD-6 contract -----------------------------------------------------

def test_contract_matches_geom_slots_slot_by_slot():
    """AMD-6: 21 = 5 shape + 9 dir + 7 ext, and the spans are read off the
    single authority rather than copied next to it."""
    assert GEOM_DIM == 21
    assert GROUP_SIZES == {"shape": 5, "dir": 9, "ext": 7}
    names = [n for n, _ in GEOM_SLOTS]
    for g, prefix in (("shape", "shape_"), ("dir", "dir_"), ("ext", "ext_")):
        a, b = GROUP_SPANS[g]
        assert all(n.startswith(prefix) for n in names[a:b])
        # and nothing of that group hides outside its span
        assert sum(n.startswith(prefix) for n in names) == b - a
    assert [GROUP_SPANS[g] for g in DISC_GROUPS] == [(0, 5), (5, 14), (14, 21)]


def test_geocode_refuses_nan_and_inconsistent_valid():
    with pytest.raises(AssertionError, match="NaN"):
        GeoCode(torch.full((GEOM_DIM,), float("nan")), torch.zeros(3),
                torch.ones(4), True)
    with pytest.raises(AssertionError, match="valid"):
        GeoCode(torch.zeros(GEOM_DIM), torch.zeros(3), torch.zeros(4), True)
    with pytest.raises(AssertionError, match="valid"):
        GeoCode(torch.zeros(GEOM_DIM), torch.zeros(3), torch.ones(4), False)
    # the contract's own null is consistent
    GeoCode.null().validate()


def test_cont_domain_assertion_raises_instead_of_clamping():
    norm = ContNorm(domain=(0.0, 1.0), source="test")
    norm.assert_in_domain(torch.tensor([0.0, 1.0, 0.5]))
    with pytest.raises(AssertionError, match="outside the domain"):
        norm.assert_in_domain(torch.tensor([0.0, 1.4, 0.5]))


# --- the two structural guarantees -----------------------------------------

def test_step0_is_bit_identical_to_m0():
    """The whole reason PCH may be added to a resumed checkpoint."""
    base, sd = _trained_checkpoint()
    arm = _base_model(geom_inject=True, geom_mode="pch", pch_impl="spec",
                      pch_size="full")
    report = load_resumable(arm, sd)
    assert report["zero_padded"] == []
    assert all(k.startswith("pch.") for k in report["from_init"])
    assert report["n_from_init"] > 0

    ref = _field(base)
    assert torch.equal(_field(arm, geom=_code()), ref), "PCH perturbed step 0"
    assert torch.equal(_field(arm, geom=GeoCode.null()), ref)
    assert torch.equal(_field(arm, geom=None), ref)


def test_lite_step0_is_bit_identical_too():
    base, sd = _trained_checkpoint()
    arm = _base_model(geom_inject=True, geom_mode="pch", pch_impl="spec",
                      pch_size="lite")
    load_resumable(arm, sd)
    assert torch.equal(_field(arm, geom=_code()), _field(base))


def test_trained_injector_moves_the_field_and_uses_both_taps():
    """A no-op at init must not stay a no-op after training -- otherwise the
    previous test passes for the trivial reason that nothing is wired."""
    arm = _base_model(geom_inject=True, geom_mode="pch", pch_impl="spec")
    with torch.no_grad():
        for p in arm.pch.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        arm.pch.gamma.fill_(0.5)
    assert not torch.equal(_field(arm, geom=_code()), _field(arm, geom=None))

    feat = torch.randn(1, CH, GH, GW)
    res, dlog = arm.pch(feat, _code())
    assert torch.count_nonzero(res) > 0, "tap B produced nothing"
    assert torch.count_nonzero(dlog) > 0, "tap A produced nothing"
    dom = arm.pch.domain_report()
    assert dom["delta_logit"]["absmax"] > 0 and dom["delta_feature"]["absmax"] > 0
    assert dom["tanh_gamma"] == pytest.approx(float(np.tanh(0.5)), abs=1e-6)


def test_conf_zero_degenerates_to_the_null_constant():
    """conf=0 must make the output independent of the code's CONTENT.

    Not "zero output": §2.2's fallback is a *learned* null constant, which is the
    path condition dropout trains.  What must not happen is code content leaking
    past the gate -- which at initialisation is invisible, because everything is
    zero for structural reasons.
    """
    pch = PCHFull(PCHFullConfig.full(feat_dim=CH))
    with torch.no_grad():
        for p in pch.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        pch.gamma.fill_(0.7)
    feat = torch.randn(1, CH, GH, GW)

    a = _code(bits=(0, 6, 15))
    b = _code(bits=(3, 11, 19), cont=(0.9, 0.1, 0.8))
    pch.assert_null_is_content_free(feat, a, b)

    # and the tokens themselves are exactly the null embeddings
    z = torch.zeros(len(GROUP_NAMES))
    toks = pch.tokens_of(a.replace(conf=z, valid=False))
    assert torch.equal(toks[0, 1:], pch.null_g.weight)
    # while a live code is not
    assert not torch.equal(pch.tokens_of(a)[0, 1:], pch.null_g.weight)


def test_conf_gate_interpolates_exactly():
    """T_g = conf_g t_g + (1-conf_g) null_g, to the bit."""
    pch = PCHFull(PCHFullConfig.full(feat_dim=CH))
    with torch.no_grad():
        for p in pch.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    code = _code()
    full = pch.tokens_of(code)[0, 1:]
    z = torch.zeros(len(GROUP_NAMES))
    null = pch.tokens_of(code.replace(conf=z, valid=False))[0, 1:]
    half = pch.tokens_of(code.replace(conf=torch.full((4,), 0.5)))[0, 1:]
    assert torch.allclose(half, 0.5 * full + 0.5 * null, atol=1e-6)


# --- the pad-mask red line --------------------------------------------------

def test_pad_mask_gives_pad_cells_exactly_zero_weight():
    pch = PCHFull(PCHFullConfig.full(feat_dim=CH))
    with torch.no_grad():
        for p in pch.parameters():
            p.add_(torch.randn_like(p) * 0.05)
        pch.gamma.fill_(0.5)
    n = GH * GW
    valid = torch.ones(1, n, dtype=torch.bool)
    valid[0, -4:] = False

    feat = torch.randn(1, CH, GH, GW)
    hs, keys = pch._run(feat, _code(), valid, keep_attn=True)
    attn = pch.transformer.layers[0].cross_attn_token_to_image.last_attn
    assert attn is not None
    assert float(attn[..., -4:].abs().max()) == 0.0, "pad cells drew attention"
    assert torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-5)

    # the token read-out must not depend on what the pad cells contain
    garbage = feat.clone()
    garbage[..., -1, -1] += 50.0            # inside the masked tail
    hs2, _ = pch._run(garbage.reshape(1, CH, GH, GW), _code(), valid)
    assert torch.allclose(hs, hs2, atol=1e-5)

    # and tap B must not write a residual into a pad cell
    res = pch._tap_b(feat, keys, valid).detach()
    assert float(res.reshape(1, CH, -1)[..., -4:].abs().max()) == 0.0


def test_wrong_feature_width_is_refused():
    pch = PCHFull(PCHFullConfig.full(feat_dim=CH))
    with pytest.raises(AssertionError, match="channel injection point"):
        pch(torch.randn(1, 64, GH, GW), _code())


# --- capacity ---------------------------------------------------------------

def test_param_budget_full_and_lite():
    """Measured, then frozen as a regression.

    They differ from the proposal's 3.96M / 0.52M because those figures assumed a
    1024-channel injection point; §0.1-3 measured 128 (`heads.py::apply_inject`),
    which shrinks the key projection and both tap-B convolutions.  The structure
    is the proposal's; the arithmetic follows the real tensor.
    """
    full = PCHFull(PCHFullConfig.full(feat_dim=CH))
    lite = PCHFull(PCHFullConfig.lite(feat_dim=CH))
    assert full.n_params() == 3_527_873
    assert lite.n_params() == 295_425
    # the Lite档 keeps tap B alive (D-11): the rank-1 tap's escape hatch must not
    # vanish with the capacity
    assert lite.cfg.tap_b and lite.cfg.tap_b_lite
    assert "tap_b_out" in dict(lite.named_parameters()).keys() or any(
        n.startswith("tap_b_out") for n, _ in lite.named_parameters())


# --- the M-series transforms ------------------------------------------------

def test_derangement_preserves_marginals_and_has_no_fixed_point():
    from q3vl.whereb.amort.mseries import assert_marginals_preserved

    rng = np.random.default_rng(0)
    codes = {}
    for i in range(64):
        v = np.zeros(GEOM_DIM, dtype=np.float32)
        v[rng.choice(GEOM_DIM, size=int(rng.integers(1, 5)), replace=False)] = 1.0
        codes[f"s{i}"] = GeoCode.from_multihot(v, c_cont=[0.5, 0.5, 0.5])
    pool = CodePool(codes, source="test")
    mapping = pool.derangement(seed=3)
    assert all(k != v for k, v in mapping.items())
    assert sorted(mapping.values()) == sorted(mapping)
    assert_marginals_preserved(pool, mapping)


def test_slot_shuffle_conserves_the_active_bit_count():
    """EPR-003's registered-but-unasserted property, now enforced at the source."""
    from q3vl.whereb.amort.geomparse import shuffle_features

    rng = np.random.default_rng(1)
    v = np.zeros(GEOM_DIM, dtype=np.float32)
    v[[0, 5, 14, 20]] = 1.0
    out = shuffle_features(v, rng)
    assert int(out.sum()) == 4
    a = GeoCode.from_multihot(v)
    b = GeoCode.from_multihot(out)
    assert_bitcount_conserved(a, b, where="slot shuffle")
    with pytest.raises(AssertionError, match="active-bit count"):
        assert_bitcount_conserved(a, GeoCode.from_multihot(np.zeros(GEOM_DIM,
                                                                    dtype=np.float32)))


def test_m1_prob_transform_smooths_labels_and_samples_conf():
    code = _code()
    sm = label_smooth(code)
    assert sorted(float(x) for x in np.unique(sm.c_disc.numpy())) == \
        pytest.approx([0.1, 0.9])
    got = sample_conf_beta(sm, np.random.default_rng(0))
    assert all(0.0 < float(x) < 1.0 for x in got.conf)
    # a group the producer never filled stays at zero rather than being invented
    partial = code.replace(conf=torch.tensor([1.0, 1.0, 0.0, 1.0]))
    assert float(sample_conf_beta(partial, np.random.default_rng(0)).conf[2]) == 0.0


def test_conf_scale_and_group_only():
    code = _code()
    assert torch.allclose(scale_conf(code, 0.25).conf, torch.full((4,), 0.25))
    assert scale_conf(code, 0.0).valid is False
    only = group_only(code, "dir")
    assert float(only.conf[1]) == 1.0 and float(only.conf.sum()) == 1.0


def test_condition_dropout_reports_its_coverage():
    rng = np.random.default_rng(0)
    code = _code()
    n, per_group = 0, {g: 0 for g in GROUP_NAMES}
    for _ in range(4000):
        out, hits = condition_dropout(code, rng, p_all=0.15, p_group=0.05)
        n += 1
        for g in GROUP_NAMES:
            per_group[g] += int(hits[g])
        if hits["all"]:
            assert not out.valid
    # 0.15 + 0.85*0.05 ~= 0.19 per group; the M5 pre-condition is >= 5%
    for g in GROUP_NAMES:
        assert 0.15 < per_group[g] / n < 0.25


# --- the runtime assertions -------------------------------------------------

class _FakeInner:
    device = torch.device("cpu")
    allow_context_fallback = False

    def build(self, samples, modes):
        return list(samples)

    def facts(self):
        return {"inner": True}


class _X:
    def __init__(self, sid):
        self.sample_id = sid
        self.geom = None
        self.meta = {"context_text": ""}


def _harness(arm: str, pool=None, **kw):
    from q3vl.whereb.amort.mseries import CodeSource, MSeriesHarness, arm_spec

    class _Src(CodeSource):
        name = "test"

        def code_for(self, sample_id, text=""):
            return _code()

    return MSeriesHarness(_FakeInner(), _Src(), arm_spec(arm), pool=pool, **kw)


def _ok_board():
    col = {"n": 10, "median": 0.8}
    return {"main_context": "gt",
            "contexts": {"gt": {"topk_iou": col, "grid_boundary_f1": col,
                                "center_prior_topk_iou": col,
                                "headline_normal_only": {"n": 6,
                                                         "topk_iou_median": 0.79}}}}


def test_assertion_blocks_a_board_whose_transform_never_ran():
    from q3vl.whereb.amort.mseries import assert_mseries_wired

    h = _harness("M3")
    with pytest.raises(AssertionError, match="injection path never ran"):
        assert_mseries_wired(_ok_board(), h)

    h.build([_X("a"), _X("b")], ["gt", "gt"])
    rep = assert_mseries_wired(_ok_board(), h)
    assert rep["checks"]["n_transform_calls"] == 2


def test_assertion_blocks_a_board_without_the_mandatory_columns():
    from q3vl.whereb.amort.mseries import assert_mseries_wired

    h = _harness("M3")
    h.build([_X("a")], ["gt"])
    board = _ok_board()
    board["contexts"]["gt"]["center_prior_topk_iou"] = {"n": 0}
    with pytest.raises(AssertionError, match="centre-prior"):
        assert_mseries_wired(board, h)

    board = _ok_board()
    board["contexts"]["gt"].pop("headline_normal_only")
    with pytest.raises(AssertionError, match="normal-only"):
        assert_mseries_wired(board, h)


def test_auc_anywhere_is_refused():
    from q3vl.whereb.amort.mseries import assert_no_auc

    assert_no_auc(_ok_board())
    with pytest.raises(AssertionError, match="AUC"):
        assert_no_auc({"contexts": {"gt": {"roc_auc": 0.83}}})
    with pytest.raises(AssertionError, match="AUC"):
        assert_no_auc({"rows": [{"nested": {"AUC": 0.5}}]})


def test_no_auc_in_the_new_sources():
    """The ban is on the code as well as on the board."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("pch_full.py", "geocode.py", "mseries.py"):
        text = (root / name).read_text(encoding="utf-8")
        code = "\n".join(l for l in text.splitlines()
                         if not l.strip().startswith("#"))
        # the word may appear in a docstring explaining the ban, never as an API
        assert not re.search(r"\bauc\s*[=(]", code, flags=re.I)
        assert "roc_auc" not in code and "roc_auc_score" not in code


def test_m3_harness_puts_every_sample_on_the_null_path():
    h = _harness("M3")
    xs = [_X("a"), _X("b"), _X("c")]
    h.build(xs, ["gt"] * 3)
    assert all(not x.geom.valid for x in xs)
    assert h.n_null == 3


def test_m2b_constant_arm_gives_every_sample_the_same_code():
    codes = {f"s{i}": _code(bits=(i % 5,)) for i in range(8)}
    pool = CodePool(codes, source="test")
    h = _harness("M2b", pool=pool)
    xs = [_X("s1"), _X("s2")]
    h.build(xs, ["gt", "gt"])
    assert torch.equal(xs[0].geom.c_disc, xs[1].geom.c_disc)
    assert torch.allclose(xs[0].geom.c_disc, pool.mean_code().c_disc)


def _fake_inputs(sid="a", geom=None):
    from q3vl.whereb.amort.data import AmortSampleInputs

    g = torch.Generator().manual_seed(11)
    return AmortSampleInputs(
        sample_id=sid,
        feat=torch.randn(1, IN_DIM, GH, GW, generator=g),
        sim=torch.rand(1, 1, GH, GW, generator=g),
        center=None,
        cond_h=torch.randn(1, 4, 2560, generator=g),
        cond_mask=torch.ones(1, 4, dtype=torch.bool),
        word_ids=torch.zeros(1, dtype=torch.long),
        word_offsets=torch.zeros(1, dtype=torch.long),
        phi_dir=torch.zeros(GH * GW, 71),
        guide_hi=None,
        gt_low=(torch.rand(GH, GW, generator=g) > 0.5).float(),
        gt_hi=None, gt_partner_low=None,
        grid_h=GH, grid_w=GW, is_fake=False, family="band",
        route_semantic=False, geom=geom, meta={"context_text": ""})


def test_one_training_step_reaches_the_injector_and_nothing_else():
    """End to end through the real loss: the frozen-base arm trains PCH only,
    and the zero-initialised branches are the ones that receive gradient first
    (ControlNet's growth behaviour, not a dead tap)."""
    from q3vl.whereb.amort.losses import LossWeights
    from q3vl.whereb.amort.trainer import compute_micro_batch

    arm = _base_model(geom_inject=True, geom_mode="pch", pch_impl="spec")
    # A *trained* base, because that is the deployment condition and because a
    # fresh one hides the interesting half: P3's `to_field` is zero-initialised,
    # so on an untrained head dL/d(tower features) is exactly zero and tap B
    # receives no gradient at all.  The M series always resumes the P3' cont2
    # checkpoint, where it is not.
    with torch.no_grad():
        for name, p in arm.named_parameters():
            if not name.startswith("pch."):
                p.add_(torch.randn_like(p) * 0.02)
    for name, p in arm.named_parameters():
        p.requires_grad = name.startswith("pch.")

    w = LossWeights(sdf=0.0, area=0.0, sep=0.0)
    total, stats, rows = compute_micro_batch(
        arm, [_fake_inputs(geom=_code())], w)
    total.backward()

    grads = {n: (None if p.grad is None else float(p.grad.abs().sum()))
             for n, p in arm.named_parameters() if p.requires_grad}
    assert grads["pch.gamma"] and grads["pch.gamma"] > 0, "tap A cannot open"
    assert grads["pch.zero_conv.weight"] > 0, "tap B cannot open"
    # and nothing outside the injector moved
    assert all(p.grad is None for n, p in arm.named_parameters()
               if not n.startswith("pch."))

    # after one step the taps are non-zero, so the rest of the module unlocks
    opt = torch.optim.AdamW([p for p in arm.parameters() if p.requires_grad],
                            lr=1e-2)
    opt.step()
    arm.zero_grad(set_to_none=True)
    total2, _, _ = compute_micro_batch(arm, [_fake_inputs(geom=_code())], w)
    total2.backward()
    assert float(arm.pch.e_shape.weight.grad.abs().sum()) > 0, \
        "the prototype bank never receives gradient"


def test_codes_jsonl_round_trip_is_the_arm_to_arm_seam(tmp_path):
    """AMD-1's reference column and every extraction arm come in this way."""
    from q3vl.whereb.amort.mseries import FileCodeSource, make_code_source

    pool = CodePool({"a": _code(bits=(0, 7)), "b": GeoCode.null()}, source="t")
    p = pool.write(tmp_path / "codes.jsonl")
    src = make_code_source("file", code_file=p)
    assert isinstance(src, FileCodeSource)
    got = src.code_for("a")
    assert torch.allclose(got.c_disc, pool.codes["a"].c_disc)
    assert torch.allclose(got.conf, pool.codes["a"].conf)
    assert src.code_for("b").valid is False
    # a sample the file has no code for is an ABSTAIN, never a crash and never a
    # dropped sample (§2.3's last row)
    assert src.code_for("not-in-the-file").valid is False


def test_train_recipe_is_the_proposal_recipe():
    from q3vl.whereb.amort.mseries import TRAIN_RECIPE, train_config_kwargs

    kw = train_config_kwargs()
    assert kw["learning_rate"] == 3e-4 and kw["weight_decay"] == 1e-4
    assert kw["effective_batch"] == 64 and kw["max_steps"] == 10_000
    assert kw["scheduler"] == "cosine"
    assert round(kw["warmup_ratio"] * kw["max_steps"]) == 300
    assert TRAIN_RECIPE["checkpoint_selection"].startswith("S-val soft-IoU")
