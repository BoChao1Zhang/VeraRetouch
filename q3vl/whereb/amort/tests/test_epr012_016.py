"""EPR-012..016 five-feature pack: equivalence + wiring checks.

The pack's contract is that every one of the five switches is independent and
OFF by default, and that with all of them off the training and inference paths
are the ST_LANG baseline **bit for bit**.  The two ``_ref_*`` functions below
are frozen transcriptions of the pre-change code (``uniq_wta_loss``'s WTA
aggregation and ``UniQ4Head.forward``); they are the baseline the equivalence
tests compare against, so they must never be "fixed" to follow the new code.
"""

from dataclasses import dataclass, field
from typing import Any

import torch

from q3vl.whereb.amort.losses import (LossWeights, amort_sample_loss,
                                      empty_mask, uniq_wta_loss)
from q3vl.whereb.amort.trainer import compute_micro_batch
from q3vl.whereb.amort.uniq import UNIQ_FAMILIES, UniQHead
from q3vl.whereb.amort.uniq4 import VARIANT4, AmortModelV4, UniQ4Head
from q3vl.whereb.amort.uniq5 import UniQ5Head


# -- frozen pre-change references -------------------------------------------
def _ref_wta(s_all, mask_of, gt, w, *, phi_sdf=None, gt_partner=None,
             is_fake=False, structural=False, family="",
             cls_logits=None, sel_logits=None):
    """``losses.uniq_wta_loss`` exactly as it stood before EPR-012/015/016."""
    import torch.nn.functional as F

    from q3vl.whereb.amort.losses import AmortLoss

    n_q = int(s_all.shape[0])
    zero = s_all.sum() * 0.0
    if is_fake:
        terms = {"fake": empty_mask(mask_of(s_all), None)}
        for k in ("bce", "sdf", "area", "sep", "curv", "mono"):
            terms[k] = zero
        return AmortLoss(total=w.fake * terms["fake"], terms=terms,
                         stats={"is_fake": 1.0})
    per = [amort_sample_loss(mask_of(s_all[k]), gt, w, valid=None,
                             phi_sdf=phi_sdf, gt_partner=gt_partner,
                             is_fake=False, structural=structural)
           for k in range(n_q)]
    j = int(torch.stack([p.total.detach() for p in per]).argmin())
    total = per[j].total
    terms = dict(per[j].terms)
    if cls_logits is not None and w.uniq_cls and family in UNIQ_FAMILIES:
        tgt = torch.tensor([UNIQ_FAMILIES.index(family)],
                           device=cls_logits.device)
        terms["uniq_cls"] = F.cross_entropy(cls_logits[j:j + 1], tgt)
        total = total + w.uniq_cls * terms["uniq_cls"]
    if sel_logits is not None and w.uniq_sel:
        tgt = torch.tensor([j], device=sel_logits.device)
        terms["uniq_sel"] = F.cross_entropy(sel_logits.reshape(1, -1), tgt)
        total = total + w.uniq_sel * terms["uniq_sel"]
    return AmortLoss(total=total, terms=terms, stats={"uniq_winner": float(j)})


def _ref_head_forward(h, feat, extra, cond, h_where, h_mask):
    """``UniQ4Head.forward`` exactly as it stood before EPR-013/016."""
    from q3vl.where.config import S_SCALE

    codes = h.tower(feat, extra, cond)
    gh, gw = codes.shape[-2:]
    pix = codes[0].reshape(codes.shape[1], gh * gw)
    if h.fourier_bands:
        four = h._fourier(gh, gw, codes.device, pix.dtype)
        pix = torch.cat([pix, four.reshape(four.shape[0], gh * gw)], dim=0)
    qh = h_where[0, -h.n_queries:, :].float()
    q = h.q_proj_in(qh)
    q = q + h.ffn(h.q_norm(q.unsqueeze(0)))[0]
    wb = h.to_mask(q)
    raw = wb[:, :-1] @ pix + wb[:, -1:]
    s_all = (S_SCALE * torch.tanh(raw / S_SCALE)).reshape(h.n_queries, gh, gw)
    return {"s_all": s_all, "cls_logits": h.cls(q),
            "sel_logits": h.sel(q).reshape(-1)}


# -- fixtures ----------------------------------------------------------------
def _head(cls=UniQ4Head, *, randomise=True, seed=7, **kw):
    torch.manual_seed(seed)
    h = cls(32, 64, 1, 2, 16, text_dim=48, n_queries=8, **kw)
    if randomise:                     # a zero-init head makes every field 0
        torch.manual_seed(seed + 1)
        with torch.no_grad():
            for p in h.parameters():
                p.copy_(torch.randn_like(p) * 0.1)
    return h


def _head_inputs(gh=6, gw=8, rows=8, t=20):
    torch.manual_seed(3)
    return (torch.randn(1, 64, gh, gw), torch.rand(1, 1, gh, gw),
            torch.randn(1, 16), torch.randn(1, t + rows, 48),
            torch.ones(1, t + rows))


def _fields(seed=0, k=8, gh=6, gw=8):
    torch.manual_seed(seed)
    s = (torch.randn(k, gh, gw) * 1.5).requires_grad_(True)
    gt = (torch.rand(gh, gw) > 0.6).float()
    return s, gt


def _mask_of(s):
    return torch.sigmoid(2.0 * s)


def _raises(fn, needle):
    """`pytest.raises` without the import -- this file also runs standalone."""
    try:
        fn()
    except AssertionError as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected an AssertionError mentioning {needle!r}")


# ===========================================================================
# equivalence: all five switches off == baseline, bit for bit
# ===========================================================================
def test_wta_loss_bit_identical_with_all_switches_off():
    w = LossWeights(uniq_cls=0.05, uniq_sel=0.05)
    for seed in range(6):
        s, gt = _fields(seed)
        cls_l = torch.randn(8, len(UNIQ_FAMILIES), requires_grad=True)
        sel_l = torch.randn(8, requires_grad=True)
        partner = (torch.rand(6, 8) > 0.7).float()
        phi = torch.randn(6, 8) * 0.1
        for fake in (False, True):
            kw = dict(phi_sdf=phi, gt_partner=partner, is_fake=fake,
                      structural=True, family="radial",
                      cls_logits=cls_l, sel_logits=sel_l)
            new = uniq_wta_loss(s, _mask_of, gt, w, valid=None, **kw)
            ref = _ref_wta(s, _mask_of, gt, w, **kw)
            assert torch.equal(new.total, ref.total), (seed, fake)
            assert set(ref.terms) <= set(new.terms)
            for key, v in ref.terms.items():
                assert torch.equal(new.terms[key], v), (seed, fake, key)
            if not fake:
                assert new.stats["uniq_winner"] == ref.stats["uniq_winner"]


def test_head_forward_bit_identical_with_all_switches_off():
    feat, extra, cond, hw, hm = _head_inputs()
    for cls in (UniQ4Head, UniQ5Head):
        h = _head(cls)
        h.eval()
        new = h(feat, extra, cond, hw, hm)
        ref = _ref_head_forward(h, feat, extra, cond, hw, hm)
        for key in ("s_all", "cls_logits", "sel_logits"):
            assert torch.equal(new[key], ref[key]), (cls.__name__, key)
        assert "aux_supervision" not in new and "aux_groups" not in new
        # the default head declares every switch off
        assert h.n_refine_layers == 0 and h.aux_groups == 0
        assert h.refine is None and not h.iou_head and h.sel_stability == 0.0
        assert getattr(h, "n_stages", 0) == 0


def test_defaults_are_off_everywhere():
    w = LossWeights()
    assert (w.uniq_iou, w.uniq_eps, w.uniq_hdrop) == (0.0, 0.0, 0.0)
    assert not w.uniq_iou_winner_only and not w.uniq_iou_mse
    assert VARIANT4.get("head_cls") is None
    d = w.to_dict()
    for key in ("uniq_iou", "uniq_eps", "uniq_hdrop", "uniq_iou_mse"):
        assert key in d                       # lands in config/ via setup()


# ===========================================================================
# EPR-012: IoU regression selection head
# ===========================================================================
def test_iou_head_shape_range_and_sam_mlp_form():
    h = _head(iou_head=True)
    feat, extra, cond, hw, hm = _head_inputs()
    out = h(feat, extra, cond, hw, hm)
    sel = out["sel_logits"].detach()
    assert sel.shape == (8,)
    assert float(sel.min()) >= 0.0 and float(sel.max()) <= 1.0
    assert len(h.sel.layers) == 3 and h.sel.sigmoid_output
    assert h.facts()["sel_head"] == "mlp_sigmoid_iou"


def test_iou_target_matches_evaluate_uniq_query_ious():
    """The regression target must be the SAME number the board reports."""
    from q3vl.whereb.metrics import gt_area_k, hard_iou, topk_mask

    s, gt = _fields(11)
    k = gt_area_k(gt)
    gt_k = topk_mask(gt, k)
    board = [hard_iou(topk_mask(_mask_of(s[q]).float(), k), gt_k)
             for q in range(8)]
    sel = torch.zeros(8, requires_grad=True)      # pred 0 -> MAE = mean(target)
    w = LossWeights(uniq_iou=0.05)
    sl = uniq_wta_loss(s, _mask_of, gt, w, sel_logits=sel)
    assert abs(sl.stats["uniq_iou_mae"] - sum(board) / 8) < 1e-6
    assert abs(float(sl.terms["uniq_iou"]) - sum(board) / 8) < 1e-6


def test_iou_loss_trains_sel_only_and_honours_ablations():
    s, gt = _fields(5)
    sel = torch.randn(8, requires_grad=True)
    base = uniq_wta_loss(s, _mask_of, gt, LossWeights(), sel_logits=sel)
    w = LossWeights(uniq_iou=1.0)
    sl = uniq_wta_loss(s, _mask_of, gt, w, sel_logits=sel)
    # the IoU term is an ADDITION to the field loss, not a change of it
    assert torch.equal(sl.total - w.uniq_iou * sl.terms["uniq_iou"], base.total)
    sl.total.backward()
    assert sel.grad is not None and float(sel.grad.abs().sum()) > 0
    # IoU must never become a field target: the target is a no_grad constant,
    # so removing the field loss leaves the IoU term with no field gradient.
    s2 = s.detach().clone().requires_grad_(True)
    sel2 = torch.randn(8, requires_grad=True)
    only = uniq_wta_loss(s2, _mask_of, gt, LossWeights(bce=0.0, sdf=0.0,
                                                      area=0.0, sep=0.0,
                                                      uniq_iou=1.0),
                         sel_logits=sel2)
    only.total.backward()
    assert s2.grad is None or float(s2.grad.abs().sum()) == 0.0
    # ablations
    won = uniq_wta_loss(s, _mask_of, gt,
                        LossWeights(uniq_iou=1.0, uniq_iou_winner_only=True),
                        sel_logits=sel)
    mse = uniq_wta_loss(s, _mask_of, gt,
                        LossWeights(uniq_iou=1.0, uniq_iou_mse=True),
                        sel_logits=sel)
    assert float(won.terms["uniq_iou"]) != float(sl.terms["uniq_iou"])
    assert float(mse.terms["uniq_iou"]) != float(sl.terms["uniq_iou"])


def test_stability_fallback_is_inference_only():
    h = _head(iou_head=True, sel_stability=0.98)
    s = torch.zeros(8, 6, 8)
    s[0] = 0.0                     # top pick: |logit| = 0 -> maximally unstable
    s[1] = 5.0
    u = {"s_all": s, "sel_logits": torch.tensor(
        [0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])}
    h.train()
    assert h.select_index(u) == 0            # training: plain argmax (SAM2 gate)
    h.eval()
    assert h.select_index(u) == 1            # unstable -> next-best pred_iou
    h.sel_stability = 0.0
    assert h.select_index(u) == 0            # off -> plain argmax


# ===========================================================================
# EPR-013: masked cross-attention refinement
# ===========================================================================
def test_refine_layers_are_identity_at_init():
    """The identity must come from ``__init__``, not from a re-zeroing here.

    Review S1: the earlier form randomised the whole head and then zeroed the
    exits by hand, so it proved "zeroed exits are an identity" -- a property of
    the arithmetic -- while the claim under test is that the CONSTRUCTOR ships
    those zeros.  A regression that dropped `nn.init.zeros_` passed it.  So:
    build fresh, randomise only the shared trunk, touch nothing under `refine.`.
    """
    torch.manual_seed(21)
    h = UniQ4Head(32, 64, 1, 2, 16, text_dim=48, n_queries=8,
                  n_refine_layers=2)
    with torch.no_grad():                    # randomise ONLY the shared trunk
        for name, p in h.named_parameters():
            if not name.startswith("refine."):
                p.copy_(torch.randn_like(p) * 0.1)
    assert len(h.refine) == 2
    feat, extra, cond, hw, hm = _head_inputs()
    h.eval()
    out = h(feat, extra, cond, hw, hm)
    aux = out["aux_supervision"]
    h.refine = None                          # same weights, no refinement
    plain = h(feat, extra, cond, hw, hm)
    assert torch.equal(out["s_all"], plain["s_all"])
    assert torch.equal(out["cls_logits"], plain["cls_logits"])
    # every intermediate field is the same field too, for the same reason
    assert len(aux) == 2
    for a in aux:
        assert torch.equal(a["s_all"], plain["s_all"])


def test_refine_emits_aux_fields_and_moves_the_query():
    h = _head(n_refine_layers=2)
    feat, extra, cond, hw, hm = _head_inputs()
    h.eval()
    out = h(feat, extra, cond, hw, hm)
    aux = out["aux_supervision"]
    assert len(aux) == 2                    # s^0 and s^1; the final is `s_all`
    assert all(a["s_all"].shape == (8, 6, 8) for a in aux)
    assert not torch.equal(aux[0]["s_all"], out["s_all"])   # trained-off layers
    out["s_all"].sum().backward()
    assert h.refine[0].xattn.out_proj.weight.grad is not None
    assert h.query_pos.grad is not None
    # the aux switch removes the intermediate supervision, not the layers
    h2 = _head(n_refine_layers=2, refine_aux_loss=False)
    h2.eval()
    assert "aux_supervision" not in h2(feat, extra, cond, hw, hm)


def test_refine_survives_an_all_negative_field():
    """p=0.15 foreign samples drive every field negative -> every attn row is
    fully masked.  Without M2F's full-row reset this is a NaN, not an edge."""
    h = _head(n_refine_layers=1)
    with torch.no_grad():
        h.to_mask.weight.zero_()
        h.to_mask.bias.fill_(-9.0)          # every field far below the threshold
    feat, extra, cond, hw, hm = _head_inputs()
    h.eval()
    out = h(feat, extra, cond, hw, hm)
    assert torch.isfinite(out["s_all"]).all()
    assert torch.isfinite(out["sel_logits"]).all()


def test_refine_anneal_schedule():
    h = _head(n_refine_layers=2, refine_anneal=True)
    assert float(h.attn_mask_probs.min()) == 1.0
    h.set_anneal_progress(1, 1200)
    assert float(h.attn_mask_probs[0]) == 1.0       # before its start step
    h.set_anneal_progress(1200, 1200)
    assert float(h.attn_mask_probs.max()) == 0.0    # fully open at the end
    off = _head(n_refine_layers=2)
    off.set_anneal_progress(1200, 1200)
    assert float(off.attn_mask_probs.min()) == 1.0  # no-op when the flag is off


def test_refine_anneal_unmasking_is_training_only():
    """Review B6: the anneal draws from the global RNG and randomly unmasks
    rows.  Under `eval()` that would make the board stochastic AND move the
    stream the training sampler rides on (N1), so the release is train-gated."""
    h = _head(n_refine_layers=1, refine_anneal=True)
    h.attn_mask_probs[0] = 0.0                  # fully annealed
    feat, extra, cond, hw, hm = _head_inputs()

    torch.manual_seed(99)
    r0 = torch.rand(3)
    h.eval()
    torch.manual_seed(99)
    ev = h(feat, extra, cond, hw, hm)
    assert torch.equal(torch.rand(3), r0)       # eval draws nothing
    torch.manual_seed(99)
    ev2 = h(feat, extra, cond, hw, hm)
    assert torch.equal(ev["s_all"], ev2["s_all"])   # ...and is deterministic

    h.train()
    torch.manual_seed(99)
    h(feat, extra, cond, hw, hm)
    assert not torch.equal(torch.rand(3), r0)   # training does anneal


# ===========================================================================
# EPR-014: K-Net kernel-update stages
# ===========================================================================
def test_kernel_update_stages_are_identity_at_init():
    torch.manual_seed(33)
    h = UniQ5Head(32, 64, 1, 2, 16, text_dim=48, n_queries=8, n_stages=3)
    with torch.no_grad():                   # randomise only the SHARED trunk
        for name, p in h.named_parameters():
            if not name.startswith("stages."):
                p.copy_(torch.randn_like(p) * 0.1)
    feat, extra, cond, hw, hm = _head_inputs()
    h.eval()
    out = h(feat, extra, cond, hw, hm)
    assert len(out["aux_supervision"]) == 3
    for a in out["aux_supervision"]:         # zero-gated stages => q unchanged
        assert torch.equal(a["s_all"], out["s_all"])
        assert torch.equal(a["cls_logits"], out["cls_logits"])
    h.n_stages = 0
    assert torch.equal(h(feat, extra, cond, hw, hm)["s_all"], out["s_all"])


def test_kernel_update_gates_start_neutral_and_train():
    h = UniQ5Head(32, 64, 1, 2, 16, text_dim=48, n_queries=8, n_stages=2)
    st = h.stages[0]
    q = torch.randn(8, 32)
    code = torch.randn(32, 48)
    fb = (torch.rand(8, 48) > 0.5).float()
    gates = st.input_gate(st.input_layer(q).split(32, -1)[0]
                          * st.dynamic_layer(fb @ code.t()).split(32, -1)[0])
    assert torch.equal(gates, torch.zeros_like(gates))   # -> sigmoid = 0.5
    out = st(q, code, fb)
    assert torch.equal(out, q)                           # zero-init exits
    out.sum().backward()
    assert st.dynamic_layer.weight.grad is not None
    assert st.out_proj.weight.grad is not None


def test_kernel_update_empty_field_gives_zero_group_feature():
    """An all-zero binarised field must contribute the zero vector (K-Net's
    sum aggregation has no division), not a NaN."""
    h = UniQ5Head(32, 64, 1, 2, 16, text_dim=48, n_queries=8, n_stages=1)
    with torch.no_grad():
        h.to_mask.weight.zero_()
        h.to_mask.bias.fill_(-9.0)
    feat, extra, cond, hw, hm = _head_inputs()
    h.eval()
    assert torch.isfinite(h(feat, extra, cond, hw, hm)["s_all"]).all()


def test_kernel_update_ablation_switches():
    kw = dict(text_dim=48, n_queries=8, n_stages=2)
    soft = UniQ5Head(32, 64, 1, 2, 16, soft_feedback=True, **kw)
    assert soft.facts()["stage_feedback"] == "soft"
    per = UniQ5Head(32, 64, 1, 2, 16, per_stage_to_mask=True, **kw)
    assert len(per.stage_to_mask) == 2
    noatt = UniQ5Head(32, 64, 1, 2, 16, stage_query_attn=False, **kw)
    assert noatt.stages[0].attn is None
    last = UniQ5Head(32, 64, 1, 2, 16, stage_supervision=False, **kw)
    last.eval()
    assert "aux_supervision" not in last(*_head_inputs())


# ===========================================================================
# EPR-015: relaxed WTA
# ===========================================================================
def test_relaxed_wta_gives_losers_gradient_and_keeps_sep_winner_only():
    s, gt = _fields(9)
    partner = (torch.rand(6, 8) > 0.7).float()
    w0 = LossWeights()
    hard = uniq_wta_loss(s, _mask_of, gt, w0, gt_partner=partner)
    hard.total.backward()
    rows_hit = int((s.grad.reshape(8, -1).abs().sum(-1) > 0).sum())
    assert rows_hit == 1                          # baseline: winner only

    s2 = s.detach().clone().requires_grad_(True)
    relaxed = uniq_wta_loss(s2, _mask_of, gt, LossWeights(uniq_eps=0.05),
                            gt_partner=partner)
    relaxed.total.backward()
    assert int((s2.grad.reshape(8, -1).abs().sum(-1) > 0).sum()) == 8

    # the delta-hat weights sum to 1 and sep stays winner-only: rebuild the
    # published formula and compare.
    w = LossWeights(uniq_eps=0.05)
    per = [amort_sample_loss(_mask_of(s[k]), gt, w, gt_partner=partner)
           for k in range(8)]
    j = int(torch.stack([p.total.detach() for p in per]).argmin())
    exp = sum(((1 - 0.05) if k == j else 0.05 / 7)
              * (per[k].total - w.sep * per[k].terms["sep"]) for k in range(8))
    exp = exp + w.sep * per[j].terms["sep"]
    assert abs(float(relaxed.total) - float(exp)) < 1e-5


def test_relaxed_wta_zero_eps_touches_no_rng():
    # The claim is "eps=0, hdrop=0 does not TOUCH the global stream", so the
    # reference draw must be taken with the loss never called at all -- calling
    # it on both sides would pass just as happily if it consumed the same
    # number of draws each time, which is a different (and weaker) property.
    s, gt = _fields(4)
    seed = 1234
    torch.manual_seed(seed)
    r0 = torch.rand(3)
    torch.manual_seed(seed)
    a = uniq_wta_loss(s, _mask_of, gt, LossWeights())      # eps=0, hdrop=0
    assert torch.equal(torch.rand(3), r0)                  # stream untouched
    torch.manual_seed(seed)
    b = uniq_wta_loss(s, _mask_of, gt, LossWeights())
    assert torch.equal(a.total, b.total)
    # hdrop > 0 does draw, and never crashes even at p = 1.0
    out = uniq_wta_loss(s, _mask_of, gt, LossWeights(uniq_eps=0.05,
                                                     uniq_hdrop=1.0))
    assert torch.isfinite(out.total)


def test_hypothesis_dropout_keeps_a_winner():
    s, gt = _fields(6)
    for _ in range(20):
        out = uniq_wta_loss(s, _mask_of, gt,
                            LossWeights(uniq_eps=0.05, uniq_hdrop=0.5))
        assert torch.isfinite(out.total)
        assert 0 <= out.stats["uniq_winner"] < 8


# ===========================================================================
# EPR-016: training-time one-to-many auxiliary query groups
# ===========================================================================
def test_aux_groups_read_extra_rows_only_while_training():
    h = _head(aux_groups=2)
    feat, extra, cond, hw, hm = _head_inputs(rows=24, t=20)
    h.train()
    tr = h(feat, extra, cond, hw, hm)
    assert tr["s_all"].shape == (24, 6, 8)
    assert tr["sel_logits"].shape == (8,)          # selection = formal group
    assert tr["aux_groups"] == 2 and tr["aux_lambda"] == 1.0
    h.eval()
    ev = h(feat, extra, cond, hw, hm)
    assert ev["s_all"].shape == (8, 6, 8)
    assert "aux_groups" not in ev
    # the eval forward reads the LAST 8 rows -- the formal group's rows are the
    # ones the no-grad encode branch appends, so this is the deployed path
    plain = _head(seed=7)
    plain.load_state_dict(h.state_dict())
    plain.eval()
    assert torch.equal(ev["s_all"], plain(feat, extra, cond, hw, hm)["s_all"])


def test_group_wta_is_main_plus_lambda_over_m_times_group_mean():
    torch.manual_seed(2)
    s = torch.randn(24, 6, 8) * 1.5
    gt = (torch.rand(6, 8) > 0.6).float()
    cls_l = torch.randn(24, len(UNIQ_FAMILIES))
    sel_l = torch.randn(8)
    w = LossWeights(uniq_cls=0.05, uniq_sel=0.05)
    got = uniq_wta_loss(s, _mask_of, gt, w, cls_logits=cls_l,
                        sel_logits=sel_l, aux_groups=2, aux_lambda=1.0)
    main = uniq_wta_loss(s[:8], _mask_of, gt, w, cls_logits=cls_l[:8],
                         sel_logits=sel_l)
    exp = main.total
    for g in (1, 2):
        sub = uniq_wta_loss(s[8 * g:8 * g + 8], _mask_of, gt, w,
                            cls_logits=cls_l[8 * g:8 * g + 8])
        exp = exp + 0.5 * sub.total                       # lambda/m = 1/2
        assert "uniq_sel" not in sub.terms                # aux never feeds sel
    assert abs(float(got.total) - float(exp)) < 1e-6
    assert "bce_aux0" in got.terms and "bce_aux1" in got.terms
    assert "uniq_sel" in got.terms                        # formal group keeps it


def test_group_wta_fake_charges_every_group():
    torch.manual_seed(8)
    s = torch.randn(16, 6, 8)
    gt = torch.zeros(6, 8)
    got = uniq_wta_loss(s, _mask_of, gt, LossWeights(), is_fake=True,
                        aux_groups=1, aux_lambda=1.0)
    solo = uniq_wta_loss(s[:8], _mask_of, gt, LossWeights(), is_fake=True)
    assert float(got.total) > float(solo.total)
    assert "fake_aux0" in got.terms


# ===========================================================================
# trainer wiring: the criteria are CALLED, not merely defined
# ===========================================================================
@dataclass
class _Stub:
    feat: Any
    cond_h: Any
    gt_low: Any
    sim: Any
    grid_h: int = 6
    grid_w: int = 8
    sample_id: str = "s0"
    family: str = "radial"
    is_fake: bool = False
    route_semantic: bool = False
    cond_mask: Any = None
    center: Any = None
    geom: Any = None
    guide_hi: Any = None
    gt_partner_low: Any = None
    word_ids: Any = field(default_factory=lambda: torch.tensor([0]))
    word_offsets: Any = field(default_factory=lambda: torch.tensor([0]))
    phi_dir: Any = field(default_factory=lambda: torch.zeros(48, 71))


def _model_and_batch(**head_kwargs):
    VARIANT4.update(n_qtok=8, lora=False, vlm_ref=None, head_cls=None,
                    head_kwargs=head_kwargs)
    if head_kwargs.get("n_stages"):
        VARIANT4["head_cls"] = UniQ5Head
    torch.manual_seed(5)
    m = AmortModelV4("UNIQ", in_dim=64, ch=32, n_blocks=2, sem_ch=32,
                     cond_text_dim=48, cond_out=8, n_words=8, word_dim=8)
    rows = 8 * (1 + head_kwargs.get("aux_groups", 0))
    x = _Stub(feat=torch.randn(1, 64, 6, 8),
              cond_h=torch.randn(1, 12 + rows, 48),
              gt_low=(torch.rand(6, 8) > 0.6).float(),
              sim=torch.rand(1, 1, 6, 8))
    VARIANT4.update(head_cls=None, head_kwargs={})       # leave no global state
    return m, [x]


def test_trainer_runs_the_deep_supervision_terms():
    for kwargs, tag in ((dict(n_refine_layers=2), "ref0"),
                        (dict(n_stages=2), "st0")):
        m, batch = _model_and_batch(**kwargs)
        m.train()
        total, stats, _ = compute_micro_batch(m, batch, LossWeights(
            uniq_cls=0.05, uniq_sel=0.05))
        assert torch.isfinite(total)
        assert f"L_bce_{tag}" in stats, (kwargs, sorted(stats))
        total.backward()


def test_trainer_runs_the_group_and_iou_terms():
    m, batch = _model_and_batch(aux_groups=2)
    m.train()
    total, stats, _ = compute_micro_batch(m, batch, LossWeights(uniq_cls=0.05,
                                                                uniq_sel=0.05))
    assert torch.isfinite(total) and "L_bce_aux0" in stats
    m2, batch2 = _model_and_batch()
    m2.train()
    _, stats2, _ = compute_micro_batch(m2, batch2,
                                       LossWeights(uniq_iou=0.05))
    assert "L_uniq_iou" in stats2                # lands in steps.jsonl


def test_uniq_mechanism_stats_land_in_the_step_row():
    """EPR-012/015's read-outs must reach `steps.jsonl`, not just the sample.

    Review B2: `_uniq_wta_group` computed `uniq_iou_mae`, `uniq_winner`,
    `uniq_sel_correct` and friends per sample and `aggregate` dropped every one
    of them, so the arms named after those mechanisms had no column showing the
    mechanism doing anything.
    """
    m, batch = _model_and_batch()
    m.train()
    _, stats, _ = compute_micro_batch(m, batch, LossWeights(
        uniq_cls=0.05, uniq_sel=0.05, uniq_iou=0.05))
    for key in ("uniq_iou_mae_mean", "uniq_iou_mae_median",
                "uniq_iou_target_best_mean", "uniq_winner_mean",
                "uniq_winner_median", "uniq_sel_correct_mean"):
        assert key in stats, (key, sorted(stats))
    assert 0.0 <= stats["uniq_iou_target_best_mean"] <= 1.0
    assert stats["uniq_winner_n"] == len(batch)
    m2, batch2 = _model_and_batch(aux_groups=2)
    m2.train()
    _, stats2, _ = compute_micro_batch(m2, batch2, LossWeights())
    assert stats2["uniq_aux_groups_mean"] == 2.0
    assert "uniq_winner_aux0_mean" in stats2


def test_configured_aux_groups_cannot_vanish_from_the_forward():
    """An eval-mode model deletes the EPR-016 rows; that must be an error."""
    m, batch = _model_and_batch(aux_groups=2)
    m.eval()                       # e.g. a quick eval that never restored train
    _raises(lambda: compute_micro_batch(m, batch, LossWeights()), "aux_groups")


def test_baseline_micro_batch_is_unchanged_by_the_pack():
    m, batch = _model_and_batch()
    m.train()
    total, stats, _ = compute_micro_batch(m, batch, LossWeights(uniq_cls=0.05,
                                                                uniq_sel=0.05))
    extra = [k for k in stats if k.endswith(("_ref0", "_st0", "_aux0"))]
    assert not extra and "L_uniq_iou" not in stats
    assert torch.isfinite(total)


def test_deep_supervision_columns_are_asserted_from_the_first_step_row():
    """Review B1 / EPR-014's pre-registered runtime assertion.

    A refine layer, a kernel-update stage and an auxiliary query group leave
    exactly one trace: a suffixed ``L_*`` column.  The board therefore has to
    prove the branches ran, from the FIRST ``steps.jsonl`` row -- a branch that
    is absent is absent from step one.
    """
    from q3vl.whereb.amort.evaluate import (assert_criteria_ran,
                                            deep_supervision_tags)

    board = {"criteria_columns": {"uniq_best": {"n": 200}}}
    facts = {"n_stages": 2, "n_refine_layers": 1, "aux_groups": 2}
    tags = deep_supervision_tags(facts)
    assert tags == ["st0", "st1", "ref0", "aux0", "aux1"]

    full = {"step": 1, "loss": 1.0, "L_bce": 0.4,
            **{f"L_bce_{t}": 0.5 for t in tags},
            **{f"L_sep_{t}": 0.1 for t in tags}}
    rep = assert_criteria_ran(board, "UNIQ", head_facts=facts, steps_row=full)
    assert rep["deep_supervision"]["checked"]
    assert rep["deep_supervision"]["columns"]["st1"] == ["L_bce_st1", "L_sep_st1"]

    # every single branch is load-bearing: drop one column set and it raises
    for tag in tags:
        partial = {k: v for k, v in full.items() if not k.endswith(f"_{tag}")}
        _raises(lambda p=partial: assert_criteria_ran(
            board, "UNIQ", head_facts=facts, steps_row=p), tag)
    # no log at all is a failure, never a silent pass
    _raises(lambda: assert_criteria_ran(board, "UNIQ", head_facts=facts),
            "steps.jsonl")

    # the pre-registered ablations that legally emit no intermediate columns
    assert deep_supervision_tags({"n_stages": 2,
                                  "stage_supervision": False}) == []
    assert deep_supervision_tags({"n_refine_layers": 2,
                                  "refine_aux_loss": False}) == []
    # a baseline head has nothing to check and needs no log
    base = assert_criteria_ran(board, "UNIQ", head_facts=UniQ4Head(
        32, 64, 1, 2, 16, text_dim=48, n_queries=8).facts())
    assert base["deep_supervision"]["tags"] == []
    assert not base["deep_supervision"]["checked"]


def test_step_row_of_a_configured_arm_satisfies_the_assertion():
    """End to end: the columns a configured head really produces are the ones
    the assertion asks for (a hand-written tag list would drift)."""
    from q3vl.whereb.amort.evaluate import assert_criteria_ran

    board = {"criteria_columns": {"uniq_best": {"n": 200}}}
    for kwargs in (dict(n_refine_layers=2), dict(n_stages=2),
                   dict(aux_groups=2), dict(n_stages=1, aux_groups=1)):
        m, batch = _model_and_batch(**kwargs)
        m.train()
        _, stats, _ = compute_micro_batch(m, batch, LossWeights(uniq_cls=0.05,
                                                                uniq_sel=0.05))
        row = {"step": 1, **stats}
        rep = assert_criteria_ran(board, "UNIQ", head_facts=m.geo.facts(),
                                  steps_row=row)
        assert rep["deep_supervision"]["tags"], kwargs
        for tag, cols in rep["deep_supervision"]["columns"].items():
            assert cols, (kwargs, tag)


if __name__ == "__main__":
    import sys

    mod = sys.modules[__name__]
    for name in [n for n in dir(mod) if n.startswith("test_")]:
        getattr(mod, name)()
        print(f"  ok {name}")
    print("EPR-012..016: all checks passed")
