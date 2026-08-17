"""UNIQ wave-3 smoke checks: variants, seams, presence statefulness, FQ forcing."""

import torch

from q3vl.whereb.amort.losses import LossWeights
from q3vl.whereb.amort.uniq import UNIQ_FAMILIES
from q3vl.whereb.amort.uniq2 import (VARIANT, AmortModelV2, UniQ2Head,
                                     dispatching_wta)


def _head(**kw):
    return UniQ2Head(32, 64, 1, 2, 16, text_dim=48, n_queries=4, **kw)


def _inputs(gh=6, gw=8, t=11):
    return (torch.randn(1, 64, gh, gw), torch.rand(1, 1, gh, gw),
            torch.randn(1, 16), torch.randn(1, t, 48), torch.ones(1, t))


def test_zero_init_holds_for_every_variant():
    for kw in ({"presence": True}, {"pix_attn": True},
               {"family_queries": True}, {"presence": True, "pix_attn": True}):
        h = _head(**kw)
        out = h(*_inputs())
        assert torch.allclose(out["s_all"], torch.zeros_like(out["s_all"])), kw


def test_presence_gates_mask_and_is_stateful():
    h = _head(presence=True)
    try:
        h.mask_of(torch.zeros(6, 8))
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("mask_of before forward must refuse")
    h(*_inputs())
    m = h.mask_of(torch.zeros(6, 8))
    p0 = torch.sigmoid(torch.tensor(2.0))
    assert torch.allclose(m, torch.full_like(m, float(p0) * 0.5), atol=1e-5)
    # foreign instruction: empty-mask loss must push presence DOWN (bias grad > 0)
    out = h(*_inputs())
    sl = dispatching_wta(out["s_all"], h.mask_of, torch.rand(6, 8),
                         LossWeights(), is_fake=True,
                         cls_logits=out["cls_logits"],
                         sel_logits=out["sel_logits"])
    sl.total.backward()
    assert h.presence.bias.grad is not None and float(h.presence.bias.grad) > 0


def test_family_queries_forces_winner():
    h = _head(family_queries=True)
    out = h(*_inputs())
    VARIANT["family_queries"] = True
    try:
        for fam in UNIQ_FAMILIES:
            sl = dispatching_wta(out["s_all"].detach(), h.mask_of,
                                 torch.rand(6, 8),
                                 LossWeights(uniq_cls=0.05, uniq_sel=0.05),
                                 family=fam,
                                 cls_logits=out["cls_logits"].detach(),
                                 sel_logits=out["sel_logits"].detach())
            assert sl.stats["uniq_winner"] == float(UNIQ_FAMILIES.index(fam))
    finally:
        VARIANT["family_queries"] = False


def test_dispatch_off_matches_original_winner_semantics():
    h = _head()
    out = h(*_inputs())
    sl = dispatching_wta(out["s_all"], h.mask_of, torch.rand(6, 8),
                         LossWeights(), family="radial",
                         cls_logits=out["cls_logits"],
                         sel_logits=out["sel_logits"])
    assert 0 <= sl.stats["uniq_winner"] < 4 and torch.isfinite(sl.total)


def test_model_v2_builds_variant_head():
    VARIANT.update(presence=True, pix_attn=True, family_queries=False)
    try:
        m = AmortModelV2("UNIQ", in_dim=64, ch=32, n_blocks=2, sem_ch=32,
                         cond_text_dim=48, cond_out=8, n_words=8, word_dim=8,
                         uniq_k=4)
        assert isinstance(m.geo, UniQ2Head)
        assert m.geo.presence_on and m.geo.pix_attn_on
        out = m.forward_geo(torch.randn(1, 64, 6, 8),
                            torch.randn(1, m.cond.out_dim),
                            torch.zeros(48, 71), sim=torch.rand(1, 1, 6, 8),
                            grid_h=6, grid_w=8,
                            h_where=torch.randn(1, 9, 48),
                            h_mask=torch.ones(1, 9))
        assert out["m_low"].shape == (6, 8)
        assert m.facts()["uniq"]["variant"]["presence"] is True
    finally:
        VARIANT.update(presence=False, pix_attn=False, family_queries=False)


if __name__ == "__main__":
    test_zero_init_holds_for_every_variant()
    test_presence_gates_mask_and_is_stateful()
    test_family_queries_forces_winner()
    test_dispatch_off_matches_original_winner_semantics()
    test_model_v2_builds_variant_head()
    print("UNIQ2 smoke: all checks passed")
