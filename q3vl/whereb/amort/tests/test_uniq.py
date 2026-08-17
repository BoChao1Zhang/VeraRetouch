"""UNIQ head smoke checks (EPR-011): shapes, zero-init, WTA wiring."""

import torch

from q3vl.whereb.amort.losses import LossWeights, uniq_wta_loss
from q3vl.whereb.amort.model import AmortModel
from q3vl.whereb.amort.uniq import UNIQ_FAMILIES, UniQHead, fourier_grid


def _head(**kw):
    return UniQHead(ch=32, in_dim=64, extra_ch=1, n_blocks=2, cond_dim=16,
                    text_dim=48, n_queries=4, **kw)


def _inputs(gh=6, gw=8, t=11):
    return (torch.randn(1, 64, gh, gw), torch.rand(1, 1, gh, gw),
            torch.randn(1, 16), torch.randn(1, t, 48),
            torch.ones(1, t))


def test_shapes_and_zero_init():
    for bands in (0, 8):
        h = _head(fourier_bands=bands)
        feat, extra, cond, hw, hm = _inputs()
        out = h(feat, extra, cond, hw, hm)
        assert out["s_all"].shape == (4, 6, 8)
        assert out["cls_logits"].shape == (4, len(UNIQ_FAMILIES))
        assert out["sel_logits"].shape == (4,)
        # zero-init discipline: every query field is exactly 0 at step 0
        assert torch.allclose(out["s_all"], torch.zeros_like(out["s_all"]))
        assert torch.allclose(h.mask_of(out["s_all"]),
                              torch.full_like(out["s_all"], 0.5))


def test_fourier_grid_deterministic_and_sized():
    a = fourier_grid(6, 8, bands=8, scale=1.0, seed=77)
    b = fourier_grid(6, 8, bands=8, scale=1.0, seed=77)
    assert a.shape == (2 + 2 * 8, 6, 8)
    assert torch.equal(a, b)
    c = fourier_grid(6, 8, bands=8, scale=1.0, seed=78)
    assert not torch.equal(a, c)


def test_wta_loss_trains_winner_and_heads():
    h = _head(fourier_bands=8)
    feat, extra, cond, hw, hm = _inputs()
    out = h(feat, extra, cond, hw, hm)
    gt = torch.rand(6, 8)
    w = LossWeights(uniq_cls=0.05, uniq_sel=0.05)
    sl = uniq_wta_loss(out["s_all"], h.mask_of, gt, w, family="radial",
                       cls_logits=out["cls_logits"],
                       sel_logits=out["sel_logits"])
    assert torch.isfinite(sl.total)
    assert "uniq_cls" in sl.terms and "uniq_sel" in sl.terms
    assert 0 <= sl.stats["uniq_winner"] < 4
    sl.total.backward()
    assert h.to_mask.weight.grad is not None
    assert h.cls.weight.grad is not None and h.sel.weight.grad is not None


def test_wta_fake_charges_all_queries():
    h = _head()
    feat, extra, cond, hw, hm = _inputs()
    out = h(feat, extra, cond, hw, hm)
    sl = uniq_wta_loss(out["s_all"], h.mask_of, torch.rand(6, 8),
                       LossWeights(), is_fake=True,
                       cls_logits=out["cls_logits"],
                       sel_logits=out["sel_logits"])
    # zero-init masks are 0.5 everywhere for all K -> fake term is exactly 0.5
    assert abs(float(sl.terms["fake"]) - 0.5) < 1e-6


def test_amort_model_uniq_arm_forward_and_guard():
    m = AmortModel("UNIQ", in_dim=64, ch=32, n_blocks=2, sem_ch=32,
                   cond_text_dim=48, cond_out=8, n_words=8, word_dim=8,
                   uniq_k=4, uniq_fourier_bands=8)
    feat = torch.randn(1, 64, 6, 8)
    sim = torch.rand(1, 1, 6, 8)
    cond = torch.randn(1, m.cond.out_dim)
    hw, hm = torch.randn(1, 9, 48), torch.ones(1, 9)
    out = m.forward_geo(feat, cond, torch.zeros(48, 71), sim=sim,
                        grid_h=6, grid_w=8, h_where=hw, h_mask=hm)
    assert out["m_low"].shape == (6, 8)
    assert "uniq" in out and out["uniq"]["s_all"].shape[0] == 4
    try:
        m.forward_geo(feat, cond, torch.zeros(48, 71), sim=sim,
                      grid_h=6, grid_w=8)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("UNIQ without h_where must refuse to run")


if __name__ == "__main__":
    test_shapes_and_zero_init()
    test_fourier_grid_deterministic_and_sized()
    test_wta_loss_trains_winner_and_heads()
    test_wta_fake_charges_all_queries()
    test_amort_model_uniq_arm_forward_and_guard()
    print("UNIQ smoke: all checks passed")
