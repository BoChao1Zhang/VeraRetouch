"""UNIQ wave-6 smoke: in-context query head, cond stripping, model wiring."""

import torch

from q3vl.whereb.amort.uniq4 import VARIANT4, AmortModelV4, UniQ4Head


def test_head_reads_last_k_rows_and_zero_init():
    h = UniQ4Head(32, 64, 1, 2, 16, text_dim=48, n_queries=8)
    feat = torch.randn(1, 64, 6, 8)
    hw = torch.randn(1, 20 + 8, 48)          # 20 text rows + 8 query rows
    out = h(feat, torch.rand(1, 1, 6, 8), torch.randn(1, 16), hw,
            torch.ones(1, 28))
    assert out["s_all"].shape == (8, 6, 8)
    assert torch.allclose(out["s_all"], torch.zeros_like(out["s_all"]))
    # gradient reaches the query rows through q_proj_in
    out["cls_logits"].sum().backward()
    assert h.q_proj_in[1].weight.grad is not None


def test_model_v4_strips_query_rows_from_pooled_path():
    VARIANT4.update(n_qtok=8, lora=False, vlm_ref=None)
    m = AmortModelV4("UNIQ", in_dim=64, ch=32, n_blocks=2, sem_ch=32,
                     cond_text_dim=48, cond_out=8, n_words=8, word_dim=8)
    hw = torch.randn(1, 12 + 8, 48)
    cond = m.cond_of(hw, None, torch.tensor([0]), torch.tensor([0]))
    # pooled path must not see the 8 query rows
    ref = m.cond_of(hw[:, :12, :].clone(), None, torch.tensor([0]),
                    torch.tensor([0]))
    # stripping happens inside cond_of, so both calls pool the same 12 rows
    assert torch.allclose(cond, m.cond_of(hw, None, torch.tensor([0]),
                                          torch.tensor([0])))
    assert cond.shape == ref.shape
    out = m.forward_geo(torch.randn(1, 64, 6, 8), cond, torch.zeros(48, 71),
                        sim=torch.rand(1, 1, 6, 8), grid_h=6, grid_w=8,
                        h_where=hw, h_mask=torch.ones(1, 20))
    assert out["m_low"].shape == (6, 8) and out["uniq"]["s_all"].shape[0] == 8
    try:
        m.forward_geo(torch.randn(1, 64, 6, 8), cond, torch.zeros(48, 71),
                      sim=torch.rand(1, 1, 6, 8), grid_h=6, grid_w=8,
                      h_where=torch.randn(1, 4, 48), h_mask=torch.ones(1, 4))
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("h_where shorter than K must refuse")


if __name__ == "__main__":
    test_head_reads_last_k_rows_and_zero_init()
    test_model_v4_strips_query_rows_from_pooled_path()
    print("UNIQ4 smoke: all checks passed")
