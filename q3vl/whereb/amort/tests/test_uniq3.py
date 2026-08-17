"""UNIQ wave-5 smoke checks: readout variants, seams, LoRA plumbing basics."""

import torch

from q3vl.whereb.amort.uniq3 import (VARIANT3, AmortModelV3, UniQ3Head,
                                     _ENCODE_NOGRADLESS)


def _head(readout, **kw):
    return UniQ3Head(32, 64, kw.pop("extra_ch", 1), 2, 16, text_dim=48,
                     n_queries=4, readout=readout, **kw)


def _inputs(gh=6, gw=8, t=11):
    return (torch.randn(1, 64, gh, gw), torch.rand(1, 1, gh, gw),
            torch.randn(1, 16), torch.randn(1, t, 48), torch.ones(1, t))


def test_hidden_readout_matches_base_shapes_and_zero_init():
    h = _head("hidden")
    out = h(*_inputs())
    assert out["s_all"].shape == (4, 6, 8)
    assert torch.allclose(out["s_all"], torch.zeros_like(out["s_all"]))


def test_code_readout_consumes_code_and_null_path():
    h = _head("code")
    feat, extra, cond, hw, hm = _inputs()
    code = torch.zeros(21)
    code[[1, 7, 16]] = 1.0
    out = h(feat, extra, cond, hw, hm, geom=code)
    assert out["s_all"].shape == (4, 6, 8)
    # empty code -> null token path, must not crash and stays zero-init
    out0 = h(feat, extra, cond, hw, hm, geom=torch.zeros(21))
    assert torch.allclose(out0["s_all"], torch.zeros_like(out0["s_all"]))
    # code path must actually be differentiable through the embedding
    loss = h(feat, extra, cond, hw, hm, geom=code)["cls_logits"].sum()
    loss.backward()
    assert h.code_embed.grad is not None


def test_attn_readout_ignores_text_and_uses_wider_stem():
    h = _head("attn", extra_ch=5)          # sim(1) + 4 attention channels
    feat = torch.randn(1, 64, 6, 8)
    extra = torch.rand(1, 5, 6, 8)
    out = h(feat, extra, torch.randn(1, 16), torch.randn(1, 11, 48),
            torch.ones(1, 11))
    assert out["s_all"].shape == (4, 6, 8)


def test_model_v3_code_kv_and_geom_stripped_from_channels():
    VARIANT3.update(readout="code", lora=False)
    try:
        m = AmortModelV3("UNIQ", in_dim=64, ch=32, n_blocks=2, sem_ch=32,
                         cond_text_dim=48, cond_out=8, n_words=8, word_dim=8,
                         uniq_k=4, geom_inject=True)
        # geom_inject intercepted: no broadcast channels on the stem
        assert m.geom_dim == 0 and m._v3_geom_kv is True
        code = torch.zeros(21)
        code[[0, 6, 15]] = 1.0
        out = m.forward_geo(torch.randn(1, 64, 6, 8),
                            torch.randn(1, m.cond.out_dim),
                            torch.zeros(48, 71), sim=torch.rand(1, 1, 6, 8),
                            geom=code, grid_h=6, grid_w=8,
                            h_where=torch.randn(1, 9, 48),
                            h_mask=torch.ones(1, 9))
        assert out["m_low"].shape == (6, 8) and "uniq" in out
    finally:
        VARIANT3.update(readout="hidden")


def test_nogradless_encode_body_is_reachable():
    # the LoRA seam depends on @torch.no_grad exposing the original
    assert callable(_ENCODE_NOGRADLESS)
    assert _ENCODE_NOGRADLESS.__name__ == "encode"


if __name__ == "__main__":
    test_hidden_readout_matches_base_shapes_and_zero_init()
    test_code_readout_consumes_code_and_null_path()
    test_attn_readout_ignores_text_and_uses_wider_stem()
    test_model_v3_code_kv_and_geom_stripped_from_channels()
    test_nogradless_encode_body_is_reachable()
    print("UNIQ3 smoke: all checks passed")
