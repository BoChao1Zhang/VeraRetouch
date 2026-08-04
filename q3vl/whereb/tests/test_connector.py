"""Connector: zero-init gates, the null context, and the block order."""

from __future__ import annotations

import torch

from q3vl.whereb.config import ConnectorConfig
from q3vl.whereb.connector import ConnectorBlock, ConnectorStream, MultiheadAttention


def _cfg(**kw) -> ConnectorConfig:
    base = {"dim": 32, "n_blocks": 2, "n_heads": 4, "ffn": 64,
            "text_dim": 48, "vision_dim": 24, "pos_bands": 4}
    base.update(kw)
    return ConnectorConfig(**base)


def test_cross_attention_gates_start_at_exactly_zero():
    blk = ConnectorBlock(_cfg())
    assert float(blk.gate_text.detach()) == 0.0
    assert float(blk.gate_vis.detach()) == 0.0


def test_block_output_at_init_ignores_both_key_sources():
    torch.manual_seed(0)
    blk = ConnectorBlock(_cfg()).eval()
    q = torch.randn(2, 5, 32)
    a = blk(q, torch.randn(2, 7, 32), None, torch.randn(2, 9, 32), None)
    b = blk(q, torch.randn(2, 7, 32) * 50, None, torch.randn(2, 9, 32) - 3, None)
    assert torch.allclose(a, b, atol=0, rtol=0)


def test_gate_actually_opens_when_it_is_non_zero():
    torch.manual_seed(0)
    blk = ConnectorBlock(_cfg()).eval()
    with torch.no_grad():
        blk.gate_text.fill_(1.0)
    q = torch.randn(2, 5, 32)
    a = blk(q, torch.randn(2, 7, 32), None, torch.randn(2, 9, 32), None)
    b = blk(q, torch.randn(2, 7, 32) * 50, None, torch.randn(2, 9, 32), None)
    assert not torch.allclose(a, b)


def test_fully_masked_keys_give_exactly_zero_not_nan():
    """The null context has no <where> token at all (protocol 5.4)."""
    torch.manual_seed(0)
    attn = MultiheadAttention(32, 4).eval()
    q = torch.randn(3, 5, 32)
    kv = torch.randn(3, 7, 32)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1] = False                                   # sample 1: null context
    out = attn(q, kv, mask)
    assert torch.isfinite(out).all()
    assert torch.equal(out[1], torch.zeros_like(out[1]))
    assert not torch.equal(out[0], torch.zeros_like(out[0]))


def test_masked_keys_do_not_influence_other_samples():
    torch.manual_seed(0)
    attn = MultiheadAttention(32, 4).eval()
    q = torch.randn(2, 4, 32)
    kv = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[:, 3:] = False
    out_masked = attn(q, kv, mask)
    out_short = attn(q, kv[:, :3], torch.ones(2, 3, dtype=torch.bool))
    assert torch.allclose(out_masked, out_short, atol=1e-5)


def test_stream_projects_both_modalities_to_the_connector_width():
    cfg = _cfg()
    s = ConnectorStream(cfg)
    assert s.text_proj.in_features == cfg.text_dim
    assert s.vision_proj.in_features == cfg.vision_dim
    assert s.text_proj.out_features == s.vision_proj.out_features == cfg.dim


def test_stream_forward_shape_and_zero_gate_report():
    torch.manual_seed(0)
    cfg = _cfg()
    s = ConnectorStream(cfg).eval()
    q = torch.randn(2, 16, cfg.dim)
    out = s(q, torch.randn(2, 5, cfg.text_dim), None,
            torch.randn(2, 12, cfg.vision_dim), torch.randn(2, 12, 2), None)
    assert out.shape == (2, 16, cfg.dim)
    g = s.gate_values()
    assert g["gate_text"] == [0.0] * cfg.n_blocks
    assert g["gate_vis"] == [0.0] * cfg.n_blocks


def test_block_applies_operations_in_the_protocol_order():
    """self-attn -> xattn(H_where) -> xattn(F_pre) -> FFN.

    Checked by disabling one branch at a time and observing which input the
    output stops depending on.
    """
    torch.manual_seed(0)
    blk = ConnectorBlock(_cfg()).eval()
    q = torch.randn(1, 4, 32)
    text, vis = torch.randn(1, 6, 32), torch.randn(1, 8, 32)
    with torch.no_grad():
        blk.gate_text.fill_(1.0)
        blk.gate_vis.fill_(0.0)
    only_text = blk(q, text, None, vis, None)
    assert not torch.allclose(only_text, blk(q, text * 2, None, vis, None))
    assert torch.allclose(only_text, blk(q, text, None, vis * 7 + 1, None))
    with torch.no_grad():
        blk.gate_text.fill_(0.0)
        blk.gate_vis.fill_(1.0)
    only_vis = blk(q, text, None, vis, None)
    assert torch.allclose(only_vis, blk(q, text * 2, None, vis, None))
    assert not torch.allclose(only_vis, blk(q, text, None, vis * 7 + 1, None))


def test_gradients_reach_the_gates():
    torch.manual_seed(0)
    cfg = _cfg()
    s = ConnectorStream(cfg)
    out = s(torch.randn(2, 9, cfg.dim), torch.randn(2, 5, cfg.text_dim), None,
            torch.randn(2, 12, cfg.vision_dim), torch.randn(2, 12, 2), None)
    out.sum().backward()
    for b in s.blocks:
        assert b.gate_text.grad is not None and torch.isfinite(b.gate_text.grad).all()
        assert b.gate_vis.grad is not None and torch.isfinite(b.gate_vis.grad).all()
