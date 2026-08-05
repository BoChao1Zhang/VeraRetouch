"""Protocol 7.1 / 7.3 -- attention parity, the colour stack, and the slot backend.

The two structural claims checked here are the ones protocol 14.8 and 14.12 ask
for evidence of, expressed as tests rather than as prose:

* ``Q_color`` reads ``H_color`` and nothing else, and at initialisation the
  zero-init cross gate makes ``M_color`` *independent of the language* -- which is
  what "zero-init gated residual" means operationally;
* head ``i`` of the 48 decoder heads depends on slot ``i`` alone, so "48
  independent decoder heads" is true of the batched implementation.
"""

from __future__ import annotations

import pytest
import torch

from q3vl.what.attention import AttentionPool, MultiheadAttention
from q3vl.what.backend import ModLN, SlotBackend, SlotHeads, n_head_params
from q3vl.what.color import ColorStack
from q3vl.what.config import BackendConfig, ColorConnectorConfig, N_SLOTS

CC = ColorConnectorConfig(dim=32, n_heads=4, ffn=64, n_blocks=2, z_style_dim=16,
                          z_style_hidden=16, text_dim=48)
BC = BackendConfig(dim=32, n_heads=4, ffn=64, z_style_dim=16, z_style_head_proj=8)


# --- attention ---------------------------------------------------------------

def test_attention_matches_the_where_b_implementation_bit_for_bit():
    from q3vl.whereb.connector import MultiheadAttention as Ref

    torch.manual_seed(0)
    a = MultiheadAttention(32, 4)
    b = Ref(32, 4)
    b.load_state_dict(a.state_dict())
    q, kv = torch.randn(3, 5, 32), torch.randn(3, 7, 32)
    mask = torch.ones(3, 7, dtype=torch.bool)
    mask[1, 3:] = False
    assert torch.equal(a(q, kv, mask), b(q, kv, mask))


def test_a_row_with_no_valid_key_returns_exactly_zero():
    torch.manual_seed(1)
    a = MultiheadAttention(32, 4)
    kv = torch.randn(2, 6, 32)
    mask = torch.ones(2, 6, dtype=torch.bool)
    mask[0] = False
    out = a(torch.randn(2, 4, 32), kv, mask)
    assert torch.equal(out[0], torch.zeros_like(out[0]))
    assert torch.isfinite(out).all()
    assert float(out[1].abs().sum()) > 0.0


def test_attention_pool_shape():
    p = AttentionPool(32, 4)
    assert p(torch.randn(3, 9, 32)).shape == (3, 32)


# --- colour stack ------------------------------------------------------------

def test_color_stack_shapes():
    s = ColorStack(CC)
    m, z = s(torch.randn(2, 11, CC.text_dim))
    assert m.shape == (2, CC.n_queries, CC.dim)
    assert z.shape == (2, CC.z_style_dim)


def test_zero_init_gate_makes_m_color_language_independent_at_step_0():
    s = ColorStack(CC).eval()
    a = s(torch.randn(2, 11, CC.text_dim))[0]
    b = s(torch.randn(2, 11, CC.text_dim) * 50)[0]
    assert torch.allclose(a, b, atol=1e-5)
    assert all(float(g) == 0.0 for g in s.connector.gate_values()["gate_text"])
    # once a gate is opened, the language matters
    with torch.no_grad():
        s.connector.blocks[0].gate_text.fill_(1.0)
    c = s(torch.randn(2, 11, CC.text_dim) * 50)[0]
    assert not torch.allclose(a, c, atol=1e-4)


def test_color_stack_signature_cannot_take_a_where_input():
    import inspect

    params = set(inspect.signature(ColorStack.forward).parameters)
    assert params == {"self", "h_color", "h_color_mask"}


def test_color_stack_handles_an_empty_language_mask():
    s = ColorStack(CC)
    mask = torch.zeros(2, 11, dtype=torch.bool)
    m, z = s(torch.randn(2, 11, CC.text_dim), mask)
    assert torch.isfinite(m).all() and torch.isfinite(z).all()


# --- backend -----------------------------------------------------------------

def test_modln_is_a_plain_layernorm_at_init():
    m = ModLN(32, 16)
    x, s = torch.randn(2, 5, 32), torch.randn(2, 16)
    assert torch.allclose(m(x, s), torch.nn.functional.layer_norm(x, (32,)), atol=1e-5)


def test_backend_stages_and_gates():
    b = SlotBackend(BC)
    style = torch.randn(2, BC.z_style_dim)
    m_color, wc = torch.randn(2, 16, BC.dim), torch.randn(2, 3, BC.dim)
    h = b.run_seed(style, m_color, None, wc, None)
    assert h.shape == (2, N_SLOTS, BC.dim)
    out = b.run_refine(h, torch.randn(2, N_SLOTS, BC.v_dim), style, m_color, None,
                       wc, None)
    assert out.shape == (2, N_SLOTS, BC.dim)
    g = b.gate_values()
    assert len(g["gate_color"]) == BC.seed_blocks + BC.refine_blocks
    assert len(g["gate_v"]) == BC.refine_blocks
    assert all(v == 0.0 for vals in g.values() for v in vals)


def test_refinement_block_refuses_to_run_without_v():
    b = SlotBackend(BC)
    style = torch.randn(1, BC.z_style_dim)
    with pytest.raises(ValueError):
        b.refine_blocks[0](torch.randn(1, N_SLOTS, BC.dim), style,
                           torch.randn(1, 4, BC.dim), None,
                           torch.randn(1, 2, BC.dim), None, None)


def test_the_48_decoder_heads_are_independent():
    """Head ``i``'s output must depend on slot ``i`` and on nothing else."""
    heads = SlotHeads(BC, n_out=23, bottleneck=8)
    with torch.no_grad():                       # the second layer is zero-init
        heads.w2.normal_(std=0.1)
        heads.b2.normal_(std=0.1)
    h = torch.randn(1, N_SLOTS, BC.dim, requires_grad=True)
    style = torch.randn(1, BC.z_style_dim)
    z_prim, _ = heads(h, style)
    grad = torch.autograd.grad(z_prim[0, 7].sum(), h, retain_graph=True)[0]
    assert float(grad[0, 7].abs().sum()) > 0.0
    others = [i for i in range(N_SLOTS) if i != 7]
    assert float(grad[0, others].abs().sum()) == 0.0


def test_heads_are_zero_at_init_and_the_count_is_the_analytic_one():
    heads = SlotHeads(BC, n_out=23, bottleneck=8)
    z_prim, z_glob = heads(torch.randn(2, N_SLOTS, BC.dim),
                           torch.randn(2, BC.z_style_dim))
    assert float(z_prim.abs().max()) == 0.0
    assert float(z_glob.abs().max()) == 0.0
    per_slot = n_head_params(BC.dim + BC.z_style_head_proj, 8, 23, N_SLOTS)
    got = sum(p.numel() for n, p in heads.named_parameters()
              if n in ("w1", "b1", "w2", "b2"))
    assert got == per_slot
