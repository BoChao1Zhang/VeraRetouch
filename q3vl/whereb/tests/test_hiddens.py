"""The H_where forward contract, asserted on the real Qwen3-VL class.

These run on a toy-sized but *genuine* ``Qwen3VLForConditionalGeneration``, so
they check transformers' actual behaviour rather than a stand-in's.  The two
facts they pin are the ones a silent mismatch would ruin:

* ``hidden_states[-1]`` is **pre** final-norm; the lm_head sees ``norm(...)``;
* attention is causal, so the ``<where>`` hidden states cannot see ``<color>`` --
  which is the numerical half of the protocol 14.8 proof.
"""

from __future__ import annotations

import torch

from q3vl.whereb.hiddens import LastLayerHook, resolve_language_model, resolve_visual


def test_module_paths_resolve_on_the_real_class(toy_qwen3vl):
    vis = resolve_visual(toy_qwen3vl)
    lm = resolve_language_model(toy_qwen3vl)
    assert hasattr(vis, "blocks") and len(vis.blocks) == 2
    assert hasattr(lm, "layers") and hasattr(lm, "norm")
    assert len(lm.layers) == 2


def test_hidden_states_last_entry_is_pre_final_norm(toy_qwen3vl):
    """VERIFIED FACT the H_where contract depends on (transformers 4.57.1)."""
    ids = torch.randint(0, 100, (2, 10))
    with torch.no_grad():
        out = toy_qwen3vl(input_ids=ids, attention_mask=torch.ones_like(ids),
                          output_hidden_states=True)
    hs = out.hidden_states
    assert len(hs) == 3                                   # num_hidden_layers + 1
    lm = resolve_language_model(toy_qwen3vl)
    with torch.no_grad():
        assert torch.allclose(toy_qwen3vl.lm_head(lm.norm(hs[-1])), out.logits, atol=1e-5)
        assert not torch.allclose(toy_qwen3vl.lm_head(hs[-1]), out.logits, atol=1e-3)


def test_last_layer_hook_equals_output_hidden_states(toy_qwen3vl):
    ids = torch.randint(0, 100, (2, 9))
    lm = resolve_language_model(toy_qwen3vl)
    hook = LastLayerHook(lm, -1)
    with torch.no_grad(), hook.attached():
        out = toy_qwen3vl(input_ids=ids, attention_mask=torch.ones_like(ids),
                          output_hidden_states=True)
    assert hook.captured is not None
    assert torch.equal(hook.captured, out.hidden_states[-1])


def test_hook_detaches_after_the_context_manager(toy_qwen3vl):
    lm = resolve_language_model(toy_qwen3vl)
    hook = LastLayerHook(lm, -1)
    with hook.attached():
        pass
    assert hook._handle is None
    n_before = len(lm.layers[-1]._forward_hooks)
    with hook.attached():
        assert len(lm.layers[-1]._forward_hooks) == n_before + 1
    assert len(lm.layers[-1]._forward_hooks) == n_before


def test_prefix_hidden_states_are_bit_identical_under_a_suffix_change(toy_qwen3vl):
    """Causal attention: <where> hidden states cannot read <color> (14.8)."""
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (2, 14))
    tail = ids.clone()
    tail[:, 8:] = torch.randint(0, 100, (2, 6))
    with torch.no_grad():
        a = toy_qwen3vl(input_ids=ids, attention_mask=torch.ones_like(ids),
                        output_hidden_states=True).hidden_states[-1]
        b = toy_qwen3vl(input_ids=tail, attention_mask=torch.ones_like(tail),
                        output_hidden_states=True).hidden_states[-1]
    assert torch.equal(a[:, :8], b[:, :8])
    assert not torch.equal(a[:, 8:], b[:, 8:])


def test_a_longer_suffix_does_not_change_the_prefix_either(toy_qwen3vl):
    """Teacher context is extracted from prompt+where only; adding <color> after
    it must not move the numbers, or teacher and generated would not compare."""
    torch.manual_seed(1)
    short = torch.randint(0, 100, (1, 10))
    long = torch.cat([short, torch.randint(0, 100, (1, 7))], dim=1)
    with torch.no_grad():
        a = toy_qwen3vl(input_ids=short, attention_mask=torch.ones_like(short),
                        output_hidden_states=True).hidden_states[-1]
        b = toy_qwen3vl(input_ids=long, attention_mask=torch.ones_like(long),
                        output_hidden_states=True).hidden_states[-1]
    assert torch.equal(a, b[:, :10])


def test_right_padding_does_not_disturb_the_real_tokens(toy_qwen3vl):
    """The batched encode path right-pads; padded columns must not leak in."""
    torch.manual_seed(2)
    a_ids = torch.randint(0, 100, (1, 9))
    b_ids = torch.randint(0, 100, (1, 6))
    n = 9
    batch = torch.zeros(2, n, dtype=torch.long)
    attn = torch.zeros(2, n, dtype=torch.long)
    batch[0], attn[0] = a_ids[0], 1
    batch[1, :6], attn[1, :6] = b_ids[0], 1
    with torch.no_grad():
        joint = toy_qwen3vl(input_ids=batch, attention_mask=attn,
                            output_hidden_states=True).hidden_states[-1]
        solo = toy_qwen3vl(input_ids=b_ids, attention_mask=torch.ones_like(b_ids),
                           output_hidden_states=True).hidden_states[-1]
    assert torch.allclose(joint[1, :6], solo[0], atol=1e-4)
