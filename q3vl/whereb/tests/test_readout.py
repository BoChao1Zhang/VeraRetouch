"""``q3vl.whereb.readout`` -- the h_cond readout shared by EPR-018..023.

CPU only.  What these pin:

* the reply span each ``--cond-readout`` kind feeds, and the row index it reads;
* that the index is **recorded**, not searched -- :func:`verify_plan` catches a
  plan whose index does not carry the token it names;
* that the live ``h_where`` path is untouched (``hiddens.py`` has no readout
  code in it at all).
"""

from __future__ import annotations

import pytest
import torch

from q3vl.whereb.readout import (READOUT_KINDS, ReadoutBuilder, ReadoutSpec,
                                 ReplyPlan, SegTokenIds, build_reply,
                                 readout_hidden, readout_vector, verify_plan)


@pytest.fixture()
def tags(tokenizer) -> SegTokenIds:
    return SegTokenIds.from_tokenizer(tokenizer)


def _spans(tokenizer):
    from q3vl.what.context import encode_color_span
    from q3vl.whereb.context import encode_where_span

    return (encode_where_span(tokenizer, "the left half of the sky"),
            encode_color_span(tokenizer, "warmer and brighter"))


# --- spec -------------------------------------------------------------------

def test_spec_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unknown --readout"):
        ReadoutSpec(kind="whatever")


def test_spec_rejects_qtok_without_k_and_k_without_qtok():
    with pytest.raises(ValueError, match="readout-qtok"):
        ReadoutSpec(kind="qtok", qtok=0)
    with pytest.raises(ValueError, match="only read under --readout qtok"):
        ReadoutSpec(kind="seg_where", qtok=4)
    with pytest.raises(ValueError, match="only read under --readout seg_where"):
        ReadoutSpec(kind="im_end", nseg=2)


def test_spec_n_vectors():
    assert ReadoutSpec().n_vectors == 1
    assert ReadoutSpec(kind="where_span_pool").n_vectors == 1
    assert ReadoutSpec(kind="qtok", qtok=8).n_vectors == 8


def test_token_ids_resolve(tags, tokenizer):
    assert tags.has_seg
    d = tags.to_dict()
    assert d["where_close_id"] == tokenizer("</where>")["input_ids"][0]
    assert d["seg_where_id"] == tokenizer("<seg_where>")["input_ids"][0]


# --- the six kinds ----------------------------------------------------------

def test_seg_where_feeds_where_plus_color_plus_seg(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("seg_where"), where_ids=w, color_ids=c)
    assert p.token_ids == list(w) + list(c) + [tags.seg_where]
    assert (p.start, p.end) == (len(p.token_ids) - 1, len(p.token_ids))
    assert p.token_ids[p.start] == tags.seg_where
    verify_plan(p)


def test_where_span_pool_feeds_the_where_span_only(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("where_span_pool"), where_ids=w, color_ids=c)
    assert p.token_ids == list(w)          # the colour span is NOT fed
    assert (p.start, p.end, p.pool) == (0, len(w), True)
    verify_plan(p)


def test_where_close_and_color_close_positions(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("where_close"), where_ids=w, color_ids=c)
    assert p.token_ids == list(w)
    assert p.token_ids[p.start] == tags.where_close

    q = build_reply(tags, ReadoutSpec("color_close"), where_ids=w, color_ids=c)
    assert q.token_ids == list(w) + list(c)
    assert q.token_ids[q.start] == tags.color_close
    verify_plan(p)
    verify_plan(q)


def test_im_end_appends_the_v2seg_tail(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("im_end"), where_ids=w, color_ids=c)
    assert p.token_ids[-3:] == [tags.seg_where, tags.seg_color, tags.im_end]
    assert p.token_ids[p.start] == tags.im_end
    assert p.flags["template"] == "v2seg"
    verify_plan(p)


def test_im_end_on_a_pre_v2seg_base_uses_the_v1_tail(tags, tokenizer):
    """checkpoint-4976 has no seg tokens; --cond-readout im_end still works."""
    v1 = SegTokenIds(where_open=tags.where_open, where_close=tags.where_close,
                     color_open=tags.color_open, color_close=tags.color_close,
                     im_end=tags.im_end)
    assert not v1.has_seg
    w, c = _spans(tokenizer)
    p = build_reply(v1, ReadoutSpec("im_end"), where_ids=w, color_ids=c)
    assert p.token_ids == list(w) + list(c) + [tags.im_end]
    assert p.flags["template"] == "v1"


def test_seg_where_refuses_a_pre_v2seg_base(tags, tokenizer):
    v1 = SegTokenIds(where_open=tags.where_open, where_close=tags.where_close,
                     color_open=tags.color_open, color_close=tags.color_close,
                     im_end=tags.im_end)
    w, c = _spans(tokenizer)
    with pytest.raises(ValueError, match="needs a v2seg base"):
        build_reply(v1, ReadoutSpec("seg_where"), where_ids=w, color_ids=c)


def test_nseg_gt_1_is_a_placeholder_not_a_silent_single_token(tags, tokenizer):
    w, c = _spans(tokenizer)
    with pytest.raises(NotImplementedError, match="supervises exactly ONE"):
        build_reply(tags, ReadoutSpec("seg_where", nseg=2), where_ids=w, color_ids=c)


@pytest.mark.parametrize("k", [1, 4, 8])
def test_qtok_slice_sits_past_the_reply(tags, tokenizer, k):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("qtok", qtok=k), where_ids=w, color_ids=c)
    assert p.n_appended == k
    assert p.start == len(p.token_ids)
    assert p.expected_len == len(p.token_ids) + k
    verify_plan(p)


def test_every_kind_is_constructible(tags, tokenizer):
    w, c = _spans(tokenizer)
    for kind in READOUT_KINDS:
        spec = ReadoutSpec(kind, qtok=4 if kind == "qtok" else 0)
        verify_plan(build_reply(tags, spec, where_ids=w, color_ids=c))


# --- format failures --------------------------------------------------------

def test_missing_close_tag_is_counted_not_silent(tags, tokenizer):
    w, c = _spans(tokenizer)
    broken = list(w)[:-1]                      # generated span that never closed
    p = build_reply(tags, ReadoutSpec("where_close"), where_ids=broken, color_ids=c,
                    on_missing_tag="last")
    assert p.flags["missing_tag"] == "<where>"
    verify_plan(p)                             # flagged, so not an assertion error
    with pytest.raises(ValueError, match="format-failure"):
        build_reply(tags, ReadoutSpec("where_close"), where_ids=broken,
                    color_ids=c, on_missing_tag="raise")


def test_null_context_is_refused_by_the_span_readouts(tags):
    with pytest.raises(ValueError, match="non-empty"):
        build_reply(tags, ReadoutSpec("where_span_pool"), where_ids=[])


# --- the wiring assertion ---------------------------------------------------

def test_verify_plan_catches_a_misplaced_index(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("seg_where"), where_ids=w, color_ids=c)
    bad = ReplyPlan(token_ids=p.token_ids, start=0, end=1, pool=False,
                    kind=p.kind, expected_ids=p.expected_ids)
    with pytest.raises(AssertionError, match="not wired to the position"):
        verify_plan(bad)


def test_readout_hidden_shapes_and_pooling(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("seg_where"), where_ids=w, color_ids=c)
    h = torch.randn(len(p.token_ids), 2560)
    assert readout_hidden(h, p).shape == (1, 2560)
    assert torch.equal(readout_vector(h, p), h[-1])

    q = build_reply(tags, ReadoutSpec("where_span_pool"), where_ids=w)
    hw = torch.randn(len(w), 2560)
    assert torch.allclose(readout_hidden(hw, q)[0], hw.mean(0), atol=1e-6)

    k = build_reply(tags, ReadoutSpec("qtok", qtok=4), where_ids=w, color_ids=c)
    hk = torch.randn(k.expected_len, 2560)
    assert torch.equal(readout_hidden(hk, k), hk[-4:])


def test_readout_hidden_refuses_a_mismatched_encode(tags, tokenizer):
    w, c = _spans(tokenizer)
    p = build_reply(tags, ReadoutSpec("seg_where"), where_ids=w, color_ids=c)
    with pytest.raises(AssertionError, match="expects"):
        readout_hidden(torch.randn(len(p.token_ids) - 1, 2560), p)


def test_readout_vector_refuses_multi_row(tags, tokenizer):
    w, c = _spans(tokenizer)
    k = build_reply(tags, ReadoutSpec("qtok", qtok=4), where_ids=w, color_ids=c)
    with pytest.raises(ValueError, match="4 vectors"):
        readout_vector(torch.randn(k.expected_len, 2560), k)


# --- builder ----------------------------------------------------------------

def test_builder_teacher_and_generated_take_the_same_route(tokenizer):
    from q3vl.whereb.context import generated_context, gt_context

    ro = ReadoutBuilder(tokenizer, ReadoutSpec("seg_where"))
    gt = gt_context(tokenizer, "s1", "the left half")
    gen_ids = tokenizer("<where> the right half </where> <color> warmer </color>"
                        )["input_ids"]
    gen = generated_context("s1", gen_ids,
                            tokenizer("</where>")["input_ids"][0])

    p_gt = ro.plan_for(sample_id="s1", where_ctx=gt, color_text="warmer")
    p_gen = ro.plan_for(sample_id="s1", where_ctx=gen,
                        color_ids=tokenizer("<color> warmer </color>")["input_ids"])
    assert p_gt.source == "teacher" and p_gen.source == "generated"
    for p in (p_gt, p_gen):
        assert p.token_ids[p.start] == ro.tags.seg_where
    f = ro.facts()
    assert f["counts"]["source_teacher"] == 1
    assert f["counts"]["source_generated"] == 1
    assert f["readout"] == "seg_where"


def test_builder_says_which_flag_is_missing(tokenizer):
    from q3vl.whereb.context import gt_context

    ro = ReadoutBuilder(tokenizer, ReadoutSpec("seg_where"))
    with pytest.raises(ValueError, match="needs the <color> span"):
        ro.plan_for(sample_id="s1",
                    where_ctx=gt_context(tokenizer, "s1", "the left half"))


def test_where_close_needs_no_colour_span(tokenizer):
    from q3vl.whereb.context import gt_context

    ro = ReadoutBuilder(tokenizer, ReadoutSpec("where_close"))
    assert not ro.needs_color()
    p = ro.plan_for(sample_id="s1",
                    where_ctx=gt_context(tokenizer, "s1", "the left half"))
    assert p.token_ids[p.start] == ro.tags.where_close


# --- the live path is untouched --------------------------------------------

def test_hiddens_module_has_no_readout_logic():
    """The four live arms' ``h_where`` path must stay exactly what it was."""
    import inspect

    from q3vl.whereb import hiddens

    src = inspect.getsource(hiddens)
    for token in ("seg_where", "ReadoutSpec", "ReplyPlan", "readout_hidden"):
        assert token not in src, f"{token} leaked into hiddens.py"
