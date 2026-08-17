"""The ``<seg_color>`` readout plan: the index is recorded, never searched for."""

from __future__ import annotations

import inspect

import pytest
import torch

from q3vl.whatb.readout import (
    SEGMENT_HIDDEN_FINAL_NORM,
    SEGMENT_HIDDEN_LAYER,
    WHATB_READOUT_KINDS,
    SegTokenIds,
    WhatReadoutBuilder,
    WhatReadoutSpec,
    build_reply,
    readout_vector,
    verify_plan,
)


def _spans(tokenizer):
    from q3vl.whatb.colorspan import encode_color_span

    w = tokenizer("<where> the sky </where>")["input_ids"]
    c = encode_color_span(tokenizer, "cooler blues")
    return w, c


def test_hidden_contract_is_imported_not_redeclared():
    import q3vl.whatb.readout as ro
    from q3vl.whereb import contracts

    assert SEGMENT_HIDDEN_LAYER is contracts.SEGMENT_HIDDEN_LAYER == -1
    assert SEGMENT_HIDDEN_FINAL_NORM is contracts.SEGMENT_HIDDEN_FINAL_NORM is True
    src = inspect.getsource(ro)
    assert "SEGMENT_HIDDEN_LAYER = " not in src        # imported, never assigned


def test_seg_color_plan_points_at_the_last_token(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    plan = build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=c)
    assert plan.token_ids == w + c + [tags.seg_where, tags.seg_color]
    assert plan.start == len(plan.token_ids) - 1 and plan.end == len(plan.token_ids)
    assert plan.expected_ids == (tags.seg_color,)
    verify_plan(plan)                     # the runtime "is it wired" assertion


def test_verify_plan_catches_a_misrecorded_index(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    plan = build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=c)
    plan.start -= 1
    plan.end -= 1
    with pytest.raises(AssertionError, match="not wired"):
        verify_plan(plan)


def test_color_span_pool_covers_exactly_the_colour_span(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    plan = build_reply(tags, WhatReadoutSpec("color_span_pool"), where_ids=w,
                       color_ids=c)
    assert plan.token_ids == w + c
    assert (plan.start, plan.end, plan.pool) == (len(w), len(w) + len(c), True)
    verify_plan(plan)
    h = torch.arange(len(plan.token_ids) * 4, dtype=torch.float32).reshape(-1, 4)
    assert torch.allclose(readout_vector(h, plan), h[len(w):].mean(0))


def test_readout_vector_reads_the_seg_color_row(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    plan = build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=c)
    h = torch.randn(len(plan.token_ids), 8)
    assert torch.equal(readout_vector(h, plan), h[-1])


def test_wrong_reply_length_is_an_assertion(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    plan = build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=c)
    with pytest.raises(AssertionError, match="reply rows"):
        readout_vector(torch.randn(len(plan.token_ids) - 1, 8), plan)


def test_other_kinds_delegate_to_the_whereb_implementation(tokenizer):
    from q3vl.whereb.readout import ReadoutSpec as WSpec, build_reply as wbuild

    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, c = _spans(tokenizer)
    for kind in ("color_close", "im_end", "seg_where"):
        mine = build_reply(tags, WhatReadoutSpec(kind), where_ids=w, color_ids=c)
        theirs = wbuild(tags, WSpec(kind), where_ids=w, color_ids=c)
        assert mine.token_ids == theirs.token_ids
        assert (mine.start, mine.end) == (theirs.start, theirs.end)


def test_seg_color_needs_a_v2seg_base(tokenizer_no_seg):
    tags = SegTokenIds.from_tokenizer(tokenizer_no_seg)
    assert not tags.has_seg
    w, c = _spans(tokenizer_no_seg)
    with pytest.raises(ValueError, match="v2seg"):
        build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=c)


def test_empty_colour_span_is_refused(tokenizer):
    tags = SegTokenIds.from_tokenizer(tokenizer)
    w, _ = _spans(tokenizer)
    with pytest.raises(ValueError, match="non-empty <color>"):
        build_reply(tags, WhatReadoutSpec("seg_color"), where_ids=w, color_ids=[])


def test_builder_uses_the_own_colorspan_and_counts(tokenizer):
    ro = WhatReadoutBuilder(tokenizer)
    w, _ = _spans(tokenizer)
    plan = ro.plan_for(sample_id="s0", where_ids=w, color_text="cooler blues",
                       source="generated", control_tag="shuffle")
    assert plan.kind == "seg_color"
    facts = ro.facts()
    assert facts["counts"]["source_generated"] == 1
    assert facts["counts"]["control_shuffle"] == 1
    assert facts["readout"] == "seg_color" and facts["needs_v2seg"] is True
    assert facts["hidden_layer"] == -1 and facts["hidden_final_norm"] is True


def test_builder_never_calls_the_contaminated_helper(tokenizer):
    ro = WhatReadoutBuilder(tokenizer)
    src = inspect.getsource(type(ro).color_ids_from_text)
    assert "q3vl.what" not in src and "encode_color_span" in src


def test_all_six_kinds_are_offered():
    assert WHATB_READOUT_KINDS == ("seg_color", "color_span_pool", "color_close",
                                   "im_end", "seg_where", "qtok")
    with pytest.raises(ValueError, match="unknown --readout"):
        WhatReadoutSpec("nope")
