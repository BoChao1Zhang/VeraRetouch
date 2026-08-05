"""Protocol 5.4: the four context flows, the 50/50 mix, and the no-fallback rule."""

from __future__ import annotations

import inspect

import pytest

from q3vl.whereb.config import WHERE_CONTEXT_MAX_TOKENS
from q3vl.whereb.context import (
    CONTEXT_MODES,
    GENERATED,
    GT,
    NULL,
    SHUFFLED,
    BalancedContextSampler,
    FormatStats,
    ShuffleIndex,
    WhereContext,
    encode_where_span,
    generated_context,
    gt_context,
    null_context,
    shuffled_context,
)

WHERE_A = "subject: the jellyfish; edit scope: stays within the jellyfish"
WHERE_B = "subject: the church; edit scope: a vertical band"


def _close_id(tok) -> int:
    return tok("</where>")["input_ids"][0]


# --- GT ---------------------------------------------------------------------

def test_gt_context_encodes_the_tagged_span(tokenizer):
    ctx = gt_context(tokenizer, "sA", WHERE_A)
    assert ctx.mode == GT and ctx.provenance == "sA"
    assert ctx.token_ids == encode_where_span(tokenizer, WHERE_A)
    assert ctx.token_ids[0] == tokenizer("<where>")["input_ids"][0]
    assert ctx.token_ids[-1] == _close_id(tokenizer)
    assert not ctx.format_failure and not ctx.truncated


def test_gt_context_refuses_to_truncate(tokenizer):
    long_text = " ".join(["word"] * (WHERE_CONTEXT_MAX_TOKENS + 5))
    with pytest.raises(ValueError, match="boundary"):
        gt_context(tokenizer, "sA", long_text)


# --- generated --------------------------------------------------------------

def test_generated_context_cuts_at_the_first_close_tag(tokenizer):
    span = encode_where_span(tokenizer, WHERE_A)
    ctx = generated_context("sA", span + [7, 8, 9], _close_id(tokenizer))
    assert ctx.token_ids == span
    assert ctx.stop_reason == "closed"
    assert not ctx.format_failure


def test_unclosed_generation_is_cut_at_the_fixed_boundary_and_flagged(tokenizer):
    runaway = [42] * (WHERE_CONTEXT_MAX_TOKENS + 50)
    ctx = generated_context("sA", runaway, _close_id(tokenizer))
    assert len(ctx.token_ids) == WHERE_CONTEXT_MAX_TOKENS
    assert ctx.format_failure and ctx.truncated
    assert ctx.stop_reason == "no_close_tag"
    assert ctx.n_generated_tokens == len(runaway)


def test_generated_context_cannot_fall_back_to_gt():
    """Structural: the function never receives the GT text (protocol 5.4)."""
    params = set(inspect.signature(generated_context).parameters)
    assert not (params & {"where_text", "gt_text", "teacher", "sample", "fallback"})
    assert params == {"sample_id", "generated_ids", "close_id", "text",
                      "max_tokens", "eos_id"}


def test_unclosed_generation_is_not_the_gt_span(tokenizer):
    gt = gt_context(tokenizer, "sA", WHERE_A)
    ctx = generated_context("sA", [42] * 200, _close_id(tokenizer))
    assert ctx.token_ids != gt.token_ids


def test_empty_generation_is_a_format_failure(tokenizer):
    ctx = generated_context("sA", [], _close_id(tokenizer))
    assert ctx.token_ids == [] and ctx.format_failure
    assert ctx.stop_reason == "empty"


def test_eos_truncates_before_the_close_scan(tokenizer):
    span = encode_where_span(tokenizer, WHERE_A)
    ctx = generated_context("sA", [5, 6, tokenizer.eos_token_id] + span,
                            _close_id(tokenizer), eos_id=tokenizer.eos_token_id)
    assert ctx.token_ids == [5, 6]
    assert ctx.format_failure and ctx.stop_reason == "no_close_tag"


def test_close_tag_beyond_the_boundary_is_still_a_failure(tokenizer):
    tail = [1] * (WHERE_CONTEXT_MAX_TOKENS + 3) + [_close_id(tokenizer)]
    ctx = generated_context("sA", tail, _close_id(tokenizer))
    assert len(ctx.token_ids) == WHERE_CONTEXT_MAX_TOKENS
    assert ctx.format_failure and ctx.stop_reason == "closed_over_boundary"


# --- null -------------------------------------------------------------------

def test_null_context_is_empty():
    ctx = null_context()
    assert ctx.mode == NULL and ctx.n_tokens == 0 and ctx.provenance == ""
    with pytest.raises(ValueError):
        WhereContext(mode=NULL, token_ids=[1])


# --- shuffled ---------------------------------------------------------------

INSTR_A = "Please make the jellyfish darker and cooler."
INSTR_B = "Please warm the church and the hillside behind it."


def test_shuffled_context_carries_the_partner_text(tokenizer):
    ctx = shuffled_context(tokenizer, "sB", WHERE_B, INSTR_B)
    assert ctx.mode == SHUFFLED and ctx.provenance == "sB"
    assert ctx.token_ids == encode_where_span(tokenizer, WHERE_B)
    assert ctx.token_ids != encode_where_span(tokenizer, WHERE_A)


def test_shuffled_context_also_swaps_the_instruction(tokenizer):
    """Protocol 5.4 swaps instruction *and* where context (review blocker B3)."""
    ctx = shuffled_context(tokenizer, "sB", WHERE_B, INSTR_B)
    assert ctx.instruction == INSTR_B
    assert ctx.to_dict()["instruction_swapped"] is True
    # the pair comes from one partner: same provenance for both halves
    assert ctx.provenance == "sB" and ctx.text == WHERE_B


def test_only_the_negative_controls_may_override_the_instruction(tokenizer):
    """A-5 widened this from `shuffled` alone to the three negative controls;
    every non-control mode must still be refused."""
    assert gt_context(tokenizer, "sA", WHERE_A).instruction is None
    assert null_context().instruction is None
    assert generated_context("sA", [1, 2], _close_id(tokenizer)).instruction is None
    for mode in (GT, GENERATED):
        with pytest.raises(ValueError, match="may not override"):
            WhereContext(mode=mode, token_ids=[1], instruction="sneaky")
    with pytest.raises(ValueError, match="may not override"):
        WhereContext(mode=NULL, token_ids=[], instruction="sneaky")


def test_shuffled_context_refuses_a_partner_without_an_instruction(tokenizer):
    with pytest.raises(ValueError, match="instruction"):
        shuffled_context(tokenizer, "sB", WHERE_B, "")
    with pytest.raises(ValueError, match="instruction"):
        shuffled_context(tokenizer, "sB", WHERE_B, "   ")


def _records(n_per_image: dict[str, int]) -> list[dict]:
    rows = []
    for img, n in n_per_image.items():
        for i in range(n):
            rows.append({"sample_id": f"{img}_{i}", "source_image_id": img,
                         "render_mode": "local", "where": f"text {img} {i}",
                         "instruction": f"instruction {img} {i}"})
    return rows


def test_shuffle_index_is_a_derangement_inside_the_group():
    idx = ShuffleIndex(_records({"a": 4, "b": 3}), seed=0)
    for sid, partner in idx.partner.items():
        assert partner != sid
        assert idx.by_id[partner]["source_image_id"] == idx.by_id[sid]["source_image_id"]
    assert len(idx.partner) == 7


def test_shuffle_index_leaves_singletons_uncovered_rather_than_crossing_images():
    idx = ShuffleIndex(_records({"a": 1, "b": 3}), seed=0)
    assert idx.partner_of("a_0") is None
    assert idx.coverage()["n_with_partner"] == 3
    assert abs(idx.coverage()["coverage"] - 0.75) < 1e-9


def test_shuffle_index_separates_global_from_local():
    rows = _records({"a": 2})
    rows.append({"sample_id": "a_g", "source_image_id": "a",
                 "render_mode": "global", "where": "global adjustment",
                 "instruction": "Please warm the whole frame."})
    idx = ShuffleIndex(rows, seed=0)
    assert idx.partner_of("a_g") is None                 # its own group of one
    assert idx.partner_of("a_0") == "a_1"


def test_shuffle_index_refuses_a_record_missing_either_half():
    """Both halves of the swap must be present (review blocker B3)."""
    base = {"sample_id": "x", "source_image_id": "i", "render_mode": "local"}
    with pytest.raises(ValueError, match="instruction"):
        ShuffleIndex([{**base, "where": "w"}], seed=0)
    with pytest.raises(ValueError, match="where"):
        ShuffleIndex([{**base, "instruction": "do a thing"}], seed=0)
    ShuffleIndex([{**base, "where": "w", "instruction": "do a thing"}], seed=0)


def test_shuffle_index_is_deterministic_in_the_seed():
    a = ShuffleIndex(_records({"x": 5}), seed=7).partner
    b = ShuffleIndex(_records({"x": 5}), seed=7).partner
    c = ShuffleIndex(_records({"x": 5}), seed=8).partner
    assert a == b
    assert a != c or len(a) < 3


# --- the 50/50 mix ----------------------------------------------------------

@pytest.mark.parametrize("micro", [2, 4, 8])
def test_every_micro_batch_is_exactly_half_teacher_half_generated(micro):
    s = BalancedContextSampler(100, micro, seed=0)
    batches = list(s)
    assert batches
    for b in batches:
        modes = [m for _, m in b]
        assert len(b) == micro
        assert modes.count(GT) == modes.count(GENERATED) == micro // 2


def test_a_sample_is_seen_once_per_epoch_and_the_pools_are_disjoint():
    s = BalancedContextSampler(64, 4, seed=1)
    seen = [i for b in s for i, _ in b]
    assert len(seen) == len(set(seen))
    assert not set(s.teacher_pool) & set(s.generated_pool)
    assert len(s.teacher_pool) + len(s.generated_pool) == 64


def test_a_sample_always_gets_the_same_context_within_an_epoch():
    s = BalancedContextSampler(40, 4, seed=2)
    mode_of: dict[int, str] = {}
    for b in s:
        for i, m in b:
            assert mode_of.setdefault(i, m) == m


def test_odd_micro_batch_is_refused():
    with pytest.raises(ValueError, match="50/50"):
        BalancedContextSampler(10, 3, seed=0)


def test_format_stats_counts_failures(tokenizer):
    st = FormatStats()
    st.update(gt_context(tokenizer, "a", WHERE_A))
    st.update(generated_context("a", [1] * 500, _close_id(tokenizer)))
    st.update(generated_context("a", [], _close_id(tokenizer)))
    d = st.to_dict()
    assert d["n"] == 3
    assert abs(d["format_failure_rate"] - 2 / 3) < 1e-9
    assert d["stop_reasons"]["no_close_tag"] == 1
