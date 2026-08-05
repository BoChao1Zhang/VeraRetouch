"""A5-B1: the three negative controls and the same-image paired difference.

The first-round failure this repeats is "an authority document asserts a check
the code cannot run": §17.2 announced a negative-control framework while two of
the three controls had no producer and the paired difference had no
implementation.  These tests assert the controls can actually be *produced*, not
that a constant equals its own literal.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import metrics as M
from q3vl.whereb.context import (
    CONTEXT_MODES,
    FIXED_PHRASE,
    FIXED_PHRASE_TEXT,
    GT,
    IRRELEVANT_WORDS,
    NEGATIVE_CONTROL_MODES,
    SHUFFLED,
    WhereContext,
    fixed_phrase_context,
    gt_context,
    irrelevant_words_context,
)

WHERE_A = "subject: the jellyfish; edit scope: stays within the jellyfish"


# --- the controls exist as producible modes ---------------------------------

def test_all_three_negative_controls_are_context_modes():
    assert NEGATIVE_CONTROL_MODES == (SHUFFLED, IRRELEVANT_WORDS, FIXED_PHRASE)
    assert C.INSTRUCTION_NEGATIVE_CONTROLS == NEGATIVE_CONTROL_MODES
    for m in NEGATIVE_CONTROL_MODES:
        assert m in CONTEXT_MODES, f"{m} is named but not a context mode"


def test_irrelevant_words_context_is_producible_and_carries_no_instruction(tokenizer):
    ctx = irrelevant_words_context(tokenizer, "sA", seed=0)
    assert ctx.mode == IRRELEVANT_WORDS
    assert ctx.token_ids and ctx.instruction == ctx.text
    gt = gt_context(tokenizer, "sA", WHERE_A)
    assert ctx.token_ids != gt.token_ids
    # the text really is unrelated: none of the instruction's content words
    assert not (set(ctx.text.split()) & {"jellyfish", "subject", "edit", "scope"})


def test_irrelevant_words_is_deterministic_per_sample_and_varies_across_samples(tokenizer):
    a1 = irrelevant_words_context(tokenizer, "sA", seed=0).text
    a2 = irrelevant_words_context(tokenizer, "sA", seed=0).text
    b = irrelevant_words_context(tokenizer, "sB", seed=0).text
    c = irrelevant_words_context(tokenizer, "sA", seed=1).text
    assert a1 == a2, "same sample + same seed must reproduce"
    assert a1 != b, "different samples must get different draws"
    assert a1 != c, "the seed must matter"


def test_fixed_phrase_is_the_red_lines_own_worked_example(tokenizer):
    ctx = fixed_phrase_context(tokenizer, "sA")
    assert ctx.mode == FIXED_PHRASE
    assert ctx.text == FIXED_PHRASE_TEXT == "the main subject"
    # constant across every sample -- that is the whole point of this control
    assert ctx.token_ids == fixed_phrase_context(tokenizer, "sZZZ").token_ids
    assert ctx.instruction == FIXED_PHRASE_TEXT


def test_the_two_new_controls_override_the_instruction_too(tokenizer):
    """Same rule as the shuffled control (D-B15): a control that leaves the real
    instruction in the prompt leaves the right answer reachable."""
    for ctx in (irrelevant_words_context(tokenizer, "sA"),
                fixed_phrase_context(tokenizer, "sA")):
        assert ctx.instruction is not None
    assert gt_context(tokenizer, "sA", WHERE_A).instruction is None


def test_a_non_control_mode_still_cannot_override_the_instruction():
    with pytest.raises(ValueError, match="may not override"):
        WhereContext(mode=GT, token_ids=[1], instruction="sneaky")


def test_batch_builder_has_a_branch_for_every_context_mode():
    """The failure mode being prevented: a mode named in CONTEXT_MODES that
    falls through to `raise ValueError(unknown context mode)`."""
    from q3vl.whereb.data import BatchBuilder

    src = inspect.getsource(BatchBuilder.context_for)
    for m in CONTEXT_MODES:
        const = {"gt": "GT", "generated": "GENERATED", "null": "NULL",
                 "shuffled": "SHUFFLED", "irrelevant_words": "IRRELEVANT_WORDS",
                 "fixed_phrase": "FIXED_PHRASE"}[m]
        assert f"== {const}" in src, f"context_for has no branch for {m}"


# --- the same-image paired difference ---------------------------------------

def _rows(n_groups=6, per=2, margin=0.30):
    rows = []
    for g in range(n_groups):
        for j in range(per):
            rows.append({"sample_id": f"s{g}_{j}", "source_image_id": f"img{g}",
                         "self_iou": 0.60 + 0.02 * j,
                         "cross_iou": 0.60 + 0.02 * j - margin})
    return rows


def test_instruction_paired_delta_detects_instruction_following():
    d = M.instruction_paired_delta(_rows())
    assert d["n_groups"] == 6 and d["n_samples"] == 12
    assert d["delta"] == pytest.approx(0.30)
    assert d["p_value"] <= 0.05
    assert d["test"] == "sign_flip_permutation"


def test_instruction_paired_delta_reports_nothing_when_the_field_ignores_the_instruction():
    d = M.instruction_paired_delta(_rows(margin=0.0))
    assert abs(d["delta"]) < 1e-9
    assert d["p_value"] > 0.05


def test_instruction_paired_delta_skips_groups_with_no_partner():
    rows = _rows(n_groups=3, per=2) + [
        {"sample_id": "lonely", "source_image_id": "solo",
         "self_iou": 0.9, "cross_iou": 0.1}]
    d = M.instruction_paired_delta(rows)
    assert d["n_groups"] == 3 and d["n_samples"] == 6


# --- N23: the p-value is a permutation test now -----------------------------

def test_p_value_is_never_exactly_zero():
    """CI inversion reported p = 0.0 for an all-positive difference set; a
    permutation test reports the honest 1/(n_perm+1) floor."""
    a = [0.5] * 40
    b = [0.1] * 40
    d = M.paired_delta(a, b, n_perm=1000, seed=0)
    assert d["p_value"] > 0.0
    assert d["p_value"] == pytest.approx(1 / 1001, rel=1e-6)


def test_permutation_p_is_symmetric_under_sign():
    a = [0.4, 0.5, 0.6, 0.55, 0.45, 0.52]
    b = [0.1, 0.2, 0.3, 0.25, 0.15, 0.22]
    fwd = M.paired_delta(a, b, n_perm=2000, seed=0)
    rev = M.paired_delta(b, a, n_perm=2000, seed=0)
    assert fwd["delta"] == pytest.approx(-rev["delta"])
    assert fwd["p_value"] == pytest.approx(rev["p_value"])


def test_permutation_test_is_declared_in_the_output():
    d = M.paired_delta([1.0, 2.0], [0.5, 1.0])
    assert d["test"] == "sign_flip_permutation"
    assert "n_perm" in d


# --- N25: the blind spots ride along with every board -----------------------

def test_attribution_note_carries_the_blind_spots():
    bs = M.ATTRIBUTION_NOTE["known_blind_spots"]
    assert "boundary_f1_alone" in bs and "instruction_conditionality" in bs
    assert "0.0357" in bs["boundary_f1_alone"], "the measured counter-example is missing"
    assert "fixed_phrase" in bs["instruction_conditionality"]


def test_attribution_section_renders_the_blind_spots():
    sec = M.attribution_section({"arm": "W01", "main_context": "generated"})
    assert "盲区" in sec or "blind spot" in sec.lower()
    assert "fixed_phrase" in sec
