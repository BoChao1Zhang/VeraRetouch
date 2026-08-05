"""The antonym invariance control (main-agent ruling on the S5.5 open item).

Where-B's mask is a function of the SUBJECT, not of the colour direction, so
flipping only the colour words must leave it alone.  Three things to hold:

1. the substitution is correct and total -- it flips what it should, on all
   three axes, and is an involution;
2. **the subject phrase survives** -- the `<where>` segment is byte-identical,
   which is what makes this an invariance control rather than a second shuffle;
3. the whole thing is producible end to end and lands in the report as a
   negative-control column, never as a gate.
"""

from __future__ import annotations

import json

import pytest
import torch

from q3vl.whereb import config as C
from q3vl.whereb import metrics as M
from q3vl.whereb.antonyms import (
    ANTONYM_AXES,
    ANTONYM_MAP,
    ANTONYM_PAIRS,
    AXES,
    axes_touched,
    coverage,
    flip_text,
    flipped_terms,
    table_digest,
)
from q3vl.whereb.context import (
    ANTONYM,
    CONTEXT_MODES,
    INSTRUCTION_OVERRIDE_MODES,
    antonym_context,
)

INSTR = ("Please make the jellyfish noticeably darker and cooler, subdue its "
         "orange and brown coloring, and make it less saturated overall.")
WHERE = "subject: the jellyfish; edit scope: stays within the jellyfish"


# --- 1. the table itself ----------------------------------------------------

def test_table_covers_the_three_ruled_axes():
    assert AXES == ("luminance", "temperature", "saturation")
    for axis in AXES:
        assert ANTONYM_AXES[axis], f"{axis} has no pairs"
    # the ruling's named examples are present
    assert ("darker", "brighter") in ANTONYM_PAIRS
    assert ("warmer", "cooler") in ANTONYM_PAIRS
    assert ("saturate", "desaturate") in ANTONYM_PAIRS


def test_table_is_one_to_one_so_the_swap_is_well_defined():
    words = [w for pair in ANTONYM_PAIRS for w in pair]
    assert len(words) == len(set(words)), "a word appears in two pairs"
    assert len(ANTONYM_MAP) == 2 * len(ANTONYM_PAIRS)


def test_flip_is_an_involution():
    for text in (INSTR, "Brighten the WARM and vivid sky", "no colour words here"):
        assert flip_text(flip_text(text)) == text


def test_flip_is_simultaneous_not_sequential():
    """darker->brighter must not then be re-flipped back by the brighter rule."""
    assert flip_text("darker and brighter") == "brighter and darker"
    assert flip_text("warmer, cooler") == "cooler, warmer"


def test_longest_match_wins_so_desaturated_is_not_corrupted():
    assert flip_text("desaturated") == "saturated"
    assert flip_text("saturated") == "desaturated"
    assert flip_text("desaturation") == "saturation"


def test_word_boundaries_are_respected():
    assert flip_text("unsaturated") == "unsaturated"      # not a standalone word
    assert flip_text("warmly") == "warmly"


def test_case_is_preserved():
    assert flip_text("Darker") == "Brighter"
    assert flip_text("WARMER") == "COOLER"
    assert flip_text("darker") == "brighter"


def test_non_colour_text_is_byte_identical():
    t = "Please make the church and its surrounding trees stand out."
    assert flip_text(t) == t
    assert flipped_terms(t) == []


def test_digest_is_stable_and_pins_the_table():
    d = table_digest()
    assert len(d) == 64 and d == table_digest()


# --- 2. the control preserves the subject -----------------------------------

def test_antonym_context_flips_the_instruction_and_keeps_the_where_text(tokenizer):
    ctx = antonym_context(tokenizer, "sA", INSTR, WHERE)
    assert ctx.mode == ANTONYM
    assert ctx.instruction != INSTR
    assert "brighter" in ctx.instruction and "warmer" in ctx.instruction
    # THE point of this control: the subject phrase is untouched
    assert ctx.text == WHERE
    from q3vl.whereb.context import encode_where_span
    assert ctx.token_ids == encode_where_span(tokenizer, WHERE)


def test_antonym_context_records_what_it_flipped(tokenizer):
    ctx = antonym_context(tokenizer, "sA", INSTR, WHERE)
    d = ctx.control_detail
    assert d["instruction_changed"] is True
    assert d["where_text_unchanged"] is True
    assert d["n_flipped"] == len(d["flipped_terms"]) >= 3
    assert d["antonym_table_digest"] == table_digest()
    assert set(axes_touched(INSTR)) >= {"luminance", "temperature"}
    assert "control_detail" in ctx.to_dict()


def test_antonym_is_an_instruction_override_mode(tokenizer):
    assert ANTONYM in CONTEXT_MODES
    assert ANTONYM in INSTRUCTION_OVERRIDE_MODES
    assert antonym_context(tokenizer, "sA", INSTR, WHERE).instruction is not None


def test_an_instruction_with_no_colour_words_is_flagged_unchanged(tokenizer):
    plain = "Please adjust the church facade."
    ctx = antonym_context(tokenizer, "sA", plain, WHERE)
    assert ctx.control_detail["instruction_changed"] is False
    assert ctx.control_detail["n_flipped"] == 0
    assert ctx.instruction == plain           # still an override, just a no-op


def test_batch_builder_has_a_branch_for_the_antonym_mode():
    import inspect

    from q3vl.whereb.data import BatchBuilder

    assert "== ANTONYM" in inspect.getsource(BatchBuilder.context_for)


# --- 3. the metric ----------------------------------------------------------

def _rows(delta):
    out = []
    for i in range(11):
        out.append({"sample_id": f"s{i}", "context": "gt", "grid_hard_iou": 0.60})
        out.append({"sample_id": f"s{i}", "context": "antonym",
                    "grid_hard_iou": 0.60 - delta})
    return out


def test_invariance_passes_when_the_mask_does_not_move():
    r = M.antonym_invariance(_rows(0.01))
    assert r["n"] == 11
    assert r["median_abs_delta"] == pytest.approx(0.01)
    assert r["within_threshold"] is True
    assert r["threshold"] == C.ANTONYM_INVARIANCE_MAX == 0.05


def test_invariance_fails_when_the_field_reads_colour_words():
    r = M.antonym_invariance(_rows(0.30))
    assert r["median_abs_delta"] == pytest.approx(0.30)
    assert r["within_threshold"] is False


def test_invariance_is_paired_per_sample_not_a_difference_of_medians():
    """Two samples moving in opposite directions must NOT cancel."""
    rows = [
        {"sample_id": "a", "context": "gt", "grid_hard_iou": 0.60},
        {"sample_id": "a", "context": "antonym", "grid_hard_iou": 0.90},
        {"sample_id": "b", "context": "gt", "grid_hard_iou": 0.60},
        {"sample_id": "b", "context": "antonym", "grid_hard_iou": 0.30},
    ]
    r = M.antonym_invariance(rows)
    assert r["median_abs_delta"] == pytest.approx(0.30)   # not ~0
    assert abs(r["signed_delta"]) < 1e-9                  # the signed mean does cancel


def test_invariance_is_not_a_gate():
    keys = [k for k, _, _ in C.GATES]
    assert not any("antonym" in k for k in keys)
    assert "not a gate" in M.antonym_invariance(_rows(0.01))["note"]
    assert "antonym_invariance" in M.ATTRIBUTION_NOTE["known_blind_spots"]


def test_invariance_handles_a_missing_antonym_board():
    r = M.antonym_invariance([{"sample_id": "a", "context": "gt", "grid_hard_iou": 0.6}])
    assert r["n"] == 0 and r["median_abs_delta"] is None


# --- corpus reality check ---------------------------------------------------

def test_table_actually_reaches_this_corpus_style_of_instruction():
    """Measured on V_where local: 98.5% of instructions carry a flippable term
    across all three axes, while only 1% of <where> segments do -- which is
    exactly why the control flips the instruction and keeps the subject."""
    cov = coverage([INSTR,
                    "Please make the sky brighter and warmer with richer colour.",
                    "Please make the wall more vivid."])
    assert cov["flippable_rate"] == 1.0
    assert cov["per_axis"]["luminance"] >= 1
    assert cov["per_axis"]["temperature"] >= 1
    assert cov["per_axis"]["saturation"] >= 1
    assert coverage([WHERE])["n_flippable"] == 0     # the subject text is clean
