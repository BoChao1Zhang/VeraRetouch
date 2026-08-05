"""Amendment A-4: the generation artefact carries both reasoning segments.

Three things these tests exist to hold:

1. the ``<color>`` segment is extracted by the **same** rule as ``<where>`` --
   first closing tag, else a fixed boundary plus a recorded format failure, and
   never a fall back to GT;
2. schema ``/2`` is **additive**: a v1 consumer reading a v2 record sees exactly
   what it saw before, because every v1 field kept its name *and* its meaning
   (the ``<where>`` segment);
3. ``forced_color`` mode -- ``<color>`` as a forced prefix, no ``<where>`` in
   front of it -- is the strict no-where control, not a where format failure.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from q3vl.whereb.config import (
    COLOR_CONTEXT_MAX_TOKENS,
    GENCTX_MODES,
    GEN_MAX_NEW_TOKENS,
    SCHEMA_GENCTX,
    SCHEMA_GENCTX_V1,
    SCHEMA_GENCTX_V1_FIELDS,
    WHERE_CONTEXT_MAX_TOKENS,
)
from q3vl.whereb.context import extract_segment, generated_context
from q3vl.whereb.gencontext import (
    SegmentIds,
    build_record,
    publish_generated,
    summarise_records,
)
from q3vl.whereb.stores import GenContextStore

from .conftest import FakeTokenizer


@pytest.fixture()
def tags(tokenizer):
    return SegmentIds(tokenizer)


def _sample(sid="sft_a"):
    return SimpleNamespace(sample_id=sid, meta={"build": "l1", "render_mode": "local",
                                                "winner_confidence": "normal"})


def _gen(tokenizer, where="a b c", color="x y z", close_color=True, eos=False):
    ids = tokenizer(f"<where>{where}</where>")["input_ids"]
    ids += tokenizer(f"<color>{color}" + ("</color>" if close_color else ""))["input_ids"]
    if eos:
        ids.append(tokenizer.eos_token_id)
    return ids


# --- the shared extractor ---------------------------------------------------

def test_extract_segment_reproduces_the_where_extractor(tokenizer):
    """The refactor must not move a single token of the where path."""
    close = tokenizer("</where>")["input_ids"][0]
    for ids in ([], [7] * 300, _gen(tokenizer), _gen(tokenizer, eos=True)):
        seg = extract_segment(ids, close, WHERE_CONTEXT_MAX_TOKENS,
                              eos_id=tokenizer.eos_token_id)
        ctx = generated_context("s", ids, close, eos_id=tokenizer.eos_token_id)
        assert seg.token_ids == ctx.token_ids
        assert seg.format_failure == ctx.format_failure
        assert seg.truncated == ctx.truncated
        assert seg.stop_reason == ctx.stop_reason


def test_color_segment_is_found_after_the_where_segment(tokenizer, tags):
    ids = _gen(tokenizer, where="a b", color="x y")
    w = extract_segment(ids, tags.where_close, WHERE_CONTEXT_MAX_TOKENS)
    c = extract_segment(ids, tags.color_close, COLOR_CONTEXT_MAX_TOKENS,
                        open_id=tags.color_open, start=w.end)
    assert c.token_ids[0] == tags.color_open
    assert c.token_ids[-1] == tags.color_close
    assert c.stop_reason == "closed" and not c.format_failure
    assert c.start == w.end


def test_missing_color_close_tag_is_cut_at_the_fixed_boundary(tokenizer, tags):
    ids = tokenizer("<where>a</where>")["input_ids"] + [tags.color_open] + [99] * 900
    w = extract_segment(ids, tags.where_close, WHERE_CONTEXT_MAX_TOKENS)
    c = extract_segment(ids, tags.color_close, COLOR_CONTEXT_MAX_TOKENS,
                        open_id=tags.color_open, start=w.end)
    assert len(c.token_ids) == COLOR_CONTEXT_MAX_TOKENS
    assert c.format_failure and c.truncated and c.stop_reason == "no_close_tag"


def test_missing_color_open_tag_is_its_own_failure_mode(tokenizer, tags):
    ids = tokenizer("<where>a</where>")["input_ids"]
    w = extract_segment(ids, tags.where_close, WHERE_CONTEXT_MAX_TOKENS)
    c = extract_segment(ids, tags.color_close, COLOR_CONTEXT_MAX_TOKENS,
                        open_id=tags.color_open, start=w.end)
    assert c.token_ids == [] and c.format_failure
    assert c.stop_reason == "no_open_tag"


def test_extract_segment_cannot_reach_any_gt_text():
    import inspect

    params = set(inspect.signature(extract_segment).parameters)
    assert params == {"generated_ids", "close_id", "max_tokens", "open_id",
                      "start", "eos_id"}
    assert not (params & {"text", "gt", "where_text", "color_text", "sample"})


# --- the record -------------------------------------------------------------

def test_two_segment_record_carries_both_segments(tokenizer, tags):
    ids = _gen(tokenizer, where="a b", color="x y z")
    rec = build_record(_sample(), ids, tags, tokenizer, split="V_where",
                       checkpoint="ckpt")
    assert rec["schema_version"] == SCHEMA_GENCTX
    assert rec["mode"] == "two_segment" and rec["where_suppressed"] is False
    assert rec["where_ids"][0] == tags.where_open
    assert rec["where_ids"][-1] == tags.where_close
    assert rec["color_ids"][0] == tags.color_open
    assert rec["color_ids"][-1] == tags.color_close
    assert not rec["format_failure"] and not rec["color_format_failure"]
    assert rec["starts_with_where_open"] and rec["starts_with_color_open"]
    # the two spans partition the generation, in order and without overlap
    assert rec["segments"]["where"]["end"] <= rec["segments"]["color"]["start"]
    assert rec["where_ids"] + rec["color_ids"] == rec["generated_ids"]


def test_a_malformed_where_does_not_cascade_into_a_colour_failure(tokenizer, tags):
    """A where span that never closed is cut at an arbitrary 96-token boundary
    that can sit past the <color> tag; the colour segment has its own tags and
    must stay recoverable."""
    ids = [55] * 5 + tokenizer("<color>x</color>")["input_ids"]
    rec = build_record(_sample(), ids, tags, tokenizer, split="V_where", checkpoint="c")
    assert rec["format_failure"] and rec["stop_reason"] == "no_close_tag"
    assert not rec["color_format_failure"]        # colour still recoverable
    assert rec["color_ids"][0] == tags.color_open
    assert rec["color_ids"][-1] == tags.color_close
    assert rec["segments_overlap"] is True        # and the overlap is declared


def test_well_formed_segments_are_disjoint_and_do_not_declare_an_overlap(tokenizer, tags):
    rec = build_record(_sample(), _gen(tokenizer), tags, tokenizer,
                       split="V_where", checkpoint="c")
    assert rec["segments_overlap"] is False
    assert rec["segments"]["where"]["end"] == rec["segments"]["color"]["start"]


def test_colour_failure_is_recorded_not_repaired(tokenizer, tags):
    ids = tokenizer("<where>a</where><color>x")["input_ids"]
    rec = build_record(_sample(), ids, tags, tokenizer, split="V_where", checkpoint="c")
    assert not rec["format_failure"]              # where is fine
    assert rec["color_format_failure"]
    assert rec["color_stop_reason"] == "no_close_tag"
    assert rec["color_ids"]                        # what the model actually said


# --- backward compatibility -------------------------------------------------

def test_every_v1_field_is_present_in_a_v2_record(tokenizer, tags):
    rec = build_record(_sample(), _gen(tokenizer), tags, tokenizer,
                       split="V_where", checkpoint="c")
    missing = [f for f in SCHEMA_GENCTX_V1_FIELDS if f not in rec]
    assert not missing, missing
    # the v1 `gen` sub-keys too
    for k in ("max_new_tokens", "max_context_tokens", "do_sample", "close_id",
              "open_id", "eos_id"):
        assert k in rec["gen"], k


def test_v1_fields_still_describe_the_where_segment(tokenizer, tags):
    """A v1 consumer must read the same numbers out of a v2 record."""
    ids = _gen(tokenizer, where="a b c", color="x y")
    rec = build_record(_sample(), ids, tags, tokenizer, split="V_where", checkpoint="c")
    ctx = generated_context(rec["sample_id"], rec["generated_ids"],
                            rec["gen"]["close_id"], eos_id=rec["gen"]["eos_id"])
    assert ctx.token_ids == rec["where_ids"]
    assert ctx.format_failure == rec["format_failure"]
    assert ctx.truncated == rec["truncated"]
    assert ctx.stop_reason == rec["stop_reason"]


def test_where_ids_do_not_change_when_the_budget_grows(tokenizer, tags):
    """Greedy decoding is prefix-deterministic: a longer generation cannot move
    the tokens before the first </where>."""
    short = _gen(tokenizer, where="a b", color="")[:len(
        tokenizer("<where>a b</where>")["input_ids"])]
    long = _gen(tokenizer, where="a b", color="x y z")
    a = build_record(_sample(), short, tags, tokenizer, split="s", checkpoint="c")
    b = build_record(_sample(), long, tags, tokenizer, split="s", checkpoint="c")
    assert a["where_ids"] == b["where_ids"]
    assert a["format_failure"] == b["format_failure"]


def test_the_whereb_consumer_path_is_untouched_by_the_new_schema(tokenizer, tags):
    """`BatchBuilder.context_for` reads generated_ids + the </where> id only."""
    rec = build_record(_sample(), _gen(tokenizer), tags, tokenizer,
                       split="V_where", checkpoint="c")
    close_id = tokenizer("</where>")["input_ids"][0]
    ctx = generated_context(rec["sample_id"], rec["generated_ids"], close_id,
                            text=rec.get("generated_text", ""),
                            eos_id=tokenizer.eos_token_id)
    assert ctx.mode == "generated"
    assert ctx.token_ids == rec["where_ids"]
    assert ctx.instruction is None


# --- forced-prefix mode -----------------------------------------------------

def test_forced_color_mode_has_no_where_segment_and_no_where_failure(tokenizer, tags):
    ids = tokenizer("<color>x y</color>")["input_ids"]
    rec = build_record(_sample(), ids, tags, tokenizer, split="V_what",
                       checkpoint="c", mode="forced_color",
                       forced_prefix_ids=[tags.color_open])
    assert rec["mode"] == "forced_color"
    assert rec["where_ids"] == [] and rec["where_suppressed"] is True
    # a suppressed segment is NOT a format failure: no <where> was asked for
    assert rec["format_failure"] is False
    assert rec["stop_reason"] == "suppressed_by_mode"
    assert rec["color_ids"][0] == tags.color_open
    assert not rec["color_format_failure"]
    assert rec["gen"]["forced_prefix_ids"] == [tags.color_open]
    assert rec["gen"]["mode"] == "forced_color"


def test_forced_color_mode_still_records_a_colour_failure(tokenizer, tags):
    rec = build_record(_sample(), [tags.color_open] + [3] * 900, tags, tokenizer,
                       split="V_what", checkpoint="c", mode="forced_color",
                       forced_prefix_ids=[tags.color_open])
    assert rec["color_format_failure"] and rec["color_truncated"]
    assert len(rec["color_ids"]) == COLOR_CONTEXT_MAX_TOKENS


def test_unknown_mode_is_rejected(tokenizer, tags):
    with pytest.raises(ValueError, match="mode"):
        build_record(_sample(), [1], tags, tokenizer, split="s", checkpoint="c",
                     mode="whatever")
    assert GENCTX_MODES == ("two_segment", "forced_color")


def test_forced_prefix_is_prepended_to_the_returned_generation():
    """`FrozenVLM.generate_where` returns the full continuation, prefix included."""
    import torch

    from q3vl.whereb.hiddens import FrozenVLM

    calls = {}

    class StubVLM:
        pad_id = 0
        device = torch.device("cpu")
        model = SimpleNamespace(
            generate=lambda **kw: (calls.update(kw) or torch.tensor(
                [[0] * kw["input_ids"].shape[1] + [41, 42]])),
            parameters=lambda: iter([torch.zeros(1)]),
        )
        processor = SimpleNamespace(image_processor=lambda **kw: {
            "pixel_values": torch.zeros(1, 3), "image_grid_thw": torch.ones(1, 3).long()})

    stub = StubVLM()
    out = FrozenVLM.generate_where(
        stub, [SimpleNamespace(sample_id="a", image=None, prompt_ids=[7, 8])],
        max_new_tokens=4, eos_token_id=None, prefix_ids=[99])
    assert out == [[99, 41, 42]]
    # the prefix was also appended to the prompt that was decoded from
    assert calls["input_ids"].tolist() == [[7, 8, 99]]


# --- summaries and the store ------------------------------------------------

def test_summary_reports_both_segments(tokenizer, tags):
    good = build_record(_sample("a"), _gen(tokenizer), tags, tokenizer,
                        split="s", checkpoint="c")
    bad = build_record(_sample("b"), tokenizer("<where>a</where><color>x")["input_ids"],
                       tags, tokenizer, split="s", checkpoint="c")
    s = summarise_records([good, bad])
    assert s["n"] == 2
    assert s["format_failure_rate"] == 0.0            # both where segments closed
    assert s["color"]["n"] == 2
    assert s["color"]["format_failure_rate"] == 0.5
    assert s["both_segments_well_formed_rate"] == 0.5
    assert s["modes"] == ["two_segment"]


def test_summary_tolerates_a_v1_record(tokenizer, tags):
    v2 = build_record(_sample("a"), _gen(tokenizer), tags, tokenizer,
                      split="s", checkpoint="c")
    v1 = {k: v for k, v in v2.items() if k in SCHEMA_GENCTX_V1_FIELDS}
    v1["schema_version"] = SCHEMA_GENCTX_V1
    v1["sample_id"] = "b"
    s = summarise_records([v1])
    assert s["n"] == 1 and "color" not in s
    mixed = summarise_records([v1, v2])
    assert mixed["n"] == 2 and mixed["color"]["n"] == 1


def test_store_round_trips_a_v2_record_and_rejects_colour_on_v1(tmp_path, tokenizer, tags):
    v2 = build_record(_sample("sft_a"), _gen(tokenizer), tags, tokenizer,
                      split="V_where", checkpoint="c")
    v1 = {k: v for k, v in v2.items() if k in SCHEMA_GENCTX_V1_FIELDS}
    v1["schema_version"] = SCHEMA_GENCTX_V1
    v1["sample_id"] = "sft_b"
    publish_generated(iter([v2, v1]), tmp_path / "gen", "V_where")
    st = GenContextStore(tmp_path / "gen")
    assert st.where_ids("sft_a") == v2["where_ids"]
    assert st.where_ids("sft_b") == v1["where_ids"]     # v1 still readable
    assert st.color_ids("sft_a") == v2["color_ids"]
    with pytest.raises(KeyError, match="amendment A-4"):
        st.color_ids("sft_b")
    s = st.summary()
    assert s["n"] == 2 and s["color"]["n"] == 1
    assert s["status"] == "complete"


def test_budget_constants_cover_the_measured_corpus():
    # tokens.color over V_where+V_what+T_final (2711 records): max 324, p99 283
    assert COLOR_CONTEXT_MAX_TOKENS >= 324 + 2
    # tokens.where + tokens.color: max 332, plus four tags
    assert GEN_MAX_NEW_TOKENS >= 332 + 4
    assert GEN_MAX_NEW_TOKENS > WHERE_CONTEXT_MAX_TOKENS + COLOR_CONTEXT_MAX_TOKENS - 100
