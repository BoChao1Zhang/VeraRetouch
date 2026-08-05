"""Amendment A-4 -- the teacher / generated ``<color>`` context.

    training   50% GT ``<color>`` hidden / 50% Base-SFT-generated ``<color>`` hidden
    evaluation GT and generated reported separately
    selection  the generated-context board is the main one
    controls   ``C01``/``C02`` generate from a prompt with no ``<where>`` and a
               forced ``<color>`` open tag

This is protocol 5.4's discipline, transposed from ``<where>`` to ``<color>``.
The original Stage-What implementation was 100% teacher-forced -- every
``color_ids`` came from the record's GT text -- and never declared it
(``docs/reviews/REVIEW-impl-What.md`` NF-1).  That left protocol 0's central
claim ("the final network receives only ``I_in + instruction``") untested in
every Stage-What number, and it is not repairable after the fact: twelve arms
trained on teacher text and evaluated on generated text would simply lose points,
and the only correct response would be to retrain all twelve.

**There is no GT fallback, and it is enforced structurally, not by a check.**
:func:`generated_color_context` never receives the GT text, so there is no value
it could fall back *to*; a generation with no closing tag is cut at
``COLOR_CONTEXT_MAX_TOKENS`` and marked ``format_failure``.  This mirrors
:mod:`q3vl.whereb.context` exactly, including the reason for mirroring it: the
two stages' context handling has to be comparable or the "generated context is
worse by X" numbers cannot be read side by side.

Only **token ids** are cached and replayed.  The hidden states are re-derived at
training time by the same :meth:`q3vl.what.hiddens.WhatVLM.encode` the teacher
context uses, which makes "same layer, same position, same normalisation" true by
construction rather than by discipline (the argument is
:mod:`q3vl.whereb.gencontext`'s, and it applies unchanged here).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from q3vl.train.constants import COLOR_CLOSE, COLOR_OPEN
# One convention for both stages: the sampler and the format counters are
# Where-B's, not re-implementations of them.  ``FormatStats.update`` reads only
# ``format_failure`` / ``truncated`` / ``n_tokens`` / ``stop_reason``, all of
# which :class:`ColorContext` provides under the same names.
from q3vl.whereb.context import BalancedContextSampler, FormatStats  # noqa: F401

from .config import (
    COLOR_CONTEXT_MAX_TOKENS,
    CONTEXT_GENERATED,
    CONTEXT_GT,
    CONTEXT_MODES,
    GENCTX_MODES,
)

__all__ = ["ColorContext", "gt_color_context", "generated_color_context",
           "encode_color_span", "BalancedContextSampler", "FormatStats",
           "CONTEXT_GT", "CONTEXT_GENERATED", "CONTEXT_MODES"]


@dataclass
class ColorContext:
    """The ``<color>`` token span a single forward will condition on."""

    mode: str
    token_ids: list[int]
    text: str = ""
    provenance: str = ""            # sample_id whose text this is
    format_failure: bool = False
    truncated: bool = False
    stop_reason: str = ""           # closed | closed_over_boundary | no_close_tag | empty
    n_generated_tokens: int = 0
    genctx_mode: str = ""           # with_where_prefix | forced_color_prefix

    def __post_init__(self) -> None:
        if self.mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode {self.mode!r}")
        if self.genctx_mode and self.genctx_mode not in GENCTX_MODES:
            raise ValueError(f"unknown genctx mode {self.genctx_mode!r}")
        if self.mode == CONTEXT_GT and self.genctx_mode:
            raise ValueError("a teacher context has no generation mode")

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "n_tokens": self.n_tokens,
                "provenance": self.provenance, "format_failure": self.format_failure,
                "truncated": self.truncated, "stop_reason": self.stop_reason,
                "n_generated_tokens": self.n_generated_tokens,
                "genctx_mode": self.genctx_mode}


def encode_color_span(tokenizer, color_text: str) -> list[int]:
    """``<color>{body}</color>`` -> ids, tokenised exactly as the SFT collator does."""
    return list(tokenizer(f"{COLOR_OPEN}{color_text}{COLOR_CLOSE}",
                          add_special_tokens=False)["input_ids"])


def gt_color_context(tokenizer, sample_id: str, color_text: str,
                     max_tokens: int = COLOR_CONTEXT_MAX_TOKENS) -> ColorContext:
    ids = encode_color_span(tokenizer, color_text)
    if len(ids) > max_tokens:
        # An assertion about the corpus, not a truncation path.  The boundary was
        # derived from a 3,745-record measurement (max 324 + tags); a sample past
        # it means the measurement has to be redone, not that a teacher context
        # should be silently shortened.
        raise ValueError(
            f"{sample_id}: GT <color> span is {len(ids)} tokens > boundary "
            f"{max_tokens}; the boundary was set from the measured corpus and "
            "must be re-derived, not silently applied to teacher context"
        )
    return ColorContext(mode=CONTEXT_GT, token_ids=ids, text=color_text,
                        provenance=sample_id, stop_reason="closed")


def generated_color_context(
    sample_id: str,
    generated_ids: Sequence[int],
    close_id: int,
    *,
    text: str = "",
    max_tokens: int = COLOR_CONTEXT_MAX_TOKENS,
    eos_id: int | None = None,
    genctx_mode: str = "",
) -> ColorContext:
    """The Base SFT model's own ``<color>`` span.  No GT is reachable from here.

    ``text`` is **metadata only** -- it lands in ``ColorContext.text`` and never
    in ``token_ids`` -- so the no-GT-fallback proof covers the conditioning path
    whatever a caller puts there (review N-22).  It is still worth being strict
    about: a call site that passed the GT body would poison the per-sample
    provenance log while the model saw the right tokens.  ``data.py`` therefore
    takes it from the published record and nothing else, and
    ``test_a4_color_context`` asserts that by reading the call site.

    ``generated_ids`` are the ids of the ``<color>`` segment as published by the
    generation job -- for ``with_where_prefix`` that is the slice after
    ``</where>``; for ``forced_color_prefix`` it is the forced ``<color>`` tag
    plus everything generated after it.  The span kept is up to and including the
    first ``</color>``; if there is none, the first ``max_tokens`` ids are kept
    and ``format_failure`` is set.
    """
    ids = list(generated_ids)
    n_gen = len(ids)
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    if close_id in ids:
        cut = ids.index(close_id) + 1
        span = ids[:cut]
        failure, truncated, reason = False, False, "closed"
        if len(span) > max_tokens:
            span = span[:max_tokens]
            truncated, failure, reason = True, True, "closed_over_boundary"
    else:
        span = ids[:max_tokens]
        failure = True
        truncated = len(ids) > max_tokens
        reason = "no_close_tag"
    if not span:
        failure, reason = True, "empty"
    return ColorContext(
        mode=CONTEXT_GENERATED, token_ids=span, text=text, provenance=sample_id,
        format_failure=failure, truncated=truncated, stop_reason=reason,
        n_generated_tokens=n_gen, genctx_mode=genctx_mode,
    )


def context_breakdown(contexts: Sequence[ColorContext]) -> dict[str, Any]:
    """Per-mode counts for a batch -- the 50/50 ratio, made auditable per step."""
    out: dict[str, Any] = {"n": len(contexts)}
    for m in CONTEXT_MODES:
        sel = [c for c in contexts if c.mode == m]
        out[f"n_{m}"] = len(sel)
        if sel:
            out[f"format_failure_rate_{m}"] = (
                sum(c.format_failure for c in sel) / len(sel))
    if contexts:
        out["teacher_fraction"] = out[f"n_{CONTEXT_GT}"] / len(contexts)
    return out


def iter_modes(sampler: BalancedContextSampler) -> Iterator[list[tuple[int, str]]]:
    """The 50/50 micro-batches, with Where-B's mode strings mapped onto ours.

    ``BalancedContextSampler`` yields ``q3vl.whereb.context``'s ``GT`` /
    ``GENERATED`` constants, which are the same two strings this module uses;
    the mapping is asserted rather than assumed so a rename on either side is a
    loud failure instead of a batch of teacher contexts labelled ``generated``.
    """
    from q3vl.whereb.context import GENERATED as WB_GENERATED, GT as WB_GT

    if (WB_GT, WB_GENERATED) != (CONTEXT_GT, CONTEXT_GENERATED):
        raise AssertionError(
            f"context mode strings diverged: Where-B uses ({WB_GT!r}, "
            f"{WB_GENERATED!r}), Stage-What uses ({CONTEXT_GT!r}, "
            f"{CONTEXT_GENERATED!r})"
        )
    yield from sampler
