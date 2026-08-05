"""Protocol 5.4 -- the four ``<where>`` context flows and the 50/50 training mix.

    "The training batch is a fixed 50% teacher context / 50% generated context:
     teacher = the GT ``<where>`` token hidden states; generated = the ``<where>``
     token hidden states of the Base SFT model's own autoregressive output.
     A generated sample that is missing its closing tag must NOT fall back to
     GT; it is cut at a fixed maximum token boundary and the format failure is
     recorded.  Model selection is driven by the generated context.

     Every checkpoint must report the four contexts separately, never as one
     average: GT / generated / null / shuffled.  ``shuffled`` swaps the
     instruction/where context inside the same image and the same local level."

The no-fallback rule is enforced structurally rather than by a runtime check:
:func:`generated_context` never receives the GT text, so there is no value it
could fall back *to*.  What it does instead is cut at
``WHERE_CONTEXT_MAX_TOKENS`` and set ``format_failure``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

from q3vl.train.constants import WHERE_CLOSE, WHERE_OPEN

from .config import SHUFFLE_GROUP_KEYS, WHERE_CONTEXT_MAX_TOKENS

__all__ = [
    "GT", "GENERATED", "NULL", "SHUFFLED", "CONTEXT_MODES",
    "WhereContext", "FormatStats", "SegmentSpan", "extract_segment",
    "gt_context", "generated_context", "null_context", "shuffled_context",
    "ShuffleIndex", "BalancedContextSampler",
]

GT = "gt"
GENERATED = "generated"
NULL = "null"
SHUFFLED = "shuffled"
CONTEXT_MODES = (GT, GENERATED, NULL, SHUFFLED)


@dataclass
class WhereContext:
    """The ``<where>`` token span a single forward will condition on.

    ``instruction`` overrides the prompt's instruction.  Only the ``shuffled``
    mode sets it: protocol 5.4 says the control "swaps the instruction/where
    context", and ``Q_where``'s *only* route to the instruction is through
    ``H_where`` (causal attention lets the ``<where>`` positions attend to the
    prompt).  Leaving the sample's own instruction in place would keep the right
    answer reachable and turn the §5.6 "instruction shuffle" gate into a control
    that catches neither a cheating model nor a good one (review blocker B3).
    """

    mode: str
    token_ids: list[int]
    text: str = ""
    provenance: str = ""            # sample_id whose text this is ("" for null)
    format_failure: bool = False
    truncated: bool = False
    stop_reason: str = ""           # closed | max_tokens | eos | empty
    n_generated_tokens: int = 0
    instruction: str | None = None  # prompt override; set by `shuffled` only

    def __post_init__(self) -> None:
        if self.mode not in CONTEXT_MODES:
            raise ValueError(f"unknown context mode {self.mode!r}")
        if self.mode == NULL and self.token_ids:
            raise ValueError("the null context must carry no tokens")
        if self.instruction is not None and self.mode != SHUFFLED:
            raise ValueError(
                f"only the shuffled context may override the instruction, not {self.mode!r}"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "n_tokens": self.n_tokens, "provenance": self.provenance,
            "format_failure": self.format_failure, "truncated": self.truncated,
            "stop_reason": self.stop_reason,
            "n_generated_tokens": self.n_generated_tokens,
            "instruction_swapped": self.instruction is not None,
        }


@dataclass
class FormatStats:
    n: int = 0
    n_format_failure: int = 0
    n_truncated: int = 0
    n_empty: int = 0
    stop_reasons: dict[str, int] = field(default_factory=dict)

    def update(self, ctx: WhereContext) -> None:
        self.n += 1
        self.n_format_failure += int(ctx.format_failure)
        self.n_truncated += int(ctx.truncated)
        self.n_empty += int(ctx.n_tokens == 0 and ctx.mode != NULL)
        self.stop_reasons[ctx.stop_reason] = self.stop_reasons.get(ctx.stop_reason, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "format_failure_rate": self.n_format_failure / self.n if self.n else None,
            "truncation_rate": self.n_truncated / self.n if self.n else None,
            "empty_rate": self.n_empty / self.n if self.n else None,
            "stop_reasons": dict(sorted(self.stop_reasons.items())),
        }


# --- builders ---------------------------------------------------------------

def encode_where_span(tokenizer, where_text: str) -> list[int]:
    """``<where>{body}</where>`` -> ids, tokenised exactly as the SFT collator does."""
    return list(tokenizer(f"{WHERE_OPEN}{where_text}{WHERE_CLOSE}",
                          add_special_tokens=False)["input_ids"])


def gt_context(tokenizer, sample_id: str, where_text: str,
               max_tokens: int = WHERE_CONTEXT_MAX_TOKENS) -> WhereContext:
    ids = encode_where_span(tokenizer, where_text)
    if len(ids) > max_tokens:
        # Measured GT spans top out at 81 tokens against a 96 boundary, so this
        # is an assertion about the data, not a truncation path.
        raise ValueError(
            f"{sample_id}: GT <where> span is {len(ids)} tokens > boundary {max_tokens}; "
            "the boundary was set from the measured corpus and must be re-derived, "
            "not silently applied to teacher context"
        )
    return WhereContext(mode=GT, token_ids=ids, text=where_text,
                        provenance=sample_id, stop_reason="closed")


@dataclass
class SegmentSpan:
    """One tagged segment carved out of a generation.

    The same rule governs ``<where>`` and ``<color>`` (amendment A-4): keep up to
    and including the first closing tag; if there is none, cut at a fixed token
    boundary and record a format failure.  **Neither ever falls back to GT** --
    this function cannot see any GT text.
    """

    token_ids: list[int]
    format_failure: bool
    truncated: bool
    stop_reason: str          # closed | closed_over_boundary | no_close_tag |
                              # no_open_tag | empty
    start: int = 0            # index in the generation where the span begins
    end: int = 0              # one past the last kept token

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"n_tokens": self.n_tokens, "format_failure": self.format_failure,
                "truncated": self.truncated, "stop_reason": self.stop_reason,
                "start": self.start, "end": self.end}


def extract_segment(
    generated_ids: Sequence[int],
    close_id: int,
    max_tokens: int,
    *,
    open_id: int | None = None,
    start: int = 0,
    eos_id: int | None = None,
) -> SegmentSpan:
    """Carve one tagged segment out of a generation.

    ``open_id=None`` (the ``<where>`` case) takes the window as it stands, which
    is what the where extractor has always done -- the first segment is expected
    at position 0 and a missing ``<where>`` shows up as ``no_close_tag`` or as
    ``starts_with_where_open=False`` rather than as a separate failure mode.
    ``open_id`` set (the ``<color>`` case) first seeks the opening tag, because
    the colour segment starts wherever the where segment ended.
    """
    ids = list(generated_ids)
    if eos_id is not None and eos_id in ids:
        ids = ids[: ids.index(eos_id)]
    window = ids[start:]
    offset = start
    if open_id is not None:
        if open_id not in window:
            return SegmentSpan([], True, False, "no_open_tag", offset, offset)
        i = window.index(open_id)
        offset += i
        window = window[i:]

    if close_id in window:
        cut = window.index(close_id) + 1
        span = window[:cut]
        failure, truncated, reason = False, False, "closed"
        if len(span) > max_tokens:
            span = span[:max_tokens]
            truncated, failure, reason = True, True, "closed_over_boundary"
    else:
        span = window[:max_tokens]
        failure = True
        truncated = len(window) > max_tokens
        reason = "no_close_tag"
    if not span:
        failure, reason = True, "empty"
    return SegmentSpan(span, failure, truncated, reason, offset, offset + len(span))


def generated_context(
    sample_id: str,
    generated_ids: Sequence[int],
    close_id: int,
    *,
    text: str = "",
    max_tokens: int = WHERE_CONTEXT_MAX_TOKENS,
    eos_id: int | None = None,
) -> WhereContext:
    """The Base SFT model's own ``<where>`` span.  No GT is reachable from here.

    ``generated_ids`` are the tokens produced after the generation prompt, i.e.
    they should start with ``<where>``.  The span kept is up to and including the
    first ``</where>``; if there is none, the first ``max_tokens`` tokens are
    kept and ``format_failure`` is set.

    Since amendment A-4 the cached generation also contains the ``<color>``
    segment.  That changes nothing here: greedy decoding is prefix-deterministic,
    so the tokens before the first ``</where>`` are identical to what a
    where-only run produced, and this function still cuts there.
    """
    seg = extract_segment(generated_ids, close_id, max_tokens, eos_id=eos_id)
    return WhereContext(
        mode=GENERATED, token_ids=seg.token_ids, text=text, provenance=sample_id,
        format_failure=seg.format_failure, truncated=seg.truncated,
        stop_reason=seg.stop_reason, n_generated_tokens=len(generated_ids),
    )


def null_context() -> WhereContext:
    return WhereContext(mode=NULL, token_ids=[], provenance="", stop_reason="empty")


def shuffled_context(tokenizer, partner_id: str, partner_where_text: str,
                     partner_instruction: str,
                     max_tokens: int = WHERE_CONTEXT_MAX_TOKENS) -> WhereContext:
    """Another sample's **instruction and** GT ``<where>`` text, taken as a pair.

    Protocol 5.4: "``shuffled`` swaps the instruction/where context inside the
    same image and the same local level".  Both come from the *same* partner, so
    the swapped condition stays internally consistent -- a mismatched
    (instruction of A, where-body of B) pair would be a third thing that the
    model never sees in training and would not measure instruction dependence.
    """
    if not isinstance(partner_instruction, str) or not partner_instruction.strip():
        raise ValueError(
            f"shuffled context for partner {partner_id!r} has no instruction; "
            "protocol 5.4 swaps instruction *and* where context (review blocker B3)"
        )
    ids = encode_where_span(tokenizer, partner_where_text)[:max_tokens]
    return WhereContext(mode=SHUFFLED, token_ids=ids, text=partner_where_text,
                        provenance=partner_id, stop_reason="closed",
                        instruction=partner_instruction)


# --- shuffling --------------------------------------------------------------

class ShuffleIndex:
    """Derangement inside ``(source_image_id, render_mode)`` groups.

    A group of one has no partner: those samples are reported as *uncovered*
    rather than paired with an unrelated image, because pairing across images
    would turn the control into "does the model react to a random instruction",
    which is a strictly weaker claim than protocol 5.4 asks for.
    """

    #: a partner has to supply both halves of the swap (protocol 5.4)
    REQUIRED_FIELDS = ("where", "instruction")

    def __init__(self, records: Iterable[dict[str, Any]], seed: int = 0,
                 group_keys: Sequence[str] = SHUFFLE_GROUP_KEYS,
                 require_texts: bool = True):
        self.group_keys = tuple(group_keys)
        groups: dict[tuple, list[str]] = {}
        self.by_id: dict[str, dict[str, Any]] = {}
        for r in records:
            sid = r["sample_id"]
            if require_texts:
                missing = [f for f in self.REQUIRED_FIELDS
                           if not str(r.get(f) or "").strip()]
                if missing:
                    raise ValueError(
                        f"{sid}: shuffle record is missing {missing}; the shuffled "
                        "context swaps instruction *and* where text as a pair"
                    )
            self.by_id[sid] = r
            groups.setdefault(tuple(str(r.get(k)) for k in self.group_keys), []).append(sid)
        self.groups = groups
        self.partner: dict[str, str] = {}
        rng = random.Random(seed)
        for members in groups.values():
            if len(members) < 2:
                continue
            order = list(members)
            rng.shuffle(order)
            for i, sid in enumerate(order):           # cyclic shift = no fixed point
                self.partner[sid] = order[(i + 1) % len(order)]

    def coverage(self) -> dict[str, Any]:
        n = len(self.by_id)
        return {
            "group_keys": list(self.group_keys),
            "n_samples": n,
            "n_groups": len(self.groups),
            "n_with_partner": len(self.partner),
            "coverage": len(self.partner) / n if n else None,
            "group_size_hist": _hist(len(v) for v in self.groups.values()),
        }

    def partner_of(self, sample_id: str) -> str | None:
        return self.partner.get(sample_id)


def _hist(xs: Iterable[int]) -> dict[str, int]:
    out: dict[str, int] = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


# --- the 50/50 training mix -------------------------------------------------

class BalancedContextSampler:
    """Emits micro-batches that are exactly 50% teacher and 50% generated.

    The dataset is split once into two disjoint halves, so a sample is seen with
    exactly one context per epoch ("1 epoch", protocol 10.3) while every batch
    still carries both (protocol 5.4).  Enforcing the ratio inside the
    *micro*-batch also makes it hold for every gradient-accumulated effective
    batch, whatever the accumulation factor is.
    """

    def __init__(self, n: int, micro_batch: int, seed: int = 0,
                 teacher_fraction: float = 0.5, drop_last: bool = True):
        if micro_batch % 2:
            raise ValueError(
                f"micro batch {micro_batch} is odd; a fixed 50/50 context split "
                "(protocol 5.4) needs an even micro batch"
            )
        self.n = n
        self.micro_batch = micro_batch
        self.half = micro_batch // 2
        self.seed = seed
        self.drop_last = drop_last
        rng = random.Random(seed)
        perm = list(range(n))
        rng.shuffle(perm)
        cut = int(round(n * teacher_fraction))
        self.teacher_pool = perm[:cut]
        self.generated_pool = perm[cut:]

    def __len__(self) -> int:
        k = min(len(self.teacher_pool), len(self.generated_pool)) // self.half
        if not self.drop_last:
            k = max(k, 1)
        return k

    def __iter__(self) -> Iterator[list[tuple[int, str]]]:
        h = self.half
        for b in range(len(self)):
            t = self.teacher_pool[b * h:(b + 1) * h]
            g = self.generated_pool[b * h:(b + 1) * h]
            if self.drop_last and (len(t) < h or len(g) < h):
                return
            batch = [(i, GT) for i in t] + [(i, GENERATED) for i in g]
            yield batch

    def facts(self) -> dict[str, Any]:
        return {
            "n": self.n, "micro_batch": self.micro_batch, "seed": self.seed,
            "n_teacher_pool": len(self.teacher_pool),
            "n_generated_pool": len(self.generated_pool),
            "n_batches": len(self),
            "n_samples_used": len(self) * self.micro_batch,
        }
