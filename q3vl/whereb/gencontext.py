"""Producing the generated ``<where>`` / ``<color>`` context in bulk (5.4 + 2.3).

What is cached
--------------
**Token ids, not hidden states.**  The Base SFT model's continuation is ~204
tokens for the two segments together (measured: ``tokens.where + tokens.color``
p50 204, max 332); caching those ids costs ~1 KB per sample, while caching the
hidden states costs ``T x 2560 x 2 B`` and, more importantly, *forks the forward
contract*: a cached hidden would have been produced by a different call than the
teacher hidden it is compared against.  Replaying the cached ids through
:meth:`q3vl.whereb.hiddens.FrozenVLM.encode` -- the exact function the teacher
context uses -- makes "same layer, same position, same normalisation" true by
construction instead of by discipline.  The training loop pays one extra text
forward, which it was going to run for ``F_pre`` anyway.

Both segments, one generation (amendment A-4)
---------------------------------------------
Stage-What adopts the same fixed 50/50 teacher/generated context split as
Where-B, so it needs the model's own ``<color>`` segment too.  The Base SFT
model emits ``<where>...</where><color>...</color>`` in a single greedy
continuation, so this job simply stops throwing the second half away.  Two
consequences worth stating:

* the ``<where>`` ids are **unchanged** by the larger budget -- greedy decoding
  is prefix-deterministic, so the tokens before the first ``</where>`` are the
  same ones a where-only run produced;
* the schema grows (``/1`` -> ``/2``) by **addition only**: every v1 field keeps
  its name and its meaning (the ``<where>`` segment), so Where-B's consumer is
  untouched.

Both segments obey the same malformed-output rule: keep up to and including the
first closing tag, otherwise cut at a fixed token boundary and record a format
failure.  Neither ever falls back to GT --
:func:`q3vl.whereb.context.extract_segment` cannot see any GT text.

Forced-prefix mode (Stage-What controls C01/C02)
------------------------------------------------
``mode="forced_color"`` forces ``<color>`` as the assistant's first token, so the
colour segment is produced without any ``<where>`` reasoning in front of it.
That is the strict no-where control: ``WC-0 ColorOnly`` still sees a causal
language state that passed through a generated ``<where>``, and protocol 8.2
asks for a control that does not.

Throughput
----------
Greedy, left-padded HF batch generation with the KV cache, single GPU, bf16.
vLLM 0.16.0 *is* available on this box (conda env ``vllm``; its registry maps
``Qwen3VLForConditionalGeneration`` -> ``qwen3_vl``, verified 2026-08-05) and
would be faster for the text, but it lives in a different torch/transformers
build, it cannot emit the hidden states, and the ids it produces would come from
a different numerical stack than the one Where-B trains in.  Since the ids are
the only artefact, determinism across the two stacks is the thing that matters,
so the conservative default is the training environment's own HF generate.

Everything is published as indexed tar shards (protocol 2.3), never as a
directory of small JSON files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from q3vl.data.shardio import build_from_memory
from q3vl.train.constants import COLOR_CLOSE, COLOR_OPEN, WHERE_CLOSE, WHERE_OPEN

from .config import (
    COLOR_CONTEXT_MAX_TOKENS,
    GENCTX_MODES,
    GENCTX_SHARD_BYTES,
    GEN_MAX_NEW_TOKENS,
    SCHEMA_GENCTX,
    WHERE_CONTEXT_MAX_TOKENS,
)
from .context import extract_segment
from .data import WhereBDataset, _PromptShim
from .hiddens import EncodeItem, FrozenVLM

__all__ = ["PRODUCER", "SUFFIX", "SegmentIds", "genwhere_payload", "build_record",
           "generate_records", "publish_generated", "summarise_records"]

PRODUCER = "q3vl.whereb.gencontext/2"
SUFFIX = ".genwhere.json"


class SegmentIds:
    """The four tag ids this job needs, resolved once from the tokenizer."""

    def __init__(self, tokenizer):
        def one(tok: str) -> int:
            ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
            if len(ids) != 1:
                raise ValueError(f"{tok!r} is not a single token: {ids}")
            return int(ids[0])

        self.where_open = one(WHERE_OPEN)
        self.where_close = one(WHERE_CLOSE)
        self.color_open = one(COLOR_OPEN)
        self.color_close = one(COLOR_CLOSE)
        self.eos = tokenizer.eos_token_id

    def to_dict(self) -> dict[str, Any]:
        return {"where_open_id": self.where_open, "where_close_id": self.where_close,
                "color_open_id": self.color_open, "color_close_id": self.color_close}


def genwhere_payload(record: dict[str, Any]) -> tuple[str, bytes]:
    sid = record["sample_id"]
    blob = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    return f"{sid}{SUFFIX}", blob


def build_record(
    sample,
    generated_ids: Sequence[int],
    tags: SegmentIds,
    tokenizer,
    *,
    split: str,
    checkpoint: str,
    mode: str = "two_segment",
    where_max_tokens: int = WHERE_CONTEXT_MAX_TOKENS,
    color_max_tokens: int = COLOR_CONTEXT_MAX_TOKENS,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    forced_prefix_ids: Sequence[int] = (),
) -> dict[str, Any]:
    """One schema-v2 record.  v1 fields keep their v1 names *and* meanings."""
    if mode not in GENCTX_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {GENCTX_MODES}")
    ids = [int(t) for t in generated_ids]

    if mode == "forced_color":
        # no <where> was asked for, so "the where segment failed" would be a lie
        where = extract_segment([], tags.where_close, where_max_tokens)
        where.stop_reason = "suppressed_by_mode"
        where.format_failure = False
        color = extract_segment(ids, tags.color_close, color_max_tokens,
                                open_id=tags.color_open, eos_id=tags.eos)
    else:
        where = extract_segment(ids, tags.where_close, where_max_tokens,
                                eos_id=tags.eos)
        # The colour segment normally starts where the where segment ended.  But
        # a where span that never closed has no meaningful end -- it was cut at
        # an arbitrary 96-token boundary that can sit *past* the <color> tag, and
        # searching from there would turn one format failure into two and throw
        # away a colour segment that is perfectly well formed.  The segments are
        # delimited by their own tags, so in that case search the whole
        # generation and record the overlap instead of hiding it.
        color_start = where.end if where.stop_reason == "closed" else 0
        color = extract_segment(ids, tags.color_close, color_max_tokens,
                                open_id=tags.color_open, start=color_start,
                                eos_id=tags.eos)

    def dec(t):
        return tokenizer.decode(list(t), skip_special_tokens=False)

    return {
        # --- schema v1 (unchanged names, unchanged meaning = <where>) --------
        "schema_version": SCHEMA_GENCTX,
        "sample_id": sample.sample_id,
        "split": split,
        "build": sample.meta.get("build"),
        "render_mode": sample.meta.get("render_mode"),
        "winner_confidence": sample.meta.get("winner_confidence"),
        "checkpoint": checkpoint,
        "generated_ids": ids,
        "generated_text": dec(ids),
        "n_generated_tokens": len(ids),
        "where_ids": list(where.token_ids),
        "where_text": dec(where.token_ids),
        "format_failure": where.format_failure,
        "truncated": where.truncated,
        "stop_reason": where.stop_reason,
        "starts_with_where_open": bool(ids and ids[0] == tags.where_open),
        # --- new in schema v2 (amendment A-4) -------------------------------
        "mode": mode,
        "where_suppressed": mode == "forced_color",
        "color_ids": list(color.token_ids),
        "color_text": dec(color.token_ids),
        "color_format_failure": color.format_failure,
        "color_truncated": color.truncated,
        "color_stop_reason": color.stop_reason,
        "starts_with_color_open": bool(
            color.token_ids and color.token_ids[0] == tags.color_open
        ),
        "segments": {"where": where.to_dict(), "color": color.to_dict()},
        # true only when a malformed where span was cut past the <color> tag;
        # the colour segment is still the authoritative one (its own tags)
        "segments_overlap": bool(color.token_ids) and color.start < where.end,
        "gen": {
            "max_new_tokens": max_new_tokens,
            "max_context_tokens": where_max_tokens,          # v1 name
            "where_max_tokens": where_max_tokens,
            "color_max_tokens": color_max_tokens,
            "do_sample": False,
            "close_id": tags.where_close,                    # v1 name
            "open_id": tags.where_open,                      # v1 name
            "eos_id": tags.eos,
            "mode": mode,
            "forced_prefix_ids": [int(t) for t in forced_prefix_ids],
            "forced_prefix_text": dec(forced_prefix_ids) if forced_prefix_ids else "",
            **tags.to_dict(),
        },
    }


def generate_records(
    vlm: FrozenVLM,
    collator,
    dataset: WhereBDataset,
    *,
    batch_size: int = 8,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    max_context_tokens: int = WHERE_CONTEXT_MAX_TOKENS,
    color_max_tokens: int = COLOR_CONTEXT_MAX_TOKENS,
    checkpoint: str = "",
    limit: int | None = None,
    progress_every: int = 200,
    mode: str = "two_segment",
) -> Iterator[dict[str, Any]]:
    """Greedy continuations for a whole split, one record per sample."""
    if mode not in GENCTX_MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {GENCTX_MODES}")
    tokenizer = collator.tokenizer
    tags = SegmentIds(tokenizer)
    prefix = [tags.color_open] if mode == "forced_color" else []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    t0 = time.time()
    done = 0

    for start in range(0, n, batch_size):
        idx = list(range(start, min(start + batch_size, n)))
        samples = [dataset[i] for i in idx]
        items = []
        for s in samples:
            enc = collator.encode_one(_PromptShim(s))
            items.append(EncodeItem(
                sample_id=s.sample_id, image=s.image,
                prompt_ids=enc["input_ids"][:enc["n_prompt_tokens"]],
            ))
        gens = vlm.generate_where(items, max_new_tokens=max_new_tokens,
                                  eos_token_id=tags.eos, prefix_ids=prefix)
        for s, ids in zip(samples, gens):
            yield build_record(
                s, ids, tags, tokenizer, split=dataset.split, checkpoint=checkpoint,
                mode=mode, where_max_tokens=max_context_tokens,
                color_max_tokens=color_max_tokens, max_new_tokens=max_new_tokens,
                forced_prefix_ids=prefix,
            )
        done += len(samples)
        if progress_every and done % progress_every < batch_size:
            rate = done / max(1e-9, time.time() - t0)
            print(json.dumps({"done": done, "of": n, "samples_per_s": round(rate, 2),
                              "eta_s": round((n - done) / max(rate, 1e-9))}), flush=True)


def publish_generated(
    records: Iterable[dict[str, Any]], out_root: Path, split: str
) -> dict[str, Any]:
    """Pack the records as an indexed tar dataset (atomic publish)."""
    return build_from_memory(
        (genwhere_payload(r) for r in records), Path(out_root),
        shard_size_bytes=GENCTX_SHARD_BYTES, producer=PRODUCER,
        source_label=f"where_b.genwhere/{split}",
    )


def _lens(xs: Sequence[int]) -> dict[str, int] | None:
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return {"min": s[0], "p50": s[n // 2],
            "p95": s[min(n - 1, int(0.95 * (n - 1)))], "max": s[-1]}


def _rate(records: Sequence[dict[str, Any]], key: str) -> float | None:
    """Rate over the records that actually carry ``key``.

    Deliberately tolerant: this summarises whatever a published shard set holds,
    which may mix schema versions, and a summary is not the place to crash.
    """
    have = [r for r in records if key in r]
    return (sum(bool(r[key]) for r in have) / len(have)) if have else None


def _counts(records: Sequence[dict[str, Any]], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        v = r.get(key)
        if v is not None:
            out[str(v)] = out.get(str(v), 0) + 1
    return dict(sorted(out.items()))


def summarise_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Per-segment rates.  Tolerant of v1 records (no colour fields)."""
    records = list(records)
    n = len(records)
    if not n:
        return {"n": 0}
    has_color = [r for r in records if "color_ids" in r]
    out: dict[str, Any] = {
        "n": n,
        "modes": sorted({str(r.get("mode", "two_segment")) for r in records}),
        "schema_versions": sorted({str(r.get("schema_version")) for r in records}),
        # v1 keys keep describing the <where> segment
        "format_failure_rate": _rate(records, "format_failure"),
        "truncation_rate": _rate(records, "truncated"),
        "starts_with_where_open_rate": _rate(records, "starts_with_where_open"),
        "stop_reasons": _counts(records, "stop_reason"),
        "where_tokens": _lens([len(r["where_ids"]) for r in records if "where_ids" in r]),
        "generated_tokens": _lens(
            [r["n_generated_tokens"] for r in records if "n_generated_tokens" in r]),
        "segments_overlap_rate": _rate(records, "segments_overlap"),
    }
    if has_color:
        out["color"] = {
            "n": len(has_color),
            "format_failure_rate": _rate(has_color, "color_format_failure"),
            "truncation_rate": _rate(has_color, "color_truncated"),
            "starts_with_color_open_rate": _rate(has_color, "starts_with_color_open"),
            "stop_reasons": _counts(has_color, "color_stop_reason"),
            "color_tokens": _lens([len(r["color_ids"]) for r in has_color]),
        }
        both = [r for r in has_color
                if "format_failure" in r and "color_format_failure" in r]
        if both:
            out["both_segments_well_formed_rate"] = sum(
                (not r["format_failure"]) and (not r["color_format_failure"])
                for r in both
            ) / len(both)
    return out
