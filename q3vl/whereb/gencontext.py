"""Producing the generated ``<where>`` context in bulk (protocol 5.4 + 2.3).

What is cached
--------------
**Token ids, not hidden states.**  The Base SFT model's ``<where>`` continuation
is ~41 tokens (measured: local p50 43, max 79); caching those ids costs ~200 B
per sample, while caching the hidden states costs ``T x 2560 x 2 B`` ~= 200 KB
(8-15 GB over the eligible corpus) and, more importantly, *forks the forward
contract*: a cached hidden would have been produced by a different call than the
teacher hidden it is compared against.  Replaying the cached ids through
:meth:`q3vl.whereb.hiddens.FrozenVLM.encode` -- the exact function the teacher
context uses -- makes "same layer, same position, same normalisation" true by
construction instead of by discipline.  The training loop pays one extra text
forward, which it was going to run for ``F_pre`` anyway.

Throughput
----------
Greedy, left-padded HF batch generation with the KV cache, single GPU, bf16.
vLLM 0.16.0 *is* available on this box (conda env ``vllm``; its registry maps
``Qwen3VLForConditionalGeneration`` -> ``qwen3_vl``, verified 2026-08-05) and
would be faster for the text, but it lives in a different torch/transformers
build, it cannot emit the hidden states, and the ids it produces would come from
a different numerical stack than the one Where-B trains in.  Since the ids are
the only artefact, determinism across the two stacks is the thing that matters,
so the conservative default is the training environment's own HF generate.  The
NOTES record vLLM as the fallback if wall-clock becomes the binding constraint.

Everything is published as indexed tar shards (protocol 2.3), never as a
directory of small JSON files.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from q3vl.data.shardio import build_from_memory
from q3vl.train.constants import WHERE_CLOSE, WHERE_OPEN

from .config import GENCTX_SHARD_BYTES, GEN_MAX_NEW_TOKENS, SCHEMA_GENCTX, WHERE_CONTEXT_MAX_TOKENS
from .context import generated_context
from .data import WhereBDataset, _PromptShim
from .hiddens import EncodeItem, FrozenVLM

__all__ = ["PRODUCER", "genwhere_payload", "generate_records", "publish_generated"]

PRODUCER = "q3vl.whereb.gencontext/1"
SUFFIX = ".genwhere.json"


def genwhere_payload(record: dict[str, Any]) -> tuple[str, bytes]:
    sid = record["sample_id"]
    blob = json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    return f"{sid}{SUFFIX}", blob


def generate_records(
    vlm: FrozenVLM,
    collator,
    dataset: WhereBDataset,
    *,
    batch_size: int = 8,
    max_new_tokens: int = GEN_MAX_NEW_TOKENS,
    max_context_tokens: int = WHERE_CONTEXT_MAX_TOKENS,
    checkpoint: str = "",
    limit: int | None = None,
    progress_every: int = 200,
) -> Iterator[dict[str, Any]]:
    """Greedy ``<where>`` continuations for a whole split, one record per sample."""
    tokenizer = collator.tokenizer
    close_id = int(tokenizer(WHERE_CLOSE, add_special_tokens=False)["input_ids"][0])
    open_id = int(tokenizer(WHERE_OPEN, add_special_tokens=False)["input_ids"][0])
    eos_id = tokenizer.eos_token_id
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
                                  eos_token_id=eos_id)
        for s, ids in zip(samples, gens):
            ctx = generated_context(
                s.sample_id, ids, close_id, max_tokens=max_context_tokens, eos_id=eos_id
            )
            yield {
                "schema_version": SCHEMA_GENCTX,
                "sample_id": s.sample_id,
                "split": dataset.split,
                "build": s.meta.get("build"),
                "render_mode": s.meta.get("render_mode"),
                "winner_confidence": s.meta.get("winner_confidence"),
                "checkpoint": checkpoint,
                "generated_ids": list(ids),
                "generated_text": tokenizer.decode(ids, skip_special_tokens=False),
                "n_generated_tokens": len(ids),
                "where_ids": list(ctx.token_ids),
                "where_text": tokenizer.decode(ctx.token_ids, skip_special_tokens=False),
                "format_failure": ctx.format_failure,
                "truncated": ctx.truncated,
                "stop_reason": ctx.stop_reason,
                "starts_with_where_open": bool(ids and ids[0] == open_id),
                "gen": {
                    "max_new_tokens": max_new_tokens,
                    "max_context_tokens": max_context_tokens,
                    "do_sample": False, "close_id": close_id, "open_id": open_id,
                    "eos_id": eos_id,
                },
            }
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


def summarise_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    n = len(records)
    if not n:
        return {"n": 0}
    reasons: dict[str, int] = {}
    for r in records:
        reasons[r["stop_reason"]] = reasons.get(r["stop_reason"], 0) + 1
    lens = sorted(len(r["where_ids"]) for r in records)
    return {
        "n": n,
        "format_failure_rate": sum(r["format_failure"] for r in records) / n,
        "truncation_rate": sum(r["truncated"] for r in records) / n,
        "starts_with_where_open_rate": sum(r["starts_with_where_open"] for r in records) / n,
        "stop_reasons": dict(sorted(reasons.items())),
        "where_tokens": {
            "min": lens[0], "p50": lens[n // 2],
            "p95": lens[min(n - 1, int(0.95 * (n - 1)))], "max": lens[-1],
        },
    }
