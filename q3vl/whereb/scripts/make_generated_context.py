#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  Cache the Base SFT model's own reasoning spans.

Protocol 5.4: half of every Where-B training batch is conditioned on the
*generated* ``<where>`` context, and evaluation reports it as its own board.
Amendment A-4 extends the same 50/50 teacher/generated split to Stage-What's
``<color>`` context.  The Base SFT model emits both segments in one greedy
continuation, so this job caches both.  Output is indexed tar shards (2.3).

    two_segment (default):
        prompt <- the SFT collator's own prompt ids (identical tokenisation)
        ids    <- greedy generate, max_new_tokens=512, do_sample=False
        where  <- up to and including the first </where>, else the first 96
                  tokens with format_failure=True     (NEVER the GT span)
        color  <- from the first <color> after that, up to and including the
                  first </color>, else the first 384 tokens with
                  color_format_failure=True           (NEVER the GT span)

    forced_color (--forced-color-prefix, Stage-What controls C01/C02):
        <color> is forced as the assistant's first token, so the colour segment
        is generated with no <where> reasoning in front of it -- the strict
        no-where control of protocol 8.2.  where_ids is empty and
        where_suppressed=True (it is not a format failure: no where was asked
        for).  Publishes to a separate root so the two artefacts cannot mix.

Only token ids are cached; the hidden states are re-derived at training time by
the same ``FrozenVLM.encode`` the teacher context uses, which is what makes the
two contexts numerically comparable (see :mod:`q3vl.whereb.gencontext`).

Requires one GPU.  **Do not start while Base SFT holds both cards.**

Usage (D-20: rm -f the log first, then verify with ``ps -p <PID>``, never pgrep):
    rm -f /home/bc/data/runs/where_b/genctx/train.log
    nohup python -m q3vl.whereb.scripts.make_generated_context \
        --split train --checkpoint /home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976 \
        > /home/bc/data/runs/where_b/genctx/train.log 2>&1 &
"""

from __future__ import annotations

# --- environment guard: sqlite3 must be imported BEFORE torch ---------------
# Verified 2026-08-05 in the campaign env (/home/bc/envs/q3vl_sft):
#   import torch; import sqlite3  -> ImportError, libstdc++ CXXABI_1.3.15 not found
#   import sqlite3; import torch  -> fine
# torch loads a libstdc++ that shadows the one `_sqlite3`'s dependency chain
# (libicui18n) needs, so any process that touches torch first can never open a
# published shard afterwards.  This job reaches sqlite3 twice over --
# `q3vl.data.shardio` (every published store and every packer) and
# `q3vl.where.maskdata` (the live mask locator opens a build catalog) -- so it
# would die on its first store access without this line.  Importing it first
# costs nothing and inoculates the whole process.  Campaign-wide bug R6, found
# by WHAT-IMPL; the guard belongs in entry points only (a guard inside a library
# module makes that module unimportable in any torch-first process, which is
# strictly worse -- tried and reverted).
import sqlite3  # noqa: F401  (import order is the point)

import argparse
import json
import time
from pathlib import Path

import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.modeling import load_model, load_processor
from q3vl.whereb.config import (
    COLOR_CONTEXT_MAX_TOKENS,
    GENCTX_WRITE_DIR,
    GEN_MAX_NEW_TOKENS,
    MODEL_DIR,
    REPORT_DIR,
    SCHEMA_GENCTX,
    SFT_CHECKPOINT,
    WHERE_CONTEXT_MAX_TOKENS,
)
from q3vl.whereb.data import open_dataset
from q3vl.whereb.gencontext import generate_records, publish_generated, summarise_records
from q3vl.whereb.hiddens import FrozenVLM
from q3vl.whereb.preflight import _env
from q3vl.whereb.stores import GenContextStore


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=GEN_MAX_NEW_TOKENS)
    ap.add_argument("--color-max-tokens", type=int, default=COLOR_CONTEXT_MAX_TOKENS)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(GENCTX_WRITE_DIR))
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--forced-color-prefix", action="store_true",
                    help="Stage-What controls C01/C02: force <color> as the "
                         "assistant's first token, so the colour segment is "
                         "generated with no <where> reasoning in front of it")
    args = ap.parse_args()

    mode = "forced_color" if args.forced_color_prefix else "two_segment"
    # a separate root per mode: the two artefacts answer different questions and
    # must never end up in the same shard set
    leaf = args.split if mode == "two_segment" else f"{args.split}-forced_color"
    out_root = Path(args.out_root) / leaf
    if out_root.exists():
        raise SystemExit(
            f"{out_root} already exists.  Publication is atomic; move the old "
            "directory aside (do NOT delete it -- CLAUDE.md long-job discipline) "
            "before re-running."
        )

    processor, special_ids = load_processor(args.model_dir, 2048)
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype=args.dtype)
    model = model.to(args.device).eval()
    vlm = FrozenVLM(model, processor, device=args.device)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)

    # need_mask=False: this job never looks at a GT mask, and resolving a
    # `.cgt.png` member for each of 159,215 samples would be pure waste
    # (review blocker B2 -- the old call had neither a maskview store nor a
    # resolver and died on the first local sample).
    dataset, ds_info = open_dataset(args.split, need_mask=False, limit=args.limit)
    setup = {
        "split": args.split, "mode": mode, "checkpoint": args.checkpoint,
        "env": _env(),
        "special_token_ids": special_ids, "vlm": vlm.facts(),
        "dataset": ds_info,
        "n_samples": len(dataset), "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": WHERE_CONTEXT_MAX_TOKENS,   # v1 name = where
        "where_max_tokens": WHERE_CONTEXT_MAX_TOKENS,
        "color_max_tokens": args.color_max_tokens,
        "schema_version": SCHEMA_GENCTX,
        "out_root": str(out_root),
    }
    print(json.dumps(setup, indent=2), flush=True)

    t0 = time.time()
    records = list(generate_records(
        vlm, collator, dataset, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, checkpoint=args.checkpoint,
        limit=args.limit, mode=mode, color_max_tokens=args.color_max_tokens,
    ))
    gen_seconds = time.time() - t0

    manifest = publish_generated(records, out_root, args.split)
    store = GenContextStore(out_root)
    report = {
        "setup": setup,
        "generation_seconds": round(gen_seconds, 1),
        "samples_per_second": round(len(records) / max(gen_seconds, 1e-9), 3),
        "summary": summarise_records(records),
        "manifest": {k: manifest.get(k) for k in
                     ("status", "member_count", "sample_count", "shard_count")},
        "store": store.summary(),
        "peak_memory_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if torch.cuda.is_available() else None),
    }
    rd = Path(args.report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / f"genctx_{leaf}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "setup"},
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
