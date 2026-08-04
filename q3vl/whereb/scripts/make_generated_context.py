#!/usr/bin/env python
"""FULL-SCALE JOB -- NOT RUN YET.  Cache the Base SFT model's own ``<where>`` spans.

Protocol 5.4: half of every Where-B training batch is conditioned on the
*generated* ``<where>`` context, and evaluation reports it as its own board.
This job produces that context once, for a whole split, as indexed tar shards
(protocol 2.3).

    for each sample in <split>:
        prompt   <- the SFT collator's own prompt ids (identical tokenisation)
        ids      <- greedy generate, max_new_tokens=128, do_sample=False
        span     <- up to and including the first </where>, else the first 96
                    tokens with format_failure=True   (NEVER the GT span)

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

import argparse
import json
import time
from pathlib import Path

import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.modeling import load_model, load_processor
from q3vl.whereb.config import (
    GENCTX_DIR,
    GEN_MAX_NEW_TOKENS,
    MODEL_DIR,
    REPORT_DIR,
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=str(GENCTX_DIR))
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    args = ap.parse_args()

    out_root = Path(args.out_root) / args.split
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
        "split": args.split, "checkpoint": args.checkpoint, "env": _env(),
        "special_token_ids": special_ids, "vlm": vlm.facts(),
        "dataset": ds_info,
        "n_samples": len(dataset), "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "max_context_tokens": WHERE_CONTEXT_MAX_TOKENS,
        "out_root": str(out_root),
    }
    print(json.dumps(setup, indent=2), flush=True)

    t0 = time.time()
    records = list(generate_records(
        vlm, collator, dataset, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens, checkpoint=args.checkpoint,
        limit=args.limit,
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
    (rd / f"genctx_{args.split}.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "setup"},
                     indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
