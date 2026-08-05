#!/usr/bin/env python
"""Does the batch-vs-B=1 tie-breaking actually damage the genwhere artifact?

`diag_batch_consistency.py` established the mechanism: at every divergence the
top-1/top-2 logits are 0 or one bf16 ulp (0.125) apart and the batching
perturbation is also ~1 ulp, so greedy decoding tosses a coin the two configs
call differently.  Padding is neither necessary (a pad=0 batch still diverges)
nor sufficient (a pad=225 row matched B=1 bit for bit), which rules out mrope.

Bit-identity with B=1 is therefore the wrong acceptance test.  What matters is
whether the *artifact* is fit for purpose:

  * Where-B consumes ONLY `where_ids`.  If divergences land in the colour tail,
    Where-B's input is bit-identical regardless of batch size.
  * Stage-What consumes `color_ids` (amendment A-4).
  * Both consume it as "the model's own reasoning", so what has to hold is the
    structural contract (tags closed, order right, no GT leakage), not equality
    with some other decoding configuration.

So this measures, per batch size, at the segment level:
  - where_ids / color_ids identical rate vs the B=1 reference
  - format-failure rates under each config (the actual quality gate)
  - token-F1 between the two configs' segment texts, for the ones that differ
"""

from __future__ import annotations

# sqlite3 before torch: campaign bug R6
import sqlite3  # noqa: F401

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.diagnostics import parse_two_segment
from q3vl.train.modeling import load_model, load_processor
from q3vl.whereb.config import MODEL_DIR, SFT_CHECKPOINT
from q3vl.whereb.data import _PromptShim, open_dataset
from q3vl.whereb.gencontext import SegmentIds, build_record
from q3vl.whereb.hiddens import EncodeItem, FrozenVLM


def token_f1(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = 0
    pool = list(tb)
    for t in ta:
        if t in pool:
            pool.remove(t)
            common += 1
    p, r = common / len(ta), common / len(tb)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def run(vlm, collator, samples, batch_size, max_new_tokens, tags, checkpoint):
    """Production path: FrozenVLM.generate_where + build_record, exactly."""
    recs = []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i: i + batch_size]
        items = []
        for s in chunk:
            enc = collator.encode_one(_PromptShim(s))
            items.append(EncodeItem(sample_id=s.sample_id, image=s.image,
                                    prompt_ids=enc["input_ids"][:enc["n_prompt_tokens"]]))
        gens = vlm.generate_where(items, max_new_tokens=max_new_tokens,
                                  eos_token_id=tags.eos)
        for s, ids in zip(chunk, gens):
            recs.append(build_record(s, ids, tags, collator.tokenizer,
                                     split="V_where", checkpoint=checkpoint,
                                     max_new_tokens=max_new_tokens))
    return recs


def compare(ref: list[dict], test: list[dict]) -> dict[str, Any]:
    n = len(ref)
    same_where = same_color = same_full = 0
    f1w, f1c = [], []
    div_in_where = div_in_color_only = 0
    for a, b in zip(ref, test):
        w = a["where_ids"] == b["where_ids"]
        c = a["color_ids"] == b["color_ids"]
        same_where += w
        same_color += c
        same_full += a["generated_ids"] == b["generated_ids"]
        if not w:
            div_in_where += 1
        elif not c:
            div_in_color_only += 1
        if not w:
            f1w.append(token_f1(a["where_text"], b["where_text"]))
        if not c:
            f1c.append(token_f1(a["color_text"], b["color_text"]))
    def struct(rs):
        ps = [parse_two_segment(r["generated_text"]) for r in rs]
        return {
            "tag_completeness": sum(p["tags_complete"] for p in ps) / len(ps),
            "order_accuracy": sum(p["order_ok"] for p in ps) / len(ps),
            "where_nonempty": sum(p["where_nonempty"] for p in ps) / len(ps),
            "color_nonempty": sum(p["color_nonempty"] for p in ps) / len(ps),
            "legacy_leak": sum(p["legacy_tag_leak"] for p in ps) / len(ps),
            "where_format_failure": sum(r["format_failure"] for r in rs) / len(rs),
            "color_format_failure": sum(r["color_format_failure"] for r in rs) / len(rs),
        }
    return {
        "n": n,
        "full_sequence_identical_rate": same_full / n,
        "where_ids_identical_rate": same_where / n,
        "color_ids_identical_rate": same_color / n,
        "n_diverged_inside_where": div_in_where,
        "n_diverged_in_color_only": div_in_color_only,
        "where_token_f1_when_differing": (
            {"n": len(f1w), "min": min(f1w), "p50": statistics.median(f1w)} if f1w else None),
        "color_token_f1_when_differing": (
            {"n": len(f1c), "min": min(f1c), "p50": statistics.median(f1c)} if f1c else None),
        "structure_reference": struct(ref),
        "structure_test": struct(test),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    processor, _ = load_processor(args.model_dir, 2048)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype=args.dtype).to(args.device).eval()
    vlm = FrozenVLM(model, processor, device=args.device)
    tags = SegmentIds(processor.tokenizer)

    ds, _ = open_dataset(args.split, need_mask=False, limit=args.n)
    samples = [ds[i] for i in range(min(args.n, len(ds)))]

    report: dict[str, Any] = {"setup": {
        "checkpoint": args.checkpoint, "n": len(samples), "batches": args.batches,
        "max_new_tokens": args.max_new_tokens, "attn": args.attn,
        "dtype": args.dtype, "split": args.split}}

    t = time.time()
    print("reference B=1 ...", flush=True)
    ref = run(vlm, collator, samples, 1, args.max_new_tokens, tags, args.checkpoint)
    report["reference_seconds"] = round(time.time() - t, 1)

    for b in args.batches:
        t = time.time()
        print(f"batched B={b} ...", flush=True)
        test = run(vlm, collator, samples, b, args.max_new_tokens, tags, args.checkpoint)
        c = compare(ref, test)
        c["seconds"] = round(time.time() - t, 1)
        c["speedup_vs_b1"] = round(report["reference_seconds"] / max(c["seconds"], 1e-9), 2)
        report[f"B={b}"] = c
        print(json.dumps({f"B={b}": {k: v for k, v in c.items()
                                     if k not in ("structure_reference", "structure_test")}},
                         indent=1), flush=True)

    out = Path(args.out) if args.out else Path(__file__).parent / "segment_impact.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
