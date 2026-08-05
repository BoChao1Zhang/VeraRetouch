#!/usr/bin/env python
"""Why does left-padded batch greedy disagree with B=1 greedy? (genctx blocker)

The checkpoint verification measured 1/4 agreement between batched and B=1
greedy decoding on checkpoint-4976, which blocks the B=64 genctx production job.
Two hypotheses have to be told apart, and they call for opposite responses:

  (a) STRUCTURAL -- Qwen3-VL's mrope positions are wrong under left padding, so
      a padded row is decoded at the wrong positions.  Divergence would then be
      early, systematic, and a function of how much padding the row carries.
      Response: fix positions / bucket by length / patch upstream.

  (b) NUMERICAL  -- bf16 batched GEMMs reduce in a different order at different
      batch shapes, perturbing logits by ~1e-2.  Any near-tie between top-1 and
      top-2 then flips, and greedy decoding amplifies one flip into a fully
      different tail.  Divergence would be late, random, and independent of
      padding.  Response: decide whether "genwhere/2 = greedy under this batch
      config" is an acceptable definition.

Five tests, in increasing order of what they can rule out:

  T1 self-consistency   the same sample duplicated across the batch -- identical
                        input, identical padding, so any row-to-row difference is
                        a real bug and nothing else
  T2 prefill logits     last-prompt-position logits, B=1 vs batched: measures the
                        perturbation directly, before decoding can amplify it
  T3 padding-controlled the same sample placed once as the LONGEST row (zero
                        padding) and once as the SHORTEST (max padding).  This is
                        the discriminator: if only the padded placement diverges,
                        it is (a); if both diverge equally, it is (b)
  T4 full greedy        B=1 vs B=4/B=8, first divergence index + the top1-top2
                        gap at that step, read from generate(output_scores=True)
  T5 length bucketing   how much padding survives if batches are formed from
                        equal-length prompts, and whether agreement recovers

Runs on one GPU, read-only, no training state touched.
"""

from __future__ import annotations

# sqlite3 before torch: campaign bug R6 (see where_b/NOTES.md section 10)
import sqlite3  # noqa: F401

import argparse
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.modeling import load_model, load_processor
from q3vl.whereb.config import MODEL_DIR, SFT_CHECKPOINT
from q3vl.whereb.data import open_dataset


# ---------------------------------------------------------------- plumbing --
def build_prompts(collator, samples) -> list[list[int]]:
    out = []
    for s in samples:
        from q3vl.whereb.data import _PromptShim

        enc = collator.encode_one(_PromptShim(s))
        out.append(enc["input_ids"][: enc["n_prompt_tokens"]])
    return out


def pack(prompts: list[list[int]], pad_id: int, device: str):
    """Left-pad, exactly as both production paths do."""
    width = max(len(p) for p in prompts)
    ids, attn = [], []
    for p in prompts:
        k = width - len(p)
        ids.append([pad_id] * k + list(p))
        attn.append([0] * k + [1] * len(p))
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(attn, dtype=torch.long, device=device), width)


def image_inputs(processor, samples, device, dtype):
    img = processor.image_processor(images=[s.image for s in samples],
                                    do_resize=False, return_tensors="pt")
    return (img["pixel_values"].to(device, dtype), img["image_grid_thw"].to(device))


@torch.no_grad()
def prefill_logits(model, processor, collator, samples, device) -> torch.Tensor:
    """Logits at each row's LAST REAL prompt position -> (B, vocab), float32."""
    prompts = build_prompts(collator, samples)
    ids, attn, width = pack(prompts, collator.pad_token_id, device)
    pv, grid = image_inputs(processor, samples, device, model.dtype)
    out = model(input_ids=ids, attention_mask=attn, pixel_values=pv,
                image_grid_thw=grid, use_cache=False)
    return out.logits[:, -1, :].float()          # left padding -> last col is real


@torch.no_grad()
def greedy(model, processor, collator, samples, device, max_new_tokens, eos_id,
           want_scores=False):
    prompts = build_prompts(collator, samples)
    ids, attn, width = pack(prompts, collator.pad_token_id, device)
    pv, grid = image_inputs(processor, samples, device, model.dtype)
    gen = model.generate(
        input_ids=ids, attention_mask=attn, pixel_values=pv, image_grid_thw=grid,
        max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
        pad_token_id=collator.pad_token_id, eos_token_id=eos_id,
        return_dict_in_generate=True, output_scores=want_scores,
    )
    # Trim at the first terminator BEFORE comparing.  `generate` runs until the
    # LAST row in the batch finishes, so an early-finishing row carries pad/eos
    # filler after its real output; diffing untrimmed rows measures the filler,
    # not the generation.  This is also exactly what build_record() stores.
    raw = [[int(t) for t in row[width:].tolist()] for row in gen.sequences]
    seqs = []
    for r in raw:
        cut = next((j for j, t in enumerate(r) if t == eos_id), None)
        seqs.append(r[:cut] if cut is not None else r)
    scores = None
    if want_scores:
        scores = [s.float().cpu() for s in gen.scores]      # list[step] -> (B, V)
    return seqs, scores, [len(p) for p in prompts]


def first_divergence(a: list[int], b: list[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def gap_stats(row: torch.Tensor) -> dict[str, float]:
    top2 = torch.topk(row, 2)
    return {"top1": float(top2.values[0]), "top2": float(top2.values[1]),
            "gap": float(top2.values[0] - top2.values[1]),
            "argmax": int(top2.indices[0])}


# ------------------------------------------------------------------ tests --
def t1_self_consistency(model, processor, collator, sample, device, args, eos_id):
    """Same sample x N.  Identical inputs and identical padding -> must agree."""
    n = args.batch
    seqs, _, _ = greedy(model, processor, collator, [sample] * n, device,
                        args.max_new_tokens, eos_id)
    same = sum(1 for s in seqs[1:] if s == seqs[0])
    return {"batch": n, "rows_identical_to_row0": same, "n_rows": n,
            "all_identical": same == n - 1,
            "first_divergences": [first_divergence(seqs[0], s) for s in seqs[1:]]}


def t2_prefill(model, processor, collator, samples, device, args):
    single = torch.cat([prefill_logits(model, processor, collator, [s], device)
                        for s in samples], dim=0)
    batched = prefill_logits(model, processor, collator, samples, device)
    prompts = build_prompts(collator, samples)
    width = max(len(p) for p in prompts)
    rows = []
    for i, s in enumerate(samples):
        d = (batched[i] - single[i]).abs()
        gs, gb = gap_stats(single[i]), gap_stats(batched[i])
        rows.append({
            "sample_id": s.sample_id,
            "prompt_len": len(prompts[i]), "pad": width - len(prompts[i]),
            "max_abs_logit_delta": float(d.max()),
            "mean_abs_logit_delta": float(d.mean()),
            "single_top1_top2_gap": gs["gap"],
            "batched_top1_top2_gap": gb["gap"],
            "argmax_flipped": gs["argmax"] != gb["argmax"],
        })
    return rows


def t3_padding_controlled(model, processor, collator, pool, device, args, eos_id):
    """The discriminator: same sample, zero padding vs maximum padding."""
    prompts = build_prompts(collator, pool)
    order = sorted(range(len(pool)), key=lambda i: len(prompts[i]))
    shortest, longest = order[0], order[-1]
    target = pool[shortest]
    spread = len(prompts[longest]) - len(prompts[shortest])

    ref, _, _ = greedy(model, processor, collator, [target], device,
                       args.max_new_tokens, eos_id)
    # placement A: target is the LONGEST row -> it gets zero padding
    companions_short = [pool[i] for i in order if len(prompts[i]) <= len(prompts[shortest])][:args.batch - 1]
    while len(companions_short) < args.batch - 1:
        companions_short.append(target)
    batch_a = [target] + companions_short
    seq_a, _, lens_a = greedy(model, processor, collator, batch_a, device,
                              args.max_new_tokens, eos_id)
    # placement B: target is the SHORTEST row -> it gets maximum padding
    companions_long = [pool[i] for i in reversed(order)][:args.batch - 1]
    batch_b = [target] + companions_long
    seq_b, _, lens_b = greedy(model, processor, collator, batch_b, device,
                              args.max_new_tokens, eos_id)
    return {
        "target": target.sample_id,
        "prompt_len_spread_in_pool": spread,
        "zero_padding_placement": {
            "target_pad": max(lens_a) - lens_a[0],
            "matches_b1": seq_a[0] == ref[0],
            "first_divergence": first_divergence(ref[0], seq_a[0]),
        },
        "max_padding_placement": {
            "target_pad": max(lens_b) - lens_b[0],
            "matches_b1": seq_b[0] == ref[0],
            "first_divergence": first_divergence(ref[0], seq_b[0]),
        },
    }


def t4_full_greedy(model, processor, collator, samples, device, args, eos_id):
    singles = []
    single_scores = []
    for s in samples:
        seq, sc, _ = greedy(model, processor, collator, [s], device,
                            args.max_new_tokens, eos_id, want_scores=True)
        singles.append(seq[0])
        single_scores.append(sc)
    out = {}
    for b in args.compare_batches:
        b = min(b, len(samples))
        seqs, scores, lens = greedy(model, processor, collator, samples[:b], device,
                                    args.max_new_tokens, eos_id, want_scores=True)
        width = max(lens)
        rows = []
        for i in range(b):
            d = first_divergence(singles[i], seqs[i])
            row = {"sample_id": samples[i].sample_id, "pad": width - lens[i],
                   "n_tokens_single": len(singles[i]), "n_tokens_batched": len(seqs[i]),
                   "identical": d is None, "first_divergence": d}
            if d is not None and d < len(single_scores[i]) and d < len(scores):
                gs = gap_stats(single_scores[i][d][0])
                gb = gap_stats(scores[d][i])
                delta = (scores[d][i] - single_scores[i][d][0]).abs()
                row.update({
                    "at_divergence": {
                        "single_top1_top2_gap": gs["gap"],
                        "batched_top1_top2_gap": gb["gap"],
                        "max_abs_logit_delta": float(delta.max()),
                        "single_pick": gs["argmax"], "batched_pick": gb["argmax"],
                    }
                })
            rows.append(row)
        out[f"B={b}"] = {
            "identical_rate": sum(r["identical"] for r in rows) / b,
            "rows": rows,
        }
    return out


def t5_bucketing(collator, pool, args) -> dict[str, Any]:
    """How much padding survives if batches are built from equal-length prompts?"""
    prompts = build_prompts(collator, pool)
    lens = [len(p) for p in prompts]

    def waste(order: list[int], b: int) -> float:
        tot = pad = 0
        for i in range(0, len(order), b):
            chunk = [lens[j] for j in order[i:i + b]]
            w = max(chunk)
            tot += w * len(chunk)
            pad += w * len(chunk) - sum(chunk)
        return pad / tot

    natural = list(range(len(pool)))
    bucketed = sorted(natural, key=lambda i: lens[i])
    return {
        "n": len(pool),
        "prompt_len": {"min": min(lens), "p50": int(statistics.median(lens)),
                       "max": max(lens), "distinct": len(set(lens))},
        "most_common_lengths": Counter(lens).most_common(5),
        "pad_fraction_natural_order": {f"B={b}": round(waste(natural, b), 4)
                                       for b in args.compare_batches},
        "pad_fraction_length_sorted": {f"B={b}": round(waste(bucketed, b), 4)
                                       for b in args.compare_batches},
    }


def t5b_bucketed_agreement(model, processor, collator, pool, device, args, eos_id):
    """Agreement when a batch is built from prompts of *identical* length."""
    prompts = build_prompts(collator, pool)
    by_len: dict[int, list] = {}
    for s, p in zip(pool, prompts):
        by_len.setdefault(len(p), []).append(s)
    group = max(by_len.values(), key=len)
    n = min(args.batch, len(group))
    if n < 2:
        return {"ran": False, "reason": "no equal-length group of size >= 2",
                "largest_group": len(group)}
    batch = group[:n]
    seqs, _, lens = greedy(model, processor, collator, batch, device,
                           args.max_new_tokens, eos_id)
    refs = [greedy(model, processor, collator, [s], device,
                   args.max_new_tokens, eos_id)[0][0] for s in batch]
    same = sum(1 for a, b in zip(seqs, refs) if a == b)
    return {"ran": True, "prompt_len": lens[0], "padding": max(lens) - min(lens),
            "batch": n, "identical": same, "identical_rate": same / n,
            "first_divergences": [first_divergence(r, s) for r, s in zip(refs, seqs)]}


# ------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(SFT_CHECKPOINT))
    ap.add_argument("--model-dir", default=str(MODEL_DIR))
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--compare-batches", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    processor, special_ids = load_processor(args.model_dir, 2048)
    collator = Sft2SegCollator(processor, max_length=2048, system_prompt=None)
    t0 = time.time()
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype=args.dtype).to(args.device).eval()
    load_s = time.time() - t0
    eos_id = processor.tokenizer.eos_token_id

    ds, ds_info = open_dataset(args.split, need_mask=False, limit=args.n * 6)
    pool = [ds[i] for i in range(min(len(ds), args.n * 6))]
    samples = pool[: args.n]

    report: dict[str, Any] = {
        "setup": {
            "checkpoint": args.checkpoint, "attn": args.attn, "dtype": args.dtype,
            "device": args.device, "n": args.n, "batch": args.batch,
            "compare_batches": args.compare_batches,
            "max_new_tokens": args.max_new_tokens, "seed": args.seed,
            "model_load_s": round(load_s, 1), "dataset": ds_info,
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        }
    }
    for name, fn in (
        ("T1_self_consistency",
         lambda: t1_self_consistency(model, processor, collator, samples[0], device=args.device,
                                     args=args, eos_id=eos_id)),
        ("T2_prefill_logits",
         lambda: t2_prefill(model, processor, collator, samples[: args.batch], args.device, args)),
        ("T3_padding_controlled",
         lambda: t3_padding_controlled(model, processor, collator, pool, args.device, args, eos_id)),
        ("T4_full_greedy",
         lambda: t4_full_greedy(model, processor, collator, samples, args.device, args, eos_id)),
        ("T5_bucketing_padding", lambda: t5_bucketing(collator, pool, args)),
        ("T5b_bucketed_agreement",
         lambda: t5b_bucketed_agreement(model, processor, collator, pool, args.device, args, eos_id)),
    ):
        t = time.time()
        print(f"--- {name} ---", flush=True)
        report[name] = fn()
        report[name if isinstance(report[name], dict) else name] = report[name]
        print(json.dumps({name: report[name]}, indent=1, default=str)[:4000], flush=True)
        print(f"    ({time.time() - t:.1f}s)", flush=True)

    out = Path(args.out) if args.out else Path(__file__).parent / "batch_consistency.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
