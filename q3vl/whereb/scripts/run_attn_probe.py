"""PR-ATT1-E1 exporter: one eager prefill per (sample, context arm) -> per-head fields.

Two context arms, both built at **run time** through the sanctioned
:mod:`q3vl.whereb.context` constructors:

* ``gt``       -- teacher-forced ``<where>`` from the sample's own record;
* ``shuffled`` -- the negative control, instruction **and** ``<where>`` swapped
  together for a partner drawn inside the ``(source_image_id, render_mode)``
  group (protocol 5.4 / review blocker B3).

The card described these as two on-disk档 under ``where_b-20260805/genwhere/``.
They are not there: that store holds the **generated** span (``do_sample:false``,
empty forced prefix), and no shuffled / fixed-phrase档 was ever published.  The
GT span lives in the sft2seg record's ``where`` field and is encoded by
``context.gt_context``; the shuffled partner is assigned by ``ShuffleIndex``.
Using those is not a substitution -- it is the same code path the training arms
use, so the probe cannot drift from the thing it is probing.

Only the **local** subset is forwarded.  V_where's 496 global samples have no GT
mask (``mask_target_hi`` returns all-ones for them by construction) and so cannot
enter a spatial criterion; forwarding them would double the wall clock to
produce fields nothing is allowed to score.  The fit/OOF split is still computed
over all 896 rows, so the fold definition is the one the card specifies.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=Path(__file__).resolve().parents[3], timeout=10,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="gt,shuffled")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap on LOCAL samples exported (smoke / P-W1 use this)")
    ap.add_argument("--stratify-resolution", action="store_true",
                    help="P-W1: pick --limit samples spread over distinct grid shapes")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seed", type=int, default=20260810)
    args = ap.parse_args(argv)

    t_start = time.time()
    out = Path(args.out)
    (out / "fields").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    from transformers import AutoProcessor

    from q3vl.train.collator import Sft2SegCollator
    from q3vl.train.modeling import load_model
    from q3vl.whereb.attnread import (
        AttentionTap, assert_model_facts, locate_image_columns, locate_query_pools,
        merged_grid, POOL_NAMES,
    )
    from q3vl.whereb.attnprobe import fit_oof_split, gt_merged_grid
    from q3vl.whereb.config import WHERE_A_MASKVIEW_DIR
    from q3vl.whereb.context import (ShuffleIndex, fixed_phrase_context, gt_context,
                                     shuffled_context)
    from q3vl.whereb.data import open_dataset, _PromptShim
    from q3vl.whereb.stores import MaskViewStore

    print(f"[{time.strftime('%H:%M:%S')}] loading processor + model (eager) ...", flush=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    tokenizer = processor.tokenizer
    # RED LINE: eager only.  assert_model_facts refuses anything else, and the
    # per-layer hook raises if any layer hands back attn_weights=None.
    model = load_model(args.checkpoint, attn_implementation="eager", dtype=args.dtype)
    model = model.to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    facts = assert_model_facts(model)
    print(f"  model facts: {facts}", flush=True)

    collator = Sft2SegCollator(processor)
    ds, ds_info = open_dataset(args.split, need_mask=False)
    meta_rows = ds.meta_rows()
    print(f"  {args.split}: {len(ds)} samples ({ds_info.get('mask_source')})", flush=True)

    # fold assignment over ALL rows, grouped by source_image_id
    shuffle_rows = ds.shuffle_records()
    by_id = {r["sample_id"]: r for r in shuffle_rows}
    for r in meta_rows:
        r["source_image_id"] = by_id.get(r["sample_id"], {}).get("source_image_id")
    folds = fit_oof_split(meta_rows, seed="verasplit-v1")

    shuffle_index = ShuffleIndex(shuffle_rows, seed=args.seed)
    maskviews = MaskViewStore(Path(WHERE_A_MASKVIEW_DIR) / args.split)

    local_idx = [i for i, r in enumerate(meta_rows) if r.get("render_mode") == "local"]
    print(f"  local subset: {len(local_idx)}", flush=True)

    if args.limit is not None:
        if args.stratify_resolution:
            # P-W1 asks for a resolution-stratified draw.  Group by the merged
            # grid shape and round-robin the groups so every distinct shape is
            # represented before any shape is sampled twice.
            buckets: dict[tuple[int, int], list[int]] = {}
            for i in local_idx:
                rec = ds.record(i)
                img = rec.get("image") or {}
                key = merged_grid(int(img["out_h"]), int(img["out_w"]))
                buckets.setdefault(key, []).append(i)
            print(f"  grid shapes present: "
                  f"{sorted((k, len(v)) for k, v in buckets.items())}", flush=True)
            picked: list[int] = []
            keys = sorted(buckets)
            r = 0
            while len(picked) < args.limit and any(len(buckets[k]) > r for k in keys):
                for k in keys:
                    if len(buckets[k]) > r and len(picked) < args.limit:
                        picked.append(buckets[k][r])
                r += 1
            local_idx = picked
        else:
            local_idx = local_idx[: args.limit]
        print(f"  limited to {len(local_idx)} samples", flush=True)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    tap = AttentionTap(model.model.language_model, store_profile=True)

    rows_path = out / "samples.jsonl"
    rows_fh = rows_path.open("w", encoding="utf-8")
    n_done = 0
    n_skipped = 0
    t0 = time.time()

    with tap.attached():
        for n_i, i in enumerate(local_idx):
            sample = ds[i]
            sid = sample.sample_id
            gh, gw = merged_grid(sample.geometry.out_h, sample.geometry.out_w)

            # GT once per sample: published low view (out/16) -> merged (out/32)
            try:
                mlow = maskviews.mask_low(sid)
                gt = gt_merged_grid(mlow, gh, gw)
            except Exception as exc:
                n_skipped += 1
                print(f"  !! {sid}: no GT mask ({type(exc).__name__}: {exc})", flush=True)
                continue
            np.save(out / "gt" / f"{sid}.npy", gt.numpy().astype(np.float32))

            for arm in arms:
                if arm == "gt":
                    ctx = gt_context(tokenizer, sid, sample.where_text)
                elif arm == "shuffled":
                    partner = shuffle_index.partner_of(sid)
                    if partner is None:
                        print(f"  !! {sid}: no shuffle partner, arm skipped", flush=True)
                        continue
                    prow = shuffle_index.by_id[partner]
                    ctx = shuffled_context(tokenizer, partner, prow["where"],
                                           prow["instruction"])
                elif arm == "fixed_phrase":
                    # PROPOSAL 2.2 step 3 / P-W3 P4: the common-mode arm.  Both the
                    # instruction and the <where> span become one constant phrase
                    # ("the main subject") for every sample in the arm -- the red
                    # line's own worked example of a string with zero per-sample
                    # information.  This is a DIFFERENT context from `shuffled`:
                    # shuffled swaps in another real instruction, this removes
                    # reference entirely.
                    ctx = fixed_phrase_context(tokenizer, sid)
                else:
                    raise ValueError(f"unknown arm {arm!r}")

                enc = collator.encode_one(_PromptShim(sample, instruction=ctx.instruction))
                n_p = enc["n_prompt_tokens"]
                input_ids = list(enc["input_ids"][:n_p]) + list(ctx.token_ids)

                cols = locate_image_columns(input_ids)
                if cols.size != gh * gw:
                    raise RuntimeError(
                        f"{sid}: {cols.size} image tokens but merged grid says {gh*gw}"
                    )
                pools = locate_query_pools(input_ids, n_p, tokenizer=tokenizer)
                union = pools.union()
                tap.set_selection(union, cols, pools.profile_rows)

                img = processor.image_processor(
                    images=[sample.image], do_resize=False, return_tensors="pt"
                )
                ids_t = torch.tensor([input_ids], dtype=torch.long, device=args.device)
                attn_t = torch.ones_like(ids_t)
                dtype = next(model.parameters()).dtype
                with torch.no_grad():
                    model(
                        input_ids=ids_t,
                        attention_mask=attn_t,
                        pixel_values=img["pixel_values"].to(args.device, dtype),
                        image_grid_thw=img["image_grid_thw"].to(args.device),
                    )
                stacked, profile = tap.stack()      # (L,H,n_rows,n_img), (L,H,n_img)

                pos = {int(r): j for j, r in enumerate(union)}
                arrays: dict[str, np.ndarray] = {}
                for name in POOL_NAMES:
                    rows = pools.rows[name]
                    if rows.size == 0:
                        continue
                    sel = torch.tensor([pos[int(r)] for r in rows], dtype=torch.long)
                    fld = stacked.index_select(2, sel).mean(dim=2)     # (L,H,n_img)
                    arrays[f"field_{name}"] = fld.numpy().astype(np.float16)
                arrays["col_profile"] = profile.numpy().astype(np.float16)

                np.savez(out / "fields" / f"{sid}__{arm}.npz", **arrays)

                prof_mean = profile.mean(dim=(0, 1)).numpy().astype(np.float64)
                rows_fh.write(json.dumps({
                    "sample_id": sid, "arm": arm, "fold": folds.get(sid),
                    "grid_h": gh, "grid_w": gw, "n_img": int(gh * gw),
                    "seq_len": len(input_ids), "n_prompt_tokens": n_p,
                    "render_mode": sample.meta.get("render_mode"),
                    "build": sample.meta.get("build"),
                    "winner_confidence": sample.meta.get("winner_confidence"),
                    "source_image_id": by_id.get(sid, {}).get("source_image_id"),
                    "region": sample.meta.get("region"),
                    "context_mode": ctx.mode,
                    "n_where_tokens": len(ctx.token_ids),
                    "pools": pools.to_dict(),
                    "profile_stats": {
                        "min": float(prof_mean.min()), "max": float(prof_mean.max()),
                        "median": float(np.median(prof_mean)),
                        "mean": float(prof_mean.mean()),
                    },
                }) + "\n")
                rows_fh.flush()
                n_done += 1

            if (n_i + 1) % 25 == 0 or n_i == 0:
                el = time.time() - t0
                rate = (n_i + 1) / el
                eta = (len(local_idx) - n_i - 1) / rate if rate > 0 else 0
                print(f"  [{n_i+1}/{len(local_idx)}] {n_done} forwards, "
                      f"{el:.0f}s elapsed, ETA {eta/60:.1f} min", flush=True)

    rows_fh.close()

    setup = {
        "experiment": "PR-ATT1-E1",
        "checkpoint": args.checkpoint,
        "split": args.split,
        "arms": arms,
        "seed": args.seed,
        "device": args.device,
        "dtype": args.dtype,
        "n_local_requested": len(local_idx),
        "n_forwards": n_done,
        "n_skipped_no_gt": n_skipped,
        "model_facts": facts,
        "dataset_info": {k: v for k, v in ds_info.items() if k != "maskview_facts"},
        "shuffle_coverage": shuffle_index.coverage(),
        "fold_counts": {
            f: sum(1 for r in meta_rows if folds.get(r["sample_id"]) == f)
            for f in ("fit", "oof")
        },
        "fold_counts_local": {
            f: sum(1 for r in meta_rows
                   if folds.get(r["sample_id"]) == f and r.get("render_mode") == "local")
            for f in ("fit", "oof")
        },
        "git_commit": _git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "hostname": platform.node(),
        "cuda_device_name": (torch.cuda.get_device_name(0)
                             if torch.cuda.is_available() else None),
        "peak_gpu_mem_gib": (torch.cuda.max_memory_allocated() / 1024 ** 3
                             if torch.cuda.is_available() else None),
        "argv": sys.argv,
        "wall_seconds": time.time() - t_start,
    }
    (out / "run_setup.json").write_text(json.dumps(setup, indent=2), encoding="utf-8")
    (out / "folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in setup.items() if k != "argv"}, indent=2))
    print(f"[{time.strftime('%H:%M:%S')}] done in {(time.time()-t_start)/60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
