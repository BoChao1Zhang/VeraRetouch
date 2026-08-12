#!/usr/bin/env python
"""A1 pilot gate: how much of the GT geometry code survives the text path.

Two stages, deliberately separable because only the second needs a GPU:

**stage 1 (CPU, no generation)** -- the model's own ``<where>`` spans are
already cached for every training sample (``GenContextStore``, 159,215 rows), so
the deployable parse can be scored without generating anything: decode the
cached ids, run the 21-d parser over them, and score the result against the
construction-side GT code.  This is the ``F1_A1`` reference line that D-10
requires every "we beat 82%" claim to be stated against, and it is the
denominator of the C-arm rule (§3.2: parsed capture >= 0.7 x GT capture => no C
arm).

**stage 2 (GPU)** -- the D-7 two-pass form: keep pass-1's free reasoning, then
ask the model to restate the geometry behind a forced ``<geom>`` prefix and
parse *that*.  The question it answers is narrow: does a canonical restatement
recover code the free-text parse drops?

A caveat that governs the whole board: **the GT code is not a superset of the
parsed code**.  ``.vrmeta.json`` carries ``slot_id`` (shape) and ``region``
(direction) and *no extent*, so the six extent slots are identically zero in the
GT vector.  Scoring them would report a perfect zero-vector match as skill and a
correct extent read as an error.  They are therefore excluded from the score and
reported separately as "not measurable against this GT".

Usage (stage 1, CPU):
    python -m q3vl.whereb.scripts.run_a1_pilot --stage 1 --n 500 \
        --out experiments/.../where_b/a1_pilot_20260812
"""

from __future__ import annotations

import sqlite3  # noqa: F401  (import order: sqlite3 before torch -- campaign bug R6)

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np


def _macro_f1(pred: np.ndarray, gt: np.ndarray, cols: list[int]) -> dict[str, Any]:
    """Per-slot F1 at threshold 0.5, macro-averaged over the given columns.

    Slots the GT never lights are dropped from the macro average rather than
    scored as perfect: a slot that is always 0 in the reference is not evidence
    about the extractor, and averaging it in would inflate the score toward 1
    exactly where the code carries no information.
    """
    per_slot, skipped = {}, []
    for c in cols:
        p, g = pred[:, c] >= 0.5, gt[:, c] >= 0.5
        if not g.any():
            skipped.append(c)
            continue
        tp = float((p & g).sum())
        fp = float((p & ~g).sum())
        fn = float((~p & g).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1e-9)
        per_slot[c] = {"f1": f1, "support": int(g.sum()),
                       "precision": tp / max(tp + fp, 1e-9),
                       "recall": tp / max(tp + fn, 1e-9)}
    macro = (float(np.mean([v["f1"] for v in per_slot.values()]))
             if per_slot else None)
    return {"macro_f1": macro, "n_slots_scored": len(per_slot),
            "slots_unsupported_in_gt": skipped, "per_slot": per_slot}


def stage1(args) -> dict[str, Any]:
    from transformers import AutoProcessor

    from q3vl.whereb.amort.geomparse import (GEOM_SLOTS, geom_features,
                                             geom_features_from_vrmeta)
    from q3vl.whereb.config import GENCTX_DIR
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.stores import GenContextStore
    from q3vl.where.maskdata import MaskResolver

    names = [n for n, _ in GEOM_SLOTS]
    grp = {"shape": [i for i, n in enumerate(names) if n.startswith("shape_")],
           "dir": [i for i, n in enumerate(names) if n.startswith("dir_")],
           "ext": [i for i, n in enumerate(names) if n.startswith("ext_")]}

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    tok = proc.tokenizer
    ds, info = open_dataset(args.split, need_mask=True, exclude_low=True)
    rows = ds.meta_rows()
    idx = [i for i, r in enumerate(rows) if r.get("render_mode") == "local"]
    print(f"{args.split}: {len(idx)} local samples", flush=True)

    store = GenContextStore(Path(GENCTX_DIR) / args.split)
    res = MaskResolver(verify="none", suffix=".vrmeta.json")

    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(idx), size=min(args.n, len(idx)), replace=False)

    P, G, per_sample = [], [], []
    n_no_genctx = n_no_vrmeta = n_empty_text = 0
    t0 = time.time()
    for j, jj in enumerate(pick):
        i = idx[int(jj)]
        sid = ds.record(i).get("sample_id") or rows[i].get("sample_id")
        try:
            ids = store.where_ids(sid)
        except Exception:
            n_no_genctx += 1
            continue
        text = tok.decode(ids, skip_special_tokens=True)
        if not text.strip():
            n_empty_text += 1
        try:
            vm = json.loads(res.read_bytes(res.resolve(ds.record(i))).decode())
            g = geom_features_from_vrmeta(vm.get("slot_id"), vm.get("region"))
        except Exception:
            n_no_vrmeta += 1
            continue
        if g is None:
            n_no_vrmeta += 1
            continue
        p = geom_features(text)
        P.append(p)
        G.append(g)
        per_sample.append({
            "sample_id": sid,
            "n_active_parsed": int((p >= 0.5).sum()),
            "n_active_gt": int((g >= 0.5).sum()),
            "family": rows[i].get("mask_type"),
            "text_len": len(text),
            "abstain": bool((p >= 0.5).sum() == 0),
        })
        if args.progress and j % 100 == 0:
            print(f"  {j}/{len(pick)}  ({time.time()-t0:.0f}s)", flush=True)

    P, G = np.asarray(P), np.asarray(G)
    out: dict[str, Any] = {
        "stage": 1,
        "what": "parsed code from the model's OWN cached <where> vs GT vrmeta code",
        "n_requested": int(args.n), "n_scored": int(len(P)),
        "n_missing_genctx": n_no_genctx, "n_missing_vrmeta": n_no_vrmeta,
        "n_empty_text": n_empty_text,
        "split": args.split, "seed": args.seed,
    }
    for name, cols in grp.items():
        out[f"capture_{name}"] = _macro_f1(P, G, cols)
    scorable = grp["shape"] + grp["dir"]
    out["capture_overall_shape_dir"] = _macro_f1(P, G, scorable)
    out["extent_note"] = (
        "extent slots are identically zero in the GT (.vrmeta carries slot_id "
        "and region only), so capture_ext is NOT a measure of extent quality; "
        "the parser's extent output is reported as an activation rate instead")
    out["extent_activation_rate_parsed"] = float(
        (P[:, grp["ext"]] >= 0.5).any(axis=1).mean()) if len(P) else None
    out["abstain_rate"] = float(np.mean([r["abstain"] for r in per_sample])) \
        if per_sample else None
    out["mean_active_parsed"] = float(np.mean([r["n_active_parsed"]
                                               for r in per_sample])) \
        if per_sample else None
    out["mean_active_gt"] = float(np.mean([r["n_active_gt"]
                                           for r in per_sample])) \
        if per_sample else None
    return out, per_sample


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1, choices=[1])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=20260812)
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    board, per_sample = stage1(args)
    (out_dir / "a1_pilot_stage1.json").write_text(
        json.dumps(board, indent=2, default=str), encoding="utf-8")
    with (out_dir / "a1_pilot_stage1_per_sample.jsonl").open("w") as fh:
        for r in per_sample:
            fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: board[k] for k in
                      ("n_scored", "abstain_rate", "mean_active_parsed",
                       "mean_active_gt")}, indent=2))
    for k in ("capture_shape", "capture_dir", "capture_overall_shape_dir"):
        print(f"{k}: macro_f1={board[k]['macro_f1']} "
              f"(slots scored {board[k]['n_slots_scored']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
