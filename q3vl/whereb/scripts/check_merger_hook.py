"""Verify :class:`MergerHook` against E1's published 400-sample cache.

The hook is new code on the one path everything else in PR-AMORT stands on, so
it is checked the only way that actually proves anything: run the *online*
forward on real samples and compare the captured tensor, element by element,
with ``merger_out`` in ``amort_cache_20260810`` -- which was produced by a
different route (``visual(...)[0]``) in a different job.

Agreement to fp16 round-off proves three things at once: the hook fires, it
captures the merger's output rather than some neighbouring tensor, and the
``(gh/2, gw/2)`` reshape is the correct de-serialisation (a wrong ordering
would still have the right shape and the right value histogram -- and would be
silently wrong forever).

Runs a handful of samples on whatever GPU has room; it does not queue.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="/home/bc/data/runs/where_b/amort_cache_20260810")
    ap.add_argument("--checkpoint",
                    default="/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976")
    ap.add_argument("--split", default="V_where")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    ap.add_argument("--attn", default="eager",
                    help="must match the cache producer (dump_amort_cache used eager)")
    args = ap.parse_args(argv)

    from transformers import AutoProcessor

    from q3vl.train.modeling import load_model
    from q3vl.whereb.data import open_dataset
    from q3vl.whereb.hiddens import EncodeItem, FrozenVLM

    cache = Path(args.cache)
    manifest = json.loads((cache / "manifest.json").read_text())
    cached_ids = {r["sample_id"]: r for r in manifest["samples"]}

    proc = AutoProcessor.from_pretrained(args.checkpoint)
    model = load_model(args.checkpoint, attn_implementation=args.attn,
                       dtype="bfloat16").to(args.device).eval()
    vlm = FrozenVLM(model, proc, device=args.device, want_merger=True)

    ds, info = open_dataset(args.split, need_mask=False)
    rows = ds.meta_rows()
    pick = [i for i, r in enumerate(rows)
            if r.get("render_mode") == "local" and ds.record(i)["sample_id"] in cached_ids]
    pick = pick[: args.n]
    if not pick:
        raise SystemExit("no overlap between the split and the cache")

    # The prompt must carry the image placeholder tokens or the LLM refuses the
    # forward ("image features and image tokens do not match"), so build it the
    # same way training does rather than hand-rolling one.
    from q3vl.train.collator import Sft2SegCollator

    from q3vl.whereb.data import _PromptShim

    collator = Sft2SegCollator(proc, max_length=2048, system_prompt=None)

    out_rows = []
    t0 = time.time()
    for i in pick:
        s = ds[i]
        enc = collator.encode_one(_PromptShim(s, instruction=s.instruction))
        n_p = enc["n_prompt_tokens"]
        res = vlm.encode([EncodeItem(sample_id=s.sample_id, image=s.image,
                                     prompt_ids=enc["input_ids"][:n_p],
                                     where_ids=())])[0]
        if res.f_merger is None:
            raise SystemExit("f_merger is None -- want_merger did not take effect")
        got = res.f_merger.reshape(-1, res.f_merger.shape[-1]).float().cpu().numpy()
        ref = np.load(cache / "cache" / f"{s.sample_id}.npz")["merger_out"].astype(np.float32)
        if got.shape != ref.shape:
            raise SystemExit(f"{s.sample_id}: shape {got.shape} != cached {ref.shape}")
        denom = float(np.abs(ref).max()) or 1.0
        amax = float(np.abs(got - ref).max())
        corr = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
        # A layout error shows up here, not in the max: a wrong de-serialisation
        # permutes rows, so per-ROW correlation collapses while the global value
        # histogram stays intact.
        gn = got / (np.linalg.norm(got, axis=1, keepdims=True) + 1e-9)
        rn = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-9)
        row_cos = float((gn * rn).sum(1).min())
        out_rows.append({
            "sample_id": s.sample_id, "shape": list(got.shape),
            "max_abs_diff": amax, "rel_max_abs_diff": amax / denom,
            "corr": corr, "min_row_cosine": row_cos,
            "median_abs_diff": float(np.median(np.abs(got - ref))),
            "grid16": [res.grid_h, res.grid_w],
        })
        print(json.dumps(out_rows[-1]), flush=True)

    worst = max(r["rel_max_abs_diff"] for r in out_rows)
    worst_corr = min(r["corr"] for r in out_rows)
    worst_row = min(r["min_row_cosine"] for r in out_rows)
    verdict = {
        "n": len(out_rows), "worst_rel_max_abs_diff": worst, "worst_corr": worst_corr,
        "worst_min_row_cosine": worst_row,
        # The criterion is the per-row cosine, not the max abs diff: the cache is
        # fp16 AND was produced in bf16 under a different attention kernel, so a
        # few large-norm "register" cells legitimately differ by O(1) in absolute
        # terms.  What must hold is that every row still IS the same vector --
        # that is what proves the ordering and the tensor identity.
        "pass": bool(worst_row > 0.999 and worst_corr > 0.998),
        "elapsed_s": round(time.time() - t0, 1),
        "rows": out_rows, "split": args.split, "cache": str(cache),
    }
    print(json.dumps({k: v for k, v in verdict.items() if k != "rows"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    return 0 if verdict["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
