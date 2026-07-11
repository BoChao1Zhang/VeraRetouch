#!/usr/bin/env python3
"""Charm scoring with multiprocess CPU preprocessing (tokenizer is the bottleneck).

Workers run Charm_Tokenizer.preprocess in parallel; the single main process owns
the GPU model and writes predictions incrementally. Same output schema as
run_charm_predictions.py.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import tempfile
from pathlib import Path

from run_charm_predictions import (  # noqa: F401
    load_rows,
    para_1_5_to_0_100,
    patch_charm_tokenizer_for_py313,
    score_to_0_100,
    seen_ids,
)

_TOK = None
_TMPDIR = None


def _init_worker(patch_selection: str, training_dataset: str, backbone_name: str, tmpdir: str) -> None:
    global _TOK, _TMPDIR
    import torch
    torch.set_num_threads(1)
    _TMPDIR = tmpdir
    from Charm_tokenizer.ImageProcessor import Charm_Tokenizer
    patch_charm_tokenizer_for_py313(Charm_Tokenizer)
    _TOK = Charm_Tokenizer(
        patch_selection=patch_selection,
        training_dataset=training_dataset,
        backbone=backbone_name,
        without_pad_or_dropping=True,
    )


def _preprocess(row: dict) -> tuple[str, str | None, str | None]:
    """Preprocess on CPU, save tensors to disk, return (id, path, err).

    Returning a path (not tensors) avoids large-payload pipe serialization,
    which is what deadlocked the pool previously.
    """
    import torch
    benchmark_id = str(row["benchmark_id"])
    try:
        tokens, pos_embed, mask_token = _TOK.preprocess(row["image_path"])
        fd, path = tempfile.mkstemp(suffix=".pt", dir=_TMPDIR)
        os.close(fd)
        torch.save((tokens, pos_embed, mask_token), path)
        return benchmark_id, path, None
    except Exception as exc:  # noqa: BLE001
        return benchmark_id, None, repr(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--patch-selection", default="frequency")
    parser.add_argument("--training-dataset", default="ava")
    parser.add_argument("--backbone", default="facebook/dinov2-large")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    import Charm_tokenizer.Backbone as charm_backbone

    checkpoint = str(args.checkpoint.expanduser())
    charm_backbone.hf_hub_download = lambda repo_id, filename: checkpoint
    scorer = charm_backbone.backbone(training_dataset=args.training_dataset, device=args.device)
    scorer.model = scorer.model.to(args.device).eval()

    rows = load_rows(args.input.expanduser(), args.limit)
    output = args.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    done = seen_ids(output) if args.resume else set()
    rows = [r for r in rows if str(r["benchmark_id"]) not in done]
    mode = "a" if args.resume else "w"

    base = {
        "model": f"Charm-{args.training_dataset}",
        "patch_selection": args.patch_selection,
        "training_dataset": args.training_dataset,
        "backbone": args.backbone,
    }
    ctx = mp.get_context("spawn")
    tmpdir = tempfile.mkdtemp(prefix="charm_prep_", dir=str(output.parent))
    n = 0
    with output.open(mode, encoding="utf-8") as out, ctx.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(args.patch_selection, args.training_dataset, args.backbone, tmpdir),
    ) as pool:
        for benchmark_id, path, err in pool.imap_unordered(_preprocess, rows, chunksize=1):
            if err is not None:
                payload = {"benchmark_id": benchmark_id, "error": err, **base}
            else:
                tokens, pos_embed, mask_token = torch.load(path, weights_only=False)
                os.remove(path)
                try:
                    with torch.inference_mode():
                        prediction = scorer.model(
                            tokens.unsqueeze(0).to(args.device),
                            pos_embed.unsqueeze(0).to(args.device),
                            mask_token.unsqueeze(0).to(args.device),
                        )
                        raw = float(scorer.mean_score(prediction)[0])
                    payload = {
                        "benchmark_id": benchmark_id,
                        "pred_score_0_100": score_to_0_100(raw, args.training_dataset),
                        "raw_score": raw,
                        **base,
                    }
                except Exception as exc:  # noqa: BLE001
                    torch.cuda.empty_cache()
                    payload = {"benchmark_id": benchmark_id, "error": repr(exc), **base}
            out.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            n += 1
            if n % 50 == 0:
                print(f"{n}/{len(rows)}", flush=True)
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass
    print(f"done {n}/{len(rows)} -> {output}")


if __name__ == "__main__":
    main()
