"""S0-JOINT item 2: exercise the spec 8.3 generation + parse pipeline once.

Single GPU, no DeepSpeed, base (un-finetuned) Qwen3-VL-4B-Instruct. The point is
that the *pipeline* runs -- prompt slicing, image tensors, `model.generate`,
decode, `parse_two_segment`, `aggregate_generation_diagnostics`. Generation
quality is explicitly out of scope: the base model has never seen the four new
special tokens, so all structural rates are expected to be 0.

The real `Qwen3VLSFTTrainer.generation_diagnostics` function object is called
(bound to a stub carrying exactly the attributes it reads), so this exercises
the shipped code rather than a re-implementation.

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python .../check_generation_pipeline.py \
        --out <json> --n 3
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from q3vl.train.collator import Sft2SegCollator
from q3vl.train.dataset import Sft2SegDataset
from q3vl.train.diagnostics import parse_two_segment
from q3vl.train.modeling import load_processor, setup_model
from q3vl.train.shards import ShardIndex, ShardStore
from q3vl.train.trainer import Qwen3VLSFTTrainer

REPO = Path(__file__).resolve().parents[5]
PROD_CONFIG = REPO / "q3vl/train/configs/sft_base.yaml"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROD_CONFIG))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["training"]

    report: dict[str, object] = {
        "config": args.config,
        "production_gen_diag_samples": tcfg.get("gen_diag_samples"),
        "production_generation_diagnostics_disabled": tcfg.get("gen_diag_samples", 0) == 0,
        "n_requested": args.n,
        "max_new_tokens": args.max_new_tokens,
    }

    index = ShardIndex.load(dcfg["eval_index"])
    store = ShardStore(dcfg["shard_root"], verify=dcfg.get("verify", "checksum"))
    ds = Sft2SegDataset(index, store, allow_local_assembly=False)
    report["eval_index"] = dcfg["eval_index"]
    report["eval_n"] = len(ds)

    t0 = time.time()
    processor, special_ids = load_processor(mcfg["model_name_or_path"], dcfg["max_length"])
    model, freeze_report, _ = setup_model(
        mcfg["model_name_or_path"], processor, special_ids,
        attn_implementation=mcfg["attn_implementation"], dtype=mcfg["dtype"],
        reinit_special_token_rows=mcfg.get("reinit_special_token_rows", True),
        seed=mcfg.get("special_token_init_seed", 0),
    )
    model = model.to("cuda:0")
    model.eval()
    report["special_token_ids"] = special_ids
    report["load_seconds"] = round(time.time() - t0, 2)
    report["use_cache_flag_after_load"] = bool(model.config.use_cache)

    collator = Sft2SegCollator(processor, max_length=dcfg["max_length"],
                               system_prompt=dcfg.get("system_prompt"),
                               supervise_eos=dcfg.get("supervise_eos", True))

    stub = SimpleNamespace(
        data_collator=collator,
        model=model,
        args=SimpleNamespace(device=torch.device("cuda:0")),
        gen_diag_samples=args.n,
        gen_diag_max_new_tokens=args.max_new_tokens,
    )

    t0 = time.time()
    metrics = Qwen3VLSFTTrainer.generation_diagnostics(stub, ds, prefix="eval_gen_")
    report["generation_seconds"] = round(time.time() - t0, 2)
    report["metrics"] = metrics
    report["pipeline_ran"] = bool(metrics)

    # keep the raw text for the report: re-run decode on the same samples so the
    # deliverable shows what actually came out, not just the aggregate rates.
    samples = []
    for i in range(min(args.n, len(ds))):
        s = ds[i]
        enc = collator.encode_one(s)
        n_prompt = enc["n_prompt_tokens"]
        ids = torch.tensor([enc["input_ids"][:n_prompt]], device="cuda:0")
        img = collator.processor.image_processor(images=[s.image], do_resize=False,
                                                 return_tensors="pt")
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=img["pixel_values"].to("cuda:0", model.dtype),
                image_grid_thw=img["image_grid_thw"].to("cuda:0"),
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
        text = processor.tokenizer.decode(gen[0][n_prompt:], skip_special_tokens=False)
        parsed = parse_two_segment(text)
        samples.append({
            "sample_id": s.sample_id,
            "n_prompt_tokens": n_prompt,
            "n_generated_tokens": int(gen.shape[1] - n_prompt),
            "seconds": round(time.time() - t1, 2),
            "generated_text_head": text[:600],
            "parsed": {k: v for k, v in parsed.items() if k not in ("where_body", "color_body")},
            "gt_where_head": s.where_text[:200],
            "gt_color_head": s.color_text[:200],
        })
    report["samples"] = samples
    report["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)

    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "samples"},
                     indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
