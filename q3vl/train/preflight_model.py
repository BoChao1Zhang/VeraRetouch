"""Model-side preflight: spec 9 items 1-4 plus the single-batch smoke (item 10).

    python -m q3vl.train.preflight_model --out-dir <dir> [--device cuda:0]

Runs on ONE GPU and never starts training. Produces ``preflight_model.json``
and a markdown fragment. Items 5-9 and 11-12 belong to the data side / joint
preflight and are reported here as ``skipped(owner=S0-DATA)``.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .args import ALLOWED_BATCH_COMBOS, SFTTrainingArguments, validate_frozen_hyperparameters
from .collator import Sft2SegCollator
from .constants import GLOBAL_BATCH_SIZE, SPECIAL_TOKENS
from .dataset import Sft2SegDataset
from .diagnostics import parse_two_segment, segment_token_stats
from .freeze import format_freeze_table, is_trainable_param
from .modeling import (
    UPSTREAM_REVISION, load_processor, model_architecture_facts, setup_model, verify_model_shards,
)
from .mock_shards import build_mock_shard
from .shards import ShardIndex, ShardStore
from .tokens import verify_reload, verify_single_token

DEFAULT_MODEL = "/home/bc/data/models/Qwen3-VL-4B-Instruct"


def _git_commit(repo: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", repo, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def environment_facts(repo_root: str) -> dict[str, Any]:
    import transformers

    facts: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "n_gpus": torch.cuda.device_count(),
        "git_commit": _git_commit(repo_root),
    }
    for mod in ("deepspeed", "flash_attn", "accelerate"):
        try:
            facts[mod] = __import__(mod).__version__
        except Exception as exc:  # noqa: BLE001
            facts[mod] = f"MISSING ({type(exc).__name__})"
    if torch.cuda.is_available():
        facts["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "total_memory_gb": round(
                    torch.cuda.get_device_properties(i).total_memory / 1024**3, 1
                ),
            }
            for i in range(torch.cuda.device_count())
        ]
    return facts


# --- item 1 ---------------------------------------------------------------
def check_shards(model_path: str) -> dict[str, Any]:
    report = verify_model_shards(model_path)
    report["upstream_source"] = (
        "https://huggingface.co/api/models/Qwen/Qwen3-VL-4B-Instruct?blobs=true "
        f"(revision {UPSTREAM_REVISION}, fetched 2026-08-04)"
    )
    report["status"] = "PASS" if report["passed"] else "FAIL"
    return report


# --- items 2 & 3 ----------------------------------------------------------
def check_freeze(model, freeze_report) -> dict[str, Any]:
    d = freeze_report.to_dict()
    d["status"] = "PASS"
    st = d["subtree_counts"]
    checks = {
        "vision_blocks_all_frozen": st["vision_blocks"]["trainable_params"] == 0,
        "vision_patch_embed_frozen": st["vision_patch_embed"]["trainable_params"] == 0,
        "vision_pos_embed_frozen": st["vision_pos_embed"]["trainable_params"] == 0,
        "main_merger_all_trainable": st["vision_merger"]["frozen_params"] == 0
        and st["vision_merger"]["params"] > 0,
        "deepstack_mergers_all_trainable": st["vision_deepstack_mergers"]["frozen_params"] == 0
        and st["vision_deepstack_mergers"]["tensors"] == 18,
        "language_all_trainable": st["language_layers"]["frozen_params"] == 0,
        "embeddings_trainable": st["language_embed_tokens"]["frozen_params"] == 0,
        "no_peft_modules": not any(
            "lora" in n.lower() or "adapter" in n.lower() for n, _ in model.named_parameters()
        ),
    }
    d["checks"] = checks
    if not all(checks.values()):
        d["status"] = "FAIL"
    d["table"] = format_freeze_table(freeze_report)
    return d


def set_gradient_checkpointing(model, enabled: bool) -> None:
    if enabled:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()
    model.config.use_cache = False


def check_gradient_evidence(model, batch, device, gradient_checkpointing: bool = False) -> dict[str, Any]:
    """Item 3: empirical proof that vision has no grad and merger/LM do.

    Run with gradient checkpointing both off and on: a checkpointed subtree
    whose inputs do not require grad is the classic way to silently lose the
    merger gradients, so the on-case is asserted rather than assumed.
    """
    set_gradient_checkpointing(model, gradient_checkpointing)
    model.train()
    model.zero_grad(set_to_none=True)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    segment_ids = inputs.pop("segment_ids", None)
    out = model(**inputs)
    loss = out.loss
    loss.backward()

    groups = {
        "vision_blocks": "model.visual.blocks.",
        "vision_patch_embed": "model.visual.patch_embed.",
        "vision_pos_embed": "model.visual.pos_embed",
        "vision_merger": "model.visual.merger.",
        "vision_deepstack_mergers": "model.visual.deepstack_merger_list.",
        "language_embed_tokens": "model.language_model.embed_tokens",
        "language_layers": "model.language_model.layers.",
        "language_norm": "model.language_model.norm",
    }
    stats: dict[str, Any] = {}
    for gname, prefix in groups.items():
        sq, n_with_grad, n_params, n_nonfinite = 0.0, 0, 0, 0
        for name, p in model.named_parameters():
            if not name.startswith(prefix):
                continue
            n_params += 1
            if p.grad is not None:
                n_with_grad += 1
                g = p.grad.detach().float()
                if not torch.isfinite(g).all():
                    n_nonfinite += 1
                sq += float(g.pow(2).sum())
        stats[gname] = {
            "n_params": n_params,
            "n_with_grad": n_with_grad,
            "grad_norm": sq ** 0.5,
            "n_nonfinite": n_nonfinite,
            "expect_grad": is_trainable_param(prefix + "w"),
        }

    per_token, correct, segs = segment_token_stats(out.logits.detach(), inputs["labels"], segment_ids)
    checks = {
        "loss_finite": bool(torch.isfinite(loss)),
        "vision_blocks_no_grad": stats["vision_blocks"]["n_with_grad"] == 0,
        "patch_embed_no_grad": stats["vision_patch_embed"]["n_with_grad"] == 0,
        "pos_embed_no_grad": stats["vision_pos_embed"]["n_with_grad"] == 0,
        "merger_has_grad": stats["vision_merger"]["n_with_grad"] == stats["vision_merger"]["n_params"]
        and stats["vision_merger"]["grad_norm"] > 0,
        "deepstack_has_grad": stats["vision_deepstack_mergers"]["n_with_grad"]
        == stats["vision_deepstack_mergers"]["n_params"]
        and stats["vision_deepstack_mergers"]["grad_norm"] > 0,
        "language_has_grad": stats["language_layers"]["n_with_grad"]
        == stats["language_layers"]["n_params"]
        and stats["language_layers"]["grad_norm"] > 0,
        "embed_has_grad": stats["language_embed_tokens"]["grad_norm"] > 0,
        "all_grads_finite": all(v["n_nonfinite"] == 0 for v in stats.values()),
    }
    result = {
        "gradient_checkpointing": gradient_checkpointing,
        "loss": float(loss.detach()),
        "n_supervised_tokens": int(segs.numel()),
        "group_grad": stats,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }
    model.zero_grad(set_to_none=True)
    return result


# --- item 4 ---------------------------------------------------------------
def check_special_tokens(processor, special_ids: dict[str, int], model, out_dir: Path) -> dict[str, Any]:
    tok = processor.tokenizer
    ids = verify_single_token(tok)
    save_dir = out_dir / "tokenizer_reload_probe"
    if save_dir.exists():
        shutil.rmtree(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(save_dir))
    reload_info = verify_reload(str(save_dir), ids)

    emb = model.get_input_embeddings()
    rows = emb.weight.data[[ids[t] for t in SPECIAL_TOKENS]].float()
    pair = torch.cdist(rows, rows)
    off_diag = pair[~torch.eye(len(SPECIAL_TOKENS), dtype=torch.bool, device=pair.device)]

    saved_files = sorted(p.name for p in save_dir.iterdir())
    # Every check below is written over ``SPECIAL_TOKENS`` rather than a literal
    # 4, so the v2seg 6-tuple (incl. <seg_where>/<seg_color>) is covered as-is.
    checks = {
        "all_single_token": True,  # verify_single_token would have raised
        "ids_distinct": len(set(ids.values())) == len(SPECIAL_TOKENS),
        "ids_stable_after_reload": reload_info["ids"] == ids,
        "ids_in_embedding_range": max(ids.values()) < emb.weight.shape[0],
        "rows_not_degenerate": float(off_diag.min()) > 0,
        "tokenizer_files_saved": "tokenizer.json" in saved_files
        and "special_tokens_map.json" in saved_files,
        "chat_template_saved": any("chat_template" in f for f in saved_files),
    }
    return {
        "ids": ids,
        "tokenizer_len": len(tok),
        "embedding_rows": int(emb.weight.shape[0]),
        "reload": reload_info,
        "saved_files": saved_files,
        "new_row_min_pairwise_distance": float(off_diag.min()),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


# --- item 10 (single-GPU smoke) -------------------------------------------
def run_smoke(
    model, collator, dataset, device, micro_batch: int, steps: int = 3,
    gradient_checkpointing: bool = True,
) -> dict[str, Any]:
    from torch.optim import AdamW

    set_gradient_checkpointing(model, gradient_checkpointing)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    mem_after_load = torch.cuda.memory_allocated(device) / 1024**3
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=1e-5, weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8)

    n = min(micro_batch, len(dataset))
    batch = collator([dataset[i] for i in range(n)])
    n_tokens = int(batch["attention_mask"].sum())
    losses, grad_norms, step_times = [], [], []
    model.train()
    for step in range(steps):
        t0 = time.perf_counter()
        inputs = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        inputs.pop("segment_ids", None)
        out = model(**inputs)
        loss = out.loss
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        step_times.append(time.perf_counter() - t0)
        losses.append(float(loss))
        grad_norms.append(float(gnorm))

    peak = torch.cuda.max_memory_allocated(device) / 1024**3
    reserved = torch.cuda.max_memory_reserved(device) / 1024**3
    total = torch.cuda.get_device_properties(device).total_memory / 1024**3
    steady = step_times[1:] or step_times
    checks = {
        "all_losses_finite": all(v == v and abs(v) != float("inf") for v in losses),
        "all_grad_norms_finite": all(v == v and abs(v) != float("inf") for v in grad_norms),
        "loss_positive": all(v > 0 for v in losses),
        "peak_memory_below_device": peak < total,
    }
    result = {
        "gradient_checkpointing": gradient_checkpointing,
        "micro_batch_size": n,
        "seq_len": int(batch["input_ids"].shape[1]),
        "batch_tokens": n_tokens,
        "visual_patches": int(batch["pixel_values"].shape[0]),
        "losses": losses,
        "grad_norms": grad_norms,
        "step_times_s": step_times,
        "steady_step_time_s": sum(steady) / len(steady),
        "samples_per_s_single_gpu": n / (sum(steady) / len(steady)),
        "mem_after_load_gb": round(mem_after_load, 2),
        "peak_allocated_gb": round(peak, 2),
        "peak_reserved_gb": round(reserved, 2),
        "device_total_gb": round(total, 1),
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "note": (
            "Single GPU, no ZeRO-3. torch.optim.AdamW keeps its moments in the "
            "parameter dtype, so this configuration holds bf16 params (8.3 GB) + bf16 "
            "grads (8.3 GB) + 2x bf16 moments (16.5 GB) = 33 GB of state, and the rest "
            "is activations. DeepSpeed ZeRO-3 instead keeps fp32 master weights and "
            "fp32 moments but shards all three across 2 GPUs: "
            "(4.13e9 x 4 x 3) / 2 = 24.8 GB per device, plus transient bf16 gathers and "
            "the same activation cost measured here. The activation figure is therefore "
            "the transferable number; the state figure is not."
        ),
    }
    model.zero_grad(set_to_none=True)
    del opt
    torch.cuda.empty_cache()
    return result


def build_worst_case_samples(collator, n: int, max_length: int = 2048) -> list:
    """The batch that actually decides 4xGAS4 vs 2xGAS8.

    Longest legal image (512x2048 -> 1024 visual tokens) and an assistant target
    grown until the full sequence just fits ``max_length``.
    """
    import random

    from PIL import Image

    from .dataset import Sft2SegSample
    from .imageproc import prepare_image

    rng = random.Random(0)
    img = Image.new("RGB", (2048, 512))
    img.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(2048 * 512)])
    image, geom = prepare_image(img)
    assert geom.n_visual_tokens == 1024, geom

    filler = "The midtones drift cool and the highlights clip toward magenta. "
    instruction = "Warm the light and lift the subject."

    def make(color_text: str) -> Sft2SegSample:
        return Sft2SegSample(
            sample_id="worst-case", image=image, geometry=geom,
            instruction=instruction, where_text="The edit covers the standing subject.",
            color_text=color_text, meta={},
        )

    reps = 1
    best = filler
    while True:
        candidate = filler * (reps + 1)
        try:
            length = len(collator.encode_one(make(candidate))["input_ids"])
        except Exception:
            break
        if length > max_length:
            break
        best, reps = candidate, reps + 1
    sample = make(best)
    return [sample] * n


def run_batch_plan_probe(model, collator, device, steps: int = 2) -> dict[str, Any]:
    """Memory probe for both spec-7.2 batch combos on the worst-case batch."""
    out: dict[str, Any] = {"probes": []}
    for micro, gas in ALLOWED_BATCH_COMBOS:
        samples = build_worst_case_samples(collator, micro)
        entry: dict[str, Any] = {
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": gas,
            "effective_global_batch": micro * gas * 2,
        }
        try:
            res = run_smoke(model, collator, samples, device, micro, steps=steps)
            entry.update({
                "seq_len": res["seq_len"],
                "visual_patches": res["visual_patches"],
                "peak_allocated_gb": res["peak_allocated_gb"],
                "peak_reserved_gb": res["peak_reserved_gb"],
                "steady_step_time_s": round(res["steady_step_time_s"], 3),
                "losses": res["losses"],
                "ok": res["status"] == "PASS",
            })
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            entry.update({"ok": False, "error": f"OOM: {exc}"})
        out["probes"].append(entry)
    preferred = out["probes"][0]
    out["recommended"] = (
        "4x4" if preferred.get("ok") else "2x8" if out["probes"][1].get("ok") else "NONE"
    )
    out["status"] = "PASS" if out["recommended"] != "NONE" else "FAIL"
    return out


# --- pipeline checks on mock data -----------------------------------------
def check_pipeline(collator, dataset) -> dict[str, Any]:
    from .constants import (
        IGNORE_INDEX, SEG_COLOR, SEG_EOS, SEG_IGNORE, SEG_SEGCOLOR, SEG_SEGWHERE, SEG_WHERE,
    )

    tok = collator.tokenizer
    ids = {t: tok(t, add_special_tokens=False)["input_ids"][0] for t in SPECIAL_TOKENS}
    concat = collator.check_concat_equivalence(dataset[0])

    n = min(4, len(dataset))
    batch = collator([dataset[i] for i in range(n)])
    labels = batch["labels"]
    segs = batch["segment_ids"]
    input_ids = batch["input_ids"]

    supervised = labels != IGNORE_INDEX
    image_token_id = collator.processor.image_token_id
    checks = {
        "concat_equivalence": concat["equal"],
        "no_image_token_supervised": int(((input_ids == image_token_id) & supervised).sum()) == 0,
        "labels_match_inputs_where_supervised": bool(
            (labels[supervised] == input_ids[supervised]).all()
        ),
        "prompt_fully_masked": bool((segs[~supervised] == SEG_IGNORE).all()),
        "where_before_color": True,
        # named for the original 4-tuple; the loop below covers every entry of
        # SPECIAL_TOKENS, i.e. all 6 under v2seg.
        "all_four_tokens_supervised": True,
        # v2seg: </color> < <seg_where> < <seg_color> < <|im_end|>
        "seg_tail_between_color_and_eos": True,
        "seq_within_limit": int(input_ids.shape[1]) <= collator.max_length,
    }
    for row in range(n):
        s = segs[row]
        w = (s == SEG_WHERE).nonzero().flatten()
        c = (s == SEG_COLOR).nonzero().flatten()
        e = (s == SEG_EOS).nonzero().flatten()
        sw = (s == SEG_SEGWHERE).nonzero().flatten()
        sc = (s == SEG_SEGCOLOR).nonzero().flatten()
        if not (len(w) and len(c) and int(w[-1]) < int(c[0])):
            checks["where_before_color"] = False
        tail_ok = len(c) and len(sw) == 1 and len(sc) == 1 and int(c[-1]) < int(sw[0]) < int(sc[0])
        if tail_ok and len(e):  # len(e)==0 only when supervise_eos is off
            tail_ok = int(sc[0]) < int(e[0])
        if not tail_ok:
            checks["seg_tail_between_color_and_eos"] = False
        row_ids = input_ids[row]
        for t, tid in ids.items():
            pos = (row_ids == tid).nonzero().flatten()
            if len(pos) != 1 or not bool(supervised[row][pos[0]]):
                checks["all_four_tokens_supervised"] = False
        if len(e) and not bool(supervised[row][e[0]]):
            checks["all_four_tokens_supervised"] = False

    geoms = [dataset[i].geometry.to_dict() for i in range(min(8, len(dataset)))]
    return {
        "concat_equivalence": concat,
        "batch_shapes": {k: list(v.shape) for k, v in batch.items() if torch.is_tensor(v)},
        "n_supervised_tokens": int(supervised.sum()),
        "n_total_tokens": int(batch["attention_mask"].sum()),
        "image_geometries": geoms,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
    }


def check_batch_plan() -> dict[str, Any]:
    out = {"allowed_combos": [list(c) for c in ALLOWED_BATCH_COMBOS], "world_size": 2}
    rows = []
    for micro, gas in ALLOWED_BATCH_COMBOS:
        rows.append({
            "per_device_train_batch_size": micro,
            "gradient_accumulation_steps": gas,
            "world_size": 2,
            "effective_global_batch": micro * gas * 2,
            "matches_spec": micro * gas * 2 == GLOBAL_BATCH_SIZE,
        })
    out["combos"] = rows
    out["status"] = "PASS" if all(r["matches_spec"] for r in rows) else "FAIL"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--smoke-steps", type=int, default=3)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--mock-dir", default=None)
    ap.add_argument("--skip-hashes", action="store_true")
    ap.add_argument("--repo-root", default="/home/bc/VeraRetouch")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "spec": "docs/QWEN3_VL_BASE_SFT_SPEC_2026-08-04.md",
        "environment": environment_facts(args.repo_root),
        "architecture": model_architecture_facts(args.model_path),
    }

    print("[1/7] verifying model shards ...", flush=True)
    report["item1_shard_integrity"] = check_shards(args.model_path)

    print("[2/7] building mock shards + dataset ...", flush=True)
    mock_dir = Path(args.mock_dir or tempfile.mkdtemp(prefix="q3vl-mock-"))
    mock = build_mock_shard(mock_dir, n_samples=8, split="train")
    index = ShardIndex.load(mock["index"], split="train")
    store = ShardStore(mock["shard_root"], verify="checksum")
    dataset = Sft2SegDataset(index, store)
    report["mock_data"] = {
        "root": str(mock_dir), "n_samples": mock["n_samples"], "layout": index.layout,
    }

    print("[3/7] loading processor + registering special tokens ...", flush=True)
    processor, special_ids = load_processor(args.model_path, max_length=2048)
    collator = Sft2SegCollator(processor, collect_stats=True)

    print("[4/7] checking the data pipeline / loss mask ...", flush=True)
    report["pipeline"] = check_pipeline(collator, dataset)
    report["collator_stats"] = collator.stats.snapshot() if collator.stats else None

    print("[5/7] loading model (this reads ~8.9 GB) ...", flush=True)
    device = torch.device(args.device)
    t0 = time.perf_counter()
    model, freeze_report, emb_info = setup_model(
        args.model_path, processor, special_ids,
        attn_implementation=args.attn, dtype="bfloat16",
    )
    model.to(device)
    report["model_load_seconds"] = round(time.perf_counter() - t0, 1)
    report["item2_freeze"] = check_freeze(model, freeze_report)
    report["embeddings"] = emb_info

    print("[6/7] special token reload check ...", flush=True)
    report["item4_special_tokens"] = check_special_tokens(processor, special_ids, model, out_dir)

    print("[7/7] gradient evidence + smoke ...", flush=True)
    batch = collator([dataset[i] for i in range(min(2, len(dataset)))])
    report["item3_gradient_evidence"] = check_gradient_evidence(
        model, batch, device, gradient_checkpointing=False
    )
    report["item3_gradient_evidence_gc"] = check_gradient_evidence(
        model, batch, device, gradient_checkpointing=True
    )
    report["item10_smoke"] = run_smoke(
        model, collator, dataset, device, args.micro_batch, args.smoke_steps,
        gradient_checkpointing=True,
    )
    report["item11_batch_plan"] = check_batch_plan()
    print("      worst-case batch memory probe ...", flush=True)
    report["item11_batch_plan"]["worst_case_probe"] = run_batch_plan_probe(model, collator, device)

    report["deferred_items"] = {
        str(i): "owner=S0-DATA / joint preflight" for i in (5, 6, 7, 8, 9, 12)
    }
    statuses = {k: v["status"] for k, v in report.items() if isinstance(v, dict) and "status" in v}
    report["status_summary"] = statuses
    report["overall"] = "PASS" if all(s == "PASS" for s in statuses.values()) else "FAIL"

    (out_dir / "preflight_model.json").write_text(
        json.dumps(report, indent=2, default=str, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(statuses, indent=2))
    print("overall:", report["overall"])
    print("written:", out_dir / "preflight_model.json")
    return 0 if report["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
