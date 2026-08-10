#!/usr/bin/env python
"""Offline checkpoint verification for the Qwen3-VL-4B base SFT (S0-TRAIN closeout).

Main-agent ruling D-J3 turned in-training generation diagnostics OFF
(``gen_diag_samples: 0``) because ZeRO-3 ``generate`` was never validated.  The
spec 8.3 structural rates therefore have to be produced *offline*, on a single
GPU, without DeepSpeed -- which is exactly what this script does, for both
protected checkpoints (step 2488 = 0.5 epoch, step 4976 = 1.0 epoch).

For every checkpoint it reports

  A. **book-keeping**: assistant-only ``eval_loss`` and the per-segment eval
     metrics, read straight out of the checkpoint's ``trainer_state.json``.
     Nothing is re-evaluated.
  B. **structural generation diagnostics** on a fixed, seeded sample of the
     V_where eval split: ``<where>``/``<color>`` tag completeness, fixed-order
     accuracy, both-segments-non-empty rate, legacy seven-tag leak rate,
     duplicate-tag rates, EOS-termination rate.
  C. **rough alignment with the GT segments**: token-F1 / exact-match for the
     ``where`` segment, and token-F1 + the "colour segment is exactly the six
     canonical bodies" line-count statistic for the ``color`` segment.

Deliberate non-goals / red lines
--------------------------------
* **Checkpoint selection must not use val loss** (CLAUDE.md 红线速查).  The
  eval_loss column is descriptive book-keeping only; this script never emits a
  "winner" and the report says so in as many words.
* The checkpoints are loaded with a plain ``from_pretrained``, **never** with
  ``q3vl.train.modeling.setup_model``: that helper re-initialises the four
  special-token embedding rows, which would silently destroy exactly the
  weights we are trying to inspect.
* Both checkpoints see the **same** sample ids and the same decoding settings,
  so the two columns are comparable.

Batched decoding uses left padding + KV cache.  Left padding through Qwen3-VL's
mrope is asserted, not assumed: ``--consistency-k`` samples are generated both
alone and inside a batch and the strings must be identical; on any mismatch the
run falls back to ``batch_size=1`` for everything and records the fallback.

Usage
-----
    CUDA_VISIBLE_DEVICES=0 python verify_checkpoints.py            # full run
    python verify_checkpoints.py --dry-run --n 4                   # no GPU
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import re
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from q3vl.train.collator import Sft2SegCollator                     # noqa: E402
from q3vl.train.constants import CANONICAL_COLOR_FIELDS             # noqa: E402
from q3vl.train.dataset import Sft2SegDataset                       # noqa: E402
from q3vl.train.diagnostics import (                                # noqa: E402
    aggregate_generation_diagnostics, parse_two_segment,
)
from q3vl.train.shards import ShardIndex, ShardStore                # noqa: E402
from q3vl.train.tokens import verify_single_token                   # noqa: E402

DEFAULT_CONFIG = REPO / "q3vl/train/configs/sft_base.yaml"
DEFAULT_RUN_DIR = Path("/home/bc/data/runs/q3vl_base_sft_20260804")
DEFAULT_OUT_DIR = (
    REPO / "experiments/Q3VL_metacanvas_where_what_20260804/s0_base_sft"
)
SCHEMA = "q3vl.s0_base_sft.checkpoint_verification/1"
N_CANONICAL_COLOR_BODIES = len(CANONICAL_COLOR_FIELDS)   # == 6

# files a usable checkpoint must carry (model / tokenizer / processor / state)
REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "trainer_state.json",
)

_WORD = re.compile(r"[0-9a-z]+")


def log(msg: str) -> None:
    print(f"[verify] {msg}", flush=True)


# --------------------------------------------------------------------------
# text statistics
# --------------------------------------------------------------------------
def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    p, g = Counter(tokens(pred)), Counter(tokens(gold))
    if not p or not g:
        return 0.0
    overlap = sum((p & g).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(g.values())
    return 2 * prec * rec / (prec + rec)


def norm(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def _quantiles(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"p50": None, "p90": None, "min": None, "max": None, "mean": None}
    s = sorted(xs)

    def q(p: float) -> float:
        return float(s[min(len(s) - 1, int(round(p * (len(s) - 1))))])

    return {
        "p50": q(0.5), "p90": q(0.9), "min": float(s[0]), "max": float(s[-1]),
        "mean": float(sum(s) / len(s)),
    }


def alignment_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rough (not headline) agreement between the generated and GT segments."""
    ok = [r for r in rows if r["parsed"]["order_ok"]]
    n_ok = len(ok)
    if n_ok == 0:
        return {"n_parsable": 0, "note": "no sample produced a well-formed two-segment answer"}

    where_f1 = [token_f1(r["pred_where"], r["gt_where"]) for r in ok]
    color_f1 = [token_f1(r["pred_color"], r["gt_color"]) for r in ok]
    where_exact = sum(1 for r in ok if norm(r["pred_where"]) == norm(r["gt_where"]))

    pred_lines = [len([ln for ln in r["pred_color"].split("\n") if ln.strip()]) for r in ok]
    gt_lines = [len([ln for ln in r["gt_color"].split("\n") if ln.strip()]) for r in ok]
    line_match = sum(1 for a, b in zip(pred_lines, gt_lines) if a == b)
    six_bodies = sum(1 for a in pred_lines if a == N_CANONICAL_COLOR_BODIES)
    gt_six = sum(1 for b in gt_lines if b == N_CANONICAL_COLOR_BODIES)

    # per-line F1, only where the line counts agree (index-aligned)
    per_line: list[float] = []
    for r, a, b in zip(ok, pred_lines, gt_lines):
        if a != b:
            continue
        pl = [ln for ln in r["pred_color"].split("\n") if ln.strip()]
        gl = [ln for ln in r["gt_color"].split("\n") if ln.strip()]
        per_line.extend(token_f1(x, y) for x, y in zip(pl, gl))

    def len_ratio(key_p: str, key_g: str) -> list[float]:
        out = []
        for r in ok:
            g = len(tokens(r[key_g]))
            if g:
                out.append(len(tokens(r[key_p])) / g)
        return out

    return {
        "n_parsable": n_ok,
        "note": "rough alignment only -- token overlap, never a headline metric",
        "where": {
            "token_f1": _quantiles(where_f1),
            "exact_match_rate": where_exact / n_ok,
            "len_ratio_pred_over_gt": _quantiles(len_ratio("pred_where", "gt_where")),
        },
        "color": {
            "token_f1": _quantiles(color_f1),
            "per_line_token_f1": _quantiles(per_line),
            "pred_line_count": _quantiles([float(x) for x in pred_lines]),
            "gt_line_count": _quantiles([float(x) for x in gt_lines]),
            "pred_is_six_bodies_rate": six_bodies / n_ok,
            "gt_is_six_bodies_rate": gt_six / n_ok,
            "line_count_matches_gt_rate": line_match / n_ok,
            "len_ratio_pred_over_gt": _quantiles(len_ratio("pred_color", "gt_color")),
        },
    }


# --------------------------------------------------------------------------
# checkpoint book-keeping
# --------------------------------------------------------------------------
def checkpoint_integrity(ckpt: Path) -> dict[str, Any]:
    present = {f: (ckpt / f).is_file() for f in REQUIRED_FILES}
    shards = sorted(p.name for p in ckpt.glob("model-*.safetensors"))
    single = (ckpt / "model.safetensors").is_file()
    weights_ok = bool(shards) or single
    if shards and (ckpt / "model.safetensors.index.json").is_file():
        idx = json.loads((ckpt / "model.safetensors.index.json").read_text())
        want = sorted(set(idx.get("weight_map", {}).values()))
        weights_ok = want == shards
    else:
        want = None
    return {
        "path": str(ckpt),
        "required_files_present": present,
        "missing_files": [f for f, ok in present.items() if not ok],
        "weight_shards": shards,
        "weight_shards_expected_by_index": want,
        "weight_bytes": sum((ckpt / s).stat().st_size for s in shards) if shards else (
            (ckpt / "model.safetensors").stat().st_size if single else 0
        ),
        "weights_complete": weights_ok,
        "ok": weights_ok and not [f for f, ok in present.items() if not ok],
    }


def trainer_state_summary(ckpt: Path) -> dict[str, Any]:
    state = json.loads((ckpt / "trainer_state.json").read_text(encoding="utf-8"))
    step = int(state["global_step"])
    evals: dict[int, dict[str, Any]] = {}
    for rec in state.get("log_history", []):
        keys = [k for k in rec if k.startswith("eval_")]
        if not keys:
            continue
        s = int(rec.get("step", -1))
        evals.setdefault(s, {"step": s, "epoch": rec.get("epoch")})
        evals[s].update({k: rec[k] for k in keys})
    history = [evals[s] for s in sorted(evals)]
    marker = ckpt / "protected_checkpoint.json"
    return {
        "global_step": step,
        "max_steps": state.get("max_steps"),
        "epoch": state.get("epoch"),
        "num_input_tokens_seen": state.get("num_input_tokens_seen"),
        "total_flos": state.get("total_flos"),
        # book-keeping only; see the red line note in the report
        "best_metric_recorded_by_trainer": state.get("best_metric"),
        "best_model_checkpoint_recorded_by_trainer": state.get("best_model_checkpoint"),
        "eval_history": history,
        "eval_at_this_step": next((e for e in history if e["step"] == step), None),
        "protected_marker": (
            json.loads(marker.read_text(encoding="utf-8")) if marker.is_file() else None
        ),
    }


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------
def build_dataset(dcfg: dict[str, Any]):
    index = ShardIndex.load(dcfg["eval_index"])
    store = ShardStore(dcfg["shard_root"], verify=dcfg.get("verify", "checksum"))
    ds = Sft2SegDataset(index, store, allow_local_assembly=False)
    return ds, store


def pick_indices(n_total: int, n: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    n = min(n, n_total)
    return sorted(rng.sample(range(n_total), n))


def load_checkpoint_model(ckpt: Path, dtype: str, attn: str, device: str):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    processor = AutoProcessor.from_pretrained(str(ckpt))
    special_ids = verify_single_token(processor.tokenizer)
    used_attn = attn
    try:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(ckpt), dtype=dtypes[dtype], attn_implementation=attn
        )
    except Exception as exc:  # noqa: BLE001 -- flash-attn absent must not kill the job
        log(f"attn_implementation={attn!r} failed ({type(exc).__name__}: {exc}); retrying with sdpa")
        used_attn = "sdpa"
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            str(ckpt), dtype=dtypes[dtype], attn_implementation="sdpa"
        )
    model.to(device)
    model.eval()
    # training left config.use_cache=False (gradient checkpointing); generation
    # without a KV cache would be quadratic, so turn it back on in every place
    # that is consulted, and pass use_cache=True to generate as well.
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    # the saved generation_config is the stock sampling one (do_sample/temp/top_p);
    # diagnostics must be deterministic, so pin greedy decoding explicitly.
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = None
    gc.top_p = None
    gc.top_k = None
    gc.use_cache = True
    return model, processor, special_ids, used_attn


def terminator_ids(model, tokenizer) -> set[int]:
    ids: set[int] = set()
    eos = getattr(model.generation_config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        ids.update(int(i) for i in eos)
    elif eos is not None:
        ids.add(int(eos))
    for tok in (tokenizer.eos_token_id, tokenizer.pad_token_id,
                getattr(model.generation_config, "pad_token_id", None)):
        if tok is not None:
            ids.add(int(tok))
    return ids


def generate_batch(model, processor, collator, samples, *, max_new_tokens: int,
                   device: str, term_ids: set[int]) -> list[dict[str, Any]]:
    """Left-padded batched greedy decoding with KV cache."""
    import torch

    encs = [collator.encode_one(s) for s in samples]
    prompts = [e["input_ids"][: e["n_prompt_tokens"]] for e in encs]
    width = max(len(p) for p in prompts)
    pad_id = collator.pad_token_id
    ids_rows, attn_rows = [], []
    for p in prompts:
        k = width - len(p)
        ids_rows.append([pad_id] * k + p)
        attn_rows.append([0] * k + [1] * len(p))
    ids = torch.tensor(ids_rows, dtype=torch.long, device=device)
    attn = torch.tensor(attn_rows, dtype=torch.long, device=device)
    img = processor.image_processor(
        images=[s.image for s in samples], do_resize=False, return_tensors="pt"
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=ids,
            attention_mask=attn,
            pixel_values=img["pixel_values"].to(device, model.dtype),
            image_grid_thw=img["image_grid_thw"].to(device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=pad_id,
        )
    results = []
    for i in range(len(samples)):
        gen = out[i][width:].tolist()
        cut = next((j for j, t in enumerate(gen) if t in term_ids), None)
        finished = cut is not None
        kept = gen[:cut] if finished else gen
        text = processor.tokenizer.decode(kept, skip_special_tokens=False)
        results.append({
            "text": text,
            "n_generated_tokens": len(kept) + (1 if finished else 0),
            "terminated": finished,
            "truncated": not finished,
            "n_prompt_tokens": len(prompts[i]),
        })
    return results


def run_generation(ckpt: Path, ds, indices: list[int], cfg: dict[str, Any],
                   args) -> dict[str, Any]:
    import torch

    t0 = time.time()
    model, processor, special_ids, used_attn = load_checkpoint_model(
        ckpt, args.dtype, args.attn_implementation, args.device
    )
    load_s = time.time() - t0
    log(f"{ckpt.name}: model loaded in {load_s:.1f}s (attn={used_attn}) "
        f"special_ids={special_ids}")

    dcfg = cfg["data"]
    collator = Sft2SegCollator(
        processor,
        max_length=dcfg["max_length"],
        system_prompt=dcfg.get("system_prompt"),
        supervise_eos=dcfg.get("supervise_eos", True),
    )
    term_ids = terminator_ids(model, processor.tokenizer)

    samples = []
    errors = []
    for i in indices:
        try:
            s = ds[i]
            collator.encode_one(s)      # fail here, not inside a batch
            samples.append(s)
        except Exception as exc:  # noqa: BLE001
            errors.append({"dataset_index": i, "error": f"{type(exc).__name__}: {exc}"})
    log(f"{ckpt.name}: {len(samples)} samples decoded, {len(errors)} dataset errors")

    # -- left-padding sanity: batch vs single must agree, else fall back -----
    batch_size = max(1, args.batch_size)
    consistency: dict[str, Any] = {"checked": 0, "identical": 0, "ran": False}
    k = min(args.consistency_k, len(samples))
    if batch_size > 1 and k > 1:
        probe = samples[:k]
        t1 = time.time()
        batched = generate_batch(model, processor, collator, probe,
                                 max_new_tokens=args.max_new_tokens,
                                 device=args.device, term_ids=term_ids)
        singles = [
            generate_batch(model, processor, collator, [s],
                           max_new_tokens=args.max_new_tokens,
                           device=args.device, term_ids=term_ids)[0]
            for s in probe
        ]
        same = sum(1 for a, b in zip(batched, singles) if a["text"] == b["text"])
        consistency = {
            "ran": True, "checked": k, "identical": same,
            "identical_rate": same / k,
            "seconds": round(time.time() - t1, 1),
            "batch_size_probed": k,
        }
        if same != k:
            log(f"{ckpt.name}: LEFT-PADDING MISMATCH {same}/{k} -- falling back to batch_size=1")
            consistency["fallback_to_batch_size_1"] = True
            batch_size = 1
        else:
            consistency["fallback_to_batch_size_1"] = False
            log(f"{ckpt.name}: left-padding consistency {same}/{k} OK")

    # -- main pass ----------------------------------------------------------
    rows: list[dict[str, Any]] = []
    t2 = time.time()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start: start + batch_size]
        tb = time.time()
        gen = generate_batch(model, processor, collator, chunk,
                             max_new_tokens=args.max_new_tokens,
                             device=args.device, term_ids=term_ids)
        for s, g in zip(chunk, gen):
            parsed = parse_two_segment(g["text"])
            rows.append({
                "sample_id": s.sample_id,
                "build": s.meta.get("build"),
                "winner_confidence": s.meta.get("winner_confidence"),
                "n_prompt_tokens": g["n_prompt_tokens"],
                "n_generated_tokens": g["n_generated_tokens"],
                "terminated": g["terminated"],
                "truncated": g["truncated"],
                "generated_text": g["text"],
                "pred_where": parsed["where_body"],
                "pred_color": parsed["color_body"],
                "gt_where": s.where_text,
                "gt_color": s.color_text,
                "parsed": {k2: v for k2, v in parsed.items()
                           if k2 not in ("where_body", "color_body")},
            })
        log(f"{ckpt.name}: {min(start + batch_size, len(samples))}/{len(samples)} "
            f"generated ({time.time() - tb:.1f}s this batch)")
    gen_s = time.time() - t2

    structural = aggregate_generation_diagnostics([r["parsed"] for r in rows], prefix="")
    n = max(1, len(rows))
    summary = {
        "n_samples_requested": len(indices),
        "n_samples_generated": len(rows),
        "n_dataset_errors": len(errors),
        "dataset_errors": errors[:10],
        "attn_implementation": used_attn,
        "batch_size_used": batch_size,
        "max_new_tokens": args.max_new_tokens,
        "left_padding_consistency": consistency,
        "special_token_ids": special_ids,
        "structural": structural,
        "termination": {
            "eos_terminated_rate": sum(1 for r in rows if r["terminated"]) / n,
            "truncated_rate": sum(1 for r in rows if r["truncated"]) / n,
            "generated_tokens": _quantiles([float(r["n_generated_tokens"]) for r in rows]),
            "prompt_tokens": _quantiles([float(r["n_prompt_tokens"]) for r in rows]),
        },
        "alignment": alignment_stats(rows),
        "seconds": {"model_load": round(load_s, 1), "generation": round(gen_s, 1)},
        "peak_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
        if args.device.startswith("cuda") else None,
    }
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    return {"summary": summary, "rows": rows}


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------
def _fmt(x: Any, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_markdown(out: dict[str, Any], path: Path) -> None:
    names = list(out["checkpoints"])
    cps = out["checkpoints"]
    L: list[str] = []
    A = L.append
    A("# S0-TRAIN 收尾 · protected checkpoint 验证")
    A("")
    A("## 要验证的结论")
    A("如果这个验证通过，我们就能说 **base SFT 的两个 protected checkpoint 真的学会了两段式"
      "`<where>`/`<color>` 输出格式**（标签完整、顺序固定、两段非空、旧七标签不再泄漏），"
      "后续 Where/What 各臂可以把它当作可用的读出起点；不通过就不能说 —— 训练 loss 下降"
      "只证明 teacher forcing 下的 token 概率，不证明自由生成时格式成立。")
    A("")
    A("## 为什么需要验证它")
    A("主 agent 裁定 D-J3 把训练内的生成诊断关掉了（ZeRO-3 `generate` 未验证），"
      "spec 8.3 的结构率因此没有任何在线证据。整条 METACANVAS 链路（Where-B 的 `s` 读出、"
      "What 的配色）都建立在「模型自由生成时输出这两段」之上；不补这一步，审稿人问"
      "「你们的两段格式在推理时成立吗」时我们只有训练 loss 可拿。")
    A("")
    A("## 怎么验的")
    A(f"离线单卡（无 DeepSpeed）加载两个 protected checkpoint，在 V_where 验证集上取"
      f"**同一批** {out['settings']['n_samples']} 条（seed={out['settings']['seed']}）"
      f"做贪心生成（左填充 + KV cache，max_new_tokens={out['settings']['max_new_tokens']}），"
      f"跑 `parse_two_segment` 结构解析 + 与 GT 的粗略 token 对齐；eval_loss 直接读"
      f"checkpoint 自带的 `trainer_state.json`，不重跑 eval。")
    A("")
    A("---")
    A("")
    A("## 1. 交付物与环境")
    A("")
    A(f"- 生成时间：`{out['generated_at']}`　主机：`{out['host']}`")
    A(f"- git commit：`{out['git_commit']}`")
    A(f"- 环境：python `{out['env']['python']}` / torch `{out['env'].get('torch')}` / "
      f"transformers `{out['env'].get('transformers')}`")
    A(f"- 训练 run：`{out['settings']['run_dir']}`")
    A(f"- 评测源：`{out['settings']['eval_index']}`（split 代号 **V_where**，"
      f"共 {out['settings']['eval_split_size']} 条，抽样 {out['settings']['n_samples']} 条）")
    A(f"- 采样的 dataset 下标（两个 checkpoint 完全相同）："
      f"`{out['settings']['sample_indices'][:8]}...`（共 {len(out['settings']['sample_indices'])} 个）")
    A("")
    A("## 2. checkpoint 完整性")
    A("")
    A("| 项 | " + " | ".join(names) + " |")
    A("|---|" + "---|" * len(names))
    for key, label in (("ok", "整体完整"), ("weights_complete", "权重分片与 index 一致"),
                       ("weight_bytes", "权重字节数")):
        A(f"| {label} | " + " | ".join(_fmt(cps[c]["integrity"][key]) for c in names) + " |")
    A("| 缺失文件 | " + " | ".join(
        ", ".join(cps[c]["integrity"]["missing_files"]) or "（无）" for c in names) + " |")
    A("| global_step | " + " | ".join(str(cps[c]["state"]["global_step"]) for c in names) + " |")
    A("| epoch | " + " | ".join(_fmt(cps[c]["state"]["epoch"], 4) for c in names) + " |")
    A("")
    A("## 3. assistant-only eval_loss（book-keeping，**不得用于 checkpoint 选择**）")
    A("")
    A("> CLAUDE.md 红线速查：**checkpoint 选择禁用 val loss**。下表只是把训练期已经算过的数字"
      "汇总在一处；本报告不据此推荐任何 checkpoint。选择依据请用第 4 节的结构率与下游任务指标。")
    A("")
    A("| 指标 | " + " | ".join(names) + " |")
    A("|---|" + "---|" * len(names))
    metric_keys = ["eval_loss", "eval_seg_assistant_loss", "eval_seg_assistant_acc",
                   "eval_seg_where_loss", "eval_seg_where_acc",
                   "eval_seg_color_loss", "eval_seg_color_acc",
                   "eval_seg_eos_loss", "eval_seg_eos_acc"]
    for k in metric_keys:
        vals = []
        for c in names:
            e = cps[c]["state"]["eval_at_this_step"] or {}
            vals.append(_fmt(e.get(k), 6))
        A(f"| `{k}` | " + " | ".join(vals) + " |")
    A("")
    A("完整 eval 曲线（step → eval_loss）：")
    A("")
    last = cps[names[-1]]["state"]["eval_history"]
    A("| step | epoch | eval_loss | assistant_acc | where_acc | color_acc |")
    A("|---|---|---|---|---|---|")
    for e in last:
        A(f"| {e['step']} | {_fmt(e.get('epoch'), 4)} | {_fmt(e.get('eval_loss'), 6)} | "
          f"{_fmt(e.get('eval_seg_assistant_acc'), 4)} | {_fmt(e.get('eval_seg_where_acc'), 4)} | "
          f"{_fmt(e.get('eval_seg_color_acc'), 4)} |")
    A("")
    A("## 4. 离线生成诊断（spec 8.3 结构率）")
    A("")
    A("预注册判据来自 spec 8.3 的结构性要求；这里并排给出实测。")
    A("")
    A("| 指标 | 期望 | " + " | ".join(names) + " |")
    A("|---|---|" + "---|" * len(names))
    gate = {
        "tag_completeness": "= 1.00",
        "order_accuracy": "= 1.00",
        "where_nonempty_rate": "= 1.00",
        "color_nonempty_rate": "= 1.00",
        "legacy_tag_leak_rate": "= 0.00",
        "duplicate_where_rate": "= 0.00",
        "duplicate_color_rate": "= 0.00",
        "where_copied_into_color_rate": "≈ 0",
    }
    for k, want in gate.items():
        vals = []
        for c in names:
            g = cps[c].get("generation") or {}
            vals.append(_fmt((g.get("structural") or {}).get(k), 4))
        A(f"| `{k}` | {want} | " + " | ".join(vals) + " |")
    for k, want in (("eos_terminated_rate", "= 1.00"), ("truncated_rate", "= 0.00")):
        vals = []
        for c in names:
            g = cps[c].get("generation") or {}
            vals.append(_fmt((g.get("termination") or {}).get(k), 4))
        A(f"| `{k}` | {want} | " + " | ".join(vals) + " |")
    A("")
    A("生成量与耗时：")
    A("")
    A("| 项 | " + " | ".join(names) + " |")
    A("|---|" + "---|" * len(names))
    for label, path_keys in (
        ("生成 token 数 p50", ("termination", "generated_tokens", "p50")),
        ("生成 token 数 max", ("termination", "generated_tokens", "max")),
        ("prompt token 数 p50", ("termination", "prompt_tokens", "p50")),
        ("batch size", ("batch_size_used",)),
        ("左填充一致性", ("left_padding_consistency", "identical_rate")),
        ("模型加载秒", ("seconds", "model_load")),
        ("生成秒", ("seconds", "generation")),
        ("显存峰值 GiB", ("peak_memory_gib",)),
    ):
        vals = []
        for c in names:
            v: Any = cps[c].get("generation") or {}
            for pk in path_keys:
                v = (v or {}).get(pk) if isinstance(v, dict) else None
            vals.append(_fmt(v, 4))
        A(f"| {label} | " + " | ".join(vals) + " |")
    A("")
    A("## 5. 与 GT 的粗略对齐（**不是 headline 指标**）")
    A("")
    A("token 重叠只用来回答「生成的两段是不是在讲同一件事」，不能当质量指标：颜色段是六个"
      "固定 body 的拼接，所以行数统计比 F1 更有信息量。")
    A("")
    A("| 指标 | " + " | ".join(names) + " |")
    A("|---|" + "---|" * len(names))
    for label, keys in (
        ("where token-F1 p50", ("where", "token_f1", "p50")),
        ("where 完全一致率", ("where", "exact_match_rate")),
        ("color token-F1 p50", ("color", "token_f1", "p50")),
        ("color 逐行 token-F1 p50", ("color", "per_line_token_f1", "p50")),
        ("color 预测行数 p50", ("color", "pred_line_count", "p50")),
        ("color 预测恰为 6 行的比例", ("color", "pred_is_six_bodies_rate",)),
        ("color GT 恰为 6 行的比例", ("color", "gt_is_six_bodies_rate",)),
        ("color 行数与 GT 一致率", ("color", "line_count_matches_gt_rate",)),
        ("color 长度比 p50", ("color", "len_ratio_pred_over_gt", "p50")),
    ):
        vals = []
        for c in names:
            v: Any = (cps[c].get("generation") or {}).get("alignment") or {}
            for pk in keys:
                v = (v or {}).get(pk) if isinstance(v, dict) else None
            vals.append(_fmt(v, 4))
        A(f"| {label} | " + " | ".join(vals) + " |")
    A("")
    A("## 6. 结论")
    A("")
    for c in names:
        g = cps[c].get("generation") or {}
        s = g.get("structural") or {}
        t = g.get("termination") or {}
        struct_ok = all(
            (s.get(k) is not None) and (
                s[k] >= 1.0 if k.endswith(("completeness", "accuracy", "nonempty_rate"))
                else s[k] <= 0.0
            )
            for k in ("tag_completeness", "order_accuracy", "where_nonempty_rate",
                      "color_nonempty_rate", "legacy_tag_leak_rate")
        )
        A(f"- **{c}**：结构率全部达标 = **{'是' if struct_ok else '否'}**；"
          f"EOS 终止率 {_fmt(t.get('eos_terminated_rate'), 4)}；"
          f"截断率 {_fmt(t.get('truncated_rate'), 4)}。")
    A("")
    A("> 本报告**不做 checkpoint 选择**：红线禁止用 val loss 选 checkpoint，而结构率若两档"
      "都满分则不构成区分度。选择应由主 agent 结合下游（Where-B / What）指标裁定。")
    A("")
    A("## 7. 机器可读产物")
    A("")
    A(f"- `{out['artifacts']['metrics_json']}`")
    A(f"- `{out['artifacts']['samples_jsonl']}`（逐样本生成文本 + 解析结果）")
    A("")
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="training config the run used (for data + collator settings)")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    ap.add_argument("--checkpoint", action="append", default=None,
                    help="checkpoint dir; repeatable. default = the two protected ones")
    ap.add_argument("--steps", default="2488,4976",
                    help="protected steps to resolve under --run-dir when no --checkpoint given")
    ap.add_argument("--n", type=int, default=64, help="samples drawn from the eval split")
    ap.add_argument("--seed", type=int, default=42)
    # 448 = measured GT assistant-target max over the 64 seeded V_where samples
    # (326 tokens: p50 210, p90 251, p99 310) plus ~37% headroom, so a correct
    # answer is never cut off and truncated_rate stays a real signal.
    ap.add_argument("--max-new-tokens", type=int, default=448)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--consistency-k", type=int, default=4,
                    help="samples generated both alone and batched to prove left padding")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--dry-run", action="store_true",
                    help="everything except loading weights / generating (no GPU touched)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    dcfg = cfg["data"]

    run_dir = Path(args.run_dir)
    if args.checkpoint:
        ckpts = [Path(c) for c in args.checkpoint]
    else:
        ckpts = [run_dir / f"checkpoint-{s.strip()}" for s in args.steps.split(",") if s.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "checkpoint_verification.json"
    samples_path = out_dir / "checkpoint_verification_samples.jsonl"
    md_path = out_dir / "CHECKPOINT_VERIFICATION.md"

    env: dict[str, Any] = {"python": platform.python_version(), "executable": sys.executable}
    try:
        import torch
        import transformers
        env["torch"] = torch.__version__
        env["transformers"] = transformers.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            env["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:  # noqa: BLE001
        env["import_error"] = f"{type(exc).__name__}: {exc}"
    env["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    try:
        commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    log(f"repo={REPO} run_dir={run_dir} checkpoints={[c.name for c in ckpts]}")
    log(f"eval_index={dcfg['eval_index']}")
    ds, store = build_dataset(dcfg)
    indices = pick_indices(len(ds), args.n, args.seed)
    log(f"eval split has {len(ds)} samples; drew {len(indices)} with seed {args.seed}")

    out: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "host": socket.gethostname(),
        "git_commit": commit,
        "env": env,
        "dry_run": bool(args.dry_run),
        "settings": {
            "config": str(Path(args.config).resolve()),
            "run_dir": str(run_dir),
            "eval_index": dcfg["eval_index"],
            "eval_split_size": len(ds),
            "n_samples": len(indices),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "consistency_k": args.consistency_k,
            "dtype": args.dtype,
            "attn_implementation": args.attn_implementation,
            "device": args.device,
            "decoding": "greedy (do_sample=False), KV cache on, left padding",
            "sample_indices": indices,
        },
        "red_lines": [
            "checkpoint selection must NOT use val loss (CLAUDE.md 红线速查); the "
            "eval_loss column here is book-keeping only",
            "checkpoints are loaded with plain from_pretrained -- setup_model would "
            "re-init the four special-token embedding rows",
        ],
        "artifacts": {
            "metrics_json": str(metrics_path),
            "samples_jsonl": str(samples_path),
            "markdown": str(md_path),
        },
        "checkpoints": {},
    }

    all_rows: list[dict[str, Any]] = []
    rc = 0
    for ckpt in ckpts:
        name = ckpt.name
        log(f"=== {name} ===")
        entry: dict[str, Any] = {"path": str(ckpt)}
        if not ckpt.is_dir():
            entry["error"] = "checkpoint directory does not exist"
            out["checkpoints"][name] = entry
            rc = 1
            continue
        entry["integrity"] = checkpoint_integrity(ckpt)
        entry["state"] = trainer_state_summary(ckpt)
        log(f"{name}: global_step={entry['state']['global_step']} "
            f"integrity_ok={entry['integrity']['ok']}")
        if not entry["integrity"]["ok"]:
            rc = 1
        if args.dry_run:
            # exercise the dataset + collator + tokenizer path without weights
            from transformers import AutoProcessor
            processor = AutoProcessor.from_pretrained(str(ckpt))
            special_ids = verify_single_token(processor.tokenizer)
            collator = Sft2SegCollator(
                processor, max_length=dcfg["max_length"],
                system_prompt=dcfg.get("system_prompt"),
                supervise_eos=dcfg.get("supervise_eos", True),
            )
            probe = min(len(indices), 4)
            encs = []
            for i in indices[:probe]:
                s = ds[i]
                e = collator.encode_one(s)
                encs.append({"sample_id": s.sample_id,
                             "n_prompt_tokens": e["n_prompt_tokens"],
                             "n_where_tokens": e["n_where_tokens"],
                             "n_color_tokens": e["n_color_tokens"],
                             "total": len(e["input_ids"])})
            entry["dry_run_probe"] = {"special_token_ids": special_ids, "encoded": encs}
            log(f"{name}: dry-run probe encoded {probe} samples OK")
        else:
            entry["generation"] = None
            res = run_generation(ckpt, ds, indices, cfg, args)
            entry["generation"] = res["summary"]
            for r in res["rows"]:
                all_rows.append({"checkpoint": name, **r})
        out["checkpoints"][name] = entry

    store.close()

    metrics_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    if all_rows:
        with samples_path.open("w", encoding="utf-8") as fh:
            for r in all_rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not args.dry_run:
        write_markdown(out, md_path)
        log(f"wrote {md_path}")
    log(f"wrote {metrics_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
