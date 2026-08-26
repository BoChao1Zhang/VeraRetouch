"""Stage 2, row A (AceTone env): AceTone-3B-PST-Preview -> 64 tokens -> LUT.

    /home/bc/data/external/acetone/venv/bin/python q3vl/whatb/acetone/pst_infer.py \\
        --inputs <inputs dir> --out <artifacts dir> --device cuda:0

Everything about the call is ``eval/predict_lut_ddp.py`` verbatim: the two-image
message with the untouched image first and the toned reference second, the
prompt text (``bridge.PST_PROMPT``), ``apply_chat_template(..., tokenize=False,
add_generation_prompt=True)``, ``process_vision_info``, the processor call, the
generation kwargs (``max_new_tokens=128, do_sample=True, temperature=0.01``), the
``<SoT>`` / ``<EoT>`` split, the ``<MM(\\d+)>`` regex, the ``num_missing``
padding branch and the ``[:64].reshape(4,4,4)``.

Two conditions per sample:

``true``     image2 = that sample's own GT after-image  (oracle reference)
``shuffle``  image2 = another sample's GT after-image, drawn once by a seeded
             derangement -- the ``N_ref_shuffle`` control

Writes ``pred_true.npz`` / ``pred_shuffle.npz`` (sample_id -> (32,32,32,3)),
``parse.jsonl`` (one line per generation: how many ``<MM..>`` tokens were found,
how many were padded in, the raw span) and ``pst_facts.json``.  No metric is
computed here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

MM_RE = re.compile(r"<MM(\d+)>")


def _load_bridge():
    path = Path(__file__).resolve().parent / "bridge.py"
    spec = importlib.util.spec_from_file_location("_acetone_bridge", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _derangement(n: int, seed: int) -> list[int]:
    """A seeded permutation with no fixed point (``shuffle`` never self-pairs)."""
    rng = np.random.default_rng(seed)
    for _ in range(1000):
        perm = rng.permutation(n)
        if n < 2 or not (perm == np.arange(n)).any():
            return [int(i) for i in perm]
    perm = list(range(1, n)) + [0]                      # pragma: no cover
    return perm


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", default="/home/bc/data/runs/what_b/acetone_inputs")
    ap.add_argument("--out",
                    default="/home/bc/data/runs/what_b/whatb_ACETONE_pst/artifacts")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--attn", default="flash_attention_2",
                    help="falls back to sdpa and records the fallback")
    ap.add_argument("--use-cache", choices=("config", "true"), default="config",
                    help=("'config' passes nothing to generate(), which is what "
                          "eval/predict_lut_ddp.py does; 'true' passes "
                          "use_cache=True.  Measured warm on this box: same "
                          "wall clock (3.3 s/generation) either way."))
    ap.add_argument("--determinism-probe", type=int, default=4,
                    help=("repeat the first row's generation N times with the "
                          "same seed and record how many distinct outputs come "
                          "back; 0 disables"))
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoProcessor

    bridge = _load_bridge()
    bridge.ensure_get_path()
    from model.config_acetone import AceToneConfig                # type: ignore
    from model.modeling_acetone import AceToneVLM                 # type: ignore
    from qwen_vl_utils import process_vision_info                 # type: ignore

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in
            (Path(args.inputs) / "rows.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    n = len(rows)
    perm = _derangement(n, args.seed)

    # -- the model, loaded the way eval/predict_lut_ddp.py:88-100 loads it ----
    config = AceToneConfig.from_pretrained(str(bridge.PST_MODEL_DIR))
    config.model_type = "acetone"
    config.mm_vocab_size = 256
    attn = args.attn
    try:
        model = AceToneVLM.from_pretrained(
            str(bridge.PST_MODEL_DIR), config=config, torch_dtype=torch.bfloat16,
            attn_implementation=attn).to(args.device)
    except (ImportError, ValueError) as exc:
        print(f"[attn] {attn} unavailable ({exc}); falling back to sdpa", flush=True)
        attn = "sdpa"
        model = AceToneVLM.from_pretrained(
            str(bridge.PST_MODEL_DIR), config=config, torch_dtype=torch.bfloat16,
            attn_implementation=attn).to(args.device)
    model.eval()
    processor = AutoProcessor.from_pretrained(str(bridge.PST_MODEL_DIR))
    vq, vq_facts = bridge.load_vq(device=args.device)

    def _generate(image: str, ref: str, seed: int) -> str:
        """One generation, exactly as ``eval/predict_lut_ddp.py`` issues it."""
        torch.manual_seed(seed)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "image", "image": ref},
                {"type": "text", "text": bridge.PST_PROMPT},
            ],
        }]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        cache_kw = {} if args.use_cache == "config" else {"use_cache": True}
        with torch.no_grad():
            gen = model.generate(**inputs, **cache_kw, **bridge.GENERATION_KWARGS)
        trimmed = [o[len(i_):] for i_, o in zip(inputs.input_ids, gen)]
        return processor.batch_decode(trimmed, skip_special_tokens=False,
                                      clean_up_tokenization_spaces=False)[0]

    # -- how reproducible is one generation at all? --------------------------
    determinism: dict[str, object] = {"n_repeats": 0}
    if args.determinism_probe and rows:
        seen = [_generate(rows[0]["image"], rows[0]["target"], args.seed)
                for _ in range(int(args.determinism_probe))]
        determinism = {
            "n_repeats": len(seen), "n_distinct": len(set(seen)),
            "sample_id": rows[0]["sample_id"],
            "quantity": ("same input, same torch.manual_seed, N generations -> "
                         "how many distinct output strings"),
        }
        print(json.dumps({"determinism_probe": determinism}), flush=True)

    preds: dict[str, dict[str, np.ndarray]] = {"true": {}, "shuffle": {}}
    parse_log: list[dict] = []
    t0 = time.time()
    with (out / "parse.jsonl").open("w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            for cond in ("true", "shuffle"):
                ref = row["target"] if cond == "true" \
                    else rows[perm[i]]["target"]
                t_gen = time.time()
                text_out = _generate(
                    row["image"], ref,
                    args.seed + 7919 * i + (0 if cond == "true" else 1))
                gen_s = time.time() - t_gen
                output_text = [text_out]
                prediction = output_text[0].split("<SoT>")[-1].split("<EoT>")[0]
                found = [int(v) for v in MM_RE.findall(prediction)]
                num_missing = 64 - len(found)
                padded = list(found)
                if num_missing > 0:
                    last = padded[0] if padded else 0
                    padded = padded + [last] * num_missing
                ids = np.asarray(padded[:64], dtype=np.int64)
                lut = bridge.decode_indices(vq, ids[None], device=args.device)[0]
                preds[cond][row["sample_id"]] = lut

                rec = {"sample_id": row["sample_id"], "condition": cond,
                       "ref_sample_id": (row["sample_id"] if cond == "true"
                                         else rows[perm[i]]["sample_id"]),
                       "n_tokens_found": len(found),
                       "num_missing": int(max(num_missing, 0)),
                       "n_tokens_over": int(max(-num_missing, 0)),
                       "padded": bool(num_missing > 0),
                       "token_ids": [int(v) for v in ids],
                       "n_nonfinite": int((~np.isfinite(lut)).sum()),
                       "gen_s": round(gen_s, 3),
                       "raw": output_text[0][:2000]}
                parse_log.append({k: v for k, v in rec.items() if k != "raw"})
                fh.write(json.dumps(rec) + "\n")
            if i % 10 == 0:
                el = time.time() - t0
                print(f"[pst] {i}/{n}  {el:.0f}s  "
                      f"eta {el / max(i, 1) * (n - i):.0f}s", flush=True)

    np.savez_compressed(out / "pred_true.npz", **preds["true"])
    np.savez_compressed(out / "pred_shuffle.npz", **preds["shuffle"])

    def _stats(cond: str) -> dict:
        rows_ = [r for r in parse_log if r["condition"] == cond]
        pad = [r for r in rows_ if r["padded"]]
        return {"n": len(rows_), "n_padded": len(pad),
                "parse_missing_rate": (len(pad) / len(rows_)) if rows_ else None,
                "n_missing_tokens_total": int(sum(r["num_missing"] for r in rows_)),
                "n_over_64_total": int(sum(r["n_tokens_over"] for r in rows_)),
                "n_zero_tokens_found": sum(1 for r in rows_
                                           if r["n_tokens_found"] == 0),
                "padded_sample_ids": [r["sample_id"] for r in pad],
                "n_nonfinite_luts": sum(1 for r in rows_ if r["n_nonfinite"])}

    facts = {"repo": bridge.repo_facts(), "vq": vq_facts,
             "model_dir": str(bridge.PST_MODEL_DIR),
             "attn_implementation": attn,
             "attn_requested": args.attn,
             "prompt": bridge.PST_PROMPT,
             "generation": bridge.GENERATION_KWARGS,
             "seed": args.seed,
             "seed_rule": "torch.manual_seed(seed + 7919*i + {0 true, 1 shuffle})",
             "use_cache": args.use_cache,
             "use_cache_note": (
                 "'config' = nothing passed to generate(), which is what "
                 "eval/predict_lut_ddp.py does.  Interleaved timing on this "
                 "box, warm: 3.3-4.4 s/generation with or without an explicit "
                 "use_cache=True, i.e. no measurable difference."),
             "determinism_probe": determinism,
             "peak_gpu_bytes": (int(torch.cuda.max_memory_allocated())
                                if str(args.device).startswith("cuda") else None),
             "n_rows": n, "device": args.device,
             "reference": ("image2 = that row's own GT after-image "
                           "(oracle-reference condition)"),
             "shuffle_control": ("image2 = another row's GT after-image; seeded "
                                 "derangement, no fixed point"),
             "A_parse": {"true": _stats("true"), "shuffle": _stats("shuffle")},
             "wall_s": time.time() - t0}
    (out / "pst_facts.json").write_text(json.dumps(facts, indent=2), encoding="utf-8")
    print(json.dumps(facts["A_parse"], indent=2)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
