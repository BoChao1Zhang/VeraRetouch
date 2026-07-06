# lens-exp capture driver: runs VeraRetouch in style mode over the manifest,
# dumping all-layer retouch-token latents (+ optional attentions) and baseline preds.
# Resumable: skips samples whose outputs already exist.
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2
import yaml
from box import Box

from llava.utils import disable_torch_init
from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
from llava.constants import (DEFAULT_RETOUCH_LIGHT_TOKEN, DEFAULT_RETOUCH_COLORTEMP_TOKEN,
                             DEFAULT_RETOUCH_COLORMIXER_TOKEN)
from transformers import AutoTokenizer
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

from common import RESULTS, DUMPS, build_manifest, load_manifest
from recorder import LensRecorder

MODEL_PATH = os.path.expanduser("~/data/models/VeraRetouch")
CFG_ADD = os.path.expanduser("~/VeraRetouch/configs/infer_config.yaml")


def load_model(max_new_tokens=1200, eager_attn=False):
    with open(CFG_ADD, "r", encoding="utf-8") as f:
        config_add = Box(yaml.safe_load(f))
    config_add.project_name = "lens_exp"
    config_add.freeze_retouch_decoder = True
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, cache_dir="./cache", model_max_length=max_new_tokens,
        padding_side="right", use_fast=False)
    extra = {"attn_implementation": "eager"} if eager_attn else {}
    model = VeraRetouchForCausalLLM_Unified.from_pretrained(
        MODEL_PATH, config_add=config_add, cache_dir="./cache",
        torch_dtype=torch.bfloat16, **extra).cuda()
    ids = [tokenizer(t, add_special_tokens=False).input_ids[0] for t in
           (DEFAULT_RETOUCH_LIGHT_TOKEN, DEFAULT_RETOUCH_COLORTEMP_TOKEN,
            DEFAULT_RETOUCH_COLORMIXER_TOKEN)]
    model.register_special_token_idx(*ids)
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="run tag: baseline / attn / readoutN")
    ap.add_argument("--attn", action="store_true")
    ap.add_argument("--readout", type=int, default=-1, help="hidden_states tuple index for readout (default -1 = original)")
    ap.add_argument("--dump-latents", action="store_true")
    ap.add_argument("--only-box", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    man_path = os.path.join(RESULTS, "manifest.json")
    if not os.path.exists(man_path):
        rows = build_manifest()
        with open(man_path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
    rows = load_manifest(man_path)
    if args.only_box:
        rows = [r for r in rows if r["has_box"]]
    if args.limit:
        rows = rows[: args.limit]

    pred_dir = os.path.join(RESULTS, f"preds_{args.tag}")
    text_dir = os.path.join(RESULTS, f"texts_{args.tag}")
    dump_dir = os.path.join(DUMPS, args.tag)
    for d in (pred_dir, text_dir, dump_dir):
        os.makedirs(d, exist_ok=True)

    # resume: keep only samples missing an output
    def done(r):
        p_ok = os.path.exists(os.path.join(pred_dir, r["key"] + ".png"))
        d_ok = (not (args.dump_latents or args.attn)) or os.path.exists(os.path.join(dump_dir, r["key"] + ".npz"))
        return p_ok and d_ok
    todo = [r for r in rows if not done(r)]
    print(f"[lens] tag={args.tag} total={len(rows)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=args.attn)
    model.lens_readout_layer = args.readout
    model.lens_track_spans = True
    recorder = None
    if args.dump_latents or args.attn:
        recorder = LensRecorder(capture_attn=args.attn)
        model.lens_recorder = recorder

    ds = Infer_Style_Dataset(
        img_paths=[r["input_path"] for r in todo],
        prompts=[r["prompt"] for r in todo],
        tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
    collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator)

    grid = model.get_vision_tower().num_patches_per_side
    print(f"[lens] patch grid = {grid}x{grid}", flush=True)
    fails = []
    for bi, batch in enumerate(loader):
        r = todo[bi]
        try:
            input_ids = batch["input_ids"].cuda()
            images = [t.to(torch.bfloat16).cuda() for t in batch["images"]]
            attention_mask = batch["attention_mask"].cuda()
            input_imgs = [t.to(torch.bfloat16).cuda() for t in batch["input_imgs"]]
            retouch_masks = [t.to(torch.bfloat16).cuda() for t in batch["retouch_masks"]]
            gen_kwargs = dict(
                do_sample=False, num_beams=1,
                max_new_tokens=args.max_new_tokens,
                output_hidden_states=True, return_dict_in_generate=True, chunk=-1)
            if args.attn:
                gen_kwargs["output_attentions"] = True
            with torch.inference_mode():
                imgs, texts = model._generate(
                    tokenizer=tokenizer, inputs=input_ids,
                    attention_mask=attention_mask, images=images,
                    image_sizes=batch["image_sizes"], retouch_masks=retouch_masks,
                    input_imgs=input_imgs, **gen_kwargs)
            cv2.imwrite(os.path.join(pred_dir, r["key"] + ".png"), imgs[0])
            with open(os.path.join(text_dir, r["key"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(texts[0])
            if recorder is not None and recorder.last is not None:
                rec = dict(recorder.last)
                rec["grid"] = np.int32(grid)
                np.savez_compressed(os.path.join(dump_dir, r["key"] + ".npz"), **rec)
                recorder.last = None
            if (bi + 1) % 10 == 0:
                print(f"[lens] {bi+1}/{len(todo)} done ({r['key']})", flush=True)
        except Exception as e:
            fails.append((r["key"], repr(e)))
            print(f"[lens] FAIL {r['key']}: {e}", flush=True)
            traceback.print_exc()
    print(f"[lens] finished. fails={len(fails)}", flush=True)
    if fails:
        with open(os.path.join(RESULTS, f"fails_{args.tag}.json"), "w") as f:
            json.dump(fails, f, indent=1)


if __name__ == "__main__":
    main()
