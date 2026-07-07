# S3 (round 3) capture: online eager re-inference of the 34 box test samples under
# language-perturbed instructions (del = referent removed, flip = referent box mirrored).
# orig condition reuses the C0 dumps (same settings). Resumable, one npz per (key, cond).
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2

from common import parse_boxes
from common_r2 import C0_DIR, load_manifest_r2, get_split_r2
from common_r3 import RESULTS_R3
from c0_capture import LensRecorderR2
from run_capture import load_model
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

S3_DIR = os.path.join(RESULTS_R3, "dumps_s3")


def mirror_box(box, axis):
    x1, y1, x2, y2 = box
    if axis == "x":
        return (1 - x2, y1, 1 - x1, y2)
    return (x1, 1 - y2, x2, 1 - y1)


def fmt_box(b):
    return "<box>%.6f %.6f %.6f %.6f</box>" % b


def build_jobs():
    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    keys = sorted(k for k in split["test"] if rows[k]["has_box"]
                  and os.path.exists(os.path.join(C0_DIR, k + ".npz")))
    pert = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "s3_prompts.json")))
    jobs = []
    for k in keys:
        p = pert[k]
        jobs.append(dict(key=k, cond="del", prompt=p["del"], input_path=rows[k]["input_path"]))
        if p["flip"]:
            boxes = parse_boxes(rows[k]["prompt"])
            fb = mirror_box(boxes[0], p["flip_axis"])
            jobs.append(dict(key=k, cond="flip", prompt=p["flip"].replace("{FBOX}", fmt_box(fb)),
                             flip_box=list(fb), flip_axis=p["flip_axis"],
                             input_path=rows[k]["input_path"]))
    return keys, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--dry", action="store_true", help="print perturbed prompts and exit")
    args = ap.parse_args()

    keys, jobs = build_jobs()
    os.makedirs(S3_DIR, exist_ok=True)
    json.dump(jobs, open(os.path.join(RESULTS_R3, "s3_jobs.json"), "w"),
              ensure_ascii=False, indent=1)
    if args.dry:
        for j in jobs:
            print(j["key"], j["cond"], "::", j["prompt"])
        return

    todo = [j for j in jobs
            if not os.path.exists(os.path.join(S3_DIR, f"{j['key']}__{j['cond']}.npz"))]
    print(f"[s3] base keys={len(keys)} jobs={len(jobs)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=True)
    model.lens_track_spans = True
    recorder = LensRecorderR2(save_full_attn=True)
    model.lens_recorder = recorder

    ds = Infer_Style_Dataset(
        img_paths=[j["input_path"] for j in todo],
        prompts=[j["prompt"] for j in todo],
        tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
    collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator)

    grid = model.get_vision_tower().num_patches_per_side
    fails = []
    for bi, batch in enumerate(loader):
        j = todo[bi]
        try:
            recorder.save_full_attn = True
            with torch.inference_mode():
                imgs, texts = model._generate(
                    tokenizer=tokenizer, inputs=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    images=[t.to(torch.bfloat16).cuda() for t in batch["images"]],
                    image_sizes=batch["image_sizes"],
                    retouch_masks=[t.to(torch.bfloat16).cuda() for t in batch["retouch_masks"]],
                    input_imgs=[t.to(torch.bfloat16).cuda() for t in batch["input_imgs"]],
                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    output_hidden_states=True, output_attentions=True,
                    return_dict_in_generate=True, chunk=-1)
            if recorder.last is not None:
                rec = dict(recorder.last)
                rec["grid"] = np.int32(grid)
                np.savez_compressed(os.path.join(S3_DIR, f"{j['key']}__{j['cond']}.npz"), **rec)
                recorder.last = None
            torch.cuda.empty_cache()
            print(f"[s3] {bi+1}/{len(todo)} {j['key']}__{j['cond']}", flush=True)
        except Exception as e:
            fails.append((j["key"], j["cond"], repr(e)))
            print(f"[s3] FAIL {j['key']}__{j['cond']}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    json.dump(fails, open(os.path.join(RESULTS_R3, "s3_fails.json"), "w"), indent=1)
    print(f"[s3] finished, fails={len(fails)}", flush=True)


if __name__ == "__main__":
    main()
