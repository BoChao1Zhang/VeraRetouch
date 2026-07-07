# C0 (round 2): unified capture over ALL Lr samples (~798), eager attention.
# Per-sample npz (resumable, one file per sample):
#   lat_{tok}            [25, 896]  fp16   all-layer retouch-token latents (E6)
#   att_{tok}_headagg    [24, 14, 2] fp32  per-head (mean, max) over image patches, "self" step (E7 screening)
#   att_{tok}_self_full  [24, 14, 256] fp16  full per-head rows, ONLY for <box> samples (E7/E8)
#   seq_l11 / seq_l23    [prompt+gen-1, 896] fp16  full-sequence activations (E9 SAE corpus)
#   prompt_len, step_{tok}, img_span, gen_len, grid
# Also writes baseline preds/texts (E8 global render reference).
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import cv2

from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2
from run_capture import load_model
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

SEQ_LAYERS = (11, 23)


class LensRecorderR2:
    """Round-2 recorder: latents + per-head attention + full-seq L11/L23."""

    def __init__(self, save_full_attn=False):
        self.save_full_attn = save_full_attn
        self.last = None

    def record(self, sample_in_batch, hidden_states, attentions, output_ids,
               token_steps, image_spans):
        i = sample_in_batch
        out = {}
        L = len(hidden_states[0])  # 25

        for name, step in token_steps.items():
            lat = torch.stack([hidden_states[step][l][i].reshape(-1) for l in range(L)])
            out[f"lat_{name}"] = lat.float().cpu().numpy().astype(np.float16)
            out[f"step_{name}"] = np.int32(step)

        prompt_len = hidden_states[0][0].shape[1]
        out["prompt_len"] = np.int32(prompt_len)
        out["gen_len"] = np.int32(output_ids.shape[1])
        if image_spans is not None:
            s, e, tot = image_spans[i]
            out["img_span"] = np.array([s, e, tot], dtype=np.int32)
        else:
            s = e = None

        # full-sequence activations at SEQ_LAYERS: prompt pass + each gen step
        # (position p of step t>=1 is prompt_len + t - 1, i.e. the token that step consumed)
        n_steps = len(hidden_states)
        for l in SEQ_LAYERS:
            seq = torch.cat([hidden_states[0][l][i]] +
                            [hidden_states[t][l][i] for t in range(1, n_steps)], dim=0)
            out[f"seq_l{l}"] = seq.float().cpu().numpy().astype(np.float16)

        # retouch-token attention over image patches, "self" step (token itself is query)
        if attentions is not None and s is not None:
            n_att = len(attentions)
            for name, step in token_steps.items():
                st = step + 1  # forward where the retouch token is the query
                if st >= n_att or attentions[st] is None or attentions[st][0] is None:
                    continue
                layers = [attentions[st][l][i][:, -1, s:e] for l in range(len(attentions[st]))]
                att = torch.stack(layers).float()          # [24, 14, n_img]
                agg = torch.stack([att.mean(dim=2), att.max(dim=2).values], dim=-1)
                out[f"att_{name}_headagg"] = agg.cpu().numpy().astype(np.float32)
                out[f"att_{name}_mass"] = att.sum(dim=2).cpu().numpy().astype(np.float32)  # [24,14] mass on image
                if self.save_full_attn:
                    out[f"att_{name}_self_full"] = att.cpu().numpy().astype(np.float16)
        self.last = out
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys-file", default=None, help="only process keys listed in this file (one per line)")
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    rows = load_manifest_r2()
    if args.keys_file:
        keep = {l.strip() for l in open(args.keys_file) if l.strip()}
        rows = [r for r in rows if r["key"] in keep]
    if args.limit:
        # smoke: take a mix guaranteeing box + CN + EN coverage
        box = [r for r in rows if r["has_box"]][:2]
        plain = [r for r in rows if not r["has_box"]]
        rows = box + [p for p in plain if p["lang"] == "CN"][:2] + [p for p in plain if p["lang"] == "EN"][:1]
        rows = rows[: args.limit]

    pred_dir = os.path.join(RESULTS_R2, "preds_c0")
    text_dir = os.path.join(RESULTS_R2, "texts_c0")
    for d in (pred_dir, text_dir, C0_DIR):
        os.makedirs(d, exist_ok=True)

    def done(r):
        return (os.path.exists(os.path.join(pred_dir, r["key"] + ".png"))
                and os.path.exists(os.path.join(C0_DIR, r["key"] + ".npz")))
    todo = [r for r in rows if not done(r)]
    print(f"[c0] total={len(rows)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=True)
    model.lens_track_spans = True
    recorder = LensRecorderR2()
    model.lens_recorder = recorder

    ds = Infer_Style_Dataset(
        img_paths=[r["input_path"] for r in todo],
        prompts=[r["prompt"] for r in todo],
        tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
    collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator)

    grid = model.get_vision_tower().num_patches_per_side
    print(f"[c0] patch grid = {grid}x{grid}", flush=True)
    fails = []
    fails_path = os.path.join(RESULTS_R2, "fails_c0.json")
    if os.path.exists(fails_path):
        fails = [tuple(x) for x in json.load(open(fails_path))]
    for bi, batch in enumerate(loader):
        r = todo[bi]
        try:
            recorder.save_full_attn = bool(r["has_box"])
            input_ids = batch["input_ids"].cuda()
            images = [t.to(torch.bfloat16).cuda() for t in batch["images"]]
            attention_mask = batch["attention_mask"].cuda()
            input_imgs = [t.to(torch.bfloat16).cuda() for t in batch["input_imgs"]]
            retouch_masks = [t.to(torch.bfloat16).cuda() for t in batch["retouch_masks"]]
            with torch.inference_mode():
                imgs, texts = model._generate(
                    tokenizer=tokenizer, inputs=input_ids,
                    attention_mask=attention_mask, images=images,
                    image_sizes=batch["image_sizes"], retouch_masks=retouch_masks,
                    input_imgs=input_imgs,
                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    output_hidden_states=True, output_attentions=True,
                    return_dict_in_generate=True, chunk=-1)
            cv2.imwrite(os.path.join(pred_dir, r["key"] + ".png"), imgs[0])
            with open(os.path.join(text_dir, r["key"] + ".txt"), "w", encoding="utf-8") as f:
                f.write(texts[0])
            if recorder.last is not None:
                rec = dict(recorder.last)
                rec["grid"] = np.int32(grid)
                np.savez_compressed(os.path.join(C0_DIR, r["key"] + ".npz"), **rec)
                recorder.last = None
            torch.cuda.empty_cache()
            if (bi + 1) % 10 == 0:
                print(f"[c0] {bi+1}/{len(todo)} done ({r['key']})", flush=True)
                json.dump(fails, open(fails_path, "w"), indent=1)
        except Exception as e:
            fails.append((r["key"], repr(e)))
            print(f"[c0] FAIL {r['key']}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    print(f"[c0] finished. fails={len(fails)}", flush=True)
    json.dump(fails, open(fails_path, "w"), indent=1)


if __name__ == "__main__":
    main()
