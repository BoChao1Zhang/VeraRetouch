# E7 (round 2) head-rescale intervention (V-SEAM style): amplify positive heads (1+lambda),
# suppress negative heads (1-lambda) on image-token attention columns, renormalize; online eager
# inference on the box test subset. Measures paired deltaE00 vs the C0 baseline + latent drift.
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import cv2

from transformers.models.qwen2 import modeling_qwen2 as qwen2_mod

from common_r2 import RESULTS_R2, C0_DIR, load_manifest_r2, get_split_r2
from run_capture import load_model
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

# ---------------- eager attention patch ----------------

class HeadScaleCfg:
    active = False
    scales = {}       # layer_idx -> torch tensor [num_heads]
    span = (0, 0)     # image token span (s, e) in kv positions

CFG = HeadScaleCfg()
_orig_eager = qwen2_mod.eager_attention_forward


def patched_eager_attention_forward(module, query, key, value, attention_mask,
                                    scaling, dropout=0.0, **kwargs):
    key_states = qwen2_mod.repeat_kv(key, module.num_key_value_groups)
    value_states = qwen2_mod.repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if CFG.active and module.layer_idx in CFG.scales:
        s, e = CFG.span
        sc = CFG.scales[module.layer_idx].to(attn_weights.device, attn_weights.dtype)
        w = attn_weights.clone()
        w[:, :, :, s:e] = w[:, :, :, s:e] * sc.view(1, -1, 1, 1)
        attn_weights = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


class MiniRecorder:
    """capture only all-layer retouch-token latents (for drift measurement)."""
    def __init__(self):
        self.last = None

    def record(self, sample_in_batch, hidden_states, attentions, output_ids,
               token_steps, image_spans):
        i = sample_in_batch
        out = {}
        L = len(hidden_states[0])
        for name, step in token_steps.items():
            lat = torch.stack([hidden_states[step][l][i].reshape(-1) for l in range(L)])
            out[f"lat_{name}"] = lat.float().cpu().numpy().astype(np.float16)
        self.last = out


def build_scales(heads_json, lam, num_heads=14):
    pos, neg = set(), set()
    for t in ("light", "colortemp", "colormixer"):
        pos |= set(map(tuple, heads_json[t]["pos"]))
        neg |= set(map(tuple, heads_json[t]["neg"]))
    neg -= pos
    scales = {}
    for (l, h) in pos:
        scales.setdefault(l, torch.ones(num_heads))[h] = 1.0 + lam
    for (l, h) in neg:
        scales.setdefault(l, torch.ones(num_heads))[h] = 1.0 - lam
    return scales, sorted(pos), sorted(neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lams", type=float, nargs="*", default=[0.3, 0.6])
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    qwen2_mod.eager_attention_forward = patched_eager_attention_forward

    heads = json.load(open(os.path.join(RESULTS_R2, "e7_heads.json")))
    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    keys = sorted(k for k in split["test"] if rows[k]["has_box"]
                  and os.path.exists(os.path.join(C0_DIR, k + ".npz")))[: args.max_samples]
    print(f"[e7int] box test samples: {len(keys)}", flush=True)

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=True)
    model.lens_track_spans = True
    rec = MiniRecorder()
    model.lens_recorder = rec

    for lam in args.lams:
        tag = f"e7int_lam{lam:g}"
        scales, pos, neg = build_scales(heads, lam)
        pred_dir = os.path.join(RESULTS_R2, f"preds_{tag}")
        lat_dir = os.path.join(RESULTS_R2, "dumps", tag)
        os.makedirs(pred_dir, exist_ok=True); os.makedirs(lat_dir, exist_ok=True)
        todo = [k for k in keys if not os.path.exists(os.path.join(pred_dir, k + ".png"))]
        print(f"[e7int] lam={lam} pos={len(pos)} neg={len(neg)} todo={len(todo)}", flush=True)
        if not todo:
            continue
        ds = Infer_Style_Dataset(
            img_paths=[rows[k]["input_path"] for k in todo],
            prompts=[rows[k]["prompt"] for k in todo],
            tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
        collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, collate_fn=collator)
        for bi, batch in enumerate(loader):
            k = todo[bi]
            try:
                d0 = np.load(os.path.join(C0_DIR, k + ".npz"))
                s, e, _ = d0["img_span"]; d0.close()
                CFG.scales = scales; CFG.span = (int(s), int(e)); CFG.active = True
                input_ids = batch["input_ids"].cuda()
                images = [t.to(torch.bfloat16).cuda() for t in batch["images"]]
                with torch.inference_mode():
                    imgs, texts = model._generate(
                        tokenizer=tokenizer, inputs=input_ids,
                        attention_mask=batch["attention_mask"].cuda(), images=images,
                        image_sizes=batch["image_sizes"],
                        retouch_masks=[t.to(torch.bfloat16).cuda() for t in batch["retouch_masks"]],
                        input_imgs=[t.to(torch.bfloat16).cuda() for t in batch["input_imgs"]],
                        do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                        output_hidden_states=True, return_dict_in_generate=True, chunk=-1)
                CFG.active = False
                cv2.imwrite(os.path.join(pred_dir, k + ".png"), imgs[0])
                if rec.last is not None:
                    np.savez_compressed(os.path.join(lat_dir, k + ".npz"), **rec.last)
                    rec.last = None
                torch.cuda.empty_cache()
                if (bi + 1) % 10 == 0:
                    print(f"[e7int] lam={lam} {bi+1}/{len(todo)}", flush=True)
            except Exception as ex:
                CFG.active = False
                print(f"[e7int] FAIL {k}: {ex}", flush=True)
                traceback.print_exc()
                torch.cuda.empty_cache()
    print("[e7int] done", flush=True)


if __name__ == "__main__":
    main()
