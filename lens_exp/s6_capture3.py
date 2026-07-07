# S6 (round 3) capture: full-sequence activations at layers 11+14+23 (one forward pass,
# self-consistent for multi-layer SAE concat; C0 only stored L11/L23). No attention capture.
# Resumable; --keys-file supports multi-GPU sharding (designed to run on exp-remote).
import os, sys, json, argparse, traceback

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common_r2 import RESULTS_R2, load_manifest_r2
from run_capture import load_model
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

SEQ_LAYERS = (11, 14, 23)
OUT_DIR = os.path.join(RESULTS_R2, "dumps", "s6_seq3")


class SeqRecorder:
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
            out[f"step_{name}"] = np.int32(step)
        out["prompt_len"] = np.int32(hidden_states[0][0].shape[1])
        out["gen_len"] = np.int32(output_ids.shape[1])
        n_steps = len(hidden_states)
        for l in SEQ_LAYERS:
            seq = torch.cat([hidden_states[0][l][i]] +
                            [hidden_states[t][l][i] for t in range(1, n_steps)], dim=0)
            out[f"seq_l{l}"] = seq.float().cpu().numpy().astype(np.float16)
        self.last = out
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keys-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    args = ap.parse_args()

    rows = load_manifest_r2()
    if args.keys_file:
        keep = {l.strip() for l in open(args.keys_file) if l.strip()}
        rows = [r for r in rows if r["key"] in keep]
    os.makedirs(OUT_DIR, exist_ok=True)
    todo = [r for r in rows if not os.path.exists(os.path.join(OUT_DIR, r["key"] + ".npz"))]
    print(f"[s6cap] total={len(rows)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=False)
    model.lens_track_spans = True
    recorder = SeqRecorder()
    model.lens_recorder = recorder

    ds = Infer_Style_Dataset(
        img_paths=[r["input_path"] for r in todo],
        prompts=[r["prompt"] for r in todo],
        tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
    collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator)

    fails = []
    for bi, batch in enumerate(loader):
        r = todo[bi]
        try:
            with torch.inference_mode():
                model._generate(
                    tokenizer=tokenizer, inputs=batch["input_ids"].cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    images=[t.to(torch.bfloat16).cuda() for t in batch["images"]],
                    image_sizes=batch["image_sizes"],
                    retouch_masks=[t.to(torch.bfloat16).cuda() for t in batch["retouch_masks"]],
                    input_imgs=[t.to(torch.bfloat16).cuda() for t in batch["input_imgs"]],
                    do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                    output_hidden_states=True, return_dict_in_generate=True, chunk=-1)
            if recorder.last is not None:
                np.savez_compressed(os.path.join(OUT_DIR, r["key"] + ".npz"), **recorder.last)
                recorder.last = None
            torch.cuda.empty_cache()
            if (bi + 1) % 10 == 0:
                print(f"[s6cap] {bi+1}/{len(todo)} ({r['key']})", flush=True)
        except Exception as e:
            fails.append((r["key"], repr(e)))
            print(f"[s6cap] FAIL {r['key']}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    print(f"[s6cap] finished, fails={len(fails)}", flush=True)
    if fails:
        json.dump(fails, open(os.path.join(RESULTS_R2, "fails_s6cap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
