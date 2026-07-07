# S7 (round 3) online capture for the anti-"language parroting" controls:
#   synth    - fixed style phrase x same 16 images (3 styles): does the style feature vary with
#              the image (fusion) or stay constant (parroting)?
#   delstyle - natural style samples with the style WORD removed: does image content alone
#              still activate the feature?
# Captures retouch-token all-layer latents only (no attention). Resumable.
import os, sys, json, argparse, traceback, re

sys.path.insert(0, os.path.expanduser("~/VeraRetouch"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from common_r2 import C0_DIR, load_manifest_r2, get_split_r2
from common_r3 import RESULTS_R3
from s6_capture3 import SeqRecorder  # captures lat_* + seq (we keep lat_* only)
from run_capture import load_model
from data.infer_dataset import Infer_Style_Dataset, DataCollatorForUnifiedTestDataset
from torch.utils.data import DataLoader

S7_DIR = os.path.join(RESULTS_R3, "dumps_s7")

TEMPLATES = {
    "cinematic": {"CN": "把这张照片调出强烈的电影感。",
                  "EN": "Give this photo a strong cinematic look."},
    "cyberpunk": {"CN": "把这张照片调成赛博朋克霓虹风格。",
                  "EN": "Make this photo look cyberpunk with neon colors."},
    "fresh":     {"CN": "把这张照片调成清新日系的风格。",
                  "EN": "Give this photo a fresh, airy style."},
}
# style-word removal patterns for delstyle (residual grammar noise is acceptable)
DEL_PATTERNS = [
    "电影感", "电影级", "影院", "大片感", "赛博朋克", "赛博", "霓虹", "清新", "日系",
    "复古", "怀旧", "年代感", "胶片", "胶卷", "菲林", "朦胧", "梦幻", "柔焦",
    "cinematic", "cyberpunk", "neon", "film grain", "retro", "vintage", "nostalgic",
    "dreamy", "hazy", "airy", "fresh", "golden hour", "movie",
]


def strip_style_words(p):
    for w in sorted(DEL_PATTERNS, key=len, reverse=True):
        p = re.sub(re.escape(w), "", p, flags=re.IGNORECASE)
    p = re.sub(r"[,，]\s*[,，]", "，", p)
    p = re.sub(r"\s{2,}", " ", p)
    p = re.sub(r"的的+", "的", p)
    return p.strip()


def build_jobs(n_imgs=16, n_del_per_style=8, seed=0):
    rows = {r["key"]: r for r in load_manifest_r2()}
    split = get_split_r2()
    lab = json.load(open(os.path.join(RESULTS_R3, "s7_labels.json")))
    labels, styles = lab["labels"], lab["styles"]
    rng = np.random.default_rng(seed)

    # neutral images: test-split, captured, not style-labeled (multi-label ones excluded too
    # is not knowable here; single-label map suffices as a filter plus template overwrite)
    neutral = [k for k in split["test"] if k not in labels
               and os.path.exists(os.path.join(C0_DIR, k + ".npz"))]
    cn = [k for k in neutral if k.startswith("CN_")][: n_imgs // 2]
    en = [k for k in neutral if k.startswith("EN_")][: n_imgs // 2]
    imgs = cn + en

    jobs = []
    for style in TEMPLATES:
        for k in imgs:
            lang = "CN" if k.startswith("CN_") else "EN"
            jobs.append(dict(kind="synth", style=style, base_key=k,
                             uid=f"synth__{style}__{k}",
                             prompt=TEMPLATES[style][lang],
                             input_path=rows[k]["input_path"]))
    for style in styles:
        ks = sorted(k for k, v in labels.items() if v == style)
        rng.shuffle(ks)
        for k in ks[:n_del_per_style]:
            jobs.append(dict(kind="delstyle", style=style, base_key=k,
                             uid=f"delstyle__{style}__{k}",
                             prompt=strip_style_words(rows[k]["prompt"]),
                             input_path=rows[k]["input_path"]))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=1200)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    jobs = build_jobs()
    os.makedirs(S7_DIR, exist_ok=True)
    json.dump(jobs, open(os.path.join(RESULTS_R3, "s7_capture_jobs.json"), "w"),
              ensure_ascii=False, indent=1)
    if args.dry:
        for j in jobs:
            print(j["uid"], "::", j["prompt"])
        return
    todo = [j for j in jobs if not os.path.exists(os.path.join(S7_DIR, j["uid"] + ".npz"))]
    print(f"[s7cap] jobs={len(jobs)} todo={len(todo)}", flush=True)
    if not todo:
        return

    model, tokenizer = load_model(args.max_new_tokens, eager_attn=False)
    model.lens_track_spans = True
    recorder = SeqRecorder()
    model.lens_recorder = recorder

    ds = Infer_Style_Dataset(
        img_paths=[j["input_path"] for j in todo],
        prompts=[j["prompt"] for j in todo],
        tokenizer=tokenizer, image_processor=model.get_vision_tower().image_processor)
    collator = DataCollatorForUnifiedTestDataset(tokenizer=tokenizer)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=4, collate_fn=collator)

    fails = []
    for bi, batch in enumerate(loader):
        j = todo[bi]
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
                rec = {k: v for k, v in recorder.last.items() if not k.startswith("seq_")}
                np.savez_compressed(os.path.join(S7_DIR, j["uid"] + ".npz"), **rec)
                recorder.last = None
            torch.cuda.empty_cache()
            print(f"[s7cap] {bi+1}/{len(todo)} {j['uid']}", flush=True)
        except Exception as e:
            fails.append((j["uid"], repr(e)))
            print(f"[s7cap] FAIL {j['uid']}: {e}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    json.dump(fails, open(os.path.join(RESULTS_R3, "s7_capture_fails.json"), "w"), indent=1)
    print(f"[s7cap] finished, fails={len(fails)}", flush=True)


if __name__ == "__main__":
    main()
