"""RO-2 · logit lens 概率场导出（EXPERIMENTS_v3 §2.1 RO-2 行 / DOSSIER §5.3）。

把每个 image token 的隐状态经**终端 RMSNorm** 投到词表，取目标语义词的概率当空间场：

    p(w | 位置 p, 层 l) = softmax_V( lm_head( model.model.norm( h_l[p] ) ) )[w]

**本项目模型的实测适配点**（DOSSIER §5.3 写的是 Qwen2.5-VL，本项目 llava_qwen2 0.5B 不同，
逐条实测见 experiments/RO2_logitlens_20260803/NOTES.md §一）：

- 模块路径：`model.model.norm`（Qwen2RMSNorm, eps=1e-6）+ `model.lm_head`。
  **没有 `language_model` 这一层**——DOSSIER 的 `language_model.norm` 写法在本项目会 AttributeError。
- `tie_word_embeddings = True`（实测 `lm_head.weight is model.embed_tokens.weight`，
  checkpoint 里根本没有 lm_head 权重）。
- **视觉 token 定位不能用 id**：本项目走 LLaVA 占位符 `IMAGE_TOKEN_INDEX = -200`，
  在 `prepare_inputs_labels_for_multimodal` 里就地展开成 256 个 embedding；
  DOSSIER 记的 151652/151653/151655 在本项目 tokenizer 里是
  `<problem_light_start>` / `<problem_globalcolor_start>` / `<problem_light_end>`，**用了就是错的**。
  定位一律用 `model.lens_image_spans`（lens-exp 插桩）。
- **无 deepstack 式二次注入**：mm_projector 只在输入端注入一次，全仓库无 deepstack 结构。

**hidden_states 索引陷阱（实测，transformers 4.57.1）**：`output_hidden_states=True` 返回
25 项，**最后一项已经过 `model.norm`**（实测 ‖h‖ 均值 332 → 114）。若照 DOSSIER 的
`lm_head(norm(hidden_states[l]))` 对 l=24 再 norm 一次就是双重归一化。本模块用
`model.model.norm` 的 forward_pre_hook 截末层的**未归一化**输出。

读出点：`emb`（mm_projector 输出）+ `L0..L23`（第 li 层 decoder 的输出），共 25 个。

**红线**：softmax 只在**词表维**做（这是方法本身）；**空间维不做任何归一化**
（无 min-max / 无分位 / 无逐图 z-score）。落盘的是原始概率。

用法（.venv-lens）：
  CUDA_VISIBLE_DEVICES=1 python tools/readout/ro2_logit_lens.py \
      --jobs-json experiments/RO2_logitlens_20260803/config/ro2_jobs.json \
      --out-dir /home/bc/data/ro2_lens_20260803 [--limit N] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "readout"))

from ro9_gl_attention import (GRID, N_IMG_TOKENS,  # noqa: E402
                              load_model, luma_to_grid)

N_LAYERS = 24
PROMPT_ORDERS = ("image_first", "instr_first")


# ---------------------------------------------------------------------------
# prompt 构造（A2：指令前置对照）
# ---------------------------------------------------------------------------
def build_inputs_ordered(tokenizer, image_processor, img_path: str, instruction: str,
                         prompt_order: str = "image_first"):
    """与 `ro9_gl_attention.build_inputs` **逐字同款**，只多一个 `<image>` 位置开关。

    - `image_first`（部署口径，与 G1/RO-9/RO-3 完全一致）：
        `<image>\n {TASK}\nNow, you are ... Instruction: {instr}`
      image token 早于指令 ⇒ 因果掩码下 image 位置的表示与指令**逐比特无关**。
    - `instr_first`（A2 变体）：
        `{TASK}\nNow, you are ... Instruction: {instr}\n<image>`
      **token 多重集完全相同，只有 `<image>` 的位置变了**——这是最干净的 A/B。
    图像预处理、conv 模板、luma 对齐一律不变。
    """
    import torch
    from PIL import Image

    from llava.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                                 TASK_STYLE_RETOUCH_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images_, tokenizer_image_token

    if prompt_order not in PROMPT_ORDERS:
        raise ValueError(f"prompt_order={prompt_order!r} 不在 {PROMPT_ORDERS}")
    body = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
            "I will provide an image and an instruction, please give me a retouch plan "
            f"and retouch tokens.\n Instruction: {instruction}")
    qs = (DEFAULT_IMAGE_TOKEN + "\n" + body if prompt_order == "image_first"
          else body + "\n" + DEFAULT_IMAGE_TOKEN)
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer,
                                      IMAGE_TOKEN_INDEX, return_tensors="pt")

    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    scale = 512 / min(w, h)
    image_512p = image.resize((int(round(w * scale)), int(round(h * scale))),
                              Image.Resampling.LANCZOS)
    image_tensor = process_images_([image_512p], image_processor)[0]
    luma = np.asarray(image_512p.convert("L"), dtype=np.float32) / 255.0
    return input_ids, image_tensor, image_512p.size, luma
N_READOUT = N_LAYERS + 1            # emb + L0..L23
READOUT_NAMES = ["emb"] + [f"L{i}" for i in range(N_LAYERS)]
CTRL_WORD = "calculator"            # 与任何图像内容无关的对照词

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "with", "to", "for",
    "from", "by", "is", "are", "its", "his", "her", "their", "this", "that",
    "into", "over", "under", "near", "against", "while", "as", "it", "one",
    "two", "front", "back", "side", "part", "image", "photo", "picture",
}
HYPERNYM = {
    "woman": "person", "man": "person", "person": "person", "girl": "person",
    "boy": "person", "child": "person", "baby": "person", "people": "person",
    "couple": "person", "bride": "person", "groom": "person", "model": "person",
    "dog": "animal", "cat": "animal", "bird": "animal", "horse": "animal",
    "fox": "animal", "wolf": "animal", "lion": "animal", "zebra": "animal",
    "eagle": "animal", "flamingo": "animal", "penguin": "animal", "owl": "animal",
    "deer": "animal", "parrot": "animal", "butterfly": "animal", "calf": "animal",
    "jellyfish": "animal", "coral": "animal",
    "flower": "plant", "plant": "plant", "tree": "plant", "leaf": "plant",
    "cactus": "plant",
    "car": "vehicle", "vehicle": "vehicle", "boat": "vehicle", "truck": "vehicle",
    "building": "building",
}
SCHEMES = ["name_last", "name_first", "desc_max", "hyper",
           "compl", "shuf_name_last", "ctrl"]


# ---------------------------------------------------------------------------
# 词 → token id（与分析脚本共用；口径必须一致）
# ---------------------------------------------------------------------------
def word_token_ids(tokenizer, word: str) -> tuple[list[int], bool]:
    """单词 → token id 列表 + 是否退化为多 token 首 token。

    取 {w, ' '+w, W, ' '+W} 里**能编码成单 token**的全部变体（概率相加）。
    一个变体都没有 → 退回 ' '+w 的**首** token（flag=True，报告里单列占比）。
    """
    w = word.strip()
    if not w:
        return [], True
    variants = [w, " " + w, w[:1].upper() + w[1:], " " + w[:1].upper() + w[1:]]
    ids: list[int] = []
    for v in variants:
        t = tokenizer(v, add_special_tokens=False).input_ids
        if len(t) == 1:
            ids.append(int(t[0]))
    if ids:
        return sorted(set(ids)), False
    t = tokenizer(" " + w, add_special_tokens=False).input_ids
    return ([int(t[0])] if t else []), True


def last_word(phrase: str) -> str:
    toks = re.findall(r"[A-Za-z]+", phrase or "")
    return toks[-1].lower() if toks else ""


def content_words(desc: str, limit: int = 12) -> list[str]:
    toks = [t.lower() for t in re.findall(r"[A-Za-z]+", desc or "")]
    out = [t for t in toks if len(t) >= 3 and t not in STOPWORDS]
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq[:limit]


def build_job_words(tokenizer, job: dict) -> dict:
    """→ {scheme: {"words": [...], "ids": [[...], ...], "multitok": bool}}"""
    name = job.get("target_name") or ""
    desc = job.get("target_desc") or ""
    shuf = job.get("shuffle_name") or ""
    spec: dict[str, list[str]] = {
        "name_last": [last_word(name)],
        "name_first": [name.strip()],        # 整短语，取首 token
        "desc_max": content_words(desc),
        "hyper": [HYPERNYM.get(last_word(name), "object")],
        "compl": ["background"],
        "shuf_name_last": [last_word(shuf)] if shuf else [""],
        "ctrl": [CTRL_WORD],
    }
    out = {}
    for sch, words in spec.items():
        ids, flags = [], []
        for w in words:
            if sch == "name_first":
                t = tokenizer(" " + w, add_special_tokens=False).input_ids
                ids.append([int(t[0])] if t else [])
                flags.append(len(t) > 1)
            else:
                i, f = word_token_ids(tokenizer, w)
                ids.append(i)
                flags.append(f)
        out[sch] = {"words": words, "ids": ids, "multitok": bool(any(flags))}
    return out


# ---------------------------------------------------------------------------
# 单样本前向 + lens
# ---------------------------------------------------------------------------
def lens_readout(model, tokenizer, image_processor, img_path: str, instruction: str,
                 job_words: dict, device: str, store_hs: bool,
                 chunk_layers: int = 5, prompt_order: str = "image_first") -> dict:
    import torch

    t0 = time.time()
    input_ids, image_tensor, image_size, luma = build_inputs_ordered(
        tokenizer, image_processor, img_path, instruction, prompt_order)
    input_ids = input_ids.unsqueeze(0).to(device)
    images = [image_tensor.to(torch.bfloat16).to(device)]

    model.lens_image_spans = []
    cap: dict = {}
    hook = model.get_model().norm.register_forward_pre_hook(
        lambda mod, args: cap.__setitem__("pre_norm", args[0].detach()))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, images=images, image_sizes=[image_size],
                        output_hidden_states=True, use_cache=False, return_dict=True)
    finally:
        hook.remove()

    hs = out.hidden_states
    if len(hs) != N_LAYERS + 1:
        raise RuntimeError(f"hidden_states 长度 {len(hs)} != {N_LAYERS + 1}")
    span = model.lens_image_spans[0]
    img_start, img_end, total = span
    if img_end - img_start != N_IMG_TOKENS:
        raise RuntimeError(f"image span {span} != {N_IMG_TOKENS} tokens")

    # 顺序守卫：实测 span 必须与请求的 prompt_order 一致（写错顺序会静默给出错误结论）
    if prompt_order == "image_first":
        # 部署口径：image token 早于指令文本 ⇒ 因果掩码下表示与指令逐比特无关
        if img_end >= total:
            raise RuntimeError(f"image_first 下 image span {span} 触及序列尾部——顺序异常")
    else:
        # A2 变体：image token 必须在指令**之后**（前面要有足量文本）
        if img_start <= 20:
            raise RuntimeError(f"instr_first 下 image span {span} 起点过早——顺序异常")
        if total - img_end > 30:
            raise RuntimeError(f"instr_first 下 image 之后仍有 {total-img_end} 个 token——顺序异常")

    # 读出点：emb=hs[0]；L{li}=hs[li+1]（li<23）；L23=终端 norm 的输入（**未归一化**）
    raw = [hs[0][0, img_start:img_end]]
    raw += [hs[i + 1][0, img_start:img_end] for i in range(N_LAYERS - 1)]
    raw.append(cap["pre_norm"][0, img_start:img_end])
    assert len(raw) == N_READOUT

    norm = model.get_model().norm
    W = model.lm_head.weight            # tied to embed_tokens（实测）
    rawnorm = np.stack([r.float().norm(dim=-1).cpu().numpy() for r in raw])  # (25,256)

    # 需要的 token id 全集（去重）→ 一次 gather
    all_ids: list[int] = []
    for sch in SCHEMES:
        for lst in job_words[sch]["ids"]:
            all_ids.extend(lst)
    uniq = sorted(set(all_ids))
    idx_of = {t: i for i, t in enumerate(uniq)}
    id_tensor = torch.tensor(uniq, device=device, dtype=torch.long)

    pn_all, lse_all, sel_all = [], [], []
    with torch.inference_mode():
        for c0 in range(0, N_READOUT, chunk_layers):
            blk = raw[c0:c0 + chunk_layers]
            h = norm(torch.stack(blk))                        # (b,256,896) bf16
            hf = h.float()
            logits = torch.nn.functional.linear(hf, W.float())  # (b,256,V) fp32
            lse_all.append(torch.logsumexp(logits, dim=-1).cpu().numpy())
            sel_all.append(logits.index_select(-1, id_tensor).cpu().numpy())
            if store_hs:
                pn_all.append(h.to(torch.float16).cpu().numpy())
            del logits, hf, h
    lse = np.concatenate(lse_all, axis=0).astype(np.float32)     # (25,256)
    sel = np.concatenate(sel_all, axis=0).astype(np.float32)     # (25,256,U)
    probs_uniq = np.exp(sel - lse[..., None])                    # (25,256,U)

    # 各 scheme 的场：同一个词的多变体**相加**；desc_max 在词间取 **max**
    fields = np.zeros((N_READOUT, N_IMG_TOKENS, len(SCHEMES)), dtype=np.float32)
    for si, sch in enumerate(SCHEMES):
        per_word = []
        for lst in job_words[sch]["ids"]:
            if not lst:
                continue
            cols = [idx_of[t] for t in lst]
            per_word.append(probs_uniq[..., cols].sum(-1))
        if not per_word:
            continue
        stk = np.stack(per_word, axis=0)
        fields[..., si] = stk.max(0) if sch == "desc_max" else stk[0]

    luma16, valid16 = luma_to_grid(luma)
    res = {
        "fields": fields.reshape(N_READOUT, GRID, GRID, len(SCHEMES)),
        "lse": lse, "rawnorm": rawnorm, "luma16": luma16, "valid16": valid16,
        "span": np.array(span, dtype=np.int64),
        "prompt_order": prompt_order,
        "t": time.time() - t0,
    }
    if store_hs:
        res["hs_pn"] = np.concatenate(pn_all, axis=0)            # (25,256,896) fp16
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default="/home/bc/data/models/VeraRetouch")
    ap.add_argument("--config-add", default=str(REPO / "configs" / "infer_config.yaml"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batches", default="", help="逗号分隔，只跑这些 batch")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--chunk-layers", type=int, default=5)
    ap.add_argument("--prompt-order", default="image_first", choices=list(PROMPT_ORDERS),
                    help="image_first = 部署口径；instr_first = A2 指令前置对照")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs_json).read_text())
    if args.batches:
        want = set(args.batches.split(","))
        jobs = [j for j in jobs if j["batch"] in want]
    if args.limit > 0:
        jobs = jobs[: args.limit]

    out_dir = Path(args.out_dir)
    (out_dir / "npz").mkdir(parents=True, exist_ok=True)
    model, tokenizer, image_processor, _ = load_model(
        args.model_path, args.config_add, args.device)

    # 元信息落盘（审阅可核）
    import torch
    meta = {
        "tie_word_embeddings": bool(model.config.tie_word_embeddings),
        "lm_head_is_embed_tokens": bool(
            model.lm_head.weight is model.get_model().embed_tokens.weight),
        "norm_type": type(model.get_model().norm).__name__,
        "norm_eps": float(model.get_model().norm.variance_epsilon),
        "n_layers": N_LAYERS, "readout_names": READOUT_NAMES,
        "vocab_size": int(model.config.vocab_size),
        "attn_impl": model.config._attn_implementation,
        "schemes": SCHEMES, "ctrl_word": CTRL_WORD,
        "prompt_order": args.prompt_order,
        "torch": torch.__version__,
    }
    (out_dir / "export_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    log = open(out_dir / "run_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    n_done = n_skip = n_err = 0
    for i, job in enumerate(jobs):
        dst = out_dir / "npz" / f"{job['key']}.npz"
        if args.skip_existing and dst.is_file():
            n_skip += 1
            continue
        try:
            jw = build_job_words(tokenizer, job)
            r = lens_readout(model, tokenizer, image_processor, job["img_path"],
                             job["instruction"], jw, args.device,
                             bool(job.get("store_hs")), args.chunk_layers,
                             args.prompt_order)
        except Exception as e:                      # 单样本失败不拖垮批次
            n_err += 1
            log.write(json.dumps({"key": job["key"], "error": repr(e)}) + "\n")
            log.flush()
            continue
        payload = {k: v for k, v in r.items() if k != "t"}
        payload["prompt_order"] = args.prompt_order
        payload["words"] = json.dumps(jw, ensure_ascii=False)
        payload["job"] = json.dumps({k: v for k, v in job.items()
                                     if k not in ("img_path",)}, ensure_ascii=False)
        tmp = dst.with_name(dst.name + ".tmp.npz")   # savez 会强行补 .npz 后缀
        np.savez_compressed(tmp, **payload)
        tmp.rename(dst)
        n_done += 1
        log.write(json.dumps({"key": job["key"], "batch": job["batch"],
                              "t": round(r["t"], 3),
                              "multitok": {s: jw[s]["multitok"] for s in SCHEMES}}) + "\n")
        if (i + 1) % 50 == 0:
            el = time.time() - t_start
            print(f"[{i+1}/{len(jobs)}] done={n_done} skip={n_skip} err={n_err} "
                  f"elapsed={el/60:.1f}min eta={el/max(n_done,1)*(len(jobs)-i-1)/60:.1f}min",
                  flush=True)
            log.flush()
    log.close()
    print(f"FINISHED jobs={len(jobs)} done={n_done} skipped={n_skip} errors={n_err} "
          f"wall={(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
