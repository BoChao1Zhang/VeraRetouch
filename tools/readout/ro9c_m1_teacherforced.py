"""RO-9c · 2026-06-09「M1_diffLMM」主体读出的**逐字复现**（teacher-forced special token）。

复现对象：回收站里的 `research/attention_grounding/scripts/veraretouch_special_token_attn.py`
（2026-06-09，单图 `sample_flower.jpg` 定性）。本工具**照抄它的读出逻辑**，只把「一张图」
换成 G1 已冻结的 S-val 区域对立批（214 源），并把逐层分量原样落盘以便任何人复算。

原脚本的读出（逐行对应，不做任何发明）：

    A        = stack([a[0] for a in out.attentions])   # [L, H, seq, seq]  post-softmax
    red      = A.mean(0).mean(0)                       # 层平均 → 头平均
    red_img  = red[:, img_block]                       # 只留 256 个 image patch 列
    post     = arange(img_block[-1] + 1, seq)          # 图像之后的所有 query 行
    common   = red_img[post].mean(0)                   # ← M1/DiffLMM 的共模（序列均值）
    M1(tok)  = red_img[tok_row] - common
    raw(tok) = red_img[tok_row]

与 DiffLMM 论文（arXiv:2410.08209, ICCV 2025）Eq.3 的关系：论文的
`A_i^norm = A_i^reduced − (1/r) Σ_j A_j^reduced` 沿**输出 token 轴**取均值；
6-09 脚本落地成「**图像之后的所有 query 行**」的均值（teacher-forced 探针里没有生成段）。
本工具**忠实 6-09**，同时落盘 `n_post_rows` 与三 token 的 raw 行，使
「只用 prompt 尾部行做共模」的鲁棒性变体可在分析侧解析地导出，无需重跑（见 NOTES D-2）。

硬规则（CLAUDE.md 红线）：
- attention 导出必须 **eager**：加载即断言 `config._attn_implementation == "eager"`，
  前向后断言 `out.attentions is not None`；FA2/SDPA 返回 None **直接 raise，不回退**。
- 落盘的是**未归一化的原始分量**（post-softmax 概率的层/头平均）。原脚本 `render()` 里的
  `m-=m.min(); m/=m.max()` 是**逐图 min-max（红线）**，只属于出图着色，绝不进入本工具。
- 不改共享模型目录：原脚本会 `os.rename` 掉 `generation_config.json`——本工具**不调用
  generate()**，因此完全不需要动它（见 NOTES §核实-3）。

用法：
  python tools/readout/ro9c_m1_teacherforced.py \
      --region-json experiments/G1_s_identifiability_20260803/config/g1_region_opp.json \
      --out-dir experiments/RO9c_subject_repro_20260805/run \
      --conditions auto reg_a reg_b fixed --device cuda:1
"""
from __future__ import annotations

# ⚑ sqlite3 必须在 torch 之前 import（战役 bug R6：libstdc++ 符号顺序）
import sqlite3  # noqa: F401  isort:skip
import argparse
import json
import os
import sys
import time
from pathlib import Path

# 本机 CPU 长期超载（load ~10 / 48 核，且卡 0 有作业）。必须在 import torch 之前设。
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ro9_gl_attention import (  # noqa: E402  复用 G1 的加载/对齐/D-0，零重写
    GRID,
    N_IMG_TOKENS,
    N_LAYERS,
    TOKEN_TAGS,
    load_model,
    luma_to_grid,
    outlier_mask_from_norms,
)

MODEL_PATH_DEFAULT = "/home/bc/data/models/VeraRetouch"
CONFIG_ADD_DEFAULT = str(REPO / "configs" / "infer_config.yaml")

# 固定短语对照（负对照 C）：对 214 张图**逐字相同**的一条指令，句框与 reg_a/reg_b 完全一致，
# 只把「区域名词短语」换成 RO-X1 用过的无名词 deictic `the main subject`。
FIXED_INSTRUCTION = ("Please brighten the main subject, "
                     "keeping the rest of the image unchanged.")


# ---------------------------------------------------------------------------
# prompt 构造
# ---------------------------------------------------------------------------
def build_prompt_ids(tokenizer, mode: str, instruction: str | None):
    """→ input_ids (1D LongTensor)。

    `mode="style"`：G1/RO-9 部署 prompt（`data/infer_dataset.py` Infer_Style_Dataset 同款，
      = `tools/readout/ro9_gl_attention.build_inputs` 逐字一致），带逐源指令。
    `mode="auto"` ：**6-09 原脚本用的那一条**（`Infer_Auto_Dataset.__getitem__`，
      `data/infer_dataset.py:61`），`<Auto_Retouch_Task>`，**完全不含指令**——
      因此它同时是本实验最强的「零指令」负对照。
    """
    from llava.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                                 TASK_AUTO_RETOUCH_TOKEN, TASK_STYLE_RETOUCH_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    if mode == "auto":
        qs = (f"{TASK_AUTO_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
              "When I provide an image, please state the problems found in the image "
              "(from 3 aspects: lighting, global_color, specific color), and give the "
              "solution and retouch tokens.")
    elif mode == "style":
        assert instruction is not None
        qs = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
              "I will provide an image and an instruction, please give me a retouch plan "
              f"and retouch tokens.\n Instruction: {instruction}")
    else:
        raise ValueError(mode)
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX,
                                 return_tensors="pt")


def prep_image(image_processor, img_path: str):
    """短边 512 LANCZOS → process_images_（expand2square 黑边 pad → 1024²）。
    与 G1 `build_inputs` / `luma_valid_from_image` 完全同一条路径。"""
    from PIL import Image

    from llava.mm_utils import process_images_

    image = Image.open(img_path).convert("RGB")
    w, h = image.size
    scale = 512 / min(w, h)
    im512 = image.resize((int(round(w * scale)), int(round(h * scale))),
                         Image.Resampling.LANCZOS)
    image_tensor = process_images_([im512], image_processor)[0]
    luma = np.asarray(im512.convert("L"), dtype=np.float32) / 255.0
    return image_tensor, im512.size, luma


# ---------------------------------------------------------------------------
# 单样本读出（= 6-09 脚本的读出逻辑）
# ---------------------------------------------------------------------------
def readout_sample(model, tokenizer, image_processor, token_ids: dict,
                   img_path: str, mode: str, instruction: str | None,
                   device: str = "cuda:1") -> dict:
    """teacher-force 3 个 retouch token 于 prompt 末尾 → 一次 eager 前向 → 抄 post-softmax。

    返回（全部 float32，**未做任何归一化**）：
      raw       (3, 24, 16, 16)  三 token 的 A_reduced 逐层分量（层未平均，供分析侧组合）
      common    (24, 16, 16)     图像之后所有 query 行的均值图（M1 的共模），逐层
      common_prompt (24,16,16)   同上但**排除**末尾 3 个 retouch 行（鲁棒性变体，见 NOTES D-2）
      bad       (24, 16, 16) bool  D-0：该层 image-token 范数 outlier
      outlier_frac (24,) / luma16 / valid16 / n_post_rows / seq_len
    """
    import torch

    input_ids = build_prompt_ids(tokenizer, mode, instruction)
    image_tensor, image_size, luma = prep_image(image_processor, img_path)
    rt = torch.tensor([token_ids[t] for t in TOKEN_TAGS], dtype=input_ids.dtype)
    ids = torch.cat([input_ids, rt]).unsqueeze(0).to(device)          # teacher-force 3 token
    images = [image_tensor.to(torch.bfloat16).to(device)]

    model.lens_image_spans = []
    t0 = time.time()
    with torch.inference_mode():
        out = model(input_ids=ids, images=images, image_sizes=[image_size],
                    output_attentions=True, output_hidden_states=True,
                    use_cache=False, return_dict=True)
    t_fwd = time.time() - t0

    # ---- 红线守卫：eager 下 attentions 必须非 None；不回退
    if out.attentions is None or out.attentions[0] is None:
        raise RuntimeError("output_attentions 返回 None —— 非 eager 路径，拒绝继续（红线）")
    if len(out.attentions) != N_LAYERS:
        raise RuntimeError(f"attention 层数 {len(out.attentions)} != {N_LAYERS}")
    span = model.lens_image_spans[0]
    img_start, img_end, total = span
    if img_end - img_start != N_IMG_TOKENS:
        raise RuntimeError(f"image span {span} != {N_IMG_TOKENS} tokens")
    if total != ids.shape[1] + (N_IMG_TOKENS - 1):
        raise RuntimeError(f"展开长度不一致: {total} vs {ids.shape[1]}+{N_IMG_TOKENS-1}")
    seq = int(out.attentions[0].shape[-1])
    if seq != total:
        raise RuntimeError(f"attention seq {seq} != 展开长度 {total}")

    # ---- 6-09 读出：层内头平均 → 只留 image 列 → 三 token 行 / 图像之后所有行的均值
    rt_rows = [total - 3, total - 2, total - 1]        # 末尾三行 = L, GC, SC（teacher-forced）
    post_lo = img_end                                  # = img_block[-1] + 1
    n_post = total - post_lo
    if n_post <= 3:
        raise RuntimeError(f"图像之后只有 {n_post} 行，共模无意义")

    raw = np.empty((3, N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
    common = np.empty((N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
    common_p = np.empty((N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
    for li in range(N_LAYERS):
        a = out.attentions[li][0]                                  # (H, seq, seq) post-softmax
        red = a[:, post_lo:total, img_start:img_end].float().mean(0)   # head-mean → (n_post,256)
        red_np = red.cpu().numpy()
        common[li] = red_np.mean(0)                                # ← 原脚本的 common
        common_p[li] = red_np[:-3].mean(0)                         # 排除 3 个 retouch 行
        for ti in range(3):
            raw[ti, li] = red_np[rt_rows[ti] - post_lo]

    # ---- D-0（G1 口径）：读出层 image-token 范数 outlier。本工具**只落盘掩膜**，
    #      修复与否留给分析侧（6-09 原脚本没有 D-0，忠实复现需要"不修"这一档）。
    hs = out.hidden_states
    bad = np.zeros((N_LAYERS, N_IMG_TOKENS), dtype=bool)
    for li in range(N_LAYERS):
        h = hs[li + 1][0, img_start:img_end].float().cpu().numpy()
        bad[li] = outlier_mask_from_norms(np.linalg.norm(h, axis=-1))

    luma16, valid16 = luma_to_grid(luma)
    g = (N_LAYERS, GRID, GRID)
    return {
        "raw": raw.reshape(3, *g), "common": common.reshape(*g),
        "common_prompt": common_p.reshape(*g), "bad": bad.reshape(*g),
        "outlier_frac": bad.mean(axis=1).astype(np.float32),
        "luma16": luma16, "valid16": valid16,
        "n_post_rows": int(n_post), "seq_len": int(total),
        "img_span": [int(img_start), int(img_end), int(total)],
        "t_fwd": t_fwd,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--region-json", required=True,
                    help="G1 已冻结的区域对立批（零重新采样）")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--conditions", nargs="+",
                    default=["auto", "reg_a", "reg_b", "fixed"])
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--config-add", default=CONFIG_ADD_DEFAULT)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--mem-frac", type=float, default=0.2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    import torch

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if args.device.startswith("cuda") and args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(
            args.mem_frac, device=int(args.device.split(":")[1]))

    sources = json.loads(Path(args.region_json).read_text())
    if args.limit > 0:
        sources = sources[: args.limit]
    out_dir = Path(args.out_dir)
    stacks = out_dir / "stacks"
    stacks.mkdir(parents=True, exist_ok=True)

    model, tokenizer, image_processor, token_ids = load_model(
        args.model_path, args.config_add, args.device)
    print(f"[load] ok · attn_impl={model.config._attn_implementation} · "
          f"token_ids={token_ids}", flush=True)

    logf = open(out_dir / "run_log.jsonl", "a", encoding="utf-8")
    n_done = n_skip = n_err = 0
    t_start = time.time()
    for i, s in enumerate(sources):
        iid, ipath = s["img_id"], s["img_path"]
        for cond in args.conditions:
            p = stacks / f"{iid}__{cond}.npz"
            if args.skip_existing and p.exists():
                n_skip += 1
                continue
            if cond == "auto":
                mode, instr = "auto", None
            elif cond == "fixed":
                mode, instr = "style", FIXED_INSTRUCTION
            else:
                mode, instr = "style", s["instructions"][cond]
            try:
                r = readout_sample(model, tokenizer, image_processor, token_ids,
                                   ipath, mode, instr, device=args.device)
            except Exception as e:
                n_err += 1
                logf.write(json.dumps({"img_id": iid, "cond": cond,
                                       "error": repr(e)[:500]}) + "\n")
                logf.flush()
                continue
            tmp = stacks / f"{iid}__{cond}.tmp.npz"
            np.savez_compressed(
                tmp, raw=r["raw"], common=r["common"],
                common_prompt=r["common_prompt"], bad=r["bad"],
                outlier_frac=r["outlier_frac"], luma16=r["luma16"],
                valid16=r["valid16"],
                meta=json.dumps({
                    "img_id": iid, "cond": cond, "mode": mode, "instr": instr,
                    "pool": s.get("pool"), "build": s.get("build"),
                    "winner_confidence": s.get("winner_confidence"),
                    "region_b_kind": s.get("region_b_kind"),
                    "subject_area": s.get("subject_area"),
                    "n_post_rows": r["n_post_rows"], "seq_len": r["seq_len"],
                    "img_span": r["img_span"],
                }))
            os.replace(tmp, p)
            logf.write(json.dumps({
                "img_id": iid, "cond": cond, "seq_len": r["seq_len"],
                "n_post_rows": r["n_post_rows"],
                "outlier_frac_mean": float(r["outlier_frac"].mean()),
                "t_fwd": round(r["t_fwd"], 3)}) + "\n")
            logf.flush()
            n_done += 1
        if (i + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"[{i+1}/{len(sources)}] done={n_done} skip={n_skip} err={n_err} "
                  f"elapsed={el/60:.1f}min "
                  f"eta={el/(i+1)*(len(sources)-i-1)/60:.1f}min", flush=True)
    logf.close()
    print(f"FINISHED sources={len(sources)} done={n_done} skip={n_skip} err={n_err} "
          f"wall={(time.time()-t_start)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
