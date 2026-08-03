"""RO-9 · GL token 原生读出（EXPERIMENTS_v3 §2.1 RO-9 行 + PLAN_v2 D-0/D-5）。

导出 VeraRetouch special token（默认 <retouch_light>，即 "GL token"）→ image token 的
**pre-softmax attention logit** 图，经 D-0 伪影修复（token 范数 outlier 剔除 + 插值），
落 tools/scache 缓存（arm=ro9）。

硬规则（CLAUDE.md 红线 / DOSSIER §5.4）：
- attention 导出必须 eager：加载 `attn_implementation="eager"`，运行时双重断言
  （config._attn_implementation == "eager" 且 forward 返回 attentions 非 None）。
  FA2/SDPA 下 output_attentions 返回 None——**不回退，直接 raise**。
- pre-softmax：monkeypatch transformers.models.qwen2.modeling_qwen2.eager_attention_forward，
  在 softmax 前抄走目标 query 行（q@k^T*scaling + causal_mask）。
- s 禁逐图归一化：缓存写入的是原始 logit 聚合值，不做任何 min-max/softmax。

用法（.venv-lens 环境）：
  python tools/readout/ro9_gl_attention.py \
      --samples-json <g1_samples.json> --scache-root /var/cache/veradata/scache \
      --out-dir <run_dir> [--limit 30] [--device cuda:0]

samples json 格式：[{"img_id": ..., "img_path": ..., "instructions":
                    {"syn_a": ..., "syn_b": ..., "opp": ...}, ...}, ...]
每 (img_id, instruction) 写一条 scache 条目（16×16 canonical s）+
run_dir/stacks/<key>.npz（per-layer head-mean logit 全栈 + 亮度图 + 元信息）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "scache"))

# ---------------------------------------------------------------------------
# 常量（与 NOTES.md §一 已核实事实一致）
# ---------------------------------------------------------------------------
MODEL_PATH_DEFAULT = "/home/bc/data/models/VeraRetouch"
CONFIG_ADD_DEFAULT = str(REPO / "configs" / "infer_config.yaml")
GRID = 16                      # mobileclip_l_1024: 1024/64 = 16 -> 256 image tokens
N_IMG_TOKENS = GRID * GRID
N_LAYERS = 24
GL_TOKEN = "<retouch_light>"   # 待决策 D1：默认 GL = retouch_light
CT_TOKEN = "<retouch_color&temp>"
CM_TOKEN = "<retouch_colormixer>"
TOKEN_TAGS = ("light", "colortemp", "colormixer")
CANON_LAYERS = tuple(range(8, 16))   # 待决策 D2：canonical 聚合层（中层先验）
MAD_K = 3.0                    # D-0: norm > median + 3*MAD -> outlier

# 方向词取反表（G1 反义构造；小写匹配，保序替换，词边界）
ANTONYM_MAP = {
    "brighten": "darken", "darken": "brighten",
    "brightening": "darkening", "darkening": "brightening",
    "brighter": "darker", "darker": "brighter",
    "brightness": "darkness",
    "lighten": "darken", "lightening": "darkening",
    "warm": "cool", "cool": "warm",
    "warmer": "cooler", "cooler": "warmer",
    "warming": "cooling", "cooling": "warming",
    "warmth": "coolness",
    "increase": "decrease", "decrease": "increase",
    "increasing": "decreasing", "decreasing": "increasing",
    "boost": "reduce", "reduce": "boost",
    "boosting": "reducing", "reducing": "boosting",
    "enhance": "mute", "enrich": "mute",
    "enhancing": "muting", "enriching": "muting",
    "richer": "flatter", "vibrant": "muted", "muted": "vibrant",
    "vivid": "dull", "saturated": "desaturated", "desaturated": "saturated",
    "saturate": "desaturate", "desaturate": "saturate",
    "deepen": "soften", "soften": "deepen",
    "stronger": "weaker", "weaker": "stronger",
    "more": "less", "less": "more",
}


def make_antonym(instr: str) -> tuple[str | None, int]:
    """方向词取反；返回 (反义句 | None, 替换数)。无可取反词 -> (None, 0)。"""
    import re

    n = 0

    def repl(m: "re.Match[str]") -> str:
        nonlocal n
        w = m.group(0)
        low = w.lower()
        out = ANTONYM_MAP[low]
        n += 1
        if w[0].isupper():
            out = out[0].upper() + out[1:]
        return out

    pat = re.compile(
        r"\b(" + "|".join(sorted(ANTONYM_MAP, key=len, reverse=True)) + r")\b",
        re.IGNORECASE,
    )
    out = pat.sub(repl, instr)
    return (out, n) if n > 0 else (None, 0)


# ---------------------------------------------------------------------------
# pre-softmax 捕获（monkeypatch eager_attention_forward）
# ---------------------------------------------------------------------------
class PreSoftmaxRecorder:
    """截取指定 query 位置在各层的 pre-softmax attention logit 行。

    仅在 `capturing` 上下文里、q_len == full_len（teacher-forced 整段前向）时记录，
    生成阶段（q_len==1 的增量步）不记录。
    """

    def __init__(self) -> None:
        self.query_positions: list[int] = []
        self.rows: dict[int, "object"] = {}   # layer_idx -> tensor (heads, nq, kv)
        self.active = False

    def install(self) -> None:
        import transformers.models.qwen2.modeling_qwen2 as m

        if getattr(m, "_ro9_patched", False):
            return
        orig = m.eager_attention_forward
        rec = self

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout: float = 0.0, **kwargs):
            if rec.active and query.shape[2] > 1 and rec.query_positions:
                import torch

                q = query[:, :, rec.query_positions, :]           # (b, h, nq, d)
                k = key
                if module.num_key_value_groups > 1:
                    k = m.repeat_kv(key, module.num_key_value_groups)
                logits = torch.matmul(q, k.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    cm = attention_mask[:, :, rec.query_positions, : k.shape[-2]]
                    logits = logits + cm
                rec.rows[module.layer_idx] = logits[0].float().cpu()
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout=dropout, **kwargs)

        m.eager_attention_forward = patched
        m._ro9_patched = True

    def start(self, query_positions: list[int]) -> None:
        self.query_positions = list(query_positions)
        self.rows = {}
        self.active = True

    def stop(self) -> None:
        self.active = False


# ---------------------------------------------------------------------------
# D-0：token 范数 outlier 剔除 + 插值
# ---------------------------------------------------------------------------
def outlier_mask_from_norms(norms: np.ndarray, k: float = MAD_K) -> np.ndarray:
    """norms: (256,) -> bool (256,)，True = outlier（norm > median + k*MAD）。"""
    med = np.median(norms)
    mad = np.median(np.abs(norms - med))
    if mad <= 0:
        return np.zeros_like(norms, dtype=bool)
    return norms > med + k * mad


def interpolate_masked(grid: np.ndarray, bad: np.ndarray, iters: int = 8) -> np.ndarray:
    """grid (16,16) float，bad (16,16) bool。4-邻域均值迭代补 NaN，兜底全局均值。"""
    out = grid.astype(np.float64).copy()
    out[bad] = np.nan
    for _ in range(iters):
        nan = np.isnan(out)
        if not nan.any():
            break
        pad = np.pad(out, 1, constant_values=np.nan)
        neigh = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
        with np.errstate(all="ignore"):
            fill = np.nanmean(neigh, axis=0)
        out[nan & np.isfinite(fill)] = fill[nan & np.isfinite(fill)]
    if np.isnan(out).any():
        out[np.isnan(out)] = np.nanmean(out)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 模型加载与单样本读出
# ---------------------------------------------------------------------------
def load_model(model_path: str = MODEL_PATH_DEFAULT,
               config_add_path: str = CONFIG_ADD_DEFAULT,
               device: str = "cuda:0"):
    import torch
    import yaml
    from box import Box
    from transformers import AutoTokenizer

    from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified
    from llava.utils import disable_torch_init

    with open(config_add_path, "r", encoding="utf-8") as f:
        config_add = Box(yaml.safe_load(f))
    config_add.project_name = "ro9_readout"
    config_add.freeze_retouch_decoder = True

    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, model_max_length=8192, padding_side="right", use_fast=False)
    model = VeraRetouchForCausalLLM_Unified.from_pretrained(
        model_path, config_add=config_add, torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    # 红线守卫 1：必须 eager（FA2/SDPA 不回退）
    impl = model.config._attn_implementation
    if impl != "eager":
        raise RuntimeError(f"attn_implementation={impl!r} != 'eager' — 拒绝继续（DOSSIER §5.4）")

    ids = {tag: tokenizer(t, add_special_tokens=False).input_ids[0]
           for tag, t in zip(TOKEN_TAGS, (GL_TOKEN, CT_TOKEN, CM_TOKEN))}
    assert ids["light"] == 151646, ids  # NOTES §一.2 核实值
    model.register_special_token_idx(ids["light"], ids["colortemp"], ids["colormixer"])
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.lens_track_spans = True   # lens-exp 插桩：记录 image token 区间
    return model, tokenizer, model.get_vision_tower().image_processor, ids


def build_inputs(tokenizer, image_processor, img_path: str, instruction: str):
    """style 模式 prompt（data/infer_dataset.py:141 同款）→ (input_ids, image_tensor, image_size)。"""
    import torch
    from PIL import Image

    from llava.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                                 TASK_STYLE_RETOUCH_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images_, tokenizer_image_token

    qs = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
          "I will provide an image and an instruction, please give me a retouch plan "
          f"and retouch tokens.\n Instruction: {instruction}")
    qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
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


def luma_to_grid(luma: np.ndarray, grid: int = GRID) -> tuple[np.ndarray, np.ndarray]:
    """亮度图 → (grid, grid) 亮度 + 有效掩膜，对齐 token 网格。

    对齐依据（llava/mm_utils.py:186 process_images_）：非方图先 expand2square
    **黑边 pad**（image_mean=0）到方形，再 resize 1024²——不是 center crop。
    返回 (luma16, valid16)：valid16 = 格子被真实图像覆盖比例 ≥0.5（黑边格 False）。
    """
    h, w = luma.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    cov = np.zeros((side, side), dtype=np.float32)
    if w >= h:
        top = (side - h) // 2
        sq[top:top + h, :] = luma
        cov[top:top + h, :] = 1.0
    else:
        left = (side - w) // 2
        sq[:, left:left + w] = luma
        cov[:, left:left + w] = 1.0
    # 面积均值下采样（side 不整除 grid 时截尾，误差 ≤ grid 像素）
    n = side // grid
    sq = sq[: n * grid, : n * grid]
    cov = cov[: n * grid, : n * grid]
    l16 = sq.reshape(grid, n, grid, n).mean(axis=(1, 3)).astype(np.float32)
    v16 = cov.reshape(grid, n, grid, n).mean(axis=(1, 3)) >= 0.5
    return l16, v16


def readout_sample(model, tokenizer, image_processor, token_ids: dict,
                   recorder: PreSoftmaxRecorder, img_path: str, instruction: str,
                   device: str = "cuda:0", max_new_tokens: int = 512) -> dict:
    """单样本：greedy 生成 → teacher-forced eager 前向截 pre-softmax → D-0 修复。

    返回 dict：
      s_canon        (16,16) float32  canonical s（D1/D2 默认口径，修复后）
      stack          (3, 24, 16, 16) float32  三 token × 24 层 head-mean logit（修复后）
      stack_raw      同上，修复前
      outlier_frac   (24,) float32   逐层 outlier 占比
      luma16         (16,16) float32
      gen_text, fallback (bool per token), positions, timings
    """
    import torch

    t0 = time.time()
    input_ids, image_tensor, image_size, luma = build_inputs(
        tokenizer, image_processor, img_path, instruction)
    input_ids = input_ids.unsqueeze(0).to(device)
    images = [image_tensor.to(torch.bfloat16).to(device)]

    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids, images=images, image_sizes=[image_size],
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_attentions=False, use_cache=True)
    gen_ids = out.sequences[0].tolist()
    t_gen = time.time() - t0

    # 三 token 在生成序列中的首现位置；缺失 -> 追加（待决策 D6 fallback）
    fallback = {}
    appended: list[int] = []
    for tag in TOKEN_TAGS:
        tid = token_ids[tag]
        if tid in gen_ids:
            fallback[tag] = False
        else:
            fallback[tag] = True
            appended.append(tid)
    if appended:
        gen_ids = gen_ids + appended

    # teacher-forced 全序列：prompt(<image> 占位) + 生成 ids
    full_ids = torch.cat([input_ids[0], torch.tensor(gen_ids, device=device)]).unsqueeze(0)

    # 展开后位置：单图 1 占位 -> 256 embed，占位之后的文本位置 +255
    from llava.constants import IMAGE_TOKEN_INDEX
    flat = full_ids[0].tolist()
    img_pos = flat.index(IMAGE_TOKEN_INDEX)
    offset = N_IMG_TOKENS - 1

    def expanded(pos: int) -> int:
        return pos + offset if pos > img_pos else pos

    qpos, first_pos = [], {}
    for tag in TOKEN_TAGS:
        tid = token_ids[tag]
        p = flat.index(tid)          # 全序列首现
        first_pos[tag] = p
        qpos.append(expanded(p))

    model.lens_image_spans = []      # prepare_inputs... 会重记
    recorder.start(qpos)
    t1 = time.time()
    with torch.inference_mode():
        fwd = model(input_ids=full_ids, images=images, image_sizes=[image_size],
                    output_attentions=True, output_hidden_states=True,
                    use_cache=False, return_dict=True)
    recorder.stop()
    t_fwd = time.time() - t1

    # 红线守卫 2：eager 下 attentions 必须非 None（SDPA/FA2 会静默 None——不回退）
    if fwd.attentions is None or fwd.attentions[0] is None:
        raise RuntimeError("output_attentions 返回 None——非 eager 路径，拒绝继续（DOSSIER §5.4）")
    if len(recorder.rows) != N_LAYERS:
        raise RuntimeError(f"捕获层数 {len(recorder.rows)} != {N_LAYERS}")

    span = model.lens_image_spans[0]
    img_start, img_end, total = span
    if img_end - img_start != N_IMG_TOKENS:
        raise RuntimeError(f"image span {span} != {N_IMG_TOKENS} tokens")
    if total != full_ids.shape[1] + offset:
        raise RuntimeError(f"展开长度不一致: {total} vs {full_ids.shape[1]} + {offset}")

    # (3, 24, 14, 256) pre-softmax logit（head 维保留） -> head-mean (3, 24, 256)
    per_layer = []
    for li in range(N_LAYERS):
        rows = recorder.rows[li]                       # (heads, 3, kv)
        per_layer.append(rows[:, :, img_start:img_end].numpy())
    raw = np.stack(per_layer, axis=0)                  # (24, heads, 3, 256)
    raw = np.transpose(raw, (2, 0, 1, 3))              # (3, 24, heads, 256)
    stack_raw = raw.mean(axis=2).reshape(3, N_LAYERS, GRID, GRID)  # head-mean

    # D-0：读出层 image-token 范数 outlier（逐层）+ 插值
    hs = fwd.hidden_states                              # tuple len 25, each (1, T, 896)
    outlier_frac = np.zeros(N_LAYERS, dtype=np.float32)
    stack = np.empty_like(stack_raw)
    bad_grids = []
    for li in range(N_LAYERS):
        h = hs[li + 1][0, img_start:img_end].float().cpu().numpy()  # 层输出
        norms = np.linalg.norm(h, axis=-1)
        bad = outlier_mask_from_norms(norms)
        outlier_frac[li] = float(bad.mean())
        bad_grid = bad.reshape(GRID, GRID)
        bad_grids.append(bad_grid)
        for ti in range(3):
            stack[ti, li] = (interpolate_masked(stack_raw[ti, li], bad_grid)
                             if bad.any() else stack_raw[ti, li])

    canon = stack[0][list(CANON_LAYERS)].mean(axis=0)   # GL=light, L8-15 平均

    luma16, valid16 = luma_to_grid(luma)
    return {
        "s_canon": canon.astype(np.float32),
        "stack": stack.astype(np.float32),
        "stack_raw": stack_raw.astype(np.float32),
        "outlier_frac": outlier_frac,
        "luma16": luma16,
        "valid16": valid16,
        "gen_text": tokenizer.decode(gen_ids, skip_special_tokens=True)[:2000],
        "fallback": fallback,
        "first_pos": first_pos,
        "gen_len": len(gen_ids),
        "t_gen": t_gen, "t_fwd": t_fwd,
    }


# ---------------------------------------------------------------------------
# CLI：批量跑 samples json -> scache + stacks
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples-json", required=True)
    ap.add_argument("--scache-root", default="/var/cache/veradata/scache")
    ap.add_argument("--arm", default="ro9")
    ap.add_argument("--out-dir", required=True, help="stacks/ 与 progress 落这里")
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--config-add", default=CONFIG_ADD_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help=">0 只跑前 N 个源（冒烟）")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    from api import SCache, instr_hash  # tools/scache

    samples = json.loads(Path(args.samples_json).read_text())
    if args.limit > 0:
        samples = samples[: args.limit]

    out_dir = Path(args.out_dir)
    stacks_dir = out_dir / "stacks"
    stacks_dir.mkdir(parents=True, exist_ok=True)
    cache = SCache(args.scache_root, args.arm, resolution=GRID, arm_version="v1")

    model, tokenizer, image_processor, token_ids = load_model(
        args.model_path, args.config_add, args.device)
    recorder = PreSoftmaxRecorder()
    recorder.install()

    log_path = out_dir / "run_log.jsonl"
    n_done = n_skip = 0
    t_start = time.time()
    with open(log_path, "a", encoding="utf-8") as logf:
        for i, s in enumerate(samples):
            img_id, img_path = s["img_id"], s["img_path"]
            for itag, instr in s["instructions"].items():
                ih = instr_hash(instr)
                if args.skip_existing and cache.exists(img_id, ih):
                    n_skip += 1
                    continue
                try:
                    r = readout_sample(model, tokenizer, image_processor, token_ids,
                                       recorder, img_path, instr,
                                       device=args.device,
                                       max_new_tokens=args.max_new_tokens)
                except Exception as e:  # 单样本失败不拖垮批次
                    logf.write(json.dumps({"img_id": img_id, "itag": itag,
                                           "error": repr(e)}) + "\n")
                    logf.flush()
                    continue
                cache.write(
                    img_id, ih, r["s_canon"],
                    layer=None,
                    norm={"kind": "presoftmax-logit-raw", "note": "no per-image norm"},
                    extra_meta={
                        "token": GL_TOKEN, "layers_agg": list(CANON_LAYERS),
                        "head_agg": "mean", "itag": itag,
                        "instruction": instr[:500],
                        "outlier_frac_mean": float(r["outlier_frac"].mean()),
                        "fallback": r["fallback"]["light"],
                        "source_pool": s.get("pool"), "build": s.get("build"),
                    })
                key = f"{img_id}__{ih}"
                np.savez_compressed(
                    stacks_dir / f"{key}.npz",
                    stack=r["stack"].astype(np.float16),
                    stack_raw=r["stack_raw"].astype(np.float16),
                    outlier_frac=r["outlier_frac"],
                    luma16=r["luma16"], valid16=r["valid16"], s_canon=r["s_canon"],
                    meta=json.dumps({"img_id": img_id, "itag": itag,
                                     "instr": instr, "pool": s.get("pool"),
                                     "fallback": r["fallback"],
                                     "gen_len": r["gen_len"]}))
                logf.write(json.dumps({
                    "img_id": img_id, "itag": itag, "ih": ih,
                    "gen_len": r["gen_len"],
                    "fallback": r["fallback"],
                    "outlier_frac_mean": float(r["outlier_frac"].mean()),
                    "t_gen": round(r["t_gen"], 2), "t_fwd": round(r["t_fwd"], 2),
                }) + "\n")
                logf.flush()
                n_done += 1
            if (i + 1) % 10 == 0:
                el = time.time() - t_start
                print(f"[{i+1}/{len(samples)}] done={n_done} skip={n_skip} "
                      f"elapsed={el/60:.1f}min eta={el/(i+1)*(len(samples)-i-1)/60:.1f}min",
                      flush=True)

    print(f"FINISHED sources={len(samples)} entries_done={n_done} skipped={n_skip} "
          f"wall={(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
