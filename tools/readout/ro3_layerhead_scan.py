"""RO-3 · LMM text→image attention **全层全头**扫描（EXPERIMENTS_v3 §2.1 RO-3 行）。

与 RO-9（`ro9_gl_attention.py`）的关系：**复用**其模型加载 / prompt 构造 / D-0 范数-MAD
outlier 判别 / 16×16 网格对齐 / scache 写入约定；**新增**的只有三件事：
  1. 不预设 token：query 端取 **text token 池**（instr / last / alltxt / gl-bridge 四池）；
  2. 不预设层与头：24 层 × 14 头**逐头**导出（RO-9 是 head-mean，会把单头信号平均掉，
     见 `experiments/RO9_layer_verdict_20260804/REPORT.md` §五.2 的明确交办）；
  3. **pre-softmax logit 与 post-softmax 概率同时导出**（本臂"绕开 attention sink"的核心假设）。

硬规则（CLAUDE.md 红线 / DOSSIER §5.4，已于 `config/verify_env.json` **实测**核实）：
- attention 导出必须 eager。实测：transformers 4.57.1 下 `attn_implementation="sdpa"` +
  `output_attentions=True` → **attentions is None**（只 warning，**不回退**）。本脚本
  加载即断言 eager，且每个样本断言 `fwd.attentions[0] is not None`，否则 raise。
- pre-softmax 捕获经 V4 实测校验：`softmax(捕获 logit)` 与 eager 返回的权重逐元素一致
  （max|Δ| = 1.95e-3 = bf16 舍入量级）。
- **s 禁逐图归一化**：落盘的是原始 logit / 原始概率，无任何 min-max/softmax/z-score 逐图操作。
- D-0（PLAN_v2 §2.2 D-0 第二件）：逐层 image-token 范数 outlier（median+3·MAD）**只落盘掩膜**，
  修复在分析端做，从而 D-0 on/off 可 A/B（PLAN D-0 判据要求）。

本项目实测常量（`config/verify_env.json`，勿信转述）：
  24 层 / 14 heads / 2 KV heads（GQA groups=7）/ hidden 896 / mobileclip_l_1024
  image token 段 = 256（16×16），prompt 内 `<image>` = llava 占位 **-200**（不在词表，
  **不是** DOSSIER §5.3 的 Qwen2.5-VL 151652/151653/151655）
  `<retouch_light>`=151646 / `<retouch_color&temp>`=151647 / `<retouch_colormixer>`=151648

用法（.venv-lens，CUDA_VISIBLE_DEVICES=1）：
  python tools/readout/ro3_layerhead_scan.py --jobs-json <jobs.json> --out-dir <stacks 根>
jobs json = [{"key": ..., "img_id": ..., "img_path": ..., "itag": ..., "instruction": ...}, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "readout"))
sys.path.insert(0, str(REPO / "tools" / "scache"))

from ro9_gl_attention import (GRID, MODEL_PATH_DEFAULT, N_IMG_TOKENS,  # noqa: E402
                              N_LAYERS, build_inputs, load_model,
                              luma_to_grid, outlier_mask_from_norms)

N_HEADS = 14
GL_TOKEN_ID = 151646                      # <retouch_light>，verify_env V6 实测
POOLS = ("instr", "last", "alltxt", "gl")  # query 池
MODES = ("pre", "post")                    # pre-softmax logit / post-softmax 概率


# ---------------------------------------------------------------------------
# 逐层逐头捕获器
# ---------------------------------------------------------------------------
class LayerHeadRecorder:
    """截取选定 query 位置在 **每一层每一个头** 上对 image token 段的注意力。

    在 softmax **之前**抄走 `q@kᵀ*scaling + causal_mask`（= pre），并在**整行**
    （全 key 集合）上做 softmax 后再切 image 列（= post，真实注意力概率，
    保留 attention sink 对分母的贡献）。
    """

    def __init__(self) -> None:
        self.qpos: list[int] = []
        self.img_slice: tuple[int, int] = (0, 0)
        self.pre: dict[int, np.ndarray] = {}
        self.post: dict[int, np.ndarray] = {}
        self.active = False

    def install(self) -> None:
        import torch
        import transformers.models.qwen2.modeling_qwen2 as m

        if getattr(m, "_ro3_patched", False):
            return
        orig = getattr(m, "_ro3_orig", None) or m.eager_attention_forward
        m._ro3_orig = orig
        rec = self

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout: float = 0.0, **kwargs):
            if rec.active and query.shape[2] > 1 and rec.qpos:
                q = query[:, :, rec.qpos, :]                      # (1, H, nq, d)
                k = m.repeat_kv(key, module.num_key_value_groups)  # GQA: 2 -> 14
                lg = torch.matmul(q, k.transpose(2, 3)) * scaling  # (1, H, nq, T)
                if attention_mask is not None:
                    lg = lg + attention_mask[:, :, rec.qpos, : k.shape[-2]]
                lg = lg[0].float()                                 # (H, nq, T)
                pr = torch.softmax(lg, dim=-1)                     # 全行 softmax（含 sink）
                a, b = rec.img_slice
                rec.pre[module.layer_idx] = lg[:, :, a:b].cpu().numpy()
                rec.post[module.layer_idx] = pr[:, :, a:b].cpu().numpy()
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout=dropout, **kwargs)

        m.eager_attention_forward = patched
        m._ro3_patched = True

    def start(self, qpos: list[int], img_slice: tuple[int, int]) -> None:
        self.qpos, self.img_slice = list(qpos), img_slice
        self.pre, self.post = {}, {}
        self.active = True

    def stop(self) -> None:
        self.active = False


# ---------------------------------------------------------------------------
# query 池定位
# ---------------------------------------------------------------------------
def locate_pools(tokenizer, instruction: str, prompt_ids: list[int], img_pos: int) -> dict:
    """在**未展开**的 prompt token 序列里定位四个 query 池。

    instr 段用「同 prompt、instruction 置空」的公共前缀/后缀差分法定位（slow tokenizer
    无 offset mapping；子串 id 匹配会因上下文切分不同而失配——verify_env V7 实测 hit=-1）。
    """
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, TASK_STYLE_RETOUCH_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    def ids_for(instr: str) -> list[int]:
        qs = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
              "I will provide an image and an instruction, please give me a retouch plan "
              f"and retouch tokens.\n Instruction: {instr}")
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        return tokenizer_image_token(conv.get_prompt(), tokenizer,
                                     IMAGE_TOKEN_INDEX, return_tensors=None)

    empty = ids_for("")
    full = prompt_ids
    # 公共前缀
    p = 0
    while p < min(len(empty), len(full)) and empty[p] == full[p]:
        p += 1
    # 公共后缀
    s = 0
    while (s < min(len(empty), len(full)) - p
           and empty[len(empty) - 1 - s] == full[len(full) - 1 - s]):
        s += 1
    instr_span = (p, len(full) - s)          # [start, end) 未展开坐标
    last_pos = len(full) - 1                 # prompt 末 token（assistant 起始处）
    alltxt = [i for i in range(img_pos + 1, len(full))]
    return {"instr_span": instr_span, "last_pos": last_pos, "alltxt": alltxt,
            "instr_decoded": tokenizer.decode(full[instr_span[0]:instr_span[1]])}


# ---------------------------------------------------------------------------
# 单样本读出（prefill only，无生成）
# ---------------------------------------------------------------------------
def readout_sample(model, tokenizer, image_processor, recorder: LayerHeadRecorder,
                   img_path: str, instruction: str, device: str = "cuda:0") -> dict:
    import torch

    t0 = time.time()
    input_ids, image_tensor, image_size, luma = build_inputs(
        tokenizer, image_processor, img_path, instruction)
    prompt_ids = input_ids.tolist()
    from llava.constants import IMAGE_TOKEN_INDEX
    img_pos = prompt_ids.index(IMAGE_TOKEN_INDEX)
    pools_u = locate_pools(tokenizer, instruction, prompt_ids, img_pos)

    # gl-bridge：在 prompt 末尾**追加** <retouch_light>（teacher-forced，不生成）
    full_ids = prompt_ids + [GL_TOKEN_ID]
    gl_pos_u = len(prompt_ids)

    offset = N_IMG_TOKENS - 1

    def exp(pos: int) -> int:                # 未展开位置 -> 展开后位置
        return pos + offset if pos > img_pos else pos

    q_instr = [exp(i) for i in range(*pools_u["instr_span"])]
    q_last = [exp(pools_u["last_pos"])]
    q_all = [exp(i) for i in pools_u["alltxt"]]
    q_gl = [exp(gl_pos_u)]
    qpos_union = sorted(set(q_instr + q_last + q_all + q_gl))
    idx = {p: i for i, p in enumerate(qpos_union)}
    pool_idx = {"instr": [idx[p] for p in q_instr], "last": [idx[p] for p in q_last],
                "alltxt": [idx[p] for p in q_all], "gl": [idx[p] for p in q_gl]}
    if not pool_idx["instr"]:
        raise RuntimeError(f"instr 池为空（定位失败）: {pools_u['instr_span']}")

    ids_t = torch.tensor(full_ids, device=device).unsqueeze(0)
    images = [image_tensor.to(torch.bfloat16).to(device)]

    model.lens_image_spans = []
    recorder.start(qpos_union, (0, 0))       # img_slice 前向内才知道，先占位
    # image span 位置在本项目里恒定：seg0(=img_pos 个 text token) 之后 256 个视觉 embedding
    recorder.img_slice = (img_pos, img_pos + N_IMG_TOKENS)
    with torch.inference_mode():
        fwd = model(input_ids=ids_t, images=images, image_sizes=[image_size],
                    output_attentions=True, output_hidden_states=True,
                    use_cache=False, return_dict=True)
    recorder.stop()

    # 红线守卫（DOSSIER §5.4）：eager 下 attentions 必须非 None
    if fwd.attentions is None or fwd.attentions[0] is None:
        raise RuntimeError("output_attentions 返回 None —— 非 eager 路径，拒绝继续（DOSSIER §5.4）")
    if len(recorder.pre) != N_LAYERS:
        raise RuntimeError(f"捕获层数 {len(recorder.pre)} != {N_LAYERS}")
    span = model.lens_image_spans[0]
    if (span[0], span[1]) != recorder.img_slice:
        raise RuntimeError(f"image span {span} != 预期 {recorder.img_slice}")
    if span[2] != len(full_ids) + offset:
        raise RuntimeError(f"展开长度 {span[2]} != {len(full_ids)}+{offset}")

    # (24, H, nq, 256) -> 逐池 mean over query -> (24, H, 16, 16)
    out: dict[str, np.ndarray] = {}
    src = {"pre": recorder.pre, "post": recorder.post}
    for mode in MODES:
        arr = np.stack([src[mode][li] for li in range(N_LAYERS)], axis=0)  # (24,H,nq,256)
        for pool in POOLS:
            f = arr[:, :, pool_idx[pool], :].mean(axis=2)                 # (24,H,256)
            # **float32 落盘**：post 的 sink 主导头概率低到 1e-8，fp16 会整片下溢成 0
            # （实测 336 个 (层,头) 里 2 个全 0、~36 个 >10% 零），会伪造并列、污染
            # 本臂的核心对比（pre-softmax 能否绕开 sink）。pre 侧 fp16 量化亦有 4/336
            # 信噪比 <10 的格，一并按 float32 处理，杜绝存储伪影进判据。
            out[f"{mode}_{pool}"] = f.reshape(N_LAYERS, N_HEADS, GRID, GRID).astype(np.float32)

    # D-0（第二件）：逐层 image-token 范数 outlier 掩膜（只落盘，不在此修复）
    hs = fwd.hidden_states
    bad = np.zeros((N_LAYERS, GRID, GRID), dtype=bool)
    for li in range(N_LAYERS):
        h = hs[li + 1][0, span[0]:span[1]].float().cpu().numpy()
        bad[li] = outlier_mask_from_norms(np.linalg.norm(h, axis=-1)).reshape(GRID, GRID)

    luma16, valid16 = luma_to_grid(luma)
    out.update({"outlier": bad, "luma16": luma16, "valid16": valid16})
    return {"fields": out, "t": time.time() - t0,
            "n_instr_tok": len(q_instr), "n_prompt_tok": len(prompt_ids),
            "instr_decoded": pools_u["instr_decoded"], "seq_len": int(span[2])}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    jobs = json.loads(Path(args.jobs_json).read_text())
    if args.limit:
        jobs = jobs[: args.limit]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch
    model, tokenizer, image_processor, _ids = load_model(
        args.model_path, device=args.device)
    rec = LayerHeadRecorder()
    rec.install()

    log = open(out_dir / "run_log.jsonl", "a", encoding="utf-8")
    t_start = time.time()
    n_done = n_skip = n_err = 0
    for i, j in enumerate(jobs):
        p = out_dir / f"{j['key']}.npz"
        if args.skip_existing and p.is_file():
            n_skip += 1
            continue
        try:
            r = readout_sample(model, tokenizer, image_processor, rec,
                               j["img_path"], j["instruction"], device=args.device)
        except Exception as e:
            n_err += 1
            log.write(json.dumps({"key": j["key"], "error": repr(e)[:400]}) + "\n")
            log.flush()
            continue
        tmp = out_dir / f".{j['key']}.tmp.npz"   # np.savez 会自动补 .npz，tmp 名必须已带
        # 不压缩：float32 全层全头场的 zlib 压缩耗时 1.3 s/样本，是前向（0.22 s）的 6 倍，
        # 而压缩比只有 ~0.7；磁盘换时间（总量 ~4.7 GB，缓存盘余量 66 GB）。
        np.savez(tmp, meta=json.dumps(
            {k: j[k] for k in ("key", "img_id", "itag")} |
            {"n_instr_tok": r["n_instr_tok"], "n_prompt_tok": r["n_prompt_tok"],
             "seq_len": r["seq_len"], "instruction": j["instruction"][:600]}),
            **r["fields"])
        tmp.rename(p)
        n_done += 1
        log.write(json.dumps({"key": j["key"], "t": round(r["t"], 2),
                              "n_instr_tok": r["n_instr_tok"],
                              "seq_len": r["seq_len"]}) + "\n")
        log.flush()
        if (i + 1) % 25 == 0:
            el = time.time() - t_start
            mem = torch.cuda.max_memory_allocated() / 2**30
            print(f"[{i+1}/{len(jobs)}] done={n_done} skip={n_skip} err={n_err} "
                  f"{el/60:.1f}min eta={el/max(n_done,1)*(len(jobs)-i-1)/60:.1f}min "
                  f"peak={mem:.2f}GB", flush=True)
    log.close()
    print(f"FINISHED jobs={len(jobs)} done={n_done} skip={n_skip} err={n_err} "
          f"wall={(time.time()-t_start)/60:.1f}min "
          f"peak_mem={torch.cuda.max_memory_allocated()/2**30:.2f}GB")


if __name__ == "__main__":
    main()
