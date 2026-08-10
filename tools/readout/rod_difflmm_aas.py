"""RO-D / G1b · DiffLMM「attend-and-segment」读出（arXiv 2410.08209, ICCV 2025 Findings）。

在 **一次前向** 内同时导出两套读出，使 G1 canonical 与 attend-and-segment（A&S）
可在完全相同的样本 / 指令 / 生成序列上做**配对**比较：

  A. **pre-softmax logit**（G1/RO-9 canonical 口径，`tools/readout/ro9_gl_attention.py`）
  B. **post-softmax 概率**（A&S 口径，论文 Eq.2 的 A_i^reduced 输入）

并落盘 A&S 归一化所需的「输出序列平均图」：

  论文 Eq.3    A_i^norm = A_i^reduced − (1/r) Σ_{j=1..r} A_j^reduced

其中 A_j^reduced = 对 **全部层、全部头** 平均后、限定在 h×w 个 image token 上的注意力图，
r = 输出（生成）序列长度。官方实现 `aas/gcg.py`：
    attn_mean = attentions.mean(dim=0);  attentions = attentions - attn_mean
（dim=0 = 输出 token 轴）。**⚑ A&S 是纯读出：不改架构、不加训练、前向完全不变**
（论文原文 "without changing their architecture or requiring additional training"）。
本工具因此不需要任何模型改动，只是把 attention 抄出来。

两种 query 行约定（同一前向内都导出，零额外成本，见 NOTES 决策 D-a）：
  - `self`：special token **自身位置**的 query 行（G1/RO-9 口径；也是 VeraRetouch
    `retouch_head` 实际取 hidden state 的位置）
  - `prev`：**产生**该 token 的前一位置 query 行（A&S/HF-generate 口径：
    官方 `aas/infer_attn.py` 取 `x[:, :, -1, ...]`，即 step i 的最后一行 → 产出 o_i）

硬规则（CLAUDE.md 红线 / DOSSIER §5.4）：
- attention 导出必须 eager；运行时双重断言（`config._attn_implementation=="eager"`
  且 forward 返回 attentions 非 None、patch 内取到的 post-softmax 权重非 None）——不回退。
- **不做任何逐图 min-max / softmax 归一化**。Eq.3 的减均值是被检验方法自身的定义
  （沿输出 token 轴），不是我们额外加的逐图归一化，且原样落盘 raw 分量供分析侧组合。
- 落盘一律 float32：post-softmax 概率量级 ~1e-3，减均值后的残差量级 ~1e-5，
  fp16 会把 A&S 的信号直接量化掉。

用法：
  python tools/readout/rod_difflmm_aas.py --samples-json <json> --out-dir <run_dir> \
      --instr-keys reg_a reg_b --device cuda:1 [--limit 20] [--skip-existing]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 机器 CPU 严重超载（load ~130 / 48 核）。torch/OMP 默认按核数开线程，进程内会起 100+ 线程，
# 与别人的作业互相抢核，实测把 greedy 生成从 ~10 s 拖到 ~140 s/样本。
# 必须在 import torch **之前**设环境变量才生效（D-20 排卡纪律：worker ≤ 4）。
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "scache"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from ro9_gl_attention import (  # noqa: E402  复用 G1 的模型加载 / 输入构造 / D-0 / 网格对齐
    CANON_LAYERS,
    GRID,
    N_IMG_TOKENS,
    N_LAYERS,
    TOKEN_TAGS,
    build_inputs,
    interpolate_masked,
    load_model,
    luma_to_grid,
    outlier_mask_from_norms,
)

MODEL_PATH_DEFAULT = "/home/bc/data/models/VeraRetouch"
CONFIG_ADD_DEFAULT = str(REPO / "configs" / "infer_config.yaml")


# ---------------------------------------------------------------------------
# prompt 顺序变体（RO-2 发现：部署 prompt 里 image span=(14,270) 严格早于指令，
# 因果掩码下 image token 表示与指令无关——跨图打乱指令 ρ=0.99995、bit 相同输入 ρ=1.0）
# ---------------------------------------------------------------------------
def build_inputs_ordered(tokenizer, image_processor, img_path: str, instruction: str,
                         order: str = "image_first"):
    """构造输入；`order` 只改 `<image>` 与指令文本的**先后**，其余逐字不变。

    - `image_first`（部署默认，G1/RO-9/RO-2 用的就是这个）：
      `<image>\\n<Style_Retouch_Task>...Instruction: {instr}`
      ⇒ image token 的 key/value 只依赖（系统前缀 + 图像），**与指令无关**。
    - `instr_first`（对照）：`<Style_Retouch_Task>...Instruction: {instr}\\n<image>`
      ⇒ image token 也能看到指令。**这是唯一改变的自变量。**

    ⚑ 注意：模型是按 image_first 做 SFT 的，instr_first 属于分布外输入，
    必须同时报生成质量（fallback 率 / gen_len），否则比较会被"模型直接崩了"混淆。
    """
    import torch
    from PIL import Image

    from llava.constants import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                                 TASK_STYLE_RETOUCH_TOKEN)
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images_, tokenizer_image_token

    body = (f"{TASK_STYLE_RETOUCH_TOKEN}\nNow, you are acting as a Retouch Agent. "
            "I will provide an image and an instruction, please give me a retouch plan "
            f"and retouch tokens.\n Instruction: {instruction}")
    if order == "image_first":
        qs = DEFAULT_IMAGE_TOKEN + "\n" + body
    elif order == "instr_first":
        qs = body + "\n" + DEFAULT_IMAGE_TOKEN
    else:
        raise ValueError(f"未知 prompt order: {order!r}")

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
    _ = torch
    return input_ids, image_tensor, image_512p.size, luma


# ---------------------------------------------------------------------------
# 记录器：一次前向同时抄 pre-softmax logit 与 post-softmax prob
# ---------------------------------------------------------------------------
class DualAttnRecorder:
    """截取给定 query 位置区间在各层的 (pre-softmax logit, post-softmax prob) 行。

    只保留 image token 列（256），并在 head 维取平均（A&S: mean over heads）。
    仅在 teacher-forced 整段前向（q_len > 1）且 `active` 时记录。
    """

    def __init__(self) -> None:
        self.qpos = None            # torch.LongTensor，待抄的 query 位置（升序）
        self.img_slice = (0, 0)     # image token 列区间
        self.pre: dict[int, np.ndarray] = {}   # layer -> (nq, 256) float32
        self.post: dict[int, np.ndarray] = {}
        self.active = False

    def install(self) -> None:
        import transformers.models.qwen2.modeling_qwen2 as m

        if getattr(m, "_rod_patched", False):
            return
        orig = m.eager_attention_forward
        rec = self

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout: float = 0.0, **kwargs):
            out = orig(module, query, key, value, attention_mask, scaling,
                       dropout=dropout, **kwargs)
            if rec.active and query.shape[2] > 1 and rec.qpos is not None:
                import torch

                s0, s1 = rec.img_slice
                # --- A. pre-softmax logit（与 ro9 同一表达式：q@k^T*scaling + causal_mask）
                q = query[:, :, rec.qpos, :]                    # (b, h, nq, d)
                k = key
                if module.num_key_value_groups > 1:
                    k = m.repeat_kv(key, module.num_key_value_groups)
                logits = torch.matmul(q, k.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    cm = attention_mask[:, :, rec.qpos, : k.shape[-2]]
                    logits = logits + cm
                rec.pre[module.layer_idx] = (
                    logits[0][:, :, s0:s1].float().mean(0).cpu().numpy())
                # --- B. post-softmax 概率（eager 返回值；红线：None 直接 raise，不回退）
                aw = out[1]
                if aw is None:
                    raise RuntimeError(
                        "eager_attention_forward 返回 attn_weights=None —— "
                        "非 eager 路径，拒绝继续（DOSSIER §5.4 红线）")
                rec.post[module.layer_idx] = (
                    aw[0][:, rec.qpos, s0:s1].float().mean(0).cpu().numpy())
            return out

        m.eager_attention_forward = patched
        m._rod_patched = True

    def start(self, qpos, img_slice: tuple[int, int]) -> None:
        self.qpos = qpos
        self.img_slice = img_slice
        self.pre, self.post = {}, {}
        self.active = True

    def stop(self) -> None:
        self.active = False


# ---------------------------------------------------------------------------
# 单样本读出
# ---------------------------------------------------------------------------
def readout_sample(model, tokenizer, image_processor, token_ids: dict,
                   recorder: DualAttnRecorder, img_path: str, instruction: str,
                   device: str = "cuda:1", max_new_tokens: int = 512,
                   order: str = "image_first") -> dict:
    """greedy 生成 → teacher-forced eager 前向 → 抄 pre/post-softmax → D-0 修复。

    返回 dict（全部 (…,24,16,16) float32，已过 D-0）：
      pre_self / pre_prev / post_self / post_prev   (3, 24, 16, 16)  三 token
      pre_mean / post_mean                          (2, 24, 16, 16)  [self 约定, prev 约定]
        = Eq.3 的 (1/r)Σ_j A_j^reduced（沿输出 token 轴的平均图）
      outlier_frac (24,) / luma16 / valid16 / gen 元信息
    """
    import torch

    from llava.constants import IMAGE_TOKEN_INDEX

    t0 = time.time()
    input_ids, image_tensor, image_size, luma = build_inputs_ordered(
        tokenizer, image_processor, img_path, instruction, order=order)
    input_ids = input_ids.unsqueeze(0).to(device)
    images = [image_tensor.to(torch.bfloat16).to(device)]

    with torch.inference_mode():
        out = model.generate(
            inputs=input_ids, images=images, image_sizes=[image_size],
            do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
            return_dict_in_generate=True, output_attentions=False, use_cache=True)
    gen_ids = out.sequences[0].tolist()
    t_gen = time.time() - t0

    # fallback（与 G1 同口径）：未自然产出的 token 追加到序列末尾当 query
    fallback, appended = {}, []
    for tag in TOKEN_TAGS:
        tid = token_ids[tag]
        fallback[tag] = tid not in gen_ids
        if fallback[tag]:
            appended.append(tid)
    if appended:
        gen_ids = gen_ids + appended
    r = len(gen_ids)                              # A&S 的输出序列长度

    prompt_ids = input_ids[0].tolist()
    flat = prompt_ids + gen_ids
    img_pos = flat.index(IMAGE_TOKEN_INDEX)
    offset = N_IMG_TOKENS - 1                     # 单图 1 占位 -> 256 embed
    img_start, img_end = img_pos, img_pos + N_IMG_TOKENS
    n_prompt = len(prompt_ids)
    assert img_pos < n_prompt - 1, "image 占位必须在 prompt 内"

    # 展开后位置：生成段整体 +255；抄 r+1 行连续区间
    #   base + 0        = 产生 g_0 的行（最后一个 prompt token）
    #   base + 1 + k    = g_k 自身的位置
    base = (n_prompt - 1) + offset
    qpos_t = torch.arange(base, base + r + 1, device=device)

    full_ids = torch.cat([input_ids[0], torch.tensor(gen_ids, device=device)]).unsqueeze(0)
    model.lens_image_spans = []
    recorder.start(qpos_t, (img_start, img_end))
    t1 = time.time()
    with torch.inference_mode():
        fwd = model(input_ids=full_ids, images=images, image_sizes=[image_size],
                    output_attentions=True, output_hidden_states=True,
                    use_cache=False, return_dict=True)
    recorder.stop()
    t_fwd = time.time() - t1

    # 红线守卫（DOSSIER §5.4）：eager 下 attentions 必须非 None，24 层全捕获
    if fwd.attentions is None or fwd.attentions[0] is None:
        raise RuntimeError("output_attentions 返回 None —— 非 eager 路径，拒绝继续")
    if len(recorder.post) != N_LAYERS or len(recorder.pre) != N_LAYERS:
        raise RuntimeError(f"捕获层数 pre={len(recorder.pre)} post={len(recorder.post)} != {N_LAYERS}")
    span = model.lens_image_spans[0]
    if span[0] != img_start or span[1] != img_end:
        raise RuntimeError(f"image span {span} != ({img_start}, {img_end})")
    if span[2] != full_ids.shape[1] + offset:
        raise RuntimeError(f"展开长度不一致: {span[2]} vs {full_ids.shape[1]} + {offset}")

    # 三 token 在生成序列中的首现下标 k（prompt 内不含 retouch token，已实测核验）
    gidx = {}
    for tag in TOKEN_TAGS:
        tid = token_ids[tag]
        assert tid not in prompt_ids, f"{tag} 出现在 prompt 内，位置约定失效"
        gidx[tag] = gen_ids.index(tid)

    def assemble(rows_by_layer: dict[int, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(nq=r+1, 256) 逐层 → (self(3,24,256), prev(3,24,256), mean(2,24,256))。"""
        sel = np.empty((3, N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
        prv = np.empty((3, N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
        mea = np.empty((2, N_LAYERS, N_IMG_TOKENS), dtype=np.float32)
        for li in range(N_LAYERS):
            rows = rows_by_layer[li]                    # (r+1, 256)
            for ti, tag in enumerate(TOKEN_TAGS):
                sel[ti, li] = rows[1 + gidx[tag]]       # g_k 自身位置
                prv[ti, li] = rows[gidx[tag]]           # 产生 g_k 的行
            mea[0, li] = rows[1:].mean(0)               # self 约定：g_0..g_{r-1}
            mea[1, li] = rows[:-1].mean(0)              # prev 约定：产生 g_0..g_{r-1}
        return sel, prv, mea

    pre_self, pre_prev, pre_mean = assemble(recorder.pre)
    post_self, post_prev, post_mean = assemble(recorder.post)

    # D-0：读出层 image-token 范数 outlier（逐层）+ 4 邻域插值。
    # 对同一层的所有分量用**同一 bad mask**，保证减法在修复后仍是逐格对齐的。
    hs = fwd.hidden_states
    outlier_frac = np.zeros(N_LAYERS, dtype=np.float32)
    bad_by_layer = []
    for li in range(N_LAYERS):
        h = hs[li + 1][0, img_start:img_end].float().cpu().numpy()
        bad = outlier_mask_from_norms(np.linalg.norm(h, axis=-1))
        outlier_frac[li] = float(bad.mean())
        bad_by_layer.append(bad.reshape(GRID, GRID))

    def repair(arr: np.ndarray) -> np.ndarray:
        g = arr.reshape(arr.shape[0], N_LAYERS, GRID, GRID).astype(np.float32)
        outg = np.empty_like(g)
        for li in range(N_LAYERS):
            bad = bad_by_layer[li]
            for ti in range(g.shape[0]):
                outg[ti, li] = (interpolate_masked(g[ti, li], bad) if bad.any()
                                else g[ti, li])
        return outg

    luma16, valid16 = luma_to_grid(luma)
    return {
        "pre_self": repair(pre_self), "pre_prev": repair(pre_prev),
        "pre_mean": repair(pre_mean),
        "post_self": repair(post_self), "post_prev": repair(post_prev),
        "post_mean": repair(post_mean),
        "outlier_frac": outlier_frac, "luma16": luma16, "valid16": valid16,
        "gen_text": tokenizer.decode(gen_ids, skip_special_tokens=True)[:3000],
        "fallback": fallback, "gen_idx": gidx, "gen_len": r,
        "img_span": [int(img_start), int(img_end), int(n_prompt)],
        "t_gen": t_gen, "t_fwd": t_fwd,
        "post_img_mass": float(post_self[0, CANON_LAYERS[0]:CANON_LAYERS[-1] + 1].sum(-1).mean()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--samples-json", help="单批模式：样本 json")
    ap.add_argument("--instr-keys", nargs="+", default=None,
                    help="要跑的 instructions 键，如 reg_a reg_b / syn_a syn_b / shuf")
    ap.add_argument("--out-dir", help="单批模式：输出目录")
    ap.add_argument("--plan", default="",
                    help="多批模式：批次计划 json（[{samples_json,instr_keys,out_dir,only_img_ids}]）。"
                         "一个进程只加载一次模型跑完全部批次——机器 CPU 超载时模型加载要 ~4 min，"
                         "多批合并是硬需求。")
    ap.add_argument("--shard", default="0/1",
                    help="'i/N'：把源列表按 idx%%N==i 切片，供多 worker 并行（确定性、无重叠）")
    ap.add_argument("--only-img-ids", default="", help="可选：换行分隔的 img_id 白名单文件")
    ap.add_argument("--model-path", default=MODEL_PATH_DEFAULT)
    ap.add_argument("--config-add", default=CONFIG_ADD_DEFAULT)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--mem-frac", type=float, default=0.15,
                    help="torch.cuda.set_per_process_memory_fraction（卡 1 共享，硬约束 ≤20GB）")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=4,
                    help="CPU 超载防护：进程内 torch 线程数（D-20：worker ≤ 4）")
    args = ap.parse_args()

    import torch

    from api import instr_hash  # tools/scache

    torch.set_num_threads(max(1, args.torch_threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass    # 已有并行工作启动过 -> 保持默认，不致命
    if args.device.startswith("cuda") and args.mem_frac > 0:
        torch.cuda.set_per_process_memory_fraction(
            args.mem_frac, device=int(args.device.split(":")[1]))

    if args.plan:
        batches = json.loads(Path(args.plan).read_text())
    else:
        batches = [{"samples_json": args.samples_json, "instr_keys": args.instr_keys,
                    "out_dir": args.out_dir, "only_img_ids": args.only_img_ids}]

    si, sn = (int(x) for x in args.shard.split("/"))
    assert 0 <= si < sn, args.shard

    model, tokenizer, image_processor, token_ids = load_model(
        args.model_path, args.config_add, args.device)
    recorder = DualAttnRecorder()
    recorder.install()

    grand_done = grand_skip = grand_err = 0
    t_all = time.time()
    for b in batches:
        samples = json.loads(Path(b["samples_json"]).read_text())
        only = b.get("only_img_ids") or ""
        if only:
            keep = {ln.strip() for ln in Path(only).read_text().splitlines() if ln.strip()}
            samples = [s for s in samples if s["img_id"] in keep]
        if args.limit > 0:
            samples = samples[: args.limit]
        samples = samples[si::sn]          # 确定性分片，worker 间无重叠
        out_dir = Path(b["out_dir"])
        stacks_dir = out_dir / "stacks"
        stacks_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / f"run_log.shard{si}of{sn}.jsonl"
        n_done = n_skip = n_err = 0
        t_start = time.time()
        print(f"### BATCH {out_dir.name} keys={b['instr_keys']} shard={si}/{sn} "
              f"sources={len(samples)}", flush=True)
        with open(log_path, "a", encoding="utf-8") as logf:
            for i, s in enumerate(samples):
                img_id, img_path = s["img_id"], s["img_path"]
                for itag in b["instr_keys"]:
                    instr = s["instructions"].get(itag)
                    if instr is None:
                        continue
                    ih = instr_hash(instr)
                    npz_path = stacks_dir / f"{img_id}__{ih}.npz"
                    if args.skip_existing and npz_path.exists():
                        n_skip += 1
                        continue
                    try:
                        r = readout_sample(model, tokenizer, image_processor, token_ids,
                                           recorder, img_path, instr, device=args.device,
                                           max_new_tokens=args.max_new_tokens,
                                           order=b.get("order", "image_first"))
                    except Exception as e:   # 单样本失败不拖垮批次
                        n_err += 1
                        logf.write(json.dumps({"img_id": img_id, "itag": itag,
                                               "error": repr(e)[:500]}) + "\n")
                        logf.flush()
                        continue
                    # np.savez_compressed 会自动补 .npz，故临时名本身就以 .npz 结尾
                    tmp = stacks_dir / f"{img_id}__{ih}.s{si}.tmp.npz"
                    np.savez_compressed(
                        tmp,
                        pre_self=r["pre_self"], pre_prev=r["pre_prev"], pre_mean=r["pre_mean"],
                        post_self=r["post_self"], post_prev=r["post_prev"],
                        post_mean=r["post_mean"],
                        outlier_frac=r["outlier_frac"], luma16=r["luma16"],
                        valid16=r["valid16"],
                        meta=json.dumps({
                            "img_id": img_id, "itag": itag, "instr": instr, "ih": ih,
                            "pool": s.get("pool"), "build": s.get("build"),
                            "fallback": r["fallback"], "gen_idx": r["gen_idx"],
                            "gen_len": r["gen_len"], "donor_img_id": s.get("donor_img_id"),
                            "prompt_order": b.get("order", "image_first"),
                            "img_span": r["img_span"], "gen_text": r["gen_text"],
                        }))
                    os.replace(tmp, npz_path)   # 原子落盘，断点续跑安全
                    logf.write(json.dumps({
                        "img_id": img_id, "itag": itag, "ih": ih, "gen_len": r["gen_len"],
                        "fallback": r["fallback"], "post_img_mass": round(r["post_img_mass"], 6),
                        "outlier_frac_mean": float(r["outlier_frac"].mean()),
                        "t_gen": round(r["t_gen"], 2), "t_fwd": round(r["t_fwd"], 2)}) + "\n")
                    logf.flush()
                    n_done += 1
                if (i + 1) % 5 == 0:
                    el = time.time() - t_start
                    print(f"[{out_dir.name} s{si} {i+1}/{len(samples)}] done={n_done} "
                          f"skip={n_skip} err={n_err} elapsed={el/60:.1f}min "
                          f"eta={el/(i+1)*(len(samples)-i-1)/60:.1f}min", flush=True)
        print(f"BATCH DONE {out_dir.name} s{si}: done={n_done} skip={n_skip} err={n_err} "
              f"wall={(time.time()-t_start)/60:.1f}min", flush=True)
        grand_done += n_done
        grand_skip += n_skip
        grand_err += n_err

    print(f"FINISHED shard={si}/{sn} done={grand_done} skipped={grand_skip} "
          f"err={grand_err} wall={(time.time()-t_all)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
