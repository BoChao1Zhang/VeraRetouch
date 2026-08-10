"""RO-9b · RO-9 读出算子的四处修复（EXPERIMENTS_v3 §2.1 RO-9 行的追加臂）。

RO-9 在 16×16 同口径下 AUC=0.648，而零训练 CLIP（RO-1）0.941。本模块把四处可疑
逐一做成可 A/B 的实验轴：

  ① 层选择    canonical L8–15 不含实测最优层（L20/L22）→ 逐头栈全导，层选择在分析端做
  ② head 聚合  pre-softmax logit 逐头量纲不同，直接算术平均让尺度大的头主导
              → 导出保留 head 维（pre / post 两读法），聚合方式在分析端 A/B
  ③ self-self  RO-1 的 +0.72 全部来自"改末层 attention"。本模块提供两组：
              组 A = 在**自家视觉塔 FastViTHD** 的 MHSA 上施加 CSA/kkᵀ/qqᵀ 手术（`VisionSurgery`）
              组 B = LM 侧 GL→image 的 q-k / k-k / q-q 三种交互（导出即三份场）
  ④ 后处理     D-0 在 RO-9 上实测空转；这里把 outlier 掩膜与 K 亲和度矩阵一并落盘，
              后处理阶梯在分析端逐档单列

**红线**（CLAUDE.md 速查表）：
- attention 导出必须 eager；FA2/SDPA 下 `output_attentions` 返回 None **不回退，直接 raise**。
- s 禁逐图 min-max/softmax 归一化：落盘一律原始 logit / 原始概率；
  head 聚合用的 z-score 统计量在**训练折的全部图全部格**上估（数据集级，非逐图）。
- 逐像素算子禁 (x,y)/邻域/MLP —— 本模块只读不训练，n/a。

**架构核实（2026-08-03，一手源）**：
  FastViTHD（`apple/ml-fastvlm@main llava/model/multimodal_encoder/mobileclip/mci.py::fastvithd`）
  layers=[2,12,24,4,2] embed_dims=[96,192,384,768,1536]
  token_mixers=("repmixer","repmixer","repmixer","attention","attention")
  ⇒ 只有 **stage 3 / stage 4** 有 self-attention（`network[7]` 4 块 dim768 @32×32、
     `network[10]` 2 块 dim1536 @16×16），**无 CLS / 无 register token**，
     空间布局始终是 (B,C,H,W)，MHSA 内部 flatten 成 (B,N,C)。
  末层 = `network[10][1].token_mixer`，其输出经 `conv_exp`(3072) → `mm_projector` → LLM。

用法见 `experiments/RO9b_readout_fix_20260803/`。
"""
from __future__ import annotations

import contextlib
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "readout"))
sys.path.insert(0, str(REPO / "tools" / "scache"))

from ro9_gl_attention import (CANON_LAYERS, GRID, N_IMG_TOKENS,  # noqa: E402
                              N_LAYERS, GL_TOKEN, build_inputs, load_model,
                              luma_to_grid, outlier_mask_from_norms)

N_HEADS = 14                     # 实测（RO-3 verify_env V1）：24 层 / 14 heads / 2 KV heads
N_KV_HEADS = 2


# ===========================================================================
# 组 B / ①② —— LM 侧逐头 pre-softmax 捕获（q-k / k-k / q-q 三种交互）
# ===========================================================================
class PerHeadRecorder:
    """截取指定 query 位置在各层**逐头**的三种交互场 + image token 的 K 矩阵。

    与 `ro9_gl_attention.PreSoftmaxRecorder` 的区别：
      - **不做 head 平均**（②的实验轴）；
      - 另出 `post`（整行 softmax 后切 image 列，量纲统一到概率）；
      - 另出 `kk`（k_GL·k_img）与 `qq`（q_GL·q_img）——③组 B 的类比改造；
      - 另出 `K_img`（2 个 KV head 的 image 段 key）——④的 self-self 亲和度传播。

    只在 `active` 且 q_len>1（teacher-forced 整段前向）时记录；生成阶段不记录。
    """

    def __init__(self) -> None:
        self.qpos: list[int] = []
        self.img_span: tuple[int, int] = (0, 0)
        self.active = False
        self.out: dict[str, dict[int, np.ndarray]] = {}

    def install(self) -> None:
        import torch
        import transformers.models.qwen2.modeling_qwen2 as m

        if getattr(m, "_ro9b_patched", False):
            return
        orig = m.eager_attention_forward
        rec = self

        def patched(module, query, key, value, attention_mask, scaling,
                    dropout: float = 0.0, **kwargs):
            if rec.active and query.shape[2] > 1 and rec.qpos:
                li = module.layer_idx
                i0, i1 = rec.img_span
                g = module.num_key_value_groups
                k_full = m.repeat_kv(key, g) if g > 1 else key      # (1,H,T,d)
                q_sel = query[:, :, rec.qpos, :]                    # (1,H,nq,d)
                k_sel = k_full[:, :, rec.qpos, :]                   # (1,H,nq,d)

                pre_full = torch.matmul(q_sel, k_full.transpose(2, 3)) * scaling
                if attention_mask is not None:
                    cm = attention_mask[:, :, rec.qpos, : k_full.shape[-2]]
                    pre_full = pre_full + cm
                post_full = torch.softmax(pre_full.float(), dim=-1)
                # k-k：把 query 端换成同位置的 key（"self-self" 的 LM 侧字面类比）
                kk = torch.matmul(k_sel, k_full.transpose(2, 3)) * scaling
                # q-q：把 key 端换成 image 位置的 query
                q_img = query[:, :, i0:i1, :]
                qq = torch.matmul(q_sel, q_img.transpose(2, 3)) * scaling

                rec.out.setdefault("pre", {})[li] = \
                    pre_full[0, :, :, i0:i1].float().cpu().numpy()
                rec.out.setdefault("post", {})[li] = \
                    post_full[0, :, :, i0:i1].cpu().numpy()
                rec.out.setdefault("kk", {})[li] = \
                    kk[0, :, :, i0:i1].float().cpu().numpy()
                rec.out.setdefault("qq", {})[li] = qq[0].float().cpu().numpy()
                rec.out.setdefault("kimg", {})[li] = \
                    key[0, :, i0:i1, :].float().cpu().numpy()      # (KV,256,d)
            return orig(module, query, key, value, attention_mask, scaling,
                        dropout=dropout, **kwargs)

        m.eager_attention_forward = patched
        m._ro9b_patched = True

    def start(self, qpos: list[int], img_span: tuple[int, int]) -> None:
        self.qpos = list(qpos)
        self.img_span = img_span
        self.out = {}
        self.active = True

    def stop(self) -> None:
        self.active = False


# ===========================================================================
# 组 A —— 视觉塔 FastViTHD 的 self-self 手术
# ===========================================================================
#   算子定义与 RO-1 的 vendored CLIP fork 逐字同源（`clip_naclip/model.py::custom_attn`，
#   出处见 `tools/readout/clip_naclip/VENDOR.md`）：
#     csa       = softmax(qqᵀ·scale) + softmax(kkᵀ·scale)          [SCLIP]
#     kk        = softmax(kkᵀ·scale)                                [NACLIP 去高斯]
#     naclip    = softmax(kkᵀ·scale + gaussian_bias(std))           [NACLIP, std=5]
#     clearclip = softmax(qqᵀ·scale)  且 arch=reduced（丢残差丢 FFN）[ClearCLIP]
#   arch：vanilla = 原 AttentionBlock（残差 + FFN）；reduced = x ← ls1·token_mixer(norm(x))
VIS_ATTN_STRATEGIES = ("vanilla", "csa", "kk", "naclip", "clearclip", "qq")
VIS_ARCHS = ("vanilla", "reduced")


def _gaussian_bias(h: int, w: int, std: float, device, dtype):
    """NACLIP 的高斯邻域偏置（`naclip` 官方 `clip/model.py::gaussian_window` 同语义）。

    加性偏置 b[i,j] = −‖pos_i − pos_j‖² / (2σ²)，softmax 前加到 kkᵀ 上。
    """
    import torch

    ys, xs = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32),
                            torch.arange(w, device=device, dtype=torch.float32),
                            indexing="ij")
    p = torch.stack([ys.reshape(-1), xs.reshape(-1)], dim=-1)       # (N,2)
    d2 = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    return (-d2 / (2.0 * std * std)).to(dtype)


class VisionSurgery:
    """上下文管理器：把 FastViTHD 指定 AttentionBlock 的注意力换成 self-self 变体。

    `blocks` 用 (stage_idx_in_network, block_idx) 指定，默认 = 末层
    `network[10][1]`（stage 4 的第 2 块，16×16 网格，dim 1536, 48 heads）。
    """

    LAST = ((10, 1),)
    STAGE4 = ((10, 0), (10, 1))
    STAGE34 = ((7, 0), (7, 1), (7, 2), (7, 3), (10, 0), (10, 1))

    def __init__(self, model, attn: str = "vanilla", arch: str = "vanilla",
                 std: float = 5.0, blocks=LAST):
        if attn not in VIS_ATTN_STRATEGIES:
            raise ValueError(f"attn={attn!r} 不在 {VIS_ATTN_STRATEGIES}")
        if arch not in VIS_ARCHS:
            raise ValueError(f"arch={arch!r} 不在 {VIS_ARCHS}")
        self.attn, self.arch, self.std = attn, arch, std
        vt = model.get_vision_tower().vision_tower.model
        self.net = vt.network
        self.blocks = [self.net[s][b] for s, b in blocks]
        self._saved: list = []
        self._gcache: dict = {}

    def _mhsa_forward(self, mhsa, x):
        import torch

        B, C, H, W = x.shape
        N = H * W
        xf = torch.flatten(x, start_dim=2).transpose(-2, -1)        # (B,N,C)
        qkv = (mhsa.qkv(xf).reshape(B, N, 3, mhsa.num_heads, mhsa.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)
        s = mhsa.scale
        if self.attn == "vanilla":
            attn = ((q * s) @ k.transpose(-2, -1)).softmax(dim=-1)
        elif self.attn in ("clearclip", "qq"):
            attn = ((q * s) @ q.transpose(-2, -1)).softmax(dim=-1)
        elif self.attn == "kk":
            attn = ((k * s) @ k.transpose(-2, -1)).softmax(dim=-1)
        elif self.attn == "naclip":
            logits = (k * s) @ k.transpose(-2, -1)
            key = (H, W, x.device, x.dtype)
            if key not in self._gcache:
                self._gcache[key] = _gaussian_bias(H, W, self.std, x.device, x.dtype)
            attn = (logits + self._gcache[key]).softmax(dim=-1)
        elif self.attn == "csa":
            attn = (((q * s) @ q.transpose(-2, -1)).softmax(dim=-1)
                    + ((k * s) @ k.transpose(-2, -1)).softmax(dim=-1))
        else:                                                       # pragma: no cover
            raise AssertionError(self.attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = mhsa.proj(out)
        return out.transpose(-2, -1).reshape(B, C, H, W)

    def _block_forward(self, blk, x):
        mixed = self._mhsa_forward(blk.token_mixer, blk.norm(x))
        if self.arch == "reduced":
            # ClearCLIP `ignore_residual=True` + 丢 FFN 的 FastViT 类比；
            # 保留 layer_scale_1（它是该 block 自身的逐通道增益，不是残差的一部分）
            return blk.layer_scale_1 * mixed if blk.use_layer_scale else mixed
        if blk.use_layer_scale:
            x = x + blk.drop_path(blk.layer_scale_1 * mixed)
            return x + blk.drop_path(blk.layer_scale_2 * blk.convffn(x))
        x = x + blk.drop_path(mixed)
        return x + blk.drop_path(blk.convffn(x))

    def __enter__(self):
        if self.attn == "vanilla" and self.arch == "vanilla":
            return self                                             # no-op 基线
        import types
        for blk in self.blocks:
            self._saved.append((blk, blk.forward))
            blk.forward = types.MethodType(
                lambda _self, x, _o=self: _o._block_forward(_self, x), blk)
        return self

    def __exit__(self, *exc):
        for blk, fwd in self._saved:
            blk.forward = fwd
        self._saved.clear()
        return False


# ===========================================================================
# 组 A 的读出：emb（mm_projector 输出）logit lens —— RO-2 已验证的唯一词特异读出点
# ===========================================================================
@contextlib.contextmanager
def _no_grad():
    import torch
    with torch.inference_mode():
        yield


def vision_features(model, image_tensor):
    """视觉塔前向 → (1, 256, 3072)。

    `MobileCLIPVisionTower.forward_images` 传 list 时返回 list（每项 (1,256,3072)），
    传 4D 张量时返回 (B,256,3072)——这里统一成后者。
    """
    import torch

    out = model.get_model().get_vision_tower()([image_tensor])
    if isinstance(out, (list, tuple)):
        out = torch.cat(list(out), dim=0)
    return out


def vision_emb_readout(model, image_tensor, image_size, word_ids: dict,
                       device: str = "cuda:0"):
    """图像 → 视觉塔 → mm_projector → 终端 RMSNorm → lm_head → 逐词概率场。

    与 RO-2 的 `emb` 读出点**逐字同一个量**（RO-2 REPORT §八 结论 1）：
    `p(w|patch) = softmax_V(lm_head(model.model.norm(mm_projector(vis_feat))))[w]`。
    softmax **只在词表维**做，空间维零归一化（红线）。

    返回 dict：{scheme: (16,16) float32 概率场} + `logit_{scheme}`（原始 logit 档）+
    `featnorm` (16,16)（emb 的 L2 范数，供 D-0 outlier 判据）。
    """
    import torch

    with _no_grad():
        vis = vision_features(model, image_tensor)                   # (1,256,3072)
        emb = model.get_model().mm_projector(vis)                    # (1,256,896)
        h = model.get_model().norm(emb)
        logits = torch.nn.functional.linear(h.float(),
                                            model.lm_head.weight.float())  # (1,256,V)
        lse = torch.logsumexp(logits, dim=-1)                        # (1,256)
        out = {"featnorm": emb[0].float().norm(dim=-1).cpu().numpy()
               .reshape(GRID, GRID).astype(np.float32)}
        for sch, ids in word_ids.items():
            if not ids:
                continue
            idx = torch.tensor(ids, device=logits.device, dtype=torch.long)
            sel = logits.index_select(-1, idx)[0]                    # (256, n_ids)
            prob = torch.exp(sel - lse[0][:, None]).sum(-1)
            out[sch] = prob.cpu().numpy().reshape(GRID, GRID).astype(np.float32)
            out["logit_" + sch] = (torch.logsumexp(sel, dim=-1).cpu().numpy()
                                   .reshape(GRID, GRID).astype(np.float32))
    return out


# ===========================================================================
# 单样本导出（LM 侧逐头栈）
# ===========================================================================
def readout_perhead(model, tokenizer, image_processor, token_ids: dict,
                    recorder: PerHeadRecorder, img_path: str, instruction: str,
                    device: str = "cuda:0", max_new_tokens: int = 512,
                    gen_ids_cache: list[int] | None = None) -> dict:
    """RO-9 同款流程（生成 → teacher-forced eager 前向），但保留 head 维。

    `gen_ids_cache` 非 None 时**跳过生成**，直接 teacher-force 给定序列——
    用于"换视觉塔算子但保持生成的 plan 不变"的受控干预（成本 0.3 s/样本 vs 10 s）。
    """
    import torch

    from llava.constants import IMAGE_TOKEN_INDEX

    t0 = time.time()
    input_ids, image_tensor, image_size, luma = build_inputs(
        tokenizer, image_processor, img_path, instruction)
    input_ids = input_ids.unsqueeze(0).to(device)
    images = [image_tensor.to(torch.bfloat16).to(device)]

    t_gen = 0.0
    if gen_ids_cache is None:
        with torch.inference_mode():
            out = model.generate(
                inputs=input_ids, images=images, image_sizes=[image_size],
                do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                return_dict_in_generate=True, output_attentions=False, use_cache=True)
        gen_ids = out.sequences[0].tolist()
        t_gen = time.time() - t0
    else:
        gen_ids = list(gen_ids_cache)

    fallback = {}
    appended: list[int] = []
    for tag, tid in token_ids.items():
        if tid in gen_ids:
            fallback[tag] = False
        else:
            fallback[tag] = True
            appended.append(tid)
    if appended:
        gen_ids = gen_ids + appended

    full_ids = torch.cat([input_ids[0], torch.tensor(gen_ids, device=device)]).unsqueeze(0)
    flat = full_ids[0].tolist()
    img_pos = flat.index(IMAGE_TOKEN_INDEX)
    offset = N_IMG_TOKENS - 1
    img_span = (img_pos, img_pos + N_IMG_TOKENS)

    def expanded(pos: int) -> int:
        return pos + offset if pos > img_pos else pos

    qpos = [expanded(flat.index(token_ids["light"]))]

    model.lens_image_spans = []
    recorder.start(qpos, img_span)
    t1 = time.time()
    with torch.inference_mode():
        fwd = model(input_ids=full_ids, images=images, image_sizes=[image_size],
                    output_attentions=True, output_hidden_states=True,
                    use_cache=False, return_dict=True)
    recorder.stop()
    t_fwd = time.time() - t1

    # 红线守卫（DOSSIER §5.4）：eager 下 attentions 必须非 None，不回退
    if fwd.attentions is None or fwd.attentions[0] is None:
        raise RuntimeError("output_attentions 返回 None——非 eager 路径，拒绝继续")
    for k in ("pre", "post", "kk", "qq", "kimg"):
        if len(recorder.out.get(k, {})) != N_LAYERS:
            raise RuntimeError(f"{k} 捕获层数 {len(recorder.out.get(k, {}))} != {N_LAYERS}")
    span = model.lens_image_spans[0]
    if (span[0], span[1]) != img_span:
        raise RuntimeError(f"image span 实测 {span[:2]} != 预算 {img_span}")

    def stack(key: str) -> np.ndarray:
        return np.stack([recorder.out[key][li] for li in range(N_LAYERS)], axis=0)

    pre = stack("pre")[:, :, 0, :].reshape(N_LAYERS, N_HEADS, GRID, GRID)
    post = stack("post")[:, :, 0, :].reshape(N_LAYERS, N_HEADS, GRID, GRID)
    kk = stack("kk")[:, :, 0, :].reshape(N_LAYERS, N_HEADS, GRID, GRID)
    qq = stack("qq")[:, :, 0, :].reshape(N_LAYERS, N_HEADS, GRID, GRID)
    kimg = stack("kimg")                                    # (24, KV, 256, d)

    hs = fwd.hidden_states
    hnorm = np.stack([hs[li + 1][0, img_span[0]:img_span[1]].float().norm(dim=-1)
                      .cpu().numpy() for li in range(N_LAYERS)])     # (24,256)
    outlier = np.stack([outlier_mask_from_norms(hnorm[li]).reshape(GRID, GRID)
                        for li in range(N_LAYERS)])

    luma16, valid16 = luma_to_grid(luma)
    return {
        "pre": pre.astype(np.float32), "post": post.astype(np.float32),
        "kk": kk.astype(np.float32), "qq": qq.astype(np.float32),
        "kimg": kimg.astype(np.float16),
        "hnorm": hnorm.astype(np.float32), "outlier": outlier,
        "luma16": luma16, "valid16": valid16,
        "gen_ids": np.array(gen_ids, dtype=np.int64),
        "gen_text": tokenizer.decode(gen_ids, skip_special_tokens=True)[:1500],
        "fallback": fallback, "gen_len": len(gen_ids),
        "t_gen": t_gen, "t_fwd": t_fwd, "t_all": time.time() - t0,
    }


__all__ = [
    "PerHeadRecorder", "VisionSurgery", "vision_emb_readout", "readout_perhead",
    "VIS_ATTN_STRATEGIES", "VIS_ARCHS", "N_HEADS", "N_KV_HEADS",
    "CANON_LAYERS", "GRID", "N_LAYERS", "N_IMG_TOKENS", "GL_TOKEN",
    "load_model", "build_inputs", "luma_to_grid",
]
