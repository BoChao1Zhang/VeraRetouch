# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/train_q3vl_adapt3.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / S2F-B：全参 S1F-FULL 之上的 stage-2（分支文件,adapt2 一字不动,E1）。

相对 train_q3vl_adapt2.py 的三处差异(config diff 亦仅限这三类键):
  (1) 基座 = `[data] base_weights_dir` 指向 **S1F-FULL 的全参 ckpt**(model/ 子目录),
      其权重**全部冻结**;不再从 stage-1 LoRA 续训,而是在其上**新挂** LoRA。
      阶段 token 的 6 行已在全参权重里,故不读 new_token_embeddings.pt。
  (2) 目标 = BK-FULL 的 edit_enc 产出(`targets_bkfull/target_latents_bkfull.pt`)。
  (3) prompt = 逐样本真实编辑指令(instruction_mode/salt,与 S1F-FULL 同规则)。

原 docstring:
EPR-051 / stage-2 变体 B/C（分支文件，723 的源码逐字冻结不动，E1）。

B (`SPRF_ADAPT_S2B`): 损失 = w_latent*SmoothL1(latent) + w_ce*CE，CE 只在被监督位置、
   分块、与 stage-1 重跑版同实现（绕开整序列 logits）。其余与 723 逐字相同。
   动机：723 的损失没有语言项，2,266 步后 LoRA 只服务 latent，自生成能力被摧毁
   （实测 0/6 阶段 token、退化重复；stage-1 单独 6/6 rank 1、格式正确）。
C (`SPRF_ADAPT_S2C`): LoRA 冻结（stage-1 权重原样）、embedding 冻结，只训 adapter。
   由 [train].freeze_lora 控制。

原 stage-2 docstring：
EPR-051 / SFT+ADAPT stage 2 -- joint LoRA + span-pool adapter -> LUT latent.

用户裁决(2026-09-02):stage 2 主线 = 在 stage-1 LoRA 基础上**继续联合训练**
LoRA 与 adapter,目标为 latent 对齐损失;读出 = 逐段 span mean-pool(带梯度)。

实验登记:

1. 数据   同一冻结快照 snap_sft1(train 18,119 / val 955);输入 = y + 冻结指令;
          teacher-forced 文本 = 该样本的 GT CoT(即 R1/oracle_text 口径)。
2. 模型   Qwen3-VL-4B + stage-1 LoRA(继续训练,r=16/alpha=16/dropout=0,
          语言塔最后 8 块 q/k/v/o)+ Adapter(2560 -> H -> H -> 128,带 stage 嵌入)。
          读出 z_m = mean_{t in span_m} norm(h^{(-1)})[t],fp32
          (SEGMENT_HIDDEN_LAYER=-1 / SEGMENT_HIDDEN_FINAL_NORM=True,
           q3vl/whereb/contracts.py:30,32)。
3. 数学   L = (1/(B*K*128)) * sum_{b,k} SmoothL1(Adapter(z)_bk - t_bk ; beta=0.05)
          t_bk = target_latents[inv_row(lut_id_bk)],即 T-ALIGN 回归的同一对象
          (align_time_film/target_latents.pt,XL formal backbone
           ckpt_formal.pt sha e61a5069…),冻结。
          w_latent = 1.0,w_pixel = 0.0(预注册为关,未实现)。
          槽位映射:读出槽 m(1..6,恢复序)-> 链位 k = 6 - m。
4. 优化器 AdamW,LoRA 与 adapter 分组 lr,见 [optim]。

与 EPR-033 的一处记录在案的偏差:EPR-033 的 h_reply 把每个 micro-batch 的计算图
留到一次 backward(32 样本 × ~167 token,峰值 68.09 GiB alloc / 76.50 GiB reserved)。
本臂的 latent 损失按样本可分,且回复长 ~1,650 token,故改为**逐 micro-batch
backward 并释放计算图**(真梯度累积)。这是显存策略差异,不是损失差异。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from veraretouch_sprf.data import cot_text as C  # noqa: E402
from veraretouch_sprf.models.vlm import q3vl_common as Q  # noqa: E402
from veraretouch_sprf.data import q3vl_text as T  # noqa: E402
from veraretouch_sprf.data.q3vl_data import CoTDataset, collate  # noqa: E402
from veraretouch_sprf.eval import guards  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


class Adapter(nn.Module):
    """z_m (2560) -> 128-d LUT latent, weights shared across the 6 slots."""

    def __init__(self, in_dim=2560, hidden=768, out_dim=128, n_slots=6):
        super().__init__()
        self.slot = nn.Embedding(n_slots, in_dim)
        nn.init.zeros_(self.slot.weight)          # identity at step 0
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, out_dim))

    def forward(self, z):                          # (B, 6, in_dim)
        k = torch.arange(z.shape[1], device=z.device)
        return self.net(z + self.slot(k).unsqueeze(0))


def build_targets(keys, assets_index, lut_names, target_latents):
    """slot m (1..6, restoration order) -> chain k = 6 - m."""
    row_of = {n: i for i, n in enumerate(lut_names)}
    idx = assets_index["index"]
    Tt = torch.zeros(len(keys), 6, target_latents.shape[1])
    for i, key in enumerate(keys):
        chain = idx[key]["chain"]
        if [c["k"] for c in chain] != list(range(6)):
            Q.die(f"{key}: chain k order {[c['k'] for c in chain]}")
        for m in range(1, 7):
            lut = chain[6 - m]["lut"]
            r = row_of.get(lut)
            if r is None:
                Q.die(f"{key}: lut_id {lut} absent from the inverse-table cache")
            Tt[i, m - 1] = target_latents[r]
    return Tt


def chunked_ce(mm_out_hidden, lm_head, batch, device, chunk: int = 512):
    """CE over ONLY supervised positions, lm_head applied per chunk under
    checkpoint(use_reentrant=False) so (B,T,151936) logits are never materialised.
    Reuses the hidden states the span readout already computed -- one forward."""
    h = mm_out_hidden[:, :-1, :]
    tgt = batch["labels"].to(device)[:, 1:]
    m = tgt != Q.IGNORE_INDEX
    hs, ts = h[m], tgt[m]
    n = int(ts.numel())
    if n == 0:
        Q.die("no supervised positions in this batch")

    def _piece(hh, tt):
        # hidden is .float() because the span-pool readout contract demands fp32,
        # but lm_head is bf16 -> cast for the matmul, then upcast the logits for CE.
        return torch.nn.functional.cross_entropy(
            lm_head(hh.to(lm_head.weight.dtype)).float(), tt, reduction="sum")

    tot = hs.new_zeros(())
    for i in range(0, n, chunk):
        tot = tot + torch.utils.checkpoint.checkpoint(
            _piece, hs[i:i + chunk], ts[i:i + chunk], use_reentrant=False)
    return tot / n


def readout_z_and_hidden(mm, batch, device, dtype):
    """Same contract as readout_z but also returns the hidden states, so the CE
    term costs no extra forward."""
    out = mm(input_ids=batch["input_ids"].to(device),
             attention_mask=batch["attention_mask"].to(device),
             pixel_values=batch["pixel_values"].to(device, dtype),
             image_grid_thw=batch["image_grid_thw"].to(device), return_dict=True)
    hidden = out.last_hidden_state.float()
    if hidden.shape[-1] != Q.Z_DIM:
        Q.die(f"h is {hidden.shape[-1]}-dim, contract says {Q.Z_DIM}")
    z = torch.stack([T.span_pool(hidden[i], batch["spans"][i])
                     for i in range(hidden.shape[0])], dim=0)
    return z, hidden


def readout_z(mm, batch, device, dtype):
    """(B, 6, 2560) fp32 span-pooled conditions, gradient intact.

    `mm` is the Qwen3VLModel.  Its `last_hidden_state` is the last decoder block's
    output ALREADY through the text tower's final norm, so this reproduces the
    EPR-033 contract (SEGMENT_HIDDEN_LAYER=-1, SEGMENT_HIDDEN_FINAL_NORM=True)
    exactly -- and NO further norm may be applied here.

    Unlike the forward-hook route this is an ordinary graph node, so gradient
    checkpointing is safe (A-9 is hook-specific), and lm_head is never invoked,
    so full-sequence vocab logits are never materialised.
    """
    out = mm(input_ids=batch["input_ids"].to(device),
             attention_mask=batch["attention_mask"].to(device),
             pixel_values=batch["pixel_values"].to(device, dtype),
             image_grid_thw=batch["image_grid_thw"].to(device),
             return_dict=True)
    hidden = out.last_hidden_state.float()
    if hidden.shape[-1] != Q.Z_DIM:
        Q.die(f"h is {hidden.shape[-1]}-dim, contract says {Q.Z_DIM}")
    return torch.stack([T.span_pool(hidden[i], batch["spans"][i])
                        for i in range(hidden.shape[0])], dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--memprobe", action="store_true",
                    help="run --max-steps steps, report the reserved PLATEAU, exit "
                         "before saving.  Must be >=100 steps: stage-1 showed "
                         "reserved climbing until ~step 100 before plateauing, so a "
                         "short probe under-reports by ~2x (D-sft1).")
    args = ap.parse_args()
    if args.memprobe and args.max_steps < 100:
        print("FATAL: --memprobe needs --max-steps >= 100 (D-sft1: the reserved "
              "plateau is only visible after ~100 steps)", flush=True)
        raise SystemExit(2)

    cfg = tomllib.load(open(args.config, "rb"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = cfg["optim"]
    torch.manual_seed(o["seed"])
    device = cfg["run"]["device"]
    dtype = dict(bf16=torch.bfloat16, fp32=torch.float32)[cfg["model"]["dtype"]]

    # ---- targets (frozen) --------------------------------------------------
    tl = torch.load(cfg["target"]["target_latents"], map_location="cpu")
    if tl["indexed_by"] != "inv_table row / lut_id":
        Q.die(f"unexpected target_latents indexing {tl['indexed_by']!r}")
    lat = tl["target_latents"]
    if lat.shape[1] != 128:
        Q.die(f"target latent dim {lat.shape[1]} != 128")
    names = list(json.loads(Path(cfg["target"]["inv_index"]).read_text())["names"])
    if lat.shape[0] != len(names) + 1:
        Q.die(f"target_latents rows {lat.shape[0]} != len(names)+1 {len(names)+1}")

    snap = Path(cfg["data"]["snapshot_dir"])
    freeze = json.loads((snap / "FREEZE.json").read_text())
    if freeze["records_sha256"] != sha256_file(snap / "records.jsonl"):
        Q.die("snapshot records.jsonl sha != FREEZE.json")
    assets_index = json.loads((snap / "assets_index.json").read_text())
    train_keys = json.loads((snap / "split_train_keys.json").read_text())
    val_keys = json.loads((snap / "split_val_keys.json").read_text())
    # Same spec-5 aspect>4 reject list that S1F-FULL needed: S2's train split still
    # contains the key that crashed that run 11 h in.  Excluded and counted here.
    rej_path = cfg["data"].get("reject_keys", "")
    rejected = sorted(json.loads(Path(rej_path).read_text())) if rej_path else []
    rej = set(rejected)
    n0t, n0v = len(train_keys), len(val_keys)
    train_keys = [k for k in train_keys if k not in rej]
    val_keys = [k for k in val_keys if k not in rej]
    reject_report = dict(file=rej_path, n_listed=len(rejected),
                         n_removed_train=n0t - len(train_keys),
                         n_removed_val=n0v - len(val_keys),
                         n_train=len(train_keys), n_val=len(val_keys))
    print(f"[adapt] spec-5 reject: {json.dumps(reject_report)}", flush=True)
    if args.limit_train:
        train_keys = train_keys[: args.limit_train]
        val_keys = val_keys[: min(len(val_keys), 16)]

    held = set(json.loads(Path(cfg["data"]["heldout_ids"]).read_text()))
    key_to_id = {json.loads(l)["key"]: json.loads(l)["id"]
                 for l in open(snap / "records.jsonl")}
    bad = [k for k in train_keys + val_keys if key_to_id[k] in held]
    if bad:
        Q.die(f"G4 held-out contamination: {len(bad)} keys e.g. {bad[:3]}")
    # G4b: the eval-only held-out CoT labels (R1 / oracle_text) must never be
    # visible to training.  Reads the directory; records the key count so a zero
    # intersection cannot be a silent consequence of reading nothing.
    g4b = guards.assert_no_eval_only_contamination(train_keys, val_keys, key_to_id)

    # ---- model: base + stage-1 LoRA, continued ------------------------------
    proc, n_before, n_new = Q.load_processor(cfg["model"]["model_max_length"])
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    model_base, emb_facts = Q.load_model(
        proc, dtype=dtype, device=device,
        attn_implementation=cfg["model"]["attn_implementation"],
        grad_checkpointing=bool(cfg["model"].get("grad_checkpointing", True)),
        weights_dir=str(Path(cfg["data"]["base_weights_dir"]) / "model"))
    base_dir = cfg["data"].get("base_weights_dir", "")
    if not base_dir:
        Q.die("[data] base_weights_dir is required for S2F-B (full-FT base)")
    from peft import PeftModel  # noqa: F401  (kept for parity; unused on this path)
    # Fresh LoRA on top of the FULL fine-tuned base.  The base was already loaded
    # from base_weights_dir by load_model(weights_dir=...), so the 6 stage-token
    # rows are the trained ones -- new_token_embeddings.pt is NOT read here.
    model, lora_facts = Q.attach_lora(model_base, r=cfg["lora"]["r"],
                                      alpha=cfg["lora"]["alpha"],
                                      dropout=cfg["lora"]["dropout"],
                                      last_n=cfg["lora"]["last_n_blocks"])
    emb_w, _mask = Q.enable_new_token_rows(model, emb_facts)
    print(f"[adapt] S2F-B base={base_dir} lora={json.dumps({k: v for k, v in lora_facts.items() if k != 'targets'})}",
          flush=True)
    lm = Q.resolve_language_model(model)
    mm = Q.resolve_mm_model(model)
    freeze_lora = bool(cfg.get("train", {}).get("freeze_lora", False))
    if freeze_lora:
        for n_, p_ in model.named_parameters():
            p_.requires_grad_(False)
        print("[adapt] variant C: LoRA + embedding FROZEN, adapter only", flush=True)
    else:
        Q.assert_frozen_area(model, extra_trainable=("embed_tokens",))
    lm_head = model.get_output_embeddings()
    lora_params = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    n_lora = sum(p.numel() for p in lora_params)
    w_ce = float(cfg["loss"].get("w_ce", 0.0))

    adapter = Adapter(in_dim=Q.Z_DIM, hidden=cfg["model"]["adapter_hidden"]).to(device)
    n_adapter = sum(p.numel() for p in adapter.parameters())
    print(f"[adapt] lora_params={n_lora} adapter_params={n_adapter}", flush=True)

    imode = cfg["data"].get("instruction_mode", "fixed")
    isalt = cfg["data"].get("instruction_salt", C.INSTRUCTION_SALT)
    ds = CoTDataset(str(snap), train_keys, proc, stage_ids,
                    include_stage_token=cfg["readout"]["include_stage_token"],
                    max_len=cfg["model"]["model_max_length"],
                    targets=build_targets(train_keys, assets_index, names, lat),
                    instruction_mode=imode, instruction_salt=isalt)
    dsv = CoTDataset(str(snap), val_keys, proc, stage_ids,
                     include_stage_token=cfg["readout"]["include_stage_token"],
                     max_len=cfg["model"]["model_max_length"],
                     targets=build_targets(val_keys, assets_index, names, lat),
                     instruction_mode=imode, instruction_salt=isalt)
    print(f"[adapt] instruction_mode={imode}", flush=True)
    pad = proc.tokenizer.pad_token_id
    mb, accum = o["micro_batch"], o["grad_accum"]
    # Length-bucketed batches: stage-1 showed reserved climbing 49 -> 94 GiB purely
    # from padding-shape variance fragmenting the caching allocator (D-sft1).
    # Grouping similar lengths keeps each batch's padded shape stable.
    def bucketed_batches(dataset, batch_size, seed, bucket_mult=32):
        # Proxy for token length = characters of the serialised 6-segment CoT
        # target.  That is the part that actually varies (1,409-1,940 tokens
        # measured); the prompt is fixed and the spec-5 visual token count varies
        # only with aspect ratio.  PNG byte size would measure image complexity,
        # not sequence length, so it is deliberately NOT used.
        lens = [len(C.target_text(dataset.by_key[k])) for k in dataset.keys]
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(dataset.keys), generator=g).tolist()
        chunk = batch_size * bucket_mult
        batches = []
        for i in range(0, len(order), chunk):
            block = sorted(order[i:i + chunk], key=lambda j: lens[j])
            for j in range(0, len(block), batch_size):
                batches.append(block[j:j + batch_size])
        perm = torch.randperm(len(batches), generator=g).tolist()
        return [batches[i] for i in perm]

    def make_loader(epoch: int):
        return DataLoader(ds, num_workers=o["num_workers"],
                          batch_sampler=bucketed_batches(ds, mb, o["seed"] + epoch),
                          collate_fn=lambda b: collate(b, pad))

    dl = make_loader(0)
    dlv = DataLoader(dsv, batch_size=mb, shuffle=False, num_workers=o["num_workers"],
                     collate_fn=lambda b: collate(b, pad), drop_last=False)

    groups = [dict(params=list(adapter.parameters()), lr=o["adapter_lr"],
                   weight_decay=o["adapter_weight_decay"])]
    if lora_params:
        groups.insert(0, dict(params=lora_params, lr=o["lora_lr"],
                              weight_decay=o["weight_decay"]))
    opt = torch.optim.AdamW(groups, betas=tuple(o["betas"]), eps=1e-8)
    i_lora = 0 if lora_params else None
    i_adapt = 1 if lora_params else 0

    beta = cfg["loss"]["smooth_l1_beta"]
    if cfg["loss"]["w_pixel"] != 0.0:
        Q.die("[loss] w_pixel != 0: 端到端像素项预注册为 0,未实现")
    w_lat = cfg["loss"]["w_latent"]

    steps_per_epoch = math.ceil(len(dl) / accum)
    total = args.max_steps or steps_per_epoch * o["epochs"]

    def lr_scale(s):
        if o["warmup"] > 0 and s < o["warmup"]:
            return (s + 1) / o["warmup"]
        t = (s - o["warmup"]) / max(1, total - o["warmup"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    @torch.no_grad()
    def evaluate():
        model.eval(); adapter.eval()
        sl, sc, s2, n = 0.0, 0.0, 0.0, 0
        for b in dlv:
            z = readout_z(mm, b, device, dtype)
            a = adapter(z)
            t = b["target"].to(device)
            sl += float(nn.functional.smooth_l1_loss(a, t, beta=beta)) * len(b["keys"])
            sc += float(nn.functional.cosine_similarity(
                a.reshape(-1, 128), t.reshape(-1, 128)).mean()) * len(b["keys"])
            s2 += float((a - t).reshape(-1, 128).norm(dim=-1).mean()) * len(b["keys"])
            n += len(b["keys"])
        model.train(); adapter.train()
        return dict(val_loss=sl / n, val_latent_cos=sc / n, val_latent_l2=s2 / n, n_val=n)

    def save_snapshot(tag: str, step_i: int, epoch_i: int):
        """LoRA + adapter only -- a few MB, so this is free.  Written to a tmp dir
        and renamed, so a kill mid-write cannot leave a half-written checkpoint.

        Exists because stage-1 (SPRF_SFT_S1) had NO periodic save: its reserved
        memory climbed to 94.3 GiB on a 95 GiB card and an OOM in the tail would
        have thrown away ~4 h with nothing on disk (NOTES D-sft2).
        """
        d = out / f"ckpt_{tag}"
        tmp = out / f".tmp_ckpt_{tag}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(tmp / "lora"))
        torch.save(dict(state_dict=adapter.state_dict(), n_params=n_adapter,
                        in_dim=Q.Z_DIM, hidden=cfg["model"]["adapter_hidden"]),
                   tmp / "adapter.pt")
        (tmp / "meta.json").write_text(json.dumps(
            dict(step=step_i, epoch=epoch_i, tag=tag, n_seen=n_seen,
                 wall_s=time.time() - t0)))
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        print(f"  [ckpt] {tag} @ step {step_i} -> {d}", flush=True)

    log = open(out / "steps.jsonl", "a")
    model.train(); adapter.train()
    step, n_seen, acc = 0, 0, 0.0
    t0 = time.time()
    done = False
    for ep in range(10 ** 6 if args.max_steps else o["epochs"]):
        if done:
            break
        opt.zero_grad(set_to_none=True)
        dl_ep = dl if ep == 0 else make_loader(ep)
        for bi, b in enumerate(dl_ep):
            if w_ce > 0.0:
                z, hid = readout_z_and_hidden(mm, b, device, dtype)
            else:
                z, hid = readout_z(mm, b, device, dtype), None
            a = adapter(z)
            loss_lat = nn.functional.smooth_l1_loss(a, b["target"].to(device), beta=beta)
            loss_ce = (chunked_ce(hid, lm_head, b, device,
                                  int(cfg["loss"].get("ce_chunk", 512)))
                       if w_ce > 0.0 else torch.zeros((), device=device))
            loss = w_lat * loss_lat + w_ce * loss_ce
            (loss / accum).backward()          # per-micro-batch, graph freed here
            if step == 0 and bi == 0:
                # A12 (span-pool gradient reaches the adapters).  EPR-033's A-9
                # trap is forward-hook-specific; the checkpointed block's RETURN
                # value is in the graph.  That is an argument, not evidence, so
                # it is measured here on the very first backward.
                g_lora = sum(float(p_.grad.abs().sum()) for p_ in lora_params
                             if p_.grad is not None)
                n_lora_with_grad = sum(1 for p_ in lora_params if p_.grad is not None)
                g_adapt = sum(float(p_.grad.abs().sum())
                              for p_ in adapter.parameters() if p_.grad is not None)
                a12 = dict(guard="A12", grad_ckpt=bool(cfg["model"].get(
                    "grad_checkpointing", True)),
                    lora_grad_abs_sum=g_lora, lora_params_with_grad=n_lora_with_grad,
                    n_lora_tensors=len(lora_params), adapter_grad_abs_sum=g_adapt)
                print("[A12] " + json.dumps(a12), flush=True)
                (out / "a12_grad_check.json").write_text(json.dumps(a12, indent=1))
                if lora_params and not (g_lora > 0.0):
                    Q.die("A12 FAILED: span-pool gradient does NOT reach the LoRA "
                          "adapters (lora_grad_abs_sum == 0). Gradient checkpointing "
                          "is severing the path -- set model.grad_checkpointing=false "
                          "in the config and re-run, and report the fallback.")
                if not (g_adapt > 0.0):
                    Q.die("A12 FAILED: adapter received no gradient")
            acc += float(loss)
            n_seen += len(b["keys"])
            if (bi + 1) % accum == 0 or (bi + 1) == len(dl_ep):
                nb = accum if (bi + 1) % accum == 0 else (bi + 1) % accum
                sc_ = lr_scale(step)
                if i_lora is not None:
                    opt.param_groups[i_lora]["lr"] = o["lora_lr"] * sc_
                opt.param_groups[i_adapt]["lr"] = o["adapter_lr"] * sc_
                gn = torch.nn.utils.clip_grad_norm_(
                    lora_params + list(adapter.parameters()), o["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                rec = dict(step=step, epoch=ep, loss=acc / nb,
                           loss_latent=float(loss_lat.detach()),
                           loss_ce=float(loss_ce.detach()), grad_norm=float(gn),
                           lr_lora=o["lora_lr"] * sc_, lr_adapter=o["adapter_lr"] * sc_,
                           n_seen=n_seen,
                           alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                           reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                           elapsed_s=time.time() - t0)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % o["print_every"] == 0:
                    print(f"  step {step}/{total} loss={rec['loss']:.6f} "
                          f"gn={rec['grad_norm']:.3f} alloc={rec['alloc_gb']:.2f}GB "
                          f"resv={rec['reserved_gb']:.2f}GB {rec['elapsed_s']:.0f}s",
                          flush=True)
                if o["eval_every"] and step % o["eval_every"] == 0 and step:
                    ev = evaluate()
                    log.write(json.dumps(dict(step=step, **ev)) + "\n"); log.flush()
                    print(f"  [val] step {step} " + json.dumps(ev), flush=True)
                if o.get("save_every", 0) and step and step % o["save_every"] == 0:
                    save_snapshot("latest", step, ep)
                acc = 0.0
                step += 1
                if step >= total:
                    done = True
                    break
        # end of epoch: LoRA+adapter are a few MB, so snapshot unconditionally
        save_snapshot(f"epoch{ep}", step, ep)

    if args.memprobe:
        rows = [json.loads(l) for l in open(out / "steps.jsonl")]
        tail = [r for r in rows if r.get("step", 0) >= max(0, step - 20) and "reserved_gb" in r]
        plateau = max(r["reserved_gb"] for r in tail) if tail else 0.0
        probe = dict(memprobe=True, steps=step,
                     peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                     peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                     reserved_plateau_last20=plateau,
                     s_per_step=(time.time() - t0) / max(1, step),
                     micro_batch=mb, grad_accum=accum,
                     alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
                     declare_mem_peak=int(plateau) + 2,
                     n_lora=n_lora, n_adapter=n_adapter)
        (out / "memprobe.json").write_text(json.dumps(probe, indent=1))
        print("ADAPT_MEMPROBE " + json.dumps(probe), flush=True)
        return

    final = evaluate()
    model.save_pretrained(str(out / "lora"))
    torch.save(dict(state_dict=adapter.state_dict(), n_params=n_adapter,
                    in_dim=Q.Z_DIM, hidden=cfg["model"]["adapter_hidden"]),
               out / "adapter.pt")
    prov = dict(
        arm=cfg["arm"]["name"], stage="adapt",
        base=dict(dir=Q.MODEL_DIR, repo=Q.MODEL_REPO, revision=Q.MODEL_REVISION,
                  attn_implementation=cfg["model"]["attn_implementation"],
                  dtype=cfg["model"]["dtype"], grad_checkpointing=False),
        base_weights_dir=str(cfg["data"]["base_weights_dir"]),
        instruction=dict(mode=cfg["data"].get("instruction_mode","fixed"),
                         salt=cfg["data"].get("instruction_salt","")),
        data=dict(snapshot=freeze, n_train=len(train_keys), n_val=len(val_keys),
                  g4b=g4b, spec5_rejects=reject_report),
        readout=dict(cfg["readout"], hidden_layer=-1, final_norm=True,
                     kind="segment_span_pool", z_dim=Q.Z_DIM),
        trainable=dict(n_lora=n_lora, n_adapter=n_adapter,
                       joint="LoRA and adapter trained together (stage-2 main row)"),
        loss=dict(kind="SmoothL1 on 128-d latent", beta=beta,
                  w_latent=w_lat, w_pixel=cfg["loss"]["w_pixel"]),
        optim=dict(o, total_steps=total, steps_per_epoch=steps_per_epoch),
        target=dict(cfg["target"], target_latents_sha256=tl.get("tensor_sha256"),
                    backbone_ckpt=tl.get("backbone_ckpt"),
                    backbone_ckpt_sha256=tl.get("backbone_ckpt_sha256")),
        index_mapping="readout slot m (1..6, restoration) -> chain k = 6 - m",
        memory_strategy="per-micro-batch backward (graph freed); EPR-033 retains graphs",
        template_sha256=C.template_sha256(),
        source_sha256={f: sha256_file(_P.src(f)) for f in
                       ["train_q3vl_adapt.py", "q3vl_common.py", "q3vl_text.py",
                        "q3vl_data.py", "cot_text.py"]},
        config_sha256=sha256_file(args.config), final=final,
        wall_s=time.time() - t0)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    summary = dict(steps=step, n_seen=n_seen, **final,
                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                   wall_s=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("ADAPT_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
