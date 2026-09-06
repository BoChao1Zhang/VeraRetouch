# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/train_q3vl_sft3.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / stage-1 FULL fine-tune (S1F-FULL). 分支文件,LoRA 版一字不动(E1)。

相对 train_q3vl_sft2.py 的差异,只有三处:
  (1) 可训面:**语言主干全部参数**(含与 lm_head 绑定的 embedding,6 行阶段 token 行
      随之一起训);`model.visual`(patch_embed / blocks / merger / deepstack / pos_embed)
      **全部冻结**。A_freeze 换成 full 口径(assert_frozen_area_full)。
  (2) 优化器与精度(coordinator 裁决 b,标准混合精度):
      **fp32 主权重 + fp32 AdamW 状态,前后向 bf16(autocast),梯度 fp32 累积后 fp32 更新。**
      不用 GradScaler:bf16 的动态范围与 fp32 同阶,不需要 loss scaling。
      实测依据:torch.optim.AdamW 的 exp_avg/exp_avg_sq **按参数 dtype 分配**,
      所以「bf16 参数 + fp32 状态」在 torch 里得不到 —— bf16 参数只会得到 bf16 状态
      (实测 allocated 38.30 GiB,与 bf16 状态的算术一致,与 fp32 状态不一致)。
  (3) ckpt:全参 checkpoint 约 8 GiB,只保留最近 2 个 + 每 epoch 1 个。
G3(旧 embedding 行逐位不变)在全参微调下**按定义不适用**,已移除。

原 docstring:
EPR-051 / SFT+ADAPT stage 1 (RERUN VARIANT) -- LoRA CoT SFT of Qwen3-VL-4B.

分支文件(E1:原 train_q3vl_sft.py 已随 SPRF_SFT_S1 冻结,一个字节都不动)。
相对原版四项改动 + 周期保存,全部来自 2026-09-02 的卡0 险情裁决:
  (1) payload 开 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  (2) 开 HF 梯度检查点(use_reentrant=False)。stage-1 走 labels 前向、本来就不挂 hook,
      A-9 从来不适用于它;原版关掉是被 q3vl_common 的一刀切 die 连累的。
  (3) **分块交叉熵,且只算被监督位置**:绕开 lm_head 的整序列调用,先取
      Qwen3VLModel.last_hidden_state,只挑 labels != -100 的位置,再按 chunk 过
      lm_head + CE,每个 chunk 用 checkpoint 重算 logits ——(B,T,151936) 永不物化。
  (4) 按长度分桶采样,削掉变长 padding 的 shape 方差(D-sft1 的碎片化根因)。
  (+) 每 save_every 步与每个 epoch 存 LoRA + 新 token 行(D-sft2:原版全程零保存)。


实验登记(四项,其余不写):

1. 数据   frozen snapshot /home/bc/data/runs/epr051_vlmsft/snap_sft1
          n=19,074 d6 CoT records;train/val = 18,119/955(sha1 salt 切分)。
          输入 = y(退化图,spec-5)+ 冻结指令;目标 = 6 段 CoT,每段以
          <vr_stage_m> 收尾。held-out 4,560 零接触(断言 G4)。
2. 模型   Qwen3-VL-4B-Instruct(hidden 2560,36 层,vocab 151,936→151,942)。
          LoRA r=16 alpha=16 dropout=0,挂最后 8 个语言块的 q/k/v/o_proj
          (32 个全限定模块,2,621,440 参数)——EPR-033 主行同款。
          其余全冻;新增 6 行 embedding 可训(权重绑定 lm_head,不训就永远发不出
          这 6 个 token),旧 151,936 行梯度掩零。
3. 数学   L = (1/|S|) * sum_{t in S} -log p(x_t | x_<t, y)
          S = assistant 段的 token 位(prompt 与图像位 = IGNORE_INDEX)。
4. 优化器 AdamW,见 [optim];LoRA lr 与新 token lr 分组记录。

Guards(失败即 die,不吸收):
  A_lora   实际挂上的模块集合 == 声明集合
  A_freeze 除 LoRA 参数与 embed_tokens 外没有可训参数
  A_span   分段 token 化与整段 token 化逐 id 相等
  G_span   每段 span 的右边界紧邻它自己的 stage token
  G3       旧 151,936 行 embedding 训练前后逐比特不变
  G5       n_seen == n_expected(drop_last=False)
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
from torch.utils.data import DataLoader  # noqa: E402

from veraretouch_sprf.data import cot_text as C  # noqa: E402
from veraretouch_sprf.models.vlm import q3vl_common as Q  # noqa: E402
from veraretouch_sprf.data.q3vl_data import CoTDataset, collate  # noqa: E402
from veraretouch_sprf.eval import guards  # noqa: E402


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def chunked_ce(mm, lm_head, batch, device, dtype, chunk: int = 512):
    """CE over ONLY the supervised positions, lm_head applied in chunks.

    The plain `model(..., labels=...)` path materialises (B, T, 151936) logits and
    their fp32 upcast for the whole sequence, then throws away ~78% of it (the
    prompt + image span are IGNORE_INDEX).  Here:
      * Qwen3VLModel gives last_hidden_state (post final norm), lm_head unused;
      * causal shift, then select only positions whose target != IGNORE_INDEX;
      * lm_head + CE run per chunk under checkpoint(use_reentrant=False), so a
        chunk's logits are recomputed in backward instead of being stored.
    Returns (loss, n_supervised).
    """
    out = mm(input_ids=batch["input_ids"].to(device),
             attention_mask=batch["attention_mask"].to(device),
             pixel_values=batch["pixel_values"].to(device, dtype),
             image_grid_thw=batch["image_grid_thw"].to(device),
             return_dict=True)
    h = out.last_hidden_state[:, :-1, :]                     # predicts token t+1
    tgt = batch["labels"].to(device)[:, 1:]
    m = tgt != Q.IGNORE_INDEX
    hs = h[m]                                                # (N, 2560)
    ts = tgt[m]                                              # (N,)
    n = int(ts.numel())
    if n == 0:
        Q.die("no supervised positions in this batch")

    def _piece(hh, tt):
        return torch.nn.functional.cross_entropy(
            lm_head(hh).float(), tt, reduction="sum")

    total = hs.new_zeros(())
    for i in range(0, n, chunk):
        total = total + torch.utils.checkpoint.checkpoint(
            _piece, hs[i:i + chunk], ts[i:i + chunk], use_reentrant=False)
    return total / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=0)
    ap.add_argument("--resume-from", default="",
                    help="checkpoint dir (containing model/) to warm-start from")
    ap.add_argument("--resume-step", type=int, default=0,
                    help="global optimizer step already completed in that ckpt")
    ap.add_argument("--memprobe", action="store_true",
                    help="run a few steps, report allocated/reserved peak, exit")
    args = ap.parse_args()

    cfg = tomllib.load(open(args.config, "rb"))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    o = cfg["optim"]
    torch.manual_seed(o["seed"])
    device = cfg["run"]["device"]

    snap = Path(cfg["data"]["snapshot_dir"])
    freeze = json.loads((snap / "FREEZE.json").read_text())
    if freeze["records_sha256"] != sha256_file(snap / "records.jsonl"):
        Q.die("snapshot records.jsonl sha != FREEZE.json")
    train_keys = json.loads((snap / "split_train_keys.json").read_text())
    val_keys = json.loads((snap / "split_val_keys.json").read_text())
    # Keys whose image violates the frozen spec-5 contract (aspect > 4:1 is
    # REJECTED by q3vl.train.imageproc.plan_geometry, never squashed).  One such
    # key killed an 11 h run at step 4,210.  Excluded here explicitly and counted
    # -- never silently skipped inside the Dataset.
    rej_path = cfg["data"].get("reject_keys", "")
    rejected = sorted(json.loads(Path(rej_path).read_text())) if rej_path else []
    rej_set = set(rejected)
    n_tr0, n_va0 = len(train_keys), len(val_keys)
    if args.resume_from:
        # Resuming must reproduce the ORIGINAL epoch-0 ordering, so the key list is
        # left unfiltered here; reject keys are dropped at the BATCH level below,
        # which leaves every other sample's position untouched.
        val_keys = [k for k in val_keys if k not in rej_set]
    else:
        train_keys = [k for k in train_keys if k not in rej_set]
        val_keys = [k for k in val_keys if k not in rej_set]
    reject_report = dict(file=rej_path, n_listed=len(rejected),
                         n_removed_train=n_tr0 - len(train_keys),
                         n_removed_val=n_va0 - len(val_keys),
                         n_train=len(train_keys), n_val=len(val_keys),
                         keys=rejected)
    print(f"[sft] spec-5 reject list: {json.dumps(reject_report)}", flush=True)
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

    if cfg["data"].get("preflight_images", True):
        from PIL import Image as _I
        _ai = json.loads((snap / "assets_index.json").read_text())["index"]
        _bad = []
        for _k in [k for k in train_keys + val_keys if k not in rej_set]:
            with _I.open(_ai[_k]["png"]) as _im:
                _w, _h = _im.size
            if max(_w, _h) / min(_w, _h) > 4.0:
                _bad.append((_k, round(max(_w, _h)/min(_w, _h), 3)))
        if _bad:
            Q.die(f"preflight: {len(_bad)} key(s) violate the spec-5 aspect<=4 "
                  f"contract and would crash mid-run, e.g. {_bad[:5]}")
        print(f"[sft] preflight OK: {len(train_keys)+len(val_keys)} images pass "
              "spec-5 aspect<=4", flush=True)

    proc, n_before, n_new = Q.load_processor(cfg["model"]["model_max_length"])
    if n_new != 6:
        Q.die(f"tokenizer added {n_new} tokens, expected 6")
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    dtype = dict(bf16=torch.bfloat16, fp32=torch.float32)[cfg["model"]["dtype"]]
    model, emb_facts = Q.load_model(
        proc, dtype=dtype, device=device,
        attn_implementation=cfg["model"]["attn_implementation"],
        grad_checkpointing=bool(cfg["model"].get("grad_checkpointing", True)),
        weights_dir=(str(Path(args.resume_from) / "model") if args.resume_from else None))
    if args.resume_from:
        print(f"[sft] RESUMED weights from {args.resume_from} at global step "
              f"{args.resume_step}; AdamW state is FRESH (not checkpointed)",
              flush=True)
    full_facts = Q.unfreeze_full(model)
    n_train_params = Q.assert_frozen_area_full(model)
    lora_facts = dict(mode="full_finetune", **full_facts)
    emb_w = model.get_input_embeddings().weight
    mm = Q.resolve_mm_model(model)
    lm_head = model.get_output_embeddings()
    print(f"[sft] FULL fine-tune: trainable={n_train_params} "
          f"frozen_visual={full_facts['n_frozen_visual']}", flush=True)
    print(f"[sft] LoRA facts: {json.dumps({k: v for k, v in lora_facts.items() if k != 'targets'})}",
          flush=True)

    imode = cfg["data"].get("instruction_mode", "fixed")
    isalt = cfg["data"].get("instruction_salt", C.INSTRUCTION_SALT)
    ds = CoTDataset(str(snap), train_keys, proc, stage_ids,
                    include_stage_token=cfg["readout"]["include_stage_token"],
                    max_len=cfg["model"]["model_max_length"],
                    instruction_mode=imode, instruction_salt=isalt)
    dsv = CoTDataset(str(snap), val_keys, proc, stage_ids,
                     include_stage_token=cfg["readout"]["include_stage_token"],
                     max_len=cfg["model"]["model_max_length"],
                     instruction_mode=imode, instruction_salt=isalt)
    mix_tr = C.instruction_mix_counts([ds.by_key[k] for k in train_keys], isalt) \
        if imode == "per_sample" else dict(mode="fixed")
    mix_va = C.instruction_mix_counts([ds.by_key[k] for k in val_keys], isalt) \
        if imode == "per_sample" else dict(mode="fixed")
    print(f"[sft] instruction_mode={imode} train_mix={json.dumps(mix_tr)}", flush=True)
    pad = proc.tokenizer.pad_token_id
    mb = o["micro_batch"]
    accum = o["grad_accum"]
    def bucketed_batches(dataset, batch_size, seed, bucket_mult=32):
        """Group similar target lengths so each batch's padded shape is stable.
        D-sft1: padding-shape variance alone drove reserved 49 -> 94 GiB."""
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
        out_b = [batches[i] for i in perm]
        if rej_set:
            drop = {i for i, k in enumerate(dataset.keys) if k in rej_set}
            if drop:
                out_b = [[j for j in b if j not in drop] for b in out_b]
                out_b = [b for b in out_b if b]
        return out_b

    def make_loader(epoch: int, skip_batches: int = 0):
        bs = bucketed_batches(ds, mb, o["seed"] + epoch)
        n_all = len(bs)
        if skip_batches:
            # Slice the batch list, do NOT iterate-and-continue: iterating would
            # push every skipped micro-batch through the workers (image decode +
            # tokenise), which costs ~9 h for 67,200 batches.
            bs = bs[skip_batches:]
            print(f"[sft] epoch {epoch}: batch list sliced {n_all} -> {len(bs)} "
                  f"(skipped {skip_batches} already-done micro-batches)", flush=True)
        return DataLoader(ds, num_workers=o["num_workers"], batch_sampler=bs,
                          collate_fn=lambda b: collate(b, pad))

    dl = make_loader(0, skip_batches=(int(args.resume_step) * accum
                                      if args.resume_from else 0))
    dlv = DataLoader(dsv, batch_size=mb, shuffle=False, num_workers=o["num_workers"],
                     collate_fn=lambda b: collate(b, pad), drop_last=False)

    train_params = [p for p in model.parameters() if p.requires_grad]
    # full fine-tune: ONE group over every trainable parameter.
    opt = torch.optim.AdamW(train_params, lr=o["lr"],
                            betas=tuple(o["betas"]),
                            weight_decay=o["weight_decay"], eps=o["eps"])

    # steps_per_epoch MUST come from the FULL batch list, not the resume-sliced
    # loader: len(dl) is short after slicing, which would shrink `total` and drive
    # the cosine past its end -> lr exactly 0.0 (observed: step 4200/832, lr 0.0).
    _full_batches = len(bucketed_batches(ds, mb, o["seed"]))
    steps_per_epoch = math.ceil(_full_batches / accum)
    total = args.max_steps or steps_per_epoch * o["epochs"]
    if args.resume_from and not args.max_steps:
        if int(args.resume_step) >= total:
            Q.die(f"resume_step {args.resume_step} >= total {total}: the LR "
                  "schedule would already be finished (lr=0)")
        print(f"[sft] schedule: total={total} steps_per_epoch={steps_per_epoch} "
              f"resume_step={args.resume_step} lr_at_resume="
              f"{o['lr'] * (0.5 * (1 + math.cos(math.pi * min(1.0, (int(args.resume_step) - o['warmup']) / max(1, total - o['warmup']))))):.3e}",
              flush=True)

    def lr_scale(s):
        if o["warmup"] > 0 and s < o["warmup"]:
            return (s + 1) / o["warmup"]
        t = (s - o["warmup"]) / max(1, total - o["warmup"])
        return 0.5 * (1 + math.cos(math.pi * min(1.0, t)))

    def save_snapshot(tag: str, step_i: int, epoch_i: int):
        """LoRA + the 6 new embedding rows only (a few MB).  tmp dir then rename,
        so a kill mid-write leaves no half-written checkpoint.  D-sft2: the
        original stage-1 saved nothing until the very end."""
        d = out / f"ckpt_{tag}"
        tmp = out / f".tmp_ckpt_{tag}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(tmp / "model"), safe_serialization=True)
        proc.tokenizer.save_pretrained(str(tmp / "tokenizer"))
        torch.save(dict(emb_facts=emb_facts, stage_ids=stage_ids),
                   tmp / "stage_tokens.pt")
        (tmp / "meta.json").write_text(json.dumps(
            dict(step=step_i, epoch=epoch_i, tag=tag, n_seen=n_seen)))
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        print(f"  [ckpt] {tag} @ step {step_i} -> {d}", flush=True)
        # retention: full checkpoints are ~8 GiB.  Keep the most recent
        # `keep_last` step-checkpoints plus every epoch checkpoint.
        keep_last = int(o.get("keep_last_ckpt", 2))
        steps_ck = sorted([q for q in out.glob("ckpt_step*") if q.is_dir()],
                          key=lambda q: int(q.name.replace("ckpt_step", "")))
        for old in steps_ck[:-keep_last]:
            shutil.rmtree(old, ignore_errors=True)
            print(f"  [ckpt] pruned {old.name}", flush=True)

    log = open(out / "steps.jsonl", "a")
    model.train()
    step = int(args.resume_step)
    resume_skip = int(args.resume_step) * accum if args.resume_from else 0
    n_seen = int(args.resume_step) * accum
    t0 = time.time()
    acc = 0.0
    done = False
    for ep in range(10 ** 6 if args.max_steps else o["epochs"]):
        if done:
            break
        opt.zero_grad(set_to_none=True)
        dl_ep = dl if ep == 0 else make_loader(ep)
        for bi, b in enumerate(dl_ep):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _n_sup = chunked_ce(mm, lm_head, b, device, dtype,
                                          chunk=o.get("ce_chunk", 512))
            (loss / accum).backward()
            acc += float(loss.detach())
            n_seen += len(b["keys"])
            if (bi + 1) % accum == 0 or (bi + 1) == len(dl_ep):
                nb = accum if (bi + 1) % accum == 0 else (bi + 1) % accum
                sc = lr_scale(step)
                for g_ in opt.param_groups:
                    g_["lr"] = o["lr"] * sc
                gn = torch.nn.utils.clip_grad_norm_(train_params, o["grad_clip"])
                opt.step()
                opt.zero_grad(set_to_none=True)
                rec = dict(step=step, epoch=ep, loss=acc / nb,
                           lr=o["lr"] * sc, grad_norm=float(gn), n_seen=n_seen,
                           alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                           reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                           elapsed_s=time.time() - t0)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                if step % o["print_every"] == 0:
                    print(f"  step {step}/{total} loss={rec['loss']:.4f} "
                          f"gn={rec['grad_norm']:.3f} alloc={rec['alloc_gb']:.2f}GB "
                          f"resv={rec['reserved_gb']:.2f}GB "
                          f"{rec['elapsed_s']:.0f}s", flush=True)
                if o.get("save_every", 0) and step and step % o["save_every"] == 0:
                    save_snapshot(f"step{step}", step, ep)
                acc = 0.0
                step += 1
                if args.memprobe and step >= max(3, args.max_steps or 3):
                    done = True
                    break
                if step >= total:
                    done = True
                    break
        save_snapshot(f"epoch{ep}", step, ep)

    # G3 (old embedding rows bit-identical) does NOT apply to a full fine-tune:
    # every row is meant to move.  Recorded as not-applicable rather than skipped.
    drift = None

    if args.memprobe:
        summary = dict(memprobe=True, steps=step,
                       peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                       peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                       s_per_step=(time.time() - t0) / max(1, step),
                       micro_batch=mb, grad_accum=accum,
                       embedding_drift="n/a (full finetune)")
        (out / "memprobe.json").write_text(json.dumps(summary, indent=1))
        print("MEMPROBE " + json.dumps(summary), flush=True)
        return

    model.eval()
    vl, vn = 0.0, 0
    with torch.no_grad():
        for b in dlv:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                vloss, _ = chunked_ce(mm, lm_head, b, device, dtype,
                                      chunk=o.get("ce_chunk", 512))
            vl += float(vloss) * len(b["keys"])
            vn += len(b["keys"])

    model.save_pretrained(str(out / "lora"))
    proc.tokenizer.save_pretrained(str(out / "tokenizer"))
    torch.save(dict(new_token_rows=emb_w.data[emb_facts["new_ids"]].detach().cpu().clone(),
                    emb_facts=emb_facts, stage_ids=stage_ids),
               out / "new_token_embeddings.pt")

    prov = dict(
        arm=cfg["arm"]["name"], stage="sft",
        base=dict(dir=Q.MODEL_DIR, repo=Q.MODEL_REPO, revision=Q.MODEL_REVISION,
                  hidden=Q.Z_DIM, n_text_layers=Q.N_TEXT_LAYERS,
                  vocab_before=n_before, vocab_after=len(proc.tokenizer),
                  embedding=emb_facts,
                  attn_implementation=cfg["model"]["attn_implementation"],
                  dtype=cfg["model"]["dtype"], grad_checkpointing=False),
        data=dict(snapshot=freeze, n_train=len(train_keys), n_val=len(val_keys),
                  g4b=g4b, spec5_rejects=reject_report),
        lora=lora_facts,
        readout=cfg["readout"],
        instruction=dict(mode=imode, salt=isalt, train_mix=mix_tr, val_mix=mix_va),
        loss="mean token CE over assistant-turn positions (teacher forcing)",
        optim=dict(o, total_steps=total, steps_per_epoch=steps_per_epoch),
        template_sha256=C.template_sha256(),
        source_sha256={f: sha256_file(_P.src(f)) for f in
                       ["train_q3vl_sft.py", "q3vl_common.py", "q3vl_text.py",
                        "q3vl_data.py", "cot_text.py"]},
        config_sha256=sha256_file(args.config))
    (out / "provenance.json").write_text(json.dumps(prov, indent=1, ensure_ascii=False))
    summary = dict(steps=step, n_seen=n_seen, val_loss=vl / max(1, vn), n_val=vn,
                   embedding_drift_pre_existing_rows="n/a (full finetune)",
                   peak_alloc_gb=torch.cuda.max_memory_allocated() / 2 ** 30,
                   peak_reserved_gb=torch.cuda.max_memory_reserved() / 2 ** 30,
                   wall_s=time.time() - t0)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print("SFT_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
