# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/dump_readout.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / SFT+ADAPT -- produce per-stage LUT latents from the stage-2 model.

Two contracts, one model, one adapter, one readout rule.  They differ ONLY in
where the 6 segment spans come from:

  oracle_text     teacher forcing.  The sample's GT CoT text is placed in the
                  assistant turn and the spans are built BY CONSTRUCTION
                  (q3vl_text.build_example), so they are exact.
  predicted_text  the model sees only y + the frozen instruction and decodes
                  greedily (do_sample=False, temperature unset -- M1 前科:
                  temp>0 is not reproducible).  The spans are then located in
                  the GENERATED id sequence by q3vl_text.spans_from_generated.

Failure accounting (never absorbed): a generated sequence missing <vr_stage_m>,
or yielding an empty span, is counted in `stage_token_missing` and the sample is
marked ok=false.  Its latent row is written as the adapter's output on a zero
pooled vector so the array stays dense, and downstream eval must read `ok`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from veraretouch_sprf.data import cot_text as C  # noqa: E402
from veraretouch_sprf.models.vlm import q3vl_common as Q  # noqa: E402
from veraretouch_sprf.data import q3vl_text as T  # noqa: E402
from veraretouch_sprf.models.vlm.adapter import Adapter  # noqa: E402


def load_full_base_single_lora(run: Path, base_dir: str, device, dtype, attn):
    """Full fine-tuned base + ONE LoRA (registered under both names so the
    gen/readout adapter switch is a no-op).  The stage-token rows live in the
    base weights, so new_token_embeddings.pt is not read."""
    proc, n_before, n_new = load_processor_checked()
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    base, emb_facts = Q.load_model(proc, dtype=dtype, device=device,
                                   attn_implementation=attn,
                                   weights_dir=str(Path(base_dir) / "model"))
    base.config.use_cache = True
    if hasattr(base.config, "text_config"):
        base.config.text_config.use_cache = True
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(run / "lora"),
                                      adapter_name="s2", is_trainable=False)
    model.load_adapter(str(run / "lora"), adapter_name="s1")
    model.eval()
    ab = torch.load(run / "adapter.pt", map_location="cpu")
    adapter = Adapter(in_dim=ab["in_dim"], hidden=ab["hidden"]).to(device)
    adapter.load_state_dict(ab["state_dict"]); adapter.eval()
    print(f"[dump] full-FT base {base_dir} + single LoRA from {run}", flush=True)
    return proc, model, Q.resolve_language_model(model), adapter, stage_ids, emb_facts


def load_two_adapter(run: Path, sft_run: Path, device, dtype, attn):
    """One base + TWO LoRA adapters: "s1" (stage-1, generation) and "s2"
    (stage-2, readout).  Not merged -- merge_and_unload cannot hold two.

    Contract `predicted_text_s1gen`: stage-2's objective carries no LM term and
    its LoRA no longer generates (measured: 0/6 stage tokens, degenerate repeat),
    while stage-1's does (6/6 teacher-forced rank 1, correct format).  So CoT is
    GENERATED under s1 and READ OUT under s2.  Generation weights != readout
    weights; every row produced this way must say so.
    """
    proc, n_before, n_new = load_processor_checked()
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    base, emb_facts = Q.load_model(proc, dtype=dtype, device=device,
                                   attn_implementation=attn)
    blob = torch.load(sft_run / "new_token_embeddings.pt", map_location="cpu")
    if blob["stage_ids"] != stage_ids:
        Q.die(f"stage ids {stage_ids} != stage-1 {blob['stage_ids']}")
    embw = base.get_input_embeddings().weight
    embw.data[emb_facts["new_ids"]] = blob["new_token_rows"].to(embw.device, embw.dtype)
    # generation needs the KV cache; load_model disables it for training
    base.config.use_cache = True
    if hasattr(base.config, "text_config"):
        base.config.text_config.use_cache = True
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(sft_run / "lora"),
                                      adapter_name="s1", is_trainable=False)
    model.load_adapter(str(run / "lora"), adapter_name="s2")
    model.eval()
    ab = torch.load(run / "adapter.pt", map_location="cpu")
    adapter = Adapter(in_dim=ab["in_dim"], hidden=ab["hidden"]).to(device)
    adapter.load_state_dict(ab["state_dict"]); adapter.eval()
    return proc, model, Q.resolve_language_model(model), adapter, stage_ids, emb_facts


def load_processor_checked():
    proc, n_before, n_new = Q.load_processor(4096)
    if n_new != 6:
        Q.die(f"tokenizer added {n_new} tokens, expected 6")
    return proc, n_before, n_new


@torch.no_grad()
def generate_batch_s1(model, proc, imgs, device, dtype, max_new, pad_id, eos_id,
                      gen_adapter="s1", instrs=None):
    """Left-padded batched greedy generation under adapter s1.

    Returns per-sample generated id lists, each truncated at its first EOS
    (inclusive) so a row's content matches what batch-1 would have returned in
    length semantics.  NOTE: batch>1 is NOT bit-identical to batch-1 on this
    model -- measured, 4/4 samples diverge (bf16 reduction order changes with the
    batch dimension; SPRF_DIAG_BATCH).  Greedy and replayable given a fixed batch
    composition, which the caller records in a manifest.
    """
    model.set_adapter(gen_adapter)
    ins = instrs if instrs is not None else [C.INSTRUCTION] * len(imgs)
    encs = [T.encode_prompt(proc, im, i) for im, i in zip(imgs, ins)]
    lens = [e["input_ids"].shape[1] for e in encs]
    P = max(lens)
    B = len(encs)
    ii = torch.full((B, P), pad_id, dtype=torch.long)
    am = torch.zeros((B, P), dtype=torch.long)
    for i, e in enumerate(encs):
        L = lens[i]
        ii[i, P - L:] = e["input_ids"][0]
        am[i, P - L:] = 1
    pv = torch.cat([e["pixel_values"] for e in encs], 0)
    gr = torch.cat([e["image_grid_thw"] for e in encs], 0)
    out = model.generate(input_ids=ii.to(device), attention_mask=am.to(device),
                         pixel_values=pv.to(device, dtype),
                         image_grid_thw=gr.to(device),
                         do_sample=False, num_beams=1, max_new_tokens=max_new,
                         use_cache=True, return_dict_in_generate=True,
                         pad_token_id=pad_id)
    gens = []
    for i in range(B):
        g = out.sequences[i][P:].tolist()
        if eos_id in g:
            g = g[: g.index(eos_id) + 1]
        gens.append(g)
    return encs, gens


@torch.no_grad()
def readout_from_generated(model, proc, enc, gen, device, dtype, include_tok,
                           stage_ids):
    """Per-sample readout under adapter s2 over prompt+generated."""
    model.set_adapter("s2")
    spans_rel, missing = T.spans_from_generated(proc.tokenizer, gen, stage_ids,
                                                include_stage_token=include_tok)
    ids = enc["input_ids"].to(device)
    full = torch.cat([ids[0], torch.tensor(gen, device=device)]).unsqueeze(0)
    mm = Q.resolve_mm_model(model)
    o = mm(input_ids=full, attention_mask=torch.ones_like(full),
           pixel_values=enc["pixel_values"].to(device, dtype),
           image_grid_thw=enc["image_grid_thw"].to(device), return_dict=True)
    h = o.last_hidden_state.float()[0]
    n_p = ids.shape[1]
    spans_abs = [None if sp is None else (n_p + sp[0], n_p + sp[1]) for sp in spans_rel]
    return T.span_pool(h, spans_abs), missing


@torch.no_grad()
def pooled_predicted_text_s1gen(model, proc, img, device, dtype, include_tok,
                                stage_ids, max_new, instr=None):
    """generate under adapter s1, read out under adapter s2."""
    enc = T.encode_prompt(proc, img, instr if instr is not None else C.INSTRUCTION)
    ids = enc["input_ids"].to(device)
    am = torch.ones_like(ids)
    model.set_adapter("s1")
    out = model.generate(
        input_ids=ids, attention_mask=am,
        pixel_values=enc["pixel_values"].to(device, dtype),
        image_grid_thw=enc["image_grid_thw"].to(device),
        do_sample=False, num_beams=1, max_new_tokens=max_new, use_cache=True,
        return_dict_in_generate=True, pad_token_id=proc.tokenizer.pad_token_id)
    gen = out.sequences[0][ids.shape[1]:].tolist()
    eos = proc.tokenizer.eos_token_id
    hit_eos = eos in gen
    spans_rel, missing = T.spans_from_generated(proc.tokenizer, gen, stage_ids,
                                                include_stage_token=include_tok)
    model.set_adapter("s2")
    full = torch.cat([ids[0], torch.tensor(gen, device=device)]).unsqueeze(0)
    mm = Q.resolve_mm_model(model)
    o = mm(input_ids=full, attention_mask=torch.ones_like(full),
           pixel_values=enc["pixel_values"].to(device, dtype),
           image_grid_thw=enc["image_grid_thw"].to(device), return_dict=True)
    h = o.last_hidden_state.float()[0]
    n_p = ids.shape[1]
    spans_abs = [None if sp is None else (n_p + sp[0], n_p + sp[1]) for sp in spans_rel]
    text = proc.tokenizer.decode(gen, skip_special_tokens=False)
    return T.span_pool(h, spans_abs), missing, text, hit_eos, len(gen)


def load_stage2(run: Path, sft_run: Path, device, dtype, attn):
    proc, n_before, n_new = Q.load_processor(4096)
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    base, emb_facts = Q.load_model(proc, dtype=dtype, device=device,
                                   attn_implementation=attn)
    blob = torch.load(sft_run / "new_token_embeddings.pt", map_location="cpu")
    if blob["stage_ids"] != stage_ids:
        Q.die(f"stage ids {stage_ids} != stage-1 {blob['stage_ids']}")
    embw = base.get_input_embeddings().weight
    embw.data[emb_facts["new_ids"]] = blob["new_token_rows"].to(embw.device, embw.dtype)
    from peft import PeftModel
    model = PeftModel.from_pretrained(base, str(run / "lora"), is_trainable=False)
    model = model.merge_and_unload()
    model.eval()
    ab = torch.load(run / "adapter.pt", map_location="cpu")
    adapter = Adapter(in_dim=ab["in_dim"], hidden=ab["hidden"]).to(device)
    adapter.load_state_dict(ab["state_dict"])
    adapter.eval()
    lm = Q.resolve_language_model(model)
    return proc, model, lm, adapter, stage_ids, emb_facts


@torch.no_grad()
def pooled_oracle_text(model, lm, proc, img, record, device, dtype, include_tok,
                       instr=None):
    ex = T.build_example(proc, img, instr if instr is not None else C.INSTRUCTION,
                         record,
                         include_stage_token=include_tok)
    ids = ex["input_ids"].unsqueeze(0).to(device)
    am = torch.ones_like(ids)
    with Q.LastLayerHook(lm, -1) as hook:
        model(input_ids=ids, attention_mask=am,
              pixel_values=ex["pixel_values"].to(device, dtype),
              image_grid_thw=ex["image_grid_thw"].to(device),
              logits_to_keep=1, return_dict=True)
        h = hook.captured
    h = lm.norm(h).float()[0]
    return T.span_pool(h, ex["spans"]), [], None


@torch.no_grad()
def pooled_predicted_text(model, lm, proc, img, device, dtype, include_tok,
                          stage_ids, max_new, instr=None):
    # TRAP: this used to hardcode C.INSTRUCTION, so --instruction-mode per_sample
    # was silently ignored for contract `predicted_text` -- the model was trained
    # on per-sample editing instructions but generated under the fixed one, and
    # the dump JSON still recorded instruction_mode=per_sample.
    enc = T.encode_prompt(proc, img, instr if instr is not None else C.INSTRUCTION)
    ids = enc["input_ids"].to(device)
    am = torch.ones_like(ids)
    out = model.generate(
        input_ids=ids, attention_mask=am,
        pixel_values=enc["pixel_values"].to(device, dtype),
        image_grid_thw=enc["image_grid_thw"].to(device),
        do_sample=False, num_beams=1, max_new_tokens=max_new,
        return_dict_in_generate=True,
        pad_token_id=proc.tokenizer.pad_token_id)
    seq = out.sequences[0]
    n_p = ids.shape[1]
    gen = seq[n_p:].tolist()
    spans_rel, missing = T.spans_from_generated(proc.tokenizer, gen, stage_ids,
                                                include_stage_token=include_tok)
    # one clean teacher-style forward over prompt+generated to get hiddens with
    # the same arithmetic as the oracle_text path (right padding, no cache)
    full = torch.cat([ids[0], torch.tensor(gen, device=device)]).unsqueeze(0)
    am2 = torch.ones_like(full)
    with Q.LastLayerHook(lm, -1) as hook:
        model(input_ids=full, attention_mask=am2,
              pixel_values=enc["pixel_values"].to(device, dtype),
              image_grid_thw=enc["image_grid_thw"].to(device),
              logits_to_keep=1, return_dict=True)
        h = hook.captured
    h = lm.norm(h).float()[0]
    spans_abs = [None if s is None else (n_p + s[0], n_p + s[1]) for s in spans_rel]
    text = proc.tokenizer.decode(gen, skip_special_tokens=False)
    return T.span_pool(h, spans_abs), missing, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapt-run", required=True)
    ap.add_argument("--sft-run", required=True)
    ap.add_argument("--contract", required=True,
                    choices=["oracle_text", "predicted_text", "predicted_text_s1gen"])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--records", required=True,
                    help="jsonl providing GT CoT (oracle_text only)")
    ap.add_argument("--assets-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-new-tokens", type=int, default=1809,
                    help="training CoT target length p99 (measured 1809; max 1916)")
    ap.add_argument("--include-stage-token", action="store_true")
    ap.add_argument("--cache-every", type=int, default=10,
                    help="flush the generation cache every N batches")
    ap.add_argument("--resume-cache", action="store_true",
                    help="reuse keys already present in the output gencache")
    ap.add_argument("--instruction-mode", default="fixed",
                    choices=["fixed", "per_sample"],
                    help="per_sample = the sample's own editing instruction "
                         "(sha1 70/20/10 tiers, same rule as training)")
    ap.add_argument("--base-weights-dir", default="",
                    help="full-FT base (…/ckpt_epochN/model). When given, the "
                         "adapt-run LoRA is attached as the SINGLE adapter 's2' "
                         "and generation+readout both use it.")
    ap.add_argument("--gen-adapter", default="s1", choices=["s1", "s2"],
                    help="which LoRA generates: s1 = stage-1 (contract "
                         "predicted_text_s1gen); s2 = this run's own weights")
    ap.add_argument("--gen-cache", default="",
                    help="reuse generated ids from a gencache_*.pt instead of "
                         "generating (readout still uses THIS run's adapters)")
    ap.add_argument("--gen-cache-assert-n", type=int, default=3,
                    help="regenerate this many keys and assert ids match the cache")
    ap.add_argument("--gen-batch", type=int, default=1,
                    help="batched left-padded greedy generation (readout stays "
                         "per-sample). >1 is NOT bit-identical to 1.")
    ap.add_argument("--attn", default="eager")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--target-latents", default="",
                    help="if given, also report latent cos/L2 against these targets")
    ap.add_argument("--inv-index", default="/home/bc/data/runs/epr051_sprf/lut_inv_cache/g17/index.json")
    args = ap.parse_args()

    dtype = torch.bfloat16
    run, sft = Path(args.adapt_run), Path(args.sft_run)
    if args.base_weights_dir:
        proc, model, lm, adapter, stage_ids, emb_facts = load_full_base_single_lora(
            run, args.base_weights_dir, args.device, dtype, args.attn)
    elif args.contract == "predicted_text_s1gen":
        proc, model, lm, adapter, stage_ids, emb_facts = load_two_adapter(
            run, sft, args.device, dtype, args.attn)
    else:
        proc, model, lm, adapter, stage_ids, emb_facts = load_stage2(
            run, sft, args.device, dtype, args.attn)

    assets = json.loads(Path(args.assets_index).read_text())["index"]
    keys = json.loads(Path(args.keys).read_text())
    if args.limit:
        keys = keys[: args.limit]
    n_eos = 0
    gen_lens = []
    recs_all = {}

    def _instr_for(k_):
        """The instruction this sample was TRAINED with; fixed mode keeps C.INSTRUCTION."""
        if args.instruction_mode != "per_sample":
            return C.INSTRUCTION
        r_ = recs_all.get(k_)
        if r_ is None:
            Q.die(f"instruction_mode=per_sample but {k_} has no record")
        return C.instruction_for(r_, k_)[0]

    if args.instruction_mode == "per_sample":
        _w = set(keys)
        for line in open(args.records):
            r = json.loads(line)
            if r.get("key") in _w:
                recs_all[r["key"]] = r
        miss = [k for k in keys if k not in recs_all]
        if miss:
            Q.die(f"per_sample instruction: {len(miss)} keys lack a record, e.g. {miss[:3]}")
    recs = {}
    if args.contract == "oracle_text":
        want = set(keys)
        for line in open(args.records):
            r = json.loads(line)
            if r.get("key") in want:
                recs[r["key"]] = r
        missing_rec = [k for k in keys if k not in recs]
        if missing_rec:
            Q.die(f"oracle_text: {len(missing_rec)} keys have no GT CoT, "
                  f"e.g. {missing_rec[:3]}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    N = len(keys)
    lat = torch.zeros(N, 6, 128)
    ok = torch.zeros(N, dtype=torch.bool)
    n_missing = 0
    n_parse_ok = 0
    samples = {}
    t0 = time.time()
    print(f"[dump:START] contract={args.contract} tag={args.tag} n={N} "
          f"gen_batch={args.gen_batch} max_new={args.max_new_tokens}", flush=True)
    batch_manifest = []
    gen_cache = {}          # key -> {"ids": [...], "text": str}
    if args.contract == "predicted_text_s1gen" and args.gen_cache:
        # Reuse generations produced by an earlier run whose GENERATION weights
        # are identical to this run's (variant C freezes LoRA at stage-1, so its
        # adapter "s1" is byte-identical to the cache producer's).  Readout still
        # runs under THIS run's s2 adapter.  Before trusting the cache, a few keys
        # are regenerated here and their ids compared token-by-token.
        blobc = torch.load(args.gen_cache, map_location="cpu")
        cache = blobc["gen_cache"]
        miss = [k for k in keys if k not in cache]
        if miss:
            Q.die(f"gen cache missing {len(miss)} keys, e.g. {miss[:3]}")
        eos_id = proc.tokenizer.eos_token_id
        nchk = min(int(args.gen_cache_assert_n), len(keys))
        for kk in keys[:nchk]:
            im = Q.prepare_image_spec5(Image.open(assets[kk]["png"]))[0]
            _e, _g = generate_batch_s1(model, proc, [im], args.device, dtype,
                                       int(blobc["max_new_tokens"]),
                                       proc.tokenizer.pad_token_id, eos_id,
                                       gen_adapter=args.gen_adapter,
                                       instrs=[_instr_for(kk)])
            if [int(t) for t in _g[0]] != list(cache[kk]["ids"]):
                Q.die(f"gen-cache MISMATCH on {kk}: regenerated ids differ from "
                      "the cache token-by-token; generation weights are not identical")
        print(f"[dump] gen-cache id parity PASS on {nchk} keys "
              f"({len(cache)} cached)", flush=True)
        for gi, kk in enumerate(keys):
            g = list(cache[kk]["ids"])
            enc = T.encode_prompt(proc, Q.prepare_image_spec5(
                Image.open(assets[kk]["png"]))[0], _instr_for(kk))
            z, miss_ = readout_from_generated(model, proc, enc, g, args.device, dtype,
                                              args.include_stage_token, stage_ids)
            lat[gi] = adapter(z.unsqueeze(0).to(args.device))[0].detach().cpu()
            ok[gi] = (len(miss_) == 0)
            n_missing += len(miss_)
            n_eos += int(eos_id in g)
            gen_lens.append(len(g))
            pp = C.parse_target_text(cache[kk]["text"])
            n_parse_ok += int(pp["ok"])
            if len(samples) < 200:
                samples[kk] = cache[kk]["text"]
            if (gi + 1) % 50 == 0 or gi + 1 == N:
                print(f"[dump:{args.contract}{args.tag}] {gi+1}/{N} "
                      f"ok={int(ok[:gi+1].sum())} missing_tok={n_missing} "
                      f"parse_ok={n_parse_ok} eos={n_eos} {time.time()-t0:.0f}s", flush=True)
        loop_keys = []
    elif (args.contract in ("predicted_text", "predicted_text_s1gen")
          and args.gen_batch > 1):
        # Incremental persistence: the cache used to be written only after the
        # whole loop, so killing the job discarded every generation (764 lost
        # 1,016 keys / ~4.3 h that way).  Flush every --cache-every batches and
        # allow resuming from a partial cache.
        cpath = out / f"gencache_{args.tag or args.contract}.pt"
        resume = {}
        if args.resume_cache and cpath.exists():
            try:
                resume = torch.load(cpath, map_location="cpu").get("gen_cache", {})
            except Exception as exc:
                print(f"[dump] partial cache unreadable ({exc}); starting fresh",
                      flush=True)
            print(f"[dump] resuming from partial cache: {len(resume)} keys", flush=True)
            if len(set(resume)) != len(resume):
                Q.die("partial cache has duplicate keys")
            gen_cache.update(resume)
        pad_id = proc.tokenizer.pad_token_id
        eos_id = proc.tokenizer.eos_token_id
        Bn = int(args.gen_batch)
        nflush = 0
        for b0 in range(0, N, Bn):
            bkeys = keys[b0: b0 + Bn]
            batch_manifest.append(dict(batch_index=len(batch_manifest),
                                       start=b0, keys=list(bkeys)))
            todo = [kk for kk in bkeys if kk not in gen_cache]
            if not todo:
                for j, kk in enumerate(bkeys):
                    gi = b0 + j
                    g = list(gen_cache[kk]["ids"])
                    enc = T.encode_prompt(proc, Q.prepare_image_spec5(
                        Image.open(assets[kk]["png"]))[0], _instr_for(kk))
                    z, mi = readout_from_generated(model, proc, enc, g, args.device,
                                                   dtype, args.include_stage_token,
                                                   stage_ids)
                    lat[gi] = adapter(z.unsqueeze(0).to(args.device))[0].detach().cpu()
                    ok[gi] = (len(mi) == 0); n_missing += len(mi)
                    n_eos += int(eos_id in g); gen_lens.append(len(g))
                    pp = C.parse_target_text(gen_cache[kk]["text"])
                    n_parse_ok += int(pp["ok"])
                continue
            imgs = [Q.prepare_image_spec5(Image.open(assets[kk]["png"]))[0]
                    for kk in bkeys]
            binstr = [_instr_for(kk) for kk in bkeys]
            encs, gens = generate_batch_s1(model, proc, imgs, args.device, dtype,
                                           args.max_new_tokens, pad_id, eos_id,
                                           gen_adapter=args.gen_adapter,
                                           instrs=binstr)
            for j, kk in enumerate(bkeys):
                gi = b0 + j
                z, miss = readout_from_generated(model, proc, encs[j], gens[j],
                                                 args.device, dtype,
                                                 args.include_stage_token, stage_ids)
                lat[gi] = adapter(z.unsqueeze(0).to(args.device))[0].detach().cpu()
                ok[gi] = (len(miss) == 0)
                n_missing += len(miss)
                n_eos += int(eos_id in gens[j])
                gen_lens.append(len(gens[j]))
                txt = proc.tokenizer.decode(gens[j], skip_special_tokens=False)
                gen_cache[kk] = dict(ids=[int(t) for t in gens[j]], text=txt)
                pp = C.parse_target_text(txt)
                n_parse_ok += int(pp["ok"])
                if len(samples) < 200:
                    samples[kk] = txt
            nflush += 1
            if args.cache_every and nflush % args.cache_every == 0:
                out.mkdir(parents=True, exist_ok=True)
                tmpc = cpath.with_suffix(".pt.tmp")
                torch.save(dict(gen_cache=gen_cache, gen_batch=Bn,
                                max_new_tokens=args.max_new_tokens,
                                sft_run=str(sft), contract=args.contract,
                                tag=args.tag, partial=True), tmpc)
                tmpc.replace(cpath)
            dn = b0 + len(bkeys)
            print(f"[dump:{args.contract}{args.tag}] {dn}/{N} "
                  f"ok={int(ok[:dn].sum())} missing_tok={n_missing} "
                  f"parse_ok={n_parse_ok} eos={n_eos} {time.time()-t0:.0f}s", flush=True)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"batch_manifest_{args.tag or args.contract}.json").write_text(
            json.dumps(dict(gen_batch=Bn, n=N, batches=batch_manifest), indent=1))
        torch.save(dict(gen_cache=gen_cache, gen_batch=Bn,
                        max_new_tokens=args.max_new_tokens, sft_run=str(sft),
                        contract=args.contract, tag=args.tag),
                   out / f"gencache_{args.tag or args.contract}.pt")
        print(f"[dump] gen cache written: {len(gen_cache)} keys", flush=True)
        loop_keys = []
    else:
        loop_keys = list(enumerate(keys))

    for i, k in loop_keys:
        img, geom = Q.prepare_image_spec5(Image.open(assets[k]["png"]))
        if args.contract == "oracle_text":
            z, miss, text = pooled_oracle_text(model, lm, proc, img, recs[k],
                                               args.device, dtype,
                                               args.include_stage_token,
                                               instr=_instr_for(k))
        elif args.contract == "predicted_text_s1gen":
            z, miss, text, hit_eos, glen = pooled_predicted_text_s1gen(
                model, proc, img, args.device, dtype, args.include_stage_token,
                stage_ids, args.max_new_tokens, instr=_instr_for(k))
            n_eos += int(hit_eos); gen_lens.append(glen)
            p = C.parse_target_text(text)
            n_parse_ok += int(p["ok"])
            if len(samples) < 200:
                samples[k] = text
        else:
            z, miss, text = pooled_predicted_text(model, lm, proc, img,
                                                  args.device, dtype,
                                                  args.include_stage_token,
                                                  stage_ids, args.max_new_tokens,
                                                  instr=_instr_for(k))
            p = C.parse_target_text(text)
            n_parse_ok += int(p["ok"])
            if len(samples) < 200:
                samples[k] = text
        with torch.no_grad():
            lat[i] = adapter(z.unsqueeze(0).to(args.device))[0].detach().cpu()
        ok[i] = (len(miss) == 0)
        n_missing += len(miss)
        if (i + 1) % 10 == 0 or i + 1 == N:
            print(f"[dump:{args.contract}{args.tag}] {i+1}/{N} "
                  f"ok={int(ok[:i+1].sum())} missing_tok={n_missing} "
                  f"parse_ok={n_parse_ok} {time.time()-t0:.0f}s", flush=True)

    # optional latent-alignment metrics (slot m -> chain k = 6 - m)
    lat_metrics = {}
    if args.target_latents:
        tl = torch.load(args.target_latents, map_location="cpu")
        if tl["indexed_by"] != "inv_table row / lut_id":
            Q.die(f"unexpected target indexing {tl['indexed_by']!r}")
        names = list(json.loads(Path(args.inv_index).read_text())["names"])
        row_of = {n: i for i, n in enumerate(names)}
        T_ = torch.zeros(N, 6, tl["target_latents"].shape[1])
        for i, k in enumerate(keys):
            chain = assets[k]["chain"]
            for m in range(1, 7):
                lut = chain[6 - m]["lut"]
                r_ = row_of.get(lut)
                if r_ is None:
                    Q.die(f"{k}: lut_id {lut} absent from the inverse-table cache")
                T_[i, m - 1] = tl["target_latents"][r_]
        okm = ok.clone()
        if int(okm.sum()) == 0:
            Q.die("no sample has a complete readout; refusing to report latent metrics")
        a_ok = lat[okm].reshape(-1, 128)
        t_ok = T_[okm].reshape(-1, 128)
        lat_metrics = dict(
            latent_cos_mean=float(torch.nn.functional.cosine_similarity(a_ok, t_ok).mean()),
            latent_cos_median=float(torch.nn.functional.cosine_similarity(a_ok, t_ok).median()),
            latent_l2_mean=float((a_ok - t_ok).norm(dim=-1).mean()),
            latent_l2_median=float((a_ok - t_ok).norm(dim=-1).median()),
            n_ok_samples=int(okm.sum()),
            target_latents=args.target_latents,
            target_tensor_sha256=tl.get("tensor_sha256"),
            backbone_ckpt=tl.get("backbone_ckpt"))
        print("LATENT_METRICS " + json.dumps(lat_metrics), flush=True)

    tag = args.tag or args.contract
    torch.save(dict(keys=keys, latents=lat, ok=ok, contract=args.contract,
                    tag=args.tag, adapt_run=str(run), sft_run=str(sft),
                    stage_ids=stage_ids),
               out / f"latents_{tag}.pt")
    summary = dict(contract=args.contract, tag=args.tag, n=N, n_ok=int(ok.sum()),
                   n_stage_token_missing=n_missing,
                   stage_token_missing_rate=n_missing / (6 * max(1, N)),
                   cot_parse_success_rate=(n_parse_ok / max(1, N)
                                           if args.contract.startswith("predicted_text") else None),
                   wall_s=time.time() - t0, adapt_run=str(run),
                   max_new_tokens=args.max_new_tokens, gen_batch=args.gen_batch,
                   gen_adapter=args.gen_adapter,
                   instruction_mode=args.instruction_mode,
                   eos_rate=(n_eos / max(1, N)) if gen_lens else None,
                   gen_len_mean=(sum(gen_lens) / len(gen_lens)) if gen_lens else None,
                   gen_len_max=(max(gen_lens) if gen_lens else None),
                   gen_hit_cap=(sum(1 for g in gen_lens if g >= args.max_new_tokens)
                                if gen_lens else None),
                   **lat_metrics)
    (out / f"dump_{tag}.json").write_text(json.dumps(summary, indent=1))
    if samples:
        (out / f"samples_{tag}.json").write_text(json.dumps(samples, indent=1,
                                                            ensure_ascii=False))
    print("DUMP_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
