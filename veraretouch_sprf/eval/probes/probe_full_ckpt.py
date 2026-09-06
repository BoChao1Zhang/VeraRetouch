# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/vlmsft/probe_full_ckpt.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 / S1F-FULL per-epoch probe: 32 val keys, greedy, field agreement.

Loads a FULL fine-tune checkpoint (a saved model dir, not LoRA adapters) and
measures whether the model actually learned the numeric content of the CoT:
per-segment mask-family hit, saturation sign agreement, |S| magnitude, and the
fraction of segments whose generated bands are all zero.  Magnitude collapse was
the failure of the LoRA line (|S| median GT 14.64 vs generated 0.000).
"""
from __future__ import annotations
import argparse, json, re, sys, time, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
import torch
from PIL import Image
from veraretouch_sprf.data import cot_text as C
from veraretouch_sprf.models.vlm import q3vl_common as Q
from veraretouch_sprf.data import q3vl_text as T

BW = r'light green|red|orange|yellow|green|aqua|blue|purple|magenta'
NUM = r'[+-]?\d+(?:\.\d+)?'
P_A = re.compile(rf'({BW})\s+hue\s*({NUM})\s*,\s*saturation\s*({NUM})\s*,\s*luminance\s*({NUM})', re.I)
P_B = re.compile(rf'({BW})\s+H\s*({NUM})\s*,\s*S\s*({NUM})\s*,\s*L\s*({NUM})', re.I)
P_SL = re.compile(rf'((?:{BW})(?:\s*/\s*(?:{BW}))+)\s*:?(.*)', re.I | re.S)
FAM = [('global', r'global|whole frame|no mask'), ('colour_range', r'colour[- ]range|color[- ]range|saturated pixels'),
       ('luminosity', r'luminosit|highlight|shadow|midtone|tonal'), ('subject', r'subject|person|skin|face|figure'),
       ('radial', r'radial|vignette'), ('gradient', r'gradient|linear mask')]


def slash_bands(t):
    m = P_SL.search(t or '')
    if not m: return {}
    names = [x.strip().lower() for x in re.split(r'\s*/\s*', m.group(1))]
    trip = {}
    for k, pat in (('h', r'hue\s*((?:' + NUM + r')(?:\s*/\s*' + NUM + r')*)'),
                   ('s', r'saturation\s*((?:' + NUM + r')(?:\s*/\s*' + NUM + r')*)'),
                   ('l', r'luminance\s*((?:' + NUM + r')(?:\s*/\s*' + NUM + r')*)')):
        mm = re.search(pat, m.group(2), re.I)
        if mm: trip[k] = [float(x) for x in re.split(r'\s*/\s*', mm.group(1))]
    if not trip: return {}
    n = min([len(names)] + [len(v) for v in trip.values()])
    return {names[i]: (trip.get('h', [0]*n)[i], trip.get('s', [0]*n)[i], trip.get('l', [0]*n)[i])
            for i in range(n)}


def bands(t):
    d = slash_bands(t)
    for p in (P_A, P_B):
        for m in p.finditer(t or ''):
            d[m.group(1).lower()] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return d


def fam(t):
    t = (t or '').lower()
    return {n for n, p in FAM if re.search(p, t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="dir containing model/ and tokenizer/")
    ap.add_argument("--keys", required=True)
    ap.add_argument("--records", required=True)
    ap.add_argument("--assets-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=1809)
    ap.add_argument("--instruction-mode", default="per_sample")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    dtype = torch.bfloat16
    ck = Path(args.ckpt)
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(Q.MODEL_DIR)
    tokdir = ck / "tokenizer"
    if tokdir.exists():
        from transformers import AutoTokenizer
        proc.tokenizer = AutoTokenizer.from_pretrained(str(tokdir), use_fast=True)
    else:
        proc.tokenizer.add_tokens(C.STAGE_TOKENS, special_tokens=True)
    stage_ids = Q.stage_token_ids(proc.tokenizer)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        str(ck / "model"), dtype=dtype, attn_implementation="sdpa").to(args.device)
    model.config.use_cache = True
    if hasattr(model.config, "text_config"):
        model.config.text_config.use_cache = True
    model.eval()
    print(f"[probe] LOADED FULL ckpt {ck} stage_ids={stage_ids}", flush=True)

    keys = json.loads(Path(args.keys).read_text())
    want = set(keys)
    recs = {}
    for l in open(args.records):
        r = json.loads(l)
        if r["key"] in want: recs[r["key"]] = r
    assets = json.loads(Path(args.assets_index).read_text())["index"]
    pad_id, eos_id = proc.tokenizer.pad_token_id, proc.tokenizer.eos_token_id

    @torch.no_grad()
    def gen(imgs, instrs):
        encs = [proc(text=[T.prompt_text(i)], images=[im], do_resize=False,
                     return_tensors="pt") for im, i in zip(imgs, instrs)]
        lens = [e["input_ids"].shape[1] for e in encs]; P = max(lens); B = len(encs)
        ii = torch.full((B, P), pad_id, dtype=torch.long); am = torch.zeros((B, P), dtype=torch.long)
        for i, e in enumerate(encs):
            ii[i, P-lens[i]:] = e["input_ids"][0]; am[i, P-lens[i]:] = 1
        o = model.generate(input_ids=ii.to(args.device), attention_mask=am.to(args.device),
                           pixel_values=torch.cat([e["pixel_values"] for e in encs], 0).to(args.device, dtype),
                           image_grid_thw=torch.cat([e["image_grid_thw"] for e in encs], 0).to(args.device),
                           do_sample=False, num_beams=1, max_new_tokens=args.max_new_tokens,
                           use_cache=True, return_dict_in_generate=True, pad_token_id=pad_id)
        out = []
        for i in range(B):
            g = o.sequences[i][P:].tolist()
            if eos_id in g: g = g[: g.index(eos_id)+1]
            out.append(g)
        return out

    per, texts = [], {}
    t0 = time.time()
    for b0 in range(0, len(keys), args.gen_batch):
        bk = keys[b0:b0+args.gen_batch]
        imgs = [Q.prepare_image_spec5(Image.open(assets[k]["png"]))[0] for k in bk]
        instrs = [C.instruction_for(recs[k], k)[0] if args.instruction_mode == "per_sample"
                  else C.INSTRUCTION for k in bk]
        gens = gen(imgs, instrs)
        for k, g in zip(bk, gens):
            txt = proc.tokenizer.decode(g, skip_special_tokens=False)
            texts[k] = txt
            pg = C.parse_target_text(txt)
            gt = C.parse_target_text('\n'.join(s+C.STAGE_TOKENS[i] for i, s in
                                               enumerate(C.target_segments(recs[k]))))
            for m in range(6):
                a, b = pg["steps"][m], gt["steps"][m]
                if not a or not b:
                    per.append(dict(key=k, m=m+1, parsed=False)); continue
                ba, bb = bands(a["adjustment"]), bands(b["adjustment"])
                com = sorted(set(ba) & set(bb))
                r = dict(key=k, m=m+1, parsed=True, n_gt=len(bb), n_gen=len(ba), n_com=len(com),
                         fam_hit=int(bool(fam(a["mask"]) & fam(b["mask"]))),
                         eos=int(eos_id in g), n_tok=len(g))
                if com:
                    r.update(signS=sum((ba[c][1] >= 0) == (bb[c][1] >= 0) for c in com)/len(com),
                             signH=sum((ba[c][0] >= 0) == (bb[c][0] >= 0) for c in com)/len(com),
                             absS_gt=st.median([abs(bb[c][1]) for c in com]),
                             absS_gen=st.median([abs(ba[c][1]) for c in com]),
                             gen_all_zero=int(all(abs(v[0]) < 1e-9 and abs(v[1]) < 1e-9
                                                  and abs(v[2]) < 1e-9 for v in ba.values())))
                per.append(r)
        print(f"[probe] {min(b0+args.gen_batch, len(keys))}/{len(keys)} {time.time()-t0:.0f}s", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = args.tag or ck.name
    def agg(f, sub=None):
        v = [r[f] for r in (sub or per) if f in r]
        return (sum(v)/len(v), st.median(v), len(v)) if v else (float('nan'), float('nan'), 0)
    ok = [r for r in per if r.get("parsed")]
    summary = dict(ckpt=str(ck), tag=tag, n_keys=len(keys), n_segments=len(per),
                   n_parsed=len(ok), eos_rate=agg("eos", ok)[0],
                   mask_family_hit=agg("fam_hit", ok)[0],
                   signS=agg("signS")[0], signH=agg("signH")[0],
                   absS_gt_median=agg("absS_gt")[1], absS_gen_median=agg("absS_gen")[1],
                   gen_all_zero_frac=agg("gen_all_zero")[0],
                   support_common_band_segments=agg("signS")[2],
                   per_seg={str(m): dict(
                       fam_hit=agg("fam_hit", [r for r in ok if r["m"] == m])[0],
                       signS=agg("signS", [r for r in per if r["m"] == m])[0],
                       absS_gen=agg("absS_gen", [r for r in per if r["m"] == m])[1],
                       all_zero=agg("gen_all_zero", [r for r in per if r["m"] == m])[0])
                       for m in range(1, 7)},
                   wall_s=time.time()-t0)
    (out / f"probe_{tag}.json").write_text(json.dumps(summary, indent=1))
    with open(out / f"per_segment_{tag}.jsonl", "w") as f:
        for r in per: f.write(json.dumps(r)+"\n")
    (out / f"texts_{tag}.json").write_text(json.dumps(texts, indent=1, ensure_ascii=False))
    print("PROBE_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
