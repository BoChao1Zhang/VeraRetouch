#!/bin/bash
# 在线判据（宿主 q3vl_sft 环境，gpu0）：32 键 val greedy 自生成 → S2F-B 读出（LoRA+adapter 冻结）→ 对 e* 余弦。
# 与 0.39 基线同一代码路径（dump_readout：full base + single LoRA，生成与读出均挂 Stage-2 LoRA）。
# usage: online_eval.sh <ckpt_dir(HF 目录: config.json+safetensors)> <tag> <out_dir>
set -euo pipefail
CK=$1; TAG=$2; OUT=$3
REPO=${VR_REPO:-/home/bc/VeraRetouch}; PY=${VR_PY_VLM:-/home/bc/envs/q3vl_sft/bin/python}
R=/home/bc/data/runs/epr051_vlmsft
mkdir -p "$OUT/base_$TAG"; ln -sfn "$CK" "$OUT/base_$TAG/model"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$REPO PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
CUDA_VISIBLE_DEVICES=${GPU:-1} $PY -u -m veraretouch_sprf.eval.eval_vlm_e2e dump \
  --adapt-run $R/adapt_s2fb/ckpt_epoch1 --sft-run $R/sft_s1f_full --base-weights-dir "$OUT/base_$TAG" \
  --contract predicted_text --gen-adapter s2 --instruction-mode per_sample \
  --keys $R/s1f_val32_keys.json --records $R/snap_sft2/records.jsonl --assets-index $R/snap_sft2/assets_index.json \
  --target-latents $R/targets_bkfull/target_latents_bkfull.pt \
  --gen-batch "${GB:-8}" --max-new-tokens 2048 --attn sdpa --device cuda:0 --tag "$TAG" --out "$OUT" > "$OUT/eval_$TAG.log" 2>&1
python3 - "$OUT" "$TAG" "$CK" <<'PY'
import json, sys, glob, time
out, tag, ck = sys.argv[1:4]
f = glob.glob(f"{out}/dump_{tag}.json")
d = json.load(open(f[0])) if f else {}
lat = None
for ln in open(f"{out}/eval_{tag}.log"):
    if ln.startswith("LATENT_METRICS "): lat = json.loads(ln[len("LATENT_METRICS "):])
rec = dict(t=time.time(), tag=tag, ckpt=ck, n=d.get("n"), n_ok=d.get("n_ok"), stage_token_missing_rate=d.get("stage_token_missing_rate"),
           cot_parse_success_rate=d.get("cot_parse_success_rate"), latent_cos_mean=(lat or {}).get("latent_cos_mean"),
           latent_cos_median=(lat or {}).get("latent_cos_median"), latent_l2_mean=(lat or {}).get("latent_l2_mean"), wall_s=d.get("wall_s"))
open(f"{out}/online_eval.jsonl", "a").write(json.dumps(rec) + "\n"); print("ONLINE_EVAL " + json.dumps(rec))
PY
