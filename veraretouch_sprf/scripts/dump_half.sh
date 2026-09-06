#!/bin/bash
# held-out d6 headline：一张卡生成 + 读出一半键（vlmsft/dump_half.sh 的主线包版，只改入口）。
# usage: dump_half.sh <keys.json> <tag> <out_dir> <adapt_ckpt> [gen_batch] [max_new]
set -euo pipefail
REPO=${VR_REPO:-/home/bc/VeraRetouch}; PY=${VR_PY_VLM:-/home/bc/envs/q3vl_sft/bin/python}
R=${VR_RUNS_VLM:-/home/bc/data/runs/epr051_vlmsft}
KEYS=$1; TAG=$2; OUT=$3; ADAPT=$4; GB=${5:-16}; MAXNEW=${6:-2048}
BASE=$R/sft_s1f_full/ckpt_epoch1
RECORDS=${RECORDS:-$R/heldout_d6_records.jsonl}
ASSETS=${ASSETS:-$R/assets_y_heldout/assets_index.json}
cd "$REPO"; mkdir -p "$OUT"
echo "[half] tag=$TAG keys=$KEYS gen_batch=$GB max_new=$MAXNEW adapt=$ADAPT"
env PYTHONPATH="$REPO" $PY -u -m veraretouch_sprf.eval.eval_vlm_e2e dump \
  --adapt-run "$ADAPT" --sft-run "$R/sft_s1f_full" --base-weights-dir "$BASE" \
  --contract predicted_text --gen-adapter s2 --instruction-mode per_sample \
  --keys "$KEYS" --records "$RECORDS" --assets-index "$ASSETS" \
  --target-latents "$R/targets_bkfull/target_latents_bkfull.pt" \
  --gen-batch "$GB" --max-new-tokens "$MAXNEW" --cache-every 5 --resume-cache \
  --attn sdpa --device cuda:0 --tag "$TAG" --out "$OUT"
echo "[half] DONE $TAG"
