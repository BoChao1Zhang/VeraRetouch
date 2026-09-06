#!/bin/bash
# held-out d6 headline：合并两半生成缓存 → 单进程读出 → BK-FULL 执行器评测（vlmsft/heldout_final.sh 的主线包版）。
# usage: heldout_final.sh <cache_a.pt> <cache_b.pt> <tag> <out_dir> <adapt_ckpt> <manifest> [max_new]
set -euo pipefail
REPO=${VR_REPO:-/home/bc/VeraRetouch}
PYV=${VR_PY_VLM:-/home/bc/envs/q3vl_sft/bin/python}; PYD=${VR_PY_DEC:-/home/bc/envs/databuild/bin/python}
R=${VR_RUNS_VLM:-/home/bc/data/runs/epr051_vlmsft}; RS=${VR_RUNS_SPRF:-/home/bc/data/runs/epr051_sprf}
CA=$1; CB=$2; TAG=$3; OUT=$4; ADAPT=$5; MAN=$6; MAXNEW=${7:-2048}
BASE=$R/sft_s1f_full/ckpt_epoch1
HELD=$REPO/experiments/prs/EPR-051_masked-restore-production/stage0/snapshot_newdata_v3.heldout_ids.json
KEYS=$($PYD -c "import json;print(json.load(open('$MAN'))['source'])")
cd "$REPO"; mkdir -p "$OUT"; export PYTHONPATH=$REPO
echo "[final] merge gen caches -> $OUT/gencache_$TAG.pt"
$PYD -u -m veraretouch_sprf.scripts.merge_gencache --half "$CA" --half "$CB" --manifest "$MAN" --out "$OUT/gencache_$TAG.pt"
echo "[final] readout over all keys (generation must be fully cached)"
$PYV -u -m veraretouch_sprf.eval.eval_vlm_e2e dump \
  --adapt-run "$ADAPT" --sft-run "$R/sft_s1f_full" --base-weights-dir "$BASE" \
  --contract predicted_text --gen-adapter s2 --instruction-mode per_sample \
  --keys "$KEYS" --records "$R/heldout_d6_records.jsonl" --assets-index "$R/assets_y_heldout/assets_index.json" \
  --target-latents "$R/targets_bkfull/target_latents_bkfull.pt" \
  --gen-batch 16 --max-new-tokens "$MAXNEW" --cache-every 5 --resume-cache \
  --attn sdpa --device cuda:0 --tag "$TAG" --out "$OUT"
echo "[final] BK-FULL executor: predicted_text + oracle_lut on the same keys"
$PYD -u -m veraretouch_sprf.eval.eval_vlm_e2e eval --backend bk --also-oracle-lut \
  --latents "$OUT/latents_$TAG.pt" --backbone-run "$RS/bkfull_adagn_ff_affhead" --ckpt ckpt_last.pt \
  --feats-cache "$RS/feature_cache" --heldout-ids "$HELD" --select-keys "$KEYS" \
  --out "$OUT/eval_$TAG.json" --device cuda:0
echo "[final] DONE"
