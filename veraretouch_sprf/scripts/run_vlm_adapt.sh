#!/bin/bash
# Stage-2 读出+adapter（S2F-B）—— vlmsft/run_s2fb.sh 的主线包版。
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=${VR_REPO:-/home/bc/VeraRetouch}
PY=${VR_PY_VLM:-/home/bc/envs/q3vl_sft/bin/python}
OUT=${VR_OUT:-/home/bc/data/runs/epr051_vlmsft/adapt_s2fb}
export PYTHONPATH=$REPO
mkdir -p "$OUT/configs_resolved"
$PY -c "from veraretouch_sprf import configs as CF; print(CF.materialize(CF.CONFIG_DIR/'vlm/adapt_s2fb.toml', '$OUT/configs_resolved'))"
cd "$REPO"
exec $PY -u -m veraretouch_sprf.train.train_vlm_adapt \
  --config "$OUT/configs_resolved/q3vl_adapt_s2fb.toml" --out-dir "$OUT" "$@"
