#!/bin/bash
# Stage-1 全参 SFT（S1F-FULL）—— vlmsft/run_s1f_full.sh 的主线包版（只改入口路径与配置物化）。
# usage: run_vlm_sft.sh [--resume-from <ckpt> --resume-step N] [其它 train_vlm_sft 参数]
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=${VR_REPO:-/home/bc/VeraRetouch}
PY=${VR_PY_VLM:-/home/bc/envs/q3vl_sft/bin/python}
OUT=${VR_OUT:-/home/bc/data/runs/epr051_vlmsft/sft_s1f_full}
export PYTHONPATH=$REPO
mkdir -p "$OUT/configs_resolved"
$PY -c "from veraretouch_sprf import configs as CF; print(CF.materialize(CF.CONFIG_DIR/'vlm/sft_full.toml', '$OUT/configs_resolved'))"
cd "$REPO"
exec $PY -u -m veraretouch_sprf.train.train_vlm_sft \
  --config "$OUT/configs_resolved/q3vl_sft_s1f_full.toml" --out-dir "$OUT" "$@"
