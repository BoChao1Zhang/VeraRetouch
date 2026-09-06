#!/bin/bash
# BK 解码器批量评测（batch_eval_bk → eval/eval_decoder.py）。usage: eval_decoder.sh <resolved arm toml> [--batch 8 --ckpt ckpt_last.pt --out metrics_batch.json ...]
set -euo pipefail
REPO=${VR_REPO:-/home/bc/VeraRetouch}; PY=${VR_PY_DEC:-/home/bc/envs/databuild/bin/python}
CFG=${1:?resolved config toml}; shift
cd "$REPO"; exec env PYTHONPATH="$REPO" "$PY" -u -m veraretouch_sprf.eval.eval_decoder --config "$CFG" "$@"
