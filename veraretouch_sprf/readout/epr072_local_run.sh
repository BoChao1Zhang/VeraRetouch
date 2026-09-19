#!/usr/bin/env bash
set -euo pipefail
ARM=${1:?PIX or NCE}
OUT=${2:?unique output directory}
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib
export PYTHONPATH=/home/bc/VeraRetouch
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_RUNTIME_DIR=/run/user/1001
export DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1001/bus
cd /home/bc/VeraRetouch
exec systemd-run --user --scope -q -p MemoryMax=40G -p MemorySwapMax=0 \
  bash -c '
    set -euo pipefail
    py=/home/bc/envs/q3vl_sft/bin/python
    "$py" -m veraretouch_sprf.readout.epr072_local_train --arm "$1" --out "$2/smoke" --smoke 2
    "$py" -m veraretouch_sprf.readout.epr072_local_train --arm "$1" --out "$2/full"
  ' -- "$ARM" "$OUT"
