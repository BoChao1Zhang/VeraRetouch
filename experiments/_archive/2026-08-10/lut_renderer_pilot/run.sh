#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$REPO_ROOT/experiments/lut_renderer_pilot/config.yaml"
OUTPUT="$REPO_ROOT/experiments/lut_renderer_pilot/outputs/native4000_online500_seed1701"
PYTHON_BIN="${VERARETOUCH_PYTHON:-/home/bc/miniconda3/bin/python}"

mkdir -p "$OUTPUT"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

cd "$REPO_ROOT"
{
  date --iso-8601=seconds
  "$PYTHON_BIN" --version
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  "$PYTHON_BIN" -m pip freeze
} > "$OUTPUT/environment.txt"

"$PYTHON_BIN" -m experiments.lut_renderer_pilot.prepare --config "$CONFIG"
"$PYTHON_BIN" -m experiments.lut_renderer_pilot.train --config "$CONFIG" --model cglut
"$PYTHON_BIN" -m experiments.lut_renderer_pilot.train --config "$CONFIG" --model vera
"$PYTHON_BIN" -m experiments.lut_renderer_pilot.evaluate --config "$CONFIG"

