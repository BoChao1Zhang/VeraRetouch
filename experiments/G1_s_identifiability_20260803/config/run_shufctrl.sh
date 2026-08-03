#!/usr/bin/env bash
# 指令条件性对照批（60 源 × shuf 指令）；链在全量任务（PID 3228455）之后
set -euo pipefail
cd /home/bc/VeraRetouch
EXP=/home/bc/VeraRetouch/experiments/G1_s_identifiability_20260803
mkdir -p "$EXP/run_shufctrl"
while kill -0 3228455 2>/dev/null; do sleep 60; done
exec .venv-lens/bin/python tools/readout/ro9_gl_attention.py \
  --samples-json "$EXP/config/g1_shuffle_ctrl.json" \
  --scache-root /var/cache/veradata/scache \
  --out-dir "$EXP/run_shufctrl" \
  --skip-existing
