#!/usr/bin/env bash
# G1 全量 300 源 × 3 指令 RO-9 读出（长任务；gbatch 提交，单卡）
set -euo pipefail
cd /home/bc/VeraRetouch
EXP=/home/bc/VeraRetouch/experiments/G1_s_identifiability_20260803
mkdir -p "$EXP/run_full"
# 等 30 源冒烟批退出再动 GPU（--skip-existing 会复用其缓存条目）
until ! pgrep -f "run_smoke30" >/dev/null 2>&1; do sleep 30; done
exec .venv-lens/bin/python tools/readout/ro9_gl_attention.py \
  --samples-json "$EXP/config/g1_samples.json" \
  --scache-root /var/cache/veradata/scache \
  --out-dir "$EXP/run_full" \
  --skip-existing
