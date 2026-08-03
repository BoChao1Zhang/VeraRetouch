#!/usr/bin/env bash
# 区域对立批（D-17 主判据；214 源 × {reg_a,reg_b}）；链在错位对照批（PID 3232359，
# 其 bash 会 exec 成 python，PID 不变）之后；冒烟 20 源已先行（run_regsmoke20，
# scache 已有 → --skip-existing 自动跳过）
set -euo pipefail
cd /home/bc/VeraRetouch
EXP=/home/bc/VeraRetouch/experiments/G1_s_identifiability_20260803
mkdir -p "$EXP/run_regfull"
while kill -0 3232359 2>/dev/null; do sleep 60; done
exec .venv-lens/bin/python tools/readout/ro9_gl_attention.py \
  --samples-json "$EXP/config/g1_region_opp.json" \
  --scache-root /var/cache/veradata/scache \
  --out-dir "$EXP/run_regfull" \
  --skip-existing
