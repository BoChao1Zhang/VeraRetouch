#!/bin/bash
# 解码器训练提交（q submit 包装；NOTES_backend.md §10 的口径）。
# usage: submit_decoder.sh <arm: clut_full|bkfull_adagn_ff_affhead> <gpu 0|1> <mem_peak_gib> [透传给 train_decoder 的参数]
# 提交后按 D-20 四步：rm -f 日志已在此做 → ps -p <pid> 判活（禁 pgrep）→ tail 实质输出 → 写 job.marker。
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
ARM=${1:?arm}; GPU=${2:?gpu}; MEM=${3:?mem_peak_gib}; shift 3
REPO=${VR_REPO:-/home/bc/VeraRetouch}
PY=${VR_PY_DEC:-/home/bc/envs/databuild/bin/python}
LOGDIR=${VR_LOGDIR:-/home/bc/data/runs/epr051_sprf/logs}
mkdir -p "$LOGDIR"
LOG=$LOGDIR/${ARM}.log
NAME=SPRF_$(echo "$ARM" | tr '[:lower:]' '[:upper:]')
set +o noclobber; rm -f "$LOG" "$LOG.pid"
q submit "$NAME" "$GPU" "$LOG" --desc "veraretouch_sprf train_decoder --arm $ARM" \
  --mem-peak "$MEM" --ready 'A12' --ready-timeout 1800 \
  -- env PYTHONPATH="$REPO" "$PY" -u -m veraretouch_sprf.train.train_decoder --arm "$ARM" "$@"
