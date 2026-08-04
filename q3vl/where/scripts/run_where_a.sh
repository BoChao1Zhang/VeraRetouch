#!/usr/bin/env bash
# Where-A: the exact order in which the pending jobs must be run.
# NOTHING IN HERE HAS BEEN EXECUTED YET.  Two H100s are held by Base SFT
# (PID 3395226/3395227); every step below either needs a GPU or hammers the same
# NFS build tree the trainer streams from.
#
# D-20 discipline is baked in: each background submission does
#   1. rm -f the target log   (zsh noclobber makes `> existing.log` kill the
#      whole redirection, and the process then never starts)
#   2. capture $! and prove liveness with `ps -p $PID`  (NEVER pgrep: any
#      `pgrep -f <pattern>` matches the very shell running the grep)
#   3. tail the log until real output appears
#   4. only then write job.marker and report upstream
set -euo pipefail

# The training env's sqlite3 needs conda's libstdc++ (CXXABI_1.3.15); without
# this, `import sqlite3` fails and the mask locator cannot open a build catalog.
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/home/bc/envs/q3vl_sft/bin/python
REPO=/home/bc/VeraRetouch
RUNS=/home/bc/data/runs/where_a
MASKVIEWS=${MASKVIEWS:-/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews}
CKPT=${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}

submit() {  # submit <logfile> <cmd...>
  local log="$1"; shift
  mkdir -p "$(dirname "$log")"
  rm -f "$log"                                   # step 1
  ( cd "$REPO" && nohup "$@" > "$log" 2>&1 & echo $! > "$log.pid" )
  local pid; pid=$(cat "$log.pid")
  sleep 5
  ps -p "$pid" > /dev/null || { echo "FAILED to start: $*"; return 1; }   # step 2
  echo "started pid=$pid log=$log"
  tail -n 20 "$log"                              # step 3
  printf 'pid=%s\ncmd=%s\nlog=%s\n' "$pid" "$*" "$log" > "$(dirname "$log")/job.marker"  # step 4
}

case "${1:-help}" in

# --- step 0: the checks that need neither GPU nor the build tree ------------
preflight-cpu)
  cd "$REPO" && $PY -m q3vl.where.preflight --skip-model
  ;;

# --- step 1: mask views (heavy IO, no GPU) ---------------------------------
maskviews)
  for split in train V_where V_what T_final T_lut_unseen; do
    cd "$REPO" && $PY -m q3vl.where.scripts.extract_maskviews --split "$split"
  done
  ;;

# --- step 2: protocol 14 items 4/5/6 on the real checkpoint (1 GPU) --------
preflight)
  cd "$REPO" && $PY -m q3vl.where.preflight --device cuda --limit 32 --checkpoint "$CKPT"
  ;;

# --- step 2b: fix D5 (guided-filter radius/eps) -- REQUIRED S2 output ------
# The E2 defaults came from a full-resolution per-channel filter, i.e. the order
# protocol 4.2 forbids, so they do not transfer (REVIEW-impl-WhereA B-4/N-16).
sweep-upsample)
  cd "$REPO" && $PY -m q3vl.where.scripts.sweep_upsample --limit 24 --checkpoint "$CKPT"
  ;;

# --- step 3: the four arms (1 GPU each; 2 at a time on this box) -----------
calibrate)
  arm="${2:?usage: run_where_a.sh calibrate <BA-0-Fixed|BA-1-Band|BA-2-CBand12|BA-3-Joint>}"
  submit "$RUNS/$arm/train.log" \
    $PY -m q3vl.where.scripts.run_calibration --arm "$arm" --checkpoint "$CKPT" \
        --maskview-root "$MASKVIEWS/train"
  ;;

# --- step 4: train-split oracle latents -- Where-B cannot start without them
# Protocol 5.5's L_s / L_curve / L_dir are *training* supervision, so they need
# per-image s*, r*(z), w_dir* on the TRAIN split, not just on V_where
# (REVIEW-impl-WhereA B-5).  Runs after S4 has frozen BA-3-Joint's B.
oracle-latents)
  submit "$RUNS/oracle_latents/train.log" \
    $PY -m q3vl.where.scripts.make_oracle_latents --arm BA-3-Joint --split train \
        --checkpoint "$CKPT" --maskview-root "$MASKVIEWS/train"
  ;;

*)
  cat <<'USAGE'
usage: run_where_a.sh {preflight-cpu|maskviews|preflight|sweep-upsample|calibrate <arm>|oracle-latents}

  preflight-cpu    protocol 14.6 + upsample-order check; no GPU, no data
  maskviews        publish GT mask views as indexed shards (AFTER Base SFT)
  preflight        protocol 14 items 4/5/6 on the real checkpoint (1 GPU)
  sweep-upsample   fix D5 radius/eps on the real path (1 GPU) -- required by S2
  calibrate <arm>  one basis-calibration arm, 1 epoch (1 GPU)
  oracle-latents   train-split oracle latents for Where-B (1 GPU), AFTER calibrate

order: preflight-cpu -> maskviews -> preflight -> sweep-upsample
       -> calibrate x4 -> oracle-latents
USAGE
  ;;
esac
