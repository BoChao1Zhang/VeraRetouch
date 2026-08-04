#!/usr/bin/env bash
# Where-B: the exact order in which the pending jobs must be run.
# NOTHING IN HERE HAS BEEN EXECUTED YET.  Two H100s are held by Base SFT
# (rank PID 3395226/3395227, ETA 2026-08-05 ~11:30), and every step below either
# needs a GPU or streams the same NFS tree the trainer reads.
#
# D-20 discipline is baked into `submit`:
#   1. rm -f the target log   (zsh noclobber makes `> existing.log` kill the
#      whole redirection, and the process then never starts)
#   2. capture $! and prove liveness with `ps -p $PID`  (NEVER pgrep: any
#      `pgrep -f <pattern>` matches the very shell running the grep)
#   3. tail the log until real output appears
#   4. only then write job.marker and report upstream
set -euo pipefail

# The training env's sqlite3 needs conda's libstdc++ (CXXABI_1.3.15); the mask
# locator opens a build catalog, so without this the data path cannot start.
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
PY=/home/bc/envs/q3vl_sft/bin/python
REPO=/home/bc/VeraRetouch
RUNS=/home/bc/data/runs/where_b
CKPT=${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}

# submit <gpu|-> <logfile> <marker-grep> <cmd...>
#
# nit N7: the previous version took no GPU argument (so `usage`'s "two arms at
# once" would have put both on GPU 0), recorded the PID of the wrapping subshell
# rather than of python, and treated `tail` as step 3 even though printing a log
# is not checking it -- an empty log still returned 0.
submit() {
  local gpu="$1"; shift
  local log="$1"; shift
  local want="$1"; shift
  mkdir -p "$(dirname "$log")"
  rm -f "$log" "$log.pid"                        # step 1 (zsh noclobber)
  if [ "$gpu" != "-" ]; then
    export CUDA_VISIBLE_DEVICES="$gpu"
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
      | sed -n "$((gpu+1))p" | sed 's/^/  target GPU: /'
  fi
  # exec so $! is the python PID itself, not a wrapper shell's
  ( cd "$REPO" && nohup setsid "$@" > "$log" 2>&1 & echo $! > "$log.pid" )
  local pid; pid=$(cat "$log.pid")

  # step 2: prove liveness with `ps -p` and show the command line, so a wrong
  # PID is visible instead of assumed.  NEVER pgrep.
  sleep 5
  ps -p "$pid" -o pid,etime,cmd --no-headers || {
    echo "FAILED to start (no such pid $pid): $*"; sed -n '1,40p' "$log"; return 1; }

  # step 3: the log must contain real output, not merely exist
  local waited=0
  until grep -q "$want" "$log" 2>/dev/null; do
    sleep 5; waited=$((waited+5))
    ps -p "$pid" > /dev/null || {
      echo "process died before printing /$want/ after ${waited}s"; tail -n 40 "$log"; return 1; }
    if [ "$waited" -ge 600 ]; then
      echo "no /$want/ in $log after ${waited}s -- refusing to claim success"
      tail -n 40 "$log"; return 1
    fi
  done
  echo "started pid=$pid gpu=${gpu} log=$log (matched /$want/ after ${waited}s)"
  tail -n 20 "$log"

  # step 4: only now is the job real
  printf 'pid=%s\ngpu=%s\ncmd=%s\nlog=%s\nstarted_at=%s\nverified=/%s/ after %ss\n' \
    "$pid" "$gpu" "$*" "$log" "$(date -Is)" "$want" "$waited" \
    > "$(dirname "$log")/job.marker"
}

case "${1:-help}" in

# --- step 0: no GPU, no weights -------------------------------------------
preflight-cpu)
  cd "$REPO" && $PY -m q3vl.whereb.preflight
  ;;

# --- step 1: protocol 14 items 7b/8b on the real checkpoint (1 GPU) --------
# Foreground on purpose: it is ~5 minutes and its exit code is the gate.
preflight)
  gpu="${2:-0}"
  cd "$REPO" && CUDA_VISIBLE_DEVICES="$gpu" $PY -m q3vl.whereb.preflight \
    --with-model --checkpoint "$CKPT" --device cuda
  ;;

# --- step 2: oracle latents for the training split (1 GPU, after Where-A) --
oracle)
  split="${2:-train}"; gpu="${3:-0}"
  submit "$gpu" "$RUNS/oracle/$split.log" '"basis"' \
    $PY -m q3vl.whereb.scripts.make_oracle_latents --split "$split" --checkpoint "$CKPT"
  ;;

# --- step 3: the generated <where> context (1 GPU) ------------------------
genctx)
  split="${2:-train}"; gpu="${3:-0}"
  submit "$gpu" "$RUNS/genctx/$split.log" '"vlm"' \
    $PY -m q3vl.whereb.scripts.make_generated_context --split "$split" --checkpoint "$CKPT"
  ;;

# --- step 4: the eight main arms (1 GPU each; 2 at a time on this box) -----
train)
  arm="${2:?usage: run_where_b.sh train <W01..W08> [gpu]}"; gpu="${3:-0}"
  submit "$gpu" "$RUNS/$arm/train.log" '"total_optimizer_steps"' \
    $PY -m q3vl.whereb.scripts.run_where_b --arm "$arm" --checkpoint "$CKPT"
  ;;

*)
  cat <<'USAGE'
usage: run_where_b.sh {preflight-cpu | preflight [gpu]
                       | oracle <split> [gpu] | genctx <split> [gpu]
                       | train <arm> [gpu]}

  preflight-cpu     protocol 14 items 7/8/9, structural half; no GPU, no weights
  preflight [gpu]   adds 7b/8b on a real VLM forward (1 GPU, foreground, gates S5)
  oracle <split>    per-image oracle latents with the frozen BA-3-Joint basis
  genctx <split>    cache the Base SFT model's own <where> spans (indexed shards)
  train <arm> [gpu] one main arm, 1 epoch (1 GPU)

Every background job pins CUDA_VISIBLE_DEVICES to its `gpu` argument, so the
two arms of a wave go to different cards:

  bash run_where_b.sh train W01 0 &
  bash run_where_b.sh train W02 1 &

order: preflight-cpu -> [Where-A basis published] -> oracle train,V_where
       -> genctx train,V_where -> preflight -> train W01..W08 (waves W1..W4)
USAGE
  ;;
esac
