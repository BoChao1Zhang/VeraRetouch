#!/usr/bin/env bash
# S4 · the four Where-A calibration arms, one GPU, strictly sequential.
#
# Sequential on purpose (D12): the CPU fit pool is the shared bottleneck, so two
# arms in parallel would each get ~16 workers, roughly double each one's fit
# phase, and finish at about the same wall clock while occupying a second card.
# One card, one arm at a time, ~11.8 h total; the other GPU stays free.
#
# Order: BA-0-Fixed (no training, ~5 min -- it is also the warm-up that proves
# the whole path before a multi-hour arm starts) -> BA-1-Band -> BA-2-CBand12
# -> BA-3-Joint (the pre-registered main arm).  After BA-3 the calibrated B's
# digest and frozen path are printed, which is what S5 (make_oracle_latents)
# consumes.
#
# D-20 per arm: rm -f the log, capture the PID, prove liveness with `ps -p`,
# tail for real output, only then write job.marker.  Never `pgrep` -- it matches
# the shell doing the grepping.
#
# Usage:
#   bash q3vl/where/scripts/run_where_a_arms.sh            # all four, in order
#   bash q3vl/where/scripts/run_where_a_arms.sh BA-3-Joint # one arm
#   GPU=1 bash q3vl/where/scripts/run_where_a_arms.sh      # pick the card
#   DRY_RUN=1 bash ...                                     # print, do not launch
set -euo pipefail

REPO=/home/bc/VeraRetouch
PY=/home/bc/envs/q3vl_sft/bin/python
RUNS=/home/bc/data/runs/where_a
SHARED="$RUNS/shared"
CKPT=${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}
MASKVIEWS=${MASKVIEWS:-/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews}
GPU=${GPU:-0}
WORKERS=${WORKERS:-32}
PREFETCH=${PREFETCH:-2}
BATCH=${BATCH:-32}
DRY_RUN=${DRY_RUN:-0}

# sqlite3 in this env needs conda's libstdc++ (CXXABI_1.3.15); the mask locator
# opens the build catalogs, so without this the whole data path fails to start.
export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# D12: one BLAS thread per fit is the canonical numeric setting, and it has to be
# in the environment before numpy/torch load.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

ARMS_DEFAULT=(BA-0-Fixed BA-1-Band BA-2-CBand12 BA-3-Joint)
if [[ $# -gt 0 ]]; then ARMS=("$@"); else ARMS=("${ARMS_DEFAULT[@]}"); fi

# D1: training uses the FULL local train pool, low included.  42,752 is only the
# reporting headline stratum.  Counted once and shared by every arm so all four
# build the identical cosine schedule (B-3).
COUNT_CACHE="$SHARED/eligible_count_train.json"

preflight() {
  [[ -x "$PY" ]] || { echo "FATAL: $PY missing"; exit 1; }
  [[ -d "$CKPT" ]] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
  [[ -d "$MASKVIEWS/train" ]] || {
    echo "FATAL: train mask views not published: $MASKVIEWS/train"
    echo "       S1 must finish first (a .train.partial.* directory means it is still writing)."
    exit 1; }
  [[ -f "$COUNT_CACHE" ]] || { echo "FATAL: no eligible count at $COUNT_CACHE"; exit 1; }
  local n
  n=$(python3 -c "import json,sys;print(json.load(open('$COUNT_CACHE'))['n_eligible'])")
  if [[ "$n" != "75544" ]]; then
    echo "FATAL: eligible count is $n, expected 75544 (D1: training includes"
    echo "       winner_confidence=low).  Delete $COUNT_CACHE and re-count."
    exit 1; fi
  echo "preflight OK  gpu=$GPU  ckpt=$CKPT  eligible=$n  workers=$WORKERS  prefetch=$PREFETCH"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

run_arm() {
  local arm="$1"
  local dir="$RUNS/$arm"
  local log="$dir/train.log"
  mkdir -p "$dir"

  if [[ -f "$dir/DONE" ]]; then
    echo "== $arm already finished (found $dir/DONE), skipping"
    return 0
  fi

  local cmd=("$PY" -m q3vl.where.scripts.run_calibration
             --arm "$arm" --checkpoint "$CKPT"
             --maskview-root "$MASKVIEWS"
             --batch-size "$BATCH" --workers "$WORKERS" --prefetch "$PREFETCH"
             --count-cache "$COUNT_CACHE")

  echo "== $arm  ->  $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "   DRY_RUN: CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"
    return 0
  fi

  # 1. rm -f first: zsh noclobber makes `> existing.log` fail the whole
  #    redirection, and the process then never starts, invisibly.
  rm -f "$log" "$log.pid"
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES="$GPU" nohup "${cmd[@]}" > "$log" 2>&1 &
    echo $! > "$log.pid" )
  local pid; pid=$(cat "$log.pid")

  # 2. prove it is alive with `ps -p $PID` (never pgrep)
  sleep 20
  if ! ps -p "$pid" > /dev/null 2>&1; then
    echo "FAILED to start $arm; last log lines:"; tail -30 "$log"; return 1
  fi

  # 3. wait for real output, not just a file
  local waited=0
  until grep -q '"projector_init"' "$log" 2>/dev/null; do
    sleep 10; waited=$((waited+10))
    if ! ps -p "$pid" > /dev/null 2>&1; then
      echo "FAILED: $arm exited during setup; last log lines:"; tail -30 "$log"; return 1
    fi
    if [[ $waited -ge 600 ]]; then
      echo "FAILED: $arm produced no setup block in 10 min"; tail -30 "$log"; return 1
    fi
  done

  # 4. only now the marker
  rm -f "$dir/job.marker"
  { printf 'arm=%s\npid=%s\ngpu=%s\nstarted=%s\nlog=%s\n' \
      "$arm" "$pid" "$GPU" "$(date -Is)" "$log"
    printf 'cmd=CUDA_VISIBLE_DEVICES=%s %s\n' "$GPU" "${cmd[*]}"
    printf 'eligible=75544 (D1: low included)\nworkers=%s prefetch=%s batch=%s\n' \
      "$WORKERS" "$PREFETCH" "$BATCH"
  } > "$dir/job.marker"
  echo "   started pid=$pid; marker written"
  sed -n '/fit_pool_self_check/p' "$log" | head -1

  # block until this arm is done: the arms are strictly sequential
  while ps -p "$pid" > /dev/null 2>&1; do sleep 60; done
  wait "$pid" 2>/dev/null || true

  if ! grep -q '"basis"\|verify' "$log" 2>/dev/null || \
     [[ ! -f "$dir/projector_final.pt" ]]; then
    echo "FAILED: $arm finished without projector_final.pt; last log lines:"
    tail -40 "$log"; return 1
  fi
  date -Is > "$dir/DONE"
  echo "   $arm finished  $(date -Is)"
  return 0
}

report_basis() {
  local arm="${1:-BA-3-Joint}"
  local basis="/mnt/nfs/bc/data/datasets/where_a-20260805/basis/$arm"
  echo
  echo "================ $arm : frozen basis ================"
  if [[ -f "$basis/basis.json" ]]; then
    python3 - "$basis" <<'PYEOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
m = json.loads((d / "basis.json").read_text())
print(f"  B path      : {d/'B.npy'}")
print(f"  shape       : {m['shape']}")
print(f"  sha256(npy) : {m['sha256']}")
print(f"  digest(B)   : {m['projector']['digest']}")
print(f"  arm         : {m['arm']}")
print(f"  oracle shards: {m.get('oracle_shards')}")
print()
print("  S5 (train oracle latents) consumes exactly this:")
print(f"    python -m q3vl.where.scripts.make_oracle_latents --arm {m['arm']} \\")
print(f"        --split train --basis {d/'B.npy'} \\")
print( "        --checkpoint $CKPT --maskview-root $MASKVIEWS")
PYEOF
  else
    echo "  MISSING: $basis/basis.json -- the arm did not publish its basis"
  fi
  echo "====================================================="
}

main() {
  preflight
  local t0=$SECONDS
  for arm in "${ARMS[@]}"; do
    run_arm "$arm" || { echo "ABORTING: $arm failed"; exit 1; }
  done
  echo
  echo "all arms done in $(( (SECONDS - t0) / 60 )) min"
  for arm in "${ARMS[@]}"; do
    [[ "$arm" == "BA-0-Fixed" ]] && continue
    report_basis "$arm"
  done
}

main
