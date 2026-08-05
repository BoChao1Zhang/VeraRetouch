#!/usr/bin/env bash
# S5 · per-image oracle latents against the frozen BA-3-Joint B.
#
# Order is deliberate: the four evaluation splits first (~15 min in total), then
# `train` (~12 h).  Where-B's gate needs the `V_where` oracle ratio before it can
# do anything, and the eval splits double as an end-to-end check of a job that
# has never run at full size -- finding a defect after 15 minutes beats finding
# it after eleven hours.
#
# Output namespace: `<ORACLE>/<arm>/s5/<split>`.
# The arms' own `evaluate()` already published `<ORACLE>/<arm>/V_where`, but those
# records carry no `curve` / `cband_normalization` -- they are the arm's ceiling
# evidence, not Where-B supervision.  S5 writes its own namespace so every split
# comes from one producer with one convention (257-point r*(z) grid, declared
# CBand normalisation), and the arms' published record stays untouched.
#
# D-20 per split: rm -f the log, capture the PID, prove liveness with `ps -p`,
# wait for real output, only then write job.marker.  Never `pgrep`.
set -euo pipefail

REPO=/home/bc/VeraRetouch
PY=/home/bc/envs/q3vl_sft/bin/python
RUNS=/home/bc/data/runs/where_a/s5
ARM=${ARM:-BA-3-Joint}
BASIS=${BASIS:-/mnt/nfs/bc/data/datasets/where_a-20260805/basis/$ARM/B.npy}
CKPT=${CKPT:-/home/bc/data/runs/q3vl_base_sft_20260804/checkpoint-4976}
MASKVIEWS=${MASKVIEWS:-/mnt/nfs/bc/data/datasets/where_a-20260805/maskviews}
ORACLE=${ORACLE:-/mnt/nfs/bc/data/datasets/where_a-20260805/oracle}
GPU=${GPU:-0}
WORKERS=${WORKERS:-32}
PREFETCH=${PREFETCH:-64}
DRY_RUN=${DRY_RUN:-0}

export LD_LIBRARY_PATH=/home/bc/miniconda3/envs/llm_factory/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# eval splits first, train last
SPLITS_DEFAULT=(V_where V_what T_final T_lut_unseen train)
if [[ $# -gt 0 ]]; then SPLITS=("$@"); else SPLITS=("${SPLITS_DEFAULT[@]}"); fi

preflight() {
  [[ -f "$BASIS" ]] || { echo "FATAL: frozen B not found: $BASIS"; exit 1; }
  [[ -d "$CKPT" ]] || { echo "FATAL: checkpoint not found: $CKPT"; exit 1; }
  [[ -d "$MASKVIEWS/train" ]] || { echo "FATAL: mask views not published"; exit 1; }
  local meta="$(dirname "$BASIS")/basis.json"
  [[ -f "$meta" ]] || { echo "FATAL: $meta missing"; exit 1; }
  echo "S5 preflight OK"
  python3 - "$meta" <<'PYEOF'
import json, sys
m = json.loads(open(sys.argv[1]).read())
print(f"  arm    : {m['arm']}")
print(f"  B      : {m['shape']}  sha256={m['sha256'][:16]}...")
print(f"  digest : {m['projector']['digest']}")
PYEOF
  echo "  gpu=$GPU workers=$WORKERS prefetch=$PREFETCH"
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
}

run_split() {
  local split="$1"
  local dir="$RUNS/$split"
  local log="$dir/run.log"
  local out="$ORACLE/$ARM/s5/$split"
  mkdir -p "$dir"

  if [[ -f "$dir/DONE" ]]; then
    echo "== $split already finished, skipping"; return 0
  fi

  local cmd=("$PY" -m q3vl.where.scripts.make_oracle_latents
             --arm "$ARM" --split "$split" --basis "$BASIS"
             --checkpoint "$CKPT" --maskview-root "$MASKVIEWS"
             --workers "$WORKERS" --prefetch "$PREFETCH"
             --out-root "$out")
  # The evaluation splits are small, so they also carry the delivery-resolution
  # tier (guide + hi-res mask); `train` is 75,544 samples and only needs the
  # low-res tier the fit actually ran on, which is the 5.6 gate's denominator.
  # NOT `[[ ... ]] && cmd+=(...)`: for `train` the test is false, the compound
  # returns non-zero, and `set -e` would abort the one split that matters most.
  if [[ "$split" != "train" ]]; then cmd+=(--attach-hi); fi

  echo "== $split  ->  $log"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "   DRY_RUN: CUDA_VISIBLE_DEVICES=$GPU ${cmd[*]}"; return 0
  fi

  rm -f "$log" "$log.pid"
  ( cd "$REPO" && CUDA_VISIBLE_DEVICES="$GPU" nohup "${cmd[@]}" > "$log" 2>&1 &
    echo $! > "$log.pid" )
  local pid; pid=$(cat "$log.pid")

  sleep 20
  ps -p "$pid" > /dev/null 2>&1 || {
    echo "FAILED to start $split:"; tail -30 "$log"; return 1; }

  # A small split can finish faster than this poll interval, so "the process is
  # gone" is not evidence of failure -- T_lut_unseen (198 samples) completed
  # successfully and was reported as a startup failure.  Absence of the process
  # just ends the wait; success is judged on artifacts below, as everywhere else.
  local waited=0
  until grep -qE '"n": [0-9]+|"n_samples"|elapsed_s' "$log" 2>/dev/null; do
    sleep 15; waited=$((waited+15))
    ps -p "$pid" > /dev/null 2>&1 || break
    [[ $waited -ge 900 ]] && { echo "FAILED: $split silent for 15 min"; tail -30 "$log"; return 1; }
  done

  rm -f "$dir/job.marker"
  { printf 'split=%s\narm=%s\npid=%s\ngpu=%s\nstarted=%s\nlog=%s\nout=%s\n' \
      "$split" "$ARM" "$pid" "$GPU" "$(date -Is)" "$log" "$out"
    printf 'basis=%s\nworkers=%s prefetch=%s\n' "$BASIS" "$WORKERS" "$PREFETCH"
  } > "$dir/job.marker"
  echo "   started pid=$pid; marker written"

  while ps -p "$pid" > /dev/null 2>&1; do sleep 60; done

  # judged on artifacts, not on log strings
  local report="$REPO/experiments/Q3VL_metacanvas_where_what_20260804/where_a/oracle_latents/${ARM}_${split}.report.json"
  if [[ ! -f "$out/manifest.json" || ! -f "$report" ]]; then
    echo "FAILED: $split left no manifest/report:"; tail -40 "$log"; return 1
  fi
  python3 - "$report" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
v = r.get("verify", {})
print(f"   {r['split']}: {r['n_samples']} samples, {r['n_rejected_fits']} rejected fits, "
      f"{r['elapsed_s']}s, verify_ok={v.get('ok')}, "
      f"curve_grid={r['curve_z_grid']['n']}, norm={r['cband_normalization']}, "
      f"b_unchanged={r['b_unchanged']}")
for ro, p in r.get("soft_iou_low", {}).items():
    if p:
        print(f"     {ro}: median soft-IoU {p['median']:.4f} (n={p['n']})")
PYEOF
  date -Is > "$dir/DONE"
  echo "   $split finished  $(date -Is)"
}

main() {
  preflight
  local t0=$SECONDS
  for split in "${SPLITS[@]}"; do
    run_split "$split" || { echo "ABORTING at $split"; exit 1; }
  done
  echo
  echo "S5 done in $(( (SECONDS - t0) / 60 )) min"
  echo "oracle latents: $ORACLE/$ARM/s5/{${SPLITS[*]// /,}}"
}

main
