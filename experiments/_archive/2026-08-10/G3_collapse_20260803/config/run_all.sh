#!/usr/bin/env bash
# G3 / Gate D3 — full grid.  GPU 1 ONLY (task card GPU discipline).
#
# 10 runs = 3 datasets x 2 arms x seeds, all at an identical budget
# (same steps / bs / lr / N / init / optimizer -- the arms differ ONLY in the
# s-axis parameterization, which is the whole point of the comparison).
set -u                       # NOT -e: one failing run must not kill the grid

cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=1
PY=/home/bc/VeraRetouch/.venv-lens/bin/python
OUT=experiments/G3_collapse_20260803
RUNS=$OUT/runs
mkdir -p "$RUNS"

STEPS=${STEPS:-25000}
PROBE_EVERY=${PROBE_EVERY:-100}
PROBE_N=${PROBE_N:-8}
FINAL_N=${FINAL_N:-48}

run () {                      # run <dataset> <arm> <seed>
  local ds=$1 arm=$2 seed=$3
  local d="$RUNS/${ds}_${arm}_s${seed}"
  if [ -f "$d/metrics.json" ]; then
    echo "[grid] SKIP $ds/$arm/s$seed (already done)"; return 0
  fi
  echo "[grid] START $ds/$arm/s$seed  $(date -Is)"
  $PY -m model.glut_repro.train_g3 \
      --arm "$arm" --dataset "$ds" --seed "$seed" \
      --steps "$STEPS" --probe-every "$PROBE_EVERY" \
      --probe-n "$PROBE_N" --final-n "$FINAL_N" \
      --outdir "$d" 2>&1 | sed "s|^|[$ds/$arm/s$seed] |"
  echo "[grid] END   $ds/$arm/s$seed  $(date -Is)"
}

# primary: s fully sufficient, 3 seeds (the headline Gate-D3 read)
for s in 0 1 2; do for a in naive anchored; do run fixed    "$a" "$s"; done; done
# intermediate: s partially sufficient
for s in 0;     do for a in naive anchored; do run tiered   "$a" "$s"; done; done
# as-shipped mixed targets: s nearly worthless (diagnostic / literal task card)
for s in 0;     do for a in naive anchored; do run mixed    "$a" "$s"; done; done

echo "[grid] all runs finished, analysing $(date -Is)"
$PY "$OUT/analyze_g3.py" --runs-dir "$RUNS" --out "$OUT"
echo "G3_FULL_DONE"
