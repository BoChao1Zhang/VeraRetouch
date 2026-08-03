#!/bin/bash
# E1 full sweep: 400 stratified LUTs x N in {8,16,24,32,48,64,96,128}.
# per_fit.jsonl append-only -> safe to restart (resume skips done pairs).
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/E1_cube_N_20260803
python3 -m model.glut_repro.run_e1 \
  --subset $D/config/e1_subset_400.txt \
  --n-list 8,16,24,32,48,64,96,128 \
  --outdir $D/runs/main --batch-luts 16 --gt-workers 12 --de-workers 12
echo "E1_FULL_DONE"
