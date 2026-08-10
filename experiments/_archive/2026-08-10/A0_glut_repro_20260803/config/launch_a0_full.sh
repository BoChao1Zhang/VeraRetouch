#!/bin/bash
# A0 full run: 75 LUTs x {rec, full} arms, GLUT recipe, full held-out eval.
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=1
D=experiments/A0_glut_repro_20260803
python3 -m model.glut_repro.run_a0 \
  --lut-list $D/config/a0_luts_75.txt --arm rec \
  --outdir $D/runs/rec --chunk-size 25 --gt-workers 8 --de-workers 8
python3 -m model.glut_repro.run_a0 \
  --lut-list $D/config/a0_luts_75.txt --arm full \
  --outdir $D/runs/full --chunk-size 25 --gt-workers 8 --de-workers 8
echo "A0_FULL_DONE"
