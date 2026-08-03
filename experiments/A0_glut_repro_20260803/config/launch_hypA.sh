#!/bin/bash
# Hypothesis (a): representation vs optimization recipe.
# E1 direct-overfit engine on the SAME 75 A0 LUTs / same GT / same N=32.
set -e
cd /home/bc/VeraRetouch
export CUDA_VISIBLE_DEVICES=0
D=experiments/A0_glut_repro_20260803
python3 -m model.glut_repro.run_e1_on_a0 \
  --lut-list $D/config/a0_luts_75.txt --steps 3000 --batch-luts 16 \
  --outdir $D/hypA --tag e1_on_a0_3k
python3 -m model.glut_repro.run_e1_on_a0 \
  --lut-list $D/config/a0_luts_75.txt --steps 12000 --batch-luts 16 \
  --outdir $D/hypA --tag e1_on_a0_12k
echo "HYPA_DONE"
